#include "io-frontend.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <iostream>
#include <cstring>
#include <memory>
#include <optional>
#include "ggml.h"
#include <map>
#include <memory>
#include "pipeline.h"
#include <mutex>

#define DEVICE_NAME "/dev/tc_ns_client"

#define ROUND_UP(x, n)   (((x) + (n)-1) & ~((n)-1))
#define ROUND_DOWN(x, n) ((x) & ~((n)-1))
#define PAGE_SIZE 0x1000

static all_ring_buffer *task_queue;
static std::once_flag task_queue_once;

static std::mutex tasks_lock;
static std::map<void *, task_entry> tasks;

#ifdef LLAMA_USE_CHCORE_API
void *cmd_queue_addr;
static void init(void)
{
    task_queue = (struct all_ring_buffer *)cmd_queue_addr;
}
#else
static void init(void)
{
    int tzd_fd = open(DEVICE_NAME, O_RDWR);
    GGML_ASSERT(tzd_fd >= 0);
    void *addr = mmap(NULL, CMD_QUEUE_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, tzd_fd, 0);
    GGML_ASSERT(addr != MAP_FAILED);
    task_queue = (struct all_ring_buffer *)addr;
}
#endif

void set_io_model_path(const char *io_model_path) {
    std::call_once(task_queue_once, init);
    strcpy(task_queue->io_model_path, io_model_path);
}

size_t io_align_up(size_t off)
{
    return ROUND_UP(off, PAGE_SIZE);
}

size_t io_align_down(size_t off)
{
    return ROUND_DOWN(off, PAGE_SIZE);
}

#ifdef LLAMA_USE_CHCORE_API
#include <chcore/memory.h>
#include <chcore/syscall.h>
#include <chcore/llm.h>
#include <chcore/bug.h>
void io_rpc(void) {
    struct smc_registers req = {0};
    req.x1 = SMC_EXIT_SHADOW;
    req.x2 = 4;
    int ret = usys_tee_switch_req(&req);
    BUG_ON(ret != 0);
}
#else
static std::once_flag io_init_once;
void io_rpc(void) {
    extern void io_init(const char *model_path);
    std::call_once(io_init_once, io_init, task_queue->io_model_path);
    extern void io_step(all_ring_buffer *task_queue);
    io_step(task_queue);
}
#endif

std::atomic<int64_t> io_time;
std::atomic<size_t> io_size;

int on_fly_cnt = 0;

void io_launch(size_t off, size_t size, int cma_index, int entry_index, task_entry entry)
{
#ifdef TZ_LLM_MEASURE
    auto start = get_micro();
#endif
    std::call_once(task_queue_once, init);
    size_t seg_begin = io_align_down(off);
    size_t seg_end = io_align_up(off + size);
    io_task task = {
        .cma_index = cma_index,
        .entry_index = entry_index,
        .len = seg_end - seg_begin,
        .io_seg = {
            .off = seg_begin,
            .len = seg_end - seg_begin,
        },
        .pipeline = entry.task.get(),
        .is_measurement = false,
    };
    {
        std::lock_guard<std::mutex> _(tasks_lock);
        tasks.emplace(entry.task.get(), entry);
    }
    task_queue->io_tasks.produce(&task);
    ++on_fly_cnt;

    io_rpc();
#ifdef TZ_LLM_MEASURE
    io_size += size;
    io_time += get_micro() - start;
#endif
}

static std::optional<task_entry> __io_try_get(void)
{
    std::call_once(task_queue_once, init);
    io_result result;
    if (!on_fly_cnt)
        return std::nullopt;
    io_rpc();
    if (task_queue->io_results.consume(&result) == 0) {
        --on_fly_cnt;
        {
            std::lock_guard<std::mutex> _(tasks_lock);
            return tasks.at(result.pipeline);
        }
    }
    return std::nullopt;
}

std::optional<task_entry> io_try_get(void)
{
#ifdef TZ_LLM_MEASURE
    auto start = get_micro();
#endif
    return __io_try_get();
#ifdef TZ_LLM_MEASURE
    io_time += get_micro() - start;
#endif
}

void record_measure(double ttft, double decoding_thpt)
{
    std::call_once(task_queue_once, init);
    io_task task = {
        .is_measurement = true,
        .ttft = ttft,
        .decoding_thpt = decoding_thpt,
    };
    task_queue->io_tasks.produce(&task);
    io_rpc();
}
