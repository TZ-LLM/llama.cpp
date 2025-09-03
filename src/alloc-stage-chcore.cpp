#include "ggml.h"
#include "pipeline.h"
#include <atomic>
#include <io-frontend.h>
#include <chcore/memory.h>
#include <chcore/syscall.h>
#include <chcore/llm.h>
#include <chcore/bug.h>

#define ROUND_UP(x, n)   (((x) + (n)-1) & ~((n)-1))
#define PAGE_SIZE 0x1000

std::mutex alloc_mtx;
std::mutex gather_mtx;
std::mutex cma_mtx[TZASC_NR];

struct tzasc_cma_meta *tzasc_cma_meta_arr;

void tzasc_cma_init(void) {
    vaddr_t vaddr;
    vaddr = chcore_alloc_vaddr(PAGE_SIZE << 10);
    BUG_ON(vaddr == 0);

    int ret = usys_map_tzasc_cma_meta(vaddr);
    BUG_ON(ret != 0);

    tzasc_cma_meta_arr = (struct tzasc_cma_meta *)vaddr;
    for (int i = 0; i < TZASC_NR; i++) {
        auto tzasc_cma_meta = tzasc_cma_meta_arr + i;
        printf("%s %d base %#lx size %#lx\n", __func__, __LINE__, tzasc_cma_meta->base, tzasc_cma_meta->size);
    }
}

static std::once_flag tzasc_flag;

int push_pages(size_t len, int cma_index) {
    std::call_once(tzasc_flag, tzasc_cma_init);

    GGML_ASSERT(cma_index >= 0 && cma_index < TZASC_NR);
    auto tzasc_cma_meta = tzasc_cma_meta_arr + cma_index;

    struct smc_registers req = {0};
    req.x1 = SMC_EXIT_SHADOW;
    req.x2 = 1;
    req.x3 = ROUND_UP(len, PAGE_SIZE) | cma_index;
    int ret = usys_tee_switch_req(&req);
    BUG_ON(ret < 0);
    return ret;
}
int pop_pages(int cma_index) {
    std::call_once(tzasc_flag, tzasc_cma_init);

    GGML_ASSERT(cma_index >= 0 && cma_index < TZASC_NR);
    auto tzasc_cma_meta = tzasc_cma_meta_arr + cma_index;

    struct smc_registers req = {0};
    req.x1 = SMC_EXIT_SHADOW;
    req.x2 = 0;
    req.x3 = cma_index;
    int ret = usys_tee_switch_req(&req);
    BUG_ON(ret != 0);
    return tzasc_cma_meta->count;
}

std::atomic<int64_t> cma_time;
std::atomic<size_t> cma_size;

extern bool is_strawman;
#define BLOCK_SIZE (is_strawman ? (8UL << 30) : (4UL << 20))

class AllocTask : public Task {
public:
    int tzd_fd;
    size_t size;
    vaddr_t vaddr;
    int cma_index;
    int entry_index;

    AllocTask(size_t size, vaddr_t vaddr, int cma_index = -1): size(size), vaddr(vaddr), cma_index(cma_index) {

    }
    void step(void) override {
#ifdef TZ_LLM_MEASURE
        auto start = get_micro();
#endif
        std::lock_guard<std::mutex> _(cma_mtx[cma_index]);
        entry_index = push_pages(size, cma_index);
        auto tzasc_cma_meta = tzasc_cma_meta_arr + cma_index;
        GGML_ASSERT(usys_map_tzasc_cma_pmo(vaddr, size, tzasc_cma_meta->entry[entry_index].paddr) == 0);
        // printf("%s %d: rgn %d paddr %p\n", __func__, __LINE__, cma_index, (void *)tzasc_cma_meta->entry[entry_index].paddr);
        // sprintf((char *)vaddr, "okokokok %d", cma_index);
        // GGML_ASSERT(usys_config_tzasc(8 + cma_index, tzasc_cma_meta->base >> 20, (tzasc_cma_meta->entry[entry_index].paddr + tzasc_cma_meta->entry[entry_index].size) >> 20) == 0);
        // while (1);
#ifdef TZ_LLM_MEASURE
        cma_size += ROUND_UP(size, BLOCK_SIZE);
        cma_time += get_micro() - start;
#endif
    }
};

std::atomic<int> last_pos;

AllocStage::AllocStage(size_t off, size_t len): addr(NULL) {
    size = io_align_up(off + len) - io_align_down(off);
    addr = (void *)chcore_alloc_vaddr(size);
    msg.buf = addr;
    msg.paddr.resize(TZASC_NR);
    GGML_ASSERT(addr);

    all_block_nr = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int cur_start = last_pos.fetch_add(all_block_nr) % TZASC_NR;
    int cur_end = (cur_start + all_block_nr) % TZASC_NR;
    for (int i = 0; i < TZASC_NR; i++) {
        block_nr[i] = all_block_nr / TZASC_NR;
        if (cur_start <= cur_end) {
            if (cur_start <= i && i < cur_end) {
                block_nr[i]++;
            }
        } else {
            if (cur_start <= i || i < cur_end) {
                block_nr[i]++;
            }
        }
    }

    GGML_ASSERT(sizeof(block_nr) / sizeof(all_block_nr) >= TZASC_NR);

    int test_sum = 0;
    for (int i = 0; i < TZASC_NR; i++) {
        test_sum += block_nr[i];
    }
    GGML_ASSERT(test_sum == all_block_nr);
}

void AllocStage::start(void *input)
{
    (void)input;
    for (int i = 0; i < TZASC_NR; i++) {
        finished_nr = 0;
        get_nr[i] = 0;
        submit_pos = 0;
    }
    msg.cma_indexes.clear();
}

std::pair<std::shared_ptr<Task>, bool> AllocStage::get_task(void *arg)
{
    std::lock_guard<std::mutex> _(submit_pos_mtx);
    int cma_index = (int)(long)arg;
    GGML_ASSERT(submit_pos < size);

    for (int i = 0; i < TZASC_NR; i++) {
        GGML_ASSERT(get_nr[i] <= block_nr[i]);
    }
    if (get_nr[cma_index] == block_nr[cma_index]) {
        for (int i = 0; i < TZASC_NR; i++) {
            if (get_nr[i] < block_nr[i]) {
                cma_index = i;
                break;
            }
        }
    }
    get_nr[cma_index]++;

    auto task = std::make_shared<AllocTask>(ROUND_UP(std::min(BLOCK_SIZE, size - submit_pos), PAGE_SIZE), (vaddr_t)addr + submit_pos, cma_index);
    submit_pos += BLOCK_SIZE;
    return { task, submit_pos >= size };
}

bool AllocStage::submit(std::shared_ptr<Task> task)
{
    AllocTask *alloc_task = dynamic_cast<AllocTask *>(task.get());
    GGML_ASSERT(alloc_task);
    {
        std::lock_guard<std::mutex> _(gather_mtx);
        msg.cma_indexes.push_back({alloc_task->cma_index, alloc_task->entry_index, alloc_task->vaddr - (vaddr_t)addr, alloc_task->size});
        msg.paddr[alloc_task->cma_index].push_back({
            tzasc_cma_meta_arr[alloc_task->cma_index].entry[alloc_task->entry_index].paddr,
            tzasc_cma_meta_arr[alloc_task->cma_index].entry[alloc_task->entry_index].paddr + alloc_task->size
        });
    }
    auto old_nr = finished_nr.fetch_add(1);
    if (old_nr + 1 == all_block_nr)
        return true;
    return false;
}

void *AllocStage::get_msg(void)
{
    GGML_ASSERT(addr);
    return &msg;
}

void AllocStage::rollback(void)
{
    int ret;

    GGML_ASSERT(addr);

    for (int cma_index = 0; cma_index < TZASC_NR; cma_index++)
        for (int i = 0; i < block_nr[cma_index]; i++)
            pop_pages(cma_index);
}
