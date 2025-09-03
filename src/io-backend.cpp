#include "io.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <iostream>
#include <cstring>
#include <memory>
#include <optional>
#include <libaio.h>
#include <queue>
#include "my_assert.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/ioctl.h>

struct aio_task {
    io_context_t ctx;
    void *pipeline;
    void *cma_buf;
    void *read_buf;
    size_t len;
    aio_task(io_context_t ctx, void *pipeline, void *cma_buf, void *read_buf, size_t len)
        : ctx(ctx), pipeline(pipeline), cma_buf(cma_buf), read_buf(read_buf), len(len) {}
};

#define IO_BLK_SIZE (2 << 20)

static std::queue<std::shared_ptr<aio_task>> tasks;
static int fd, tzd_fd;
// static const char *model_path = "/data/ssd/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
#if DUMMY_WEIGHT
static void *global_read_buf;
static size_t global_read_buf_len;
#endif

struct llm_client_op_pages {
	int cma_index;
	int entry_index;
	unsigned long size;
};

#define DEVICE_NAME "/dev/tc_ns_client"
#define TC_NS_CLIENT_IOC_MAGIC  't'
#define LLM_CLIENT_IOCTL_SET_PAGES \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 27, struct llm_client_op_pages)

static std::queue<io_context_t> ctxs;

io_context_t get_ctx(void) {
    if (ctxs.empty()) {
        io_context_t new_ctx = NULL;
        GGML_ASSERT(io_setup(1, &new_ctx) == 0);
        ctxs.push(new_ctx);
    }
    auto ctx = ctxs.front();
    ctxs.pop();
    return ctx;
}

void put_ctx(io_context_t ctx) {
    ctxs.push(ctx);
}

static void *get_buf(int cma_index, int entry_index, size_t len) {
    if (cma_index == -1)
        return NULL;
    struct llm_client_op_pages index = {
        .cma_index = cma_index,
        .entry_index = entry_index,
    };
    int ret = ioctl(tzd_fd, LLM_CLIENT_IOCTL_SET_PAGES, &index);
    GGML_ASSERT(ret >= 0);

    void *addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, tzd_fd, 0);
    GGML_ASSERT(addr != MAP_FAILED);

    return addr;
}

static void launch_io(void *dst, int fd, const io_seg &io_seg, void *pipeline) {
#if DUMMY_WEIGHT
    if (io_seg.len > global_read_buf_len) {
        printf("[warn] extend global read buffer from %ldB to %ldB\n", global_read_buf_len, io_seg.len);
        // munmap(global_read_buf, global_read_buf_len);
        global_read_buf = mmap(NULL, io_seg.len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        global_read_buf_len = io_seg.len;
    }
    void *read_buf = global_read_buf;
#else
    void *read_buf = mmap(NULL, io_seg.len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
    GGML_ASSERT(read_buf != MAP_FAILED);

    auto task = std::make_shared<aio_task>(get_ctx(), pipeline, dst, read_buf, io_seg.len);
    
    struct iocb *cbs[1];
    iocb cb;
    memset(&cb, 0, sizeof(cb));
    io_prep_pread(&cb, fd, read_buf, io_seg.len, io_seg.off);
    cbs[0] = &cb;

    GGML_ASSERT(io_submit(task->ctx, 1, cbs) == 1);
    tasks.push(task);
}

static void *wait_io(void) {
    struct io_event event;
    timespec timeout = { .tv_sec = 0, .tv_nsec = 0 };
    if (tasks.empty()) return NULL;
    auto task = tasks.front();
    int ret = io_getevents(task->ctx, 1, 1, &event, &timeout);
    GGML_ASSERT(ret >= 0);
    if (ret > 0) {
        GGML_ASSERT((int)event.res >= 0);
#if not(DUMMY_WEIGHT)
        memcpy(task->cma_buf, task->read_buf, task->len);
        munmap(task->read_buf, task->len);
#endif
        tasks.pop();
        put_ctx(task->ctx);
        return task->pipeline;
    }
    return NULL;
}

#define IO_TEST_FILE "/data/ssd/Meta-Llama-3-8B-Instruct.Q8_0.gguf"
#define IO_TEST_FILE_SIZE (8UL << 30)
#define IO_PRE_LAUNCH_CNT (16)

void io_init(const char *model_path) {
    printf("backend %s %d %s\n", __func__, __LINE__, model_path);
    tzd_fd = open(DEVICE_NAME, O_RDWR);
    GGML_ASSERT(tzd_fd > 0);

    fd = open(model_path, O_RDONLY | O_DIRECT);
    GGML_ASSERT(fd != -1);

#if DUMMY_WEIGHT
    global_read_buf = mmap(NULL, IO_BLK_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    global_read_buf_len = IO_BLK_SIZE;
    if (0) {
        printf("begin io test\n");
        auto start = get_micro();
        fd = open(IO_TEST_FILE, O_RDONLY | O_DIRECT);
        int all = 0, wait = 0;
        for (size_t i = 0; i < IO_TEST_FILE_SIZE; i += IO_BLK_SIZE) {
            struct io_seg io_seg = {
                .off = i,
                .len = IO_BLK_SIZE,
            };
            launch_io(global_read_buf, fd, io_seg, (void *)1);
            if (++all >= IO_PRE_LAUNCH_CNT) {
                while (wait_io());
                ++wait;
            }
        }
        for (; wait < all; wait++) {
            while (wait_io());
        }
        printf("io test %ld us thpt %.2f GB/s\n", get_micro() - start, 0.001f * IO_TEST_FILE_SIZE / (get_micro() - start));
    }
#endif
}

static void write_measurement(const io_task &task) {
    // Use POSIX shared memory object under /dev/shm
    const char *shm_name = "/current_measure";  // results in /dev/shm/current_measure
    int fd = shm_open(shm_name, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        perror("shm_open");
        return;
    }

    FILE *fp = fdopen(fd, "w");
    if (!fp) {
        perror("fdopen");
        close(fd);
        return;
    }

    // Overwrite with the latest measurement (no append)
    // Exact format requested by user
    fprintf(fp, "ttft: %.2f\ndecoding_thpt: %.2f\n", task.ttft, task.decoding_thpt);
    fflush(fp);
    fsync(fd);
    fclose(fp); // also closes fd
}

void io_step(all_ring_buffer *task_queue) {
    io_task task;
    while (task_queue->io_tasks.consume(&task) == 0) {
        if (task.is_measurement) {
            write_measurement(task);
            return;
        } else {
            void *buf = get_buf(task.cma_index, task.entry_index, task.len);
            launch_io(buf, fd, task.io_seg, task.pipeline);
        }
    }
    void *pipeline;
    while (pipeline = wait_io()) {
        io_result result = {
            .pipeline = pipeline
        };
        task_queue->io_results.produce(&result);
    }
}
