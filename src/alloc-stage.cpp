#include "ggml.h"
#include "pipeline.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <atomic>
#include <io-frontend.h>

#define ROUND_UP(x, n)   (((x) + (n)-1) & ~((n)-1))
#define PAGE_SIZE 0x1000

struct llm_client_op_pages {
	int cma_index;
	int entry_index;
	unsigned long size;
};

#define DEVICE_NAME "/dev/tc_ns_client"
#define TC_NS_CLIENT_IOC_MAGIC  't'
#define LLM_CLIENT_IOCTL_PUSH_PAGES \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 25, struct llm_client_op_pages)
#define LLM_CLIENT_IOCTL_POP_PAGES \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 26, struct llm_client_op_pages)
#define LLM_CLIENT_IOCTL_SET_PAGES \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 27, struct llm_client_op_pages)

std::atomic<int64_t> cma_time;
std::atomic<size_t> cma_size;

std::mutex alloc_mtx;

class AllocTask : public Task {
public:
    int tzd_fd;
    size_t size;
    void *addr;
    int cma_index;
    int entry_index;

    AllocTask(size_t size): size(size), addr(NULL) {
#if not(DUMMY_WEIGHT)
        tzd_fd = open(DEVICE_NAME, O_RDWR);
        GGML_ASSERT(tzd_fd >= 0);
#endif
    }
#if DUMMY_WEIGHT
    void step(void) override {
        cma_index = entry_index = -1;
        addr = malloc(size);
    }
#else
    void step(void) override {
        std::lock_guard<std::mutex> _(alloc_mtx);
#ifdef TZ_LLM_MEASURE
            auto start = get_micro();
#endif
            struct llm_client_op_pages index;

            GGML_ASSERT(addr == NULL);

            int ret;
            index.size = size;
            ret = ioctl(tzd_fd, LLM_CLIENT_IOCTL_PUSH_PAGES, &index);
            GGML_ASSERT(ret == 0);

            cma_index = index.cma_index;
            entry_index = index.entry_index;
            ret = ioctl(tzd_fd, LLM_CLIENT_IOCTL_SET_PAGES, &index);
            GGML_ASSERT(ret == 0);

            addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, tzd_fd, 0);
            GGML_ASSERT(addr != MAP_FAILED);

            // addr = malloc(size);

#ifdef TZ_LLM_MEASURE
            cma_size += size;
            cma_time += get_micro() - start;
#endif
    }
#endif
};

AllocStage::AllocStage(size_t off, size_t len): addr(NULL) {
    size = io_align_up(off + len) - io_align_down(off);
    tzd_fd = open(DEVICE_NAME, O_RDWR);
    GGML_ASSERT(tzd_fd >= 0);
}

void AllocStage::start(void *input)
{
    (void)input;
}

std::pair<std::shared_ptr<Task>, bool> AllocStage::get_task(void *)
{
    return { std::make_shared<AllocTask>(size), true };
}

bool AllocStage::submit(std::shared_ptr<Task> task)
{
    AllocTask *alloc_task = dynamic_cast<AllocTask *>(task.get());
    GGML_ASSERT(alloc_task);
    GGML_ASSERT(!addr);
    addr = alloc_task->addr;
    msg.buf = alloc_task->addr;
    msg.cma_indexes.push_back({alloc_task->cma_index, alloc_task->entry_index, 0, size});
    return true;
}

void *AllocStage::get_msg(void)
{
    GGML_ASSERT(addr);
    return &msg;
}

void AllocStage::rollback(void)
{
#if DUMMY_WEIGHT
    free(addr);
#else
    int ret;

    GGML_ASSERT(addr);

    ret = munmap(addr, size);
    GGML_ASSERT(ret == 0);

    ret = ioctl(tzd_fd, LLM_CLIENT_IOCTL_POP_PAGES, &size);
    GGML_ASSERT(ret == 0);

#endif
    addr = NULL;
}

