#include "mapping.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <iostream>
#include <cstring>
#include <memory>
#include <vector>
#include <optional>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <atomic>
#include "interface.h"

#define ROUND_UP(x, n)   (((x) + (n)-1) & ~((n)-1))
#define PAGE_SIZE 0x1000

#define DEVICE_NAME "/dev/tc_ns_client"
#define TC_NS_CLIENT_IOC_MAGIC  't'
#define LLM_CLIENT_IOCTL_PUSH_PAGES \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 25, unsigned long)
#define LLM_CLIENT_IOCTL_POP_PAGES \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 26, unsigned long)
#define LLM_CLIENT_IOCTL_SET_PAGES \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 27, int)

cma_region::cma_region(int tzd_fd, size_t size, std::function<void(void)> destructor)
    : done(false), len(size), destructor(destructor), tzd_fd(tzd_fd) {}

std::atomic<int64_t> cma_time;

void cma_region::ready(void) {
    if (done) return;
#ifdef TZ_LLM_MEASURE
    auto start = get_micro();
#endif

    int ret;
    ret = ioctl(tzd_fd, LLM_CLIENT_IOCTL_PUSH_PAGES, &len);
    if (ret < 0) {
        printf("%s %d err %d\n", __func__, __LINE__, errno);
    }
    GGML_ASSERT(ret >= 0);

    int index = ret;
    ret = ioctl(tzd_fd, LLM_CLIENT_IOCTL_SET_PAGES, &index);
    if (ret < 0) {
        printf("%s %d err %d\n", __func__, __LINE__, errno);
    }
    GGML_ASSERT(ret >= 0);

    addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, tzd_fd, 0);
    GGML_ASSERT(addr);

    // addr = malloc(len);

    done = true;

#ifdef TZ_LLM_MEASURE
    cma_time += get_micro() - start;
#endif
}

cma_region::~cma_region() {
    munmap(addr, len);
    int ret = ioctl(tzd_fd, LLM_CLIENT_IOCTL_PUSH_PAGES, &len);
    if (ret < 0) {
        printf("%s %d err %d\n", __func__, __LINE__, errno);
    }
    GGML_ASSERT(ret >= 0);
    destructor();
}

mappings::mappings(void) {
    const char *path = "/dev/dma_heap/reserved";
    fd = open(path, O_RDWR);
    GGML_ASSERT(fd > 0);
}

mappings::~mappings(void) {
    while (!tensors.empty())
        tensors.pop_back();
    close(fd);
}

void mappings::push(size_t offset, size_t size, std::function<void(void)> destructor) {
    std::shared_ptr<struct cma_region> cma_region;
    auto cma_iter = cma_regions.find(offset);
    if (cma_iter == cma_regions.end() || !(cma_region = cma_iter->second.lock())) {
        cma_region = std::make_shared<struct cma_region>(fd, size, destructor);
        cma_regions.emplace(offset, cma_region);
    }
    std::lock_guard<std::mutex> _(work_queue_mutex);
    work_queue.emplace(cma_region);
}

void mappings::step(void) {
    std::shared_ptr<cma_region> cma_region;
    {
        std::lock_guard<std::mutex> _(work_queue_mutex);
        if (!work_queue.empty()) {
            cma_region = work_queue.front();
            work_queue.pop();
        }
    }
    if (cma_region) {
        cma_region->ready();
        std::lock_guard<std::mutex> _(tensors_mutex);
        tensors.push_back(cma_region);
    }
}

void *mappings::try_get(size_t offset) {
    std::shared_ptr<struct cma_region> cma_region;
    auto cma_iter = cma_regions.find(offset);
    if (cma_iter == cma_regions.end())
        return NULL;
    cma_region = cma_iter->second.lock();
    if (!cma_region)
        return NULL;
    if (!cma_region->done)
        return NULL;
    return cma_region->addr;
}

void *mappings::get(size_t offset) {
    void *addr = try_get(offset);
    while (!addr) {
        step();
        addr = try_get(offset);
    }
    return addr;
}

void mappings::pop(void) {
    std::lock_guard<std::mutex> _(tensors_mutex);
    tensors.pop_back();
}