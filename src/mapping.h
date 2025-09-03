#pragma once

#include "ggml.h"
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <mutex>
#include <queue>

struct cma_region {
    bool done;
    void *addr;
    size_t len;
    std::function<void(void)> destructor;

    int tzd_fd;

    cma_region(int tzd_fd, size_t size, std::function<void(void)> destructor);
    void ready(void);
    ~cma_region();
};

struct mappings {
    std::unordered_map<size_t, std::weak_ptr<cma_region>> cma_regions;
    std::mutex tensors_mutex;
    std::vector<std::shared_ptr<cma_region>> tensors;
    int fd;
    std::mutex work_queue_mutex;
    std::queue<std::shared_ptr<cma_region>> work_queue;

    mappings(void);
    ~mappings(void);

    void push(size_t offset, size_t size, std::function<void(void)> destructor);
    void step(void);
    void *try_get(size_t offset);
    void *get(size_t offset);
    void pop(void);
};
