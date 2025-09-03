#pragma once

#include <sys/types.h>
#include <unistd.h>
#include "my_assert.h"
#include <cstdio>
#include <pthread.h>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <atomic>

#define TZ_LLM_MEASURE

#define DUMMY_WEIGHT 1

#define ENABLE_PIPELINE 1

#define CMD_QUEUE_SHM_NAME "/cmd_queue_shm"
#define CMD_QUEUE_SHM_SIZE (256 * 4096)

#define IO_BUFFER_SHM_NAME "/io_buffer_shm"
#define IO_BUFFER_SHM_SIZE (600 * 1024 * 1024)
#define IO_BUFFER_NR (2)

#define IO_SEG_NR (10)

struct io_seg {
    size_t off;
    size_t len;
};

struct io_segs {
    int seg_nr;
    io_seg io_segs[4];
};

struct io_task {
    int cma_index;
    int entry_index;
    size_t len;
    struct io_seg io_seg;
    void *pipeline;

    bool is_measurement;
    double ttft;
    double decoding_thpt;
};

struct io_result {
    void *pipeline;
};

typedef enum {
    NPU_TASK_OP_CREATE_MEM_FROM_FD,
    NPU_TASK_OP_DESTROY_MEM,
    NPU_TASK_OP_MATMUL_CREATE,
    NPU_TASK_OP_MATMUL_DESTROY,
    NPU_TASK_OP_MATMUL_SET_CORE_MASK,
    NPU_TASK_OP_MATMUL_SET_IO_MEM,
    NPU_TASK_OP_MATMUL_RUN,
} npu_task_op_t;

struct npu_task {
    npu_task_op_t op;
};

struct npu_result {
    npu_task_op_t op;
};

typedef enum {
    PAGE_TASK_OP_ALLOC,
    PAGE_TASK_OP_FREE,
} page_task_op_t;

struct page_task {
    page_task_op_t op;
    unsigned long entry_begin;
    unsigned long entry_end;
    size_t off;
};

struct page_result {
    unsigned long entry_begin;
    unsigned long entry_end;
    size_t off;
};

template<typename T, int BUFFER_SIZE>
struct ring_buffer {
    T buffer[BUFFER_SIZE];
    std::atomic<int> head;
    std::atomic<int> tail;
    std::atomic<int> count;

    void init(void) {
        head = tail = count = 0;
    }

    ring_buffer(void): head(0), tail(0), count(0) {}
    int produce(const T *item) {
        GGML_ASSERT(this->count.load() != BUFFER_SIZE);
        while (true) {
            int _head = this->head.load();
            bool success = this->head.compare_exchange_strong(_head, (_head + 1) % BUFFER_SIZE);
            if (success) {
                memcpy(this->buffer + _head, item, sizeof(T));
                break;
            }
        }
        this->count++;
        return 0;
    }

    int consume(T *item) {
        if (this->count.load() == 0) {
            return -1;
        }
        while (true) {
            int _tail = this->tail.load();
            bool success = this->tail.compare_exchange_strong(_tail, (_tail + 1) % BUFFER_SIZE);
            if (success) {
                memcpy(item, this->buffer + _tail, sizeof(T));
                break;
            }
        }
        this->count--;
        return 0;
    }
};

const int IO_BUFFER_SIZE = 10240;
const int NPU_BUFFER_SIZE = 128;
const int PAGE_BUFFER_SIZE = 128;

struct all_ring_buffer {
    char io_model_path[256];
    char cache_p[256];
    char inner_model_path[256];
    char prompt[256];
    char n[256];
    bool is_strawman;

    ring_buffer<io_task, IO_BUFFER_SIZE> io_tasks;
    ring_buffer<io_result, IO_BUFFER_SIZE> io_results;
    // ring_buffer<npu_task, NPU_BUFFER_SIZE> npu_tasks;
    // ring_buffer<npu_result, NPU_BUFFER_SIZE> npu_results;
    // ring_buffer<page_task, PAGE_BUFFER_SIZE> page_tasks;
    // ring_buffer<page_result, PAGE_BUFFER_SIZE> page_results;

    void init(void) {
        GGML_ASSERT(sizeof(all_ring_buffer) <= CMD_QUEUE_SHM_SIZE);
        io_tasks.init();
        io_results.init();
        // npu_tasks.init();
        // npu_results.init();
        // page_tasks.init();
        // page_results.init();
    }
};

static inline int64_t get_micro(void) {
    sizeof(all_ring_buffer);
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
}


static_assert(sizeof(all_ring_buffer) <= CMD_QUEUE_SHM_SIZE, "sizeof(all_ring_buffer) > CMD_QUEUE_SHM_SIZE");

static inline void *shm_client(const char *shm_name, size_t shm_size) {
    int shm_fd = shm_open(shm_name, O_RDWR, 0666);
    GGML_ASSERT(shm_fd != -1);
    void *ptr = mmap(0, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    GGML_ASSERT(ptr != MAP_FAILED);
    return ptr;
}

static inline void *shm_server(const char *shm_name, size_t shm_size) {
    shm_unlink(shm_name);
    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    GGML_ASSERT(shm_fd != -1);
    GGML_ASSERT(ftruncate(shm_fd, shm_size) == 0);
    void *ptr = mmap(0, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    GGML_ASSERT(ptr != MAP_FAILED);
    return ptr;
}
