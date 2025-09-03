#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <thread>
#include <mutex>
#include "interface.h"

static int tzd_fd;
static int shm_fd;
static std::once_flag once_flag;
static all_ring_buffer *task_queues;

#define DEVICE_NAME "/dev/tc_ns_client"

void ca_backend_init(const char *io_model_path) {
    tzd_fd = open(DEVICE_NAME, O_RDWR);
    GGML_ASSERT(tzd_fd >= 0);
    void *addr = mmap(NULL, CMD_QUEUE_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, tzd_fd, 0);
    GGML_ASSERT(addr != MAP_FAILED);
    task_queues = (struct all_ring_buffer *)addr;
    task_queues->init();
    extern void io_init(const char *model_path);
    io_init(io_model_path);
}

void ca_backend_io_step(void) {
    GGML_ASSERT(task_queues);
    extern void io_step(all_ring_buffer *task_queues);
    io_step(task_queues);
}

void ca_backend_set_cache(int p) {
    GGML_ASSERT(task_queues);
    sprintf(task_queues->cache_p, "%d\0", p);
}

void ca_backend_set_prompt(const char *model, int len) {
    GGML_ASSERT(task_queues);
    sprintf(task_queues->prompt, "%s#%d\0", model, len);
    if (!strcmp(model, "tinyllama")) {
        strcpy(task_queues->inner_model_path, "tinyllama-1.1b-chat-v1.0.Q8_0-meta.gguf");
    } else if (!strcmp(model, "gemma")) {
        strcpy(task_queues->inner_model_path, "gemma-2-2b-it-Q8_0-meta.gguf");
    } else if (!strcmp(model, "qwen")) {
        strcpy(task_queues->inner_model_path, "qwen2.5-3b-instruct-q8_0-meta.gguf");
    } else if (!strcmp(model, "phi")) {
        strcpy(task_queues->inner_model_path, "Phi-3-mini-4k-instruct.Q8_0-meta.gguf");
    } else if (!strcmp(model, "llama")) {
        strcpy(task_queues->inner_model_path, "Meta-Llama-3-8B-Instruct.Q8_0-meta.gguf");
    } else {
        GGML_ABORT("model: %s\n", model);
    }
}

void ca_backend_set_n(int n) {
    GGML_ASSERT(task_queues);
    sprintf(task_queues->n, "%d\0", n);
}

void ca_backend_set_strawman(bool is_strawman) {
    GGML_ASSERT(task_queues);
    task_queues->is_strawman = is_strawman;
}
