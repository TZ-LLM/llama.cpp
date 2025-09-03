#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "ggml.h"
#include <sys/ioctl.h>
#include <cstring>
#include <pthread.h>
#include <cstdlib>
#include <thread>
#include <vector>
#include <getopt.h>
#include <string>

struct llm_client_op_pages {
	unsigned long entry_begin;
	unsigned long entry_end;
};

#define DEVICE_NAME "/dev/tc_ns_client"
#define TC_NS_CLIENT_IOC_MAGIC  't'
#define LLM_CLIENT_IOCTL_RUN \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 24, int)
#define LLM_CLIENT_IOCTL_ALLOC_PAGES \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 25, struct llm_client_op_pages)
#define LLM_CLIENT_IOCTL_FREE_PAGES \
	_IOWR(TC_NS_CLIENT_IOC_MAGIC, 26, struct llm_client_op_pages)

enum smc_loop_exit {
    SMC_LOOP_EXIT_FINISH = 1,
    SMC_LOOP_EXIT_NPU_SUBMIT,
    SMC_LOOP_EXIT_NPU_DONE,
    SMC_LOOP_EXIT_IO_STEP,
};

extern void ca_backend_init(const char *io_model_path);
extern void ca_backend_io_step(void);
extern void ca_backend_set_cache(int p);
extern void ca_backend_set_prompt(const char *model, int len);
extern void ca_backend_set_n(int n);
extern void ca_backend_set_strawman(bool is_strawman);

void ca_thread(int fd, int index) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(4 + index, &cpuset);
    GGML_ASSERT(pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0);

    printf("%s %d run llm\n", __func__, __LINE__);
    while (true) {
        int out_cmd;
        unsigned long ret = ioctl(fd, LLM_CLIENT_IOCTL_RUN, fd, &out_cmd);
        // printf("%s %d: ret %d error %d\n", __func__, __LINE__, ret, errno);
        // GGML_ASSERT(ret >= 0);
        // switch (out_cmd) {
        // case SMC_LOOP_EXIT_IO_STEP:
        //     printf("%s %d: receive SMC_LOOP_EXIT_IO_STEP\n", __func__, __LINE__);
            ca_backend_io_step();
        //     break;
        // default:
        //     GGML_ASSERT(false);
        //     break;
        // }
    }
}
int main(int argc, char *argv[]) {
    int fd;
    char *mapped_mem;
    size_t length = getpagesize();

    fd = open(DEVICE_NAME, O_RDWR);
    GGML_ASSERT(fd > 0);

    mapped_mem = (char *)mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    GGML_ASSERT(mapped_mem != MAP_FAILED);

    printf("msg in shm %s\n", (char *)mapped_mem);

    const char *model = "qwen";
    int cache = 0;
    int len = 128;
    int n = 64;
    int strawman = 0;

    struct option long_options[] = {
        {"model", required_argument, NULL, 'm'},
        {"cache", required_argument, NULL, 'c'},
        {"len", required_argument, NULL, 'l'},
        {"nn", required_argument, NULL, 'n'},
        {"strawman", required_argument, NULL, 's'},
        {NULL, 0, NULL, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:c:l:n:s:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'm':
                model = optarg;
                break;
            case 'c':
                cache = atoi(optarg);
                break;
            case 'l':
                len = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 's':
                strawman = atoi(optarg);
                printf("strawman %d\n", strawman);
                break;
            case '?':
                // Invalid argument or missing value
                break;
        }
    }

    std::string model_string(model);
    const char *io_model_path;
    if (model_string == "tinyllama") {
        io_model_path = "/data/ssd/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
    } else if (model_string == "gemma") {
        io_model_path = "/data/ssd/gemma-2-2b-it-Q8_0.gguf";
    } else if (model_string == "qwen") {
        io_model_path = "/data/ssd/qwen2.5-3b-instruct-q8_0.gguf";
    } else if (model_string == "phi") {
        io_model_path = "/data/ssd/Phi-3-mini-4k-instruct.Q8_0.gguf";
    } else if (model_string == "llama") {
        io_model_path = "/data/ssd/Meta-Llama-3-8B-Instruct.Q8_0.gguf";
    } else {
        GGML_ABORT("invalid model %s\n", model);
    }

    ca_backend_init(io_model_path);
    ca_backend_set_cache(cache);
    ca_backend_set_prompt(model, len);
    ca_backend_set_n(n);
    ca_backend_set_strawman((bool)strawman);

    std::vector<std::thread> ca_threads;
    for (int i = 0; i < 4; i++) {
        ca_threads.emplace_back(ca_thread, fd, i);
    }
    for (int i = 0; i < 4; i++) {
        ca_threads[i].join();
    }

    // GGML_ASSERT(ioctl(fd, LLM_CLIENT_IOCTL_RUN, fd) >= 0);

    while (1);

#define LLM_MODEL_SIZE (4ul * 1024 * 1024 * 1024)
    struct llm_client_op_pages op = { .entry_begin = 0, .entry_end = LLM_MODEL_SIZE / (1 << 12) };
    printf("%s %d\n", __func__, __LINE__);
    GGML_ASSERT(ioctl(fd, LLM_CLIENT_IOCTL_ALLOC_PAGES, &op) >= 0);
    printf("%s %d\n", __func__, __LINE__);
    GGML_ASSERT(ioctl(fd, LLM_CLIENT_IOCTL_RUN, fd) >= 0);
    printf("%s %d\n", __func__, __LINE__);
    GGML_ASSERT(ioctl(fd, LLM_CLIENT_IOCTL_FREE_PAGES, &op) >= 0);
    printf("%s %d\n", __func__, __LINE__);

    close(fd);

    return 0;
}
