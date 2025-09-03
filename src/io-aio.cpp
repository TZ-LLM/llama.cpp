#include "io.h"
#include <fcntl.h>
#include <unistd.h>
#include <aio.h>
#include <unordered_map>
#include <string>
#include <mutex>
#include <iostream>
#include <cstring>
#include <memory>

std::mutex lock;
std::unordered_map<void *, std::shared_ptr<struct aiocb>> aios;
int counter = 0;

void launch_io(void *dst, int fd, size_t off, size_t len) {
    std::lock_guard<std::mutex> _(lock);

    GGML_ASSERT(aios.find(dst) == aios.end());

    auto aio_cb = std::make_shared<struct aiocb>();
    aio_cb->aio_fildes = fd;
    aio_cb->aio_buf = dst;
    aio_cb->aio_nbytes = len;
    aio_cb->aio_offset = off;
    int ret = aio_read(aio_cb.get());
    GGML_ASSERT(ret != -1);
    aios[dst] = aio_cb;
}

void wait_io(void *dst) {
    std::shared_ptr<aiocb> aio_cb;
    {
        std::lock_guard<std::mutex> _(lock);
        GGML_ASSERT(aios.find(dst) != aios.end());
        aio_cb = aios[dst];
        aios.erase(aios.find(dst));
    }
    const struct aiocb *aiocb_list[1] = { aio_cb.get() };
    int ret = aio_suspend(aiocb_list, 1, nullptr);
    GGML_ASSERT(ret != -1);
    GGML_ASSERT(aio_error(aio_cb.get()) == 0);
    ssize_t bytes_read = aio_return(aio_cb.get());
    GGML_ASSERT(bytes_read == aio_cb->aio_nbytes);
}
