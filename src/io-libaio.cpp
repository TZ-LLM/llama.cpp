#include "io.h"
#include "ggml.h"
#include <fcntl.h>
#include <unistd.h>
#include <libaio.h>
#include <unordered_map>
#include <string>
#include <mutex>
#include <iostream>
#include <cstring>
#include <memory>
#include <optional>

std::mutex lock;
std::unordered_map<void *, io_context_t> aios;
int counter = 0;

void launch_io(void *dst, int fd, size_t off, size_t len) {
    std::lock_guard<std::mutex> _(lock);

    GGML_ASSERT(aios.find(dst) == aios.end());

    io_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    GGML_ASSERT(io_setup(1, &ctx) == 0);
    
    iocb cb;
    struct iocb *cbs[1];
    memset(&cb, 0, sizeof(cb));
    io_prep_pread(&cb, fd, dst, len, off);
    cbs[0] = &cb;

    GGML_ASSERT(io_submit(ctx, 1, cbs) == 1);

    aios[dst] = ctx;
}

void wait_io(void *dst) {
    io_context_t ctx;
    struct io_event events[1];

    {
        std::lock_guard<std::mutex> _(lock);
        GGML_ASSERT(aios.find(dst) != aios.end());
        ctx = aios[dst];
        aios.erase(aios.find(dst));
    }

    GGML_ASSERT(io_getevents(ctx, 1, 1, events, NULL) == 1);
    // GGML_ASSERT(events[0].res == events[0].obj->u.c.nbytes);
}
