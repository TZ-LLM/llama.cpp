#pragma once

#include "interface.h"

struct io_buf {
    void *buf;
    size_t size;
    io_buf(void *buf, size_t size): buf(buf), size(size) {}
};
