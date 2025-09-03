#pragma once

#include <map>
#include <vector>
#include <optional>
#include "io.h"
#include "pipeline.h"

struct task_entry {
    std::shared_ptr<Pipeline> pipeline;
    std::shared_ptr<Task> task;
    task_entry(std::shared_ptr<Pipeline> pipeline, std::shared_ptr<Task> task)
        : pipeline(pipeline), task(task) {}
};

size_t io_align_up(size_t off);
size_t io_align_down(size_t off);
void io_launch(size_t off, size_t size, int cma_index, int entry_index, task_entry entry);
std::optional<task_entry> io_try_get(void);