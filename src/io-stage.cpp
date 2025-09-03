#include "ggml.h"
#include "pipeline.h"
#include "interface.h"
#include "io-frontend.h"
#include <algorithm>

class IOTask : public std::enable_shared_from_this<IOTask>, public Task {
public:
    size_t off;
    size_t len;
    int cma_index;
    int entry_index;
    std::shared_ptr<Pipeline> pipeline;
    IOTask(size_t off, size_t len, int cma_index, int entry_index, std::shared_ptr<Pipeline> pipeline)
        : off(off), len(len), cma_index(cma_index), entry_index(entry_index), pipeline(pipeline) {}
    void step(void) override {
        std::shared_ptr<Task> self = shared_from_this();
        io_launch(off, len, cma_index, entry_index, task_entry(pipeline, self));
    }
};

extern bool is_strawman;
#define BLOCK_SIZE (is_strawman ? (8UL << 30) : (4UL << 20))

void IOStage::start(void *input)
{
    auto msg = (alloc_io_msg *)input;
    buf = msg->buf;
    cma_indexes.swap(msg->cma_indexes);
    cnt_to_finish = cma_indexes.size() * 2;
#ifdef LLAMA_USE_CHCORE_API
    if (!msg->paddr.empty()) {
        id_msg.cma_region.resize(msg->paddr.size());
        for (int cma_index = 0; cma_index < msg->paddr.size(); cma_index++) {
            std::sort(msg->paddr[cma_index].begin(), msg->paddr[cma_index].end());
            if (msg->paddr[cma_index].empty())
                id_msg.cma_region[cma_index] = {0, 0};
            else
                id_msg.cma_region[cma_index] = {msg->paddr[cma_index].front().first, msg->paddr[cma_index].back().second};
        }
    }
#endif
}

std::pair<std::shared_ptr<Task>, bool> IOStage::get_task(void *)
{
    GGML_ASSERT(pipeline);
    GGML_ASSERT(!cma_indexes.empty());
    auto [cma_index, entry_index, cma_offset, cma_size] = cma_indexes.back();
    cma_indexes.pop_back();
    return { std::make_shared<IOTask>(io_align_down(off) + cma_offset, cma_size, cma_index, entry_index, pipeline), cma_indexes.empty() };
}

bool IOStage::submit(std::shared_ptr<Task> task)
{
    auto old_nr = cnt_to_finish.fetch_sub(1);
    if (old_nr == 1)
        return true;
    return false;
}

void *IOStage::get_msg(void)
{
    GGML_ASSERT(buf);
    id_msg.buf = buf + off - io_align_down(off);
    return &id_msg;
}

void IOStage::rollback(void)
{

}

void IOStage::set_pipeline(std::shared_ptr<Pipeline> pipe)
{
    pipeline = pipe;
}
