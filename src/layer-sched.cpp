#include "pipeline.h"
#include "ggml.h"
#include "io-frontend.h"
#ifdef LLAMA_USE_CHCORE_API
#include <chcore/llm.h>
#endif

std::pair<std::shared_ptr<Pipeline>, std::shared_ptr<Task>> LayerScheduler::get_task(layer_queue_t &queue, void *arg)
{
    while (true) {
        if (queue.empty()) {
            return std::make_pair(nullptr, nullptr);
        }
        auto pipeline = queue.top();
        GGML_ASSERT(!pipeline->is_finished());
        auto task = pipeline->get_current_stage()->get_task(arg);
        if (task.second) {
            queue.pop();
        }
        return std::make_pair(pipeline, task.first);
    }
}

pid_t main_tid = -1;

#ifdef LLAMA_USE_CHCORE_API
std::atomic<int> index;
thread_local int my_cma_index = -1;

bool is_pipelining = false;

int get_cma_index(void) {
    if (my_cma_index == -1) {
        my_cma_index = index.fetch_add(1) % TZASC_NR;
    }
    return my_cma_index;
}
#else
int get_cma_index(void) {
    return 0;
}
#endif

int io_cnt = 0;

std::mutex io_lock;
bool LayerScheduler::step(void) {
extern bool is_strawman;
if (!is_strawman) {
    GGML_ASSERT(main_tid != -1);
    if (gettid() == main_tid) {
        auto entry = io_try_get();
        if (entry.has_value()) {
            io_cnt--;
            auto pipeline = entry->pipeline;
            auto task = entry->task;
            if (pipeline->get_current_stage()->submit(task)) {
                pipeline->finish_stage();
                if (!pipeline->is_finished()) {
                    enqueue(pipeline);
                }
            }
            // return false;
        }
    }

    std::pair<std::shared_ptr<Pipeline>, std::shared_ptr<Task>> res;
    {
        std::lock_guard<std::mutex> _(lock);

        while (true) {
            if (gettid() == main_tid && io_cnt <= 32) {
                res = get_task(io, NULL);
                if (res.first) {
                    io_cnt++;
                    break;
                }
            }
            res = get_task(decrypt, NULL);
            if (res.first) break;
            GGML_ASSERT(main_tid != -1);
#ifdef LLAMA_USE_CHCORE_API
            res = get_task(alloc, (void *)(long)get_cma_index());
#else
            if (gettid() == main_tid) {
                res = get_task(alloc, (void *)(long)get_cma_index());
            }
#endif
            if (res.first) break;
            return false;
        }
    }

    auto pipeline = res.first;
    auto task = res.second;
    GGML_ASSERT(pipeline && task);

    task->step();
    if (pipeline->get_current_stage()->submit(task)) {
        pipeline->finish_stage();
        if (!pipeline->is_finished()) {
            enqueue(pipeline);
        }
    }

    return true;
} else {
    bool is_io = false;

    {
        std::lock_guard<std::mutex> _(io_lock);
        auto entry = io_try_get();
        if (entry.has_value()) {
            auto pipeline = entry->pipeline;
            auto task = entry->task;
            if (pipeline->get_current_stage()->submit(task)) {
                pipeline->finish_stage();
                if (!pipeline->is_finished()) {
                    enqueue(pipeline);
                }
            }
            return true;
        }
    }

    std::pair<std::shared_ptr<Pipeline>, std::shared_ptr<Task>> res;
    {
        std::lock_guard<std::mutex> _(lock);
        GGML_ASSERT(main_tid != -1);
        while (true) {
#ifdef LLAMA_USE_CHCORE_API
            res = get_task(alloc, (void *)(long)get_cma_index());
#else
            if (gettid() == main_tid) {
                res = get_task(alloc, (void *)(long)get_cma_index());
            }
#endif
            if (res.first) break;
            res = get_task(io, NULL);
            if (res.first) {
                is_io = true;
                break;
            }
            
            res = get_task(decrypt, NULL);
            if (res.first) break;
            return false;
        }
    }

    auto pipeline = res.first;
    auto task = res.second;
    GGML_ASSERT(pipeline && task);

    if (is_io) io_lock.lock();
    task->step();
    if (is_io) io_lock.unlock();
    if (pipeline->get_current_stage()->submit(task)) {
        pipeline->finish_stage();
        if (!pipeline->is_finished()) {
            enqueue(pipeline);
        }
    }

    return !is_io;
}
}

void LayerScheduler::enqueue(std::shared_ptr<Pipeline> pipeline)
{
    auto current_stage = pipeline->get_current_stage();
    GGML_ASSERT(current_stage);

    std::lock_guard<std::mutex> _(lock);
    if (std::dynamic_pointer_cast<AllocStage>(current_stage)) {
        alloc.push(pipeline);
    } else if (std::dynamic_pointer_cast<IOStage>(current_stage)) {
        io.push(pipeline);
    } else if (std::dynamic_pointer_cast<DecryptStage>(current_stage)) {
        decrypt.push(pipeline);
    } else {
        GGML_ASSERT(false);
    }
}