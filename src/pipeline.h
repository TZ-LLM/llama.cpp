#pragma once

#include <functional>
#include <memory>
#include <map>
#include <queue>
#include <mutex>
#include <atomic>

class Task {
public:
    virtual ~Task() = default;

    virtual void step(void) = 0;
};

class Stage {
public:
    virtual ~Stage() = default;

    virtual void start(void *input) = 0;
    virtual std::pair<std::shared_ptr<Task>, bool> get_task(void *) = 0;
    virtual bool submit(std::shared_ptr<Task> task) = 0;
    virtual void *get_msg(void) = 0;
    virtual void rollback(void) = 0;
};

struct Pipeline;

struct alloc_io_msg {
    void *buf;
    std::vector<std::tuple<int, int, off_t, size_t>> cma_indexes;
    std::vector<std::vector<std::pair<unsigned long, size_t>>> paddr;
};

class AllocStage : public Stage {
private:
    size_t size;
    void *addr;
    alloc_io_msg msg;
    int tzd_fd;

    std::mutex submit_pos_mtx;
    size_t submit_pos;
    int get_nr[10];
    int all_block_nr;
    int block_nr[10];
    std::atomic<int> finished_nr;

public:
    AllocStage(size_t off, size_t len);
    void start(void *input) override;
    std::pair<std::shared_ptr<Task>, bool> get_task(void *) override;
    bool submit(std::shared_ptr<Task> task) override;
    void *get_msg(void) override;
    void rollback(void) override;

};

struct io_decrypt_msg {
    void *buf;
    std::vector<std::pair<unsigned long, unsigned long>> cma_region;
};

class IOStage : public Stage {
private:
    size_t off;
    size_t size;
    std::shared_ptr<Pipeline> pipeline;
    io_decrypt_msg id_msg;

    void *buf;
    std::vector<std::tuple<int, int, off_t, size_t>> cma_indexes;

    std::atomic<int> cnt_to_finish;

public:
    IOStage(size_t off, size_t size): off(off), size(size), pipeline(nullptr) {}
    void start(void *input) override;
    std::pair<std::shared_ptr<Task>, bool> get_task(void *) override;
    bool submit(std::shared_ptr<Task> task) override;
    void *get_msg(void) override;
    void rollback(void) override;
    void set_pipeline(std::shared_ptr<Pipeline> pipeline);

};

class DecryptStage : public Stage {
private:
    void *buf;
    size_t size;
    int block_nr;
    std::atomic<int> finished_nr;
    size_t submit_pos;

public:
    DecryptStage(size_t size);
    void start(void *input) override;
    std::pair<std::shared_ptr<Task>, bool> get_task(void *) override;
    bool submit(std::shared_ptr<Task> task) override;
    void *get_msg(void) override;
    void rollback(void) override;

};

class Pipeline : public std::enable_shared_from_this<Pipeline> {
private:
    std::shared_ptr<AllocStage> alloc;
    std::shared_ptr<IOStage> io;
    std::shared_ptr<DecryptStage> decrypt;
    void *sched_info;
    std::shared_ptr<Stage> current_stage;
    void *final_msg;

public:
    Pipeline(
        std::shared_ptr<AllocStage> alloc,
        std::shared_ptr<IOStage> io,
        std::shared_ptr<DecryptStage> decrypt,
        void *sched_info
    ) : alloc(alloc), io(io), decrypt(decrypt), sched_info(sched_info), current_stage(alloc) {}

    void rollback(void);
    std::shared_ptr<Stage> get_current_stage(void);
    void finish_stage(void);
    bool is_finished(void);
    void *get_sched_info(void);
    void *get_final_msg(void);
    void set_self(void);
};

class Scheduler {
public:
    virtual ~Scheduler() = default;

    virtual bool step(void) = 0;
    virtual void enqueue(std::shared_ptr<Pipeline> pipeline) = 0;
};

class LayerScheduler : public Scheduler {
    struct pipeline_cmp {
        bool operator()(const std::shared_ptr<Pipeline> &left, const std::shared_ptr<Pipeline> &right) {
            auto left_sched_info = (int64_t)left->get_sched_info();
            auto right_sched_info = (int64_t)right->get_sched_info();
            return left_sched_info > right_sched_info;
        }
    };
    typedef std::priority_queue<std::shared_ptr<Pipeline>, std::vector<std::shared_ptr<Pipeline>>, pipeline_cmp> layer_queue_t;

private:
    layer_queue_t alloc, io, decrypt;
    std::mutex lock;

public:
    bool step(void) override;
    void enqueue(std::shared_ptr<Pipeline> pipeline) override;

private:
    std::pair<std::shared_ptr<Pipeline>, std::shared_ptr<Task>> get_task(layer_queue_t &queue, void *arg);
};
