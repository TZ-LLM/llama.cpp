#include "ggml.h"
#include "pipeline.h"

void Pipeline::rollback(void)
{
    decrypt->rollback();
    io->rollback();
    alloc->rollback();
    current_stage = alloc;
}

std::shared_ptr<Stage> Pipeline::get_current_stage(void)
{
    return current_stage;
}

void Pipeline::finish_stage(void)
{
    GGML_ASSERT(current_stage);
    if (std::dynamic_pointer_cast<AllocStage>(current_stage)) {
        io->start(alloc->get_msg());
        current_stage = io;
    } else if (std::dynamic_pointer_cast<IOStage>(current_stage)) {
        decrypt->start(io->get_msg());
        current_stage = decrypt;
    } else if (std::dynamic_pointer_cast<DecryptStage>(current_stage)) {
        final_msg = decrypt->get_msg();
        current_stage = nullptr;
    } else {
        GGML_ASSERT(false);
    }
}

bool Pipeline::is_finished(void)
{
    return !current_stage;
}

void *Pipeline::get_sched_info(void)
{
    return sched_info;
}

void *Pipeline::get_final_msg(void)
{
    return final_msg;
}

void Pipeline::set_self(void)
{
    io->set_pipeline(shared_from_this());
}
