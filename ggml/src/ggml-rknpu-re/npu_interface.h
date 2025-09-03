#ifndef NPU_INTERFACE_H
#define NPU_INTERFACE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void* mem_allocate(size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint64_t *handle);
void mem_destroy(void *addr, size_t len, uint64_t handle, uint64_t obj_addr);

int npu_reset(void);
int npu_submit(__u64 task_obj_addr, __u32 core_mask);

#ifdef __cplusplus
}
#endif
#endif // NPU_INTERFACE_H
