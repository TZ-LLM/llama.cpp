#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#define GGML_RKNPU2_NAME "RKNPURE"
#define GGML_RKNPU2_MAX_DEVICES 1
// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_rknpure_init(int32_t device);

GGML_API GGML_CALL bool ggml_backend_is_rknpu2(ggml_backend_t backend);

// number of threads used for conversion to float
// for openblas and blis, this will also set the number of threads used for blas operations
GGML_API GGML_CALL void ggml_backend_rknpu2_set_n_threads(ggml_backend_t backend_rknpu2, int n_threads);
int ggml_rknpure_can_mul_mat_b(const ggml_tensor * tensor);
int ggml_rknpure_transform_tensor(const void * data, ggml_tensor * tensor, size_t offset, size_t size);
void ggml_rknpu2_transform_tensor_back(void * data, const ggml_tensor * tensor, size_t offset, size_t size);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_rknpu2_host_buffer_type(void);
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_rknpure_buffer_type(int32_t dev_num);
GGML_API GGML_CALL int32_t ggml_backend_rknpu2_get_device_count(void);
