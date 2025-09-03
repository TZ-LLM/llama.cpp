#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <vector>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

#include <ggml-rknpu2.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"
#define GGML_RKNPU2_INPUT_SCALE 9.0f

#define NDEBUG
#define MAT_COPY

#define THREAD_NR 3
#define NON_NPU_THREAD 0xdeadbeef

thread_local int ttid = NON_NPU_THREAD;

std::mutex B_mtx;
std::unordered_map<rknn_tensor_mem *, rknn_context> Bs;

static inline void add_B(rknn_tensor_mem *B, rknn_context ctx) {
    std::lock_guard<std::mutex> lock(B_mtx);
    Bs.emplace(B, ctx);
}
static inline void del_B(rknn_tensor_mem *B) {
    std::lock_guard<std::mutex> lock(B_mtx);
    auto B_iter = Bs.find(B);
    GGML_ASSERT(B_iter != Bs.end());
    Bs.erase(B_iter);
}
static inline void clr_B(void) {
    std::lock_guard<std::mutex> lock(B_mtx);
    for (auto &B: Bs) {
        rknn_destroy_mem(B.second, B.first);
    }
    Bs.clear();
}

struct ggml_backend_rknpu2_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
#ifndef GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
    int device;
    struct ggml_backend *         backend;
    char                          name[GGML_MAX_NAME];
};
struct ggml_rknpu2_data_pack
{
    rknn_tensor_type type;
    // save data used for mat mul
    void* ordered_data;
    int initialized;

    // RKNPU2 API structs
    rknn_tensor_mem* B;
};
// // host buffer type

// GGML_CALL static const char * ggml_backend_rknpu2_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
//     return GGML_RKNPU2_NAME "_Host";

//     GGML_UNUSED(buft);
// }
/**
 * @brief Free resources associated with a CANN host buffer.
 *
 * This function frees the resources associated with a CANN host buffer, including
 * its context.
 *
 * @param buffer The CANN host buffer to free.
 */
// GGML_CALL static void ggml_backend_rknpu2_host_buffer_free(ggml_backend_buffer_t buffer) {
//     // ACL_CHECK(aclrtFreeHost(buffer->context));
//     printf("host buffer free\n");
// }
/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
static void * ggml_rknpu2_host_malloc(size_t size) {
    // if (getenv("GGML_RKNPU2_NO_PINNED") != nullptr) {
    //     return nullptr;
    // }

/*    void * hostPtr = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.B.size);
    aclError err = aclrtMallocHost((void **) &hostPtr, size);
    if (err != ACL_SUCCESS) {

        GGML_CANN_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, aclGetRecentErrMsg());
        return nullptr;
    }
    return hostPtr;*/
    // It seems that posix_memalign is mandatory
    void * data = nullptr;
    int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), size);
    if (result != 0) {
        printf("%s: error: posix_memalign failed\n", __func__);
        return nullptr;
    }

    return data;
}
// /**
//  * @brief Retrieves the name associated with a CANN host buffer.
//  *
//  * This function returns the descriptive name associated with the specified
//  * CANN host buffer context.
//  *
//  * @param buft Pointer to the host buffer context.
//  * @return Const pointer to the C-style string containing the name.
//  */
// GGML_CALL static const char * ggml_backend_rknpu2_host_buffer_name(ggml_backend_buffer_t buffer) {
//     return "RKNPU2_Host";

//     GGML_UNUSED(buffer);
// }
// /**
//  * @brief Allocates a new CANN host buffer of the specified type and size.
//  *
//  * @param buft Pointer to the host buffer type context.
//  * @param size Size in bytes of the host buffer to allocate.
//  * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
//  */
// GGML_CALL static ggml_backend_buffer_t ggml_backend_rknpu2_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
//     void * hostPtr = ggml_rknpu2_host_malloc(size);

//     if (hostPtr == nullptr) {
//         // fallback to cpu buffer
//         return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
//     }

//     ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(hostPtr, size);
//     buffer->buft = buft;
//     buffer->iface.get_name = ggml_backend_rknpu2_host_buffer_name;
//     buffer->iface.free_buffer = ggml_backend_rknpu2_host_buffer_free;

//     return buffer;
// }

static const char * ggml_backend_rknpu2_buffer_get_name(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return "RKNPU2";
}

struct ggml_backend_rknpu2_buffer_context {
    ggml_backend_rknpu2_buffer_context(size_t device)
            : device(device)
            , name(std::string("RKNPU2") + (std::to_string(device))) {}

    ~ggml_backend_rknpu2_buffer_context() {
        if (buffer) {
            free(buffer);
        }

        /*for (auto * sub_buffer : sub_buffers) {
            free(sub_buffer);
        }

        for (auto * qnn_tensor : qnn_tensors) {
            free_qnn_tensor(*qnn_tensor);
            free(qnn_tensor);
        }*/

        // sub_buffers.clear();
        // qnn_tensors.clear();
    }
    void * buffer = nullptr;

    struct ggml_backend_rknpu2_context * backend_ctx = nullptr;

    size_t                      buffer_size = 0;
    // std::vector<void *>         sub_buffers;
    // std::vector<Qnn_Tensor_t *> qnn_tensors;
    size_t                      device;
    std::string                 name;
};
GGML_CALL static void ggml_backend_rknpu2_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    printf("zzh: %s called\n", __func__);
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;

    if (ctx->buffer) {
        free(ctx->buffer);
        ctx->buffer = NULL;
    }
    clr_B();
    // delete ctx;
}

GGML_CALL static void * ggml_backend_rknpu2_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;

    return ctx->buffer;
}



GGML_CALL static void ggml_backend_rknpu2_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                        ggml_tensor * tensor) {
    // Dont't know what should do.
    // Qnn_ErrorHandle_t               error = QNN_SUCCESS;
    /*ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;

    static int idx                        = 0;
    char       tensor_name[GGML_MAX_NAME] = {0};
    snprintf(tensor_name, GGML_MAX_NAME, "tensor_%04d", idx++);

    uint32_t dimensions[] = {(uint32_t) tensor->ne[0], (uint32_t) tensor->ne[1],
                             (uint32_t) tensor->ne[2],
                             (uint32_t) tensor->ne[3]};
    Qnn_DataType_t qnn_data_type =
        qnn_datatype_from_ggml_datatype(tensor->type);
    Qnn_TensorType_t qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;

    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
    } else if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
        qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
    }
    Qnn_Tensor_t qnn_tensor = QNN_TENSOR_INIT;

    Qnn_TensorMemType_t qnn_mem_type = QNN_TENSORMEMTYPE_RAW;
    if (ctx->device == QNN_BACKEND_GPU) {
        qnn_mem_type = QNN_TENSORMEMTYPE_MEMHANDLE;
    }

    qnn_tensor = {
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {.id         = 0,
                    .name       = tensor_name,
                    .type       = qnn_tensor_type,
                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    .dataType   = qnn_data_type,
                    .quantizeParams =
                            {QNN_DEFINITION_UNDEFINED,
                             QNN_QUANTIZATION_ENCODING_UNDEFINED,
                             {.scaleOffsetEncoding = {.scale  = 0.0000000000000000f,
                                     .offset = 0}}},
                    .rank       = qnn_get_ggml_tensor_rank(tensor),
                    .dimensions = dimensions,
                    .memType    = qnn_mem_type,
                    {.clientBuf = {.data = nullptr, .dataSize = 0}}}}};

    Qnn_Tensor_t * p_qnn_tensor =
        (Qnn_Tensor_t *)calloc(1, sizeof(Qnn_Tensor_t));
    if (nullptr == p_qnn_tensor) {
        QNN_LOG_WARN("calloc failed");
        return;
    }
    error = deep_copy_qnn_tensors(qnn_tensor, *p_qnn_tensor);
    if (error != QNN_SUCCESS) {
        free(p_qnn_tensor);
        QNN_LOG_WARN("init tensor failed");
        return;
    }
    tensor->extra = p_qnn_tensor;
    ctx->qnn_tensors.push_back(p_qnn_tensor);*/
}

// in fact, this is never called when not using mmap
GGML_CALL static void ggml_backend_rknpu2_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                       ggml_tensor * tensor, const void * data,
                                       size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    // TODO: how to handle offset and size?

    if (/*rknpu2_backend &&*/ ggml_rknpu2_can_mul_mat_b(tensor) == false) { // we don't test this!
        printf("ggml_rknpu2_can_mul_mat_b NOT OK\n");
        memcpy((char *) tensor->data + offset, data, size); // We must have this, otherwise will give meaningful output
        return;
    }
    // printf("ggml_backend_rknpu2_buffer_set_tensor, offset=%lu\n", offset);
    GGML_ASSERT(offset == 0);

    
    if (ggml_rknpu2_transform_tensor(data, tensor, offset, size)) {
        printf("ggml_rknpu2_transform_tensor failed\n");
    }
    memcpy((char *) tensor->data + offset, data, size);
}

GGML_CALL static void ggml_backend_rknpu2_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                       const ggml_tensor * tensor, void * data,
                                       size_t offset, size_t size) {
    GGML_UNUSED(buffer);
#ifndef NDEBUG
    printf("ggml_backend_rknpu2_buffer_get_tensor\n");
#endif
    GGML_ASSERT(offset == 0);
    memcpy(data, (char *) tensor->data + offset, size);
    return;
    abort();
    // memcpy(data, (const char *) tensor->data + offset, size);
    
        // We need to transform RKNPU2 tensor back
    ggml_rknpu2_transform_tensor_back(data, tensor, offset, size);
}

GGML_CALL static bool ggml_backend_rknpu2_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                       const struct ggml_tensor * src,
                                       struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    printf("ggml_backend_rknpu2_buffer_cpy_tensor\n");
    abort();
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

GGML_CALL static void ggml_backend_rknpu2_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;

    printf("zzh: %s called\n", __func__);
    memset(ctx->buffer, value, ctx->buffer_size);
}
/**
 * @brief Retrieves the name associated with a CANN buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * CANN buffer type context.
 *
 * @param buft Pointer to the buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
GGML_CALL static const char* ggml_backend_rknpu2_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    return "RKNPU2";

    GGML_UNUSED(buft);
}
// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
struct ggml_backend_rknpu2_buffer_type_context {
    int32_t
        device; /**< Device identifier associated with the buffer context. */
    std::string name; /**< Name associated with the buffer context. */
};
static ggml_backend_buffer_i ggml_backend_rknpu2_buffer_interface = {
    /* .get_name        = */ ggml_backend_rknpu2_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_rknpu2_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rknpu2_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rknpu2_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_rknpu2_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rknpu2_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rknpu2_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rknpu2_buffer_clear,
    /* .reset           = */ nullptr,
};

static struct ggml_backend_rknpu2_context g_rknpu2_mgr[GGML_RKNPU2_MAX_DEVICES];/* = {
    [0] = {.device               = 0,
                         .threads              = 1,
                         .name                 = "rknpu2",
                         .instance             = nullptr,
                         .backend              = nullptr,
                         .raw_interface        = {},
                         .raw_system_interface = {},
                         .socinfo              = {}},
};*/
/**
 * @brief Allocates a new CANN buffer of the specified type and size.
 *
 * This function allocates a new CANN buffer on the specified device with the
 * given size.
 *
 * @param buft Pointer to the buffer type context.
 * @param size Size in bytes of the buffer to allocate.
 * @return Pointer to the allocated buffer, or nullptr if allocation fails.
 */
GGML_CALL static ggml_backend_buffer_t
ggml_backend_rknpu2_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    ggml_backend_rknpu2_buffer_type_context* buft_ctx =
        (ggml_backend_rknpu2_buffer_type_context*)buft->context;


    size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    ggml_backend_rknpu2_buffer_context* ctx =
        new ggml_backend_rknpu2_buffer_context(buft_ctx->device);
    ctx->buffer = ggml_rknpu2_host_malloc(size);
    ctx->buffer_size = size;
    ctx->backend_ctx = &g_rknpu2_mgr[buft_ctx->device];
    if (nullptr == ctx->buffer) {
        printf("%s: failed to allocate %.2f MiB\n", __func__, (double)size / (1 << 20));
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_rknpu2_buffer_interface,
                                    ctx, size);
}
GGML_CALL static bool ggml_backend_rknpu2_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}
GGML_CALL static size_t ggml_backend_rknpu2_buffer_type_get_alignment(
    ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 32;
}
// TODO: this value is an experimental value, works fine with whisper/llm/minicpm-v inference on Android
GGML_CALL static size_t ggml_backend_rknpu2_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);

    return (4095UL * 1024 * 1024); // original QNN: 96 * 1024 * 1024
}


/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
static ggml_backend_buffer_type_i ggml_backend_rknpu2_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rknpu2_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rknpu2_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rknpu2_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rknpu2_buffer_type_get_max_size,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ nullptr,
    /* .is_host          = */ ggml_backend_rknpu2_buffer_is_host,
};
/**
 * @brief Retrieves the CANN buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
GGML_CALL ggml_backend_buffer_type_t
ggml_backend_rknpu2_buffer_type(int32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_rknpu2_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type
        ggml_backend_rknpu2_buffer_types[GGML_RKNPU2_MAX_DEVICES];

    static bool ggml_backend_rknpu2_buffer_type_initialized = false;

    if (!ggml_backend_rknpu2_buffer_type_initialized) {
        for (int32_t i = 0; i < GGML_RKNPU2_MAX_DEVICES; i++) {
            // auto & context = ggml_backend_rknpu2_buffer_type_contexts[i];
            // context = { /*i,*/ std::string(RKNPU2_BACKEND_NAME) + std::to_string(i); }
            ggml_backend_rknpu2_buffer_types[i] = {
                /* .iface    = */ ggml_backend_rknpu2_buffer_type_interface,
                /* .context  = */
                 new ggml_backend_rknpu2_buffer_type_context{
                    i, "RKNPU2" + std::to_string(i)},
            };
        }
        ggml_backend_rknpu2_buffer_type_initialized = true;
    }

    return &ggml_backend_rknpu2_buffer_types[device];
}

// g++ doesn't allow restrict? removed.
static void ggml_rknpu2_transposed_to_native_fp16(__fp16 * dst,
                                                  const float * src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 16 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/16, K/32, 16, 32)
  const size_t rknpu_strides[4] = {k / 32 * 16 * 32, 16 * 32, 32, 1};

  // Block copy 32x16 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 16; i++) {
      for (size_t ii = 0; ii < 16; ii++) {
        size_t partial_src_idx = j * 32 + (i * 16 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

// native fp16 => float (32-bit)
static void ggml_rknpu2_transposed_from_native_fp16(float * dst,
                                                  const __fp16 * src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 16 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/16, K/32, 16, 32)
  const size_t rknpu_strides[4] = {k / 32 * 16 * 32, 16 * 32, 32, 1};

  // Block copy 32x16 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 16; i++) {
      for (size_t ii = 0; ii < 16; ii++) {
        size_t partial_src_idx = j * 32 + (i * 16 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          dst[src_idx] = src[dst_idx];
        }
      }
    }
  }
}

static void ggml_rknpu2_transposed_to_native_int8(int8_t * dst,
                                                  const float * src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 32 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/32, K/32, 32, 32)
  const size_t rknpu_strides[4] = {k / 32 * 32 * 32, 32 * 32, 32, 1};

  // Block copy 32x32 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 32; i++) {
      for (size_t ii = 0; ii < 32; ii++) {
        size_t partial_src_idx = j * 32 + (i * 32 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          /*if (src[dst_idx] > 1.0f || src[dst_idx] < -1.0f) {
            printf("error: src[dst_idx]=%f exceeds 1.0f\n", src[dst_idx]);
          }*/
          dst[dst_idx] = roundf(fminf(fmaxf(src[src_idx], -1.0f), 1.0f) * 127.0f);
            // dst[dst_idx] = roundf(fminf(fmaxf(src[dst_idx], -1.0f), 1.0f) * 127.0f);
        }
      }
    }
  }
}

// RKNN native int8 -> float32
static void ggml_rknpu2_transposed_from_native_int8(float * dst,
                                                  const int8_t * src,
                                                  size_t k, size_t n) {
  GGML_ASSERT(k % 32 == 0 && n % 32 == 0 && k > 0 && n > 0);

  // RKNN native layout is (N/32, K/32, 32, 32)
  const size_t rknpu_strides[4] = {k / 32 * 32 * 32, 32 * 32, 32, 1};

  // Block copy 32x32 at a time to improve cache locality
  for (size_t j = 0; j < k / 32; j++) {
    for (size_t i = 0; i < n / 32; i++) {
      for (size_t ii = 0; ii < 32; ii++) {
        size_t partial_src_idx = j * 32 + (i * 32 + ii) * k;
        size_t partial_dst_idx =
            i * rknpu_strides[0] + j * rknpu_strides[1] + ii * rknpu_strides[2];

        for (size_t jj = 0; jj < 32; jj++) {
          size_t src_idx = partial_src_idx + jj;
          size_t dst_idx = partial_dst_idx + jj;
          /*if (src[dst_idx] > 1.0f || src[dst_idx] < -1.0f) {
            printf("error: src[dst_idx]=%f exceeds 1.0f\n", src[dst_idx]);
          }*/
          dst[src_idx] = float(src[dst_idx]) / 127.0f;
            // dst[dst_idx] = roundf(fminf(fmaxf(src[dst_idx], -1.0f), 1.0f) * 127.0f);
        }
      }
    }
  }
}

// memcpy(data, (const char *) tensor->data + offset, size)
int ggml_rknpu2_transform_tensor(const void * data, struct ggml_tensor * tensor, size_t offset, size_t size)
{
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne3 = tensor->ne[3];
    const int64_t nb0 = tensor->nb[0];
    const int64_t nb1 = tensor->nb[1];

    // this is the original type of tensor
    const enum ggml_type type = tensor->type;
    rknn_tensor_type inference_type;
    // switch (type) {
    //     case GGML_TYPE_Q8_0:
    //         inference_type = RKNN_TENSOR_INT8;
    //         break;
    //     case GGML_TYPE_F16:
    //         inference_type = RKNN_TENSOR_FLOAT16;
    //         break;
    //     default:
    //         printf("ERROR: unsupported tensor transform\n");
    //         abort();
    //         return 1;
    // }
    inference_type = RKNN_TENSOR_FLOAT16;

    if (offset) {
        printf("Error: offset not zero!\n");
        abort();
        return -2;
    }
    GGML_ASSERT(ne2 == 1 && ne3 == 1 && ne1 > 0 && ne0 > 0);
    // GGML_ASSERT(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F16);
    // if (type != GGML_TYPE_F16) {
    //     GGML_ASSERT(ggml_is_quantized(type));
    // }

    tensor->extra = malloc(sizeof(struct ggml_rknpu2_data_pack *) * THREAD_NR);
    struct ggml_rknpu2_data_pack **packs = (struct ggml_rknpu2_data_pack **)tensor->extra;
    for (int i = 0; i < THREAD_NR; i++) {
        struct ggml_rknpu2_data_pack* pack = (struct ggml_rknpu2_data_pack*)malloc(sizeof(struct ggml_rknpu2_data_pack));
        memset(pack, 0, sizeof(struct ggml_rknpu2_data_pack));

        pack->ordered_data = nullptr;
        pack->initialized = 0;
        pack->type = inference_type;
        packs[i] = pack; 
    }


    return 0;
}

// convert tensor back (tensor->extra->ordered_data -> float32 -> tensor-> type(F16/Q8)) (actually not called)
void ggml_rknpu2_transform_tensor_back(void * data, const struct ggml_tensor * tensor, size_t offset, size_t size)
{
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne3 = tensor->ne[3];
    const int64_t nb0 = tensor->nb[0];
    const int64_t nb1 = tensor->nb[1];

    const enum ggml_type type = tensor->type;
    printf("ggml_rknpu2_transform_tensor_back\n");
    abort();

    GGML_ASSERT(ne2 == 1 && ne3 == 1 && ne1 > 0 && ne0 > 0);
    GGML_ASSERT(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F16);
    // type is transformed target type
    GGML_ASSERT(ggml_is_quantized(type));

    rknn_tensor_type inference_type;
    switch (type) {
        case GGML_TYPE_Q8_0:
            inference_type = RKNN_TENSOR_INT8;
            break;
        case GGML_TYPE_F16:
            inference_type = RKNN_TENSOR_FLOAT16;
            break;
        default:
            printf("ERROR: unsupported tensor transform\n");
            abort();
            return;
    }
    for (int i = 0; i < THREAD_NR; i++) {
        GGML_ASSERT(ttid == NON_NPU_THREAD);
        struct ggml_rknpu2_data_pack* pack = ((struct ggml_rknpu2_data_pack**)tensor->extra)[i];
        if (pack->type != inference_type) {
            printf("ERROR: inference type mismatch!\n");
            abort();
        }

        ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
        GGML_ASSERT(traits.from_float != NULL);

        float* reordered_data = NULL;
        const size_t nelements = ne0 * ne1;
        reordered_data = (float*)malloc(nelements * sizeof(float));
        if(inference_type == RKNN_TENSOR_FLOAT16) {
            ggml_rknpu2_transposed_from_native_fp16((float*)reordered_data, (const __fp16*)pack->ordered_data, ne1, ne0);

        }
        else if(inference_type == RKNN_TENSOR_INT8) {
            ggml_rknpu2_transposed_from_native_int8((float*)reordered_data, (const int8_t*)pack->ordered_data, ne1, ne0);
            for (int i = 0; i < ne1 * ne0; ++i) {
                // printf("reordered=%d, fdata=%f\n", ((int8_t*)reordered_data)[i] ,fdata[i]);
            }
        }
        else {
            // free(fdata);
            GGML_ASSERT(/*0 && */"Unsupported inference type");
            abort();
        }

        // float* qdata = (float *)malloc(nelements * sizeof(float));
        traits.from_float(reordered_data, data, nelements);



        // GGML_ASSERT(reordered_data != NULL);
        free(reordered_data);
        // struct ggml_rknpu2_data_pack* pack = (struct ggml_rknpu2_data_pack*)malloc(sizeof(struct ggml_rknpu2_data_pack));
        // memset(pack, 0, sizeof(struct ggml_rknpu2_data_pack));

        // pack->ordered_data = reordered_data;
        // pack->initialized = 0;
        // pack->type = inference_type;

        // tensor->extra = pack;
    }
}

int ggml_rknpu2_can_mul_mat_b(const struct ggml_tensor * tensor)
{

    return 1;
    const int64_t k = tensor->ne[0];
    const int64_t n = tensor->ne[1];
    if(k > 8192 || n > 4096) // RKNPU2 limit
    {
        printf("%s: exceed limit\n", __func__);
        return 0;
    }

    // k and n size must align to 32 bytes
    if(k % 32 != 0 || n % 32 != 0) {
        printf("%s: not align\n", __func__);
        return 0;
    }

    // make sure the tensor has assosiated data
    // zzh: deprecated.
    if(strcmp(tensor->buffer->buft->iface.get_name(tensor->buffer->buft), "RKNPU2")) {
    // if(tensor->backend != GGML_BACKEND_TYPE_GPU)
        printf("iface not RKNPU2\n");
        return 0;
    }

    if(tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_F16)
    {
        printf("zzh: tensor->type != GGML_TYPE_Q8_0!\n");
        return 0;
    }

    return 1;
}

struct ggml_rknpu2_matmul_kernel
{
    int n, k;
    rknn_matmul_info matmul_info;
    rknn_matmul_ctx matmul_ctx;
    rknn_matmul_io_attr matmul_io_attr;

    rknn_tensor_mem* A;
    rknn_tensor_mem* C;

    rknn_core_mask core_mask;
};

#define GGML_RKNPU2_USE_OUTSIDE_ALLOC 1

#if GGML_RKNPU2_USE_OUTSIDE_ALLOC
struct dma_heap_allocation_data {
	uint64_t len;
	uint32_t fd;
	uint32_t fd_flags;
	uint64_t heap_flags;
};

#define DMA_HEAP_IOC_MAGIC		'H'
#define DMA_HEAP_IOCTL_ALLOC	_IOWR(DMA_HEAP_IOC_MAGIC, 0x0,\
				      struct dma_heap_allocation_data)

#define DMA_BUF_SYNC_READ      (1 << 0)
#define DMA_BUF_SYNC_WRITE     (2 << 0)
#define DMA_BUF_SYNC_RW        (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START     (0 << 2)
#define DMA_BUF_SYNC_END       (1 << 2)
#define DMA_BUF_BASE		'b'
#define DMA_BUF_IOCTL_SYNC	_IOW(DMA_BUF_BASE, 0, uint64_t)
#define CMA_HEAP_SIZE	(1024 * 1024)

//Helper function to manually allocate buffer from dma_heap for RKNPU2
//The internal RKNPU2 API will allocate buffer from DMA32 heap, which is only 4GiB, not enough for large models.
//WARNING: Memory leak will not be released on exit!! But it will be released on next run...?
int dma_alloc(size_t size, int *fd, void **va) {
    int ret;
    int prot;
    void *mmap_va;
    int dma_heap_fd = -1;
    struct dma_heap_allocation_data buf_data;
    const char* path = "/dev/dma_heap/system";

    /* open dma_heap fd */
    dma_heap_fd = open(path, O_RDWR);
    if (dma_heap_fd < 0) {
        printf("open %s fail!\n", path);
        return dma_heap_fd;
    }

    /* alloc buffer */
    memset(&buf_data, 0x0, sizeof(struct dma_heap_allocation_data));

    buf_data.len = size;
    buf_data.fd_flags = O_CLOEXEC | O_RDWR;
    ret = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &buf_data);
    if (ret < 0) {
        printf("RK_DMA_HEAP_ALLOC_BUFFER failed\n");
        return ret;
    }

    /* mmap va */
    if (fcntl(buf_data.fd, F_GETFL) & O_RDWR)
        prot = PROT_READ | PROT_WRITE;
    else
        prot = PROT_READ;

    /* mmap contiguors buffer to user */
    mmap_va = (void *)mmap(NULL, buf_data.len, prot, MAP_SHARED, buf_data.fd, 0);
    if (mmap_va == MAP_FAILED) {
        printf("mmap failed: %s\n", strerror(errno));
        return -errno;
    }

    *va = mmap_va;
    *fd = buf_data.fd;

    close(dma_heap_fd);

    return 0;
}

int dma_sync_device_to_cpu(int fd) {
    uint64_t flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}

int dma_sync_cpu_to_device(int fd) {
    uint64_t flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}
void dma_buf_free(size_t size, int *fd, void *va) {
    int len;

    len =  size;
    munmap(va, len);

    close(*fd);
    *fd = -1;
}

#endif

#define GGML_RKNPU2_MAX_MATMUL_KERNELS 64
thread_local struct ggml_rknpu2_matmul_kernel matmul_kernels[GGML_RKNPU2_MAX_MATMUL_KERNELS];
thread_local int matmul_kernels_count = 0;

static uint64_t rknpu2_allocated_bytes = 0;
rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return RKNN_TENSOR_FLOAT16;
        case RKNN_INT8_MM_INT8_TO_INT32:
            return RKNN_TENSOR_INT8;
        case RKNN_INT4_MM_INT4_TO_INT16:
            return RKNN_TENSOR_INT4;
        default:
            GGML_ASSERT(0);
    }
}

rknn_tensor_type rknpu2_matmul_type_to_rknn_type_output(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return RKNN_TENSOR_FLOAT32;
        case RKNN_INT8_MM_INT8_TO_INT32:
            return RKNN_TENSOR_INT32;
        case RKNN_INT4_MM_INT4_TO_INT16:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_INT8_MM_INT8_TO_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_INT4_MM_INT4_TO_INT16;
        default:
            GGML_ASSERT(0);
    }
}

rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_TENSOR_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_TENSOR_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

const char* rknpu2_tensor_type_to_string(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT32:
            return "FLOAT32";
        case RKNN_TENSOR_FLOAT16:
            return "FLOAT16";
        case RKNN_TENSOR_INT8:
            return "INT8";
        case RKNN_TENSOR_INT16:
            return "INT16";
        case RKNN_TENSOR_INT32:
            return "INT32";
        case RKNN_TENSOR_UINT8:
            return "UINT8";
        case RKNN_TENSOR_UINT16:
            return "UINT16";
        default:
            GGML_ASSERT(0);
    }
}
static struct ggml_rknpu2_matmul_kernel *
ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_tensor_type type) {
  for (int i = 0; i < matmul_kernels_count; i++) {
    struct ggml_rknpu2_matmul_kernel *kernel = &matmul_kernels[i];
    if (kernel->matmul_info.M == m && kernel->k == k && kernel->n == n)
      return kernel;
  }
  return NULL;
}
// first find from buffer, then reuse them
static struct ggml_rknpu2_matmul_kernel* ggml_rknpu2_matmul_kernel_create(int m, int k, int n, rknn_tensor_type type, rknn_core_mask core_mask)
{
    struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type);
    if(kernel != NULL)
        return kernel;

    GGML_ASSERT(matmul_kernels_count < GGML_RKNPU2_MAX_MATMUL_KERNELS);
    kernel = &matmul_kernels[matmul_kernels_count++];
    memset(kernel, 0, sizeof(struct ggml_rknpu2_matmul_kernel));

    // if (n > 4096 && n / 4 <= 4096) {
    //     GGML_ASSERT(n % 4 == 0);
    //     kernel->matmul_info.N = n / 4;
    // } else if (n > 4096 && n / 8 <= 4096) {
    //     GGML_ASSERT(n % 8 == 0);
    //     kernel->matmul_info.N = n / 8;
    // } else if (n > 4096 && n % 4096 % 32 == 0) {
    //     kernel->matmul_info.N = 4096;
    // } else {
    //     GGML_ASSERT(n <= 4096);
    //     kernel->n_split = 1;
    // }

    if (n < 4096) {
        GGML_ASSERT(n % 32 == 0);
        kernel->matmul_info.N = n;
    } else {
        // GGML_ASSERT(n % 4096 % 32 == 0);
        kernel->matmul_info.N = 4096;
    }
    if (k < 8192) {
        GGML_ASSERT(k % 32 == 0);
        kernel->matmul_info.K = k;
    } else {
        kernel->matmul_info.K = 8192;
    }
    // GGML_ASSERT(k % 32 == 0);

    kernel->matmul_info.M = m;
    // kernel->matmul_info.K = k; // int8 type must be aligned with 32byte
    kernel->k = k;
    kernel->n = n;
    // kernel->matmul_info.type = rknpu2_matmul_type_from_rknn_type(type);
    kernel->matmul_info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    kernel->matmul_info.B_layout = 2; // B use transposed layout (weight)
    kernel->matmul_info.AC_layout = 0; // A and C use original layout (intermediate)
    // kernel->matmul_info.B_quant_type = RKNN_QUANT_TYPE_PER_CHANNEL_SYM;

    int ret = rknn_matmul_create(&kernel->matmul_ctx, &kernel->matmul_info, &kernel->matmul_io_attr);
    /*if (kernel->matmul_info.K * kernel->matmul_info.N * 4 != kernel->matmul_io_attr.B.size) {
        printf("Error: %u  mismatches %lu\n", kernel->matmul_info.K * kernel->matmul_info.N * 4, kernel->matmul_io_attr.B.size);
    }*/
    GGML_ASSERT(ret == 0);

    kernel->core_mask = core_mask;
    rknn_matmul_set_core_mask(kernel->matmul_ctx, core_mask);
#ifndef NDEBUG
    printf("Created RKNPU2 matmul kernel: src0(%d, %d) x src1(%d, %d) = dst(%d, %d) %s\n", m, k, k, n, m, n, rknpu2_tensor_type_to_string(type));
#endif

    if (kernel->matmul_info.B_quant_type == RKNN_QUANT_TYPE_PER_CHANNEL_SYM)
    {
      rknn_quant_params params_b;
      memcpy(params_b.name, kernel->matmul_io_attr.B.name, RKNN_MAX_NAME_LEN);
      params_b.scale_len = n;
      params_b.scale = (float *)malloc(params_b.scale_len * sizeof(float));
      for (int i = 0; i < params_b.scale_len; i++)
        params_b.scale[i] = 127.0f / GGML_RKNPU2_INPUT_SCALE;
      params_b.zp_len = n;
      params_b.zp = (int32_t *)malloc(params_b.zp_len * sizeof(int32_t));
      memset(params_b.zp, 0, sizeof(int32_t) * params_b.zp_len);

      rknn_matmul_set_quant_params(kernel->matmul_ctx, &params_b);
      free(params_b.scale);
      free(params_b.zp);
    }


    kernel->A = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.A.size);
    kernel->C = rknn_create_mem(kernel->matmul_ctx, kernel->matmul_io_attr.C.size);


    return kernel;
}

static void ggml_backend_rknpu2_mul_mat_mul_npu(
    struct ggml_rknpu2_data_pack** packs,
    const int64_t m,
    const int64_t k,
    const int64_t base_n,
    const int64_t size_n,
    const int64_t n,
    const enum ggml_type type,
    float *A,
    float *B,
    float *C,
    /*const */struct ggml_tensor * src0,
    rknn_core_mask core_mask
);

struct mul_mat_mul_npu_params {
    struct ggml_rknpu2_data_pack** packs;
    int64_t m;
    int64_t k;
    int64_t base_n;
    int64_t size_n;
    int64_t n;
    enum ggml_type type;
    float *A;
    float *B;
    float *C;
    struct ggml_tensor * src0;
};
mul_mat_mul_npu_params params[THREAD_NR];
std::mutex mtx;
std::condition_variable cv_worker;
std::condition_variable cv_master;

int worker_nr;
int finished_int, wake_int;
std::mutex mtx_int;
std::condition_variable cv_int;

bool done;
bool ready[THREAD_NR];
std::atomic<int> finished_nr;
static std::once_flag once_flag;

void set_to_big_core(void) {
#ifdef GGML_USE_RKNPU2
    GGML_ASSERT(ttid != NON_NPU_THREAD);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(ttid + 4, &cpuset);
    GGML_ASSERT(pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0);
#else
    GGML_ABORT("not using RKNPU2");
#endif
}

void set_to_little_core(void) {
#ifdef GGML_USE_RKNPU2
    GGML_ASSERT(ttid != NON_NPU_THREAD);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(ttid, &cpuset);
    GGML_ASSERT(pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0);
#else
    GGML_ABORT("not using RKNPU2");
#endif
}

static void worker(int tid) {
    struct sched_param param;
    int policy = SCHED_FIFO;
    int priority = 40;
    param.sched_priority = priority;
    GGML_ASSERT(pthread_setschedparam(pthread_self(), policy, &param) == 0);

    ttid = tid;
    set_to_big_core();
    rknn_core_mask core_mask;
    if (tid == 0) core_mask = RKNN_NPU_CORE_0;
    if (tid == 1) core_mask = RKNN_NPU_CORE_1;
    if (tid == 2) core_mask = RKNN_NPU_CORE_2;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_worker.wait(lock, [tid] { return ready[tid] || done; });
        }

        if (done) break;

        auto &p = params[tid];
        if (p.m) {
            ggml_backend_rknpu2_mul_mat_mul_npu(
                p.packs,
                p.m,
                p.k,
                p.base_n,
                p.size_n,
                p.n,
                p.type,
                p.A,
                p.B,
                p.C,
                p.src0,
                core_mask
            );
        } else {

        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            finished_nr++;
            ready[tid] = false;
            if (finished_nr == THREAD_NR) {
                cv_master.notify_one();
            }
        }
    }

    for (int32_t i = 0; i < matmul_kernels_count; ++i) {
        struct ggml_rknpu2_matmul_kernel* kernel = &matmul_kernels[i];
        rknn_destroy_mem(kernel->matmul_ctx, kernel->A);
        // rknn_destroy_mem(kernel->matmul_ctx, kernel->B);
        rknn_destroy_mem(kernel->matmul_ctx, kernel->C);
        rknn_matmul_destroy(kernel->matmul_ctx);
    }
}

std::vector<std::thread> npu_threads;

static void init_npu_thread(int thread_nr) {
    GGML_ASSERT(thread_nr <= 3);
    for (int i = 0; i < thread_nr; i++) {
        npu_threads.emplace_back(worker, i);
    }
}

extern "C" {
    extern void ggml_thread_cpu_relax_out(void);
}

void ggml_backend_rknpu2_mul_mat_mul_npu_dispatcher(
    struct ggml_rknpu2_data_pack** packs,
    const int64_t m,
    const int64_t k,
    const int64_t n,
    const enum ggml_type type,
    float *A,
    float *B,
    float *C,
    /*const */struct ggml_tensor * src0
) {
    std::call_once(once_flag, init_npu_thread, THREAD_NR);
    memset(C, 0, m * n * sizeof(float));
    mul_mat_mul_npu_params p = {
        .packs = packs,
        .m = m,
        .k = k,
        .n = n,
        .type = type,
        .A = A,
        .B = B,
        .C = C,
        .src0 = src0,
    };
    const int64_t nn = n / THREAD_NR / 32 * 32;
    finished_nr = 0;
    for (int i = THREAD_NR - 1; i >= 0; i--) {
    // for (int i = 0; i < THREAD_NR; i++) {
        p.base_n = i * nn;
        if (i + 1 != THREAD_NR) {
            p.size_n = nn;
        } else {
            p.size_n = n - i * nn;
        }
        memcpy(params + i, &p, sizeof(p));
        ready[i] = true;
        // if (p.m) {
        //     ggml_backend_rknpu2_mul_mat_mul_npu(
        //         p.pack,
        //         p.m,
        //         p.k,
        //         p.n,
        //         p.type,
        //         p.A,
        //         p.B,
        //         p.C,
        //         p.src0,
        //         RKNN_NPU_CORE_0
        //     );
        // }
    }
    worker_nr = std::min(m, (int64_t)THREAD_NR);
    cv_worker.notify_all();
    // std::unique_lock<std::mutex> lock(mtx);
    // cv_master.wait(lock, [] { return finished_nr == THREAD_NR; });
    while (finished_nr.load() != THREAD_NR) {
        ggml_thread_cpu_relax_out();
    } 
}

static void ggml_backend_rknpu2_mul_mat(ggml_backend_rknpu2_context * ctx, struct ggml_tensor * dst) {
    /* src0 => matrix B */
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne12 == 1 && ne13 == 1);
    GGML_ASSERT(ne2 == 1 && ne3 == 1);

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

// printf("zzh: ggml_rknpu2_mul_mat\n");
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    struct ggml_rknpu2_data_pack** packs = (struct ggml_rknpu2_data_pack **)src0->extra;
    GGML_ASSERT(packs != NULL);

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    float *B;
    ggml_type_traits_t traits = ggml_internal_get_type_traits(src0->type);
    GGML_ASSERT(traits.to_float != NULL);


    if (type == GGML_TYPE_F16) {
        /* B is already F16, so do nothing */
    } else if (type != GGML_TYPE_F32) {
        /* need to change B to F16 */
        ggml_to_float_t const to_float = traits.to_float;

        B = (float*)malloc(k * n * sizeof(float));

        size_t id = 0;
        for (int64_t i01 = 0; i01 < ne01; ++i01) {
            to_float((const char *)src0->data + i01*nb01, B + id, ne00);
            id += ne00;
        }

        // assert(id*sizeof(float) <= params->wsize);
    } else {
        /* it seems that F32 can be directly converted to F16 */
        B = (float*)src0->data;
    }

    float *A = (float*)src1->data;
    float *C = (float*)dst->data;

    // ggml_backend_rknpu2_mul_mat_mul_npu(pack, m, k, n, type, A, B, C, src0);
    ggml_backend_rknpu2_mul_mat_mul_npu_dispatcher(packs, m, k, n, type, A, B, C, src0);

    if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) {
        free(B);
    }
}

extern "C" {
void ggml_backend_rknpu2_mul_mat_out(struct ggml_tensor * dst) {
    return ggml_backend_rknpu2_mul_mat(NULL, dst);
}
}

std::mutex init_mutex;

static rknn_tensor_mem *alloc_B(
    size_t size,
    rknn_matmul_ctx ctx
) {
    thread_local static std::unordered_map<size_t, std::pair<int, uint8_t *>> Bs;

    auto B_iter = Bs.find(size);
    if (B_iter == Bs.end()) {
        int fd = -1;
        uint8_t *va = NULL;
        if (size == 0) {
            printf("zzh: error: 976: B.size=0!\n");
        }
        int ret = dma_alloc(size, &fd, (void **)&va);
        if (ret < 0) {
            printf("dma_alloc returns %d\n", ret);
            ret = 0;
        }
        ret = dma_sync_device_to_cpu(fd);
        if (ret < 0) {
            printf("dma_sync_device_to_cpu returns %d\n", ret);
            ret = 0;
        }
        if (size == 0) {
            printf("zzh: error: B.size=0!\n");
        }
        ret = dma_sync_cpu_to_device(fd);
        if (ret < 0) {
            printf("dma_sync_cpu_to_device returns %d\n", ret);
            ret = 0;
        }

        Bs.emplace(size, std::make_pair(fd, va));
        return rknn_create_mem_from_fd(ctx, fd, va, size, 0);
    } else {
        return rknn_create_mem_from_fd(ctx, B_iter->second.first, B_iter->second.second, size, 0);
    }
}

void ggml_backend_rknpu2_mul_mat_mul_npu(
    struct ggml_rknpu2_data_pack** packs,
    const int64_t m,
    const int64_t k,
    const int64_t base_n,
    const int64_t size_n,
    const int64_t n,
    const enum ggml_type type,
    float *A,
    float *B,
    float *C,
    /*const */struct ggml_tensor * src0,
    rknn_core_mask core_mask
) {
    struct ggml_rknpu2_data_pack *pack = packs[ttid];

    // First time called. Initialize RKNPU2 API structs
    if(pack->initialized == 0) {
        struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_create(m, k, size_n, RKNN_TENSOR_FLOAT32, core_mask);
        // allocate B
        pack->B = alloc_B(kernel->matmul_io_attr.B.size, kernel->matmul_ctx);
        // should change ordered_data to int8

        // rknn_B_normal_layout_to_native_layout(pack->ordered_data, pack->B->virt_addr, k, n, &kernel->matmul_info); // 1224: disable native layout
            // 这个不是零拷贝的接口
    // int ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->A, &kernel->matmul_io_attr.A);
    // GGML_ASSERT(ret == 0);
    // ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->C, &kernel->matmul_io_attr.C);
    // GGML_ASSERT(ret == 0);
        free(pack->ordered_data);
        pack->ordered_data = NULL;
        pack->initialized = 1;
        if (kernel->matmul_io_attr.B.size == 0) {
            printf("zzh: error: B.size=0!\n");
        }
        rknpu2_allocated_bytes += kernel->matmul_io_attr.B.size;
#ifndef NDEBUG
        printf("RKNPU2 allocated %f MiB\n",
               rknpu2_allocated_bytes / 1024.0F / 1024.0F);
#endif
    }

    struct ggml_rknpu2_matmul_kernel* kernel = ggml_rknpu2_matmul_kernel_find(m, k, size_n, RKNN_TENSOR_FLOAT32);
    // GGML will switch batch size on the fly. So we need to create a new kernel if the batch size is different
    if(kernel == NULL) {
        // GGML_ASSERT(pack->type == 0); // I don't understand
        kernel = ggml_rknpu2_matmul_kernel_create(m, k, size_n, RKNN_TENSOR_FLOAT32, core_mask);
            // 这个不是零拷贝的接口
    // int ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->A, &kernel->matmul_io_attr.A);
    // GGML_ASSERT(ret == 0);
    // ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->C, &kernel->matmul_io_attr.C);
    // GGML_ASSERT(ret == 0);
    }

    GGML_ASSERT(kernel->matmul_io_attr.A.type == pack->type);
    GGML_ASSERT(kernel->matmul_io_attr.C.type == rknpu2_matmul_input_type_to_output_type(pack->type));
    // rknn_tensor_type inference_type = pack->type;

    // __fp16 *data = (__fp16 *)kernel->A->virt_addr;
    // for (int64_t i = 0; i < m; i++) {
    //     for (int64_t j = 0; j < k; j++) {
    //         data[i * k + j] = *(float *)(src1->data + i * src1->nb[1] + j * src1->nb[0]);
    //     }
    // }
    // for (int64_t i = 0; i < m * k; i++) {
    //     data[i] = A[i];
    // }


    // int64_t step_B = k * n / kernel->n_split;
    // int64_t step_C = m * n / kernel->n_split;
    // int64_t step_n = n / kernel->n_split;
    int64_t K = kernel->matmul_info.K;
    int64_t N = kernel->matmul_info.N;
    for (int64_t pk = 0; pk * K < k; pk++) {
        int64_t step_k = (pk + 1) * K < k ? K : k - pk * K;

#ifdef MAT_COPY
        if (step_k < K) {
            memset((void *)pack->B->virt_addr, 0, K * N * sizeof(__fp16));
        }
#endif

        {
            __fp16 *data = (__fp16 *)kernel->A->virt_addr;
            for (int64_t i = 0; i < m; i++) {
                for (int64_t j = 0; j < step_k; j++) {
                    int64_t jj = pk * K + j;
                    data[i * K + j] = A[i * k + jj];
                }
            }
        }

        for (int64_t pn = 0; pn * N < size_n; pn++) {
            int64_t step_n = (pn + 1) * N < size_n ? N : size_n - pn * N;

#ifdef MAT_COPY
            if (src0->type != GGML_TYPE_F16) {
                __fp16 *data = (__fp16 *)pack->B->virt_addr;
                for (int64_t i = 0; i < step_n; i++) {
                    for (int64_t j = 0; j < step_k; j++) {
                        int64_t ii = base_n + pn * N + i;
                        int64_t jj = pk * K + j;
                        data[i * K + j] = B[ii * k + jj];
                    }
                }
            } else {
                __fp16 *data = (__fp16 *)pack->B->virt_addr;
                __fp16 *B = (__fp16 *)src0->data;
                for (int64_t i = 0; i < step_n; i++) {
                    for (int64_t j = 0; j < step_k; j++) {
                        int64_t ii = base_n + pn * N + i;
                        int64_t jj = pk * K + j;
                        data[i * K + j] = B[ii * k + jj];
                    }
                }
            }
#endif

            int ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->A, &kernel->matmul_io_attr.A);
            GGML_ASSERT(ret == 0);
            ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, kernel->C, &kernel->matmul_io_attr.C);
            GGML_ASSERT(ret == 0);
            ret = rknn_matmul_set_io_mem(kernel->matmul_ctx, pack->B, &kernel->matmul_io_attr.B);
            GGML_ASSERT(ret == 0);
            ret = rknn_matmul_run(kernel->matmul_ctx);
            GGML_ASSERT(ret == 0);

            {
                std::unique_lock<std::mutex> lock(mtx_int);
                finished_int++;
                if (finished_int == worker_nr) {
                    wake_int = 0;
                    cv_int.notify_all();
                } else {
                    cv_int.wait(lock, [] { return finished_int == worker_nr; });
                }
                wake_int++;
                if (wake_int == worker_nr) {
                    finished_int = 0;
                    cv_int.notify_all();
                } else {
                    cv_int.wait(lock, [] { return wake_int == worker_nr; });
                }
            }

            {
                float *data = (float*)kernel->C->virt_addr;
                for (int64_t i = 0; i < m; i++) {
                    for (int64_t j = 0; j < step_n; j++) {
                        int64_t jj = base_n + pn * N + j;
                        C[i * n + jj] += data[i * N + j];
                        // *(float *)(dst->data + i * dst->nb[1] + jj * dst->nb[0]) = data[i * step_n + j];
                    }
                }
            }
        }
    }
}

// backend interface

GGML_CALL static const char * ggml_backend_rknpu2_name(ggml_backend_t backend) {
    return "RKNPU2";

    GGML_UNUSED(backend);
}

GGML_CALL static void ggml_backend_rknpu2_free(ggml_backend_t backend) {
    ggml_backend_rknpu2_context * ctx = (ggml_backend_rknpu2_context *)backend->context;
    // delete ctx;
    // delete backend;


    // rknpu2_instance * instance = (rknpu2_instance *)g_rknpu2_mgr[ctx->device].instance;
    // if (instance != nullptr) {
    //     std::map<std::string,
    //              std::tuple<Qnn_GraphHandle_t, Qnn_Tensor_t *, Qnn_Tensor_t *,
    //                         Qnn_Tensor_t *>>::iterator graph_it;
    //     for (graph_it = instance->_qnn_graph_map.begin();
    //          graph_it != instance->_qnn_graph_map.end(); graph_it++) {
    //         auto & graph_item   = graph_it->second;
    //         Qnn_GraphHandle_t & graph_handle = std::get<0>(graph_item);
    //         GGML_UNUSED(graph_handle);
    //         QNN_LOG_INFO("graph type:%s", graph_it->first.c_str());
    //     }
    //     instance->_qnn_graph_map.clear();

    //     instance->qnn_finalize();
    //     delete instance;
    //     g_qnn_mgr[ctx->device].instance = nullptr;
    // }

    if (g_rknpu2_mgr[ctx->device].backend != nullptr) {
        delete backend;
        g_rknpu2_mgr[ctx->device].backend = nullptr;
    }
    for (int32_t i = 0; i < matmul_kernels_count; ++i) {
        struct ggml_rknpu2_matmul_kernel* kernel = &matmul_kernels[i];
        rknn_destroy_mem(kernel->matmul_ctx, kernel->A);
        // rknn_destroy_mem(kernel->matmul_ctx, kernel->B);
        rknn_destroy_mem(kernel->matmul_ctx, kernel->C);
        rknn_matmul_destroy(kernel->matmul_ctx);
    }

    done = true;
    cv_worker.notify_all();
    for (auto &thread: npu_threads) {
        thread.join();
    }
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_rknpu2_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_rknpu2_context * ctx = (ggml_backend_rknpu2_context *) backend->context;
    return ggml_backend_rknpu2_buffer_type(ctx->device);
}

GGML_CALL static enum ggml_status ggml_backend_rknpu2_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_rknpu2_context * ctx = (ggml_backend_rknpu2_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_rknpu2_mul_mat(ctx, node);
                break;

            // case GGML_OP_OUT_PROD:
            //     // ggml_backend_rknpu2_out_prod(ctx, node); // we don't support currently
            //     break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            // case GGML_OP_PERMUTE:
            // case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_rknpu2_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst = op;

    if (op->op != GGML_OP_MUL_MAT) {
        // printf("zzh: op is %d, not mul mat\n", op->op);
        return false;
    }

    // printf("ggml_backend_rknpu2_supports_op, %d, %d, %p\n", src1->type, dst->type, src0->extra);
    // return false; // DEBUG: first, never use this backend

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    if (ggml_is_contiguous(src0) &&
        ggml_is_contiguous(src1) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src0->extra != NULL) {
        const int64_t k = src0->ne[0];
        const int64_t n = src0->ne[1];
        return true;
        // k > 8192 时，B 会被分成 T 段，int T = std::ceil(K / 8192)，推荐使用 rknn_B_normal_layout_to_native_layout 接口直接进行数据转换
        if(k > 8192 || n > 4096) // RKNPU2 limit （原来是10240）
        {
            // printf("oversize: k=%ld, n=%ld\n", k, n);
            return 0;
        }

        // k and n size must align to 32 bytes
        if(k % 32 != 0 || n % 32 != 0)
        {
            printf("not align: k=%ld, n=%ld\n", k, n);
            return 0;
        }

        // make sure the tensor has assosiated data
        // printf("zzh: %s\n", src0->buffer->buft->iface.get_name(src0->buffer->buft));
        if (strcmp(src0->buffer->buft->iface.get_name(src0->buffer->buft), "RKNPU2")) {
            return 0;
        }

        if(src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_F16)
        {
            printf("zzh: tensor->type wrong\n");
            return 0;
        }

        /*printf("RKNPU2: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
        return true;
    }

    // printf("rknpu2 not support this MUL_MAT\n");
    return false;

    GGML_UNUSED(backend);
}

extern "C" {
bool ggml_backend_rknpu2_supports_op_out(const struct ggml_tensor * op) {
    ggml_backend_t backend;
    return ggml_backend_rknpu2_supports_op(backend, op);
}
}

GGML_CALL static bool ggml_backend_rknpu2_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    // zzh: maybe this is wrong! however qnn doesn't have this.
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(backend);
}


GGML_CALL static bool ggml_backend_rknpu2_offload_op(ggml_backend_t backend, const ggml_tensor *tensor) {
    // ggml_backend_rknpu2_context *ctx = (ggml_backend_rknpu2_context *)backend->context;
    // return ggml_rknpu2_compute_forward(ctx, nullptr, (ggml_tensor *)tensor);
    return tensor->op == GGML_OP_MUL_MAT;
}

static struct ggml_backend_i rknpu2_backend_i = {
    /* .get_name                = */ ggml_backend_rknpu2_name,
    /* .free                    = */ ggml_backend_rknpu2_free,
    /* .get_default_buffer_type = */ ggml_backend_rknpu2_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rknpu2_graph_compute,
    /* .supports_op             = */ ggml_backend_rknpu2_supports_op,
    /* .supports_buft           = */ ggml_backend_rknpu2_supports_buft,
    /* .offload_op              = */ ggml_backend_rknpu2_offload_op, // qnn set this
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_rknpu2_guid(void) {
    static ggml_guid guid = { 0xc3, 0x92, 0x9a, 0xca, 0x45, 0x27, 0x40, 0x55, 0x89, 0xf2, 0xdc, 0x83, 0xf1, 0xba, 0x02, 0xc3 };
    return &guid;
}

static ggml_backend_t ggml_backend_rknpu2_reg_init(const char *params, void * user_data) {
    ggml_backend_t rknpu2_backend = ggml_backend_rknpu2_init((int) (unsigned long) user_data);
    return rknpu2_backend;
}

ggml_backend_t ggml_backend_rknpu2_init(int32_t device) {
    // g_rknpu2_mgr[0].interface 
    // ggml_backend_rknpu2_context * ctx = new ggml_backend_rknpu2_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_rknpu2_guid(),
        /* .interface = */ rknpu2_backend_i,
        /* .context   = */ &g_rknpu2_mgr[device],
    };

    g_rknpu2_mgr[device].backend = backend;
    return backend;
}

GGML_CALL bool ggml_backend_is_rknpu2(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rknpu2_guid());
}

GGML_CALL int32_t ggml_backend_rknpu2_get_device_count() {
    return 1;
}

void ggml_backend_rknpu2_set_n_threads(ggml_backend_t backend_rknpu2, int n_threads) {
    GGML_ASSERT(ggml_backend_is_rknpu2(backend_rknpu2));

    ggml_backend_rknpu2_context * ctx = (ggml_backend_rknpu2_context *)backend_rknpu2->context;
    ctx->n_threads = n_threads;
}

extern "C" GGML_CALL int ggml_backend_rknpu2_reg_devices();

GGML_CALL int ggml_backend_rknpu2_reg_devices() {
    // ggml_rknpu2_instance_init();
    uint32_t device_count = ggml_backend_rknpu2_get_device_count();

    for (size_t i = 0; i < device_count; i++) {
        //char name[128];
        //snprintf(name, sizeof(name), "%s%ld", GGML_VK_NAME, i);
        ggml_backend_register(g_rknpu2_mgr[i].name, ggml_backend_rknpu2_reg_init, ggml_backend_rknpu2_buffer_type(i), (void *) (intptr_t) i);  // NOLINT
    }
    return device_count;
}