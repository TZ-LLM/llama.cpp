#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <sys/mman.h>
#include <vector>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <map>

#include <chrono>
#include <queue>

static inline int64_t get_micro(void) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
}

struct measure_point {
    const char *file;
    int line;
    int64_t micro;

    measure_point(const char *file, int line, int64_t micro)
        : file(file), line(line), micro(micro) {}
};

#define LINE_NR (3000)
int last_lines[LINE_NR];
std::atomic<int64_t> micros[LINE_NR];

struct measure_stack {
    std::vector<measure_point> stack;
    std::unordered_map<int, int> regions;

    void begin_measure(const char *file, int line, int64_t micro) {
        stack.emplace_back(file, line, micro);
    }
    void end_measure(const char *file, int line, int64_t micro) {
        assert(!stack.empty());
        auto last = stack.back();
        stack.pop_back();
        if (regions.find(line) != regions.end()) {
            assert(last.line == regions[line]);
        } else {
            regions[line] = last.line;
        }
        last_lines[line] = last.line;
        micros[line] += micro - last.micro;
    }
};
thread_local measure_stack mstack;

#define RKNPU_MEASURE 1

#if RKNPU_MEASURE

#define BEGIN_MEASURE mstack.begin_measure(__FILE__, __LINE__, get_micro())
#define END_MEASURE mstack.end_measure(__FILE__, __LINE__, get_micro())

#else

#define BEGIN_MEASURE
#define END_MEASURE

#endif

#define BEGIN_MEASURE_0 if (ith == 0) BEGIN_MEASURE
#define END_MEASURE_0 if (ith == 0) END_MEASURE

#ifdef GGML_USE_CHCORE
extern "C" void npu_dump_measure(bool clear);
#endif

void ggml_rknpu_dump_measure(void) {
#if RKNPU_MEASURE
    printf("********************************begin rknpu measure dump********************************\n");
    for (int i = 0; i < LINE_NR; i++) {
        if (micros[i].load()) {
            printf("%s:%d:(from %d) %ld ms\n", __FILE__, i, last_lines[i], micros[i].load() / 1000);
        }
        last_lines[i] = 0;
        micros[i] = 0;
    }
    printf("*********************************end rknpu measure dump*********************************\n");

#ifdef GGML_USE_CHCORE
    npu_dump_measure(true);
#endif

#endif
}

// #define USE_CPU_CHECK

#include <ggml-rknpu-re.h>
#include <ggml-rknpu-re/rknpu-ioctl.h>
#include <ggml-rknpu-re/npu_interface.h>
#include <ggml-rknpu-re/npu_matmul.h>
#ifdef USE_CPU_CHECK
#include <ggml-rknpu-re/matmul_cpu_check.h>
#endif
#ifdef __MUSL__
#include <sys/syscall.h>
#endif

#define RKNN_TENSOR_INT8 0x9
#define RKNN_TENSOR_FLOAT16 0x16
#define RKNN_TENSOR_FLOAT32 0x33
#define GGML_RKNPU2_INPUT_SCALE 9.0f
typedef int rknn_tensor_type;
typedef int rknn_core_mask;

const float SCALE_MIN = 1e-9;

// #define MAT_COPY

#include <sys/ioctl.h>

#ifndef MAT_COPY
#ifndef GGML_USE_CHCORE
#define FAKE_CACHE
#endif
#endif
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

#ifdef GGML_USE_CHCORE
#define NPU_CORE_NUM (3)
#define THREAD_NR (1)
#else
#define THREAD_NR (3)
#define NPU_CORE_NUM THREAD_NR
#endif
#define NON_NPU_THREAD 0xdeadbeef

thread_local int ttid = NON_NPU_THREAD;


struct ggml_backend_rknpure_context {
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
    int type;
    // save data used for mat mul
    void* ordered_data;
    int initialized;
};
struct rknpure_weight {
    void *weights; // B
    uint64_t weights_dma, weights_obj;
    uint64_t weights_handle;
    uint64_t buffer_size;
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
    return "RKNPURE";
}

struct ggml_backend_rknpu2_buffer_context {
    ggml_backend_rknpu2_buffer_context(size_t device)
            : device(device)
            , name(std::string("RKNPURE") + (std::to_string(device))) {}

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

    struct ggml_backend_rknpure_context * backend_ctx = nullptr;

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
    // clr_B();

    // delete ctx;
}

GGML_CALL static void * ggml_backend_rknpu2_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu2_buffer_context * ctx = (ggml_backend_rknpu2_buffer_context *) buffer->context;

    return ctx->buffer;
}



GGML_CALL static void ggml_backend_rknpu2_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                        ggml_tensor * tensor) {
    // Dont't know what should do.
}

// in fact, this is never called when not using mmap
GGML_CALL static void ggml_backend_rknpu2_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                       ggml_tensor * tensor, const void * data,
                                       size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    // TODO: how to handle offset and size?

    if (/*rknpu2_backend &&*/ ggml_rknpure_can_mul_mat_b(tensor) == false) { // we don't test this!
        printf("ggml_rknpure_can_mul_mat_b NOT OK\n");
        memcpy((char *) tensor->data + offset, data, size); // We must have this, otherwise will give meaningful output
        return;
    }
    // printf("ggml_backend_rknpu2_buffer_set_tensor, offset=%lu\n", offset);
    GGML_ASSERT(offset == 0);

    
    if (ggml_rknpure_transform_tensor(data, tensor, offset, size)) {
        printf("ggml_rknpure_transform_tensor failed\n");
    }
    memcpy((char *) tensor->data + offset, data, size);
}

GGML_CALL static void ggml_backend_rknpu2_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                       const ggml_tensor * tensor, void * data,
                                       size_t offset, size_t size) {
    GGML_UNUSED(buffer);
#ifndef NDEBUG
    // printf("ggml_backend_rknpu2_buffer_get_tensor\n");
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
    return "RKNPURE";

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

static struct ggml_backend_rknpure_context g_rknpu2_mgr[GGML_RKNPU2_MAX_DEVICES];

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
ggml_backend_rknpure_buffer_type(int32_t device) {
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
                    i, "RKNPURE" + std::to_string(i)},
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
int ggml_rknpure_transform_tensor(const void * data, struct ggml_tensor * tensor, size_t offset, size_t size)
{
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne3 = tensor->ne[3];
    const int64_t nb0 = tensor->nb[0];
    const int64_t nb1 = tensor->nb[1];

    // this is the original type of tensor
    const enum ggml_type type = tensor->type;
    int inference_type;
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

    int inference_type;
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
}

int ggml_rknpure_can_mul_mat_b(const struct ggml_tensor * tensor)
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
    if(strcmp(tensor->buffer->buft->iface.get_name(tensor->buffer->buft), "RKNPURE")) {
    // if(tensor->backend != GGML_BACKEND_TYPE_GPU)
        printf("iface not RKNPURE\n");
        return 0;
    }

    if(tensor->type != GGML_TYPE_Q8_0 && tensor->type != GGML_TYPE_F16)
    {
        printf("zzh: tensor->type != GGML_TYPE_Q8_0!\n");
        return 0;
    }

    return 1;
}

#ifdef GGML_USE_CHCORE
extern "C" {
    int usys_set_prio(int thread_cap, int prio);
    int usys_cache_flush(unsigned long start, unsigned long size, int op_type);
    void usys_disable_local_irq(void);
    void usys_enable_local_irq(void);
    void usys_yield(void);
}

/* cache operations */
#define CACHE_CLEAN         1
#define CACHE_INVALIDATE    2
#define CACHE_CLEAN_AND_INV 3
#define SYNC_IDCACHE        4
#endif

inline size_t rknn_type_size_A(rknn_tensor_type type) {
    if (type == RKNN_TENSOR_FLOAT32) return sizeof(__fp16);
    if (type == RKNN_TENSOR_INT8) return sizeof(int8_t);
    GGML_ASSERT(false);
}
inline size_t rknn_type_size_B(rknn_tensor_type type) {
    if (type == RKNN_TENSOR_FLOAT32) return sizeof(__fp16);
    if (type == RKNN_TENSOR_INT8) return sizeof(int8_t);
    GGML_ASSERT(false);
}
inline size_t rknn_type_size_C(rknn_tensor_type type) {
    if (type == RKNN_TENSOR_FLOAT32) return sizeof(float);
    if (type == RKNN_TENSOR_INT8) return sizeof(int32_t);
    GGML_ASSERT(false);
}

struct rknn_mem {
    size_t size;
    void *ptr;
    int fd;
    void *dma_ptr;
    uint64_t dma, obj;
    uint64_t handle;
    float scale;
    pthread_spinlock_t scale_lock;

    std::atomic<int> pre_scale_cnt;
    std::atomic<int> pre1_cnt;
    std::atomic<int> post_cnt;

    rknn_mem(size_t size): size(size) {
        dma_ptr = mem_allocate(size, &dma, &obj,
            RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT | RKNPU_MEM_CACHEABLE, &handle);
#ifdef FAKE_CACHE
        GGML_ASSERT(dma_alloc(size, &fd, &ptr) == 0);
#else
        ptr = dma_ptr;
#endif
        GGML_ASSERT(dma_ptr);
        scale = 1.0;
        pthread_spin_init(&scale_lock, 0);
    }
    ~rknn_mem(void) {
#ifdef FAKE_CACHE
        dma_buf_free(size, &fd, ptr);
#endif
        mem_destroy(dma_ptr, size, handle, obj);
    }
    void init_scale(void) { scale = SCALE_MIN; }
    void commit_scale(float _scale) {
        pthread_spin_lock(&scale_lock);
        scale = std::max(scale, _scale);
        pthread_spin_unlock(&scale_lock);
    }
    void reset_cnt(void) {
        pre_scale_cnt = 0;
        pre1_cnt = 0;
        post_cnt = 0;
    }
};

#define A_buf_size (rknn_type_size_A(type) * M * K)
#define B_buf_size (rknn_type_size_B(type) * K * N)
#define C_buf_size (rknn_type_size_C(type) * M * N)
#define A_size (rknn_type_size_A(type) * m * k)
#define B_size (rknn_type_size_B(type) * k * n)
#define C_size (rknn_type_size_C(type) * m * n)

struct A_bufs {
    int M, K, m, k;
    rknn_tensor_type type;
    std::map<std::tuple<int, int>, std::shared_ptr<rknn_mem>> As;

    A_bufs(int M, int K, int m, int k, rknn_tensor_type type)
        : M(M), K(K), m(m), k(k), type(type) {
        for (int mm = 0; mm < m; mm += M)
            for (int kk = 0; kk < k; kk += K)
                As.emplace(std::make_tuple(mm, kk), std::make_shared<rknn_mem>(A_buf_size));
    }
};
struct B_bufs {
    int K, N, k, n;
    rknn_tensor_type type;
    std::map<std::tuple<int, int>, std::shared_ptr<rknn_mem>> Bs;

    B_bufs(int K, int N, int k, int n, rknn_tensor_type type)
        : K(K), N(N), k(k), n(n), type(type) {
        for (int nn = 0; nn < n; nn += N)
            for (int kk = 0; kk < k; kk += K)
                Bs.emplace(std::make_tuple(nn, kk), std::make_shared<rknn_mem>(B_buf_size));
    }
};
struct C_bufs {
    int M, K, N, m, k, n;
    rknn_tensor_type type;
    std::map<std::tuple<int, int, int>, std::shared_ptr<rknn_mem>> Cs;

    C_bufs(int M, int K, int N, int m, int k, int n, rknn_tensor_type type)
        : M(M), K(K), N(N), m(m), k(k), n(n), type(type) {
        for (int mm = 0; mm < m; mm += M)
            for (int nn = 0; nn < n; nn += N)
                for (int kk = 0; kk < k; kk += K)
                    Cs.emplace(std::make_tuple(mm, nn, kk), std::make_shared<rknn_mem>(C_buf_size));
    }
};

struct matmul_buffer_mgr {
    std::map<std::tuple<int, int, int, int, rknn_tensor_type>, std::shared_ptr<A_bufs>> A_map;
    std::map<std::tuple<int, int, int, int, rknn_tensor_type>, std::shared_ptr<B_bufs>> B_map;
    std::map<std::tuple<int, int, int, int, int, int, rknn_tensor_type>, std::shared_ptr<C_bufs>> C_map;

    std::shared_ptr<A_bufs> get_A_bufs(int M, int K, int m, int k, rknn_tensor_type type) {
        auto A_bufs_iter = A_map.find(std::make_tuple(M, K, m, k, type));
        std::shared_ptr<A_bufs> ret;
        if (A_bufs_iter == A_map.end()) {
            ret = std::make_shared<A_bufs>(M, K, m, k, type);
            A_map.emplace(std::make_tuple(M, K, m, k, type), ret);
        } else {
            ret = A_bufs_iter->second;
        }
        return ret;
    }
    std::shared_ptr<B_bufs> get_B_bufs(int K, int N, int k, int n, rknn_tensor_type type) {
        auto B_bufs_iter = B_map.find(std::make_tuple(K, N, k, n, type));
        std::shared_ptr<B_bufs> ret;
        if (B_bufs_iter == B_map.end()) {
            ret = std::make_shared<B_bufs>(K, N, k, n, type);
            B_map.emplace(std::make_tuple(K, N, k, n, type), ret);
        } else {
            ret = B_bufs_iter->second;
        }
        return ret;
    }
    std::shared_ptr<C_bufs> get_C_bufs(int M, int K, int N, int m, int k, int n, rknn_tensor_type type) {
        auto C_bufs_iter = C_map.find(std::make_tuple(M, K, N, m, k, n, type));
        std::shared_ptr<C_bufs> ret;
        if (C_bufs_iter == C_map.end()) {
            ret = std::make_shared<C_bufs>(M, K, N, m, k, n, type);
            C_map.emplace(std::make_tuple(M, K, N, m, k, n, type), ret);
        } else {
            ret = C_bufs_iter->second;
        }
        return ret;
    }
    void clear(void) {
        A_map.clear();
        B_map.clear();
        C_map.clear();
    }
};

static struct matmul_buffer_mgr matmul_buffer_mgr;

struct npu_task {
    int M, N, K;
    uint64_t *regcmd;
    uint64_t regcmd_dma, regcmd_obj;
    uint64_t regcmd_handle;
    struct rknpu_task *tasks;
    uint64_t tasks_dma, tasks_obj;
    uint64_t tasks_handle;
    std::shared_ptr<rknn_mem> input; // A
    std::shared_ptr<rknn_mem> weight; // B
    std::shared_ptr<rknn_mem> output; // C
    uint64_t npu_regs[112];
    matmul_params_t params;
    rknn_tensor_type type;

    npu_task(int M, int N, int K, rknn_tensor_type type,
             std::shared_ptr<rknn_mem> input, std::shared_ptr<rknn_mem> weight, std::shared_ptr<rknn_mem> output)
        : M(M), N(N), K(K), type(type), input(input), weight(weight), output(output) {
        regcmd = (uint64_t*)mem_allocate(1024, &regcmd_dma, &regcmd_obj, 0, &regcmd_handle);
        GGML_ASSERT(regcmd);

        tasks = (rknpu_task *)mem_allocate(1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING, &tasks_handle);
        GGML_ASSERT(tasks);

        params.m = M;
        params.k = K;
        params.n = N;
        params.input_dma = input->dma;
        params.weights_dma = weight->dma;
        params.output_dma = output->dma;
        params.tasks = (uint64_t *)&npu_regs;
        if (type == RKNN_TENSOR_FLOAT32) {
            params.fp32tofp16 = 0;
            int ret = gen_matmul_fp16(&params);
            GGML_ASSERT(ret == 0);
        } else if (type == RKNN_TENSOR_INT8) {
            int ret = gen_matmul_int8(&params);
            GGML_ASSERT(ret == 0);
        } else {
            GGML_ASSERT(false);
        }
        memcpy(regcmd, npu_regs,sizeof(npu_regs));

        tasks[0].flags  = 0;
        tasks[0].op_idx = 0;
        tasks[0].enable_mask = 0xd;
        tasks[0].int_mask = 0x300; // wait for DPU to finish
        tasks[0].int_clear = 0x1ffff;
        tasks[0].int_status = 0;
        tasks[0].regcfg_amount = sizeof(npu_regs)/sizeof(uint64_t)-(RKNPU_PC_DATA_EXTRA_AMOUNT+4);
        tasks[0].regcfg_offset = 0;
        tasks[0].regcmd_addr = regcmd_dma;
    }

    ~npu_task(void) {
        mem_destroy(regcmd, 1024, regcmd_handle, regcmd_obj);
        mem_destroy(tasks, 1024, tasks_handle, tasks_obj);
    }

#ifdef GGML_USE_CHCORE
    void flush_cache(void) {
        usys_cache_flush((unsigned long)input->ptr, A_buf_size, CACHE_CLEAN);
#ifdef MAT_COPY
        usys_cache_flush((unsigned long)weight->ptr, B_buf_size, CACHE_CLEAN);
#endif
        usys_cache_flush((unsigned long)output->ptr, C_buf_size, CACHE_CLEAN_AND_INV);
    }
#else
    void flush_cache(void) {
#ifdef FAKE_CACHE
        GGML_ASSERT(dma_sync_cpu_to_device(input->fd) == 0);
#ifdef MAT_COPY
        GGML_ASSERT(dma_sync_cpu_to_device(weight->fd) == 0);
#endif
        GGML_ASSERT(dma_sync_cpu_to_device(output->fd) == 0);
        GGML_ASSERT(dma_sync_device_to_cpu(output->fd) == 0);
#endif
    }
    void submit(int core_mask) {
        flush_cache();
        int ret = npu_submit(tasks_obj, (__u32)core_mask);
        if (ret) {
            printf("RKNPU_SUBMIT returned %d, submitted m=%hu, k=%hu, n=%hu, errno %d\n",
                ret, M, K, N, errno);
        }
        GGML_ASSERT(ret == 0);
    }
#endif

    void apply_scale(void) {
        output->scale = input->scale * weight->scale;
    }
};

extern "C" {
    int npu_submit_multi(__u64 task_obj_addr[], int task_num, void *poll);
    void ggml_thread_cpu_relax_out(void);
}

struct npu_task_multi_core {
    std::vector<std::shared_ptr<npu_task>> npu_tasks;

    npu_task_multi_core(std::shared_ptr<npu_task> task)
        : npu_tasks(1, task) {}
    npu_task_multi_core(const std::vector<std::shared_ptr<npu_task>> &tasks)
        : npu_tasks(tasks) {}

#ifdef GGML_USE_CHCORE
    void submit(void) {
        GGML_ASSERT(npu_tasks.size() <= NPU_CORE_NUM);
        std::vector<uint64_t> tasks_objs;
        for (auto npu_task: npu_tasks) {
            npu_task->flush_cache();
            tasks_objs.push_back(npu_task->tasks_obj);
        }
        int ret = npu_submit_multi(tasks_objs.data(), tasks_objs.size(), (void *)ggml_thread_cpu_relax_out);
        GGML_ASSERT(ret == 0);
    }
#endif
    void apply_scale(void) {
        for (auto task: npu_tasks) {
            task->apply_scale();
        }
    }
};

struct matmul_kernel {
    const int MAX_N = 4096;
    const int ALIGN_N = 32;
    const int MAX_K = 4096;
    const int ALIGN_K = 32;
    int m, n, k;
    int M, N, K;
    rknn_tensor_type type;
    std::vector<std::shared_ptr<npu_task_multi_core>> npu_tasks;
    std::shared_ptr<A_bufs> inputs;
    std::shared_ptr<B_bufs> weights;
    std::shared_ptr<C_bufs> outputs;
    static inline int partition(int num, int max, int align) {
        int div = (num + max - 1) / max;
        int part = ((num + div - 1) / div + align - 1) / align * align;
        GGML_ASSERT(part <= max && part % align == 0);
        return part;
    }
    matmul_kernel(int m, int n, int k, rknn_tensor_type type)
        : m(m), n(n), k(k), type(type) {
        // partition m, n, k into M, N, K;

        N = partition(n / NPU_CORE_NUM, MAX_N, ALIGN_N);
        K = partition(k, MAX_K, ALIGN_K);
        // N = std::min(n / THREAD_NR / 32 * 32 + 32, 4096); K = std::min(k / 32 * 32 + 32, 4096);
        // let N be 4096, experiments show the following limitations: 
        if (type == RKNN_TENSOR_FLOAT32) {
            if (K <= 416) {
                M = std::min(m, 384);
            } else if (K <= 448) {
                M = std::min(m, 364);
            } else if (K <= 480) {
                M = std::min(m, 340);
            } else if (K <= 512) {
                M = std::min(m, 352);
            } else if (K <= 544) {
                M = std::min(m, 300);
            } else if (K <= 576) {
                M = std::min(m, 284);
            } else if (K <= 608) {
                M = std::min(m, 268);
            } else if (K <= 640) {
                M = std::min(m, 256);
            } else if (K <= 736) {
                M = std::min(m, 220);
            } else if (K <= 768) {
                M = std::min(m, 212);
            } else if (K <= 800) {
                M = std::min(m, 204);
            } else if (K <= 832) {
                M = std::min(m, 196);
            } else if (K <= 864) {
                M = std::min(m, 188);
            } else if (K <= 896) {
                M = std::min(m, 180);
            } else if (K <= 928) {
                M = std::min(m, 176);
            } else if (K <= 960) {
                M = std::min(m, 168);
            } else if (K <= 992) {
                M = std::min(m, 164);
            } else if (K <= 1024) {
                M = std::min(m, 176);
            } else if (K <= 1056) {
                M = std::min(m, 152);
            } else if (K <= 1088) {
                M = std::min(m, 148);
            } else if (K <= 1120) {
                M = std::min(m, 128);
            } else if (K <= 1152) {
                M = std::min(m, 140);
            } else if (K <= 1184) {
                M = std::min(m, 124);
            } else if (K <= 1216) {
                M = std::min(m, 120);
            } else if (K <= 1248) {
                M = std::min(m, 116);
            } else if (K <= 1280) {
                M = std::min(m, 128);
            } else if (K <= 1312) {
                M = std::min(m, 112);
            } else if (K <= 1344) {
                M = std::min(m, 108);
            } else if (K <= 1408) {
                M = std::min(m, 104);
            } else if (K <= 1472) {
                M = std::min(m, 100);
            } else if (K <= 1504) {
                M = std::min(m, 96);
            } else if (K <= 1536) {
                M = std::min(m, 100);
            } else if (K <= 1600) {
                M = std::min(m, 92);
            } else if (K <= 1664) {
                M = std::min(m, 88);
            } else if (K <= 1728) {
                M = std::min(m, 84);
            } else if (K <= 1824) {
                M = std::min(m, 80);
            } else if (K <= 1920) {
                M = std::min(m, 76);
            } else if (K <= 2016) {
                M = std::min(m, 72);
            } else if (K <= 2048) {
                M = std::min(m, 80);
            } else if (K <= 2112) {
                M = std::min(m, 68);
            } else if (K <= 2144) {
                M = std::min(m, 60);
            } else if (K <= 2176) {
                M = std::min(m, 64);
            } else if (K <= 2272) {
                M = std::min(m, 56);
            } else if (K <= 2304) {
                M = std::min(m, 64);
            } else if (K <= 2336) {
                M = std::min(m, 56);
            } else if (K <= 2496) {
                M = std::min(m, 52);
            } else if (K <= 2528) {
                M = std::min(m, 48);
            } else if (K <= 2560) {
                M = std::min(m, 56);
            } else if (K <= 2720) {
                M = std::min(m, 48);
            } else if (K <= 2976) {
                M = std::min(m, 44);
            } else if (K <= 3040) {
                M = std::min(m, 40);
            } else if (K <= 3072) {
                M = std::min(m, 48);
            } else if (K <= 3136) {
                M = std::min(m, 40);
            } else if (K <= 3168) {
                M = std::min(m, 36);
            } else if (K <= 3200) {
                M = std::min(m, 40);
            } else if (K <= 3296) {
                M = std::min(m, 32);
            } else if (K <= 3328) {
                M = std::min(m, 36);
            } else if (K <= 3552) {
                M = std::min(m, 32);
            } else if (K <= 3584) {
                M = std::min(m, 36);
            } else if (K <= 4064) {
                M = std::min(m, 28);
            } else {
                M = std::min(m, 32);
            }
        } else if (type == RKNN_TENSOR_INT8) {
            if (K <= 576) {
                M = std::min(m, 544);
            } else if (K <= 768) {
                M = std::min(m, 424);
            } else if (K <= 960) {
                M = std::min(m, 340);
            } else if (K <= 992) {
                M = std::min(m, 320);
            } else if (K <= 1024) {
                M = std::min(m, 352);
            } else if (K <= 1280) {
                M = std::min(m, 256);
            } else if (K <= 1472) {
                M = std::min(m, 200);
            } else if (K <= 1504) {
                M = std::min(m, 192);
            } else if (K <= 1536) {
                M = std::min(m, 212);
            } else if (K <= 1984) {
                M = std::min(m, 148);
            } else if (K <= 2016) {
                M = std::min(m, 144);
            } else if (K <= 2048) {
                M = std::min(m, 160);
            } else if (K <= 2528) {
                M = std::min(m, 100);
            } else if (K <= 2560) {
                M = std::min(m, 112);
            } else if (K <= 3040) {
                M = std::min(m, 84);
            } else if (K <= 3072) {
                M = std::min(m, 96);
            } else if (K <= 3552) {
                M = std::min(m, 64);
            } else if (K <= 3584) {
                M = std::min(m, 72);
            } else if (K <= 4064) {
                M = std::min(m, 56);
            } else {
                M = std::min(m, 64);
            }
        }
        // Besides, the open source driver could only handle (m==1) or (m%4==0).
        // Otherwise the results are not accurate.
        if (M != 1 && M % 4 != 0) {
            // Change to next multiple of 4
            M = (M + 3) / 4 * 4;
        }

        // printf("partition (m, k, n) = (%d, %d, %d) to (M, K, N) = (%d, %d, %d)\n", m, k, n, M, K, N);
        
        inputs = matmul_buffer_mgr.get_A_bufs(M, K, m, k, type);
#ifdef MAT_COPY
        weights = matmul_buffer_mgr.get_B_bufs(K, N, k, n, type);
#else
        weights = nullptr;
#endif
        outputs = matmul_buffer_mgr.get_C_bufs(M, K, N, m, k, n, type);

        for (int kk = 0; kk < k; kk += K) {
            for (int mm = 0; mm < m; mm += M) {
                std::vector<std::shared_ptr<npu_task>> tmp_tasks;
                for (int nn = 0; nn < n; nn += N) {
                    auto input = inputs->As.find(std::make_tuple(mm, kk))->second;
#ifdef MAT_COPY
                    auto weight = weights->Bs.find(std::make_tuple(nn, kk))->second;
#else
                    static std::shared_ptr<rknn_mem> global_weight = std::make_shared<rknn_mem>(4096 * 4096 * rknn_type_size_B(type));
                    std::shared_ptr<rknn_mem> weight(global_weight);
#endif
                    auto output = outputs->Cs.find(std::make_tuple(mm, nn, kk))->second;
                    tmp_tasks.emplace_back(std::make_shared<npu_task>(M, N, K, type, input, weight, output));
                    if (tmp_tasks.size() == NPU_CORE_NUM || nn + N >= n) {
                        npu_tasks.emplace_back(std::make_shared<npu_task_multi_core>(tmp_tasks));
                        tmp_tasks.clear();
                    }
                }
                GGML_ASSERT(tmp_tasks.empty());
            }
        }
    }

    void for_all_inputs(std::function<void(int, int, int, int, std::shared_ptr<rknn_mem>)> closure) {
        for (const auto &[shape, mem]: inputs->As) {
            auto [mm, kk] = shape;
            closure(mm, kk, M, K, mem);
        }
    }
    void for_all_weights(std::function<void(int, int, int, int, std::shared_ptr<rknn_mem>)> closure) {
#ifdef MAT_COPY
        for (const auto &[shape, mem]: weights->Bs) {
            auto [nn, kk] = shape;
            closure(nn, kk, N, K, mem);
        }
#endif
    }
    void for_all_outputs(std::function<void(int, int, int, int, std::shared_ptr<rknn_mem>)> closure) {
        for (const auto &[shape, mem]: outputs->Cs) {
            auto [mm, nn, _] = shape;
            closure(mm, nn, M, N, mem);
        }
    }

#ifndef GGML_USE_CHCORE
    std::vector<std::vector<std::shared_ptr<npu_task>>> to_multi_npu(int npu_nr) const {
        std::vector<std::vector<std::shared_ptr<npu_task>>> tasks;
        tasks.resize(npu_nr);
        for (auto task: npu_tasks) {
            GGML_ASSERT(task->npu_tasks.size() <= npu_nr);
            for (std::size_t i = 0; i < task->npu_tasks.size(); i++) {
                tasks[i].push_back(task->npu_tasks[i]);
            }
        }
        return tasks;
    }
#endif
};

std::vector<std::shared_ptr<matmul_kernel>> matmul_kernels;

static std::shared_ptr<matmul_kernel>
ggml_rknpu2_matmul_kernel_find(int m, int k, int n, rknn_tensor_type type) {
    for (const auto &kernel: matmul_kernels) {
        if (kernel->m == m && kernel->k == k && kernel->n == n && kernel->type == type) {
            return kernel;
        }
    }
    return nullptr;
}
// first find from buffer, then reuse them
static std::shared_ptr<matmul_kernel>
ggml_rknpu2_matmul_kernel_create(int m, int k, int n, rknn_tensor_type type)
{
    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, type);
    if (kernel != NULL)
        return kernel;

    kernel = std::make_shared<matmul_kernel>(m, n, k, type);
    matmul_kernels.emplace_back(kernel);
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

std::mutex mtx;
std::condition_variable cv_worker;
std::condition_variable cv_master;
bool done;
bool ready[THREAD_NR];
std::atomic<int> finished_nr;
static std::once_flag once_flag;

#ifdef GGML_USE_CHCORE
std::vector<std::shared_ptr<npu_task_multi_core>> npu_tasks[THREAD_NR];
#else
std::vector<std::shared_ptr<npu_task>> npu_tasks[THREAD_NR];
#endif

void npu_worker(int tid) {
    int ith = tid;
#ifndef GGML_USE_CHCORE
    struct sched_param param;
    int policy = SCHED_FIFO;
    int priority = 40;
    param.sched_priority = priority;
    GGML_ASSERT(pthread_setschedparam(pthread_self(), policy, &param) == 0);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid + 4, &cpuset);
    printf("npu worker: bind to core %d\n", tid + 4);
    GGML_ASSERT(pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0);
#else
    usys_set_prio(0, 54);
#endif

    rknn_core_mask core_mask;
    if (tid == 0) core_mask = 0x1;
    if (tid == 1) core_mask = 0x2;
    if (tid == 2) core_mask = 0x4;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_worker.wait(lock, [tid] { return ready[tid] || done; });
        }
        BEGIN_MEASURE_0;

        if (done) break;

        for (auto task: npu_tasks[tid]) {
            task->apply_scale();
#ifdef GGML_USE_CHCORE
            task->submit();
#else
            task->submit(core_mask);
#endif
        }

        {
            std::lock_guard<std::mutex> lock(mtx);
            finished_nr++;
            ready[tid] = false;
            if (finished_nr == THREAD_NR) {
                cv_master.notify_one();
            }
        }
        END_MEASURE_0;
    }
}

std::vector<std::thread> npu_threads;

static void init_npu_thread(int thread_nr) {
    GGML_ASSERT(thread_nr <= 3);
    for (int i = 0; i < thread_nr; i++) {
        npu_threads.emplace_back(npu_worker, i);
    }
}

static inline rknn_tensor_type ggml_type_to_rknn_type(enum ggml_type type) {
    if (type == GGML_TYPE_F16) return RKNN_TENSOR_FLOAT32;
    if (type == GGML_TYPE_Q8_0) return RKNN_TENSOR_INT8;
    GGML_ASSERT(false);
}

static inline int8_t f32_to_i8(float x, float scale) {
    return (int8_t)std::min(std::max(x / scale, -127.f), 127.f);
}
static inline float i32_to_f32(int32_t x, float scale) {
    return (float)x * scale;
}

/*
void matmul_perf_test(int m, int k, int n) {
    rknn_tensor_type tensor_type = RKNN_TENSOR_INT8;
    auto kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

    auto tasks = kernel->to_multi_npu(1);
    GGML_ASSERT(tasks.size() == 1 && tasks[0].size() == 1);

    std::queue<std::pair<int64_t, int64_t>> tss;
    int64_t sum = 0;

    while (1) {
        auto start = get_micro();

        tasks[0][0]->submit(1);
        
        auto end = get_micro();
        auto cur = end - start;
        tss.push({end, cur});
        sum += cur;
        while (!tss.empty()) {
            if (end - tss.front().first <= 10 * 1000000) break;
            sum -= tss.front().second;
            tss.pop();
        }
        printf("%s %d: (m, k, n) = (%d %d %d) cnt %ld average %ld us current %ld us\n", __func__, __LINE__, m, k, n, tss.size(), sum / tss.size(), cur);
    }
}
*/

extern "C" {

void rknpu2_matmul_begin_measure(int ith) {
    BEGIN_MEASURE_0;
}
void rknpu2_matmul_end_measure(int ith) {
    END_MEASURE_0;
}
void rknpu2_matmul_begin_measure_npu(int ith) {
    BEGIN_MEASURE_0;
}
void rknpu2_matmul_end_measure_npu(int ith) {
    END_MEASURE_0;
}

void rknpu2_matmul_pre0(struct ggml_tensor * dst, int nth, int ith) {
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

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    static bool cleared;
    if (ith == 0) {
        /* now in decoding stage, clear all prefill buffers */
        if (m == 1 && cleared == false) {
            matmul_kernels.clear();
            matmul_buffer_mgr.clear();
            cleared = true;
            ggml_rknpu_dump_measure();
        }
    }

    BEGIN_MEASURE_0;

    GGML_ASSERT(GGML_TYPE_F16 == type || GGML_TYPE_Q8_0 == type);

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(type);

    if (ith == 0) {
        auto kernel = ggml_rknpu2_matmul_kernel_create(m, k, n, tensor_type);
        GGML_ASSERT(kernel);
        memset(dst->data, 0, m * n * sizeof(float));

        float *A = (float*)src1->data;
        void *B = src0->data;

        kernel->for_all_inputs(
            [&](int mm, int kk, int M, int K, std::shared_ptr<rknn_mem> input_mem) {
                if (tensor_type == RKNN_TENSOR_INT8) {
                    input_mem->init_scale();
                }
                input_mem->reset_cnt();
            }
        );
        kernel->for_all_weights(
            [&](int nn, int kk, int N, int K, std::shared_ptr<rknn_mem> weight_mem) {
                if (tensor_type == RKNN_TENSOR_INT8) {
                    weight_mem->init_scale();
                }
                weight_mem->reset_cnt();
            }
        );
        kernel->for_all_outputs(
            [&](int mm, int nn, int M, int N, std::shared_ptr<rknn_mem> output_mem) {
                output_mem->reset_cnt();
            }
        );
    }
    END_MEASURE_0;
}

void rknpu2_matmul_pre_scale(struct ggml_tensor * dst, int nth, int ith) {
    BEGIN_MEASURE_0;
    /* src0 => matrix B */
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(src0->type);

    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

    float *A = (float*)src1->data;
    void *B = src0->data;

    kernel->for_all_inputs(
        [&](int mm, int kk, int M, int K, std::shared_ptr<rknn_mem> input_mem) {
            if (tensor_type == RKNN_TENSOR_INT8) {
                float scale = SCALE_MIN;
                for (int i = input_mem->pre_scale_cnt.fetch_add(1); i < M; i = input_mem->pre_scale_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    if (ii >= m) break;
                    for (int j = 0; j < K; j++) {
                        int jj = kk + j;
                        if (jj >= k) break;
                        scale = std::max(scale, std::abs(A[ii * k + jj]));
                    }
                }
                input_mem->commit_scale(scale / 127.f);
            }
        }
    );
    kernel->for_all_weights(
        [&](int nn, int kk, int N, int K, std::shared_ptr<rknn_mem> weight_mem) {
            auto weight = weight_mem->ptr;
            if (tensor_type == RKNN_TENSOR_INT8) {
                float scale = SCALE_MIN;
                ggml_type_traits_t traits = ggml_internal_get_type_traits(src0->type);
                GGML_ASSERT(traits.to_float != NULL);
                int nele = k * n;
                float *fB = (float *)malloc(nele * sizeof(*fB));
                traits.to_float(B, fB, nele);
                for (int i = weight_mem->pre_scale_cnt.fetch_add(1); i < N; i = weight_mem->pre_scale_cnt.fetch_add(1))
                    for (int j = 0; j < K; j++) {
                        int ii = nn + i;
                        int jj = kk + j;
                        if (ii >= n || jj >= k) continue;
                        scale = std::max(scale, std::abs(fB[ii * k + jj]));
                    }
                weight_mem->commit_scale(scale / 127.f);
                free(fB);
            }
        }
    );
    END_MEASURE_0;
}

std::atomic<bool> finish;

void rknpu2_matmul_pre1(struct ggml_tensor * dst, int nth, int ith) {
    BEGIN_MEASURE_0;
    /* src0 => matrix B */
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    float *A = (float*)src1->data;
    void *B = src0->data;

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(src0->type);

    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

    kernel->for_all_inputs(
        [&](int mm, int kk, int M, int K, std::shared_ptr<rknn_mem> input_mem) {
            auto input = input_mem->ptr;
            if (tensor_type == RKNN_TENSOR_FLOAT32) {
                GGML_ASSERT(k % 8 == 0);
                for (int i = input_mem->pre1_cnt.fetch_add(1); i < M; i = input_mem->pre1_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    int off_in_sub_mat = i * 8;
                    if (ii >= m) break;
                    for (int j = 0; j < K; j += 8) {
                        int jj = kk + j;
                        if (jj >= k) break;
                        int base_sub_mat = M * j;
                        for (int t = 0; t < 8; t++) {
                            ((__fp16 *)input)[base_sub_mat + off_in_sub_mat + t] = A[ii * k + jj + t];
                        }
                    }
                }
            } else {
                GGML_ASSERT(tensor_type == RKNN_TENSOR_INT8);
                GGML_ASSERT(k % 16 == 0);
                for (int i = input_mem->pre1_cnt.fetch_add(1); i < M; i = input_mem->pre1_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    int off_in_sub_mat = i * 16;
                    if (ii >= m) break;
                    for (int j = 0; j < K; j += 16) {
                        int jj = kk + j;
                        if (jj >= k) break;
                        int base_sub_mat = M * j;
                        for (int t = 0; t < 16; t++) {
                            ((int8_t *)input)[base_sub_mat + off_in_sub_mat + t] = f32_to_i8(A[ii * k + jj + t], input_mem->scale);
                        }
                    }
                }
            }
        }
    );
    kernel->for_all_weights(
        [&](int nn, int kk, int N, int K, std::shared_ptr<rknn_mem> weight_mem) {
            auto weight = weight_mem->ptr;
            if (tensor_type == RKNN_TENSOR_FLOAT32) {
                for (int i = weight_mem->pre1_cnt.fetch_add(1); i < N; i = weight_mem->pre1_cnt.fetch_add(1))
                    for (int j = 0; j < K; j++) {
                        int ii = nn + i;
                        int jj = kk + j;
                        if (ii >= n || jj >= k) continue;
                        ((__fp16 *)weight)[weight_fp16(K, i + 1, j + 1)] = ((__fp16 *)B)[ii * k + jj];
                    }
            } else if (tensor_type == RKNN_TENSOR_INT8) {
                ggml_type_traits_t traits = ggml_internal_get_type_traits(src0->type);
                GGML_ASSERT(traits.to_float != NULL);
                int nele = k * n;
                float *fB = (float *)malloc(nele * sizeof(*fB));
                traits.to_float(B, fB, nele);
                for (int i = weight_mem->pre1_cnt.fetch_add(1); i < N; i = weight_mem->pre1_cnt.fetch_add(1))
                    for (int j = 0; j < K; j++) {
                        int ii = nn + i;
                        int jj = kk + j;
                        if (ii >= n || jj >= k) continue;
                        ((int8_t *)weight)[weight_int8(K, i + 1, j + 1)] = f32_to_i8(fB[ii * k + jj], weight_mem->scale);
                    }
                free(fB);
            }
        }
    );
    END_MEASURE_0;
    finish = false;
}

void rknpu2_matmul_submit(struct ggml_tensor * dst, int nth, int ith) {
    if (ith) {
        // while (!finish.load())
        //     ggml_thread_cpu_relax_out();
        return;
    }
    BEGIN_MEASURE_0;
    /* src0 => matrix B */
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(src0->type);

    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

#ifdef GGML_USE_CHCORE
    for (auto task: kernel->npu_tasks) {
        task->apply_scale();
        task->submit();
    }
#else
    auto tasks = kernel->to_multi_npu(THREAD_NR);
    for (int i = 0; i < THREAD_NR; i++)
        npu_tasks[i].assign(tasks[i].begin(), tasks[i].end());

    std::call_once(once_flag, init_npu_thread, THREAD_NR);

    finished_nr = 0;
    for (int i = 0; i < THREAD_NR; i++)
        ready[i] = true;
    cv_worker.notify_all();
    // std::unique_lock<std::mutex> lock(mtx);
    // cv_master.wait(lock, [] { return finished_nr == THREAD_NR; });
    while (finished_nr.load() != THREAD_NR) {
        ggml_thread_cpu_relax_out();
    }
    for (int i = 0; i < THREAD_NR; i++)
        npu_tasks[i].clear();
#endif
    finish = true;
    END_MEASURE_0;
}

void rknpu2_matmul_post(struct ggml_tensor * dst, int nth, int ith) {
    /*const */struct ggml_tensor * src0 = dst->src[0];
    /*const*/ struct ggml_tensor * src1 = dst->src[1];

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = dst->ne[0];

    rknn_tensor_type tensor_type = ggml_type_to_rknn_type(src0->type);

    float *C = (float*)dst->data;

    auto kernel = ggml_rknpu2_matmul_kernel_find(m, k, n, tensor_type);
    GGML_ASSERT(kernel);

    BEGIN_MEASURE_0;
    kernel->for_all_outputs(
        [&](int mm, int nn, int M, int N, std::shared_ptr<rknn_mem> output_mem) {
            auto output = output_mem->ptr;
            if (tensor_type == RKNN_TENSOR_FLOAT32) {
                GGML_ASSERT(n % 4 == 0);
                for (int i = output_mem->post_cnt.fetch_add(1); i < M; i = output_mem->post_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    int off_in_sub_mat = i * 4;
                    if (ii >= m) break;
                    for (int j = 0; j < N; j += 4) {
                        int jj = nn + j;
                        if (jj >= n) break;
                        int base_sub_mat = M * j;
                        for (int t = 0; t < 4; t++) {
                            C[ii * n + jj + t] += ((float *)output)[base_sub_mat + off_in_sub_mat + t];
                        }
                    }
                }
            } else {
                GGML_ASSERT(tensor_type == RKNN_TENSOR_INT8);
                GGML_ASSERT(n % 4 == 0);
                for (int i = output_mem->post_cnt.fetch_add(1); i < M; i = output_mem->post_cnt.fetch_add(1)) {
                    int ii = mm + i;
                    int off_in_sub_mat = i * 4;
                    if (ii >= m) break;
                    for (int j = 0; j < N; j += 4) {
                        int jj = nn + j;
                        if (jj >= n) break;
                        int base_sub_mat = M * j;
                        for (int t = 0; t < 4; t++) {
                            C[ii * n + jj + t] += i32_to_f32(((int32_t *)output)[base_sub_mat + off_in_sub_mat + t], output_mem->scale);
                        }
                    }
                }
            }
        }
    );
    END_MEASURE_0;
}

}

static void ggml_backend_rknpu2_mul_mat(ggml_backend_rknpure_context * ctx, struct ggml_tensor * dst) {
    rknpu2_matmul_pre0(dst, 1, 0);
    rknpu2_matmul_pre1(dst, 1, 0);
    rknpu2_matmul_submit(dst, 1, 0);
    rknpu2_matmul_post(dst, 1, 0);
}

// backend interface

GGML_CALL static const char * ggml_backend_rknpu2_name(ggml_backend_t backend) {
    return "RKNPURE";

    GGML_UNUSED(backend);
}

GGML_CALL static void ggml_backend_rknpu2_free(ggml_backend_t backend) {
    ggml_backend_rknpure_context * ctx = (ggml_backend_rknpure_context *)backend->context;
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
    matmul_kernels.clear();

    done = true;
    cv_worker.notify_all();
    for (auto &thread: npu_threads) {
        thread.join();
    }
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_rknpu2_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_rknpure_context * ctx = (ggml_backend_rknpure_context *) backend->context;
    return ggml_backend_rknpure_buffer_type(ctx->device);
}

GGML_CALL static enum ggml_status ggml_backend_rknpu2_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_rknpure_context * ctx = (ggml_backend_rknpure_context *)backend->context;

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

bool is_strawman = false;
void ggml_backend_rknpure_set_strawman(bool strawman) {
    is_strawman = strawman;
}

GGML_CALL static bool ggml_backend_rknpure_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst = op;

    if (is_strawman) {
        return false;
    }

    if (op->op != GGML_OP_MUL_MAT) {
        // printf("zzh: op is %d, not mul mat\n", op->op);
        return false;
    }

    // /* cpu computation when batch size == 1, i.e., decoding stage */
    // if (src1->ne[1] == 1) {
    //     return false;
    // }

    {
        const int64_t m = src1->ne[1];
        const int64_t k = src0->ne[0];
        const int64_t n = dst->ne[0];
        /* can not allocate large B buffers for large vocab_size. just use cpu to perform these matmuls */
        if (k >= 50000 || n >= 50000)
            return false;
    }

    // printf("ggml_backend_rknpure_supports_op, %d, %d, %p\n", src1->type, dst->type, src0->extra);
    // return false; // DEBUG: first, never use this backend

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    if (ggml_is_contiguous(src0) &&
        ggml_is_contiguous(src1) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
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
        if (strcmp(src0->buffer->buft->iface.get_name(src0->buffer->buft), "RKNPURE")) {
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
    bool ggml_backend_rknpure_supports_op_out(const struct ggml_tensor *op) {
        ggml_backend_t backend;
        return ggml_backend_rknpure_supports_op(backend, op);
    }
}
GGML_CALL static bool ggml_backend_rknpu2_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    // zzh: maybe this is wrong! however qnn doesn't have this.
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(backend);
}

extern "C" {
void ggml_backend_rknpure_mul_mat_out(struct ggml_tensor * dst) {
    return ggml_backend_rknpu2_mul_mat(NULL, dst);
}
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
    /* .supports_op             = */ ggml_backend_rknpure_supports_op,
    /* .supports_buft           = */ ggml_backend_rknpu2_supports_buft,
    /* .offload_op              = */ ggml_backend_rknpu2_offload_op, // qnn set this
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_rknpu2_guid(void) {
    static ggml_guid guid = { 0x3c, 0x29, 0xa9, 0xac, 0x54, 0x27, 0x40, 0x55, 0x89, 0xf2, 0xdc, 0x83, 0xf1, 0xba, 0x02, 0xc4 };
    return &guid;
}

static ggml_backend_t ggml_backend_rknpu2_reg_init(const char *params, void * user_data) {
    ggml_backend_t rknpu2_backend = ggml_backend_rknpure_init((int) (unsigned long) user_data);
    return rknpu2_backend;
}

ggml_backend_t ggml_backend_rknpure_init(int32_t device) {
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

    ggml_backend_rknpure_context * ctx = (ggml_backend_rknpure_context *)backend_rknpu2->context;
    ctx->n_threads = n_threads;
}

extern "C" GGML_CALL int ggml_backend_rknpure_reg_devices();

GGML_CALL int ggml_backend_rknpure_reg_devices() {
    // ggml_rknpu2_instance_init();
    uint32_t device_count = ggml_backend_rknpu2_get_device_count();

    for (size_t i = 0; i < device_count; i++) {
        //char name[128];
        //snprintf(name, sizeof(name), "%s%ld", GGML_VK_NAME, i);
        ggml_backend_register(g_rknpu2_mgr[i].name, ggml_backend_rknpu2_reg_init, ggml_backend_rknpure_buffer_type(i), (void *) (intptr_t) i);  // NOLINT
    }
    return device_count;
}