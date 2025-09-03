#include "ggml.h"

void record_tensor_size(size_t size);
void register_param_tensor(
    ggml_tensor *tensor,
    size_t off,
    size_t len,
    int fd
);
void reset_param_tensor(void);
struct ggml_tensor * ggml_use_param_wrapper(
        struct ggml_context * ctx,
        struct ggml_tensor  * self,
        struct ggml_tensor  * dep
);
