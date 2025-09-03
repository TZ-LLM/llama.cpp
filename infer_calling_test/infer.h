#ifndef __LLAMA_TEE__
#define __LLAMA_TEE__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
int infer(int argc, char **argv, char *input, size_t input_len, char *output,
          size_t output_len);

#ifdef __cplusplus
}
#endif

#endif // __LLAMA_TEE__
