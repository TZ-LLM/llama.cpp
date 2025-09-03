#ifndef MATMUL_CPU_CHECK_H
#define MATMUL_CPU_CHECK_H
#ifdef __cplusplus
extern "C" {
#endif

void cpu_matmul_fp16_fp16_fp32(int m, int k, int n, const __fp16 *src0 , const __fp16 *src1, float* dst);

#ifdef __cplusplus
}
#endif
#endif // MATMUL_CPU_CHECK_H
