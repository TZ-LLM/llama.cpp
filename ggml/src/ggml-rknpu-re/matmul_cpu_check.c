void cpu_matmul_fp16_fp16_fp32(int m, int k, int n, const __fp16 *src0 , const __fp16 *src1, float* dst) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0;
      for (int l = 0; l < k; l++) {
        sum += src0[i*k + l] * src1[j*k + l];
      }
     dst[i*n + j] = sum;
    }
  }
}
