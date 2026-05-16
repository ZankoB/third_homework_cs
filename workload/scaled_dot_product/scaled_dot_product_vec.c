#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include <gem5/m5ops.h>

#define SEQ_LEN 256
#define D_K 64
#define SCALE 0.125f  // 1/sqrt(64)

void softmax_row(float *scores, float *output, int len) {
    float max_val = scores[0];

    /*
     * Vectorized max reduction over scores.
     */
    int i = 0;
    size_t vl = __riscv_vsetvl_e32m1(len);
    vfloat32m1_t max_vec = __riscv_vle32_v_f32m1(&scores[0], vl);

    i += vl;

    while (i < len) {
        vl = __riscv_vsetvl_e32m1(len - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(&scores[i], vl);
        max_vec = __riscv_vfmax_vv_f32m1(max_vec, v, vl);
        i += vl;
    }

    /*
     * Store vector max lanes and finish scalar reduction.
     * This is safe because SEQ_LEN is small and fixed.
     */
    float tmp[SEQ_LEN];
    vl = __riscv_vsetvl_e32m1(len);
    __riscv_vse32_v_f32m1(tmp, max_vec, vl);

    max_val = tmp[0];
    for (int k = 1; k < (int)vl; k++) {
        if (tmp[k] > max_val)
            max_val = tmp[k];
    }

    /*
     * Keep expf and sum scalar to preserve the same numerical behavior
     * as the original code.
     */
    float sum = 0.0f;
    for (int k = 0; k < len; k++) {
        output[k] = expf((scores[k] - max_val) * SCALE);
        sum += output[k];
    }

    /*
     * Vectorized normalization: output[k] /= sum.
     */
    i = 0;
    while (i < len) {
        vl = __riscv_vsetvl_e32m1(len - i);
        vfloat32m1_t out = __riscv_vle32_v_f32m1(&output[i], vl);
        out = __riscv_vfdiv_vf_f32m1(out, sum, vl);
        __riscv_vse32_v_f32m1(&output[i], out, vl);
        i += vl;
    }
}

float dot_product(float *q, float *k, int dim) {
    /*
     * Vectorized multiply, scalar ordered accumulation.
     *
     * This preserves the original addition order:
     * result += q[i] * k[i]
     *
     * A fully vector-reduced dot product would be faster, but it may slightly
     * change floating-point roundoff and therefore could change the printed
     * values.
     */
    float products[D_K];

    int i = 0;
    while (i < dim) {
        size_t vl = __riscv_vsetvl_e32m1(dim - i);

        vfloat32m1_t qv = __riscv_vle32_v_f32m1(&q[i], vl);
        vfloat32m1_t kv = __riscv_vle32_v_f32m1(&k[i], vl);
        vfloat32m1_t pv = __riscv_vfmul_vv_f32m1(qv, kv, vl);

        __riscv_vse32_v_f32m1(&products[i], pv, vl);

        i += vl;
    }

    float result = 0.0f;
    for (int j = 0; j < dim; j++) {
        result += products[j];
    }

    return result;
}

int main() {
    float Q[SEQ_LEN][D_K];
    float K[SEQ_LEN][D_K];
    float scores[SEQ_LEN];
    float attn_weights[SEQ_LEN];

    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < D_K; j++) {
            Q[i][j] = (float)(i + j) * 0.01f;
            K[i][j] = (float)(i - j) * 0.01f;
        }

#ifdef GEM5
    m5_reset_stats(0, 0);
#endif

    for (int j = 0; j < SEQ_LEN; j++)
        scores[j] = dot_product(Q[0], K[j], D_K);

    softmax_row(scores, attn_weights, SEQ_LEN);

#ifdef GEM5
    m5_dump_stats(0, 0);
#endif

    for (int i = 0; i < 4; i++)
        printf("attn[%d] = %.6f\n", i, attn_weights[i]);

    return 0;
}