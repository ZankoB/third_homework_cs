#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include <gem5/m5ops.h>

#define SEQ_LEN 256
#define D_K 64
#define SCALE 0.125f  // 1/sqrt(64)

float dot_product(float *q, float *k, int dim) {
    float result = 0.0f;
    int i = 0;

    while (i < dim) {
        size_t vl = __riscv_vsetvl_e32m1(dim - i);

        vfloat32m1_t qv = __riscv_vle32_v_f32m1(&q[i], vl);
        vfloat32m1_t kv = __riscv_vle32_v_f32m1(&k[i], vl);
        vfloat32m1_t prod = __riscv_vfmul_vv_f32m1(qv, kv, vl);

        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t sum_vec =
            __riscv_vfredusum_vs_f32m1_f32m1(prod, zero, vl);

        result += __riscv_vfmv_f_s_f32m1_f32(sum_vec);

        i += vl;
    }

    return result;
}

void softmax_row(float *scores, float *output, int len) {
    float max_val = -INFINITY;
    int i = 0;

    while (i < len) {
        size_t vl = __riscv_vsetvl_e32m1(len - i);

        vfloat32m1_t v = __riscv_vle32_v_f32m1(&scores[i], vl);

        vfloat32m1_t init = __riscv_vfmv_v_f_f32m1(-INFINITY, vl);
        vfloat32m1_t max_vec =
            __riscv_vfredmax_vs_f32m1_f32m1(v, init, vl);

        float chunk_max = __riscv_vfmv_f_s_f32m1_f32(max_vec);

        if (chunk_max > max_val)
            max_val = chunk_max;

        i += vl;
    }

    float sum = 0.0f;

    for (int k = 0; k < len; k++) {
        output[k] = expf((scores[k] - max_val) * SCALE);
        sum += output[k];
    }

    i = 0;

    while (i < len) {
        size_t vl = __riscv_vsetvl_e32m1(len - i);

        vfloat32m1_t out = __riscv_vle32_v_f32m1(&output[i], vl);
        out = __riscv_vfdiv_vf_f32m1(out, sum, vl);
        __riscv_vse32_v_f32m1(&output[i], out, vl);

        i += vl;
    }
}

int main() {
    float Q[SEQ_LEN][D_K];
    float K[SEQ_LEN][D_K];
    float scores[SEQ_LEN];
    float attn_weights[SEQ_LEN];

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < D_K; j++) {
            Q[i][j] = (float)(i + j) * 0.01f;
            K[i][j] = (float)(i - j) * 0.01f;
        }
    }

#ifdef GEM5
    m5_reset_stats(0, 0);
#endif

    for (int j = 0; j < SEQ_LEN; j++) {
        scores[j] = dot_product(Q[0], K[j], D_K);
    }

    softmax_row(scores, attn_weights, SEQ_LEN);

#ifdef GEM5
    m5_dump_stats(0, 0);
#endif

    for (int i = 0; i < 4; i++) {
        printf("attn[%d] = %.6f\n", i, attn_weights[i]);
    }

    return 0;
}