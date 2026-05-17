# Benchmark Results

## 1. Vectorizing Kernel (Scaled Dot-Product)

- **L1 Cache Size:** 8KB
- **Metrics:** CPI, Execution Time

| VPU Size (bits) | Scalar CPI | Scalar Exec Time | Vectorized CPI | Vectorized Exec Time |
| --------------- | ---------- | ---------------- | -------------- | -------------------- |
| 128             | 0.775324   | 0.000110         | 2.491596       | 0.000852             |
| 256             | 0.775324   | 0.000110         | 2.460773       | 0.000451             |
| 512             | 0.775324   | 0.000110         | 2.412990       | 0.000251             |
| 1024            | 0.775324   | 0.000110         | 2.329477       | 0.000150             |
| 2048            | 0.775324   | 0.000110         | 2.229050       | 0.000099             |
| 4096            | 0.775324   | 0.000110         | 2.227282       | 0.000099             |

## 2. Stencil Kernel (Heat Diffusion)

- **Metrics:** CPI, L1 Cache Misses

### 8KB L1 Cache Size

| VPU Size (bits) | Scalar CPI | Scalar L1 Misses | Vectorized CPI | Vectorized L1 Misses |
| --------------- | ---------- | ---------------- | -------------- | -------------------- |
| 128             | 0.847457   | 65               | 2.547448       | 50                   |
| 256             | 0.847457   | 65               | 2.568728       | 52                   |
| 512             | 0.847457   | 65               | 2.612624       | 55                   |
| 1024            | 0.847457   | 65               | 2.676691       | 68                   |
| 2048            | 0.847457   | 65               | 2.810465       | 61                   |
| 4096            | 0.847457   | 65               | 3.901307       | 355                  |

### 64KB L1 Cache Size

| VPU Size (bits) | Scalar CPI | Scalar L1 Misses | Vectorized CPI | Vectorized L1 Misses |
| --------------- | ---------- | ---------------- | -------------- | -------------------- |
| 128             | 0.827531   | 0                | 2.539969       | 42                   |
| 256             | 0.827531   | 0                | 2.552192       | 42                   |
| 512             | 0.827531   | 0                | 2.578073       | 42                   |
| 1024            | 0.827531   | 2                | 2.608516       | 44                   |
| 2048            | 0.827531   | 6                | 2.677283       | 48                   |
| 4096            | 0.827531   | 9                | 2.805427       | 51                   |

## 3. Sparse Matrix-Vector Multiplication (SpMV)

- **Metrics:** CPI, L1 Cache Misses

### 8KB L1 Cache Size

| Kernel Pattern  | VPU Size (bits) | CPI      | L1 Cache Misses |
| --------------- | --------------- | -------- | --------------- |
| Unit-stride     | 256             | 2.504482 | 32              |
| Unit-stride     | 512             | 2.572486 | 41              |
| Unit-stride     | 1024            | 2.711817 | 44              |
| Strided         | 256             | 2.562149 | 107             |
| Strided         | 512             | 3.579013 | 613             |
| Strided         | 1024            | 4.276478 | 820             |
| Gather (sorted) | 256             | 2.453685 | 37              |
| Gather (sorted) | 512             | 2.558479 | 53              |
| Gather (sorted) | 1024            | 2.647485 | 69              |
| Gather (random) | 256             | 3.106025 | 515             |
| Gather (random) | 512             | 3.363072 | 535             |
| Gather (random) | 1024            | 3.866654 | 569             |

### 64KB L1 Cache Size

| Kernel Pattern  | VPU Size (bits) | CPI      | L1 Cache Misses |
| --------------- | --------------- | -------- | --------------- |
| Unit-stride     | 256             | 2.464176 | 22              |
| Unit-stride     | 512             | 2.542392 | 30              |
| Unit-stride     | 1024            | 2.602150 | 31              |
| Strided         | 256             | 2.549792 | 95              |
| Strided         | 512             | 3.577364 | 604             |
| Strided         | 1024            | 4.289290 | 821             |
| Gather (sorted) | 256             | 2.439296 | 28              |
| Gather (sorted) | 512             | 2.537564 | 44              |
| Gather (sorted) | 1024            | 2.572118 | 61              |
| Gather (random) | 256             | 2.431806 | 2               |
| Gather (random) | 512             | 2.467215 | 2               |
| Gather (random) | 1024            | 2.524782 | 4               |
