# Benchmark Results

## 1. Vectorizing Kernel (Scaled Dot-Product)
- **L1 Cache Size:** 8KB
- **Metrics:** CPI, Execution Time

| VPU Size (bits) | Scalar CPI | Scalar Exec Time | Vectorized CPI | Vectorized Exec Time |
|-----------------|------------|------------------|----------------|----------------------|
| 128             | 0.775324   | 0.000110         | 2.322715       | 0.001394             |
| 256             | 0.775324   | 0.000110         | 2.325065       | 0.001097             |
| 512             | 0.775324   | 0.000110         | 2.325375       | 0.000948             |
| 1024            | 0.775324   | 0.000110         | 2.323931       | 0.000874             |
| 2048            | 0.775324   | 0.000110         | 2.323034       | 0.000838             |
| 4096            | 0.775324   | 0.000110         | 2.324310       | 0.000841             |

## 2. Stencil Kernel (Heat Diffusion)
- **Metrics:** CPI, L1 Cache Misses

### 8KB L1 Cache Size
| VPU Size (bits) | Scalar CPI | Scalar L1 Misses | Vectorized CPI | Vectorized L1 Misses |
|-----------------|------------|------------------|----------------|----------------------|
| 128             |            |                  |                |                      |
| 256             |            |                  |                |                      |
| 512             |            |                  |                |                      |
| 1024            |            |                  |                |                      |
| 2048            |            |                  |                |                      |
| 4096            |            |                  |                |                      |

### 64KB L1 Cache Size
| VPU Size (bits) | Scalar CPI | Scalar L1 Misses | Vectorized CPI | Vectorized L1 Misses |
|-----------------|------------|------------------|----------------|----------------------|
| 128             |            |                  |                |                      |
| 256             |            |                  |                |                      |
| 512             |            |                  |                |                      |
| 1024            |            |                  |                |                      |
| 2048            |            |                  |                |                      |
| 4096            |            |                  |                |                      |

## 3. Sparse Matrix-Vector Multiplication (SpMV)
- **Metrics:** CPI, L1 Cache Misses

### 8KB L1 Cache Size
| Kernel Pattern   | VPU Size (bits) | CPI | L1 Cache Misses |
|------------------|-----------------|-----|-----------------|
| Unit-stride      | 256             |     |                 |
| Unit-stride      | 512             |     |                 |
| Unit-stride      | 1024            |     |                 |
| Strided          | 256             |     |                 |
| Strided          | 512             |     |                 |
| Strided          | 1024            |     |                 |
| Gather (sorted)  | 256             |     |                 |
| Gather (sorted)  | 512             |     |                 |
| Gather (sorted)  | 1024            |     |                 |
| Gather (random)  | 256             |     |                 |
| Gather (random)  | 512             |     |                 |
| Gather (random)  | 1024            |     |                 |

### 64KB L1 Cache Size
| Kernel Pattern   | VPU Size (bits) | CPI | L1 Cache Misses |
|------------------|-----------------|-----|-----------------|
| Unit-stride      | 256             |     |                 |
| Unit-stride      | 512             |     |                 |
| Unit-stride      | 1024            |     |                 |
| Strided          | 256             |     |                 |
| Strided          | 512             |     |                 |
| Strided          | 1024            |     |                 |
| Gather (sorted)  | 256             |     |                 |
| Gather (sorted)  | 512             |     |                 |
| Gather (sorted)  | 1024            |     |                 |
| Gather (random)  | 256             |     |                 |
| Gather (random)  | 512             |     |                 |
| Gather (random)  | 1024            |     |                 |
