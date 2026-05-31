---
title: "Writing Your First CUDA Kernels: GPU Execution, Memory, and Optimization"
date: 2026-05-31
description: "A complete walkthrough of CUDA programming from the ground up: thread indexing, vector addition, tiled matrix multiplication with shared memory, atomics, and async streams. Built around an RTX 5050 and CUDA 12.6."
authors:
  - name: Luan Costa
---

Modern deep learning, physics simulations, and scientific computing are built on one insight: some problems are embarrassingly parallel, and the GPU is the hardware that was built to exploit that. But to actually write GPU code, you need to understand a programming model that is fundamentally different from anything on the CPU.

This post is a complete walkthrough of CUDA from scratch. We'll cover the GPU execution model, thread indexing in 1D/2D/3D, the standard memory management pattern, vector addition and matrix multiplication kernels, shared memory tiling, race conditions and atomics, and finally concurrent execution with streams. Every concept is grounded in working code.

The examples were written and tested on an **NVIDIA GeForce RTX 5050** with **CUDA 12.6**.

---

## The GPU Execution Model: Why It's Different

Before touching a single line of CUDA code, it's worth understanding why the GPU thinks the way it does. The CPU and GPU are designed around opposite philosophies:

| | CPU | GPU |
|---|---|---|
| Cores | Few (8–32) | Thousands (RTX 5050: ~2,560) |
| Strategy | Low latency per thread | High throughput across many threads |
| Thread context | Heavy (full state save on context switch) | Extremely light (switch in 1 cycle) |
| Ideal workload | Serial logic, complex branches | Massively parallel, uniform operations |

The CPU is optimized to run a small number of threads as fast as possible. The GPU takes the opposite bet: run an enormous number of threads simultaneously, and hide the latency of slow memory operations by switching to a different thread while one is waiting for data. This is called **latency hiding through occupancy**, and it only works if you give the GPU enough threads to keep busy.

This is why CUDA kernels launch millions of threads. The GPU isn't just providing parallelism; it needs the volume of work to function efficiently.

### The Grid → Block → Thread Hierarchy

Every CUDA kernel launch creates a three-level hierarchy of threads:

```
┌─────────────────────────────────────────────┐
│                    GRID                      │
│  (all blocks from one kernel launch)         │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Block   │  │  Block   │  │  Block   │   │
│  │ (0,0,0)  │  │ (1,0,0)  │  │ (2,0,0)  │   │
│  │          │  │          │  │          │   │
│  │ T T T T  │  │ T T T T  │  │ T T T T  │   │
│  │ T T T T  │  │ T T T T  │  │ T T T T  │   │
│  └──────────┘  └──────────┘  └──────────┘   │
│                                              │
└─────────────────────────────────────────────┘
```

- **Thread**: the smallest unit of execution. Each thread runs the kernel code independently, and crucially, each thread has a unique combination of `threadIdx` and `blockIdx` that lets it determine which piece of data to work on.
- **Block**: a group of threads that share **shared memory** (`__shared__`) and can synchronize with `__syncthreads()`. An entire block runs on a single **Streaming Multiprocessor (SM)**: a block never splits across SMs.
- **Grid**: all the blocks in one kernel launch. Blocks in different parts of the grid do **not** communicate with each other except through global memory.

### Streaming Multiprocessors (SMs)

Each SM is an independent processing unit inside the GPU. The RTX 5050 has multiple SMs, each capable of running dozens of blocks simultaneously. The GPU's scheduler distributes blocks across available SMs automatically; you don't control this assignment, and that's by design.

The independence between blocks is what makes CUDA scalable: the same kernel binary runs equally well on a GPU with 10 SMs or 1,000 SMs. More SMs means more blocks run simultaneously, but the result is identical.

### Warps: The Real Unit of Execution

Inside a block, threads are grouped into **warps of 32 threads**. All threads in a warp execute the **same instruction at the same time**: this is the SIMT (Single Instruction, Multiple Threads) model.

This has an important consequence: if threads in a warp diverge at an `if/else`, the GPU executes *both branches sequentially*, masking off the threads that don't belong to each branch. This is called **warp divergence** and is a common source of inefficiency.

The practical rule: **always choose block sizes that are multiples of 32** (256, 512, or 1024 are typical choices). Anything else leaves partial warps that waste execution slots.

---

## Thread Indexing

For a thread to know which element to process, it needs to compute a globally unique ID. CUDA provides four built-in variables for this:

| Variable | Meaning |
|---|---|
| `threadIdx.{x,y,z}` | Thread's position within its block |
| `blockIdx.{x,y,z}` | Block's position within the grid |
| `blockDim.{x,y,z}` | Size of a block in each dimension |
| `gridDim.{x,y,z}` | Size of the grid in each dimension |

### The `whoami` Kernel

The classic way to understand indexing is the `whoami` kernel, which computes every thread's global ID and prints it:

```c
__global__ void whoami(void) {
    // Step 1: compute the LINEAR block ID in the 3D grid
    int block_id =
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;

    // Step 2: how many threads came before this block?
    int block_offset =
        block_id * blockDim.x * blockDim.y * blockDim.z;

    // Step 3: compute the LINEAR thread ID within the block
    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    // Step 4: global ID = threads before me + my local position
    int id = block_offset + thread_offset;
}
```

The math in Step 1 is exactly what C uses to linearize a 3D array index `(x, y, z)` into a flat position: `x + y*dimX + z*dimX*dimY`. You're doing the same thing: finding a unique flat index from a 3D coordinate.

The launch configuration for this kernel:

```c
const int b_x = 2, b_y = 3, b_z = 4;   // grid: 2×3×4 = 24 blocks
const int t_x = 4, t_y = 4, t_z = 4;   // block: 4×4×4 = 64 threads per block

// Total: 24 × 64 = 1,536 threads running in parallel
dim3 blocksPerGrid(b_x, b_y, b_z);
dim3 threadsPerBlock(t_x, t_y, t_z);
whoami<<<blocksPerGrid, threadsPerBlock>>>();
```

The `<<<blocksPerGrid, threadsPerBlock>>>` syntax is the **launch configuration**: this is where you tell the GPU how many threads to create.

**Try this:** run the kernel and pipe the output through `sort`:
```bash
nvcc -o idxing 01_idxing.cu
./idxing | sort -n | head -20   # sorted by ID
./idxing | head -20             # actual execution order
```

Here is what the first few lines look like on the RTX 5050. First, raw output as the GPU schedules it:

```
24 blocks/grid
64 threads/block
1536 total threads
0256 | Block(0 2 0) =   4 | Thread(0 0 0) =   0
0257 | Block(0 2 0) =   4 | Thread(1 0 0) =   1
0258 | Block(0 2 0) =   4 | Thread(2 0 0) =   2
0263 | Block(0 2 0) =   4 | Thread(3 1 0) =   7
0260 | Block(0 2 0) =   4 | Thread(0 1 0) =   4
```

The first block to actually print is block `(0,2,0)` — not `(0,0,0)`. After sorting by global ID:

```
0000 | Block(0 0 0) =   0 | Thread(0 0 0) =   0
0001 | Block(0 0 0) =   0 | Thread(1 0 0) =   1
0002 | Block(0 0 0) =   0 | Thread(2 0 0) =   2
0003 | Block(0 0 0) =   0 | Thread(3 0 0) =   3
0004 | Block(0 0 0) =   0 | Thread(0 1 0) =   4
```

The GPU scheduler assigned blocks to SMs based on availability, not launch order. This is intentional and expected.

### 1D Indexing: The Everyday Case

In practice, most kernels use 1D indexing:

```c
__global__ void kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = ...; // process element i
    }
}
```

The `if (i < n)` is called a **guard** and it's essential. When `n` is not divisible by `blockDim.x`, the last block has threads that would index beyond the end of the array. The guard prevents out-of-bounds access.

---

## The CUDA Memory Management Pattern

Every CUDA program follows the same skeleton for moving data between the CPU and GPU:

```
CPU (Host)                          GPU (Device)
─────────                          ────────────
1. Allocate host memory
2. Initialize data
3. cudaMalloc() ──────────────────► Allocate VRAM
4. cudaMemcpy(H→D) ───────────────► Copy data to VRAM
5. kernel<<<grid, block>>>() ─────► Execute kernel
6. cudaMemcpy(D→H) ◄──────────────  Copy result back
7. Use the results
8. free() + cudaFree()
```

This pattern appears in every example that follows. The key abstraction is that the GPU has its own memory (VRAM) that is separate from RAM, and all data must be explicitly transferred between them before and after a kernel runs.

---

## Vector Addition: The "Hello World" of GPU Computing

Adding two vectors element-by-element is the simplest parallel kernel; every element is independent, so every thread can work on its own pair without ever talking to another thread.

### CPU Reference

```c
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

The CPU processes 10 million additions serially. Even with SIMD auto-vectorization, this is fundamentally sequential at the loop level.

### GPU Kernel

```c
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

With `BLOCK_SIZE = 256` and `N = 10,000,000`:

```
num_blocks = (10,000,000 + 256 - 1) / 256 = 39,063 blocks
```

The GPU creates ~10 million threads. Each adds a single pair of elements.

### Ceiling Division

The formula `(N + BLOCK_SIZE - 1) / BLOCK_SIZE` is integer **ceiling division**: rounding up instead of down:

```
N = 1,000, BLOCK_SIZE = 256
Without rounding up: 1000 / 256 = 3 blocks → covers 768 elements (missing 232!)
With rounding up:  (1000 + 255) / 256 = 4 blocks → covers 1,024 elements
                                        ↑ guard protects the extra 24
```

### Warm-up Runs

When benchmarking GPU code, always run a few **warm-up iterations** before measuring time. The first kernel execution has overhead from:
- JIT compilation of GPU code
- The GPU ramping up from a low-power P-state to full clock speed
- Cold caches causing artificially high latency

```bash
nvcc -O2 -o vec_add 00_vector_add_v1.cu
./vec_add
```

Running this on the RTX 5050 with `N = 10,000,000` elements and 10 timed runs:

```
CPU average time:  4.360 ms
GPU average time:  0.577 ms
Speedup:           7.55x
Results are correct
```

The GPU wins by 7.55x on a simple element-wise addition — and this is one of the less impressive cases for the GPU, since memory bandwidth (moving 120 MB of data) is the bottleneck here, not arithmetic. For compute-heavy kernels the gap is much larger.

**1D vs 3D grid** on the same 10M-element problem:

```
GPU 1D average time:  0.634 ms  (6.43x vs CPU)
GPU 3D average time:  0.776 ms  (5.25x vs CPU)
GPU 1D vs GPU 3D:     1D is 1.22x faster
```

The 3D kernel is slower purely because it does more index arithmetic per thread (3 multiplications instead of 1). The extra compute overhead is measurable even for a trivially cheap operation like addition.

### A Note on 3D Grids for 1D Problems

It's also valid to organize threads in a 3D grid even for a 1D problem; the kernel just linearizes the 3D index back to a flat array index:

```c
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        c[idx] = a[idx] + b[idx];
    }
}
```

This pattern becomes natural when working with actual 3D data (volumetric simulations, 3D images, tensor operations). For a simple vector problem, the 1D version is faster because the 3D version does more arithmetic per thread to compute the index.

---

## Matrix Multiplication: The Most Important Kernel

Matrix multiplication is the single most important kernel in modern computing. It underpins neural network training, computer vision, and scientific simulation. Understanding its performance characteristics is foundational.

For `C = A × B` where `A` is `(M × K)` and `B` is `(K × N)`:

$$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]$$

### Naive GPU Kernel

The straightforward approach: one thread per output element.

```c
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}
```

The grid is **2D** here because the output matrix is 2D. The block is typically `32×32 = 1,024 threads` (the hardware maximum per block on modern NVIDIA GPUs).

Launch configuration for `M×K × K×N`:
```c
dim3 block(32, 32);
dim3 grid((N + 31) / 32, (M + 31) / 32);
matmul_gpu<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
```

### The Bottleneck: Global Memory Bandwidth

The naive kernel works, but it's memory-bound. For a `1024×1024` matrix:
- Each thread makes `K = 1,024` reads from global memory for A and 1,024 reads for B
- There are `1,024 × 1,024 = 1,048,576` threads
- **Total global memory reads: ~2 billion**

Worse, the reads are redundant. Thread `(0,0)` and thread `(0,1)` both read the entire row `A[0][0..K-1]`. That row is read **N times**, once per column of C. The same data is fetched from VRAM over and over.

This is not a compute bottleneck; it's a **memory bandwidth bottleneck**. The GPU's arithmetic units are idle, waiting for data to arrive from VRAM.

Running the naive kernel against a CPU baseline on a `256×256 × 256×256` matrix (`K=256`):

```
CPU average time:  4388 µs
GPU average time:   115 µs
Speedup:           38x
```

38x is a big number, but the naive GPU kernel is far from optimal — it's leaving most of the hardware performance on the table.

---

## Optimizing with Shared Memory

### The GPU Memory Hierarchy

Before the optimization, understand where data lives:

```
┌──────────────────────────────────────────────────┐
│  Registers (per thread)                           │
│  Latency: ~1 cycle | Size: ~255 per thread        │
├──────────────────────────────────────────────────┤
│  Shared Memory / L1 Cache (per block)             │
│  Latency: ~5–30 cycles | Size: 48–96 KB/SM        │
├──────────────────────────────────────────────────┤
│  L2 Cache (per GPU)                               │
│  Latency: ~100 cycles | Size: several MB          │
├──────────────────────────────────────────────────┤
│  Global Memory / VRAM                             │
│  Latency: ~200–800 cycles | Size: 8 GB            │
└──────────────────────────────────────────────────┘
```

Accessing global memory is **200–800× slower** than registers. The core optimization strategy in CUDA is to minimize global memory traffic by using shared memory as a manually managed cache.

### Tiled Matrix Multiplication

The key insight: instead of each thread fetching its data independently from VRAM, the whole block cooperates to load a **tile** of A and a tile of B into shared memory, then all threads compute using that fast local copy.

```
Matrix A (M×K)              Matrix B (K×N)
┌─────────────────┐        ┌─────────────────┐
│ tile │ tile │..│        │tile│tile│tile│..│
│  A0  │  A1  │  │        │ B0 │ B1 │ B2 │  │
└──────┴──────┴──┘        └────┴────┴────┴──┘

To compute one tile of C:
C_tile += A_tile0 × B_tile0  +  A_tile1 × B_tile1  + ...
            ↑ loaded into shared memory, one tile at a time
```

Here's the full tiled kernel:

```c
#define TILE_SIZE 16

__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;  // lives in a register - fast!

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {

        // === PHASE 1: Collaborative load into shared memory ===
        // Each thread loads ONE element of A and ONE of B
        if (row < M && tile * TILE_SIZE + tx < K)
            sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;

        if (col < N && tile * TILE_SIZE + ty < K)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            sharedB[ty][tx] = 0.0f;

        // === BARRIER: all threads must finish loading before computing ===
        __syncthreads();

        // === PHASE 2: Compute using fast shared memory ===
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sharedA[ty][k] * sharedB[k][tx];

        // === BARRIER: all threads must finish computing before next tile ===
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### Why Two `__syncthreads()`?

Each barrier serves a distinct purpose:

**First barrier**: after loading, before computing. Without it, a fast thread could start reading `sharedA[ty][k]` before a slow thread has written that value. You'd be reading garbage.

**Second barrier**: after computing, before the next tile's load. Without it, a fast thread could start overwriting shared memory with the next tile while a slow thread is still reading the current tile's values.

The timing diagram makes the failure mode concrete:

```
Thread A (fast)         Thread B (slow)
────────────────        ─────────────────
loads sharedA[0][0]     ... still computing last tile ...
                          ← __syncthreads() ← Thread A waits here
                        loads sharedA[0][0]  (would overwrite!)
computes ...            computes ...
```

### The Bandwidth Reduction

With `TILE_SIZE = 16`, one block of 16×16 threads loads:
- 256 floats of A (one tile)
- 256 floats of B (one tile)

Each float in shared memory is read **16 times** (once per thread in the other dimension), instead of once. The number of global memory accesses drops from `2K` reads per thread to `2K/TILE_SIZE`; a **16× reduction in VRAM traffic**.

You can measure this directly with NVIDIA's profiler:

```bash
# Measure global memory bytes read
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum ./naive_matmul
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum ./tiled_matmul
```

Here are the measured results on the RTX 5050 for a `1024×1024` matrix, averaged over 20 runs:

```
Kernel                    Time (ms)    TFLOPS
Naive (16×16 blocks)        2.977      0.721
Tiled (TILE_SIZE=16)        2.088      1.029

Speedup: 1.43x
```

The tiled kernel delivers 1.43x the throughput and crosses the 1 TFLOP/s mark on a matrix that fits in a single kernel launch. The speedup grows with matrix size — at 1024×1024 the L2 cache already helps the naive kernel somewhat; at larger sizes where data no longer fits in cache, the tiled advantage widens further.

---

## Atomics and Race Conditions

### The Problem

When multiple threads write to the same memory location simultaneously, the result is undefined. This is a **race condition**.

Consider a kernel that increments a shared counter:

```c
__global__ void incrementNonAtomic(int *counter) {
    int old = *counter;        // read current value
    int new_val = old + 1;     // increment locally
    *counter = new_val;        // write back
}
```

These three operations look sequential but they're not atomic on hardware. With 1,000,000 threads running simultaneously:

```
Time   Thread A           Thread B
──────────────────────────────────────
  1    reads counter = 42  reads counter = 42
  2    computes 43          computes 43
  3    writes 43            writes 43

Two increments happened, but counter only went up by 1.
This is the "lost update problem."
```

Run this with `NUM_BLOCKS = 1000` and `NUM_THREADS = 1000`. Here is the actual output from 5 consecutive runs on the RTX 5050:

```
Non-atomic counter value: 151
Atomic counter value:     1000000

Non-atomic counter value: 148
Atomic counter value:     1000000

Non-atomic counter value: 148
Atomic counter value:     1000000

Non-atomic counter value: 148
Atomic counter value:     1000000

Non-atomic counter value: 148
Atomic counter value:     1000000
```

Expected: 1,000,000. The non-atomic version returns ~148-151 — it lost 99.98% of the increments to race conditions. The number is also not entirely random: it tends to stabilize around a low value because the GPU schedules all warps in tight lockstep, so most threads read the same stale value before any of them write back.

### The Fix: `atomicAdd`

```c
__global__ void incrementAtomic(int *counter) {
    atomicAdd(counter, 1);
}
```

`atomicAdd` is a hardware instruction that performs read-modify-write as an indivisible unit. If 32 threads in a warp all try to `atomicAdd` the same address, the hardware serializes them: all 32 increments happen, one at a time, and every update is preserved.

### The Full Atomic Toolkit

| Function | Operation |
|---|---|
| `atomicAdd(addr, val)` | `*addr += val` |
| `atomicSub(addr, val)` | `*addr -= val` |
| `atomicExch(addr, val)` | swap: returns old, writes `val` |
| `atomicMin(addr, val)` | `*addr = min(*addr, val)` |
| `atomicMax(addr, val)` | `*addr = max(*addr, val)` |
| `atomicCAS(addr, cmp, val)` | Compare-And-Swap: if `*addr == cmp`, write `val` |
| `atomicAnd/Or/Xor` | bitwise operations |

### When Atomics Hurt Performance

Atomics serialize conflicting accesses. If many threads compete for the same address (**high contention**), performance degrades severely; threads queue up waiting for their turn.

The efficient pattern for reductions is to reduce within the block first using shared memory, then use a single atomic per block to combine with the global result:

```c
__global__ void reduceSum(float *data, float *result, int n) {
    __shared__ float partial[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partial[threadIdx.x] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    // Tree reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    // Only thread 0 of each block touches global memory
    if (threadIdx.x == 0)
        atomicAdd(result, partial[0]);
}
```

This reduces the number of atomic operations from `N` (one per thread) to `num_blocks` (one per block). For a million elements with block size 256, that's a reduction from 1,000,000 atomic ops to ~3,906.

---

## Streams and Asynchronous Execution

### What is a CUDA Stream?

A **CUDA stream** is a queue of GPU operations that execute in order within the stream, but streams can execute **in parallel with each other**. This lets you overlap:
- Host-to-device memory transfers
- Device-to-host memory transfers
- Kernel execution

Without streams, everything goes through the default stream and executes sequentially:

```
Without streams:
──────────────────────────────────────────────────►
[  H2D copy  ] [  kernel  ] [  D2H copy  ]

With streams:
stream1: [  H2D copy A  ] [  kernel A  ] [  D2H copy A  ]
stream2:      [  H2D copy B  ] [  kernel B  ] [  D2H copy B  ]
         ──────────────────────────────────────────────────►
         total time is reduced!
```

### Pinned Memory Requirement

Asynchronous transfers require **pinned (page-locked) memory** on the host. Normal `malloc` memory can be moved by the OS at any time, which breaks DMA transfers. Pinned memory is locked in place:

```c
// Normal paginable memory - async transfers not guaranteed
float *h_data = (float*)malloc(size);

// Pinned memory - true async DMA
float *h_data;
cudaMallocHost(&h_data, size);
// ... use h_data ...
cudaFreeHost(h_data);
```

The trade-off: pinned memory cannot be swapped, so excessive use can hurt overall system performance. Use it only for transfer buffers where async behavior actually matters.

### Stream Basics

```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Copy A and B in parallel - different streams!
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);

// Must wait for d_B before launching a kernel that uses it
cudaStreamSynchronize(stream2);

// Kernel in stream1
vectorAdd<<<blocks, threads, 0, stream1>>>(d_A, d_B, d_C, n);

// Copy result back
cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

The `cudaStreamSynchronize(stream2)` before the kernel launch is necessary because the kernel uses `d_B`, which is being copied in stream2. Different streams have no implicit ordering dependency; if you don't explicitly synchronize, the kernel can start before the copy finishes.

### Events: Fine-Grained Cross-Stream Synchronization

Events let you express more precise dependencies without blocking the CPU:

```c
cudaEvent_t event;
cudaEventCreate(&event);

// Stream1 does its work
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1);
kernel1<<<grid, block, 0, stream1>>>(d_data, N);

// Mark the point in stream1 where the event fires
cudaEventRecord(event, stream1);

// Stream2 waits until stream1 reaches the event - GPU manages this
cudaStreamWaitEvent(stream2, event, 0);

// Safe to use data produced by stream1
kernel2<<<grid, block, 0, stream2>>>(d_data, N);
```

`cudaStreamWaitEvent` is more efficient than `cudaStreamSynchronize` because the CPU doesn't block; the GPU scheduler manages the dependency internally.

### Stream Priorities

For workloads where some operations are more time-sensitive than others:

```c
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

cudaStreamCreateWithPriority(&high_stream, cudaStreamNonBlocking, greatestPriority);
cudaStreamCreateWithPriority(&low_stream, cudaStreamNonBlocking, leastPriority);
```

Under load, the GPU scheduler prefers operations in higher-priority streams. This is useful for guaranteeing that latency-sensitive work (real-time frame decoding, interactive inference) isn't delayed by batch processing.

`cudaStreamNonBlocking` means the stream doesn't block and isn't blocked by the default stream (stream 0).

### Stream Callbacks: GPU Notifying the CPU

```c
void CUDART_CB onComplete(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU work complete, notifying CPU\n");
    // wake a CPU thread, signal a semaphore, enqueue more work, etc.
}

cudaStreamAddCallback(stream, onComplete, NULL, 0);
```

The callback executes on the CPU when the GPU reaches that point in the stream. This enables producer-consumer pipelines where the CPU needs to know when a batch of GPU work is done so it can start preparing the next batch.

---

## Memory Hierarchy Reference

| Type | Scope | Latency | Declaration | Use Case |
|---|---|---|---|---|
| Registers | per thread | ~1 cycle | local variables | loop vars, accumulators |
| Shared Memory | per block | ~5–30 cycles | `__shared__ float arr[N]` | tiles, inter-thread communication within block |
| Global Memory | whole GPU | ~200–800 cycles | `cudaMalloc` | main data, input/output |
| Constant Memory | whole GPU (read-only) | ~5 cycles (cached) | `__constant__ float arr[N]` | fixed params (filter weights, etc.) |
| Texture Memory | whole GPU (read-only) | ~5 cycles (cached) | `cudaBindTexture` | data with 2D spatial locality |
| Pinned Memory | host | N/A | `cudaMallocHost` | async transfer buffers |

---

## Where to Go Next

This covers the foundational layer of CUDA programming. The natural next steps:

**cuBLAS**: NVIDIA's production-grade linear algebra library. Run `cublasSgemm` on the same matrices and compare against your tiled kernel; it will be noticeably faster because it uses all the hardware tricks (vectorized loads, tensor cores, bank conflict avoidance) that take months to master.

**Occupancy**: learn to calculate SM occupancy (the ratio of active warps to the theoretical maximum) and how it relates to throughput. Shared memory usage, register count, and block size all affect occupancy, and optimizing for it is often the lever that closes the gap to cuBLAS.

**Coalesced Memory Access**: the access patterns that maximize global memory bandwidth. The canonical example is matrix transposition: a naive implementation reads rows (coalesced) but writes columns (strided, slow). Understanding coalescing is essential before optimizing anything memory-bound.

**Triton**: a Python-embedded language for writing GPU kernels at a higher level of abstraction, which handles tiling and shared memory automatically. Triton is what most ML practitioners reach for today. But it generates CUDA under the hood, and understanding what's in this post is the prerequisite for knowing whether Triton's output is any good.
