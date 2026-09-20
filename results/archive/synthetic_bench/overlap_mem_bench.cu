// Does an H2D copy overlap a MEMORY-BANDWIDTH-BOUND kernel?
//
// overlap_bench.cu used an ALU-only spin kernel (fmaf chain, no memory traffic) and showed
// full overlap. A real prefill GEMM is memory bound, so the copy engine and the SMs would
// contend on the same memory subsystem. This test uses a kernel that saturates device
// memory bandwidth, which is the honest comparison.
//
//   both ~= max(kernel, copy)  -> overlap still works, the problem is in llama.cpp
//   both ~= kernel + copy      -> the copy cannot overlap real compute on this GPU
//
// build: nvcc -O2 -o overlap_mem_bench overlap_mem_bench.cu

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
        exit(1); \
    } } while (0)

// memory bandwidth bound: stream through a large buffer
__global__ void mem_kernel(const float4 * __restrict__ in, float4 * __restrict__ out,
                           size_t n4, int reps) {
    const size_t i      = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x*blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int r = 0; r < reps; ++r) {
        for (size_t j = i; j < n4; j += stride) {
            const float4 v = in[j];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    if (i == 0) out[0] = acc;
}

static float ms_between(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s, %d SMs, %d copy engines\n",
            prop.name, prop.multiProcessorCount, prop.asyncEngineCount);

    const size_t kb_bytes = 256ull*1024*1024;   // kernel working set
    const size_t cp_bytes = 512ull*1024*1024;   // copy size, ~ one big layer
    const int    reps     = 48;                 // tuned so the kernel is long enough to hide the copy

    float4 * kb_in = nullptr, * kb_out = nullptr;
    CK(cudaMalloc(&kb_in,  kb_bytes));
    CK(cudaMalloc(&kb_out, 16));
    CK(cudaMemset(kb_in, 1, kb_bytes));

    void * cp_dst = nullptr;
    CK(cudaMalloc(&cp_dst, cp_bytes));
    void * cp_src = nullptr;
    CK(cudaHostAlloc(&cp_src, cp_bytes, cudaHostAllocDefault));
    for (size_t i = 0; i < cp_bytes; i += 4096) ((char *) cp_src)[i] = 1;

    cudaStream_t sa = nullptr, sb = nullptr;
    CK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));

    const int blocks = prop.multiProcessorCount * 16;
    const int threads = 256;
    const size_t n4 = kb_bytes/16;

    // ---- kernel alone
    float k_ms;
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, sa));
        mem_kernel<<<blocks, threads, 0, sa>>>(kb_in, kb_out, n4, reps);
        CK(cudaEventRecord(e1, sa));
        CK(cudaEventSynchronize(e1));
        k_ms = ms_between(e0, e1);
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }

    // ---- copy alone
    float c_ms;
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, sb));
        CK(cudaMemcpyAsync(cp_dst, cp_src, cp_bytes, cudaMemcpyHostToDevice, sb));
        CK(cudaEventRecord(e1, sb));
        CK(cudaEventSynchronize(e1));
        c_ms = ms_between(e0, e1);
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }

    printf("\n-- separately --\n");
    printf("kernel alone (256 MiB x %d reps)   %8.2f ms  (%.1f GiB/s device)\n",
            reps, k_ms, (double) kb_bytes*reps/(k_ms/1000.0)/(1024.0*1024*1024));
    printf("copy alone   (512 MiB H2D)         %8.2f ms  (%.2f GB/s)\n",
            c_ms, cp_bytes/(c_ms/1000.0)/1e9);

    // ---- both at once
    {
        cudaEvent_t g0, gk, gc;
        CK(cudaEventCreate(&g0));
        CK(cudaEventCreate(&gk));
        CK(cudaEventCreate(&gc));

        CK(cudaEventRecord(g0, sa));
        mem_kernel<<<blocks, threads, 0, sa>>>(kb_in, kb_out, n4, reps);
        CK(cudaEventRecord(gk, sa));

        CK(cudaEventRecord(g0, sb));     // same instant, second stream
        CK(cudaMemcpyAsync(cp_dst, cp_src, cp_bytes, cudaMemcpyHostToDevice, sb));
        CK(cudaEventRecord(gc, sb));

        CK(cudaEventSynchronize(gk));
        CK(cudaEventSynchronize(gc));

        const float both_k = ms_between(g0, gk);
        const float both_c = ms_between(g0, gc);
        const float both   = both_k > both_c ? both_k : both_c;

        printf("\n-- overlapped (kernel on stream A, H2D copy on stream B) --\n");
        printf("kernel finished at                 %8.2f ms\n", both_k);
        printf("copy finished at                   %8.2f ms\n", both_c);
        printf("total                              %8.2f ms\n", both);
        printf("sum if serialized                  %8.2f ms\n", k_ms + c_ms);
        printf("max if fully overlapped            %8.2f ms\n", k_ms > c_ms ? k_ms : c_ms);
        printf("copy rate while overlapped           %8.2f GB/s\n",
                cp_bytes/(both_c/1000.0)/1e9);
        printf("\n");

        const float serial = k_ms + c_ms;
        if (both < serial*0.80f) {
            printf("=> overlap WORKS even against a memory bound kernel.\n");
            printf("   the serialization is in the llama.cpp submission/event pattern.\n");
        } else {
            printf("=> the copy CANNOT overlap real compute on this GPU.\n");
            printf("   the copy engine serializes with memory bound kernels.\n");
        }

        CK(cudaEventDestroy(g0)); CK(cudaEventDestroy(gk)); CK(cudaEventDestroy(gc));
    }

    CK(cudaFreeHost(cp_src));
    CK(cudaFree(cp_dst));
    CK(cudaFree(kb_in));
    CK(cudaFree(kb_out));
    return 0;
}
