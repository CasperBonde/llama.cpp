// Can this GPU overlap a long-running kernel with a host-to-device copy?
//
// The prefill streaming design depends on this. It runs layer compute on one stream and
// slot refill copies on a second stream, and expects the overlap.
//
// The test launches a long kernel on stream A and a large H2D copy on stream B, then
// compares the total against each part alone:
//
//   total ~= max(kernel, copy)  -> the two overlap
//   total ~= kernel + copy      -> the two are serialized by the driver or the device
//
// It also reports the achieved copy rate while the kernel runs, which shows whether the
// link slows down under compute load.
//
// build: nvcc -O2 -o overlap_bench overlap_bench.cu

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

static float * g_out = nullptr;

// spin for a while so the kernel is long enough to overlap with
__global__ void spin_kernel(float * out, long long iters) {
    float v = 0.0f;
    long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    for (long long n = 0; n < iters; ++n) {
        v = fmaf(v, 1.000001f, 0.000001f);
        v = fmaf(v, 1.000001f, 0.000001f);
        v = fmaf(v, 1.000001f, 0.000001f);
        v = fmaf(v, 1.000001f, 0.000001f);
        v = fmaf(v, 1.000001f, 0.000001f);
        v = fmaf(v, 1.000001f, 0.000001f);
        v = fmaf(v, 1.000001f, 0.000001f);
        v = fmaf(v, 1.000001f, 0.000001f);
    }
    if (i == 0) out[0] = v;
}

static float time_kernel_alone(int blocks, int threads, long long iters, cudaStream_t s) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, s));
    spin_kernel<<<blocks, threads, 0, s>>>(g_out, iters);
    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return ms;
}

static float time_copy_alone(void * dst, void * src, size_t bytes, int n_chunks, cudaStream_t s) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    size_t chunk = bytes / n_chunks;
    CUDA_CHECK(cudaEventRecord(e0, s));
    for (int i = 0; i < n_chunks; ++i) {
        CUDA_CHECK(cudaMemcpyAsync((char *) dst + i*chunk, (char *) src + i*chunk, chunk,
                                   cudaMemcpyHostToDevice, s));
    }
    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return ms;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s, cc %d.%d, %d SMs, %d copy engines\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.asyncEngineCount);

    const size_t bytes   = 512ull*1024ull*1024ull;   // 512 MiB
    const int    n_chunks = 8;                        // 64 MiB each, like one layer

    cudaStream_t sa, sb;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking));

    void * dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMalloc(&g_out, sizeof(float)));

    void * pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault));
    for (size_t i = 0; i < bytes; i += 4096) ((char *) pinned)[i] = 1;

    // tune the spin so the kernel alone lands near the copy time
    long long iters = 400000;
    const int blocks  = prop.multiProcessorCount * 4;
    const int threads = 256;

    float k_ms = time_kernel_alone(blocks, threads, iters, sa);
    float c_ms = time_copy_alone(dst, pinned, bytes, n_chunks, sb);

    printf("\n-- separately --\n");
    printf("kernel alone                 %8.2f ms\n", k_ms);
    printf("copy alone (512 MiB, %d chunks) %8.2f ms  (%.2f GB/s)\n",
            n_chunks, c_ms, bytes/(c_ms/1000.0)/1e9);

    // overlapping: start the kernel on A, issue the copy on B, wait for both
    {
        cudaEvent_t e0, e1, done_a, done_b;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventCreate(&done_a));
        CUDA_CHECK(cudaEventCreate(&done_b));

        CUDA_CHECK(cudaEventRecord(e0, sa));
        spin_kernel<<<blocks, threads, 0, sa>>>(g_out, iters);
        CUDA_CHECK(cudaEventRecord(done_a, sa));

        size_t chunk = bytes / n_chunks;
        for (int i = 0; i < n_chunks; ++i) {
            CUDA_CHECK(cudaMemcpyAsync((char *) dst + i*chunk, (char *) pinned + i*chunk, chunk,
                                       cudaMemcpyHostToDevice, sb));
        }
        CUDA_CHECK(cudaEventRecord(done_b, sb));

        CUDA_CHECK(cudaEventSynchronize(done_a));
        CUDA_CHECK(cudaEventSynchronize(done_b));
        CUDA_CHECK(cudaEventRecord(e1, sa));
        CUDA_CHECK(cudaEventSynchronize(e1));

        float both_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&both_ms, e0, e1));

        printf("\n-- overlapped (kernel on stream A, copy on stream B) --\n");
        printf("both together                %8.2f ms\n", both_ms);
        printf("sum of the parts             %8.2f ms\n", k_ms + c_ms);
        printf("max of the parts             %8.2f ms\n", k_ms > c_ms ? k_ms : c_ms);
        printf("\n");

        if (both_ms < (k_ms + c_ms) * 0.80f) {
            printf("=> the copy IS hidden. overlap works.\n");
            printf("   copy effective rate while overlapped: %.2f GB/s\n",
                    bytes/(both_ms/1000.0)/1e9);
        } else {
            printf("=> the copy is NOT hidden. it serializes with the kernel.\n");
        }
    }

    cudaFreeHost(pinned);
    cudaFree(dst);
    return 0;
}
