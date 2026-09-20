// Which stream should the slot refill use?
//
// The refill path reaches ggml_backend_cuda_buffer_set_tensor, which copies on
// cudaStreamPerThread. Unless the build uses --default-stream per-thread, that is the legacy
// default stream, and the legacy default stream is not an ordinary stream: it serializes
// against every blocking stream on the device. This test measures whether a copy issued that
// way can run at the same time as a kernel issued on a non-blocking stream.
//
// Each case runs a device-DRAM-bound kernel and a 512 MiB host-to-device copy, both finite and
// of similar duration, then compares the combined wall time against max() and the sum.
//
// build: nvcc -O2 -o stream_test stream_test.cu

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <atomic>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__global__ void dram_kernel(const float4 * __restrict__ a, float4 * __restrict__ b, size_t n) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float4 v = a[i];
        v.x += 1.0f; v.y += 1.0f; v.z += 1.0f; v.w += 1.0f;
        b[i] = v;
    }
}

static const size_t DRAM_N   = 512ull*1024ull*1024ull / 16ull;
static const int    DRAM_REP = 2;
static const size_t COPY_SZ  = 512ull*1024ull*1024ull;
static const int    CHUNKS   = 8;

static int                 g_blocks = 112;
static float4 *            g_a      = nullptr;
static float4 *            g_b      = nullptr;
static void *              g_dst    = nullptr;
static void *              g_pinned = nullptr;

static void run_kernel(cudaStream_t s) {
    for (int r = 0; r < DRAM_REP; ++r) {
        dram_kernel<<<g_blocks, 256, 0, s>>>(g_a, g_b, DRAM_N);
    }
}

static void run_copy(cudaStream_t s) {
    const size_t chunk = COPY_SZ / CHUNKS;
    for (int c = 0; c < CHUNKS; ++c) {
        CUDA_CHECK(cudaMemcpyAsync((char *) g_dst + c*chunk, (char *) g_pinned + c*chunk, chunk,
                                   cudaMemcpyHostToDevice, s));
    }
}

// a dedicated host thread that issues the copy, modelling the copy thread in llama.cpp
static std::atomic<int>   g_go{0};
static std::atomic<int>   g_done{0};
static cudaStream_t       g_copy_stream = nullptr;

static void copy_thread_fn() {
    while (g_go.load() == 0) { }
    run_copy(g_copy_stream);
    g_done.store(1);
}

static void report(const char * label, cudaStream_t ks, cudaStream_t cs, bool separate_thread) {
    // kernel alone, wall clock
    auto t0 = std::chrono::steady_clock::now();
    run_kernel(ks);
    CUDA_CHECK(cudaStreamSynchronize(ks));
    auto t1 = std::chrono::steady_clock::now();
    const float k_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // copy alone, wall clock
    t0 = std::chrono::steady_clock::now();
    run_copy(cs);
    CUDA_CHECK(cudaStreamSynchronize(cs));
    t1 = std::chrono::steady_clock::now();
    const float c_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // together
    float both = 0.0f;
    if (!separate_thread) {
        t0 = std::chrono::steady_clock::now();
        run_kernel(ks);
        run_copy(cs);
        CUDA_CHECK(cudaStreamSynchronize(ks));
        CUDA_CHECK(cudaStreamSynchronize(cs));
        t1 = std::chrono::steady_clock::now();
    } else {
        g_go.store(0);
        g_done.store(0);
        std::thread t(copy_thread_fn);
        t0 = std::chrono::steady_clock::now();
        run_kernel(ks);
        g_go.store(1);
        while (g_done.load() == 0) { }
        CUDA_CHECK(cudaStreamSynchronize(ks));
        CUDA_CHECK(cudaStreamSynchronize(g_copy_stream));
        t1 = std::chrono::steady_clock::now();
        t.join();
    }
    both = std::chrono::duration<float, std::milli>(t1 - t0).count();

    const float best = k_ms > c_ms ? k_ms : c_ms;
    const bool  ok   = both < 1.30f*best;
    printf("%-52s kernel %7.2f  copy %7.2f  both %7.2f  sum %7.2f  max %7.2f  %s\n",
            label, k_ms, c_ms, both, k_ms+c_ms, best, ok ? "OVERLAP" : "SERIALIZED");
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    g_blocks = prop.multiProcessorCount * 8;

    CUDA_CHECK(cudaMalloc(&g_a, DRAM_N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g_b, DRAM_N * sizeof(float4)));
    CUDA_CHECK(cudaMemset(g_a, 1, DRAM_N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g_dst, COPY_SZ));
    CUDA_CHECK(cudaHostAlloc(&g_pinned, COPY_SZ, cudaHostAllocDefault));
    for (size_t i = 0; i < COPY_SZ; i += 4096) ((char *) g_pinned)[i] = 1;

    cudaStream_t snb1, snb2, sblk;
    CUDA_CHECK(cudaStreamCreateWithFlags(&snb1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&snb2, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreate(&sblk));

    printf("device %s, %d SMs, asyncEngineCount %d\n", prop.name, prop.multiProcessorCount, prop.asyncEngineCount);
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    printf("compiled with per-thread default stream: yes\n");
#else
    printf("compiled with per-thread default stream: no, cudaStreamPerThread is the legacy stream\n");
#endif
    printf("\n");

    // A: both on non-blocking streams (the control, this is what hw_bench used)
    report("A kernel(nonblock) + copy(nonblock)  1 thread", snb1, snb2, false);
    // B: copy on the per-thread default stream, which is what ggml_set_tensor uses
    report("B kernel(nonblock) + copy(perThread) 1 thread", snb1, cudaStreamPerThread, false);
    // C: copy on an explicit blocking stream
    report("C kernel(nonblock) + copy(blocking)  1 thread", snb1, sblk, false);
    // D: copy on the true legacy default stream
    report("D kernel(nonblock) + copy(stream 0)  1 thread", snb1, (cudaStream_t) 0, false);
    // E: copy on the per-thread stream, issued from a second host thread
    g_copy_stream = cudaStreamPerThread;
    report("E kernel(nonblock) + copy(perThread) 2 threads", snb1, cudaStreamPerThread, true);
    // F: copy on a non-blocking stream, issued from a second host thread
    g_copy_stream = snb2;
    report("F kernel(nonblock) + copy(nonblock)  2 threads", snb1, snb2, true);
    // G: copy on the true legacy default stream, issued from a second host thread
    g_copy_stream = (cudaStream_t) 0;
    report("G kernel(nonblock) + copy(stream 0)  2 threads", snb1, (cudaStream_t) 0, true);

    return 0;
}
