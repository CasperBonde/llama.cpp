// Does a host-to-device copy from PAGEABLE memory overlap with kernel execution?
//
// The slot refill reads the model weights, which live in a plain malloc'd buffer
// (ggml_aligned_malloc -> posix_memalign). That is pageable memory, not pinned. The CUDA
// programming guide says pageable copies are staged through a driver bounce buffer and, for
// host-to-device, are not guaranteed to make progress independently of kernel execution.
//
// Every earlier micro-benchmark on this machine used cudaHostAlloc (pinned) source memory and
// overlapped perfectly. This test runs the same overlap with both kinds of source, so the two
// can be compared directly.
//
// build: nvcc -O2 -std=c++17 -o pageable_test pageable_test.cu

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

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

static const size_t DRAM_N  = 512ull*1024ull*1024ull / 16ull;
static const int    DRAM_REP = 2;
static const size_t COPY_SZ = 512ull*1024ull*1024ull;
static const int    CHUNKS  = 64;   // ~8 MiB each, like one layer's tensors combined

static float4 * g_a = nullptr;
static float4 * g_b = nullptr;
static void *   g_dst = nullptr;
static cudaStream_t g_ks = nullptr;
static cudaStream_t g_cs = nullptr;
static int      g_blocks = 112;

static float wall_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static void run_kernel() {
    for (int r = 0; r < DRAM_REP; ++r) {
        dram_kernel<<<g_blocks, 256, 0, g_ks>>>(g_a, g_b, DRAM_N);
    }
}

static void run_copy(void * src) {
    const size_t chunk = COPY_SZ / CHUNKS;
    for (int c = 0; c < CHUNKS; ++c) {
        CUDA_CHECK(cudaMemcpyAsync((char *) g_dst + c*chunk, (char *) src + c*chunk, chunk,
                                   cudaMemcpyHostToDevice, g_cs));
    }
}

static void test(const char * label, void * src) {
    auto t0 = std::chrono::steady_clock::now();
    run_kernel();
    CUDA_CHECK(cudaStreamSynchronize(g_ks));
    const float k_ms = wall_ms(t0);

    t0 = std::chrono::steady_clock::now();
    run_copy(src);
    CUDA_CHECK(cudaStreamSynchronize(g_cs));
    const float c_ms = wall_ms(t0);

    // both, issued back to back and without a host-side wait in between
    t0 = std::chrono::steady_clock::now();
    run_kernel();
    run_copy(src);
    CUDA_CHECK(cudaStreamSynchronize(g_ks));
    CUDA_CHECK(cudaStreamSynchronize(g_cs));
    const float both = wall_ms(t0);

    const float best = k_ms > c_ms ? k_ms : c_ms;
    printf("%-28s kernel %7.2f  copy %7.2f  both %7.2f  sum %7.2f  max %7.2f  %s\n",
            label, k_ms, c_ms, both, k_ms + c_ms, best,
            both < 1.30f*best ? "OVERLAP" : "SERIALIZED");
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    g_blocks = prop.multiProcessorCount * 8;

    CUDA_CHECK(cudaMalloc(&g_a, DRAM_N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g_b, DRAM_N * sizeof(float4)));
    CUDA_CHECK(cudaMemset(g_a, 1, DRAM_N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&g_dst, COPY_SZ));

    CUDA_CHECK(cudaStreamCreateWithFlags(&g_ks, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_cs, cudaStreamNonBlocking));

    // pinned source
    void * pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&pinned, COPY_SZ, cudaHostAllocDefault));
    for (size_t i = 0; i < COPY_SZ; i += 4096) ((char *) pinned)[i] = 1;

    // pageable source, the same allocation ggml uses for host weights
    void * pageable = nullptr;
    CUDA_CHECK(cudaMallocHost(&pageable, 1));  // keep the runtime happy
    pageable = aligned_alloc(64, COPY_SZ);
    if (!pageable) { printf("aligned_alloc failed\n"); return 1; }
    for (size_t i = 0; i < COPY_SZ; i += 4096) ((char *) pageable)[i] = 1;

    printf("device %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    printf("source memory for the copy:\n\n");
    test("pinned  (cudaHostAlloc)", pinned);
    test("pageable (posix_memalign)", pageable);
    test("pinned  (cudaHostAlloc)", pinned);
    test("pageable (posix_memalign)", pageable);

    return 0;
}
