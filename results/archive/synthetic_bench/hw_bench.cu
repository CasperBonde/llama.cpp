// Hardware limits of the prefill streaming overlap.
//
// The streaming design copies ~46 MiB per layer into a VRAM slot while the previous layer
// computes. It only pays off if those two run concurrently. This test finds out whether the
// hardware can do it, and if not, which of the three classic limits is hit:
//
//   1. the PCIe link negotiates a low generation (power saving) and never ramps up
//   2. a host-to-device copy cannot proceed while the GPU saturates its own DRAM bandwidth
//   3. the driver serializes the copy behind the compute stream
//
// The copy rate is measured while a kernel runs and compared against the copy alone. If the
// rate holds, the link is fine and the kernel does not contend. If it collapses, the GPU
// memory bus is the shared resource and no amount of stream juggling will fix it.
//
// build: nvcc -O2 -o hw_bench hw_bench.cu

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

// alu bound: no memory traffic, keeps the SMs busy only
__global__ void alu_kernel(float * out, long long iters) {
    float v = 0.0f;
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
    if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = v;
}

// dram bound: stream through device memory, reads and writes the full buffer
__global__ void dram_kernel(const float4 * __restrict__ a, float4 * __restrict__ b, size_t n) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float4 v = a[i];
        v.x += 1.0f; v.y += 1.0f; v.z += 1.0f; v.w += 1.0f;
        b[i] = v;
    }
}

static float elapsed(cudaEvent_t e0, cudaEvent_t e1) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int async_engines = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&async_engines, cudaDevAttrAsyncEngineCount, 0));

    printf("device            : %s cc %d.%d, %d SMs\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("cudaDevAttrAsyncEngineCount : %d\n", async_engines);
    printf("prop.asyncEngineCount       : %d\n", prop.asyncEngineCount);
    printf("memoryClockRate   : %d kHz, busWidth %d bit\n", prop.memoryClockRate, prop.memoryBusWidth);
    printf("pciBusID          : %s\n", prop.pciBusID == 0 ? "?" : "");
    printf("\n");

    cudaStream_t s_alu, s_dram, s_copy;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_alu,  cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_dram, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s_copy, cudaStreamNonBlocking));

    // ---------------------------------------------------------------- device DRAM bandwidth
    // must exceed L2 by a wide margin, otherwise the kernel is served from cache and never
    // touches the VRAM bus at all. the T1000 has a small L2, so use 512 MiB per buffer.
    const size_t dram_n = 512ull*1024ull*1024ull / 16ull;  // 512 MiB of float4
    float4 * dbuf_a = nullptr;
    float4 * dbuf_b = nullptr;
    CUDA_CHECK(cudaMalloc(&dbuf_a, dram_n * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dbuf_b, dram_n * sizeof(float4)));
    CUDA_CHECK(cudaMemset(dbuf_a, 1, dram_n * sizeof(float4)));

    const int dram_blocks = prop.multiProcessorCount * 8;
    const int dram_reps   = 4;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // one rep alone, so the overlap below reuses the same finite kernel
    CUDA_CHECK(cudaEventRecord(e0, s_dram));
    dram_kernel<<<dram_blocks, 256, 0, s_dram>>>(dbuf_a, dbuf_b, dram_n);
    CUDA_CHECK(cudaEventRecord(e1, s_dram));
    CUDA_CHECK(cudaEventSynchronize(e1));
    const float dram_one_ms = elapsed(e0, e1);

    CUDA_CHECK(cudaEventRecord(e0, s_dram));
    for (int r = 0; r < dram_reps; ++r) {
        dram_kernel<<<dram_blocks, 256, 0, s_dram>>>(dbuf_a, dbuf_b, dram_n);
    }
    CUDA_CHECK(cudaEventRecord(e1, s_dram));
    CUDA_CHECK(cudaEventSynchronize(e1));
    const float dram_ms = elapsed(e0, e1);
    const double dram_gbs = 2.0 * (dram_n * 16.0) / (dram_one_ms / 1000.0) / 1e9;
    printf("-- device DRAM --\n");
    printf("512 MiB read+write x1      %8.2f ms  ->  %.1f GB/s\n", dram_one_ms, dram_gbs);
    printf("512 MiB read+write x%-2d     %8.2f ms\n\n", dram_reps, dram_ms);

    // ---------------------------------------------------------------- host to device copy
    const size_t copy_bytes = 512ull*1024ull*1024ull;
    void * dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, copy_bytes));
    void * pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&pinned, copy_bytes, cudaHostAllocDefault));
    for (size_t i = 0; i < copy_bytes; i += 4096) ((char *) pinned)[i] = 1;

    const int    chunks = 8;
    const size_t chunk  = copy_bytes / chunks;

    CUDA_CHECK(cudaEventRecord(e0, s_copy));
    for (int c = 0; c < chunks; ++c) {
        CUDA_CHECK(cudaMemcpyAsync((char *) dst + c*chunk, (char *) pinned + c*chunk, chunk,
                                   cudaMemcpyHostToDevice, s_copy));
    }
    CUDA_CHECK(cudaEventRecord(e1, s_copy));
    CUDA_CHECK(cudaEventSynchronize(e1));
    const float copy_ms = elapsed(e0, e1);
    printf("-- host to device --\n");
    printf("copy alone 512 MiB         %8.2f ms  ->  %.2f GB/s\n", copy_ms, copy_bytes/(copy_ms/1000.0)/1e9);

    // ---------------------------------------------------------------- overlap with DRAM kernel
    // issue the dram kernel and the copy back to back on two streams, then wait for both. the
    // two are finite and about the same length, so comparing the total against max() and
    // against the sum shows whether they shared the device at all.
    {
        cudaEvent_t d0, d_done;
        CUDA_CHECK(cudaEventCreate(&d0));
        CUDA_CHECK(cudaEventCreate(&d_done));

        CUDA_CHECK(cudaEventRecord(e0, s_copy));
        for (int c = 0; c < chunks; ++c) {
            CUDA_CHECK(cudaMemcpyAsync((char *) dst + c*chunk, (char *) pinned + c*chunk, chunk,
                                       cudaMemcpyHostToDevice, s_copy));
        }
        CUDA_CHECK(cudaEventRecord(e1, s_copy));

        CUDA_CHECK(cudaEventRecord(d0, s_dram));
        for (int r = 0; r < dram_reps; ++r) {
            dram_kernel<<<dram_blocks, 256, 0, s_dram>>>(dbuf_a, dbuf_b, dram_n);
        }
        CUDA_CHECK(cudaEventRecord(d_done, s_dram));

        CUDA_CHECK(cudaEventSynchronize(e1));
        const float copy_under_load = elapsed(e0, e1);
        CUDA_CHECK(cudaEventSynchronize(d_done));
        const float dram_busy = elapsed(d0, d_done);

        printf("\n-- overlap: DRAM kernel + H2D copy --\n");
        printf("dram alone                 %8.2f ms\n", dram_ms);
        printf("copy alone                 %8.2f ms\n", copy_ms);
        printf("dram while copying         %8.2f ms  (%.0f%% of alone)\n",
                dram_busy, 100.0*dram_busy/dram_ms);
        printf("copy while dram runs       %8.2f ms  ->  %.2f GB/s (%.0f%% of alone rate)\n",
                copy_under_load, copy_bytes/(copy_under_load/1000.0)/1e9,
                100.0*(copy_ms/copy_under_load));
        printf("serialized would be        %8.2f ms\n", dram_ms + copy_ms);
        printf("max() would be             %8.2f ms\n", dram_ms > copy_ms ? dram_ms : copy_ms);
        printf("verdict: %s\n",
                copy_under_load < 1.25f*(dram_ms > copy_ms ? dram_ms : copy_ms)
                    ? "copy coexists with a DRAM-hungry kernel -> VRAM bus is NOT the blocker"
                    : "copy stretched by the DRAM kernel -> VRAM bus is shared/contended");

        CUDA_CHECK(cudaEventDestroy(d_done));
    }

    // ---------------------------------------------------------------- overlap with ALU kernel
    {
        float * aout = nullptr;
        CUDA_CHECK(cudaMalloc(&aout, sizeof(float)));

        CUDA_CHECK(cudaEventRecord(e0, s_alu));
        alu_kernel<<<prop.multiProcessorCount * 4, 256, 0, s_alu>>>(aout, 400000);
        CUDA_CHECK(cudaEventRecord(e1, s_alu));
        CUDA_CHECK(cudaEventSynchronize(e1));
        const float alu_ms = elapsed(e0, e1);

        CUDA_CHECK(cudaEventRecord(e0, s_alu));
        alu_kernel<<<prop.multiProcessorCount * 4, 256, 0, s_alu>>>(aout, 400000);
        for (int c = 0; c < chunks; ++c) {
            CUDA_CHECK(cudaMemcpyAsync((char *) dst + c*chunk, (char *) pinned + c*chunk, chunk,
                                       cudaMemcpyHostToDevice, s_copy));
        }
        // both streams must be tracked. recording only on s_alu measures the kernel alone and
        // silently ignores the copy, which is the bug this section used to have.
        cudaEvent_t ev_alu_done, ev_copy_done;
        CUDA_CHECK(cudaEventCreate(&ev_alu_done));
        CUDA_CHECK(cudaEventCreate(&ev_copy_done));
        CUDA_CHECK(cudaEventRecord(ev_alu_done,  s_alu));
        CUDA_CHECK(cudaEventRecord(ev_copy_done, s_copy));

        CUDA_CHECK(cudaEventSynchronize(ev_alu_done));
        CUDA_CHECK(cudaEventSynchronize(ev_copy_done));

        float t_alu_done = 0.0f, t_copy_done = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&t_alu_done,  e0, ev_alu_done));
        CUDA_CHECK(cudaEventElapsedTime(&t_copy_done, e0, ev_copy_done));
        const float both = t_alu_done > t_copy_done ? t_alu_done : t_copy_done;

        printf("\n-- overlap: ALU kernel + H2D copy --\n");
        printf("alu alone                  %8.2f ms\n", alu_ms);
        printf("copy alone                 %8.2f ms\n", copy_ms);
        printf("alu  finished at           %8.2f ms\n", t_alu_done);
        printf("copy finished at           %8.2f ms\n", t_copy_done);
        printf("both (max of the two)      %8.2f ms\n", both);
        printf("serialized would be        %8.2f ms\n", alu_ms + copy_ms);
        printf("max() would be             %8.2f ms\n", alu_ms > copy_ms ? alu_ms : copy_ms);
        printf("verdict: %s\n", both < 1.25f*(alu_ms > copy_ms ? alu_ms : copy_ms)
                    ? "OVERLAP works when only the SMs are busy" : "no overlap even for pure ALU");

        CUDA_CHECK(cudaEventDestroy(ev_alu_done));
        CUDA_CHECK(cudaEventDestroy(ev_copy_done));
        CUDA_CHECK(cudaFree(aout));
    }

    return 0;
}
