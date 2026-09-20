// Does TU117 (Quadro T1000 / sm_75) have a real int8 tensor pipe?
//
// The J=128 MMQ kernel in this build contains 0 HMMA, 1024 IMMA.8816.S8.S8 and
// 2048 FFMA. The SASS IMMA.8816.S8.S8 corresponds to the PTX
// mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 that mma.cuh uses on Turing
// (m16n8k16 needs Ampere, so Turing gets 2x m8n8k16).
//
// ncu reports sm__pipe_tensor_op_hmma_cycles_active = 39.45% for a kernel with
// no HMMA at all, so that counter cannot tell us anything. Measure the
// instruction directly instead, with operands in registers and zero memory
// traffic, and compare against __dp4a and FFMA.
//
// The timing is a scaling test on purpose: it prints time against iteration
// count, so a loop that the compiler eliminated shows up as flat time instead of
// a slope. Nothing is hidden behind a single average.
//
// Build: nvcc -O3 -arch=sm_75 --cudart shared imma.cu -o imma -lnvidia-ml -lpthread
// Run:   ./imma

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <thread>

#include <cuda_runtime.h>
#include <nvml.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (line %d)\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static nvmlDevice_t g_dev;

static double power_now(void) {
    unsigned mw = 0;
    nvmlDeviceGetPowerUsage(g_dev, &mw);
    return mw / 1000.0;
}

// ---------------------------------------------------------------- kernels

// the exact pair of mma llama.cpp uses on Turing
__global__ void imma_kernel(int iters, int * out) {
    unsigned a0 = threadIdx.x * 2654435761u + 1u;
    unsigned a1 = threadIdx.x * 40503u + 3u;
    unsigned b0 = threadIdx.x * 2246822519u + 7u;

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    int d4 = 0, d5 = 0, d6 = 0, d7 = 0;

    // 4 independent accumulators. 2 would leave the kernel latency-bound on the
    // mma dependency chain, and then the measurement says nothing about the
    // pipe. the operands also depend on the counter, or the loop becomes
    // loop-invariant and is collapsed (the scaling check below caught that).
    for (int i = 0; i < iters; ++i) {
        a0 += (unsigned) i;
        b0 ^= (unsigned) i;
        asm volatile ("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(d0), "+r"(d1) : "r"(a0), "r"(b0));
        asm volatile ("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(d2), "+r"(d3) : "r"(a1), "r"(b0));
        asm volatile ("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(d4), "+r"(d5) : "r"(a0), "r"(b0));
        asm volatile ("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(d6), "+r"(d7) : "r"(a1), "r"(b0));
    }

    // a data-dependent write: the compiler cannot prove this never fires
    if (d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7 == 0x5a5a5a5a) { out[0] = d0; }
}

// same int8 MACs through __dp4a: 2 mma = 2*8*8*16 = 2048 MACs = 512 dp4a,
// done 8-way to give the scheduler similar ILP
__global__ void dp4a_kernel(int iters, int * out) {
    const unsigned a0 = threadIdx.x * 2654435761u;
    int m = (int) a0;
    int c = (int) (a0 ^ 0x5a5a5a5au);
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    for (int i = 0; i < iters; ++i) {
        m += i;
        c ^= i;
        for (int k = 0; k < 64; ++k) {
            acc0 = __dp4a(m, c, acc0);  acc1 = __dp4a(m, c, acc1);
            acc2 = __dp4a(m, c, acc2);  acc3 = __dp4a(m, c, acc3);
            acc4 = __dp4a(m, c, acc4);  acc5 = __dp4a(m, c, acc5);
            acc6 = __dp4a(m, c, acc6);  acc7 = __dp4a(m, c, acc7);
        }
    }

    if (acc0+acc1+acc2+acc3+acc4+acc5+acc6+acc7 == 0x5a5a5a5a) { out[0] = acc0; }
}

// FP32 pipe reference, same shape
__global__ void ffma_kernel(int iters, int * out) {
    float x = threadIdx.x * 1.0e-3f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int i = 0; i < iters; ++i) {
        x += 1.0e-7f;
        for (int k = 0; k < 256; ++k) {
            acc0 = fmaf(x, 1.0001f, acc0);  acc1 = fmaf(x, 1.0002f, acc1);
            acc2 = fmaf(x, 1.0003f, acc2);  acc3 = fmaf(x, 1.0004f, acc3);
        }
    }

    if ((int) (acc0+acc1+acc2+acc3) == 0x5a5a5a5a) { out[0] = (int) acc0; }
}

// ---------------------------------------------------------------- harness

typedef void (*kern_t)(int, int *);

struct job {
    const char * name;
    kern_t       k;
    // MACs per iteration per THREAD. mma is per-warp: 2 * 8*8*16 = 2048 per
    // warp, so /32 per thread, because all 32 lanes execute the same asm.
    double       macs_per_iter_thread;
};

static void run(const job & j, int grid, int block, double secs) {
    const double threads = (double) grid * block;

    int * out = nullptr;
    CHECK(cudaMalloc(&out, sizeof(int)));

    j.k<<<grid, block>>>(1000, out);    // warm up
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // scaling check: a loop the compiler removed shows flat time here
    printf("  scaling:");
    int    last_iters = 0;
    double last_ms    = 0.0;
    int    cal_iters  = 8000;
    double cal_ms     = 1.0;
    const int counts[] = { 2000, 8000, 32000, 128000 };
    for (int idx = 0; idx < 4; ++idx) {
        const int iters = counts[idx];
        cudaEventRecord(e0);
        j.k<<<grid, block>>>(iters, out);
        CHECK(cudaGetLastError());
        cudaEventRecord(e1);
        CHECK(cudaDeviceSynchronize());
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        printf(" %d:%.4fms", iters, ms);
        last_iters = iters;
        last_ms    = ms;
        if (idx == 1) { cal_iters = iters; cal_ms = ms; }
    }
    printf("\n");

    if (last_ms <= 0.0) {
        printf("  SKIPPED: zero time, the loop was eliminated\n\n");
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(out);
        return;
    }

    long long n_launch = (long long) (secs * 1000.0 / cal_ms);
    if (n_launch < 1) { n_launch = 1; }
    const int n_iter = cal_iters;

    std::vector<double> samples;
    bool run_sampler = true;
    std::thread sampler([&]() {
        while (run_sampler) {
            samples.push_back(power_now());
            for (volatile int i = 0; i < 20000; i++) {}
        }
    });

    cudaEventRecord(e0);
    for (long long i = 0; i < n_launch; ++i) { j.k<<<grid, block>>>(n_iter, out); }
    CHECK(cudaGetLastError());
    cudaEventRecord(e1);
    CHECK(cudaDeviceSynchronize());

    run_sampler = false;
    sampler.join();

    double w_avg = 0.0, w_max = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        w_avg += samples[i];
        if (samples[i] > w_max) { w_max = samples[i]; }
    }
    if (!samples.empty()) { w_avg /= (double) samples.size(); }

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, e0, e1);

    const double macs = (double) n_launch * (double) n_iter * threads * j.macs_per_iter_thread;
    const double t    = (double) ms_total / 1000.0;

    printf("  %lld launches x %d iters in %.1f ms -> %.3f GMAC/s (%.1f W avg / %.1f W max)\n\n",
           n_launch, n_iter, (double) ms_total, macs / t / 1e9, w_avg, w_max);

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(out);
}

int main(void) {
    if (nvmlInit() != NVML_SUCCESS) { fprintf(stderr, "nvml init failed\n"); return 1; }
    if (nvmlDeviceGetHandleByIndex(0, &g_dev) != NVML_SUCCESS) { fprintf(stderr, "nvml handle failed\n"); return 1; }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    const int sms   = prop.multiProcessorCount;
    const int block = 256;
    const int grid  = sms * 8;

    printf("%s, sm_%d%d, %d SMs, grid %d x block %d = %.0f threads\n\n",
           prop.name, prop.major, prop.minor, sms, grid, block, (double) grid * block);

    const job jobs[] = {
        { "imma m8n8k16.s8.s8 x4", imma_kernel, 4.0 * 8 * 8 * 16 / 32.0 },
        { "dp4a (512 per iter)",   dp4a_kernel, 64.0 * 8 * 4.0           },
        { "ffma (1024 per iter)",  ffma_kernel, 256.0 * 4.0              },
    };

    for (int i = 0; i < 3; ++i) {
        printf("%s  [%.0f MACs/iter/thread]\n", jobs[i].name, jobs[i].macs_per_iter_thread);
        run(jobs[i], grid, block, 3.0);
    }

    nvmlShutdown();
    return 0;
}
