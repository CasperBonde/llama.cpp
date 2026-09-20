// Measure the H2D path the prefill streaming slot fill uses, in isolation.
//
// The slot fill copies host-resident layer weights into a device buffer. This program
// measures the same transfer with the variables that the design can control:
//
//   - source memory is pageable (ggml_aligned_malloc) or pinned (cudaHostAlloc)
//   - transfer is one large contiguous block, or split into chunks of a given size
//   - transfer is host to device only, or host to device mixed with device to host
//
// The mixed case matters because llama.cpp moves MoE expert weights across the same PCIe
// link in both directions at the same time.
//
// build: nvcc -O2 -o copy_bench copy_bench.cu

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

static const size_t GiB = 1024ull*1024ull*1024ull;

// total bytes to move, and the chunk size to move them in
static double bench_h2d(void * dst_dev, void * src_host, size_t total, size_t chunk, int n_iter) {
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // warmup, also faults in the source pages
    CUDA_CHECK(cudaMemcpy(dst_dev, src_host, chunk, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int it = 0; it < n_iter; ++it) {
        for (size_t off = 0; off + chunk <= total; off += chunk) {
            CUDA_CHECK(cudaMemcpy((char *) dst_dev + off, (char *) src_host + off, chunk, cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    const double bytes = (double) total * n_iter;
    const double gbs   = bytes / (ms/1000.0) / 1e9;

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return gbs;
}

// interleave H2D and D2H over the same link, to mimic the MoE expert path
static double bench_mixed(void * dst_dev, void * src_host, void * src_dev, void * dst_host,
                          size_t total, size_t chunk, int n_iter) {
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaMemcpy(src_dev, src_host, chunk, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int it = 0; it < n_iter; ++it) {
        for (size_t off = 0; off + chunk <= total; off += chunk) {
            CUDA_CHECK(cudaMemcpy((char *) dst_dev + off, (char *) src_host + off, chunk, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dst_host, (char *) src_dev + off, chunk, cudaMemcpyDeviceToHost));
        }
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    // bytes moved over the link is both directions summed
    const double bytes = 2.0 * (double) total * n_iter;
    const double gbs   = bytes / (ms/1000.0) / 1e9;

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return gbs;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("device: %s, cc %d.%d\n", prop.name, prop.major, prop.minor);
    printf("pcie link: gen 3 x16 (from nvidia-smi)\n");
    printf("pcie theoretical: %.2f GB/s\n", 8.0*16.0*(128.0/130.0));
    printf("\n");

    const size_t total = 2*GiB;
    const int n_iter = 3;

    void * pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&pinned, total, cudaHostAllocDefault));

    void * pageable = malloc(total);
    if (pageable == nullptr) { printf("malloc failed\n"); return 1; }
    for (size_t i = 0; i < total; i += 4096) {
        ((char *) pageable)[i] = (char) i;
    }

    void * dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, total));

    void * host_out = malloc(total);

    printf("%-46s %8s\n", "transfer", "GB/s");
    printf("%-46s %8s\n", "----------------------------------------------", "--------");

    const size_t chunk_sizes[] = { total, 64u*1024u*1024u, 16u*1024u*1024u, 6u*1024u*1024u, 1u*1024u*1024u };
    const char * chunk_names[]  = { "one block", "64 MiB chunks", "16 MiB chunks", "6 MiB chunks", "1 MiB chunks" };

    printf("\n-- pageable source (what llama.cpp uses today) --\n");
    for (int i = 0; i < 5; ++i) {
        printf("%-46s %8.2f\n", chunk_names[i], bench_h2d(dev, pageable, total, chunk_sizes[i], n_iter));
    }

    printf("\n-- pinned source --\n");
    for (int i = 0; i < 5; ++i) {
        printf("%-46s %8.2f\n", chunk_names[i], bench_h2d(dev, pinned, total, chunk_sizes[i], n_iter));
    }

    printf("\n-- mixed: H2D interleaved with D2H (mimics the MoE expert path) --\n");
    printf("%-46s %8.2f\n", "pageable, 6 MiB chunks", bench_mixed(dev, pageable, dev, host_out, total, 6u*1024u*1024u, n_iter));
    printf("%-46s %8.2f\n", "pinned, 6 MiB chunks",   bench_mixed(dev, pinned,   dev, host_out, total, 6u*1024u*1024u, n_iter));

    cudaFreeHost(pinned);
    cudaFree(dev);
    free(pageable);
    free(host_out);

    return 0;
}
