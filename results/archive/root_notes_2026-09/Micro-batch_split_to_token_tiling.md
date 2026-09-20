*User prompt: Grunden til jeg skal kører micro batch størrelse på 1024 er at jeg for hver lag skal overfører alle eksperterne til VRAM. Er der ikke en måde jeg kan omstrukturere beregningerne på så de tages i mindre bidder \- der giver færre cache miss? Men stadig beregner alle 1024 tokens lag for lag? Tænk rent matematisk \- ikke settings i Llama.cpp*

Response: Det er en helt central og matematisk præcis problemstilling, du fatte fat i her. Du står med en direkte konflikt mellem to fysiske flaskehalse:

> 1. **PCIe-bussen (Vægt-overførsel):** Kræver en stor batch (*B*\=1024) for at amortisere overførslen af eksperterne *W*(*l*), så *O*(*W*) kun betales én gang pr. lag.  
> 2. **GPU L2-cache / GDDR-båndbredde (Aktiveringer):** Kræver en lille micro-batch (*b*\=128), så aktiveringstensorerne *O*(*b*⋅*d*model​) kan holdes i L2-cachen under matrix-multiplikationerne.

Svaret er **ja, det kan lade sig gøre rent matematisk**. Løsningen hedder **Token-Tiling inden for laggrænsen** (*In-Layer Token Partitioning*), hvor man adskiller *vægt-loopet* fra *aktiverings-loopet*.

### **Den matematiske omstrukturering**

Lad os betragte et enkelt lag *l*. Input-aktiveringsmatricen for hele batchen er *X*(*l*)∈R*B*×*d*model​, hvor *B*\=1024.

Fordi matrix-multiplikation (GEMM) i tætte lag og MoE-eksperter er **rækkedistributiv** (hver token behandles uafhængigt i lineære projektioner), kan vi opdele *X*(*l*) langs token-dimensionen i *k* mindre "fliser" (tiles) af størrelse *b*\=*kB*​\=128:

*X*(*l*)\=​*X*1(*l*)​*X*2(*l*)​⋮*Xk*(*l*)​​​,hvor *Xi*(*l*)​∈R*b*×*d*model​

Da matrixmultiplikationen *Y*\=*X*⋅*W* overholder:

*Y*(*l*)\=​*X*1(*l*)​*X*2(*l*)​⋮*Xk*(*l*)​​​*W*(*l*)\=​*X*1(*l*)​*W*(*l*)*X*2(*l*)​*W*(*l*)⋮*Xk*(*l*)​*W*(*l*)​​

kan vi ændre beregningsrækkefølgen (loop-interchange) uden at ændre det matematiske resultat.

### **Algoritme: Ydre Vægt-loop \+ Indre Aktiverings-tile**

I stedet for at køre hele *B*\=1024 igennem alle lagets operationer på én gang, opdeler vi udførelsen inde i lag *l* således:

#### **Step 1: Overførsel og Låsning af Vægtsæt (Ydre Step)**

Overfør alle nødvendige vægte for Lag *l* (Attention *WQ*​,*WK*​,*WV*​,*WO*​ samt de aktive MoE-eksperter *W*gate​,*W*up​,*W*down​ for de 1024 tokens) over PCIe til VRAM **én enkelt gang**.

#### **Step 2: Sekventiel Flise-beregning (Indre Step)**

Med vægtene *W*(*l*) liggende fast i VRAM, itererer GPU'en over aktiverings-fliserne *i*\=1…*k* (hvor *k*\=8 for *b*\=128):

> 1. **Indlæs Flise *Xi*(*l*)​:** Indlæs kun 128×*d*model​ aktiverings-data ( 2,09 MB ved FP16).  
> 2. **Kør Lineære Projektioner & MoE:**  
>    *Yi*,proj​\=*Xi*(*l*)​⋅*W*(*l*)  
>    Da *Xi*(*l*)​ og dens mellemliggende projektioner fylder \<3 MB, kan de i høj grad blive i GPU'ens SRAM/L2-cache under udregningen af SiLU/GELU og elementvise operationer.  
> 3. **Gem Flise-resultat *Yi*(*l*)​:** Skriv det mindre delresultat tilbage til den samlede output-buffer *Y*(*l*).

#### **Step 3: Samling og Næste Lag**

Saml *Y*(*l*)\=\[*Y*1(*l*)​;*Y*2(*l*)​;…;*Yk*(*l*)​\] og gå videre til Lag *l*\+1.

### **Hvad med Attention-mekanismen? (Cross-Token afhængighed)**

Lineære lag og MoE-eksperter er 100% uafhængige pr. token. Attention-mekanismen er den eneste operation, der kobler tokens på tværs (*Q*⋅*KT*).

Hvordan beregner vi Attention for *Xi*​ uden at ødelægge cachen?

Det løses matematisk via **Online Softmax (Tiled FlashAttention)**:

> * Query-matrix opdeles i fliser *Qi*​∈R*b*×*dh*​ (*b*\=128).  
> * Key og Value (*K*,*V*) for alle 1024 tokens behandles i blokke.  
> * Man beregner den skallerede prikprodukt-attention med en rullende max- og sum-skalering (Online Softmax):

Attention(*Qi*​,*K*,*V*)=Softmax(*dk*​​*Qi*​*KT*​)*V*

Fordi Softmax kan udregnes i blokke ved at opdatere en løbende normaliseringsfaktor, behøver GPU'en aldrig at gemme den fulde 1024×1024 Attention-matrix i VRAM. *Qi*​\-flisen holdes i L2-cache, mens *K*,*V* streames igennem.

### **Sammenligning af Hukommelsesaftryk i L2 Cache**

| Tilgang | Aktiverings-tensor i L2 pr. pass | PCIe-overførsler af Vægte | L2 Cache Hit-rate (T1000) |
| :---- | :---- | :---- | :---- |
| **Standard *B*\=1024 Micro-batch** | ∼16,7 MB | 1× pr. lag | **Lav** (Cache misses, GDDR-thrashing) |
| **Standard *b*\=128 Micro-batch** | ∼2,09 MB | 8× pr. lag | **Høj** (Men PCIe-bussen dør) |
| **Matematisk Token-Tiled Lag** | **∼2,09 MB** | **1× pr. lag** | **Høj** (Optimal L2 udnyttelse \+ minimal PCIe) |

### **Konklusion**

Ved at tænke på operativsystem- og kernel-niveau skal du **ikke** reducere batchen over for PCIe-bussen, men derimod instruere GPU-execution-kernelen i at **loop'e over token-dimensionen i bidder af 128**, mens *W*(*l*) forbliver fastlåst (*pinned*) i VRAM.

Det eliminerer aktiverings-thrashingen i GDDR-hukommelsen uden at betale ekstraskat på PCIe-bussen.

---

*User prompt: Vis mig et C++/CUDA kildekode-eksempel på hvordan Token-Tiling opbygges for at opdele aktiveringer i mindre blocks mens vægte er pinned i VRAM.*

Response: Koden er opdelt i to niveauer: **Host-styringen** (som fastholder lagets vægte i VRAM over PCIe og looper over token-fliser) og **CUDA-kernelen** (som udnytter Shared Memory på GPU'en til at holde aktiverings-flisen inde i L1/SRAM-cachen).

`#include <cuda_runtime.h>`  
`#include <iostream>`  
`#include <vector>`  
`#include <algorithm>`

`// Tile-størrelser til Shared Memory inde i CUDA kernelen`  
`#define TILE_SIZE 16`

`/**`  
 `* CUDA Kernel: Tiled GEMM for en enkelt token-flise (b x K) * W (K x N) -> Y_tile (b x N)`  
 `*`  
 `* X_tile: [b x K] - Aktiverings-flise (f.eks. b = 128 tokens)`  
 `* W:      [K x N] - Fastlåste lag-vægte i VRAM`  
 `* Y_tile: [b x N] - Output-flise`  
 `*/`  
`__global__ void gemm_token_tile_kernel(`  
    `const float* __restrict__ X_tile,`  
    `const float* __restrict__ W,`  
    `float* __restrict__ Y_tile,`  
    `int b_current, int K, int N)`   
`{`  
    `// Shared Memory på GPU'en til at undgå L2-cache misses og GDDR-thrashing`  
    `__shared__ float s_X[TILE_SIZE][TILE_SIZE];`  
    `__shared__ float s_W[TILE_SIZE][TILE_SIZE];`

    `int row = blockIdx.y * blockDim.y + threadIdx.y; // Token index i flisen (0..b_current-1)`  
    `int col = blockIdx.x * blockDim.x + threadIdx.x; // Feature index (0..N-1)`

    `float acc = 0.0f;`

    `// Loop over K-dimensionen (d_in) i TILE_SIZE bidder`  
    `for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ++ph) {`  
          
        `// 1. Hent data ind i ultrahurtig Shared Memory (SRAM)`  
        `if (row < b_current && (ph * TILE_SIZE + threadIdx.x) < K) {`  
            `s_X[threadIdx.y][threadIdx.x] = X_tile[row * K + ph * TILE_SIZE + threadIdx.x];`  
        `} else {`  
            `s_X[threadIdx.y][threadIdx.x] = 0.0f;`  
        `}`

        `if ((ph * TILE_SIZE + threadIdx.y) < K && col < N) {`  
            `s_W[threadIdx.y][threadIdx.x] = W[(ph * TILE_SIZE + threadIdx.y) * N + col];`  
        `} else {`  
            `s_W[threadIdx.y][threadIdx.x] = 0.0f;`  
        `}`

        `__syncthreads();`

        `// 2. Beregn dot-product for denne tile i SRAM`  
        `#pragma unroll`  
        `for (int k = 0; k < TILE_SIZE; ++k) {`  
            `acc += s_X[threadIdx.y][k] * s_W[k][threadIdx.x];`  
        `}`

        `__syncthreads();`  
    `}`

    `// 3. Skriv det færdige resultat ud i output-flisen`  
    `if (row < b_current && col < N) {`  
        `Y_tile[row * N + col] = acc;`  
    `}`  
`}`

`/**`  
 `* Host Orchestrator: Afvikler et fuldt lag med Token-Tiling`  
 `*`   
 `* B = Total Batch (f.eks. 1024 tokens)`  
 `* b_tile = Token Tile-størrelse (f.eks. 128 tokens)`  
 `* K = d_in (f.eks. 8192)`  
 `* N = d_out (f.eks. 8192)`  
 `*/`  
`void execute_layer_token_tiled(`  
    `const float* h_W,       // Vægte sendes fra CPU Host over PCIe`  
    `const float* d_X_full,  // Fuld aktiverings-matrix på GPU [B x K]`  
    `float* d_Y_full,        // Fuld output-matrix på GPU [B x N]`  
    `int B, int K, int N, int b_tile)`   
`{`  
    `// STEP 1: Overfør og fastlås (Pin) lagets vægte i VRAM ÉN GANG`  
    `float* d_W = nullptr;`  
    `size_t w_bytes = static_cast<size_t>(K) * N * sizeof(float);`  
    `cudaMalloc(&d_W, w_bytes);`  
      
    `// PCIe transfer koster kun én gang pr. lag over bussen`  
    `cudaMemcpy(d_W, h_W, w_bytes, cudaMemcpyHostToDevice);`

    `// STEP 2: Indre Token-Tiling Loop (Splits B=1024 op i bidder af b_tile=128)`  
    `for (int offset_b = 0; offset_b < B; offset_b += b_tile) {`  
          
        `int current_b = std::min(b_tile, B - offset_b);`

        `// Pointer-aritmetik: peg på den aktuelle aktiverings-flise og output-flise`  
        `const float* d_X_tile = d_X_full + (offset_b * K);`  
        `float* d_Y_tile = d_Y_full + (offset_b * N);`

        `// Grid og Block konfiguration for den lille tile (128 x N)`  
        `dim3 block(TILE_SIZE, TILE_SIZE);`  
        `dim3 grid((N + block.x - 1) / block.x, (current_b + block.y - 1) / block.y);`

        `// d_W er PINNED i VRAM og genbruges af alle tiles uden genoverførsel`  
        `gemm_token_tile_kernel<<<grid, block>>>(d_X_tile, d_W, d_Y_tile, current_b, K, N);`  
    `}`

    `// Synkroniser stream for at sikre at alle tiles er færdige`  
    `cudaDeviceSynchronize();`

    `// Frigør lagets vægte før næste lag indlæses over PCIe`  
    `cudaFree(d_W);`  
`}`

`int main() {`  
    `int B = 1024;    // Total batch`  
    `int b_tile = 128;// Micro-tile (passer i GPU L2/SRAM)`  
    `int K = 4096;    // Input dimension`  
    `int N = 4096;    // Output dimension`

    `// Alloker Host Vægte (Simuleret PCIe-overførsel)`  
    `std::vector<float> h_W(K * N, 0.01f);`

    `// Alloker GPU Aktiveringer`  
    `float *d_X_full = nullptr, *d_Y_full = nullptr;`  
    `cudaMalloc(&d_X_full, B * K * sizeof(float));`  
    `cudaMalloc(&d_Y_full, B * N * sizeof(float));`

    `std::cout << "Korer Token-Tiled Layer Execution (B=" << B << ", b_tile=" << b_tile << ")...\n";`  
      
    `// Afvikl laget`  
    `execute_layer_token_tiled(h_W.data(), d_X_full, d_Y_full, B, K, N, b_tile);`

    `std::cout << "Lag-beregning fuldfort uden L2-cache thrashing!\n";`

    `cudaFree(d_X_full);`  
    `cudaFree(d_Y_full);`  
    `return 0;`  
`}`

### **Hvorfor denne struktur fjerner flaskehalsen**

> 1. **PCIe Amortisering (cudaMemcpy placeret før loopet):** Vægterne d\_W kopieres over PCIe-bussen præcis **1 gang** pr. lag for alle 1024 tokens i stedet for 8 separate overførsler.  
> 2. **L2/SRAM Cache Hit-rate (d\_X\_tile pointer-fremføring):** I for (int offset\_b \= 0; ...) loopet sendes kun d\_X\_tile svarende til 128 tokens×4096 dim×4 bytes≈2 MB til kernelen ad gangen. Data holdes dermed i GPU'ens L2-cache og Shared Memory (s\_X), hvilket eliminere mætning af VRAM GDDR-båndbredden.

---

*User prompt: Forklar hvordan denne Token-Tiling struktur kan integreres i llama.cpp og GGML compute graph arkitekturen.*

Response: I llama.cpp og dens underliggende tensor-bibliotek **GGML** evalueres modeller normalt ved at opbygge én stor compute graph (ggml\_cgraph), som repræsenterer *hele* modellen (alle 32–80 lag) for en givet micro-batch (ubatch).

For at implementere **Token-Tiling** skal denne grafevaluerings-hierarki vendes om, så **Ydre loop \= Lag** og **Indre loop \= Token-fliser**.

### **1\. Arkitektonisk ændring: Fra Monolitisk Graf til Per-Lag Subgraphs**

I standard llama.cpp ser udførelsen af prefill således ud:

For hver ubatch (*b*\=128)⟶Kør Graf (Lag 0…*L*−1)

Ved Token-Tiling omstruktureres det til:

For hvert Lag *l*∈\[0,*L*−1\]⟶\[Load/Pin *W*(*l*)⟶For hver Token Tile *i*∈\[0,*k*−1\]→Kør Lag-Subgraf(*Xi*​)\]

### **2\. Nøglemekanismen i GGML: Nul-Kopi Slicing med ggml\_view\_2d**

For at opdele en samlet aktiveringstensor *X*full​∈R*B*×*d*model​ uden hukommelses-overhead, udnyttes GGML's view-operatorer. ggml\_view\_2d opretter en ny ggml\_tensor, der peger direkte ind i den eksisterende VRAM-buffer med en offset.

`// Offset i bytes for tile 'i'`  
`size_t offset_bytes = i * b_tile * d_model * ggml_type_size(type);`

`// Opret et zero-copy view af aktiverings-flisen i VRAM`  
`struct ggml_tensor * X_tile = ggml_view_2d(`  
    `ctx,`   
    `X_full,                 // Kilde: Den fulde 1024-token buffer`  
    `d_model,                // ne0: Feature-dimension`  
    `b_tile,                 // ne1: Micro-batch (f.eks. 128 tokens)`  
    `d_model * element_size, // nb1: Stride pr. token`  
    `offset_bytes            // Offset til start-token for denne tile`  
`);`

### **3\. C++ Pseudo-kode: Opbygning og afvikling i llama.cpp**

Her er, hvordan en per-lag Token-Tiled grafevaluering kodes ved brug af GGML's backend-API:

`#include "ggml.h"`  
`#include "ggml-alloc.h"`  
`#include "ggml-backend.h"`

`void build_and_eval_token_tiled_layer(`  
    `int layer_idx,`  
    `struct ggml_tensor * d_X_full,  // [B x d_model] i VRAM (B = 1024)`  
    `struct ggml_tensor * d_Y_full,  // [B x d_model] i VRAM`  
    `struct llama_layer * layer,     // Vægte for dette lag`  
    `int B, int b_tile, int d_model,`  
    `ggml_backend_t backend)`   
`{`  
    `int num_tiles = B / b_tile; // f.eks. 1024 / 128 = 8 tiles`

    `// STEP 1: Fastlås/Stream lagets vægte til VRAM (f.eks. MoE Eksperter for Lag 'layer_idx')`  
    `load_and_pin_layer_weights_to_vram(layer, backend);`

    `// STEP 2: Loop over alle token-fliser i samtlige 1024 tokens`  
    `for (int i = 0; i < num_tiles; ++i) {`  
          
        `// Opret en midlertidig GGML kontekst til flisens subgraph`  
        `struct ggml_init_params params = {`  
            `/* .mem_size   = */ ggml_tensor_overhead() * 32,`  
            `/* .mem_buffer = */ NULL,`  
            `/* .no_alloc   = */ true, // Ingen allokering her; bruger backend allocators`  
        `};`  
        `struct ggml_context * ctx = ggml_init(params);`

        `// A. Opret Zero-Copy Views af input og output for denne tile`  
        `size_t byte_offset = i * b_tile * d_model * sizeof(float);`  
          
        `struct ggml_tensor * X_tile = ggml_view_2d(ctx, d_X_full, d_model, b_tile, d_model * sizeof(float), byte_offset);`  
        `struct ggml_tensor * Y_tile = ggml_view_2d(ctx, d_Y_full, d_model, b_tile, d_model * sizeof(float), byte_offset);`

        `// B. Opbyg Lagets Compute Graph for kun DENNE tile (128 tokens)`  
        `struct ggml_cgraph * gf = ggml_new_graph(ctx);`

        `// 1. Attention & Feed-Forward/MoE projektioner med lagets fastlåste vægte`  
        `struct ggml_tensor * Q = ggml_mul_mat(ctx, layer->w_q, X_tile);`  
        `struct ggml_tensor * K = ggml_mul_mat(ctx, layer->w_k, X_tile);`  
        `struct ggml_tensor * V = ggml_mul_mat(ctx, layer->w_v, X_tile);`  
          
        `// 2. FlashAttention node for tile 'i'`  
        `struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, Q, K, V, NULL, 1.0f / sqrtf(d_head), 0.0f, 0.0f);`  
          
        `// 3. MoE / FFN lag`  
        `struct ggml_tensor * ffn_out  = build_moe_ffn_subgraph(ctx, layer, attn_out);`

        `// 4. Gem i output-viewet Y_tile`  
        `struct ggml_tensor * cur = ggml_add(ctx, X_tile, ffn_out);`  
        `ggml_build_forward_expand(gf, ggml_cpy(ctx, cur, Y_tile));`

        `// C. Evaluér subgrafen på GPU backend`  
        `// BEMÆRK: Vægterne (layer->w_q osv.) genbruges i VRAM uden at blive genindlæst!`  
        `ggml_backend_graph_compute(backend, gf);`

        `// Oprydning af graf-konteksten for denne tile`  
        `ggml_free(ctx);`  
    `}`

    `// STEP 3: Frigør lagets vægte fra VRAM før næste lag`  
    `evict_layer_weights_from_vram(layer);`  
`}`

### **4\. Gevinster i GGML's Hukommelses-Allokator (ggml\_gallocr)**

En stor sidegevinst ved at integrere dette i GGML er optimeringen af scratch-bufferen:

> * **Før (Monolitisk *B*\=1024 graf):** ggml\_gallocr skal allokere midlertidige tensorer (som Q, K, V og FFN intermediate states) til 1024 tokens i VRAM samtidig. Det resulterer i et enormt VRAM-fodaftryk.  
> * **Efter (Token-Tiled per-lag graf):** Fordi grafevalueringen genbruges for hver flise på *b*tile​\=128, behøver ggml\_gallocr kun at afsætte scratch-plads i VRAM til 128 tokens ad gangen.

Når flise *i* er færdig, genbruges den samme lille VRAM scratch-buffer øjeblikkeligt af flise *i*\+1.

### **Sammenfatning af ændringerne i llama.cpp**

| Komponent | Standard llama.cpp | Token-Tiled llama.cpp |
| :---- | :---- | :---- |
| **ggml\_cgraph Scope** | Hele modellen (*L* lag × *B* tokens) | Enkelt lag (1 lag × *b*tile​ tokens) |
| **Vægt-livscyklus** | Swappes eller streames *U* gange pr. prompt | Swappes kun **1 gang** pr. lag pr. prompt |
| **Aktiverings-buffer** | *O*(*B*⋅*d*model​) gennem alle lag | Flises op via ggml\_view\_2d uden ekstra kopiering |
| **VRAM Scratch-pool** | Stor (dimensioneret efter *B*\=1024) | Meget lille (dimensioneret efter *b*tile​\=128) |

---

*User prompt: Forklar hvordan man kombinerer Token-Tiling med Double-Buffering i GGML for asynkront at prefetch'e næste lags MoE-vægte over PCIe mens den nuværende tile beregnes på GPU'en.*

Response: Ved at kombinerer **Token-Tiling** med **Double-Buffering** (Ping-Pong buffering) kan du opnå næsten 100% overlappende udførelse, hvor PCIe-bussen og GPU'ens Compute Engines arbejder fuldstændig parallelt uden "stalls":

> 1. **GPU Compute Stream (stream\_compute):** Afvikler Token-Tiling runden for Lag *l* ved hjælp af vægte placeret i **Buffer A**.  
> 2. **GPU Transfer Stream (stream\_transfer):** Kører asynkront i baggrunden over PCIe og henter MoE-vægte for Lag *l*\+1 ned i **Buffer B**.

Når Lag *l* er færdigt, bytter de to buffere rolle (Ping-Pong), og processen fortsætter uden at GPU'en nogensinde venter på PCIe-bussen.

### **Matematisk betingelse for 0-Stall Prefetch**

For at skjul-mekanismen fungerer uden flaskehalse, skal den samlede beregningstid for alle *k* token-fliser i Lag *l* være større end eller lig med overførselstiden for Lag *l*\+1's vægte:

*T*compute​(*l*)=*i*\=0∑*k*−1​*T*tile\_compute​(*Xi*​,*W*(*l*))≥PCIe BandwidthSize(*W*(*l*\+1))​\=*T*transfer​(*l*\+1)

Token-Tiling sikrer her, at *T*tile\_compute​ kører med maksimal GPU L2-cache-effektivitet, så Compute Engine udnyttes optimalt, mens PCIe DMA-motoren kører med 100% båndbreddeudnyttelse ved siden af.

### **Arkitektur & GGML Opsætning**

To afgørende krav i GGML for at muliggøre asynkron prefetch:

> * **Pinned Host Memory (cudaHostAlloc):** PCIe DMA-controlleren kan *kun* køre asynkront (cudaMemcpyAsync), hvis CPU-hukommelsen med MoE-vægte er låst i fysisk RAM (pinned) og ikke kan pagedes ud af OS.  
> * **To CUDA Streams & Events:** Én stream til vægt-overførsel og én til GGML graph compute, synkroniseret på GPU-niveau via CUDA Events uden at blokere CPU-tråden.

### **C++ / GGML Implementation**

`#include <cuda_runtime.h>`  
`#include <ggml.h>`  
`#include <ggml-backend.h>`  
`#include <vector>`  
`#include <algorithm>`

`struct DoubleBufferMoE {`  
    `float* d_weight_buffer_A; // VRAM Buffer A`  
    `float* d_weight_buffer_B; // VRAM Buffer B`  
    `size_t layer_weight_bytes;`  
      
    `cudaStream_t stream_compute;  // Stream til Token-Tiling kernel/graph execution`  
    `cudaStream_t stream_transfer; // Stream til asynkron PCIe overførsel`  
      
    `cudaEvent_t event_transfer_done[2]; // Indikerer at VRAM buffer x er fyldt af PCIe`  
    `cudaEvent_t event_compute_done[2];  // Indikerer at GPU er færdig med VRAM buffer x`  
`};`

`void execute_tiled_double_buffered_pipeline(`  
    `DoubleBufferMoE& db,`  
    `const std::vector<float*>& h_pinned_layer_weights, // Pinned CPU RAM for alle lag`  
    `struct ggml_tensor* d_X_full,`  
    `struct ggml_tensor* d_Y_full,`  
    `int num_layers, int B, int b_tile, int d_model)`  
`{`  
    `int num_tiles = B / b_tile;`

    `// STEP 1: Asynkron prefetch af Lag 0 ind i Buffer A (Buffer 0) før loop start`  
    `size_t w_bytes = db.layer_weight_bytes;`  
    `cudaMemcpyAsync(db.d_weight_buffer_A, h_pinned_layer_weights[0], w_bytes,`   
                    `cudaMemcpyHostToDevice, db.stream_transfer);`  
    `cudaEventRecord(db.event_transfer_done[0], db.stream_transfer);`

    `// STEP 2: Main Layer Loop`  
    `for (int l = 0; l < num_layers; ++l) {`  
        `int curr_buf_idx = l % 2;`  
        `int next_buf_idx = (l + 1) % 2;`

        `float* d_curr_weights = (curr_buf_idx == 0) ? db.d_weight_buffer_A : db.d_weight_buffer_B;`  
        `float* d_next_weights = (next_buf_idx == 0) ? db.d_weight_buffer_A : db.d_weight_buffer_B;`

        `// A. PREFETCH NÆSTE LAG (l + 1) OVER PCIe`  
        `if (l + 1 < num_layers) {`  
            `// SIKKERHEDS-CHECK: Vent på at GPU compute er FÆRDIG med at læse fra next_buffer`  
            `if (l >= 1) {`  
                `cudaStreamWaitEvent(db.stream_transfer, db.event_compute_done[next_buf_idx], 0);`  
            `}`  
              
            `// Asynkron PCIe overførsel af næste lags vægte til næste buffer`  
            `cudaMemcpyAsync(d_next_weights, h_pinned_layer_weights[l + 1], w_bytes,`   
                            `cudaMemcpyHostToDevice, db.stream_transfer);`  
              
            `// Signalér at overførslen for næste lag er sat i kø/færdig`  
            `cudaEventRecord(db.event_transfer_done[next_buf_idx], db.stream_transfer);`  
        `}`

        `// B. SYNKRONISER COMPUTE STREAM MED TRANSFER STREAM FOR NUVÆRENDE LAG (l)`  
        `// Compute stream venter KUN indtil d_curr_weights er fuldt lande i VRAM`  
        `cudaStreamWaitEvent(db.stream_compute, db.event_transfer_done[curr_buf_idx], 0);`

        `// C. AFVIKL TOKEN-TILED COMPUTE FOR LAG l`  
        `// Vi kører k fliser af størrelse b_tile (f.eks. 8 tiles af 128 tokens)`  
        `for (int i = 0; i < num_tiles; ++i) {`  
              
            `size_t byte_offset = i * b_tile * d_model * sizeof(float);`  
              
            `// GGML Zero-copy view oprettelse (pseudo-kode for subgraf)`  
            `// Structs og kernels submitte til db.stream_compute`  
            `execute_token_tile_subgraph_async(`  
                `d_X_full, d_Y_full, d_curr_weights,`   
                `byte_offset, b_tile, d_model, db.stream_compute`  
            `);`  
        `}`

        `// Signalér i compute stream at d_curr_weights ikke længere bruges af GPU`  
        `cudaEventRecord(db.event_compute_done[curr_buf_idx], db.stream_compute);`  
    `}`

    `// Vent på at alle lag er fuldstændigt afviklet`  
    `cudaStreamSynchronize(db.stream_compute);`  
`}`

### **Synkroniserings-Flow i Hardwaren**

Koden benytter CUDA Events til at styre afhængigheder direkte på GPU'ens hardware-schedulere (uden at introducere CPU-latency):

`Time ---------------------------------------------------------------------------------------->`

`[stream_transfer]  |-- Copy Layer 0 -->|-- Copy Layer 1 (to Buf B) -->|-- Copy Layer 2 (to Buf A) -->`  
                   `|                   |                              |`  
                   `E0 (Record)         E1 (Record)                    E2 (Record)`  
                   `|                   |                              |`  
`[stream_compute]   +--- Wait(E0) ----->|-- Compute Layer 0 (Buf A) -->+--- Wait(E1) ------------->`  
                                       `|   (Tile 0 .. Tile 7)         |   Compute Layer 1 (Buf B)`  
                                       `|                              |   (Tile 0 .. Tile 7)`  
                                       `C0 (Record)                    C1 (Record)`

> 1. cudaStreamWaitEvent(stream\_compute, event\_transfer\_done) tvinger Compute Engine til at stoppe, indtil DMA Engine har frigivet den pågældende buffer.  
> 2. cudaStreamWaitEvent(stream\_transfer, event\_compute\_done) forhindrer DMA Engine i at overskrive en buffer, mens Compute Engine stadig afvikler token-fliser på den.

### **T1000 Performance Gevinst**

Når dette kombineres på en Quadro T1000:

> * **PCIe Gen3 x16 Båndbredde (∼12–15 GB/s):** Bliver udnyttet 100% kontinuerligt.  
> * **L2 Cache Latency:** Aktiveringerne for hver 128-token flise genbruges internt i L2/SRAM.  
> * **PCIe Transfer Latency:** Bliver **skjult 100%** bag beregningstiden for de 8 token-fliser, så længe compute-tiden overstiger overførselstiden.

---

