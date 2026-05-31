---
title: "Escrevendo seus Primeiros Kernels CUDA: Execução na GPU, Memória e Otimização"
date: 2026-05-31
description: "Um guia completo de programação CUDA do zero — indexação de threads, adição de vetores, multiplicação de matrizes com tiling em shared memory, atomics e streams assíncronos. Desenvolvido com RTX 5050 e CUDA 12.6."
authors:
  - name: Luan Costa
---

O deep learning moderno, simulações físicas e computação científica estão construídos sobre uma constatação: alguns problemas são embaraçosamente paralelos, e a GPU é o hardware que foi feito para explorar isso. Mas para escrever código de GPU de verdade, você precisa entender um modelo de programação que é fundamentalmente diferente de tudo que existe na CPU.

Este post é um guia completo de CUDA do zero. Vamos cobrir o modelo de execução da GPU, indexação de threads em 1D/2D/3D, o padrão de gerenciamento de memória, adição de vetores e multiplicação de matrizes, tiling com shared memory, race conditions e atomics, e por fim execução concorrente com streams. Cada conceito está ancorado em código que funciona de verdade.

Os exemplos foram escritos e testados em uma **NVIDIA GeForce RTX 5050** com **CUDA 12.6**.

---

## O Modelo de Execução da GPU: Por que É Diferente

Antes de tocar em uma única linha de código CUDA, vale a pena entender por que a GPU pensa do jeito que pensa. A CPU e a GPU foram projetadas em torno de filosofias opostas:

| | CPU | GPU |
|---|---|---|
| Núcleos | Poucos (8–32) | Milhares (RTX 5050: ~2.560) |
| Estratégia | Baixa latência por thread | Alto throughput com muitos threads |
| Contexto de thread | Pesado (salva estado completo na troca) | Levíssimo (troca em 1 ciclo) |
| Carga ideal | Lógica serial, branches complexos | Operações massivamente paralelas e uniformes |

A CPU é otimizada para rodar um número pequeno de threads o mais rápido possível. A GPU faz a aposta oposta: executar um número enorme de threads simultaneamente, e esconder a latência de operações lentas de memória trocando para outra thread enquanto uma está esperando dados. Isso se chama **hiding latency through occupancy**, e só funciona se você der à GPU threads suficientes para se manter ocupada.

É por isso que kernels CUDA lançam milhões de threads. A GPU não está apenas oferecendo paralelismo — ela precisa do volume de trabalho para funcionar com eficiência.

### A Hierarquia Grid → Block → Thread

Todo lançamento de kernel CUDA cria uma hierarquia de três níveis de threads:

```
┌─────────────────────────────────────────────┐
│                    GRID                      │
│  (todos os blocos de um lançamento de kernel)│
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

- **Thread**: a menor unidade de execução. Cada thread roda o código do kernel de forma independente, e crucialmente, cada thread tem uma combinação única de `threadIdx` e `blockIdx` que permite determinar qual pedaço de dado processar.
- **Block (Bloco)**: um grupo de threads que compartilha **memória compartilhada** (`__shared__`) e pode se sincronizar com `__syncthreads()`. Um bloco inteiro roda em um único **Streaming Multiprocessor (SM)** — um bloco nunca se divide entre SMs.
- **Grid**: todos os blocos de um lançamento. Blocos em partes diferentes do grid **não** se comunicam entre si, exceto através da memória global.

### Streaming Multiprocessors (SMs)

Cada SM é uma unidade de processamento independente dentro da GPU. A RTX 5050 tem múltiplos SMs, cada um capaz de rodar dezenas de blocos simultaneamente. O scheduler da GPU distribui os blocos pelos SMs disponíveis de forma automática — você não controla esse mapeamento, e isso é intencional.

A independência entre blocos é o que torna o CUDA escalável: o mesmo kernel roda igualmente bem em uma GPU com 10 SMs ou com 1.000 SMs. Mais SMs significa mais blocos executando ao mesmo tempo, mas o resultado é idêntico.

### Warps: A Unidade Real de Execução

Dentro de um bloco, as threads são agrupadas em **warps de 32 threads**. Todos os threads de um warp executam a **mesma instrução ao mesmo tempo** — esse é o modelo SIMT (Single Instruction, Multiple Threads).

Isso tem uma consequência importante: se threads de um mesmo warp divergem em um `if/else`, a GPU executa os dois ramos *sequencialmente*, desativando as threads que não pertencem a cada ramo. Isso se chama **divergência de warp** e é uma fonte comum de ineficiência.

A regra prática: **sempre escolha tamanhos de bloco que sejam múltiplos de 32** (256, 512 ou 1024 são escolhas típicas). Qualquer outro valor deixa warps parciais que desperdiçam slots de execução.

---

## Indexação de Threads

Para que uma thread saiba qual elemento processar, ela precisa calcular um ID globalmente único. O CUDA fornece quatro variáveis built-in para isso:

| Variável | Significado |
|---|---|
| `threadIdx.{x,y,z}` | Posição do thread dentro do seu bloco |
| `blockIdx.{x,y,z}` | Posição do bloco dentro do grid |
| `blockDim.{x,y,z}` | Tamanho do bloco em cada dimensão |
| `gridDim.{x,y,z}` | Tamanho do grid em cada dimensão |

### O Kernel `whoami`

A forma clássica de entender a indexação é o kernel `whoami`, que computa o ID global de cada thread e o imprime:

```c
__global__ void whoami(void) {
    // Passo 1: calcular o ID LINEAR do bloco no grid 3D
    int block_id =
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;

    // Passo 2: quantas threads existem antes do nosso bloco?
    int block_offset =
        block_id * blockDim.x * blockDim.y * blockDim.z;

    // Passo 3: calcular o ID LINEAR do thread dentro do bloco
    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    // Passo 4: ID global = threads antes de mim + minha posição local
    int id = block_offset + thread_offset;
}
```

A matemática do Passo 1 é exatamente o que o C usa para linearizar um índice de array 3D `(x, y, z)` em uma posição plana: `x + y*dimX + z*dimX*dimY`. Você está fazendo o mesmo — encontrando um índice único a partir de uma coordenada 3D.

A configuração de lançamento para esse kernel:

```c
const int b_x = 2, b_y = 3, b_z = 4;   // grid: 2×3×4 = 24 blocos
const int t_x = 4, t_y = 4, t_z = 4;   // bloco: 4×4×4 = 64 threads por bloco

// Total: 24 × 64 = 1.536 threads rodando em paralelo
dim3 blocksPerGrid(b_x, b_y, b_z);
dim3 threadsPerBlock(t_x, t_y, t_z);
whoami<<<blocksPerGrid, threadsPerBlock>>>();
```

A sintaxe `<<<blocksPerGrid, threadsPerBlock>>>` é a **configuração de lançamento** — é onde você diz à GPU quantas threads criar.

**Experimente:** rode o kernel e passe a saída por `sort`:
```bash
nvcc -o idxing 01_idxing.cu
./idxing | sort -n | head -20   # ordenado por ID
./idxing | head -20             # ordem real de execução
```

Você vai ver que a saída não está em ordem. O scheduler da GPU atribui blocos a SMs com base na disponibilidade — não há ordenação garantida entre blocos. Isso é intencional e esperado.

### Indexação 1D — O Caso do Dia a Dia

Na prática, a maioria dos kernels usa indexação 1D:

```c
__global__ void kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = ...; // processa o elemento i
    }
}
```

O `if (i < n)` é chamado de **guard** e é essencial. Quando `n` não é divisível por `blockDim.x`, o último bloco tem threads que apontariam para além do final do array. O guard previne acesso fora dos limites.

---

## O Padrão de Gerenciamento de Memória CUDA

Todo programa CUDA segue o mesmo esqueleto para mover dados entre CPU e GPU:

```
CPU (Host)                          GPU (Device)
─────────                          ────────────
1. Aloca memória no host
2. Inicializa os dados
3. cudaMalloc() ──────────────────► Aloca VRAM
4. cudaMemcpy(H→D) ───────────────► Copia dados para a VRAM
5. kernel<<<grid, block>>>() ─────► Executa o kernel
6. cudaMemcpy(D→H) ◄──────────────  Copia resultado de volta
7. Usa os resultados
8. free() + cudaFree()
```

Esse padrão aparece em todos os exemplos a seguir. A abstração central é que a GPU tem sua própria memória (VRAM) separada da RAM, e todos os dados precisam ser explicitamente transferidos entre elas antes e depois de um kernel executar.

---

## Adição de Vetores: O "Hello World" da GPU

Somar dois vetores elemento a elemento é o kernel paralelo mais simples — cada elemento é independente, então cada thread pode trabalhar no seu par sem nunca precisar falar com outra thread.

### Referência na CPU

```c
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

A CPU processa 10 milhões de somas em série. Mesmo com auto-vetorização SIMD, isso é fundamentalmente sequencial no nível do loop.

### Kernel na GPU

```c
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

Com `BLOCK_SIZE = 256` e `N = 10.000.000`:

```
num_blocks = (10.000.000 + 256 - 1) / 256 = 39.063 blocos
```

A GPU cria ~10 milhões de threads. Cada um soma um único par de elementos.

### Divisão com Arredondamento Para Cima

A fórmula `(N + BLOCK_SIZE - 1) / BLOCK_SIZE` é a **ceiling division** (divisão arredondada para cima):

```
N = 1.000, BLOCK_SIZE = 256
Sem arredondamento: 1000 / 256 = 3 blocos → cobre 768 elementos (faltam 232!)
Com arredondamento: (1000 + 255) / 256 = 4 blocos → cobre 1.024 elementos
                                         ↑ o guard protege os extras
```

### Execuções de Aquecimento (Warm-up)

Ao fazer benchmark de código GPU, sempre rode algumas **iterações de aquecimento** antes de medir o tempo. A primeira execução do kernel tem overhead de:
- Compilação JIT do código GPU
- A GPU saindo de um estado de baixo consumo (P-state) para clock máximo
- Caches frios causando latências artificialmente altas

```bash
nvcc -O2 -o vec_add 00_vector_add_v1.cu
./vec_add
```

### Uma Nota sobre Grids 3D para Problemas 1D

É perfeitamente válido organizar threads em um grid 3D mesmo para um problema 1D — o kernel simplesmente lineariza o índice 3D de volta para um índice plano de array:

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

Esse padrão se torna natural ao trabalhar com dados 3D reais (simulações volumétricas, imagens médicas 3D, operações com tensores). Para um problema de vetor simples, a versão 1D é mais rápida porque a 3D executa mais aritmética por thread para calcular o índice.

---

## Multiplicação de Matrizes: O Kernel Mais Importante

A multiplicação de matrizes é o kernel individualmente mais importante da computação moderna. Ela sustenta o treinamento de redes neurais, visão computacional e simulação científica. Entender suas características de desempenho é fundamental.

Para `C = A × B` onde `A` é `(M × K)` e `B` é `(K × N)`:

$$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]$$

### Kernel Ingênuo na GPU

A abordagem direta: uma thread por elemento de saída.

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

O grid é **2D** aqui porque a matriz de saída C é 2D. O bloco é tipicamente `32×32 = 1.024 threads` (o máximo por bloco no hardware moderno da NVIDIA).

Configuração de lançamento para `M×K × K×N`:
```c
dim3 block(32, 32);
dim3 grid((N + 31) / 32, (M + 31) / 32);
matmul_gpu<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
```

### O Gargalo: Largura de Banda da Memória Global

O kernel ingênuo funciona, mas é limitado pela memória. Para uma matriz `1024×1024`:
- Cada thread faz `K = 1.024` leituras da memória global de A e 1.024 de B
- Existem `1.024 × 1.024 = 1.048.576` threads no total
- **Total de leituras da VRAM: ~2 bilhões**

Pior ainda, as leituras são redundantes. A thread `(0,0)` e a thread `(0,1)` leem a linha inteira `A[0][0..K-1]`. Essa linha é lida **N vezes**, uma por coluna de C. O mesmo dado é buscado da VRAM repetidamente.

Esse não é um gargalo de computação — é um **gargalo de largura de banda de memória**. As unidades aritméticas da GPU ficam ociosas, esperando os dados chegarem da VRAM.

---

## Otimizando com Shared Memory

### A Hierarquia de Memória da GPU

Antes da otimização, entenda onde os dados vivem:

```
┌──────────────────────────────────────────────────┐
│  Registradores (por thread)                       │
│  Latência: ~1 ciclo | Tamanho: ~255 por thread    │
├──────────────────────────────────────────────────┤
│  Shared Memory / L1 Cache (por bloco)             │
│  Latência: ~5–30 ciclos | Tamanho: 48–96 KB/SM   │
├──────────────────────────────────────────────────┤
│  L2 Cache (por GPU)                               │
│  Latência: ~100 ciclos | Tamanho: vários MB       │
├──────────────────────────────────────────────────┤
│  Memória Global / VRAM                            │
│  Latência: ~200–800 ciclos | Tamanho: 8 GB        │
└──────────────────────────────────────────────────┘
```

Acessar a memória global é **200–800× mais lento** que registradores. A estratégia central de otimização em CUDA é minimizar o tráfego de memória global usando a shared memory como um cache gerenciado manualmente.

### Multiplicação de Matrizes com Tiling

A ideia central: em vez de cada thread buscar seus dados independentemente da VRAM, o bloco inteiro coopera para carregar um **tile** de A e um tile de B para a shared memory, e então todas as threads calculam usando essa cópia local e rápida.

```
Matriz A (M×K)              Matriz B (K×N)
┌─────────────────┐        ┌─────────────────┐
│ tile │ tile │..│        │tile│tile│tile│..│
│  A0  │  A1  │  │        │ B0 │ B1 │ B2 │  │
└──────┴──────┴──┘        └────┴────┴────┴──┘

Para calcular um tile de C:
C_tile += A_tile0 × B_tile0  +  A_tile1 × B_tile1  + ...
            ↑ carregado na shared memory, um tile por vez
```

Aqui está o kernel completo com tiling:

```c
#define TILE_SIZE 16

__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;  // fica em registrador — rápido!

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {

        // === FASE 1: Carga colaborativa para a shared memory ===
        // Cada thread carrega UM elemento de A e UM de B
        if (row < M && tile * TILE_SIZE + tx < K)
            sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;

        if (col < N && tile * TILE_SIZE + ty < K)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            sharedB[ty][tx] = 0.0f;

        // === BARREIRA: todas as threads devem terminar o carregamento antes de computar ===
        __syncthreads();

        // === FASE 2: Computação usando a shared memory (rápida) ===
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sharedA[ty][k] * sharedB[k][tx];

        // === BARREIRA: todas as threads devem terminar de computar antes do próximo tile ===
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### Por que Dois `__syncthreads()`?

Cada barreira serve a um propósito distinto:

**Primeira barreira** — após o carregamento, antes de computar. Sem ela, uma thread rápida poderia começar a ler `sharedA[ty][k]` antes de uma thread lenta ter escrito esse valor. Você leria lixo.

**Segunda barreira** — após computar, antes do carregamento do próximo tile. Sem ela, uma thread rápida poderia começar a sobrescrever a shared memory com o próximo tile enquanto uma thread lenta ainda está lendo os valores do tile atual.

O diagrama de tempo deixa o modo de falha concreto:

```
Thread A (rápida)       Thread B (lenta)
────────────────        ─────────────────
carrega sharedA[0][0]   ... ainda computando o tile anterior ...
                          ← __syncthreads() ← Thread A espera aqui
                        carrega sharedA[0][0]  (sobrescreveria!)
computa ...             computa ...
```

### A Redução no Tráfego de Memória

Com `TILE_SIZE = 16`, um bloco de 16×16 threads carrega:
- 256 floats de A (um tile)
- 256 floats de B (um tile)

Cada float na shared memory é lido **16 vezes** (uma vez por thread na outra dimensão), em vez de uma vez. O número de acessos à memória global cai de `2K` leituras por thread para `2K/TILE_SIZE` — uma **redução de 16× no tráfego de VRAM**.

Você pode medir isso diretamente com o profiler da NVIDIA:

```bash
# Mede bytes lidos da memória global
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum ./naive_matmul
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum ./tiled_matmul
```

A diferença é dramática. O kernel ingênuo é gargalado pela largura de banda de memória; o kernel com tiling move muito mais da computação para o regime da shared memory rápida.

---

## Atomics e Race Conditions

### O Problema

Quando múltiplas threads escrevem na mesma posição de memória simultaneamente, o resultado é indefinido. Isso é uma **race condition**.

Considere um kernel que incrementa um contador compartilhado:

```c
__global__ void incrementNaoAtomico(int *counter) {
    int old = *counter;        // lê o valor atual
    int new_val = old + 1;     // incrementa localmente
    *counter = new_val;        // escreve de volta
}
```

Essas três operações parecem sequenciais, mas não são atômicas no hardware. Com 1.000.000 de threads executando simultaneamente:

```
Tempo  Thread A              Thread B
──────────────────────────────────────
  1    lê counter = 42       lê counter = 42
  2    calcula 43            calcula 43
  3    escreve 43            escreve 43

Dois incrementos aconteceram, mas o counter só subiu 1.
Isso é o "problema do update perdido".
```

Rode isso com `NUM_BLOCKS = 1000` e `NUM_THREADS = 1000`: o resultado esperado é 1.000.000, mas a versão não-atômica produz um número aleatório menor — e diferente a cada execução.

### A Solução: `atomicAdd`

```c
__global__ void incrementAtomico(int *counter) {
    atomicAdd(counter, 1);
}
```

`atomicAdd` é uma instrução de hardware que executa leitura-modificação-escrita como uma unidade indivisível. Se 32 threads de um warp tentam `atomicAdd` no mesmo endereço, o hardware as serializa: todos os 32 incrementos acontecem, um de cada vez, e cada atualização é preservada.

### O Toolkit Completo de Atomics

| Função | Operação |
|---|---|
| `atomicAdd(addr, val)` | `*addr += val` |
| `atomicSub(addr, val)` | `*addr -= val` |
| `atomicExch(addr, val)` | troca: retorna o antigo, escreve `val` |
| `atomicMin(addr, val)` | `*addr = min(*addr, val)` |
| `atomicMax(addr, val)` | `*addr = max(*addr, val)` |
| `atomicCAS(addr, cmp, val)` | Compare-And-Swap: se `*addr == cmp`, escreve `val` |
| `atomicAnd/Or/Xor` | operações bit a bit |

### Quando Atomics Prejudicam o Desempenho

Atomics serializam acessos conflitantes. Se muitas threads competem pelo mesmo endereço (**alta contenção**), o desempenho cai drasticamente — as threads ficam na fila esperando sua vez.

O padrão eficiente para reduções é reduzir dentro do bloco primeiro usando shared memory, depois usar um único atomic por bloco para combinar com o resultado global:

```c
__global__ void reduceSum(float *data, float *result, int n) {
    __shared__ float partial[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    partial[threadIdx.x] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    // Redução em árvore dentro do bloco
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    // Apenas a thread 0 de cada bloco acessa a memória global
    if (threadIdx.x == 0)
        atomicAdd(result, partial[0]);
}
```

Isso reduz o número de operações atômicas de `N` (uma por thread) para `num_blocks` (uma por bloco). Para um milhão de elementos com tamanho de bloco 256, isso é uma redução de 1.000.000 para ~3.906 operações atômicas.

---

## Streams e Execução Assíncrona

### O que é um CUDA Stream?

Um **CUDA stream** é uma fila de operações de GPU que executam em ordem dentro do stream, mas streams diferentes podem executar **em paralelo entre si**. Isso permite sobrepor:
- Transferências de memória host-para-device
- Transferências device-para-host
- Execução de kernels

Sem streams, tudo vai para o stream padrão e executa sequencialmente:

```
Sem streams:
──────────────────────────────────────────────────►
[  cópia H2D  ] [  kernel  ] [  cópia D2H  ]

Com streams:
stream1: [  cópia H2D A  ] [  kernel A  ] [  cópia D2H A  ]
stream2:      [  cópia H2D B  ] [  kernel B  ] [  cópia D2H B  ]
         ──────────────────────────────────────────────────►
         tempo total reduzido!
```

### Requisito: Pinned Memory

Transferências assíncronas requerem **pinned memory (memória fixada/pinada)** no host. A memória normal do `malloc` pode ser movida pelo SO a qualquer momento, o que quebra transferências DMA. Pinned memory é travada no lugar:

```c
// Memória paginável normal — transferências async não são garantidas
float *h_data = (float*)malloc(size);

// Pinned memory — DMA verdadeiramente assíncrono
float *h_data;
cudaMallocHost(&h_data, size);
// ... usa h_data ...
cudaFreeHost(h_data);
```

O trade-off: pinned memory não pode ser swapped, então uso excessivo pode prejudicar o desempenho geral do sistema. Use-a apenas para buffers de transferência onde o comportamento assíncrono realmente importa.

### Fundamentos de Streams

```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Copia A e B em paralelo — streams diferentes!
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2);

// Precisa esperar d_B antes de lançar um kernel que o usa
cudaStreamSynchronize(stream2);

// Kernel no stream1
vectorAdd<<<blocks, threads, 0, stream1>>>(d_A, d_B, d_C, n);

// Copia resultado de volta
cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

O `cudaStreamSynchronize(stream2)` antes do lançamento do kernel é necessário porque o kernel usa `d_B`, que está sendo copiado em stream2. Streams diferentes não têm dependência de ordem implícita — se você não sincronizar explicitamente, o kernel pode iniciar antes que a cópia termine.

### Events: Sincronização Precisa Entre Streams

Events permitem expressar dependências mais precisas sem bloquear a CPU:

```c
cudaEvent_t event;
cudaEventCreate(&event);

// Stream1 faz seu trabalho
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1);
kernel1<<<grid, block, 0, stream1>>>(d_data, N);

// Marca o ponto no stream1 onde o evento dispara
cudaEventRecord(event, stream1);

// Stream2 espera até o stream1 chegar no evento — a GPU gerencia isso
cudaStreamWaitEvent(stream2, event, 0);

// Seguro usar dados produzidos pelo stream1
kernel2<<<grid, block, 0, stream2>>>(d_data, N);
```

`cudaStreamWaitEvent` é mais eficiente que `cudaStreamSynchronize` porque a CPU não bloqueia — o scheduler da GPU gerencia a dependência internamente.

### Prioridades de Stream

Para cargas de trabalho onde algumas operações são mais sensíveis ao tempo:

```c
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

cudaStreamCreateWithPriority(&high_stream, cudaStreamNonBlocking, greatestPriority);
cudaStreamCreateWithPriority(&low_stream, cudaStreamNonBlocking, leastPriority);
```

Sob carga, o scheduler da GPU prefere operações em streams de maior prioridade. Útil para garantir que trabalho sensível à latência (decodificação de frames em tempo real, inferência interativa) não seja atrasado por processamento em batch.

`cudaStreamNonBlocking` significa que o stream não bloqueia e não é bloqueado pelo stream padrão (stream 0).

### Callbacks de Stream: A GPU Notificando a CPU

```c
void CUDART_CB aoTerminar(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Trabalho da GPU completo, notificando a CPU\n");
    // acordar uma thread da CPU, sinalizar um semáforo, enfileirar mais trabalho, etc.
}

cudaStreamAddCallback(stream, aoTerminar, NULL, 0);
```

O callback é executado na CPU quando a GPU chega a esse ponto no stream. Isso habilita pipelines produtor-consumidor onde a CPU precisa saber quando um batch de trabalho da GPU terminou para começar a preparar o próximo.

---

## Referência: Hierarquia de Memória

| Tipo | Escopo | Latência | Declaração | Caso de Uso |
|---|---|---|---|---|
| Registradores | por thread | ~1 ciclo | variáveis locais | variáveis de loop, acumuladores |
| Shared Memory | por bloco | ~5–30 ciclos | `__shared__ float arr[N]` | tiles, comunicação entre threads do bloco |
| Memória Global | GPU toda | ~200–800 ciclos | `cudaMalloc` | dados principais, input/output |
| Memória Constante | GPU toda (read-only) | ~5 ciclos (com cache) | `__constant__ float arr[N]` | parâmetros fixos (pesos de filtro, etc.) |
| Memória de Textura | GPU toda (read-only) | ~5 ciclos (com cache) | `cudaBindTexture` | dados com localidade espacial 2D |
| Pinned Memory | host | N/A | `cudaMallocHost` | buffers de transferência async |

---

## Próximos Passos

Isso cobre a camada fundamental da programação CUDA. Os próximos passos naturais:

**cuBLAS** — a biblioteca de álgebra linear production-grade da NVIDIA. Rode `cublasSgemm` nas mesmas matrizes e compare com seu kernel com tiling — será notavelmente mais rápido porque usa todos os truques de hardware (loads vetorizados, tensor cores, evitar bank conflicts) que levam meses para dominar.

**Occupancy** — aprenda a calcular a ocupância do SM (a razão entre warps ativos e o máximo teórico) e como ela se relaciona com o throughput. Uso de shared memory, contagem de registradores e tamanho do bloco afetam a ocupância, e otimizá-la é frequentemente a alavanca que fecha a diferença com o cuBLAS.

**Acesso Coalescido à Memória** — os padrões de acesso que maximizam a largura de banda da memória global. O exemplo canônico é a transposição de matrizes: uma implementação ingênua lê linhas (coalescido) mas escreve colunas (strided, lento). Entender coalescing é essencial antes de otimizar qualquer coisa limitada por memória.

**Triton** — uma linguagem embutida em Python para escrever kernels de GPU em um nível mais alto de abstração, que cuida automaticamente de tiling e shared memory. Triton é o que a maioria dos praticantes de ML usa hoje. Mas ele gera CUDA por baixo dos panos, e entender o que está neste post é o pré-requisito para avaliar se a saída do Triton é boa ou não.
