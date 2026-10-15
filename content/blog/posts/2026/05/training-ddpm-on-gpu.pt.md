---
title: "Treinando um Modelo de Difusão do Zero: Execução na GPU e Comunicação CPU-GPU"
date: 2026-05-31
description: "Uma análise orientada por medições de como o PyTorch move dados e coordena CPU e GPU durante o treino e o sampling de um DDPM no MNIST."
authors:
  - name: Luan Costa
---

Um loop de treino em PyTorch parece sequencial: carregar um batch, movê-lo para a GPU, executar o modelo, calcular os gradientes e atualizar os pesos. CUDA não executa dessa forma. Em geral, o Python enfileira o trabalho e continua; a GPU consome esse trabalho de forma assíncrona até que o host solicite um resultado que precisa ler.

Este post acompanha uma carga de trabalho real — um Denoising Diffusion Probabilistic Model (DDPM) treinado no MNIST — para mostrar onde os bytes se movem, onde o Python apenas submete trabalho e onde CPU e GPU precisam esperar uma pela outra. O DDPM não é, intencionalmente, o assunto principal. Ele fornece uma carga pequena e visual cujos caminhos completos de treino e sampling são fáceis de inspecionar.

Todas as medições foram coletadas em uma **NVIDIA GeForce RTX 5050 Laptop GPU** (Blackwell, 8 GB), usando PyTorch `2.12.0.dev20260407+cu128`, batch size 128 e quatro workers no DataLoader. O código e os scripts de benchmark estão disponíveis em [luanhcosta/ddpm-mnist](https://github.com/luanhcosta/ddpm-mnist).

---

## A Carga de Trabalho em Uma Equação

Durante o treino, uma imagem limpa $x_0$ é corrompida em um timestep aleatório $t$:

$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0,I)$$

Uma U-Net com 9,7 milhões de parâmetros recebe $x_t$ e $t$ e aprende a prever $\varepsilon$ usando uma loss MSE. Esse é todo o contexto necessário para a discussão de sistemas: cada step transfere um batch de 128 imagens em escala de cinza com 28×28 pixels e depois executa forward pass, backward pass e uma atualização Adam na GPU.

O modelo funciona — depois de 100 épocas, ele gera dígitos reconhecíveis —, mas aqui ele serve principalmente como instrumento para observar o pipeline de execução.

{{< img src="images/samples_final.png" alt="64 dígitos semelhantes aos do MNIST gerados pelo DDPM treinado" >}}

---

## O Mapa de Um Step de Treino

A distinção útil não é apenas entre “código de CPU” e “código de GPU”. Existem três eventos diferentes:

1. **Movimentação de dados:** bytes atravessam uma fronteira, como da RAM para a VRAM no batch de entrada ou da VRAM para a RAM em um escalar usado no logging.
2. **Submissão de trabalho:** o Python pede ao CUDA que execute uma operação. A chamada pode retornar antes de a GPU terminá-la.
3. **Sincronização:** a CPU precisa de um resultado da GPU e deve esperar até que o trabalho necessário seja concluído.

O caminho deste projeto se parece com isto:

```text
CPU / RAM                                      GPU / VRAM
─────────                                      ──────────
Workers do DataLoader
    │
    ├─ decodificam e transformam o MNIST
    └─ preparam batch no host ── cópia H2D ────► tensor de entrada
                                                   │
Python submete operações de timestep/ruído ───────► q_sample
Python submete operações do modelo ───────────────► forward + loss
Python submete operações do autograd ─────────────► backward
Python submete operações do otimizador ───────────► atualização Adam
                                                   │
loss.item() ◄──── cópia D2H do escalar + espera ───┘
```

As setas do Python para a GPU não significam que o Python espera depois de cada linha. As operações na mesma CUDA stream permanecem ordenadas, mas o host pode enfileirar várias operações antes que a GPU as conclua. Uma chamada como `loss.item()` muda essa relação porque o Python não consegue produzir o número até que o resultado exista na GPU.

---

## Memória Pinned: Cópia Mais Rápida, Não Overlap Automático

A memória comum do host é paginável: o sistema operacional pode realocar ou enviar suas páginas para swap. O CUDA não pode executar com segurança uma transferência DMA assíncrona diretamente de uma memória cuja localização física pode mudar, então o driver pode precisar primeiro copiar os dados para uma região page-locked.

A memória pinned é page-locked. Isso fornece ao driver CUDA uma origem estável para DMA e permite uma cópia H2D assíncrona. Em um DataLoader do PyTorch, `pin_memory=True` adiciona uma etapa no host que fixa a memória dos batches antes de retorná-los.

O benchmark de largura de banda usa o mesmo buffer pré-alocado na GPU e a mesma operação `copy_` para as duas origens. Cada caso é executado 30 vezes; o gráfico calcula a largura de banda a partir da latência mediana. A ordem alterna entre memória paginável e pinned para reduzir viés causado por clock e temperatura.

{{< img src="images/h2d_bandwidth.png" alt="Largura de banda host-to-device de memória paginável e pinned usando operações de cópia equivalentes" >}}

| Tamanho do tensor | Paginável | Pinned | Speedup com pinned |
|---|---:|---:|---:|
| 1 MB | 3,75 GB/s | 9,03 GB/s | 2,41× |
| 10 MB | 4,63 GB/s | 8,81 GB/s | 1,90× |
| 50 MB | 8,25 GB/s | 9,77 GB/s | 1,18× |
| 100 MB | 8,37 GB/s | 9,76 GB/s | 1,17× |
| 500 MB | 8,29 GB/s | 10,88 GB/s | 1,31× |

A memória pinned melhora claramente a cópia isolada. O ganho relativo é maior quando o custo fixo de staging é relevante e diminui à medida que os dois caminhos se aproximam da largura de banda sustentada.

O código de treino usa:

```python
x = x.to(device, non_blocking=True)
```

Para uma origem pinned, isso permite que a chamada no host retorne sem esperar a cópia terminar. Isso **não** significa, por si só, que a cópia se sobrepõe à computação da GPU. A cópia e as operações seguintes do modelo usam a default stream neste projeto, portanto permanecem ordenadas. A sobreposição com o batch anterior ou seguinte exige um pipeline explícito — normalmente prefetching ou double buffering, uma CUDA stream separada e sincronização correta entre streams.

---

## Microbenchmark versus Pipeline de Entrada

Uma versão anterior deste experimento se apresentava como benchmark de “throughput do DataLoader”, embora cronometrasse apenas `.to(device)`. O benchmark revisado separa duas perguntas:

- **Somente transferência:** o batch já foi produzido; mede-se apenas H2D.
- **Loop somente de entrada:** mede-se `next(iterator)` junto da cópia H2D e espera-se sua conclusão.

{{< img src="images/dataloader_comparison.png" alt="Comparação de memória pinned em uma transferência isolada e em um loop de entrada do DataLoader" >}}

Para o batch real do MNIST, a memória pinned reduziu a latência mediana da transferência isolada de **0,260 ms para 0,158 ms**. No loop deliberadamente limitado à entrada, porém, a mediana foi de **0,600 ms para memória paginável e 3,226 ms para pinned**.

Isso não é uma contradição. Fazer o pinning também exige trabalho no host, e esse loop quase não oferece computação com a qual esconder esse custo. O resultado também mostra por que um microbenchmark não deve ser apresentado como throughput end-to-end. O benefício para a aplicação depende do tamanho do batch, do comportamento dos workers, do prefetching e de quanto trabalho a GPU executa entre duas solicitações de batch.

A latência de cauda conta outra parte da história: nesta execução, o p95 foi de 28,58 ms para a entrada paginável e 17,54 ms para pinned. Uma única média ou taxa de throughput esconderia tanto o overhead quanto a variabilidade.

---

## Medindo CUDA sem Medir Apenas o Dispatch

Este padrão de medição normalmente está errado para trabalho na GPU:

```python
start = time.perf_counter()
y = model(x)
elapsed = time.perf_counter() - start
```

Ele pode medir quanto tempo o Python levou para enfileirar os kernels, e não quanto tempo os kernels levaram para executar. O benchmark delimita cada fase com sincronização:

```python
torch.cuda.synchronize()
start = time.perf_counter()
y = model(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

CUDA Events são outra boa opção para medir tempo no device. Aqui, a sincronização explícita é usada porque o objetivo é dividir um step completo em fases de wall-clock fáceis de interpretar.

O breakdown agora reproduz o caminho real do treino: AMP autocast, GradScaler, gradient clipping e Adam. A operação de noising do forward process está incluída, e o batch é transferido exatamente uma vez.

{{< img src="images/training_breakdown.png" alt="Breakdown de um step AMP mostrando H2D, noising, forward, loss, backward e otimizador" >}}

| Fase | Tempo | Participação |
|---|---:|---:|
| dados H2D | 0,134 ms | 0,1% |
| ruído + `q_sample` | 0,270 ms | 0,2% |
| forward | 47,099 ms | 33,4% |
| loss | 0,222 ms | 0,2% |
| backward | 87,023 ms | 61,8% |
| otimizador + clipping | 6,119 ms | 4,3% |
| **Total** | **140,867 ms** | **100%** |

Para esta carga de trabalho, H2D não é o gargalo do treino. Mesmo uma grande melhoria relativa em uma fase que representa 0,1% dificilmente altera o step completo. A computação do backward domina.

O total das fases sincronizadas serve para explicar onde o tempo do device é gasto; ele não significa que inserir uma sincronização entre cada fase seja a forma ideal de executar o treino.

---

## Observar a GPU pelo Host Também É Comunicação

O sampling expõe outra fronteira. Um DDPM realiza 1.000 avaliações dependentes do modelo. O Python pode continuar submetendo trabalho sem ler os tensores intermediários, mas código de monitoramento frequentemente materializa um valor no host:

```python
stat = x.mean().item()
```

Isso não mede uma “sincronização pura”. A operação adiciona uma redução na GPU, uma cópia D2H do escalar e a espera necessária para produzir um número Python. Essa combinação representa exatamente o que logging ou monitoramento de progresso pode acrescentar a um loop real.

O experimento revisado observa o tensor nunca, a cada 100 steps, a cada 10 steps ou em todos os steps. Ele executa um sampling completo de warmup, alterna a ordem dos casos ao longo de cinco rodadas e reporta a mediana.

{{< img src="images/sampling_comparison.png" alt="Tempo de sampling do DDPM para diferentes frequências de observação pelo host" >}}

| Frequência de observação | Tempo do DDPM-1000 |
|---|---:|
| nunca | 4,740 s |
| a cada 100 steps | 4,729 s |
| a cada 10 steps | 4,709 s |
| em todos os steps | 4,844 s |

Observar todos os steps adicionou aproximadamente **104 ms**, ou **2%**, em relação ao caso sem observação intermediária. As diferenças entre os casos intermediários estão dentro da variação normal entre execuções; o resultado importante é que até 1.000 observações no host custaram pouco diante de 1.000 avaliações da U-Net nesta máquina.

Esse resultado é específico desta carga. Kernels menores, um modelo mais rápido, estatísticas mais caras ou um serviço sensível a latência podem tornar o mesmo padrão de sincronização muito mais importante.

Para dar escala à comparação, DDIM com 50 avaliações do modelo levou **0,241 s**, um speedup de 19,7× sobre o baseline DDPM de 1.000 steps. Isso não é uma otimização de comunicação: ele remove 95% das avaliações do modelo. O resultado está incluído para mostrar a diferença entre otimizar overhead de coordenação e remover a computação dominante.

---

## Onde o Modelo Ainda Importa

O DDPM continua sendo útil porque seu loop de sampling é sequencial e sua saída é fácil de inspecionar. A animação abaixo captura oito tensores da GPU para a CPU a cada 100 steps para visualização — outra transferência device-to-host explícita que não existiria em um sampler de produção se não fosse solicitada.

{{< img src="images/denoising.gif" alt="Oito amostras progredindo de ruído Gaussiano para dígitos reconhecíveis" >}}

O modelo, a curva de loss, o checkpoint, as imagens geradas, o JSON bruto dos benchmarks e os scripts que regeneram os gráficos estão no repositório. A principal lição é mais ampla que modelos de difusão:

> Otimize o caminho completo de execução, não uma cópia isolada. Saiba se uma linha move bytes, submete trabalho ou força o host a esperar — e então meça a fronteira que realmente importa.

---

## Reproduzindo as Medições

```bash
python scripts/benchmark.py \
  --checkpoint results/checkpoints/epoch_100.pt \
  --assets-dir /caminho/para/blog/static/images
```

O comando registra os metadados do ambiente em `results/profiling/benchmarks.json`, gera os quatro gráficos de benchmark e exporta um trace do PyTorch Profiler. Implementação completa: [luanhcosta/ddpm-mnist](https://github.com/luanhcosta/ddpm-mnist).
