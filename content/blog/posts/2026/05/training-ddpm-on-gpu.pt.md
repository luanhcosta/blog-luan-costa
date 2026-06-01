---
title: "Treinando um Modelo de Difusão do Zero: Execução na GPU e Comunicação CPU-GPU"
date: 2026-05-31
description: "Uma análise completa do treinamento de um DDPM no MNIST do zero — cobrindo a matemática da difusão, a arquitetura U-Net, e cada ponto onde CPU e GPU trocam dados durante o treino e a inferência. Todos os benchmarks rodados em uma RTX 5050."
authors:
  - name: Luan Costa
---

Modelos generativos estão entre as cargas de trabalho mais intensas em computação no deep learning. Entender não apenas se um modelo treina, mas *como* o hardware está sendo usado durante o treino, é o que separa código que funciona de código que roda bem.

Este post documenta a implementação completa de um Denoising Diffusion Probabilistic Model (DDPM) treinado no MNIST do zero, com foco específico na comunicação CPU-GPU: onde os dados se movem, quando CPU e GPU se sincronizam, e quais são os gargalos reais. Todas as medições foram coletadas em uma **NVIDIA GeForce RTX 5050 Laptop GPU** (Blackwell, 8 GB). O código completo está em [luanhcosta/ddpm-mnist](https://github.com/luanhcosta/ddpm-mnist).

---

## O Processo Forward

O DDPM enquadra a geração de imagens como a reversão de um processo de adição de ruído. O processo forward $q(x_t \mid x_0)$ corrompe progressivamente uma imagem limpa $x_0$ adicionando ruído gaussiano ao longo de $T$ timesteps:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

onde $\bar{\alpha}_t$ é o produto cumulativo do noise schedule:

$$\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$$ Com um schedule linear de $\beta_1 = 10^{-4}$ a $\beta_T = 0.02$ ao longo de $T = 1000$ steps, o schedule se comporta assim:

{{< img src="images/noise_schedule.png" alt="DDPM Linear Noise Schedule — β, ᾱ, amplitude de sinal/ruído e SNR" >}}

Três observações desse gráfico que informam tanto o treino quanto a inferência:

- As amplitudes de sinal e ruído se cruzam em **t ≈ 258** — menos de 26% do processo. Após esse ponto, o ruído já domina a imagem.
- $\bar{\alpha}_t$ decai em forma de S, não linearmente. A maior parte do sinal é perdida nos primeiros 300 steps.
- O SNR (canto inferior direito, escala log) colapsa exponencialmente. Em t = 500 ele já caiu três ordens de grandeza em relação ao valor inicial.

O efeito sobre um dígito real do MNIST:

{{< img src="images/noising_trajectory.png" alt="Processo forward: um dígito 2 sendo progressivamente ruidizado em t=0, 100, 250, 500, 750, 999" >}}

Em t = 250 a forma ainda é fracamente visível — coerente com o cruzamento do schedule em t ≈ 258. Em t = 500 o dígito já é irreconhecível, e em t = 999 é indistinguível de ruído gaussiano puro.

---

## O Modelo: U-Net com Condicionamento Temporal

O processo reverso $p_\theta(x_{t-1} \mid x_t)$ é parametrizado por uma U-Net que prediz o ruído $\varepsilon$ adicionado no timestep $t$, treinada com uma loss MSE simples:

$$\mathcal{L} = \mathbb{E}_{x_0,\, t,\, \varepsilon}\left[\lVert\varepsilon - \varepsilon_\theta(x_t, t)\rVert^2\right]$$

A arquitetura para imagens 28×28 usa três níveis de resolução com um bottleneck em 7×7:

| Estágio | Resolução | Canais | Notas |
|---|---|---|---|
| Encoder L0 | 28×28 | 64 | 2× ResBlock |
| Encoder L1 | 14×14 | 128 | 2× ResBlock |
| Encoder L2 | 7×7 | 256 | 2× ResBlock + SelfAttention |
| Bottleneck | 7×7 | 256 | ResBlock + SelfAttention + ResBlock |
| Decoder L2→1 | 14×14 | 128 | upsample + 2× ResBlock |
| Decoder L1→0 | 28×28 | 64 | upsample + 2× ResBlock |

Cada ResBlock recebe o embedding de timestep — uma codificação sinusoidal de $t$ projetada para 256 dimensões via MLP — somado ao feature map após a primeira convolução. É assim que o modelo distingue "denoise a partir de t=900" de "denoise a partir de t=50". Total de parâmetros: **9.744.641**.

---

## Comunicação CPU-GPU Durante o Treino

Cada step de treino segue o pipeline padrão: carregar batch da CPU, transferir para a GPU, rodar o forward pass, calcular a loss, rodar o backward pass, atualizar os pesos. A questão é onde o tempo realmente vai.

### Largura de Banda de Transferência Host-para-Device

O conselho padrão é usar `pin_memory=True` no DataLoader para que os tensores do host sejam fixados na memória antes de serem transferidos via DMA. O ganho de bandwidth real depende muito do tamanho do tensor:

{{< img src="images/h2d_bandwidth.png" alt="Bandwidth H2D: memória pageable vs pinned em diferentes tamanhos de tensor" >}}

Em **1 MB**, memória pinned é **2.5× mais rápida** (17.3 vs 6.9 GB/s) — o overhead do SO para paginar domina em transferências pequenas. Em **10 MB**, a diferença quase desaparece (10.6 vs 10.9 GB/s). Em **500 MB**, pinned recupera uma vantagem de 1.24× à medida que a eficiência sustentada do DMA volta a importar.

Para batches do MNIST (batch size 128, imagens 28×28 float32 = ~400 KB por batch), estamos firmemente no regime de tensor pequeno onde memória pinned tem mais impacto.

### Throughput do DataLoader

{{< img src="images/dataloader_comparison.png" alt="Throughput do DataLoader: pin_memory=True vs False" >}}

`pin_memory=True` entrega **1.78× mais throughput**: 14.662 vs 8.292 batches/segundo, ou 0,068 vs 0,121 ms por batch. Para um loop de treino que processa ~460 batches por época, isso economiza ~24 ms por época — modesto isoladamente, mas gratuito.

### Onde o Tempo Realmente Vai em um Step de Treino

{{< img src="images/training_breakdown.png" alt="Breakdown do step de treino: data H2D, forward, loss, backward, optimizer" >}}

Esta é a medição mais reveladora. Com média de 100 steps com o modelo treinado:

| Fase | Tempo | Parcela |
|---|---|---|
| data H2D | 0,08 ms | **0,0%** |
| amostragem de ruído + timestep | 0,05 ms | 0,0% |
| forward pass | 70,1 ms | 32,8% |
| loss | 0,09 ms | 0,0% |
| backward pass | 140,4 ms | **65,8%** |
| optimizer step | 2,7 ms | 1,3% |
| **Total** | **213,5 ms** | |

A transferência de dados CPU→GPU é **três ordens de grandeza menor** que o backward pass. Para um modelo de ~10M parâmetros em imagens 28×28, o gargalo é o cálculo do gradiente — não a comunicação de memória. Otimizar o carregamento de dados aqui seria otimizar os 0,0% ignorando os 65,8%.

---

## Resultados do Treino

100 épocas, Adam com lr=2e-4, batch size 128, AMP (float16), gradient clipping em 1.0.

{{< img src="images/loss_curve.png" alt="Loss de treino do DDPM ao longo de 47.000 steps" >}}

A loss cai abruptamente nos primeiros ~2.000 steps e depois entra em uma longa fase de refinamento estável. Nenhuma instabilidade — uma das vantagens estruturais do DDPM sobre GANs: há apenas uma rede para treinar, e o landscape da loss MSE é bem comportado. O modelo atingiu sua loss final de ~0,02 e permaneceu lá pelas últimas 80 épocas.

Após 100 épocas, amostrando 64 imagens a partir de ruído gaussiano puro:

{{< img src="images/samples_final.png" alt="64 imagens geradas pelo DDPM treinado a partir de ruído puro" >}}

Todas as 10 classes de dígitos aparecem com traços limpos e sem mode collapse. Algumas amostras são ambíguas — o modelo aprendeu a distribuição completa de estilos de escrita, não apenas formas canônicas.

---

## Inferência: O Custo de Comunicação de 1000 Steps

Gerar uma imagem com DDPM requer 1000 avaliações sequenciais do modelo — uma por timestep reverso. Isso cria uma questão natural: qual é o custo de sincronizar CPU e GPU a cada step?

Em código, qualquer operação Python em um tensor GPU que retorna um escalar Python — como `x.mean().item()` — força um `cudaDeviceSynchronize()`. A CPU bloqueia até a GPU terminar aquele kernel antes do interpretador Python continuar. Código de monitoramento ingênuo dentro do loop de denoising dispara isso a cada um dos 1000 steps.

{{< img src="images/sampling_comparison.png" alt="Tempo de sampling: DDPM-1000 com syncs forçados vs GPU pura, vs DDIM-50" >}}

O resultado para esse modelo: o overhead de sync é **1,04×** — 3,94s vs 3,78s. As 1000 chamadas a `.item()` adicionam cerca de 160 ms ao longo do loop completo.

Isso é menor do que o esperado, e o motivo é informativo: o forward pass da U-Net em uma entrada 28×28 leva ~70 ms (conforme medido no breakdown do treino). Cada sync forçado adiciona ~0,16 ms — menos de 0,25% do tempo de execução do kernel por step. Quando a GPU está fazendo trabalho real, o overhead de sincronização se torna relativamente barato.

A solução estrutural é o **DDIM** — uma reformulação do processo reverso que alcança qualidade de amostra equivalente em 50 steps em vez de 1000, escolhendo deterministicamente uma trajetória de steps pulados através do mesmo modelo treinado. O speedup: **19,2×** (0,20s vs 3,78s). Reduzir a contagem de steps de 1000 para 50 é muito mais eficaz do que eliminar o overhead de sincronização.

---

## O Processo de Denoising

{{< img src="images/denoising.gif" alt="8 amostras sendo denoised de ruído puro a dígitos reconhecíveis" >}}

Oito amostras geradas em paralelo, capturadas a cada 100 steps. O primeiro frame é ruído puro. Pelos primeiros ~700 steps, os frames permanecem indistinguíveis de estática — coerente com o noise schedule: o SNR é tão baixo que o modelo está operando quase inteiramente no regime de ruído. A estrutura só se compromete nos últimos 200–300 steps, quando $\bar{\alpha}_t$ é grande o suficiente para que o gradiente da função de score direcione para formas reconhecíveis.

Esse comprometimento tardio não é uma falha do treino — é o modelo seguindo fielmente o schedule com o qual foi treinado.

---

## Código

Implementação completa, benchmarks e scripts de treino em [luanhcosta/ddpm-mnist](https://github.com/luanhcosta/ddpm-mnist). O `scripts/benchmark.py` reproduz todas as medições de tempo deste post, e o `scripts/generate_assets.py` gera os gráficos conceituais sem precisar de um modelo treinado.
