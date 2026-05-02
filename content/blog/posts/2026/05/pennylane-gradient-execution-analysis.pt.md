---
title: "Quantas Vezes Seu Circuito Quântico Executa? Uma Análise de Execuções de Gradiente no PennyLane"
date: 2026-05-02
description: "Uma análise empírica completa das contagens de execução de circuitos quânticos na implementação QAOA MaxCut do PennyLane, cobrindo todos os métodos de diferenciação — parameter-shift, finite-diff, hadamard, SPSA, adjoint e backprop."
authors:
  - name: Luan Costa
---

Quantas execuções de circuito quântico o PennyLane realiza por etapa de otimização no QAOA? Essa pergunta surgiu durante meu mestrado, pois não encontrei uma resposta satisfatória na literatura ou na documentação. Usando uma análise orientada a código, gerei um relatório sobre os custos exatos.

Este post é uma análise completa desse número. A análise completa é open-source em [luanhcosta/pennylane-gd-analysis](https://github.com/luanhcosta/pennylane-gd-analysis).

---

## Definição de Variáveis

Toda fórmula neste post utiliza os quatro símbolos a seguir:

| Símbolo | Significado |
|---------|-------------|
| $E$ | Número de arestas do grafo = termos $ZZ$ de Pauli não-triviais em $H_\text{cost}$ |
| $N$ | Número de nós do grafo = termos $X$ no Hamiltoniano mixer padrão |
| $\text{depth}$ | Número de camadas QAOA ($p$) |
| $P$ | Parâmetros treináveis `PauliRot` após decomposição da tape $= (E+N) \times \text{depth}$ |
| $D$ | Número de direções aleatórias para o SPSA |

---

## A Fórmula do `parameter-shift`

$$\text{execuções} = 1 + 2P = 1 + 2(E + N) \times \text{depth}$$

O $1$ vem da passagem forward que o `GradientDescentOptimizer.step_and_cost` realiza para avaliar o custo. O $2P$ vem da regra de parameter-shift de dois termos aplicada a cada um dos $P$ parâmetros treináveis `PauliRot`:

$$\frac{\partial f}{\partial \theta_k} = \frac{f(\theta_k + \pi/2) - f(\theta_k - \pi/2)}{2}$$

### Por que $(E+N) \times \text{depth}$ parâmetros?

`qml.qaoa.cost_layer` e `qml.qaoa.mixer_layer` do PennyLane usam internamente `ApproxTimeEvolution`. Esse operador não possui `grad_recipe` ou `parameter_frequencies` embutidos, então o `parameter-shift` não consegue diferenciá-lo diretamente. Antes de qualquer regra de gradiente ser aplicada, o PennyLane o decompõe:

```
tape do usuário:
  [Hadamard × N] + [ApproxTimeEvolution(H_cost, γ_k) + ApproxTimeEvolution(H_mixer, α_k)] × depth

     ↓  _expand_transform_param_shift → decompose()

tape de gradiente:
  [Hadamard × N] + [PauliRot × E (termos de custo) + PauliRot × N (termos do mixer)] × depth
```

Durante essa decomposição, `ApproxTimeEvolution.compute_decomposition()` ignora palavras de Pauli Identidade (o guard `if len(pw):` na linha 228 de `approx_time_evolution.py`). O Hamiltoniano de custo MaxCut tem $E$ termos $ZZ$ não-triviais e uma constante identidade — a identidade é descartada. O mixer tem $N$ termos $X$. Por camada QAOA: $E + N$ gates PauliRot. Ao longo de $\text{depth}$ camadas: $P = (E+N) \times \text{depth}$ parâmetros treináveis.

### Verificação Empírica

**Variando depth** (grafo ciclo com 10 nós, $E = N = 10$):

| depth | $E+N$ | Previsto $1+2(E+N)\cdot\text{depth}$ | Real | Bateu |
|-------|-------|--------------------------------------|------|-------|
| 1 | 20 | 41 | 41 | ✓ |
| 2 | 20 | 81 | 81 | ✓ |
| 5 | 20 | 201 | 201 | ✓ |
| 10 | 20 | 401 | 401 | ✓ |
| 20 | 20 | 801 | 801 | ✓ |

**Variando número de arestas** (10 nós, depth = 5):

| Grafo | $E$ | $N$ | Previsto | Real | Bateu |
|-------|-----|-----|----------|------|-------|
| Ciclo (10 arestas) | 10 | 10 | 201 | 201 | ✓ |
| Caminho (9 arestas) | 9 | 10 | 191 | 191 | ✓ |
| Completo $K_{10}$ (45 arestas) | 45 | 10 | 551 | 551 | ✓ |
| Petersen (15 arestas) | 15 | 10 | 251 | 251 | ✓ |
| Esparso (2 arestas, 4 nós) | 2 | 4 | 61 | 61 | ✓ |

**Variando número de nós** (grafo ciclo, depth = 3):

| Nós | $E$ | $N$ | Previsto | Real | Bateu |
|-----|-----|-----|----------|------|-------|
| 3 | 3 | 3 | 37 | 37 | ✓ |
| 4 | 4 | 4 | 49 | 49 | ✓ |
| 5 | 5 | 5 | 61 | 61 | ✓ |
| 6 | 6 | 6 | 73 | 73 | ✓ |
| 8 | 8 | 8 | 97 | 97 | ✓ |
| 10 | 10 | 10 | 121 | 121 | ✓ |

Zero divergências de previsão em todos os 16 casos.

### Verificação da Tape Decomposta

Para confirmar a contagem de parâmetros treináveis diretamente, a análise inspecionou a tape no nível de gradiente usando `qml.workflow.construct_batch(..., level="gradient")`:

| Circuito | depth | Params treináveis ($P$) | Execuções previstas ($1+2P$) |
|----------|-------|-------------------------|------------------------------|
| Ciclo (10 nós) | 2 | 40 | 81 |
| Ciclo (10 nós) | 5 | 100 | 201 |
| Ciclo (5 nós) | 3 | 30 | 61 |

---

## Comparação de Todos os `diff_method`

Testado em grafo ciclo com 10 nós, depth = 5, resultando em $P = (10+10) \times 5 = 100$.

| `diff_method` | Execuções de circuito (simulações) | Execuções no Tracker | Qualidade do gradiente | Compatível com hardware |
|---|---|---|---|---|
| `parameter-shift` | 201 | 201 | Analítico exato | ✓ |
| `finite-diff` (forward, ordem 1) | 102 | 102 | Aprox. $O(h)$ | ✓ |
| `finite-diff` (forward, ordem 2) | 202 | 202 | Aprox. $O(h^2)$ | ✓ |
| `finite-diff` (center, ordem 2) | 201 | 201 | Aprox. $O(h^2)$ | ✓ |
| `hadamard` | **101** (simulações) | 1101 (artefato do tracker) | Analítico exato | ✓ (precisa de ancilla) |
| `spsa` ($D=1$) | 3 | 3 | Estocástico | ✓ |
| `adjoint` | **1** | **1** | Analítico exato | Apenas simulador |
| `backprop` | **1** | **1** | Analítico exato | Apenas simulador |

---

## `finite-diff` — Estratégias e Ordens

O mesmo pipeline de decomposição do `parameter-shift` se aplica aqui, gerando os mesmos $P = (E+N)\times\text{depth}$ parâmetros treináveis PauliRot. A diferença está na receita de shifts escolhida por `finite_diff_coeffs(n, approx_order, strategy)`:

| Estratégia | Ordem | Pontos de shift | Fórmula | Execuções ($P=100$) |
|------------|-------|-----------------|---------|----------------------|
| `forward` | 1 | $[0, +h]$ | $2 + P$ | 102 |
| `forward` | 2 | $[0, +h, +2h]$ | $2 + 2P$ | 202 |
| `backward` | 1 | $[0, -h]$ | $2 + P$ | 102 |
| `backward` | 2 | $[0, -h, -2h]$ | $2 + 2P$ | 202 |
| `center` | 2 | $[-h, +h]$ | $1 + 2P$ | 201 |

**Por que forward/backward custa uma avaliação extra?** O `step_and_cost` do otimizador avalia $f(\theta)$ para o custo. As receitas de gradiente forward e backward também incluem um termo de shift zero $f(\theta)$ — mas o PennyLane não reutiliza a avaliação do otimizador, disparando uma segunda passagem forward independente. O esquema center não tem termo de shift zero, então apenas 1 avaliação forward no total.

---

## `hadamard` — A Inflação do Tracker

O método de gradiente Hadamard gera 1 circuito por parâmetro treinável usando um qubit ancilla: ele transforma o observável de medição para $H_\text{cost} \otimes Y(\text{aux\_wire})$.

Para MaxCut, `cost_h.pauli_rep` tem $E$ palavras $ZZ$ não-triviais mais 1 palavra identidade. Após a transformação Hadamard, o observável se torna um `Sum` com $E+1$ termos. O `default.qubit` conta cada termo de um `Sum` como uma execução separada, inflando o tracker:

$$\text{execuções no tracker} = 1 + (E+1) \times P$$

$$\text{custo real de hardware (simulações)} = 1 + P$$

Verificação ($E = 10$, $N = 10$):

| depth | $P$ | Simulações ($1+P$) | Execuções no tracker ($1+(E+1)P$) | Fórmula |
|-------|-----|--------------------|------------------------------------|---------|
| 1 | 20 | 21 | 221 | ✓ |
| 2 | 40 | 41 | 441 | ✓ |
| 5 | 100 | 101 | 1101 | ✓ |
| 10 | 200 | 201 | 2201 | ✓ |

**A chave `"simulations"` no `qml.Tracker` é o custo real de hardware.** A chave `"executions"` fica inflada por um fator de $E+1$ devido à forma como o `default.qubit` trata observáveis `Sum` internamente. Em hardware quântico real, a contagem de preparações de circuito equivale à contagem de simulações.

O método exige um qubit extra (`gradient_kwargs={"aux_wire": nodes}`, dispositivo inicializado com `wires=nodes+1`).

---

## `spsa` — Custo Fixo Independente de $P$

O SPSA (Simultaneous Perturbation Stochastic Approximation) estima o gradiente completo a partir de apenas duas avaliações de circuito por direção aleatória:

$$\nabla f(\theta) \approx \frac{f(\theta + h\delta) - f(\theta - h\delta)}{2h\delta}$$

onde $\delta$ é um vetor aleatório $\pm 1$ sobre todos os parâmetros simultaneamente. O custo é completamente independente de $P$:

$$\text{execuções} = 1 + 2D$$

Verificação:

| $D$ | Previsto $1+2D$ | Real |
|-----|----------------|------|
| 1 | 3 | 3 |
| 2 | 5 | 5 |
| 4 | 9 | 9 |
| 8 | 17 | 17 |
| 16 | 33 | 33 |

O SPSA é configurado via `gradient_kwargs={"num_directions": D, "sampler_rng": np.random.default_rng(42)}`. O trade-off são estimativas de gradiente ruidosas — o SPSA converge, mas exige mais iterações de otimização do que métodos exatos.

---

## `adjoint` e `backprop` — Métodos de Passagem Única

Ambos os métodos custam exatamente **1 execução de circuito**, sem overhead adicional de tapes shiftadas.

**adjoint:** A passagem forward armazena o vetor de estado $|\psi_k\rangle$ em cada fronteira de gate. A passagem backward propaga um bra de volta pelo circuito, acumulando $\partial f / \partial \theta_k$ em cada gate. O overhead de memória é $O(\text{depth})$ vetores de estado, não $O(P)$ avaliações de circuito. Compatível com `default.qubit` e `lightning.qubit`, mas não com hardware real.

**backprop:** Incorpora toda a simulação de vetor de estado em um grafo de diferenciação automática clássica (autograd, JAX ou PyTorch). Os gradientes fluem de volta pela simulação quântica. Requer um simulador completamente diferenciável e não generaliza além da simulação.

Ambos reportam `tracker.totals["executions"] = 1`.

---

## Modelo de Custo Completo

| `diff_method` | Fórmula de execuções | Execuções de circuito | Qualidade do gradiente | Hardware |
|---|---|---|---|---|
| `parameter-shift` | $1 + 2P$ | $1 + 2P$ | Analítico exato | ✓ |
| `finite-diff` fwd/1 | $2 + P$ | $2 + P$ | Aprox. $O(h)$ | ✓ |
| `finite-diff` fwd/2 | $2 + 2P$ | $2 + 2P$ | Aprox. $O(h^2)$ | ✓ |
| `finite-diff` ctr/2 | $1 + 2P$ | $1 + 2P$ | Aprox. $O(h^2)$ | ✓ |
| `hadamard` | $1 + (E+1)P$ (tracker) | $1 + P$ | Analítico exato | ✓ (ancilla necessária) |
| `spsa` | $1 + 2D$ | $1 + 2D$ | Estocástico | ✓ |
| `adjoint` | $1$ | $1$ | Analítico exato | Apenas simulador |
| `backprop` | $1$ | $1$ | Analítico exato | Apenas simulador |

Onde $P = (E + N) \times \text{depth}$.

### Sensibilidade ao Escalonamento

| Mudança | Impacto no `parameter-shift` |
|---------|------------------------------|
| +1 aresta | $+2 \times \text{depth}$ execuções |
| +1 nó | $+2 \times \text{depth}$ execuções |
| $\times 2$ depth | $\times 2$ execuções (linear) |
| $K_{10}$ completo vs ciclo | $E = 45$ vs $E = 10$; em depth 20: 2201 vs 801 execuções |

Para o grafo completo $K_{10}$: $E = 45$, $P = 55 \times \text{depth}$. Em depth 20: $P = 1100$, execuções $= 2201$ — quase $3\times$ o custo do grafo ciclo.

---

## Guia de Decisão

**Apenas simulação:**
- Use `adjoint` para gradientes exatos mais rápidos com baixo overhead de memória.
- Use `backprop` se precisar de integração com autodiff JAX/PyTorch.

**Hardware-realista ou hardware real:**
- `parameter-shift` — gradientes exatos, $O((E+N)\times\text{depth})$ circuitos. A escolha padrão segura.
- `hadamard` — mesma contagem real de circuitos que o parameter-shift; precisa de 1 qubit ancilla; ignore a chave `"executions"` no `Tracker`.
- `finite-diff center/2` — mesma contagem que o parameter-shift, erro $O(h^2)$, sem ancilla, mais simples de configurar.
- `finite-diff forward/1` — opção aproximada mais barata quando $P$ é pequeno (menos de 3 parâmetros).
- `spsa` — custo fixo $1 + 2D$ independente do tamanho do circuito; útil quando $P$ é muito grande; converge com ruído.

---

## Reproduzindo a Análise

```bash
git clone https://github.com/luanhcosta/pennylane-gd-analysis
cd pennylane-gd-analysis
python3 -m venv .venv
.venv/bin/pip install pennylane==0.44.1 networkx==3.6.1
.venv/bin/python analysis.py
```

Todas as previsões nas 8 seções de análise são verificadas programaticamente. Zero divergências foram observadas em qualquer uma das 16+ configurações testadas.
