---
title: "How Many Times Does Your Quantum Circuit Run? A PennyLane Gradient Execution Analysis"
date: 2026-05-02
description: "A complete empirical breakdown of quantum circuit execution counts in PennyLane's QAOA MaxCut implementation across every differentiation method — parameter-shift, finite-diff, hadamard, SPSA, adjoint, and backprop."
authors:
  - name: Luan Costa
---

How many quantum circuit executions does PennyLane run per optimization step in QAOA? This question arose during my master's degree, as I could not find a satisfactory answer in the literature or documentation. Using a code-driven analysis, I generated a report on the exact costs.

This post is a complete teardown of that number. The full analysis is open-source at [luanhcosta/pennylane-gd-analysis](https://github.com/luanhcosta/pennylane-gd-analysis).

---

## Variable Definitions

Every formula in this post uses the following four symbols:

| Symbol | Meaning |
|--------|---------|
| $E$ | Number of graph edges = non-trivial $ZZ$ Pauli terms in $H_\text{cost}$ |
| $N$ | Number of graph nodes = $X$ terms in the default mixer Hamiltonian |
| $\text{depth}$ | Number of QAOA layers ($p$) |
| $P$ | Trainable `PauliRot` parameters after tape decomposition $= (E+N) \times \text{depth}$ |
| $D$ | Number of random directions for SPSA |

---

## The `parameter-shift` Formula

$$\text{executions} = 1 + 2P = 1 + 2(E + N) \times \text{depth}$$

The $1$ comes from the forward pass that `GradientDescentOptimizer.step_and_cost` performs to evaluate the cost. The $2P$ comes from the two-term parameter-shift rule applied to each of the $P$ trainable `PauliRot` parameters:

$$\frac{\partial f}{\partial \theta_k} = \frac{f(\theta_k + \pi/2) - f(\theta_k - \pi/2)}{2}$$

### Why $(E+N) \times \text{depth}$ parameters?

PennyLane's `qml.qaoa.cost_layer` and `qml.qaoa.mixer_layer` internally use `ApproxTimeEvolution`. This operator has no built-in `grad_recipe` or `parameter_frequencies`, so `parameter-shift` cannot differentiate it directly. Before any gradient rule fires, PennyLane decomposes it:

```
user tape:
  [Hadamard × N] + [ApproxTimeEvolution(H_cost, γ_k) + ApproxTimeEvolution(H_mixer, α_k)] × depth

     ↓  _expand_transform_param_shift → decompose()

gradient tape:
  [Hadamard × N] + [PauliRot × E (cost terms) + PauliRot × N (mixer terms)] × depth
```

During this decomposition, `ApproxTimeEvolution.compute_decomposition()` skips Identity Pauli words (the `if len(pw):` guard at line 228 of `approx_time_evolution.py`). The MaxCut cost Hamiltonian has $E$ non-trivial $ZZ$ terms and one identity constant, the identity is dropped. The mixer has $N$ $X$ terms. Per QAOA layer: $E + N$ PauliRot gates. Over $\text{depth}$ layers: $P = (E+N) \times \text{depth}$ trainable parameters.

### Empirical Verification

**Varying depth** (10-node cycle graph, $E = N = 10$):

| depth | $E+N$ | Predicted $1+2(E+N)\cdot\text{depth}$ | Actual | Match |
|-------|-------|---------------------------------------|--------|-------|
| 1 | 20 | 41 | 41 | ✓ |
| 2 | 20 | 81 | 81 | ✓ |
| 5 | 20 | 201 | 201 | ✓ |
| 10 | 20 | 401 | 401 | ✓ |
| 20 | 20 | 801 | 801 | ✓ |

**Varying edge count** (10 nodes, depth = 5):

| Graph | $E$ | $N$ | Predicted | Actual | Match |
|-------|-----|-----|-----------|--------|-------|
| Cycle (10 edges) | 10 | 10 | 201 | 201 | ✓ |
| Path (9 edges) | 9 | 10 | 191 | 191 | ✓ |
| Complete $K_{10}$ (45 edges) | 45 | 10 | 551 | 551 | ✓ |
| Petersen (15 edges) | 15 | 10 | 251 | 251 | ✓ |
| Sparse (2 edges, 4 nodes) | 2 | 4 | 61 | 61 | ✓ |

**Varying node count** (cycle graph, depth = 3):

| Nodes | $E$ | $N$ | Predicted | Actual | Match |
|-------|-----|-----|-----------|--------|-------|
| 3 | 3 | 3 | 37 | 37 | ✓ |
| 4 | 4 | 4 | 49 | 49 | ✓ |
| 5 | 5 | 5 | 61 | 61 | ✓ |
| 6 | 6 | 6 | 73 | 73 | ✓ |
| 8 | 8 | 8 | 97 | 97 | ✓ |
| 10 | 10 | 10 | 121 | 121 | ✓ |

Zero prediction mismatches across all 16 cases.

### Decomposed Tape Verification

To confirm the trainable parameter count directly, the analysis inspected the gradient-level tape using `qml.workflow.construct_batch(..., level="gradient")`:

| Circuit | depth | Trainable params ($P$) | Predicted executions ($1+2P$) |
|---------|-------|------------------------|-------------------------------|
| Cycle(10 nodes) | 2 | 40 | 81 |
| Cycle(10 nodes) | 5 | 100 | 201 |
| Cycle(5 nodes) | 3 | 30 | 61 |

---

## All `diff_method` Options Compared

Tested on a 10-node cycle graph, depth = 5, giving $P = (10+10) \times 5 = 100$.

| `diff_method` | Circuit runs (simulations) | Tracker executions | Gradient quality | Hardware-compatible |
|---|---|---|---|---|
| `parameter-shift` | 201 | 201 | Exact analytic | ✓ |
| `finite-diff` (forward, order 1) | 102 | 102 | $O(h)$ approximate | ✓ |
| `finite-diff` (forward, order 2) | 202 | 202 | $O(h^2)$ approximate | ✓ |
| `finite-diff` (center, order 2) | 201 | 201 | $O(h^2)$ approximate | ✓ |
| `hadamard` | **101** (simulations) | 1101 (tracker artifact) | Exact analytic | ✓ (needs ancilla) |
| `spsa` ($D=1$) | 3 | 3 | Stochastic | ✓ |
| `adjoint` | **1** | **1** | Exact analytic | Simulator only |
| `backprop` | **1** | **1** | Exact analytic | Simulator only |

---

## `finite-diff` — Strategies and Orders

The same decomposition pipeline applies as with `parameter-shift`, producing the same $P = (E+N)\times\text{depth}$ trainable PauliRot parameters. The difference is the shift recipe chosen by `finite_diff_coeffs(n, approx_order, strategy)`:

| Strategy | Order | Shift points | Formula | Executions ($P=100$) |
|----------|-------|--------------|---------|----------------------|
| `forward` | 1 | $[0, +h]$ | $2 + P$ | 102 |
| `forward` | 2 | $[0, +h, +2h]$ | $2 + 2P$ | 202 |
| `backward` | 1 | $[0, -h]$ | $2 + P$ | 102 |
| `backward` | 2 | $[0, -h, -2h]$ | $2 + 2P$ | 202 |
| `center` | 2 | $[-h, +h]$ | $1 + 2P$ | 201 |

**Why does forward/backward cost an extra evaluation?** The optimizer's `step_and_cost` evaluates $f(\theta)$ for the cost. The forward and backward gradient recipes also include a zero-shift term $f(\theta)$ — but PennyLane does not reuse the optimizer's evaluation, so it fires an independent second forward pass. The center scheme has no zero-shift term, so only 1 forward evaluation total.

---

## `hadamard` — The Tracker Inflation

The Hadamard gradient method generates 1 circuit per trainable parameter by using an ancilla qubit: it transforms the measurement observable to $H_\text{cost} \otimes Y(\text{aux\_wire})$.

For MaxCut, `cost_h.pauli_rep` has $E$ non-trivial $ZZ$ words plus 1 identity word. After the Hadamard transform, the observable becomes a `Sum` with $E+1$ terms. `default.qubit` counts each term in a `Sum` as a separate execution, inflating the tracker:

$$\text{tracker executions} = 1 + (E+1) \times P$$

$$\text{actual hardware cost (simulations)} = 1 + P$$

Verification ($E = 10$, $N = 10$):

| depth | $P$ | Simulations ($1+P$) | Tracker executions ($1+(E+1)P$) | Formula check |
|-------|-----|---------------------|---------------------------------|---------------|
| 1 | 20 | 21 | 221 | ✓ |
| 2 | 40 | 41 | 441 | ✓ |
| 5 | 100 | 101 | 1101 | ✓ |
| 10 | 200 | 201 | 2201 | ✓ |

**The `"simulations"` key in `qml.Tracker` is the correct hardware cost.** The `"executions"` key is inflated by a factor of $E+1$ due to how `default.qubit` handles `Sum` observables internally. On real quantum hardware, the actual circuit preparation count equals the simulations count.

The method requires one extra qubit (`gradient_kwargs={"aux_wire": nodes}`, device initialized with `wires=nodes+1`).

---

## `spsa` — Fixed Cost Regardless of $P$

SPSA (Simultaneous Perturbation Stochastic Approximation) estimates the full gradient from just two circuit evaluations per random direction:

$$\nabla f(\theta) \approx \frac{f(\theta + h\delta) - f(\theta - h\delta)}{2h\delta}$$

where $\delta$ is a random $\pm 1$ vector over all parameters simultaneously. The cost is completely $P$-independent:

$$\text{executions} = 1 + 2D$$

Verification:

| $D$ | Predicted $1+2D$ | Actual |
|-----|-----------------|--------|
| 1 | 3 | 3 |
| 2 | 5 | 5 |
| 4 | 9 | 9 |
| 8 | 17 | 17 |
| 16 | 33 | 33 |

SPSA is configured via `gradient_kwargs={"num_directions": D, "sampler_rng": np.random.default_rng(42)}`. The trade-off is noisy gradient estimates — SPSA converges, but requires more optimization iterations than exact methods.

---

## `adjoint` and `backprop` — Single-Pass Methods

Both methods cost exactly **1 circuit execution**, with no additional shifted-tape overhead.

**adjoint:** The forward pass stores the state vector $|\psi_k\rangle$ at each gate boundary. The backward pass propagates a bra back through the circuit, accumulating $\partial f / \partial \theta_k$ at each gate. The memory overhead is $O(\text{depth})$ state vectors, not $O(P)$ circuit evaluations. Compatible with `default.qubit` and `lightning.qubit` but not real hardware.

**backprop:** Embeds the entire state-vector simulation inside a classical automatic differentiation graph (autograd, JAX, or PyTorch). Gradients flow backwards through the quantum simulation. Requires a fully differentiable simulator, does not generalize beyond simulation.

Both report `tracker.totals["executions"] = 1`.

---

## Complete Cost Model

| `diff_method` | Executions formula | Circuit runs | Gradient quality | Hardware |
|---|---|---|---|---|
| `parameter-shift` | $1 + 2P$ | $1 + 2P$ | Exact analytic | ✓ |
| `finite-diff` fwd/1 | $2 + P$ | $2 + P$ | $O(h)$ approx | ✓ |
| `finite-diff` fwd/2 | $2 + 2P$ | $2 + 2P$ | $O(h^2)$ approx | ✓ |
| `finite-diff` ctr/2 | $1 + 2P$ | $1 + 2P$ | $O(h^2)$ approx | ✓ |
| `hadamard` | $1 + (E+1)P$ (tracker) | $1 + P$ | Exact analytic | ✓ (ancilla needed) |
| `spsa` | $1 + 2D$ | $1 + 2D$ | Stochastic | ✓ |
| `adjoint` | $1$ | $1$ | Exact analytic | Simulator only |
| `backprop` | $1$ | $1$ | Exact analytic | Simulator only |

Where $P = (E + N) \times \text{depth}$.

### Scaling Sensitivity

| Change | `parameter-shift` impact |
|--------|--------------------------|
| +1 edge | $+2 \times \text{depth}$ executions |
| +1 node | $+2 \times \text{depth}$ executions |
| $\times 2$ depth | $\times 2$ executions (linear) |
| Complete $K_{10}$ vs cycle | $E = 45$ vs $E = 10$; at depth 20: 2201 vs 801 executions |

For a complete graph $K_{10}$: $E = 45$, $P = 55 \times \text{depth}$. At depth 20: $P = 1100$, executions $= 2201$ — nearly $3\times$ the cycle graph cost.

---

## Decision Guide

**Simulation only:**
- Use `adjoint` for fastest exact gradients with low memory overhead.
- Use `backprop` if you need JAX/PyTorch autodiff integration.

**Hardware-realistic or real hardware:**
- `parameter-shift` — exact gradients, $O((E+N)\times\text{depth})$ circuits. The default safe choice.
- `hadamard` — same real circuit count as parameter-shift; needs 1 ancilla qubit; ignore the `"executions"` key in `Tracker`.
- `finite-diff center/2` — same count as parameter-shift, $O(h^2)$ error, no ancilla needed, simpler to configure.
- `finite-diff forward/1` — cheapest approximate option when $P$ is small (fewer than 3 parameters).
- `spsa` — fixed cost $1 + 2D$ regardless of circuit size; useful when $P$ is very large; converges noisily.

---

## Reproducing the Analysis

```bash
git clone https://github.com/luanhcosta/pennylane-gd-analysis
cd pennylane-gd-analysis
python3 -m venv .venv
.venv/bin/pip install pennylane==0.44.1 networkx==3.6.1
.venv/bin/python analysis.py
```

All predictions across the 8 analysis sections are verified programmatically. Zero mismatches were observed in any of the 16+ configurations tested.
