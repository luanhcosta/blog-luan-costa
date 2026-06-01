---
title: "Training a Diffusion Model from Scratch: GPU Execution and CPU-GPU Communication"
date: 2026-05-31
description: "A complete breakdown of training a DDPM on MNIST from scratch — covering the diffusion math, U-Net architecture, and every point where the CPU and GPU exchange data during training and inference. All benchmarks run on an RTX 5050."
authors:
  - name: Luan Costa
---

Generative models are among the most compute-intensive workloads in deep learning. Understanding not just whether a model trains, but *how* the hardware is being used during training, is what separates writing code that works from writing code that runs well.

This post documents the full implementation of a Denoising Diffusion Probabilistic Model (DDPM) trained on MNIST from scratch, with a specific focus on CPU-GPU communication: where data moves, when the CPU and GPU synchronize, and what the actual bottlenecks are. All measurements were collected on an **NVIDIA GeForce RTX 5050 Laptop GPU** (Blackwell, 8 GB). The full source is at [luanhcosta/ddpm-mnist](https://github.com/luanhcosta/ddpm-mnist).

---

## The Forward Process

DDPM frames image generation as the reversal of a noise-adding process. The forward process $q(x_t \mid x_0)$ progressively corrupts a clean image $x_0$ by adding Gaussian noise over $T$ timesteps:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

where $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$ is the cumulative product of the noise schedule. With a linear schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ over $T = 1000$ steps, the schedule behaves as follows:

{{< img src="images/noise_schedule.png" alt="DDPM Linear Noise Schedule — β, ᾱ, signal/noise amplitude and SNR" >}}

Three observations from this plot that inform both training and inference:

- Signal and noise amplitudes cross at **t ≈ 258** — less than 26% into the process. After this point, noise already dominates the image.
- $\bar{\alpha}_t$ decays in an S-curve, not linearly. Most signal is lost in the first 300 steps.
- The SNR (bottom right, log scale) collapses exponentially. By t = 500 it has already dropped by three orders of magnitude from its initial value.

The effect on a real MNIST digit:

{{< img src="images/noising_trajectory.png" alt="Forward process: a digit 2 being progressively noised at t=0, 100, 250, 500, 750, 999" >}}

At t = 250 the shape is still faintly visible — consistent with the noise schedule crossing at t ≈ 258. By t = 500 the digit is unrecognizable, and by t = 999 it is indistinguishable from pure Gaussian noise.

---

## The Model: U-Net with Time Conditioning

The reverse process $p_\theta(x_{t-1} \mid x_t)$ is parameterized by a U-Net that predicts the noise $\varepsilon$ added at timestep $t$, trained with a simple MSE loss:

$$\mathcal{L} = \mathbb{E}_{x_0, t, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]$$

The architecture for 28×28 images uses three resolution levels with a bottleneck at 7×7:

| Stage | Resolution | Channels | Notes |
|---|---|---|---|
| Encoder L0 | 28×28 | 64 | 2× ResBlock |
| Encoder L1 | 14×14 | 128 | 2× ResBlock |
| Encoder L2 | 7×7 | 256 | 2× ResBlock + SelfAttention |
| Bottleneck | 7×7 | 256 | ResBlock + SelfAttention + ResBlock |
| Decoder L2→1 | 14×14 | 128 | upsample + 2× ResBlock |
| Decoder L1→0 | 28×28 | 64 | upsample + 2× ResBlock |

Each ResBlock receives the timestep embedding — a sinusoidal encoding of $t$ projected to 256 dimensions via an MLP — added into the feature map after the first convolution. This is how the model distinguishes "denoise from t=900" from "denoise from t=50". Total parameters: **9,744,641**.

---

## CPU-GPU Communication During Training

Every training step follows the standard pipeline: load batch from CPU, transfer to GPU, run forward pass, compute loss, run backward pass, update weights. The question is where time actually goes.

### Host-to-Device Transfer Bandwidth

The standard advice is to use `pin_memory=True` in the DataLoader so host tensors are page-locked before being transferred via DMA. The actual bandwidth gain depends heavily on tensor size:

{{< img src="images/h2d_bandwidth.png" alt="H2D Transfer Bandwidth: pageable vs pinned memory across tensor sizes" >}}

At **1 MB**, pinned memory is **2.5× faster** (17.3 vs 6.9 GB/s) — the OS overhead of paging dominates for small transfers. At **10 MB**, the gap nearly vanishes (10.6 vs 10.9 GB/s). At **500 MB**, pinned recovers to a 1.24× advantage as sustained DMA efficiency matters again.

For MNIST batches (batch size 128, 28×28 float32 images = ~400 KB per batch), we are firmly in the small-tensor regime where pinned memory has the most impact.

### DataLoader Throughput

{{< img src="images/dataloader_comparison.png" alt="DataLoader throughput: pin_memory=True vs False" >}}

`pin_memory=True` delivers **1.78× more throughput**: 14,662 vs 8,292 batches/second, or 0.068 vs 0.121 ms per batch. For a training loop that processes ~460 batches per epoch, this saves roughly 24 ms per epoch — modest in isolation, but free.

### Where Time Actually Goes in a Training Step

{{< img src="images/training_breakdown.png" alt="Training step phase breakdown: data H2D, forward, loss, backward, optimizer" >}}

This is the most revealing measurement. Averaged over 100 steps with the trained model:

| Phase | Time | Share |
|---|---|---|
| data H2D | 0.08 ms | **0.0%** |
| noise + timestep sample | 0.05 ms | 0.0% |
| forward pass | 70.1 ms | 32.8% |
| loss | 0.09 ms | 0.0% |
| backward pass | 140.4 ms | **65.8%** |
| optimizer step | 2.7 ms | 1.3% |
| **Total** | **213.5 ms** | |

The CPU-to-GPU data transfer is **three orders of magnitude smaller** than the backward pass. For a ~10M parameter model on 28×28 images, the bottleneck is the gradient computation — not memory communication. Obsessing over data loading optimization here would optimize the 0.0% while ignoring the 65.8%.

---

## Training Results

100 epochs, Adam with lr=2e-4, batch size 128, AMP (float16), gradient clipping at 1.0.

{{< img src="images/loss_curve.png" alt="DDPM training loss over 47,000 steps" >}}

The loss drops sharply in the first ~2,000 steps and then enters a long, stable refinement phase. No instability — one of the structural advantages of DDPM over GANs: there is only one network to train, and the MSE loss landscape is well-behaved. The model reached its final loss of ~0.02 and stayed there for the last 80 epochs.

After 100 epochs, sampling 64 images from pure Gaussian noise:

{{< img src="images/samples_final.png" alt="64 images generated by the trained DDPM from pure noise" >}}

All 10 digit classes appear with clean strokes and no mode collapse. Some samples are ambiguous — the model learned the full distribution of handwriting styles, not just canonical forms.

---

## Inference: The Communication Cost of 1000 Steps

Generating one image with DDPM requires 1000 sequential model evaluations — one per reverse timestep. This creates a natural question: what is the cost of synchronizing the CPU and GPU at every step?

In code, any Python operation on a GPU tensor that returns a Python scalar — like `x.mean().item()` — forces a `cudaDeviceSynchronize()`. The CPU blocks until the GPU finishes that kernel before the Python interpreter can continue. Naive monitoring code inside the denoising loop triggers this at every one of the 1000 steps.

{{< img src="images/sampling_comparison.png" alt="Sampling time: DDPM-1000 with forced syncs vs pure GPU, vs DDIM-50" >}}

The result for this model: the sync overhead is **1.04×** — 3.94s vs 3.78s. The 1000 `.item()` calls add roughly 160 ms over the full loop.

This is smaller than expected, and the reason is informative: the U-Net forward pass on a 28×28 input takes ~70 ms (as measured in the training breakdown). Each forced sync adds ~0.16 ms — less than 0.25% of the kernel execution time per step. When the GPU is doing real work, synchronization overhead becomes relatively cheap.

The structural solution is **DDIM** — a reformulation of the reverse process that achieves equivalent sample quality in 50 steps instead of 1000, by deterministically choosing a skip-step trajectory through the same trained model. The speedup: **19.2×** (0.20s vs 3.78s). Reducing the step count from 1000 to 50 is far more effective than eliminating synchronization overhead.

---

## The Denoising Process

{{< img src="images/denoising.gif" alt="8 samples being denoised from pure noise to recognizable digits" >}}

Eight samples generated in parallel, captured every 100 steps. The first frame is pure noise. For the first ~700 steps the frames remain indistinguishable from static — consistent with the noise schedule: the SNR is so low that the model is operating almost entirely in the noise regime. Structure only commits in the final 200–300 steps, when $\bar{\alpha}_t$ is large enough that the gradient of the score function can steer toward recognizable forms.

This late commitment is not a failure of training — it is the model faithfully following the schedule it was trained on.

---

## Source

Full implementation, benchmarks, and training scripts at [luanhcosta/ddpm-mnist](https://github.com/luanhcosta/ddpm-mnist). The `scripts/benchmark.py` reproduces all timing measurements in this post, and `scripts/generate_assets.py` generates the conceptual plots without requiring a trained model.
