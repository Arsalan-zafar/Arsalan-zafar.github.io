---
title: Overview of cuDNN algorithms
collection: talks
type: "Talk"
permalink: /talks/Overview-Of-cuDNN-Algorithms
venue: "Online talk"
date: 2020-11-24
location: "London"
---

In this video, I provide an overview of various cuDNN algorithms by reviewing the 2019 paper by Marc Jorda et al. 

The paper benchmarks NVIDIA cuDNN’s convolution algorithms (GEMM, FFT, Winograd and their cuDNN variants) on a Tesla V100 (Volta) across 602 layer configurations drawn from AlexNet, VGG19, GoogLeNet, ResNet-50, and SqueezeNet, evaluating both FP32 and FP16. The main takeaways are practical selection rules: for FP32, filter size and input-channel count dominate—GEMM variants are best for 1×1 (batch-dependent), Winograd is generally best for 3×3 except on very large inputs where GEMM can win, and for 5×5 Winograd (non-fused) leads up to moderate batch sizes, with FFT/FFT-tiled taking over as input size grows. For FP16, exploiting Tensor Cores is decisive: use GEMM-impl-precomp-TC for 1×1 and 5×5, and Winograd-TC for 3×3. The authors also report memory-workspace considerations (capping at ~1 GB) and provide concise guidelines to choose the fastest algorithm given kernel size, batch size, and precision.

<iframe width="560" height="315" src="https://www.youtube.com/embed/l8xo6ll3AxI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

