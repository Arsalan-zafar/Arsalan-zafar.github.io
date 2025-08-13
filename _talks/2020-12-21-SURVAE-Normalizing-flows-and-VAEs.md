---
title: Normalizing flows and VAEs
collection: talks
type: "Talk"
permalink: /talks/Normalizing_flows_and_VAEs
venue: "Online talk"
date: 2021-12-21
location: "London"
---

In this video, Chris Finlay and I provide an overview of VAEs and normalizing flows through presenting the SURVAE paper which aims bridge the gap between them.  

The paper introduces a modular framework that unifies variational autoencoders (VAEs) and normalizing flows by allowing surjective transformations: deterministic in one direction (enabling exact likelihood) and stochastic in the reverse (yielding a tractable lower bound). This perspective recovers VAEs and bijective flows as special cases and shows that techniques like dequantization, variational data augmentation, and augmented flows can be expressed within the same compositional toolkit. The authors also add practical layers—absolute value, max pooling, sorting, and stochastic permutations—to model symmetries, perform downsampling with exact likelihoods, and handle exchangeable data. Experiments on synthetic datasets, point clouds, and images demonstrate that these layers improve modeling of symmetric/exchangeable structure and that max-pooling surjections can trade a small likelihood drop for better sample quality (e.g., better Inception/FID)

<iframe width="560" height="315" src="https://www.youtube.com/embed/vC0F_XMnv3k" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

