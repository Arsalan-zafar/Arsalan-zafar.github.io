---
title: 'Cross-Entropy, KL and NLL are the same objective in AI compression'
date: 2019-05-19
permalink: /posts/2020/05/cross-entropy-kl-nll-ai-compression/
tags:
  - AI based compression
  - Information Theory
  - Likelihood
  - Entropy coding
published: True
mathjax: true
---

In end-to-end learned compression, we need to model the distribution of data, so we can entropy encode it. The training loss we use is the cross-entropy between the unknown true data distribution and our model distribution which in a simplest case is a fully factorized distribution of standard normals. There is often some confusion about the objective used in compression so I thought I'd use this post to clarify it. I'll show that for data sampled from the true distribution $$p$$ and a parametric model $$q_\theta$$ we learn, the cross-entropy, the Kullback–Leibler divergence and the negative log-likelihood are optimization-equivalent objectives. In practice this means we are doing maximum likelihood estimation (MLE) and simultaneously minimizing expected code length.

## Setup

- We assume samples $$x \sim p$$ (the real data distribution).
- We train a model $$q_\theta(x)$$ (density or probability mass) used for entropy coding and for likelihood.
- Expectations are with respect to $$p$$: $$\mathbb{E}_p[\cdot] = \mathbb{E}_{x\sim p}[\cdot]$$.

## Definitions (in nats)

- Cross-entropy of $$p$$ under $$q_\theta$$:

$$H(p, q_\theta) := \mathbb{E}_p\big[-\log q_\theta(x)\big].$$

- Shannon entropy of \(p\):

$$H(p) := \mathbb{E}_p\big[-\log p(x)\big].$$

- KL divergence (forward KL):

$$D_{\mathrm{KL}}(p\,\|\,q_\theta) := \mathbb{E}_p\big[\log p(x) - \log q_\theta(x)\big].$$

- Negative log-likelihood (NLL):

$$\mathrm{NLL}(\theta) := \mathbb{E}_p\big[-\log q_\theta(x)\big].$$

## Identities and immediate consequences

By simple algebra,

$$\begin{aligned}
H(p, q_\theta)
&= \mathbb{E}_p\big[-\log q_\theta(x)\big]\\
&= \mathbb{E}_p\big[\log p(x) - \log q_\theta(x)\big] + \mathbb{E}_p\big[-\log p(x)\big]\\
&= D_{\mathrm{KL}}(p\,\|\,q_\theta) + H(p).
\end{aligned}$$

Therefore

$$H(p, q_\theta) \equiv \mathrm{NLL}(\theta) = D_{\mathrm{KL}}(p\,\|\,q_\theta) + H(p).$$

Since \(H(p)\) does not depend on \(\theta\), the three quantities are **minimization-equivalent** in \(\theta\):

$$\arg\min_\theta H(p, q_\theta) \;=\; \arg\min_\theta D_{\mathrm{KL}}(p\,\|\,q_\theta) \;=\; \arg\min_\theta \mathrm{NLL}(\theta).$$

Equivalently,

$$\arg\max_\theta \mathbb{E}_p\big[\log q_\theta(x)\big]$$

which is the **maximum likelihood estimator** in expectation.


The latent space we model has dependencies, so a fully factorized mean-field approximation is too simple and will result in a large KL divergence. To improve modeling the joint distribution of our latents we need to compress, we can use a hyperprior model which introduces a latent that captures the dependencies and allows us to assume a fully factorized model, or we can break down our joint into a product of conditionals and use PixelCNN-like autoregressive models. Using conditional models (e.g., autoregressive context, hyperprior latents) preserves all identities by replacing $$q_\theta(x)$$ with $$q_\theta(x\mid y)$$ and taking expectations over the joint $$p(x,y)$$. 

## Takeaways

- Training with cross-entropy loss in AI compression is the same as minimizing forward KL from the data distribution to the model and the same as minimizing population NLL.
- Consequently, we are performing MLE (maximizing $$\mathbb{E}_p[\log q_\theta(x)]$$).
- Minimizing this loss also minimizes the expected code length produced by an ideal entropy coder fed with $$q_\theta$$.

