---
title: 'Cross-Entropy, KL and NLL are the same objective in AI compression'
date: 2020-05-19
permalink: /posts/2020/05/cross-entropy-kl-nll-ai-compression/
tags:
  - AI based compression
  - Information Theory
  - Likelihood
  - Entropy coding
published: True
mathjax: true
---

In AI-based compression we model the distribution of data and then entropy-code it. The training loss we use is the cross-entropy between the unknown true data distribution and our model. This post shows, cleanly and with expectations, that for data sampled from the true distribution \(p\) and a parametric model \(q_\theta\) we learn, the cross-entropy, the Kullback–Leibler divergence and the negative log-likelihood are optimization-equivalent objectives. In practice this means we are doing maximum likelihood estimation (MLE) and simultaneously minimizing expected code length.

## Setup

- We assume samples \(x \sim p\) (the real data distribution).
- We train a model \(q_\theta(x)\) (density or probability mass) used for entropy coding and for likelihood.
- Expectations are with respect to \(p\): \(\mathbb{E}_p[\cdot] = \mathbb{E}_{x\sim p}[\cdot]\).

## Definitions (in nats)

- Cross-entropy of \(p\) under \(q_\theta\):

$$H(p, q_\theta) := \mathbb{E}_p\big[-\log q_\theta(x)\big].$$

- Shannon entropy of \(p\):

$$H(p) := \mathbb{E}_p\big[-\log p(x)\big].$$

- KL divergence (forward KL):

$$D_{\mathrm{KL}}(p\,\|\,q_\theta) := \mathbb{E}_p\big[\log p(x) - \log q_\theta(x)\big].$$

- Population negative log-likelihood (NLL):

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

## Finite-sample view (empirical MLE)

Given i.i.d. data \(\{x_i\}_{i=1}^n\) drawn from \(p\), the empirical NLL is

$$\widehat{\mathrm{NLL}}_n(\theta) := \frac{1}{n} \sum_{i=1}^n -\log q_\theta(x_i).$$

By the law of large numbers,

$$\widehat{\mathrm{NLL}}_n(\theta) \xrightarrow[]{\;n\to\infty\;} \mathbb{E}_p\big[-\log q_\theta(x)\big] = H(p, q_\theta),$$

so minimizing the empirical NLL (the standard training loss) is a consistent estimator of the population objective, hence of \(D_{\mathrm{KL}}(p\,\|\,q_\theta)\) as well.

## Compression perspective (bits)

When entropy coding with model \(q_\theta\), the expected code length in bits is

$$\mathbb{E}_p\big[-\log_2 q_\theta(x)\big] = \frac{1}{\ln 2} \, \mathbb{E}_p\big[-\log q_\theta(x)\big] = \frac{1}{\ln 2}\, H(p, q_\theta).$$

Thus minimizing NLL in nats directly minimizes expected bits-per-sample (e.g., bits-per-pixel) up to the constant \(1/\ln 2\). Using conditional models (e.g., autoregressive context, hyperprior latents) preserves all identities by replacing \(q_\theta(x)\) with \(q_\theta(x\mid y)\) and taking expectations over the joint \(p(x,y)\).

## Takeaways

- Training with cross-entropy loss in AI compression is the same as minimizing forward KL from the data distribution to the model and the same as minimizing population NLL.
- Consequently, we are performing MLE (maximizing \(\mathbb{E}_p[\log q_\theta(x)]\)).
- Minimizing this loss also minimizes the expected code length produced by an ideal entropy coder fed with \(q_\theta\).

This unifies likelihood learning and compression: better likelihood under the data distribution means fewer expected bits to transmit the data. 