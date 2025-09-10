---
title: 'The evidence lower bound is compression'
date: 2019-02-28
permalink: /posts/2019/02/elbo-end-to-end-compression-equivalence/
tags:
  - variational inference
  - compression
  - ELBO
  - information theory
  - machine learning
published: False 
mathjax: true  
---

The connection between variational inference and compression has been one of the most profound insights in modern machine learning. Ballé's groundbreaking work on variational image compression revealed a fundamental mathematical equivalence: the Evidence Lower BOund (ELBO) objective from variational inference is mathematically equivalent to the rate-distortion objective used in end-to-end learned compression. This equivalence has transformed how we approach compression, moving from heuristic methods to principled variational frameworks.

Understanding this connection is crucial for anyone working in AI-based compression, as it provides the theoretical foundation that enables us to leverage the full power of variational methods for compression applications.

# The fundamental insight

At the heart of modern compression lies a trade-off between two competing objectives: how much information we need to transmit (rate) and how accurately we can reconstruct the original data (distortion). Traditional compression methods approached this through carefully engineered codecs. However, Ballé's insight was that this rate-distortion optimization is mathematically equivalent to maximizing the Evidence Lower BOund in variational inference.

The implications are profound: every advancement in variational methods can potentially improve compression performance, and every insight from compression can inform variational inference.

# Mathematical framework: The compression objective

## Rate-distortion formulation

In learned compression, we seek to minimize the expected rate-distortion Lagrangian:

$$\mathcal{L}_{RD} = \mathbb{E}_{p_X(x)} [R(x) + \lambda D(x, \hat{x})]$$

Where:
- $$R(x)$$ is the rate (number of bits) required to encode input $$x$$
- $$D(x, \hat{x})$$ is the distortion between original $$x$$ and reconstruction $$\hat{x}$$
- $$\lambda$$ is the rate-distortion trade-off parameter

## Encoder-decoder framework

Consider an autoencoder with stochastic encoder $$q_\phi(y \mid x)$$ and decoder $$p_\theta(\hat{x} \mid y)$$:

- **Encoder**: Maps input $$x$$ to latent representation $$y \sim q_\phi(y \mid x)$$
- **Decoder**: Reconstructs $$\hat{x}$$ from latent $$y$$ via $$p_\theta(\hat{x} \mid y)$$

The rate can be expressed using the compressed representation:

$$R(x) = \mathbb{E}_{q_\phi(y \mid x)} [-\log p_Y(y)]$$

Where $$p_Y(y)$$ is the prior distribution over latents, determining the coding efficiency.

# Mathematical framework: The ELBO objective

## Variational inference setup

In variational inference, we approximate an intractable posterior $$p(y \mid x)$$ with a tractable variational distribution $$q_\phi(y \mid x)$$. The ELBO provides a lower bound on the log marginal likelihood:

$$\log p(x) \geq \mathcal{L}_{ELBO}(x) = \mathbb{E}_{q_\phi(y \mid x)} [\log p_\theta(x \mid y)] - D_{KL}(q_\phi(y \mid x) \| p(y))$$

## Decomposition of the ELBO

The ELBO can be written as:

$$\mathcal{L}_{ELBO}(x) = \underbrace{\mathbb{E}_{q_\phi(y \mid x)} [\log p_\theta(x \mid y)]}_{\text{Reconstruction term}} - \underbrace{D_{KL}(q_\phi(y \mid x) \| p(y))}_{\text{Regularization term}}$$

The first term encourages accurate reconstruction, while the second term keeps the variational posterior close to the prior.

# The mathematical equivalence

## Key insight: Reinterpreting ELBO components

The profound insight is that each term in the ELBO directly corresponds to a component in the compression objective:

### Reconstruction term equivalence

The ELBO reconstruction term:
$$\mathbb{E}_{q_\phi(y \mid x)} [\log p_\theta(x \mid y)]$$

Under a Gaussian decoder assumption $$p_\theta(x \mid y) = \mathcal{N}(x; \mu_\theta(y), \sigma^2 I)$$, this becomes:

$$\mathbb{E}_{q_\phi(y \mid x)} [\log p_\theta(x \mid y)] = -\frac{1}{2\sigma^2} \mathbb{E}_{q_\phi(y \mid x)} [\|x - \mu_\theta(y)\|^2] + \text{const}$$

This is exactly the negative expected distortion:
$$\boxed{\mathbb{E}_{q_\phi(y \mid x)} [\log p_\theta(x \mid y)] \propto -D(x, \hat{x})}$$

### Rate term equivalence

The KL divergence term can be rewritten using the definition:

$$D_{KL}(q_\phi(y \mid x) \| p(y)) = \mathbb{E}_{q_\phi(y \mid x)} \left[\log \frac{q_\phi(y \mid x)}{p(y)}\right]$$

$$= \mathbb{E}_{q_\phi(y \mid x)} [\log q_\phi(y \mid x)] - \mathbb{E}_{q_\phi(y \mid x)} [\log p(y)]$$

The second term is exactly the expected code length under the prior:
$$\mathbb{E}_{q_\phi(y \mid x)} [-\log p(y)] = R(x)$$

Therefore:
$$\boxed{D_{KL}(q_\phi(y \mid x) \| p(y)) = H(q_\phi(y \mid x)) + R(x)}$$

Where $$H(q_\phi(y \mid x))$$ is the entropy of the encoder distribution.

## Complete equivalence derivation

Starting with the ELBO:
$$\mathcal{L}_{ELBO}(x) = \mathbb{E}_{q_\phi(y \mid x)} [\log p_\theta(x \mid y)] - D_{KL}(q_\phi(y \mid x) \| p(y))$$

Substituting our equivalences:
$$\mathcal{L}_{ELBO}(x) = -\frac{1}{2\sigma^2} D(x, \hat{x}) - H(q_\phi(y \mid x)) - R(x) + \text{const}$$

Rearranging and noting that $$H(q_\phi(y \mid x))$$ doesn't affect optimization when the encoder architecture is fixed:

$$\boxed{\max_{\phi,\theta} \mathcal{L}_{ELBO}(x) \equiv \min_{\phi,\theta} [R(x) + \lambda D(x, \hat{x})]}$$

Where $$\lambda = \frac{1}{2\sigma^2}$$ connects the decoder variance to the rate-distortion trade-off parameter.

# Practical implications

## Unified optimization framework

This equivalence means we can optimize compression systems using standard variational inference techniques:

**Reparameterization trick**: Enable gradient-based optimization through stochastic layers
**Amortized inference**: Learn encoder networks that efficiently approximate posteriors
**Hierarchical priors**: Use complex prior structures to improve compression efficiency

## Rate allocation

The KL term $$D_{KL}(q_\phi(y \mid x) \| p(y))$$ provides a principled way to allocate bits across different components of the latent representation. Elements of $$y$$ that deviate significantly from the prior $$p(y)$$ consume more bits, creating automatic rate allocation.

## Prior design

The choice of prior $$p(y)$$ directly impacts compression efficiency. This equivalence provides theoretical justification for learning optimal priors, leading to advances like:

**Hyperpriors**: Learning priors from data to minimize rate
**Context models**: Using spatial/temporal context to improve prior accuracy


# Looking forward

This mathematical equivalence between ELBO and compression objectives represents a foundational insight that continues to drive progress in AI-based compression. By establishing this connection, Ballé's work opened the door to leveraging decades of research in variational inference for compression applications.

The implications extend beyond compression itself. This equivalence demonstrates how information theory, variational inference, and practical engineering can be unified through careful mathematical analysis. As we continue to push the boundaries of what's possible with learned compression, this theoretical foundation ensures that our advances are principled and well-grounded.

The marriage of variational inference and compression exemplifies the kind of cross-pollination between fields that drives real progress in machine learning. By understanding these deep connections, we can build systems that are both theoretically sound and practically effective—exactly what's needed to realize the full potential of AI-based compression.

---

*This mathematical equivalence, first rigorously established in Ballé et al.'s work, continues to be fundamental to modern compression research and has enabled the practical deployment of learned compression systems that we see today.*
