---
title: 'Theory on GANs: Adding Pixel-Wise Losses for Local Stability'
date: 2025-01-15
permalink: /posts/2025/01/theory-on-gans-pixel-wise-losses/
tags:
  - GANs
  - Generative Adversarial Networks
  - Machine Learning
  - Deep Learning
  - Pixel-wise losses
  - Generative Compression Networks
published: True 
mathjax: true
---

*Originally by Vira Koshkina, July 21, 2021*

Generative Adversarial Networks are notoriously unstable and hard to train. Researchers have to employ a variety of tricks to make the adversarial training converge, constantly battling issues such as mode collapse and training divergence. Fortunately for us, when it comes to Generative Compression Networks (GCN), the adversarial training seems to be on its best behaviour, causing much less trouble.

# Introduction

Many tricks that are absolutely necessary for training GANs are not required in the compression pipeline. For example, we never observed a mode collapse, our GCNs train without gradient penalty, we don't need to add noise to our input samples to the decoder, and previous experiments indicate that pre-training of encoder-decoder is not necessary for getting stable discriminator training. 

Recently in the visual-loss team, we aim to understand how exactly our pipeline differs from standard GANs, what it means in terms of stability and convergence and why traditional GAN techniques are often not applicable.

One obvious difference is that in GCN, by nature of compression, we always have access to the ground truth image that we aim to generate. That allows us to use pixel-wise distortion losses in generator (encoder-decoder) training. In this blog post, we look at a toy example of GAN, examine the conditions of its convergence and discuss how our specific loss function impacts convergence.

# Convergence of GANs

For detailed proof of why eigenvalues need to lie in a unit circle, see [4] proposition 4.4.1, page 226. This condition directly translates to the eigenvalues of the Jacobian of the gradient vector **v**:

$$\mathbf{v} = \begin{bmatrix} \nabla_{\theta_g} L_g \\ \nabla_{\theta_d} L_d \end{bmatrix}$$

![Convergence Analysis](/images/gans_theory_page_2.png)

The convergence behavior depends on the eigenvalues:

- **If all eigenvalues have negative real-part**, the system converges with a linear convergence rate (with a small enough learning rate).
- **If there are eigenvalues with positive real-part**, the system diverges.
- **If all eigenvalues have zero real-part**, it can be convergent, divergent or neither, but if it is convergent, it will generally converge with a sublinear rate.

In fact, for a GAN with positive real parts to converge, we would require a negative step size, which is impossible since our learning rate is always above 0.

Intuitively we can understand this requirement if we think of GAN updates as a fixed point iteration. Then we can think of the updates as the Euler discretization of the first-order ODE of the gradient vector:

$$\frac{d\mathbf{x}}{dt} = \mathbf{v}(\mathbf{x})$$

where $\mathbf{x} = [\theta_g, \theta_d]^T$ represents the parameters.

![ODE Analysis](/images/gans_theory_page_3.png)

# Dirac-GAN

Having the criteria for convergence and a tool to visualise the behaviour in a 2D case, we can start the analysis of GAN models. To analyse the behaviour of different training methods and visualise the findings, the authors in [1] propose to look at a toy example called Dirac GAN.

Under this formulation, both the discriminator and the generator has exactly one parameter. This simplicity allows us to easily plot the vector field for the GAN in a 2D space. For a great explanation of vector fields and convergence of GANs, check out this [inFERENCe blog post](https://www.inference.vc/my-notes-on-the-numerics-of-gans/).

![Vector Field Classification](/images/gans_theory_page_4.png)

*Figure 1. Classification of critical points of a 2D vector field according to eigenvalues. R₁, R₂ denote the real parts of the eigenvalues of the Jacobian matrix and I₁, I₂ - the imaginary parts. The figure was modified from Figure 1 in [5].*

The non-saturated GAN generator loss is:

$$L_g = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

# Adding MSE

Above we look at the non-saturating GAN loss, used in standard generative modelling. Now, let us look at what happens once we adapt this to GCN loss, by adding MSE to the loss for the generator parameters θ. 

The discriminator loss stays the same, but the generator loss becomes:

$$L_g = -\mathbb{E}_{z \sim p_z}[\log D(G(z))] + \lambda \mathbb{E}_{x \sim p_{data}}[\|x - G(E(x))\|_2^2]$$

where $E$ is the encoder, $G$ is the generator (decoder), and $\lambda$ is the weighting parameter for the MSE term.

![MSE Loss Analysis](/images/gans_theory_page_5.png)

# Vanilla GAN Loss

We can repeat the analysis for the vanilla GAN loss function. The gradient vector is then:

$$\mathbf{v} = \begin{bmatrix} \nabla_{\theta_g} L_g \\ \nabla_{\theta_d} L_d \end{bmatrix} = \begin{bmatrix} \nabla_{\theta_g}[\mathbb{E}_{z}[\log(1-D(G(z)))] + \lambda \mathbb{E}_{x}[\|x - G(E(x))\|_2^2]] \\ \nabla_{\theta_d}[\mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]] \end{bmatrix}$$

![Vanilla GAN Analysis](/images/gans_theory_page_6.png)

The gradient vector behavior changes significantly when we incorporate the MSE component.

# Key Insights

This analysis shows that adding MSE has the same impact as gradient regularisation and instance noise, which also remove the circular behaviours in the gradient field and force the negative real part in the eigenvalues. This would explain why these solutions were not impactful when applied to GCNs.

# Conclusion

Adding MSE loss for the generator parameter completely changes the training behaviour of the system, making it converge and do it fast. **MSE loss acts as a regulariser.** This can potentially explain why our GCNs, which are always trained with MSE components in the loss, show much better convergence than standard GANs.

The theoretical foundation provided in this analysis helps explain the empirical success we've observed with Generative Compression Networks and their superior stability compared to traditional GANs.

![Analysis Summary](/images/gans_theory_page_7.png)

# References

[1] Mescheder, Lars, Andreas Geiger, and Sebastian Nowozin. "Which training methods for GANs do actually converge?." International conference on machine learning. PMLR, 2018. [https://arxiv.org/pdf/1801.04406.pdf](https://arxiv.org/pdf/1801.04406.pdf)

[2] Mescheder, Lars, Sebastian Nowozin, and Andreas Geiger. "The numerics of gans." arXiv preprint arXiv:1705.10461 (2017). [https://arxiv.org/pdf/1705.10461.pdf](https://arxiv.org/pdf/1705.10461.pdf)

[3] Huszár Ferenc "GANs are Broken in More than One Way: The Numerics of GANs."

[4] Bertsekas, Dimitri P. "Nonlinear programming." (1999). [https://nms.kcl.ac.uk/osvaldo.simeone/bert.pdf](https://nms.kcl.ac.uk/osvaldo.simeone/bert.pdf)

[5] Theisel, Holger, and Tino Weinkauf. "Vector field metrics based on distance measures of first order critical points." (2002). [http://wscg.zcu.cz/wscg2002/Papers_2002/D49.pdf](http://wscg.zcu.cz/wscg2002/Papers_2002/D49.pdf) 