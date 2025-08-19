---
title: 'Proof: MSE makes GANs locally stable'
date: 2021-07-21
permalink: /posts/2021/07/locally-stable-gans/
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


Generative Adversarial Networks are notoriously unstable due to issues such as mode collapse and training divergence. However, in AI compression, the adversarial training is generally stable and reliable without the neccessary tricks such as gradient penalty and adding noise to our input samples. I found this intriguing so I set out to explore why.    

One obvious difference is that in compression GANs, we always have access to the ground truth image that we aim to generate. That allows us to use pixel-wise distortion losses in generator (encoder-decoder) training. 

## Convergence of GANs

Our starting point is the (Mescheder et al., 2017)[https://arxiv.org/abs/1705.10461] Numerics of GANs paper. We can think of GAN training as a two-player non-cooperative game. The first player is a generator $$G_\theta(z)$$ with parameters $$\theta$$ that wants to maximize its payoff $$g(\theta, \psi)$$, the second player is a discriminator $$D_\psi(x)$$, with parameters $$\psi$$ that aims to maximize $$d(\theta, \psi)$$. The game is at a Nash equilibrium at $$(\theta^*, \psi^*)$$ when neither player can improve its payoff by changing its parameters slightly. When GAN reaches a Nash equilibrium, we can say that it reached local convergence.

One method to train a GAN is to use a Simultaneous Gradient Descent, which can be thought of as a fixed point algorithm that applies an operator $$F(\theta, \psi)$$ to the parameters of the generator and discriminator $$(\theta, \psi)$$ respectively:

$$F(\theta, \psi) = (\theta, \psi) + hv(\theta, \psi),$$

where $$h$$ is a learning rate and $$v(\theta, \psi)$$ is the Jacobian of $$L$$, our gradient:

$$v(\theta, \psi) = \begin{bmatrix} -\nabla_\theta L(\theta, \psi) \\ \nabla_\psi L(\theta, \psi) \end{bmatrix}$$

Mescheder et al. demonstrate that the convergence near an equilibrium point $$(\theta^*, \psi^*)$$ can be assessed by looking at the spectrum of Jacobian of our update operator $$F_h'(\theta, \psi)$$ at the point $$(\theta^*, \psi^*)$$:

• If all eigenvalues have absolute value **less than 1**, the system **converges to** $$(\theta^*, \psi^*)$$ with a linear rate (our desired case).

• If there are any eigenvalues with absolute values **greater than 1**, the system **diverges**.

• If all eigenvalues have an absolute value **equal to 1** (lie on the unit circle), it can be convergent, divergent or neither, but if it is convergent, it will generally converge with a sublinear rate.

Now let's look at a toy example to examine the conditions of convergence of GANs and discuss how the AI commpression objective impacts convergence. We'll use the Dirac-GAN for simplicity with the non-saturating and vanillar loss for our analysis. 

## Dirac-GAN

The Dirac-GAN consists of a (univariate) generator distribution $$p_g = \delta_\theta$$ and a linear discriminator $$D_\psi(x) = \psi x$$. The true data distribution $$p_D$$ is given by a Dirac-distribution concentrated at 0.

Under this formulation, both the discriminator and the generator has exactly one parameter. This simplicity allows us to easily plot the vector field for the GAN in a 2D space to assess convergence behaviour. For a great explanation of vector fields and convergence of GANs, check out this [inFERENCe blog post](https://inference.vc/my-notes-on-the-gan-literature/).

The non-saturated GAN generator loss is:

$$\max_\theta L(\theta, \psi) = f(-\psi\theta)$$

where $$f(t) = -\log(1 + e^{-t})$$. The discriminator loss is:

$$\max_\psi L(\theta, \psi) = f(\psi\theta) - const$$

The gradient vector is then:

$$v(\theta, \psi) = \begin{bmatrix} -\psi f'(-\psi\theta) \\ \theta f'(\psi\theta) \end{bmatrix}$$

The system has a unique equilibrium point of the training objective, at point $$(\theta, \psi) = (0, 0)$$. Indeed, since $$f(0) = const$$, $$L(\theta, 0) = L(0, \psi) = const$$ for all $$\theta, \psi \in \mathbb{R}$$. Therefore $$(\theta^*, \psi^*) = (0, 0)$$ is a Nash-equilibrium. Now, assuming that $$f'(0) \neq 0$$ we have that $$v(\theta, \psi) = 0$$ only if $$\theta = \psi = 0$$.

Now we can analyse the convergence of the Dirac-GAN around the equilibrium point $$(\theta^*, \psi^*) = (0, 0)$$ by calculating the Jacobian of the gradient vector field at the equilibrium point:

$$v'(\theta, \psi) = \begin{bmatrix} f''(-\theta\psi)\psi^2 & -f'(-\theta\psi) + f''(-\theta\psi)\theta\psi \\ f'(\theta\psi) + f''(\theta\psi)\theta\psi & f''(\theta\psi)\theta^2 \end{bmatrix}$$

$$v'(0, 0) = \begin{bmatrix} 0 & -f'(0) \\ f'(0) & 0 \end{bmatrix}$$

which has two eigenvalues, $$\pm f'(0)i$$, which are both on the imaginary axis. This means that the model is not convergent with a linear rate, as it has zero real parts and we expect to see a circular vector field.

## Adding MSE

Now, let us look at what happens by adding MSE to the loss for the generator parameters $$\theta$$. The discriminator loss stays the same, but the generator loss becomes:

$$\max_\theta L(\theta, \psi) = f(-\psi\theta) - (0 - \theta)^2$$

The gradient vector field now has an additional $$-2\theta$$ term is the $$\theta$$ partial derivative:

$$v(\theta, \psi) = \begin{bmatrix} -\psi f'(-\psi\theta) - 2\theta \\ \theta f'(-\psi\theta) \end{bmatrix}$$

The Jacobian of the vector field becomes:

$$v'(0, 0) = \begin{bmatrix} -2 & -f'(0) \\ f'(0) & 0 \end{bmatrix}$$

And eigenvalues of the Jacobian of the gradient vector field at the equilibrium point is $$-1 \pm \sqrt{1 - f'(0)}$$, both of which have negative real parts and 0 imaginary parts, which means that the system should be locally convergent around $$(0, 0)$$. The lack of imaginary parts means that there should not be any circular non-convergent behaviour in the gradient vector field. 

## Vanilla GAN loss

We can repeat the analysis for the vanilla GAN loss function. The gradient vector is then:

$$v(\theta, \psi) = \begin{bmatrix} -\psi f'(\psi\theta) \\ \theta f'(\psi\theta) \end{bmatrix}$$

As before, the system has a unique equilibrium point of the training objective at the point $$(\theta^*, \psi^*) = (0, 0)$$.

The Jacobian of the gradient vector field at the equilibrium point:

$$v'(\theta, \psi) = \begin{bmatrix} -f''(\theta\psi)\psi^2 & -f'(\theta\psi) - f''(\theta\psi)\theta\psi \\ f'(\theta\psi) + f''(\theta\psi)\theta\psi & f''(\theta\psi)\theta^2 \end{bmatrix}$$

$$v'(0, 0) = \begin{bmatrix} 0 & -f'(0) \\ f'(0) & 0 \end{bmatrix}$$

which has two eigenvalues, $$\pm f'(0)i$$, which are both on the imaginary axis. This means that the model is not convergent as it has zero real parts and we expect to see a circular vector field. Figure 4 demonstrates that Dirac-GAN with vanilla GAN loss is divergent around the equilibrium $$(0, 0)$$.

However, for similar reasons as above, the behaviour completely changes once we add the MSE component to the generator loss. The gradient vector field now has an additional $$-2\theta$$ term in the $$\theta$$ partial derivative:

$$v(\theta, \psi) = \begin{bmatrix} -\psi f'(\psi\theta) - 2\theta \\ \theta f'(\psi\theta) \end{bmatrix}$$

The Jacobian of the vector field becomes:

$$v'(0, 0) = \begin{bmatrix} -2 & -f'(0) \\ f'(0) & 0 \end{bmatrix}$$

And, just like in the case of non-saturating loss, eigenvalues of the Jacobian of the gradient vector field at the equilibrium point are $$-1 \pm \sqrt{1 - f'(0)}$$.

This analysis shows that adding MSE has the same impact as gradient regularisation and instance noise, which also remove the circular behaviours in the gradient field and force the negative real part in the eigenvalues. This analysis explains how GANs in compresison can sidestep the need for these regaularization methods buy simple using the MSE. 



## Further reading:

[1] Mescheder, Lars, Andreas Geiger, and Sebastian Nowozin. "Which training methods for GANs do actually converge?."
International conference on machine learning. PMLR, 2018. https://arxiv.org/pdf/1801.04406.pdf
[2] Mescheder, Lars, Sebastian Nowozin, and Andreas Geiger. "The numerics of gans."
arXiv preprint arXiv:1705.10461 (2017). https://arxiv.org/pdf/1705.10461.pdf
[3] Huszár Ferenc "GANs are Broken in More than One Way: The Numerics of GANs."
GANs are Broken in More than One Way: The Numerics of GANs. https://www.inference.vc/my-notes-on-the-numerics-of-gans/
[4] Bertsekas, Dimitri P. "Nonlinear programming." (1999). https://nms.kcl.ac.uk/osvaldo.simeone/bert.pdf
[5] Theisel, Holger, and Tino Weinkauf. "Vector field metrics based on distance measures of first order critical points." (2002). http://wscg.zcu.cz/wscg2002/Papers_2002/D49.pdf



