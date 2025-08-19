---
title: 'Locally stable GANs'
date: 2021-07-21
permalink: /posts/2025/01/theory-on-gans-pixel-wise-losses/
tags:
  - GANs
  - Generative Adversarial Networks
  - Machine Learning
  - Deep Learning
  - Pixel-wise losses
  - Generative Compression Networks
published: False 
mathjax: true
---

 

Generative Adversarial Networks are notoriously unstable due to issues such as mode collapse and training divergence. 

In AI compression, the adversarial training is generally stable and reliable without the neccessary tricks gradient penalty and adding noise to our input samples to the decoder. So how do GANs in a compression pipeline differ from standard GANs? 

One obvious difference is that in compression GANs, we always have access to the ground truth image that we aim to generate. That allows us to use pixel-wise distortion losses in generator (encoder-decoder) training. 

let's look at a toy example of GAN, examine the conditions of its convergence and discuss how our specific loss function impacts convergence.

## Dirac-GAN

The Dirac-GAN consists of a (univariate) generator distribution $$p_g = \delta_\theta$$ and a linear discriminator $$D_\psi(x) = \psi x$$. The true data distribution $$p_D$$ is given by a Dirac-distribution concentrated at 0.

Therefore, under this formulation, both the discriminator and the generator has exactly one parameter. This simplicity allows us to easily plot the vector field for the GAN in a 2D space. For a great explanation of vector fields and convergence of GANs, check out this [inFERENCe blog post](https://inference.vc/my-notes-on-the-gan-literature/).

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

Above we look at the non-saturating GAN loss, used in standard generative modelling. Now, let us look at what happens once we adapt this to GCN loss, by adding MSE to the loss for the generator parameters $$\theta$$. The discriminator loss stays the same, but the generator loss becomes:

$$\max_\theta L(\theta, \psi) = f(-\psi\theta) - (0 - \theta)^2$$

The gradient vector field now has an additional $$-2\theta$$ term is the $$\theta$$ partial derivative:

$$v(\theta, \psi) = \begin{bmatrix} -\psi f'(-\psi\theta) - 2\theta \\ \theta f'(-\psi\theta) \end{bmatrix}$$

The Jacobian of the vector field becomes:

$$v'(0, 0) = \begin{bmatrix} -2 & -f'(0) \\ f'(0) & 0 \end{bmatrix}$$

And eigenvalues of the Jacobian of the gradient vector field at the equilibrium point is $$-1 \pm \sqrt{1 - f'(0)}$$, both of which have negative real parts and 0 imaginary parts, which means that the system should be locally convergent around $$(0, 0)$$. The lack of imaginary parts means that there should not be any circular non-convergent behaviour in the gradient vector field. If we now run the Dirac GAN training with this updated generator loss function, we should see convergence and attracting gradient field if the theory matches the practical results. We demonstrate the results in Figure 3.

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

*Figure 5. Training behaviours of Dirac-GAN with added MSE. Left: Vector field and the training state of the system visualise for 500 steps. The starting position is marked in red. Right: Current value of the generator parameter $$\theta$$. The green line indicated the value of the gt parameter $$\theta^* = 0$$, and the blue line shows the estimated $$\theta$$.*

*The orange line shows the current angle of the gradient.*

This analysis shows that adding MSE has the same impact as gradient regularisation and instance noise, which also remove the circular behaviours in the gradient field and force the negative real part in the eigenvalues. This would explain why these solutions were not impactful when applied to GCNs.


