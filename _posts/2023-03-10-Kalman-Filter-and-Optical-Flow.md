---
title: 'Kalman filter and optical flow'
date: 2021-05-14
permalink: /posts/2023/03/kalman-filter-and-optical-flow/
tags:
  - AI based compression
  - AI codecs
  - video compression
published: True 
mathjax: true  
---

In video compression, we have to send optical flow information to enable a warping operation between the previous frame and the current frame. Though this costs only <10% of the total bits, it would still be beneficial if we could predict this cheaply rather than send it through the bitstream. 

There is regularity and structure in motion, think about panning motion, which should be well modelled by linear quadratic approximator with gaussian error assumptions like a Kalman filter. This post explores how we can leverage Kalman filters to predict optical flow motion vectors, potentially eliminating the need to transmit this information entirely.

# The fundamental question

Can we use Kalman filters to predict motion at time $$t$$, given the motion at time $$t-1$$, $$t-2$$, and so on? The answer is promising. Given a set of optical flow maps, where each pixel predicts x, y motion, we can absolutely use the motion pixels from previous flow maps to predict the motion for the current flow map using a carefully designed Kalman filter model.

# Background: The Kalman filter framework

The Kalman filter is an optimal recursive estimator that provides the best linear unbiased estimate of a system's state given noisy observations. It operates under the assumption that both the system dynamics and observations follow Gaussian distributions - a reasonable assumption for many motion patterns in video sequences.

## Core mathematical framework

The Kalman filter operates on two fundamental equations that describe the system dynamics and observations:

**State transition model:**
$$\mathbf{x}_t = \mathbf{F}_t \mathbf{x}_{t-1} + \mathbf{B}_t \mathbf{u}_t + \mathbf{w}_t$$

**Observation model:**
$$\mathbf{z}_t = \mathbf{H}_t \mathbf{x}_t + \mathbf{v}_t$$

Where:
- $$\mathbf{x}_t$$ is the state vector at time $$t$$
- $$\mathbf{F}_t$$ is the state transition matrix
- $$\mathbf{B}_t$$ is the control input matrix  
- $$\mathbf{u}_t$$ is the control vector
- $$\mathbf{w}_t$$ is the process noise $$\sim \mathcal{N}(0, \mathbf{Q}_t)$$
- $$\mathbf{z}_t$$ is the observation vector
- $$\mathbf{H}_t$$ is the observation matrix
- $$\mathbf{v}_t$$ is the observation noise $$\sim \mathcal{N}(0, \mathbf{R}_t)$$

## The two-step prediction process

The Kalman filter operates through a predict-update cycle:

**Prediction step:**
$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}_t \hat{\mathbf{x}}_{t-1|t-1} + \mathbf{B}_t \mathbf{u}_t$$
$$\mathbf{P}_{t|t-1} = \mathbf{F}_t \mathbf{P}_{t-1|t-1} \mathbf{F}_t^T + \mathbf{Q}_t$$

**Update step:**
$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}_t^T (\mathbf{H}_t \mathbf{P}_{t|t-1} \mathbf{H}_t^T + \mathbf{R}_t)^{-1}$$
$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t (\mathbf{z}_t - \mathbf{H}_t \hat{\mathbf{x}}_{t|t-1})$$
$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-1}$$

Where $$\mathbf{K}_t$$ is the Kalman gain, $$\mathbf{P}_{t|t}$$ is the state covariance matrix, and $$\hat{\mathbf{x}}_{t|t}$$ represents the optimal state estimate.

# Modelling optical flow with Kalman filters

## State space design for motion prediction

For optical flow prediction, we need to design a state space that captures the motion dynamics of individual pixels. A natural choice is to model each pixel's motion vector along with its velocity:

$$\mathbf{x}_t = \begin{bmatrix} u_t \\ v_t \\ \dot{u}_t \\ \dot{v}_t \end{bmatrix}$$

Where:
- $$u_t, v_t$$ are the horizontal and vertical motion components
- $$\dot{u}_t, \dot{v}_t$$ are the motion velocities (change in motion)

## Motion dynamics model

For many real-world scenarios, we can assume constant velocity motion with some acceleration noise:

$$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

This model assumes that motion at time $$t$$ equals motion at time $$t-1$$ plus the velocity multiplied by the time step $$\Delta t$$. The process noise matrix $$\mathbf{Q}$$ captures uncertainty in acceleration:

$$\mathbf{Q} = \sigma_a^2 \begin{bmatrix} \frac{\Delta t^4}{4} & 0 & \frac{\Delta t^3}{2} & 0 \\ 0 & \frac{\Delta t^4}{4} & 0 & \frac{\Delta t^3}{2} \\ \frac{\Delta t^3}{2} & 0 & \Delta t^2 & 0 \\ 0 & \frac{\Delta t^3}{2} & 0 & \Delta t^2 \end{bmatrix}$$

## Observation model

Our observations are the motion vectors from optical flow estimation:

$$\mathbf{z}_t = \begin{bmatrix} u_{obs,t} \\ v_{obs,t} \end{bmatrix}$$

The observation matrix simply extracts the motion components:

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

# Practical example: Predicting the next motion pixel

Let's walk through a concrete example of how we'd predict motion for a single pixel across multiple frames.

## Initial setup

Consider a pixel that has exhibited the following motion vectors over the past three frames:
- $$t-2$$: $$(u, v) = (2.1, 0.3)$$
- $$t-1$$: $$(u, v) = (2.3, 0.4)$$  
- $$t-0$$: $$(u, v) = (2.5, 0.5)$$

## State initialisation

We initialise our state vector using the observed motion and estimated velocity:

$$\hat{\mathbf{x}}_{0|0} = \begin{bmatrix} 2.5 \\ 0.5 \\ 0.2 \\ 0.1 \end{bmatrix}$$

The initial velocities (0.2, 0.1) are estimated from the motion differences between consecutive frames.

## Prediction for next frame

Using our state transition model with $$\Delta t = 1$$:

$$\hat{\mathbf{x}}_{t+1|t} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 2.5 \\ 0.5 \\ 0.2 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 2.7 \\ 0.6 \\ 0.2 \\ 0.1 \end{bmatrix}$$

This predicts the next motion vector as $$(2.7, 0.6)$$ - a natural continuation of the observed motion pattern.

## Handling complex motion patterns

Real video sequences exhibit diverse motion patterns beyond simple linear motion. The Kalman filter framework can be extended to handle:

**Panning motions**: Global camera movement creates coherent motion fields that are highly predictable using simple constant velocity models.

**Rotating motions**: By incorporating angular velocity states, we can model rotational motion patterns common in camera movements.

**Zoom motions**: Radial motion patterns can be captured by modelling distance from frame center and radial velocity.

**Scene-specific adaptation**: Different motion models can be selected based on scene analysis, allowing the filter to adapt to specific motion characteristics.

# Practical implementation considerations

## Computational efficiency

A key advantage of this approach is computational efficiency. Rather than computing expensive optical flow estimation for every frame, we can:

1. Compute optical flow periodically (every N frames)
2. Use Kalman filter predictions for intermediate frames
3. Update predictions when new optical flow is available

This reduces computational cost while maintaining motion prediction accuracy.

## Integration with video codecs

In production video compression systems, this predictive approach offers several benefits:

**Bitrate reduction**: Predicted motion vectors require fewer bits to encode than raw optical flow
**Error resilience**: The Kalman filter provides graceful degradation when predictions are inaccurate
**Adaptive precision**: Prediction confidence can guide quantisation decisions for motion vector residuals

The regularity and structure in motion, particularly for common patterns like panning, make them well-suited for linear quadratic approximation with Gaussian error assumptions - exactly what Kalman filters excel at.

# Looking forward

This approach represents a paradigm shift in video compression, moving from transmitting all motion information to predicting it intelligently. As AI-based compression continues to evolve, sophisticated motion prediction models like Kalman filters will play an increasingly important role in achieving the compression efficiency gains needed for next-generation video applications.

The marriage of classical signal processing techniques like Kalman filtering with modern AI compression represents the kind of hybrid approach that will define the future of video technology. By leveraging the best of both worlds, we can build systems that are both theoretically grounded and practically effective.


