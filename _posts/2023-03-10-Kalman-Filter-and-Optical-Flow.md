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

Coming soon: Can we use Kalman filters predictive model to predict motion at time $$t$$, given the motion at time $$t-1$$, $$t-2$$? 

In video compression, we have to send optical flow information to enable a warping opertation between the previous frame and the current frame. Though this costs only <10% of the total bits, it would still be benificial if we could predict this cheaply rather than send it though the bitsteeam. 

There is regularity and structure in motion, think about panning motion, which should be well modelled by linear quadratic approximator with gaussian error assumptions like a kalman filter.


