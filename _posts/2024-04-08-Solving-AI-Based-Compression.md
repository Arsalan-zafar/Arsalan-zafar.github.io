---
title: 'Solving AI based compression'
date: 2024-04-08
permalink: /posts/2024/04/solving-ai-based-compression/
tags:
  - AI based compression
  - AI codecs
  - video compression
published: True 
---

Deep Render recently shipped the world’s first AI codec to its customers. This result follows two years of careful research, gruelling engineering, and relentless product focus. I’ll take some time to share some thoughts on the research and engineering that went into solving AI-based compression.  

# The research

A key critique we have of the current AI-based compression field is their disregard for model complexity. The community has opted for improved compression performance through increased model size at the cost of production feasibility. At the extreme, you can argue that most gains made in AI-based compression over the last few years have simply been due to increasing model complexity. A symptom of this doctrine is the annual CLIC competition, which, to this day, lacks any target for model complexity.  

Model complexity matters because it’s intricately linked with compression performance. If we want to see AI codecs in production in the short term, we must optimise compression performance per operation (model complexity). It will come as no surprise that this is our north star. 

During the research phase for our codec, we jointly optimise compression performance and model complexity. We’ve developed systems that enable us to implement, train, and measure compression performance and model complexity within a few days. Given this streamlined workflow, we have diligently worked through major unsolved problems limiting AI-based compression: INT8 to float compression performance gap, temporal visual consistency, efficient edge device execution, and cross-platform consistency. 

As a fun sidenote, AI-based codecs, if not trained correctly, can produce fascinating videos. Here’s an example:

# The engineering

Productisation requires an immense amount of raw engineering. Historically, AI-based compression-esque companies have been heavily focused on research, demonstrated by the unimodal composition of their teams. A well-balanced team should be able to bridge innovations over the non-trivial gap between research and production, enabling the widespread adoption of exciting technologies. It will come as no surprise that Deep Render places an equal importance on engineering and research. We have teams that focus on ML engineering, Device integration and Infrastructure.  

## ML engineering

Developing several independent and complex innovations during the research phase is only part of the puzzle. The final model must provide the joint benefits of these innovations. It falls on our ML Engineering team to organise innovations, perform integration and training to deliver production-ready checkpoints. They run between 200-500 integration experiments to achieve the final stable models for each iteration of our codec. Training schedules, datasets and learning rate length are all hyperparameters that are ablated in parallel, as they can significantly impact compression performance.

Once the final weights are ready, our evaluation team uses their extensive in-house benchmarking library to thoroughly benchmark our models against all traditional codecs, resulting in an array of objective and subjective metrics. As part of these evaluations, we partner with Subjectify to crowd-source subjective evaluations worldwide, resulting in over 10,000 votes per evaluation.

## Device integration 

AI is a cloud-centric industry. The majority of the widely used AI applications run inference on the cloud. The simple reason for this is that executing well-performing methods efficiently on edge devices is challenging. To alleviate this, chip manufacturers are continually improving the hardware and software stack for edge AI. We’re seeing incredible inter-generation improvement in hardware capabilities. However, the APIs, the gateway to the hardware, are often rigid, poorly documented and bug-prone. This is to be expected, given where we are in the maturity cycle for AI. 

AI-based compression is a field that does not have the luxury of being cloud-centric, as decodes have to be run on edge devices. As an antidote to the nascency of edge AI APIs, Deep Render has partnered with all major hardware providers to collaborate on their development. This has created a feedback loop between our implementation process and the vendor’s development process, greatly speeding up our development. 

Deep Render follows the standard device porting logic. We produce JIT-traced models using TorchScript, which are consumed by vendor API converters to create modules capable of executing on edge devices. Deep Render often uses custom operators that are currently unsupported by vendor APIs. For these, we write highly optimised implementations in OpenCL or Metal. Once we have all modules on the device, we optimise memory movement, execution graph, operator selection, and parallelism using custom software developed at Deep Render. Finally, the optimised codec undergoes extensive testing using a framework set up by our infrastructure team, reporting back power consumption, frame rate and compression performance across various devices and sequences. Once the tests have passed, the codec is tagged for release. 

## Infrastructure

To support our engineering effort, we have an infrastructure team dedicated to maintaining code, testing, and hardware needs. Their primary aim is to create a pipeline that allows us to go from research code to results efficiently. They turn code into coding gains. Alongside optimising the software and testing infrastructure, they maintain our on-premise training systems, with over 150 compute nodes custom-built to solve compression.      

# Looking forward

Our fundamental belief is that AI is the tool that will allow humanity to reach the compression limit. Deep Render is spearheading this goal and is perfectly placed to make the most progress towards it. Our models have surpassed 45 years of incumbent research in two years, and we have internal research showing that we’ve barely scratched the surface. Who’s to say what the next two will bring?