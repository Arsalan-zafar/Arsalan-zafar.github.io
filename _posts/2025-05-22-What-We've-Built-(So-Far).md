---
title: 'What We\'ve Built (So Far)'
date: 2025-05-22
permalink: /posts/2025/05/what-we-have-built-so-far/
tags:
  - AI based compression
  - AI codecs
  - video compression
published: True 
---

Deep Render was built from the [ground up](https://arsalan-zafar.github.io/posts/2024/04/solving-ai-based-compression/) to develop a production ready codec. We focused on developing an AI codec with low computational complexity and high compression efficiency. We worked with a relentless product focus, efficient tooling and process driven research. After 4 years in the trenches, we have developed a codec that can achieve real-time encode and decode with over 45% BD rate saving w.r.t. SVT-AV1. In other words, the world's first AI codec.

To demonstrate what we've built so far, we've made our visual comparison app ([eval.deeprender.ai](https://eval.deeprender.ai)) public - so everyone can assess our visual quality. We've also recently had [Jan Ozer](https://www.linkedin.com/in/jan-ozer/) (streaming media) and [Vittorio Baroncini](https://fub.academia.edu/VittorioBaroncini/CurriculumVitae) (ITU and MPEG testing chair) test our [codec and compression performance claims](https://streaminglearningcenter.com/codecs/deep-render-an-ai-codec-that-encodes-in-ffmpeg-plays-in-vlc-and-outperforms-svt-av1.html). In this blog, I'll highlight our key achievements to date and touch on what we're working towards now.

# Compression efficiency

## Subjective metrics 

Our primary focus for compression efficiency is with respect to visual quality, the gold standard for evaluating quality in the compression industry. At Deep Render, we have a department dedicated to AI-based explicit and implicit density estimation, which allows us to leverage methods like diffusion models and GANs to enhance visual quality. 

Verifying visual quality can be difficult, tedious and expensive. We have to pick between numerous methods that define how to structure visual studies, decide how to standardise viewing conditions, settle on a distribution strategy, crowdsource the results and finally, draw meaningful conclusions from the collected data in a format that is comprehensible. At Deep Render, we use three methods depending on the use case:

* **Internal subjective evaluations**: We use an application to distribute clips within Deep Render to collect votes.
* **Remotely crowdsourced votes through [Subjectify.us](https://subjectify.us)**: We use Subjectify to distribute clips through the internet to thousands of participants and produce [BSQ rate plots](https://www.researchgate.net/publication/340060891_BSQ-rate_a_new_approach_for_video-codec_performance_comparison_and_drawbacks_of_current_solutions).
* **In-person standardized professional lab evaluations**: We use a professional lab, such as Vittorio Baroncini VABTech lab to perform in-person standard p.910 DSIS ITU-T evaluations and retrieve BD-rates. This method is also used by the standard bodies when developing the ITU/MPEG codecs.

To demonstrate our compression performance rigorously, we recently had an [evaluation report](https://drive.google.com/file/d/1eU3gsBxfEFowRzxVYGcrsGK8BxgJLOe3/view) produced by Vittorio Baroncini, ITU/MPEG testing chair, using the P.910 DSIS method. These evaluations were performed in a professional laboratory setting with expert and naive viewers under the P.910 protocol. The results show that Deep Render achieves a ~45% bitrate reduction over SVT-AV1. Additionally, Jan Ozer published an evaluation of our subjective quality [here](https://streaminglearningcenter.com/codecs/deep-render-an-ai-codec-that-encodes-in-ffmpeg-plays-in-vlc-and-outperforms-svt-av1.html) and reached a similar conclusion, demonstrating the visual quality gain in [this](https://www.youtube.com/watch?v=D49ckpIoXB8&t=2s&ab_channel=JanOzer) video.

We believe this gain in visual quality is a significant result for the compression industry. It demonstrates that AI is the right tool to build the future of compression, and that future is here. 

## Objective metrics

Deep Render as an organisation does not focus on objective metrics, but we still share them for academic purposes. Generally, we've found traditional metrics (PSNR/VMAF/SSIM) to not fully capture the subjective gains provided by AI codecs. Nonetheless, our traditional metrics are generally on par in BD-rate with SVT-AV1, all other things being equal. Jan Ozer evaluated and published these results [here](https://streaminglearningcenter.com/codecs/deep-render-an-ai-codec-that-encodes-in-ffmpeg-plays-in-vlc-and-outperforms-svt-av1.html). We think this is still a significant result as it demonstrates that even after ignoring the subjective improvements, AI codecs are already on par with traditional codecs while providing significant improvements in encoding complexity, roll-out speed and rate of progress. 

## Future work

Our codec currently operates in low-delay, p-frame only mode. We are actively working on building random access features such as hierarchical mini GOPs and B-frames. Excitingly, these models already provide a 10% BD rate improvement over SVT-AV1 in RA mode, and we expect to be outperforming traditional codecs by 40-50% on subjective metrics by the end of the year. 

In addition to random access features, we're also working on adding presets and additional rate control modes to our codec. These will enable encoding houses to trade complexity and coding efficiency based on their use cases. 

# Computational complexity

A key critique of AI codecs is their inability to encode and decode without expensive GPUs. This is true for AI compression modes in academia, such as[DCVC](https://github.com/microsoft/DCVC) series from Microsoft, however, as demonstrated by Jan's testing, Deep Render's AI codec can seamlessly and efficiently encode and decode on everyday, widely available hardware while providing 45% BD rate gains. Without a doubt, this has been a key achievement of our team. Without this capability, AI codecs remain a mythical technology only future generations will unlock. 

## Encode & Decode

When encoding and decoding on Apple M series chips, our codec was able to achieve 22 and 70+ respectively as verified by Jan Ozer [here](https://www.youtube.com/watch?v=D49ckpIoXB8&t=2s&ab_channel=JanOzer). Internally, we have models that can hit 30+ and 100+ fps on encode and decode respectively which will go into production later this year. These internal models have benefited from research breakthroughs in the past two months, unlocking significant computational efficiency gains. Achieving a decode speed of 100 FPS on widely available hardware while providing a 45% BD-rate gain over AV1 demonstrates that AI is the right tool for compression and AI codecs are imminently ready to dominate the codec industry, with Deep Render leading the charge. 

## BDT (Battery drain time) 

Besides encode and decode speed, the concept of battery drain time is important for a production ready codec. Battery drain time is defined as the time taken to drain the battery from 100% to 0% on a given device while playing back a sequence. On phones from the last three years, our next production model will be able to achieve up to 15 hours of BDT, while the dav1d decoder is able to achieve around 16-18 hours, all other things being equal. Even though the dav1d decoder currently beats out Deep Render, we don’t see this as an issue since we’re seeing an impressive and consistent reduction in computational complexity every quarter. We also believe that a BDT of 15 hours on phones is above the threshold for widespread deployment.

## Future work

While we think a BDT of 15 hours is sufficient for significant deployment, we are confident we can increase this to 18-20 hours through methods such as content-adaptive encoding paths. We believe this can be achieved by early next year and will enable us to deploy AI codecs to an even wider audience by targeting lower-end and older devices. 

# Device reach

A key attraction of the Deep Render codec is that it does not require specialised hardware to decode. Deep Render uses widely available commodity hardware like the NPU available on Apple, Qualcomm, MediaTek and many other devices. This has two significant advantages.

Firstly, Deep Render is decoupled from the custom hardware development and adoption cycles as it does not require them for efficient encoding and decoding. Already, there are billions of devices that can encode and decode our AI codec efficiently. What's even more exciting is the rate at which the capability of these NPUs is improving. In the last four years, Apple NPUs have gone from 11 TOPs to 35 TOPs, Qualcomm NPUs from 26 to 50 TOPs and Intel NPUs will be going from 11 TOPs in Meteor Lake to 50+ in Lunar and Panther Lake chips. Deep Render can easily leverage this trend to further improve compression performance, BDT, and playback frame rates. 

Secondly, since our codec is essentially a software codec, we can readily deploy improved codecs with a higher frequency as we unlock more gains through our ongoing research efforts. The current 45% BD rate gain over AV1 is just the start, and we expect our coding gains to continue to improve as we continue to explore the frontier of AI. On top of this, we can build techniques such as [specialisation](https://arsalan-zafar.github.io/posts/2025/03/future-of-ai-based-compression/) into production systems to really enhance products and improve engagement.  

The codec world is used to having a new codec every 10 years and dealing with a slow adoption curve as hardware lags behind. AI codecs remove these constraints entirely while providing significantly more gains, flexibility and a brighter future 

# Deployability

Having a codec without any deployment infrastructure hinders testing, roll-out and ultimately adoption. Deep Render's Engineering team has been focused on building real world integrations. We've focused on making our AI codec easy to test, playback and deploy. 

To allow seamless testing and deployment, we developed [FFmpeg binaries](https://www.youtube.com/watch?v=Bk8iCrvZt5w&ab_channel=JanOzer) and parallel production APIs. Encoded clips are containerised using MP4 containers, ideal for DASH streaming. To stream encoded clips, we support the generation of manifest files (e.g., MPD for DASH) that describe available video qualities, bitrates, segment locations and other metadata. Finally, for playback on edge devices, we support playback through VLC which can playback DASH. 

While we think we've made significant progress in our integrations, we're continuing to build out our APIs to enable deeper and wider integration into the compression ecosystem. This year, we're adding a low-power mode to enable wider device reach, additional rate control modes, presets, and random access features. 

# Looking ahead

Our team has made incredible progress in the last few years, and we fundamentally believe that AI is the right tool to build codecs, as demonstrated by our results. However, we're just at the beginning of the AI codec development and progress arc. If we project ourselves 5 years into the future, without a doubt, the codec industry will be dominated by AI codecs with the following features:

* There will be multiple proprietary AI codecs on the market, and encoding houses will own their own AI codec, reducing reliance on standard bodies
* There will be different codecs per use case, media type, film title and even person
* NPUs will be widespread and boast 100+ TOPs of compute, enabling efficient and seamless deployment
* AI codecs will be providing 3- 4x improvements over traditional codecs

We think the world will benefit greatly from this future, and the Deep Render team will continue working toward this future.

---

*Special thanks to Allie, Clare, Chris and Sebastjan for the reviews.* 