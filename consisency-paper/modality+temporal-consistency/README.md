<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### Modality + Temporal Consistency

**Capability Profile**: Models that transform textual descriptions or static images into temporally coherent, dynamic video sequences.

**Significance**: Currently the most prominent integration direction, enabling high-quality text-to-video and image-to-video generation.

#### Representative Works

<details>
<summary><b>Lumiere: A Space-Time Diffusion Model for Video Generation</b></summary>

* **Authors:** Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, Inbar Mosseri  
* **arXiv ID:** 2401.12945  
* **One-liner:** A text-to-video diffusion model with a space-time U-Net that generates the full video in one pass for realistic, coherent motion. 
* **Published in:** SIGGRAPH-ASIA 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2401.12945) | [[PDF]](https://arXiv.org/pdf/2401.12945.pdf) | [[Project Page]](https://lumiere-video.github.io/)

> **Core Innovation**  
> Proposes a Space-Time U-Net architecture that processes both spatial and temporal dimensions in a single forward pass, generating full-duration videos in one shot and thereby eliminating the temporal-inconsistency issues inherent in previous multi-stage key-frame + temporal super-resolution cascades.

<details>
    <summary>Abstract</summary>
    We introduce Lumiere -- a text-to-video diffusion model designed for synthesizing videos that portray realistic, diverse and coherent motion -- a pivotal challenge in video synthesis. To this end, we introduce a Space-Time U-Net architecture that generates the entire temporal duration of the video at once, through a single pass in the model. This is in contrast to existing video models which synthesize distant keyframes followed by temporal super-resolution -- an approach that inherently makes global temporal consistency difficult to achieve. By deploying both spatial and (importantly) temporal down- and up-sampling and leveraging a pre-trained text-to-image diffusion model, our model learns to directly generate a full-frame-rate, low-resolution video by processing it in multiple space-time scales. We demonstrate state-of-the-art text-to-video generation results, and show that our design easily facilitates a wide range of content creation tasks and video editing applications, including image-to-video, video inpainting, and stylized generation.
</details>

<details>
    <summary>Key points</summary>
    * Space-Time U-Net that jointly handles spatial and temporal down/up-sampling in one pass.  
    * Bypasses multi-stage temporal super-resolution cascades.  
    * Demonstrates improved motion coherence and diverse video generation.  
</details>
</details>

---

<details>
<summary><b>Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets</b></summary>

* **Authors:** Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach  
* **arXiv ID:** 2311.15127  
* **One-liner:** A latent video diffusion framework scaling to large datasets for high-resolution text-to-video and image-to-video generation.
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2311.15127) | [[PDF]](https://arxiv.org/pdf/2311.15127.pdf) | [[Code]](https://github.com/Stability-AI/generative-models)

> **Core Innovation**  
> We present a systematic three-stage pipeline for training latent video diffusion models—text-to-image pretraining, video pretraining, and high-quality video fine-tuning—and underscore the critical role of curated data selection and captioning, enabling high-resolution generation and multi-view 3D priors at scale.

<details>
    <summary>Abstract</summary>
    We present Stable Video Diffusion - a latent video diffusion model for high-resolution, state-of-the-art text-to-video and image-to-video generation. Recently, latent diffusion models trained for 2D image synthesis have been turned into generative video models by inserting temporal layers and finetuning them on small, high-quality video datasets. However, training methods in the literature vary widely, and the field has yet to agree on a unified strategy for curating video data. In this paper, we identify and evaluate three different stages for successful training of video LDMs: text-to-image pretraining, video pretraining, and high-quality video finetuning. Furthermore, we demonstrate the necessity of a well-curated pretraining dataset for generating high-quality videos and present a systematic curation process to train a strong base model, including captioning and filtering strategies. We then explore the impact of finetuning our base model on high-quality data and train a text-to-video model that is competitive with closed-source video generation. We also show that our base model provides a powerful motion representation for downstream tasks such as image-to-video generation and adaptability to camera motion-specific LoRA modules. Finally, we demonstrate that our model provides a strong multi-view 3D-prior and can serve as a base to finetune a multi-view diffusion model that jointly generates multiple views of objects in a feedforward fashion, outperforming image-based methods at a fraction of their compute budget.
</details>

<details>
    <summary>Key points</summary>
    * Identifies three training stages: text-to-image pretraining, video pretraining, high-quality video fine-tuning. 
    * Highlights the importance of well-curated large scale video data for video diffusion performance.  
    * Demonstrates latent video diffusion’s capability to act as a multi-view 3D prior and to enable image-to-video generation and camera motion control.  
</details>
</details>

---

<details>
<summary><b>AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning</b></summary>

* **Authors:** Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, Bo Dai  
* **arXiv ID:** 2307.04725  
* **One-liner:** A plug-and-play motion module that animates any personalized text-to-image diffusion model without requiring model-specific tuning. 
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.04725) | [[PDF]](https://arxiv.org/pdf/2307.04725.pdf) | [[Code]](https://github.com/guoyww/AnimateDiff)

> **Core Innovation**  
> Proposes a plug-and-play Motion Module: trained once on top of a frozen text-to-image diffusion model, it integrates seamlessly to turn any existing T2I checkpoint into an animation generator. Further introduces MotionLoRA—a lightweight fine-tuning recipe for fast adaptation to new motion patterns without retraining the full model.

<details>
    <summary>Abstract</summary>
    With the advance of text-to-image (T2I) diffusion models (e.g., Stable Diffusion) and corresponding personalization techniques such as DreamBooth and LoRA, everyone can manifest their imagination into high-quality images at an affordable cost. However, adding motion dynamics to existing high-quality personalized T2Is and enabling them to generate animations remains an open challenge. In this paper, we present AnimateDiff, a practical framework for animating personalized T2I models without requiring model-specific tuning. At the core of our framework is a plug-and-play motion module that can be trained once and seamlessly integrated into any personalized T2Is originating from the same base T2I. Through our proposed training strategy, the motion module effectively learns transferable motion priors from real-world videos. Once trained, the motion module can be inserted into a personalized T2I model to form a personalized animation generator. We further propose MotionLoRA, a lightweight fine-tuning technique for AnimateDiff that enables a pre-trained motion module to adapt to new motion patterns, such as different shot types, at a low training and data collection cost. We evaluate AnimateDiff and MotionLoRA on several public representative personalized T2I models collected from the community. The results demonstrate that our approaches help these models generate temporally smooth animation clips while preserving the visual quality and motion diversity. 
</details>

<details>
    <summary>Key points</summary>
    * A motion module that can be plugged into any personalized T2I model – no model-specific re-training required.  
    * MotionLoRA: lightweight fine-tuning for adapting motion patterns with low cost.  
    * Enables high-quality animation generation from existing personalized image models.  
</details>
</details>

---

<details>
<summary><b>Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning</b></summary>

* **Authors:** Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, Ishan Misra  
* **arXiv ID:** 2311.10709  
* **One-liner:** A two-stage text-to-video generation: first image then video conditioned on both text and generated image. 
* **Published in:** ECCV 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2311.10709) | [[PDF]](https://arxiv.org/pdf/2311.10709.pdf) | [[Project Page]](https://emu-video.metademolab.com/)

> **Core Innovation**  
> Proposes a factorized generation pipeline: first synthesize an image conditioned on text, then generate a video conditioned on both that image and the same text. With custom noise schedules and multi-stage training strategies, it bypasses the limitations of previous deep cascades and directly produces high-quality, high-resolution videos.

<details>
    <summary>Abstract</summary>
    We present Emu Video, a text-to-video generation model that factorizes the generation into two steps: first generating an image conditioned on the text, and then generating a video conditioned on the text and the generated image. We identify critical design decisions--adjusted noise schedules for diffusion, and multi-stage training that enable us to directly generate high quality and high resolution videos, without requiring a deep cascade of models as in prior work. In human evaluations, our generated videos are strongly preferred in quality compared to all prior work--81% vs. Google's Imagen Video, 90% vs. Nvidia's PYOCO, and 96% vs. Meta's Make-A-Video. Our model outperforms commercial solutions such as RunwayML's Gen2 and Pika Labs. Finally, our factorizing approach naturally lends itself to animating images based on a user's text prompt, where our generations are preferred 96% over prior work.
</details>

<details>
    <summary>Key points</summary>
    * Factorization into image generation → video generation conditioned on (text + generated image).  
    * Adjusted diffusion noise schedules and multi-stage training to enable high-quality, high-resolution video generation.  
    * Demonstrates significant quality improvements over previous text-to-video approaches.  
</details>
</details>

---

<details>
<summary><b>VideoPoet: A Large Language Model for Zero-Shot Video Generation</b></summary>

* **Authors:** Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang  
* **arXiv ID:** 2312.14125  
* **One-liner:** A decoder-only transformer trained like an LLM to perform zero-shot video generation from a wide variety of conditioning signals (text, image, audio, video). 
* **Published in:** ICML 2024 
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14125) | [[PDF]](https://arxiv.org/pdf/2312.14125.pdf) | [[Project Page]](https://sites.research.google/videopoet/)

> **Core Innovation**  
> Extends the large language model training paradigm to video generation: a decoder-only Transformer processes multimodal inputs (text, image, audio, video) and produces high-quality videos (with matching audio) in a zero-shot setting, opening a new route for generative video modeling.

<details>
    <summary>Abstract</summary>
    We present VideoPoet, a language model capable of synthesizing high-quality video, with matching audio, from a large variety of conditioning signals. VideoPoet employs a decoder-only transformer architecture that processes multimodal inputs -- including images, videos, text, and audio. The training protocol follows that of Large Language Models (LLMs), consisting of two stages: pretraining and task-specific adaptation. During pretraining, VideoPoet incorporates a mixture of multimodal generative objectives within an autoregressive Transformer framework. The pretrained LLM serves as a foundation that can be adapted for a range of video generation tasks. We present empirical results demonstrating the model's state-of-the-art capabilities in zero-shot video generation, specifically highlighting VideoPoet's ability to generate high-fidelity motions.
</details>

<details>
    <summary>Key points</summary>
    * Uses a decoder-only transformer that ingests multimodal signals (text, image, video, audio).  
    * Training protocol similar to LLMs: large-scale pretraining + adaptation.  
    * Enables zero-shot video generation across many conditioning types.  
</details>
</details>

---

<details>
<summary><b>UniVG: Towards UNIfied-modal Video Generation</b></summary>

* **Authors:** Ludan Ruan, Lei Tian, Chuanwei Huang, Xu Zhang, Xinyan Xiao
* **arXiv ID:** 2401.09084
* **One-liner:** This work introduces UniVG, a unified modal video generation system that reclassifies multi-modal video tasks such as text/image-driven generation based on “generative degrees of freedom.” Within a single diffusion framework, it simultaneously addresses high-degree-of-freedom (text/image-driven generation) and low-degree-of-freedom (image animation, super-resolution) video generation using multi-modal cross-attention and biased Gaussian noise. UniVG uniformly supports diverse tasks while outperforming existing methods in both metrics and subjective quality.
* **Published in:** ICME 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2401.09084) | [[PDF]](https://arxiv.org/pdf/2401.09084) | [[Code]](https://univg-baidu.github.io/)

> **核心创新**  
> propose a unified perspective of “generative freedom” to categorize diverse video generation tasks into high-freedom and low-freedom classes. For both categories, we design a comprehensive mechanism—combining multi-condition cross-attention with biased Gaussian noise—within a unified diffusion framework. This approach achieves true unification of multimodal, multi-task video generation within a single model.
<details>
    <summary>Abstract</summary>
    Diffusion based video generation has received extensive attention and achieved considerable success within both the academic and industrial communities. However, current efforts are mainly concentrated on single-objective or single-task video generation, such as generation driven by text, by image, or by a combination of text and image. This cannot fully meet the needs of real-world application scenarios, as users are likely to input images and text conditions in a flexible manner, either individually or in combination. To address this, we propose a Unified-modal Video Genearation system that is capable of handling multiple video generation tasks across text and image modalities. To this end, we revisit the various video generation tasks within our system from the perspective of generative freedom, and classify them into high-freedom and low-freedom video generation categories. For high-freedom video generation, we employ Multi-condition Cross Attention to generate videos that align with the semantics of the input images or text. For low-freedom video generation, we introduce Biased Gaussian Noise to replace the pure random Gaussian Noise, which helps to better preserve the content of the input conditions. Our method achieves the lowest Fréchet Video Distance (FVD) on the public academic benchmark MSR-VTT, surpasses the current open-source methods in human evaluations, and is on par with the current close-source method Gen2. 
</details>

<details>
    <summary>Key points</summary>
    * Proposes the UniVG unified modality video generation system, supporting simultaneous text-to-video, image-to-video, and text+image joint-condition video generation within a diffusion framework
    * Introduces the concept of “generative freedom,” categorizing video generation tasks into two main types: high-freedom (creative generation) and low-freedom (content-constrained/editing). This perspective reorganizes task and model design
    * For high-freedom tasks, designs Multi-condition Cross Attention within U-Net to simultaneously process text and image conditions, enabling semantically consistent video generation under arbitrary combinations.
    * For low-freedom tasks (e.g., image animation, video super-resolution), proposes Biased Gaussian Noise to align diffusion noise with input condition distributions, thereby better preserving input content and structure.
    * Achieved the lowest FVD on benchmarks like MSR-VTT, with subjective evaluations surpassing existing open-source methods and approaching Gen-2, demonstrating the unified system's generalization capabilities and practical value across multiple tasks
</details>
</details>

---

<details>
<summary><b>VideoGen: A Reference-Guided Latent Diffusion Approach for High Definition Text-to-Video Generation</b></summary>

* **Authors:** Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, Jingdong Wang
* **arXiv ID:** 2309.00398
* **One-liner:** VideoGen proposes a text-to-video generation framework combining “reference images + cascaded latent space diffusion + streaming temporal upsampling.” By first generating high-quality reference images from text, then guiding latent space video diffusion and high-definition decoding, it significantly enhances the spatial clarity and temporal consistency of generated videos, achieving state-of-the-art performance at the time.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2309.00398) | [[PDF]](https://arxiv.org/pdf/2309.00398) | [[Code]](https://videogen.github.io/VideoGen/)

> **核心创新**  
> VideoGen integrates “T2I for generating high-quality reference images + cascaded latent space diffusion for video generation + streaming temporal upscaling + separately trained video decoder” into a unified reference image-guided T2V framework. By adopting the approach of “image-first content definition and diffusion-focused motion learning,” it simultaneously achieves significant improvements in both clarity and temporal consistency for text-to-video generation.
<details>
    <summary>Abstract</summary>
    In this paper, we present VideoGen, a text-to-video generation approach, which can generate a high-definition video with high frame fidelity and strong temporal consistency using reference-guided latent diffusion. We leverage an off-the-shelf text-to-image generation model, e.g., Stable Diffusion, to generate an image with high content quality from the text prompt, as a reference image to guide video generation. Then, we introduce an efficient cascaded latent diffusion module conditioned on both the reference image and the text prompt, for generating latent video representations, followed by a flow-based temporal upsampling step to improve the temporal resolution. Finally, we map latent video representations into a high-definition video through an enhanced video decoder. During training, we use the first frame of a ground-truth video as the reference image for training the cascaded latent diffusion module. The main characterises of our approach include: the reference image generated by the text-to-image model improves the visual fidelity; using it as the condition makes the diffusion model focus more on learning the video dynamics; and the video decoder is trained over unlabeled video data, thus benefiting from high-quality easily-available videos. VideoGen sets a new state-of-the-art in text-to-video generation in terms of both qualitative and quantitative evaluation.
</details>

<details>
    <summary>Key points</summary>
    * use a mature T2I model to generate high-quality reference images from text, then conditionally guide subsequent video generation to enhance spatial clarity and semantic alignment
    * Design a cascaded latent diffusion module within the VAE latent space to uniformly handle generation from text + reference images to video latent representations, reducing computational costs in the pixel space
    * After obtaining low-frame-rate latent videos, introduce optical flow-based temporal upsampling to extend short, low-frame-rate clips into longer, higher-frame-rate videos, enhancing temporal continuity.
    * Train a dedicated high-quality video decoder to map latent-space videos into high-definition outputs, fully leveraging unlabeled real video data during training to improve detail and realism.
    * Achieves new SOTA on multiple text-to-video generation benchmarks, outperforming mainstream methods in both frame quality and temporal consistency, validating the effectiveness of the “image-first content definition + latent diffusion motion learning” approach
</details>
</details>

---

<details>
<summary><b>VideoPoet: A Large Language Model for Zero-Shot Video Generation</b></summary>

* **Authors:** Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, Lu Jiang
* **arXiv ID:** 2312.14125
* **One-liner:** This work introduces VideoPoet—a foundational video generation model that uniformly tokenizes video (including audio) into discrete units. It employs a decoder-based large language model for pre-training and fine-tuning across multimodal, multi-task objectives, enabling high-quality zero-shot video generation from diverse inputs such as text, images, and video.
* **Published in:** ICML 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14125) | [[PDF]](https://arxiv.org/pdf/2312.14125) | [[Code]](https://sites.research.google/videopoet/)

> **核心创新**  
> Unify the discretization of video (including audio) into tokens, then employ a purely decoder-based, LLM-style pre-trained multimodal Transformer to simultaneously model text/image/video/audio. This directly transforms the “video generation” problem into multimodal autoregressive language modeling, enabling zero-shot video generation within a unified framework.
<details>
    <summary>Abstract</summary>
    We present VideoPoet, a language model capable of synthesizing high-quality video, with matching audio, from a large variety of conditioning signals. VideoPoet employs a decoder-only transformer architecture that processes multimodal inputs -- including images, videos, text, and audio. The training protocol follows that of Large Language Models (LLMs), consisting of two stages: pretraining and task-specific adaptation. During pretraining, VideoPoet incorporates a mixture of multimodal generative objectives within an autoregressive Transformer framework. The pretrained LLM serves as a foundation that can be adapted for a range of video generation tasks. We present empirical results demonstrating the model's state-of-the-art capabilities in zero-shot video generation, specifically highlighting VideoPoet's ability to generate high-fidelity motions
</details>

<details>
    <summary>Key points</summary>
    * Uniformly discretize video frame sequences and audio into “visual/audio tokens,” processing them with the same discrete representation as text for large language models
    * Employ a decoder-only Transformer architecture similar to LLMs, simultaneously processing multimodal inputs like text, images, video, and audio to autoregressively generate video (and audio) tokens
    * Adhering to the “large language model” paradigm, perform large-scale multimodal autoregressive pretraining followed by minimal task-specific fine-tuning to cover diverse video generation tasks within a unified framework
    * Supports multiple tasks within a single model, including text-to-video, image-to-video, video editing, stylization, completion, and video-to-audio conversion, demonstrating strong zero-shot generalization across diverse evaluations
    * Achieves state-of-the-art performance in motion coherence and content diversity without retraining for individual tasks, excelling particularly in modeling large-scale, complex motion
</details>
</details>

---

<summary><b>Imagen Video: High Definition Video Generation with Diffusion Models</b></summary>

* **Authors:** Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, Tim Salimans
* **arXiv ID:** 2210.02303
* **One-liner:** This work introduces Imagen Video, a high-resolution text-to-video generation system built upon a cascaded video diffusion model. Through its design—comprising “base video generation + spatial/temporal super-resolution cascading + v-parameterization + progressive distillation”—it significantly elevates text-to-video resolution, duration, and temporal consistency to high-definition levels.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2210.02303) | [[PDF]](https://arxiv.org/pdf/2210.02303) | [[Code]](https://imagen.research.google/video/)

> **核心创新**  
> Using a foundational video diffusion model combined with an alternating temporal/spatial super-resolution submodel, alongside v-parameterization, classifier-free guidance, and progressive distillation, we achieve the first unified architecture to simultaneously push text-to-video resolution, duration, and temporal consistency to high-definition levels.
<details>
    <summary>Abstract</summary>
    We present Imagen Video, a text-conditional video generation system based on a cascade of video diffusion models. Given a text prompt, Imagen Video generates high definition videos using a base video generation model and a sequence of interleaved spatial and temporal video super-resolution models. We describe how we scale up the system as a high definition text-to-video model including design decisions such as the choice of fully-convolutional temporal and spatial super-resolution models at certain resolutions, and the choice of the v-parameterization of diffusion models. In addition, we confirm and transfer findings from previous work on diffusion-based image generation to the video generation setting. Finally, we apply progressive distillation to our video models with classifier-free guidance for fast, high quality sampling. We find Imagen Video not only capable of generating videos of high fidelity, but also having a high degree of controllability and world knowledge, including the ability to generate diverse videos and text animations in various artistic styles and with 3D object understanding.
</details>

<details>
    <summary>Key points</summary>
    * Proposes a 7-stage cascaded video diffusion system comprising “base video generation + spatial super-resolution + temporal super-resolution,” enabling high-resolution text-to-video conversion within a unified framework
    * Generates approximately 5.3 seconds of high-definition video at 1280×768 resolution, 128 frames, and 24fps—significantly improving resolution and duration compared to prior work
    * Validates and transfers key techniques from text-to-image diffusion to the video domain, including freezing the text encoder, classifier-free guidance, and conditioning augmentation
    * Introduced v-parameterization to enhance sampling quality and substantially reduced sampling steps via progressive distillation while maintaining video generation quality under text conditioning
    * Demonstrated strong controllability and language understanding capabilities in 3D object comprehension, text animation, and artistic style transfer, enabling generation of multi-style, highly consistent complex scene videos
</details>
</details>


<summary><b>Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models</b></summary>

* **Authors:** Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, Karsten Kreis
* **arXiv ID:** 2304.08818
* **One-liner:** This work introduces VideoLDM: by simply adding and aligning the temporal dimension to a pre-trained image LDM, it achieves high-resolution, temporally consistent text/condition-based video generation and video super-resolution. It delivers high-quality results on real-world driving scenarios and text-to-video tasks while significantly reducing computational costs for video generation.
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2304.08818) | [[PDF]](https://arxiv.org/pdf/2304.08818) | [[Code]](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)

> **核心创新**  
> By introducing the temporal dimension with minimal modifications to existing latent diffusion models, we extend static image generators into video generators through temporal convolutions and cross-frame attention. This approach enables high-resolution, temporally coherent text/condition-based video generation and video super-resolution with negligible increases in parameters and training costs.
<details>
    <summary>Abstract</summary>
    Latent Diffusion Models (LDMs) enable high-quality image synthesis while avoiding excessive compute demands by training a diffusion model in a compressed lower-dimensional latent space. Here, we apply the LDM paradigm to high-resolution video generation, a particularly resource-intensive task. We first pre-train an LDM on images only; then, we turn the image generator into a video generator by introducing a temporal dimension to the latent space diffusion model and fine-tuning on encoded image sequences, i.e., videos. Similarly, we temporally align diffusion model upsamplers, turning them into temporally consistent video super resolution models. We focus on two relevant real-world applications: Simulation of in-the-wild driving data and creative content creation with text-to-video modeling. In particular, we validate our Video LDM on real driving videos of resolution 512 x 1024, achieving state-of-the-art performance. Furthermore, our approach can easily leverage off-the-shelf pre-trained image LDMs, as we only need to train a temporal alignment model in that case. Doing so, we turn the publicly available, state-of-the-art text-to-image LDM Stable Diffusion into an efficient and expressive text-to-video model with resolution up to 1280 x 2048. We show that the temporal layers trained in this way generalize to different fine-tuned text-to-image LDMs. Utilizing this property, we show the first results for personalized text-to-video generation, opening exciting directions for future content creation.
</details>

<details>
    <summary>Key points</summary>
    * Building upon the pre-trained image LDM, we seamlessly extend the static image generator into a video generator by introducing a temporal dimension in the latent space and fine-tuning the temporal layer.
    * Training temporal convolutional/attention layers within the VAE latent space aligns latent representations across frames, significantly enhancing temporal consistency while preserving high-resolution details.
    * By performing “temporal alignment” fine-tuning on the diffusion upscaler, the original image super-resolution diffusion model is transformed into a “temporally coherent” video super-resolution model.
    * Achieved SOTA or near-SOTA results on real-world driving scene video synthesis and text-to-video generation tasks, demonstrating the method's practicality for industrial and creative content scenarios
    * The temporal layer is transferable across different Stable Diffusion variants, enabling personalized text-to-video generation (e.g., DreamBooth-style customization) and reducing the cost of building dedicated T2V models
</details>
</details>

---

<summary><b>Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation</b></summary>

* **Authors:** David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, Mike Zheng Shou
* **arXiv ID:** 2309.15818
* **One-liner:** This work introduces Show-1, the first hybrid text-to-video model featuring a “pixel diffusion + latent space diffusion” cascade: first generating low-resolution yet text-aligned videos via pixel-domain video diffusion, then performing super-resolution using latent space diffusion. This approach combines the strong alignment of pixel models with the efficiency of latent models, achieving state-of-the-art performance across multiple T2V benchmarks.
* **Published in:** IJCV 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2309.15818) | [[PDF]](https://arxiv.org/pdf/2309.15818) | [[Code]](https://github.com/showlab/Show-1)

> **核心创新**  
> Show-1 pioneers a truly collaborative cascaded hybrid framework combining “pixel-domain video diffusion + latent-space video diffusion”: it uses pixel diffusion to generate low-resolution videos with strong text alignment, then upscales them to high resolution via latent-space “expert translation.” This simultaneously resolves the classic trade-off between pixel models being computationally expensive and latent models having poor alignment.
<details>
    <summary>Abstract</summary>
    Significant advancements have been achieved in the realm of large-scale pre-trained text-to-video Diffusion Models (VDMs). However, previous methods either rely solely on pixel-based VDMs, which come with high computational costs, or on latent-based VDMs, which often struggle with precise text-video alignment. In this paper, we are the first to propose a hybrid model, dubbed as Show-1, which marries pixel-based and latent-based VDMs for text-to-video generation. Our model first uses pixel-based VDMs to produce a low-resolution video of strong text-video correlation. After that, we propose a novel expert translation method that employs the latent-based VDMs to further upsample the low-resolution video to high resolution, which can also remove potential artifacts and corruptions from low-resolution videos. Compared to latent VDMs, Show-1 can produce high-quality videos of precise text-video alignment; Compared to pixel VDMs, Show-1 is much more efficient (GPU memory usage during inference is 15G vs 72G). Furthermore, our Show-1 model can be readily adapted for motion customization and video stylization applications through simple temporal attention layer finetuning. Our model achieves state-of-the-art performance on standard video generation benchmarks. 
</details>

<details>
    <summary>Key points</summary>
    * First proposed a hybrid T2V framework cascading pixel-domain VDM with latent-space VDM, where the pixel model handles strong text-to-video alignment while the latent model performs efficient super-resolution
    * First generates a low-resolution, semantically aligned video segment via pixel diffusion, then employs “expert translation” to enable latent-space diffusion for high-resolution reconstruction and detail restoration
    * Maintains pixel-level text-to-video alignment quality while reducing inference memory from ~72GB to 15GB, significantly lowering computational costs
    * Simple fine-tuning of the temporal attention layer enables adaptation to downstream tasks like motion customization and video stylization
    * Achieved state-of-the-art or competitive results on standard text-to-video generation benchmarks like UCF-101 and MSR-VTT, validating the effectiveness of the hybrid diffusion strategy
</details>
</details>

---

<details>
<summary><b>Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets</b></summary>

* **Authors:** Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach
* **arXiv ID:** 2311.15127
* **One-liner:** This work introduces Stable Video Diffusion, which systematically decomposes the process into three stages: “image pre-training → large-scale video pre-training → high-quality video fine-tuning.” Combined with a meticulous subtitle and data filtering workflow, it trains an open-source latent space video diffusion model that achieves state-of-the-art performance across multiple tasks. This enables high-resolution text/image-to-video generation and downstream motion prior transfer.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2311.15127) | [[PDF]](https://arxiv.org/pdf/2311.15127) | [[Code]](https://github.com/Stability-AI/generative-models)

> **核心创新**  
> By integrating the three-stage training workflow—“text-image pretraining → large-scale video pretraining → high-quality video fine-tuning”—with systematic video data screening and caption annotation strategies, we present a reusable, comprehensive paradigm for correctly training latent video diffusion models on massive datasets. This approach demonstrates strong generalizability as a motion prior across multiple tasks, including T2V/I2V and multi-view processing.
<details>
    <summary>Abstract</summary>
    We present Stable Video Diffusion - a latent video diffusion model for high-resolution, state-of-the-art text-to-video and image-to-video generation. Recently, latent diffusion models trained for 2D image synthesis have been turned into generative video models by inserting temporal layers and finetuning them on small, high-quality video datasets. However, training methods in the literature vary widely, and the field has yet to agree on a unified strategy for curating video data. In this paper, we identify and evaluate three different stages for successful training of video LDMs: text-to-image pretraining, video pretraining, and high-quality video finetuning. Furthermore, we demonstrate the necessity of a well-curated pretraining dataset for generating high-quality videos and present a systematic curation process to train a strong base model, including captioning and filtering strategies. We then explore the impact of finetuning our base model on high-quality data and train a text-to-video model that is competitive with closed-source video generation. We also show that our base model provides a powerful motion representation for downstream tasks such as image-to-video generation and adaptability to camera motion-specific LoRA modules. Finally, we demonstrate that our model provides a strong multi-view 3D-prior and can serve as a base to finetune a multi-view diffusion model that jointly generates multiple views of objects in a feedforward fashion, outperforming image-based methods at a fraction of their compute budget.
</details>

<details>
    <summary>Key points</summary>
    * The system proposes a three-stage workflow—“text-image pretraining → large-scale video pretraining → high-quality video fine-tuning”—establishing a universal approach for training latent video diffusion models.
    * A comprehensive video data construction and filtering pipeline (quality filtering, deduplication, caption/description generation, etc.) is designed, demonstrating that pretraining data quality has a sustained impact on final video generation outcomes.
    * On the same latent video diffusion backbone, supports text-to-video and image-to-video generation via distinct conditional encoders, enabling the model to learn highly generalizable “motion and 3D priors”
    * Demonstrated the effectiveness of fine-tuning the base model for diverse tasks including I2V, camera motion LoRA, and multi-view 3D synthesis, proving its utility as a versatile foundational model for video generation and motion prior learning
    * Released model weights and training code, establishing a standard baseline for subsequent research in video editing, motion control, accelerated distillation, and related domains
</details>
</details>

---

<details>
<summary><b>MagicVideo: Efficient Video Generation With Latent Diffusion Models</b></summary>

* **Authors:** Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, Jiashi Feng
* **arXiv ID:** 2211.11018
* **One-liner:** This work introduces MagicVideo, an efficient text-to-video generation framework based on latent diffusion. By leveraging low-dimensional latent space modeling, a 3D U-Net architecture, and reusing T2I weights, it generates 256×256 resolution videos with strong semantic alignment and smooth temporal transitions on a single GPU. Compared to pixel-domain VDM, it achieves approximately 64 times lower computational cost.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2211.11018) | [[PDF]](https://arxiv.org/pdf/2211.11018) | [[Code]](https://magicvideo.github.io/)

> **核心创新**  
> Adapting the latent diffusion model from text-to-image to serve as a video backbone, we introduce a triple-component system in the latent space: a 3D U-Net, inter-frame directional temporal attention, and a lightweight per-frame adapter. This enables the U-Net—originally designed solely for image generation—to learn to produce temporally coherent videos with minimal computational overhead.
<details>
    <summary>Abstract</summary>
    We present an efficient text-to-video generation framework based on latent diffusion models, termed MagicVideo. MagicVideo can generate smooth video clips that are concordant with the given text descriptions. Due to a novel and efficient 3D U-Net design and modeling video distributions in a low-dimensional space, MagicVideo can synthesize video clips with 256x256 spatial resolution on a single GPU card, which takes around 64x fewer computations than the Video Diffusion Models (VDM) in terms of FLOPs. In specific, unlike existing works that directly train video models in the RGB space, we use a pre-trained VAE to map video clips into a low-dimensional latent space and learn the distribution of videos' latent codes via a diffusion model. Besides, we introduce two new designs to adapt the U-Net denoiser trained on image tasks to video data: a frame-wise lightweight adaptor for the image-to-video distribution adjustment and a directed temporal attention module to capture temporal dependencies across frames. Thus, we can exploit the informative weights of convolution operators from a text-to-image model for accelerating video training. To ameliorate the pixel dithering in the generated videos, we also propose a novel VideoVAE auto-encoder for better RGB reconstruction. We conduct extensive experiments and demonstrate that MagicVideo can generate high-quality video clips with either realistic or imaginary content
</details>

<details>
    <summary>Key points</summary>
    * Video diffusion modeling in the VAE latent space significantly reduces computational and GPU memory overhead compared to pixel-domain video diffusion, while maintaining high generation quality
    * Extends pre-trained text-to-image latent diffusion to video with minimal modifications, leveraging existing image generation capabilities through weight initialization and sharing
    * Employing a 3D U-Net to simultaneously model spatial and temporal structures in the latent space, introducing cross-frame temporal information to generate videos with more coherent and natural motion
    * Achieving a favorable balance between training and inference costs by adding a few temporal layers/adaptation modules to the original 2D architecture, rather than training a complete video model from scratch
    * Generates semantically aligned, temporally smooth videos at 256×256 resolution, offering orders-of-magnitude acceleration and resource savings compared to pixel-domain VDMs of comparable quality, laying the foundation for subsequent latent T2V work
</details>
</details>

---

<details>
<summary><b>Make Pixels Dance: High-Dynamic Video Generation</b></summary>

* **Authors:** Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, Hang Li
* **arXiv ID:** 2311.10982
* **One-liner:** This work introduces PixelDance, a high-motion video generation method that simultaneously incorporates “first-frame + last-frame image prompts + text conditions” into diffusion models. Specifically addressing the lack of dynamic content in existing T2V methods, it significantly enhances video synthesis capabilities for complex scenes and substantial motion.
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2311.10982) | [[PDF]](https://arxiv.org/pdf/2311.10982) | [[Code]](https://makepixelsdance.github.io/)

> **核心创新**  
> By simultaneously introducing the triple conditions of “first frame + last frame image prompts + text prompts” in diffusion video generation, coupled with corresponding training/inference mechanisms, the model learns complex scenes and significant motion directly from image prompts. This significantly alleviates the “insufficient motion” issue prevalent in existing T2V models.
<details>
    <summary>Abstract</summary>
    Creating high-dynamic videos such as motion-rich actions and sophisticated visual effects poses a significant challenge in the field of artificial intelligence. Unfortunately, current state-of-the-art video generation methods, primarily focusing on text-to-video generation, tend to produce video clips with minimal motions despite maintaining high fidelity. We argue that relying solely on text instructions is insufficient and suboptimal for video generation. In this paper, we introduce PixelDance, a novel approach based on diffusion models that incorporates image instructions for both the first and last frames in conjunction with text instructions for video generation. Comprehensive experimental results demonstrate that PixelDance trained with public data exhibits significantly better proficiency in synthesizing videos with complex scenes and intricate motions, setting a new standard for video generation.
</details>

<details>
    <summary>Key points</summary>
    * Specifically addresses the limitations of existing T2V models—limited motion and weak effects—by focusing on generating highly dynamic videos featuring substantial movement, complex effects, and camera angle changes
    * Utilizes text prompts + first-frame image prompts + last-frame image prompts simultaneously within the diffusion framework, enabling the model to capture semantic information while directly learning complex scenes and motion states from images
    * Performs latent diffusion modeling of videos within the VAE latent space, balancing generation quality and computational efficiency through latent diffusion
    * Utilizes the first/last frames of ground truth videos as training supervision, and employs inference techniques like segmented generation and frame blending to generate longer videos with coherent storylines
    * Trained solely on public datasets (e.g., WebVid), it significantly outperforms existing methods in synthesizing complex scenes and videos with substantial motion, generating three-minute continuous narrative videos and establishing a new baseline for high-dynamic video generation.
</details>
</details>

---

<details>
<summary><b>Latent Video Diffusion Models for High-Fidelity Long Video Generation</b></summary>

* **Authors:** Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, Qifeng Chen
* **arXiv ID:** 2211.13221
* **One-liner:** This work introduces Latent Video Diffusion Models (LVDM), which employ hierarchical diffusion modeling in a low-dimensional 3D latent space. By integrating hierarchical generation, conditional noise perturbation, and unconditional guidance, LVDM achieves the generation of high-fidelity long videos extending to thousands of frames with limited computational resources.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2211.13221) | [[PDF]](https://arxiv.org/pdf/2211.13221) | [[Code]](https://github.com/YingqingHe/LVDM)

> **核心创新**  
> We migrate video generation to a low-dimensional 3D latent space, where we propose a comprehensive mechanism comprising “layered diffusion + conditional latent variable perturbation + unconditional guidance.” This approach achieves, for the first time under limited computational resources, both high fidelity and stable generation of videos spanning thousands of frames.
<details>
    <summary>Abstract</summary>
    AI-generated content has attracted lots of attention recently, but photo-realistic video synthesis is still challenging. Although many attempts using GANs and autoregressive models have been made in this area, the visual quality and length of generated videos are far from satisfactory. Diffusion models have shown remarkable results recently but require significant computational resources. To address this, we introduce lightweight video diffusion models by leveraging a low-dimensional 3D latent space, significantly outperforming previous pixel-space video diffusion models under a limited computational budget. In addition, we propose hierarchical diffusion in the latent space such that longer videos with more than one thousand frames can be produced. To further overcome the performance degradation issue for long video generation, we propose conditional latent perturbation and unconditional guidance that effectively mitigate the accumulated errors during the extension of video length. Extensive experiments on small domain datasets of different categories suggest that our framework generates more realistic and longer videos than previous strong baselines. We additionally provide an extension to large-scale text-to-video generation to demonstrate the superiority of our work. 
</details>

<details>
    <summary>Key points</summary>
    * Use video VAEs to map high-dimensional videos onto compact 3D latent spaces, enabling training of video diffusion models while drastically reducing computational and GPU memory overhead
    * Design a hierarchical diffusion architecture within the latent space, first generating short segments or low-frequency information before progressively expanding and refining details to support long-video generation spanning thousands of frames
    * Enable the model to continuously append existing segments in the latent space through a unified video diffusion model + autoregressive generation strategy, theoretically generating videos of “arbitrary length”
    * Proposes conditional latent perturbation combined with unconditional guidance to suppress error accumulation during long video expansion, enhancing temporal consistency and stability
    * Generates significantly higher-quality long videos than previous methods across multi-category small-domain datasets, and further extends to large-scale text-to-video tasks, validating the framework's generalization capability
</details>
</details>

---

<details>
<summary><b>TriDet: Temporal Action Detection with Relative Boundary Modeling</b></summary>

* **Authors:** Dingfeng Shi, Yujie Zhong, Qiong Cao, Lin Ma, Jia Li, Dacheng Tao
* **arXiv ID:** 2303.07347
* **One-liner:** This work introduces TriDet, a one-stage framework for temporal action detection. By employing a Trident-head with “relative boundary probability distribution” and an Scalable Granularity-Aware (SGP) feature pyramid to finely model action boundaries, it significantly improves detection accuracy while reducing computational overhead across multiple mainstream datasets.
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2303.07347) | [[PDF]](https://arxiv.org/pdf/2303.07347) | [[Code]](https://github.com/dingfengshi/TriDet)

> **核心创新**  
> Trident-head employs a “relative boundary probability distribution” to finely model action start and end points. Combined with an extensible granularity-aware (SGP) feature pyramid that enhances discriminative power across different temporal scales, this approach significantly improves the accuracy and efficiency of temporal action boundary localization within a single-stage framework.
<details>
    <summary>Abstract</summary>
    In this paper, we present a one-stage framework TriDet for temporal action detection. Existing methods often suffer from imprecise boundary predictions due to the ambiguous action boundaries in videos. To alleviate this problem, we propose a novel Trident-head to model the action boundary via an estimated relative probability distribution around the boundary. In the feature pyramid of TriDet, we propose an efficient Scalable-Granularity Perception (SGP) layer to mitigate the rank loss problem of self-attention that takes place in the video features and aggregate information across different temporal granularities. Benefiting from the Trident-head and the SGP-based feature pyramid, TriDet achieves state-of-the-art performance on three challenging benchmarks: THUMOS14, HACS and EPIC-KITCHEN 100, with lower computational costs, compared to previous methods. For example, TriDet hits an average mAP of 69.3\% on THUMOS14, outperforming the previous best by 2.5\%, but with only 74.6\% of its latency. 
</details>

<details>
    <summary>Key points</summary>
    * TriDet proposes a simple and efficient one-stage temporal action detection framework that simultaneously performs action classification and bounding box regression within an end-to-end architecture
    * By modeling action start and end points as “relative probability distributions near the boundary” via the Trident-head, it significantly mitigates the inaccurate boundary prediction issues caused by blurred boundaries in traditional methods
    * Introduces Scalable-Granularity Perception (SGP) layers in the feature pyramid to mitigate rank-loss issues in self-attention on video features while aggregating information across multiple temporal scales
    * Achieves SOTA or competitive results on datasets including THUMOS14, HACS, and EPIC-KITCHEN 100, while demonstrating significantly lower inference latency than previous methods (e.g., achieving higher mAP on THUMOS14 with only approximately 74.6% of the latency)
    * Demonstrates stable performance across multiple TAD datasets with hierarchical labels, showcasing strong generalization capabilities across diverse scenarios and annotation formats.
</details>
</details>

---

<details>
<summary><b>Temporal Action Localization with Enhanced Instant Discriminability</b></summary>

* **Authors:** Dingfeng Shi, Qiong Cao, Yujie Zhong, Shan An, Jian Cheng, Haogang Zhu, Dacheng Tao
* **arXiv ID:** 2309.05590
* **One-liner:** This work builds upon TriDet to propose a temporal action detection framework called “Enhanced Instantaneous Discrimination,” which combines an extensible granularity-aware layer, pre-trained large model features, and a decoupled feature pyramid. These designs systematically enhance the discrimination capability at each video time point, thereby achieving new state-of-the-art performance across multiple TAD datasets, including those with hierarchical multi-label tasks.
* **Published in:** IJCV 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2309.05590) | [[PDF]](https://arxiv.org/pdf/2309.05590) | [[Code]](https://github.com/dingfengshi/tridetplus)

> **核心创新**  
> By introducing an extensible granularity-aware layer, pre-trained large model features, and a decoupled feature pyramid within TriDet's one-stage framework, the system enhances its discriminative capability at each video frame. This significantly improves temporal action localization accuracy without increasing computational complexity.
<details>
    <summary>Abstract</summary>
    Temporal action detection (TAD) aims to detect all action boundaries and their corresponding categories in an untrimmed video. The unclear boundaries of actions in videos often result in imprecise predictions of action boundaries by existing methods. To resolve this issue, we propose a one-stage framework named TriDet. First, we propose a Trident-head to model the action boundary via an estimated relative probability distribution around the boundary. Then, we analyze the rank-loss problem (i.e. instant discriminability deterioration) in transformer-based methods and propose an efficient scalable-granularity perception (SGP) layer to mitigate this issue. To further push the limit of instant discriminability in the video backbone, we leverage the strong representation capability of pretrained large models and investigate their performance on TAD. Last, considering the adequate spatial-temporal context for classification, we design a decoupled feature pyramid network with separate feature pyramids to incorporate rich spatial context from the large model for localization. Experimental results demonstrate the robustness of TriDet and its state-of-the-art performance on multiple TAD datasets, including hierarchical (multilabel) TAD datasets.
</details>

<details>
    <summary>Key points</summary>
    * Focusing on the issue of “rank-loss causing a drop in instantaneous discriminability” in Transformer-based TAD methods, emphasizing the need to make each time point itself more easily distinguishable between “action/no action”
    * Operates within a single-stage TriDet framework: employs a Trident-head to precisely model action boundaries via “relative boundary probability distributions” while maintaining an end-to-end, highly efficient detection pipeline.
    * Introduces the Scalable-Granularity Perception (SGP) layer, which mitigates rank-loss through multi-granularity feature branching, making instantaneous features more “distinctly separable” across different temporal scales.
    * Leverages strong visual representations from pre-trained large models, integrating their features into the video backbone to significantly enhance discrimination capabilities for local temporal segments.
    * Decouples the feature pyramid to better align the large model's spatial context with localization tasks, achieving SOTA or competitive results across multiple hierarchical multi-label TAD datasets, validating the method's generalization and robustness.
</details>
</details>

---

<details>
<summary><b>BasicTAD: an Astounding RGB-Only Baseline for Temporal Action Detection</b></summary>

* **Authors:** Min Yang, Guo Chen, Yin-Dong Zheng, Tong Lu, Limin Wang
* **arXiv ID:** 2205.02717
* **One-liner:** BasicTAD systematically rebuilt a “pure RGB, end-to-end, modular” temporal action detection baseline (BasicTAD/PlusTAD). Without introducing complex structures or additional modalities like optical flow, it achieved significant improvements or approached SOTA performance across multiple mainstream TAD datasets through meticulous design and combination of data augmentation, backbone, neck, and detection heads. This work redefines the performance ceiling for RGB-only baselines.
* **Published in:** CVIU 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2205.02717) | [[PDF]](https://arxiv.org/pdf/2205.02717) | [[Code]](https://github.com/MCG-NJU/BasicTAD)

> **核心创新**  
> Without inventing new modules, we decompose the TAD pipeline system into four fundamental components: sampling, backbone, neck, and detection head. Based on RGB-only and end-to-end training, we present a minimalist modular design thoroughly validated through empirical testing. This “most basic combination” achieves performance approaching or even surpassing state-of-the-art (SOTA) models, thereby redefining the strong baseline in the TAD domain.
<details>
    <summary>Abstract</summary>
    Temporal action detection (TAD) is extensively studied in the video understanding community by generally following the object detection pipeline in images. However, complex designs are not uncommon in TAD, such as two-stream feature extraction, multi-stage training, complex temporal modeling, and global context fusion. In this paper, we do not aim to introduce any novel technique for TAD. Instead, we study a simple, straightforward, yet must-known baseline given the current status of complex design and low detection efficiency in TAD. In our simple baseline (termed BasicTAD), we decompose the TAD pipeline into several essential components: data sampling, backbone design, neck construction, and detection head. We extensively investigate the existing techniques in each component for this baseline, and more importantly, perform end-to-end training over the entire pipeline thanks to the simplicity of design. As a result, this simple BasicTAD yields an astounding and real-time RGB-Only baseline very close to the state-of-the-art methods with two-stream inputs. In addition, we further improve the BasicTAD by preserving more temporal and spatial information in network representation (termed as PlusTAD). Empirical results demonstrate that our PlusTAD is very efficient and significantly outperforms the previous methods on the datasets of THUMOS14 and FineAction. Meanwhile, we also perform in-depth visualization and error analysis on our proposed method and try to provide more insights on the TAD problem. Our approach can serve as a strong baseline for future TAD research.
</details>

<details>
    <summary>Key points</summary>
    * We built and refined a baseline TAD model, BasicTAD, trained end-to-end using only RGB input. Without introducing optical flow or complex architectures, it approaches or even matches the performance of state-of-the-art methods that rely on two streams.
    * Decomposed the TAD pipeline into four fundamental modules: sampling, backbone, neck, and detector head. Within each module, systematically compared existing designs to propose a set of “simple yet effective” combination practices.
    * Leveraged the advantage of a simple architecture to achieve true end-to-end training from video input to action boundary and category prediction. This approach is more unified and efficient compared to multi-stage methods relying on pre-extracted features.
    * Develops PlusTAD from BasicTAD with minimal modifications focused on “preserving more spatio-temporal information,” significantly outperforming existing methods on datasets like THUMOS14 and FineAction
    * Reveals the impact of different design choices on TAD performance through extensive ablation studies and error analysis, providing reusable strong baselines and design guidelines for future work
</details>
</details>

---

<details>
<summary><b>Action Sensitivity Learning for Temporal Action Localization</b></summary>

* **Authors:** Jiayi Shao, Xiaohan Wang, Ruijie Quan, Junjun Zheng, Jiang Yang, Yi Yang
* **arXiv ID:** 2305.15701
* **One-liner:** This work proposes the Action Sensitivity Learning (ASL) framework for temporal action localization. By learning the “action sensitivity” of each frame at both the category and instance levels, it explicitly distinguishes key frames from redundant frames through reweighted gradient training and a designed action-sensitive contrastive loss. This approach achieves significant improvements in average mAP across multiple TAL datasets.
* **Published in:** ICCV 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2305.15701) | [[PDF]](https://arxiv.org/pdf/2305.15701) 
> **核心创新**  
> The core innovation of ASL lies in explicitly learning action sensitivity that quantifies “how important each frame is.” It employs a lightweight evaluator to simultaneously provide category-level and instance-level sensitivity, using this to reweight training gradients for classification/boundary regression. Combined with action-sensitive contrastive loss, this approach distinguishes key frames from redundant ones, significantly enhancing temporal action localization performance.
<details>
    <summary>Abstract</summary>
   Temporal action localization (TAL), which involves recognizing and locating action instances, is a challenging task in video understanding. Most existing approaches directly predict action classes and regress offsets to boundaries, while overlooking the discrepant importance of each frame. In this paper, we propose an Action Sensitivity Learning framework (ASL) to tackle this task, which aims to assess the value of each frame and then leverage the generated action sensitivity to recalibrate the training procedure. We first introduce a lightweight Action Sensitivity Evaluator to learn the action sensitivity at the class level and instance level, respectively. The outputs of the two branches are combined to reweight the gradient of the two sub-tasks. Moreover, based on the action sensitivity of each frame, we design an Action Sensitive Contrastive Loss to enhance features, where the action-aware frames are sampled as positive pairs to push away the action-irrelevant frames. The extensive studies on various action localization benchmarks (i.e., MultiThumos, Charades, Ego4D-Moment Queries v1.0, Epic-Kitchens 100, Thumos14 and ActivityNet1.3) show that ASL surpasses the state-of-the-art in terms of average-mAP under multiple types of scenarios, e.g., single-labeled, densely-labeled and egocentric.
</details>

<details>
    <summary>Key points</summary>
    * Points out that existing TAL methods generally “treat all frames equally,” ignoring the importance difference between key frames and redundant frames, and explicitly models “action sensitivity” as a target
    * Proposes a lightweight Action Sensitivity Evaluator that outputs both category-level and instance-level sensitivity to measure each frame's contribution to classification and boundary localization
    * Utilizes frame-level sensitivity to reweight gradients for classification and bounding box regression losses, enabling the model to automatically “focus more on key frames and be less influenced by background noise” during training.
    * Designs an Action-Sensitive Contrastive Loss based on frame sensitivity, treating action-related frames as positive samples and action-irrelevant frames as negative samples to further widen feature distances and enhance discriminative power.
    * Achieves superior average mAP performance across multiple (single-label/dense/egocentric) TAL benchmarks including MultiThumos, Charades, Ego4D-MQ, EPIC-Kitchens 100, THUMOS14, and ActivityNet1.3, validating the framework's versatility and effectiveness.
</details>
</details>


<summary><b>TemporalMaxer: Maximize Temporal Context with only Max Pooling for Temporal Action Localization</b></summary>

* **Authors:** Tuan N. Tang, Kwonyoung Kim, Kwanghoon Sohn
* **arXiv ID:** 2303.09055
* **One-liner:** This work introduces TemporalMaxer, a minimalist temporal attention framework that replaces complex self-attention and other long-range modeling mechanisms with “local max-pooling blocks.” We demonstrate that even with only the most fundamental max-pooling operations and significantly reduced parameters and computational complexity, TemporalMaxer can achieve performance on par with or surpass existing state-of-the-art methods across multiple temporal action localization datasets.
* **Published in:** ICCV 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2303.09055) | [[PDF]](https://arxiv.org/pdf/2303.09055) | [[Code]](https://github.com/TuanTNG/TemporalMaxer)
> **核心创新**  
> The core innovation of TemporalMaxer lies in completely abandoning complex long-term self-attention/graph structures, employing only a “local, parameter-free max-pooling block” as the temporal modeling unit within its backbone. This approach maximizes the utilization of pre-trained 3D CNN feature information while minimizing the complexity of long-term modeling, thereby achieving and even surpassing the temporal attention learning (TAL) performance of heavyweight models like Transformers with an extremely minimalist architecture.
<details>
    <summary>Abstract</summary>
    Temporal Action Localization (TAL) is a challenging task in video understanding that aims to identify and localize actions within a video sequence. Recent studies have emphasized the importance of applying long-term temporal context modeling (TCM) blocks to the extracted video clip features such as employing complex self-attention mechanisms. In this paper, we present the simplest method ever to address this task and argue that the extracted video clip features are already informative to achieve outstanding performance without sophisticated architectures. To this end, we introduce TemporalMaxer, which minimizes long-term temporal context modeling while maximizing information from the extracted video clip features with a basic, parameter-free, and local region operating max-pooling block. Picking out only the most critical information for adjacent and local clip embeddings, this block results in a more efficient TAL model. We demonstrate that TemporalMaxer outperforms other state-of-the-art methods that utilize long-term TCM such as self-attention on various TAL datasets while requiring significantly fewer parameters and computational resources. The code for our approach is publicly available at this https URL
</details>

<details>
    <summary>Key points</summary>
    * Completely abandoning complex TCMs like Transformers, self-attention, and graph structures, this approach employs only local, parameter-free max-pooling as the sole temporal modeling unit, forming one of the simplest TAL backbones to date.
    * Utilizes clip features extracted via pre-trained 3D CNNs (e.g., I3D/TSN), asserting these features are inherently robust enough. Max-pooling merely “extracts the most critical local information” without layering additional complex structures.
    * Performs local max-pooling across adjacent time windows to concentrate the most salient responses within local segments, capturing effective temporal context at minimal computational cost.
    * On multiple TAL benchmarks including THUMOS14, TemporalMaxer achieves or surpasses SOTA performance with significantly fewer parameters and FLOPs than self-attention/Transformer methods
    * Subsequent benchmark studies demonstrate that TemporalMaxer remains outstanding in scenarios with limited data or computational constraints, representing a TAL model that excels in both data efficiency and computational efficiency
</details>
</details>

---

<details>
<summary><b>Contextual Object Detection with Multimodal Large Language Models</b></summary>

* **Authors:** Yuhang Zang, Wei Li, Jun Han, Kaiyang Zhou, Chen Change Loy
* **arXiv ID:** 2305.18279
* **One-liner:** This work introduces the novel problem of “contextual object detection” and constructs the unified multimodal model ContextDET. It integrates a visual encoder, a pre-trained large language model, and a detection head into a “semantic generation-first, object detection-second” framework. This enables MLLMs to locate, identify, and describe visible objects using natural language in interactive scenarios such as language cloze tests, image description, and question-answering, even with open-ended vocabularies.
* **Published in:** IJCV 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2305.18279) | [[PDF]](https://arxiv.org/pdf/2305.18279) | [[Code]](https://github.com/yuhangzang/ContextDET)
> **核心创新**  
> The core innovation of ContextDET lies in embedding “object detection” into the language reasoning process of multimodal large models. Using a unified framework of “visual encoder + frozen LLM + detection head,” it transforms the traditionally independent detection task into context-aware language generation results. This enables true “context-based detection” in interactive scenarios such as open vocabulary, question-answering, and description tasks.
<details>
    <summary>Abstract</summary>
    Recent Multimodal Large Language Models (MLLMs) are remarkable in vision-language tasks, such as image captioning and question answering, but lack the essential perception ability, i.e., object detection. In this work, we address this limitation by introducing a novel research problem of contextual object detection -- understanding visible objects within different human-AI interactive contexts. Three representative scenarios are investigated, including the language cloze test, visual captioning, and question answering. Moreover, we present ContextDET, a unified multimodal model that is capable of end-to-end differentiable modeling of visual-language contexts, so as to locate, identify, and associate visual objects with language inputs for human-AI interaction. Our ContextDET involves three key submodels: (i) a visual encoder for extracting visual representations, (ii) a pre-trained LLM for multimodal context decoding, and (iii) a visual decoder for predicting bounding boxes given contextual object words. The new generate-then-detect framework enables us to detect object words within human vocabulary. Extensive experiments show the advantages of ContextDET on our proposed CODE benchmark, open-vocabulary detection, and referring image segmentation. 
</details>

<details>
    <summary>Key points</summary>
    * Formally introduces the novel task of “Contextual Object Detection,” enabling models to identify and localize contextually relevant visible objects within linguistic contexts such as cloze tests, descriptions, and question-answering
    * Designed the unified ContextDET framework, integrating a visual encoder, pre-trained LLM, and visual decoder to end-to-end model interactions between image and linguistic contexts
    * Proposed a “generate-then-detect” workflow: the LLM first generates object words within the context, then the visual decoder predicts corresponding bounding boxes based on these words
    * Leveraging the lexical and reasoning capabilities of LLMs, the model performs object detection and language alignment with an open vocabulary, unrestricted by fixed category sets
    * Constructed the CODE benchmark and conducted experiments on open-vocabulary detection and referring segmentation tasks, demonstrating that ContextDET significantly outperforms existing methods across multiple context-aware visual tasks
</details>
</details>

---

<details>
<summary><b>ActionFormer: Localizing Moments of Actions with Transformers</b></summary>

* **Authors:** Chenlin Zhang, Jianxin Wu, Yin Li
* **arXiv ID:** 2202.07925
* **One-liner:** This work introduces Latent Video Diffusion Models (LVDM), which employ hierarchical diffusion modeling in a low-dimensional 3D latent space. By integrating hierarchical generation, conditional noise perturbation, and unconditional guidance, LVDM achieves the generation of high-fidelity long videos extending to thousands of frames with limited computational resources.
* **Published in:** ECCV 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2202.07925) | [[PDF]](https://arxiv.org/pdf/2202.07925) | [[Code]](https://github.com/happyharrycn/actionformer_release)

> **核心创新**  
> We migrate video generation to a low-dimensional 3D latent space, where we propose a comprehensive mechanism comprising “layered diffusion + conditional latent variable perturbation + unconditional guidance.” This approach achieves, for the first time under limited computational resources, both high fidelity and stable generation of videos spanning thousands of frames.
<details>
    <summary>Abstract</summary>
    Self-attention based Transformer models have demonstrated impressive results for image classification and object detection, and more recently for video understanding. Inspired by this success, we investigate the application of Transformer networks for temporal action localization in videos. To this end, we present ActionFormer -- a simple yet powerful model to identify actions in time and recognize their categories in a single shot, without using action proposals or relying on pre-defined anchor windows. ActionFormer combines a multiscale feature representation with local self-attention, and uses a light-weighted decoder to classify every moment in time and estimate the corresponding action boundaries. We show that this orchestrated design results in major improvements upon prior works. Without bells and whistles, ActionFormer achieves 71.0% mAP at tIoU=0.5 on THUMOS14, outperforming the best prior model by 14.1 absolute percentage points. Further, ActionFormer demonstrates strong results on ActivityNet 1.3 (36.6% average mAP) and EPIC-Kitchens 100 (+13.5% average mAP over prior works). 
</details>

<details>
    <summary>Key points</summary>
    * Use video VAEs to map high-dimensional videos onto compact 3D latent spaces, enabling training of video diffusion models while drastically reducing computational and GPU memory overhead
    * Design a hierarchical diffusion architecture within the latent space, first generating short segments or low-frequency information before progressively expanding and refining details to support long-video generation spanning thousands of frames
    * Enable the model to continuously append existing segments in the latent space through a unified video diffusion model + autoregressive generation strategy, theoretically generating videos of “arbitrary length”
    * Proposes conditional latent perturbation combined with unconditional guidance to suppress error accumulation during long video expansion, enhancing temporal consistency and stability
    * Generates significantly higher-quality long videos than previous methods across multi-category small-domain datasets, and further extends to large-scale text-to-video tasks, validating the framework's generalization capability
</details>
</details>
---
