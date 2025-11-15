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
