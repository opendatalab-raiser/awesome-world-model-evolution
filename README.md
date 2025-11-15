# Awesome World Model Evolution - Forging the World Model Universe from Unified Multimodal Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<div align="center">
  <img src="https://img.shields.io/github/stars/opendatalab-raiser/awesome-world-model-evolution" alt="Stars">
  <img src="https://img.shields.io/github/forks/opendatalab-raiser/awesome-world-model-evolution" alt="Forks">
  <img src="https://img.shields.io/github/license/opendatalab-raiser/awesome-world-model-evolution" alt="License">
  <img src="https://img.shields.io/github/last-commit/opendatalab-raiser/awesome-world-model-evolution" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</div>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

<p align="center">
  <strong>A curated collection of research papers, models, and resources tracing the evolution from specialized models to unified world models.</strong>
</p>

---

## 📋 Table of Contents

- [Awesome World Model Evolution - Forging the World Model Universe from Unified Multimodal Models](#awesome-world-model-evolution---forging-the-world-model-universe-from-unified-multimodal-models-)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Introduction](#-introduction)
    - [What is a World Model?](#what-is-a-world-model)
    - [The Three Fundamental Consistencies](#the-three-fundamental-consistencies)
      - [1️⃣ **Modality Consistency**](#1️⃣-modality-consistency)
      - [2️⃣ **Spatial Consistency**](#2️⃣-spatial-consistency)
      - [3️⃣ **Temporal Consistency**](#3️⃣-temporal-consistency)
    - [Why Unified Multimodal Models?](#why-unified-multimodal-models)
  - [🔬 Independent Exploration: Specialized Models](#-independent-exploration-specialized-models)
    - [Modality Consistency](#modality-consistency)
      - [Representative Works](#representative-works)
    - [Spatial Consistency](#spatial-consistency)
      - [Representative Works](#representative-works-1)
    - [Temporal Consistency](#temporal-consistency)
      - [Representative Works](#representative-works-2)
  - [🔗 Preliminary Integration: Unified Multimodal Models](#-preliminary-integration-unified-multimodal-models)
    - [Modality + Spatial Consistency](#modality--spatial-consistency)
      - [Representative Works](#representative-works-3)
    - [Modality + Temporal Consistency](#modality--temporal-consistency)
      - [Representative Works](#representative-works-4)
    - [Spatial + Temporal Consistency](#spatial--temporal-consistency)
      - [Representative Works](#representative-works-5)
  - [🌟 The "Trinity" Prototype: Emerging World Models](#-the-trinity-prototype-emerging-world-models)
    - [Text-to-World Generators](#text-to-world-generators)
      - [Representative Works](#representative-works-6)
    - [Embodied Intelligence Systems](#embodied-intelligence-systems)
      - [Representative Works](#representative-works-7)
  - [📊 Benchmarks and Evaluation](#-benchmarks-and-evaluation)
      - [Representative Works](#representative-works-8)
  - [📝 Contributing](#-contributing)
    - [How to Contribute](#how-to-contribute)
    - [Contribution Guidelines](#contribution-guidelines)
  - [⭐ Star History](#-star-history)
  - [📄 License](#-license)

---

## 🎯 Introduction

### What is a World Model?

A **World Model** is a computational system that learns the intrinsic laws governing our physical world and constructs an executable, dynamic simulation environment internally. Unlike conventional AI models that merely perform pattern matching or classification, a world model serves as an **internal simulator** capable of:

- 🔄 **Reproducing** observed events and phenomena
- 🎲 **Predicting** future states and outcomes
- 🤔 **Reasoning** about counterfactual scenarios (what-if analysis)
- 🎯 **Planning** long-term strategies based on simulated consequences

**Why World Models Matter:**

World models are considered a cornerstone toward achieving **Artificial General Intelligence (AGI)**:

- 🤖 **Embodied AI**: Robots can "rehearse" actions before execution
- 🚗 **Autonomous Systems**: Simulation of hazardous edge cases for safer deployment
- 🔬 **Scientific Discovery**: Accelerated understanding of complex systems through digital experimentation
- 🎮 **Interactive Environments**: Creation of controllable, physics-aware virtual worlds

### The Three Fundamental Consistencies

A fully functional world model must master three core competencies, which we term the **Three Fundamental Consistencies**:

#### 1️⃣ **Modality Consistency**
The "linguistic interface" between the model and reality.

- **Capability**: Bidirectional translation between high-dimensional sensory inputs (images, video, audio) and abstract symbolic representations (language, structured data)
- **Example Tasks**: Image captioning, text-to-image generation, visual question answering
- **Significance**: Enables the world model to receive instructions and communicate observations in human-interpretable formats

#### 2️⃣ **Spatial Consistency**
The "static 3D comprehension" of the physical world.

- **Capability**: Understanding that objects possess fixed geometric forms, occupy space, exhibit occlusion relationships, and maintain identity across viewpoints
- **Example Tasks**: Novel view synthesis, 3D reconstruction, multi-view consistency
- **Significance**: Forms the foundational "scene graph" enabling accurate spatial reasoning and navigation

#### 3️⃣ **Temporal Consistency**
The "physics engine" for dynamic simulation.

- **Capability**: Modeling how the world evolves over time, including object motion, physical interactions, and causal event chains
- **Example Tasks**: Video prediction, dynamics modeling, future state forecasting
- **Significance**: Enables anticipation of consequences and long-term planning capabilities

### Why Unified Multimodal Models?

This repository traces the evolution toward world models through **Unified Multimodal Models** (particularly Large Multimodal Models - LMMs), which we argue represent the most promising pathway forward:

**Key Advantages:**

- 🏗️ **Architectural Unity**: A central processing core (LLM backbone) with modular sensory interfaces naturally facilitates cross-modal integration
- ✨ **Emergent Understanding**: Pre-training on massive, diverse multimodal datasets enables implicit learning of world regularities rather than relying on manually engineered rules
- 📈 **Scalability**: Demonstrated scaling laws suggest capabilities improve predictably with increased parameters, data, and compute
- 🔄 **Synergistic Learning**: Different consistencies can emerge and reinforce each other through joint training

**Our Thesis:** End-to-end trained unified models are superior to task-specific specialized models for building world models, as they can learn the deep interconnections between modalities, space, and time.

---

## 🔬 Independent Exploration: Specialized Models

Before the era of unified models, researchers pursued a "divide-and-conquer" approach, developing specialized architectures for each consistency challenge. This foundational work established key techniques and insights that inform current unified approaches.

### <a href="./consisency-paper/modality-consistency/README.md">Modality Consistency</a>

**Objective**: Establish bidirectional mappings between symbolic (language) and perceptual (vision) representations.

**Historical Significance**: These models created the first "symbol-perception bridges," solving the fundamental I/O problem for world models.

#### Representative Works


<details>
<summary><b>CLIP: Learning Transferable Visual Models From Natural Language Supervision</b></summary>

* **Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
* **arXiv ID:** 2103.00020
* **One-liner:** Large-scale contrastive learning on 400M image–text pairs enabling strong zero-shot transfer across vision tasks
* **Published in:** ICML 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2103.00020) | [[PDF]](https://arxiv.org/pdf/2103.00020.pdf) | [[Code]](https://github.com/openai/CLIP)

> **Core Innovation**  
> Contrastively trained on 400 million image–text pairs to establish a unified vision–language representation space, enabling zero-shot cross-task transfer and pioneering the large-scale multimodal training paradigm of “using text as a supervisory signal.”

<details>
    <summary>Abstract</summary>
    State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on.
</details>

<details>
    <summary>Key points</summary>
    * Contrastive learning between image and text features  
    * Weakly supervised web-scale dataset (400M pairs)  
    * Zero-shot classification via text prompts  
    * Sparked multimodal foundation model wave  
</details>
</details>

---

<details>
<summary><b>DALL-E: Zero-Shot Text-to-Image Generation</b></summary>

* **Authors:** Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever
* **arXiv ID:** 2102.12092
* **One-liner:** First large transformer trained to generate images from text, demonstrating compositional text-to-image creativity
* **Published in:** ICML 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2102.12092) | [[PDF]](https://arxiv.org/pdf/2102.12092.pdf) | [[Project Page]](https://openai.com/research/dall-e)

> **Core Innovation**  
> First to scale a Transformer for text-to-image generation, paired with a discrete VAE codec, demonstrating strong compositional generation of semantics (shape + style + attributes).

<details>
    <summary>Abstract</summary>
    Text-to-image generation has traditionally focused on finding better modeling assumptions for training on a fixed dataset. These assumptions might involve complex architectures, auxiliary losses, or side information such as object part labels or segmentation masks supplied during training. We describe a simple approach for this task based on a transformer that autoregressively models the text and image tokens as a single stream of data. With sufficient data and scale, our approach is competitive with previous domain-specific models when evaluated in a zero-shot fashion.
</details>

<details>
    <summary>Key points</summary>
    * Text-conditioned transformer for image synthesis  
    * Uses VQ-VAE tokenization for images  
    * Compositional reasoning (shape + style + attribute)  
    * Launch of prompt-driven generative vision  
</details>
</details>

---

<details>
<summary><b>Show, Attend and Tell: Neural Image Caption Generation with Visual Attention</b></summary>

* **Authors:** Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio
* **arXiv ID:** 1502.03044
* **One-liner:** First visual attention mechanism for image captioning; pioneered attention in vision models
* **Published in:** ICML 2015
* **Links:** [[Paper]](https://arxiv.org/abs/1502.03044) | [[PDF]](https://arxiv.org/pdf/1502.03044.pdf)

> **Core Innovation**  
> First to introduce visual attention into image captioning, enabling the model to focus on key regions and produce interpretable attention maps—laying the groundwork for later Vision Transformers.

<details>
    <summary>Abstract</summary>
    Inspired by recent work in machine translation and object detection, we introduce an attention based model that automatically learns to describe the content of images. We describe how we can train this model in a deterministic manner using standard backpropagation techniques and stochastically by maximizing a variational lower bound. We also show through visualization how the model is able to automatically learn to fix its gaze on salient objects while generating the corresponding words in the output sequence. We validate the use of attention with state-of-the-art performance on three benchmark datasets: Flickr8k, Flickr30k and MS COCO.
</details>

<details>
    <summary>Key points</summary>
    * Early attention mechanism for vision  
    * CNN encoder + RNN decoder  
    * Visual heatmaps → interpretability  
    * Influenced attention-based transformers in vision  
</details>
</details>

---

<details>
<summary><b>AttnGAN: Fine-Grained Text-to-Image Generation with Attentional GANs</b></summary>

* **Authors:** Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He
* **arXiv ID:** 1711.10485
* **One-liner:** Word-region attention and semantic alignment loss for high-fidelity text-to-image synthesis
* **Published in:** CVPR 2018
* **Links:** [[Paper]](https://arxiv.org/abs/1711.10485) | [[PDF]](https://arxiv.org/pdf/1711.10485.pdf) | [[Code]](https://github.com/taoxugit/AttnGAN)

> **Core Innovation**  
> Achieves fine-grained text–image correspondence via word-level attention and semantic-alignment losses, markedly improving both the fidelity and semantic consistency of text-to-image generation.

<details>
    <summary>Abstract</summary>
    In this paper, we propose an Attentional Generative Adversarial Network (AttnGAN) that allows attention-driven, multi-stage refinement for fine-grained text-to-image generation. With a novel attentional generative network, the AttnGAN can synthesize fine-grained details at different subregions of the image by paying attentions to the relevant words in the natural language description. In addition, a deep attentional multimodal similarity model is proposed to compute a fine-grained image-text matching loss for training the generator. The proposed AttnGAN significantly outperforms the previous state of the art, boosting the best reported inception score by 14.14% on the CUB dataset and 170.25% on the more challenging COCO dataset. A detailed analysis is also performed by visualizing the attention layers of the AttnGAN. It for the first time shows that the layered attentional GAN is able to automatically select the condition at the word level for generating different parts of the image.
</details>

<details>
    <summary>Key points</summary>
    * Fine-grained word-to-region attention  
    * Multi-stage image refinement  
    * DAMSM loss for text-image consistency  
    * Major milestone before diffusion models  
</details>
</details>

---

### Spatial Consistency

**Objective**: Enable models to understand and generate 3D spatial structure from 2D observations.

**Historical Significance**: Provided methodologies for constructing internal "3D scene graphs" and understanding geometric relationships.

#### Representative Works

<details>
<summary><b>NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis</b></summary>

* **Authors:** Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng  
* **arXiv ID:** 2003.08934  
* **One-liner:** Represent a scene as a continuous volumetric neural radiance field and render novel views via volume rendering of the MLP output  
* **Published in:** ECCV 2020  
* **Links:** [[Paper]](https://arxiv.org/abs/2003.08934) | [[PDF]](https://arxiv.org/pdf/2003.08934.pdf)  

> **Core Innovation**  
> A scene is encoded by an MLP as a Neural Radiance Field (NeRF): given any 5D coordinate (x, y, z, θ, φ) it outputs the volume density plus view-dependent radiance at that location; novel-view images are then synthesized via differentiable volume rendering.

<details>
    <summary>Abstract</summary>
    We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Our algorithm represents a scene using a fully-connected (non-convolutional) deep network, whose input is a single continuous 5D coordinate (spatial location  and viewing direction ) and whose output is the volume density and view-dependent emitted radiance at that spatial location. We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally differentiable, the only input required to optimize our representation is a set of images with known camera poses. We describe how to effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance, and demonstrate results that outperform prior work on neural rendering and view synthesis. View synthesis results are best viewed as videos, so we urge readers to view our supplementary video for convincing comparisons.
</details>

<details>
    <summary>Key points</summary>
    * Uses an MLP to represent a 5D function mapping (x, y, z, θ, φ) → (density, emitted radiance)  
    * Applies differentiable volume rendering for image synthesis  
    * Requires only input images + known camera poses (no explicit surface reconstruction)  
    * Demonstrates high-quality novel view synthesis of complex scenes  
</details>
</details>

---

<details>
<summary><b>3D Gaussian Splatting for Real-Time Radiance Field Rendering</b></summary>

* **Authors:** Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis  
* **arXiv ID:** 2308.04079  
* **One-liner:** Use anisotropic 3D Gaussians + splatting renderer to achieve real-time (≥30fps) high-quality novel-view synthesis of large scenes.  
* **Published in:** SIGGRAPH 2023  
* **Links:** [[Paper]](https://arxiv.org/abs/2308.04079) | [[PDF]](https://arxiv.org/pdf/2308.04079.pdf) | [[Code]](https://github.com/graphdeco-inria/gaussian-splatting)

> **Core Innovation**  
> Proposes representing scenes with thousands of anisotropic 3D Gaussians instead of a pure MLP, coupled with a visibility-aware tile-based splatting rasterizer, enabling real-time (1080p ≥ 30 fps) novel-view synthesis on large-scale scenes.

<details>
    <summary>Abstract</summary>
    Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (>= 30 fps) novel-view synthesis at 1080p resolution. First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space; Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene; Third, we develop a fast visibility-aware rendering algorithm that supports anisotropic splatting and both accelerates training and allows realtime rendering. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets.
</details>

<details>
    <summary>Key points</summary>
    * Represents scene using anisotropic 3D Gaussians rather than only MLP or voxels  
    * Optimizes Gaussian parameters (density, covariance, color) for large scenes  
    * Employs a high-performance splatting renderer enabling real-time novel-view rendering at high resolution  
    * Targets full real-world scenes with real-time performance constraints  
</details>
</details>

---

<details>
<summary><b>EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks</b></summary>

* **Authors:** Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, Gordon Wetzstein  
* **arXiv ID:** 2112.07945  
* **One-liner:** Hybrid explicit-implicit 3D GAN (tri-plane) that synthesizes high-resolution, multi-view consistent imagery and high-quality geometry in real time.  
* **Published in:** CVPR 2022  
* **Links:** [[Paper]](https://arxiv.org/abs/2112.07945) | [[PDF]](https://arxiv.org/pdf/2112.07945.pdf) | [[Code]](https://github.com/NVlabs/eg3d)

> **Core Innovation**  
> Introduces a tri-plane-based hybrid explicit–implicit generator that equips 3D GANs with efficient computation while producing high-resolution, multi-view-consistent images and high-quality geometry.

<details>
    <summary>Abstract</summary>
    Unsupervised generation of high-quality multi-view-consistent images and 3D shapes using only collections of single-view 2D photographs has been a long-standing challenge. Existing 3D GANs are either compute-intensive or make approximations that are not 3D-consistent; the former limits quality and resolution of the generated images and the latter adversely affects multi-view consistency and shape quality. In this work, we improve the computational efficiency and image quality of 3D GANs without overly relying on these approximations. We introduce an expressive hybrid explicit-implicit network architecture that, together with other design choices, synthesizes not only high-resolution multi-view-consistent images in real time but also produces high-quality 3D geometry. By decoupling feature generation and neural rendering, our framework is able to leverage state-of-the-art 2D CNN generators, such as StyleGAN2, and inherit their efficiency and expressiveness. We demonstrate state-of-the-art 3D-aware synthesis with FFHQ and AFHQ Cats, among other experiments.
</details>

<details>
    <summary>Key points</summary>
    * Introduces a tri-plane representation combining explicit 2D feature planes with implicit 3D decoding  
    * Enables use of strong 2D GAN architectures for 3D generation without sacrificing quality  
    * Produces multi-view consistent, high-resolution images plus underlying geometry  
    * Improves efficiency and realism of 3D GANs, bridging geometry and image synthesis  
</details>
</details>

---

<details>
<summary><b>Instant Neural Graphics Primitives with a Multiresolution Hash Encoding</b></summary>

* **Authors:** Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller  
* **arXiv ID:** 2201.05989  
* **One-liner:** Use a multiresolution hash table encoding to massively speed up training and inference of neural graphics primitives (NeRF/SDF etc.).  
* **Published in:** SIGGRAPH 2022 
* **Links:** [[Paper]](https://arxiv.org/abs/2201.05989) | [[PDF]](https://arxiv.org/pdf/2201.05989.pdf) | [[Code]](https://github.com/NVlabs/instant-ngp)

> **Core Innovation**  
> Proposes multiresolution hash encoding that maps coordinates to trainable feature vectors and couples them with a lightweight MLP, slashing FLOPs and memory traffic while preserving quality—delivering second-level training and millisecond rendering of neural graphics primitives.

<details>
    <summary>Abstract</summary>
    Neural graphics primitives, parameterized by fully connected neural networks, can be costly to train and evaluate. We reduce this cost with a versatile new input encoding that permits the use of a smaller network without sacrificing quality, thus significantly reducing the number of floating point and memory access operations: a small neural network is augmented by a multiresolution hash table of trainable feature vectors whose values are optimized through stochastic gradient descent. The multiresolution structure allows the network to disambiguate hash collisions, making for a simple architecture that is trivial to parallelize on modern GPUs. We leverage this parallelism by implementing the whole system using fully-fused CUDA kernels with a focus on minimizing wasted bandwidth and compute operations. We achieve a combined speed-up of several orders of magnitude, enabling training of high-quality neural graphics primitives in a matter of seconds, and rendering in tens of milliseconds at a resolution of 1920 × 1080.  
</details>

<details>
    <summary>Key points</summary>
    * Employs a multiresolution hash table of trainable feature vectors as input encoding  
    * Combines with a small neural network to drastically reduce computation and memory access  
    * Fully fused GPU implementation (CUDA kernels) for maximum parallelism and efficiency  
    * Enables neural graphics primitives (NeRF/SDF/image) to train in seconds and render in tens of milliseconds  
</details>
</details>

---

### Temporal Consistency

**Objective**: Model temporal dynamics, object motion, and causal relationships in video sequences.

**Historical Significance**: Early explorations of the world's "physics engine," capturing regularities in how scenes evolve over time.

#### Representative Works

<details>
<summary><b>PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning</b></summary>

* **Authors:** Yunbo Wang, Haixu Wu, Jianjin Zhang, Zhifeng Gao, Jianmin Wang, Philip S. Yu, Mingsheng Long
* **arXiv ID:** 2103.09504
* **One-liner:** A novel recurrent architecture that decouples memory cells to capture spatio-temporal dynamics in predictive learning of video frames.  
* **Published in:** TPAMI 2023 
* **Links:** [[Paper]](https://arxiv.org/abs/2103.09504) | [[PDF]](https://arxiv.org/pdf/2103.09504.pdf) | [[Code]](https://github.com/thuml/predrnn-pytorch)

> **Core Innovation**  
> Introduces decoupled memory cells embedded in an RNN backbone to separately capture spatial appearance and temporal dynamics, then fuses them into a unified complex-scene representation. A “zig-zag” memory flow propagates information across layers, while a curriculum-learning strategy enables modeling of long-term temporal dependencies.

<details>
    <summary>Abstract</summary>
    The predictive learning of spatiotemporal sequences aims to generate future images by learning from the historical context, where the visual dynamics are believed to have modular structures that can be learned with compositional subsystems. This paper models these structures by presenting PredRNN, a new recurrent network, in which a pair of memory cells are explicitly decoupled, operate in nearly independent transition manners, and finally form unified representations of the complex environment. Concretely, besides the original memory cell of LSTM, this network is featured by a zigzag memory flow that propagates in both bottom-up and top-down directions across all layers, enabling the learned visual dynamics at different levels of RNNs to communicate. It also leverages a memory decoupling loss to keep the memory cells from learning redundant features. We further propose a new curriculum learning strategy to force PredRNN to learn long-term dynamics from context frames, which can be generalized to most sequence-to-sequence models. We provide detailed ablation studies to verify the effectiveness of each component. Our approach is shown to obtain highly competitive results on five datasets for both action-free and action-conditioned predictive learning scenarios.
</details>

<details>
    <summary>Key points</summary>
    * Introduces a memory‐pair structure: two memory cells decoupled in transition to separately capture spatial appearances and temporal dynamics. 
    * Unified representation by merging the outputs of the two memory streams.  
    * Applied to spatiotemporal predictive learning without explicit geometry or flow guidance.  
    * Demonstrates improved future-frame prediction performance on benchmark datasets.  
</details>
</details>

---

<details>
<summary><b>SimVP: Simpler yet Better Video Prediction</b></summary>

* **Authors:** Zhangyang Gao, Cheng Tan, Lirong Wu, Stan Z. Li
* **arXiv ID:** 2206.05099
* **One-liner:** A purely convolutional video-prediction model trained end-to-end with MSE loss, showing that simplicity can outperform more complex architectures.  
* **Published in:** CVPR 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2206.05099) | [[PDF]](https://arxiv.org/pdf/2206.05099.pdf) | [[Code]](https://github.com/A4Bio/SimVP)

> **Core Innovation**  
> Proposes an ultra-simple end-to-end video-prediction framework that uses only a CNN encoder–decoder trained with plain MSE loss; without RNNs, Transformers, or any sophisticated modules, it matches or surpasses far more complex models on multiple benchmarks.

<details>
    <summary>Abstract</summary>
    From CNN, RNN, to ViT, we have witnessed remarkable advancements in video prediction, incorporating auxiliary inputs, elaborate neural architectures, and sophisticated training strategies. We admire these progresses but are confused about the necessity: is there a simple method that can perform comparably well? This paper proposes SimVP, a simple video prediction model that is completely built upon CNN and trained by MSE loss in an end-to-end fashion. Without introducing any additional tricks and complicated strategies, we can achieve state-of-the-art performance on five benchmark datasets. Through extended experiments, we demonstrate that SimVP has strong generalization and extensibility on real-world datasets. The significant reduction of training cost makes it easier to scale to complex scenarios. We believe SimVP can serve as a solid baseline to stimulate the further development of video prediction. 
</details>

<details>
    <summary>Key points</summary>
    * Model based only on convolutional encoder-decoder structure (no RNN, no transformer).
    * Trained with straightforward Mean Squared Error (MSE) loss end-to-end.  
    * Demonstrates that simpler architectures can achieve competitive or superior results in video prediction benchmarks.  
    * Reduces architectural complexity and computational overhead compared to more sophisticated models.  
</details>
</details>

---

<details>
<summary><b>Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning</b></summary>

* **Authors:** Cheng Tan, Zhangyang Gao, Lirong Wu, Yongjie Xu, Jun Xia, Siyuan Li, Stan Z. Li  
* **arXiv ID:** 2206.12126 
* **One-liner:** Introduces a parallelizable temporal‐attention module (TAU) that splits intra-frame and inter-frame attention to improve efficiency in spatiotemporal predictive learning.  
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2206.12126) | [[PDF]](https://arxiv.org/pdf/2206.12126.pdf)

> **Core Innovation**  
> Introduces the Temporal Attention Unit (TAU) that disentles “intra-frame static attention” from “inter-frame dynamic attention” within spatiotemporal prediction modules, enabling fully parallel computation, and further strengthens inter-frame change detection via differential-divergence regularization.

<details>
    <summary>Abstract</summary>
    Spatiotemporal predictive learning aims to generate future frames by learning from historical frames. In this paper, we investigate existing methods and present a general framework of spatiotemporal predictive learning, in which the spatial encoder and decoder capture intra-frame features and the middle temporal module catches inter-frame correlations. While the mainstream methods employ recurrent units to capture long-term temporal dependencies, they suffer from low computational efficiency due to their unparallelizable architectures. To parallelize the temporal module, we propose the Temporal Attention Unit (TAU), which decomposes the temporal attention into intra-frame statical attention and inter-frame dynamical attention. Moreover, while the mean squared error loss focuses on intra-frame errors, we introduce a novel differential divergence regularization to take inter-frame variations into account. Extensive experiments demonstrate that the proposed method enables the derived model to achieve competitive performance on various spatiotemporal prediction benchmarks.
</details>

<details>
    <summary>Key points</summary>
    * Proposes “Temporal Attention Unit (TAU)” that separates intra-frame static attention from inter-frame dynamic attention for efficiency and parallelism.
    * Introduces differential divergence regularization to better account for temporal variations (inter-frame changes).  
    * Framework decouples spatial encoding/decoding and temporal module, enabling better scalability.  
    * Shows improved computational efficiency while maintaining predictive accuracy on benchmark datasets.  
</details>
</details>

---

<details>
<summary><b>VideoGPT: Video Generation using VQ-VAE and Transformers</b></summary>

* **Authors:** Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas  
* **arXiv ID:** 2104.10157
* **One-liner:** A minimal architecture combining VQ-VAE and GPT-style transformer to scale generative modeling from images to natural videos.  
* **Published in:** arXiv 2021  
* **Links:** [[Paper]](https://arxiv.org/abs/2104.10157) | [[PDF]](https://arxiv.org/pdf/2104.10157.pdf) | [[Code]](https://github.com/wilson1yan/VideoGPT)

> **Core Innovation**  
> Applies VQ-VAE to learn discrete video latents (3D conv + axial self-attention) and then performs autoregressive modeling with a GPT-style Transformer over these tokens, enabling likelihood-based generation on natural videos.

<details>
    <summary>Abstract</summary>
    We present VideoGPT: a conceptually simple architecture for scaling likelihood based generative modeling to natural videos. VideoGPT uses VQ-VAE that learns downsampled discrete latent representations of a raw video by employing 3D convolutions and axial self-attention. A simple GPT-like architecture is then used to autoregressively model the discrete latents using spatio-temporal position encodings. Despite the simplicity in formulation and ease of training, our architecture is able to generate samples competitive with state-of-the-art GAN models for video generation on the BAIR Robot dataset, and generate high fidelity natural videos from UCF-101 and Tumbler GIF Dataset (TGIF). We hope our proposed architecture serves as a reproducible reference for a minimalistic implementation of transformer based video generation models. 
</details>

<details>
    <summary>Key points</summary>
    * Uses VQ-VAE to learn discrete latent representations of video using 3D convolutions + axial self-attention.  
    * Uses a GPT-style autoregressive transformer on those discrete latents with spatio-temporal positional encoding.  
    * Achieves competitive quality compared to GAN-based video generation models, but with simpler likelihood-based framework.  
    * Provides a reproducible baseline for transformer-based video generation on natural video datasets.  
</details>
</details>

---

<details>
<summary><b>Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions</b></summary>

* **Authors:** Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, Dumitru Erhan  
* **arXiv ID:** 2210.02399
* **One-liner:** Generates variable-length videos from a sequence of open-domain text prompts by tokenizing video representation and using a masked‐transformer conditioned on text.  
* **Published in:** ICLR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2210.02399) | [[PDF]](https://arxiv.org/pdf/2210.02399.pdf)

> **Core Innovation**  
> Proposes a video-tokenizer that discretizes variable-length videos into tokens and a conditional masked Transformer that generates these tokens before decoding them into frames. By jointly training on large-scale image-text data plus a small number of video-text pairs, it enables open-domain, variable-duration video generation from text prompts.

<details>
    <summary>Abstract</summary>
    We present Phenaki, a model capable of realistic video synthesis, given a sequence of textual prompts. Generating videos from text is particularly challenging due to the computational cost, limited quantities of high quality text-video data and variable length of videos. To address these issues, we introduce a new model for learning video representation which compresses the video to a small representation of discrete tokens. This tokenizer uses causal attention in time, which allows it to work with variable-length videos. To generate video tokens from text we are using a bidirectional masked transformer conditioned on pre-computed text tokens. The generated video tokens are subsequently de-tokenized to create the actual video. To address data issues, we demonstrate how joint training on a large corpus of image-text pairs as well as a smaller number of video-text examples can result in generalization beyond what is available in the video datasets. Compared to the previous video generation methods, Phenaki can generate arbitrary long videos conditioned on a sequence of prompts (i.e. time variable text or a story) in open domain. To the best of our knowledge, this is the first time a paper studies generating videos from time variable prompts. In addition, compared to the per-frame baselines, the proposed video encoder-decoder computes fewer tokens per video but results in better spatio-temporal consistency.
</details>

<details>
    <summary>Key points</summary>
    * Introduces a discrete tokenization of video representation, enabling variable-length output.  
    * Uses a bidirectional masked transformer conditioned on text tokens to generate video tokens, then de-tokenizes to actual frames.  
    * Leverages joint training on large image-text corpus and smaller video-text datasets to improve generalization.  
    * Capable of generating long videos from sequences of prompts, rather than single static text.  
</details>
</details>

---

## 🔗 Preliminary Integration: Unified Multimodal Models

Current state-of-the-art models are beginning to break down the barriers between individual consistencies. This section showcases models that successfully integrate **two** of the three fundamental consistencies, representing crucial intermediate steps toward complete world models.

### Modality + Spatial Consistency

**Capability Profile**: Models that can translate text/image descriptions into spatially coherent 3D representations or multi-view consistent outputs.

**Significance**: These models demonstrate "3D imagination" - they are no longer mere "2D painters" but "digital sculptors" understanding spatial structure.

#### Representative Works

<details>
<summary><b>Zero-1-to-3: Zero-shot One Image to 3D Object</b></summary>

* **Authors:** Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, Carl Vondrick  
* **arXiv ID:** 2303.11328  
* **One-liner:** A framework that takes a single RGB image and enables novel-view synthesis and implicit 3D reconstruction via viewpoint-conditioned diffusion.  
* **Published in:** ICCV 2023  
* **Links:** [[Paper]](https://arxiv.org/abs/2303.11328) | [[PDF]](https://arxiv.org/pdf/2303.11328.pdf) | [[Code]](https://github.com/cvlab-columbia/zero123)

> **Core Innovation**  
> Leverages the implicit geometric priors of large-scale pretrained diffusion models to learn camera-viewpoint changes from a single image, enabling synthesis of novel views of the same object under specified transformations and further supporting single-image implicit 3D reconstruction.

<details>
    <summary>Abstract</summary>
    We introduce Zero-1-to-3, a framework for changing the camera viewpoint of an object given just a single RGB image. To perform novel view synthesis in this under-constrained setting, we capitalize on the geometric priors that large-scale diffusion models learn about natural images. Our conditional diffusion model uses a synthetic dataset to learn controls of the relative camera viewpoint, which allow new images to be generated of the same object under a specified camera transformation. Even though it is trained on a synthetic dataset, our model retains a strong zero-shot generalization ability to out-of-distribution datasets as well as in-the-wild images, including impressionist paintings. Our viewpoint-conditioned diffusion approach can further be used for the task of 3D reconstruction from a single image. Qualitative and quantitative experiments show that our method significantly outperforms state-of-the-art single-view 3D reconstruction and novel view synthesis models by leveraging Internet-scale pre-training.
</details>

<details>
    <summary>Key points</summary>
    * Leverages large-scale pre-trained 2D diffusion models to extract implicit 3D/geometric priors.  
    * Conditions generation on a single RGB image + relative camera transformation (rotation + translation).  
    * Achieves zero-shot generalization to out-of-distribution real-world and artistic images.  
    * Enables both novel-view synthesis and implicit 3D reconstruction from only one image.  
</details>
</details>

---

<details>
<summary><b>MVDream: Multi-view Diffusion for 3D Generation</b></summary>

* **Authors:** Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, Xiao Yang  
* **arXiv ID:** 2308.16512  
* **One-liner:** A multi-view diffusion model that generates geometrically consistent multi-view images from text prompts, enabling 3D generation via 2D+3D synergy.  
* **Published in:** ICLR 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2308.16512) | [[PDF]](https://arxiv.org/pdf/2308.16512.pdf) | [[Code]](https://github.com/bytedance/MVDream)

> **Core Innovation**  
> Proposes a diffusion model that produces geometrically-consistent multi-view images by first pretraining on 2D images and then fine-tuning on 3D data, inheriting the generalization power of 2D diffusion while guaranteeing 3D-render consistency—serving as a universal prior for 3D content generation.

<details>
    <summary>Abstract</summary>
    We introduce MVDream, a diffusion model that is able to generate consistent multi-view images from a given text prompt. Learning from both 2D and 3D data, a multi-view diffusion model can achieve the generalizability of 2D diffusion models and the consistency of 3D renderings. We demonstrate that such a multi-view diffusion model is implicitly a generalizable 3D prior agnostic to 3D representations. It can be applied to 3D generation via Score Distillation Sampling, significantly enhancing the consistency and stability of existing 2D-lifting methods. It can also learn new concepts from a few 2D examples, akin to DreamBooth, but for 3D generation.
</details>

<details>
    <summary>Key points</summary>
    * Combines web-scale 2D diffusion pre-training with fine-tuning on multi-view 3D asset rendered dataset.  
    * Generates sets of consistent multi-view images conditioned on text prompts, facilitating 3D reconstruction/generation tasks.  
    * Serves as a 3D-agnostic prior: the model does not commit to one explicit 3D representation but still ensures multi-view consistency.  
    * Demonstrates improved geometry fidelity using Score Distillation Sampling driven by multi-view images.  
</details>
</details>

---

<details>
<summary><b>Wonder3D: Single Image to 3D using Cross-Domain Diffusion</b></summary>

* **Authors:** Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, Wenping Wang  
* **arXiv ID:** 2310.15008  
* **One-liner:** Efficiently reconstruct high-fidelity textured meshes from a single view image via a cross-domain diffusion generating multi-view normal maps + color images.  
* **Published in:** CVPR 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2310.15008) | [[PDF]](https://arxiv.org/pdf/2310.15008.pdf) | [[Code]](https://github.com/xxlong0/Wonder3D)

> **Core Innovation**  
> Introduces a single-image-to-high-fidelity-textured-mesh pipeline: a cross-domain diffusion model first synthesizes multi-view normals and RGB images; multi-view cross-domain attention enforces inter-view/inter-modal consistency; finally, a geometry-aware normal-fusion algorithm fuses the 2D representations into a high-quality textured mesh.

<details>
    <summary>Abstract</summary>
    In this work, we introduce Wonder3D, a novel method for efficiently generating high-fidelity textured meshes from single-view images. Recent methods based on Score Distillation Sampling (SDS) have shown the potential to recover 3D geometry from 2D diffusion priors, but they typically suffer from time-consuming per-shape optimization and inconsistent geometry. In contrast, certain works directly produce 3D information via fast network inferences, but their results are often of low quality and lack geometric details. To holistically improve the quality, consistency, and efficiency of image-to-3D tasks, we propose a cross-domain diffusion model that generates multi-view normal maps and the corresponding color images. To ensure consistency, we employ a multi-view cross-domain attention mechanism that facilitates information exchange across views and modalities. Lastly, we introduce a geometry-aware normal fusion algorithm that extracts high-quality surfaces from the multi-view 2D representations. Our extensive evaluations demonstrate that our method achieves high-quality reconstruction results, robust generalization, and reasonably good efficiency compared to prior works.  
</details>

<details>
    <summary>Key points</summary>
    * Uses a cross-domain diffusion model to generate multi-view normal maps + color images from a single input view.  
    * Multi-view cross-domain attention mechanism ensures consistency across generated views and domains (normals vs colors).  
    * Geometry-aware normal fusion algorithm transforms the multi-view 2D outputs into a high-quality textured mesh efficiently.  
    * Achieves high-fidelity textured mesh reconstruction with strong generalization and relatively fast inference (minutes) from a single image.  
</details>
</details>

---

<details>
<summary><b>SyncDreamer: Generating Multiview-consistent Images from a Single-view Image</b></summary>

* **Authors:** Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, Wenping Wang  
* **arXiv ID:** 2309.03453  
* **One-liner:** A synchronized multiview diffusion model that, from one input view, produces multiple consistent views via a joint reverse-diffusion process.  
* **Published in:** ICLR 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2309.03453) | [[PDF]](https://arxiv.org/pdf/2309.03453.pdf) | [[Code]](https://github.com/liuyuan-pal/SyncDreamer)

> **Core Innovation**  
> Introduces SyncDreamer, a synchronized multi-view generation framework that jointly produces multiple views during the reverse diffusion process. By leveraging 3D-aware feature attention to synchronize intermediate states across all viewpoints, it enforces both geometric and color consistency, enabling the creation of a consistent multi-view image set from a single input—providing strong support for single-image-to-3D and text-to-3D tasks.

<details>
    <summary>Abstract</summary>
    In this paper, we present a novel diffusion model called SyncDreamer that generates multiview-consistent images from a single-view image. Using pretrained large-scale 2D diffusion models, recent work Zero123 demonstrates the ability to generate plausible novel views from a single-view image of an object. However, maintaining consistency in geometry and colors for the generated images remains a challenge. To address this issue, we propose a synchronized multiview diffusion model that models the joint probability distribution of multiview images, enabling the generation of multiview-consistent images in a single reverse process. SyncDreamer synchronizes the intermediate states of all the generated images at every step of the reverse process through a 3D-aware feature attention mechanism that correlates the corresponding features across different views. Experiments show that SyncDreamer generates images with high consistency across different views, thus making it well-suited for various 3D generation tasks such as novel-view-synthesis, text-to-3D, and image-to-3D.  
</details>

<details>
    <summary>Key points</summary>
    * Models the joint distribution of multiple views in a single diffusion reverse process, rather than generating each view independently.  
    * Introduces a 3D-aware feature attention mechanism that synchronizes intermediate states across views to maintain geometry and color consistency.  
    * From a single view image, produces multiple coherent novel view images suitable for downstream 3D tasks.  
    * Demonstrates strong multiview consistency, facilitating improved image-to-3D or text-to-3D workflows.  
</details>
</details>

---

<details>
<summary><b>DreamFusion: Text-to-3D using 2D Diffusion</b></summary>

* **Authors:** Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall
* **arXiv ID:** 2209.14988
* **One-liner:** DreamFusion pioneers text-to-3D by distilling a pre-trained 2D diffusion model into 3D NeRFs via Score Distillation Sampling, requiring zero 3D training data.
* **Published in:** ICRL 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2209.14988) | [[PDF]](https://arxiv.org/pdf/2209.14988.pdf) | [[Project Page]](https://dreamfusion3d.github.io/)

> **Core Innovation**  
> Introduces Score Distillation Sampling (SDS)—a loss that leverages 2D diffusion score functions as priors to optimize a randomly-initialized NeRF in parameter space, enabling text-to-3D without any 3D supervision.

<details>
    <summary>Abstract</summary>
    Recent breakthroughs in text-to-image synthesis have been driven by diffusion models trained on billions of image-text pairs. Adapting this approach to 3D synthesis would require large-scale datasets of labeled 3D data and efficient architectures for denoising 3D data, neither of which currently exist. In this work, we circumvent these limitations by using a pretrained 2D text-to-image diffusion model to perform text-to-3D synthesis. We introduce a loss based on probability density distillation that enables the use of a 2D diffusion model as a prior for optimization of a parametric image generator. Using this loss in a DeepDream-like procedure, we optimize a randomly-initialized 3D model (a Neural Radiance Field, or NeRF) via gradient descent such that its 2D renderings from random angles achieve a low loss. The resulting 3D model of the given text can be viewed from any angle, relit by arbitrary illumination, or composited into any 3D environment. Our approach requires no 3D training data and no modifications to the image diffusion model, demonstrating the effectiveness of pretrained image diffusion models as priors.
</details>

<details>
    <summary>Key points</summary>
    * Zero 3D annotations—only a large-scale 2D text-to-image diffusion model is required.
    * SDS replaces reconstruction loss; updates NeRF by matching injected noise to diffusion-predicted scores.
    * Integrates mip-NeRF 360 with differentiable shading for coherent geometry and appearance.
    * View-dependent text conditioning plus geometric regularizers mitigate “cardboard” local minima. 
</details>
</details>

---

<details>
<summary><b>ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding</b></summary>

* **Authors:** Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, Silvio Savarese
* **arXiv ID:** 2305.08275
* **One-liner:** ULIP-2 introduces a scalable, annotation-free multimodal pre-training framework that aligns 3D point clouds with images and automatically generated text, enabling superior 3D understanding.
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2305.08275) | [[PDF]](https://arxiv.org/pdf/2305.08275.pdf) | [[Code]](https://github.com/salesforce/ULIP)

> **Core Innovation**  
> Leverages large multimodal models (BLIP-2) to automatically generate holistic language descriptions from rendered 2D views of 3D objects, enabling scalable tri-modal alignment without human annotations.

<details>
    <summary>Abstract</summary>
    Recent advancements in multimodal pre-training have shown promising efficacy in 3D representation learning by aligning multimodal features across 3D shapes, their 2D counterparts, and language descriptions. However, the methods used by existing frameworks to curate such multimodal data, in particular language descriptions for 3D shapes, are not scalable, and the collected language descriptions are not diverse. To address this, we introduce ULIP-2, a simple yet effective tri-modal pre-training framework that leverages large multimodal models to automatically generate holistic language descriptions for 3D shapes. It only needs 3D data as input, eliminating the need for any manual 3D annotations, and is therefore scalable to large datasets. ULIP-2 is also equipped with scaled-up backbones for better multimodal representation learning. We conduct experiments on two large-scale 3D datasets, Objaverse and ShapeNet, and augment them with tri-modal datasets of 3D point clouds, images, and language for training ULIP-2. Experiments show that ULIP-2 demonstrates substantial benefits in three downstream tasks: zero-shot 3D classification, standard 3D classification with fine-tuning, and 3D captioning (3D-to-language generation). It achieves a new SOTA of 50.6% (top-1) on Objaverse-LVIS and 84.7% (top-1) on ModelNet40 in zero-shot classification. In the ScanObjectNN benchmark for standard fine-tuning, ULIP-2 reaches an overall accuracy of 91.5% with a compact model of only 1.4 million parameters. ULIP-2 sheds light on a new paradigm for scalable multimodal 3D representation learning without human annotations and shows significant improvements over existing baselines.
</details>

<details>
    <summary>Key points</summary>
    * Annotation-free: only 3D data is required, making the framework highly scalable.
    * Uses BLIP-2 to generate detailed, multi-view language descriptions, enhancing text modality quality and diversity.
    * Introduces two large-scale tri-modal datasets: ULIP-Objaverse and ULIP-ShapeNet.
    * Achieves SOTA performance on zero-shot 3D classification, standard classification, and 3D-to-language captioning.
</details>
</details>

---

<details>
<summary><b>OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding</b></summary>

* **Authors:** Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, Hao Su
* **arXiv ID:** 2305.10764
* **One-liner:** OpenShape achieves powerful open-world zero-shot 3D shape recognition by scaling up multimodal contrastive learning aligned with CLIP across large-scale 3D datasets.
* **Published in:** NeurIPS 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2305.10764) | [[PDF]](https://arxiv.org/pdf/2305.10764.pdf) | [[Code]](https://github.com/Colin97/OpenShape_code)

> **Core Innovation**  
> Introduces a scalable tri-modal contrastive learning framework with automated text filtering/enrichment, 3D backbone scaling, and hard negative mining to greatly improve open-world generalization of 3D representations.

<details>
    <summary>Abstract</summary>
    We introduce OpenShape, a method for learning multi-modal joint representations of text, image, and point clouds. We adopt the commonly used multi-modal contrastive learning framework for representation alignment, but with a specific focus on scaling up 3D representations to enable open-world 3D shape understanding. To achieve this, we scale up training data by ensembling multiple 3D datasets and propose several strategies to automatically filter and enrich noisy text descriptions. We also explore and compare strategies for scaling 3D backbone networks and introduce a novel hard negative mining module for more efficient training. We evaluate OpenShape on zero-shot 3D classification benchmarks and demonstrate its superior capabilities for open-world recognition. Specifically, OpenShape achieves a zero-shot accuracy of 46.8% on the 1,156-category Objaverse-LVIS benchmark, compared to less than 10% for existing methods. OpenShape also achieves an accuracy of 85.3% on ModelNet40, outperforming previous zero-shot baseline methods by 20% and performing on par with some fully-supervised methods. Furthermore, we show that our learned embeddings encode a wide range of visual and semantic concepts (e.g., subcategories, color, shape, style) and facilitate fine-grained text-3D and image-3D interactions. Due to their alignment with CLIP embeddings, our learned shape representations can also be integrated with off-the-shelf CLIP-based models for various applications, such as point cloud captioning and point cloud-conditioned image generation.
</details>

<details>
    <summary>Key points</summary>
    * Ensembles 876k 3D shapes across diverse categories to address data scarcity.
    * Proposes three text enrichment strategies—filtering, captioning, and image retrieval—to improve text quality and semantics.
    * Evaluates and scales multiple 3D backbones (e.g., PointBERT, SparseConv) for large-scale training.
    * Introduces hard negative mining to address class imbalance and enhance model discrimination.
    * Learned 3D representations are CLIP-aligned and readily usable for cross-modal tasks like image generation and captioning.
</details>
</details>

---

<details>
<summary><b>DreamLLM: Synergistic Multimodal Comprehension and Creation</b></summary>

* **Authors:** Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, Li Yi
* **arXiv ID:** 2309.11499
* **One-liner:** DREAMLLM is the first multimodal large language model that unifies comprehension and creation by end-to-end generating raw interleaved text and images, enabling true synergy between understanding and generation.
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2309.11499) | [[PDF]](https://arxiv.org/pdf/2309.11499.pdf) | [[Code]](https://github.com/RunpeiDong/DreamLLM)

> **Core Innovation**  
> Introduces a novel Interleaved Generative Pre-training (I-GPT) framework that uses dream queries and score distillation to directly model image posteriors without relying on intermediate representations like CLIP, enabling native generation and understanding of interleaved multimodal content.

<details>
    <summary>Abstract</summary>
    This paper presents DreamLLM, a learning framework that first achieves versatile Multimodal Large Language Models (MLLMs) empowered with frequently overlooked synergy between multimodal comprehension and creation. DreamLLM operates on two fundamental principles. The first focuses on the generative modeling of both language and image posteriors by direct sampling in the raw multimodal space. This approach circumvents the limitations and information loss inherent to external feature extractors like CLIP, and a more thorough multimodal understanding is obtained. Second, DreamLLM fosters the generation of raw, interleaved documents, modeling both text and image contents, along with unstructured layouts. This allows DreamLLM to learn all conditional, marginal, and joint multimodal distributions effectively. As a result, DreamLLM is the first MLLM capable of generating free-form interleaved content. Comprehensive experiments highlight DreamLLM's superior performance as a zero-shot multimodal generalist, reaping from the enhanced learning synergy.
</details>

<details>
    <summary>Key points</summary>
    * First MLLM capable of free-form interleaved text-image generation with real synergy between creation and comprehension.
    * Introduces dream queries and dream label
    * Uses score distillation for direct pixel-space image sampling, avoiding CLIP-induced semantic loss and modality gaps.
    * Supports in-context multimodal generation such as image editing, subject-driven generation, and compositional creation.
</details>
</details>

---

<details>
<summary><b>EditWorld: Simulating World Dynamics for Instruction-Following Image Editing</b></summary>

* **Authors:** Ling Yang, Bohan Zeng, Jiaming Liu, Hong Li, Minghao Xu, Wentao Zhang, Shuicheng Yan
* **arXiv ID:** 2405.14785
* **One-liner:** EDITWORLD introduces the first world-instructed image editing task, enabling more intelligent and physically plausible image editing by simulating real and virtual world dynamics.
* **Published in:** ACM Multimedia 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2405.14785) | [[PDF]](https://arxiv.org/pdf/2405.14785.pdf) | [[Code]](https://github.com/YangLing0818/EditWorld)

> **Core Innovation**  
> Proposes the concept of world instructions, systematically defines and categorizes seven types of editing instructions that reflect world dynamics, constructs a multimodal dataset with complex semantic changes, and introduces a post-edit strategy to improve editing quality and consistency.

<details>
    <summary>Abstract</summary>
    Diffusion models have significantly improved the performance of image editing. Existing methods realize various approaches to achieve high-quality image editing, including but not limited to text control, dragging operation, and mask-and-inpainting. Among these, instruction-based editing stands out for its convenience and effectiveness in following human instructions across diverse scenarios. However, it still focuses on simple editing operations like adding, replacing, or deleting, and falls short of understanding aspects of world dynamics that convey the realistic dynamic nature in the physical world. Therefore, this work, EditWorld, introduces a new editing task, namely world-instructed image editing, which defines and categorizes the instructions grounded by various world scenarios. We curate a new image editing dataset with world instructions using a set of large pretrained models (e.g., GPT-3.5, Video-LLava and SDXL). To enable sufficient simulation of world dynamics for image editing, our EditWorld trains model in the curated dataset, and improves instruction-following ability with designed post-edit strategy. Extensive experiments demonstrate our method significantly outperforms existing editing methods in this new task.
</details>

<details>
    <summary>Key points</summary>
    * Introduces world-instructed image editing, going beyond traditional add/remove/replace operations.
    * Constructs a high-quality dataset with 10K+ triplets covering complex dynamics from real and virtual worlds.
    * Designs a dual-branch data generation pipeline: text-to-image synthesis + video frame extraction, leveraging GPT, Video-LLaVA, SDXL, etc.
    * Proposes MLLM Score, a new metric for better evaluating semantic alignment of editing results.
    * Introduces a Post-Edit method combining SAM and inpainting to improve editing precision and non-edited region consistency.
</details>
</details>

---

<details>
<summary><b>MIO: A Foundation Model on Multimodal Tokens</b></summary>

* **Authors:** Zekun Wang, King Zhu, Chunpu Xu, Wangchunshu Zhou, Jiaheng Liu, Yibo Zhang, Jiashuo Wang, Ning Shi, Siyu Li, Yizhi Li, Haoran Que, Zhaoxiang Zhang, Yuanxing Zhang, Ge Zhang, Ke Xu, Jie Fu, Wenhao Huang
* **arXiv ID:** 2409.17692
* **One-liner:** MIO is the first open-source any-to-any foundation model that simultaneously understands and generates text, image, speech and video in a unified, end-to-end, autoregressive way.
* **Published in:** EMNLP 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2409.17692) | [[PDF]](https://arxiv.org/pdf/2409.17692.pdf) | [[Code]](https://github.com/MIO-Team/MIO)

> **Core Innovation**  
> MIO is the first open-source any-to-any foundation model that simultaneously understands and generates text, image, speech and video in a unified, end-to-end, autoregressive way.

<details>
    <summary>Abstract</summary>
    State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on.
</details>

<details>
    <summary>Key points</summary>
    * DIDO discrete-token framework guarantees I/O consistency and natively supports multimodal interleaved generation.
    * SpeechTokenizer & SEED-Tokenizer provide compact, causal tokens for speech/video/image, ready for next-token prediction.
    * Three-stage pre-training (alignment → interleaved → speech-enhanced) + large-scale multi-task SFT deliver strong uni/bi-modal performance.
    * Competitive or SOTA results on image/text/speech/video benchmarks; emergent abilities: chain-of-visual-thought, visual story, instructional editing.
    * Fully open-source, 7 B-scale, end-to-end trainable, no external diffusion or TTS tools required.
</details>
</details>

---

<details>
<summary><b>SGEdit: Bridging LLM with Text2Image Generative Model for Scene Graph-based Image Editing</b></summary>

* **Authors:** Zhiyuan Zhang, DongDong Chen, Jing Liao
* **arXiv ID:** 2410.11815
* **One-liner:** SGEdit marries LLM reasoning with a fine-tuned diffusion model to turn a scene-graph interface into precise, open-vocabulary image edits—add, remove, replace or re-relate objects in one unified “remove-then-generate” pipeline.
* **Published in:** SIGGRAPH Asia 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2410.11815) | [[PDF]](https://arxiv.org/pdf/2410.11815.pdf) | [[Code]](https://github.com/bestzzhang/SGEdit-code)

> **Core Innovation**  
> First system that uses an LLM both as (1) an open-vocabulary scene parser that outputs masks + detailed captions and (2) an editing controller that plans attention-modulated remove/generate steps, all empowered by hybrid concept learning (optimized token + rich prompt) for identity-preserving yet editable objects.

<details>
    <summary>Abstract</summary>
    Scene graphs offer a structured, hierarchical representation of images, with nodes and edges symbolizing objects and the relationships among them. It can serve as a natural interface for image editing, dramatically improving precision and flexibility. Leveraging this benefit, we introduce a new framework that integrates large language model (LLM) with Text2Image generative model for scene graph-based image editing. This integration enables precise modifications at the object level and creative recomposition of scenes without compromising overall image integrity. Our approach involves two primary stages: ① Utilizing a LLM-driven scene parser, we construct an image's scene graph, capturing key objects and their interrelationships, as well as parsing fine-grained attributes such as object masks and descriptions. These annotations facilitate concept learning with a fine-tuned diffusion model, representing each object with an optimized token and detailed description prompt. ② During the image editing phase, a LLM editing controller guides the edits towards specific areas. These edits are then implemented by an attention-modulated diffusion editor, utilizing the fine-tuned model to perform object additions, deletions, replacements, and adjustments. Through extensive experiments, we demonstrate that our framework significantly outperforms existing image editing methods in terms of editing precision and scene aesthetics.
</details>

<details>
    <summary>Key points</summary>
    * Scene-graph-driven UI: users edit nodes/edges; system translates to executable ops.
    * LLM parser → graph + per-object mask & caption; hybrid concept learning fine-tunes SD for each object.
    * LLM controller decomposes any edit into sequential “remove → generate” with text prompts & boxes.
    * Attention-modulated removal fills masked regions by attending only to unmasked features; insertion enhances cross-attn alignment and suppresses inter-box self-attn. 
</details>
</details>

---

<details>
<summary><b>UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics</b></summary>

* **Authors:** Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, Hui Ding, Zhe Lin, Hengshuang Zhao
* **arXiv ID:** 2412.07774
* **One-liner:** UniReal treats any image task as “discontinuous video” inside a single 5 B diffusion transformer, learning real-world dynamics from massive video frames to unify generation, editing, customization and composition with one model.
* **Published in:** CVPR 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2412.07774) | [[PDF]](https://arxiv.org/pdf/2412.07774.pdf) | [[Code]](https://github.com/XavierCHEN34/UniReal)

> **Core Innovation**  
> First universal image model that reformulates diverse tasks into pseudo-video frame generation, leverages scalable video supervision for natural consistency/variation, and introduces hierarchical prompt (context + image role) plus index embedding to disambiguate multi-image inputs under one text prompt.

<details>
    <summary>Abstract</summary>
    We introduce UniReal, a unified framework designed to address various image generation and editing tasks. Existing solutions often vary by tasks, yet share fundamental principles: preserving consistency between inputs and outputs while capturing visual variations. Inspired by recent video generation models that effectively balance consistency and variation across frames, we propose a unifying approach that treats image-level tasks as discontinuous video generation. Specifically, we treat varying numbers of input and output images as frames, enabling seamless support for tasks such as image generation, editing, customization, composition, etc. Although designed for image-level tasks, we leverage videos as a scalable source for universal supervision. UniReal learns world dynamics from large-scale videos, demonstrating advanced capability in handling shadows, reflections, pose variation, and object interaction, while also exhibiting emergent capability for novel applications.
</details>

<details>
    <summary>Key points</summary>
    * Single 5B diffusion transformer; full attention across any number of input/output frames.
    * Three image roles—canvas, asset, control—plus learnable category & index embeddings bind visuals to prompt terms like “IMG1”.
    * Hierarchical prompt: base prompt + context tags (realistic/synthetic, static/dynamic, w/ or w/o ref) + image role tags; composable at inference.
    * Auto pipeline mines 23M+ frame pairs from videos with captions, masks, depth, edges → universal supervision for add, remove, replace, stylize, customize, insert, perception.
    * Emergent zero-shot abilities: multi-object insertion, depth-conditioned generation, reference inpainting, object resizing, etc.
    * Training: 256 → 512 → 1024 progressive resolution; flow-matching loss; handles arbitrary aspect ratios.
</details>
</details>

---

<details>
<summary><b>ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding</b></summary>

* **Authors:** Junliang Ye, Zhengyi Wang, Ruowen Zhao, Shenghao Xie, Jun Zhu
* **arXiv ID:** 2506.01853
* **One-liner:** ShapeLLM-Omni is the first native 3D-LLM that unifies text ⇄ image ⇄ 3D mesh generation, understanding and editing in one autoregressive next-token framework.
* **Published in:** NeurIPS 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2506.01853) | [[PDF]](https://arxiv.org/pdf/2506.01853.pdf) | [[Code]](https://github.com/JAMESYJL/ShapeLLM-Omni)

> **Core Innovation**  
> A 3D VQVAE compresses voxels into 1024 discrete tokens, enabling a frozen Qwen-2.5-VL-7B to perform fully autoregressive 3D generation/editing/understanding without extra networks; the 3D-Alpaca dataset (3.46 B tokens) makes it scalable.

<details>
    <summary>Abstract</summary>
    Recently, the powerful text-to-image capabilities of ChatGPT-4o have led to growing appreciation for native multimodal large language models. However, its multimodal capabilities remain confined to images and text. Yet beyond images, the ability to understand and generate 3D content is equally crucial. To address this gap, we propose ShapeLLM-Omni-a native 3D large language model capable of understanding and generating 3D assets and text in any sequence. First, we train a 3D vector-quantized variational autoencoder (VQVAE), which maps 3D objects into a discrete latent space to achieve efficient and accurate shape representation and reconstruction. Building upon the 3D-aware discrete tokens, we innovatively construct a large-scale continuous training dataset named 3D-Alpaca, encompassing generation, comprehension, and editing, thus providing rich resources for future research and training. Finally, by performing instruction-based training of the Qwen-2.5-vl-7B-Instruct model on the 3D-Alpaca dataset. Our work provides an effective attempt at extending multimodal models with basic 3D capabilities, which contributes to future research in 3D-native AI.
</details>

<details>
    <summary>Key points</summary>
    * Unified tokenizer: 643 → 163 latent → 1024 discrete tokens; 8192-codebook; reversible to high-quality mesh via Rectified-Flow.
    * Native 3D-LLM: same transformer handles text, image features, 3D tokens in any order; next-token prediction end-to-end.
    * 3D-Alpaca: 712k text-to-3D, 712k image-to-3D, 712k 3D caption, 420k 3D editing pairs + UltraChat; 2.56 M samples, 3.46 B tokens.
    * Tasks: text-to-3D, image-to-3D, 3D-to-text, interactive 3D editing (add/remove/modify), multi-turn dialogue.
</details>
</details>

---

<details>
<summary><b>Step1X-Edit: A Practical Framework for General Image Editing</b></summary>

* **Authors:** Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, Daxin Jiang
* **arXiv ID:** 2504.17761
* **One-liner:** Step1X-Edit closes the gap between open-source and proprietary editors by fusing MLLM reasoning with a DiT decoder, delivering GPT-4o-level quality on 11 real-world editing tasks without masks.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2504.17761) | [[PDF]](https://arxiv.org/pdf/2504.17761.pdf) | [[Code]](https://github.com/stepfun-ai/Step1X-Edit)

> **Core Innovation**  
> A lightweight token-refiner connector injects MLLM edit embeddings directly into a latent DiT via token-concatenation, enabling single-pass, high-fidelity editing trained only with diffusion loss—no masks, no extra self-attention, no T5 text encoder.

<details>
    <summary>Abstract</summary>
    In recent years, image editing models have witnessed remarkable and rapid development. The recent unveiling of cutting-edge multimodal models such as GPT-4o and Gemini2 Flash has introduced highly promising image editing capabilities. These models demonstrate an impressive aptitude for fulfilling a vast majority of user-driven editing requirements, marking a significant advancement in the field of image manipulation. However, there is still a large gap between the open-source algorithm with these closed-source models. Thus, in this paper, we aim to release a state-of-the-art image editing model, called Step1X-Edit, which can provide comparable performance against the closed-source models like GPT-4o and Gemini2 Flash. More specifically, we adopt the Multimodal LLM to process the reference image and the user's editing instruction. A latent embedding has been extracted and integrated with a diffusion image decoder to obtain the target image. To train the model, we build a data generation pipeline to produce a high-quality dataset. For evaluation, we develop the GEdit-Bench, a novel benchmark rooted in real-world user instructions. Experimental results on GEdit-Bench demonstrate that Step1X-Edit outperforms existing open-source baselines by a substantial margin and approaches the performance of leading proprietary models, thereby making significant contributions to the field of image editing.
</details>

<details>
    <summary>Key points</summary>
    * Unified MLLM+DiT pipeline: Qwen-VL parses image+instruction → compact multimodal tokens → DiT synthesizes edited image.
    * 20 M synthetic triplets filtered to 1 M high-quality (Step1X-Edit-HQ); covers 11 tasks: add/remove/replace, motion, material, color, style, tone, text, portrait, background.
    * GEdit-Bench: 606 real-user prompts (EN/CN), privacy-safe, first benchmark with genuine instructions & human filtering.
    * Token-concatenation conditioning preserves fine detail; global guidance vector from mean MLLM embedding enhances semantic alignment.
    * Trained from SD3.5-class base; supports plug-in replacement (FLUX, HiDream, etc.).
</details>
</details>

---

<details>
<summary><b>LangScene-X: Reconstruct Generalizable 3D Language-Embedded Scenes with TriMap Video Diffusion</b></summary>

* **Authors:** Fangfu Liu, Hao Li, Jiawei Chi, Hanyang Wang, Minghui Yang, Fudong Wang, Yueqi Duan
* **arXiv ID:** 2507.02813
* **One-liner:** LangScene-X bootstraps a TriMap video diffusion model with a Language-Quantized Compressor to build 3-D language-embedded Gaussian fields from only two images and answer open-vocabulary queries at novel views—without any per-scene optimization.
* **Published in:** ICCV 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2507.02813) | [[PDF]](https://arxiv.org/pdf/2507.02813.pdf) | [[Code]](https://github.com/liuff19/LangScene-X)

> **Core Innovation**  
> A progressive multi-task video diffusion first jointly generates RGB, normals and multi-granularity segmentation; a lightweight vector-quantized compressor (LQC) trained on COCO reduces 512-D CLIP features to 3-D discrete codes, enabling cross-scene 3-D Gaussian language fields that render in real time.

<details>
    <summary>Abstract</summary>
    Recovering 3D structures with open-vocabulary scene understanding from 2D images is a fundamental but daunting task. Recent developments have achieved this by performing per-scene optimization with embedded language information. However, they heavily rely on the calibrated dense-view reconstruction paradigm, thereby suffering from severe rendering artifacts and implausible semantic synthesis when limited views are available. In this paper, we introduce a novel generative framework, coined LangScene-X, to unify and generate 3D consistent multi-modality information for reconstruction and understanding. Powered by the generative capability of creating more consistent novel observations, we can build generalizable 3D language-embedded scenes from only sparse views. Specifically, we first train a TriMap video diffusion model that can generate appearance (RGBs), geometry (normals), and semantics (segmentation maps) from sparse inputs through progressive knowledge integration. Furthermore, we propose a Language Quantized Compressor (LQC), trained on large-scale image datasets, to efficiently encode language embeddings, enabling cross-scene generalization without per-scene retraining. Finally, we reconstruct the language surface fields by aligning language information onto the surface of 3D scenes, enabling open-ended language queries. Extensive experiments on real-world data demonstrate the superiority of our LangScene-X over state-of-the-art methods in terms of quality and generalizability.
</details>

<details>
    <summary>Key points</summary>
    * TriMap diffusion: 4-stage training (key-frame → 3-D consistent → add normals → add semantics) produces video-level coherent triplets.
    * LQC: VQ-VAE with 2048 codes; 1e-4 L2 loss, gradient bypass; compresses any CLIP field to 3-channel index, zero per-scene cost.
    * Language surface fields: DUSt3R sparse init + progressive normal regularization + 2-D/3-D semantic clustering; 5k steps RGB/Geo, 5k steps sem.
    * Inference: 2 input views → 49-frame TriMap → language Gaussians → real-time novel-view query.
    * Generalizable: one model handles indoor, object-level, in-the-wild; no scene-wise auto-encoder, no dense calibration.
</details>
</details>

---

<details>
<summary><b>MENTOR: Efficient Multimodal-Conditioned Tuning for Autoregressive Vision Generation Models</b></summary>

* **Authors:** Haozhe Zhao, Zefan Cai, Shuzheng Si, Liang Chen, Jiuxiang Gu, Wen Xiao, Junjie Hu
* **arXiv ID:** 2507.09574
* **One-liner:** MENTOR introduces an efficient autoregressive framework for multimodal image generation, achieving fine-grained alignment and balanced control via a two-stage training paradigm.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2507.09574) | [[PDF]](https://arxiv.org/pdf/2507.09574.pdf) | [[Code]](https://github.com/HaozheZhao/MENTOR)

> **Core Innovation**  
> The core innovation lies in an autoregressive image generation method that eliminates the need for cross-attention or auxiliary adapters, using a two-stage training strategy to achieve semantic and pixel-level alignment across multimodal inputs, greatly improving training efficiency and controllability.

<details>
    <summary>Abstract</summary>
    Recent text-to-image models produce high-quality results but still struggle with precise visual control, balancing multimodal inputs, and requiring extensive training for complex multimodal image generation. To address these limitations, we propose MENTOR, a novel autoregressive (AR) framework for efficient Multimodal-conditioned Tuning for Autoregressive multimodal image generation. MENTOR combines an AR image generator with a two-stage training paradigm, enabling fine-grained, token-level alignment between multimodal inputs and image outputs without relying on auxiliary adapters or cross-attention modules. The two-stage training consists of: (1) a multimodal alignment stage that establishes robust pixel- and semantic-level alignment, followed by (2) a multimodal instruction tuning stage that balances the integration of multimodal inputs and enhances generation controllability. Despite modest model size, suboptimal base components, and limited training resources, MENTOR achieves strong performance on the DreamBench++ benchmark, outperforming competitive baselines in concept preservation and prompt following. Additionally, our method delivers superior image reconstruction fidelity, broad task adaptability, and improved training efficiency compared to diffusion-based methods.
</details>

<details>
    <summary>Key points</summary>
    * Unified autoregressive architecture for joint image and text processing with simplified model design.
    * Two-stage training: Stage 1 aligns multimodal inputs; Stage 2 enhances instruction-following and cross-modal fusion.
    * Eliminates cross-attention modules, reducing training costs.
    * Supports various downstream tasks (e.g., segmentation, subject-driven generation, in-context generation) without architectural changes.
</details>
</details>

---

<details>
<summary><b>GSFixer: Improving 3D Gaussian Splatting with Reference-Guided Video Diffusion Priors</b></summary>

* **Authors:** Xingyilang Yin, Qi Zhang, Jiahao Chang, Ying Feng, Qingnan Fan, Xi Yang, Chi-Man Pun, Huaqi Zhang, Xiaodong Cun
* **arXiv ID:** 2508.09667
* **One-liner:** GSFixer introduces a reference-guided video diffusion prior method to restore artifacts in 3D Gaussian Splatting and improve reconstruction quality under sparse-view conditions.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2508.09667) | [[PDF]](https://arxiv.org/pdf/2508.09667.pdf) | [[Code]](https://github.com/GVCLab/GSFixer)

> **Core Innovation**  
> The core innovation lies in fusing 2D semantic and 3D geometric features as conditions to guide a video diffusion model for artifact restoration, along with a reference-guided trajectory sampling strategy for high-quality 3DGS reconstruction.

<details>
    <summary>Abstract</summary>
    Reconstructing 3D scenes using 3D Gaussian Splatting (3DGS) from sparse views is an ill-posed problem due to insufficient information, often resulting in noticeable artifacts. While recent approaches have sought to leverage generative priors to complete information for under-constrained regions, they struggle to generate content that remains consistent with input observations. To address this challenge, we propose GSFixer, a novel framework designed to improve the quality of 3DGS representations reconstructed from sparse inputs. The core of our approach is the reference-guided video restoration model, built upon a DiT-based video diffusion model trained on paired artifact 3DGS renders and clean frames with additional reference-based conditions. Considering the input sparse views as references, our model integrates both 2D semantic features and 3D geometric features of reference views extracted from the visual geometry foundation model, enhancing the semantic coherence and 3D consistency when fixing artifact novel views. Furthermore, considering the lack of suitable benchmarks for 3DGS artifact restoration evaluation, we present DL3DV-Res which contains artifact frames rendered using low-quality 3DGS. Extensive experiments demonstrate our GSFixer outperforms current state-of-the-art methods in 3DGS artifact restoration and sparse-view 3D reconstruction.
</details>

<details>
    <summary>Key points</summary>
    * Proposes a DiT-based video diffusion model for restoring artifact-prone views in 3DGS rendering.
    * Utilizes DINOv2 and VGGT to extract 2D semantic and 3D geometric features, enhancing view consistency and semantic fidelity.
    * Introduces a reference-guided trajectory sampling strategy to balance view coverage and restoration quality.
    * Presents DL3DV-Res dataset for evaluating 3DGS artifact restoration performance.
    * Significantly outperforms existing methods in sparse-view reconstruction and generalizes well across datasets.
</details>
</details>

---

<details>
<summary><b>CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields</b></summary>

* **Authors:** Can Wang, Menglei Chai, Mingming He, Dongdong Chen, Jing Liao
* **arXiv ID:** 2112.05139
* **One-liner:** CLIP-NeRF presents a multi-modal 3D object manipulation method using the CLIP model，enabling users to manipulate NeRF via brief text prompts or example images.
* **Published in:** CVPR 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2112.05139) | [[PDF]](https://arxiv.org/pdf/2112.05139.pdf) | [[Code]](https://github.com/cassiePython/CLIPNeRF)

> **Core Innovation**  
> The core innovation lies in the disentangled conditional NeRF architecture, which uses shape and appearance codes to separately control shape and appearance, combined with CLIP's multi-modal capabilities for text and image-driven 3D manipulation.

<details>
    <summary>Abstract</summary>
    We present CLIP-NeRF, a multi-modal 3D object manipulation method for neural radiance fields (NeRF). By leveraging the joint language-image embedding space of the recent Contrastive Language-Image Pre-Training (CLIP) model, we propose a unified framework that allows manipulating NeRF in a user-friendly way, using either a short text prompt or an exemplar image. Specifically, to combine the novel view synthesis capability of NeRF and the controllable manipulation ability of latent representations from generative models, we introduce a disentangled conditional NeRF architecture that allows individual control over both shape and appearance. This is achieved by performing the shape conditioning via applying a learned deformation field to the positional encoding and deferring color conditioning to the volumetric rendering stage. To bridge this disentangled latent representation to the CLIP embedding, we design two code mappers that take a CLIP embedding as input and update the latent codes to reflect the targeted editing. The mappers are trained with a CLIP-based matching loss to ensure the manipulation accuracy. Furthermore, we propose an inverse optimization method that accurately projects an input image to the latent codes for manipulation to enable editing on real images. We evaluate our approach by extensive experiments on a variety of text prompts and exemplar images and also provide an intuitive interface for interactive editing.
</details>

<details>
    <summary>Key points</summary>
    * Proposes a disentangled conditional NeRF architecture for separate shape and appearance control.
    * Designs a shape deformation network to update positional encodings via displacement vectors for precise shape control.
    * Leverages CLIP's text and image encoders to map text prompts or example images to the disentangled latent space for NeRF manipulation.
    * Introduces feedforward code mappers for fast inference, supporting editing of different objects within the same category.
    * Presents an inverse optimization method to project real images into latent codes for manipulation.
</details>
</details>

---

<details>
<summary><b>Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency</b></summary>

* **Authors:** Xiangyu Guo, Zhanqian Wu, Kaixin Xiong, Ziyang Xu, Lijun Zhou, Gangwei Xu, Shaoqing Xu, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Wenyu Liu, Xinggang Wang
* **arXiv ID:** 2506.07497
* **One-liner:** Genesis presents a unified framework for jointly generating multi-view driving videos and LiDAR sequences, achieving spatio-temporal and cross-modal consistency.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2506.07497) | [[PDF]](https://arxiv.org/pdf/2506.07497.pdf) | [[Code]](https://github.com/xiaomi-research/genesis)

> **Core Innovation**  
> The core innovation of Genesis lies in its unified multimodal generation architecture that enables coherent evolution of video and LiDAR data across visual and geometric domains through shared conditional inputs and structured semantic guidance from DataCrafter.

<details>
    <summary>Abstract</summary>
    We present Genesis, a unified framework for joint generation of multi-view driving videos and LiDAR sequences with spatio-temporal and cross-modal consistency. Genesis employs a two-stage architecture that integrates a DiT-based video diffusion model with 3D-VAE encoding, and a BEV-aware LiDAR generator with NeRF-based rendering and adaptive sampling. Both modalities are directly coupled through a shared latent space, enabling coherent evolution across visual and geometric domains. To guide the generation with structured semantics, we introduce DataCrafter, a captioning module built on vision-language models that provides scene-level and instance-level supervision. Extensive experiments on the nuScenes benchmark demonstrate that Genesis achieves state-of-the-art performance across video and LiDAR metrics (FVD 16.95, FID 4.24, Chamfer 0.611), and benefits downstream tasks including segmentation and 3D detection, validating the semantic fidelity and practical utility of the generated data.
</details>

<details>
    <summary>Key points</summary>
    * Genesis uses a dual-branch design where both video and LiDAR generation are conditioned on shared inputs (e.g., scene descriptions and layouts), ensuring cross-modal consistency.
    * Built on vision-language models, DataCrafter generates scene-level and instance-level descriptions to provide detailed semantic guidance for multimodal generation.
    * The video branch leverages a DiT-based diffusion model with 3D-VAE encoding for fine-grained visual dynamics, while the LiDAR branch uses BEV representation and NeRF-based rendering for accurate geometry.
    * BEV features extracted from image pathways are incorporated into the LiDAR diffusion model to enhance consistency between geometric and visual domains.
</details>
</details>

---

<details>
<summary><b>Hunyuan3D-Omni: A Unified Framework for Controllable Generation of 3D Assets</b></summary>

* **Authors:** Team Hunyuan3D: Bowen Zhang, Chunchao Guo, Haolin Liu, Hongyu Yan, Huiwen Shi, Jingwei Huang, Junlin Yu, Kunhong Li, Linus, Penghao Wang, Qingxiang Lin, Sicong Liu, Xianghui Yang, Yixuan Tang, Yunfei Zhao, Zeqiang Lai, Zhihao Liang, Zibo Zhao
* **arXiv ID:** 2509.21245
* **One-liner:** Hunyuan3D-Omni is a unified framework for fine-grained, controllable 3D asset generation based on multi-modal conditioning signals.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2509.21245) | [[PDF]](https://arxiv.org/pdf/2509.21245.pdf) | [[Code]](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni)

> **Core Innovation**  
> The core innovation lies in the unified control encoder that integrates diverse conditioning signals like point clouds, voxels, skeletons, and bounding boxes. Combined with a progressive, difficulty-aware training strategy, it enables precise control over the 3D generation process.

<details>
    <summary>Abstract</summary>
    Recent advances in 3D-native generative models have accelerated asset creation for games, film, and design. However, most methods still rely primarily on image or text conditioning and lack fine-grained, cross-modal controls, which limits controllability and practical adoption. To address this gap, we present Hunyuan3D-Omni, a unified framework for fine-grained, controllable 3D asset generation built on Hunyuan3D 2.1. In addition to images, Hunyuan3D-Omni accepts point clouds, voxels, bounding boxes, and skeletal pose priors as conditioning signals, enabling precise control over geometry, topology, and pose. Instead of separate heads for each modality, our model unifies all signals in a single cross-modal architecture. We train with a progressive, difficulty-aware sampling strategy that selects one control modality per example and biases sampling toward harder signals (e.g., skeletal pose) while downweighting easier ones (e.g., point clouds), encouraging robust multi-modal fusion and graceful handling of missing inputs. Experiments show that these additional controls improve generation accuracy, enable geometry-aware transformations, and increase robustness for production workflows.
</details>

<details>
    <summary>Key points</summary>
    * Hunyuan3D-Omni introduces a unified framework supporting diverse conditioning signals for precise control over geometry, topology, and pose in 3D generation.
    * It designs a unified control encoder to convert various conditions into embeddings, which are combined with image features to achieve controllable 3D generation.
    * A progressive, difficulty-aware sampling strategy is proposed, prioritizing harder signals during training to enhance the model's robustness to multimodal fusion.
    * Experiments show that additional control signals improve generation accuracy, enhance geometric fidelity, and increase robustness in production workflows compared to image-only 3D generation.
</details>
</details>

---

<details>
<summary><b>LERF: Language Embedded Radiance Fields</b></summary>

* **Authors:** Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, Matthew Tancik
* **arXiv ID:** 2303.09553
* **One-liner:** LERF optimizes CLIP language embeddings into a dense, multi-scale 3D field within NeRF, enabling real-time, open-vocabulary natural language queries of 3D scenes without fine-tuning or region proposals.
* **Published in:** ICCV 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2303.09553) | [[PDF]](https://arxiv.org/pdf/2303.09553.pdf) | [[Code]](https://github.com/kerrj/lerf)

> **Core Innovation**  
> The key innovation is learning a volumetric language field that renders CLIP embeddings through volume rendering, supervised by multi-scale feature pyramids from image crops, enabling hierarchical, view-consistent 3D semantic queries while preserving the original CLIP model's zero-shot capabilities.

<details>
    <summary>Abstract</summary>
    Humans describe the physical world using natural language to refer to specific 3D locations based on a vast range of properties: visual appearance, semantics, abstract associations, or actionable affordances. In this work we propose Language Embedded Radiance Fields (LERFs), a method for grounding language embeddings from off-the-shelf models like CLIP into NeRF, which enable these types of open-ended language queries in 3D. LERF learns a dense, multi-scale language field inside NeRF by volume rendering CLIP embeddings along training rays, supervising these embeddings across training views to provide multi-view consistency and smooth the underlying language field. After optimization, LERF can extract 3D relevancy maps for a broad range of language prompts interactively in real-time, which has potential use cases in robotics, understanding vision-language models, and interacting with 3D scenes. LERF enables pixel-aligned, zero-shot queries on the distilled 3D CLIP embeddings without relying on region proposals or masks, supporting long-tail open-vocabulary queries hierarchically across the volume. 
</details>

<details>
    <summary>Key points</summary>
    * Multi-Scale Language Field: Learns a dense 3D field conditioned on position and physical scale to capture hierarchical semantics
    * Volumetric CLIP Rendering: Renders CLIP embeddings along rays using NeRF's volume rendering weights, then normalizes to unit sphere
    * View-Consistency: Averaging embeddings across multiple training views produces more localized and 3D-consistent relevancy maps than 2D CLIP
    * No Fine-Tuning Required: Directly uses off-the-shelf CLIP model without training on segmentation datasets, preserving open-vocabulary capabilities
    * DINO Regularization: Incorporates self-supervised DINO features as a regularizer to improve field smoothness and object boundaries
    * Decoupled Architecture: Trains separate networks for language features (CLIP/DINO) and radiance field (RGB/density) to prevent language gradients from affecting geometry
</details>
</details>

---

<details>
<summary><b>Viewset Diffusion: (0-)Image-Conditioned 3D Generative Models from 2D Data</b></summary>

* **Authors:** Stanislaw Szymanowicz, Christian Rupprecht, Andrea Vedaldi
* **arXiv ID:** 2306.07881
* **One-liner:** Viewset Diffusion trains a diffusion model to generate viewsets (multi-view images) while internally reconstructing 3D radiance fields, enabling both probabilistic single-view reconstruction and unconditional 3D generation using only 2D multi-view supervision.
* **Published in:** ICCV 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2306.07881) | [[PDF]](https://arxiv.org/pdf/2306.07881.pdf) | [[Code]](https://github.com/szymanowiczs/viewset-diffusion)

> **Core Innovation**  
> The key innovation is framing 3D generation as viewset diffusion: by generating a set of 2D views and designing the denoiser as a 3D reconstructor followed by differentiable rendering, the model learns a 3D generative prior from 2D data alone, unifying ambiguous-aware reconstruction and generation in a single framework without requiring 3D ground truth.

<details>
    <summary>Abstract</summary>
    We present Viewset Diffusion, a diffusion-based generator that outputs 3D objects while only using multi-view 2D data for supervision. We note that there exists a one-to-one mapping between viewsets, i.e., collections of several 2D views of an object, and 3D models. Hence, we train a diffusion model to generate viewsets, but design the neural network generator to reconstruct internally corresponding 3D models, thus generating those too. We fit a diffusion model to a large number of viewsets for a given category of objects. The resulting generator can be conditioned on zero, one or more input views. Conditioned on a single view, it performs 3D reconstruction accounting for the ambiguity of the task and allowing to sample multiple solutions compatible with the input. The model performs reconstruction efficiently, in a feed-forward manner, and is trained using only rendering losses using as few as three views per viewset.
</details>

<details>
    <summary>Key points</summary>
    * Exploits the bijective mapping between viewsets and 3D models to enable 3D generation via 2D supervision only
    * The diffusion denoising function Φ reconstructs an explicit 3D radiance field (voxel grid) from noisy viewsets, decoded by differentiable rendering Ψ
    * Supports zero (unconditional), single, or few-view conditioning by applying different noise levels to different views (noised views for generation, clean views for conditioning)
    * 2D features are lifted to 3D via camera-aware unprojection, creating view-aligned feature volumes before aggregation
    * Attention-based Multi-view Fusion: Uses cross-attention to aggregate features from multiple views in an occlusion-aware manner, with learned per-category queries at coarsest level
    * Handles inherent ambiguity in single-view reconstruction by sampling from the learned distribution, producing diverse plausible 3D shapes instead of blurry averages
    * Requires only 3 views per object during training, significantly less than optimization-based methods (50+ views)
</details>
</details>

---

<details>
<summary><b>Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation</b></summary>

* **Authors:** Min-Seop Kwak, Junho Kim, Sangdoo Yun, Dongyoon Han, Taekyoung Kim, Seungryong Kim, Jin-Hwa Kim
* **arXiv ID:** 2506.11924
* **One-liner:** MoAI is a diffusion-based warping-and-inpainting framework that synthesizes aligned novel view images and geometry from sparse unposed reference images by distilling attention maps from the image branch to a parallel geometry branch, enabling extrapolative view synthesis without dense posed data or pose-embedded models.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2506.11924) | [[PDF]](https://arxiv.org/pdf/2506.11924.pdf) | [[Code]](https://github.com/cvlab-kaist/MoAI)

> **Core Innovation**  
> The key innovation is cross-modal attention instillation (MoAI), where spatial attention maps from the image diffusion branch are injected into the geometry diffusion branch during both training and inference, creating a synergistic multi-task learning framework that regularizes image synthesis with deterministic geometry cues while enhancing geometry completion with rich semantic image features, ensuring precise alignment between generated images and geometry.

<details>
    <summary>Abstract</summary>
    We introduce a diffusion-based framework that performs aligned novel view image and geometry generation via a warping-and-inpainting methodology. Unlike prior methods that require dense posed images or pose-embedded generative models limited to in-domain views, our method leverages off-the-shelf geometry predictors to predict partial geometries viewed from reference images, and formulates novel-view synthesis as an inpainting task for both image and geometry. To ensure accurate alignment between generated images and geometry, we propose cross-modal attention distillation, where attention maps from the image diffusion branch are injected into a parallel geometry diffusion branch during both training and inference. This multi-task approach achieves synergistic effects, facilitating geometrically robust image synthesis as well as well-defined geometry prediction. We further introduce proximity-based mesh conditioning to integrate depth and normal cues, interpolating between point cloud and filtering erroneously predicted geometry from influencing the generation process. Empirically, our method achieves high-fidelity extrapolative view synthesis on both image and geometry across a range of unseen scenes, delivers competitive reconstruction quality under interpolation settings, and produces geometrically aligned colored point clouds for comprehensive 3D completion.
</details>

<details>
    <summary>Key points</summary>
    * Warping-and-Inpainting Paradigm: Leverages off-the-shelf geometry predictors to estimate partial point clouds from reference images, projecting them to target viewpoints as sparse * geometric conditioning for inpainting
    * Transfers attention maps from image U-Net to geometry U-Net, aligning modalities via shared structural information while preventing harmful feature mixing
    * Converts sparse point clouds to mesh via ball-pivoting, concatenating depth and normal maps to filter erroneous projections and provide denser geometric cues
    * Handles target viewpoints outside the convex hull of reference cameras, generating plausible occluded/unseen regions using learned scene priors
    * Works with unposed reference images, eliminating the need for known camera poses as required by prior diffusion-based NVS methods
    * Jointly predicts novel view images and pointmaps, producing geometrically consistent colored point clouds for comprehensive 3D completion
    * Geometry completion's deterministic nature provides stronger training signals to regularize the more ambiguous image inpainting task
    * Concatenates key/value features from all reference views with target queries, enabling occlusion-aware feature aggregation across arbitrary numbers of inputs
</details>
</details>

---

<details>
<summary><b>Scaling Transformer-Based Novel View Synthesis Models with Token Disentanglement and Synthetic Data</b></summary>

* **Authors:** Nithin Gopalakrishnan Nair, Srinivas Kaza, Xuan Luo, Vishal M. Patel, Stephen Lombardi, Jungyeon Park
* **arXiv ID:** 2509.06950
* **One-liner:** This paper introduces a Token-Disentangled (Tok-D) Transformer block that explicitly distinguishes source and target tokens through layer-wise modulation, enabling efficient and robust training on synthetic multi-view data generated by diffusion models, thereby achieving state-of-the-art novel view synthesis with improved generalization and reduced computational cost.
* **Published in:** ICCV 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2509.06950) | [[PDF]](https://arxiv.org/pdf/2509.06950.pdf) | [[Project Page]](https://scaling3dnvs.github.io/)

> **Core Innovation**  
> The core innovation is the Token-Disentangled (Tok-D) Transformer block, which addresses the fundamental inefficiency in decoder-only NVS architectures by using indicator-based modulation to separately process source and target tokens at each layer (pre-modulation for scaling/shifting attention and MLP layers, post-modulation for scaling). This disentanglement eliminates redundant feature alignment, improves computational efficiency, and critically, makes the model robust to synthetic data artifacts, enabling effective data scaling via synthetic augmentation—an approach that previously degraded baseline model performance.

<details>
    <summary>Abstract</summary>
    Large transformer-based models have made significant progress in generalizable novel view synthesis (NVS) from sparse input views, generating novel viewpoints without the need for test-time optimization. However, these models are constrained by the limited diversity of publicly available scene datasets, making most real-world (in-the-wild) scenes out-of-distribution. To overcome this, we incorporate synthetic training data generated from diffusion models, which improves generalization across unseen domains. While synthetic data offers scalability, we identify artifacts introduced during data generation as a key bottleneck affecting reconstruction quality. To address this, we propose a token disentanglement process within the transformer architecture, enhancing feature separation and ensuring more effective learning. This refinement not only improves reconstruction quality over standard transformers but also enables scalable training with synthetic data. As a result, our method outperforms existing models on both in-dataset and cross-dataset evaluations, achieving state-of-the-art results across multiple benchmarks while significantly reducing computational costs.
</details>

<details>
    <summary>Key points</summary>
    * Reveals that decoder-only NVS transformers (e.g., LVSM) waste capacity aligning source/target features and are vulnerable to source noise, limiting scalability.
    * Introduces token-type-aware modulation (δ indicator) that generates separate style vectors and scale/shift parameters for source vs. target tokens at every transformer layer.
    * The disentanglement mechanism prevents artifact propagation from synthetic source views to target generation, unlike naive transformers.
    * Uses CAT3D to generate multi-view data but crucially reverses the conditioning/target roles (real conditioned image becomes target, synthetic views become sources), forcing realistic output and robustness.
    * Proposes 3D-consistent noise initialization for diffusion models via warping and noise blending to improve synthetic data quality.
</details>
</details>

---

<details>
<summary><b>NeRF-HuGS: Improved Neural Radiance Fields in Non-static Scenes Using Heuristics-Guided Segmentation</b></summary>

* **Authors:** Jiahao Chen, Yipeng Qin, Lingjie Liu, Jiangbo Lu, Guanbin Li
* **arXiv ID:** 2403.17537
* **One-liner:** NeRF-HuGS introduces a "Heuristics-Guided Segmentation" paradigm that synergizes hand-crafted cues (SfM features & color residuals) with SAM to accurately separate static scenes from transient distractors without prior knowledge, significantly boosting NeRF quality in non-static environments.
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2403.17537) | [[PDF]](https://arxiv.org/pdf/2403.17537.pdf) | [[Code]](https://github.com/cnhaox/NeRF-HuGS)

> **Core Innovation**  
> The core innovation is the Heuristics-Guided Segmentation (HuGS) framework that reframes static/transient separation as a collaborative process: instead of relying purely on heuristics or segmentation, it uses complementary heuristics (SfM-based for high-frequency textures, color-residual for low-frequency) to provide coarse static cues that guide SAM into producing precise, boundary-aware static maps, thus overcoming the fundamental trade-off between generality and accuracy in prior methods.

<details>
    <summary>Abstract</summary>
    Neural Radiance Field (NeRF) has been widely recognized for its excellence in novel view synthesis and 3D scene reconstruction. However, their effectiveness is inherently tied to the assumption of static scenes, rendering them susceptible to undesirable artifacts when confronted with transient distractors such as moving objects or shadows. In this work, we propose a novel paradigm, namely "Heuristics-Guided Segmentation" (HuGS), which significantly enhances the separation of static scenes from transient distractors by harmoniously combining the strengths of hand-crafted heuristics and state-of-the-art segmentation models, thus significantly transcending the limitations of previous solutions. Furthermore, we delve into the meticulous design of heuristics, introducing a seamless fusion of Structure-from-Motion (SfM)-based heuristics and color residual heuristics, catering to a diverse range of texture profiles. Extensive experiments demonstrate the superiority and robustness of our method in mitigating transient distractors for NeRFs trained in non-static scenes.
</details>

<details>
    <summary>Key points</summary>
    * NeRF's static scene assumption causes artifacts in real-world data with transient distractors like moving objects/shadows
    * Static objects' SfM feature points have significantly more cross-view matches than transient ones, and high/low-frequency textures require different heuristics
    * HuGS thresholds SfM static features → SAM generates initial static map → partially-trained Nerfacto provides color residual map → dual heuristics fusion → SAM again for final static mask
</details>
</details>

---

<details>
<summary><b>RealFusion: 360° Reconstruction of Any Object from a Single Image</b></summary>

* **Authors:** Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, Andrea Vedaldi
* **arXiv ID:** 2302.10663
* **One-liner:** Proposes RealFusion, a method that uses single-image textual inversion to generate custom prompts for Stable Diffusion, combined with SDS loss and efficient InstantNGP reconstruction, enabling 360° photorealistic 3D reconstruction of arbitrary objects from a single image.
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2302.10663) | [[PDF]](https://arxiv.org/pdf/2302.10663.pdf) | [[Code]](https://github.com/lukemelas/realfusion)

> **Core Innovation**  
> The core innovation is the single-image textual inversion technique that creates a custom text token embedding from a single input image (via heavy augmentations), which conditions the diffusion model to generate view-consistent, object-specific priors—bridging the gap between generic 2D diffusion and precise single-view 3D reconstruction without any 3D supervision.

<details>
    <summary>Abstract</summary>
    We consider the problem of reconstructing a full 360° photographic model of an object from a single image of it. We do so by fitting a neural radiance field to the image, but find this problem to be severely ill-posed. We thus take an off-the-self conditional image generator based on diffusion and engineer a prompt that encourages it to "dream up" novel views of the object. Using an approach inspired by DreamFields and DreamFusion, we fuse the given input view, the conditional prior, and other regularizers in a final, consistent reconstruction. We demonstrate state-of-the-art reconstruction results on benchmark images when compared to prior methods for monocular 3D reconstruction of objects. Qualitatively, our reconstructions provide a faithful match of the input view and a plausible extrapolation of its appearance and 3D shape, including to the side of the object not visible in the image.
</details>

<details>
    <summary>Key points</summary>
    * Methodology: Combines two simultaneous objectives—reconstruction loss (fitting the input view) and SDS-based prior loss (constraining novel views using Stable Diffusion).
    * Single-image textual inversion optimizes a custom token ⟨e⟩ using heavily-augmented versions of the input image, making the diffusion model object-specific rather than category-generic.
    * Uses InstantNGP for fast training and introduces coarse-to-fine training to reduce surface artifacts.
    * Adds normal smoothness regularization in 2D and mask loss to improve geometry.
</details>
</details>

---

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

### Spatial + Temporal Consistency

Capability Profile: Models that maintain 3D spatial structure while simulating temporal dynamics, but may have limited language understanding or controllability.

Significance: These models represent crucial technical achievements in understanding "how the 3D world moves," forming the physics engine component of world models.

#### Representative Works

<details>
<summary><b>DUSt3R: Geometric 3D Vision Made Easy</b></summary>

* **Authors:** Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jérôme Revaud  
* **arXiv ID:** 2312.14132  
* **One-liner:** Dense, unconstrained stereo 3D reconstruction from arbitrary image collections with no calibration/preset pose required  
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14132) | [[PDF]](https://arxiv.org/pdf/2312.14132.pdf)

> **Core Innovation**  
> Introduces a paradigm-shifting pipeline that replaces traditional multi-view stereo (MVS): it directly regresses dense point-maps without ever estimating camera intrinsics or extrinsics. By unifying single- and multi-view depth estimation, camera pose recovery, and reconstruction into one network, the approach excels even in casual, “in-the-wild” capture scenarios.

<details>
    <summary>Abstract</summary>
    Multi-view stereo reconstruction (MVS) in the wild requires to first estimate the camera parameters e.g. intrinsic and extrinsic parameters. These are usually tedious and cumbersome to obtain, yet they are mandatory to triangulate corresponding pixels in 3D space, which is the core of all best performing MVS algorithms. In this work, we take an opposite stance and introduce DUSt3R, a radically novel paradigm for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections, i.e. operating without prior information about camera calibration nor viewpoint poses. We cast the pairwise reconstruction problem as a regression of pointmaps, relaxing the hard constraints of usual projective camera models. We show that this formulation smoothly unifies the monocular and binocular reconstruction cases. In the case where more than two images are provided, we further propose a simple yet effective global alignment strategy that expresses all pairwise pointmaps in a common reference frame. We base our network architecture on standard Transformer encoders and decoders, allowing us to leverage powerful pretrained models. Our formulation directly provides a 3D model of the scene as well as depth information, but interestingly, we can seamlessly recover from it, pixel matches, relative and absolute camera. Exhaustive experiments on all these tasks showcase that the proposed DUSt3R can unify various 3D vision tasks and set new SoTAs on monocular/multi-view depth estimation as well as relative pose estimation. In summary, DUSt3R makes many geometric 3D vision tasks easy.
</details>

<details>
    <summary>Key points</summary>
    * Reformulates multi-view stereo as point-map regression, removing need for camera intrinsics/extrinsics.  
    * Unified framework that handles monocular and multi-view reconstruction seamlessly.  
    * Uses Transformer-based architecture to directly predict dense 3D geometry from images.  
    * Achieves state-of-the-art on depth, pose, and reconstruction benchmarks under unconstrained settings.  
</details>
</details>

---

<details>
<summary><b>4D Gaussian Splatting for Real-Time Dynamic Scene Rendering</b></summary>

* **Authors:** Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang  
* **arXiv ID:** 2310.08528  
* **One-liner:** A holistic 4D Gaussian splatting representation for dynamic scenes enabling real-time rendering of space-time geometry and appearance  
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2310.08528) | [[PDF]](https://arxiv.org/pdf/2310.08528.pdf) | [[Code]](https://github.com/hustvl/4DGaussians)

> **Core Innovation**  
> Proposes 4D Gaussian Splatting (4D-GS) that models space and time jointly: each Gaussian primitive carries spatio-temporal extent, dramatically boosting both training speed and real-time rendering performance on dynamic scenes.

<details>
    <summary>Abstract</summary>
    Representing and rendering dynamic scenes has been an important but challenging task. Especially, to accurately model complex motions, high efficiency is usually hard to guarantee. To achieve real-time dynamic scene rendering while also enjoying high training and storage efficiency, we propose 4D Gaussian Splatting (4D-GS) as a holistic representation for dynamic scenes rather than applying 3D-GS for each individual frame. In 4D-GS, a novel explicit representation containing both 3D Gaussians and 4D neural voxels is proposed. A decomposed neural voxel encoding algorithm inspired by HexPlane is proposed to efficiently build Gaussian features from 4D neural voxels and then a lightweight MLP is applied to predict Gaussian deformations at novel timestamps. Our 4D-GS method achieves real-time rendering under high resolutions, 82 FPS at an 800800 resolution on an RTX 3090 GPU while maintaining comparable or better quality than previous state-of-the-art methods.
</details>

<details>
    <summary>Key points</summary>
    * Introduces 4D Gaussian primitives that model space + time jointly (rather than separate per-frame 3D models).  
    * Enables real-time rendering (30+ fps) for dynamic scenes with high resolution.  
    * Efficient deformation field network and splatting renderer designed for dynamic geometry + appearance.  
    * Training and storage efficiency improved relative to prior dynamic scene representations.  
</details>
</details>

---

<details>
<summary><b>Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes</b></summary>

* **Authors:** Zhengqi Li, Simon Niklaus, Noah Snavely, Oliver Wang  
* **arXiv ID:** 2011.13084  
* **One-liner:** A continuous time-varying function representation capturing appearance, geometry and motion (scene flow) to perform novel view & time synthesis from monocular video  
* **Published in:** CVPR 2021  
* **Links:** [[Paper]](https://arxiv.org/abs/2011.13084) | [[PDF]](https://arxiv.org/pdf/2011.13084.pdf) | [[Code]](https://github.com/zhengqili/Neural-Scene-Flow-Fields)

> **Core Innovation**  
> Proposes Neural Scene Flow Fields (NSFF): a time-varying continuous representation that jointly encodes geometry, appearance, and 3D scene flow for dynamic scenes. Trained on monocular videos with known camera trajectories, NSFF enables simultaneous novel-view and temporal interpolation.

<details>
    <summary>Abstract</summary>
    We present a method to perform novel view and time synthesis of dynamic scenes, requiring only a monocular video with known camera poses as input. To do this, we introduce Neural Scene Flow Fields, a new representation that models the dynamic scene as a time-variant continuous function of appearance, geometry, and 3D scene motion. Our representation is optimized through a neural network to fit the observed input views. We show that our representation can be used for complex dynamic scenes, including thin structures, view-dependent effects, and natural degrees of motion. We conduct a number of experiments that demonstrate our approach significantly outperforms recent monocular view synthesis methods, and show qualitative results of space-time view synthesis on a variety of real-world videos.
</details>

<details>
    <summary>Key points</summary>
    * Represents dynamic scenes by a continuous 5D/6D function (space + time + view) capturing geometry, appearance & scene-flow.  
    * Optimized from input monocular video + known camera poses (no multi-view capture required).  
    * Supports novel view and novel time synthesis (i.e., view + motion interpolation).  
    * Handles challenging dynamic phenomena such as thin structures, specularities and complex motion.  
</details>
</details>

---

<details>
<summary><b>CoTracker: It is Better to Track Together</b></summary>

* **Authors:** Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht 
* **arXiv ID:** 2307.07635  
* **One-liner:** A transformer-based model that jointly tracks tens of thousands of 2D points across video frames rather than treating them independently  
* **Published in:** ECCV 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2307.07635) | [[PDF]](https://arxiv.org/pdf/2307.07635.pdf) | [[Code]](https://github.com/facebookresearch/co-tracker)

> **Core Innovation**  
> Introduces CoTracker: a Transformer-based architecture that jointly tracks thousands of 2D point trajectories, explicitly modeling inter-point dependencies to boost tracking accuracy for occluded or out-of-view points.

<details>
    <summary>Abstract</summary>
    We introduce CoTracker, a transformer-based model that tracks a large number of 2D points in long video sequences. Differently from most existing approaches that track points independently, CoTracker tracks them jointly, accounting for their dependencies. We show that joint tracking significantly improves tracking accuracy and robustness, and allows CoTracker to track occluded points and points outside of the camera view. We also introduce several innovations for this class of trackers, including using token proxies that significantly improve memory efficiency and allow CoTracker to track 70k points jointly and simultaneously at inference on a single GPU. CoTracker is an online algorithm that operates causally on short windows. However, it is trained utilizing unrolled windows as a recurrent network, maintaining tracks for long periods of time even when points are occluded or leave the field of view. Quantitatively, CoTracker substantially outperforms prior trackers on standard point-tracking benchmarks.
</details>

<details>
    <summary>Key points</summary>
    * Joint point-tracking: tracks many points together, modeling correlations rather than independent tracking.  
    * Uses transformer architecture with token proxies to manage memory and scale to tens of thousands of points.  
    * Works online in causal short windows but is trained as an unrolled long-window recurrent to handle long sequences and occlusion.  
    * Significant improvements in robustness and accuracy over traditional independent point tracking methods, especially under occlusion and out-of-view.  
</details>
</details>

---

<details>
<summary><b>GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control</b></summary>

* **Authors:** Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, Jun Gao
* **arXiv ID:** 2503.03751
*   **One-liner:** Image (first frame/seed frame) → Video.3D caching + precise camera control, emphasizing world-consistent video generation
*   **Published in:** CVPR 2025
*   **Links:** [[Paper]](https://arxiv.org/abs/2503.03751) | [[PDF]](https://arxiv.org/pdf/2503.03751) | [[Code]](https://github.com/nv-tlabs/GEN3C)
   
> **Core Innovation**  
> Turns camera poses into directly-renderable conditions via a 3D cache, leaving the model to fill only the dis-occluded regions for the new viewpoint—eliminating both imprecise camera control and temporal inconsistency in one shot.

<details>
    <summary>Abstract</summary>
    We present GEN3C, a generative video model with precise Camera Control and temporal 3D Consistency. Prior video models already generate realistic videos, but they tend to leverage little 3D information, leading to inconsistencies, such as objects popping in and out of existence. Camera control, if implemented at all, is imprecise, because camera parameters are mere inputs to the neural network which must then infer how the video depends on the camera. In contrast, GEN3C is guided by a 3D cache: point clouds obtained by predicting the pixel-wise depth of seed images or previously generated frames. When generating the next frames, GEN3C is conditioned on the 2D renderings of the 3D cache with the new camera trajectory provided by the user. Crucially, this means that GEN3C neither has to remember what it previously generated nor does it have to infer the image structure from the camera pose. The model, instead, can focus all its generative power on previously unobserved regions, as well as advancing the scene state to the next frame. Our results demonstrate more precise camera control than prior work, as well as state-of-the-art results in sparse-view novel view synthesis, even in challenging settings such as driving scenes and monocular dynamic video. Results are best viewed in videos.
</details>
<details>
    <summary>Key points</summary>
   Depth Anything v2 (metric version) is used to predict metric depth for a single reference image, and the pixels are back-projected into camera space to form a stable 3D representation (point cloud). Then, it is registered/scaled with the point cloud obtained by COLMAP triangulation of each training video, thereby unifying the camera trajectory from relative scale to metric scale. During inference, the camera trajectory preview is drawn and rendered in this interactive 3D point cloud.
</details>
</details>

---

## 🌟 The "Trinity" Prototype: Emerging World Models

This section highlights models that demonstrate **preliminary integration of all three consistencies**, exhibiting emergent world model capabilities. These systems represent the current frontier, showing glimpses of true world simulation.

### Text-to-World Generators

Models that generate dynamic, spatially consistent virtual environments from language descriptions.

**Key Characteristics:**
- ✅ Modality: Natural language understanding and pixel-space generation
- ✅ Spatial: 3D-aware scene composition with object permanence
- ✅ Temporal: Physically plausible dynamics and motion

#### Representative Works

<details>
<summary><b>OpenAI Sora</b></summary>

* **Authors:** OpenAI Research Team
* **Model:** Sora (2024)
* **One-liner:** A large-scale space-time generative model that treats videos as sequences of visual patches, enabling high-fidelity text-to-video, image-to-video, and video editing.
* **Published in:** 2024 (open release)
* **Links:** [[Model Page]](https://openai.com/sora) | [[Technical Overview]](https://openai.com/index/video-generation-models-as-world-simulators/)

> **Core Innovation**  
> Sora introduces a large-scale universal generative model that treats visual data (e.g., videos) as “patches,” supporting multiple input modalities (text, images, videos) and outputting new videos with flexible durations, resolutions, and aspect ratios.

<details>
    <summary>Abstract</summary>
    Sora is a general visual generation model capable of synthesizing videos and images across diverse durations, resolutions, and aspect ratios. It leverages a spatiotemporal representation of visual data and trains on patches, similar to token-based language models. Sora supports flexible conditioning (text, image, video) and demonstrates strong temporal coherence, physical realism, and camera motion control, pointing toward scalable video models as world simulators.
</details>

<details>
    <summary>Key points</summary>
    * Treats video as patch-based latent sequences (analogous to tokenization in LLMs)  
    * Unified model for video & image generation across resolutions and durations  
    * Strong physical consistency and camera control  
    * Supports text → video, image → video, video editing, and stylization  
</details>
</details>

---

<details>
<summary><b>Runway Gen-3 Alpha</b></summary>

* **Authors:** Runway Research Team
* **Model:** Gen-3 Alpha (2024)
* **One-liner:** Next-generation multimodal video foundation model with improved motion realism, fidelity, and controllability over Gen-2.
* **Published in:** 2024 (Alpha)
* **Links:** [[Model Page]](https://runwayml.com/research/introducing-gen-3-alpha)

> **Core Innovation**  
> Gen-3 Alpha is built on a brand-new, large-scale multimodal training infrastructure that accepts text, image, and video inputs, delivering superior motion fidelity, structural consistency, and controllability.

<details>
    <summary>Abstract</summary>
    Gen-3 Alpha is a new multimodal video generation model built on a scalable training stack designed for high-fidelity, controllable visual synthesis. It supports text-to-video, image-to-video, and hybrid creative workflows with strong temporal stability, realistic motion, and cinematic quality. The system introduces new control tools such as Motion Brush and camera path control for creator-centric video production.
</details>

<details>
    <summary>Key points</summary>
    * Text-to-video, image-to-video, and mixed creative inputs  
    * Enhanced motion, consistency, and cinematic realism vs. Gen-2  
    * New control tools (Motion Brush, camera path control)  
    * Designed for professional film & creator workflows  
</details>
</details>

---

<details>
<summary><b>Pika 1.0</b></summary>

* **Authors:** Pika Labs Team
* **Model:** Pika 1.0 (2023)
* **One-liner:** A user-friendly generative video platform supporting text-to-video, image-to-video, and video editing with multiple artistic and cinematic styles.
* **Published in:** 2023 (November)
* **Links:** [[Product Page]](https://pika.art/) | [[Launch Blog]](https://pika.art/blog)

> **Core Innovation**  
> Pika 1.0 democratizes video generation and editing: accessible via Web and Discord, it lets users create videos from text or images and edit existing footage in a wide range of visual styles.

<details>
    <summary>Abstract</summary>
    Pika 1.0 introduces a generative video model embedded in a web-based creation platform. It supports multi-style video synthesis (3D animation, anime, cinematic, cartoon) and allows users to animate images, edit video clips, and generate scenes through natural prompts. With an emphasis on accessibility and fast iteration, Pika aims to democratize AI-powered video creation for everyday creative workflows.
</details>

<details>
    <summary>Key points</summary>
    * Text-to-video, image-to-video, and video editing  
    * Multiple stylistic modes (3D, anime, cinematic, cartoon)  
    * Web UI + Discord integration for accessible creation  
    * Consumer-focused, fast iteration and community-driven growth  
</details>
</details>

---

### Embodied Intelligence Systems

Models designed for robotic control and autonomous agents that must integrate perception, spatial reasoning, and temporal prediction for real-world task execution.

**Key Characteristics:**
- Multimodal instruction following
- 3D spatial navigation and manipulation planning
- Predictive modeling of action consequences

#### Representative Works

<details>
<summary><b>RT-2: Vision-Language-Action Models</b></summary>

* **Authors:** Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, Brianna Zitkovich
* **arXiv ID:** 2307.15818
* **One-liner:** Trains a VLM on web-scale internet data + robot trajectories to map text + images directly to robotic actions for general-purpose real-world manipulation.
* **Published in:** CoRL 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.15818) | [[PDF]](https://arxiv.org/pdf/2307.15818.pdf) | [[Project Page]](https://robotics-transformer2.github.io/)

> **Core Innovation**  
> Extends large models from “see and talk” to “see, understand, and act,” pioneering the transfer of internet-scale vision-language knowledge to real-world robotic control, endowing robots with open-world reasoning and zero-shot manipulation capabilities.

<details>
    <summary>Abstract</summary>
    We study how vision-language models trained on Internet-scale data can be incorporated directly into end-to-end robotic control to boost generalization and enable emergent semantic reasoning. Our goal is to enable a single end-to-end trained model to both learn to map robot observations to actions and enjoy the benefits of large-scale pretraining on language and vision-language data from the web. To this end, we propose to co-fine-tune state-of-the-art vision-language models on both robotic trajectory data and Internet-scale vision-language tasks, such as visual question answering. In contrast to other approaches, we propose a simple, general recipe to achieve this goal: in order to fit both natural language responses and robotic actions into the same format, we express the actions as text tokens and incorporate them directly into the training set of the model in the same way as natural language tokens. We refer to such category of models as vision-language-action models (VLA) and instantiate an example of such a model, which we call RT-2. Our extensive evaluation (6k evaluation trials) shows that our approach leads to performant robotic policies and enables RT-2 to obtain a range of emergent capabilities from Internet-scale training. This includes significantly improved generalization to novel objects, the ability to interpret commands not present in the robot training data (such as placing an object onto a particular number or icon), and the ability to perform rudimentary reasoning in response to user commands (such as picking up the smallest or largest object, or the one closest to another object). We further show that incorporating chain of thought reasoning allows RT-2 to perform multi-stage semantic reasoning, for example figuring out which object to pick up for use as an improvised hammer (a rock), or which type of drink is best suited for someone who is tired (an energy drink).
</details>

<details>
    <summary>Key points</summary>
    * Extends LLM/VLM foundation modeling to robot control
    * Trained on *internet vision-language data + robot action data*
    * Outputs robotic action tokens
    * Strong zero-shot generalization to new objects, tasks, environment changes
</details>
</details>

---

<details>
<summary><b>GAIA-1: A Generative World Model for Autonomous Driving</b></summary>

* **Authors:** Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado
* **arXiv ID:** 2311.07541
* **One-liner:** A large-scale generative world model that simulates diverse driving scenarios and predicts future multi-agent behavior for closed-loop autonomous driving.
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2309.17080) | [[PDF]](https://arxiv.org/pdf/2309.17080.pdf)

> **Core Innovation**  
> Beyond trajectory forecasting, it learns “world evolution” itself—simultaneously generating multi-agent dynamics, realistic traffic behaviors, and authentic sensor signals for closed-loop simulation and synthesis of rare safety-critical scenarios.

<details>
    <summary>Abstract</summary>
    Autonomous driving promises transformative improvements to transportation, but building systems capable of safely navigating the unstructured complexity of real-world scenarios remains challenging. A critical problem lies in effectively predicting the various potential outcomes that may emerge in response to the vehicle's actions as the world evolves.
    To address this challenge, we introduce GAIA-1 ('Generative AI for Autonomy'), a generative world model that leverages video, text, and action inputs to generate realistic driving scenarios while offering fine-grained control over ego-vehicle behavior and scene features. Our approach casts world modeling as an unsupervised sequence modeling problem by mapping the inputs to discrete tokens, and predicting the next token in the sequence. Emerging properties from our model include learning high-level structures and scene dynamics, contextual awareness, generalization, and understanding of geometry. The power of GAIA-1's learned representation that captures expectations of future events, combined with its ability to generate realistic samples, provides new possibilities for innovation in the field of autonomy, enabling enhanced and accelerated training of autonomous driving technology.
</details>

<details>
    <summary>Key points</summary>
    * Generative world model for driving — not just trajectory prediction
    * Generates future agent motion, scene geometry, and sensor data
    * Closed-loop simulation for AD evaluation + training
    * Enables rare / edge-case generation for safety
</details>
</details>

---

<details>
<summary><b>PaLM-E: An Embodied Multimodal Language Model</b></summary>

* **Authors:** Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence
* **arXiv ID:** 2303.03378
* **One-liner:** Multimodal LLM that integrates real-world sensor inputs (vision, robotics state) into PaLM, enabling robotic reasoning and action planning.
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2303.03378) | [[PDF]](https://arxiv.org/pdf/2303.03378.pdf) | [[Project Page]](https://palm-e.github.io/)

> **Core Innovation**  
> Treats robot vision and state information as “tokens” fed into an LLM, turning the language model into an embodied-intelligence decision system that can understand the environment, plan actions, and execute tasks.

<details>
    <summary>Abstract</summary>
    Large language models excel at a wide range of complex tasks. However, enabling general inference in the real world, e.g., for robotics problems, raises the challenge of grounding. We propose embodied language models to directly incorporate real-world continuous sensor modalities into language models and thereby establish the link between words and percepts. Input to our embodied language model are multi-modal sentences that interleave visual, continuous state estimation, and textual input encodings. We train these encodings end-to-end, in conjunction with a pre-trained large language model, for multiple embodied tasks including sequential robotic manipulation planning, visual question answering, and captioning. Our evaluations show that PaLM-E, a single large embodied multimodal model, can address a variety of embodied reasoning tasks, from a variety of observation modalities, on multiple embodiments, and further, exhibits positive transfer: the model benefits from diverse joint training across internet-scale language, vision, and visual-language domains. Our largest model, PaLM-E-562B with 562B parameters, in addition to being trained on robotics tasks, is a visual-language generalist with state-of-the-art performance on OK-VQA, and retains generalist language capabilities with increasing scale.
</details>

<details>
    <summary>Key points</summary>
    * Embodied LLM combining robot sensor tokens + language
    * Joint vision-robot-language pretraining improves generalization
    * Handles long-horizon task reasoning and grounding
    * Builds toward unified *robotics-LLM world models*
</details>
</details>

---

## 📊 Benchmarks and Evaluation

**Current Challenge**: Existing metrics (FID, FVD, CLIP Score) inadequately assess world model capabilities, focusing on perceptual quality rather than physical understanding.

**Need for Comprehensive Benchmarks:**

A true world model benchmark should evaluate:
- 🧩 **Commonsense Physics Understanding**: Does the model respect gravity, momentum, conservation laws?
- 🔮 **Counterfactual Reasoning**: Can it predict outcomes of hypothetical interventions?
- ⏳ **Long-term Consistency**: Does coherence break down over extended simulation horizons?
- 🎯 **Goal-Directed Planning**: Can it chain actions to achieve complex objectives?
- 🎛️ **Controllability**: How precisely can users manipulate simulated elements?

#### Representative Works

<details>
<summary><b>WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation</b></summary>

* **Authors:** Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, Li Yuan
* **arXiv ID:** 2503.07265
* **One-liner:** Introduces WISE, a world-knowledge-aware evaluation framework that measures semantic alignment and common-sense correctness in text-to-image generation.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2503.07265) | [[PDF]](https://arxiv.org/pdf/2503.07265.pdf) | [[Code]](https://github.com/PKU-YuanGroup/WISE)

> **Core Innovation**  
> WISE argues that “evaluation must incorporate world knowledge” and constructs a text-to-image consistency assessment framework grounded in real-world semantic constraints, conceptual relations, and commonsense reasoning. This approach significantly improves measurement of text–image alignment and comprehension beyond mere CLIP similarity.

<details>
    <summary>Abstract</summary>
    Text-to-Image (T2I) models are capable of generating high-quality artistic creations and visual content. However, existing research and evaluation standards predominantly focus on image realism and shallow text-image alignment, lacking a comprehensive assessment of complex semantic understanding and world knowledge integration in text to image generation. To address this challenge, we propose WISE, the first benchmark specifically designed for World Knowledge-Informed Semantic Evaluation. WISE moves beyond simple word-pixel mapping by challenging models with 1000 meticulously crafted prompts across 25 sub-domains in cultural common sense, spatio-temporal reasoning, and natural science. To overcome the limitations of traditional CLIP metric, we introduce WIScore, a novel quantitative metric for assessing knowledge-image alignment. Through comprehensive testing of 20 models (10 dedicated T2I models and 10 unified multimodal models) using 1,000 structured prompts spanning 25 subdomains, our findings reveal significant limitations in their ability to effectively integrate and apply world knowledge during image generation, highlighting critical pathways for enhancing knowledge incorporation and application in next-generation T2I models. Code and data are available at this https URL.
</details>

<details>
    <summary>Key points</summary>
    * Evaluates world knowledge and common-sense reasoning in T2I models  
    * Measures entity correctness, relational consistency, attributes, and scene realism  
    * Shows stronger correlation with human preference vs CLIPScore/BERTScore  
    * Designed as a standardized benchmark for semantic T2I evaluation  
</details>
</details>

---

<details>
<summary><b>Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark</b></summary>

* **Authors:** Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, Pheng-Ann Heng
* **arXiv ID:** 2510.26802
* **One-liner:** Proposes MME-COF benchmark to evaluate zero-shot video reasoning ability across memory, motion, events, causality, and forecasting — showing that current video models lag behind VLMs.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2510.26802) | [[PDF]](https://arxiv.org/pdf/2510.26802.pdf) | [[Code]](https://github.com/ZiyuGuo99/MME-CoF)

> **Core Innovation**  
> For the first time systematically evaluates “zero-shot reasoning” capability of large video models, introduces the MME-CoF benchmark (Memory, Motion, Event, Causality, Object, Future prediction), and reveals that video models lag markedly behind vision-language LLMs in fine-grained causal understanding.

<details>
    <summary>Abstract</summary>
    Recent video generation models can produce high-fidelity, temporally coherent videos, indicating that they may encode substantial world knowledge. Beyond realistic synthesis, they also exhibit emerging behaviors indicative of visual perception, modeling, and manipulation. Yet, an important question still remains: Are video models ready to serve as zero-shot reasoners in challenging visual reasoning scenarios? In this work, we conduct an empirical study to comprehensively investigate this question, focusing on the leading and popular Veo-3. We evaluate its reasoning behavior across 12 dimensions, including spatial, geometric, physical, temporal, and embodied logic, systematically characterizing both its strengths and failure modes. To standardize this study, we curate the evaluation data into MME-CoF, a compact benchmark that enables in-depth and thorough assessment of Chain-of-Frame (CoF) reasoning. Our findings reveal that while current video models demonstrate promising reasoning patterns on short-horizon spatial coherence, fine-grained grounding, and locally consistent dynamics, they remain limited in long-horizon causal reasoning, strict geometric constraints, and abstract logic. Overall, they are not yet reliable as standalone zero-shot reasoners, but exhibit encouraging signs as complementary visual engines alongside dedicated reasoning models.
</details>

<details>
    <summary>Key points</summary>
    * MME-COF benchmark: Memory, Motion, Event, Causality, Object, Future prediction  
    * Zero-shot evaluation — no task-specific video fine-tuning  
    * Reveals video models are weaker in causal & common-sense reasoning  
    * Shows video models rely on pattern recognition rather than deep inference  
</details>
</details>

---

## 📝 Contributing

We welcome contributions of all forms! Including but not limited to:

- 🆕 Adding new papers, tools, or datasets
- 📝 Improving descriptions of existing entries
- 🐛 Fixing errors or outdated information
- 💡 Suggesting improvements

### How to Contribute

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- When adding new entries, please ensure they include:
  - 📄 Paper/tool name and link
  - 📝 Clear and concise description (1-2 sentences)
  - 🔗 Related links (code, dataset, blog, etc.)
- Maintain entries in alphabetical or importance order
- Ensure links are valid and point to official resources
- Use English for descriptions

---

## ⭐ Star History

If this project helps you, please give us a Star ⭐️!

[![Star History Chart](https://api.star-history.com/svg?repos=opendatalab-raiser/awesome-world-model-evolution&type=Date)](https://star-history.com/#opendatalab-raiser/awesome-world-model-evolution&Date)

---

## 📄 License

This project is licensed under [MIT License](LICENSE).




