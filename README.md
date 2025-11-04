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

- [Awesome World Model Evolution - Forging the World Model Universe from Unified Multimodal Models ](#awesome-world-model-evolution---forging-the-world-model-universe-from-unified-multimodal-models-)
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

### Modality Consistency

**Objective**: Establish bidirectional mappings between symbolic (language) and perceptual (vision) representations.

**Historical Significance**: These models created the first "symbol-perception bridges," solving the fundamental I/O problem for world models.

#### Representative Works

**CLIP: Learning Transferable Visual Models From Natural Language Supervision**

* **Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
* **arXiv ID:** 2103.00020
* **One-liner:** Large-scale contrastive learning on 400M image–text pairs enabling strong zero-shot transfer across vision tasks
* **Published in:** ICML 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2103.00020) | [[PDF]](https://arxiv.org/pdf/2103.00020.pdf) | [[Code]](https://github.com/openai/CLIP)

> **Core Innovation**  
> Contrastively trained on 400 million image–text pairs to establish a unified vision–language representation space, enabling zero-shot cross-task transfer and pioneering the large-scale multimodal training paradigm of “using text as a supervisory signal.”

<details>
    <summary>Abstract</summary>
    CLIP trains separate image and text encoders jointly via contrastive learning over 400M image–text pairs collected from the internet. The model demonstrates strong generalization, achieving competitive zero-shot accuracy on many vision benchmarks without finetuning.
</details>

<details>
    <summary>Key points</summary>
    * Contrastive learning between image and text features  
    * Weakly supervised web-scale dataset (400M pairs)  
    * Zero-shot classification via text prompts  
    * Sparked multimodal foundation model wave  
</details>

---

**DALL-E: Zero-Shot Text-to-Image Generation**

* **Authors:** Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever
* **arXiv ID:** 2102.12092
* **One-liner:** First large transformer trained to generate images from text, demonstrating compositional text-to-image creativity
* **Published in:** ICML 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2102.12092) | [[PDF]](https://arxiv.org/pdf/2102.12092.pdf) | [[Project Page]](https://openai.com/research/dall-e)

> **Core Innovation**  
> First to scale a Transformer for text-to-image generation, paired with a discrete VAE codec, demonstrating strong compositional generation of semantics (shape + style + attributes).

<details>
    <summary>Abstract</summary>
    DALL-E is a 12-billion parameter transformer trained to generate images from text prompts using discrete VAE tokens. It exhibits strong compositionality and creative visual synthesis abilities across diverse prompts.
</details>

<details>
    <summary>Key points</summary>
    * Text-conditioned transformer for image synthesis  
    * Uses VQ-VAE tokenization for images  
    * Compositional reasoning (shape + style + attribute)  
    * Launch of prompt-driven generative vision  
</details>

---

**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**

* **Authors:** Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio
* **arXiv ID:** 1502.03044
* **One-liner:** First visual attention mechanism for image captioning; pioneered attention in vision models
* **Published in:** ICML 2015
* **Links:** [[Paper]](https://arxiv.org/abs/1502.03044) | [[PDF]](https://arxiv.org/pdf/1502.03044.pdf)

> **Core Innovation**  
> First to introduce visual attention into image captioning, enabling the model to focus on key regions and produce interpretable attention maps—laying the groundwork for later Vision Transformers.

<details>
    <summary>Abstract</summary>
    This work introduces a soft and hard attention mechanism for image captioning, enabling models to dynamically focus on relevant regions while generating natural language descriptions.
</details>

<details>
    <summary>Key points</summary>
    * Early attention mechanism for vision  
    * CNN encoder + RNN decoder  
    * Visual heatmaps → interpretability  
    * Influenced attention-based transformers in vision  
</details>

---

**AttnGAN: Fine-Grained Text-to-Image Generation with Attentional GANs**

* **Authors:** Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He
* **arXiv ID:** 1711.10485
* **One-liner:** Word-region attention and semantic alignment loss for high-fidelity text-to-image synthesis
* **Published in:** CVPR 2018
* **Links:** [[Paper]](https://arxiv.org/abs/1711.10485) | [[PDF]](https://arxiv.org/pdf/1711.10485.pdf) | [[Code]](https://github.com/taoxugit/AttnGAN)

> **Core Innovation**  
> Achieves fine-grained text–image correspondence via word-level attention and semantic-alignment losses, markedly improving both the fidelity and semantic consistency of text-to-image generation.

<details>
    <summary>Abstract</summary>
    AttnGAN introduces fine-grained word-attention over image regions and a DAMSM semantic alignment loss to improve realism and semantic fidelity in text-to-image generation.
</details>

<details>
    <summary>Key points</summary>
    * Fine-grained word-to-region attention  
    * Multi-stage image refinement  
    * DAMSM loss for text-image consistency  
    * Major milestone before diffusion models  
</details>

---

### Spatial Consistency

**Objective**: Enable models to understand and generate 3D spatial structure from 2D observations.

**Historical Significance**: Provided methodologies for constructing internal "3D scene graphs" and understanding geometric relationships.

#### Representative Works

**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**

* **Authors:** Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng  
* **arXiv ID:** 2003.08934  
* **One-liner:** Represent a scene as a continuous volumetric neural radiance field and render novel views via volume rendering of the MLP output  
* **Published in:** ECCV 2020  
* **Links:** [[Paper]](https://arxiv.org/abs/2003.08934) | [[PDF]](https://arxiv.org/pdf/2003.08934.pdf)  

> **Core Innovation**  
> A scene is encoded by an MLP as a Neural Radiance Field (NeRF): given any 5D coordinate (x, y, z, θ, φ) it outputs the volume density plus view-dependent radiance at that location; novel-view images are then synthesized via differentiable volume rendering.

<details>
    <summary>Abstract</summary>
    We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Our algorithm represents a scene using a fully-connected (non-convolutional) deep network, whose input is a single continuous 5D coordinate (spatial location (x, y, z) and viewing direction (θ, φ)) and whose output is the volume density and view-dependent emitted radiance at that spatial location. We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally differentiable, the only input required to optimize our representation is a set of images with known camera poses. We describe how to effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance, and demonstrate results that outperform prior work on neural rendering and view synthesis.  
</details>

<details>
    <summary>Key points</summary>
    * Uses an MLP to represent a 5D function mapping (x, y, z, θ, φ) → (density, emitted radiance)  
    * Applies differentiable volume rendering for image synthesis  
    * Requires only input images + known camera poses (no explicit surface reconstruction)  
    * Demonstrates high-quality novel view synthesis of complex scenes  
</details>

---

**3D Gaussian Splatting for Real-Time Radiance Field Rendering**

* **Authors:** Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis  
* **arXiv ID:** 2308.04079  
* **One-liner:** Use anisotropic 3D Gaussians + splatting renderer to achieve real-time (≥30fps) high-quality novel-view synthesis of large scenes.  
* **Published in:** SIGGRAPH 2023  
* **Links:** [[Paper]](https://arxiv.org/abs/2308.04079) | [[PDF]](https://arxiv.org/pdf/2308.04079.pdf) | [[Code]](https://github.com/graphdeco-inria/gaussian-splatting)

> **Core Innovation**  
> Proposes representing scenes with thousands of anisotropic 3D Gaussians instead of a pure MLP, coupled with a visibility-aware tile-based splatting rasterizer, enabling real-time (1080p ≥ 30 fps) novel-view synthesis on large-scale scenes.

<details>
    <summary>Abstract</summary>
    In this paper we boost 3D GANs in both computational efficiency and image quality without leaning on heavy approximations. To this end we introduce a highly expressive hybrid explicit-implicit backbone that, together with several architectural refinements, synthesizes high-resolution, multi-view-consistent images in real time while yielding high-quality 3D geometry. By disentangling feature generation from neural rendering, our framework can plug in state-of-the-art 2D CNN generators such as StyleGAN2 and directly inherit their speed and expressiveness.
</details>

<details>
    <summary>Key points</summary>
    * Represents scene using anisotropic 3D Gaussians rather than only MLP or voxels  
    * Optimizes Gaussian parameters (density, covariance, color) for large scenes  
    * Employs a high-performance splatting renderer enabling real-time novel-view rendering at high resolution  
    * Targets full real-world scenes with real-time performance constraints  
</details>

---

**EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks**

* **Authors:** Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, Gordon Wetzstein  
* **arXiv ID:** 2112.07945  
* **One-liner:** Hybrid explicit-implicit 3D GAN (tri-plane) that synthesizes high-resolution, multi-view consistent imagery and high-quality geometry in real time.  
* **Published in:** CVPR 2022  
* **Links:** [[Paper]](https://arxiv.org/abs/2112.07945) | [[PDF]](https://arxiv.org/pdf/2112.07945.pdf) | [[Code]](https://github.com/NVlabs/eg3d)

> **Core Innovation**  
> Introduces a tri-plane-based hybrid explicit–implicit generator that equips 3D GANs with efficient computation while producing high-resolution, multi-view-consistent images and high-quality geometry.

<details>
    <summary>Abstract</summary>
    Neural graphics primitives parameterized by fully-connected networks incur high training and evaluation costs. We reduce these costs with a generic, novel input encoding: a small neural network is augmented by trainable multi-resolution hash-table feature vectors optimized via stochastic gradient descent. The multi-resolution structure lets the network resolve hash collisions automatically, yielding an architecture that is both simple and readily parallelized on modern GPUs. Exploiting this parallelism, we implement a fully-fused CUDA kernel that minimizes memory bandwidth and computational waste. Our system achieves orders-of-magnitude speed-ups, enabling high-quality neural primitives to be trained in seconds and rendered in tens of milliseconds.
</details>

<details>
    <summary>Key points</summary>
    * Introduces a tri-plane representation combining explicit 2D feature planes with implicit 3D decoding  
    * Enables use of strong 2D GAN architectures for 3D generation without sacrificing quality  
    * Produces multi-view consistent, high-resolution images plus underlying geometry  
    * Improves efficiency and realism of 3D GANs, bridging geometry and image synthesis  
</details>

---

**Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**

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

---

### Temporal Consistency

**Objective**: Model temporal dynamics, object motion, and causal relationships in video sequences.

**Historical Significance**: Early explorations of the world's "physics engine," capturing regularities in how scenes evolve over time.

#### Representative Works

**PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning**

* **Authors:** Yunbo Wang, Haixu Wu, Jianjin Zhang, Zhifeng Gao, Jianmin Wang, Philip S. Yu, Mingsheng Long
* **arXiv ID:** 2103.09504
* **One-liner:** A novel recurrent architecture that decouples memory cells to capture spatio-temporal dynamics in predictive learning of video frames.  
* **Published in:** TPAMI 2023 
* **Links:** [[Paper]](https://arxiv.org/abs/2103.09504) | [[PDF]](https://arxiv.org/pdf/2103.09504.pdf) | [[Code]](https://github.com/thuml/predrnn-pytorch)

> **Core Innovation**  
> Introduces decoupled memory cells embedded in an RNN backbone to separately capture spatial appearance and temporal dynamics, then fuses them into a unified complex-scene representation. A “zig-zag” memory flow propagates information across layers, while a curriculum-learning strategy enables modeling of long-term temporal dependencies.

<details>
    <summary>Abstract</summary>
    The predictive learning of spatiotemporal sequences aims to generate future images by learning from historical frames, where the visual dynamics are believed to have modular structures that can be learned with compositional subsystems. This paper models these structures by presenting a predictive recurrent neural network (PredRNN), in which a pair of memory cells are explicitly decoupled, operate in nearly independent transition manners, and finally form unified representations of the complex environment.  
</details>

<details>
    <summary>Key points</summary>
    * Introduces a memory‐pair structure: two memory cells decoupled in transition to separately capture spatial appearances and temporal dynamics. 
    * Unified representation by merging the outputs of the two memory streams.  
    * Applied to spatiotemporal predictive learning without explicit geometry or flow guidance.  
    * Demonstrates improved future-frame prediction performance on benchmark datasets.  
</details>

---

**SimVP: Simpler yet Better Video Prediction**

* **Authors:** Zhangyang Gao, Cheng Tan, Lirong Wu, Stan Z. Li
* **arXiv ID:** 2206.05099
* **One-liner:** A purely convolutional video-prediction model trained end-to-end with MSE loss, showing that simplicity can outperform more complex architectures.  
* **Published in:** CVPR 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2206.05099) | [[PDF]](https://arxiv.org/pdf/2206.05099.pdf) | [[Code]](https://github.com/A4Bio/SimVP)

> **Core Innovation**  
> Proposes an ultra-simple end-to-end video-prediction framework that uses only a CNN encoder–decoder trained with plain MSE loss; without RNNs, Transformers, or any sophisticated modules, it matches or surpasses far more complex models on multiple benchmarks.

<details>
    <summary>Abstract</summary>
    From CNN, RNN, to ViT, we have witnessed remarkable advancements in video prediction, incorporating auxiliary inputs, elaborate neural architectures, and sophisticated training strategies. We admire these progresses but are confused about the necessity: is there a simple method that can perform comparably well? This paper proposes SimVP, a simple video prediction model that is completely built upon CNN and trained by MSE loss in an end-to-end fashion. 
</details>

<details>
    <summary>Key points</summary>
    * Model based only on convolutional encoder-decoder structure (no RNN, no transformer).
    * Trained with straightforward Mean Squared Error (MSE) loss end-to-end.  
    * Demonstrates that simpler architectures can achieve competitive or superior results in video prediction benchmarks.  
    * Reduces architectural complexity and computational overhead compared to more sophisticated models.  
</details>

---

**Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning**

* **Authors:** Cheng Tan, Zhangyang Gao, Lirong Wu, Yongjie Xu, Jun Xia, Siyuan Li, Stan Z. Li  
* **arXiv ID:** 2206.12126 
* **One-liner:** Introduces a parallelizable temporal‐attention module (TAU) that splits intra-frame and inter-frame attention to improve efficiency in spatiotemporal predictive learning.  
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2206.12126) | [[PDF]](https://arxiv.org/pdf/2206.12126.pdf)

> **Core Innovation**  
> Introduces the Temporal Attention Unit (TAU) that disentles “intra-frame static attention” from “inter-frame dynamic attention” within spatiotemporal prediction modules, enabling fully parallel computation, and further strengthens inter-frame change detection via differential-divergence regularization.

<details>
    <summary>Abstract</summary>
    Spatiotemporal predictive learning aims to generate future frames by learning from historical frames. In this paper, we investigate existing methods and present a general framework of spatiotemporal predictive learning, in which the spatial encoder and decoder capture intra-frame features and the middle temporal module catches inter-frame correlations. While the mainstream methods employ recurrent units to capture long-term temporal dependencies, they suffer from low computational efficiency due to their unparallelizable architectures. To parallelize the temporal module, we propose the Temporal Attention Unit (TAU), which decomposes the temporal attention into intra-frame statical attention and inter-frame dynamical attention. Moreover, while the mean squared error loss focuses on intra-frame errors, we introduce a novel differential divergence regularization to take inter‐frame variations into account. Extensive experiments demonstrate that the proposed method enables the derived model to achieve competitive performance on various spatiotemporal prediction benchmarks.
</details>

<details>
    <summary>Key points</summary>
    * Proposes “Temporal Attention Unit (TAU)” that separates intra-frame static attention from inter-frame dynamic attention for efficiency and parallelism.
    * Introduces differential divergence regularization to better account for temporal variations (inter-frame changes).  
    * Framework decouples spatial encoding/decoding and temporal module, enabling better scalability.  
    * Shows improved computational efficiency while maintaining predictive accuracy on benchmark datasets.  
</details>

---

**VideoGPT: Video Generation using VQ-VAE and Transformers**

* **Authors:** Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas  
* **arXiv ID:** 2104.10157
* **One-liner:** A minimal architecture combining VQ-VAE and GPT-style transformer to scale generative modeling from images to natural videos.  
* **Published in:** arXiv 2021  
* **Links:** [[Paper]](https://arxiv.org/abs/2104.10157) | [[PDF]](https://arxiv.org/pdf/2104.10157.pdf) | [[Code]](https://github.com/wilson1yan/VideoGPT)

> **Core Innovation**  
> Applies VQ-VAE to learn discrete video latents (3D conv + axial self-attention) and then performs autoregressive modeling with a GPT-style Transformer over these tokens, enabling likelihood-based generation on natural videos.

<details>
    <summary>Abstract</summary>
    We present VideoGPT: a conceptually simple architecture for scaling likelihood-based generative modeling to natural videos. VideoGPT uses VQ-VAE that learns down-sampled discrete latent representations of a raw video by employing 3D convolutions and axial self-attention. A simple GPT-like architecture is then used to autoregressively model the discrete latents using spatio-temporal position encodings. Despite the simplicity in formulation and ease of training, our architecture is able to generate samples competitive with state-of-the-art GAN models for video generation on the BAIR Robot dataset, and generate high fidelity natural videos from UCF-101 and Tumbler GIF Dataset (TGIF). We hope our proposed architecture serves as a reproducible reference for a minimalistic implementation of transformer based video generation models. 
</details>

<details>
    <summary>Key points</summary>
    * Uses VQ-VAE to learn discrete latent representations of video using 3D convolutions + axial self-attention.  
    * Uses a GPT-style autoregressive transformer on those discrete latents with spatio-temporal positional encoding.  
    * Achieves competitive quality compared to GAN-based video generation models, but with simpler likelihood-based framework.  
    * Provides a reproducible baseline for transformer-based video generation on natural video datasets.  
</details>

---

**Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions**

* **Authors:** Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, Dumitru Erhan  
* **arXiv ID:** 2210.02399
* **One-liner:** Generates variable-length videos from a sequence of open-domain text prompts by tokenizing video representation and using a masked‐transformer conditioned on text.  
* **Published in:** ICLR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2210.02399) | [[PDF]](https://arxiv.org/pdf/2210.02399.pdf)

> **Core Innovation**  
> Proposes a video-tokenizer that discretizes variable-length videos into tokens and a conditional masked Transformer that generates these tokens before decoding them into frames. By jointly training on large-scale image-text data plus a small number of video-text pairs, it enables open-domain, variable-duration video generation from text prompts.

<details>
    <summary>Abstract</summary>
    We present Phenaki, a model capable of realistic video synthesis, given a sequence of textual prompts. Generating videos from text is particularly challenging due to the computational cost, limited quantities of high-quality text-video data and variable length of videos. To address these issues, we introduce a new model for learning video representation which compresses the video to a small representation of discrete tokens. This tokenizer uses causal attention in time, which allows it to work with variable-length videos. To generate video tokens from text we are using a bidirectional masked transformer conditioned on pre-computed text tokens. The generated video tokens are subsequently de-tokenized to create the actual video. To address data issues, we demonstrate how joint training on a large corpus of image-text pairs as well as a smaller number of video-text examples can result in generalization beyond what is available in the video datasets. Compared to the previous video generation methods, Phenaki can generate arbitrary long videos conditioned on a sequence of prompts (i.e., time-variable text or a story) in open domain.
</details>

<details>
    <summary>Key points</summary>
    * Introduces a discrete tokenization of video representation, enabling variable-length output.  
    * Uses a bidirectional masked transformer conditioned on text tokens to generate video tokens, then de-tokenizes to actual frames.  
    * Leverages joint training on large image-text corpus and smaller video-text datasets to improve generalization.  
    * Capable of generating long videos from sequences of prompts, rather than single static text.  
</details>

---

## 🔗 Preliminary Integration: Unified Multimodal Models

Current state-of-the-art models are beginning to break down the barriers between individual consistencies. This section showcases models that successfully integrate **two** of the three fundamental consistencies, representing crucial intermediate steps toward complete world models.

### Modality + Spatial Consistency

**Capability Profile**: Models that can translate text/image descriptions into spatially coherent 3D representations or multi-view consistent outputs.

**Significance**: These models demonstrate "3D imagination" - they are no longer mere "2D painters" but "digital sculptors" understanding spatial structure.

#### Representative Works

**Zero-1-to-3: Zero-shot One Image to 3D Object**

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

---

**MVDream: Multi-view Diffusion for 3D Generation**

* **Authors:** Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, Xiao Yang  
* **arXiv ID:** 2308.16512  
* **One-liner:** A multi-view diffusion model that generates geometrically consistent multi-view images from text prompts, enabling 3D generation via 2D+3D synergy.  
* **Published in:** ICLR 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2308.16512) | [[PDF]](https://arxiv.org/pdf/2308.16512.pdf) | [[Code]](https://github.com/bytedance/MVDream)

> **Core Innovation**  
> Proposes a diffusion model that produces geometrically-consistent multi-view images by first pretraining on 2D images and then fine-tuning on 3D data, inheriting the generalization power of 2D diffusion while guaranteeing 3D-render consistency—serving as a universal prior for 3D content generation.

<details>
    <summary>Abstract</summary>
    We introduce MVDream, a diffusion model that is able to generate consistent multi-view images from a given text prompt. Learning from both 2D and 3D data, a multi-view diffusion model can achieve the generalizability of 2D diffusion models and the consistency of 3D renderings. We demonstrate that such a multi-view diffusion model is implicitly a generalizable 3D prior agnostic to 3D representations. It can be applied to 3D generation via Score Distillation Sampling, significantly improving geometry consistency without requiring explicit 3D supervision.  
</details>

<details>
    <summary>Key points</summary>
    * Combines web-scale 2D diffusion pre-training with fine-tuning on multi-view 3D asset rendered dataset.  
    * Generates sets of consistent multi-view images conditioned on text prompts, facilitating 3D reconstruction/generation tasks.  
    * Serves as a 3D-agnostic prior: the model does not commit to one explicit 3D representation but still ensures multi-view consistency.  
    * Demonstrates improved geometry fidelity using Score Distillation Sampling driven by multi-view images.  
</details>

---

**Wonder3D: Single Image to 3D using Cross-Domain Diffusion**

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

---

**SyncDreamer: Generating Multiview-consistent Images from a Single-view Image**

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

---

### Modality + Temporal Consistency

**Capability Profile**: Models that transform textual descriptions or static images into temporally coherent, dynamic video sequences.

**Significance**: Currently the most prominent integration direction, enabling high-quality text-to-video and image-to-video generation.

#### Representative Works

**Lumiere: A Space-Time Diffusion Model for Video Generation**

* **Authors:** Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, Inbar Mosseri  
* **arXiv ID:** 2401.12945  
* **One-liner:** A text-to-video diffusion model with a space-time U-Net that generates the full video in one pass for realistic, coherent motion. 
* **Published in:** SIGGRAPH-ASIA 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2401.12945) | [[PDF]](https://arXiv.org/pdf/2401.12945.pdf) | [[Project Page]](https://lumiere-video.github.io/)

> **Core Innovation**  
> Proposes a Space-Time U-Net architecture that processes both spatial and temporal dimensions in a single forward pass, generating full-duration videos in one shot and thereby eliminating the temporal-inconsistency issues inherent in previous multi-stage key-frame + temporal super-resolution cascades.

<details>
    <summary>Abstract</summary>
    We introduce Lumiere -- a text-to-video diffusion model designed for synthesizing videos that portray realistic, diverse and coherent motion. To this end, we introduce a Space-Time U-Net architecture that generates the entire temporal duration of the video at once, through a single pass in the model. 
</details>

<details>
    <summary>Key points</summary>
    * Space-Time U-Net that jointly handles spatial and temporal down/up-sampling in one pass.  
    * Bypasses multi-stage temporal super-resolution cascades.  
    * Demonstrates improved motion coherence and diverse video generation.  
</details>

---

**Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets**

* **Authors:** Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach  
* **arXiv ID:** 2311.15127  
* **One-liner:** A latent video diffusion framework scaling to large datasets for high-resolution text-to-video and image-to-video generation.
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2311.15127) | [[PDF]](https://arxiv.org/pdf/2311.15127.pdf) | [[Code]](https://github.com/Stability-AI/generative-models)

> **Core Innovation**  
> We present a systematic three-stage pipeline for training latent video diffusion models—text-to-image pretraining, video pretraining, and high-quality video fine-tuning—and underscore the critical role of curated data selection and captioning, enabling high-resolution generation and multi-view 3D priors at scale.

<details>
    <summary>Abstract</summary>
    We present Stable Video Diffusion – a latent video diffusion model for high-resolution, state-of-the-art text-to-video and image-to-video generation. Recently, latent diffusion models trained for 2D image synthesis have been turned into generative video models by inserting temporal layers and finetuning them on small, high-quality video datasets. However, training methods in the literature vary widely, and the field has yet to agree on a unified strategy for curating video data. 
</details>

<details>
    <summary>Key points</summary>
    * Identifies three training stages: text-to-image pretraining, video pretraining, high-quality video fine-tuning. 
    * Highlights the importance of well-curated large scale video data for video diffusion performance.  
    * Demonstrates latent video diffusion’s capability to act as a multi-view 3D prior and to enable image-to-video generation and camera motion control.  
</details>

---

**AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning**

* **Authors:** Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, Bo Dai  
* **arXiv ID:** 2307.04725  
* **One-liner:** A plug-and-play motion module that animates any personalized text-to-image diffusion model without requiring model-specific tuning. 
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.04725) | [[PDF]](https://arxiv.org/pdf/2307.04725.pdf) | [[Code]](https://github.com/guoyww/AnimateDiff)

> **Core Innovation**  
> Proposes a plug-and-play Motion Module: trained once on top of a frozen text-to-image diffusion model, it integrates seamlessly to turn any existing T2I checkpoint into an animation generator. Further introduces MotionLoRA—a lightweight fine-tuning recipe for fast adaptation to new motion patterns without retraining the full model.

<details>
    <summary>Abstract</summary>
    With the advance of text-to-image (T2I) diffusion models and corresponding personalization techniques, generating animations remains a challenge. This paper presents AnimateDiff, a practical framework for animating personalized T2I models without requiring model-specific tuning. At the core is a plug-and-play motion module trained once and integrated into any personalized T2I model. A lightweight fine-tuning technique MotionLoRA allows adaptation to new motion patterns. Empirical results show the capability to generate temporally smooth animation clips while preserving image quality and motion diversity. 
</details>

<details>
    <summary>Key points</summary>
    * A motion module that can be plugged into any personalized T2I model – no model-specific re-training required.  
    * MotionLoRA: lightweight fine-tuning for adapting motion patterns with low cost.  
    * Enables high-quality animation generation from existing personalized image models.  
</details>

---

**Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning**

* **Authors:** Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, Ishan Misra  
* **arXiv ID:** 2311.10709  
* **One-liner:** A two-stage text-to-video generation: first image then video conditioned on both text and generated image. 
* **Published in:** ECCV 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2311.10709) | [[PDF]](https://arxiv.org/pdf/2311.10709.pdf) | [[Project Page]](https://emu-video.metademolab.com/)

> **Core Innovation**  
> Proposes a factorized generation pipeline: first synthesize an image conditioned on text, then generate a video conditioned on both that image and the same text. With custom noise schedules and multi-stage training strategies, it bypasses the limitations of previous deep cascades and directly produces high-quality, high-resolution videos.

<details>
    <summary>Abstract</summary>
    We present Emu Video, a text-to-video generation model that factorizes the generation into two steps: first generating an image conditioned on the text, and then generating a video conditioned on the text and the generated image. We identify critical design decisions—adjusted noise schedules for diffusion, and multi-stage training—that enable us to directly generate high quality and high resolution videos, without requiring a deep cascade of models as in prior work. 
</details>

<details>
    <summary>Key points</summary>
    * Factorization into image generation → video generation conditioned on (text + generated image).  
    * Adjusted diffusion noise schedules and multi-stage training to enable high-quality, high-resolution video generation.  
    * Demonstrates significant quality improvements over previous text-to-video approaches.  
</details>

---

**VideoPoet: A Large Language Model for Zero-Shot Video Generation**

* **Authors:** Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang  
* **arXiv ID:** 2312.14125  
* **One-liner:** A decoder-only transformer trained like an LLM to perform zero-shot video generation from a wide variety of conditioning signals (text, image, audio, video). 
* **Published in:** ICML 2024 
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14125) | [[PDF]](https://arxiv.org/pdf/2312.14125.pdf) | [[Project Page]](https://sites.research.google/videopoet/)

> **Core Innovation**  
> Extends the large language model training paradigm to video generation: a decoder-only Transformer processes multimodal inputs (text, image, audio, video) and produces high-quality videos (with matching audio) in a zero-shot setting, opening a new route for generative video modeling.

<details>
    <summary>Abstract</summary>
    We present VideoPoet, a model capable of synthesizing high-quality video, with matching audio, from a large variety of conditioning signals. VideoPoet employs a decoder-only transformer architecture that processes multimodal inputs – including images, videos, text, and audio. The training protocol follows that of Large Language Models (LLMs), consisting of two stages: pretraining and task-specific adaptation. 
</details>

<details>
    <summary>Key points</summary>
    * Uses a decoder-only transformer that ingests multimodal signals (text, image, video, audio).  
    * Training protocol similar to LLMs: large-scale pretraining + adaptation.  
    * Enables zero-shot video generation across many conditioning types.  
</details>

---

### Spatial + Temporal Consistency

Capability Profile: Models that maintain 3D spatial structure while simulating temporal dynamics, but may have limited language understanding or controllability.

Significance: These models represent crucial technical achievements in understanding "how the 3D world moves," forming the physics engine component of world models.

#### Representative Works

**DUSt3R: Geometric 3D Vision Made Easy**

* **Authors:** Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jérôme Revaud  
* **arXiv ID:** 2312.14132  
* **One-liner:** Dense, unconstrained stereo 3D reconstruction from arbitrary image collections with no calibration/preset pose required  
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14132) | [[PDF]](https://arxiv.org/pdf/2312.14132.pdf)

> **Core Innovation**  
> Introduces a paradigm-shifting pipeline that replaces traditional multi-view stereo (MVS): it directly regresses dense point-maps without ever estimating camera intrinsics or extrinsics. By unifying single- and multi-view depth estimation, camera pose recovery, and reconstruction into one network, the approach excels even in casual, “in-the-wild” capture scenarios.

<details>
    <summary>Abstract</summary>
    We present DUSt3R, a radically novel approach for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections. Multi-view stereo reconstruction (MVS) in the wild requires to first estimate the camera parameters, which are tedious and mandatory in all best-performing MVS algorithms. In this work we cast the problem as a regression of dense point-maps, and show that we can drop calibration and pose estimation altogether. We outperform prior methods on multiple 3D vision tasks including monocular/multi-view depth estimation and relative pose estimation.  
</details>

<details>
    <summary>Key points</summary>
    * Reformulates multi-view stereo as point-map regression, removing need for camera intrinsics/extrinsics.  
    * Unified framework that handles monocular and multi-view reconstruction seamlessly.  
    * Uses Transformer-based architecture to directly predict dense 3D geometry from images.  
    * Achieves state-of-the-art on depth, pose, and reconstruction benchmarks under unconstrained settings.  
</details>

---

**4D Gaussian Splatting for Real-Time Dynamic Scene Rendering**

* **Authors:** Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang  
* **arXiv ID:** 2310.08528  
* **One-liner:** A holistic 4D Gaussian splatting representation for dynamic scenes enabling real-time rendering of space-time geometry and appearance  
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2310.08528) | [[PDF]](https://arxiv.org/pdf/2310.08528.pdf) | [[Code]](https://github.com/hustvl/4DGaussians)

> **Core Innovation**  
> Proposes 4D Gaussian Splatting (4D-GS) that models space and time jointly: each Gaussian primitive carries spatio-temporal extent, dramatically boosting both training speed and real-time rendering performance on dynamic scenes.

<details>
    <summary>Abstract</summary>
    Representing and rendering dynamic scenes has been an important but challenging task. Especially, to accurately model complex motions, high efficiency is usually hard to guarantee. To achieve real-time dynamic scene rendering while also enjoying high training and storage efficiency, we propose 4D Gaussian Splatting (4D-GS) as a holistic representation for dynamic scenes rather than applying 3D-GS for each individual frame. Our experiments show high-fidelity rendering at real-time frame rates under large scenes.  
</details>

<details>
    <summary>Key points</summary>
    * Introduces 4D Gaussian primitives that model space + time jointly (rather than separate per-frame 3D models).  
    * Enables real-time rendering (30+ fps) for dynamic scenes with high resolution.  
    * Efficient deformation field network and splatting renderer designed for dynamic geometry + appearance.  
    * Training and storage efficiency improved relative to prior dynamic scene representations.  
</details>

---

**Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes**

* **Authors:** Zhengqi Li, Simon Niklaus, Noah Snavely, Oliver Wang  
* **arXiv ID:** 2011.13084  
* **One-liner:** A continuous time-varying function representation capturing appearance, geometry and motion (scene flow) to perform novel view & time synthesis from monocular video  
* **Published in:** CVPR 2021  
* **Links:** [[Paper]](https://arxiv.org/abs/2011.13084) | [[PDF]](https://arxiv.org/pdf/2011.13084.pdf) | [[Code]](https://github.com/zhengqili/Neural-Scene-Flow-Fields)

> **Core Innovation**  
> Proposes Neural Scene Flow Fields (NSFF): a time-varying continuous representation that jointly encodes geometry, appearance, and 3D scene flow for dynamic scenes. Trained on monocular videos with known camera trajectories, NSFF enables simultaneous novel-view and temporal interpolation.

<details>
    <summary>Abstract</summary>
    We present a method to perform novel view and time synthesis of dynamic scenes, requiring only a monocular video with known camera poses as input. To do this, we introduce Neural Scene Flow Fields, a new representation that models the dynamic scene as a time-variant continuous function of appearance, geometry, and 3D scene motion. Our representation is optimized through a neural network to fit the observed input views. We show that our representation can be used for complex dynamic scenes, including thin structures, view-dependent effects, and natural degrees of motion.  
</details>

<details>
    <summary>Key points</summary>
    * Represents dynamic scenes by a continuous 5D/6D function (space + time + view) capturing geometry, appearance & scene-flow.  
    * Optimized from input monocular video + known camera poses (no multi-view capture required).  
    * Supports novel view and novel time synthesis (i.e., view + motion interpolation).  
    * Handles challenging dynamic phenomena such as thin structures, specularities and complex motion.  
</details>

---

**CoTracker: It is Better to Track Together**

* **Authors:** Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht 
* **arXiv ID:** 2307.07635  
* **One-liner:** A transformer-based model that jointly tracks tens of thousands of 2D points across video frames rather than treating them independently  
* **Published in:** ECCV 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2307.07635) | [[PDF]](https://arxiv.org/pdf/2307.07635.pdf) | [[Code]](https://github.com/facebookresearch/co-tracker)

> **Core Innovation**  
> Introduces CoTracker: a Transformer-based architecture that jointly tracks thousands of 2D point trajectories, explicitly modeling inter-point dependencies to boost tracking accuracy for occluded or out-of-view points.

<details>
    <summary>Abstract</summary>
    We show that joint tracking significantly improves tracking accuracy and robustness, and allows CoTracker to track occluded points and points outside of the camera view. We introduce CoTracker, a transformer-based model that tracks a large number of 2D points in long video sequences, modeling dependencies among points. Technical innovations include “token proxies” for memory efficiency, enabling CoTracker to track ~70k points jointly on one GPU, online.  
</details>

<details>
    <summary>Key points</summary>
    * Joint point-tracking: tracks many points together, modeling correlations rather than independent tracking.  
    * Uses transformer architecture with token proxies to manage memory and scale to tens of thousands of points.  
    * Works online in causal short windows but is trained as an unrolled long-window recurrent to handle long sequences and occlusion.  
    * Significant improvements in robustness and accuracy over traditional independent point tracking methods, especially under occlusion and out-of-view.  
</details>

---

**GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control**

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

**OpenAI Sora**

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

---

**Runway Gen-3 Alpha**

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

---

**Pika 1.0**

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

---

### Embodied Intelligence Systems

Models designed for robotic control and autonomous agents that must integrate perception, spatial reasoning, and temporal prediction for real-world task execution.

**Key Characteristics:**
- Multimodal instruction following
- 3D spatial navigation and manipulation planning
- Predictive modeling of action consequences

#### Representative Works

**RT-2: Vision-Language-Action Models**

* **Authors:** Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, Brianna Zitkovich
* **arXiv ID:** 2307.15818
* **One-liner:** Trains a VLM on web-scale internet data + robot trajectories to map text + images directly to robotic actions for general-purpose real-world manipulation.
* **Published in:** CoRL 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.15818) | [[PDF]](https://arxiv.org/pdf/2307.15818.pdf) | [[Project Page]](https://robotics-transformer2.github.io/)

> **Core Innovation**  
> Extends large models from “see and talk” to “see, understand, and act,” pioneering the transfer of internet-scale vision-language knowledge to real-world robotic control, endowing robots with open-world reasoning and zero-shot manipulation capabilities.

<details>
    <summary>Abstract</summary>
    RT-2 is a vision-language-action (VLA) model built on large-scale internet and robotics data, enabling robots to learn general skills from web-scale knowledge. Unlike traditional robotic policies trained only on expert demonstrations, RT-2 leverages multimodal LLM pretraining and fine-tuning on robotic trajectories to map visual inputs and text instructions directly to robotic actions. RT-2 shows strong zero-shot generalization to real-world robotic tasks and novel concepts unseen during robot-specific training.
</details>

<details>
    <summary>Key points</summary>
    * Extends LLM/VLM foundation modeling to robot control
    * Trained on *internet vision-language data + robot action data*
    * Outputs robotic action tokens
    * Strong zero-shot generalization to new objects, tasks, environment changes
</details>

---

**GAIA-1: A Generative World Model for Autonomous Driving**

* **Authors:** Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado
* **arXiv ID:** 2311.07541
* **One-liner:** A large-scale generative world model that simulates diverse driving scenarios and predicts future multi-agent behavior for closed-loop autonomous driving.
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2309.17080) | [[PDF]](https://arxiv.org/pdf/2309.17080.pdf)

> **Core Innovation**  
> Beyond trajectory forecasting, it learns “world evolution” itself—simultaneously generating multi-agent dynamics, realistic traffic behaviors, and authentic sensor signals for closed-loop simulation and synthesis of rare safety-critical scenarios.

<details>
    <summary>Abstract</summary>
    GAIA-1 is a generative model for simulating complex real-world driving environments. Trained on large fleets of driving data, it learns a latent world representation enabling high-fidelity multi-agent behavior prediction and scene evolution. GAIA-1 can act as a self-supervised autonomous driving simulator for closed-loop evaluation and safety-critical scenario synthesis, improving planning and control systems without manual scenario engineering.
</details>

<details>
    <summary>Key points</summary>
    * Generative world model for driving — not just trajectory prediction
    * Generates future agent motion, scene geometry, and sensor data
    * Closed-loop simulation for AD evaluation + training
    * Enables rare / edge-case generation for safety
</details>

---

**PaLM-E: An Embodied Multimodal Language Model**

* **Authors:** Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence
* **arXiv ID:** 2303.03378
* **One-liner:** Multimodal LLM that integrates real-world sensor inputs (vision, robotics state) into PaLM, enabling robotic reasoning and action planning.
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2303.03378) | [[PDF]](https://arxiv.org/pdf/2303.03378.pdf) | [[Project Page]](https://palm-e.github.io/)

> **Core Innovation**  
> Treats robot vision and state information as “tokens” fed into an LLM, turning the language model into an embodied-intelligence decision system that can understand the environment, plan actions, and execute tasks.

<details>
    <summary>Abstract</summary>
    PaLM-E is an embodied multimodal LLM connecting large language models with robot perception and control inputs. It combines a transformer-based vision encoder with PaLM-type LLMs to ingest text, images, and robot state observations as tokens. The model outputs natural language, task descriptions, or robot commands. PaLM-E demonstrates strong transfer across tasks, allowing robots to execute long-horizon, real-world manipulation instructions and reason over multimodal sensory input.
</details>

<details>
    <summary>Key points</summary>
    * Embodied LLM combining robot sensor tokens + language
    * Joint vision-robot-language pretraining improves generalization
    * Handles long-horizon task reasoning and grounding
    * Builds toward unified *robotics-LLM world models*
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

**WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation**

* **Authors:** Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, Li Yuan
* **arXiv ID:** 2503.07265
* **One-liner:** Introduces WISE, a world-knowledge-aware evaluation framework that measures semantic alignment and common-sense correctness in text-to-image generation.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2503.07265) | [[PDF]](https://arxiv.org/pdf/2503.07265.pdf) | [[Code]](https://github.com/PKU-YuanGroup/WISE)

> **Core Innovation**  
> WISE argues that “evaluation must incorporate world knowledge” and constructs a text-to-image consistency assessment framework grounded in real-world semantic constraints, conceptual relations, and commonsense reasoning. This approach significantly improves measurement of text–image alignment and comprehension beyond mere CLIP similarity.

<details>
    <summary>Abstract</summary>
    Text-to-image model evaluation typically relies on metrics like CLIPScore, which often reward surface-level similarity rather than true semantic grounding. WISE is proposed as a semantic and world-knowledge-guided evaluation framework designed to assess whether generated images correctly reflect entity relations, attributes, and real-world facts. Experiments demonstrate significantly higher correlation with human judgements compared to existing metrics.
</details>

<details>
    <summary>Key points</summary>
    * Evaluates world knowledge and common-sense reasoning in T2I models  
    * Measures entity correctness, relational consistency, attributes, and scene realism  
    * Shows stronger correlation with human preference vs CLIPScore/BERTScore  
    * Designed as a standardized benchmark for semantic T2I evaluation  
</details>

---

**Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark**

* **Authors:** Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, Pheng-Ann Heng
* **arXiv ID:** 2510.26802
* **One-liner:** Proposes MME-COF benchmark to evaluate zero-shot video reasoning ability across memory, motion, events, causality, and forecasting — showing that current video models lag behind VLMs.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2510.26802) | [[PDF]](https://arxiv.org/pdf/2510.26802.pdf) | [[Code]](https://github.com/ZiyuGuo99/MME-CoF)

> **Core Innovation**  
> For the first time systematically evaluates “zero-shot reasoning” capability of large video models, introduces the MME-CoF benchmark (Memory, Motion, Event, Causality, Object, Future prediction), and reveals that video models lag markedly behind vision-language LLMs in fine-grained causal understanding.

<details>
    <summary>Abstract</summary>
    Although video models have advanced rapidly, their reasoning capabilities remain unclear. This work introduces the MME-COF benchmark that evaluates zero-shot video reasoning across six dimensions: object recognition, memory, motion understanding, event reasoning, causality, and future prediction. Experiments indicate that current state-of-the-art video models still significantly underperform vision-language models in causal and semantic reasoning, highlighting video reasoning as an open challenge.
</details>

<details>
    <summary>Key points</summary>
    * MME-COF benchmark: Memory, Motion, Event, Causality, Object, Future prediction  
    * Zero-shot evaluation — no task-specific video fine-tuning  
    * Reveals video models are weaker in causal & common-sense reasoning  
    * Shows video models rely on pattern recognition rather than deep inference  
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
