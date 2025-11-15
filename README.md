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

* CLIP: Learning Transferable Visual Models From Natural Language Supervision, ICML 2021
* DALL-E: Zero-Shot Text-to-Image Generation, ICML 2021
* Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, ICML 2015
* AttnGAN: Fine-Grained Text-to-Image Generation with Attentional GANs, CVPR 2018

### <a href="./consisency-paper/spatial-consistency/README.md">Spatial Consistency</a>

**Objective**: Enable models to understand and generate 3D spatial structure from 2D observations.

**Historical Significance**: Provided methodologies for constructing internal "3D scene graphs" and understanding geometric relationships.

#### Representative Works

* NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV 2020
* 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
* EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks, CVPR 2022
* Instant Neural Graphics Primitives with a Multiresolution Hash Encoding, SIGGRAPH 2022

### <a href="./consisency-paper/temporal-consistency/README.md">Temporal Consistency</a>

**Objective**: Model temporal dynamics, object motion, and causal relationships in video sequences.

**Historical Significance**: Early explorations of the world's "physics engine," capturing regularities in how scenes evolve over time.

#### Representative Works

* PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning, TPAMI 2023
* SimVP: Simpler yet Better Video Prediction, CVPR 2022
* Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning, CVPR 2023
* VideoGPT: Video Generation using VQ-VAE and Transformers, arXiv 2021
* Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions, ICLR 2023

## 🔗 Preliminary Integration: Unified Multimodal Models

Current state-of-the-art models are beginning to break down the barriers between individual consistencies. This section showcases models that successfully integrate **two** of the three fundamental consistencies, representing crucial intermediate steps toward complete world models.

### <a href="./consisency-paper/modality+spatial-consistency/README.md">Modality + Spatial Consistency</a>

**Capability Profile**: Models that can translate text/image descriptions into spatially coherent 3D representations or multi-view consistent outputs.

**Significance**: These models demonstrate "3D imagination" - they are no longer mere "2D painters" but "digital sculptors" understanding spatial structure.

#### Representative Works

* Zero-1-to-3: Zero-shot One Image to 3D Object, ICCV 2023
* MVDream: Multi-view Diffusion for 3D Generation, ICLR 2024
* Wonder3D: Single Image to 3D using Cross-Domain Diffusion, CVPR 2024
* SyncDreamer: Generating Multiview-consistent Images from a Single-view Image, ICLR 2024
* DreamFusion: Text-to-3D using 2D Diffusion, ICRL 2023
* ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding, CVPR 2024
* OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding, NeurIPS 2023
* DreamLLM: Synergistic Multimodal Comprehension and Creation, ICLR 2024
* EditWorld: Simulating World Dynamics for Instruction-Following Image Editing, ACM Multimedia 2025
* MIO: A Foundation Model on Multimodal Tokens, EMNLP 2025
* SGEdit: Bridging LLM with Text2Image Generative Model for Scene Graph-based Image Editing, SIGGRAPH Asia 2024
* UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics, CVPR 2025
* ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding, NeurIPS 2025
* Step1X-Edit: A Practical Framework for General Image Editing, arXiv 2025
* LangScene-X: Reconstruct Generalizable 3D Language-Embedded Scenes with TriMap Video Diffusion, ICCV 2025
* MENTOR: Efficient Multimodal-Conditioned Tuning for Autoregressive Vision Generation Models, arXiv 2025
* GSFixer: Improving 3D Gaussian Splatting with Reference-Guided Video Diffusion Priors, arXiv 2025
* CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields, CVPR 2022
* Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency, arXiv 2025
* Hunyuan3D-Omni: A Unified Framework for Controllable Generation of 3D Assets, arXiv 2025
* LERF: Language Embedded Radiance Fields, ICCV 2023
* Viewset Diffusion: (0-)Image-Conditioned 3D Generative Models from 2D Data, ICCV 2023
* Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation, arXiv 2025
* Scaling Transformer-Based Novel View Synthesis Models with Token Disentanglement and Synthetic Data, ICCV 2025
* NeRF-HuGS: Improved Neural Radiance Fields in Non-static Scenes Using Heuristics-Guided Segmentation, CVPR 2024
* RealFusion: 360° Reconstruction of Any Object from a Single Image, CVPR 2023

### <a href="./consisency-paper/modality+temporal-consistency/README.md">Modality + Temporal Consistency</a>

**Capability Profile**: Models that transform textual descriptions or static images into temporally coherent, dynamic video sequences.

**Significance**: Currently the most prominent integration direction, enabling high-quality text-to-video and image-to-video generation.

#### Representative Works

* Lumiere: A Space-Time Diffusion Model for Video Generation, SIGGRAPH-ASIA 2024
* Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets, arXiv 2023
* AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning, arXiv 2023
* Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning, ECCV 2024
* VideoPoet: A Large Language Model for Zero-Shot Video Generation, ICML 2024

### <a href="./consisency-paper/spatial-temporal-consistency/README.md">Spatial + Temporal Consistency</a>

Capability Profile: Models that maintain 3D spatial structure while simulating temporal dynamics, but may have limited language understanding or controllability.

Significance: These models represent crucial technical achievements in understanding "how the 3D world moves," forming the physics engine component of world models.

#### Representative Works

* DUSt3R: Geometric 3D Vision Made Easy, CVPR 2024
* 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering, CVPR 2024
* Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes, CVPR 2021
* CoTracker: It is Better to Track Together, ECCV 2024
* GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, 2024 (open release)

## 🌟 The "Trinity" Prototype: Emerging World Models

This section highlights models that demonstrate **preliminary integration of all three consistencies**, exhibiting emergent world model capabilities. These systems represent the current frontier, showing glimpses of true world simulation.

### <a href="./consisency-paper/world-models/README.md">Text-to-World Generators</a>

Models that generate dynamic, spatially consistent virtual environments from language descriptions.

**Key Characteristics:**
- ✅ Modality: Natural language understanding and pixel-space generation
- ✅ Spatial: 3D-aware scene composition with object permanence
- ✅ Temporal: Physically plausible dynamics and motion

#### Representative Works

* Runway Gen-3 Alpha, 2024 (Alpha)
* Pika 1.0, 2023 (November)

### <a href="./consisency-paper/embodied-intelligence-systems/README.md">Embodied Intelligence Systems</a>

Models designed for robotic control and autonomous agents that must integrate perception, spatial reasoning, and temporal prediction for real-world task execution.

**Key Characteristics:**
- Multimodal instruction following
- 3D spatial navigation and manipulation planning
- Predictive modeling of action consequences

#### Representative Works

* RT-2: Vision-Language-Action Models, CoRL 2023
* GAIA-1: A Generative World Model for Autonomous Driving, arXiv 2023
* PaLM-E: An Embodied Multimodal Language Model, ICLR 2024

## 📊 <a href="./consisency-paper/benchmarks+evaluation/README.md">Benchmarks and Evaluation</a>

**Current Challenge**: Existing metrics (FID, FVD, CLIP Score) inadequately assess world model capabilities, focusing on perceptual quality rather than physical understanding.

**Need for Comprehensive Benchmarks:**

A true world model benchmark should evaluate:
- 🧩 **Commonsense Physics Understanding**: Does the model respect gravity, momentum, conservation laws?
- 🔮 **Counterfactual Reasoning**: Can it predict outcomes of hypothetical interventions?
- ⏳ **Long-term Consistency**: Does coherence break down over extended simulation horizons?
- 🎯 **Goal-Directed Planning**: Can it chain actions to achieve complex objectives?
- 🎛️ **Controllability**: How precisely can users manipulate simulated elements?

#### Representative Works

* WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation, arXiv 2025
* Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark, arXiv 2025

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






