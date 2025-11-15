# Awesome World Model Evolution - 铸向世界模型宇宙：基于统一多模态模型的融合之路 [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<div align="center">
  <img src="https://img.shields.io/github/stars/opendatalab-raiser/awesome-world-model-evolution" alt="Stars">
  <img src="https://img.shields.io/github/forks/opendatalab-raiser/awesome-world-model-evolution" alt="Forks">
  <img src="https://img.shields.io/github/license/opendatalab-raiser/awesome-world-model-evolution" alt="License">
  <img src="https://img.shields.io/github/last-commit/opendatalab-raiser/awesome-world-model-evolution" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</div>

<p align="center">
  <strong>精心收集的研究论文、模型和资源合集，追溯从专用模型到统一世界模型的演化历程。</strong>
</p>

---

## 📋 目录

- [Awesome World Model Evolution - 铸向世界模型宇宙：基于统一多模态模型的融合之路 ](#awesome-world-model-evolution---铸向世界模型宇宙基于统一多模态模型的融合之路-)
  - [📋 目录](#-目录)
  - [🎯 引言](#-引言)
    - [什么是世界模型？](#什么是世界模型)
    - [三大基础一致性](#三大基础一致性)
      - [1️⃣ **模态一致性（Modality Consistency）**](#1️⃣-模态一致性modality-consistency)
      - [2️⃣ **空间一致性（Spatial Consistency）**](#2️⃣-空间一致性spatial-consistency)
      - [3️⃣ **时间一致性（Temporal Consistency）**](#3️⃣-时间一致性temporal-consistency)
    - [为何选择统一多模态模型？](#为何选择统一多模态模型)
  - [🔬 独立探索：专用模型](#-独立探索专用模型)
    - [模态一致性](#模态一致性)
      - [代表性工作](#代表性工作)
    - [空间一致性](#空间一致性)
      - [代表性工作](#代表性工作-1)
    - [时间一致性](#时间一致性)
      - [代表性工作](#代表性工作-2)
  - [🔗 初步融合：统一多模态模型](#-初步融合统一多模态模型)
    - [模态 + 空间一致性](#模态--空间一致性)
      - [代表性工作](#代表性工作-3)
    - [模态 + 时间一致性](#模态--时间一致性)
      - [代表性工作](#代表性工作-4)
    - [空间一致性 + 时间一致性](#空间一致性--时间一致性)
      - [代表性工作](#代表性工作-5)
  - [🌟 "三位一体"原型：涌现的世界模型](#-三位一体原型涌现的世界模型)
    - [文本到世界生成器](#文本到世界生成器)
      - [代表性工作](#代表性工作-6)
    - [具身智能系统](#具身智能系统)
      - [代表性工作](#代表性工作-7)
  - [📊 基准测试与评估](#-基准测试与评估)
      - [代表性工作](#代表性工作-8)
  - [📝 贡献指南](#-贡献指南)
    - [如何贡献](#如何贡献)
    - [条目格式模板](#条目格式模板)
  - [⭐ Star History](#-star-history)
  - [📄 License](#-license)

---

## 🎯 引言

### 什么是世界模型？

**世界模型（World Model）** 是一种计算系统，能够学习支配我们物理世界的内在规律，并在内部构建一个可执行的、动态的模拟环境。与仅执行模式匹配或分类的传统 AI 模型不同，世界模型作为一个**内部模拟器**，具备以下能力：

- 🔄 **复现**已观察到的事件和现象
- 🎲 **预测**未来状态和结果
- 🤔 **推理**反事实场景（假设分析）
- 🎯 **规划**基于模拟后果的长期策略

**世界模型的重要性：**

世界模型被视为实现**通用人工智能（AGI）** 的基石：

- 🤖 **具身智能**：机器人可以在执行前"预演"动作
- 🚗 **自动驾驶系统**：模拟危险边缘场景以实现更安全的部署
- 🔬 **科学发现**：通过数字实验加速对复杂系统的理解
- 🎮 **交互式环境**：创建可控的、符合物理规律的虚拟世界

### 三大基础一致性

一个功能完备的世界模型必须掌握三项核心能力，我们称之为**三大基础一致性**：

#### 1️⃣ **模态一致性（Modality Consistency）**
模型与现实世界之间的"语言接口"。

- **能力**：在高维感官输入（图像、视频、音频）与抽象符号表示（语言、结构化数据）之间进行双向转换
- **示例任务**：图像描述生成、文本到图像生成、视觉问答
- **意义**：使世界模型能够接收指令并以人类可理解的格式交流观察结果

#### 2️⃣ **空间一致性（Spatial Consistency）**
对物理世界的"静态三维理解"。

- **能力**：理解物体具有固定的几何形态、占据空间、存在遮挡关系，并在不同视角下保持身份
- **示例任务**：新视角合成、三维重建、多视角一致性
- **意义**：构建支持精确空间推理和导航的基础"场景图"

#### 3️⃣ **时间一致性（Temporal Consistency）**
用于动态模拟的"物理引擎"。

- **能力**：建模世界如何随时间演化，包括物体运动、物理交互和因果事件链
- **示例任务**：视频预测、动力学建模、未来状态预测
- **意义**：实现后果预判和长期规划能力

### 为何选择统一多模态模型？

本仓库通过**统一多模态模型**（特别是大型多模态模型 - LMMs）追溯世界模型的演化路径，我们认为这代表了最有前景的发展方向：

**核心优势：**

- 🏗️ **架构统一性**：中央处理核心（LLM 主干）配合模块化感知接口，自然促进跨模态整合
- ✨ **涌现式理解**：在海量、多样化的多模态数据集上预训练，使模型能够隐式学习世界规律，而非依赖人工设计的规则
- 📈 **可扩展性**：已验证的缩放定律表明，能力随参数、数据和计算量的增加而可预见地提升
- 🔄 **协同学习**：不同的一致性能力可以通过联合训练相互涌现和强化

**我们的论点：** 端到端训练的统一模型优于特定任务的专用模型来构建世界模型，因为它们能够学习模态、空间和时间之间的深层关联。

---

## 🔬 独立探索：专用模型

在统一模型时代之前，研究人员采用"分而治之"的方法，针对每个一致性挑战开发专门的架构。这些奠基性工作建立了关键技术和洞见，为当前的统一方法提供了基础。

### <a href="./consisency-paper/modality-consistency/README_zh.md">模态一致性</a>

**目标**：在符号（语言）和感知（视觉）表示之间建立双向映射。

**历史意义**：这些模型创建了首批"符号-感知桥梁"，解决了世界模型的基本输入/输出问题。

#### 代表性工作

* CLIP: Learning Transferable Visual Models From Natural Language Supervision, ICML 2021
* DALL-E: Zero-Shot Text-to-Image Generation, ICML 2021
* Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, ICML 2015
* AttnGAN: Fine-Grained Text-to-Image Generation with Attentional GANs, CVPR 2018
* NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV 2020

### <a href="./consisency-paper/spatial-consistency/README_zh.md">空间一致性</a>

**目标**：使模型能够从二维观察中理解和生成三维空间结构。

**历史意义**：为构建内部"3D 场景图"和理解几何关系提供了方法论。

#### 代表性工作

* NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV 2020
* 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
* EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks, CVPR 2022
* Instant Neural Graphics Primitives with a Multiresolution Hash Encoding, SIGGRAPH 2022

### <a href="./consisency-paper/temporal-consistency/README_zh.md">时间一致性</a>

**目标**：建模视频序列中的时间动态、物体运动和因果关系。

**历史意义**：对世界"物理引擎"的早期探索，捕捉场景随时间演化的规律。

#### 代表性工作

* PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning, TPAMI 2023
* SimVP: Simpler yet Better Video Prediction, CVPR 2022
* Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning, CVPR 2023
* VideoGPT: Video Generation using VQ-VAE and Transformers, arXiv 2021
* Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions, ICLR 2023

## 🔗 初步融合：统一多模态模型

当前最先进的模型正开始打破各个一致性之间的壁垒。本节展示成功整合**三大一致性中的两项**的模型，它们代表了通向完整世界模型的关键中间步骤。

### <a href="./consisency-paper/modality+spatial-consistency/README_zh.md">模态 + 空间一致性</a>

**能力特征**：能够将文本/图像描述转换为空间连贯的 3D 表示或多视角一致输出的模型。

**意义**：这些模型展示了"3D 想象力"——它们不再是简单的"2D 画家"，而是理解空间结构的"数字雕塑家"。

#### 代表性工作

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

  
### <a href="./consisency-paper/modality+temporal-consistency/README_zh.md">模态 + 时间一致性</a>

**能力特征**：将文本描述或静态图像转换为时间连贯的动态视频序列的模型。

**意义**：目前最突出的融合方向，实现高质量的文本到视频和图像到视频生成。

#### 代表性工作

* Lumiere: A Space-Time Diffusion Model for Video Generation, SIGGRAPH-ASIA 2024
* Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets, arXiv 2023
* AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning, arXiv 2023
* Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning, ECCV 2024
* VideoPoet: A Large Language Model for Zero-Shot Video Generation, ICML 2024

### <a href="./consisency-paper/spatial-temporal-consistency/README_zh.md">空间一致性 + 时间一致性</a>

**能力特征**：这类模型能够在模拟时间动态演化的同时保持三维空间结构的一致性，但可能在语言理解或可控性方面存在一定局限。

**意义**：这些模型代表了理解"三维世界如何运动"的关键技术成就，构成了世界模型的物理引擎组成部分。

#### 代表性工作

* DUSt3R: Geometric 3D Vision Made Easy, CVPR 2024
* 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering, CVPR 2024
* Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes, CVPR 2021
* CoTracker: It is Better to Track Together, ECCV 2024
* GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control, 2024 (open release)

## 🌟 "三位一体"原型：涌现的世界模型

本节重点介绍展示**三大一致性初步整合**的模型，表现出涌现的世界模型能力。这些系统代表了当前前沿，展现了真正世界模拟的雏形。

### <a href="./consisency-paper/world-models/README_zh.md">文本到世界生成器</a>

从语言描述生成动态、空间一致的虚拟环境的模型。

**关键特征：**
- ✅ 模态一致性：自然语言理解和像素空间生成
- ✅ 空间一致性：具有物体恒存性的 3D 感知场景组合
- ✅ 时间一致性：物理上可信的动力学和运动

#### 代表性工作

* Runway Gen-3 Alpha, 2024 (Alpha)
* Pika 1.0, 2023 (November)

### <a href="./consisency-paper/embodied-intelligence-systems/README_zh.md">具身智能系统</a>

为机器人控制和自主智能体设计的模型，必须整合感知、空间推理和时间预测以执行真实世界任务。

**关键特征：**
- 多模态指令遵循
- 3D 空间导航和操作规划
- 动作后果的预测建模

#### 代表性工作

<details>
<summary><b>RT-2: Vision-Language-Action Models</b></summary>

* **Authors:** Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, Brianna Zitkovich
* **arXiv ID:** 2307.15818
* **One-liner:** 将大规模网络 VLM 与机器人动作训练结合，将视觉与语言直接转化为真实机器人动作，实现零样本现实操控。
* **Published in:** CoRL 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.15818) | [[PDF]](https://arxiv.org/pdf/2307.15818.pdf) | [[Project Page]](https://robotics-transformer2.github.io/)

> **核心创新**  
> 将大模型从“看图 + 说话”扩展为“看图 + 理解 + 动作执行”，首次将互联网视觉-语言知识迁移到真实机器人动作层面，使机器人具备开放世界推理和零样本动作能力。

* RT-2: Vision-Language-Action Models, CoRL 2023
* GAIA-1: A Generative World Model for Autonomous Driving, arXiv 2023
* PaLM-E: An Embodied Multimodal Language Model, ICLR 2024

## 📊 <a href="./consisency-paper/benchmarks+evaluation/README_zh.md">基准测试与评估</a>

**当前挑战**：现有指标（FID、FVD、CLIP Score）无法充分评估世界模型能力，侧重于感知质量而非物理理解。

**综合基准测试的需求：**

真正的世界模型基准应评估：
- 🧩 **常识物理理解**：模型是否遵守重力、动量、守恒定律？
- 🔮 **反事实推理**：能否预测假设干预的结果？
- ⏳ **长期一致性**：在扩展的模拟时间范围内连贯性是否会崩溃？
- 🎯 **目标导向规划**：能否链接动作以实现复杂目标？
- 🎛️ **可控性**：用户能多精确地操控模拟元素？

#### 代表性工作

* WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation, arXiv 2025
* Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark, arXiv 2025

## 📝 贡献指南

我们欢迎各种形式的贡献！包括但不限于：

- 🆕 添加新论文、工具或数据集
- 📝 改进现有条目的描述
- 🐛 修复错误或过时信息
- 💡 提出改进建议

### 如何贡献

1. **Fork** 本仓库
2. **创建**特性分支 (`git checkout -b feature/AmazingPaper`)
3. **提交**更改 (`git commit -m 'Add Paper: Amazing World Model'`)
4. **推送**到分支 (`git push origin feature/AmazingPaper`)
5. **开启** Pull Request

### 条目格式模板

添加论文时，请使用以下格式：

**[论文标题]**

* **作者：** [作者列表]
* **发表于：** [会议/期刊] [年份]
* **arXiv ID:** [arXiv 编号（如适用）]
* **核心创新：** [1-2 句话概述]
* **链接：** [[Paper]](url) | [[Code]](url) | [[Project Page]](url)

<details>
<summary>Abstract</summary>
[官方摘要]
</details>

<details>
<summary>Key points</summary>
[要点列举主要贡献]
</details>

---

## ⭐ Star History

如果这个项目对您的研究或理解有帮助，请给我们一个 Star ⭐️！

[![Star History Chart](https://api.star-history.com/svg?repos=opendatalab-raiser/awesome-world-model-evolution&type=Date)](https://star-history.com/#opendatalab-raiser/awesome-world-model-evolution&Date)

---

## 📄 License

This project is licensed under [MIT License](LICENSE).
