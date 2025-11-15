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

<details>
<summary><b>NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis</b></summary>

* **Authors:** Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
* **arXiv ID:** 2003.08934  
* **One-liner:** Represent a scene as a continuous volumetric neural radiance field and render novel views via volume rendering of the MLP output  
* **Published in:** ECCV 2020
* **Links:** [[Paper]](https://arxiv.org/abs/2003.08934) | [[PDF]](https://arxiv.org/pdf/2003.08934.pdf)  

> **核心创新**  
> 将场景用一个 MLP 表示为“神经辐射场”（Neural Radiance Field, NeRF）：输入任意 5D 坐标（x,y,z,θ,φ）输出该位置的体密度 + 视角依赖的辐射量，再通过可微体渲染 (volume rendering) 实现从新视角生成图像。  

<details>
    <summary>Abstract</summary>
    我们通过仅用稀疏输入视图优化底层连续体积场景函数，实现了复杂场景新视角合成的最先进效果：用一个全连接深度网络把单条5维坐标直接映射到该点的体积密度和视角相关辐射，再沿相机射线查询并借助经典体渲染把颜色和密度投影成图像，凭借体渲染的可微性只需已知相机姿态的图像即可优化，从而渲染出几何与外观皆逼真的新视图并在神经渲染与视角合成任务上超越先前工作，建议观看补充视频以获得更直观的对比体验。 
</details>

<details>
    <summary>Key points</summary>
    * 使用 MLP 来表示场景，输入 (x,y,z,θ,φ)，输出体密度 + 辐射。  
    * 通过体渲染 (volume rendering) 将网络输出合成为图像。  
    * 无需构建显式几何重建，只用多视角图像和相机位姿即可优化。  
    * 展示出高质量的新视角合成效果，推动了视图合成和隐式表示的发展。  
</details>
</details>

---

<details>
<summary><b>3D Gaussian Splatting for Real-Time Radiance Field Rendering</b></summary>

* **Authors:** Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
* **arXiv ID:** 2308.04079
* **One-liner:** Use anisotropic 3D Gaussians + splatting renderer to achieve real-time (>=30 fps) high-quality novel-view synthesis of large scenes.  
* **Published in:** SIGGRAPH 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2308.04079) | [[PDF]](https://arxiv.org/pdf/2308.04079.pdf) | [[Code]](https://github.com/graphdeco-inria/gaussian-splatting)

> **核心创新**
> 提出用成千上万的各向异性 3D 高斯分布（3D Gaussians）而非纯 MLP 表征场景，再结合一个可视性感知的高效 splatting 渲染器，实现大场景下的实时 (1080p, >=30fps) 新视角合成。  

<details>
    <summary>Abstract</summary>
    辐射场方法最近彻底革新了用多张照片或视频捕获场景的新视角合成，然而要获得高视觉质量仍需昂贵训练和渲染的神经网络，而近期更快的方法难免在速度和质量间折中，对于无界完整场景而非孤立物体且需1080p分辨率渲染，尚无方法能实现实时显示，我们引入三个关键要素——从相机校准产生的稀疏点出发用保留连续体积辐射场优良特性又避免空区域不必要计算的3D高斯表示场景、对3D高斯进行交错优化/密度控制特别是优化各向异性协方差以精确表达场景、开发支持各向异性splatting且加速训练并支持实时渲染的快速可见性感知渲染算法——在保持有竞争力训练时间的同时实现最先进视觉质量并首次在1080p分辨率下达成≥30 fps的实时新视角合成，在多个基准数据集上验证了最先进视觉质量与实时渲染能力。
</details>

<details>
    <summary>Key points</summary>
    * 使用 3D 高斯而非传统隐式 MLP 表示场景，避开空白空间浪费。  
    * 对每个高斯分布使用各向异性协方差优化，提高场景表达精度。  
    * 引入可视性感知的 splatting 渲染器，实现实时新视角合成。  
    * 在大场景、1080p 分辨率下实现 ≥30 fps 的实时渲染，并且质量与此前高质量方法可比。  
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

> **核心创新**  
> 引入基于三平面 (tri-plane) 的混合显式‐隐式网络架构，使 3D GAN 既具备高效计算、又能生成高分辨率、多视角一致性的图像和优质几何形状。

<details>
    <summary>Abstract</summary>
    仅利用单张2D照片集合进行无监督生成高质量多视角一致图像与3D形状一直是长期难题，现有3D GAN要么计算密集要么采用非3D一致近似，前者限制生成图像质量与分辨率后者损害多视角一致性和形状质量，本文通过引入表达力强的混合显隐式网络架构并配合其他设计选择，在不过度依赖这些近似的情况下提升3D GAN计算效率与图像质量，实时合成高分辨率多视角一致图像并产出高质量3D几何，且通过解耦特征生成与神经渲染得以利用StyleGAN2等最先进2D CNN生成器并继承其高效性与表现力，在FFHQ和AFHQ Cats等实验上展示了最先进的3D感知合成效果。
</details>

<details>
    <summary>Key points</summary>
    * 三平面 (tri-plane) 表示：将场景编码到三组 2D 隐式特征平面，从而高效查询 3D 特征。  
    * 解耦特征生成与神经渲染，允许使用强大的 2D 生成器架构。  
    * 实时生成高分辨率且多视角一致的图像，并产出优质几何。  
    * 提升 3D GAN 的效率与视觉质量，推动 3D 世界生成研究。  
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

> **核心创新**  
> 提出了一个多分辨率哈希编码 (multiresolution hash encoding)，将坐标映射到可训练特征向量并与小型网络联合使用，从而在保持质量的同时，大幅降低 FLOPs 和内存访问，从而实现“秒级训练”、毫秒渲染的神经图形原语。  

<details>
    <summary>Abstract</summary>
    我们用一种通用的新输入编码——多分辨率可训练特征向量哈希表——给小型全连接神经网络外挂“字典”，既保质量又巨幅削减浮点与访存开销，哈希冲突靠多分辨率结构自动消解，架构极简且在现代GPU上全并行化，通过全融合CUDA内核进一步压缩带宽与计算浪费，最终实现数秒训完、1920×1080分辨率下数十毫秒绘出的神经图形基元，速度提升数个量级。
</details>

<details>
    <summary>Key points</summary>
    * 多分辨率哈希编码：将坐标映射到多个尺度的哈希表特征向量。  
    * 小型神经网络 + 哈希特征表：保持高质量同时大幅降低计算与内存访问。  
    * 完全 GPU 并行化（融合 CUDA 核心）以最大化效率。  
    * 实现“瞬时”训练与极快推理（神经渲染毫秒级），推动隐式表示与神经渲染实时化。  
</details>
</details>

---

### <a href="./consisency-paper/temporal-consistency/README_zh.md">时间一致性</a>

**目标**：建模视频序列中的时间动态、物体运动和因果关系。

**历史意义**：对世界"物理引擎"的早期探索，捕捉场景随时间演化的规律。

#### 代表性工作

<details>
<summary><b>PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning</b></summary>

* **Authors:** Yunbo Wang, Haixu Wu, Jianjin Zhang, Zhifeng Gao, Jianmin Wang, Philip S. Yu, Mingsheng Long
* **arXiv ID:** 2103.09504  
* **One-liner:** 新的循环结构解耦时空记忆单元，用于视频帧的时空预测学习  
* **Published in:** TPAMI 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2103.09504) | [[PDF]](https://arxiv.org/pdf/2103.09504.pdf) | [[Code]](https://github.com/thuml/predrnn-pytorch)

> **核心创新**  
> 提出将记忆单元对(decoupled memory cells)融入 RNN 架构，以分离地捕捉空间外观与时间动态，再统一形成复杂环境表示。此外引入“之”字形（zig-zag）记忆流上下层传播，以及课程化学习策略以学习长期时动态。 

<details>
    <summary>Abstract</summary>
    PredRNN通过一对显式解耦、几乎独立过渡并最终统一表征复杂环境的记忆单元，以及贯穿所有层自下而上又自上而下的之字形记忆流和防止冗余的记忆解耦损失，建模时空序列预测学习中被认为可由组合子系统习得的模块化视觉动态，并辅以迫使模型从上下文帧学习长期依赖的课程策略，在五个数据集的无动作与给定动作预测场景均取得极具竞争力的结果。
</details>

<details>
    <summary>Key points</summary>
    * 引入记忆对结构：两记忆单元解耦转移分别捕捉空间外观与时间动态。   
    * zig-zag 记忆流：跨层上下传播，促进不同时间层次动态通信。  
    * 课程化学习策略（curriculum learning）：从上下文帧学习长期预测。  
    * 在多个基准数据集上取得竞争性预测性能。  
</details>
</details>

---

<details>
<summary><b>SimVP: Simpler yet Better Video Prediction</b></summary>

* **Authors:** Zhangyang Gao, Cheng Tan, Lirong Wu, Stan Z. Li
* **arXiv ID:** 2206.05099  
* **One-liner:** 完全基于 CNN 的视频预测模型，少即是多，简而优于复杂结构  
* **Published in:** CVPR 2022  
* **Links:** [[Paper]](https://arxiv.org/abs/2206.05099) | [[PDF]](https://arxiv.org/pdf/2206.05099.pdf) | [[Code]](https://github.com/A4Bio/SimVP)

> **核心创新**  
> 提出一个极简的端到端视频预测框架，仅用 CNN 编码—解码结构，训练损失为 MSE，无需 RNN、Transformer、复杂机制即可在多个基准数据集上达到或超越更复杂模型。 

<details>
    <summary>Abstract</summary>
    从CNN、RNN到ViT，视频预测领域借助辅助输入、精巧架构与复杂训练策略不断取得突破，而我们却困惑其必要性：是否存在同样高效的极简方案？本文提出SimVP，一个仅用CNN、仅用MSE损失端到端训练的简单模型，无需任何额外技巧便在五个基准数据集上达到SOTA，扩展实验显示其在真实场景具强泛化与可扩展性，训练成本大幅降低，易于扩展到复杂场景，可作为坚实基线推动视频预测进一步发展。 
</details>

<details>
    <summary>Key points</summary>
    * 完全用卷积编码器–解码器架构（无 RNN、无 Transformer）
    * 用标准 MSE 训练，无额外复杂策略。  
    * 显示出更简单结构也能实现竞争/领先性能。  
    * 降低架构复杂度与计算开销，更适用于实际场景。  
</details>
</details>

---

<details>
<summary><b>Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning</b></summary>

* **Authors:** Cheng Tan, Zhangyang Gao, Lirong Wu, Yongjie Xu, Jun Xia, Siyuan Li, Stan Z. Li
* **arXiv ID:** 2206.12126  
* **One-liner:** 引入可并行的时域注意力模块（TAU），分离帧内／帧间注意力以提升时空预测效率  
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2206.12126) | [[PDF]](https://arxiv.org/pdf/2206.12126.pdf)

> **核心创新**  
> 提出 Temporal Attention Unit (TAU)，将时空预测模块中的“帧内静态注意力”与“帧间动态注意力”分离，支持并行化计算；并且通过差分散度正则化(differential divergence regularization)增强帧间变化捕捉。  

<details>
    <summary>Abstract</summary>
    时空预测学习旨在通过历史帧生成未来帧，本文在回顾现有方法后提出通用框架：由空间编码器-解码器提取帧内特征，中间并行化时间模块捕捉帧间关联；针对主流循环单元因串行而计算低效的问题，我们设计 Temporal Attention Unit（TAU），将时间注意力分解为帧内静态与帧间动态两部分，并配套差分散度正则化显式建模帧间变化，仅使用平方误差关注帧内误差，最终在多个基准数据集上取得有竞争力的预测性能。
</details>

<details>
    <summary>Key points</summary>
    * 提出 TAU：将时域注意力分为帧内静态与帧间动态两部分，可并行执行。  
    * 引入差分散度正则（differential divergence regularisation）以增强帧间变化捕捉。  
    * 构建空间编码–解码模块 + 时间模块的解耦架构，提升可扩展性。  
    * 显著提高计算效率，同时在预测准确性上保持竞争水平。  
</details>
</details>

---

<details>
<summary><b>VideoGPT: Video Generation using VQ-VAE and Transformers</b></summary>

* **Authors:** Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas
* **arXiv ID:** 2104.10157
* **One-liner:** 使用 VQ-VAE + GPT 风格 Transformer 将图像生成范式扩展至自然视频生成  
* **Published in:** arXiv 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2104.10157) | [[PDF]](https://arxiv.org/pdf/2104.10157.pdf) | [[Code]](https://github.com/wilson1yan/VideoGPT)

> **核心创新**  
> 将 VQ-VAE 用于视频离散潜变量学习（3D 卷积 + 轴向自注意力），然后用 GPT 式 Transformer 在这些离散潜变量上自回归建模，实现在自然视频上基于似然的生成。  

<details>
    <summary>Abstract</summary>
    我们提出 VideoGPT：一个概念简单的架构，用于将基于似然的生成建模扩展至自然视频。VideoGPT 采用 VQ-VAE 来通过 3D 卷积与轴向自注意力学习原始视频的下采样离散潜变量。然后，一个类似 GPT 的架构在这些离散潜变量上自回归地进行建模，并使用时空位置编码。尽管结构简洁且训练方便，我们的模型在 BAIR Robot 数据集上生成质量与先进的 GAN 模型竞争，并能从 UCF-101 和 TGIF 等数据集生成自然视频。我们希望所提出架构可作为 Transformer 基础的视频生成模型的可复现参考。  
</details>

<details>
    <summary>Key points</summary>
    * 使用 VQ-VAE 学习视频的离散潜变量（3D 卷积 + 轴向自注意力）  
    * 使用 GPT 风格 Transformer 在这些离散潜变量上进行自回归生成，并加入时空位置编码  
    * 与 GAN 方法相比，结构更简单但生成质量具有竞争力  
    * 提供一个可复现、基于 Transformer 的自然视频生成基准  
</details>
</details>

---

<details>
<summary><b>Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions</b></summary>

* **Authors:** Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, Dumitru Erhan
* **arXiv ID:** 2210.02399  
* **One-liner:** 从开放域文本提示生成可变长度视频：文本→离散视频 token →解码为帧  
* **Published in:** ICLR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2210.02399) | [[PDF]](https://arxiv.org/pdf/2210.02399.pdf)

> **核心创新**  
> 提出将视频表示离散化为 token 的 tokenizer（支持可变长度视频）、并用条件遮蔽 Transformer 生成这些 token，再解码生成帧。通过联合训练大规模图文数据和少量视频-文对，实现开放域文本提示下任意长度视频生成。  

<details>
    <summary>Abstract</summary>
    我们提出 Phenaki，这一模型可在仅输入一串文本提示的条件下合成逼真且任意时长的视频：通过引入一种以因果时间注意力压缩视频为离散 token 的紧凑表示来应对可变长度与计算开销，再用基于预计算文本 token 的双向掩码变换器自文本生成这些视频 token 并最终解码成画面；同时利用大规模图文对与小规模视频文本对联合训练突破数据稀缺限制，首次实现按时间变化提示（故事）在开放域生成连贯长视频，且相比逐帧基线token更少却具备更优时空一致性。
</details>

<details>
    <summary>Key points</summary>
    * 将视频表示离散化为 token，实现可变长度视频输出。  
    * 使用条件遮蔽 Transformer 在文本 token 条件下生成视频 token，然后解码为帧。  
    * 联合训练图文大数据与视频-文本少数据以提升泛化。  
    * 支持开放域文本提示生成任意长度视频，而不仅限于固定短片。  
</details>
</details>

---

## 🔗 初步融合：统一多模态模型

当前最先进的模型正开始打破各个一致性之间的壁垒。本节展示成功整合**三大一致性中的两项**的模型，它们代表了通向完整世界模型的关键中间步骤。

### <a href="./consisency-paper/modality+spatial-consistency/README_zh.md">模态 + 空间一致性</a>

**能力特征**：能够将文本/图像描述转换为空间连贯的 3D 表示或多视角一致输出的模型。

**意义**：这些模型展示了"3D 想象力"——它们不再是简单的"2D 画家"，而是理解空间结构的"数字雕塑家"。

#### 代表性工作

<details>
<summary><b>Zero-1-to-3: Zero-shot One Image to 3D Object</b></summary>

* **Authors:** Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, Carl Vondrick
* **arXiv ID:** 2303.11328  
* **One-liner:** 给定单张 RGB 图像，实现零样本视点变化与隐式 3D 重建  
* **Published in:** ICCV 2023  
* **Links:** [[Paper]](https://arxiv.org/abs/2303.11328) | [[PDF]](https://arxiv.org/pdf/2303.11328.pdf) | [[Code]](https://github.com/cvlab-columbia/zero123)

> **核心创新**  
> 利用大规模预训练扩散模型所隐含的几何先验，仅从一张图像学习控制摄像机视点变化，从而生成同一物体在指定变换下的新视角图像，并可进一步进行单张图像的隐式3D重建。

<details>
    <summary>Abstract</summary>
    我们介绍了 Zero-1-to-3 框架，用于仅给定一张 RGB 图像的物体，改变其摄像机视点。为在该欠定设置下执行新视角合成，我们利用大规模扩散模型从自然图像中学习到的几何先验。我们的条件扩散模型使用合成数据集来学习相对摄像机视点的控制，从而允许在指定的摄像机变换下生成同一物体的新图像。尽管模型在合成数据集上训练，它仍然拥有强大的零样本泛化能力，适用于分布外数据集甚至野外图像（包括印象派画作）。我们的视点条件扩散方法还可用于单图像的 3D 重建任务。定性与定量实验表明，本方法通过利用互联网规模预训练显著优于当前单视图 3D 重建与新视角合成模型。
</details>

<details>
    <summary>Key points</summary>
    * 利用大规模预训练的 2D 扩散模型提取隐含的 3D/几何先验。  
    * 条件生成：输入单张 RGB 图像 + 相对摄像机变换（旋转+平移）生成新视角。  
    * 实现强零样本泛化至分布外真实图像和艺术风格图片。  
    * 支持从单张图像实现新视角合成与隐式 3D 重建。  
</details>
</details>

---

<details>
<summary><b>MVDream: Multi-view Diffusion for 3D Generation</b></summary>

* **Authors:** Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, Xiao Yang
* **arXiv ID:** 2308.16512  
* **One-liner:** 一个多视角扩散模型：从文本提示生成几何一致的多视角图像，从而用于 3D 生成  
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2308.16512) | [[PDF]](https://arxiv.org/pdf/2308.16512.pdf) | [[Code]](https://github.com/bytedance/MVDream)

> **核心创新**  
> 提出一个可生成几何一致多视角图像的扩散模型，通过结合2D图像预训练和3D数据微调，使其既具有 2D 扩散模型的泛化能力 又具有 3D 渲染的一致性，从而可作为 3D 内容生成的通用先验。

<details>
    <summary>Abstract</summary>
    我们提出 MVDream，一种扩散模型，能够根据给定文本提示生成几何一致的多视角图像。通过同时学习2D与3D数据兼具2D扩散模型的泛化性和3D渲染的一致性，隐式地成为与3D表示无关的可泛化3D先验，可经分数蒸馏采样用于3D生成显著提升现有2D提升方法的稳定性与一致性，也能像DreamBooth那样仅用少量2D样例学习新概念但服务于3D生成。
</details>

<details>
    <summary>Key points</summary>
    * 将 web 规模 2D 扩散模型预训练 + 多视角 3D 资产渲染数据微调结合。  
    * 条件生成：根据文本提示生成一组多视角图像，保证视角间几何一致性。  
    * 将该模型作为 3D 生成的先验，并通过 SDS（Score Distillation Sampling）提升经典 2D → 3D 方法的稳定性。  
    * 支持少样本微调（如 DreamBooth3D）以实现个性化 3D 生成。  
</details>
</details>

---

<details>
<summary><b>Wonder3D: Single Image to 3D using Cross-Domain Diffusion</b></summary>

* **Authors:** Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, Wenping Wang  
* **arXiv ID:** 2310.15008  
* **One-liner:** 通过跨域扩散模型从单张图像生成高保真有纹理网格  
* **Published in:** CVPR 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2310.15008) | [[PDF]](https://arxiv.org/pdf/2310.15008.pdf) | [[Code]](https://github.com/xxlong0/Wonder3D)

> **核心创新**  
> 提出从单张图像生成高保真纹理网格的方法：先用跨域扩散模型生成多视角法线图与彩色图，再通过多视角跨域注意力机制保证视角间/域间一致性，最后采用几何感知法线融合算法高效提取多视角2D表示为高质量纹理网格。  

<details>
    <summary>Abstract</summary>
    本工作介绍 Wonder3D，一种能够从单视图图像高效生成高保真纹理网格的方法。近期基于 Score Distillation Sampling 的方法已展示从 2D 扩散先验恢复 3D 几何的潜力，但它们通常训练时间长、几何或纹理质量不稳定。与此不同，一些方法直接通过快速网络推理生成 3D 信息，但结果常常质量低、细节差、几何不一致。为全面提升图像-to-3D 任务的质量、一致性与效率，我们提出一种跨域扩散模型，该模型生成多视角法线图与对应彩色图。为确保一致性，我们采用多视角跨域注意力机制以促进视角与域间信息交换。最后，我们引入几何感知法线融合算法，将多视角2D表示高效转换为高质量纹理网格。大量评估表明本方法在重建质量、泛化能力和推理效率方面优于以往工作。  
</details>

<details>
    <summary>Key points</summary>
    * 使用跨域扩散模型，从单视图生成多视角法线图 + 彩色图。  
    * 多视角跨域注意力机制：确保不同视角、不同域（法线 vs 颜色）之间信息一致。  
    * 几何感知法线融合算法：将多视角2D输出高效转换为高质量纹理网格。  
    * 实现从单张图像生成高保真纹理网格，具备强泛化和较快推理速度。  
</details>
</details>

---

<details>
<summary><b>SyncDreamer: Generating Multiview-consistent Images from a Single-view Image</b></summary>

* **Authors:** Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, Wenping Wang
* **arXiv ID:** 2309.03453  
* **One-liner:** 从一张视图生成多视角一致图像的同步扩散模型  
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2309.03453) | [[PDF]](https://arxiv.org/pdf/2309.03453.pdf) | [[Code]](https://github.com/liuyuan-pal/SyncDreamer)

> **核心创新**  
> 提出 SyncDreamer，一种同步多视角生成框架：在逆扩散过程中同时生成多个视图，并通过 3D 感知特征注意力机制同步各视图中间状态，实现几何与颜色的一致性，从而从单视图创建多视角图像集合，为图像-to-3D、文本-to-3D 提供强支撑。

<details>
    <summary>Abstract</summary>
    本文提出一种名为 SyncDreamer 的扩散模型，能够从单视图图像生成多视角一致的图像。利用大规模预训练的 2D 扩散模型，近期如 Zero123 等工作展示了从单图生成可行的新视角图像能力，但在生成图像的几何与颜色一致性方面仍存挑战。为此，我们提出同步多视角扩散模型，从单视图图像直接生成多视图。该模型在逆扩散过程中同步多个视图的中间状态，通过 3D 感知特征注意力机制关联不同视图中的对应特征。实验表明，SyncDreamer 在不同视角间生成高一致性图像，适用于多视角生成任务，如新视角合成、文本-to-3D、图像-to-3D。  
</details>

<details>
    <summary>Key points</summary>
    * 建模多视图**联合概率分布**，在单次逆扩散过程中生成多个视图，而非分别独立生成。  
    * 引入 3D 感知特征注意力机制，同步各视图中间状态，维护几何与颜色一致性。  
    * 从单视图生成多个连贯视角图像，适用于继后续 3D 生成任务。  
    * 展现了强的多视角一致性，为图像-to-3D或文本-to-3D工作提供更可靠输入。  
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

> **核心创新**  
> 提出“分数蒸馏采样”（SDS）损失，将2D扩散模型的分数函数作为先验，在参数空间而非像素空间优化随机初始化的NeRF，实现无3D数据的文本驱动3D生成。

<details>
    <summary>Abstract</summary>
    文本到图像合成的最新突破由基于数十亿图文对训练的扩散模型推动。若将这一范式迁移至三维合成，需大规模标注3D数据与高效去噪架构，而两者目前均不存在。本文绕过这些限制，利用预训练二维文本到图像扩散模型实现文本到三维合成。我们提出基于概率密度蒸馏的损失，使二维扩散模型成为参数化图像生成器的优化先验。借鉴DeepDream式流程，该损失通过梯度下降优化随机初始化的三维模型（神经辐射场，NeRF），使其任意视角的二维渲染损失最小。所得三维模型可任意视角观察、重光照或合成至任意三维环境，无需3D训练数据，也无需修改图像扩散模型，充分验证了预训练图像扩散模型作为先验的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 无需3D标注数据，仅依赖大规模2D文本-图像扩散模型。
    * SDS代替传统重建损失，通过噪声-分数匹配指导NeRF参数更新。
    * 结合mip-NeRF 360与可微分着色，提升几何与外观一致性。
    * 引入视角相关文本提示与几何正则项，减少“扁平化”局部最优。
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

> **核心创新**  
> 利用大模型（BLIP-2）从 3D 渲染图像中自动生成多角度语言描述，构建三模态（点云-图像-文本）对齐数据，实现无需人工标注的 3D 表示学习。

<details>
    <summary>Abstract</summary>
    多模态预训练通过联合对齐三维形状、二维图像与语言描述，在三维表征学习上展现出显著潜力。然而，现有框架在构建此类数据、尤其是为三维形状配语言描述时，依赖手工标注，既难以扩展，描述也缺乏多样性。为此，我们提出ULIP-2：一种简洁高效的三模态预训练框架，仅输入三维数据即可借助大型多模态模型自动生成全面语言描述，无需任何人工标注，可轻松扩展至大规模数据集；同时引入更大容量骨干网络以提升表征能力。我们在Objaverse与ShapeNet两大三维数据集上构建了点云-图像-语言三模态数据，用于训练ULIP-2。实验表明，ULIP-2在零样本三维分类、微调三维分类和三维字幕生成三项下游任务中均带来显著提升：在Objaverse-LVIS零样本分类取得50.6%新SOTA（top-1），ModelNet40达84.7%；在ScanObjectNN微调基准上，仅用140万参数的紧凑模型即实现91.5%整体准确率。ULIP-2开创了无需人工标注即可扩展的三维多模态表征新范式，相较基线方法实现显著跃升。
</details>

<details>
    <summary>Key points</summary>
    * 无需人工标注，仅依赖 3D 数据即可生成语言描述，具备高度可扩展性。
    * 引入 BLIP-2 自动生成多角度、细粒度的语言描述，提升文本模态质量与多样性。
    * 构建两个大规模三模态数据集：ULIP-Objaverse 和 ULIP-ShapeNet。
    * 在零样本 3D 分类、标准分类和 3D 字幕生成任务中均取得 SOTA 性能。
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

> **核心创新**  
> 提出一个可扩展的三模态（点云-图像-文本）对比学习框架，结合自动化文本清洗与增强、3D 骨干网络扩展及难负样本挖掘策略，显著提升 3D 表示的开放世界泛化能力。

<details>
    <summary>Abstract</summary>
    我们提出 OpenShape，一种学习文本、图像与点云多模态联合表征的方法。沿用多模态对比学习框架对齐表征，但聚焦放大三维表征以实现开放世界形状理解。为此，我们整合多个三维数据集扩大训练规模，并设计策略自动过滤与丰富含噪文本；同时探索三维骨干网络扩容方案，引入新型困难负例挖掘模块加速训练。在零样本三维分类基准上，OpenShape 开放世界识别能力显著：在含 1156 类的 Objaverse-LVIS 取得 46.8% 零样本准确率，远超现有方法不足 10%；ModelNet40 达 85.3%，领先前零样本基线 20%，媲美部分全监督方法。所学嵌入涵盖颜色、形状、风格等丰富语义，支持细粒度文本-三维与图像-三维交互；与 CLIP 嵌入对齐后，可直接用于点云字幕、点云条件图像生成等下游任务。
</details>

<details>
    <summary>Key points</summary>
    * 整合 876k 3D 形状数据，覆盖更丰富类别，解决数据规模瓶颈。
    * 提出三种文本增强策略（过滤、图像字幕、图像检索）提升文本质量与语义丰富度。
    * 系统评估并扩展多种 3D 骨干网络（如 PointBERT、SparseConv）以适应大规模训练。
    * 引入难负样本挖掘策略，缓解类别不平衡并增强模型判别力。
    * 学习到的 3D 表示与 CLIP 空间对齐，可直接用于图像生成、字幕生成等跨模态任务。
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

> **核心创新**  
> 提出一种全新的交错生成预训练（I-GPT）框架，使用 dream queries 和 score distillation 直接建模图像后验分布，避免依赖 CLIP 等中间表示，实现图文交错内容的原生生成与理解。

<details>
    <summary>Abstract</summary>
    本文提出 DreamLLM，首个实现多模态理解与创作协同的多模态大型语言模型学习框架。DreamLLM 遵循两大核心原则：一是在原始多模态空间中直接采样建模语言与图像后验，避免 CLIP 等外部特征提取器带来的信息损失，获得更深入的多模态理解；二是生成原始交错的文档，同时建模文本、图像内容及非结构化布局，有效学习所有条件、边缘与联合多模态分布。因此，DreamLLM 成为首个能够生成自由形式交错内容的多模态大型语言模型。综合实验表明，DreamLLM 作为零样本多模态通才表现卓越，充分受益于增强的学习协同效应。
</details>

<details>
    <summary>Key points</summary>
    * 首个支持自由形式图文交错内容生成的 MLLM，具备真正的创作与理解协同能力。
    * 引入 dream queries 和 dream标签
    * 通过 score distillation 在像素空间直接采样图像，避免 CLIP 语义损失与模态鸿沟问题。
    * 支持多模态上下文生成，如图像编辑、主体驱动生成、组合生成等复杂任务。
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

> **核心创新**  
> 提出“世界指令”概念，系统定义并分类七类反映世界动态的编辑指令，构建包含复杂语义变化的多模态数据集，并设计后编辑策略提升编辑质量与一致性。

<details>
    <summary>Abstract</summary>
    扩散模型显著提升了图像编辑性能。现有方法通过文本控制、拖拽操作、掩膜重绘等多种手段实现高质量编辑，其中基于指令的编辑因便捷通用而突出。然而，这些方法仍局限于增删替换等简单操作，无法理解呈现真实物理动态的世界规律。为此，本工作提出EditWorld，首次定义并分类“世界指令图像编辑”任务，依托GPT-3.5、Video-LLaVA、SDXL等预训练大模型构建含世界动态指令的新编辑数据集。EditWorld在 curated 数据上训练，并引入设计的后编辑策略强化指令遵循能力。大量实验表明，新方法在该任务上显著优于现有编辑方法。
</details>

<details>
    <summary>Key points</summary>
    * 提出新任务“世界指令图像编辑”，突破传统编辑仅限于添加/删除/替换的局限。
    * 构建包含10K+三元组的高质量数据集，涵盖真实与虚拟世界中的复杂动态变化。
    * 设计双分支数据生成流程：文本生成图像 + 视频帧提取，结合 GPT、Video-LLaVA、SDXL 等模型。
    * 引入 MLLM Score 新指标，更准确地评估编辑结果是否符合指令语义。
    * 提出后编辑方法（Post-Edit），结合 SAM 与图像修复技术，提升编辑区域精度与非编辑区域一致性。 
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

> **核心创新**  
> 统一多模态词元化 + 四阶段渐进式预训练，实现四模态交错序列的真·任意模态生成，且输入输出表示一致。

<details>
    <summary>Abstract</summary>
    本文提出MIO，一种基于多模态token的基础模型，可端到端自回归地理解并生成语音、文本、图像与视频。现有LLM与MM-LLM虽推动通用智能进展，却未实现真正的任意到任意跨模态理解与生成。近期GPT-4o展示了任意到任意潜力，但闭源且不支持多模态交错序列生成。为此，MIO以离散token混合训练四模态，采用因果多模态建模，分四阶段：对齐预训练、交错预训练、语音增强预训练、多元文本视觉语音监督微调。实验表明，MIO在双模、任意到任意及单模基线上均具竞争力或更优，并展现交错视频文本生成、视觉思维链、视觉指南生成、指令图像编辑等任意到任意高级能力。
</details>

<details>
    <summary>Key points</summary>
    * DIDO 离散词元框架保证输入输出一致，天然支持多模态交错生成。
    * SpeechTokenizer 与 SEED-Tokenizer 将语音/视频/图像压缩为因果词元，可直接做下一词元预测。
    * 三阶段预训练（对齐→交错→语音增强）+ 多任务监督微调，单/双向任务性能均优。
    * 在图文音视频基准上达到或超越现有模型，展现视觉思维链、视觉故事、指令编辑等新能力。
    * 全开源、70 亿参数、端到端训练，无需外部扩散或 TTS 工具。
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

> **核心创新**  
> 首个以 LLM 同时作为开放词汇场景解析器（输出掩码+详细描述）和编辑控制器（规划注意力调制的删/生步骤）的系统，并通过“优化词元+丰富提示”的混合概念学习保持物体身份且可编辑。

<details>
    <summary>Abstract</summary>
    场景图以节点和边结构化地描述图像中的对象及其关系，可作为自然接口实现精准灵活的编辑。基于此，我们提出将大语言模型与文本到图像生成模型融合，通过场景图进行图像编辑。方法分两阶段：①由LLM驱动的场景解析器构建图像场景图，提取对象、关系及掩码、描述等细粒度属性，并用微调扩散模型为每个对象学习优化令牌与描述提示；②编辑阶段由LLM编辑控制器定位修改区域，注意力调制扩散编辑器利用微调模型完成对象增删替换与属性调整。大量实验表明，该框架在编辑精度与场景美观度上显著优于现有方法。
</details>

<details>
    <summary>Key points</summary>
    * 场景图交互：改节点/边即生成可执行操作。
    * LLM 解析器输出图及每物体掩码与描述，混合概念学习定制 SD。
    * LLM 控制器将任意编辑拆为“删除→生成”序列，自动生成提示与框。
    * 删除时仅关注未掩区特征实现无缝填充；插入时强化文本-区域交叉注意力并隔离框间自注意力。
</details>
</details>

---

<details>
<summary><b>UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics</b></summary>

* **Authors:** Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, Hui Ding, Zhe Lin, Hengshuang Zhao
* **arXiv ID:** 2412.07774
* **One-liner:** UniReal treats any image task as “discontinuous video” inside a single 5B diffusion transformer, learning real-world dynamics from massive video frames to unify generation, editing, customization and composition with one model.
* **Published in:** CVPR 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2412.07774) | [[PDF]](https://arxiv.org/pdf/2412.07774.pdf) | [[Code]](https://github.com/XavierCHEN34/UniReal)

> **核心创新**  
> 首个将多样图像任务重构成伪视频帧生成的通用模型，利用可扩展的视频监督自然获得一致性与变化，并引入分层提示（上下文+图像角色）和索引嵌入，在单文本提示下消除多图像输入的歧义。

<details>
    <summary>Abstract</summary>
    我们提出 UniReal，一个用于多种图像生成与编辑任务的统一框架。现有方法往往因任务而异，但核心原则一致：在保持输入输出一致性的同时捕捉视觉变化。受近期视频生成模型在帧间平衡一致性与变化的启发，我们将图像级任务视为不连续的视频生成，把任意数量的输入输出图像当作帧，从而无缝支持图像生成、编辑、定制、合成等任务。尽管面向图像级任务，我们利用视频作为可扩展的通用监督来源，让 UniReal 从大规模视频中学习世界动态，在处理阴影、反射、姿态变化及物体交互方面展现先进能力，并涌现出新应用的潜力。
</details>

<details>
    <summary>Key points</summary>
    * 单个 5B 扩散 Transformer，任意帧间全注意力。
    * 三类图像角色（画布、资产、控制）+ 可学习类别与索引嵌入，将视觉与“IMG1”等提示词绑定。
    * 分层提示：基础提示 + 上下文标签（真实/合成、静态/动态、是否含参考）+ 图像角色标签；推理时可组合。
    * 自动管线从视频挖掘 2300 万+ 帧对，附带描述、掩码、深度、边缘 → 统一监督，覆盖增、删、换、风格化、定制、插入、感知。
    * 零样本涌现多对象插入、深度条件生成、参考补图、物体缩放等能力。
    * 训练：256→512→1024 渐进分辨率；流匹配损失；支持任意宽高比。
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

> **核心创新**  
> 3D VQVAE 将体素压缩为 1024 个离散词元，使冻结的 Qwen-2.5-VL-7B 无需额外网络即可进行完全自回归的 3D 生成/编辑/理解；3D-Alpaca 数据集（34.6 亿词元）保证可扩展性。

<details>
    <summary>Abstract</summary>
    近期，ChatGPT-4o 强大的图文能力让原生多模态大模型备受关注，但其模态仍限于图像与文本。三维内容的理解与生成同样关键。为此，我们提出 ShapeLLM-Omni——首个可任意序列理解并生成三维资产与文本的原生三维大模型。首先训练三维向量量化变分自编码器（3D VQVAE），将物体映射到离散潜空间，实现高效精确的形状表达与重建；基于此构建大规模连续训练集 3D-Alpaca，涵盖生成、理解、编辑三类任务，为后续研究与训练提供丰富资源；最后在 3D-Alpaca 上对 Qwen-2.5-vl-7B-Instruct 进行指令微调。我们的工作首次把基础三维能力融入多模态模型，为原生三维 AI 研究提供有效范式。
</details>

<details>
    <summary>Key points</summary>
    * 统一词元化：643→163 隐式→1024 离散词元，8192 码本，可经 Rectified-Flow 还原高质量网格。
    * 原生 3D-LLM：同一 Transformer 任意顺序处理文本、图像特征、3D 词元，端到端 next-token 预测。
    * 3D-Alpaca：71.2 万文本生 3D、71.2 万图像生 3D、71.2 万 3D 描述、42 万 3D 编辑对 + UltraChat；共 256 万样本、34.6 亿词元。
    * 任务：文本/图像→3D、3D→文本、交互式 3D 编辑（增删改）、多轮对话。
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

> **核心创新**  
> 轻量级 token-refiner 连接器通过词元拼接将 MLLM 编辑嵌入一次性注入潜空间 DiT，仅用扩散损失完成训练，实现免掩膜、免额外自注意力、免 T5 文本编码器的高保真单步编辑。

<details>
    <summary>Abstract</summary>
    近年来，图像编辑模型发展迅猛。GPT-4o、Gemini2 Flash 等前沿多模态模型展现出优异的编辑能力，基本满足用户主流需求，但开源算法与闭源模型差距依然显著。为此，我们发布开源图像编辑模型 Step1X-Edit，性能对标 GPT-4o 与 Gemini2 Flash。具体而言，以多模态大语言模型处理参考图与编辑指令，提取潜嵌入并与扩散图像解码器融合生成目标图；配套构建高质量数据生产管线，同时推出基于真实用户指令的评测基准 GEdit-Bench。实验表明，Step1X-Edit 大幅领先现有开源基线，逼近顶尖闭源模型，为图像编辑领域贡献可复现的先进方案。
</details>

<details>
    <summary>Key points</summary>
    * 统一 MLLM+DiT 管线：Qwen-VL 解析图文 → 压缩多模态词元 → DiT 合成编辑图。
    * 2000 万合成三元组过滤为 100 万高质量数据，覆盖 11 类任务：增删替换、动作、材质、颜色、风格、色调、文字、人像、背景。
    * GEdit-Bench：606 条真实用户指令（中英），脱敏处理，首个经人工筛选的真实编辑基准。
    * 词元拼接条件保留细节，全局引导向量强化语义对齐。
    * 基于 SD3.5 级底座训练，可即插替换 FLUX、HiDream 等。
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

> **核心创新**  
> 渐进式多任务视频扩散首次联合生成 RGB、法线与多粒度分割；在 COCO 上训练的轻量级矢量量化压缩器（LQC）将 512 维 CLIP 特征降至 3 维离散码，实现可跨场景、实时渲染的 3D 高斯语言场。

<details>
    <summary>Abstract</summary>
    从单张或稀疏二维图像中恢复三维结构并实现开放词汇场景理解，是计算机视觉的核心难题。现有方法依赖每场景稠密视角优化，在视角受限时渲染伪影严重、语义合成失真。为此，我们提出生成式框架 LangScene-X，统一生成三维一致的多模态信息用于重建与理解。该方法首先训练 TriMap 视频扩散模型，通过渐进知识整合，从稀疏输入联合生成外观（RGB）、几何（法向）与语义（分割图）；其次引入语言量化压缩器（LQC），在大规模图像数据集上训练，高效编码语言嵌入，实现跨场景泛化而无需逐场景重训；最后将语言信息对齐至三维表面，重建语言表面场，支持开放式语言查询。真实数据实验表明，LangScene-X 在质量与泛化性上均优于现有最佳方法。
</details>

<details>
    <summary>Key points</summary>
    * TriMap 扩散：四阶段训练（关键帧→3D 一致→加法线→加语义）输出视频级一致三元组。
    * LQC：2048 码 VQ-VAE，梯度旁路，将任意 CLIP 场压至 3 通道索引，零场景成本。
    * 语言表面场：DUSt3R 稀疏初始化 + 渐进法线正则 + 2D/3D 语义聚类；5k 步 RGB/几何，5k 步语义。
    * 推理：2 张图→49 帧 TriMap→语言高斯→实时新视角查询。
    * 通用：单模型覆盖室内、物体、野外场景，无需逐场景自编码器或稠密标定。
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

> **核心创新**  
> 提出一种无需交叉注意力或辅助适配器的自回归图像生成方法，通过两阶段训练实现多模态输入的语义与像素级对齐，显著提升训练效率与生成可控性。

<details>
    <summary>Abstract</summary>
    最新文本到图像模型虽质量高，却在精细视觉控制、多模态输入平衡及复杂场景训练成本上仍存短板。为此，我们提出MENTOR——一种高效自回归框架，实现多模态条件微调生成图像。MENTOR将自回归图像生成器与两阶段训练结合，无需额外适配器或交叉注意力，即可完成多模态输入与图像输出的token级细粒度对齐。训练分为：1）多模态对齐阶段，建立像素与语义级稳健对齐；2）多模态指令微调阶段，平衡多模态融合并提升可控生成。尽管模型体量小、基线组件一般、训练资源有限，MENTOR仍在DreamBench++基准上取得优异表现，在概念保持与指令遵循方面超越竞对，并相比扩散模型具备更高重建保真、更广任务适应与更佳训练效率。
</details>

<details>
    <summary>Key points</summary>
    * 自回归架构统一处理图像与文本输入，简化模型结构。
    * 两阶段训练策略：第一阶段实现多模态对齐，第二阶段增强指令遵循与跨模态融合能力。
    * 无需交叉注意力机制，降低训练成本。
    * 支持多种下游任务（如图像分割、主题驱动生成、上下文图像生成）且无需修改架构。
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

> **核心创新**  
> 提出将 2D 语义特征与 3D 几何特征融合为条件，引导视频扩散模型修复伪影视图，并设计参考视图引导的轨迹采样策略，实现高质量 3DGS 重建。

<details>
    <summary>Abstract</summary>
    稀疏视角下用3D高斯溅射重建3D场景因信息不足而成病态问题，常伴显著伪影。近期方法尝试引入生成先验补全缺失区域，却难以保持与输入观测一致。为此，我们提出GSFixer，一种提升稀疏输入3DGS质量的新框架。核心是基于DiT视频扩散模型的参考引导视频修复，其以成对伪影3DGS渲染与干净帧训练，并接受额外参考条件。模型将输入稀疏视图作为参考，融合视觉几何基础模型提取的2D语义与3D几何特征，在修复伪影新视角时增强语义连贯性与三维一致性。此外，针对3DGS伪影修复缺乏基准，我们构建DL3DV-Res，含低质量3DGS渲染的伪影帧。大量实验表明，GSFixer在3DGS伪影修复与稀疏视角重建上均优于现有最佳方法。
</details>

<details>
    <summary>Key points</summary>
    * 构建基于 DiT 的视频扩散模型，用于修复 3DGS 渲染中的伪影视图。
    * 引入 DINOv2 和 VGGT 提取 2D 语义与 3D 几何特征，增强视图一致性与语义保真度。
    * 设计参考视图引导的轨迹采样策略，兼顾视角覆盖与修复质量。
    * 提出 DL3DV-Res 数据集，用于评估 3DGS 伪影修复性能。
    * 在稀疏视角重建任务中显著优于现有方法，具备良好的跨数据集泛化能力。
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

> **核心创新**  
> 论文的核心创新在于设计了一种解耦的条件神经辐射场（disentangled conditional NeRF）架构，通过引入形状码和外观码分别控制形状变形和外观颜色，并利用 CLIP 模型的多模态能力实现文本和图像驱动的 3D 操控。

<details>
    <summary>Abstract</summary>
    我们提出 CLIP-NeRF，这是一种用于神经辐射场（NeRF）的多模态 3D 对象操作方法。通过利用最近的对比语言-图像预训练（CLIP）模型的联合语言-图像嵌入空间，我们提出了一个统一框架，允许用户通过简短的文本提示或示例图像以用户友好的方式操作 NeRF。具体来说，为了将 NeRF 的新视图合成能力与生成模型的潜在表示的可控操作能力相结合，我们引入了一种解耦的条件 NeRF 架构，允许分别控制形状和外观。这是通过将形状条件通过学习到的形变场应用于位置编码，并将颜色条件推迟到体积渲染阶段来实现的。为了将这种解耦的潜在表示与 CLIP 嵌入联系起来，我们设计了两个代码映射器，它们以 CLIP 嵌入作为输入，并更新潜在代码以反映目标编辑。这些映射器通过基于 CLIP 的匹配损失进行训练，以确保操作的准确性。此外，我们提出了一种逆优化方法，可以将输入图像准确地投影到潜在代码中以进行操作，从而实现对真实图像的编辑。我们通过在各种文本提示和示例图像上的广泛实验来评估我们的方法，并提供了一个直观的界面以实现交互式编辑。
</details>

<details>
    <summary>Key points</summary>
    * 提出了解耦的条件神经辐射场架构，分别控制形状和外观。
    * 设计形状变形网络，通过位移向量更新位置编码，实现对形状的精确控制。
    * 借助 CLIP 模型的文本和图像编码器，将文本提示或示例图像映射到解耦的潜在空间，实现对神经辐射场的操控。
    * 引入前馈代码映射器，实现快速推理，支持同一类别中不同对象的编辑。
    * 提出一种逆优化方法，能够将真实图像投影到潜在代码中进行编辑。
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

> **核心创新**  
> Genesis 的核心创新在于设计了一种统一的多模态生成架构，通过共享条件输入和结构化语义描述模块 DataCrafter，实现视频和 LiDAR 数据在视觉和几何领域的协同演化。

<details>
    <summary>Abstract</summary>
    我们提出Genesis，一个统一框架，用于联合生成多视角驾驶视频和LiDAR序列，确保时空一致性和跨模态一致性。Genesis采用两阶段架构，整合基于DiT的视频扩散模型与3D-VAE编码，以及具有BEV感知的LiDAR生成器，后者结合基于NeRF的渲染和自适应采样。两种模态通过共享潜在空间直接耦合，实现视觉和几何领域的连贯演变。为引导生成structured semantics，我们引入DataCrafter，一个基于视觉语言模型的字幕生成模块，提供场景级和实例级监督。在nuScenes基准上的广泛实验表明，Genesis在视频和LiDAR指标（FVD 16.95，FID 4.24，Chamfer 0.611）上达到最佳性能，并有助于分割和3D检测等下游任务，验证了生成数据的语义保真度和实用性。
</details>

<details>
    <summary>Key points</summary>
    * Genesis 采用双分支架构，通过共享条件输入（如场景描述和布局）实现视频和 LiDAR 的联合生成，确保跨模态一致性。
    * 基于视觉语言模型，DataCrafter 提供场景级别和实例级别的描述，生成详细的语义先验，指导多模态生成。
    * 视频生成分支采用 DiT 扩散模型结合 3D-VAE 编码器，捕捉细粒度视觉动态，而 LiDAR 生成分支则利用 BEV 表示和 NeRF 渲染实现精确的几何重建。
    * 通过将图像特征提取的 BEV 特征融入 LiDAR 扩散模型，增强几何和视觉领域之间的一致性。
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

> **核心创新**  
> 论文的核心创新在于设计了一个统一的控制编码器，整合点云、体素、骨骼姿态和边界框等多种条件信号，通过渐进式、难度感知的训练策略，实现对3D生成过程的精确控制。

<details>
    <summary>Abstract</summary>
    我们提出了Hunyuan3D-Omni，这是一个基于Hunyuan3D 2.1的统一框架，用于实现精细且可控的3D资产生成。除了图像之外，它还接受点云、体素、包围盒和骨骼姿态先验作为条件信号，从而能够精确控制几何形状、拓扑结构和姿态。我们的模型并非为每种模态单独设置头部，而是将所有信号统一在一个跨模态架构中。我们采用了一种渐进式的、具有难度感知的采样策略进行训练，该策略为每个示例选择一种控制模态，并倾向于对更难的信号（例如骨骼姿态）进行采样，同时减少对更容易的信号（例如点云）的权重，从而鼓励稳健的多模态融合和对缺失输入的优雅处理。实验表明，这些额外的控制措施提高了生成精度，实现了几何感知的变换，并增强了生产工作流程的鲁棒性。
</details>

<details>
    <summary>Key points</summary>
    * Hunyuan3D-Omni 提出了一个支持多种条件信号的统一框架，实现了对几何、拓扑和姿态的精确控制。
    * 设计了一个统一控制编码器，将各种额外条件信号编码为嵌入特征，与图像特征结合，实现可控的3D生成。
    * 提出了一种渐进式的、难度感知的采样策略，优先训练更具挑战性的条件信号，提高模型对多模态融合的鲁棒性。
    * 实验表明，与仅基于图像的3D生成相比，额外的控制信号显著提高了生成精度，增强了几何保真度，并改善了生产流程中的稳健性。
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

> **核心创新**  
> 核心创新在于通过体积渲染学习一个三维语言场，利用多尺度图像块特征金字塔进行监督，实现分层、视角一致的3D语义查询，同时保留原始CLIP模型的零样本能力。

<details>
    <summary>Abstract</summary>
    Language Embedded Radiance Fields (LERFs)是一种创新方法，它将自然语言描述与3D空间中的特定位置关联起来。LERFs通过在NeRF（神经辐射场）中体积渲染CLIP嵌入，学习一个密集的、多尺度的语言场。这种方法使得LERFs能够实时、交互式地为各种语言提示提取3D相关性图，而无需依赖区域提议或掩码。LERFs在机器人技术、理解视觉-语言模型以及与3D场景交互方面具有潜在的应用价值，支持长尾开放词汇查询，并在3D场景中实现像素级对齐的零样本查询。
</details>

<details>
    <summary>Key points</summary>
    * 多尺度语言场：学习基于位置和物理尺度的密集3D场，捕捉分层语义
    * 体积渲染CLIP嵌入：沿射线使用NeRF体积渲染权重渲染CLIP嵌入，并归一化到单位球面
    * 视角一致性：跨多个训练视角平均嵌入，比2D CLIP产生更局部化且3D一致的相关性热图
    * 无需微调：直接使用预训练CLIP模型，无需在分割数据集上训练，保留开放词汇能力
    * DINO正则化：引入自监督DINO特征作为正则化器，改善场的平滑性和物体边界
    * 解耦架构：为语言特征（CLIP/DINO）和辐射场（RGB/密度）训练独立网络，防止语言梯度影响几何
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

> **核心创新**  
> 核心创新在于将三维生成重构为视图集扩散：通过生成一组二维视图并将去噪器设计为三维重建器加可微渲染，模型仅从零二维数据中学习三维生成先验，在统一框架内实现模糊感知重建与生成，无需三维真实标注。

<details>
    <summary>Abstract</summary>
    Viewset Diffusion是一种基于扩散模型的3D对象生成器，仅使用多视角2D数据进行监督。它利用视图集（多个2D视角的集合）与3D模型之间的一一映射关系，训练扩散模型生成视图集，并设计神经网络生成器内部重建对应的3D模型。该模型可基于零、一或多个输入视图进行条件生成。在单个视图条件下，它能考虑任务的模糊性，生成多个与输入兼容的解决方案。模型训练仅使用渲染损失，每个视图集最少仅需三视图，且能高效地以前馈方式完成重建。
</details>

<details>
    <summary>Key points</summary>
    * 视图集-三维等价性：利用视图集与三维模型之间的双射映射，仅通过二维监督实现三维生成
    * 内部三维重建：扩散去噪函数Φ从噪声视图集重建显式三维辐射场（体素网格），由可微渲染Ψ解码
    * 任意条件输入：支持零视图（无条件）、单视图或少视图条件生成，通过对不同视图施加不同噪声水平实现
    * 几何反投影：二维特征通过相机感知的反投影提升到三维，在聚合前创建视图对齐的特征体积
    * 基于注意力的多视图融合：使用交叉注意力以遮挡感知方式聚合多视图特征，最粗层级使用学习到的类别特定查询
    * 概率式重建：通过学习分布采样处理单视图重建的固有模糊性，生成多样化合理三维形状而非模糊平均
    * 数据高效性：训练时每物体仅需3个视图，远少于基于优化方法（50+视图）
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

> **核心创新**  
> 核心创新在于跨模态注意力蒸馏（MoAI），在训练和推理期间将图像扩散分支的空间注意力图注入几何扩散分支，构建协同多任务学习框架：用确定性几何线索正则化图像合成，同时利用丰富的语义图像特征增强几何补全，确保生成的图像与几何精确对齐。

<details>
    <summary>Abstract</summary>
    我们提出了一种基于扩散模型的框架，通过变形与修复方法实现图像和几何严格对齐的新视图生成。与依赖密集带姿态图像或局限于固定视域的位姿嵌入模型不同，我们的方法借助现有几何预测器，从参考图像中提取局部几何信息，并将新视图合成转化为图像与几何的修复任务。为确保生成图像与几何的精确对齐，我们提出跨模态注意力蒸馏技术，将图像扩散分支的注意力图注入到并行的几何扩散分支，在训练与推理中实现信息共享。我们进一步引入基于邻近性的网格条件，整合深度与法向信息，通过插值点云与过滤错误几何预测来优化生成结果。实验表明，我们的方法在未见场景中实现了高保真外推视图生成，具备竞争性的插值重建质量，并生成对齐的彩色点云以支持全面的三维补全。
</details>

<details>
    <summary>Key points</summary>
    * 利用现成的几何预测器从参考图像估计部分点云，将其投影至目标视角作为稀疏几何条件进行修复
    * 将图像U-Net的注意力图转移至几何U-Net，通过共享结构信息对齐模态，同时防止有害的特征混合
    * 通过球旋转算法将稀疏点云转为网格，拼接深度和法线图以过滤错误投影并提供更密集的几何线索
    * 处理位于参考相机凸包外的目标视角，利用学习到的场景先验生成合理的遮挡/未见区域
    * 支持未配准的参考图像，无需先前扩散模型NVS方法所需的已知相机位姿
    * 联合预测新视角图像和点图，生成几何一致的彩色点云以实现完整3D补全
    * 几何补全的确定性为更模糊图像修复任务提供更强训练信号进行正则化
    * 将所有参考视图的键/值特征与目标查询拼接，实现跨任意数量输入的遮挡感知特征聚合
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

> **核心创新**  
> 核心创新是Token-Disentangled (Tok-D) Transformer块，它通过基于指示变量的调制在每层分别处理源token和目标token（对注意力和MLP层进行前置缩放/移位调制，后置缩放调制），解决了仅解码器NVS架构的根本低效问题。这种解耦消除了冗余的特征对齐，提升了计算效率，关键地使模型对合成数据伪影具有鲁棒性，从而能够通过合成数据增强实现有效的数据扩展——这是此前会降低基线模型性能的方法。

<details>
    <summary>Abstract</summary>
    为解决稀疏输入视图下新视图合成的泛化问题，我们提出了一种结合合成数据和token解耦的Transformer架构方法。首先，利用扩散模型生成的合成数据扩展训练集，增强模型对未见场景的泛化能力。其次，针对合成数据中的伪影问题，设计了token解耦过程，通过区分真实特征和伪影特征的token，增强特征分离，提高模型对伪影的鲁棒性。实验表明，该方法在多个基准测试中均取得了优异的性能，同时显著降低了计算成本。
</details>

<details>
    <summary>Key points</summary>
    * 揭示仅解码器NVS Transformer（如LVSM）在对齐源/目标特征时浪费容量且易受源噪声影响，限制了可扩展性。
    * 引入token类型感知调制（δ指示器），为每层Transformer的源token和目标token生成独立的风格向量及缩放/移位参数。
    * 解耦机制防止合成源视图的伪影传播至目标生成，与朴素Transformer不同。
    * 使用CAT3D生成多视角数据，但关键地反转条件/目标角色（真实条件图像作为目标，合成视图作为源），强制输出真实性并提升鲁棒性。
    * 提出通过warping和噪声混合的3D一致噪声初始化方法，提升合成数据质量。
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

> **核心创新**  
> 核心创新是启发式引导分割（HuGS）框架，它将静/瞬态分离重构为协同过程：不再单纯依赖启发式或分割，而是通过互补启发式（基于SfM的高频纹理检测 + 基于颜色残差的低频纹理检测）提供粗略静态线索，引导SAM生成精确、边界感知的静态掩码，从而打破了现有方法在通用性与准确性之间的根本权衡。

<details>
    <summary>Abstract</summary>
    NeRF在处理静态场景时表现出色，但在动态场景中易受干扰。为解决这一问题，我们提出了“启发式引导分割”（HuGS）方法，结合手工启发式规则和先进分割模型，有效分离静态场景与动态干扰物。该方法融合了基于运动恢复结构（SfM）的启发式规则和颜色残差启发式规则，以适应不同纹理特征。实验表明，HuGS方法能显著减少动态干扰对NeRF的影响，提升其在非静态场景中的性能。
</details>

<details>
    <summary>Key points</summary>
    * NeRF的静态场景假设使其在包含移动物体、阴影等瞬态干扰物的真实数据中产生伪影
    * 静态物体的SfM特征点在跨视图匹配次数上显著多于瞬态物体，且高频与低频纹理需不同启发式处理
    * HuGS通过阈值筛选SfM静态特征点→SAM生成初始静态图→部分训练Nerfacto提供颜色残差图→双启发式融合→再次SAM生成最终静态掩码
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

> **核心创新**  
> 核心创新在于单图像文本反转技术：通过对单张输入图像进行大量增强，创建自定义文本token嵌入，使扩散模型生成视角一致、物体特定的先验信息，从而在无需任何3D监督的情况下，弥合了通用2D扩散模型与精确单视角3D重建之间的鸿沟。

<details>
    <summary>Abstract</summary>
    单图像重建完整360°对象模型极具挑战性，因为二维图像信息不足以推断出完整的三维结构。为解决这一问题，研究者采用了一种基于扩散模型的条件图像生成器，通过精心设计的提示词（prompt）来引导模型“想象”出对象的多角度视图。接着，他们借鉴DreamFields和DreamFusion的方法，将输入图像、条件先验和其他约束条件融合，最终生成一个与输入视角一致且能合理外推到未见视角的3D模型。实验表明，该方法在多个基准测试中均取得了最佳的重建效果，无论是定性还是定量评估，都显著优于其他单目3D重建方法。
</details>

<details>
    <summary>Key points</summary>
    * 结合两个同时的目标——重建损失（拟合输入视角）和基于SDS的先验损失（使用Stable Diffusion约束新视角）。
    * 单图像文本反转通过对输入图像进行重度增强来优化自定义token⟨e⟩，使扩散模型针对特定物体而非通用类别。
    * 使用InstantNGP实现快速训练，并引入从粗到精的训练策略以减少表面伪影。
    * 添加2D法向量平滑正则化和掩码损失以改善几何质量。
</details>
</details>

---

### <a href="./consisency-paper/modality+temporal-consistency/README_zh.md">模态 + 时间一致性</a>

**能力特征**：将文本描述或静态图像转换为时间连贯的动态视频序列的模型。

**意义**：目前最突出的融合方向，实现高质量的文本到视频和图像到视频生成。

#### 代表性工作

<details>
<summary><b>Lumiere: A Space-Time Diffusion Model for Video Generation</b></summary>

* **Authors:** Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, Inbar Mosseri
* **arXiv ID:** 2401.12945 
* **One-liner:** 一种文本到视频扩散模型，通过 Space-Time U-Net 一次生成完整视频时长，强化运动连贯性  
* **Published in:** SIGGRAPH-ASIA 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2401.12945) | [[PDF]](https://arXiv.org/pdf/2401.12945.pdf) | [[Project Page]](https://lumiere-video.github.io/)

> **核心创新**  
> 提出一种 Space-Time U-Net 架构，可在一次模型运行中同时处理空间与时间维度，生成完整时长视频，从而克服此前多阶段关键帧＋时间超分辨 cascade 所带来的时间一致性问题。 

<details>
    <summary>Abstract</summary>
    我们提出Lumiere——一个为合成具有真实、多样且连贯运动的视频而设计的文本到视频扩散模型，通过一次性单次前向传递的时空U-Net架构在多个时空尺度上联合上下采样并借助预训练文生图模型直接生成全帧率低分辨率完整视频，而非现有模型先合成远端关键帧再做时域超分的做法，从而根本解决全局时间一致性难题，在文生视频任务上取得最先进效果，并轻松支持图生视频、视频补绘与风格化等广泛应用。
</details>

<details>
    <summary>Key points</summary>
    * 引入 Space-Time U-Net，可同时进行空间与时间的下/上采样一次生成整色视频。  
    * 避免传统关键帧 + 逐步超分辨策略，从根源提升时间一致性。  
    * 利用预训练文本到图像扩散模型结构再扩展至视频生成。  
    * 支持多任务：文本到视频、图像到视频、视频修补、风格化生成。  
</details>
</details>

---

<details>
<summary><b>Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets</b></summary>

* **Authors:** Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach
* **arXiv ID:** 2311.15127 
* **One-liner:** 隐空间视频扩散模型，拓展至大规模数据集，实现高分辨率文本→视频与图像→视频生成  
* **Published in:** arXiv 2023  
* **Links:** [[Paper]](https://arxiv.org/abs/2311.15127) | [[PDF]](https://arxiv.org/pdf/2311.15127.pdf) | [[Code]](https://github.com/Stability-AI/generative-models)

> **核心创新**  
> 系统提出训练隐空间视频扩散模型的三阶段流程（文本→图像预训练、视频预训练、高质量视频微调）并强调数据集筛选与注释的重要性，从而在大规模视频数据上实现高分辨率生成与多视角 3D 先验。   

<details>
    <summary>Abstract</summary>
    我们提出Stable Video Diffusion——一种用于高分辨率文本到视频和图像到视频生成的潜视频扩散模型，通过系统梳理并执行文生图预训练、视频预训练与高质量微调三阶段流程，配合精心设计的视频筛选与标注策略，先训练出强大基础模型，再在其上微调出媲美闭源系统的文生视频模型，同时该基础模型还可驱动图生视频、适配相机运动LoRA，并能作为多视图3D先验快速微调出一次生成多视图且计算量远低于图像法的新模型。
</details>

<details>
    <summary>Key points</summary>
    * 提出三阶段训练流程：文本→图像预训练 → 视频预训练 → 高质量视频微调。  
    * 强调数据集策划（captioning & filtering）对生成质量的影响。  
    * 模型兼具图像→视频和文本→视频能力，并提供多视角 3D 模型潜力。  
    * 展示在大规模视频生成任务上的效率与效果提升。  
</details>
</details>

---

<details>
<summary><b>AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning</b></summary>

* **Authors:** Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, Bo Dai
* **arXiv ID:** 2307.04725 
* **One-liner:** 插入可复用运动模块将任何个性化文本→图像扩散模型转变为动画生成器，无需模型专用调优  
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.04725) | [[PDF]](https://arxiv.org/pdf/2307.04725.pdf) | [[Code]](https://github.com/guoyww/AnimateDiff)

> **核心创新**  
> 提出一个可插拔的“运动模块”（motion module），在冻结的文本→图像扩散模型基础上训练一次后即可无缝集成，实现在已有图像生成模型上直接生成动画。并提出 MotionLoRA 轻量微调方案适配新的运动模式。

<details>
    <summary>Abstract</summary>
    随着文本到图像（T2I）扩散模型（例如Stable Diffusion）及其个性化技术（如DreamBooth和LoRA）的发展，每个人都能以可承受的成本将想象转化为高质量图像。然而，为现有高质量个性化T2I添加运动动态并使其生成动画仍是开放难题。本文提出AnimateDiff，一种无需针对特定模型调优即可为个性化T2I模型赋予动画能力的实用框架。其核心是可即插即用的运动模块，一次训练后即可无缝集成源自同一基础T2I的任意个性化模型。通过所提出的训练策略，该模块从真实视频中有效学习可迁移的运动先验。训练完成后，模块可插入个性化T2I模型，构成个性化动画生成器。我们进一步提出MotionLoRA，一种轻量级微调技术，使预训练运动模块以极低训练与数据采集成本适应新运动模式，如不同镜头类型。我们在社区收集的多个代表性个性化T2I模型上评估AnimateDiff与MotionLoRA，结果表明该方法在保持视觉质量与运动多样性的同时，帮助这些模型生成时序平滑的动画片段。
</details>

<details>
    <summary>Key points</summary>
    * 插入可复用运动模块，可直接嵌入任意个性化 T2I 模型。  
    * MotionLoRA：轻量级微调方案，快速适配新运动模式。  
    * 保持原图像生成模型的高质量输出，同时赋予运动能力。  
    * 简化动画生成流程，降低模型-调优负担。  
</details>
</details>

---

<details>
<summary><b>Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning</b></summary>

* **Authors:** Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, Ishan Misra
* **arXiv ID:** 2311.10709
* **One-liner:** 采用分两步生成流程：文本→图像、再图像+文本→视频，从而实现高质量文本→视频生成  
* **Published in:** ECCV 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2311.10709) | [[PDF]](https://arxiv.org/pdf/2311.10709.pdf) | [[Project Page]](https://emu-video.metademolab.com/)

> **核心创新**  
> 提出生成流程的因式分解（factorization）：先生成图像（text-conditioned），再基于该图像和文本生成视频。并设计特定噪声调度和多阶段训练策略，突破以往深度模型级联限制，直接生成高质量高分辨率视频。   

<details>
    <summary>Abstract</summary>
    我们提出Emu Video，一种文本到视频生成模型，将生成过程分解为两步：首先根据文本生成图像，然后根据文本和生成的图像生成视频。我们确定了关键设计决策——调整扩散噪声调度和多阶段训练，使我们能够直接生成高质量、高分辨率的视频，而无需像先前工作那样依赖深度级联模型。在人类评估中，我们的生成视频在质量上显著优于所有先前工作——相比谷歌Imagen Video为81%，相比英伟达PYOCO为90%，相比Meta Make-A-Video为96%。我们的模型优于RunwayML Gen2和Pika Labs等商业方案。最后，我们的分解方法天然支持基于用户文本提示为图像添加动画，其生成结果相比先前工作有96%的偏好率。
</details>

<details>
    <summary>Key points</summary>
    * 生成流程分两步：文本→图像 →（文本＋图像）→视频。  
    * 调整扩散模型的噪声调度与多阶段训练流程。  
    * 实现高质量、高分辨率视频生成，并提升用户图像动画能力。  
    * 在人工评测中优于先前文本→视频方法。  
</details>
</details>

---

<details>
<summary><b>VideoPoet: A Large Language Model for Zero-Shot Video Generation</b></summary>

* **Authors:** Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, Lu Jiang
* **arXiv ID:** 2312.14125
* **One-liner:** 一个解码器-仅 Transformers 架构的大语言模型，可零样本生成视频，支持多模态条件输入（文本、图像、音频）  
* **Published in:** ICML 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14125) | [[PDF]](https://arxiv.org/pdf/2312.14125.pdf) | [[Project Page]](https://sites.research.google/videopoet/)

> **核心创新**  
> 将大语言模型训练范式扩展至视频生成：使用解码器-仅 Transformer 架构处理多模态输入（文本、图像、音频、视频）并在零样本设置下生成高质量视频（含匹配音频），开拓视频生成的新路径。 

<details>
    <summary>Abstract</summary>
    我们推出VideoPoet，这一语言模型能将多种条件信号合成为带匹配音频的高质量视频。它采用仅解码器Transformer架构，可处理图像、视频、文本与音频等多模态输入。训练流程遵循大语言模型范式，分预训练与任务适配两阶段：预训练时在自回归Transformer框架内融合多模态生成目标，所得模型可迁移至各类视频生成任务。实验表明，该模型在零样本视频生成中达到领先水平，尤擅生成高保真运动。
</details>

<details>
    <summary>Key points</summary>
    * 使用解码器-仅 Transformer，输入为多模态（文本、图像、视频、音频）。  
    * 训练流程遵循 LLM：大规模预训练 + 任务适配阶段。  
    * 支持零样本视频生成，满足不同条件输入。  
    * 展现多模态视频生成新范式，推动视频生成研究。  
</details>
</details>

---

### <a href="./consisency-paper/spatial-temporal-consistency/README_zh.md">空间一致性 + 时间一致性</a>

**能力特征**：这类模型能够在模拟时间动态演化的同时保持三维空间结构的一致性，但可能在语言理解或可控性方面存在一定局限。

**意义**：这些模型代表了理解"三维世界如何运动"的关键技术成就，构成了世界模型的物理引擎组成部分。

#### 代表性工作

<details>
<summary><b>DUSt3R: Geometric 3D Vision Made Easy</b></summary>

* **Authors:** Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud
* **arXiv ID:** 2312.14132  
* **One-liner:** 无需相机标定与外参，即可实现任意图像集合的密集、无约束立体 3D 重建  
* **Published in:** CVPR 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14132) | [[PDF]](https://arxiv.org/pdf/2312.14132.pdf)

> **核心创新**  
> 提出一种颠覆传统多视角立体重建（MVS）流程的新范式——直接回归稠密点图（point-maps）而省略相机内参／外参估计。该方法统一单视图与多视图深度估计、相机位姿估计与重建任务，在“采集环境随意”的野外场景下也能表现优异。

<details>
    <summary>Abstract</summary>
    野外多视角立体重建需先估算相机内参与位姿，这些参数繁琐却又是像素三维三角化的核心。我们反其道而行，提出DUSt3R，一种无需任何标定与位姿即可对任意图像集进行稠密自由立体重建的新范式。我们把成对重建转化为点图回归，摆脱传统投影模型的硬约束，自然统一单目与双目情形。图像多于两张时，我们引入简单有效的全局对齐策略，将所有成对点图归到同一坐标系。网络采用标准Transformer编解码器，可借力强大预训练权重。该方法直接输出场景三维模型与深度，并可从中无缝恢复像素匹配、相对与绝对相机参数。在单目/多视角深度及相对位姿等任务上的全面实验表明，DUSt3R统一多种三维视觉任务并刷新最佳成绩，让几何三维视觉变得轻而易举。
</details>

<details>
    <summary>Key points</summary>
    * 将多视角立体重建问题转化为点图回归，无需显式三角化与相机矩阵。  
    * 统一框架处理单视图、两视图及多视图场景。  
    * Transformer-基模型直接从图像预测稠密 3D 几何，跳过传统 SfM／MVS 模块。  
    * 在深度、位姿与重建任务上取得新的性能标杆。  
</details>
</details>

---

<details>
<summary><b>4D Gaussian Splatting for Real-Time Dynamic Scene Rendering</b></summary>

* **Authors:** Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang
* **arXiv ID:** 2310.08528  
* **One-liner:** 引入 4D 高斯 splatting 表示法，用于动态场景的实时渲染  
* **Published in:** CVPR 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2310.08528) | [[PDF]](https://arxiv.org/pdf/2310.08528.pdf) | [[Code]](https://github.com/hustvl/4DGaussians)

> **核心创新**  
> 提出 4D 高斯 splatting（4D-GS）——将空间加时间维度统一建模，每一个高斯原语具备时空扩展属性，从而显著提升动态场景的训练效率与渲染实时性能。

<details>
    <summary>Abstract</summary>
    动态场景的表征与渲染一直是一项重要却充满挑战的任务。尤其当需要精确建模复杂运动时，往往难以保证高效率。为实现实时动态渲染并兼顾训练与存储效率，我们提出4D高斯抛雪球（4D-GS），将动态场景作为整体表征，而非逐帧使用3D-GS。4D-GS创新地融合3D高斯与4D神经体素，并借鉴HexPlane设计分解式神经体素编码，高效构建高斯特征，再由轻量级MLP预测新时刻的高斯形变。该方法在RTX 3090上800×800分辨率下实现82 FPS实时渲染，质量媲美甚至超越现有最佳方案。
</details>

<details>
    <summary>Key points</summary>
    * 引入时空高斯原语：同时建模 XYZ + 时间轴。  
    * 设计高效的变形场 (deformation field) 和 splatting 渲染器以支持动态几何与外观。  
    * 在大规模场景下实现实时（30 fps+）渲染。  
    * 相比传统逐帧建模显著提升训练/存储效率。  
</details>
</details>

---

<details>
<summary><b>Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes</b></summary>

* **Authors:** Zhengqi Li, Simon Niklaus, Noah Snavely, Oliver Wang  
* **arXiv ID:** 2011.13084  
* **One-liner:** 用于动态场景的新视角+新时间合成的神经场景流域（NSFF）  
* **Published in:** CVPR 2021  
* **Links:** [[Paper]](https://arxiv.org/abs/2011.13084) | [[PDF]](https://arxiv.org/pdf/2011.13084.pdf) | [[Code]](https://github.com/zhengqili/Neural-Scene-Flow-Fields)

> **核心创新**  
> 提出 Neural Scene Flow Fields (NSFF)：一种以时间变化的连续函数表示动态场景，其同时编码几何、外观与三维场景流（scene flow），从单目视频及已知摄像机轨迹中学习，实现联合视角与时间插值。 

<details>
    <summary>Abstract</summary>
    我们提出一种仅需单目视频与已知相机位姿即可实现动态场景新视角与时间合成的方法。为此，我们引入神经场景流场，这一新表征将动态场景建模为外观、几何与三维运动的时变连续函数，并通过神经网络优化以拟合观测视角。实验表明，该表征可处理含薄结构、视角相关效应及自然运动复杂场景，在多项测试中显著优于最新单目视角合成方法，并在真实视频中展示时空视角合成的定性结果。
</details>

<details>
    <summary>Key points</summary>
    * 将动态场景表示为连续函数 f(x,y,z,t,θ,φ) → (radiance, density, scene-flow)。  
    * 使用单目视频 + 已知相机轨迹，不依赖多视图阵列。  
    * 支持新视角合成（空间）与新时间合成（插帧/时间方向）。  
    * 能处理薄结构、视角依赖反射与复杂运动。  
</details>
</details>

---

<details>
<summary><b>CoTracker: It is Better to Track Together</b></summary>

* **Authors:** Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht
* **arXiv ID:** 2307.07635  
* **One-liner:** 基于 Transformer 的联合点追踪模型，可同时追踪成千上万点，提高准确性与鲁棒性
* **Published in:** ECCV 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2307.07635) | [[PDF]](https://arxiv.org/pdf/2307.07635.pdf) | [[Code]](https://github.com/facebookresearch/co-tracker)

> **核心创新**  
> 提出 CoTracker：通过 Transformer 架构联合追踪大量 2D 点流，建模点与点之间的依赖关系，从而提升遮挡点、视野外点的追踪能力与整体准确性。 

<details>
    <summary>Abstract</summary>
    我们推出CoTracker，一种基于Transformer的模型，可在长视频中同时追踪大量二维点。与多数独立逐点追踪的方法不同，CoTracker联合建模点间依赖，显著提升精度与鲁棒性，并能追踪被遮挡或已出画面的点。针对此类追踪器，我们提出多项创新，包括使用token代理大幅提升内存效率，使CoTracker在单GPU上同时联合追踪7万个点。该算法以在线方式因果滑动短窗，却以循环网络形式展开训练，即使点被遮挡或离开视野也能保持长时轨迹。在标准点追踪基准上，CoTracker大幅超越先前方法。
</details>

<details>
    <summary>Key points</summary>
    * 联合追踪机制：同时追踪大量点、建模点间依赖，而非独立追踪。  
    * Transformer 架构 + “token proxies”技术，支持大规模 (~70k 点) 在线追踪。  
    * 在线短窗口执行，训练采用长序列反复展开，增强长时段追踪能力。  
    * 在遮挡、视野外点场景下明显优于传统独立追踪方法。  
</details>
</details>

---

<details>
<summary><b>GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control</b></summary>

* **作者：** Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, Jun Gao
* **arXiv ID:** 2503.03751
* **核心创新：** 图像（首帧/种子帧）→ 视频。3D 缓存 + 精确相机控制，强调世界一致性的视频生成
* **发表于：** CVPR 2025
* **链接：** [[Paper]](https://arxiv.org/abs/2503.03751) | [[PDF]](https://arxiv.org/pdf/2503.03751) | [[Code]](https://github.com/nv-tlabs/GEN3C)

> **核心创新**  
> 用 3D 缓存把相机位姿直接变成可渲染条件，让模型只负责补全新视角下的空缺，从而一次性解决相机控制不准和时序不一致两大痛点。 

<details>
    <summary>Abstract</summary>
    我们提出 GEN3C，一个具备精确相机控制与时序 3D 一致性的生成式视频模型。已有的视频模型虽能生成逼真视频，但往往利用的3D 信息极少，导致时序不一致现象，例如物体突然弹出或消失。即便实现了相机控制，其精度也有限，因为相机参数仅作为神经网络输入，网络必须自行推断视频与相机位姿的依赖关系。
    相比之下，GEN3C 受 3D 缓存引导：该缓存由种子图像或已生成帧的逐像素深度预测得到的点云构成。在生成后续帧时，GEN3C 以用户指定的新相机轨迹下对 3D 缓存的 2D 渲染作为条件。关键在于，GEN3C 既无需记忆先前生成的内容，也无需从相机位姿反推图像结构；模型可集中全部生成能力于此前未观测区域，同时将场景状态推进至下一帧。实验结果表明，GEN3C 在相机控制精度上超越现有方法，并在稀疏视角新视角合成任务中达到SOTA，即便在驾驶场景与单目动态视频等挑战性设定下依然有效。
</details>

<details>
    <summary>Key points</summary>
    * 使用 Depth Anything v2（metric 版本）对单张参考图预测米制深度
    * 将像素回投影到相机空间形成稳定的 3D 表示（点云）
    * 与每个训练视频用 COLMAP 三角化得到的点云进行配准/尺度对齐
    * 将相机轨迹从相对尺度统一到米制尺度
    * 推理时在交互式 3D 点云中绘制并渲染相机轨迹预览
</details>
</details>

---

## 🌟 "三位一体"原型：涌现的世界模型

本节重点介绍展示**三大一致性初步整合**的模型，表现出涌现的世界模型能力。这些系统代表了当前前沿，展现了真正世界模拟的雏形。

### <a href="./consisency-paper/world-models/README_zh.md">文本到世界生成器</a>

从语言描述生成动态、空间一致的虚拟环境的模型。

**关键特征：**
- ✅ 模态一致性：自然语言理解和像素空间生成
- ✅ 空间一致性：具有物体恒存性的 3D 感知场景组合
- ✅ 时间一致性：物理上可信的动力学和运动

#### 代表性工作

<details>
<summary><b>OpenAI Sora</b></summary>

* **Authors:** OpenAI (团队)  
* **Model ID:** Sora (2024)  
* **One-liner:** 支持从文本、图像、视频输入生成新视频的通用视觉数据模型。 
* **Published in:** 2024（开放形式）
* **Links:** [[Model Page]](https://openai.com/sora/) | [[Technical Overview]](https://openai.com/index/video-generation-models-as-world-simulators/)

> **核心创新**  
> Sora 引入了将视觉数据（如视频）视作“补丁（patches）”训练的大规模通用生成模型，支持多种输入模态（例如文本、图像、视频）并输出新视频，时长、分辨率、长短多样。

<details>
    <summary>Abstract</summary>
    Sora 是一种通用视觉数据生成模型——能够生成视频与图像，跨越不同持续时间、纵横比与分辨率。我们借鉴大型语言模型中 token 化方式，将视觉数据转换为补丁表示，并训练一个压缩编码网络以把原始视频降维，再训练对应解码器将生成的 latent 映射回像素空间。
</details>

<details>
    <summary>Key points</summary>
    * 使用补丁（patch）作为视觉数据单位，构建可扩展的生成模型。   
    * 引入视频压缩网络，将原始视频编码为时空 latent 表示。
    * 支持灵活的输入模态（文本、图像、视频）与多样的视频输出格式。  
    * 强调通用性与规模化：目标是构建类似“世界模型”的视觉生成系统。  
</details>
</details>

---

<details>
<summary><b>Runway Gen-3 Alpha</b></summary>

* **Authors:** Runway AI (团队)  
* **Model ID:** Gen-3 Alpha (2024)  
* **One-liner:** 下一代多模态视频生成基础模型，在忠实度、一致性、运动表现方面较 Gen-2 有大幅提升。
* **Published in:** 2024 (Alpha) 
* **Links:** [[Model Page]](https://runwayml.com/research/introducing-gen-3-alpha)

> **核心创新**  
> Gen-3 Alpha 构建于全新大规模多模态训练基础设施之上，支持文本、图像、视频三种输入，并提供更优秀的运动表现、结构一致性与控制能力。

<details>
    <summary>Abstract</summary>
    该模型是 Runway 在其新一代基础模型上的开端，旨在提升视频生成的视觉忠实度、镜头运动一致性以及创意控制能力。用户可从静态图像或文本出发，生成动态视频、控制镜头运动、人物表现等。
</details>

<details>
    <summary>Key points</summary>
    * 支持文本→视频、图像→视频等多模态输入形式。  
    * 引入“Motion Brush”“高级摄像机控制”等工具，提升创意可控性。 
    * 在生成质量、速度、运动流畅度方面比上代显著改进。  
    * 面向创作者与影视制作流程，强调实用性与制作体验。  
</details>
</details>

---

<details>
<summary><b>Pika 1.0</b></summary>

* **Authors:** Pika Labs (团队)  
* **Model ID:** Pika 1.0 (2023)  
* **One-liner:** “想法到视频”平台中推出的版本 1.0 模型，可生成与编辑多种风格（3D 动画、卡通、电影风格）视频。 
* **Published in:** 2023 (11 月) 
* **Links:** [[Product Page]](https://pika.art/) | [[Launch Blog]](https://pika.art/blog) 

> **核心创新**  
> Pika 1.0 将视频生成编辑工具普及化：通过 Web 与 Discord 接入、支持从文本或图像创建视频及编辑已有视频，覆盖多种视觉风格。 

<details>
    <summary>Abstract</summary>
    Pika Labs 在 2023 年 11 月推出其 1.0 版本，引入了一个新的 AI 模型，能够在多种风格下生成视频（包括 3D 动画、动漫、卡通与电影风格），提供全新的网页使用体验。公司同期宣布融资 5500 万美元。用户可从文本或图像提示出发，部分支持已有视频编辑。
</details>

<details>
    <summary>Key points</summary>
    * 支持文本→视频、图像→视频与视频→视频编辑。
    * 多种视觉风格：3D 动画、动漫、卡通、电影级。  
    * 简化创作流程：Web 界面 + Discord 等平台接入，降低视频生成门槛。  
    * 快速增长用户基础：推出半年内已有数十万用户、每周生成百万级视频。 
</details>
</details>


---

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

<details>
    <summary>Abstract</summary>
    我们研究如何将互联网级视觉-语言模型直接融入端到端机器人控制，以提升泛化能力并激发语义推理。目标是让单一端到端模型既能将观测映射为动作，又能享用网络图文预训练成果。为此，我们在机器人轨迹与互联网级视觉-语言任务（如视觉问答）上联合微调前沿视觉-语言模型。不同于其他方法，我们给出简洁通用方案：把动作表示为文本token，与语言token同等对待并纳入训练集。这类模型称为视觉-语言-动作模型（VLA），我们实例化出RT-2。6千次评估表明，该方法催生高性能策略，使RT-2从互联网训练中获得涌现能力：显著泛化到新物体，理解训练数据未见的指令（如把物体放到特定数字或图标上），并执行基础推理（如取最小/最大或离某物最近的物体）。引入思维链后，RT-2可进行多阶段语义推理，例如判断用哪块石头当临时锤子，或给疲惫的人挑能量饮料。
</details>

<details>
    <summary>Key points</summary>
    * 将 VLM 扩展到机器人动作空间  
    * 互联网数据 + 机器人数据联合训练  
    * 输出离散化机器人动作 token 序列  
    * 强零样本操控能力：可执行未示教任务/对象/指令  
</details>
</details>

---

<details>
<summary><b>GAIA-1: A Generative World Model for Autonomous Driving</b></summary>

* **Authors:** Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado
* **arXiv ID:** 2311.07541
* **One-liner:** 面向自动驾驶的生成式世界模型，实现多智能体运动预测与闭环模拟训练。
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2309.17080) | [[PDF]](https://arxiv.org/pdf/2309.17080.pdf)

> **核心创新**  
> 不止是轨迹预测，而是学习“世界演化”——可生成多智能体动态、真实交通行为与传感器信号，用于闭环仿真与安全稀有场景合成。

<details>
    <summary>Abstract</summary>
    自动驾驶有望带来交通领域的变革性改善，但构建能在真实世界复杂无序场景中安全导航的系统仍面临挑战。关键难题在于，如何有效预测随车辆动作演进可能出现的多种潜在结果。为此，我们提出GAIA-1（Generative AI for Autonomy），一个生成式世界模型，它利用视频、文本与动作输入生成逼真驾驶场景，并可精细控制自车行为与场景特征。该方法将世界建模转化为无监督序列建模：把输入映射为离散token，再预测序列中的下一token。模型涌现的能力包括学习高层结构与场景动力学、上下文感知、泛化及几何理解。GAIA-1对未来事件的期望表征与逼真采样能力相结合，为自动驾驶领域开辟新可能，加速并提升自动驾驶技术的训练水平。
</details>

<details>
    <summary>Key points</summary>
    * 生成式世界模型：不仅预测轨迹，还模拟整个交通生态  
    * 支持多智能体时空建模（车-人-交通灯-环境）  
    * 用于闭环 AD 仿真、鲁棒性测试和长尾案例生成  
    * 提高数据效率与泛化能力的同时提升安全性  
</details>
</details>

---

<details>
<summary><b>PaLM-E: An Embodied Multimodal Language Model</b></summary>

* **Authors:** Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence
* **arXiv ID:** 2303.03378
* **One-liner:** 将真实世界传感输入嵌入 PaLM 大模型，实现具身智能中的视觉-语言-机器人统一推理与任务执行。
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2303.03378) | [[PDF]](https://arxiv.org/pdf/2303.03378.pdf) | [[Project Page]](https://palm-e.github.io/)

> **核心创新**  
> 把机器人视觉、状态信息作为“token”输入 LLM，使语言模型成为具身智能决策系统，能理解环境、规划动作并执行任务。

<details>
    <summary>Abstract</summary>
    大语言模型在诸多复杂任务中表现卓越，但在现实世界（如机器人问题）中进行通用推理时，面临 grounding 挑战。我们提出具身语言模型，将连续的实境传感器模态直接融入语言模型，从而建立词汇与感知之间的连接。模型输入为多模态句子，交织视觉、连续状态估计与文本编码。我们与预训练大语言模型端到端联合训练这些编码，用于序列机器人操作规划、视觉问答和描述等具身任务。评估显示，单一大规模具身多模态模型 PaLM-E 可处理多种具身推理任务，适配不同观测模态与 embodiment，并表现出正向迁移：模型受益于互联网级语言、视觉及视觉-语言联合训练。最大的 PaLM-E-562B 含 5620 亿参数，除机器人任务外，还是视觉-语言通才，在 OK-VQA 上达最佳水平，并随规模保持通用语言能力。
</details>

<details>
    <summary>Key points</summary>
    * 将机器人传感信息 token 化输入 LLM  
    * 视觉-语言-动作统一模型  
    * 具备长期规划、任务分解与泛化能力  
    * 推动“机器人 = LLM + 具身输入”研究范式  
</details>
</details>

---

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

<details>
<summary><b>WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation</b></summary>

* **Authors:** Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, Li Yuan
* **arXiv ID:** 2503.07265
* **One-liner:** 提出可感知世界知识的语义一致性评价指标（WISE），用以衡量文生图模型是否真正理解常识与世界事实。
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2503.07265) | [[PDF]](https://arxiv.org/pdf/2503.07265.pdf) | [[Code]](https://github.com/PKU-YuanGroup/WISE)

> **核心创新**  
> WISE 强调“评价应包含世界知识”，构建了一个包含真实世界语义约束、概念关系和常识推理的文生图一致性评估体系，能够显著更准确地衡量文本-图像对齐与理解能力，而非只依赖 CLIP 相似度。

<details>
    <summary>Abstract</summary>
    文本到图像模型能生成高质量的艺术作品与视觉内容，然而现有研究与评估标准主要关注图像逼真度和浅层图文对齐，缺乏对复杂语义理解与世界知识融入的综合衡量。为此，我们提出首个面向世界知识引导语义评估的基准 WISE，超越简单词-像素映射，用涵盖文化常识、时空推理、自然科学 25 子域的 1000 条精心构造提示挑战模型。针对传统 CLIP 指标局限，我们提出新的量化度量 WIScore 评估知识-图像对齐度。通过对 20 个模型（10 个专用 T2I 与 10 个统一多模态）在 1000 条结构化提示上的全面测试，发现其生成图像时整合并应用世界知识的能力显著不足，为下一代模型增强知识融入指明关键路径。代码与数据见 https 链接。
</details>

<details>
    <summary>Key points</summary>
    * 引入“世界知识 + 常识推理”评估框架  
    * 评价维度：实体正确性、关系合理性、属性一致性、场景常识  
    * 与人工偏好高度相关，优于 CLIPScore/BERTScore 等指标  
    * 可用作未来文生图模型 benchmark 的通用评价层  
</details>
</details>

---

<details>
<summary><b>Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark</b></summary>

* **Authors:** Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, Pheng-Ann Heng
* **arXiv ID:** 2510.26802
* **One-liner:** 构建 MME-COF 基准测试多模态视频模型的零样本常识与因果推理能力，发现当前视频模型仍缺乏真正时序-因果理解。
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2510.26802) | [[PDF]](https://arxiv.org/pdf/2510.26802.pdf) | [[Code]](https://github.com/ZiyuGuo99/MME-CoF)

> **核心创新**  
> 首次系统评估视频大模型的“零样本推理”能力，提出 MME-COF 基准（Memory, Motion, Event, Causality, Object, Future prediction），发现视频模型在细粒度因果理解方面显著落后于视觉-语言大模型。

<details>
    <summary>Abstract</summary>
    最新视频生成模型可产出高保真、时序连贯的视频，暗示其编码了丰富的世界知识。除逼真合成外，它们还表现出视觉感知、建模与操控的涌现行为。然而，一个重要问题仍未解答：在挑战性视觉推理场景中，视频模型是否已具备零样本推理能力？为此，我们对领先且流行的 Veo-3 开展实证研究，从空间、几何、物理、时间、具身逻辑等 12 个维度系统评估其推理表现，刻画优势与失效模式。为标准化评估，我们构建紧凑基准 MME-CoF，深入测评逐帧链式（CoF）推理。结果显示，当前视频模型在短程空间连贯、细粒度定位与局部一致动态上展现可喜模式，但在长程因果、严格几何与抽象逻辑上仍受限。总体而言，它们尚不足以作为独立零样本推理器，但已显露作为专用推理模型补充视觉引擎的潜力。
</details>

<details>
    <summary>Key points</summary>
    * MME-COF 六大测试维度：对象识别、记忆、动作、事件、因果、未来预测  
    * 面向“零样本”评估，无视频任务微调  
    * 视频模型 vs 文本-图像模型：前者推理能力不足  
    * 揭示了当前视频模型更偏“模式拟合”而非“语义推理”  
</details>
</details>

---

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
