# Awesome World Model Evolution - 铸向世界模型宇宙：基于统一多模态模型的融合之路 [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

<div align="center">
  <img src="https://img.shields.io/github/stars/opendatalab-raiser/awesome-world-model-evolution" alt="Stars">
  <img src="https://img.shields.io/github/forks/opendatalab-raiser/awesome-world-model-evolution" alt="Forks">
  <img src="https://img.shields.io/github/license/opendatalab-raiser/awesome-world-model-evolution" alt="License">
  <img src="https://img.shields.io/github/last-commit/opendatalab-raiser/awesome-world-model-evolution" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</div>

<p align="center">
  <strong>精心策划的研究论文、模型和资源合集，追溯从专用模型到统一世界模型的演化历程。</strong>
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
      - [代表性工作：](#代表性工作)
    - [空间一致性](#空间一致性)
      - [代表性工作：](#代表性工作-1)
    - [时间一致性](#时间一致性)
      - [代表性工作：](#代表性工作-2)
  - [🔗 初步融合：统一多模态模型](#-初步融合统一多模态模型)
    - [模态 + 空间一致性](#模态--空间一致性)
      - [代表性工作：](#代表性工作-3)
    - [模态 + 时间一致性](#模态--时间一致性)
      - [代表性工作：](#代表性工作-4)
    - [空间一致性 + 时间一致性](#空间一致性--时间一致性)
      - [代表性工作](#代表性工作-5)
  - [🌟 "三位一体"原型：涌现的世界模型](#-三位一体原型涌现的世界模型)
    - [文本到世界生成器](#文本到世界生成器)
      - [代表性工作](#代表性工作-6)
    - [具身智能系统](#具身智能系统)
      - [代表性工作：](#代表性工作-7)
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

### 模态一致性

**目标**：在符号（语言）和感知（视觉）表示之间建立双向映射。

**历史意义**：这些模型创建了首批"符号-感知桥梁"，解决了世界模型的基本输入/输出问题。

#### 代表性工作：

**CLIP: Learning Transferable Visual Models From Natural Language Supervision**

* **Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
* **arXiv ID:** 2103.00020
* **One-liner:** Large-scale contrastive learning on 400M image–text pairs enabling strong zero-shot transfer across vision tasks
* **Published in:** ICML 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2103.00020) | [[PDF]](https://arxiv.org/pdf/2103.00020.pdf) | [[Code]](https://github.com/openai/CLIP)

> **核心创新**  
> 基于4亿图文对进行对比学习，建立统一的视觉-语言表征空间，实现零样本跨任务迁移能力，首创“文本作为监督信号”的大规模多模态训练范式。

<details>
    <summary>Abstract</summary>
    CLIP 通过对比学习联合训练图像编码器与文本编码器，使用来自互联网的4亿图文对作为弱监督数据。模型展现出极强的泛化能力，无需微调即可在多个视觉任务中实现接近监督方法的表现。
</details>

<details>
    <summary>Key points</summary>
    * 图像与文本特征的对比学习  
    * 大规模弱监督网络数据（4亿对）  
    * 通过文本提示进行零样本分类  
    * 引发多模态基础模型浪潮  
</details>

---

**DALL-E: Zero-Shot Text-to-Image Generation**

* **Authors:** Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever
* **arXiv ID:** 2102.12092
* **One-liner:** First large transformer trained to generate images from text, demonstrating compositional text-to-image creativity
* **Published in:** ICML 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2102.12092) | [[PDF]](https://arxiv.org/pdf/2102.12092.pdf) | [[Project Page]](https://openai.com/research/dall-e)

> **核心创新**  
> 首次将大规模Transformer应用于文本到图像生成，搭配离散VAE编码，展示了语义组合生成（形状+风格+属性）的强能力。

<details>
    <summary>Abstract</summary>
    DALL-E 是一个拥有120亿参数的 Transformer 模型，利用离散VAE图像tokens，从文本提示生成图像。其表现出强大的组合性与视觉创造能力，能够根据多样且复杂的指令生成连贯图像。
</details>

<details>
    <summary>Key points</summary>
    * 文本条件 Transformer 用于图像生成  
    * 使用 VQ-VAE 进行图像离散化编码  
    * 具备组合推理能力（内容+风格+属性）  
    * 预示 Prompt 驱动生成新时代  
</details>

---

**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**

* **Authors:** Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio
* **arXiv ID:** 1502.03044
* **One-liner:** First visual attention mechanism for image captioning; pioneered attention in vision models
* **Published in:** ICML 2015
* **Links:** [[Paper]](https://arxiv.org/abs/1502.03044) | [[PDF]](https://arxiv.org/pdf/1502.03044.pdf)

> **核心创新**  
> 首次在图像描述任务中引入视觉注意力机制，使模型能聚焦关键区域并生成解释性注意图，为后续视觉Transformers奠基。

<details>
    <summary>Abstract</summary>
    本工作提出软/硬两种注意力机制用于图像描述，使模型在生成语言过程动态关注图像关键区域，从而输出更准确的自然语言描述。
</details>

<details>
    <summary>Key points</summary>
    * 视觉领域的早期注意力机制  
    * CNN 编码 + RNN 解码架构  
    * 注意力热力图带来可解释性  
    * 为视觉 Transformer 技术铺路  
</details>

---

**AttnGAN: Fine-Grained Text-to-Image Generation with Attentional GANs**

* **Authors:** Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He
* **arXiv ID:** 1711.10485
* **One-liner:** Word-region attention and semantic alignment loss for high-fidelity text-to-image synthesis
* **Published in:** CVPR 2018
* **Links:** [[Paper]](https://arxiv.org/abs/1711.10485) | [[PDF]](https://arxiv.org/pdf/1711.10485.pdf) | [[Code]](https://github.com/taoxugit/AttnGAN)

> **核心创新**  
> 通过单词级注意力和语义对齐损失实现细粒度文图对应，大幅提升文本生成图像的清晰度与语义一致性。

<details>
    <summary>Abstract</summary>
    AttnGAN 利用精细的单词-区域注意力与 DAMSM 语义匹配损失提升生成质量，使得生成图像在细节和语义一致性方面均显著改善。
</details>

<details>
    <summary>Key points</summary>
    * 基于单词的细粒度文本注意力  
    * 多阶段逐步细化图像生成  
    * DAMSM 损失强化文图语义一致性  
    * 扩展GAN在文本生成图像方向的上限  
</details>

---

### 空间一致性

**目标**：使模型能够从二维观察中理解和生成三维空间结构。

**历史意义**：为构建内部"3D 场景图"和理解几何关系提供了方法论。

#### 代表性工作：

**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**

* **Authors:** Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
* **arXiv ID:** 2003.08934  
* **One-liner:** Represent a scene as a continuous volumetric neural radiance field and render novel views via volume rendering of the MLP output  
* **Published in:** ECCV 2020
* **Links:** [[Paper]](https://arxiv.org/abs/2003.08934) | [[PDF]](https://arxiv.org/pdf/2003.08934.pdf)  

> **核心创新**  
> 将场景用一个 MLP 表示为“神经辐射场”（Neural Radiance Field, NeRF）：输入任意 5D 坐标（x,y,z,θ,φ）输出该位置的体密度 + 视角依赖的辐射量，再通过可微体渲染 (volume rendering) 实现从新视角生成图像。  

<details>
    <summary>Abstract</summary>
    本文提出一种方法，通过优化一个底层的连续体场函数（神经网络）来合成复杂场景的新视角图像。网络输入为一个连续 5D 坐标（空间位置 x,y,z 以及视角方向 θ,φ），输出该位置的体密度和视角依赖的发射辐射。我们沿摄像机射线查询这些 5D 坐标，并使用经典体渲染技术将颜色和密度投影成图像。因为体渲染是可微的，唯一需要的输入是已知相机姿态的一组图像。我们描述了如何有效优化神经辐射场以渲染具有复杂几何与外观的场景，并展示了优于此前神经渲染与视角合成方法的结果。  
</details>

<details>
    <summary>Key points</summary>
    * 使用 MLP 来表示场景，输入 (x,y,z,θ,φ)，输出体密度 + 辐射。  
    * 通过体渲染 (volume rendering) 将网络输出合成为图像。  
    * 无需构建显式几何重建，只用多视角图像和相机位姿即可优化。  
    * 展示出高质量的新视角合成效果，推动了视图合成和隐式表示的发展。  
</details>

---

**3D Gaussian Splatting for Real-Time Radiance Field Rendering**

* **Authors:** Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
* **arXiv ID:** 2308.04079
* **One-liner:** Use anisotropic 3D Gaussians + splatting renderer to achieve real-time (>=30 fps) high-quality novel-view synthesis of large scenes.  
* **Published in:** SIGGRAPH 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2308.04079) | [[PDF]](https://arxiv.org/pdf/2308.04079.pdf) | [[Code]](https://github.com/graphdeco-inria/gaussian-splatting)

> **核心创新**
> 提出用成千上万的各向异性 3D 高斯分布（3D Gaussians）而非纯 MLP 表征场景，再结合一个可视性感知的高效 splatting 渲染器，实现大场景下的实时 (1080p, >=30fps) 新视角合成。  

<details>
    <summary>Abstract</summary>
    辐射场方法近年来改变了使用多照片或视频捕捉场景并合成新视角的方式。但要达到高质量仍需神经网络进行训练和渲染，而最近的加速方法往往牺牲质量。对于非封闭、完整场景（不是孤立物体）和 1080p 分辨率渲染，目前还没有方法能实现实时显示。我们引入三个关键元素，使我们在保持训练时间竞争力的同时，实现高质量实时（≥30fps）1080p 新视角合成。首先，从相机标定得到的稀疏点出发，用 3D 高斯表示场景，这保留了连续体辐射场的优良属性，同时避免在空白空间做不必要计算；其次，我们执行交叉优化／密度控制（尤其优化各向异性协方差）以准确表示场景；第三，我们开发了一个快速的可视性感知渲染算法，支持各向异性 splatting，同时加速训练并允许实时渲染。我们在多个既定数据集上展示了高质量效果与实时渲染能力。  
</details>

<details>
    <summary>Key points</summary>
    * 使用 3D 高斯而非传统隐式 MLP 表示场景，避开空白空间浪费。  
    * 对每个高斯分布使用各向异性协方差优化，提高场景表达精度。  
    * 引入可视性感知的 splatting 渲染器，实现实时新视角合成。  
    * 在大场景、1080p 分辨率下实现 ≥30 fps 的实时渲染，并且质量与此前高质量方法可比。  
</details>

---

**EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks**

* **Authors:** Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, Gordon Wetzstein 
* **arXiv ID:** 2112.07945  
* **One-liner:** Hybrid explicit-implicit 3D GAN (tri-plane) that synthesizes high-resolution, multi-view consistent imagery and high-quality geometry in real time.  
* **Published in:** CVPR 2022  
* **Links:** [[Paper]](https://arxiv.org/abs/2112.07945) | [[PDF]](https://arxiv.org/pdf/2112.07945.pdf) | [[Code]](https://github.com/NVlabs/eg3d)

> **核心创新**  
> 引入基于三平面 (tri-plane) 的混合显式‐隐式网络架构，使 3D GAN 既具备高效计算、又能生成高分辨率、多视角一致性的图像和优质几何形状。

<details>
    <summary>Abstract</summary>
    本文我们提升了 3D GAN 的计算效率与图像质量，同时不过度依赖近似。为此我们引入一种表达能力强的混合显式‐隐式网络架构，结合其他设计选择，不仅实时合成高分辨率且多视角一致的图像，还生成高质量 3D 几何。通过将特征生成与神经渲染解耦，我们的框架能够利用如 StyleGAN2 等最先进的 2D CNN 生成器，并继承其高效与表达力。
</details>

<details>
    <summary>Key points</summary>
    * 三平面 (tri-plane) 表示：将场景编码到三组 2D 隐式特征平面，从而高效查询 3D 特征。  
    * 解耦特征生成与神经渲染，允许使用强大的 2D 生成器架构。  
    * 实时生成高分辨率且多视角一致的图像，并产出优质几何。  
    * 提升 3D GAN 的效率与视觉质量，推动 3D 世界生成研究。  
</details>

---

**Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**

* **Authors:** Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller
* **arXiv ID:** 2201.05989
* **One-liner:** Use a multiresolution hash table encoding to massively speed up training and inference of neural graphics primitives (NeRF/SDF etc.).  
* **Published in:** SIGGRAPH 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2201.05989) | [[PDF]](https://arxiv.org/pdf/2201.05989.pdf) | [[Code]](https://github.com/NVlabs/instant-ngp)

> **核心创新**  
> 提出了一个多分辨率哈希编码 (multiresolution hash encoding)，将坐标映射到可训练特征向量并与小型网络联合使用，从而在保持质量的同时，大幅降低 FLOPs 和内存访问，从而实现“秒级训练”、毫秒渲染的神经图形原语。  

<details>
    <summary>Abstract</summary>
    神经图形原语（neural graphics primitives），由全连接神经网络参数化，训练和评估成本较高。我们通过一种通用的新输入编码降低了该成本：一个小神经网络被一个可训练的多分辨率哈希表特征向量所增强，其值通过随机梯度下降进行优化。多分辨率结构允许网络消解哈希冲突，从而形成一个简单且易于在现代 GPU 上并行化的架构。我们利用这种并行性，通过完全融合的 CUDA 核心实现系统，注重最小化带宽与计算浪费。我们达到数个数量级的速度提升，使得高质量神经图形原语的训练可以在数秒内完成，渲染在几十毫秒级别即可。  
</details>

<details>
    <summary>Key points</summary>
    * 多分辨率哈希编码：将坐标映射到多个尺度的哈希表特征向量。  
    * 小型神经网络 + 哈希特征表：保持高质量同时大幅降低计算与内存访问。  
    * 完全 GPU 并行化（融合 CUDA 核心）以最大化效率。  
    * 实现“瞬时”训练与极快推理（神经渲染毫秒级），推动隐式表示与神经渲染实时化。  
</details>


---

### 时间一致性

**目标**：建模视频序列中的时间动态、物体运动和因果关系。

**历史意义**：对世界"物理引擎"的早期探索，捕捉场景随时间演化的规律。

#### 代表性工作：

**PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning**

* **Authors:** Yunbo Wang, Haixu Wu, Jianjin Zhang, Zhifeng Gao, Jianmin Wang, Philip S. Yu, Mingsheng Long
* **arXiv ID:** 2103.09504  
* **One-liner:** 新的循环结构解耦时空记忆单元，用于视频帧的时空预测学习  
* **Published in:** TPAMI 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2103.09504) | [[PDF]](https://arxiv.org/pdf/2103.09504.pdf) | [[Code]](https://github.com/thuml/predrnn-pytorch)

> **核心创新**  
> 提出将记忆单元对(decoupled memory cells)融入 RNN 架构，以分离地捕捉空间外观与时间动态，再统一形成复杂环境表示。此外引入“之”字形（zig-zag）记忆流上下层传播，以及课程化学习策略以学习长期时动态。 

<details>
    <summary>Abstract</summary>
    时空序列的预测学习旨在通过历史帧学习生成未来图像，其中视觉动态被认为包含可组合的模块结构。本文通过提出 PredRNN 来建模这些结构。在该网络中，一对记忆单元被显式解耦，以近乎独立的转移方式操作，并最终形成复杂环境的统一表示。具体地，除了传统 LSTM 的记忆单元外，本网络具有一个在所有层中上下传播的 zig-zag 记忆流，允许不同级别 RNN 的视觉动态相互通信。它还采用记忆解耦损失以防止记忆单元学习冗余特征。我们进一步提出一种新的课程学习策略，促使 PredRNN 从上下文帧中学习长期动态。我们在五个数据集上（包含动作自由和动作条件情景）验证了各组件的有效性。  
</details>

<details>
    <summary>Key points</summary>
    * 引入记忆对结构：两记忆单元解耦转移分别捕捉空间外观与时间动态。   
    * zig-zag 记忆流：跨层上下传播，促进不同时间层次动态通信。  
    * 课程化学习策略（curriculum learning）：从上下文帧学习长期预测。  
    * 在多个基准数据集上取得竞争性预测性能。  
</details>

---

**SimVP: Simpler yet Better Video Prediction**

* **Authors:** Zhangyang Gao, Cheng Tan, Lirong Wu, Stan Z. Li
* **arXiv ID:** 2206.05099  
* **One-liner:** 完全基于 CNN 的视频预测模型，少即是多，简而优于复杂结构  
* **Published in:** CVPR 2022  
* **Links:** [[Paper]](https://arxiv.org/abs/2206.05099) | [[PDF]](https://arxiv.org/pdf/2206.05099.pdf) | [[Code]](https://github.com/A4Bio/SimVP)

> **核心创新**  
> 提出一个极简的端到端视频预测框架，仅用 CNN 编码—解码结构，训练损失为 MSE，无需 RNN、Transformer、复杂机制即可在多个基准数据集上达到或超越更复杂模型。 

<details>
    <summary>Abstract</summary>
    从 CNN、RNN 到 ViT，我们见证了视频预测的显著进步，涉及辅助输入、精巧网络结构与复杂训练策略。但我们思考：是否有一个简单方法也能取得可比成绩？本文提出 SimVP：一个完全基于 CNN 且端到端训练的简单视频预测模型。它未引入额外技巧和复杂策略，却在五个基准数据集上取得最先进表现。通过扩展实验，我们展示 SimVP 在真实数据集上具有强泛化与可扩展性。训练成本显著降低，更易于扩展至复杂场景。我们认为 SimVP 可作为视频预测的坚实基线。  
</details>

<details>
    <summary>Key points</summary>
    * 完全用卷积编码器–解码器架构（无 RNN、无 Transformer）
    * 用标准 MSE 训练，无额外复杂策略。  
    * 显示出更简单结构也能实现竞争/领先性能。  
    * 降低架构复杂度与计算开销，更适用于实际场景。  
</details>

---

**Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning**

* **Authors:** Cheng Tan, Zhangyang Gao, Lirong Wu, Yongjie Xu, Jun Xia, Siyuan Li, Stan Z. Li
* **arXiv ID:** 2206.12126  
* **One-liner:** 引入可并行的时域注意力模块（TAU），分离帧内／帧间注意力以提升时空预测效率  
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2206.12126) | [[PDF]](https://arxiv.org/pdf/2206.12126.pdf)

> **核心创新**  
> 提出 Temporal Attention Unit (TAU)，将时空预测模块中的“帧内静态注意力”与“帧间动态注意力”分离，支持并行化计算；并且通过差分散度正则化(differential divergence regularization)增强帧间变化捕捉。  

<details>
    <summary>Abstract</summary>
    时空预测学习旨在通过历史帧生成未来帧。本论文探讨现有方法，并提出一个通用框架，其中空间编码器与解码器负责帧内特征，时域模块负责帧间关联。主流方法采用循环单元捕捉长期时依赖，但架构不可并行、效率低。为并行化时模块，我们提出 Temporal Attention Unit (TAU)，它将时间注意力分解为帧内静态注意力与帧间动态注意力。同时，由于 MSE 损失专注于帧内误差，我们引入差分散度正则化以考虑帧间变化。大量实验表明所提方法使模型在多个时空预测基准上取得竞争性能。  
</details>

<details>
    <summary>Key points</summary>
    * 提出 TAU：将时域注意力分为帧内静态与帧间动态两部分，可并行执行。  
    * 引入差分散度正则（differential divergence regularisation）以增强帧间变化捕捉。  
    * 构建空间编码–解码模块 + 时间模块的解耦架构，提升可扩展性。  
    * 显著提高计算效率，同时在预测准确性上保持竞争水平。  
</details>

---

**VideoGPT: Video Generation using VQ-VAE and Transformers**

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

---

**Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions**

* **Authors:** Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, Dumitru Erhan
* **arXiv ID:** 2210.02399  
* **One-liner:** 从开放域文本提示生成可变长度视频：文本→离散视频 token →解码为帧  
* **Published in:** ICLR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2210.02399) | [[PDF]](https://arxiv.org/pdf/2210.02399.pdf)

> **核心创新**  
> 提出将视频表示离散化为 token 的 tokenizer（支持可变长度视频）、并用条件遮蔽 Transformer 生成这些 token，再解码生成帧。通过联合训练大规模图文数据和少量视频-文对，实现开放域文本提示下任意长度视频生成。  

<details>
    <summary>Abstract</summary>
    我们提出 Phenaki，一种模型，能够根据一系列文本提示生成真实视频。由于从文本生成视频面临计算成本高、文本-视频数据稀缺和可变长度视频等挑战。为解决这些问题，我们提出一种新的视频表示学习模型，将视频压缩为少量离散 token 表示。该 tokenizer 使用时间方向的因果注意力，从而支持可变长度视频。为从文本生成视频 token，我们使用一个在预计算文本 token 条件下的双向遮蔽 Transformer。生成的视频 token 随后被解码为实际视频。为应对数据匮乏问题，我们演示了如何在大量图文对与少量视频-文本对上联合训练，从而实现超越视频数据集本身的泛化。与先前视频生成方法相比，Phenaki 可以根据一系列（即时间可变）文本提示生成任意长度的视频。  
</details>

<details>
    <summary>Key points</summary>
    * 将视频表示离散化为 token，实现可变长度视频输出。  
    * 使用条件遮蔽 Transformer 在文本 token 条件下生成视频 token，然后解码为帧。  
    * 联合训练图文大数据与视频-文本少数据以提升泛化。  
    * 支持开放域文本提示生成任意长度视频，而不仅限于固定短片。  
</details>

---

## 🔗 初步融合：统一多模态模型

当前最先进的模型正开始打破各个一致性之间的壁垒。本节展示成功整合**三大一致性中的两项**的模型，它们代表了通向完整世界模型的关键中间步骤。

### 模态 + 空间一致性

**能力特征**：能够将文本/图像描述转换为空间连贯的 3D 表示或多视角一致输出的模型。

**意义**：这些模型展示了"3D 想象力"——它们不再是简单的"2D 画家"，而是理解空间结构的"数字雕塑家"。

#### 代表性工作：

**Zero-1-to-3: Zero-shot One Image to 3D Object**

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

---

**MVDream: Multi-view Diffusion for 3D Generation**

* **Authors:** Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, Xiao Yang
* **arXiv ID:** 2308.16512  
* **One-liner:** 一个多视角扩散模型：从文本提示生成几何一致的多视角图像，从而用于 3D 生成  
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2308.16512) | [[PDF]](https://arxiv.org/pdf/2308.16512.pdf) | [[Code]](https://github.com/bytedance/MVDream)

> **核心创新**  
> 提出一个可生成几何一致多视角图像的扩散模型，通过结合2D图像预训练和3D数据微调，使其既具有 2D 扩散模型的泛化能力 又具有 3D 渲染的一致性，从而可作为 3D 内容生成的通用先验。

<details>
    <summary>Abstract</summary>
    我们提出 MVDream，一种扩散模型，能够根据给定文本提示生成几何一致的多视角图像。通过学习2D和3D数据，该多视角扩散模型兼具2D扩散模型的泛化能力与3D渲染的一致性。我们证明这种多视角扩散模型隐式地成为一个与 3D 表示无关的可泛化 3D 先验。它可通过 Score Distillation Sampling 应用于 3D 生成大幅提升几何一致性，且支持少量示例微调实现个性化 3D 生成。  
</details>

<details>
    <summary>Key points</summary>
    * 将 web 规模 2D 扩散模型预训练 + 多视角 3D 资产渲染数据微调结合。  
    * 条件生成：根据文本提示生成一组多视角图像，保证视角间几何一致性。  
    * 将该模型作为 3D 生成的先验，并通过 SDS（Score Distillation Sampling）提升经典 2D → 3D 方法的稳定性。  
    * 支持少样本微调（如 DreamBooth3D）以实现个性化 3D 生成。  
</details>

---

**Wonder3D: Single Image to 3D using Cross-Domain Diffusion**

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

---

**SyncDreamer: Generating Multiview-consistent Images from a Single-view Image**

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

---

### 模态 + 时间一致性

**能力特征**：将文本描述或静态图像转换为时间连贯的动态视频序列的模型。

**意义**：目前最突出的融合方向，实现高质量的文本到视频和图像到视频生成。

#### 代表性工作：

**Lumiere: A Space-Time Diffusion Model for Video Generation**

* **Authors:** Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, Inbar Mosseri
* **arXiv ID:** 2401.12945 
* **One-liner:** 一种文本到视频扩散模型，通过 Space-Time U-Net 一次生成完整视频时长，强化运动连贯性  
* **Published in:** SIGGRAPH-ASIA 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2401.12945) | [[PDF]](https://arXiv.org/pdf/2401.12945.pdf) | [[Project Page]](https://lumiere-video.github.io/)

> **核心创新**  
> 提出一种 Space-Time U-Net 架构，可在一次模型运行中同时处理空间与时间维度，生成完整时长视频，从而克服此前多阶段关键帧＋时间超分辨 cascade 所带来的时间一致性问题。 

<details>
    <summary>Abstract</summary>
    我们提出 Lumiere — 一种文本到视频的扩散模型，旨在合成表现出真实、多样和连贯运动的视频。为此，我们引入 Space-Time U-Net 架构，该架构通过一次模型前向实现完整视频时长的生成。与现有视频模型通常首先合成关键帧然后再做时间超分辨处理的方式不同，该设计直接从空间‐时间多尺度下采样与上采样生成低分辨率整色视频。我们在多个任务上展示最先进的文本-视频生成效果，并显示该设计轻松支持图像-到-视频、视频修补、风格化生成等内容创作任务。
</details>

<details>
    <summary>Key points</summary>
    * 引入 Space-Time U-Net，可同时进行空间与时间的下/上采样一次生成整色视频。  
    * 避免传统关键帧 + 逐步超分辨策略，从根源提升时间一致性。  
    * 利用预训练文本到图像扩散模型结构再扩展至视频生成。  
    * 支持多任务：文本到视频、图像到视频、视频修补、风格化生成。  
</details>

---

**Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets**

* **Authors:** Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach
* **arXiv ID:** 2311.15127 
* **One-liner:** 隐空间视频扩散模型，拓展至大规模数据集，实现高分辨率文本→视频与图像→视频生成  
* **Published in:** arXiv 2023  
* **Links:** [[Paper]](https://arxiv.org/abs/2311.15127) | [[PDF]](https://arxiv.org/pdf/2311.15127.pdf) | [[Code]](https://github.com/Stability-AI/generative-models)

> **核心创新**  
> 系统提出训练隐空间视频扩散模型的三阶段流程（文本→图像预训练、视频预训练、高质量视频微调）并强调数据集筛选与注释的重要性，从而在大规模视频数据上实现高分辨率生成与多视角 3D 先验。   

<details>
    <summary>Abstract</summary>
    我们提出 Stable Video Diffusion — 一种用于高分辨率文本→视频和图像→视频生成的隐空间视频扩散模型。近期，将用于2D图像合成的隐空间扩散模型通过插入时间层并在小规模高质量视频数据集上微调后，已转向视频生成。但训练方法差异大，数据集整理尚无统一策略。本文我们识别并评估了成功训练视频 LDM（latent diffusion models）三阶段：文本→图像预训练、视频预训练、高质量视频微调；进一步展示了精心筛选的视频数据对生成质量的重要性。我们训练了一个文本→视频模型，与闭源视频生成模型竞争，并展示了其强大的多视角 3D 先验能力，可作为图像→视频与摄像机运动控制下的生成基底。 
</details>

<details>
    <summary>Key points</summary>
    * 提出三阶段训练流程：文本→图像预训练 → 视频预训练 → 高质量视频微调。  
    * 强调数据集策划（captioning & filtering）对生成质量的影响。  
    * 模型兼具图像→视频和文本→视频能力，并提供多视角 3D 模型潜力。  
    * 展示在大规模视频生成任务上的效率与效果提升。  
</details>

---

**AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning**

* **Authors:** Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, Bo Dai
* **arXiv ID:** 2307.04725 
* **One-liner:** 插入可复用运动模块将任何个性化文本→图像扩散模型转变为动画生成器，无需模型专用调优  
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.04725) | [[PDF]](https://arxiv.org/pdf/2307.04725.pdf) | [[Code]](https://github.com/guoyww/AnimateDiff)

> **核心创新**  
> 提出一个可插拔的“运动模块”（motion module），在冻结的文本→图像扩散模型基础上训练一次后即可无缝集成，实现在已有图像生成模型上直接生成动画。并提出 MotionLoRA 轻量微调方案适配新的运动模式。

<details>
    <summary>Abstract</summary>
    随着文本→图像扩散模型（如 Stable Diffusion）与个性化技术（如 DreamBooth、LoRA）进步，每个人都可低成本生成高质量图像。但将已有高质个性化 T2I 模型扩展至动画仍然具有挑战。本文提出 AnimateDiff：一个实用框架，将任何个性化 T2I 模型转变为动画生成器，无需模型-特定微调。核心为一个训练一次即可适用于所有个性化 T2I 模型的可插拔运动模块。我们通过训练策略使该模块从真实视频中学习可迁移运动先验。实验证明可生成时序平滑、图像质量高、运动多样的动画片段。 
</details>

<details>
    <summary>Key points</summary>
    * 插入可复用运动模块，可直接嵌入任意个性化 T2I 模型。  
    * MotionLoRA：轻量级微调方案，快速适配新运动模式。  
    * 保持原图像生成模型的高质量输出，同时赋予运动能力。  
    * 简化动画生成流程，降低模型-调优负担。  
</details>

---

**Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning**

* **Authors:** Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, Ishan Misra
* **arXiv ID:** 2311.10709
* **One-liner:** 采用分两步生成流程：文本→图像、再图像+文本→视频，从而实现高质量文本→视频生成  
* **Published in:** ECCV 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2311.10709) | [[PDF]](https://arxiv.org/pdf/2311.10709.pdf) | [[Project Page]](https://emu-video.metademolab.com/)

> **核心创新**  
> 提出生成流程的因式分解（factorization）：先生成图像（text-conditioned），再基于该图像和文本生成视频。并设计特定噪声调度和多阶段训练策略，突破以往深度模型级联限制，直接生成高质量高分辨率视频。   

<details>
    <summary>Abstract</summary>
    我们提出 Emu Video — 一种文本→视频生成模型，将生成流程因式分解为两步：首先基于文本生成图像，然后在文本与生成图像条件下生成视频。我们识别出关键设计决策——调整扩散噪声调度和多阶段训练，使我们可以直接生成高质量、高分辨率视频，而无需如以往那样依赖深度模型级联。在人工评估中，我们生成的视频在人类偏好中大幅领先于先前工作。模型还可用于基于文本提示或用户提供图像的动画生成。 
</details>

<details>
    <summary>Key points</summary>
    * 生成流程分两步：文本→图像 →（文本＋图像）→视频。  
    * 调整扩散模型的噪声调度与多阶段训练流程。  
    * 实现高质量、高分辨率视频生成，并提升用户图像动画能力。  
    * 在人工评测中优于先前文本→视频方法。  
</details>

---

**VideoPoet: A Large Language Model for Zero-Shot Video Generation**

* **Authors:** Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, Lu Jiang
* **arXiv ID:** 2312.14125
* **One-liner:** 一个解码器-仅 Transformers 架构的大语言模型，可零样本生成视频，支持多模态条件输入（文本、图像、音频）  
* **Published in:** ICML 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14125) | [[PDF]](https://arxiv.org/pdf/2312.14125.pdf) | [[Project Page]](https://sites.research.google/videopoet/)

> **核心创新**  
> 将大语言模型训练范式扩展至视频生成：使用解码器-仅 Transformer 架构处理多模态输入（文本、图像、音频、视频）并在零样本设置下生成高质量视频（含匹配音频），开拓视频生成的新路径。 

<details>
    <summary>Abstract</summary>
    我们提出 VideoPoet，一种能从多种条件信号中生成高质量视频（含匹配音频）的模型。VideoPoet 采用解码器-仅 Transformer 架构，处理包括图像、视频、文本和音频在内的多模态输入。训练协议遵循大语言模型（LLM）范式，包含预训练与任务特定适配阶段。实验证明模型在零样本视频生成任务上展示了最先进能力，尤其在大幅运动和多模态条件下表现出色。  
</details>

<details>
    <summary>Key points</summary>
    * 使用解码器-仅 Transformer，输入为多模态（文本、图像、视频、音频）。  
    * 训练流程遵循 LLM：大规模预训练 + 任务适配阶段。  
    * 支持零样本视频生成，满足不同条件输入。  
    * 展现多模态视频生成新范式，推动视频生成研究。  
</details>

---

### 空间一致性 + 时间一致性

**能力特征**：这类模型能够在模拟时间动态演化的同时保持三维空间结构的一致性，但可能在语言理解或可控性方面存在一定局限。

**意义**：这些模型代表了理解"三维世界如何运动"的关键技术成就，构成了世界模型的物理引擎组成部分。

#### 代表性工作

**DUSt3R: Geometric 3D Vision Made Easy**

* **Authors:** Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud
* **arXiv ID:** 2312.14132  
* **One-liner:** 无需相机标定与外参，即可实现任意图像集合的密集、无约束立体 3D 重建  
* **Published in:** CVPR 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14132) | [[PDF]](https://arxiv.org/pdf/2312.14132.pdf)

> **核心创新**  
> 提出一种颠覆传统多视角立体重建（MVS）流程的新范式——直接回归稠密点图（point-maps）而省略相机内参／外参估计。该方法统一单视图与多视图深度估计、相机位姿估计与重建任务，在“采集环境随意”的野外场景下也能表现优异。

<details>
    <summary>Abstract</summary>
    我们提出 DUSt3R，一种面向任意图像集合的“密集且无约束”的立体 3D 重建方法。传统野外多视角立体重建 (MVS) 必须先估计相机的内参与外参，这既繁琐又必须，而这恰是许多顶尖算法的核心。本文反其道而行，将问题重新表述为稠密点图的回归，从而完全省略相机校准与位姿估计。我们展示了该方法在深度估计、位姿估计及重建任务上均优于现有方法。
</details>

<details>
    <summary>Key points</summary>
    * 将多视角立体重建问题转化为点图回归，无需显式三角化与相机矩阵。  
    * 统一框架处理单视图、两视图及多视图场景。  
    * Transformer-基模型直接从图像预测稠密 3D 几何，跳过传统 SfM／MVS 模块。  
    * 在深度、位姿与重建任务上取得新的性能标杆。  
</details>

---

**4D Gaussian Splatting for Real-Time Dynamic Scene Rendering**

* **Authors:** Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang
* **arXiv ID:** 2310.08528  
* **One-liner:** 引入 4D 高斯 splatting 表示法，用于动态场景的实时渲染  
* **Published in:** CVPR 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2310.08528) | [[PDF]](https://arxiv.org/pdf/2310.08528.pdf) | [[Code]](https://github.com/hustvl/4DGaussians)

> **核心创新**  
> 提出 4D 高斯 splatting（4D-GS）——将空间加时间维度统一建模，每一个高斯原语具备时空扩展属性，从而显著提升动态场景的训练效率与渲染实时性能。

<details>
    <summary>Abstract</summary>
    表示与渲染动态场景一直是挑战性任务，尤其在准确建模复杂运动的同时仍要保障高效性。为实现实时动态场景渲染且保持训练／存储效率，我们提出 4D Gaussian Splatting (4D-GS) 作为一种整体性表示，而非对每帧单独应用 3D-GS。实验表明，该方法可在大场景下实现高保真渲染与实时帧率。  
</details>

<details>
    <summary>Key points</summary>
    * 引入时空高斯原语：同时建模 XYZ + 时间轴。  
    * 设计高效的变形场 (deformation field) 和 splatting 渲染器以支持动态几何与外观。  
    * 在大规模场景下实现实时（30 fps+）渲染。  
    * 相比传统逐帧建模显著提升训练/存储效率。  
</details>

---

**Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes**

* **Authors:** Zhengqi Li, Simon Niklaus, Noah Snavely, Oliver Wang  
* **arXiv ID:** 2011.13084  
* **One-liner:** 用于动态场景的新视角+新时间合成的神经场景流域（NSFF）  
* **Published in:** CVPR 2021  
* **Links:** [[Paper]](https://arxiv.org/abs/2011.13084) | [[PDF]](https://arxiv.org/pdf/2011.13084.pdf) | [[Code]](https://github.com/zhengqili/Neural-Scene-Flow-Fields)

> **核心创新**  
> 提出 Neural Scene Flow Fields (NSFF)：一种以时间变化的连续函数表示动态场景，其同时编码几何、外观与三维场景流（scene flow），从单目视频及已知摄像机轨迹中学习，实现联合视角与时间插值。 

<details>
    <summary>Abstract</summary>
    我们提出一种方法，仅使用带已知相机位姿的单目视频即可执行动态场景的新视角与新时间合成。为此，我们引入 Neural Scene Flow Fields，一种时间变化的连续函数，用于建模动态场景的外观、几何及三维场景流。该表示通过神经网络拟合输入视图。我们展示该表示可用于处理复杂动态场景，包括细结构、视角依赖效应与自然运动。 

<details>
    <summary>Key points</summary>
    * 将动态场景表示为连续函数 f(x,y,z,t,θ,φ) → (radiance, density, scene-flow)。  
    * 使用单目视频 + 已知相机轨迹，不依赖多视图阵列。  
    * 支持新视角合成（空间）与新时间合成（插帧/时间方向）。  
    * 能处理薄结构、视角依赖反射与复杂运动。  
</details>

---

**CoTracker: It is Better to Track Together**

* **Authors:** Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht
* **arXiv ID:** 2307.07635  
* **One-liner:** 基于 Transformer 的联合点追踪模型，可同时追踪成千上万点，提高准确性与鲁棒性
* **Published in:** ECCV 2024  
* **Links:** [[Paper]](https://arxiv.org/abs/2307.07635) | [[PDF]](https://arxiv.org/pdf/2307.07635.pdf) | [[Code]](https://github.com/facebookresearch/co-tracker)

> **核心创新**  
> 提出 CoTracker：通过 Transformer 架构联合追踪大量 2D 点流，建模点与点之间的依赖关系，从而提升遮挡点、视野外点的追踪能力与整体准确性。 

<details>
    <summary>Abstract</summary>
    我们展示联合追踪显著提升追踪精度与鲁棒性，并使 CoTracker 能够追踪被遮挡的点或摄像机视野之外的点。我们提出 CoTracker，一种基于 Transformer 的模型，可在线追踪大量 2D 点。技术创新包括“token proxies”以提高内存效率，使 CoTracker 在单 GPU 上实时联合追踪约 70 k 个点。虽然模型在线执行于短窗口，但训练采用展开的长窗口循环结构，以处理长时段序列及遮挡情况。定量结果显示 CoTracker 显著优于现有追踪器。 
</details>

<details>
    <summary>Key points</summary>
    * 联合追踪机制：同时追踪大量点、建模点间依赖，而非独立追踪。  
    * Transformer 架构 + “token proxies”技术，支持大规模 (~70k 点) 在线追踪。  
    * 在线短窗口执行，训练采用长序列反复展开，增强长时段追踪能力。  
    * 在遮挡、视野外点场景下明显优于传统独立追踪方法。  
</details>

---

**GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control**

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

---

## 🌟 "三位一体"原型：涌现的世界模型

本节重点介绍展示**三大一致性初步整合**的模型，表现出涌现的世界模型能力。这些系统代表了当前前沿，展现了真正世界模拟的雏形。

### 文本到世界生成器

从语言描述生成动态、空间一致的虚拟环境的模型。

**关键特征：**
- ✅ 模态一致性：自然语言理解和像素空间生成
- ✅ 空间一致性：具有物体恒存性的 3D 感知场景组合
- ✅ 时间一致性：物理上可信的动力学和运动

#### 代表性工作

**OpenAI Sora**

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

---

**Runway Gen-3 Alpha**

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

---

**Pika 1.0**

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


---

### 具身智能系统

为机器人控制和自主智能体设计的模型，必须整合感知、空间推理和时间预测以执行真实世界任务。

**关键特征：**
- 多模态指令遵循
- 3D 空间导航和操作规划
- 动作后果的预测建模

#### 代表性工作：

**RT-2: Vision-Language-Action Models**

* **Authors:** Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, Brianna Zitkovich
* **arXiv ID:** 2307.15818
* **One-liner:** 将大规模网络 VLM 与机器人动作训练结合，将视觉与语言直接转化为真实机器人动作，实现零样本现实操控。
* **Published in:** CoRL 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.15818) | [[PDF]](https://arxiv.org/pdf/2307.15818.pdf) | [[Project Page]](https://robotics-transformer2.github.io/)

> **核心创新**  
> 将大模型从“看图 + 说话”扩展为“看图 + 理解 + 动作执行”，首次将互联网视觉-语言知识迁移到真实机器人动作层面，使机器人具备开放世界推理和零样本动作能力。

<details>
    <summary>Abstract</summary>
    RT-2 是一种视觉-语言-动作（VLA）模型，结合大规模互联网视觉-语言数据与机器人操作数据训练，直接将图像和语言输入映射为机器人动作 token。RT-2 展示出对新任务、新物体和非训练分布场景的零样本泛化能力，代表机器人从专用示教走向互联网级通用学习的重要一步。
</details>

<details>
    <summary>Key points</summary>
    * 将 VLM 扩展到机器人动作空间  
    * 互联网数据 + 机器人数据联合训练  
    * 输出离散化机器人动作 token 序列  
    * 强零样本操控能力：可执行未示教任务/对象/指令  
</details>

---

**GAIA-1: A Generative World Model for Autonomous Driving**

* **Authors:** Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado
* **arXiv ID:** 2311.07541
* **One-liner:** 面向自动驾驶的生成式世界模型，实现多智能体运动预测与闭环模拟训练。
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2309.17080) | [[PDF]](https://arxiv.org/pdf/2309.17080.pdf)

> **核心创新**  
> 不止是轨迹预测，而是学习“世界演化”——可生成多智能体动态、真实交通行为与传感器信号，用于闭环仿真与安全稀有场景合成。

<details>
    <summary>Abstract</summary>
    GAIA-1 是一个面向真实高速驾驶系统的生成式世界模型，基于大量车队数据学习时空动态，能够重建并预测多智能体行为、道路动态与场景变化。模型可用于闭环自动驾驶仿真与决策测试，生成具有高真实性与多样性的复杂驾驶场景，包括危险与长尾案例。
</details>

<details>
    <summary>Key points</summary>
    * 生成式世界模型：不仅预测轨迹，还模拟整个交通生态  
    * 支持多智能体时空建模（车-人-交通灯-环境）  
    * 用于闭环 AD 仿真、鲁棒性测试和长尾案例生成  
    * 提高数据效率与泛化能力的同时提升安全性  
</details>

---

**PaLM-E: An Embodied Multimodal Language Model**

* **Authors:** Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence
* **arXiv ID:** 2303.03378
* **One-liner:** 将真实世界传感输入嵌入 PaLM 大模型，实现具身智能中的视觉-语言-机器人统一推理与任务执行。
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2303.03378) | [[PDF]](https://arxiv.org/pdf/2303.03378.pdf) | [[Project Page]](https://palm-e.github.io/)

> **核心创新**  
> 把机器人视觉、状态信息作为“token”输入 LLM，使语言模型成为具身智能决策系统，能理解环境、规划动作并执行任务。

<details>
    <summary>Abstract</summary>
    PaLM-E 是一个具身多模态语言模型，将视觉编码、机器人状态信号与大型语言模型相结合。系统以 token 形式接收传感信息与指令，并输出自然语言或机器人动作计划。PaLM-E 在长时序操控、环境理解与跨任务泛化中表现优异，展示通用具身智能的关键方向。
</details>

<details>
    <summary>Key points</summary>
    * 将机器人传感信息 token 化输入 LLM  
    * 视觉-语言-动作统一模型  
    * 具备长期规划、任务分解与泛化能力  
    * 推动“机器人 = LLM + 具身输入”研究范式  
</details>

---

## 📊 基准测试与评估

**当前挑战**：现有指标（FID、FVD、CLIP Score）无法充分评估世界模型能力，侧重于感知质量而非物理理解。

**综合基准测试的需求：**

真正的世界模型基准应评估：
- 🧩 **常识物理理解**：模型是否遵守重力、动量、守恒定律？
- 🔮 **反事实推理**：能否预测假设干预的结果？
- ⏳ **长期一致性**：在扩展的模拟时间范围内连贯性是否会崩溃？
- 🎯 **目标导向规划**：能否链接动作以实现复杂目标？
- 🎛️ **可控性**：用户能多精确地操控模拟元素？

#### 代表性工作

**WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation**

* **Authors:** Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, Li Yuan
* **arXiv ID:** 2503.07265
* **One-liner:** 提出可感知世界知识的语义一致性评价指标（WISE），用以衡量文生图模型是否真正理解常识与世界事实。
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2503.07265) | [[PDF]](https://arxiv.org/pdf/2503.07265.pdf) | [[Code]](https://github.com/PKU-YuanGroup/WISE)

> **核心创新**  
> WISE 强调“评价应包含世界知识”，构建了一个包含真实世界语义约束、概念关系和常识推理的文生图一致性评估体系，能够显著更准确地衡量文本-图像对齐与理解能力，而非只依赖 CLIP 相似度。

<details>
    <summary>Abstract</summary>
    文本到图像模型的快速发展呼唤更强的自动化评价方法。然而现有指标（如 CLIPScore）无法区分表面匹配与真实语义一致，尤其在涉及常识、世界知识与逻辑关系时。WISE 通过世界知识驱动的语义检查框架，对生成图像的内容与文本语义进行深入比对，包括物体关系、属性一致性与常识合理性。实验显示，WISE 与人工评价的相关性显著优于现有方法。
</details>

<details>
    <summary>Key points</summary>
    * 引入“世界知识 + 常识推理”评估框架  
    * 评价维度：实体正确性、关系合理性、属性一致性、场景常识  
    * 与人工偏好高度相关，优于 CLIPScore/BERTScore 等指标  
    * 可用作未来文生图模型 benchmark 的通用评价层  
</details>

---

**Are Video Models Ready as Zero-Shot Reasoners?An Empirical Study with the MME-COF Benchmark**

* **Authors:** Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, Pheng-Ann Heng
* **arXiv ID:** 2510.26802
* **One-liner:** 构建 MME-COF 基准测试多模态视频模型的零样本常识与因果推理能力，发现当前视频模型仍缺乏真正时序-因果理解。
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2510.26802) | [[PDF]](https://arxiv.org/pdf/2510.26802.pdf) | [[Code]](https://github.com/ZiyuGuo99/MME-CoF)

> **核心创新**  
> 首次系统评估视频大模型的“零样本推理”能力，提出 MME-COF 基准（Memory, Motion, Event, Causality, Object, Future prediction），发现视频模型在细粒度因果理解方面显著落后于视觉-语言大模型。

<details>
    <summary>Abstract</summary>
    尽管视频生成与理解模型快速进步，是否具备类人“推理”能力仍不明确。本文提出 MME-COF，一个面向零样本视频推理的 benchmark，涵盖对象识别、事件推断、记忆依赖、时序理解、动作因果与未来预测等能力。实证研究表明，当前领先的视频模型在常识与因果推理任务上落后于 VLMs，提示视频推理能力仍处早期阶段。
</details>

<details>
    <summary>Key points</summary>
    * MME-COF 六大测试维度：对象识别、记忆、动作、事件、因果、未来预测  
    * 面向“零样本”评估，无视频任务微调  
    * 视频模型 vs 文本-图像模型：前者推理能力不足  
    * 揭示了当前视频模型更偏“模式拟合”而非“语义推理”  
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
