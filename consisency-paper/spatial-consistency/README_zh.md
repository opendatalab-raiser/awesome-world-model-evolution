<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### 空间一致性

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

<details>
<summary><b>Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Scale Models</b></summary>

* **Authors:** Lichao Sun, Yue Huang, Haoran Wang et al.
* **arXiv ID:** 2402.17177
* **One-liner:** 全面审查Sora技术栈，聚焦时空分块化和扩散Transformer架构的长程依赖建模，用于长序列视频生成。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2402.17177) | [[PDF]](https://arxiv.org/pdf/2402.17177.pdf)

> **核心创新**  
> 论文的核心创新在于系统分析了Sora的时空分块化方法和扩散Transformer架构，这些技术实现了连贯的长序列视频生成并改进了物理一致性。

<details>
    <summary>Abstract</summary>
    本文对OpenAI的大规模文本到视频生成模型Sora进行了全面综述。我们分析了Sora的技术基础，重点关注其时空分块化策略，该策略在空间和时间维度上统一了视觉表示。扩散Transformer架构能够建模长程依赖关系，这对于在扩展视频序列中保持连贯性至关重要。我们研究了Sora如何应对视频生成中的关键挑战，包括时间一致性、物理合理性和组合推理。该综述涵盖了技术突破和剩余限制，如精确时间控制和复杂物理交互。我们还讨论了大规模视频生成模型在各种应用中的潜在机会和影响，以及更广泛的AI研究前景。
</details>

<details>
    <summary>Key points</summary>
    * 分析Sora的时空分块化实现统一视觉表示
    * 研究扩散Transformer架构用于长程依赖建模
    * 讨论视频生成中的物理一致性约束
    * 识别时间控制和复杂物理交互的局限性
    * 探索大规模视频生成模型的机遇
</details>
</details>

---

<details>
<summary><b>Point-E: A System for Generating 3D Point Clouds from Complex Prompts</b></summary>

* **Authors:** Alex Nichol, Heewoo Jun et al.
* **arXiv ID:** 2212.08751
* **One-liner:** 首个文本驱动端到端3D点云生成系统，使用两阶段扩散方法。
* **Published in:** arXiv 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2212.08751) | [[PDF]](https://arxiv.org/pdf/2212.08751.pdf) | [[Code]](https://github.com/openai/point-e)

> **核心创新**  
> 论文的核心创新在于提出两阶段扩散模型，首先生成隐式场，然后解码为3D点云，有效将2D先验知识迁移到3D生成，同时解决3D数据稀缺问题。

<details>
    <summary>Abstract</summary>
    我们提出了Point-E，一个从复杂文本提示生成3D点云的系统。虽然先前关于3D生成的工作主要集中在光栅化表示如网格或体素上，但点云提供了一种简单灵活的替代方案，可以高效表示复杂几何形状。我们的方法使用两阶段方法：首先，一个文本条件扩散模型生成对象的单个合成视图；然后，第二个扩散模型根据生成的图像生成3D点云。这种方法允许我们在训练期间利用大规模2D图像-文本数据集，同时仅需要较小的3D数据集用于第二阶段。我们在大型文本-3D对数据集上训练我们的模型，并证明Point-E可以从复杂文本描述生成多样化和连贯的3D形状。该系统在单个GPU上快速运行，在1-2分钟内生成点云。
</details>

<details>
    <summary>Key points</summary>
    * 首个文本驱动端到端3D点云生成系统
    * 使用两阶段扩散：图像生成后接点云生成
    * 利用2D先验知识解决3D数据稀缺问题
    * 从复杂文本提示生成多样化3D形状
    * 在单个GPU上1-2分钟内运行
</details>
</details>

---

<details>
<summary><b>Shap-E: Generating Conditional 3D Implicit Functions</b></summary>

* **Authors:** Heewoo Jun, Alex Nichol
* **arXiv ID:** 2305.02463
* **One-liner:** Shap-E通过生成隐式函数参数实现文本和图像条件的3D生成，输出高质量3D资产。
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2305.02463) | [[PDF]](https://arxiv.org/pdf/2305.02463.pdf) | [[Code]](https://github.com/openai/shap-e)

> **核心创新**  
> 论文的核心创新在于提出条件生成模型，直接输出隐式函数（NeRF或有符号距离函数）的参数，实现从文本和图像输入的快速高质量3D生成。

<details>
    <summary>Abstract</summary>
    我们提出了Shap-E，一个用于3D资产的条件生成模型，生成可以渲染为纹理网格或神经辐射场的隐式函数参数。与生成显式3D表示（如点云或网格）的先前方法不同，Shap-E直接生成表示3D形状的隐式函数参数。这允许更高质量的结果和更灵活的3D表示。我们在两个阶段训练Shap-E：首先，我们训练一个编码器，将3D资产映射到隐式函数的参数；然后，我们在这些参数上训练条件扩散模型。这种方法使我们能够在几秒钟内生成3D资产，同时支持文本和图像条件。实验表明，Shap-E产生比Point-E更高质量的3D资产，同时生成速度更快并支持更多输出格式。
</details>

<details>
    <summary>Key points</summary>
    * 生成隐式函数参数而非显式3D表示
    * 支持文本和图像条件的3D生成
    * 两阶段训练：编码器训练后接条件扩散
    * 比Point-E生成速度快10倍且质量更高
    * 输出可渲染为网格或神经辐射场
</details>
</details>

---

<details>
<summary><b>DreamFusion: Text-to-3D using 2D Diffusion</b></summary>

* **Authors:** Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall
* **arXiv ID:** 2209.14988
* **One-liner:** DreamFusion引入分数蒸馏采样（SDS），通过利用2D扩散先验实现无需3D训练数据的文本到3D生成。
* **Published in:**  ICLR 2023 
* **Links:** [[Paper]](https://arxiv.org/abs/2209.14988) | [[PDF]](https://arxiv.org/pdf/2209.14988.pdf)

> **核心创新**  
> 论文的核心创新在于提出分数蒸馏采样（SDS），该方法通过可微渲染将2D扩散模型的知识蒸馏到3D表示（NeRF）中，实现无需任何3D训练数据的文本到3D生成。

<details>
    <summary>Abstract</summary>
    我们提出了DreamFusion，一种从文本提示生成3D模型而无需任何3D训练数据的方法。我们的方法使用预训练的2D文本到图像扩散模型来优化随机初始化的3D模型，该模型由神经辐射场（NeRF）表示。关键创新是分数蒸馏采样（SDS），这是一个损失函数，允许我们使用2D扩散模型作为3D生成的先验。SDS通过在随机相机位置从3D模型渲染图像，然后使用扩散模型估计条件图像分布的分数函数，最后反向传播该分数来更新3D模型。这种方法能够生成多样化、高质量的3D模型，具有复杂的几何形状和纹理，与输入文本提示匹配。我们证明DreamFusion可以生成各种物体和场景的3D模型，优于需要3D监督的先前方法。
</details>

<details>
    <summary>Key points</summary>
    * 引入分数蒸馏采样（SDS）用于3D生成
    * 使用2D扩散先验而无需3D训练数据
    * 通过可微渲染优化神经辐射场
    * 实现具有复杂几何形状的文本到3D生成
    * 开创了使用2D扩散模型作为3D先验的先河
</details>
</details>

---

<details>
<summary><b>Magic3D: High-Resolution Text-to-3D Content Creation</b></summary>

* **Authors:** Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, Tsung-Yi Lin
* **arXiv ID:** 2211.10440
* **One-liner:** Magic3D引入两阶段优化框架，使用粗糙NeRF初始化和精细3D高斯溅射精炼实现高分辨率文本到3D生成。
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2211.10440) | [[PDF]](https://arxiv.org/pdf/2211.10440.pdf)

> **核心创新**  
> 论文的核心创新在于提出从粗糙到精细的两阶段框架，首先优化低分辨率NeRF，然后使用高分辨率3D高斯表示进行精炼，在实现效率的同时获得高质量结果。

<details>
    <summary>Abstract</summary>
    我们提出了Magic3D，一个高分辨率文本到3D内容创建框架，从文本提示生成高质量的3D模型。我们的方法使用两阶段优化过程：首先，我们使用分数蒸馏采样（SDS）优化低分辨率神经辐射场（NeRF）以获得粗糙的3D结构；然后，我们从NeRF中提取网格，并使用具有3D感知超分辨率的可微光栅器对其进行精炼，实现高分辨率纹理合成。这种从粗糙到精细的策略使我们能够比先前方法实现更高的分辨率和更好的质量，同时计算效率高。Magic3D可以在约40分钟内生成512×512分辨率的3D模型，比DreamFusion显著更快，同时产生更详细和连贯的3D资产。我们通过广泛的定性和定量评估证明了我们方法的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 两阶段从粗糙到精细优化框架
    * 结合低分辨率NeRF和高分辨率网格精炼
    * 在约40分钟内实现512×512分辨率生成
    * 减少多视角不一致性30%
    * 提高文本-3D对齐，达到85%匹配准确率
</details>
</details>

---

<details>
<summary><b>DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation</b></summary>

* **Authors:** Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, Gang Zeng
* **arXiv ID:** 2309.16653
* **One-liner:** DreamGaussian将分数蒸馏采样直接应用于3D高斯溅射表示，实现高效可编辑的3D内容生成。
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2309.16653) | [[PDF]](https://arxiv.org/pdf/2309.16653.pdf) | [[Code]](https://github.com/dreamgaussian/dreamgaussian)

> **核心创新**  
> 论文的核心创新在于将SDS直接应用于3D高斯溅射表示而非NeRF，实现120倍更快的生成速度，同时保持高质量并支持实时渲染和编辑。

<details>
    <summary>Abstract</summary>
    我们提出了DreamGaussian，一个新颖的3D内容生成框架，利用3D高斯溅射实现高效高质量的文本到3D和图像到3D生成。与先前使用神经辐射场（NeRF）的方法不同，我们直接使用分数蒸馏采样（SDS）优化3D高斯分布。这种方法将3D高斯表示的表达能力与扩散模型的生成能力相结合。我们的方法实现了显著的加速，在大约30秒内生成高质量的3D资产，这比基于NeRF的方法快120倍。生成的3D高斯支持实时渲染并支持灵活的编辑操作。我们证明DreamGaussian产生具有高视觉质量和与输入文本提示准确语义对齐的3D资产，达到88%的文本-3D空间语义匹配准确率。
</details>

<details>
    <summary>Key points</summary>
    * 将SDS直接应用于3D高斯溅射表示
    * 比基于NeRF的方法快120倍（约30秒）
    * 支持实时渲染和灵活编辑
    * 达到88%文本-3D空间语义匹配
    * 实现高效的文本到3D和图像到3D生成
</details>
</details>

---

<details>
<summary><b>GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images</b></summary>

* **Authors:** Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, Sanja Fidler
* **arXiv ID:** 2209.11163
* **One-liner:** GET3D是一个端到端的3D原生扩散模型，直接在3D数据域中生成高保真纹理3D形状，无需显式3D监督。
* **Published in:** NeurIPS 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2209.11163) | [[PDF]](https://arxiv.org/pdf/2209.11163.pdf) | [[Code]](https://github.com/nv-tlabs/GET3D)

> **核心创新**  
> 论文的核心创新在于直接在3D网格数据域上训练扩散模型，实现端到端的完整纹理3D形状生成，无需依赖显式3D监督或2D到3D提升。

<details>
    <summary>Abstract</summary>
    我们介绍了GET3D，一个直接从图像生成显式纹理3D网格的生成模型。我们的关键洞察是在3D数据表示上直接训练3D生成模型，而不是依赖2D监督或可微渲染。GET3D由两个主要组件组成：一个生成3D表面场的几何生成器，和一个合成高分辨率纹理的纹理生成器。该模型在3D形状数据集上端到端训练，学习生成具有复杂几何形状和详细纹理的多样化高质量3D资产。与需要2D监督或多视角一致性损失的先前方法不同，GET3D直接针对3D质量进行优化。我们在ShapeNet数据集上展示了最先进的结果，在类别条件生成中实现了16.3的FID，在文本-网格匹配中达到86%的准确率。
</details>

<details>
    <summary>Key points</summary>
    * 用于网格生成的端到端3D原生扩散模型
    * 直接在3D网格数据上训练，无需显式3D监督
    * 生成具有复杂几何形状的完整纹理3D形状
    * 在ShapeNet类别条件生成中实现FID 16.3
    * 无需2D监督达到86%文本-网格匹配准确率
</details>
</details>

---

<details>
<summary><b>Mamba3D: Enhancing Local Features for 3D Point Cloud Analysis via State Space Model</b></summary>

* **Authors:** Xu Han, Xiaohui Wang, Muzi Zhuge, Wenyi Li, Hang Xu, Zhen Yang, Chen Zhao, Guosheng Lin, Jiannong Cao
* **arXiv ID:** 2404.14966
* **One-liner:** Mamba3D引入双向状态空间模型实现高效的3D点云分析，通过增强局部特征提取和线性复杂度提升性能。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2404.14966) | [[PDF]](https://arxiv.org/pdf/2404.14966.pdf)

> **核心创新**  
> 论文的核心创新在于用双向状态空间模型（SSM）替代Transformer的二次复杂度，实现3D点云的线性复杂度处理，同时通过局部规范池化增强局部几何特征提取。

<details>
    <summary>Abstract</summary>
    我们提出了Mamba3D，一种用于3D点云分析的新架构，利用状态空间模型（SSM）解决基于Transformer方法的局限性。传统的Transformer模型在处理大规模点云时存在二次计算复杂度，导致信息丢失和局部特征提取能力弱。Mamba3D引入双向SSM（bi-SSM）结合局部规范池化，有效捕捉细粒度几何细节。我们的方法实现了相对于序列长度的线性复杂度，使其能够扩展到大规模点云。在各种3D理解任务上的广泛实验表明，Mamba3D在准确性和效率方面均优于基于Transformer的方法，在ScanObjectNN分类上达到92.6%的OA，在ModelNet40上通过单模态预训练达到95.1%。
</details>

<details>
    <summary>Key points</summary>
    * 首个专门为3D点云设计的Mamba变体
    * 双向状态空间模型实现线性复杂度
    * 局部规范池化增强几何特征提取
    * 在ScanObjectNN上从零训练达到92.6% OA
    * 在准确性和效率方面均优于Transformer
</details>
</details>

---

<details>
<summary><b>Zero-1-to-3: Zero-shot One Image to 3D Object</b></summary>

* **Authors:** Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, Carl Vondrick
* **arXiv ID:** 2303.11328
* **One-liner:** 零样本框架利用Stable Diffusion特征和相机姿态编码，无需3D训练数据即可从单图生成新颖视图。
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2303.11328) | [[PDF]](https://arxiv.org/abs/2303.11328.pdf) | [[Code]](https://github.com/cvlab-columbia/zero123)

> **核心创新**  
> 论文的核心创新在于提出零样本框架，通过相对相机姿态条件和视角一致性损失从单图推断3D结构，实现无需3D监督的新颖视图合成。

<details>
    <summary>Abstract</summary>
    我们提出了Zero-1-to-3，一个仅给定单个RGB图像即可改变物体相机视角的框架。我们的方法利用大规模扩散模型学习到的几何先验，同时克服它们在3D一致性方面的局限性。我们在多视图图像的合成数据集上微调预训练的2D扩散模型，教导它根据单个输入图像和相对相机变换生成物体的新颖视图。关键洞察是将模型条件化为输入视图和目标视图之间的相对相机姿态，使其能够学习3D感知的变换而无需显式3D监督。我们的方法在零样本新颖视图合成方面达到了最先进的结果，在不同视角下产生几何一致的输出，保持输入图像的身份和细节。
</details>

<details>
    <summary>Key points</summary>
    * 从单图进行零样本新颖视图合成
    * 相对相机姿态条件实现3D感知
    * 利用预训练的2D扩散模型
    * 无需3D训练数据
    * 在FFHQ上达到89%多视图结构一致性
</details>
</details>

---

<details>
<summary><b>One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds Without Per-Shape Optimization</b></summary>

* **Authors:** Minghua Liu, Chao Xu, Haian Jin, Lingteng Qiu, Chen Wang, Yuchen Rao, Yukang Cao, Zexiang Xu, Hao Su
* **arXiv ID:** 2306.16928
* **One-liner:** 前向传播流水线通过多视图合成和多视图立体视觉重建，无需逐形状优化即可快速实现单图到3D网格生成。
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2306.16928) | [[PDF]](https://arxiv.org/pdf/2306.16928.pdf) | [[Code]](https://github.com/One-2-3-45/One-2-3-45)

> **核心创新**  
> 论文的核心创新在于提出前向传播流水线，首先生成多视图一致图像，然后应用多视图立体视觉（MVS）重建水密3D网格，在45秒内无需测试时优化。

<details>
    <summary>Abstract</summary>
    我们提出了One-2-3-45，一种新颖的方法，可在仅45秒内从单张图像重建3D网格而无需逐形状优化。我们的方法包括两个主要阶段：首先，我们使用视角条件扩散模型从不同视角生成物体的多视图一致图像；其次，我们应用多视图立体视觉重建将这些生成的视图融合成带有纹理的水密3D网格。这种前向设计消除了先前方法（如分数蒸馏采样）中使用的耗时测试时优化的需要。One-2-3-45产生高质量的3D网格，保留输入图像的几何细节和纹理，同时比基于优化的方法显著更快。我们证明我们的方法相比Zero-1-to-3在网格水密性方面提高了27%，同时生成完整的360°纹理网格。
</details>

<details>
    <summary>Key points</summary>
    * 无需测试时优化的前向传播流水线
    * 多视图合成后接MVS重建
    * 在45秒内生成水密纹理网格
    * 相比Zero-1-to-3网格水密性提高27%
    * 从单图实现完整360°重建
</details>
</details>

---

<details>
<summary><b>Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels</b></summary>

* **Authors:** Zhening Xing, Huayi Wang, Yitong Li, Yansong Peng, Jie Zhou, Jiwen Lu
* **arXiv ID:** 2405.16822
* **One-liner:** Vidu4D使用动态高斯面元从单视频重建高保真4D资产，捕捉时空动态并达到85%物理依从性。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2405.16822) | [[PDF]](https://arxiv.org/pdf/2405.16822.pdf)

> **核心创新**  
> 论文的核心创新在于提出动态高斯面元表示，从单目视频捕捉3D几何和时间演化，实现具有85%物理依从性的高质量4D重建。

<details>
    <summary>Abstract</summary>
    我们介绍了Vidu4D，一种从单个生成视频重建高保真4D（3D + 时间）资产的新方法。我们的方法使用随时间演化的高斯面元表示动态场景，捕捉几何细节和时间动态。与静态3D重建方法不同，Vidu4D建模场景的完整时空演化，实现动态场景理解、动画和虚拟现实等应用。我们开发了一个可微渲染流水线，优化高斯面元在时间上的位置、尺度、旋转和外观，受物理原理约束以确保真实运动。我们的方法在重建运动中达到85%的物理依从性，并忠实地再现了对先前方法具有挑战性的复杂动态细节。
</details>

<details>
    <summary>Key points</summary>
    * 用于4D场景表示的动态高斯面元
    * 单视频到4D重建流水线
    * 运动重建中85%物理依从性
    * 捕捉复杂时空动态
    * 高保真动态场景重建
</details>
</details>

---

<details>
<summary><b>GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation</b></summary>

* **Authors:** Quankai Gao, Qiangeng Xu, Zhe Cao, Ben Mildenhall, Wenchao Ma, Le Chen, Danhang Tang, Ulrich Neumann
* **arXiv ID:** 2403.12334
* **One-liner:** GaussianFlow引入高斯流概念连接3D高斯动态与像素速度，实现高质量4D内容创建和新型视图合成。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2403.12334) | [[PDF]](https://arxiv.org/pdf/2403.12334.pdf)

> **核心创新**  
> 论文的核心创新在于提出高斯流概念，将3D高斯动态溅射到2D图像空间并利用光流监督，实现从单目视频的高效4D内容创建。

<details>
    <summary>Abstract</summary>
    我们提出了GaussianFlow，一种用于4D内容创建的新方法，使用时序演化的3D高斯建模动态场景。我们的关键贡献是高斯流概念，它通过可微溅射将3D高斯的动态连接到2D像素速度。这使我们能够使用从视频中容易获得的光流估计来监督动态高斯的优化。GaussianFlow可以从单目视频输入生成高质量的4D表示，支持新型视图合成和时间插值。高斯表示的显式性质使得能够实时渲染动态场景，同时保持高视觉质量。我们在4D重建和新型视图合成任务上展示了最先进的性能，在运动一致性和渲染效率方面有显著改进。
</details>

<details>
    <summary>Key points</summary>
    * 用于动态3D高斯建模的高斯流概念
    * 光流监督实现时间一致性
    * 动态4D内容的实时渲染
    * 单目视频到4D重建
    * 动态场景新型视图合成的最先进性能
</details>
</details>

---

<details>
<summary><b>LLM-to-Phy3D: Physically Conform Online 3D Object Generation with LLMs</b></summary>

* **Authors:** Melvin Wong, Yew-Soon Ong, Hisashi Kashima
* **arXiv ID:** 2506.11148
* **One-liner:** 首个物理一致在线3D生成框架，使用LLM引导提示优化和实时物理评估，适用于工程应用。
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2506.11148) | [[PDF]](https://arxiv.org/pdf/2506.11148.pdf)

> **核心创新**  
> 论文的核心创新在于提出在线黑盒迭代精炼框架，结合视觉和物理评估来引导LLM生成物理一致的3D对象，适用于工程应用。

<details>
    <summary>Abstract</summary>
    我们提出了LLM-to-Phy3D，首个使用大型语言模型的物理一致在线3D对象生成框架。当前的LLM-to-3D方法通常产生几何新颖但物理上不可行的对象，无法用于实际工程应用。我们的方法通过在线迭代精炼过程解决了这一限制，其中生成的3D对象通过视觉指标和物理引擎（牛顿力学、流体）进行评估。评估结果指导LLM自适应地优化生成提示，创建一个收敛到物理可行3D设计的闭环系统。该框架支持从文本描述端到端生成可制造的3D对象，同时保持几何新颖性和物理一致性。我们展示了在机械设计、建筑和产品开发等物理可行性至关重要的应用。
</details>

<details>
    <summary>Key points</summary>
    * 物理一致性的在线迭代精炼
    * LLM引导提示优化与物理评估
    * 使用视觉和物理指标的实时适应
    * 可制造3D对象的端到端生成
    * 工程设计和产品开发应用
</details>
</details>

---

<details>
<summary><b>Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Multi-View Diffusion and Reconstruction</b></summary>

* **Authors:** Wei Liu, Jian Yang, Gang Zhang, Jiashi Li, Xiaojuan Qi, Xuemiao Xu, Shengfeng He
* **arXiv ID:** 2506.16504
* **One-liner:** 工业级3D资产生成套件，使用多视图扩散和重建实现高保真几何和纹理生成。
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2506.16504) | [[PDF]](https://arxiv.org/pdf/2506.16504.pdf)

> **核心创新**  
> 论文的核心创新在于提出全面的3D扩散模型套件，结合多视图扩散和高分辨率重建，专门针对需要详细几何和纹理保真度的工业应用进行优化。

<details>
    <summary>Abstract</summary>
    我们提出了Hunyuan3D 2.5，一个为需要高保真几何和纹理的工业应用设计的先进3D资产生成系统。我们的方法利用多视图扩散模型，从文本或图像输入生成一致的多视图图像，然后通过复杂的重建流水线将这些视图转换为高分辨率纹理网格。该系统解决了工业3D内容创建中的关键挑战，包括精细几何细节、材料准确性和生产就绪的网格质量。Hunyuan3D 2.5在几何精度和纹理保真度方面相比先前方法展示了显著改进，使其适用于游戏、虚拟制作和数字孪生等资产质量至关重要的应用。
</details>

<details>
    <summary>Key points</summary>
    * 一致3D结构生成的多视图扩散
    * 高分辨率纹理网格重建
    * 工业级资产质量优化
    * 文本和图像条件的3D生成
    * 游戏、VFX和数字孪生应用
</details>
</details>

---

<details>
<summary><b>Efficient 3D Shape Generation via Diffusion Mamba with Bidirectional SSMs</b></summary>

* **Authors:** Shentong Mo, Yuantao Chen, Weiming Zhai, Yichen Zhou, Jie Yang, Linqi Song
* **arXiv ID:** 2406.05038
* **One-liner:** 扩散Mamba架构结合双向状态空间模型，实现线性复杂度的3D形状生成，提升效率和长序列建模能力。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2406.05038) | [[PDF]](https://arxiv.org/pdf/2406.05038.pdf)

> **核心创新**  
> 论文的核心创新在于使用Mamba架构和双向状态空间模型将2D扩散Transformer扩展到3D，实现大规模3D形状生成的线性复杂度。

<details>
    <summary>Abstract</summary>
    我们提出了一种新颖的3D形状生成框架，利用扩散Mamba和双向状态空间模型（SSM）来解决基于Transformer的3D生成器的效率限制。传统的Transformer在处理3D体素数据时存在二次计算复杂度，限制了其可扩展性。我们的方法通过用双向SSM替换自注意力机制，将成功的2D扩散Transformer（DiT）架构扩展到3D，实现了相对于序列长度的线性复杂度。我们引入了3D分块化将体素网格转换为补丁序列，并采用双向SSM高效捕捉长程依赖关系。在ModelNet10上的实验表明，相比GET3D在生成准确率上提高了22%，在大规模3D形状生成的训练和推理效率方面都有显著提升。
</details>

<details>
    <summary>Key points</summary>
    * 用于3D生成的扩散Mamba与双向SSM
    * 相比Transformer的二次复杂度实现线性复杂度
    * 3D分块化实现高效序列处理
    * 在ModelNet10上相比GET3D准确率提高22%
    * 增强的训练和推理效率
</details>
</details>

---

<details>
<summary><b>Progressive3D: Progressively Local Editing for Text-to-3D Content Creation with Complex Textural Descriptions</b></summary>

* **Authors:** Jun Hao Liew, Hanshu Yan, Jianfeng Zhang, Zhongcong Xu, Jiashi Feng
* **arXiv ID:** 2310.11784
* **One-liner:** 渐进式局部编辑框架将复杂文本驱动的3D生成分解为局部优化步骤，实现精确控制。
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2310.11784) | [[PDF]](https://arxiv.org/pdf/2310.11784.pdf)

> **核心创新**  
> 论文的核心创新在于提出渐进式局部编辑框架，将复杂的文本到3D生成分解为顺序的局部优化步骤，实现对3D资产不同区域的精确控制。

<details>
    <summary>Abstract</summary>
    我们介绍了Progressive3D，一个用于文本到3D内容创建的新框架，解决了在处理复杂文本描述时全局优化的局限性。传统方法由于同时优化所有方面的挑战，往往无法捕捉提示中描述的复杂纹理细节。我们的方法将生成过程分解为渐进式局部编辑步骤，每个步骤根据文本描述的相应部分专注于精炼3D资产的特定区域。这种局部优化策略使得能够精确控制复杂纹理和几何细节，这些细节在全局优化中难以实现。Progressive3D在处理复杂文本描述方面展示了显著改进，同时在整个3D资产中保持空间一致性。
</details>

<details>
    <summary>Key points</summary>
    * 复杂文本到3D的渐进式局部编辑
    * 将全局优化分解为局部步骤
    * 实现对区域细节的精确控制
    * 改进复杂纹理描述处理
    * 在整个3D资产中保持空间一致性
</details>
</details>

---

<details>
<summary><b>Hallo3D: Multi-Modal Hallucination Detection and Mitigation for Consistent 3D Content Generation</b></summary>

* **Authors:** Fushuai Shi, Zhaoyu Chen, Yichen Zhu, Mingkui Tan, Yan Wang
* **arXiv ID:** 2410.11784
* **One-liner:** 多模态幻觉检测和缓解框架使用GPT-4V解决3D生成中的几何不一致问题，如Janus问题。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2410.11784) | [[PDF]](https://arxiv.org/pdf/2410.11784.pdf)

> **核心创新**  
> 论文的核心创新在于使用多模态大模型（GPT-4V）检测和解决3D生成中的几何幻觉，通过分析渲染的2D视图并提供校正信号。

<details>
    <summary>Abstract</summary>
    我们提出了Hallo3D，一个用于检测和缓解3D内容生成中几何幻觉的新框架。常见问题如Janus问题（多面）和结构不一致源于当前3D生成方法的局限性。Hallo3D通过利用多模态大模型（GPT-4V）分析来自多个视角的渲染2D视图并识别几何矛盾来解决这些挑战。检测到的不一致性用于生成校正信号，指导优化过程朝向更连贯的3D结构。我们的方法将Janus问题的发生率降低了68%，并在各种3D生成方法中显著改善了多视角几何一致性，实现了更可靠和连贯的3D内容创建。
</details>

<details>
    <summary>Key points</summary>
    * 使用GPT-4V进行多模态幻觉检测
    * 从2D渲染进行几何不一致性分析
    * Janus问题发生率降低68%
    * 为3D优化生成校正信号
    * 改进多视角几何一致性
</details>
</details>

---

<details>
<summary><b>SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion</b></summary>

* **Authors:** Nikhil Singh, Kyle Genova, Guandao Yang, Jonathan Barron, Dmitry Lagun, Thomas Funkhouser, Leonidas Guibas
* **arXiv ID:** 2403.12008
* **One-liner:** SV3D使用潜在视频扩散和动态相机轨迹，实现单图到多视图视频和3D生成，增强视角一致性。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2403.12008) | [[PDF]](https://arxiv.org/pdf/2403.12008.pdf)

> **核心创新**  
> 论文的核心创新在于提出潜在视频扩散模型，使用动态相机轨迹从单图生成多视图视频，然后通过两阶段3D优化（NeRF→DMTet）实现几何和光照解耦。

<details>
    <summary>Abstract</summary>
    我们介绍了SV3D，一个从单个输入图像生成多视图视频和3D模型的新框架。我们的方法利用以动态相机轨迹为条件的潜在视频扩散模型来产生一致的多视图序列。生成的视频作为输入到一个两阶段3D重建流水线，该流水线首先优化神经辐射场（NeRF），然后使用DMTet进行高质量网格提取的精炼。这种方法实现了几何和光照的有效解耦，产生在所有视角下保持一致性的3D资产。SV3D在跨视角结构相似性（SSIM）方面实现了32%的改进，并支持完整的360°覆盖，解决了先前单图像3D生成方法的局限性。
</details>

<details>
    <summary>Key points</summary>
    * 具有动态相机轨迹的潜在视频扩散
    * 两阶段3D优化（NeRF → DMTet）
    * 几何和光照解耦
    * 跨视角一致性SSIM提高32%
    * 从单图实现完整360°覆盖
</details>
</details>

---

<details>
<summary><b>Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing</b></summary>

* **Authors:** Xinyu Liang, Zhenyu Zhang, Xiong Zhou, Yujiu Yang, Tiejun Huang
* **arXiv ID:** 2411.16375
* **One-liner:** 因果生成与缓存共享机制实现高效长视频扩散，具有线性计算扩展性。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2411.16375) | [[PDF]](https://arxiv.org/pdf/2411.16375.pdf)

> **核心创新**  
> 论文的核心创新在于提出具有KV缓存共享的因果生成框架，优化长视频生成的计算效率，同时保持时空连贯性。

<details>
    <summary>Abstract</summary>
    我们提出了Ca2-VDM，一个为长视频生成设计的高效自回归视频扩散模型。我们的方法通过引入具有缓存共享机制的因果生成来解决传统视频扩散模型的计算挑战。该模型以因果方式处理视频序列，跨帧重用键值（KV）缓存以减少冗余计算。这种设计实现了相对于序列长度的线性计算扩展，使得长视频生成在保持质量的同时变得可行。Ca2-VDM在推理效率方面展示了显著改进，同时在扩展视频序列中保持空间逻辑连贯性，使其适用于需要分钟级视频生成的应用。
</details>

<details>
    <summary>Key points</summary>
    * 时间一致性的因果生成
    * KV缓存共享实现计算效率
    * 线性计算扩展
    * 改进长视频推理效率
    * 增强空间逻辑连贯性
</details>
</details>

---

<details>
<summary><b>Flex3D: Feed-Forward 3D Generation with Flexible Reconstruction Model and Input View Curation</b></summary>

* **Authors:** Junlin Han, Jianyuan Wang, Weihao Yuan, Yichao Yan, Jiaming Sun, Hujun Bao, Zhaopeng Cui, Xiaowei Zhou
* **arXiv ID:** 2410.00890
* **One-liner:** 灵活输入视图整理和3D高斯重建，实现可变形物体建模并改进形状一致性。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2410.00890) | [[PDF]](https://arxiv.org/pdf/2410.00890.pdf)

> **核心创新**  
> 论文的核心创新在于提出两阶段框架，结合输入视图整理和灵活3D高斯重建，处理可变形物体和变化的输入视图配置。

<details>
    <summary>Abstract</summary>
    我们介绍了Flex3D，一个支持灵活输入视图配置和可变形物体建模的前向传播3D生成框架。传统的3D重建方法通常假设刚性物体和固定视图排列，限制了它们在真实世界场景中的适用性。Flex3D通过两阶段方法解决这些限制：首先，输入视图整理模块选择和对齐用于重建的最佳视图；其次，灵活的3D高斯表示捕捉可变形物体的动态。这种方法在可变形物体一致性得分方面实现了30%的改进，并增强了对多样化输入视图配置的适应性，使得能够从真实世界捕捉场景进行稳健的3D生成。
</details>

<details>
    <summary>Key points</summary>
    * 两阶段：视图整理 + 灵活3D高斯重建
    * 可变形物体建模能力
    * 柔性物体形状一致性提高30%
    * 自适应输入视图配置处理
    * 增强真实世界场景适用性
</details>
</details>

---

<details>
<summary><b>Zero3D: Semantic-Driven Multi-Category 3D Shape Generation</b></summary>

* **Authors:** Zishun Yu, Timur Bagautdinov, Shunshi Zhang, Yuanlu Xu, Tony Tung, Leonidas Guibas
* **arXiv ID:** 2301.13591
* **One-liner:** 语义驱动的零样本多类别3D形状生成，使用单视图对比学习实现通用3D语义表示。
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2301.13591) | [[PDF]](https://arxiv.org/pdf/2301.13591.pdf)

> **核心创新**  
> 论文的核心创新在于单视图对比学习从2D图像中提取通用3D语义先验，实现无需类别特定训练的零样本多类别3D形状生成。

<details>
    <summary>Abstract</summary>
    我们提出了Zero3D，一个用于零样本多类别3D形状生成的语义驱动框架。与需要大量每类别训练的类别特定3D生成器不同，Zero3D通过在2D图像上进行单视图对比学习来学习通用3D语义表示。这种方法使模型能够捕捉跨类别的几何和语义关系，允许它生成在训练期间未见过的类别的3D形状。Zero3D在零样本3D生成任务上相比DreamFusion实现了28%的FID降低，并展示了强大的跨类别适应性，使其适用于需要从语义描述生成多样化3D形状的应用。
</details>

<details>
    <summary>Key points</summary>
    * 用于3D语义的单视图对比学习
    * 零样本多类别形状生成
    * 相比DreamFusion FID降低28%
    * 跨类别适应性
    * 通用3D语义表示学习
</details>
</details>

---

<details>
<summary><b>LinGen: Enhancing Long Video Generation with Linear Attention</b></summary>

* **Authors:** Anisha Jha, Ming-Hsuan Yang, Xiaolong Wang, Zhuowen Tu, Joseph Lim
* **arXiv ID:** 2412.09856
* **One-liner:** 用线性注意力替换标准自回归注意力，实现分钟级长视频生成，具有线性复杂度和改进的空间连贯性。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2412.09856) | [[PDF]](https://arxiv.org/pdf/2412.09856.pdf)

> **核心创新**  
> 论文的核心创新在于用线性注意力机制替换视频生成模型中的二次自注意力，实现长视频序列的高效处理，同时保持时空连贯性。

<details>
    <summary>Abstract</summary>
    我们介绍了LinGen，一种用于长视频生成的新方法，解决了传统自注意力机制的二次计算复杂度问题。通过用线性注意力变体替换标准注意力，LinGen实现了相对于序列长度的线性复杂度，同时保持了长程时空依赖关系的建模能力。这一创新使得能够生成长达16K帧的分钟级视频，相比传统方法在空间逻辑连贯性方面提高了35%。LinGen的高效架构使得长视频生成在计算上变得可行，而不会影响视频质量或时间一致性。
</details>

<details>
    <summary>Key points</summary>
    * 线性注意力实现二次复杂度降低
    * 分钟级视频生成（高达16K帧）
    * 空间逻辑连贯性提高35%
    * 线性计算扩展
    * 高效长程时空建模
</details>
</details>

---

<details>
<summary><b>ByTheWay: Boost Your Text-to-Video Generation Model to Higher Quality in a Training-free Way</b></summary>

* **Authors:** Jialong Bu, Hongkai Ling, Bo Zhao, Seungryong Kim, Jaesik Park, Seongtae Kim
* **arXiv ID:** 2410.06241
* **One-liner:** 无需训练的观察注入精炼方法通过帧间特征约束增强空间一致性，提升文本到视频生成质量。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2410.06241) | [[PDF]](https://arxiv.org/pdf/2410.06241.pdf)

> **核心创新**  
> 论文的核心创新在于提出无需训练的细化方法，通过注入观察约束来增强文本到视频生成中的空间一致性，无需额外训练即可减少40%的场景跳跃。

<details>
    <summary>Abstract</summary>
    我们提出了ByTheWay，一种无需训练的提高文本到视频生成质量的新方法。我们的方法不是重新训练或微调现有模型，而是作为后处理细化步骤运行，注入观察约束以增强帧间的空间一致性。通过分析帧间特征关系并应用一致性约束，ByTheWay将场景跳跃减少了40%，并显著改善了生成视频的整体连贯性。这种无需训练的方法使得质量增强无需模型重新训练的计算成本即可实现，为改进不同模型和架构的视频生成输出提供了实用的解决方案。
</details>

<details>
    <summary>Key points</summary>
    * 无需训练的视频质量增强
    * 观察约束注入
    * 场景跳跃减少40%
    * 帧间特征一致性约束
    * 模型无关的细化方法
</details>
</details>

---

<details>
<summary><b>Pandora: Towards General World Model with Natural Language Actions and Video States</b></summary>

* **Authors:** Xianghua Liu, Kai Ye, Zheyuan Zhang, Yizhou Wang, Gao Huang, Wen Gao
* **arXiv ID:** 2406.09455
* **One-liner:** 混合自回归-扩散模型结合视频状态和自然语言动作，实现通用世界建模并改进物理合规性。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2406.09455) | [[PDF]](https://arxiv.org/pdf/2406.09455.pdf)

> **核心创新**  
> 论文的核心创新在于提出混合自回归-扩散架构，将视频状态表示与自然语言动作命令集成，实现通用世界建模，物理定律合规性提高30%。

<details>
    <summary>Abstract</summary>
    我们介绍了Pandora，一个结合视频状态表示和自然语言动作的通用世界模型，用于动态场景模拟。与特定领域的世界模型不同，Pandora旨在实现跨不同环境和任务的通用适用性。混合自回归-扩散架构以自回归方式处理视频状态，同时使用扩散模型进行动作条件状态转换。这种设计使模型能够学习通用物理原理和因果关系，相比专门的世界模型在物理定律合规性方面实现了30%的改进。Pandora展示了强大的跨领域模拟能力，并支持动态场景的自然语言控制。
</details>

<details>
    <summary>Key points</summary>
    * 混合自回归-扩散架构
    * 视频状态 + 自然语言动作集成
    * 物理合规性提高30%
    * 跨领域动态模拟
    * 通用世界建模能力
</details>
</details>

---

<details>
<summary><b>PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation</b></summary>

* **Authors:** Shaowei Liu, Zhongsheng Wang, Jiaming Liu, Zhangyang Wang, Tianfan Xue
* **arXiv ID:** 2412.00596
* **One-liner:** LLM引导的迭代自精炼与物理感知损失实现基于物理的文本到视频生成，无需物理训练数据。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2412.00596) | [[PDF]](https://arxiv.org/pdf/2412.00596.pdf)

> **核心创新**  
> 论文的核心创新在于提出LLM引导的迭代精炼与物理感知损失函数，强制执行牛顿力学约束，实现无需物理数据监督的物理合理视频生成。

<details>
    <summary>Abstract</summary>
    我们提出了PhyT2V，一种基于物理的文本到视频生成的新方法，解决了生成视频中常见的物理伪影问题。我们的方法采用大型语言模型来指导迭代自精炼过程，其中生成的视频根据物理一致性标准进行评估和改进。物理感知损失函数编码牛顿力学原理，确保物体运动遵循真实的物理定律。无需物理训练数据，PhyT2V在弹道运动物理合理性得分方面实现了28%的改进，并显著减少了非物理伪影，使得能够从文本描述生成物理连贯的视频。
</details>

<details>
    <summary>Key points</summary>
    * LLM引导的迭代自精炼
    * 牛顿力学的物理感知损失
    * 物理合理性提高28%
    * 无需物理训练数据要求
    * 减少非物理伪影
</details>
</details>

---

<details>
<summary><b>DSC-PoseNet: Learning 6DoF Object Pose Estimation via Dual-Scale Consistency</b></summary>

* **Authors:** Yiming Yang, Sixian Chan, Xiangyu Xu, Haibin Ling, Xiaoming Liu
* **arXiv ID:** 2104.03658
* **One-liner:** 双尺度一致性学习框架从2D边界框弱监督学习6DoF物体姿态，误差减少18%。
* **Published in:** CVPR 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2104.03658) | [[PDF]](https://arxiv.org/pdf/2104.03658.pdf)

> **核心创新**  
> 论文的核心创新在于提出双尺度一致性框架，通过在不同空间尺度和视角间强制执行一致性，从弱2D边界框监督中学习6DoF物体姿态。

<details>
    <summary>Abstract</summary>
    我们提出了DSC-PoseNet，一种用于6DoF物体姿态估计的新方法，减少了对昂贵3D标注的依赖。我们的方法利用双尺度一致性学习从容易获得的2D边界框标注中提取3D姿态信息。该框架在局部特征对应和全局空间关系之间强制执行一致性，使得无需完整3D监督即可实现准确的姿态估计。DSC-PoseNet在PASCAL3D+数据集上实现了18%的姿态估计误差减少，并为文本驱动的3D生成应用提供了有价值的姿态约束，弥合了2D感知和3D理解之间的差距。
</details>

<details>
    <summary>Key points</summary>
    * 双尺度一致性学习
    * 来自2D边界框的弱监督
    * 在PASCAL3D+上误差减少18%
    * 2D到3D姿态映射
    * 3D生成的姿态约束
</details>
</details>

---

<details>
<summary><b>A Quantitative Evaluation of Score Distillation Sampling Based Text-to-3D</b></summary>

* **Authors:** Kai Wang, Weiyang Liu, Hao Liu, Bo Zhao, Yisen Wang, James Zou
* **arXiv ID:** 2402.18780
* **One-liner:** 基于人类验证的定量评估框架用于SDS文本到3D方法，零样本FID改进28%并识别失败模式。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2402.18780) | [[PDF]](https://arxiv.org/pdf/2402.18780.pdf)

> **核心创新**  
> 论文的核心创新在于提出全面的定量评估框架，具有人类验证的指标，系统评估SDS文本到3D方法并识别常见失败模式。

<details>
    <summary>Abstract</summary>
    我们提出了首个用于基于分数蒸馏采样（SDS）的文本到3D方法的全面定量评估框架。虽然SDS已经彻底改变了文本到3D生成，但其评估在很大程度上仍然是定性和主观的。我们的框架引入了人类验证的定量指标，系统评估文本-3D对齐、几何质量和视觉保真度。通过广泛的实验，我们识别了如过度平滑等常见失败模式，并为未来改进提供了见解。该框架实现了28%的零样本FID降低，并能够精确识别SDS基3D生成中的优化挑战，为这个快速发展的领域的客观比较和进步奠定了坚实基础。
</details>

<details>
    <summary>Key points</summary>
    * 人类验证的定量评估框架
    * SDS基文本到3D的系统评估
    * 零样本FID改进28%
    * 失败模式识别（过度平滑等）
    * 文本-3D对齐的客观指标
</details>
</details>

---

<details>
<summary><b>NeRF++: Analyzing and Improving Neural Radiance Fields</b></summary>

* **Authors:** Kai Zhang, Gernot Riegler, Noah Snavely, Vladlen Koltun
* **arXiv ID:** 2010.07492
* **One-liner:** 球坐标扩展和形状-辐射歧义分析改进NeRF的无边界场景建模，具有更好的泛化性。
* **Published in:** CVPR 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2010.07492) | [[PDF]](https://arxiv.org/pdf/2010.07492.pdf) | [[Code]](https://github.com/Kai-46/nerfplusplus)

> **核心创新**  
> 论文的核心创新在于将NeRF扩展到球坐标进行无边界场景建模，并分析形状-辐射歧义以改进泛化性和减少伪影。

<details>
    <summary>Abstract</summary>
    我们提出了NeRF++，对原始神经辐射场方法的分析和改进。我们的工作解决了两个关键限制：处理无边界场景和解决形状-辐射歧义。我们引入了一种球面参数化，自然地扩展了NeRF到无边界环境，实现了大规模场景的高质量新视角合成。此外，我们提供了对NeRF中形状-辐射歧义问题的彻底分析，并提出了减轻其影响的解决方案。NeRF++展示了对大规模无边界场景的改进泛化性，并减少了由辐射场歧义引起的伪影，为神经场景表示建立了更稳健的基础。
</details>

<details>
    <summary>Key points</summary>
    * 无边界场景的球坐标扩展
    * 形状-辐射歧义分析和缓解
    * 改进对大规模环境的泛化性
    * 减少辐射场伪影
    * 增强神经场景表示的稳健性
</details>
</details>

---

<details>
<summary><b>Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding</b></summary>

* **Authors:** Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Karsten Kreis, David Fleet, Mohammad Norouzi
* **arXiv ID:** 2205.11487
* **One-liner:** 深度条件扩散模型增强空间关系理解和几何推理，在文本到图像生成中布局准确率提高35%。
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2205.11487) | [[PDF]](https://arxiv.org/pdf/2205.11487.pdf)

> **核心创新**  
> 论文的核心创新在于将深度图作为中间控制信号集成到扩散模型中，强制执行空间层次关系并改进文本到图像生成中的几何推理。

<details>
    <summary>Abstract</summary>
    我们提出了一种先进的文本到图像扩散模型，在实现前所未有的照片级真实感的同时，展示了深度语言理解和空间推理能力。通过将深度图作为中间条件信号纳入，我们的模型学习强制执行空间层次关系和生成图像中的几何一致性。这种方法解决了2D生成中常见的"平面化"问题，其中前景-背景关系和空间布局往往缺乏连贯性。我们的方法在室内场景空间布局准确率方面实现了35%的改进，并实现了更好的文本-图像-3D对齐，为3D提升和几何推理任务提供了更强的基础。
</details>

<details>
    <summary>Key points</summary>
    * 空间推理的深度条件扩散
    * 空间布局准确率提高35%
    * 增强前景-背景层次结构
    * 更好的文本-图像-3D对齐
    * 减少2D生成中的"平面化"
</details>
</details>

---

<details>
<summary><b>LiftImage3D: Lifting Any Single Image to 3D Gaussians with Video Diffusion Priors</b></summary>

* **Authors:** Yuyang Yin, Shuhan Shen et al.
* **arXiv ID:** 2412.09597
* **One-liner:** 使用鲁棒神经匹配和3D高斯溅射的单图像到3D一致视频生成框架，结合失真感知渲染。
* **Published in:** arXiv 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2412.09597) | [[PDF]](https://arxiv.org/pdf/2412.09597.pdf)

> **核心创新**  
> 论文的核心创新在于使用鲁棒神经匹配模型MASt3R估计相机姿态和点云，然后通过失真感知渲染提升到3DGS表示，实现多视角一致性。

<details>
    <summary>Abstract</summary>
    我们提出了LiftImage3D，一种利用视频扩散先验将任何单张图像提升到3D高斯表示的新框架。我们的方法解决了单图像3D重建中的多视角不一致和几何错位挑战。我们首先使用MASt3R鲁棒神经匹配模型从输入图像估计相机姿态并生成初始点云。然后将这些提升到3D高斯溅射（3DGS）表示，结合失真感知渲染和视频扩散先验以增强空间一致性。该框架相比DreamFusion实现了42%的多视角几何一致性误差降低，实现了精确的图像到3D对齐，并从单张图像生成一致的3D表示。
</details>

<details>
    <summary>Key points</summary>
    * 使用MASt3R进行鲁棒相机姿态和点云估计
    * 将单张图像提升到3D高斯溅射表示
    * 融入视频扩散先验以增强空间一致性
    * 相比DreamFusion多视角几何一致性误差降低42%
    * 失真感知渲染改进几何对齐
</details>
</details>
