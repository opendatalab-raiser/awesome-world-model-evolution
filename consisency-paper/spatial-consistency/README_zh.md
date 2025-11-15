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
