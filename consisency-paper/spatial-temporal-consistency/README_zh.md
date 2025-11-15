<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### 空间一致性 + 时间一致性

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
