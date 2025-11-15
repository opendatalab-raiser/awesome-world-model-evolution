<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### 模态 + 时间一致性

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
