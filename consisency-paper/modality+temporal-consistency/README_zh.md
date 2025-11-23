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

<details>
<summary><b>UniVG: Towards UNIfied-modal Video Generation</b></summary>

* **Authors:** Ludan Ruan, Lei Tian, Chuanwei Huang, Xu Zhang, Xinyan Xiao
* **arXiv ID:** 2401.09084
* **One-liner:** This work introduces UniVG, a unified modal video generation system that reclassifies multi-modal video tasks such as text/image-driven generation based on “generative degrees of freedom.” Within a single diffusion framework, it simultaneously addresses high-degree-of-freedom (text/image-driven generation) and low-degree-of-freedom (image animation, super-resolution) video generation using multi-modal cross-attention and biased Gaussian noise. UniVG uniformly supports diverse tasks while outperforming existing methods in both metrics and subjective quality.
* **Published in:** ICME 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2401.09084) | [[PDF]](https://arxiv.org/pdf/2401.09084) | [[Code]](https://univg-baidu.github.io/)

> **核心创新**  
> 提出用“生成自由度”这一统一视角把不同视频生成任务划分为高自由度/低自由度两类，并针对两类任务在同一扩散框架中设计了“多条件交叉注意力 + 偏置高斯噪声”这一成套机制，从而真正把多模态、多任务视频生成统一到一个模型里

<details>
    <summary>Abstract</summary>
    基于扩散的视频生成在学术界和工业界都受到了广泛关注，并取得了可观的成果。然而，当前的研究主要集中在单一目标或单一任务的视频生成上，例如由文本驱动、由图像驱动，或由文本与图像结合驱动的生成。这无法完全满足现实应用场景的需求，因为用户可能会以灵活的方式输入图像和文本条件，单独输入或组合输入。为了解决这一问题，我们提出了一个统一模态的视频生成系统，能够跨文本和图像模态处理多种视频生成任务。为此，我们从生成自由度的角度重新审视了系统中的各种视频生成任务，并将其分类为高自由度视频生成和低自由度视频生成。在高自由度视频生成中，我们采用多条件交叉注意力（Multi-condition Cross Attention）生成与输入图像或文本语义一致的视频。在低自由度视频生成中，我们引入有偏高斯噪声（Biased Gaussian Noise）替代纯随机高斯噪声，以更好地保留输入条件的内容。我们的方法在公共学术基准 MSR-VTT 上实现了最低的 Fréchet 视频距离（FVD），在人类评估中超过了当前的开源方法，并且与当前闭源方法 Gen2 相当。
</details>

<details>
    <summary>Key points</summary>
    * 提出 UniVG 统一模态视频生成系统，在一个扩散框架下同时支持文本→视频、图像→视频以及文本+图像联合条件的视频生成
    * 引入“生成自由度（generative freedom）”概念，把视频生成任务统一划分为高自由度（创意生成）和低自由度（内容约束/编辑）两大类，用这一视角重新整理任务和模型设计
    * 针对高自由度任务，在 U-Net 中设计 Multi-condition Cross Attention，同时接收文本与图像多种条件，实现任意组合条件下的语义一致视频生成
    * 针对低自由度任务（如图像动画、视频超分），提出 Biased Gaussian Noise，将扩散噪声偏向输入条件分布，从而更好地保持输入内容与结构
    * 在 MSR-VTT 等基准上获得最低 FVD，主观评价优于现有开源方法并接近 Gen-2，证明统一系统在多任务上的泛化能力和实际应用价值
</details>
</details>

---

<details>
<summary><b>VideoGen: A Reference-Guided Latent Diffusion Approach for High Definition Text-to-Video Generation</b></summary>

* **Authors:** Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, Jingdong Wang
* **arXiv ID:** 2309.00398
* **One-liner:** VideoGen proposes a text-to-video generation framework combining “reference images + cascaded latent space diffusion + streaming temporal upsampling.” By first generating high-quality reference images from text, then guiding latent space video diffusion and high-definition decoding, it significantly enhances the spatial clarity and temporal consistency of generated videos, achieving state-of-the-art performance at the time.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2309.00398) | [[PDF]](https://arxiv.org/pdf/2309.00398) | [[Code]](https://videogen.github.io/VideoGen/)

> **核心创新**  
> VideoGen 把“先用 T2I 生成高质量参考图像 + 级联潜空间扩散建视频 + 流式时间上采样 + 单独训练的视频解码器”整合成一套参考图像引导的 T2V 框架，用“图像先定内容、扩散专注学运动”的思路同时大幅提升了文本到视频的清晰度和时间一致性
<details>
    <summary>Abstract</summary>
    在本文中，我们提出了 VideoGen，一种文本到视频的生成方法，该方法可以使用参考引导的潜在扩散生成高帧率保真度和强时间一致性的高清视频。我们利用现成的文本到图像生成模型，例如 Stable Diffusion，从文本提示生成高内容质量的图像，作为参考图像来引导视频生成。然后，我们引入了一个高效的级联潜在扩散模块，该模块以参考图像和文本提示为条件，用于生成潜在视频表示，随后通过基于流的时间上采样步骤提高时间分辨率。最后，我们通过增强的视频解码器将潜在视频表示映射为高清视频。在训练过程中，我们使用真实视频的第一帧作为训练级联潜在扩散模块的参考图像。我们方法的主要特性包括：由文本到图像模型生成的参考图像提高了视觉保真度；将其作为条件使扩散模型更专注于学习视频动态；视频解码器在未标注视频数据上进行训练，因此能够受益于高质量、易获取的视频资源。VideoGen 在文本到视频生成方面，在定性和定量评估上均创下了新的最先进水平。
</details>

<details>
    <summary>Key points</summary>
    * 先用成熟的 T2I 模型从文本生成高质量参考图像，再作为条件引导后续视频生成，提高空间清晰度和语义对齐度
    * 在 VAE 潜空间中设计级联 latent diffusion 模块，统一负责从文本+参考图像到视频潜表示的生成，减少像素空间计算成本
    * 在获得低帧率潜视频后，引入基于光流的 temporal upsampling，将短、低帧率片段扩展为更长、更高帧率的视频，增强时间连续性
    * 单独训练高质量视频解码器，将潜空间视频映射为高清输出，训练时充分利用无标注真实视频数据，提高细节与逼真度
    * 在多项文本到视频生成基准上实现新的 SOTA，在帧质量和时间一致性上均优于当时主流方法，证明“图像先定内容 + 潜扩散学运动”的有效性
</details>
</details>

---

<details>
<summary><b>VideoPoet: A Large Language Model for Zero-Shot Video Generation</b></summary>

* **Authors:** Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Josh Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, Ming-Hsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, Lu Jiang
* **arXiv ID:** 2312.14125
* **One-liner:** This work introduces VideoPoet—a foundational video generation model that uniformly tokenizes video (including audio) into discrete units. It employs a decoder-based large language model for pre-training and fine-tuning across multimodal, multi-task objectives, enabling high-quality zero-shot video generation from diverse inputs such as text, images, and video.
* **Published in:** ICML 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14125) | [[PDF]](https://arxiv.org/pdf/2312.14125) | [[Code]](https://sites.research.google/videopoet/)

> **核心创新**  
> 把视频（含音频）统一离散成 token，用一个纯解码式、按 LLM 方式预训练的多模态 Transformer 来同时建模文本/图像/视频/音频，从而把“视频生成”问题直接转化为多模态自回归语言建模，实现统一框架下的零样本视频生成
<details>
    <summary>Abstract</summary>
    我们介绍了 VideoPoet，这是一种能够从各种条件信号生成高质量视频的模型。VideoPoet 采用仅解码器的 Transformer 架构，处理多模态输入——包括图像、视频、文本和音频。训练方案遵循大型语言模型（LLM）的流程，包括两个阶段：预训练和任务特定适应。在预训练阶段，VideoPoet 在自回归 Transformer 框架中融合了多模态生成目标。预训练的 LLM 作为基础，可适应各种视频生成任务。我们展示的结果表明，该模型在零样本视频生成方面具有最先进的能力，特别强调生成高保真动作。
</details>

<details>
    <summary>Key points</summary>
    * 将视频帧序列和音频统一离散化为“视觉/音频 token”，用和文本相同的离散表示形式交给大模型处理
    * 采用类似 LLM 的 decoder-only Transformer，同时接收文本、图像、视频、音频等多模态输入，并自回归生成视频（及音频）token
    * 沿用“大语言模型”范式，先进行大规模多模态自回归预训练，再通过少量任务特定适配，实现在统一框架下覆盖多种视频生成任务
    * 在同一个模型里支持文本→视频、图像→视频、视频编辑、风格化、补全、视频→音频等多种任务，并在多类评测中展现出强零样本泛化能力
    * 在无需针对单一任务重训的前提下，生成的视频在运动连贯性和内容多样性上达到 SOTA 水平，尤其擅长大幅度、复杂运动的建模
</details>
</details>

---

<details>
<summary><b>Imagen Video: High Definition Video Generation with Diffusion Models</b></summary>

* **Authors:** Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, Tim Salimans
* **arXiv ID:** 2210.02303
* **One-liner:** This work introduces Imagen Video, a high-resolution text-to-video generation system built upon a cascaded video diffusion model. Through its design—comprising “base video generation + spatial/temporal super-resolution cascading + v-parameterization + progressive distillation”—it significantly elevates text-to-video resolution, duration, and temporal consistency to high-definition levels.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2210.02303) | [[PDF]](https://arxiv.org/pdf/2210.02303) | [[Code]](https://imagen.research.google/video/)

> **核心创新**  
> 用一个基础视频扩散模型 + 交替的时间/空间超分辨子模型，再配合 v-parameterization、classifier-free guidance 和渐进蒸馏，在统一架构下首次把文本到视频的分辨率、时长和时间一致性同时推到高清级别
<details>
    <summary>Abstract</summary>
    我们呈现了 Imagen Video，一种基于级联视频扩散模型的文本条件视频生成系统。给定一个文本提示，Imagen Video 使用基础视频生成模型以及交替的空间和时间视频超分辨率模型序列生成高分辨率视频。我们描述了如何将该系统扩展为高分辨率文本到视频模型，包括设计决策，例如在特定分辨率下选择全卷积时间和空间超分辨率模型，以及选择扩散模型的 v 参数化。此外，我们确认并将之前在基于扩散的图像生成方面的研究成果转移到视频生成场景中。最后，我们将渐进蒸馏应用于我们的视频在线模型，并结合无分类器引导实现快速、高质量采样。我们发现，Imagen Video 不仅能够生成高保真度视频，还具有高度的可控性和世界知识，包括生成多样化视频和文本动画的能力，可呈现各种艺术风格并具备 3D 对象理解能力。
</details>

<details>
    <summary>Key points</summary>
    * 提出由“基础视频生成 + 空间超分辨 + 时间超分辨”组成的 7 级级联视频扩散系统，在统一框架下实现高分辨率文本转视频
    * 在 1280×768 分辨率、128 帧、24fps 设置下生成约 5.3 秒高清视频，相比以往工作显著提升了分辨率与时长
    * 验证并迁移了文本到图像扩散中的关键技巧到视频领域，如冻结文本编码器、classifier-free guidance 和 conditioning augmentation
    * 引入 v-parameterization 提升采样质量，并通过 progressive distillation 大幅减少采样步数，同时保持文本条件下视频生成的质量
    * 展示了在 3D 物体理解、文字动画、艺术风格迁移等方面的强可控性和语言理解能力，能生成多风格、高一致性的复杂场景视频
</details>
</details>

---

<details>
<summary><b>Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models</b></summary>

* **Authors:** Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, Karsten Kreis
* **arXiv ID:** 2304.08818
* **One-liner:** This work introduces VideoLDM: by simply adding and aligning the temporal dimension to a pre-trained image LDM, it achieves high-resolution, temporally consistent text/condition-based video generation and video super-resolution. It delivers high-quality results on real-world driving scenarios and text-to-video tasks while significantly reducing computational costs for video generation.
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2304.08818) | [[PDF]](https://arxiv.org/pdf/2304.08818) | [[Code]](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)

> **核心创新**  
> 在现有图像 latent diffusion 模型上“最小改动”地引入时间维度，通过时序卷积和跨帧注意力把静态图像生成器扩展为视频生成器，从而在几乎不增加参数和训练成本的前提下，实现高分辨率、时间一致的文本/条件视频生成与视频超分辨率
<details>
    <summary>Abstract</summary>
    潜在扩散模型（LDMs）通过在压缩的低维潜空间中训练扩散模型，实现高质量图像合成，同时避免过高的计算需求。在此，我们将LDM范式应用于高分辨率视频生成，这是一个尤其耗费资源的任务。我们首先仅在图像上对LDM进行预训练；然后，通过在潜空间扩散模型中引入时间维度，并在编码的图像序列（即视频）上进行微调，将图像生成器转变为视频生成器。同样地，我们对扩散模型的上采样器进行时间对齐，使其成为时间一致的视频超分辨率模型。我们关注两个相关的现实应用：野外驾驶数据模拟和基于文本的视频创作。特别是，我们在分辨率为512 × 1024的真实驾驶视频上验证了我们的Video LDM，实现了最先进的性能。此外，我们的方法可以轻松利用现成的预训练图像LDM，因为在这种情况下我们只需训练一个时间对齐模型。通过这样做，我们将公开可用、最先进的文本到图像LDM——Stable Diffusion，转变为高效且表现力强的文本到视频模型，分辨率可达1280 × 2048。我们展示了以这种方式训练的时间层可以推广到不同的微调文本到图像LDM。利用这一特性，我们展示了个性化文本到视频生成的首批结果，为未来的内容创作开辟了令人兴奋的方向。
</details>

<details>
    <summary>Key points</summary>
    * 在预训练图像 LDM 的基础上，仅通过在潜空间中引入时间维度并微调时间层，就把静态图像生成器“平滑”扩展为视频生成器
    * 在 VAE 潜空间里训练时序卷积/注意力层，使跨帧的 latent 表达对齐，显著提升视频的时间一致性，同时保持高分辨率细节
    * 通过对扩散上采样器进行“时间对齐”微调，把原本的图像超分辨扩散模型变成“时间一致”的视频超分模型
    * 在真实驾驶场景视频合成和文本生成视频任务上取得 SOTA 或接近 SOTA 的结果，证明方法在工业和创意内容场景中的实用性
    * 时间层可以迁移到不同的 Stable Diffusion 变体上，支持个性化文本生成视频（如 DreamBooth 风格的个性化），降低了构建专用 T2V 模型的成本
</details>
</details>


---

<details>
<summary><b>Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation</b></summary>

* **Authors:** David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, Mike Zheng Shou
* **arXiv ID:** 2309.15818
* **One-liner:** This work introduces Show-1, the first hybrid text-to-video model featuring a “pixel diffusion + latent space diffusion” cascade: first generating low-resolution yet text-aligned videos via pixel-domain video diffusion, then performing super-resolution using latent space diffusion. This approach combines the strong alignment of pixel models with the efficiency of latent models, achieving state-of-the-art performance across multiple T2V benchmarks.
* **Published in:** IJCV 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2309.15818) | [[PDF]](https://arxiv.org/pdf/2309.15818) | [[Code]](https://github.com/showlab/Show-1)

> **核心创新**  
> Show-1 首次把“像素域视频扩散 + 潜空间视频扩散”做成真正协同的级联混合框架：用像素扩散生成强文本对齐的低分辨率视频，再通过潜空间“专家翻译”超分到高分辨率，从而同时解决了像素模型太贵、潜模型对齐差这对典型矛盾。
<details>
    <summary>Abstract</summary>
    在大规模预训练文本到视频扩散模型（VDMs）领域取得了显著进展。然而，以往的方法要么完全依赖基于像素的VDMs，这会带来高计算成本，要么依赖基于潜在空间的VDMs，这通常在精确的文本-视频对齐方面存在困难。在本文中，我们首次提出了一种混合模型，称为Show-1，它结合了基于像素和基于潜在空间的VDMs用于文本到视频的生成。我们的模型首先使用基于像素的VDMs生成具有强文本-视频相关性的低分辨率视频。在此之后，我们提出了一种新颖的专家翻译方法，利用基于潜在空间的VDMs进一步将低分辨率视频上采样到高分辨率，同时还可以去除低分辨率视频中的潜在伪影和损坏。与潜在视频扩散模型相比，Show-1 能够生成具有精确文本-视频对齐的高质量视频；与像素级视频扩散模型相比，Show-1 的效率要高得多（推理期间的 GPU 内存使用为 15G，而不是 72G）。此外，我们的 Show-1 模型可以通过简单的时间注意力层微调，轻松适应运动定制和视频风格化应用。我们的模型在标准视频生成基准上达到了最先进的性能。
</details>

<details>
    <summary>Key points</summary>
    * 首次提出将像素域 VDM 与潜空间 VDM 级联的混合 T2V 框架，用像素模型负责强文本对齐、潜模型负责高效超分
    * 先用像素扩散生成低分辨率、语义对齐好的一段视频，再通过“expert translation”让潜空间扩散进行高分辨率重建与细节修复
    * 在保持像素模型级别文本-视频对齐质量的同时，将推理显存从约 72GB 降到 15GB，大幅降低计算成本
    * 通过对时间注意力层进行简单微调，即可适配运动定制（motion customization）和视频风格化等下游应用
    * 在 UCF-101、MSR-VTT 等标准文本到视频生成基准上取得 SOTA 或竞争性结果，证明混合扩散策略的有效性
</details>
</details>

---

<details>
<summary><b>Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets</b></summary>

* **Authors:** Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach
* **arXiv ID:** 2311.15127
* **One-liner:** This work introduces Stable Video Diffusion, which systematically decomposes the process into three stages: “image pre-training → large-scale video pre-training → high-quality video fine-tuning.” Combined with a meticulous subtitle and data filtering workflow, it trains an open-source latent space video diffusion model that achieves state-of-the-art performance across multiple tasks. This enables high-resolution text/image-to-video generation and downstream motion prior transfer.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2311.15127) | [[PDF]](https://arxiv.org/pdf/2311.15127) | [[Code]](https://github.com/Stability-AI/generative-models)

> **核心创新**  
> 把“文本图像预训练 → 大规模视频预训练 → 高质量视频微调”三阶段训练流程和系统化视频数据筛选/字幕标注策略结合起来，给出了一个可复用的“如何在大规模数据上正确训练 latent 视频扩散模型”的完整范式，并证明其在 T2V / I2V、多视角等多任务上都能作为强通用运动先验
<details>
    <summary>Abstract</summary>
    我们介绍了稳定视频扩散（Stable Video Diffusion）——一种潜在的视频扩散模型，用于高分辨率、最先进的文本到视频和图像到视频生成。最近，经过训练用于二维图像合成的潜在扩散模型通过插入时间层并在小型高质量视频数据集上进行微调，已经转化为生成视频的模型。然而，文献中的训练方法差异很大，该领域尚未就视频数据的统一策​​略达成一致。在本文中，我们识别并评估了视频潜在扩散模型成功训练的三个不同阶段：文本到图像预训练、视频预训练以及高质量视频微调。此外，我们展示了策划良好的预训练数据集对于生成高质量视频的必要性，并提出了一个系统化的策划流程来训练强大的基础模型，包括字幕生成和过滤策略。随后，我们探索了在高质量数据上微调基础模型的影响，并训练了一个能够与闭源视频生成竞争的文本到视频模型。我们还展示了我们的基础模型为下游任务提供了强大的运动表示，例如图像到视频生成以及适配于特定摄像机运动的LoRA模块。最后，我们展示了我们的模型提供了强大的多视角三维先验，并能够作为基础，微调一个多视角扩散模型，该模型能够以前馈方式联合生成对象的多视角，性能超过基于图像的方法，同时计算成本仅为其一小部分。
</details>

<details>
    <summary>Key points</summary>
    * 系统提出“文本图像预训练 → 大规模视频预训练 → 高质量视频微调”三阶段流程，给出训练 latent 视频扩散模型的通用套路
    * 设计完整的视频数据构建与筛选管线（质量过滤、去重、字幕/描述生成等），证明预训练数据质量对最终视频生成效果具有持续影响
    * 在同一 latent 视频扩散 backbone 上，通过不同条件编码器支持文本生成视频和图像生成视频，模型内部学到强泛化的“运动与三维先验”
    * 展示了将基础模型微调用于 I2V、相机运动 LoRA、多视角 3D 合成等多种任务的效果，证明其可作为通用视频生成/运动先验基座模型
    * 公开模型权重与训练代码，为后续工作在此基础上做视频编辑、运动控制、加速蒸馏等研究提供了标准基线
</details>
</details>


---

<details>
<summary><b>MagicVideo: Efficient Video Generation With Latent Diffusion Models</b></summary>

* **Authors:** Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, Jiashi Feng
* **arXiv ID:** 2211.11018
* **One-liner:** This work introduces MagicVideo, an efficient text-to-video generation framework based on latent diffusion. By leveraging low-dimensional latent space modeling, a 3D U-Net architecture, and reusing T2I weights, it generates 256×256 resolution videos with strong semantic alignment and smooth temporal transitions on a single GPU. Compared to pixel-domain VDM, it achieves approximately 64 times lower computational cost.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2211.11018) | [[PDF]](https://arxiv.org/pdf/2211.11018) | [[Code]](https://magicvideo.github.io/)

> **核心创新**  
> 把文本到图像的 latent diffusion 模型“迁过去当视频骨干”，通过在潜空间中引入 3D U-Net + 帧间定向时间注意力 + 逐帧轻量适配器三件套，让原本只会生图的 U-Net 在几乎不增加计算量的情况下学会生成时序连贯的视频
<details>
    <summary>Abstract</summary>
    我们提出了一个高效的文本到视频生成框架，基于潜在扩散模型，称为 MagicVideo。MagicVideo 可以生成与给定文本描述一致的平滑视频片段。由于采用了新颖且高效的 3D U-Net 设计，并在低维空间中建模视频分布，MagicVideo 可以在单张 GPU 卡上合成 256×256 空间分辨率的视频片段，这相比视频扩散模型（VDM）在 FLOPs 上的计算量减少了约 64 倍。具体来说，与现有直接在 RGB 空间训练视频模型的方法不同，我们使用预训练的 VAE 将视频片段映射到低维潜空间，并通过扩散模型学习视频潜在编码的分布。此外，我们提出了两种新设计，以将训练于图像任务的 U-Net 去噪器适配到视频数据：用于图像到视频分布调整的逐帧轻量适配器，以及用于捕捉帧间时序依赖的定向时序注意力模块。因此，我们可以利用文本到图像模型中卷积算子的有效权重来加速视频训练。为改善生成视频中的像素抖动，我们还提出了一种新型 VideoVAE 自编码器，以实现更好的 RGB 重建。我们进行了大量实验，并证明 MagicVideo 可以生成高质量的、无论是现实还是虚构内容的视频片段。
</details>

<details>
    <summary>Key points</summary>
    * 在 VAE 潜空间中进行视频扩散建模，相比像素域视频扩散大幅降低计算和显存开销，同时保持较高生成质量
    * 在预训练文本到图像 latent diffusion 的基础上进行最小改动扩展到视频，通过初始化与权重共享，充分利用已有图像生成能力
    * 使用 3D U-Net 在潜空间同时建模空间和时间结构，引入跨帧的时序信息，使生成视频在运动上更加连贯自然
    * 在原有 2D 结构上增加少量时间层/适配模块，而非从头训练完整视频模型，实现训练成本与推理成本之间的良好平衡
    * 在 256×256 分辨率上能生成语义对齐好、时序平滑的视频，相比类似质量的像素域 VDM 具有数量级的加速和资源节省，为后续 latent T2V 工作奠定基础
</details>
</details>

---

<details>
<summary><b>Make Pixels Dance: High-Dynamic Video Generation</b></summary>

* **Authors:** Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, Hang Li
* **arXiv ID:** 2311.10982
* **One-liner:** This work introduces PixelDance, a high-motion video generation method that simultaneously incorporates “first-frame + last-frame image prompts + text conditions” into diffusion models. Specifically addressing the lack of dynamic content in existing T2V methods, it significantly enhances video synthesis capabilities for complex scenes and substantial motion.
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2311.10982) | [[PDF]](https://arxiv.org/pdf/2311.10982) | [[Code]](https://makepixelsdance.github.io/)

> **核心创新**  
> 在扩散视频生成中同时引入“首帧 + 末帧图像指令 + 文本指令”的三重条件，并配套相应的训练/推理机制，让模型直接从图像指令中学习复杂场景与大幅度运动，从而把现有 T2V 中“动作太少”的问题显著缓解。
<details>
    <summary>Abstract</summary>
    在人工智能领域，创建高动态的视频，如动作丰富的场景和复杂的视觉特效，是一个重大挑战。不幸的是，目前最先进的视频生成方法，主要集中于文本到视频的生成，尽管保持了高保真度，但生成的视频片段运动量通常非常有限。我们认为，仅依靠文本指令对于视频生成是不充分且次优的。在本文中，我们提出了PixelDance，这是一种基于扩散模型的新方法，在视频生成中结合了用于第一帧和最后一帧的图像指令以及文本指令。全面的实验结果表明，使用公开数据训练的PixelDance在合成复杂场景和精细动作的视频方面表现出显著更高的能力，为视频生成设立了新的标准。
</details>

<details>
    <summary>Key points</summary>
    * 专门针对现有 T2V 模型“动作少、特效弱”的问题，聚焦生成包含大幅运动、复杂效果和机位变化的高动态视频
    * 在扩散框架中同时使用文本指令 + 首帧图像指令 + 末帧图像指令，让模型既抓住语义，又从图像中直接学习复杂场景与运动起止状态
    * 在 VAE 潜空间里对视频进行扩散建模，以 latent diffusion 方式兼顾生成质量与计算效率
    * 利用真值视频的第一/最后一帧作为训练监督，并设计分段生成、帧衔接等推理技巧，可叠加生成更长、有清晰故事线的视频
    * 仅用公开数据（如 WebVid）训练，就能在复杂场景和大幅动作视频的合成能力上显著超越现有方法，生成三分钟连续剧情视频，树立高动态视频生成的新基线
</details>
</details>


---

<details>
<summary><b>Latent Video Diffusion Models for High-Fidelity Long Video Generation</b></summary>

* **Authors:** Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, Qifeng Chen
* **arXiv ID:** 2211.13221
* **One-liner:** This work introduces Latent Video Diffusion Models (LVDM), which employ hierarchical diffusion modeling in a low-dimensional 3D latent space. By integrating hierarchical generation, conditional noise perturbation, and unconditional guidance, LVDM achieves the generation of high-fidelity long videos extending to thousands of frames with limited computational resources.
* **Published in:** arXiv preprint
* **Links:** [[Paper]](https://arxiv.org/abs/2211.13221) | [[PDF]](https://arxiv.org/pdf/2211.13221) | [[Code]](https://github.com/YingqingHe/LVDM)

> **核心创新**  
> 把视频生成迁移到低维 3D 潜空间，并在其中提出“分层扩散 + 条件潜变量扰动 + 无条件引导”的整套机制，在有限算力下首次同时兼顾高保真与上千帧长视频的稳定生成。
<details>
    <summary>Abstract</summary>
    人工智能生成的内容最近吸引了大量关注，但照片级真实感的视频合成仍然具有挑战性。尽管在这一领域已经使用 GAN 和自回归模型进行了许多尝试，但生成视频的视觉质量和长度仍远未令人满意。扩散模型最近显示出了显著的效果，但需要大量计算资源。为了解决这一问题，我们通过利用低维 3D 潜在空间引入轻量级视频扩散模型，在有限的计算预算下显著优于以往的像素空间视频扩散模型。此外，我们在潜在空间中提出了分层扩散，使得能够生成超过一千帧的长视频。为进一步克服长视频生成中性能下降的问题，我们提出了条件潜在扰动和无条件引导，有效缓解了视频长度延长过程中累积的误差。在不同类别的小规模数据集上进行的大量实验表明，我们的框架生成的视频比以往的强基线更真实、更长。我们还提供了对大规模文本到视频生成的扩展，以展示我们工作的优越性。
</details>

<details>
    <summary>Key points</summary>
    * 用视频 VAE 把高维视频映射到紧凑的 3D 潜空间，在极大降低计算与显存开销的前提下训练视频扩散模型
    * 在潜空间中设计分层/层级扩散架构，先生成短片段或低频信息，再逐步扩展和细化，从而支持上千帧的长视频生成
    * 通过统一的视频扩散模型 + 自回归生成策略，让模型可以在潜空间中不断接续已有片段，理论上生成“任意长度”的视频
    * 提出 conditional latent perturbation 结合 unconditional guidance，在扩展长视频时抑制误差累积，提升长时序的一致性与稳定性
    * 在多类小域数据集上生成的长视频质量明显优于前人，并进一步扩展到大规模文本生成视频任务，验证了框架的泛化能力
</details>
</details>

---

<details>
<summary><b>TriDet: Temporal Action Detection with Relative Boundary Modeling</b></summary>

* **Authors:** Dingfeng Shi, Yujie Zhong, Qiong Cao, Lin Ma, Jia Li, Dacheng Tao
* **arXiv ID:** 2303.07347
* **One-liner:** This work introduces TriDet, a one-stage framework for temporal action detection. By employing a Trident-head with “relative boundary probability distribution” and an Scalable Granularity-Aware (SGP) feature pyramid to finely model action boundaries, it significantly improves detection accuracy while reducing computational overhead across multiple mainstream datasets.
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2303.07347) | [[PDF]](https://arxiv.org/pdf/2303.07347) | [[Code]](https://github.com/dingfengshi/TriDet)

> **核心创新**  
> 用“相对边界概率分布”的 Trident-head 精细建模动作起止点，并结合可扩展粒度感知（SGP）特征金字塔提升不同时间尺度上的判别力，从而在一阶段框架下显著提高时序动作边界定位精度与效率。
<details>
    <summary>Abstract</summary>
    人工智能生成的内容最近吸引了大量关注，但照片级真实感的视频合成仍然具有挑战性。尽管在这一领域已经使用 GAN 和自回归模型进行了许多尝试，但生成视频的视觉质量和长度仍远未令人满意。扩散模型最近显示出了显著的效果，但需要大量计算资源。为了解决这一问题，我们通过利用低维 3D 潜在空间引入轻量级视频扩散模型，在有限的计算预算下显著优于以往的像素空间视频扩散模型。此外，我们在潜在空间中提出了分层扩散，使得能够生成超过一千帧的长视频。为进一步克服长视频生成中性能下降的问题，我们提出了条件潜在扰动和无条件引导，有效缓解了视频长度延长过程中累积的误差。在不同类别的小规模数据集上进行的大量实验表明，我们的框架生成的视频比以往的强基线更真实、更长。我们还提供了对大规模文本到视频生成的扩展，以展示我们工作的优越性。
</details>

<details>
    <summary>Key points</summary>
    * TriDet 提出一个简单高效的一阶段时序动作检测框架，在端到端结构中同时完成动作分类与边界回归
    * 通过 Trident-head 将动作起止点建模为“边界附近的相对概率分布”，显著缓解传统方法因边界模糊导致的边界预测不准问题
    * 在特征金字塔中引入 Scalable-Granularity Perception（SGP）层，缓解自注意力在视频特征上的 rank-loss 问题，并在多时间尺度上聚合信息
    * 在 THUMOS14、HACS、EPIC-KITCHEN 100 等数据集上取得 SOTA 或竞争性结果，同时推理延迟明显低于以往方法（如在 THUMOS14 上提高 mAP 的同时仅用约 74.6% 的延迟）
    * 在包含层次标签的多种 TAD 数据集上都表现稳定，展示出对不同场景和标注形式的较强泛化能力。
</details>
</details>

---

<details>
<summary><b>Temporal Action Localization with Enhanced Instant Discriminability</b></summary>

* **Authors:** Dingfeng Shi, Qiong Cao, Yujie Zhong, Shan An, Jian Cheng, Haogang Zhu, Dacheng Tao
* **arXiv ID:** 2309.05590
* **One-liner:** This work builds upon TriDet to propose a temporal action detection framework called “Enhanced Instantaneous Discrimination,” which combines an extensible granularity-aware layer, pre-trained large model features, and a decoupled feature pyramid. These designs systematically enhance the discrimination capability at each video time point, thereby achieving new state-of-the-art performance across multiple TAD datasets, including those with hierarchical multi-label tasks.
* **Published in:** IJCV 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2309.05590) | [[PDF]](https://arxiv.org/pdf/2309.05590) | [[Code]](https://github.com/dingfengshi/tridetplus)

> **核心创新**  
> 在 TriDet 的一阶段框架上引入可扩展粒度感知层、预训练大模型特征与解耦特征金字塔等设计，系统增强视频每个时间点的判别能力，从而在不增加复杂度的前提下显著提升时序动作定位精度。
<details>
    <summary>Abstract</summary>
    时间动作检测（TAD）旨在在未经剪辑的视频中检测所有动作边界及其对应的类别。视频中动作边界的不清晰性常导致现有方法对动作边界的预测不准确。为了解决这一问题，我们提出了一个名为 TriDet 的单阶段框架。首先，我们提出了 Trident 头来通过边界周围的估计相对概率分布建模动作边界。然后，我们分析了基于 Transformer 方法中的排序损失问题（即瞬时判别能力下降），并提出了高效的可扩展粒度感知（SGP）层来缓解这一问题。为了进一步提高视频主干网络中的瞬时判别能力，我们利用预训练大型模型的强大表示能力，并研究它们在 TAD 上的性能。最后，考虑到分类所需的充足时空上下文，我们设计了一个解耦特征金字塔网络，使用独立的特征金字塔将大型模型的丰富空间上下文用于定位。实验结果表明，TriDet 具有稳健性，并在多个 TAD 数据集上表现出最先进的性能，包括分层（多标签）TAD 数据集。
</details>

<details>
    <summary>Key points</summary>
    * 聚焦 Transformer 类 TAD 方法中“rank-loss 导致瞬时可判别性下降”的问题，强调要让每个时间点本身更容易区分“动作 / 非动作”
    * 在一阶段 TriDet 框架下工作：用 Trident-head 通过“相对边界概率分布”精细建模动作起止点，同时保持端到端、高效率的检测流程
    * 提出 Scalable-Granularity Perception（SGP）层，通过多粒度特征分支缓解 rank-loss，让瞬时特征在不同时间尺度上都更“清晰可分”
    * 利用预训练大模型的强视觉表征，把其特征引入视频 backbone，显著增强局部时间片段的判别能力。
    * 通过解耦 feature pyramid，将大模型的空间上下文更好地服务于定位，在多种含层次多标签的 TAD 数据集上取得 SOTA 或竞争性结果，验证方法泛化性和鲁棒性
</details>
</details>

---

<details>
<summary><b>BasicTAD: an Astounding RGB-Only Baseline for Temporal Action Detection</b></summary>

* **Authors:** Min Yang, Guo Chen, Yin-Dong Zheng, Tong Lu, Limin Wang
* **arXiv ID:** 2205.02717
* **One-liner:** BasicTAD systematically rebuilt a “pure RGB, end-to-end, modular” temporal action detection baseline (BasicTAD/PlusTAD). Without introducing complex structures or additional modalities like optical flow, it achieved significant improvements or approached SOTA performance across multiple mainstream TAD datasets through meticulous design and combination of data augmentation, backbone, neck, and detection heads. This work redefines the performance ceiling for RGB-only baselines.
* **Published in:** CVIU 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2205.02717) | [[PDF]](https://arxiv.org/pdf/2205.02717) | [[Code]](https://github.com/MCG-NJU/BasicTAD)

> **核心创新**  
> 在不发明新模块的前提下，把 TAD 管线系统拆解为采样、骨干、neck 和检测头四个基础部件，基于 RGB-only 与端到端训练，给出一套经过充分实证验证的极简模块化设计，并把这种“最朴素组合”直接推到接近乃至赶超两流/SOTA 的性能上，从而重新定义了 TAD 领域的强基线
<details>
    <summary>Abstract</summary>
    时序动作检测（Temporal Action Detection, TAD）在视频理解领域得到了广泛研究，通常沿用图像中的目标检测流程。然而，在TAD中复杂设计并不少见，例如双流特征提取、多阶段训练、复杂的时序建模以及全局上下文融合。在本文中，我们并不打算为TAD引入任何新技术。相反，我们研究了一个简单、直接但必须了解的基线模型，鉴于当前TAD中复杂设计和低检测效率的现状。在我们的简单基线（BasicTAD）中，我们将TAD流程分解为几个核心组成部分：数据采样、主干网络设计、颈部构建和检测头。我们对每个组成部分的现有技术进行了广泛研究，更重要的是，由于设计的简单性，我们能够对整个流程进行端到端训练。因此，这个简单的BasicTAD实现了一个令人惊叹且实时的仅RGB基线，其性能非常接近使用双流输入的最先进方法。此外，我们通过在网络表示中保留更多的时间和空间信息进一步改进了BasicTAD（称为PlusTAD）。实验结果表明，我们的PlusTAD非常高效，并且在THUMOS14和FineAction数据集上显著优于之前的方法。同时，我们还对提出的方法进行了深入的可视化和错误分析，并尝试对TAD问题提供更多见解。我们的方法可以作为未来TAD研究的强基线。
</details>

<details>
    <summary>Key points</summary>
    * 系统搭建并打磨了一个仅用 RGB 输入、端到端训练的 TAD 基线 BasicTAD，在不引入光流或复杂结构的前提下接近甚至追平两流 SOTA 方法
    * 把 TAD 流程拆成采样、骨干、neck、检测头四个基础模块，在每个模块内系统比较现有设计，给出一套“简单但好用”的组合实践
    * 利用结构简单的优势，实现从视频输入到动作边界与类别预测的真正端到端训练，相比依赖预提特征的多阶段方法更统一、更高效
    * 在 BasicTAD 基础上通过“保留更多时空信息”的最小改动得到 PlusTAD，在 THUMOS14、FineAction 等数据集上显著超过现有方法
    * 通过大量消融和误差分析揭示不同设计对 TAD 性能的影响，为后续方法提供可复用的强基线和设计指南。
</details>
</details>

---

<details>
<summary><b>Action Sensitivity Learning for Temporal Action Localization</b></summary>

* **Authors:** Jiayi Shao, Xiaohan Wang, Ruijie Quan, Junjun Zheng, Jiang Yang, Yi Yang
* **arXiv ID:** 2305.15701
* **One-liner:** This work proposes the Action Sensitivity Learning (ASL) framework for temporal action localization. By learning the “action sensitivity” of each frame at both the category and instance levels, it explicitly distinguishes key frames from redundant frames through reweighted gradient training and a designed action-sensitive contrastive loss. This approach achieves significant improvements in average mAP across multiple TAL datasets.
* **Published in:** ICCV 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2305.15701) | [[PDF]](https://arxiv.org/pdf/2305.15701) 
> **核心创新**  
> ASL 的核心创新在于显式学习“每一帧有多重要”的动作敏感度，用一个轻量评估器同时给出类别级和实例级敏感度，并用它来重加权分类/边界回归的训练梯度、再配合动作敏感对比损失，从而把关键帧和冗余帧区分开来，大幅提升时序动作定位性能
<details>
    <summary>Abstract</summary>
    时序动作定位（Temporal Action Localization, TAL）涉及识别和定位动作实例，是视频理解中的一项具有挑战性的任务。大多数现有方法直接预测动作类别并回归边界偏移，而忽略了每一帧的重要性差异。在本文中，我们提出了一种动作敏感学习框架（Action Sensitivity Learning, ASL）来解决这一任务，旨在评估每一帧的价值，并利用生成的动作敏感性重新校准训练过程。我们首先引入了一个轻量级动作敏感性评估器，分别在类别级别和实例级别学习动作敏感性。两个分支的输出被结合起来，用于对两个子任务的梯度进行重新加权。此外，基于每帧的动作敏感性，我们设计了一种动作敏感对比损失（Action Sensitive Contrastive Loss）来增强特征，其中动作相关帧被采样为正样本对，以推动动作无关帧远离。针对各种动作定位基准（如 MultiThumos、Charades、Ego4D-Moment Queries v1.0、Epic-Kitchens 100、Thumos14 和 ActivityNet1.3）的广泛研究表明，ASL 在多种场景（如单标签、密集标签和自我视角）下的平均 mAP 上超越了现有最先进的方法。
</details>

<details>
    <summary>Key points</summary>
    * 指出现有 TAL 方法普遍“把所有帧一视同仁”，忽略关键帧与冗余帧的重要性差异，并把“动作敏感度”作为显式建模目标
    * 提出轻量级 Action Sensitivity Evaluator，同时输出类别级和实例级两种敏感度，用于衡量每一帧对分类和边界定位的贡献
    * 利用帧级敏感度对分类与边界回归损失的梯度进行重加权，让模型在训练中自动“多关注关键帧、少被背景噪声牵着走”
    * 基于帧敏感度设计 Action Sensitive Contrastive Loss，将动作相关帧作为正样本、动作无关帧作为负样本，进一步拉开特征间距、提升判别性
    * 在 MultiThumos、Charades、Ego4D-MQ、EPIC-Kitchens 100、THUMOS14 和 ActivityNet1.3 等多种（单标签/密集/自中心）TAL 基准上平均 mAP 全面领先当前方法，验证框架的通用性与有效性。
</details>
</details>

---

<details>
<summary><b>TemporalMaxer: Maximize Temporal Context with only Max Pooling for Temporal Action Localization</b></summary>

* **Authors:** Tuan N. Tang, Kwonyoung Kim, Kwanghoon Sohn
* **arXiv ID:** 2303.09055
* **One-liner:** This work introduces TemporalMaxer, a minimalist temporal attention framework that replaces complex self-attention and other long-range modeling mechanisms with “local max-pooling blocks.” We demonstrate that even with only the most fundamental max-pooling operations and significantly reduced parameters and computational complexity, TemporalMaxer can achieve performance on par with or surpass existing state-of-the-art methods across multiple temporal action localization datasets.
* **Published in:** ICCV 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2303.09055) | [[PDF]](https://arxiv.org/pdf/2303.09055) | [[Code]](https://github.com/TuanTNG/TemporalMaxer)
> **核心创新**  
> TemporalMaxer 的核心创新在于彻底抛弃复杂的长时序自注意/图结构，仅用一个“局部、无参数的 max-pooling 块”作为骨干里的时序建模单元，在最大化预训练 3D CNN 特征信息利用的同时最小化长时序建模复杂度，从而以极简结构实现与甚至超越 Transformer 等重型模型的 TAL 性能
<details>
    <summary>Abstract</summary>
    时序动作定位（TAL）是视频理解领域的一项具有挑战性的任务，旨在识别并定位视频序列中的动作。近期的研究强调了将长期时序上下文建模（TCM）模块应用于提取的视频片段特征的重要性，例如采用复杂的自注意力机制。在本文中，我们提出了迄今为止最简便的方法来解决这一任务，并且认为提取的视频片段特征本身已经足够信息量，可以在不依赖复杂架构的情况下实现出色的性能。为此，我们引入了TemporalMaxer，它在尽量减少长期时序上下文建模的同时，通过一个基本、无参数且在局部区域操作的最大池化模块最大化提取的视频片段特征的信息。该模块仅提取相邻和局部片段嵌入中最关键的信息，从而形成一个更加高效的TAL模型。我们证明，TemporalMaxer在多个TAL数据集上优于其他利用长期TCM（如自注意力机制）的最先进方法，同时所需的参数和计算资源显著减少。
</details>

<details>
    <summary>Key points</summary>
    * 完全放弃 Transformer、自注意力和图结构等复杂 TCM，仅用局部、无参数的 max-pooling 作为唯一的时序建模单元，构成目前最简单的 TAL 主干之一
    * 基于预训练 3D CNN（如 I3D/TSN）提取的 clip 特征，主张这些特征本身已足够强，通过 max-pooling 只“挑出最关键的局部信息”，而不是再堆复杂结构
    * 通过在相邻时间窗口上做局部 max-pooling，将局部片段内最显著的响应集中起来，在极低计算代价下仍能捕获有效的时序上下文
    * 在 THUMOS14 等多种 TAL 基准上，TemporalMaxer 以远少于自注意/Transformer 方法的参数量和 FLOPs，达到或超过 SOTA 性能
    * 后续基准研究表明，在数据有限或算力受限场景下，TemporalMaxer 依然表现突出，是兼具数据效率与计算效率的 TAL 模型代表
</details>
</details>

---

<details>
<summary><b>Contextual Object Detection with Multimodal Large Language Models</b></summary>

* **Authors:** Yuhang Zang, Wei Li, Jun Han, Kaiyang Zhou, Chen Change Loy
* **arXiv ID:** 2305.18279
* **One-liner:** This work introduces the novel problem of “contextual object detection” and constructs the unified multimodal model ContextDET. It integrates a visual encoder, a pre-trained large language model, and a detection head into a “semantic generation-first, object detection-second” framework. This enables MLLMs to locate, identify, and describe visible objects using natural language in interactive scenarios such as language cloze tests, image description, and question-answering, even with open-ended vocabularies.
* **Published in:** IJCV 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2305.18279) | [[PDF]](https://arxiv.org/pdf/2305.18279) | [[Code]](https://github.com/yuhangzang/ContextDET)
> **核心创新**  
> ContextDET 的核心创新在于把“目标检测”嵌进多模态大模型的语言推理过程中，用“视觉编码器 + 冻结 LLM + 检测头”的统一框架，把原本独立的检测任务变成上下文感知的语言生成结果，从而在开放词汇、问答、描述等交互场景下实现真正“按语境来检测”的上下文目标检测
<details>
    <summary>Abstract</summary>
    近期多模态大型语言模型（MLLM）在视觉语言任务（如图像描述生成与问答）中表现卓越，却缺乏关键感知能力——即物体检测。本研究通过提出情境化物体检测这一新研究范式，致力于突破该局限：理解不同人机交互场景中的可见物体。我们研究了三个典型场景：语言填空测试、视觉描述生成及问答系统。同时提出统一的多模态模型ContextDET，该模型能对视觉-语言上下文进行端到端可微建模，从而实现视觉对象定位、识别及与语言输入的关联，支持人机交互。ContextDET包含三大核心子模型：(i) 视觉编码器用于提取视觉表征，(ii) 预训练大型语言模型实现多模态上下文解码，(iii) 视觉解码器根据上下文物体词预测边界框。创新的生成-检测框架使我们能够检测人类词汇中的物体词汇。大量实验表明，ContextDET在我们提出的CODE基准测试、开放词汇检测以及指称图像分割任务中均展现出显著优势。
</details>

<details>
    <summary>Key points</summary>
    * 正式提出“Contextual Object Detection”新问题，让模型在完形填空、描述、问答等语言上下文中，识别并定位与语境相关的可见物体
    * 设计统一的 ContextDET 框架，将视觉编码器、预训练 LLM 和视觉解码器整合起来，端到端建模图像与语言上下文的交互
    * 提出“先生成语义再检测”的 generate-then-detect 流程：LLM 先在上下文中生成对象词，再由视觉解码器根据这些词去预测对应的目标框
    * 得益于 LLM 的词汇与推理能力，模型能够在开放词表下进行目标检测和语言对齐，而不局限于固定的类别集合
    * 构建 CODE 基准并在开放词汇检测、referring segmentation 等任务上实验，表明 ContextDET 在多种上下文感知视觉任务上明显优于现有方法
</details>
</details>

---

<details>
<summary><b>ActionFormer: Localizing Moments of Actions with Transformers</b></summary>

* **Authors:** Chenlin Zhang, Jianxin Wu, Yin Li
* **arXiv ID:** 2202.07925
* **One-liner:** This work introduces Latent Video Diffusion Models (LVDM), which employ hierarchical diffusion modeling in a low-dimensional 3D latent space. By integrating hierarchical generation, conditional noise perturbation, and unconditional guidance, LVDM achieves the generation of high-fidelity long videos extending to thousands of frames with limited computational resources.
* **Published in:** ECCV 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2202.07925) | [[PDF]](https://arxiv.org/pdf/2202.07925) | [[Code]](https://github.com/happyharrycn/actionformer_release)

> **核心创新**  
> 把视频生成迁移到低维 3D 潜空间，并在其中提出“分层扩散 + 条件潜变量扰动 + 无条件引导”的整套机制，在有限算力下首次同时兼顾高保真与上千帧长视频的稳定生成。
<details>
    <summary>Abstract</summary>
    人工智能生成的内容最近吸引了大量关注，但照片级真实感的视频合成仍然具有挑战性。尽管在这一领域已经使用 GAN 和自回归模型进行了许多尝试，但生成视频的视觉质量和长度仍远未令人满意。扩散模型最近显示出了显著的效果，但需要大量计算资源。为了解决这一问题，我们通过利用低维 3D 潜在空间引入轻量级视频扩散模型，在有限的计算预算下显著优于以往的像素空间视频扩散模型。此外，我们在潜在空间中提出了分层扩散，使得能够生成超过一千帧的长视频。为进一步克服长视频生成中性能下降的问题，我们提出了条件潜在扰动和无条件引导，有效缓解了视频长度延长过程中累积的误差。在不同类别的小规模数据集上进行的大量实验表明，我们的框架生成的视频比以往的强基线更真实、更长。我们还提供了对大规模文本到视频生成的扩展，以展示我们工作的优越性。
</details>

<details>
    <summary>Key points</summary>
    * 用视频 VAE 把高维视频映射到紧凑的 3D 潜空间，在极大降低计算与显存开销的前提下训练视频扩散模型
    * 在潜空间中设计分层/层级扩散架构，先生成短片段或低频信息，再逐步扩展和细化，从而支持上千帧的长视频生成
    * 通过统一的视频扩散模型 + 自回归生成策略，让模型可以在潜空间中不断接续已有片段，理论上生成“任意长度”的视频
    * 提出 conditional latent perturbation 结合 unconditional guidance，在扩展长视频时抑制误差累积，提升长时序的一致性与稳定性
    * 在多类小域数据集上生成的长视频质量明显优于前人，并进一步扩展到大规模文本生成视频任务，验证了框架的泛化能力
</details>
</details>

---
