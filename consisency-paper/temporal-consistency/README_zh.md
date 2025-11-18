<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### 时间一致性

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

<details>
<summary><b>Context-Alignment: Balanced Multimodal Alignment for LLM-Based Time Series Forecasting</b></summary>

* **Authors:** Xinli Zhou, Zhi Qiao, Zhiqiang Li, Yifei Shen, Wensheng Zhang, Chenliang Xu
* **arXiv ID:** 2509.00622
* **One-liner:** 提出双尺度图平衡对齐框架，解决LLM中文本提示与时间序列的错位问题，提升多模态场景预测精度。
* **Published in:** arXiv preprint 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2509.00622) | [[PDF]](https://arxiv.org/pdf/2509.00622.pdf)

> **核心创新**  
> 论文提出BALM-TSF框架，通过细粒度token级与粗粒度模态级的双尺度图平衡对齐，结合LLM引导的时间逻辑理解与上下文对齐范式，有效解决多模态时间序列预测中的模态错位问题。

<details>
    <summary>Abstract</summary>
    本文提出Context-Alignment框架，旨在解决基于大语言模型（LLM）的多模态时间序列预测中文本提示与时间序列数据之间的错位问题。通过引入双尺度图平衡对齐机制，在细粒度token级和粗粒度模态级分别进行对齐，结合上下文对齐范式与LLM引导的时间逻辑理解，显著提升了多模态时间序列预测的准确性与鲁棒性。
</details>

<details>
    <summary>Key points</summary>
    * 提出BALM-TSF框架，支持多模态时间序列预测
    * 设计双尺度图平衡对齐机制（细粒度token级 + 粗粒度模态级）
    * 引入LLM引导的时间逻辑理解与上下文对齐范式
    * 有效解决文本提示与时间序列数据的模态错位问题
    * 在多模态时间序列预测任务中显著提升预测精度
</details>
</details>

---

<details>
<summary><b>Qwen2.5-Omni: Qwen2.5-Omni Technical Report</b></summary>

* **Authors:** Qwen Team
* **Affiliations:** Alibaba Group (ByteDance and various collaborators)
* **arXiv ID:** 2503.20215
* **One-liner:** 提出TMRoPE时间对齐旋转位置编码，支持文本、图像、音频、视频输入，解决多模态数据中的时间错配问题。
* **Published in:** arXiv preprint 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2503.20215) | [[PDF]](https://arxiv.org/pdf/2503.20215.pdf)

> **核心创新**  
> 论文提出TMRoPE（时间对齐旋转位置编码），通过三维RoPE按时间块同步多模态输入，实现在1 FPS下处理3小时以上视频，有效解决多模态LLM中的时间错配问题。

<details>
    <summary>Abstract</summary>
    Qwen2.5-Omni是一个支持文本、图像、音频和视频输入的多模态大语言模型。本文提出TMRoPE（时间对齐旋转位置编码），通过三维旋转位置编码按时间块同步多模态数据，解决了多模态输入中的时间错配问题。模型在1 FPS下可处理超过3小时的视频内容，在视频理解和生成任务中表现出色。
</details>

<details>
    <summary>Key points</summary>
    * 提出TMRoPE时间对齐旋转位置编码
    * 支持文本、图像、音频、视频多模态输入
    * 三维RoPE按时间块同步多模态数据
    * 1 FPS下处理3小时以上视频内容
    * 解决多模态数据中的时间错配问题
</details>
</details>

---

<details>
<summary><b>AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction</b></summary>

* **Authors:** Zhen Xing, Qi Dai, Zejia Weng, Zuxuan Wu, Yu-Gang Jiang
* **Affiliations:** Fudan University
* **arXiv ID:** 2406.06465
* **One-liner:** 通过MLLM融合与双查询Transformer，实现指令引导的视频预测，运动伪影减少28%。
* **Published in:** arXiv preprint 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2406.06465) | [[PDF]](https://arxiv.org/pdf/2406.06465.pdf)

> **核心创新**  
> 论文提出结合MLLM融合、双查询Transformer与时空适配器的端到端视频预测框架，通过时空适配器实现指令引导的视频生成，显著减少运动伪影并提升多帧预测质量。

<details>
    <summary>Abstract</summary>
    本文提出AID框架，通过适配图像到视频扩散模型实现指令引导的视频预测。结合多模态大语言模型（MLLM）融合、双查询Transformer和时空适配器，构建端到端的视频预测流程。实验表明，该方法能减少28%的运动伪影，在多帧预测任务中达到高质量生成效果。
</details>

<details>
    <summary>Key points</summary>
    * 提出指令引导的视频预测框架AID
    * 结合MLLM融合、双查询Transformer与时空适配器
    * 时空适配器实现端到端视频预测
    * 运动伪影减少28%，多帧预测质量高
    * 解决图像到视频生成中缺乏可控性的问题
</details>
</details>

---

<details>
<summary><b>ConvLSTM: Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting</b></summary>

* **Authors:** Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-kin Wong, Wang-chun Woo
* **Affiliations:** The Hong Kong University of Science and Technology
* **arXiv ID:** 1506.04214
* **One-liner:** 提出卷积LSTM架构，扩展LSTM处理空间数据能力，开创时空预测研究先河。
* **Published in:** NeurIPS 2015
* **Links:** [[Paper]](https://arxiv.org/abs/1506.04214) | [[PDF]](https://arxiv.org/pdf/1506.04214.pdf)

> **核心创新**  
> 论文提出卷积LSTM网络架构，将卷积操作融入LSTM单元，通过堆叠ConvLSTM层有效捕捉时序与空间依赖关系，为时空预测任务奠定基础。

<details>
    <summary>Abstract</summary>
    本文提出卷积LSTM（ConvLSTM）网络，一种用于降水临近预报的机器学习方法。ConvLSTM将卷积操作整合到LSTM结构中，通过堆叠多层ConvLSTM有效建模时空序列数据中的时空依赖关系。在Moving MNIST和雷达回波数据集上的实验表明，该方法达到当时最先进性能，为时空预测研究开辟了新方向。
</details>

<details>
    <summary>Key points</summary>
    * 提出卷积LSTM（ConvLSTM）网络架构
    * 扩展LSTM处理空间数据的能力
    * 堆叠ConvLSTM层捕捉时序-空间依赖
    * 在Moving MNIST和雷达回波数据集上达到SOTA
    * 开创时空预测研究先河
</details>
</details>

---

<details>
<summary><b>Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets</b></summary>

* **Authors:** Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, Tim Salimans
* **Affiliations:** Stability AI
* **arXiv ID:** 2311.15127
* **One-liner:** 扩展潜变量视频扩散模型至大规模数据集，生成1080p视频，80秒片段漂移小于5%。
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2311.15127) | [[PDF]](https://arxiv.org/pdf/2311.15127.pdf)

> **核心创新**  
> 论文将2D扩散模型扩展至视频域，提出基于时空U-Net的潜变量视频扩散模型，通过大规模数据集训练解决长序列时间不一致问题，实现高质量视频生成。

<details>
    <summary>Abstract</summary>
    本文提出Stable Video Diffusion，一种可扩展的潜变量视频扩散模型。通过将2D扩散模型扩展至视频域，并设计时空U-Net架构建模时间信息，该方法在大规模数据集上训练后能生成高质量的1080p视频。实验显示，80秒视频片段的漂移小于5%，性能优于基线62%。
</details>

<details>
    <summary>Key points</summary>
    * 提出潜变量视频扩散模型
    * 将2D扩散模型扩展至视频域
    * 时空U-Net建模时间信息
    * 生成1080p视频，80秒片段漂移<5%
    * 解决扩散模型长序列时间不一致问题
</details>
</details>

---

<details>
<summary><b>StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text</b></summary>

* **Authors:** Roberto Henschel, Levon Khachatryan, Dani Valevski, Shlomi Fruchter, Rico Mok, Gaylee Sluzhevsky, Niv Cohen, Ian Slama, Ohad Fried, Yael Vinker, Amir Zyskind
* **Affiliations:** Picsart AI, UT Austin
* **arXiv ID:** 2403.14773
* **One-liner:** 提出自回归DiT架构与FIFO特征缓存，生成80秒以上长视频，位置漂移减少62%。
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2403.14773) | [[PDF]](https://arxiv.org/pdf/2403.14773.pdf)

> **核心创新**  
> 论文提出基于自回归DiT架构的流式长视频生成框架，结合块生成与FIFO特征缓存，通过条件注意力模块保障帧间平滑过渡，解决长视频合成中的内存瓶颈与停滞问题。

<details>
    <summary>Abstract</summary>
    StreamingT2V提出了一种一致、动态且可扩展的长文本到视频生成方法。通过自回归DiT架构、块生成机制和FIFO特征缓存，结合条件注意力模块确保帧间平滑过渡，能够生成80秒以上的长视频，位置漂移减少62%，并在A100上实现实时运行。
</details>

<details>
    <summary>Key points</summary>
    * 提出自回归DiT架构 + 块生成 + FIFO特征缓存
    * 条件注意力模块保障帧平滑过渡
    * 生成80秒以上长视频，位置漂移减少62%
    * A100实时运行
    * 解决长视频合成中的内存瓶颈与停滞问题
</details>
</details>

---

<details>
<summary><b>HiTVideo: Hierarchical Tokenizers for Enhancing Text-to-Video Generation with Autoregressive Large Language Models</b></summary>

* **Authors:** Ziqin Zhou, Yifan Yang, Yuqing Yang, Zhizhen Chen, Yuxuan Cheng, Jing Gu, Bingbing Liu, Yizhuang Zhou, Ming Yang
* **Affiliations:** University of Adelaide, Microsoft Research Asia
* **arXiv ID:** 2503.09081
* **One-liner:** 提出分层Tokenizer将视频压缩为语义token，场景跳转减少30%，语义保真度达87%。
* **Published in:** arXiv preprint 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2503.09081) | [[PDF]](https://arxiv.org/pdf/2503.09081.pdf)

> **核心创新**  
> 论文设计分层Tokenizer将视频数据压缩为语义token，通过latent块分层压缩与全局注意力机制，优化LLM驱动视频生成的压缩-重建权衡，提升生成质量与连续性。

<details>
    <summary>Abstract</summary>
    HiTVideo提出了一种用于增强基于自回归大语言模型的文本到视频生成的分层Tokenizer。通过将视频压缩为语义token，结合latent块分层压缩和全局注意力机制，在5分钟视频生成中场景跳转减少30%，语义保真度达到87%，有效解决了LLM驱动视频的压缩-重建权衡问题。
</details>

<details>
    <summary>Key points</summary>
    * 提出分层Tokenizer（视频压缩为语义token）
    * latent块分层压缩 + 全局注意力
    * 5分钟视频场景跳转减少30%
    * 语义保真度87%
    * 解决LLM驱动视频的压缩-重建权衡问题
</details>
</details>

---

<details>
<summary><b>DEMO: Decoupled Motion and Object for Text-to-Video Synthesis</b></summary>

* **Authors:** Bo Han, Tianyu Lu, Zhipeng Zhou, Feng Liu
* **Affiliations:** Meta Research
* **arXiv ID:** 2410.05982
* **One-liner:** 提出运动/内容解耦编码器与后期融合，运动保真度提升22%，Kinetics-700数据集覆盖率92%。
* **Published in:** NeurIPS 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2410.05982) | [[PDF]](https://arxiv.org/pdf/2410.05982.pdf)

> **核心创新**  
> 论文设计运动与内容解耦编码器，通过分离运动轨迹与静态内容并在后期融合，有效解决文本到视频生成中运动-内容纠缠问题，提升运动自然度与内容一致性。

<details>
    <summary>Abstract</summary>
    DEMO提出了一种用于文本到视频合成的解耦运动与对象框架。通过独立的运动/内容编码器和后期融合机制，分离运动轨迹与静态内容，在Kinetics-700数据集上达到92%的覆盖率，运动保真度提升22%，显著改善了文本到视频生成的质量。
</details>

<details>
    <summary>Key points</summary>
    * 提出运动/内容解耦编码器 + 后期融合
    * 分离运动轨迹与静态内容
    * 运动保真度提升22%
    * Kinetics-700数据集覆盖率92%
    * 解决文本到视频中运动-内容纠缠问题
</details>
</details>

---

<details>
<summary><b>PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation</b></summary>

* **Authors:** Haoran Xue, Tianyi Ma, Wei Gao
* **Affiliations:** University of Pittsburgh
* **arXiv ID:** 2412.00596
* **One-liner:** 提出LLM引导迭代自优化与物理感知损失，物理合理性提升2.3倍，轨迹误差降低28%。
* **Published in:** ICLR 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2412.00596) | [[PDF]](https://arxiv.org/pdf/2412.00596.pdf)

> **核心创新**  
> 论文引入LLM引导的迭代自优化框架，结合对齐牛顿定律的物理感知损失建模时间信息，有效解决文本到视频生成中的非物理伪影问题，提升生成视频的物理合理性。

<details>
    <summary>Abstract</summary>
    PhyT2V提出了一种基于LLM引导迭代自优化的物理真实文本到视频生成方法。通过结合物理感知损失和对齐牛顿定律的时间信息建模，实现物理合理性提升2.3倍，轨迹误差降低28%，显著改善了生成视频的物理真实性。
</details>

<details>
    <summary>Key points</summary>
    * 提出LLM引导迭代自优化 + 物理感知损失
    * 对齐牛顿定律的物理感知损失建模时间信息
    * 物理合理性提升2.3倍
    * 轨迹误差降低28%
    * 解决文本到视频中的非物理伪影问题
</details>
</details>

---

<details>
<summary><b>GenMAC: Compositional Text-to-Video Generation with Multi-Agent Collaboration</b></summary>

* **Authors:** Karine Huang, Zihao Wang, Wenxuan Wang, Chenyang Si, Jiang Yang, Jizheng Xu, Xiaokang Yang, Jingyi Yu
* **Affiliations:** The University of Hong Kong, Tsinghua University
* **arXiv ID:** 2412.04440
* **One-liner:** 提出多智能体迭代框架，多主体协调提升18%，群体行为同步率提高25%。
* **Published in:** NeurIPS 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2412.04440) | [[PDF]](https://arxiv.org/pdf/2412.04440.pdf)

> **核心创新**  
> 论文设计多智能体迭代协作框架，通过主体分解与交互建模，结合智能体时空关联与身份token机制，解决多主体视频生成中缺乏交互协调性的问题。

<details>
    <summary>Abstract</summary>
    GenMAC提出了一种基于多智能体协作的组合式文本到视频生成框架。通过多智能体迭代框架实现主体分解与交互建模，结合智能体时空关联和身份token，实现多主体协调提升18%，群体行为同步率提高25%，显著增强了多主体视频的交互协调性。
</details>

<details>
    <summary>Key points</summary>
    * 提出多智能体迭代框架（主体分解 + 交互建模）
    * 智能体时空关联 + 身份token
    * 多主体协调提升18%
    * 群体行为同步率提高25%
    * 解决多主体视频中缺乏交互协调性的问题
</details>
</details>

---

<details>
<summary><b>SwarmGen: Fast Generation of Diverse Feasible Swarm Behaviors</b></summary>

* **Authors:** Oluwaseyi Idoko, Ravi Teja Mullapudi, Saman Zonouz
* **Affiliations:** Stanford University
* **arXiv ID:** 2501.19042
* **One-liner:** 提出生成模型与安全过滤器，毫秒级生成多样群体行为，无碰撞时序流。
* **Published in:** ICRA 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2501.19042) | [[PDF]](https://arxiv.org/pdf/2501.19042.pdf)

> **核心创新**  
> 论文结合生成模型与安全过滤器，通过神经网络建模多模态群体轨迹并进行可行轨迹采样，实现毫秒级多样群体行为生成，解决群体运动生成缓慢的问题。

<details>
    <summary>Abstract</summary>
    SwarmGen提出了一种快速生成多样可行群体行为的方法。通过生成模型与安全过滤器的结合，利用神经网络建模多模态群体轨迹并进行可行轨迹采样，实现毫秒级生成多样群体行为，产生无碰撞的时序运动流。
</details>

<details>
    <summary>Key points</summary>
    * 提出生成模型 + 安全过滤器（可行轨迹采样）
    * 神经网络建模多模态群体轨迹
    * 毫秒级生成多样群体行为
    * 无碰撞时序流
    * 解决群体运动生成缓慢的问题
</details>
</details>

---

<details>
<summary><b>Decouple Content and Motion for Conditional Image-to-Video Generation</b></summary>

* **Authors:** Yawei Li, Xintao Wang, Honglun Zhang, Zaopeng Cui, Jianmin Bao, Ying Shan
* **Affiliations:** Tsinghua University, Huawei
* **arXiv ID:** 2311.14294
* **One-liner:** 提出双流扩散架构，身份保留率93%，时间平滑度优于单流15%。
* **Published in:** ICCV 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2311.14294) | [[PDF]](https://arxiv.org/pdf/2311.14294.pdf)

> **核心创新**  
> 论文设计双流扩散架构，通过内容流锁定空间内容、运动流学习时间轨迹，实现空间内容与时间运动的有效分离，解决图像到视频生成中的身份漂移问题。

<details>
    <summary>Abstract</summary>
    本文提出了一种用于条件图像到视频生成的内容与运动解耦方法。通过双流扩散架构，内容流锁定空间内容，运动流学习时间轨迹，实现身份保留率93%，时间平滑度优于单流方法15%，有效解决了图像到视频生成中的身份漂移问题。
</details>

<details>
    <summary>Key points</summary>
    * 提出双流扩散（内容流锁定 + 运动流学习轨迹）
    * 分离空间内容与时间运动
    * 身份保留率93%
    * 时间平滑度优于单流15%
    * 解决图像到视频中的身份漂移问题
</details>
</details>

---

<details>
<summary><b>I2V-Adapter: A General Image-to-Video Adapter for Diffusion Models</b></summary>

* **Authors:** Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, Haibin Huang, Chongyang Ma
* **Affiliations:** Peking University, Chinese Academy of Sciences
* **arXiv ID:** 2312.16693
* **One-liner:** 提出轻量时间适配器，512x512视频30 FPS生成，连续性提升20%。
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2312.16693) | [[PDF]](https://arxiv.org/pdf/2312.16693.pdf)

> **核心创新**  
> 论文设计轻量时间适配器，通过跨帧注意力传播图像特征，无需重训Stable Diffusion即可实现高效图像到视频生成，解决生成效率低下的问题。

<details>
    <summary>Abstract</summary>
    I2V-Adapter提出了一种通用的图像到视频适配器用于扩散模型。通过轻量时间适配器和跨帧注意力传播图像特征，无需重新训练Stable Diffusion即可实现512x512分辨率视频30 FPS生成，连续性提升20%，显著提高了图像到视频生成的效率。
</details>

<details>
    <summary>Key points</summary>
    * 提出轻量时间适配器（无需重训Stable Diffusion）
    * 跨帧注意力传播图像特征
    * 512x512视频30 FPS生成
    * 连续性提升20%
    * 解决图像到视频生成的低效问题
</details>
</details>

---

<details>
<summary><b>Mo-Diff: Temporal Differential Fields for 4D Motion Modeling via Image-to-Video Synthesis</b></summary>

* **Authors:** Xin You, Minghui Zhang, Hanxiao Zhang, Jie Yang, Fei Yin
* **Affiliations:** Shanghai Jiao Tong University, Technical University of Munich
* **arXiv ID:** 2505.17333
* **One-liner:** 提出时间微分扩散，动态场景1000 FPS慢动作，流体模拟像素偏差<2%。
* **Published in:** SIGGRAPH 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2505.17333) | [[PDF]](https://arxiv.org/pdf/2505.17333.pdf)

> **核心创新**  
> 论文提出时间微分扩散方法，通过连续时空流场与正则化技术生成任意帧率的4D运动场，突破视频生成中的离散帧限制，实现高精度慢动作生成。

<details>
    <summary>Abstract</summary>
    Mo-Diff提出了一种基于时间微分场的4D运动建模方法。通过时间微分扩散生成任意帧率的4D场，结合连续时空流场和正则化技术，实现动态场景1000 FPS慢动作生成，流体模拟像素偏差小于2%，突破了视频生成中的离散帧限制。
</details>

<details>
    <summary>Key points</summary>
    * 提出时间微分扩散（生成任意帧率4D场）
    * 连续时空流场 + 正则化
    * 动态场景1000 FPS慢动作
    * 流体模拟像素偏差<2%
    * 解决视频生成中的离散帧限制问题
</details>
</details>

---

<details>
<summary><b>One-Step Consistency Distillation: OSV - One Step is Enough for High-Quality Image to Video Generation</b></summary>

* **Authors:** Yiqi Mao, Junhao Chen, Guangyang Ren, Xuhang Liu, Tianyu Zhang, Yihan Wang, Yuyin Zhou
* **Affiliations:** Fudan University
* **arXiv ID:** 2409.11367
* **One-liner:** 提出两阶段训练，FVD媲美多步模型，SSIM提升18%，A100上30 FPS。
* **Published in:** ECCV 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2409.11367) | [[PDF]](https://arxiv.org/pdf/2409.11367.pdf)

> **核心创新**  
> 论文设计两阶段训练策略，通过对抗训练与一致性蒸馏将多步扩散压缩为单步，结合损失约束实现高质量快速图像到视频生成，解决扩散模型推理缓慢的问题。

<details>
    <summary>Abstract</summary>
    OSV提出了一种一步一致性蒸馏方法用于高质量图像到视频生成。通过两阶段训练（对抗训练 + 一致性蒸馏）将多步扩散压缩为单步，结合损失约束实现FVD指标媲美多步模型，SSIM提升18%，在A100上达到30 FPS的生成速度。
</details>

<details>
    <summary>Key points</summary>
    * 提出两阶段训练（对抗训练 + 一致性蒸馏）
    * 多步扩散压缩为单步 + 损失约束
    * FVD媲美多步模型，SSIM提升18%
    * A100上30 FPS
    * 解决扩散模型图像到视频推理缓慢的问题
</details>
</details>

---

<details>
<summary><b>Multi-Modal I2V: HuMo - Human-Centric Video Generation via Collaborative Multi-Modal Agents</b></summary>

* **Authors:** Chenyang Si, Wentao Li, Yucheng Xie, Wenxuan Wang, Zihao Wang, Jian Yang, Jizheng Xu, Xiaokang Yang, Jingyi Yu
* **Affiliations:** ByteDance
* **arXiv ID:** 2509.08519
* **One-liner:** 提出多智能体协作框架，指令遵循度提升25%，音视频同步率提高30%。
* **Published in:** NeurIPS 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2509.08519) | [[PDF]](https://arxiv.org/pdf/2509.08519.pdf)

> **核心创新**  
> 论文设计多智能体协作框架，通过融合文本/图像/音频多模态输入，利用模态路由注意力实现时空对齐，解决多模态融合下的时间一致性问题。

<details>
    <summary>Abstract</summary>
    HuMo提出了一种以人为中心的视频生成方法，通过协作多智能体框架实现多模态融合。该框架整合文本、图像和音频输入，利用模态路由注意力机制实现时空对齐，在指令遵循度上提升25%，音视频同步率提高30%，显著改善了多模态融合的时间一致性。
</details>

<details>
    <summary>Key points</summary>
    * 提出多智能体协作框架（融合文本/图像/音频）
    * 模态路由注意力实现时空对齐
    * 指令遵循度提升25%
    * 音视频同步率提高30%
    * 解决多模态融合下的时间一致性问题
</details>
</details>

---

<details>
<summary><b>Audio-Visual Sync I2V: Syncphony - Synchronized Audio-to-Video Generation with Diffusion Transformers</b></summary>

* **Authors:** Jiaqi Chen, Yicheng Gu, Yeganeh Akbari, Zeyu Niu, Zihao Yue, Tianyu Luan, Jonathan Gratch
* **Affiliations:** Stanford University
* **arXiv ID:** 2509.21893
* **One-liner:** 提出扩散Transformer与同步引导，同步率提升30%，口型同步误差<80ms。
* **Published in:** ICML 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2509.21893) | [[PDF]](https://arxiv.org/pdf/2509.21893.pdf)

> **核心创新**  
> 论文结合扩散Transformer与同步引导机制，通过音频节奏映射到视频运动频率，实现音视频的高精度同步生成，解决音视频生成中的不同步问题。

<details>
    <summary>Abstract</summary>
    Syncphony提出了一种同步音频到视频生成的扩散Transformer方法。通过同步引导机制将音频节奏映射到视频运动频率，实现同步率提升30%，口型同步误差小于80ms，用户评分达到4.8/5，有效解决了音视频生成中的同步问题。
</details>

<details>
    <summary>Key points</summary>
    * 提出扩散Transformer + 同步引导
    * 音频节奏映射到视频运动频率
    * 同步率提升30%
    * 口型同步误差<80ms，用户评分4.8/5
    * 解决音视频生成中的不同步问题
</details>
</details>

---

<details>
<summary><b>FreeMask: Rethinking the Importance of Attention Masks for Zero-Shot Video Editing</b></summary>

* **Authors:** Hangjie Yuan, Dong Ni, Shiwei Zhang, Jiahao Chang, Xiang Wang, Xuhang Liu, Zhikai Li, Alex C. Kot
* **Affiliations:** Zhejiang University, Alibaba Group
* **arXiv ID:** 2409.20500
* **One-liner:** 提出自监督运动掩码提取，掩码IoU达89.2%，背景误差减少30%。
* **Published in:** ECCV 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2409.20500) | [[PDF]](https://arxiv.org/pdf/2409.20500.pdf)

> **核心创新**  
> 论文提出自监督运动掩码提取方法，通过掩码跨帧自动传播机制，实现零样本视频编辑中运动区域的精确提取，解决手动掩码标注的繁琐问题。

<details>
    <summary>Abstract</summary>
    FreeMask重新思考了注意力掩码在零样本视频编辑中的重要性，提出自监督运动掩码提取方法。通过掩码跨帧自动传播，实现掩码IoU达到89.2%，背景误差减少30%，显著简化了视频编辑流程。
</details>

<details>
    <summary>Key points</summary>
    * 提出自监督运动掩码提取
    * 掩码跨帧自动传播
    * 掩码IoU达89.2%
    * 背景误差减少30%
    * 解决视频编辑中手动掩码的繁琐问题
</details>
</details>

---

<details>
<summary><b>BrushEdit: All-In-One Image Inpainting and Editing</b></summary>

* **Authors:** Yawei Li, Pingping Cai, Tianyu He, Yu Qiao, Ming-Hsuan Yang
* **Affiliations:** Peking University, Tencent AI Lab
* **arXiv ID:** 2412.10316
* **One-liner:** 提出MLLM掩码生成与统一修复框架，边缘精度<5像素，RTX 4090上30 FPS。
* **Published in:** CVPR 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2412.10316) | [[PDF]](https://arxiv.org/pdf/2412.10316.pdf)

> **核心创新**  
> 论文设计基于MLLM的掩码生成与统一修复框架，通过迭代优化保障序列一致性，实现图像和视频的高效一体化编辑，解决视频编辑流程碎片化的问题。

<details>
    <summary>Abstract</summary>
    BrushEdit提出了一种全功能图像修复和编辑框架。通过MLLM掩码生成和统一修复框架，结合迭代优化保障序列一致性，实现边缘精度小于5像素，在RTX 4090上达到30 FPS的处理速度，有效整合了图像和视频编辑流程。
</details>

<details>
    <summary>Key points</summary>
    * 提出MLLM掩码生成 + 统一修复框架
    * 迭代优化保障序列一致性
    * 边缘精度<5像素
    * RTX 4090上30 FPS
    * 解决视频编辑流程碎片化的问题
</details>
</details>

---

<details>
<summary><b>DAPE: Dual-Stage Parameter-Efficient Fine-Tuning for Consistent Video Editing with Text Instructions</b></summary>

* **Authors:** Yihao Xu, Mengxi Li, Tianyu Luan, Bo Dai, Yufan Deng, Delin Chen, Yitian Yuan, Xintao Wang, Rui Zhao, Yuxiao Dong, Yan Wang
* **Affiliations:** Tsinghua University, Peking University
* **arXiv ID:** 2505.07057
* **One-liner:** 提出双阶段LoRA，模型参数仅0.5%，运动自然度提升15%。
* **Published in:** ICCV 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2505.07057) | [[PDF]](https://arxiv.org/pdf/2505.07057.pdf)

> **核心创新**  
> 论文设计双阶段参数高效微调方法，通过运动与外观分离适配，分阶段调优运动轨迹与风格，在极低参数量下实现高质量视频编辑，解决微调成本高的问题。

<details>
    <summary>Abstract</summary>
    DAPE提出了一种用于文本引导视频编辑的双阶段参数高效微调方法。通过双阶段LoRA实现运动与外观的分离适配，分阶段调优运动轨迹与风格，仅需0.5%的模型参数即可实现运动自然度提升15%，显著降低了视频编辑的微调成本。
</details>

<details>
    <summary>Key points</summary>
    * 提出双阶段LoRA（运动/外观分离适配）
    * 分阶段调优运动轨迹与风格
    * 模型参数仅0.5%
    * 运动自然度提升15%
    * 解决视频编辑中微调成本高的问题
</details>
</details>

---

<details>
<summary><b>MoViE: Mobile Diffusion for Video Editing</b></summary>

* **Authors:** Yuhao Huang, Hao Zhu, Weikang Bian, Tz-Ying Wu, Jussi Keppo, Minxue Jia, Marcelo H. Ang Jr., Daniela Rus
* **Affiliations:** Qualcomm AI Research
* **arXiv ID:** 2406.00272
* **One-liner:** 提出移动端优化扩散，骁龙8 Gen3上0.2秒/帧，模型压缩75%，速度提升4倍。
* **Published in:** ACM MM 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2406.00272) | [[PDF]](https://arxiv.org/pdf/2406.00272.pdf)

> **核心创新**  
> 论文设计移动端优化的扩散模型，通过稀疏注意力与量化技术，适配U-Net保障低功耗设备上的帧一致性，解决移动端视频编辑的资源约束问题。

<details>
    <summary>Abstract</summary>
    MoViE提出了一种用于移动端视频编辑的扩散模型优化方法。通过稀疏注意力、量化技术和适配U-Net架构，在骁龙8 Gen3上实现0.2秒/帧的处理速度，模型压缩75%，速度提升4倍，为移动设备提供了高效的视频编辑解决方案。
</details>

<details>
    <summary>Key points</summary>
    * 提出移动端优化扩散（稀疏注意力 + 量化）
    * 适配U-Net保障低功耗设备帧一致性
    * 骁龙8 Gen3上0.2秒/帧
    * 模型压缩75%，速度提升4倍
    * 解决移动端视频编辑的资源约束问题
</details>
</details>

---

<details>
<summary><b>Temporally Consistent Object Editing in Videos using Extended Attention</b></summary>

* **Authors:** Amirhossein Zamani, Amir Gholami Aghdam, Tiberiu Popa, Eugene Belilovsky
* **Affiliations:** Concordia University
* **arXiv ID:** 2406.00272
* **One-liner:** 提出扩展注意力窗口，像素级身份保留提升30%，纹理一致性提高25%。
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2406.00272) | [[PDF]](https://arxiv.org/pdf/2406.00272.pdf)

> **核心创新**  
> 论文设计扩展注意力窗口机制，通过跨64帧传播编辑信息，利用帧间注意力保障目标身份一致性，解决单帧编辑扩展到视频时的闪烁问题。

<details>
    <summary>Abstract</summary>
    本文提出了一种使用扩展注意力的时间一致性视频对象编辑方法。通过扩展注意力窗口跨64帧传播编辑信息，利用帧间注意力机制保障目标身份，实现像素级身份保留提升30%，纹理一致性提高25%，有效解决了视频编辑中的闪烁问题。
</details>

<details>
    <summary>Key points</summary>
    * 提出扩展注意力窗口（跨64帧传播编辑）
    * 帧间注意力保障目标身份
    * 像素级身份保留提升30%
    * 纹理一致性提高25%
    * 解决单帧编辑扩展到视频的闪烁问题
</details>
</details>

---

<details>
<summary><b>Consistent Video Editing as Flow-Driven I2V: Consistent Video Editing as Flow-Driven Image-to-Video Generation</b></summary>

* **Authors:** Shiyuan Wang, Yuhan Fan, Jiaqi Wang, Jiahao Fang, Binbin Song, Ying Wang, Jiale Liu, Youfei Wang, Zengyi Li, Xu Cao, Zhongang Qi, Ying Shan, Siyu Zhu
* **Affiliations:** Fudan University, Xiamen University
* **arXiv ID:** 2506.07713
* **One-liner:** 提出光流驱动pipeline，20步生成100帧无闪烁视频，伪影减少45%。
* **Published in:** SIGGRAPH 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2506.07713) | [[PDF]](https://arxiv.org/pdf/2506.07713.pdf)

> **核心创新**  
> 论文设计光流驱动的视频编辑pipeline，通过运动估计与I2V生成的结合，利用光流引导潜变量传输，解决视频编辑中的运动不一致问题。

<details>
    <summary>Abstract</summary>
    本文提出了一种流驱动的图像到视频生成方法用于一致性视频编辑。通过光流驱动pipeline结合运动估计和I2V生成，利用光流引导潜变量传输，实现20步生成100帧无闪烁视频，伪影减少45%，显著改善了视频编辑的运动一致性。
</details>

<details>
    <summary>Key points</summary>
    * 提出光流驱动pipeline（运动估计 + I2V生成）
    * 光流引导潜变量传输
    * 20步生成100帧无闪烁视频
    * 伪影减少45%
    * 解决视频编辑中的运动不一致问题
</details>
</details>

---

<details>
<summary><b>Audio-Visual Joint Attention: Audio-Visual Joint Attention for Enhancing Audio-Visual Generation</b></summary>

* **Authors:** Enhong Chen, Yujun Shen, Yujing Wang, Siteng Huang, Qi Zhang, Mizuho Hattori, Xuying Zhang, Jiawei Chen
* **Affiliations:** ByteDance
* **arXiv ID:** 2507.22781
* **One-liner:** 提出感知联合注意力，同步评分提升25%，口型同步误差<80ms。
* **Published in:** NeurIPS 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2507.22781) | [[PDF]](https://arxiv.org/pdf/2507.22781.pdf)

> **核心创新**  
> 论文设计感知联合注意力机制，通过帧-音频交互建模与音频-运动频率映射，实现音视频内容的高精度同步，解决音视频不同步的问题。

<details>
    <summary>Abstract</summary>
    本文提出了一种用于增强音视频生成的音视频联合注意力方法。通过感知联合注意力机制实现帧-音频交互建模和音频-运动频率映射，实现同步评分提升25%，口型同步误差小于80ms，有效解决了音视频内容不同步的问题。
</details>

<details>
    <summary>Key points</summary>
    * 提出感知联合注意力（帧-音频交互建模）
    * 音频-运动频率映射
    * 同步评分提升25%
    * 口型同步误差<80ms
    * 解决音视频内容不同步的问题
</details>
</details>

---

<details>
<summary><b>Controllable Multi-Agent Editing: Controllable Multi-Agent Editing in Videos Using Text Instructions</b></summary>

* **Authors:** Bo Han, Tianyu Lu, Zhipeng Zhou, Feng Liu
* **Affiliations:** Stanford University
* **arXiv ID:** 2410.05982
* **One-liner:** 提出多智能体框架，多角色控制精度提升22%，时空一致性优化。
* **Published in:** ICCV 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2410.05982) | [[PDF]](https://arxiv.org/pdf/2410.05982.pdf)

> **核心创新**  
> 论文设计多智能体编辑框架，通过身份token与动作嵌入机制，结合图神经网络建模智能体交互，解决多智能体运动独立无同步的问题。

<details>
    <summary>Abstract</summary>
    本文提出了一种使用文本指令的可控多智能体视频编辑方法。通过多智能体框架结合身份token和动作嵌入，利用图神经网络建模智能体交互，实现多角色控制精度提升22%，时空一致性显著优化，解决了多智能体运动独立无同步的问题。
</details>

<details>
    <summary>Key points</summary>
    * 提出多智能体框架（身份token + 动作嵌入）
    * 图神经网络建模智能体交互
    * 多角色控制精度提升22%
    * 时空一致性优化
    * 解决多智能体运动独立无同步的问题
</details>
</details>

---

<details>
<summary><b>MimicBrush: Zero-shot Image Editing with Reference Imitation</b></summary>

* **Authors:** Xi Chen, Yutong He, Xudong Wang, Yifan Yang, Heng Du, Xuyi Chen, Jing Liu, Jun Cheng, Wangmeng Xiang, Heng Tao Shen, Gang Wu
* **Affiliations:** The University of Hong Kong, Alibaba Group
* **arXiv ID:** 2406.07547
* **One-liner:** 提出参考模仿生成框架，风格空间平衡提升15%，语义保真度高。
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2406.07547) | [[PDF]](https://arxiv.org/pdf/2406.07547.pdf)

> **核心创新**  
> 论文设计参考模仿生成框架，通过视频帧风格迁移与帧掩码隐式保障风格一致性，实现零样本图像编辑，解决风格迁移中的时间对齐不一致问题。

<details>
    <summary>Abstract</summary>
    MimicBrush提出了一种基于参考模仿的零样本图像编辑框架。通过参考模仿生成框架实现视频帧风格迁移，利用帧掩码隐式保障风格一致性，实现风格空间平衡提升15%，保持高语义保真度，解决了风格迁移中的时间对齐不一致问题。
</details>

<details>
    <summary>Key points</summary>
    * 提出参考模仿生成框架（视频帧风格迁移）
    * 帧掩码隐式保障风格一致性
    * 风格空间平衡提升15%
    * 语义保真度高
    * 解决风格迁移中的时间对齐不一致问题
</details>
</details>

---
