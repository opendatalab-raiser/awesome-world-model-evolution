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
