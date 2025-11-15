<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### 模态一致性

**目标**：在符号（语言）和感知（视觉）表示之间建立双向映射。

**历史意义**：这些模型创建了首批"符号-感知桥梁"，解决了世界模型的基本输入/输出问题。

#### 代表性工作

<details>
<summary><b>CLIP: Learning Transferable Visual Models From Natural Language Supervision</b></summary>

* **Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
* **arXiv ID:** 2103.00020
* **One-liner:** Large-scale contrastive learning on 400M image–text pairs enabling strong zero-shot transfer across vision tasks
* **Published in:** ICML 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2103.00020) | [[PDF]](https://arxiv.org/pdf/2103.00020.pdf) | [[Code]](https://github.com/openai/CLIP)

> **核心创新**  
> 基于4亿图文对进行对比学习，建立统一的视觉-语言表征空间，实现零样本跨任务迁移能力，首创“文本作为监督信号”的大规模多模态训练范式。

<details>
    <summary>Abstract</summary>
    当前最先进的计算机视觉系统仍被训练来预测一组“固定且预先设定”的目标类别。这种受限的监督形式限制了它们的通用性和可用性，因为若想指定任何其他视觉概念，就必须再收集相应的标注数据。直接从关于图像的原始文本进行学习是一种颇具前景的替代方案，可利用远为广泛的监督来源。我们证明，仅需完成一项简单的预训练任务——预测哪段文字描述与哪张图像匹配——即可从头学出最先进的图像表征。我们在从互联网收集的4亿对（图像，文本）数据上完成该预训练。预训练后，借助自然语言即可引用已学到的视觉概念（或描述全新概念），使模型能够以零样本方式迁移到下游任务。我们在涵盖30余个现有计算机视觉数据集上对该方法进行评测，任务包括OCR、视频动作识别、地理定位以及各种细粒度物体分类。该模型在大多数任务上均表现出非平凡的迁移能力，且常常可与完全监督的基线相媲美，而无需任何针对特定数据集的训练。例如，我们在ImageNet上零样本达到了原始ResNet-50的精度，却无需使用后者赖以训练的128万张图片。
</details>

<details>
    <summary>Key points</summary>
    * 图像与文本特征的对比学习
    * 大规模弱监督网络数据（4亿对）  
    * 通过文本提示进行零样本分类  
    * 引发多模态基础模型浪潮  
</details>
</details>

---

<details>
<summary><b>DALL-E: Zero-Shot Text-to-Image Generation</b></summary>

* **Authors:** Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever
* **arXiv ID:** 2102.12092
* **One-liner:** First large transformer trained to generate images from text, demonstrating compositional text-to-image creativity
* **Published in:** ICML 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2102.12092) | [[PDF]](https://arxiv.org/pdf/2102.12092.pdf) | [[Project Page]](https://openai.com/research/dall-e)

> **核心创新**  
> 首次将大规模Transformer应用于文本到图像生成，搭配离散VAE编码，展示了语义组合生成（形状+风格+属性）的强能力。

<details>
    <summary>Abstract</summary>
    传统上，文本到图像生成研究一直致力于“在固定数据集上”寻找更优的建模假设。这些假设可能涉及复杂的网络架构、辅助损失，或在训练阶段引入对象部件标签、分割掩码等旁路信息。我们提出了一种极简方案：用单个Transformer以自回归方式把文本token与图像token当作同一数据流统一建模。只要数据量与模型规模足够，该方法在零样本评测下便可与此前专为特定领域设计的模型相抗衡。
</details>

<details>
    <summary>Key points</summary>
    * 文本条件 Transformer 用于图像生成  
    * 使用 VQ-VAE 进行图像离散化编码  
    * 具备组合推理能力（内容+风格+属性）  
    * 预示 Prompt 驱动生成新时代  
</details>
</details>

---

<details>
<summary><b>Show, Attend and Tell: Neural Image Caption Generation with Visual Attention</b></summary>

* **Authors:** Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio
* **arXiv ID:** 1502.03044
* **One-liner:** First visual attention mechanism for image captioning; pioneered attention in vision models
* **Published in:** ICML 2015
* **Links:** [[Paper]](https://arxiv.org/abs/1502.03044) | [[PDF]](https://arxiv.org/pdf/1502.03044.pdf)

> **核心创新**  
> 首次在图像描述任务中引入视觉注意力机制，使模型能聚焦关键区域并生成解释性注意图，为后续视觉Transformers奠基。

<details>
    <summary>Abstract</summary>
    受近期机器翻译与目标检测研究的启发，我们提出了一种基于“注意力”的模型，可自动学习并描述图像内容。我们阐述了如何以确定性方式（标准反向传播）和随机方式（最大化变分下界）对该模型进行训练。通过可视化，我们还展示了模型在生成输出序列中的对应词语时，能自动学会将“注视”锁定在显著物体上。在 Flickr8k、Flickr30k 和 MS COCO 三大基准数据集上，我们验证了注意力机制的有效性，并取得了当时最先进的性能。
</details>

<details>
    <summary>Key points</summary>
    * 视觉领域的早期注意力机制  
    * CNN 编码 + RNN 解码架构  
    * 注意力热力图带来可解释性  
    * 为视觉 Transformer 技术铺路  
</details>
</details>

---

<details>
<summary><b>AttnGAN: Fine-Grained Text-to-Image Generation with Attentional GANs</b></summary>

* **Authors:** Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He
* **arXiv ID:** 1711.10485
* **One-liner:** Word-region attention and semantic alignment loss for high-fidelity text-to-image synthesis
* **Published in:** CVPR 2018
* **Links:** [[Paper]](https://arxiv.org/abs/1711.10485) | [[PDF]](https://arxiv.org/pdf/1711.10485.pdf) | [[Code]](https://github.com/taoxugit/AttnGAN)

> **核心创新**  
> 通过单词级注意力和语义对齐损失实现细粒度文图对应，大幅提升文本生成图像的清晰度与语义一致性。

<details>
    <summary>Abstract</summary>
    本文提出注意力生成对抗网络（AttnGAN），实现“注意力驱动、多阶段细化”的细粒度文本到图像生成。借助新颖的注意力生成器，AttnGAN 能够关注自然语言描述中的相关词汇，在图像的不同子区域逐步合成精细细节。此外，我们设计了深度注意力多模态相似度模型，用于计算细粒度的图像-文本匹配损失，以训练生成器。AttnGAN 显著超越此前最佳方法：在 CUB 数据集上将 Inception Score 提高 14.14%，在更具挑战的 COCO 数据集上提升 170.25%。通过对 AttnGAN 各注意力层的可视化分析，我们首次证实，分层注意力 GAN 能够自动在“词”级别选择条件，从而生成图像的不同部分。
</details>

<details>
    <summary>Key points</summary>
    * 基于单词的细粒度文本注意力  
    * 多阶段逐步细化图像生成  
    * DAMSM 损失强化文图语义一致性  
    * 扩展GAN在文本生成图像方向的上限  
</details>
</details>

---
