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

<details>
<summary><b> Zero-Shot Text-to-Image Generation</b></summary>

* **Authors:** Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever
* **arXiv ID:** 2102.12092
* **One-liner:** Developed a simple transformer-based autoregressive model for text-to-image generation that is competitive with domain-specific models in zero-shot evaluation.
* **Published in:** arxiv (24 Feb 2021)
* **Links:** [[Paper]](https://arxiv.org/abs/2102.12092) | [[PDF]](https://arxiv.org/pdf/2102.12092) | [[Code]](https://github.com/openai/dall-e)

> **核心创新**
> 通过将文本和图像标记建模为单一数据流并使用变换器架构，实现了竞争性的零样本文本到图像生成。

<details>
    <summary>Abstract</summary>
    文本到图像生成传统上侧重于为在固定数据集上的训练寻找更好的建模假设。这些假设可能涉及复杂架构、辅助损失或训练期间提供的侧信息，如对象部分标签或分割掩码。我们描述了一种基于变换器的简单方法，该方法自回归地将文本和图像标记建模为单一数据流。在足够的数据和规模下，我们的方法在零样本评估中与先前领域特定模型竞争。
</details>

<details>
    <summary>Key points</summary>
    * 将文本和图像标记自回归建模为单一流
    * 使用变换器架构进行序列预测
    * 通过足够的数据和规模训练实现零样本能力
</details>
</details>

---


<details>
<summary><b> High-Resolution Image Synthesis with Latent Diffusion Models</b></summary>

* **Authors:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer
* **arXiv ID:** 2112.10752
* **One-liner:** Introduced latent diffusion models (LDMs) that reduce computational costs while maintaining high image quality and flexibility for tasks like text-to-image generation.
* **Published in:** arxiv (20 Dec 2021)
* **Links:** [[Paper]](https://arxiv.org/abs/2112.10752) | [[PDF]](https://arxiv.org/pdf/2112.10752) | [[Code]](https://github.com/CompVis/stable-diffusion)

> **核心创新**
> 在预训练自编码器的潜在空间中应用扩散模型以平衡复杂度降低和细节保留，实现了高效的高保真图像合成。

<details>
    <summary>Abstract</summary>
    通过将图像形成过程分解为去噪自编码器的顺序应用，扩散模型在图像数据及其他领域实现了最先进的合成结果。此外，它们的公式允许一种引导机制来控制图像生成过程而无需重新训练。然而，由于这些模型通常直接在像素空间中操作，强大扩散模型的优化往往消耗数百GPU天，并且由于顺序评估，推理成本高昂。为了在有限计算资源上实现扩散模型训练，同时保持其质量和灵活性，我们在强大预训练自编码器的潜在空间中应用它们。与先前工作相比，在这种表示上训练扩散模型首次实现了复杂度降低和细节保留之间的近最优平衡，大大提升了视觉保真度。通过在模型架构中引入交叉注意力层，我们将扩散模型转变为强大且灵活的生成器，用于一般条件输入，如文本或边界框，并且高分辨率合成以卷积方式成为可能。我们的潜在扩散模型在图像修复中实现了新的最先进水平，并在各种任务中表现出高度竞争性性能，包括无条件图像生成、语义场景合成和超分辨率，同时与基于像素的扩散模型相比显著降低了计算需求。代码可在<a href="https://github.com/CompVis/latent-diffusion" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 使用预训练自编码器在潜在空间中训练扩散模型
    * 引入交叉注意力层以条件化输入如文本
    * 在图像修复中实现最先进水平，并在各种任务中表现竞争性性能
</details>
</details>

---


<details>
<summary><b> Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding</b></summary>

* **Authors:** Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, Mohammad Norouzi
* **arXiv ID:** 2205.11487
* **One-liner:** Created Imagen, a text-to-image diffusion model with exceptional photorealism and language understanding, leveraging large language models for text encoding.
* **Published in:** arxiv (23 May 2022)
* **Links:** [[Paper]](https://arxiv.org/abs/2205.11487) | [[PDF]](https://arxiv.org/pdf/2205.11487) | [[Code]](https://github.com/GACWR/Deep-Floyd-IF?tab=readme-ov-file)

> **核心创新**
> 展示了在Imagen中扩展语言模型比扩展图像扩散模型更能改善样本保真度和图像-文本对齐，实现了最先进的FID分数。

<details>
    <summary>Abstract</summary>
    我们提出了Imagen，一种文本到图像的扩散模型，具有前所未有的照片真实感和深层次的语言理解能力。Imagen基于大型变换器语言模型在理解文本方面的能力，并依赖扩散模型在高保真图像生成方面的优势。我们的关键发现是，通用大型语言模型（如T5）在仅文本语料库上预训练后，在编码文本以进行图像合成方面出奇地有效：在Imagen中增加语言模型的规模比增加图像扩散模型的规模更能提升样本保真度和图像-文本对齐度。Imagen在COCO数据集上实现了新的最先进FID分数7.27，而无需在COCO上训练，并且人类评估者发现Imagen样本在图像-文本对齐方面与COCO数据本身相当。为了更深入地评估文本到图像模型，我们引入了DrawBench，一个全面且具有挑战性的文本到图像模型基准。通过DrawBench，我们将Imagen与最近的方法（包括VQ-GAN+CLIP、潜在扩散模型和DALL-E 2）进行比较，并发现人类评估者在并排比较中更偏好Imagen，无论是在样本质量还是图像-文本对齐方面。结果概述请参见<a href="https://imagen.research.google/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 集成大型变换器语言模型（如T5）进行文本编码
    * 使用扩散模型进行高保真图像生成
    * 引入DrawBench进行全面评估
</details>
</details>

---


<details>
<summary><b> Scaling Autoregressive Models for Content-Rich Text-to-Image Generation</b></summary>

* **Authors:** Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, Yonghui Wu
* **arXiv ID:** 2206.10789
* **One-liner:** Developed Parti, a sequence-to-sequence model for text-to-image generation that scales to 20B parameters, achieving high-fidelity and content-rich image synthesis.
* **Published in:** arxiv (22 Jun 2022)
* **Links:** [[Paper]](https://arxiv.org/abs/2206.10789) | [[PDF]](https://arxiv.org/pdf/2206.10789) | [[Code]](https://github.com/google-research/parti)

> **核心创新**
> 将文本到图像生成视为序列到序列问题，使用基于变换器的图像标记器并扩展模型规模以改进性能。

<details>
    <summary>Abstract</summary>
    我们提出了Pathways自回归文本到图像模型，该模型生成高保真照片真实图像，并支持涉及复杂组合和世界知识的内容丰富合成。Parti将文本到图像生成视为序列到序列建模问题，类似于机器翻译，其中图像标记序列作为目标输出，而非另一种语言的文本标记。这一策略自然可以利用先前在大型语言模型上的丰富工作，这些模型通过扩展数据和模型规模在能力和性能上持续进步。我们的方法简单：首先，Parti使用基于变换器的图像标记器ViT-VQGAN将图像编码为离散标记序列。其次，我们通过将编码器-解码器变换器模型扩展到200亿参数，实现了持续的质量改进，在MS-COCO上获得了新的最先进零样本FID分数7.23和微调FID分数3.22。我们在Localized Narratives以及PartiPrompts上的详细分析，这是一个包含1600多个英语提示的新整体基准，展示了Parti在广泛类别和难度方面的有效性。我们还探索并强调了模型的局限性，以定义和示例化进一步改进的关键领域。高分辨率图像请参见<a href="https://parti.research.google/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 以图像标记为目标的序列到序列建模
    * 使用ViT-VQGAN进行图像标记化
    * 将编码器-解码器变换器扩展到200亿参数
</details>
</details>

---


<details>
<summary><b> Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment</b></summary>

* **Authors:** Hao Liu, Wilson Yan, Pieter Abbeel
* **arXiv ID:** 2302.00902
* **One-liner:** Proposed LQAE, an unsupervised method to align text and image modalities using pretrained language models, enabling few-shot image classification.
* **Published in:** arxiv (2 Feb 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2302.00902) | [[PDF]](https://arxiv.org/pdf/2302.00902) | [[Code]](https://github.com/haoliuhl/language-quantized-autoencoders)

> **核心创新**
> 通过使用预训练语言码本量化图像嵌入，学习将图像表示为文本标记序列，从而在没有对齐数据的情况下促进多模态任务。

<details>
    <summary>Abstract</summary>
    最近在扩展大型语言模型方面的进展显示了在广泛基于文本的任务中执行少样本学习的令人印象深刻的能力。然而，一个关键限制是这些语言模型从根本上缺乏视觉感知——这是扩展这些模型以能够与现实世界交互并解决视觉任务（如视觉问答和机器人技术）所需的关键属性。先前工作主要通过预训练和/或在策划的图像-文本数据集上微调来连接图像和文本，这可能是一个成本高昂的过程。为了解决这一限制，我们提出了一种简单而有效的方法，称为语言量化自编码器，这是VQ-VAE的一种修改，通过利用预训练语言模型（如BERT、RoBERTa）以无监督方式学习对齐文本-图像数据。我们的主要思想是通过直接使用预训练语言码本量化图像嵌入来将图像编码为文本标记序列。然后，我们应用随机掩码，随后使用BERT模型，并让解码器从BERT预测的文本标记嵌入中重建原始图像。通过这样做，LQAE学习用相似的文本标记簇表示相似图像，从而在没有对齐文本-图像对的情况下对齐这两种模态。这使得能够使用大型语言模型（如GPT-3）进行少样本图像分类，以及基于BERT文本特征的图像线性分类。据我们所知，我们的工作是第一个通过利用预训练语言模型的能力使用未对齐图像进行多模态任务的工作。
</details>

<details>
    <summary>Key points</summary>
    * 使用预训练语言码本量化图像嵌入
    * 应用随机掩码和BERT进行重建
    * 无监督对齐文本和图像模态
</details>
</details>

---


<details>
<summary><b> IconShop: Text-Guided Vector Icon Synthesis with Autoregressive Transformers</b></summary>

* **Authors:** Ronghuan Wu, Wanchao Su, Kede Ma, Jing Liao
* **arXiv ID:** 2304.14400
* **One-liner:** Introduced IconShop, a text-guided vector icon synthesis method using autoregressive transformers for high-quality and diverse SVG generation.
* **Published in:** arxiv (27 Apr 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2304.14400) | [[PDF]](https://arxiv.org/pdf/2304.14400) | [[Code]](https://github.com/kingnobro/IconShop)

> **核心创新**
> 将SVG路径和文本描述序列化和标记化为可解码序列，以利用自回归变换器实现无条件和文本条件图标合成。

<details>
    <summary>Abstract</summary>
    可缩放矢量图形是一种流行的矢量图像格式，提供了良好的交互性和动画支持。尽管具有吸引人的特性，但由于需要陡峭的学习曲线来理解SVG语法或熟悉专业编辑软件，创建自定义SVG内容对用户来说可能具有挑战性。最近文本到图像生成的进展激发了研究人员探索矢量图形合成，使用基于图像的方法（即文本 -> 光栅图像 -> 矢量图形）结合文本到图像生成模型和图像矢量化，或基于语言的方法（即文本 -> 矢量图形脚本）通过预训练大型语言模型。然而，这些方法在生成质量、多样性和灵活性方面仍然存在限制。在本文中，我们介绍了IconShop，一种使用自回归变换器的文本引导矢量图标合成方法。我们方法成功的关键是将SVG路径（和文本描述作为引导）序列化和标记化为唯一可解码的标记序列。这样，我们能够充分利用自回归变换器的序列学习能力，同时实现无条件和文本条件图标合成。通过在伴随文本描述的大规模矢量图标数据集上进行标准训练以预测下一个标记，所提出的IconShop在定量和定性上一致表现出比现有基于图像和基于语言方法更好的图标合成能力。同时，我们观察到生成多样性的显著改善，这通过客观的唯一性和新颖性度量得到验证。更重要的是，我们通过多个新颖图标合成任务展示了IconShop的灵活性，包括图标编辑、图标插值、图标语义组合和图标设计自动建议。
</details>

<details>
    <summary>Key points</summary>
    * 将SVG路径和文本标记化为可解码序列
    * 使用自回归变换器进行序列学习
    * 在任务如图标编辑和插值中展示灵活性
</details>
</details>

---


<details>
<summary><b> SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis</b></summary>

* **Authors:** Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach
* **arXiv ID:** 2307.01952
* **One-liner:** Presented SDXL, an enhanced latent diffusion model with a larger UNet backbone and novel conditioning schemes for improved text-to-image synthesis.
* **Published in:** arxiv (4 Jul 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2307.01952) | [[PDF]](https://arxiv.org/pdf/2307.01952) | [[Code]](https://github.com/Stability-AI/generative-models)

> **核心创新**
> 增加了模型参数并引入了多个条件方案和宽高比训练，以及一个精炼模型以改善视觉保真度。

<details>
    <summary>Abstract</summary>
    我们提出了SDXL，一种用于文本到图像合成的潜在扩散模型。与先前版本的Stable Diffusion相比，SDXL利用了三倍大的UNet骨干：模型参数增加主要是由于更多的注意力块和更大的交叉注意力上下文，因为SDXL使用了第二个文本编码器。我们设计了多个新颖的条件方案，并在多种宽高比上训练SDXL。我们还引入了一个精炼模型，用于通过后处理图像到图像技术改进SDXL生成样本的视觉保真度。我们证明SDXL与先前版本的Stable Diffusion相比显示出显著改进的性能，并实现了与黑盒最先进图像生成器竞争的结果。本着促进开放研究和在大型模型训练和评估中培养透明度的精神，我们在<a href="https://github.com/Stability-AI/generative-models" rel="external noopener nofollow" class="link-external link-https">此https URL</a>提供代码和模型权重访问。
</details>

<details>
    <summary>Key points</summary>
    * 更大的UNet骨干，具有更多注意力块和交叉注意力上下文
    * 使用第二个文本编码器和多个条件方案
    * 引入精炼模型进行后处理改进
</details>
</details>

---


<details>
<summary><b> Emu: Generative Pretraining in Multimodality</b></summary>

* **Authors:** Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, Xinlong Wang
* **arXiv ID:** 2307.05222
* **One-liner:** Developed Emu, a multimodal foundation model that generates images and texts from any single or multimodal input using a unified autoregressive training process.
* **Published in:** arxiv (11 Jul 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2307.05222) | [[PDF]](https://arxiv.org/pdf/2307.05222) | [[Code]](https://github.com/baaivision/Emu)

> **核心创新**
> 通过统一目标进行端到端训练，用于多模态序列中的下一个标记预测，实现了如图像描述和文本到图像生成等多样化任务。

<details>
    <summary>Abstract</summary>
    我们提出了Emu，一种基于变换器的多模态基础模型，可以在多模态上下文中无缝生成图像和文本。这种全能模型可以通过单一模型适用于所有自回归训练过程，无差别地接受任何单模态或多模态数据输入（例如，交织的图像、文本和视频）。首先，视觉信号被编码为嵌入，并与文本标记一起形成交织的输入序列。然后，Emu通过统一目标进行端到端训练，该目标是在多模态序列中分类下一个文本标记或回归下一个视觉嵌入。这种多功能多模态性使得能够大规模探索多样化的预训练数据源，例如带有交织帧和文本的视频、带有交织图像和文本的网页，以及网络规模的图像-文本对和视频-文本对。Emu可以作为通用多模态接口，用于图像到文本和文本到图像任务，并支持上下文图像和文本生成。在广泛的零样本/少样本任务中，包括图像描述、视觉问答、视频问答和文本到图像生成，Emu与最先进的大型多模态模型相比表现出卓越性能。通过指令调优扩展的能力，如多模态助手，也展示了令人印象深刻的性能。
</details>

<details>
    <summary>Key points</summary>
    * 多模态序列的统一自回归训练
    * 将视觉信号编码为嵌入并与文本标记结合
    * 支持上下文图像和文本生成
</details>
</details>

---


<details>
<summary><b> SDXL-Lightning: Progressive Adversarial Diffusion Distillation</b></summary>

* **Authors:** Shanchuan Lin, Anran Wang, Xiao Yang
* **arXiv ID:** 2402.13929
* **One-liner:** Proposed a diffusion distillation method for efficient one-step/few-step text-to-image generation based on SDXL, achieving state-of-the-art results.
* **Published in:** arxiv (21 Feb 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2402.13929) | [[PDF]](https://arxiv.org/pdf/2402.13929) | [[Code]](https://github.com/Stability-AI/generative-models)

> **核心创新**
> 结合渐进式和对抗性蒸馏以平衡质量和模式覆盖，实现了快速推理和高保真度。

<details>
    <summary>Abstract</summary>
    我们提出了一种扩散蒸馏方法，在基于SDXL的一步/少步1024px文本到图像生成中实现了新的最先进水平。我们的方法结合了渐进式和对抗性蒸馏，以在质量和模式覆盖之间实现平衡。在本文中，我们讨论了理论分析、判别器设计、模型公式和训练技术。我们开源了我们的蒸馏SDXL-Lightning模型，包括LoRA和完整UNet权重。
</details>

<details>
    <summary>Key points</summary>
    * 渐进式和对抗性蒸馏技术
    * 将蒸馏应用于SDXL以实现一步/少步生成
    * 开源蒸馏模型作为LoRA和完整UNet权重
</details>
</details>

---


<details>
<summary><b> Beyond Text: Frozen Large Language Models in Visual Signal Comprehension</b></summary>

* **Authors:** Lei Zhu, Fangyun Wei, Yanye Lu
* **arXiv ID:** 2403.07874
* **One-liner:** Introduced V2T Tokenizer to enable large language models to comprehend and process images as linguistic entities without fine-tuning on multimodal data.
* **Published in:** arxiv (12 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.07874) | [[PDF]](https://arxiv.org/pdf/2403.07874) | [[Code]](https://github.com/zh460045050/V2L-Tokenizer)

> **核心创新**
> 使用编码器-解码器和CLIP模型将图像转换为离散词，允许LLM以自回归方式执行视觉任务。

<details>
    <summary>Abstract</summary>
    在这项工作中，我们研究了大型语言模型直接理解视觉信号的潜力，而无需在多模态数据集上微调。我们方法的基本概念将图像视为语言实体，并将其翻译为从LLM词汇表中派生的一组离散词。为了实现这一点，我们提出了视觉到语言标记器，缩写为V2T标记器，它在编码器-解码器、LLM词汇表和CLIP模型的联合帮助下将图像转换为“外语”。通过这种创新的图像编码，LLM不仅获得了视觉理解能力，还能够以自回归方式进行图像去噪和恢复——关键的是，无需任何微调。我们进行了严格的实验来验证我们的方法，涵盖理解任务如图像识别、图像描述和视觉问答，以及图像去噪任务如修复、外绘、去模糊和移位恢复。代码和模型可在<a href="https://github.com/zh460045050/V2L-Tokenizer" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 将图像翻译为LLM词汇表中的离散词
    * 使用编码器-解码器和CLIP进行图像编码
    * 自回归处理用于视觉理解和去噪任务
</details>
</details>

---


<details>
<summary><b> Kandinsky 3.0 Technical Report</b></summary>

* **Authors:** Vladimir Arkhipkin, Andrei Filatov, Viacheslav Vasilev, Anastasia Maltseva, Said Azizov, Igor Pavlov, Julia Agafonova, Andrey Kuznetsov, Denis Dimitrov
* **arXiv ID:** 2312.03511
* **One-liner:** Introduced Kandinsky 3.0, a high-quality text-to-image model based on latent diffusion.
* **Published in:** arxiv (6 Dec 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2312.03511) | [[PDF]](https://arxiv.org/pdf/2312.03511) | [[Code]](https://github.com/ai-forever/Kandinsky-3)

> **核心创新**
> 通过架构改进和训练技术，实现了图像生成的更高真实感和质量。

<details>
    <summary>Abstract</summary>
    我们介绍了Kandinsky 3.0，一个基于潜在扩散的大规模文本到图像生成模型，延续了Kandinsky文本到图像模型系列，并反映了我们在实现更高图像生成质量和真实感方面的进展。
</details>

<details>
    <summary>Key points</summary>
    * 潜在扩散架构
    * 训练技术优化
    * 扩展功能如超分辨率和修复
    * 蒸馏版本以实现更快推理
</details>
</details>

---


<details>
<summary><b> PixArt-$α$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis</b></summary>

* **Authors:** Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, Zhenguo Li
* **arXiv ID:** 2310.00426
* **One-liner:** Developed PIXART-α, a low-cost Transformer-based text-to-image model with competitive quality.
* **Published in:** arxiv (30 Sep 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2310.00426) | [[PDF]](https://arxiv.org/pdf/2310.00426) | [[Code]](https://github.com/PixArt-alpha/PixArt-alpha)

> **核心创新**
> 在保持高分辨率图像合成的同时，降低了训练成本和二氧化碳排放。

<details>
    <summary>Abstract</summary>
    最先进的文本到图像（T2I）模型需要大量训练成本（例如，数百万GPU小时），严重阻碍了AIGC社区的根本创新，同时增加了二氧化碳排放。本文介绍了PIXART-$\alpha$，一个基于Transformer的T2I扩散模型，其图像生成质量与最先进的图像生成器（如Imagen、SDXL，甚至Midjourney）竞争，达到接近商业应用标准。此外，它支持高达1024像素分辨率的高分辨率图像合成，且训练成本低，如图1和图2所示。为实现这一目标，提出了三个核心设计：（1）训练策略分解：我们设计了三个不同的训练步骤，分别优化像素依赖、文本-图像对齐和图像美学质量；（2）高效的T2I Transformer：我们将交叉注意力模块整合到扩散Transformer（DiT）中，以注入文本条件并简化计算密集的类条件分支；（3）高信息数据：我们强调文本-图像对中概念密度的重要性，并利用大型视觉语言模型自动标注密集伪标题，以辅助文本-图像对齐学习。因此，PIXART-$\alpha$的训练速度显著超过现有大规模T2I模型，例如，PIXART-$\alpha$仅占Stable Diffusion v1.5训练时间的10.8%（675 vs. 6,250 A100 GPU天），节省近30万美元（26,000 vs. 320,000美元），并减少90%的二氧化碳排放。此外，与更大的SOTA模型RAPHAEL相比，我们的训练成本仅为1%。广泛实验表明，PIXART-$\alpha$在图像质量、艺术性和语义控制方面表现出色。我们希望PIXART-$\alpha$能为AIGC社区和初创公司提供新见解，加速从零开始构建高质量且低成本的生成模型。
</details>

<details>
    <summary>Key points</summary>
    * 训练策略分解为三个步骤
    * 高效的T2I Transformer与交叉注意力
    * 使用高信息数据与伪标题
</details>
</details>

---


<details>
<summary><b> EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models</b></summary>

* **Authors:** Jingyuan Yang, Jiawei Feng, Hui Huang
* **arXiv ID:** 2401.04608
* **One-liner:** Proposed Emotional Image Content Generation (EICG) for generating emotion-faithful images.
* **Published in:** arxiv (9 Jan 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2401.04608) | [[PDF]](https://arxiv.org/pdf/2401.04608) | [[Code]]()

> **核心创新**
> 通过CLIP对齐，实现了基于抽象情感的语义清晰图像生成。

<details>
    <summary>Abstract</summary>
    近年来，图像生成任务取得了显著进展，用户能够创建视觉上惊人的高质量图像。然而，现有的文本到图像扩散模型擅长生成具体概念（如狗），但在处理更抽象概念（如情感）时遇到挑战。已有一些努力通过颜色和风格调整来修改图像情感，但在有效传达固定图像内容的情感方面存在局限性。在这项工作中，我们引入了情感图像内容生成（EICG），一个新任务，旨在给定情感类别生成语义清晰且情感忠实的图像。具体来说，我们提出了一个情感空间，并构建了一个映射网络将其与强大的对比语言-图像预训练（CLIP）空间对齐，为抽象情感提供具体解释。进一步提出了属性损失和情感置信度，以确保生成图像的语义多样性和情感保真度。我们的方法在定量和定性上均优于最先进的文本到图像方法，其中我们推导了三个自定义指标，即情感准确性、语义清晰度和语义多样性。除了生成，我们的方法还有助于情感理解并激发情感艺术设计。
</details>

<details>
    <summary>Key points</summary>
    * 情感空间映射到CLIP空间
    * 属性损失用于语义多样性
    * 情感置信度用于保真度
</details>
</details>

---


<details>
<summary><b> Parrot: Pareto-optimal Multi-Reward Reinforcement Learning Framework for Text-to-Image Generation</b></summary>

* **Authors:** Seung Hyun Lee, Yinxiao Li, Junjie Ke, Innfarn Yoo, Han Zhang, Jiahui Yu, Qifei Wang, Fei Deng, Glenn Entis, Junfeng He, Gang Li, Sangpil Kim, Irfan Essa, Feng Yang
* **arXiv ID:** 2401.05675
* **One-liner:** Introduced Parrot for multi-objective optimization in text-to-image generation.
* **Published in:** arxiv (11 Jan 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2401.05675) | [[PDF]](https://arxiv.org/pdf/2401.05675) | [[Code]]()

> **核心创新**
> 自动化奖励权衡，无需手动权重调整即可提高图像质量。

<details>
    <summary>Abstract</summary>
    最近的研究表明，使用强化学习（RL）与多个质量奖励可以提高文本到图像（T2I）生成中生成图像的质量。然而，手动调整奖励权重具有挑战性，并可能导致某些指标的过度优化。为解决这一问题，我们提出了Parrot，通过多目标优化解决该问题，并引入了一种有效的多奖励优化策略来近似帕累托最优。利用批量帕累托最优选择，Parrot自动识别不同奖励之间的最优权衡。我们使用新颖的多奖励优化算法联合优化T2I模型和提示扩展网络，显著提高了图像质量，并允许在推理时使用与奖励相关的提示来控制不同奖励的权衡。此外，我们在推理时引入了原始提示中心指导，确保在提示扩展后对用户输入的忠实性。广泛实验和用户研究验证了Parrot在多个质量标准（包括美学、人类偏好、文本-图像对齐和图像情感）上优于多个基线。
</details>

<details>
    <summary>Key points</summary>
    * 多目标优化与帕累托最优选择
    * T2I模型和提示扩展的联合优化
    * 原始提示中心指导
</details>
</details>

---


<details>
<summary><b> DiffusionGPT: LLM-Driven Text-to-Image Generation System</b></summary>

* **Authors:** Jie Qin, Jie Wu, Weifeng Chen, Yuxi Ren, Huixia Li, Hefeng Wu, Xuefeng Xiao, Rui Wang, Shilei Wen
* **arXiv ID:** 2401.10061
* **One-liner:** Proposed DiffusionGPT, a unified system for diverse prompts and model integration using LLMs.
* **Published in:** arxiv (18 Jan 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2401.10061) | [[PDF]](https://arxiv.org/pdf/2401.10061) | [[Code]](https://github.com/DiffusionGPT/DiffusionGPT)

> **核心创新**
> 利用LLM根据输入提示选择适当的生成模型。

<details>
    <summary>Abstract</summary>
    扩散模型为图像生成领域开辟了新途径，导致高质量模型在开源平台上激增。然而，当前文本到图像系统的一个主要挑战是往往无法处理多样输入，或仅限于单一模型结果。当前统一尝试通常落入两个正交方面：i）在输入阶段解析多样提示；ii）激活专家模型以输出。为结合两者优势，我们提出了DiffusionGPT，它利用大型语言模型（LLM）提供一个统一生成系统，能够无缝容纳各种类型的提示并整合领域专家模型。DiffusionGPT基于先验知识为各种生成模型构建领域特定树。当提供输入时，LLM解析提示并使用思想树来指导选择适当模型，从而放宽输入约束并确保跨多样领域的卓越性能。此外，我们引入了优势数据库，其中思想树通过人类反馈丰富，使模型选择过程与人类偏好对齐。通过广泛实验和比较，我们展示了DiffusionGPT的有效性，展示了其在多样领域推动图像合成边界的潜力。
</details>

<details>
    <summary>Key points</summary>
    * 领域特定树用于模型选择
    * 使用大型语言模型进行解析
    * 带有人类反馈的优势数据库
</details>
</details>

---


<details>
<summary><b> MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer</b></summary>

* **Authors:** Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, Hongsheng Li, Yu Qiao, Jifeng Dai
* **arXiv ID:** 2401.10208
* **One-liner:** Presented MM-Interleaved, an end-to-end model for interleaved image-text data generation.
* **Published in:** arxiv (18 Jan 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2401.10208) | [[PDF]](https://arxiv.org/pdf/2401.10208) | [[Code]](https://github.com/OpenGVLab/MM-Interleaved)

> **核心创新**
> 通过细粒度特征访问，改进了多图像场景的处理。

<details>
    <summary>Abstract</summary>
    开发用于交错图像-文本数据的生成模型具有研究和实用价值。它要求模型理解交错序列并随后生成图像和文本。然而，现有尝试受限于固定数量的视觉标记无法有效捕捉图像细节的问题，这在多图像场景中尤为突出。为解决这一问题，本文提出了MM-Interleaved，一个用于交错图像-文本数据的端到端生成模型。它引入了多尺度和多图像特征同步器模块，允许在生成过程中直接访问先前上下文中的细粒度图像特征。MM-Interleaved在配对和交错图像-文本语料库上进行了端到端预训练。通过监督微调阶段进一步增强，模型提高了遵循复杂多模态指令的能力。实验证明了MM-Interleaved在识别多模态指令后的视觉细节和生成遵循文本和视觉条件的一致图像方面的多功能性。代码和模型可在\url{<a href="https://github.com/OpenGVLab/MM-Interleaved" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}获取。
</details>

<details>
    <summary>Key points</summary>
    * 多尺度和多图像特征同步器
    * 交错数据的端到端预训练
    * 多模态指令的监督微调
</details>
</details>

---


<details>
<summary><b> Taiyi-Diffusion-XL: Advancing Bilingual Text-to-Image Generation with Large Vision-Language Model Support</b></summary>

* **Authors:** Xiaojun Wu, Dixiang Zhang, Ruyi Gan, Junyu Lu, Ziwei Wu, Renliang Sun, Jiaxing Zhang, Pingjian Zhang, Yan Song
* **arXiv ID:** 2401.14688
* **One-liner:** Developed Taiyi-Diffusion-XL, a bilingual text-to-image model for Chinese and English.
* **Published in:** arxiv (26 Jan 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2401.14688) | [[PDF]](https://arxiv.org/pdf/2401.14688) | [[Code]](https://github.com/IDEA-CCNL/Taiyi-Diffusion-XL)

> **核心创新**
> 通过双语支持和增强提示扩展了CLIP和Stable-Diffusion-XL。

<details>
    <summary>Abstract</summary>
    文本到图像模型的最新进展显著增强了图像生成能力，但开源模型在双语或中文语言支持方面仍存在显著差距。为满足这一需求，我们提出了Taiyi-Diffusion-XL，一个新的中英文双语文本到图像模型，通过扩展CLIP和Stable-Diffusion-XL的能力，通过双语连续预训练过程开发。该方法包括通过将最常用的中文字符集成到CLIP的分词器和嵌入层中来高效扩展词汇表，并结合绝对位置编码扩展。此外，我们通过大型视觉语言模型丰富文本提示，导致更好的图像标题和更高的视觉质量。这些增强随后应用于下游文本到图像模型。我们的实证结果表明，开发的CLIP模型在双语图像-文本检索方面表现出色。此外，Taiyi-Diffusion-XL的双语图像生成能力超越了先前模型。这项研究导致了Taiyi-Diffusion-XL模型的开发和开源，代表了图像生成领域的显著进步，特别是针对中文语言应用。这一贡献是解决多模态研究中更多样语言需求的一步。模型和演示在\href{<a href="https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}公开提供，促进该领域的进一步研究和合作。
</details>

<details>
    <summary>Key points</summary>
    * 中文字符的词汇扩展
    * 双语连续预训练
    * 使用视觉语言模型丰富文本提示
</details>
</details>

---


<details>
<summary><b> IntentTuner: An Interactive Framework for Integrating Human Intents in Fine-tuning Text-to-Image Generative Models</b></summary>

* **Authors:** Xingchen Zeng, Ziyao Gao, Yilin Ye, Wei Zeng
* **arXiv ID:** 2401.15559
* **One-liner:** Proposed IntentTuner, an interactive framework for intent-aligned fine-tuning of T2I models.
* **Published in:** arxiv (28 Jan 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2401.15559) | [[PDF]](https://arxiv.org/pdf/2401.15559) | [[Code]]()

> **核心创新**
> 通过整合用户意图和自动化数据增强，简化了微调过程。

<details>
    <summary>Abstract</summary>
    微调促进了文本到图像生成模型适应新概念（例如，风格和肖像），使用户能够创造性地定制内容。最近关于微调的努力集中在减少训练数据和减轻计算负担上，但忽视了与用户意图的对齐，特别是在手动策划多模态训练数据和意图导向评估方面。通过与微调从业者进行形成性研究以理解用户意图，我们提出了IntentTuner，一个交互式框架，智能地将人类意图整合到微调工作流的每个阶段。IntentTuner使用户能够用图像示例和文本描述表达训练意图，自动将其转换为有效的数据增强策略。此外，IntentTuner引入了新指标来衡量用户意图对齐，允许意图感知的模型训练监控和评估。应用示例和用户研究表明，IntentTuner简化了微调，减少了认知努力，并产生了优于常见基线工具的模型。
</details>

<details>
    <summary>Key points</summary>
    * 用示例和描述表达意图
    * 自动数据增强策略
    * 意图对齐评估的新指标
</details>
</details>

---


<details>
<summary><b> Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation</b></summary>

* **Authors:** Huizhuo Yuan, Zixiang Chen, Kaixuan Ji, Quanquan Gu
* **arXiv ID:** 2402.10210
* **One-liner:** Introduced SPIN-Diffusion, a self-play fine-tuning method for diffusion models.
* **Published in:** arxiv (15 Feb 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2402.10210) | [[PDF]](https://arxiv.org/pdf/2402.10210) | [[Code]]()

> **核心创新**
> 无需额外人类偏好数据，实现了迭代自我改进。

<details>
    <summary>Abstract</summary>
    扩散模型的微调在生成人工智能（GenAI）中仍是一个未充分探索的前沿领域，尤其是在与大型语言模型（LLM）微调取得的显著进展相比时。虽然尖端扩散模型如Stable Diffusion（SD）和SDXL依赖于监督微调，但它们的性能在接触到一定量数据后不可避免地达到平台期。最近，强化学习（RL）被用于使用人类偏好数据微调扩散模型，但它需要每个文本提示至少两个图像（“赢家”和“输家”图像）。在本文中，我们引入了一种创新技术，称为扩散模型的自对弈微调（SPIN-Diffusion），其中扩散模型与其早期版本竞争，促进迭代自我改进过程。我们的方法提供了对传统监督微调和RL策略的替代方案，显著提高了模型性能和对齐度。我们在Pick-a-Pic数据集上的实验表明，SPIN-Diffusion从第一次迭代起就在人类偏好对齐和视觉吸引力方面优于现有监督微调方法。到第二次迭代时，它在所有指标上超过了基于RLHF的方法，并以更少的数据实现了这些结果。
</details>

<details>
    <summary>Key points</summary>
    * 与早期模型版本的自对弈竞争
    * 监督微调和RL的替代方案
    * 改进的对齐度和视觉吸引力
</details>
</details>

---


<details>
<summary><b> Universal Prompt Optimizer for Safe Text-to-Image Generation</b></summary>

* **Authors:** Zongyu Wu, Hongcheng Gao, Yueze Wang, Xiang Zhang, Suhang Wang
* **arXiv ID:** 2402.10882
* **One-liner:** Proposed POSI, a universal prompt optimizer for safe text-to-image generation.
* **Published in:** arxiv (16 Feb 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2402.10882) | [[PDF]](https://arxiv.org/pdf/2402.10882) | [[Code]]()

> **核心创新**
> 通过将有毒提示转换为清洁提示，减少了不安全内容的生成。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）模型在基于文本提示生成图像方面表现出色。然而，这些模型容易受到不安全输入的影响，生成不安全内容，如性、骚扰和非法活动图像。现有基于图像检查器、模型微调和嵌入阻塞的研究在实际应用中不实用。因此，我们提出了第一个在黑盒场景中用于安全T2I生成的通用提示优化器（POSI）。我们首先通过GPT-3.5 Turbo构建了一个包含有毒-清洁提示对的数据集。为了指导优化器具备将有毒提示转换为清洁提示同时保留语义信息的能力，我们设计了一个新颖的奖励函数，测量生成图像的毒性和文本对齐，并通过近端策略优化训练优化器。实验表明，我们的方法能有效降低各种T2I模型生成不当图像的可能性，对文本对齐没有显著影响。它还可以灵活地与其他方法结合以实现更好性能。我们的代码可在<a href="https://github.com/wu-zongyu/POSI" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 有毒-清洁对的数据集构建
    * 毒性和文本对齐的奖励函数
    * 通过近端策略优化进行训练
</details>
</details>

---


<details>
<summary><b> Visual Concept-driven Image Generation with Text-to-Image Diffusion Model</b></summary>

* **Authors:** Tanzila Rahman, Shweta Mahajan, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Leonid Sigal
* **arXiv ID:** 2402.11487
* **One-liner:** Proposed a concept-driven personalization framework for TTI models that enables generation with multiple interacting and entangled concepts.
* **Published in:** arxiv (18 Feb 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2402.11487) | [[PDF]](https://arxiv.org/pdf/2402.11487) | [[Code]]()

> **核心创新**
> 通过EM类优化联合学习自定义标记和潜在分割掩码，以解缠用户提供图像中的概念。

<details>
    <summary>Abstract</summary>
    文本到图像（TTI）扩散模型在生成复杂且富有想象力的高分辨率图像方面已展现出令人瞩目的成果。近期方法通过个性化技术进一步扩展了这些方法，允许其使用少量样本图像插图整合用户描绘的概念（例如用户自身）。然而，生成包含多个交互概念（如人类主体）以及可能在单个或多个图像插图中纠缠的概念的图像能力仍然难以实现。在本工作中，我们提出了一个概念驱动的TTI个性化框架，以解决这些核心挑战。我们基于现有工作，学习用户描绘概念的自定义标记，使其能够与TTI模型中的现有文本标记交互。但重要的是，为了解缠并更好地学习相关概念，我们联合学习（潜在）分割掩码，以在用户提供的图像插图中解缠这些概念。我们通过引入一种期望最大化（EM）类优化过程来实现这一点，在该过程中我们交替学习自定义标记和估计用户提供图像中对应概念的（潜在）掩码。我们基于U-Net参数化潜在扩散模型内的交叉注意力和随后的DenseCRF优化来获取这些掩码。我们证明这种联合交替细化导致学习到更好的概念标记，并作为副产品，产生潜在掩码。我们通过多个示例和用例定性和定量地说明了所提出方法的益处，这些用例可以组合三个或更多纠缠概念。
</details>

<details>
    <summary>Key points</summary>
    * 学习用户描绘概念的自定义标记
    * 通过交叉注意力和DenseCRF优化联合学习潜在分割掩码
    * 交替优化标记和掩码以改进概念学习
</details>
</details>

---


<details>
<summary><b> A User-Friendly Framework for Generating Model-Preferred Prompts in Text-to-Image Synthesis</b></summary>

* **Authors:** Nailei Hei, Qianyu Guo, Zihao Wang, Yan Wang, Haofen Wang, Wenqiang Zhang
* **arXiv ID:** 2402.12760
* **One-liner:** Developed an automated prompt optimization framework to bridge the gap between novice user inputs and model-preferred prompts.
* **Published in:** arxiv (20 Feb 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2402.12760) | [[PDF]](https://arxiv.org/pdf/2402.12760) | [[Code]]()

> **核心创新**
> 引入了用户友好细粒度文本生成框架和粗-细粒度提示数据集用于自动提示精炼。

<details>
    <summary>Abstract</summary>
    精心设计的提示已证明能够引导文本到图像模型生成惊人图像。尽管现有提示工程方法可以提供高级指导，但由于新手用户输入提示与模型偏好提示之间存在差异，新手用户通过手动输入提示实现期望结果具有挑战性。为了弥合用户输入行为与模型训练数据集之间的分布差距，我们首先构建了一个新颖的粗-细粒度提示数据集（CFP），并提出了一个新颖的用户友好细粒度文本生成框架（UF-FGTG）用于自动提示优化。对于CFP，我们为文本到图像任务构建了一个新颖数据集，结合粗和细粒度提示，以促进自动提示生成方法的发展。对于UF-FGTG，我们提出了一个新颖框架，自动将用户输入提示转换为模型偏好提示。具体来说，我们提出了一个提示精炼器，持续重写提示，使用户能够选择符合其独特需求的结果。同时，我们将文本到图像模型中的图像相关损失函数集成到文本生成的训练过程中，以生成模型偏好提示。此外，我们提出了一个自适应特征提取模块，以确保生成结果的多样性。实验证明，我们的方法能够生成比先前最先进方法更具视觉吸引力和多样性的图像，在六个质量和美学指标上平均提高了5%。
</details>

<details>
    <summary>Key points</summary>
    * 构建了包含粗和细粒度提示的CFP数据集
    * 实现了用于持续重写的提示精炼器
    * 集成了图像相关损失函数和自适应特征提取以增强多样性
</details>
</details>

---


<details>
<summary><b> Scaling Rectified Flow Transformers for High-Resolution Image Synthesis</b></summary>

* **Authors:** Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, Robin Rombach
* **arXiv ID:** 2403.03206
* **One-liner:** Improved rectified flow models with biased noise sampling and introduced a novel transformer architecture for superior text-to-image synthesis.
* **Published in:** arxiv (5 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.03206) | [[PDF]](https://arxiv.org/pdf/2403.03206) | [[Code]](https://github.com/Stability-AI/sd3.5)

> **核心创新**
> 通过偏向感知相关尺度的噪声采样和改进的双向Transformer架构，增强了文本理解和排版。

<details>
    <summary>Abstract</summary>
    扩散模型通过将数据向噪声的正向路径反转，从噪声中创建数据，并已成为高维感知数据（如图像和视频）的强大生成建模技术。整流流是最近的生成模型公式，将数据和噪声以直线连接。尽管其具有更好的理论性质和概念简单性，但尚未被确立为标准实践。在本工作中，我们通过将噪声采样偏向感知相关尺度，改进了训练整流流模型的现有噪声采样技术。通过大规模研究，我们证明了这种方法在高分辨率文本到图像合成中优于已建立的扩散公式。此外，我们提出了一种新颖的基于Transformer的架构用于文本到图像生成，该架构对两种模态使用独立权重，并实现图像和文本标记之间的双向信息流，改进了文本理解、排版和人类偏好评分。我们证明该架构遵循可预测的缩放趋势，并将较低的验证损失与通过各种指标和人类评估衡量的改进文本到图像合成相关联。我们的最大模型优于最先进模型，我们将公开我们的实验数据、代码和模型权重。
</details>

<details>
    <summary>Key points</summary>
    * 偏向感知相关尺度的噪声采样
    * 新颖Transformer架构，具有模态独立权重和双向流
    * 展示了可预测缩放和与改进合成指标的相关性
</details>
</details>

---


<details>
<summary><b> PromptCharm: Text-to-Image Generation through Multi-modal Prompting and Refinement</b></summary>

* **Authors:** Zhijie Wang, Yuheng Huang, Da Song, Lei Ma, Tianyi Zhang
* **arXiv ID:** 2403.04014
* **One-liner:** Created PromptCharm, a mixed-initiative system to assist novice users in text-to-image creation through prompt engineering and refinement.
* **Published in:** arxiv (6 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.04014) | [[PDF]](https://arxiv.org/pdf/2403.04014) | [[Code]]()

> **核心创新**
> 结合了自动提示优化、风格探索、模型解释可视化和交互式精炼工具。

<details>
    <summary>Abstract</summary>
    生成式AI的最新进展显著推动了文本到图像生成领域的发展。最先进的文本到图像模型Stable Diffusion现在能够合成具有强烈美学感的高质量图像。因此，制作与模型解释和用户意图对齐的文本提示变得至关重要。然而，由于Stable Diffusion模型的复杂性和迭代编辑和精炼文本提示所需的非平凡努力，提示对于新手用户仍然具有挑战性。为了解决这些挑战，我们提出了PromptCharm，一个混合主动系统，通过多模态提示工程和精炼促进文本到图像创作。为了帮助新手用户提示，PromptCharm首先自动精炼和优化用户的初始提示。此外，PromptCharm支持用户在大型数据库中探索和选择不同的图像风格。为了帮助用户有效精炼其提示和图像，PromptCharm通过可视化模型的注意力值来呈现模型解释。如果用户在生成图像中注意到任何不满意区域，他们可以通过模型注意力调整或图像修复在PromptCharm的丰富反馈循环中进一步精炼图像。为了评估PromptCharm的有效性和可用性，我们进行了控制用户研究（12名参与者）和探索性用户研究（另12名参与者）。这两项研究表明，与使用缺乏交互或可视化支持的PromptCharm变体相比，使用PromptCharm的参与者能够创建更高质量且更符合用户期望的图像。
</details>

<details>
    <summary>Key points</summary>
    * 自动提示精炼和优化
    * 从大型数据库中选择风格
    * 模型注意力可视化和调整，图像修复用于反馈
</details>
</details>

---


<details>
<summary><b> Discriminative Probing and Tuning for Text-to-Image Generation</b></summary>

* **Authors:** Leigang Qu, Wenjie Wang, Yongqi Li, Hanwang Zhang, Liqiang Nie, Tat-Seng Chua
* **arXiv ID:** 2403.04321
* **One-liner:** Enhanced text-to-image alignment by bolstering discriminative abilities through a discriminative adapter and fine-tuning.
* **Published in:** arxiv (7 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.04321) | [[PDF]](https://arxiv.org/pdf/2403.04321) | [[Code]](https://github.com/LgQu/DPT-T2I)

> **核心创新**
> 提出使用判别式建模改进生成对齐，并在推理时使用自校正机制。

<details>
    <summary>Abstract</summary>
    尽管文本到图像生成（T2I）取得了进展，先前方法经常面临文本-图像不对齐问题，如生成图像中的关系混淆。现有解决方案涉及交叉注意力操作以改进组合理解，或集成大型语言模型以改进布局规划。然而，T2I模型的固有对齐能力仍然不足。通过回顾生成式和判别式建模之间的联系，我们假设T2I模型的判别能力可能反映其在生成过程中的文本-图像对齐熟练度。基于此，我们主张增强T2I模型的判别能力，以实现更精确的文本到图像对齐生成。我们提出了一个基于T2I模型的判别适配器，以探测其在两个代表性任务上的判别能力，并利用判别微调改进其文本-图像对齐。作为判别适配器的额外好处，自校正机制可以在推理过程中利用判别梯度更好地将生成图像与文本提示对齐。在三个基准数据集（包括分布内和分布外场景）上的综合评估证明了我们方法的优越生成性能。同时，与其他生成模型相比，它在两个判别任务上实现了最先进的判别性能。
</details>

<details>
    <summary>Key points</summary>
    * 构建了T2I模型的判别适配器
    * 在代表性任务上进行判别微调
    * 使用判别梯度的自校正机制
</details>
</details>

---


<details>
<summary><b> PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation</b></summary>

* **Authors:** Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, Zhenguo Li
* **arXiv ID:** 2403.04692
* **One-liner:** Introduced PixArt-Σ, a Diffusion Transformer model for efficient 4K resolution image generation with high fidelity and text alignment.
* **Published in:** arxiv (7 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.04692) | [[PDF]](https://arxiv.org/pdf/2403.04692) | [[Code]](https://github.com/PixArt-alpha/PixArt-sigma)

> **核心创新**
> 通过弱到强训练与高质量数据和高效标记压缩实现了训练效率。

<details>
    <summary>Abstract</summary>
    在本文中，我们介绍了PixArt-Σ，一种扩散Transformer模型（DiT），能够直接生成4K分辨率图像。PixArt-Σ相比其前身PixArt-α有显著进步，提供明显更高保真度和改进文本提示对齐的图像。PixArt-Σ的一个关键特征是其训练效率。利用PixArt-α的基础预训练，它通过纳入更高质量数据从“较弱”基线演变为“较强”模型，这一过程我们称为“弱到强训练”。PixArt-Σ的进步有两方面：（1）高质量训练数据：PixArt-Σ纳入了更高质量的图像数据，配以更精确和详细的图像标题。（2）高效标记压缩：我们在DiT框架内提出了一种新颖的注意力模块，压缩键和值，显著提高效率并促进超高分辨率图像生成。得益于这些改进，PixArt-Σ以显著更小的模型大小（0.6B参数）实现了优越的图像质量和用户提示遵循能力，优于现有文本到图像扩散模型，如SDXL（2.6B参数）和SD Cascade（5.1B参数）。此外，PixArt-Σ生成4K图像的能力支持高分辨率海报和壁纸的创作，有效促进电影和游戏等行业高质量视觉内容的生产。
</details>

<details>
    <summary>Key points</summary>
    * 纳入了高质量训练数据与详细标题
    * 在DiT框架中提出了高效标记压缩
    * 以较小模型大小实现4K图像生成
</details>
</details>

---


<details>
<summary><b> A 28.6 mJ/iter Stable Diffusion Processor for Text-to-Image Generation with Patch Similarity-based Sparsity Augmentation and Text-based Mixed-Precision</b></summary>

* **Authors:** Jiwon Choi, Wooyoung Jo, Seongyon Hong, Beomseok Kwon, Wonhoon Park, Hoi-Jun Yoo
* **arXiv ID:** 2403.04982
* **One-liner:** Designed an energy-efficient stable diffusion processor for mobile deployment with high throughput and reduced power consumption.
* **Published in:** arxiv (8 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.04982) | [[PDF]](https://arxiv.org/pdf/2403.04982) | [[Code]]()

> **核心创新**
> 实现了包括稀疏增强、重要像素定位和双模核心架构的硬件优化。

<details>
    <summary>Abstract</summary>
    本文提出了一种用于文本到图像生成的高效能Stable Diffusion处理器。尽管Stable Diffusion因高质量图像合成结果而受到关注，但其固有特性阻碍了在移动平台上的部署。所提出的处理器通过三个关键特征实现高吞吐量和能效：1）基于补丁相似性的稀疏增强（PSSA）将自注意力得分的外部存储器访问（EMA）能量减少60.3%，导致总EMA能量减少37.8%。2）基于文本的重要像素定位（TIPS）允许44.8%的FFN层工作负载以低精度激活处理。3）双模位切片核心（DBSC）架构将FFN层的能效提高43.0%。所提出的处理器在28 nm CMOS技术中实现，达到3.84 TOPS峰值吞吐量，平均功耗为225.6 mW。总之，在MS-COCO数据集上可以实现28.6 mJ/迭代的高能效文本到图像生成处理器。
</details>

<details>
    <summary>Key points</summary>
    * 基于补丁相似性的稀疏增强以减少存储器能量
    * 基于文本的重要像素定位用于低精度处理
    * 双模位切片核心以增强FFN层能效
</details>
</details>

---


<details>
<summary><b> CogView3: Finer and Faster Text-to-Image Generation via Relay Diffusion</b></summary>

* **Authors:** Wendi Zheng, Jiayan Teng, Zhuoyi Yang, Weihan Wang, Jidong Chen, Xiaotao Gu, Yuxiao Dong, Ming Ding, Jie Tang
* **arXiv ID:** 2403.05121
* **One-liner:** Proposed CogView3, a cascaded framework using relay diffusion for efficient and high-quality text-to-image generation.
* **Published in:** arxiv (8 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.05121) | [[PDF]](https://arxiv.org/pdf/2403.05121) | [[Code]]()

> **核心创新**
> 引入了基于中继的超分辨率以减少训练和推理成本，同时提高性能。

<details>
    <summary>Abstract</summary>
    文本到图像生成系统的最新进展主要由扩散模型驱动。然而，单阶段文本到图像扩散模型仍然面临挑战，包括计算效率和图像细节精炼方面。为了解决这个问题，我们提出了CogView3，一个创新的级联框架，增强了文本到图像扩散的性能。CogView3是第一个在文本到图像生成领域实现中继扩散的模型，通过首先生成低分辨率图像，然后应用中继超分辨率来执行任务。这种方法不仅产生有竞争力的文本到图像输出，还大大降低了训练和推理成本。我们的实验结果表明，CogView3在人类评估中优于当前最先进的开源文本到图像扩散模型SDXL 77.0%，同时仅需要约1/2的推理时间。CogView3的蒸馏变体实现了可比性能，同时仅利用SDXL推理时间的1/10。
</details>

<details>
    <summary>Key points</summary>
    * 实现了中继扩散用于从低分辨率到高分辨率生成
    * 以减少推理时间实现有竞争力输出
    * 蒸馏变体以进一步效率增益
</details>
</details>

---


<details>
<summary><b> DivCon: Divide and Conquer for Complex Numerical and Spatial Reasoning in Text-to-Image Generation</b></summary>

* **Authors:** Yuhao Jia, Wenhan Tan
* **arXiv ID:** 2403.06400
* **One-liner:** Introduced a divide-and-conquer approach for layout-based text-to-image generation to handle complex spatial relationships with lightweight LLMs.
* **Published in:** arxiv (11 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.06400) | [[PDF]](https://arxiv.org/pdf/2403.06400) | [[Code]]()

> **核心创新**
> 将布局预测和图像生成解耦为子任务以改进准确性和感知质量。

<details>
    <summary>Abstract</summary>
    扩散驱动的文本到图像（T2I）生成近年来取得了显著进展。为了进一步提高T2I模型在数值和空间推理方面的能力，布局被用作中介来桥接大型语言模型和基于布局的扩散模型。然而，这些方法通常依赖闭源、大规模LLM进行布局预测，限制了可访问性和可扩展性。它们还难以从具有多个对象和复杂空间关系的提示生成图像。为了解决这些挑战，我们引入了一种分而治之的方法，将生成任务解耦为多个子任务。首先，布局预测阶段被分为数值和空间推理与边界框视觉规划，使即使轻量级LLM也能实现与大规模模型相当的布局准确性。其次，布局到图像生成阶段被分为两步，从简单对象到困难对象合成对象。实验在HRS和NSR-1K基准上进行，我们的方法以显著优势优于先前方法。此外，视觉结果和用户研究表明，我们的方法显著提高了感知质量，特别是在从复杂文本提示生成多个对象时。
</details>

<details>
    <summary>Key points</summary>
    * 将布局预测分为推理和视觉规划
    * 从简单到困难的顺序对象合成
    * 在基准测试中增强多对象性能
</details>
</details>

---


<details>
<summary><b> Block-wise LoRA: Revisiting Fine-grained LoRA for Effective Personalization and Stylization in Text-to-Image Generation</b></summary>

* **Authors:** Likun Li, Haoqi Zeng, Changpeng Yang, Haozhe Jia, Di Xu
* **arXiv ID:** 2403.07500
* **One-liner:** Proposed block-wise Low-Rank Adaptation for effective personalization and stylization in text-to-image generation.
* **Published in:** arxiv (12 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.07500) | [[PDF]](https://arxiv.org/pdf/2403.07500) | [[Code]]()

> **核心创新**
> 通过对不同SD块进行细粒度微调实现忠实和风格化图像生成。

<details>
    <summary>Abstract</summary>
    文本到图像中个性化和风格化的目标是指导预训练扩散模型分析用户引入的新概念，并将其融入期望风格。最近，参数高效微调（PEFT）方法已被广泛采用以解决此任务，并极大地推动了该领域的发展。尽管其流行，现有高效微调方法仍然难以在T2I生成中实现有效的个性化和风格化。为了解决这个问题，我们提出了块级低秩适应（LoRA）来对SD的不同块进行细粒度微调，这可以生成忠实于输入提示和目标身份且具有期望风格的图像。广泛实验证明了所提出方法的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 应用块级LoRA进行参数高效微调
    * 微调Stable Diffusion的不同块
    * 生成与提示、身份和风格对齐的图像
</details>
</details>

---


<details>
<summary><b> Optimizing Negative Prompts for Enhanced Aesthetics and Fidelity in Text-To-Image Generation</b></summary>

* **Authors:** Michael Ogezi, Ning Shi
* **arXiv ID:** 2403.07605
* **One-liner:** Proposed NegOpt for automated negative prompt optimization, enhancing image quality by 25% in Inception Score.
* **Published in:** arxiv (12 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.07605) | [[PDF]](https://arxiv.org/pdf/2403.07605) | [[Code]](https://huggingface.co/datasets/mikeogezi/negopt_full)

> **核心创新**
> 开发了一种结合监督微调和强化学习的方法，自动生成有效的负向提示，优于手动方法。

<details>
    <summary>Abstract</summary>
    在文本到图像生成中，使用负向提示来描述不希望的图像特征可以显著提升图像质量。然而，生成良好的负向提示是手动且繁琐的。为解决这一问题，我们提出了NegOpt，一种新颖的方法，通过监督微调和强化学习来优化负向提示生成，以增强图像生成。我们的组合方法使Inception Score相比其他方法提高了25%，并超越了测试集中的真实负向提示。此外，使用NegOpt，我们可以优先优化对我们最重要的指标。最后，我们构建了Negative Prompts DB（<a href="https://huggingface.co/datasets/mikeogezi/negopt_full" rel="external noopener nofollow" class="link-external link-https">此链接</a>），一个公开可用的负向提示数据集。
</details>

<details>
    <summary>Key points</summary>
    * 监督微调用于初始模型训练
    * 强化学习用于优化负向提示
    * 创建公开数据集（Negative Prompts DB）
</details>
</details>

---


<details>
<summary><b> Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation</b></summary>

* **Authors:** Shihao Zhao, Shaozhe Hao, Bojia Zi, Huaizhe Xu, Kwan-Yee K. Wong
* **arXiv ID:** 2403.07860
* **One-liner:** Introduced LaVi-Bridge, enabling flexible integration of diverse language and vision models for improved text-to-image generation.
* **Published in:** arxiv (12 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.07860) | [[PDF]](https://arxiv.org/pdf/2403.07860) | [[Code]](https://github.com/ShihaoZhaoZSH/LaVi-Bridge)

> **核心创新**
> 创建了一个使用LoRA和适配器的即插即用管道，以结合预训练模型而不修改原始权重，增强了文本对齐和图像质量等能力。

<details>
    <summary>Abstract</summary>
    文本到图像生成在文本到图像扩散模型的引入下取得了显著进展。这些模型通常包括一个解释用户提示的语言模型和一个生成相应图像的视觉模型。随着语言和视觉模型在各自领域的不断进步，探索用更先进的对应组件替换文本到图像扩散模型中的组件具有巨大潜力。因此，一个更广泛的研究目标是调查任何两个不相关的语言和生成视觉模型在文本到图像生成中的集成。在本文中，我们探索了这一目标，并提出了LaVi-Bridge，一个管道，使得能够集成多样化的预训练语言模型和生成视觉模型用于文本到图像生成。通过利用LoRA和适配器，LaVi-Bridge提供了一种灵活且即插即用的方法，无需修改语言和视觉模型的原始权重。我们的管道兼容各种语言模型和生成视觉模型，适应不同的结构。在此框架内，我们证明了集成更先进的模块（如更先进的语言模型或生成视觉模型）可以显著提升能力，如文本对齐或图像质量。我们进行了广泛评估以验证LaVi-Bridge的有效性。代码可在<a href="https://github.com/ShihaoZhaoZSH/LaVi-Bridge" rel="external noopener nofollow" class="link-external link-https">此链接</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 使用LoRA和适配器进行集成
    * 兼容各种语言和生成视觉模型
    * 展示使用更先进模块的改进
</details>
</details>

---


<details>
<summary><b> DialogGen: Multi-modal Interactive Dialogue System for Multi-turn Text-to-Image Generation</b></summary>

* **Authors:** Minbin Huang, Yanxin Long, Xinchi Deng, Ruihang Chu, Jiangfeng Xiong, Xiaodan Liang, Hong Cheng, Qinglin Lu, Wei Liu
* **arXiv ID:** 2403.08857
* **One-liner:** Proposed DialogGen for multi-turn text-to-image generation via MLLM alignment, improving output coherence and modality switching.
* **Published in:** arxiv (13 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.08857) | [[PDF]](https://arxiv.org/pdf/2403.08857) | [[Code]](https://github.com/Centaurusalpha/DialogGen)

> **核心创新**
> 开发了一个包含绘制提示对齐、数据整理和错误纠正的管道，以构建多模态交互对话系统，并在DialogBen基准上进行了验证。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）生成模型近年来取得了显著进展。然而，与这些模型的有效交互对普通用户来说具有挑战性，因为需要专门的提示工程知识，且无法进行多轮图像生成，阻碍了动态和迭代的创作过程。最近的尝试试图将多模态大语言模型（MLLMs）与T2I模型结合，以将用户的自然语言指令变为现实。因此，MLLMs的输出模态被扩展，T2I模型的多轮生成质量得益于MLLMs强大的多模态理解能力而得到增强。然而，随着输出模态数量的增加和对话的深入，许多工作面临识别正确输出模态并相应生成连贯图像的挑战。因此，我们提出了DialogGen，一个有效的管道，用于对齐现成的MLLMs和T2I模型，以构建一个多模态交互对话系统（MIDS），用于多轮文本到图像生成。它包括绘制提示对齐、仔细的训练数据整理和错误纠正。此外，随着MIDS领域的繁荣，迫切需要全面的基准来公平评估MIDS在输出模态正确性和多模态输出连贯性方面的能力。为解决这一问题，我们引入了多模态对话基准（DialogBen），一个全面的双语基准，旨在评估MLLMs生成准确和连贯多模态内容以支持图像编辑的能力。它包含两个评估指标，用于衡量模型切换模态的能力和输出图像的连贯性。我们在DialogBen上的广泛实验和用户研究证明了DialogGen与其他最先进模型相比的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 绘制提示对齐用于MLLM-T2I集成
    * 仔细的训练数据整理
    * 引入DialogBen基准用于评估
</details>
</details>

---


<details>
<summary><b> CLIP-VQDiffusion : Langauge Free Training of Text To Image generation using CLIP and vector quantized diffusion model</b></summary>

* **Authors:** Seungdae Han, Joohee Kim
* **arXiv ID:** 2403.14944
* **One-liner:** Introduced CLIP-VQDiffusion for text-to-image generation without paired captions, achieving state-of-the-art performance on FFHQ.
* **Published in:** arxiv (22 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.14944) | [[PDF]](https://arxiv.org/pdf/2403.14944) | [[Code]](https://github.com/INFINIQ-AI1/CLIPVQDiffusion)

> **核心创新**
> 利用预训练的CLIP进行多模态表示，从文本生成逼真图像，即使在分布外情况下，clipscore比先前方法提高了4.4%。

<details>
    <summary>Abstract</summary>
    文本条件图像生成模型取得了显著进展。该领域的最新进展不仅依赖于模型结构的改进，还依赖于大量文本-图像配对数据集。然而，创建这类数据集成本高昂，需要大量劳动力。著名的面部数据集没有相应的文本标题，使得在这些数据集上开发文本条件图像生成模型变得困难。一些研究专注于仅使用没有文本标题的图像开发文本到图像生成模型。在此，我们提出了CLIP-VQDiffusion，它利用预训练的CLIP模型提供多模态文本-图像表示和强大的图像生成能力。在FFHQ数据集上，我们的模型在clipscore上比先前的最先进方法提高了4.4%，并且在文本分布内和分布外时都生成了非常逼真的图像。预训练模型和代码将很快在<a href="https://github.com/INFINIQ-AI1/CLIPVQDiffusion" rel="external noopener nofollow" class="link-external link-https">此链接</a>提供。
</details>

<details>
    <summary>Key points</summary>
    * 使用CLIP进行文本-图像表示
    * 与VQDiffusion集成用于图像生成
    * 在FFHQ数据集上的评估
</details>
</details>

---


<details>
<summary><b> FlashEval: Towards Fast and Accurate Evaluation of Text-to-image Diffusion Generative Models</b></summary>

* **Authors:** Lin Zhao, Tianchen Zhao, Zinan Lin, Xuefei Ning, Guohao Dai, Huazhong Yang, Yu Wang
* **arXiv ID:** 2403.16379
* **One-liner:** Developed FlashEval for efficient evaluation of text-to-image models, achieving 10x speedup with representative subset selection.
* **Published in:** arxiv (25 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.16379) | [[PDF]](https://arxiv.org/pdf/2403.16379) | [[Code]](https://github.com/thu-nics/FlashEval)

> **核心创新**
> 提出了一种迭代搜索算法来选择文本-图像数据集的子集，在COCO和DiffusionDB上使用更少样本实现可比较的评估质量。

<details>
    <summary>Abstract</summary>
    近年来，文本到图像生成模型的开发取得了显著进展。评估生成模型的质量是开发过程中的关键步骤。不幸的是，评估过程可能消耗大量计算资源，使得所需的周期性模型性能评估（例如，监控训练进度）不切实际。因此，我们寻求通过选择文本-图像数据集的代表性子集来提高评估效率。我们系统地调查了设计选择，包括选择标准（文本特征或基于图像的指标）和选择粒度（提示级别或集合级别）。我们发现先前关于训练数据子集选择的研究见解不适用于此问题，我们提出了FlashEval，一种针对评估数据选择的迭代搜索算法。我们在COCO和DiffusionDB数据集上，针对各种配置（包括架构、量化水平和采样器调度）的扩散模型排名，证明了FlashEval的有效性。我们搜索的50项子集在未见模型上可以达到与随机采样500项子集相当的评估质量，实现了10倍的评估加速。我们发布了这些常用数据集的浓缩子集，以帮助促进扩散算法设计和评估，并将FlashEval开源为未来数据集浓缩的工具，可在<a href="https://github.com/thu-nics/FlashEval" rel="external noopener nofollow" class="link-external link-https">此链接</a>访问。
</details>

<details>
    <summary>Key points</summary>
    * 用于子集选择的迭代搜索算法
    * 调查选择标准和粒度
    * 应用于各种模型配置
</details>
</details>

---


<details>
<summary><b> Skews in the Phenomenon Space Hinder Generalization in Text-to-Image Generation</b></summary>

* **Authors:** Yingshan Chang, Yasi Zhang, Zhiyuan Fang, Yingnian Wu, Yonatan Bisk, Feng Gao
* **arXiv ID:** 2403.16394
* **One-liner:** Identified dataset skew as a cause of generalization failures in entity-relation compositions and proposed metrics to quantify it.
* **Published in:** arxiv (25 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.16394) | [[PDF]](https://arxiv.org/pdf/2403.16394) | [[Code]]()

> **核心创新**
> 引入了语言和视觉偏斜的统计指标，表明受控扰动在不增加数据大小的情况下改善泛化。

<details>
    <summary>Abstract</summary>
    文本到图像生成的文献中，忠实组合实体与关系的问题一直存在。但对于如何有效学习实体-关系组合缺乏形式化理解。此外，有意义反映问题结构的基础现象空间定义不清，导致了对更大规模数据的竞赛，希望泛化能力从大规模预训练中涌现。我们假设基础现象覆盖没有按比例扩展，导致呈现现象的偏斜，从而损害泛化。我们引入了统计指标，量化数据集在关系学习中的语言和视觉偏斜，并表明文本到图像生成的泛化失败是现象覆盖不完整或不平衡的直接结果。我们首先在合成领域进行实验，证明系统控制的指标强烈预测泛化性能。然后我们转向自然图像，并展示根据我们理论进行的简单分布扰动可以提升泛化，而不增加绝对数据大小。这项工作指示了一个重要方向，即提高数据多样性或平衡，与扩大绝对规模正交。我们的讨论指出了重要开放问题：1）生成实体-关系组合的评估，和2）用于抽象关系推理的更好模型。
</details>

<details>
    <summary>Key points</summary>
    * 开发数据集偏斜指标
    * 在合成和自然领域进行实验
    * 通过分布扰动展示泛化改进
</details>
</details>

---


<details>
<summary><b> Refining Text-to-Image Generation: Towards Accurate Training-Free Glyph-Enhanced Image Generation</b></summary>

* **Authors:** Sanyam Lakhanpal, Shivang Chopra, Vinija Jain, Aman Chadha, Man Luo
* **arXiv ID:** 2403.16422
* **One-liner:** Proposed a training-free framework to enhance visual text generation, improving OCR metrics by over 23% on LenCom-Eval benchmark.
* **Published in:** arxiv (25 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.16422) | [[PDF]](https://arxiv.org/pdf/2403.16422) | [[Code]]()

> **核心创新**
> 引入了LenCom-Eval基准用于长且复杂文本图像，以及一种无需训练的方法来增强两阶段生成方法。

<details>
    <summary>Abstract</summary>
    过去几年中，基于扩散模型的文本到图像（T2I）生成方法获得了广泛关注。然而，普通扩散模型通常在生成图像中显示的文本拼写不准确。生成视觉文本的能力至关重要，具有学术兴趣和广泛的实际应用。为生成准确的视觉文本图像，最先进的技术采用字形控制图像生成方法，包括一个文本布局生成器，后跟一个以生成文本布局为条件的图像生成器。然而，我们的研究揭示这些模型仍面临三个主要挑战，促使我们开发一个测试平台以促进未来研究。我们引入了一个基准，LenCom-Eval，专门设计用于测试模型生成长且复杂视觉文本图像的能力。随后，我们引入了一个无需训练的框架来增强两阶段生成方法。我们在LenCom-Eval和MARIO-Eval基准上检验了我们方法的有效性，并在一系列评估指标上展示了显著改进，包括CLIPScore、OCR精确率、召回率、F1分数、准确率和编辑距离分数。例如，我们提出的框架将骨干模型TextDiffuser在LenCom-Eval和MARIO-Eval上的OCR单词F1分别提高了超过23%和13.5%。我们的工作通过专注于生成长且稀有文本序列的图像，为该领域做出了独特贡献，这是现有文献未探索的利基。
</details>

<details>
    <summary>Key points</summary>
    * 创建LenCom-Eval基准
    * 无需训练增强字形控制生成
    * 在多个OCR指标上的评估
</details>
</details>

---


<details>
<summary><b> Isolated Diffusion: Optimizing Multi-Concept Text-to-Image Generation Training-Freely with Isolated Diffusion Guidance</b></summary>

* **Authors:** Jingyuan Zhu, Huimin Ma, Jiansheng Chen, Jian Yuan
* **arXiv ID:** 2403.16954
* **One-liner:** Introduced Isolated Diffusion to address concept bleeding in multi-concept generation, improving text-image consistency.
* **Published in:** arxiv (25 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.16954) | [[PDF]](https://arxiv.org/pdf/2403.16954) | [[Code]]()

> **核心创新**
> 提出了一种无需训练策略，使用分割提示和主题隔离与预训练模型，兼容SD和SDXL。

<details>
    <summary>Abstract</summary>
    大规模文本到图像扩散模型在给定目标文本提示下合成高质量和多样化图像方面取得了巨大成功。尽管具有革命性的图像生成能力，当前最先进的模型在许多情况下仍难以准确处理多概念生成。这种现象被称为“概念渗漏”，表现为不同概念的意外重叠或合并。本文提出了一种通用方法，用于文本到图像扩散模型解决复杂场景中不同主题及其附件之间的相互干扰，追求更好的文本-图像一致性。核心思想是隔离不同概念的合成过程。我们建议使用分割文本提示将每个附件分别绑定到相应主题。此外，我们引入了一种修正方法来解决多主题合成中的概念渗漏问题。我们首先依赖预训练的对象检测和分割模型来获取主题的布局。然后我们隔离并单独重新合成每个主题，使用相应的文本提示以避免相互干扰。总体而言，我们实现了一种无需训练的策略，名为Isolated Diffusion，以优化多概念文本到图像合成。它与最新的Stable Diffusion XL（SDXL）和先前的Stable Diffusion（SD）模型兼容。我们使用各种多概念文本提示将我们的方法与替代方法进行比较，并在文本-图像一致性和用户研究中展示了其有效性，具有明显优势。
</details>

<details>
    <summary>Key points</summary>
    * 分割文本提示用于主题绑定
    * 使用对象检测和分割获取布局
    * 隔离和重新合成主题
</details>
</details>

---


<details>
<summary><b> Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation</b></summary>

* **Authors:** Omer Dahary, Or Patashnik, Kfir Aberman, Daniel Cohen-Or
* **arXiv ID:** 2403.16990
* **One-liner:** Developed Bounded Attention to prevent semantic leakage in multi-subject generation, enhancing subject individuality.
* **Published in:** arxiv (25 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.16990) | [[PDF]](https://arxiv.org/pdf/2403.16990) | [[Code]](https://github.com/omer11a/bounded-attention)

> **核心创新**
> 分析了注意力层问题，并引入了一种无需训练的方法来限制采样过程中的信息流，改善与提示和布局的对齐。

<details>
    <summary>Abstract</summary>
    文本到图像扩散模型具有前所未有的能力，生成多样化和高质量的图像。然而，它们通常难以忠实捕捉包含多个主题的复杂输入提示的预期语义。最近，引入了许多布局到图像扩展以改善用户控制，旨在定位由特定令牌表示的主题。然而，这些方法通常产生语义不准确的图像，尤其是在处理多个语义或视觉相似主题时。在这项工作中，我们研究并分析了这些限制的原因。我们的探索揭示，主要问题源于去噪过程中主题之间的无意语义泄漏。这种泄漏归因于扩散模型的注意力层，这些层倾向于混合不同主题的视觉特征。为解决这些问题，我们引入了Bounded Attention，一种无需训练的方法，用于在采样过程中限制信息流。Bounded Attention防止主题之间的有害泄漏，并能够引导生成以促进每个主题的个性，即使在复杂多主题条件下。通过广泛实验，我们证明了我们的方法能够生成与给定提示和布局更好对齐的多个主题。
</details>

<details>
    <summary>Key points</summary>
    * 分析注意力层中的语义泄漏
    * 引入Bounded Attention方法
    * 复杂多主题条件下的实验
</details>
</details>

---


<details>
<summary><b> Capability-aware Prompt Reformulation Learning for Text-to-Image Generation</b></summary>

* **Authors:** Jingtao Zhan, Qingyao Ai, Yiqun Liu, Jia Chen, Shaoping Ma
* **arXiv ID:** 2403.19716
* **One-liner:** Proposed CAPR for automatic prompt reformulation based on user capability, improving interaction with text-to-image systems.
* **Published in:** arxiv (27 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.19716) | [[PDF]](https://arxiv.org/pdf/2403.19716) | [[Code]]()

> **核心创新**
> 开发了一个框架，包含条件重新表述模型和可配置能力特征，以学习多样化策略并模拟高能力用户。

<details>
    <summary>Abstract</summary>
    文本到图像生成系统已成为艺术创作领域的革命性工具，提供了前所未有的便利，将文本提示转化为视觉艺术。然而，这些系统的效能与用户提供的提示质量密切相关，这对不熟悉提示制作的用户来说往往构成挑战。本文通过利用交互日志中的用户重新表述数据来应对这一挑战，开发了一个自动提示重新表述模型。我们对这些日志的深入分析揭示，用户提示重新表述严重依赖于个体用户的能力，导致重新表述对的质量存在显著差异。为有效使用这些数据进行训练，我们引入了能力感知提示重新表述（CAPR）框架。CAPR通过两个关键组件创新性地将用户能力整合到重新表述过程中：条件重新表述模型（CRM）和可配置能力特征（CCF）。CRM根据指定的用户能力（由CCF表示）重新表述提示。CCF反过来提供了调整和指导CRM行为的灵活性。这使得CAPR能够有效学习跨不同用户能力的多样化重新表述策略，并在推理过程中模拟高能力用户的重新表述。在标准文本到图像生成基准上的广泛实验展示了CAPR优于现有基线的性能及其在未见系统上的显著鲁棒性。此外，全面分析验证了不同组件的有效性。CAPR可以促进用户与文本到图像系统的友好交互，并使更广泛的用户更容易实现高级艺术创作。
</details>

<details>
    <summary>Key points</summary>
    * 将用户能力整合到重新表述中
    * 使用条件重新表述模型
    * 可配置能力特征用于调整行为
</details>
</details>

---


<details>
<summary><b> Evaluating Text-to-Visual Generation with Image-to-Text Generation</b></summary>

* **Authors:** Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, Deva Ramanan
* **arXiv ID:** 2404.01291
* **One-liner:** Introduced VQAScore for improved image-text alignment evaluation using VQA models.
* **Published in:** arxiv (1 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.01291) | [[PDF]](https://arxiv.org/pdf/2404.01291) | [[Code]](https://github.com/linzhiqiu/t2v_metrics)

> **核心创新**
> VQAScore通过使用VQA模型计算对齐分数，解决了CLIPScore的局限性，在基准测试中实现了最先进的结果，并支持复杂组合提示的评估。

<details>
    <summary>Abstract</summary>
    尽管生成式AI取得了显著进展，但由于缺乏有效指标和标准化基准，全面评估仍然具有挑战性。
</details>

<details>
    <summary>Key points</summary>
    * 使用VQA模型计算'这个图是否显示{文本}？'问题的'是'答案概率
    * 开发了具有双向编码器的内部CLIP-FlanT5模型
    * 引入了包含1600个组合提示的GenAI-Bench基准用于基准测试
</details>
</details>

---


<details>
<summary><b> InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation</b></summary>

* **Authors:** Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, Anthony Chen
* **arXiv ID:** 2404.02733
* **One-liner:** Proposed InstantStyle for tuning-free style-consistent image generation.
* **Published in:** arxiv (3 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.02733) | [[PDF]](https://arxiv.org/pdf/2404.02733) | [[Code]](https://github.com/instantX-research/InstantStyle)

> **核心创新**
> InstantStyle通过特征空间解耦风格和内容，并将特征仅注入风格特定块，以防止风格退化并避免权重调优。

<details>
    <summary>Abstract</summary>
    免调优的扩散模型在图像个性化和定制领域显示出巨大潜力，但当前模型在生成风格一致的图像时仍面临复杂挑战。
</details>

<details>
    <summary>Key points</summary>
    * 通过特征加减法解耦风格和内容
    * 仅将参考图像特征注入风格特定块
    * 消除权重调优需求并防止风格泄漏
</details>
</details>

---


<details>
<summary><b> MULAN: A Multi Layer Annotated Dataset for Controllable Text-to-Image Generation</b></summary>

* **Authors:** Petru-Daniel Tudosiu, Yongxin Yang, Shifeng Zhang, Fei Chen, Steven McDonagh, Gerasimos Lampouras, Ignacio Iacobacci, Sarah Parisot
* **arXiv ID:** 2404.02790
* **One-liner:** Created MuLAn dataset for instance-wise image decomposition to aid text-to-image generation.
* **Published in:** arxiv (3 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.02790) | [[PDF]](https://arxiv.org/pdf/2404.02790) | [[Code]](https://mulan-dataset.github.io/)

> **核心创新**
> MuLAn提供了RGB图像的多层注释作为RGBA分解，无需训练即可支持分层生成和编辑研究。

<details>
    <summary>Abstract</summary>
    文本到图像生成取得了惊人成果，但精确的空间可控性和提示保真度仍然极具挑战性。
</details>

<details>
    <summary>Key points</summary>
    * 开发了免训练管道，将图像分解为RGBA层
    * 包括实例发现、补全和重组模块
    * 构建了包含超过44K注释的MuLAn-COCO和MuLAn-LAION数据集
</details>
</details>

---


<details>
<summary><b> On the Scalability of Diffusion-based Text-to-Image Generation</b></summary>

* **Authors:** Hao Li, Yang Zou, Ying Wang, Orchid Majumder, Yusheng Xie, R. Manmatha, Ashwin Swaminathan, Zhuowen Tu, Stefano Ermon, Stefano Soatto
* **arXiv ID:** 2404.02883
* **One-liner:** Empirically studied scaling laws for diffusion-based text-to-image models to optimize performance and cost.
* **Published in:** arxiv (3 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.02883) | [[PDF]](https://arxiv.org/pdf/2404.02883) | [[Code]]()

> **核心创新**
> 识别了高效的UNet设计和数据缩放策略，表明Transformer块和数据集质量/多样性对文本-图像对齐至关重要。

<details>
    <summary>Abstract</summary>
    扩大模型和数据规模对LLMs的演进非常成功，但扩散式文本到图像模型的缩放规律尚未充分探索。
</details>

<details>
    <summary>Key points</summary>
    * 对0.4B到4B参数的UNet和Transformer变体进行了消融实验
    * 发现增加Transformer块比增加通道数更具参数效率
    * 显示标题密度和多样性提高对齐性和学习效率
</details>
</details>

---


<details>
<summary><b> RL for Consistency Models: Faster Reward Guided Text-to-Image Generation</b></summary>

* **Authors:** Owen Oertell, Jonathan D. Chang, Yiyi Zhang, Kianté Brantley, Wen Sun
* **arXiv ID:** 2404.03673
* **One-liner:** Developed RLCM for fast reinforcement learning fine-tuning of consistency models in text-to-image generation.
* **Published in:** arxiv (25 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.03673) | [[PDF]](https://arxiv.org/pdf/2404.03673) | [[Code]](https://github.com/Owen-Oertell/rlcm)

> **核心创新**
> RLCM将一致性模型推理框架化为RL过程，实现更快训练和推理，在少数步骤内生成高质量图像，可适应各种奖励。

<details>
    <summary>Abstract</summary>
    强化学习通过直接优化奖励改进了扩散模型的引导图像生成，但生成过程缓慢。
</details>

<details>
    <summary>Key points</summary>
    * 将一致性模型的迭代推理框架化为RL过程
    * 实现仅两步生成
    * 适应图像可压缩性和人类反馈等目标
</details>
</details>

---


<details>
<summary><b> Dynamic Prompt Optimizing for Text-to-Image Generation</b></summary>

* **Authors:** Wenyi Mo, Tianyu Zhang, Yalong Bai, Bing Su, Ji-Rong Wen, Qing Yang
* **arXiv ID:** 2404.04095
* **One-liner:** Introduced PAE for automatic prompt editing to improve image generation quality.
* **Published in:** arxiv (5 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.04095) | [[PDF]](https://arxiv.org/pdf/2404.04095) | [[Code]](https://github.com/Mowenyii/PAE)

> **核心创新**
> PAE使用强化学习动态调整提示中的词权重和注入时间步，增强美学和语义对齐，无需手动干预。

<details>
    <summary>Abstract</summary>
    文本到图像生成模型，特别是基于扩散的模型，取得了实质性进展，但精细提示调整需要大量手动干预。
</details>

<details>
    <summary>Key points</summary>
    * 采用在线RL探索词权重和注入时间步
    * 奖励函数考虑美学分数、语义一致性和用户偏好
    * 改进提示以生成更具视觉吸引力和对齐的图像
</details>
</details>

---


<details>
<summary><b> SafeGen: Mitigating Sexually Explicit Content Generation in Text-to-Image Models</b></summary>

* **Authors:** Xinfeng Li, Yuchen Yang, Jiangyi Deng, Chen Yan, Yanjiao Chen, Xiaoyu Ji, Wenyuan Xu
* **arXiv ID:** 2404.06666
* **One-liner:** Proposed SafeGen to mitigate NSFW content generation in text-to-image models in a text-agnostic manner.
* **Published in:** arxiv (10 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.06666) | [[PDF]](https://arxiv.org/pdf/2404.06666) | [[Code]]()

> **核心创新**
> SafeGen从模型内部消除显式视觉表示，使其对对抗提示具有抵抗力，同时保持良性图像保真度。

<details>
    <summary>Abstract</summary>
    文本到图像模型可能被诱导生成不安全内容，现有对策大多关注过滤输入输出或抑制不当文本嵌入。
</details>

<details>
    <summary>Key points</summary>
    * 无论文本输入如何，都阻碍不安全的视觉表示
    * 实现99.4%的性内容去除性能
    * 优于八种基线方法并提供对抗提示基准
</details>
</details>

---


<details>
<summary><b> TextCenGen: Attention-Guided Text-Centric Background Adaptation for Text-to-Image Generation</b></summary>

* **Authors:** Tianyi Liang, Jiangqi Liu, Yifei Huang, Shiqi Jiang, Jianshen Shi, Changbo Wang, Chenhui Li
* **arXiv ID:** 2404.11824
* **One-liner:** Developed TextCenGen for generating text-friendly backgrounds in text-to-image models without training.
* **Published in:** arxiv (18 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.11824) | [[PDF]](https://arxiv.org/pdf/2404.11824) | [[Code]](https://github.com/tianyilt/TextCenGen_Background_Adapt)

> **核心创新**
> TextCenGen使用交叉注意力图和力导向图重新定位冲突对象，确保文本放置的平滑背景，同时保持语义保真度。

<details>
    <summary>Abstract</summary>
    文本到图像生成在生成高质量图像方面取得了显著进展，但创建适合文本放置的背景仍然是一个基本挑战。
</details>

<details>
    <summary>Key points</summary>
    * 分析交叉注意力图以识别冲突对象
    * 使用力导向图进行对象重新定位
    * 应用注意力排除约束以优化背景
</details>
</details>

---


<details>
<summary><b> EdgeFusion: On-Device Text-to-Image Generation</b></summary>

* **Authors:** Thibault Castells, Hyoung-Kyu Song, Tairen Piao, Shinkook Choi, Bo-Kyeong Kim, Hanyoung Yim, Changgwun Lee, Jae Gon Kim, Tae-Ho Kim
* **arXiv ID:** 2404.11925
* **One-liner:** Optimized Stable Diffusion for fast, high-quality image generation on edge devices.
* **Published in:** arxiv (18 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.11925) | [[PDF]](https://arxiv.org/pdf/2404.11925) | [[Code]]()

> **核心创新**
> 通过高质量数据和高级蒸馏增强BK-SDM与LCM，在资源受限设备上实现亚秒延迟和两步生成的光真实图像。

<details>
    <summary>Abstract</summary>
    Stable Diffusion的密集计算负担对其实际应用构成重大障碍。
</details>

<details>
    <summary>Key points</summary>
    * 利用生成模型的高质量图像-文本对
    * 为LCM设计了高级蒸馏过程
    * 在边缘设备上实现两步生成和亚秒延迟
</details>
</details>

---


<details>
<summary><b> Object-Attribute Binding in Text-to-Image Generation: Evaluation and Control</b></summary>

* **Authors:** Maria Mihaela Trusca, Wolf Nuyts, Jonathan Thomm, Robert Honig, Thomas Hofmann, Tinne Tuytelaars, Marie-Francine Moens
* **arXiv ID:** 2404.13766
* **One-liner:** Proposed FCA and DisCLIP embeddings to improve attribute-object binding in text-to-image generation.
* **Published in:** arxiv (21 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.13766) | [[PDF]](https://arxiv.org/pdf/2404.13766) | [[Code]]()

> **核心创新**
> FCA使用句法约束控制视觉注意力图，DisCLIP解耦CLIP嵌入，无需额外模型训练即可增强对齐。

<details>
    <summary>Abstract</summary>
    当前扩散模型在给定文本提示时能创建光真实图像，但难以正确将文本中的属性绑定到图像中的对象。
</details>

<details>
    <summary>Key points</summary>
    * 使用句法约束控制交叉注意力图
    * 将多模态CLIP嵌入解耦为DisCLIP
    * 轻松集成到现有扩散模型中以改善绑定
</details>
</details>

---


<details>
<summary><b> Towards Better Text-to-Image Generation Alignment via Attention Modulation</b></summary>

* **Authors:** Yihang Wu, Xiao Cao, Kaixin Li, Zitan Chen, Haonan Wang, Lei Meng, Zhiyong Huang
* **arXiv ID:** 2404.13899
* **One-liner:** Proposed a training-free attribution-focusing mechanism for diffusion models to improve text-image alignment.
* **Published in:** arxiv (22 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.13899) | [[PDF]](https://arxiv.org/pdf/2404.13899) | [[Code]]()

> **核心创新**
> 通过增强注意力调制，在不进行额外训练的情况下解决实体泄漏和属性错位问题。

<details>
    <summary>Abstract</summary>
    在文本到图像生成任务中，扩散模型的进展提升了生成结果的保真度。然而，这些模型在处理包含多个实体和属性的文本提示时面临挑战。注意力分布不均导致实体泄漏和属性错位问题。从头开始训练来解决这一问题需要大量标注数据且资源消耗高。受此启发，我们提出了一种属性聚焦机制，这是一种无需训练的阶段性机制，通过调制扩散模型的注意力来实现。我们的核心思想之一是引导模型在不同时间步长上专注于提示的相应句法成分。为实现这一点，我们在自注意力模块的早期阶段引入了温度控制机制，以减轻实体泄漏问题。对象聚焦掩码方案和阶段性动态权重控制机制被整合到交叉注意力模块中，使模型能更有效地辨别实体间语义信息的归属。在各种对齐场景下的实验结果表明，我们的模型以最小的额外计算成本实现了更好的图像-文本对齐。
</details>

<details>
    <summary>Key points</summary>
    * 阶段性注意力调制
    * 自注意力中的温度控制
    * 交叉注意力中的对象聚焦掩码
    * 动态权重控制
</details>
</details>

---


<details>
<summary><b> Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation</b></summary>

* **Authors:** Xun Wu, Shaohan Huang, Furu Wei
* **arXiv ID:** 2404.15100
* **One-liner:** Created VisionPrefer, a high-quality preference dataset using multimodal LLMs to improve text-to-image model alignment.
* **Published in:** arxiv (23 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.15100) | [[PDF]](https://arxiv.org/pdf/2404.15100) | [[Code]]()

> **核心创新**
> 利用AI生成的合成数据进行指令调优，以增强模型在多个偏好方面的对齐。

<details>
    <summary>Abstract</summary>
    最近的研究表明，利用人类偏好数据集来优化文本到图像生成模型具有巨大潜力，可以增强生成图像与文本提示之间的对齐。尽管有这些进展，当前的人类偏好数据集要么构建成本过高，要么在偏好维度上缺乏多样性，导致在开源文本到图像生成模型的指令调优中适用性有限，并阻碍了进一步探索。为解决这些挑战并促进通过指令调优实现生成模型的对齐，我们利用多模态大语言模型创建了VisionPrefer，一个高质量、细粒度的偏好数据集，捕捉了多个偏好方面。我们聚合了AI标注者在四个方面的反馈：提示遵循、美学、保真度和无害性，以构建VisionPrefer。为验证VisionPrefer的有效性，我们在VisionPrefer上训练了一个奖励模型VP-Score，以指导文本到图像生成模型的训练，且VP-Score的偏好预测准确性与人类标注者相当。此外，我们使用两种强化学习方法对生成模型进行监督微调，以评估VisionPrefer的性能，广泛的实验结果表明，VisionPrefer在组合图像生成中显著提高了文本-图像对齐，例如在美学方面，并在各种图像分布上比先前的人类偏好指标泛化得更好。此外，VisionPrefer表明，将AI生成的合成数据作为监督信号整合，是实现视觉生成模型与人类偏好更好对齐的有前景途径。
</details>

<details>
    <summary>Key points</summary>
    * 使用AI标注者构建数据集
    * 训练VP-Score奖励模型
    * 使用强化学习进行微调
    * 在组合生成上的评估
</details>
</details>

---


<details>
<summary><b> ID-Aligner: Enhancing Identity-Preserving Text-to-Image Generation with Reward Feedback Learning</b></summary>

* **Authors:** Weifeng Chen, Jiacheng Zhang, Jie Wu, Hefeng Wu, Xuefeng Xiao, Liang Lin
* **arXiv ID:** 2404.15449
* **One-liner:** Introduced ID-Aligner, a feedback learning framework for identity-preserving text-to-image generation.
* **Published in:** arxiv (23 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.15449) | [[PDF]](https://arxiv.org/pdf/2404.15449) | [[Code]](https://github.com/Weifeng-Chen/ID-Aligner)

> **核心创新**
> 使用基于奖励的微调改进了生成图像中的身份保持和美学吸引力。

<details>
    <summary>Abstract</summary>
    扩散模型的快速发展引发了多样化的应用。身份保持的文本到图像生成（ID-T2I）尤其受到广泛关注，因为它在AI肖像和广告等应用场景中具有广泛用途。尽管现有的ID-T2I方法已展示出令人印象深刻的结果，但仍存在几个关键挑战：（1）难以准确保持参考肖像的身份特征，（2）在强制身份保留时，生成的图像缺乏美学吸引力，以及（3）存在无法同时兼容基于LoRA和基于Adapter的方法的局限性。为解决这些问题，我们提出了ID-Aligner，一个通用的反馈学习框架，以增强ID-T2I性能。为解决身份特征丢失问题，我们引入了身份一致性奖励微调，利用来自人脸检测和识别模型的反馈来改进生成的身份保持。此外，我们提出了身份美学奖励微调，利用人类标注的偏好数据和自动构建的关于角色结构生成的反馈来提供美学调优信号。得益于其通用的反馈微调框架，我们的方法可以轻松应用于LoRA和Adapter模型，实现一致的性能提升。在SD1.5和SDXL扩散模型上的广泛实验验证了我们方法的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 身份一致性奖励微调
    * 身份美学奖励微调
    * 与LoRA和Adapter模型的兼容性
    * 在SD1.5和SDXL上的验证
</details>
</details>

---


<details>
<summary><b> G-Refine: A General Quality Refiner for Text-to-Image Generation</b></summary>

* **Authors:** Chunyi Li, Haoning Wu, Hongkun Hao, Zicheng Zhang, Tengchaun Kou, Chaofeng Chen, Lei Bai, Xiaohong Liu, Weisi Lin, Guangtao Zhai
* **arXiv ID:** 2404.18343
* **One-liner:** Developed G-Refine, a general image quality refiner for enhancing AI-generated images.
* **Published in:** arxiv (29 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.18343) | [[PDF]](https://arxiv.org/pdf/2404.18343) | [[Code]](https://github.com/Q-Future/Q-Refine)

> **核心创新**
> 通过模块化指示器和增强解决了感知和对齐质量缺陷。

<details>
    <summary>Abstract</summary>
    随着文本到图像（T2I）模型的发展，AI生成图像（AIGIs）的质量缺陷对其广泛应用构成了重大障碍。在感知和对齐方面，现有模型无法始终保证高质量结果。为缓解这一限制，我们引入了G-Refine，一个通用的图像质量优化器，旨在增强低质量图像而不损害高质量图像的完整性。该模型由三个相互连接的模块组成：感知质量指示器、对齐质量指示器和通用质量增强模块。基于人类视觉系统（HVS）和句法树的机制，前两个指示器可以分别识别感知和对齐缺陷，最后一个模块可以相应地应用针对性质量增强。广泛的实验显示，与替代优化方法相比，经过G-Refine处理的AIGIs在4个数据库的10多个质量指标上表现更优。这一改进显著促进了当代T2I模型的实际应用，为其更广泛采用铺平了道路。
</details>

<details>
    <summary>Key points</summary>
    * 基于HVS的感知质量指示器
    * 使用句法树的对齐质量指示器
    * 通用质量增强模块
    * 在多个质量指标上的评估
</details>
</details>

---


<details>
<summary><b> On Mechanistic Knowledge Localization in Text-to-Image Generative Models</b></summary>

* **Authors:** Samyadeep Basu, Keivan Rezaei, Priyatham Kattakinda, Ryan Rossi, Cherry Zhao, Vlad Morariu, Varun Manjunatha, Soheil Feizi
* **arXiv ID:** 2405.01008
* **One-liner:** Proposed Mechanistic Localization for efficient model editing in text-to-image models.
* **Published in:** arxiv (2 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.01008) | [[PDF]](https://arxiv.org/pdf/2405.01008) | [[Code]](https://github.com/samyadeepbasu/LocoGen)

> **核心创新**
> 将知识定位到特定UNet层以促进闭式更新和编辑。

<details>
    <summary>Abstract</summary>
    识别文本到图像模型中控制视觉属性的层可以通过闭式更新促进高效模型编辑。最近的工作利用因果追踪表明，早期Stable-Diffusion变体将知识主要限制在CLIP文本编码器的第一层，而知识在UNet中扩散。扩展这一框架，我们观察到对于最近的模型（如SD-XL、DeepFloyd），因果追踪无法精确定位局部知识，突显了模型编辑的挑战。为解决这一问题，我们引入了文本到图像模型中的机制定位概念，其中关于各种视觉属性（如“风格”、“对象”、“事实”）的知识可以被机制性地定位到UNet中的一小部分层，从而促进高效模型编辑。我们使用我们的方法LocoGen来定位知识，该方法通过在UNet的交叉注意力层进行干预来测量中间层对输出生成的直接影响。然后，我们采用LocoEdit，一种快速的闭式编辑方法，应用于流行的开源文本到图像模型（包括最新的SD-XL），并探索神经元级模型编辑的可能性。通过机制定位，我们的工作为基于定位的文本到图像模型编辑的成功和失败提供了更好的视角。
</details>

<details>
    <summary>Key points</summary>
    * 用于定位的LocoGen方法
    * 在交叉注意力层进行干预
    * 用于快速模型编辑的LocoEdit
    * 在SD-XL和其他模型上的应用
</details>
</details>

---


<details>
<summary><b> FlexEControl: Flexible and Efficient Multimodal Control for Text-to-Image Generation</b></summary>

* **Authors:** Xuehai He, Jian Zheng, Jacob Zhiyuan Fang, Robinson Piramuthu, Mohit Bansal, Vicente Ordonez, Gunnar A Sigurdsson, Nanyun Peng, Xin Eric Wang
* **arXiv ID:** 2405.04834
* **One-liner:** Designed FlexEControl, an efficient method for controllable text-to-image generation with multimodal inputs.
* **Published in:** arxiv (8 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.04834) | [[PDF]](https://arxiv.org/pdf/2405.04834) | [[Code]]()

> **核心创新**
> 通过权重分解增强了忠实度并减少了计算开销。

<details>
    <summary>Abstract</summary>
    可控文本到图像（T2I）扩散模型根据文本提示和其他模态的语义输入（如边缘图）生成图像。然而，当前的可控T2I方法通常面临效率和忠实度的挑战，尤其是在基于来自相同或不同模态的多个输入进行条件化时。在本文中，我们提出了一种新颖的灵活高效方法FlexEControl，用于可控T2I生成。FlexEControl的核心是一种独特的权重分解策略，允许对各种输入类型进行流线型整合。这种方法不仅增强了生成图像对控制的忠实度，还显著减少了通常与多模态条件化相关的计算开销。与Uni-ControlNet相比，我们的方法实现了可训练参数减少41%和内存使用减少30%。此外，它加倍了数据效率，并可以灵活地在各种模态的多个输入条件指导下生成图像。
</details>

<details>
    <summary>Key points</summary>
    * 权重分解策略
    * 各种输入的流线型整合
    * 参数和内存使用的减少
    * 与多模态的灵活性
</details>
</details>

---


<details>
<summary><b> TriLoRA: Integrating SVD for Advanced Style Personalization in Text-to-Image Generation</b></summary>

* **Authors:** Chengcheng Feng, Mu He, Qiuyu Tian, Haojie Yin, Xiaofang Zhao, Hongwei Tang, Xingqiang Wei
* **arXiv ID:** 2405.11236
* **One-liner:** Integrated SVD into LoRA for improved fine-tuning of image generation models.
* **Published in:** arxiv (18 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.11236) | [[PDF]](https://arxiv.org/pdf/2405.11236) | [[Code]]()

> **核心创新**
> 增强了模型输出的稳定性，减少了过拟合，并更好地捕捉了特征。

<details>
    <summary>Abstract</summary>
    随着深度学习技术的不断进步，图像生成模型，尤其是像Stable Diffusion这样的模型，在视觉艺术创作中的应用日益广泛。然而，这些模型经常面临过拟合、生成结果缺乏稳定性以及在微调过程中难以准确捕捉创作者期望特征的挑战。为应对这些挑战，我们提出了一种创新方法，将奇异值分解（SVD）整合到低秩适应（LoRA）参数更新策略中，旨在提高图像生成模型的微调效率和输出质量。通过在LoRA框架中整合SVD，我们的方法不仅有效降低了过拟合风险，还增强了模型输出的稳定性，并更准确地捕捉了创作者期望的细微特征调整。我们在多个数据集上评估了我们的方法，结果表明，与传统微调方法相比，我们的方法在保持生成质量的同时显著提高了模型的泛化能力和创作灵活性。此外，该方法在资源受限条件下保持了LoRA的优异性能，允许在不牺牲原始效率和资源优势的情况下显著提高图像生成质量。
</details>

<details>
    <summary>Key points</summary>
    * 在LoRA框架中整合SVD
    * 过拟合风险的降低
    * 输出稳定性的改进
    * 在约束条件下保持效率
</details>
</details>

---


<details>
<summary><b> An Empirical Study and Analysis of Text-to-Image Generation Using Large Language Model-Powered Textual Representation</b></summary>

* **Authors:** Zhiyu Tan, Mengping Yang, Luozheng Qin, Hao Yang, Ye Qian, Qiang Zhou, Cheng Zhang, Hao Li
* **arXiv ID:** 2405.12914
* **One-liner:** Investigated LLMs as text encoders for multilingual and longer-context text-to-image generation.
* **Published in:** arxiv (21 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.12914) | [[PDF]](https://arxiv.org/pdf/2405.12914) | [[Code]](https://github.com/llm-conditioned-diffusion/OmniDiffusion)

> **核心创新**
> 通过轻量级适配器实现了更优的文本表示和生成质量。

<details>
    <summary>Abstract</summary>
    忠实文本到图像生成的一个关键前提是准确理解文本输入。现有方法利用CLIP模型的文本编码器来表示输入提示。然而，预训练的CLIP模型仅能编码英语，且最大标记长度为77。此外，CLIP的文本编码器模型容量相对于大语言模型（LLMs）较为有限，后者支持多语言输入、容纳更长上下文并实现更优的文本表示。在本文中，我们研究使用LLMs作为文本编码器来改进文本到图像生成中的语言理解。不幸的是，从头开始使用LLMs训练文本到图像生成模型需要大量计算资源和数据。为此，我们引入了一个三阶段训练流程，有效且高效地将现有文本到图像模型与LLMs整合。具体来说，我们提出了一种轻量级适配器，使得能够使用来自LLMs的文本表示快速训练文本到图像模型。广泛的实验表明，我们的模型不仅支持多语言，还支持更长输入上下文，并具有更优的图像生成质量。
</details>

<details>
    <summary>Key points</summary>
    * 三阶段训练流程
    * 用于LLM整合的轻量级适配器
    * 对多语言输入的支持
    * 对更长上下文的容纳
</details>
</details>

---


<details>
<summary><b> Personalized Residuals for Concept-Driven Text-to-Image Generation</b></summary>

* **Authors:** Cusuh Ham, Matthew Fisher, James Hays, Nicholas Kolkin, Yuchen Liu, Richard Zhang, Tobias Hinz
* **arXiv ID:** 2405.12978
* **One-liner:** Introduced personalized residuals and localized sampling for efficient concept-driven generation.
* **Published in:** arxiv (21 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.12978) | [[PDF]](https://arxiv.org/pdf/2405.12978) | [[Code]](https://github.com/richzhang/webpage-template)

> **核心创新**
> 以最少的参数和计算成本实现了快速概念学习和采样。

<details>
    <summary>Abstract</summary>
    我们提出了个性化残差和局部注意力引导采样，用于使用文本到图像扩散模型进行高效概念驱动生成。我们的方法首先通过冻结预训练文本条件扩散模型的权重并学习模型一小部分层的低秩残差来表示概念。然后，残差方法直接启用了我们提出的采样技术的应用，该技术仅通过交叉注意力在概念局部化的区域应用学习到的残差，并在所有其他区域应用原始扩散权重。因此，局部采样将学习到的概念身份与底层扩散模型的现有生成先验相结合。我们表明，个性化残差在单个GPU上约3分钟内有效捕捉概念身份，无需使用正则化图像，且参数少于先前模型，而局部采样允许将原始模型用作图像大部分区域的强先验。
</details>

<details>
    <summary>Key points</summary>
    * 低秩残差学习
    * 局部注意力引导采样
    * 预训练模型权重的冻结
    * 无需正则化图像的应用
</details>
</details>

---


<details>
<summary><b> Kaleido Diffusion: Improving Conditional Diffusion Models with Autoregressive Latent Modeling</b></summary>

* **Authors:** Jiatao Gu, Ying Shen, Shuangfei Zhai, Yizhe Zhang, Navdeep Jaitly, Joshua M. Susskind
* **arXiv ID:** 2405.21048
* **One-liner:** Presented Kaleido to enhance diversity in diffusion model image generation using autoregressive latent priors.
* **Published in:** arxiv (31 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.21048) | [[PDF]](https://arxiv.org/pdf/2405.21048) | [[Code]]()

> **核心创新**
> 通过潜在变量指导拓宽了输出多样性，同时保持质量。

<details>
    <summary>Abstract</summary>
    扩散模型已成为从文本描述生成高质量图像的强大工具。尽管取得了成功，这些模型在采样图像中往往表现出有限的多样性，尤其是在使用高分类器自由引导权重进行采样时。为解决这一问题，我们提出了Kaleido，一种新颖方法，通过整合自回归潜在先验来增强样本的多样性。Kaleido整合了一个自回归语言模型，该模型编码原始标题并生成潜在变量，作为指导和促进图像生成过程的抽象和中间表示。在本文中，我们探索了多种离散潜在表示，包括文本描述、检测边界框、对象斑块和视觉标记。这些表示多样化并丰富了扩散模型的输入条件，实现了更多样化的输出。我们的实验结果表明，Kaleido有效拓宽了给定文本描述生成图像样本的多样性，同时保持高图像质量。此外，我们表明Kaleido紧密遵循生成的潜在变量提供的指导，展示了其有效控制和引导图像生成过程的能力。
</details>

<details>
    <summary>Key points</summary>
    * 自回归语言模型的整合
    * 离散潜在表示的使用
    * 输入条件的多样化
    * 对生成过程的控制
</details>
</details>

---


<details>
<summary><b> AttnDreamBooth: Towards Text-Aligned Personalized Text-to-Image Generation</b></summary>

* **Authors:** Lianyu Pang, Jian Yin, Baoquan Zhao, Feize Wu, Fu Lee Wang, Qing Li, Xudong Mao
* **arXiv ID:** 2406.05000
* **One-liner:** Introduced AttnDreamBooth to improve text-to-image personalization by addressing embedding alignment issues.
* **Published in:** arxiv (7 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.05000) | [[PDF]](https://arxiv.org/pdf/2406.05000) | [[Code]]()

> **核心创新**
> 在不同训练阶段分别学习嵌入对齐、注意力图和主体身份，并采用交叉注意力正则化。

<details>
    <summary>Abstract</summary>
    文本到图像模型的最新进展使得用户提供的概念能够通过灵活的文本控制实现高质量的个性化图像合成。本文分析了文本到图像个性化中两种主要技术——Textual Inversion 和 DreamBooth 的局限性。在将学习到的概念集成到新提示时，Textual Inversion 倾向于过拟合概念，而 DreamBooth 常常忽略它。我们将这些问题归因于概念嵌入对齐的错误学习。我们提出了 AttnDreamBooth，一种新颖的方法，通过在不同训练阶段分别学习嵌入对齐、注意力图和主体身份来解决这些问题。我们还引入了交叉注意力图正则化项来增强注意力图的学习。与基线方法相比，我们的方法在身份保持和文本对齐方面显示出显著改进。
</details>

<details>
    <summary>Key points</summary>
    * 分析了 Textual Inversion 和 DreamBooth 的局限性
    * 将问题归因于错误的嵌入对齐
    * 提出了多阶段训练以分离组件
    * 引入了交叉注意力图正则化
</details>
</details>

---


<details>
<summary><b> Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance</b></summary>

* **Authors:** Kuan Heng Lin, Sicheng Mo, Ben Klingher, Fangzhou Mu, Bolei Zhou
* **arXiv ID:** 2406.07540
* **One-liner:** Developed Ctrl-X for efficient structure and appearance control in text-to-image generation without training.
* **Published in:** arxiv (11 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.07540) | [[PDF]](https://arxiv.org/pdf/2406.07540) | [[Code]](https://github.com/genforce/ctrl-x)

> **核心创新**
> 实现了前馈结构对齐和语义感知的外观迁移，以提供即插即用功能。

<details>
    <summary>Abstract</summary>
    最近的可控生成方法，如 FreeControl 和 Diffusion Self-Guidance，为文本到图像（T2I）扩散模型带来了细粒度的空间和外观控制，而无需训练辅助模块。然而，这些方法为每种类型的得分函数优化潜在嵌入，使用更长的扩散步骤，使得生成过程耗时，并限制了其灵活性和使用。本文提出了 Ctrl-X，一个简单的框架，用于 T2I 扩散控制结构和外观，无需额外训练或引导。Ctrl-X 设计了前馈结构控制，以实现与结构图像的结构对齐，以及语义感知的外观迁移，以促进从用户输入图像的外观迁移。广泛的定性和定量实验展示了 Ctrl-X 在各种条件输入和模型检查点上的优越性能。特别是，Ctrl-X 支持任意模态的条件图像的新颖结构和外观控制，与现有工作相比，展现出卓越的图像质量和外观迁移，并为任何 T2I 和文本到视频（T2V）扩散模型提供即插即用功能。
</details>

<details>
    <summary>Key points</summary>
    * 设计了前馈结构控制
    * 实现了语义感知的外观迁移
    * 支持任意条件图像
    * 为 T2I 和 T2V 模型提供即插即用功能
</details>
</details>

---


<details>
<summary><b> Commonsense-T2I Challenge: Can Text-to-Image Generation Models Understand Commonsense?</b></summary>

* **Authors:** Xingyu Fu, Muyu He, Yujie Lu, William Yang Wang, Dan Roth
* **arXiv ID:** 2406.07546
* **One-liner:** Created Commonsense-T2I benchmark to evaluate text-to-image models on commonsense reasoning.
* **Published in:** arxiv (11 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.07546) | [[PDF]](https://arxiv.org/pdf/2406.07546) | [[Code]](https://github.com/zeyofu/Commonsense-T2I)

> **核心创新**
> 提供了带有细粒度标注的对抗性文本提示，以评估模型与真实场景的对齐。

<details>
    <summary>Abstract</summary>
    我们提出了一个新任务和基准，用于评估文本到图像（T2I）生成模型生成与现实生活中常识对齐的图像的能力，我们称之为 Commonsense-T2I。给定两个包含相同动作词集合但细微差异的对抗性文本提示，例如“没有电的灯泡”与“有电的灯泡”，我们评估 T2I 模型是否能够进行视觉常识推理，例如相应地生成符合“灯泡未亮”与“灯泡亮”的图像。Commonsense-T2I 提出了一个对抗性挑战，提供了成对文本提示以及预期输出。该数据集由专家精心手工策划，并标注了细粒度标签，如常识类型和预期输出的可能性，以帮助分析模型行为。我们对各种最先进的 T2I 模型进行了基准测试，并惊讶地发现，图像合成与现实生活照片之间仍存在巨大差距——即使是 DALL-E 3 模型在 Commonsense-T2I 上也只能达到 48.92%，而稳定扩散 XL 模型仅达到 24.92% 的准确率。我们的实验表明，GPT 增强的提示无法解决这一挑战，我们还包括了对这种缺陷可能原因的详细分析。我们旨在让 Commonsense-T2I 作为 T2I 常识检查的高质量评估基准，促进现实生活图像生成的进步。
</details>

<details>
    <summary>Key points</summary>
    * 定义了对抗性成对文本提示
    * 手工策划了带有专家标注的数据集
    * 对最先进的 T2I 模型进行了基准测试
    * 分析了常识推理的差距
</details>
</details>

---


<details>
<summary><b> Improving Compositional Attribute Binding in Text-to-Image Generative Models via Enhanced Text Embeddings</b></summary>

* **Authors:** Arman Zarei, Keivan Rezaei, Samyadeep Basu, Mehrdad Saberi, Mazda Moayeri, Priyatham Kattakinda, Soheil Feizi
* **arXiv ID:** 2406.07844
* **One-liner:** Identified and addressed compositional attribute binding failures in text-to-image models.
* **Published in:** arxiv (12 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.07844) | [[PDF]](https://arxiv.org/pdf/2406.07844) | [[Code]]()

> **核心创新**
> 对 CLIP 表示空间的线性投影进行微调，以改进属性-对象关联而不损害 FID。

<details>
    <summary>Abstract</summary>
    基于扩散的文本到图像生成模型具有生成逼真图像的能力，并在挑战性图像生成基准上达到了最先进的低 FID 分数。然而，这些文本到图像生成模型的主要失败模式之一是将属性、对象及其关联关系准确组合到图像中。在本文中，我们研究了组合属性绑定失败，其中模型未能正确将描述性属性（如颜色、形状或纹理）与生成图像中的相应对象关联，并强调使用 CLIP 文本编码器的不完美文本调节是这些模型无法生成高保真组合场景的主要原因之一。具体来说，我们表明（i）存在一个最优的文本嵌入空间，可以生成高度一致的组合场景，表明 CLIP 文本编码器的输出空间是次优的，以及（ii）CLIP 中的最终令牌嵌入是错误的，因为它们常常包含组合提示中不相关令牌的注意力贡献。我们的主要发现表明，通过在 Stable-Diffusion 变体中使用少量组合图像-文本对，仅对 CLIP 表示空间进行简单且参数高效的线性投影微调，即可实现显著的组合改进（而不损害模型的 FID 分数）。
</details>

<details>
    <summary>Key points</summary>
    * 研究了组合属性绑定失败
    * 显示了 CLIP 文本编码器输出的次优性
    * 提出了微调线性投影
    * 使用了少量组合图像-文本对
</details>
</details>

---


<details>
<summary><b> FairCoT: Enhancing Fairness in Text-to-Image Generation via Chain of Thought Reasoning with Multimodal Large Language Models</b></summary>

* **Authors:** Zahraa Al Sahili, Ioannis Patras, Matthew Purver
* **arXiv ID:** 2406.09070
* **One-liner:** Introduced FairCoT to enhance fairness in text-to-image models using Chain of Thought reasoning.
* **Published in:** arxiv (13 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.09070) | [[PDF]](https://arxiv.org/pdf/2406.09070) | [[Code]]()

> **核心创新**
> 采用迭代 CoT 精炼来减轻偏见并调整提示以实现多样表示。

<details>
    <summary>Abstract</summary>
    在文本到图像生成模型领域，训练数据集中固有的偏见常常传播到生成内容中，带来显著的伦理挑战，特别是在社会敏感情境中。我们引入了 FairCoT，一个新颖的框架，通过在多模态生成大语言模型中使用思维链（CoT）推理来增强文本到图像模型的公平性。FairCoT 采用迭代 CoT 精炼来系统性地减轻偏见，并动态调整文本提示，确保生成图像中的多样性和公平表示。通过整合迭代推理过程，FairCoT 解决了零样本 CoT 在敏感情境中的局限性，平衡了创造力与伦理责任。在包括 DALLE 和各种稳定扩散变体在内的流行文本到图像系统上的实验评估表明，FairCoT 显著增强了公平性和多样性，而不牺牲图像质量或语义保真度。通过结合稳健推理、轻量级部署和可扩展到多个模型，FairCoT 代表了迈向更社会责任和透明 AI 驱动内容生成的有希望的一步。
</details>

<details>
    <summary>Key points</summary>
    * 整合了迭代思维链推理
    * 动态调整了文本提示
    * 解决了敏感情境中的偏见
    * 平衡了创造力与伦理责任
</details>
</details>

---


<details>
<summary><b> STAR: Scale-wise Text-conditioned AutoRegressive image generation</b></summary>

* **Authors:** Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Biye Li, Huaian Chen, Yi Jin
* **arXiv ID:** 2406.10797
* **One-liner:** Developed STAR, a scale-wise auto-regressive model for high-resolution text-to-image generation.
* **Published in:** arxiv (16 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.10797) | [[PDF]](https://arxiv.org/pdf/2406.10797) | [[Code]](https://github.com/krennic999/STAR)

> **核心创新**
> 实现了文本驱动的 1024x1024 图像合成，采用稳定采样和归一化 RoPE 以确保结构一致性。

<details>
    <summary>Abstract</summary>
    我们引入了 STAR，一个采用尺度自回归范式的文本到图像模型。与 VAR 局限于类条件合成图像至 256×256 不同，STAR 通过三个关键设计实现了文本驱动的图像生成至 1024×1024。首先，我们引入了一个预训练文本编码器来提取和采用文本约束的表示，增强细节和泛化能力。其次，考虑到不同尺度之间固有的结构相关性，我们利用 2D 旋转位置编码（RoPE）并将其调整为归一化版本，确保跨令牌映射的相对位置一致解释，并稳定训练过程。第三，我们观察到在单一尺度内同时采样所有令牌会破坏令牌间关系，导致结构不稳定，特别是在高分辨率生成中。为了解决这个问题，我们提出了一种新颖的稳定采样方法，将因果关系纳入采样过程，确保丰富的细节和稳定的结构。与之前的扩散模型和自回归模型相比，STAR 在保真度、文本-图像一致性和美学质量上超越了现有基准，在 A100 上仅需 2.21 秒生成 1024×1024 图像。这突显了自回归方法在高质量图像合成中的潜力，为文本到图像生成提供了新方向。
</details>

<details>
    <summary>Key points</summary>
    * 引入了预训练文本编码器
    * 应用了归一化 2D 旋转位置编码
    * 提出了带有因果关系的稳定采样
    * 在 A100 GPU 上实现了快速生成
</details>
</details>

---


<details>
<summary><b> AITTI: Learning Adaptive Inclusive Token for Text-to-Image Generation</b></summary>

* **Authors:** Xinyu Hou, Xiaoming Li, Chen Change Loy
* **arXiv ID:** 2406.12805
* **One-liner:** Proposed adaptive inclusive tokens to mitigate stereotypical biases in text-to-image generation.
* **Published in:** arxiv (18 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.12805) | [[PDF]](https://arxiv.org/pdf/2406.12805) | [[Code]](https://github.com/itsmag11/AITTI)

> **核心创新**
> 使用轻量级自适应映射网络定制令牌，无需属性指定。

<details>
    <summary>Abstract</summary>
    尽管文本到图像生成具有高质量结果，但在其生成内容中发现了刻板印象偏见，损害了生成模型的公平性。在这项工作中，我们提出学习自适应包容性令牌来转移最终生成输出的属性分布。与现有的去偏见方法不同，我们的方法既不需要显式属性指定，也不需要偏见分布的先前知识。具体来说，我们方法的核心是一个轻量级自适应映射网络，它可以为需要去偏见的概念定制包容性令牌，使令牌能够泛化到未见概念，无论其原始偏见分布如何。这是通过使用锚定损失对自适应映射网络进行少量平衡和包容性样本调优来实现的。实验结果表明，我们的方法在不指定属性的情况下优于先前的偏见缓解方法，同时保持生成结果与文本描述之间的对齐。此外，我们的方法达到了与需要特定属性或编辑方向进行生成的模型相当的性能。广泛实验展示了我们的自适应包容性令牌在减轻文本到图像生成中刻板印象偏见方面的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 设计了自适应映射网络
    * 使用锚定损失对平衡样本进行调优
    * 泛化到未见概念
    * 在减少偏见的同时保持文本对齐
</details>
</details>

---


<details>
<summary><b> Beyond Thumbs Up/Down: Untangling Challenges of Fine-Grained Feedback for Text-to-Image Generation</b></summary>

* **Authors:** Katherine M. Collins, Najoung Kim, Yonatan Bitton, Verena Rieser, Shayegan Omidshafiei, Yushi Hu, Sherol Chen, Senjuti Dutta, Minsuk Chang, Kimin Lee, Youwei Liang, Georgina Evans, Sahil Singla, Gang Li, Adrian Weller, Junfeng He, Deepak Ramachandran, Krishnamurthy Dj Dvijotham
* **arXiv ID:** 2406.16807
* **One-liner:** Investigated the effectiveness of fine-grained vs. coarse-grained human feedback for text-to-image reward models.
* **Published in:** arxiv (24 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.16807) | [[PDF]](https://arxiv.org/pdf/2406.16807) | [[Code]]()

> **核心创新**
> 分析了反馈类型的复杂性及其在各种设置下对模型准确性的影响。

<details>
    <summary>Abstract</summary>
    人类反馈在学习文本到图像生成的奖励模型中起着关键作用，但反馈应采取的优化形式尚未明确确立。本文研究了细粒度反馈（捕捉图像质量和提示对齐的细微区别）与传统粗粒度反馈（例如，点赞/点踩或选项间排名）的有效性。虽然细粒度反馈具有潜力，特别是对于迎合多样化社会偏好的系统，但我们表明证明其优于粗粒度反馈并非自动。通过在真实和合成偏好数据上的实验，我们揭示了由于模型选择、反馈类型以及人类判断与计算解释之间对齐的相互作用，构建有效模型的复杂性。我们识别了在引出和利用细粒度反馈中的关键挑战，促使重新评估其假设的好处和实用性。我们的发现——例如，在固定预算下，细粒度反馈在某些设置中可能导致更差的模型；然而，在具有已知属性的受控设置中，细粒度奖励确实更有帮助——呼吁仔细考虑反馈属性，并可能呼唤新颖的建模方法来适当解锁细粒度反馈在野外的潜在价值。
</details>

<details>
    <summary>Key points</summary>
    * 比较了细粒度和粗粒度反馈
    * 识别了反馈引出中的挑战
    * 显示了取决于设置的混合结果
    * 呼吁新颖的建模方法
</details>
</details>

---


<details>
<summary><b> MUMU: Bootstrapping Multimodal Image Generation from Text-to-Image Data</b></summary>

* **Authors:** William Berman, Alexander Peysakhovich
* **arXiv ID:** 2406.18790
* **One-liner:** Trained MUMU model for image generation from multimodal prompts with interleaved text and images.
* **Published in:** arxiv (26 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.18790) | [[PDF]](https://arxiv.org/pdf/2406.18790) | [[Code]]()

> **核心创新**
> 实现了来自不同图像输入的连贯组合，用于风格迁移和角色一致性等任务。

<details>
    <summary>Abstract</summary>
    我们训练了一个模型，从交错文本和图像的多模态提示生成图像，例如“一个<男人图片>男人和他的<狗图片>狗在<卡通图片>动画风格中。”我们通过从合成生成和公开可用的文本-图像数据中提取与图像标题中词语对应的语义有意义的图像裁剪来引导一个多模态数据集。我们的模型 MUMU 由一个视觉-语言模型编码器和一个扩散解码器组成，并在单个 8xH100 GPU 节点上训练。尽管仅在同一图像的裁剪上训练，MUMU 学会了将来自不同图像的输入组合成连贯的输出。例如，一个现实人物和卡通的输入将输出同一人物的卡通风格，一个站立主体和滑板车的输入将输出主体骑滑板车。因此，我们的模型泛化到风格迁移和角色一致性等任务。我们的结果显示了使用多模态模型作为图像生成的通用控制器的前景。
</details>

<details>
    <summary>Key points</summary>
    * 从图像裁剪引导了多模态数据集
    * 使用了视觉-语言编码器与扩散解码器
    * 学会了组合来自不同图像的输入
    * 泛化到风格迁移和一致性任务
</details>
</details>

---


<details>
<summary><b> AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation</b></summary>

* **Authors:** Yanan Sun, Yanchen Liu, Yinhao Tang, Wenjie Pei, Kai Chen
* **arXiv ID:** 2406.18958
* **One-liner:** Proposed AnyControl for multi-control image synthesis with arbitrary combinations of control signals.
* **Published in:** arxiv (27 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.18958) | [[PDF]](https://arxiv.org/pdf/2406.18958) | [[Code]](https://github.com/open-mmlab/AnyControl)

> **核心创新**
> 开发了多控制编码器以提取统一嵌入，处理多样化输入并保持语义对齐。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）生成领域近年来取得了显著进展，主要由扩散模型的进步驱动。语言控制实现了有效的内容创作，但在图像生成的细粒度控制方面存在困难。这一挑战在很大程度上通过将额外的用户提供的空间条件（如深度图和边缘图）通过额外编码整合到预训练 T2I 模型中来探索。然而，多控制图像合成仍面临几个挑战。具体来说，当前方法在处理多样化输入控制信号的自由组合方面有限，忽视了多个空间条件之间的复杂关系，并且常常无法保持与提供文本提示的语义对齐。这可能导致次优的用户体验。为了解决这些挑战，我们提出了 AnyControl，一个多控制图像合成框架，支持多样化控制信号的任意组合。AnyControl 开发了一个新颖的多控制编码器，提取统一的多模态嵌入来指导生成过程。这种方法使用户输入有整体理解，并在多样化控制信号下产生高质量、忠实的结果，如广泛的定量和定性评估所示。
</details>

<details>
    <summary>Key points</summary>
    * 引入了多控制编码器
    * 支持任意控制信号组合
    * 解决了空间条件间的关系
    * 确保了与文本提示的语义对齐
</details>
</details>

---


<details>
<summary><b> PopAlign: Population-Level Alignment for Fair Text-to-Image Generation</b></summary>

* **Authors:** Shufan Li, Harkanwar Singh, Aditya Grover
* **arXiv ID:** 2406.19668
* **One-liner:** Introduced PopAlign for population-level bias mitigation in T2I models.
* **Published in:** arxiv (28 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.19668) | [[PDF]](https://arxiv.org/pdf/2406.19668) | [[Code]](https://github.com/jacklishufan/PopAlignSDXL)

> **核心创新**
> 开发了一种新颖的优化方法，在群体层面解决偏见，不同于现有的成对偏好方法。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）模型通过在大型数据集上的广泛训练实现了高保真度生成。然而，这些模型可能会无意中吸收训练数据中的不良偏见，例如在性别或种族中性提示下对特定身份的过度代表。现有的对齐方法，如基于人类反馈的强化学习（RLHF）和直接偏好优化（DPO），无法有效解决这一问题，因为它们基于由单个样本组成的成对偏好操作，而上述偏见只能在群体层面衡量。例如，对于提示“医生”的单个样本可能是男性或女性，但模型在重复采样时主要生成男性医生则反映了性别偏见。为了解决这一局限，我们引入了PopAlign，一种新颖的群体级偏好优化方法，而标准优化会偏好整个样本集而非其他。我们进一步推导了一个随机下界，直接优化来自偏好群体的单个样本，以实现可扩展训练。通过人类评估和标准图像质量与偏见指标，我们表明PopAlign显著减轻了预训练T2I模型的偏见，同时很大程度上保持了生成质量。代码可在<a href="https://github.com/jacklishufan/PopAlignSDXL" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 群体级偏好优化
    * 用于可扩展训练的随机下界
    * 减轻偏见同时保持生成质量
</details>
</details>

---


<details>
<summary><b> Prompt Refinement with Image Pivot for Text-to-Image Generation</b></summary>

* **Authors:** Jingtao Zhan, Qingyao Ai, Yiqun Liu, Yingwei Pan, Ting Yao, Jiaxin Mao, Shaoping Ma, Tao Mei
* **arXiv ID:** 2407.00247
* **One-liner:** Proposed PRIP for zero-shot prompt refinement using image pivots.
* **Published in:** arxiv (28 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.00247) | [[PDF]](https://arxiv.org/pdf/2407.00247) | [[Code]]()

> **核心创新**
> 创新了一种方法，通过利用图像表示作为中介，将用户提示翻译成系统语言。

<details>
    <summary>Abstract</summary>
    对于文本到图像生成，自动将用户提供的自然语言提示精炼为系统偏好的关键词丰富提示对用户体验至关重要。这种提示精炼过程类似于将提示从“用户语言”翻译成“系统语言”。然而，这种平行语料的稀缺使得训练提示精炼模型变得困难。受零样本机器翻译技术的启发，我们引入了基于图像枢轴的提示精炼（PRIP）。PRIP创新性地使用用户偏好图像的潜在表示作为用户语言和系统语言之间的中间“枢轴”。它将精炼过程分解为两个数据丰富的任务：从用户语言推断用户偏好图像的表示，然后将图像表示翻译成系统语言。因此，它可以利用丰富的数据进行训练。广泛的实验表明，PRIP显著优于一系列基线，并以零样本方式有效迁移到未见系统。
</details>

<details>
    <summary>Key points</summary>
    * 分解为推断图像表示和翻译成系统语言
    * 使用零样本机器翻译技术
    * 有效迁移到未见系统
</details>
</details>

---


<details>
<summary><b> Efficient Personalized Text-to-image Generation by Leveraging Textual Subspace</b></summary>

* **Authors:** Shian Du, Xiaotian Cheng, Qi Qian, Henglu Wei, Yi Xu, Xiangyang Ji
* **arXiv ID:** 2407.00608
* **One-liner:** Efficient personalized T2I generation by optimizing in a textual subspace.
* **Published in:** arxiv (30 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.00608) | [[PDF]](https://arxiv.org/pdf/2407.00608) | [[Code]]()

> **核心创新**
> 解决了嵌入优化中的低效问题，并改进了与新颖提示的对齐。

<details>
    <summary>Abstract</summary>
    个性化文本到图像生成在近些年吸引了前所未有的关注，因为它能够通过输入概念数据集和新颖文本提示生成高度个性化的图像。然而，先前的方法仅专注于重建任务的性能，降低了其与不同文本提示结合的能力。此外，在高维嵌入空间中进行优化通常导致不必要的耗时训练过程和缓慢收敛。为了解决这些问题，我们提出了一种高效的方法，在文本子空间中探索目标嵌入，灵感来自自表达性属性。此外，我们提出了一种高效的基向量选择策略，用于确定文本子空间的基向量。实验评估表明，学习到的嵌入不仅能够忠实重建输入图像，还显著提高了其与新颖输入文本提示的对齐。此外，我们观察到在文本子空间中进行优化显著提高了对初始词的鲁棒性，放宽了要求用户输入最相关初始词的约束。我们的方法为个性化文本到图像生成开启了更高效表示学习的大门。
</details>

<details>
    <summary>Key points</summary>
    * 在文本子空间中探索目标嵌入
    * 基向量选择策略
    * 增强对初始词输入的鲁棒性
</details>
</details>

---


<details>
<summary><b> LLM4GEN: Leveraging Semantic Representation of LLMs for Text-to-Image Generation</b></summary>

* **Authors:** Mushui Liu, Yuhang Ma, Yang Zhen, Jun Dan, Yunlong Yu, Zeng Zhao, Zhipeng Hu, Bai Liu, Changjie Fan
* **arXiv ID:** 2407.00737
* **One-liner:** Enhanced T2I diffusion models with LLM features via LLM4GEN framework.
* **Published in:** arxiv (30 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.00737) | [[PDF]](https://arxiv.org/pdf/2407.00737) | [[Code]](https://github.com/YUHANG-Ma/LLM4GEN)

> **核心创新**
> 在复杂提示场景中使用LLMs改进了语义理解和对齐。

<details>
    <summary>Abstract</summary>
    扩散模型在文本到图像生成中取得了显著成功。然而，在处理涉及多个对象、属性绑定和长描述的复杂密集提示时，它们经常遇到挑战。在本文中，我们提出了一个名为\textbf{LLM4GEN}的新颖框架，通过利用大型语言模型（LLMs）的表示来增强文本到图像扩散模型的语义理解。它可以作为即插即用组件无缝集成到各种扩散模型中。一个特别设计的交叉适配模块（CAM）将文本到图像模型的原始文本特征与LLM特征集成，从而增强文本到图像生成。此外，为了促进和纠正文本提示中的实体-属性关系，我们开发了一个实体引导正则化损失，以进一步提高生成性能。我们还引入了DensePrompts，包含$7,000$个密集提示，为文本到图像生成任务提供全面评估。实验表明，LLM4GEN显著提高了SD1.5和SDXL的语义对齐，在T2I-CompBench上的颜色指标分别提高了9.69\%和12.90\%。此外，它在样本质量、图像-文本对齐和人类评估方面超越了现有模型。
</details>

<details>
    <summary>Key points</summary>
    * 使用交叉适配模块集成LLM特征
    * 实体引导正则化损失
    * 引入DensePrompts数据集进行评估
</details>
</details>

---


<details>
<summary><b> InstantStyle-Plus: Style Transfer with Content-Preserving in Text-to-Image Generation</b></summary>

* **Authors:** Haofan Wang, Peng Xing, Renyuan Huang, Hao Ai, Qixun Wang, Xu Bai
* **arXiv ID:** 2407.00788
* **One-liner:** Introduced InstantStyle-Plus for balanced style transfer in diffusion models.
* **Published in:** arxiv (30 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.00788) | [[PDF]](https://arxiv.org/pdf/2407.00788) | [[Code]](https://github.com/instantX-research/InstantStyle-Plus)

> **核心创新**
> 通过模块化组件实现了风格的无缝集成，同时保持内容完整性。

<details>
    <summary>Abstract</summary>
    风格迁移是一种创造性过程，旨在创建保持原始图像本质同时拥抱另一图像视觉风格的图像。尽管扩散模型在个性化主题驱动或风格驱动应用中展示了令人印象深刻的生成能力，现有最先进方法在实现内容保存和风格增强之间的无缝平衡方面仍然遇到困难。例如，放大风格影响往往会破坏内容的结构完整性。为了解决这些挑战，我们将风格迁移任务解构为三个核心元素：1）风格，关注图像的美学特征；2）空间结构，涉及视觉元素的几何排列和构图；3）语义内容，捕捉图像的概念意义。在这些原则指导下，我们引入了InstantStyle-Plus，一种优先考虑原始内容完整性同时无缝集成目标风格的方法。具体来说，我们的方法通过高效、轻量级过程实现风格注入，利用前沿的InstantStyle框架。为了加强内容保存，我们使用反转内容潜在噪声和多功能即插即用平铺ControlNet来启动过程，以保留原始图像的内在布局。我们还集成了一个全局语义适配器来增强语义内容的保真度。为了防止风格信息稀释，使用风格提取器作为判别器提供补充风格指导。代码将在<a href="https://github.com/instantX-research/InstantStyle-Plus" rel="external noopener nofollow" class="link-external link-https">此https URL</a>提供。
</details>

<details>
    <summary>Key points</summary>
    * 解构为风格、空间结构和语义内容
    * 使用反转内容潜在噪声和平铺ControlNet
    * 全局语义适配器和风格提取器用于指导
</details>
</details>

---


<details>
<summary><b> JeDi: Joint-Image Diffusion Models for Finetuning-Free Personalized Text-to-Image Generation</b></summary>

* **Authors:** Yu Zeng, Vishal M. Patel, Haochen Wang, Xun Huang, Ting-Chun Wang, Ming-Yu Liu, Yogesh Balaji
* **arXiv ID:** 2407.06187
* **One-liner:** Developed JEDI for finetuning-free personalized T2I generation.
* **Published in:** arxiv (8 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.06187) | [[PDF]](https://arxiv.org/pdf/2407.06187) | [[Code]]()

> **核心创新**
> 通过学习文本-图像对的联合分布，实现了无需优化的快速个性化。

<details>
    <summary>Abstract</summary>
    个性化文本到图像生成模型使用户能够在多样场景中描绘其个人物品，应用于各种领域。为实现个性化能力，现有方法依赖于在用户自定义数据集上微调文本到图像基础模型，这对普通用户来说可能非平凡、资源密集且耗时。尽管尝试开发免微调方法，但它们的生成质量远低于微调对应方法。在本文中，我们提出了联合图像扩散（\jedi），一种学习免微调个性化模型的有效技术。我们的关键思想是学习共享共同主题的多个相关文本-图像对的联合分布。为了促进学习，我们提出了一种可扩展的合成数据集生成技术。一旦训练完成，我们的模型通过在采样过程中简单使用参考图像作为输入，实现快速简便的个性化。我们的方法不需要任何昂贵的优化过程或额外模块，并且能够忠实保留由任意数量参考图像表示的身份。实验结果表明，我们的模型在定量和定性上均实现了最先进的生成质量，显著优于先前基于微调和免微调的个性化基线。
</details>

<details>
    <summary>Key points</summary>
    * 学习相关文本-图像对的联合分布
    * 可扩展合成数据集生成
    * 在采样过程中使用参考图像输入
</details>
</details>

---


<details>
<summary><b> Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning</b></summary>

* **Authors:** Fanyue Wei, Wei Zeng, Zhenyang Li, Dawei Yin, Lixin Duan, Wen Li
* **arXiv ID:** 2407.06642
* **One-liner:** Applied reinforcement learning with DPG for improved personalized T2I generation.
* **Published in:** arxiv (9 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.06642) | [[PDF]](https://arxiv.org/pdf/2407.06642) | [[Code]](https://github.com/wfanyue/DPG-T2I-Personalization)

> **核心创新**
> 通过灵活目标整合增强了结构一致性和视觉保真度。

<details>
    <summary>Abstract</summary>
    个性化文本到图像模型允许用户为对象（由一组参考图像指定）生成各种风格的图像（由句子指定）。尽管使用基于扩散的生成模型取得了显著成果，但在扩散过程中，对象的视觉结构和细节经常意外改变。一个主要原因是这些基于扩散的方法在训练中通常采用简单的重建目标，难以强制执行生成图像与参考图像之间的适当结构一致性。为此，在本文中，我们设计了一个新颖的强化学习框架，利用确定性策略梯度方法进行个性化文本到图像生成，通过该框架可以轻松整合各种目标，无论是可微分还是不可微分，以监督扩散模型提高生成图像的质量。在个性化文本到图像生成基准数据集上的实验结果表明，我们提出的方法在视觉保真度上大幅优于现有最先进方法，同时保持文本对齐。我们的代码可在：\url{<a href="https://github.com/wfanyue/DPG-T2I-Personalization" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}获取。
</details>

<details>
    <summary>Key points</summary>
    * 利用确定性策略梯度方法
    * 整合各种目标
    * 改进文本对齐和视觉保真度
</details>
</details>

---


<details>
<summary><b> MARS: Mixture of Auto-Regressive Models for Fine-grained Text-to-image Synthesis</b></summary>

* **Authors:** Wanggui He, Siming Fu, Mushui Liu, Xierui Wang, Wenyi Xiao, Fangxun Shu, Yi Wang, Lei Zhang, Zhelun Yu, Haoyuan Li, Ziwei Huang, LeiLei Gan, Hao Jiang
* **arXiv ID:** 2407.07614
* **One-liner:** Introduced MARS for efficient bilingual T2I generation using LLMs.
* **Published in:** arxiv (10 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.07614) | [[PDF]](https://arxiv.org/pdf/2407.07614) | [[Code]](https://github.com/fusiming3/MARS)

> **核心创新**
> 在多阶段训练策略中结合语言和视觉处理，实现高效率。

<details>
    <summary>Abstract</summary>
    自回归模型在语言生成领域取得了显著进展，但在图像合成领域表现不如扩散模型。在这项工作中，我们引入了MARS，一个用于T2I生成的新颖框架，集成了一个特别设计的语义视觉语言集成专家（SemVIE）。这个创新组件通过独立处理语言和视觉信息来集成预训练LLMs，冻结文本组件同时微调视觉组件。这种方法保留了LLMs的NLP能力，同时赋予它们卓越的视觉理解能力。基于预训练Qwen-7B的强大基础，MARS以其对应英语和中文提示的双语生成能力以及联合图像和文本生成能力脱颖而出。该框架的灵活性使其能够迁移到任意到任意任务适应性。此外，MARS采用多阶段训练策略，首先通过互补双向任务建立鲁棒的图像-文本对齐，随后专注于精炼T2I生成过程，显著增强文本-图像同步性和图像细节的粒度。值得注意的是，MARS仅需要SD1.5所需GPU天数的9%，但在各种基准测试中取得了显著成果，展示了训练效率和快速部署在各种应用中的潜力。
</details>

<details>
    <summary>Key points</summary>
    * 语义视觉语言集成专家（SemVIE）
    * 用于图像-文本对齐的多阶段训练
    * 双语能力和减少的GPU使用
</details>
</details>

---


<details>
<summary><b> Addressing Image Hallucination in Text-to-Image Generation through Factual Image Retrieval</b></summary>

* **Authors:** Youngsun Lim, Hyunjung Shim
* **arXiv ID:** 2407.10683
* **One-liner:** Addressed image hallucination in T2I models using factual image retrieval.
* **Published in:** arxiv (15 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.10683) | [[PDF]](https://arxiv.org/pdf/2407.10683) | [[Code]]()

> **核心创新**
> 提出了一种方法，通过利用外部来源和编辑工具生成事实一致的图像。

<details>
    <summary>Abstract</summary>
    文本到图像生成随着扩散模型的出现显示出显著进展。然而，这些模型经常生成事实不一致的图像，未能准确反映输入文本提示传达的事实信息和常识。我们将此问题称为图像幻觉。借鉴语言模型中幻觉的研究，我们将此问题分为三种类型，并提出一种方法，使用从外部来源检索的事实图像生成真实图像。根据幻觉的性质，我们使用现成的图像编辑工具，如InstructPix2Pix或IP-Adapter，来利用检索图像中的事实信息。这种方法能够生成准确反映事实和常识的图像。
</details>

<details>
    <summary>Key points</summary>
    * 幻觉类型的分类
    * 使用检索的事实图像与编辑工具
    * 确保事实和常识的准确性
</details>
</details>

---


<details>
<summary><b> Subject-driven Text-to-Image Generation via Preference-based Reinforcement Learning</b></summary>

* **Authors:** Yanting Miao, William Loh, Suraj Kothawade, Pascal Poupart, Abdullah Rashwan, Yeqing Li
* **arXiv ID:** 2407.12164
* **One-liner:** Proposed RPO with λ-Harmonic reward for efficient subject-driven T2I generation.
* **Published in:** arxiv (16 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.12164) | [[PDF]](https://arxiv.org/pdf/2407.12164) | [[Code]](https://github.com/andrew-miao/RPO)

> **核心创新**
> 简化了训练设置并改进了正则化，以实现更好的文本-图像对齐。

<details>
    <summary>Abstract</summary>
    文本到图像生成模型最近引起了相当大的兴趣，能够从文本提示合成高质量图像。然而，这些模型通常缺乏从给定参考图像生成特定主题或在变化条件下合成新颖呈现的能力。像DreamBooth和主题驱动文本到图像（SuTI）等方法在这一领域取得了显著进展。然而，这两种方法主要专注于增强与参考图像的相似性，并且需要昂贵的设置，经常忽视高效训练的需求和避免对参考图像的过拟合。在这项工作中，我们提出了$\lambda$-Harmonic奖励函数，它提供可靠的奖励信号，并允许早期停止以实现更快训练和有效正则化。通过结合Bradley-Terry偏好模型，$\lambda$-Harmonic奖励函数还为主题驱动生成任务提供偏好标签。我们提出了奖励偏好优化（RPO），它提供了更简单的设置（仅需要DreamBooth使用的负样本的$3\%）和更少的梯度步数进行微调。与大多数现有方法不同，我们的方法不需要训练文本编码器或优化文本嵌入，并且通过仅微调U-Net组件实现文本-图像对齐。经验上，$\lambda$-Harmonic被证明是主题驱动生成任务中模型选择的可靠方法。基于$\lambda$-Harmonic奖励函数的偏好标签和早期停止验证，我们的算法在DreamBench上实现了最先进的CLIP-I分数0.833和CLIP-T分数0.314。
</details>

<details>
    <summary>Key points</summary>
    * λ-Harmonic奖励函数用于可靠信号
    * 奖励偏好优化（RPO）与早期停止
    * 仅微调U-Net组件以提高效率
</details>
</details>

---


<details>
<summary><b> GreenStableYolo: Optimizing Inference Time and Image Quality of Text-to-Image Generation</b></summary>

* **Authors:** Jingzhi Gong, Sisi Li, Giordano d&#39;Aloisio, Zishuo Ding, Yulong Ye, William B. Langdon, Federica Sarro
* **arXiv ID:** 2407.14982
* **One-liner:** Improved text-to-image generation efficiency and quality with multi-objective optimization.
* **Published in:** arxiv (20 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.14982) | [[PDF]](https://arxiv.org/pdf/2407.14982) | [[Code]]()

> **核心创新**
> GreenStableYolo使用NSGA-II和Yolo优化Stable Diffusion的参数和提示，将GPU推理时间减少266%，超体积提高526%，同时图像质量略有折衷。

<details>
    <summary>Abstract</summary>
    调整参数和提示以改进基于AI的文生图生成一直是一个重要但尚未解决的挑战。因此，我们引入了GreenStableYolo，它使用NSGA-II和Yolo优化Stable Diffusion的参数和提示，既减少GPU推理时间又提高图像生成质量。
<br>我们的实验表明，尽管与StableYolo（仅考虑图像质量）相比，图像质量略有折衷（18%），但GreenStableYolo实现了推理时间的大幅减少（减少266%）和超体积提高526%，从而推进了文生图生成的最新技术水平。
</details>

<details>
    <summary>Key points</summary>
    * 使用NSGA-II进行参数和提示调优
    * 与Yolo集成进行优化
    * 多目标优化平衡时间和质量
</details>
</details>

---


<details>
<summary><b> VersusDebias: Universal Zero-Shot Debiasing for Text-to-Image Models via SLM-Based Prompt Engineering and Generative Adversary</b></summary>

* **Authors:** Hanjun Luo, Ziye Deng, Haoyu Huang, Xuecheng Liu, Ruizhe Chen, Zuozhu Liu
* **arXiv ID:** 2407.19524
* **One-liner:** Introduced a universal debiasing framework for text-to-image models without model-specific training.
* **Published in:** arxiv (28 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.19524) | [[PDF]](https://arxiv.org/pdf/2407.19524) | [[Code]](https://github.com/VersusDebias/VersusDebias)

> **核心创新**
> VersusDebias采用数组生成模块处理幻觉和去偏多个属性，以及图像生成模块与小型语言模型进行提示修改，实现跨性别、种族和年龄的零样本去偏。

<details>
    <summary>Abstract</summary>
    随着文生图（T2I）模型的快速发展，人类图像生成中对人口社会群体的偏见成为一个重要问题，影响了AI的公平性和伦理标准。一些研究人员提出了他们的方法来应对这一问题。然而，现有方法针对特定模型和固定提示设计，限制了它们对快速演进模型和多样化实际场景的适应性。此外，它们忽视了幻觉的影响，导致预期结果与实际结果之间的差异。为了解决这些问题，我们引入了VersusDebias，一个新颖且通用的去偏框架，适用于任意T2I模型中的偏见，包括数组生成（AG）模块和图像生成（IG）模块。自适应AG模块生成专门的属性数组以后处理幻觉并同时去偏多个属性。IG模块使用一个小型语言模型根据数组修改提示，并驱动T2I模型生成去偏图像，实现零样本去偏。广泛实验证明了VersusDebias能够同时去偏任何模型在性别、种族和年龄方面的偏见。在零样本和少样本场景中，VersusDebias优于现有方法，展示了其卓越的实用性。我们的工作可在<a href="https://github.com/VersusDebias/VersusDebias" rel="external noopener nofollow" class="link-external link-https">此https URL</a>访问，以确保可重复性并促进进一步研究。
</details>

<details>
    <summary>Key points</summary>
    * 数组生成模块用于后处理幻觉
    * 图像生成模块与提示修改
    * 零样本和少样本去偏能力
</details>
</details>

---


<details>
<summary><b> Reproducibility Study of &#34;ITI-GEN: Inclusive Text-to-Image Generation&#34;</b></summary>

* **Authors:** Daniel Gallo Fernández, Răzvan-Andrei Matisan, Alejandro Monroy Muñoz, Janusz Partyka
* **arXiv ID:** 2407.19996
* **One-liner:** Reproduced and improved upon ITI-GEN for inclusive text-to-image generation, addressing limitations with Hard Prompt Search.
* **Published in:** arxiv (29 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.19996) | [[PDF]](https://arxiv.org/pdf/2407.19996) | [[Code]](https://github.com/humansensinglab/ITI-GEN)

> **核心创新**
> 该研究验证了ITI-GEN在多样性和可扩展性方面的改进，但识别了代理特征和属性解耦的问题；提出将ITI-GEN与带负提示的硬提示搜索相结合，以更好地处理否定。

<details>
    <summary>Abstract</summary>
    文生图生成模型通常在涉及某些敏感属性（如性别或肤色）的公平性方面存在问题。本研究旨在复现Zhang等人（2023a）在'ITI-GEN：包容性文生图生成'中提出的结果，该模型旨在提高这类模型的包容性。我们表明，作者关于ITI-GEN的大多数主张成立：它提高了生成图像的多样性和质量，可扩展到不同领域，具有即插即用能力，并且在计算上高效。然而，ITI-GEN有时使用不期望的属性作为代理特征，并且无法解耦一些（相关的）属性对，如性别和秃头。此外，当考虑属性数量增加时，训练时间呈指数增长，ITI-GEN难以生成联合分布中所有元素的包容性图像。为了解决这些问题，我们提出使用带负提示的硬提示搜索，这是一种无需训练的方法，比普通硬提示搜索更好地处理否定。尽管如此，硬提示搜索（带或不带负提示）不能用于难以用自然语言表达的连续属性，而ITI-GEN在这方面表现出色，因为它在训练中由图像引导。最后，我们提出将ITI-GEN与带负提示的硬提示搜索相结合。
</details>

<details>
    <summary>Key points</summary>
    * 复现ITI-GEN的主张
    * 识别ITI-GEN的局限性
    * 提出带负提示的硬提示搜索
    * 方法组合以增强包容性
</details>
</details>

---


<details>
<summary><b> VAR-CLIP: Text-to-Image Generator with Visual Auto-Regressive Modeling</b></summary>

* **Authors:** Qian Zhang, Xiangzi Dai, Ninghua Yang, Xiang An, Ziyong Feng, Xingyu Ren
* **arXiv ID:** 2408.01181
* **One-liner:** Developed a text-to-image model integrating visual auto-regressive techniques with CLIP for enhanced generation.
* **Published in:** arxiv (2 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.01181) | [[PDF]](https://arxiv.org/pdf/2408.01181) | [[Code]](https://github.com/daixiangzi/VAR-CLIP)

> **核心创新**
> VAR-CLIP在自回归变换器中使用下一尺度预测，用CLIP将标题编码为文本嵌入，并在大型数据集（如ImageNet）上训练，实现幻想图像的高保真度和文本一致性。

<details>
    <summary>Abstract</summary>
    VAR是一种新一代范式，采用'下一尺度预测'而非'下一令牌预测'。这种创新转变使自回归（AR）变换器能够快速学习视觉分布并实现鲁棒泛化。然而，原始VAR模型仅限于类条件合成，仅依赖文本标题进行指导。在本文中，我们引入了VAR-CLIP，一种新颖的文生图模型，将视觉自回归技术与CLIP的能力相结合。VAR-CLIP框架将标题编码为文本嵌入，然后用作图像生成的文本条件。为了在大型数据集（如ImageNet）上训练，我们利用BLIP2构建了一个大规模的图像-文本数据集。此外，我们深入探讨了CLIP中词位置对于标题指导的重要性。广泛实验证实了VAR-CLIP在生成具有高保真度、文本一致性和美学卓越性的幻想图像方面的熟练度。我们的项目页面在<a href="https://github.com/daixiangzi/VAR-CLIP" rel="external noopener nofollow" class="link-external link-https">此https URL</a>
</details>

<details>
    <summary>Key points</summary>
    * VAR与CLIP的集成
    * 使用下一尺度预测
    * 在大型图像-文本数据集上训练
    * 探索CLIP中词位置
</details>
</details>

---


<details>
<summary><b> Lumina-mGPT: Illuminate Flexible Photorealistic Text-to-Image Generation with Multimodal Generative Pretraining</b></summary>

* **Authors:** Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yi Xin, Xinyue Li, Qi Qin, Yu Qiao, Hongsheng Li, Peng Gao
* **arXiv ID:** 2408.02657
* **One-liner:** Created a multimodal autoregressive model for versatile image generation and other vision-language tasks.
* **Published in:** arxiv (5 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.02657) | [[PDF]](https://arxiv.org/pdf/2408.02657) | [[Code]](https://github.com/Alpha-VLLM/Lumina-mGPT)

> **核心创新**
> Lumina-mGPT从mGPT初始化，使用灵活渐进监督微调和无歧义图像表示生成不同纵横比的高质量图像，并扩展到全能监督微调以实现统一多模态能力。

<details>
    <summary>Abstract</summary>
    我们提出了Lumina-mGPT，一个多模态自回归模型家族，能够执行各种视觉和语言任务，特别擅长从文本描述生成灵活的照片级真实图像。通过从多模态生成预训练（mGPT）初始化，我们证明仅解码器自回归（AR）模型可以通过灵活渐进监督微调（FP-SFT）实现与现代扩散模型相当的图像生成性能和高效率。配备我们提出的无歧义图像表示（UniRep），Lumina-mGPT可以灵活生成不同纵横比的高质量图像。基于强大的图像生成能力，我们进一步探索全能监督微调（Omni-SFT），这是将Lumina-mGPT提升为统一多模态通用专家的初步尝试。所得模型展示了多才多艺的多模态能力，包括视觉生成任务（如文生图/多视图生成和可控生成）、视觉识别任务（如分割和深度估计）以及视觉语言任务（如多轮视觉问答），显示了该技术方向的广阔潜力。代码和检查点可在<a href="https://github.com/Alpha-VLLM/Lumina-mGPT" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 从多模态生成预训练初始化
    * 灵活渐进监督微调
    * 无歧义图像表示
    * 全能监督微调以实现通用能力
</details>
</details>

---


<details>
<summary><b> FRAP: Faithful and Realistic Text-to-Image Generation with Adaptive Prompt Weighting</b></summary>

* **Authors:** Liyao Jiang, Negar Hassanpour, Mohammad Salameh, Mohan Sai Singamsetti, Fengyu Sun, Wei Lu, Di Niu
* **arXiv ID:** 2408.11706
* **One-liner:** Proposed a method to improve prompt-image alignment and authenticity in text-to-image generation without latent code optimization.
* **Published in:** arxiv (21 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.11706) | [[PDF]](https://arxiv.org/pdf/2408.11706) | [[Code]]()

> **核心创新**
> FRAP使用在线算法自适应调整每令牌提示权重，以最小化对象存在和绑定的统一目标函数，减少延迟并改进对齐和真实感。

<details>
    <summary>Abstract</summary>
    文生图（T2I）扩散模型在给定文本提示生成高质量图像方面展示了令人印象深刻的能力。然而，确保提示-图像对齐仍然是一个相当大的挑战，即生成忠实对齐提示语义的图像。最近的工作尝试通过优化潜在代码来提高忠实度，但这可能导致潜在代码超出分布，从而产生不真实的图像。在本文中，我们提出了FRAP，一种简单而有效的方法，基于自适应调整每令牌提示权重，以改进提示-图像对齐和生成图像的真实性。我们设计了一个在线算法来自适应更新每个令牌的权重系数，这是通过最小化一个统一目标函数实现的，该函数鼓励对象存在和对象-修饰符对的绑定。通过广泛评估，我们显示FRAP在复杂数据集的提示上生成具有显著更高提示-图像对齐的图像，同时与最近的潜在代码优化方法相比具有更低的平均延迟，例如在COCO-Subject数据集上比D&B快4秒。此外，通过视觉比较和CLIP-IQA-Real指标评估，我们显示FRAP不仅改进了提示-图像对齐，还生成了更真实的图像，具有逼真的外观。我们还探索了将FRAP与提示重写LLM相结合，以恢复它们退化的提示-图像对齐，我们观察到在提示-图像对齐和图像质量方面的改进。我们在以下链接发布代码：<a href="https://github.com/LiyaoJiang1998/FRAP/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 自适应每令牌权重调整
    * 权重更新的在线算法
    * 对象存在和绑定的统一目标函数
    * 与提示重写LLM的组合
</details>
</details>

---


<details>
<summary><b> Rethinking Training for De-biasing Text-to-Image Generation: Unlocking the Potential of Stable Diffusion</b></summary>

* **Authors:** Eunji Kim, Siwon Kim, Minjun Park, Rahim Entezari, Sungroh Yoon
* **arXiv ID:** 2408.12692
* **One-liner:** Discovered and leveraged minority regions in Stable Diffusion for bias reduction without additional training.
* **Published in:** arxiv (22 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.12692) | [[PDF]](https://arxiv.org/pdf/2408.12692) | [[Code]]()

> **核心创新**
> 该方法识别少数属性的聚类初始噪声，并使用弱引导将随机噪声引导到这些区域，保留语义完整性和核心功能，同时减少偏见。

<details>
    <summary>Abstract</summary>
    最近文生图模型（如Stable Diffusion）的进展显示出显著的人口偏见。现有的去偏技术严重依赖额外训练，这带来了高计算成本和损害核心图像生成功能的风险。这阻碍了它们在现实世界应用中的广泛采用。在本文中，我们探索了Stable Diffusion被忽视的潜力，即无需额外训练即可减少偏见。通过我们的分析，我们发现与少数属性相关的初始噪声形成'少数区域'而非分散。我们将这些'少数区域'视为SD中减少偏见的机会。为了解锁这一潜力，我们提出了一种新颖的去偏方法，称为'弱引导'，精心设计以将随机噪声引导到少数区域，而不损害语义完整性。通过对各种版本SD的分析和实验，我们证明了我们提出的方法有效减少偏见，无需额外训练，实现了效率和核心图像生成功能的保留。
</details>

<details>
    <summary>Key points</summary>
    * 噪声中少数区域的识别
    * 用于偏见减少的弱引导技术
    * 无需额外训练
    * 图像生成功能的保留
</details>
</details>

---


<details>
<summary><b> Taming Text-to-Image Synthesis for Novices: User-centric Prompt Generation via Multi-turn Guidance</b></summary>

* **Authors:** Yilun Liu, Minggui He, Feiyu Yao, Yuhe Ji, Shimin Tao, Jingzhou Du, Duan Li, Jian Gao, Li Zhang, Hao Yang, Boxing Chen, Osamu Yoshie
* **arXiv ID:** 2408.12910
* **One-liner:** Introduced a dialogue-based prompt generation model to enhance user-centricity for novice users in text-to-image synthesis.
* **Published in:** arxiv (23 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.12910) | [[PDF]](https://arxiv.org/pdf/2408.12910) | [[Code]]()

> **核心创新**
> DialPrompt使用多轮对话工作流引导用户在15个维度上表达偏好，在策划数据集上训练，提高用户中心性，同时保持图像质量。

<details>
    <summary>Abstract</summary>
    文生图合成（TIS）模型的出现通过从书面描述生成高质量视觉内容，显著影响了数字图像创作。然而，这些模型对文本提示敏感，对不熟悉TIS提示写作的新手用户构成了挑战。现有解决方案通过自动提示扩展或从用户查询生成提示来缓解这一问题。然而，这种单轮方式在结果可解释性和用户交互性方面用户中心性有限。因此，我们提出了DialPrompt，一个基于对话的TIS提示生成模型，强调新手用户的用户体验。DialPrompt设计为遵循多轮工作流，在每轮对话中，模型引导用户在生成最终TIS提示前表达他们对可能优化维度的偏好。为了实现这一点，我们从高级用户中挖掘了15个高质量提示的基本维度，并策划了一个多轮数据集。通过在这个数据集上训练，DialPrompt通过允许用户感知和控制TIS提示的创建过程来提高用户中心性。实验表明，与现有方法相比，DialPrompt在用户中心性得分上显著提高，同时保持合成图像的竞争性质量。在我们的用户评估中，DialPrompt受到19名人类评审员（尤其是新手）的高度评价。
</details>

<details>
    <summary>Key points</summary>
    * 多轮对话工作流
    * 优化维度上的引导
    * 在策划多轮数据集上训练
    * 关注用户中心性和可解释性
</details>
</details>

---


<details>
<summary><b> Focus on Neighbors and Know the Whole: Towards Consistent Dense Multiview Text-to-Image Generator for 3D Creation</b></summary>

* **Authors:** Bonan Li, Zicheng Zhang, Xingyi Yang, Xinchao Wang
* **arXiv ID:** 2408.13149
* **One-liner:** Developed a consistent dense multiview text-to-image generator for high-quality 3D asset creation.
* **Published in:** arxiv (23 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.13149) | [[PDF]](https://arxiv.org/pdf/2408.13149) | [[Code]]()

> **核心创新**
> CoSER通过密集交互和运动路径聚合实现邻视图一致性，并通过螺旋双向扫描和加权下采样增强跨视图一致性，与注意力和状态空间模型集成。

<details>
    <summary>Abstract</summary>
    从文本提示生成密集多视图图像对于创建高保真3D资产至关重要。然而，现有方法在空间-视图对应方面存在困难，导致稀疏和低质量输出。在本文中，我们引入了CoSER，一种新颖的一致密集多视图文生图生成器，用于文生3D，通过细致学习邻视图一致性并通过快速遍历所有视图进一步缓解模糊性，实现效率和质量。为实现邻视图一致性，每个视点与相邻视点密集交互以感知全局空间结构，并沿物理原理明确定义的运动路径聚合信息以细化细节。为进一步增强跨视图一致性和缓解内容漂移，CoSER以螺旋双向方式快速扫描所有视图以感知整体信息，然后基于语义材料对每个点评分。随后，我们基于分数沿空间维度进行加权下采样，从而促进所有视图的突出信息融合与轻量计算。技术上，核心模块通过将注意力机制与选择性状态空间模型集成构建，利用前者的强大学习能力和后者的低开销。广泛评估显示，CoSER能够生成密集、高保真、内容一致的多视图图像，可以灵活集成到各种3D生成模型中。
</details>

<details>
    <summary>Key points</summary>
    * 通过密集交互实现邻视图一致性
    * 螺旋双向扫描以感知整体信息
    * 加权下采样用于信息融合
    * 注意力与状态空间模型的集成
</details>
</details>

---


<details>
<summary><b> CSGO: Content-Style Composition in Text-to-Image Generation</b></summary>

* **Authors:** Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, Zechao Li
* **arXiv ID:** 2408.16766
* **One-liner:** Constructed a large-scale style transfer dataset and proposed an end-to-end model for enhanced style control.
* **Published in:** arxiv (29 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.16766) | [[PDF]](https://arxiv.org/pdf/2408.16766) | [[Code]](https://github.com/instantX-research/CSGO)

> **核心创新**
> IMAGStyle数据集包含210k三元组，使CSGO能够训练，该模型解耦内容和风格特征，实现统一风格迁移、文本驱动合成和文本编辑驱动合成。

<details>
    <summary>Abstract</summary>
    扩散模型在受控图像生成中展示了卓越能力，这进一步激发了图像风格迁移的兴趣。现有工作主要关注基于免训练的方法（例如，图像反转），因为特定数据的稀缺性。在本研究中，我们提出了一个数据构建流程，用于内容-风格-风格化图像三元组，生成并自动清理风格化数据三元组。基于此流程，我们构建了IMAGStyle数据集，这是第一个大规模风格迁移数据集，包含210k图像三元组，可供社区探索和研究。配备IMAGStyle，我们提出了CSGO，一个基于端到端训练的风格迁移模型，它通过独立特征注入显式解耦内容和风格特征。统一的CSGO实现了图像驱动风格迁移、文本驱动风格化合成和文本编辑驱动风格化合成。广泛实验证明了我们方法在增强图像生成中风格控制能力方面的有效性。额外的可视化和源代码访问可在项目页面找到：\url{<a href="https://csgo-gen.github.io/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}。
</details>

<details>
    <summary>Key points</summary>
    * 风格迁移三元组的数据构建流程
    * IMAGStyle数据集创建
    * 具有特征解耦的CSGO模型
    * 多风格迁移任务的统一实现
</details>
</details>

---


<details>
<summary><b> Text-to-Image Generation Via Energy-Based CLIP</b></summary>

* **Authors:** Roy Ganz, Michael Elad
* **arXiv ID:** 2408.17046
* **One-liner:** Scaled Joint Energy Models to multimodal vision-language domain using CLIP.
* **Published in:** arxiv (30 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.17046) | [[PDF]](https://arxiv.org/pdf/2408.17046) | [[Code]]()

> **核心创新**
> 通过整合生成和判别目标，使用CLIP扩展JEMs，实现从文本生成逼真图像和在基准测试中的竞争性能。

<details>
    <summary>Abstract</summary>
    尽管联合能量模型（JEMs）吸引了大量研究关注，但尚未成功扩展到真实世界的高分辨率数据集。我们提出了CLIP-JEM，一种新颖方法，利用CLIP将JEMs扩展到多模态视觉-语言领域，整合了生成和判别目标。对于生成目标，我们基于CLIP空间中的余弦相似度引入了图像-文本联合能量函数，训练CLIP为真实图像-标题对分配低能量，否则分配高能量。对于判别目标，我们采用对比对抗损失，将对抗训练目标扩展到多模态领域。CLIP-JEM不仅从文本生成逼真图像，还在组合性基准测试中取得竞争性结果，以更少参数超越领先方法。此外，我们通过增强基于CLIP的生成框架和将无条件扩散模型转换为基于文本的模型，展示了CLIP-JEM的优越引导能力。最后，我们表明我们的模型可以作为比CLIP更鲁棒的文本到图像生成任务评估指标。
</details>

<details>
    <summary>Key points</summary>
    * 在CLIP空间中引入基于余弦相似度的图像-文本联合能量函数
    * 采用对比对抗损失进行多模态对抗训练
    * 展示了在增强CLIP框架和转换扩散模型中的优越引导能力
</details>
</details>

---


<details>
<summary><b> SPDiffusion: Semantic Protection Diffusion Models for Multi-concept Text-to-image Generation</b></summary>

* **Authors:** Yang Zhang, Rui Zhang, Xuecheng Nie, Haochen Li, Jikun Chen, Yifan Hao, Xin Zhang, Luoqi Liu, Ling Li
* **arXiv ID:** 2409.01327
* **One-liner:** Addressed semantic entanglement in multi-concept text-to-image generation.
* **Published in:** arxiv (2 Sep 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2409.01327) | [[PDF]](https://arxiv.org/pdf/2409.01327) | [[Code]]()

> **核心创新**
> 提出了SPDiffusion，采用概念区域提取和保护机制解决概念和属性纠缠，实现最先进结果。

<details>
    <summary>Abstract</summary>
    最近的文本到图像模型在生成高质量图像方面取得了令人印象深刻的成果。然而，当任务涉及多概念生成，即创建包含多个角色或对象的图像时，现有方法常常遭受语义纠缠，包括概念纠缠和属性绑定不当，导致显著的文本-图像不一致。我们发现，当潜在特征的某些区域关注错误的概念和属性标记时，语义纠缠就会发生。在这项工作中，我们提出了语义保护扩散模型（SPDiffusion），仅使用文本提示作为输入来解决概念纠缠和属性绑定不当问题。SPDiffusion框架引入了一种新颖的概念区域提取方法SP-Extraction来解决交叉注意力中的区域纠缠，以及SP-Attn，它保护概念区域免受不相关属性和概念的影响。为了评估我们的方法，我们在现有基准测试中进行了测试，SPDiffusion取得了最先进的结果，证明了其有效性。
</details>

<details>
    <summary>Key points</summary>
    * 引入SP-Extraction用于概念区域提取
    * 开发SP-Attn以保护概念区域免受不相关影响
    * 在基准测试中评估文本-图像一致性
</details>
</details>

---


<details>
<summary><b> Qihoo-T2X: An Efficient Proxy-Tokenized Diffusion Transformer for Text-to-Any-Task</b></summary>

* **Authors:** Jing Wang, Ao Ma, Jiasong Feng, Dawei Leng, Yuhui Yin, Xiaodan Liang
* **arXiv ID:** 2409.04005
* **One-liner:** Reduced computational complexity in diffusion transformers with proxy tokens.
* **Published in:** arxiv (6 Sep 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2409.04005) | [[PDF]](https://arxiv.org/pdf/2409.04005) | [[Code]](https://github.com/360CVGroup/Qihoo-T2X)

> **核心创新**
> 设计了PT-DiT，使用稀疏代表性标记注意力高效捕获全局语义，导致图像和视频生成中计算显著减少。

<details>
    <summary>Abstract</summary>
    扩散变换器中的全局自注意力机制由于视觉信息的稀疏和冗余性质，涉及冗余计算，且空间窗口内标记的注意力图显示出显著相似性。为了解决这种冗余，我们提出了代理标记化扩散变换器（PT-DiT），它采用稀疏代表性标记注意力（其中代表性标记数量远少于总标记数）来高效建模全局视觉信息。具体来说，在每个变换器块内，我们从每个时空窗口计算平均标记作为该区域的代理标记。全局语义通过这些代理标记的自注意力捕获，然后通过交叉注意力注入所有潜在标记。同时，我们引入窗口和移位窗口注意力来解决稀疏注意力机制在细节建模方面的限制。基于精心设计的PT-DiT，我们进一步开发了Qihoo-T2X系列，包括用于T2I、T2V和T2MV任务的各种模型。实验结果表明，PT-DiT在图像和视频生成任务中实现了竞争性能，同时显著降低了计算复杂度（例如，与DiT相比减少49%，与PixArt-α相比减少34%）。Qihoo-T2X的视觉展示和源代码可在指定URL获取。
</details>

<details>
    <summary>Key points</summary>
    * 采用时空窗口的代理标记进行全局注意力
    * 引入窗口和移位窗口注意力用于细节建模
    * 构建Qihoo-T2X系列用于多种生成任务
</details>
</details>

---


<details>
<summary><b> IFAdapter: Instance Feature Control for Grounded Text-to-Image Generation</b></summary>

* **Authors:** Yinwei Wu, Xianpan Zhou, Bing Ma, Xuefeng Su, Kai Ma, Xinchao Wang
* **arXiv ID:** 2409.08240
* **One-liner:** Enhanced instance feature generation with accurate positioning and fidelity.
* **Published in:** arxiv (12 Sep 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2409.08240) | [[PDF]](https://arxiv.org/pdf/2409.08240) | [[Code]]()

> **核心创新**
> 引入了IFG任务和IFAdapter以对齐实例级特征与空间位置，在定量和定性评估中优于其他模型。

<details>
    <summary>Abstract</summary>
    虽然文本到图像（T2I）扩散模型在生成单个实例的视觉吸引力图像方面表现出色，但它们在准确定位和控制多个实例的特征生成方面存在困难。布局到图像（L2I）任务被引入以通过结合边界框作为空间控制信号来解决定位挑战，但它仍然在生成精确实例特征方面不足。为此，我们提出了实例特征生成（IFG）任务，旨在确保生成实例的位置准确性和特征保真度。为了解决IFG任务，我们引入了实例特征适配器（IFAdapter）。IFAdapter通过结合额外外观标记和利用实例语义图来对齐实例级特征与空间位置，从而增强特征描绘。IFAdapter作为即插即用模块指导扩散过程，使其适应各种社区模型。为了评估，我们贡献了一个IFG基准并开发了一个验证流程，以客观比较模型生成具有准确定位和特征的实例的能力。实验结果表明，IFAdapter在定量和定性评估中均优于其他模型。
</details>

<details>
    <summary>Key points</summary>
    * 提出了带有外观标记的实例特征适配器
    * 利用实例语义图进行特征对齐
    * 开发了IFG基准和验证流程
</details>
</details>

---


<details>
<summary><b> Generalizing Alignment Paradigm of Text-to-Image Generation with Preferences through $f$-divergence Minimization</b></summary>

* **Authors:** Haoyuan Sun, Bo Xia, Yongzhe Chang, Xueqian Wang
* **arXiv ID:** 2409.09774
* **One-liner:** Extended alignment paradigm to f-divergence for better trade-off in text-to-image models.
* **Published in:** arxiv (15 Sep 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2409.09774) | [[PDF]](https://arxiv.org/pdf/2409.09774) | [[Code]]()

> **核心创新**
> 将DPO推广到f-散度，分析梯度场并发现Jensen-Shannon散度在对齐和多样性方面最优。

<details>
    <summary>Abstract</summary>
    直接偏好优化（DPO）最近从对齐大型语言模型（LLMs）成功扩展到对齐文本到图像模型与人类偏好，这在社区内引起了相当大的兴趣。然而，我们观察到这些方法在微调模型和参考模型之间的对齐过程中仅依赖于最小化反向Kullback-Leibler散度，忽略了其他散度约束的整合。在本研究中，我们专注于将文本到图像模型对齐范式中的反向Kullback-Leibler散度扩展到f-散度，旨在获得更好的对齐性能以及良好的生成多样性。我们提供了f-散度条件下对齐范式的广义公式，并从梯度场角度深入分析了不同散度约束对对齐过程的影响。我们在不同散度约束下对图像-文本对齐性能、人类价值对齐性能和生成多样性性能进行了全面评估，结果表明基于Jensen-Shannon散度的对齐在其中实现了最佳权衡。用于对齐文本到图像模型的散度选择显著影响对齐性能（尤其是人类价值对齐）和生成多样性之间的权衡，这突显了在实际应用中选择适当散度的必要性。
</details>

<details>
    <summary>Key points</summary>
    * 提供了f-散度条件下的广义公式
    * 分析了不同散度对对齐过程的影响
    * 进行了对齐性能和多样性的评估
</details>
</details>

---


<details>
<summary><b> Evaluating Image Hallucination in Text-to-Image Generation with Question-Answering</b></summary>

* **Authors:** Youngsun Lim, Hojun Choi, Hyunjung Shim
* **arXiv ID:** 2409.12784
* **One-liner:** Introduced automated metric for evaluating factual accuracy in generated images.
* **Published in:** arxiv (19 Sep 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2409.12784) | [[PDF]](https://arxiv.org/pdf/2409.12784) | [[Code]](https://github.com/kaist-cvml/I-HallA-v1.0)

> **核心创新**
> 开发了I-HallA，使用VQA测量图像幻觉，基准数据集显示与人类判断强相关。

<details>
    <summary>Abstract</summary>
    尽管文本到图像（TTI）生成模型取得了令人印象深刻的成功，现有研究忽视了这些模型是否准确传达事实信息的问题。在本文中，我们关注图像幻觉问题，即生成模型创建的图像未能忠实描绘事实内容。为了解决这个问题，我们引入了I-HallA（基于问答的图像幻觉评估），一种新颖的自动化评估指标，通过视觉问答（VQA）测量生成图像的事实性。我们还引入了I-HallA v1.0，一个为此目的策划的基准数据集。作为这个过程的一部分，我们开发了一个使用多个基于GPT-4 Omni的代理生成高质量问答对的流程，并通过人类判断确保准确性。我们的评估协议通过测试现有TTI模型的图像是否能正确响应这些问题来测量图像幻觉。I-HallA v1.0数据集包含1.2K个跨九个类别的多样化图像-文本对，以及1,000个涵盖各种组合挑战的严格策划问题。我们使用I-HallA评估了五个TTI模型，并揭示这些最先进模型常常无法准确传达事实信息。此外，我们通过展示与人类判断的强Spearman相关性（ρ=0.95）验证了我们指标的可靠性。我们相信我们的基准数据集和指标可以作为开发事实准确的TTI生成模型的基础。额外资源可在项目页面找到。
</details>

<details>
    <summary>Key points</summary>
    * 创建了I-HallA v1.0数据集，包含策划的问答对
    * 使用基于GPT-4 Omni的代理进行流程开发
    * 评估了多个TTI模型的事实性
</details>
</details>

---


<details>
<summary><b> Text Image Generation for Low-Resource Languages with Dual Translation Learning</b></summary>

* **Authors:** Chihiro Noguchi, Shun Fukuda, Shoichiro Mihara, Masao Yamanaka
* **arXiv ID:** 2409.17747
* **One-liner:** Improved scene text recognition for low-resource languages via synthetic image generation.
* **Published in:** arxiv (26 Sep 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2409.17747) | [[PDF]](https://arxiv.org/pdf/2409.17747) | [[Code]]()

> **核心创新**
> 提出了以二元状态为条件的扩散模型来生成文本图像，增强识别模型性能。

<details>
    <summary>Abstract</summary>
    低资源语言中的场景文本识别经常因真实场景训练数据集的有限可用性而面临挑战。本研究提出了一种新颖方法，通过模拟高资源语言中真实文本图像的风格来生成低资源语言的文本图像。我们的方法利用了一个扩散模型，该模型以二元状态为条件：“合成”和“真实”。该模型的训练涉及双重翻译任务，其中它将纯文本图像转换为合成或真实文本图像，基于二元状态。这种方法不仅有效区分了两个领域，还促进了模型对目标语言字符的显式识别。此外，为了增强生成文本图像的准确性和多样性，我们引入了两种引导技术：保真度-多样性平衡引导和保真度增强引导。我们的实验结果表明，我们提出的框架生成的文本图像可以显著提高低资源语言场景文本识别模型的性能。
</details>

<details>
    <summary>Key points</summary>
    * 利用双重翻译任务进行领域区分
    * 引入了保真度-多样性平衡和增强引导
    * 生成模拟高资源语言真实风格的文本图像
</details>
</details>

---


<details>
<summary><b> MCGM: Mask Conditional Text-to-Image Generative Model</b></summary>

* **Authors:** Rami Skaik, Leonardo Rossi, Tomaso Fontanini, Andrea Prati
* **arXiv ID:** 2410.00483
* **One-liner:** Enabled pose-specific image generation with mask conditioning.
* **Published in:** arxiv (1 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.00483) | [[PDF]](https://arxiv.org/pdf/2410.00483) | [[Code]]()

> **核心创新**
> 开发了MCGM，通过将掩码嵌入注入扩散模型，允许从单张图像灵活控制主题姿态。

<details>
    <summary>Abstract</summary>
    生成模型的最近进展彻底改变了人工智能领域，使得能够创建高度逼真和详细的图像。在本研究中，我们提出了一种新颖的掩码条件文本到图像生成模型（MCGM），它利用条件扩散模型的力量生成具有特定姿态的图片。我们的模型建立在Break-a-scene [1]模型在使用单张图像生成新场景方面的成功基础上，并整合了掩码嵌入注入，允许对生成过程进行条件化。通过引入这种额外的控制级别，MCGM提供了一种灵活直观的方法，用于从单张图像学习的一个或多个主题生成特定姿态，使用户能够根据需求影响输出。通过广泛的实验和评估，我们证明了我们提出的模型在生成满足预定义掩码条件的高质量图像以及改进当前Break-a-scene生成模型方面的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 基于Break-a-scene模型进行场景生成
    * 引入了掩码嵌入注入用于条件化
    * 展示了在高质量图像生成中的有效性
</details>
</details>

---


<details>
<summary><b> Accelerating Auto-regressive Text-to-Image Generation with Training-free Speculative Jacobi Decoding</b></summary>

* **Authors:** Yao Teng, Han Shi, Xian Liu, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, Xihui Liu
* **arXiv ID:** 2410.01699
* **One-liner:** Accelerated auto-regressive text-to-image generation with probabilistic parallel decoding.
* **Published in:** arxiv (2 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.01699) | [[PDF]](https://arxiv.org/pdf/2410.01699) | [[Code]](https://github.com/tyshiwo1/Accelerating-T2I-AR-with-SJD/)

> **核心创新**
> 提出了SJD算法，使用概率收敛标准减少推理步骤，同时保持多样性和质量。

<details>
    <summary>Abstract</summary>
    当前的大型自回归模型可以生成高质量、高分辨率的图像，但这些模型在推理过程中需要数百甚至数千步的下一个标记预测，导致大量时间消耗。在现有研究中，Jacobi解码，一种迭代并行解码算法，已被用于加速自回归生成，并且可以在无需训练的情况下执行。然而，Jacobi解码依赖于确定性标准来确定迭代的收敛性。因此，它适用于贪婪解码，但与基于采样的解码不兼容，而基于采样的解码对于当前自回归文本到图像生成中的视觉质量和多样性至关重要。在本文中，我们提出了一种无需训练的概率并行解码算法，推测性Jacobi解码（SJD），以加速自回归文本到图像生成。通过引入概率收敛标准，我们的SJD加速了自回归文本到图像生成的推理，同时保持了基于采样的标记解码中的随机性，并允许模型生成多样化图像。具体来说，SJD促进模型在每一步预测多个标记，并根据概率标准接受标记，使模型能够以比传统下一个标记预测范式更少的步骤生成图像。我们还研究了利用视觉数据空间局部性的标记初始化策略，以在特定场景下进一步提高加速比。我们在多个自回归文本到图像生成模型上对我们的SJD进行了实验，显示了在不牺牲视觉质量的情况下模型加速的有效性。我们工作的代码可在指定URL获取。
</details>

<details>
    <summary>Key points</summary>
    * 引入了概率标准用于标记接受
    * 研究了标记初始化策略以加速
    * 在多个模型上进行了无需训练的实验
</details>
</details>

---


<details>
<summary><b> ComfyGen: Prompt-Adaptive Workflows for Text-to-Image Generation</b></summary>

* **Authors:** Rinon Gal, Adi Haviv, Yuval Alaluf, Amit H. Bermano, Daniel Cohen-Or, Gal Chechik
* **arXiv ID:** 2410.01731
* **One-liner:** Automated prompt-adaptive workflow generation for text-to-image tasks.
* **Published in:** arxiv (2 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.01731) | [[PDF]](https://arxiv.org/pdf/2410.01731) | [[Code]](https://github.com/google/nerfies)

> **核心创新**
> 引入了基于LLM的方法来定制工作流以适应用户提示，提高图像质量超过单一模型。

<details>
    <summary>Abstract</summary>
    文本到图像生成的实际使用已从简单的单一模型演变为结合多个专门组件的复杂工作流。虽然基于工作流的方法可以导致图像质量的提高，但设计有效的工作流需要大量专业知识，因为可用组件数量众多、它们之间相互依赖复杂，并且它们依赖于生成提示。在这里，我们引入了提示自适应工作流生成的新任务，其目标是自动为每个用户提示定制工作流。我们提出了两种基于LLM的方法来解决这个任务：一种基于调优的方法，从用户偏好数据中学习；另一种无需训练的方法，使用LLM选择现有工作流。两种方法在与单一模型或通用、提示无关的工作流相比时，都导致图像质量的提高。我们的工作表明，提示依赖的工作流预测提供了一条改进文本到图像生成质量的新途径，补充了该领域的现有研究方向。
</details>

<details>
    <summary>Key points</summary>
    * 提出了基于调优和无需训练的LLM方法
    * 解决了组件选择和相互依赖问题
    * 展示了在提示依赖工作流预测中的有效性
</details>
</details>

---


<details>
<summary><b> A Spark of Vision-Language Intelligence: 2-Dimensional Autoregressive Transformer for Efficient Finegrained Image Generation</b></summary>

* **Authors:** Liang Chen, Sinan Tan, Zefan Cai, Weichu Xie, Haozhe Zhao, Yichi Zhang, Junyang Lin, Jinze Bai, Tianyu Liu, Baobao Chang
* **arXiv ID:** 2410.01912
* **One-liner:** Introduced the DnD-Transformer to address information loss in VQ autoregressive image generation by adding a model depth direction.
* **Published in:** arxiv (2 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.01912) | [[PDF]](https://arxiv.org/pdf/2410.01912) | [[Code]](https://github.com/chenllliang/DnD-Transformer)

> **核心创新**
> 通过二维自回归方法增强自回归图像生成，提高质量并实现自监督生成包含文本和图形的图像。

<details>
    <summary>Abstract</summary>
    本研究通过引入一种新颖的模型架构——二维自回归（DnD）Transformer，解决了向量量化（VQ）自回归图像生成中的信息损失瓶颈问题。DnD-Transformer通过引入一个新的自回归方向——模型深度，以及序列长度方向，预测图像的更多代码。与传统的1D自回归和先前利用类似2D图像分解的工作（如RQ-Transformer）相比，DnD-Transformer是一个端到端模型，能够在相同骨干模型大小和序列长度下生成更高质量的图像，为自回归图像生成开辟了新的优化视角。此外，我们的实验表明，DnD-Transformer的潜力不仅限于生成自然图像，它甚至能以自监督方式生成包含丰富文本和图形元素的图像，展示了这些组合模态的理解能力。这在流行的视觉生成模型（如扩散模型）中尚未被证明，表明仅通过图像训练即可激发视觉语言智能。代码、数据集和模型已在<a href="https://github.com/chenllliang/DnD-Transformer" rel="external noopener nofollow" class="link-external link-https">此https URL</a>开源。
</details>

<details>
    <summary>Key points</summary>
    * 引入具有模型深度方向的二维自回归
    * 端到端模型用于生成更高质量图像
    * 从仅图像训练中展示视觉语言智能
</details>
</details>

---


<details>
<summary><b> EvolveDirector: Approaching Advanced Text-to-Image Generation with Large Vision-Language Models</b></summary>

* **Authors:** Rui Zhao, Hangjie Yuan, Yujie Wei, Shiwei Zhang, Yuchao Gu, Lingmin Ran, Xiang Wang, Zhangjie Wu, Junhao Zhang, Yingya Zhang, Mike Zheng Shou
* **arXiv ID:** 2410.07133
* **One-liner:** Developed EvolveDirector to train text-to-image models using public APIs and VLM guidance, reducing data needs and costs.
* **Published in:** arxiv (9 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.07133) | [[PDF]](https://arxiv.org/pdf/2410.07133) | [[Code]](https://github.com/showlab/EvolveDirector)

> **核心创新**
> 提出了一个框架，利用API生成数据和VLM反馈来演化基础模型，以更少数据实现优越性能。

<details>
    <summary>Abstract</summary>
    生成模型的近期进展展示了生成精彩内容的卓越能力。然而，大多数模型在专有高质量数据上训练，一些模型保留其参数仅提供可访问的应用程序编程接口（API），限制了它们对下游任务的益处。为了探索使用公开可用资源训练与先进模型相当的文本到图像生成模型的可行性，我们引入了EvolveDirector。该框架通过公共API与先进模型交互，获取文本-图像数据对以训练基础模型。我们的大规模数据实验表明，在先进模型生成数据上训练的模型可以近似其生成能力，但需要1000万或更多的大规模样本，这导致时间、计算资源以及调用付费API相关成本的显著开销。为了解决这个问题，我们利用预训练的大型视觉语言模型（VLM）来指导基础模型的演化。VLM在训练期间持续评估基础模型，并通过判别、扩展、删除和突变操作动态更新和精炼训练数据集。实验结果显示，这种范式显著减少了所需数据量。此外，当接近多个先进模型时，EvolveDirector可以选择它们生成的最佳样本来学习强大且平衡的能力。最终训练的模型Edgen被证明优于这些先进模型。代码和模型权重可在<a href="https://github.com/showlab/EvolveDirector" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 通过API与先进模型交互以收集数据
    * 使用VLM进行动态数据集精炼
    * 将所需数据量减少至1000万以下
</details>
</details>

---


<details>
<summary><b> IterComp: Iterative Composition-Aware Feedback Learning from Model Gallery for Text-to-Image Generation</b></summary>

* **Authors:** Xinchen Zhang, Ling Yang, Guohao Li, Yaqi Cai, Jiake Xie, Yong Tang, Yujiu Yang, Mengdi Wang, Bin Cui
* **arXiv ID:** 2410.07171
* **One-liner:** Introduced IterComp to enhance compositional text-to-image generation by aggregating model preferences and iterative feedback learning.
* **Published in:** arxiv (9 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.07171) | [[PDF]](https://arxiv.org/pdf/2410.07171) | [[Code]](https://github.com/YangLing0818/IterComp)

> **核心创新**
> 通过训练奖励模型和迭代精炼生成，改进了扩散模型的组合能力。

<details>
    <summary>Abstract</summary>
    先进扩散模型如RPG、Stable Diffusion 3和FLUX在组合文本到图像生成方面取得了显著进展。然而，这些方法通常在组合生成中表现出不同的优势，一些在属性绑定方面表现出色，另一些在空间关系方面表现优异。这种差异凸显了需要一种方法能够利用各种模型的互补优势来全面改进组合能力。为此，我们引入了IterComp，一个新颖的框架，它聚合来自多个模型的组合感知模型偏好，并采用迭代反馈学习方法来增强组合生成。具体来说，我们策划了一个包含六个强大开源扩散模型的画廊，并评估它们的三个关键组合指标：属性绑定、空间关系和非空间关系。基于这些指标，我们开发了一个组合感知模型偏好数据集，包含大量图像-排名对，以训练组合感知奖励模型。然后，我们提出了一种迭代反馈学习方法，以闭环方式增强组合性，使基础扩散模型和奖励模型在多次迭代中逐步自我精炼。理论证明展示了其有效性，广泛实验显示我们在先前SOTA方法（如Omost和FLUX）上的显著优势，特别是在多类别对象组合和复杂语义对齐方面。IterComp为扩散模型的奖励反馈学习和组合生成开辟了新的研究途径。代码：<a href="https://github.com/YangLing0818/IterComp" rel="external noopener nofollow" class="link-external link-https">此https URL</a>
</details>

<details>
    <summary>Key points</summary>
    * 策划扩散模型画廊进行评估
    * 开发组合感知奖励模型
    * 采用迭代反馈学习进行自我精炼
</details>
</details>

---


<details>
<summary><b> Minority-Focused Text-to-Image Generation via Prompt Optimization</b></summary>

* **Authors:** Soobin Um, Jong Chul Ye
* **arXiv ID:** 2410.07838
* **One-liner:** Presented a framework to generate minority samples in T2I diffusion models via prompt optimization and likelihood objectives.
* **Published in:** arxiv (10 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.07838) | [[PDF]](https://arxiv.org/pdf/2410.07838) | [[Code]](https://github.com/soobin-um/MinorityPrompt)

> **核心创新**
> 通过优化提示来鼓励少数特征生成，解决了T2I模型的高密度聚焦问题。

<details>
    <summary>Abstract</summary>
    我们研究了使用预训练文本到图像（T2I）潜在扩散模型生成少数样本。在T2I生成背景下，少数实例可以定义为位于文本条件数据分布低密度区域的实例。它们对于现代T2I生成器的各种应用（如数据增强和创意AI）具有价值。不幸的是，现有预训练T2I扩散模型主要关注高密度区域，这很大程度上是由于引导采样器（如CFG）的影响，这些采样器对于高质量生成至关重要。为了解决这个问题，我们提出了一个新颖框架来对抗T2I扩散模型的高密度聚焦。具体来说，我们首先开发了一个在线提示优化框架，鼓励在推理过程中出现所需属性，同时保留用户提供提示的语义内容。随后，我们将这个通用提示优化器定制为一个专门求解器，通过融入精心设计的似然目标来促进少数特征的生成。在各种T2I模型上进行的广泛实验表明，与现有采样器相比，我们的方法显著增强了生成高质量少数实例的能力。代码可在<a href="https://github.com/soobin-um/MinorityPrompt" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 开发在线提示优化框架
    * 融入少数特征的似然目标
    * 增强低密度实例的生成能力
</details>
</details>

---


<details>
<summary><b> DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation</b></summary>

* **Authors:** Jiatao Gu, Yuyang Wang, Yizhe Zhang, Qihang Zhang, Dinghuai Zhang, Navdeep Jaitly, Josh Susskind, Shuangfei Zhai
* **arXiv ID:** 2410.08159
* **One-liner:** Proposed DART, a transformer-based model unifying autoregressive and diffusion approaches in a non-Markovian framework.
* **Published in:** arxiv (10 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.08159) | [[PDF]](https://arxiv.org/pdf/2410.08159) | [[Code]](https://github.com/daixiangzi/VAR-CLIP)

> **核心创新**
> 将自回归和扩散统一用于图像生成，无需量化，实现与文本和图像数据的高效训练。

<details>
    <summary>Abstract</summary>
    扩散模型已成为视觉生成的主导方法。它们通过去噪一个逐渐向输入添加噪声的马尔可夫过程进行训练。我们认为马尔可夫性质限制了模型充分利用生成轨迹的能力，导致训练和推理过程中的低效率。在本文中，我们提出了DART，一个基于Transformer的模型，将自回归（AR）和扩散统一在一个非马尔可夫框架中。DART使用与标准语言模型相同架构的自回归模型，在空间和频谱上迭代去噪图像块。DART不依赖于图像量化，从而在保持灵活性的同时实现更有效的图像建模。此外，DART在统一模型中无缝训练文本和图像数据。我们的方法在类条件和文本到图像生成任务上展示了竞争性能，为传统扩散模型提供了一个可扩展、高效的替代方案。通过这个统一框架，DART为可扩展、高质量图像合成设定了新基准。
</details>

<details>
    <summary>Key points</summary>
    * 使用非马尔可夫框架进行去噪
    * 在空间和频谱上迭代去噪块
    * 无缝训练多模态数据
</details>
</details>

---


<details>
<summary><b> Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis</b></summary>

* **Authors:** Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Xiangtai Li, Zhen Dong, Lei Zhu, Shuicheng Yan
* **arXiv ID:** 2410.08261
* **One-liner:** Elevated MIM text-to-image generation to SOTA levels with Meissonic, incorporating architectural innovations and high-quality data.
* **Published in:** arxiv (10 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.08261) | [[PDF]](https://arxiv.org/pdf/2410.08261) | [[Code]](https://github.com/viiika/Meissonic)

> **核心创新**
> 通过多种优化改进了非自回归MIM，在质量和分辨率上匹配或超越扩散模型。

<details>
    <summary>Abstract</summary>
    我们提出了Meissonic，它将非自回归掩码图像建模（MIM）文本到图像提升到与最先进扩散模型（如SDXL）相当的水平。通过整合一套全面的架构创新、先进位置编码策略和优化采样条件，Meissonic显著提高了MIM的性能和效率。此外，我们利用高质量训练数据，整合基于人类偏好分数的微条件，并采用特征压缩层来进一步增强图像保真度和分辨率。我们的模型不仅匹配而且经常超过现有模型（如SDXL）在生成高质量、高分辨率图像方面的性能。广泛实验验证了Meissonic的能力，展示了其作为文本到图像合成新标准的潜力。我们发布了一个能够生成$1024 \times 1024$分辨率图像的模型检查点。
</details>

<details>
    <summary>Key points</summary>
    * 整合架构和位置编码创新
    * 使用来自人类偏好的微条件
    * 将图像保真度和分辨率提升至1024x1024
</details>
</details>

---


<details>
<summary><b> Text-To-Image with Generative Adversarial Networks</b></summary>

* **Authors:** Mehrshad Momen-Tayefeh
* **arXiv ID:** 2410.08608
* **One-liner:** Conducted a comparison of GAN-based text-to-image methods, identifying the best model based on accuracy metrics.
* **Published in:** arxiv (11 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.08608) | [[PDF]](https://arxiv.org/pdf/2410.08608) | [[Code]]()

> **核心创新**
> 分析了五种GAN架构用于文本到图像合成，重点关注分辨率和准确性比较。

<details>
    <summary>Abstract</summary>
    从人类文本生成逼真图像是计算机视觉（CV）领域最具挑战性的问题之一。现有文本到图像方法可以大致反映所给描述的含义。在本文中，我们的主要目的是基于生成对抗网络（GAN）对五种不同方法进行简要比较，以从文本生成图像。此外，每种模型架构合成不同分辨率的图像。最佳和最差获得的分辨率分别为64*64和256*256。然而，我们检查并比较了一些引入每个模型准确性的指标。通过这项研究，我们通过比较这些不同方法的关键指标，找出了解决此问题的最佳模型。
</details>

<details>
    <summary>Key points</summary>
    * 比较五种不同的GAN模型
    * 评估图像分辨率和准确性指标
    * 识别性能最佳模型
</details>
</details>

---


<details>
<summary><b> Generating Intermediate Representations for Compositional Text-To-Image Generation</b></summary>

* **Authors:** Ran Galun, Sagie Benaim
* **arXiv ID:** 2410.09792
* **One-liner:** Proposed a compositional approach for text-to-image generation using intermediate representations to improve spatial accuracy.
* **Published in:** arxiv (13 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.09792) | [[PDF]](https://arxiv.org/pdf/2410.09792) | [[Code]](https://github.com/RANG1991/Public-Intermediate-Semantics-For-Generation)

> **核心创新**
> 通过首先生成对齐中间图，然后映射到最终图像，增强了扩散模型，以改善FID分数。

<details>
    <summary>Abstract</summary>
    文本到图像扩散模型展示了生成高质量输出的令人印象深刻的能力。然而，它们通常难以准确遵循输入文本中的细粒度空间信息。为此，我们提出了一种基于两个阶段的组合方法用于文本到图像生成。在第一阶段，我们设计了一个基于扩散的生成模型，以生成一个或多个对齐的中间表示（如深度或分割图），条件于文本。在第二阶段，我们使用另一个基于扩散的生成模型将这些表示与文本一起映射到最终输出图像。我们的发现表明，这种组合方法可以改进图像生成，与标准非组合基线相比，FID分数显著提高，CLIP分数相当。
</details>

<details>
    <summary>Key points</summary>
    * 使用两阶段扩散过程
    * 生成如深度图的中间表示
    * 与基线相比改善FID分数
</details>
</details>

---


<details>
<summary><b> FlexGen: Flexible Multi-View Generation from Text and Image Inputs</b></summary>

* **Authors:** Xinli Xu, Wenhang Ge, Jiantao Lin, Jiawei Feng, Lie Xu, HanFeng Zhao, Shunsi Zhang, Ying-Cong Chen
* **arXiv ID:** 2410.10745
* **One-liner:** Introduced FlexGen for controllable multi-view image generation using 3D-aware text annotations from GPT-4V.
* **Published in:** arxiv (14 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.10745) | [[PDF]](https://arxiv.org/pdf/2410.10745) | [[Code]](https://xxu068.github.io/flexgen.github.io/)

> **核心创新**
> 从单视图或文本输入实现灵活多视图合成，具有增强的属性可控性。

<details>
    <summary>Abstract</summary>
    在这项工作中，我们引入了FlexGen，一个灵活的框架，设计用于生成可控且一致的多视图图像，条件于单视图图像、文本提示或两者。FlexGen通过额外条件于3D感知文本注释来解决可控多视图合成的挑战。我们利用GPT-4V的强大推理能力生成3D感知文本注释。通过分析以平铺多视图图像排列的对象的四个正交视图，GPT-4V可以生成包含空间关系的3D感知信息的文本注释。通过将控制信号与提出的自适应双控制模块集成，我们的模型可以生成与指定文本对应的多视图图像。FlexGen支持多种可控能力，允许用户修改文本提示以生成合理且对应的未见部分。此外，用户可以影响属性，如外观和材料属性，包括金属性和粗糙度。广泛实验表明，我们的方法提供了增强的多重可控性，标志着对现有多视图扩散模型的显著进步。这项工作对需要快速灵活3D内容创建的领域（包括游戏开发、动画和虚拟现实）具有重要影响。项目页面：<a href="https://xxu068.github.io/flexgen.github.io/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 利用GPT-4V生成3D感知文本注释
    * 整合自适应双控制模块
    * 支持多种可控能力
</details>
</details>

---


<details>
<summary><b> HART: Efficient Visual Generation with Hybrid Autoregressive Transformer</b></summary>

* **Authors:** Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, Song Han
* **arXiv ID:** 2410.10812
* **One-liner:** Developed HART, a hybrid autoregressive model combining discrete and continuous tokens for high-quality 1024x1024 image generation.
* **Published in:** arxiv (14 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.10812) | [[PDF]](https://arxiv.org/pdf/2410.10812) | [[Code]](https://github.com/mit-han-lab/hart)

> **核心创新**
> 通过混合标记器克服AR模型限制，实现比扩散模型更好的FID和效率。

<details>
    <summary>Abstract</summary>
    我们引入了混合自回归Transformer（HART），一个自回归（AR）视觉生成模型，能够直接生成1024x1024图像，在图像生成质量上与扩散模型相媲美。现有AR模型面临其离散标记器图像重建质量差和生成1024px图像相关训练成本过高的限制。为了解决这些挑战，我们提出了混合标记器，它将自编码器的连续潜在分解为两个组件：表示大局的离散标记和表示无法由离散标记表示的残差组件的连续标记。离散组件由可扩展分辨率的离散AR模型建模，而连续组件由仅37M参数的轻量残差扩散模块学习。与仅离散的VAR标记器相比，我们的混合方法将重建FID从2.11提高到0.30（在MJHQ-30K上），导致生成FID从7.85提高到5.38，改善了31%。HART在FID和CLIP分数上均优于最先进的扩散模型，吞吐量高4.5-7.7倍，MACs低6.9-13.4倍。我们的代码在<a href="https://github.com/mit-han-lab/hart" rel="external noopener nofollow" class="link-external link-https">此https URL</a>开源。
</details>

<details>
    <summary>Key points</summary>
    * 使用离散和连续组件的混合标记器
    * 用AR建模离散标记，用残差扩散建模连续标记
    * 实现高吞吐量和低计算成本
</details>
</details>

---


<details>
<summary><b> SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image And Video Generation</b></summary>

* **Authors:** Jaehong Yoon, Shoubin Yu, Vaidehi Patil, Huaxiu Yao, Mohit Bansal
* **arXiv ID:** 2410.12761
* **One-liner:** Proposed SAFREE, a training-free method for safe text-to-image and text-to-video generation without altering model weights.
* **Published in:** arxiv (16 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.12761) | [[PDF]](https://arxiv.org/pdf/2410.12761) | [[Code]](https://github.com/jaehong31/SAFREE)

> **核心创新**
> 在文本嵌入中检测有毒概念子空间并引导提示远离，结合自验证过滤和自适应重注意机制以平衡安全性和质量。

<details>
    <summary>Abstract</summary>
    扩散模型的最新进展显著提升了生成高质量图像和视频的能力，但也增加了产生不安全内容的风险。现有的基于遗忘/编辑的安全生成方法从模型中移除有害概念，但面临几个挑战：(1) 它们无法在不训练的情况下即时移除有害概念。(2) 它们的安全生成能力依赖于收集的训练数据。(3) 它们改变模型权重，可能降低与有毒概念无关内容的质量。为解决这些问题，我们提出了SAFREE，一种新颖的、无需训练的安全文本到图像和文本到视频方法，不改变模型权重。具体来说，我们在文本嵌入空间中检测与一组有毒概念对应的子空间，并将提示嵌入引导远离该子空间，从而过滤有害内容同时保留预期语义。为平衡过滤毒性与保留安全概念之间的权衡，SAFREE引入了一种新颖的自验证过滤机制，在应用过滤嵌入时动态调整去噪步骤。此外，我们在扩散潜在空间中整合自适应重注意机制，以在像素级别选择性削弱与有毒概念相关的特征影响。最终，SAFREE确保一致的安全检查，保持输出的保真度、质量和安全性。与无需训练的基线相比，SAFREE在文本到图像生成中抑制不安全内容方面达到最先进性能，并在保持高质量图像的同时有效过滤目标概念。它还在与基于训练的方法比较中显示出竞争性结果。我们将SAFREE扩展到各种文本到图像主干和文本到视频任务，展示了其灵活性和泛化能力。SAFREE为安全视觉生成提供了鲁棒且可适应的保障。
</details>

<details>
    <summary>Key points</summary>
    * 文本嵌入空间中有毒概念子空间检测
    * 引导提示嵌入远离有毒子空间
    * 自验证过滤机制用于动态去噪调整
    * 扩散潜在空间中的自适应重注意机制
</details>
</details>

---


<details>
<summary><b> Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens</b></summary>

* **Authors:** Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, Yonglong Tian
* **arXiv ID:** 2410.13863
* **One-liner:** Introduced Fluid, a random-order autoregressive model on continuous tokens, achieving state-of-the-art zero-shot FID and GenEval scores.
* **Published in:** arxiv (17 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.13863) | [[PDF]](https://arxiv.org/pdf/2410.13863) | [[Code]]()

> **核心创新**
> 实证表明连续令牌和随机生成顺序改善了文本到图像生成中的扩展和性能。

<details>
    <summary>Abstract</summary>
    在视觉领域，自回归模型的扩展并未像在大语言模型中那样带来显著益处。在本工作中，我们在文本到图像生成的背景下研究这一扩展问题，重点关注两个关键因素：模型是否使用离散或连续令牌，以及令牌是否使用BERT或GPT类Transformer架构以随机或固定光栅顺序生成。我们的实证结果表明，尽管所有模型在验证损失方面都能有效扩展，但它们的评估性能——通过FID、GenEval分数和视觉质量衡量——呈现不同趋势。基于连续令牌的模型比使用离散令牌的模型实现显著更好的视觉质量。此外，生成顺序和注意机制显著影响GenEval分数：随机顺序模型相比光栅顺序模型实现明显更好的GenEval分数。受这些发现启发，我们训练了Fluid，一种在连续令牌上的随机顺序自回归模型。Fluid 10.5B模型在MS-COCO 30K上实现了6.16的新最先进零样本FID，并在GenEval基准上达到0.69的总分。我们希望我们的发现和结果能鼓励未来努力进一步弥合视觉和语言模型之间的扩展差距。
</details>

<details>
    <summary>Key points</summary>
    * 离散与连续令牌及生成顺序的调查
    * 在连续令牌上训练随机顺序自回归Fluid模型
    * 在MS-COCO上实现6.16零样本FID和GenEval上0.69的最先进性能
</details>
</details>

---


<details>
<summary><b> Synergistic Dual Spatial-aware Generation of Image-to-Text and Text-to-Image</b></summary>

* **Authors:** Yu Zhao, Hao Fei, Xiangtai Li, Libo Qin, Jiayi Ji, Hongyuan Zhu, Meishan Zhang, Min Zhang, Jianguo Wei
* **arXiv ID:** 2410.15312
* **One-liner:** Developed a dual learning framework with 3D scene graphs for spatial image-to-text and text-to-image tasks, improving spatial understanding.
* **Published in:** arxiv (20 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.15312) | [[PDF]](https://arxiv.org/pdf/2410.15312) | [[Code]]()

> **核心创新**
> 使用空间对偶离散扩散利用中间3D特征指导困难过程，增强任务间相互益处。

<details>
    <summary>Abstract</summary>
    在视觉空间理解领域，空间图像到文本和空间文本到图像是两个以对偶形式出现的基本任务。由于3D空间特征建模的困难，现有的独立SI2T或ST2I方法在空间理解方面表现不完美。在本工作中，我们考虑在对偶学习框架下共同建模SI2T和ST2I。在对偶框架中，我们提出使用新颖的3D场景图表示来表示3D空间场景特征，该表示可以共享并有益于两个任务。进一步，受3D到图像和3D到文本过程在ST2I和SI2T中对称存在的直觉启发，我们提出了空间对偶离散扩散框架，该框架利用3D到X过程的中间特征来指导困难的X到3D过程，从而使整体ST2I和SI2T相互受益。在视觉空间理解数据集VSD上，我们的系统显著优于主流T2I和I2T方法。进一步的深入分析揭示了我们的对偶学习策略如何推进性能。
</details>

<details>
    <summary>Key points</summary>
    * SI2T和ST2I的对偶学习框架
    * 空间特征的3D场景图表示
    * 空间对偶离散扩散框架
    * 利用3D→X过程指导X→3D过程
</details>
</details>

---


<details>
<summary><b> Progressive Compositionality in Text-to-Image Generative Models</b></summary>

* **Authors:** Evans Xu Han, Linghao Jin, Xiaofeng Liu, Paul Pu Liang
* **arXiv ID:** 2410.16719
* **One-liner:** Created ConPair dataset using LLMs and VQA for contrastive learning, and proposed EvoGen curriculum to improve compositional T2I generation.
* **Published in:** arxiv (22 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.16719) | [[PDF]](https://arxiv.org/pdf/2410.16719) | [[Code]](https://github.com/evansh666/EvoGen)

> **核心创新**
> 自动策划高质量对比图像对并应用多阶段课程学习以应对组合挑战。

<details>
    <summary>Abstract</summary>
    尽管扩散模型在文本到图像合成方面表现出色，但它们常常难以理解对象和属性之间的组合关系，尤其是在复杂设置中。现有解决方案通过优化交叉注意机制或从语义变化最小的标题对中学习来应对这些挑战。然而，我们能否生成高质量的复杂对比图像，使扩散模型能直接基于视觉表示进行区分？在本工作中，我们利用大语言模型来组合现实、复杂的场景，并利用视觉问答系统和扩散模型自动策划一个对比数据集ConPair，包含15k对高质量对比图像。这些对具有最小的视觉差异，并覆盖广泛的属性类别，尤其是复杂和自然场景。为从这些错误案例（即硬负样本图像）中有效学习，我们提出了EvoGen，一种用于扩散模型对比学习的新多阶段课程。通过在广泛组合场景上的大量实验，我们在组合T2I基准上展示了所提框架的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 利用LLM和VQA组合和策划ConPair数据集
    * 提出EvoGen多阶段课程进行对比学习
    * 专注于硬负样本图像以有效学习
</details>
</details>

---


<details>
<summary><b> FairQueue: Rethinking Prompt Learning for Fair Text-to-Image Generation</b></summary>

* **Authors:** Christopher T.H Teo, Milad Abdollahzadeh, Xinda Ma, Ngai-man Cheung
* **arXiv ID:** 2410.18615
* **One-liner:** Identified quality degradation in prompt learning for fair T2I generation and proposed Prompt Queuing and Attention Amplification to improve it.
* **Published in:** arxiv (24 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.18615) | [[PDF]](https://arxiv.org/pdf/2410.18615) | [[Code]](https://github.com/Bearwithchris/FairQueue_Code)

> **核心创新**
> 分析交叉注意图以揭示异常，并引入方法以在保持公平性的同时增强生成质量。

<details>
    <summary>Abstract</summary>
    最近，提示学习已成为公平文本到图像生成的最先进方法。具体来说，这种方法利用现成的参考图像为每个目标敏感属性学习包容性提示，从而实现公平图像生成。在本工作中，我们首先揭示这种基于提示学习的方法导致样本质量下降。我们的分析表明，该方法的训练目标——旨在对齐学习提示和参考图像的嵌入差异——可能次优，导致学习提示的扭曲和生成图像的退化。为进一步证实这一主张，作为我们的主要贡献，我们深入探究T2I模型的去噪子网络，通过分析交叉注意图来追踪这些学习提示的影响。在我们的分析中，我们提出了一种新颖的提示切换分析：I2H和H2I。此外，我们提出了交叉注意图的新定量表征。我们的分析揭示了早期去噪步骤中的异常，这些异常延续了不适当的全局结构，导致生成样本的退化。基于分析中的见解，我们提出了两个想法：(i) 提示排队和(ii) 注意放大以解决质量问题。在广泛tSA上的大量实验结果表明，我们提出的方法在图像生成质量上优于最先进方法，同时实现竞争性公平性。更多资源请访问FairQueue项目网站：此HTTPS URL。
</details>

<details>
    <summary>Key points</summary>
    * 交叉注意图和提示切换分析
    * 提出提示排队和注意放大
    * 交叉注意异常的定量表征
</details>
</details>

---


<details>
<summary><b> Diff-Instruct++: Training One-step Text-to-image Generator Model to Align with Human Preferences</b></summary>

* **Authors:** Weijian Luo
* **arXiv ID:** 2410.18881
* **One-liner:** Introduced Diff-Instruct++ (DI++), a fast-converging, image data-free method for aligning one-step T2I generators with human preferences.
* **Published in:** arxiv (24 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.18881) | [[PDF]](https://arxiv.org/pdf/2410.18881) | [[Code]](https://github.com/pkulwj1994/diff_instruct_pp)

> **核心创新**
> 将对齐问题表述为最大化人类奖励与KL散度，表明CFG用于扩散蒸馏等同于RLHF。

<details>
    <summary>Abstract</summary>
    一步文本到图像生成器模型具有快速推理效率、灵活架构和最先进生成性能等优势。在本文中，我们首次研究将一步生成器模型与人类偏好对齐的问题。受使用人类反馈的强化学习成功启发，我们将对齐问题表述为最大化期望人类奖励函数，同时添加积分Kullback-Leibler散度项以防止生成器偏离。通过克服技术挑战，我们引入了Diff-Instruct++，第一种快速收敛且无需图像数据的人类偏好对齐方法，用于一步文本到图像生成器。我们还引入了新颖的理论见解，表明使用CFG进行扩散蒸馏实际上是在用DI++进行RLHF。这一有趣发现为未来涉及CFG的研究带来了理解和潜在贡献。在实验部分，我们使用DI++对齐了基于UNet和DiT的一步生成器，这些生成器使用Stable Diffusion 1.5和PixelArt-α作为参考扩散过程。得到的基于DiT的一步文本到图像模型在COCO验证提示数据集上实现了6.19的高美学分数和1.24的图像奖励。它还实现了28.48的领先人类偏好分数，优于其他开源模型如Stable Diffusion XL、DMD2、SD-Turbo以及PixelArt-α。理论贡献和实证证据均表明DI++是一种强大的人类偏好对齐方法，用于一步文本到图像模型。论文主页是此HTTPS URL。
</details>

<details>
    <summary>Key points</summary>
    * 用人类奖励和KL散度表述对齐问题
    * 开发Diff-Instruct++方法
    * 将CFG与RLHF关联的理论见解
    * 应用于基于UNet和DiT的一步生成器
</details>
</details>

---


<details>
<summary><b> Kandinsky 3: Text-to-Image Synthesis for Multifunctional Generative Framework</b></summary>

* **Authors:** Vladimir Arkhipkin, Viacheslav Vasilev, Andrei Filatov, Igor Pavlov, Julia Agafonova, Nikolai Gerasimenko, Anna Averchenkova, Evelina Mironova, Anton Bukashkin, Konstantin Kulikov, Andrey Kuznetsov, Denis Dimitrov
* **arXiv ID:** 2410.21061
* **One-liner:** Presented Kandinsky 3, a latent diffusion-based T2I model with high quality and adaptability for various generation tasks.
* **Published in:** arxiv (28 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.21061) | [[PDF]](https://arxiv.org/pdf/2410.21061) | [[Code]](https://github.com/ai-forever/Kandinsky-3)

> **核心创新**
> 扩展基础模型用于修复、外绘、图像融合、I2V、T2V，并创建蒸馏版本以实现更快推理。

<details>
    <summary>Abstract</summary>
    文本到图像扩散模型因引入图像操作方法（如编辑、图像融合、修复等）而流行。同时，图像到视频和文本到视频模型也构建在T2I模型之上。我们提出了Kandinsky 3，一种基于潜在扩散的新颖T2I模型，实现了高质量和照片级真实感。新架构的关键特征是其适应多种生成任务类型的简单性和效率。我们扩展基础T2I模型用于各种应用，并创建一个多功能生成系统，包括文本引导修复/外绘、图像融合、文本-图像融合、图像变体生成、I2V和T2V生成。我们还提出了T2I模型的蒸馏版本，在反向过程的4步中评估推理，不降低图像质量，且比基础模型快3倍。我们部署了一个用户友好的演示系统，所有功能可在公共领域测试。此外，我们发布了Kandinsky 3及扩展模型的源代码和检查点。人类评估显示，Kandinsky 3在开源生成系统中展示了最高质量分数之一。
</details>

<details>
    <summary>Key points</summary>
    * 开发Kandinsky 3潜在扩散模型
    * 扩展到多种应用
    * 创建蒸馏版本以实现高效推理
    * 部署用户友好演示系统
</details>
</details>

---


<details>
<summary><b> Diffusion Beats Autoregressive: An Evaluation of Compositional Generation in Text-to-Image Models</b></summary>

* **Authors:** Arash Marioriyad, Parham Rezaei, Mahdieh Soleymani Baghshah, Mohammad Hossein Rohban
* **arXiv ID:** 2410.22775
* **One-liner:** Evaluated compositional generation capabilities of FLUX and LlamaGen, finding FLUX comparable to DALL-E3 while LlamaGen lags behind diffusion models.
* **Published in:** arxiv (30 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.22775) | [[PDF]](https://arxiv.org/pdf/2410.22775) | [[Code]]()

> **核心创新**
> 使用T2I-CompBench评估模型，突出处理复杂组合的优缺点。

<details>
    <summary>Abstract</summary>
    文本到图像生成模型，如Stable Diffusion和DALL-E，在从文本描述生成高质量、真实和自然图像方面表现出卓越能力。然而，这些模型有时无法准确捕捉输入提示中的所有细节，特别是关于实体、属性和空间关系。当提示包含新颖或复杂组合时，这一问题更加突出，导致所谓的组合生成失败模式。最近，一种新的开源基于扩散的T2I模型FLUX被引入，展示了在高质量图像生成方面的强大性能。此外，自回归T2I模型如LlamaGen声称在视觉质量性能上与基于扩散的模型竞争。在本研究中，我们使用T2I-CompBench基准评估这些新引入模型与现有模型的组合生成能力。我们的发现揭示，作为普通自回归模型，LlamaGen在相同标准（如模型大小和推理时间）下，在组合生成任务上尚未与最先进的扩散模型相媲美。另一方面，开源基于扩散的模型FLUX表现出与最先进闭源模型DALL-E3相当组合生成能力。
</details>

<details>
    <summary>Key points</summary>
    * 使用T2I-CompBench基准进行评估
    * FLUX和LlamaGen与现有模型的比较
    * 关于FLUX竞争性能和LlamaGen局限性的发现
</details>
</details>

---


<details>
<summary><b> Image2Text2Image: A Novel Framework for Label-Free Evaluation of Image-to-Text Generation with Text-to-Image Diffusion Models</b></summary>

* **Authors:** Jia-Hong Huang, Hongyi Zhu, Yixian Shen, Stevan Rudinac, Evangelos Kanoulas
* **arXiv ID:** 2411.05706
* **One-liner:** Proposed Image2Text2Image framework for evaluating image captioning models using diffusion models without human references.
* **Published in:** arxiv (8 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.05706) | [[PDF]](https://arxiv.org/pdf/2411.05706) | [[Code]]()

> **核心创新**
> 利用T2I生成测量原始和生成图像之间的相似性，通过实验和人类评估验证。

<details>
    <summary>Abstract</summary>
    评估自动生成图像描述的质量是一项复杂任务，需要捕捉多个维度的指标，如语法性、覆盖度、准确性和真实性。尽管人类评估提供了宝贵见解，但其成本和时间消耗性带来限制。现有自动化指标如BLEU、ROUGE、METEOR和CIDEr试图填补这一空白，但它们常常与人类判断表现出弱相关性。为应对这一挑战，我们提出了一种新颖的评估框架Image2Text2Image，该框架利用扩散模型（如Stable Diffusion或DALL-E）进行文本到图像生成。在Image2Text2Image框架中，输入图像首先由选定的图像描述模型（用于评估）处理以生成文本描述。使用此生成描述，扩散模型然后创建新图像。通过比较从原始和生成图像中提取的特征，我们使用指定相似性度量来衡量它们的相似性。高相似性分数表明模型生成了忠实的文本描述，而低分数突出差异，揭示模型性能的潜在弱点。值得注意的是，我们的框架不依赖人类注释的参考描述，使其成为评估图像描述模型的有价值工具。大量实验和人类评估验证了我们提出的Image2Text2Image评估框架的有效性。代码和数据集将发布以支持社区进一步研究。
</details>

<details>
    <summary>Key points</summary>
    * 使用图像描述和T2I生成的评估框架
    * 原始和生成图像之间的相似性测量
    * 通过大量实验和人类评估进行验证
</details>
</details>

---


<details>
<summary><b> Region-Aware Text-to-Image Generation via Hard Binding and Soft Refinement</b></summary>

* **Authors:** Zhennan Chen, Yajie Li, Haofan Wang, Zhibo Chen, Zhengkai Jiang, Jun Li, Qian Wang, Jian Yang, Ying Tai
* **arXiv ID:** 2411.06558
* **One-liner:** Introduced RAG, a tuning-free method for regional-aware text-to-image generation with precise layout control and repainting capabilities.
* **Published in:** arxiv (10 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.06558) | [[PDF]](https://arxiv.org/pdf/2411.06558) | [[Code]](https://github.com/NJU-PCALab/RAG-Diffusion)

> **核心创新**
> 将多区域生成解耦为区域硬绑定和软细化，在不额外训练的情况下增强控制。

<details>
    <summary>Abstract</summary>
    区域提示或组合生成能够实现细粒度空间控制，因其在实际应用中的实用性而日益受到关注。然而，先前方法要么引入额外可训练模块，因此仅适用于特定模型，要么在交叉注意层中使用注意掩码操作分数图，导致区域数量增加时控制强度有限。为应对这些限制，我们提出了RAG，一种区域感知的文本到图像生成方法，以区域描述为条件实现精确布局组合。RAG将多区域生成解耦为两个子任务：单个区域构建（区域硬绑定）确保区域提示正确执行，以及区域间整体细节细化（区域软细化）消除视觉边界并增强相邻交互。此外，RAG新颖地使重绘可行，用户可以在最后生成中修改特定不满意区域，同时保持所有其他区域不变，而无需依赖额外修复模型。我们的方法无需调优，并可作为提示跟随属性的增强应用于其他框架。定量和定性实验表明，RAG在属性绑定和对象关系上优于先前无需调优方法。
</details>

<details>
    <summary>Key points</summary>
    * 区域感知生成方法
    * 解耦为区域硬绑定和区域软细化
    * 无需修复模型的重绘可行性
    * 作为增强应用于其他框架的适用性
</details>
</details>

---


<details>
<summary><b> JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation</b></summary>

* **Authors:** Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai yu, Liang Zhao, Yisong Wang, Jiaying Liu, Chong Ruan
* **arXiv ID:** 2411.07975
* **One-liner:** JanusFlow unifies image understanding and generation in a single model.
* **Published in:** arxiv (12 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.07975) | [[PDF]](https://arxiv.org/pdf/2411.07975) | [[Code]](https://github.com/deepseek-ai/Janus)

> **核心创新**
> 将自回归语言模型与整流流集成，用于统一视觉语言建模。

<details>
    <summary>Abstract</summary>
    我们提出了JanusFlow，一个强大的框架，将图像理解与生成统一在单一模型中。JanusFlow引入了一种极简架构，将自回归语言模型与整流流（一种生成建模中的先进方法）相结合。我们的关键发现表明，整流流可以在大型语言模型框架内直接训练，无需复杂的架构修改。为了进一步提升统一模型的性能，我们采用了两个关键策略：（i）解耦理解与生成编码器，以及（ii）在统一训练中对齐它们的表示。大量实验显示，JanusFlow在各自领域内达到或超越了专用模型的性能，同时在标准基准测试中显著优于现有统一方法。这项工作代表了向更高效、多功能的视觉语言模型迈出的一步。
</details>

<details>
    <summary>Key points</summary>
    * 引入带有整流流的极简架构
    * 解耦理解与生成编码器
    * 在统一训练中对齐表示
</details>
</details>

---


<details>
<summary><b> Visual question answering based evaluation metrics for text-to-image generation</b></summary>

* **Authors:** Mizuki Miyamoto, Ryugo Morita, Jinjia Zhou
* **arXiv ID:** 2411.10183
* **One-liner:** Proposes new evaluation metrics for detailed text-image alignment.
* **Published in:** arxiv (15 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.10183) | [[PDF]](https://arxiv.org/pdf/2411.10183) | [[Code]]()

> **核心创新**
> 使用ChatGPT和VQA评估每个个体对象的对齐。

<details>
    <summary>Abstract</summary>
    文本到图像生成和文本引导的图像编辑在图像生成任务领域受到了广泛关注。然而，这些任务的主流评估方法难以评估输入文本中的所有信息是否准确反映在生成图像中，且主要关注输入文本与生成图像的整体对齐。本文提出了新的评估指标，用于评估输入文本与生成图像中每个个体对象的对齐。首先，根据输入文本，利用ChatGPT为生成图像生成问题。之后，我们使用视觉问答（VQA）来测量生成图像与输入文本的相关性，从而实现对对齐的更详细评估，优于现有方法。此外，我们使用无参考图像质量评估（NR-IQA）来评估文本-图像对齐以及生成图像的质量。实验结果表明，我们提出的评估方法是优越的指标，能够同时评估更精细的文本-图像对齐和图像质量，并允许调整这些比例。
</details>

<details>
    <summary>Key points</summary>
    * 利用ChatGPT为图像生成问题
    * 使用VQA进行详细对齐评估
    * 结合NR-IQA进行图像质量评估
</details>
</details>

---


<details>
<summary><b> Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding</b></summary>

* **Authors:** Huming Qiu, Guanxu Chen, Mi Zhang, Xiaohan Zhang, Xiaoyu You, Min Yang
* **arXiv ID:** 2411.10329
* **One-liner:** Enhances T2I model safety by sanitizing prompt embeddings.
* **Published in:** arxiv (15 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.10329) | [[PDF]](https://arxiv.org/pdf/2411.10329) | [[Code]]()

> **核心创新**
> 可解释框架，用于识别和减轻文本提示中的危害内容。

<details>
    <summary>Abstract</summary>
    近年来，文本到图像（T2I）生成模型在生成与文本描述对齐的高质量图像方面取得了显著进展。然而，这些模型也面临不安全生成的风险，可能产生违反使用政策的危害内容，如露骨材料。现有的安全生成方法通常通过从视觉表示中擦除不期望的概念来抑制不当内容，但忽略了文本表示的净化。尽管这些方法在一定程度上帮助减轻了滥用风险，但在处理对抗攻击时，其鲁棒性仍然不足。鉴于输入文本与输出图像之间的语义一致性是T2I模型的核心要求，我们识别出文本表示可能是不安全生成的主要来源。为此，我们提出了嵌入净化器（ES），通过净化提示嵌入中的不当概念来增强T2I模型的安全性。据我们所知，ES是第一个可解释的安全生成框架，为提示中的每个令牌分配分数以指示其潜在危害性。此外，ES采用即插即用的模块化设计，提供与各种T2I模型和其他安全措施的兼容性。在五个提示基准测试上的评估显示，ES优于十一个现有安全基线，在保持高质量图像生成的同时，实现了最先进的鲁棒性。
</details>

<details>
    <summary>Key points</summary>
    * 为提示中的令牌分配危害性分数
    * 采用即插即用的模块化设计
    * 在鲁棒性上优于现有安全措施
</details>
</details>

---


<details>
<summary><b> High-Resolution Image Synthesis via Next-Token Prediction</b></summary>

* **Authors:** Dengsheng Chen, Jie Hu, Tiezhu Yue, Xiaoming Wei, Enhua Wu
* **arXiv ID:** 2411.14808
* **One-liner:** Achieves state-of-the-art high-resolution image synthesis via next-token prediction.
* **Published in:** arxiv (22 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.14808) | [[PDF]](https://arxiv.org/pdf/2411.14808) | [[Code]]()

> **核心创新**
> 自回归模型，用于生成高达4K任意分辨率的逼真图像。

<details>
    <summary>Abstract</summary>
    最近，自回归模型在类条件图像生成中表现出卓越性能。然而，将下一令牌预测应用于高分辨率文本到图像生成仍未被充分探索。在本文中，我们介绍了\textbf{D-JEPA$\cdot$T2I}，一种基于连续令牌的自回归模型，结合了架构和训练策略的创新，以生成高达4K任意分辨率的高质量、逼真图像。在架构上，我们采用去噪联合嵌入预测架构（D-JEPA），同时利用多模态视觉变换器有效整合文本和视觉特征。此外，我们引入流匹配损失与提出的视觉旋转位置嵌入（VoPE），以实现连续分辨率学习。在训练策略方面，我们提出了一种数据反馈机制，基于统计分析动态调整采样过程，并使用在线学习批评模型。这鼓励模型超越其舒适区，减少对已掌握场景的冗余训练，并迫使其处理生成质量较差的更具挑战性案例。我们首次通过下一令牌预测实现了最先进的高分辨率图像合成。
</details>

<details>
    <summary>Key points</summary>
    * 结合D-JEPA架构与多模态视觉变换器
    * 使用流匹配损失和视觉旋转位置嵌入
    * 实现数据反馈机制以进行自适应训练
</details>
</details>

---


<details>
<summary><b> Automatic Evaluation for Text-to-image Generation: Task-decomposed Framework, Distilled Training, and Meta-evaluation Benchmark</b></summary>

* **Authors:** Rong-Cheng Tu, Zi-Ao Ma, Tian Lan, Yuehao Zhao, Heyan Huang, Xian-Ling Mao
* **arXiv ID:** 2411.15488
* **One-liner:** Distills GPT-4o's evaluation capabilities into a smaller open-source MLLM.
* **Published in:** arxiv (23 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.15488) | [[PDF]](https://arxiv.org/pdf/2411.15488) | [[Code]](https://github.com/maziao/T2I-Eval)

> **核心创新**
> 任务分解框架，用于高效和可扩展的图像质量评估。

<details>
    <summary>Abstract</summary>
    受扩散模型显著进展的推动，文本到图像生成取得了重大进步，对生成图像的自动质量评估产生了迫切需求。当前最先进的自动评估方法严重依赖多模态大语言模型（MLLMs），特别是像GPT-4o这样的强大商业模型。尽管这些模型非常有效，但其高昂成本限制了大规模评估的可扩展性。采用开源MLLMs是一种替代方案；然而，由于在处理多模态数据方面与商业MLLMs相比存在显著限制，其性能不足。为了解决这些问题，我们首先提出一个基于GPT-4o的任务分解评估框架，自动构建新的训练数据集，其中复杂评估任务被解耦为更简单的子任务，有效降低学习复杂度。基于此数据集，我们设计创新训练策略，将GPT-4o的评估能力有效蒸馏到一个7B开源MLLM，MiniCPM-V-2.6。此外，为了可靠且全面地评估先前工作和我们提出的模型，我们手动标注了一个元评估基准，包括生成图像的思维链解释和质量分数。实验结果表明，我们蒸馏的开源MLLM显著优于当前最先进的GPT-4o基线VIEScore，在与人类判断的Spearman和Kendall相关性上提高了超过4.6%。
</details>

<details>
    <summary>Key points</summary>
    * 将评估解耦为更简单的子任务
    * 使用创新策略训练7B MLLM
    * 创建带有人工标注的元评估基准
</details>
</details>

---


<details>
<summary><b> Interactive Visual Assessment for Text-to-Image Generation Models</b></summary>

* **Authors:** Xiaoyue Mi, Fan Tang, Juan Cao, Qiang Sheng, Ziyao Huang, Peng Li, Yang Liu, Tong-Yee Lee
* **arXiv ID:** 2411.15509
* **One-liner:** Facilitates dynamic interactive assessment of T2I models to uncover failures.
* **Published in:** arxiv (23 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.15509) | [[PDF]](https://arxiv.org/pdf/2411.15509) | [[Code]]()

> **核心创新**
> LLM驱动的框架，用于协作人机评估与自适应输入生成。

<details>
    <summary>Abstract</summary>
    视觉生成模型在计算机图形应用中取得了显著进展，但在实际部署中仍面临重大挑战。当前视觉生成任务的评估方法通常遵循孤立的三阶段框架：测试输入收集、模型输出生成和用户评估。这些模式存在固定覆盖、演化难度和数据泄露风险，限制了其在全面评估日益复杂生成模型方面的有效性。为了解决这些限制，我们提出了DyEval，一个LLM驱动的动态交互式视觉评估框架，促进人类与生成模型在文本到图像系统中的协作评估。DyEval具有直观的视觉界面，使用户能够交互式探索和分析模型行为，同时基于反馈自适应生成分层、细粒度和多样化的文本输入，以持续探测模型的能力边界。此外，为了提供可解释的分析以帮助用户进一步改进测试模型，我们开发了一个上下文反思模块，挖掘测试输入的失败触发器，并利用LLM的逻辑推理能力反映模型潜在失败模式，支持深入分析。定性和定量实验表明，DyEval能有效帮助用户识别最多2.56倍的生成失败，比传统方法更多，并揭示复杂和罕见的失败模式，如代词生成和特定文化背景生成问题。我们的框架为改进生成模型提供了宝贵见解，并对推进视觉生成系统在各种领域的可靠性和能力具有广泛意义。
</details>

<details>
    <summary>Key points</summary>
    * 具有直观视觉界面以支持用户交互
    * 生成分层和多样化的文本输入
    * 包括上下文反思模块以进行失败分析
</details>
</details>

---


<details>
<summary><b> ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting</b></summary>

* **Authors:** Chengyou Jia, Changliang Xia, Zhuohang Dang, Weijia Wu, Hangwei Qian, Minnan Luo
* **arXiv ID:** 2411.17176
* **One-liner:** Automates T2I generation steps to reduce user trial-and-error.
* **Published in:** arxiv (26 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.17176) | [[PDF]](https://arxiv.org/pdf/2411.17176) | [[Code]](https://github.com/chengyou-jia/ChatGen)

> **核心创新**
> 多阶段进化策略，用于自动化提示制作和模型配置。

<details>
    <summary>Abstract</summary>
    尽管文本到图像（T2I）生成模型取得了显著进展，但在实际场景中，用户常常面临试错挑战。这一挑战源于繁琐步骤的复杂性和不确定性，如制作合适的提示、选择适当的模型和配置特定参数，使用户不得不进行劳动密集型尝试以获得期望图像。本文提出自动T2I生成，旨在自动化这些繁琐步骤，允许用户以自由聊天方式简单描述其需求。为了系统研究此问题，我们首先介绍了ChatGenBench，一个为自动T2I设计的新基准。它具有高质量配对数据和多样化自由风格输入，能够全面评估自动T2I模型在所有步骤上的表现。此外，认识到自动T2I是一个复杂的多步推理任务，我们提出了ChatGen-Evo，一种多阶段进化策略，逐步赋予模型必要的自动化技能。通过跨步骤准确性和图像质量的广泛评估，ChatGen-Evo显著提升了各种基线的性能。我们的评估还揭示了推进自动T2I的宝贵见解。我们所有的数据、代码和模型将在\url{<a href="https://chengyou-jia.github.io/ChatGen-Home" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}中提供。
</details>

<details>
    <summary>Key points</summary>
    * 引入ChatGenBench基准以进行评估
    * 提出ChatGen-Evo并逐步获取技能
    * 在步骤准确性和图像质量上提升性能
</details>
</details>

---


<details>
<summary><b> Interleaved Scene Graphs for Interleaved Text-and-Image Generation Assessment</b></summary>

* **Authors:** Dongping Chen, Ruoxi Chen, Shu Pu, Zhaoyi Liu, Yanru Wu, Caixi Chen, Benlin Liu, Yue Huang, Yao Wan, Pan Zhou, Ranjay Krishna
* **arXiv ID:** 2411.17188
* **One-liner:** Presents evaluation framework for interleaved text-and-image generation consistency.
* **Published in:** arxiv (26 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.17188) | [[PDF]](https://arxiv.org/pdf/2411.17188) | [[Code]](https://github.com/Dongping-Chen/ISG)

> **核心创新**
> 基于场景图的评估，用于多模态内容的一致性和准确性。

<details>
    <summary>Abstract</summary>
    许多现实世界用户查询（例如“如何制作蛋炒饭？”）可以受益于能够生成带有伴随图像的文本步骤响应的系统，类似于食谱。设计用于生成交错文本和图像的模型在确保这些模态内部和跨模态的一致性方面面临挑战。为了解决这些挑战，我们提出了ISG，一个用于交错文本和图像生成的全面评估框架。ISG利用场景图结构捕捉文本和图像块之间的关系，在四个粒度级别上评估响应：整体、结构、块级和图像特定。这种多层次评估允许对一致性、连贯性和准确性进行细致评估，并提供可解释的问答反馈。与ISG一起，我们引入了一个基准ISG-Bench，涵盖8个类别和21个子类别的1,150个样本。该基准数据集包括复杂的语言-视觉依赖关系和黄金答案，以在视觉中心任务（如风格转移，当前模型的一个挑战领域）上有效评估模型。使用ISG-Bench，我们证明了最近的统一视觉语言模型在生成交错内容方面表现不佳。虽然组合方法结合了独立的语言和图像模型在整体级别上比统一模型提高了111%，但它们在块和图像级别的性能仍然不理想。为了促进未来工作，我们开发了ISG-Agent，一个基线代理，采用“计划-执行-精炼”管道调用工具，实现了122%的性能提升。
</details>

<details>
    <summary>Key points</summary>
    * 在整体、结构、块级和图像特定粒度上评估
    * 引入具有多样类别的ISG-Bench基准
    * 开发带计划-执行-精炼管道的ISG-Agent
</details>
</details>

---


<details>
<summary><b> Reward Incremental Learning in Text-to-Image Generation</b></summary>

* **Authors:** Maorong Wang, Jiafeng Mao, Xueting Wang, Toshihiko Yamasaki
* **arXiv ID:** 2411.17310
* **One-liner:** Addresses catastrophic forgetting in diffusion model fine-tuning for multiple rewards.
* **Published in:** arxiv (26 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.17310) | [[PDF]](https://arxiv.org/pdf/2411.17310) | [[Code]]()

> **核心创新**
> 奖励增量学习方法，用于稳定适应序列目标。

<details>
    <summary>Abstract</summary>
    去噪扩散模型的近期成功显著推进了文本到图像生成。尽管这些大规模预训练模型在通用图像合成中表现出色，但下游目标通常需要微调以满足特定标准，如美学或人类偏好。基于奖励梯度的策略在此背景下很有前景，但现有方法仅限于单奖励任务，限制了其在需要适应随时间增量引入的多个目标的现实场景中的适用性。在本文中，我们首先定义了这个更现实且未探索的问题，称为奖励增量学习（RIL），其中模型期望增量适应多个下游目标。此外，当模型适应不断出现的新目标时，我们观察到扩散模型微调中一种独特的灾难性遗忘形式，影响度量级和视觉结构级的图像质量。为了解决这一灾难性遗忘挑战，我们提出了奖励增量蒸馏（RID），一种以最小计算开销减轻遗忘的方法，实现在序列奖励任务中的稳定性能。实验结果表明，RID在RIL场景中实现了一致、高质量生成的有效性。我们工作的源代码将在接受后公开提供。
</details>

<details>
    <summary>Key points</summary>
    * 定义奖励增量学习（RIL）问题
    * 提出奖励增量蒸馏（RID）以减轻遗忘
    * 确保跨任务的一致高质量生成
</details>
</details>

---


<details>
<summary><b> Type-R: Automatically Retouching Typos for Text-to-Image Generation</b></summary>

* **Authors:** Wataru Shimoda, Naoto Inoue, Daichi Haraguchi, Hayato Mitani, Seiichi Uchida, Kota Yamaguchi
* **arXiv ID:** 2411.18159
* **One-liner:** Improves text rendering accuracy in generated images through post-processing.
* **Published in:** arxiv (27 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.18159) | [[PDF]](https://arxiv.org/pdf/2411.18159) | [[Code]](https://github.com/CyberAgentAILab/Type-R)

> **核心创新**
> 修饰管道，用于纠正T2I输出中的排版错误。

<details>
    <summary>Abstract</summary>
    尽管最近的文本到图像模型可以从反映详细指令的文本提示生成逼真图像，但在准确渲染图像中的文字方面仍面临重大挑战。在本文中，我们提出在后处理管道中修饰错误的文本渲染。我们的方法称为Type-R，识别生成图像中的排版错误，擦除错误文本，为缺失单词重新生成文本框，并最终纠正渲染单词中的错别字。通过大量实验，我们显示Type-R与最新文本到图像模型（如Stable Diffusion或Flux）结合，实现了最高的文本渲染准确性，同时保持图像质量，并且在平衡文本准确性和图像质量方面优于专注于文本生成的基线。
</details>

<details>
    <summary>Key points</summary>
    * 识别并擦除错误文本
    * 为缺失单词重新生成文本框
    * 纠正错别字同时保持图像质量
</details>
</details>

---


<details>
<summary><b> Enhancing MMDiT-Based Text-to-Image Models for Similar Subject Generation</b></summary>

* **Authors:** Tianyi Wei, Dongdong Chen, Yifan Zhou, Xingang Pan
* **arXiv ID:** 2411.18301
* **One-liner:** Enhanced MMDiT by mitigating subject neglect in multi-subject prompts.
* **Published in:** arxiv (27 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.18301) | [[PDF]](https://arxiv.org/pdf/2411.18301) | [[Code]](https://github.com/wtybest/EnMMDiT)

> **核心创新**
> 提出了测试时优化与损失函数和策略，以解决MMDiT中的模糊性问题。

<details>
    <summary>Abstract</summary>
    作为文本到图像模型的前沿技术，最新的多模态扩散变换器（MMDiT）在很大程度上缓解了先前模型存在的许多生成问题。然而，我们发现，当输入文本提示包含多个语义或外观相似的主题时，它仍然存在主题忽略或混淆的问题。我们识别了MMDiT架构中导致此问题的三种可能模糊性：块间模糊性、文本编码器模糊性和语义模糊性。为了解决这些问题，我们提出在早期去噪步骤中通过测试时优化来动态修复模糊潜在表示。具体而言，我们设计了三种损失函数：块对齐损失、文本编码器对齐损失和重叠损失，每种都针对缓解这些模糊性而定制。尽管有显著改进，我们观察到在生成多个相似主题时，语义模糊性仍然存在，因为重叠损失提供的指导不够明确。因此，我们进一步提出了重叠在线检测和回退到起始采样策略来缓解该问题。在一个新构建的具有挑战性的相似主题数据集上的实验结果验证了我们方法的有效性，显示出优于现有方法的生成质量和更高的成功率。我们的代码将在<a href="https://github.com/wtybest/EnMMDiT" rel="external noopener nofollow" class="link-external link-https">此https URL</a>提供。
</details>

<details>
    <summary>Key points</summary>
    * 识别了块间、文本编码器和语义模糊性
    * 设计了块对齐损失、文本编码器对齐损失和重叠损失
    * 引入了重叠在线检测和回退到起始采样策略
</details>
</details>

---


<details>
<summary><b> All Seeds Are Not Equal: Enhancing Compositional Text-to-Image Generation with Reliable Random Seeds</b></summary>

* **Authors:** Shuangqi Li, Hieu Le, Jingyi Xu, Mathieu Salzmann
* **arXiv ID:** 2411.18810
* **One-liner:** Improved compositional ability of text-to-image models by leveraging initial noise.
* **Published in:** arxiv (27 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.18810) | [[PDF]](https://arxiv.org/pdf/2411.18810) | [[Code]]()

> **核心创新**
> 挖掘了可靠的初始噪声案例以创建用于微调的训练集。

<details>
    <summary>Abstract</summary>
    文本到图像扩散模型在从任意文本提示生成逼真图像方面展现出卓越能力。然而，对于组合提示如'两只狗'或'碗右侧的企鹅'，它们往往产生不一致的结果。理解这些不一致性对于可靠的图像生成至关重要。在本文中，我们强调了初始噪声在这些不一致性中的重要作用，其中某些噪声模式对于组合提示比其他模式更可靠。我们的分析揭示，不同的初始随机种子倾向于引导模型将对象放置在图像的不同区域，可能遵循与种子相关的特定相机角度和图像构图模式。为了改进模型的组合能力，我们提出了一种挖掘这些可靠案例的方法，从而构建了一个无需手动注释的生成图像精选训练集。通过在生成的图像上对文本到图像模型进行微调，我们显著增强了它们的组合能力。对于数值组合，我们观察到Stable Diffusion和PixArt-α分别相对提高了29.3%和19.5%。空间组合的增益更大，Stable Diffusion为60.7%，PixArt-α为21.1%。
</details>

<details>
    <summary>Key points</summary>
    * 分析了初始噪声在对象放置不一致性中的作用
    * 从生成图像中构建了无需手动注释的精选训练集
    * 通过微调模型增强了数值和空间组合能力
</details>
</details>

---


<details>
<summary><b> Switti: Designing Scale-Wise Transformers for Text-to-Image Synthesis</b></summary>

* **Authors:** Anton Voronov, Denis Kuznedelev, Mikhail Khoroshikh, Valentin Khrulkov, Dmitry Baranchuk
* **arXiv ID:** 2412.01819
* **One-liner:** Developed Switti, a fast and high-quality scale-wise transformer for T2I generation.
* **Published in:** arxiv (2 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.01819) | [[PDF]](https://arxiv.org/pdf/2412.01819) | [[Code]](https://github.com/yandex-research/switti)

> **核心创新**
> 引入了非因果架构并优化了指导以提高效率和品质。

<details>
    <summary>Abstract</summary>
    本工作提出了Switti，一种用于文本到图像生成的尺度级变换器。我们首先将现有的下一尺度预测自回归（AR）架构适配到T2I生成中，并在过程中调查和缓解训练稳定性问题。接下来，我们认为尺度级变换器不需要因果性，并提出了一个非因果对应物，实现了约21%更快的采样和更低的内存使用，同时获得了略好的生成质量。此外，我们揭示了在高分辨率尺度上的无分类器指导通常是不必要的，甚至可能降低性能。通过在这些尺度上禁用指导，我们实现了额外的约32%采样加速，并改善了细粒度细节的生成。广泛的人类偏好研究和自动化评估显示，Switti优于现有的T2I AR模型，并与最先进的T2I扩散模型竞争，同时速度提高了7倍。
</details>

<details>
    <summary>Key points</summary>
    * 将下一尺度预测AR架构适配到T2I
    * 提出了非因果变换器以实现更快采样和更低内存
    * 在高分辨率尺度上禁用了无分类器指导以改善细节
</details>
</details>

---


<details>
<summary><b> Cross-Attention Head Position Patterns Can Align with Human Visual Concepts in Text-to-Image Generative Models</b></summary>

* **Authors:** Jungwon Park, Jungmin Ko, Dongnam Byun, Jangwon Suh, Wonjong Rhee
* **arXiv ID:** 2412.02237
* **One-liner:** Advanced interpretability of cross-attention layers in diffusion models.
* **Published in:** arxiv (3 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.02237) | [[PDF]](https://arxiv.org/pdf/2412.02237) | [[Code]]()

> **核心创新**
> 引入了头部相关性向量用于概念对齐和控制。

<details>
    <summary>Abstract</summary>
    最近的文本到图像扩散模型利用交叉注意力层，这些层已被有效用于增强一系列视觉生成任务。然而，我们对交叉注意力层的理解仍然有限。在本研究中，我们引入了一种扩散模型的机制可解释性方法，通过构建与人类指定视觉概念对齐的头部相关性向量（HRVs）。给定视觉概念的HRV长度等于交叉注意力头部的总数，每个元素代表相应头部对于给定视觉概念的重要性。为了验证HRVs作为可解释特征，我们开发了一种有序弱化分析，证明了它们的有效性。此外，我们提出了概念强化和概念调整方法，并将它们应用于增强三个视觉生成任务。我们的结果显示，HRVs可以减少图像生成中多义词的误解，成功修改图像编辑中的五个挑战性属性，并缓解多概念生成中的灾难性忽略。总体而言，我们的工作在理解交叉注意力层方面提供了进展，并引入了在头部级别精细控制这些层的新方法。
</details>

<details>
    <summary>Key points</summary>
    * 构建了HRVs以表示视觉概念重要性
    * 开发了有序弱化分析用于验证
    * 应用了概念强化和调整到生成任务
</details>
</details>

---


<details>
<summary><b> DynamicControl: Adaptive Condition Selection for Improved Text-to-Image Generation</b></summary>

* **Authors:** Qingdong He, Jinlong Peng, Pengcheng Xu, Boyuan Jiang, Xiaobin Hu, Donghao Luo, Yong Liu, Yabiao Wang, Chengjie Wang, Xiangtai Li, Jiangning Zhang
* **arXiv ID:** 2412.03255
* **One-liner:** Proposed DynamicControl for adaptive multi-condition T2I generation.
* **Published in:** arxiv (4 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.03255) | [[PDF]](https://arxiv.org/pdf/2412.03255) | [[Code]]()

> **核心创新**
> 集成了双循环控制器和MLLM用于条件优化。

<details>
    <summary>Abstract</summary>
    为了增强文本到图像扩散模型的可控性，当前的ControlNet类模型探索了各种控制信号来指定图像属性。然而，现有方法要么处理条件效率低下，要么使用固定数量的条件，这没有完全解决多个条件及其潜在冲突的复杂性。这强调了需要创新方法来有效管理多个条件，以实现更可靠和详细的图像合成。为了解决这个问题，我们提出了一个新框架DynamicControl，它支持动态组合多样控制信号，允许自适应选择不同数量和类型的条件。我们的方法从一个双循环控制器开始，该控制器利用预训练的条件生成模型和判别模型，为所有输入条件生成初始真实分数排序。该控制器评估提取条件与输入条件之间的相似性，以及与源图像的像素级相似性。然后，我们集成一个多模态大语言模型（MLLM）来构建一个高效的条件评估器。该评估器基于双循环控制器的分数排名优化条件排序。我们的方法联合优化MLLMs和扩散模型，利用MLLMs的推理能力促进多条件文本到图像（T2I）任务。最终排序的条件被输入到一个并行多控制适配器中，该适配器从动态视觉条件中学习特征图，并将它们集成以调制ControlNet，从而增强对生成图像的控制。通过定量和定性比较，DynamicControl在各种条件控制下，在可控性、生成质量和可组合性方面展示了优于现有方法的优越性。
</details>

<details>
    <summary>Key points</summary>
    * 使用双循环控制器进行初始条件评分
    * 利用MLLM优化条件排序
    * 应用并行多控制适配器进行特征集成
</details>
</details>

---


<details>
<summary><b> Safeguarding Text-to-Image Generation via Inference-Time Prompt-Noise Optimization</b></summary>

* **Authors:** Jiangweizhi Peng, Zhiwei Tang, Gaowen Liu, Charles Fleming, Mingyi Hong
* **arXiv ID:** 2412.03876
* **One-liner:** Introduced PNO to prevent unsafe image generation without training.
* **Published in:** arxiv (5 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.03876) | [[PDF]](https://arxiv.org/pdf/2412.03876) | [[Code]]()

> **核心创新**
> 优化了提示嵌入和噪声轨迹以实现安全性和对齐性。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）扩散模型因其基于文本提示生成高质量和多样化图像的能力而广受认可。然而，尽管最近有进展，这些模型仍然容易生成包含敏感或不适当内容的不安全图像，这可能对用户有害。当前防止扩散模型生成不适当图像的努力容易被绕过，并且容易受到对抗攻击。如何确保T2I模型与特定安全目标对齐仍然是一个重大挑战。在这项工作中，我们提出了一种新颖的、无需训练的方法，称为提示-噪声优化（PNO），以减轻不安全图像生成。我们的方法引入了一个新颖的优化框架，利用连续提示嵌入和采样过程中注入的噪声轨迹来生成安全图像。广泛的数值结果表明，我们的框架在抑制有毒图像生成方面实现了最先进的性能，并展示了对对抗攻击的鲁棒性，而无需调整模型参数。此外，与现有方法相比，PNO使用相当的生成时间，同时在安全生成和提示-图像对齐这两个冲突目标之间提供了最佳权衡。
</details>

<details>
    <summary>Key points</summary>
    * 利用了连续提示嵌入和噪声注入
    * 在有毒图像抑制中实现了最先进的性能
    * 保持了对对抗攻击的鲁棒性
</details>
</details>

---


<details>
<summary><b> Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis</b></summary>

* **Authors:** Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, Xiaobing Liu
* **arXiv ID:** 2412.04431
* **One-liner:** Created Infinity, a fast and high-quality bitwise autoregressive T2I model.
* **Published in:** arxiv (5 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.04431) | [[PDF]](https://arxiv.org/pdf/2412.04431) | [[Code]](https://github.com/FoundationVision/Infinity)

> **核心创新**
> 扩展了令牌化器和变换器以提高生成和速度。

<details>
    <summary>Abstract</summary>
    我们提出了Infinity，一种能够根据语言指令生成高分辨率、逼真图像的位级视觉自回归建模。Infinity在位级令牌预测框架下重新定义了视觉自回归模型，具有无限词汇令牌化器和分类器以及位级自校正机制，显著提高了生成能力和细节。通过理论上将令牌化器词汇大小扩展到无限，并同时扩展变换器大小，我们的方法相比普通VAR显著释放了强大的扩展能力。Infinity为自回归文本到图像模型设定了新纪录，超越了像SD3-Medium和SDXL这样的顶级扩散模型。值得注意的是，Infinity通过将GenEval基准分数从0.62提高到0.73，ImageReward基准分数从0.87提高到0.96，胜率为66%，超越了SD3-Medium。无需额外优化，Infinity在0.8秒内生成高质量的1024x1024图像，比SD3-Medium快2.6倍，使其成为最快的文本到图像模型。模型和代码将被发布，以促进Infinity在视觉生成和统一令牌化器建模方面的进一步探索。
</details>

<details>
    <summary>Key points</summary>
    * 实现了位级令牌预测与无限词汇
    * 使用了位级自校正机制
    * 实现了创纪录的速度和基准分数
</details>
</details>

---


<details>
<summary><b> LayerFusion: Harmonized Multi-Layer Text-to-Image Generation with Generative Priors</b></summary>

* **Authors:** Yusuf Dalva, Yijun Li, Qing Liu, Nanxuan Zhao, Jianming Zhang, Zhe Lin, Pinar Yanardag
* **arXiv ID:** 2412.04460
* **One-liner:** Developed a pipeline for layered content generation with LDMs.
* **Published in:** arxiv (5 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.04460) | [[PDF]](https://arxiv.org/pdf/2412.04460) | [[Code]]()

> **核心创新**
> 实现了前景和背景层的协调生成。

<details>
    <summary>Abstract</summary>
    大规模扩散模型在从文本描述生成高质量图像方面取得了显著成功，在各种应用中广受欢迎。然而，分层内容的生成，如具有前景和背景层的透明图像，仍然是一个未被充分探索的领域。分层内容生成对于图形设计、动画和数字艺术等领域的创意工作流程至关重要，其中基于层的方法对于灵活编辑和合成是基础。在本文中，我们提出了一种基于潜在扩散模型（LDMs）的新图像生成流水线，生成具有两个层的图像：一个具有透明度信息的前景层（RGBA）和一个背景层（RGB）。与现有方法顺序生成这些层不同，我们的方法引入了一种协调生成机制，使层之间能够进行动态交互，以获得更一致的输出。我们通过广泛的定性和定量实验证明了我们方法的有效性，显示出在视觉一致性、图像质量和层一致性方面相比基线方法的显著改进。
</details>

<details>
    <summary>Key points</summary>
    * 生成了RGBA前景和RGB背景层
    * 引入了层间的动态交互
    * 改善了视觉一致性和层一致性
</details>
</details>

---


<details>
<summary><b> Proactive Agents for Multi-Turn Text-to-Image Generation Under Uncertainty</b></summary>

* **Authors:** Meera Hahn, Wenjun Zeng, Nithish Kannen, Rich Galt, Kartikeya Badola, Been Kim, Zi Wang
* **arXiv ID:** 2412.06771
* **One-liner:** Proposed proactive T2I agents to align user intent through clarification.
* **Published in:** arxiv (9 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.06771) | [[PDF]](https://arxiv.org/pdf/2412.06771) | [[Code]](https://github.com/google-deepmind/proactive_t2i_agents)

> **核心创新**
> 使用了信念图和提问以实现更好的对齐。

<details>
    <summary>Abstract</summary>
    生成AI模型的用户提示往往不够明确，导致用户意图与模型理解之间的错位。因此，用户通常需要费力地精炼他们的提示。我们研究了文本到图像（T2I）生成中的这种对齐问题，并提出了一个主动T2I代理的原型，该代理配备了一个接口来（1）在不确定时主动询问澄清问题，以及（2）将关于用户意图的不确定性表示为一个可理解和可编辑的信念图。我们为这样的代理构建了简单原型，并提出了一种新的可扩展和自动化评估方法，使用两个代理，一个具有真实意图（一张图像），而另一个试图尽可能少地提问以与真实意图对齐。我们在三个图像-文本数据集上进行了实验：ImageInWords（Garg等人，2024）、COCO（Lin等人，2014）和DesignBench，一个我们策划的具有强烈艺术和设计元素的基准。在三个数据集上的实验证明了所提出的T2I代理能够询问信息丰富的问题并引出关键信息，以实现成功的对齐，VQAScore（Lin等人，2024）至少比标准T2I生成高2倍。此外，我们进行了人类研究，观察到至少90%的人类受试者发现这些代理及其信念图对他们的T2I工作流程有帮助，突出了我们方法的有效性。代码和DesignBench可在<a href="https://github.com/google-deepmind/proactive_t2i_agents" rel="external noopener nofollow" class="link-external link-https">此https URL</a>找到。
</details>

<details>
    <summary>Key points</summary>
    * 构建了询问澄清问题的代理
    * 将不确定性表示为可编辑信念图
    * 在评估中实现了更高的对齐分数
</details>
</details>

---


<details>
<summary><b> Boosting Alignment for Post-Unlearning Text-to-Image Generative Models</b></summary>

* **Authors:** Myeongseob Ko, Henry Li, Zhun Wang, Jonathan Patsenker, Jiachen T. Wang, Qinbin Li, Ming Jin, Dawn Song, Ruoxi Jia
* **arXiv ID:** 2412.07808
* **One-liner:** Enhanced machine unlearning for diffusion models to remove harmful content.
* **Published in:** arxiv (9 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.07808) | [[PDF]](https://arxiv.org/pdf/2412.07808) | [[Code]](https://github.com/reds-lab/Restricted_gradient_diversity_unlearning.git)

> **核心创新**
> 优化了模型更新以平衡遗忘和对齐。

<details>
    <summary>Abstract</summary>
    大规模生成模型通过海量数据展现了令人印象深刻的图像生成能力。然而，这往往无意中导致有害或不适当内容的生成，并引发版权担忧。受这些担忧驱动，机器遗忘变得至关重要，以有效从模型中清除不良知识。虽然现有文献研究了各种遗忘技术，但由于这些目标的竞争性质，这些技术往往要么遗忘质量差，要么在遗忘后文本-图像对齐退化。为了解决这些挑战，我们提出了一个框架，在每次遗忘迭代中寻求最优模型更新，确保在两个目标上单调改进。我们进一步推导了这种更新的特征。此外，我们设计了程序来战略性地多样化遗忘和剩余数据集，以提升性能改进。我们的评估表明，我们的方法有效地从最近的基于扩散的生成模型中移除目标类别，并从稳定扩散模型中移除概念，同时保持与模型原始训练状态的紧密对齐，从而优于最先进的基线。我们的代码将在<a href="https://github.com/reds-lab/Restricted_gradient_diversity_unlearning.git" rel="external noopener nofollow" class="link-external link-https">此https URL</a>提供。
</details>

<details>
    <summary>Key points</summary>
    * 寻求了最优更新以实现单调改进
    * 多样化了遗忘和剩余数据集
    * 在遗忘后保持了文本-图像对齐
</details>
</details>

---


<details>
<summary><b> Fast Prompt Alignment for Text-to-Image Generation</b></summary>

* **Authors:** Khalil Mrini, Hanlin Lu, Linjie Yang, Weilin Huang, Heng Wang
* **arXiv ID:** 2412.08639
* **One-liner:** Introduced Fast Prompt Alignment (FPA) for efficient text-to-image alignment.
* **Published in:** arxiv (11 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.08639) | [[PDF]](https://arxiv.org/pdf/2412.08639) | [[Code]](https://github.com/tiktok/fast_prompt_alignment)

> **核心创新**
> 开发了一种使用LLM进行改写的单次提示优化框架，减少计算开销同时保持对齐保真度。

<details>
    <summary>Abstract</summary>
    文本到图像生成技术发展迅速，但将复杂文本提示与生成图像对齐仍具挑战性，特别是在处理精细对象关系和细节时。本文介绍了快速提示对齐（FPA），一种利用单次方法的提示优化框架，提高了文本到图像对齐效率，无需当前方法如OPT2I的迭代开销。FPA使用大型语言模型进行单次迭代提示改写，然后通过优化提示进行微调或上下文学习，实现实时推理，减少计算需求同时保持对齐保真度。在COCO Captions和PartiPrompts数据集上的广泛评估表明，FPA在少量处理时间内实现了竞争性的文本-图像对齐分数，并通过自动指标（TIFA、VQA）和人工评估验证。一项由专家标注者进行的人工研究进一步揭示了人工对齐判断与自动分数之间的强相关性，突显了FPA改进的鲁棒性。该方法展示了可扩展、高效的迭代提示优化替代方案，使实时高需求场景具有更广泛适用性。代码库已提供以促进进一步研究：<a href="https://github.com/tiktok/fast_prompt_alignment" rel="external noopener nofollow" class="link-external link-https">此https URL</a>
</details>

<details>
    <summary>Key points</summary>
    * 利用大型语言模型进行单次迭代提示改写
    * 使用优化提示进行微调或上下文学习
    * 以减少的处理时间实现竞争性对齐分数
</details>
</details>

---


<details>
<summary><b> Preference Adaptive and Sequential Text-to-Image Generation</b></summary>

* **Authors:** Ofir Nabati, Guy Tennenholtz, ChihWei Hsu, Moonkyung Ryu, Deepak Ramachandran, Yinlam Chow, Xiang Li, Craig Boutilier
* **arXiv ID:** 2412.10419
* **One-liner:** Designed PASTA, an RL-based agent for interactive text-to-image generation.
* **Published in:** arxiv (10 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.10419) | [[PDF]](https://arxiv.org/pdf/2412.10419) | [[Code]]()

> **核心创新**
> 实现了强化学习方法，通过自适应提示扩展在多轮交互中改进图像集。

<details>
    <summary>Abstract</summary>
    我们解决了交互式文本到图像生成问题，设计了一个强化学习代理，通过一系列提示扩展迭代改进用户生成的图像集。利用人工评分者，我们创建了一个新颖的顺序偏好数据集，并结合大规模开源非顺序数据集。我们使用EM策略构建用户偏好和用户选择模型，并识别不同的用户偏好类型。然后，我们利用大型多模态语言模型和基于值的强化学习方法，向用户提供自适应和多样化的提示扩展建议。我们的偏好自适应顺序文本到图像代理（PASTA）扩展了T2I模型的自适应多轮能力，促进协作共创，并解决用户意图的不确定性或不足指定问题。我们使用人工评分者评估PASTA，显示相比基线方法的显著改进。我们还开源了顺序评分者数据集和模拟用户-评分者交互，以支持未来以用户为中心的多轮T2I系统研究。
</details>

<details>
    <summary>Key points</summary>
    * 使用强化学习代理进行迭代提示扩展
    * 创建并利用顺序偏好数据集
    * 集成LMM和基于值的强化学习以提供自适应建议
</details>
</details>

---


<details>
<summary><b> AlignGuard: Scalable Safety Alignment for Text-to-Image Generation</b></summary>

* **Authors:** Runtao Liu, I Chieh Chen, Jindong Gu, Jipeng Zhang, Renjie Pi, Qifeng Chen, Philip Torr, Ashkan Khakzar, Fabio Pizzati
* **arXiv ID:** 2412.10493
* **One-liner:** Proposed AlignGuard for safety alignment in text-to-image models.
* **Published in:** arxiv (13 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.10493) | [[PDF]](https://arxiv.org/pdf/2412.10493) | [[Code]]()

> **核心创新**
> 应用直接偏好优化与合成数据集，使用LoRA专家移除有害概念。

<details>
    <summary>Abstract</summary>
    文本到图像模型广泛使用，但其有限的安全防护使终端用户暴露于有害内容，并可能导致模型滥用。当前安全措施通常限于基于文本的过滤或概念移除策略，只能从模型生成能力中移除少数概念。在这项工作中，我们介绍了AlignGuard，一种用于T2I模型安全对齐的方法。我们通过合成生成有害和安全图像-文本对数据集（称为CoProV2），使直接偏好优化应用于T2I模型的安全目的。使用自定义DPO策略和该数据集，我们训练安全专家，以低秩适应矩阵形式，能够引导生成过程远离特定安全相关概念。然后，我们使用新颖的合并策略将专家合并为单个LoRA，以实现最优扩展性能。这种基于专家的方法实现了可扩展性，使我们能够从T2I模型中移除比基线多7倍的有害概念。AlignGuard在多个基准测试中一致优于最先进方法，并为T2I网络的安全对齐建立了新实践。代码和数据将在<a href="https://safetydpo.github.io/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>共享。
</details>

<details>
    <summary>Key points</summary>
    * 生成有害和安全图像-文本对的合成数据集
    * 使用DPO策略训练安全专家
    * 合并LoRA专家以实现可扩展概念移除
</details>
</details>

---


<details>
<summary><b> Efficient Scaling of Diffusion Transformers for Text-to-Image Generation</b></summary>

* **Authors:** Hao Li, Shamit Lal, Zhiheng Li, Yusheng Xie, Ying Wang, Yang Zou, Orchid Majumder, R. Manmatha, Zhuowen Tu, Stefano Ermon, Stefano Soatto, Ashwin Swaminathan
* **arXiv ID:** 2412.12391
* **One-liner:** Empirically studied scaling properties of Diffusion Transformers for text-to-image generation.
* **Published in:** arxiv (16 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.12391) | [[PDF]](https://arxiv.org/pdf/2412.12391) | [[Code]]()

> **核心创新**
> 识别U-ViT作为可扩展DiT模型，性能优于变体，并探索了数据扩展效应。

<details>
    <summary>Abstract</summary>
    我们通过进行广泛且严格的消融实验，实证研究了各种扩散变换器在文本到图像生成中的扩展特性，包括训练从0.3B到8B参数的缩放DiT，数据集规模达6亿图像。我们发现，U-ViT，一种纯自注意力基础的DiT模型，提供了更简单的设计，并比基于交叉注意力的DiT变体更有效地扩展，允许直接扩展以支持额外条件和其他模态。我们识别出2.3B U-ViT模型在受控设置中可以获得比SDXL UNet和其他DiT变体更好的性能。在数据扩展方面，我们研究了增加数据集大小和增强长标题如何提高文本-图像对齐性能和学习效率。
</details>

<details>
    <summary>Key points</summary>
    * 训练从0.3B到8B参数的缩放DiT
    * 发现U-ViT比交叉注意力变体更有效地扩展
    * 研究数据集大小和标题增强对对齐的影响
</details>
</details>

---


<details>
<summary><b> ArtAug: Enhancing Text-to-Image Generation through Synthesis-Understanding Interaction</b></summary>

* **Authors:** Zhongjie Duan, Qianyi Zhao, Cen Chen, Daoyuan Chen, Wenmeng Zhou, Yaliang Li, Yingda Chen
* **arXiv ID:** 2412.12888
* **One-liner:** Introduced ArtAug for enhancing text-to-image models via model interactions.
* **Published in:** arxiv (17 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.12888) | [[PDF]](https://arxiv.org/pdf/2412.12888) | [[Code]]()

> **核心创新**
> 利用图像理解模型提供美学建议，并迭代融合增强到合成模型中。

<details>
    <summary>Abstract</summary>
    扩散模型的出现显著推进了图像合成。最近在大型语言模型中模型交互和自我纠正推理方法的研究为增强文本到图像模型提供了新见解。受这些研究启发，我们在本文中提出了一种名为ArtAug的新方法。据我们所知，ArtAug是第一个通过模型交互与理解模型来改进图像合成模型的方法。在交互中，我们利用图像理解模型隐式学习的人类偏好，为图像合成模型提供细粒度建议。交互可以修改图像内容以使其更具美感，例如调整曝光、改变拍摄角度和添加大气效果。通过交互带来的增强通过额外的增强模块迭代融合到合成模型本身中。这使得合成模型能够直接生成美观图像，无需额外计算成本。在实验中，我们在现有文本到图像模型上训练ArtAug增强模块。各种评估指标一致证明ArtAug增强了文本到图像模型的生成能力，而不产生额外计算成本。源代码和模型将公开发布。
</details>

<details>
    <summary>Key points</summary>
    * 使用模型交互与理解模型提供细粒度建议
    * 通过额外模块迭代融合增强
    * 实现直接生成美观图像，无需额外成本
</details>
</details>

---


<details>
<summary><b> GALOT: Generative Active Learning via Optimizable Zero-shot Text-to-image Generation</b></summary>

* **Authors:** Hanbin Hong, Shenao Yan, Shuya Feng, Yan Yan, Yuan Hong
* **arXiv ID:** 2412.16227
* **One-liner:** Integrated zero-shot text-to-image synthesis with active learning for efficient model training.
* **Published in:** arxiv (18 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.16227) | [[PDF]](https://arxiv.org/pdf/2412.16227) | [[Code]]()

> **核心创新**
> 开发了一种使用主动学习标准优化文本输入以生成信息丰富合成数据集的框架。

<details>
    <summary>Abstract</summary>
    主动学习是机器学习中的关键方法，强调识别和利用最具信息量的样本以高效训练模型。然而，主动学习的一个显著挑战是其对有限标记数据样本和数据分布的依赖，导致性能受限。为解决这一限制，本文通过设计一种新颖框架，将零样本文本到图像合成与主动学习相结合，能够仅使用文本描述高效训练机器学习模型。具体而言，我们利用主动学习标准优化文本输入，以生成更具信息量和多样性的数据样本，这些样本由从文本生成的伪标签标注，然后作为合成数据集用于主动学习。这种方法减少了数据收集和标注成本，同时通过提供信息丰富的训练样本提高了模型训练效率，实现了从文本描述到视觉模型的端到端机器学习任务。通过全面评估，我们的框架在传统主动学习方法基础上展示了持续且显著的改进。
</details>

<details>
    <summary>Key points</summary>
    * 利用主动学习标准优化文本输入以生成数据
    * 使用从文本生成的伪标签进行标注
    * 减少数据收集成本并提高训练效率
</details>
</details>

---


<details>
<summary><b> Self-Corrected Flow Distillation for Consistent One-Step and Few-Step Text-to-Image Generation</b></summary>

* **Authors:** Quan Dao, Hao Phung, Trung Dao, Dimitris Metaxas, Anh Tran
* **arXiv ID:** 2412.16906
* **One-liner:** Introduced a self-corrected flow distillation method for generative models.
* **Published in:** arxiv (22 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.16906) | [[PDF]](https://arxiv.org/pdf/2412.16906) | [[Code]]()

> **核心创新**
> 在流匹配中结合一致性模型和对抗训练，实现了少步和单步采样的高质量生成。

<details>
    <summary>Abstract</summary>
    流匹配已成为训练生成模型的有前景框架，展示了令人印象深刻的实证性能，同时相比基于扩散的模型训练相对容易。然而，该方法在采样过程中仍需要多次函数评估。为解决这些限制，我们引入了一种自校正流蒸馏方法，有效将一致性模型和对抗训练整合到流匹配框架中。这项工作在少步和单步采样中实现了一致的生成质量。我们的广泛实验验证了该方法的有效性，在CelebA-HQ和COCO数据集的零样本基准测试中，定量和定性均获得优越结果。我们的实现在<a href="https://github.com/VinAIResearch/SCFlow" rel="external noopener nofollow" class="link-external link-https">此https URL</a>发布。
</details>

<details>
    <summary>Key points</summary>
    * 在流匹配中整合一致性模型和对抗训练
    * 在少步和单步采样中实现一致的生成质量
    * 在CelebA-HQ和COCO数据集上验证
</details>
</details>

---


<details>
<summary><b> Hierarchical Vision-Language Alignment for Text-to-Image Generation via Diffusion Models</b></summary>

* **Authors:** Emily Johnson, Noah Wilson
* **arXiv ID:** 2501.00917
* **One-liner:** Proposed VLAD model for improved text-to-image generation with dual-stream strategy.
* **Published in:** arxiv (1 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.00917) | [[PDF]](https://arxiv.org/pdf/2501.00917) | [[Code]]()

> **核心创新**
> 利用语义对齐和分层扩散增强图像质量和文本渲染准确性。

<details>
    <summary>Abstract</summary>
    文本到图像生成在集成大型视觉语言模型方面取得了显著进展，但在对齐复杂文本描述与高质量、视觉一致图像方面仍存在挑战。本文介绍了视觉语言对齐扩散模型，一种生成框架，通过结合语义对齐和分层扩散的双流策略解决这些挑战。VLAD利用上下文组合模块将文本提示分解为全局和局部表示，确保与视觉特征的精确对齐。此外，它结合了具有分层指导的多阶段扩散过程，以生成高保真图像。在MARIO-Eval和INNOVATOR-Eval基准测试上进行的实验表明，VLAD在图像质量、语义对齐和文本渲染准确性方面显著优于最先进方法。人工评估进一步验证了VLAD的优越性能，使其成为复杂场景中文本到图像生成的有前景方法。
</details>

<details>
    <summary>Key points</summary>
    * 采用双流策略结合语义对齐和分层扩散
    * 使用上下文组合模块进行提示分解
    * 在基准测试和人工评估中优于最先进方法
</details>
</details>

---


<details>
<summary><b> Evaluating Image Caption via Cycle-consistent Text-to-Image Generation</b></summary>

* **Authors:** Tianyu Cui, Jinbin Bai, Guo-Hua Wang, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, Ye Shi
* **arXiv ID:** 2501.03567
* **One-liner:** Introduced CAMScore, a cyclic reference-free evaluation metric for image captioning.
* **Published in:** arxiv (7 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.03567) | [[PDF]](https://arxiv.org/pdf/2501.03567) | [[Code]]()

> **核心创新**
> 开发了一种使用文本到图像生成评估标题的框架，通过比较生成图像与原始图像避免模态差距。

<details>
    <summary>Abstract</summary>
    评估图像标题通常依赖于参考标题，这些参考标题获取成本高且表现出显著多样性和主观性。虽然已提出无参考评估指标，但大多数关注标题和图像之间的跨模态评估。最近研究揭示了对比学习基础多模态系统中普遍存在的模态差距，削弱了如CLIPScore等跨模态指标的可靠性。在本文中，我们提出了CAMScore，一种用于图像标题模型的循环无参考自动评估指标。为规避上述模态差距，CAMScore利用文本到图像模型从标题生成图像，并随后将这些生成图像与原始图像进行比较评估。此外，为提供更全面评估的细粒度信息，我们为CAMScore设计了一个三级评估框架，涵盖像素级、语义级和客观级视角。在多个基准数据集上的广泛实验结果显示，CAMScore与现有基于参考和无参考指标相比，与人工判断的相关性更优，证明了该框架的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 使用文本到图像模型从标题生成图像
    * 将生成图像与原始图像进行比较评估
    * 包含像素级、语义级和客观级视角
</details>
</details>

---


<details>
<summary><b> Boosting Text-To-Image Generation via Multilingual Prompting in Large Multimodal Models</b></summary>

* **Authors:** Yongyu Mu, Hengyu Li, Junxin Wang, Xiaoxuan Zhou, Chenglong Wang, Yingfeng Luo, Qiaozhi He, Tong Xiao, Guocheng Chen, Jingbo Zhu
* **arXiv ID:** 2501.07086
* **One-liner:** Extended multilingual capabilities in text-to-image generation with PMT2I.
* **Published in:** arxiv (13 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.07086) | [[PDF]](https://arxiv.org/pdf/2501.07086) | [[Code]](https://github.com/takagi97/PMT2I)

> **核心创新**
> 构建了并行多语言提示，以利用LMM的多语言能力增强理解和多样性。

<details>
    <summary>Abstract</summary>
    先前关于增强大型多模态模型用于文本到图像生成的工作集中在丰富上下文学习的输入空间。这包括提供少量演示和优化图像描述以更详细和逻辑。然而，随着对更复杂和灵活图像描述的需求增长，在ICL范式中增强输入文本理解仍是一个关键但未充分探索的领域。在这项工作中，我们通过构建并行多语言提示扩展这一研究方向，旨在利用LMM的多语言能力。更具体地说，我们将输入文本翻译成多种语言，并向模型提供原始文本和翻译。在两个LMM上的三个基准测试实验表明，我们的方法PMT2I在一般、组合和细粒度评估中实现了优越性能，特别是在人类偏好对齐方面。此外，凭借其生成更多样化图像的优势，PMT2I在结合重排方法时显著优于基线提示。我们的代码和并行多语言数据可在<a href="https://github.com/takagi97/PMT2I" rel="external noopener nofollow" class="link-external link-https">此https URL</a>找到。
</details>

<details>
    <summary>Key points</summary>
    * 将输入文本翻译成多种语言用于提示
    * 利用LMM的多语言能力
    * 在组合和细粒度评估中提高性能
</details>
</details>

---


<details>
<summary><b> Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens</b></summary>

* **Authors:** Dongwon Kim, Ju He, Qihang Yu, Chenglin Yang, Xiaohui Shen, Suha Kwak, Liang-Chieh Chen
* **arXiv ID:** 2501.07730
* **One-liner:** Introduced TA-TiTok, an efficient image tokenizer that integrates text during decoding for improved performance.
* **Published in:** arxiv (13 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.07730) | [[PDF]](https://arxiv.org/pdf/2501.07730) | [[Code]](https://github.com/bytedance/1d-tokenizer)

> **核心创新**
> 开发了一种一阶段训练过程用于图像分词，消除了复杂蒸馏并实现了可扩展性。

<details>
    <summary>Abstract</summary>
    图像分词器构成了现代文生图生成模型的基础，但众所周知难以训练。此外，大多数现有文生图模型依赖于大规模、高质量的私有数据集，使其难以复现。在本工作中，我们引入了文本感知的基于Transformer的一维分词器（TA-TiTok），一种高效且强大的图像分词器，可以利用离散或连续的一维词元。TA-TiTok在分词器解码阶段（即去分词化）独特地整合了文本信息，加速了收敛并提升了性能。TA-TiTok还受益于简化而有效的一阶段训练过程，消除了先前一维分词器中使用的复杂两阶段蒸馏需求。这种设计允许无缝扩展到大型数据集。基于此，我们引入了一系列文生图掩码生成模型（MaskGen），仅使用开放数据训练，同时实现了与私有数据训练模型相媲美的性能。我们旨在发布高效、强大的TA-TiTok分词器以及开放数据、开放权重的MaskGen模型，以促进更广泛的访问并民主化文生图掩码生成模型领域。
</details>

<details>
    <summary>Key points</summary>
    * 文本感知的基于Transformer的一维分词器（TA-TiTok）
    * 在解码阶段整合文本信息
    * 简化的一阶段训练过程
    * 基于开放数据训练的掩码生成模型（MaskGen）
</details>
</details>

---


<details>
<summary><b> SHYI: Action Support for Contrastive Learning in High-Fidelity Text-to-Image Generation</b></summary>

* **Authors:** Tianxiang Xia, Lin Xiao, Yannick Montorfani, Francesco Pavia, Enis Simsar, Thomas Hofmann
* **arXiv ID:** 2501.09055
* **One-liner:** Improved text-to-image generation fidelity for actions involving multiple objects using enhanced contrastive learning.
* **Published in:** arxiv (15 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.09055) | [[PDF]](https://arxiv.org/pdf/2501.09055) | [[Code]](https://polybox.ethz.ch/index.php/s/dJm3SWyRohUrFxn)

> **核心创新**
> 采用语义超图对比邻接学习和InteractDiffusion来修正动作理解。

<details>
    <summary>Abstract</summary>
    在本项目中，我们解决了文生图生成中的不忠实问题，特别是涉及多个对象的动作。为此，我们基于CONFORM框架构建，该框架使用对比学习来提高多对象生成图像的准确性。然而，涉及多个不同对象的动作描绘仍有很大的改进空间。为了改进，我们采用了语义超图对比邻接学习，一种增强对比结构的理解以及“对比但链接”技术。我们进一步通过InteractDiffusion修正了Stable Diffusion对动作的理解。作为评估指标，我们使用了图像-文本相似度CLIP和TIFA。此外，我们进行了用户研究。我们的方法即使在Stable Diffusion理解一般的动词上也显示出有希望的结果。然后，我们通过分析结果提供了未来方向。我们的代码库可以在polybox上找到，链接：this https URL。
</details>

<details>
    <summary>Key points</summary>
    * 使用对比学习的CONFORM框架
    * 语义超图对比邻接学习
    * 对比但链接技术
    * 用于动作修正的InteractDiffusion
</details>
</details>

---


<details>
<summary><b> IMAGINE-E: Image Generation Intelligence Evaluation of State-of-the-art Text-to-Image Models</b></summary>

* **Authors:** Jiayi Lei, Renrui Zhang, Xiangfei Hu, Weifeng Lin, Zhen Li, Wenjian Sun, Ruoyi Du, Le Zhuo, Zhongyu Li, Xinyue Li, Shitian Zhao, Ziyu Guo, Yiting Lu, Peng Gao, Hongsheng Li
* **arXiv ID:** 2501.13920
* **One-liner:** Developed IMAGINE-E, a comprehensive evaluation framework for text-to-image models across multiple domains.
* **Published in:** arxiv (23 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.13920) | [[PDF]](https://arxiv.org/pdf/2501.13920) | [[Code]](https://github.com/jylei16/Imagine-e)

> **核心创新**
> 评估了六个模型在结构化输出、真实感、特定领域、挑战性场景和多风格任务中的表现。

<details>
    <summary>Abstract</summary>
    随着扩散模型的快速发展，文生图模型取得了显著进展，在提示跟随和图像生成方面展示了令人印象深刻的能力。最近发布的模型如FLUX.1和Ideogram2.0，以及其他如Dall-E3和Stable Diffusion 3，在各种复杂任务中表现出卓越性能，引发了文生图模型是否正朝着通用适用性发展的疑问。除了传统图像生成，这些模型在一系列领域展示了能力，包括可控生成、图像编辑、视频、音频、3D和运动生成，以及计算机视觉任务如语义分割和深度估计。然而，当前评估框架不足以全面评估这些模型在扩展领域中的性能。为了彻底评估这些模型，我们开发了IMAGINE-E并测试了六个突出模型：FLUX.1、Ideogram2.0、Midjourney、Dall-E3、Stable Diffusion 3和Jimeng。我们的评估分为五个关键领域：结构化输出生成、真实感和物理一致性、特定领域生成、挑战性场景生成和多风格创建任务。这一全面评估突出了每个模型的优势和局限性，特别是FLUX.1和Ideogram2.0在结构化和特定领域任务中的出色表现，强调了文生图模型作为基础AI工具的应用扩展和潜力。本研究为文生图模型当前状态和未来轨迹提供了宝贵见解，因为它们正朝着通用可用性演进。评估脚本将在this https URL发布。
</details>

<details>
    <summary>Key points</summary>
    * IMAGINE-E评估框架
    * 五个关键领域：结构化输出、真实感、特定领域、挑战性场景、多风格
    * 测试模型如FLUX.1和Ideogram2.0
</details>
</details>

---


<details>
<summary><b> Text-to-Image Generation for Vocabulary Learning Using the Keyword Method</b></summary>

* **Authors:** Nuwan T. Attygalle, Matjaž Kljun, Aaron Quigley, Klen čOpič Pucihar, Jens Grubert, Verena Biener, Luis A. Leiva, Juri Yoneyama, Alice Toniolo, Angela Miguel, Hirokazu Kato, Maheshya Weerasinghe
* **arXiv ID:** 2501.17099
* **One-liner:** Enhanced vocabulary memorization by combining the keyword method with text-to-image generators.
* **Published in:** arxiv (28 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.17099) | [[PDF]](https://arxiv.org/pdf/2501.17099) | [[Code]]()

> **核心创新**
> 将心理视觉链接外化为图像，显著改善了语言学习中的记忆保留。

<details>
    <summary>Abstract</summary>
    '关键词方法'是一种学习外语词汇的有效技术。它涉及在单词含义与其在外语中的发音在学习者母语中的发音之间创建可记忆的视觉链接。然而，这些可记忆的视觉链接在人们头脑中保持隐含，对于大量单词不易记忆。为了增强词汇的记忆和回忆，我们开发了一个应用程序，将关键词方法与文生图生成器结合，将可记忆的视觉链接外化为视觉图像。这些视觉图像在记忆过程中代表额外的刺激。为了探索这种方法的有效性，我们首先进行了一项试点研究，通过要求参与者写下它们来调查外化心理可视化描述的难度。我们使用这些描述作为文生图生成器（DALL-E2）的提示，将其转换为图像，并要求参与者选择他们最喜欢的。接下来，我们比较了不同的文生图生成器（DALL-E2、Midjourney、Stable和Latent Diffusion），以评估每个生成图像的感知质量。尽管结果异质，参与者大多偏好DALL-E2生成的图像，该生成器也用于最终研究。在这项研究中，我们调查了提供此类图像是否比仅使用关键词方法增强了学习词汇的保留。我们的结果表明，人们在描述其可记忆链接的可视化时没有遇到困难，并且提供相应图像显著改善了记忆保留。
</details>

<details>
    <summary>Key points</summary>
    * 关键词方法与文生图生成结合
    * 关于外化可视化的试点研究
    * 生成器如DALL-E2的比较
    * 关于图像增强保留的最终研究
</details>
</details>

---


<details>
<summary><b> TextAtlas5M: A Large-scale Dataset for Dense Text Image Generation</b></summary>

* **Authors:** Alex Jinpeng Wang, Dongxing Mao, Jiawei Zhang, Weiming Han, Zhuobai Dong, Linjie Li, Yiqi Lin, Zhengyuan Yang, Libo Qin, Fuwei Zhang, Lijuan Wang, Min Li
* **arXiv ID:** 2502.07870
* **One-liner:** Introduced TextAtlas5M, a dataset for evaluating long-text rendering in text-conditioned image generation.
* **Published in:** arxiv (11 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.07870) | [[PDF]](https://arxiv.org/pdf/2502.07870) | [[Code]](https://github.com/CSU-JPG/TextAtlas)

> **核心创新**
> 策划了TextAtlasEval基准，挑战了先进模型并突出了性能差距。

<details>
    <summary>Abstract</summary>
    文本条件图像生成近年来获得了显著关注，并且正在处理越来越长和全面的文本提示。在日常生活中，密集和复杂的文本出现在广告、信息图和标牌等上下文中，其中文本和视觉的整合对于传达复杂信息至关重要。然而，尽管有这些进展，生成长文本图像仍然是一个持续挑战，主要由于现有数据集的限制，这些数据集通常关注较短和较简单的文本。为了解决这一差距，我们引入了TextAtlas5M，一个专门设计用于评估文本条件图像生成长文本渲染的新数据集。我们的数据集包含500万张长文本生成和收集的图像，涵盖多种数据类型，使得能够全面评估大规模生成模型在长文本图像生成上的表现。我们进一步策划了3000张人类改进的测试集TextAtlasEval，跨越3个数据领域，建立了文本条件生成中最广泛的基准之一。评估表明，TextAtlasEval基准即使对于最先进的专有模型（例如GPT4o与DallE-3）也提出了显著挑战，而它们的开源对应物显示出更大的性能差距。这些证据将TextAtlas5M定位为一个有价值的训练和评估未来代文本条件图像生成模型的数据集。
</details>

<details>
    <summary>Key points</summary>
    * 包含500万张图像的TextAtlas5M数据集
    * 跨越3个领域的TextAtlasEval基准
    * 长文本生成挑战的评估
</details>
</details>

---


<details>
<summary><b> Skrr: Skip and Re-use Text Encoder Layers for Memory Efficient Text-to-Image Generation</b></summary>

* **Authors:** Hoigi Seo, Wongi Jeong, Jae-sun Seo, Se Young Chun
* **arXiv ID:** 2502.08690
* **One-liner:** Proposed Skrr, a pruning strategy for text encoders in T2I models to reduce memory usage without performance loss.
* **Published in:** arxiv (12 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.08690) | [[PDF]](https://arxiv.org/pdf/2502.08690) | [[Code]]()

> **核心创新**
> 选择性跳过或重用Transformer层，实现了最先进的内存效率。

<details>
    <summary>Abstract</summary>
    文生图扩散模型中的大规模文本编码器在从文本提示生成高质量图像方面展示了卓越性能。与依赖多个迭代步骤的去噪模块不同，文本编码器仅需单次前向传递即可产生文本嵌入。然而，尽管它们对总推理时间和浮点操作（FLOPs）的贡献最小，文本编码器需要显著更高的内存使用，高达去噪模块的八倍。为了解决这种低效问题，我们提出了跳过和重用层（Skrr），一种简单而有效的剪枝策略，专门为文生图扩散模型中的文本编码器设计。Skrr通过以针对文生图任务定制的方式选择性跳过或重用某些层，来利用Transformer块中的固有冗余，从而在不损害性能的情况下减少内存消耗。广泛实验表明，Skrr即使在高度稀疏水平下也保持与原始模型相当的图像质量，优于现有的块级剪枝方法。此外，Skrr在多个评估指标（包括FID、CLIP、DreamSim和GenEval分数）上实现了最先进的内存效率，同时保持性能。
</details>

<details>
    <summary>Key points</summary>
    * 跳过和重用层（Skrr）剪枝
    * 利用Transformer块中的冗余
    * 使用FID、CLIP、DreamSim、GenEval分数进行评估
</details>
</details>

---


<details>
<summary><b> FlexControl: Computation-Aware ControlNet with Differentiable Router for Text-to-Image Generation</b></summary>

* **Authors:** Zheng Fang, Lichuan Xiang, Xu Cai, Kaicheng Zhou, Hongkai Wen
* **arXiv ID:** 2502.10451
* **One-liner:** Introduced FlexControl, a framework for dynamic block selection in controlled diffusion models.
* **Published in:** arxiv (11 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.10451) | [[PDF]](https://arxiv.org/pdf/2502.10451) | [[Code]](https://github.com/Anonymousuuser/FlexControl)

> **核心创新**
> 使用可训练门控和计算感知损失来增强适应性并减少计算开销。

<details>
    <summary>Abstract</summary>
    ControlNet提供了一种强大的方式来指导基于扩散的生成模型，但大多数实现依赖于临时启发式方法来选择要控制的网络块——这种方法在不同任务中变化不可预测。为了解决这一差距，我们提出了FlexControl，一种新颖的框架，在训练期间复制所有扩散块，并采用可训练的门控机制来动态选择在每个去噪步骤中激活哪些块。通过引入计算感知损失，我们可以鼓励控制块仅在有益于生成质量时激活。通过消除手动块选择，FlexControl增强了跨多样任务的适应性，并简化了设计流程，以端到端训练方式结合计算感知训练损失。通过对UNet（例如SD1.5）和DiT（例如SD3.0）的全面实验，我们表明我们的方法在某些关键方面优于现有ControlNet变体。正如定量和定性评估所证明的，FlexControl保持或增强了图像保真度，同时通过选择性激活最相关块减少了计算开销。这些结果强调了灵活、数据驱动方法在受控扩散中的潜力，并为高效生成模型设计开辟了新途径。代码将很快在this https URL提供。
</details>

<details>
    <summary>Key points</summary>
    * 具有可训练门控机制的FlexControl
    * 用于选择性激活的计算感知损失
    * 在UNet和DiT架构上的实验
</details>
</details>

---


<details>
<summary><b> REAL: Realism Evaluation of Text-to-Image Generation Models for Effective Data Augmentation</b></summary>

* **Authors:** Ran Li, Xiaomeng Jin, Heng ji
* **arXiv ID:** 2502.10663
* **One-liner:** Proposed REAL, an automatic evaluation framework for assessing realism in T2I outputs.
* **Published in:** arxiv (15 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.10663) | [[PDF]](https://arxiv.org/pdf/2502.10663) | [[Code]]()

> **核心创新**
> 评估细粒度属性、不寻常关系和风格，与人类判断对齐。

<details>
    <summary>Abstract</summary>
    文生图生成模型的最新进展已经改变了该领域。然而，在生成反映要求苛刻的文本描述的图像方面仍然存在挑战，特别是对于细粒度细节和不寻常关系。现有评估指标关注文本-图像对齐，但忽略了生成图像的真实感，这对于下游应用如机器学习中的数据增强可能至关重要。为了解决这一差距，我们提出了REAL，一个自动评估框架，从三个维度评估T2I输出的真实感：细粒度视觉属性、不寻常视觉关系和视觉风格。REAL在与人类判断对齐方面实现了高达0.62的Spearman's rho分数，并展示了在排名和过滤增强数据用于图像描述、分类和视觉关系检测等任务中的实用性。实证结果表明，通过我们的指标评估的高分图像将图像分类的F1分数提高了高达11.3%，而低分图像则降低了高达4.95%。我们在真实感维度上基准测试了四个主要T2I模型，为T2I输出真实感的未来改进提供了见解。
</details>

<details>
    <summary>Key points</summary>
    * 具有三个真实感维度的REAL框架
    * Spearman's rho分数高达0.62
    * 在分类等任务中数据增强的实用性
</details>
</details>

---


<details>
<summary><b> Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation</b></summary>

* **Authors:** Taeyoung Yun, Dinghuai Zhang, Jinkyoo Park, Ling Pan
* **arXiv ID:** 2502.11477
* **One-liner:** Introduced PAG, using GFlowNets for diverse and effective prompt adaptation in text-to-image generation.
* **Published in:** arxiv (17 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.11477) | [[PDF]](https://arxiv.org/pdf/2502.11477) | [[Code]]()

> **核心创新**
> 将提示适应框架为概率推理，解决模式崩溃和神经可塑性损失。

<details>
    <summary>Abstract</summary>
    文生图扩散模型的最新进展已经实现了令人印象深刻的图像生成能力。然而，用期望属性（例如美学质量、用户意图）控制生成过程仍然具有挑战性，这些属性可以表示为黑盒奖励函数。在本文中，我们关注提示适应，它将原始提示细化为模型偏好的提示以生成期望图像。虽然先前工作使用强化学习（RL）来优化提示，但我们观察到应用RL通常导致生成相似的后缀和确定性行为。为此，我们引入了\textbf{提示适应与生成流网络（PAG）}，一种新颖的方法，将提示适应框架为概率推理问题。我们的关键见解是，利用生成流网络（GFlowNets）允许我们从奖励最大化转向从未归一化密度函数中采样，实现高质量和多样化的提示生成。然而，我们识别到GFlowNets的简单应用遭受模式崩溃，并揭示了一个先前被忽视的现象：模型中神经可塑性的渐进损失，这在顺序提示生成中与低效信用分配相结合。为了解决这一关键挑战，我们在PAG中开发了一种系统方法，包括流重新激活、奖励优先采样和奖励分解用于提示适应。广泛实验验证了PAG成功学习为文生图生成采样有效和多样化的提示。我们还表明，PAG在各种奖励函数和不同文生图模型之间表现出强鲁棒性和可转移性。
</details>

<details>
    <summary>Key points</summary>
    * 提示适应与生成流网络（PAG）
    * 流重新激活、奖励优先采样、奖励分解
    * 跨奖励函数和模型的鲁棒性
</details>
</details>

---


<details>
<summary><b> CHATS: Combining Human-Aligned Optimization and Test-Time Sampling for Text-to-Image Generation</b></summary>

* **Authors:** Minghao Fu, Guo-Hua Wang, Liangfu Cao, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang
* **arXiv ID:** 2502.12579
* **One-liner:** Introduced CHATS, a framework combining human-aligned optimization and test-time sampling for T2I models.
* **Published in:** arxiv (18 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.12579) | [[PDF]](https://arxiv.org/pdf/2502.12579) | [[Code]](https://github.com/AIDC-AI/CHATS)

> **核心创新**
> 建模偏好和非偏好分布，以数据效率实现高性能。

<details>
    <summary>Abstract</summary>
    扩散模型已经成为文生图生成的主导方法。关键组件如人类偏好对齐和无分类器指导在确保生成质量方面扮演关键角色。然而，它们在当前文生图模型中的独立应用在实现强文本-图像对齐、高生成质量和与人类美学标准一致性方面继续面临显著挑战。在这项工作中，我们首次探索促进人类性能对齐和测试时采样的协作，以解锁文生图模型的潜力。因此，我们引入了CHATS（结合人类对齐优化和测试时采样），一种新颖的生成框架，分别建模偏好和非偏好分布，并采用基于代理提示的采样策略来利用两个分布中包含的有用信息。我们观察到CHATS表现出卓越的数据效率，仅使用小型高质量微调数据集即可实现强性能。广泛实验表明，CHATS超越了传统偏好对齐方法，在各种标准基准上设定了新的最先进水平。
</details>

<details>
    <summary>Key points</summary>
    * 具有基于代理提示采样的CHATS框架
    * 分布的分别建模
    * 使用小型微调数据集的数据效率
</details>
</details>

---


<details>
<summary><b> FlipConcept: Tuning-Free Multi-Concept Personalization for Text-to-Image Generation</b></summary>

* **Authors:** Young Beom Woo, Sun Eung Kim, Seong-Whan Lee
* **arXiv ID:** 2502.15203
* **One-liner:** Proposed FlipConcept for seamless multi-concept personalization in T2I without additional tuning.
* **Published in:** arxiv (21 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.15203) | [[PDF]](https://arxiv.org/pdf/2502.15203) | [[Code]]()

> **核心创新**
> 引入引导外观注意力、掩码引导噪声混合和背景稀释，以增强保真度并防止概念泄漏。

<details>
    <summary>Abstract</summary>
    近年来，将多个个性化概念整合到单一图像中在文本到图像（T2I）生成领域受到关注。然而，现有方法在复杂场景中常因非个性化区域失真和需要额外微调而导致性能下降，限制了其实用性。为解决这一问题，我们提出FlipConcept，一种无需额外调优即可无缝整合多个个性化概念的新方法。我们引入引导外观注意力以增强个性化概念的视觉保真度。此外，我们采用掩码引导噪声混合来保护概念整合过程中的非个性化区域。最后，我们应用背景稀释以最小化概念泄漏，即个性化概念与图像中其他对象的不期望混合。实验中，我们证明所提方法尽管无需调优，但在单概念和多概念个性化推理中均优于现有模型，展示了其可扩展、高质量多概念个性化的有效性和实用性。
</details>

<details>
    <summary>Key points</summary>
    * 引导外观注意力用于视觉保真度
    * 掩码引导噪声混合以保护非个性化区域
    * 背景稀释以最小化概念泄漏
</details>
</details>

---


<details>
<summary><b> Multi-Agent Multimodal Models for Multicultural Text to Image Generation</b></summary>

* **Authors:** Parth Bhalerao, Mounika Yalamarty, Brian Trinh, Oana Ignat
* **arXiv ID:** 2502.15972
* **One-liner:** Introduced MosAIG, a multi-agent framework for multicultural image generation using LLMs.
* **Published in:** arxiv (21 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.15972) | [[PDF]](https://arxiv.org/pdf/2502.15972) | [[Code]](https://github.com/OanaIgnat/MosAIG)

> **核心创新**
> 利用LLMs中的不同文化角色并提供多文化数据集，显示多智能体交互优于无智能体模型。

<details>
    <summary>Abstract</summary>
    大型语言模型（LLMs）在各种多模态任务中展现出卓越性能。然而，由于现有数据和模型主要基于西方中心主义，其在跨文化背景下的有效性仍有限。同时，多智能体模型在解决复杂任务中表现出强大能力。本文中，我们评估了LLMs在多智能体交互设置下用于多文化图像生成这一新任务的性能。我们的主要贡献包括：（1）引入MosAIG，一个多智能体框架，通过利用具有不同文化角色的LLMs来增强多文化图像生成；（2）提供一个包含9000张多文化图像的数据集，涵盖五个国家、三个年龄段、两种性别、25个历史地标和五种语言；（3）证明多智能体交互在多个评估指标上优于无智能体模型，为未来研究提供宝贵洞见。我们的数据集和模型可在<a href="https://github.com/OanaIgnat/MosAIG" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 具有文化角色的多智能体框架
    * 9000张多文化图像的数据集
    * 在评估指标中展示优越性
</details>
</details>

---


<details>
<summary><b> Multimodal Representation Alignment for Image Generation: Text-Image Interleaved Control Is Easier Than You Think</b></summary>

* **Authors:** Liang Chen, Shuai Bai, Wenhao Chai, Weichu Xie, Haozhe Zhao, Leon Vinci, Junyang Lin, Baobao Chang
* **arXiv ID:** 2502.20172
* **One-liner:** Proposed Dream Engine for arbitrary text-image interleaved control in image generation.
* **Published in:** arxiv (27 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.20172) | [[PDF]](https://arxiv.org/pdf/2502.20172) | [[Code]](https://github.com/chenllliang/DreamEngine)

> **核心创新**
> 利用大型多模态模型实现共享表示和两阶段训练范式以进行有效控制。

<details>
    <summary>Abstract</summary>
    先进文本到图像生成领域正涌现出统一框架，将强大文本编码器（如CLIP和T5）与扩散Transformer骨干网络整合。尽管已有努力通过额外条件（如边缘图和深度图）控制输出图像，但针对任意文本-图像交错控制的全面框架仍缺乏。这一差距在尝试生成过程中合并多个图像的概念或视觉元素时尤为明显。为弥补此差距，我们进行了初步实验，显示大型多模态模型（LMMs）提供了一个有效的共享表示空间，其中图像和文本可以良好对齐，作为外部扩散模型的条件。基于此发现，我们提出Dream Engine，一个高效统一的框架，专为图像生成模型中的任意文本-图像交错控制而设计。基于强大文本到图像模型（如SD3.5），我们通过整合多功能多模态信息编码器（如QwenVL）替换原始仅文本编码器。我们的方法采用两阶段训练范式，包括联合文本-图像对齐和多模态交错指令调优。实验证明此训练方法有效，在GenEval基准上获得0.69总分，并与最先进文本到图像模型（如SD3.5和FLUX）性能相当。
</details>

<details>
    <summary>Key points</summary>
    * 与LMMs的共享表示空间
    * 两阶段训练：联合对齐和指令调优
    * 用多模态编码器替换仅文本编码器
</details>
</details>

---


<details>
<summary><b> Fine-Grained Alignment and Noise Refinement for Compositional Text-to-Image Generation</b></summary>

* **Authors:** Amir Mohammad Izadi, Seyed Mohammad Hadi Hosseini, Soroush Vafaie Tabar, Ali Abdollahi, Armin Saghafian, Mahdieh Soleymani Baghshah
* **arXiv ID:** 2503.06506
* **One-liner:** Developed a training-free method to improve text-to-image compositionality with constraint-based losses.
* **Published in:** arxiv (9 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.06506) | [[PDF]](https://arxiv.org/pdf/2503.06506) | [[Code]](https://github.com/hadi-hosseini/noise-refinement)

> **核心创新**
> 整合实体和属性约束作为损失，并引入反馈驱动的噪声优化系统。

<details>
    <summary>Abstract</summary>
    文本到图像生成模型近年来取得显著进展；然而，准确捕捉文本提示中的精细细节（如实体缺失、属性绑定错误和关系错误）仍是一个严峻挑战。为此，我们提出一种创新的免训练方法，通过整合定制目标来考虑文本约束，直接应对这些挑战。与基于布局的方法不同，后者强制执行刚性结构并限制多样性，我们的方法仅施加从文本提取的约束，提供更灵活的场景安排。这些约束被公式化为损失函数——实体缺失、实体混合、属性绑定和空间关系——整合为统一损失，应用于首先生成阶段。此外，我们引入反馈驱动的细粒度初始噪声优化系统。该系统整合一个验证器，评估生成图像、识别不一致并提供纠正反馈。利用此反馈，我们的优化方法首先通过优化与这些约束相关的选择性损失，来优化由初始噪声引起的错误注意力图，以针对未满足约束。随后，重新应用统一损失函数以进行第二阶段生成。实验结果表明，我们的方法仅依赖所提目标函数，显著增强了组合性，在人类评估中提升24%，在空间关系中提升25%。此外，我们的细粒度噪声优化证明有效，性能提升高达5%。代码可在\href{<a href="https://github.com/hadi-hosseini/noise-refinement" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}{<a href="https://github.com/hadi-hosseini/noise-refinement" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}获取。
</details>

<details>
    <summary>Key points</summary>
    * 文本约束的统一损失
    * 反馈驱动的噪声优化
    * 用于不一致识别的验证器
</details>
</details>

---


<details>
<summary><b> Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment</b></summary>

* **Authors:** Xing Xie, Jiawei Liu, Ziyue Lin, Huijie Fan, Zhi Han, Yandong Tang, Liangqiong Qu
* **arXiv ID:** 2503.07334
* **One-liner:** Introduced ARRA for global-coherent text-to-image generation in autoregressive LLMs without architectural changes.
* **Published in:** arxiv (10 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.07334) | [[PDF]](https://arxiv.org/pdf/2503.07334) | [[Code]](https://github.com/xiexing0916/ARRA)

> **核心创新**
> 使用全局视觉对齐损失和混合令牌将LLM隐藏状态与视觉表示对齐。

<details>
    <summary>Abstract</summary>
    我们提出自回归表示对齐（ARRA），一种新训练框架，无需架构修改即可在自回归LLMs中解锁全局一致文本到图像生成。不同于先前需要复杂架构重新设计的工作，ARRA通过全局视觉对齐损失和混合令牌[object Object]将LLMs的隐藏状态与外部视觉基础模型的视觉表示对齐。该令牌强制执行双重约束：局部下一令牌预测和全局语义蒸馏，使LLMs在保留原始自回归范式的同时隐式学习空间和上下文一致性。广泛实验验证了ARRA的即插即用多功能性。在从头训练T2I LLMs时，ARRA将自回归LLMs（如LlamaGen）的FID降低16.6%（ImageNet）和12.0%（LAION-COCO），而无需修改原始架构和推理机制。在从仅文本生成LLMs训练时，ARRA将高级LLMs（如Chameleon）的FID降低25.5%（MIMIC-CXR）和8.8%（DeepEyeNet）。对于领域适应，ARRA将通用LLMs与专用模型（如BioMedCLIP）对齐，在医学成像（MIMIC-CXR）上比直接微调实现18.6%的FID降低。这些结果表明，训练目标重新设计而非架构修改可以解决跨模态全局一致性挑战。ARRA为推进自回归模型提供了补充范式。代码可在<a href="https://github.com/xiexing0916/ARRA" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 全局视觉对齐损失
    * 用于双重约束的混合令牌
    * 各种LLMs的即插即用多功能性
</details>
</details>

---


<details>
<summary><b> SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation</b></summary>

* **Authors:** Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, Enze Xie
* **arXiv ID:** 2503.09641
* **One-liner:** Presented SANA-Sprint for ultra-fast T2I generation with hybrid distillation and step-adaptive inference.
* **Published in:** arxiv (12 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.09641) | [[PDF]](https://arxiv.org/pdf/2503.09641) | [[Code]](https://github.com/NVlabs/Sana)

> **核心创新**
> 结合sCM和LADD进行高效蒸馏，在1-4步内实现高质量生成和实时控制。

<details>
    <summary>Abstract</summary>
    本文提出SANA-Sprint，一种用于超快速文本到图像（T2I）生成的高效扩散模型。SANA-Sprint基于预训练基础模型，并通过混合蒸馏增强，将推理步骤从20步大幅减少至1-4步。我们引入三个关键创新：（1）提出免训练方法，将预训练流匹配模型转换为连续时间一致性蒸馏（sCM），避免从头开始的昂贵训练并实现高训练效率。我们的混合蒸馏策略结合sCM与潜在对抗蒸馏（LADD）：sCM确保与教师模型对齐，而LADD增强单步生成保真度。（2）SANA-Sprint是统一的自适应步数模型，在1-4步内实现高质量生成，消除步数特定训练并提高效率。（3）我们将ControlNet与SANA-Sprint整合，实现实时交互式图像生成，为用户交互提供即时视觉反馈。SANA-Sprint在速度-质量权衡中建立了新的帕累托前沿，仅在1步内实现最先进性能，FID为7.59，GenEval为0.74——优于FLUX-schnell（FID 7.94 / GenEval 0.71），同时速度快10倍（H100上0.1秒 vs 1.1秒）。在H100上，它还实现1024 x 1024图像的0.1秒（T2I）和0.25秒（ControlNet）延迟，在RTX 4090上为0.31秒（T2I），展示了其卓越效率和AI驱动消费应用（AIPC）的潜力。代码和预训练模型将开源。
</details>

<details>
    <summary>Key points</summary>
    * 混合蒸馏：sCM和LADD
    * 自适应步数模型用于1-4步
    * 与ControlNet整合以实现交互性
</details>
</details>

---


<details>
<summary><b> ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation</b></summary>

* **Authors:** Zirun Guo, Tao Jin
* **arXiv ID:** 2503.10358
* **One-liner:** Proposed ConceptGuard to address catastrophic forgetting in sequential diffusion customization.
* **Published in:** arxiv (13 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.10358) | [[PDF]](https://arxiv.org/pdf/2503.10358) | [[Code]]()

> **核心创新**
> 结合移位嵌入、概念绑定提示、记忆保留和优先级队列以实现动态概念管理。

<details>
    <summary>Abstract</summary>
    扩散定制方法仅用少量用户提供图像即取得显著成果。然而，现有方法集体定制概念，而实际应用常需顺序概念整合。这种顺序性可能导致灾难性遗忘，即先前学习概念丢失。本文中，我们研究持续定制中的概念遗忘和概念混淆。为应对这些挑战，我们提出ConceptGuard，一种综合方法，结合移位嵌入、概念绑定提示和记忆保留正则化，辅以优先级队列，可自适应更新不同概念的重要性和出现顺序。这些策略能动态更新、解绑和学习先前概念的关系，从而缓解概念遗忘和混淆。通过全面实验，我们展示我们的方法在定量和定性分析中一致且显著优于所有基线方法。
</details>

<details>
    <summary>Key points</summary>
    * 移位嵌入和概念绑定提示
    * 记忆保留正则化
    * 用于概念重要性的优先级队列
</details>
</details>

---


<details>
<summary><b> DiT-Air: Revisiting the Efficiency of Diffusion Model Architecture Design in Text to Image Generation</b></summary>

* **Authors:** Chen Chen, Rui Qian, Wenze Hu, Tsu-Jui Fu, Jialing Tong, Xinze Wang, Lezhi Li, Bowen Zhang, Alex Schwing, Wei Liu, Yinfei Yang
* **arXiv ID:** 2503.10618
* **One-liner:** Empirically studied DiTs and introduced DiT-Air for efficient and high-performance text-to-image generation.
* **Published in:** arxiv (13 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.10618) | [[PDF]](https://arxiv.org/pdf/2503.10618) | [[Code]]()

> **核心创新**
> 发现标准DiT与专用模型相当，层间共享减少模型大小并实现SOTA性能。

<details>
    <summary>Abstract</summary>
    本工作中，我们实证研究扩散Transformer（DiTs）用于文本到图像生成，聚焦架构选择、文本条件策略和训练协议。我们评估一系列基于DiT的架构——包括PixArt风格和MMDiT变体——并与标准DiT变体比较，后者直接处理拼接文本和噪声输入。令人惊讶地，我们的发现显示标准DiT性能与这些专用模型相当，同时展示出更优的参数效率，尤其在放大时。利用层间参数共享策略，我们相比MMDiT架构进一步减少66%模型大小，性能影响最小。基于对关键组件（如文本编码器和变分自编码器（VAEs））的深入分析，我们引入DiT-Air和DiT-Air-Lite。通过监督和奖励微调，DiT-Air在GenEval和T2I CompBench上实现最先进性能，而DiT-Air-Lite保持高度竞争力，尽管尺寸紧凑，仍超越大多数现有模型。
</details>

<details>
    <summary>Key points</summary>
    * DiTs的架构评估
    * 层间参数共享
    * 监督和奖励微调
</details>
</details>

---


<details>
<summary><b> TF-TI2I: Training-Free Text-and-Image-to-Image Generation via Multi-Modal Implicit-Context Learning in Text-to-Image Models</b></summary>

* **Authors:** Teng-Fang Hsiao, Bo-Kai Ruan, Yi-Lun Wu, Tzu-Ling Lin, Hong-Han Shuai
* **arXiv ID:** 2503.15283
* **One-liner:** Introduced TF-TI2I for training-free text-and-image-to-image generation with enhanced multimodal interactions.
* **Published in:** arxiv (19 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.15283) | [[PDF]](https://arxiv.org/pdf/2503.15283) | [[Code]](https://github.com/BlueDyee/TF-TI2I)

> **核心创新**
> 使用MM-DiT架构与参考上下文掩码和赢家通吃模块处理复杂指令。

<details>
    <summary>Abstract</summary>
    文本和图像到图像（TI2I）作为文本到图像（T2I）的扩展，整合图像输入与文本指令以增强图像生成。现有方法常部分利用图像输入，聚焦特定元素如对象或风格，或在复杂多图像指令下生成质量下降。为克服这些挑战，我们引入免训练文本和图像到图像（TF-TI2I），无需额外训练即可适应尖端T2I模型如SD3。我们的方法利用MM-DiT架构，其中我们指出文本令牌可从视觉令牌隐式学习视觉信息。我们通过从参考图像提取浓缩视觉表示来增强此交互，通过参考上下文掩码实现选择性信息共享——该技术将上下文令牌使用限制于指令相关视觉信息。此外，我们的赢家通吃模块通过为每个视觉令牌优先选择最相关参考来缓解分布偏移。针对TI2I评估的空白，我们还引入FG-TI2I Bench，一个专为TI2I设计并与现有T2I方法兼容的综合基准。我们的方法在各种基准上展示稳健性能，证实其在处理复杂图像生成任务中的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 参考上下文掩码
    * 赢家通吃模块
    * 用于评估的FG-TI2I Bench
</details>
</details>

---


<details>
<summary><b> Zero-Shot Styled Text Image Generation, but Make It Autoregressive</b></summary>

* **Authors:** Vittorio Pippi, Fabio Quattrini, Silvia Cascianelli, Alessio Tonioni, Rita Cucchiara
* **arXiv ID:** 2503.17074
* **One-liner:** Proposed Emuru, an autoregressive model for styled handwritten text generation with zero-shot generalization.
* **Published in:** arxiv (21 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.17074) | [[PDF]](https://arxiv.org/pdf/2503.17074) | [[Code]]()

> **核心创新**
> 利用VAE和Transformer在合成数据上训练，生成无背景伪影的风格化文本图像。

<details>
    <summary>Abstract</summary>
    风格化手写文本生成（HTG）最近受到计算机视觉和文档分析社区关注，已开发出多种基于GAN或扩散的解决方案，取得有希望结果。然而，这些策略未能泛化到新风格，并存在技术限制，尤其在最大输出长度和训练效率方面。为克服这些限制，本工作中我们提出一种新文本图像生成框架，称为Emuru。我们的方法利用强大文本图像表示模型（变分自编码器）结合自回归Transformer。我们的方法能够生成风格化文本图像，条件于文本内容和风格示例，如特定字体或手写风格。我们仅在多样化合成数据集上训练模型，该数据集包含超过100,000种打字和书法字体渲染的英文文本，使其具备零样本泛化到未见风格（包括字体和用户手写）的能力。据我们所知，Emuru是首个用于HTG的自回归模型，且首个专为泛化到新风格而设计。此外，我们的模型生成无背景伪影的图像，更易于下游应用使用。在打字和手写、任意长度文本图像生成场景的广泛评估中，证明了我们方法的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 自回归Transformer与VAE
    * 在多样化合成数据集上训练
    * 零样本泛化到新风格
</details>
</details>

---


<details>
<summary><b> Progressive Prompt Detailing for Improved Alignment in Text-to-Image Generative Models</b></summary>

* **Authors:** Ketan Suhaas Saichandran, Xavier Thomas, Prakhar Kaushik, Deepti Ghadiyaram
* **arXiv ID:** 2503.17794
* **One-liner:** Proposed SCoPE, a training-free method to improve text-to-image alignment for complex scenes.
* **Published in:** arxiv (22 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.17794) | [[PDF]](https://arxiv.org/pdf/2503.17794) | [[Code]]()

> **核心创新**
> 在推理过程中逐步从粗粒度到细粒度优化输入提示。

<details>
    <summary>Abstract</summary>
    文本到图像生成模型在处理包含复杂场景、多样对象及其视觉特征和空间关系的长提示时常常遇到困难。在本研究中，我们提出了SCoPE（计划性粗到细提示嵌入插值），一种无需训练的方法，通过从粗粒度到细粒度的方式逐步优化输入提示，以改善文本到图像的对齐。给定一个详细的输入提示，我们首先将其分解为多个子提示，这些子提示从描述广泛场景布局演变为高度精细的细节。在推理过程中，我们在这些子提示之间进行插值，从而逐步将更细粒度的细节引入生成的图像中。我们的无需训练即插即用方法显著增强了提示对齐，在GenAI-Bench数据集的83%提示上，相对于Stable Diffusion基线，视觉问答（VQA）得分平均提高了超过8分。
</details>

<details>
    <summary>Key points</summary>
    * 将提示分解为从广泛布局到精细细节演变的子提示
    * 在子提示之间插值以引入更细粒度的细节
    * 在GenAI-Bench数据集上实现VQA得分平均提高8分
</details>
</details>

---


<details>
<summary><b> Plug-and-Play Interpretable Responsible Text-to-Image Generation via Dual-Space Multi-facet Concept Control</b></summary>

* **Authors:** Basim Azam, Naveed Akhtar
* **arXiv ID:** 2503.18324
* **One-liner:** Introduced a scalable technique for responsible T2I generation addressing fairness and safety.
* **Published in:** arxiv (24 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.18324) | [[PDF]](https://arxiv.org/pdf/2503.18324) | [[Code]]()

> **核心创新**
> 通过知识蒸馏和概念白化使用可解释复合负责任空间。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）模型的伦理问题需要对生成内容进行全面控制。现有技术旨在使生成内容公平且安全（非暴力/露骨），以解决负责任T2I模型的问题。然而，这些方法仍局限于单独处理责任概念的各个方面，同时缺乏可解释性。此外，它们通常需要对原始模型进行修改，这会损害模型性能。在本研究中，我们提出了一种独特技术，通过同时考虑广泛概念，以可扩展的方式实现公平和安全的负责任T2I生成。关键思想是通过外部即插即用机制蒸馏目标T2I管道，该机制学习针对目标T2I管道的可解释复合负责任空间。我们使用知识蒸馏和概念白化来实现这一点。在推理时，学习的空间被用于调制生成内容。典型的T2I管道为我们的方法提供了两个插入点，即文本嵌入空间和扩散模型潜在空间。我们为这两个点开发了模块，并通过一系列强结果展示了我们方法的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 用即插即用机制蒸馏T2I管道
    * 学习可解释复合负责任空间
    * 在文本嵌入和潜在空间调制生成内容
</details>
</details>

---


<details>
<summary><b> Beyond Words: Advancing Long-Text Image Generation via Multimodal Autoregressive Models</b></summary>

* **Authors:** Alex Jinpeng Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, Min Li
* **arXiv ID:** 2503.20198
* **One-liner:** Developed a model for generating high-quality long-text images with unprecedented fidelity.
* **Published in:** arxiv (26 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.20198) | [[PDF]](https://arxiv.org/pdf/2503.20198) | [[Code]](https://fingerrec.github.io/longtextar/)

> **核心创新**
> 通过新颖二进制分词器解决文本生成质量的瓶颈问题。

<details>
    <summary>Abstract</summary>
    自回归和扩散模型的最新进展在生成带有短场景文本的图像方面取得了强劲性能。然而，生成连贯的长文本图像，如幻灯片或文档中的段落，仍然是当前生成模型的主要挑战。我们提出了首个专门针对长文本图像生成的工作，解决了现有文本到图像系统通常只处理简短短语或单个句子的关键空白。通过对最先进的自回归生成模型的综合分析，我们识别出图像分词器是文本生成质量的关键瓶颈。为了解决这个问题，我们引入了一种新颖的文本聚焦二进制分词器，优化用于捕捉详细的场景文本特征。利用我们的分词器，我们开发了\ModelName，一种多模态自回归模型，在生成高质量长文本图像方面表现出前所未有的保真度。我们的模型提供强大的可控性，允许定制文本属性，如字体样式、大小、颜色和对齐方式。广泛的实验表明，\ModelName~在准确、一致和灵活地生成长文本方面显著优于SD3.5 Large~\cite{sd3}和GPT4o~\cite{gpt4o}与DALL-E 3~\cite{dalle3}。除了技术成就外，\ModelName~为创新应用如交错文档和PowerPoint生成开辟了令人兴奋的机会，确立了长文本图像生成的新前沿。
</details>

<details>
    <summary>Key points</summary>
    * 引入文本聚焦二进制分词器以捕捉详细场景文本
    * 构建多模态自回归模型用于长文本生成
    * 实现文本属性如字体和对齐的自定义
</details>
</details>

---


<details>
<summary><b> Lumina-Image 2.0: A Unified and Efficient Image Generative Framework</b></summary>

* **Authors:** Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, Xiangyang Zhu, Manyuan Zhang, Will Beddow, Erwann Millon, Victor Perez, Wenhai Wang, Conghui He, Bo Zhang, Xiaohong Liu, Hongsheng Li, Yu Qiao, Chang Xu, Peng Gao
* **arXiv ID:** 2503.21758
* **One-liner:** Advanced text-to-image generation with Lumina-Image 2.0, emphasizing unification and efficiency.
* **Published in:** arxiv (27 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.21758) | [[PDF]](https://arxiv.org/pdf/2503.21758) | [[Code]](https://github.com/Alpha-VLLM/Lumina-Image-2.0?tab=readme-ov-file)

> **核心创新**
> 采用统一架构和字幕系统以改善跨模态交互。

<details>
    <summary>Abstract</summary>
    我们介绍了Lumina-Image 2.0，一个先进的文本到图像生成框架，与先前工作Lumina-Next相比取得了显著进展。Lumina-Image 2.0基于两个关键原则构建：（1）统一性——它采用统一架构（Unified Next-DiT），将文本和图像令牌视为联合序列，实现自然的跨模态交互并允许无缝任务扩展。此外，由于高质量字幕器可以提供语义对齐良好的文本-图像训练对，我们引入了统一字幕系统Unified Captioner（UniCap），专门为T2I生成任务设计。UniCap擅长生成全面准确的字幕，加速收敛并增强提示遵循。（2）效率——为了提高我们提出模型的效率，我们开发了多阶段渐进训练策略，并引入了不损害图像质量的推理加速技术。在学术基准和公共文本到图像竞技场上的广泛评估显示，Lumina-Image 2.0即使只有26亿参数也能提供强劲性能，突显其可扩展性和设计效率。我们已在<a href="https://github.com/Alpha-VLLM/Lumina-Image-2.0" rel="external noopener nofollow" class="link-external link-https">此https URL</a>发布了训练细节、代码和模型。
</details>

<details>
    <summary>Key points</summary>
    * 使用Unified Next-DiT处理联合文本-图像令牌序列
    * 引入Unified Captioner以生成准确字幕
    * 应用多阶段训练和推理加速技术
</details>
</details>

---


<details>
<summary><b> Geometrical Properties of Text Token Embeddings for Strong Semantic Binding in Text-to-Image Generation</b></summary>

* **Authors:** Hoigi Seo, Junseo Bang, Haechang Lee, Joohoon Lee, Byung Hyun Lee, Se Young Chun
* **arXiv ID:** 2503.23011
* **One-liner:** Proposed TokeBi, a training-free framework for strong semantic binding in T2I models.
* **Published in:** arxiv (29 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.23011) | [[PDF]](https://arxiv.org/pdf/2503.23011) | [[Code]]()

> **核心创新**
> 利用令牌嵌入的几何性质改进交叉注意力图。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）模型在涉及多个对象和属性的复杂场景中常常遭受文本-图像错位。语义绑定试图通过文本或潜在优化与交叉注意力（CA）图的调制，将生成的属性和对象与其对应的名词短语（NP）关联起来；然而，影响语义绑定的因素仍未充分探索。在这里，我们研究了文本令牌嵌入及其CA图的几何性质。我们发现令牌嵌入的几何性质，特别是角度距离和范数，是CA图分化的关键因素。这些理论发现导致了我们提出的无需训练文本嵌入感知T2I框架，称为\textbf{TokeBi}，用于强语义绑定。TokeBi包括因果感知投影输出（CAPO）用于区分NP间CA图，以及自适应令牌混合（ATM）用于增强NP间分离同时保持NP内内聚在CA图中。广泛的实验证实，TokeBi在多种基线和数据集上优于先前方法。
</details>

<details>
    <summary>Key points</summary>
    * 研究令牌嵌入的角度距离和范数
    * 实现因果感知投影输出以区分NP间CA图
    * 使用自适应令牌混合保持NP内内聚
</details>
</details>

---


<details>
<summary><b> LayerCraft: Enhancing Text-to-Image Generation with CoT Reasoning and Layered Object Integration</b></summary>

* **Authors:** Yuyao Zhang, Jinghao Li, Yu-Wing Tai
* **arXiv ID:** 2504.00010
* **One-liner:** Introduced LayerCraft, a modular framework for structured image generation and editing using LLMs.
* **Published in:** arxiv (25 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.00010) | [[PDF]](https://arxiv.org/pdf/2504.00010) | [[Code]](https://github.com/PeterYYZhang/LayerCraft)

> **核心创新**
> 实现可控场景分解和对象集成而无需重新训练。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）生成已取得显著进展，但现有系统仍缺乏对空间组合、对象一致性和多步编辑的直观控制。我们提出了$\textbf{LayerCraft}$，一个模块化框架，使用大型语言模型（LLMs）作为自主代理来编排结构化、分层图像生成和编辑。LayerCraft支持两个关键能力：（1）通过链式思维（CoT）推理从简单提示进行$\textit{结构化生成}$，使其能够分解场景、推理对象放置并以可控、可解释的方式指导组合；（2）$\textit{分层对象集成}$，允许用户在多样图像或场景中插入和自定义对象——如角色或道具——同时保持身份、上下文和风格。该系统包括一个协调代理，$\textbf{ChainArchitect}$用于CoT驱动的布局规划，以及$\textbf{对象集成网络（OIN）}$用于使用现成T2I模型进行无缝图像编辑而无需重新训练。通过批量拼贴编辑和叙事场景生成等应用，LayerCraft使非专家能够以最少的手动努力迭代设计、自定义和优化视觉内容。代码将在<a href="https://github.com/PeterYYZhang/LayerCraft" rel="external noopener nofollow" class="link-external link-https">此https URL</a>发布。
</details>

<details>
    <summary>Key points</summary>
    * 使用LLMs作为自主代理进行链式思维推理
    * 实现ChainArchitect用于布局规划
    * 开发对象集成网络用于无缝编辑
</details>
</details>

---


<details>
<summary><b> Compass Control: Multi Object Orientation Control for Text-to-Image Generation</b></summary>

* **Authors:** Rishubh Parihar, Vaibhav Agrawal, Sachidanand VS, R. Venkatesh Babu
* **arXiv ID:** 2504.06752
* **One-liner:** Enabled precise multi-object orientation control in T2I diffusion models.
* **Published in:** arxiv (9 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.06752) | [[PDF]](https://arxiv.org/pdf/2504.06752) | [[Code]](https://github.com/rishubhpar/compass-control-src)

> **核心创新**
> 通过方向感知指南针令牌和约束交叉注意力图来条件化模型。

<details>
    <summary>Abstract</summary>
    现有控制文本到图像扩散模型的方法虽然强大，但不允许显式的3D对象中心控制，如精确控制对象方向。在本研究中，我们解决了文本到图像扩散模型中的多对象方向控制问题。这使得能够生成具有每个对象精确方向控制的多样多对象场景。关键思想是通过一组方向感知的\textbf{指南针}令牌（每个对象一个）以及文本令牌来条件化扩散模型。一个轻量级编码器网络以对象方向为输入预测这些指南针令牌。该模型在合成数据集上训练，该数据集包含程序生成的场景，每个场景在纯背景上包含一个或两个3D资源。然而，直接训练此框架会导致方向控制差以及对象间纠缠。为了缓解这一点，我们在生成过程中进行干预，并将每个指南针令牌的交叉注意力图约束到其对应的对象区域。训练后的模型能够实现精确方向控制，适用于a）训练期间未见过的复杂对象和b）超过两个对象的多对象场景，表明强大的泛化能力。此外，当与个性化方法结合时，我们的方法能在多样上下文中精确控制新对象的方向。我们的方法在方向控制和文本对齐方面达到最先进水平，通过广泛评估和用户研究量化。
</details>

<details>
    <summary>Key points</summary>
    * 用轻量级编码器预测指南针令牌
    * 将交叉注意力图约束到对象区域
    * 实现未见对象和多对象场景的泛化
</details>
</details>

---


<details>
<summary><b> Towards NSFW-Free Text-to-Image Generation via Safety-Constraint Direct Preference Optimization</b></summary>

* **Authors:** Shouwei Ruan, Zhenyu Wu, Yao Huang, Ruochen Zhang, Yitong Sun, Caixin Kang, Xingxing Wei
* **arXiv ID:** 2504.14290
* **One-liner:** Proposed SC-DPO for safety alignment in T2I models, balancing safety and quality.
* **Published in:** arxiv (19 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.14290) | [[PDF]](https://arxiv.org/pdf/2504.14290) | [[Code]]()

> **核心创新**
> 将安全约束集成到偏好优化中，并使用安全成本模型。

<details>
    <summary>Abstract</summary>
    确保生成内容的安全性仍然是文本到图像（T2I）生成的基本挑战。现有研究要么无法在潜在有害概念下保证完全安全，要么难以平衡安全性与生成质量。为了解决这些问题，我们提出了安全约束直接偏好优化（SC-DPO），一种用于T2I模型安全对齐的新框架。SC-DPO将安全约束集成到一般人类偏好校准中，旨在最大化生成人类偏好样本的可能性，同时最小化生成输出的安全成本。在SC-DPO中，我们引入了一个安全成本模型来准确量化图像的有害程度，并使用提出的对比学习和成本锚定目标有效训练它。为了应用SC-DPO进行有效的T2I安全对齐，我们构建了SCP-10K，一个包含丰富有害概念的安全约束偏好数据集，该数据集混合了有害和清洁指令下的安全约束偏好对，进一步缓解安全性与样本质量之间的权衡。此外，我们为SC-DPO提出了动态聚焦机制（DFM），促进模型学习困难偏好对样本。广泛的实验表明，SC-DPO优于现有方法，有效防御各种NSFW内容，同时保持最佳样本质量和人类偏好对齐。此外，SC-DPO对旨在生成有害内容的对抗提示表现出韧性。
</details>

<details>
    <summary>Key points</summary>
    * 引入安全成本模型以量化有害程度
    * 使用对比学习和成本锚定目标
    * 构建SCP-10K数据集并应用动态聚焦机制
</details>
</details>

---


<details>
<summary><b> LLM-Enabled Style and Content Regularization for Personalized Text-to-Image Generation</b></summary>

* **Authors:** Anran Yu, Wei Feng, Yaochen Zhang, Xiang Li, Lei Meng, Lei Wu, Xiangxu Meng
* **arXiv ID:** 2504.15309
* **One-liner:** Enhanced personalized T2I generation with style refinement and content preservation strategies.
* **Published in:** arxiv (19 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.15309) | [[PDF]](https://arxiv.org/pdf/2504.15309) | [[Code]]()

> **核心创新**
> 优化风格嵌入并保留模型泛化以改善可控性。

<details>
    <summary>Abstract</summary>
    个性化文本到图像生成随着Stable Diffusion的出现而迅速发展。现有方法通常使用嵌入标识符微调模型，但由于文本可控性降低，常常难以实现充分的风格化和准确的图像内容。在本文中，我们提出了风格优化和内容保留策略。风格优化策略利用视觉推理提示和参考图像的语义信息来优化风格嵌入，允许更精确和一致的风格信息表示。内容保留策略通过保留模型的泛化能力来解决内容偏差问题，确保在不损害风格化的情况下增强文本可控性。实验结果验证了我们的方法在生成一致和个性化文本到图像输出方面实现了优越性能。
</details>

<details>
    <summary>Key points</summary>
    * 利用语义信息优化风格嵌入
    * 解决内容偏差以保持文本可控性
    * 实现一致和个性化输出
</details>
</details>

---


<details>
<summary><b> RefVNLI: Towards Scalable Evaluation of Subject-driven Text-to-image Generation</b></summary>

* **Authors:** Aviv Slobodkin, Hagai Taitelbaum, Yonatan Bitton, Brian Gordon, Michal Sokolik, Nitzan Bitton Guetta, Almog Gueta, Royi Rassin, Dani Lischinski, Idan Szpektor
* **arXiv ID:** 2504.17502
* **One-liner:** Introduced RefVNLI, a cost-effective metric for evaluating subject-driven T2I generation.
* **Published in:** arxiv (24 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.17502) | [[PDF]](https://arxiv.org/pdf/2504.17502) | [[Code]]()

> **核心创新**
> 在单次运行中评估文本对齐和主题保留。

<details>
    <summary>Abstract</summary>
    主题驱动文本到图像（T2I）生成旨在生成与给定文本描述对齐的图像，同时保留参考主题图像的视觉身份。尽管其下游应用广泛——从图像生成中的增强个性化到视频渲染中的一致角色表示——该领域的进展受到缺乏可靠自动评估的限制。现有方法要么只评估任务的一个方面（即文本对齐或主题保留），要么与人类判断不一致，要么依赖昂贵的基于API的评估。为了解决这一空白，我们引入了RefVNLI，一种成本效益高的指标，在单次运行中评估文本对齐和主题保留。在从视频推理基准和图像扰动衍生的大规模数据集上训练，RefVNLI在多个基准和主题类别（例如\emph{动物}、\emph{对象}）上优于或统计上匹配现有基线，在文本对齐上实现高达6.4分的增益，在主题保留上实现高达5.9分的增益。
</details>

<details>
    <summary>Key points</summary>
    * 在从视频推理基准衍生的大规模数据集上训练
    * 在文本对齐和主题保留上优于基线
    * 提供高达6.4分对齐增益和5.9分保留增益
</details>
</details>

---


<details>
<summary><b> TextTIGER: Text-based Intelligent Generation with Entity Prompt Refinement for Text-to-Image Generation</b></summary>

* **Authors:** Shintaro Ozaki, Kazuki Hayashi, Yusuke Sakai, Jingun Kwon, Hidetaka Kamigaito, Katsuhiko Hayashi, Manabu Okumura, Taro Watanabe
* **arXiv ID:** 2504.18269
* **One-liner:** Improved image generation by refining prompts with augmented entity knowledge.
* **Published in:** arxiv (25 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.18269) | [[PDF]](https://arxiv.org/pdf/2504.18269) | [[Code]]()

> **核心创新**
> 提出了TextTIGER，使用LLM总结增强的实体描述以改善图像生成。

<details>
    <summary>Abstract</summary>
    从包含特定实体的提示生成图像需要模型尽可能保留实体特定知识。然而，由于实体数量庞大且不断涌现，完全记忆这些知识不切实际。为此，我们提出了基于文本的智能生成与实体提示优化（TextTIGER），该方法增强提示中实体的知识，然后使用大型语言模型（LLM）总结增强的描述，以减轻较长输入导致的性能下降。为评估我们的方法，我们引入了WiT-Cub（带标题和简单背景解释的WiT），一个包含标题、图像和实体列表的数据集。在四个图像生成模型和五个LLM上的实验表明，与仅使用标题提示相比，TextTIGER在标准指标（IS、FID和CLIPScore）上提高了图像生成性能。此外，多位注释者的评估证实总结的描述更具信息性，验证了LLM生成简洁而丰富描述的能力。这些发现表明，通过增强和总结实体相关描述来优化提示，可以增强图像生成能力。代码和数据集将在接受后提供。
</details>

<details>
    <summary>Key points</summary>
    * 增强提示中实体的知识
    * 使用LLM总结增强的描述
    * 使用WiT-Cub数据集和标准指标进行评估
</details>
</details>

---


<details>
<summary><b> T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT</b></summary>

* **Authors:** Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, Hongsheng Li
* **arXiv ID:** 2505.00703
* **One-liner:** Enhanced text-to-image generation with bi-level chain-of-thought reasoning.
* **Published in:** arxiv (1 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.00703) | [[PDF]](https://arxiv.org/pdf/2505.00703) | [[Code]](https://github.com/CaraJ7/T2I-R1)

> **核心创新**
> 引入了T2I-R1，一种使用RL和双层CoT进行语义和令牌级规划的模型。

<details>
    <summary>Abstract</summary>
    大型语言模型的最新进展展示了思维链（CoT）和强化学习（RL）如何提高性能。然而，在视觉生成领域应用此类推理策略仍未被充分探索。在本文中，我们提出了T2I-R1，一种新颖的推理增强文本到图像生成模型，由具有双层CoT推理过程的RL驱动。具体来说，我们识别出两个层次的CoT可用于增强生成的不同阶段：（1）语义级CoT用于提示的高层规划，以及（2）令牌级CoT用于逐块生成过程中的低层像素处理。为了更好地协调这两个层次的CoT，我们引入了BiCoT-GRPO与生成奖励集成，在同一训练步骤中无缝优化两个生成CoT。通过将我们的推理策略应用于基线模型Janus-Pro，我们在T2I-CompBench上实现了13%的改进，在WISE基准上实现了19%的改进，甚至超过了最先进模型FLUX.1。代码可在：此HTTPS URL获取。
</details>

<details>
    <summary>Key points</summary>
    * 语义级CoT用于高层提示规划
    * 令牌级CoT用于低层像素处理
    * 使用BiCoT-GRPO与集成奖励进行优化
</details>
</details>

---


<details>
<summary><b> Deconstructing Bias: A Multifaceted Framework for Diagnosing Cultural and Compositional Inequities in Text-to-Image Generative Models</b></summary>

* **Authors:** Muna Numan Said, Aarib Zaidi, Rabia Usman, Sonia Okon, Praneeth Medepalli, Kevin Zhu, Vasu Sharma, Sean O&#39;Brien
* **arXiv ID:** 2505.01430
* **One-liner:** Benchmarked cultural biases in text-to-image models using CIS metric.
* **Published in:** arxiv (5 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.01430) | [[PDF]](https://arxiv.org/pdf/2505.01430) | [[Code]]()

> **核心创新**
> 开发了CIS以评估文化保真度并识别图像生成中的偏见。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）模型的变革潜力取决于其从文本提示合成文化多样、逼真图像的能力。然而，这些模型常常延续其训练数据中嵌入的文化偏见，导致系统性误表征。本文基准测试了组件包含分数（CIS），一种设计用于评估跨文化上下文图像生成保真度的指标。通过涉及2,400张图像的广泛分析，我们量化了组合脆弱性和上下文错位方面的偏见，揭示了西方和非西方文化提示之间的显著性能差距。我们的发现强调了数据不平衡、注意力熵和嵌入叠加对模型公平性的影响。通过使用CIS基准测试如Stable Diffusion等模型，我们为增强AI生成图像中文化包容性的架构和数据中心干预提供了见解。这项工作通过提供诊断和减轻T2I生成中偏见的全面工具，推动了该领域的发展，倡导更公平的AI系统。
</details>

<details>
    <summary>Key points</summary>
    * 设计组件包含分数（CIS）指标
    * 使用2,400张图像分析偏见
    * 为架构和数据中心干预提供见解
</details>
</details>

---


<details>
<summary><b> Improving Physical Object State Representation in Text-to-Image Generative Systems</b></summary>

* **Authors:** Tianle Chen, Chaitanya Chakka, Deepti Ghadiyaram
* **arXiv ID:** 2505.02236
* **One-liner:** Improved generation of object states in text-to-image models.
* **Published in:** arxiv (4 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.02236) | [[PDF]](https://arxiv.org/pdf/2505.02236) | [[Code]](https://github.com/cskyl/Object-State-Bench)

> **核心创新**
> 创建了一个用于合成数据的管道，并微调模型以更好地表示对象状态。

<details>
    <summary>Abstract</summary>
    当前文本到图像生成模型难以准确表示对象状态（例如，“没有瓶子的桌子”，“空杯子”）。在这项工作中，我们首先设计了一个全自动管道来生成高质量合成数据，准确捕捉不同状态下的对象。接下来，我们在这个合成数据上微调了几个开源文本到图像模型。我们通过使用GPT4o-mini量化生成图像与其提示的对齐来评估微调模型的性能，并在公共GenAI-Bench数据集上实现了四个模型平均绝对改进8%以上。我们还策划了200个提示集合，特别关注常见对象的不同物理状态。我们展示了在该数据集上平均24%以上的显著改进。我们发布了所有评估提示和代码。
</details>

<details>
    <summary>Key points</summary>
    * 设计自动管道用于合成数据生成
    * 在合成数据上微调模型
    * 使用GPT4o-mini和自定义数据集进行评估
</details>
</details>

---


<details>
<summary><b> MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation</b></summary>

* **Authors:** Mingcheng Li, Xiaolu Hou, Ziyang Liu, Dingkang Yang, Ziyun Qian, Jiawei Chen, Jinjie Wei, Yue Jiang, Qingyao Xu, Lihua Zhang
* **arXiv ID:** 2505.02648
* **One-liner:** Enhanced complex scene generation with multi-agent collaboration.
* **Published in:** arxiv (5 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.02648) | [[PDF]](https://arxiv.org/pdf/2505.02648) | [[Code]]()

> **核心创新**
> 提出了MCCD，使用多智能体场景解析和分层组合扩散。

<details>
    <summary>Abstract</summary>
    扩散模型在文本到图像生成中表现出色。然而，现有方法在处理涉及多个对象、特征和关系的复杂提示时常常遇到性能瓶颈。因此，我们提出了基于多智能体协作的组合扩散（MCCD）用于复杂场景的文本到图像生成。具体来说，我们设计了一个基于多智能体协作的场景解析模块，生成一个包含多个具有不同任务智能体的智能体系统，利用MLLM有效提取各种场景元素。此外，分层组合扩散使用高斯掩码和滤波来细化边界框区域并通过区域增强增强对象，从而实现复杂场景的准确和高保真生成。综合实验表明，我们的MCCD在无需训练的情况下显著提高了基线模型的性能，在复杂场景生成中提供了显著优势。
</details>

<details>
    <summary>Key points</summary>
    * 多智能体协作用于场景解析
    * 分层组合扩散与高斯掩码
    * 无需训练的复杂场景改进
</details>
</details>

---


<details>
<summary><b> HCMA: Hierarchical Cross-model Alignment for Grounded Text-to-Image Generation</b></summary>

* **Authors:** Hang Wang, Zhi-Qi Cheng, Chenhao Lin, Chao Shen, Lei Zhang
* **arXiv ID:** 2505.06512
* **One-liner:** Achieved better spatial control and semantic fidelity in text-to-image generation.
* **Published in:** arxiv (10 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.06512) | [[PDF]](https://arxiv.org/pdf/2505.06512) | [[Code]](https://github.com/hwang-cs-ime/HCMA)

> **核心创新**
> 引入了HCMA框架，具有全局和局部对齐模块用于基于生成。

<details>
    <summary>Abstract</summary>
    文本到图像合成已发展到模型可以从自然语言提示生成视觉上引人入胜的图像。然而，现有方法常常无法协调高层语义保真度与显式空间控制，特别是在涉及多个对象、细微关系或复杂布局的场景中。为弥合这一差距，我们提出了分层跨模态对齐（HCMA）框架用于基于文本到图像的生成。HCMA将两个对齐模块集成到每个扩散采样步骤中：一个全局模块持续对齐潜在表示与文本描述以确保场景级一致性，以及一个局部模块使用边界框布局将对象锚定在指定位置，实现细粒度空间控制。在MS-COCO 2014验证集上的广泛实验显示，HCMA超越了最先进的基线，在Frechet Inception Distance（FID）上实现了0.69的改进，在CLIP Score上实现了0.0295的增益。这些结果证明了HCMA在忠实捕捉复杂文本语义的同时遵守用户定义空间约束的有效性，为语义基础的图像生成提供了稳健解决方案。我们的代码可在此HTTPS URL获取。
</details>

<details>
    <summary>Key points</summary>
    * 全局对齐用于场景级一致性
    * 局部对齐与边界框布局
    * 将模块集成到扩散采样步骤中
</details>
</details>

---


<details>
<summary><b> IMAGE-ALCHEMY: Advancing subject fidelity in personalised text-to-image generation</b></summary>

* **Authors:** Amritanshu Tiwari, Cherish Puniani, Kaustubh Sharma, Ojasva Nema
* **arXiv ID:** 2505.10743
* **One-liner:** Improved personalization in text-to-image models with reduced forgetting.
* **Published in:** arxiv (15 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.10743) | [[PDF]](https://arxiv.org/pdf/2505.10743) | [[Code]]()

> **核心创新**
> 开发了一个两阶段管道，使用LoRA微调和分割进行主题集成。

<details>
    <summary>Abstract</summary>
    文本到图像扩散模型的最新进展，特别是Stable Diffusion，使得生成高度详细和语义丰富的图像成为可能。然而，个性化这些模型以基于少量参考图像表示新主题仍然具有挑战性。这常常导致灾难性遗忘、过拟合或大计算开销。我们提出了一个两阶段管道，通过利用Stable Diffusion XL（SDXL）模型U-Net中注意力权重的LoRA微调来解决这些限制。首先，我们使用未修改的SDXL通过将主题替换为其类别标签来生成通用场景。然后，我们通过分割驱动的图像到图像（Img2Img）管道选择性插入个性化主题，该管道使用训练的LoRA权重。该框架将主题编码与整体组合隔离，从而在集成新主题时以高保真方式保留SDXL的更广泛生成能力。我们的方法在SDXL上实现了0.789的DINO相似度分数，优于现有的个性化文本到图像方法。
</details>

<details>
    <summary>Key points</summary>
    * 基于注意力权重的LoRA微调
    * 分割驱动的图像到图像管道
    * 隔离主题编码以保留生成能力
</details>
</details>

---


<details>
<summary><b> CompAlign: Improving Compositional Text-to-Image Generation with a Complex Benchmark and Fine-Grained Feedback</b></summary>

* **Authors:** Yixin Wan, Kai-Wei Chang
* **arXiv ID:** 2505.11178
* **One-liner:** Advanced evaluation and improvement of compositional image generation.
* **Published in:** arxiv (16 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.11178) | [[PDF]](https://arxiv.org/pdf/2505.11178) | [[Code]]()

> **核心创新**
> 引入了CompAlign基准和CompQuest框架用于细粒度评估和对齐。

<details>
    <summary>Abstract</summary>
    最先进的T2I模型能够根据文本提示生成高分辨率图像。然而，它们仍然难以准确描绘指定多个对象、属性和空间关系的组合场景。我们提出了CompAlign，一个强调评估3D空间关系描绘的挑战性基准，用于评估和改进模型在组合图像生成上的性能。CompAlign包含900个复杂多主题图像生成提示，结合了数值和3D空间关系以及变化的属性绑定。我们的基准极具挑战性，包含具有3+个生成主题和复杂3D空间关系的生成任务。此外，我们提出了CompQuest，一个可解释且准确的评估框架，将复杂提示分解为原子子问题，然后利用MLLM对模型生成图像中生成元素每个方面的正确性提供细粒度二元反馈。这实现了生成图像与组合提示之间对齐的精确量化。进一步，我们提出了一个对齐框架，使用CompQuest的反馈作为偏好信号来改进扩散模型的组合图像生成能力。使用可调整的每图像偏好，我们的方法易于扩展和适应不同任务。对9个T2I模型的评估揭示：（1）模型在具有更复杂3D空间配置的组合任务上表现更差，以及（2）开源可访问模型与闭源商业模型之间存在显著性能差距。使用CompAlign进行模型对齐的进一步实证研究产生了有希望的结果：对齐后的扩散模型在组合准确性上实现了显著改进，特别是在复杂生成任务上，优于先前方法。
</details>

<details>
    <summary>Key points</summary>
    * 创建CompAlign基准与复杂提示
    * 开发CompQuest用于原子子问题评估
    * 使用反馈进行模型对齐和改进
</details>
</details>

---


<details>
<summary><b> Diff-MM: Exploring Pre-trained Text-to-Image Generation Model for Unified Multi-modal Object Tracking</b></summary>

* **Authors:** Shiyu Xuan, Zechao Li, Jinhui Tang
* **arXiv ID:** 2505.12606
* **One-liner:** Enhanced multi-modal object tracking using text-to-image model features.
* **Published in:** arxiv (19 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.12606) | [[PDF]](https://arxiv.org/pdf/2505.12606) | [[Code]]()

> **核心创新**
> 提出了Diff-MM，利用Stable Diffusion UNet进行统一多模态跟踪。

<details>
    <summary>Abstract</summary>
    多模态对象跟踪整合了深度、热红外、事件流和语言等辅助模态，提供RGB图像之外的额外信息，在复杂场景中显示出提高跟踪稳定性的巨大潜力。现有方法通常从基于RGB的跟踪器开始，仅从训练数据中学习理解辅助模态。受限于有限的多模态训练数据，这些方法的性能不尽人意。为缓解这一限制，本工作提出了统一多模态跟踪器Diff-MM，通过利用预训练文本到图像生成模型的多模态理解能力。Diff-MM通过提出的并行特征提取管道，利用预训练Stable Diffusion的UNet作为跟踪特征提取器，使得对象跟踪能够处理成对图像输入。我们进一步引入了多模态子模块调优方法，学习获取不同模态之间的互补信息。通过利用生成模型中的广泛先验知识，我们实现了具有统一参数的RGB-N/D/T/E跟踪的统一跟踪器。实验结果表明，我们的方法相比最近提出的跟踪器表现出有希望的性能，例如，其在TNL2K上的AUC优于OneTracker 8.3%。
</details>

<details>
    <summary>Key points</summary>
    * 使用Stable Diffusion的UNet作为特征提取器
    * 并行特征提取管道
    * 多模态子模块调优用于互补信息
</details>
</details>

---


<details>
<summary><b> Emerging Properties in Unified Multimodal Pretraining</b></summary>

* **Authors:** Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, Haoqi Fan
* **arXiv ID:** 2505.14683
* **One-liner:** Developed an open-source unified multimodal model with advanced reasoning.
* **Published in:** arxiv (20 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.14683) | [[PDF]](https://arxiv.org/pdf/2505.14683) | [[Code]](https://github.com/bytedance-seed/BAGEL)

> **核心创新**
> 引入了BAGEL，一个在多样多模态数据上预训练的仅解码器模型，用于理解和生成。

<details>
    <summary>Abstract</summary>
    统一多模态理解和生成在尖端专有系统中显示出令人印象深刻的能力。在这项工作中，我们引入了BAGEL，一个开源基础模型，原生支持多模态理解和生成。BAGEL是一个统一的仅解码器模型，在大规模交错文本、图像、视频和网络数据中策划的数万亿令牌上预训练。当用这种多样多模态交错数据扩展时，BAGEL在复杂多模态推理中展现出新兴能力。因此，它在标准基准上在生成和理解方面显著优于开源统一模型，同时展现出先进的多模态推理能力，如自由形式图像操作、未来帧预测、3D操作和世界导航。希望促进多模态研究的进一步机会，我们分享了关键发现、预训练细节、数据创建协议，并向社区发布了我们的代码和检查点。项目页面在此HTTPS URL。
</details>

<details>
    <summary>Key points</summary>
    * 在交错数据上预训练数万亿令牌
    * 支持多模态理解和生成
    * 在复杂推理中展现新兴能力
</details>
</details>

---


<details>
<summary><b> Harnessing Caption Detailness for Data-Efficient Text-to-Image Generation</b></summary>

* **Authors:** Xinran Wang, Muxi Diao, Yuanzhi Liu, Chunyu Wang, Kongming Liang, Zhanyu Ma, Jun Guo
* **arXiv ID:** 2505.15172
* **One-liner:** Proposed a new metric for caption detailness in T2I training, improving model performance with efficient data selection.
* **Published in:** arxiv (21 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.15172) | [[PDF]](https://arxiv.org/pdf/2505.15172) | [[Code]]()

> **核心创新**
> 引入ICR和AOD指标来评估描述详细度，使T2I模型能在高详细描述上实现更优训练。

<details>
    <summary>Abstract</summary>
    使用详细描述训练文生图模型可显著提升生成质量。现有方法常依赖简单指标（如描述长度）来代表训练集中描述的详细程度。本文提出一种基于两方面的新指标来估计描述详细度：图像覆盖率（ICR），评估描述是否覆盖图像中所有区域/对象；平均对象详细度（AOD），量化每个对象描述的详细程度。通过在COCO数据集上使用ShareGPT4V描述进行实验，我们证明使用高ICR和AOD描述训练的T2I模型在DPG及其他基准测试中表现更优。值得注意的是，我们的指标实现了更有效的数据选择——仅使用20%的高详细数据训练即可超越全数据集训练和基于长度的选择方法，提升了对齐和重建能力。这些发现突显了在T2I任务中，细节感知指标比基于长度的启发式方法更为关键。
</details>

<details>
    <summary>Key points</summary>
    * 定义图像覆盖率（ICR）以评估描述对图像区域的覆盖程度。
    * 量化平均对象详细度（AOD）以衡量对象描述的详细程度。
    * 证明使用20%高详细数据训练优于全数据集和基于长度的方法。
</details>
</details>

---


<details>
<summary><b> IA-T2I: Internet-Augmented Text-to-Image Generation</b></summary>

* **Authors:** Chuanhao Li, Jianwen Sun, Yukang Feng, Mingliang Zhai, Yifan Chang, Kaipeng Zhang
* **arXiv ID:** 2505.15779
* **One-liner:** Developed an Internet-Augmented T2I framework to handle uncertain knowledge in prompts using reference images.
* **Published in:** arxiv (21 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.15779) | [[PDF]](https://arxiv.org/pdf/2505.15779) | [[Code]]()

> **核心创新**
> 集成主动检索、分层图像选择和自反思机制，以增强T2I生成在不确定场景下的表现。

<details>
    <summary>Abstract</summary>
    当前文生图生成模型取得了显著成果，但在文本提示中隐含知识不确定的场景下表现不佳。例如，二月发布的T2I模型难以生成四月首映电影的合适海报，因为角色设计和风格对模型而言不确定。为解决此问题，我们提出一个互联网增强的文生图生成框架，通过提供参考图像来使T2I模型明确此类不确定知识。具体而言，设计了一个主动检索模块，根据给定文本提示判断是否需要参考图像；引入分层图像选择模块，从图像搜索引擎返回的结果中寻找最合适的图像以增强T2I模型；提出自反思机制，持续评估和优化生成图像，确保与文本提示忠实对齐。为评估所提框架性能，我们收集了一个名为Img-Ref-T2I的数据集，其中文本提示包含三种不确定知识类型：（1）已知但罕见；（2）未知；（3）模糊。此外，我们精心设计了一个复杂提示来指导GPT-4o进行偏好评估，其评估准确度与人类偏好评估相似。实验结果表明我们的框架有效，在人类评估中优于GPT-4o约30%。
</details>

<details>
    <summary>Key points</summary>
    * 设计主动检索模块以判断是否需要参考图像。
    * 引入分层图像选择模块以检索合适图像。
    * 实现自反思机制以持续优化图像。
</details>
</details>

---


<details>
<summary><b> Self-Rewarding Large Vision-Language Models for Optimizing Prompts in Text-to-Image Generation</b></summary>

* **Authors:** Hongji Yang, Yucheng Zhou, Wencheng Han, Jianbing Shen
* **arXiv ID:** 2505.16763
* **One-liner:** Created a prompt optimization framework using LVLMs for AI feedback, reducing reliance on human annotations.
* **Published in:** arxiv (22 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.16763) | [[PDF]](https://arxiv.org/pdf/2505.16763) | [[Code]]()

> **核心创新**
> 使用LVLM作为求解器和奖励模型，在强化学习中实现自我改进的提示重写。

<details>
    <summary>Abstract</summary>
    文生图模型能基于给定文本提示生成高质量图像，但制作这些提示常需专业词汇。现有方法通过大量人工标注数据和训练的美学评估模型来监督训练重写模型。为减轻模型训练对数据规模的依赖和训练模型引入的偏差，我们提出一种新颖的提示优化框架，旨在将简单用户提示重述为文生图模型的复杂提示。具体而言，我们使用大型视觉语言模型作为求解器来重写用户提示，同时使用LVLM作为奖励模型，对优化提示生成图像的美学和对齐度进行评分。我们利用LVLM的先验知识提供奖励，即AI反馈，而非繁琐的人类反馈。求解器和奖励模型统一为一个模型，并在强化学习中迭代，通过给出解决方案和自我评判实现自我改进。在两个流行数据集上的结果表明，我们的方法优于其他强竞争者。
</details>

<details>
    <summary>Key points</summary>
    * 使用LVLM将用户提示重写为复杂版本。
    * 应用LVLM作为奖励模型来评分图像美学和对齐度。
    * 在强化学习中集成求解器和奖励模型以迭代自我改进。
</details>
</details>

---


<details>
<summary><b> RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning</b></summary>

* **Authors:** Mingrui Wu, Lu Wang, Pu Zhao, Fangkai Yang, Jianjin Zhang, Jianfeng Liu, Yuefeng Zhan, Weihao Han, Hao Sun, Jiayi Ji, Xiaoshuai Sun, Qingwei Lin, Weiwei Deng, Dongmei Zhang, Feng Sun, Qi Zhang, Rongrong Ji
* **arXiv ID:** 2505.17540
* **One-liner:** Introduced RePrompt, a reinforcement learning-based reprompting framework for better T2I alignment and composition.
* **Published in:** arxiv (23 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.17540) | [[PDF]](https://arxiv.org/pdf/2505.17540) | [[Code]](https://github.com/microsoft/DKI_LLM/tree/main/RePrompt)

> **核心创新**
> 训练语言模型生成结构化、自反思提示，针对图像结果进行优化。

<details>
    <summary>Abstract</summary>
    尽管文生图生成近期取得进展，现有模型常难以从简短和未指定提示中忠实捕捉用户意图。先前工作尝试使用大语言模型增强提示，但这些方法因视觉语义和真实世界组合基础不足而常生成风格化或不现实内容。受语言模型推理进展启发，我们提出RePrompt，一种新颖的重提示框架，通过强化学习在提示增强过程中引入显式推理。我们的方法训练语言模型生成结构化、自反思提示，通过优化图像级结果。定制奖励模型评估生成图像的人类偏好、语义对齐和视觉组合，提供间接监督以优化提示生成。我们的方法无需人工标注数据即可实现端到端训练。在GenEval和T2I-Compbench上的实验显示，RePrompt显著提升了空间布局保真度和组合泛化能力，在各种T2I骨干网络上建立了新的最先进结果。
</details>

<details>
    <summary>Key points</summary>
    * 使用强化学习训练提示生成，无需人类数据。
    * 开发奖励模型以评估人类偏好、语义对齐和视觉组合。
    * 增强T2I模型的空间布局保真度和组合泛化能力。
</details>
</details>

---


<details>
<summary><b> Align Beyond Prompts: Evaluating World Knowledge Alignment in Text-to-Image Generation</b></summary>

* **Authors:** Wenchao Zhang, Jiahe Tian, Runze He, Jizhong Han, Jiao Dai, Miaomiao Feng, Wei Mi, Xiaodan Zhang
* **arXiv ID:** 2505.18730
* **One-liner:** Established the ABP benchmark to evaluate T2I model alignment with real-world knowledge beyond prompts.
* **Published in:** arxiv (24 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.18730) | [[PDF]](https://arxiv.org/pdf/2505.18730) | [[Code]](https://github.com/smile365317/ABP)

> **核心创新**
> 提出ABPScore指标和ITKI策略，以改进T2I生成中的知识整合。

<details>
    <summary>Abstract</summary>
    近期文生图生成模型进展显著，能从文本提示创建高保真图像。但现有评估基准主要关注生成图像与提示的显式对齐，忽略了与提示外真实世界知识的对齐。为填补此空白，我们引入Align Beyond Prompts，一个全面基准，旨在衡量生成图像与超出显式用户提示的真实世界知识的对齐度。ABP包含超过2,000个精心设计的提示，覆盖六个不同场景的真实世界知识。我们进一步引入ABPScore，一种利用现有多模态大语言模型评估生成图像与提示外世界知识对齐度的指标，其与人类判断有强相关性。通过对8个流行T2I模型使用ABP进行全面评估，我们发现即使最先进模型如GPT-4o在将简单真实世界知识整合到生成图像中也存在局限。为缓解此问题，我们在ABP中引入一种无需训练的策略，名为推理时知识注入。通过将此策略应用于优化200个挑战性样本，我们在ABPScore上实现了约43%的提升。数据集和代码可在指定链接获取。
</details>

<details>
    <summary>Key points</summary>
    * 创建ABP基准，包含超过2,000个提示，覆盖六个场景。
    * 引入ABPScore，使用MLLM进行对齐评估。
    * 开发推理时知识注入以实现无需训练的改进。
</details>
</details>

---


<details>
<summary><b> Training-free Stylized Text-to-Image Generation with Fast Inference</b></summary>

* **Authors:** Xin Ma, Yaohui Wang, Xinyuan Chen, Tien-Tsin Wong, Cunjian Chen
* **arXiv ID:** 2505.19063
* **One-liner:** Proposed OmniPainter for stylized image generation without fine-tuning, using pre-trained diffusion models.
* **Published in:** arxiv (25 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.19063) | [[PDF]](https://arxiv.org/pdf/2505.19063) | [[Code]](https://github.com/maxin-cn/OmniPainter)

> **核心创新**
> 利用潜在一致性模型的自一致性和自注意力范数混合进行风格迁移。

<details>
    <summary>Abstract</summary>
    尽管扩散模型展现出令人印象深刻的生成能力，现有基于这些模型的风格化图像生成方法常需文本反转或使用风格图像进行微调，这耗时且限制大规模扩散模型的实际应用。为解决这些挑战，我们提出一种新颖的风格化图像生成方法，利用预训练大规模扩散模型而无需微调或额外优化，称为OmniPainter。具体而言，我们利用潜在一致性模型的自一致性属性从参考风格图像中提取代表性风格统计量以指导风格化过程。此外，我们引入自注意力范数混合，使模型能从这些统计量中查询最相关的风格模式以用于中间输出内容特征。此机制还确保风格化结果与参考风格图像的分布紧密对齐。我们的定性和定量实验结果表明，所提方法优于最先进方法。
</details>

<details>
    <summary>Key points</summary>
    * 使用自一致性属性从参考图像中提取风格统计量。
    * 引入自注意力范数混合以查询相关风格模式。
    * 确保风格化结果与参考风格分布对齐。
</details>
</details>

---


<details>
<summary><b> Alchemist: Turning Public Text-to-Image Data into Generative Gold</b></summary>

* **Authors:** Valerii Startsev, Alexander Ustyuzhanin, Alexey Kirillov, Dmitry Baranchuk, Sergey Kastryulin
* **arXiv ID:** 2505.19297
* **One-liner:** Introduced a methodology for creating high-impact SFT datasets using pre-trained models, releasing Alchemist dataset.
* **Published in:** arxiv (25 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.19297) | [[PDF]](https://arxiv.org/pdf/2505.19297) | [[Code]](https://huggingface.co/datasets/yandex/alchemist)

> **核心创新**
> 利用生成模型估计有效训练样本，用于T2I模型微调。

<details>
    <summary>Abstract</summary>
    预训练赋予文生图模型广泛世界知识，但这通常不足以实现高美学质量和对齐度。因此，监督微调对于进一步精炼至关重要。但其效果高度依赖于微调数据集的质量。现有公共SFT数据集常针对狭窄领域（如动漫或特定艺术风格），而高质量通用SFT数据集的创建仍是一个重大挑战。当前策展方法常成本高昂且难以识别真正有影响的样本。此挑战因公共通用数据集的稀缺而进一步复杂化，因为领先模型常依赖大型、专有且文档不全的内部数据，阻碍更广泛研究进展。本文引入一种新颖方法，利用预训练生成模型作为高影响训练样本的估计器来创建通用SFT数据集。我们应用此方法构建并发布Alchemist，一个紧凑（3,350样本）但高效的SFT数据集。实验表明，Alchemist显著提升了五个公共T2I模型的生成质量，同时保持多样性和风格。此外，我们向公众发布微调模型权重。
</details>

<details>
    <summary>Key points</summary>
    * 使用预训练生成模型识别高影响SFT样本。
    * 构建Alchemist数据集，包含3,350个样本。
    * 证明在公共T2I模型中提升生成质量和多样性。
</details>
</details>

---


<details>
<summary><b> StyleAR: Customizing Multimodal Autoregressive Model for Style-Aligned Text-to-Image Generation</b></summary>

* **Authors:** Yi Wu, Lingting Zhu, Shengju Qian, Lei Liu, Wandi Qiao, Lequan Yu, Bin Li
* **arXiv ID:** 2505.19874
* **One-liner:** Developed StyleAR for style-aligned T2I generation using binary data and enhanced AR models.
* **Published in:** arxiv (26 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.19874) | [[PDF]](https://arxiv.org/pdf/2505.19874) | [[Code]]()

> **核心创新**
> 结合数据策展与AR模型，利用文本-图像二元数据实现风格一致性。

<details>
    <summary>Abstract</summary>
    在当前研究环境中，多模态自回归模型在视觉理解和生成等多个领域展现出卓越能力。但复杂任务如风格对齐的文生图生成带来重大挑战，尤其在数据获取方面。类比于AR模型的图像编辑指令跟随调优，风格对齐生成需要参考风格图像和提示，形成文本-图像-图像三元组，其中输出共享输入的风格和语义。然而，获取大量具有特定风格的此类三元组数据比获取用于训练生成模型的常规文生图数据更具挑战性。为解决此问题，我们提出StyleAR，一种创新方法，将特别设计的数据策展方法与所提AR模型结合，以有效利用文生图二元数据进行风格对齐的文生图生成。我们的方法使用参考风格图像和提示合成目标风格化数据，但仅将目标风格化图像作为图像模态以创建高质量二元数据。为促进二元数据训练，我们引入带有感知重采样器的CLIP图像编码器，将图像输入转换为与AR模型中多模态令牌对齐的风格令牌，并实现风格增强令牌技术以防止内容泄漏（这是先前工作中的常见问题）。此外，我们将从大规模文本-图像数据集中提取的原始图像与风格化图像混合，以增强StyleAR提取更丰富风格特征的能力并确保风格一致性。广泛的定性和定量实验证明了我们的优越性能。
</details>

<details>
    <summary>Key points</summary>
    * 设计数据策展方法以合成风格化二元数据。
    * 引入带感知重采样器的CLIP编码器以实现风格令牌对齐。
    * 使用风格增强令牌防止内容泄漏，并混合原始图像以丰富特征。
</details>
</details>

---


<details>
<summary><b> Identity-Preserving Text-to-Image Generation via Dual-Level Feature Decoupling and Expert-Guided Fusion</b></summary>

* **Authors:** Kewen Chen, Xiaobin Hu, Wenqi Ren
* **arXiv ID:** 2505.22360
* **One-liner:** Proposed a framework for subject-driven T2I generation with improved identity feature disentanglement and fusion.
* **Published in:** arxiv (28 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.22360) | [[PDF]](https://arxiv.org/pdf/2505.22360) | [[Code]](https://github.com/wuyi2020/StyleAR)

> **核心创新**
> 引入IEDM进行隐式-显式解耦和FFM与MoE进行特征整合。

<details>
    <summary>Abstract</summary>
    大规模文生图生成模型的近期进展导致主题驱动文生图生成的激增，其旨在生成与文本描述对齐且保留特定主题身份的定制图像。尽管取得显著进展，当前方法难以从输入图像中解耦身份相关和身份无关细节，导致过拟合或无法维持主题身份。本工作中，我们提出一种新颖框架，改进身份相关和身份无关特征的分离，并引入创新特征融合机制以提升生成图像的质量和文本对齐度。我们的框架包含两个关键组件：隐式-显式前景-背景解耦模块和基于专家混合的特征融合模块。IEDM结合可学习适配器在特征级进行隐式解耦与修复技术在图像级进行显式前景-背景分离。FFM动态整合身份无关特征与身份相关特征，即使在解耦不完全时也能实现精炼特征表示。此外，我们引入三个互补损失函数以指导解耦过程。广泛实验证明了我们方法在提升图像生成质量、改进场景适应灵活性和增加各种文本描述下生成输出多样性方面的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 开发隐式-显式解耦模块以分离前景和背景。
    * 创建特征融合模块，使用专家混合。
    * 应用互补损失函数以指导解耦并提升生成质量。
</details>
</details>

---


<details>
<summary><b> HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer</b></summary>

* **Authors:** Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, Yimeng Wang, Kai Yu, Wenxuan Chen, Ziwei Feng, Zijian Gong, Jianzhuang Pan, Yi Peng, Rui Tian, Siyu Wang, Bo Zhao, Ting Yao, Tao Mei
* **arXiv ID:** 2505.22705
* **One-liner:** Introduced HiDream-I1, a sparse DiT-based model for fast, high-quality image generation and editing.
* **Published in:** arxiv (28 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.22705) | [[PDF]](https://arxiv.org/pdf/2505.22705) | [[Code]](https://github.com/HiDream-ai/HiDream-I1)

> **核心创新**
> 构建双流和单流稀疏DiT与MoE，用于高效多模态交互。

<details>
    <summary>Abstract</summary>
    图像生成基础模型的近期进展优先考虑质量改进，但常以增加计算复杂度和推理延迟为代价。为解决这一关键权衡，我们引入HiDream-I1，一个新的开源图像生成基础模型，具有170亿参数，能在数秒内实现最先进的图像生成质量。HiDream-I1采用新的稀疏扩散Transformer结构构建。具体而言，它始于稀疏DiT的双流解耦设计，具有动态专家混合架构，其中两个独立编码器首先参与以分别处理图像和文本令牌。然后，采用具有动态MoE架构的单流稀疏DiT结构，以成本高效方式触发多模态交互以进行图像生成。为支持不同模型能力的灵活可访问性，我们提供HiDream-I1的三个变体：HiDream-I1-Full、HiDream-I1-Dev和HiDream-I1-Fast。此外，我们超越典型文生图生成，通过额外图像条件重塑HiDream-I1以执行基于指令的图像编辑，产生一个新的基于指令的图像编辑模型即HiDream-E1。最终，通过整合文生图生成和基于指令的图像编辑，HiDream-I1演变为一个全面的图像代理，能够完全交互式图像创建和优化。为加速多模态AIGC研究，我们已开源HiDream-I1-Full、HiDream-I1-Dev、HiDream-I1-Fast、HiDream-E1的所有代码和模型权重，通过项目网站。所有功能可通过指定链接直接体验。
</details>

<details>
    <summary>Key points</summary>
    * 设计稀疏扩散Transformer，具有动态MoE架构。
    * 提供变体以支持灵活模型能力。
    * 扩展到基于指令的图像编辑和交互式代理。
</details>
</details>

---


<details>
<summary><b> Rhetorical Text-to-Image Generation via Two-layer Diffusion Policy Optimization</b></summary>

* **Authors:** Yuxi Zhang, Yueting Li, Xinyu Du, Sibo Wang
* **arXiv ID:** 2505.22792
* **One-liner:** Proposed Rhet2Pix, a framework that improves rhetorical text-to-image generation by formulating it as a multi-step policy optimization problem.
* **Published in:** arxiv (28 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.22792) | [[PDF]](https://arxiv.org/pdf/2505.22792) | [[Code]]()

> **核心创新**
> Rhet2Pix通过使用双层MDP扩散模块来细化子句和优化动作，解决了从修辞语言生成图像的挑战，并优于SOTA模型。

<details>
    <summary>Abstract</summary>
    从修辞语言生成图像仍然是文本到图像模型的关键挑战。即使是最先进的多模态大语言模型也未能基于修辞语言中隐含的深层含义生成图像——尽管这些内容对人类来说易于映射到视觉表示。一个关键限制是当前模型强调对象级词嵌入对齐，导致隐喻表达引导图像生成趋向其字面视觉，而忽略预期语义含义。为此，我们提出Rhet2Pix框架，将修辞文本到图像生成制定为多步策略优化问题，并集成双层MDP扩散模块。在外层，Rhet2Pix将输入提示转换为逐步细化的子句并执行相应图像生成动作，构建语义更丰富的视觉。在内层，Rhet2Pix通过折扣最终奖励并优化扩散去噪轨迹上的每个相邻动作对，缓解图像生成中的奖励稀疏性。广泛实验证明Rhet2Pix在修辞文本到图像生成中的有效性。我们的模型在定性和定量评估中均优于SOTA MLLMs如GPT-4o、Grok-3及领先学术基线。本工作中使用的代码和数据集已公开可用。
</details>

<details>
    <summary>Key points</summary>
    * 将修辞文本到图像生成制定为多步策略优化问题
    * 集成双层MDP扩散模块，包括外层和内层
    * 将输入提示转换为逐步细化的子句
    * 通过折扣最终奖励和优化相邻动作对缓解奖励稀疏性
</details>
</details>

---


<details>
<summary><b> Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model</b></summary>

* **Authors:** Qingyu Shi, Jinbin Bai, Zhuoran Zhao, Wenhao Chai, Kaidong Yu, Jianzong Wu, Shuangyong Song, Yunhai Tong, Xiangtai Li, Xuelong Li, Shuicheng Yan
* **arXiv ID:** 2505.23606
* **One-liner:** Introduced Muddit, a unified discrete diffusion transformer for fast and parallel generation across text and image modalities.
* **Published in:** arxiv (29 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.23606) | [[PDF]](https://arxiv.org/pdf/2505.23606) | [[Code]](https://github.com/M-E-AGI-Lab/Muddit)

> **核心创新**
> Muddit通过集成预训练文本到图像骨干的强视觉先验与轻量文本解码器，在质量和效率上实现竞争性性能。

<details>
    <summary>Abstract</summary>
    统一生成模型旨在通过单一架构和解码范式处理跨模态的多样化任务——如文本生成、图像生成和视觉语言推理。自回归统一模型因顺序解码而推理缓慢，非自回归统一模型因预训练骨干有限而泛化能力弱。我们引入Muddit，一种统一离散扩散变换器，能够在文本和图像模态上实现快速并行生成。与先前从头训练的统一直散模型不同，Muddit将预训练文本到图像骨干的强视觉先验与轻量文本解码器集成，在统一架构下实现灵活高质量的多模态生成。实证结果显示，Muddit在质量和效率上均达到或优于显著更大的自回归模型。该工作突显了纯离散扩散在配备强视觉先验时作为统一生成可扩展有效骨干的潜力。
</details>

<details>
    <summary>Key points</summary>
    * 使用统一离散扩散变换器进行多模态生成
    * 集成预训练骨干的强视觉先验
    * 实现跨文本和图像模态的快速并行生成
    * 突显离散扩散与视觉先验的潜力
</details>
</details>

---


<details>
<summary><b> OSPO: Object-centric Self-improving Preference Optimization for Text-to-Image Generation</b></summary>

* **Authors:** Yoonjin Oh, Yongjin Kim, Hyomin Kim, Donghwan Chi, Sungwoong Kim
* **arXiv ID:** 2506.02015
* **One-liner:** Developed OSPO, an object-centric self-improving framework to enhance fine-grained text-image alignment in text-to-image generation.
* **Published in:** arxiv (28 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.02015) | [[PDF]](https://arxiv.org/pdf/2506.02015) | [[Code]]()

> **核心创新**
> OSPO通过构建硬负数据和使用对象中心偏好优化解决对象幻觉问题，在细粒度对齐上超越先前方法。

<details>
    <summary>Abstract</summary>
    多模态大语言模型的最新进展使模型能够以统一方式执行多模态数据的理解和生成。然而，在输入提示与生成图像之间实现细粒度对齐仍然是一个主要挑战，尤其是在文本到图像生成中。因此，近期工作引入了基于自生成数据和自反馈的自改进机制，以在不依赖外部大规模数据或模型的情况下高效缓解这一挑战。然而，现有自改进方法在生成训练数据或提供反馈时未关注细粒度视觉细节，尤其是在对象级别，因此它们仍难以解决文本到图像生成中的对象幻觉问题。为解决此问题，我们提出对象中心自改进偏好优化，一种用于增强对象级文本图像对齐的自改进框架。OSPO旨在明确满足构建和利用对象级硬负数据以及对象中心优化的需求，以改进对象特定保真度。具体而言，OSPO包括：(1)初始提示生成(2)硬偏好对生成(3)过滤和选择(4)带条件偏好损失的对象中心偏好优化。在组合图像生成基准上的广泛实验表明，OSPO显著改进文本到图像生成中的细粒度对齐，不仅超越先前自改进方法，还优于基于扩散的专用图像生成模型。
</details>

<details>
    <summary>Key points</summary>
    * 引入对象中心自改进偏好优化
    * 生成硬偏好对并进行过滤
    * 使用带条件偏好损失的对象中心偏好优化
    * 专注于改进对象级文本图像对齐
</details>
</details>

---


<details>
<summary><b> DIMCIM: A Quantitative Evaluation Framework for Default-mode Diversity and Generalization in Text-to-Image Generative Models</b></summary>

* **Authors:** Revant Teotia, Candace Ross, Karen Ullrich, Sumit Chopra, Adriana Romero-Soriano, Melissa Hall, Matthew J. Muckley
* **arXiv ID:** 2506.05108
* **One-liner:** Introduced DIM-CIM, a reference-free framework for measuring diversity and generalization in text-to-image models.
* **Published in:** arxiv (5 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.05108) | [[PDF]](https://arxiv.org/pdf/2506.05108) | [[Code]](https://github.com/facebookresearch/DIMCIM)

> **核心创新**
> DIM-CIM使用COCO-DIMCIM基准评估默认模式多样性和泛化能力，揭示模型扩展中的权衡和失败案例。

<details>
    <summary>Abstract</summary>
    文本到图像模型的最新进展已实现令人印象深刻的生成质量和一致性。然而，这以表示多样性为代价。虽然存在用于基准测试模型多样性的自动评估方法，但它们要么需要参考图像数据集，要么缺乏对测量多样性类型的特异性，限制了其适应性和可解释性。为填补这一空白，我们引入Does-it/Can-it框架DIM-CIM，一种无参考的默认模式多样性测量和泛化能力评估。我们构建COCO-DIMCIM基准，该基准基于COCO概念和标题，并通过大语言模型增强。使用COCO-DIMCIM，我们发现广泛使用的模型在从1.5B扩展到8.1B参数时，以默认模式多样性为代价提高泛化能力。DIMCIM还识别细粒度失败案例，例如在通用提示下生成的属性在明确请求时很少生成。最后，我们使用DIMCIM评估T2I模型的训练数据，并观察到训练图像多样性与默认模式多样性之间的0.85相关性。我们的工作提供了一个灵活可解释的框架，用于评估T2I模型多样性和泛化能力，实现更全面的模型性能理解。
</details>

<details>
    <summary>Key points</summary>
    * 提出Does-it/Can-it框架用于多样性测量
    * 构建基于COCO概念和LLM增强的COCO-DIMCIM基准
    * 测量默认模式多样性和泛化能力
    * 识别细粒度失败案例和与训练数据的相关性
</details>
</details>

---


<details>
<summary><b> FocusDiff: Advancing Fine-Grained Text-Image Alignment for Autoregressive Visual Generation through RL</b></summary>

* **Authors:** Kaihang Pan, Wendong Bu, Yuruo Wu, Yang Wu, Kai Shen, Yunfei Li, Hang Zhao, Juncheng Li, Siliang Tang, Yueting Zhuang
* **arXiv ID:** 2506.05501
* **One-liner:** Proposed FocusDiff, a method to enhance fine-grained text-image alignment by focusing on subtle semantic differences.
* **Published in:** arxiv (5 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.05501) | [[PDF]](https://arxiv.org/pdf/2506.05501) | [[Code]](https://github.com/wendell0218/FocusDiff)

> **核心创新**
> FocusDiff使用新颖强化学习算法和配对文本数据集，在PairComp等基准上实现SOTA性能。

<details>
    <summary>Abstract</summary>
    近期研究将自回归范式扩展到文本到图像生成，实现与扩散模型相当的性能。然而，我们的新PairComp基准——包含具有相似语法但不同细粒度语义的配对提示测试案例——揭示现有模型在细粒度文本图像对齐上存在困难，因此无法实现对视觉令牌的精确控制。为此，我们提出FocusDiff，通过聚焦相似文本图像对之间的细微差异来增强细粒度文本图像语义对齐。我们构建了一个新数据集，包含具有相似整体表达但不同局部语义的配对文本和图像，并进一步引入一种新颖强化学习算法，以强调此类细粒度语义差异用于期望图像生成。我们的方法在现有文本到图像基准上实现最先进性能，并在PairComp上显著优于先前方法。
</details>

<details>
    <summary>Key points</summary>
    * 增强细粒度文本图像语义对齐
    * 构建具有不同局部语义的配对文本数据集
    * 引入强化学习算法以强调语义差异
    * 在文本到图像基准上实现最先进性能
</details>
</details>

---


<details>
<summary><b> A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation</b></summary>

* **Authors:** Andrew Z. Wang, Songwei Ge, Tero Karras, Ming-Yu Liu, Yogesh Balaji
* **arXiv ID:** 2506.08210
* **One-liner:** Investigated using modern LLMs as text encoders for text-to-image diffusion models, improving alignment with complex prompts.
* **Published in:** arxiv (9 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.08210) | [[PDF]](https://arxiv.org/pdf/2506.08210) | [[Code]]()

> **核心创新**
> 通过探索各层嵌入并使用层归一化平均，研究表明LLMs在文本到图像生成中优于T5。

<details>
    <summary>Abstract</summary>
    文本到图像生成和大语言模型均取得显著进展。然而，许多文本到图像模型仍使用相对过时的T5和CLIP作为其文本编码器。在本工作中，我们研究使用现代仅解码器LLMs作为文本到图像扩散模型文本编码器的有效性。我们构建了一个标准化训练和评估管道，使我们能够隔离和评估不同文本嵌入的效果。我们训练了总计27个文本到图像模型，使用12种不同文本编码器，以分析可能影响文本到图像生成的LLMs关键方面，包括提取嵌入的方法、不同LLMs变体和模型大小。我们的实验揭示，使用最后一层嵌入作为调节的默认方式导致性能较差。相反，我们探索来自各层的嵌入，并发现使用跨所有层的层归一化平均显著提高与复杂提示的对齐。大多数LLMs在这种调节下优于基线T5模型，显示出在高级视觉语言推理技能上的增强性能。
</details>

<details>
    <summary>Key points</summary>
    * 评估现代仅解码器LLMs作为扩散模型文本编码器
    * 训练27个模型和12种文本编码器以分析嵌入效果
    * 发现跨所有层的层归一化平均提高对齐
    * 显示LLMs在视觉语言推理上增强性能
</details>
</details>

---


<details>
<summary><b> A High-Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation</b></summary>

* **Authors:** Yukang Feng, Jianwen Sun, Chuanhao Li, Zizhen Li, Jiaxin Ai, Fanrui Zhang, Yifan Chang, Sizhuo Zhou, Shenglin Zhang, Yu Dai, Kaipeng Zhang
* **arXiv ID:** 2506.09427
* **One-liner:** Introduced InterSyn, a large-scale multimodal dataset, and SynJudge, an automatic evaluation model for interleaved image-text outputs.
* **Published in:** arxiv (11 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.09427) | [[PDF]](https://arxiv.org/pdf/2506.09427) | [[Code]]()

> **核心创新**
> InterSyn使用SEIR方法构建，提供丰富对话用于训练LMMs，而SynJudge评估多模态输出的多个维度。

<details>
    <summary>Abstract</summary>
    大型多模态模型的最新进展显著改进了多模态理解和生成。然而，这些模型仍难以生成紧密交织的图像文本输出，主要由于当前训练数据集的规模、质量和指令丰富性有限。为解决此问题，我们引入InterSyn，一个使用自评估与迭代精化方法构建的大规模多模态数据集。InterSyn特征多轮指令驱动对话与紧密交织的图像文本响应，提供丰富的对象多样性和严格的自动化质量精化，使其非常适合训练下一代指令跟随LMMs。此外，为解决缺乏可靠评估工具来评估交织多模态输出的问题，我们引入SynJudge，一种自动评估模型，设计用于沿四个维度定量评估多模态输出：文本内容、图像内容、图像质量和图像文本协同性。实验研究表明，SEIR方法相比无精化的相同过程导致数据集质量显著提高。此外，在InterSyn上训练的LMMs在所有评估指标上均实现一致性能提升，确认InterSyn在推进多模态系统方面的实用性。
</details>

<details>
    <summary>Key points</summary>
    * 使用自评估与迭代精化方法构建InterSyn数据集
    * 特征多轮指令驱动对话与交织图像文本响应
    * 引入SynJudge用于多模态输出的自动评估
    * 提高LMMs在所有评估指标上的性能
</details>
</details>

---


<details>
<summary><b> ELBO-T2IAlign: A Generic ELBO-Based Method for Calibrating Pixel-level Text-Image Alignment in Diffusion Models</b></summary>

* **Authors:** Qin Zhou, Zhiyang Zhang, Jinglong Wang, Xiaobin Li, Jing Zhang, Qian Yu, Lu Sheng, Dong Xu
* **arXiv ID:** 2506.09740
* **One-liner:** Proposed ELBO-T2IAlign, a training-free method to calibrate pixel-text alignment in diffusion models using zero-shot referring image segmentation.
* **Published in:** arxiv (11 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.09740) | [[PDF]](https://arxiv.org/pdf/2506.09740) | [[Code]]()

> **核心创新**
> ELBO-T2IAlign通过利用证据下界解决错位问题，在各种扩散架构上工作而无需识别具体原因。

<details>
    <summary>Abstract</summary>
    扩散模型在图像生成方面表现出色。近期研究表明，这些模型不仅生成高质量图像，还通过注意力图或损失函数编码文本图像对齐信息。这些信息对于各种下游任务具有价值，包括分割、文本引导图像编辑和组合图像生成。然而，当前方法严重依赖扩散模型中完美文本图像对齐的假设，但实际情况并非如此。在本文中，我们提出使用零样本参考图像分割作为代理任务，以评估流行扩散模型的像素级图像和类级文本对齐。我们从训练数据偏差的角度对扩散模型中的像素文本错位进行深入分析。我们发现错位发生在具有小尺寸、遮挡或稀有对象类的图像中。因此，我们提出ELBO-T2IAlign，一种基于证据下界的简单有效方法，用于校准扩散模型中的像素文本对齐。我们的方法无需训练且通用，无需识别错位的具体原因，并在各种扩散模型架构上工作良好。在常用基准数据集上的广泛实验验证了我们提出的校准方法的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 使用零样本参考图像分割作为对齐评估代理任务
    * 从训练数据偏差角度分析像素文本错位
    * 提出基于证据下界的ELBO-T2IAlign进行校准
    * 无需训练且通用的方法适用于各种扩散模型
</details>
</details>

---


<details>
<summary><b> Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention</b></summary>

* **Authors:** Jeonghoon Park, Juyoung Lee, Chaeyeon Chung, Jaeseong Lee, Jaegul Choo, Jindong Gu
* **arXiv ID:** 2506.13298
* **One-liner:** Introduced Entanglement-Free Attention (EFA) to mitigate societal biases in text-to-image models while preserving non-target attributes.
* **Published in:** arxiv (16 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.13298) | [[PDF]](https://arxiv.org/pdf/2506.13298) | [[Code]]()

> **核心创新**
> EFA调整交叉注意力层以等概率纳入目标属性，在偏见缓解中优于现有方法且无分布偏移。

<details>
    <summary>Abstract</summary>
    基于扩散的文本到图像模型的最新进展使得从文本生成高质量和逼真图像成为可能。然而，它们常表现出与社会相关的偏见，如性别、种族和社会经济地位，从而可能强化有害刻板印象并以意外方式塑造公众认知。虽然现有偏见缓解方法显示有效性，但它们常遇到属性纠缠，其中对与偏见相关的属性的调整无意中改变与偏见无关的属性，导致不希望的分布偏移。为应对这一挑战，我们引入无纠缠注意力，一种在偏见缓解期间准确纳入目标属性同时保留非目标属性的方法。在推理时，EFA以等概率随机采样目标属性，并调整选定层中的交叉注意力以纳入采样属性，实现目标属性的公平分布。广泛实验证明，EFA在缓解偏见同时保留非目标属性方面优于现有方法，从而维持原始模型的输出分布和生成能力。
</details>

<details>
    <summary>Key points</summary>
    * 提出无纠缠注意力用于偏见缓解
    * 随机采样目标属性并调整选定层中的交叉注意力
    * 在偏见调整期间保留非目标属性
    * 实现目标属性的公平分布并维持生成能力
</details>
</details>

---


<details>
<summary><b> Discrete JEPA: Learning Discrete Token Representations without Reconstruction</b></summary>

* **Authors:** Junyeob Baek, Hosung Lee, Christopher Hoang, Mengye Ren, Sungjin Ahn
* **arXiv ID:** 2506.14373
* **One-liner:** Proposed Discrete-JEPA, a method for robust tokenization to enhance symbolic reasoning in AI systems.
* **Published in:** arxiv (17 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.14373) | [[PDF]](https://arxiv.org/pdf/2506.14373) | [[Code]]()

> **核心创新**
> Discrete-JEPA扩展潜在预测编码与语义令牌化和互补目标，在视觉符号预测任务上优于基线。

<details>
    <summary>Abstract</summary>
    认知智能的基石在于从观察中提取隐藏模式，并利用这些原则系统预测未来结果。然而，当前图像令牌化方法在需要符号抽象和逻辑推理能力的任务中表现出显著局限性，这些能力对于系统推理至关重要。为应对这一挑战，我们提出Discrete-JEPA，扩展潜在预测编码框架与语义令牌化和新颖互补目标，以创建用于符号推理任务的鲁棒令牌化。Discrete-JEPA在视觉符号预测任务上显著优于基线，同时显著的视觉证据揭示了学习语义令牌空间内自发出现的刻意系统模式。尽管是初始模型，我们的方法有望对推进人工智能系统中的符号世界建模和规划能力产生重大影响。
</details>

<details>
    <summary>Key points</summary>
    * 扩展潜在预测编码框架与语义令牌化
    * 引入新颖互补目标用于鲁棒令牌化
    * 在视觉符号预测任务上优于基线
    * 显示语义令牌空间中系统模式的出现
</details>
</details>

---


<details>
<summary><b> Cost-Aware Routing for Efficient Text-To-Image Generation</b></summary>

* **Authors:** Qinchan Li, Kenneth Chen, Changyue Su, Wittawat Jitkrittum, Qi Sun, Patsorn Sangkloy
* **arXiv ID:** 2506.14753
* **One-liner:** Developed a framework to optimize the trade-off between image quality and computational cost in text-to-image generation by routing prompts to appropriate models.
* **Published in:** arxiv (17 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.14753) | [[PDF]](https://arxiv.org/pdf/2506.14753) | [[Code]]()

> **核心创新**
> 提出了一个框架，根据复杂性自动将提示路由到不同的文本到图像模型或去噪步骤，实现比任何单一模型更高的平均质量。

<details>
    <summary>Abstract</summary>
    扩散模型以其通过迭代去噪过程为输入提示生成高保真图像的能力而闻名。然而，由于固有的顺序生成过程，高保真性也带来了高计算成本。在本工作中，我们寻求在质量和计算成本之间实现最优平衡，并提出了一个框架，允许每个提示的计算量根据其复杂性而变化。每个提示被自动路由到最合适的文本到图像生成函数，这可能对应于扩散模型的不同去噪步骤数量，或不同的独立文本到图像模型。与统一成本减少技术（例如，蒸馏、模型量化）不同，我们的方法通过学习仅为少数复杂提示保留昂贵选择（例如，100+去噪步骤），并为不太复杂的提示采用更经济的选择（例如，小型蒸馏模型），实现了最优权衡。我们在COCO和DiffusionDB上实证证明，通过学习路由到九个已训练的文本到图像模型，我们的方法能够提供比任何单一模型单独实现更高的平均质量。
</details>

<details>
    <summary>Key points</summary>
    * 自动将提示路由到最合适的生成函数
    * 学习为复杂提示使用昂贵模型，为简单提示使用经济模型
    * 在COCO和DiffusionDB数据集上的实证验证
</details>
</details>

---


<details>
<summary><b> NSFW-Classifier Guided Prompt Sanitization for Safe Text-to-Image Generation</b></summary>

* **Authors:** Yu Xie, Chengjie Zeng, Lingyun Zhang, Yanwei Fu
* **arXiv ID:** 2506.18325
* **One-liner:** Introduced PromptSan to detoxify harmful prompts in text-to-image models without altering model architecture.
* **Published in:** arxiv (23 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.18325) | [[PDF]](https://arxiv.org/pdf/2506.18325) | [[Code]]()

> **核心创新**
> 提出了NSFW分类器引导的提示净化及其两个变体，以减少有害内容生成，同时保持可用性。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）模型（如Stable Diffusion）的快速发展增强了它们从文本提示合成图像的能力。然而，这一进展也带来了显著的滥用风险，包括生成有害内容（例如，色情、暴力、歧视），这与T2I技术的伦理目标相悖，并阻碍其可持续发展。受大型语言模型中'越狱'攻击的启发，该攻击通过微妙的提示修改绕过限制，本文提出了NSFW分类器引导的提示净化（PromptSan），一种在不改变模型架构或降低生成能力的情况下净化有害提示的新方法。PromptSan包括两个变体：PromptSan-Modify，在推理过程中使用文本NSFW分类器迭代识别和替换输入提示中的有害令牌；以及PromptSan-Suffix，训练一个优化的后缀令牌序列以中和有害意图，同时通过文本和图像NSFW分类器检查。广泛的实验表明，PromptSan在减少有害内容生成方面，在多个指标上实现了最先进的性能，有效平衡了安全性和可用性。
</details>

<details>
    <summary>Key points</summary>
    * PromptSan-Modify：使用文本NSFW分类器迭代替换令牌
    * PromptSan-Suffix：训练优化的后缀令牌以中和有害意图
    * 广泛实验显示在安全指标上的最先进性能
</details>
</details>

---


<details>
<summary><b> Ovis-U1 Technical Report</b></summary>

* **Authors:** Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, Yang Li, Qing-Guo Chen
* **arXiv ID:** 2506.23044
* **One-liner:** Introduced Ovis-U1, a unified model integrating multimodal understanding, text-to-image generation, and image editing.
* **Published in:** arxiv (29 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.23044) | [[PDF]](https://arxiv.org/pdf/2506.23044) | [[Code]](https://github.com/AIDC-AI/Ovis-U1)

> **核心创新**
> 开发了一个30亿参数模型，采用统一训练，在多个理解和生成任务基准上获得高分。

<details>
    <summary>Abstract</summary>
    在本报告中，我们介绍了Ovis-U1，一个30亿参数的统一模型，集成了多模态理解、文本到图像生成和图像编辑能力。基于Ovis系列的基础，Ovis-U1结合了基于扩散的视觉解码器和双向令牌精炼器，实现了与领先模型如GPT-4o相当的图像生成任务。与一些先前使用冻结MLLM进行生成任务的模型不同，Ovis-U1采用了一种新的统一训练方法，从语言模型开始。与仅在理解或生成任务上训练相比，统一训练产生了更好的性能，展示了集成这两个任务所带来的增强。Ovis-U1在OpenCompass多模态学术基准上获得了69.6分，超越了最近的最先进模型，如Ristretto-3B和SAIL-VL-1.5-2B。在文本到图像生成中，它在DPG-Bench和GenEval基准上分别取得了83.72和0.89的分数。对于图像编辑，它在ImgEdit-Bench和GEdit-Bench-EN上分别获得了4.00和6.42的分数。作为Ovis统一模型系列的初始版本，Ovis-U1推动了多模态理解、生成和编辑的边界。
</details>

<details>
    <summary>Key points</summary>
    * 从语言模型开始的统一训练方法
    * 基于扩散的视觉解码器与双向令牌精炼器
    * 在OpenCompass、DPG-Bench、GenEval、ImgEdit-Bench和GEdit-Bench-EN上的优越性能
</details>
</details>

---


<details>
<summary><b> RichControl: Structure- and Appearance-Rich Training-Free Spatial Control for Text-to-Image Generation</b></summary>

* **Authors:** Liheng Zhang, Lexi Pang, Hang Ye, Xiaoxuan Ma, Yizhou Wang
* **arXiv ID:** 2507.02792
* **One-liner:** Proposed a training-free framework for feature injection in text-to-image models to improve structure guidance and visual quality.
* **Published in:** arxiv (3 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.02792) | [[PDF]](https://arxiv.org/pdf/2507.02792) | [[Code]]()

> **核心创新**
> 将条件特征采样调度与去噪过程解耦，引入了简单调度和重启精炼以改善对齐。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）扩散模型在从文本提示生成高质量图像方面显示出显著成功。最近的努力扩展了这些模型以纳入条件图像（例如，Canny边缘）进行细粒度空间控制。其中，特征注入方法已成为传统基于微调方法的无需训练替代方案。然而，它们经常遭受结构错位、条件泄漏和视觉伪影，特别是当条件图像与自然RGB分布显著偏离时。通过对现有方法的实证分析，我们识别出一个关键限制：条件特征的采样调度，先前未被探索，未能考虑在扩散步骤中结构保存和域对齐之间不断演变的相互作用。受此观察启发，我们提出了一个灵活的无需训练框架，将条件特征的采样调度与去噪过程解耦，并系统研究特征注入调度的频谱，以实现特征空间中更高质量的结构指导。具体来说，我们发现从单个时间步采样的条件特征就足够了，产生了一个简单而高效的调度，平衡了结构对齐和外观质量。我们进一步通过引入重启精炼调度来增强采样过程，并通过外观丰富的提示策略提高视觉质量。这些设计共同实现了既结构丰富又外观丰富的无需训练生成。广泛的实验表明，我们的方法在多样零样本条件场景中实现了最先进的结果。
</details>

<details>
    <summary>Key points</summary>
    * 解耦条件特征采样与去噪过程
    * 条件特征的单个时间步采样
    * 重启精炼调度和外观丰富的提示策略
</details>
</details>

---


<details>
<summary><b> Subject-Consistent and Pose-Diverse Text-to-Image Generation</b></summary>

* **Authors:** Zhanxin Gao, Beier Zhu, Liang Yao, Jian Yang, Ying Tai
* **arXiv ID:** 2507.08396
* **One-liner:** Developed CoDi for subject-consistent generation with diverse poses and layouts in text-to-image models.
* **Published in:** arxiv (11 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.08396) | [[PDF]](https://arxiv.org/pdf/2507.08396) | [[Code]](https://github.com/NJU-PCALab/CoDi)

> **核心创新**
> 引入了使用身份传输和精炼的两阶段策略，以保持主题一致性，同时实现姿态多样性。

<details>
    <summary>Abstract</summary>
    主题一致生成（SCG）——旨在在不同场景中保持一致的主题身份——对文本到图像（T2I）模型仍然是一个挑战。现有的无需训练SCG方法通常以布局和姿态多样性为代价实现一致性，阻碍了表达性视觉叙事。为了解决这一限制，我们提出了主题一致和姿态多样T2I框架，称为CoDi，该框架能够实现一致主题生成，同时具有多样姿态和布局。受扩散过程渐进性质的启发，其中粗结构早期出现，细节后期精炼，CoDi采用两阶段策略：身份传输（IT）和身份精炼（IR）。IT在早期去噪步骤中操作，使用最优传输以姿态感知方式将身份特征传输到每个目标图像。这促进了主题一致性，同时保留了姿态多样性。IR应用于后期去噪步骤，选择最显著的身份特征以进一步精炼主题细节。广泛的定性和定量结果在主题一致性、姿态多样性和提示保真度方面表明，CoDi在所有指标上实现了更好的视觉感知和更强性能。代码在提供的链接中。
</details>

<details>
    <summary>Key points</summary>
    * 早期去噪步骤中的身份传输（IT）使用最优传输
    * 后期步骤中的身份精炼（IR）用于细节精炼
    * 评估显示在一致性、多样性和保真度方面的改进
</details>
</details>

---


<details>
<summary><b> Visual Semantic Description Generation with MLLMs for Image-Text Matching</b></summary>

* **Authors:** Junyu Chen, Yihua Gao, Mingyong Li
* **arXiv ID:** 2507.08590
* **One-liner:** Proposed a framework for image-text matching using multimodal large language models to bridge the modality gap.
* **Published in:** arxiv (11 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.08590) | [[PDF]](https://arxiv.org/pdf/2507.08590) | [[Code]](https://github.com/Image-Text-Matching/VSD)

> **核心创新**
> 利用MLLMs生成视觉语义描述，用于ITM任务中的实例级和原型级对齐。

<details>
    <summary>Abstract</summary>
    图像文本匹配（ITM）旨在解决对齐视觉和文本模态的基本挑战，这些模态在表示上固有不同：连续、高维图像特征与离散、结构化文本。我们提出了一个新颖框架，通过利用多模态大语言模型（MLLMs）作为视觉语义解析器来弥合模态差距。通过生成丰富的视觉语义描述（VSD），MLLMs提供语义锚点，促进跨模态对齐。我们的方法结合：（1）实例级对齐，通过融合视觉特征与VSD来增强图像表示的语言表达性；以及（2）原型级对齐，通过VSD聚类确保类别级一致性。这些模块可以无缝集成到现有ITM模型中。在Flickr30K和MSCOCO上的广泛实验显示了显著的性能改进。该方法还表现出在跨域任务（包括新闻和遥感ITM）中的显著零样本泛化能力。代码和模型检查点在提供的链接中可用。
</details>

<details>
    <summary>Key points</summary>
    * 通过融合视觉特征与VSD进行实例级对齐
    * 通过VSD聚类进行原型级对齐
    * 在Flickr30K和MSCOCO上的广泛实验，具有零样本泛化
</details>
</details>

---


<details>
<summary><b> RaDL: Relation-aware Disentangled Learning for Multi-Instance Text-to-Image Generation</b></summary>

* **Authors:** Geon Park, Seon Bin Kim, Gunho Jung, Seong-Whan Lee
* **arXiv ID:** 2507.11947
* **One-liner:** Introduced RaDL for multi-instance image generation with improved relationship and attribute handling.
* **Published in:** arxiv (16 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.11947) | [[PDF]](https://arxiv.org/pdf/2507.11947) | [[Code]]()

> **核心创新**
> 提出了关系感知解缠学习，以增强实例特定属性并生成关系感知特征。

<details>
    <summary>Abstract</summary>
    随着文本到图像（T2I）模型的最新进展，在单个图像提示中有效生成多个实例已成为一个关键挑战。现有方法虽然在生成单个实例位置方面成功，但往往难以处理关系差异和多个属性泄漏。为了解决这些限制，本文提出了关系感知解缠学习（RaDL）框架。RaDL通过可学习参数增强实例特定属性，并通过关系注意力生成关系感知图像特征，利用从全局提示中提取的动作动词。通过在COCO-Position、COCO-MIG和DrawBench等基准上的广泛评估，我们证明RaDL优于现有方法，在位置准确性、多个属性考虑和实例间关系方面显示出显著改进。我们的结果展示了RaDL作为生成考虑多实例图像中每个实例的关系和多个属性的图像的解决方案。
</details>

<details>
    <summary>Key points</summary>
    * 用于实例特定属性的可学习参数
    * 使用提示中动作动词的关系注意力
    * 在COCO-Position、COCO-MIG和DrawBench基准上的评估
</details>
</details>

---


<details>
<summary><b> ID-EA: Identity-driven Text Enhancement and Adaptation with Textual Inversion for Personalized Text-to-Image Generation</b></summary>

* **Authors:** Hyun-Jun Jin, Young-Eun Kim, Seong-Whan Lee
* **arXiv ID:** 2507.11990
* **One-liner:** Developed ID-EA to improve identity preservation in personalized portrait generation using text-to-image models.
* **Published in:** arxiv (16 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.11990) | [[PDF]](https://arxiv.org/pdf/2507.11990) | [[Code]]()

> **核心创新**
> 引入了ID-Enhancer和ID-Adapter，以对齐文本嵌入与视觉身份嵌入，改善一致性。

<details>
    <summary>Abstract</summary>
    最近，使用文本到图像扩散模型的个性化肖像生成随着文本反转的兴起显著进步，成为创建高保真个性化图像的有前景方法。尽管有潜力，当前文本反转方法由于文本和视觉嵌入空间在身份方面的语义错位，难以保持一致的面部身份。我们引入了ID-EA，一个新颖框架，指导文本嵌入与视觉身份嵌入对齐，从而改善个性化生成中的身份保存。ID-EA包括两个关键组件：ID驱动增强器（ID-Enhancer）和ID条件适配器（ID-Adapter）。首先，ID-Enhancer将身份嵌入与文本ID锚点集成，使用代表性文本嵌入精炼从人脸识别模型导出的视觉身份嵌入。然后，ID-Adapter利用身份增强嵌入来适应文本条件，通过调整预训练UNet模型中的交叉注意力模块确保身份保存。这个过程鼓励文本特征在前景片段中找到最相关的视觉线索。广泛的定量和定性评估表明，ID-EA在身份保存指标上显著优于最先进方法，同时实现了显著的计算效率，生成个性化肖像比现有方法快约15倍。
</details>

<details>
    <summary>Key points</summary>
    * ID-Enhancer使用文本ID锚点精炼视觉身份嵌入
    * ID-Adapter调整交叉注意力以保存身份
    * 定量和定性评估显示优越性能和效率
</details>
</details>

---


<details>
<summary><b> Local Representative Token Guided Merging for Text-to-Image Generation</b></summary>

* **Authors:** Min-Jeong Lee, Hee-Dong Kim, Seong-Whan Lee
* **arXiv ID:** 2507.12771
* **One-liner:** Proposed ReToM, a token merging strategy to reduce computational cost in stable diffusion models while maintaining quality.
* **Published in:** arxiv (17 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.12771) | [[PDF]](https://arxiv.org/pdf/2507.12771) | [[Code]]()

> **核心创新**
> 引入了局部代表性令牌引导合并，以保留显著特征并提高注意力操作中的效率。

<details>
    <summary>Abstract</summary>
    Stable diffusion是文本到图像领域的杰出图像生成模型，但由于注意力操作的二次复杂性，其耗时的生成过程仍然是一个挑战。最近的令牌合并方法通过减少注意力操作中的令牌数量来提高效率，但往往忽略了基于注意力的图像生成模型的特征，限制了其有效性。在本文中，我们提出了局部代表性令牌引导合并（ReToM），一种适用于图像生成中任何注意力机制的新颖令牌合并策略。为了基于各种上下文信息合并令牌，ReToM将局部边界定义为注意力输入中的窗口，并调整窗口大小。此外，我们引入了一个代表性令牌，它通过在特定时间步计算相似度并选择具有最高平均相似度的令牌来代表每个窗口中最具代表性的令牌。这种方法在最小化计算开销的同时保留了最显著的局部特征。实验结果显示，与基线相比，ReToM在FID上实现了6.2%的改进和更高的CLIP分数，同时保持可比的推理时间。我们实证证明，ReToM在平衡视觉质量和计算效率方面是有效的。
</details>

<details>
    <summary>Key points</summary>
    * 将局部边界定义为窗口以进行令牌合并
    * 基于相似度选择代表性令牌
    * 实验结果显示FID和CLIP分数改进，推理时间可比
</details>
</details>

---


<details>
<summary><b> LSSGen: Leveraging Latent Space Scaling in Flow and Diffusion for Efficient Text to Image Generation</b></summary>

* **Authors:** Jyun-Ze Tang, Chih-Fan Hsu, Jeng-Lin Li, Ming-Ching Chang, Wei-Chao Chen
* **arXiv ID:** 2507.16154
* **One-liner:** Introduced LSSGen for efficient and high-quality multi-resolution image generation in latent space.
* **Published in:** arxiv (22 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.16154) | [[PDF]](https://arxiv.org/pdf/2507.16154) | [[Code]]()

> **核心创新**
> 提出了潜在空间缩放与轻量级上采样器，以避免传统像素缩放方法产生的伪影。

<details>
    <summary>Abstract</summary>
    流匹配和扩散模型在文本到图像生成中显示出令人印象深刻的结果，通过迭代去噪过程产生逼真图像。加速合成的常见策略是在较低分辨率下执行早期去噪。然而，在像素空间中进行下采样和上采样的传统方法经常引入伪影和失真。当上采样图像被重新编码到潜在空间时，这些问题会导致最终图像质量下降。为了解决这个问题，我们提出了潜在空间缩放生成（LSSGen），一个使用轻量级潜在上采样器直接在潜在空间执行分辨率缩放的框架。在不改变Transformer或U-Net架构的情况下，LSSGen提高了效率和视觉质量，同时支持灵活的多分辨率生成。我们涵盖文本图像对齐和感知质量的全面评估显示，LSSGen显著优于传统缩放方法。在生成$1024^2$图像时，以相似速度，它实现了高达246%的TOPIQ分数改进。
</details>

<details>
    <summary>Key points</summary>
    * 直接在潜在空间执行分辨率缩放
    * 使用轻量级潜在上采样器而不改变核心架构
    * 评估显示TOPIQ分数和视觉质量的显著改进
</details>
</details>

---


<details>
<summary><b> Lumina-mGPT 2.0: Stand-Alone AutoRegressive Image Modeling</b></summary>

* **Authors:** Yi Xin, Juncheng Yan, Qi Qin, Zhen Li, Dongyang Liu, Shicheng Li, Victor Shea-Jay Huang, Yupeng Zhou, Renrui Zhang, Le Zhuo, Tiancheng Han, Xiaoqing Sun, Siqi Luo, Mengmeng Wang, Bin Fu, Yuewen Cao, Hongsheng Li, Guangtao Zhai, Xiaohong Liu, Yu Qiao, Peng Gao
* **arXiv ID:** 2507.17801
* **One-liner:** Introduced Lumina-mGPT 2.0, a high-quality autoregressive image generation model trained from scratch, matching state-of-the-art diffusion models.
* **Published in:** arxiv (23 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.17801) | [[PDF]](https://arxiv.org/pdf/2507.17801) | [[Code]](https://github.com/Alpha-VLLM/Lumina-mGPT-2.0)

> **核心创新**
> 振兴自回归建模，实现统一多模态生成的单一框架，具备灵活性和组合性。

<details>
    <summary>Abstract</summary>
    我们提出了Lumina-mGPT 2.0，一个独立的、仅解码器的自回归模型，重新审视并振兴了自回归范式，用于高质量图像生成及其他任务。与依赖预训练组件或混合架构的现有方法不同，Lumina-mGPT 2.0完全从头开始训练，实现了无限制的架构设计和许可自由。它在生成质量上达到了与最先进扩散模型（如DALL-E 3和SANA）相当的水平，同时保留了自回归建模固有的灵活性和组合性。我们的统一标记化方案使模型能够在一个生成框架内无缝处理广泛任务，包括主题驱动生成、图像编辑、可控合成和密集预测。为了进一步提升可用性，我们整合了高效解码策略，如推理时间缩放和推测性雅可比采样，分别提高质量和速度。在标准文本到图像基准（如GenEval、DPG）上的广泛评估表明，Lumina-mGPT 2.0不仅匹配而且在某些情况下超越了基于扩散的模型。此外，我们在Graph200K基准上确认了其多任务能力，原生Lumina-mGPT 2.0表现优异。这些结果将Lumina-mGPT 2.0定位为一个强大、灵活的统一多模态生成基础模型。我们已在GitHub上发布了训练细节、代码和模型。
</details>

<details>
    <summary>Key points</summary>
    * 完全从头开始训练，实现无限制的架构设计和许可自由
    * 统一标记化方案，用于处理多个任务，如主题驱动生成和图像编辑
    * 整合高效解码策略，如推理时间缩放和推测性雅可比采样
</details>
</details>

---


<details>
<summary><b> T2I-Copilot: A Training-Free Multi-Agent Text-to-Image System for Enhanced Prompt Interpretation and Interactive Generation</b></summary>

* **Authors:** Chieh-Yun Chen, Min Shi, Gong Zhang, Humphrey Shi
* **arXiv ID:** 2507.20536
* **One-liner:** Developed T2I-Copilot, a training-free multi-agent system that automates prompt engineering and model selection for T2I generation.
* **Published in:** arxiv (28 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.20536) | [[PDF]](https://arxiv.org/pdf/2507.20536) | [[Code]]()

> **核心创新**
> 通过协作代理提高生成质量和文本图像对齐，减少手动提示优化的需求。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）生成模型已经彻底改变了内容创作，但对提示措辞高度敏感，通常需要用户多次重复优化提示而没有明确反馈。尽管自动提示工程、受控文本嵌入、去噪和多轮生成等技术缓解了这些问题，但它们提供的可控性有限，或通常需要额外训练，限制了泛化能力。因此，我们引入了T2I-Copilot，一个无需训练的多代理系统，利用（多模态）大型语言模型之间的协作来自动化提示措辞、模型选择和迭代优化。这种方法显著简化了提示工程，同时与直接生成相比，提高了生成质量和文本图像对齐。具体来说，T2I-Copilot包括三个代理：（1）输入解释器，解析输入提示，解决歧义，并生成标准化报告；（2）生成引擎，从不同类型的T2I模型中选择适当模型，并组织视觉和文本提示以启动生成；（3）质量评估器，评估美学质量和文本图像对齐，提供分数和反馈以进行潜在重新生成。T2I-Copilot可以完全自主运行，同时支持人机交互干预以实现细粒度控制。在GenAI-Bench上，使用开源生成模型，T2I-Copilot实现了与商业模型RecraftV3和Imagen 3相当的VQA分数，以仅16.59%的成本超越FLUX1.1-pro 6.17%，并优于FLUX.1-dev和SD 3.5 Large 9.11%和6.36%。代码将在GitHub上发布。
</details>

<details>
    <summary>Key points</summary>
    * 输入解释器解析和标准化提示
    * 生成引擎选择适当的T2I模型并组织提示
    * 质量评估器评估美学质量和对齐以进行迭代优化
</details>
</details>

---


<details>
<summary><b> Multimodal LLMs as Customized Reward Models for Text-to-Image Generation</b></summary>

* **Authors:** Shijie Zhou, Ruiyi Zhang, Huaisheng Zhu, Branislav Kveton, Yufan Zhou, Jiuxiang Gu, Jian Chen, Changyou Chen
* **arXiv ID:** 2507.21391
* **One-liner:** Proposed LLaVA-Reward, an efficient reward model for automatic evaluation of T2I generations using MLLMs.
* **Published in:** arxiv (28 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.21391) | [[PDF]](https://arxiv.org/pdf/2507.21391) | [[Code]](https://github.com/sjz5202/LLaVA-Reward)

> **核心创新**
> 通过跳跃连接交叉注意力模块改进文本图像相关性推理，实现人类对齐评分。

<details>
    <summary>Abstract</summary>
    我们引入了LLaVA-Reward，一个高效的奖励模型，旨在利用预训练多模态大型语言模型（MLLMs）自动评估文本到图像（T2I）生成的多个方面。现有的基于MLLM的方法需要指令跟随数据进行监督微调，并在分析文本响应时评估生成质量，这耗时且难以训练。为了解决这个问题，我们提出了LLaVA-Reward，它直接利用MLLMs在给定文本图像对时的隐藏状态。为了增强仅解码器MLLMs中视觉和文本表示之间的双向交互，我们进一步提出了添加跳跃连接交叉注意力（SkipCA）模块。这种设计通过将早期层视觉特征与后期层隐藏表示连接起来，增强了文本图像相关性推理。此外，LLaVA-Reward支持不同类型的偏好数据进行高效微调，包括配对偏好数据和非配对数据。我们在四个评估视角上训练LLaVA-Reward：文本图像对齐、保真度/伪影、安全性和总体排名。实证结果表明，LLaVA-Reward在生成人类对齐分数用于自动评估和文本到图像生成中的推理时间缩放方面优于传统和基于MLLM的方法。
</details>

<details>
    <summary>Key points</summary>
    * 利用MLLMs的隐藏状态进行直接评估，无需微调
    * SkipCA模块增强视觉和文本特征之间的双向交互
    * 支持配对和非配对偏好数据的微调，用于多个评估视角
</details>
</details>

---


<details>
<summary><b> Qwen-Image Technical Report</b></summary>

* **Authors:** Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, Zenan Liu
* **arXiv ID:** 2508.02324
* **One-liner:** Achieved state-of-the-art performance in complex text rendering and precise image editing with Qwen-Image.
* **Published in:** arxiv (4 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.02324) | [[PDF]](https://arxiv.org/pdf/2508.02324) | [[Code]](https://github.com/QwenLM/Qwen-Image)

> **核心创新**
> 通过渐进式训练和改进的多任务对齐增强文本渲染能力，确保一致性。

<details>
    <summary>Abstract</summary>
    我们提出了Qwen-Image，Qwen系列中的一个图像生成基础模型，在复杂文本渲染和精确图像编辑方面取得了显著进展。为了解决复杂文本渲染的挑战，我们设计了一个全面的数据管道，包括大规模数据收集、过滤、标注、合成和平衡。此外，我们采用了一种渐进式训练策略，从非文本到文本渲染开始，从简单到复杂的文本输入演进，并逐步扩展到段落级描述。这种课程学习方法显著增强了模型的本地文本渲染能力。因此，Qwen-Image不仅在字母语言（如英语）上表现优异，而且在更具挑战性的象形文字语言（如中文）上取得了显著进展。为了增强图像编辑一致性，我们引入了一个改进的多任务训练范式，不仅包括传统的文本到图像（T2I）和文本图像到图像（TI2I）任务，还包括图像到图像（I2I）重建，有效对齐了Qwen2.5-VL和MMDiT之间的潜在表示。此外，我们分别将原始图像输入Qwen2.5-VL和VAE编码器，以获得语义和重建表示。这种双重编码机制使编辑模块能够在保持语义一致性和视觉保真度之间取得平衡。Qwen-Image在多个基准上实现了最先进的性能，展示了其在图像生成和编辑方面的强大能力。
</details>

<details>
    <summary>Key points</summary>
    * 全面的文本渲染数据管道，包括合成和平衡
    * 渐进式训练策略，从非文本到段落级描述
    * 双重编码机制，用于编辑中的语义和重建表示
</details>
</details>

---


<details>
<summary><b> Documenting Patterns of Exoticism of Marginalized Populations within Text-to-Image Generators</b></summary>

* **Authors:** Sourojit Ghosh, Sanjana Gautam, Pranav Venkit, Avijit Ghosh
* **arXiv ID:** 2508.02937
* **One-liner:** Identified and analyzed exoticism in GAI tools, highlighting biases against non-Western and marginalized communities.
* **Published in:** arxiv (4 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.02937) | [[PDF]](https://arxiv.org/pdf/2508.02937) | [[Code]]()

> **核心创新**
> 记录了全球南方国家服装描绘中的异国情调模式，并提出了危害感知设计的启示。

<details>
    <summary>Abstract</summary>
    大多数AI公平性研究关注GAI工具的有害结果，但忽视了非西方社区和背景，因此需要加强这方面的覆盖。我们扩展了先前关于GAI工具描绘的全球南方国家异国情调的研究（Ghosh等人，2024）。我们分析了来自13个国家的个体生成的图像——印度、孟加拉国、巴布亚新几内亚、埃及、埃塞俄比亚、突尼斯、苏丹、利比亚、委内瑞拉、哥伦比亚、印度尼西亚、洪都拉斯和墨西哥——执行日常活动（如在家、上班、购物等），与来自3个全球北方国家——美国、英国、澳大利亚——的个体执行相同活动的图像进行比较。虽然全球北方的输出在图像和穿着活动适当服装的人之间显示出差异，但全球南方国家的个体被描绘为穿着相似服装，无论执行的活动如何，这表明一种异国情调模式，其中服装或其他文化特征被过度放大，牺牲了准确性。我们进一步展示了定性分析的案例研究，表明异国情调不仅针对全球南方国家，也针对西方背景中的边缘化群体，因为我们观察到全球北方中土著群体的类似异国情调化，以及对全球南方国家内边缘化群体的双重异国情调化。我们记录了这种工具的危害感知使用模式的启示，以及通过社区中心努力设计更好GAI工具的步骤。
</details>

<details>
    <summary>Key points</summary>
    * 分析了来自13个全球南方和3个全球北方国家的生成图像
    * 定性案例研究显示边缘化群体中的异国情调
    * 提出了社区中心步骤以设计更好的GAI工具
</details>
</details>

---


<details>
<summary><b> Draw Your Mind: Personalized Generation via Condition-Level Modeling in Text-to-Image Diffusion Models</b></summary>

* **Authors:** Hyungjin Kim, Seokho Ahn, Young-Duk Seo
* **arXiv ID:** 2508.03481
* **One-liner:** Proposed DrUM, a method for personalized T2I generation using user profiling and transformer-based adapters.
* **Published in:** arxiv (5 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.03481) | [[PDF]](https://arxiv.org/pdf/2508.03481) | [[Code]]()

> **核心创新**
> 通过潜在空间中的条件级建模实现准确个性化，无需微调。

<details>
    <summary>Abstract</summary>
    T2I扩散模型中的个性化生成旨在以最小用户干预自然地将个体用户偏好整合到生成过程中。然而，现有研究主要依赖大规模模型的提示级建模，往往由于T2I扩散模型输入标记容量有限而导致个性化不准确。为了解决这些限制，我们提出了DrUM，一种新方法，将用户分析与基于变压器的适配器集成，通过潜在空间中的条件级建模实现个性化生成。DrUM在大规模数据集上表现出强大性能，并与开源文本编码器无缝集成，使其与广泛使用的基础T2I模型兼容，无需额外微调。
</details>

<details>
    <summary>Key points</summary>
    * 将用户分析与基于变压器的适配器集成
    * 与开源文本编码器和基础T2I模型兼容
    * 在大规模数据集上表现出强大性能
</details>
</details>

---


<details>
<summary><b> LumiGen: An LVLM-Enhanced Iterative Framework for Fine-Grained Text-to-Image Generation</b></summary>

* **Authors:** Xiaoqi Dong, Xiangyu Zhou, Nicholas Evans, Yujia Lin
* **arXiv ID:** 2508.04732
* **One-liner:** Introduced LumiGen, an LVLM-enhanced iterative framework for fine-grained control in T2I generation.
* **Published in:** arxiv (5 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.04732) | [[PDF]](https://arxiv.org/pdf/2508.04732) | [[Code]]()

> **核心创新**
> 通过闭环反馈提升T2I性能，改进文本渲染和姿态表达。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）生成在扩散模型方面取得了显著进展，但在处理复杂指令、确保细粒度内容控制和保持深度语义一致性方面仍存在挑战。现有T2I模型往往在准确文本渲染、精确姿态生成或复杂组合一致性等任务上表现不佳。同时，视觉语言模型（LVLMs）在跨模态理解和指令跟随方面展示了强大能力。我们提出了LumiGen，一种新颖的LVLM增强迭代框架，旨在通过闭环、LVLM驱动的反馈机制提升T2I模型性能，特别是在需要细粒度控制的领域。LumiGen包括一个智能提示解析与增强（IPPA）模块，用于主动提示增强，以及一个迭代视觉反馈与优化（IVFR）模块，作为“视觉批评家”迭代纠正和优化生成图像。在挑战性的LongBench-T2I基准上评估，LumiGen实现了3.08的优异平均分数，优于最先进的基线。值得注意的是，我们的框架在文本渲染和姿态表达等关键维度上显示出显著改进，验证了LVLM集成对更可控和更高质量图像生成的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 智能提示解析与增强模块用于主动增强
    * 迭代视觉反馈与优化模块作为视觉批评家
    * 在LongBench-T2I基准上实现优异分数
</details>
</details>

---


<details>
<summary><b> UNCAGE: Contrastive Attention Guidance for Masked Generative Transformers in Text-to-Image Generation</b></summary>

* **Authors:** Wonjun Kang, Byeongkeun Ahn, Minjae Lee, Kevin Galim, Seunghyuk Oh, Hyung Il Koo, Nam Ik Cho
* **arXiv ID:** 2508.05399
* **One-liner:** Proposed UNCAGE, a training-free method to improve compositional fidelity in Masked Generative Transformers.
* **Published in:** arxiv (7 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.05399) | [[PDF]](https://arxiv.org/pdf/2508.05399) | [[Code]](https://github.com/furiosa-ai/uncage)

> **核心创新**
> 通过利用注意力图优先解掩码对象标记，增强文本图像对齐。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）生成已使用扩散模型和自回归模型积极研究。最近，掩码生成变压器作为自回归模型的替代方案受到关注，以克服因果注意力和自回归解码的固有局限性，通过双向注意力和并行解码实现高效高质量图像生成。然而，组合T2I生成仍然具有挑战性，因为即使最先进的扩散模型也常常无法准确绑定属性并实现适当的文本图像对齐。虽然扩散模型已在这方面被广泛研究，但掩码生成变压器表现出类似局限性，但尚未在此背景下被探索。为了解决这个问题，我们提出了Unmasking with Contrastive Attention Guidance（UNCAGE），一种新颖的无需训练方法，通过利用注意力图优先解掩码明确表示单个对象的标记来提高组合保真度。UNCAGE在多个基准和指标上一致提高了性能，且推理开销可忽略。我们的代码在GitHub上可用。
</details>

<details>
    <summary>Key points</summary>
    * 利用注意力图进行标记解掩码
    * 在基准上一致提高性能，开销可忽略
    * 解决组合T2I生成中的局限性
</details>
</details>

---


<details>
<summary><b> CoAR: Concept Injection into Autoregressive Models for Personalized Text-to-Image Generation</b></summary>

* **Authors:** Fangtai Wu, Mushui Liu, Weijie He, Wanggui He, Hao Jiang, Zhao Wang, Yunlong Yu
* **arXiv ID:** 2508.07341
* **One-liner:** Developed CoAR, a framework for customized image generation in unified AR models with minimal parameter tuning.
* **Published in:** arxiv (10 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.07341) | [[PDF]](https://arxiv.org/pdf/2508.07341) | [[Code]](https://github.com/KZF-kzf/CoAR)

> **核心创新**
> 在保持预训练参数冻结的同时实现主题和风格定制，提高效率。

<details>
    <summary>Abstract</summary>
    统一自回归（AR）模型在多模态理解和生成方面表现出色，但其在定制图像生成方面的潜力尚未被充分探索。现有定制生成方法依赖全微调或适配器，使其成本高昂且容易过拟合或灾难性遗忘。在本文中，我们提出了CoAR，一种新颖框架，用于将主题概念注入统一AR模型，同时保持所有预训练参数完全冻结。CoAR使用层间多模态上下文学习策略，仅用最少数量的参数学习有效、特定的主题表示。为了解决过拟合和语言漂移，我们进一步引入了正则化，以保留预训练分布并锚定上下文标记，提高主题保真度和重新上下文化。此外，CoAR支持在用户提供风格中的无需训练主题定制。实验表明，CoAR在主题驱动个性化和风格个性化方面均实现优异性能，同时在计算和内存效率上带来显著增益。值得注意的是，CoAR调整少于0.05%的参数，同时实现与最近Proxy-Tuning竞争的性能。代码在GitHub上可用。
</details>

<details>
    <summary>Key points</summary>
    * 层间多模态上下文学习用于主题表示
    * 正则化防止过拟合和语言漂移
    * 调整少于0.05%的参数以实现竞争性能
</details>
</details>

---


<details>
<summary><b> Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion</b></summary>

* **Authors:** Jiwon Kim, Pureum Kim, SeonHwa Kim, Soobin Park, Eunju Cha, Kyong Hwan Jin
* **arXiv ID:** 2508.09575
* **One-liner:** Proposed DRF, a training-free system for fine-grained control in controllable T2I diffusion models.
* **Published in:** arxiv (13 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.09575) | [[PDF]](https://arxiv.org/pdf/2508.09575) | [[Code]](https://github.com/jwonkm/DRF)

> **核心创新**
> 递归优化潜在表示以更好地反映外观和结构条件，确保一致性。

<details>
    <summary>Abstract</summary>
    可控文本到图像（T2I）扩散模型的最新进展，如Ctrl-X和FreeControl，已展示了无需辅助模块训练的强大空间和外观控制。然而，这些模型往往难以准确保留空间结构，并无法捕捉与对象姿态和场景布局相关的细粒度条件。为了解决这些挑战，我们提出了一种无需训练的双重递归反馈（DRF）系统，在可控T2I模型中正确反映控制条件。所提出的DRF包括外观反馈和生成反馈，递归优化中间潜在表示，以更好地反映给定外观信息和用户意图。这种双重更新机制引导潜在表示朝向可靠流形，有效整合结构和外观属性。我们的方法即使在类不变结构-外观融合中也能实现细粒度生成，例如将人类运动转移到老虎形态上。广泛实验证明了我们的方法在生成高质量、语义连贯和结构一致图像方面的有效性。我们的源代码在GitHub上可用。
</details>

<details>
    <summary>Key points</summary>
    * 双重递归反馈，包括外观和生成反馈
    * 实现类不变结构-外观融合
    * 生成高质量、语义连贯的图像
</details>
</details>

---


<details>
<summary><b> High Fidelity Text to Image Generation with Contrastive Alignment and Structural Guidance</b></summary>

* **Authors:** Danyi Gao
* **arXiv ID:** 2508.10280
* **One-liner:** Proposed a high-fidelity image generation method integrating text-image contrastive constraints with structural guidance mechanisms.
* **Published in:** arxiv (14 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.10280) | [[PDF]](https://arxiv.org/pdf/2508.10280) | [[Code]]()

> **核心创新**
> 在不增加计算复杂度的前提下，增强了文本驱动图像生成中的语义对齐和结构一致性。

<details>
    <summary>Abstract</summary>
    本文针对现有文本驱动图像生成方法在语义对齐精度和结构一致性方面的性能瓶颈，提出了一种通过整合文本-图像对比约束与结构引导机制的高保真图像生成方法。
</details>

<details>
    <summary>Key points</summary>
    * 用于跨模态对齐的对比学习模块
    * 如语义布局图或边缘草图等结构先验用于空间级结构建模
    * 对比损失、结构一致性损失和语义保持损失的联合优化
</details>
</details>

---


<details>
<summary><b> CEIDM: A Controlled Entity and Interaction Diffusion Model for Enhanced Text-to-Image Generation</b></summary>

* **Authors:** Mingyue Yang, Dianxi Shi, Jialu Zhou, Xinyu Wei, Leqian Li, Shaowu Yang, Chunping Qiu
* **arXiv ID:** 2508.17760
* **One-liner:** Introduced CEIDM, a diffusion-based method with dual controls for entity and interaction in T2I generation.
* **Published in:** arxiv (25 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.17760) | [[PDF]](https://arxiv.org/pdf/2508.17760) | [[Code]]()

> **核心创新**
> 通过挖掘隐式关系和聚类动作，改进了对实体及其交互的控制。

<details>
    <summary>Abstract</summary>
    在文本到图像（T2I）生成中，实体及其复杂交互的复杂性对基于扩散模型的T2I方法提出了重大挑战：如何有效控制实体及其交互以生成高质量图像。为此，我们提出了CEIDM，一种基于扩散模型的图像生成方法，具有实体和交互的双重控制。
</details>

<details>
    <summary>Key points</summary>
    * 基于大型语言模型（LLM）的实体交互关系挖掘
    * 交互动作聚类和偏移方法
    * 具有多尺度和动态特征融合的实体控制网络
</details>
</details>

---


<details>
<summary><b> HADIS: Hybrid Adaptive Diffusion Model Serving for Efficient Text-to-Image Generation</b></summary>

* **Authors:** Qizheng Yang, Tung-I Chen, Siyu Zhao, Ramesh K. Sitaraman, Hui Guan
* **arXiv ID:** 2509.00642
* **One-liner:** Developed HADIS, a hybrid adaptive diffusion model serving system for efficient real-time deployment.
* **Published in:** arxiv (31 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.00642) | [[PDF]](https://arxiv.org/pdf/2509.00642) | [[Code]]()

> **核心创新**
> 优化了级联模型选择、查询路由和资源分配，以提高响应质量并降低延迟。

<details>
    <summary>Abstract</summary>
    文本到图像扩散模型已实现卓越的视觉质量，但计算成本高昂，使得实时、可扩展部署具有挑战性。现有的查询感知服务系统通过级联轻量级和重量级模型来缓解成本，但大多依赖固定级联配置，并将所有提示路由通过初始轻量级阶段，浪费资源于复杂查询。我们提出了HADIS，一种混合自适应扩散模型服务系统，联合优化级联模型选择、查询路由和资源分配。
</details>

<details>
    <summary>Key points</summary>
    * 基于规则的提示路由器，用于直接路由到重量级模型
    * 离线分析生成帕累托最优级联配置表
    * 运行时根据延迟和工作负载约束选择配置和GPU分配
</details>
</details>

---


<details>
<summary><b> Data-Driven Loss Functions for Inference-Time Optimization in Text-to-Image Generation</b></summary>

* **Authors:** Sapir Esther Yiflach, Yuval Atzmon, Gal Chechik
* **arXiv ID:** 2509.02295
* **One-liner:** Introduced Learn-to-Steer, a framework for learning data-driven objectives to improve spatial reasoning in T2I models.
* **Published in:** arxiv (2 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.02295) | [[PDF]](https://arxiv.org/pdf/2509.02295) | [[Code]](https://github.com/nerfies/nerfies.github.io)

> **核心创新**
> 通过从交叉注意力图解码关系并使用学习到的损失，显著提高了空间准确性。

<details>
    <summary>Abstract</summary>
    文本到图像扩散模型可以生成令人惊叹的视觉效果，但它们在儿童认为简单的任务上经常失败——例如将狗放在泰迪熊的右边而不是左边。当组合变得更加不寻常时——如长颈鹿在飞机上方——这些失败变得更加明显。现有方法试图通过模型微调或使用手工损失进行测试时优化来修复这些空间推理失败，但这些损失是次优的。我们没有强加关于空间编码的假设，而是提出直接从模型的内部表示中学习这些目标。我们引入了Learn-to-Steer，一种新颖的框架，学习数据驱动的目标用于测试时优化，而不是手工制作它们。我们的关键洞察是训练一个轻量级分类器，从扩散模型的交叉注意力图中解码空间关系，然后在推理过程中部署该分类器作为学习到的损失函数。训练这样的分类器提出了一个惊人的挑战：它们可以通过检测语言痕迹而不是学习真实的空间模式来走捷径。我们通过双重反转策略解决了这个问题，该策略强制执行几何理解。
</details>

<details>
    <summary>Key points</summary>
    * 在交叉注意力图上训练的轻量级分类器
    * 双重反转策略以强制执行几何理解
    * 在推理过程中作为学习到的损失函数部署
</details>
</details>

---


<details>
<summary><b> PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting</b></summary>

* **Authors:** Linqing Wang, Ximing Xing, Yiji Cheng, Zhiyuan Zhao, Donghao Li, Tiankai Hang, Jiale Tao, Qixun Wang, Ruihuang Li, Comi Chen, Xin Li, Mingrui Wu, Xinchi Deng, Shuyang Gu, Chunyu Wang, Qinglin Lu
* **arXiv ID:** 2509.04545
* **One-liner:** Proposed PromptEnhancer, a universal prompt rewriting framework to enhance T2I model alignment without weight modifications.
* **Published in:** arxiv (4 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.04545) | [[PDF]](https://arxiv.org/pdf/2509.04545) | [[Code]](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1)

> **核心创新**
> 通过使用细粒度奖励模型训练CoT重写器，改善了图像-文本对齐。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）扩散模型的最新进展在生成高保真图像方面展示了卓越能力。然而，这些模型往往难以忠实渲染复杂的用户提示，特别是在属性绑定、否定和组合关系等方面。这导致用户意图与生成输出之间的显著不匹配。为了解决这一挑战，我们引入了PromptEnhancer，一种新颖且通用的提示重写框架，无需修改其权重即可增强任何预训练的T2I模型。与先前依赖模型特定微调或隐式奖励信号（如图像奖励分数）的方法不同，我们的框架将重写器与生成器解耦。我们通过强化学习训练一个思维链（CoT）重写器来实现这一点，该重写器由一个我们称为AlignEvaluator的专用奖励模型指导。AlignEvaluator经过训练，基于从常见T2I失败模式综合分析得出的24个关键点的系统分类法，提供显式和细粒度的反馈。通过优化CoT重写器以最大化来自AlignEvaluator的奖励，我们的框架学会生成更精确被T2I模型解释的提示。在HunyuanImage 2.1模型上的广泛实验表明，PromptEnhancer在广泛的语义和组合挑战中显著改善了图像-文本对齐。此外，我们引入了一个新的高质量人类偏好基准，以促进未来在这一方向的研究。
</details>

<details>
    <summary>Key points</summary>
    * 通过强化学习训练的思维链重写器
    * 基于24个关键点的AlignEvaluator奖励模型
    * 解耦的重写器和生成器，用于通用应用
</details>
</details>

---


<details>
<summary><b> EditIDv2: Editable ID Customization with Data-Lubricated ID Feature Integration for Text-to-Image Generation</b></summary>

* **Authors:** Guandong Li, Zhaobin Chu
* **arXiv ID:** 2509.05659
* **One-liner:** Presented EditIDv2, a tuning-free solution for high-complexity narrative scenes and long text inputs in character editing.
* **Published in:** arxiv (6 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.05659) | [[PDF]](https://arxiv.org/pdf/2509.05659) | [[Code]]()

> **核心创新**
> 使用最小数据和先进注意力分解，实现了具有身份一致性的深度语义编辑。

<details>
    <summary>Abstract</summary>
    我们提出了EditIDv2，一种专门为高复杂性叙事场景和长文本输入设计的免调优解决方案。现有角色编辑方法在简单提示下表现良好，但在面对包含多个语义层、时间逻辑和复杂上下文关系的长文本叙事时，往往遭受编辑能力下降、语义理解偏差和身份一致性崩溃的问题。在EditID中，我们分析了ID集成模块对可编辑性的影响。在EditIDv2中，我们进一步探索并解决了ID特征集成模块的影响。EditIDv2的核心是在最小数据润滑下讨论可编辑性注入问题。通过PerceiverAttention的精细分解、ID损失和与扩散模型的联合动态训练，以及集成模块的离线融合策略，我们仅使用少量数据润滑，在复杂叙事环境中实现了深度、多级语义编辑，同时保持身份一致性。这满足了长提示和高质量图像生成的需求，并在IBench评估中取得了优异结果。
</details>

<details>
    <summary>Key points</summary>
    * PerceiverAttention的分解
    * ID损失和联合动态训练
    * 集成模块的离线融合策略
</details>
</details>

---


<details>
<summary><b> BiasMap: Leveraging Cross-Attentions to Discover and Mitigate Hidden Social Biases in Text-to-Image Generation</b></summary>

* **Authors:** Rajatsubhra Chakraborty, Xujun Che, Depeng Xu, Cori Faklaris, Xi Niu, Shuhan Yuan
* **arXiv ID:** 2509.13496
* **One-liner:** Proposed BiasMap, a framework for uncovering and mitigating latent concept-level representational biases in stable diffusion models.
* **Published in:** arxiv (16 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.13496) | [[PDF]](https://arxiv.org/pdf/2509.13496) | [[Code]]()

> **核心创新**
> 使用交叉注意力归因揭示了人口统计和语义之间的概念纠缠，并减少了它。

<details>
    <summary>Abstract</summary>
    偏差发现对于黑盒生成模型至关重要，尤其是文本到图像（TTI）模型。现有工作主要关注输出级的人口统计分布，这不一定保证在缓解后概念表示被解耦。我们提出了BiasMap，一个模型无关的框架，用于揭示稳定扩散模型中潜在概念级表示偏差。BiasMap利用交叉注意力归因图来揭示人口统计（如性别、种族）和语义（如职业）之间的结构纠缠，更深入地探究图像生成过程中的表示偏差。使用这些概念的归因图，我们通过交并比（IoU）量化空间人口统计-语义概念纠缠，提供了一个洞察隐藏于现有公平性发现方法中的偏差的视角。此外，我们进一步利用BiasMap进行偏差缓解，通过能量引导扩散采样直接修改潜在噪声空间，并在去噪过程中最小化期望SoftIoU。我们的发现表明，现有的公平性干预可能减少输出分布差距，但往往未能解耦概念级耦合，而我们的缓解方法可以在图像生成中缓解概念纠缠，同时补充分布偏差缓解。
</details>

<details>
    <summary>Key points</summary>
    * 用于偏差发现的交叉注意力归因图
    * 基于IoU的概念纠缠量化
    * 用于缓解的能量引导扩散采样
</details>
</details>

---


<details>
<summary><b> DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models</b></summary>

* **Authors:** Komal Kumar, Rao Muhammad Anwer, Fahad Shahbaz Khan, Salman Khan, Ivan Laptev, Hisham Cholakkal
* **arXiv ID:** 2509.22793
* **One-liner:** Introduced DEFT, a decompositional efficient fine-tuning framework for T2I models.
* **Published in:** arxiv (26 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.22793) | [[PDF]](https://arxiv.org/pdf/2509.22793) | [[Code]](https://github.com/MAXNORM8650/DEFT)

> **核心创新**
> 在个性化和多任务适应中，以最少的可训练参数实现了最先进的性能。

<details>
    <summary>Abstract</summary>
    预训练文本到图像（T2I）模型的高效微调涉及调整模型以适应特定任务或数据集，同时最小化计算资源并限制可训练参数数量。然而，它经常面临在目标分布对齐之间权衡的挑战：从有限图像中学习新概念以进行个性化，并保留统一多个任务所需的指令能力，同时保持可编辑性（与各种提示或上下文生成对齐）。在这项工作中，我们引入了DEFT，分解高效微调，一种高效微调框架，通过将其更新分解为两个具有两个可训练矩阵的组件来适应预训练权重矩阵：（1）投影到由低秩矩阵张成的低秩子空间的补空间上，和（2）低秩更新。单个可训练低秩矩阵定义了子空间，而另一个可训练低秩矩阵在该子空间内实现灵活的参数适应。我们在Dreambooth和Dreambench Plus数据集上进行了广泛实验用于个性化，在InsDet数据集上用于对象和场景适应，在VisualCloze数据集上通过视觉上下文学习与Stable Diffusion和统一模型构建通用图像生成框架。我们的结果展示了最先进的性能，突出了高效微调的新兴特性。
</details>

<details>
    <summary>Key points</summary>
    * 权重矩阵更新分解为两个组件
    * 低秩子空间投影和更新
    * 在各种数据集上的广泛实验
</details>
</details>

---


<details>
<summary><b> No Concept Left Behind: Test-Time Optimization for Compositional Text-to-Image Generation</b></summary>

* **Authors:** Mohammad Hossein Sameti, Amir M. Mansourian, Arash Marioriyad, Soheil Fadaee Oshyani, Mohammad Hossein Rohban, Mahdieh Soleymani Baghshah
* **arXiv ID:** 2509.23457
* **One-liner:** Proposed a fine-grained test-time optimization framework to enhance compositional faithfulness in T2I generation.
* **Published in:** arxiv (27 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.23457) | [[PDF]](https://arxiv.org/pdf/2509.23457) | [[Code]](https://github.com/AmirMansurian/NoConceptLeftBehind)

> **核心创新**
> 通过分解提示并使用概念级反馈，改善了概念覆盖和忠实度。

<details>
    <summary>Abstract</summary>
    尽管文本到图像（T2I）模型最近取得了进展，但它们往往无法忠实渲染复杂提示的所有元素，经常遗漏或错误表示特定对象和属性。测试时优化已成为一种有前景的方法，通过在不需重新训练的情况下精炼生成来解决这一限制。在本文中，我们提出了一种细粒度测试时优化框架，增强了T2I生成中的组合忠实度。与大多数先前仅依赖全局图像/文本相似性分数的方法不同，我们的方法将输入提示分解为语义概念，并在全局和概念级别评估对齐。使用细粒度变体的CLIP计算概念级对应，产生关于缺失或不准确概念的详细反馈。该反馈被输入到迭代提示精炼循环中，使大型语言模型能够提出改进的提示。在DrawBench和CompBench提示上的实验表明，我们的方法在概念覆盖和人类判断的忠实度上显著优于标准测试时优化和基础T2I模型。
</details>

<details>
    <summary>Key points</summary>
    * 用于概念级对齐的细粒度CLIP
    * 带有LLM的迭代提示精炼循环
    * 在全局和概念级别上的评估
</details>
</details>

---


<details>
<summary><b> Free Lunch Alignment of Text-to-Image Diffusion Models without Preference Image Pairs</b></summary>

* **Authors:** Jia Jun Cheng Xian, Muchen Li, Haotian Yang, Xin Tao, Pengfei Wan, Leonid Sigal, Renjie Liao
* **arXiv ID:** 2509.25771
* **One-liner:** Introduced Text Preference Optimization (TPO), a framework for aligning T2I models without paired image preference data.
* **Published in:** arxiv (30 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.25771) | [[PDF]](https://arxiv.org/pdf/2509.25771) | [[Code]](https://github.com/DSL-Lab/T2I-Free-Lunch-Alignment)

> **核心创新**
> 通过使用LLM在扰动提示上训练，实现了更好的人类偏好分数和对齐。

<details>
    <summary>Abstract</summary>
    基于扩散的文本到图像（T2I）模型的最新进展在从文本提示生成高质量图像方面取得了显著成功。然而，确保文本与生成图像之间的准确对齐对于最先进的扩散模型仍然是一个重大挑战。为了解决这个问题，现有研究采用带有人类反馈的强化学习（RLHF）来将T2I输出与人类偏好对齐。这些方法要么直接依赖配对的图像偏好数据，要么需要学习到的奖励函数，两者都严重依赖成本高昂的高质量人类注释，因此面临可扩展性限制。在这项工作中，我们引入了文本偏好优化（TPO），一个框架，实现了T2I模型的“免费午餐”对齐，无需配对的图像偏好数据即可实现对齐。TPO通过训练模型偏好匹配提示而非不匹配提示来工作，这些不匹配提示是使用大型语言模型扰动原始标题构建的。我们的框架是通用的，与现有的基于偏好的算法兼容。我们将DPO和KTO扩展到我们的设置，产生了TDPO和TKTO。在多个基准上的定量和定性评估表明，我们的方法 consistently 优于其原始对应物，提供更好的人类偏好分数和改进的文本到图像对齐。
</details>

<details>
    <summary>Key points</summary>
    * 使用匹配与不匹配提示进行训练
    * 将DPO和KTO扩展到TDPO和TKTO
    * 与基于偏好算法兼容的通用框架
</details>
</details>

---


<details>
<summary><b> VQGAN-CLIP: Open Domain Image Generation and Editing with Natural Language Guidance</b></summary>

* **Authors:** Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, Edward Raff
* **arXiv ID:** 2204.08583
* **One-liner:** Introduced a training-free method using CLIP to guide VQGAN for high-quality image generation and editing from complex text prompts.
* **Published in:** arxiv (18 Apr 2022)
* **Links:** [[Paper]](https://arxiv.org/abs/2204.08583) | [[PDF]](https://arxiv.org/pdf/2204.08583) | [[Code]]()

> **核心创新**
> 通过利用多模态指导，在无需模型训练的情况下实现了图像生成和编辑的卓越视觉质量。

<details>
    <summary>Abstract</summary>
    从开放域文本提示生成和编辑图像是一项具有挑战性的任务，以往需要昂贵且专门训练的模型。我们展示了一种新颖方法，能够从语义复杂的文本提示中生成高视觉质量的图像，无需任何训练，通过使用多模态编码器指导图像生成。我们在多种任务上展示了使用CLIP [37] 指导VQGAN [11] 比先前灵活性较低的方法（如DALL-E [38]、GLIDE [33] 和 Open-Edit [24]）产生更高视觉质量的输出，尽管未针对所呈现任务进行训练。我们的代码可在公共仓库中获取。
</details>

<details>
    <summary>Key points</summary>
    * 使用CLIP指导VQGAN进行图像生成
    * 应用于多种任务而无需额外训练
    * 在视觉质量上超越了DALL-E和GLIDE等模型
</details>
</details>

---


<details>
<summary><b> InstructPix2Pix: Learning to Follow Image Editing Instructions</b></summary>

* **Authors:** Tim Brooks, Aleksander Holynski, Alexei A. Efros
* **arXiv ID:** 2211.09800
* **One-liner:** Developed InstructPix2Pix, a model for fast image editing from human instructions using generated training data.
* **Published in:** arxiv (17 Nov 2022)
* **Links:** [[Paper]](https://arxiv.org/abs/2211.09800) | [[PDF]](https://arxiv.org/pdf/2211.09800) | [[Code]](https://github.com/timothybrooks/instruct-pix2pix)

> **核心创新**
> 通过在GPT-3和Stable Diffusion合成的数据集上训练，实现了在几秒钟内快速编辑图像。

<details>
    <summary>Abstract</summary>
    我们提出了一种根据人类指令编辑图像的方法：给定输入图像和告诉模型要做什么的书面指令，我们的模型遵循这些指令编辑图像。为了获取此问题的训练数据，我们结合了两个大型预训练模型的知识——语言模型（GPT-3）和文本到图像模型（Stable Diffusion）——以生成一个大型图像编辑示例数据集。我们的条件扩散模型InstructPix2Pix在我们的生成数据上训练，并在推理时泛化到真实图像和用户编写的指令。由于它在正向传递中执行编辑，不需要每个示例的微调或反转，我们的模型在几秒钟内快速编辑图像。我们展示了针对多样输入图像和书面指令的引人注目的编辑结果。
</details>

<details>
    <summary>Key points</summary>
    * 结合GPT-3和Stable Diffusion生成训练数据
    * 训练条件扩散模型用于基于指令的编辑
    * 无需微调即可泛化到真实图像和用户指令
</details>
</details>

---


<details>
<summary><b> MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing</b></summary>

* **Authors:** Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, Yu Su
* **arXiv ID:** 2306.10012
* **One-liner:** Created MagicBrush, the first large-scale manually annotated dataset for instruction-guided image editing.
* **Published in:** arxiv (16 Jun 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2306.10012) | [[PDF]](https://arxiv.org/pdf/2306.10012) | [[Code]](https://github.com/OSU-NLP-Group/MagicBrush)

> **核心创新**
> 通过在高质量标注数据上微调，提高了模型性能，解决了自动数据集中的噪声问题。

<details>
    <summary>Abstract</summary>
    文本引导的图像编辑在日常生活中广泛需要，从个人使用到专业应用如Photoshop。然而，现有方法要么是零样本的，要么在自动合成的数据集上训练，这些数据集包含大量噪声。因此，在实际应用中，它们仍需要大量手动调整才能产生理想结果。为了解决这个问题，我们引入了MagicBrush（<a href="https://osu-nlp-group.github.io/MagicBrush/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>），这是第一个大规模、手动标注的指令引导真实图像编辑数据集，涵盖多样场景：单轮、多轮、提供掩码和无掩码编辑。MagicBrush包含超过10K手动标注的三元组（源图像、指令、目标图像），支持训练大规模文本引导图像编辑模型。我们在MagicBrush上微调InstructPix2Pix，并显示新模型根据人类评估能产生更好的图像。我们进一步进行广泛实验，从多个维度评估当前图像编辑基线，包括定量、定性和人类评估。结果揭示了我们数据集的挑战性以及当前基线与现实世界编辑需求之间的差距。
</details>

<details>
    <summary>Key points</summary>
    * 收集了超过10K手动标注的三元组（源图像、指令、目标图像）
    * 在MagicBrush上微调了InstructPix2Pix
    * 通过评估显示了更好的图像质量和数据集的挑战性
</details>
</details>

---


<details>
<summary><b> An Item is Worth a Prompt: Versatile Image Editing with Disentangled Control</b></summary>

* **Authors:** Aosong Feng, Weikang Qiu, Jinbin Bai, Xiao Zhang, Zhen Dong, Kaicheng Zhou, Rex Ying, Leandros Tassiulas
* **arXiv ID:** 2403.04880
* **One-liner:** Proposed D-Edit, a framework for versatile image editing by disentangling item-prompt interactions in diffusion models.
* **Published in:** arxiv (7 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.04880) | [[PDF]](https://arxiv.org/pdf/2403.04880) | [[Code]]()

> **核心创新**
> 在多种类型（基于图像、基于文本、基于掩码、项目移除）的编辑中实现了最先进的结果，采用统一方法。

<details>
    <summary>Abstract</summary>
    基于文本到图像扩散模型（DPMs）的成功，图像编辑是使人类与AI生成内容交互的重要应用。在各种编辑方法中，提示空间内的编辑因其控制语义的能力和简单性而受到更多关注。然而，由于扩散模型通常在描述性文本标题上预训练，直接编辑文本提示中的词语通常导致完全不同的生成图像，违反了图像编辑的要求。另一方面，现有编辑方法通常考虑引入空间掩码以保留未编辑区域的身份，但这些通常被DPMs忽略，从而导致不和谐的编辑结果。针对这两个挑战，在本工作中，我们提出将全面的图像-提示交互解耦为多个项目-提示交互，每个项目链接到一个特殊学习到的提示。所得框架名为D-Edit，基于预训练扩散模型，解耦交叉注意力层，并采用两步优化构建项目-提示关联。然后，可以通过操作相应提示对特定项目应用多功能图像编辑。我们在四种编辑操作中展示了最先进的结果，包括基于图像、基于文本、基于掩码的编辑和项目移除，覆盖大多数编辑应用类型，全部在一个统一框架内。值得注意的是，D-Edit是第一个能够（1）通过掩码编辑实现项目编辑和（2）结合基于图像和文本的编辑的框架。我们通过定性和定量评估展示了针对多样图像集合的编辑结果的质量和多功能性。
</details>

<details>
    <summary>Key points</summary>
    * 解耦了扩散模型中的交叉注意力层
    * 使用两步优化构建项目-提示关联
    * 通过掩码操作实现项目编辑并组合编辑类型
</details>
</details>

---


<details>
<summary><b> InstructGIE: Towards Generalizable Image Editing</b></summary>

* **Authors:** Zichong Meng, Changdi Yang, Jun Liu, Hao Tang, Pu Zhao, Yanzhi Wang
* **arXiv ID:** 2403.05018
* **One-liner:** Introduced a robust image editing framework with enhanced in-context learning and language unification.
* **Published in:** arxiv (8 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.05018) | [[PDF]](https://arxiv.org/pdf/2403.05018) | [[Code]]()

> **核心创新**
> 通过整合VMamba块、编辑移位匹配和选择性区域匹配，提高了泛化能力和质量。

<details>
    <summary>Abstract</summary>
    图像编辑的最新进展由去噪扩散模型的发展推动，标志着该领域的重大飞跃。尽管有这些进展，最近图像编辑方法的泛化能力仍然受限。针对这一挑战，我们的研究引入了一种新颖的图像编辑框架，通过增强上下文学习能力和统一语言指令来提高泛化鲁棒性。该框架包含一个专门为图像编辑任务优化的模块，利用VMamba块和编辑移位匹配策略来增强上下文学习。此外，我们揭示了一种选择性区域匹配技术，专门设计用于解决和纠正生成图像中的损坏细节，如人脸特征，以进一步提高质量。我们方法的另一个关键创新是语言统一技术的整合，该技术对齐语言嵌入与编辑语义以提升图像编辑质量。此外，我们编译了第一个用于图像编辑的视觉提示和编辑指令数据集，可用于增强上下文能力。在该数据集上训练后，我们的方法不仅为训练任务实现了卓越的合成质量，还通过定制提示展示了在未见视觉任务上的鲁棒泛化能力。
</details>

<details>
    <summary>Key points</summary>
    * 利用VMamba块和编辑移位匹配进行上下文学习
    * 应用选择性区域匹配纠正损坏细节
    * 使用语言统一对齐嵌入与编辑语义
</details>
</details>

---


<details>
<summary><b> Leveraging LLMs for On-the-Fly Instruction Guided Image Editing</b></summary>

* **Authors:** Rodrigo Santos, João Silva, António Branco
* **arXiv ID:** 2403.08004
* **One-liner:** Developed a preparation-free method for instruction-guided image editing using image captioning and DDIM inversion.
* **Published in:** arxiv (12 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.08004) | [[PDF]](https://arxiv.org/pdf/2403.08004) | [[Code]]()

> **核心创新**
> 无需初步训练或微调，实现了具有竞争力的性能。

<details>
    <summary>Abstract</summary>
    语言处理和图像处理的结合持续吸引更多兴趣，鉴于最近利用两个研究领域优势的令人印象深刻的进展。在这些进展中，仅基于自然语言指令编辑图像的任务作为一个最具挑战性的努力脱颖而出。虽然最近针对此任务的方法以某种方式诉诸某种形式的初步准备、训练或微调，本文探索了一种新颖方法：我们提出一种无需准备的方法，允许即时进行指令引导的图像编辑。该方法组织为三个适当编排的步骤，诉诸图像描述和DDIM反转，随后获取编辑方向嵌入，然后是图像编辑本身。尽管无需初步准备，我们的方法被证明是有效和具有竞争力的，在MAGICBRUSH数据集上评估时，超越了最近最先进的模型。
</details>

<details>
    <summary>Key points</summary>
    * 编排了三个步骤：图像描述、DDIM反转和编辑方向嵌入
    * 无需准备步骤
    * 在MAGICBRUSH数据集上超越了最先进模型
</details>
</details>

---


<details>
<summary><b> Enhancing Text-to-Image Editing via Hybrid Mask-Informed Fusion</b></summary>

* **Authors:** Aoxue Li, Mingyang Yi, Zhenguo Li
* **arXiv ID:** 2405.15313
* **One-liner:** Proposed MaSaFusion to improve text-guided image editing by incorporating mask-informed fusion in self-attention modules.
* **Published in:** arxiv (24 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.15313) | [[PDF]](https://arxiv.org/pdf/2405.15313) | [[Code]]()

> **核心创新**
> 通过减少纹理保留和新特征创建中的干扰，解决了基于扩散的编辑中的不一致性问题。

<details>
    <summary>Abstract</summary>
    最近，通过应用扩散模型，文本到图像（T2I）编辑得到了极大推动。尽管生成图像在视觉上具有前景，但与预期文本提示的不一致性仍然普遍存在。本文旨在通过解决其局限性，系统性地改进基于扩散模型的文本引导图像编辑技术。值得注意的是，基于扩散的编辑中的常见思想首先通过反转技术（如DDIM反转）重建源图像。然后遵循一个融合过程，仔细整合源中间（隐藏）状态（通过反转获得）与目标图像的中间状态。不幸的是，这种标准流程在许多情况下失败，由于纹理保留的干扰和某些区域中新特征的创建。为了缓解这一点，我们引入人类标注作为外部知识，将编辑限制在“掩码知情”区域内。然后我们在模型的自注意力模块中仔细融合编辑图像与源图像和构建的中间图像。广泛的实证结果表明，提出的“MaSaFusion”显著改进了现有的T2I编辑技术。
</details>

<details>
    <summary>Key points</summary>
    * 使用人类标注定义掩码知情区域
    * 在自注意力模块中融合源图像和目标图像
    * 实证上提高了编辑准确性和质量
</details>
</details>

---


<details>
<summary><b> Text Guided Image Editing with Automatic Concept Locating and Forgetting</b></summary>

* **Authors:** Jia Li, Lijie Hu, Zhixian He, Jingfeng Zhang, Tianhang Zheng, Di Wang
* **arXiv ID:** 2405.19708
* **One-liner:** Introduced Locate and Forget (LaF) method for precise text-guided image editing by comparing syntactic trees.
* **Published in:** arxiv (30 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.19708) | [[PDF]](https://arxiv.org/pdf/2405.19708) | [[Code]]()

> **核心创新**
> 通过定位和遗忘目标概念，增强了文本提示和图像修改之间的语义对齐。

<details>
    <summary>Abstract</summary>
    随着文本引导的图像到图像扩散模型的进步，图像编辑取得了显著进展。然而，一个持续的挑战在于基于文本指令无缝将对象融入图像，而不依赖额外的用户提供指导。文本和图像本质上是不同的模态，带来了充分捕捉通过语言传达的语义意图并准确将其转化为期望视觉修改的困难。因此，文本引导的图像编辑模型通常产生具有残留对象属性的生成结果，这些属性不完全符合人类期望。为了解决这一挑战，模型应有效理解图像内容，远离提供的文本编辑提示与实际图像修改之间的脱节。在我们的论文中，我们提出了一种名为“定位与遗忘”（LaF）的新方法，该方法通过比较目标提示和输入图像中场景描述的句法树，有效定位图像中潜在的目标概念进行修改，意图在生成图像中遗忘它们的存在线索。与基线相比，我们的方法在文本引导图像编辑任务中定性和定量地展示了其优越性。
</details>

<details>
    <summary>Key points</summary>
    * 比较了目标提示和场景描述的句法树
    * 定位了用于修改的目标概念
    * 在生成图像中遗忘存在线索以改善对齐
</details>
</details>

---


<details>
<summary><b> Empowering Visual Creativity: A Vision-Language Assistant to Image Editing Recommendations</b></summary>

* **Authors:** Tiancheng Shen, Jun Hao Liew, Long Mai, Lu Qi, Jiashi Feng, Jiaya Jia
* **arXiv ID:** 2406.00121
* **One-liner:** Proposed Image Editing Recommendation (IER) task and Creativity-VLA framework for generating creative editing instructions from vague prompts.
* **Published in:** arxiv (31 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.00121) | [[PDF]](https://arxiv.org/pdf/2406.00121) | [[Code]]()

> **核心创新**
> 通过自动生成多样化和相关的编辑指令，弥合了用户构思中的差距。

<details>
    <summary>Abstract</summary>
    基于文本的图像生成和编辑的进展彻底改变了内容创作，使用户能够从富有想象力的文本提示创建令人印象深刻的内容。然而，现有方法并非设计用于在典型场景中处理过于简化的提示，这些场景中用户通常以模糊或抽象的目的开始编辑。这些场景要求用户付出精心构思的努力，以弥合这种模糊起点与描述期望结果所需的详细创意想法之间的差距。在本文中，我们引入了图像编辑推荐（IER）任务。该任务旨在从输入图像和代表用户未充分指定编辑目的的简单提示自动生成多样化的创意编辑指令。为此，我们引入了Creativity-Vision Language Assistant（Creativity-VLA），一个专门为编辑指令生成设计的多模态框架。我们在专门为IER策划的编辑指令数据集上训练Creativity-VLA。我们进一步通过一种新颖的“用于定位的令牌”机制增强我们的模型，使其支持全局和局部编辑操作。我们的实验结果表明，我们的方法在建议指令方面的有效性，这些指令不仅包含引人入胜的创意元素，而且保持与输入图像和用户初始提示的高度相关性。
</details>

<details>
    <summary>Key points</summary>
    * 引入了IER任务用于自动指令生成
    * 在策划的编辑指令数据集上训练了Creativity-VLA
    * 使用用于定位的令牌机制支持全局和局部编辑
</details>
</details>

---


<details>
<summary><b> The Curious Case of End Token: A Zero-Shot Disentangled Image Editing using CLIP</b></summary>

* **Authors:** Hidir Yesiltepe, Yusuf Dalva, Pinar Yanardag
* **arXiv ID:** 2406.00457
* **One-liner:** Demonstrated that CLIP enables disentangled editing in diffusion models in a zero-shot manner.
* **Published in:** arxiv (1 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.00457) | [[PDF]](https://arxiv.org/pdf/2406.00457) | [[Code]]()

> **核心创新**
> 提供了一种轻量级方法，用于在不损害图像连贯性的情况下进行精确属性操作。

<details>
    <summary>Abstract</summary>
    扩散模型在创建高质量图像方面已变得突出。然而，与GAN模型因其以解耦方式编辑图像的能力而受到赞誉不同，基于扩散的文本到图像模型难以在不损害图像连贯性的情况下实现相同水平的精确属性操作。在本文中，CLIP（常用于流行的文本到图像扩散模型如Stable Diffusion）能够以零样本方式执行解耦编辑。通过与最先进编辑方法的定性和定量比较，我们显示我们的方法产生了具有竞争力的结果。这一见解可能为将这种方法应用于各种任务（包括图像和视频编辑）打开机会，提供一种轻量级且高效的解耦编辑方法。
</details>

<details>
    <summary>Key points</summary>
    * 应用CLIP进行零样本解耦编辑
    * 与最先进方法进行了定性和定量比较
    * 强调了在图像和视频编辑应用中的潜力
</details>
</details>

---


<details>
<summary><b> UltraEdit: Instruction-based Fine-Grained Image Editing at Scale</b></summary>

* **Authors:** Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, Baobao Chang
* **arXiv ID:** 2407.05282
* **One-liner:** Created a large-scale, high-quality dataset for instruction-based image editing.
* **Published in:** arxiv (7 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.05282) | [[PDF]](https://arxiv.org/pdf/2407.05282) | [[Code]](https://github.com/HaozheZhao/UltraEdit)

> **核心创新**
> 系统化生成了一个具有更广泛编辑指令、真实图像源和基于区域编辑支持的数据集。

<details>
    <summary>Abstract</summary>
    本文提出了UltraEdit，一个大规模（约400万编辑样本）、自动生成的基于指令的图像编辑数据集。我们的核心思想是解决现有图像编辑数据集（如InstructPix2Pix和MagicBrush）的缺陷，并提供一种系统化方法以产生大量高质量图像编辑样本。UltraEdit具有几个显著优势：1）通过利用大型语言模型（LLM）的创造力以及人类评分者的上下文编辑示例，提供了更广泛的编辑指令范围；2）其数据源基于真实图像，包括照片和艺术作品，与仅由文本到图像模型生成的数据集相比，提供了更大的多样性和减少的偏见；3）它还支持基于区域的编辑，并通过高质量、自动生成的区域注释得到增强。我们的实验表明，在UltraEdit上训练的规范基于扩散的编辑基线在MagicBrush和Emu-Edit基准测试中创下了新纪录。我们的分析进一步证实了真实图像锚点和基于区域的编辑数据的关键作用。数据集、代码和模型可在<a href="https://ultra-editing.github.io" rel="external noopener nofollow" class="link-external link-https">此https URL</a>找到。
</details>

<details>
    <summary>Key points</summary>
    * 利用LLM和人类示例生成多样指令
    * 使用真实图像以提高多样性和减少偏见
    * 包含基于区域的编辑与自动注释
</details>
</details>

---


<details>
<summary><b> EditScribe: Non-Visual Image Editing with Natural Language Verification Loops</b></summary>

* **Authors:** Ruei-Che Chang, Yuxuan Liu, Lotus Zhang, Anhong Guo
* **arXiv ID:** 2408.06632
* **One-liner:** Developed an accessible image editing system for blind and low-vision users.
* **Published in:** arxiv (13 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.06632) | [[PDF]](https://arxiv.org/pdf/2408.06632) | [[Code]]()

> **核心创新**
> 引入了自然语言验证循环用于非视觉图像编辑和反馈。

<details>
    <summary>Abstract</summary>
    图像编辑是一个迭代过程，需要精确的视觉评估和操作，以使输出与编辑意图匹配。然而，当前图像编辑工具未提供盲人和低视力个体可访问的交互或足够的反馈以实现这种控制水平。为此，我们开发了EditScribe，一个原型系统，利用大型多模态模型驱动的自然语言验证循环使图像编辑可访问。使用EditScribe，用户首先通过初始一般和对象描述理解图像内容，然后使用开放式自然语言提示指定编辑动作。EditScribe执行图像编辑，并提供四种类型的验证反馈供用户验证执行的编辑，包括视觉变化摘要、AI判断以及更新的通用和对象描述。用户可以提出后续问题以澄清和探究编辑或验证反馈，然后执行另一个编辑。在一项针对十名盲人或低视力用户的研究中，我们发现EditScribe支持参与者非视觉地执行和验证图像编辑动作。我们观察到参与者的不同提示策略，以及他们对各种类型验证反馈的看法。最后，我们讨论了利用自然语言验证循环使视觉创作非视觉可访问的启示。
</details>

<details>
    <summary>Key points</summary>
    * 使用初始描述进行图像理解
    * 提供四种类型的验证反馈
    * 允许后续问题以澄清
</details>
</details>

---


<details>
<summary><b> FastEdit: Fast Text-Guided Single-Image Editing via Semantic-Aware Diffusion Fine-Tuning</b></summary>

* **Authors:** Zhi Chen, Zecheng Zhao, Yadan Luo, Zi Huang
* **arXiv ID:** 2408.03355
* **One-liner:** Accelerated text-guided single-image editing to 17 seconds.
* **Published in:** arxiv (6 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.03355) | [[PDF]](https://arxiv.org/pdf/2408.03355) | [[Code]]()

> **核心创新**
> 简化了微调并使用语义感知扩散实现更快处理。

<details>
    <summary>Abstract</summary>
    传统的文本引导单图像编辑方法需要一个两步过程，包括对目标文本嵌入进行超过1000次迭代的微调，以及对生成模型进行另外1500次迭代的微调。尽管这确保了结果图像与输入图像和目标文本紧密对齐，但该过程通常每张图像需要7分钟，由于其耗时性，对实际应用构成挑战。为解决这一瓶颈，我们引入了FastEdit，一种快速的文本引导单图像编辑方法，采用语义感知扩散微调，将编辑过程显著加速至仅17秒。FastEdit简化了生成模型的微调阶段，将其从1500次迭代减少到仅50次。对于扩散微调，我们根据输入图像和目标文本之间的语义差异采用特定时间步值。此外，FastEdit通过使用基于特征空间而非文本嵌入空间的图像到图像模型，绕过了初始微调步骤。它能有效在相同特征空间中对齐目标文本提示和输入图像，并节省大量处理时间。另外，我们将参数高效微调技术LoRA应用于U-net。通过LoRA，FastEdit将模型的可训练参数最小化至原始大小的0.37%。同时，我们可以在显著减少计算开销的情况下实现可比较的编辑结果。我们进行了广泛实验以验证我们方法的编辑性能，并展示了有前景的编辑能力，包括内容添加、风格转移、背景替换和姿态操纵等。
</details>

<details>
    <summary>Key points</summary>
    * 将微调迭代从1500次减少到50次
    * 基于语义差异采用时间步
    * 应用LoRA以提高参数效率
</details>
</details>

---


<details>
<summary><b> TurboEdit: Instant text-based image editing</b></summary>

* **Authors:** Zongze Wu, Nicholas Kolkin, Jonathan Brandt, Richard Zhang, Eli Shechtman
* **arXiv ID:** 2408.08332
* **One-liner:** Enabled precise image inversion and disentangled editing in few-step diffusion models.
* **Published in:** arxiv (14 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.08332) | [[PDF]](https://arxiv.org/pdf/2408.08332) | [[Code]]()

> **核心创新**
> 使用基于编码器的迭代反演和文本条件实现实时编辑。

<details>
    <summary>Abstract</summary>
    我们解决了在少步扩散模型中精确图像反演和解缠图像编辑的挑战。我们引入了一种基于编码器的迭代反演技术。该反演网络以输入图像和上一步重建图像为条件，允许对下一次重建进行校正以朝向输入图像。我们证明，通过以（自动生成的）详细文本提示为条件，可以在少步扩散模型中轻松实现解缠控制。为了操作反演图像，我们冻结噪声图并修改文本提示中的一个属性（手动或通过LLM驱动的基于指令的编辑），从而生成一个新图像，类似于输入图像但仅改变一个属性。它还能控制编辑强度并接受指导性文本提示。我们的方法促进了实时文本引导图像编辑，反演仅需8次功能评估（NFE）（一次性成本），每次编辑仅需4次NFE。我们的方法不仅快速，而且在性能上显著优于最先进的多步扩散编辑技术。
</details>

<details>
    <summary>Key points</summary>
    * 引入带校正的迭代反演
    * 冻结噪声图以实现属性变更
    * 以低NFE实现编辑（反演8次，每次编辑4次）
</details>
</details>

---


<details>
<summary><b> ReEdit: Multimodal Exemplar-Based Image Editing with Diffusion Models</b></summary>

* **Authors:** Ashutosh Srivastava, Tarun Ram Menta, Abhinav Java, Avadhoot Jadhav, Silky Singh, Surgan Jandial, Balaji Krishnamurthy
* **arXiv ID:** 2411.03982
* **One-liner:** Proposed an efficient exemplar-based image editing framework.
* **Published in:** arxiv (6 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.03982) | [[PDF]](https://arxiv.org/pdf/2411.03982) | [[Code]]()

> **核心创新**
> 在文本和图像模态中捕获编辑以实现高保真度和速度。

<details>
    <summary>Abstract</summary>
    现代文本到图像（T2I）扩散模型通过生成高质量逼真图像，彻底改变了图像编辑。虽然使用T2I模型执行编辑的默认方法是通过文本指令，但由于自然语言与图像之间复杂的多对多映射，这种方法非平凡。在这项工作中，我们处理基于示例的图像编辑——将编辑从示例对转移到内容图像的任务。我们提出了ReEdit，一个模块化且高效的端到端框架，在文本和图像模态中捕获编辑，同时确保编辑图像的保真度。我们通过与最先进基线的广泛比较和关键设计选择的敏感性分析，验证了ReEdit的有效性。我们的结果表明，ReEdit在定性和定量上均一致优于当代方法。此外，ReEdit具有高实际适用性，因为它不需要任何任务特定优化，并且比次优基线快四倍。
</details>

<details>
    <summary>Key points</summary>
    * 模块化端到端框架
    * 无需任务特定优化
    * 比基线快四倍
</details>
</details>

---


<details>
<summary><b> Multi-Reward as Condition for Instruction-based Image Editing</b></summary>

* **Authors:** Xin Gu, Ming Li, Libo Zhang, Fan Chen, Longyin Wen, Tiejian Luo, Sijie Zhu
* **arXiv ID:** 2411.04713
* **One-liner:** Improved training data quality with multi-perspective reward data.
* **Published in:** arxiv (6 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.04713) | [[PDF]](https://arxiv.org/pdf/2411.04713) | [[Code]](https://github.com/bytedance/Multi-Reward-Editing)

> **核心创新**
> 设计了一个指标系统和训练框架以改进编辑模型。

<details>
    <summary>Abstract</summary>
    高质量训练三元组（指令、原始图像、编辑图像）对于基于指令的图像编辑至关重要。主流训练数据集（如InsPix2Pix）是使用文本到图像生成模型（如Stable Diffusion、DALL-E）创建的，这些模型未针对图像编辑进行训练。因此，这些数据集存在指令遵循不准确、细节保留差和生成伪影的问题。在本文中，我们提出通过多视角奖励数据解决训练数据质量问题，而不是改进地面真实图像质量。1）我们首先设计了一个基于最佳大型视觉语言模型（LVLM）的定量指标系统，即GPT-4o，从三个角度评估生成质量：指令遵循、细节保留和生成质量。对于每个角度，我们收集了0到5的定量分数和关于地面真实编辑图像中特定失败点的文本描述反馈，从而产生了一个高质量编辑奖励数据集，即RewardEdit20K。2）我们进一步提出了一种新颖的训练框架，将指标输出（视为多奖励）无缝集成到编辑模型中，以从不完美的训练三元组中学习。在训练期间，奖励分数和文本描述被编码为嵌入，并作为辅助条件馈送到编辑模型的潜在空间和U-Net中。3）我们还构建了一个具有真实世界图像/照片和多样编辑指令的具有挑战性的评估基准，名为Real-Edit。实验表明，我们的多奖励条件模型在两种流行编辑流水线（即InsPix2Pix和SmartEdit）上优于其无奖励对应模型。代码发布于<a href="https://github.com/bytedance/Multi-Reward-Editing" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 使用LVLM从三个角度进行评估
    * 将奖励作为嵌入集成到模型中
    * 构建了具有挑战性的基准（Real-Edit）
</details>
</details>

---


<details>
<summary><b> OmniEdit: Building Image Editing Generalist Models Through Specialist Supervision</b></summary>

* **Authors:** Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, Wenhu Chen
* **arXiv ID:** 2411.07199
* **One-liner:** Developed an omnipotent editor for multiple tasks and aspect ratios.
* **Published in:** arxiv (11 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.07199) | [[PDF]](https://arxiv.org/pdf/2411.07199) | [[Code]](https://github.com/TIGER-AI-Lab/OmniEdit)

> **核心创新**
> 利用专家模型监督并改进了数据质量。

<details>
    <summary>Abstract</summary>
    指令引导图像编辑方法通过在自动合成或手动标注的图像编辑对上训练扩散模型，已显示出巨大潜力。然而，这些方法距离实际、现实生活应用仍很远。我们识别了导致这一差距的三个主要挑战。首先，现有模型由于有偏的合成过程而编辑技能有限。其次，这些方法使用具有高噪声和伪影的数据集进行训练。这是由于应用了简单过滤方法如CLIP分数。第三，所有这些数据集都限制在单一低分辨率和固定宽高比，限制了处理现实世界用例的多样性。在本文中，我们提出了\omniedit，它是一个全能编辑器，能够无缝处理七种不同图像编辑任务和任何宽高比。我们的贡献有四个方面：（1）\omniedit通过利用七个不同专家模型的监督进行训练，以确保任务覆盖。（2）我们使用基于大型多模态模型（如GPT-4o）提供的分数的重要性采样，而不是CLIP分数，以提高数据质量。（3）我们提出了一种新的编辑架构EditNet，以大幅提升编辑成功率。（4）我们提供不同宽高比的图像，以确保我们的模型可以处理野外任何图像。我们策划了一个测试集，包含不同宽高比的图像，并配有覆盖不同任务的多样指令。自动评估和人工评估均表明，\omniedit可以显著优于所有现有模型。我们的代码、数据集和模型将在<a href="https://tiger-ai-lab.github.io/OmniEdit/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>提供。
</details>

<details>
    <summary>Key points</summary>
    * 使用七个专家模型进行训练
    * 使用LMM分数进行重要性采样
    * 提出EditNet架构以提高成功率
</details>
</details>

---


<details>
<summary><b> AnyEdit: Mastering Unified High-Quality Image Editing for Any Idea</b></summary>

* **Authors:** Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, Yueting Zhuang
* **arXiv ID:** 2411.15738
* **One-liner:** Created a comprehensive multi-modal instruction editing dataset.
* **Published in:** arxiv (24 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.15738) | [[PDF]](https://arxiv.org/pdf/2411.15738) | [[Code]](https://github.com/DCDmllm/AnyEdit)

> **核心创新**
> 确保了跨编辑类型和领域的多样性和质量。

<details>
    <summary>Abstract</summary>
    基于指令的图像编辑旨在用自然语言指令修改特定图像元素。然而，该领域的当前模型通常难以准确执行复杂用户指令，因为它们是在低质量数据和有限编辑类型上训练的。我们提出了AnyEdit，一个全面的多模态指令编辑数据集，包含250万高质量编辑对，涵盖超过20种编辑类型和五个领域。我们通过三个方面确保AnyEdit集合的多样性和质量：初始数据多样性、自适应编辑过程和编辑结果的自动选择。使用该数据集，我们进一步训练了一个新颖的AnyEdit Stable Diffusion，具有任务感知路由和可学习任务嵌入，用于统一图像编辑。在三个基准数据集上的综合实验表明，AnyEdit一致提升了基于扩散的编辑模型的性能。这为开发支持人类创造力的指令驱动图像编辑模型提供了前景。
</details>

<details>
    <summary>Key points</summary>
    * 收集了250万高质量对
    * 使用自适应编辑和自动选择
    * 训练模型具有任务感知路由
</details>
</details>

---


<details>
<summary><b> TPIE: Topology-Preserved Image Editing With Text Instructions</b></summary>

* **Authors:** Nivetha Jayakumar, Srivardhan Reddy Gadila, Tonmoy Hossain, Yangfeng Ji, Miaomiao Zhang
* **arXiv ID:** 2411.16714
* **One-liner:** Introduced topology-preserved image editing for sensitive domains.
* **Published in:** arxiv (22 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.16714) | [[PDF]](https://arxiv.org/pdf/2411.16714) | [[Code]]()

> **核心创新**
> 通过可变形变体和扩散模型确保几何保持完整。

<details>
    <summary>Abstract</summary>
    在现实世界应用中，特别是在医疗和医学等敏感领域，保持拓扑结构很重要，其中人体解剖的正确性至关重要。然而，大多数现有图像编辑模型专注于操作强度和纹理特征，常常忽略图像中的对象几何。为解决这一问题，本文引入了一种新方法，拓扑保持图像编辑（TPIE），首次通过文本引导生成扩散模型确保编辑图像中的拓扑和几何保持完整。更具体地说，我们的方法将新生成的样本视为给定输入模板的可变形变体，允许可控和结构保持的编辑。我们提出的TPIE框架包括两个关键模块：（i）一个基于自动编码器的配准网络，从成对训练图像中学习对象变换的潜在表示，参数化为速度场；（ii）一个新颖的潜在条件几何扩散（LCDG）模型，高效捕获学习到的变换特征的数据分布，条件于自定义文本指令。我们在多样2D和3D图像上验证TPIE，并将其与最先进图像编辑方法进行比较。实验结果表明，我们的方法在生成更逼真且拓扑保持良好的图像方面优于其他基线。我们的代码将在Github上公开提供。
</details>

<details>
    <summary>Key points</summary>
    * 基于自动编码器的配准网络
    * 潜在条件几何扩散模型
    * 在2D和3D图像上验证
</details>
</details>

---


<details>
<summary><b> InsightEdit: Towards Better Instruction Following for Image Editing</b></summary>

* **Authors:** Yingjing Xu, Jie Kong, Jiazhi Wang, Xiao Pan, Bo Lin, Qiang Liu
* **arXiv ID:** 2411.17323
* **One-liner:** Enhanced instruction-based editing with better dataset and feature utilization.
* **Published in:** arxiv (26 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.17323) | [[PDF]](https://arxiv.org/pdf/2411.17323) | [[Code]](https://github.com/poppyxu/InsightEdit)

> **核心创新**
> 策划了高质量数据集并使用双流桥接实现精确指导。

<details>
    <summary>Abstract</summary>
    在本文中，我们专注于基于指令的图像编辑任务。先前工作如InstructPix2Pix、InstructDiffusion和SmartEdit已探索端到端编辑。然而，两个局限性仍然存在：首先，现有数据集分辨率低、背景一致性差且指令过于简单。其次，当前方法主要条件于文本，而丰富的图像信息未被充分探索，因此在复杂指令遵循和保持背景一致性方面表现较差。针对这些问题，我们首先使用新颖的数据构建流程策划了AdvancedEdit数据集，形成了一个大规模、高视觉质量、复杂指令和良好背景一致性的数据集。然后，为了进一步注入丰富图像信息，我们引入了一种双流桥接机制，利用强大多模态大语言模型（MLLM）推理的文本和视觉特征，以更精确地指导图像编辑过程。广泛结果表明，我们的方法InsightEdit实现了最先进性能，在复杂指令遵循和保持与原始图像的高背景一致性方面表现出色。
</details>

<details>
    <summary>Key points</summary>
    * 具有复杂指令的AdvancedEdit数据集
    * 使用文本和视觉特征的双流机制
    * 改进了背景一致性和指令遵循
</details>
</details>

---


<details>
<summary><b> UIP2P: Unsupervised Instruction-based Image Editing via Edit Reversibility Constraint</b></summary>

* **Authors:** Enis Simsar, Alessio Tonioni, Yongqin Xian, Thomas Hofmann, Federico Tombari
* **arXiv ID:** 2412.15216
* **One-liner:** Proposed an unsupervised instruction-based image editing approach eliminating the need for ground-truth edited images during training.
* **Published in:** arxiv (19 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.15216) | [[PDF]](https://arxiv.org/pdf/2412.15216) | [[Code]]()

> **核心创新**
> 引入了编辑可逆性约束（ERC）来应用正向和反向编辑，使得能够在没有真实编辑图像的数据集上进行训练。

<details>
    <summary>Abstract</summary>
    我们提出了一种无监督的基于指令的图像编辑方法，无需在训练期间使用真实编辑图像。
</details>

<details>
    <summary>Key points</summary>
    * 编辑可逆性约束（ERC）用于正向和反向编辑
    * 在图像、文本和注意力空间中的对齐强化
    * 在无真实编辑图像的数据集上进行训练
</details>
</details>

---


<details>
<summary><b> Textualize Visual Prompt for Image Editing via Diffusion Bridge</b></summary>

* **Authors:** Pengcheng Xu, Qingnan Fan, Fei Kou, Shuai Qin, Hong Gu, Ruoyu Zhao, Charles Ling, Boyu Wang
* **arXiv ID:** 2501.03495
* **One-liner:** Developed a framework for visual prompt-based image editing using any single text-to-image model without relying on an explicit image-to-image model.
* **Published in:** arxiv (7 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.03495) | [[PDF]](https://arxiv.org/pdf/2501.03495) | [[Code]]()

> **核心创新**
> 利用概率流常微分方程构建扩散桥进行分布转移，并通过差分注意力控制优化文本嵌入。

<details>
    <summary>Abstract</summary>
    视觉提示，即一对前后编辑图像，可以传达难以描述的图像变换，并在图像编辑中蓬勃发展。
</details>

<details>
    <summary>Key points</summary>
    * 使用概率流常微分方程构建扩散桥
    * 文本优化以实现嵌入文本化
    * 差分注意力控制用于变换解耦
</details>
</details>

---


<details>
<summary><b> Hands-off Image Editing: Language-guided Editing without any Task-specific Labeling, Masking or even Training</b></summary>

* **Authors:** Rodrigo Santos, António Branco, João Silva, João Rodrigues
* **arXiv ID:** 2502.10064
* **One-liner:** Proposed a novel unsupervised approach for instruction-guided image editing without task-specific supervision.
* **Published in:** arxiv (14 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.10064) | [[PDF]](https://arxiv.org/pdf/2502.10064) | [[Code]]()

> **核心创新**
> 通过消除任务特定标注、掩码或训练的需求，实现了竞争性性能。

<details>
    <summary>Abstract</summary>
    指令引导的图像编辑涉及获取图像和指令，并根据该指令交付修改后的图像。
</details>

<details>
    <summary>Key points</summary>
    * 无任务特定监督的无监督学习
    * 有效处理指令引导的编辑
    * 竞争性性能评估
</details>
</details>

---


<details>
<summary><b> PromptArtisan: Multi-instruction Image Editing in Single Pass with Complete Attention Control</b></summary>

* **Authors:** Kunal Swami, Raghu Chittersu, Pranav Adlinge, Rajeev Irny, Shashavali Doodekula, Alok Shukla
* **arXiv ID:** 2502.10258
* **One-liner:** Introduced PromptArtisan for multi-instruction image editing in a single pass without iterative refinement.
* **Published in:** arxiv (14 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.10258) | [[PDF]](https://arxiv.org/pdf/2502.10258) | [[Code]]()

> **核心创新**
> 将预训练的InstructPix2Pix模型与完整注意力控制机制（CACM）集成，实现精确的零样本编辑。

<details>
    <summary>Abstract</summary>
    我们提出了PromptArtisan，一种突破性的多指令图像编辑方法，在单次处理中实现显著结果，无需耗时迭代优化。
</details>

<details>
    <summary>Key points</summary>
    * 多指令编辑与掩码关联
    * 完整注意力控制机制（CACM）
    * 零样本、单次处理效率
</details>
</details>

---


<details>
<summary><b> Instruct-CLIP: Improving Instruction-Guided Image Editing with Automated Data Refinement Using Contrastive Learning</b></summary>

* **Authors:** Sherry X. Chen, Misha Sra, Pradeep Sen
* **arXiv ID:** 2503.18406
* **One-liner:** Presented Instruct-CLIP, a self-supervised method to refine instruction-image alignment in datasets for training latent diffusion models.
* **Published in:** arxiv (24 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.18406) | [[PDF]](https://arxiv.org/pdf/2503.18406) | [[Code]](https://github.com/SherryXTChen/Instruct-CLIP.git)

> **核心创新**
> 将Instruct-CLIP适配以处理噪声潜在图像和扩散时间步，实现在潜在空间中的对齐强化。

<details>
    <summary>Abstract</summary>
    尽管自然语言指令提供了一种直观的方式来指导自动化图像编辑，但深度学习模型往往难以实现高质量结果，主要由于创建大规模高质量训练数据集的困难。
</details>

<details>
    <summary>Key points</summary>
    * 自监督学习用于语义变化精炼
    * 处理噪声潜在图像和扩散时间步
    * 基于I-CLIP的损失函数用于模型微调
</details>
</details>

---


<details>
<summary><b> Tuning-Free Image Editing with Fidelity and Editability via Unified Latent Diffusion Model</b></summary>

* **Authors:** Qi Mao, Lan Chen, Yuchao Gu, Mike Zheng Shou, Ming-Hsuan Yang
* **arXiv ID:** 2504.05594
* **One-liner:** Introduced UnifyEdit, a tuning-free method for balancing fidelity and editability in text-based image editing.
* **Published in:** arxiv (8 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.05594) | [[PDF]](https://arxiv.org/pdf/2504.05594) | [[Code]](https://github.com/CUC-MIPG/UnifyEdit)

> **核心创新**
> 开发了自注意力和交叉注意力约束，并引入自适应时间步调度器以防止梯度冲突。

<details>
    <summary>Abstract</summary>
    在基于文本的图像编辑（TIE）中，平衡保真度和可编辑性至关重要，失败通常导致过度或不足编辑问题。
</details>

<details>
    <summary>Key points</summary>
    * 自注意力保留约束用于保真度
    * 交叉注意力对齐约束用于可编辑性
    * 自适应时间步调度器用于平衡优化
</details>
</details>

---


<details>
<summary><b> Omni$^2$: Unifying Omnidirectional Image Generation and Editing in an Omni Model</b></summary>

* **Authors:** Liu Yang, Huiyu Duan, Yucheng Zhu, Xiaohong Liu, Lu Liu, Zitong Xu, Guangji Ma, Xiongkuo Min, Guangtao Zhai, Patrick Le Callet
* **arXiv ID:** 2504.11379
* **One-liner:** Constructed Any2Omni, the first comprehensive dataset for omnidirectional image generation and editing, and proposed the Omni² model.
* **Published in:** arxiv (15 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.11379) | [[PDF]](https://arxiv.org/pdf/2504.11379) | [[Code]](https://github.com/IntMeGroup/Omni2)

> **核心创新**
> 使得能够使用单一模型处理各种ODI任务，适应多样输入条件。

<details>
    <summary>Abstract</summary>
    360°全向图像（ODI）近年来受到广泛关注，并广泛应用于各种虚拟现实（VR）和增强现实（AR）应用中。
</details>

<details>
    <summary>Key points</summary>
    * Any2Omni数据集包含60,000+训练样本
    * Omni²模型用于多任务ODI生成和编辑
    * 支持多样输入条件
</details>
</details>

---


<details>
<summary><b> X-Edit: Detecting and Localizing Edits in Images Altered by Text-Guided Diffusion Models</b></summary>

* **Authors:** Valentina Bazyleva, Nicolo Bonettini, Gaurav Bharaj
* **arXiv ID:** 2505.11753
* **One-liner:** Introduced X-Edit, a method for localizing diffusion-based edits in images to detect deepfake manipulations.
* **Published in:** arxiv (16 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.11753) | [[PDF]](https://arxiv.org/pdf/2505.11753) | [[Code]]()

> **核心创新**
> 使用预训练扩散模型进行图像反转，结合分割网络和组合分割与相关性损失，实现精确掩码预测。

<details>
    <summary>Abstract</summary>
    文本引导扩散模型显著推进了图像编辑，使得能够基于文本提示进行高度真实和局部修改。
</details>

<details>
    <summary>Key points</summary>
    * 使用预训练扩散模型进行图像反转
    * 具有通道和空间注意力的分割网络
    * 组合分割与相关性损失用于定位
</details>
</details>

---


<details>
<summary><b> Step1X-Edit: A Practical Framework for General Image Editing</b></summary>

* **Authors:** Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, Daxin Jiang
* **arXiv ID:** 2504.17761
* **One-liner:** Released Step1X-Edit, an open-source image editing model achieving performance comparable to closed-source models like GPT-4o.
* **Published in:** arxiv (24 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.17761) | [[PDF]](https://arxiv.org/pdf/2504.17761) | [[Code]](https://github.com/stepfun-ai/Step1X-Edit)

> **核心创新**
> 集成多模态LLM用于指令处理和扩散解码器，在高质量数据集上训练，并使用GEdit-Bench进行评估。

<details>
    <summary>Abstract</summary>
    近年来，图像编辑模型见证了显著且快速的发展。
</details>

<details>
    <summary>Key points</summary>
    * 多模态LLM用于指令和图像处理
    * 高质量数据集生成流程
    * GEdit-Bench用于真实世界评估
</details>
</details>

---


<details>
<summary><b> FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space</b></summary>

* **Authors:** Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, Luke Smith
* **arXiv ID:** 2506.15742
* **One-liner:** Evaluated FLUX.1 Kontext, a unified generative flow matching model for image generation and editing with improved consistency.
* **Published in:** arxiv (17 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.15742) | [[PDF]](https://arxiv.org/pdf/2506.15742) | [[Code]](https://github.com/black-forest-labs/flux)

> **核心创新**
> 实现了竞争性性能和更快的生成时间，在KontextBench上验证了多任务性能。

<details>
    <summary>Abstract</summary>
    我们展示了FLUX.1 Kontext的评估结果，这是一种生成流匹配模型，统一了图像生成和编辑。
</details>

<details>
    <summary>Key points</summary>
    * 统一架构用于生成和编辑
    * 序列连接用于上下文处理
    * KontextBench用于全面评估
</details>
</details>

---


<details>
<summary><b> Towards Efficient Exemplar Based Image Editing with Multimodal VLMs</b></summary>

* **Authors:** Avadhoot Jadhav, Ashutosh Srivastava, Abhinav Java, Silky Singh, Tarun Ram Menta, Surgan Jandial, Balaji Krishnamurthy
* **arXiv ID:** 2506.20155
* **One-liner:** Enabled exemplar-based image editing using pretrained models without optimization.
* **Published in:** arxiv (25 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.20155) | [[PDF]](https://arxiv.org/pdf/2506.20155) | [[Code]]()

> **核心创新**
> 利用预训练的文本到图像扩散模型和多模态视觉语言模型，将编辑从示例对转移到内容图像。

<details>
    <summary>Abstract</summary>
    文本到图像扩散模型已实现广泛的图像编辑应用，但仅通过文本捕捉所有类型的编辑可能具有挑战性且繁琐。某些图像编辑的模糊性通过示例对（即分别描绘编辑前和编辑后图像的一对图像）能更好地表达。在本工作中，我们处理基于示例的图像编辑——利用预训练的文本到图像扩散模型和多模态视觉语言模型，将编辑从示例对转移到内容图像的任务。尽管我们的端到端流程无需优化，但实验表明，它在多种编辑类型上仍优于基线方法，同时速度提高约4倍。
</details>

<details>
    <summary>Key points</summary>
    * 使用示例对处理模糊编辑
    * 无需优化的流程
    * 多模态视觉语言模型集成
</details>
</details>

---


<details>
<summary><b> Reasoning to Edit: Hypothetical Instruction-Based Image Editing with Visual Reasoning</b></summary>

* **Authors:** Qingdong He, Xueqin Chen, Chaoyi Wang, Yanjie Pan, Xiaobin Hu, Zhenye Gan, Yabiao Wang, Chengjie Wang, Xiangtai Li, Jiangning Zhang
* **arXiv ID:** 2507.01908
* **One-liner:** Introduced a dataset and framework for reasoning-aware image editing.
* **Published in:** arxiv (2 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.01908) | [[PDF]](https://arxiv.org/pdf/2507.01908) | [[Code]](https://github.com/hithqd/ReasonBrain)

> **核心创新**
> 提出了Reason50K数据集和ReasonBrain框架，以处理具有细粒度推理的隐式假设指令。

<details>
    <summary>Abstract</summary>
    基于指令的图像编辑（IIE）随着扩散模型的成功而迅速发展。然而，现有工作主要关注简单和显式的指令来执行编辑操作，如添加、删除、移动或交换对象。它们难以处理更复杂的隐式假设指令，这些指令需要更深层次的推理来推断合理的视觉变化和用户意图。此外，当前数据集对训练和评估推理感知编辑能力的支持有限。在架构上，这些方法也缺乏支持此类推理的细粒度细节提取机制。为了解决这些限制，我们提出了Reason50K，一个专门为训练和评估假设指令推理图像编辑而策划的大规模数据集，以及ReasonBrain，一个设计用于在多样化场景中推理和执行隐式假设指令的新框架。Reason50K包含超过50K个样本，涵盖四个关键推理场景：物理、时间、因果和故事推理。ReasonBrain利用多模态大语言模型（MLLMs）生成编辑指导，并使用扩散模型进行图像合成，结合细粒度推理线索提取（FRCE）模块来捕获支持指令推理所需的详细视觉和文本语义。为了减轻语义损失，我们进一步引入了跨模态增强器（CME），使细粒度线索和MLLM衍生特征之间能够进行丰富交互。广泛实验表明，ReasonBrain在推理场景上始终优于最先进的基线方法，同时在传统IIE任务上表现出强大的零样本泛化能力。我们的数据集和代码将公开发布。
</details>

<details>
    <summary>Key points</summary>
    * Reason50K数据集包含50K样本
    * 细粒度推理线索提取（FRCE）
    * 跨模态增强器（CME）
</details>
</details>

---


<details>
<summary><b> Beyond Simple Edits: X-Planner for Complex Instruction-Based Image Editing</b></summary>

* **Authors:** Chun-Hsiao Yeh, Yilin Wang, Nanxuan Zhao, Richard Zhang, Yuheng Li, Yi Ma, Krishna Kumar Singh
* **arXiv ID:** 2507.05259
* **One-liner:** Developed an MLLM-based planning system for complex instruction decomposition.
* **Published in:** arxiv (7 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.05259) | [[PDF]](https://arxiv.org/pdf/2507.05259) | [[Code]](https://danielchyeh.github.io/x-planner/)

> **核心创新**
> 引入了X-Planner，将指令分解为子指令，并自动生成掩码以实现身份保持的编辑。

<details>
    <summary>Abstract</summary>
    最近基于扩散的图像编辑方法在文本引导任务上取得了显著进展，但往往难以解释复杂的间接指令。此外，当前模型经常遭受身份保持不佳、意外编辑或严重依赖手动掩码的问题。为了解决这些挑战，我们引入了X-Planner，一个基于多模态大语言模型（MLLM）的规划系统，有效桥接用户意图与编辑模型能力。X-Planner采用思维链推理，系统地将复杂指令分解为更简单、清晰的子指令。对于每个子指令，X-Planner自动生成精确的编辑类型和分割掩码，消除手动干预并确保局部化、身份保持的编辑。此外，我们提出了一种新颖的自动化流程，用于生成大规模数据来训练X-Planner，该流程在现有基准和我们新引入的复杂编辑基准上均达到了最先进的结果。
</details>

<details>
    <summary>Key points</summary>
    * 思维链推理
    * 自动掩码生成
    * 自动化数据生成流程
</details>
</details>

---


<details>
<summary><b> NoHumansRequired: Autonomous High-Quality Image Editing Triplet Mining</b></summary>

* **Authors:** Maksim Kuprashevich, Grigorii Alekseenko, Irina Tolstykh, Georgii Fedorov, Bulat Suleimanov, Vladimir Dokholyan, Aleksandr Gordeev
* **arXiv ID:** 2507.14119
* **One-liner:** Automated the creation of high-quality training data for image editing.
* **Published in:** arxiv (18 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.14119) | [[PDF]](https://arxiv.org/pdf/2507.14119) | [[Code]]()

> **核心创新**
> 提出了一个使用生成模型和Gemini验证器挖掘和验证三元组的流程，并发布了NHR-Edit数据集。

<details>
    <summary>Abstract</summary>
    生成建模的最新进展使得图像编辑助手能够遵循自然语言指令而无需额外用户输入。它们的监督训练需要数百万个三元组（原始图像、指令、编辑后图像），但挖掘像素级准确的示例很困难。每个编辑必须仅影响提示指定的区域、保持风格一致性、尊重物理合理性并保留视觉吸引力。缺乏鲁棒的自动化编辑质量指标阻碍了大规模可靠自动化。我们提出了一个自动化、模块化的流程，用于跨领域、分辨率、指令复杂性和风格挖掘高保真三元组。该系统基于公共生成模型运行，无需人工干预，使用任务调整的Gemini验证器直接评分指令遵循性和美学，消除了对分割或基础模型的需求。反转和组合引导将挖掘集扩大了约2.6倍，实现了大规模高保真训练数据。通过自动化最重复的标注步骤，该方法允许在没有人工标注努力的情况下进行新规模的训练。为了在这个资源密集型领域民主化研究，我们发布了NHR-Edit，一个包含720K高质量三元组的开放数据集，通过数百万次引导生成和验证器传递进行工业规模策划，并分析了流程的阶段生存率，提供了一个估计不同模型堆栈计算努力的框架。在最大的跨数据集评估中，它超越了所有公共替代方案。我们还发布了Bagel-NHR-Edit，一个具有最先进指标的微调Bagel模型。
</details>

<details>
    <summary>Key points</summary>
    * 自动化三元组挖掘
    * Gemini验证器用于评分
    * 反转和引导用于数据扩展
</details>
</details>

---


<details>
<summary><b> Qwen-Image Technical Report</b></summary>

* **Authors:** Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, Zenan Liu
* **arXiv ID:** 2508.02324
* **One-liner:** Advanced text rendering and editing consistency in image generation.
* **Published in:** arxiv (4 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.02324) | [[PDF]](https://arxiv.org/pdf/2508.02324) | [[Code]](https://github.com/QwenLM/Qwen-Image)

> **核心创新**
> 改进了Qwen-Image，采用渐进式训练和双重编码机制，以实现更好的文本渲染和编辑保真度。

<details>
    <summary>Abstract</summary>
    我们提出了Qwen-Image，Qwen系列中的图像生成基础模型，在复杂文本渲染和精确图像编辑方面取得了显著进展。为了解决复杂文本渲染的挑战，我们设计了一个全面的数据流程，包括大规模数据收集、过滤、标注、合成和平衡。此外，我们采用了一种渐进式训练策略，从非文本到文本渲染开始，从简单到复杂的文本输入演进，并逐渐扩展到段落级描述。这种课程学习方法显著增强了模型的本地文本渲染能力。因此，Qwen-Image不仅在英语等字母语言中表现优异，还在更具挑战性的象形文字如中文上取得了显著进展。为了增强图像编辑一致性，我们引入了一种改进的多任务训练范式，不仅包括传统的文本到图像（T2I）和文本图像到图像（TI2I）任务，还包括图像到图像（I2I）重建，有效对齐了Qwen2.5-VL和MMDiT之间的潜在表示。此外，我们分别将原始图像输入Qwen2.5-VL和VAE编码器，以获得语义和重建表示。这种双重编码机制使编辑模块能够在保持语义一致性和视觉保真度之间取得平衡。Qwen-Image在多个基准测试中实现了最先进的性能，展示了其在图像生成和编辑方面的强大能力。
</details>

<details>
    <summary>Key points</summary>
    * 渐进式训练策略
    * 双重编码机制
    * 多任务训练范式
</details>
</details>

---


<details>
<summary><b> Talk2Image: A Multi-Agent System for Multi-Turn Image Generation and Editing</b></summary>

* **Authors:** Shichao Ma, Yunhe Guo, Jiahao Su, Qihe Huang, Zhengyang Zhou, Yang Wang
* **arXiv ID:** 2508.06916
* **One-liner:** Enabled interactive multi-turn image editing with a multi-agent system.
* **Published in:** arxiv (9 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.06916) | [[PDF]](https://arxiv.org/pdf/2508.06916) | [[Code]]()

> **核心创新**
> 引入了Talk2Image，用于基于对话的编辑，具有意图解析和协作代理。

<details>
    <summary>Abstract</summary>
    文本到图像生成任务推动了多样化媒体应用的显著进展，但大多数关注单轮场景，难以处理迭代、多轮创意任务。最近的基于对话的系统试图弥合这一差距，但其单代理、顺序范式经常导致意图漂移和不连贯编辑。为了解决这些限制，我们提出了Talk2Image，一个用于多轮对话场景中交互式图像生成和编辑的新颖多代理系统。我们的方法集成了三个关键组件：从对话历史中解析意图、跨专业代理的任务分解和协作执行，以及基于多视图评估机制的反馈驱动细化。Talk2Image实现了与用户意图的逐步对齐和一致的图像编辑。实验表明，Talk2Image在迭代图像生成和编辑任务中，在可控性、连贯性和用户满意度方面优于现有基线。
</details>

<details>
    <summary>Key points</summary>
    * 多代理系统
    * 从对话中解析意图
    * 反馈驱动细化
</details>
</details>

---


<details>
<summary><b> CannyEdit: Selective Canny Control and Dual-Prompt Guidance for Training-Free Image Editing</b></summary>

* **Authors:** Weiyan Xie, Han Gao, Didan Deng, Kaican Li, April Hua Liu, Yongxiang Huang, Nevin L. Zhang
* **arXiv ID:** 2508.06937
* **One-liner:** Achieved balanced local editing with training-free structural guidance.
* **Published in:** arxiv (9 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.06937) | [[PDF]](https://arxiv.org/pdf/2508.06937) | [[Code]](https://github.com/vaynexie/CannyEdit)

> **核心创新**
> 提出了CannyEdit，使用选择性Canny控制和双提示指导实现无缝编辑。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）模型的最新进展使得利用基础模型的生成先验进行无需训练的区域图像编辑成为可能。然而，现有方法难以平衡编辑区域中的文本遵循性、未编辑区域中的上下文保真度以及编辑的无缝集成。我们引入了CannyEdit，一个新颖的无需训练框架，通过两个关键创新解决这一三难问题。首先，选择性Canny控制将来自Canny ControlNet的结构指导仅应用于未编辑区域，保留原始图像的细节，同时在指定可编辑区域允许精确的文本驱动变化。其次，双提示指导利用局部提示进行特定编辑和全局提示进行整体场景一致性。通过这种协同方法，这些组件实现了对象添加、替换和移除的可控局部编辑，在当前基于区域的方法中实现了文本遵循性、上下文保真度和编辑无缝性的优越权衡。除此之外，CannyEdit提供了卓越的灵活性：它在添加任务中除了粗糙掩码外，甚至单点提示也能有效操作。此外，该框架可以以无需训练的方式与视觉语言模型无缝集成，用于需要规划和推理的复杂基于指令的编辑。我们的广泛评估表明，CannyEdit在复杂对象添加场景中相对于领先的基于指令编辑器表现出强大性能。
</details>

<details>
    <summary>Key points</summary>
    * 选择性Canny控制
    * 双提示指导
    * 无需训练框架
</details>
</details>

---


<details>
<summary><b> Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing</b></summary>

* **Authors:** Joonghyuk Shin, Alchan Hwang, Yujin Kim, Daneul Kim, Jaesik Park
* **arXiv ID:** 2508.07519
* **One-liner:** Adapted editing methods for MM-DiT architectures with bidirectional attention.
* **Published in:** arxiv (11 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.07519) | [[PDF]](https://arxiv.org/pdf/2508.07519) | [[Code]](https://github.com/SNU-VGILab/exploring-mmdit)

> **核心创新**
> 分析了MM-DiT注意机制，并提出了一种基于提示的编辑方法，支持从全局到局部编辑。

<details>
    <summary>Abstract</summary>
    基于Transformer的扩散模型最近已取代传统的U-Net架构，多模态扩散Transformer（MM-DiT）成为最先进模型（如Stable Diffusion 3和Flux.1）中的主导方法。先前的方法依赖于单向交叉注意机制，信息从文本嵌入流向图像潜在表示。相比之下，MMDiT引入了一种统一的注意机制，将来自两种模态的输入投影连接起来，并执行单一的全注意操作，允许文本和图像分支之间的双向信息流。这种架构转变对现有编辑技术提出了重大挑战。在本文中，我们通过将注意矩阵分解为四个不同的块来系统分析MM-DiT的注意机制，揭示其固有特性。通过这些分析，我们提出了一种鲁棒的、基于提示的MM-DiT图像编辑方法，支持从全局到局部编辑的各种MM-DiT变体，包括少步模型。我们相信我们的发现弥合了现有基于U-Net的方法与新兴架构之间的差距，为MM-DiT的行为模式提供了更深入的见解。
</details>

<details>
    <summary>Key points</summary>
    * 注意矩阵分解
    * 双向信息流
    * 与MM-DiT变体的兼容性
</details>
</details>

---


<details>
<summary><b> X2Edit: Revisiting Arbitrary-Instruction Image Editing through Self-Constructed Data and Task-Aware Representation Learning</b></summary>

* **Authors:** Jian Ma, Xujie Zhu, Zihao Pan, Qirong Peng, Xu Guo, Chen Chen, Haonan Lu
* **arXiv ID:** 2508.07607
* **One-liner:** Created a large-scale dataset and efficient model for diverse editing tasks.
* **Published in:** arxiv (11 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.07607) | [[PDF]](https://arxiv.org/pdf/2508.07607) | [[Code]](https://github.com/OPPO-Mente-Lab/X2Edit)

> **核心创新**
> 引入了X2Edit数据集和任务感知MoE-LoRA训练，结合对比学习。

<details>
    <summary>Abstract</summary>
    现有的开源任意指令图像编辑数据集仍然不理想，而一个与社区流行生成模型兼容的即插即用编辑模块明显缺失。在本文中，我们首先引入了X2Edit数据集，一个涵盖14种多样化编辑任务（包括主题驱动生成）的综合数据集。我们利用行业领先的统一图像生成模型和专家模型来构建数据。同时，我们使用VLM设计合理的编辑指令，并实施各种评分机制来过滤数据。结果，我们构建了3.7百万个高质量数据，类别平衡。其次，为了更好地与社区图像生成模型无缝集成，我们基于FLUX.1设计了任务感知的MoE-LoRA训练，仅使用完整模型参数的8%。为了进一步提高最终性能，我们利用扩散模型的内部表示，并基于图像编辑类型定义正/负样本，引入对比学习。广泛实验表明，该模型的编辑性能在许多优秀模型中具有竞争力。此外，构建的数据集相对于现有开源数据集表现出显著优势。X2Edit的开源代码、检查点和数据集可在以下链接找到：<a href="https://github.com/OPPO-Mente-Lab/X2Edit" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * X2Edit数据集包含3.7M样本
    * MoE-LoRA训练
    * 对比学习集成
</details>
</details>

---


<details>
<summary><b> An LLM-LVLM Driven Agent for Iterative and Fine-Grained Image Editing</b></summary>

* **Authors:** Zihan Liang, Jiahao Sun, Haoran Ma
* **arXiv ID:** 2508.17435
* **One-liner:** Developed a training-free agent for iterative and context-aware editing.
* **Published in:** arxiv (24 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.17435) | [[PDF]](https://arxiv.org/pdf/2508.17435) | [[Code]]()

> **核心创新**
> 引入了RefineEdit-Agent，集成LLM和LVLM，用于复杂编辑和反馈循环。

<details>
    <summary>Abstract</summary>
    尽管文本到图像（T2I）生成模型具有显著能力，但现实世界应用通常需要细粒度、迭代的图像编辑，而现有方法难以提供。关键挑战包括细粒度指令理解、修改过程中的鲁棒上下文保持，以及缺乏用于迭代细化的智能反馈机制。本文介绍了RefineEdit-Agent，一个新颖的、无需训练的智能代理框架，旨在通过实现复杂、迭代和上下文感知的图像编辑来解决这些限制。RefineEdit-Agent利用大语言模型（LLMs）的强大规划能力以及视觉语言大模型（LVLMs）的高级视觉理解和评估能力，在一个闭环系统中。我们的框架包括一个LVLM驱动的指令解析器和场景理解模块、一个多级LLM驱动的编辑规划器用于目标分解、工具选择和序列生成、一个迭代图像编辑模块，以及一个关键的LVLM驱动的反馈和评估循环。为了严格评估RefineEdit-Agent，我们提出了LongBench-T2I-Edit，一个新基准，包含500个初始图像，具有跨九个视觉维度的复杂多轮编辑指令。广泛实验表明，RefineEdit-Agent显著优于最先进的基线方法，在LongBench-T2I-Edit上平均得分3.67，而直接重新提示为2.29，InstructPix2Pix为2.91，基于GLIGEN的编辑为3.16，ControlNet-XL为3.39。消融研究、人工评估以及对迭代细化、骨干选择、工具使用和指令复杂性鲁棒性的分析进一步验证了我们代理设计在提供优越编辑保真度和上下文保持方面的有效性。
</details>

<details>
    <summary>Key points</summary>
    * LVLM驱动的指令解析
    * LLM驱动的规划
    * 迭代反馈和评估
</details>
</details>

---


<details>
<summary><b> Describe, Don&#39;t Dictate: Semantic Image Editing with Natural Language Intent</b></summary>

* **Authors:** En Ci, Shanyan Guan, Yanhao Ge, Yilin Zhang, Wei Li, Zhenyu Zhang, Jian Yang, Ying Tai
* **arXiv ID:** 2508.20505
* **One-liner:** Proposed a descriptive-prompt-based editing framework to improve semantic image editing.
* **Published in:** arxiv (28 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.20505) | [[PDF]](https://arxiv.org/pdf/2508.20505) | [[Code]]()

> **核心创新**
> 将基于指令的图像编辑重新定义为基于参考图像的文本到图像生成，以利用训练良好的模型而无需修改。

<details>
    <summary>Abstract</summary>
    尽管文本到图像生成取得了进展，语义图像编辑仍然是一个挑战。基于反转的算法不可避免地引入重建误差，而基于指令的模型主要受限于数据集质量和规模。为了解决这些问题，我们提出了一个基于描述性提示的编辑框架，名为DescriptiveEdit。其核心思想是将'基于指令的图像编辑'重新定义为'基于参考图像的文本到图像生成'，从而保留训练良好的文本到图像模型的生成能力，而无需架构修改或反转。具体来说，以参考图像和提示作为输入，我们引入了交叉注意力UNet，通过新增注意力桥接将参考图像特征注入到提示到编辑图像的生成过程中。由于其文本到图像的本质，DescriptiveEdit克服了指令数据集质量的限制，与ControlNet、IP-Adapter等扩展无缝集成，并具有更好的可扩展性。在Emu Edit基准测试上的实验表明，它提高了编辑准确性和一致性。
</details>

<details>
    <summary>Key points</summary>
    * 引入了带有注意力桥接的交叉注意力UNet
    * 将参考图像特征注入生成过程
    * 与ControlNet和IP-Adapter无缝集成
</details>
</details>

---


<details>
<summary><b> Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing</b></summary>

* **Authors:** Ziyun Zeng, Junhao Zhang, Wei Li, Mike Zheng Shou
* **arXiv ID:** 2509.01986
* **One-liner:** Introduced a unified model with balanced responsibilities for precise image editing.
* **Published in:** arxiv (2 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.01986) | [[PDF]](https://arxiv.org/pdf/2509.01986) | [[Code]](https://github.com/showlab/DIM)

> **核心创新**
> 通过使用带有思维链想象的数据集，将设计职责分配给理解模块以解决不平衡问题。

<details>
    <summary>Abstract</summary>
    近年来，将多模态理解和生成集成到单一统一模型中已成为一种有前景的范式。尽管这种方法在文本到图像生成中取得了强劲结果，但在精确图像编辑方面仍存在困难。我们将这一限制归因于职责分配的不平衡。理解模块主要作为翻译器，将用户指令编码为语义条件，而生成模块必须同时充当设计师和画家，推断原始布局、识别目标编辑区域并渲染新内容。这种不平衡是反直觉的，因为理解模块通常在复杂推理任务上比生成模块训练了数倍的数据。为了解决这个问题，我们引入了Draw-In-Mind，一个包含两个互补子集的数据集：DIM-T2I，包含1400万长上下文图像-文本对以增强复杂指令理解；以及DIM-Edit，包含23.3万由GPT-4o生成的思维链想象，作为图像编辑的显式设计蓝图。我们通过一个轻量级两层MLP连接冻结的Qwen2.5-VL-3B和可训练的SANA1.5-1.6B，并在提出的DIM数据集上训练，得到DIM-4.6B-T2I/Edit。尽管参数规模适中，DIM-4.6B-Edit在ImgEdit和GEdit-Bench基准测试中实现了SOTA或竞争性性能，优于UniWorld-V1和Step1X-Edit等更大模型。这些发现表明，将设计职责显式分配给理解模块对图像编辑有显著益处。我们的数据集和模型可在指定URL获取。
</details>

<details>
    <summary>Key points</summary>
    * 创建了包含DIM-T2I和DIM-Edit子集的DIM数据集
    * 通过MLP连接冻结的Qwen2.5-VL-3B与可训练的SANA1.5-1.6B
    * 在基准测试中实现了SOTA性能
</details>
</details>

---


<details>
<summary><b> MultiEdit: Advancing Instruction-based Image Editing on Diverse and Challenging Tasks</b></summary>

* **Authors:** Mingsong Li, Lin Liu, Hongjun Wang, Haoxing Chen, Xijun Gu, Shizhan Liu, Dong Gong, Junbo Zhao, Zhenzhong Lan, Jianguo Li
* **arXiv ID:** 2509.14638
* **One-liner:** Developed a comprehensive dataset to enhance instruction-based image editing.
* **Published in:** arxiv (18 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.14638) | [[PDF]](https://arxiv.org/pdf/2509.14638) | [[Code]](https://huggingface.co/datasets/inclusionAI/MultiEdit)

> **核心创新**
> 构建了MultiEdit数据集，包含多样化编辑任务和类型，以克服数据集限制。

<details>
    <summary>Abstract</summary>
    当前基于指令的图像编辑方法在处理挑战性编辑任务时存在困难，因为现有数据集的编辑类型和样本数量有限。此外，传统数据集构建常包含噪声图像-标题对，可能引入偏见并限制模型在复杂编辑场景中的能力。为了解决这些限制，我们引入了MultiEdit，一个包含超过10.7万高质量图像编辑样本的综合数据集。它通过18种非风格迁移编辑类型和38种风格迁移操作的多样化集合，涵盖了6个挑战性编辑任务，范围从复杂风格迁移到如人物参考编辑和图像内文本编辑等复杂语义操作。我们采用一种新颖的数据集构建流程，利用两个多模态大语言模型分别生成视觉自适应编辑指令和产生高保真编辑图像。广泛实验表明，使用我们的MultiEdit-Train集微调基础开源模型，在我们提出的MultiEdit-Test基准测试中显著提高了模型在复杂编辑任务上的性能，同时有效保留了其在标准编辑基准上的能力。我们相信MultiEdit为推进更多样化和挑战性的IBIE能力研究提供了宝贵资源。我们的数据集可在指定URL获取。
</details>

<details>
    <summary>Key points</summary>
    * 使用MLLMs生成指令和编辑图像
    * 涵盖18种非风格迁移和38种风格迁移操作
    * 提高了模型在复杂编辑任务上的性能
</details>
</details>

---


<details>
<summary><b> AutoEdit: Automatic Hyperparameter Tuning for Image Editing</b></summary>

* **Authors:** Chau Pham, Quan Dao, Mahesh Bhosale, Yunjie Tian, Dimitris Metaxas, David Doermann
* **arXiv ID:** 2509.15031
* **One-liner:** Proposed a reinforcement learning framework for efficient hyperparameter tuning in diffusion-based editing.
* **Published in:** arxiv (18 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.15031) | [[PDF]](https://arxiv.org/pdf/2509.15031) | [[Code]]()

> **核心创新**
> 将超参数搜索建模为顺序决策任务以减少计算成本。

<details>
    <summary>Abstract</summary>
    扩散模型的近期进展革新了文本引导图像编辑，但现有编辑方法在超参数识别方面面临关键挑战。为了获得合理的编辑性能，这些方法通常需要用户暴力调优多个相互依赖的超参数，如反转时间步和注意力修改。由于巨大的超参数搜索空间，这一过程带来高计算成本。我们将搜索最优编辑超参数视为扩散去噪过程中的顺序决策任务。具体来说，我们提出了一个强化学习框架，建立马尔可夫决策过程，动态调整去噪步骤中的超参数，将编辑目标集成到奖励函数中。该方法通过近端策略优化实现时间效率，同时保持最优超参数配置。实验表明，与现有暴力方法相比，显著减少了搜索时间和计算开销，推进了基于扩散的图像编辑框架在现实世界中的实际部署。代码可在指定URL获取。
</details>

<details>
    <summary>Key points</summary>
    * 建立马尔可夫决策过程以动态调整超参数
    * 使用近端策略优化实现时间效率
    * 将编辑目标集成到奖励函数中
</details>
</details>

---


<details>
<summary><b> CAMILA: Context-Aware Masking for Image Editing with Language Alignment</b></summary>

* **Authors:** Hyunseung Kim, Chiho Choi, Srikanth Malla, Sai Prahladh Padmanabhan, Saurabh Bagchi, Joon Hee Choi
* **arXiv ID:** 2509.19731
* **One-liner:** Introduced a context-aware method to handle infeasible instructions in image editing.
* **Published in:** arxiv (24 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.19731) | [[PDF]](https://arxiv.org/pdf/2509.19731) | [[Code]]()

> **核心创新**
> 设计CAMILA以验证指令一致性并仅应用相关编辑。

<details>
    <summary>Abstract</summary>
    文本引导图像编辑允许用户通过自然语言指令转换和合成图像，提供了相当大的灵活性。然而，大多数现有图像编辑模型天真地尝试遵循所有用户指令，即使这些指令本质上不可行或矛盾，常常导致无意义的输出。为了解决这些挑战，我们提出了一种上下文感知的图像编辑方法，名为CAMILA。CAMILA旨在验证指令与图像之间的上下文一致性，确保仅对指定区域应用相关编辑，同时忽略不可执行指令。为了全面评估这一新方法，我们构建了单指令和多指令图像编辑的数据集，包含不可行请求。我们的方法实现了比最先进模型更好的性能和更高的语义对齐，展示了其在处理复杂指令挑战同时保持图像完整性的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 构建了单指令和多指令编辑的数据集
    * 确保指令与图像之间的上下文一致性
    * 实现了比SOTA模型更好的语义对齐
</details>
</details>

---


<details>
<summary><b> EditVerse: Unifying Image and Video Editing and Generation with In-Context Learning</b></summary>

* **Authors:** Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, Daniil Pakhomov, Zhe Lin, Soo Ye Kim, Qiang Xu
* **arXiv ID:** 2509.20360
* **One-liner:** Created a unified framework for image and video generation and editing.
* **Published in:** arxiv (24 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.20360) | [[PDF]](https://arxiv.org/pdf/2509.20360) | [[Code]]()

> **核心创新**
> 将所有模态表示为统一令牌序列以实现跨模态学习。

<details>
    <summary>Abstract</summary>
    基础模型的近期进展凸显了统一和扩展的明确趋势，在多样化领域展现出涌现能力。尽管图像生成和编辑已从任务特定快速过渡到统一框架，但由于架构限制和数据稀缺，视频生成和编辑仍然碎片化。在这项工作中，我们引入了EditVerse，一个在单一模型中统一图像和视频生成与编辑的框架。通过将所有模态表示为统一令牌序列，EditVerse利用自注意力实现鲁棒上下文学习、自然跨模态知识转移，以及灵活处理任意分辨率和持续时间的输入和输出。为了解决视频编辑训练数据的缺乏，我们设计了一个可扩展数据管道，策划了23.2万视频编辑样本，并将其与大规模图像和视频数据集结合进行联合训练。此外，我们提出了EditVerseBench，第一个基于指令的视频编辑基准，涵盖多样化任务和分辨率。广泛实验和用户研究表明，EditVerse实现了最先进性能，超越现有开源和商业模型，同时在跨模态中展现出涌现编辑和生成能力。
</details>

<details>
    <summary>Key points</summary>
    * 设计了包含23.2万视频编辑样本的可扩展数据管道
    * 利用自注意力进行上下文学习
    * 引入了EditVerseBench基准
</details>
</details>

---


<details>
<summary><b> EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling</b></summary>

* **Authors:** Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Liu, Zheng liu
* **arXiv ID:** 2509.23909
* **One-liner:** Developed a high-fidelity reward model to enable reinforcement learning in image editing.
* **Published in:** arxiv (28 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.23909) | [[PDF]](https://arxiv.org/pdf/2509.23909) | [[Code]](https://github.com/VectorSpaceLab/EditScore)

> **核心创新**
> 引入EditScore奖励模型和基准以克服RL采用障碍。

<details>
    <summary>Abstract</summary>
    指令引导图像编辑已取得显著进展，但当前模型在处理复杂指令时仍面临挑战，且通常需要多个样本来产生期望结果。强化学习提供了一个有前景的解决方案，但其在图像编辑中的应用因缺乏高保真、高效的奖励信号而严重受阻。在这项工作中，我们提出了一种全面方法来克服这一障碍，核心是开发一个最先进的专用奖励模型。我们首先引入EditReward-Bench，一个系统评估编辑质量奖励模型的综合基准。基于此基准，我们开发了EditScore，一系列用于评估指令引导图像编辑质量的奖励模型。通过细致的数据策划和过滤，EditScore有效匹配了学习专有VLM的性能。此外，结合针对EditScore生成性质定制的有效自集成策略，我们最大变体甚至在基准中超越了GPT-5。然后我们证明，高保真奖励模型是解锁图像编辑在线RL的关键。实验表明，即使最大的开源VLM也无法提供有效学习信号，而EditScore实现了高效和鲁棒的政策优化。将我们的框架应用于强基础模型OmniGen2，得到的最终模型显示出实质性和一致的性能提升。总体而言，这项工作提供了从基准测试到奖励建模再到RL训练在图像编辑中的第一个系统路径，表明高保真、领域专用奖励模型是解锁RL在该领域全部潜力的关键。
</details>

<details>
    <summary>Key points</summary>
    * 创建了EditReward-Bench进行系统评估
    * 使用自集成策略改进性能
    * 通过EditScore实现高效政策优化
</details>
</details>

---


<details>
<summary><b> EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing</b></summary>

* **Authors:** Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, Wenhu Chen
* **arXiv ID:** 2509.26346
* **One-liner:** Built a reward model to scale high-quality synthetic training data for image editing.
* **Published in:** arxiv (30 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.26346) | [[PDF]](https://arxiv.org/pdf/2509.26346) | [[Code]](https://github.com/TIGER-AI-Lab/EditReward)

> **核心创新**
> 在大型人类偏好数据集上训练\mname以对齐人类偏好。

<details>
    <summary>Abstract</summary>
    最近，我们在自然语言指令图像编辑方面见证了巨大进展。几个闭源模型如GPT-Image-1、Seedream和Google-Nano-Banana显示出高度有前景的进展。然而，开源模型仍然落后。主要瓶颈是缺乏可靠的奖励模型来扩展高质量合成训练数据。为了解决这一关键瓶颈，我们构建了\mname，使用我们新的大规模人类偏好数据集训练，该数据集由训练有素的专家按照严格协议精心标注，包含超过20万偏好对。\mname在指令引导图像编辑任务中表现出与人类偏好的卓越对齐。实验显示，\mname在GenAI-Bench、AURORA-Bench、ImagenHub和我们新的\benchname等既定基准上实现了最先进的人类相关性，优于一系列VLM作为评判模型。此外，我们使用\mname从现有噪声ShareGPT-4o-Image数据集中选择高质量子集。我们在所选子集上训练Step1X-Edit，与在全集上训练相比显示出显著改进。这证明了\mname作为奖励模型扩展高质量图像编辑训练数据的能力。此外，其强对齐表明在如基于强化学习的后训练和图像编辑模型的测试时扩展等高级应用中的潜力。\mname及其训练数据集将被发布，以帮助社区构建更高质量的图像编辑训练数据集。
</details>

<details>
    <summary>Key points</summary>
    * 专家标注了超过20万偏好对
    * 使用\mname选择高质量数据子集
    * 在基准测试中实现了SOTA人类相关性
</details>
</details>

---


<details>
<summary><b> Query-Kontext: An Unified Multimodal Model for Image Generation and Editing</b></summary>

* **Authors:** Yuxin Song, Wenkai Dong, Shizun Wang, Qi Zhang, Song Xue, Tao Yuan, Hu Yang, Haocheng Feng, Hang Zhou, Xinyan Xiao, Jingdong Wang
* **arXiv ID:** 2509.26641
* **One-liner:** Introduced a novel approach to disentangle generative reasoning from synthesis in unified models.
* **Published in:** arxiv (30 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.26641) | [[PDF]](https://arxiv.org/pdf/2509.26641) | [[Code]]()

> **核心创新**
> 通过多模态kontext桥接VLM和扩散模型以改进编辑和生成。

<details>
    <summary>Abstract</summary>
    统一多模态模型在文本到图像生成和编辑中表现出卓越性能，无论是实例化为组装统一框架，将强大视觉语言模型与基于扩散的生成器耦合，还是作为早期融合理解和生成模态的朴素统一多模态模型。我们认为，在当前统一框架中，多模态生成推理的关键能力，包括指令理解、接地和图像引用以保持身份和忠实重建，本质上与高保真合成纠缠。在这项工作中，我们引入了Query-Kontext，一种通过多模态'kontext'桥接VLM和扩散模型的新方法，该kontext由从多模态输入编码的语义线索和粗粒度图像条件组成。这一设计将复杂多模态生成推理能力委托给强大的VLM，同时保留扩散模型用于高质量视觉合成的角色。为实现此，我们提出了三阶段渐进训练策略。首先，我们通过多模态kontext令牌将VLM连接到轻量级扩散头，以释放VLM的生成推理能力。其次，我们将此头扩展到大型预训练扩散模型以增强视觉细节和真实感。最后，我们引入低级图像编码器以提高图像保真度，并在下游任务上进行指令调优。此外，我们构建了一个综合数据管道，整合真实、合成和开源数据集，涵盖多样化多模态参考到图像场景，包括图像生成、指令驱动编辑、定制生成和多主题组合。实验表明，我们的方法匹配强统一基线，并在多个案例中甚至优于任务特定最先进方法。
</details>

<details>
    <summary>Key points</summary>
    * 提出了三阶段渐进训练策略
    * 使用多模态kontext令牌进行语义线索
    * 构建了涵盖多样化场景的综合数据管道
</details>
</details>

---


<details>
<summary><b> GODIVA: Generating Open-DomaIn Videos from nAtural Descriptions</b></summary>

* **Authors:** Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, Nan Duan
* **arXiv ID:** 2104.14806
* **One-liner:** Proposed an open-domain text-to-video pretrained model with good generalization.
* **Published in:** arxiv (30 Apr 2021)
* **Links:** [[Paper]](https://arxiv.org/abs/2104.14806) | [[PDF]](https://arxiv.org/pdf/2104.14806) | [[Code]]()

> **核心创新**
> 在大型数据集上预训练GODIVA以进行自回归视频生成。

<details>
    <summary>Abstract</summary>
    从文本生成视频是一个具有挑战性的任务，因为其训练计算需求高且评估答案无限可能。现有工作通常在简单或小数据集上实验，泛化能力相当有限。在这项工作中，我们提出了GODIVA，一个开放域文本到视频预训练模型，可以使用三维稀疏注意力机制以自回归方式从文本生成视频。我们在Howto100M上预训练我们的模型，这是一个大规模文本-视频数据集，包含超过1.36亿文本-视频对。实验表明，GODIVA不仅可以在下游视频生成任务上微调，而且在未见文本上具有良好的零样本能力。我们还提出了一个新指标称为相对匹配，以自动评估视频生成质量。几个挑战被列出并讨论为未来工作。
</details>

<details>
    <summary>Key points</summary>
    * 使用三维稀疏注意力机制
    * 在Howto100M上训练，包含1.36亿文本-视频对
    * 引入了相对匹配指标进行评估
</details>
</details>

---


<details>
<summary><b> NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion</b></summary>

* **Authors:** Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, Nan Duan
* **arXiv ID:** 2111.12417
* **One-liner:** Introduced NÜWA, a unified multimodal pre-trained model for generating and manipulating visual data.
* **Published in:** arxiv (24 Nov 2021)
* **Links:** [[Paper]](https://arxiv.org/abs/2111.12417) | [[PDF]](https://arxiv.org/pdf/2111.12417) | [[Code]](https://github.com/microsoft/NUWA)

> **核心创新**
> 设计了一个3D Transformer编码器-解码器框架，并采用3DNA机制，以高效处理文本、图像和视频数据。

<details>
    <summary>Abstract</summary>
    本文提出了一种名为NÜWA的统一多模态预训练模型，能够为各种视觉合成任务生成新的或操作现有的视觉数据（即图像和视频）。
</details>

<details>
    <summary>Key points</summary>
    * 3D Transformer编码器-解码器框架
    * 3D邻近注意力（3DNA）机制
    * 在多个任务上取得最先进结果
    * 在操作任务上展示零样本能力
</details>
</details>

---


<details>
<summary><b> VideoGen: A Reference-Guided Latent Diffusion Approach for High Definition Text-to-Video Generation</b></summary>

* **Authors:** Xin Li, Wenqing Chu, Ye Wu, Weihang Yuan, Fanglong Liu, Qi Zhang, Fu Li, Haocheng Feng, Errui Ding, Jingdong Wang
* **arXiv ID:** 2309.00398
* **One-liner:** Developed VideoGen, a text-to-video generation method using reference-guided latent diffusion.
* **Published in:** arxiv (1 Sep 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2309.00398) | [[PDF]](https://arxiv.org/pdf/2309.00398) | [[Code]]()

> **核心创新**
> 利用文本到图像模型生成参考图像，并通过级联潜在扩散模块实现高保真视频生成。

<details>
    <summary>Abstract</summary>
    本文提出了VideoGen，一种文本到视频生成方法，利用参考引导的潜在扩散生成高保真度和强时间一致性的高清视频。
</details>

<details>
    <summary>Key points</summary>
    * 从文本到图像模型获取参考图像
    * 级联潜在扩散模块
    * 基于流的时序上采样
    * 增强的视频解码器
</details>
</details>

---


<details>
<summary><b> Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation</b></summary>

* **Authors:** Zhiwu Qing, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yujie Wei, Yingya Zhang, Changxin Gao, Nong Sang
* **arXiv ID:** 2312.04483
* **One-liner:** Proposed HiGen, a diffusion model that decouples spatial and temporal factors for improved video generation.
* **Published in:** arxiv (7 Dec 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2312.04483) | [[PDF]](https://arxiv.org/pdf/2312.04483) | [[Code]](https://github.com/ali-vilab/VGen)

> **核心创新**
> 通过解耦视频的空间和时间因素，在结构和内容层面进行空间推理和时序推理，并使用内容线索指导运动和外貌变化。

<details>
    <summary>Abstract</summary>
    尽管扩散模型在生成逼真图像方面表现出强大能力，但生成真实且多样化的视频仍处于起步阶段。
</details>

<details>
    <summary>Key points</summary>
    * 解耦空间和时间因素
    * 空间推理和时序推理步骤
    * 提取运动和外观线索
</details>
</details>

---


<details>
<summary><b> UniCtrl: Improving the Spatiotemporal Consistency of Text-to-Video Diffusion Models via Training-Free Unified Attention Control</b></summary>

* **Authors:** Tian Xia, Xuweiyi Chen, Sihan Xu
* **arXiv ID:** 2403.02332
* **One-liner:** Introduced UniCtrl, a plug-and-play method to enhance spatiotemporal consistency in text-to-video models.
* **Published in:** arxiv (4 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.02332) | [[PDF]](https://arxiv.org/pdf/2403.02332) | [[Code]](https://github.com/XuweiyiChen/UniCtrl)

> **核心创新**
> 采用跨帧自注意力控制和运动注入，无需额外训练即可提高时空一致性和运动多样性。

<details>
    <summary>Abstract</summary>
    视频扩散模型已用于视频生成，通常结合文本和图像条件以增强对生成内容的控制。
</details>

<details>
    <summary>Key points</summary>
    * 跨帧自注意力控制
    * 运动注入
    * 时空同步
</details>
</details>

---


<details>
<summary><b> ViD-GPT: Introducing GPT-style Autoregressive Generation in Video Diffusion Models</b></summary>

* **Authors:** Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao
* **arXiv ID:** 2406.10981
* **One-liner:** Presented ViD-GPT, a causal video diffusion model for long-term consistent video generation.
* **Published in:** arxiv (16 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.10981) | [[PDF]](https://arxiv.org/pdf/2406.10981) | [[Code]](https://github.com/Dawn-LX/CausalCache-VDM)

> **核心创新**
> 引入因果时序注意力和帧作为提示机制，结合kv-cache以提升推理效率。

<details>
    <summary>Abstract</summary>
    随着扩散模型的进展，当前视频生成已实现令人印象深刻的品质，但生成时间一致的长视频仍具挑战性。
</details>

<details>
    <summary>Key points</summary>
    * 因果时序注意力
    * 帧作为提示机制
    * Kv-cache以加速
</details>
</details>

---


<details>
<summary><b> TALC: Time-Aligned Captions for Multi-Scene Text-to-Video Generation</b></summary>

* **Authors:** Hritik Bansal, Yonatan Bitton, Michal Yarom, Idan Szpektor, Aditya Grover, Kai-Wei Chang
* **arXiv ID:** 2405.04682
* **One-liner:** Developed TALC framework for generating multi-scene videos from text descriptions.
* **Published in:** arxiv (7 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.04682) | [[PDF]](https://arxiv.org/pdf/2405.04682) | [[Code]]()

> **核心创新**
> 通过时间对齐的文本条件增强机制，并在多场景数据上进行微调。

<details>
    <summary>Abstract</summary>
    大多数文本到视频生成模型通常生成单场景视频片段，但现实世界中多场景视频更为普遍。
</details>

<details>
    <summary>Key points</summary>
    * 时间对齐的标题
    * 场景的时间对齐
    * 使用多场景数据微调
</details>
</details>

---


<details>
<summary><b> DisenStudio: Customized Multi-subject Text-to-Video Generation with Disentangled Spatial Control</b></summary>

* **Authors:** Hong Chen, Xin Wang, Yipeng Zhang, Yuwei Zhou, Zeyang Zhang, Siao Tang, Wenwu Zhu
* **arXiv ID:** 2405.12796
* **One-liner:** Proposed DisenStudio for generating videos with multiple customized subjects.
* **Published in:** arxiv (21 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.12796) | [[PDF]](https://arxiv.org/pdf/2405.12796) | [[Code]]()

> **核心创新**
> 采用空间解耦交叉注意力机制和运动保持的解耦微调策略，以处理多主题生成。

<details>
    <summary>Abstract</summary>
    生成视频中的定制内容近来受到越来越多的关注，但现有工作主要关注单主题定制，存在主题缺失和属性绑定问题。
</details>

<details>
    <summary>Key points</summary>
    * 空间解耦交叉注意力
    * 多主题共现调优
    * 运动保持微调
</details>
</details>

---


<details>
<summary><b> MotionBooth: Motion-Aware Customized Text-to-Video Generation</b></summary>

* **Authors:** Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, Kai Chen
* **arXiv ID:** 2406.17758
* **One-liner:** Introduced MotionBooth for animating customized subjects with precise motion control.
* **Published in:** arxiv (25 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.17758) | [[PDF]](https://arxiv.org/pdf/2406.17758) | [[Code]](https://github.com/jianzongwu/MotionBooth)

> **核心创新**
> 通过微调结合专门损失函数和无训练技术，实现运动控制。

<details>
    <summary>Abstract</summary>
    本文提出了MotionBooth，一个创新框架，用于动画化定制主题，并精确控制对象和相机运动。
</details>

<details>
    <summary>Key points</summary>
    * 主题区域和视频保持损失
    * 交叉注意力图操作
    * 潜在偏移模块用于相机控制
</details>
</details>

---


<details>
<summary><b> Text-Animator: Controllable Visual Text Video Generation</b></summary>

* **Authors:** Lin Liu, Quande Liu, Shengju Qian, Yuan Zhou, Wengang Zhou, Houqiang Li, Lingxi Xie, Qi Tian
* **arXiv ID:** 2406.17777
* **One-liner:** Presented Text-Animator for generating videos with visualized text.
* **Published in:** arxiv (25 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.17777) | [[PDF]](https://arxiv.org/pdf/2406.17777) | [[Code]](https://github.com/laulampaul/text-animator)

> **核心创新**
> 开发了文本嵌入注入和相机控制模块，以提高生成文本的稳定性。

<details>
    <summary>Abstract</summary>
    视频生成在游戏、电子商务和广告等行业中是一项关键但具有挑战性的任务。
</details>

<details>
    <summary>Key points</summary>
    * 文本嵌入注入模块
    * 相机控制模块
    * 文本精炼模块
</details>
</details>

---


<details>
<summary><b> CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer</b></summary>

* **Authors:** Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, Jie Tang
* **arXiv ID:** 2408.06072
* **One-liner:** Developed CogVideoX, a large-scale diffusion transformer model for coherent long-duration video generation.
* **Published in:** arxiv (12 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.06072) | [[PDF]](https://arxiv.org/pdf/2408.06072) | [[Code]](https://github.com/zai-org/CogVideo)

> **核心创新**
> 采用3D变分自编码器、专家Transformer和渐进式训练，以提升文本-视频对齐质量。

<details>
    <summary>Abstract</summary>
    本文提出了CogVideoX，一种基于扩散Transformer的大规模文本到视频生成模型，能够生成与文本提示对齐的10秒连续视频。
</details>

<details>
    <summary>Key points</summary>
    * 3D变分自编码器
    * 具有自适应LayerNorm的专家Transformer
    * 渐进式训练和多分辨率帧打包
</details>
</details>

---


<details>
<summary><b> Qihoo-T2X: An Efficient Proxy-Tokenized Diffusion Transformer for Text-to-Any-Task</b></summary>

* **Authors:** Jing Wang, Ao Ma, Jiasong Feng, Dawei Leng, Yuhui Yin, Xiaodan Liang
* **arXiv ID:** 2409.04005
* **One-liner:** Proposed PT-DiT for efficient diffusion transformers by reducing redundant computation with proxy tokens.
* **Published in:** arxiv (6 Sep 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2409.04005) | [[PDF]](https://arxiv.org/pdf/2409.04005) | [[Code]](https://github.com/360CVGroup/Qihoo-T2X)

> **核心创新**
> 引入了一种稀疏代表性令牌注意力机制，使用代理令牌高效捕获全局语义，并结合窗口和移位窗口注意力进行细节建模。

<details>
    <summary>Abstract</summary>
    扩散变换器中的全局自注意力机制由于视觉信息的稀疏和冗余性质而涉及冗余计算，且空间窗口内令牌的注意力图显示出显著相似性。为解决这种冗余，我们提出了代理令牌化扩散变换器（PT-DiT），它采用稀疏代表性令牌注意力（其中代表性令牌数量远少于总令牌数）来高效建模全局视觉信息。具体而言，在每个变换器块内，我们计算每个时空窗口的平均令牌作为该区域的代理令牌。全局语义通过这些代理令牌的自注意力捕获，然后通过交叉注意力注入所有潜在令牌。同时，我们引入了窗口和移位窗口注意力以解决稀疏注意力机制在细节建模方面的限制。基于精心设计的PT-DiT，我们进一步开发了Qihoo-T2X系列，包括多种用于T2I、T2V和T2MV任务的模型。实验结果表明，PT-DiT在图像和视频生成任务中实现了竞争性性能，同时降低了计算复杂度（例如，与DiT相比减少49%，与PixArt-α相比减少34%）。Qihoo-T2X的视觉展示和源代码可在指定链接获取。
</details>

<details>
    <summary>Key points</summary>
    * 使用平均令牌作为每个时空窗口的代理令牌
    * 利用代理令牌的自注意力和交叉注意力将全局语义注入潜在令牌
    * 整合窗口和移位窗口注意力以增强细节建模
</details>
</details>

---


<details>
<summary><b> Loong: Generating Minute-level Long Videos with Autoregressive Language Models</b></summary>

* **Authors:** Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, Xihui Liu
* **arXiv ID:** 2410.02757
* **One-liner:** Developed Loong, an autoregressive LLM-based video generator for minute-long videos.
* **Published in:** arxiv (3 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.02757) | [[PDF]](https://arxiv.org/pdf/2410.02757) | [[Code]]()

> **核心创新**
> 将文本和视频令牌建模为统一序列，采用渐进式短到长训练和推理策略以减少误差累积。

<details>
    <summary>Abstract</summary>
    生成内容丰富、长达数分钟的长视频是理想但具有挑战性的。自回归大语言模型（LLMs）在自然语言处理领域已成功生成长序列令牌，而基于自回归LLMs的视频生成探索仅限于生成几秒的短视频。在这项工作中，我们深入分析了阻碍基于自回归LLM的视频生成器生成长视频的挑战。基于观察和分析，我们提出了Loong，一种新的基于自回归LLM的视频生成器，能够生成长达数分钟的视频。具体而言，我们将文本令牌和视频令牌建模为统一序列用于自回归LLMs，并从零开始训练模型。我们提出了渐进式短到长训练与损失重新加权方案，以缓解长视频训练中的损失不平衡问题。我们进一步研究了推理策略，包括视频令牌重新编码和采样策略，以减少推理过程中的误差累积。我们提出的Loong可以在10秒视频上训练，并扩展到根据文本提示生成长达数分钟的视频，如结果所示。更多样本可在指定链接获取。
</details>

<details>
    <summary>Key points</summary>
    * 统一文本和视频令牌用于自回归建模
    * 应用渐进式短到长训练与损失重新加权
    * 在推理中使用视频令牌重新编码和采样策略
</details>
</details>

---


<details>
<summary><b> DiCoDe: Diffusion-Compressed Deep Tokens for Autoregressive Video Generation with Language Models</b></summary>

* **Authors:** Yizhuo Li, Yuying Ge, Yixiao Ge, Ping Luo, Ying Shan
* **arXiv ID:** 2412.04446
* **One-liner:** Introduced DiCoDe for scalable video generation using deep tokens with high compression.
* **Published in:** arxiv (5 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.04446) | [[PDF]](https://arxiv.org/pdf/2412.04446) | [[Code]](https://github.com/liyz15/Diffusion-Compressed-Deep-Tokens)

> **核心创新**
> 利用扩散压缩深度令牌进行自回归视频生成，实现高效训练和与AR语言模型的可扩展性。

<details>
    <summary>Abstract</summary>
    视频本质上是时间序列。在这项工作中，我们探索了使用自回归（AR）语言模型以时序和可扩展方式建模视频的潜力，受其在自然语言处理中成功的启发。我们引入了DiCoDe，一种新颖方法，利用扩散压缩深度令牌以自回归方式生成视频。与使用压缩率有限的低级表示的现有方法不同，DiCoDe利用具有显著压缩率（令牌数量减少1000倍）的深度令牌。这种显著压缩通过利用视频扩散模型先验知识训练的分词器实现。深度令牌使DiCoDe能够使用普通AR语言模型进行视频生成，类似于将一种视觉“语言”翻译成另一种。通过将视频视为时间序列，DiCoDe充分利用了语言模型的自回归生成能力。DiCoDe使用现成的AR架构可扩展，并能够在仅使用4个A100 GPU训练的情况下生成从几秒到一分钟的视频。我们在定量和定性上评估DiCoDe，证明其在质量上与现有方法相当，同时确保高效训练。为展示其可扩展性，我们发布了一系列具有不同参数大小的DiCoDe配置，并观察到随着模型大小从100M增加到3B，性能持续改进。我们认为DiCoDe在学术界的探索代表了使用AR语言模型进行可扩展视频建模的有希望的第一步，为开发更大、更强大的视频生成模型铺平了道路。
</details>

<details>
    <summary>Key points</summary>
    * 使用通过视频扩散先验训练的分词器实现1000倍压缩的深度令牌
    * 采用普通AR语言模型进行视频生成
    * 将模型大小从100M扩展到3B参数以实现性能持续改进
</details>
</details>

---


<details>
<summary><b> InstanceCap: Improving Text-to-Video Generation via Instance-aware Structured Caption</b></summary>

* **Authors:** Tiehan Fan, Kepan Nan, Rui Xie, Penghao Zhou, Zhenheng Yang, Chaoyou Fu, Xiang Li, Jian Yang, Ying Tai
* **arXiv ID:** 2412.09283
* **One-liner:** Proposed InstanceCap for instance-level fine-grained video captioning to improve fidelity.
* **Published in:** arxiv (12 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.09283) | [[PDF]](https://arxiv.org/pdf/2412.09283) | [[Code]](https://github.com/NJU-PCALab/InstanceCap)

> **核心创新**
> 设计了一个实例感知结构化字幕框架，结合辅助模型和策划数据集以减少幻觉并增强字幕精度。

<details>
    <summary>Abstract</summary>
    文本到视频生成近年来迅速发展，取得了显著成果。训练通常依赖于视频-字幕配对数据，这在提高生成性能中起关键作用。然而，当前视频字幕往往存在细节不足、幻觉和运动描述不精确的问题，影响生成视频的保真度和一致性。在这项工作中，我们提出了一种新颖的实例感知结构化字幕框架，称为InstanceCap，首次实现实例级和细粒度视频字幕。基于此方案，我们设计了一个辅助模型集群，将原始视频转换为实例以增强实例保真度。视频实例进一步用于将密集提示精炼为结构化短语，实现简洁而精确的描述。此外，我们策划了一个22K的InstanceVid数据集用于训练，并提出了一个针对InstanceCap结构定制的增强管道用于推理。实验结果表明，我们提出的InstanceCap显著优于先前模型，确保字幕和视频之间的高保真度，同时减少幻觉。
</details>

<details>
    <summary>Key points</summary>
    * 使用辅助模型将视频转换为实例以增强保真度
    * 将密集提示精炼为结构化短语以实现简洁描述
    * 策划了一个22K InstanceVid数据集并提出了增强管道
</details>
</details>

---


<details>
<summary><b> TIV-Diffusion: Towards Object-Centric Movement for Text-driven Image to Video Generation</b></summary>

* **Authors:** Xingrui Wang, Xin Li, Yaosi Hu, Hanxin Zhu, Chen Hou, Cuiling Lan, Zhibo Chen
* **arXiv ID:** 2412.10275
* **One-liner:** Developed TIV-Diffusion for precise text-driven image-to-video generation via object-centric alignment.
* **Published in:** arxiv (13 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.10275) | [[PDF]](https://arxiv.org/pdf/2412.10275) | [[Code]]()

> **核心创新**
> 结合融合文本和视觉知识与尺度偏移调制以及对象中心对齐模块，确保运动一致性和高质量。

<details>
    <summary>Abstract</summary>
    文本驱动图像到视频生成（TI2V）旨在根据第一帧和相应文本描述生成可控视频。该任务的主要挑战在于两部分：（i）如何识别目标对象并确保运动轨迹与文本描述的一致性。（ii）如何提高生成视频的主观质量。为解决上述挑战，我们提出了一种新的基于扩散的TI2V框架，称为TIV-Diffusion，通过对象中心文本-视觉对齐，旨在基于文本描述的运动实现精确控制和高质量视频生成。具体而言，我们通过尺度偏移调制融合文本和视觉知识，使TIV-Diffusion模型能够感知文本描述的对象及其运动轨迹。此外，为缓解对象消失和对象与运动不对齐的问题，我们引入了一个对象中心文本-视觉对齐模块，通过解耦参考图像中的对象并将文本特征与每个对象单独对齐，减少对象/运动不对齐的风险。基于上述创新，我们的TIV-Diffusion在与现有TI2V方法相比，实现了最先进的高质量视频生成。
</details>

<details>
    <summary>Key points</summary>
    * 应用尺度偏移调制进行文本-视觉融合
    * 引入对象中心文本-视觉对齐模块以解耦和对齐对象
    * 解决对象消失和不对齐问题
</details>
</details>

---


<details>
<summary><b> LTX-Video: Realtime Video Latent Diffusion</b></summary>

* **Authors:** Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, Ofir Bibi
* **arXiv ID:** 2501.00103
* **One-liner:** Introduced LTX-Video, a holistic latent diffusion model for efficient high-resolution video generation.
* **Published in:** arxiv (30 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.00103) | [[PDF]](https://arxiv.org/pdf/2501.00103) | [[Code]](https://github.com/Lightricks/LTX-Video)

> **核心创新**
> 集成Video-VAE和去噪变换器，具有高压缩和全时空自注意力，实现快速生成而无需单独上采样。

<details>
    <summary>Abstract</summary>
    我们介绍了LTX-Video，一种基于变换器的潜在扩散模型，通过无缝集成Video-VAE和去噪变换器的职责，采用整体方法进行视频生成。与现有方法将这些组件视为独立不同，LTX-Video旨在优化它们的交互以提高效率和质量。其核心是一个精心设计的Video-VAE，实现了1:192的高压缩比，时空下采样为每令牌32 x 32 x 8像素，通过将分块操作从变换器输入重新定位到VAE输入实现。在这种高度压缩的潜在空间中操作使变换器能够高效执行全时空自注意力，这对于生成具有时间一致性的高分辨率视频至关重要。然而，高压缩固有地限制了精细细节的表示。为解决这一问题，我们的VAE解码器负责潜在到像素转换和最终去噪步骤，直接在像素空间中生成干净结果。这种方法保留了生成精细细节的能力，而无需承担单独上采样模块的运行成本。我们的模型支持多样化用例，包括文本到视频和图像到视频生成，两种能力同时训练。它实现了比实时更快的生成，在Nvidia H100 GPU上仅需2秒生成5秒24 fps的768x512分辨率视频，优于所有类似规模的现有模型。源代码和预训练模型公开可用，为可访问和可扩展视频生成设定了新基准。
</details>

<details>
    <summary>Key points</summary>
    * 设计了具有1:192压缩比和时空下采样的Video-VAE
    * 在潜在空间中使用全时空自注意力
    * 使VAE解码器负责潜在到像素转换和最终去噪
</details>
</details>

---


<details>
<summary><b> Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model</b></summary>

* **Authors:** Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, Yu Zhou, Deshan Sun, Deyu Zhou, Jian Zhou, Kaijun Tan, Kang An, Mei Chen, Wei Ji, Qiling Wu, Wen Sun, Xin Han, Yanan Wei, Zheng Ge, Aojie Li, Bin Wang, Bizhu Huang, Bo Wang, Brian Li, Changxing Miao, Chen Xu, Chenfei Wu, Chenguang Yu, Dapeng Shi, Dingyuan Hu, Enle Liu, Gang Yu, Ge Yang, Guanzhe Huang, Gulin Yan, Haiyang Feng, Hao Nie, Haonan Jia, Hanpeng Hu, Hanqi Chen, Haolong Yan, Heng Wang, Hongcheng Guo, Huilin Xiong, Huixin Xiong, Jiahao Gong, Jianchang Wu, Jiaoren Wu, Jie Wu, Jie Yang, Jiashuai Liu, Jiashuo Li, Jingyang Zhang, Junjing Guo, Junzhe Lin, Kaixiang Li, Lei Liu, Lei Xia, Liang Zhao, Liguo Tan, Liwen Huang, Liying Shi, Ming Li, Mingliang Li, Muhua Cheng, Na Wang, Qiaohui Chen, Qinglin He, Qiuyan Liang, Quan Sun, Ran Sun, Rui Wang, Shaoliang Pang, Shiliang Yang, Sitong Liu, Siqi Liu, Shuli Gao, Tiancheng Cao, Tianyu Wang, Weipeng Ming, Wenqing He, Xu Zhao, Xuelin Zhang, Xianfang Zeng, Xiaojia Liu, Xuan Yang, Yaqi Dai, Yanbo Yu, Yang Li, Yineng Deng, Yingming Wang, Yilei Wang, Yuanwei Lu, Yu Chen, Yu Luo, Yuchu Luo, Yuhe Yin, Yuheng Feng, Yuxiang Yang, Zecheng Tang, Zekai Zhang, Zidong Yang, Binxing Jiao, Jiansheng Chen, Jing Li, Shuchang Zhou, Xiangyu Zhang, Xinhao Zhang, Yibo Zhu, Heung-Yeung Shum, Daxin Jiang
* **arXiv ID:** 2502.10248
* **One-liner:** Presented Step-Video-T2V, a large-scale text-to-video model with 30B parameters for high-quality generation.
* **Published in:** arxiv (14 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.10248) | [[PDF]](https://arxiv.org/pdf/2502.10248) | [[Code]](https://github.com/stepfun-ai/Step-Video-T2V)

> **核心创新**
> 采用深度压缩Video-VAE、具有3D全注意力的DiT和Video-DPO以减少伪影并提高视觉质量。

<details>
    <summary>Abstract</summary>
    我们展示了Step-Video-T2V，一种最先进的文本到视频预训练模型，具有30B参数和生成长达204帧视频的能力。一个深度压缩变分自编码器，Video-VAE，专为视频生成任务设计，实现了16x16空间和8x时间压缩比，同时保持卓越的视频重建质量。用户提示使用两个双语文本编码器编码以处理英语和中文。一个具有3D全注意力的DiT使用流匹配训练，并用于将输入噪声去噪为潜在帧。一种基于视频的DPO方法，Video-DPO，被应用以减少伪影并提高生成视频的视觉质量。我们还详细介绍了我们的训练策略并分享了关键观察和见解。Step-Video-T2V的性能在一个新颖的视频生成基准Step-Video-T2V-Eval上评估，证明了其与开源和商业引擎相比的最先进文本到视频质量。此外，我们讨论了当前基于扩散的模型范式的局限性，并概述了视频基础模型的未来方向。我们在指定链接提供Step-Video-T2V和Step-Video-T2V-Eval。在线版本也可从指定链接访问。我们的目标是加速视频基础模型的创新并赋能视频内容创作者。
</details>

<details>
    <summary>Key points</summary>
    * 设计了具有16x16空间和8x时间压缩的Video-VAE
    * 使用流匹配训练DiT进行去噪
    * 应用Video-DPO以增强视频质量
</details>
</details>

---


<details>
<summary><b> Wan: Open and Advanced Large-Scale Video Generative Models</b></summary>

* **Authors:** Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, Ziyu Liu
* **arXiv ID:** 2503.20314
* **One-liner:** Introduced Wan, an open suite of video foundation models with leading performance and comprehensiveness.
* **Published in:** arxiv (26 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.20314) | [[PDF]](https://arxiv.org/pdf/2503.20314) | [[Code]](https://github.com/Wan-Video/Wan2.1)

> **核心创新**
> 通过新颖VAE、可扩展预训练、大规模数据策划和自动化评估实现进步，涵盖多个任务和模型大小。

<details>
    <summary>Abstract</summary>
    本报告介绍了Wan，一套全面且开放的视频基础模型套件，旨在推动视频生成的边界。基于主流扩散变换器范式，Wan通过一系列创新实现了生成能力的显著进步，包括我们的新颖VAE、可扩展预训练策略、大规模数据策划和自动化评估指标。这些贡献共同增强了模型的性能和多功能性。具体而言，Wan具有四个关键特征：领先性能：Wan的14B模型在包含数十亿图像和视频的大规模数据集上训练，展示了视频生成在数据和模型大小方面的缩放定律。它在多个内部和外部基准测试中持续优于现有开源模型以及最先进的商业解决方案，显示出明显且显著的性能优势。全面性：Wan提供两个能力模型，即1.3B和14B参数，分别用于效率和效果。它还涵盖多个下游应用，包括图像到视频、指令引导视频编辑和个人视频生成，涵盖多达八个任务。消费级效率：1.3B模型展示了卓越的资源效率，仅需8.19 GB VRAM，使其与广泛的消费级GPU兼容。开放性：我们开源了整个Wan系列，包括源代码和所有模型，目标是促进视频生成社区的发展。这种开放性旨在显著扩展行业视频制作的创意可能性，并为学术界提供高质量的视频基础模型。所有代码和模型可在指定链接获取。
</details>

<details>
    <summary>Key points</summary>
    * 开发了新颖VAE和可扩展预训练策略
    * 策划了包含数十亿图像和视频的大规模数据集
    * 提供从1.3B到14B参数的模型用于各种应用
</details>
</details>

---


<details>
<summary><b> VPO: Aligning Text-to-Video Generation Models with Prompt Optimization</b></summary>

* **Authors:** Jiale Cheng, Ruiliang Lyu, Xiaotao Gu, Xiao Liu, Jiazheng Xu, Yida Lu, Jiayan Teng, Zhuoyi Yang, Yuxiao Dong, Jie Tang, Hongning Wang, Minlie Huang
* **arXiv ID:** 2503.20491
* **One-liner:** Proposed VPO, a framework for prompt optimization to enhance video generation safety and quality.
* **Published in:** arxiv (26 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.20491) | [[PDF]](https://arxiv.org/pdf/2503.20491) | [[Code]](https://github.com/thu-coai/VPO)

> **核心创新**
> 采用两阶段优化，结合SFT和基于无害性、准确性和有用性原则的偏好学习。

<details>
    <summary>Abstract</summary>
    视频生成模型在文本到视频任务中取得了显著进展。这些模型通常在具有高度详细和精心制作描述的文本-视频对上训练，而推理期间真实世界用户输入往往简洁、模糊或结构不良。这种差距使得提示优化对于生成高质量视频至关重要。当前方法通常依赖大语言模型（LLMs）通过上下文学习精炼提示，但存在几个局限性：它们可能扭曲用户意图、忽略关键细节或引入安全风险。此外，它们在优化提示时不考虑对最终视频质量的影响，这可能导致次优结果。为解决这些问题，我们引入了VPO，一个基于无害性、准确性和有用性三个核心原则的优化提示框架。生成的提示忠实保留用户意图，更重要的是，提高了生成视频的安全性和质量。为实现这一点，VPO采用两阶段优化方法。首先，我们基于安全和对齐原则构建并精炼一个监督微调（SFT）数据集。其次，我们引入文本级和视频级反馈，通过偏好学习进一步优化SFT模型。我们的广泛实验表明，VPO与基线方法相比显著提高了安全性、对齐性和视频质量。此外，VPO在视频生成模型中表现出强泛化能力。进一步，我们证明VPO可以在视频生成模型上优于并结合RLHF方法，强调了VPO在对齐视频生成模型中的有效性。我们的代码和数据公开可用在指定链接。
</details>

<details>
    <summary>Key points</summary>
    * 构建并精炼SFT数据集用于安全和对齐
    * 使用文本级和视频级反馈进行偏好学习
    * 确保提示保留用户意图并提高视频质量
</details>
</details>

---


<details>
<summary><b> DyST-XL: Dynamic Layout Planning and Content Control for Compositional Text-to-Video Generation</b></summary>

* **Authors:** Weijie He, Mushui Liu, Yunlong Yu, Zhao Wang, Chao Wu
* **arXiv ID:** 2504.15032
* **One-liner:** Developed DyST-XL, a training-free framework for compositional text-to-video generation with precise control.
* **Published in:** arxiv (21 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.15032) | [[PDF]](https://arxiv.org/pdf/2504.15032) | [[Code]](https://github.com/XiaoBuL/DyST-XL)

> **核心创新**
> 整合动态布局规划器、双提示控制注意力和实体一致性约束以改进布局和实体一致性。

<details>
    <summary>Abstract</summary>
    组合文本到视频生成，需要合成具有多个交互实体和精确时空关系的动态场景，对于基于扩散的模型仍然是一个关键挑战。现有方法由于无约束的交叉注意力机制和不足的物理感知推理，在布局不连续性、实体身份漂移和不可信交互动态方面存在困难。为解决这些限制，我们提出了DyST-XL，一个无需训练的框架，通过帧感知控制增强现成的文本到视频模型（例如CogVideoX-5B）。DyST-XL整合了三个关键创新：（1）一个动态布局规划器，利用大语言模型（LLMs）将输入提示解析为实体-属性图，并生成物理感知关键帧布局，中间帧通过轨迹优化插值；（2）一个双提示控制注意力机制，通过帧感知注意力掩码强制执行局部化文本-视频对齐，实现对单个实体的精确控制；（3）一个实体一致性约束策略，在去噪过程中将第一帧特征嵌入传播到后续帧，无需手动注释即可保留对象身份。实验表明，DyST-XL在组合文本到视频生成中表现出色，显著提高了复杂提示的性能，并弥合了无需训练视频合成中的关键差距。代码在指定链接发布。
</details>

<details>
    <summary>Key points</summary>
    * 利用LLMs进行物理感知关键帧布局规划
    * 应用帧感知注意力掩码进行局部化对齐
    * 传播第一帧特征以保留对象身份
</details>
</details>

---


<details>
<summary><b> ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models</b></summary>

* **Authors:** Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M. Rehg, Tobias Hinz
* **arXiv ID:** 2505.07652
* **One-liner:** Enabled text-to-multi-shot video generation with character consistency and user control over shots.
* **Published in:** arxiv (12 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.07652) | [[PDF]](https://arxiv.org/pdf/2505.07652) | [[Code]]()

> **核心创新**
> 提出了一个框架，包括数据集收集和视频扩散模型的架构扩展，用于生成多镜头视频作为单一视频，并在帧之间保持完整注意力。

<details>
    <summary>Abstract</summary>
    当前基于扩散的文本到视频方法仅限于生成单镜头短视频片段，缺乏生成多镜头视频的能力，其中同一角色在不同或相同背景下执行不同活动。为解决这一限制，我们提出了一个框架，包括数据集收集流程和视频扩散模型的架构扩展，以实现文本到多镜头视频生成。我们的方法能够生成多镜头视频作为单一视频，在所有镜头的所有帧之间保持完整注意力，确保角色和背景一致性，并允许用户通过镜头特定条件控制镜头的数量、时长和内容。这是通过在文本到视频模型中引入过渡标记来控制新镜头开始的帧，以及使用局部注意力掩蔽策略来控制过渡标记的效果并实现镜头特定提示。为获取训练数据，我们提出了一种新颖的数据收集流程，从现有的单镜头视频数据集中构建多镜头视频数据集。大量实验表明，对预训练的文本到视频模型进行数千次迭代的微调足以使模型随后能够生成具有镜头特定控制的多镜头视频，优于基线方法。更多细节请参见<a href="https://shotadapter.github.io/" rel="external noopener nofollow" class="link-external link-https">此 https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 引入了过渡标记来控制镜头的开始。
    * 使用局部注意力掩蔽策略实现镜头特定提示。
    * 从单镜头数据集构建了多镜头视频数据集。
    * 对预训练模型进行微调以实现高效多镜头生成。
</details>
</details>

---


<details>
<summary><b> MOVi: Training-free Text-conditioned Multi-Object Video Generation</b></summary>

* **Authors:** Aimon Rahman, Jiang Liu, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Yusheng Su, Vishal M. Patel, Zicheng Liu, Emad Barsoum
* **arXiv ID:** 2505.22980
* **One-liner:** Enhanced multi-object video generation without training using LLM-guided trajectories and attention manipulation.
* **Published in:** arxiv (29 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.22980) | [[PDF]](https://arxiv.org/pdf/2505.22980) | [[Code]]()

> **核心创新**
> 引入了一种无训练方法，利用LLMs进行对象轨迹指导和噪声重新初始化来控制运动并防止对象间干扰。

<details>
    <summary>Abstract</summary>
    基于扩散的文本到视频（T2V）模型的最新进展显示出显著进步，但这些模型在生成包含多个对象的视频时仍面临挑战。大多数模型难以准确捕捉复杂的对象交互，常将某些对象视为静态背景元素并限制其运动。此外，它们经常无法生成提示中指定的多个不同对象，导致生成错误或对象间特征混合。本文提出了一种新颖的无训练方法，用于多对象视频生成，利用扩散模型的开放世界知识和大型语言模型（LLMs）。我们使用LLM作为对象轨迹的“导演”，并通过噪声重新初始化应用这些轨迹来实现精确的运动控制。我们进一步通过操纵注意力机制来更好地捕捉对象特定特征和运动模式，并防止对象间特征干扰，从而优化生成过程。大量实验验证了我们无训练方法的有效性，显著增强了现有视频扩散模型的多对象生成能力，在运动动态和对象生成准确性上实现了42%的绝对提升，同时保持了高保真度和运动平滑性。
</details>

<details>
    <summary>Key points</summary>
    * 使用LLM作为对象轨迹的导演。
    * 应用噪声重新初始化进行精确运动控制。
    * 操纵注意力机制以捕捉对象特定特征。
    * 防止对象间特征干扰。
</details>
</details>

---


<details>
<summary><b> Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion</b></summary>

* **Authors:** Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman
* **arXiv ID:** 2506.08009
* **One-liner:** Addressed exposure bias in autoregressive video diffusion models for real-time streaming generation.
* **Published in:** arxiv (9 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.08009) | [[PDF]](https://arxiv.org/pdf/2506.08009) | [[Code]](https://github.com/guandeh17/Self-Forcing)

> **核心创新**
> 提出了自强制训练范式，使用自回归展开和KV缓存将生成条件设定在自生成输出上，并应用整体视频级别损失。

<details>
    <summary>Abstract</summary>
    我们引入了自强制（Self Forcing），一种用于自回归视频扩散模型的新颖训练范式。它解决了长期存在的曝光偏差问题，即在推理过程中，基于真实上下文训练的模型必须基于自身不完美输出生成序列。与先前基于真实上下文帧去噪未来帧的方法不同，自强制通过在训练期间执行自回归展开和键值（KV）缓存，将每一帧的生成条件设定在先前自生成输出上。这一策略通过视频级别的整体损失实现监督，直接评估整个生成序列的质量，而非仅依赖传统的逐帧目标。为确保训练效率，我们采用少步扩散模型和随机梯度截断策略，有效平衡计算成本和性能。我们进一步引入了滚动KV缓存机制，实现高效的自回归视频外推。大量实验表明，我们的方法在单个GPU上实现了亚秒延迟的实时流视频生成，同时匹配甚至超越了显著较慢和非因果扩散模型的生成质量。项目网站：<a href="http://self-forcing.github.io/" rel="external noopener nofollow" class="link-external link-http">此 http URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 在训练期间执行自回归展开和KV缓存。
    * 使用整体损失评估整个生成序列。
    * 采用少步扩散和随机梯度截断以提高效率。
    * 引入滚动KV缓存以实现高效视频外推。
</details>
</details>

---


<details>
<summary><b> Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective</b></summary>

* **Authors:** Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, Zhihui Lin, Tao Feng, Pengwei Liu, Jiazheng Xing, Hao Luo, Jiasheng Tang, Fan Wang, Yi Yang
* **arXiv ID:** 2507.08801
* **One-liner:** Developed an autoregressive video generator (Lumos-1) with minimal LLM modifications for efficient multimodal spatiotemporal modeling.
* **Published in:** arxiv (11 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.08801) | [[PDF]](https://arxiv.org/pdf/2507.08801) | [[Code]](https://github.com/alibaba-damo-academy/Lumos)

> **核心创新**
> 保留了LLM架构，使用MM-RoPE实现平衡频率谱，并引入AR-DF处理逐帧损失不平衡和空间冗余。

<details>
    <summary>Abstract</summary>
    自回归大型语言模型（LLMs）已统一了广泛的语言任务，激发了自回归视频生成的初步努力。现有的自回归视频生成器要么偏离标准LLM架构，要么依赖笨重的外部文本编码器，或因下一个标记解码而产生过高延迟。本文中，我们介绍了Lumos-1，一种自回归视频生成器，保留了LLM架构，仅进行最小架构修改。为在LLMs中注入时空相关性，我们识别了纳入3D RoPE的有效性，并诊断了其不平衡的频率谱范围。因此，我们提出了MM-RoPE，一种RoPE方案，保留了原始文本RoPE，同时提供全面的频率谱和缩放的3D位置，用于建模多模态时空数据。此外，Lumos-1采用了一种标记依赖策略，遵循帧内双向性和帧间时间因果性。基于此依赖策略，我们识别了由空间信息冗余引起的逐帧损失不平衡问题，并通过提出自回归离散扩散强制（AR-DF）来解决。AR-DF在训练期间引入时间管掩蔽，并采用兼容的推理时掩蔽策略以避免质量下降。通过使用内存高效训练技术，我们在仅48个GPU上预训练Lumos-1，在GenEval上达到与EMU3相当的性能，在VBench-I2V上与COSMOS-Video2World相当，在VBench-T2V上与OpenSoraPlan相当。代码和模型可在<a href="https://github.com/alibaba-damo-academy/Lumos" rel="external noopener nofollow" class="link-external link-https">此 https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 纳入MM-RoPE以增强时空相关性。
    * 使用标记依赖策略，具有帧内双向性和帧间因果性。
    * 应用AR-DF，在训练中使用时间管掩蔽。
    * 采用内存高效训练技术。
</details>
</details>

---


<details>
<summary><b> TITAN-Guide: Taming Inference-Time AligNment for Guided Text-to-Video Diffusion Models</b></summary>

* **Authors:** Christian Simon, Masato Ishii, Akio Hayakawa, Zhi Zhong, Shusuke Takahashi, Takashi Shibuya, Yuki Mitsufuji
* **arXiv ID:** 2508.00289
* **One-liner:** Improved training-free guidance for text-to-video diffusion models with efficient memory usage and optimal control.
* **Published in:** arxiv (1 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.00289) | [[PDF]](https://arxiv.org/pdf/2508.00289) | [[Code]]()

> **核心创新**
> 提出了TITAN-Guide，使用前向梯度下降进行潜在优化，无需反向传播，以克服内存和控制限制。

<details>
    <summary>Abstract</summary>
    在条件扩散模型的近期发展中，仍需要大量监督微调来执行一类任务的控制。使用现成模型进行无训练指导是一种有利的替代方案，以避免对基础模型的进一步微调。然而，现有的无训练指导框架要么内存需求高，要么由于粗略估计而提供次优控制。这些缺点限制了其在需要密集计算的扩散模型（如文本到视频（T2V）扩散模型）中的应用。本工作中，我们提出了驯服推理时对齐的引导文本到视频扩散模型，即TITAN-Guide，它克服了内存空间问题，并在指导过程中提供比同类方法更优的控制。具体而言，我们开发了一种高效方法，用于优化扩散潜在变量，无需从判别性指导模型进行反向传播。我们特别研究了前向梯度下降在引导扩散任务中的应用，并探索了各种方向性指令选项。在实验中，我们证明了我们的方法在高效管理潜在优化内存方面的有效性，而先前方法则不足。我们提出的方法不仅最小化了内存需求，还显著提升了T2V在多个扩散指导基准上的性能。代码、模型和演示可在<a href="https://titanguide.github.io" rel="external noopener nofollow" class="link-external link-https">此 https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 开发了无需反向传播的高效潜在优化方法。
    * 使用前向梯度下降和方向性指令。
    * 在推理期间最小化内存需求。
    * 在多个基准上提升了T2V性能。
</details>
</details>

---


<details>
<summary><b> VidCLearn: A Continual Learning Approach for Text-to-Video Generation</b></summary>

* **Authors:** Luca Zanchetta, Lorenzo Papa, Luca Maiano, Irene Amerini
* **arXiv ID:** 2509.16956
* **One-liner:** Enabled continual learning for text-to-video generation without retraining from scratch.
* **Published in:** arxiv (21 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.16956) | [[PDF]](https://arxiv.org/pdf/2509.16956) | [[Code]]()

> **核心创新**
> 引入了VidCLearn，采用学生-教师架构、生成重放、时间一致性损失和视频检索，用于增量更新和知识保留。

<details>
    <summary>Abstract</summary>
    文本到视频生成是生成式AI中的一个新兴领域，能够从文本提示创建逼真、语义准确的视频。虽然当前模型实现了令人印象深刻的视觉质量和与输入文本的对齐，但它们通常依赖静态知识，难以在不从头重新训练的情况下纳入新数据。为解决这一限制，我们提出了VidCLearn，一种用于基于扩散的文本到视频生成的持续学习框架。VidCLearn采用学生-教师架构，其中学生模型通过新文本-视频对进行增量更新，教师模型通过生成重放帮助保留先前学到的知识。此外，我们引入了一种新颖的时间一致性损失以增强运动平滑性，以及一个视频检索模块以在推理时提供结构指导。我们的架构还设计为比现有模型更计算高效，同时保持满意的生成性能。实验结果显示，VidCLearn在视觉质量、语义对齐和时间一致性方面优于基线方法。
</details>

<details>
    <summary>Key points</summary>
    * 使用学生-教师架构进行增量学习。
    * 应用生成重放以保留知识。
    * 引入时间一致性损失以增强运动平滑性。
    * 纳入视频检索以提供结构指导。
</details>
</details>

---


<details>
<summary><b> Wan-Alpha: High-Quality Text-to-Video Generation with Alpha Channel</b></summary>

* **Authors:** Haotian Dong, Wenjing Wang, Chen Li, Di Lin
* **arXiv ID:** 2509.24979
* **One-liner:** Generated high-quality transparent videos with superior visual and motion realism.
* **Published in:** arxiv (29 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.24979) | [[PDF]](https://arxiv.org/pdf/2509.24979) | [[Code]](https://github.com/WeChatCV/Wan-Alpha)

> **核心创新**
> 提出了Wan-Alpha框架，使用VAE将alpha通道编码到RGB潜在空间，并在多样化RGBA数据集上训练，用于联合RGB和alpha生成。

<details>
    <summary>Abstract</summary>
    RGBA视频生成，包括表示透明度的alpha通道，正在各种应用中获得越来越多的关注。然而，现有方法常忽视视觉质量，限制了其实际可用性。本文中，我们提出了Wan-Alpha，一种新框架，通过联合学习RGB和alpha通道生成透明视频。我们设计了一种有效的变分自编码器（VAE），将alpha通道编码到RGB潜在空间中。然后，为支持我们的扩散变换器训练，我们构建了一个高质量和多样化的RGBA视频数据集。与最先进方法相比，我们的模型在视觉质量、运动真实性和透明度渲染方面表现出优越性能。值得注意的是，我们的模型能够生成各种半透明对象、发光效果和细粒度细节，如发丝。发布的模型可在我们的网站上获取：<a href="https://donghaotian123.github.io/Wan-Alpha/" rel="external noopener nofollow" class="link-external link-https">此 https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 设计了VAE用于alpha通道编码到RGB潜在空间。
    * 构建了高质量RGBA视频数据集。
    * 使用扩散变换器进行训练。
    * 生成半透明对象和细粒度细节。
</details>
</details>

---


<details>
<summary><b> TempoControl: Temporal Attention Guidance for Text-to-Video Models</b></summary>

* **Authors:** Shira Schiber, Ofir Lindenbaum, Idan Schwartz
* **arXiv ID:** 2510.02226
* **One-liner:** Enabled fine-grained temporal control in text-to-video generation without retraining.
* **Published in:** arxiv (2 Oct 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2510.02226) | [[PDF]](https://arxiv.org/pdf/2510.02226) | [[Code]]()

> **核心创新**
> 引入了TempoControl，使用交叉注意力图和优化原则在推理期间进行视觉概念的时间对齐。

<details>
    <summary>Abstract</summary>
    生成视频模型的最新进展使得能够基于自然语言提示创建高质量视频。然而，这些模型常缺乏细粒度时间控制，即不允许用户在生成序列中指定特定视觉元素何时出现。本工作中，我们介绍了TempoControl，一种方法，允许在推理期间进行视觉概念的时间对齐，无需重新训练或额外监督。TempoControl利用交叉注意力图（文本到视频扩散模型的关键组件），通过一种新颖的优化方法指导概念的时间安排。我们的方法使用三个互补原则引导注意力：将其时间形状与控制信号对齐（通过相关性），在需要可见性的地方放大它（通过能量），并保持空间焦点（通过熵）。TempoControl允许精确控制时间安排，同时确保高视频质量和多样性。我们证明了其在各种视频生成应用中的有效性，包括单对象和多对象的时间重排序，以及动作和音频对齐生成。
</details>

<details>
    <summary>Key points</summary>
    * 利用交叉注意力图进行时间指导。
    * 应用优化，包括相关性、能量和熵原则。
    * 允许精确控制概念时间安排。
    * 保持高视频质量和多样性。
</details>
</details>

---


<details>
<summary><b> CamViG: Camera Aware Image-to-Video Generation with Multimodal Transformers</b></summary>

* **Authors:** Andrew Marmon, Grant Schindler, José Lezama, Dan Kondratyuk, Bryan Seybold, Irfan Essa
* **arXiv ID:** 2405.13195
* **One-liner:** Integrated 3D camera motion as a conditioning signal for video generation.
* **Published in:** arxiv (21 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.13195) | [[PDF]](https://arxiv.org/pdf/2405.13195) | [[Code]]()

> **核心创新**
> 扩展了多模态变换器以包括3D相机控制，实现了从单帧和相机信号生成准确相机路径。

<details>
    <summary>Abstract</summary>
    我们将多模态变换器扩展为包括3D相机运动作为视频生成任务的调节信号。生成视频模型正变得越来越强大，因此研究重点转向控制此类模型输出的方法。我们提议通过将生成视频条件设定在三维相机运动在生成视频过程中的编码上，为生成视频方法添加虚拟3D相机控制。结果表明，我们（1）能够成功在视频生成期间控制相机，从单帧和相机信号开始，以及（2）我们使用传统计算机视觉方法证明了生成的3D相机路径的准确性。
</details>

<details>
    <summary>Key points</summary>
    * 将视频生成条件设定在3D相机运动编码上。
    * 使用传统计算机视觉方法验证准确性。
    * 从单帧控制相机生成。
    * 展示了准确的3D相机路径。
</details>
</details>

---


<details>
<summary><b> CamCo: Camera-Controllable 3D-Consistent Image-to-Video Generation</b></summary>

* **Authors:** Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, Arash Vahdat
* **arXiv ID:** 2406.02509
* **One-liner:** Achieved fine-grained camera pose control for image-to-video generation with improved 3D consistency.
* **Published in:** arxiv (4 Jun 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2406.02509) | [[PDF]](https://arxiv.org/pdf/2406.02509) | [[Code]]()

> **核心创新**
> 提出了CamCo，使用Plücker坐标进行相机姿态输入和极线注意力以强制执行3D约束，并在真实世界视频上微调。

<details>
    <summary>Abstract</summary>
    近期，视频扩散模型已成为高质量视频内容创作的表达性生成工具，广泛供普通用户使用。然而，这些模型常不提供对视频生成中相机姿态的精确控制，限制了电影语言的表达和用户控制。为解决这一问题，我们介绍了CamCo，它允许在图像到视频生成中进行细粒度相机姿态控制。我们使用Plücker坐标为预训练的图像到视频生成器配备精确参数化的相机姿态输入。为增强生成视频的3D一致性，我们在每个注意力块中集成了一个极线注意力模块，对特征图强制执行极线约束。此外，我们通过结构从运动算法估计相机姿态的真实世界视频上微调CamCo，以更好地合成对象运动。我们的实验显示，CamCo在3D一致性和相机控制能力方面显著优于先前模型，同时有效生成合理的对象运动。项目页面：<a href="https://ir1d.github.io/CamCo/" rel="external noopener nofollow" class="link-external link-https">此 https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 使用Plücker坐标进行相机姿态参数化。
    * 集成极线注意力模块以增强3D一致性。
    * 在估计相机姿态的真实世界视频上微调。
    * 增强了对象运动合成和相机控制。
</details>
</details>

---


<details>
<summary><b> JVID: Joint Video-Image Diffusion for Visual-Quality and Temporal-Consistency in Video Generation</b></summary>

* **Authors:** Hadrien Reynaud, Matthew Baugh, Mischa Dombrowski, Sarah Cechnicka, Qingjie Meng, Bernhard Kainz
* **arXiv ID:** 2409.14149
* **One-liner:** Introduced JVID for high-quality, temporally coherent video generation by integrating image and video diffusion models.
* **Published in:** arxiv (21 Sep 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2409.14149) | [[PDF]](https://arxiv.org/pdf/2409.14149) | [[Code]]()

> **核心创新**
> 在反向扩散过程中结合潜在图像扩散模型（LIDM）和潜在视频扩散模型（LVDM）以处理时空动态。

<details>
    <summary>Abstract</summary>
    我们提出了联合视频-图像扩散模型（JVID），这是一种生成高质量且时间一致视频的新方法。
</details>

<details>
    <summary>Key points</summary>
    * LIDM和LVDM的集成
    * 反向扩散过程以增强图像质量和时间一致性
    * 视频真实性和一致性的定量和定性改进
</details>
</details>

---


<details>
<summary><b> FrameBridge: Improving Image-to-Video Generation with Bridge Models</b></summary>

* **Authors:** Yuji Wang, Zehua Chen, Xiaoyu Chen, Yixiang Wei, Jun Zhu, Jianfei Chen
* **arXiv ID:** 2410.15371
* **One-liner:** Proposed FrameBridge for improved image-to-video generation by modeling frame-to-frames as a data-to-data process.
* **Published in:** arxiv (20 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.15371) | [[PDF]](https://arxiv.org/pdf/2410.15371) | [[Code]]()

> **核心创新**
> 使用桥模型将生成过程与I2V任务对齐，利用给定图像信息以获得更好的一致性。

<details>
    <summary>Abstract</summary>
    扩散模型在图像到视频（I2V）生成方面取得了显著进展，但其噪声到数据的生成过程与该任务本质上不匹配，可能导致次优合成质量。在这项工作中，我们提出了FrameBridge。通过基于数据到数据生成过程的桥模型建模帧到帧的生成过程，我们能够充分利用给定图像中的信息，并改进生成过程与I2V任务之间的一致性。此外，我们针对训练I2V模型的两种流行设置分别提出了两种新技术。首先，我们提出了信噪比对齐微调（SAF），首次尝试将扩散模型微调为桥模型，从而允许我们利用预训练的基于扩散的文本到视频（T2V）模型。其次，我们提出了神经先验，进一步提高了FrameBridge在从头开始训练时的合成质量。在WebVid-2M和UCF-101上进行的实验表明，FrameBridge相比扩散对应方法具有优越的质量（在MSR-VTT上零样本FVD 95 vs. 192，在UCF-101上非零样本FVD 122 vs. 171），以及我们提出的SAF和神经先验在基于桥的I2V模型中的优势。
</details>

<details>
    <summary>Key points</summary>
    * 帧到帧桥模型
    * 信噪比对齐微调（SAF）用于微调扩散模型
    * 神经先验用于从头开始训练
    * 在WebVid-2M和UCF-101上的优越性能
</details>
</details>

---


<details>
<summary><b> TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation</b></summary>

* **Authors:** Wenhao Wang, Yi Yang
* **arXiv ID:** 2411.04709
* **One-liner:** Introduced TIP-I2V, the first large-scale dataset for image-to-video prompts with over 1.70 million entries.
* **Published in:** arxiv (5 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.04709) | [[PDF]](https://arxiv.org/pdf/2411.04709) | [[Code]](https://github.com/WangWenhao0716/TIP-I2V)

> **核心创新**
> 提供用户提供的文本和图像提示及生成视频，以促进I2V研究和模型评估。

<details>
    <summary>Abstract</summary>
    视频生成模型正在革新内容创作，其中图像到视频模型因其增强的可控性、视觉一致性和实际应用而受到越来越多的关注。然而，尽管这些模型很受欢迎，它们依赖于用户提供的文本和图像提示，目前还没有专门用于研究这些提示的数据集。在本文中，我们介绍了TIP-I2V，这是第一个大规模数据集，包含超过170万个独特的用户提供的文本和图像提示，专门用于图像到视频生成。此外，我们提供了来自五个最先进的图像到视频模型的相应生成视频。我们首先概述了策划这个大规模数据集的耗时且昂贵的过程。接下来，我们将TIP-I2V与两个流行的提示数据集VidProM（文本到视频）和DiffusionDB（文本到图像）进行比较，突出了基本和语义信息的差异。该数据集促进了图像到视频研究的进展。例如，为了开发更好的模型，研究人员可以使用TIP-I2V中的提示来分析用户偏好并评估他们训练模型的多维性能；为了增强模型安全性，他们可以专注于解决由图像到视频模型引起的错误信息问题。由TIP-I2V启发的新研究以及与现有数据集的差异强调了专门图像到视频提示数据集的重要性。
</details>

<details>
    <summary>Key points</summary>
    * 策划大规模数据集与独特提示
    * 与VidProM和DiffusionDB的比较
    * 支持用户偏好分析和模型安全性评估
</details>
</details>

---


<details>
<summary><b> SG-I2V: Self-Guided Trajectory Control in Image-to-Video Generation</b></summary>

* **Authors:** Koichi Namekata, Sherwin Bahmani, Ziyi Wu, Yash Kant, Igor Gilitschenski, David B. Lindell
* **arXiv ID:** 2411.04989
* **One-liner:** Developed SG-I2V for zero-shot controllable image-to-video generation without fine-tuning.
* **Published in:** arxiv (7 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.04989) | [[PDF]](https://arxiv.org/pdf/2411.04989) | [[Code]](https://github.com/Kmcode1/SG-I2V)

> **核心创新**
> 依赖预训练模型知识进行自引导控制，提高视觉质量和运动保真度。

<details>
    <summary>Abstract</summary>
    图像到视频生成方法已经实现了令人印象深刻的照片级真实质量。然而，调整生成视频中的特定元素，如物体运动或相机移动，通常是一个繁琐的试错过程，例如涉及使用不同随机种子重新生成视频。最近的技术通过微调预训练模型以遵循条件信号（如边界框或点轨迹）来解决这个问题。然而，这种微调过程可能计算成本高昂，并且需要带有注释物体运动的数据集，这些数据集可能难以获取。在这项工作中，我们介绍了SG-I2V，一个用于可控图像到视频生成的框架，它是自引导的——提供零样本控制，仅依赖于预训练图像到视频扩散模型中的知识，无需微调或外部知识。我们的零样本方法在视觉质量和运动保真度方面优于无监督基线，同时显著缩小了与监督模型的性能差距。
</details>

<details>
    <summary>Key points</summary>
    * 零样本控制框架
    * 无需微调或外部知识
    * 优于无监督基线并缩小与监督模型的差距
</details>
</details>

---


<details>
<summary><b> OmniDrag: Enabling Motion Control for Omnidirectional Image-to-Video Generation</b></summary>

* **Authors:** Weiqi Li, Shijie Zhao, Chong Mou, Xuhan Sheng, Zhenyu Zhang, Qian Wang, Junlin Li, Li Zhang, Jian Zhang
* **arXiv ID:** 2412.09623
* **One-liner:** Proposed OmniDrag for accurate, high-quality omnidirectional image-to-video generation with motion control.
* **Published in:** arxiv (12 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.09623) | [[PDF]](https://arxiv.org/pdf/2412.09623) | [[Code]](https://github.com/lwq20020127/OmniDrag)

> **核心创新**
> 引入全向控制模块和球形运动估计器以处理复杂球形运动。

<details>
    <summary>Abstract</summary>
    随着虚拟现实的普及，对可控创建沉浸式和动态的全向视频（ODV）的需求正在增加。虽然先前的文本到ODV生成方法取得了令人印象深刻的结果，但由于仅依赖文本输入，它们难以处理内容不准确和不一致的问题。尽管最近的运动控制技术为视频生成提供了细粒度控制，但直接将这些方法应用于ODV通常会导致空间扭曲和不令人满意的性能，尤其是在复杂的球形运动中。为了解决这些挑战，我们提出了OmniDrag，这是第一种实现场景级和物体级运动控制的方法，用于准确、高质量的全向图像到视频生成。基于预训练的视频扩散模型，我们引入了一个全向控制模块，该模块与时间注意力层联合微调，以有效处理复杂的球形运动。此外，我们开发了一种新颖的球形运动估计器，能够准确提取运动控制信号，并允许用户通过简单绘制手柄和目标点来执行拖拽式ODV生成。我们还提出了一个新的数据集Move360，解决了具有大场景和物体运动的ODV数据稀缺问题。实验表明，OmniDrag在实现整体场景级和细粒度物体级控制方面具有显著优势。
</details>

<details>
    <summary>Key points</summary>
    * 全向控制模块
    * 球形运动估计器用于拖拽式生成
    * Move360数据集用于ODV数据
    * 与时间注意力的联合微调
</details>
</details>

---


<details>
<summary><b> TIV-Diffusion: Towards Object-Centric Movement for Text-driven Image to Video Generation</b></summary>

* **Authors:** Xingrui Wang, Xin Li, Yaosi Hu, Hanxin Zhu, Chen Hou, Cuiling Lan, Zhibo Chen
* **arXiv ID:** 2412.10275
* **One-liner:** Introduced TIV-Diffusion for precise control and high-quality video generation via object-centric textual-visual alignment.
* **Published in:** arxiv (13 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.10275) | [[PDF]](https://arxiv.org/pdf/2412.10275) | [[Code]]()

> **核心创新**
> 结合融合文本和视觉知识及对齐模块，确保物体一致性和运动准确性。

<details>
    <summary>Abstract</summary>
    文本驱动的图像到视频生成（TI2V）旨在根据第一帧和相应的文本描述生成可控视频。该任务的主要挑战在于两部分：（i）如何识别目标物体并确保运动轨迹与文本描述之间的一致性。（ii）如何提高生成视频的主观质量。为了解决上述挑战，我们提出了一种新的基于扩散的TI2V框架，称为TIV-Diffusion，通过以物体为中心的文本-视觉对齐，旨在基于不同物体的文本描述运动实现精确控制和高质量视频生成。具体来说，我们通过尺度偏移调制融合文本和视觉知识，使我们的TIV-Diffusion模型能够感知文本描述的物体及其运动轨迹。此外，为了缓解物体消失和物体与运动不对齐的问题，我们引入了一个以物体为中心的文本-视觉对齐模块，该模块通过解耦参考图像中的物体并单独将文本特征与每个物体对齐，减少了物体/运动不对齐的风险。基于以上创新，我们的TIV-Diffusion与现有TI2V方法相比，实现了最先进的高质量视频生成。
</details>

<details>
    <summary>Key points</summary>
    * 以物体为中心的文本-视觉对齐
    * 尺度偏移调制用于感知
    * 解耦并单独对齐物体
    * 最先进的视频生成质量
</details>
</details>

---


<details>
<summary><b> Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation</b></summary>

* **Authors:** Guy Yariv, Yuval Kirstain, Amit Zohar, Shelly Sheynin, Yaniv Taigman, Yossi Adi, Sagie Benaim, Adam Polyak
* **arXiv ID:** 2501.03059
* **One-liner:** Proposed a two-stage compositional framework for I2V generation using mask-based motion trajectory as intermediate representation.
* **Published in:** arxiv (6 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.03059) | [[PDF]](https://arxiv.org/pdf/2501.03059) | [[Code]](https://guyyariv.github.io/TTM/)

> **核心创新**
> 将生成分解为明确表示和视频阶段，使用物体级注意力实现一致性和真实性。

<details>
    <summary>Abstract</summary>
    我们考虑图像到视频（I2V）生成任务，该任务涉及将静态图像转换为基于文本描述的现实视频序列。虽然最近的进展产生了照片级真实的输出，但它们经常难以创建具有准确和一致物体运动的视频，尤其是在多物体场景中。为了解决这些限制，我们提出了一个两阶段组合框架，将I2V生成分解为：（i）一个明确的中间表示生成阶段，随后是（ii）一个基于该表示的视频生成阶段。我们的关键创新是引入了基于掩码的运动轨迹作为中间表示，该表示捕获了语义物体信息和运动，实现了运动和语义的表达性但紧凑的表示。为了在第二阶段整合学习到的表示，我们利用物体级注意力目标。具体来说，我们考虑一个空间、每物体的掩码交叉注意力目标，将物体特定提示整合到相应的潜在空间区域，以及一个掩码时空自注意力目标，确保每个物体的帧到帧一致性。我们在具有多物体和高运动场景的挑战性基准上评估我们的方法，并经验证明所提出的方法在时间一致性、运动真实性和文本提示忠实度方面实现了最先进的结果。此外，我们引入了\benchmark，一个新的挑战性基准，用于单物体和多物体I2V生成，并展示了我们的方法在该基准上的优越性。
</details>

<details>
    <summary>Key points</summary>
    * 两阶段框架与中间表示
    * 基于掩码的运动轨迹
    * 物体级注意力目标
    * 在多物体和高运动基准上的评估
</details>
</details>

---


<details>
<summary><b> VidCRAFT3: Camera, Object, and Lighting Control for Image-to-Video Generation</b></summary>

* **Authors:** Sixiao Zheng, Zimian Peng, Yanpeng Zhou, Yi Zhu, Hang Xu, Xiangru Huang, Yanwei Fu
* **arXiv ID:** 2502.07531
* **One-liner:** Presented VidCRAFT3 for unified control over camera motion, object motion, and lighting direction in I2V generation.
* **Published in:** arxiv (11 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.07531) | [[PDF]](https://arxiv.org/pdf/2502.07531) | [[Code]]()

> **核心创新**
> 整合Image2Cloud、ObjMotionNet和空间三重注意力变换器，采用三阶段训练策略。

<details>
    <summary>Abstract</summary>
    可控图像到视频（I2V）生成将参考图像转换为由用户指定控制信号引导的一致视频。在内容创建工作流中，对相机运动、物体运动和光照方向的精确和同时控制提高了准确性和灵活性。然而，现有方法通常将这些控制信号分开处理，主要是由于缺乏具有高质量联合注释的数据集以及跨模态的不匹配控制空间。我们提出了VidCRAFT3，一个统一且灵活的I2V框架，通过整合三个核心组件，支持对相机运动、物体运动和光照方向的独立和联合控制。Image2Cloud从参考图像重建3D点云以实现精确相机运动控制。ObjMotionNet将稀疏物体轨迹编码为多尺度光流特征以引导物体运动。空间三重注意力变换器通过并行交叉注意力整合光照方向嵌入。为了解决联合注释数据的稀缺性，我们策划了VideoLightingDirection（VLD）数据集，包含带有每帧光照方向标签的合成静态场景视频片段，并采用三阶段训练策略，使得在没有完全联合注释的情况下实现鲁棒学习。广泛实验表明，VidCRAFT3在控制精度和视觉一致性方面优于现有方法。
</details>

<details>
    <summary>Key points</summary>
    * 统一框架与三个核心组件
    * Image2Cloud用于3D重建
    * ObjMotionNet用于物体运动编码
    * 空间三重注意力用于光照控制
    * VLD数据集用于训练
</details>
</details>

---


<details>
<summary><b> RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control</b></summary>

* **Authors:** Teng Li, Guangcong Zheng, Rui Jiang, Shuigen Zhan, Tao Wu, Yehao Lu, Yining Lin, Chuanyun Deng, Yepan Xiong, Min Chen, Lin Cheng, Xi Li
* **arXiv ID:** 2502.10059
* **One-liner:** Developed RealCam-I2V for precise camera control in video generation using monocular depth estimation and 3D scene reconstruction.
* **Published in:** arxiv (14 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.10059) | [[PDF]](https://arxiv.org/pdf/2502.10059) | [[Code]](https://github.com/ZGCTroy/RealCam-I2V)

> **核心创新**
> 实现直观相机轨迹绘制和场景约束噪声整形，以提高可控性和质量。

<details>
    <summary>Abstract</summary>
    最近在相机轨迹引导的图像到视频生成方面的进展提供了比基于文本的方法更高的精度和更好的复杂相机控制支持。然而，它们也引入了显著的可用性挑战，因为用户在处理任意真实世界图像时，往往难以提供精确的相机参数，而不知道其深度或场景尺度。为了解决这些真实世界应用问题，我们提出了RealCam-I2V，一种新颖的基于扩散的视频生成框架，在预处理步骤中整合单目度量深度估计以建立3D场景重建。在训练期间，重建的3D场景使得相机参数从相对尺度缩放到度量尺度，确保跨多样真实世界图像的兼容性和尺度一致性。在推理中，RealCam-I2V提供了一个直观界面，用户可以在3D场景内通过拖拽精确绘制相机轨迹。为了进一步增强精确相机控制和场景一致性，我们提出了场景约束噪声整形，该技术不仅塑造高级噪声，还允许框架在较低噪声阶段保持动态和一致视频生成。RealCam-I2V在RealEstate10K和域外图像上实现了可控性和视频质量的显著改进。我们进一步启用了应用，如相机控制循环视频生成和生成帧插值。
</details>

<details>
    <summary>Key points</summary>
    * 单目度量深度估计的整合
    * 3D场景重建用于尺度一致性
    * 场景约束噪声整形
    * 循环视频和帧插值的应用
</details>
</details>

---


<details>
<summary><b> TextOCVP: Object-Centric Video Prediction with Language Guidance</b></summary>

* **Authors:** Angel Villar-Corrales, Gjergj Plepi, Sven Behnke
* **arXiv ID:** 2502.11655
* **One-liner:** Proposed TextOCVP for object-centric video prediction guided by textual descriptions, improving controllability and robustness.
* **Published in:** arxiv (17 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.11655) | [[PDF]](https://arxiv.org/pdf/2502.11655) | [[Code]](https://github.com/angelvillar96/TextOCVP)

> **核心创新**
> 使用物体槽和文本条件变换器预测未来状态，具有结构化潜在空间。

<details>
    <summary>Abstract</summary>
    理解和预测未来场景状态对于自主代理在复杂环境中有效规划和行动至关重要。以物体为中心的模型，具有结构化潜在空间，在建模物体动态和预测未来场景状态方面显示出潜力，但往往难以扩展到简单合成数据集之外，并且难以整合外部指导，限制了它们在机器人学中的适用性。为了解决这些限制，我们提出了TextOCVP，一种以物体为中心的模型，用于由文本描述引导的视频预测。TextOCVP将观察到的场景解析为称为槽的物体表示，并利用文本条件变换器预测器来预测未来物体状态和视频帧。我们的方法联合建模物体动态和交互，同时整合文本指导，实现准确和可控的预测。TextOCVP的结构化潜在空间提供了对预测过程的更精确控制，在两个数据集上优于多个视频预测基线。此外，我们展示了结构化以物体为中心的表示提供了对新颖场景配置的优越鲁棒性，以及改进的可控性和可解释性，实现了更精确和可理解的预测。
</details>

<details>
    <summary>Key points</summary>
    * 以物体为中心的模型与槽表示
    * 文本条件变换器预测器
    * 结构化潜在空间用于精确控制
    * 预测中的优越鲁棒性和可解释性
</details>
</details>

---


<details>
<summary><b> Dynamic-I2V: Exploring Image-to-Video Generation Models via Multimodal LLM</b></summary>

* **Authors:** Peng Liu, Xiaoming Ren, Fengkai Liu, Qingsong Xie, Quanlong Zheng, Yanhao Zhang, Haonan Lu, Yujiu Yang
* **arXiv ID:** 2505.19901
* **One-liner:** Proposed Dynamic-I2V framework with MLLM integration for enhanced motion control and temporal coherence in I2V generation.
* **Published in:** arxiv (26 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.19901) | [[PDF]](https://arxiv.org/pdf/2505.19901) | [[Code]]()

> **核心创新**
> 集成多模态大语言模型（MLLMs）编码视觉和文本条件，用于扩散变换器，提高运动可控性和时间一致性，并引入DIVE基准进行动态质量评估。

<details>
    <summary>Abstract</summary>
    近期图像到视频（I2V）生成在常规场景中展现出良好性能，但这些方法在处理需要深入理解细微运动和复杂对象-动作关系的复杂场景时仍面临显著挑战。为解决这些挑战，我们提出了Dynamic-I2V，一个创新框架，集成多模态大语言模型（MLLMs）来联合编码视觉和文本条件，用于扩散变换器（DiT）架构。通过利用MLLMs的先进多模态理解能力，我们的模型显著提高了合成视频中的运动可控性和时间一致性。Dynamic-I2V的固有多模态性进一步支持多样条件输入的灵活适配，扩展了其在下游生成任务中的应用。通过系统分析，我们识别出现有I2V基准的一个关键局限：显著偏向于低动态视频，源于运动复杂性与视觉质量指标之间的不平衡。为解决这一评估差距，我们提出了DIVE——一个专门为I2V生成中全面动态质量测量设计的新评估基准。总之，广泛的定量和定性实验证实，Dynamic-I2V在图像到视频生成中达到了最先进性能，特别是在动态范围、可控性和质量方面，相比现有方法，在DIVE指标下分别实现了42.5%、7.9%和11.8%的显著提升。
</details>

<details>
    <summary>Key points</summary>
    * 使用MLLMs联合编码视觉和文本条件
    * 采用扩散变换器（DiT）架构
    * 引入DIVE基准进行评估
    * 在动态范围、可控性和质量方面实现显著提升
</details>
</details>

---


<details>
<summary><b> MotionPro: A Precise Motion Controller for Image-to-Video Generation</b></summary>

* **Authors:** Zhongwei Zhang, Fuchen Long, Zhaofan Qiu, Yingwei Pan, Wu Liu, Ting Yao, Tao Mei
* **arXiv ID:** 2505.20287
* **One-liner:** Introduced MotionPro for precise motion control in I2V generation using region-wise trajectories and motion masks.
* **Published in:** arxiv (26 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.20287) | [[PDF]](https://arxiv.org/pdf/2505.20287) | [[Code]](https://github.com/HiDream-ai/MotionPro)

> **核心创新**
> 利用区域轨迹和运动掩码调节细粒度运动合成并区分对象与相机移动，增强控制性和真实感。

<details>
    <summary>Abstract</summary>
    通过交互式运动控制对图像进行动画化在图像到视频（I2V）生成中日益流行。现代方法通常依赖大高斯核来扩展运动轨迹作为条件，而不明确定义移动区域，导致粗糙的运动控制，并无法分离对象和相机移动。为缓解这些问题，我们提出了MotionPro，一个精确的运动控制器，创新地利用区域轨迹和运动掩码分别调节细粒度运动合成和识别目标运动类别（即对象或相机移动）。技术上，MotionPro首先通过跟踪模型估计每个训练视频的流图，然后采样区域轨迹以模拟推理场景。与通过大高斯核扩展流不同，我们的区域轨迹方法通过直接利用局部区域内的轨迹实现更精确的控制，从而有效表征细粒度运动。同时，从预测流图中导出运动掩码以捕捉移动区域的整体运动动态。为实现自然运动控制，MotionPro通过特征调制进一步结合区域轨迹和运动掩码来增强视频去噪。更值得注意的是，我们精心构建了一个基准，即MC-Bench，包含1.1K用户标注的图像-轨迹对，用于评估细粒度和对象级I2V运动控制。在WebVid-10M和MC-Bench上进行的广泛实验证明了MotionPro的有效性。
</details>

<details>
    <summary>Key points</summary>
    * 通过跟踪模型估计流图
    * 使用区域轨迹进行精确控制
    * 导出运动掩码以捕捉整体动态
    * 通过特征调制结合轨迹和掩码进行去噪
</details>
</details>

---


<details>
<summary><b> Consistent Video Editing as Flow-Driven Image-to-Video Generation</b></summary>

* **Authors:** Ge Wang, Songlin Fan, Hangxu Liu, Quanjian Song, Hewei Wang, Jinfeng Xu
* **arXiv ID:** 2506.07713
* **One-liner:** Presented FlowV2V for video editing by re-investigating it as flow-driven I2V generation.
* **Published in:** arxiv (9 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.07713) | [[PDF]](https://arxiv.org/pdf/2506.07713) | [[Code]](https://github.com/stepfun-ai/Step-Video-Ti2V)

> **核心创新**
> 将视频编辑分解为第一帧编辑和条件I2V生成，使用伪流序列确保时间一致性并处理形状变形。

<details>
    <summary>Abstract</summary>
    随着视频扩散模型的繁荣，视频编辑等下游应用在不消耗大量计算成本的情况下得到了显著推动。该任务中的一个特定挑战在于从源视频到编辑视频的运动转移过程，需要考虑形状变形，同时保持生成视频序列的时间一致性。然而，现有方法未能建模复杂运动模式用于视频编辑，并基本局限于对象替换，其中涉及非刚性对象运动的任务（如多对象和肖像编辑）被大量忽视。在本文中，我们观察到光流在复杂运动建模中提供了一个有前景的替代方案，并提出了FlowV2V，将视频编辑重新视为流驱动的图像到视频（I2V）生成任务。具体来说，FlowV2V将整个流程分解为第一帧编辑和条件I2V生成，并模拟与变形形状对齐的伪流序列，从而确保编辑过程中的一致性。在DAVIS-EDIT上的实验结果显示，在DOVER和扭曲误差上分别提升了13.67%和50.66%，表明FlowV2V相比现有最先进方法在时间一致性和样本质量上的优越性。此外，我们进行了全面的消融研究，分析了第一帧范式和流对齐在提出方法中的内部功能。
</details>

<details>
    <summary>Key points</summary>
    * 分解为第一帧编辑和I2V生成
    * 模拟与形状变形对齐的伪流序列
    * 使用光流进行复杂运动建模
    * 在时间一致性和样本质量方面实现改进
</details>
</details>

---


<details>
<summary><b> Versatile Transition Generation with Image-to-Video Diffusion</b></summary>

* **Authors:** Zuhao Yang, Jiahui Zhang, Yingchen Yu, Shijian Lu, Song Bai
* **arXiv ID:** 2508.01698
* **One-liner:** Developed VTG framework for smooth and coherent transition video generation between frames.
* **Published in:** arxiv (3 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.01698) | [[PDF]](https://arxiv.org/pdf/2508.01698) | [[Code]](https://mwxely.github.io/projects/yang2025vtg/index)

> **核心创新**
> 引入基于插值的初始化、双向运动微调和表示对齐正则化，以增强过渡中的运动平滑性和保真度。

<details>
    <summary>Abstract</summary>
    利用文本、图像、结构图或运动轨迹作为条件指导，扩散模型在自动化和高质量视频生成方面取得了巨大成功。然而，给定第一帧和最后一帧视频帧以及描述性文本提示，生成平滑且合理的过渡视频仍远未充分探索。我们提出了VTG，一个多功能过渡视频生成框架，能够生成平滑、高保真和语义连贯的视频过渡。VTG引入了基于插值的初始化，帮助有效保留对象身份和处理突然内容变化。此外，它结合了双向运动微调和表示对齐正则化，分别缓解预训练图像到视频扩散模型在运动平滑性和生成保真度方面的限制。为评估VTG并促进未来统一过渡生成研究，我们收集了TransitBench，一个全面的过渡生成基准，涵盖两个代表性过渡任务：概念混合和场景过渡。广泛实验显示，VTG在所有四个任务中一致实现了优越的过渡性能。
</details>

<details>
    <summary>Key points</summary>
    * 基于插值的初始化用于身份保留
    * 双向运动微调
    * 表示对齐正则化
    * 引入TransitBench进行评估
</details>
</details>

---


<details>
<summary><b> Zero-shot 3D-Aware Trajectory-Guided image-to-video generation via Test-Time Training</b></summary>

* **Authors:** Ruicheng Zhang, Jun Zhou, Zunnan Xu, Zihao Liu, Jiehui Huang, Mingyang Zhang, Yu Sun, Xiu Li
* **arXiv ID:** 2509.06723
* **One-liner:** Proposed Zo3T, a zero-shot test-time-training framework for trajectory-guided I2V generation with 3D awareness.
* **Published in:** arxiv (8 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.06723) | [[PDF]](https://arxiv.org/pdf/2509.06723) | [[Code]]()

> **核心创新**
> 结合3D感知运动投影、轨迹引导测试时LoRA和指导场校正，以增强运动准确性和真实感，无需微调。

<details>
    <summary>Abstract</summary>
    轨迹引导的图像到视频（I2V）生成旨在合成符合用户指定运动指令的视频。现有方法通常依赖于在稀缺标注数据集上进行计算昂贵的微调。尽管一些零样本方法尝试在潜在空间中进行轨迹控制，但它们可能因忽略3D透视而产生不真实运动，并在操纵潜在状态与网络噪声预测之间造成不对齐。为解决这些挑战，我们引入了Zo3T，一个新颖的零样本测试时训练框架，用于轨迹引导生成，具有三个核心创新：首先，我们结合了3D感知运动投影，利用推断场景深度来推导目标区域的透视校正仿射变换。其次，我们引入了轨迹引导测试时LoRA，一种机制，在去噪网络中动态注入和优化临时LoRA适配器，与潜在状态一起。通过区域特征一致性损失的驱动，这种共适应有效强制执行运动约束，同时允许预训练模型局部调整其内部表示以适应操纵的潜在状态，从而确保生成保真度和流形一致性。最后，我们开发了指导场校正，通过一步前瞻策略优化条件指导场，精炼去噪演化路径，确保高效生成向目标轨迹的进展。Zo3T显著增强了轨迹控制I2V生成中的3D真实感和运动准确性，展示了优于现有基于训练和零样本方法的性能。
</details>

<details>
    <summary>Key points</summary>
    * 3D感知运动投影用于透视校正
    * 轨迹引导测试时LoRA用于动态适配
    * 指导场校正用于高效进展
    * 零样本方法结合区域特征一致性
</details>
</details>

---


<details>
<summary><b> Vid-Freeze: Protecting Images from Malicious Image-to-Video Generation via Temporal Freezing</b></summary>

* **Authors:** Rohit Chowdhury, Aniruddha Bala, Rohan Jaiswal, Siddharth Roheda
* **arXiv ID:** 2509.23279
* **One-liner:** Introduced Vid-Freeze, an adversarial attack to block motion synthesis in I2V models while preserving image semantics.
* **Published in:** arxiv (27 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.23279) | [[PDF]](https://arxiv.org/pdf/2509.23279) | [[Code]]()

> **核心创新**
> 使用注意力抑制对抗扰动破坏运动合成，生成静态视频以防止I2V生成的恶意使用。

<details>
    <summary>Abstract</summary>
    图像到视频（I2V）生成模型的快速进展引入了显著风险，使得从静态图像合成视频成为可能，并促进了欺骗性或恶意内容的创建。尽管先前的防御如I2VGuard尝试免疫图像，但有效且原则性的保护以阻止运动仍远未充分探索。在这项工作中，我们引入了Vid-Freeze——一种新颖的注意力抑制对抗攻击，向图像添加精心设计的对抗扰动。我们的方法明确针对I2V模型的注意力机制，完全破坏运动合成，同时保留输入图像的语义保真度。由此免疫的图像生成静止或近静态视频，有效阻止恶意内容创建。我们的实验证明了提出方法提供的令人印象深刻的保护，突出了注意力攻击作为针对I2V生成模型滥用的鲁棒和主动防御的有前景方向的重要性。
</details>

<details>
    <summary>Key points</summary>
    * 针对I2V模型的注意力机制
    * 添加对抗扰动
    * 保留语义保真度
    * 评估显示有效保护
</details>
</details>

---


<details>
<summary><b> MotionRAG: Motion Retrieval-Augmented Image-to-Video Generation</b></summary>

* **Authors:** Chenhui Zhu, Yilu Wu, Shuai Wang, Gangshan Wu, Limin Wang
* **arXiv ID:** 2509.26391
* **One-liner:** Proposed MotionRAG, a retrieval-augmented framework for enhancing motion realism in video generation.
* **Published in:** arxiv (30 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.26391) | [[PDF]](https://arxiv.org/pdf/2509.26391) | [[Code]](https://github.com/MCG-NJU/MotionRAG)

> **核心创新**
> 通过上下文感知运动适应（CAMA）从参考视频中适配运动先验，具有检索基于管道和基于注意力的注入，实现零样本泛化。

<details>
    <summary>Abstract</summary>
    图像到视频生成随着扩散模型的进步取得了显著进展，但生成具有真实运动的视频仍然极具挑战性。这一困难源于准确建模运动的复杂性，涉及捕捉物理约束、对象交互和领域特定动态，这些不易在多样场景中泛化。为解决这一问题，我们提出了MotionRAG，一个检索增强框架，通过上下文感知运动适应（CAMA）从相关参考视频中适配运动先验，以增强运动真实感。关键技术创新包括：（i）一个检索基于的管道，使用视频编码器和专门重采样器提取高级运动特征，以蒸馏语义运动表示；（ii）一种上下文学习方法用于运动适应，通过因果变换器架构实现；（iii）一个基于注意力的运动注入适配器，无缝将转移的运动特征集成到预训练视频扩散模型中。广泛实验证明，我们的方法在多个领域和各种基础模型上实现了显著改进，所有在推理过程中计算开销可忽略。此外，我们的模块化设计通过简单更新检索数据库而不重新训练任何组件，实现了对新领域的零样本泛化。这项研究通过有效检索和转移运动先验，促进了真实运动动态的合成，增强了视频生成系统的核心能力。
</details>

<details>
    <summary>Key points</summary>
    * 检索基于管道用于运动特征
    * 上下文感知运动适应（CAMA）与因果变换器
    * 基于注意力的运动注入适配器
    * 模块化设计实现零样本泛化
</details>
</details>

---


<details>
<summary><b> Moonshot: Towards Controllable Video Generation and Editing with Multimodal Conditions</b></summary>

* **Authors:** David Junhao Zhang, Dongxu Li, Hung Le, Mike Zheng Shou, Caiming Xiong, Doyen Sahoo
* **arXiv ID:** 2401.01827
* **One-liner:** Presented Moonshot, a video generation model with multimodal conditioning on image and text inputs.
* **Published in:** arxiv (3 Jan 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2401.01827) | [[PDF]](https://arxiv.org/pdf/2401.01827) | [[Code]](https://github.com/salesforce/LAVIS)

> **核心创新**
> 使用多模态视频块（MVB）与解耦交叉注意力进行外观调节，并可选集成ControlNet用于几何，改进视觉质量和一致性。

<details>
    <summary>Abstract</summary>
    大多数现有视频扩散模型（VDMs）仅限于文本条件。因此，它们通常在控制生成视频的视觉外观和几何结构方面不足。这项工作提出了Moonshot，一个新的视频生成模型，同时以图像和文本的多模态输入为条件。该模型基于一个核心模块，称为多模态视频块（MVB），它由常规时空层组成，用于表示视频特征，以及一个解耦交叉注意力层来处理图像和文本输入以进行外观调节。此外，我们精心设计了模型架构，使其可以选择性地与预训练图像ControlNet模块集成，用于几何视觉条件，而不需要像先前方法那样的额外训练开销。实验显示，通过多功能多模态调节机制，Moonshot在视觉质量和时间一致性方面相比现有模型实现了显著改进。此外，该模型可以轻松重新用于各种生成应用，如个性化视频生成、图像动画和视频编辑，揭示了其作为可控视频生成基础架构的潜力。
</details>

<details>
    <summary>Key points</summary>
    * 多模态视频块（MVB）与时空层
    * 解耦交叉注意力用于图像和文本输入
    * 可选ControlNet集成用于几何
    * 多功能应用在个性化和编辑中
</details>
</details>

---


<details>
<summary><b> LAVE: LLM-Powered Agent Assistance and Language Augmentation for Video Editing</b></summary>

* **Authors:** Bryan Wang, Yuliang Li, Zhaoyang Lv, Haijun Xia, Yan Xu, Raj Sodhi
* **arXiv ID:** 2402.10294
* **One-liner:** Developed LAVE, an LLM-powered system to assist in video editing for beginners.
* **Published in:** arxiv (15 Feb 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2402.10294) | [[PDF]](https://arxiv.org/pdf/2402.10294) | [[Code]]()

> **核心创新**
> 自动生成视频素材的语言描述，使LLM代理能够基于用户目标计划和执行编辑任务，并支持手动精炼。

<details>
    <summary>Abstract</summary>
    视频创作日益流行，但编辑所需的专业知识和努力通常对初学者构成障碍。在本文中，我们探索将大语言模型（LLMs）集成到视频编辑工作流中以减少这些障碍。我们的设计愿景体现在LAVE中，一个新颖系统，提供LLM驱动的代理辅助和语言增强编辑功能。LAVE自动为用户素材生成语言描述，作为使LLM能够处理视频并协助编辑任务的基础。当用户提供编辑目标时，代理计划并执行相关操作以实现它们。此外，LAVE允许用户通过代理或直接UI操作编辑视频，提供灵活性并支持手动精炼代理操作。我们的用户研究包括从新手到熟练编辑者的八名参与者，证明了LAVE的有效性。结果还揭示了用户对提出的LLM辅助编辑范式的看法及其对用户创造力和共同创造感的影响。基于这些发现，我们提出了设计启示，以指导未来代理辅助内容编辑的发展。
</details>

<details>
    <summary>Key points</summary>
    * 自动生成视频语言描述
    * LLM代理用于计划和执行编辑
    * 支持代理和直接UI操作
    * 用户研究显示有效性和共同创造
</details>
</details>

---


<details>
<summary><b> ExpressEdit: Video Editing with Natural Language and Sketching</b></summary>

* **Authors:** Bekzat Tilekbay, Saelyne Yang, Michal Lewkowicz, Alex Suryapranata, Juho Kim
* **arXiv ID:** 2403.17693
* **One-liner:** Introduced ExpressEdit, a multimodal system for video editing using natural language and sketching.
* **Published in:** arxiv (26 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.17693) | [[PDF]](https://arxiv.org/pdf/2403.17693) | [[Code]]()

> **核心创新**
> 解释NL命令和草图中的时间、空间和操作引用，迭代实施编辑以增强新手表达和效率。

<details>
    <summary>Abstract</summary>
    信息视频作为向新手和专家解释概念和程序知识的关键来源。在制作信息视频时，编辑者通过叠加文本/图像或修剪素材来编辑视频，以增强视频质量并使其更具吸引力。然而，视频编辑可能困难且耗时，特别是对于新手视频编辑者，他们常常在表达和实施编辑想法上挣扎。为应对这一挑战，我们首先探索了多模态——自然语言（NL）和草图，这是人类表达的自然模态——如何用于支持视频编辑者表达视频编辑想法。我们收集了10名视频编辑者的176个多模态编辑命令表达，揭示了在描述编辑意图时NL和草图的使用模式。基于这些发现，我们提出了ExpressEdit，一个系统，通过NL文本和在视频帧上草图来编辑视频。由LLM和视觉模型驱动，该系统解释（1）时间、（2）空间和（3）操作引用在NL命令中，以及从草图中的空间引用。系统实施解释的编辑，然后用户可以迭代。一项观察研究（N=10）显示，ExpressEdit增强了新手视频编辑者表达和实施编辑想法的能力。该系统通过基于用户多模态编辑命令生成编辑并支持编辑命令的迭代，使参与者更高效地执行编辑并生成更多想法。这项工作为未来多模态界面和基于AI的视频编辑管道设计提供了见解。
</details>

<details>
    <summary>Key points</summary>
    * 解释NL和草图用于编辑命令
    * 处理时间、空间和操作引用
    * 迭代实施编辑
    * 观察研究显示改进效率和想法生成
</details>
</details>

---


<details>
<summary><b> GenVideo: One-shot Target-image and Shape Aware Video Editing using T2I Diffusion Models</b></summary>

* **Authors:** Sai Sree Harsha, Ambareesh Revanur, Dhwanit Agarwal, Shradha Agrawal
* **arXiv ID:** 2404.12541
* **One-liner:** Proposed GenVideo for precise video editing using target images, handling objects of varying shapes and sizes.
* **Published in:** arxiv (18 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.12541) | [[PDF]](https://arxiv.org/pdf/2404.12541) | [[Code]]()

> **核心创新**
> 利用目标图像感知的T2I模型，结合新颖的InvEdit掩码和潜在噪声校正，确保时间一致性。

<details>
    <summary>Abstract</summary>
    基于扩散模型的视频编辑方法仅依赖文本提示进行编辑，受限于文本提示的表达能力不足。因此，引入参考目标图像作为视觉引导，以实现对编辑的精确控制。此外，大多数现有方法在目标图像中对象的形状和大小与源对象不同时，难以准确编辑视频。为解决这些挑战，我们提出'GenVideo'，利用目标图像感知的T2I模型编辑视频。我们的方法处理不同形状和大小的目标对象编辑，同时通过新颖的目标和形状感知InvEdit掩码保持编辑的时间一致性。进一步，我们提出一种新颖的目标图像感知潜在噪声校正策略，在推理过程中提高编辑的时间一致性。实验分析表明，GenVideo能有效处理形状变化的对象编辑，而现有方法在此方面失败。
</details>

<details>
    <summary>Key points</summary>
    * 使用目标图像感知的T2I模型
    * 采用目标和形状感知的InvEdit掩码
    * 实施潜在噪声校正策略
</details>
</details>

---


<details>
<summary><b> Edit-Your-Motion: Space-Time Diffusion Decoupling Learning for Video Motion Editing</b></summary>

* **Authors:** Yi Zuo, Lingling Li, Licheng Jiao, Fang Liu, Xu Liu, Wenping Ma, Shuyuan Yang, Yuwei Guo
* **arXiv ID:** 2405.04496
* **One-liner:** Introduced Edit-Your-Motion for human motion editing with reduced ghosting and distortion in unseen cases.
* **Published in:** arxiv (7 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.04496) | [[PDF]](https://arxiv.org/pdf/2405.04496) | [[Code]]()

> **核心创新**
> 利用一次性微调、DDIM反演、运动注意力适配器和时空学习策略。

<details>
    <summary>Abstract</summary>
    现有基于扩散的方法在人体运动编辑方面取得了显著成果。然而，这些方法在未见过的野外案例中常出现显著的鬼影和身体扭曲。本文中，我们介绍Edit-Your-Motion，一种视频运动编辑方法，通过对未见案例进行一次性微调来解决这些挑战。具体而言，首先，我们利用DDIM反演初始化噪声，保留源视频的外观，并设计了一个轻量级运动注意力适配器模块以增强运动保真度。DDIM反演旨在通过估计源视频的预测噪声来获取隐式表示，作为采样过程的起点，确保源视频和编辑视频之间的外观一致性。运动注意力模块通过解决骨架特征和外观特征之间的冲突来增强模型的运动编辑能力。其次，为了有效解耦源视频的运动和外观，我们设计了一个时空两阶段学习策略。在第一阶段，我们专注于学习人体运动的时间特征，并提出循环因果注意力以确保视频帧之间的一致性。在第二阶段，我们将焦点转移到学习源视频的外观特征。通过Edit-Your-Motion，用户可以编辑源视频中的人体运动，创造更具吸引力和多样性的内容。广泛的定性和定量实验，以及用户偏好研究，显示Edit-Your-Motion优于其他方法。
</details>

<details>
    <summary>Key points</summary>
    * 采用DDIM反演进行噪声初始化
    * 使用运动注意力适配器模块
    * 实施时空两阶段学习
</details>
</details>

---


<details>
<summary><b> Learning Action and Reasoning-Centric Image Editing from Videos and Simulations</b></summary>

* **Authors:** Benno Krojer, Dheeraj Vattikonda, Luis Lara, Varun Jampani, Eva Portelance, Christopher Pal, Siva Reddy
* **arXiv ID:** 2407.03471
* **One-liner:** Curated AURORA Dataset and developed a model for diverse image edits, excelling in action and reasoning tasks.
* **Published in:** arxiv (3 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.03471) | [[PDF]](https://arxiv.org/pdf/2407.03471) | [[Code]]()

> **核心创新**
> 专注于高质量数据，具有最小变化，并提出新的自动评估指标。

<details>
    <summary>Abstract</summary>
    图像编辑模型应能执行多样化的编辑，包括对象替换、改变属性或风格、执行动作或移动，这需要多种形式的推理。当前通用的指令引导编辑模型在动作和推理中心编辑方面存在显著不足。对象、属性或风格变化可以从视觉静态数据集中学习。另一方面，高质量的动作和推理中心编辑数据稀缺，必须来自完全不同的来源，涵盖物理动态、时间性和空间推理。为此，我们精心策划了AURORA数据集，一个高质量的训练数据集合，从视频和模拟引擎中人工标注和筛选。我们关注高质量训练数据的一个关键方面：三元组包含由提示描述的单一有意义的视觉变化，即源图像和目标图像之间真正的最小变化。为展示我们数据集的价值，我们在一个新的专家策划基准上评估AURORA微调模型，覆盖8种多样化编辑任务。我们的模型在人类评分者判断下显著优于先前编辑模型。对于自动评估，我们发现先前指标存在重要缺陷，并警告其在语义困难编辑任务中的使用。相反，我们提出一个新的自动指标，专注于判别性理解。我们希望我们的努力：策划高质量训练数据集和评估基准、开发关键评估以及发布最先进模型，将推动通用图像编辑的进一步进展。
</details>

<details>
    <summary>Key points</summary>
    * 策划AURORA数据集，包含人工标注的三元组
    * 在AURORA上微调模型以进行多样化编辑
    * 提出判别性自动指标
</details>
</details>

---


<details>
<summary><b> A Reinforcement Learning-Based Automatic Video Editing Method Using Pre-trained Vision-Language Model</b></summary>

* **Authors:** Panwen Hu, Nan Xiao, Feifei Li, Yongquan Chen, Rui Huang
* **arXiv ID:** 2411.04942
* **One-liner:** Proposed a two-stage scheme for general video editing using VLM-based context and RL-based framework.
* **Published in:** arxiv (7 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.04942) | [[PDF]](https://arxiv.org/pdf/2411.04942) | [[Code]]()

> **核心创新**
> 利用预训练VLM提取编辑上下文，并使用强化学习在通用场景中进行顺序决策。

<details>
    <summary>Abstract</summary>
    在这个视频时代，自动视频编辑技术吸引了越来越多工业和学术界的关注，因为它们可以减少工作量并降低对人类编辑的要求。现有自动编辑系统主要是场景或事件特定的，例如足球比赛广播，但通用编辑的自动系统，如覆盖各种场景和事件的电影或vlog编辑，之前很少被研究，将事件驱动编辑方法转换为通用场景并非易事。本文中，我们提出一个两阶段方案用于通用编辑。首先，与先前提取场景特定特征的工作不同，我们利用预训练的视觉语言模型提取编辑相关表示作为编辑上下文。此外，为缩小专业外观视频与简单指南生成的自动产品之间的差距，我们提出一个基于强化学习的编辑框架来制定编辑问题并训练虚拟编辑器做出更好的顺序编辑决策。最后，我们在一个更通用的编辑任务上使用真实电影数据集评估所提方法。实验结果证明了所提上下文表示的有效性和益处，以及我们基于强化学习的编辑框架的学习能力。
</details>

<details>
    <summary>Key points</summary>
    * 使用视觉语言模型进行上下文提取
    * 实施强化学习框架
    * 在真实电影数据集上评估
</details>
</details>

---


<details>
<summary><b> VideoDirector: Precise Video Editing via Text-to-Video Models</b></summary>

* **Authors:** Yukun Wang, Longguang Wang, Zhiyuan Ma, Qibin Hu, Kai Xu, Yulan Guo
* **arXiv ID:** 2411.17592
* **One-liner:** Developed VideoDirector to harness T2V models for video editing with improved temporal coherence.
* **Published in:** arxiv (26 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.17592) | [[PDF]](https://arxiv.org/pdf/2411.17592) | [[Code]](https://github.com/Yukun66/Video_Director)

> **核心创新**
> 引入时空解耦指导和多帧空文本优化以实现精确反演。

<details>
    <summary>Abstract</summary>
    尽管使用文本到图像模型的典型反演-然后-编辑范式已展示出有希望的结果，但直接将其扩展到文本到视频模型仍遭受严重伪影，如颜色闪烁和内容扭曲。因此，当前视频编辑方法主要依赖T2I模型，这些模型本质上缺乏时间一致性生成能力，常导致编辑结果不佳。本文中，我们将典型编辑范式的失败归因于：1）紧密的时空耦合。基于关键点的反演策略难以解耦视频扩散模型中的时空信息；2）复杂的时空布局。普通交叉注意力控制在保留未编辑内容方面不足。为解决这些限制，我们提出时空解耦指导和多帧空文本优化策略，为更精确的关键点反演提供关键时间线索。此外，我们引入自注意力控制策略，以保持更高保真度，实现精确部分内容编辑。实验结果表明，我们的方法有效利用T2V模型的强大时间生成能力，在准确性、运动平滑度、真实性和未编辑内容保真度方面产生最先进性能的编辑视频。
</details>

<details>
    <summary>Key points</summary>
    * 提出时空解耦指导
    * 使用多帧空文本优化
    * 实施自注意力控制
</details>
</details>

---


<details>
<summary><b> SPAgent: Adaptive Task Decomposition and Model Selection for General Video Generation and Editing</b></summary>

* **Authors:** Rong-Cheng Tu, Wenhao Sun, Zhao Jin, Jingyi Liao, Jiaxing Huang, Dacheng Tao
* **arXiv ID:** 2411.18983
* **One-liner:** Created SPAgent system for coordinating multiple models in video generation and editing tasks.
* **Published in:** arxiv (28 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.18983) | [[PDF]](https://arxiv.org/pdf/2411.18983) | [[Code]]()

> **核心创新**
> 集成工具库，通过意图识别、路径规划和模型选择实现自动协调。

<details>
    <summary>Abstract</summary>
    尽管开源视频生成和编辑模型取得了显著进展，但单个模型通常限于特定任务，无法满足用户的多样化需求。有效协调这些模型可以解锁广泛的视频生成和编辑能力。然而，手动协调复杂且耗时，要求用户深入理解任务需求并具备每个模型性能、适用性和局限性的全面知识，从而增加入门门槛。为解决这些挑战，我们提出一个新颖的视频生成和编辑系统，由我们的语义规划代理驱动。SPAgent弥合了多样化用户意图与现有生成模型有效利用之间的差距，增强了视频生成和编辑的适应性、效率和整体质量。具体而言，SPAgent集成一个工具库，整合最先进的开源图像和视频生成和编辑模型作为工具。在我们手动标注的数据集上微调后，SPAgent可以通过我们新颖设计的三步框架自动协调工具进行视频生成和编辑：解耦意图识别、原则引导的路径规划和基于能力的执行模型选择。此外，我们增强SPAgent的视频质量评估能力，使其能够自主评估并无需人工干预将新视频生成和编辑模型纳入其工具库。实验结果表明，SPAgent有效协调模型生成或编辑视频，突显其在各种视频任务中的多功能性和适应性。
</details>

<details>
    <summary>Key points</summary>
    * 将模型集成到工具库中
    * 使用三步框架进行协调
    * 增强视频质量评估能力
</details>
</details>

---


<details>
<summary><b> DIVE: Taming DINO for Subject-Driven Video Editing</b></summary>

* **Authors:** Yi Huang, Wei Xiong, He Zhang, Chaoqi Chen, Jianzhuang Liu, Mingfu Yan, Shifeng Chen
* **arXiv ID:** 2412.03347
* **One-liner:** Proposed DIVE for subject-driven video editing using DINOv2 features for motion consistency and identity registration.
* **Published in:** arxiv (4 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.03347) | [[PDF]](https://arxiv.org/pdf/2412.03347) | [[Code]]()

> **核心创新**
> 利用DINO特征进行运动对齐，并使用LoRAs进行主体身份学习。

<details>
    <summary>Abstract</summary>
    基于扩散模型在图像生成和编辑中的成功，视频编辑最近获得了大量关注。然而，保持时间一致性和运动对齐仍然具有挑战性。为解决这些问题，本文提出DINO引导的视频编辑，一个框架设计用于在源视频中基于目标文本提示或具有特定身份的参考图像进行主体驱动编辑。DIVE的核心在于利用从预训练DINOv2模型中提取的强大语义特征作为隐式对应关系来指导编辑过程。具体而言，为确保时间运动一致性，DIVE使用DINO特征与源视频的运动轨迹对齐。为精确主体编辑，DIVE将参考图像的DINO特征整合到预训练文本到图像模型中，学习低秩适应，有效注册目标主体的身份。在多样化真实世界视频上的广泛实验表明，我们的框架可以实现高质量编辑结果，具有鲁棒的运动一致性，突显DINO对视频编辑的潜在贡献。
</details>

<details>
    <summary>Key points</summary>
    * 使用DINOv2特征进行语义指导
    * 采用LoRAs进行身份注册
    * 确保时间运动一致性
</details>
</details>

---


<details>
<summary><b> Re-Attentional Controllable Video Diffusion Editing</b></summary>

* **Authors:** Yuanzhi Wang, Yong Li, Mengyi Liu, Xiaoya Zhang, Xin Liu, Zhen Cui, Antoni B. Chan
* **arXiv ID:** 2412.11710
* **One-liner:** Introduced ReAtCo method for controllable video editing with improved spatial alignment and artifact reduction.
* **Published in:** arxiv (16 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.11710) | [[PDF]](https://arxiv.org/pdf/2412.11710) | [[Code]](https://github.com/mdswyz/ReAtCo)

> **核心创新**
> 使用重新注意力扩散和不变区域引导联合采样以保持保真度和一致性。

<details>
    <summary>Abstract</summary>
    使用文本指导编辑视频因其简化过程而广受欢迎，该过程仅要求用户编辑与源视频对应的文本提示。最近研究探索并利用大规模文本到图像扩散模型进行文本引导视频编辑，产生了显著的视频编辑能力。然而，它们仍可能遭受一些限制，如对象位置错误、对象数量不正确。因此，视频编辑的可控性仍然是一个严峻挑战。本文中，我们旨在通过提出重新注意力可控视频扩散编辑方法来挑战上述限制。特别地，为在无需训练的方式下将目标对象的空间放置与编辑文本提示对齐，我们提出重新注意力扩散，在去噪阶段重新聚焦编辑文本提示和目标视频之间的交叉注意力激活响应，产生空间位置对齐和语义高保真度的操纵视频。特别是，为忠实保留不变区域内容并减少边界伪影，我们提出不变区域引导联合采样策略，以减轻每个去噪时间步不变区域的内在采样误差，并约束生成内容与不变区域内容和谐。实验结果验证，ReAtCo持续提高视频扩散编辑的可控性，并实现优越的视频编辑性能。
</details>

<details>
    <summary>Key points</summary>
    * 实施重新注意力扩散
    * 使用不变区域引导联合采样
    * 重新聚焦交叉注意力激活
</details>
</details>

---


<details>
<summary><b> Edit as You See: Image-guided Video Editing via Masked Motion Modeling</b></summary>

* **Authors:** Zhi-Lin Huang, Yixuan Liu, Chujun Qin, Zhongdao Wang, Dong Zhou, Dong Li, Emad Barsoum
* **arXiv ID:** 2501.04325
* **One-liner:** Developed IVEDiff for image-guided video editing with learnable motion modules and optical flow guidance.
* **Published in:** arxiv (8 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.04325) | [[PDF]](https://arxiv.org/pdf/2501.04325) | [[Code]]()

> **核心创新**
> 结合掩码运动建模和运动参考网络以保持时间一致性。

<details>
    <summary>Abstract</summary>
    扩散模型的最近进展显著促进了文本引导视频编辑。然而，关于图像引导视频编辑的研究相对较少，该方法使用户仅通过在初始帧指示目标对象并提供RGB图像作为参考来编辑视频，而无需依赖文本提示。本文中，我们提出一个新颖的图像引导视频编辑扩散模型，称为IVEDiff。IVEDiff构建在图像编辑模型之上，并配备可学习运动模块以保持编辑视频的时间一致性。受自监督学习概念启发，我们引入掩码运动建模微调策略，增强运动模块捕捉帧间运动动态的能力，同时保留基础图像编辑模型的帧内语义相关性建模能力。此外，提出光流引导运动参考网络，确保编辑视频帧之间信息的准确传播，减轻无效信息的误导效应。我们还构建一个基准以促进进一步研究。综合实验表明，我们的方法能够生成时间平滑的编辑视频，同时鲁棒处理各种编辑对象，具有高质量。
</details>

<details>
    <summary>Key points</summary>
    * 构建于图像编辑模型，配备运动模块
    * 使用掩码运动建模微调
    * 实施光流引导运动参考
</details>
</details>

---


<details>
<summary><b> VideoPainter: Any-length Video Inpainting and Editing with Plug-and-Play Context Control</b></summary>

* **Authors:** Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, Qiang Xu
* **arXiv ID:** 2503.05639
* **One-liner:** Proposed VideoPainter for video inpainting with dual-stream paradigm and any-length capability.
* **Published in:** arxiv (7 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.05639) | [[PDF]](https://arxiv.org/pdf/2503.05639) | [[Code]](https://github.com/TencentARC/VideoPainter)

> **核心创新**
> 使用上下文编码器和目标区域ID重采样，并建立VPData和VPBench。

<details>
    <summary>Abstract</summary>
    视频修复旨在恢复损坏的视频内容，已取得实质性进展。尽管有这些进步，现有方法，无论是通过光流和感受野先验传播未掩码区域像素，还是将图像修复模型扩展到时间维度，都面临生成完全掩码对象或在一个模型中平衡背景上下文保留和前景生成的竞争目标的挑战。为解决这些限制，我们提出一种新颖的双流范式VideoPainter，集成一个高效上下文编码器处理掩码视频，并将骨干感知背景上下文线索注入任何预训练视频DiT，以即插即用方式产生语义一致内容。这种架构分离显著降低模型学习复杂性，同时实现关键背景上下文的细致整合。我们还引入一种新颖的目标区域ID重采样技术，实现任意长度视频修复，大大增强我们的实际适用性。此外，我们建立一个可扩展数据集管道，利用当前视觉理解模型，贡献VPData和VPBench以促进基于分割的修复训练和评估，这是迄今为止最大的视频修复数据集和基准，拥有超过390K多样化片段。使用修复作为管道基础，我们还探索下游应用，包括视频编辑和视频编辑对数据生成，展示竞争性能和显著实际潜力。广泛实验证明VideoPainter在任意长度视频修复和编辑中的优越性能，跨越八个关键指标，包括视频质量、掩码区域保留和文本一致性。
</details>

<details>
    <summary>Key points</summary>
    * 实施双流范式，配备上下文编码器
    * 使用目标区域ID重采样
    * 建立VPData和VPBench数据集
</details>
</details>

---


<details>
<summary><b> VACE: All-in-One Video Creation and Editing</b></summary>

* **Authors:** Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, Yu Liu
* **arXiv ID:** 2503.07598
* **One-liner:** VACE enables unified video creation and editing in an all-in-one framework.
* **Published in:** arxiv (10 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.07598) | [[PDF]](https://arxiv.org/pdf/2503.07598) | [[Code]](https://github.com/ali-vilab/VACE)

> **核心创新**
> 开发了一个使用视频条件单元和上下文适配器的统一模型，用于多种视频任务。

<details>
    <summary>Abstract</summary>
    扩散变换器在生成高质量图像和视频方面展现了强大的能力和可扩展性。进一步追求生成与编辑任务的统一在图像内容创作领域取得了显著进展。然而，由于时空动态一致性的内在需求，实现视频合成的统一方法仍然具有挑战性。我们引入了VACE，它使用户能够在全功能框架中执行视频创作和编辑任务。这些任务包括参考到视频生成、视频到视频编辑和掩码视频到视频编辑。具体而言，我们通过将视频任务输入（如编辑、参考和掩码）组织成一个统一接口（称为视频条件单元VCU），有效整合了各种任务的需求。此外，通过利用上下文适配器结构，我们使用时空维度的形式化表示将不同任务概念注入模型，使其能够灵活处理任意视频合成任务。大量实验表明，VACE的统一模型在各种子任务上实现了与任务特定模型相当的性能，同时通过多功能任务组合实现了多样化应用。项目页面：<a href="https://ali-vilab.github.io/VACE-Page/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 将视频任务输入组织成统一接口（VCU）
    * 使用上下文适配器注入任务概念，并使用时空表示
    * 实现了与任务特定模型相当的性能
</details>
</details>

---


<details>
<summary><b> VEGGIE: Instructional Editing and Reasoning Video Concepts with Grounded Generation</b></summary>

* **Authors:** Shoubin Yu, Difan Liu, Ziqiao Ma, Yicong Hong, Yang Zhou, Hao Tan, Joyce Chai, Mohit Bansal
* **arXiv ID:** 2503.14350
* **One-liner:** VEGGIE unifies video editing, grounding, and reasoning based on user instructions.
* **Published in:** arxiv (18 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.14350) | [[PDF]](https://arxiv.org/pdf/2503.14350) | [[Code]](https://github.com/Yui010206/VEGGIE-VidEdit/)

> **核心创新**
> 引入了一个端到端框架，使用MLLM进行意图解释和扩散模型进行视频渲染。

<details>
    <summary>Abstract</summary>
    最近的视频扩散模型提升了视频编辑能力，但在统一框架内处理指令编辑和多样化任务（例如添加、移除、更改）仍然具有挑战性。在本文中，我们引入了VEGGIE，一种基于指令的接地生成视频编辑器，这是一个简单的端到端框架，统一了基于多样化用户指令的视频概念编辑、接地和推理。具体而言，给定视频和文本查询，VEGGIE首先利用多模态大语言模型（MLLM）解释指令中的用户意图，并将其接地到视频上下文，生成帧特定的接地任务查询以用于像素空间响应。然后，扩散模型渲染这些计划并生成与用户意图一致的编辑视频。为了支持多样化任务和复杂指令，我们采用课程学习策略：首先使用大规模指令图像编辑数据对齐MLLM和视频扩散模型，然后在高质量多任务视频数据上进行端到端微调。此外，我们引入了一种新颖的数据合成流程，通过利用图像到视频模型注入动态性，将静态图像数据转化为多样化、高质量的视频编辑样本，以生成配对的指令视频编辑数据用于模型训练。VEGGIE在不同编辑技能的指令视频编辑中表现出强大性能，作为多功能模型优于最佳指令基线，而其他模型在多任务处理上存在困难。VEGGIE在视频对象接地和推理分割方面也表现出色，而其他基线则失败。我们进一步揭示了多任务如何相互促进，并突出了有前景的应用，如零样本多模态指令和上下文视频编辑。
</details>

<details>
    <summary>Key points</summary>
    * 使用MLLM解释指令并将其接地到视频上下文
    * 采用课程学习，结合图像编辑数据和视频微调
    * 开发了指令视频编辑数据的数据合成流程
</details>
</details>

---


<details>
<summary><b> InstructVEdit: A Holistic Approach for Instructional Video Editing</b></summary>

* **Authors:** Chi Zhang, Chengjian Feng, Feng Yan, Qiming Zhang, Mingjin Zhang, Yujie Zhong, Jing Zhang, Lin Ma
* **arXiv ID:** 2503.17641
* **One-liner:** InstructVEdit provides a full-cycle approach for instructional video editing with improved data and model strategies.
* **Published in:** arxiv (22 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.17641) | [[PDF]](https://arxiv.org/pdf/2503.17641) | [[Code]](https://o937-blip.github.io/InstructVEdit/)

> **核心创新**
> 建立了数据集整理工作流和模型改进，以提升编辑质量和时间一致性。

<details>
    <summary>Abstract</summary>
    基于指令的视频编辑是一项极具挑战性的任务，因为难以收集大规模、高质量的编辑视频对数据。这种稀缺性不仅限制了训练数据的可用性，还阻碍了模型架构和训练策略的系统性探索。尽管先前工作改进了视频编辑的特定方面（例如使用图像编辑技术合成视频数据集或分解视频编辑训练），但解决上述挑战的整体框架仍未充分探索。在本研究中，我们引入了InstructVEdit，一种全周期指令视频编辑方法，它：（1）建立可靠的数据集整理工作流以初始化训练，（2）整合两种模型架构改进以增强编辑质量同时保持时间一致性，（3）提出一种迭代优化策略，利用真实世界数据增强泛化能力并最小化训练-测试差异。大量实验表明，InstructVEdit在基于指令的视频编辑中实现了最先进的性能，展现出对多样化真实世界场景的强大适应性。项目页面：<a href="https://o937-blip.github.io/InstructVEdit" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 创建了可靠的数据集整理工作流
    * 整合了架构改进以增强编辑质量和时间一致性
    * 提出了基于真实世界数据的迭代优化策略
</details>
</details>

---


<details>
<summary><b> OmniV2V: Versatile Video Generation and Editing via Dynamic Content Manipulation</b></summary>

* **Authors:** Sen Liang, Zhentao Yu, Zhengguang Zhou, Teng Hu, Hongmei Wang, Yi Chen, Qin Lin, Yuan Zhou, Xin Li, Qinglin Lu, Zhibo Chen
* **arXiv ID:** 2506.01801
* **One-liner:** OmniV2V enables diverse video generation and editing across multiple scenarios.
* **Published in:** arxiv (2 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.01801) | [[PDF]](https://arxiv.org/pdf/2506.01801) | [[Code]]()

> **核心创新**
> 设计了一个具有动态内容操作和视觉-文本指令模块的统一模型。

<details>
    <summary>Abstract</summary>
    扩散变换器（DiT）的出现为视频生成带来了显著进展，尤其是在文本到视频和图像到视频任务中。尽管视频生成广泛应用于各种领域，但大多数现有模型局限于单一场景，无法通过动态内容操作执行多样化视频生成和编辑。我们提出了OmniV2V，一种能够基于各种操作在不同场景中生成和编辑视频的模型，包括：对象移动、对象添加、掩码引导视频编辑、试穿、修复、扩展、人体动画和可控角色视频合成。我们探索了一个统一的动态内容操作注入模块，有效整合了上述任务的需求。此外，我们设计了一个基于LLaVA的视觉-文本指令模块，使模型能够有效理解视觉内容与指令之间的对应关系。进一步，我们构建了一个全面的多任务数据处理系统。由于各种任务之间存在数据重叠，该系统能够高效提供数据增强。使用该系统，我们构建了一个多类型、多场景的OmniV2V数据集及其相应的OmniV2V-Test基准。大量实验表明，OmniV2V在许多视频生成和编辑任务上表现与最佳现有开源和商业模型相当，有时甚至更优。
</details>

<details>
    <summary>Key points</summary>
    * 开发了统一的动态内容操作注入模块
    * 使用基于LLaVA的视觉-文本指令模块进行理解
    * 构建了多任务数据处理系统用于数据增强
</details>
</details>

---


<details>
<summary><b> Yan: Foundational Interactive Video Generation</b></summary>

* **Authors:** Deheng Ye, Fangyun Zhou, Jiacheng Lv, Jianqi Ma, Jun Zhang, Junyan Lv, Junyou Li, Minwen Deng, Mingyu Yang, Qiang Fu, Wei Yang, Wenkai Lv, Yangbin Yu, Yewen Wang, Yonghang Guan, Zhihao Hu, Zhongbin Fang, Zhongqian Sun
* **arXiv ID:** 2508.08601
* **One-liner:** Yan integrates simulation, generation, and editing for interactive video creation.
* **Published in:** arxiv (12 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.08601) | [[PDF]](https://arxiv.org/pdf/2508.08601) | [[Code]](https://greatx3.github.io/Yan/)

> **核心创新**
> 在基础框架中结合了实时模拟、多模态生成和多粒度编辑。

<details>
    <summary>Abstract</summary>
    我们提出了Yan，一个用于交互式视频生成的基础框架，涵盖了从模拟、生成到编辑的整个流程。具体而言，Yan包含三个核心模块。AAA级模拟：我们设计了一个高度压缩、低延迟的3D-VAE，结合基于KV缓存的移位窗口去噪推理过程，实现了实时1080P/60FPS的交互式模拟。多模态生成：我们引入了一种分层自回归描述方法，将游戏特定知识注入开放域多模态视频扩散模型（VDM），然后将VDM转化为帧级、动作可控、实时无限的交互式视频生成器。值得注意的是，当文本和视觉提示来自不同领域时，该模型展现出强大的泛化能力，使其能够根据用户提示灵活地混合和组合跨领域的风格和机制。多粒度编辑：我们提出了一种混合模型，明确解耦交互机制模拟与视觉渲染，使得在交互过程中能够通过文本进行多粒度视频内容编辑。总体而言，Yan整合了这些模块，将交互式视频生成从孤立能力推向全面的AI驱动交互创作范式，为下一代创意工具、媒体和娱乐铺平道路。项目页面：<a href="https://greatx3.github.io/Yan/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 实现了AAA级模拟，使用3D-VAE和KV缓存去噪
    * 引入了分层自回归描述用于多模态生成
    * 提出了混合模型用于解耦机制和渲染的编辑
</details>
</details>

---


<details>
<summary><b> EditDuet: A Multi-Agent System for Video Non-Linear Editing</b></summary>

* **Authors:** Marcelo Sandoval-Castaneda, Bryan Russell, Josef Sivic, Gregory Shakhnarovich, Fabian Caba Heilbron
* **arXiv ID:** 2509.10761
* **One-liner:** Automated video editing via multi-agent sequential decision making.
* **Published in:** arxiv (13 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.10761) | [[PDF]](https://arxiv.org/pdf/2509.10761) | [[Code]]()

> **核心创新**
> 将视频编辑表述为序列过程，使用编辑和批评代理。

<details>
    <summary>Abstract</summary>
    视频编辑和组装的自动化工具在从电影制作、广告到社交媒体内容创作等领域有广泛应用。先前的视频编辑工作主要集中于检索或用户界面，将实际编辑留给用户。相比之下，我们提出自动化视频编辑的核心任务，将其表述为序列决策过程。我们的方法是一种多代理方法。我们设计了一个编辑代理和一个批评代理。编辑代理以视频片段集合和自然语言指令作为输入，并使用视频编辑软件中常见的工具生成编辑序列。另一方面，批评代理根据生成的序列提供自然语言反馈，或在序列满意时进行渲染。我们引入了一种基于学习的方法，以在专门代理之间实现有效通信，解决语言驱动视频编辑任务。最后，我们探索了LLM作为评判指标来评估视频编辑系统的质量，并将其与一般人类偏好进行比较。我们通过用户研究定性和定量评估了我们系统的输出视频序列，发现我们的系统在覆盖率、时间约束满足和人类偏好方面大大优于现有方法。
</details>

<details>
    <summary>Key points</summary>
    * 设计了使用工具进行视频编辑的编辑代理
    * 使用批评代理进行反馈和渲染
    * 采用了基于学习的方法进行代理通信
</details>
</details>

---


<details>
<summary><b> Prompt-Driven Agentic Video Editing System: Autonomous Comprehension of Long-Form, Story-Driven Media</b></summary>

* **Authors:** Zihan Ding, Xinyi Wang, Junlong Chen, Per Ola Kristensson, Junxiao Shen
* **arXiv ID:** 2509.16811
* **One-liner:** Modular system for prompt-driven editing of long-form narrative videos.
* **Published in:** arxiv (20 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.16811) | [[PDF]](https://arxiv.org/pdf/2509.16811) | [[Code]]()

> **核心创新**
> 开发了语义索引流程用于全局叙事和透明编辑。

<details>
    <summary>Abstract</summary>
    创作者在编辑长篇、叙事丰富的视频时遇到困难，不是因为用户界面复杂，而是由于搜索、故事板设计和序列编排小时级素材的认知需求。现有的基于转录或嵌入的方法在创意工作流中表现不足，因为模型难以跟踪角色、推断动机和连接分散事件。我们提出了一种提示驱动、模块化的编辑系统，帮助创作者通过自由形式提示而非时间线来重构多小​​时内容。其核心是一个语义索引流程，通过时间分割、引导内存压缩和跨粒度融合构建全局叙事，生成情节、对话、情感和上下文的可解释轨迹。用户获得电影式编辑，同时可选地优化透明中间输出。在400多个视频上通过专家评分、QA和偏好研究评估，我们的系统扩展了提示驱动编辑，保持了叙事连贯性，并平衡了自动化与创作者控制。
</details>

<details>
    <summary>Key points</summary>
    * 构建了语义索引，包括时间分割和内存压缩
    * 实现了跨粒度融合用于情节和上下文轨迹
    * 允许用户优化中间输出
</details>
</details>

---


<details>
<summary><b> EditVerse: Unifying Image and Video Editing and Generation with In-Context Learning</b></summary>

* **Authors:** Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, Daniil Pakhomov, Zhe Lin, Soo Ye Kim, Qiang Xu
* **arXiv ID:** 2509.20360
* **One-liner:** EditVerse unifies image and video generation and editing in a single model.
* **Published in:** arxiv (24 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.20360) | [[PDF]](https://arxiv.org/pdf/2509.20360) | [[Code]]()

> **核心创新**
> 将模态表示为统一令牌序列以处理跨模态任务。

<details>
    <summary>Abstract</summary>
    基础模型的近期进展突显了统一和扩展的明确趋势，在多样化领域展现出涌现能力。尽管图像生成和编辑已迅速从任务特定过渡到统一框架，但由于架构限制和数据稀缺，视频生成和编辑仍然碎片化。在这项工作中，我们引入了EditVerse，一个在单一模型中统一图像和视频生成与编辑的框架。通过将所有模态（即文本、图像和视频）表示为统一令牌序列，EditVerse利用自注意力实现鲁棒的上下文学习、自然跨模态知识转移，以及灵活处理任意分辨率和持续时间的输入和输出。为了解决视频编辑训练数据的缺乏，我们设计了一个可扩展的数据流程，整理了232K个视频编辑样本，并将其与大规模图像和视频数据集结合进行联合训练。此外，我们提出了EditVerseBench，第一个覆盖多样化任务和分辨率的基于指令的视频编辑基准。大量实验和用户研究表明，EditVerse实现了最先进的性能，超越了现有开源和商业模型，同时在跨模态中展现出涌现的编辑和生成能力。
</details>

<details>
    <summary>Key points</summary>
    * 使用统一令牌序列表示所有模态
    * 设计了可扩展数据流程用于视频编辑样本
    * 创建了EditVerseBench基准用于视频编辑
</details>
</details>

---


<details>
<summary><b> ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation</b></summary>

* **Authors:** Han Zhang, Weichong Yin, Yewei Fang, Lanxin Li, Boqiang Duan, Zhihua Wu, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang
* **arXiv ID:** 2112.15283
* **One-liner:** ERNIE-ViLG enables bidirectional image-text generation with unified pre-training.
* **Published in:** arxiv (31 Dec 2021)
* **Links:** [[Paper]](https://arxiv.org/abs/2112.15283) | [[PDF]](https://arxiv.org/pdf/2112.15283) | [[Code]]()

> **核心创新**
> 将图像和文本生成表述为自回归任务以实现语义对齐。

<details>
    <summary>Abstract</summary>
    传统方法在图像-文本生成任务中主要分别处理自然双向生成任务，专注于设计任务特定框架以提高生成样本的质量和保真度。最近，视觉-语言预训练模型极大地提升了图像到文本生成任务的性能，但用于文本到图像合成任务的大规模预训练模型仍不发达。在本文中，我们提出了ERNIE-ViLG，一个基于Transformer模型的双向图像-文本生成的统一生成预训练框架。基于图像量化模型，我们将图像生成和文本生成都表述为以文本/图像输入为条件的自回归生成任务。双向图像-文本生成建模缓解了视觉和语言之间的语义对齐。对于文本到图像生成过程，我们进一步提出了一种端到端训练方法，以联合学习视觉序列生成器和图像重建器。为了探索双向文本-图像生成的大规模预训练前景，我们在1.45亿（中文）图像-文本对的大规模数据集上训练了一个100亿参数的ERNIE-ViLG模型，该模型在文本到图像和图像到文本任务上均实现了最先进的性能，在MS-COCO上文本到图像合成的FID为7.9，并在COCO-CN和AIC-ICC上获得图像描述的最佳结果。
</details>

<details>
    <summary>Key points</summary>
    * 将双向生成建模为自回归任务
    * 提出了文本到图像合成的端到端训练方法
    * 在大规模数据集上训练实现最先进性能
</details>
</details>

---


<details>
<summary><b> Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks</b></summary>

* **Authors:** Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, Aniruddha Kembhavi
* **arXiv ID:** 2206.08916
* **One-liner:** Unified-IO performs diverse AI tasks across vision and language with a single model.
* **Published in:** arxiv (17 Jun 2022)
* **Links:** [[Paper]](https://arxiv.org/abs/2206.08916) | [[PDF]](https://arxiv.org/pdf/2206.08916) | [[Code]](https://github.com/allenai/unified-io-inference)

> **核心创新**
> 将输入和输出同质化为令牌序列以进行统一Transformer训练。

<details>
    <summary>Abstract</summary>
    我们提出了Unified-IO，一个执行多种AI任务的模型，涵盖经典计算机视觉任务，包括姿态估计、物体检测、深度估计和图像生成，视觉-语言任务如区域描述和指代表达，以及自然语言处理任务如问答和释义。为如此多样化任务开发单一统一模型带来了独特挑战，因为每个任务涉及异构输入和输出，包括RGB图像、逐像素图、二值掩码、边界框和语言。我们通过将每个支持的输入和输出同质化为离散词汇令牌序列来实现这种统一。所有任务的这种共同表示允许我们在视觉和语言领域的90多个多样化数据集上联合训练单一基于Transformer的架构。Unified-IO是第一个能够在GRIT基准上执行所有7个任务的模型，并在NYUv2-Depth、ImageNet、VQA2.0、OK-VQA、Swig、VizWizGround、BoolQ和SciTail等16个多样化基准上产生强大结果，无需任务特定微调。Unified-IO的代码和演示可在：<a href="https://unified-io.allenai.org" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 将所有输入和输出表示为离散令牌序列
    * 在90多个多样化数据集上训练
    * 在多个基准上实现强大结果，无需微调
</details>
</details>

---


<details>
<summary><b> Grounding Language Models to Images for Multimodal Inputs and Outputs</b></summary>

* **Authors:** Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried
* **arXiv ID:** 2301.13823
* **One-liner:** Enabled text-only language models to process and generate interleaved image-text data.
* **Published in:** arxiv (31 Jan 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2301.13823) | [[PDF]](https://arxiv.org/pdf/2301.13823) | [[Code]](https://github.com/kohjingyu/fromage)

> **核心创新**
> 通过微调输入/输出线性层同时保持语言模型冻结，将预训练的纯文本语言模型接地到视觉领域。

<details>
    <summary>Abstract</summary>
    我们提出了一种高效的方法，将预训练的纯文本语言模型接地到视觉领域，使其能够处理任意交织的图像和文本数据，并生成与检索图像交织的文本。我们的方法利用了从大规模纯文本预训练中学到的语言模型能力，如上下文学习和自由形式文本生成。我们保持语言模型冻结，并微调输入和输出线性层以实现跨模态交互。这使得我们的模型能够处理任意交织的图像和文本输入，并生成与检索图像交织的自由形式文本。我们在接地任务（如上下文图像检索和多模态对话）上实现了强大的零样本性能，并展示了引人入胜的交互能力。我们的方法适用于任何现成的语言模型，并为在视觉接地设置中有效利用预训练语言模型铺平了道路。
</details>

<details>
    <summary>Key points</summary>
    * 利用大规模文本预训练中的上下文学习和自由形式文本生成能力
    * 冻结语言模型并微调线性层以实现跨模态交互
    * 在接地任务（如上下文图像检索和多模态对话）上实现了零样本性能
</details>
</details>

---


<details>
<summary><b> Any-to-Any Generation via Composable Diffusion</b></summary>

* **Authors:** Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, Mohit Bansal
* **arXiv ID:** 2305.11846
* **One-liner:** Developed a generative model that can generate any combination of output modalities from any input modalities.
* **Published in:** arxiv (19 May 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2305.11846) | [[PDF]](https://arxiv.org/pdf/2305.11846) | [[Code]](https://github.com/microsoft/i-Code/tree/main/i-Code-V3)

> **核心创新**
> 引入了可组合扩散（CoDi），通过在输入和输出空间中对齐模态，实现多种模态的并行生成。

<details>
    <summary>Abstract</summary>
    我们提出了可组合扩散（CoDi），一种新颖的生成模型，能够从任何输入模态组合生成任何输出模态组合，如语言、图像、视频或音频。与现有的生成AI系统不同，CoDi可以并行生成多种模态，其输入不限于文本或图像等模态子集。尽管缺乏许多模态组合的训练数据集，我们提出在输入和输出空间中对齐模态。这使得CoDi能够自由地以任何输入组合为条件，并生成任何模态组，即使它们在训练数据中不存在。CoDi采用了一种新颖的可组合生成策略，通过桥接扩散过程中的对齐来构建共享多模态空间，从而实现交织模态（如时间对齐的视频和音频）的同步生成。高度可定制和灵活，CoDi实现了强大的联合模态生成质量，并在单模态合成方面优于或与单模态最先进技术相当。项目页面包含演示和代码，位于此链接。
</details>

<details>
    <summary>Key points</summary>
    * 在输入和输出空间中对齐模态以处理未见过的组合
    * 使用可组合生成策略与共享多模态空间
    * 实现了强大的联合模态生成质量和竞争性的单模态合成
</details>
</details>

---


<details>
<summary><b> Planting a SEED of Vision in Large Language Model</b></summary>

* **Authors:** Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, Ying Shan
* **arXiv ID:** 2307.08041
* **One-liner:** Empowered LLMs with the ability to see and draw using a discrete image tokenizer.
* **Published in:** arxiv (16 Jul 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2307.08041) | [[PDF]](https://arxiv.org/pdf/2307.08041) | [[Code]](https://github.com/AILab-CVC/SEED)

> **核心创新**
> 设计了SEED，一种具有1D因果依赖和高层语义对齐的图像分词器，用于统一多模态训练。

<details>
    <summary>Abstract</summary>
    我们提出了SEED，一种精细的图像分词器，赋予大型语言模型（LLMs）同时看和画的涌现能力。图像分词器的研究先前已陷入僵局，因为采用量化视觉标记的框架由于在多模态理解（与BLIP-2等相比）或生成（与Stable Diffusion等相比）中表现不佳和收敛性差而失去突出地位。尽管有这些限制，我们对其统一视觉和文本表示的自然能力保持信心，促进了使用LLM原始配方进行可扩展多模态训练。在本研究中，我们确定了SEED架构和训练的两个关键原则，有效简化了与LLMs的后续对齐。（1）图像标记应独立于2D物理补丁位置，而是以1D因果依赖产生，表现出与LLMs中从左到右自回归预测机制对齐的内在相互依赖。（2）图像标记应捕获与词语语义抽象程度一致的高层语义，并在分词器训练阶段优化区分性和重建性。因此，现成的LLM能够通过高效LoRA调优整合我们的SEED，执行图像到文本和文本到图像的生成。全面的多模态预训练和指令调优（可能产生改进结果）保留给未来研究。此版本SEED使用仅64个V100 GPU和500万公开可用的图像-文本对在5.7天内训练完成。我们的初步研究强调了离散视觉标记在多功能多模态LLMs中的巨大潜力，以及适当图像分词器在更广泛研究中的重要性。
</details>

<details>
    <summary>Key points</summary>
    * 使图像标记独立于2D位置，具有1D因果依赖
    * 优化标记的区分性和重建性
    * 通过高效LoRA调优实现图像到文本和文本到图像的生成
</details>
</details>

---


<details>
<summary><b> Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization</b></summary>

* **Authors:** Yang Jin, Kun Xu, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Quzhe Huang, Bin Chen, Chenyi Lei, An Liu, Chengru Song, Xiaoqiang Lei, Di Zhang, Wenwu Ou, Kun Gai, Yadong Mu
* **arXiv ID:** 2309.04669
* **One-liner:** Unified vision and language representation for generative multimodal tasks.
* **Published in:** arxiv (9 Sep 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2309.04669) | [[PDF]](https://arxiv.org/pdf/2309.04669) | [[Code]](https://github.com/jy0205/LaVIT)

> **核心创新**
> 引入了LaVIT，一种基础模型，在生成学习范式下无差别地处理图像和文本。

<details>
    <summary>Abstract</summary>
    最近，大型语言模型（LLM）的显著进展激发了研究人员将其非凡推理能力转移到视觉和语言数据上。然而，主流方法主要将视觉输入视为提示，并专注于通过冻结的LLM优化以视觉内容为条件的文本生成过程。这种对视觉和语言的不平等处理严重限制了模型的潜力。在本文中，我们通过以统一形式表示视觉和语言来突破这一限制。具体来说，我们引入了一个精心设计的视觉分词器，将非语言图像翻译成离散标记序列，如同LLM可以阅读的外语。生成的视觉标记包含值得一个词语的高层语义，并支持根据图像变化的动态序列长度。结合此分词器，提出的基础模型LaVIT可以在相同的生成学习范式下无差别地处理图像和文本。这种统一使LaVIT能够作为一个令人印象深刻的通用接口，同时理解和生成多模态内容。广泛的实验进一步表明，它在大量视觉语言任务上大幅优于现有模型。我们的代码和模型在此链接可用。
</details>

<details>
    <summary>Key points</summary>
    * 使用视觉分词器将图像转换为离散标记，如同外语
    * 实现两种模态的统一生成学习
    * 在各种视觉语言任务上实现了卓越性能
</details>
</details>

---


<details>
<summary><b> VL-GPT: A Generative Pre-trained Transformer for Vision and Language Understanding and Generation</b></summary>

* **Authors:** Jinguo Zhu, Xiaohan Ding, Yixiao Ge, Yuying Ge, Sijie Zhao, Hengshuang Zhao, Xiaohua Wang, Ying Shan
* **arXiv ID:** 2312.09251
* **One-liner:** Created a transformer model for concurrent perception and generation of visual and linguistic data.
* **Published in:** arxiv (14 Dec 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2312.09251) | [[PDF]](https://arxiv.org/pdf/2312.09251) | [[Code]](https://github.com/AILab-CVC/VL-GPT)

> **核心创新**
> 开发了VL-GPT，使用图像分词器-去分词器框架，通过统一自回归目标进行多模态预训练。

<details>
    <summary>Abstract</summary>
    在这项工作中，我们介绍了视觉语言生成预训练Transformer（VL-GPT），一种Transformer模型，精通同时感知和生成视觉和语言数据。VL-GPT通过采用简单的自回归目标，实现了图像和文本模态的统一预训练方法，从而使模型能够像语言模型处理文本一样无缝处理图像和文本。为实现此目标，我们首先提出了一个新颖的图像分词器-去分词器框架，专门设计用于将原始图像转换为连续嵌入序列并相应重建它们。结合现有的文本分词器和去分词器，该框架允许将交织的图像-文本数据编码为多模态序列，随后可以馈入Transformer模型。因此，VL-GPT可以使用统一的自回归目标（即下一个标记预测）在多模态语料库上进行大规模预训练。预训练完成后，VL-GPT在广泛的视觉和语言理解和生成任务（如图像描述、视觉问答、文本到图像生成等）上表现出卓越的零样本和少样本性能。此外，预训练模型在提供多模态提示时保留了上下文学习能力。我们进一步对VL-GPT进行指令调优，突显了其作为多模态助手的卓越潜力。源代码和模型权重将发布。
</details>

<details>
    <summary>Key points</summary>
    * 提出了图像分词器-去分词器用于连续嵌入
    * 使用统一自回归预训练于多模态语料库
    * 展示了在视觉语言任务上的零样本和少样本能力
</details>
</details>

---


<details>
<summary><b> Unifying Generation and Compression: Ultra-low bitrate Image Coding Via Multi-stage Transformer</b></summary>

* **Authors:** Naifu Xue, Qi Mao, Zijian Wang, Yuan Zhang, Siwei Ma
* **arXiv ID:** 2403.03736
* **One-liner:** Merged image generation and compression for ultra-low bitrate scenarios.
* **Published in:** arxiv (6 Mar 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2403.03736) | [[PDF]](https://arxiv.org/pdf/2403.03736) | [[Code]]()

> **核心创新**
> 引入了统一图像生成-压缩（UIGC）范式，使用VQ分词化和多阶段Transformer。

<details>
    <summary>Abstract</summary>
    生成压缩技术的最新进展显著提高了压缩数据的感知质量。然而，这些进步主要集中于产生高频细节，常常忽视了生成模型捕获图像内容先验分布的能力，从而阻碍了在极端压缩场景（<0.05 bpp）中进一步降低比特率。受预测语言模型在无损压缩中能力的启发，本文引入了一种新颖的统一图像生成-压缩（UIGC）范式，合并了生成和压缩过程。UIGC框架的一个关键特征是采用向量量化（VQ）图像模型进行分词化，以及一个多阶段Transformer，设计用于利用空间上下文信息来建模先验分布。因此，这种双重用途框架有效利用学习到的先验进行熵估计，并协助重建丢失的标记。广泛的实验证明了所提出的UIGC框架在感知质量和人类感知上优于现有编解码器，特别是在超低比特率场景（<=0.03 bpp）中，开创了生成压缩的新方向。
</details>

<details>
    <summary>Key points</summary>
    * 采用向量量化图像模型进行分词化
    * 使用多阶段Transformer建模先验分布
    * 在超低比特率压缩中实现了优越的感知质量
</details>
</details>

---


<details>
<summary><b> In-Context Translation: Towards Unifying Image Recognition, Processing, and Generation</b></summary>

* **Authors:** Han Xue, Qianru Sun, Li Song, Wenjun Zhang, Zhiwu Huang
* **arXiv ID:** 2404.09633
* **One-liner:** Unified diverse vision tasks into a single framework using in-context learning.
* **Published in:** arxiv (15 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.09633) | [[PDF]](https://arxiv.org/pdf/2404.09633) | [[Code]]()

> **核心创新**
> 提出了上下文翻译（ICT），将输入-输出标准化为RGB图像对，并使用上下文学习进行训练。

<details>
    <summary>Abstract</summary>
    我们提出了上下文翻译（ICT），一种通用学习框架，用于统一视觉识别（如语义分割）、低级图像处理（如去噪）和条件图像生成（如边缘到图像合成）。得益于统一，ICT显著减少了为特定任务设计模型所带来的固有归纳偏差，并最大化相似任务之间的相互增强。然而，由于各种数据格式和训练流程，统一大量任务并非易事。为此，ICT引入了两个设计。首先，它将不同任务的输入-输出数据标准化为RGB图像对，例如，语义分割数据将RGB图像与其分割掩码配对为相同RGB格式。这将不同任务转化为两个RGB图像之间的一般翻译任务。其次，它将不同任务的训练标准化为一般上下文学习，其中“上下文”意味着输入包括目标任务的示例输入-输出对和一个查询图像。学习目标是生成与查询配对的“缺失”数据。因此，隐式翻译过程发生在查询和生成图像之间。在实验中，ICT统一了十个视觉任务，并在各自基准上展示了令人印象深刻的性能。值得注意的是，ICT在三大类计算机视觉任务中表现良好，而其两个竞争对手（Painter和PromptDiffusion）仅在最多两个任务类别中有效。此外，与竞争对手相比，ICT仅在4个RTX 3090 GPU上训练，显示出更高的训练效率和更低的成本。
</details>

<details>
    <summary>Key points</summary>
    * 将任务标准化为RGB图像对以进行一般翻译
    * 使用上下文学习与示例对和查询图像
    * 统一了十个视觉任务并展示了高效训练
</details>
</details>

---


<details>
<summary><b> SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation</b></summary>

* **Authors:** Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, Ying Shan
* **arXiv ID:** 2404.14396
* **One-liner:** Enhanced multimodal foundation model for real-world applicability with arbitrary image sizes and multi-granularity generation.
* **Published in:** arxiv (22 Apr 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2404.14396) | [[PDF]](https://arxiv.org/pdf/2404.14396) | [[Code]](https://github.com/AILab-CVC/SEED-X)

> **核心创新**
> 开发了SEED-X，为理解和生成任务建模多粒度视觉语义。

<details>
    <summary>Abstract</summary>
    多模态基础模型的快速发展在视觉语言理解和生成方面展示了显著进展，例如我们之前的工作SEED-LLaMA。然而，其能力与现实世界适用性之间仍存在差距，主要由于模型在有效响应各种用户指令和与多样视觉数据交互方面的能力有限。在这项工作中，我们专注于通过整合两个增强功能来弥合这一差距：（1）理解任意大小和比例的图像，以及（2）启用多粒度图像生成。我们提出了一个统一且多功能的基础模型，即SEED-X，它能够为理解和生成任务建模多粒度视觉语义。除了在公共基准上的竞争性结果外，SEED-X在指令调优后展示了在处理跨多个领域的现实世界应用中的有效性。我们希望我们的工作将激发未来研究，探索多功能多模态基础模型在现实世界应用中的潜力。模型、代码和数据集在此链接发布。
</details>

<details>
    <summary>Key points</summary>
    * 启用任意图像大小和比例的理解
    * 支持多粒度图像生成
    * 在指令调优后在现实世界应用中实现了有效性
</details>
</details>

---


<details>
<summary><b> Chameleon: Mixed-Modal Early-Fusion Foundation Models</b></summary>

* **Authors:** Chameleon Team
* **arXiv ID:** 2405.09818
* **One-liner:** Built a mixed-modal model capable of understanding and generating images and text in any sequence.
* **Published in:** arxiv (16 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.09818) | [[PDF]](https://arxiv.org/pdf/2405.09818) | [[Code]](https://github.com/facebookresearch/chameleon)

> **核心创新**
> 引入了Chameleon，具有早期融合基于标记的架构和稳定训练，用于统一多模态建模。

<details>
    <summary>Abstract</summary>
    我们提出了Chameleon，一个基于早期融合标记的混合模态模型家族，能够理解和生成任意序列中的图像和文本。我们概述了从初始开始的稳定训练方法、对齐配方和专为早期融合、基于标记、混合模态设置量身定制的架构参数化。这些模型在全面范围的任务上进行了评估，包括视觉问答、图像描述、文本生成、图像生成和长形式混合模态生成。Chameleon展示了广泛和通用的能力，包括在图像描述任务中的最先进性能，在纯文本任务中优于Llama-2，同时与Mixtral 8x7B和Gemini-Pro等模型竞争，并在单一模型中执行非平凡图像生成。根据人类判断，在新长形式混合模态生成评估中，它匹配或超过了包括Gemini Pro和GPT-4V在内的更大模型的性能，其中提示或输出包含图像和文本的混合序列。Chameleon标志着在完整多模态文档统一建模方面的重要进展。
</details>

<details>
    <summary>Key points</summary>
    * 使用早期融合基于标记的架构处理混合模态
    * 开发了稳定训练和对齐配方
    * 在各种任务中展示了最先进性能，包括长形式混合模态生成
</details>
</details>

---


<details>
<summary><b> Libra: Building Decoupled Vision System on Large Language Models</b></summary>

* **Authors:** Yifan Xu, Xiaoshan Yang, Yaguang Song, Changsheng Xu
* **arXiv ID:** 2405.10140
* **One-liner:** Designed a decoupled vision system for effective multimodal comprehension with LLMs.
* **Published in:** arxiv (16 May 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2405.10140) | [[PDF]](https://arxiv.org/pdf/2405.10140) | [[Code]](https://github.com/YifanXu74/Libra)

> **核心创新**
> 创建了Libra，具有路由视觉专家和跨模态桥模块，用于离散自回归建模。

<details>
    <summary>Abstract</summary>
    在这项工作中，我们介绍了Libra，一个在大型语言模型（LLM）上具有解耦视觉系统的原型模型。解耦视觉系统解耦了内部模态建模和跨模态交互，产生了独特的视觉信息建模和有效的跨模态理解。Libra通过对视觉和语言输入进行离散自回归建模进行训练。具体来说，我们将一个路由视觉专家与跨模态桥模块整合到预训练的LLM中，以在注意力计算期间路由视觉和语言流，从而在内部模态建模和跨模态交互场景中启用不同的注意力模式。实验结果表明，Libra的专门设计实现了一个强大的MLLM基线，在图像到文本场景中与现有工作竞争，仅使用5000万训练数据，为未来多模态基础模型提供了新视角。代码在此链接可用。
</details>

<details>
    <summary>Key points</summary>
    * 解耦内部模态建模和跨模态交互
    * 在注意力中使用路由视觉专家与跨模态桥
    * 使用最小训练数据实现了强大基线
</details>
</details>

---


<details>
<summary><b> GenArtist: Multimodal LLM as an Agent for Unified Image Generation and Editing</b></summary>

* **Authors:** Zhenyu Wang, Aoxue Li, Zhenguo Li, Xihui Liu
* **arXiv ID:** 2407.05600
* **One-liner:** Proposed GenArtist, a unified image generation and editing system coordinated by an MLLM agent.
* **Published in:** arxiv (8 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.05600) | [[PDF]](https://arxiv.org/pdf/2407.05600) | [[Code]](https://github.com/zhenyuw16/GenArtist)

> **核心创新**
> 通过分解复杂问题并使用带有逐步验证的工具库，在各种生成和编辑任务中实现了最先进的性能。

<details>
    <summary>Abstract</summary>
    尽管现有图像生成和编辑方法已取得成功，但当前模型在处理复杂问题时仍面临挑战，包括复杂文本提示以及缺乏验证和自校正机制，导致生成图像不可靠。同时，单个模型往往专精于特定任务，无法满足所有用户需求。我们提出了GenArtist，一个由多模态大语言模型（MLLM）代理协调的统一图像生成和编辑系统。我们将广泛的现有模型集成到工具库中，并利用代理进行工具选择和执行。对于复杂问题，MLLM代理将其分解为更简单的子问题，并构建树状结构来系统规划生成、编辑和自校正过程，进行逐步验证。通过自动生成缺失的位置相关输入并整合位置信息，可以有效使用适当工具解决每个子问题。实验表明，GenArtist能够执行各种生成和编辑任务，实现了最先进的性能，超越了SDXL和DALL-E 3等现有模型，如图1所示。项目页面位于此链接。
</details>

<details>
    <summary>Key points</summary>
    * 将复杂问题分解为更简单的子问题
    * 利用MLLM代理进行工具选择和执行
    * 整合自校正和验证机制
    * 自动生成缺失的位置相关输入
</details>
</details>

---


<details>
<summary><b> ANOLE: An Open, Autoregressive, Native Large Multimodal Models for Interleaved Image-Text Generation</b></summary>

* **Authors:** Ethan Chern, Jiadi Su, Yan Ma, Pengfei Liu
* **arXiv ID:** 2407.06135
* **One-liner:** Introduced Anole, an open, autoregressive, native large multimodal model for interleaved image-text generation.
* **Published in:** arxiv (8 Jul 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2407.06135) | [[PDF]](https://arxiv.org/pdf/2407.06135) | [[Code]](https://github.com/GAIR-NLP/anole)

> **核心创新**
> 通过从Chameleon构建并采用高效微调，实现了高质量、连贯的多模态生成，无需独立扩散模型。

<details>
    <summary>Abstract</summary>
    先前开源的多模态大模型面临若干限制：（1）它们通常缺乏原生集成，需要适配器将视觉表示与预训练大语言模型对齐；（2）许多仅限于单模态生成；（3）虽然一些支持多模态生成，但它们依赖独立的扩散模型进行视觉建模和生成。为缓解这些限制，我们提出了Anole，一个用于交错图像-文本生成的开放、自回归、原生多模态大模型。我们基于Meta AI的Chameleon构建Anole，采用创新的数据高效和参数高效的微调策略。Anole展示了高质量、连贯的多模态生成能力。我们开源了模型、训练框架和指令调优数据。
</details>

<details>
    <summary>Key points</summary>
    * 采用创新的数据高效和参数高效微调策略
    * 支持交错图像-文本生成
    * 开源模型、训练框架和指令调优数据
</details>
</details>

---


<details>
<summary><b> Show-o: One Single Transformer to Unify Multimodal Understanding and Generation</b></summary>

* **Authors:** Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, Mike Zheng Shou
* **arXiv ID:** 2408.12528
* **One-liner:** Developed Show-o, a unified transformer for multimodal understanding and generation.
* **Published in:** arxiv (22 Aug 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2408.12528) | [[PDF]](https://arxiv.org/pdf/2408.12528) | [[Code]](https://github.com/showlab/Show-o)

> **核心创新**
> 结合自回归和离散扩散建模处理混合模态，实现了与专用模型相当或更优的性能。

<details>
    <summary>Abstract</summary>
    我们提出了一个统一变换器，即Show-o，它统一了多模态理解和生成。与完全自回归模型不同，Show-o统一了自回归和离散扩散建模，以自适应处理各种和混合模态的输入和输出。该统一模型灵活支持广泛的视觉语言任务，包括视觉问答、文本到图像生成、文本引导修复/外推以及混合模态生成。在各种基准测试中，它展示了与现有专为理解或生成设计的同等或更大参数模型相当或更优的性能。这显著突显了其作为下一代基础模型的潜力。代码和模型已在此链接发布。
</details>

<details>
    <summary>Key points</summary>
    * 统一自回归和离散扩散建模
    * 支持各种视觉语言任务
    * 自适应处理混合模态的输入和输出
</details>
</details>

---


<details>
<summary><b> PUMA: Empowering Unified MLLM with Multi-granular Visual Generation</b></summary>

* **Authors:** Rongyao Fang, Chengqi Duan, Kun Wang, Hao Li, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, Hongsheng Li, Xihui Liu
* **arXiv ID:** 2410.13861
* **One-liner:** Proposed PUMA, a unified MLLM with multi-granular visual generation.
* **Published in:** arxiv (17 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.13861) | [[PDF]](https://arxiv.org/pdf/2410.13861) | [[Code]](https://github.com/rongyaofang/PUMA)

> **核心创新**
> 在单一框架中解决图像生成任务的粒度需求差异，展示了在多模态任务中的熟练能力。

<details>
    <summary>Abstract</summary>
    多模态基础模型的最新进展在视觉语言理解方面取得了显著进步。初步尝试也探索了多模态大语言模型（MLLM）在视觉内容生成方面的潜力。然而，现有工作在统一MLLM范式中未能充分解决不同图像生成任务的粒度需求差异——从文本到图像生成所需的多样性到图像操作所需的精确可控性。在这项工作中，我们提出了PUMA，通过多粒度视觉生成赋能统一MLLM。PUMA将多粒度视觉特征统一为MLLM的输入和输出，优雅地解决了统一MLLM框架中各种图像生成任务的不同粒度需求。经过多模态预训练和任务特定指令调优后，PUMA展示了在广泛多模态任务中的熟练能力。这项工作代表了向真正统一MLLM迈出的重要一步，能够适应各种视觉任务的粒度需求。代码和模型将在此链接发布。
</details>

<details>
    <summary>Key points</summary>
    * 将多粒度视觉特征统一为输入和输出
    * 使用多模态预训练和任务特定指令调优
    * 优雅处理不同粒度需求
</details>
</details>

---


<details>
<summary><b> VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation</b></summary>

* **Authors:** Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, Song Han, Yao Lu
* **arXiv ID:** 2409.04429
* **One-liner:** Presented VILA-U, a unified foundation model for video, image, language understanding and generation.
* **Published in:** arxiv (6 Sep 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2409.04429) | [[PDF]](https://arxiv.org/pdf/2409.04429) | [[Code]](https://github.com/mit-han-lab/vila-u)

> **核心创新**
> 使用单一自回归下一个令牌预测框架，简化模型并实现接近最先进的性能。

<details>
    <summary>Abstract</summary>
    VILA-U是一个统一基础模型，集成了视频、图像、语言理解和生成。传统视觉语言模型使用独立模块进行理解和生成视觉内容，这可能导致不对齐和复杂性增加。相比之下，VILA-U采用单一自回归下一个令牌预测框架处理这两个任务，消除了对扩散模型等额外组件的需求。这种方法不仅简化了模型，还在视觉语言理解和生成中实现了接近最先进的性能。VILA-U的成功归因于两个主要因素：统一视觉塔在预训练期间对齐离散视觉令牌与文本输入，增强了视觉感知；自回归图像生成可以在高质量数据集上实现与扩散模型相似的质量。这使得VILA-U能够使用完全基于令牌的自回归框架，与更复杂模型相媲美。
</details>

<details>
    <summary>Key points</summary>
    * 采用统一视觉塔对齐视觉令牌与文本
    * 使用自回归图像生成，无需扩散模型
    * 消除额外组件的需求
</details>
</details>

---


<details>
<summary><b> UniMuMo: Unified Text, Music and Motion Generation</b></summary>

* **Authors:** Han Yang, Kun Su, Yutong Zhang, Jiaben Chen, Kaizhi Qian, Gaowen Liu, Chuang Gan
* **arXiv ID:** 2410.04534
* **One-liner:** Introduced UniMuMo, a unified multimodal model for text, music, and motion generation.
* **Published in:** arxiv (6 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.04534) | [[PDF]](https://arxiv.org/pdf/2410.04534) | [[Code]](https://github.com/hanyangclarence/UniMuMo)

> **核心创新**
> 对齐未配对音乐和运动数据并使用统一编码器-解码器变换器，在跨模态中取得竞争性结果。

<details>
    <summary>Abstract</summary>
    我们介绍了UniMuMo，一个统一多模态模型，能够以任意文本、音乐和运动数据作为输入条件，生成所有三种模态的输出。为解决时间同步数据缺乏的问题，我们基于节奏模式对齐未配对音乐和运动数据，以利用现有大规模纯音乐和纯运动数据集。通过将音乐、运动和文本转换为基于令牌的表示，我们的模型通过统一编码器-解码器变换器架构桥接这些模态。为在单一框架内支持多个生成任务，我们引入了若干架构改进。我们提出用音乐码本编码运动，将运动映射到与音乐相同的特征空间。我们引入了音乐-运动并行生成方案，将所有音乐和运动生成任务统一到单一变换器解码器架构中，以音乐-运动联合生成的单一训练任务进行训练。此外，该模型通过微调现有预训练单模态模型设计，显著降低了计算需求。广泛实验表明，UniMuMo在音乐、运动和文本模态的所有单向生成基准测试中取得了竞争性结果。定量结果可在项目页面获取。
</details>

<details>
    <summary>Key points</summary>
    * 基于节奏模式对齐未配对音乐和运动数据
    * 使用基于令牌的表示和统一变换器架构
    * 提出音乐-运动并行生成方案
</details>
</details>

---


<details>
<summary><b> Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation</b></summary>

* **Authors:** Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, Ping Luo
* **arXiv ID:** 2410.13848
* **One-liner:** Developed Janus, an autoregressive framework that unifies multimodal understanding and generation.
* **Published in:** arxiv (17 Oct 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2410.13848) | [[PDF]](https://arxiv.org/pdf/2410.13848) | [[Code]](https://github.com/deepseek-ai/Janus)

> **核心创新**
> 将视觉编码解耦为独立路径以增强灵活性和性能，超越了先前统一模型。

<details>
    <summary>Abstract</summary>
    在本文中，我们介绍了Janus，一个统一多模态理解和生成的自回归框架。先前研究通常依赖单一视觉编码器处理这两个任务，例如Chameleon。然而，由于多模态理解和生成所需的信息粒度不同，这种方法可能导致次优性能，特别是在多模态理解方面。为解决此问题，我们将视觉编码解耦为独立路径，同时仍利用单一统一变换器架构进行处理。这种解耦不仅缓解了视觉编码器在理解和生成中角色的冲突，还增强了框架的灵活性。例如，多模态理解和生成组件可以独立选择最合适的编码方法。实验显示，Janus超越了先前统一模型，并匹配或超过了任务特定模型的性能。Janus的简洁性、高灵活性和有效性使其成为下一代统一多模态模型的有力候选。
</details>

<details>
    <summary>Key points</summary>
    * 将视觉编码解耦为独立路径
    * 使用单一统一变换器架构
    * 允许理解和生成组件独立选择编码方法
</details>
</details>

---


<details>
<summary><b> Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads</b></summary>

* **Authors:** Siqi Kou, Jiachun Jin, Zhihong Liu, Chang Liu, Ye Ma, Jian Jia, Quan Chen, Peng Jiang, Zhijie Deng
* **arXiv ID:** 2412.00127
* **One-liner:** Introduced Orthus, an autoregressive transformer for multimodal tasks with continuous image features.
* **Published in:** arxiv (28 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.00127) | [[PDF]](https://arxiv.org/pdf/2412.00127) | [[Code]](https://github.com/zhijie-group/Orthus)

> **核心创新**
> 在自回归建模下结合离散文本令牌和连续图像特征，通过高效训练在基准测试中取得高分。

<details>
    <summary>Abstract</summary>
    我们介绍了Orthus，一个自回归变换器，擅长在给定文本提示下生成图像、基于视觉输入回答问题，甚至创作长篇幅图像-文本交错内容。与先前统一多模态建模方法不同，Orthus在自回归建模原则下同时处理离散文本令牌和连续图像特征。对视觉信号的连续处理最小化了图像理解和生成的信息损失，而完全自回归公式使模态间相关性表征变得直接。Orthus利用这些优势的关键机制在于其模态特定头——一个常规语言建模头预测离散文本令牌，一个扩散头基于主干输出生成连续图像特征。我们设计了一种高效构建Orthus的策略——通过将现有统一自回归模型中的向量量化操作替换为软替代，引入扩散头，并调优添加模块以重建图像，我们可以轻松创建Orthus-base模型（例如，仅需72 A100 GPU小时）。Orthus-base可以进一步接受后训练以更好地建模交错图像和文本。实证上，Orthus在标准基准测试中超越了包括Show-o和Chameleon在内的竞争基线，使用7B参数实现了GenEval得分0.58和MME-P得分1265.8。Orthus还展示了卓越的混合模态生成能力，反映了处理复杂实际生成任务的潜力。
</details>

<details>
    <summary>Key points</summary>
    * 使用模态特定头：语言建模头处理文本，扩散头处理图像
    * 用软替代替换向量量化以高效构建模型
    * 支持混合模态生成和后训练
</details>
</details>

---


<details>
<summary><b> MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding</b></summary>

* **Authors:** Rongchang Xie, Chen Du, Ping Song, Chang Liu
* **arXiv ID:** 2411.17762
* **One-liner:** Proposed MUSE-VL, a unified vision-language model using Semantic Discrete Encoding.
* **Published in:** arxiv (26 Nov 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2411.17762) | [[PDF]](https://arxiv.org/pdf/2411.17762) | [[Code]]()

> **核心创新**
> 通过语义约束改进视觉和语言令牌对齐，减少训练数据需求并提升性能。

<details>
    <summary>Abstract</summary>
    我们介绍了MUSE-VL，一个通过语义离散编码的统一视觉语言模型，用于多模态理解和生成。最近，研究社区开始探索用于视觉生成和理解的统一模型。然而，现有视觉标记器仅考虑低级信息，难以与语言令牌对齐。这导致高训练复杂性，并需要大量训练数据以实现最优性能。此外，它们的性能仍远低于专用理解模型。本文提出语义离散编码，通过向视觉标记器添加语义约束，有效对齐视觉令牌和语言令牌的信息。这大大减少了训练数据量，并提高了统一模型的性能。在相同LLM大小下，我们的方法将理解性能比先前SOTA Emu3提高了4.8%，并超越了专用理解模型LLaVA-NeXT 34B的3.7%。我们的模型在视觉生成基准测试中也超越了现有统一模型。
</details>

<details>
    <summary>Key points</summary>
    * 引入语义离散编码以改进令牌对齐
    * 减少训练复杂性和数据需求
    * 相比SOTA提升理解和生成性能
</details>
</details>

---


<details>
<summary><b> Liquid: Language Models are Scalable and Unified Multi-modal Generators</b></summary>

* **Authors:** Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, Xiang Bai
* **arXiv ID:** 2412.04332
* **One-liner:** Presented Liquid, an auto-regressive paradigm for visual comprehension and generation using a single LLM.
* **Published in:** arxiv (5 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.04332) | [[PDF]](https://arxiv.org/pdf/2412.04332) | [[Code]](https://github.com/FoundationVision/Liquid)

> **核心创新**
> 揭示了统一训练的缩放定律，使任务相互增强，并超越了Chameleon和SD-XL等模型。

<details>
    <summary>Abstract</summary>
    我们提出了Liquid，一种自回归生成范式，通过将图像标记为离散代码并在共享特征空间中学习这些代码嵌入与文本令牌，无缝集成视觉理解和生成。与先前多模态大语言模型不同，Liquid使用单一大型语言模型实现此集成，无需外部预训练视觉嵌入如CLIP。Liquid首次揭示了一个缩放定律：随着模型大小增加，由视觉和语言任务统一训练带来的性能下降不可避免性减小。此外，统一令牌空间使视觉生成和理解任务相互增强，有效消除了早期模型中常见的干扰。我们展示了现有LLM可以作为Liquid的强大基础，节省100倍训练成本，同时在多模态能力上超越Chameleon，并保持与主流LLM如LLAMA2相当的语言性能。Liquid还在视觉语言和纯文本任务中表现出色，超越了SD v2.1和SD-XL等模型。这项工作证明了LLM是强大的多模态生成器，为增强视觉语言理解和生成提供了可扩展解决方案。代码和模型将在此链接发布。
</details>

<details>
    <summary>Key points</summary>
    * 将图像标记为离散代码在共享特征空间中
    * 使用单一LLM，无需外部视觉嵌入
    * 展示缩放定律，性能随模型大小提升
</details>
</details>

---


<details>
<summary><b> SILMM: Self-Improving Large Multimodal Models for Compositional Text-to-Image Generation</b></summary>

* **Authors:** Leigang Qu, Haochuan Li, Wenjie Wang, Xiang Liu, Juncheng Li, Liqiang Nie, Tat-Seng Chua
* **arXiv ID:** 2412.05818
* **One-liner:** Introduced SILMM, a model-agnostic iterative self-improvement framework for LMMs.
* **Published in:** arxiv (8 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.05818) | [[PDF]](https://arxiv.org/pdf/2412.05818) | [[Code]](https://github.com/LgQu/SILMM)

> **核心创新**
> 通过DPO使LMMs能够提供自反馈并优化文本-图像对齐，并针对连续视觉特征进行适应性调整。

<details>
    <summary>Abstract</summary>
    大型多模态模型（LMMs）在多模态理解和生成方面展现出令人印象深刻的能力，推动了文本到图像生成的进步。然而，在组合场景中实现准确的文本-图像对齐仍然具有挑战性。现有方法，如多步生成的布局规划以及从人类反馈或AI反馈中学习，严重依赖于提示工程、昂贵的人工标注和持续升级，限制了灵活性和可扩展性。在本工作中，我们引入了一个模型无关的迭代自改进框架（SILMM），该框架能够使LMMs提供有用且可扩展的自反馈，并通过直接偏好优化（DPO）优化文本-图像对齐。DPO可以轻松应用于使用离散视觉标记作为中间图像表示的LMMs；而对于具有连续视觉特征的LMMs，由于获取生成概率具有挑战性，DPO不太适用。为了使SILMM适应具有连续特征的LMMs，我们提出了一种多样性机制来获取多样化的表示，以及一种基于核的连续DPO用于对齐。在三个组合文本到图像生成基准上的广泛实验验证了SILMM的有效性和优越性，在T2I-CompBench++上显示出超过30%的改进，在DPG-Bench上显示出约20%的改进。
</details>

<details>
    <summary>Key points</summary>
    * 迭代自改进框架（SILMM）
    * 直接偏好优化（DPO）用于对齐
    * 针对连续特征的多样性机制
    * 基于核的连续DPO
</details>
</details>

---


<details>
<summary><b> ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance</b></summary>

* **Authors:** Chunwei Wang, Guansong Lu, Junwei Yang, Runhui Huang, Jianhua Han, Lu Hou, Wei Zhang, Hang Xu
* **arXiv ID:** 2412.06673
* **One-liner:** Developed ILLUME, a unified MLLM with integrated understanding and generation.
* **Published in:** arxiv (9 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.06673) | [[PDF]](https://arxiv.org/pdf/2412.06673) | [[Code]]()

> **核心创新**
> 通过视觉标记器和自增强对齐方案实现数据效率和协同增强。

<details>
    <summary>Abstract</summary>
    本文中，我们介绍了ILLUME，一个统一的多模态大语言模型（MLLM），通过统一的下一标记预测公式，将多模态理解和生成能力无缝集成在单个大语言模型中。为了解决通常需要的大数据集规模问题，我们提出通过设计一个包含语义信息的视觉标记器和渐进式多阶段训练过程来提高数据效率。这种方法将预训练所需的数据集大小减少到仅15M——比通常所需少四倍以上——同时在与现有统一MLLMs（如Janus）相比时，实现了竞争性甚至更优的性能。此外，为了促进理解和生成能力之间的协同增强，这在先前工作中较少探索，我们引入了一种新颖的自增强多模态对齐方案。该方案监督MLLM自评估文本描述与自生成图像之间的一致性，促进模型更准确地解释图像，并避免因图像生成中的不对齐而导致的非现实和错误预测。基于广泛实验，我们提出的ILLUME在多个多模态理解、生成和编辑基准上脱颖而出，与最先进的统一MLLMs和专门模型竞争。
</details>

<details>
    <summary>Key points</summary>
    * 统一下一标记预测公式
    * 包含语义信息的视觉标记器
    * 渐进式多阶段训练
    * 自增强多模态对齐方案
</details>
</details>

---


<details>
<summary><b> Visual Lexicon: Rich Image Features in Language Space</b></summary>

* **Authors:** XuDong Wang, Xingyi Zhou, Alireza Fathi, Trevor Darrell, Cordelia Schmid
* **arXiv ID:** 2412.06774
* **One-liner:** Created Visual Lexicon (ViLex), a visual language encoding image information into text tokens.
* **Published in:** arxiv (9 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.06774) | [[PDF]](https://arxiv.org/pdf/2412.06774) | [[Code]]()

> **核心创新**
> 通过自监督方式捕捉语义和细节，实现高质量图像生成和理解。

<details>
    <summary>Abstract</summary>
    我们提出了视觉词典（ViLex），一种新颖的视觉语言，将丰富的图像信息编码为词汇标记的文本空间，同时保留通常难以用自然语言传达的精细视觉细节。与优先考虑高级语义（如CLIP）或像素级重建（如VAE）的传统方法不同，ViLex同时捕捉丰富的语义内容和精细视觉细节，实现高质量图像生成和全面视觉场景理解。通过自监督学习流程，ViLex生成优化用于使用冻结文本到图像（T2I）扩散模型重建输入图像的标记，保留高保真语义级重建所需的详细信息。作为语言空间中的图像嵌入，ViLex标记利用自然语言的组合性，允许它们独立用作“文本标记”或与自然语言标记结合，以提示预训练T2I模型具有视觉和文本输入，模拟我们与视觉语言模型（VLMs）的交互方式。实验表明，ViLex在图像重建中比文本嵌入实现更高的保真度——即使使用单个ViLex标记。此外，ViLex成功以零样本、无监督方式执行各种DreamBooth任务，无需微调T2I模型。另外，ViLex作为强大的视觉编码器，相对于强SigLIP基线，在15个基准上持续改进视觉语言模型性能。
</details>

<details>
    <summary>Key points</summary>
    * 自监督学习流程
    * 用于图像重建的标记生成
    * 与T2I模型一起用作文本标记
    * 零样本DreamBooth任务
</details>
</details>

---


<details>
<summary><b> SynerGen-VL: Towards Synergistic Image Understanding and Generation with Vision Experts and Token Folding</b></summary>

* **Authors:** Hao Li, Changyao Tian, Jie Shao, Xizhou Zhu, Zhaokai Wang, Jinguo Zhu, Wenhan Dou, Xiaogang Wang, Hongsheng Li, Lewei Lu, Jifeng Dai
* **arXiv ID:** 2412.09604
* **One-liner:** Proposed SynerGen-VL, an encoder-free MLLM for image understanding and generation.
* **Published in:** arxiv (12 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.09604) | [[PDF]](https://arxiv.org/pdf/2412.09604) | [[Code]]()

> **核心创新**
> 通过标记折叠和渐进对齐预训练简化模型设计，实现竞争性性能。

<details>
    <summary>Abstract</summary>
    大型语言模型（LLMs）的显著成功已扩展到多模态领域，在图像理解和生成方面取得了卓越性能。最近开发统一多模态大语言模型（MLLMs）以整合这些能力的努力显示出有希望的结果。然而，现有方法通常涉及模型架构或训练流程的复杂设计，增加了模型训练和扩展的难度。在本文中，我们提出SynerGen-VL，一个简单而强大的无编码器MLLM，能够同时进行图像理解和生成。为了解决现有无编码器统一MLLMs中识别的挑战，我们引入了标记折叠机制和基于视觉专家的渐进对齐预训练策略，这些策略有效支持高分辨率图像理解，同时减少训练复杂性。在通过统一下一标记预测目标在大规模混合图像-文本数据上训练后，SynerGen-VL在可比或更小参数规模下达到或超越现有无编码器统一MLLMs的性能，并缩小了与任务特定最先进模型的差距，突显了未来统一MLLMs的有希望路径。我们的代码和模型将发布。
</details>

<details>
    <summary>Key points</summary>
    * 无编码器架构
    * 标记折叠机制
    * 基于视觉专家的渐进对齐预训练
    * 统一下一标记预测目标
</details>
</details>

---


<details>
<summary><b> MetaMorph: Multimodal Understanding and Generation via Instruction Tuning</b></summary>

* **Authors:** Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, Zhuang Liu
* **arXiv ID:** 2412.14164
* **One-liner:** Introduced VPiT for instruction tuning to enable LLMs to generate text and visual tokens.
* **Published in:** arxiv (18 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.14164) | [[PDF]](https://arxiv.org/pdf/2412.14164) | [[Code]](https://tsb0601.github.io/metamorph/)

> **核心创新**
> 通过高效数据使用，将视觉生成解锁为改进理解的副产品。

<details>
    <summary>Abstract</summary>
    在本工作中，我们提出视觉预测指令调优（VPiT）——一种简单有效的视觉指令调优扩展，使预训练LLM能够快速转变为能够生成文本和视觉标记的统一自回归模型。VPiT教导LLM从以指令跟随格式策划的任何图像和文本数据输入序列中预测离散文本标记和连续视觉标记。我们的实证研究揭示了VPiT的几个有趣特性：（1）视觉生成能力作为改进视觉理解的副产品自然出现，并可以通过少量生成数据高效解锁；（2）虽然我们发现理解和生成相互有益，但理解数据对两种能力的贡献比生成数据更有效。基于这些发现，我们训练了MetaMorph模型，并在视觉理解和生成上实现了竞争性性能。在视觉生成中，MetaMorph可以利用从LLM预训练中获得的世界知识和推理能力，并克服其他生成模型常见的失败模式。我们的结果表明，LLMs可能具有强大的“先验”视觉能力，可以通过相对简单的指令调优过程高效适应视觉理解和生成。
</details>

<details>
    <summary>Key points</summary>
    * 视觉预测指令调优（VPiT）
    * 离散文本和连续视觉标记的预测
    * 理解和生成的相互益处
    * 使用少量生成数据的高效适应
</details>
</details>

---


<details>
<summary><b> LMFusion: Adapting Pretrained Language Models for Multimodal Generation</b></summary>

* **Authors:** Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, Lili Yu
* **arXiv ID:** 2412.15188
* **One-liner:** Developed LMFusion, a framework to add multimodal capabilities to text-only LLMs.
* **Published in:** arxiv (19 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.15188) | [[PDF]](https://arxiv.org/pdf/2412.15188) | [[Code]]()

> **核心创新**
> 通过并行模块保留语言能力，同时实现视觉理解和生成。

<details>
    <summary>Abstract</summary>
    我们提出了LMFusion，一个框架，用于赋能预训练纯文本大型语言模型（LLMs）具有多模态生成能力，使它们能够理解和生成任意序列的文本和图像。LMFusion利用现有Llama-3的权重自回归处理文本，同时引入额外且并行的Transformer模块，用于使用扩散处理图像。在训练期间，来自每个模态的数据被路由到其专用模块：模态特定前馈层、查询-键-值投影和归一化层独立处理每个模态，而共享的自注意力层允许跨文本和图像特征的交互。通过冻结文本特定模块并仅训练图像特定模块，LMFusion保留了纯文本LLMs的语言能力，同时发展出强大的视觉理解和生成能力。与从零开始预训练多模态生成模型的方法相比，我们的实验表明，LMFusion仅使用50%的FLOPs就将图像理解提高了20%，图像生成提高了3.6%，同时保持Llama-3的语言能力。我们还证明该框架可以适应现有视觉语言模型具有多模态生成能力。总体而言，该框架不仅利用了纯文本LLMs的现有计算投资，还实现了语言和视觉能力的并行发展，为高效多模态模型开发提供了一个有希望的方向。
</details>

<details>
    <summary>Key points</summary>
    * 模态特定模块与共享自注意力
    * 冻结文本模块，训练图像模块
    * 高效利用现有LLM权重
    * 语言和视觉的并行发展
</details>
</details>

---


<details>
<summary><b> Dual Diffusion for Unified Image Generation and Understanding</b></summary>

* **Authors:** Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, Peng Wang
* **arXiv ID:** 2501.00289
* **One-liner:** Proposed a large-scale end-to-end diffusion model for multimodal understanding and generation.
* **Published in:** arxiv (31 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.00289) | [[PDF]](https://arxiv.org/pdf/2501.00289) | [[Code]](https://github.com/zijieli-Jlee/Dual-Diffusion)

> **核心创新**
> 首个支持完整视觉语言能力的扩散模型，采用跨模态似然框架。

<details>
    <summary>Abstract</summary>
    扩散模型在文本到图像生成中取得了巨大成功，但在视觉理解任务上仍然落后，该领域由自回归视觉语言模型主导。我们提出了一个大规模、完全端到端的扩散模型，用于多模态理解和生成，显著改进了现有基于扩散的多模态模型，并且是首个支持全套视觉语言建模能力的模型。受多模态扩散Transformer（MM-DiT）和离散扩散语言建模最新进展的启发，我们利用跨模态最大似然估计框架，该框架在单一损失函数下同时训练图像和文本的条件似然，并通过扩散Transformer的两个分支进行反向传播。所得模型高度灵活，能够执行广泛任务，包括图像生成、字幕生成和视觉问答。我们的模型在图像理解和生成方面与最近的统一模型相比达到了竞争性性能，展示了多模态扩散建模作为自回归下一标记预测模型的有希望替代方案的潜力。
</details>

<details>
    <summary>Key points</summary>
    * 跨模态最大似然估计
    * 图像和文本分支的联合训练
    * 扩散Transformer架构
    * 支持生成、字幕生成和VQA
</details>
</details>

---


<details>
<summary><b> Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling</b></summary>

* **Authors:** Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan
* **arXiv ID:** 2501.17811
* **One-liner:** Enhanced Janus to Janus-Pro with optimized training, data, and scaling.
* **Published in:** arxiv (29 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.17811) | [[PDF]](https://arxiv.org/pdf/2501.17811) | [[Code]](https://github.com/deepseek-ai/Janus)

> **核心创新**
> 通过改进的稳定性和扩展，在多模态理解和文本到图像生成上取得进步。

<details>
    <summary>Abstract</summary>
    在本工作中，我们介绍了Janus-Pro，先前工作Janus的先进版本。具体来说，Janus-Pro整合了（1）优化的训练策略，（2）扩展的训练数据，以及（3）扩展到更大模型规模。通过这些改进，Janus-Pro在多模态理解和文本到图像指令跟随能力上取得了显著进步，同时增强了文本到图像生成的稳定性。我们希望这项工作将激发该领域的进一步探索。代码和模型公开可用。
</details>

<details>
    <summary>Key points</summary>
    * 优化的训练策略
    * 扩展的训练数据
    * 更大模型规模扩展
    * 生成稳定性的改进
</details>
</details>

---


<details>
<summary><b> QLIP: Text-Aligned Visual Tokenization Unifies Auto-Regressive Multimodal Understanding and Generation</b></summary>

* **Authors:** Yue Zhao, Fuzhao Xue, Scott Reed, Linxi Fan, Yuke Zhu, Jan Kautz, Zhiding Yu, Philipp Krähenbühl, De-An Huang
* **arXiv ID:** 2502.05178
* **One-liner:** Introduced QLIP, a visual tokenization method for multimodal understanding and generation.
* **Published in:** arxiv (7 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.05178) | [[PDF]](https://arxiv.org/pdf/2502.05178) | [[Code]](https://github.com/NVlabs/QLIP/tree/main)

> **核心创新**
> 平衡重建和对齐目标，通过高效训练实现统一模型。

<details>
    <summary>Abstract</summary>
    我们介绍了量化语言图像预训练（QLIP），一种视觉标记化方法，结合了最先进的重建质量和最先进的零样本图像理解。QLIP训练一个基于二值球面量化的自编码器，具有重建和语言-图像对齐目标。我们首次表明这两个目标不需要相互冲突。我们在训练期间动态平衡两个损失项，并展示一个两阶段训练流程有效混合了图像-语言预训练的大批量需求与重建目标施加的内存瓶颈。我们验证了QLIP在多模态理解和文本条件图像生成中的有效性，使用单一模型。具体来说，QLIP作为LLaVA视觉编码器和LlamaGen图像标记器的即插即用替代品，性能相当甚至更好。最后，我们证明QLIP支持统一混合模态自回归模型用于理解和生成。
</details>

<details>
    <summary>Key points</summary>
    * 基于二值球面量化的自编码器
    * 损失项的动态平衡
    * 两阶段训练流程
    * 编码器和标记器的即插即用替代
</details>
</details>

---


<details>
<summary><b> UniCMs: A Unified Consistency Model For Efficient Multimodal Generation and Understanding</b></summary>

* **Authors:** Chenkai Xu, Xu Wang, Zhenyi Liao, Yishun Li, Tianqi Hou, Zhijie Deng
* **arXiv ID:** 2502.05415
* **One-liner:** Developed UniCMs, a unified consistency model for efficient multimodal generation and understanding.
* **Published in:** arxiv (8 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.05415) | [[PDF]](https://arxiv.org/pdf/2502.05415) | [[Code]](https://github.com/zhijie-group/UniCMs)

> **核心创新**
> 使用离散标记和并行解码用于文本，在速度和性能上超越现有模型。

<details>
    <summary>Abstract</summary>
    一致性模型（CMs）在图像和文本的高效生成中显示出潜力。这自然引发了一个问题：我们是否可以学习一个统一的CM用于高效多模态生成（例如，文本到图像）和理解（例如，图像到文本）。直观上，这样的模型可以通过对现有统一多模态模型应用一致性蒸馏（CD）来获得。然而，关键挑战是为图像和文本生成建立统一的去噪视角，这对于建立一致性映射至关重要。为了解决这个问题，在表示层面，我们主张对两种模态使用离散标记，以最好地保留语言建模能力。关键的是，我们不是通过最近的离散扩散语言建模原理定义文本去噪轨迹，而是使用自回归语言模型的并行解码轨迹来指定它，受益于后者在一般文本生成任务中的优越性能。图像标记的去噪轨迹遵循标准离散扩散。我们在这些组合的多模态轨迹上同时训练我们的统一一致性模型（UniCMs），使用统一目标。我们引入轨迹分割策略以进一步改进训练收敛。实证上，在文本到图像生成中，UniCMs在GenEval、Image Reward和CLIP Score指标上优于SD3，同时仅需约1/8的采样时间。同时，在图像到文本生成中，UniCMs在MMMU基准上超越Show-o，同时在长序列生成速度上快1.5倍。代码可在<a href="https://github.com/zhijie-group/UniCMs" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 统一模型的一致性蒸馏
    * 两种模态的离散标记
    * 文本去噪的并行解码轨迹
    * 轨迹分割策略
</details>
</details>

---


<details>
<summary><b> UniTok: A Unified Tokenizer for Visual Generation and Understanding</b></summary>

* **Authors:** Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, Xiaojuan Qi
* **arXiv ID:** 2502.20321
* **One-liner:** Introduced UniTok, a unified tokenizer that sets new performance records in image generation and understanding.
* **Published in:** arxiv (27 Feb 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2502.20321) | [[PDF]](https://arxiv.org/pdf/2502.20321) | [[Code]](https://github.com/FoundationVision/UniTok)

> **核心创新**
> 通过多码本量化扩展词汇表大小和瓶颈维度，实现重建和语义监督的统一。

<details>
    <summary>Abstract</summary>
    视觉生成和理解模型通常依赖不同的分词器处理图像，这为在单一框架内统一它们带来了关键挑战。近期研究尝试通过连接VQVAE（用于自回归生成）和CLIP（用于理解）的训练来构建统一分词器。然而，直接结合这些训练目标已被观察到会导致严重的损失冲突。本文中，我们表明重建和语义监督本质上并不冲突；相反，潜在瓶颈源于离散标记空间的有限表示能力。基于这些洞见，我们引入了UniTok，一种统一分词器，采用新颖的多码本量化机制，有效扩展词汇表大小和瓶颈维度。在最终性能方面，UniTok在ImageNet上创下了0.38 rFID和78.6%零样本准确率的新纪录。此外，UniTok可以无缝集成到MLLMs中，解锁原生视觉生成能力，而不损害理解性能。另外，我们显示UniTok支持无cfg生成，在ImageNet 256×256基准上将gFID从14.6降低到2.5。
</details>

<details>
    <summary>Key points</summary>
    * 多码本量化机制
    * 扩展词汇表大小和瓶颈维度
    * 无缝集成到MLLMs中实现原生视觉生成
    * 支持无cfg生成
</details>
</details>

---


<details>
<summary><b> MMGen: Unified Multi-modal Image Generation and Understanding in One Go</b></summary>

* **Authors:** Jiepeng Wang, Zhaoqing Wang, Hao Pan, Yuan Liu, Dongdong Yu, Changhu Wang, Wenping Wang
* **arXiv ID:** 2503.20644v1
* **One-liner:** Introduced MMGen, a unified diffusion framework for multi-modal generation and understanding.
* **Published in:** arxiv (26 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.20644v1) | [[PDF]](https://arxiv.org/pdf/2503.20644v1) | [[Code]](https://github.com/jiepengwang/MMGen)

> **核心创新**
> 通过新颖的扩散变换器和模态解耦策略，将多种生成任务集成到单一扩散模型中。

<details>
    <summary>Abstract</summary>
    统一扩散框架在多种模态生成和理解方面具有变革潜力，可实现无缝可控的图像扩散和其他跨模态任务。本文中，我们介绍了MMGen，一个统一框架，将多种生成任务集成到单一扩散模型中。这包括：（1）多模态类别条件生成，其中给定类别信息，通过单一推理过程同时生成多模态输出；（2）多模态视觉理解，从RGB图像准确预测深度、表面法线和分割图；（3）多模态条件生成，基于特定模态条件和其他对齐模态生成相应的RGB图像。我们的方法开发了一种新颖的扩散变换器，灵活支持多模态输出，以及简单的模态解耦策略来统一各种任务。广泛的实验和应用展示了MMGen在多样任务和条件下的有效性和优越性，突显其在需要同时生成和理解的应用中的潜力。
</details>

<details>
    <summary>Key points</summary>
    * 新颖的扩散变换器用于多模态输出
    * 模态解耦策略
    * 多模态类别条件生成
    * 多模态视觉理解
    * 多模态条件生成
</details>
</details>

---


<details>
<summary><b> Towards Enhanced Image Generation Via Multi-modal Chain of Thought in Unified Generative Models</b></summary>

* **Authors:** Yi Wang, Mushui Liu, Wanggui He, Hanyang Yuan, Longxiang Zhang, Ziwei Huang, Guanghao Zhang, Wenkai Fang, Haoze Jiang, Shengxuming Zhang, Dong She, Jinlong Liu, Weilong Dai, Mingli Song, Hao Jiang, Jie Song
* **arXiv ID:** 2503.01298
* **One-liner:** Introduced FoX with MCoT to enhance complex image generation in unified models.
* **Published in:** arxiv (3 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.01298) | [[PDF]](https://arxiv.org/pdf/2503.01298) | [[Code]]()

> **核心创新**
> 通过集成链式思维（CoT）与功能导向专家架构，解决复杂组合指令问题。

<details>
    <summary>Abstract</summary>
    统一生成模型在文本和图像生成方面表现出显著性能。对于图像合成任务，它们采用直接的文本到图像（T2I）生成。然而，直接T2I生成限制了模型处理复杂组合指令的能力，这在现实场景中频繁出现。尽管这个问题至关重要，现有工作主要集中于改进模型的基本图像生成能力。虽然这些改进在一定程度上有所帮助，但它们仍未能充分解决问题。受链式思维（CoT）逐步解决复杂问题的启发，本工作旨在将CoT引入统一生成模型，以解决直接T2I生成无法有效处理的复杂图像生成挑战，从而赋予模型增强的图像生成能力。为实现此目标，我们首先提出功能导向专家（FoXperts），我们模型FoX中的专家并行架构，按功能分配专家。FoXperts解耦了主流模态导向设计中的潜在冲突，为CoT提供了坚实基础。在引入CoT时，第一个问题是如何为复杂图像生成设计它。为此，我们模拟类似人类的艺术工作流程——规划、行动、反思和修正——并提出多模态链式思维（MCoT）方法，因为数据涉及文本和图像。为解决后续挑战——设计有效的MCoT训练范式——我们开发了一种多任务联合训练方案，以解耦方式赋予模型每个MCoT步骤所需的所有能力。该范式避免了收集一致多步数据元组的困难。广泛实验显示，FoX在各种T2I基准上持续优于现有统一模型，在复杂图像生成方面带来显著改进。
</details>

<details>
    <summary>Key points</summary>
    * 功能导向专家（FoXperts）架构
    * 多模态链式思维（MCoT）方法
    * 多任务联合训练方案
    * 模拟人类艺术工作流程
</details>
</details>

---


<details>
<summary><b> SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation</b></summary>

* **Authors:** Zisheng Chen, Chunwei Wang, Xiuwei Chen, Hongbin Xu, Runhui Huang, Jun Zhou, Jianhua Han, Hang Xu, Xiaodan Liang
* **arXiv ID:** 2503.06764
* **One-liner:** Introduced SemHiTok, a unified image tokenizer achieving SOTA in reconstruction and understanding.
* **Published in:** arxiv (9 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.06764) | [[PDF]](https://arxiv.org/pdf/2503.06764) | [[Code]]()

> **核心创新**
> 通过语义引导分层码本解耦语义和像素特征，实现一致表示。

<details>
    <summary>Abstract</summary>
    本文中，我们介绍了SemHiTok，一种通过语义引导分层码本提供一致离散表示的统一图像分词器，用于多模态理解和生成。近期，统一图像分词器激发了研究社区的探索，旨在捕获高级语义特征以用于理解，并保留低级像素特征以用于生成。先前工作尝试通过结合语义蒸馏和像素重建的损失来训练统一图像分词器。然而，由于多模态理解和生成优先考虑的特征级别不同，联合训练方法在实现良好权衡方面面临显著挑战。SemHiTok通过新颖的语义引导分层码本解决了这一挑战，该码本在预训练语义码本上构建像素子码本。这种设计在结构和训练策略上解耦了语义和像素，使分词器能够捕获像素特征，同时保留其理解高级语义信息的能力。我们的实验表明，SemHiTok在LLaVA-v1.5设置下实现了图像重建和多模态理解的SOTA性能。进一步，我们开发了基于SemHiTok的统一MLLM，在多模态理解和生成任务中表现出优越性能。对于理解，SemHiTok在大多数基准上取得了令人印象深刻的性能。对于生成，我们的模型在统一MLLMs中在MJHQ30K上实现了SOTA性能。
</details>

<details>
    <summary>Key points</summary>
    * 语义引导分层码本
    * 在结构和训练上解耦语义和像素特征
    * 在预训练语义码本上构建像素子码本
    * 在MLLM理解和生成中表现出优越性能
</details>
</details>

---


<details>
<summary><b> WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation</b></summary>

* **Authors:** Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, Li Yuan
* **arXiv ID:** 2503.07265
* **One-liner:** Proposed WISE benchmark and WiScore metric for evaluating world knowledge in T2I models.
* **Published in:** arxiv (10 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.07265) | [[PDF]](https://arxiv.org/pdf/2503.07265) | [[Code]](https://github.com/PKU-YuanGroup/WISE)

> **核心创新**
> 超越传统指标，评估复杂语义理解和知识整合在图像生成中的表现。

<details>
    <summary>Abstract</summary>
    文本到图像（T2I）模型能够生成高质量的艺术创作和视觉内容。然而，现有研究和评估标准主要关注图像真实性和浅层文本-图像对齐，缺乏对文本到图像生成中复杂语义理解和世界知识整合的全面评估。为应对这一挑战，我们提出WISE，第一个专门为世界知识驱动的语义评估设计的基准。WISE超越简单的词-像素映射，通过挑战模型使用1000个精心设计的提示，覆盖文化常识、时空推理和自然科学等25个子领域。为克服传统CLIP指标的局限性，我们引入WiScore，一种新颖的定量指标，用于评估知识-图像对齐。通过全面测试20个模型（10个专用T2I模型和10个统一多模态模型）使用1000个结构化提示跨越25个子领域，我们的发现揭示了它们在图像生成过程中有效整合和应用世界知识的能力存在显著局限性，突显了在下一代T2I模型中增强知识融入和应用的关键路径。
</details>

<details>
    <summary>Key points</summary>
    * WISE基准，包含1000个提示覆盖25个子领域
    * WiScore指标用于知识-图像对齐
    * 全面测试20个模型
    * 突显知识整合的局限性
</details>
</details>

---


<details>
<summary><b> DualToken: Towards Unifying Visual Understanding and Generation with Dual Visual Vocabularies</b></summary>

* **Authors:** Wei Song, Yuran Wang, Zijia Song, Yadong Li, Haoze Sun, Weipeng Chen, Zenan Zhou, Jianhua Xu, Jiaqi Wang, Kaicheng Yu
* **arXiv ID:** 2503.14324
* **One-liner:** Introduced DualToken, a unified tokenizer achieving SOTA in both reconstruction and semantic tasks.
* **Published in:** arxiv (18 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.14324) | [[PDF]](https://arxiv.org/pdf/2503.14324) | [[Code]]()

> **核心创新**
> 通过独立码本解耦高和低级特征，解决重建和语义目标之间的冲突。

<details>
    <summary>Abstract</summary>
    视觉理解和生成所需的不同表示空间在大型语言模型的自回归范式中统一它们构成了挑战。为重建训练的视觉分词器擅长捕获低级感知细节，使其非常适合视觉生成，但缺乏用于理解任务的高级语义表示。相反，通过对比学习训练的视觉编码器与语言对齐良好，但难以解码回像素空间以用于生成任务。为弥合这一差距，我们提出DualToken，一种在单一分词器中统一理解和生成表示的方法。然而，直接在单一分词器中集成重建和语义目标会产生冲突，导致重建质量和语义性能下降。DualToken不强制单一码本处理语义和感知信息，而是通过引入高和低级特征的独立码本来解耦它们，有效将其固有冲突转化为协同关系。因此，DualToken在重建和语义任务中均实现了SOTA性能，同时在下游MLLM理解和生成任务中表现出显著有效性。值得注意的是，我们还显示DualToken作为统一分词器，超越了两种不同类型视觉编码器的简单组合，在统一MLLM中提供优越性能。
</details>

<details>
    <summary>Key points</summary>
    * 高和低级特征的独立码本
    * 将冲突转化为协同关系
    * 在MLLM理解和生成中表现出优越性能
    * 超越不同类型编码器的简单组合
</details>
</details>

---


<details>
<summary><b> Unified Multimodal Discrete Diffusion</b></summary>

* **Authors:** Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, Katerina Fragkiadaki
* **arXiv ID:** 2503.20853
* **One-liner:** Introduced UniDisc, a unified multimodal discrete diffusion model outperforming AR models.
* **Published in:** arxiv (26 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.20853) | [[PDF]](https://arxiv.org/pdf/2503.20853) | [[Code]](https://github.com/alexanderswerdlow/unidisc)

> **核心创新**
> 利用离散扩散实现联合文本和图像任务，具有改进的可控性和效率。

<details>
    <summary>Abstract</summary>
    能够理解和生成多种模态的多模态生成模型主要由自回归（AR）方法主导，这些方法从左到右或从上到下顺序处理标记。这些模型联合处理图像、文本、视频和音频，用于图像描述、问答和图像生成等任务。在本工作中，我们探索离散扩散模型作为联合文本和图像领域的统一生成公式，基于其在文本生成中的近期成功。离散扩散模型相对于AR模型具有多个优势，包括改进生成样本质量与多样性的控制、执行联合多模态修复（跨文本和图像领域）的能力，以及通过引导实现生成中更大的可控性。利用这些优势，我们提出了第一个统一多模态离散扩散（UniDisc）模型，能够联合理解和生成文本和图像，用于各种下游任务。我们将UniDisc与多模态AR模型进行比较，进行缩放分析，并证明UniDisc在性能和推理时间计算、增强可控性、可编辑性、修复以及推理时间与生成质量之间的灵活权衡方面优于它们。
</details>

<details>
    <summary>Key points</summary>
    * 统一多模态离散扩散（UniDisc）模型
    * 相对于AR模型在控制和修复方面的优势
    * 联合多模态修复和生成
    * 推理时间与生成质量之间的灵活权衡
</details>
</details>

---


<details>
<summary><b> Harmonizing Visual Representations for Unified Multimodal Understanding and Generation</b></summary>

* **Authors:** Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, Chen Change Loy
* **arXiv ID:** 2503.21979
* **One-liner:** Introduced Harmon, a unified autoregressive framework harmonizing understanding and generation.
* **Published in:** arxiv (27 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.21979) | [[PDF]](https://arxiv.org/pdf/2503.21979) | [[Code]](https://github.com/wusize/Harmon)

> **核心创新**
> 使用共享MAR编码器和三阶段训练过程，在生成和理解中实现SOTA性能。

<details>
    <summary>Abstract</summary>
    在单一多模态框架内统一视觉理解和生成仍然是一个重大挑战，因为这两个固有异质任务需要不同粒度的表示。当前利用向量量化（VQ）或变分自编码器（VAE）进行统一视觉表示的方法优先考虑内在图像特征而非语义，损害了理解性能。在本工作中，我们受掩码图像建模（MIM）的启发，它通过掩码-重建预训练学习丰富语义，并成功扩展到掩码自回归（MAR）图像生成。对MAR编码器表示的初步研究揭示了卓越的线性探测精度和对视觉概念的精确特征响应，这表明MAR在视觉理解任务中具有超越其原始生成角色的潜力。基于这些洞见，我们提出Harmon，一个统一自回归框架，通过共享MAR编码器协调理解和生成任务。通过逐步优化理解和生成能力的三阶段训练过程，Harmon在GenEval、MJHQ30K和WISE基准上实现了SOTA图像生成结果，同时在图像理解基准上匹配专用语义编码器（如Janus）的性能。
</details>

<details>
    <summary>Key points</summary>
    * 共享MAR编码器用于理解和生成
    * 三阶段训练过程
    * 在GenEval、MJHQ30K和WISE上的SOTA结果
    * 匹配专用语义编码器的理解性能
</details>
</details>

---


<details>
<summary><b> YoChameleon: Personalized Vision and Language Generation</b></summary>

* **Authors:** Thao Nguyen, Krishna Kumar Singh, Jing Shi, Trung Bui, Yong Jae Lee, Yuheng Li
* **arXiv ID:** 2504.20998
* **One-liner:** Introduced Yo'Chameleon, the first personalization method for large multimodal models.
* **Published in:** arxiv (29 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.20998) | [[PDF]](https://arxiv.org/pdf/2504.20998) | [[Code]](https://github.com/WisconsinAIVision/YoChameleon)

> **核心创新**
> 使用软提示调优和少样本示例实现个性化图像生成和理解。

<details>
    <summary>Abstract</summary>
    大型多模态模型（例如GPT-4、Gemini、Chameleon）已演变成拥有数百万用户的强大工具。然而，它们仍然是通用模型，缺乏对特定用户概念的个性化知识。先前工作探索了文本生成的个性化，但这些方法如何适应新模态（如图像生成）仍不清楚。本文中，我们介绍Yo'Chameleon，第一个研究大型多模态模型个性化的尝试。给定特定概念的3-5张图像，Yo'Chameleon利用软提示调优嵌入主题特定信息，以（i）回答关于该主题的问题和（ii）重新创建像素级细节以在新上下文中生成该主题的图像。Yo'Chameleon通过（i）自提示优化机制平衡多个模态的性能，和（ii）“软正”图像生成方法在少样本设置中增强图像质量进行训练。
</details>

<details>
    <summary>Key points</summary>
    * 软提示调优用于个性化
    * 自提示优化机制
    * 软正图像生成方法
    * 处理多模态少样本数据
</details>
</details>

---


<details>
<summary><b> X-Fusion: Introducing New Modality to Frozen Large Language Models</b></summary>

* **Authors:** Sicheng Mo, Thao Nguyen, Xun Huang, Siddharth Srinivasan Iyer, Yijun Li, Yuchen Liu, Abhishek Tandon, Eli Shechtman, Krishna Kumar Singh, Yong Jae Lee, Bolei Zhou, Yuheng Li
* **arXiv ID:** 2504.20996
* **One-liner:** Introduced X-Fusion, a framework extending LLMs for multimodal tasks while preserving language capabilities.
* **Published in:** arxiv (29 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.20996) | [[PDF]](https://arxiv.org/pdf/2504.20996) | [[Code]](https://sichengmo.github.io/XFusion/)

> **核心创新**
> 采用双塔设计和冻结LLM参数，实现高效多模态集成。

<details>
    <summary>Abstract</summary>
    我们提出X-Fusion，一个扩展预训练大型语言模型（LLMs）用于多模态任务的框架，同时保留其语言能力。X-Fusion采用双塔设计，具有模态特定权重，保持LLM参数冻结，同时集成视觉特定信息以用于理解和生成。我们的实验表明，X-Fusion在图像到文本和文本到图像任务上持续优于替代架构。我们发现，整合理解导向数据提高了生成质量，减少图像数据噪声增强了整体性能，特征对齐加速了较小模型的收敛，但对较大模型影响最小。我们的发现为构建高效统一多模态模型提供了宝贵洞见。
</details>

<details>
    <summary>Key points</summary>
    * 双塔设计，具有模态特定权重
    * 保持LLM参数冻结
    * 通过理解导向数据改进生成
    * 特征对齐加速收敛
</details>
</details>

---


<details>
<summary><b> Nexus-Gen: A Unified Model for Image Understanding, Generation, and Editing</b></summary>

* **Authors:** Hong Zhang, Zhongjie Duan, Xingjun Wang, Yingda Chen, Yuze Zhao, Yu Zhang
* **arXiv ID:** 2504.21356v1
* **One-liner:** Developed Nexus-Gen, a unified model combining LLM reasoning and diffusion model synthesis for multimodal tasks.
* **Published in:** arxiv (30 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.21356v1) | [[PDF]](https://arxiv.org/pdf/2504.21356v1) | [[Code]]()

> **核心创新**
> 通过双阶段对齐训练和预填充自回归策略，协同语言推理和图像合成。

<details>
    <summary>Abstract</summary>
    统一多模态大语言模型（MLLMs）旨在通过单一框架整合多模态理解和生成能力。尽管它们具有多功能性，现有的开源统一模型在性能上与领域特定架构存在差距。为弥合这一差距，我们提出了Nexus-Gen，一个统一模型，将LLM的语言推理能力与扩散模型的图像合成能力协同结合。为对齐LLM和扩散模型的嵌入空间，我们进行了双阶段对齐训练过程。（1）自回归LLM学习基于多模态输入预测图像嵌入，而（2）视觉解码器被训练从这些嵌入中重建高保真图像。在训练LLM时，我们识别出自回归范式的训练和推理阶段之间存在关键差异，其中连续嵌入空间中的误差累积严重降低生成质量。为避免此问题，我们引入了预填充自回归策略，该策略使用位置嵌入的特殊令牌预填充输入序列，而非连续嵌入。通过双阶段训练，Nexus-Gen发展了综合能力，全面处理图像理解、生成和编辑任务。所有模型、数据集和代码已发布在<a href="https://github.com/modelscope/Nexus-Gen.git" rel="external noopener nofollow" class="link-external link-https">此https URL</a>，以促进该领域的进一步进展。
</details>

<details>
    <summary>Key points</summary>
    * 双阶段对齐训练用于嵌入空间对齐
    * 预填充自回归策略以防止误差累积
    * LLM和扩散模型的整合以实现统一能力
</details>
</details>

---


<details>
<summary><b> T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT</b></summary>

* **Authors:** Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, Hongsheng Li
* **arXiv ID:** 2505.00703
* **One-liner:** Introduced T2I-R1, a reasoning-enhanced text-to-image model using RL and bi-level CoT.
* **Published in:** arxiv (1 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.00703) | [[PDF]](https://arxiv.org/pdf/2505.00703) | [[Code]](https://github.com/CaraJ7/T2I-R1)

> **核心创新**
> 通过BiCoT-GRPO优化的语义级和令牌级CoT推理增强生成。

<details>
    <summary>Abstract</summary>
    大语言模型的最新进展展示了思维链（CoT）和强化学习（RL）如何提高性能。然而，将此类推理策略应用于视觉生成领域仍很大程度上未被探索。在本文中，我们提出了T2I-R1，一种新颖的推理增强文本到图像生成模型，由具有双层CoT推理过程的RL驱动。具体来说，我们识别出两个层级的CoT可用于增强生成的不同阶段：（1）语义级CoT用于提示的高层规划，以及（2）令牌级CoT用于逐块生成过程中的低层像素处理。为更好地协调这两个层级的CoT，我们引入了BiCoT-GRPO，具有生成奖励的集成，在同一训练步骤中无缝优化两个生成CoT。通过将我们的推理策略应用于基线模型Janus-Pro，我们在T2I-CompBench上实现了13%的提升，在WISE基准上实现了19%的提升，甚至超越了最先进模型FLUX.1。代码可在：<a href="https://github.com/CaraJ7/T2I-R1" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 双层CoT：语义级用于提示规划，令牌级用于像素处理
    * 具有集成奖励的BiCoT-GRPO用于协调优化
    * 应用于基线模型Janus-Pro以获得性能增益
</details>
</details>

---


<details>
<summary><b> Ming-Lite-Uni: Advancements in Unified Architecture for Natural Multimodal Interaction</b></summary>

* **Authors:** Inclusion AI, Biao Gong, Cheng Zou, Dandan Zheng, Hu Yu, Jingdong Chen, Jianxin Sun, Junbo Zhao, Jun Zhou, Kaixiang Ji, Lixiang Ru, Libin Wang, Qingpei Guo, Rui Liu, Weilong Chai, Xinyu Xiao, Ziyuan Huang
* **arXiv ID:** 2505.02471
* **One-liner:** Created Ming-Lite-Uni, an open-source unified multimodal framework with native AR capabilities.
* **Published in:** arxiv (5 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.02471) | [[PDF]](https://arxiv.org/pdf/2505.02471) | [[Code]](https://github.com/inclusionAI/Ming/tree/Ming-Lite-Omni-Preview/Ming-unify)

> **核心创新**
> 使用多尺度令牌和对齐的统一视觉生成器和多模态AR模型。

<details>
    <summary>Abstract</summary>
    我们介绍了Ming-Lite-Uni，一个开源多模态框架，具有新设计的统一视觉生成器和针对统一视觉与语言的原生多模态自回归模型。具体来说，该项目提供了集成MetaQueries和M2-omni框架的开源实现，同时引入了新颖的多尺度可学习令牌和多尺度表示对齐策略。通过利用固定MLLM和可学习扩散模型，Ming-Lite-Uni使原生多模态AR模型能够执行文本到图像生成和基于指令的图像编辑任务，扩展其能力超越纯视觉理解。我们的实验结果展示了Ming-Lite-Uni的强大性能，并说明了其交互过程的令人印象深刻的流畅性。所有代码和模型权重已开源，以促进社区内的进一步探索。值得注意的是，这项工作与并发的多模态AI里程碑（如2025年3月25日更新的具有原生图像生成的ChatGPT-4o）保持一致，强调了像Ming-Lite-Uni这样的统一模型在通往AGI道路上的更广泛意义。Ming-Lite-Uni处于alpha阶段，并将很快进一步精炼。
</details>

<details>
    <summary>Key points</summary>
    * 多尺度可学习令牌和表示对齐
    * 固定MLLM和可学习扩散模型用于生成和编辑
    * 支持交互过程的开源实现
</details>
</details>

---


<details>
<summary><b> TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation</b></summary>

* **Authors:** Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, Ying Shan
* **arXiv ID:** 2505.05422
* **One-liner:** Proposed TokLIP, a visual tokenizer enhancing comprehension and generation with semanticized VQ tokens.
* **Published in:** arxiv (8 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.05422) | [[PDF]](https://arxiv.org/pdf/2505.05422) | [[Code]](https://github.com/TencentARC/TokLIP)

> **核心创新**
> 解耦理解和生成的训练目标，实现高效AR训练。

<details>
    <summary>Abstract</summary>
    开创性的基于令牌的工作，如Chameleon和Emu3，为多模态统一奠定了基础，但面临高训练计算开销和由于缺乏高层语义而导致的有限理解性能的挑战。在本文中，我们介绍了TokLIP，一种视觉令牌化器，通过语义化向量量化（VQ）令牌并融入CLIP级语义来增强理解，同时使用标准VQ令牌实现端到端多模态自回归训练。TokLIP将低层离散VQ令牌化器与基于ViT的令牌编码器集成，以捕获高层连续语义。与先前方法（例如VILA-U）将高层特征离散化不同，TokLIP解耦了理解和生成的训练目标，允许直接应用先进的VQ令牌化器，无需定制量化操作。我们的实证结果表明，TokLIP实现了卓越的数据效率，赋予视觉令牌高层语义理解，同时增强低层生成能力，使其非常适合自回归Transformer在理解和生成任务中的应用。代码和模型可在<a href="https://github.com/TencentARC/TokLIP" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * VQ令牌的语义化与CLIP级语义
    * 基于ViT的令牌编码器用于高层语义
    * 无需定制量化的端到端训练
</details>
</details>

---


<details>
<summary><b> Selftok: Discrete Visual Tokens of Autoregression, by Diffusion, and for Reasoning</b></summary>

* **Authors:** Bohan Wang, Zhongqi Yue, Fengda Zhang, Shuo Chen, Li&#39;an Bi, Junzhe Zhang, Xue Song, Kennard Yanting Chan, Jiachun Pan, Weijia Wu, Mingze Zhou, Wang Lin, Kaihang Pan, Saining Zhang, Liyu Jia, Wentao Hu, Wei Zhao, Hanwang Zhang
* **arXiv ID:** 2505.07538
* **One-liner:** Introduced Selftok, a discrete visual tokenizer with AR prior for unified VLM and RL support.
* **Published in:** arxiv (12 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.07538) | [[PDF]](https://arxiv.org/pdf/2505.07538) | [[Code]](https://github.com/selftok-team/SelftokTokenizer)

> **核心创新**
> 为VLMs启用纯AR架构和视觉生成中的有效RL。

<details>
    <summary>Abstract</summary>
    我们完全摒弃了图像表示中的传统空间先验，并引入了一种新颖的离散视觉令牌化器：自一致性令牌化器（Selftok）。在其设计核心，我们通过使用图像生成的反向扩散过程，将自回归（AR）先验——镜像语言的因果结构——组合到视觉令牌中。AR属性使Selftok在以下两个关键方面与传统空间令牌根本不同：- Selftok提供了一种优雅且极简的方法来统一扩散和AR用于视觉语言模型（VLMs）：通过用Selftok令牌表示图像，我们可以使用纯离散自回归架构——类似于LLM中的架构——训练VLM，无需额外模块或训练目标。- 我们在理论上表明AR先验满足Bellman方程，而空间先验不满足。因此，Selftok支持视觉生成的强化学习（RL），其有效性可与LLM中实现的相媲美。除了AR属性，Selftok也是一个最先进的令牌化器，在高质量重建和压缩率之间实现了有利的权衡。我们使用Selftok构建了一个纯AR VLM，用于视觉理解和生成任务。令人印象深刻的是，不使用任何文本-图像训练对，一个简单的策略梯度RL在视觉令牌中工作，可以显著提升视觉生成基准，大幅超越所有现有模型。因此，我们相信Selftok有效解决了视觉令牌无法支持有效RL的长期挑战。当与LLM中RL的成熟优势结合时，这使我们更接近实现真正的多模态LLM。项目页面：<a href="https://selftok-team.github.io/report/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 来自反向扩散过程的AR先验
    * 理论Bellman方程满足用于RL
    * 高质量重建和压缩权衡
</details>
</details>

---


<details>
<summary><b> BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset</b></summary>

* **Authors:** Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, Ran Xu
* **arXiv ID:** 2505.09568
* **One-liner:** Developed BLIP3-o, a unified model with diffusion transformer and sequential pretraining.
* **Published in:** arxiv (14 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.09568) | [[PDF]](https://arxiv.org/pdf/2505.09568) | [[Code]](https://github.com/JiuhaiChen/BLIP3o)

> **核心创新**
> 通过策划数据集和创新设计提高训练效率和生成质量。

<details>
    <summary>Abstract</summary>
    统一图像理解和生成在最近的多模态模型研究中受到越来越多的关注。尽管图像理解的设计选择已被广泛研究，但具有图像生成的统一框架的最优模型架构和训练方法仍未充分探索。受自回归和扩散模型在高品质生成和可扩展性方面的强大潜力驱动，我们对它们在统一多模态设置中的使用进行了全面研究，重点关注图像表示、建模目标和训练策略。基于这些调查，我们引入了一种新颖方法，使用扩散Transformer生成语义丰富的CLIP图像特征，与传统的基于VAE的表示形成对比。这种设计既提高了训练效率，又改善了生成质量。此外，我们证明了统一模型的顺序预训练策略——首先训练图像理解，随后训练图像生成——通过保留图像理解能力同时发展强大的图像生成能力，提供了实际优势。最后，我们通过使用GPT-4o提示多样化的字幕集，覆盖各种场景、物体、人类姿态等，精心策划了一个高质量的指令调优数据集BLIP3o-60k用于图像生成。基于我们的创新模型设计、训练方法和数据集，我们开发了BLIP3-o，一套最先进的统一多模态模型。BLIP3-o在大多数流行基准上实现了卓越性能，涵盖图像理解和生成任务。为促进未来研究，我们完全开源了我们的模型，包括代码、模型权重、训练脚本以及预训练和指令调优数据集。
</details>

<details>
    <summary>Key points</summary>
    * 用于CLIP图像特征的扩散Transformer
    * 顺序预训练：先理解后生成
    * BLIP3o-60k指令调优数据集
</details>
</details>

---


<details>
<summary><b> Exploring the Deep Fusion of Large Language Models and Diffusion Transformers for Text-to-Image Synthesis</b></summary>

* **Authors:** Bingda Tang, Boyang Zheng, Xichen Pan, Sayak Paul, Saining Xie
* **arXiv ID:** 2505.10046
* **One-liner:** Conducted empirical study on LLM-DiT fusion for text-to-image generation.
* **Published in:** arxiv (15 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.10046) | [[PDF]](https://arxiv.org/pdf/2505.10046) | [[Code]](https://github.com/tang-bd/fuse-dit)

> **核心创新**
> 提供了多模态生成的受控比较和可复现训练方法。

<details>
    <summary>Abstract</summary>
    本文未描述新方法；而是对与文本到图像合成最新进展相关的重要但未被充分研究的设计空间进行了彻底探索——具体来说，大语言模型（LLMs）和扩散Transformer（DiTs）的深度融合用于多模态生成。先前研究主要关注整体系统性能，而非与替代方法的详细比较，且关键设计细节和训练方法常未公开。这些差距导致对该方法真正潜力的不确定性。为填补这些差距，我们对文本到图像生成进行了实证研究，执行与已建立基线的受控比较，分析重要设计选择，并提供清晰、可复现的大规模训练方法。我们希望这项工作为多模态生成的未来研究提供有意义的数据点和实用指南。
</details>

<details>
    <summary>Key points</summary>
    * 与已建立基线的受控比较
    * 设计选择和训练策略的分析
    * 清晰、可扩展的训练方法
</details>
</details>

---


<details>
<summary><b> End-to-End Vision Tokenizer Tuning</b></summary>

* **Authors:** Wenxuan Wang, Fan Zhang, Yufeng Cui, Haiwen Diao, Zhuoyan Luo, Huchuan Lu, Jing Liu, Xinlong Wang
* **arXiv ID:** 2505.10562
* **One-liner:** Proposed ETT, an end-to-end vision tokenizer tuning method for joint optimization.
* **Published in:** arxiv (15 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.10562) | [[PDF]](https://arxiv.org/pdf/2505.10562) | [[Code]]()

> **核心创新**
> 实现令牌化和AR任务的联合优化，在不改变架构的情况下提高性能。

<details>
    <summary>Abstract</summary>
    现有视觉令牌化将视觉令牌化器的优化与下游训练隔离，隐含假设视觉令牌能在各种任务（例如图像生成和视觉问答）中良好泛化。为低层重建优化的视觉令牌化器对需要不同表示和语义的下游任务不可知。这种解耦范式引入了关键错位：视觉令牌化的损失可能是目标任务的表示瓶颈。例如，在给定图像中令牌化文本时的错误导致在识别或生成它们时结果不佳。为解决此问题，我们提出了ETT，一种端到端视觉令牌化器调优方法，实现视觉令牌化与目标自回归任务之间的联合优化。与先前仅使用冻结视觉令牌化器的离散索引的自回归模型不同，ETT利用令牌化器码本的视觉嵌入，并使用重建和字幕目标端到端优化视觉令牌化器。ETT可以以最小架构修改无缝集成到现有训练流程中。我们的ETT易于实现和集成，无需调整所使用大语言模型的原始码本或架构。广泛实验表明，我们提出的端到端视觉令牌化器调优解锁了显著的性能增益，即与冻结令牌化器基线相比，在多模态理解和视觉生成任务上提高2-6%，同时保留原始重建能力。我们希望这种非常简单且强大的方法能赋能除图像生成和理解之外的多模态基础模型。
</details>

<details>
    <summary>Key points</summary>
    * 具有重建和字幕目标的端到端调优
    * 以最小修改集成到现有流程中
    * 理解和生成任务的性能增益
</details>
</details>

---


<details>
<summary><b> UniCTokens: Boosting Personalized Understanding and Generation via Unified Concept Tokens</b></summary>

* **Authors:** Ruichuan An, Sihan Yang, Renrui Zhang, Zijun Shen, Ming Lu, Gaole Dai, Hao Liang, Ziyu Guo, Shilin Yan, Yulin Luo, Bocheng Zou, Chaoqun Yang, Wentao Zhang
* **arXiv ID:** 2505.14671
* **One-liner:** Introduced UniCTokens for unified personalized VLM with attribute-reasoning generation.
* **Published in:** arxiv (20 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.14671) | [[PDF]](https://arxiv.org/pdf/2505.14671) | [[Code]](https://github.com/arctanxarc/UniCTokens)

> **核心创新**
> 通过渐进训练增强理解和生成之间的互惠互利。

<details>
    <summary>Abstract</summary>
    个性化模型在理解和生成用户提供的概念方面已展示出显著成功。然而，现有方法使用单独的概念令牌进行理解和生成，将这些任务孤立处理。这可能导致生成具有复杂提示的图像时存在限制。例如，给定概念$\langle bo\rangle$，生成“$\langle bo\rangle$戴着它的帽子”而无需其帽子的额外文本描述。我们称这种生成为\textit{\textbf{个性化属性推理生成}}。为解决这一限制，我们提出了UniCTokens，一种新颖框架，有效将个性化信息集成到统一视觉语言模型（VLM）中进行理解和生成。UniCTokens训练一组统一概念令牌以利用互补语义，提升两个个性化任务。此外，我们提出了一个渐进训练策略，包含三个阶段：理解预热、从理解引导生成、以及从生成深化理解，以增强两个任务之间的互惠互利。为定量评估统一VLM个性化，我们提出了UnifyBench，第一个评估概念理解、概念生成和属性推理生成的基准。在UnifyBench上的实验结果表明，UniCTokens在概念理解、概念生成方面显示出与领先方法竞争的性能，并在个性化属性推理生成中实现了最先进的结果。我们的研究表明，增强的理解改善生成，且生成过程可以为理解提供有价值的见解。我们的代码和数据集将发布在：\href{<a href="https://github.com/arctanxarc/UniCTokens" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}{<a href="https://github.com/arctanxarc/UniCTokens" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}。
</details>

<details>
    <summary>Key points</summary>
    * 用于互补语义的统一概念令牌
    * 三阶段渐进训练策略
    * 用于评估的UnifyBench基准
</details>
</details>

---


<details>
<summary><b> UniGen: Enhanced Training &amp; Test-Time Strategies for Unified Multimodal Understanding and Generation</b></summary>

* **Authors:** Rui Tian, Mingfei Gao, Mingze Xu, Jiaming Hu, Jiasen Lu, Zuxuan Wu, Yinfei Yang, Afshin Dehghan
* **arXiv ID:** 2505.14682
* **One-liner:** Developed UniGen, a unified MLLM with CoT-V strategy for test-time scaling.
* **Published in:** arxiv (20 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.14682) | [[PDF]](https://arxiv.org/pdf/2505.14682) | [[Code]]()

> **核心创新**
> 通过完整训练流程和CoT验证实现SOTA性能。

<details>
    <summary>Abstract</summary>
    我们介绍了UniGen，一个能够进行图像理解和生成的统一多模态大语言模型（MLLM）。我们从数据中心视角研究了UniGen的完整训练流程，包括多阶段预训练、监督微调和直接偏好优化。更重要的是，我们提出了一种新的思维链验证（CoT-V）策略用于测试时扩展，该策略使用简单的Best-of-N测试时策略显著提升了UniGen的图像生成质量。具体来说，CoT-V使UniGen在测试时既能作为图像生成器又能作为验证器，以逐步CoT方式评估文本提示与其生成图像之间的语义对齐。在所有阶段完全使用开源数据集训练，UniGen在一系列图像理解和生成基准上实现了最先进性能，在GenEval上最终得分为0.78，在DPG-Bench上为85.19。通过广泛消融研究，我们的工作提供了可操作的见解，并解决了构建统一MLLMs全生命周期中的关键挑战，为未来研究贡献了有意义的方向。
</details>

<details>
    <summary>Key points</summary>
    * 用于测试时扩展的思维链验证（CoT-V）
    * 多阶段训练：预训练、微调、DPO
    * 开源数据集使用和消融研究
</details>
</details>

---


<details>
<summary><b> OpenUni: A Simple Baseline for Unified Multimodal Understanding and Generation</b></summary>

* **Authors:** Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, Chen Change Loy
* **arXiv ID:** 2505.23661
* **One-liner:** Developed OpenUni, a lightweight open-source baseline for unified multimodal understanding and generation.
* **Published in:** arxiv (29 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.23661) | [[PDF]](https://arxiv.org/pdf/2505.23661) | [[Code]](https://github.com/wusize/OpenUni)

> **核心创新**
> 通过可学习查询和Transformer连接器桥接LLMs与扩散模型，以最小参数实现高质量图像生成和卓越基准性能。

<details>
    <summary>Abstract</summary>
    在本报告中，我们提出了OpenUni，一个简单、轻量级且完全开源的基线模型，用于统一多模态理解与生成。受统一模型学习主流实践的启发，我们采用了一种高效的训练策略，通过一组可学习的查询和一个轻量级基于Transformer的连接器，桥接现成的多模态大语言模型（LLMs）和扩散模型，从而最小化训练复杂性和开销。通过极简的架构选择，我们证明OpenUni能够：1）生成高质量且与指令对齐的图像，2）在标准基准测试（如GenEval、DPG-Bench和WISE）上仅用1.1B和3.1B激活参数就实现卓越性能。为支持开放研究和社区进步，我们在<a href="https://github.com/wusize/OpenUni" rel="external noopener nofollow" class="link-external link-https">此https URL</a>发布了所有模型权重、训练代码和我们整理的训练数据集（包括2300万图像-文本对）。
</details>

<details>
    <summary>Key points</summary>
    * 采用带可学习查询的高效训练策略
    * 使用轻量级基于Transformer的连接器
    * 最小化训练复杂性和开销
    * 发布模型权重、代码和数据集以支持开放研究
</details>
</details>

---


<details>
<summary><b> Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model</b></summary>

* **Authors:** Qingyu Shi, Jinbin Bai, Zhuoran Zhao, Wenhao Chai, Kaidong Yu, Jianzong Wu, Shuangyong Song, Yunhai Tong, Xiangtai Li, Xuelong Li, Shuicheng Yan
* **arXiv ID:** 2505.23606
* **One-liner:** Introduced Muddit, a unified discrete diffusion transformer for fast parallel generation across text and image modalities.
* **Published in:** arxiv (29 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.23606) | [[PDF]](https://arxiv.org/pdf/2505.23606) | [[Code]](https://github.com/M-E-AGI-Lab/Muddit)

> **核心创新**
> 通过将预训练骨干的强视觉先验与轻量级文本解码器集成于统一架构，实现快速并行的多模态生成。

<details>
    <summary>Abstract</summary>
    统一生成模型旨在单一架构和解码范式下处理跨模态的多样化任务——如文本生成、图像生成和视觉语言推理。自回归统一模型因顺序解码而推理缓慢，非自回归统一模型因预训练骨干有限而泛化能力弱。我们引入了Muddit，一个统一的离散扩散Transformer，能够在文本和图像模态上实现快速并行生成。与先前从头训练的统一直散模型不同，Muddit将预训练文本到图像骨干的强视觉先验与轻量级文本解码器集成，在统一架构下实现灵活高质量的多模态生成。实证结果表明，Muddit在质量和效率上与显著更大的自回归模型相比，达到了竞争性或更优的性能。这项工作突显了纯离散扩散在配备强视觉先验时，作为统一生成的可扩展有效骨干的潜力。
</details>

<details>
    <summary>Key points</summary>
    * 利用离散扩散实现可扩展生成
    * 集成预训练文本到图像骨干
    * 采用轻量级文本解码器
    * 在质量和效率上实现竞争性性能
</details>
</details>

---


<details>
<summary><b> UniRL: Self-Improving Unified Multimodal Models via Supervised and Reinforcement Learning</b></summary>

* **Authors:** Weijia Mao, Zhenheng Yang, Mike Zheng Shou
* **arXiv ID:** 2505.23380
* **One-liner:** Proposed UniRL, a self-improving post-training approach for unified multimodal models.
* **Published in:** arxiv (29 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.23380) | [[PDF]](https://arxiv.org/pdf/2505.23380) | [[Code]](https://github.com/showlab/UniRL)

> **核心创新**
> 通过在迭代中生成并使用图像作为训练数据，无需外部数据提升生成和理解性能。

<details>
    <summary>Abstract</summary>
    统一多模态大语言模型（如Show-o和Janus）在生成和理解任务上均取得了强劲性能。然而，这些模型通常依赖大规模数据集，并在预训练阶段需要大量计算。此外，已提出多种后训练方法，但它们往往依赖外部数据或局限于任务特定定制。在本工作中，我们引入了UniRL，一种自我改进的后训练方法。我们的方法使模型能够从提示生成图像，并在每次迭代中将其用作训练数据，无需任何外部图像数据。而且，它使两个任务相互增强：生成的图像用于理解，理解结果用于监督生成。我们探索了监督微调（SFT）和组相对策略优化（GRPO）来优化模型。UniRL提供三个关键优势：（1）无需外部图像数据，因为所有训练样本均在训练期间由模型自身生成；（2）不仅提升单个任务性能，还减少生成与理解之间的不平衡；（3）在后训练阶段仅需几个额外训练步骤。我们在Show-o和Janus上评估UniRL，实现了Show-o的GenEval分数0.77和Janus的0.65。代码和模型将在<a href="https://github.com/showlab/UniRL" rel="external noopener nofollow" class="link-external link-https">此https URL</a>发布。
</details>

<details>
    <summary>Key points</summary>
    * 使用生成图像进行训练，无需外部数据
    * 应用监督微调和GRPO
    * 实现任务间相互增强
    * 仅需额外后训练步骤
</details>
</details>

---


<details>
<summary><b> Are Unified Vision-Language Models Necessary: Generalization Across Understanding and Generation</b></summary>

* **Authors:** Jihai Zhang, Tianle Li, Linjie Li, Zhengyuan Yang, Yu Cheng
* **arXiv ID:** 2505.23043
* **One-liner:** Systematically investigated generalization across understanding and generation tasks in unified VLMs.
* **Published in:** arxiv (29 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.23043) | [[PDF]](https://arxiv.org/pdf/2505.23043) | [[Code]](https://github.com/MajorDavidZhang/Generalization_unified_VLM)

> **核心创新**
> 发现理解和生成任务间的相互益处，以及统一架构中更好的对齐和跨任务知识迁移。

<details>
    <summary>Abstract</summary>
    统一视觉语言模型（VLMs）的最新进展整合了视觉理解和生成能力，引起了广泛关注。其基本假设是，在理解和生成任务上混合训练的统一架构能够实现任务间的相互增强。然而，这一假设在先前统一VLM工作中尚未充分探索。为填补这一空白，本文系统研究了统一VLM中理解和生成任务的泛化能力。具体而言，我们设计了一个与现实场景紧密对齐的数据集，以促进广泛实验和定量评估。我们评估了多种统一VLM架构以验证发现。我们的关键发现如下：首先，使用混合数据训练的统一VLM在各种架构中表现出理解和生成任务的相互益处，且这种益处随数据增加而扩展；其次，多模态输入和输出空间间更好的对齐将导致更好的泛化；第三，生成任务中获取的知识可以迁移到理解任务，且这种跨任务泛化发生在基础语言模型内，超越模态适配器。我们的发现强调了在VLM中统一理解和生成的必要性，为统一VLM的设计和优化提供了宝贵见解。
</details>

<details>
    <summary>Key points</summary>
    * 设计现实对齐数据集用于实验
    * 评估多种统一VLM架构
    * 识别相互益处及随数据扩展
    * 强调基础模型内的跨任务泛化
</details>
</details>

---


<details>
<summary><b> UniWorld-V1: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation</b></summary>

* **Authors:** Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, Yatian Pang, Li Yuan
* **arXiv ID:** 2506.03147
* **One-liner:** Developed UniWorld-V1, a unified generative framework for image understanding, generation, manipulation, and perception.
* **Published in:** arxiv (3 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.03147) | [[PDF]](https://arxiv.org/pdf/2506.03147) | [[Code]](https://github.com/TencentARC/MindOmni)

> **核心创新**
> 使用多模态LLMs和对比编码器的语义特征，以最小训练数据在多样化任务上实现卓越性能。

<details>
    <summary>Abstract</summary>
    尽管现有统一模型在视觉语言理解和文本到图像生成中表现出强劲性能，但在处理图像感知和操作方面仍有限制——这些能力在实际应用中日益重要。最近，OpenAI引入了强大的GPT-4o-Image模型，展示了在全面图像感知和操作方面的先进能力，引发了广泛兴趣。通过精心设计的实验，我们观察到GPT-4o-Image可能依赖语义编码器而非VAE进行特征提取，尽管VAE通常被视为图像操作任务的关键。受此启发，我们提出了UniWorld-V1，一个基于从强大多模态大语言模型和对比语义编码器提取的语义特征构建的统一生成框架。仅使用270万训练数据，UniWorld-V1在多样化任务（包括图像理解、生成、操作和感知）上实现了令人印象深刻的性能。我们完全开源UniWorld-V1框架，包括模型权重、训练和评估脚本及数据集，以促进可重复性和进一步研究。
</details>

<details>
    <summary>Key points</summary>
    * 基于多模态LLMs的语义特征构建
    * 使用对比语义编码器
    * 仅用270万数据训练
    * 开源框架以促进可重复性
</details>
</details>

---


<details>
<summary><b> LaTtE-Flow: Layerwise Timestep-Expert Flow-based Transformer</b></summary>

* **Authors:** Ying Shen, Zhiyang Xu, Jiuhai Chen, Shizhe Diao, Jiaxin Zhang, Yuguang Yao, Joy Rimchala, Ismini Lourentzou, Lifu Huang
* **arXiv ID:** 2506.06952
* **One-liner:** Proposed LaTtE-Flow, an efficient architecture unifying image understanding and generation with fast inference.
* **Published in:** arxiv (8 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.06952) | [[PDF]](https://arxiv.org/pdf/2506.06952) | [[Code]](https://github.com/yingShen-ys/LaTtE-Flow)

> **核心创新**
> 通过将流匹配分布在专门层组并使用时间步条件残差注意力，提高采样效率和推理速度。

<details>
    <summary>Abstract</summary>
    统一图像理解和生成的多模态基础模型的最新进展为在单一框架内处理广泛视觉语言任务开辟了激动人心的途径。尽管有进展，现有统一模型通常需要大量预训练，且难以达到专门模型的性能水平。此外，许多模型图像生成速度慢，限制了其在实时或资源受限环境中的实际部署。在本工作中，我们提出了层间时间步专家流基Transformer（LaTtE-Flow），一种新颖高效的架构，在单一多模态模型中统一图像理解与生成。LaTtE-Flow基于强大预训练视觉语言模型（VLMs）继承强大多模态理解能力，并通过新颖的层间时间步专家流基架构扩展以实现高效图像生成。LaTtE-Flow将流匹配过程分布在专门Transformer层组上，每组负责不同时间步子集。该设计通过每个采样时间步仅激活一小部分层，显著提高采样效率。为进一步提升性能，我们提出了时间步条件残差注意力机制，用于跨层高效信息重用。实验表明，LaTtE-Flow在多模态理解任务上实现强劲性能，同时以约6倍更快推理速度达到与近期统一多模态模型竞争的图像生成质量。
</details>

<details>
    <summary>Key points</summary>
    * 扩展预训练VLMs为流基架构
    * 将流匹配分布在层组间
    * 使用时间步条件残差注意力
    * 实现6倍更快推理且质量竞争
</details>
</details>

---


<details>
<summary><b> Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models</b></summary>

* **Authors:** Francisco Caetano, Christiaan Viviers, Peter H.N. De With, Fons van der Sommen
* **arXiv ID:** 2506.10634
* **One-liner:** Introduced SymmFlow, a symmetrical flow matching framework for unifying semantic segmentation, classification, and image generation.
* **Published in:** arxiv (12 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.10634) | [[PDF]](https://arxiv.org/pdf/2506.10634) | [[Code]]()

> **核心创新**
> 通过双向一致性的对称学习目标，实现语义图像合成的最先进性能。

<details>
    <summary>Abstract</summary>
    流匹配已成为学习分布间连续变换的强大框架，实现高保真生成建模。本工作引入对称流匹配（SymmFlow），一种新公式，在单一模型中统一语义分割、分类和图像生成。使用对称学习目标，SymmFlow联合建模前向和反向变换，确保双向一致性，同时保留足够熵以维持生成多样性。引入新训练目标以在流间显式保留语义信息，实现高效采样同时保留语义结构，允许无需迭代细化的一步分割和分类。与先前在掩码和图像间施加严格一对一映射的方法不同，SymmFlow泛化到灵活条件，支持像素级和图像级类标签。在各种基准上的实验结果表明，SymmFlow在语义图像合成上达到最先进性能，在CelebAMask-HQ上FID分数11.9，COCO-Stuff上7.0，仅用25推理步骤。此外，它在语义分割上提供竞争结果，并在分类任务中展示有前景能力。代码将公开可用。
</details>

<details>
    <summary>Key points</summary>
    * 使用对称学习目标确保双向一致性
    * 引入语义保留训练目标
    * 支持带标签的灵活条件
    * 实现高效一步分割和分类
</details>
</details>

---


<details>
<summary><b> Scale Your Instructions: Enhance the Instruction-Following Fidelity of Unified Image Generation Model by Self-Adaptive Attention Scaling</b></summary>

* **Authors:** Chao Zhou, Tianyi Wei, Nenghai Yu
* **arXiv ID:** 2507.16240
* **One-liner:** Proposed SaaS, a method to address text instruction neglect in unified image generation models.
* **Published in:** arxiv (22 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.16240) | [[PDF]](https://arxiv.org/pdf/2507.16240) | [[Code]](https://github.com/zhouchao-ops/SaaS)

> **核心创新**
> 通过基于时间步间交叉注意力一致性的动态注意力缩放，无需额外训练增强指令跟随保真度。

<details>
    <summary>Abstract</summary>
    统一图像生成模型（如OmniGen）的最新进展使得在单一框架内处理多样化图像生成和编辑任务成为可能，接受多模态、交错文本和图像的任意形式。这种统一架构消除了文本编码器的需求，极大降低模型复杂性并标准化各种图像生成和编辑任务，使其更用户友好。然而，我们发现它存在文本指令忽略问题，尤其是当文本指令包含多个子指令时。为探索此问题，我们对输入进行扰动分析以识别关键步骤和层。通过检查这些关键步骤的交叉注意力图，我们观察到被忽略子指令与输入图像激活间的显著冲突。作为响应，我们提出自适应性注意力缩放（SaaS），一种利用相邻时间步间交叉注意力一致性动态缩放每个子指令注意力激活的方法。我们的SaaS无需额外训练或测试时优化即可增强指令跟随保真度。在基于指令的图像编辑和视觉条件图像生成上的实验结果验证了SaaS的有效性，显示其优于现有方法的指令跟随保真度。代码可在<a href="https://github.com/zhouchao-ops/SaaS" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 通过扰动分析识别关键步骤和层
    * 利用交叉注意力一致性进行动态缩放
    * 应用于图像编辑和生成任务
    * 无需额外训练或优化
</details>
</details>

---


<details>
<summary><b> OneReward: Unified Mask-Guided Image Generation via Multi-Task Human Preference Learning</b></summary>

* **Authors:** Yuan Gong, Xionghui Wang, Jie Wu, Shiyin Wang, Yitong Wang, Xinglong Wu
* **arXiv ID:** 2508.21066
* **One-liner:** Developed OneReward, a unified reinforcement learning framework for multi-task generation using a single reward model.
* **Published in:** arxiv (28 Aug 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2508.21066) | [[PDF]](https://arxiv.org/pdf/2508.21066) | [[Code]](https://github.com/bytedance/OneReward)

> **核心创新**
> 通过使用视觉语言模型作为生成奖励模型，消除任务特定微调，在多样化任务上实现一致性能。

<details>
    <summary>Abstract</summary>
    本文中，我们引入OneReward，一个统一强化学习框架，仅使用\textit{一个奖励}模型即可在不同评估标准下增强模型跨多个任务的生成能力。通过采用单一视觉语言模型（VLM）作为生成奖励模型，该模型能区分给定任务和评估标准下的赢家和输家，它可以有效应用于多任务生成模型，特别是在数据和任务目标多样化的上下文中。我们使用OneReward进行掩码引导图像生成，这可进一步分为多个子任务，如图像填充、图像扩展、对象移除和文本渲染，涉及二进制掩码作为编辑区域。尽管这些领域特定任务共享相同条件范式，它们在底层数据分布和评估指标上差异显著。现有方法通常依赖任务特定监督微调（SFT），限制了泛化和训练效率。基于OneReward，我们开发了Seedream 3.0 Fill，一个掩码引导生成模型，通过多任务强化学习直接在预训练基础模型上训练，无需任务特定SFT。实验结果表明，我们的统一编辑模型在多个评估维度上一致优于商业和开源竞争对手，如Ideogram、Adobe Photoshop和FLUX Fill [Pro]。代码和模型可在：<a href="https://one-reward.github.io" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取
</details>

<details>
    <summary>Key points</summary>
    * 使用单一VLM作为生成奖励模型
    * 应用于掩码引导图像生成任务
    * 消除任务特定SFT需求
    * 在评估中优于竞争对手
</details>
</details>

---


<details>
<summary><b> Reconstruction Alignment Improves Unified Multimodal Models</b></summary>

* **Authors:** Ji Xie, Trevor Darrell, Luke Zettlemoyer, XuDong Wang
* **arXiv ID:** 2509.07295
* **One-liner:** Introduced RecA, a resource-efficient post-training method for aligning understanding and generation in unified multimodal models.
* **Published in:** arxiv (8 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.07295) | [[PDF]](https://arxiv.org/pdf/2509.07295) | [[Code]](https://github.com/HorizonWind2004/reconstruction-alignment)

> **核心创新**
> 通过条件化模型于自身视觉嵌入并使用自监督重建损失，提高生成和编辑保真度。

<details>
    <summary>Abstract</summary>
    统一多模态模型（UMMs）在单一架构内统一视觉理解与生成。然而，传统训练依赖图像-文本对（或序列），其标题通常稀疏且缺失细粒度视觉细节——即使它们用数百词描述简单图像。我们引入重建对齐（RecA），一种资源高效的后训练方法，利用视觉理解编码器嵌入作为密集“文本提示”，提供无需标题的丰富监督。具体而言，RecA将UMM条件化于其自身视觉理解嵌入，并通过自监督重建损失优化以重构输入图像，从而重新对齐理解与生成。尽管简单，RecA广泛适用：在自回归、掩码自回归和基于扩散的UMMs中，它一致提高生成和编辑保真度。仅用27 GPU小时，后训练RecA显著提升图像生成性能在GenEval（0.73$\rightarrow$0.90）和DPGBench（80.93$\rightarrow$88.15），同时提升编辑基准（ImgEdit 3.38$\rightarrow$3.75，GEdit 6.94$\rightarrow$7.25）。值得注意的是，RecA超越更大开源模型，并广泛适用于多样UMM架构，确立其为UMMs的高效通用后训练对齐策略。
</details>

<details>
    <summary>Key points</summary>
    * 利用视觉理解嵌入作为密集提示
    * 使用自监督重建损失
    * 应用于多样UMM架构
    * 以低资源成本实现显著基准提升
</details>
</details>

---


<details>
<summary><b> GenExam: A Multidisciplinary Text-to-Image Exam</b></summary>

* **Authors:** Zhaokai Wang, Penghao Yin, Xiangyu Zhao, Changyao Tian, Yu Qiao, Wenhai Wang, Jifeng Dai, Gen Luo
* **arXiv ID:** 2509.14232
* **One-liner:** Introduced GenExam, the first benchmark for multidisciplinary text-to-image exams.
* **Published in:** arxiv (17 Sep 2025)
* **Links:** [[Paper]](https://www.arxiv.org/abs/2509.14232) | [[PDF]](https://www.arxiv.org/pdf/2509.14232) | [[Code]](https://github.com/OpenGVLab/GenExam)

> **核心创新**
> 通过将图像生成框架化为具有细粒度评分的考试，建立了严格的评估框架。

<details>
    <summary>Abstract</summary>
    考试是专家级智能的基本测试，需要综合理解、推理和生成能力。现有的考试式基准主要关注理解和推理任务，而当前的生成基准强调世界知识和视觉概念的展示，忽略了严格绘图考试的评价。我们引入了GenExam，首个多学科文本到图像考试基准，包含10个学科的1000个样本，考试式提示按四级分类法组织。每个问题配备真实图像和细粒度评分点，以实现语义正确性和视觉合理性的精确评估。实验表明，即使是最先进的模型如GPT-Image-1和Gemini-2.5-Flash-Image，严格得分也低于15%，大多数模型得分几乎为0%，表明我们的基准具有巨大挑战性。通过将图像生成框架化为考试，GenExam对模型整合理解、推理和生成的能力进行了严格评估，为通往通用AGI的道路提供了洞见。我们的基准和评估代码发布于<a href="https://github.com/OpenGVLab/GenExam" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 开发了包含10个学科的1000个样本的基准
    * 按四级分类法组织提示
    * 提供真实图像和评分点以进行精确评估
    * 显示最先进模型得分低，突显挑战性
</details>
</details>

---


<details>
<summary><b> MANZANO: A Simple and Scalable Unified Multimodal Model with a Hybrid Vision Tokenizer</b></summary>

* **Authors:** Yanghao Li, Rui Qian, Bowen Pan, Haotian Zhang, Haoshuo Huang, Bowen Zhang, Jialing Tong, Haoxuan You, Xianzhi Du, Zhe Gan, Hyunjik Kim, Chao Jia, Zhenbang Wang, Yinfei Yang, Mingfei Gao, Zi-Yi Dou, Wenze Hu, Chang Gao, Dongxu Li, Philipp Dufter, Zirui Wang, Guoli Yin, Zhengdong Zhang, Chen Chen, Yang Zhao, Ruoming Pang, Zhifeng Chen
* **arXiv ID:** 2509.16197
* **One-liner:** Presented Manzano, a unified framework reducing performance trade-off between understanding and generation in multimodal LLMs.
* **Published in:** arxiv (19 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.16197) | [[PDF]](https://arxiv.org/pdf/2509.16197) | [[Code]]()

> **核心创新**
> 通过结合混合图像分词器和精心策划的训练配方，在统一模型中实现了最先进的结果。

<details>
    <summary>Abstract</summary>
    能够理解和生成视觉内容的统一多模态大型语言模型（LLMs）具有巨大潜力。然而，现有的开源模型往往在这些能力之间存在性能权衡。我们提出了Manzano，一个简单且可扩展的统一框架，通过结合混合图像分词器和精心策划的训练配方，显著减少了这种张力。一个共享的视觉编码器馈送两个轻量级适配器，在共同语义空间中为图像到文本理解生成连续嵌入，为文本到图像生成生成离散标记。一个统一的自回归LLM预测文本和图像标记的高级语义，随后辅助扩散解码器将图像标记转换为像素。该架构与理解和生成数据的统一训练配方相结合，实现了两种能力的可扩展联合学习。Manzano在统一模型中取得了最先进的结果，并在文本丰富评估中与专业模型竞争。我们的研究表明任务冲突最小，且从模型规模扩展中获得一致增益，验证了混合分词器的设计选择。
</details>

<details>
    <summary>Key points</summary>
    * 使用共享视觉编码器与轻量级适配器生成连续和离散标记
    * 采用统一自回归LLM进行文本和图像标记预测
    * 集成辅助扩散解码器进行像素生成
    * 展示了最小任务冲突和规模扩展增益
</details>
</details>

---


<details>
<summary><b> EditVerse: Unifying Image and Video Editing and Generation with In-Context Learning</b></summary>

* **Authors:** Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, Daniil Pakhomov, Zhe Lin, Soo Ye Kim, Qiang Xu
* **arXiv ID:** 2509.20360
* **One-liner:** Introduced EditVerse, a unified framework for image and video generation and editing in a single model.
* **Published in:** arxiv (24 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.20360) | [[PDF]](https://arxiv.org/pdf/2509.20360) | [[Code]]()

> **核心创新**
> 通过将所有模态表示为统一标记序列，实现了鲁棒的上下文学习和跨模态知识转移。

<details>
    <summary>Abstract</summary>
    基础模型的近期进展突显了统一和扩展的明确趋势，展示了跨多个领域的新兴能力。尽管图像生成和编辑已从任务特定快速过渡到统一框架，但视频生成和编辑由于架构限制和数据稀缺而仍然分散。在这项工作中，我们引入了EditVerse，一个在单一模型中统一图像和视频生成与编辑的框架。通过将所有模态（即文本、图像和视频）表示为统一标记序列，EditVerse利用自注意力实现鲁棒的上下文学习、自然的跨模态知识转移，以及灵活处理任意分辨率和持续时间的输入和输出。为解决视频编辑训练数据的缺乏，我们设计了一个可扩展的数据管道，策划了232K视频编辑样本，并将其与大规模图像和视频数据集结合进行联合训练。此外，我们提出了EditVerseBench，首个基于指令的视频编辑基准，涵盖多样任务和分辨率。广泛的实验和用户研究表明，EditVerse实现了最先进的性能，超越了现有的开源和商业模型，同时在跨模态中展现出新兴的编辑和生成能力。
</details>

<details>
    <summary>Key points</summary>
    * 将文本、图像和视频表示为统一标记序列
    * 设计了包含232K视频编辑样本的可扩展数据管道
    * 创建了EditVerseBench用于基于指令的视频编辑评估
    * 实现了最先进的性能和跨模态新兴能力
</details>
</details>

---


<details>
<summary><b> UniAlignment: Semantic Alignment for Unified Image Generation, Understanding, Manipulation and Perception</b></summary>

* **Authors:** Xinyang Song, Libin Wang, Weining Wang, Shaozhen Liu, Dandan Zheng, Jingdong Chen, Qi Li, Zhenan Sun
* **arXiv ID:** 2509.23760
* **One-liner:** Proposed UniAlignment, a unified multimodal generation framework within a single diffusion transformer.
* **Published in:** arxiv (28 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.23760) | [[PDF]](https://arxiv.org/pdf/2509.23760) | [[Code]]()

> **核心创新**
> 通过双流扩散训练增强了跨模态一致性和指令遵循鲁棒性。

<details>
    <summary>Abstract</summary>
    扩散模型在文本到图像生成中的显著成功引发了将其能力扩展到各种多模态任务的日益增长的兴趣，包括图像理解、操作和感知。这些任务需要跨视觉和文本模态的高级语义理解，特别是在涉及复杂语义指令的场景中。然而，现有方法往往严重依赖视觉语言模型（VLMs）或模块化设计进行语义指导，导致架构分散和计算效率低下。为解决这些挑战，我们提出了UniAlignment，一个在单一扩散变换器内的统一多模态生成框架。UniAlignment引入了双流扩散训练策略，结合了模态内语义对齐和跨模态语义对齐，从而增强了模型的跨模态一致性和指令遵循鲁棒性。此外，我们提出了SemGen-Bench，一个新基准，专门设计用于评估复杂文本指令下的多模态语义一致性。在多个任务和基准上的广泛实验表明，UniAlignment优于现有基线，突显了扩散模型在统一多模态生成中的巨大潜力。
</details>

<details>
    <summary>Key points</summary>
    * 在训练中引入了模态内和跨模态语义对齐
    * 开发了SemGen-Bench用于评估多模态语义一致性
    * 在多个任务中优于现有基线
    * 展示了扩散模型在统一多模态生成中的潜力
</details>
</details>

---


<details>
<summary><b> Query-Kontext: An Unified Multimodal Model for Image Generation and Editing</b></summary>

* **Authors:** Yuxin Song, Wenkai Dong, Shizun Wang, Qi Zhang, Song Xue, Tao Yuan, Hu Yang, Haocheng Feng, Hang Zhou, Xinyan Xiao, Jingdong Wang
* **arXiv ID:** 2509.26641
* **One-liner:** Introduced Query-Kontext, a novel approach bridging VLM and diffusion models for multimodal generative reasoning.
* **Published in:** arxiv (30 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.26641) | [[PDF]](https://arxiv.org/pdf/2509.26641) | [[Code]]()

> **核心创新**
> 将生成推理委托给VLM，同时使用扩散进行高保真合成，匹配或优于最先进方法。

<details>
    <summary>Abstract</summary>
    统一多模态模型（UMMs）在文本到图像生成（T2I）和编辑（TI2I）中展示了卓越性能，无论是实例化为组装统一框架（将强大视觉语言模型与基于扩散的生成器耦合），还是作为朴素统一多模态模型（早期融合理解和生成模态）。我们认为，在当前统一框架中，多模态生成推理的关键能力（包括指令理解、接地和图像引用以保持身份和忠实重建）与高保真合成内在纠缠。在这项工作中，我们引入了Query-Kontext，一种新方法，通过多模态“kontext”（从多模态输入编码的语义线索和粗粒度图像条件）桥接VLM和扩散模型。这种设计将多模态生成推理的复杂能力委托给强大VLM，同时保留扩散模型用于高质量视觉合成的角色。为实现此，我们提出了三阶段渐进训练策略。首先，我们通过多模态kontext标记将VLM连接到轻量级扩散头，以释放VLM的生成推理能力。其次，我们将此头扩展到大型预训练扩散模型以增强视觉细节和真实感。最后，我们引入低级图像编码器以提高图像保真度，并在下游任务上进行指令调优。此外，我们构建了一个全面的数据管道，整合真实、合成和开源数据集，涵盖多样多模态参考到图像场景，包括图像生成、指令驱动编辑、定制生成和多主体组合。实验表明，我们的方法匹配强大统一基线，并在某些情况下甚至优于任务特定最先进方法。
</details>

<details>
    <summary>Key points</summary>
    * 使用多模态kontext标记进行语义线索和图像条件
    * 实施了三阶段渐进训练策略
    * 构建了涵盖多样参考到图像场景的全面数据管道
    * 实现了与统一和任务特定模型的竞争性能
</details>
</details>

---


<details>
<summary><b> M6: A Chinese Multimodal Pretrainer</b></summary>

* **Authors:** Junyang Lin, Rui Men, An Yang, Chang Zhou, Ming Ding, Yichang Zhang, Peng Wang, Ang Wang, Le Jiang, Xianyan Jia, Jie Zhang, Jianwei Zhang, Xu Zou, Zhikang Li, Xiaodong Deng, Jie Liu, Jinbao Xue, Huiling Zhou, Jianxin Ma, Jin Yu, Yong Li, Wei Lin, Jingren Zhou, Jie Tang, Hongxia Yang
* **arXiv ID:** 2103.00823
* **One-liner:** Constructed the largest Chinese multimodal pretraining dataset and proposed the M6 model for unified pretraining.
* **Published in:** arxiv (1 Mar 2021)
* **Links:** [[Paper]](https://arxiv.org/abs/2103.00823) | [[PDF]](https://arxiv.org/pdf/2103.00823) | [[Code]]()

> **核心创新**
> 将模型规模扩展到1000亿参数，并在下游应用中展示了卓越性能，包括文本引导图像生成。

<details>
    <summary>Abstract</summary>
    在这项工作中，我们构建了中文多模态预训练的最大数据集，包含超过1.9TB图像和292GB文本，覆盖广泛领域。我们提出了一种跨模态预训练方法M6，指多模态到多模态多任务巨型变换器，用于在单模态和多模态数据上进行统一预训练。我们将模型规模扩展到100亿和1000亿参数，并构建了中文最大的预训练模型。我们将模型应用于一系列下游应用，并展示了其与强大基线相比的卓越性能。此外，我们专门设计了文本引导图像生成的下游任务，并显示微调后的M6可以生成高分辨率且细节丰富的高质量图像。
</details>

<details>
    <summary>Key points</summary>
    * 构建了包含超过1.9TB图像和292GB文本的数据集
    * 将M6模型扩展到100亿和1000亿参数
    * 应用于各种下游任务并实现高性能
    * 微调用于高分辨率高质量图像生成
</details>
</details>

---


<details>
<summary><b> Flamingo: a Visual Language Model for Few-Shot Learning</b></summary>

* **Authors:** Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karen Simonyan
* **arXiv ID:** 2204.14198
* **One-liner:** Introduced Flamingo, a family of VLMs with in-context few-shot learning capabilities for multimodal tasks.
* **Published in:** arxiv (29 Apr 2022)
* **Links:** [[Paper]](https://arxiv.org/abs/2204.14198) | [[PDF]](https://arxiv.org/pdf/2204.14198) | [[Code]]()

> **核心创新**
> 使用少样本学习在各种图像和视频任务上实现了最先进性能，无需大量微调。

<details>
    <summary>Abstract</summary>
    构建能够仅使用少量标注示例快速适应新任务的模型是多模态机器学习研究的开放挑战。我们引入了Flamingo，一个具有此能力的视觉语言模型（VLM）家族。我们提出了关键架构创新：（i）桥接强大预训练视觉专用和语言专用模型，（ii）处理任意交错视觉和文本数据序列，以及（iii）无缝摄入图像或视频作为输入。得益于其灵活性，Flamingo模型可以在包含任意交错文本和图像的大规模多模态网络语料库上训练，这是赋予其上下文少样本学习能力的关键。我们对模型进行了彻底评估，探索并测量其快速适应各种图像和视频任务的能力。这些包括开放任务如视觉问答，其中模型被提示一个问题需要回答；描述任务，评估描述场景或事件的能力；以及封闭任务如多项选择视觉问答。对于此频谱上的任何任务，单个Flamingo模型可以通过少样本学习实现新的最先进水平，只需用任务特定示例提示模型。在众多基准上，Flamingo优于在数千倍更多任务特定数据上微调的模型。
</details>

<details>
    <summary>Key points</summary>
    * 桥接了预训练视觉专用和语言专用模型
    * 处理了交错视觉和文本数据序列
    * 在大规模多模态网络语料库上训练
    * 在基准上优于更多数据微调的模型
</details>
</details>

---


<details>
<summary><b> SPAE: Semantic Pyramid AutoEncoder for Multimodal Generation with Frozen LLMs</b></summary>

* **Authors:** Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David A. Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, Kevin Murphy, Alexander G. Hauptmann, Lu Jiang
* **arXiv ID:** 2306.17842
* **One-liner:** Introduced Semantic Pyramid AutoEncoder (SPAE) for enabling frozen LLMs to perform multimodal understanding and generation.
* **Published in:** arxiv (30 Jun 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2306.17842) | [[PDF]](https://arxiv.org/pdf/2306.17842) | [[Code]]()

> **核心创新**
> 使冻结LLMs能够生成图像并在理解任务中实现超过25%的改进，标志着此类能力的首次成功。

<details>
    <summary>Abstract</summary>
    在这项工作中，我们引入了语义金字塔自动编码器（SPAE），用于使冻结LLMs能够执行涉及非语言模态（如图像或视频）的理解和生成任务。SPAE在原始像素和从LLM词汇表中提取的可解释词汇标记（或单词）之间进行转换。生成的标记捕获语义含义和视觉重建所需的细粒度细节，有效地将视觉内容翻译为LLM可理解的语言，并赋予其执行广泛多模态任务的能力。我们的方法通过在冻结PaLM 2和GPT 3.5上的上下文学习实验在多样图像理解和生成任务上得到验证。我们的方法标志着首次成功尝试使冻结LLM生成图像内容，同时在相同设置下在图像理解任务中超越最先进性能超过25%。
</details>

<details>
    <summary>Key points</summary>
    * 将像素转换为LLM词汇表中的可解释词汇标记
    * 通过冻结PaLM 2和GPT 3.5的上下文学习验证
    * 在相同设置下在图像理解中超越最先进性能
    * 赋予LLMs多样多模态任务能力
</details>
</details>

---


<details>
<summary><b> Emu: Generative Pretraining in Multimodality</b></summary>

* **Authors:** Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, Xinlong Wang
* **arXiv ID:** 2307.05222
* **One-liner:** Presented Emu, a Transformer-based multimodal foundation model for seamless image and text generation in multimodal contexts.
* **Published in:** arxiv (11 Jul 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2307.05222) | [[PDF]](https://arxiv.org/pdf/2307.05222) | [[Code]](https://github.com/baaivision/Emu)

> **核心创新**
> 通过统一自回归训练在图像和视频模态的零样本/少样本任务中实现了卓越性能。

<details>
    <summary>Abstract</summary>
    我们提出了Emu，一个基于变换器的多模态基础模型，可以在多模态上下文中无缝生成图像和文本。这个全模态模型可以通过单一模型适用于所有自回归训练过程，无差别地接受任何单模态或多模态数据输入（例如交错图像、文本和视频）。首先，视觉信号被编码为嵌入，并与文本标记一起形成交错输入序列。Emu然后通过分类下一个文本标记或回归下一个视觉嵌入的统一目标进行端到端训练。这种多功能多模态性使得能够大规模探索多样预训练数据源，如带有交错帧和文本的视频、带有交错图像和文本的网页，以及网络规模图像-文本对和视频-文本对。Emu可以作为图像到文本和文本到图像任务的通用多模态接口，并支持上下文图像和文本生成。在广泛的零样本/少样本任务中，包括图像描述、视觉问答、视频问答和文本到图像生成，Emu展示了与最先进大型多模态模型相比的卓越性能。通过指令调优扩展的能力如多模态助手也展示了令人印象深刻的性能。
</details>

<details>
    <summary>Key points</summary>
    * 将视觉信号编码为嵌入以形成交错输入序列
    * 使用统一目标训练下一个标记或嵌入预测
    * 支持上下文图像和文本生成
    * 通过指令调优展示了扩展能力
</details>
</details>

---


<details>
<summary><b> NExT-GPT: Any-to-Any Multimodal LLM</b></summary>

* **Authors:** Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, Tat-Seng Chua
* **arXiv ID:** 2309.05519
* **One-liner:** Presented NExT-GPT, an end-to-end any-to-any MM-LLM system for input and output in text, images, videos, and audio.
* **Published in:** arxiv (11 Sep 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2309.05519) | [[PDF]](https://arxiv.org/pdf/2309.05519) | [[Code]](https://github.com/NExT-GPT/NExT-GPT)

> **核心创新**
> 实现了任意多模态组合的感知和生成，具有低成本训练和扩展潜力。

<details>
    <summary>Abstract</summary>
    尽管最近多模态大型语言模型（MM-LLMs）取得了令人兴奋的进展，但它们大多局限于仅输入侧多模态理解，而没有能力在多个模态中产生内容。由于我们人类总是通过各种模态感知世界并与他人交流，开发能够接受和传递任何模态内容的任意到任意MM-LLMs对于人类级AI变得至关重要。为填补这一空白，我们提出了一个端到端通用任意到任意MM-LLM系统NExT-GPT。我们连接一个LLM与多模态适配器和不同扩散解码器，使NExT-GTP能够感知输入并在文本、图像、视频和音频的任意组合中生成输出。通过利用现有训练良好的高性能编码器和解码器，NExT-GPT仅使用某些投影层的少量参数（1%）进行调优，这不仅有利于低成本训练，还便于扩展到更多潜在模态。此外，我们引入了模态切换指令调优（MosIT），并手动策划了一个高质量数据集用于MosIT，基于此NExT-GPT被赋予复杂跨模态语义理解和内容生成能力。总体而言，我们的研究展示了构建能够建模通用模态的AI代理的有希望可能性，为社区中更类似人类AI的研究铺平道路。项目页面：<a href="https://next-gpt.github.io/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>
</details>

<details>
    <summary>Key points</summary>
    * 连接LLM与多模态适配器和扩散解码器
    * 仅使用1%参数进行投影层调优
    * 引入了模态切换指令调优（MosIT）
    * 策划了高质量数据集用于复杂跨模态理解和生成
</details>
</details>

---


<details>
<summary><b> Making LLaMA SEE and Draw with SEED Tokenizer</b></summary>

* **Authors:** Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, Ying Shan
* **arXiv ID:** 2310.01218
* **One-liner:** SEED enables LLMs to perform scalable multimodal autoregression for both comprehension and generation tasks.
* **Published in:** arxiv (2 Oct 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2310.01218) | [[PDF]](https://arxiv.org/pdf/2310.01218) | [[Code]](https://github.com/AILab-CVC/SEED)

> **核心创新**
> SEED引入了一种图像分词器，使文本和图像能够在统一的自动回归Transformer中互换表示和处理。

<details>
    <summary>Abstract</summary>
    大型语言模型（LLM）的巨大成功扩展了多模态的潜力，推动了通用人工智能（AGI）的逐步演进。真正的AGI智能体不仅应具备执行预定义多任务的能力，还应在开放世界环境中展现出涌现能力。然而，尽管近期多模态LLM取得了显著进展，它们仍难以有效统一理解与生成任务，更不用说开放世界的涌现能力。我们认为，突破当前瓶颈的关键在于使文本和图像能够在统一的自动回归Transformer中互换表示和处理。为此，我们提出了SEED，一种精细的图像分词器，赋予LLM同时看和画的能力。我们确定了两个关键设计原则：（1）图像令牌应独立于2D物理补丁位置，并以1D因果依赖生成，展现出与LLM中从左到右自动回归预测机制一致的内在互依性。（2）图像令牌应捕获与词语语义抽象程度一致的高层语义，并在分词器训练阶段针对区分性和重建进行优化。通过SEED令牌，LLM能够在原始训练方案（即下一个词预测）下执行可扩展的多模态自动回归。因此，SEED-LLaMA通过对交错文本和视觉数据进行大规模预训练和指令调优而产生，在广泛的多模态理解和生成任务上展现出令人印象深刻的性能。更重要的是，SEED-LLaMA展现出组合涌现能力，如多轮上下文多模态生成，类似于您的AI助手。
</details>

<details>
    <summary>Key points</summary>
    * 图像令牌设计为独立于2D补丁位置的1D因果依赖。
    * 令牌捕获高层语义，针对区分性和重建进行优化。
    * SEED-LLaMA在交错文本和视觉数据上训练，实现如多轮上下文生成等涌现能力。
</details>
</details>

---


<details>
<summary><b> Kosmos-G: Generating Images in Context with Multimodal Large Language Models</b></summary>

* **Authors:** Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, Furu Wei
* **arXiv ID:** 2310.02992
* **One-liner:** Kosmos-G achieves zero-shot subject-driven image generation with interleaved multi-image and text input.
* **Published in:** arxiv (4 Oct 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2310.02992) | [[PDF]](https://arxiv.org/pdf/2310.02992) | [[Code]](https://github.com/xichenpan/Kosmos-G)

> **核心创新**
> Kosmos-G使用文本模态作为锚点将MLLM输出空间与CLIP对齐，并执行组合指令调优。

<details>
    <summary>Abstract</summary>
    主题驱动图像生成的最新进展取得了显著进步。然而，当前方法在多样化应用场景中仍显不足，因为它们需要测试时调优，且无法接受交错多图像和文本输入。这些限制使它们远离'图像作为图像生成中的外语'的终极目标。本文提出Kosmos-G，一种利用多模态大语言模型（MLLM）的先进多模态感知能力来解决上述挑战的模型。我们的方法使用文本模态作为锚点，将MLLM的输出空间与CLIP对齐，并在精选数据上执行组合指令调优。Kosmos-G展示了在交错多图像和文本输入下零样本主题驱动生成的强大能力。值得注意的是，分数蒸馏指令调优无需修改图像解码器。这使得可以无缝替换CLIP，并轻松集成各种U-Net技术，从细粒度控制到个性化图像解码器变体。我们将Kosmos-G视为实现'图像作为图像生成中的外语'目标的初步尝试。代码可在<a href="https://aka.ms/Kosmos-G" rel="external noopener nofollow" class="link-external link-https">此https URL</a>找到。
</details>

<details>
    <summary>Key points</summary>
    * 使用文本作为锚点将MLLM输出空间与CLIP对齐。
    * 采用分数蒸馏指令调优，无需修改图像解码器。
    * 实现与各种U-Net技术的无缝集成，用于细粒度控制。
</details>
</details>

---


<details>
<summary><b> Gemini: A Family of Highly Capable Multimodal Models</b></summary>

* **Authors:** Gemini Team Google, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul R. Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Jack Krawczyk, Cosmo Du, Ed Chi, Heng-Tze Cheng, Eric Ni, Purvi Shah, Patrick Kane, Betty Chan, Manaal Faruqui, Aliaksei Severyn, Hanzhao Lin, YaGuang Li, Yong Cheng, Abe Ittycheriah, Mahdis Mahdieh, Mia Chen, Pei Sun, Dustin Tran, Sumit Bagri, Balaji Lakshminarayanan, Jeremiah Liu, Andras Orban, Fabian Güra, Hao Zhou, Xinying Song, Aurelien Boffy, Harish Ganapathy, Steven Zheng, HyunJeong Choe, Ágoston Weisz, Tao Zhu, Yifeng Lu, Siddharth Gopal, Jarrod Kahn, Maciej Kula, Jeff Pitman, Rushin Shah, Emanuel Taropa, Majd Al Merey, Martin Baeuml, Zhifeng Chen, Laurent El Shafey, Yujing Zhang, Olcan Sercinoglu, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, Alexandre Frechette, Charlotte Smith, Laura Culp, Lev Proleev, Yi Luan, Xi Chen, James Lottes, Nathan Schucher, Federico Lebron, Alban Rrustemi, Natalie Clay, Phil Crone, Tomas Kocisky, Jeffrey Zhao, Bartek Perz, Dian Yu, Heidi Howard, Adam Bloniarz, Jack W. Rae, Han Lu, Laurent Sifre, Marcello Maggioni, Fred Alcober, Dan Garrette, Megan Barnes, Shantanu Thakoor, Jacob Austin, Gabriel Barth-Maron, William Wong, Rishabh Joshi, Rahma Chaabouni, Deeni Fatiha, Arun Ahuja, Gaurav Singh Tomar, Evan Senter, Martin Chadwick, Ilya Kornakov, Nithya Attaluri, Iñaki Iturrate, Ruibo Liu, Yunxuan Li, Sarah Cogan, Jeremy Chen, Chao Jia, Chenjie Gu, Qiao Zhang, Jordan Grimstad, Ale Jakse Hartman, Xavier Garcia, Thanumalayan Sankaranarayana Pillai, Jacob Devlin, Michael Laskin, Diego de Las Casas, Dasha Valter, Connie Tao, Lorenzo Blanco, Adrià Puigdomènech Badia, David Reitter, Mianna Chen, Jenny Brennan, Clara Rivera, Sergey Brin, Shariq Iqbal, Gabriela Surita, Jane Labanowski, Abhi Rao, Stephanie Winkler, Emilio Parisotto, Yiming Gu, Kate Olszewska, Ravi Addanki, Antoine Miech, Annie Louis, Denis Teplyashin, Geoff Brown, Elliot Catt, Jan Balaguer, Jackie Xiang, Pidong Wang, Zoe Ashwood, Anton Briukhov, Albert Webson, Sanjay Ganapathy, Smit Sanghavi, Ajay Kannan, Ming-Wei Chang, Axel Stjerngren, Josip Djolonga, Yuting Sun, Ankur Bapna, Matthew Aitchison, Pedram Pejman, Henryk Michalewski, Tianhe Yu, Cindy Wang, Juliette Love, Junwhan Ahn, Dawn Bloxwich, Kehang Han, Peter Humphreys, Thibault Sellam, James Bradbury, Varun Godbole, Sina Samangooei, Bogdan Damoc, Alex Kaskasoli, Sébastien M. R. Arnold, Vijay Vasudevan, Shubham Agrawal, Jason Riesa, Dmitry Lepikhin, Richard Tanburn, Srivatsan Srinivasan, Hyeontaek Lim, Sarah Hodkinson, Pranav Shyam, Johan Ferret, Steven Hand, Ankush Garg, Tom Le Paine, Jian Li, Yujia Li, Minh Giang, Alexander Neitz, Zaheer Abbas, Sarah York, Machel Reid, Elizabeth Cole, Aakanksha Chowdhery, Dipanjan Das, Dominika Rogozińska, Vitaliy Nikolaev, Pablo Sprechmann, Zachary Nado, Lukas Zilka, Flavien Prost, Luheng He, Marianne Monteiro, Gaurav Mishra, Chris Welty, Josh Newlan, Dawei Jia, Miltiadis Allamanis, Clara Huiyi Hu, Raoul de Liedekerke, Justin Gilmer, Carl Saroufim, Shruti Rijhwani, Shaobo Hou, Disha Shrivastava, Anirudh Baddepudi, Alex Goldin, Adnan Ozturel, Albin Cassirer, Yunhan Xu, Daniel Sohn, Devendra Sachan, Reinald Kim Amplayo, Craig Swanson, Dessie Petrova, Shashi Narayan, Arthur Guez, Siddhartha Brahma, Jessica Landon, Miteyan Patel, Ruizhe Zhao, Kevin Villela, Luyu Wang, Wenhao Jia, Matthew Rahtz, Mai Giménez, Legg Yeung, James Keeling, Petko Georgiev, Diana Mincu, Boxi Wu, Salem Haykal, Rachel Saputro, Kiran Vodrahalli, James Qin, Zeynep Cankara, Abhanshu Sharma, Nick Fernando, Will Hawkins, Behnam Neyshabur, Solomon Kim, Adrian Hutter, Priyanka Agrawal, Alex Castro-Ros, George van den Driessche, Tao Wang, Fan Yang, Shuo-yiin Chang, Paul Komarek, Ross McIlroy, Mario Lučić, Guodong Zhang, Wael Farhan, Michael Sharman, Paul Natsev, Paul Michel, Yamini Bansal, Siyuan Qiao, Kris Cao, Siamak Shakeri, Christina Butterfield, Justin Chung, Paul Kishan Rubenstein, Shivani Agrawal, Arthur Mensch, Kedar Soparkar, Karel Lenc, Timothy Chung, Aedan Pope, Loren Maggiore, Jackie Kay, Priya Jhakra, Shibo Wang, Joshua Maynez, Mary Phuong, Taylor Tobin, Andrea Tacchetti, Maja Trebacz, Kevin Robinson, Yash Katariya, Sebastian Riedel, Paige Bailey, Kefan Xiao, Nimesh Ghelani, Lora Aroyo, Ambrose Slone, Neil Houlsby, Xuehan Xiong, Zhen Yang, Elena Gribovskaya, Jonas Adler, Mateo Wirth, Lisa Lee, Music Li, Thais Kagohara, Jay Pavagadhi, Sophie Bridgers, Anna Bortsova, Sanjay Ghemawat, Zafarali Ahmed, Tianqi Liu, Richard Powell, Vijay Bolina, Mariko Iinuma, Polina Zablotskaia, James Besley, Da-Woon Chung, Timothy Dozat, Ramona Comanescu, Xiance Si, Jeremy Greer, Guolong Su, Martin Polacek, Raphaël Lopez Kaufman, Simon Tokumine, Hexiang Hu, Elena Buchatskaya, Yingjie Miao, Mohamed Elhawaty, Aditya Siddhant, Nenad Tomasev, Jinwei Xing, Christina Greer, Helen Miller, Shereen Ashraf, Aurko Roy, Zizhao Zhang, Ada Ma, Angelos Filos, Milos Besta, Rory Blevins, Ted Klimenko, Chih-Kuan Yeh, Soravit Changpinyo, Jiaqi Mu, Oscar Chang, Mantas Pajarskas, Carrie Muir, Vered Cohen, Charline Le Lan, Krishna Haridasan, Amit Marathe, Steven Hansen, Sholto Douglas, Rajkumar Samuel, Mingqiu Wang, Sophia Austin, Chang Lan, Jiepu Jiang, Justin Chiu, Jaime Alonso Lorenzo, Lars Lowe Sjösund, Sébastien Cevey, Zach Gleicher, Thi Avrahami, Anudhyan Boral, Hansa Srinivasan, Vittorio Selo, Rhys May, Konstantinos Aisopos, Léonard Hussenot, Livio Baldini Soares, Kate Baumli, Michael B. Chang, Adrià Recasens, Ben Caine, Alexander Pritzel, Filip Pavetic, Fabio Pardo, Anita Gergely, Justin Frye, Vinay Ramasesh, Dan Horgan, Kartikeya Badola, Nora Kassner, Subhrajit Roy, Ethan Dyer, Víctor Campos Campos, Alex Tomala, Yunhao Tang, Dalia El Badawy, Elspeth White, Basil Mustafa, Oran Lang, Abhishek Jindal, Sharad Vikram, Zhitao Gong, Sergi Caelles, Ross Hemsley, Gregory Thornton, Fangxiaoyu Feng, Wojciech Stokowiec, Ce Zheng, Phoebe Thacker, Çağlar Ünlü, Zhishuai Zhang, Mohammad Saleh, James Svensson, Max Bileschi, Piyush Patil, Ankesh Anand, Roman Ring, Katerina Tsihlas, Arpi Vezer, Marco Selvi, Toby Shevlane, Mikel Rodriguez, Tom Kwiatkowski, Samira Daruki, Keran Rong, Allan Dafoe, Nicholas FitzGerald, Keren Gu-Lemberg, Mina Khan, Lisa Anne Hendricks, Marie Pellat, Vladimir Feinberg, James Cobon-Kerr, Tara Sainath, Maribeth Rauh, Sayed Hadi Hashemi, Richard Ives, Yana Hasson, Eric Noland, Yuan Cao, Nathan Byrd, Le Hou, Qingze Wang, Thibault Sottiaux, Michela Paganini, Jean-Baptiste Lespiau, Alexandre Moufarek, Samer Hassan, Kaushik Shivakumar, Joost van Amersfoort, Amol Mandhane, Pratik Joshi, Anirudh Goyal, Matthew Tung, Andrew Brock, Hannah Sheahan, Vedant Misra, Cheng Li, Nemanja Rakićević, Mostafa Dehghani, Fangyu Liu, Sid Mittal, Junhyuk Oh, Seb Noury, Eren Sezener, Fantine Huot, Matthew Lamm, Nicola De Cao, Charlie Chen, Sidharth Mudgal, Romina Stella, Kevin Brooks, Gautam Vasudevan, Chenxi Liu, Mainak Chain, Nivedita Melinkeri, Aaron Cohen, Venus Wang, Kristie Seymore, Sergey Zubkov, Rahul Goel, Summer Yue, Sai Krishnakumaran, Brian Albert, Nate Hurley, Motoki Sano, Anhad Mohananey, Jonah Joughin, Egor Filonov, Tomasz Kępa, Yomna Eldawy, Jiawern Lim, Rahul Rishi, Shirin Badiezadegan, Taylor Bos, Jerry Chang, Sanil Jain, Sri Gayatri Sundara Padmanabhan, Subha Puttagunta, Kalpesh Krishna, Leslie Baker, Norbert Kalb, Vamsi Bedapudi, Adam Kurzrok, Shuntong Lei, Anthony Yu, Oren Litvin, Xiang Zhou, Zhichun Wu, Sam Sobell, Andrea Siciliano, Alan Papir, Robby Neale, Jonas Bragagnolo, Tej Toor, Tina Chen, Valentin Anklin, Feiran Wang, Richie Feng, Milad Gholami, Kevin Ling, Lijuan Liu, Jules Walter, Hamid Moghaddam, Arun Kishore, Jakub Adamek, Tyler Mercado, Jonathan Mallinson, Siddhinita Wandekar, Stephen Cagle, Eran Ofek, Guillermo Garrido, Clemens Lombriser, Maksim Mukha, Botu Sun, Hafeezul Rahman Mohammad, Josip Matak, Yadi Qian, Vikas Peswani, Pawel Janus, Quan Yuan, Leif Schelin, Oana David, Ankur Garg, Yifan He, Oleksii Duzhyi, Anton Älgmyr, Timothée Lottaz, Qi Li, Vikas Yadav, Luyao Xu, Alex Chinien, Rakesh Shivanna, Aleksandr Chuklin, Josie Li, Carrie Spadine, Travis Wolfe, Kareem Mohamed, Subhabrata Das, Zihang Dai, Kyle He, Daniel von Dincklage, Shyam Upadhyay, Akanksha Maurya, Luyan Chi, Sebastian Krause, Khalid Salama, Pam G Rabinovitch, Pavan Kumar Reddy M, Aarush Selvan, Mikhail Dektiarev, Golnaz Ghiasi, Erdem Guven, Himanshu Gupta, Boyi Liu, Deepak Sharma, Idan Heimlich Shtacher, Shachi Paul, Oscar Akerlund, François-Xavier Aubet, Terry Huang, Chen Zhu, Eric Zhu, Elico Teixeira, Matthew Fritze, Francesco Bertolini, Liana-Eleonora Marinescu, Martin Bölle, Dominik Paulus, Khyatti Gupta, Tejasi Latkar, Max Chang, Jason Sanders, Roopa Wilson, Xuewei Wu, Yi-Xuan Tan, Lam Nguyen Thiet, Tulsee Doshi, Sid Lall, Swaroop Mishra, Wanming Chen, Thang Luong, Seth Benjamin, Jasmine Lee, Ewa Andrejczuk, Dominik Rabiej, Vipul Ranjan, Krzysztof Styrc, Pengcheng Yin, Jon Simon, Malcolm Rose Harriott, Mudit Bansal, Alexei Robsky, Geoff Bacon, David Greene, Daniil Mirylenka, Chen Zhou, Obaid Sarvana, Abhimanyu Goyal, Samuel Andermatt, Patrick Siegler, Ben Horn, Assaf Israel, Francesco Pongetti, Chih-Wei &#34;Louis&#34; Chen, Marco Selvatici, Pedro Silva, Kathie Wang, Jackson Tolins, Kelvin Guu, Roey Yogev, Xiaochen Cai, Alessandro Agostini, Maulik Shah, Hung Nguyen, Noah Ó Donnaile, Sébastien Pereira, Linda Friso, Adam Stambler, Adam Kurzrok, Chenkai Kuang, Yan Romanikhin, Mark Geller, ZJ Yan, Kane Jang, Cheng-Chun Lee, Wojciech Fica, Eric Malmi, Qijun Tan, Dan Banica, Daniel Balle, Ryan Pham, Yanping Huang, Diana Avram, Hongzhi Shi, Jasjot Singh, Chris Hidey, Niharika Ahuja, Pranab Saxena, Dan Dooley, Srividya Pranavi Potharaju, Eileen O&#39;Neill, Anand Gokulchandran, Ryan Foley, Kai Zhao, Mike Dusenberry, Yuan Liu, Pulkit Mehta, Ragha Kotikalapudi, Chalence Safranek-Shrader, Andrew Goodman, Joshua Kessinger, Eran Globen, Prateek Kolhar, Chris Gorgolewski, Ali Ibrahim, Yang Song, Ali Eichenbaum, Thomas Brovelli, Sahitya Potluri, Preethi Lahoti, Cip Baetu, Ali Ghorbani, Charles Chen, Andy Crawford, Shalini Pal, Mukund Sridhar, Petru Gurita, Asier Mujika, Igor Petrovski, Pierre-Louis Cedoz, Chenmei Li, Shiyuan Chen, Niccolò Dal Santo, Siddharth Goyal, Jitesh Punjabi, Karthik Kappaganthu, Chester Kwak, Pallavi LV, Sarmishta Velury, Himadri Choudhury, Jamie Hall, Premal Shah, Ricardo Figueira, Matt Thomas, Minjie Lu, Ting Zhou, Chintu Kumar, Thomas Jurdi, Sharat Chikkerur, Yenai Ma, Adams Yu, Soo Kwak, Victor Ähdel, Sujeevan Rajayogam, Travis Choma, Fei Liu, Aditya Barua, Colin Ji, Ji Ho Park, Vincent Hellendoorn, Alex Bailey, Taylan Bilal, Huanjie Zhou, Mehrdad Khatir, Charles Sutton, Wojciech Rzadkowski, Fiona Macintosh, Roopali Vij, Konstantin Shagin, Paul Medina, Chen Liang, Jinjing Zhou, Pararth Shah, Yingying Bi, Attila Dankovics, Shipra Banga, Sabine Lehmann, Marissa Bredesen, Zifan Lin, John Eric Hoffmann, Jonathan Lai, Raynald Chung, Kai Yang, Nihal Balani, Arthur Bražinskas, Andrei Sozanschi, Matthew Hayes, Héctor Fernández Alcalde, Peter Makarov, Will Chen, Antonio Stella, Liselotte Snijders, Michael Mandl, Ante Kärrman, Paweł Nowak, Xinyi Wu, Alex Dyck, Krishnan Vaidyanathan, Raghavender R, Jessica Mallet, Mitch Rudominer, Eric Johnston, Sushil Mittal, Akhil Udathu, Janara Christensen, Vishal Verma, Zach Irving, Andreas Santucci, Gamaleldin Elsayed, Elnaz Davoodi, Marin Georgiev, Ian Tenney, Nan Hua, Geoffrey Cideron, Edouard Leurent, Mahmoud Alnahlawi, Ionut Georgescu, Nan Wei, Ivy Zheng, Dylan Scandinaro, Heinrich Jiang, Jasper Snoek, Mukund Sundararajan, Xuezhi Wang, Zack Ontiveros, Itay Karo, Jeremy Cole, Vinu Rajashekhar, Lara Tumeh, Eyal Ben-David, Rishub Jain, Jonathan Uesato, Romina Datta, Oskar Bunyan, Shimu Wu, John Zhang, Piotr Stanczyk, Ye Zhang, David Steiner, Subhajit Naskar, Michael Azzam, Matthew Johnson, Adam Paszke, Chung-Cheng Chiu, Jaume Sanchez Elias, Afroz Mohiuddin, Faizan Muhammad, Jin Miao, Andrew Lee, Nino Vieillard, Jane Park, Jiageng Zhang, Jeff Stanway, Drew Garmon, Abhijit Karmarkar, Zhe Dong, Jong Lee, Aviral Kumar, Luowei Zhou, Jonathan Evens, William Isaac, Geoffrey Irving, Edward Loper, Michael Fink, Isha Arkatkar, Nanxin Chen, Izhak Shafran, Ivan Petrychenko, Zhe Chen, Johnson Jia, Anselm Levskaya, Zhenkai Zhu, Peter Grabowski, Yu Mao, Alberto Magni, Kaisheng Yao, Javier Snaider, Norman Casagrande, Evan Palmer, Paul Suganthan, Alfonso Castaño, Irene Giannoumis, Wooyeol Kim, Mikołaj Rybiński, Ashwin Sreevatsa, Jennifer Prendki, David Soergel, Adrian Goedeckemeyer, Willi Gierke, Mohsen Jafari, Meenu Gaba, Jeremy Wiesner, Diana Gage Wright, Yawen Wei, Harsha Vashisht, Yana Kulizhskaya, Jay Hoover, Maigo Le, Lu Li, Chimezie Iwuanyanwu, Lu Liu, Kevin Ramirez, Andrey Khorlin, Albert Cui, Tian LIN, Marcus Wu, Ricardo Aguilar, Keith Pallo, Abhishek Chakladar, Ginger Perng, Elena Allica Abellan, Mingyang Zhang, Ishita Dasgupta, Nate Kushman, Ivo Penchev, Alena Repina, Xihui Wu, Tom van der Weide, Priya Ponnapalli, Caroline Kaplan, Jiri Simsa, Shuangfeng Li, Olivier Dousse, Fan Yang, Jeff Piper, Nathan Ie, Rama Pasumarthi, Nathan Lintz, Anitha Vijayakumar, Daniel Andor, Pedro Valenzuela, Minnie Lui, Cosmin Paduraru, Daiyi Peng, Katherine Lee, Shuyuan Zhang, Somer Greene, Duc Dung Nguyen, Paula Kurylowicz, Cassidy Hardin, Lucas Dixon, Lili Janzer, Kiam Choo, Ziqiang Feng, Biao Zhang, Achintya Singhal, Dayou Du, Dan McKinnon, Natasha Antropova, Tolga Bolukbasi, Orgad Keller, David Reid, Daniel Finchelstein, Maria Abi Raad, Remi Crocker, Peter Hawkins, Robert Dadashi, Colin Gaffney, Ken Franko, Anna Bulanova, Rémi Leblond, Shirley Chung, Harry Askham, Luis C. Cobo, Kelvin Xu, Felix Fischer, Jun Xu, Christina Sorokin, Chris Alberti, Chu-Cheng Lin, Colin Evans, Alek Dimitriev, Hannah Forbes, Dylan Banarse, Zora Tung, Mark Omernick, Colton Bishop, Rachel Sterneck, Rohan Jain, Jiawei Xia, Ehsan Amid, Francesco Piccinno, Xingyu Wang, Praseem Banzal, Daniel J. Mankowitz, Alex Polozov, Victoria Krakovna, Sasha Brown, MohammadHossein Bateni, Dennis Duan, Vlad Firoiu, Meghana Thotakuri, Tom Natan, Matthieu Geist, Ser tan Girgin, Hui Li, Jiayu Ye, Ofir Roval, Reiko Tojo, Michael Kwong, James Lee-Thorp, Christopher Yew, Danila Sinopalnikov, Sabela Ramos, John Mellor, Abhishek Sharma, Kathy Wu, David Miller, Nicolas Sonnerat, Denis Vnukov, Rory Greig, Jennifer Beattie, Emily Caveness, Libin Bai, Julian Eisenschlos, Alex Korchemniy, Tomy Tsai, Mimi Jasarevic, Weize Kong, Phuong Dao, Zeyu Zheng, Frederick Liu, Fan Yang, Rui Zhu, Tian Huey Teh, Jason Sanmiya, Evgeny Gladchenko, Nejc Trdin, Daniel Toyama, Evan Rosen, Sasan Tavakkol, Linting Xue, Chen Elkind, Oliver Woodman, John Carpenter, George Papamakarios, Rupert Kemp, Sushant Kafle, Tanya Grunina, Rishika Sinha, Alice Talbert, Diane Wu, Denese Owusu-Afriyie, Cosmo Du, Chloe Thornton, Jordi Pont-Tuset, Pradyumna Narayana, Jing Li, Saaber Fatehi, John Wieting, Omar Ajmeri, Benigno Uria, Yeongil Ko, Laura Knight, Amélie Héliou, Ning Niu, Shane Gu, Chenxi Pang, Yeqing Li, Nir Levine, Ariel Stolovich, Rebeca Santamaria-Fernandez, Sonam Goenka, Wenny Yustalim, Robin Strudel, Ali Elqursh, Charlie Deck, Hyo Lee, Zonglin Li, Kyle Levin, Raphael Hoffmann, Dan Holtmann-Rice, Olivier Bachem, Sho Arora, Christy Koh, Soheil Hassas Yeganeh, Siim Põder, Mukarram Tariq, Yanhua Sun, Lucian Ionita, Mojtaba Seyedhosseini, Pouya Tafti, Zhiyu Liu, Anmol Gulati, Jasmine Liu, Xinyu Ye, Bart Chrzaszcz, Lily Wang, Nikhil Sethi, Tianrun Li, Ben Brown, Shreya Singh, Wei Fan, Aaron Parisi, Joe Stanton, Vinod Koverkathu, Christopher A. Choquette-Choo, Yunjie Li, TJ Lu, Abe Ittycheriah, Prakash Shroff, Mani Varadarajan, Sanaz Bahargam, Rob Willoughby, David Gaddy, Guillaume Desjardins, Marco Cornero, Brona Robenek, Bhavishya Mittal, Ben Albrecht, Ashish Shenoy, Fedor Moiseev, Henrik Jacobsson, Alireza Ghaffarkhah, Morgane Rivière, Alanna Walton, Clément Crepy, Alicia Parrish, Zongwei Zhou, Clement Farabet, Carey Radebaugh, Praveen Srinivasan, Claudia van der Salm, Andreas Fidjeland, Salvatore Scellato, Eri Latorre-Chimoto, Hanna Klimczak-Plucińska, David Bridson, Dario de Cesare, Tom Hudson, Piermaria Mendolicchio, Lexi Walker, Alex Morris, Matthew Mauger, Alexey Guseynov, Alison Reid, Seth Odoom, Lucia Loher, Victor Cotruta, Madhavi Yenugula, Dominik Grewe, Anastasia Petrushkina, Tom Duerig, Antonio Sanchez, Steve Yadlowsky, Amy Shen, Amir Globerson, Lynette Webb, Sahil Dua, Dong Li, Surya Bhupatiraju, Dan Hurt, Haroon Qureshi, Ananth Agarwal, Tomer Shani, Matan Eyal, Anuj Khare, Shreyas Rammohan Belle, Lei Wang, Chetan Tekur, Mihir Sanjay Kale, Jinliang Wei, Ruoxin Sang, Brennan Saeta, Tyler Liechty, Yi Sun, Yao Zhao, Stephan Lee, Pandu Nayak, Doug Fritz, Manish Reddy Vuyyuru, John Aslanides, Nidhi Vyas, Martin Wicke, Xiao Ma, Evgenii Eltyshev, Nina Martin, Hardie Cate, James Manyika, Keyvan Amiri, Yelin Kim, Xi Xiong, Kai Kang, Florian Luisier, Nilesh Tripuraneni, David Madras, Mandy Guo, Austin Waters, Oliver Wang, Joshua Ainslie, Jason Baldridge, Han Zhang, Garima Pruthi, Jakob Bauer, Feng Yang, Riham Mansour, Jason Gelman, Yang Xu, George Polovets, Ji Liu, Honglong Cai, Warren Chen, XiangHai Sheng, Emily Xue, Sherjil Ozair, Christof Angermueller, Xiaowei Li, Anoop Sinha, Weiren Wang, Julia Wiesinger, Emmanouil Koukoumidis, Yuan Tian, Anand Iyer, Madhu Gurumurthy, Mark Goldenson, Parashar Shah, MK Blake, Hongkun Yu, Anthony Urbanowicz, Jennimaria Palomaki, Chrisantha Fernando, Ken Durden, Harsh Mehta, Nikola Momchev, Elahe Rahimtoroghi, Maria Georgaki, Amit Raul, Sebastian Ruder, Morgan Redshaw, Jinhyuk Lee, Denny Zhou, Komal Jalan, Dinghua Li, Blake Hechtman, Parker Schuh, Milad Nasr, Kieran Milan, Vladimir Mikulik, Juliana Franco, Tim Green, Nam Nguyen, Joe Kelley, Aroma Mahendru, Andrea Hu, Joshua Howland, Ben Vargas, Jeffrey Hui, Kshitij Bansal, Vikram Rao, Rakesh Ghiya, Emma Wang, Ke Ye, Jean Michel Sarr, Melanie Moranski Preston, Madeleine Elish, Steve Li, Aakash Kaku, Jigar Gupta, Ice Pasupat, Da-Cheng Juan, Milan Someswar, Tejvi M., Xinyun Chen, Aida Amini, Alex Fabrikant, Eric Chu, Xuanyi Dong, Amruta Muthal, Senaka Buthpitiya, Sarthak Jauhari, Nan Hua, Urvashi Khandelwal, Ayal Hitron, Jie Ren, Larissa Rinaldi, Shahar Drath, Avigail Dabush, Nan-Jiang Jiang, Harshal Godhia, Uli Sachs, Anthony Chen, Yicheng Fan, Hagai Taitelbaum, Hila Noga, Zhuyun Dai, James Wang, Chen Liang, Jenny Hamer, Chun-Sung Ferng, Chenel Elkind, Aviel Atias, Paulina Lee, Vít Listík, Mathias Carlen, Jan van de Kerkhof, Marcin Pikus, Krunoslav Zaher, Paul Müller, Sasha Zykova, Richard Stefanec, Vitaly Gatsko, Christoph Hirnschall, Ashwin Sethi, Xingyu Federico Xu, Chetan Ahuja, Beth Tsai, Anca Stefanoiu, Bo Feng, Keshav Dhandhania, Manish Katyal, Akshay Gupta, Atharva Parulekar, Divya Pitta, Jing Zhao, Vivaan Bhatia, Yashodha Bhavnani, Omar Alhadlaq, Xiaolin Li, Peter Danenberg, Dennis Tu, Alex Pine, Vera Filippova, Abhipso Ghosh, Ben Limonchik, Bhargava Urala, Chaitanya Krishna Lanka, Derik Clive, Yi Sun, Edward Li, Hao Wu, Kevin Hongtongsak, Ianna Li, Kalind Thakkar, Kuanysh Omarov, Kushal Majmundar, Michael Alverson, Michael Kucharski, Mohak Patel, Mudit Jain, Maksim Zabelin, Paolo Pelagatti, Rohan Kohli, Saurabh Kumar, Joseph Kim, Swetha Sankar, Vineet Shah, Lakshmi Ramachandruni, Xiangkai Zeng, Ben Bariach, Laura Weidinger, Tu Vu, Alek Andreev, Antoine He, Kevin Hui, Sheleem Kashem, Amar Subramanya, Sissie Hsiao, Demis Hassabis, Koray Kavukcuoglu, Adam Sadovsky, Quoc Le, Trevor Strohman, Yonghui Wu, Slav Petrov, Jeffrey Dean, Oriol Vinyals
* **arXiv ID:** 2312.11805
* **One-liner:** Gemini advances state-of-the-art in multimodal understanding across image, audio, video, and text.
* **Published in:** arxiv (19 Dec 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2312.11805) | [[PDF]](https://arxiv.org/pdf/2312.11805) | [[Code]]()

> **核心创新**
> Gemini模型在MMLU等基准上达到人类专家水平，并在32个基准中的30个上改进结果。

<details>
    <summary>Abstract</summary>
    本报告介绍了Gemini，一种新的多模态模型家族，在图像、音频、视频和文本理解方面展现出卓越能力。Gemini家族包括Ultra、Pro和Nano尺寸，适用于从复杂推理任务到设备上内存受限用例的各种应用。在广泛基准测试上的评估显示，我们能力最强的Gemini Ultra模型在32个基准中的30个上推进了技术前沿——尤其值得注意的是，它是首个在广泛研究的考试基准MMLU上达到人类专家水平的模型，并在我们检查的20个多模态基准中的每一个都改进了技术水平。我们相信，Gemini家族在跨模态推理和语言理解方面的新能力将支持广泛用例。我们讨论了负责任地对Gemini模型进行后训练并通过服务（包括Gemini、Gemini Advanced、Google AI Studio和Cloud Vertex AI）部署给用户的方法。
</details>

<details>
    <summary>Key points</summary>
    * 包括Ultra、Pro和Nano尺寸，适用于多样化应用。
    * 展现出强大的跨模态推理和语言理解能力。
    * 通过Gemini Advanced和Google AI Studio等服务负责任部署。
</details>
</details>

---


<details>
<summary><b> Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action</b></summary>

* **Authors:** Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, Aniruddha Kembhavi
* **arXiv ID:** 2312.17172
* **One-liner:** Unified-IO 2 is the first autoregressive multimodal model for understanding and generating image, text, audio, and action.
* **Published in:** arxiv (28 Dec 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2312.17172) | [[PDF]](https://arxiv.org/pdf/2312.17172) | [[Code]](https://github.com/allenai/unified-io-2)

> **核心创新**
> Unified-IO 2将多样模态分词到共享语义空间中，并使用单个编码器-解码器Transformer。

<details>
    <summary>Abstract</summary>
    我们提出了Unified-IO 2，首个能够理解和生成图像、文本、音频和动作的自动回归多模态模型。为了统一不同模态，我们将输入和输出——图像、文本、音频、动作、边界框等——分词到共享语义空间中，然后用单个编码器-解码器Transformer模型处理它们。由于使用如此多样化的模态进行训练具有挑战性，我们提出了各种架构改进以稳定模型训练。我们在来自不同来源的大规模多模态预训练语料库上从头开始训练我们的模型，使用多模态去噪混合目标。为了学习广泛的技能，如遵循多模态指令，我们构建并在120个数据集上进行微调，使用提示和增强。通过单个统一模型，Unified-IO 2在GRIT基准上实现了最先进的性能，并在超过35个基准上取得强劲结果，包括图像生成与理解、自然语言理解、视频和音频理解以及机器人操作。我们向研究社区发布所有模型。
</details>

<details>
    <summary>Key points</summary>
    * 将输入和输出（如图像、文本、音频）分词到统一空间。
    * 采用架构改进，使用多模态去噪混合目标稳定训练。
    * 在120个数据集上微调，在GRIT等基准上实现最先进性能。
</details>
</details>

---


<details>
<summary><b> VARGPT: Unified Understanding and Generation in a Visual Autoregressive Multimodal Large Language Model</b></summary>

* **Authors:** Xianwei Zhuang, Yuxin Xie, Yufan Deng, Liming Liang, Jinghan Ru, Yuguo Yin, Yuexian Zou
* **arXiv ID:** 2501.12327
* **One-liner:** VARGPT unifies visual understanding and generation in a single autoregressive framework.
* **Published in:** arxiv (21 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.12327) | [[PDF]](https://arxiv.org/pdf/2501.12327) | [[Code]](https://vargpt-1.github.io)

> **核心创新**
> VARGPT使用下一个令牌预测进行理解，下一个尺度预测进行生成，扩展了LLaVA架构。

<details>
    <summary>Abstract</summary>
    我们提出了VARGPT，一种新颖的多模态大语言模型（MLLM），在单个自动回归框架内统一了视觉理解和生成。VARGPT采用下一个令牌预测范式进行视觉理解，以及下一个尺度预测范式进行视觉自动回归生成。VARGPT创新性地扩展了LLaVA架构，在MLLMs内实现高效的尺度级自动回归视觉生成，同时在单个模型框架内无缝适应混合模态输入和输出。我们的VARGPT在特别精选的数据集上经历三阶段统一训练过程，包括预训练阶段和两个混合视觉指令调优阶段。统一训练策略旨在分别实现视觉和文本特征的对齐、增强理解和生成的指令遵循能力，以及提高视觉生成质量。尽管基于LLaVA架构进行多模态理解，VARGPT在各种以视觉为中心的基准（如视觉问答和推理任务）上显著优于LLaVA-1.5。值得注意的是，VARGPT自然支持自动回归视觉生成和指令到图像合成的能力，展示了其在视觉理解和生成任务中的多功能性。项目页面位于：\url{<a href="https://vargpt-1.github.io/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}。
</details>

<details>
    <summary>Key points</summary>
    * 采用下一个令牌和下一个尺度预测范式。
    * 经历三阶段统一训练以实现对齐和质量增强。
    * 支持自动回归视觉生成和指令到图像合成。
</details>
</details>

---


<details>
<summary><b> Generative Multimodal Models are In-Context Learners</b></summary>

* **Authors:** Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, Xinlong Wang
* **arXiv ID:** 2312.13286
* **One-liner:** Emu2 enhances multimodal in-context learning with 37B parameters, achieving state-of-the-art in few-shot tasks.
* **Published in:** arxiv (20 Dec 2023)
* **Links:** [[Paper]](https://arxiv.org/abs/2312.13286) | [[PDF]](https://arxiv.org/pdf/2312.13286) | [[Code]](https://github.com/baaivision/Emu2)

> **核心创新**
> Emu2使用统一自动回归目标在大规模多模态序列上训练。

<details>
    <summary>Abstract</summary>
    人类轻松解决上下文多模态任务（即仅通过少量演示或简单指令）的能力，是当前多模态系统大多难以模仿的。在这项工作中，我们证明，通过有效扩展，可以显著增强大型多模态模型的任务无关上下文学习能力。我们引入了Emu2，一种具有370亿参数的生成式多模态模型，使用统一自动回归目标在大规模多模态序列上训练。Emu2展现出强大的多模态上下文学习能力，甚至涌现出解决需要即时推理的任务，如视觉提示和对象接地生成。该模型在少样本设置下的多个多模态理解任务上创下新纪录。当指令调优以遵循特定指令时，Emu2进一步在挑战性任务上实现新的最先进水平，如大型多模态模型的问答基准和开放端主题驱动生成。这些成就表明，Emu2可以作为广泛多模态任务的基模型和通用接口。代码和模型公开可用，以促进未来研究。
</details>

<details>
    <summary>Key points</summary>
    * 扩展到370亿参数以改进上下文学习。
    * 展现出视觉提示和对象接地生成等涌现能力。
    * 指令调优后在问答和主题驱动生成上达到顶级性能。
</details>
</details>

---


<details>
<summary><b> MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer</b></summary>

* **Authors:** Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, Hongsheng Li, Yu Qiao, Jifeng Dai
* **arXiv ID:** 2401.10208
* **One-liner:** MM-Interleaved is an end-to-end generative model for interleaved image-text data with improved detail capture.
* **Published in:** arxiv (18 Jan 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2401.10208) | [[PDF]](https://arxiv.org/pdf/2401.10208) | [[Code]](https://github.com/OpenGVLab/MM-Interleaved)

> **核心创新**
> MM-Interleaved使用多尺度和多图像特征同步器访问细粒度图像特征。

<details>
    <summary>Abstract</summary>
    开发用于交错图像-文本数据的生成模型具有研究和实用价值。它要求模型理解交错序列并随后生成图像和文本。然而，现有尝试受限于固定数量的视觉令牌无法有效捕获图像细节的问题，这在多图像场景中尤为突出。为了解决这个问题，本文提出了MM-Interleaved，一种用于交错图像-文本数据的端到端生成模型。它引入了多尺度和多图像特征同步器模块，允许在生成过程中直接访问先前上下文中的细粒度图像特征。MM-Interleaved在配对和交错图像-文本语料库上端到端预训练。它通过监督微调阶段进一步增强，其中模型提高了遵循复杂多模态指令的能力。实验证明了MM-Interleaved在遵循多模态指令识别视觉细节和遵循文本和视觉条件生成一致图像方面的多功能性。代码和模型可在\url{<a href="https://github.com/OpenGVLab/MM-Interleaved" rel="external noopener nofollow" class="link-external link-https">此https URL</a>}找到。
</details>

<details>
    <summary>Key points</summary>
    * 引入多尺度和多图像特征同步器模块。
    * 在配对和交错图像-文本语料库上端到端预训练。
    * 通过监督微调增强复杂多模态指令遵循能力。
</details>
</details>

---


<details>
<summary><b> AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling</b></summary>

* **Authors:** Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, Hang Yan, Jie Fu, Tao Gui, Tianxiang Sun, Yu-Gang Jiang, Xipeng Qiu
* **arXiv ID:** 2402.12226
* **One-liner:** AnyGPT enables any-to-any multimodal conversation using discrete representations without altering LLM architecture.
* **Published in:** arxiv (19 Feb 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2402.12226) | [[PDF]](https://arxiv.org/pdf/2402.12226) | [[Code]](https://github.com/OpenMOSS/AnyGPT)

> **核心创新**
> AnyGPT利用离散表示统一处理语音、文本、图像和音乐。

<details>
    <summary>Abstract</summary>
    我们介绍了AnyGPT，一种任意到任意多模态语言模型，利用离散表示统一处理各种模态，包括语音、文本、图像和音乐。AnyGPT可以在不改变当前大型语言模型（LLM）架构或训练范式的情况下稳定训练。相反，它完全依赖于数据级预处理，促进新模态无缝集成到LLMs中，类似于新语言的整合。我们构建了一个多模态以文本为中心的数据集用于多模态对齐预训练。利用生成模型，我们合成了首个大规模任意到任意多模态指令数据集。它包含108k个多轮对话样本，错综复杂地交织各种模态，从而使模型能够处理任意组合的多模态输入和输出。实验结果表明，AnyGPT能够促进任意到任意多模态对话，同时在所有模态上实现与专用模型相当的性能，证明离散表示可以有效地在语言模型中统一多个模态。演示见<a href="https://junzhan2000.github.io/AnyGPT.github.io/" rel="external noopener nofollow" class="link-external link-https">此https URL</a>。
</details>

<details>
    <summary>Key points</summary>
    * 依赖数据级预处理实现无缝模态集成。
    * 在多模态以文本为中心数据集和合成指令数据集上训练。
    * 在所有模态上实现与专用模型相当的性能。
</details>
</details>

---


<details>
<summary><b> PVC: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models</b></summary>

* **Authors:** Chenyu Yang, Xuan Dong, Xizhou Zhu, Weijie Su, Jiahao Wang, Hao Tian, Zhe Chen, Wenhai Wang, Lewei Lu, Jifeng Dai
* **arXiv ID:** 2412.09613
* **One-liner:** PVC unifies token compression for images and videos, achieving state-of-the-art in video understanding without image performance loss.
* **Published in:** arxiv (12 Dec 2024)
* **Links:** [[Paper]](https://arxiv.org/abs/2412.09613) | [[PDF]](https://arxiv.org/pdf/2412.09613) | [[Code]](https://github.com/OpenGVLab/PVC)

> **核心创新**
> PVC渐进编码和压缩视觉令牌，将图像视为静态视频以保留细节。

<details>
    <summary>Abstract</summary>
    大型视觉语言模型（VLM）已被扩展以理解图像和视频。视觉令牌压缩被用来减少视觉输入的显著令牌长度。为了满足不同任务的需求，现有高性能模型通常使用不同的令牌压缩策略分别处理图像和视频，限制了结合图像和视频的能力。为此，我们将每个图像扩展为'静态'视频，并引入一种统一的令牌压缩策略，称为渐进视觉令牌压缩（PVC），其中每帧的令牌被渐进编码并自适应压缩，以补充先前帧未提取的信息。视频令牌通过利用固有时间冗余高效压缩。图像被重复为静态视频，空间细节可以在多帧中逐渐补充。PVC统一了图像和视频的令牌压缩。在每帧有限令牌数（默认为64个令牌）下，空间细节和时间变化仍可保留。实验显示，我们的模型在各种视频理解基准上实现最先进性能，包括长视频任务和细粒度短视频任务。同时，我们的统一令牌压缩策略在图像基准上无性能损失，特别是在细节敏感任务中。
</details>

<details>
    <summary>Key points</summary>
    * 将图像扩展为静态视频以实现统一令牌压缩。
    * 渐进编码令牌并自适应压缩以补充信息。
    * 在细节敏感图像任务中保持性能，同时在视频基准上表现出色。
</details>
</details>

---


<details>
<summary><b> Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks</b></summary>

* **Authors:** Miran Heo, Min-Hung Chen, De-An Huang, Sifei Liu, Subhashree Radhakrishnan, Seon Joo Kim, Yu-Chiang Frank Wang, Ryo Hachiuma
* **arXiv ID:** 2501.08326
* **One-liner:** Omni-RGPT facilitates region-level comprehension for images and videos using Token Mark for consistent representation.
* **Published in:** arxiv (14 Jan 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2501.08326) | [[PDF]](https://arxiv.org/pdf/2501.08326) | [[Code]]()

> **核心创新**
> Omni-RGPT将Token Mark令牌嵌入视觉特征和文本提示中以建立直接连接。

<details>
    <summary>Abstract</summary>
    我们提出了Omni-RGPT，一种多模态大语言模型，旨在促进图像和视频的区域级理解。为了实现跨时空维度的一致区域表示，我们引入了Token Mark，一组在视觉特征空间内突出目标区域的令牌。这些令牌使用区域提示（如边界框或掩码）直接嵌入到空间区域中，并同时纳入文本提示以指定目标，建立视觉和文本令牌之间的直接连接。为了进一步支持无需轨迹的稳健视频理解，我们引入了一个辅助任务，通过利用令牌的一致性来引导Token Mark，实现跨视频的稳定区域解释。此外，我们引入了一个大规模区域级视频指令数据集（RegVID-300k）。Omni-RGPT在图像和视频基于常识推理基准上实现最先进结果，同时在描述和指代表达理解任务中展现出强劲性能。
</details>

<details>
    <summary>Key points</summary>
    * 引入Token Mark以在视觉特征空间中突出目标区域。
    * 使用辅助任务实现无需轨迹的稳定视频区域解释。
    * 在大规模区域级视频指令数据集（RegVID-300k）上训练。
</details>
</details>

---


<details>
<summary><b> UGen: Unified Autoregressive Multimodal Model with Progressive Vocabulary Learning</b></summary>

* **Authors:** Hongxuan Tang, Hao Liu, Xinyan Xiao
* **arXiv ID:** 2503.21193
* **One-liner:** Introduced UGen, a unified autoregressive multimodal model achieving strong performance in text processing, image understanding, and generation.
* **Published in:** arxiv (27 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.21193) | [[PDF]](https://arxiv.org/pdf/2503.21193) | [[Code]]()

> **核心创新**
> UGen将文本和图像转换为离散标记序列，并使用单一变换器进行统一自回归生成，通过渐进式词汇学习增强。

<details>
    <summary>Abstract</summary>
    我们提出了UGen，一个统一的自回归多模态模型，在文本处理、图像理解和图像生成任务中同时展现出强大性能。
</details>

<details>
    <summary>Key points</summary>
    * 渐进式词汇学习以逐步整合视觉标记
    * 使用单一变换器进行统一自回归生成
    * 相比原始统一自回归方法实现了13.3%的整体性能提升
</details>
</details>

---


<details>
<summary><b> OmniMamba: Efficient and Unified Multimodal Understanding and Generation via State Space Models</b></summary>

* **Authors:** Jialv Zou, Bencheng Liao, Qian Zhang, Wenyu Liu, Xinggang Wang
* **arXiv ID:** 2503.08686
* **One-liner:** Presented OmniMamba, the first linear-architecture-based multimodal generation model with high efficiency and data efficiency.
* **Published in:** arxiv (11 Mar 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2503.08686) | [[PDF]](https://arxiv.org/pdf/2503.08686) | [[Code]](https://github.com/hustvl/OmniMamba)

> **核心创新**
> OmniMamba利用Mamba-2的高计算和内存效率，并引入解耦词汇和任务特定LoRA以实现高效多模态生成。

<details>
    <summary>Abstract</summary>
    我们提出了OmniMamba，首个基于线性架构的多模态生成模型，通过统一的下一个标记预测范式生成文本和图像。
</details>

<details>
    <summary>Key points</summary>
    * 基于Mamba-2的线性架构以提高效率
    * 解耦词汇用于模态特定生成
    * 任务特定LoRA用于参数高效适应
    * 仅使用200万图像-文本对训练，性能具有竞争力
</details>
</details>

---


<details>
<summary><b> VARGPT-v1.1: Improve Visual Autoregressive Large Unified Model via Iterative Instruction Tuning and Reinforcement Learning</b></summary>

* **Authors:** Xianwei Zhuang, Yuxin Xie, Yufan Deng, Dongchao Yang, Liming Liang, Jinghan Ru, Yuguo Yin, Yuexian Zou
* **arXiv ID:** 2504.02949
* **One-liner:** Advanced VARGPT-v1.1 to achieve state-of-the-art performance in multimodal understanding and text-to-image instruction-following.
* **Published in:** arxiv (3 Apr 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2504.02949) | [[PDF]](https://arxiv.org/pdf/2504.02949) | [[Code]](https://github.com/VARGPT-family/VARGPT-v1.1)

> **核心创新**
> VARGPT-v1.1整合迭代视觉指令调优与DPO、扩展语料库和升级主干网络，实现增强生成和新兴编辑功能。

<details>
    <summary>Abstract</summary>
    我们提出了VARGPT-v1.1，一个先进的统一视觉自回归模型，结合迭代视觉指令调优和DPO以增强生成和编辑能力。
</details>

<details>
    <summary>Key points</summary>
    * 迭代视觉指令调优与直接偏好优化
    * 扩展训练语料库包含830万视觉生成指令对
    * 升级语言模型主干使用Qwen2
    * 增强图像生成分辨率和新兴编辑能力
</details>
</details>

---


<details>
<summary><b> FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities</b></summary>

* **Authors:** Jin Wang, Yao Lai, Aoxue Li, Shifeng Zhang, Jiacheng Sun, Ning Kang, Chengyue Wu, Zhenguo Li, Ping Luo
* **arXiv ID:** 2505.20147
* **One-liner:** Introduced FUDOKI, a unified multimodal model based on discrete flow matching as an alternative to autoregressive paradigms.
* **Published in:** arxiv (26 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.20147) | [[PDF]](https://arxiv.org/pdf/2505.20147) | [[Code]](https://github.com/fudoki-hku/FUDOKI)

> **核心创新**
> FUDOKI利用度量诱导概率路径进行迭代精炼和双向上下文集成，并从预训练AR模型初始化。

<details>
    <summary>Abstract</summary>
    我们提出了FUDOKI，一个基于离散流匹配的统一多模态模型，作为传统自回归范式的替代方案。
</details>

<details>
    <summary>Key points</summary>
    * 离散流匹配框架用于迭代精炼
    * 生成过程中的双向上下文集成
    * 从预训练AR基MLLMs初始化
    * 性能与最先进AR模型相当
</details>
</details>

---


<details>
<summary><b> MMaDA: Multimodal Large Diffusion Language Models</b></summary>

* **Authors:** Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, Mengdi Wang
* **arXiv ID:** 2505.15809
* **One-liner:** Developed MMaDA, a unified multimodal diffusion foundation model with strong generalization across reasoning, understanding, and generation.
* **Published in:** arxiv (21 May 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2505.15809) | [[PDF]](https://arxiv.org/pdf/2505.15809) | [[Code]](https://github.com/Gen-Verse/MMaDA)

> **核心创新**
> MMaDA采用统一扩散架构、混合长思维链微调和UniGRPO RL算法，实现无缝多模态集成。

<details>
    <summary>Abstract</summary>
    我们提出了MMaDA，一种新颖的多模态扩散基础模型，在文本推理、多模态理解和文本到图像生成等多样化领域实现卓越性能。
</details>

<details>
    <summary>Key points</summary>
    * 统一扩散架构与模态无关设计
    * 混合长思维链微调
    * UniGRPO统一策略梯度RL算法
    * 在多种任务中超越LLaMA-3-7B和SDXL等模型
</details>
</details>

---


<details>
<summary><b> Many-for-Many: Unify the Training of Multiple Video and Image Generation and Manipulation Tasks</b></summary>

* **Authors:** Tao Yang, Ruibin Li, Yangming Shi, Yuqi Zhang, Qide Dong, Haoran Cheng, Weiguo Feng, Shilei Wen, Bingyue Peng, Lei Zhang
* **arXiv ID:** 2506.01758
* **One-liner:** Introduced a many-for-many unified framework for visual generation and manipulation tasks with improved video generation.
* **Published in:** arxiv (2 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.01758) | [[PDF]](https://arxiv.org/pdf/2506.01758) | [[Code]](https://github.com/leeruibin/MfM)

> **核心创新**
> 该框架使用轻量级适配器和联合图像-视频学习训练单一模型，并整合深度图以增强3D感知。

<details>
    <summary>Abstract</summary>
    我们提出了一个多对多统一框架，利用轻量级适配器和联合图像-视频学习训练单一模型用于多个任务。
</details>

<details>
    <summary>Key points</summary>
    * 轻量级适配器以统一不同条件
    * 联合图像-视频学习策略
    * 使用深度图进行3D空间感知
    * 支持超过10个不同任务，性能具有竞争力
</details>
</details>

---


<details>
<summary><b> Pisces: An Auto-regressive Foundation Model for Image Understanding and Generation</b></summary>

* **Authors:** Zhiyang Xu, Jiuhai Chen, Zhaojiang Lin, Xichen Pan, Lifu Huang, Tianyi Zhou, Madian Khabsa, Qifan Wang, Di Jin, Michihiro Yasunaga, Lili Yu, Xi Victoria Lin, Shaoliang Nie
* **arXiv ID:** 2506.10395
* **One-liner:** Presented Pisces, an autoregressive multimodal foundation model achieving competitive performance in both image understanding and generation.
* **Published in:** arxiv (12 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.10395) | [[PDF]](https://arxiv.org/pdf/2506.10395) | [[Code]]()

> **核心创新**
> Pisces采用解耦视觉编码架构和定制训练技术，结合精细数据管理，实现理解和生成的竞争性能。

<details>
    <summary>Abstract</summary>
    我们提出了Pisces，一个自回归多模态基础模型，通过解耦视觉编码架构和定制训练技术解决模态差异。
</details>

<details>
    <summary>Key points</summary>
    * 解耦视觉编码架构
    * 多模态生成的定制训练技术
    * 精细数据管理、预训练和微调
    * 在超过20个基准测试中表现出强大性能
</details>
</details>

---


<details>
<summary><b> Ming-Omni: A Unified Multimodal Model for Perception and Generation</b></summary>

* **Authors:** Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, Furong Xu, GuangMing Yao, Jun Zhou, Jingdong Chen, Jianxin Sun, Jiajia Liu, Jianjiang Zhu, Jun Peng, Kaixiang Ji, Kaiyou Song, Kaimeng Ren, Libin Wang, Lixiang Ru, Lele Xie, Longhua Tan, Lyuxin Xue, Lan Wang, Mochen Bai, Ning Gao, Pei Chen, Qingpei Guo, Qinglong Zhang, Qiang Xu, Rui Liu, Ruijie Xiong, Sirui Gao, Tinghao Liu, Taisong Li, Weilong Chai, Xinyu Xiao, Xiaomei Wang, Xiaoxue Chen, Xiao Lu, Xiaoyu Li, Xingning Dong, Xuzheng Yu, Yi Yuan, Yuting Gao, Yunxiao Sun, Yipeng Chen, Yifei Wu, Yongjie Lyu, Ziping Ma, Zipeng Feng, Zhijiang Fang, Zhihao Qiu, Ziyuan Huang, Zhengyu He
* **arXiv ID:** 2506.09344
* **One-liner:** Proposed Ming-Omni, a unified multimodal model supporting image, text, audio, and video processing with generation capabilities.
* **Published in:** arxiv (11 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.09344) | [[PDF]](https://arxiv.org/pdf/2506.09344) | [[Code]](https://github.com/inclusionAI/Ming/tree/main)

> **核心创新**
> Ming-Omni使用专用编码器和具有模态特定路由器的MoE架构，实现高效多模态融合和生成。

<details>
    <summary>Abstract</summary>
    我们提出了Ming-Omni，一个统一多模态模型，能够处理图像、文本、音频和视频，并在语音和图像生成中表现出强大能力。
</details>

<details>
    <summary>Key points</summary>
    * 不同模态的专用编码器
    * 具有模态特定路由器的MoE架构
    * 集成先进音频和图像解码器
    * 在模态支持上匹配GPT-4o并开源
</details>
</details>

---


<details>
<summary><b> UniCode$^2$: Cascaded Large-scale Codebooks for Unified Multimodal Understanding and Generation</b></summary>

* **Authors:** Yanzhe Chen, Huasong Zhong, Yan Li, Zhenheng Yang
* **arXiv ID:** 2506.20214
* **One-liner:** Introduced UniCode^2, a cascaded codebook framework for stable and semantically aligned visual tokenization.
* **Published in:** arxiv (25 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.20214) | [[PDF]](https://arxiv.org/pdf/2506.20214) | [[Code]]()

> **核心创新**
> UniCode^2通过聚类SigLIP序列嵌入构建500K条目码本，实现高利用率和与扩散解码器的无缝集成。

<details>
    <summary>Abstract</summary>
    我们提出了UniCode^2，一个级联码本框架，实现大规模、语义对齐和稳定的视觉标记化。
</details>

<details>
    <summary>Key points</summary>
    * 级联码本设计包含冻结和可训练组件
    * 聚类SigLIP序列嵌入以实现对齐
    * 大规模500K条目码本
    * 与预训练扩散解码器的无缝集成
</details>
</details>

---


<details>
<summary><b> Ovis-U1 Technical Report</b></summary>

* **Authors:** Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, Yang Li, Qing-Guo Chen
* **arXiv ID:** 2506.23044
* **One-liner:** Developed Ovis-U1, a unified model integrating multimodal understanding, text-to-image generation, and image editing.
* **Published in:** arxiv (29 Jun 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2506.23044) | [[PDF]](https://arxiv.org/pdf/2506.23044) | [[Code]](https://github.com/AIDC-AI/Ovis-U1)

> **核心创新**
> Ovis-U1使用基于扩散的视觉解码器和统一训练方法，在多个基准测试中取得高分。

<details>
    <summary>Abstract</summary>
    我们提出了Ovis-U1，一个30亿参数统一模型，整合多模态理解、文本到图像生成和图像编辑能力。
</details>

<details>
    <summary>Key points</summary>
    * 基于扩散的视觉解码器与双向标记精炼器
    * 从语言模型出发的统一训练方法
    * 在OpenCompass、DPG-Bench和GenEval上取得高分
    * 在理解、生成和编辑方面性能增强
</details>
</details>

---


<details>
<summary><b> UniLiP: Adapting CLIP for Unified Multimodal Understanding, Generation and Editing</b></summary>

* **Authors:** Hao Tang, Chenwei Xie, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, Liwei Wang
* **arXiv ID:** 2507.23278
* **One-liner:** Proposed UniLIP, a unified framework adapting CLIP for multimodal understanding, generation, and editing with high performance.
* **Published in:** arxiv (31 Jul 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2507.23278) | [[PDF]](https://arxiv.org/pdf/2507.23278) | [[Code]]()

> **核心创新**
> 引入了一种结合自蒸馏的两阶段训练方案，以增强CLIP的重建能力，同时保持理解性能，在GenEval和WISE等基准测试中实现了最先进的结果。

<details>
    <summary>Abstract</summary>
    本文提出了UniLIP，一个统一框架，将CLIP适配用于多模态理解、生成和编辑。尽管CLIP在理解方面表现出色，但缺乏作为统一视觉编码器所需的重建能力。然而，先前基于CLIP的统一方法未能平衡理解和重建，导致语义退化或重建不一致。相比之下，我们引入了一种新颖的两阶段训练方案，结合自蒸馏策略，逐步赋予CLIP高保真重建能力，同时保留其原始理解性能。为了增强生成和编辑中的推理和一致性，我们进一步开发了一种基于MetaQuery框架的双条件架构。该架构联合利用多模态隐藏状态以获取丰富的上下文细节，以及可学习查询嵌入以利用多模态大语言模型（MLLMs）的强大推理能力。凭借先进的图像表示和架构设计，UniLIP展示了卓越的指令遵循和编辑保真度。仅使用1B和3B参数，UniLIP就能超越更大的统一模型，如BAGEL（7B）和Uniworld-V1（12B），在GenEval上达到0.90、WISE上0.63、ImgEdit上3.94的最先进性能。这些结果表明，UniLIP成功扩展了CLIP的应用，确立其连续特征不仅作为理解任务的最佳选择，还在生成和编辑任务中实现了高度竞争力。代码和模型可在<a href="https://github.com/nnnth/UniLIP" rel="external noopener nofollow" class="link-external link-https">此https URL</a>获取。
</details>

<details>
    <summary>Key points</summary>
    * 两阶段训练与自蒸馏
    * 基于MetaQuery的双条件架构
    * 利用多模态隐藏状态和可学习查询嵌入
    * 与多模态大语言模型（MLLMs）的集成
</details>
</details>

---


<details>
<summary><b> Hyper-Bagel: A Unified Acceleration Framework for Multimodal Understanding and Generation</b></summary>

* **Authors:** Yanzuo Lu, Xin Xia, Manlin Zhang, Huafeng Kuang, Jianbin Zheng, Yuxi Ren, Xuefeng Xiao
* **arXiv ID:** 2509.18824
* **One-liner:** Developed Hyper-Bagel, a unified acceleration framework for multimodal tasks, significantly speeding up understanding and generation.
* **Published in:** arxiv (23 Sep 2025)
* **Links:** [[Paper]](https://arxiv.org/abs/2509.18824) | [[PDF]](https://arxiv.org/pdf/2509.18824) | [[Code]]()

> **核心创新**
> 采用推测性解码和多阶段蒸馏，实现了理解任务超过2倍和生成任务高达22倍的加速，同时保持输出质量。

<details>
    <summary>Abstract</summary>
    统一多模态模型因其在联合理解和生成多样化内容方面的卓越能力而近期备受关注。然而，随着上下文整合越来越多的交错多模态令牌，扩散去噪和自回归解码的迭代过程带来了显著的计算开销。为解决此问题，我们提出了Hyper-Bagel，一个统一加速框架，旨在同时加速多模态理解和生成任务。我们的方法采用分治策略，使用推测性解码进行下一令牌预测，以及多阶段蒸馏过程用于扩散去噪。该框架实现了显著的性能提升，在多模态理解中达到超过2倍的加速。对于生成任务，我们得到的无损6-NFE模型在文本到图像生成中实现了16.67倍的加速，在图像编辑中实现了22倍的加速，同时保留了原始模型的高质量输出。我们进一步开发了一种高效的1-NFE模型，能够实现近乎实时的交互式编辑和生成。通过将先进对抗蒸馏与人类反馈学习相结合，该模型实现了终极的成本效益和响应性，使复杂的多模态交互变得无缝且即时。
</details>

<details>
    <summary>Key points</summary>
    * 分治策略与推测性解码
    * 用于扩散去噪的多阶段蒸馏
    * 开发无损6-NFE和高效1-NFE模型
    * 结合对抗蒸馏与人类反馈学习
</details>
</details>

---
