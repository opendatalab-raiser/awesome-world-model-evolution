<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### Modality Consistency

**Objective**: Establish bidirectional mappings between symbolic (language) and perceptual (vision) representations.

**Historical Significance**: These models created the first "symbol-perception bridges," solving the fundamental I/O problem for world models.

#### Representative Works


<details>
<summary><b>CLIP: Learning Transferable Visual Models From Natural Language Supervision</b></summary>

* **Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
* **arXiv ID:** 2103.00020
* **One-liner:** Large-scale contrastive learning on 400M image–text pairs enabling strong zero-shot transfer across vision tasks
* **Published in:** ICML 2021
* **Links:** [[Paper]](https://arxiv.org/abs/2103.00020) | [[PDF]](https://arxiv.org/pdf/2103.00020.pdf) | [[Code]](https://github.com/openai/CLIP)

> **Core Innovation**  
> Contrastively trained on 400 million image–text pairs to establish a unified vision–language representation space, enabling zero-shot cross-task transfer and pioneering the large-scale multimodal training paradigm of “using text as a supervisory signal.”

<details>
    <summary>Abstract</summary>
    State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on.
</details>

<details>
    <summary>Key points</summary>
    * Contrastive learning between image and text features  
    * Weakly supervised web-scale dataset (400M pairs)  
    * Zero-shot classification via text prompts  
    * Sparked multimodal foundation model wave  
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

> **Core Innovation**  
> First to scale a Transformer for text-to-image generation, paired with a discrete VAE codec, demonstrating strong compositional generation of semantics (shape + style + attributes).

<details>
    <summary>Abstract</summary>
    Text-to-image generation has traditionally focused on finding better modeling assumptions for training on a fixed dataset. These assumptions might involve complex architectures, auxiliary losses, or side information such as object part labels or segmentation masks supplied during training. We describe a simple approach for this task based on a transformer that autoregressively models the text and image tokens as a single stream of data. With sufficient data and scale, our approach is competitive with previous domain-specific models when evaluated in a zero-shot fashion.
</details>

<details>
    <summary>Key points</summary>
    * Text-conditioned transformer for image synthesis  
    * Uses VQ-VAE tokenization for images  
    * Compositional reasoning (shape + style + attribute)  
    * Launch of prompt-driven generative vision  
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

> **Core Innovation**  
> First to introduce visual attention into image captioning, enabling the model to focus on key regions and produce interpretable attention maps—laying the groundwork for later Vision Transformers.

<details>
    <summary>Abstract</summary>
    Inspired by recent work in machine translation and object detection, we introduce an attention based model that automatically learns to describe the content of images. We describe how we can train this model in a deterministic manner using standard backpropagation techniques and stochastically by maximizing a variational lower bound. We also show through visualization how the model is able to automatically learn to fix its gaze on salient objects while generating the corresponding words in the output sequence. We validate the use of attention with state-of-the-art performance on three benchmark datasets: Flickr8k, Flickr30k and MS COCO.
</details>

<details>
    <summary>Key points</summary>
    * Early attention mechanism for vision  
    * CNN encoder + RNN decoder  
    * Visual heatmaps → interpretability  
    * Influenced attention-based transformers in vision  
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

> **Core Innovation**  
> Achieves fine-grained text–image correspondence via word-level attention and semantic-alignment losses, markedly improving both the fidelity and semantic consistency of text-to-image generation.

<details>
    <summary>Abstract</summary>
    In this paper, we propose an Attentional Generative Adversarial Network (AttnGAN) that allows attention-driven, multi-stage refinement for fine-grained text-to-image generation. With a novel attentional generative network, the AttnGAN can synthesize fine-grained details at different subregions of the image by paying attentions to the relevant words in the natural language description. In addition, a deep attentional multimodal similarity model is proposed to compute a fine-grained image-text matching loss for training the generator. The proposed AttnGAN significantly outperforms the previous state of the art, boosting the best reported inception score by 14.14% on the CUB dataset and 170.25% on the more challenging COCO dataset. A detailed analysis is also performed by visualizing the attention layers of the AttnGAN. It for the first time shows that the layered attentional GAN is able to automatically select the condition at the word level for generating different parts of the image.
</details>

<details>
    <summary>Key points</summary>
    * Fine-grained word-to-region attention  
    * Multi-stage image refinement  
    * DAMSM loss for text-image consistency  
    * Major milestone before diffusion models  
</details>
</details>

---
