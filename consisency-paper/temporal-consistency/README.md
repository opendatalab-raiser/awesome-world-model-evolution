### Temporal Consistency

**Objective**: Model temporal dynamics, object motion, and causal relationships in video sequences.

**Historical Significance**: Early explorations of the world's "physics engine," capturing regularities in how scenes evolve over time.

#### Representative Works

<details>
<summary><b>PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning</b></summary>

* **Authors:** Yunbo Wang, Haixu Wu, Jianjin Zhang, Zhifeng Gao, Jianmin Wang, Philip S. Yu, Mingsheng Long
* **arXiv ID:** 2103.09504
* **One-liner:** A novel recurrent architecture that decouples memory cells to capture spatio-temporal dynamics in predictive learning of video frames.  
* **Published in:** TPAMI 2023 
* **Links:** [[Paper]](https://arxiv.org/abs/2103.09504) | [[PDF]](https://arxiv.org/pdf/2103.09504.pdf) | [[Code]](https://github.com/thuml/predrnn-pytorch)

> **Core Innovation**  
> Introduces decoupled memory cells embedded in an RNN backbone to separately capture spatial appearance and temporal dynamics, then fuses them into a unified complex-scene representation. A “zig-zag” memory flow propagates information across layers, while a curriculum-learning strategy enables modeling of long-term temporal dependencies.

<details>
    <summary>Abstract</summary>
    The predictive learning of spatiotemporal sequences aims to generate future images by learning from the historical context, where the visual dynamics are believed to have modular structures that can be learned with compositional subsystems. This paper models these structures by presenting PredRNN, a new recurrent network, in which a pair of memory cells are explicitly decoupled, operate in nearly independent transition manners, and finally form unified representations of the complex environment. Concretely, besides the original memory cell of LSTM, this network is featured by a zigzag memory flow that propagates in both bottom-up and top-down directions across all layers, enabling the learned visual dynamics at different levels of RNNs to communicate. It also leverages a memory decoupling loss to keep the memory cells from learning redundant features. We further propose a new curriculum learning strategy to force PredRNN to learn long-term dynamics from context frames, which can be generalized to most sequence-to-sequence models. We provide detailed ablation studies to verify the effectiveness of each component. Our approach is shown to obtain highly competitive results on five datasets for both action-free and action-conditioned predictive learning scenarios.
</details>

<details>
    <summary>Key points</summary>
    * Introduces a memory‐pair structure: two memory cells decoupled in transition to separately capture spatial appearances and temporal dynamics. 
    * Unified representation by merging the outputs of the two memory streams.  
    * Applied to spatiotemporal predictive learning without explicit geometry or flow guidance.  
    * Demonstrates improved future-frame prediction performance on benchmark datasets.  
</details>
</details>

---

<details>
<summary><b>SimVP: Simpler yet Better Video Prediction</b></summary>

* **Authors:** Zhangyang Gao, Cheng Tan, Lirong Wu, Stan Z. Li
* **arXiv ID:** 2206.05099
* **One-liner:** A purely convolutional video-prediction model trained end-to-end with MSE loss, showing that simplicity can outperform more complex architectures.  
* **Published in:** CVPR 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2206.05099) | [[PDF]](https://arxiv.org/pdf/2206.05099.pdf) | [[Code]](https://github.com/A4Bio/SimVP)

> **Core Innovation**  
> Proposes an ultra-simple end-to-end video-prediction framework that uses only a CNN encoder–decoder trained with plain MSE loss; without RNNs, Transformers, or any sophisticated modules, it matches or surpasses far more complex models on multiple benchmarks.

<details>
    <summary>Abstract</summary>
    From CNN, RNN, to ViT, we have witnessed remarkable advancements in video prediction, incorporating auxiliary inputs, elaborate neural architectures, and sophisticated training strategies. We admire these progresses but are confused about the necessity: is there a simple method that can perform comparably well? This paper proposes SimVP, a simple video prediction model that is completely built upon CNN and trained by MSE loss in an end-to-end fashion. Without introducing any additional tricks and complicated strategies, we can achieve state-of-the-art performance on five benchmark datasets. Through extended experiments, we demonstrate that SimVP has strong generalization and extensibility on real-world datasets. The significant reduction of training cost makes it easier to scale to complex scenarios. We believe SimVP can serve as a solid baseline to stimulate the further development of video prediction. 
</details>

<details>
    <summary>Key points</summary>
    * Model based only on convolutional encoder-decoder structure (no RNN, no transformer).
    * Trained with straightforward Mean Squared Error (MSE) loss end-to-end.  
    * Demonstrates that simpler architectures can achieve competitive or superior results in video prediction benchmarks.  
    * Reduces architectural complexity and computational overhead compared to more sophisticated models.  
</details>
</details>

---

<details>
<summary><b>Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning</b></summary>

* **Authors:** Cheng Tan, Zhangyang Gao, Lirong Wu, Yongjie Xu, Jun Xia, Siyuan Li, Stan Z. Li  
* **arXiv ID:** 2206.12126 
* **One-liner:** Introduces a parallelizable temporal‐attention module (TAU) that splits intra-frame and inter-frame attention to improve efficiency in spatiotemporal predictive learning.  
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2206.12126) | [[PDF]](https://arxiv.org/pdf/2206.12126.pdf)

> **Core Innovation**  
> Introduces the Temporal Attention Unit (TAU) that disentles “intra-frame static attention” from “inter-frame dynamic attention” within spatiotemporal prediction modules, enabling fully parallel computation, and further strengthens inter-frame change detection via differential-divergence regularization.

<details>
    <summary>Abstract</summary>
    Spatiotemporal predictive learning aims to generate future frames by learning from historical frames. In this paper, we investigate existing methods and present a general framework of spatiotemporal predictive learning, in which the spatial encoder and decoder capture intra-frame features and the middle temporal module catches inter-frame correlations. While the mainstream methods employ recurrent units to capture long-term temporal dependencies, they suffer from low computational efficiency due to their unparallelizable architectures. To parallelize the temporal module, we propose the Temporal Attention Unit (TAU), which decomposes the temporal attention into intra-frame statical attention and inter-frame dynamical attention. Moreover, while the mean squared error loss focuses on intra-frame errors, we introduce a novel differential divergence regularization to take inter-frame variations into account. Extensive experiments demonstrate that the proposed method enables the derived model to achieve competitive performance on various spatiotemporal prediction benchmarks.
</details>

<details>
    <summary>Key points</summary>
    * Proposes “Temporal Attention Unit (TAU)” that separates intra-frame static attention from inter-frame dynamic attention for efficiency and parallelism.
    * Introduces differential divergence regularization to better account for temporal variations (inter-frame changes).  
    * Framework decouples spatial encoding/decoding and temporal module, enabling better scalability.  
    * Shows improved computational efficiency while maintaining predictive accuracy on benchmark datasets.  
</details>
</details>

---

<details>
<summary><b>VideoGPT: Video Generation using VQ-VAE and Transformers</b></summary>

* **Authors:** Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas  
* **arXiv ID:** 2104.10157
* **One-liner:** A minimal architecture combining VQ-VAE and GPT-style transformer to scale generative modeling from images to natural videos.  
* **Published in:** arXiv 2021  
* **Links:** [[Paper]](https://arxiv.org/abs/2104.10157) | [[PDF]](https://arxiv.org/pdf/2104.10157.pdf) | [[Code]](https://github.com/wilson1yan/VideoGPT)

> **Core Innovation**  
> Applies VQ-VAE to learn discrete video latents (3D conv + axial self-attention) and then performs autoregressive modeling with a GPT-style Transformer over these tokens, enabling likelihood-based generation on natural videos.

<details>
    <summary>Abstract</summary>
    We present VideoGPT: a conceptually simple architecture for scaling likelihood based generative modeling to natural videos. VideoGPT uses VQ-VAE that learns downsampled discrete latent representations of a raw video by employing 3D convolutions and axial self-attention. A simple GPT-like architecture is then used to autoregressively model the discrete latents using spatio-temporal position encodings. Despite the simplicity in formulation and ease of training, our architecture is able to generate samples competitive with state-of-the-art GAN models for video generation on the BAIR Robot dataset, and generate high fidelity natural videos from UCF-101 and Tumbler GIF Dataset (TGIF). We hope our proposed architecture serves as a reproducible reference for a minimalistic implementation of transformer based video generation models. 
</details>

<details>
    <summary>Key points</summary>
    * Uses VQ-VAE to learn discrete latent representations of video using 3D convolutions + axial self-attention.  
    * Uses a GPT-style autoregressive transformer on those discrete latents with spatio-temporal positional encoding.  
    * Achieves competitive quality compared to GAN-based video generation models, but with simpler likelihood-based framework.  
    * Provides a reproducible baseline for transformer-based video generation on natural video datasets.  
</details>
</details>

---

<details>
<summary><b>Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions</b></summary>

* **Authors:** Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, Dumitru Erhan  
* **arXiv ID:** 2210.02399
* **One-liner:** Generates variable-length videos from a sequence of open-domain text prompts by tokenizing video representation and using a masked‐transformer conditioned on text.  
* **Published in:** ICLR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2210.02399) | [[PDF]](https://arxiv.org/pdf/2210.02399.pdf)

> **Core Innovation**  
> Proposes a video-tokenizer that discretizes variable-length videos into tokens and a conditional masked Transformer that generates these tokens before decoding them into frames. By jointly training on large-scale image-text data plus a small number of video-text pairs, it enables open-domain, variable-duration video generation from text prompts.

<details>
    <summary>Abstract</summary>
    We present Phenaki, a model capable of realistic video synthesis, given a sequence of textual prompts. Generating videos from text is particularly challenging due to the computational cost, limited quantities of high quality text-video data and variable length of videos. To address these issues, we introduce a new model for learning video representation which compresses the video to a small representation of discrete tokens. This tokenizer uses causal attention in time, which allows it to work with variable-length videos. To generate video tokens from text we are using a bidirectional masked transformer conditioned on pre-computed text tokens. The generated video tokens are subsequently de-tokenized to create the actual video. To address data issues, we demonstrate how joint training on a large corpus of image-text pairs as well as a smaller number of video-text examples can result in generalization beyond what is available in the video datasets. Compared to the previous video generation methods, Phenaki can generate arbitrary long videos conditioned on a sequence of prompts (i.e. time variable text or a story) in open domain. To the best of our knowledge, this is the first time a paper studies generating videos from time variable prompts. In addition, compared to the per-frame baselines, the proposed video encoder-decoder computes fewer tokens per video but results in better spatio-temporal consistency.
</details>

<details>
    <summary>Key points</summary>
    * Introduces a discrete tokenization of video representation, enabling variable-length output.  
    * Uses a bidirectional masked transformer conditioned on text tokens to generate video tokens, then de-tokenizes to actual frames.  
    * Leverages joint training on large image-text corpus and smaller video-text datasets to improve generalization.  
    * Capable of generating long videos from sequences of prompts, rather than single static text.  
</details>
</details>

---
