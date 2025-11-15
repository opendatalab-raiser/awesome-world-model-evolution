<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">中文</a>
</p>

## 📊 Benchmarks and Evaluation

**Current Challenge**: Existing metrics (FID, FVD, CLIP Score) inadequately assess world model capabilities, focusing on perceptual quality rather than physical understanding.

**Need for Comprehensive Benchmarks:**

A true world model benchmark should evaluate:
- 🧩 **Commonsense Physics Understanding**: Does the model respect gravity, momentum, conservation laws?
- 🔮 **Counterfactual Reasoning**: Can it predict outcomes of hypothetical interventions?
- ⏳ **Long-term Consistency**: Does coherence break down over extended simulation horizons?
- 🎯 **Goal-Directed Planning**: Can it chain actions to achieve complex objectives?
- 🎛️ **Controllability**: How precisely can users manipulate simulated elements?

#### Representative Works

<details>
<summary><b>WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation</b></summary>

* **Authors:** Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, Li Yuan
* **arXiv ID:** 2503.07265
* **One-liner:** Introduces WISE, a world-knowledge-aware evaluation framework that measures semantic alignment and common-sense correctness in text-to-image generation.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2503.07265) | [[PDF]](https://arxiv.org/pdf/2503.07265.pdf) | [[Code]](https://github.com/PKU-YuanGroup/WISE)

> **Core Innovation**  
> WISE argues that “evaluation must incorporate world knowledge” and constructs a text-to-image consistency assessment framework grounded in real-world semantic constraints, conceptual relations, and commonsense reasoning. This approach significantly improves measurement of text–image alignment and comprehension beyond mere CLIP similarity.

<details>
    <summary>Abstract</summary>
    Text-to-Image (T2I) models are capable of generating high-quality artistic creations and visual content. However, existing research and evaluation standards predominantly focus on image realism and shallow text-image alignment, lacking a comprehensive assessment of complex semantic understanding and world knowledge integration in text to image generation. To address this challenge, we propose WISE, the first benchmark specifically designed for World Knowledge-Informed Semantic Evaluation. WISE moves beyond simple word-pixel mapping by challenging models with 1000 meticulously crafted prompts across 25 sub-domains in cultural common sense, spatio-temporal reasoning, and natural science. To overcome the limitations of traditional CLIP metric, we introduce WIScore, a novel quantitative metric for assessing knowledge-image alignment. Through comprehensive testing of 20 models (10 dedicated T2I models and 10 unified multimodal models) using 1,000 structured prompts spanning 25 subdomains, our findings reveal significant limitations in their ability to effectively integrate and apply world knowledge during image generation, highlighting critical pathways for enhancing knowledge incorporation and application in next-generation T2I models. Code and data are available at this https URL.
</details>

<details>
    <summary>Key points</summary>
    * Evaluates world knowledge and common-sense reasoning in T2I models  
    * Measures entity correctness, relational consistency, attributes, and scene realism  
    * Shows stronger correlation with human preference vs CLIPScore/BERTScore  
    * Designed as a standardized benchmark for semantic T2I evaluation  
</details>
</details>

---

<details>
<summary><b>Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark</b></summary>

* **Authors:** Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, Pheng-Ann Heng
* **arXiv ID:** 2510.26802
* **One-liner:** Proposes MME-COF benchmark to evaluate zero-shot video reasoning ability across memory, motion, events, causality, and forecasting — showing that current video models lag behind VLMs.
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2510.26802) | [[PDF]](https://arxiv.org/pdf/2510.26802.pdf) | [[Code]](https://github.com/ZiyuGuo99/MME-CoF)

> **Core Innovation**  
> For the first time systematically evaluates “zero-shot reasoning” capability of large video models, introduces the MME-CoF benchmark (Memory, Motion, Event, Causality, Object, Future prediction), and reveals that video models lag markedly behind vision-language LLMs in fine-grained causal understanding.

<details>
    <summary>Abstract</summary>
    Recent video generation models can produce high-fidelity, temporally coherent videos, indicating that they may encode substantial world knowledge. Beyond realistic synthesis, they also exhibit emerging behaviors indicative of visual perception, modeling, and manipulation. Yet, an important question still remains: Are video models ready to serve as zero-shot reasoners in challenging visual reasoning scenarios? In this work, we conduct an empirical study to comprehensively investigate this question, focusing on the leading and popular Veo-3. We evaluate its reasoning behavior across 12 dimensions, including spatial, geometric, physical, temporal, and embodied logic, systematically characterizing both its strengths and failure modes. To standardize this study, we curate the evaluation data into MME-CoF, a compact benchmark that enables in-depth and thorough assessment of Chain-of-Frame (CoF) reasoning. Our findings reveal that while current video models demonstrate promising reasoning patterns on short-horizon spatial coherence, fine-grained grounding, and locally consistent dynamics, they remain limited in long-horizon causal reasoning, strict geometric constraints, and abstract logic. Overall, they are not yet reliable as standalone zero-shot reasoners, but exhibit encouraging signs as complementary visual engines alongside dedicated reasoning models.
</details>

<details>
    <summary>Key points</summary>
    * MME-COF benchmark: Memory, Motion, Event, Causality, Object, Future prediction  
    * Zero-shot evaluation — no task-specific video fine-tuning  
    * Reveals video models are weaker in causal & common-sense reasoning  
    * Shows video models rely on pattern recognition rather than deep inference  
</details>
</details>

---
