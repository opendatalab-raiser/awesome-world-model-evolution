<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

## 📊 基准测试与评估

**当前挑战**：现有指标（FID、FVD、CLIP Score）无法充分评估世界模型能力，侧重于感知质量而非物理理解。

**综合基准测试的需求：**

真正的世界模型基准应评估：
- 🧩 **常识物理理解**：模型是否遵守重力、动量、守恒定律？
- 🔮 **反事实推理**：能否预测假设干预的结果？
- ⏳ **长期一致性**：在扩展的模拟时间范围内连贯性是否会崩溃？
- 🎯 **目标导向规划**：能否链接动作以实现复杂目标？
- 🎛️ **可控性**：用户能多精确地操控模拟元素？

#### 代表性工作

<details>
<summary><b>WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation</b></summary>

* **Authors:** Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, Li Yuan
* **arXiv ID:** 2503.07265
* **One-liner:** 提出可感知世界知识的语义一致性评价指标（WISE），用以衡量文生图模型是否真正理解常识与世界事实。
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2503.07265) | [[PDF]](https://arxiv.org/pdf/2503.07265.pdf) | [[Code]](https://github.com/PKU-YuanGroup/WISE)

> **核心创新**  
> WISE 强调“评价应包含世界知识”，构建了一个包含真实世界语义约束、概念关系和常识推理的文生图一致性评估体系，能够显著更准确地衡量文本-图像对齐与理解能力，而非只依赖 CLIP 相似度。

<details>
    <summary>Abstract</summary>
    文本到图像模型能生成高质量的艺术作品与视觉内容，然而现有研究与评估标准主要关注图像逼真度和浅层图文对齐，缺乏对复杂语义理解与世界知识融入的综合衡量。为此，我们提出首个面向世界知识引导语义评估的基准 WISE，超越简单词-像素映射，用涵盖文化常识、时空推理、自然科学 25 子域的 1000 条精心构造提示挑战模型。针对传统 CLIP 指标局限，我们提出新的量化度量 WIScore 评估知识-图像对齐度。通过对 20 个模型（10 个专用 T2I 与 10 个统一多模态）在 1000 条结构化提示上的全面测试，发现其生成图像时整合并应用世界知识的能力显著不足，为下一代模型增强知识融入指明关键路径。代码与数据见 https 链接。
</details>

<details>
    <summary>Key points</summary>
    * 引入“世界知识 + 常识推理”评估框架  
    * 评价维度：实体正确性、关系合理性、属性一致性、场景常识  
    * 与人工偏好高度相关，优于 CLIPScore/BERTScore 等指标  
    * 可用作未来文生图模型 benchmark 的通用评价层  
</details>
</details>

---

<details>
<summary><b>Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark</b></summary>

* **Authors:** Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, Pheng-Ann Heng
* **arXiv ID:** 2510.26802
* **One-liner:** 构建 MME-COF 基准测试多模态视频模型的零样本常识与因果推理能力，发现当前视频模型仍缺乏真正时序-因果理解。
* **Published in:** arXiv 2025
* **Links:** [[Paper]](https://arxiv.org/abs/2510.26802) | [[PDF]](https://arxiv.org/pdf/2510.26802.pdf) | [[Code]](https://github.com/ZiyuGuo99/MME-CoF)

> **核心创新**  
> 首次系统评估视频大模型的“零样本推理”能力，提出 MME-COF 基准（Memory, Motion, Event, Causality, Object, Future prediction），发现视频模型在细粒度因果理解方面显著落后于视觉-语言大模型。

<details>
    <summary>Abstract</summary>
    最新视频生成模型可产出高保真、时序连贯的视频，暗示其编码了丰富的世界知识。除逼真合成外，它们还表现出视觉感知、建模与操控的涌现行为。然而，一个重要问题仍未解答：在挑战性视觉推理场景中，视频模型是否已具备零样本推理能力？为此，我们对领先且流行的 Veo-3 开展实证研究，从空间、几何、物理、时间、具身逻辑等 12 个维度系统评估其推理表现，刻画优势与失效模式。为标准化评估，我们构建紧凑基准 MME-CoF，深入测评逐帧链式（CoF）推理。结果显示，当前视频模型在短程空间连贯、细粒度定位与局部一致动态上展现可喜模式，但在长程因果、严格几何与抽象逻辑上仍受限。总体而言，它们尚不足以作为独立零样本推理器，但已显露作为专用推理模型补充视觉引擎的潜力。
</details>

<details>
    <summary>Key points</summary>
    * MME-COF 六大测试维度：对象识别、记忆、动作、事件、因果、未来预测  
    * 面向“零样本”评估，无视频任务微调  
    * 视频模型 vs 文本-图像模型：前者推理能力不足  
    * 揭示了当前视频模型更偏“模式拟合”而非“语义推理”  
</details>
</details>

---
