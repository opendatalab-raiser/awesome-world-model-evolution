<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

## 🌟 "三位一体"原型：涌现的世界模型

本节重点介绍展示**三大一致性初步整合**的模型，表现出涌现的世界模型能力。这些系统代表了当前前沿，展现了真正世界模拟的雏形。

### 文本到世界生成器

从语言描述生成动态、空间一致的虚拟环境的模型。

**关键特征：**
- ✅ 模态一致性：自然语言理解和像素空间生成
- ✅ 空间一致性：具有物体恒存性的 3D 感知场景组合
- ✅ 时间一致性：物理上可信的动力学和运动

#### 代表性工作

<details>
<summary><b>OpenAI Sora</b></summary>

* **Authors:** OpenAI (团队)  
* **Model ID:** Sora (2024)  
* **One-liner:** 支持从文本、图像、视频输入生成新视频的通用视觉数据模型。 
* **Published in:** 2024（开放形式）
* **Links:** [[Model Page]](https://openai.com/sora/) | [[Technical Overview]](https://openai.com/index/video-generation-models-as-world-simulators/)

> **核心创新**  
> Sora 引入了将视觉数据（如视频）视作“补丁（patches）”训练的大规模通用生成模型，支持多种输入模态（例如文本、图像、视频）并输出新视频，时长、分辨率、长短多样。

<details>
    <summary>Abstract</summary>
    Sora 是一种通用视觉数据生成模型——能够生成视频与图像，跨越不同持续时间、纵横比与分辨率。我们借鉴大型语言模型中 token 化方式，将视觉数据转换为补丁表示，并训练一个压缩编码网络以把原始视频降维，再训练对应解码器将生成的 latent 映射回像素空间。
</details>

<details>
    <summary>Key points</summary>
    * 使用补丁（patch）作为视觉数据单位，构建可扩展的生成模型。   
    * 引入视频压缩网络，将原始视频编码为时空 latent 表示。
    * 支持灵活的输入模态（文本、图像、视频）与多样的视频输出格式。  
    * 强调通用性与规模化：目标是构建类似“世界模型”的视觉生成系统。  
</details>
</details>

---

<details>
<summary><b>Runway Gen-3 Alpha</b></summary>

* **Authors:** Runway AI (团队)  
* **Model ID:** Gen-3 Alpha (2024)  
* **One-liner:** 下一代多模态视频生成基础模型，在忠实度、一致性、运动表现方面较 Gen-2 有大幅提升。
* **Published in:** 2024 (Alpha) 
* **Links:** [[Model Page]](https://runwayml.com/research/introducing-gen-3-alpha)

> **核心创新**  
> Gen-3 Alpha 构建于全新大规模多模态训练基础设施之上，支持文本、图像、视频三种输入，并提供更优秀的运动表现、结构一致性与控制能力。

<details>
    <summary>Abstract</summary>
    该模型是 Runway 在其新一代基础模型上的开端，旨在提升视频生成的视觉忠实度、镜头运动一致性以及创意控制能力。用户可从静态图像或文本出发，生成动态视频、控制镜头运动、人物表现等。
</details>

<details>
    <summary>Key points</summary>
    * 支持文本→视频、图像→视频等多模态输入形式。  
    * 引入“Motion Brush”“高级摄像机控制”等工具，提升创意可控性。 
    * 在生成质量、速度、运动流畅度方面比上代显著改进。  
    * 面向创作者与影视制作流程，强调实用性与制作体验。  
</details>
</details>

---

<details>
<summary><b>Pika 1.0</b></summary>

* **Authors:** Pika Labs (团队)  
* **Model ID:** Pika 1.0 (2023)  
* **One-liner:** “想法到视频”平台中推出的版本 1.0 模型，可生成与编辑多种风格（3D 动画、卡通、电影风格）视频。 
* **Published in:** 2023 (11 月) 
* **Links:** [[Product Page]](https://pika.art/) | [[Launch Blog]](https://pika.art/blog) 

> **核心创新**  
> Pika 1.0 将视频生成编辑工具普及化：通过 Web 与 Discord 接入、支持从文本或图像创建视频及编辑已有视频，覆盖多种视觉风格。 

<details>
    <summary>Abstract</summary>
    Pika Labs 在 2023 年 11 月推出其 1.0 版本，引入了一个新的 AI 模型，能够在多种风格下生成视频（包括 3D 动画、动漫、卡通与电影风格），提供全新的网页使用体验。公司同期宣布融资 5500 万美元。用户可从文本或图像提示出发，部分支持已有视频编辑。
</details>

<details>
    <summary>Key points</summary>
    * 支持文本→视频、图像→视频与视频→视频编辑。
    * 多种视觉风格：3D 动画、动漫、卡通、电影级。  
    * 简化创作流程：Web 界面 + Discord 等平台接入，降低视频生成门槛。  
    * 快速增长用户基础：推出半年内已有数十万用户、每周生成百万级视频。 
</details>
</details>


---
