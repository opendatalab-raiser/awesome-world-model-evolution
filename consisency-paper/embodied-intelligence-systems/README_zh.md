<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### 具身智能系统

为机器人控制和自主智能体设计的模型，必须整合感知、空间推理和时间预测以执行真实世界任务。

**关键特征：**
- 多模态指令遵循
- 3D 空间导航和操作规划
- 动作后果的预测建模

#### 代表性工作

<details>
<summary><b>RT-2: Vision-Language-Action Models</b></summary>

* **Authors:** Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, Brianna Zitkovich
* **arXiv ID:** 2307.15818
* **One-liner:** 将大规模网络 VLM 与机器人动作训练结合，将视觉与语言直接转化为真实机器人动作，实现零样本现实操控。
* **Published in:** CoRL 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.15818) | [[PDF]](https://arxiv.org/pdf/2307.15818.pdf) | [[Project Page]](https://robotics-transformer2.github.io/)

> **核心创新**  
> 将大模型从“看图 + 说话”扩展为“看图 + 理解 + 动作执行”，首次将互联网视觉-语言知识迁移到真实机器人动作层面，使机器人具备开放世界推理和零样本动作能力。

<details>
    <summary>Abstract</summary>
    我们研究如何将互联网级视觉-语言模型直接融入端到端机器人控制，以提升泛化能力并激发语义推理。目标是让单一端到端模型既能将观测映射为动作，又能享用网络图文预训练成果。为此，我们在机器人轨迹与互联网级视觉-语言任务（如视觉问答）上联合微调前沿视觉-语言模型。不同于其他方法，我们给出简洁通用方案：把动作表示为文本token，与语言token同等对待并纳入训练集。这类模型称为视觉-语言-动作模型（VLA），我们实例化出RT-2。6千次评估表明，该方法催生高性能策略，使RT-2从互联网训练中获得涌现能力：显著泛化到新物体，理解训练数据未见的指令（如把物体放到特定数字或图标上），并执行基础推理（如取最小/最大或离某物最近的物体）。引入思维链后，RT-2可进行多阶段语义推理，例如判断用哪块石头当临时锤子，或给疲惫的人挑能量饮料。
</details>

<details>
    <summary>Key points</summary>
    * 将 VLM 扩展到机器人动作空间  
    * 互联网数据 + 机器人数据联合训练  
    * 输出离散化机器人动作 token 序列  
    * 强零样本操控能力：可执行未示教任务/对象/指令  
</details>
</details>

---

<details>
<summary><b>GAIA-1: A Generative World Model for Autonomous Driving</b></summary>

* **Authors:** Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado
* **arXiv ID:** 2311.07541
* **One-liner:** 面向自动驾驶的生成式世界模型，实现多智能体运动预测与闭环模拟训练。
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2309.17080) | [[PDF]](https://arxiv.org/pdf/2309.17080.pdf)

> **核心创新**  
> 不止是轨迹预测，而是学习“世界演化”——可生成多智能体动态、真实交通行为与传感器信号，用于闭环仿真与安全稀有场景合成。

<details>
    <summary>Abstract</summary>
    自动驾驶有望带来交通领域的变革性改善，但构建能在真实世界复杂无序场景中安全导航的系统仍面临挑战。关键难题在于，如何有效预测随车辆动作演进可能出现的多种潜在结果。为此，我们提出GAIA-1（Generative AI for Autonomy），一个生成式世界模型，它利用视频、文本与动作输入生成逼真驾驶场景，并可精细控制自车行为与场景特征。该方法将世界建模转化为无监督序列建模：把输入映射为离散token，再预测序列中的下一token。模型涌现的能力包括学习高层结构与场景动力学、上下文感知、泛化及几何理解。GAIA-1对未来事件的期望表征与逼真采样能力相结合，为自动驾驶领域开辟新可能，加速并提升自动驾驶技术的训练水平。
</details>

<details>
    <summary>Key points</summary>
    * 生成式世界模型：不仅预测轨迹，还模拟整个交通生态  
    * 支持多智能体时空建模（车-人-交通灯-环境）  
    * 用于闭环 AD 仿真、鲁棒性测试和长尾案例生成  
    * 提高数据效率与泛化能力的同时提升安全性  
</details>
</details>

---

<details>
<summary><b>PaLM-E: An Embodied Multimodal Language Model</b></summary>

* **Authors:** Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence
* **arXiv ID:** 2303.03378
* **One-liner:** 将真实世界传感输入嵌入 PaLM 大模型，实现具身智能中的视觉-语言-机器人统一推理与任务执行。
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2303.03378) | [[PDF]](https://arxiv.org/pdf/2303.03378.pdf) | [[Project Page]](https://palm-e.github.io/)

> **核心创新**  
> 把机器人视觉、状态信息作为“token”输入 LLM，使语言模型成为具身智能决策系统，能理解环境、规划动作并执行任务。

<details>
    <summary>Abstract</summary>
    大语言模型在诸多复杂任务中表现卓越，但在现实世界（如机器人问题）中进行通用推理时，面临 grounding 挑战。我们提出具身语言模型，将连续的实境传感器模态直接融入语言模型，从而建立词汇与感知之间的连接。模型输入为多模态句子，交织视觉、连续状态估计与文本编码。我们与预训练大语言模型端到端联合训练这些编码，用于序列机器人操作规划、视觉问答和描述等具身任务。评估显示，单一大规模具身多模态模型 PaLM-E 可处理多种具身推理任务，适配不同观测模态与 embodiment，并表现出正向迁移：模型受益于互联网级语言、视觉及视觉-语言联合训练。最大的 PaLM-E-562B 含 5620 亿参数，除机器人任务外，还是视觉-语言通才，在 OK-VQA 上达最佳水平，并随规模保持通用语言能力。
</details>

<details>
    <summary>Key points</summary>
    * 将机器人传感信息 token 化输入 LLM  
    * 视觉-语言-动作统一模型  
    * 具备长期规划、任务分解与泛化能力  
    * 推动“机器人 = LLM + 具身输入”研究范式  
</details>
</details>

---
