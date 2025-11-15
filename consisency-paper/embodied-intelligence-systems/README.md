<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### Embodied Intelligence Systems

Models designed for robotic control and autonomous agents that must integrate perception, spatial reasoning, and temporal prediction for real-world task execution.

**Key Characteristics:**
- Multimodal instruction following
- 3D spatial navigation and manipulation planning
- Predictive modeling of action consequences

#### Representative Works

<details>
<summary><b>RT-2: Vision-Language-Action Models</b></summary>

* **Authors:** Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, Brianna Zitkovich
* **arXiv ID:** 2307.15818
* **One-liner:** Trains a VLM on web-scale internet data + robot trajectories to map text + images directly to robotic actions for general-purpose real-world manipulation.
* **Published in:** CoRL 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2307.15818) | [[PDF]](https://arxiv.org/pdf/2307.15818.pdf) | [[Project Page]](https://robotics-transformer2.github.io/)

> **Core Innovation**  
> Extends large models from “see and talk” to “see, understand, and act,” pioneering the transfer of internet-scale vision-language knowledge to real-world robotic control, endowing robots with open-world reasoning and zero-shot manipulation capabilities.

<details>
    <summary>Abstract</summary>
    We study how vision-language models trained on Internet-scale data can be incorporated directly into end-to-end robotic control to boost generalization and enable emergent semantic reasoning. Our goal is to enable a single end-to-end trained model to both learn to map robot observations to actions and enjoy the benefits of large-scale pretraining on language and vision-language data from the web. To this end, we propose to co-fine-tune state-of-the-art vision-language models on both robotic trajectory data and Internet-scale vision-language tasks, such as visual question answering. In contrast to other approaches, we propose a simple, general recipe to achieve this goal: in order to fit both natural language responses and robotic actions into the same format, we express the actions as text tokens and incorporate them directly into the training set of the model in the same way as natural language tokens. We refer to such category of models as vision-language-action models (VLA) and instantiate an example of such a model, which we call RT-2. Our extensive evaluation (6k evaluation trials) shows that our approach leads to performant robotic policies and enables RT-2 to obtain a range of emergent capabilities from Internet-scale training. This includes significantly improved generalization to novel objects, the ability to interpret commands not present in the robot training data (such as placing an object onto a particular number or icon), and the ability to perform rudimentary reasoning in response to user commands (such as picking up the smallest or largest object, or the one closest to another object). We further show that incorporating chain of thought reasoning allows RT-2 to perform multi-stage semantic reasoning, for example figuring out which object to pick up for use as an improvised hammer (a rock), or which type of drink is best suited for someone who is tired (an energy drink).
</details>

<details>
    <summary>Key points</summary>
    * Extends LLM/VLM foundation modeling to robot control
    * Trained on *internet vision-language data + robot action data*
    * Outputs robotic action tokens
    * Strong zero-shot generalization to new objects, tasks, environment changes
</details>
</details>

---

<details>
<summary><b>GAIA-1: A Generative World Model for Autonomous Driving</b></summary>

* **Authors:** Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado
* **arXiv ID:** 2311.07541
* **One-liner:** A large-scale generative world model that simulates diverse driving scenarios and predicts future multi-agent behavior for closed-loop autonomous driving.
* **Published in:** arXiv 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2309.17080) | [[PDF]](https://arxiv.org/pdf/2309.17080.pdf)

> **Core Innovation**  
> Beyond trajectory forecasting, it learns “world evolution” itself—simultaneously generating multi-agent dynamics, realistic traffic behaviors, and authentic sensor signals for closed-loop simulation and synthesis of rare safety-critical scenarios.

<details>
    <summary>Abstract</summary>
    Autonomous driving promises transformative improvements to transportation, but building systems capable of safely navigating the unstructured complexity of real-world scenarios remains challenging. A critical problem lies in effectively predicting the various potential outcomes that may emerge in response to the vehicle's actions as the world evolves.
    To address this challenge, we introduce GAIA-1 ('Generative AI for Autonomy'), a generative world model that leverages video, text, and action inputs to generate realistic driving scenarios while offering fine-grained control over ego-vehicle behavior and scene features. Our approach casts world modeling as an unsupervised sequence modeling problem by mapping the inputs to discrete tokens, and predicting the next token in the sequence. Emerging properties from our model include learning high-level structures and scene dynamics, contextual awareness, generalization, and understanding of geometry. The power of GAIA-1's learned representation that captures expectations of future events, combined with its ability to generate realistic samples, provides new possibilities for innovation in the field of autonomy, enabling enhanced and accelerated training of autonomous driving technology.
</details>

<details>
    <summary>Key points</summary>
    * Generative world model for driving — not just trajectory prediction
    * Generates future agent motion, scene geometry, and sensor data
    * Closed-loop simulation for AD evaluation + training
    * Enables rare / edge-case generation for safety
</details>
</details>

---

<details>
<summary><b>PaLM-E: An Embodied Multimodal Language Model</b></summary>

* **Authors:** Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence
* **arXiv ID:** 2303.03378
* **One-liner:** Multimodal LLM that integrates real-world sensor inputs (vision, robotics state) into PaLM, enabling robotic reasoning and action planning.
* **Published in:** ICLR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2303.03378) | [[PDF]](https://arxiv.org/pdf/2303.03378.pdf) | [[Project Page]](https://palm-e.github.io/)

> **Core Innovation**  
> Treats robot vision and state information as “tokens” fed into an LLM, turning the language model into an embodied-intelligence decision system that can understand the environment, plan actions, and execute tasks.

<details>
    <summary>Abstract</summary>
    Large language models excel at a wide range of complex tasks. However, enabling general inference in the real world, e.g., for robotics problems, raises the challenge of grounding. We propose embodied language models to directly incorporate real-world continuous sensor modalities into language models and thereby establish the link between words and percepts. Input to our embodied language model are multi-modal sentences that interleave visual, continuous state estimation, and textual input encodings. We train these encodings end-to-end, in conjunction with a pre-trained large language model, for multiple embodied tasks including sequential robotic manipulation planning, visual question answering, and captioning. Our evaluations show that PaLM-E, a single large embodied multimodal model, can address a variety of embodied reasoning tasks, from a variety of observation modalities, on multiple embodiments, and further, exhibits positive transfer: the model benefits from diverse joint training across internet-scale language, vision, and visual-language domains. Our largest model, PaLM-E-562B with 562B parameters, in addition to being trained on robotics tasks, is a visual-language generalist with state-of-the-art performance on OK-VQA, and retains generalist language capabilities with increasing scale.
</details>

<details>
    <summary>Key points</summary>
    * Embodied LLM combining robot sensor tokens + language
    * Joint vision-robot-language pretraining improves generalization
    * Handles long-horizon task reasoning and grounding
    * Builds toward unified *robotics-LLM world models*
</details>
</details>

---
