## 🌟 The "Trinity" Prototype: Emerging World Models

This section highlights models that demonstrate **preliminary integration of all three consistencies**, exhibiting emergent world model capabilities. These systems represent the current frontier, showing glimpses of true world simulation.

### Text-to-World Generators

Models that generate dynamic, spatially consistent virtual environments from language descriptions.

**Key Characteristics:**
- ✅ Modality: Natural language understanding and pixel-space generation
- ✅ Spatial: 3D-aware scene composition with object permanence
- ✅ Temporal: Physically plausible dynamics and motion

#### Representative Works

<details>
<summary><b>OpenAI Sora</b></summary>

* **Authors:** OpenAI Research Team
* **Model:** Sora (2024)
* **One-liner:** A large-scale space-time generative model that treats videos as sequences of visual patches, enabling high-fidelity text-to-video, image-to-video, and video editing.
* **Published in:** 2024 (open release)
* **Links:** [[Model Page]](https://openai.com/sora) | [[Technical Overview]](https://openai.com/index/video-generation-models-as-world-simulators/)

> **Core Innovation**  
> Sora introduces a large-scale universal generative model that treats visual data (e.g., videos) as “patches,” supporting multiple input modalities (text, images, videos) and outputting new videos with flexible durations, resolutions, and aspect ratios.

<details>
    <summary>Abstract</summary>
    Sora is a general visual generation model capable of synthesizing videos and images across diverse durations, resolutions, and aspect ratios. It leverages a spatiotemporal representation of visual data and trains on patches, similar to token-based language models. Sora supports flexible conditioning (text, image, video) and demonstrates strong temporal coherence, physical realism, and camera motion control, pointing toward scalable video models as world simulators.
</details>

<details>
    <summary>Key points</summary>
    * Treats video as patch-based latent sequences (analogous to tokenization in LLMs)  
    * Unified model for video & image generation across resolutions and durations  
    * Strong physical consistency and camera control  
    * Supports text → video, image → video, video editing, and stylization  
</details>
</details>

---

<details>
<summary><b>Runway Gen-3 Alpha</b></summary>

* **Authors:** Runway Research Team
* **Model:** Gen-3 Alpha (2024)
* **One-liner:** Next-generation multimodal video foundation model with improved motion realism, fidelity, and controllability over Gen-2.
* **Published in:** 2024 (Alpha)
* **Links:** [[Model Page]](https://runwayml.com/research/introducing-gen-3-alpha)

> **Core Innovation**  
> Gen-3 Alpha is built on a brand-new, large-scale multimodal training infrastructure that accepts text, image, and video inputs, delivering superior motion fidelity, structural consistency, and controllability.

<details>
    <summary>Abstract</summary>
    Gen-3 Alpha is a new multimodal video generation model built on a scalable training stack designed for high-fidelity, controllable visual synthesis. It supports text-to-video, image-to-video, and hybrid creative workflows with strong temporal stability, realistic motion, and cinematic quality. The system introduces new control tools such as Motion Brush and camera path control for creator-centric video production.
</details>

<details>
    <summary>Key points</summary>
    * Text-to-video, image-to-video, and mixed creative inputs  
    * Enhanced motion, consistency, and cinematic realism vs. Gen-2  
    * New control tools (Motion Brush, camera path control)  
    * Designed for professional film & creator workflows  
</details>
</details>

---

<details>
<summary><b>Pika 1.0</b></summary>

* **Authors:** Pika Labs Team
* **Model:** Pika 1.0 (2023)
* **One-liner:** A user-friendly generative video platform supporting text-to-video, image-to-video, and video editing with multiple artistic and cinematic styles.
* **Published in:** 2023 (November)
* **Links:** [[Product Page]](https://pika.art/) | [[Launch Blog]](https://pika.art/blog)

> **Core Innovation**  
> Pika 1.0 democratizes video generation and editing: accessible via Web and Discord, it lets users create videos from text or images and edit existing footage in a wide range of visual styles.

<details>
    <summary>Abstract</summary>
    Pika 1.0 introduces a generative video model embedded in a web-based creation platform. It supports multi-style video synthesis (3D animation, anime, cinematic, cartoon) and allows users to animate images, edit video clips, and generate scenes through natural prompts. With an emphasis on accessibility and fast iteration, Pika aims to democratize AI-powered video creation for everyday creative workflows.
</details>

<details>
    <summary>Key points</summary>
    * Text-to-video, image-to-video, and video editing  
    * Multiple stylistic modes (3D, anime, cinematic, cartoon)  
    * Web UI + Discord integration for accessible creation  
    * Consumer-focused, fast iteration and community-driven growth  
</details>
</details>

---

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
