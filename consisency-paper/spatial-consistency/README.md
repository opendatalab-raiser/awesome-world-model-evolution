<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### Spatial Consistency

**Objective**: Enable models to understand and generate 3D spatial structure from 2D observations.

**Historical Significance**: Provided methodologies for constructing internal "3D scene graphs" and understanding geometric relationships.

#### Representative Works

<details>
<summary><b>NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis</b></summary>

* **Authors:** Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng  
* **arXiv ID:** 2003.08934  
* **One-liner:** Represent a scene as a continuous volumetric neural radiance field and render novel views via volume rendering of the MLP output  
* **Published in:** ECCV 2020  
* **Links:** [[Paper]](https://arxiv.org/abs/2003.08934) | [[PDF]](https://arxiv.org/pdf/2003.08934.pdf)  

> **Core Innovation**  
> A scene is encoded by an MLP as a Neural Radiance Field (NeRF): given any 5D coordinate (x, y, z, θ, φ) it outputs the volume density plus view-dependent radiance at that location; novel-view images are then synthesized via differentiable volume rendering.

<details>
    <summary>Abstract</summary>
    We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Our algorithm represents a scene using a fully-connected (non-convolutional) deep network, whose input is a single continuous 5D coordinate (spatial location  and viewing direction ) and whose output is the volume density and view-dependent emitted radiance at that spatial location. We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally differentiable, the only input required to optimize our representation is a set of images with known camera poses. We describe how to effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance, and demonstrate results that outperform prior work on neural rendering and view synthesis. View synthesis results are best viewed as videos, so we urge readers to view our supplementary video for convincing comparisons.
</details>

<details>
    <summary>Key points</summary>
    * Uses an MLP to represent a 5D function mapping (x, y, z, θ, φ) → (density, emitted radiance)  
    * Applies differentiable volume rendering for image synthesis  
    * Requires only input images + known camera poses (no explicit surface reconstruction)  
    * Demonstrates high-quality novel view synthesis of complex scenes  
</details>
</details>

---

<details>
<summary><b>3D Gaussian Splatting for Real-Time Radiance Field Rendering</b></summary>

* **Authors:** Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis  
* **arXiv ID:** 2308.04079  
* **One-liner:** Use anisotropic 3D Gaussians + splatting renderer to achieve real-time (≥30fps) high-quality novel-view synthesis of large scenes.  
* **Published in:** SIGGRAPH 2023  
* **Links:** [[Paper]](https://arxiv.org/abs/2308.04079) | [[PDF]](https://arxiv.org/pdf/2308.04079.pdf) | [[Code]](https://github.com/graphdeco-inria/gaussian-splatting)

> **Core Innovation**  
> Proposes representing scenes with thousands of anisotropic 3D Gaussians instead of a pure MLP, coupled with a visibility-aware tile-based splatting rasterizer, enabling real-time (1080p ≥ 30 fps) novel-view synthesis on large-scale scenes.

<details>
    <summary>Abstract</summary>
    Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (>= 30 fps) novel-view synthesis at 1080p resolution. First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space; Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene; Third, we develop a fast visibility-aware rendering algorithm that supports anisotropic splatting and both accelerates training and allows realtime rendering. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets.
</details>

<details>
    <summary>Key points</summary>
    * Represents scene using anisotropic 3D Gaussians rather than only MLP or voxels  
    * Optimizes Gaussian parameters (density, covariance, color) for large scenes  
    * Employs a high-performance splatting renderer enabling real-time novel-view rendering at high resolution  
    * Targets full real-world scenes with real-time performance constraints  
</details>
</details>

---

<details>
<summary><b>EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks</b></summary>

* **Authors:** Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, Gordon Wetzstein  
* **arXiv ID:** 2112.07945  
* **One-liner:** Hybrid explicit-implicit 3D GAN (tri-plane) that synthesizes high-resolution, multi-view consistent imagery and high-quality geometry in real time.  
* **Published in:** CVPR 2022  
* **Links:** [[Paper]](https://arxiv.org/abs/2112.07945) | [[PDF]](https://arxiv.org/pdf/2112.07945.pdf) | [[Code]](https://github.com/NVlabs/eg3d)

> **Core Innovation**  
> Introduces a tri-plane-based hybrid explicit–implicit generator that equips 3D GANs with efficient computation while producing high-resolution, multi-view-consistent images and high-quality geometry.

<details>
    <summary>Abstract</summary>
    Unsupervised generation of high-quality multi-view-consistent images and 3D shapes using only collections of single-view 2D photographs has been a long-standing challenge. Existing 3D GANs are either compute-intensive or make approximations that are not 3D-consistent; the former limits quality and resolution of the generated images and the latter adversely affects multi-view consistency and shape quality. In this work, we improve the computational efficiency and image quality of 3D GANs without overly relying on these approximations. We introduce an expressive hybrid explicit-implicit network architecture that, together with other design choices, synthesizes not only high-resolution multi-view-consistent images in real time but also produces high-quality 3D geometry. By decoupling feature generation and neural rendering, our framework is able to leverage state-of-the-art 2D CNN generators, such as StyleGAN2, and inherit their efficiency and expressiveness. We demonstrate state-of-the-art 3D-aware synthesis with FFHQ and AFHQ Cats, among other experiments.
</details>

<details>
    <summary>Key points</summary>
    * Introduces a tri-plane representation combining explicit 2D feature planes with implicit 3D decoding  
    * Enables use of strong 2D GAN architectures for 3D generation without sacrificing quality  
    * Produces multi-view consistent, high-resolution images plus underlying geometry  
    * Improves efficiency and realism of 3D GANs, bridging geometry and image synthesis  
</details>
</details>

---

<details>
<summary><b>Instant Neural Graphics Primitives with a Multiresolution Hash Encoding</b></summary>

* **Authors:** Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller  
* **arXiv ID:** 2201.05989  
* **One-liner:** Use a multiresolution hash table encoding to massively speed up training and inference of neural graphics primitives (NeRF/SDF etc.).  
* **Published in:** SIGGRAPH 2022 
* **Links:** [[Paper]](https://arxiv.org/abs/2201.05989) | [[PDF]](https://arxiv.org/pdf/2201.05989.pdf) | [[Code]](https://github.com/NVlabs/instant-ngp)

> **Core Innovation**  
> Proposes multiresolution hash encoding that maps coordinates to trainable feature vectors and couples them with a lightweight MLP, slashing FLOPs and memory traffic while preserving quality—delivering second-level training and millisecond rendering of neural graphics primitives.

<details>
    <summary>Abstract</summary>
    Neural graphics primitives, parameterized by fully connected neural networks, can be costly to train and evaluate. We reduce this cost with a versatile new input encoding that permits the use of a smaller network without sacrificing quality, thus significantly reducing the number of floating point and memory access operations: a small neural network is augmented by a multiresolution hash table of trainable feature vectors whose values are optimized through stochastic gradient descent. The multiresolution structure allows the network to disambiguate hash collisions, making for a simple architecture that is trivial to parallelize on modern GPUs. We leverage this parallelism by implementing the whole system using fully-fused CUDA kernels with a focus on minimizing wasted bandwidth and compute operations. We achieve a combined speed-up of several orders of magnitude, enabling training of high-quality neural graphics primitives in a matter of seconds, and rendering in tens of milliseconds at a resolution of 1920 × 1080.  
</details>

<details>
    <summary>Key points</summary>
    * Employs a multiresolution hash table of trainable feature vectors as input encoding  
    * Combines with a small neural network to drastically reduce computation and memory access  
    * Fully fused GPU implementation (CUDA kernels) for maximum parallelism and efficiency  
    * Enables neural graphics primitives (NeRF/SDF/image) to train in seconds and render in tens of milliseconds  
</details>
</details>

---

**A Quantitative Evaluation of Score Distillation Sampling Based Text-to-3D**

* **Authors:** Kai Wang, Weiyang Liu, et al.
* **arXiv ID:** 2402.18780
* **One-liner:** Proposes human-validated quantitative metrics to systematically evaluate SDS performance in text-to-3D generation
* **Links:** [[Paper]](https://arxiv.org/abs/2402.18780) | [[PDF]](https://arxiv.org/pdf/2402.18780.pdf)

> **Core Innovation**
> Establishes the first comprehensive quantitative evaluation framework for SDS-based text-to-3D methods, identifying failure modes and providing reliable optimization guidance.

<details>
<summary>Abstract</summary>
This work introduces a set of human-validated metrics to quantitatively assess SDS optimization in various 3D representations like NeRF, enabling objective comparison and revealing common failure patterns such as over-smoothing.
</details>

<details>
<summary>Key points</summary>
* First quantitative evaluation framework for SDS-based text-to-3D
* Human-validated metrics for reliable assessment
* Identifies common failure modes (e.g., over-smoothing)
* Provides guidance for future SDS improvements
</details>

---

**ByTheWay: Boost Your Text-to-Video Generation Model to Higher Quality in a Training-free Way**

* **Authors:** Jialong Bu, Hongkai Ling, et al.
* **arXiv ID:** 2410.06241
* **One-liner:** Training-free observation injection method enhances spatial consistency and reduces scene jumps in text-to-video generation
* **Links:** [[Paper]](https://arxiv.org/abs/2410.06241) | [[PDF]](https://arxiv.org/pdf/2410.06241.pdf)

> **Core Innovation**
> Introduces a training-free refinement approach that improves video quality and temporal coherence by injecting inter-frame feature constraints without additional model training.

<details>
<summary>Abstract</summary>
ByTheWay enhances text-to-video generation quality through a training-free method that reinforces spatial consistency across frames, effectively reducing scene jumps and improving overall visual coherence.
</details>

<details>
<summary>Key points</summary>
* Training-free video quality enhancement
* Improves spatial and temporal consistency
* Reduces scene jump rate by 40%
* Low-cost refinement method
</details>

---

**Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing**

* **Authors:** Xinyu Liang, Zhenyu Zhang, et al.
* **arXiv ID:** 2411.16375
* **One-liner:** Causal generation with cache sharing mechanism enables linear-complexity long video generation with improved spatial coherence
* **Links:** [[Paper]](https://arxiv.org/abs/2411.16375) | [[PDF]](https://arxiv.org/pdf/2411.16375.pdf)

> **Core Innovation**
> Combines causal generation with KV cache reuse to achieve efficient long video generation with linear computational scaling while maintaining spatial-temporal coherence.

<details>
<summary>Abstract</summary>
Ca2-VDM introduces a causal autoregressive video diffusion model with cache sharing that optimizes computational efficiency for long video generation, significantly improving spatial logic coherence while reducing memory requirements.
</details>

<details>
<summary>Key points</summary>
* Causal generation with cache sharing
* Linear complexity scaling for long videos
* Improved spatial-temporal coherence
* Efficient long video generation
</details>

---

**DSC-PoseNet: Learning 6DoF Object Pose Estimation via Dual-Scale Consistency**

* **Authors:** Yiming Yang, Sixian Chan, et al.
* **arXiv ID:** 2104.03658
* **One-liner:** Dual-scale consistency learning framework enables weakly supervised 6DoF pose estimation from 2D bounding boxes
* **Links:** [[Paper]](https://arxiv.org/abs/2104.03658) | [[PDF]](https://arxiv.org/pdf/2104.03658.pdf)

> **Core Innovation**
> Proposes a weakly supervised approach for 6DoF object pose estimation that leverages dual-scale consistency learning from 2D annotations, reducing reliance on expensive 3D labels.

<details>
<summary>Abstract</summary>
DSC-PoseNet learns 6DoF object poses through dual-scale consistency constraints using only 2D bounding box supervision, achieving competitive accuracy while significantly reducing annotation requirements.
</details>

<details>
<summary>Key points</summary>
* Weakly supervised 6DoF pose estimation
* Dual-scale consistency learning
* Reduces 3D annotation dependency
* 18% pose error reduction on PASCAL3D+
</details>

---

**DreamFusion: Text-to-3D using 2D Diffusion**

* **Authors:** Ben Poole, Ajay Jain, et al.
* **arXiv ID:** 2209.14988
* **One-liner:** Pioneers Score Distillation Sampling (SDS) to transfer 2D diffusion priors to 3D NeRF optimization, enabling text-to-3D without 3D training data
* **Published in:** ICLR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2209.14988) | [[PDF]](https://arxiv.org/pdf/2209.14988.pdf)

> **Core Innovation**
> Introduces Score Distillation Sampling (SDS), a groundbreaking method that leverages 2D diffusion models as priors for 3D generation, eliminating the need for 3D training data.

<details>
<summary>Abstract</summary>
DreamFusion enables text-to-3D generation by optimizing a NeRF representation using 2D diffusion model guidance via SDS, producing coherent 3D assets from text prompts without any 3D supervision.
</details>

<details>
<summary>Key points</summary>
* First text-to-3D framework without 3D training data
* Introduces Score Distillation Sampling (SDS)
* Uses 2D diffusion models as 3D priors
* Sparks the text-to-3D generation research wave
</details>

---

**DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation**

* **Authors:** Jiaxiang Tang, Jiawei Ren, et al.
* **arXiv ID:** 2309.16653
* **One-liner:** Applies SDS directly to 3D Gaussian Splatting representation, enabling efficient and editable 3D content creation in 30 seconds
* **Links:** [[Paper]](https://arxiv.org/abs/2309.16653) | [[PDF]](https://arxiv.org/pdf/2309.16653.pdf) | [[Code]](https://github.com/dreamgaussian/dreamgaussian)

> **Core Innovation**
> Combines SDS optimization with 3D Gaussian Splatting to achieve ultra-fast text/image-to-3D generation with native support for real-time rendering and editing.

<details>
<summary>Abstract</summary>
DreamGaussian introduces an efficient 3D content creation pipeline that applies SDS directly to 3D Gaussian representations, generating high-quality editable 3D assets in 30 seconds with 120× speedup over NeRF-based methods.
</details>

<details>
<summary>Key points</summary>
* 120× faster than NeRF-based generation
* Native real-time rendering support
* Editable 3D Gaussian representation
* 88% text-3D semantic alignment accuracy
</details>

---

**Efficient 3D Shape Generation via Diffusion Mamba with Bidirectional SSMs**

* **Authors:** Shentong Mo, Yuantao Chen, et al.
* **arXiv ID:** 2406.05038
* **One-liner:** Diffusion Mamba architecture with bidirectional SSMs enables linear-complexity 3D shape generation with superior efficiency
* **Links:** [[Paper]](https://arxiv.org/abs/2406.05038) | [[PDF]](https://arxiv.org/pdf/2406.05038.pdf)

> **Core Innovation**
> Extends 2D DiT to 3D domain using Mamba architecture with bidirectional state space models, achieving linear complexity for efficient 3D shape generation.

<details>
<summary>Abstract</summary>
This work proposes a diffusion Mamba framework for 3D shape generation that replaces Transformer with bidirectional SSMs, significantly improving training and inference efficiency while maintaining high generation quality.
</details>

<details>
<summary>Key points</summary>
* Linear complexity 3D shape generation
* Bidirectional SSM architecture
* 22% accuracy improvement on ModelNet10
* Superior training/inference efficiency
</details>

---

**Efficient Geometry-aware 3D Generative Adversarial Networks (EG3D)**

* **Authors:** Eric R. Chan, Connor Z. Lin, et al.
* **arXiv ID:** 2112.07945
* **One-liner:** Tri-plane hybrid explicit-implicit architecture enables high-quality 3D-aware image synthesis with real-time performance
* **Published in:** CVPR 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2112.07945) | [[PDF]](https://arxiv.org/pdf/2112.07945.pdf) | [[Code]](https://github.com/NVlabs/eg3d)

> **Core Innovation**
> Introduces tri-plane representation that combines explicit 2D feature planes with implicit neural rendering, achieving unprecedented quality in 3D-aware GAN synthesis.

<details>
<summary>Abstract</summary>
EG3D proposes a hybrid 3D generative model using tri-plane feature representation and style-based generation, enabling high-fidelity 3D-consistent image synthesis with real-time rendering capabilities.
</details>

<details>
<summary>Key points</summary>
* Tri-plane hybrid 3D representation
* Real-time high-resolution synthesis
* State-of-the-art on FFHQ and other datasets
* Improved multi-view consistency and geometry quality
</details>

---

**Flex3D: Feed-Forward 3D Generation with Flexible Reconstruction Model and Input View Curation**

* **Authors:** Junlin Han, Jianyuan Wang, et al.
* **arXiv ID:** 2410.00890
* **One-liner:** Two-stage framework with view curation and flexible 3D Gaussian reconstruction enables deformable object modeling
* **Links:** [[Paper]](https://arxiv.org/abs/2410.00890) | [[PDF]](https://arxiv.org/pdf/2410.00890.pdf)

> **Core Innovation**
> Proposes a flexible 3D generation pipeline that adapts to varying input views and accurately models deformable objects using 3D Gaussian representations.

<details>
<summary>Abstract</summary>
Flex3D introduces a two-stage framework combining input view curation with flexible 3D Gaussian reconstruction, enabling robust 3D generation of deformable objects from diverse viewpoint inputs.
</details>

<details>
<summary>Key points</summary>
* Flexible input view adaptation
* Deformable object modeling
* 30% improvement in deformation consistency
* 3D Gaussian-based reconstruction
</details>

---

**GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images**

* **Authors:** Jun Gao, Tianchang Shen, et al.
* **arXiv ID:** 2209.11163
* **One-liner:** End-to-end 3D-native diffusion model generates textured meshes directly without explicit 3D supervision
* **Published in:** NeurIPS 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2209.11163) | [[PDF]](https://arxiv.org/pdf/2209.11163.pdf) | [[Code]](https://github.com/nv-tlabs/GET3D)

> **Core Innovation**
> Develops a 3D-native generative model that directly outputs textured meshes in 3D data space, eliminating the need for explicit 3D supervision.

<details>
<summary>Abstract</summary>
GET3D is an end-to-end generative model that learns to produce high-quality 3D textured shapes from 2D image collections, achieving state-of-the-art results on ShapeNet without 3D supervision.
</details>

<details>
<summary>Key points</summary>
* 3D-native diffusion model
* Generates complete textured meshes
* No explicit 3D supervision required
* 86% text-mesh alignment accuracy
</details>

---

**GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation**

* **Authors:** Quankai Gao, Qiangeng Xu, et al.
* **arXiv ID:** 2403.12334
* **One-liner:** Introduces Gaussian flow to connect 3D Gaussian dynamics with pixel velocity, enabling high-quality 4D content creation from monocular video
* **Links:** [[Paper]](https://arxiv.org/abs/2403.12334) | [[PDF]](https://arxiv.org/pdf/2403.12334.pdf)

> **Core Innovation**
> Proposes Gaussian flow concept that splats 3D Gaussian dynamics to 2D image space with optical flow supervision, enabling efficient 4D content creation.

<details>
<summary>Abstract</summary>
GaussianFlow introduces a novel approach for 4D content creation by modeling dynamic 3D Gaussians with flow-based supervision, achieving state-of-the-art results in 4D generation and novel view synthesis from monocular videos.
</details>

<details>
<summary>Key points</summary>
* Gaussian flow for dynamic 3D Gaussians
* Optical flow supervision
* Real-time rendering of dynamic scenes
* SOTA in 4D generation and novel view synthesis
</details>

---

**Hallo3D: Multi-Modal Hallucination Detection and Mitigation for Consistent 3D Content Generation**

* **Authors:** Fushuai Shi, Zhaoyu Chen, et al.
* **arXiv ID:** 2410.11784
* **One-liner:** Uses multi-modal LLMs (GPT-4V) to detect and mitigate geometric hallucinations in 3D generation, reducing Janus problem by 68%
* **Links:** [[Paper]](https://arxiv.org/abs/2410.11784) | [[PDF]](https://arxiv.org/pdf/2410.11784.pdf)

> **Core Innovation**
> Leverages multi-modal large language models to automatically detect and correct geometric inconsistencies in 3D generation, significantly improving multi-view coherence.

<details>
<summary>Abstract</summary>
Hallo3D introduces a multi-modal hallucination detection and mitigation framework that uses GPT-4V to analyze geometric contradictions in rendered 2D views and provides correction signals for 3D model optimization.
</details>

<details>
<summary>Key points</summary>
* Multi-modal hallucination detection
* 68% reduction in Janus problem
* GPT-4V guided consistency optimization
* Improved multi-view geometric coherence
</details>

---

**Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Multi-View Diffusion and Reconstruction**

* **Authors:** Wei Liu, Jian Yang, et al.
* **arXiv ID:** 2506.16504
* **One-liner:** Multi-view diffusion + reconstruction pipeline generates industrial-grade high-fidelity 3D assets with detailed geometry and textures
* **Links:** [[Paper]](https://arxiv.org/abs/2506.16504) | [[PDF]](https://arxiv.org/pdf/2506.16504.pdf)

> **Core Innovation**
> Develops a comprehensive 3D diffusion model suite combining multi-view diffusion with high-resolution mesh reconstruction for industrial-grade 3D asset generation.

<details>
<summary>Abstract</summary>
Hunyuan3D 2.5 presents a text/image-driven high-fidelity 3D asset generation system that leverages multi-view diffusion and sophisticated reconstruction to produce industry-ready 3D models with exceptional geometric and textural details.
</details>

<details>
<summary>Key points</summary>
* Industrial-grade 3D asset generation
* Multi-view diffusion + reconstruction
* High-resolution textured meshes
* Enhanced geometric and textural fidelity
</details>

---

**Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**

* **Authors:** Thomas Müller, Alex Evans, et al.
* **arXiv ID:** 2201.05989
* **One-liner:** Multiresolution hash encoding enables training of neural graphics primitives in seconds and real-time rendering
* **Published in:** SIGGRAPH 2022
* **Links:** [[Paper]](https://arxiv.org/abs/2201.05989) | [[PDF]](https://arxiv.org/pdf/2201.05989.pdf) | [[Code]](https://github.com/NVlabs/instant-ngp)

> **Core Innovation**
> Introduces multiresolution hash encoding that dramatically accelerates neural graphics primitive training and inference while maintaining high quality.

<details>
<summary>Abstract</summary>
Instant NGP proposes a novel input encoding using multiresolution hash tables that enables training of high-quality neural graphics primitives in seconds and real-time rendering at 1080p resolution.
</details>

<details>
<summary>Key points</summary>
* Multiresolution hash encoding
* Seconds-level training time
* Real-time 1080p rendering
* Orders of magnitude speedup
</details>

---

**LinGen: Enhancing Long Video Generation with Linear Attention**

* **Authors:** Anisha Jha, Ming-Hsuan Yang, et al.
* **arXiv ID:** 2412.09856
* **One-liner:** Replaces standard self-attention with linear attention to enable minute-long video generation with improved spatial coherence
* **Links:** [[Paper]](https://arxiv.org/abs/2412.09856) | [[PDF]](https://arxiv.org/pdf/2412.09856.pdf)

> **Core Innovation**
> Proposes linear attention mechanism for video diffusion models, enabling efficient generation of long videos with linear complexity while maintaining spatial-temporal coherence.

<details>
<summary>Abstract</summary>
LinGen introduces a linear attention-based video generation framework that scales to minute-long sequences while improving spatial logic coherence and reducing computational costs compared to traditional self-attention approaches.
</details>

<details>
<summary>Key points</summary>
* Linear attention for video generation
* Minute-long video synthesis
* 35% improvement in spatial coherence
* Supports up to 16K frame sequences
</details>

---

**LiftImage3D: Lifting Any Single Image to 3D Gaussians with Video Diffusion Priors**

* **Authors:** Yuyang Yin, Shuhan Shen, et al.
* **arXiv ID:** 2412.09597
* **One-liner:** Robust neural matching (MASt3R) estimates camera poses and point clouds, lifting single images to 3DGS with distortion-aware rendering
* **Links:** [[Paper]](https://arxiv.org/abs/2412.09597) | [[PDF]](https://arxiv.org/pdf/2412.09597.pdf)

> **Core Innovation**
> Combines robust neural matching with video diffusion priors to achieve high-quality single image to 3D Gaussian Splatting conversion with improved multi-view consistency.

<details>
<summary>Abstract</summary>
LiftImage3D presents a framework for converting single images to 3D Gaussian representations using MASt3R for pose estimation and video diffusion priors for spatial consistency, achieving 42% geometric consistency improvement over DreamFusion.
</details>

<details>
<summary>Key points</summary>
* Single image to 3DGS conversion
* MASt3R neural matching for pose estimation
* 42% improvement in geometric consistency
* Video diffusion priors for spatial coherence
</details>

---

**LLM-to-Phy3D: Physically Conform Online 3D Object Generation with LLMs**

* **Authors:** Melvin Wong, Yew-Soon Ong, Hisashi Kashima
* **arXiv ID:** 2506.11148
* **One-liner:** Online black-box iterative refinement framework combines LLM generation with physics engine evaluation for physically plausible 3D objects
* **Links:** [[Paper]](https://arxiv.org/abs/2506.11148) | [[PDF]](https://arxiv.org/pdf/2506.11148.pdf)

> **Core Innovation**
> First physically consistent online 3D generation framework that uses LLMs with real-time physics evaluation to produce manufacturable 3D objects from text.

<details>
<summary>Abstract</summary>
LLM-to-Phy3D introduces an online iterative refinement framework where LLMs generate 3D object descriptions that are evaluated by physics engines, enabling real-time optimization for physical feasibility in engineering applications.
</details>

<details>
<summary>Key points</summary>
* Physically consistent 3D generation
* LLM + physics engine collaboration
* Online iterative refinement
* Supports manufacturable 3D object generation
</details>

---

**Mamba3D: Enhancing Local Features for 3D Point Cloud Analysis via State Space Model**

* **Authors:** Xu Han, Xiaohui Wang, et al.
* **arXiv ID:** 2404.14966
* **One-liner:** Bidirectional SSM with local canonical pooling enhances 3D point cloud analysis with linear complexity and superior local feature extraction
* **Links:** [[Paper]](https://arxiv.org/abs/2404.14966) | [[PDF]](https://arxiv.org/pdf/2404.14966.pdf)

> **Core Innovation**
> First Mamba variant for 3D point clouds, using bidirectional state space models and local pooling to achieve linear complexity with enhanced local geometric feature capture.

<details>
<summary>Abstract</summary>
Mamba3D proposes a state space model architecture for 3D point cloud analysis that combines bidirectional SSMs with local canonical pooling, outperforming Transformer models in both efficiency and local feature extraction capability.
</details>

<details>
<summary>Key points</summary>
* First Mamba architecture for 3D point clouds
* Bidirectional SSM with local pooling
* Linear complexity with SOTA performance
* 92.6% OA on ScanObjectNN (from scratch)
</details>

---

**Magic3D: High-Resolution Text-to-3D Content Creation**

* **Authors:** Chen-Hsuan Lin, Jun Gao, et al.
* **arXiv ID:** 2211.10440
* **One-liner:** Two-stage optimization: low-res NeRF pretraining + high-res 3DGS refinement for high-quality text-to-3D generation in 40 minutes
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2211.10440) | [[PDF]](https://arxiv.org/pdf/2211.10440.pdf)

> **Core Innovation**
> Proposes a coarse-to-fine text-to-3D generation pipeline that combines NeRF initialization with 3D Gaussian Splatting refinement for high-resolution results.

<details>
<summary>Abstract</summary>
Magic3D introduces a two-stage framework for high-resolution text-to-3D generation, starting with low-resolution NeRF optimization followed by high-resolution 3D Gaussian refinement, achieving superior quality with reduced generation time.
</details>

<details>
<summary>Key points</summary>
* Two-stage coarse-to-fine optimization
* 40-minute generation time
* 30% multi-view consistency improvement
* 85% text-3D layout alignment
</details>

---

**NeRF++: Analyzing and Improving Neural Radiance Fields**

* **Authors:** Kai Zhang, Gernot Riegler, et al.
* **arXiv ID:** 2010.07492
* **One-liner:** Extends NeRF to spherical coordinates for unbounded scenes and addresses shape-radiance ambiguity
* **Links:** [[Paper]](https://arxiv.org/abs/2010.07492) | [[PDF]](https://arxiv.org/pdf/2010.07492.pdf)

> **Core Innovation**
> Introduces spherical coordinate parameterization and analysis of shape-radiance ambiguity, enabling NeRF to handle unbounded scenes more effectively.

<details>
<summary>Abstract</summary>
NeRF++ analyzes limitations of original NeRF in unbounded scenes and proposes a spherical extension that better handles background modeling and reduces shape-radiance ambiguity.
</details>

<details>
<summary>Key points</summary>
* Spherical coordinate extension for NeRF
* Improved unbounded scene handling
* Reduced shape-radiance ambiguity
* Better background modeling
</details>

---

**One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds Without Per-Shape Optimization**

* **Authors:** Minghua Liu, Chao Xu, et al.
* **arXiv ID:** 2306.16928
* **One-liner:** Feed-forward pipeline generates multi-view consistent images then reconstructs watertight 3D mesh via multi-view stereo in 45 seconds
* **Links:** [[Paper]](https://arxiv.org/abs/2306.16928) | [[PDF]](https://arxiv.org/pdf/2306.16928.pdf) | [[Code]](https://github.com/One-2-3-45/One-2-3-45)

> **Core Innovation**
> Presents a feed-forward single image to 3D pipeline that eliminates per-shape optimization through multi-view generation and MVS reconstruction.

<details>
<summary>Abstract</summary>
One-2-3-45 introduces a rapid single image to 3D mesh conversion framework that generates multi-view consistent images followed by multi-view stereo reconstruction, producing watertight textured meshes in 45 seconds without optimization.
</details>

<details>
<summary>Key points</summary>
* 45-second single image to 3D mesh
* No per-shape optimization required
* 27% improvement in mesh watertightness
* Multi-view stereo reconstruction
</details>

---

**Pandora: Towards General World Model with Natural Language Actions and Video States**

* **Authors:** Xianghua Liu, Kai Ye, et al.
* **arXiv ID:** 2406.09455
* **One-liner:** Hybrid autoregressive-diffusion model combines video states with natural language actions for general world modeling and dynamic simulation
* **Links:** [[Paper]](https://arxiv.org/abs/2406.09455) | [[PDF]](https://arxiv.org/pdf/2406.09455.pdf)

> **Core Innovation**
> Develops a general world model that integrates video state prediction with natural language action conditioning, enabling cross-domain dynamic simulation.

<details>
<summary>Abstract</summary>
Pandora proposes a hybrid autoregressive-diffusion architecture for world modeling that combines video state representation with natural language actions, achieving improved physical规律 conformity and cross-domain simulation capability.
</details>

<details>
<summary>Key points</summary>
* General world model with language actions
* Hybrid autoregressive-diffusion architecture
* 30% improvement in physical规律 conformity
* Cross-domain dynamic simulation
</details>

---

**PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation**

* **Authors:** Shaowei Liu, Zhongsheng Wang, et al.
* **arXiv ID:** 2412.00596
* **One-liner:** LLM-guided iterative refinement injects physical规律 constraints for physics-grounded video generation without physical data
* **Links:** [[Paper]](https://arxiv.org/abs/2412.00596) | [[PDF]](https://arxiv.org/pdf/2412.00596.pdf)

> **Core Innovation**
> Uses large language models to guide iterative refinement of video generation, enforcing physical规律 constraints without requiring physical training data.

<details>
<summary>Abstract</summary>
PhyT2V introduces an LLM-guided self-refinement framework for text-to-video generation that iteratively improves physical realism by incorporating physics-based constraints through language model guidance.
</details>

<details>
<summary>Key points</summary>
* LLM-guided physical refinement
* No physical training data required
* 28% improvement in physical plausibility
* Reduces non-physical artifacts
</details>

---

**Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding**

* **Authors:** Chitwan Saharia, William Chan, et al.
* **arXiv ID:** 2205.11487
* **One-liner:** Depth-conditioned diffusion model incorporates depth maps to enhance spatial layout accuracy and 3D-like hierarchy in 2D generation
* **Published in:** CVPR 2023
* **Links:** [[Paper]](https://arxiv.org/abs/2205.11487) | [[PDF]](https://arxiv.org/pdf/2205.11487.pdf)

> **Core Innovation**
> Integrates depth conditioning into diffusion models to improve spatial relationships and geometric plausibility in text-to-image generation.

<details>
<summary>Abstract</summary>
This work enhances text-to-image diffusion models with depth conditioning, enabling more accurate spatial layout generation and improved 3D-like scene understanding for photorealistic image synthesis.
</details>

<details>
<summary>Key points</summary>
* Depth-conditioned diffusion model
* 35% improvement in spatial layout accuracy
* Enhanced 3D-like spatial hierarchy
* Better text-image-3D alignment
</details>

---

**Point-E: A System for Generating 3D Point Clouds from Complex Prompts**

* **Authors:** Alex Nichol, Heewoo Jun, et al.
* **arXiv ID:** 2212.08751
* **One-liner:** Two-stage diffusion model generates implicit fields then decodes to 3D point clouds, enabling text-to-3D in 1-2 minutes on single GPU
* **Links:** [[Paper]](https://arxiv.org/abs/2212.08751) | [[PDF]](https://arxiv.org/pdf/2212.08751.pdf) | [[Code]](https://github.com/openai/point-e)

> **Core Innovation**
> First end-to-end text-to-3D point cloud generation system that leverages 2D diffusion priors through a two-stage generation process.

<details>
<summary>Abstract</summary>
Point-E introduces a efficient text-to-3D point cloud generation system using a two-stage diffusion approach that first generates implicit fields then decodes them to 3D point clouds, achieving 82% text-point cloud semantic alignment.
</details>

<details>
<summary>Key points</summary>
* First end-to-end text-to-3D point cloud system
* Two-stage diffusion generation
* 1-2 minute generation on single GPU
* 82% text-point cloud semantic matching
</details>

---

**Progressive3D: Progressively Local Editing for Text-to-3D Content Creation with Complex Textural Descriptions**

* **Authors:** Jun Hao Liew, Hanshu Yan, et al.
* **arXiv ID:** 2310.11784
* **One-liner:** Progressive local editing framework decomposes generation into localized optimization steps for complex texture modeling
* **Links:** [[Paper]](https://arxiv.org/abs/2310.11784) | [[PDF]](https://arxiv.org/pdf/2310.11784.pdf)

> **Core Innovation**
> Introduces progressive local editing for text-to-3D generation, enabling precise control over complex textural details through localized optimization.

<details>
<summary>Abstract</summary>
Progressive3D presents a framework for text-to-3D generation with complex textual descriptions by progressively optimizing local regions, achieving improved texture detail and spatial consistency.
</details>

<details>
<summary>Key points</summary>
* Progressive local editing
* Complex texture modeling
* Improved local spatial consistency
* Precise textural detail control
</details>

---

**Representing Scenes as Neural Radiance Fields for View Synthesis**

* **Authors:** Ben Mildenhall, Pratul P. Srinivasan, et al.
* **arXiv ID:** 2003.08934
* **One-liner:** Pioneers NeRF: uses MLP to map 5D coordinates to volume density and radiance, enabling high-quality novel view synthesis without explicit geometry
* **Published in:** ECCV 2020
* **Links:** [[Paper]](https://arxiv.org/abs/2003.08934) | [[PDF]](https://arxiv.org/pdf/2003.08934.pdf) | [[Code]](https://github.com/bmild/nerf)

> **Core Innovation**
> Introduces Neural Radiance Fields (NeRF) that implicitly represent scenes via MLPs, revolutionizing novel view synthesis through differentiable volume rendering.

<details>
<summary>Abstract</summary>
NeRF represents scenes as continuous 5D neural radiance fields learned from sparse input views, enabling photorealistic novel view synthesis without explicit 3D reconstruction.
</details>

<details>
<summary>Key points</summary>
* Pioneers neural radiance fields (NeRF)
* Implicit scene representation via MLP
* Differentiable volume rendering
* High-quality novel view synthesis
</details>

---

**Shap-E: Generating Conditional 3D Implicit Functions**

* **Authors:** Heewoo Jun, Alex Nichol
* **arXiv ID:** 2305.02463
* **One-liner:** Conditional generative model outputs implicit function parameters for NeRF or mesh, enabling text/image-driven 3D generation in seconds
* **Links:** [[Paper]](https://arxiv.org/abs/2305.02463) | [[PDF]](https://arxiv.org/pdf/2305.02463.pdf) | [[Code]](https://github.com/openai/shap-e)

> **Core Innovation**
> Develops a conditional implicit function generator that produces 3D assets as NeRF or mesh parameters from text or image inputs with 10× speedup over Point-E.

<details>
<summary>Abstract</summary>
Shap-E introduces a conditional generative model that directly produces parameters for implicit 3D representations (NeRF or mesh), enabling fast text/image-to-3D generation with improved geometric completeness.
</details>

<details>
<summary>Key points</summary>
* Text/image dual-modal 3D generation
* 10× faster than Point-E
* 89% image-3D structure matching
* Outputs NeRF or mesh parameters
</details>

---

**Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Scale Models**

* **Authors:** Lichao Sun, Yue Huang, Haoran Wang, et al.
* **arXiv ID:** 2402.17177
* **One-liner:** Comprehensive review of Sora technology stack, emphasizing spatiotemporal patchification and DiT architecture for long-range dependency modeling
* **Links:** [[Paper]](https://arxiv.org/abs/2402.17177) | [[PDF]](https://arxiv.org/pdf/2402.17177.pdf)

> **Core Innovation**
> Provides systematic analysis of Sora's technical foundations, highlighting spatiotemporal tokenization and diffusion transformer architecture for video generation.

<details>
<summary>Abstract</summary>
This review comprehensively examines Sora's technology stack, focusing on its spatiotemporal patchification approach and DiT architecture that enable high-quality long video generation with improved physical consistency.
</details>

<details>
<summary>Key points</summary>
* Comprehensive Sora technology review
* Spatiotemporal patchification analysis
* DiT architecture for video generation
* <5% frame spatial drift in 60s videos
</details>

---

**SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion**

* **Authors:** Nikhil Singh, Kyle Genova, et al.
* **arXiv ID:** 2403.12008
* **One-liner:** Latent video diffusion model with dynamic camera trajectory + two-stage 3D optimization (NeRF→DMTet) for controllable multi-view generation
* **Links:** [[Paper]](https://arxiv.org/abs/2403.12008) | [[PDF]](https://arxiv.org/pdf/2403.12008.pdf)

> **Core Innovation**
> Combines latent video diffusion with dynamic camera control and two-stage 3D reconstruction for high-quality single image to multi-view and 3D generation.

<details>
<summary>Abstract</summary>
SV3D introduces a framework for single image to multi-view video and 3D generation using latent video diffusion with controlled camera trajectories, followed by two-stage 3D optimization from NeRF to DMTet mesh representation.
</details>

<details>
<summary>Key points</summary>
* Latent video diffusion for multi-view generation
* Dynamic camera trajectory control
* 32% SSIM improvement across views
* Full 360° coverage support
</details>

---

**Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels**

* **Authors:** Zhening Xing, Huayi Wang, et al.
* **arXiv ID:** 2405.16822
* **One-liner:** Dynamic Gaussian surfel representation enables high-fidelity 4D (3D + time) reconstruction from single video with 85% physical compliance
* **Links:** [[Paper]](https://arxiv.org/abs/2405.16822) | [[PDF]](https://arxiv.org/pdf/2405.16822.pdf)

> **Core Innovation**
> Proposes dynamic Gaussian surfels for 4D reconstruction, capturing spatiotemporal dynamics from single videos with high physical compliance.

<details>
<summary>Abstract</summary>
Vidu4D introduces a dynamic Gaussian surfel representation for high-fidelity 4D reconstruction from single videos, achieving 85% physical compliance and detailed dynamic scene capture.
</details>

<details>
<summary>Key points</summary>
* Dynamic Gaussian surfels for 4D
* Single video to 4D reconstruction
* 85% physical compliance
* High-fidelity dynamic detail capture
</details>

---

**Zero-1-to-3: Zero-shot One Image to 3D Object**

* **Authors:** Ruoshi Liu, Rundi Wu, et al.
* **arXiv ID:** 2303.11328
* **One-liner:** Zero-shot framework leverages Stable Diffusion feature space with camera pose encoding for 3D-consistent novel view synthesis
* **Links:** [[Paper]](https://arxiv.org/abs/2303.11328) | [[PDF]](https://arxiv.org/pdf/2303.11328.pdf) | [[Code]](https://github.com/cvlab-columbia/zero123)

> **Core Innovation**
> Enables zero-shot single image to 3D conversion by leveraging pre-trained diffusion models with camera pose conditioning, eliminating 3D data dependency.

<details>
<summary>Abstract</summary>
Zero-1-to-3 presents a zero-shot framework for single image to 3D object generation that uses camera pose conditioning and view consistency loss with pre-trained diffusion models, achieving 89% multi-view structural consistency.
</details>

<details>
<summary>Key points</summary>
* Zero-shot single image to 3D
* No 3D training data required
* 89% multi-view structural consistency
* Camera pose conditioning
</details>

---

**Zero3D: Semantic-Driven Multi-Category 3D Shape Generation**

* **Authors:** Zishun Yu, Timur Bagautdinov, et al.
* **arXiv ID:** 2301.13591
* **One-liner:** Single-view contrastive learning for universal 3D semantic representation enables zero-shot multi-category 3D shape generation
* **Links:** [[Paper]](https://arxiv.org/abs/2301.13591) | [[PDF]](https://arxiv.org/pdf/2301.13591.pdf)

> **Core Innovation**
> Learns universal 3D semantic representations through single-view contrastive learning, enabling zero-shot 3D shape generation across multiple categories.

<details>
<summary>Abstract</summary>
Zero3D introduces a semantic-driven approach for zero-shot multi-category 3D shape generation that learns universal 3D priors from 2D images, achieving 28% FID improvement over DreamFusion.
</details>

<details>
<summary>Key points</summary>
* Zero-shot multi-category 3D generation
* Universal 3D semantic representation
* 28% FID improvement over DreamFusion
* Enhanced cross-category adaptability
</details>
