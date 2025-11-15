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
