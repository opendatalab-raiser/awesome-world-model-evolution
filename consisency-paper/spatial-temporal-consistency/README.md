<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</p>

### Spatial + Temporal Consistency

Capability Profile: Models that maintain 3D spatial structure while simulating temporal dynamics, but may have limited language understanding or controllability.

Significance: These models represent crucial technical achievements in understanding "how the 3D world moves," forming the physics engine component of world models.

#### Representative Works

<details>
<summary><b>DUSt3R: Geometric 3D Vision Made Easy</b></summary>

* **Authors:** Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jérôme Revaud  
* **arXiv ID:** 2312.14132  
* **One-liner:** Dense, unconstrained stereo 3D reconstruction from arbitrary image collections with no calibration/preset pose required  
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2312.14132) | [[PDF]](https://arxiv.org/pdf/2312.14132.pdf)

> **Core Innovation**  
> Introduces a paradigm-shifting pipeline that replaces traditional multi-view stereo (MVS): it directly regresses dense point-maps without ever estimating camera intrinsics or extrinsics. By unifying single- and multi-view depth estimation, camera pose recovery, and reconstruction into one network, the approach excels even in casual, “in-the-wild” capture scenarios.

<details>
    <summary>Abstract</summary>
    Multi-view stereo reconstruction (MVS) in the wild requires to first estimate the camera parameters e.g. intrinsic and extrinsic parameters. These are usually tedious and cumbersome to obtain, yet they are mandatory to triangulate corresponding pixels in 3D space, which is the core of all best performing MVS algorithms. In this work, we take an opposite stance and introduce DUSt3R, a radically novel paradigm for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections, i.e. operating without prior information about camera calibration nor viewpoint poses. We cast the pairwise reconstruction problem as a regression of pointmaps, relaxing the hard constraints of usual projective camera models. We show that this formulation smoothly unifies the monocular and binocular reconstruction cases. In the case where more than two images are provided, we further propose a simple yet effective global alignment strategy that expresses all pairwise pointmaps in a common reference frame. We base our network architecture on standard Transformer encoders and decoders, allowing us to leverage powerful pretrained models. Our formulation directly provides a 3D model of the scene as well as depth information, but interestingly, we can seamlessly recover from it, pixel matches, relative and absolute camera. Exhaustive experiments on all these tasks showcase that the proposed DUSt3R can unify various 3D vision tasks and set new SoTAs on monocular/multi-view depth estimation as well as relative pose estimation. In summary, DUSt3R makes many geometric 3D vision tasks easy.
</details>

<details>
    <summary>Key points</summary>
    * Reformulates multi-view stereo as point-map regression, removing need for camera intrinsics/extrinsics.  
    * Unified framework that handles monocular and multi-view reconstruction seamlessly.  
    * Uses Transformer-based architecture to directly predict dense 3D geometry from images.  
    * Achieves state-of-the-art on depth, pose, and reconstruction benchmarks under unconstrained settings.  
</details>
</details>

---

<details>
<summary><b>4D Gaussian Splatting for Real-Time Dynamic Scene Rendering</b></summary>

* **Authors:** Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang  
* **arXiv ID:** 2310.08528  
* **One-liner:** A holistic 4D Gaussian splatting representation for dynamic scenes enabling real-time rendering of space-time geometry and appearance  
* **Published in:** CVPR 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2310.08528) | [[PDF]](https://arxiv.org/pdf/2310.08528.pdf) | [[Code]](https://github.com/hustvl/4DGaussians)

> **Core Innovation**  
> Proposes 4D Gaussian Splatting (4D-GS) that models space and time jointly: each Gaussian primitive carries spatio-temporal extent, dramatically boosting both training speed and real-time rendering performance on dynamic scenes.

<details>
    <summary>Abstract</summary>
    Representing and rendering dynamic scenes has been an important but challenging task. Especially, to accurately model complex motions, high efficiency is usually hard to guarantee. To achieve real-time dynamic scene rendering while also enjoying high training and storage efficiency, we propose 4D Gaussian Splatting (4D-GS) as a holistic representation for dynamic scenes rather than applying 3D-GS for each individual frame. In 4D-GS, a novel explicit representation containing both 3D Gaussians and 4D neural voxels is proposed. A decomposed neural voxel encoding algorithm inspired by HexPlane is proposed to efficiently build Gaussian features from 4D neural voxels and then a lightweight MLP is applied to predict Gaussian deformations at novel timestamps. Our 4D-GS method achieves real-time rendering under high resolutions, 82 FPS at an 800800 resolution on an RTX 3090 GPU while maintaining comparable or better quality than previous state-of-the-art methods.
</details>

<details>
    <summary>Key points</summary>
    * Introduces 4D Gaussian primitives that model space + time jointly (rather than separate per-frame 3D models).  
    * Enables real-time rendering (30+ fps) for dynamic scenes with high resolution.  
    * Efficient deformation field network and splatting renderer designed for dynamic geometry + appearance.  
    * Training and storage efficiency improved relative to prior dynamic scene representations.  
</details>
</details>

---

<details>
<summary><b>Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes</b></summary>

* **Authors:** Zhengqi Li, Simon Niklaus, Noah Snavely, Oliver Wang  
* **arXiv ID:** 2011.13084  
* **One-liner:** A continuous time-varying function representation capturing appearance, geometry and motion (scene flow) to perform novel view & time synthesis from monocular video  
* **Published in:** CVPR 2021  
* **Links:** [[Paper]](https://arxiv.org/abs/2011.13084) | [[PDF]](https://arxiv.org/pdf/2011.13084.pdf) | [[Code]](https://github.com/zhengqili/Neural-Scene-Flow-Fields)

> **Core Innovation**  
> Proposes Neural Scene Flow Fields (NSFF): a time-varying continuous representation that jointly encodes geometry, appearance, and 3D scene flow for dynamic scenes. Trained on monocular videos with known camera trajectories, NSFF enables simultaneous novel-view and temporal interpolation.

<details>
    <summary>Abstract</summary>
    We present a method to perform novel view and time synthesis of dynamic scenes, requiring only a monocular video with known camera poses as input. To do this, we introduce Neural Scene Flow Fields, a new representation that models the dynamic scene as a time-variant continuous function of appearance, geometry, and 3D scene motion. Our representation is optimized through a neural network to fit the observed input views. We show that our representation can be used for complex dynamic scenes, including thin structures, view-dependent effects, and natural degrees of motion. We conduct a number of experiments that demonstrate our approach significantly outperforms recent monocular view synthesis methods, and show qualitative results of space-time view synthesis on a variety of real-world videos.
</details>

<details>
    <summary>Key points</summary>
    * Represents dynamic scenes by a continuous 5D/6D function (space + time + view) capturing geometry, appearance & scene-flow.  
    * Optimized from input monocular video + known camera poses (no multi-view capture required).  
    * Supports novel view and novel time synthesis (i.e., view + motion interpolation).  
    * Handles challenging dynamic phenomena such as thin structures, specularities and complex motion.  
</details>
</details>

---

<details>
<summary><b>CoTracker: It is Better to Track Together</b></summary>

* **Authors:** Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht 
* **arXiv ID:** 2307.07635  
* **One-liner:** A transformer-based model that jointly tracks tens of thousands of 2D points across video frames rather than treating them independently  
* **Published in:** ECCV 2024
* **Links:** [[Paper]](https://arxiv.org/abs/2307.07635) | [[PDF]](https://arxiv.org/pdf/2307.07635.pdf) | [[Code]](https://github.com/facebookresearch/co-tracker)

> **Core Innovation**  
> Introduces CoTracker: a Transformer-based architecture that jointly tracks thousands of 2D point trajectories, explicitly modeling inter-point dependencies to boost tracking accuracy for occluded or out-of-view points.

<details>
    <summary>Abstract</summary>
    We introduce CoTracker, a transformer-based model that tracks a large number of 2D points in long video sequences. Differently from most existing approaches that track points independently, CoTracker tracks them jointly, accounting for their dependencies. We show that joint tracking significantly improves tracking accuracy and robustness, and allows CoTracker to track occluded points and points outside of the camera view. We also introduce several innovations for this class of trackers, including using token proxies that significantly improve memory efficiency and allow CoTracker to track 70k points jointly and simultaneously at inference on a single GPU. CoTracker is an online algorithm that operates causally on short windows. However, it is trained utilizing unrolled windows as a recurrent network, maintaining tracks for long periods of time even when points are occluded or leave the field of view. Quantitatively, CoTracker substantially outperforms prior trackers on standard point-tracking benchmarks.
</details>

<details>
    <summary>Key points</summary>
    * Joint point-tracking: tracks many points together, modeling correlations rather than independent tracking.  
    * Uses transformer architecture with token proxies to manage memory and scale to tens of thousands of points.  
    * Works online in causal short windows but is trained as an unrolled long-window recurrent to handle long sequences and occlusion.  
    * Significant improvements in robustness and accuracy over traditional independent point tracking methods, especially under occlusion and out-of-view.  
</details>
</details>

---

<details>
<summary><b>GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control</b></summary>

* **Authors:** Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, Jun Gao
* **arXiv ID:** 2503.03751
*   **One-liner:** Image (first frame/seed frame) → Video.3D caching + precise camera control, emphasizing world-consistent video generation
*   **Published in:** CVPR 2025
*   **Links:** [[Paper]](https://arxiv.org/abs/2503.03751) | [[PDF]](https://arxiv.org/pdf/2503.03751) | [[Code]](https://github.com/nv-tlabs/GEN3C)
   
> **Core Innovation**  
> Turns camera poses into directly-renderable conditions via a 3D cache, leaving the model to fill only the dis-occluded regions for the new viewpoint—eliminating both imprecise camera control and temporal inconsistency in one shot.

<details>
    <summary>Abstract</summary>
    We present GEN3C, a generative video model with precise Camera Control and temporal 3D Consistency. Prior video models already generate realistic videos, but they tend to leverage little 3D information, leading to inconsistencies, such as objects popping in and out of existence. Camera control, if implemented at all, is imprecise, because camera parameters are mere inputs to the neural network which must then infer how the video depends on the camera. In contrast, GEN3C is guided by a 3D cache: point clouds obtained by predicting the pixel-wise depth of seed images or previously generated frames. When generating the next frames, GEN3C is conditioned on the 2D renderings of the 3D cache with the new camera trajectory provided by the user. Crucially, this means that GEN3C neither has to remember what it previously generated nor does it have to infer the image structure from the camera pose. The model, instead, can focus all its generative power on previously unobserved regions, as well as advancing the scene state to the next frame. Our results demonstrate more precise camera control than prior work, as well as state-of-the-art results in sparse-view novel view synthesis, even in challenging settings such as driving scenes and monocular dynamic video. Results are best viewed in videos.
</details>
<details>
    <summary>Key points</summary>
   Depth Anything v2 (metric version) is used to predict metric depth for a single reference image, and the pixels are back-projected into camera space to form a stable 3D representation (point cloud). Then, it is registered/scaled with the point cloud obtained by COLMAP triangulation of each training video, thereby unifying the camera trajectory from relative scale to metric scale. During inference, the camera trajectory preview is drawn and rendered in this interactive 3D point cloud.
</details>
</details>

---
