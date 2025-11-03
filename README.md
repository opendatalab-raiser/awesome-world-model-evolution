# Awesome World Models Evolution

[English](README.md) | [中文](README_zh.md)

**[GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control]**

* **Authors:** [Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, Jun Gao]
* **arXiv ID:** [2503.03751]
*   **One-liner:** [Image (first frame/seed frame) → Video.3D caching + precise camera control, emphasizing world-consistent video generation]
*   **Published in:** arXiv [2025]
*   **Links:** [[Paper]](https://arxiv.org/abs/2503.03751) | [[PDF]](https://arxiv.org/pdf/2503.03751) | [[Code]](link_to_code_if_available) | [[Project Page]](https://research.nvidia.com/labs/toronto-ai/GEN3C/)
<details>
    <summary>Abstract</summary>
    
    [We present GEN3C, a generative video model with precise Camera Control and temporal 3D Consistency. Prior video models already generate realistic videos, but they tend to leverage little 3D information, leading to inconsistencies, such as objects popping in and out of existence. Camera control, if implemented at all, is imprecise, because camera parameters are mere inputs to the neural network which must then infer how the video depends on the camera. In contrast, GEN3C is guided by a 3D cache: point clouds obtained by predicting the pixel-wise depth of seed images or previously generated frames. When generating the next frames, GEN3C is conditioned on the 2D renderings of the 3D cache with the new camera trajectory provided by the user. Crucially, this means that GEN3C neither has to remember what it previously generated nor does it have to infer the image structure from the camera pose. The model, instead, can focus all its generative power on previously unobserved regions, as well as advancing the scene state to the next frame. Our results demonstrate more precise camera control than prior work, as well as state-of-the-art results in sparse-view novel view synthesis, even in challenging settings such as driving scenes and monocular dynamic video. Results are best viewed in videos. Check out our webpage! this https URL]
</details>
<details>
    <summary>Key points</summary>
    
   [Depth Anything v2 (metric version) is used to predict metric depth for a single reference image, and the pixels are back-projected into camera space to form a stable 3D representation (point cloud). Then, it is registered/scaled with the point cloud obtained by COLMAP triangulation of each training video, thereby unifying the camera trajectory from relative scale to metric scale. During inference, the camera trajectory preview is drawn and rendered in this interactive 3D point cloud.]
</details>

---