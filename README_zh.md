# Awesome World Models Evolution

[English](README.md) | [中文](README_zh.md)

**[GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control]**

* **作者：:** [Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, Jun Gao]
* **arXiv ID:** [2503.03751]
*   **简要介绍:** [图像（首帧/种子帧）→ 视频。3D缓存 + 精确相机控制，强调世界一致性的视频生成]
*   **发表时间:** arXiv [2025]
*   **链接:** [[Paper]](https://arxiv.org/abs/2503.03751) | [[PDF]](https://arxiv.org/pdf/2503.03751) | [[Code]](link_to_code_if_available) | [[Project Page]](https://research.nvidia.com/labs/toronto-ai/GEN3C/)
<details>
    <summary>Abstract</summary>
    
    [We present GEN3C, a generative video model with precise Camera Control and temporal 3D Consistency. Prior video models already generate realistic videos, but they tend to leverage little 3D information, leading to inconsistencies, such as objects popping in and out of existence. Camera control, if implemented at all, is imprecise, because camera parameters are mere inputs to the neural network which must then infer how the video depends on the camera. In contrast, GEN3C is guided by a 3D cache: point clouds obtained by predicting the pixel-wise depth of seed images or previously generated frames. When generating the next frames, GEN3C is conditioned on the 2D renderings of the 3D cache with the new camera trajectory provided by the user. Crucially, this means that GEN3C neither has to remember what it previously generated nor does it have to infer the image structure from the camera pose. The model, instead, can focus all its generative power on previously unobserved regions, as well as advancing the scene state to the next frame. Our results demonstrate more precise camera control than prior work, as well as state-of-the-art results in sparse-view novel view synthesis, even in challenging settings such as driving scenes and monocular dynamic video. Results are best viewed in videos. Check out our webpage! this https URL]
</details>
<details>
    <summary>Key points</summary>
      [使用 Depth Anything v2（metric 版本）对单张参考图预测米制深度，将像素回投影到相机空间形成稳定的 3D 表示（点云）；再与每个训练视频用 COLMAP 三角化得到的点云进行配准/尺度对齐，据此把相机轨迹从相对尺度统一到米制尺度；推理时在该交互式 3D 点云中绘制并渲染相机轨迹预览。]
</details>

---