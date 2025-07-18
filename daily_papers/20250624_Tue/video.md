# 计算机视觉 cs.CV

- **最新发布 222 篇**

- **更新 113 篇**

## 最新发布

#### [new 001] ELMAR: Enhancing LiDAR Detection with 4D Radar Motion Awareness and Cross-modal Uncertainty
- **分类: cs.CV**

- **简介: 该论文属于多模态感知任务，旨在解决LiDAR与4D雷达数据对齐问题。通过引入运动信息和不确定性估计，提升检测精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.17958v1](http://arxiv.org/pdf/2506.17958v1)**

> **作者:** Xiangyuan Peng; Miao Tang; Huawei Sun; Bierzynski Kay; Lorenzo Servadei; Robert Wille
>
> **备注:** 7 pages. Accepted by IROS2025
>
> **摘要:** LiDAR and 4D radar are widely used in autonomous driving and robotics. While LiDAR provides rich spatial information, 4D radar offers velocity measurement and remains robust under adverse conditions. As a result, increasing studies have focused on the 4D radar-LiDAR fusion method to enhance the perception. However, the misalignment between different modalities is often overlooked. To address this challenge and leverage the strengths of both modalities, we propose a LiDAR detection framework enhanced by 4D radar motion status and cross-modal uncertainty. The object movement information from 4D radar is first captured using a Dynamic Motion-Aware Encoding module during feature extraction to enhance 4D radar predictions. Subsequently, the instance-wise uncertainties of bounding boxes are estimated to mitigate the cross-modal misalignment and refine the final LiDAR predictions. Extensive experiments on the View-of-Delft (VoD) dataset highlight the effectiveness of our method, achieving state-of-the-art performance with the mAP of 74.89% in the entire area and 88.70% within the driving corridor while maintaining a real-time inference speed of 30.02 FPS.
>
---
#### [new 002] Enhancing Wireless Device Identification through RF Fingerprinting: Leveraging Transient Energy Spectrum Analysis
- **分类: cs.CV**

- **简介: 该论文属于无线设备识别任务，旨在解决复杂电磁环境中准确识别RF设备的问题。通过分析瞬态能量谱并使用CNN-Bi-GRU模型进行分类，实现了高精度的设备识别。**

- **链接: [http://arxiv.org/pdf/2506.17439v1](http://arxiv.org/pdf/2506.17439v1)**

> **作者:** Nisar Ahmed; Gulshan Saleem; Hafiz Muhammad Shahzad Asif; Muhammad Usman Younus; Kalsoom Safdar
>
> **备注:** Submitted in Wireless Personal Communications
>
> **摘要:** In recent years, the rapid growth of the Internet of Things technologies and the widespread adoption of 5G wireless networks have led to an exponential increase in the number of radiation devices operating in complex electromagnetic environments. A key challenge in managing and securing these devices is accurate identification and classification. To address this challenge, specific emitter identification techniques have emerged as a promising solution that aims to provide reliable and efficient means of identifying individual radiation devices in a unified and standardized manner. This research proposes an approach that leverages transient energy spectrum analysis using the General Linear Chirplet Transform to extract features from RF devices. A dataset comprising nine RF devices is utilized, with each sample containing 900 attributes and a total of 1080 equally distributed samples across the devices. These features are then used in a classification modeling framework. To overcome the limitations of conventional machine learning methods, we introduce a hybrid deep learning model called the CNN-Bi-GRU for learning the identification of RF devices based on their transient characteristics. The proposed approach provided a 10-fold cross-validation performance with a precision of 99.33%, recall of 99.53%, F1-score of 99.43%, and classification accuracy of 99.17%. The results demonstrate the promising classification performance of the CNN-Bi-GRU approach, indicating its suitability for accurately identifying RF devices based on their transient characteristics and its potential for enhancing device identification and classification in complex wireless environments.
>
---
#### [new 003] AViLA: Asynchronous Vision-Language Agent for Streaming Multimodal Data Interaction
- **分类: cs.CV**

- **简介: 该论文属于多模态交互任务，解决流数据中查询与证据异步的问题。提出AViLA模型，具备时间感知的响应能力。**

- **链接: [http://arxiv.org/pdf/2506.18472v1](http://arxiv.org/pdf/2506.18472v1)**

> **作者:** Gengyuan Zhang; Tanveer Hannan; Hermine Kleiner; Beste Aydemir; Xinyu Xie; Jian Lan; Thomas Seidl; Volker Tresp; Jindong Gu
>
> **备注:** preprint version; 23 pages (including references and appendix)
>
> **摘要:** An ideal vision-language agent serves as a bridge between the human users and their surrounding physical world in real-world applications like autonomous driving and embodied agents, and proactively provides accurate and timely responses given user intents. An intriguing challenge arises when agents interact with the world as a dynamic data stream and ad-hoc queries from users: supporting knowledge for queries, namely evidence, usually appears asynchronously with the arrival time of queries, and agents need to ground their responses in historical data, present observations, and even future streams. We frame this challenge as Query-Evidence Asynchrony, where user queries and their supporting evidence typically arrive asynchronously in the streaming setting. This setting requires not only strong reasoning capabilities but also the ability to retain past observations and respond to queries with temporal awareness. In this paper, we introduce a diagnostic benchmark that evaluates Multimodal Large Language Models (MLLMs) on their ability to handle interaction with streaming data. Further, we present AViLA, Asynchronous Video-Language Agent for streaming data interaction that can handle ad-hoc queries and give time-aware responses. For this purpose, AViLA consists of three key modules: comprehensive memory retention, evidence identification, and evidence-grounded trigger, that are designed to maintain a general-purpose memory and respond readily and timely to queries. Our experiments show that existing models often fail to respond at appropriate times, while AViLA significantly improves both accuracy and temporal awareness. Our code and dataset will be publicly available.
>
---
#### [new 004] GEMeX-ThinkVG: Towards Thinking with Visual Grounding in Medical VQA via Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉问答任务，旨在提升模型答案的可靠性与可解释性。通过构建ThinkVG数据集和引入验证奖励机制，实现更准确的视觉推理与解释。**

- **链接: [http://arxiv.org/pdf/2506.17939v1](http://arxiv.org/pdf/2506.17939v1)**

> **作者:** Bo Liu; Xiangyu Zhao; Along He; Yidi Chen; Huazhu Fu; Xiao-Ming Wu
>
> **备注:** Work in Progress
>
> **摘要:** Medical visual question answering aims to support clinical decision-making by enabling models to answer natural language questions based on medical images. While recent advances in multi-modal learning have significantly improved performance, current methods still suffer from limited answer reliability and poor interpretability, impairing the ability of clinicians and patients to understand and trust model-generated answers. To address this, this work first proposes a Thinking with Visual Grounding (ThinkVG) dataset wherein the answer generation is decomposed into intermediate reasoning steps that explicitly ground relevant visual regions of the medical image, thereby providing fine-grained explainability. Furthermore, we introduce a novel verifiable reward mechanism for reinforcement learning to guide post-training, improving the alignment between the model's reasoning process and its final answer. Remarkably, our method achieves comparable performance using only one-eighth of the training data, demonstrating the efficiency and effectiveness of the proposal. The dataset is available at https://huggingface.co/datasets/BoKelvin/GEMeX-ThinkVG.
>
---
#### [new 005] A Novel Multi-layer Task-centric and Data Quality Framework for Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶领域，旨在解决多源数据质量与任务需求不匹配的问题。提出多层框架以提升系统可靠性与性能。**

- **链接: [http://arxiv.org/pdf/2506.17346v1](http://arxiv.org/pdf/2506.17346v1)**

> **作者:** Yuhan Zhou; Haihua Chen; Kewei Sha
>
> **摘要:** The next-generation autonomous vehicles (AVs), embedded with frequent real-time decision-making, will rely heavily on a large volume of multisource and multimodal data. In real-world settings, the data quality (DQ) of different sources and modalities usually varies due to unexpected environmental factors or sensor issues. However, both researchers and practitioners in the AV field overwhelmingly concentrate on models/algorithms while undervaluing the DQ. To fulfill the needs of the next-generation AVs with guarantees of functionality, efficiency, and trustworthiness, this paper proposes a novel task-centric and data quality vase framework which consists of five layers: data layer, DQ layer, task layer, application layer, and goal layer. The proposed framework aims to map DQ with task requirements and performance goals. To illustrate, a case study investigating redundancy on the nuScenes dataset proves that partially removing redundancy on multisource image data could improve YOLOv8 object detection task performance. Analysis on multimodal data of image and LiDAR further presents existing redundancy DQ issues. This paper opens up a range of critical but unexplored challenges at the intersection of DQ, task orchestration, and performance-oriented system development in AVs. It is expected to guide the AV community toward building more adaptive, explainable, and resilient AVs that respond intelligently to dynamic environments and heterogeneous data streams. Code, data, and implementation details are publicly available at: https://anonymous.4open.science/r/dq4av-framework/README.md.
>
---
#### [new 006] Let Your Video Listen to Your Music!
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频与音乐节奏对齐任务，旨在自动调整视频节奏以匹配音乐，同时保留原始画面内容。通过分步框架实现关键帧对齐与视频修复。**

- **链接: [http://arxiv.org/pdf/2506.18881v1](http://arxiv.org/pdf/2506.18881v1)**

> **作者:** Xinyu Zhang; Dong Gong; Zicheng Duan; Anton van den Hengel; Lingqiao Liu
>
> **备注:** project page: https://zhangxinyu-xyz.github.io/MVAA/
>
> **摘要:** Aligning the rhythm of visual motion in a video with a given music track is a practical need in multimedia production, yet remains an underexplored task in autonomous video editing. Effective alignment between motion and musical beats enhances viewer engagement and visual appeal, particularly in music videos, promotional content, and cinematic editing. Existing methods typically depend on labor-intensive manual cutting, speed adjustments, or heuristic-based editing techniques to achieve synchronization. While some generative models handle joint video and music generation, they often entangle the two modalities, limiting flexibility in aligning video to music beats while preserving the full visual content. In this paper, we propose a novel and efficient framework, termed MVAA (Music-Video Auto-Alignment), that automatically edits video to align with the rhythm of a given music track while preserving the original visual content. To enhance flexibility, we modularize the task into a two-step process in our MVAA: aligning motion keyframes with audio beats, followed by rhythm-aware video inpainting. Specifically, we first insert keyframes at timestamps aligned with musical beats, then use a frame-conditioned diffusion model to generate coherent intermediate frames, preserving the original video's semantic content. Since comprehensive test-time training can be time-consuming, we adopt a two-stage strategy: pretraining the inpainting module on a small video set to learn general motion priors, followed by rapid inference-time fine-tuning for video-specific adaptation. This hybrid approach enables adaptation within 10 minutes with one epoch on a single NVIDIA 4090 GPU using CogVideoX-5b-I2V as the backbone. Extensive experiments show that our approach can achieve high-quality beat alignment and visual smoothness.
>
---
#### [new 007] Trustworthy Few-Shot Transfer of Medical VLMs through Split Conformal Prediction
- **分类: cs.CV**

- **简介: 该论文属于医学视觉语言模型的迁移学习任务，旨在提升模型在少量标注数据下的可靠性。通过改进的分拆共形预测方法，提高模型的置信度和泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.17503v1](http://arxiv.org/pdf/2506.17503v1)**

> **作者:** Julio Silva-Rodríguez; Ismail Ben Ayed; Jose Dolz
>
> **备注:** MICCAI 2025. Code: https://github.com/jusiro/SCA-T
>
> **摘要:** Medical vision-language models (VLMs) have demonstrated unprecedented transfer capabilities and are being increasingly adopted for data-efficient image classification. Despite its growing popularity, its reliability aspect remains largely unexplored. This work explores the split conformal prediction (SCP) framework to provide trustworthiness guarantees when transferring such models based on a small labeled calibration set. Despite its potential, the generalist nature of the VLMs' pre-training could negatively affect the properties of the predicted conformal sets for specific tasks. While common practice in transfer learning for discriminative purposes involves an adaptation stage, we observe that deploying such a solution for conformal purposes is suboptimal since adapting the model using the available calibration data breaks the rigid exchangeability assumptions for test data in SCP. To address this issue, we propose transductive split conformal adaptation (SCA-T), a novel pipeline for transfer learning on conformal scenarios, which performs an unsupervised transductive adaptation jointly on calibration and test data. We present comprehensive experiments utilizing medical VLMs across various image modalities, transfer tasks, and non-conformity scores. Our framework offers consistent gains in efficiency and conditional coverage compared to SCP, maintaining the same empirical guarantees.
>
---
#### [new 008] Cross-Architecture Knowledge Distillation (KD) for Retinal Fundus Image Anomaly Detection on NVIDIA Jetson Nano
- **分类: cs.CV; cs.AI; cs.LG; 68T07; I.2.6; I.5.1; J.3**

- **简介: 该论文属于医学图像分类任务，旨在解决资源匮乏地区视网膜疾病检测难题。通过跨架构知识蒸馏，将高性能ViT模型压缩为轻量CNN模型，实现准确且高效的诊断。**

- **链接: [http://arxiv.org/pdf/2506.18220v1](http://arxiv.org/pdf/2506.18220v1)**

> **作者:** Berk Yilmaz; Aniruddh Aiyengar
>
> **备注:** 15 pages, 10 figures. Berk Yilmaz and Aniruddh Aiyengar contributed equally to this work
>
> **摘要:** Early and accurate identification of retinal ailments is crucial for averting ocular decline; however, access to dependable diagnostic devices is not often available in low-resourced settings. This project proposes to solve that by developing a lightweight, edge-device deployable disease classifier using cross-architecture knowledge distilling. We first train a high-capacity vision transformer (ViT) teacher model, pre-trained using I-JEPA self-supervised learning, to classify fundus images into four classes: Normal, Diabetic Retinopathy, Glaucoma, and Cataract. We kept an Internet of Things (IoT) focus when compressing to a CNN-based student model for deployment in resource-limited conditions, such as the NVIDIA Jetson Nano. This was accomplished using a novel framework which included a Partitioned Cross-Attention (PCA) projector, a Group-Wise Linear (GL) projector, and a multi-view robust training method. The teacher model has 97.4 percent more parameters than the student model, with it achieving 89 percent classification with a roughly 93 percent retention of the teacher model's diagnostic performance. The retention of clinical classification behavior supports our method's initial aim: compression of the ViT while retaining accuracy. Our work serves as an example of a scalable, AI-driven triage solution for retinal disorders in under-resourced areas.
>
---
#### [new 009] Limitations of NERF with pre-trained Vision Features for Few-Shot 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决少样本场景下的重建问题。通过对比不同特征融合方法，发现预训练视觉特征反而降低性能，揭示其局限性。**

- **链接: [http://arxiv.org/pdf/2506.18208v1](http://arxiv.org/pdf/2506.18208v1)**

> **作者:** Ankit Sanjyal
>
> **备注:** 5 pages, 1 table, 2 figures. First submission. Code available at: \url{https://github.com/ANKITSANJYAL/nerf-few-shot-limitations}
>
> **摘要:** Neural Radiance Fields (NeRF) have revolutionized 3D scene reconstruction from sparse image collections. Recent work has explored integrating pre-trained vision features, particularly from DINO, to enhance few-shot reconstruction capabilities. However, the effectiveness of such approaches remains unclear, especially in extreme few-shot scenarios. In this paper, we present a systematic evaluation of DINO-enhanced NeRF models, comparing baseline NeRF, frozen DINO features, LoRA fine-tuned features, and multi-scale feature fusion. Surprisingly, our experiments reveal that all DINO variants perform worse than the baseline NeRF, achieving PSNR values around 12.9 to 13.0 compared to the baseline's 14.71. This counterintuitive result suggests that pre-trained vision features may not be beneficial for few-shot 3D reconstruction and may even introduce harmful biases. We analyze potential causes including feature-task mismatch, overfitting to limited data, and integration challenges. Our findings challenge common assumptions in the field and suggest that simpler architectures focusing on geometric consistency may be more effective for few-shot scenarios.
>
---
#### [new 010] MedSeg-R: Medical Image Segmentation with Clinical Reasoning
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决小病灶识别困难和重叠结构分割不准的问题。提出MedSeg-R框架，结合临床推理增强语义先验，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.18669v1](http://arxiv.org/pdf/2506.18669v1)**

> **作者:** Hao Shao; Qibin Hou
>
> **摘要:** Medical image segmentation is challenging due to overlapping anatomies with ambiguous boundaries and a severe imbalance between the foreground and background classes, which particularly affects the delineation of small lesions. Existing methods, including encoder-decoder networks and prompt-driven variants of the Segment Anything Model (SAM), rely heavily on local cues or user prompts and lack integrated semantic priors, thus failing to generalize well to low-contrast or overlapping targets. To address these issues, we propose MedSeg-R, a lightweight, dual-stage framework inspired by inspired by clinical reasoning. Its cognitive stage interprets medical report into structured semantic priors (location, texture, shape), which are fused via transformer block. In the perceptual stage, these priors modulate the SAM backbone: spatial attention highlights likely lesion regions, dynamic convolution adapts feature filters to expected textures, and deformable sampling refines spatial support. By embedding this fine-grained guidance early, MedSeg-R disentangles inter-class confusion and amplifies minority-class cues, greatly improving sensitivity to small lesions. In challenging benchmarks, MedSeg-R produces large Dice improvements in overlapping and ambiguous structures, demonstrating plug-and-play compatibility with SAM-based systems.
>
---
#### [new 011] Programmable-Room: Interactive Textured 3D Room Meshes Generation Empowered by Large Language Models
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于3D场景生成任务，旨在通过自然语言指令交互生成和编辑3D房间网格。工作包括分解任务、引入视觉编程框架及优化纹理生成。**

- **链接: [http://arxiv.org/pdf/2506.17707v1](http://arxiv.org/pdf/2506.17707v1)**

> **作者:** Jihyun Kim; Junho Park; Kyeongbo Kong; Suk-Ju Kang
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** We present Programmable-Room, a framework which interactively generates and edits a 3D room mesh, given natural language instructions. For precise control of a room's each attribute, we decompose the challenging task into simpler steps such as creating plausible 3D coordinates for room meshes, generating panorama images for the texture, constructing 3D meshes by integrating the coordinates and panorama texture images, and arranging furniture. To support the various decomposed tasks with a unified framework, we incorporate visual programming (VP). VP is a method that utilizes a large language model (LLM) to write a Python-like program which is an ordered list of necessary modules for the various tasks given in natural language. We develop most of the modules. Especially, for the texture generating module, we utilize a pretrained large-scale diffusion model to generate panorama images conditioned on text and visual prompts (i.e., layout, depth, and semantic map) simultaneously. Specifically, we enhance the panorama image generation quality by optimizing the training objective with a 1D representation of a panorama scene obtained from bidirectional LSTM. We demonstrate Programmable-Room's flexibility in generating and editing 3D room meshes, and prove our framework's superiority to an existing model quantitatively and qualitatively. Project page is available in https://jihyun0510.github.io/Programmable_Room_Page/.
>
---
#### [new 012] TEM^3-Learning: Time-Efficient Multimodal Multi-Task Learning for Advanced Assistive Driving
- **分类: cs.CV**

- **简介: 该论文属于多任务学习领域，旨在解决辅助驾驶中的多模态任务协同问题。提出TEM^3-Learning框架，提升实时性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.18084v1](http://arxiv.org/pdf/2506.18084v1)**

> **作者:** Wenzhuo Liu; Yicheng Qiao; Zhen Wang; Qiannan Guo; Zilong Chen; Meihua Zhou; Xinran Li; Letian Wang; Zhiwei Li; Huaping Liu; Wenshuo Wang
>
> **摘要:** Multi-task learning (MTL) can advance assistive driving by exploring inter-task correlations through shared representations. However, existing methods face two critical limitations: single-modality constraints limiting comprehensive scene understanding and inefficient architectures impeding real-time deployment. This paper proposes TEM^3-Learning (Time-Efficient Multimodal Multi-task Learning), a novel framework that jointly optimizes driver emotion recognition, driver behavior recognition, traffic context recognition, and vehicle behavior recognition through a two-stage architecture. The first component, the mamba-based multi-view temporal-spatial feature extraction subnetwork (MTS-Mamba), introduces a forward-backward temporal scanning mechanism and global-local spatial attention to efficiently extract low-cost temporal-spatial features from multi-view sequential images. The second component, the MTL-based gated multimodal feature integrator (MGMI), employs task-specific multi-gating modules to adaptively highlight the most relevant modality features for each task, effectively alleviating the negative transfer problem in MTL. Evaluation on the AIDE dataset, our proposed model achieves state-of-the-art accuracy across all four tasks, maintaining a lightweight architecture with fewer than 6 million parameters and delivering an impressive 142.32 FPS inference speed. Rigorous ablation studies further validate the effectiveness of the proposed framework and the independent contributions of each module. The code is available on https://github.com/Wenzhuo-Liu/TEM3-Learning.
>
---
#### [new 013] Make It Efficient: Dynamic Sparse Attention for Autoregressive Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，解决自回归模型推理时内存和计算效率问题。提出ADSA方法动态优化注意力机制，提升生成质量和资源效率。**

- **链接: [http://arxiv.org/pdf/2506.18226v1](http://arxiv.org/pdf/2506.18226v1)**

> **作者:** Xunzhi Xiang; Qi Fan
>
> **摘要:** Autoregressive conditional image generation models have emerged as a dominant paradigm in text-to-image synthesis. These methods typically convert images into one-dimensional token sequences and leverage the self-attention mechanism, which has achieved remarkable success in natural language processing, to capture long-range dependencies, model global context, and ensure semantic coherence. However, excessively long contexts during inference lead to significant memory overhead caused by KV-cache and computational delays. To alleviate these challenges, we systematically analyze how global semantics, spatial layouts, and fine-grained textures are formed during inference, and propose a novel training-free context optimization method called Adaptive Dynamic Sparse Attention (ADSA). Conceptually, ADSA dynamically identifies historical tokens crucial for maintaining local texture consistency and those essential for ensuring global semantic coherence, thereby efficiently streamlining attention computation. Additionally, we introduce a dynamic KV-cache update mechanism tailored for ADSA, reducing GPU memory consumption during inference by approximately $50\%$. Extensive qualitative and quantitative experiments demonstrate the effectiveness and superiority of our approach in terms of both generation quality and resource efficiency.
>
---
#### [new 014] Matrix-Game: Interactive World Foundation Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Matrix-Game，属于交互式世界生成任务，解决Minecraft世界可控生成问题。通过两阶段训练和自建数据集，实现高精度控制与物理一致性。**

- **链接: [http://arxiv.org/pdf/2506.18701v1](http://arxiv.org/pdf/2506.18701v1)**

> **作者:** Yifan Zhang; Chunli Peng; Boyang Wang; Puyi Wang; Qingcheng Zhu; Fei Kang; Biao Jiang; Zedong Gao; Eric Li; Yang Liu; Yahui Zhou
>
> **备注:** Technical Report
>
> **摘要:** We introduce Matrix-Game, an interactive world foundation model for controllable game world generation. Matrix-Game is trained using a two-stage pipeline that first performs large-scale unlabeled pretraining for environment understanding, followed by action-labeled training for interactive video generation. To support this, we curate Matrix-Game-MC, a comprehensive Minecraft dataset comprising over 2,700 hours of unlabeled gameplay video clips and over 1,000 hours of high-quality labeled clips with fine-grained keyboard and mouse action annotations. Our model adopts a controllable image-to-world generation paradigm, conditioned on a reference image, motion context, and user actions. With over 17 billion parameters, Matrix-Game enables precise control over character actions and camera movements, while maintaining high visual quality and temporal coherence. To evaluate performance, we develop GameWorld Score, a unified benchmark measuring visual quality, temporal quality, action controllability, and physical rule understanding for Minecraft world generation. Extensive experiments show that Matrix-Game consistently outperforms prior open-source Minecraft world models (including Oasis and MineWorld) across all metrics, with particularly strong gains in controllability and physical consistency. Double-blind human evaluations further confirm the superiority of Matrix-Game, highlighting its ability to generate perceptually realistic and precisely controllable videos across diverse game scenarios. To facilitate future research on interactive image-to-world generation, we will open-source the Matrix-Game model weights and the GameWorld Score benchmark at https://github.com/SkyworkAI/Matrix-Game.
>
---
#### [new 015] CSDN: A Context-Gated Self-Adaptive Detection Network for Real-Time Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决CNN检测器因感受野有限导致的全局上下文信息不足问题。提出CSDN网络，通过门控机制提升特征利用效率和多尺度适应能力。**

- **链接: [http://arxiv.org/pdf/2506.17679v1](http://arxiv.org/pdf/2506.17679v1)**

> **作者:** Wei Haolin
>
> **备注:** 15pages, 11figures
>
> **摘要:** Convolutional neural networks (CNNs) have long been the cornerstone of target detection, but they are often limited by limited receptive fields, which hinders their ability to capture global contextual information. This paper believes that the effective utilization of extracted features is as important as the feature extraction process itself. We critically re-evaluated the DETR-inspired header network architecture, questioning the indispensable nature of its self-attention mechanism, and discovering significant information redundancies. To solve these problems, we introduced the Context-Gated Scale-Adaptive Detection Network (CSDN), a Transformer-based detection header inspired by natural language processing architecture and human visual perception. CSDN aims to efficiently utilize the characteristics of the CNN backbone network by replacing the traditional stacked self-attention and cross-attention layers with a novel gating mechanism. This mechanism enables each region of interest (ROI) to adaptively select and combine feature dimensions and scale information from multiple attention patterns. CSDN provides more powerful global context modeling capabilities and can better adapt to objects of different sizes and structures. Our proposed detection head can directly replace the native heads of various CNN-based detectors, and only a few rounds of fine-tuning on the pre-training weights can significantly improve the detection accuracy, thus avoiding the need to achieve small improvements. Various layer modules undergo extensive re-training.
>
---
#### [new 016] Targeted False Positive Synthesis via Detector-guided Adversarial Diffusion Attacker for Robust Polyp Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决polyp检测中的假阳性问题。通过对抗扩散框架生成高价值假阳性数据，提升检测器性能。**

- **链接: [http://arxiv.org/pdf/2506.18134v1](http://arxiv.org/pdf/2506.18134v1)**

> **作者:** Quan Zhou; Gan Luo; Qiang Hu; Qingyong Zhang; Jinhua Zhang; Yinjiao Tian; Qiang Li; Zhiwei Wang
>
> **备注:** Early Accepted by MICCAI 2025
>
> **摘要:** Polyp detection is crucial for colorectal cancer screening, yet existing models are limited by the scale and diversity of available data. While generative models show promise for data augmentation, current methods mainly focus on enhancing polyp diversity, often overlooking the critical issue of false positives. In this paper, we address this gap by proposing an adversarial diffusion framework to synthesize high-value false positives. The extensive variability of negative backgrounds presents a significant challenge in false positive synthesis. To overcome this, we introduce two key innovations: First, we design a regional noise matching strategy to construct a negative synthesis space using polyp detection datasets. This strategy trains a negative-centric diffusion model by masking polyp regions, ensuring the model focuses exclusively on learning diverse background patterns. Second, we introduce the Detector-guided Adversarial Diffusion Attacker (DADA) module, which perturbs the negative synthesis process to disrupt a pre-trained detector's decision, guiding the negative-centric diffusion model to generate high-value, detector-confusing false positives instead of low-value, ordinary backgrounds. Our approach is the first to apply adversarial diffusion to lesion detection, establishing a new paradigm for targeted false positive synthesis and paving the way for more reliable clinical applications in colorectal cancer screening. Extensive results on public and in-house datasets verify the superiority of our method over the current state-of-the-arts, with our synthesized data improving the detectors by at least 2.6% and 2.7% in F1-score, respectively, over the baselines. Codes are at https://github.com/Huster-Hq/DADA.
>
---
#### [new 017] RadarSeq: A Temporal Vision Framework for User Churn Prediction via Radar Chart Sequences
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于用户流失预测任务，解决非订阅型平台中隐式行为的早期检测问题。通过雷达图序列建模用户行为，结合CNN与LSTM提升预测效果。**

- **链接: [http://arxiv.org/pdf/2506.17325v1](http://arxiv.org/pdf/2506.17325v1)**

> **作者:** Sina Najafi; M. Hadi Sepanj; Fahimeh Jafari
>
> **摘要:** Predicting user churn in non-subscription gig platforms, where disengagement is implicit, poses unique challenges due to the absence of explicit labels and the dynamic nature of user behavior. Existing methods often rely on aggregated snapshots or static visual representations, which obscure temporal cues critical for early detection. In this work, we propose a temporally-aware computer vision framework that models user behavioral patterns as a sequence of radar chart images, each encoding day-level behavioral features. By integrating a pretrained CNN encoder with a bidirectional LSTM, our architecture captures both spatial and temporal patterns underlying churn behavior. Extensive experiments on a large real-world dataset demonstrate that our method outperforms classical models and ViT-based radar chart baselines, yielding gains of 17.7 in F1 score, 29.4 in precision, and 16.1 in AUC, along with improved interpretability. The framework's modular design, explainability tools, and efficient deployment characteristics make it suitable for large-scale churn modeling in dynamic gig-economy platforms.
>
---
#### [new 018] MARL-MambaContour: Unleashing Multi-Agent Deep Reinforcement Learning for Active Contour Optimization in Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决传统方法在拓扑约束和结构感知上的不足。通过多智能体强化学习优化轮廓，提升分割精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.18679v1](http://arxiv.org/pdf/2506.18679v1)**

> **作者:** Ruicheng Zhang; Yu Sun; Zeyu Zhang; Jinai Li; Xiaofan Liu; Au Hoi Fan; Haowei Guo; Puxin Yan
>
> **摘要:** We introduce MARL-MambaContour, the first contour-based medical image segmentation framework based on Multi-Agent Reinforcement Learning (MARL). Our approach reframes segmentation as a multi-agent cooperation task focused on generate topologically consistent object-level contours, addressing the limitations of traditional pixel-based methods which could lack topological constraints and holistic structural awareness of anatomical regions. Each contour point is modeled as an autonomous agent that iteratively adjusts its position to align precisely with the target boundary, enabling adaptation to blurred edges and intricate morphologies common in medical images. This iterative adjustment process is optimized by a contour-specific Soft Actor-Critic (SAC) algorithm, further enhanced with the Entropy Regularization Adjustment Mechanism (ERAM) which dynamically balance agent exploration with contour smoothness. Furthermore, the framework incorporates a Mamba-based policy network featuring a novel Bidirectional Cross-attention Hidden-state Fusion Mechanism (BCHFM). This mechanism mitigates potential memory confusion limitations associated with long-range modeling in state space models, thereby facilitating more accurate inter-agent information exchange and informed decision-making. Extensive experiments on five diverse medical imaging datasets demonstrate the state-of-the-art performance of MARL-MambaContour, highlighting its potential as an accurate and robust clinical application.
>
---
#### [new 019] Mechanistic Interpretability of Diffusion Models: Circuit-Level Analysis and Causal Validation
- **分类: cs.CV**

- **简介: 该论文属于机器学习解释性研究，旨在解析扩散模型的内部机制。通过分析计算路径和注意力机制，揭示其在处理不同数据时的差异与复杂性。**

- **链接: [http://arxiv.org/pdf/2506.17237v1](http://arxiv.org/pdf/2506.17237v1)**

> **作者:** Dip Roy
>
> **摘要:** We present a quantitative circuit-level analysis of diffusion models, establishing computational pathways and mechanistic principles underlying image generation processes. Through systematic intervention experiments across 2,000 synthetic and 2,000 CelebA facial images, we discover fundamental algorithmic differences in how diffusion architectures process synthetic versus naturalistic data distributions. Our investigation reveals that real-world face processing requires circuits with measurably higher computational complexity (complexity ratio = 1.084 plus/minus 0.008, p < 0.001), exhibiting distinct attention specialization patterns with entropy divergence ranging from 0.015 to 0.166 across denoising timesteps. We identify eight functionally distinct attention mechanisms showing specialized computational roles: edge detection (entropy = 3.18 plus/minus 0.12), texture analysis (entropy = 4.16 plus/minus 0.08), and semantic understanding (entropy = 2.67 plus/minus 0.15). Intervention analysis demonstrates critical computational bottlenecks where targeted ablations produce 25.6% to 128.3% performance degradation, providing causal evidence for identified circuit functions. These findings establish quantitative foundations for algorithmic understanding and control of generative model behavior through mechanistic intervention strategies.
>
---
#### [new 020] SELFI: Selective Fusion of Identity for Generalizable Deepfake Detection
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决模型泛化能力不足的问题。通过引入SELFI框架，动态融合身份特征与视觉特征，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.17592v1](http://arxiv.org/pdf/2506.17592v1)**

> **作者:** Younghun Kim; Minsuk Jang; Myung-Joon Kwon; Wonjun Lee; Changick Kim
>
> **摘要:** Face identity provides a powerful signal for deepfake detection. Prior studies show that even when not explicitly modeled, classifiers often learn identity features implicitly. This has led to conflicting views: some suppress identity cues to reduce bias, while others rely on them as forensic evidence. To reconcile these views, we analyze two hypotheses: (1) whether face identity alone is discriminative for detecting deepfakes, and (2) whether such identity features generalize poorly across manipulation methods. Our experiments confirm that identity is informative but context-dependent. While some manipulations preserve identity-consistent artifacts, others distort identity cues and harm generalization. We argue that identity features should neither be blindly suppressed nor relied upon, but instead be explicitly modeled and adaptively controlled based on per-sample relevance. We propose \textbf{SELFI} (\textbf{SEL}ective \textbf{F}usion of \textbf{I}dentity), a generalizable detection framework that dynamically modulates identity usage. SELFI consists of: (1) a Forgery-Aware Identity Adapter (FAIA) that extracts identity embeddings from a frozen face recognition model and projects them into a forgery-relevant space via auxiliary supervision; and (2) an Identity-Aware Fusion Module (IAFM) that selectively integrates identity and visual features using a relevance-guided fusion mechanism. Experiments on four benchmarks show that SELFI improves cross-manipulation generalization, outperforming prior methods by an average of 3.1\% AUC. On the challenging DFDC dataset, SELFI exceeds the previous best by 6\%. Code will be released upon paper acceptance.
>
---
#### [new 021] Multimodal Fusion SLAM with Fourier Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉SLAM任务，旨在解决噪声、光照变化和黑暗环境下的定位与建图问题。提出FMF-SLAM方法，结合多模态数据与傅里叶注意力机制提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.18204v1](http://arxiv.org/pdf/2506.18204v1)**

> **作者:** Youjie Zhou; Guofeng Mei; Yiming Wang; Yi Wan; Fabio Poiesi
>
> **摘要:** Visual SLAM is particularly challenging in environments affected by noise, varying lighting conditions, and darkness. Learning-based optical flow algorithms can leverage multiple modalities to address these challenges, but traditional optical flow-based visual SLAM approaches often require significant computational resources.To overcome this limitation, we propose FMF-SLAM, an efficient multimodal fusion SLAM method that utilizes fast Fourier transform (FFT) to enhance the algorithm efficiency. Specifically, we introduce a novel Fourier-based self-attention and cross-attention mechanism to extract features from RGB and depth signals. We further enhance the interaction of multimodal features by incorporating multi-scale knowledge distillation across modalities. We also demonstrate the practical feasibility of FMF-SLAM in real-world scenarios with real time performance by integrating it with a security robot by fusing with a global positioning module GNSS-RTK and global Bundle Adjustment. Our approach is validated using video sequences from TUM, TartanAir, and our real-world datasets, showcasing state-of-the-art performance under noisy, varying lighting, and dark conditions.Our code and datasets are available at https://github.com/youjie-zhou/FMF-SLAM.git.
>
---
#### [new 022] PlanMoGPT: Flow-Enhanced Progressive Planning for Text to Motion Synthesis
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于文本到动作生成任务，解决LLM在该任务中表现不佳的问题。通过引入渐进式规划和增强的运动分词方法，提升生成质量与多样性。**

- **链接: [http://arxiv.org/pdf/2506.17912v1](http://arxiv.org/pdf/2506.17912v1)**

> **作者:** Chuhao Jin; Haosen Li; Bingzi Zhang; Che Liu; Xiting Wang; Ruihua Song; Wenbing Huang; Ying Qin; Fuzheng Zhang; Di Zhang
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Recent advances in large language models (LLMs) have enabled breakthroughs in many multimodal generation tasks, but a significant performance gap still exists in text-to-motion generation, where LLM-based methods lag far behind non-LLM methods. We identify the granularity of motion tokenization as a critical bottleneck: fine-grained tokenization induces local dependency issues, where LLMs overemphasize short-term coherence at the expense of global semantic alignment, while coarse-grained tokenization sacrifices motion details. To resolve this issue, we propose PlanMoGPT, an LLM-based framework integrating progressive planning and flow-enhanced fine-grained motion tokenization. First, our progressive planning mechanism leverages LLMs' autoregressive capabilities to hierarchically generate motion tokens by starting from sparse global plans and iteratively refining them into full sequences. Second, our flow-enhanced tokenizer doubles the downsampling resolution and expands the codebook size by eight times, minimizing detail loss during discretization, while a flow-enhanced decoder recovers motion nuances. Extensive experiments on text-to-motion benchmarks demonstrate that it achieves state-of-the-art performance, improving FID scores by 63.8% (from 0.380 to 0.141) on long-sequence generation while enhancing motion diversity by 49.9% compared to existing methods. The proposed framework successfully resolves the diversity-quality trade-off that plagues current non-LLM approaches, establishing new standards for text-to-motion generation.
>
---
#### [new 023] Fetuses Made Simple: Modeling and Tracking of Fetal Shape and Pose
- **分类: cs.CV**

- **简介: 该论文属于胎儿形态与姿态分析任务，旨在解决传统方法在捕捉完整形状和运动时的不足。通过构建3D统计胎儿模型，实现更准确的胎儿运动与形态分析。**

- **链接: [http://arxiv.org/pdf/2506.17858v1](http://arxiv.org/pdf/2506.17858v1)**

> **作者:** Yingcheng Liu; Peiqi Wang; Sebastian Diaz; Esra Abaci Turk; Benjamin Billot; Patricia Ellen Grant; Polina Golland
>
> **摘要:** Analyzing fetal body motion and shape is paramount in prenatal diagnostics and monitoring. Existing methods for fetal MRI analysis mainly rely on anatomical keypoints or volumetric body segmentations. Keypoints simplify body structure to facilitate motion analysis, but may ignore important details of full-body shape. Body segmentations capture complete shape information but complicate temporal analysis due to large non-local fetal movements. To address these limitations, we construct a 3D articulated statistical fetal body model based on the Skinned Multi-Person Linear Model (SMPL). Our algorithm iteratively estimates body pose in the image space and body shape in the canonical pose space. This approach improves robustness to MRI motion artifacts and intensity distortions, and reduces the impact of incomplete surface observations due to challenging fetal poses. We train our model on segmentations and keypoints derived from $19,816$ MRI volumes across $53$ subjects. Our model captures body shape and motion across time series and provides intuitive visualization. Furthermore, it enables automated anthropometric measurements traditionally difficult to obtain from segmentations and keypoints. When tested on unseen fetal body shapes, our method yields a surface alignment error of $3.2$ mm for $3$ mm MRI voxel size. To our knowledge, this represents the first 3D articulated statistical fetal body model, paving the way for enhanced fetal motion and shape analysis in prenatal diagnostics. The code is available at https://github.com/MedicalVisionGroup/fetal-smpl .
>
---
#### [new 024] STACT-Time: Spatio-Temporal Cross Attention for Cine Thyroid Ultrasound Time Series Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在提高甲状腺结节良恶性判断的准确性，减少不必要的活检。通过引入时空交叉注意力机制，提升模型对超声视频序列的特征学习能力。**

- **链接: [http://arxiv.org/pdf/2506.18172v1](http://arxiv.org/pdf/2506.18172v1)**

> **作者:** Irsyad Adam; Tengyue Zhang; Shrayes Raman; Zhuyu Qiu; Brandon Taraku; Hexiang Feng; Sile Wang; Ashwath Radhachandran; Shreeram Athreya; Vedrana Ivezic; Peipei Ping; Corey Arnold; William Speier
>
> **摘要:** Thyroid cancer is among the most common cancers in the United States. Thyroid nodules are frequently detected through ultrasound (US) imaging, and some require further evaluation via fine-needle aspiration (FNA) biopsy. Despite its effectiveness, FNA often leads to unnecessary biopsies of benign nodules, causing patient discomfort and anxiety. To address this, the American College of Radiology Thyroid Imaging Reporting and Data System (TI-RADS) has been developed to reduce benign biopsies. However, such systems are limited by interobserver variability. Recent deep learning approaches have sought to improve risk stratification, but they often fail to utilize the rich temporal and spatial context provided by US cine clips, which contain dynamic global information and surrounding structural changes across various views. In this work, we propose the Spatio-Temporal Cross Attention for Cine Thyroid Ultrasound Time Series Classification (STACT-Time) model, a novel representation learning framework that integrates imaging features from US cine clips with features from segmentation masks automatically generated by a pretrained model. By leveraging self-attention and cross-attention mechanisms, our model captures the rich temporal and spatial context of US cine clips while enhancing feature representation through segmentation-guided learning. Our model improves malignancy prediction compared to state-of-the-art models, achieving a cross-validation precision of 0.91 (plus or minus 0.02) and an F1 score of 0.89 (plus or minus 0.02). By reducing unnecessary biopsies of benign nodules while maintaining high sensitivity for malignancy detection, our model has the potential to enhance clinical decision-making and improve patient outcomes.
>
---
#### [new 025] OpenMAP-BrainAge: Generalizable and Interpretable Brain Age Predictor
- **分类: cs.CV**

- **简介: 该论文属于脑年龄预测任务，旨在解决模型可解释性与泛化能力问题。通过Transformer架构融合多视角和体积信息，提升预测精度与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.17597v1](http://arxiv.org/pdf/2506.17597v1)**

> **作者:** Pengyu Kan; Craig Jones; Kenichi Oishi
>
> **摘要:** Purpose: To develop an age prediction model which is interpretable and robust to demographic and technological variances in brain MRI scans. Materials and Methods: We propose a transformer-based architecture that leverages self-supervised pre-training on large-scale datasets. Our model processes pseudo-3D T1-weighted MRI scans from three anatomical views and incorporates brain volumetric information. By introducing a stem architecture, we reduce the conventional quadratic complexity of transformer models to linear complexity, enabling scalability for high-dimensional MRI data. We trained our model on ADNI2 $\&$ 3 (N=1348) and OASIS3 (N=716) datasets (age range: 42 - 95) from the North America, with an 8:1:1 split for train, validation and test. Then, we validated it on the AIBL dataset (N=768, age range: 60 - 92) from Australia. Results: We achieved an MAE of 3.65 years on ADNI2 $\&$ 3 and OASIS3 test set and a high generalizability of MAE of 3.54 years on AIBL. There was a notable increase in brain age gap (BAG) across cognitive groups, with mean of 0.15 years (95% CI: [-0.22, 0.51]) in CN, 2.55 years ([2.40, 2.70]) in MCI, 6.12 years ([5.82, 6.43]) in AD. Additionally, significant negative correlation between BAG and cognitive scores was observed, with correlation coefficient of -0.185 (p < 0.001) for MoCA and -0.231 (p < 0.001) for MMSE. Gradient-based feature attribution highlighted ventricles and white matter structures as key regions influenced by brain aging. Conclusion: Our model effectively fused information from different views and volumetric information to achieve state-of-the-art brain age prediction accuracy, improved generalizability and interpretability with association to neurodegenerative disorders.
>
---
#### [new 026] Selective Social-Interaction via Individual Importance for Fast Human Trajectory Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于轨迹预测任务，旨在快速准确预测行人轨迹。通过重要性估计模块选择关键邻近人，提升预测效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.18291v1](http://arxiv.org/pdf/2506.18291v1)**

> **作者:** Yota Urano; Hiromu Taketsugu; Norimichi Ukita
>
> **备注:** MIRU 2025
>
> **摘要:** This paper presents an architecture for selecting important neighboring people to predict the primary person's trajectory. To achieve effective neighboring people selection, we propose a people selection module called the Importance Estimator which outputs the importance of each neighboring person for predicting the primary person's future trajectory. To prevent gradients from being blocked by non-differentiable operations when sampling surrounding people based on their importance, we employ the Gumbel Softmax for training. Experiments conducted on the JRDB dataset show that our method speeds up the process with competitive prediction accuracy.
>
---
#### [new 027] SWA-SOP: Spatially-aware Window Attention for Semantic Occupancy Prediction in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶中的语义占用预测任务，解决传感器在遮挡和稀疏数据下的感知不足问题。提出SWA机制，增强空间结构建模，提升场景补全效果。**

- **链接: [http://arxiv.org/pdf/2506.18785v1](http://arxiv.org/pdf/2506.18785v1)**

> **作者:** Helin Cao; Rafael Materla; Sven Behnke
>
> **备注:** under reviewed
>
> **摘要:** Perception systems in autonomous driving rely on sensors such as LiDAR and cameras to perceive the 3D environment. However, due to occlusions and data sparsity, these sensors often fail to capture complete information. Semantic Occupancy Prediction (SOP) addresses this challenge by inferring both occupancy and semantics of unobserved regions. Existing transformer-based SOP methods lack explicit modeling of spatial structure in attention computation, resulting in limited geometric awareness and poor performance in sparse or occluded areas. To this end, we propose Spatially-aware Window Attention (SWA), a novel mechanism that incorporates local spatial context into attention. SWA significantly improves scene completion and achieves state-of-the-art results on LiDAR-based SOP benchmarks. We further validate its generality by integrating SWA into a camera-based SOP pipeline, where it also yields consistent gains across modalities.
>
---
#### [new 028] CLGRPO: Reasoning Ability Enhancement for Small VLMs
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在提升小规模模型的推理能力。通过构造COT数据并采用增量训练策略，显著提高了1B模型的性能。**

- **链接: [http://arxiv.org/pdf/2506.18048v1](http://arxiv.org/pdf/2506.18048v1)**

> **作者:** Fanyi Wang; Binzhi Dong; Haotian Hu; Jinjin Xu; Zhiwang Zhang
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Small Vision Language Models (SVLMs) generally refer to models with parameter sizes less than or equal to 2B. Their low cost and power consumption characteristics confer high commercial value. However, their reasoning abilities are limited by the number of parameters. To address this issue, this paper proposes a post-training optimization paradigm called the Incremental Training Strategy to enhance the reasoning ability of SVLMs. Firstly, we constructed a Self-Supervised Chain-of-Thought (COT) Data Construction System, which leverages multiple LVLMs with 7B parameters or more to transform original data into COT data in a self-supervised manner. Our proposed Incremental Training Strategy consists of four stages. Stage 1 injects domain knowledge by performing Supervised Fine-Tuning (SFT) to the pretrained model on the COT data. Stage 2 aligns the COT data format by conducting a small amount of Group Relative Policy Optimization (GRPO) training constrained only by format rewards on the COT data. Stage 3 enhances reasoning ability by applying GRPO training on the COT data with constraints on both format and accuracy rewards. The resulting model shows significant improvement compared to the baseline. Stage 4 addresses the limited capacity of the SVLMs and the weak ability to capture complex patterns by proposing ClipLow GRPO (CLGRPO) to constrain the capture space of the training process. We conducted extensive comparative and ablation experiments on the abstract semantic recognition dataset EMOSet-118K. Experimental results demonstrate that our method significantly improves the reasoning ability of 1B SVLM. Compared to the baseline model fine-tuned on the original data, accuracy increased by 2.77 and recall by 0.69, achieving performance comparable to that of 8B models.
>
---
#### [new 029] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态语言模型个性化任务，旨在解决模型生成个性化图像描述能力不足的问题。提出基于强化学习的后训练框架，提升模型的视觉识别与个性化生成能力。**

- **链接: [http://arxiv.org/pdf/2506.18369v1](http://arxiv.org/pdf/2506.18369v1)**

> **作者:** Yeongtak Oh; Jisoo Mok; Dohyun Chung; Juhyeon Shin; Sangha Park; Johan Barthelemy; Sungroh Yoon
>
> **备注:** Project Page: https://github.com/oyt9306/RePIC
>
> **摘要:** Recent multi-modal large language models (MLLMs) often struggle to generate personalized image captions, even when trained on high-quality captions. In this work, we observe that such limitations persist in existing post-training-based MLLM personalization methods. Specifically, despite being post-tuned with large-scale caption data through supervised fine-tuning (SFT), these models frequently fail to produce faithful descriptions in real-world scenarios, such as multi-concept image captioning. However, acquiring large-scale, high-quality captions for such complex settings is both costly and difficult. To address the data-centric nature of SFT, we propose a reinforcement learning (RL)-based post-training framework. To the best of our knowledge, this is the first RL-based approach to post-train MLLMs for personalized image captioning. Our method significantly enhances both visual recognition and personalized generation capabilities of MLLMs, and consistently outperforms existing SFT-based baselines, especially in the challenging multi-concept image captioning task.
>
---
#### [new 030] Histopathology Image Report Generation by Vision Language Model with Multimodal In-Context Learning
- **分类: cs.CV**

- **简介: 该论文属于医学图像报告生成任务，旨在解决从组织病理图像自动生成准确报告的问题。提出PathGenIC框架，结合多模态上下文学习提升生成质量。**

- **链接: [http://arxiv.org/pdf/2506.17645v1](http://arxiv.org/pdf/2506.17645v1)**

> **作者:** Shih-Wen Liu; Hsuan-Yu Fan; Wei-Ta Chu; Fu-En Yang; Yu-Chiang Frank Wang
>
> **备注:** Accepted to MIDL 2025
>
> **摘要:** Automating medical report generation from histopathology images is a critical challenge requiring effective visual representations and domain-specific knowledge. Inspired by the common practices of human experts, we propose an in-context learning framework called PathGenIC that integrates context derived from the training set with a multimodal in-context learning (ICL) mechanism. Our method dynamically retrieves semantically similar whole slide image (WSI)-report pairs and incorporates adaptive feedback to enhance contextual relevance and generation quality. Evaluated on the HistGen benchmark, the framework achieves state-of-the-art results, with significant improvements across BLEU, METEOR, and ROUGE-L metrics, and demonstrates robustness across diverse report lengths and disease categories. By maximizing training data utility and bridging vision and language with ICL, our work offers a solution for AI-driven histopathology reporting, setting a strong foundation for future advancements in multimodal clinical applications.
>
---
#### [new 031] OmniGen2: Exploration to Advanced Multimodal Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出OmniGen2，解决多模态生成任务，如文本到图像、图像编辑和上下文生成。通过双解码路径和反射机制提升性能。**

- **链接: [http://arxiv.org/pdf/2506.18871v1](http://arxiv.org/pdf/2506.18871v1)**

> **作者:** Chenyuan Wu; Pengfei Zheng; Ruiran Yan; Shitao Xiao; Xin Luo; Yueze Wang; Wanli Li; Xiyan Jiang; Yexin Liu; Junjie Zhou; Ze Liu; Ziyi Xia; Chaofan Li; Haoge Deng; Jiahao Wang; Kun Luo; Bo Zhang; Defu Lian; Xinlong Wang; Zhongyuan Wang; Tiejun Huang; Zheng Liu
>
> **摘要:** In this work, we introduce OmniGen2, a versatile and open-source generative model designed to provide a unified solution for diverse generation tasks, including text-to-image, image editing, and in-context generation. Unlike OmniGen v1, OmniGen2 features two distinct decoding pathways for text and image modalities, utilizing unshared parameters and a decoupled image tokenizer. This design enables OmniGen2 to build upon existing multimodal understanding models without the need to re-adapt VAE inputs, thereby preserving the original text generation capabilities. To facilitate the training of OmniGen2, we developed comprehensive data construction pipelines, encompassing image editing and in-context generation data. Additionally, we introduce a reflection mechanism tailored for image generation tasks and curate a dedicated reflection dataset based on OmniGen2. Despite its relatively modest parameter size, OmniGen2 achieves competitive results on multiple task benchmarks, including text-to-image and image editing. To further evaluate in-context generation, also referred to as subject-driven tasks, we introduce a new benchmark named OmniContext. OmniGen2 achieves state-of-the-art performance among open-source models in terms of consistency. We will release our models, training code, datasets, and data construction pipeline to support future research in this field. Project Page: https://vectorspacelab.github.io/OmniGen2; GitHub Link: https://github.com/VectorSpaceLab/OmniGen2
>
---
#### [new 032] Trans${^2}$-CBCT: A Dual-Transformer Framework for Sparse-View CBCT Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于CT图像重建任务，旨在解决稀疏视角CBCT的伪影和空间覆盖问题。通过引入双Transformer框架提升重建质量。**

- **链接: [http://arxiv.org/pdf/2506.17425v1](http://arxiv.org/pdf/2506.17425v1)**

> **作者:** Minmin Yang; Huantao Ren; Senem Velipasalar
>
> **摘要:** Cone-beam computed tomography (CBCT) using only a few X-ray projection views enables faster scans with lower radiation dose, but the resulting severe under-sampling causes strong artifacts and poor spatial coverage. We address these challenges in a unified framework. First, we replace conventional UNet/ResNet encoders with TransUNet, a hybrid CNN-Transformer model. Convolutional layers capture local details, while self-attention layers enhance global context. We adapt TransUNet to CBCT by combining multi-scale features, querying view-specific features per 3D point, and adding a lightweight attenuation-prediction head. This yields Trans-CBCT, which surpasses prior baselines by 1.17 dB PSNR and 0.0163 SSIM on the LUNA16 dataset with six views. Second, we introduce a neighbor-aware Point Transformer to enforce volumetric coherence. This module uses 3D positional encoding and attention over k-nearest neighbors to improve spatial consistency. The resulting model, Trans$^2$-CBCT, provides an additional gain of 0.63 dB PSNR and 0.0117 SSIM. Experiments on LUNA16 and ToothFairy show consistent gains from six to ten views, validating the effectiveness of combining CNN-Transformer features with point-based geometry reasoning for sparse-view CBCT reconstruction.
>
---
#### [new 033] Shape from Polarization of Thermal Emission and Reflection
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决透明物体形状估计难题。通过融合发射与反射的偏振模型，并利用神经网络和真实数据集提升精度。**

- **链接: [http://arxiv.org/pdf/2506.18217v1](http://arxiv.org/pdf/2506.18217v1)**

> **作者:** Kazuma Kitazawa; Tsuyoshi Takatani
>
> **备注:** ICCP2025
>
> **摘要:** Shape estimation for transparent objects is challenging due to their complex light transport. To circumvent these difficulties, we leverage the Shape from Polarization (SfP) technique in the Long-Wave Infrared (LWIR) spectrum, where most materials are opaque and emissive. While a few prior studies have explored LWIR SfP, these attempts suffered from significant errors due to inadequate polarimetric modeling, particularly the neglect of reflection. Addressing this gap, we formulated a polarization model that explicitly accounts for the combined effects of emission and reflection. Based on this model, we estimated surface normals using not only a direct model-based method but also a learning-based approach employing a neural network trained on a physically-grounded synthetic dataset. Furthermore, we modeled the LWIR polarimetric imaging process, accounting for inherent systematic errors to ensure accurate polarimetry. We implemented a prototype system and created ThermoPol, the first real-world benchmark dataset for LWIR SfP. Through comprehensive experiments, we demonstrated the high accuracy and broad applicability of our method across various materials, including those transparent in the visible spectrum.
>
---
#### [new 034] Including Semantic Information via Word Embeddings for Skeleton-based Action Recognition
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于动作识别任务，旨在解决骨架方法丢失关键点语义的问题。通过引入词嵌入增强语义信息，提升模型性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.18721v1](http://arxiv.org/pdf/2506.18721v1)**

> **作者:** Dustin Aganian; Erik Franze; Markus Eisenbach; Horst-Michael Gross
>
> **备注:** IEEE International Joint Conference on Neural Networks (IJCNN) 2025
>
> **摘要:** Effective human action recognition is widely used for cobots in Industry 4.0 to assist in assembly tasks. However, conventional skeleton-based methods often lose keypoint semantics, limiting their effectiveness in complex interactions. In this work, we introduce a novel approach to skeleton-based action recognition that enriches input representations by leveraging word embeddings to encode semantic information. Our method replaces one-hot encodings with semantic volumes, enabling the model to capture meaningful relationships between joints and objects. Through extensive experiments on multiple assembly datasets, we demonstrate that our approach significantly improves classification performance, and enhances generalization capabilities by simultaneously supporting different skeleton types and object classes. Our findings highlight the potential of incorporating semantic information to enhance skeleton-based action recognition in dynamic and diverse environments.
>
---
#### [new 035] Deep Supervised LSTM for 3D morphology estimation from Multi-View RGB Images of Wheat Spikes
- **分类: cs.CV**

- **简介: 该论文属于3D形态估计任务，旨在解决从多视角RGB图像中准确估算小麦穗体积的问题。通过结合DINOv2和LSTM的深度监督模型，提升体积预测精度。**

- **链接: [http://arxiv.org/pdf/2506.18060v1](http://arxiv.org/pdf/2506.18060v1)**

> **作者:** Olivia Zumsteg; Nico Graf; Aaron Haeusler; Norbert Kirchgessner; Nicola Storni; Lukas Roth; Andreas Hund
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** Estimating three-dimensional morphological traits from two-dimensional RGB images presents inherent challenges due to the loss of depth information, projection distortions, and occlusions under field conditions. In this work, we explore multiple approaches for non-destructive volume estimation of wheat spikes, using RGB image sequences and structured-light 3D scans as ground truth references. Due to the complex geometry of the spikes, we propose a neural network approach for volume estimation in 2D images, employing a transfer learning pipeline that combines DINOv2, a self-supervised Vision Transformer, with a unidirectional Long Short-Term Memory (LSTM) network. By using deep supervision, the model is able to learn more robust intermediate representations, which enhances its generalisation ability across varying evaluation sequences. We benchmark our model against two conventional baselines: a 2D area-based projection and a geometric reconstruction using axis-aligned cross-sections. Our deep supervised model achieves a mean absolute percentage error (MAPE) of 6.46% on six-view indoor images, outperforming the area (9.36%) and geometric (13.98%) baselines. Fine-tuning the model on field-based single-image data enables domain adaptation, yielding a MAPE of 10.82%. We demonstrate that object shape significantly impacts volume prediction accuracy, with irregular geometries such as wheat spikes posing greater challenges for geometric methods compared to our deep learning approach.
>
---
#### [new 036] From Drawings to Decisions: A Hybrid Vision-Language Framework for Parsing 2D Engineering Drawings into Structured Manufacturing Knowledge
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文属于2D工程图解析任务，旨在解决手动提取信息效率低、OCR模型不准确的问题。通过融合视觉与语言模型，实现结构化制造知识的高效提取。**

- **链接: [http://arxiv.org/pdf/2506.17374v1](http://arxiv.org/pdf/2506.17374v1)**

> **作者:** Muhammad Tayyab Khan; Lequn Chen; Zane Yong; Jun Ming Tan; Wenhe Feng; Seung Ki Moon
>
> **备注:** Preprint submitted to Elsevier
>
> **摘要:** Efficient and accurate extraction of key information from 2D engineering drawings is essential for advancing digital manufacturing workflows. Such information includes geometric dimensioning and tolerancing (GD&T), measures, material specifications, and textual annotations. Manual extraction is slow and labor-intensive, while generic OCR models often fail due to complex layouts, engineering symbols, and rotated text, leading to incomplete and unreliable outputs. These limitations result in incomplete and unreliable outputs. To address these challenges, we propose a hybrid vision-language framework that integrates a rotation-aware object detection model (YOLOv11-obb) with a transformer-based vision-language parser. Our structured pipeline applies YOLOv11-OBB to localize annotations and extract oriented bounding box (OBB) patches, which are then parsed into structured outputs using a fine-tuned, lightweight vision-language model (VLM). We curate a dataset of 1,367 2D mechanical drawings annotated across nine key categories. YOLOv11-OBB is trained on this dataset to detect OBBs and extract annotation patches. These are parsed using two open-source VLMs: Donut and Florence-2. Both models are lightweight and well-suited for specialized industrial tasks under limited computational overhead. Following fine-tuning of both models on the curated dataset of image patches paired with structured annotation labels, a comparative experiment is conducted to evaluate parsing performance across four key metrics. Donut outperforms Florence-2, achieving 88.5% precision, 99.2% recall, and a 93.5% F1-score, with a hallucination rate of 11.5%. Finally, a case study demonstrates how the extracted structured information supports downstream manufacturing tasks such as process and tool selection, showcasing the practical utility of the proposed framework in modernizing 2D drawing interpretation.
>
---
#### [new 037] Sequential keypoint density estimator: an overlooked baseline of skeleton-based video anomaly detection
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在通过人体骨骼关键点检测异常行为。提出SeeKer方法，基于关键点密度估计判断异常。**

- **链接: [http://arxiv.org/pdf/2506.18368v1](http://arxiv.org/pdf/2506.18368v1)**

> **作者:** Anja Delić; Matej Grcić; Siniša Šegvić
>
> **摘要:** Detecting anomalous human behaviour is an important visual task in safety-critical applications such as healthcare monitoring, workplace safety, or public surveillance. In these contexts, abnormalities are often reflected with unusual human poses. Thus, we propose SeeKer, a method for detecting anomalies in sequences of human skeletons. Our method formulates the skeleton sequence density through autoregressive factorization at the keypoint level. The corresponding conditional distributions represent probable keypoint locations given prior skeletal motion. We formulate the joint distribution of the considered skeleton as causal prediction of conditional Gaussians across its constituent keypoints. A skeleton is flagged as anomalous if its keypoint locations surprise our model (i.e. receive a low density). In practice, our anomaly score is a weighted sum of per-keypoint log-conditionals, where the weights account for the confidence of the underlying keypoint detector. Despite its conceptual simplicity, SeeKer surpasses all previous methods on the UBnormal and MSAD-HR datasets while delivering competitive performance on the ShanghaiTech dataset.
>
---
#### [new 038] SegChange-R1:Augmented Reasoning for Remote Sensing Change Detection via Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于遥感变化检测任务，旨在提升模型对变化区域的识别能力。通过融合文本信息和设计空间变换模块，提高了检测精度与效率。**

- **链接: [http://arxiv.org/pdf/2506.17944v1](http://arxiv.org/pdf/2506.17944v1)**

> **作者:** Fei Zhou
>
> **摘要:** Remote sensing change detection is widely used in a variety of fields such as urban planning, terrain and geomorphology analysis, and environmental monitoring, mainly by analyzing the significant change differences of features (e.g., building changes) in the same spatial region at different time phases. In this paper, we propose a large language model (LLM) augmented inference approach (SegChange-R1), which enhances the detection capability by integrating textual descriptive information and aims at guiding the model to segment the more interested change regions, thus accelerating the convergence speed. Moreover, we design a spatial transformation module (BEV) based on linear attention, which solves the problem of modal misalignment in change detection by unifying features from different temporal perspectives onto the BEV space. In addition, we construct the first dataset for building change detection from UAV viewpoints (DVCD ), and our experiments on four widely-used change detection datasets show a significant improvement over existing methods. The code and pre-trained models are available in https://github.com/Yu-Zhouz/SegChange-R1.
>
---
#### [new 039] TC-Light: Temporally Consistent Relighting for Dynamic Long Videos
- **分类: cs.CV**

- **简介: 该论文属于视频重光照任务，解决长视频动态场景下光照编辑的时序一致性和计算效率问题。提出TC-Light方法，通过两阶段优化实现高质量重光照。**

- **链接: [http://arxiv.org/pdf/2506.18904v1](http://arxiv.org/pdf/2506.18904v1)**

> **作者:** Yang Liu; Chuanchen Luo; Zimo Tang; Yingyan Li; Yuran Yang; Yuanyong Ning; Lue Fan; Junran Peng; Zhaoxiang Zhang
>
> **备注:** Project Page: https://dekuliutesla.github.io/tclight/ Code: https://github.com/Linketic/TC-Light
>
> **摘要:** Editing illumination in long videos with complex dynamics has significant value in various downstream tasks, including visual content creation and manipulation, as well as data scaling up for embodied AI through sim2real and real2real transfer. Nevertheless, existing video relighting techniques are predominantly limited to portrait videos or fall into the bottleneck of temporal consistency and computation efficiency. In this paper, we propose TC-Light, a novel paradigm characterized by the proposed two-stage post optimization mechanism. Starting from the video preliminarily relighted by an inflated video relighting model, it optimizes appearance embedding in the first stage to align global illumination. Then it optimizes the proposed canonical video representation, i.e., Unique Video Tensor (UVT), to align fine-grained texture and lighting in the second stage. To comprehensively evaluate performance, we also establish a long and highly dynamic video benchmark. Extensive experiments show that our method enables physically plausible relighting results with superior temporal coherence and low computation cost. The code and video demos are available at https://dekuliutesla.github.io/tclight/.
>
---
#### [new 040] Time-Contrastive Pretraining for In-Context Image and Video Segmentation
- **分类: cs.CV**

- **简介: 该论文属于图像和视频分割任务，解决传统方法在上下文灵活性和分辨率上的不足。通过时间对比预训练，提升分割效果。**

- **链接: [http://arxiv.org/pdf/2506.17837v1](http://arxiv.org/pdf/2506.17837v1)**

> **作者:** Assefa Wahd; Jacob Jaremko; Abhilash Hareendranathan
>
> **摘要:** In-context learning (ICL) enables generalization to new tasks with minimal labeled data. However, mainstream ICL approaches rely on a gridding strategy, which lacks the flexibility required for vision applications. We introduce Temporal, a time-contrastive self-supervised objective that pretrains a prompt retriever for visual ICL, and formulate ICL as a video object segmentation (VOS) task. Temporal addresses key limitations of grid-based methods that restrict the number and resolution of context images. By reframing ICL as a VOS problem, our approach supports a variable number of context images while preserving their full resolution. To address the challenge of selecting optimal context sets for queries, we pretrain a prompt retriever on videos via self-supervised learning, where adjacent frames serve as positives and distant frames as negatives. For image segmentation, the prompt retriever selects relevant sequences that, when combined with the query, form coherent videos for VOS processing. For video segmentation, it identifies keyframes, predicts their masks using our ICL pipeline, and propagates them throughout the sequence. When evaluated on MICCAI FLARE 2022, our method achieves substantial improvements over baselines: 90.95% Dice score for image segmentation (10.64% improvement) and 92.45% Dice for video segmentation (14.88% improvement).
>
---
#### [new 041] NSFW-Classifier Guided Prompt Sanitization for Safe Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本生成安全任务，旨在解决T2I模型生成有害内容的问题。通过PromptSan方法，有效净化提示词，提升生成安全性。**

- **链接: [http://arxiv.org/pdf/2506.18325v1](http://arxiv.org/pdf/2506.18325v1)**

> **作者:** Yu Xie; Chengjie Zeng; Lingyun Zhang; Yanwei Fu
>
> **摘要:** The rapid advancement of text-to-image (T2I) models, such as Stable Diffusion, has enhanced their capability to synthesize images from textual prompts. However, this progress also raises significant risks of misuse, including the generation of harmful content (e.g., pornography, violence, discrimination), which contradicts the ethical goals of T2I technology and hinders its sustainable development. Inspired by "jailbreak" attacks in large language models, which bypass restrictions through subtle prompt modifications, this paper proposes NSFW-Classifier Guided Prompt Sanitization (PromptSan), a novel approach to detoxify harmful prompts without altering model architecture or degrading generation capability. PromptSan includes two variants: PromptSan-Modify, which iteratively identifies and replaces harmful tokens in input prompts using text NSFW classifiers during inference, and PromptSan-Suffix, which trains an optimized suffix token sequence to neutralize harmful intent while passing both text and image NSFW classifier checks. Extensive experiments demonstrate that PromptSan achieves state-of-the-art performance in reducing harmful content generation across multiple metrics, effectively balancing safety and usability.
>
---
#### [new 042] CDG-MAE: Learning Correspondences from Diffusion Generated Views
- **分类: cs.CV**

- **简介: 该论文属于自监督学习任务，旨在解决视频标签传播中密集对应关系学习的问题。通过生成多样化的合成视图，提升预训练效果。**

- **链接: [http://arxiv.org/pdf/2506.18164v1](http://arxiv.org/pdf/2506.18164v1)**

> **作者:** Varun Belagali; Pierre Marza; Srikar Yellapragada; Zilinghan Li; Tarak Nath Nandi; Ravi K Madduri; Joel Saltz; Stergios Christodoulidis; Maria Vakalopoulou; Dimitris Samaras
>
> **摘要:** Learning dense correspondences, critical for application such as video label propagation, is hindered by tedious and unscalable manual annotation. Self-supervised methods address this by using a cross-view pretext task, often modeled with a masked autoencoder, where a masked target view is reconstructed from an anchor view. However, acquiring effective training data remains a challenge - collecting diverse video datasets is difficult and costly, while simple image crops lack necessary pose variations. This paper introduces CDG-MAE, a novel MAE-based self-supervised method that uses diverse synthetic views generated from static images via an image-conditioned diffusion model. These generated views exhibit substantial changes in pose and perspective, providing a rich training signal that overcomes the limitations of video and crop-based anchors. We present a quantitative method to evaluate local and global consistency of generated images, discussing their use for cross-view self-supervised pretraining. Furthermore, we enhance the standard single-anchor MAE setting to a multi-anchor strategy to effectively modulate the difficulty of pretext task. CDG-MAE significantly outperforms state-of-the-art MAE methods reliant only on images and substantially narrows the performance gap to video-based approaches.
>
---
#### [new 043] When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的异常检测任务，旨在解决实时性与准确性兼备的问题。提出一种多模态异步混合网络，融合事件相机与RGB图像数据，实现毫秒级精准检测。**

- **链接: [http://arxiv.org/pdf/2506.17457v1](http://arxiv.org/pdf/2506.17457v1)**

> **作者:** Dong Xiao; Guangyao Chen; Peixi Peng; Yangru Huang; Yifan Zhao; Yongxing Dai; Yonghong Tian
>
> **备注:** ICML 2025 Spotlight
>
> **摘要:** Anomaly detection is essential for the safety and reliability of autonomous driving systems. Current methods often focus on detection accuracy but neglect response time, which is critical in time-sensitive driving scenarios. In this paper, we introduce real-time anomaly detection for autonomous driving, prioritizing both minimal response time and high accuracy. We propose a novel multimodal asynchronous hybrid network that combines event streams from event cameras with image data from RGB cameras. Our network utilizes the high temporal resolution of event cameras through an asynchronous Graph Neural Network and integrates it with spatial features extracted by a CNN from RGB images. This combination effectively captures both the temporal dynamics and spatial details of the driving environment, enabling swift and precise anomaly detection. Extensive experiments on benchmark datasets show that our approach outperforms existing methods in both accuracy and response time, achieving millisecond-level real-time performance.
>
---
#### [new 044] Audit & Repair: An Agentic Framework for Consistent Story Visualization in Text-to-Image Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于故事可视化任务，旨在解决多面板图像中角色和物体视觉不一致的问题。提出一种多智能体框架，自动检测并修复不一致性，提升整体连贯性。**

- **链接: [http://arxiv.org/pdf/2506.18900v1](http://arxiv.org/pdf/2506.18900v1)**

> **作者:** Kiymet Akdemir; Tahira Kazimi; Pinar Yanardag
>
> **备注:** Project webpage: https://auditandrepair.github.io/
>
> **摘要:** Story visualization has become a popular task where visual scenes are generated to depict a narrative across multiple panels. A central challenge in this setting is maintaining visual consistency, particularly in how characters and objects persist and evolve throughout the story. Despite recent advances in diffusion models, current approaches often fail to preserve key character attributes, leading to incoherent narratives. In this work, we propose a collaborative multi-agent framework that autonomously identifies, corrects, and refines inconsistencies across multi-panel story visualizations. The agents operate in an iterative loop, enabling fine-grained, panel-level updates without re-generating entire sequences. Our framework is model-agnostic and flexibly integrates with a variety of diffusion models, including rectified flow transformers such as Flux and latent diffusion models such as Stable Diffusion. Quantitative and qualitative experiments show that our method outperforms prior approaches in terms of multi-panel consistency.
>
---
#### [new 045] Focus Your Attention: Towards Data-Intuitive Lightweight Vision Transformers
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉Transformer优化任务，旨在解决计算复杂度高和迁移学习困难的问题。提出SPPP和LLA模块以提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.18791v1](http://arxiv.org/pdf/2506.18791v1)**

> **作者:** Suyash Gaurav; Muhammad Farhan Humayun; Jukka Heikkonen; Jatin Chaudhary
>
> **摘要:** The evolution of Vision Transformers has led to their widespread adaptation to different domains. Despite large-scale success, there remain significant challenges including their reliance on extensive computational and memory resources for pre-training on huge datasets as well as difficulties in task-specific transfer learning. These limitations coupled with energy inefficiencies mainly arise due to the computation-intensive self-attention mechanism. To address these issues, we propose a novel Super-Pixel Based Patch Pooling (SPPP) technique that generates context-aware, semantically rich, patch embeddings to effectively reduce the architectural complexity and improve efficiency. Additionally, we introduce the Light Latent Attention (LLA) module in our pipeline by integrating latent tokens into the attention mechanism allowing cross-attention operations to significantly reduce the time and space complexity of the attention module. By leveraging the data-intuitive patch embeddings coupled with dynamic positional encodings, our approach adaptively modulates the cross-attention process to focus on informative regions while maintaining the global semantic structure. This targeted attention improves training efficiency and accelerates convergence. Notably, the SPPP module is lightweight and can be easily integrated into existing transformer architectures. Extensive experiments demonstrate that our proposed architecture provides significant improvements in terms of computational efficiency while achieving comparable results with the state-of-the-art approaches, highlighting its potential for energy-efficient transformers suitable for edge deployment. (The code is available on our GitHub repository: https://github.com/zser092/Focused-Attention-ViT).
>
---
#### [new 046] Deep CNN Face Matchers Inherently Support Revocable Biometric Templates
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于生物特征认证任务，旨在解决生物特征模板被泄露后无法撤销的问题。通过深度CNN模型生成可撤销的生物特征模板，实现安全的身份验证。**

- **链接: [http://arxiv.org/pdf/2506.18731v1](http://arxiv.org/pdf/2506.18731v1)**

> **作者:** Aman Bhatta; Michael C. King; Kevin W. Bowyer
>
> **摘要:** One common critique of biometric authentication is that if an individual's biometric is compromised, then the individual has no recourse. The concept of revocable biometrics was developed to address this concern. A biometric scheme is revocable if an individual can have their current enrollment in the scheme revoked, so that the compromised biometric template becomes worthless, and the individual can re-enroll with a new template that has similar recognition power. We show that modern deep CNN face matchers inherently allow for a robust revocable biometric scheme. For a given state-of-the-art deep CNN backbone and training set, it is possible to generate an unlimited number of distinct face matcher models that have both (1) equivalent recognition power, and (2) strongly incompatible biometric templates. The equivalent recognition power extends to the point of generating impostor and genuine distributions that have the same shape and placement on the similarity dimension, meaning that the models can share a similarity threshold for a 1-in-10,000 false match rate. The biometric templates from different model instances are so strongly incompatible that the cross-instance similarity score for images of the same person is typically lower than the same-instance similarity score for images of different persons. That is, a stolen biometric template that is revoked is of less value in attempting to match the re-enrolled identity than the average impostor template. We also explore the feasibility of using a Vision Transformer (ViT) backbone-based face matcher in the revocable biometric system proposed in this work and demonstrate that it is less suitable compared to typical ResNet-based deep CNN backbones.
>
---
#### [new 047] MUPA: Towards Multi-Path Agentic Reasoning for Grounded Video Question Answering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频问答任务，旨在解决视频语言理解中答案与视觉证据对齐不足的问题。提出MUPA方法，通过多路径协同推理提升 grounding 精度与答案准确性。**

- **链接: [http://arxiv.org/pdf/2506.18071v1](http://arxiv.org/pdf/2506.18071v1)**

> **作者:** Jisheng Dang; Huilin Song; Junbin Xiao; Bimei Wang; Han Peng; Haoxuan Li; Xun Yang; Meng Wang; Tat-Seng Chua
>
> **摘要:** Grounded Video Question Answering (Grounded VideoQA) requires aligning textual answers with explicit visual evidence. However, modern multimodal models often rely on linguistic priors and spurious correlations, resulting in poorly grounded predictions. In this work, we propose MUPA, a cooperative MUlti-Path Agentic approach that unifies video grounding, question answering, answer reflection and aggregation to tackle Grounded VideoQA. MUPA features three distinct reasoning paths on the interplay of grounding and QA agents in different chronological orders, along with a dedicated reflection agent to judge and aggregate the multi-path results to accomplish consistent QA and grounding. This design markedly improves grounding fidelity without sacrificing answer accuracy. Despite using only 2B parameters, our method outperforms all 7B-scale competitors. When scaled to 7B parameters, MUPA establishes new state-of-the-art results, with Acc@GQA of 30.3% and 47.4% on NExT-GQA and DeVE-QA respectively, demonstrating MUPA' effectiveness towards trustworthy video-language understanding. Our code is available in https://github.com/longmalongma/MUPA.
>
---
#### [new 048] Pattern-Based Phase-Separation of Tracer and Dispersed Phase Particles in Two-Phase Defocusing Particle Tracking Velocimetry
- **分类: cs.CV; physics.app-ph; physics.flu-dyn**

- **简介: 该论文属于两相流粒子追踪中的相分离任务，旨在解决传统方法在复杂场景下失效的问题。通过卷积神经网络和生成对抗网络实现高效准确的相分离。**

- **链接: [http://arxiv.org/pdf/2506.18157v1](http://arxiv.org/pdf/2506.18157v1)**

> **作者:** Christian Sax; Jochen Kriegseis
>
> **摘要:** This work investigates the feasibility of a post-processing-based approach for phase separation in defocusing particle tracking velocimetry for dispersed two-phase flows. The method enables the simultaneous 3D localization determination of both tracer particles and particles of the dispersed phase, using a single-camera setup. The distinction between phases is based on pattern differences in defocused particle images, which arise from distinct light scattering behaviors of tracer particles and bubbles or droplets. Convolutional neural networks, including Faster R-CNN and YOLOv4 variants, are trained to detect and classify particle images based on these pattern features. To generate large, labeled training datasets, a generative adversarial network based framework is introduced, allowing the generation of auto-labeled data that more closely reflects experiment-specific visual appearance. Evaluation across six datasets, comprising synthetic two-phase and real single- and two-phase flows, demonstrates high detection precision and classification accuracy (95-100%), even under domain shifts. The results confirm the viability of using CNNs for robust phase separation in disperse two-phase DPTV, particularly in scenarios where traditional wavelength-, size-, or ensemble correlation-based methods are impractical.
>
---
#### [new 049] OC-SOP: Enhancing Vision-Based 3D Semantic Occupancy Prediction by Object-Centric Awareness
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶中的语义占用预测任务，旨在解决遮挡和场景数据不全导致的预测不准问题。通过引入基于目标的感知线索，提升预测精度。**

- **链接: [http://arxiv.org/pdf/2506.18798v1](http://arxiv.org/pdf/2506.18798v1)**

> **作者:** Helin Cao; Sven Behnke
>
> **备注:** under review
>
> **摘要:** Autonomous driving perception faces significant challenges due to occlusions and incomplete scene data in the environment. To overcome these issues, the task of semantic occupancy prediction (SOP) is proposed, which aims to jointly infer both the geometry and semantic labels of a scene from images. However, conventional camera-based methods typically treat all categories equally and primarily rely on local features, leading to suboptimal predictions, especially for dynamic foreground objects. To address this, we propose Object-Centric SOP (OC-SOP), a framework that integrates high-level object-centric cues extracted via a detection branch into the semantic occupancy prediction pipeline. This object-centric integration significantly enhances the prediction accuracy for foreground objects and achieves state-of-the-art performance among all categories on SemanticKITTI.
>
---
#### [new 050] See-in-Pairs: Reference Image-Guided Comparative Vision-Language Models for Medical Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医学影像诊断任务，旨在解决单图分析不足与对比推理缺失的问题。通过引入参考图像和临床提示，提升VLM的诊断准确性。**

- **链接: [http://arxiv.org/pdf/2506.18140v1](http://arxiv.org/pdf/2506.18140v1)**

> **作者:** Ruinan Jin; Gexin Huang; Xinwei Shen; Qiong Zhang; Yan Shuo Tan; Xiaoxiao Li
>
> **备注:** 25 pages, four figures
>
> **摘要:** Medical imaging diagnosis presents inherent challenges due to diseases that mimic normal anatomy and exhibit significant inter-patient variability. Clinicians routinely employ comparative reasoning-using reference images from healthy controls or previous patient examinations-to discern subtle yet diagnostically critical abnormalities. However, existing medical vision-language models (VLMs) focus primarily on single-image or single-series analyses and lack explicit mechanisms for comparative reasoning. Conversely, general-purpose VLMs demonstrate strong multi-image comparative reasoning capabilities but lack essential medical-domain knowledge to identify nuanced clinical differences. This work aims to bridge this gap by exploring clinically-inspired comparative analysis within VLMs, leveraging reference images to enhance diagnostic accuracy. Through extensive empirical analysis, we show that providing general-purpose VLMs with query and normative matched reference images, accompanied by clinically-informed comparative prompts, significantly improves diagnostic outcomes compared to single-image baselines, especially after supervised finetuning (SFT). Our contributions highlight the clinical relevance of comparative analysis introduce novel strategies for leveraging reference images in VLMs, empirically demonstrate enhanced performance across multiple medical visual question answering (VQA) tasks, and provide theoretical insights into the efficacy of comparative image analysis in medical diagnosis.
>
---
#### [new 051] On the Robustness of Human-Object Interaction Detection against Distribution Shift
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于人-物交互检测任务，旨在解决模型在分布偏移下的鲁棒性问题。通过构建基准、分析模型并提出增强方法，提升模型在不同分布下的性能。**

- **链接: [http://arxiv.org/pdf/2506.18021v1](http://arxiv.org/pdf/2506.18021v1)**

> **作者:** Chi Xie; Shuang Liang; Jie Li; Feng Zhu; Rui Zhao; Yichen Wei; Shengjie Zhao
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Human-Object Interaction (HOI) detection has seen substantial advances in recent years. However, existing works focus on the standard setting with ideal images and natural distribution, far from practical scenarios with inevitable distribution shifts. This hampers the practical applicability of HOI detection. In this work, we investigate this issue by benchmarking, analyzing, and enhancing the robustness of HOI detection models under various distribution shifts. We start by proposing a novel automated approach to create the first robustness evaluation benchmark for HOI detection. Subsequently, we evaluate more than 40 existing HOI detection models on this benchmark, showing their insufficiency, analyzing the features of different frameworks, and discussing how the robustness in HOI is different from other tasks. With the insights from such analyses, we propose to improve the robustness of HOI detection methods through: (1) a cross-domain data augmentation integrated with mixup, and (2) a feature fusion strategy with frozen vision foundation models. Both are simple, plug-and-play, and applicable to various methods. Our experimental results demonstrate that the proposed approach significantly increases the robustness of various methods, with benefits on standard benchmarks, too. The dataset and code will be released.
>
---
#### [new 052] TAMMs: Temporal-Aware Multimodal Model for Satellite Image Change Understanding and Forecasting
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦卫星图像时序分析任务，解决多模态模型在时空推理上的不足。提出TAMMs模型，结合时间模块与语义融合机制，提升变化理解和未来图像生成能力。**

- **链接: [http://arxiv.org/pdf/2506.18862v1](http://arxiv.org/pdf/2506.18862v1)**

> **作者:** Zhongbin Guo; Yuhao Wang; Ping Jian; Xinyue Chen; Wei Peng; Ertai E
>
> **备注:** Submitted to the 33rd ACM International Conference on Multimedia. Our dataset can be found at https://huggingface.co/datasets/IceInPot/TAMMs
>
> **摘要:** Satellite image time-series analysis demands fine-grained spatial-temporal reasoning, which remains a challenge for existing multimodal large language models (MLLMs). In this work, we study the capabilities of MLLMs on a novel task that jointly targets temporal change understanding and future scene generation, aiming to assess their potential for modeling complex multimodal dynamics over time. We propose TAMMs, a Temporal-Aware Multimodal Model for satellite image change understanding and forecasting, which enhances frozen MLLMs with lightweight temporal modules for structured sequence encoding and contextual prompting. To guide future image generation, TAMMs introduces a Semantic-Fused Control Injection (SFCI) mechanism that adaptively combines high-level semantic reasoning and structural priors within an enhanced ControlNet. This dual-path conditioning enables temporally consistent and semantically grounded image synthesis. Experiments demonstrate that TAMMs outperforms strong MLLM baselines in both temporal change understanding and future image forecasting tasks, highlighting how carefully designed temporal reasoning and semantic fusion can unlock the full potential of MLLMs for spatio-temporal understanding.
>
---
#### [new 053] OSDMamba: Enhancing Oil Spill Detection from Remote Sensing Images Using Selective State Space Model
- **分类: cs.CV**

- **简介: 该论文属于油污检测任务，解决标签样本少和小目标检测难的问题，提出OSDMamba模型提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.18006v1](http://arxiv.org/pdf/2506.18006v1)**

> **作者:** Shuaiyu Chen; Fu Wang; Peng Ren; Chunbo Luo; Zeyu Fu
>
> **摘要:** Semantic segmentation is commonly used for Oil Spill Detection (OSD) in remote sensing images. However, the limited availability of labelled oil spill samples and class imbalance present significant challenges that can reduce detection accuracy. Furthermore, most existing methods, which rely on convolutional neural networks (CNNs), struggle to detect small oil spill areas due to their limited receptive fields and inability to effectively capture global contextual information. This study explores the potential of State-Space Models (SSMs), particularly Mamba, to overcome these limitations, building on their recent success in vision applications. We propose OSDMamba, the first Mamba-based architecture specifically designed for oil spill detection. OSDMamba leverages Mamba's selective scanning mechanism to effectively expand the model's receptive field while preserving critical details. Moreover, we designed an asymmetric decoder incorporating ConvSSM and deep supervision to strengthen multi-scale feature fusion, thereby enhancing the model's sensitivity to minority class samples. Experimental results show that the proposed OSDMamba achieves state-of-the-art performance, yielding improvements of 8.9% and 11.8% in OSD across two publicly available datasets.
>
---
#### [new 054] EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉转换任务，旨在将第三人称视角转换为第一人称视角。针对现有方法依赖2D线索和理想条件的问题，提出EgoWorld框架，利用多模态数据生成高质量第一人称图像。**

- **链接: [http://arxiv.org/pdf/2506.17896v1](http://arxiv.org/pdf/2506.17896v1)**

> **作者:** Junho Park; Andrew Sangwoo Ye; Taein Kwon
>
> **备注:** Project Page: https://redorangeyellowy.github.io/EgoWorld/
>
> **摘要:** Egocentric vision is essential for both human and machine visual understanding, particularly in capturing the detailed hand-object interactions needed for manipulation tasks. Translating third-person views into first-person views significantly benefits augmented reality (AR), virtual reality (VR) and robotics applications. However, current exocentric-to-egocentric translation methods are limited by their dependence on 2D cues, synchronized multi-view settings, and unrealistic assumptions such as necessity of initial egocentric frame and relative camera poses during inference. To overcome these challenges, we introduce EgoWorld, a novel two-stage framework that reconstructs an egocentric view from rich exocentric observations, including projected point clouds, 3D hand poses, and textual descriptions. Our approach reconstructs a point cloud from estimated exocentric depth maps, reprojects it into the egocentric perspective, and then applies diffusion-based inpainting to produce dense, semantically coherent egocentric images. Evaluated on the H2O and TACO datasets, EgoWorld achieves state-of-the-art performance and demonstrates robust generalization to new objects, actions, scenes, and subjects. Moreover, EgoWorld shows promising results even on unlabeled real-world examples.
>
---
#### [new 055] Training-free Test-time Improvement for Explainable Medical Image Classification
- **分类: cs.CV**

- **简介: 该论文属于医疗图像分类任务，解决CBM在新环境中的概念偏移问题。通过无需训练的方法提升模型性能，增强可解释性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.18070v1](http://arxiv.org/pdf/2506.18070v1)**

> **作者:** Hangzhou He; Jiachen Tang; Lei Zhu; Kaiwen Li; Yanye Lu
>
> **备注:** This is the initial version of our work accepted by MICCAI 2025. We'll include a link to the version on SpringerLink after this becomes available
>
> **摘要:** Deep learning-based medical image classification techniques are rapidly advancing in medical image analysis, making it crucial to develop accurate and trustworthy models that can be efficiently deployed across diverse clinical scenarios. Concept Bottleneck Models (CBMs), which first predict a set of explainable concepts from images and then perform classification based on these concepts, are increasingly being adopted for explainable medical image classification. However, the inherent explainability of CBMs introduces new challenges when deploying trained models to new environments. Variations in imaging protocols and staining methods may induce concept-level shifts, such as alterations in color distribution and scale. Furthermore, since CBM training requires explicit concept annotations, fine-tuning models solely with image-level labels could compromise concept prediction accuracy and faithfulness - a critical limitation given the high cost of acquiring expert-annotated concept labels in medical domains. To address these challenges, we propose a training-free confusion concept identification strategy. By leveraging minimal new data (e.g., 4 images per class) with only image-level labels, our approach enhances out-of-domain performance without sacrificing source domain accuracy through two key operations: masking misactivated confounding concepts and amplifying under-activated discriminative concepts. The efficacy of our method is validated on both skin and white blood cell images. Our code is available at: https://github.com/riverback/TF-TTI-XMed.
>
---
#### [new 056] Latent Space Analysis for Melanoma Prevention
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决 melanoma 早期诊断问题。通过构建可解释的潜在空间，实现对皮肤病变的连续风险评估与分类。**

- **链接: [http://arxiv.org/pdf/2506.18414v1](http://arxiv.org/pdf/2506.18414v1)**

> **作者:** Ciro Listone; Aniello Murano
>
> **备注:** 11 pages, 4 figures, under review
>
> **摘要:** Melanoma represents a critical health risk due to its aggressive progression and high mortality, underscoring the need for early, interpretable diagnostic tools. While deep learning has advanced in skin lesion classification, most existing models provide only binary outputs, offering limited clinical insight. This work introduces a novel approach that extends beyond classification, enabling interpretable risk modelling through a Conditional Variational Autoencoder. The proposed method learns a structured latent space that captures semantic relationships among lesions, allowing for a nuanced, continuous assessment of morphological differences. An SVM is also trained on this representation effectively differentiating between benign nevi and melanomas, demonstrating strong and consistent performance. More importantly, the learned latent space supports visual and geometric interpretation of malignancy, with the spatial proximity of a lesion to known melanomas serving as a meaningful indicator of risk. This approach bridges predictive performance with clinical applicability, fostering early detection, highlighting ambiguous cases, and enhancing trust in AI-assisted diagnosis through transparent and interpretable decision-making.
>
---
#### [new 057] Robust Foreground-Background Separation for Severely-Degraded Videos Using Convolutional Sparse Representation Modeling
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于视频分割任务，旨在解决低帧率和噪声环境下前景与背景分离的问题。通过引入基于卷积稀疏表示的模型，提升分离效果。**

- **链接: [http://arxiv.org/pdf/2506.17838v1](http://arxiv.org/pdf/2506.17838v1)**

> **作者:** Kazuki Naganuma; Shunsuke Ono
>
> **备注:** Submitted to IEEE Transactions on Image Processing. The code is available at https://drive.google.com/file/d/1tuVuIgkArCryVSifJDyG7R468DCLMkF2/view?usp=sharing
>
> **摘要:** This paper proposes a foreground-background separation (FBS) method with a novel foreground model based on convolutional sparse representation (CSR). In order to analyze the dynamic and static components of videos acquired under undesirable conditions, such as hardware, environmental, and power limitations, it is essential to establish an FBS method that can handle videos with low frame rates and various types of noise. Existing FBS methods have two limitations that prevent us from accurately separating foreground and background components from such degraded videos. First, they only capture either data-specific or general features of the components. Second, they do not include explicit models for various types of noise to remove them in the FBS process. To this end, we propose a robust FBS method with a CSR-based foreground model. This model can adaptively capture specific spatial structures scattered in imaging data. Then, we formulate FBS as a constrained multiconvex optimization problem that incorporates CSR, functions that capture general features, and explicit noise characterization functions for multiple types of noise. Thanks to these functions, our method captures both data-specific and general features to accurately separate the components from various types of noise even under low frame rates. To obtain a solution of the optimization problem, we develop an algorithm that alternately solves its two convex subproblems by newly established algorithms. Experiments demonstrate the superiority of our method over existing methods using two types of degraded videos: infrared and microscope videos.
>
---
#### [new 058] Feedback Driven Multi Stereo Vision System for Real-Time Event Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉领域，旨在解决复杂环境中实时事件分析问题。通过多立体视觉系统融合，实现场景重建与任务处理，提升交互系统的可靠性与适应性。**

- **链接: [http://arxiv.org/pdf/2506.17910v1](http://arxiv.org/pdf/2506.17910v1)**

> **作者:** Mohamed Benkedadra; Matei Mancas; Sidi Ahmed Mahmoudi
>
> **摘要:** 2D cameras are often used in interactive systems. Other systems like gaming consoles provide more powerful 3D cameras for short range depth sensing. Overall, these cameras are not reliable in large, complex environments. In this work, we propose a 3D stereo vision based pipeline for interactive systems, that is able to handle both ordinary and sensitive applications, through robust scene understanding. We explore the fusion of multiple 3D cameras to do full scene reconstruction, which allows for preforming a wide range of tasks, like event recognition, subject tracking, and notification. Using possible feedback approaches, the system can receive data from the subjects present in the environment, to learn to make better decisions, or to adapt to completely new environments. Throughout the paper, we introduce the pipeline and explain our preliminary experimentation and results. Finally, we draw the roadmap for the next steps that need to be taken, in order to get this pipeline into production
>
---
#### [new 059] SRKD: Towards Efficient 3D Point Cloud Segmentation via Structure- and Relation-aware Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文属于3D点云分割任务，旨在解决大模型计算复杂和部署困难的问题。通过知识蒸馏方法SRKD，将大模型的知识迁移至轻量模型，提升分割效果与效率。**

- **链接: [http://arxiv.org/pdf/2506.17290v1](http://arxiv.org/pdf/2506.17290v1)**

> **作者:** Yuqi Li; Junhao Dong; Zeyu Dong; Chuanguang Yang; Zhulin An; Yongjun Xu
>
> **备注:** 13 pages
>
> **摘要:** 3D point cloud segmentation faces practical challenges due to the computational complexity and deployment limitations of large-scale transformer-based models. To address this, we propose a novel Structure- and Relation-aware Knowledge Distillation framework, named SRKD, that transfers rich geometric and semantic knowledge from a large frozen teacher model (>100M) to a lightweight student model (<15M). Specifically, we propose an affinity matrix-based relation alignment module, which distills structural dependencies from the teacher to the student through point-wise similarity matching, enhancing the student's capability to learn contextual interactions. Meanwhile, we introduce a cross-sample mini-batch construction strategy that enables the student to perceive stable and generalized geometric structure. This aligns across diverse point cloud instances of the teacher, rather than within a single sample. Additionally, KL divergence is applied to align semantic distributions, and ground-truth supervision further reinforces accurate segmentation. Our method achieves state of the art performance with significantly reduced model complexity, demonstrating its effectiveness and efficiency in real-world deployment scenarios. Our Code is available at https://github.com/itsnotacie/SRKD.
>
---
#### [new 060] HIRE: Lightweight High-Resolution Image Feature Enrichment for Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型任务，旨在降低高分辨率图像特征提取的计算成本。通过轻量级特征增强方法，提升效率并减少FLOPs。**

- **链接: [http://arxiv.org/pdf/2506.17608v1](http://arxiv.org/pdf/2506.17608v1)**

> **作者:** Nikitha SR; Aradhya Neeraj Mathur; Tarun Ram Menta; Rishabh Jain; Mausoom Sarkar
>
> **备注:** Accepted in CVPR 2025 Workshop on What's Next in Multimodal Foundational Models
>
> **摘要:** The integration of high-resolution image features in modern multimodal large language models has demonstrated significant improvements in fine-grained visual understanding tasks, achieving high performance across multiple benchmarks. Since these features are obtained from large image encoders like ViT, they come with a significant increase in computational costs due to multiple calls to these encoders. In this work, we first develop an intuition for feature upsampling as a natural extension of high-resolution feature generation. Through extensive experiments and ablations, we demonstrate how a shallow feature enricher can achieve competitive results with tremendous reductions in training and inference time as well as computational cost, with upto 1.5x saving in FLOPs.
>
---
#### [new 061] CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于 embodied visual reasoning 任务，旨在解决长视频中复杂指令理解与推理问题。提出 CLiViS 框架，结合语言模型与视觉模型优势，构建动态认知地图提升推理效果。**

- **链接: [http://arxiv.org/pdf/2506.17629v1](http://arxiv.org/pdf/2506.17629v1)**

> **作者:** Kailing Li; Qi'ao Xu; Tianwen Qian; Yuqian Fu; Yang Jiao; Xiaoling Wang
>
> **摘要:** Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end-to-end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning. Consider the complementary strengths of LLMs in reasoning and VLMs in perception, we propose CLiViS. It is a novel training-free framework that leverages LLMs for high-level task planning and orchestrates VLM-driven open-world visual perception to iteratively update the scene context. Building on this synergy, the core of CLiViS is a dynamic Cognitive Map that evolves throughout the reasoning process. This map constructs a structured representation of the embodied scene, bridging low-level perception and high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long-term visual dependencies. Code is available at https://github.com/Teacher-Tom/CLiViS.
>
---
#### [new 062] Enabling PSO-Secure Synthetic Data Sharing Using Diversity-Aware Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于隐私保护数据共享任务，旨在解决合成数据的法律合规与性能不足问题。通过增强多样性实现PSO安全，提升合成数据质量与隐私保护。**

- **链接: [http://arxiv.org/pdf/2506.17975v1](http://arxiv.org/pdf/2506.17975v1)**

> **作者:** Mischa Dombrowski; Bernhard Kainz
>
> **摘要:** Synthetic data has recently reached a level of visual fidelity that makes it nearly indistinguishable from real data, offering great promise for privacy-preserving data sharing in medical imaging. However, fully synthetic datasets still suffer from significant limitations: First and foremost, the legal aspect of sharing synthetic data is often neglected and data regulations, such as the GDPR, are largley ignored. Secondly, synthetic models fall short of matching the performance of real data, even for in-domain downstream applications. Recent methods for image generation have focused on maximising image diversity instead of fidelity solely to improve the mode coverage and therefore the downstream performance of synthetic data. In this work, we shift perspective and highlight how maximizing diversity can also be interpreted as protecting natural persons from being singled out, which leads to predicate singling-out (PSO) secure synthetic datasets. Specifically, we propose a generalisable framework for training diffusion models on personal data which leads to unpersonal synthetic datasets achieving performance within one percentage point of real-data models while significantly outperforming state-of-the-art methods that do not ensure privacy. Our code is available at https://github.com/MischaD/Trichotomy.
>
---
#### [new 063] Pre-Trained LLM is a Semantic-Aware and Generalizable Segmentation Booster
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于医学图像分割任务，旨在提升分割性能。通过引入预训练LLM层，增强模型的语义理解与局部建模能力，实现高效且通用的分割方法。**

- **链接: [http://arxiv.org/pdf/2506.18034v1](http://arxiv.org/pdf/2506.18034v1)**

> **作者:** Fenghe Tang; Wenxin Ma; Zhiyang He; Xiaodong Tao; Zihang Jiang; S. Kevin Zhou
>
> **备注:** Accepted by MICCAI 2025. Code: https://github.com/FengheTan9/LLM4Seg
>
> **摘要:** With the advancement of Large Language Model (LLM) for natural language processing, this paper presents an intriguing finding: a frozen pre-trained LLM layer can process visual tokens for medical image segmentation tasks. Specifically, we propose a simple hybrid structure that integrates a pre-trained, frozen LLM layer within the CNN encoder-decoder segmentation framework (LLM4Seg). Surprisingly, this design improves segmentation performance with a minimal increase in trainable parameters across various modalities, including ultrasound, dermoscopy, polypscopy, and CT scans. Our in-depth analysis reveals the potential of transferring LLM's semantic awareness to enhance segmentation tasks, offering both improved global understanding and better local modeling capabilities. The improvement proves robust across different LLMs, validated using LLaMA and DeepSeek.
>
---
#### [new 064] ViDAR: Video Diffusion-Aware 4D Reconstruction From Monocular Inputs
- **分类: cs.CV**

- **简介: 该论文属于动态新视角合成任务，解决单目视频中结构与运动分离困难的问题。通过引入ViDAR框架，利用扩散模型生成伪多视角监督信号，提升重建质量与几何一致性。**

- **链接: [http://arxiv.org/pdf/2506.18792v1](http://arxiv.org/pdf/2506.18792v1)**

> **作者:** Michal Nazarczuk; Sibi Catley-Chandar; Thomas Tanay; Zhensong Zhang; Gregory Slabaugh; Eduardo Pérez-Pellitero
>
> **摘要:** Dynamic Novel View Synthesis aims to generate photorealistic views of moving subjects from arbitrary viewpoints. This task is particularly challenging when relying on monocular video, where disentangling structure from motion is ill-posed and supervision is scarce. We introduce Video Diffusion-Aware Reconstruction (ViDAR), a novel 4D reconstruction framework that leverages personalised diffusion models to synthesise a pseudo multi-view supervision signal for training a Gaussian splatting representation. By conditioning on scene-specific features, ViDAR recovers fine-grained appearance details while mitigating artefacts introduced by monocular ambiguity. To address the spatio-temporal inconsistency of diffusion-based supervision, we propose a diffusion-aware loss function and a camera pose optimisation strategy that aligns synthetic views with the underlying scene geometry. Experiments on DyCheck, a challenging benchmark with extreme viewpoint variation, show that ViDAR outperforms all state-of-the-art baselines in visual quality and geometric consistency. We further highlight ViDAR's strong improvement over baselines on dynamic regions and provide a new benchmark to compare performance in reconstructing motion-rich parts of the scene. Project page: https://vidar-4d.github.io
>
---
#### [new 065] Fine-Scale Soil Mapping in Alaska with Multimodal Machine Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于土壤制图任务，旨在解决阿拉斯加细尺度土壤地图生成问题，通过多模态机器学习方法提升预测精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.17302v1](http://arxiv.org/pdf/2506.17302v1)**

> **作者:** Yijun Lin; Theresa Chen; Colby Brungard; Grunwald Sabine; Sue Ives; Matt Macander; Timm Nawrocki; Yao-Yi Chiang; Nic Jelinski
>
> **备注:** 12 pages, Submitted to SIGSPATIAL 2025
>
> **摘要:** Fine-scale soil mapping in Alaska, traditionally relying on fieldwork and localized simulations, remains a critical yet underdeveloped task, despite the region's ecological importance and extensive permafrost coverage. As permafrost thaw accelerates due to climate change, it threatens infrastructure stability and key ecosystem services, such as soil carbon storage. High-resolution soil maps are essential for characterizing permafrost distribution, identifying vulnerable areas, and informing adaptation strategies. We present MISO, a vision-based machine learning (ML) model to produce statewide fine-scale soil maps for near-surface permafrost and soil taxonomy. The model integrates a geospatial foundation model for visual feature extraction, implicit neural representations for continuous spatial prediction, and contrastive learning for multimodal alignment and geo-location awareness. We compare MISO with Random Forest (RF), a traditional ML model that has been widely used in soil mapping applications. Spatial cross-validation and regional analysis across Permafrost Zones and Major Land Resource Areas (MLRAs) show that MISO generalizes better to remote, unseen locations and achieves higher recall than RF, which is critical for monitoring permafrost thaw and related environmental processes. These findings demonstrate the potential of advanced ML approaches for fine-scale soil mapping and provide practical guidance for future soil sampling and infrastructure planning in permafrost-affected landscapes. The project will be released at https://github.com/knowledge-computing/Peatland-permafrost.
>
---
#### [new 066] Spatial frequency information fusion network for few-shot learning
- **分类: cs.CV**

- **简介: 该论文属于少样本学习任务，旨在解决数据量少导致的过拟合和泛化能力差问题。通过融合空间与频率域信息提升特征表示，增强分类性能。**

- **链接: [http://arxiv.org/pdf/2506.18364v1](http://arxiv.org/pdf/2506.18364v1)**

> **作者:** Wenqing Zhao; Guojia Xie; Han Pan; Biao Yang; Weichuan Zhang
>
> **摘要:** The objective of Few-shot learning is to fully leverage the limited data resources for exploring the latent correlations within the data by applying algorithms and training a model with outstanding performance that can adequately meet the demands of practical applications. In practical applications, the number of images in each category is usually less than that in traditional deep learning, which can lead to over-fitting and poor generalization performance. Currently, many Few-shot classification models pay more attention to spatial domain information while neglecting frequency domain information, which contains more feature information. Ignoring frequency domain information will prevent the model from fully exploiting feature information, which would effect the classification performance. Based on conventional data augmentation, this paper proposes an SFIFNet with innovative data preprocessing. The key of this method is enhancing the accuracy of image feature representation by integrating frequency domain information with spatial domain information. The experimental results demonstrate the effectiveness of this method in enhancing classification performance.
>
---
#### [new 067] Object-aware Sound Source Localization via Audio-Visual Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于音频-视觉声源定位任务，旨在解决复杂场景中声源准确定位的问题。通过引入多模态大语言模型和新损失函数，提升对声源与静止物体的区分能力。**

- **链接: [http://arxiv.org/pdf/2506.18557v1](http://arxiv.org/pdf/2506.18557v1)**

> **作者:** Sung Jin Um; Dongjin Kim; Sangmin Lee; Jung Uk Kim
>
> **备注:** Accepted at CVPR 2025
>
> **摘要:** Audio-visual sound source localization task aims to spatially localize sound-making objects within visual scenes by integrating visual and audio cues. However, existing methods struggle with accurately localizing sound-making objects in complex scenes, particularly when visually similar silent objects coexist. This limitation arises primarily from their reliance on simple audio-visual correspondence, which does not capture fine-grained semantic differences between sound-making and silent objects. To address these challenges, we propose a novel sound source localization framework leveraging Multimodal Large Language Models (MLLMs) to generate detailed contextual information that explicitly distinguishes between sound-making foreground objects and silent background objects. To effectively integrate this detailed information, we introduce two novel loss functions: Object-aware Contrastive Alignment (OCA) loss and Object Region Isolation (ORI) loss. Extensive experimental results on MUSIC and VGGSound datasets demonstrate the effectiveness of our approach, significantly outperforming existing methods in both single-source and multi-source localization scenarios. Code and generated detailed contextual information are available at: https://github.com/VisualAIKHU/OA-SSL.
>
---
#### [new 068] Improving Weakly Supervised Temporal Action Localization by Exploiting Multi-resolution Information in Temporal Domain
- **分类: cs.CV**

- **简介: 该论文属于弱监督时序动作定位任务，旨在利用多尺度时间信息提升伪标签质量，通过两阶段方法优化动作定位性能。**

- **链接: [http://arxiv.org/pdf/2506.18261v1](http://arxiv.org/pdf/2506.18261v1)**

> **作者:** Rui Su; Dong Xu; Luping Zhou; Wanli Ouyang
>
> **备注:** 13 pages
>
> **摘要:** Weakly supervised temporal action localization is a challenging task as only the video-level annotation is available during the training process. To address this problem, we propose a two-stage approach to fully exploit multi-resolution information in the temporal domain and generate high quality frame-level pseudo labels based on both appearance and motion streams. Specifically, in the first stage, we generate reliable initial frame-level pseudo labels, and in the second stage, we iteratively refine the pseudo labels and use a set of selected frames with highly confident pseudo labels to train neural networks and better predict action class scores at each frame. We fully exploit temporal information at multiple scales to improve temporal action localization performance. Specifically, in order to obtain reliable initial frame-level pseudo labels, in the first stage, we propose an Initial Label Generation (ILG) module, which leverages temporal multi-resolution consistency to generate high quality class activation sequences (CASs), which consist of a number of sequences with each sequence measuring how likely each video frame belongs to one specific action class. In the second stage, we propose a Progressive Temporal Label Refinement (PTLR) framework. In our PTLR framework, two networks called Network-OTS and Network-RTS, which are respectively used to generate CASs for the original temporal scale and the reduced temporal scales, are used as two streams (i.e., the OTS stream and the RTS stream) to refine the pseudo labels in turn. By this way, the multi-resolution information in the temporal domain is exchanged at the pseudo label level, and our work can help improve each stream (i.e., the OTS/RTS stream) by exploiting the refined pseudo labels from another stream (i.e., the RTS/OTS stream).
>
---
#### [new 069] ShareGPT-4o-Image: Aligning Multimodal Models with GPT-4o-Level Image Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于多模态生成任务，旨在解决高质量图像生成的可访问性问题。通过构建数据集和模型，实现文本到图像及图文到图像的生成。**

- **链接: [http://arxiv.org/pdf/2506.18095v1](http://arxiv.org/pdf/2506.18095v1)**

> **作者:** Junying Chen; Zhenyang Cai; Pengcheng Chen; Shunian Chen; Ke Ji; Xidong Wang; Yunjin Yang; Benyou Wang
>
> **摘要:** Recent advances in multimodal generative models have unlocked photorealistic, instruction-aligned image generation, yet leading systems like GPT-4o-Image remain proprietary and inaccessible. To democratize these capabilities, we present ShareGPT-4o-Image, the first dataset comprising 45K text-to-image and 46K text-and-image-to-image data, all synthesized using GPT-4o's image generation capabilities for distilling its advanced image generation abilities. Leveraging this dataset, we develop Janus-4o, a multimodal large language model capable of both text-to-image and text-and-image-to-image generation. Janus-4o not only significantly improves text-to-image generation over its predecessor, Janus-Pro, but also newly supports text-and-image-to-image generation. Notably, it achieves impressive performance in text-and-image-to-image generation from scratch, using only 91K synthetic samples and 6 hours of training on an 8 A800-GPU machine. We hope the release of ShareGPT-4o-Image and Janus-4o will foster open research in photorealistic, instruction-aligned image generation.
>
---
#### [new 070] Normality Prior Guided Multi-Semantic Fusion Network for Unsupervised Image Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于无监督图像异常检测任务，旨在解决逻辑异常难以被现有方法检测的问题。通过引入正常先验的多语义融合网络，提升异常重建的准确性。**

- **链接: [http://arxiv.org/pdf/2506.18544v1](http://arxiv.org/pdf/2506.18544v1)**

> **作者:** Muhao Xu; Xueying Zhou; Xizhan Gao; Weiye Song; Guang Feng; Sijie Niu
>
> **摘要:** Recently, detecting logical anomalies is becoming a more challenging task compared to detecting structural ones. Existing encoder decoder based methods typically compress inputs into low-dimensional bottlenecks on the assumption that the compression process can effectively suppress the transmission of logical anomalies to the decoder. However, logical anomalies present a particular difficulty because, while their local features often resemble normal semantics, their global semantics deviate significantly from normal patterns. Thanks to the generalisation capabilities inherent in neural networks, these abnormal semantic features can propagate through low-dimensional bottlenecks. This ultimately allows the decoder to reconstruct anomalous images with misleading fidelity. To tackle the above challenge, we propose a novel normality prior guided multi-semantic fusion network for unsupervised anomaly detection. Instead of feeding the compressed bottlenecks to the decoder directly, we introduce the multi-semantic features of normal samples into the reconstruction process. To this end, we first extract abstract global semantics of normal cases by a pre-trained vision-language network, then the learnable semantic codebooks are constructed to store representative feature vectors of normal samples by vector quantisation. Finally, the above multi-semantic features are fused and employed as input to the decoder to guide the reconstruction of anomalies to approximate normality. Extensive experiments are conducted to validate the effectiveness of our proposed method, and it achieves the SOTA performance on the MVTec LOCO AD dataset with improvements of 5.7% in pixel-sPRO and 2.6% in image-AUROC. The source code is available at https://github.com/Xmh-L/NPGMF.
>
---
#### [new 071] Spatial-Temporal Pre-Training for Embryo Viability Prediction Using Time-Lapse Videos
- **分类: cs.CV**

- **简介: 该论文属于胚胎活力预测任务，解决标注数据少和视频时序对齐难题。提出时空预训练方法STPT，提升预测效果。**

- **链接: [http://arxiv.org/pdf/2506.17403v1](http://arxiv.org/pdf/2506.17403v1)**

> **作者:** Zhiyi Shi; Junsik Kim; Helen Y. Yang; Yonghyun Song; Hyun-Jic Oh; Dalit Ben-Yosef; Daniel Needleman; Hanspeter Pfister
>
> **备注:** Preprint submitted to Medical Image Analysis
>
> **摘要:** Automating embryo viability prediction for in vitro fertilization (IVF) is important but challenging due to the limited availability of labeled pregnancy outcome data, as only a small fraction of embryos are labeled after transfer. Self-supervised learning (SSL) can leverage both labeled and unlabeled data to improve prediction. However, existing SSL methods for videos are not directly applicable to embryo development videos due to two challenges: (1) embryo time-lapse videos contain hundreds of frames, requiring significant GPU memory for conventional SSL; (2) the dataset contains videos with varying lengths and many outlier frames, causing traditional video alignment methods to struggle with semantic misalignment. We propose Spatial-Temporal Pre-Training (STPT) to address these challenges. STPT includes two stages: spatial and temporal. In each stage, only one encoder is trained while the other is frozen, reducing memory demands. To handle temporal misalignment, STPT avoids frame-by-frame alignment across videos. The spatial stage learns from alignments within each video and its temporally consistent augmentations. The temporal stage then models relationships between video embeddings. Our method efficiently handles long videos and temporal variability. On 23,027 time-lapse videos (3,286 labeled), STPT achieves the highest AUC of 0.635 (95% CI: 0.632-0.638) compared to baselines, with limited computational resources.
>
---
#### [new 072] PP-DocBee2: Improved Baselines with Efficient Data for Multimodal Document Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态文档理解任务，旨在提升模型性能与效率。通过优化数据质量、改进特征融合策略，显著提升了模型效果并降低了延迟。**

- **链接: [http://arxiv.org/pdf/2506.18023v1](http://arxiv.org/pdf/2506.18023v1)**

> **作者:** Kui Huang; Xinrong Chen; Wenyu Lv; Jincheng Liao; Guanzhong Wang; Yi Liu
>
> **摘要:** This report introduces PP-DocBee2, an advanced version of the PP-DocBee, designed to enhance multimodal document understanding. Built on a large multimodal model architecture, PP-DocBee2 addresses the limitations of its predecessor through key technological improvements, including enhanced synthetic data quality, improved visual feature fusion strategy, and optimized inference methodologies. These enhancements yield an $11.4\%$ performance boost on internal benchmarks for Chinese business documents, and reduce inference latency by $73.0\%$ to the vanilla version. A key innovation of our work is a data quality optimization strategy for multimodal document tasks. By employing a large-scale multimodal pre-trained model to evaluate data, we apply a novel statistical criterion to filter outliers, ensuring high-quality training data. Inspired by insights into underutilized intermediate features in multimodal models, we enhance the ViT representational capacity by decomposing it into layers and applying a novel feature fusion strategy to improve complex reasoning. The source code and pre-trained model are available at \href{https://github.com/PaddlePaddle/PaddleMIX}{https://github.com/PaddlePaddle/PaddleMIX}.
>
---
#### [new 073] Enhancing VICReg: Random-Walk Pairing for Improved Generalization and Better Global Semantics Capturing
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于自监督学习任务，旨在解决VICReg在泛化能力和全局语义捕捉上的不足。通过引入SAG-VICReg提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.18104v1](http://arxiv.org/pdf/2506.18104v1)**

> **作者:** Idan Simai; Ronen Talmon; Uri Shaham
>
> **摘要:** In this paper, we argue that viewing VICReg-a popular self-supervised learning (SSL) method--through the lens of spectral embedding reveals a potential source of sub-optimality: it may struggle to generalize robustly to unseen data due to overreliance on the training data. This observation invites a closer look at how well this method achieves its goal of producing meaningful representations of images outside of the training set as well. Here, we investigate this issue and introduce SAG-VICReg (Stable and Generalizable VICReg), a method that builds on VICReg by incorporating new training techniques. These enhancements improve the model's ability to capture global semantics within the data and strengthen the generalization capabilities. Experiments demonstrate that SAG-VICReg effectively addresses the generalization challenge while matching or surpassing diverse state-of-the-art SSL baselines. Notably, our method exhibits superior performance on metrics designed to evaluate global semantic understanding, while simultaneously maintaining competitive results on local evaluation metrics. Furthermore, we propose a new standalone evaluation metric for embeddings that complements the standard evaluation methods and accounts for the global data structure without requiring labels--a key issue when tagged data is scarce or not available.
>
---
#### [new 074] YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决YOLO系列模型在复杂场景下全局高阶关联建模不足的问题。通过引入HyperACE机制和FullPAD范式，提升检测性能并降低计算量。**

- **链接: [http://arxiv.org/pdf/2506.17733v1](http://arxiv.org/pdf/2506.17733v1)**

> **作者:** Mengqi Lei; Siqi Li; Yihong Wu; Han Hu; You Zhou; Xinhu Zheng; Guiguang Ding; Shaoyi Du; Zongze Wu; Yue Gao
>
> **摘要:** The YOLO series models reign supreme in real-time object detection due to their superior accuracy and computational efficiency. However, both the convolutional architectures of YOLO11 and earlier versions and the area-based self-attention mechanism introduced in YOLOv12 are limited to local information aggregation and pairwise correlation modeling, lacking the capability to capture global multi-to-multi high-order correlations, which limits detection performance in complex scenarios. In this paper, we propose YOLOv13, an accurate and lightweight object detector. To address the above-mentioned challenges, we propose a Hypergraph-based Adaptive Correlation Enhancement (HyperACE) mechanism that adaptively exploits latent high-order correlations and overcomes the limitation of previous methods that are restricted to pairwise correlation modeling based on hypergraph computation, achieving efficient global cross-location and cross-scale feature fusion and enhancement. Subsequently, we propose a Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm based on HyperACE, which effectively achieves fine-grained information flow and representation synergy within the entire network by distributing correlation-enhanced features to the full pipeline. Finally, we propose to leverage depthwise separable convolutions to replace vanilla large-kernel convolutions, and design a series of blocks that significantly reduce parameters and computational complexity without sacrificing performance. We conduct extensive experiments on the widely used MS COCO benchmark, and the experimental results demonstrate that our method achieves state-of-the-art performance with fewer parameters and FLOPs. Specifically, our YOLOv13-N improves mAP by 3.0\% over YOLO11-N and by 1.5\% over YOLOv12-N. The code and models of our YOLOv13 model are available at: https://github.com/iMoonLab/yolov13.
>
---
#### [new 075] SpaNN: Detecting Multiple Adversarial Patches on CNNs by Spanning Saliency Thresholds
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于对抗攻击检测任务，旨在解决多补丁攻击检测问题。提出SpaNN方法，通过阈值分割和聚类实现高效准确的检测。**

- **链接: [http://arxiv.org/pdf/2506.18591v1](http://arxiv.org/pdf/2506.18591v1)**

> **作者:** Mauricio Byrd Victorica; György Dán; Henrik Sandberg
>
> **备注:** 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML2025)
>
> **摘要:** State-of-the-art convolutional neural network models for object detection and image classification are vulnerable to physically realizable adversarial perturbations, such as patch attacks. Existing defenses have focused, implicitly or explicitly, on single-patch attacks, leaving their sensitivity to the number of patches as an open question or rendering them computationally infeasible or inefficient against attacks consisting of multiple patches in the worst cases. In this work, we propose SpaNN, an attack detector whose computational complexity is independent of the expected number of adversarial patches. The key novelty of the proposed detector is that it builds an ensemble of binarized feature maps by applying a set of saliency thresholds to the neural activations of the first convolutional layer of the victim model. It then performs clustering on the ensemble and uses the cluster features as the input to a classifier for attack detection. Contrary to existing detectors, SpaNN does not rely on a fixed saliency threshold for identifying adversarial regions, which makes it robust against white box adversarial attacks. We evaluate SpaNN on four widely used data sets for object detection and classification, and our results show that SpaNN outperforms state-of-the-art defenses by up to 11 and 27 percentage points in the case of object detection and the case of image classification, respectively. Our code is available at https://github.com/gerkbyrd/SpaNN.
>
---
#### [new 076] Geometry-Aware Preference Learning for 3D Texture Generation
- **分类: cs.CV**

- **简介: 该论文属于3D纹理生成任务，旨在解决生成内容与人类偏好及3D结构不匹配的问题。提出一种端到端的几何感知偏好学习框架，通过可微奖励函数优化生成过程。**

- **链接: [http://arxiv.org/pdf/2506.18331v1](http://arxiv.org/pdf/2506.18331v1)**

> **作者:** AmirHossein Zamani; Tianhao Xie; Amir G. Aghdam; Tiberiu Popa; Eugene Belilovsky
>
> **摘要:** Recent advances in 3D generative models have achieved impressive results but 3D contents generated by these models may not align with subjective human preferences or task-specific criteria. Moreover, a core challenge in the 3D texture generation domain remains: most existing approaches rely on repeated calls to 2D text-to-image generative models, which lack an inherent understanding of the 3D structure of the input 3D mesh object. To address this, we propose an end-to-end differentiable preference learning framework that back-propagates human preferences, represented by differentiable reward functions, through the entire 3D generative pipeline, making the process inherently geometry-aware. We demonstrate the effectiveness of our framework using four proposed novel geometry-aware reward functions, offering a more controllable and interpretable pathway for high-quality 3D content creation from natural language.
>
---
#### [new 077] Benchmarking Foundation Models and Parameter-Efficient Fine-Tuning for Prognosis Prediction in Medical Imaging
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像预后预测任务，旨在评估AI模型在数据稀缺情况下的适应能力。通过对比不同微调方法和预训练模型，探索其在临床场景中的有效性。**

- **链接: [http://arxiv.org/pdf/2506.18434v1](http://arxiv.org/pdf/2506.18434v1)**

> **作者:** Filippo Ruffini; Elena Mulero Ayllon; Linlin Shen; Paolo Soda; Valerio Guarrasi
>
> **摘要:** Artificial Intelligence (AI) holds significant promise for improving prognosis prediction in medical imaging, yet its effective application remains challenging. In this work, we introduce a structured benchmark explicitly designed to evaluate and compare the transferability of Convolutional Neural Networks and Foundation Models in predicting clinical outcomes in COVID-19 patients, leveraging diverse publicly available Chest X-ray datasets. Our experimental methodology extensively explores a wide set of fine-tuning strategies, encompassing traditional approaches such as Full Fine-Tuning and Linear Probing, as well as advanced Parameter-Efficient Fine-Tuning methods including Low-Rank Adaptation, BitFit, VeRA, and IA3. The evaluations were conducted across multiple learning paradigms, including both extensive full-data scenarios and more clinically realistic Few-Shot Learning settings, which are critical for modeling rare disease outcomes and rapidly emerging health threats. By implementing a large-scale comparative analysis involving a diverse selection of pretrained models, including general-purpose architectures pretrained on large-scale datasets such as CLIP and DINOv2, to biomedical-specific models like MedCLIP, BioMedCLIP, and PubMedCLIP, we rigorously assess each model's capacity to effectively adapt and generalize to prognosis tasks, particularly under conditions of severe data scarcity and pronounced class imbalance. The benchmark was designed to capture critical conditions common in prognosis tasks, including variations in dataset size and class distribution, providing detailed insights into the strengths and limitations of each fine-tuning strategy. This extensive and structured evaluation aims to inform the practical deployment and adoption of robust, efficient, and generalizable AI-driven solutions in real-world clinical prognosis prediction workflows.
>
---
#### [new 078] Distributed Poisson multi-Bernoulli filtering via generalised covariance intersection
- **分类: cs.CV; math.ST; stat.TH**

- **简介: 该论文属于多目标跟踪任务，解决分布式多目标滤波问题。提出基于广义协方差交叉的PMB滤波方法，通过近似融合得到PMBM形式，提升滤波效果。**

- **链接: [http://arxiv.org/pdf/2506.18397v1](http://arxiv.org/pdf/2506.18397v1)**

> **作者:** Ángel F. García-Fernández; Giorgio Battistelli
>
> **摘要:** This paper presents the distributed Poisson multi-Bernoulli (PMB) filter based on the generalised covariance intersection (GCI) fusion rule for distributed multi-object filtering. Since the exact GCI fusion of two PMB densities is intractable, we derive a principled approximation. Specifically, we approximate the power of a PMB density as an unnormalised PMB density, which corresponds to an upper bound of the PMB density. Then, the GCI fusion rule corresponds to the normalised product of two unnormalised PMB densities. We show that the result is a Poisson multi-Bernoulli mixture (PMBM), which can be expressed in closed form. Future prediction and update steps in each filter preserve the PMBM form, which can be projected back to a PMB density before the next fusion step. Experimental results show the benefits of this approach compared to other distributed multi-object filters.
>
---
#### [new 079] 4Real-Video-V2: Fused View-Time Attention and Feedforward Reconstruction for 4D Scene Generation
- **分类: cs.CV**

- **简介: 该论文属于4D场景生成任务，解决视频与3D结构同步生成问题。提出融合时空注意力和前馈重建的框架，提升生成质量与重建精度。**

- **链接: [http://arxiv.org/pdf/2506.18839v1](http://arxiv.org/pdf/2506.18839v1)**

> **作者:** Chaoyang Wang; Ashkan Mirzaei; Vidit Goel; Willi Menapace; Aliaksandr Siarohin; Avalon Vinella; Michael Vasilkovsky; Ivan Skorokhodov; Vladislav Shakhrai; Sergey Korolev; Sergey Tulyakov; Peter Wonka
>
> **摘要:** We propose the first framework capable of computing a 4D spatio-temporal grid of video frames and 3D Gaussian particles for each time step using a feed-forward architecture. Our architecture has two main components, a 4D video model and a 4D reconstruction model. In the first part, we analyze current 4D video diffusion architectures that perform spatial and temporal attention either sequentially or in parallel within a two-stream design. We highlight the limitations of existing approaches and introduce a novel fused architecture that performs spatial and temporal attention within a single layer. The key to our method is a sparse attention pattern, where tokens attend to others in the same frame, at the same timestamp, or from the same viewpoint. In the second part, we extend existing 3D reconstruction algorithms by introducing a Gaussian head, a camera token replacement algorithm, and additional dynamic layers and training. Overall, we establish a new state of the art for 4D generation, improving both visual quality and reconstruction capability.
>
---
#### [new 080] Resampling Augmentation for Time Series Contrastive Learning: Application to Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于遥感时间序列的对比学习任务，旨在解决标签数据稀缺问题。提出一种基于重采样的增强方法，提升农业分类性能。**

- **链接: [http://arxiv.org/pdf/2506.18587v1](http://arxiv.org/pdf/2506.18587v1)**

> **作者:** Antoine Saget; Baptiste Lafabregue; Antoine Cornuéjols; Pierre Gançarski
>
> **备注:** 10 pages, 2 figures, accepted at 42nd International Conference on Machine Learning (ICML 2025) Terrabytes workshop
>
> **摘要:** Given the abundance of unlabeled Satellite Image Time Series (SITS) and the scarcity of labeled data, contrastive self-supervised pretraining emerges as a natural tool to leverage this vast quantity of unlabeled data. However, designing effective data augmentations for contrastive learning remains challenging for time series. We introduce a novel resampling-based augmentation strategy that generates positive pairs by upsampling time series and extracting disjoint subsequences while preserving temporal coverage. We validate our approach on multiple agricultural classification benchmarks using Sentinel-2 imagery, showing that it outperforms common alternatives such as jittering, resizing, and masking. Further, we achieve state-of-the-art performance on the S2-Agri100 dataset without employing spatial information or temporal encodings, surpassing more complex masked-based SSL frameworks. Our method offers a simple, yet effective, contrastive learning augmentation for remote sensing time series.
>
---
#### [new 081] Biased Teacher, Balanced Student
- **分类: cs.CV**

- **简介: 该论文属于知识蒸馏任务，解决长尾数据下教师模型偏见导致的学生学习效果差的问题。通过分解KL散度损失，提出LTKD框架提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.18496v1](http://arxiv.org/pdf/2506.18496v1)**

> **作者:** Seonghak Kim
>
> **备注:** 12 pages, 5 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Knowledge Distillation (KD) is a widely adopted model compression technique where a compact student model learns from the output of a larger, pre-trained teacher. While effective in balanced settings, conventional KD suffers significantly when applied to long-tailed data distributions, as the teacher model tends to be biased toward head classes and provides limited supervision for tail classes. In this paper, we propose Long-Tailed Knowledge Distillation (LTKD), a novel framework tailored for class-imbalanced scenarios. We begin by reformulating the standard KD objective into two components: inter-group and intra-group Kullback-Leibler (KL) divergence, corresponding to the prediction distributions across and within class groups (head, medium, tail), respectively. This decomposition allows us to identify and quantify the sources of teacher bias. To address them, we introduce (1) a rebalanced inter-group loss that calibrates the teacher's group-level predictions and (2) a uniform intra-group loss that ensures equal contribution from all groups during distillation. Extensive experiments on CIFAR-100-LT, TinyImageNet-LT, and ImageNet-LT show that LTKD consistently outperforms existing KD methods, achieving significant gains in both overall accuracy and tail-class performance. Our results demonstrate that LTKD enables effective knowledge transfer even from biased teachers, making it a strong candidate for real-world deployment in resource-constrained and imbalanced settings.
>
---
#### [new 082] Incorporating Rather Than Eliminating: Achieving Fairness for Skin Disease Diagnosis Through Group-Specific Expert
- **分类: cs.CV**

- **简介: 该论文属于医疗AI公平性任务，旨在解决皮肤疾病诊断中的算法偏见问题。通过引入群体特定专家模型，提升公平性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.17787v1](http://arxiv.org/pdf/2506.17787v1)**

> **作者:** Gelei Xu; Yuying Duan; Zheyuan Liu; Xueyang Li; Meng Jiang; Michael Lemmon; Wei Jin; Yiyu Shi
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** AI-based systems have achieved high accuracy in skin disease diagnostics but often exhibit biases across demographic groups, leading to inequitable healthcare outcomes and diminished patient trust. Most existing bias mitigation methods attempt to eliminate the correlation between sensitive attributes and diagnostic prediction, but those methods often degrade performance due to the lost of clinically relevant diagnostic cues. In this work, we propose an alternative approach that incorporates sensitive attributes to achieve fairness. We introduce FairMoE, a framework that employs layer-wise mixture-of-experts modules to serve as group-specific learners. Unlike traditional methods that rigidly assign data based on group labels, FairMoE dynamically routes data to the most suitable expert, making it particularly effective for handling cases near group boundaries. Experimental results show that, unlike previous fairness approaches that reduce performance, FairMoE achieves substantial accuracy improvements while preserving comparable fairness metrics.
>
---
#### [new 083] Rapeseed population point cloud completion network (RP-PCN) with dynamic graph convolution for 3D reconstruction of crop canopy occlusion architecture
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决作物冠层遮挡导致的点云不完整问题。通过提出RP-PCN模型，实现精准点云补全，提升产量预测精度。**

- **链接: [http://arxiv.org/pdf/2506.18292v1](http://arxiv.org/pdf/2506.18292v1)**

> **作者:** Ziyue Guo; Xin Yang; Yutao Shen; Yang Zhu; Lixi Jiang; Haiyan Cen
>
> **摘要:** Quantitative descriptions of complete canopy architecture are crucial for evaluating crop photosynthesis and yield to guide ideotype design. Although three-dimensional (3D) sensing technologies have been developed for plant and canopy reconstruction, severe occlusion and complex architectures hinder accurate canopy descriptions. In this study, we propose a point cloud completion model for 3D reconstruction of rapeseed populations from seeding to silique stages using multi-view imaging. A complete point cloud generation framework was developed with the virtual-real integration (VRI) simulation method and occlusion point detection algorithm to annotate the training dataset by distinguishing surface from occluded points. The rapeseed population point cloud completion network (RP-PCN) was designed with a multi-resolution dynamic graph convolutional encoder (MRDG) and point pyramid decoder (PPD) to predict occluded points based on input surface point clouds. A dynamic graph convolutional feature extractor (DGCFE) was introduced to capture structural variations across the growth period. The effectiveness of point cloud completion was validated by predicting yield using architectural indicators from complete point clouds of rapeseed population. The results demonstrated that RP-PCN achieved chamfer distance (CD) values of 3.35 cm, 3.46 cm, 4.32 cm, and 4.51 cm at the seedling, bolting, flowering, and silique stages, respectively. Ablation studies showed the effectiveness of the MRDG and DGCFE modules, reducing CD values by 10% and 23%, respectively. The silique efficiency index (SEI) from RP-PCN improved yield prediction accuracy by 11.2% compared to incomplete point clouds. The RP-PCN pipeline proposed in this study has the potential to be extended to other crops, significantly enhancing the analysis of population canopy architectures in field environments.
>
---
#### [new 084] RAG-6DPose: Retrieval-Augmented 6D Pose Estimation via Leveraging CAD as Knowledge Base
- **分类: cs.CV**

- **简介: 该论文属于6D姿态估计任务，旨在提升机器人操作中物体定位的准确性。通过整合CAD模型作为知识库，结合视觉与几何信息，解决遮挡和新视角下的姿态估计问题。**

- **链接: [http://arxiv.org/pdf/2506.18856v1](http://arxiv.org/pdf/2506.18856v1)**

> **作者:** Kuanning Wang; Yuqian Fu; Tianyu Wang; Yanwei Fu; Longfei Liang; Yu-Gang Jiang; Xiangyang Xue
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Accurate 6D pose estimation is key for robotic manipulation, enabling precise object localization for tasks like grasping. We present RAG-6DPose, a retrieval-augmented approach that leverages 3D CAD models as a knowledge base by integrating both visual and geometric cues. Our RAG-6DPose roughly contains three stages: 1) Building a Multi-Modal CAD Knowledge Base by extracting 2D visual features from multi-view CAD rendered images and also attaching 3D points; 2) Retrieving relevant CAD features from the knowledge base based on the current query image via our ReSPC module; and 3) Incorporating retrieved CAD information to refine pose predictions via retrieval-augmented decoding. Experimental results on standard benchmarks and real-world robotic tasks demonstrate the effectiveness and robustness of our approach, particularly in handling occlusions and novel viewpoints. Supplementary material is available on our project website: https://sressers.github.io/RAG-6DPose .
>
---
#### [new 085] Phantom-Data : Towards a General Subject-Consistent Video Generation Dataset
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决文本指令与生成视频不一致的问题。通过构建Phantom-Data数据集，提升视频生成的主体一致性与质量。**

- **链接: [http://arxiv.org/pdf/2506.18851v1](http://arxiv.org/pdf/2506.18851v1)**

> **作者:** Zhuowei Chen; Bingchuan Li; Tianxiang Ma; Lijie Liu; Mingcong Liu; Yi Zhang; Gen Li; Xinghui Li; Siyu Zhou; Qian He; Xinglong Wu
>
> **备注:** Project page:https://phantom-video.github.io/Phantom-Data/
>
> **摘要:** Subject-to-video generation has witnessed substantial progress in recent years. However, existing models still face significant challenges in faithfully following textual instructions. This limitation, commonly known as the copy-paste problem, arises from the widely used in-pair training paradigm. This approach inherently entangles subject identity with background and contextual attributes by sampling reference images from the same scene as the target video. To address this issue, we introduce \textbf{Phantom-Data, the first general-purpose cross-pair subject-to-video consistency dataset}, containing approximately one million identity-consistent pairs across diverse categories. Our dataset is constructed via a three-stage pipeline: (1) a general and input-aligned subject detection module, (2) large-scale cross-context subject retrieval from more than 53 million videos and 3 billion images, and (3) prior-guided identity verification to ensure visual consistency under contextual variation. Comprehensive experiments show that training with Phantom-Data significantly improves prompt alignment and visual quality while preserving identity consistency on par with in-pair baselines.
>
---
#### [new 086] Escaping the SpuriVerse: Can Large Vision-Language Models Generalize Beyond Seen Spurious Correlations?
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉问答任务，旨在解决LVLMs对虚假相关性的泛化问题。通过构建SpuriVerse基准，评估模型并验证合成数据提升性能的效果。**

- **链接: [http://arxiv.org/pdf/2506.18322v1](http://arxiv.org/pdf/2506.18322v1)**

> **作者:** Yiwei Yang; Chung Peng Lee; Shangbin Feng; Dora Zhao; Bingbing Wen; Anthony Z. Liu; Yulia Tsvetkov; Bill Howe
>
> **摘要:** Finetuning can cause spurious correlations to arise between non-essential features and the target labels, but benchmarks to study these effects involve contrived settings and narrow tasks. In contrast, we consider spurious correlations in multi-modal Large Vision Language Models (LVLMs) pretrained on extensive and diverse datasets without explicit task supervision. We develop a benchmark by sourcing GPT-4o errors on real-world visual-question-answering (VQA) benchmarks, then curating a subset through LVLM-human annotation and synthetic counterfactual evaluation to identify errors caused by spurious correlations. This process yields SpuriVerse, a novel benchmark comprised of 124 distinct types of spurious correlations extracted from real-world datasets, each containing 1 realistic and 10 synthetic VQA samples for a total of 1364 multiple choice questions. We evaluate 15 open and closed-source LVLMs on SpuriVerse, finding that even state-of-the-art closed-source models struggle significantly, achieving at best only 37.1% accuracy. Fine-tuning on synthetic examples that emphasize the spurious correlation improves performance to 78.40%, suggesting that training on diverse spurious patterns generalizes to unseen situations: models appear to learn to avoid "shortcuts" and attend to the overall image context.
>
---
#### [new 087] Scene-R1: Video-Grounded Large Language Models for 3D Scene Reasoning without 3D Annotations
- **分类: cs.CV**

- **简介: 该论文提出Scene-R1，解决3D场景理解任务，无需3D标注。通过视频引导和强化学习，实现透明、准确的3D推理。**

- **链接: [http://arxiv.org/pdf/2506.17545v1](http://arxiv.org/pdf/2506.17545v1)**

> **作者:** Zhihao Yuan; Shuyi Jiang; Chun-Mei Feng; Yaolun Zhang; Shuguang Cui; Zhen Li; Na Zhao
>
> **摘要:** Currently, utilizing large language models to understand the 3D world is becoming popular. Yet existing 3D-aware LLMs act as black boxes: they output bounding boxes or textual answers without revealing how those decisions are made, and they still rely on pre-trained 3D detectors to supply object proposals. We introduce Scene-R1, a video-grounded framework that learns to reason about 3D scenes without any point-wise 3D instance supervision by pairing reinforcement-learning-driven reasoning with a two-stage grounding pipeline. In the temporal grounding stage, we explicitly reason about the video and select the video snippets most relevant to an open-ended query. In the subsequent image grounding stage, we analyze the image and predict the 2D bounding box. After that, we track the object using SAM2 to produce pixel-accurate masks in RGB frames, and project them back into 3D, thereby eliminating the need for 3D detector-based proposals while capturing fine geometry and material cues. Scene-R1 can also adapt to the 3D visual question answering task to answer free-form questions directly from video. Our training pipeline only needs task-level 2D boxes or textual labels without dense 3D point-wise labels. Scene-R1 surpasses existing open-vocabulary baselines on multiple datasets, while delivering transparent, step-by-step rationales. These results show that reinforcement-learning-based reasoning combined with RGB-D video alone offers a practical, annotation-efficient route to trustworthy 3D scene understanding.
>
---
#### [new 088] P2MFDS: A Privacy-Preserving Multimodal Fall Detection System for Elderly People in Bathroom Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于老年人跌倒检测任务，旨在解决浴室环境中传统单模态系统精度不足的问题，通过融合雷达与振动传感数据，提出多模态检测系统P2MFDS。**

- **链接: [http://arxiv.org/pdf/2506.17332v1](http://arxiv.org/pdf/2506.17332v1)**

> **作者:** Haitian Wang; Yiren Wang; Xinyu Wang; Yumeng Miao; Yuliang Zhang; Yu Zhang; Atif Mansoor
>
> **备注:** Accepted to appear in the 2025 IEEE International Workshop on AIoT and Smart Systems (AIoTSys'25). Nominated for Best Paper Award and Best IoT System Implementation Award. Code and pretrained models available at: https://github.com/HaitianWang/P2MFDS-A-Privacy-Preserving-Multimodal-Fall-Detection-Network-for-Elderly-Individuals-in-Bathroom
>
> **摘要:** By 2050, people aged 65 and over are projected to make up 16 percent of the global population. As aging is closely associated with increased fall risk, particularly in wet and confined environments such as bathrooms where over 80 percent of falls occur. Although recent research has increasingly focused on non-intrusive, privacy-preserving approaches that do not rely on wearable devices or video-based monitoring, these efforts have not fully overcome the limitations of existing unimodal systems (e.g., WiFi-, infrared-, or mmWave-based), which are prone to reduced accuracy in complex environments. These limitations stem from fundamental constraints in unimodal sensing, including system bias and environmental interference, such as multipath fading in WiFi-based systems and drastic temperature changes in infrared-based methods. To address these challenges, we propose a Privacy-Preserving Multimodal Fall Detection System for Elderly People in Bathroom Environments. First, we develop a sensor evaluation framework to select and fuse millimeter-wave radar with 3D vibration sensing, and use it to construct and preprocess a large-scale, privacy-preserving multimodal dataset in real bathroom settings, which will be released upon publication. Second, we introduce P2MFDS, a dual-stream network combining a CNN-BiLSTM-Attention branch for radar motion dynamics with a multi-scale CNN-SEBlock-Self-Attention branch for vibration impact detection. By uniting macro- and micro-scale features, P2MFDS delivers significant gains in accuracy and recall over state-of-the-art approaches. Code and pretrained models will be made available at: https://github.com/HaitianWang/P2MFDS-A-Privacy-Preserving-Multimodal-Fall-Detection-Network-for-Elderly-Individuals-in-Bathroom.
>
---
#### [new 089] Vision as a Dialect: Unifying Visual Understanding and Generation via Text-Aligned Representations
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于多模态任务，旨在统一视觉理解和生成。通过文本对齐的离散表示，构建共享接口，提升跨模态效率与效果。**

- **链接: [http://arxiv.org/pdf/2506.18898v1](http://arxiv.org/pdf/2506.18898v1)**

> **作者:** Jiaming Han; Hao Chen; Yang Zhao; Hanyu Wang; Qi Zhao; Ziyan Yang; Hao He; Xiangyu Yue; Lu Jiang
>
> **备注:** Project page: https://tar.csuhan.com
>
> **摘要:** This paper presents a multimodal framework that attempts to unify visual understanding and generation within a shared discrete semantic representation. At its core is the Text-Aligned Tokenizer (TA-Tok), which converts images into discrete tokens using a text-aligned codebook projected from a large language model's (LLM) vocabulary. By integrating vision and text into a unified space with an expanded vocabulary, our multimodal LLM, Tar, enables cross-modal input and output through a shared interface, without the need for modality-specific designs. Additionally, we propose scale-adaptive encoding and decoding to balance efficiency and visual detail, along with a generative de-tokenizer to produce high-fidelity visual outputs. To address diverse decoding needs, we utilize two complementary de-tokenizers: a fast autoregressive model and a diffusion-based model. To enhance modality fusion, we investigate advanced pre-training tasks, demonstrating improvements in both visual understanding and generation. Experiments across benchmarks show that Tar matches or surpasses existing multimodal LLM methods, achieving faster convergence and greater training efficiency. Code, models, and data are available at https://tar.csuhan.com
>
---
#### [new 090] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，解决Transformer中平移不变性缺失问题，提出TEAFormer模型通过自适应滑动索引提升效果与收敛性。**

- **链接: [http://arxiv.org/pdf/2506.18520v1](http://arxiv.org/pdf/2506.18520v1)**

> **作者:** JiaKui Hu; Zhengjian Yao; Lujia Jin; Hangzhou He; Yanye Lu
>
> **摘要:** Translation equivariance is a fundamental inductive bias in image restoration, ensuring that translated inputs produce translated outputs. Attention mechanisms in modern restoration transformers undermine this property, adversely impacting both training convergence and generalization. To alleviate this issue, we propose two key strategies for incorporating translation equivariance: slide indexing and component stacking. Slide indexing maintains operator responses at fixed positions, with sliding window attention being a notable example, while component stacking enables the arrangement of translation-equivariant operators in parallel or sequentially, thereby building complex architectures while preserving translation equivariance. However, these strategies still create a dilemma in model design between the high computational cost of self-attention and the fixed receptive field associated with sliding window attention. To address this, we develop an adaptive sliding indexing mechanism to efficiently select key-value pairs for each query, which are then concatenated in parallel with globally aggregated key-value pairs. The designed network, called the Translation Equivariance Adaptive Transformer (TEAFormer), is assessed across a variety of image restoration tasks. The results highlight its superiority in terms of effectiveness, training convergence, and generalization.
>
---
#### [new 091] Historical Report Guided Bi-modal Concurrent Learning for Pathology Report Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于病理报告生成任务，旨在解决WSI语义不足和信息冗余问题。提出BiGen框架，结合视觉与文本模态，提升报告质量与分类性能。**

- **链接: [http://arxiv.org/pdf/2506.18658v1](http://arxiv.org/pdf/2506.18658v1)**

> **作者:** Ling Zhang; Boxiang Yun; Qingli Li; Yan Wang
>
> **摘要:** Automated pathology report generation from Whole Slide Images (WSIs) faces two key challenges: (1) lack of semantic content in visual features and (2) inherent information redundancy in WSIs. To address these issues, we propose a novel Historical Report Guided \textbf{Bi}-modal Concurrent Learning Framework for Pathology Report \textbf{Gen}eration (BiGen) emulating pathologists' diagnostic reasoning, consisting of: (1) A knowledge retrieval mechanism to provide rich semantic content, which retrieves WSI-relevant knowledge from pre-built medical knowledge bank by matching high-attention patches and (2) A bi-modal concurrent learning strategy instantiated via a learnable visual token and a learnable textual token to dynamically extract key visual features and retrieved knowledge, where weight-shared layers enable cross-modal alignment between visual features and knowledge features. Our multi-modal decoder integrates both modals for comprehensive diagnostic reports generation. Experiments on the PathText (BRCA) dataset demonstrate our framework's superiority, achieving state-of-the-art performance with 7.4\% relative improvement in NLP metrics and 19.1\% enhancement in classification metrics for Her-2 prediction versus existing methods. Ablation studies validate the necessity of our proposed modules, highlighting our method's ability to provide WSI-relevant rich semantic content and suppress information redundancy in WSIs. Code is publicly available at https://github.com/DeepMed-Lab-ECNU/BiGen.
>
---
#### [new 092] ShowFlow: From Robust Single Concept to Condition-Free Multi-Concept Generation
- **分类: cs.CV**

- **简介: 该论文属于可控图像生成任务，解决单概念和无条件多概念生成中的身份丢失与概念遗漏问题。提出ShowFlow框架，通过新方法提升生成质量。**

- **链接: [http://arxiv.org/pdf/2506.18493v1](http://arxiv.org/pdf/2506.18493v1)**

> **作者:** Trong-Vu Hoang; Quang-Binh Nguyen; Thanh-Toan Do; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** Customizing image generation remains a core challenge in controllable image synthesis. For single-concept generation, maintaining both identity preservation and prompt alignment is challenging. In multi-concept scenarios, relying solely on a prompt without additional conditions like layout boxes or semantic masks, often leads to identity loss and concept omission. In this paper, we introduce ShowFlow, a comprehensive framework designed to tackle these challenges. We propose ShowFlow-S for single-concept image generation, and ShowFlow-M for handling multiple concepts. ShowFlow-S introduces a KronA-WED adapter, which integrates a Kronecker adapter with weight and embedding decomposition, and employs a disentangled learning approach with a novel attention regularization objective to enhance single-concept generation. Building on this foundation, ShowFlow-M directly reuses the learned models from ShowFlow-S to support multi-concept generation without extra conditions, incorporating a Subject-Adaptive Matching Attention (SAMA) and a layout consistency strategy as the plug-and-play module. Extensive experiments and user studies validate ShowFlow's effectiveness, highlighting its potential in real-world applications like advertising and virtual dressing.
>
---
#### [new 093] YouTube-Occ: Learning Indoor 3D Semantic Occupancy Prediction from YouTube Videos
- **分类: cs.CV**

- **简介: 该论文属于3D语义占据预测任务，解决复杂室内环境数据获取困难的问题。通过YouTube视频构建数据集，提出自监督模型实现精准3D感知。**

- **链接: [http://arxiv.org/pdf/2506.18266v1](http://arxiv.org/pdf/2506.18266v1)**

> **作者:** Haoming Chen; Lichen Yuan; TianFang Sun; Jingyu Gong; Xin Tan; Zhizhong Zhang; Yuan Xie
>
> **摘要:** 3D semantic occupancy prediction in the past was considered to require precise geometric relationships in order to enable effective training. However, in complex indoor environments, the large-scale and widespread collection of data, along with the necessity for fine-grained annotations, becomes impractical due to the complexity of data acquisition setups and privacy concerns. In this paper, we demonstrate that 3D spatially-accurate training can be achieved using only indoor Internet data, without the need for any pre-knowledge of intrinsic or extrinsic camera parameters. In our framework, we collect a web dataset, YouTube-Occ, which comprises house tour videos from YouTube, providing abundant real house scenes for 3D representation learning. Upon on this web dataset, we establish a fully self-supervised model to leverage accessible 2D prior knowledge for reaching powerful 3D indoor perception. Specifically, we harness the advantages of the prosperous vision foundation models, distilling the 2D region-level knowledge into the occupancy network by grouping the similar pixels into superpixels. Experimental results show that our method achieves state-of-the-art zero-shot performance on two popular benchmarks (NYUv2 and OccScanNet
>
---
#### [new 094] A Set-to-Set Distance Measure in Hyperbolic Space
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于度量学习任务，旨在解决超球空间中集合间相似性度量问题。提出HS2SD方法，融合全局与局部结构信息，提升集合比较效果。**

- **链接: [http://arxiv.org/pdf/2506.18529v1](http://arxiv.org/pdf/2506.18529v1)**

> **作者:** Pengxiang Li; Wei Wu; Zhi Gao; Xiaomeng Fan; Peilin Yu; Yuwei Wu; Zhipeng Lu; Yunde Jia; Mehrtash Harandi
>
> **备注:** 24 pages
>
> **摘要:** We propose a hyperbolic set-to-set distance measure for computing dissimilarity between sets in hyperbolic space. While point-to-point distances in hyperbolic space effectively capture hierarchical relationships between data points, many real-world applications require comparing sets of hyperbolic data points, where the local structure and the global structure of the sets carry crucial semantic information. The proposed the \underline{h}yperbolic \underline{s}et-\underline{to}-\underline{s}et \underline{d}istance measure (HS2SD) integrates both global and local structural information: global structure through geodesic distances between Einstein midpoints of hyperbolic sets, and local structure through topological characteristics of the two sets. To efficiently compute topological differences, we prove that using a finite Thue-Morse sequence of degree and adjacency matrices can serve as a robust approximation to capture the topological structure of a set. In this case, by considering the topological differences, HS2SD provides a more nuanced understanding of the relationships between two hyperbolic sets. Empirical evaluation on entity matching, standard image classification, and few-shot image classification demonstrates that our distance measure outperforms existing methods by effectively modeling the hierarchical and complex relationships inherent in hyperbolic sets.
>
---
#### [new 095] VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory
- **分类: cs.CV**

- **简介: 该论文属于视频场景生成任务，旨在解决长期场景一致性与相机控制问题。提出VMem机制，通过几何索引记忆过去视图，提升生成效率与连贯性。**

- **链接: [http://arxiv.org/pdf/2506.18903v1](http://arxiv.org/pdf/2506.18903v1)**

> **作者:** Runjia Li; Philip Torr; Andrea Vedaldi; Tomas Jakab
>
> **备注:** Project page: https://v-mem.github.io
>
> **摘要:** We propose a novel memory mechanism to build video generators that can explore environments interactively. Similar results have previously been achieved by out-painting 2D views of the scene while incrementally reconstructing its 3D geometry, which quickly accumulates errors, or by video generators with a short context window, which struggle to maintain scene coherence over the long term. To address these limitations, we introduce Surfel-Indexed View Memory (VMem), a mechanism that remembers past views by indexing them geometrically based on the 3D surface elements (surfels) they have observed. VMem enables the efficient retrieval of the most relevant past views when generating new ones. By focusing only on these relevant views, our method produces consistent explorations of imagined environments at a fraction of the computational cost of using all past views as context. We evaluate our approach on challenging long-term scene synthesis benchmarks and demonstrate superior performance compared to existing methods in maintaining scene coherence and camera control.
>
---
#### [new 096] A Multimodal In Vitro Diagnostic Method for Parkinson's Disease Combining Facial Expressions and Behavioral Gait Data
- **分类: cs.CV**

- **简介: 该论文属于帕金森病诊断任务，旨在解决单一模态诊断准确性低的问题，通过融合面部表情和步态数据提出一种新型多模态诊断方法。**

- **链接: [http://arxiv.org/pdf/2506.17596v1](http://arxiv.org/pdf/2506.17596v1)**

> **作者:** Wei Huang; Yinxuan Xu; Yintao Zhou; Zhengyu Li; Jing Huang; Meng Pang
>
> **备注:** 8 pages, 4 figures, accepted by CogSci 2025
>
> **摘要:** Parkinson's disease (PD), characterized by its incurable nature, rapid progression, and severe disability, poses significant challenges to the lives of patients and their families. Given the aging population, the need for early detection of PD is increasing. In vitro diagnosis has garnered attention due to its non-invasive nature and low cost. However, existing methods present several challenges: 1) limited training data for facial expression diagnosis; 2) specialized equipment and acquisition environments required for gait diagnosis, resulting in poor generalizability; 3) the risk of misdiagnosis or missed diagnosis when relying on a single modality. To address these issues, we propose a novel multimodal in vitro diagnostic method for PD, leveraging facial expressions and behavioral gait. Our method employs a lightweight deep learning model for feature extraction and fusion, aimed at improving diagnostic accuracy and facilitating deployment on mobile devices. Furthermore, we have established the largest multimodal PD dataset in collaboration with a hospital and conducted extensive experiments to validate the effectiveness of our proposed method.
>
---
#### [new 097] Rethinking Decoder Design: Improving Biomarker Segmentation Using Depth-to-Space Restoration and Residual Linear Attention
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决生物标志物分割中特征提取不足和解码器效率低的问题。提出一种新解码器设计，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.18335v1](http://arxiv.org/pdf/2506.18335v1)**

> **作者:** Saad Wazir; Daeyoung Kim
>
> **备注:** Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025, pp. 30861-30871
>
> **摘要:** Segmenting biomarkers in medical images is crucial for various biotech applications. Despite advances, Transformer and CNN based methods often struggle with variations in staining and morphology, limiting feature extraction. In medical image segmentation, where datasets often have limited sample availability, recent state-of-the-art (SOTA) methods achieve higher accuracy by leveraging pre-trained encoders, whereas end-to-end methods tend to underperform. This is due to challenges in effectively transferring rich multiscale features from encoders to decoders, as well as limitations in decoder efficiency. To address these issues, we propose an architecture that captures multi-scale local and global contextual information and a novel decoder design, which effectively integrates features from the encoder, emphasizes important channels and regions, and reconstructs spatial dimensions to enhance segmentation accuracy. Our method, compatible with various encoders, outperforms SOTA methods, as demonstrated by experiments on four datasets and ablation studies. Specifically, our method achieves absolute performance gains of 2.76% on MoNuSeg, 3.12% on DSB, 2.87% on Electron Microscopy, and 4.03% on TNBC datasets compared to existing SOTA methods. Code: https://github.com/saadwazir/MCADS-Decoder
>
---
#### [new 098] SynDaCaTE: A Synthetic Dataset For Evaluating Part-Whole Hierarchical Inference
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于计算机视觉任务，旨在解决部分-整体层次推理的评估问题。提出SynDaCaTE数据集，用于验证模型是否真正学习到部分-整体结构。**

- **链接: [http://arxiv.org/pdf/2506.17558v1](http://arxiv.org/pdf/2506.17558v1)**

> **作者:** Jake Levi; Mark van der Wilk
>
> **备注:** Accepted at Methods and Opportunities at Small Scale (MOSS), ICML 2025, Vancouver, Canada
>
> **摘要:** Learning to infer object representations, and in particular part-whole hierarchies, has been the focus of extensive research in computer vision, in pursuit of improving data efficiency, systematic generalisation, and robustness. Models which are \emph{designed} to infer part-whole hierarchies, often referred to as capsule networks, are typically trained end-to-end on supervised tasks such as object classification, in which case it is difficult to evaluate whether such a model \emph{actually} learns to infer part-whole hierarchies, as claimed. To address this difficulty, we present a SYNthetic DAtaset for CApsule Testing and Evaluation, abbreviated as SynDaCaTE, and establish its utility by (1) demonstrating the precise bottleneck in a prominent existing capsule model, and (2) demonstrating that permutation-equivariant self-attention is highly effective for parts-to-wholes inference, which motivates future directions for designing effective inductive biases for computer vision.
>
---
#### [new 099] Generalizing Vision-Language Models to Novel Domains: A Comprehensive Survey
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型泛化任务，旨在解决模型在新领域表现下降的问题，综述了不同方法、基准及与多模态大模型的关系。**

- **链接: [http://arxiv.org/pdf/2506.18504v1](http://arxiv.org/pdf/2506.18504v1)**

> **作者:** Xinyao Li; Jingjing Li; Fengling Li; Lei Zhu; Yang Yang; Heng Tao Shen
>
> **摘要:** Recently, vision-language pretraining has emerged as a transformative technique that integrates the strengths of both visual and textual modalities, resulting in powerful vision-language models (VLMs). Leveraging web-scale pretraining data, these models exhibit strong zero-shot capabilities. However, their performance often deteriorates when confronted with domain-specific or specialized generalization tasks. To address this, a growing body of research focuses on transferring or generalizing the rich knowledge embedded in VLMs to various downstream applications. This survey aims to comprehensively summarize the generalization settings, methodologies, benchmarking and results in VLM literatures. Delving into the typical VLM structures, current literatures are categorized into prompt-based, parameter-based and feature-based methods according to the transferred modules. The differences and characteristics in each category are furthered summarized and discussed by revisiting the typical transfer learning (TL) settings, providing novel interpretations for TL in the era of VLMs. Popular benchmarks for VLM generalization are further introduced with thorough performance comparisons among the reviewed methods. Following the advances in large-scale generalizable pretraining, this survey also discusses the relations and differences between VLMs and up-to-date multimodal large language models (MLLM), e.g., DeepSeek-VL. By systematically reviewing the surging literatures in vision-language research from a novel and practical generalization prospective, this survey contributes to a clear landscape of current and future multimodal researches.
>
---
#### [new 100] DIP: Unsupervised Dense In-Context Post-training of Visual Representations
- **分类: cs.CV**

- **简介: 该论文提出DIP方法，用于提升视觉模型的密集图像表示，解决无监督上下文场景理解问题。通过伪任务训练，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.18463v1](http://arxiv.org/pdf/2506.18463v1)**

> **作者:** Sophia Sirko-Galouchenko; Spyros Gidaris; Antonin Vobecky; Andrei Bursuc; Nicolas Thome
>
> **摘要:** We introduce DIP, a novel unsupervised post-training method designed to enhance dense image representations in large-scale pretrained vision encoders for in-context scene understanding. Unlike prior approaches that rely on complex self-distillation architectures, our method trains the vision encoder using pseudo-tasks that explicitly simulate downstream in-context scenarios, inspired by meta-learning principles. To enable post-training on unlabeled data, we propose an automatic mechanism for generating in-context tasks that combines a pretrained diffusion model and the vision encoder itself. DIP is simple, unsupervised, and computationally efficient, requiring less than 9 hours on a single A100 GPU. By learning dense representations through pseudo in-context tasks, it achieves strong performance across a wide variety of downstream real-world in-context scene understanding tasks. It outperforms both the initial vision encoder and prior methods, offering a practical and effective solution for improving dense representations. Code available here: https://github.com/sirkosophia/DIP
>
---
#### [new 101] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于少样本分布外检测任务，旨在解决传统方法依赖大量数据的问题。提出AMCN网络，通过多提示对比学习提升检测效果。**

- **链接: [http://arxiv.org/pdf/2506.17633v1](http://arxiv.org/pdf/2506.17633v1)**

> **作者:** Xiang Fang; Arvind Easwaran; Blaise Genest
>
> **备注:** ICML 2025
>
> **摘要:** Out-of-distribution (OOD) detection attempts to distinguish outlier samples to prevent models trained on the in-distribution (ID) dataset from producing unavailable outputs. Most OOD detection methods require many IID samples for training, which seriously limits their real-world applications. To this end, we target a challenging setting: few-shot OOD detection, where {Only a few {\em labeled ID} samples are available.} Therefore, few-shot OOD detection is much more challenging than the traditional OOD detection setting. Previous few-shot OOD detection works ignore the distinct diversity between different classes. In this paper, we propose a novel network: Adaptive Multi-prompt Contrastive Network (AMCN), which adapts the ID-OOD separation boundary by learning inter- and intra-class distribution. To compensate for the absence of OOD and scarcity of ID {\em image samples}, we leverage CLIP, connecting text with images, engineering learnable ID and OOD {\em textual prompts}. Specifically, we first generate adaptive prompts (learnable ID prompts, label-fixed OOD prompts and label-adaptive OOD prompts). Then, we generate an adaptive class boundary for each class by introducing a class-wise threshold. Finally, we propose a prompt-guided ID-OOD separation module to control the margin between ID and OOD prompts. Experimental results show that AMCN outperforms other state-of-the-art works.
>
---
#### [new 102] MedTVT-R1: A Multimodal LLM Empowering Medical Reasoning and Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于多模态医学诊断任务，旨在解决多疾病准确诊断问题。提出MedTVT-R1模型，融合多模态数据并优化诊断推理。**

- **链接: [http://arxiv.org/pdf/2506.18512v1](http://arxiv.org/pdf/2506.18512v1)**

> **作者:** Yuting Zhang; Kaishen Yuan; Hao Lu; Yutao Yue; Jintai Chen; Kaishun Wu
>
> **摘要:** Accurate and interpretable multi-disease diagnosis remains a critical challenge in medical research, particularly when leveraging heterogeneous multimodal medical data. Current approaches often rely on single-modal data, limiting their ability to comprehensively understand complex diseases. To address this, we propose MedTVT-R1, a novel Multimodal Large Language Model (MLLM) framework designed to integrate clinical multimodal data for reasoning and diagnosing multiple diseases. We construct MedTVT-QA, a curated instruction dataset that provides question-answer pairs for physiological-level interpretations and disease-level diagnoses with a Chain of Evidence approach. MedTVT-R1 incorporates a modality perception layer to capture inter-modal dependencies and adaptively weight modality contributions. Additionally, we employ Group Relative Policy Optimization (GRPO)-based Reinforcement Fine-Tuning with a Jaccard Reward function to enhance diagnostic reasoning. Experimental results demonstrate MedTVT-R1's superiority in multimodal feature utilization and multi-disease diagnosis, offering significant potential for clinical applications such as diagnostic report generation and comorbidity reasoning. The dataset and code are available at https://github.com/keke-nice/MedTVT-R1.
>
---
#### [new 103] Cross-modal State Space Modeling for Real-time RGB-thermal Wild Scene Semantic Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于RGB-热红外图像语义分割任务，旨在解决多模态数据处理计算开销大的问题。提出CM-SSM架构，采用跨模态状态空间建模方法，提升分割效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.17869v1](http://arxiv.org/pdf/2506.17869v1)**

> **作者:** Xiaodong Guo; Zi'ang Lin; Luwen Hu; Zhihong Deng; Tong Liu; Wujie Zhou
>
> **摘要:** The integration of RGB and thermal data can significantly improve semantic segmentation performance in wild environments for field robots. Nevertheless, multi-source data processing (e.g. Transformer-based approaches) imposes significant computational overhead, presenting challenges for resource-constrained systems. To resolve this critical limitation, we introduced CM-SSM, an efficient RGB-thermal semantic segmentation architecture leveraging a cross-modal state space modeling (SSM) approach. Our framework comprises two key components. First, we introduced a cross-modal 2D-selective-scan (CM-SS2D) module to establish SSM between RGB and thermal modalities, which constructs cross-modal visual sequences and derives hidden state representations of one modality from the other. Second, we developed a cross-modal state space association (CM-SSA) module that effectively integrates global associations from CM-SS2D with local spatial features extracted through convolutional operations. In contrast with Transformer-based approaches, CM-SSM achieves linear computational complexity with respect to image resolution. Experimental results show that CM-SSM achieves state-of-the-art performance on the CART dataset with fewer parameters and lower computational cost. Further experiments on the PST900 dataset demonstrate its generalizability. Codes are available at https://github.com/xiaodonguo/CMSSM.
>
---
#### [new 104] MCN-SLAM: Multi-Agent Collaborative Neural SLAM with Hybrid Implicit Neural Scene Representation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于SLAM任务，解决多智能体协同定位与建图问题，提出MCN-SLAM框架，结合混合隐式场景表示和在线蒸馏方法，提升大场景下的地图一致性与通信效率。**

- **链接: [http://arxiv.org/pdf/2506.18678v1](http://arxiv.org/pdf/2506.18678v1)**

> **作者:** Tianchen Deng; Guole Shen; Xun Chen; Shenghai Yuan; Hongming Shen; Guohao Peng; Zhenyu Wu; Jingchuan Wang; Lihua Xie; Danwei Wang; Hesheng Wang; Weidong Chen
>
> **摘要:** Neural implicit scene representations have recently shown promising results in dense visual SLAM. However, existing implicit SLAM algorithms are constrained to single-agent scenarios, and fall difficulties in large-scale scenes and long sequences. Existing NeRF-based multi-agent SLAM frameworks cannot meet the constraints of communication bandwidth. To this end, we propose the first distributed multi-agent collaborative neural SLAM framework with hybrid scene representation, distributed camera tracking, intra-to-inter loop closure, and online distillation for multiple submap fusion. A novel triplane-grid joint scene representation method is proposed to improve scene reconstruction. A novel intra-to-inter loop closure method is designed to achieve local (single-agent) and global (multi-agent) consistency. We also design a novel online distillation method to fuse the information of different submaps to achieve global consistency. Furthermore, to the best of our knowledge, there is no real-world dataset for NeRF-based/GS-based SLAM that provides both continuous-time trajectories groundtruth and high-accuracy 3D meshes groundtruth. To this end, we propose the first real-world Dense slam (DES) dataset covering both single-agent and multi-agent scenarios, ranging from small rooms to large-scale outdoor scenes, with high-accuracy ground truth for both 3D mesh and continuous-time camera trajectory. This dataset can advance the development of the research in both SLAM, 3D reconstruction, and visual foundation model. Experiments on various datasets demonstrate the superiority of the proposed method in both mapping, tracking, and communication. The dataset and code will open-source on https://github.com/dtc111111/mcnslam.
>
---
#### [new 105] GANs vs. Diffusion Models for virtual staining with the HER2match dataset
- **分类: cs.CV**

- **简介: 该论文属于虚拟染色任务，旨在解决H&E-HER2染色转移问题。通过引入HER2match数据集并比较GAN与扩散模型性能，提升染色效果。**

- **链接: [http://arxiv.org/pdf/2506.18484v1](http://arxiv.org/pdf/2506.18484v1)**

> **作者:** Pascal Klöckner; José Teixeira; Diana Montezuma; Jaime S. Cardoso; Hugo M. Horlings; Sara P. Oliveira
>
> **摘要:** Virtual staining is a promising technique that uses deep generative models to recreate histological stains, providing a faster and more cost-effective alternative to traditional tissue chemical staining. Specifically for H&E-HER2 staining transfer, despite a rising trend in publications, the lack of sufficient public datasets has hindered progress in the topic. Additionally, it is currently unclear which model frameworks perform best for this particular task. In this paper, we introduce the HER2match dataset, the first publicly available dataset with the same breast cancer tissue sections stained with both H&E and HER2. Furthermore, we compare the performance of several Generative Adversarial Networks (GANs) and Diffusion Models (DMs), and implement a novel Brownian Bridge Diffusion Model for H&E-HER2 translation. Our findings indicate that, overall, GANs perform better than DMs, with only the BBDM achieving comparable results. Furthermore, we emphasize the importance of data alignment, as all models trained on HER2match produced vastly improved visuals compared to the widely used consecutive-slide BCI dataset. This research provides a new high-quality dataset ([available upon publication acceptance]), improving both model training and evaluation. In addition, our comparison of frameworks offers valuable guidance for researchers working on the topic.
>
---
#### [new 106] Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决VLM在推理与规划间的脱节问题。通过构建Drive-R1模型，结合监督微调和强化学习，提升视觉输入到规划决策的连贯性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.18234v1](http://arxiv.org/pdf/2506.18234v1)**

> **作者:** Yue Li; Meng Tian; Dechang Zhu; Jiangtong Zhu; Zhenyu Lin; Zhiwei Xiong; Xinhai Zhao
>
> **摘要:** Large vision-language models (VLMs) for autonomous driving (AD) are evolving beyond perception and cognition tasks toward motion planning. However, we identify two critical challenges in this direction: (1) VLMs tend to learn shortcuts by relying heavily on history input information, achieving seemingly strong planning results without genuinely understanding the visual inputs; and (2) the chain-ofthought (COT) reasoning processes are always misaligned with the motion planning outcomes, and how to effectively leverage the complex reasoning capability to enhance planning remains largely underexplored. In this paper, we start from a small-scale domain-specific VLM and propose Drive-R1 designed to bridges the scenario reasoning and motion planning for AD. Drive-R1 first undergoes the supervised finetuning on a elaborate dataset containing both long and short COT data. Drive-R1 is encouraged to reason step-by-step from visual input to final planning decisions. Subsequently, Drive-R1 is trained within a reinforcement learning framework that incentivizes the discovery of reasoning paths that are more informative for planning, guided by rewards based on predicted trajectories and meta actions. Experimental evaluations on the nuScenes and DriveLM-nuScenes benchmarks demonstrate that Drive-R1 achieves superior performance compared to existing state-of-the-art VLMs. We believe that Drive-R1 presents a promising direction for bridging reasoning and planning in AD, offering methodological insights for future research and applications.
>
---
#### [new 107] Frequency-Domain Fusion Transformer for Image Inpainting
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决传统方法在处理复杂纹理和大范围遮挡时细节丢失的问题。通过引入频域融合的Transformer结构，提升细节保留与计算效率。**

- **链接: [http://arxiv.org/pdf/2506.18437v1](http://arxiv.org/pdf/2506.18437v1)**

> **作者:** Sijin He; Guangfeng Lin; Tao Li; Yajun Chen
>
> **摘要:** Image inpainting plays a vital role in restoring missing image regions and supporting high-level vision tasks, but traditional methods struggle with complex textures and large occlusions. Although Transformer-based approaches have demonstrated strong global modeling capabilities, they often fail to preserve high-frequency details due to the low-pass nature of self-attention and suffer from high computational costs. To address these challenges, this paper proposes a Transformer-based image inpainting method incorporating frequency-domain fusion. Specifically, an attention mechanism combining wavelet transform and Gabor filtering is introduced to enhance multi-scale structural modeling and detail preservation. Additionally, a learnable frequency-domain filter based on the fast Fourier transform is designed to replace the feedforward network, enabling adaptive noise suppression and detail retention. The model adopts a four-level encoder-decoder structure and is guided by a novel loss strategy to balance global semantics and fine details. Experimental results demonstrate that the proposed method effectively improves the quality of image inpainting by preserving more high-frequency information.
>
---
#### [new 108] Multi-Scale Spectral Attention Module-based Hyperspectral Segmentation in Autonomous Driving Scenarios
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于高光谱图像语义分割任务，旨在提升自动驾驶中的环境感知。通过引入多尺度光谱注意力模块，有效处理高维光谱数据，提高分割性能。**

- **链接: [http://arxiv.org/pdf/2506.18682v1](http://arxiv.org/pdf/2506.18682v1)**

> **作者:** Imad Ali Shah; Jiarong Li; Tim Brophy; Martin Glavin; Edward Jones; Enda Ward; Brian Deegan
>
> **摘要:** Recent advances in autonomous driving (AD) have highlighted the potential of Hyperspectral Imaging (HSI) for enhanced environmental perception, particularly in challenging weather and lighting conditions. However, efficiently processing its high-dimensional spectral data remains a significant challenge. This paper introduces a Multi-scale Spectral Attention Module (MSAM) that enhances spectral feature extraction through three parallel 1D convolutions with varying kernel sizes between 1 to 11, coupled with an adaptive feature aggregation mechanism. By integrating MSAM into UNet's skip connections (UNet-SC), our proposed UNet-MSAM achieves significant improvements in semantic segmentation performance across multiple HSI datasets: HyKo-VIS v2, HSI-Drive v2, and Hyperspectral City v2. Our comprehensive experiments demonstrate that with minimal computational overhead (on average 0.02% in parameters and 0.82% GFLOPS), UNet-MSAM consistently outperforms UNet-SC, achieving average improvements of 3.61% in mean IoU and 3.80% in mF1 across the three datasets. Through extensive ablation studies, we have established that multi-scale kernel combinations perform better than single-scale configurations. These findings demonstrate the potential of HSI processing for AD and provide valuable insights into designing robust, multi-scale spectral feature extractors for real-world applications.
>
---
#### [new 109] Adaptive Mask-guided K-space Diffusion for Accelerated MRI Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于MRI重建任务，旨在解决传统方法未考虑k空间频率重要性的问题。提出自适应掩码引导的扩散模型，提升重建质量。**

- **链接: [http://arxiv.org/pdf/2506.18270v1](http://arxiv.org/pdf/2506.18270v1)**

> **作者:** Qinrong Cai; Yu Guan; Zhibo Chen; Dong Liang; Qiuyun Fan; Qiegen Liu
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** As the deep learning revolution marches on, masked modeling has emerged as a distinctive approach that involves predicting parts of the original data that are proportionally masked during training, and has demonstrated exceptional performance in multiple fields. Magnetic Resonance Imaging (MRI) reconstruction is a critical task in medical imaging that seeks to recover high-quality images from under-sampled k-space data. However, previous MRI reconstruction strategies usually optimized the entire image domain or k-space, without considering the importance of different frequency regions in the k-space This work introduces a diffusion model based on adaptive masks (AMDM), which utilizes the adaptive adjustment of frequency distribution based on k-space data to develop a hybrid masks mechanism that adapts to different k-space inputs. This enables the effective separation of high-frequency and low-frequency components, producing diverse frequency-specific representations. Additionally, the k-space frequency distribution informs the generation of adaptive masks, which, in turn, guide a closed-loop diffusion process. Experimental results verified the ability of this method to learn specific frequency information and thereby improved the quality of MRI reconstruction, providing a flexible framework for optimizing k-space data using masks in the future.
>
---
#### [new 110] LoLA-SpecViT: Local Attention SwiGLU Vision Transformer with LoRA for Hyperspectral Imaging
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像分类任务，旨在解决数据高维、冗余及标注样本少的问题。提出LoLA-SpecViT模型，结合局部注意力与LoRA技术，提升性能与效率。**

- **链接: [http://arxiv.org/pdf/2506.17759v1](http://arxiv.org/pdf/2506.17759v1)**

> **作者:** Fadi Abdeladhim Zidi; Djamel Eddine Boukhari; Abdellah Zakaria Sellam; Abdelkrim Ouafi; Cosimo Distante; Salah Eddine Bekhouche; Abdelmalik Taleb-Ahmed
>
> **摘要:** Hyperspectral image classification remains a challenging task due to the high dimensionality of spectral data, significant inter-band redundancy, and the limited availability of annotated samples. While recent transformer-based models have improved the global modeling of spectral-spatial dependencies, their scalability and adaptability under label-scarce conditions remain limited. In this work, we propose \textbf{LoLA-SpecViT}(Low-rank adaptation Local Attention Spectral Vision Transformer), a lightweight spectral vision transformer that addresses these limitations through a parameter-efficient architecture tailored to the unique characteristics of hyperspectral imagery. Our model combines a 3D convolutional spectral front-end with local window-based self-attention, enhancing both spectral feature extraction and spatial consistency while reducing computational complexity. To further improve adaptability, we integrate low-rank adaptation (LoRA) into attention and projection layers, enabling fine-tuning with over 80\% fewer trainable parameters. A novel cyclical learning rate scheduler modulates LoRA adaptation strength during training, improving convergence and generalisation. Extensive experiments on three benchmark datasets WHU-Hi LongKou, WHU-Hi HongHu, and Salinas demonstrate that LoLA-SpecViT consistently outperforms state-of-the-art baselines, achieving up to 99.91\% accuracy with substantially fewer parameters and enhanced robustness under low-label regimes. The proposed framework provides a scalable and generalizable solution for real-world HSI applications in agriculture, environmental monitoring, and remote sensing analytics. Our code is available in the following \href{https://github.com/FadiZidiDz/LoLA-SpecViT}{GitHub Repository}.
>
---
#### [new 111] Cause-Effect Driven Optimization for Robust Medical Visual Question Answering with Language Biases
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉问答任务，旨在解决语言偏见问题。提出CEDO框架，通过三种机制缓解因果和效应偏见，提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.17903v1](http://arxiv.org/pdf/2506.17903v1)**

> **作者:** Huanjia Zhu; Yishu Liu; Xiaozhao Fang; Guangming Lu; Bingzhi Chen
>
> **备注:** Accepted at IJCAI 2025
>
> **摘要:** Existing Medical Visual Question Answering (Med-VQA) models often suffer from language biases, where spurious correlations between question types and answer categories are inadvertently established. To address these issues, we propose a novel Cause-Effect Driven Optimization framework called CEDO, that incorporates three well-established mechanisms, i.e., Modality-driven Heterogeneous Optimization (MHO), Gradient-guided Modality Synergy (GMS), and Distribution-adapted Loss Rescaling (DLR), for comprehensively mitigating language biases from both causal and effectual perspectives. Specifically, MHO employs adaptive learning rates for specific modalities to achieve heterogeneous optimization, thus enhancing robust reasoning capabilities. Additionally, GMS leverages the Pareto optimization method to foster synergistic interactions between modalities and enforce gradient orthogonality to eliminate bias updates, thereby mitigating language biases from the effect side, i.e., shortcut bias. Furthermore, DLR is designed to assign adaptive weights to individual losses to ensure balanced learning across all answer categories, effectively alleviating language biases from the cause side, i.e., imbalance biases within datasets. Extensive experiments on multiple traditional and bias-sensitive benchmarks consistently demonstrate the robustness of CEDO over state-of-the-art competitors.
>
---
#### [new 112] PDC-Net: Pattern Divide-and-Conquer Network for Pelvic Radiation Injury Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决骨盆放射性损伤的自动分割问题。通过设计PDC-Net网络，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.17712v1](http://arxiv.org/pdf/2506.17712v1)**

> **作者:** Xinyu Xiong; Wuteng Cao; Zihuang Wu; Lei Zhang; Chong Gao; Guanbin Li; Qiyuan Qin
>
> **备注:** MICCAI 2025
>
> **摘要:** Accurate segmentation of Pelvic Radiation Injury (PRI) from Magnetic Resonance Images (MRI) is crucial for more precise prognosis assessment and the development of personalized treatment plans. However, automated segmentation remains challenging due to factors such as complex organ morphologies and confusing context. To address these challenges, we propose a novel Pattern Divide-and-Conquer Network (PDC-Net) for PRI segmentation. The core idea is to use different network modules to "divide" various local and global patterns and, through flexible feature selection, to "conquer" the Regions of Interest (ROI) during the decoding phase. Specifically, considering that our ROI often manifests as strip-like or circular-like structures in MR slices, we introduce a Multi-Direction Aggregation (MDA) module. This module enhances the model's ability to fit the shape of the organ by applying strip convolutions in four distinct directions. Additionally, to mitigate the challenge of confusing context, we propose a Memory-Guided Context (MGC) module. This module explicitly maintains a memory parameter to track cross-image patterns at the dataset level, thereby enhancing the distinction between global patterns associated with the positive and negative classes. Finally, we design an Adaptive Fusion Decoder (AFD) that dynamically selects features from different patterns based on the Mixture-of-Experts (MoE) framework, ultimately generating the final segmentation results. We evaluate our method on the first large-scale pelvic radiation injury dataset, and the results demonstrate the superiority of our PDC-Net over existing approaches.
>
---
#### [new 113] AQUA20: A Benchmark Dataset for Underwater Species Classification under Challenging Conditions
- **分类: cs.CV**

- **简介: 该论文属于 underwater species classification 任务，旨在解决复杂水下环境下的物种识别问题。研究构建了AQUA20数据集，并评估了多种深度学习模型的性能。**

- **链接: [http://arxiv.org/pdf/2506.17455v1](http://arxiv.org/pdf/2506.17455v1)**

> **作者:** Taufikur Rahman Fuad; Sabbir Ahmed; Shahriar Ivan
>
> **备注:** Submitted to AJSE Springer
>
> **摘要:** Robust visual recognition in underwater environments remains a significant challenge due to complex distortions such as turbidity, low illumination, and occlusion, which severely degrade the performance of standard vision systems. This paper introduces AQUA20, a comprehensive benchmark dataset comprising 8,171 underwater images across 20 marine species reflecting real-world environmental challenges such as illumination, turbidity, occlusions, etc., providing a valuable resource for underwater visual understanding. Thirteen state-of-the-art deep learning models, including lightweight CNNs (SqueezeNet, MobileNetV2) and transformer-based architectures (ViT, ConvNeXt), were evaluated to benchmark their performance in classifying marine species under challenging conditions. Our experimental results show ConvNeXt achieving the best performance, with a Top-3 accuracy of 98.82% and a Top-1 accuracy of 90.69%, as well as the highest overall F1-score of 88.92% with moderately large parameter size. The results obtained from our other benchmark models also demonstrate trade-offs between complexity and performance. We also provide an extensive explainability analysis using GRAD-CAM and LIME for interpreting the strengths and pitfalls of the models. Our results reveal substantial room for improvement in underwater species recognition and demonstrate the value of AQUA20 as a foundation for future research in this domain. The dataset is publicly available at: https://huggingface.co/datasets/taufiktrf/AQUA20.
>
---
#### [new 114] RDPO: Real Data Preference Optimization for Physics Consistency Video Generation
- **分类: cs.CV; I.2.6; I.2.10**

- **简介: 该论文属于视频生成任务，旨在提升生成视频的物理一致性。通过无标注数据的RDPO框架，从真实视频中提取物理先验，优化生成模型，增强视频动作连贯性和物理真实性。**

- **链接: [http://arxiv.org/pdf/2506.18655v1](http://arxiv.org/pdf/2506.18655v1)**

> **作者:** Wenxu Qian; Chaoyue Wang; Hou Peng; Zhiyu Tan; Hao Li; Anxiang Zeng
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Video generation techniques have achieved remarkable advancements in visual quality, yet faithfully reproducing real-world physics remains elusive. Preference-based model post-training may improve physical consistency, but requires costly human-annotated datasets or reward models that are not yet feasible. To address these challenges, we present Real Data Preference Optimisation (RDPO), an annotation-free framework that distills physical priors directly from real-world videos. Specifically, the proposed RDPO reverse-samples real video sequences with a pre-trained generator to automatically build preference pairs that are statistically distinguishable in terms of physical correctness. A multi-stage iterative training schedule then guides the generator to obey physical laws increasingly well. Benefiting from the dynamic information explored from real videos, our proposed RDPO significantly improves the action coherence and physical realism of the generated videos. Evaluations on multiple benchmarks and human evaluations have demonstrated that RDPO achieves improvements across multiple dimensions. The source code and demonstration of this paper are available at: https://wwenxu.github.io/RDPO/
>
---
#### [new 115] CmFNet: Cross-modal Fusion Network for Weakly-supervised Segmentation of Medical Images
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决弱监督下分割性能下降和过拟合问题。提出CmFNet模型，通过跨模态融合与混合监督策略提升分割效果。**

- **链接: [http://arxiv.org/pdf/2506.18042v1](http://arxiv.org/pdf/2506.18042v1)**

> **作者:** Dongdong Meng; Sheng Li; Hao Wu; Suqing Tian; Wenjun Ma; Guoping Wang; Xueqing Yan
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Accurate automatic medical image segmentation relies on high-quality, dense annotations, which are costly and time-consuming. Weakly supervised learning provides a more efficient alternative by leveraging sparse and coarse annotations instead of dense, precise ones. However, segmentation performance degradation and overfitting caused by sparse annotations remain key challenges. To address these issues, we propose CmFNet, a novel 3D weakly supervised cross-modal medical image segmentation approach. CmFNet consists of three main components: a modality-specific feature learning network, a cross-modal feature learning network, and a hybrid-supervised learning strategy. Specifically, the modality-specific feature learning network and the cross-modal feature learning network effectively integrate complementary information from multi-modal images, enhancing shared features across modalities to improve segmentation performance. Additionally, the hybrid-supervised learning strategy guides segmentation through scribble supervision, intra-modal regularization, and inter-modal consistency, modeling spatial and contextual relationships while promoting feature alignment. Our approach effectively mitigates overfitting, delivering robust segmentation results. It excels in segmenting both challenging small tumor regions and common anatomical structures. Extensive experiments on a clinical cross-modal nasopharyngeal carcinoma (NPC) dataset (including CT and MR imaging) and the publicly available CT Whole Abdominal Organ dataset (WORD) show that our approach outperforms state-of-the-art weakly supervised methods. In addition, our approach also outperforms fully supervised methods when full annotation is used. Our approach can facilitate clinical therapy and benefit various specialists, including physicists, radiologists, pathologists, and oncologists.
>
---
#### [new 116] VisualChef: Generating Visual Aids in Cooking via Mask Inpainting
- **分类: cs.CV**

- **简介: 该论文提出VisualChef，用于生成烹饪场景的视觉辅助图像。任务是解决烹饪过程中缺乏一致视觉引导的问题，通过掩码修复生成动作执行和结果图像，保持环境一致性。**

- **链接: [http://arxiv.org/pdf/2506.18569v1](http://arxiv.org/pdf/2506.18569v1)**

> **作者:** Oleh Kuzyk; Zuoyue Li; Marc Pollefeys; Xi Wang
>
> **摘要:** Cooking requires not only following instructions but also understanding, executing, and monitoring each step - a process that can be challenging without visual guidance. Although recipe images and videos offer helpful cues, they often lack consistency in focus, tools, and setup. To better support the cooking process, we introduce VisualChef, a method for generating contextual visual aids tailored to cooking scenarios. Given an initial frame and a specified action, VisualChef generates images depicting both the action's execution and the resulting appearance of the object, while preserving the initial frame's environment. Previous work aims to integrate knowledge extracted from large language models by generating detailed textual descriptions to guide image generation, which requires fine-grained visual-textual alignment and involves additional annotations. In contrast, VisualChef simplifies alignment through mask-based visual grounding. Our key insight is identifying action-relevant objects and classifying them to enable targeted modifications that reflect the intended action and outcome while maintaining a consistent environment. In addition, we propose an automated pipeline to extract high-quality initial, action, and final state frames. We evaluate VisualChef quantitatively and qualitatively on three egocentric video datasets and show its improvements over state-of-the-art methods.
>
---
#### [new 117] Learning golf swing signatures from a single wrist-worn inertial sensor
- **分类: cs.CV**

- **简介: 该论文属于运动分析任务，旨在解决高尔夫挥杆分析的不足。通过单手腕传感器数据，实现精准动作识别与技术评估。**

- **链接: [http://arxiv.org/pdf/2506.17505v1](http://arxiv.org/pdf/2506.17505v1)**

> **作者:** Jessy Lauer
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Despite its importance for performance and injury prevention, golf swing analysis is limited by isolated metrics, underrepresentation of professional athletes, and a lack of rich, interpretable movement representations. We address these gaps with a holistic, data-driven framework for personalized golf swing analysis from a single wrist-worn sensor. We build a large dataset of professional swings from publicly available videos, reconstruct full-body 3D kinematics using biologically accurate human mesh recovery, and generate synthetic inertial data to train neural networks that infer motion and segment swing phases from wrist-based input. We learn a compositional, discrete vocabulary of motion primitives that facilitates the detection and visualization of technical flaws, and is expressive enough to predict player identity, club type, sex, and age. Our system accurately estimates full-body kinematics and swing events from wrist data, delivering lab-grade motion analysis on-course and supporting early detection of anomalous movement patterns. Explainability methods reveal subtle, individualized movement signatures, reinforcing the view that variability is a hallmark of skilled performance. Longitudinal tracking demonstrates practical value: as one player's handicap improved from 50 to 2.2 over 1.5 years, our system captured measurable technical progress and provided targeted, actionable feedback. Our findings challenge common assumptions, such as swing consistency across clubs and the existence of a single "ideal" swing, and uncover latent biomarkers shaped by both intrinsic traits and task-specific constraints. This work bridges lab and field-based biomechanics, offering scalable, accessible, high-fidelity motion analysis for research, coaching, and injury prevention, while opening new directions in movement-based phenotyping, personalized equipment design, and motor skill development.
>
---
#### [new 118] Light of Normals: Unified Feature Representation for Universal Photometric Stereo
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决通用光度立体中光照与法线特征耦合及高频率几何细节丢失的问题。**

- **链接: [http://arxiv.org/pdf/2506.18882v1](http://arxiv.org/pdf/2506.18882v1)**

> **作者:** Hong Li; Houyuan Chen; Chongjie Ye; Zhaoxi Chen; Bohan Li; Shaocong Xu; Xianda Guo; Xuhui Liu; Yikai Wang; Baochang Zhang; Satoshi Ikehata; Boxin Shi; Anyi Rao; Hao Zhao
>
> **摘要:** Universal photometric stereo (PS) aims to recover high-quality surface normals from objects under arbitrary lighting conditions without relying on specific illumination models. Despite recent advances such as SDM-UniPS and Uni MS-PS, two fundamental challenges persist: 1) the deep coupling between varying illumination and surface normal features, where ambiguity in observed intensity makes it difficult to determine whether brightness variations stem from lighting changes or surface orientation; and 2) the preservation of high-frequency geometric details in complex surfaces, where intricate geometries create self-shadowing, inter-reflections, and subtle normal variations that conventional feature processing operations struggle to capture accurately.
>
---
#### [new 119] SurgVidLM: Towards Multi-grained Surgical Video Understanding with Large Language Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医疗视频理解任务，旨在解决手术视频的细粒度分析问题。提出SurgVidLM模型及SVU-31K数据集，采用多粒度机制提升视频理解能力。**

- **链接: [http://arxiv.org/pdf/2506.17873v1](http://arxiv.org/pdf/2506.17873v1)**

> **作者:** Guankun Wang; Wenjin Mo; Junyi Wang; Long Bai; Kun Yuan; Ming Hu; Jinlin Wu; Junjun He; Yiming Huang; Nicolas Padoy; Zhen Lei; Hongbin Liu; Nassir Navab; Hongliang Ren
>
> **摘要:** Recent advances in Multimodal Large Language Models have demonstrated great potential in the medical domain, facilitating users to understand surgical scenes and procedures. Beyond image-based methods, the exploration of Video Large Language Models (Vid-LLMs) has emerged as a promising avenue for capturing the complex sequences of information involved in surgery. However, there is still a lack of Vid-LLMs specialized for fine-grained surgical video understanding tasks, which is crucial for analyzing specific processes or details within a surgical procedure. To bridge this gap, we propose SurgVidLM, the first video language model designed to address both full and fine-grained surgical video comprehension. To train our SurgVidLM, we construct the SVU-31K dataset which consists of over 31K video-instruction pairs, enabling both holistic understanding and detailed analysis of surgical procedures. Furthermore, we introduce the StageFocus mechanism which is a two-stage framework performing the multi-grained, progressive understanding of surgical videos. We also develop the Multi-frequency Fusion Attention to effectively integrate low and high-frequency visual tokens, ensuring the retention of critical information. Experimental results demonstrate that SurgVidLM significantly outperforms state-of-the-art Vid-LLMs in both full and fine-grained video understanding tasks, showcasing its superior capability in capturing complex procedural contexts.
>
---
#### [new 120] Photogranulometry -- Dataset of soil images with corresponding particle size distributions
- **分类: cs.CV; I.5.4; I.2.10**

- **简介: 该论文属于地质图像分析任务，旨在解决传统颗粒分析耗时昂贵的问题，通过提供高分辨率土壤图像及粒径分布数据集，支持CNN模型训练。**

- **链接: [http://arxiv.org/pdf/2506.17469v1](http://arxiv.org/pdf/2506.17469v1)**

> **作者:** Thomas Plante St-Cyr; François Duhaime; Jean-Sébastien Dubé; Simon Grenier
>
> **备注:** 8 pages, 10 figures, conference
>
> **摘要:** Traditional particle size distribution (PSD) analyses create significant downtime and are expensive in labor and maintenance. These drawbacks could be alleviated using optical grain size analysis integrated into routine geotechnical laboratory workflow. This paper presents a high-resolution dataset of 12,714 images of 321 different soil samples collected in the Montreal, Quebec region, alongside their PSD analysis. It is designed to provide a robust starting point for training convolutional neural networks (CNN) in geotechnical applications. Soil samples were photographed in a standardized top-view position with a resolution of 45 MP and a minimum scale of 39.4 micrometers per pixel, both in their moist and dry states. A custom test bench employing 13x9 inch white aluminum trays, on which the samples are spread in a thin layer, was used. For samples exceeding a size limit, a coning and quartering method was employed for mass reduction.
>
---
#### [new 121] From Virtual Games to Real-World Play
- **分类: cs.CV**

- **简介: 该论文提出RealPlay，一种基于神经网络的实时游戏引擎，用于生成逼真、连贯的视频。任务是实现从用户控制信号到真实场景视频的交互式生成，解决低延迟、时间一致性和控制响应问题。**

- **链接: [http://arxiv.org/pdf/2506.18901v1](http://arxiv.org/pdf/2506.18901v1)**

> **作者:** Wenqiang Sun; Fangyun Wei; Jinjing Zhao; Xi Chen; Zilong Chen; Hongyang Zhang; Jun Zhang; Yan Lu
>
> **备注:** Project page: https://wenqsun.github.io/RealPlay/
>
> **摘要:** We introduce RealPlay, a neural network-based real-world game engine that enables interactive video generation from user control signals. Unlike prior works focused on game-style visuals, RealPlay aims to produce photorealistic, temporally consistent video sequences that resemble real-world footage. It operates in an interactive loop: users observe a generated scene, issue a control command, and receive a short video chunk in response. To enable such realistic and responsive generation, we address key challenges including iterative chunk-wise prediction for low-latency feedback, temporal consistency across iterations, and accurate control response. RealPlay is trained on a combination of labeled game data and unlabeled real-world videos, without requiring real-world action annotations. Notably, we observe two forms of generalization: (1) control transfer-RealPlay effectively maps control signals from virtual to real-world scenarios; and (2) entity transfer-although training labels originate solely from a car racing game, RealPlay generalizes to control diverse real-world entities, including bicycles and pedestrians, beyond vehicles. Project page can be found: https://wenqsun.github.io/RealPlay/
>
---
#### [new 122] OpenEvents V1: Large-Scale Benchmark Dataset for Multimodal Event Grounding
- **分类: cs.CV**

- **简介: 该论文提出OpenEvents V1数据集，用于多模态事件定位任务，解决事件级视觉语言理解问题，通过生成事件感知的图像描述和基于文本查询的图像检索进行研究。**

- **链接: [http://arxiv.org/pdf/2506.18372v1](http://arxiv.org/pdf/2506.18372v1)**

> **作者:** Hieu Nguyen; Phuc-Tan Nguyen; Thien-Phuc Tran; Minh-Quang Nguyen; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** We introduce OpenEvents V1, a large-scale benchmark dataset aimed at advancing event-centric vision-language understanding. Unlike conventional image captioning and retrieval datasets that emphasize surface-level descriptions, OpenEvents V1 focuses on contextual and temporal grounding through two primary tasks: (1) generating rich, event-aware image captions and (2) retrieving event-relevant images based on narrative-style textual queries. The dataset contains over 200,000 news articles and 400,000 associated images sourced from CNN and The Guardian, spanning diverse domains and time periods. We provide extensive baseline results and standardized evaluation protocols for both tasks. OpenEvents V1 establishes a robust foundation for developing multimodal models capable of deep reasoning over complex real-world events. The dataset is available at https://ltnghia.github.io/eventa/openevents-v1
>
---
#### [new 123] Universal Video Temporal Grounding with Generative Multi-modal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于视频时序定位任务，旨在准确识别视频中与自然语言查询对应的时间段。通过引入UniTime模型，利用多模态大语言模型处理不同长度和类型的视频，提升时序定位效果。**

- **链接: [http://arxiv.org/pdf/2506.18883v1](http://arxiv.org/pdf/2506.18883v1)**

> **作者:** Zeqian Li; Shangzhe Di; Zhonghua Zhai; Weilin Huang; Yanfeng Wang; Weidi Xie
>
> **摘要:** This paper presents a computational model for universal video temporal grounding, which accurately localizes temporal moments in videos based on natural language queries (e.g., questions or descriptions). Unlike existing methods that are often limited to specific video domains or durations, we propose UniTime, a robust and universal video grounding model leveraging the strong vision-language understanding capabilities of generative Multi-modal Large Language Models (MLLMs). Our model effectively handles videos of diverse views, genres, and lengths while comprehending complex language queries. The key contributions include: (i) We consider steering strong MLLMs for temporal grounding in videos. To enable precise timestamp outputs, we incorporate temporal information by interleaving timestamp tokens with video tokens. (ii) By training the model to handle videos with different input granularities through adaptive frame scaling, our approach achieves robust temporal grounding for both short and long videos. (iii) Comprehensive experiments show that UniTime outperforms state-of-the-art approaches in both zero-shot and dataset-specific finetuned settings across five public temporal grounding benchmarks. (iv) When employed as a preliminary moment retriever for long-form video question-answering (VideoQA), UniTime significantly improves VideoQA accuracy, highlighting its value for complex video understanding tasks.
>
---
#### [new 124] BSMamba: Brightness and Semantic Modeling for Long-Range Interaction in Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，解决亮度提升与语义一致性难以兼顾的问题。提出BSMamba模型，通过亮度和语义建模实现长程交互。**

- **链接: [http://arxiv.org/pdf/2506.18346v1](http://arxiv.org/pdf/2506.18346v1)**

> **作者:** Tongshun Zhang; Pingping Liu; Mengen Cai; Zijian Zhang; Yubing Lu; Qiuzhan Zhou
>
> **摘要:** Current low-light image enhancement (LLIE) methods face significant limitations in simultaneously improving brightness while preserving semantic consistency, fine details, and computational efficiency. With the emergence of state-space models, particularly Mamba, image restoration has achieved remarkable performance, yet existing visual Mamba approaches flatten 2D images into 1D token sequences using fixed scanning rules, critically limiting interactions between distant tokens with causal relationships and constraining their ability to capture meaningful long-range dependencies. To address these fundamental limitations, we propose BSMamba, a novel visual Mamba architecture comprising two specially designed components: Brightness Mamba and Semantic Mamba. The Brightness Mamba revolutionizes token interaction patterns by prioritizing connections between distant tokens with similar brightness levels, effectively addressing the challenge of brightness restoration in LLIE tasks through brightness-guided selective attention. Complementing this, the Semantic Mamba establishes priority interactions between tokens sharing similar semantic meanings, allowing the model to maintain contextual consistency by connecting semantically related regions across the image, thus preserving the hierarchical nature of image semantics during enhancement. By intelligently modeling tokens based on brightness and semantic similarity rather than arbitrary scanning patterns, BSMamba transcends the constraints of conventional token sequencing while adhering to the principles of causal modeling. Extensive experiments demonstrate that BSMamba achieves state-of-the-art performance in LLIE while preserving semantic consistency.
>
---
#### [new 125] Mobile Image Analysis Application for Mantoux Skin Test
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决传统结核菌素试验存在的主观性强、准确性低的问题。通过开发移动应用，利用图像处理和深度学习技术实现自动测量与诊断。**

- **链接: [http://arxiv.org/pdf/2506.17954v1](http://arxiv.org/pdf/2506.17954v1)**

> **作者:** Liong Gele; Tan Chye Cheah
>
> **摘要:** This paper presents a newly developed mobile application designed to diagnose Latent Tuberculosis Infection (LTBI) using the Mantoux Skin Test (TST). Traditional TST methods often suffer from low follow-up return rates, patient discomfort, and subjective manual interpretation, particularly with the ball-point pen method, leading to misdiagnosis and delayed treatment. Moreover, previous developed mobile applications that used 3D reconstruction, this app utilizes scaling stickers as reference objects for induration measurement. This mobile application integrates advanced image processing technologies, including ARCore, and machine learning algorithms such as DeepLabv3 for robust image segmentation and precise measurement of skin indurations indicative of LTBI. The system employs an edge detection algorithm to enhance accuracy. The application was evaluated against standard clinical practices, demonstrating significant improvements in accuracy and reliability. This innovation is crucial for effective tuberculosis management, especially in resource-limited regions. By automating and standardizing TST evaluations, the application enhances the accessibility and efficiency of TB di-agnostics. Future work will focus on refining machine learning models, optimizing measurement algorithms, expanding functionalities to include comprehensive patient data management, and enhancing ARCore's performance across various lighting conditions and operational settings.
>
---
#### [new 126] Open Set Recognition for Endoscopic Image Classification: A Deep Learning Approach on the Kvasir Dataset
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于开放集识别任务，旨在解决医学图像分类中未见类别识别问题，通过对比不同模型在Kvasir数据集上的表现，评估其开放集识别能力。**

- **链接: [http://arxiv.org/pdf/2506.18284v1](http://arxiv.org/pdf/2506.18284v1)**

> **作者:** Kasra Moazzami; Seoyoun Son; John Lin; Sun Min Lee; Daniel Son; Hayeon Lee; Jeongho Lee; Seongji Lee
>
> **备注:** 9 pages, 3 figures, 3 tables
>
> **摘要:** Endoscopic image classification plays a pivotal role in medical diagnostics by identifying anatomical landmarks and pathological findings. However, conventional closed-set classification frameworks are inherently limited in open-world clinical settings, where previously unseen conditions can arise andcompromise model reliability. To address this, we explore the application of Open Set Recognition (OSR) techniques on the Kvasir dataset, a publicly available and diverse endoscopic image collection. In this study, we evaluate and compare the OSR capabilities of several representative deep learning architectures, including ResNet-50, Swin Transformer, and a hybrid ResNet-Transformer model, under both closed-set and open-set conditions. OpenMax is adopted as a baseline OSR method to assess the ability of these models to distinguish known classes from previously unseen categories. This work represents one of the first efforts to apply open set recognition to the Kvasir dataset and provides a foundational benchmark for evaluating OSR performance in medical image analysis. Our results offer practical insights into model behavior in clinically realistic settings and highlight the importance of OSR techniques for the safe deployment of AI systems in endoscopy.
>
---
#### [new 127] Reconstructing Tornadoes in 3D with Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决缺乏可控 tornado 数据集的问题。研究者构建了首个实验室 tornado 多视角数据集，并成功用3DGS技术进行3D重建与可视化。**

- **链接: [http://arxiv.org/pdf/2506.18677v1](http://arxiv.org/pdf/2506.18677v1)**

> **作者:** Adam Yang; Nadula Kadawedduwa; Tianfu Wang; Maria Molina; Christopher Metzler
>
> **摘要:** Accurately reconstructing the 3D structure of tornadoes is critically important for understanding and preparing for this highly destructive weather phenomenon. While modern 3D scene reconstruction techniques, such as 3D Gaussian splatting (3DGS), could provide a valuable tool for reconstructing the 3D structure of tornados, at present we are critically lacking a controlled tornado dataset with which to develop and validate these tools. In this work we capture and release a novel multiview dataset of a small lab-based tornado. We demonstrate one can effectively reconstruct and visualize the 3D structure of this tornado using 3DGS.
>
---
#### [new 128] Auto-Regressively Generating Multi-View Consistent Images
- **分类: cs.CV**

- **简介: 该论文属于多视角图像生成任务，旨在解决多视图一致性与复杂条件下的图像合成问题。提出MV-AR方法，通过自回归模型和条件注入实现高质量多视角图像生成。**

- **链接: [http://arxiv.org/pdf/2506.18527v1](http://arxiv.org/pdf/2506.18527v1)**

> **作者:** JiaKui Hu; Yuxiao Yang; Jialun Liu; Jinbo Wu; Chen Zhao; Yanye Lu
>
> **摘要:** Generating multi-view images from human instructions is crucial for 3D content creation. The primary challenges involve maintaining consistency across multiple views and effectively synthesizing shapes and textures under diverse conditions. In this paper, we propose the Multi-View Auto-Regressive (MV-AR) method, which leverages an auto-regressive model to progressively generate consistent multi-view images from arbitrary prompts. Firstly, the next-token-prediction capability of the AR model significantly enhances its effectiveness in facilitating progressive multi-view synthesis. When generating widely-separated views, MV-AR can utilize all its preceding views to extract effective reference information. Subsequently, we propose a unified model that accommodates various prompts via architecture designing and training strategies. To address multiple conditions, we introduce condition injection modules for text, camera pose, image, and shape. To manage multi-modal conditions simultaneously, a progressive training strategy is employed. This strategy initially adopts the text-to-multi-view (t2mv) model as a baseline to enhance the development of a comprehensive X-to-multi-view (X2mv) model through the randomly dropping and combining conditions. Finally, to alleviate the overfitting problem caused by limited high-quality data, we propose the "Shuffle View" data augmentation technique, thus significantly expanding the training data by several magnitudes. Experiments demonstrate the performance and versatility of our MV-AR, which consistently generates consistent multi-view images across a range of conditions and performs on par with leading diffusion-based multi-view image generation models. Code and models will be released at https://github.com/MILab-PKU/MVAR.
>
---
#### [new 129] BeltCrack: the First Sequential-image Industrial Conveyor Belt Crack Detection Dataset and Its Baseline with Triple-domain Feature Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于工业缺陷检测任务，旨在解决 conveyor belt 裂纹智能检测问题。构建了首个真实工业裂纹数据集，并提出三域特征融合方法进行检测。**

- **链接: [http://arxiv.org/pdf/2506.17892v1](http://arxiv.org/pdf/2506.17892v1)**

> **作者:** Jianghong Huang; Luping Ji; Xin Ma; Mao Ye
>
> **备注:** 32 pages, 10 figures
>
> **摘要:** Conveyor belt is a category of important equipments in modern industry, widely applied in production and manufacturing Fields. Its health status is much critical to operation efficiency and safety hazards. Among the factors affecting belt health, crack is often one of the most threatening risks. Currently, considering safety, how to intelligently detect belt cracks is catching an increasing attention. To implement the intelligent detection with machine learning, real crack samples are believed to be necessary. However, existing crack datasets primarily focus on pavement scenarios or synthetic data, no real-world industrial belt crack datasets at all. To propel machine learning advancement in this field, this paper constructs the first sequential-image belt crack detection datasets (BeltCrack14ks and BeltCrack9kd), from real-world factory scenes. Furthermore, to validate usability and effectiveness, we propose a special baseline method with triple-domain (i.e., time-space-frequency) feature hierarchical fusion learning for the two whole-new datasets. Experimental results demonstrate the availability and effectiveness of our dataset. Besides, they also show that our baseline is obviously superior to other similar detection methods. Our datasets and source codes are available at https://github.com/UESTC-nnLab/BeltCrack.
>
---
#### [new 130] Few-Shot, Now for Real: Medical VLMs Adaptation without Balanced Sets or Validation
- **分类: cs.CV**

- **简介: 该论文属于医学视觉语言模型适应任务，解决真实场景下的少样本学习问题。提出无需平衡数据和验证集的适应方法，提升模型在实际应用中的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.17500v1](http://arxiv.org/pdf/2506.17500v1)**

> **作者:** Julio Silva-Rodríguez; Fereshteh Shakeri; Houda Bahig; Jose Dolz; Ismail Ben Ayed
>
> **备注:** MICCAI 2025. Code: https://github.com/jusiro/SS-Text
>
> **摘要:** Vision-language models (VLMs) are gaining attention in medical image analysis. These are pre-trained on large, heterogeneous data sources, yielding rich and transferable representations. Notably, the combination of modality-specialized VLMs with few-shot adaptation has provided fruitful results, enabling the efficient deployment of high-performing solutions. However, previous works on this topic make strong assumptions about the distribution of adaptation data, which are unrealistic in the medical domain. First, prior art assumes access to a balanced support set, a condition that breaks the natural imbalance in disease prevalence found in real-world scenarios. Second, these works typically assume the presence of an additional validation set to fix critical hyper-parameters, which is highly data-inefficient. This work challenges these favorable deployment scenarios and introduces a realistic, imbalanced, validation-free adaptation setting. Our extensive benchmark across various modalities and downstream tasks demonstrates that current methods systematically compromise their performance when operating under realistic conditions, occasionally even performing worse than zero-shot inference. Also, we introduce a training-free linear probe that adaptively blends visual and textual supervision. Detailed studies demonstrate that the proposed solver is a strong, efficient baseline, enabling robust adaptation in challenging scenarios.
>
---
#### [new 131] BPCLIP: A Bottom-up Image Quality Assessment from Distortion to Semantics Based on CLIP
- **分类: cs.CV**

- **简介: 该论文属于图像质量评估任务，旨在解决现有方法对失真影响语义内容捕捉不足的问题。提出BPCLIP模型，结合CLIP增强语义理解，提升评估效果。**

- **链接: [http://arxiv.org/pdf/2506.17969v1](http://arxiv.org/pdf/2506.17969v1)**

> **作者:** Chenyue Song; Chen Hui; Wei Zhang; Haiqi Zhu; Shaohui Liu; Hong Huang; Feng Jiang
>
> **备注:** Accepted to ICME 2025
>
> **摘要:** Image Quality Assessment (IQA) aims to evaluate the perceptual quality of images based on human subjective perception. Existing methods generally combine multiscale features to achieve high performance, but most rely on straightforward linear fusion of these features, which may not adequately capture the impact of distortions on semantic content. To address this, we propose a bottom-up image quality assessment approach based on the Contrastive Language-Image Pre-training (CLIP, a recently proposed model that aligns images and text in a shared feature space), named BPCLIP, which progressively extracts the impact of low-level distortions on high-level semantics. Specifically, we utilize an encoder to extract multiscale features from the input image and introduce a bottom-up multiscale cross attention module designed to capture the relationships between shallow and deep features. In addition, by incorporating 40 image quality adjectives across six distinct dimensions, we enable the pre-trained CLIP text encoder to generate representations of the intrinsic quality of the image, thereby strengthening the connection between image quality perception and human language. Our method achieves superior results on most public Full-Reference (FR) and No-Reference (NR) IQA benchmarks, while demonstrating greater robustness.
>
---
#### [new 132] FilMaster: Bridging Cinematic Principles and Generative AI for Automated Film Generation
- **分类: cs.CV**

- **简介: 该论文属于电影生成任务，旨在解决AI生成影片缺乏专业镜头语言和节奏的问题。通过整合电影原则与生成模型，构建FilMaster系统提升影片质量。**

- **链接: [http://arxiv.org/pdf/2506.18899v1](http://arxiv.org/pdf/2506.18899v1)**

> **作者:** Kaiyi Huang; Yukun Huang; Xintao Wang; Zinan Lin; Xuefei Ning; Pengfei Wan; Di Zhang; Yu Wang; Xihui Liu
>
> **备注:** Project Page: https://filmaster-ai.github.io/
>
> **摘要:** AI-driven content creation has shown potential in film production. However, existing film generation systems struggle to implement cinematic principles and thus fail to generate professional-quality films, particularly lacking diverse camera language and cinematic rhythm. This results in templated visuals and unengaging narratives. To address this, we introduce FilMaster, an end-to-end AI system that integrates real-world cinematic principles for professional-grade film generation, yielding editable, industry-standard outputs. FilMaster is built on two key principles: (1) learning cinematography from extensive real-world film data and (2) emulating professional, audience-centric post-production workflows. Inspired by these principles, FilMaster incorporates two stages: a Reference-Guided Generation Stage which transforms user input to video clips, and a Generative Post-Production Stage which transforms raw footage into audiovisual outputs by orchestrating visual and auditory elements for cinematic rhythm. Our generation stage highlights a Multi-shot Synergized RAG Camera Language Design module to guide the AI in generating professional camera language by retrieving reference clips from a vast corpus of 440,000 film clips. Our post-production stage emulates professional workflows by designing an Audience-Centric Cinematic Rhythm Control module, including Rough Cut and Fine Cut processes informed by simulated audience feedback, for effective integration of audiovisual elements to achieve engaging content. The system is empowered by generative AI models like (M)LLMs and video generation models. Furthermore, we introduce FilmEval, a comprehensive benchmark for evaluating AI-generated films. Extensive experiments show FilMaster's superior performance in camera language design and cinematic rhythm control, advancing generative AI in professional filmmaking.
>
---
#### [new 133] 3D Arena: An Open Platform for Generative 3D Evaluation
- **分类: cs.CV**

- **简介: 该论文属于生成3D模型评估任务，旨在解决现有评价指标与人类感知不一致的问题。通过构建3D Arena平台，收集大规模用户偏好数据，提升评估准确性。**

- **链接: [http://arxiv.org/pdf/2506.18787v1](http://arxiv.org/pdf/2506.18787v1)**

> **作者:** Dylan Ebert
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Evaluating Generative 3D models remains challenging due to misalignment between automated metrics and human perception of quality. Current benchmarks rely on image-based metrics that ignore 3D structure or geometric measures that fail to capture perceptual appeal and real-world utility. To address this gap, we present 3D Arena, an open platform for evaluating image-to-3D generation models through large-scale human preference collection using pairwise comparisons. Since launching in June 2024, the platform has collected 123,243 votes from 8,096 users across 19 state-of-the-art models, establishing the largest human preference evaluation for Generative 3D. We contribute the iso3d dataset of 100 evaluation prompts and demonstrate quality control achieving 99.75% user authenticity through statistical fraud detection. Our ELO-based ranking system provides reliable model assessment, with the platform becoming an established evaluation resource. Through analysis of this preference data, we present insights into human preference patterns. Our findings reveal preferences for visual presentation features, with Gaussian splat outputs achieving a 16.6 ELO advantage over meshes and textured models receiving a 144.1 ELO advantage over untextured models. We provide recommendations for improving evaluation methods, including multi-criteria assessment, task-oriented evaluation, and format-aware comparison. The platform's community engagement establishes 3D Arena as a benchmark for the field while advancing understanding of human-centered evaluation in Generative 3D.
>
---
#### [new 134] VLA-OS: Structuring and Dissecting Planning Representations and Paradigms in Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决规划范式与表示对性能影响的分析问题。提出VLA-OS架构，通过系统实验验证不同规划方法的效果。**

- **链接: [http://arxiv.org/pdf/2506.17561v1](http://arxiv.org/pdf/2506.17561v1)**

> **作者:** Chongkai Gao; Zixuan Liu; Zhenghao Chi; Junshan Huang; Xin Fei; Yiwen Hou; Yuxuan Zhang; Yudi Lin; Zhirui Fang; Zeyu Jiang; Lin Shao
>
> **摘要:** Recent studies on Vision-Language-Action (VLA) models have shifted from the end-to-end action-generation paradigm toward a pipeline involving task planning followed by action generation, demonstrating improved performance on various complex, long-horizon manipulation tasks. However, existing approaches vary significantly in terms of network architectures, planning paradigms, representations, and training data sources, making it challenging for researchers to identify the precise sources of performance gains and components to be further improved. To systematically investigate the impacts of different planning paradigms and representations isolating from network architectures and training data, in this paper, we introduce VLA-OS, a unified VLA architecture series capable of various task planning paradigms, and design a comprehensive suite of controlled experiments across diverse object categories (rigid and deformable), visual modalities (2D and 3D), environments (simulation and real-world), and end-effectors (grippers and dexterous hands). Our results demonstrate that: 1) visually grounded planning representations are generally better than language planning representations; 2) the Hierarchical-VLA paradigm generally achieves superior or comparable performance than other paradigms on task performance, pretraining, generalization ability, scalability, and continual learning ability, albeit at the cost of slower training and inference speeds.
>
---
#### [new 135] CPAM: Context-Preserving Adaptive Manipulation for Zero-Shot Real Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决零样本真实图像编辑中对象与背景一致性问题。提出CPAM框架，通过自适应机制和掩码策略实现精准编辑。**

- **链接: [http://arxiv.org/pdf/2506.18438v1](http://arxiv.org/pdf/2506.18438v1)**

> **作者:** Dinh-Khoi Vo; Thanh-Toan Do; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** Editing natural images using textual descriptions in text-to-image diffusion models remains a significant challenge, particularly in achieving consistent generation and handling complex, non-rigid objects. Existing methods often struggle to preserve textures and identity, require extensive fine-tuning, and exhibit limitations in editing specific spatial regions or objects while retaining background details. This paper proposes Context-Preserving Adaptive Manipulation (CPAM), a novel zero-shot framework for complicated, non-rigid real image editing. Specifically, we propose a preservation adaptation module that adjusts self-attention mechanisms to preserve and independently control the object and background effectively. This ensures that the objects' shapes, textures, and identities are maintained while keeping the background undistorted during the editing process using the mask guidance technique. Additionally, we develop a localized extraction module to mitigate the interference with the non-desired modified regions during conditioning in cross-attention mechanisms. We also introduce various mask-guidance strategies to facilitate diverse image manipulation tasks in a simple manner. Extensive experiments on our newly constructed Image Manipulation BenchmArk (IMBA), a robust benchmark dataset specifically designed for real image editing, demonstrate that our proposed method is the preferred choice among human raters, outperforming existing state-of-the-art editing techniques.
>
---
#### [new 136] Efficient Feedback Gate Network for Hyperspectral Image Super-Resolution
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于高光谱图像超分辨率任务，旨在提升图像的空间分辨率。通过设计反馈门网络，增强光谱与空间信息的融合，提高重建质量。**

- **链接: [http://arxiv.org/pdf/2506.17361v1](http://arxiv.org/pdf/2506.17361v1)**

> **作者:** Xufei Wang; Mingjian Zhang; Fei Ge; Jinchen Zhu; Wen Sha; Jifen Ren; Zhimeng Hou; Shouguo Zheng; ling Zheng; Shizhuang Weng
>
> **备注:** 20 pages,17 figures
>
> **摘要:** Even without auxiliary images, single hyperspectral image super-resolution (SHSR) methods can be designed to improve the spatial resolution of hyperspectral images. However, failing to explore coherence thoroughly along bands and spatial-spectral information leads to the limited performance of the SHSR. In this study, we propose a novel group-based SHSR method termed the efficient feedback gate network, which uses various feedbacks and gate operations involving large kernel convolutions and spectral interactions. In particular, by providing different guidance for neighboring groups, we can learn rich band information and hierarchical hyperspectral spatial information using channel shuffling and dilatation convolution in shuffled and progressive dilated fusion module(SPDFM). Moreover, we develop a wide-bound perception gate block and a spectrum enhancement gate block to construct the spatial-spectral reinforcement gate module (SSRGM) and obtain highly representative spatial-spectral features efficiently. Additionally, we apply a three-dimensional SSRGM to enhance holistic information and coherence for hyperspectral data. The experimental results on three hyperspectral datasets demonstrate the superior performance of the proposed network over the state-of-the-art methods in terms of spectral fidelity and spatial content reconstruction.
>
---
#### [new 137] OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于音频驱动的人体动画生成任务，旨在解决面部与全身动作同步不足及控制精度低的问题。提出OmniAvatar模型，通过多层级音频嵌入和LoRA训练提升生成效果。**

- **链接: [http://arxiv.org/pdf/2506.18866v1](http://arxiv.org/pdf/2506.18866v1)**

> **作者:** Qijun Gan; Ruizi Yang; Jianke Zhu; Shaofei Xue; Steven Hoi
>
> **备注:** Project page: https://omni-avatar.github.io/
>
> **摘要:** Significant progress has been made in audio-driven human animation, while most existing methods focus mainly on facial movements, limiting their ability to create full-body animations with natural synchronization and fluidity. They also struggle with precise prompt control for fine-grained generation. To tackle these challenges, we introduce OmniAvatar, an innovative audio-driven full-body video generation model that enhances human animation with improved lip-sync accuracy and natural movements. OmniAvatar introduces a pixel-wise multi-hierarchical audio embedding strategy to better capture audio features in the latent space, enhancing lip-syncing across diverse scenes. To preserve the capability for prompt-driven control of foundation models while effectively incorporating audio features, we employ a LoRA-based training approach. Extensive experiments show that OmniAvatar surpasses existing models in both facial and semi-body video generation, offering precise text-based control for creating videos in various domains, such as podcasts, human interactions, dynamic scenes, and singing. Our project page is https://omni-avatar.github.io/.
>
---
#### [new 138] DreamJourney: Perpetual View Generation with Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出DreamJourney，解决动态场景长期视角生成问题。通过两阶段框架，结合3D重建与视频扩散模型，实现连贯动态视图生成。**

- **链接: [http://arxiv.org/pdf/2506.17705v1](http://arxiv.org/pdf/2506.17705v1)**

> **作者:** Bo Pan; Yang Chen; Yingwei Pan; Ting Yao; Wei Chen; Tao Mei
>
> **摘要:** Perpetual view generation aims to synthesize a long-term video corresponding to an arbitrary camera trajectory solely from a single input image. Recent methods commonly utilize a pre-trained text-to-image diffusion model to synthesize new content of previously unseen regions along camera movement. However, the underlying 2D diffusion model lacks 3D awareness and results in distorted artifacts. Moreover, they are limited to generating views of static 3D scenes, neglecting to capture object movements within the dynamic 4D world. To alleviate these issues, we present DreamJourney, a two-stage framework that leverages the world simulation capacity of video diffusion models to trigger a new perpetual scene view generation task with both camera movements and object dynamics. Specifically, in stage I, DreamJourney first lifts the input image to 3D point cloud and renders a sequence of partial images from a specific camera trajectory. A video diffusion model is then utilized as generative prior to complete the missing regions and enhance visual coherence across the sequence, producing a cross-view consistent video adheres to the 3D scene and camera trajectory. Meanwhile, we introduce two simple yet effective strategies (early stopping and view padding) to further stabilize the generation process and improve visual quality. Next, in stage II, DreamJourney leverages a multimodal large language model to produce a text prompt describing object movements in current view, and uses video diffusion model to animate current view with object movements. Stage I and II are repeated recurrently, enabling perpetual dynamic scene view generation. Extensive experiments demonstrate the superiority of our DreamJourney over state-of-the-art methods both quantitatively and qualitatively. Our project page: https://dream-journey.vercel.app.
>
---
#### [new 139] HalluRNN: Mitigating Hallucinations via Recurrent Cross-Layer Reasoning in Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型生成不准确内容（幻觉）的问题。提出HalluRNN架构，通过跨层推理减少幻觉，提升模型稳定性。**

- **链接: [http://arxiv.org/pdf/2506.17587v1](http://arxiv.org/pdf/2506.17587v1)**

> **作者:** Le Yu; Kaishen Wang; Jianlong Xiong; Yue Cao; Tao He
>
> **备注:** 6 figures, 9 tables
>
> **摘要:** Though Large Vision-Language Models (LVLMs) have achieved remarkable performance across various tasks, they are still prone to hallucinations-generating outputs that are textually plausible but visually ungrounded. While prior approaches generally address this issue through data-centric fine-tuning or innovative decoding strategies, these methods often require substantial resources or task-specific configurations. In this work, we introduce an architecture-level solution, HalluRNN, which enhances model stability through recurrent cross-layer reasoning. Specifically, we propose a novel Dual-Gated Depth Propagation Unit (DG-DPU) module, which is shared across layers and recurrently refines hidden states. This allows for the adaptive propagation of information throughout the model, enforces consistency across layers, and mitigates hallucinations caused by representational drift. By fine-tuning only the DG-DPU module, HalluRNN achieves strong and robust performance across multiple benchmarks.
>
---
#### [new 140] PostAlign: Multimodal Grounding as a Corrective Lens for MLLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决MLLMs依赖语言偏见、忽视视觉信息的问题。提出PostAlign框架，通过多模态对齐和消歧机制提升视觉理解与减少幻觉。**

- **链接: [http://arxiv.org/pdf/2506.17901v1](http://arxiv.org/pdf/2506.17901v1)**

> **作者:** Yixuan Wu; Yang Zhang; Jian Wu; Philip Torr; Jindong Gu
>
> **摘要:** Multimodal Large Language Models (MLLMs) excel in vision-language tasks, such as image captioning and visual question answering. However, they often suffer from over-reliance on spurious correlations, primarily due to linguistic priors that distract the model from leveraging actual visual information. To address these issues, we introduce MMGrounded-PostAlign, a post-multimodal alignment framework designed to enhance the visual understanding capabilities and mitigate the hallucinations of MLLMs. Our framework incorporates a multimodal grounding module for both visual grounding, which identifies the referred object in the image, and textual grounding, which generates the rationale for the final answer, ensuring that outputs are anchored in both visual and textual evidence. To mitigate the hallucinations, we introduce a negative rejection mechanism in the visual grounding module to distinguish grounded entities from non-existent objects influenced by linguistic biases. On the textual grounding side, we propose a selective reasoning mechanism that adjusts the model's reasoning strategy based on query complexity. Extensive evaluations are conducted on benchmarks such as POPE, HaloQuest, VQAv2, MME, and MMBench showing significant improvements in fine-grained visual understanding and hallucination suppression.
>
---
#### [new 141] Referring Expression Instance Retrieval and A Strong End-to-End Baseline
- **分类: cs.CV**

- **简介: 该论文提出REIR任务，解决跨大图库的实例检索与定位问题。构建了REIRCOCO数据集，设计CLARE模型实现端到端优化。**

- **链接: [http://arxiv.org/pdf/2506.18246v1](http://arxiv.org/pdf/2506.18246v1)**

> **作者:** Xiangzhao Hao; Kuan Zhu; Hongyu Guo; Haiyun Guo; Ming Tang; JinQiao Wang
>
> **摘要:** Natural language querying of visual content underpins many vision-language tasks, typically categorized by text granularity and visual search scope. Text-Image Retrieval (TIR) retrieves whole images using coarse descriptions, while Referring Expression Comprehension (REC) localizes objects using fine-grained expressions within a single image. However, real-world scenarios often require both instance-level retrieval and localization across large galleries -- tasks where TIR lacks precision and REC lacks scalability. To address this gap, we propose a new task: Referring Expression Instance Retrieval (REIR), which jointly supports instance-level retrieval and localization. We introduce REIRCOCO, a large-scale benchmark constructed by prompting vision-language models to generate fine-grained expressions for MSCOCO and RefCOCO instances. We also present a baseline method, CLARE, featuring a dual-stream architecture with a Mix of Relation Experts (MORE) module for capturing inter-instance relationships. CLARE integrates object detection and REC pretraining with Contrastive Language-Instance Alignment (CLIA) for end-to-end optimization. Experiments show that CLARE achieves state-of-the-art performance on REIR and generalizes well to TIR and REC, highlighting its effectiveness and versatility.
>
---
#### [new 142] Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D实例分割任务，旨在解决点云中对象实例的精确分割问题。通过引入关系建模方法提升特征表示与注意力机制，提高分割性能。**

- **链接: [http://arxiv.org/pdf/2506.17891v1](http://arxiv.org/pdf/2506.17891v1)**

> **作者:** Jiahao Lu; Jiacheng Deng
>
> **备注:** Accepted by CVPR 2025. Code: https://github.com/Howard-coder191/Relation3D
>
> **摘要:** 3D instance segmentation aims to predict a set of object instances in a scene, representing them as binary foreground masks with corresponding semantic labels. Currently, transformer-based methods are gaining increasing attention due to their elegant pipelines and superior predictions. However, these methods primarily focus on modeling the external relationships between scene features and query features through mask attention. They lack effective modeling of the internal relationships among scene features as well as between query features. In light of these disadvantages, we propose \textbf{Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation}. Specifically, we introduce an adaptive superpoint aggregation module and a contrastive learning-guided superpoint refinement module to better represent superpoint features (scene features) and leverage contrastive learning to guide the updates of these features. Furthermore, our relation-aware self-attention mechanism enhances the capabilities of modeling relationships between queries by incorporating positional and geometric relationships into the self-attention mechanism. Extensive experiments on the ScanNetV2, ScanNet++, ScanNet200 and S3DIS datasets demonstrate the superior performance of Relation3D.
>
---
#### [new 143] MDSAM:Memory-Driven Sparse Attention Matrix for LVLMs Hallucination Mitigation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型生成中的幻觉问题。提出MDSAM方法，通过动态调整注意力机制减少幻觉，提升可靠性。**

- **链接: [http://arxiv.org/pdf/2506.17664v1](http://arxiv.org/pdf/2506.17664v1)**

> **作者:** Shuaiye Lu; Linjiang Zhou; Xiaochuan Shi
>
> **摘要:** Hallucinations in large vision-language models (LVLMs) often stem from the model's sensitivity to image tokens during decoding, as evidenced by attention peaks observed when generating both real and hallucinated entities. To address this, we propose Memory-Driven Sparse Attention Matrix (MDSAM) , a novel training-free approach that dynamically captures and refines the attention allocated to image tokens at each layer. MDSAM memorizes attention patterns and activates updates through alignment during decoding, enhancing focus on relevant image tokens while effectively reducing hallucinations. We evaluate MDSAM on multiple benchmarks for tasks such as image captioning and visual question answering, demonstrating its ability to consistently reduce hallucinations and improve reliability. Compatible with various LVLM architectures, MDSAM highlights its adaptability and effectiveness in mitigating hallucinations without requiring additional training or external tools.
>
---
#### [new 144] Attention-Based Ensemble Learning for Crop Classification Using Landsat 8-9 Fusion
- **分类: cs.CV**

- **简介: 该论文属于作物分类任务，旨在提高灌溉区作物识别精度。通过融合Landsat 8-9数据并结合深度学习方法进行分类建模。**

- **链接: [http://arxiv.org/pdf/2506.18321v1](http://arxiv.org/pdf/2506.18321v1)**

> **作者:** Zeeshan Ramzan; Nisar Ahmed; Qurat-ul-Ain Akram; Shahzad Asif; Muhammad Shahbaz; Rabin Chakrabortty; Ahmed F. Elaksher
>
> **备注:** Under review in Earth Systems and Environment
>
> **摘要:** Remote sensing offers a highly effective method for obtaining accurate information on total cropped area and crop types. The study focuses on crop cover identification for irrigated regions of Central Punjab. Data collection was executed in two stages: the first involved identifying and geocoding six target crops through field surveys conducted in January and February 2023. The second stage involved acquiring Landsat 8-9 imagery for each geocoded field to construct a labelled dataset. The satellite imagery underwent extensive pre-processing, including radiometric calibration for reflectance values, atmospheric correction, and georeferencing verification to ensure consistency within a common coordinate system. Subsequently, image fusion techniques were applied to combine Landsat 8 and 9 spectral bands, creating a composite image with enhanced spectral information, followed by contrast enhancement. During data acquisition, farmers were interviewed, and fields were meticulously mapped using GPS instruments, resulting in a comprehensive dataset of 50,835 data points. This dataset facilitated the extraction of vegetation indices such as NDVI, SAVO, RECI, and NDRE. These indices and raw reflectance values were utilized for classification modeling using conventional classifiers, ensemble learning, and artificial neural networks. A feature selection approach was also incorporated to identify the optimal feature set for classification learning. This study demonstrates the effectiveness of combining remote sensing data and advanced modeling techniques to improve crop classification accuracy in irrigated agricultural regions.
>
---
#### [new 145] VQ-Insight: Teaching VLMs for AI-Generated Video Quality Understanding via Progressive Visual Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于AI生成视频质量评估任务，旨在解决现有方法依赖标注数据、缺乏时序感知等问题。提出VQ-Insight框架，结合多维度奖励与渐进学习提升视频质量评估效果。**

- **链接: [http://arxiv.org/pdf/2506.18564v1](http://arxiv.org/pdf/2506.18564v1)**

> **作者:** Xuanyu Zhang; Weiqi Li; Shijie Zhao; Junlin Li; Li Zhang; Jian Zhang
>
> **备注:** Technical Report
>
> **摘要:** Recent advances in AI-generated content (AIGC) have led to the emergence of powerful text-to-video generation models. Despite these successes, evaluating the quality of AIGC-generated videos remains challenging due to limited generalization, lack of temporal awareness, heavy reliance on large-scale annotated datasets, and the lack of effective interaction with generation models. Most current approaches rely on supervised finetuning of vision-language models (VLMs), which often require large-scale annotated datasets and tend to decouple understanding and generation. To address these shortcomings, we propose VQ-Insight, a novel reasoning-style VLM framework for AIGC video quality assessment. Our approach features: (1) a progressive video quality learning scheme that combines image quality warm-up, general task-specific temporal learning, and joint optimization with the video generation model; (2) the design of multi-dimension scoring rewards, preference comparison rewards, and temporal modeling rewards to enhance both generalization and specialization in video quality evaluation. Extensive experiments demonstrate that VQ-Insight consistently outperforms state-of-the-art baselines in preference comparison, multi-dimension scoring, and natural video scoring, bringing significant improvements for video generation tasks.
>
---
#### [new 146] ReFrame: Rectification Framework for Image Explaining Architectures
- **分类: cs.CV**

- **简介: 该论文属于图像解释任务，旨在解决对象幻觉和不完整问题。提出ReFrame框架，提升图像描述、VQA和提示模型的解释准确性。**

- **链接: [http://arxiv.org/pdf/2506.18272v1](http://arxiv.org/pdf/2506.18272v1)**

> **作者:** Debjyoti Das Adhikary; Aritra Hazra; Partha Pratim Chakrabarti
>
> **备注:** Accepted in CODS-COMAD December 2024
>
> **摘要:** Image explanation has been one of the key research interests in the Deep Learning field. Throughout the years, several approaches have been adopted to explain an input image fed by the user. From detecting an object in a given image to explaining it in human understandable sentence, to having a conversation describing the image, this problem has seen an immense change throughout the years, However, the existing works have been often found to (a) hallucinate objects that do not exist in the image and/or (b) lack identifying the complete set of objects present in the image. In this paper, we propose a novel approach to mitigate these drawbacks of inconsistency and incompleteness of the objects recognized during the image explanation. To enable this, we propose an interpretable framework that can be plugged atop diverse image explaining frameworks including Image Captioning, Visual Question Answering (VQA) and Prompt-based AI using LLMs, thereby enhancing their explanation capabilities by rectifying the incorrect or missing objects. We further measure the efficacy of the rectified explanations generated through our proposed approaches leveraging object based precision metrics, and showcase the improvements in the inconsistency and completeness of image explanations. Quantitatively, the proposed framework is able to improve the explanations over the baseline architectures of Image Captioning (improving the completeness by 81.81% and inconsistency by 37.10%), Visual Question Answering(average of 9.6% and 37.10% in completeness and inconsistency respectively) and Prompt-based AI model (0.01% and 5.2% for completeness and inconsistency respectively) surpassing the current state-of-the-art by a substantial margin.
>
---
#### [new 147] LLM-driven Medical Report Generation via Communication-efficient Heterogeneous Federated Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医疗报告生成任务，解决多中心数据隐私和通信效率问题。提出FedMRG框架，通过联邦学习实现高效、隐私保护的模型训练。**

- **链接: [http://arxiv.org/pdf/2506.17562v1](http://arxiv.org/pdf/2506.17562v1)**

> **作者:** Haoxuan Che; Haibo Jin; Zhengrui Guo; Yi Lin; Cheng Jin; Hao Chen
>
> **摘要:** LLMs have demonstrated significant potential in Medical Report Generation (MRG), yet their development requires large amounts of medical image-report pairs, which are commonly scattered across multiple centers. Centralizing these data is exceptionally challenging due to privacy regulations, thereby impeding model development and broader adoption of LLM-driven MRG models. To address this challenge, we present FedMRG, the first framework that leverages Federated Learning (FL) to enable privacy-preserving, multi-center development of LLM-driven MRG models, specifically designed to overcome the critical challenge of communication-efficient LLM training under multi-modal data heterogeneity. To start with, our framework tackles the fundamental challenge of communication overhead in FL-LLM tuning by employing low-rank factorization to efficiently decompose parameter updates, significantly reducing gradient transmission costs and making LLM-driven MRG feasible in bandwidth-constrained FL settings. Furthermore, we observed the dual heterogeneity in MRG under the FL scenario: varying image characteristics across medical centers, as well as diverse reporting styles and terminology preferences. To address this, we further enhance FedMRG with (1) client-aware contrastive learning in the MRG encoder, coupled with diagnosis-driven prompts, which capture both globally generalizable and locally distinctive features while maintaining diagnostic accuracy; and (2) a dual-adapter mutual boosting mechanism in the MRG decoder that harmonizes generic and specialized adapters to address variations in reporting styles and terminology. Through extensive evaluation of our established FL-MRG benchmark, we demonstrate the generalizability and adaptability of FedMRG, underscoring its potential in harnessing multi-center data and generating clinically accurate reports while maintaining communication efficiency.
>
---
#### [new 148] MiCo: Multiple Instance Learning with Context-Aware Clustering for Whole Slide Image Analysis
- **分类: cs.CV**

- **简介: 该论文属于病理图像分析任务，解决WSI中空间异质性带来的挑战，提出MiCo框架增强组织间关联与语义一致性。**

- **链接: [http://arxiv.org/pdf/2506.18028v1](http://arxiv.org/pdf/2506.18028v1)**

> **作者:** Junjian Li; Hulin Kuang; Jin Liu; Hailin Yue; Mengshen He; Jianxin Wang
>
> **备注:** MICCAI 2025
>
> **摘要:** Multiple instance learning (MIL) has shown significant promise in histopathology whole slide image (WSI) analysis for cancer diagnosis and prognosis. However, the inherent spatial heterogeneity of WSIs presents critical challenges, as morphologically similar tissue types are often dispersed across distant anatomical regions. Conventional MIL methods struggle to model these scattered tissue distributions and capture cross-regional spatial interactions effectively. To address these limitations, we propose a novel Multiple instance learning framework with Context-Aware Clustering (MiCo), designed to enhance cross-regional intra-tissue correlations and strengthen inter-tissue semantic associations in WSIs. MiCo begins by clustering instances to distill discriminative morphological patterns, with cluster centroids serving as semantic anchors. To enhance cross-regional intra-tissue correlations, MiCo employs a Cluster Route module, which dynamically links instances of the same tissue type across distant regions via feature similarity. These semantic anchors act as contextual hubs, propagating semantic relationships to refine instance-level representations. To eliminate semantic fragmentation and strengthen inter-tissue semantic associations, MiCo integrates a Cluster Reducer module, which consolidates redundant anchors while enhancing information exchange between distinct semantic groups. Extensive experiments on two challenging tasks across nine large-scale public cancer datasets demonstrate the effectiveness of MiCo, showcasing its superiority over state-of-the-art methods. The code is available at https://github.com/junjianli106/MiCo.
>
---
#### [new 149] Multi-Scale Representation of Follicular Lymphoma Pathology Images in a Single Hyperbolic Space
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决淋巴瘤病理图像多尺度表示问题。通过自监督学习在单个双曲空间中嵌入组织与细胞核图像，捕捉疾病状态和细胞类型变化。**

- **链接: [http://arxiv.org/pdf/2506.18523v1](http://arxiv.org/pdf/2506.18523v1)**

> **作者:** Kei Taguchi; Kazumasa Ohara; Tatsuya Yokota; Hiroaki Miyoshi; Noriaki Hashimoto; Ichiro Takeuchi; Hidekata Hontani
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** We propose a method for representing malignant lymphoma pathology images, from high-resolution cell nuclei to low-resolution tissue images, within a single hyperbolic space using self-supervised learning. To capture morphological changes that occur across scales during disease progression, our approach embeds tissue and corresponding nucleus images close to each other based on inclusion relationships. Using the Poincar\'e ball as the feature space enables effective encoding of this hierarchical structure. The learned representations capture both disease state and cell type variations.
>
---
#### [new 150] USVTrack: USV-Based 4D Radar-Camera Tracking Dataset for Autonomous Driving in Inland Waterways
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于水路自动驾驶中的目标跟踪任务，旨在提升复杂水环境下的跟踪精度。通过USV采集4D雷达与相机数据，构建了USVTrack数据集，并提出一种有效的雷达-相机匹配方法RCM。**

- **链接: [http://arxiv.org/pdf/2506.18737v1](http://arxiv.org/pdf/2506.18737v1)**

> **作者:** Shanliang Yao; Runwei Guan; Yi Ni; Sen Xu; Yong Yue; Xiaohui Zhu; Ryan Wen Liu
>
> **备注:** Accepted by IROS
>
> **摘要:** Object tracking in inland waterways plays a crucial role in safe and cost-effective applications, including waterborne transportation, sightseeing tours, environmental monitoring and surface rescue. Our Unmanned Surface Vehicle (USV), equipped with a 4D radar, a monocular camera, a GPS, and an IMU, delivers robust tracking capabilities in complex waterborne environments. By leveraging these sensors, our USV collected comprehensive object tracking data, which we present as USVTrack, the first 4D radar-camera tracking dataset tailored for autonomous driving in new generation waterborne transportation systems. Our USVTrack dataset presents rich scenarios, featuring diverse various waterways, varying times of day, and multiple weather and lighting conditions. Moreover, we present a simple but effective radar-camera matching method, termed RCM, which can be plugged into popular two-stage association trackers. Experimental results utilizing RCM demonstrate the effectiveness of the radar-camera matching in improving object tracking accuracy and reliability for autonomous driving in waterborne environments. The USVTrack dataset is public on https://usvtrack.github.io.
>
---
#### [new 151] StainPIDR: A Pathological Image Decouplingand Reconstruction Method for StainNormalization Based on Color VectorQuantization and Structure Restaining
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于病理图像染色归一化任务，旨在解决颜色差异导致的诊断系统性能下降问题。通过解耦结构与颜色特征并进行重染，实现图像归一化。**

- **链接: [http://arxiv.org/pdf/2506.17879v1](http://arxiv.org/pdf/2506.17879v1)**

> **作者:** Zheng Chen
>
> **摘要:** The color appearance of a pathological image is highly related to the imaging protocols, the proportion of different dyes, and the scanning devices. Computer-aided diagnostic systems may deteriorate when facing these color-variant pathological images. In this work, we propose a stain normalization method called StainPIDR. We try to eliminate this color discrepancy by decoupling the image into structure features and vector-quantized color features, restaining the structure features with the target color features, and decoding the stained structure features to normalized pathological images. We assume that color features decoupled by different images with the same color should be exactly the same. Under this assumption, we train a fixed color vector codebook to which the decoupled color features will map. In the restaining part, we utilize the cross-attention mechanism to efficiently stain the structure features. As the target color (decoupled from a selected template image) will also affect the performance of stain normalization, we further design a template image selection algorithm to select a template from a given dataset. In our extensive experiments, we validate the effectiveness of StainPIDR and the template image selection algorithm. All the results show that our method can perform well in the stain normalization task. The code of StainPIDR will be publicly available later.
>
---
#### [new 152] VMRA-MaR: An Asymmetry-Aware Temporal Framework for Longitudinal Breast Cancer Risk Prediction
- **分类: cs.CV**

- **简介: 该论文属于乳腺癌风险预测任务，旨在利用时间序列影像数据提升长期风险预测准确性。通过引入VMRNN和不对称模块，有效捕捉乳腺组织变化趋势。**

- **链接: [http://arxiv.org/pdf/2506.17412v1](http://arxiv.org/pdf/2506.17412v1)**

> **作者:** Zijun Sun; Solveig Thrun; Michael Kampffmeyer
>
> **备注:** MICCAI 2025, Provisional Accept
>
> **摘要:** Breast cancer remains a leading cause of mortality worldwide and is typically detected via screening programs where healthy people are invited in regular intervals. Automated risk prediction approaches have the potential to improve this process by facilitating dynamically screening of high-risk groups. While most models focus solely on the most recent screening, there is growing interest in exploiting temporal information to capture evolving trends in breast tissue, as inspired by clinical practice. Early methods typically relied on two time steps, and although recent efforts have extended this to multiple time steps using Transformer architectures, challenges remain in fully harnessing the rich temporal dynamics inherent in longitudinal imaging data. In this work, we propose to instead leverage Vision Mamba RNN (VMRNN) with a state-space model (SSM) and LSTM-like memory mechanisms to effectively capture nuanced trends in breast tissue evolution. To further enhance our approach, we incorporate an asymmetry module that utilizes a Spatial Asymmetry Detector (SAD) and Longitudinal Asymmetry Tracker (LAT) to identify clinically relevant bilateral differences. This integrated framework demonstrates notable improvements in predicting cancer onset, especially for the more challenging high-density breast cases and achieves superior performance at extended time points (years four and five), highlighting its potential to advance early breast cancer recognition and enable more personalized screening strategies. Our code is available at https://github.com/Mortal-Suen/VMRA-MaR.git.
>
---
#### [new 153] SSAVSV: Towards Unified Model for Self-Supervised Audio-Visual Speaker Verification
- **分类: cs.CV**

- **简介: 该论文属于音频-视觉说话人验证任务，旨在解决传统方法依赖大量标注数据和计算成本高的问题。提出一种统一的自监督学习框架，使用共享主干网络提升效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.17694v1](http://arxiv.org/pdf/2506.17694v1)**

> **作者:** Gnana Praveen Rajasekhar; Jahangir Alam
>
> **摘要:** Conventional audio-visual methods for speaker verification rely on large amounts of labeled data and separate modality-specific architectures, which is computationally expensive, limiting their scalability. To address these problems, we propose a self-supervised learning framework based on contrastive learning with asymmetric masking and masked data modeling to obtain robust audiovisual feature representations. In particular, we employ a unified framework for self-supervised audiovisual speaker verification using a single shared backbone for audio and visual inputs, leveraging the versatility of vision transformers. The proposed unified framework can handle audio, visual, or audiovisual inputs using a single shared vision transformer backbone during training and testing while being computationally efficient and robust to missing modalities. Extensive experiments demonstrate that our method achieves competitive performance without labeled data while reducing computational costs compared to traditional approaches.
>
---
#### [new 154] Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于对抗攻击任务，旨在提升生成式对抗样本的迁移性。通过利用生成器的语义结构信息，增强扰动与目标区域的对齐，从而提高攻击效果。**

- **链接: [http://arxiv.org/pdf/2506.18248v1](http://arxiv.org/pdf/2506.18248v1)**

> **作者:** Jongoh Jeong; Hunmin Yang; Jaeseok Jeong; Kuk-Jin Yoon
>
> **摘要:** Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).
>
---
#### [new 155] A Multi-Scale Spatial Attention-Based Zero-Shot Learning Framework for Low-Light Image Enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于低光图像增强任务，解决无配对数据下的图像增强问题。提出LucentVisionNet框架，结合多尺度空间注意力和深度曲线估计，提升图像质量与结构一致性。**

- **链接: [http://arxiv.org/pdf/2506.18323v1](http://arxiv.org/pdf/2506.18323v1)**

> **作者:** Muhammad Azeem Aslam; Hassan Khalid; Nisar Ahmed
>
> **摘要:** Low-light image enhancement remains a challenging task, particularly in the absence of paired training data. In this study, we present LucentVisionNet, a novel zero-shot learning framework that addresses the limitations of traditional and deep learning-based enhancement methods. The proposed approach integrates multi-scale spatial attention with a deep curve estimation network, enabling fine-grained enhancement while preserving semantic and perceptual fidelity. To further improve generalization, we adopt a recurrent enhancement strategy and optimize the model using a composite loss function comprising six tailored components, including a novel no-reference image quality loss inspired by human visual perception. Extensive experiments on both paired and unpaired benchmark datasets demonstrate that LucentVisionNet consistently outperforms state-of-the-art supervised, unsupervised, and zero-shot methods across multiple full-reference and no-reference image quality metrics. Our framework achieves high visual quality, structural consistency, and computational efficiency, making it well-suited for deployment in real-world applications such as mobile photography, surveillance, and autonomous navigation.
>
---
#### [new 156] Benchmarking histopathology foundation models in a multi-center dataset for skin cancer subtyping
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于皮肤癌亚型分类任务，旨在评估病理学基础模型在多中心数据集中的表现，并提出新指标FM-SI以衡量模型一致性。**

- **链接: [http://arxiv.org/pdf/2506.18668v1](http://arxiv.org/pdf/2506.18668v1)**

> **作者:** Pablo Meseguer; Rocío del Amor; Valery Naranjo
>
> **备注:** Accepeted for oral presentation at Medical Image Understanding and Analysis (MIUA) 2025
>
> **摘要:** Pretraining on large-scale, in-domain datasets grants histopathology foundation models (FM) the ability to learn task-agnostic data representations, enhancing transfer learning on downstream tasks. In computational pathology, automated whole slide image analysis requires multiple instance learning (MIL) frameworks due to the gigapixel scale of the slides. The diversity among histopathology FMs has highlighted the need to design real-world challenges for evaluating their effectiveness. To bridge this gap, our work presents a novel benchmark for evaluating histopathology FMs as patch-level feature extractors within a MIL classification framework. For that purpose, we leverage the AI4SkIN dataset, a multi-center cohort encompassing slides with challenging cutaneous spindle cell neoplasm subtypes. We also define the Foundation Model - Silhouette Index (FM-SI), a novel metric to measure model consistency against distribution shifts. Our experimentation shows that extracting less biased features enhances classification performance, especially in similarity-based MIL classifiers.
>
---
#### [new 157] Context Consistency Learning via Sentence Removal for Semi-Supervised Video Paragraph Grounding
- **分类: cs.CV**

- **简介: 该论文属于视频段落定位任务，解决半监督学习中监督信号不足的问题。提出CCL框架，通过上下文一致性学习增强模型性能。**

- **链接: [http://arxiv.org/pdf/2506.18476v1](http://arxiv.org/pdf/2506.18476v1)**

> **作者:** Yaokun Zhong; Siyu Jiang; Jian Zhu; Jian-Fang Hu
>
> **备注:** Accepted by ICME2025
>
> **摘要:** Semi-Supervised Video Paragraph Grounding (SSVPG) aims to localize multiple sentences in a paragraph from an untrimmed video with limited temporal annotations. Existing methods focus on teacher-student consistency learning and video-level contrastive loss, but they overlook the importance of perturbing query contexts to generate strong supervisory signals. In this work, we propose a novel Context Consistency Learning (CCL) framework that unifies the paradigms of consistency regularization and pseudo-labeling to enhance semi-supervised learning. Specifically, we first conduct teacher-student learning where the student model takes as inputs strongly-augmented samples with sentences removed and is enforced to learn from the adequately strong supervisory signals from the teacher model. Afterward, we conduct model retraining based on the generated pseudo labels, where the mutual agreement between the original and augmented views' predictions is utilized as the label confidence. Extensive experiments show that CCL outperforms existing methods by a large margin.
>
---
#### [new 158] Domain Generalization using Action Sequences for Egocentric Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于 egocentric action recognition 任务，旨在提升模型在未见环境中的泛化能力。通过引入动作序列重建和混合训练方法，增强模型的跨域识别性能。**

- **链接: [http://arxiv.org/pdf/2506.17685v1](http://arxiv.org/pdf/2506.17685v1)**

> **作者:** Amirshayan Nasirimajd; Chiara Plizzari; Simone Alberto Peirone; Marco Ciccone; Giuseppe Averta; Barbara Caputo
>
> **备注:** Accepted at Pattern Recognition Letters. 9 pages including references. Code and Data: https://github.com/Ashayan97/SeqDG
>
> **摘要:** Recognizing human activities from visual inputs, particularly through a first-person viewpoint, is essential for enabling robots to replicate human behavior. Egocentric vision, characterized by cameras worn by observers, captures diverse changes in illumination, viewpoint, and environment. This variability leads to a notable drop in the performance of Egocentric Action Recognition models when tested in environments not seen during training. In this paper, we tackle these challenges by proposing a domain generalization approach for Egocentric Action Recognition. Our insight is that action sequences often reflect consistent user intent across visual domains. By leveraging action sequences, we aim to enhance the model's generalization ability across unseen environments. Our proposed method, named SeqDG, introduces a visual-text sequence reconstruction objective (SeqRec) that uses contextual cues from both text and visual inputs to reconstruct the central action of the sequence. Additionally, we enhance the model's robustness by training it on mixed sequences of actions from different domains (SeqMix). We validate SeqDG on the EGTEA and EPIC-KITCHENS-100 datasets. Results on EPIC-KITCHENS-100, show that SeqDG leads to +2.4% relative average improvement in cross-domain action recognition in unseen environments, and on EGTEA the model achieved +0.6% Top-1 accuracy over SOTA in intra-domain action recognition.
>
---
#### [new 159] DExNet: Combining Observations of Domain Adapted Critics for Leaf Disease Classification with Limited Data
- **分类: cs.CV**

- **简介: 该论文属于植物病害分类任务，旨在解决小样本下分类性能差的问题。提出DExNet框架，通过融合多个预训练模型的特征提升分类准确率。**

- **链接: [http://arxiv.org/pdf/2506.18173v1](http://arxiv.org/pdf/2506.18173v1)**

> **作者:** Sabbir Ahmed; Md. Bakhtiar Hasan; Tasnim Ahmed; Md. Hasanul Kabir
>
> **备注:** Submitted to ACPR Springer, 15 pages, 1 Figure, 7 Tables, and lots of efforts :)
>
> **摘要:** While deep learning-based architectures have been widely used for correctly detecting and classifying plant diseases, they require large-scale datasets to learn generalized features and achieve state-of-the-art performance. This poses a challenge for such models to obtain satisfactory performance in classifying leaf diseases with limited samples. This work proposes a few-shot learning framework, Domain-adapted Expert Network (DExNet), for plant disease classification that compensates for the lack of sufficient training data by combining observations of a number of expert critics. It starts with extracting the feature embeddings as 'observations' from nine 'critics' that are state-of-the-art pre-trained CNN-based architectures. These critics are 'domain adapted' using a publicly available leaf disease dataset having no overlapping classes with the specific downstream task of interest. The observations are then passed to the 'Feature Fusion Block' and finally to a classifier network consisting of Bi-LSTM layers. The proposed pipeline is evaluated on the 10 classes of tomato leaf images from the PlantVillage dataset, achieving promising accuracies of 89.06%, 92.46%, and 94.07%, respectively, for 5-shot, 10-shot, and 15-shot classification. Furthermore, an accuracy of 98.09+-0.7% has been achieved in 80-shot classification, which is only 1.2% less than state-of-the-art, allowing a 94.5% reduction in the training data requirement. The proposed pipeline also outperforms existing works on leaf disease classification with limited data in both laboratory and real-life conditions in single-domain, mixed-domain, and cross-domain scenarios.
>
---
#### [new 160] SIM-Net: A Multimodal Fusion Network Using Inferred 3D Object Shape Point Clouds from RGB Images for 2D Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于2D图像分类任务，旨在解决背景复杂、遮挡等问题。通过融合从RGB图像推断的3D点云，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2506.18683v1](http://arxiv.org/pdf/2506.18683v1)**

> **作者:** Youcef Sklab; Hanane Ariouat; Eric Chenin; Edi Prifti; Jean-Daniel Zucker
>
> **备注:** 25 pages, 9 figures, 14 tables
>
> **摘要:** We introduce the Shape-Image Multimodal Network (SIM-Net), a novel 2D image classification architecture that integrates 3D point cloud representations inferred directly from RGB images. Our key contribution lies in a pixel-to-point transformation that converts 2D object masks into 3D point clouds, enabling the fusion of texture-based and geometric features for enhanced classification performance. SIM-Net is particularly well-suited for the classification of digitized herbarium specimens (a task made challenging by heterogeneous backgrounds), non-plant elements, and occlusions that compromise conventional image-based models. To address these issues, SIM-Net employs a segmentation-based preprocessing step to extract object masks prior to 3D point cloud generation. The architecture comprises a CNN encoder for 2D image features and a PointNet-based encoder for geometric features, which are fused into a unified latent space. Experimental evaluations on herbarium datasets demonstrate that SIM-Net consistently outperforms ResNet101, achieving gains of up to 9.9% in accuracy and 12.3% in F-score. It also surpasses several transformer-based state-of-the-art architectures, highlighting the benefits of incorporating 3D structural reasoning into 2D image classification tasks.
>
---
#### [new 161] JarvisArt: Liberating Human Artistic Creativity via an Intelligent Photo Retouching Agent
- **分类: cs.CV**

- **简介: 该论文提出 JarvisArt，一个基于多模态大语言模型的智能照片修图代理，解决传统工具专业性强、AI方案调整有限的问题，实现更自然和精准的图像编辑。**

- **链接: [http://arxiv.org/pdf/2506.17612v1](http://arxiv.org/pdf/2506.17612v1)**

> **作者:** Yunlong Lin; Zixu Lin; Kunjie Lin; Jinbin Bai; Panwang Pan; Chenxin Li; Haoyu Chen; Zhongdao Wang; Xinghao Ding; Wenbo Li; Shuicheng Yan
>
> **备注:** 40 pages, 26 figures
>
> **摘要:** Photo retouching has become integral to contemporary visual storytelling, enabling users to capture aesthetics and express creativity. While professional tools such as Adobe Lightroom offer powerful capabilities, they demand substantial expertise and manual effort. In contrast, existing AI-based solutions provide automation but often suffer from limited adjustability and poor generalization, failing to meet diverse and personalized editing needs. To bridge this gap, we introduce JarvisArt, a multi-modal large language model (MLLM)-driven agent that understands user intent, mimics the reasoning process of professional artists, and intelligently coordinates over 200 retouching tools within Lightroom. JarvisArt undergoes a two-stage training process: an initial Chain-of-Thought supervised fine-tuning to establish basic reasoning and tool-use skills, followed by Group Relative Policy Optimization for Retouching (GRPO-R) to further enhance its decision-making and tool proficiency. We also propose the Agent-to-Lightroom Protocol to facilitate seamless integration with Lightroom. To evaluate performance, we develop MMArt-Bench, a novel benchmark constructed from real-world user edits. JarvisArt demonstrates user-friendly interaction, superior generalization, and fine-grained control over both global and local adjustments, paving a new avenue for intelligent photo retouching. Notably, it outperforms GPT-4o with a 60% improvement in average pixel-level metrics on MMArt-Bench for content fidelity, while maintaining comparable instruction-following capabilities. Project Page: https://jarvisart.vercel.app/.
>
---
#### [new 162] 2D Triangle Splatting for Direct Differentiable Mesh Training
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决传统方法在渲染速度和效果上的不足。通过引入2D三角形贴图，实现高效且高质量的网格训练。**

- **链接: [http://arxiv.org/pdf/2506.18575v1](http://arxiv.org/pdf/2506.18575v1)**

> **作者:** Kaifeng Sheng; Zheng Zhou; Yingliang Peng; Qianwei Wang
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Differentiable rendering with 3D Gaussian primitives has emerged as a powerful method for reconstructing high-fidelity 3D scenes from multi-view images. While it offers improvements over NeRF-based methods, this representation still encounters challenges with rendering speed and advanced rendering effects, such as relighting and shadow rendering, compared to mesh-based models. In this paper, we propose 2D Triangle Splatting (2DTS), a novel method that replaces 3D Gaussian primitives with 2D triangle facelets. This representation naturally forms a discrete mesh-like structure while retaining the benefits of continuous volumetric modeling. By incorporating a compactness parameter into the triangle primitives, we enable direct training of photorealistic meshes. Our experimental results demonstrate that our triangle-based method, in its vanilla version (without compactness tuning), achieves higher fidelity compared to state-of-the-art Gaussian-based methods. Furthermore, our approach produces reconstructed meshes with superior visual quality compared to existing mesh reconstruction methods.
>
---
#### [new 163] InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出InternSpatial数据集及评估基准，用于提升视觉语言模型的空间推理能力，解决现有数据规模小、多样性不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.18385v1](http://arxiv.org/pdf/2506.18385v1)**

> **作者:** Nianchen Deng; Lixin Gu; Shenglong Ye; Yinan He; Zhe Chen; Songze Li; Haomin Wang; Xingguang Wei; Tianshuo Yang; Min Dou; Tong He; Wenqi Shao; Kaipeng Zhang; Yi Wang; Botian Shi; Yanting Zhang; Jifeng Dai; Yu Qiao; Hongjie Zhang; Wenhai Wang
>
> **摘要:** Recent benchmarks and datasets have been proposed to improve spatial reasoning in vision-language models (VLMs), yet existing open resources remain limited in scale, visual diversity, and instruction expressiveness. In this work, we introduce InternSpatial, the largest open-source dataset for spatial reasoning in VLMs, along with InternSpatial-Bench, a corresponding evaluation benchmark designed to assess spatial understanding under diverse instruction formats. InternSpatial comprises 12 million QA pairs spanning both single-view and multi-view settings, drawn from diverse visual environments and supporting 19 instruction formats that reflect varied query styles. For evaluation, we propose InternSpatial-Bench for single-view tasks and expand multi-view reasoning by introducing a novel rotation angle prediction task that has not been explored in prior work. Experimental results show that models trained on InternSpatial achieve 12.1% improvement on InternSpatial-Bench and 10.7% on VSI-Bench, while maintaining strong performance on general-purpose benchmarks. We hope these resources will support the development of spatially capable VLMs in practical applications such as robotics and embodied AI.
>
---
#### [new 164] Optimization-Free Patch Attack on Stereo Depth Estimation
- **分类: cs.CV**

- **简介: 该论文属于立体深度估计的对抗攻击任务，旨在设计物理可实现、场景自适应且可迁移的攻击方法。提出PatchHunter，无需优化的强化学习攻击框架，有效提升攻击成功率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.17632v1](http://arxiv.org/pdf/2506.17632v1)**

> **作者:** Hangcheng Liu; Xu Kuang; Xingshuo Han; Xingwan Wu; Haoran Ou; Shangwei Guo; Xingyi Huang; Tao Xiang; Tianwei Zhang
>
> **摘要:** Stereo Depth Estimation (SDE) is essential for scene understanding in vision-based systems like autonomous driving. However, recent studies show that SDE models are vulnerable to adversarial attacks, which are often limited to unrealistic settings, e.g., digital perturbations on separate stereo views in static scenes, restricting their real-world applicability. This raises a critical question: how can we design physically realizable, scene-adaptive, and transferable attacks against SDE under realistic constraints? To answer this, we make two key contributions. First, we propose a unified attack framework that extends optimization-based techniques to four core stages of stereo matching: feature extraction, cost-volume construction, cost aggregation, and disparity regression. A comprehensive stage-wise evaluation across 9 mainstream SDE models, under constraints like photometric consistency, reveals that optimization-based patches suffer from poor transferability. Interestingly, partially transferable patches suggest that patterns, rather than pixel-level perturbations, may be key to generalizable attacks. Motivated by this, we present PatchHunter, the first optimization-free adversarial patch attack against SDE. PatchHunter formulates patch generation as a reinforcement learning-driven search over a structured space of visual patterns crafted to disrupt SDE assumptions. We validate PatchHunter across three levels: the KITTI dataset, the CARLA simulator, and real-world vehicle deployment. PatchHunter not only surpasses optimization-based methods in effectiveness but also achieves significantly better black-box transferability. Even under challenging physical conditions like low light, PatchHunter maintains high attack success (e.g., D1-all > 0.4), whereas optimization-based methods fail.
>
---
#### [new 165] Classification of Tents in Street Bazaars Using CNN
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决街市帐篷自动分类问题。通过对比自定义CNN与EfficientNetB0模型，提升分类准确率。**

- **链接: [http://arxiv.org/pdf/2506.17946v1](http://arxiv.org/pdf/2506.17946v1)**

> **作者:** Azamat Ibragimov; Ruslan Isaev; Remudin Reshid Mekuria; Gulnaz Gimaletdinova; Dim Shaiakhmetov
>
> **摘要:** This research paper proposes an improved deep learning model for classifying tents in street bazaars, comparing a custom Convolutional Neural Network (CNN) with EfficientNetB0. This is a critical task for market organization with a tent classification, but manual methods in the past have been inefficient. Street bazaars represent a vital economic hub in many regions, yet their unstructured nature poses significant challenges for the automated classification of market infrastructure, such as tents. In Kyrgyzstan, more than a quarter of the country's GDP is derived from bazaars. While CNNs have been widely applied to object recognition, their application to bazaar-specific tasks remains underexplored. Here, we build upon our original approach by training on an extended set of 126 original photographs that were augmented to generate additional images. This dataset is publicly available for download on Kaggle. A variety of performance metrics, such as accuracy, precision, recall, F1 score, and mean average precision (mAP), were used to assess the models comparatively, providing a more extensive analysis of classification performance. The results show that the CNN custom model achieved 92.8% accuracy, and EfficientNetB0 showed 98.4% accuracy results, confirming the effectiveness of transfer learning in the bazaar image classification. Also, when analyzing the confusion matrix, the analysis reveals the weaknesses and strengths of each model. These findings suggest that using a pre-trained model such as EfficientNetB0 significantly improves classification accuracy and generalization.
>
---
#### [new 166] Geometry-aware Distance Measure for Diverse Hierarchical Structures in Hyperbolic Spaces
- **分类: cs.CV**

- **简介: 该论文属于机器学习任务，旨在解决超球空间中层次结构建模问题。提出一种自适应距离度量方法，动态适应不同层次结构，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.18533v1](http://arxiv.org/pdf/2506.18533v1)**

> **作者:** Pengxiang Li; Yuwei Wu; Zhi Gao; Xiaomeng Fan; Wei Wu; Zhipeng Lu; Yunde Jia; Mehrtash Harandi
>
> **备注:** 24 pages
>
> **摘要:** Learning in hyperbolic spaces has attracted increasing attention due to its superior ability to model hierarchical structures of data. Most existing hyperbolic learning methods use fixed distance measures for all data, assuming a uniform hierarchy across all data points. However, real-world hierarchical structures exhibit significant diversity, making this assumption overly restrictive. In this paper, we propose a geometry-aware distance measure in hyperbolic spaces, which dynamically adapts to varying hierarchical structures. Our approach derives the distance measure by generating tailored projections and curvatures for each pair of data points, effectively mapping them to an appropriate hyperbolic space. We introduce a revised low-rank decomposition scheme and a hard-pair mining mechanism to mitigate the computational cost of pair-wise distance computation without compromising accuracy. We present an upper bound on the low-rank approximation error using Talagrand's concentration inequality, ensuring theoretical robustness. Extensive experiments on standard image classification (MNIST, CIFAR-10 and CIFAR-100), hierarchical classification (5-level CIFAR-100), and few-shot learning tasks (mini-ImageNet, tiered-ImageNet) demonstrate the effectiveness of our method. Our approach consistently outperforms learning methods that use fixed distance measures, with notable improvements on few-shot learning tasks, where it achieves over 5\% gains on mini-ImageNet. The results reveal that adaptive distance measures better capture diverse hierarchical structures, with visualization showing clearer class boundaries and improved prototype separation in hyperbolic spaces.
>
---
#### [new 167] IDAL: Improved Domain Adaptive Learning for Natural Images Dataset
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于无监督域适应任务，旨在解决自然图像在域间分布差异下的模型泛化问题。通过改进网络结构和损失函数，提升模型在目标域的准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.17931v1](http://arxiv.org/pdf/2506.17931v1)**

> **作者:** Ravi Kant Gupta; Shounak Das; Amit Sethi
>
> **备注:** Accepted in ICPR'24 (International Conference on Pattern Recognition)
>
> **摘要:** We present a novel approach for unsupervised domain adaptation (UDA) for natural images. A commonly-used objective for UDA schemes is to enhance domain alignment in representation space even if there is a domain shift in the input space. Existing adversarial domain adaptation methods may not effectively align different domains of multimodal distributions associated with classification problems. Our approach has two main features. Firstly, its neural architecture uses the deep structure of ResNet and the effective separation of scales of feature pyramidal network (FPN) to work with both content and style features. Secondly, it uses a combination of a novel loss function and judiciously selected existing loss functions to train the network architecture. This tailored combination is designed to address challenges inherent to natural images, such as scale, noise, and style shifts, that occur on top of a multi-modal (multi-class) distribution. The combined loss function not only enhances model accuracy and robustness on the target domain but also speeds up training convergence. Our proposed UDA scheme generalizes better than state-of-the-art for CNN-based methods on Office-Home, Office-31, and VisDA-2017 datasets and comaparable for DomainNet dataset.
>
---
#### [new 168] Fast Neural Inverse Kinematics on Human Body Motions
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人体运动捕捉任务，解决无标记运动捕捉的实时性问题。提出一种快速可靠的神经逆运动学框架，通过网络设计与训练优化实现高效推理。**

- **链接: [http://arxiv.org/pdf/2506.17996v1](http://arxiv.org/pdf/2506.17996v1)**

> **作者:** David Tolpin; Sefy Kagarlitsky
>
> **备注:** Work in progress
>
> **摘要:** Markerless motion capture enables the tracking of human motion without requiring physical markers or suits, offering increased flexibility and reduced costs compared to traditional systems. However, these advantages often come at the expense of higher computational demands and slower inference, limiting their applicability in real-time scenarios. In this technical report, we present a fast and reliable neural inverse kinematics framework designed for real-time capture of human body motions from 3D keypoints. We describe the network architecture, training methodology, and inference procedure in detail. Our framework is evaluated both qualitatively and quantitatively, and we support key design decisions through ablation studies.
>
---
#### [new 169] 4D-LRM: Large Space-Time Reconstruction Model From and To Any View at Any Time
- **分类: cs.CV**

- **简介: 该论文提出4D-LRM，解决从任意视角和时间重建物体的问题。通过大规模时空预训练，实现高效高质的4D重建与渲染。**

- **链接: [http://arxiv.org/pdf/2506.18890v1](http://arxiv.org/pdf/2506.18890v1)**

> **作者:** Ziqiao Ma; Xuweiyi Chen; Shoubin Yu; Sai Bi; Kai Zhang; Chen Ziwen; Sihan Xu; Jianing Yang; Zexiang Xu; Kalyan Sunkavalli; Mohit Bansal; Joyce Chai; Hao Tan
>
> **备注:** Project page: https://4dlrm.github.io/
>
> **摘要:** Can we scale 4D pretraining to learn general space-time representations that reconstruct an object from a few views at some times to any view at any time? We provide an affirmative answer with 4D-LRM, the first large-scale 4D reconstruction model that takes input from unconstrained views and timestamps and renders arbitrary novel view-time combinations. Unlike prior 4D approaches, e.g., optimization-based, geometry-based, or generative, that struggle with efficiency, generalization, or faithfulness, 4D-LRM learns a unified space-time representation and directly predicts per-pixel 4D Gaussian primitives from posed image tokens across time, enabling fast, high-quality rendering at, in principle, infinite frame rate. Our results demonstrate that scaling spatiotemporal pretraining enables accurate and efficient 4D reconstruction. We show that 4D-LRM generalizes to novel objects, interpolates across time, and handles diverse camera setups. It reconstructs 24-frame sequences in one forward pass with less than 1.5 seconds on a single A100 GPU.
>
---
#### [new 170] Deep Learning-based Alignment Measurement in Knee Radiographs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像分析任务，旨在自动化测量膝关节对齐度。通过深度学习定位100多个解剖标志，实现准确的膝关节角度评估。**

- **链接: [http://arxiv.org/pdf/2506.18209v1](http://arxiv.org/pdf/2506.18209v1)**

> **作者:** Zhisen Hu; Dominic Cullen; Peter Thompson; David Johnson; Chang Bian; Aleksei Tiulpin; Timothy Cootes; Claudia Lindner
>
> **备注:** Accepted to MICCAI 2025
>
> **摘要:** Radiographic knee alignment (KA) measurement is important for predicting joint health and surgical outcomes after total knee replacement. Traditional methods for KA measurements are manual, time-consuming and require long-leg radiographs. This study proposes a deep learning-based method to measure KA in anteroposterior knee radiographs via automatically localized knee anatomical landmarks. Our method builds on hourglass networks and incorporates an attention gate structure to enhance robustness and focus on key anatomical features. To our knowledge, this is the first deep learning-based method to localize over 100 knee anatomical landmarks to fully outline the knee shape while integrating KA measurements on both pre-operative and post-operative images. It provides highly accurate and reliable anatomical varus/valgus KA measurements using the anatomical tibiofemoral angle, achieving mean absolute differences ~1{\deg} when compared to clinical ground truth measurements. Agreement between automated and clinical measurements was excellent pre-operatively (intra-class correlation coefficient (ICC) = 0.97) and good post-operatively (ICC = 0.86). Our findings demonstrate that KA assessment can be automated with high accuracy, creating opportunities for digitally enhanced clinical workflows.
>
---
#### [new 171] PhysID: Physics-based Interactive Dynamics from a Single-view Image
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在将静态图像转换为物理交互动态。通过单视角图像生成3D模型并模拟物理行为，实现高效、实时的交互体验。**

- **链接: [http://arxiv.org/pdf/2506.17746v1](http://arxiv.org/pdf/2506.17746v1)**

> **作者:** Sourabh Vasant Gothe; Ayon Chattopadhyay; Gunturi Venkata Sai Phani Kiran; Pratik; Vibhav Agarwal; Jayesh Rajkumar Vachhani; Sourav Ghosh; Parameswaranath VM; Barath Raj KR
>
> **备注:** Published in 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). Project page: https://physid.github.io/
>
> **摘要:** Transforming static images into interactive experiences remains a challenging task in computer vision. Tackling this challenge holds the potential to elevate mobile user experiences, notably through interactive and AR/VR applications. Current approaches aim to achieve this either using pre-recorded video responses or requiring multi-view images as input. In this paper, we present PhysID, that streamlines the creation of physics-based interactive dynamics from a single-view image by leveraging large generative models for 3D mesh generation and physical property prediction. This significantly reduces the expertise required for engineering-intensive tasks like 3D modeling and intrinsic property calibration, enabling the process to be scaled with minimal manual intervention. We integrate an on-device physics-based engine for physically plausible real-time rendering with user interactions. PhysID represents a leap forward in mobile-based interactive dynamics, offering real-time, non-deterministic interactions and user-personalization with efficient on-device memory consumption. Experiments evaluate the zero-shot capabilities of various Multimodal Large Language Models (MLLMs) on diverse tasks and the performance of 3D reconstruction models. These results demonstrate the cohesive functioning of all modules within the end-to-end framework, contributing to its effectiveness.
>
---
#### [new 172] ThermalLoc: A Vision Transformer-Based Approach for Robust Thermal Camera Relocalization in Large-Scale Environments
- **分类: cs.CV**

- **简介: 该论文属于视觉重定位任务，解决热成像相机在大场景中的位姿估计问题。提出ThermalLoc方法，结合EfficientNet与Transformer提取特征，实现高精度定位。**

- **链接: [http://arxiv.org/pdf/2506.18268v1](http://arxiv.org/pdf/2506.18268v1)**

> **作者:** Yu Liu; Yangtao Meng; Xianfei Pan; Jie Jiang; Changhao Chen
>
> **备注:** 8 pages, 3 figures, accepted to IROS 2025
>
> **摘要:** Thermal cameras capture environmental data through heat emission, a fundamentally different mechanism compared to visible light cameras, which rely on pinhole imaging. As a result, traditional visual relocalization methods designed for visible light images are not directly applicable to thermal images. Despite significant advancements in deep learning for camera relocalization, approaches specifically tailored for thermal camera-based relocalization remain underexplored. To address this gap, we introduce ThermalLoc, a novel end-to-end deep learning method for thermal image relocalization. ThermalLoc effectively extracts both local and global features from thermal images by integrating EfficientNet with Transformers, and performs absolute pose regression using two MLP networks. We evaluated ThermalLoc on both the publicly available thermal-odometry dataset and our own dataset. The results demonstrate that ThermalLoc outperforms existing representative methods employed for thermal camera relocalization, including AtLoc, MapNet, PoseNet, and RobustLoc, achieving superior accuracy and robustness.
>
---
#### [new 173] DRAMA-X: A Fine-grained Intent Prediction and Risk Reasoning Benchmark For Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DRAMA-X基准，用于驾驶场景中的意图预测与风险推理，解决自动驾驶中对弱势道路使用者行为理解不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.17590v1](http://arxiv.org/pdf/2506.17590v1)**

> **作者:** Mihir Godbole; Xiangbo Gao; Zhengzhong Tu
>
> **备注:** 19 pages, 5 figures, Preprint under review. Code available at: https://github.com/taco-group/DRAMA-X
>
> **摘要:** Understanding the short-term motion of vulnerable road users (VRUs) like pedestrians and cyclists is critical for safe autonomous driving, especially in urban scenarios with ambiguous or high-risk behaviors. While vision-language models (VLMs) have enabled open-vocabulary perception, their utility for fine-grained intent reasoning remains underexplored. Notably, no existing benchmark evaluates multi-class intent prediction in safety-critical situations, To address this gap, we introduce DRAMA-X, a fine-grained benchmark constructed from the DRAMA dataset via an automated annotation pipeline. DRAMA-X contains 5,686 accident-prone frames labeled with object bounding boxes, a nine-class directional intent taxonomy, binary risk scores, expert-generated action suggestions for the ego vehicle, and descriptive motion summaries. These annotations enable a structured evaluation of four interrelated tasks central to autonomous decision-making: object detection, intent prediction, risk assessment, and action suggestion. As a reference baseline, we propose SGG-Intent, a lightweight, training-free framework that mirrors the ego vehicle's reasoning pipeline. It sequentially generates a scene graph from visual input using VLM-backed detectors, infers intent, assesses risk, and recommends an action using a compositional reasoning stage powered by a large language model. We evaluate a range of recent VLMs, comparing performance across all four DRAMA-X tasks. Our experiments demonstrate that scene-graph-based reasoning enhances intent prediction and risk assessment, especially when contextual cues are explicitly modeled.
>
---
#### [new 174] PicoSAM2: Low-Latency Segmentation In-Sensor for Edge Vision Applications
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，解决边缘设备实时分割问题。提出PicoSAM2模型，实现低延迟、隐私保护的摄像头端分割。**

- **链接: [http://arxiv.org/pdf/2506.18807v1](http://arxiv.org/pdf/2506.18807v1)**

> **作者:** Pietro Bonazzi; Nicola Farronato; Stefan Zihlmann; Haotong Qi; Michele Magno
>
> **摘要:** Real-time, on-device segmentation is critical for latency-sensitive and privacy-aware applications like smart glasses and IoT devices. We introduce PicoSAM2, a lightweight (1.3M parameters, 336M MACs) promptable segmentation model optimized for edge and in-sensor execution, including the Sony IMX500. It builds on a depthwise separable U-Net, with knowledge distillation and fixed-point prompt encoding to learn from the Segment Anything Model 2 (SAM2). On COCO and LVIS, it achieves 51.9% and 44.9% mIoU, respectively. The quantized model (1.22MB) runs at 14.3 ms on the IMX500-achieving 86 MACs/cycle, making it the only model meeting both memory and compute constraints for in-sensor deployment. Distillation boosts LVIS performance by +3.5% mIoU and +5.1% mAP. These results demonstrate that efficient, promptable segmentation is feasible directly on-camera, enabling privacy-preserving vision without cloud or host processing.
>
---
#### [new 175] Cloud-Aware SAR Fusion for Enhanced Optical Sensing in Space Missions
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感图像处理任务，旨在解决云层干扰光学影像的问题。通过融合SAR与光学数据，利用深度学习生成无云光学图像。**

- **链接: [http://arxiv.org/pdf/2506.17885v1](http://arxiv.org/pdf/2506.17885v1)**

> **作者:** Trong-An Bui; Thanh-Thoai Le
>
> **摘要:** Cloud contamination significantly impairs the usability of optical satellite imagery, affecting critical applications such as environmental monitoring, disaster response, and land-use analysis. This research presents a Cloud-Attentive Reconstruction Framework that integrates SAR-optical feature fusion with deep learning-based image reconstruction to generate cloud-free optical imagery. The proposed framework employs an attention-driven feature fusion mechanism to align complementary structural information from Synthetic Aperture Radar (SAR) with spectral characteristics from optical data. Furthermore, a cloud-aware model update strategy introduces adaptive loss weighting to prioritize cloud-occluded regions, enhancing reconstruction accuracy. Experimental results demonstrate that the proposed method outperforms existing approaches, achieving a PSNR of 31.01 dB, SSIM of 0.918, and MAE of 0.017. These outcomes highlight the framework's effectiveness in producing high-fidelity, spatially and spectrally consistent cloud-free optical images.
>
---
#### [new 176] BlenderFusion: 3D-Grounded Visual Editing and Generative Compositing
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出BlenderFusion，用于3D场景的生成式合成与编辑，解决复杂场景重组问题，通过分层、编辑、融合流程实现高效场景生成。**

- **链接: [http://arxiv.org/pdf/2506.17450v1](http://arxiv.org/pdf/2506.17450v1)**

> **作者:** Jiacheng Chen; Ramin Mehran; Xuhui Jia; Saining Xie; Sanghyun Woo
>
> **备注:** Project page: https://blenderfusion.github.io
>
> **摘要:** We present BlenderFusion, a generative visual compositing framework that synthesizes new scenes by recomposing objects, camera, and background. It follows a layering-editing-compositing pipeline: (i) segmenting and converting visual inputs into editable 3D entities (layering), (ii) editing them in Blender with 3D-grounded control (editing), and (iii) fusing them into a coherent scene using a generative compositor (compositing). Our generative compositor extends a pre-trained diffusion model to process both the original (source) and edited (target) scenes in parallel. It is fine-tuned on video frames with two key training strategies: (i) source masking, enabling flexible modifications like background replacement; (ii) simulated object jittering, facilitating disentangled control over objects and camera. BlenderFusion significantly outperforms prior methods in complex compositional scene editing tasks.
>
---
#### [new 177] Temporal Neural Cellular Automata: Application to modeling of contrast enhancement in breast MRI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决乳腺MRI中对比增强的建模问题。通过引入TeNCA模型，提升时间序列图像生成的连贯性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.18720v1](http://arxiv.org/pdf/2506.18720v1)**

> **作者:** Daniel M. Lang; Richard Osuala; Veronika Spieker; Karim Lekadir; Rickmer Braren; Julia A. Schnabel
>
> **备注:** MICCAI 2025
>
> **摘要:** Synthetic contrast enhancement offers fast image acquisition and eliminates the need for intravenous injection of contrast agent. This is particularly beneficial for breast imaging, where long acquisition times and high cost are significantly limiting the applicability of magnetic resonance imaging (MRI) as a widespread screening modality. Recent studies have demonstrated the feasibility of synthetic contrast generation. However, current state-of-the-art (SOTA) methods lack sufficient measures for consistent temporal evolution. Neural cellular automata (NCA) offer a robust and lightweight architecture to model evolving patterns between neighboring cells or pixels. In this work we introduce TeNCA (Temporal Neural Cellular Automata), which extends and further refines NCAs to effectively model temporally sparse, non-uniformly sampled imaging data. To achieve this, we advance the training strategy by enabling adaptive loss computation and define the iterative nature of the method to resemble a physical progression in time. This conditions the model to learn a physiologically plausible evolution of contrast enhancement. We rigorously train and test TeNCA on a diverse breast MRI dataset and demonstrate its effectiveness, surpassing the performance of existing methods in generation of images that align with ground truth post-contrast sequences.
>
---
#### [new 178] A workflow for generating synthetic LiDAR datasets in simulation environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主系统感知与传感器安全任务，旨在生成高保真合成LiDAR数据集，解决真实数据不足和安全评估难题。**

- **链接: [http://arxiv.org/pdf/2506.17378v1](http://arxiv.org/pdf/2506.17378v1)**

> **作者:** Abhishek Phadke; Shakib Mahmud Dipto; Pratip Rana
>
> **摘要:** This paper presents a simulation workflow for generating synthetic LiDAR datasets to support autonomous vehicle perception, robotics research, and sensor security analysis. Leveraging the CoppeliaSim simulation environment and its Python API, we integrate time-of-flight LiDAR, image sensors, and two dimensional scanners onto a simulated vehicle platform operating within an urban scenario. The workflow automates data capture, storage, and annotation across multiple formats (PCD, PLY, CSV), producing synchronized multimodal datasets with ground truth pose information. We validate the pipeline by generating large-scale point clouds and corresponding RGB and depth imagery. The study examines potential security vulnerabilities in LiDAR data, such as adversarial point injection and spoofing attacks, and demonstrates how synthetic datasets can facilitate the evaluation of defense strategies. Finally, limitations related to environmental realism, sensor noise modeling, and computational scalability are discussed, and future research directions, such as incorporating weather effects, real-world terrain models, and advanced scanner configurations, are proposed. The workflow provides a versatile, reproducible framework for generating high-fidelity synthetic LiDAR datasets to advance perception research and strengthen sensor security in autonomous systems. Documentation and examples accompany this framework; samples of animated cloud returns and image sensor data can be found at this Link.
>
---
#### [new 179] Taming Vision-Language Models for Medical Image Analysis: A Comprehensive Review
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决将通用视觉-语言模型适配到医疗领域的挑战，通过总结现有方法、分析问题并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2506.18378v1](http://arxiv.org/pdf/2506.18378v1)**

> **作者:** Haoneng Lin; Cheng Xu; Jing Qin
>
> **备注:** 34 pages
>
> **摘要:** Modern Vision-Language Models (VLMs) exhibit unprecedented capabilities in cross-modal semantic understanding between visual and textual modalities. Given the intrinsic need for multi-modal integration in clinical applications, VLMs have emerged as a promising solution for a wide range of medical image analysis tasks. However, adapting general-purpose VLMs to medical domain poses numerous challenges, such as large domain gaps, complicated pathological variations, and diversity and uniqueness of different tasks. The central purpose of this review is to systematically summarize recent advances in adapting VLMs for medical image analysis, analyzing current challenges, and recommending promising yet urgent directions for further investigations. We begin by introducing core learning strategies for medical VLMs, including pretraining, fine-tuning, and prompt learning. We then categorize five major VLM adaptation strategies for medical image analysis. These strategies are further analyzed across eleven medical imaging tasks to illustrate their current practical implementations. Furthermore, we analyze key challenges that impede the effective adaptation of VLMs to clinical applications and discuss potential directions for future research. We also provide an open-access repository of related literature to facilitate further research, available at https://github.com/haonenglin/Awesome-VLM-for-MIA. It is anticipated that this article can help researchers who are interested in harnessing VLMs in medical image analysis tasks have a better understanding on their capabilities and limitations, as well as current technical barriers, to promote their innovative, robust, and safe application in clinical practice.
>
---
#### [new 180] TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **简介: 该论文属于音乐驱动的群体舞蹈生成任务，解决多舞者碰撞、单人脚滑和长序列 abrupt 位移问题，提出 TCDiff++ 模型优化轨迹控制。**

- **链接: [http://arxiv.org/pdf/2506.18671v1](http://arxiv.org/pdf/2506.18671v1)**

> **作者:** Yuqin Dai; Wanlu Zhu; Ronghui Li; Xiu Li; Zhenyu Zhang; Jun Li; Jian Yang
>
> **摘要:** Music-driven dance generation has garnered significant attention due to its wide range of industrial applications, particularly in the creation of group choreography. During the group dance generation process, however, most existing methods still face three primary issues: multi-dancer collisions, single-dancer foot sliding and abrupt swapping in the generation of long group dance. In this paper, we propose TCDiff++, a music-driven end-to-end framework designed to generate harmonious group dance. Specifically, to mitigate multi-dancer collisions, we utilize a dancer positioning embedding to better maintain the relative positioning among dancers. Additionally, we incorporate a distance-consistency loss to ensure that inter-dancer distances remain within plausible ranges. To address the issue of single-dancer foot sliding, we introduce a swap mode embedding to indicate dancer swapping patterns and design a Footwork Adaptor to refine raw motion, thereby minimizing foot sliding. For long group dance generation, we present a long group diffusion sampling strategy that reduces abrupt position shifts by injecting positional information into the noisy input. Furthermore, we integrate a Sequence Decoder layer to enhance the model's ability to selectively process long sequences. Extensive experiments demonstrate that our TCDiff++ achieves state-of-the-art performance, particularly in long-duration scenarios, ensuring high-quality and coherent group dance generation.
>
---
#### [new 181] LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation
- **分类: cs.IR; cs.CV**

- **简介: 该论文属于跨领域序列推荐任务，旨在提升用户兴趣建模。通过融合视觉与文本信息，并利用大语言模型增强，提高推荐效果。**

- **链接: [http://arxiv.org/pdf/2506.17966v1](http://arxiv.org/pdf/2506.17966v1)**

> **作者:** Wangyu Wu; Zhenhong Chen; Xianglin Qiu; Siqi Song; Xiaowei Huang; Fei Ma; Jimin Xiao
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2504.15085
>
> **摘要:** Cross-Domain Sequential Recommendation (CDSR) predicts user behavior by leveraging historical interactions across multiple domains, focusing on modeling cross-domain preferences and capturing both intra- and inter-sequence item relationships. We propose LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation (LLM-EMF), a novel and advanced approach that enhances textual information with Large Language Models (LLM) knowledge and significantly improves recommendation performance through the fusion of visual and textual data. Using the frozen CLIP model, we generate image and text embeddings, thereby enriching item representations with multimodal data. A multiple attention mechanism jointly learns both single-domain and cross-domain preferences, effectively capturing and understanding complex user interests across diverse domains. Evaluations conducted on four e-commerce datasets demonstrate that LLM-EMF consistently outperforms existing methods in modeling cross-domain user preferences, thereby highlighting the effectiveness of multimodal data integration and its advantages in enhancing sequential recommendation systems. Our source code will be released.
>
---
#### [new 182] Origins of Creativity in Attention-Based Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究扩散模型中创造力的来源，聚焦于自注意力机制在生成图像一致性中的作用，通过理论分析与实验验证其影响。**

- **链接: [http://arxiv.org/pdf/2506.17324v1](http://arxiv.org/pdf/2506.17324v1)**

> **作者:** Emma Finn; T. Anderson Keller; Manos Theodosis; Demba E. Ba
>
> **摘要:** As diffusion models have become the tool of choice for image generation and as the quality of the images continues to improve, the question of how `creativity' originates in diffusion has become increasingly important. The score matching perspective on diffusion has proven particularly fruitful for understanding how and why diffusion models generate images that remain plausible while differing significantly from their training images. In particular, as explained in (Kamb \& Ganguli, 2024) and others, e.g., (Ambrogioni, 2023), theory suggests that if our score matching were optimal, we would only be able to recover training samples through our diffusion process. However, as shown by Kamb \& Ganguli, (2024), in diffusion models where the score is parametrized by a simple CNN, the inductive biases of the CNN itself (translation equivariance and locality) allow the model to generate samples that globally do not match any training samples, but are rather patch-wise `mosaics'. Notably, however, this theory does not extend to describe the role of self-attention in this process. In this work, we take a preliminary step in this direction to extend this theory to the case of diffusion models whose score is parametrized by a CNN with a final self-attention layer. We show that our theory suggests that self-attention will induce a globally image-consistent arrangement of local features beyond the patch-level in generated samples, and we verify this behavior empirically on a carefully crafted dataset.
>
---
#### [new 183] Unfolding the Past: A Comprehensive Deep Learning Approach to Analyzing Incunabula Pages
- **分类: cs.DL; cs.CV**

- **简介: 该论文属于文档分析任务，旨在自动解析早期印刷书籍的结构与内容。通过深度学习方法，解决文本、图片等元素的识别与分类问题。**

- **链接: [http://arxiv.org/pdf/2506.18069v1](http://arxiv.org/pdf/2506.18069v1)**

> **作者:** Klaudia Ropel; Krzysztof Kutt; Luiz do Valle Miranda; Grzegorz J. Nalepa
>
> **备注:** 10 pages, 8 figures; submitted to TPDL 2025
>
> **摘要:** We developed a proof-of-concept method for the automatic analysis of the structure and content of incunabula pages. A custom dataset comprising 500 annotated pages from five different incunabula was created using resources from the Jagiellonian Digital Library. Each page was manually labeled with five predefined classes: Text, Title, Picture, Table, and Handwriting. Additionally, the publicly available DocLayNet dataset was utilized as supplementary training data. To perform object detection, YOLO11n and YOLO11s models were employed and trained using two strategies: a combined dataset (DocLayNet and the custom dataset) and the custom dataset alone. The highest performance (F1 = 0.94) was achieved by the YOLO11n model trained exclusively on the custom data. Optical character recognition was then conducted on regions classified as Text, using both Tesseract and Kraken OCR, with Tesseract demonstrating superior results. Subsequently, image classification was applied to the Picture class using a ResNet18 model, achieving an accuracy of 98.7% across five subclasses: Decorative_letter, Illustration, Other, Stamp, and Wrong_detection. Furthermore, the CLIP model was utilized to generate semantic descriptions of illustrations. The results confirm the potential of machine learning in the analysis of early printed books, while emphasizing the need for further advancements in OCR performance and visual content interpretation.
>
---
#### [new 184] DSA-NRP: No-Reflow Prediction from Angiographic Perfusion Dynamics in Stroke EVT
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在预测急性脑梗死患者EVT后无复流现象。通过分析DSA影像和临床数据，构建机器学习模型实现早期准确预测。**

- **链接: [http://arxiv.org/pdf/2506.17501v1](http://arxiv.org/pdf/2506.17501v1)**

> **作者:** Shreeram Athreya; Carlos Olivares; Ameera Ismail; Kambiz Nael; William Speier; Corey Arnold
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Following successful large-vessel recanalization via endovascular thrombectomy (EVT) for acute ischemic stroke (AIS), some patients experience a complication known as no-reflow, defined by persistent microvascular hypoperfusion that undermines tissue recovery and worsens clinical outcomes. Although prompt identification is crucial, standard clinical practice relies on perfusion magnetic resonance imaging (MRI) within 24 hours post-procedure, delaying intervention. In this work, we introduce the first-ever machine learning (ML) framework to predict no-reflow immediately after EVT by leveraging previously unexplored intra-procedural digital subtraction angiography (DSA) sequences and clinical variables. Our retrospective analysis included AIS patients treated at UCLA Medical Center (2011-2024) who achieved favorable mTICI scores (2b-3) and underwent pre- and post-procedure MRI. No-reflow was defined as persistent hypoperfusion (Tmax > 6 s) on post-procedural imaging. From DSA sequences (AP and lateral views), we extracted statistical and temporal perfusion features from the target downstream territory to train ML classifiers for predicting no-reflow. Our novel method significantly outperformed a clinical-features baseline(AUC: 0.7703 $\pm$ 0.12 vs. 0.5728 $\pm$ 0.12; accuracy: 0.8125 $\pm$ 0.10 vs. 0.6331 $\pm$ 0.09), demonstrating that real-time DSA perfusion dynamics encode critical insights into microvascular integrity. This approach establishes a foundation for immediate, accurate no-reflow prediction, enabling clinicians to proactively manage high-risk patients without reliance on delayed imaging.
>
---
#### [new 185] DRIMV_TSK: An Interpretable Surgical Evaluation Model for Incomplete Multi-View Rectal Cancer Data
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于医疗评估任务，解决多视角直肠癌数据不完整下的手术难度评估问题。提出DRIMV_TSK模型，融合多视角数据与模糊系统进行可解释评估。**

- **链接: [http://arxiv.org/pdf/2506.17552v1](http://arxiv.org/pdf/2506.17552v1)**

> **作者:** Wei Zhang; Zi Wang; Hanwen Zhou; Zhaohong Deng; Weiping Ding; Yuxi Ge; Te Zhang; Yuanpeng Zhang; Kup-Sze Choi; Shitong Wang; Shudong Hu
>
> **摘要:** A reliable evaluation of surgical difficulty can improve the success of the treatment for rectal cancer and the current evaluation method is based on clinical data. However, more data about rectal cancer can be collected with the development of technology. Meanwhile, with the development of artificial intelligence, its application in rectal cancer treatment is becoming possible. In this paper, a multi-view rectal cancer dataset is first constructed to give a more comprehensive view of patients, including the high-resolution MRI image view, pressed-fat MRI image view, and clinical data view. Then, an interpretable incomplete multi-view surgical evaluation model is proposed, considering that it is hard to obtain extensive and complete patient data in real application scenarios. Specifically, a dual representation incomplete multi-view learning model is first proposed to extract the common information between views and specific information in each view. In this model, the missing view imputation is integrated into representation learning, and second-order similarity constraint is also introduced to improve the cooperative learning between these two parts. Then, based on the imputed multi-view data and the learned dual representation, a multi-view surgical evaluation model with the TSK fuzzy system is proposed. In the proposed model, a cooperative learning mechanism is constructed to explore the consistent information between views, and Shannon entropy is also introduced to adapt the view weight. On the MVRC dataset, we compared it with several advanced algorithms and DRIMV_TSK obtained the best results.
>
---
#### [new 186] A Deep Convolutional Neural Network-Based Novel Class Balancing for Imbalance Data Segmentation
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，解决视网膜血管分割中的数据不平衡问题，提出BLCB-CNN方法实现有效分割。**

- **链接: [http://arxiv.org/pdf/2506.18474v1](http://arxiv.org/pdf/2506.18474v1)**

> **作者:** Atifa Kalsoom; M. A. Iftikhar; Amjad Ali; Zubair Shah; Shidin Balakrishnan; Hazrat Ali
>
> **备注:** This is preprint of the paper submitted to Scientific Reports journal
>
> **摘要:** Retinal fundus images provide valuable insights into the human eye's interior structure and crucial features, such as blood vessels, optic disk, macula, and fovea. However, accurate segmentation of retinal blood vessels can be challenging due to imbalanced data distribution and varying vessel thickness. In this paper, we propose BLCB-CNN, a novel pipeline based on deep learning and bi-level class balancing scheme to achieve vessel segmentation in retinal fundus images. The BLCB-CNN scheme uses a Convolutional Neural Network (CNN) architecture and an empirical approach to balance the distribution of pixels across vessel and non-vessel classes and within thin and thick vessels. Level-I is used for vessel/non-vessel balancing and Level-II is used for thick/thin vessel balancing. Additionally, pre-processing of the input retinal fundus image is performed by Global Contrast Normalization (GCN), Contrast Limited Adaptive Histogram Equalization (CLAHE), and gamma corrections to increase intensity uniformity as well as to enhance the contrast between vessels and background pixels. The resulting balanced dataset is used for classification-based segmentation of the retinal vascular tree. We evaluate the proposed scheme on standard retinal fundus images and achieve superior performance measures, including an area under the ROC curve of 98.23%, Accuracy of 96.22%, Sensitivity of 81.57%, and Specificity of 97.65%. We also demonstrate the method's efficacy through external cross-validation on STARE images, confirming its generalization ability.
>
---
#### [new 187] Multimodal Political Bias Identification and Neutralization
- **分类: cs.CY; cs.AI; cs.CV**

- **简介: 该论文属于多模态政治偏见识别与消除任务，旨在解决文本和图像中的主观偏见问题，通过四步模型实现文本和图像的去偏处理。**

- **链接: [http://arxiv.org/pdf/2506.17372v1](http://arxiv.org/pdf/2506.17372v1)**

> **作者:** Cedric Bernard; Xavier Pleimling; Amun Kharel; Chase Vickery
>
> **摘要:** Due to the presence of political echo chambers, it becomes imperative to detect and remove subjective bias and emotionally charged language from both the text and images of political articles. However, prior work has focused on solely the text portion of the bias rather than both the text and image portions. This is a problem because the images are just as powerful of a medium to communicate information as text is. To that end, we present a model that leverages both text and image bias which consists of four different steps. Image Text Alignment focuses on semantically aligning images based on their bias through CLIP models. Image Bias Scoring determines the appropriate bias score of images via a ViT classifier. Text De-Biasing focuses on detecting biased words and phrases and neutralizing them through BERT models. These three steps all culminate to the final step of debiasing, which replaces the text and the image with neutralized or reduced counterparts, which for images is done by comparing the bias scores. The results so far indicate that this approach is promising, with the text debiasing strategy being able to identify many potential biased words and phrases, and the ViT model showcasing effective training. The semantic alignment model also is efficient. However, more time, particularly in training, and resources are needed to obtain better results. A human evaluation portion was also proposed to ensure semantic consistency of the newly generated text and images.
>
---
#### [new 188] Deciphering Emotions in Children Storybooks: A Comparative Analysis of Multimodal LLMs in Educational Applications
- **分类: cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于情感识别任务，旨在提升多模态AI在阿拉伯儿童绘本中的情感理解能力。研究对比了GPT-4o与Gemini 1.5 Pro的表现，分析了不同提示策略的效果。**

- **链接: [http://arxiv.org/pdf/2506.18201v1](http://arxiv.org/pdf/2506.18201v1)**

> **作者:** Bushra Asseri; Estabraq Abdelaziz; Maha Al Mogren; Tayef Alhefdhi; Areej Al-Wabil
>
> **摘要:** Emotion recognition capabilities in multimodal AI systems are crucial for developing culturally responsive educational technologies, yet remain underexplored for Arabic language contexts where culturally appropriate learning tools are critically needed. This study evaluates the emotion recognition performance of two advanced multimodal large language models, GPT-4o and Gemini 1.5 Pro, when processing Arabic children's storybook illustrations. We assessed both models across three prompting strategies (zero-shot, few-shot, and chain-of-thought) using 75 images from seven Arabic storybooks, comparing model predictions with human annotations based on Plutchik's emotional framework. GPT-4o consistently outperformed Gemini across all conditions, achieving the highest macro F1-score of 59% with chain-of-thought prompting compared to Gemini's best performance of 43%. Error analysis revealed systematic misclassification patterns, with valence inversions accounting for 60.7% of errors, while both models struggled with culturally nuanced emotions and ambiguous narrative contexts. These findings highlight fundamental limitations in current models' cultural understanding and emphasize the need for culturally sensitive training approaches to develop effective emotion-aware educational technologies for Arabic-speaking learners.
>
---
#### [new 189] Radar and Event Camera Fusion for Agile Robot Ego-Motion Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人自主导航任务，旨在解决高速运动下姿态估计困难的问题。通过融合事件相机和毫米波雷达数据，提出无IMU和特征匹配的运动估计框架，提升动态环境下的鲁棒性和效率。**

- **链接: [http://arxiv.org/pdf/2506.18443v1](http://arxiv.org/pdf/2506.18443v1)**

> **作者:** Yang Lyu; Zhenghao Zou; Yanfeng Li; Chunhui Zhao; Quan Pan
>
> **摘要:** Achieving reliable ego motion estimation for agile robots, e.g., aerobatic aircraft, remains challenging because most robot sensors fail to respond timely and clearly to highly dynamic robot motions, often resulting in measurement blurring, distortion, and delays. In this paper, we propose an IMU-free and feature-association-free framework to achieve aggressive ego-motion velocity estimation of a robot platform in highly dynamic scenarios by combining two types of exteroceptive sensors, an event camera and a millimeter wave radar, First, we used instantaneous raw events and Doppler measurements to derive rotational and translational velocities directly. Without a sophisticated association process between measurement frames, the proposed method is more robust in texture-less and structureless environments and is more computationally efficient for edge computing devices. Then, in the back-end, we propose a continuous-time state-space model to fuse the hybrid time-based and event-based measurements to estimate the ego-motion velocity in a fixed-lagged smoother fashion. In the end, we validate our velometer framework extensively in self-collected experiment datasets. The results indicate that our IMU-free and association-free ego motion estimation framework can achieve reliable and efficient velocity output in challenging environments. The source code, illustrative video and dataset are available at https://github.com/ZzhYgwh/TwistEstimator.
>
---
#### [new 190] Can Generated Images Serve as a Viable Modality for Text-Centric Multimodal Learning?
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于多模态学习任务，探讨生成图像是否可作为文本任务的补充模态。研究解决如何利用T2I模型增强文本理解的问题，通过实验分析影响效果的关键因素。**

- **链接: [http://arxiv.org/pdf/2506.17623v1](http://arxiv.org/pdf/2506.17623v1)**

> **作者:** Yuesheng Huang; Peng Zhang; Riliang Liu; Jiaqi Liang
>
> **备注:** 4 figures,7 tables
>
> **摘要:** A significant ``modality gap" exists between the abundance of text-only data and the increasing power of multimodal models. This work systematically investigates whether images generated on-the-fly by Text-to-Image (T2I) models can serve as a valuable complementary modality for text-centric tasks. Through a comprehensive evaluation framework on text classification, we analyze the impact of critical variables, including T2I model quality, prompt engineering strategies, and multimodal fusion architectures. Our findings demonstrate that this``synthetic perception" can yield significant performance gains, even when augmenting strong large language model baselines. However, we find the effectiveness of this approach is highly conditional, depending critically on the semantic alignment between text and the generated image, the inherent ``visual groundability" of the task, and the generative fidelity of the T2I model. Our work establishes the first rigorous benchmark for this paradigm, providing a clear analysis of its potential and current limitations, and demonstrating its viability as a pathway to enrich language understanding in traditionally unimodal scenarios.
>
---
#### [new 191] Transforming H&E images into IHC: A Variance-Penalized GAN for Precision Oncology
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像翻译任务，旨在解决HER2检测成本高、依赖抗体的问题。通过改进GAN模型，从H&E图像生成IHC图像，提升诊断效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.18371v1](http://arxiv.org/pdf/2506.18371v1)**

> **作者:** Sara Rehmat; Hafeez Ur Rehman
>
> **摘要:** The overexpression of the human epidermal growth factor receptor 2 (HER2) in breast cells is a key driver of HER2-positive breast cancer, a highly aggressive subtype requiring precise diagnosis and targeted therapy. Immunohistochemistry (IHC) is the standard technique for HER2 assessment but is costly, labor-intensive, and highly dependent on antibody selection. In contrast, hematoxylin and eosin (H&E) staining, a routine histopathological procedure, offers broader accessibility but lacks HER2 specificity. This study proposes an advanced deep learning-based image translation framework to generate highfidelity IHC images from H&E-stained tissue samples, enabling cost-effective and scalable HER2 assessment. By modifying the loss function of pyramid pix2pix, we mitigate mode collapse, a fundamental limitation in generative adversarial networks (GANs), and introduce a novel variance-based penalty that enforces structural diversity in generated images. Our model particularly excels in translating HER2-positive (IHC 3+) images, which have remained challenging for existing methods due to their complex morphological variations. Extensive evaluations on the BCI histopathological dataset demonstrate that our model surpasses state-of-the-art methods in terms of peak signal-tonoise ratio (PSNR), structural similarity index (SSIM), and Frechet Inception Distance (FID), particularly in accurately translating HER2-positive (IHC 3+) images. Beyond medical imaging, our model exhibits superior performance in general image-to-image translation tasks, showcasing its potential across multiple domains. This work marks a significant step toward AI-driven precision oncology, offering a reliable and efficient alternative to traditional HER2 diagnostics.
>
---
#### [new 192] GRAND-SLAM: Local Optimization for Globally Consistent Large-Scale Multi-Agent Gaussian SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多智能体SLAM任务，解决大尺度户外环境中多机器人协同定位与建图问题，提出GRAND-SLAM方法提升跟踪精度和场景一致性。**

- **链接: [http://arxiv.org/pdf/2506.18885v1](http://arxiv.org/pdf/2506.18885v1)**

> **作者:** Annika Thomas; Aneesa Sonawalla; Alex Rose; Jonathan P. How
>
> **摘要:** 3D Gaussian splatting has emerged as an expressive scene representation for RGB-D visual SLAM, but its application to large-scale, multi-agent outdoor environments remains unexplored. Multi-agent Gaussian SLAM is a promising approach to rapid exploration and reconstruction of environments, offering scalable environment representations, but existing approaches are limited to small-scale, indoor environments. To that end, we propose Gaussian Reconstruction via Multi-Agent Dense SLAM, or GRAND-SLAM, a collaborative Gaussian splatting SLAM method that integrates i) an implicit tracking module based on local optimization over submaps and ii) an approach to inter- and intra-robot loop closure integrated into a pose-graph optimization framework. Experiments show that GRAND-SLAM provides state-of-the-art tracking performance and 28% higher PSNR than existing methods on the Replica indoor dataset, as well as 91% lower multi-agent tracking error and improved rendering over existing multi-agent methods on the large-scale, outdoor Kimera-Multi dataset.
>
---
#### [new 193] Multimodal Medical Image Binding via Shared Text Embeddings
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决多模态医学图像对齐问题。通过共享文本嵌入空间实现无需配对数据的跨模态对齐，提升分类与检索性能。**

- **链接: [http://arxiv.org/pdf/2506.18072v1](http://arxiv.org/pdf/2506.18072v1)**

> **作者:** Yunhao Liu; Suyang Xi; Shiqi Liu; Hong Ding; Chicheng Jin; Chenxi Yang; Junjun He; Yiqing Shen
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Medical image analysis increasingly relies on the integration of multiple imaging modalities to capture complementary anatomical and functional information, enabling more accurate diagnosis and treatment planning. Achieving aligned feature representations across these diverse modalities is therefore important for effective multimodal analysis. While contrastive language-image pre-training (CLIP) and its variant have enabled image-text alignments, they require explicitly paired data between arbitrary two modalities, which is difficult to acquire in medical contexts. To address the gap, we present Multimodal Medical Image Binding with Text (M\textsuperscript{3}Bind), a novel pre-training framework that enables seamless alignment of multiple medical imaging modalities through a shared text representation space without requiring explicit paired data between any two medical image modalities. Specifically, based on the insight that different images can naturally bind with text, M\textsuperscript{3}Bind first fine-tunes pre-trained CLIP-like image-text models to align their modality-specific text embedding space while preserving their original image-text alignments. Subsequently, we distill these modality-specific text encoders into a unified model, creating a shared text embedding space. Experiments on X-ray, CT, retina, ECG, and pathological images on multiple downstream tasks demonstrate that M\textsuperscript{3}Bind achieves state-of-the-art performance in zero-shot, few-shot classification and cross-modal retrieval tasks compared to its CLIP-like counterparts. These results validate M\textsuperscript{3}Bind's effectiveness in achieving cross-image-modal alignment for medical analysis.
>
---
#### [new 194] AI-based Multimodal Biometrics for Detecting Smartphone Distractions: Application to Online Learning
- **分类: cs.CY; cs.AI; cs.CV; cs.HC**

- **简介: 该论文属于注意力检测任务，旨在解决在线学习中因手机使用导致的分心问题。通过多模态生物特征数据，提出AI方法提升检测准确性。**

- **链接: [http://arxiv.org/pdf/2506.17364v1](http://arxiv.org/pdf/2506.17364v1)**

> **作者:** Alvaro Becerra; Roberto Daza; Ruth Cobos; Aythami Morales; Mutlu Cukurova; Julian Fierrez
>
> **备注:** Accepted in EC-TEL25: 20th European Conference on Technology Enhanced Learning, Newcastle and Durham, UK, 15-19 September 2025
>
> **摘要:** This work investigates the use of multimodal biometrics to detect distractions caused by smartphone use during tasks that require sustained attention, with a focus on computer-based online learning. Although the methods are applicable to various domains, such as autonomous driving, we concentrate on the challenges learners face in maintaining engagement amid internal (e.g., motivation), system-related (e.g., course design) and contextual (e.g., smartphone use) factors. Traditional learning platforms often lack detailed behavioral data, but Multimodal Learning Analytics (MMLA) and biosensors provide new insights into learner attention. We propose an AI-based approach that leverages physiological signals and head pose data to detect phone use. Our results show that single biometric signals, such as brain waves or heart rate, offer limited accuracy, while head pose alone achieves 87%. A multimodal model combining all signals reaches 91% accuracy, highlighting the benefits of integration. We conclude by discussing the implications and limitations of deploying these models for real-time support in online learning environments.
>
---
#### [new 195] DuetGen: Music Driven Two-Person Dance Generation via Hierarchical Masked Modeling
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于舞蹈生成任务，旨在解决两人舞蹈与音乐同步的问题。通过分阶段的编码与生成模型，实现高质量的双人舞蹈动作生成。**

- **链接: [http://arxiv.org/pdf/2506.18680v1](http://arxiv.org/pdf/2506.18680v1)**

> **作者:** Anindita Ghosh; Bing Zhou; Rishabh Dabral; Jian Wang; Vladislav Golyanik; Christian Theobalt; Philipp Slusallek; Chuan Guo
>
> **备注:** 11 pages, 7 figures, 2 tables, accepted in ACM Siggraph 2025 conference track
>
> **摘要:** We present DuetGen, a novel framework for generating interactive two-person dances from music. The key challenge of this task lies in the inherent complexities of two-person dance interactions, where the partners need to synchronize both with each other and with the music. Inspired by the recent advances in motion synthesis, we propose a two-stage solution: encoding two-person motions into discrete tokens and then generating these tokens from music. To effectively capture intricate interactions, we represent both dancers' motions as a unified whole to learn the necessary motion tokens, and adopt a coarse-to-fine learning strategy in both the stages. Our first stage utilizes a VQ-VAE that hierarchically separates high-level semantic features at a coarse temporal resolution from low-level details at a finer resolution, producing two discrete token sequences at different abstraction levels. Subsequently, in the second stage, two generative masked transformers learn to map music signals to these dance tokens: the first producing high-level semantic tokens, and the second, conditioned on music and these semantic tokens, producing the low-level tokens. We train both transformers to learn to predict randomly masked tokens within the sequence, enabling them to iteratively generate motion tokens by filling an empty token sequence during inference. Through the hierarchical masked modeling and dedicated interaction representation, DuetGen achieves the generation of synchronized and interactive two-person dances across various genres. Extensive experiments and user studies on a benchmark duet dance dataset demonstrate state-of-the-art performance of DuetGen in motion realism, music-dance alignment, and partner coordination.
>
---
#### [new 196] DRO-Augment Framework: Robustness by Synergizing Wasserstein Distributionally Robust Optimization and Data Augmentation
- **分类: stat.ML; cs.CV; cs.LG**

- **简介: 该论文属于图像分类任务，旨在提升模型对数据扰动和对抗攻击的鲁棒性。通过结合Wasserstein分布鲁棒优化与数据增强，提出DRO-Augment框架，显著增强模型稳定性。**

- **链接: [http://arxiv.org/pdf/2506.17874v1](http://arxiv.org/pdf/2506.17874v1)**

> **作者:** Jiaming Hu; Debarghya Mukherjee; Ioannis Ch. Paschalidis
>
> **备注:** 26 pages,3 figures
>
> **摘要:** In many real-world applications, ensuring the robustness and stability of deep neural networks (DNNs) is crucial, particularly for image classification tasks that encounter various input perturbations. While data augmentation techniques have been widely adopted to enhance the resilience of a trained model against such perturbations, there remains significant room for improvement in robustness against corrupted data and adversarial attacks simultaneously. To address this challenge, we introduce DRO-Augment, a novel framework that integrates Wasserstein Distributionally Robust Optimization (W-DRO) with various data augmentation strategies to improve the robustness of the models significantly across a broad spectrum of corruptions. Our method outperforms existing augmentation methods under severe data perturbations and adversarial attack scenarios while maintaining the accuracy on the clean datasets on a range of benchmark datasets, including but not limited to CIFAR-10-C, CIFAR-100-C, MNIST, and Fashion-MNIST. On the theoretical side, we establish novel generalization error bounds for neural networks trained using a computationally efficient, variation-regularized loss function closely related to the W-DRO problem.
>
---
#### [new 197] MAARTA:Multi-Agentic Adaptive Radiology Teaching Assistant
- **分类: cs.CY; cs.CV; cs.LG**

- **简介: 该论文属于医学教育任务，旨在解决放射学学生因缺乏指导而产生的感知错误。工作是提出MAARTA系统，通过分析眼动和报告提供个性化反馈。**

- **链接: [http://arxiv.org/pdf/2506.17320v1](http://arxiv.org/pdf/2506.17320v1)**

> **作者:** Akash Awasthi; Brandon V. Chang; Anh M. Vu; Ngan Le; Rishi Agrawal; Zhigang Deng; Carol Wu; Hien Van Nguyen
>
> **备注:** Accepted to MICCAI 2025 (Main Conference)
>
> **摘要:** Radiology students often struggle to develop perceptual expertise due to limited expert mentorship time, leading to errors in visual search and diagnostic interpretation. These perceptual errors, such as missed fixations, short dwell times, or misinterpretations, are not adequately addressed by current AI systems, which focus on diagnostic accuracy but fail to explain how and why errors occur. To address this gap, we introduce MAARTA (Multi-Agentic Adaptive Radiology Teaching Assistant), a multi-agent framework that analyzes gaze patterns and radiology reports to provide personalized feedback. Unlike single-agent models, MAARTA dynamically selects agents based on error complexity, enabling adaptive and efficient reasoning. By comparing expert and student gaze behavior through structured graphs, the system identifies missed findings and assigns Perceptual Error Teacher agents to analyze discrepancies. MAARTA then uses step-by-step prompting to help students understand their errors and improve diagnostic reasoning, advancing AI-driven radiology education.
>
---
#### [new 198] Decoding Federated Learning: The FedNAM+ Conformal Revolution
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于联邦学习任务，旨在解决模型不确定性量化与可解释性问题。提出FedNAM+框架，结合NAM和新置信预测方法，实现可解释的可靠不确定性估计。**

- **链接: [http://arxiv.org/pdf/2506.17872v1](http://arxiv.org/pdf/2506.17872v1)**

> **作者:** Sree Bhargavi Balija; Amitash Nanda; Debashis Sahoo
>
> **摘要:** Federated learning has significantly advanced distributed training of machine learning models across decentralized data sources. However, existing frameworks often lack comprehensive solutions that combine uncertainty quantification, interpretability, and robustness. To address this, we propose FedNAM+, a federated learning framework that integrates Neural Additive Models (NAMs) with a novel conformal prediction method to enable interpretable and reliable uncertainty estimation. Our method introduces a dynamic level adjustment technique that utilizes gradient-based sensitivity maps to identify key input features influencing predictions. This facilitates both interpretability and pixel-wise uncertainty estimates. Unlike traditional interpretability methods such as LIME and SHAP, which do not provide confidence intervals, FedNAM+ offers visual insights into prediction reliability. We validate our approach through experiments on CT scan, MNIST, and CIFAR datasets, demonstrating high prediction accuracy with minimal loss (e.g., only 0.1% on MNIST), along with transparent uncertainty measures. Visual analysis highlights variable uncertainty intervals, revealing low-confidence regions where model performance can be improved with additional data. Compared to Monte Carlo Dropout, FedNAM+ delivers efficient and global uncertainty estimates with reduced computational overhead, making it particularly suitable for federated learning scenarios. Overall, FedNAM+ provides a robust, interpretable, and computationally efficient framework that enhances trust and transparency in decentralized predictive modeling.
>
---
#### [new 199] Collaborative Texture Filtering
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于纹理过滤任务，解决GPU无法高效使用纹理单元的问题。通过波通信技术减少重复解压，实现高效高质量过滤。**

- **链接: [http://arxiv.org/pdf/2506.17770v1](http://arxiv.org/pdf/2506.17770v1)**

> **作者:** Tomas Akenine-Möller; Pontus Ebelin; Matt Pharr; Bartlomiej Wronski
>
> **备注:** Accepted to ACM/EG Symposium on High Performance Graphics (HPG), 2025
>
> **摘要:** Recent advances in texture compression provide major improvements in compression ratios, but cannot use the GPU's texture units for decompression and filtering. This has led to the development of stochastic texture filtering (STF) techniques to avoid the high cost of multiple texel evaluations with such formats. Unfortunately, those methods can give undesirable visual appearance changes under magnification and may contain visible noise and flicker despite the use of spatiotemporal denoisers. Recent work substantially improves the quality of magnification filtering with STF by sharing decoded texel values between nearby pixels (Wronski 2025). Using GPU wave communication intrinsics, this sharing can be performed inside actively executing shaders without memory traffic overhead. We take this idea further and present novel algorithms that use wave communication between lanes to avoid repeated texel decompression prior to filtering. By distributing unique work across lanes, we can achieve zero-error filtering using <=1 texel evaluations per pixel given a sufficiently large magnification factor. For the remaining cases, we propose novel filtering fallback methods that also achieve higher quality than prior approaches.
>
---
#### [new 200] Auto-Regressive Surface Cutting
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于表面切割任务，旨在解决现有方法生成的纹理图过于碎片化、缺乏语义一致的问题。工作是提出SeamGPT模型，通过自回归方式生成连贯的切割线。**

- **链接: [http://arxiv.org/pdf/2506.18017v1](http://arxiv.org/pdf/2506.18017v1)**

> **作者:** Yang Li; Victor Cheung; Xinhai Liu; Yuguang Chen; Zhongjin Luo; Biwen Lei; Haohan Weng; Zibo Zhao; Jingwei Huang; Zhuo Chen; Chunchao Guo
>
> **备注:** Tech. report. https://victorcheung12.github.io/seamgpt
>
> **摘要:** Surface cutting is a fundamental task in computer graphics, with applications in UV parameterization, texture mapping, and mesh decomposition. However, existing methods often produce technically valid but overly fragmented atlases that lack semantic coherence. We introduce SeamGPT, an auto-regressive model that generates cutting seams by mimicking professional workflows. Our key technical innovation lies in formulating surface cutting as a next token prediction task: sample point clouds on mesh vertices and edges, encode them as shape conditions, and employ a GPT-style transformer to sequentially predict seam segments with quantized 3D coordinates. Our approach achieves exceptional performance on UV unwrapping benchmarks containing both manifold and non-manifold meshes, including artist-created, and 3D-scanned models. In addition, it enhances existing 3D segmentation tools by providing clean boundaries for part decomposition.
>
---
#### [new 201] LIGHTHOUSE: Fast and precise distance to shoreline calculations from anywhere on earth
- **分类: cs.DB; cs.CV; cs.LG**

- **简介: 该论文属于地理信息处理任务，旨在解决高精度海岸线距离计算问题。通过构建高分辨率数据集和优化算法Lighthouse，实现快速准确的全球海岸线距离计算。**

- **链接: [http://arxiv.org/pdf/2506.18842v1](http://arxiv.org/pdf/2506.18842v1)**

> **作者:** Patrick Beukema; Henry Herzog; Yawen Zhang; Hunter Pitelka; Favyen Bastani
>
> **备注:** 8 pages, 7 figures, 1 table, ICML 2025 ML4RS
>
> **摘要:** We introduce a new dataset and algorithm for fast and efficient coastal distance calculations from Anywhere on Earth (AoE). Existing global coastal datasets are only available at coarse resolution (e.g. 1-4 km) which limits their utility. Publicly available satellite imagery combined with computer vision enable much higher precision. We provide a global coastline dataset at 10 meter resolution, a 100+ fold improvement in precision over existing data. To handle the computational challenge of querying at such an increased scale, we introduce a new library: Layered Iterative Geospatial Hierarchical Terrain-Oriented Unified Search Engine (Lighthouse). Lighthouse is both exceptionally fast and resource-efficient, requiring only 1 CPU and 2 GB of RAM to achieve millisecond online inference, making it well suited for real-time applications in resource-constrained environments.
>
---
#### [new 202] Morse: Dual-Sampling for Lossless Acceleration of Diffusion Models
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于图像生成任务，旨在加速扩散模型。提出Morse框架，通过双模型协作实现无损加速，提升采样效率。**

- **链接: [http://arxiv.org/pdf/2506.18251v1](http://arxiv.org/pdf/2506.18251v1)**

> **作者:** Chao Li; Jiawei Fan; Anbang Yao
>
> **备注:** This work is accepted to ICML 2025. The project page: https://github.com/deep-optimization/Morse
>
> **摘要:** In this paper, we present Morse, a simple dual-sampling framework for accelerating diffusion models losslessly. The key insight of Morse is to reformulate the iterative generation (from noise to data) process via taking advantage of fast jump sampling and adaptive residual feedback strategies. Specifically, Morse involves two models called Dash and Dot that interact with each other. The Dash model is just the pre-trained diffusion model of any type, but operates in a jump sampling regime, creating sufficient space for sampling efficiency improvement. The Dot model is significantly faster than the Dash model, which is learnt to generate residual feedback conditioned on the observations at the current jump sampling point on the trajectory of the Dash model, lifting the noise estimate to easily match the next-step estimate of the Dash model without jump sampling. By chaining the outputs of the Dash and Dot models run in a time-interleaved fashion, Morse exhibits the merit of flexibly attaining desired image generation performance while improving overall runtime efficiency. With our proposed weight sharing strategy between the Dash and Dot models, Morse is efficient for training and inference. Our method shows a lossless speedup of 1.78X to 3.31X on average over a wide range of sampling step budgets relative to 9 baseline diffusion models on 6 image generation tasks. Furthermore, we show that our method can be also generalized to improve the Latent Consistency Model (LCM-SDXL, which is already accelerated with consistency distillation technique) tailored for few-step text-to-image synthesis. The code and models are available at https://github.com/deep-optimization/Morse.
>
---
#### [new 203] No Training Wheels: Steering Vectors for Bias Correction at Inference Time
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于分类模型偏见修正任务，旨在解决数据分布不均导致的分类偏差问题。提出一种无需训练、在推理时使用方向向量修正偏见的方法。**

- **链接: [http://arxiv.org/pdf/2506.18598v1](http://arxiv.org/pdf/2506.18598v1)**

> **作者:** Aviral Gupta; Armaan Sethi; Ameesh Sethi
>
> **摘要:** Neural network classifiers trained on datasets with uneven group representation often inherit class biases and learn spurious correlations. These models may perform well on average but consistently fail on atypical groups. For example, in hair color classification, datasets may over-represent females with blond hair, reinforcing stereotypes. Although various algorithmic and data-centric methods have been proposed to address such biases, they often require retraining or significant compute. In this work, we propose a cheap, training-free method inspired by steering vectors used to edit behaviors in large language models. We compute the difference in mean activations between majority and minority groups to define a "bias vector," which we subtract from the model's residual stream. This leads to reduced classification bias and improved worst-group accuracy. We explore multiple strategies for extracting and applying these vectors in transformer-like classifiers, showing that steering vectors, traditionally used in generative models, can also be effective in classification. More broadly, we showcase an extremely cheap, inference time, training free method to mitigate bias in classification models.
>
---
#### [new 204] General-Purpose Robotic Navigation via LVLM-Orchestrated Perception, Reasoning, and Acting
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人导航任务，解决未知环境中通用导航问题。提出ARNA框架，结合LVLM实现自主感知、推理与行动，无需预设地图或人工规划。**

- **链接: [http://arxiv.org/pdf/2506.17462v1](http://arxiv.org/pdf/2506.17462v1)**

> **作者:** Bernard Lange; Anil Yildiz; Mansur Arief; Shehryar Khattak; Mykel Kochenderfer; Georgios Georgakis
>
> **摘要:** Developing general-purpose navigation policies for unknown environments remains a core challenge in robotics. Most existing systems rely on task-specific neural networks and fixed data flows, limiting generalizability. Large Vision-Language Models (LVLMs) offer a promising alternative by embedding human-like knowledge suitable for reasoning and planning. Yet, prior LVLM-robot integrations typically depend on pre-mapped spaces, hard-coded representations, and myopic exploration. We introduce the Agentic Robotic Navigation Architecture (ARNA), a general-purpose navigation framework that equips an LVLM-based agent with a library of perception, reasoning, and navigation tools available within modern robotic stacks. At runtime, the agent autonomously defines and executes task-specific workflows that iteratively query the robotic modules, reason over multimodal inputs, and select appropriate navigation actions. This approach enables robust navigation and reasoning in previously unmapped environments, providing a new perspective on robotic stack design. Evaluated in Habitat Lab on the HM-EQA benchmark, ARNA achieves state-of-the-art performance, demonstrating effective exploration, navigation, and embodied question answering without relying on handcrafted plans, fixed input representations, or pre-existing maps.
>
---
#### [new 205] RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文属于双臂机器人操作任务，旨在解决合成数据不足与仿真环境简化的问题。提出RoboTwin 2.0框架，实现大规模、多样化的数据生成与评估。**

- **链接: [http://arxiv.org/pdf/2506.18088v1](http://arxiv.org/pdf/2506.18088v1)**

> **作者:** Tianxing Chen; Zanxin Chen; Baijun Chen; Zijian Cai; Yibin Liu; Qiwei Liang; Zixuan Li; Xianliang Lin; Yiheng Ge; Zhenyu Gu; Weiliang Deng; Yubin Guo; Tian Nian; Xuanbing Xie; Qiangyu Chen; Kailun Su; Tianling Xu; Guodong Liu; Mengkang Hu; Huan-ang Gao; Kaixuan Wang; Zhixuan Liang; Yusen Qin; Xiaokang Yang; Ping Luo; Yao Mu
>
> **备注:** Project Page: https://robotwin-platform.github.io/
>
> **摘要:** Simulation-based data synthesis has emerged as a powerful paradigm for enhancing real-world robotic manipulation. However, existing synthetic datasets remain insufficient for robust bimanual manipulation due to two challenges: (1) the lack of an efficient, scalable data generation method for novel tasks, and (2) oversimplified simulation environments that fail to capture real-world complexity. We present RoboTwin 2.0, a scalable simulation framework that enables automated, large-scale generation of diverse and realistic data, along with unified evaluation protocols for dual-arm manipulation. We first construct RoboTwin-OD, a large-scale object library comprising 731 instances across 147 categories, each annotated with semantic and manipulation-relevant labels. Building on this foundation, we develop an expert data synthesis pipeline that combines multimodal large language models (MLLMs) with simulation-in-the-loop refinement to generate task-level execution code automatically. To improve sim-to-real transfer, RoboTwin 2.0 incorporates structured domain randomization along five axes: clutter, lighting, background, tabletop height and language instructions, thereby enhancing data diversity and policy robustness. We instantiate this framework across 50 dual-arm tasks spanning five robot embodiments, and pre-collect over 100,000 domain-randomized expert trajectories. Empirical results show a 10.9% gain in code generation success and improved generalization to novel real-world scenarios. A VLA model fine-tuned on our dataset achieves a 367% relative improvement (42.0% vs. 9.0%) on unseen scene real-world tasks, while zero-shot models trained solely on our synthetic data achieve a 228% relative gain, highlighting strong generalization without real-world supervision. We release the data generator, benchmark, dataset, and code to support scalable research in robust bimanual manipulation.
>
---
#### [new 206] EASE: Embodied Active Event Perception via Self-Supervised Energy Minimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人感知任务，旨在解决动态环境中事件感知的适应性问题。提出EASE框架，通过自监督学习实现无需标注数据的事件检测与跟踪。**

- **链接: [http://arxiv.org/pdf/2506.17516v1](http://arxiv.org/pdf/2506.17516v1)**

> **作者:** Zhou Chen; Sanjoy Kundu; Harsimran S. Baweja; Sathyanarayanan N. Aakur
>
> **备注:** Accepted to IEEE Robotics and Automation Letters, 2025
>
> **摘要:** Active event perception, the ability to dynamically detect, track, and summarize events in real time, is essential for embodied intelligence in tasks such as human-AI collaboration, assistive robotics, and autonomous navigation. However, existing approaches often depend on predefined action spaces, annotated datasets, and extrinsic rewards, limiting their adaptability and scalability in dynamic, real-world scenarios. Inspired by cognitive theories of event perception and predictive coding, we propose EASE, a self-supervised framework that unifies spatiotemporal representation learning and embodied control through free energy minimization. EASE leverages prediction errors and entropy as intrinsic signals to segment events, summarize observations, and actively track salient actors, operating without explicit annotations or external rewards. By coupling a generative perception model with an action-driven control policy, EASE dynamically aligns predictions with observations, enabling emergent behaviors such as implicit memory, target continuity, and adaptability to novel environments. Extensive evaluations in simulation and real-world settings demonstrate EASE's ability to achieve privacy-preserving and scalable event perception, providing a robust foundation for embodied systems in unscripted, dynamic tasks.
>
---
#### [new 207] Adapting Vision-Language Models for Evaluating World Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于评估任务，解决世界模型滚动生成的评价问题。通过适配视觉语言模型，提出UNIVERSE方法，实现高效、语义感知的评估。**

- **链接: [http://arxiv.org/pdf/2506.17967v1](http://arxiv.org/pdf/2506.17967v1)**

> **作者:** Mariya Hendriksen; Tabish Rashid; David Bignell; Raluca Georgescu; Abdelhak Lemkhenter; Katja Hofmann; Sam Devlin; Sarah Parisot
>
> **摘要:** World models -- generative models that simulate environment dynamics conditioned on past observations and actions -- are gaining prominence in planning, simulation, and embodied AI. However, evaluating their rollouts remains a fundamental challenge, requiring fine-grained, temporally grounded assessment of action alignment and semantic consistency -- capabilities not captured by existing metrics. Vision-Language Models (VLMs) have shown promise as automatic evaluators of generative content due to their strong multimodal reasoning abilities. Yet, their use in fine-grained, temporally sensitive evaluation tasks remains limited and requires targeted adaptation. We introduce a evaluation protocol targeting two recognition tasks -- action recognition and character recognition -- each assessed across binary, multiple-choice, and open-ended formats. To support this, we present UNIVERSE (UNIfied Vision-language Evaluator for Rollouts in Simulated Environments), a method for adapting VLMs to rollout evaluation under data and compute constraints. We conduct a large-scale study comparing full, partial, and parameter-efficient finetuning across task formats, context lengths, sampling strategies, and data compositions. The resulting unified evaluator matches the performance of task-specific baselines using a single checkpoint. Human studies confirm strong alignment with human judgments, establishing UNIVERSE as a scalable, semantics-aware evaluator for world models.
>
---
#### [new 208] Learning to Adapt Frozen CLIP for Few-Shot Test-Time Domain Adaptation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于少样本测试时域适应任务，旨在提升冻结CLIP模型在新域的性能。通过输入空间学习和特征融合，增强模型对目标域的适应能力。**

- **链接: [http://arxiv.org/pdf/2506.17307v1](http://arxiv.org/pdf/2506.17307v1)**

> **作者:** Zhixiang Chi; Li Gu; Huan Liu; Ziqiang Wang; Yanan Wu; Yang Wang; Konstantinos N Plataniotis
>
> **备注:** ICLR2025,https://github.com/chi-chi-zx/L2C
>
> **摘要:** Few-shot Test-Time Domain Adaptation focuses on adapting a model at test time to a specific domain using only a few unlabeled examples, addressing domain shift. Prior methods leverage CLIP's strong out-of-distribution (OOD) abilities by generating domain-specific prompts to guide its generalized, frozen features. However, since downstream datasets are not explicitly seen by CLIP, solely depending on the feature space knowledge is constrained by CLIP's prior knowledge. Notably, when using a less robust backbone like ViT-B/16, performance significantly drops on challenging real-world benchmarks. Departing from the state-of-the-art of inheriting the intrinsic OOD capability of CLIP, this work introduces learning directly on the input space to complement the dataset-specific knowledge for frozen CLIP. Specifically, an independent side branch is attached in parallel with CLIP and enforced to learn exclusive knowledge via revert attention. To better capture the dataset-specific label semantics for downstream adaptation, we propose to enhance the inter-dispersion among text features via greedy text ensemble and refinement. The text and visual features are then progressively fused in a domain-aware manner by a generated domain prompt to adapt toward a specific domain. Extensive experiments show our method's superiority on 5 large-scale benchmarks (WILDS and DomainNet), notably improving over smaller networks like ViT-B/16 with gains of \textbf{+5.1} in F1 for iWildCam and \textbf{+3.1\%} in WC Acc for FMoW.
>
---
#### [new 209] MTSIC: Multi-stage Transformer-based GAN for Spectral Infrared Image Colorization
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于红外图像彩色化任务，旨在解决TIR图像缺乏颜色和纹理的问题。提出MTSIC框架，利用多阶段Transformer网络提升色彩还原与语义准确性。**

- **链接: [http://arxiv.org/pdf/2506.17540v1](http://arxiv.org/pdf/2506.17540v1)**

> **作者:** Tingting Liu; Yuan Liu; Jinhui Tang; Liyin Yuan; Chengyu Liu; Chunlai Li; Xiubao Sui; Qian Chen
>
> **摘要:** Thermal infrared (TIR) images, acquired through thermal radiation imaging, are unaffected by variations in lighting conditions and atmospheric haze. However, TIR images inherently lack color and texture information, limiting downstream tasks and potentially causing visual fatigue. Existing colorization methods primarily rely on single-band images with limited spectral information and insufficient feature extraction capabilities, which often result in image distortion and semantic ambiguity. In contrast, multiband infrared imagery provides richer spectral data, facilitating the preservation of finer details and enhancing semantic accuracy. In this paper, we propose a generative adversarial network (GAN)-based framework designed to integrate spectral information to enhance the colorization of infrared images. The framework employs a multi-stage spectral self-attention Transformer network (MTSIC) as the generator. Each spectral feature is treated as a token for self-attention computation, and a multi-head self-attention mechanism forms a spatial-spectral attention residual block (SARB), achieving multi-band feature mapping and reducing semantic confusion. Multiple SARB units are integrated into a Transformer-based single-stage network (STformer), which uses a U-shaped architecture to extract contextual information, combined with multi-scale wavelet blocks (MSWB) to align semantic information in the spatial-frequency dual domain. Multiple STformer modules are cascaded to form MTSIC, progressively optimizing the reconstruction quality. Experimental results demonstrate that the proposed method significantly outperforms traditional techniques and effectively enhances the visual quality of infrared images.
>
---
#### [new 210] What You Think Is What You Get: Bridge User Intent and Transfer Function Design through Multimodal Large Language Models
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于可视化任务，解决用户意图与传输函数设计之间的语义鸿沟问题。通过多模态大语言模型引导优化，提升传输函数设计的有效性和通用性。**

- **链接: [http://arxiv.org/pdf/2506.18407v1](http://arxiv.org/pdf/2506.18407v1)**

> **作者:** Yiyao Wang; Bo Pan; Ke Wang; Han Liu; Jinyuan Mao; Yuxin Liu; Minfeng Zhu; Bo Zhang; Weifeng Chen; Xiuqi Huang; Wei Chen
>
> **摘要:** Direct volume rendering (DVR) is a fundamental technique for visualizing volumetric data, with transfer functions (TFs) playing a crucial role in extracting meaningful structures. However, designing effective TFs remains unintuitive due to the semantic gap between user intent and TF parameter space. Researchers have developed numerous TF optimization methods to bridge this gap. However, existing methods still face two challenges: large exploration space and weak generalizability. To address these issues, we propose What You Think is What You Get (WYTWYG) framework, which leveraging Multi-model Large Language Models (MLLMs) to guide the TF optimization based on user intent. Specifically, we first introduce a novel TF optimization approach comprising two core components: (1) an evolution-based explorer for effective exploration of the TF space, and (2) a volume rendering quality evaluator based on MLLMs to provide generalizable visual guidance. We further propose a TF interactive design system based on this approach. We demonstrate the general applicability of our framework through three case studies, and validate the effectiveness of each component through extensive experiments. Our code is available at: https://github.com/wyysteelhead/TFevolve.
>
---
#### [new 211] TDACloud: Point Cloud Recognition Using Topological Data Analysis
- **分类: cs.RO; cs.CG; cs.CV**

- **简介: 该论文属于点云识别任务，旨在解决噪声和变换下点云匹配难题。提出TDACloud方法，利用拓扑数据分析提取局部描述符，无需GPU训练，效果优于基线。**

- **链接: [http://arxiv.org/pdf/2506.18725v1](http://arxiv.org/pdf/2506.18725v1)**

> **作者:** Anirban Ghosh; Ian Dahlin; Ayan Dutta
>
> **摘要:** Point cloud-based object/place recognition remains a problem of interest in applications such as autonomous driving, scene reconstruction, and localization. Extracting meaningful local descriptors from a query point cloud that can be matched with the descriptors of the collected point clouds is a challenging problem. Furthermore, when the query point cloud is noisy or has been transformed (e.g., rotated), it adds to the complexity. To this end, we propose a novel methodology, named TDACloud, using Topological Data Analysis (TDA) for local descriptor extraction from a point cloud, which does not need resource-intensive GPU-based machine learning training. More specifically, we used the ATOL vectorization method to generate vectors for point clouds. Unlike voxelization, our proposed technique can take raw point clouds as inputs and outputs a fixed-size TDA-descriptor vector. To test the quality of the proposed TDACloud technique, we have implemented it on multiple real-world (e.g., Oxford RobotCar, KITTI-360) and realistic (e.g., ShapeNet) point cloud datasets for object and place recognition. We have also tested TDACloud on noisy and transformed test cases where the query point cloud has been scaled, translated, or rotated. Our results demonstrate high recognition accuracies in noisy conditions and large-scale real-world place recognition while outperforming the baselines by up to approximately 14%.
>
---
#### [new 212] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于4D重建任务，旨在解决单目视频重建中的未见区域和深度模糊问题。通过生成模型修正并补全动态场景表示，提升重建效果。**

- **链接: [http://arxiv.org/pdf/2506.18601v1](http://arxiv.org/pdf/2506.18601v1)**

> **作者:** Denys Rozumnyi; Jonathon Luiten; Numair Khan; Johannes Schönberger; Peter Kontschieder
>
> **摘要:** Transforming casually captured, monocular videos into fully immersive dynamic experiences is a highly ill-posed task, and comes with significant challenges, e.g., reconstructing unseen regions, and dealing with the ambiguity in monocular depth estimation. In this work we introduce BulletGen, an approach that takes advantage of generative models to correct errors and complete missing information in a Gaussian-based dynamic scene representation. This is done by aligning the output of a diffusion-based video generation model with the 4D reconstruction at a single frozen "bullet-time" step. The generated frames are then used to supervise the optimization of the 4D Gaussian model. Our method seamlessly blends generative content with both static and dynamic scene components, achieving state-of-the-art results on both novel-view synthesis, and 2D/3D tracking tasks.
>
---
#### [new 213] PCaM: A Progressive Focus Attention-Based Information Fusion Method for Improving Vision Transformer Domain Adaptation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于无监督域适应任务，针对视觉Transformer中前景对象不匹配问题，提出PCaM机制以提升跨域注意力一致性与特征融合效果。**

- **链接: [http://arxiv.org/pdf/2506.17232v1](http://arxiv.org/pdf/2506.17232v1)**

> **作者:** Zelin Zang; Fei Wang; Liangyu Li; Jinlin Wu; Chunshui Zhao; Zhen Lei; Baigui Sun
>
> **摘要:** Unsupervised Domain Adaptation (UDA) aims to transfer knowledge from a labeled source domain to an unlabeled target domain. Recent UDA methods based on Vision Transformers (ViTs) have achieved strong performance through attention-based feature alignment. However, we identify a key limitation: foreground object mismatch, where the discrepancy in foreground object size and spatial distribution across domains weakens attention consistency and hampers effective domain alignment. To address this issue, we propose the Progressive Focus Cross-Attention Mechanism (PCaM), which progressively filters out background information during cross-attention, allowing the model to focus on and fuse discriminative foreground semantics across domains. We further introduce an attentional guidance loss that explicitly directs attention toward task-relevant regions, enhancing cross-domain attention consistency. PCaM is lightweight, architecture-agnostic, and easy to integrate into existing ViT-based UDA pipelines. Extensive experiments on Office-Home, DomainNet, VisDA-2017, and remote sensing datasets demonstrate that PCaM significantly improves adaptation performance and achieves new state-of-the-art results, validating the effectiveness of attention-guided foreground fusion for domain adaptation.
>
---
#### [new 214] Chain-of-Memory: Enhancing GUI Agents for Cross-Application Navigation
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于GUI代理任务，旨在解决跨应用导航中任务状态理解与信息存储的问题。提出Chain-of-Memory方法，通过显式记忆建模提升性能，并构建了相关数据集。**

- **链接: [http://arxiv.org/pdf/2506.18158v1](http://arxiv.org/pdf/2506.18158v1)**

> **作者:** Xinzge Gao; Chuanrui Hu; Bin Chen; Teng Li
>
> **摘要:** Multimodal large language models (MLLMs) are attracting growing attention in the development of Graphical User Interface (GUI) agents. Existing approaches often rely on historical screenshots or actions to implicitly represent the task state. This reliance poses challenges for GUI agents in accurately understanding task states and underscores the absence of effective mechanisms to store critical information in complex and lengthy cross-app tasks. To address these challenges, we propose Chain-of-Memory (CoM), a novel approach for explicitly modeling short-term and long-term memory in GUI agents. CoM achieves this by capturing action descriptions, integrating task-relevant screen information, and maintaining a dedicated memory module to store and manage this information. By leveraging explicit memory representations, CoM enables GUI agents to better understand task states and retain critical historical information persistently. To equip GUI agents with memory management capabilities and evaluate the effectiveness of CoM, we developed the GUI Odyssey-CoM, a dataset comprising 111k screen-action pairs annotated with Chain-of-Memory. Experimental results demonstrate that CoM significantly improves GUI agents' performance in cross-application tasks. Additionally, GUI Odyssey-CoM enables 7B models to achieve memory management capabilities comparable to 72B models. The dataset and code will be open-sourced.
>
---
#### [new 215] h-calibration: Rethinking Classifier Recalibration with Probabilistic Error-Bounded Objective
- **分类: cs.LG; cs.AI; cs.CV; math.PR; stat.ML**

- **简介: 该论文属于模型校准任务，解决深度学习模型概率输出不可靠的问题。通过提出h-calibration框架，实现更准确的校准，提升分类可靠性。**

- **链接: [http://arxiv.org/pdf/2506.17968v1](http://arxiv.org/pdf/2506.17968v1)**

> **作者:** Wenjian Huang; Guiping Cao; Jiahao Xia; Jingkun Chen; Hao Wang; Jianguo Zhang
>
> **摘要:** Deep neural networks have demonstrated remarkable performance across numerous learning tasks but often suffer from miscalibration, resulting in unreliable probability outputs. This has inspired many recent works on mitigating miscalibration, particularly through post-hoc recalibration methods that aim to obtain calibrated probabilities without sacrificing the classification performance of pre-trained models. In this study, we summarize and categorize previous works into three general strategies: intuitively designed methods, binning-based methods, and methods based on formulations of ideal calibration. Through theoretical and practical analysis, we highlight ten common limitations in previous approaches. To address these limitations, we propose a probabilistic learning framework for calibration called h-calibration, which theoretically constructs an equivalent learning formulation for canonical calibration with boundedness. On this basis, we design a simple yet effective post-hoc calibration algorithm. Our method not only overcomes the ten identified limitations but also achieves markedly better performance than traditional methods, as validated by extensive experiments. We further analyze, both theoretically and experimentally, the relationship and advantages of our learning objective compared to traditional proper scoring rule. In summary, our probabilistic framework derives an approximately equivalent differentiable objective for learning error-bounded calibrated probabilities, elucidating the correspondence and convergence properties of computational statistics with respect to theoretical bounds in canonical calibration. The theoretical effectiveness is verified on standard post-hoc calibration benchmarks by achieving state-of-the-art performance. This research offers valuable reference for learning reliable likelihood in related fields.
>
---
#### [new 216] LVPNet: A Latent-variable-based Prediction-driven End-to-end Framework for Lossless Compression of Medical Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像无损压缩任务，解决现有方法中潜在变量利用效率低的问题，提出LVPNet框架，结合全局潜在变量和量化补偿模块提升压缩效率。**

- **链接: [http://arxiv.org/pdf/2506.17983v1](http://arxiv.org/pdf/2506.17983v1)**

> **作者:** Chenyue Song; Chen Hui; Qing Lin; Wei Zhang; Siqiao Li; Shengping Zhang; Haiqi Zhu; Zhixuan Li; Shaohui Liu; Feng Jiang; Xiang Li
>
> **备注:** Accepted to MICCAI 2025
>
> **摘要:** Autoregressive Initial Bits is a framework that integrates sub-image autoregression and latent variable modeling, demonstrating its advantages in lossless medical image compression. However, in existing methods, the image segmentation process leads to an even distribution of latent variable information across each sub-image, which in turn causes posterior collapse and inefficient utilization of latent variables. To deal with these issues, we propose a prediction-based end-to-end lossless medical image compression method named LVPNet, leveraging global latent variables to predict pixel values and encoding predicted probabilities for lossless compression. Specifically, we introduce the Global Multi-scale Sensing Module (GMSM), which extracts compact and informative latent representations from the entire image, effectively capturing spatial dependencies within the latent space. Furthermore, to mitigate the information loss introduced during quantization, we propose the Quantization Compensation Module (QCM), which learns the distribution of quantization errors and refines the quantized features to compensate for quantization loss. Extensive experiments on challenging benchmarks demonstrate that our method achieves superior compression efficiency compared to state-of-the-art lossless image compression approaches, while maintaining competitive inference speed. The code is at https://github.com/Anonymity00000/Anonymity-repository/.
>
---
#### [new 217] ConciseHint: Boosting Efficient Reasoning via Continuous Concise Hints during Generation
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于自然语言处理中的推理任务，旨在解决大模型生成过程冗长的问题。通过引入连续简洁提示框架ConciseHint，提升推理效率并保持性能。**

- **链接: [http://arxiv.org/pdf/2506.18810v1](http://arxiv.org/pdf/2506.18810v1)**

> **作者:** Siao Tang; Xinyin Ma; Gongfan Fang; Xinchao Wang
>
> **备注:** Codes are available at https://github.com/tsa18/ConciseHint
>
> **摘要:** Recent advancements in large reasoning models (LRMs) like DeepSeek-R1 and OpenAI o1 series have achieved notable performance enhancements on complex reasoning tasks by scaling up the generation length by Chain-of-Thought (CoT). However, an emerging issue is their inclination to produce excessively verbose reasoning processes, leading to the inefficiency problem. Existing literature on improving efficiency mainly adheres to the before-reasoning paradigms such as prompting and reasoning or fine-tuning and reasoning, but ignores the promising direction of directly encouraging the model to speak concisely by intervening during the generation of reasoning. In order to fill the blank, we propose a framework dubbed ConciseHint, which continuously encourages the reasoning model to speak concisely by injecting the textual hint (manually designed or trained on the concise data) during the token generation of the reasoning process. Besides, ConciseHint is adaptive to the complexity of the query by adaptively adjusting the hint intensity, which ensures it will not undermine model performance. Experiments on the state-of-the-art LRMs, including DeepSeek-R1 and Qwen-3 series, demonstrate that our method can effectively produce concise reasoning processes while maintaining performance well. For instance, we achieve a reduction ratio of 65\% for the reasoning length on GSM8K benchmark with Qwen-3 4B with nearly no accuracy loss.
>
---
#### [new 218] 3D Gaussian Splatting for Fine-Detailed Surface Reconstruction in Large-Scale Scene
- **分类: cs.GR; cs.CV; eess.IV**

- **简介: 该论文属于3D表面重建任务，解决大场景细粒度重建问题。通过分阶段策略、外观解耦和单视角正则化提升精度与效率。**

- **链接: [http://arxiv.org/pdf/2506.17636v1](http://arxiv.org/pdf/2506.17636v1)**

> **作者:** Shihan Chen; Zhaojin Li; Zeyu Chen; Qingsong Yan; Gaoyang Shen; Ran Duan
>
> **备注:** IROS 2025
>
> **摘要:** Recent developments in 3D Gaussian Splatting have made significant advances in surface reconstruction. However, scaling these methods to large-scale scenes remains challenging due to high computational demands and the complex dynamic appearances typical of outdoor environments. These challenges hinder the application in aerial surveying and autonomous driving. This paper proposes a novel solution to reconstruct large-scale surfaces with fine details, supervised by full-sized images. Firstly, we introduce a coarse-to-fine strategy to reconstruct a coarse model efficiently, followed by adaptive scene partitioning and sub-scene refining from image segments. Additionally, we integrate a decoupling appearance model to capture global appearance variations and a transient mask model to mitigate interference from moving objects. Finally, we expand the multi-view constraint and introduce a single-view regularization for texture-less areas. Our experiments were conducted on the publicly available dataset GauU-Scene V2, which was captured using unmanned aerial vehicles. To the best of our knowledge, our method outperforms existing NeRF-based and Gaussian-based methods, achieving high-fidelity visual results and accurate surface from full-size image optimization. Open-source code will be available on GitHub.
>
---
#### [new 219] Pitfalls of Conformal Predictions for Medical Image Classification
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于医学图像分类任务，探讨了置信预测在医疗领域的局限性，指出其在分布偏移和小类别场景下的不可靠性。**

- **链接: [http://arxiv.org/pdf/2506.18162v1](http://arxiv.org/pdf/2506.18162v1)**

> **作者:** Hendrik Mehrtens; Tabea Bucher; Titus J. Brinker
>
> **摘要:** Reliable uncertainty estimation is one of the major challenges for medical classification tasks. While many approaches have been proposed, recently the statistical framework of conformal predictions has gained a lot of attention, due to its ability to provide provable calibration guarantees. Nonetheless, the application of conformal predictions in safety-critical areas such as medicine comes with pitfalls, limitations and assumptions that practitioners need to be aware of. We demonstrate through examples from dermatology and histopathology that conformal predictions are unreliable under distributional shifts in input and label variables. Additionally, conformal predictions should not be used for selecting predictions to improve accuracy and are not reliable for subsets of the data, such as individual classes or patient attributes. Moreover, in classification settings with a small number of classes, which are common in medical image classification tasks, conformal predictions have limited practical value.
>
---
#### [new 220] Can Common VLMs Rival Medical VLMs? Evaluation and Strategic Insights
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医疗影像任务，探讨通用视觉语言模型（VLM）能否通过微调 rival 医疗专用VLM。研究评估了模型在疾病诊断和视觉问答中的表现，发现通用VLM经微调后可达到或超越医疗专用模型。**

- **链接: [http://arxiv.org/pdf/2506.17337v1](http://arxiv.org/pdf/2506.17337v1)**

> **作者:** Yuan Zhong; Ruinan Jin; Xiaoxiao Li; Qi Dou
>
> **摘要:** Medical vision-language models (VLMs) leverage large-scale pretraining for diverse imaging tasks but require substantial computational and data resources. Meanwhile, common or general-purpose VLMs (e.g., CLIP, LLaVA), though not trained for medical use, show promise with fine-tuning. This raises a key question: Can efficient fine-tuned common VLMs rival generalist medical VLMs for solving specific medical imaging tasks? This study systematically evaluates common and medical VLMs across disease diagnosis and visual question answering (VQA). Using CLIP-based and LLaVA-based models, we examine (1) off-the-shelf performance gaps in in-domain (ID) settings, (2) whether fine-tuning bridges these gaps, and (3) generalization to out-of-domain (OOD) tasks on unseen medical modalities. While medical-specific pretraining provides advantages in ID settings, common VLMs match or surpass medical-specific models after lightweight fine-tuning, with LoRA-based adaptation proving highly effective among different tasks. In OOD tasks, common VLMs demonstrate strong adaptability in some tasks, challenging the assumption that medical-specific pre-training is essential. These findings suggest that leveraging common VLMs with fine-tuning offers a scalable and cost-effective alternative to developing large-scale medical VLMs, providing crucial insights for future research in the medical imaging field.
>
---
#### [new 221] Reproducible Evaluation of Camera Auto-Exposure Methods in the Field: Platform, Benchmark and Lessons Learned
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决相机自动曝光方法的可复现评估问题。通过构建数据集和模拟器，实现离线 benchmark，验证了传统方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.18844v1](http://arxiv.org/pdf/2506.18844v1)**

> **作者:** Olivier Gamache; Jean-Michel Fortin; Matěj Boxan; François Pomerleau; Philippe Giguère
>
> **备注:** 19 pages, 11 figures, pre-print version of the accepted paper for IEEE Transactions on Field Robotics (T-FR)
>
> **摘要:** Standard datasets often present limitations, particularly due to the fixed nature of input data sensors, which makes it difficult to compare methods that actively adjust sensor parameters to suit environmental conditions. This is the case with Automatic-Exposure (AE) methods, which rely on environmental factors to influence the image acquisition process. As a result, AE methods have traditionally been benchmarked in an online manner, rendering experiments non-reproducible. Building on our prior work, we propose a methodology that utilizes an emulator capable of generating images at any exposure time. This approach leverages BorealHDR, a unique multi-exposure stereo dataset, along with its new extension, in which data was acquired along a repeated trajectory at different times of the day to assess the impact of changing illumination. In total, BorealHDR covers 13.4 km over 59 trajectories in challenging lighting conditions. The dataset also includes lidar-inertial-odometry-based maps with pose estimation for each image frame, as well as Global Navigation Satellite System (GNSS) data for comparison. We demonstrate that by using images acquired at various exposure times, we can emulate realistic images with a Root-Mean-Square Error (RMSE) below 1.78% compared to ground truth images. Using this offline approach, we benchmarked eight AE methods, concluding that the classical AE method remains the field's best performer. To further support reproducibility, we provide in-depth details on the development of our backpack acquisition platform, including hardware, electrical components, and performance specifications. Additionally, we share valuable lessons learned from deploying the backpack over more than 25 km across various environments. Our code and dataset are available online at this link: https://github.com/norlab-ulaval/TFR24 BorealHDR
>
---
#### [new 222] Pix2Geomodel: A Next-Generation Reservoir Geomodeling with Property-to-Property Translation
- **分类: physics.geo-ph; cs.CE; cs.CV; cs.LG; cs.NE**

- **简介: 该论文属于地质建模任务，旨在解决传统方法在复杂地下结构建模中的不足。通过生成对抗网络实现属性间的翻译与预测。**

- **链接: [http://arxiv.org/pdf/2506.17747v1](http://arxiv.org/pdf/2506.17747v1)**

> **作者:** Abdulrahman Al-Fakih; Ardiansyah Koeshidayatullah; Nabil A. Saraih; Tapan Mukerji; Rayan Kanfar; Abdulmohsen Alali; SanLinn I. Kaka
>
> **备注:** 34 pages, 13 figures
>
> **摘要:** Accurate geological modeling is critical for reservoir characterization, yet traditional methods struggle with complex subsurface heterogeneity, and they have problems with conditioning to observed data. This study introduces Pix2Geomodel, a novel conditional generative adversarial network (cGAN) framework based on Pix2Pix, designed to predict reservoir properties (facies, porosity, permeability, and water saturation) from the Rotliegend reservoir of the Groningen gas field. Utilizing a 7.6 million-cell dataset from the Nederlandse Aardolie Maatschappij, accessed via EPOS-NL, the methodology included data preprocessing, augmentation to generate 2,350 images per property, and training with a U-Net generator and PatchGAN discriminator over 19,000 steps. Evaluation metrics include pixel accuracy (PA), mean intersection over union (mIoU), frequency weighted intersection over union (FWIoU), and visualizations assessed performance in masked property prediction and property-to-property translation tasks. Results demonstrated high accuracy for facies (PA 0.88, FWIoU 0.85) and water saturation (PA 0.96, FWIoU 0.95), with moderate success for porosity (PA 0.70, FWIoU 0.55) and permeability (PA 0.74, FWIoU 0.60), and robust translation performance (e.g., facies-to-facies PA 0.98, FWIoU 0.97). The framework captured spatial variability and geological realism, as validated by variogram analysis, and calculated the training loss curves for the generator and discriminator for each property. Compared to traditional methods, Pix2Geomodel offers enhanced fidelity in direct property mapping. Limitations include challenges with microstructural variability and 2D constraints, suggesting future integration of multi-modal data and 3D modeling (Pix2Geomodel v2.0). This study advances the application of generative AI in geoscience, supporting improved reservoir management and open science initiatives.
>
---
## 更新

#### [replaced 001] MIFNet: Learning Modality-Invariant Features for Generalizable Multimodal Image Matching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.11299v2](http://arxiv.org/pdf/2501.11299v2)**

> **作者:** Yepeng Liu; Zhichao Sun; Baosheng Yu; Yitian Zhao; Bo Du; Yongchao Xu; Jun Cheng
>
> **备注:** Accept by IEEE TIP 2025
>
> **摘要:** Many keypoint detection and description methods have been proposed for image matching or registration. While these methods demonstrate promising performance for single-modality image matching, they often struggle with multimodal data because the descriptors trained on single-modality data tend to lack robustness against the non-linear variations present in multimodal data. Extending such methods to multimodal image matching often requires well-aligned multimodal data to learn modality-invariant descriptors. However, acquiring such data is often costly and impractical in many real-world scenarios. To address this challenge, we propose a modality-invariant feature learning network (MIFNet) to compute modality-invariant features for keypoint descriptions in multimodal image matching using only single-modality training data. Specifically, we propose a novel latent feature aggregation module and a cumulative hybrid aggregation module to enhance the base keypoint descriptors trained on single-modality data by leveraging pre-trained features from Stable Diffusion models. %, our approach generates robust and invariant features across diverse and unknown modalities. We validate our method with recent keypoint detection and description methods in three multimodal retinal image datasets (CF-FA, CF-OCT, EMA-OCTA) and two remote sensing datasets (Optical-SAR and Optical-NIR). Extensive experiments demonstrate that the proposed MIFNet is able to learn modality-invariant feature for multimodal image matching without accessing the targeted modality and has good zero-shot generalization ability. The code will be released at https://github.com/lyp-deeplearning/MIFNet.
>
---
#### [replaced 002] Ultra-high resolution multimodal MRI densely labelled holistic structural brain atlas
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.16879v3](http://arxiv.org/pdf/2501.16879v3)**

> **作者:** José V. Manjón; Sergio Morell-Ortega; Marina Ruiz-Perez; Boris Mansencal; Edern Le Bot; Marien Gadea; Enrique Lanuza; Gwenaelle Catheline; Thomas Tourdias; Vincent Planche; Rémi Giraud; Denis Rivière; Jean-François Mangin; Nicole Labra-Avila; Roberto Vivo-Hernando; Gregorio Rubio; Fernando Aparici; Maria de la Iglesia-Vaya; Pierrick Coupé
>
> **摘要:** In this paper, we introduce a novel structural holistic Atlas (holiAtlas) of the human brain anatomy based on multimodal and high-resolution MRI that covers several anatomical levels from the organ to the substructure level, using a new densely labelled protocol generated from the fusion of multiple local protocols at different scales. This atlas was constructed by averaging images and segmentations of 75 healthy subjects from the Human Connectome Project database. Specifically, MR images of T1, T2 and WMn (White Matter nulled) contrasts at 0.125 $mm^{3}$ resolution were selected for this project. The images of these 75 subjects were nonlinearly registered and averaged using symmetric group-wise normalisation to construct the atlas. At the finest level, the proposed atlas has 350 different labels derived from 7 distinct delineation protocols. These labels were grouped at multiple scales, offering a coherent and consistent holistic representation of the brain across different levels of detail. This multiscale and multimodal atlas can be used to develop new ultra-high-resolution segmentation methods, potentially improving the early detection of neurological disorders. We make it publicly available to the scientific community.
>
---
#### [replaced 003] Discrete JEPA: Learning Discrete Token Representations without Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14373v2](http://arxiv.org/pdf/2506.14373v2)**

> **作者:** Junyeob Baek; Hosung Lee; Christopher Hoang; Mengye Ren; Sungjin Ahn
>
> **摘要:** The cornerstone of cognitive intelligence lies in extracting hidden patterns from observations and leveraging these principles to systematically predict future outcomes. However, current image tokenization methods demonstrate significant limitations in tasks requiring symbolic abstraction and logical reasoning capabilities essential for systematic inference. To address this challenge, we propose Discrete-JEPA, extending the latent predictive coding framework with semantic tokenization and novel complementary objectives to create robust tokenization for symbolic reasoning tasks. Discrete-JEPA dramatically outperforms baselines on visual symbolic prediction tasks, while striking visual evidence reveals the spontaneous emergence of deliberate systematic patterns within the learned semantic token space. Though an initial model, our approach promises a significant impact for advancing Symbolic world modeling and planning capabilities in artificial intelligence systems.
>
---
#### [replaced 004] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09283v2](http://arxiv.org/pdf/2503.09283v2)**

> **作者:** Xiangbin Wei; Yuanfeng Wang; Ao XU; Lingyu Zhu; Dongyong Sun; Keren Li; Yang Li; Qi Qin
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2502.16826
>
> **摘要:** Building on recent advances in Bayesian statistics and image denoising, we propose Noise2Score3D, a fully unsupervised framework for point cloud denoising. Noise2Score3D learns the score function of the underlying point cloud distribution directly from noisy data, eliminating the need for clean data during training. Using Tweedie's formula, our method performs denoising in a single step, avoiding the iterative processes used in existing unsupervised methods, thus improving both accuracy and efficiency. Additionally, we introduce Total Variation for Point Clouds as a denoising quality metric, which allows for the estimation of unknown noise parameters. Experimental results demonstrate that Noise2Score3D achieves state-of-the-art performance on standard benchmarks among unsupervised learning methods in Chamfer distance and point-to-mesh metrics. Noise2Score3D also demonstrates strong generalization ability beyond training datasets. Our method, by addressing the generalization issue and challenge of the absence of clean data in learning-based methods, paves the way for learning-based point cloud denoising methods in real-world applications.
>
---
#### [replaced 005] Emergent Temporal Correspondences from Video Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17220v2](http://arxiv.org/pdf/2506.17220v2)**

> **作者:** Jisu Nam; Soowon Son; Dahyun Chung; Jiyoung Kim; Siyoon Jin; Junhwa Hur; Seungryong Kim
>
> **备注:** Project page is available at https://cvlab-kaist.github.io/DiffTrack
>
> **摘要:** Recent advancements in video diffusion models based on Diffusion Transformers (DiTs) have achieved remarkable success in generating temporally coherent videos. Yet, a fundamental question persists: how do these models internally establish and represent temporal correspondences across frames? We introduce DiffTrack, the first quantitative analysis framework designed to answer this question. DiffTrack constructs a dataset of prompt-generated video with pseudo ground-truth tracking annotations and proposes novel evaluation metrics to systematically analyze how each component within the full 3D attention mechanism of DiTs (e.g., representations, layers, and timesteps) contributes to establishing temporal correspondences. Our analysis reveals that query-key similarities in specific, but not all, layers play a critical role in temporal matching, and that this matching becomes increasingly prominent during the denoising process. We demonstrate practical applications of DiffTrack in zero-shot point tracking, where it achieves state-of-the-art performance compared to existing vision foundation and self-supervised video models. Further, we extend our findings to motion-enhanced video generation with a novel guidance method that improves temporal consistency of generated videos without additional training. We believe our work offers crucial insights into the inner workings of video DiTs and establishes a foundation for further research and applications leveraging their temporal understanding.
>
---
#### [replaced 006] SurgSora: Object-Aware Diffusion Model for Controllable Surgical Video Generation
- **分类: cs.CV; cs.AI; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.14018v3](http://arxiv.org/pdf/2412.14018v3)**

> **作者:** Tong Chen; Shuya Yang; Junyi Wang; Long Bai; Hongliang Ren; Luping Zhou
>
> **备注:** MICCAI 2025
>
> **摘要:** Surgical video generation can enhance medical education and research, but existing methods lack fine-grained motion control and realism. We introduce SurgSora, a framework that generates high-fidelity, motion-controllable surgical videos from a single input frame and user-specified motion cues. Unlike prior approaches that treat objects indiscriminately or rely on ground-truth segmentation masks, SurgSora leverages self-predicted object features and depth information to refine RGB appearance and optical flow for precise video synthesis. It consists of three key modules: (1) the Dual Semantic Injector, which extracts object-specific RGB-D features and segmentation cues to enhance spatial representations; (2) the Decoupled Flow Mapper, which fuses multi-scale optical flow with semantic features for realistic motion dynamics; and (3) the Trajectory Controller, which estimates sparse optical flow and enables user-guided object movement. By conditioning these enriched features within the Stable Video Diffusion, SurgSora achieves state-of-the-art visual authenticity and controllability in advancing surgical video synthesis, as demonstrated by extensive quantitative and qualitative comparisons. Our human evaluation in collaboration with expert surgeons further demonstrates the high realism of SurgSora-generated videos, highlighting the potential of our method for surgical training and education. Our project is available at https://surgsora.github.io/surgsora.github.io.
>
---
#### [replaced 007] Leveraging Foundation Models for Content-Based Image Retrieval in Radiology
- **分类: cs.CV; cs.IR**

- **链接: [http://arxiv.org/pdf/2403.06567v4](http://arxiv.org/pdf/2403.06567v4)**

> **作者:** Stefan Denner; David Zimmerer; Dimitrios Bounias; Markus Bujotzek; Shuhan Xiao; Raphael Stock; Lisa Kausch; Philipp Schader; Tobias Penzkofer; Paul F. Jäger; Klaus Maier-Hein
>
> **摘要:** Content-based image retrieval (CBIR) has the potential to significantly improve diagnostic aid and medical research in radiology. However, current CBIR systems face limitations due to their specialization to certain pathologies, limiting their utility. On the other hand, several vision foundation models have been shown to produce general-purpose visual features. Therefore, in this work, we propose using vision foundation models as powerful and versatile off-the-shelf feature extractors for content-based image retrieval. Our contributions include: (1) benchmarking a diverse set of vision foundation models on an extensive dataset comprising 1.6 million 2D radiological images across four modalities and 161 pathologies; (2) identifying weakly-supervised models, particularly BiomedCLIP, as highly effective, achieving a achieving a P@1 of up to 0.594 (P@3: 0.590, P@5: 0.588, P@10: 0.583), comparable to specialized CBIR systems but without additional training; (3) conducting an in-depth analysis of the impact of index size on retrieval performance; (4) evaluating the quality of embedding spaces generated by different models; and (5) investigating specific challenges associated with retrieving anatomical versus pathological structures. Despite these challenges, our research underscores the vast potential of foundation models for CBIR in radiology, proposing a shift towards versatile, general-purpose medical image retrieval systems that do not require specific tuning. Our code, dataset splits and embeddings are publicly available under https://github.com/MIC-DKFZ/foundation-models-for-cbmir.
>
---
#### [replaced 008] Noise2Score3D:Unsupervised Tweedie's Approach for Point Cloud Denoising
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16826v3](http://arxiv.org/pdf/2502.16826v3)**

> **作者:** Xiangbin Wei
>
> **备注:** There is already another version: "Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising". Please see arXiv:2503.09283
>
> **摘要:** Building on recent advances in Bayesian statistics and image denoising, we propose Noise2Score3D, a fully unsupervised framework for point cloud denoising that addresses the critical challenge of limited availability of clean data. Noise2Score3D learns the gradient of the underlying point cloud distribution directly from noisy data, eliminating the need for clean data during training. By leveraging Tweedie's formula, our method performs inference in a single step, avoiding the iterative processes used in existing unsupervised methods, thereby improving both performance and efficiency. Experimental results demonstrate that Noise2Score3D achieves state-of-the-art performance on standard benchmarks, outperforming other unsupervised methods in Chamfer distance and point-to-mesh metrics, and rivaling some supervised approaches. Furthermore, Noise2Score3D demonstrates strong generalization ability beyond training datasets. Additionally, we introduce Total Variation for Point Cloud, a criterion that allows for the estimation of unknown noise parameters, which further enhances the method's versatility and real-world utility.
>
---
#### [replaced 009] VesselGPT: Autoregressive Modeling of Vascular Geometry
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.13318v2](http://arxiv.org/pdf/2505.13318v2)**

> **作者:** Paula Feldman; Martin Sinnona; Claudio Delrieux; Viviana Siless; Emmanuel Iarussi
>
> **备注:** Accepted for MICCAI 2025
>
> **摘要:** Anatomical trees are critical for clinical diagnosis and treatment planning, yet their complex and diverse geometry make accurate representation a significant challenge. Motivated by the latest advances in large language models, we introduce an autoregressive method for synthesizing anatomical trees. Our approach first embeds vessel structures into a learned discrete vocabulary using a VQ-VAE architecture, then models their generation autoregressively with a GPT-2 model. This method effectively captures intricate geometries and branching patterns, enabling realistic vascular tree synthesis. Comprehensive qualitative and quantitative evaluations reveal that our technique achieves high-fidelity tree reconstruction with compact discrete representations. Moreover, our B-spline representation of vessel cross-sections preserves critical morphological details that are often overlooked in previous' methods parameterizations. To the best of our knowledge, this work is the first to generate blood vessels in an autoregressive manner. Code is available at https://github.com/LIA-DiTella/VesselGPT-MICCAI.
>
---
#### [replaced 010] MM-R5: MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12364v2](http://arxiv.org/pdf/2506.12364v2)**

> **作者:** Mingjun Xu; Jinhan Dong; Jue Hou; Zehui Wang; Sihang Li; Zhifeng Gao; Renxin Zhong; Hengxing Cai
>
> **摘要:** Multimodal document retrieval systems enable information access across text, images, and layouts, benefiting various domains like document-based question answering, report analysis, and interactive content summarization. Rerankers improve retrieval precision by reordering retrieved candidates. However, current multimodal reranking methods remain underexplored, with significant room for improvement in both training strategies and overall effectiveness. Moreover, the lack of explicit reasoning makes it difficult to analyze and optimize these methods further. In this paper, We propose MM-R5, a MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval, aiming to provide a more effective and reliable solution for multimodal reranking tasks. MM-R5 is trained in two stages: supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we focus on improving instruction-following and guiding the model to generate complete and high-quality reasoning chains. To support this, we introduce a novel data construction strategy that produces rich, high-quality reasoning data. In the RL stage, we design a task-specific reward framework, including a reranking reward tailored for multimodal candidates and a composite template-based reward to further refine reasoning quality. We conduct extensive experiments on MMDocIR, a challenging public benchmark spanning multiple domains. MM-R5 achieves state-of-the-art performance on most metrics and delivers comparable results to much larger models on the remaining ones. Moreover, compared to the best retrieval-only method, MM-R5 improves recall@1 by over 4%. These results validate the effectiveness of our reasoning-enhanced training pipeline. Our code is available at https://github.com/i2vec/MM-R5 .
>
---
#### [replaced 011] Memorization to Generalization: Emergence of Diffusion Models from Associative Memory
- **分类: cs.LG; cond-mat.dis-nn; cs.CV; q-bio.NC; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.21777v2](http://arxiv.org/pdf/2505.21777v2)**

> **作者:** Bao Pham; Gabriel Raya; Matteo Negri; Mohammed J. Zaki; Luca Ambrogioni; Dmitry Krotov
>
> **摘要:** Hopfield networks are associative memory (AM) systems, designed for storing and retrieving patterns as local minima of an energy landscape. In the classical Hopfield model, an interesting phenomenon occurs when the amount of training data reaches its critical memory load $- spurious\,\,states$, or unintended stable points, emerge at the end of the retrieval dynamics, leading to incorrect recall. In this work, we examine diffusion models, commonly used in generative modeling, from the perspective of AMs. The training phase of diffusion model is conceptualized as memory encoding (training data is stored in the memory). The generation phase is viewed as an attempt of memory retrieval. In the small data regime the diffusion model exhibits a strong memorization phase, where the network creates distinct basins of attraction around each sample in the training set, akin to the Hopfield model below the critical memory load. In the large data regime, a different phase appears where an increase in the size of the training set fosters the creation of new attractor states that correspond to manifolds of the generated samples. Spurious states appear at the boundary of this transition and correspond to emergent attractor states, which are absent in the training set, but, at the same time, have distinct basins of attraction around them. Our findings provide: a novel perspective on the memorization-generalization phenomenon in diffusion models via the lens of AMs, theoretical prediction of existence of spurious states, empirical validation of this prediction in commonly-used diffusion models.
>
---
#### [replaced 012] A Prior-Guided Joint Diffusion Model in Projection Domain for PET Tracer Conversion
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16733v2](http://arxiv.org/pdf/2506.16733v2)**

> **作者:** Fang Chen; Weifeng Zhang; Xingyu Ai; BingXuan Li; An Li; Qiegen Liu
>
> **摘要:** Positron emission tomography (PET) is widely used to assess metabolic activity, but its application is limited by the availability of radiotracers. 18F-labeled fluorodeoxyglucose (18F-FDG) is the most commonly used tracer but shows limited effectiveness for certain tumors. In contrast, 6-18F-fluoro-3,4-dihydroxy-L-phenylalanine (18F-DOPA) offers higher specificity for neuroendocrine tumors and neurological disorders. However, the complexity of its synthesis process and constraints on transportation time have limited its clinical application. Among different forms of raw data acquired by the scanner, sinogram is a commonly used representation in PET imaging. Therefore, modeling in projection domain enables more direct utilization of the original information, potentially reducing the accumulation errors during the image reconstruction process. Inspired by these factors, this study proposes a prior-guided joint diffusion model (PJDM) for transforming 18F-FDG PET sinograms into 18F-DOPA PET sinograms. During inference, an initial synthetic 18F-DOPA PET sinogram is first generated using a higher-order hybrid sampler. This sinogram is then degraded and serves as an additional condition to guide the iterative refinement process. Experimental results demonstrated that PJDM effectively improved both sinogram quality and the final synthetic outcomes. The code is available at: https://github.com/yqx7150/PJDM.
>
---
#### [replaced 013] MGHF: Multi-Granular High-Frequency Perceptual Loss for Image Super-Resolution
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.13548v2](http://arxiv.org/pdf/2411.13548v2)**

> **作者:** Shoaib Meraj Sami; Md Mahedi Hasan; Mohammad Saeed Ebrahimi Saadabadi; Jeremy Dawson; Nasser Nasrabadi; Raghuveer Rao
>
> **备注:** 14 pages
>
> **摘要:** While different variants of perceptual losses have been employed in super-resolution literature to synthesize more realistic, appealing, and detailed high-resolution images, most are convolutional neural networks-based, causing information loss during guidance and often relying on complicated architectures and training procedures. We propose an invertible neural network (INN)-based naive \textbf{M}ulti-\textbf{G}ranular \textbf{H}igh-\textbf{F}requency (MGHF-n) perceptual loss trained on ImageNet to overcome these issues. Furthermore, we develop a comprehensive framework (MGHF-c) with several constraints to preserve, prioritize, and regularize information across multiple perspectives: texture and style preservation, content preservation, regional detail preservation, and joint content-style regularization. Information is prioritized through adaptive entropy-based pruning and reweighting of INN features. We utilize Gram matrix loss for style preservation and mean-squared error loss for content preservation. Additionally, we propose content-style consistency through correlation loss to regulate unnecessary texture generation while preserving content information. Since small image regions may contain intricate details, we employ modulated PatchNCE in the INN features as a local information preservation objective. Extensive experiments on various super-resolution algorithms, including GAN- and diffusion-based methods, demonstrate that our MGHF framework significantly improves performance. After the review process, our code will be released in the public repository.
>
---
#### [replaced 014] Shaken, Not Stirred: A Novel Dataset for Visual Understanding of Glasses in Human-Robot Bartending Tasks
- **分类: cs.RO; cs.CV; 68T40; I.2.9; I.4.8**

- **链接: [http://arxiv.org/pdf/2503.04308v2](http://arxiv.org/pdf/2503.04308v2)**

> **作者:** Lukáš Gajdošech; Hassan Ali; Jan-Gerrit Habekost; Martin Madaras; Matthias Kerzel; Stefan Wermter
>
> **备注:** Submitted and Accepted for Presentation at the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Datasets for object detection often do not account for enough variety of glasses, due to their transparent and reflective properties. Specifically, open-vocabulary object detectors, widely used in embodied robotic agents, fail to distinguish subclasses of glasses. This scientific gap poses an issue to robotic applications that suffer from accumulating errors between detection, planning, and action execution. The paper introduces a novel method for the acquisition of real-world data from RGB-D sensors that minimizes human effort. We propose an auto-labeling pipeline that generates labels for all the acquired frames based on the depth measurements. We provide a novel real-world glass object dataset that was collected on the Neuro-Inspired COLlaborator (NICOL), a humanoid robot platform. The data set consists of 7850 images recorded from five different cameras. We show that our trained baseline model outperforms state-of-the-art open-vocabulary approaches. In addition, we deploy our baseline model in an embodied agent approach to the NICOL platform, on which it achieves a success rate of 81% in a human-robot bartending scenario.
>
---
#### [replaced 015] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06659v2](http://arxiv.org/pdf/2506.06659v2)**

> **作者:** Wenhao Yao; Zhenxin Li; Shiyi Lan; Zi Wang; Xinglong Sun; Jose M. Alvarez; Zuxuan Wu
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** In complex driving environments, autonomous vehicles must navigate safely. Relying on a single predicted path, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each, but face optimization challenges in precisely selecting the best option from thousands of possibilities and distinguishing subtle but safety-critical differences, especially in rare or underrepresented scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, demonstrating superior safetycritical capabilities, including collision avoidance and compliance with rules, while maintaining high trajectory quality in various driving scenarios.
>
---
#### [replaced 016] Evaluating Sensitivity Parameters in Smartphone-Based Gaze Estimation: A Comparative Study of Appearance-Based and Infrared Eye Trackers
- **分类: cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.11932v3](http://arxiv.org/pdf/2506.11932v3)**

> **作者:** Nishan Gunawardena; Gough Yumu Lui; Bahman Javadi; Jeewani Anupama Ginige
>
> **摘要:** This study evaluates a smartphone-based, deep-learning eye-tracking algorithm by comparing its performance against a commercial infrared-based eye tracker, the Tobii Pro Nano. The aim is to investigate the feasibility of appearance-based gaze estimation under realistic mobile usage conditions. Key sensitivity factors, including age, gender, vision correction, lighting conditions, device type, and head position, were systematically analysed. The appearance-based algorithm integrates a lightweight convolutional neural network (MobileNet-V3) with a recurrent structure (Long Short-Term Memory) to predict gaze coordinates from grayscale facial images. Gaze data were collected from 51 participants using dynamic visual stimuli, and accuracy was measured using Euclidean distance. The deep learning model produced a mean error of 17.76 mm, compared to 16.53 mm for the Tobii Pro Nano. While overall accuracy differences were small, the deep learning-based method was more sensitive to factors such as lighting, vision correction, and age, with higher failure rates observed under low-light conditions among participants using glasses and in older age groups. Device-specific and positional factors also influenced tracking performance. These results highlight the potential of appearance-based approaches for mobile eye tracking and offer a reference framework for evaluating gaze estimation systems across varied usage conditions.
>
---
#### [replaced 017] Multi-label Scene Classification for Autonomous Vehicles: Acquiring and Accumulating Knowledge from Diverse Datasets
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17101v2](http://arxiv.org/pdf/2506.17101v2)**

> **作者:** Ke Li; Chenyu Zhang; Yuxin Ding; Xianbiao Hu; Ruwen Qin
>
> **摘要:** Driving scene identification, which assigns multiple non-exclusive class labels to a scene, provides the contextual awareness necessary for enhancing autonomous vehicles' ability to understand, reason about, and interact with the complex driving environment. As a multi-label classification problem, it is better tackled via multitasking learning. However, directly training a multi-label classification model for driving scene identification through multitask learning presents two main challenges: acquiring a balanced, comprehensively annotated multi-label dataset and balancing learning across different tasks. This paper introduces a novel learning system that synergizes knowledge acquisition and accumulation (KAA) with consistency-based active learning (CAL) to address those challenges. KAA acquires and accumulates knowledge about scene identification from various single-label datasets via monotask learning. Subsequently, CAL effectively resolves the knowledge gap caused by the discrepancy between single-label and multi-label data. An ablation study on our Driving Scene Identification (DSI) dataset demonstrates a 56.1% performance increase over the baseline model pretrained on ImageNet. Of this, KAA accounts for 31.3% of the gain, and CAL contributes 24.8%. Moreover, KAA-CAL stands out as the best performer when compared to state-of-the-art (SOTA) multi-label models on two public datasets, BDD100K and HSD, achieving this while using 85% less data. The DSI dataset and the implementation code for KAA-CAL are available at https://github.com/KELISBU/KAA-CAL .
>
---
#### [replaced 018] Multi-contrast laser endoscopy for in vivo gastrointestinal imaging
- **分类: eess.IV; cs.CV; physics.med-ph; physics.optics**

- **链接: [http://arxiv.org/pdf/2505.10492v2](http://arxiv.org/pdf/2505.10492v2)**

> **作者:** Taylor L. Bobrow; Mayank Golhar; Suchapa Arayakarnkul; Anthony A. Song; Saowanee Ngamruengphong; Nicholas J. Durr
>
> **摘要:** White light endoscopy is the clinical gold standard for detecting diseases in the gastrointestinal tract. Most applications involve identifying visual abnormalities in tissue color, texture, and shape. Unfortunately, the contrast of these features is often subtle, causing many clinically relevant cases to go undetected. To overcome this challenge, we introduce Multi-contrast Laser Endoscopy (MLE): a platform for widefield clinical imaging with rapidly tunable spectral, coherent, and directional illumination. We demonstrate three capabilities of MLE: enhancing tissue chromophore contrast with multispectral diffuse reflectance, quantifying blood flow using laser speckle contrast imaging, and characterizing mucosal topography using photometric stereo. We validate MLE with benchtop models, then demonstrate MLE in vivo during clinical colonoscopies. MLE images from 31 polyps demonstrate an approximate three-fold improvement in contrast and a five-fold improvement in color difference compared to white light and narrow band imaging. With the ability to reveal multiple complementary types of tissue contrast while seamlessly integrating into the clinical environment, MLE shows promise as an investigative tool to improve gastrointestinal imaging.
>
---
#### [replaced 019] Recent Trends in Artificial Intelligence Technology: A Scoping Review
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2305.04532v3](http://arxiv.org/pdf/2305.04532v3)**

> **作者:** Teemu Niskanen; Tuomo Sipola; Olli Väänänen
>
> **摘要:** Artificial intelligence is more ubiquitous in multiple domains. Smartphones, social media platforms, search engines, and autonomous vehicles are just a few examples of applications that utilize artificial intelligence technologies to enhance their performance. This study carries out a scoping review of the current state-of-the-art artificial intelligence technologies following the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) framework. The goal was to find the most advanced technologies used in different domains of artificial intelligence technology research. Three recognized journals were used from artificial intelligence and machine learning domain: Journal of Artificial Intelligence Research, Journal of Machine Learning Research, and Machine Learning, and articles published in 2022 were observed. Certain qualifications were laid for the technological solutions: the technology must be tested against comparable solutions, commonly approved or otherwise well justified datasets must be used while applying, and results must show improvements against comparable solutions. One of the most important parts of the technology development appeared to be how to process and exploit the data gathered from multiple sources. The data can be highly unstructured, and the technological solution should be able to utilize the data with minimum manual work from humans. The results of this review indicate that creating labeled datasets is very laborious, and solutions exploiting unsupervised or semi-supervised learning technologies are more and more researched. The learning algorithms should be able to be updated efficiently, and predictions should be interpretable. Using artificial intelligence technologies in real-world applications, safety and explainable predictions are mandatory to consider before mass adoption can occur.
>
---
#### [replaced 020] PotatoGANs: Utilizing Generative Adversarial Networks, Instance Segmentation, and Explainable AI for Enhanced Potato Disease Identification and Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.07332v2](http://arxiv.org/pdf/2405.07332v2)**

> **作者:** Fatema Tuj Johora Faria; Mukaffi Bin Moin; Mohammad Shafiul Alam; Ahmed Al Wase; Md. Rabius Sani; Khan Md Hasib
>
> **摘要:** Numerous applications have resulted from the automation of agricultural disease segmentation using deep learning techniques. However, when applied to new conditions, these applications frequently face the difficulty of overfitting, resulting in lower segmentation performance. In the context of potato farming, where diseases have a large influence on yields, it is critical for the agricultural economy to quickly and properly identify these diseases. Traditional data augmentation approaches, such as rotation, flip, and translation, have limitations and frequently fail to provide strong generalization results. To address these issues, our research employs a novel approach termed as PotatoGANs. In this novel data augmentation approach, two types of Generative Adversarial Networks (GANs) are utilized to generate synthetic potato disease images from healthy potato images. This approach not only expands the dataset but also adds variety, which helps to enhance model generalization. Using the Inception score as a measure, our experiments show the better quality and realisticness of the images created by PotatoGANs, emphasizing their capacity to resemble real disease images closely. The CycleGAN model outperforms the Pix2Pix GAN model in terms of image quality, as evidenced by its higher IS scores CycleGAN achieves higher Inception scores (IS) of 1.2001 and 1.0900 for black scurf and common scab, respectively. This synthetic data can significantly improve the training of large neural networks. It also reduces data collection costs while enhancing data diversity and generalization capabilities. Our work improves interpretability by combining three gradient-based Explainable AI algorithms (GradCAM, GradCAM++, and ScoreCAM) with three distinct CNN architectures (DenseNet169, Resnet152 V2, InceptionResNet V2) for potato disease classification.
>
---
#### [replaced 021] Open-world machine learning: A review and new outlooks
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.01759v4](http://arxiv.org/pdf/2403.01759v4)**

> **作者:** Fei Zhu; Shijie Ma; Zhen Cheng; Xu-Yao Zhang; Zhaoxiang Zhang; Dacheng Tao; Cheng-Lin Liu
>
> **摘要:** Machine learning has achieved remarkable success in many applications. However, existing studies are largely based on the closed-world assumption, which assumes that the environment is stationary, and the model is fixed once deployed. In many real-world applications, this fundamental and rather naive assumption may not hold because an open environment is complex, dynamic, and full of unknowns. In such cases, rejecting unknowns, discovering novelties, and then continually learning them, could enable models to be safe and evolve continually as biological systems do. This article presents a holistic view of open-world machine learning by investigating unknown rejection, novelty discovery, and continual learning in a unified paradigm. The challenges, principles, and limitations of current methodologies are discussed in detail. Furthermore, widely used benchmarks, metrics, and performances are summarized. Finally, we discuss several potential directions for further progress in the field. By providing a comprehensive introduction to the emerging open-world machine learning paradigm, this article aims to help researchers build more powerful AI systems in their respective fields, and to promote the development of artificial general intelligence.
>
---
#### [replaced 022] CLIP-GS: CLIP-Informed Gaussian Splatting for View-Consistent 3D Indoor Semantic Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.14249v2](http://arxiv.org/pdf/2404.14249v2)**

> **作者:** Guibiao Liao; Jiankun Li; Zhenyu Bao; Xiaoqing Ye; Qing Li; Kanglin Liu
>
> **备注:** ACM TOMM 2025
>
> **摘要:** Exploiting 3D Gaussian Splatting (3DGS) with Contrastive Language-Image Pre-Training (CLIP) models for open-vocabulary 3D semantic understanding of indoor scenes has emerged as an attractive research focus. Existing methods typically attach high-dimensional CLIP semantic embeddings to 3D Gaussians and leverage view-inconsistent 2D CLIP semantics as Gaussian supervision, resulting in efficiency bottlenecks and deficient 3D semantic consistency. To address these challenges, we present CLIP-GS, efficiently achieving a coherent semantic understanding of 3D indoor scenes via the proposed Semantic Attribute Compactness (SAC) and 3D Coherent Regularization (3DCR). SAC approach exploits the naturally unified semantics within objects to learn compact, yet effective, semantic Gaussian representations, enabling highly efficient rendering (>100 FPS). 3DCR enforces semantic consistency in 2D and 3D domains: In 2D, 3DCR utilizes refined view-consistent semantic outcomes derived from 3DGS to establish cross-view coherence constraints; in 3D, 3DCR encourages features similar among 3D Gaussian primitives associated with the same object, leading to more precise and coherent segmentation results. Extensive experimental results demonstrate that our method remarkably suppresses existing state-of-the-art approaches, achieving mIoU improvements of 21.20% and 13.05% on ScanNet and Replica datasets, respectively, while maintaining real-time rendering speed. Furthermore, our approach exhibits superior performance even with sparse input data, substantiating its robustness.
>
---
#### [replaced 023] FARCLUSS: Fuzzy Adaptive Rebalancing and Contrastive Uncertainty Learning for Semi-Supervised Semantic Segmentation
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.11142v2](http://arxiv.org/pdf/2506.11142v2)**

> **作者:** Ebenezer Tarubinga; Jenifer Kalafatovich; Seong-Whan Lee
>
> **备注:** Submitted to Neural Networks
>
> **摘要:** Semi-supervised semantic segmentation (SSSS) faces persistent challenges in effectively leveraging unlabeled data, such as ineffective utilization of pseudo-labels, exacerbation of class imbalance biases, and neglect of prediction uncertainty. Current approaches often discard uncertain regions through strict thresholding favouring dominant classes. To address these limitations, we introduce a holistic framework that transforms uncertainty into a learning asset through four principal components: (1) fuzzy pseudo-labeling, which preserves soft class distributions from top-K predictions to enrich supervision; (2) uncertainty-aware dynamic weighting, that modulate pixel-wise contributions via entropy-based reliability scores; (3) adaptive class rebalancing, which dynamically adjust losses to counteract long-tailed class distributions; and (4) lightweight contrastive regularization, that encourage compact and discriminative feature embeddings. Extensive experiments on benchmarks demonstrate that our method outperforms current state-of-the-art approaches, achieving significant improvements in the segmentation of under-represented classes and ambiguous regions.
>
---
#### [replaced 024] Boosting Virtual Agent Learning and Reasoning: A Step-Wise, Multi-Dimensional, and Generalist Reward Model with Benchmark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18665v2](http://arxiv.org/pdf/2503.18665v2)**

> **作者:** Bingchen Miao; Yang Wu; Minghe Gao; Qifan Yu; Wendong Bu; Wenqiao Zhang; Yunfei Li; Siliang Tang; Tat-Seng Chua; Juncheng Li
>
> **备注:** Home page is available at https://dcd-ant-similar.github.io
>
> **摘要:** The development of Generalist Virtual Agents (GVAs) has shown significant promise in autonomous task execution. However, current training paradigms face critical limitations, including reliance on outcome supervision and labor-intensive human annotations. To address these challenges, we propose Similar, a Step-Wise Multi-Dimensional Generalist Reward Model, which offers fine-grained signals for agent training and can choose better action for inference-time scaling. Specifically, we begin by systematically defining five dimensions for evaluating agent actions. Building on this framework, we design an MCTS-P algorithm to automatically collect and annotate step-wise, five-dimensional agent execution data. Using this data, we train Similar with the Triple-M strategy. Furthermore, we introduce the first benchmark in the virtual agent domain for step-wise, multi-dimensional reward model training and evaluation, named SRM. This benchmark consists of two components: SRMTrain, which serves as the training set for Similar, and SRMEval, a manually selected test set for evaluating the reward model. Experimental results demonstrate that Similar, through its step-wise, multi-dimensional assessment and synergistic gain, provides GVAs with effective intermediate signals during both training and inference-time scaling. The project is available at https://github.com/antgroup/Similar.
>
---
#### [replaced 025] CLIP-HandID: Vision-Language Model for Hand-Based Person Identification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12447v2](http://arxiv.org/pdf/2506.12447v2)**

> **作者:** Nathanael L. Baisa; Babu Pallam; Amudhavel Jayavel
>
> **摘要:** This paper introduces a novel approach to person identification using hand images, designed specifically for criminal investigations. The method is particularly valuable in serious crimes such as sexual abuse, where hand images are often the only identifiable evidence available. Our proposed method, CLIP-HandID, leverages a pre-trained foundational vision-language model - CLIP - to efficiently learn discriminative deep feature representations from hand images (input to CLIP's image encoder) using textual prompts as semantic guidance. Since hand images are labeled with indexes rather than text descriptions, we employ a textual inversion network to learn pseudo-tokens that encode specific visual contexts or appearance attributes. These learned pseudo-tokens are then incorporated into textual prompts, which are fed into CLIP's text encoder to leverage its multi-modal reasoning and enhance generalization for identification. Through extensive evaluations on two large, publicly available hand datasets with multi-ethnic representation, we demonstrate that our method significantly outperforms existing approaches.
>
---
#### [replaced 026] R3eVision: A Survey on Robust Rendering, Restoration, and Enhancement for 3D Low-Level Vision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16262v2](http://arxiv.org/pdf/2506.16262v2)**

> **作者:** Weeyoung Kwon; Jeahun Sung; Minkyu Jeon; Chanho Eom; Jihyong Oh
>
> **备注:** Please visit our project page at https://github.com/CMLab-Korea/Awesome-3D-Low-Level-Vision
>
> **摘要:** Neural rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved significant progress in photorealistic 3D scene reconstruction and novel view synthesis. However, most existing models assume clean and high-resolution (HR) multi-view inputs, which limits their robustness under real-world degradations such as noise, blur, low-resolution (LR), and weather-induced artifacts. To address these limitations, the emerging field of 3D Low-Level Vision (3D LLV) extends classical 2D Low-Level Vision tasks including super-resolution (SR), deblurring, weather degradation removal, restoration, and enhancement into the 3D spatial domain. This survey, referred to as R\textsuperscript{3}eVision, provides a comprehensive overview of robust rendering, restoration, and enhancement for 3D LLV by formalizing the degradation-aware rendering problem and identifying key challenges related to spatio-temporal consistency and ill-posed optimization. Recent methods that integrate LLV into neural rendering frameworks are categorized to illustrate how they enable high-fidelity 3D reconstruction under adverse conditions. Application domains such as autonomous driving, AR/VR, and robotics are also discussed, where reliable 3D perception from degraded inputs is critical. By reviewing representative methods, datasets, and evaluation protocols, this work positions 3D LLV as a fundamental direction for robust 3D content generation and scene-level reconstruction in real-world environments.
>
---
#### [replaced 027] Multi-level Compositional Feature Augmentation for Unbiased Scene Graph Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2308.06712v2](http://arxiv.org/pdf/2308.06712v2)**

> **作者:** Lin Li; Xingchen Li; Chong Sun; Chen Li; Long Chen
>
> **备注:** Journal version, extension of conference paper (arXiv:2308.06712v1)
>
> **摘要:** Scene Graph Generation (SGG) aims to detect all the visual relation triplets <sub, pred, obj> in a given image. With the emergence of various advanced techniques for better utilizing both the intrinsic and extrinsic information in each relation triplet, SGG has achieved great progress over the recent years. However, due to the ubiquitous long-tailed predicate distributions, today's SGG models are still easily biased to the head predicates. Currently, the most prevalent debiasing solutions for SGG are re-balancing methods, e.g., changing the distributions of original training samples. In this paper, we argue that all existing re-balancing strategies fail to increase the diversity of the relation triplet features of each predicate, which is critical for robust SGG. To this end, we propose a novel Multi-level Compositional Feature Augmentation (MCFA) strategy, which aims to mitigate the bias issue from the perspective of increasing the diversity of triplet features. Specifically, we enhance relationship diversity on not only feature-level, i.e., replacing the intrinsic or extrinsic visual features of triplets with other correlated samples to create novel feature compositions for tail predicates, but also image-level, i.e., manipulating the image to generate brand new visual appearance for triplets. Due to its model-agnostic nature, MCFA can be seamlessly incorporated into various SGG frameworks. Extensive ablations have shown that MCFA achieves a new state-of-the-art performance on the trade-off between different metrics.
>
---
#### [replaced 028] Segmentation-Aware Generative Reinforcement Network (GRN) for Tissue Layer Segmentation in 3-D Ultrasound Images for Chronic Low-back Pain (cLBP) Assessment
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.17690v3](http://arxiv.org/pdf/2501.17690v3)**

> **作者:** Zixue Zeng; Xiaoyan Zhao; Matthew Cartier; Tong Yu; Jing Wang; Xin Meng; Zhiyu Sheng; Maryam Satarpour; John M Cormack; Allison Bean; Ryan Nussbaum; Maya Maurer; Emily Landis-Walkenhorst; Dinesh Kumbhare; Kang Kim; Ajay Wasan; Jiantao Pu
>
> **摘要:** We introduce a novel segmentation-aware joint training framework called generative reinforcement network (GRN) that integrates segmentation loss feedback to optimize both image generation and segmentation performance in a single stage. An image enhancement technique called segmentation-guided enhancement (SGE) is also developed, where the generator produces images tailored specifically for the segmentation model. Two variants of GRN were also developed, including GRN for sample-efficient learning (GRN-SEL) and GRN for semi-supervised learning (GRN-SSL). GRN's performance was evaluated using a dataset of 69 fully annotated 3D ultrasound scans from 29 subjects. The annotations included six anatomical structures: dermis, superficial fat, superficial fascial membrane (SFM), deep fat, deep fascial membrane (DFM), and muscle. Our results show that GRN-SEL with SGE reduces labeling efforts by up to 70% while achieving a 1.98% improvement in the Dice Similarity Coefficient (DSC) compared to models trained on fully labeled datasets. GRN-SEL alone reduces labeling efforts by 60%, GRN-SSL with SGE decreases labeling requirements by 70%, and GRN-SSL alone by 60%, all while maintaining performance comparable to fully supervised models. These findings suggest the effectiveness of the GRN framework in optimizing segmentation performance with significantly less labeled data, offering a scalable and efficient solution for ultrasound image analysis and reducing the burdens associated with data annotation.
>
---
#### [replaced 029] Improved Baselines with Synchronized Encoding for Universal Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.09886v4](http://arxiv.org/pdf/2408.09886v4)**

> **作者:** Sihan Yang; Jiadong Feng; Xuande Mi; Haixia Bi; Hai Zhang; Jian Sun
>
> **摘要:** Large foundation models, known for their strong zero-shot generalization capabilities, can be applied to a wide range of downstream tasks. However, developing foundation models for medical image segmentation poses a significant challenge due to the domain gap between natural and medical images. While fine-tuning techniques based on the Segment Anything Model (SAM) have been explored, they primarily focus on scaling up data or refining inference strategies without incorporating domain-specific architectural designs, limiting their zero-shot performance. To optimize segmentation performance under standard inference settings and provide a strong baseline for future research, we introduce SyncSAM, which employs a synchronized dual-branch encoder that integrates convolution and Transformer features in a synchronized manner to enhance medical image encoding, and a multi-scale dual-branch decoder to preserve image details. SyncSAM is trained on two of the largest medical image segmentation datasets, SA-Med2D-20M and IMed-361M, resulting in a series of pre-trained models for universal medical image segmentation. Experimental results demonstrate that SyncSAM not only achieves state-of-the-art performance on test sets but also exhibits strong zero-shot capabilities on unseen datasets. Code and checkpoints are available at https://github.com/Hhankyangg/SyncSAM.
>
---
#### [replaced 030] Cost-Aware Routing for Efficient Text-To-Image Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.14753v2](http://arxiv.org/pdf/2506.14753v2)**

> **作者:** Qinchan Li; Kenneth Chen; Changyue Su; Wittawat Jitkrittum; Qi Sun; Patsorn Sangkloy
>
> **摘要:** Diffusion models are well known for their ability to generate a high-fidelity image for an input prompt through an iterative denoising process. Unfortunately, the high fidelity also comes at a high computational cost due the inherently sequential generative process. In this work, we seek to optimally balance quality and computational cost, and propose a framework to allow the amount of computation to vary for each prompt, depending on its complexity. Each prompt is automatically routed to the most appropriate text-to-image generation function, which may correspond to a distinct number of denoising steps of a diffusion model, or a disparate, independent text-to-image model. Unlike uniform cost reduction techniques (e.g., distillation, model quantization), our approach achieves the optimal trade-off by learning to reserve expensive choices (e.g., 100+ denoising steps) only for a few complex prompts, and employ more economical choices (e.g., small distilled model) for less sophisticated prompts. We empirically demonstrate on COCO and DiffusionDB that by learning to route to nine already-trained text-to-image models, our approach is able to deliver an average quality that is higher than that achievable by any of these models alone.
>
---
#### [replaced 031] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.15698v2](http://arxiv.org/pdf/2506.15698v2)**

> **作者:** Yunhak Oh; Junseok Lee; Yeongmin Kim; Sangwoo Seo; Namkyeong Lee; Chanyoung Park
>
> **备注:** ICML 2025
>
> **摘要:** Spatially Resolved Transcriptomics (SRT) is a cutting-edge technique that captures the spatial context of cells within tissues, enabling the study of complex biological networks. Recent graph-based methods leverage both gene expression and spatial information to identify relevant spatial domains. However, these approaches fall short in obtaining meaningful spot representations, especially for spots near spatial domain boundaries, as they heavily emphasize adjacent spots that have minimal feature differences from an anchor node. To address this, we propose Spotscape, a novel framework that introduces the Similarity Telescope module to capture global relationships between multiple spots. Additionally, we propose a similarity scaling strategy to regulate the distances between intra- and inter-slice spots, facilitating effective multi-slice integration. Extensive experiments demonstrate the superiority of Spotscape in various downstream tasks, including single-slice and multi-slice scenarios. Our code is available at the following link: https: //github.com/yunhak0/Spotscape.
>
---
#### [replaced 032] An Exploratory Approach Towards Investigating and Explaining Vision Transformer and Transfer Learning for Brain Disease Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16039v3](http://arxiv.org/pdf/2505.16039v3)**

> **作者:** Shuvashis Sarker; Shamim Rahim Refat; Faika Fairuj Preotee; Shifat Islam; Tashreef Muhammad; Mohammad Ashraful Hoque
>
> **备注:** Accepted for publication in 2024 27th International Conference on Computer and Information Technology (ICCIT)
>
> **摘要:** The brain is a highly complex organ that manages many important tasks, including movement, memory and thinking. Brain-related conditions, like tumors and degenerative disorders, can be hard to diagnose and treat. Magnetic Resonance Imaging (MRI) serves as a key tool for identifying these conditions, offering high-resolution images of brain structures. Despite this, interpreting MRI scans can be complicated. This study tackles this challenge by conducting a comparative analysis of Vision Transformer (ViT) and Transfer Learning (TL) models such as VGG16, VGG19, Resnet50V2, MobilenetV2 for classifying brain diseases using MRI data from Bangladesh based dataset. ViT, known for their ability to capture global relationships in images, are particularly effective for medical imaging tasks. Transfer learning helps to mitigate data constraints by fine-tuning pre-trained models. Furthermore, Explainable AI (XAI) methods such as GradCAM, GradCAM++, LayerCAM, ScoreCAM, and Faster-ScoreCAM are employed to interpret model predictions. The results demonstrate that ViT surpasses transfer learning models, achieving a classification accuracy of 94.39%. The integration of XAI methods enhances model transparency, offering crucial insights to aid medical professionals in diagnosing brain diseases with greater precision.
>
---
#### [replaced 033] One Step Diffusion via Shortcut Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.12557v3](http://arxiv.org/pdf/2410.12557v3)**

> **作者:** Kevin Frans; Danijar Hafner; Sergey Levine; Pieter Abbeel
>
> **摘要:** Diffusion models and flow-matching models have enabled generating diverse and realistic images by learning to transfer noise to data. However, sampling from these models involves iterative denoising over many neural network passes, making generation slow and expensive. Previous approaches for speeding up sampling require complex training regimes, such as multiple training phases, multiple networks, or fragile scheduling. We introduce shortcut models, a family of generative models that use a single network and training phase to produce high-quality samples in a single or multiple sampling steps. Shortcut models condition the network not only on the current noise level but also on the desired step size, allowing the model to skip ahead in the generation process. Across a wide range of sampling step budgets, shortcut models consistently produce higher quality samples than previous approaches, such as consistency models and reflow. Compared to distillation, shortcut models reduce complexity to a single network and training phase and additionally allow varying step budgets at inference time.
>
---
#### [replaced 034] Training A Neural Network For Partially Occluded Road Sign Identification In The Context Of Autonomous Vehicles
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18177v2](http://arxiv.org/pdf/2503.18177v2)**

> **作者:** Gulnaz Gimaletdinova; Dim Shaiakhmetov; Madina Akpaeva; Mukhammadmuso Abduzhabbarov; Kadyrmamat Momunov
>
> **摘要:** The increasing number of autonomous vehicles and the rapid development of computer vision technologies underscore the particular importance of conducting research on the accuracy of traffic sign recognition. Numerous studies in this field have already achieved significant results, demonstrating high effectiveness in addressing traffic sign recognition tasks. However, the task becomes considerably more complex when a sign is partially obscured by surrounding objects, such as tree branches, billboards, or other elements of the urban environment. In our study, we investigated how partial occlusion of traffic signs affects their recognition. For this purpose, we collected a dataset comprising 5,746 images, including both fully visible and partially occluded signs, and made it publicly available. Using this dataset, we compared the performance of our custom convolutional neural network (CNN), which achieved 96% accuracy, with models trained using transfer learning. The best result was obtained by VGG16 with full layer unfreezing, reaching 99% accuracy. Additional experiments revealed that models trained solely on fully visible signs lose effectiveness when recognizing occluded signs. This highlights the critical importance of incorporating real-world data with partial occlusion into training sets to ensure robust model performance in complex practical scenarios and to enhance the safety of autonomous driving.
>
---
#### [replaced 035] Efficient Feature Aggregation and Scale-Aware Regression for Monocular 3D Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.02747v2](http://arxiv.org/pdf/2411.02747v2)**

> **作者:** Yifan Wang; Xiaochen Yang; Fanqi Pu; Qingmin Liao; Wenming Yang
>
> **摘要:** Monocular 3D object detection has attracted great attention due to simplicity and low cost. Existing methods typically follow conventional 2D detection paradigms, first locating object centers and then predicting 3D attributes via neighboring features. However, these methods predominantly rely on progressive cross-scale feature aggregation and focus solely on local information, which may result in a lack of global awareness and the omission of small-scale objects. In addition, due to large variation in object scales across different scenes and depths, inaccurate receptive fields often lead to background noise and degraded feature representation. To address these issues, we introduces MonoASRH, a novel monocular 3D detection framework composed of Efficient Hybrid Feature Aggregation Module (EH-FAM) and Adaptive Scale-Aware 3D Regression Head (ASRH). Specifically, EH-FAM employs multi-head attention with a global receptive field to extract semantic features for small-scale objects and leverages lightweight convolutional modules to efficiently aggregate visual features across different scales. The ASRH encodes 2D bounding box dimensions and then fuses scale features with the semantic features aggregated by EH-FAM through a scale-semantic feature fusion module. The scale-semantic feature fusion module guides ASRH in learning dynamic receptive field offsets, incorporating scale priors into 3D position prediction for better scale-awareness. Extensive experiments on the KITTI and Waymo datasets demonstrate that MonoASRH achieves state-of-the-art performance.
>
---
#### [replaced 036] Visual Prompt Engineering for Vision Language Models in Radiology
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.15802v3](http://arxiv.org/pdf/2408.15802v3)**

> **作者:** Stefan Denner; Markus Bujotzek; Dimitrios Bounias; David Zimmerer; Raphael Stock; Klaus Maier-Hein
>
> **备注:** Accepted at ECCV 2024 Workshop on Emergent Visual Abilities and Limits of Foundation Models & Medical Imaging with Deep Learning 2025
>
> **摘要:** Medical image classification plays a crucial role in clinical decision-making, yet most models are constrained to a fixed set of predefined classes, limiting their adaptability to new conditions. Contrastive Language-Image Pretraining (CLIP) offers a promising solution by enabling zero-shot classification through multimodal large-scale pretraining. However, while CLIP effectively captures global image content, radiology requires a more localized focus on specific pathology regions to enhance both interpretability and diagnostic accuracy. To address this, we explore the potential of incorporating visual cues into zero-shot classification, embedding visual markers, such as arrows, bounding boxes, and circles, directly into radiological images to guide model attention. Evaluating across four public chest X-ray datasets, we demonstrate that visual markers improve AUROC by up to 0.185, highlighting their effectiveness in enhancing classification performance. Furthermore, attention map analysis confirms that visual cues help models focus on clinically relevant areas, leading to more interpretable predictions.To support further research, we use public datasets and provide our codebase and preprocessing pipeline under https://github.com/MIC-DKFZ/VPE-in-Radiology, serving as a reference point for future work on localized classification in medical imaging.
>
---
#### [replaced 037] How Far is Video Generation from World Model: A Physical Law Perspective
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.02385v2](http://arxiv.org/pdf/2411.02385v2)**

> **作者:** Bingyi Kang; Yang Yue; Rui Lu; Zhijie Lin; Yang Zhao; Kaixin Wang; Gao Huang; Jiashi Feng
>
> **备注:** ICML 2025
>
> **摘要:** OpenAI's Sora highlights the potential of video generation for developing world models that adhere to fundamental physical laws. However, the ability of video generation models to discover such laws purely from visual data without human priors can be questioned. A world model learning the true law should give predictions robust to nuances and correctly extrapolate on unseen scenarios. In this work, we evaluate across three key scenarios: in-distribution, out-of-distribution, and combinatorial generalization. We developed a 2D simulation testbed for object movement and collisions to generate videos deterministically governed by one or more classical mechanics laws. This provides an unlimited supply of data for large-scale experimentation and enables quantitative evaluation of whether the generated videos adhere to physical laws. We trained diffusion-based video generation models to predict object movements based on initial frames. Our scaling experiments show perfect generalization within the distribution, measurable scaling behavior for combinatorial generalization, but failure in out-of-distribution scenarios. Further experiments reveal two key insights about the generalization mechanisms of these models: (1) the models fail to abstract general physical rules and instead exhibit "case-based" generalization behavior, i.e., mimicking the closest training example; (2) when generalizing to new cases, models are observed to prioritize different factors when referencing training data: color > size > velocity > shape. Our study suggests that scaling alone is insufficient for video generation models to uncover fundamental physical laws, despite its role in Sora's broader success. See our project page at https://phyworld.github.io
>
---
#### [replaced 038] PC-SRGAN: Physically Consistent Super-Resolution Generative Adversarial Network for General Transient Simulations
- **分类: eess.IV; cs.CE; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.06502v2](http://arxiv.org/pdf/2505.06502v2)**

> **作者:** Md Rakibul Hasan; Pouria Behnoudfar; Dan MacKinlay; Thomas Poulet
>
> **摘要:** Machine Learning, particularly Generative Adversarial Networks (GANs), has revolutionised Super Resolution (SR). However, generated images often lack physical meaningfulness, which is essential for scientific applications. Our approach, PC-SRGAN, enhances image resolution while ensuring physical consistency for interpretable simulations. PC-SRGAN significantly improves both the Peak Signal-to-Noise Ratio and the Structural Similarity Index Measure compared to conventional methods, even with limited training data (e.g., only 13% of training data required for SRGAN). Beyond SR, PC-SRGAN augments physically meaningful machine learning, incorporating numerically justified time integrators and advanced quality metrics. These advancements promise reliable and causal machine-learning models in scientific domains. A significant advantage of PC-SRGAN over conventional SR techniques is its physical consistency, which makes it a viable surrogate model for time-dependent problems. PC-SRGAN advances scientific machine learning, offering improved accuracy and efficiency for image processing, enhanced process understanding, and broader applications to scientific research. We publicly release the complete source code at https://github.com/hasan-rakibul/PC-SRGAN.
>
---
#### [replaced 039] RealSR-R1: Reinforcement Learning for Real-World Image Super-Resolution with Vision-Language Chain-of-Thought
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16796v2](http://arxiv.org/pdf/2506.16796v2)**

> **作者:** Junbo Qiao; Miaomiao Cai; Wei Li; Yutong Liu; Xudong Huang; Gaoqi He; Jiao Xie; Jie Hu; Xinghao Chen; Shaohui Lin
>
> **摘要:** Real-World Image Super-Resolution is one of the most challenging task in image restoration. However, existing methods struggle with an accurate understanding of degraded image content, leading to reconstructed results that are both low-fidelity and unnatural. We present RealSR-R1 in this work, which empowers the RealSR models with understanding and reasoning capabilities. Inspired by the success of Chain of Thought (CoT) in large language models (LLMs), we simulate the human process of handling degraded images and propose the VLCoT framework, which integrates vision and language reasoning. The framework aims to precisely restore image details by progressively generating more comprehensive text and higher-resolution images. To overcome the challenge of traditional supervised learning CoT failing to generalize to real-world scenarios, we introduce, for the first time, Group Relative Policy Optimization (GRPO) into the Real-World Image Super-Resolution task. We propose VLCoT-GRPO as a solution, which designs four reward functions: (1) Format reward, used to standardize the CoT process; (2) Degradation reward, to incentivize accurate degradation estimation; (3) Understanding reward, to ensure the accuracy of the generated content; and (4) Generation reward, where we propose using a visual expert model to evaluate the quality of generated images, encouraging the model to generate more realistic images. Extensive experiments demonstrate that our proposed RealSR-R1 can generate realistic details and accurately understand image content, particularly in semantically rich scenes or images with severe degradation.
>
---
#### [replaced 040] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17265v3](http://arxiv.org/pdf/2411.17265v3)**

> **作者:** Lehan He; Zeren Chen; Zhelun Shi; Tianyu Yu; Jing Shao; Lu Sheng
>
> **摘要:** The success of Direct Preference Optimization (DPO) in mitigating hallucinations in Vision Language Models (VLMs) critically hinges on the true reward gaps within preference pairs. However, current methods, typically relying on ranking or rewriting strategies, often struggle to optimize these reward gaps in a systematic way during data curation. A core difficulty lies in precisely characterizing and strategically manipulating the overall reward gap configuration, that is, the deliberate design of how to shape these reward gaps within each preference pair across the data. To address this, we introduce Topic-level Preference Rewriting(TPR), a novel framework designed for the systematic optimization of reward gap configuration. Through selectively replacing semantic topics within VLM responses with model's own resampled candidates for targeted rewriting, TPR can provide topic-level control over fine-grained semantic details. This precise control enables advanced data curation strategies, such as progressively adjusting the difficulty of rejected responses, thereby sculpting an effective reward gap configuration that guides the model to overcome challenging hallucinations. Comprehensive experiments demonstrate TPR achieves state-of-the-art performance on multiple hallucination benchmarks, outperforming previous methods by an average of 20%. Notably, it significantly reduces hallucinations by up to 93% on ObjectHal-Bench, and also exhibits superior data efficiency towards robust and cost-effective VLM alignment.
>
---
#### [replaced 041] HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10631v3](http://arxiv.org/pdf/2503.10631v3)**

> **作者:** Jiaming Liu; Hao Chen; Pengju An; Zhuoyang Liu; Renrui Zhang; Chenyang Gu; Xiaoqi Li; Ziyu Guo; Sixiang Chen; Mengzhen Liu; Chengkai Hou; Mengdi Zhao; KC alex Zhou; Pheng-Ann Heng; Shanghang Zhang
>
> **摘要:** A fundamental objective of manipulation policy design is to endow robots to comprehend human instructions, reason about scene cues, and execute generalized actions in dynamic environments. Recent autoregressive vision-language-action (VLA) methods inherit common-sense reasoning capabilities from vision-language models (VLMs) for next action-token prediction. However, these methods quantize actions into discrete bins, which disrupts the continuity required for precise control. In contrast, existing diffusion-based VLA methods incorporate an additional diffusion head to predict continuous actions solely conditioned on feature representations extracted by the VLM, without fully leveraging the VLM's pretrained reasoning capabilities through token-level generation. To address these limitations, we introduce HybridVLA, a unified framework that absorbs the continuous nature of diffusion-based actions and the contextual reasoning of autoregression within a single large language model. To mitigate interference between the two generation paradigms, we propose a collaborative training recipe that seamlessly incorporates diffusion denoising into the next-token prediction process. With this recipe, we find these two action prediction methods not only reinforce each other but also exhibit varying strength across different tasks. Therefore, we design a collaborative action ensemble mechanism that adaptively fuses both predictions, leading to more robust control. HybridVLA outperforms previous state-of-the-art VLA methods by 14\% and 19\% in mean success rate on simulation and real-world tasks, respectively, while demonstrating stable manipulation in unseen configurations.
>
---
#### [replaced 042] Cross-Camera Distracted Driver Classification through Feature Disentanglement and Contrastive Learning
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2411.13181v2](http://arxiv.org/pdf/2411.13181v2)**

> **作者:** Simone Bianco; Luigi Celona; Paolo Napoletano
>
> **摘要:** The classification of distracted drivers is pivotal for ensuring safe driving. Previous studies demonstrated the effectiveness of neural networks in automatically predicting driver distraction, fatigue, and potential hazards. However, recent research has uncovered a significant loss of accuracy in these models when applied to samples acquired under conditions that differ from the training data. In this paper, we introduce a robust model designed to withstand changes in camera position within the vehicle. Our Driver Behavior Monitoring Network (DBMNet) relies on a lightweight backbone and integrates a disentanglement module to discard camera view information from features, coupled with contrastive learning to enhance the encoding of various driver actions. Experiments conducted using a leave-one-camera-out protocol on the daytime and nighttime subsets of the 100-Driver dataset validate the effectiveness of our approach. Cross-dataset and cross-camera experiments conducted on three benchmark datasets, namely AUCDD-V1, EZZ2021 and SFD, demonstrate the superior generalization capabilities of the proposed method. Overall DBMNet achieves an improvement of 7% in Top-1 accuracy compared to existing approaches. Moreover, a quantized version of the DBMNet and all considered methods has been deployed on a Coral Dev Board board. In this deployment scenario, DBMNet outperforms alternatives, achieving the lowest average error while maintaining a compact model size, low memory footprint, fast inference time, and minimal power consumption.
>
---
#### [replaced 043] Model-Agnostic, Temperature-Informed Sampling Enhances Cross-Year Crop Mapping with Deep Learning
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.12885v2](http://arxiv.org/pdf/2506.12885v2)**

> **作者:** Mehmet Ozgur Turkoglu; Selene Ledain; Helge Aasen
>
> **备注:** under review
>
> **摘要:** Conventional benchmarks for crop type classification from optical satellite time series typically assume access to labeled data from the same year and rely on fixed calendar-day sampling. This limits generalization across seasons, where crop phenology shifts due to interannual climate variability, and precludes real-time application when current-year labels are unavailable. Furthermore, uncertainty quantification is often neglected, making such approaches unreliable for crop monitoring applications. Inspired by ecophysiological principles of plant growth, we propose a simple, model-agnostic sampling strategy that leverages growing degree days (GDD), based on daily average temperature, to replace calendar time with thermal time. By uniformly subsampling time series in this biologically meaningful domain, the method emphasizes phenologically active growth stages while reducing temporal redundancy and noise. We evaluate the method on a multi-year Sentinel-2 dataset spanning all of Switzerland, training on one growing season and testing on other seasons. Compared to state-of-the-art baselines, our method delivers substantial gains in classification accuracy and, critically, produces more calibrated uncertainty estimates. Notably, our method excels in low-data regimes and enables significantly more accurate early-season classification. With only 10 percent of the training data, our method surpasses the state-of-the-art baseline in both predictive accuracy and uncertainty estimation, and by the end of June, it achieves performance similar to a baseline trained on the full season. These results demonstrate that leveraging temperature data not only improves predictive performance across seasons but also enhances the robustness and trustworthiness of crop-type mapping in real-world applications.
>
---
#### [replaced 044] Novel Multicolumn Kernel Extreme Learning Machine for Food Detection via Optimal Features from CNN
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2205.07348v2](http://arxiv.org/pdf/2205.07348v2)**

> **作者:** Ghalib Ahmed Tahir; Chu Kiong Loo
>
> **摘要:** Automatic food detection is an emerging topic of interest due to its wide array of applications ranging from detecting food images on social media platforms to filtering non-food photos from the users in dietary assessment apps. Recently, during the COVID-19 pandemic, it has facilitated enforcing an eating ban by automatically detecting eating activities from cameras in public places. Therefore, to tackle the challenge of recognizing food images with high accuracy, we proposed the idea of a hybrid framework for extracting and selecting optimal features from an efficient neural network. There on, a nonlinear classifier is employed to discriminate between linearly inseparable feature vectors with great precision. In line with this idea, our method extracts features from MobileNetV3, selects an optimal subset of attributes by using Shapley Additive exPlanations (SHAP) values, and exploits kernel extreme learning machine (KELM) due to its nonlinear decision boundary and good generalization ability. However, KELM suffers from the 'curse of dimensionality problem' for large datasets due to the complex computation of kernel matrix with large numbers of hidden nodes. We solved this problem by proposing a novel multicolumn kernel extreme learning machine (MCKELM) which exploited the k-d tree algorithm to divide data into N subsets and trains separate KELM on each subset of data. Then, the method incorporates KELM classifiers into parallel structures and selects the top k nearest subsets during testing by using the k-d tree search for classifying input instead of the whole network. For evaluating a proposed framework large food/non-food dataset is prepared using nine publically available datasets. Experimental results showed the superiority of our method on an integrated set of measures while solving the problem of 'curse of dimensionality in KELM for large datasets.
>
---
#### [replaced 045] VR-FuseNet: A Fusion of Heterogeneous Fundus Data and Explainable Deep Network for Diabetic Retinopathy Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21464v3](http://arxiv.org/pdf/2504.21464v3)**

> **作者:** Shamim Rahim Refat; Ziyan Shirin Raha; Shuvashis Sarker; Faika Fairuj Preotee; MD. Musfikur Rahman; Tashreef Muhammad; Mohammad Shafiul Alam
>
> **备注:** 33 pages, 49 figures
>
> **摘要:** Diabetic retinopathy is a severe eye condition caused by diabetes where the retinal blood vessels get damaged and can lead to vision loss and blindness if not treated. Early and accurate detection is key to intervention and stopping the disease progressing. For addressing this disease properly, this paper presents a comprehensive approach for automated diabetic retinopathy detection by proposing a new hybrid deep learning model called VR-FuseNet. Diabetic retinopathy is a major eye disease and leading cause of blindness especially among diabetic patients so accurate and efficient automated detection methods are required. To address the limitations of existing methods including dataset imbalance, diversity and generalization issues this paper presents a hybrid dataset created from five publicly available diabetic retinopathy datasets. Essential preprocessing techniques such as SMOTE for class balancing and CLAHE for image enhancement are applied systematically to the dataset to improve the robustness and generalizability of the dataset. The proposed VR-FuseNet model combines the strengths of two state-of-the-art convolutional neural networks, VGG19 which captures fine-grained spatial features and ResNet50V2 which is known for its deep hierarchical feature extraction. This fusion improves the diagnostic performance and achieves an accuracy of 91.824%. The model outperforms individual architectures on all performance metrics demonstrating the effectiveness of hybrid feature extraction in Diabetic Retinopathy classification tasks. To make the proposed model more clinically useful and interpretable this paper incorporates multiple XAI techniques. These techniques generate visual explanations that clearly indicate the retinal features affecting the model's prediction such as microaneurysms, hemorrhages and exudates so that clinicians can interpret and validate.
>
---
#### [replaced 046] Leveraging Model Guidance to Extract Training Data from Personalized Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.03039v2](http://arxiv.org/pdf/2410.03039v2)**

> **作者:** Xiaoyu Wu; Jiaru Zhang; Zhiwei Steven Wu
>
> **备注:** Accepted at the International Conference on Machine Learning (ICML) 2025
>
> **摘要:** Diffusion Models (DMs) have become powerful image generation tools, especially for few-shot fine-tuning where a pretrained DM is fine-tuned on a small image set to capture specific styles or objects. Many people upload these personalized checkpoints online, fostering communities such as Civitai and HuggingFace. However, model owners may overlook the data leakage risks when releasing fine-tuned checkpoints. Moreover, concerns regarding copyright violations arise when unauthorized data is used during fine-tuning. In this paper, we ask: "Can training data be extracted from these fine-tuned DMs shared online?" A successful extraction would present not only data leakage threats but also offer tangible evidence of copyright infringement. To answer this, we propose FineXtract, a framework for extracting fine-tuning data. Our method approximates fine-tuning as a gradual shift in the model's learned distribution -- from the original pretrained DM toward the fine-tuning data. By extrapolating the models before and after fine-tuning, we guide the generation toward high-probability regions within the fine-tuned data distribution. We then apply a clustering algorithm to extract the most probable images from those generated using this extrapolated guidance. Experiments on DMs fine-tuned with datasets including WikiArt, DreamBooth, and real-world checkpoints posted online validate the effectiveness of our method, extracting about 20% of fine-tuning data in most cases. The code is available https://github.com/Nicholas0228/FineXtract.
>
---
#### [replaced 047] ReNeg: Learning Negative Embedding with Reward Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.19637v3](http://arxiv.org/pdf/2412.19637v3)**

> **作者:** Xiaomin Li; Yixuan Liu; Takashi Isobe; Xu Jia; Qinpeng Cui; Dong Zhou; Dong Li; You He; Huchuan Lu; Zhongdao Wang; Emad Barsoum
>
> **备注:** Code: https://github.com/AMD-AIG-AIMA/ReNeg
>
> **摘要:** In text-to-image (T2I) generation applications, negative embeddings have proven to be a simple yet effective approach for enhancing generation quality. Typically, these negative embeddings are derived from user-defined negative prompts, which, while being functional, are not necessarily optimal. In this paper, we introduce ReNeg, an end-to-end method designed to learn improved Negative embeddings guided by a Reward model. We employ a reward feedback learning framework and integrate classifier-free guidance (CFG) into the training process, which was previously utilized only during inference, thus enabling the effective learning of negative embeddings. We also propose two strategies for learning both global and per-sample negative embeddings. Extensive experiments show that the learned negative embedding significantly outperforms null-text and handcrafted counterparts, achieving substantial improvements in human preference alignment. Additionally, the negative embedding learned within the same text embedding space exhibits strong generalization capabilities. For example, using the same CLIP text encoder, the negative embedding learned on SD1.5 can be seamlessly transferred to text-to-image or even text-to-video models such as ControlNet, ZeroScope, and VideoCrafter2, resulting in consistent performance improvements across the board.
>
---
#### [replaced 048] Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03262v2](http://arxiv.org/pdf/2503.03262v2)**

> **作者:** Nadya Abdel Madjid; Abdulrahman Ahmad; Murad Mebrahtu; Yousef Babaa; Abdelmoamen Nasser; Sumbal Malik; Bilal Hassan; Naoufel Werghi; Jorge Dias; Majid Khonji
>
> **摘要:** As the potential for autonomous vehicles to be integrated on a large scale into modern traffic systems continues to grow, ensuring safe navigation in dynamic environments is crucial for smooth integration. To guarantee safety and prevent collisions, autonomous vehicles must be capable of accurately predicting the trajectories of surrounding traffic agents. Over the past decade, significant efforts from both academia and industry have been dedicated to designing solutions for precise trajectory forecasting. These efforts have produced a diverse range of approaches, raising questions about the differences between these methods and whether trajectory prediction challenges have been fully addressed. This paper reviews a substantial portion of recent trajectory prediction methods proposing a taxonomy to classify existing solutions. A general overview of the prediction pipeline is also provided, covering input and output modalities, modeling features, and prediction paradigms existing in the literature. In addition, the paper discusses active research areas within trajectory prediction, addresses the posed research questions, and highlights the remaining research gaps and challenges.
>
---
#### [replaced 049] Zero-Shot NAS via the Suppression of Local Entropy Decrease
- **分类: cs.LG; cs.CV; cs.NE**

- **链接: [http://arxiv.org/pdf/2411.06236v3](http://arxiv.org/pdf/2411.06236v3)**

> **作者:** Ning Wu; Han Huang; Yueting Xu; Zhifeng Hao
>
> **备注:** 8 pages, 2 figures. Corrected typos and latex template
>
> **摘要:** Architecture performance evaluation is the most time-consuming part of neural architecture search (NAS). Zero-Shot NAS accelerates the evaluation by utilizing zero-cost proxies instead of training. Though effective, existing zero-cost proxies require invoking backpropagations or running networks on input data, making it difficult to further accelerate the computation of proxies. To alleviate this issue, architecture topologies are used to evaluate the performance of networks in this study. We prove that particular architectural topologies decrease the local entropy of feature maps, which degrades specific features to a bias, thereby reducing network performance. Based on this proof, architectural topologies are utilized to quantify the suppression of local entropy decrease (SED) as a data-free and running-free proxy. Experimental results show that SED outperforms most state-of-the-art proxies in terms of architecture selection on five benchmarks, with computation time reduced by three orders of magnitude. We further compare the SED-based NAS with state-of-the-art proxies. SED-based NAS selects the architecture with higher accuracy and fewer parameters in only one second. The theoretical analyses of local entropy and experimental results demonstrate that the suppression of local entropy decrease facilitates selecting optimal architectures in Zero-Shot NAS.
>
---
#### [replaced 050] AnchorCrafter: Animate Cyber-Anchors Selling Your Products via Human-Object Interacting Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17383v2](http://arxiv.org/pdf/2411.17383v2)**

> **作者:** Ziyi Xu; Ziyao Huang; Juan Cao; Yong Zhang; Xiaodong Cun; Qing Shuai; Yuchen Wang; Linchao Bao; Jintao Li; Fan Tang
>
> **摘要:** The generation of anchor-style product promotion videos presents promising opportunities in e-commerce, advertising, and consumer engagement. Despite advancements in pose-guided human video generation, creating product promotion videos remains challenging. In addressing this challenge, we identify the integration of human-object interactions (HOI) into pose-guided human video generation as a core issue. To this end, we introduce AnchorCrafter, a novel diffusion-based system designed to generate 2D videos featuring a target human and a customized object, achieving high visual fidelity and controllable interactions. Specifically, we propose two key innovations: the HOI-appearance perception, which enhances object appearance recognition from arbitrary multi-view perspectives and disentangles object and human appearance, and the HOI-motion injection, which enables complex human-object interactions by overcoming challenges in object trajectory conditioning and inter-occlusion management. Extensive experiments show that our system improves object appearance preservation by 7.5\% and doubles the object localization accuracy compared to existing state-of-the-art approaches. It also outperforms existing approaches in maintaining human motion consistency and high-quality video generation. Project page including data, code, and Huggingface demo: https://github.com/cangcz/AnchorCrafter.
>
---
#### [replaced 051] Transformer-based RGB-T Tracking with Channel and Spatial Feature Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.03177v3](http://arxiv.org/pdf/2405.03177v3)**

> **作者:** Yunfeng Li; Bo Wang; Ye Li
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** The main problem in RGB-T tracking is the correct and optimal merging of the cross-modal features of visible and thermal images. Some previous methods either do not fully exploit the potential of RGB and TIR information for channel and spatial feature fusion or lack a direct interaction between the template and the search area, which limits the model's ability to fully utilize the original semantic information of both modalities. To address these limitations, we investigate how to achieve a direct fusion of cross-modal channels and spatial features in RGB-T tracking and propose CSTNet. It uses the Vision Transformer (ViT) as the backbone and adds a Joint Spatial and Channel Fusion Module (JSCFM) and Spatial Fusion Module (SFM) integrated between the transformer blocks to facilitate cross-modal feature interaction. The JSCFM module achieves joint modeling of channel and multi-level spatial features. The SFM module includes a cross-attention-like architecture for cross modeling and joint learning of RGB and TIR features. Comprehensive experiments show that CSTNet achieves state-of-the-art performance. To enhance practicality, we retrain the model without JSCFM and SFM modules and use CSNet as the pretraining weight, and propose CSTNet-small, which achieves 50% speedup with an average decrease of 1-2% in SR and PR performance. CSTNet and CSTNet-small achieve real-time speeds of 21 fps and 33 fps on the Nvidia Jetson Xavier, meeting actual deployment requirements. Code is available at https://github.com/LiYunfengLYF/CSTNet.
>
---
#### [replaced 052] Exploring Diffusion with Test-Time Training on Efficient Image Restoration
- **分类: cs.CV; I.4.9**

- **链接: [http://arxiv.org/pdf/2506.14541v2](http://arxiv.org/pdf/2506.14541v2)**

> **作者:** Rongchang Lu; Tianduo Luo; Yunzhi Jiang; Conghan Yue; Pei Yang; Guibao Liu; Changyang Gu
>
> **摘要:** Image restoration faces challenges including ineffective feature fusion, computational bottlenecks and inefficient diffusion processes. To address these, we propose DiffRWKVIR, a novel framework unifying Test-Time Training (TTT) with efficient diffusion. Our approach introduces three key innovations: (1) Omni-Scale 2D State Evolution extends RWKV's location-dependent parameterization to hierarchical multi-directional 2D scanning, enabling global contextual awareness with linear complexity O(L); (2) Chunk-Optimized Flash Processing accelerates intra-chunk parallelism by 3.2x via contiguous chunk processing (O(LCd) complexity), reducing sequential dependencies and computational overhead; (3) Prior-Guided Efficient Diffusion extracts a compact Image Prior Representation (IPR) in only 5-20 steps, proving 45% faster training/inference than DiffIR while solving computational inefficiency in denoising. Evaluated across super-resolution and inpainting benchmarks (Set5, Set14, BSD100, Urban100, Places365), DiffRWKVIR outperforms SwinIR, HAT, and MambaIR/v2 in PSNR, SSIM, LPIPS, and efficiency metrics. Our method establishes a new paradigm for adaptive, high-efficiency image restoration with optimized hardware utilization.
>
---
#### [replaced 053] Exploring the Potential of Encoder-free Architectures in 3D LMMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09620v3](http://arxiv.org/pdf/2502.09620v3)**

> **作者:** Yiwen Tang; Zoey Guo; Zhuhao Wang; Ray Zhang; Qizhi Chen; Junli Liu; Delin Qu; Zhigang Wang; Dong Wang; Xuelong Li; Bin Zhao
>
> **备注:** During the review process, we discovered that a portion of the test dataset used in our submission contained content that may have infringed upon the commercial copyrights of others. Due to the conflict regarding these commercial copyrights, we have unfortunately had to retract the submission
>
> **摘要:** Encoder-free architectures have been preliminarily explored in the 2D visual domain, yet it remains an open question whether they can be effectively applied to 3D understanding scenarios. In this paper, we present the first comprehensive investigation into the potential of encoder-free architectures to alleviate the challenges of encoder-based 3D Large Multimodal Models (LMMs). These challenges include the failure to adapt to varying point cloud resolutions and the point features from the encoder not meeting the semantic needs of Large Language Models (LLMs). We identify key aspects for 3D LMMs to remove the encoder and enable the LLM to assume the role of the 3D encoder: 1) We propose the LLM-embedded Semantic Encoding strategy in the pre-training stage, exploring the effects of various point cloud self-supervised losses. And we present the Hybrid Semantic Loss to extract high-level semantics. 2) We introduce the Hierarchical Geometry Aggregation strategy in the instruction tuning stage. This incorporates inductive bias into the LLM layers to focus on the local details of the point clouds. To the end, we present the first Encoder-free 3D LMM, ENEL. Our 7B model rivals the current state-of-the-art model, ShapeLLM-13B, achieving 55.10%, 50.98%, and 43.10% on the classification, captioning, and VQA tasks, respectively. Our results demonstrate that the encoder-free architecture is highly promising for replacing encoder-based architectures in the field of 3D understanding. The code is released at https://github.com/Ivan-Tang-3D/ENEL
>
---
#### [replaced 054] Step1X-Edit: A Practical Framework for General Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17761v4](http://arxiv.org/pdf/2504.17761v4)**

> **作者:** Shiyu Liu; Yucheng Han; Peng Xing; Fukun Yin; Rui Wang; Wei Cheng; Jiaqi Liao; Yingming Wang; Honghao Fu; Chunrui Han; Guopeng Li; Yuang Peng; Quan Sun; Jingwei Wu; Yan Cai; Zheng Ge; Ranchen Ming; Lei Xia; Xianfang Zeng; Yibo Zhu; Binxing Jiao; Xiangyu Zhang; Gang Yu; Daxin Jiang
>
> **备注:** code: https://github.com/stepfun-ai/Step1X-Edit
>
> **摘要:** In recent years, image editing models have witnessed remarkable and rapid development. The recent unveiling of cutting-edge multimodal models such as GPT-4o and Gemini2 Flash has introduced highly promising image editing capabilities. These models demonstrate an impressive aptitude for fulfilling a vast majority of user-driven editing requirements, marking a significant advancement in the field of image manipulation. However, there is still a large gap between the open-source algorithm with these closed-source models. Thus, in this paper, we aim to release a state-of-the-art image editing model, called Step1X-Edit, which can provide comparable performance against the closed-source models like GPT-4o and Gemini2 Flash. More specifically, we adopt the Multimodal LLM to process the reference image and the user's editing instruction. A latent embedding has been extracted and integrated with a diffusion image decoder to obtain the target image. To train the model, we build a data generation pipeline to produce a high-quality dataset. For evaluation, we develop the GEdit-Bench, a novel benchmark rooted in real-world user instructions. Experimental results on GEdit-Bench demonstrate that Step1X-Edit outperforms existing open-source baselines by a substantial margin and approaches the performance of leading proprietary models, thereby making significant contributions to the field of image editing.
>
---
#### [replaced 055] Multi-entity Video Transformers for Fine-Grained Video Representation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.10873v2](http://arxiv.org/pdf/2311.10873v2)**

> **作者:** Matthew Walmer; Rose Kanjirathinkal; Kai Sheng Tai; Keyur Muzumdar; Taipeng Tian; Abhinav Shrivastava
>
> **备注:** Published at the 12th Workshop on Fine-Grained Visual Categorization (CVPRW 2025)
>
> **摘要:** The area of temporally fine-grained video representation learning focuses on generating frame-by-frame representations for temporally dense tasks, such as fine-grained action phase classification and frame retrieval. In this work, we advance the state-of-the-art for self-supervised models in this area by re-examining the design of transformer architectures for video representation learning. A key aspect of our approach is the improved sharing of scene information in the temporal pipeline by representing multiple salient entities per frame. Prior works use late-fusion architectures that reduce frames to a single-dimensional vector before modeling any cross-frame dynamics. In contrast, our Multi-entity Video Transformer (MV-Former) processes the frames as groups of entities represented as tokens linked across time. To achieve this, we propose a Learnable Spatial Token Pooling strategy to identify and extract features for multiple salient regions per frame. Through our experiments, we show that MV-Former outperforms previous self-supervised methods, and also surpasses some prior works that use additional supervision or training data. When combined with additional pre-training data from Kinetics-400, MV-Former achieves a further performance boost. Overall, our MV-Former achieves state-of-the-art results on multiple fine-grained video benchmarks and shows that parsing video scenes as collections of entities can enhance performance in video tasks.
>
---
#### [replaced 056] S4Fusion: Saliency-aware Selective State Space Model for Infrared Visible Image Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.20881v3](http://arxiv.org/pdf/2405.20881v3)**

> **作者:** Haolong Ma; Hui Li; Chunyang Cheng; Gaoang Wang; Xiaoning Song; Xiaojun Wu
>
> **摘要:** As one of the tasks in Image Fusion, Infrared and Visible Image Fusion aims to integrate complementary information captured by sensors of different modalities into a single image. The Selective State Space Model (SSSM), known for its ability to capture long-range dependencies, has demonstrated its potential in the field of computer vision. However, in image fusion, current methods underestimate the potential of SSSM in capturing the global spatial information of both modalities. This limitation prevents the simultaneous consideration of the global spatial information from both modalities during interaction, leading to a lack of comprehensive perception of salient targets. Consequently, the fusion results tend to bias towards one modality instead of adaptively preserving salient targets. To address this issue, we propose the Saliency-aware Selective State Space Fusion Model (S4Fusion). In our S4Fusion, the designed Cross-Modal Spatial Awareness Module (CMSA) can simultaneously focus on global spatial information from both modalities while facilitating their interaction, thereby comprehensively capturing complementary information. Additionally, S4Fusion leverages a pre-trained network to perceive uncertainty in the fused images. By minimizing this uncertainty, S4Fusion adaptively highlights salient targets from both images. Extensive experiments demonstrate that our approach produces high-quality images and enhances performance in downstream tasks.
>
---
#### [replaced 057] LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13794v4](http://arxiv.org/pdf/2503.13794v4)**

> **作者:** Yang Zhou; Shiyu Zhao; Yuxiao Chen; Zhenting Wang; Can Jin; Dimitris N. Metaxas
>
> **摘要:** Large foundation models trained on large-scale vision-language data can boost Open-Vocabulary Object Detection (OVD) via synthetic training data, yet the hand-crafted pipelines often introduce bias and overfit to specific prompts. We sidestep this issue by directly fusing hidden states from Large Language Models (LLMs) into detectors-an avenue surprisingly under-explored. This paper presents a systematic method to enhance visual grounding by utilizing decoder layers of the LLM of an MLLM. We introduce a zero-initialized cross-attention adapter to enable efficient knowledge fusion from LLMs to object detectors, a new approach called LED (LLM Enhanced Open-Vocabulary Object Detection). We find that intermediate LLM layers already encode rich spatial semantics; adapting only the early layers yields most of the gain. With Swin-T as the vision encoder, Qwen2-0.5B + LED lifts GroundingDINO by 3.82 % on OmniLabel at just 8.7 % extra GFLOPs, and a larger vision backbone pushes the improvement to 6.22 %. Extensive ablations on adapter variants, LLM scales and fusion depths further corroborate our design.
>
---
#### [replaced 058] DiffDesign: Controllable Diffusion with Meta Prior for Efficient Interior Design Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.16301v3](http://arxiv.org/pdf/2411.16301v3)**

> **作者:** Yuxuan Yang; Tao Geng
>
> **摘要:** Interior design is a complex and creative discipline involving aesthetics, functionality, ergonomics, and materials science. Effective solutions must meet diverse requirements, typically producing multiple deliverables such as renderings and design drawings from various perspectives. Consequently, interior design processes are often inefficient and demand significant creativity. With advances in machine learning, generative models have emerged as a promising means of improving efficiency by creating designs from text descriptions or sketches. However, few generative works focus on interior design, leading to substantial discrepancies between outputs and practical needs, such as differences in size, spatial scope, and the lack of controllable generation quality. To address these challenges, we propose DiffDesign, a controllable diffusion model with meta priors for efficient interior design generation. Specifically, we utilize the generative priors of a 2D diffusion model pre-trained on a large image dataset as our rendering backbone. We further guide the denoising process by disentangling cross-attention control over design attributes, such as appearance, pose, and size, and introduce an optimal transfer-based alignment module to enforce view consistency. Simultaneously, we construct an interior design-specific dataset, DesignHelper, consisting of over 400 solutions across more than 15 spatial types and 15 design styles. This dataset helps fine-tune DiffDesign. Extensive experiments conducted on various benchmark datasets demonstrate the effectiveness and robustness of DiffDesign.
>
---
#### [replaced 059] ILIAS: Instance-Level Image retrieval At Scale
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11748v3](http://arxiv.org/pdf/2502.11748v3)**

> **作者:** Giorgos Kordopatis-Zilos; Vladan Stojnić; Anna Manko; Pavel Šuma; Nikolaos-Antonios Ypsilantis; Nikos Efthymiadis; Zakaria Laskar; Jiří Matas; Ondřej Chum; Giorgos Tolias
>
> **备注:** CVPR 2025
>
> **摘要:** This work introduces ILIAS, a new test dataset for Instance-Level Image retrieval At Scale. It is designed to evaluate the ability of current and future foundation models and retrieval techniques to recognize particular objects. The key benefits over existing datasets include large scale, domain diversity, accurate ground truth, and a performance that is far from saturated. ILIAS includes query and positive images for 1,000 object instances, manually collected to capture challenging conditions and diverse domains. Large-scale retrieval is conducted against 100 million distractor images from YFCC100M. To avoid false negatives without extra annotation effort, we include only query objects confirmed to have emerged after 2014, i.e. the compilation date of YFCC100M. An extensive benchmarking is performed with the following observations: i) models fine-tuned on specific domains, such as landmarks or products, excel in that domain but fail on ILIAS ii) learning a linear adaptation layer using multi-domain class supervision results in performance improvements, especially for vision-language models iii) local descriptors in retrieval re-ranking are still a key ingredient, especially in the presence of severe background clutter iv) the text-to-image performance of the vision-language foundation models is surprisingly close to the corresponding image-to-image case. website: https://vrg.fel.cvut.cz/ilias/
>
---
#### [replaced 060] EmoAgent: A Multi-Agent Framework for Diverse Affective Image Manipulation
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.11290v3](http://arxiv.org/pdf/2503.11290v3)**

> **作者:** Qi Mao; Haobo Hu; Yujie He; Difei Gao; Haokun Chen; Libiao Jin
>
> **摘要:** Affective Image Manipulation (AIM) aims to alter visual elements within an image to evoke specific emotional responses from viewers. However, existing AIM approaches rely on rigid \emph{one-to-one} mappings between emotions and visual cues, making them ill-suited for the inherently subjective and diverse ways in which humans perceive and express emotion.To address this, we introduce a novel task setting termed \emph{Diverse AIM (D-AIM)}, aiming to generate multiple visually distinct yet emotionally consistent image edits from a single source image and target emotion. We propose \emph{EmoAgent}, the first multi-agent framework tailored specifically for D-AIM. EmoAgent explicitly decomposes the manipulation process into three specialized phases executed by collaborative agents: a Planning Agent that generates diverse emotional editing strategies, an Editing Agent that precisely executes these strategies, and a Critic Agent that iteratively refines the results to ensure emotional accuracy. This collaborative design empowers EmoAgent to model \emph{one-to-many} emotion-to-visual mappings, enabling semantically diverse and emotionally faithful edits.Extensive quantitative and qualitative evaluations demonstrate that EmoAgent substantially outperforms state-of-the-art approaches in both emotional fidelity and semantic diversity, effectively generating multiple distinct visual edits that convey the same target emotion.
>
---
#### [replaced 061] HGO-YOLO: Advancing Anomaly Behavior Detection with Hierarchical Features and Lightweight Optimized Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07371v2](http://arxiv.org/pdf/2503.07371v2)**

> **作者:** Qizhi Zheng; Zhongze Luo; Meiyan Guo; Xinzhu Wang; Renqimuge Wu; Qiu Meng; Guanghui Dong
>
> **备注:** 12 pages
>
> **摘要:** Accurate, real-time object detection on resource-constrained hardware is critical for anomaly-behavior monitoring. We introduce HGO-YOLO, a lightweight detector that combines GhostHGNetv2 with an optimized parameter-sharing head (OptiConvDetect) to deliver an outstanding accuracy-efficiency trade-off. By embedding GhostConv into the HGNetv2 backbone with multi-scale residual fusion, the receptive field is enlarged while redundant computation is reduced by 50%. OptiConvDetect shares a partial-convolution layer for the classification and regression branches, cutting detection-head FLOPs by 41% without accuracy loss. On three anomaly datasets (fall, fight, smoke), HGO-YOLO attains 87.4% mAP@0.5 and 81.1% recall at 56 FPS on a single CPU with just 4.3 GFLOPs and 4.6 MB-surpassing YOLOv8n by +3.0% mAP, -51.7% FLOPs, and 1.7* speed. Real-world tests on a Jetson Orin Nano further confirm a stable throughput gain of 42 FPS.
>
---
#### [replaced 062] Holistic White-light Polyp Classification via Alignment-free Dense Distillation of Auxiliary Optical Chromoendoscopy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.19319v2](http://arxiv.org/pdf/2505.19319v2)**

> **作者:** Qiang Hu; Qimei Wang; Jia Chen; Xuantao Ji; Qiang Li; Zhiwei Wang
>
> **摘要:** White Light Imaging (WLI) and Narrow Band Imaging (NBI) are the two main colonoscopic modalities for polyp classification. While NBI, as optical chromoendoscopy, offers valuable vascular details, WLI remains the most common and often the only available modality in resource-limited settings. However, WLI-based methods typically underperform, limiting their clinical applicability. Existing approaches transfer knowledge from NBI to WLI through global feature alignment but often rely on cropped lesion regions, which are susceptible to detection errors and neglect contextual and subtle diagnostic cues. To address this, this paper proposes a novel holistic classification framework that leverages full-image diagnosis without requiring polyp localization. The key innovation lies in the Alignment-free Dense Distillation (ADD) module, which enables fine-grained cross-domain knowledge distillation regardless of misalignment between WLI and NBI images. Without resorting to explicit image alignment, ADD learns pixel-wise cross-domain affinities to establish correspondences between feature maps, guiding the distillation along the most relevant pixel connections. To further enhance distillation reliability, ADD incorporates Class Activation Mapping (CAM) to filter cross-domain affinities, ensuring the distillation path connects only those semantically consistent regions with equal contributions to polyp diagnosis. Extensive results on public and in-house datasets show that our method achieves state-of-the-art performance, relatively outperforming the other approaches by at least 2.5% and 16.2% in AUC, respectively. Code is available at: https://github.com/Huster-Hq/ADD.
>
---
#### [replaced 063] ZigzagPointMamba: Spatial-Semantic Mamba for Point Cloud Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21381v3](http://arxiv.org/pdf/2505.21381v3)**

> **作者:** Linshuang Diao; Dayong Ren; Sensen Song; Yurong Qian
>
> **备注:** The format of the document has an error and needs to be revised
>
> **摘要:** State Space models (SSMs) such as PointMamba enable efficient feature extraction for point cloud self-supervised learning with linear complexity, outperforming Transformers in computational efficiency. However, existing PointMamba-based methods depend on complex token ordering and random masking, which disrupt spatial continuity and local semantic correlations. We propose ZigzagPointMamba to tackle these challenges. The core of our approach is a simple zigzag scan path that globally sequences point cloud tokens, enhancing spatial continuity by preserving the proximity of spatially adjacent point tokens. Nevertheless, random masking undermines local semantic modeling in self-supervised learning. To address this, we introduce a Semantic-Siamese Masking Strategy (SMS), which masks semantically similar tokens to facilitate reconstruction by integrating local features of original and similar tokens. This overcomes the dependence on isolated local features and enables robust global semantic modeling. Our pre-trained ZigzagPointMamba weights significantly improve downstream tasks, achieving a 1.59% mIoU gain on ShapeNetPart for part segmentation, a 0.4% higher accuracy on ModelNet40 for classification, and 0.19%, 1.22%, and 0.72% higher accuracies respectively for the classification tasks on the OBJ-BG, OBJ-ONLY, and PB-T50-RS subsets of ScanObjectNN.
>
---
#### [replaced 064] InstructAttribute: Fine-grained Object Attributes editing with Instruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.00751v2](http://arxiv.org/pdf/2505.00751v2)**

> **作者:** Xingxi Yin; Jingfeng Zhang; Yue Deng; Zhi Li; Yicheng Li; Yin Zhang
>
> **摘要:** Text-to-image (T2I) diffusion models are widely used in image editing due to their powerful generative capabilities. However, achieving fine-grained control over specific object attributes, such as color and material, remains a considerable challenge. Existing methods often fail to accurately modify these attributes or compromise structural integrity and overall image consistency. To fill this gap, we introduce Structure Preservation and Attribute Amplification (SPAA), a novel training-free framework that enables precise generation of color and material attributes for the same object by intelligently manipulating self-attention maps and cross-attention values within diffusion models. Building on SPAA, we integrate multi-modal large language models (MLLMs) to automate data curation and instruction generation. Leveraging this object attribute data collection engine, we construct the Attribute Dataset, encompassing a comprehensive range of colors and materials across diverse object categories. Using this generated dataset, we propose InstructAttribute, an instruction-tuned model that enables fine-grained and object-level attribute editing through natural language prompts. This capability holds significant practical implications for diverse fields, from accelerating product design and e-commerce visualization to enhancing virtual try-on experiences. Extensive experiments demonstrate that InstructAttribute outperforms existing instruction-based baselines, achieving a superior balance between attribute modification accuracy and structural preservation.
>
---
#### [replaced 065] Kimi-VL Technical Report
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07491v3](http://arxiv.org/pdf/2504.07491v3)**

> **作者:** Kimi Team; Angang Du; Bohong Yin; Bowei Xing; Bowen Qu; Bowen Wang; Cheng Chen; Chenlin Zhang; Chenzhuang Du; Chu Wei; Congcong Wang; Dehao Zhang; Dikang Du; Dongliang Wang; Enming Yuan; Enzhe Lu; Fang Li; Flood Sung; Guangda Wei; Guokun Lai; Han Zhu; Hao Ding; Hao Hu; Hao Yang; Hao Zhang; Haoning Wu; Haotian Yao; Haoyu Lu; Heng Wang; Hongcheng Gao; Huabin Zheng; Jiaming Li; Jianlin Su; Jianzhou Wang; Jiaqi Deng; Jiezhong Qiu; Jin Xie; Jinhong Wang; Jingyuan Liu; Junjie Yan; Kun Ouyang; Liang Chen; Lin Sui; Longhui Yu; Mengfan Dong; Mengnan Dong; Nuo Xu; Pengyu Cheng; Qizheng Gu; Runjie Zhou; Shaowei Liu; Sihan Cao; Tao Yu; Tianhui Song; Tongtong Bai; Wei Song; Weiran He; Weixiao Huang; Weixin Xu; Xiaokun Yuan; Xingcheng Yao; Xingzhe Wu; Xinhao Li; Xinxing Zu; Xinyu Zhou; Xinyuan Wang; Y. Charles; Yan Zhong; Yang Li; Yangyang Hu; Yanru Chen; Yejie Wang; Yibo Liu; Yibo Miao; Yidao Qin; Yimin Chen; Yiping Bao; Yiqin Wang; Yongsheng Kang; Yuanxin Liu; Yuhao Dong; Yulun Du; Yuxin Wu; Yuzhi Wang; Yuzi Yan; Zaida Zhou; Zhaowei Li; Zhejun Jiang; Zheng Zhang; Zhilin Yang; Zhiqi Huang; Zihao Huang; Zijia Zhao; Ziwei Chen; Zongyu Lin
>
> **备注:** Updated Kimi-VL-A3B-Thinking-2506 information
>
> **摘要:** We present Kimi-VL, an efficient open-source Mixture-of-Experts (MoE) vision-language model (VLM) that offers advanced multimodal reasoning, long-context understanding, and strong agent capabilities - all while activating only 2.8B parameters in its language decoder (Kimi-VL-A3B). Kimi-VL demonstrates strong performance across challenging domains: as a general-purpose VLM, Kimi-VL excels in multi-turn agent tasks (e.g., OSWorld), matching flagship models. Furthermore, it exhibits remarkable capabilities across diverse challenging vision language tasks, including college-level image and video comprehension, OCR, mathematical reasoning, and multi-image understanding. In comparative evaluations, it effectively competes with cutting-edge efficient VLMs such as GPT-4o-mini, Qwen2.5-VL-7B, and Gemma-3-12B-IT, while surpassing GPT-4o in several key domains. Kimi-VL also advances in processing long contexts and perceiving clearly. With a 128K extended context window, Kimi-VL can process diverse long inputs, achieving impressive scores of 64.5 on LongVideoBench and 35.1 on MMLongBench-Doc. Its native-resolution vision encoder, MoonViT, further allows it to see and understand ultra-high-resolution visual inputs, achieving 83.2 on InfoVQA and 34.5 on ScreenSpot-Pro, while maintaining lower computational cost for common tasks. Building upon Kimi-VL, we introduce an advanced long-thinking variant: Kimi-VL-Thinking-2506. Developed through long chain-of-thought (CoT) supervised fine-tuning (SFT) and reinforcement learning (RL), the latest model exhibits strong long-horizon reasoning capabilities (64.0 on MMMU, 46.3 on MMMU-Pro, 56.9 on MathVision, 80.1 on MathVista, 65.2 on VideoMMMU) while obtaining robust general abilities. Code and models are publicly accessible at https://github.com/MoonshotAI/Kimi-VL.
>
---
#### [replaced 066] Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is Secretly a GAN Discriminator
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.01103v3](http://arxiv.org/pdf/2503.01103v3)**

> **作者:** Kaiwen Zheng; Yongxin Chen; Huayu Chen; Guande He; Ming-Yu Liu; Jun Zhu; Qinsheng Zhang
>
> **备注:** ICML 2025 Spotlight Project Page: https://research.nvidia.com/labs/dir/ddo/ Code: https://github.com/NVlabs/DDO
>
> **摘要:** While likelihood-based generative models, particularly diffusion and autoregressive models, have achieved remarkable fidelity in visual generation, the maximum likelihood estimation (MLE) objective, which minimizes the forward KL divergence, inherently suffers from a mode-covering tendency that limits the generation quality under limited model capacity. In this work, we propose Direct Discriminative Optimization (DDO) as a unified framework that integrates likelihood-based generative training and GAN-type discrimination to bypass this fundamental constraint by exploiting reverse KL and self-generated negative signals. Our key insight is to parameterize a discriminator implicitly using the likelihood ratio between a learnable target model and a fixed reference model, drawing parallels with the philosophy of Direct Preference Optimization (DPO). Unlike GANs, this parameterization eliminates the need for joint training of generator and discriminator networks, allowing for direct, efficient, and effective finetuning of a well-trained model to its full potential beyond the limits of MLE. DDO can be performed iteratively in a self-play manner for progressive model refinement, with each round requiring less than 1% of pretraining epochs. Our experiments demonstrate the effectiveness of DDO by significantly advancing the previous SOTA diffusion model EDM, reducing FID scores from 1.79/1.58/1.96 to new records of 1.30/0.97/1.26 on CIFAR-10/ImageNet-64/ImageNet 512x512 datasets without any guidance mechanisms, and by consistently improving both guidance-free and CFG-enhanced FIDs of visual autoregressive models on ImageNet 256x256.
>
---
#### [replaced 067] Interpretation of Deep Learning Model in Embryo Selection for In Vitro Fertilization (IVF) Treatment
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06680v2](http://arxiv.org/pdf/2506.06680v2)**

> **作者:** Radha Kodali; Venkata Rao Dhulipalla; Venkata Siva Kishor Tatavarty; Madhavi Nadakuditi; Bharadwaj Thiruveedhula; Suryanarayana Gunnam; Durga Prasad Bavirisetti
>
> **摘要:** Infertility has a considerable impact on individuals' quality of life, affecting them socially and psychologically, with projections indicating a rise in the upcoming years. In vitro fertilization (IVF) emerges as one of the primary techniques within economically developed nations, employed to address the rising problem of low fertility. Expert embryologists conventionally grade embryos by reviewing blastocyst images to select the most optimal for transfer, yet this process is time-consuming and lacks efficiency. Blastocyst images provide a valuable resource for assessing embryo viability. In this study, we introduce an explainable artificial intelligence (XAI) framework for classifying embryos, employing a fusion of convolutional neural network (CNN) and long short-term memory (LSTM) architecture, referred to as CNN-LSTM. Utilizing deep learning, our model achieves high accuracy in embryo classification while maintaining interpretability through XAI.
>
---
#### [replaced 068] Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning
- **分类: cs.LG; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01837v2](http://arxiv.org/pdf/2503.01837v2)**

> **作者:** Adrià López Escoriza; Nicklas Hansen; Stone Tao; Tongzhou Mu; Hao Su
>
> **备注:** Project page can be found at https://adrialopezescoriza.github.io/demo3/
>
> **摘要:** Long-horizon tasks in robotic manipulation present significant challenges in reinforcement learning (RL) due to the difficulty of designing dense reward functions and effectively exploring the expansive state-action space. However, despite a lack of dense rewards, these tasks often have a multi-stage structure, which can be leveraged to decompose the overall objective into manageable subgoals. In this work, we propose DEMO3, a framework that exploits this structure for efficient learning from visual inputs. Specifically, our approach incorporates multi-stage dense reward learning, a bi-phasic training scheme, and world model learning into a carefully designed demonstration-augmented RL framework that strongly mitigates the challenge of exploration in long-horizon tasks. Our evaluations demonstrate that our method improves data-efficiency by an average of 40% and by 70% on particularly difficult tasks compared to state-of-the-art approaches. We validate this across 16 sparse-reward tasks spanning four domains, including challenging humanoid visual control tasks using as few as five demonstrations.
>
---
#### [replaced 069] UniDrive: Towards Universal Driving Perception Across Camera Configurations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.13864v2](http://arxiv.org/pdf/2410.13864v2)**

> **作者:** Ye Li; Wenzhao Zheng; Xiaonan Huang; Kurt Keutzer
>
> **备注:** ICLR 2025; 15 pages, 7 figures, 2 tables; Code at https://github.com/ywyeli/UniDrive
>
> **摘要:** Vision-centric autonomous driving has demonstrated excellent performance with economical sensors. As the fundamental step, 3D perception aims to infer 3D information from 2D images based on 3D-2D projection. This makes driving perception models susceptible to sensor configuration (e.g., camera intrinsics and extrinsics) variations. However, generalizing across camera configurations is important for deploying autonomous driving models on different car models. In this paper, we present UniDrive, a novel framework for vision-centric autonomous driving to achieve universal perception across camera configurations. We deploy a set of unified virtual cameras and propose a ground-aware projection method to effectively transform the original images into these unified virtual views. We further propose a virtual configuration optimization method by minimizing the expected projection error between original and virtual cameras. The proposed virtual camera projection can be applied to existing 3D perception methods as a plug-and-play module to mitigate the challenges posed by camera parameter variability, resulting in more adaptable and reliable driving perception models. To evaluate the effectiveness of our framework, we collect a dataset on CARLA by driving the same routes while only modifying the camera configurations. Experimental results demonstrate that our method trained on one specific camera configuration can generalize to varying configurations with minor performance degradation.
>
---
#### [replaced 070] Towards Reflected Object Detection: A Benchmark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.05575v2](http://arxiv.org/pdf/2407.05575v2)**

> **作者:** Yiquan Wu; Zhongtian Wang; You Wu; Ling Huang; Hui Zhou; Shuiwang Li
>
> **摘要:** Object detection has greatly improved over the past decade thanks to advances in deep learning and large-scale datasets. However, detecting objects reflected in surfaces remains an underexplored area. Reflective surfaces are ubiquitous in daily life, appearing in homes, offices, public spaces, and natural environments. Accurate detection and interpretation of reflected objects are essential for various applications. This paper addresses this gap by introducing a extensive benchmark specifically designed for Reflected Object Detection. Our Reflected Object Detection Dataset (RODD) features a diverse collection of images showcasing reflected objects in various contexts, providing standard annotations for both real and reflected objects. This distinguishes it from traditional object detection benchmarks. RODD encompasses 10 categories and includes 21,059 images of real and reflected objects across different backgrounds, complete with standard bounding box annotations and the classification of objects as real or reflected. Additionally, we present baseline results by adapting five state-of-the-art object detection models to address this challenging task. Experimental results underscore the limitations of existing methods when applied to reflected object detection, highlighting the need for specialized approaches. By releasing RODD, we aim to support and advance future research on detecting reflected objects. Dataset and code are available at: https://github.com/jirouvan/ROD.
>
---
#### [replaced 071] One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.22366v4](http://arxiv.org/pdf/2410.22366v4)**

> **作者:** Viacheslav Surkov; Chris Wendler; Antonio Mari; Mikhail Terekhov; Justin Deschenaux; Robert West; Caglar Gulcehre; David Bau
>
> **摘要:** For large language models (LLMs), sparse autoencoders (SAEs) have been shown to decompose intermediate representations that often are not interpretable directly into sparse sums of interpretable features, facilitating better control and subsequent analysis. However, similar analyses and approaches have been lacking for text-to-image models. We investigate the possibility of using SAEs to learn interpretable features for SDXL Turbo, a few-step text-to-image diffusion model. To this end, we train SAEs on the updates performed by transformer blocks within SDXL Turbo's denoising U-net in its 1-step setting. Interestingly, we find that they generalize to 4-step SDXL Turbo and even to the multi-step SDXL base model (i.e., a different model) without additional training. In addition, we show that their learned features are interpretable, causally influence the generation process, and reveal specialization among the blocks. We do so by creating RIEBench, a representation-based image editing benchmark, for editing images while they are generated by turning on and off individual SAE features. This allows us to track which transformer blocks' features are the most impactful depending on the edit category. Our work is the first investigation of SAEs for interpretability in text-to-image diffusion models and our results establish SAEs as a promising approach for understanding and manipulating the internal mechanisms of text-to-image models.
>
---
#### [replaced 072] Cross from Left to Right Brain: Adaptive Text Dreamer for Vision-and-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20897v2](http://arxiv.org/pdf/2505.20897v2)**

> **作者:** Pingrui Zhang; Yifei Su; Pengyuan Wu; Dong An; Li Zhang; Zhigang Wang; Dong Wang; Yan Ding; Bin Zhao; Xuelong Li
>
> **摘要:** Vision-and-Language Navigation (VLN) requires the agent to navigate by following natural instructions under partial observability, making it difficult to align perception with language. Recent methods mitigate this by imagining future scenes, yet they rely on vision-based synthesis, leading to high computational cost and redundant details. To this end, we propose to adaptively imagine key environmental semantics via \textit{language} form, enabling a more reliable and efficient strategy. Specifically, we introduce a novel Adaptive Text Dreamer (ATD), a dual-branch self-guided imagination policy built upon a large language model (LLM). ATD is designed with a human-like left-right brain architecture, where the left brain focuses on logical integration, and the right brain is responsible for imaginative prediction of future scenes. To achieve this, we fine-tune only the Q-former within both brains to efficiently activate domain-specific knowledge in the LLM, enabling dynamic updates of logical reasoning and imagination during navigation. Furthermore, we introduce a cross-interaction mechanism to regularize the imagined outputs and inject them into a navigation expert module, allowing ATD to jointly exploit both the reasoning capacity of the LLM and the expertise of the navigation model. We conduct extensive experiments on the R2R benchmark, where ATD achieves state-of-the-art performance with fewer parameters. The code is \href{https://github.com/zhangpingrui/Adaptive-Text-Dreamer}{here}.
>
---
#### [replaced 073] FullLoRA: Efficiently Boosting the Robustness of Pretrained Vision Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.01752v2](http://arxiv.org/pdf/2401.01752v2)**

> **作者:** Zheng Yuan; Jie Zhang; Shiguang Shan; Xilin Chen
>
> **备注:** Accepted by IEEE Transactions on Image Processing (TIP). 11 pages, 3 figures, 8 tables
>
> **摘要:** In recent years, the Vision Transformer (ViT) model has gradually become mainstream in various computer vision tasks, and the robustness of the model has received increasing attention. However, existing large models tend to prioritize performance during training, potentially neglecting the robustness, which may lead to serious security concerns. In this paper, we establish a new challenge: exploring how to use a small number of additional parameters for adversarial finetuning to quickly and effectively enhance the adversarial robustness of a standardly trained model. To address this challenge, we develop novel LNLoRA module, incorporating a learnable layer normalization before the conventional LoRA module, which helps mitigate magnitude differences in parameters between the adversarial and standard training paradigms. Furthermore, we propose the FullLoRA framework by integrating the learnable LNLoRA modules into all key components of ViT-based models while keeping the pretrained model frozen, which can significantly improve the model robustness via adversarial finetuning in a parameter-efficient manner. Extensive experiments on several datasets demonstrate the superiority of our proposed FullLoRA framework. It achieves comparable robustness with full finetuning while only requiring about 5\% of the learnable parameters. This also effectively addresses concerns regarding extra model storage space and enormous training time caused by adversarial finetuning.
>
---
#### [replaced 074] STAGE: A Stream-Centric Generative World Model for Long-Horizon Driving-Scene Simulation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13138v2](http://arxiv.org/pdf/2506.13138v2)**

> **作者:** Jiamin Wang; Yichen Yao; Xiang Feng; Hang Wu; Yaming Wang; Qingqiu Huang; Yuexin Ma; Xinge Zhu
>
> **摘要:** The generation of temporally consistent, high-fidelity driving videos over extended horizons presents a fundamental challenge in autonomous driving world modeling. Existing approaches often suffer from error accumulation and feature misalignment due to inadequate decoupling of spatio-temporal dynamics and limited cross-frame feature propagation mechanisms. To address these limitations, we present STAGE (Streaming Temporal Attention Generative Engine), a novel auto-regressive framework that pioneers hierarchical feature coordination and multi-phase optimization for sustainable video synthesis. To achieve high-quality long-horizon driving video generation, we introduce Hierarchical Temporal Feature Transfer (HTFT) and a novel multi-stage training strategy. HTFT enhances temporal consistency between video frames throughout the video generation process by modeling the temporal and denoising process separately and transferring denoising features between frames. The multi-stage training strategy is to divide the training into three stages, through model decoupling and auto-regressive inference process simulation, thereby accelerating model convergence and reducing error accumulation. Experiments on the Nuscenes dataset show that STAGE has significantly surpassed existing methods in the long-horizon driving video generation task. In addition, we also explored STAGE's ability to generate unlimited-length driving videos. We generated 600 frames of high-quality driving videos on the Nuscenes dataset, which far exceeds the maximum length achievable by existing methods.
>
---
#### [replaced 075] A Comparative Analysis of Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) as Dimensionality Reduction Techniques
- **分类: cs.CV; cs.NA; math.NA**

- **链接: [http://arxiv.org/pdf/2506.16663v2](http://arxiv.org/pdf/2506.16663v2)**

> **作者:** Michael Gyimadu; Gregory Bell; Ph. D
>
> **摘要:** High-dimensional image data often require dimensionality reduction before further analysis. This paper provides a purely analytical comparison of two linear techniques-Principal Component Analysis (PCA) and Singular Value Decomposition (SVD). After the derivation of each algorithm from first principles, we assess their interpretability, numerical stability, and suitability for differing matrix shapes. building on classical and recent numerical literature, We synthesize rule-of-thumb guidelines for choosing one out of the two algorithms without empirical benchmarking, building on classical and recent numerical literature. Limitations and directions for future experimental work are outlined at the end.
>
---
#### [replaced 076] FRAMES-VQA: Benchmarking Fine-Tuning Robustness across Multi-Modal Shifts in Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21755v2](http://arxiv.org/pdf/2505.21755v2)**

> **作者:** Chengyue Huang; Brisa Maneechotesuwan; Shivang Chopra; Zsolt Kira
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Visual question answering (VQA) systems face significant challenges when adapting to real-world data shifts, especially in multi-modal contexts. While robust fine-tuning strategies are essential for maintaining performance across in-distribution (ID) and out-of-distribution (OOD) scenarios, current evaluation settings are primarily unimodal or particular to some types of OOD, offering limited insight into the complexities of multi-modal contexts. In this work, we propose a new benchmark FRAMES-VQA (Fine-Tuning Robustness across Multi-Modal Shifts in VQA) for evaluating robust fine-tuning for VQA tasks. We utilize ten existing VQA benchmarks, including VQAv2, IV-VQA, VQA-CP, OK-VQA and others, and categorize them into ID, near and far OOD datasets covering uni-modal, multi-modal and adversarial distribution shifts. We first conduct a comprehensive comparison of existing robust fine-tuning methods. We then quantify the distribution shifts by calculating the Mahalanobis distance using uni-modal and multi-modal embeddings extracted from various models. Further, we perform an extensive analysis to explore the interactions between uni- and multi-modal shifts as well as modality importance for ID and OOD samples. These analyses offer valuable guidance on developing more robust fine-tuning methods to handle multi-modal distribution shifts. The code is available at https://github.com/chengyuehuang511/FRAMES-VQA .
>
---
#### [replaced 077] Image Captions are Natural Prompts for Text-to-Image Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2307.08526v2](http://arxiv.org/pdf/2307.08526v2)**

> **作者:** Shiye Lei; Hao Chen; Sen Zhang; Bo Zhao; Dacheng Tao
>
> **备注:** 31 pages, 2 figure, 15 tables. Codes are available at https://github.com/LeavesLei/Caption_in_Prompt
>
> **摘要:** With the rapid development of Artificial Intelligence Generated Content (AIGC), it has become a common practice to train models on synthetic data due to data-scarcity and privacy leakage problems. Owing to massive and diverse information conveyed in real images, it is challenging for text-to-image generative models to synthesize informative training data with hand-crafted prompts. Considering the impressive ability of large generative models, could such models directly synthesize good training images for prediction tasks with proper prompts? We offer an affirmative response to this question by proposing a simple yet effective method, validated through ImageNet classification. Specifically, we caption each real image with the advanced captioning model to obtain informative and faithful prompts that extract class-relevant information and clarify the polysemy of class names. The image captions and class names are concatenated to prompt generative models for training image synthesis. We show that this simple caption incorporation significantly boosts the informativeness of synthetic data therefore enhancing downstream model generalization. More importantly, besides improvements in data augmentation and privacy preservation, our experiments demonstrate that synthesized images can exceed real data in terms of out-of-distribution robustness.
>
---
#### [replaced 078] Benchmarking Large Language Models for Handwritten Text Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.15195v3](http://arxiv.org/pdf/2503.15195v3)**

> **作者:** Giorgia Crosilla; Lukas Klic; Giovanni Colavizza
>
> **摘要:** Traditional machine learning models for Handwritten Text Recognition (HTR) rely on supervised training, requiring extensive manual annotations, and often produce errors due to the separation between layout and text processing. In contrast, Multimodal Large Language Models (MLLMs) offer a general approach to recognizing diverse handwriting styles without the need for model-specific training. The study benchmarks various proprietary and open-source LLMs against Transkribus models, evaluating their performance on both modern and historical datasets written in English, French, German, and Italian. In addition, emphasis is placed on testing the models' ability to autonomously correct previously generated outputs. Findings indicate that proprietary models, especially Claude 3.5 Sonnet, outperform open-source alternatives in zero-shot settings. MLLMs achieve excellent results in recognizing modern handwriting and exhibit a preference for the English language due to their pre-training dataset composition. Comparisons with Transkribus show no consistent advantage for either approach. Moreover, LLMs demonstrate limited ability to autonomously correct errors in zero-shot transcriptions.
>
---
#### [replaced 079] CGS-GAN: 3D Consistent Gaussian Splatting GANs for High Resolution Human Head Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17590v2](http://arxiv.org/pdf/2505.17590v2)**

> **作者:** Florian Barthel; Wieland Morgenstern; Paul Hinzer; Anna Hilsmann; Peter Eisert
>
> **备注:** Main paper 12 pages, supplementary materials 8 pages
>
> **摘要:** Recently, 3D GANs based on 3D Gaussian splatting have been proposed for high quality synthesis of human heads. However, existing methods stabilize training and enhance rendering quality from steep viewpoints by conditioning the random latent vector on the current camera position. This compromises 3D consistency, as we observe significant identity changes when re-synthesizing the 3D head with each camera shift. Conversely, fixing the camera to a single viewpoint yields high-quality renderings for that perspective but results in poor performance for novel views. Removing view-conditioning typically destabilizes GAN training, often causing the training to collapse. In response to these challenges, we introduce CGS-GAN, a novel 3D Gaussian Splatting GAN framework that enables stable training and high-quality 3D-consistent synthesis of human heads without relying on view-conditioning. To ensure training stability, we introduce a multi-view regularization technique that enhances generator convergence with minimal computational overhead. Additionally, we adapt the conditional loss used in existing 3D Gaussian splatting GANs and propose a generator architecture designed to not only stabilize training but also facilitate efficient rendering and straightforward scaling, enabling output resolutions up to $2048^2$. To evaluate the capabilities of CGS-GAN, we curate a new dataset derived from FFHQ. This dataset enables very high resolutions, focuses on larger portions of the human head, reduces view-dependent artifacts for improved 3D consistency, and excludes images where subjects are obscured by hands or other objects. As a result, our approach achieves very high rendering quality, supported by competitive FID scores, while ensuring consistent 3D scene generation. Check our our project page here: https://fraunhoferhhi.github.io/cgs-gan/
>
---
#### [replaced 080] Harmony: A Joint Self-Supervised and Weakly-Supervised Framework for Learning General Purpose Visual Representations
- **分类: cs.LG; cs.CV; 68T07, 68T45; I.2.10**

- **链接: [http://arxiv.org/pdf/2405.14239v3](http://arxiv.org/pdf/2405.14239v3)**

> **作者:** Mohammed Baharoon; Jonathan Klein; Dominik L. Michels
>
> **备注:** 27 pages
>
> **摘要:** Vision-language contrastive learning frameworks such as CLIP enable learning representations from natural language supervision and provide strong zero-shot classification capabilities. However, due to the nature of the supervisory signal in these paradigms, they lack the ability to learn localized features, leading to degraded performance on dense prediction tasks such as segmentation and detection. On the other hand, self-supervised learning methods have shown the ability to learn granular representations, complementing the high-level features in vision-language training. In this work, we present Harmony, a framework that combines vision-language training with discriminative and generative self-supervision to learn visual features that can be generalized across different downstream vision tasks. Our framework is specifically designed to work on web-scraped data by not relying on negative examples in the self-supervised learning path and addressing the one-to-one correspondence issue using soft CLIP targets generated by an EMA model. Moreover, Harmony optimizes for five different objectives simultaneously, efficiently utilizing the supervision in each data example, making it even more suited in data-constrained settings. We comprehensively evaluate Harmony across various vision downstream tasks and find that it significantly outperforms the baseline CLIP and outperforms the previously leading joint self- and weakly supervised methods, SLIP, MaskCLIP, and DetailCLIP.
>
---
#### [replaced 081] PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views
- **分类: cs.CV; I.2.10; I.4.8; I.5.1**

- **链接: [http://arxiv.org/pdf/2505.23481v2](http://arxiv.org/pdf/2505.23481v2)**

> **作者:** Mohamed Rayan Barhdadi; Hasan Kurban; Hussein Alnuweiri
>
> **备注:** 4 pages, 2 figures, 2 tables. Appearing in Building Physically Plausible World Models at the 42nd International Conference on Machine Learning (ICML 2025), Vancouver, Canada
>
> **摘要:** PhysicsNeRF is a physically grounded framework for 3D reconstruction from sparse views, extending Neural Radiance Fields with four complementary constraints: depth ranking, RegNeRF-style consistency, sparsity priors, and cross-view alignment. While standard NeRFs fail under sparse supervision, PhysicsNeRF employs a compact 0.67M-parameter architecture and achieves 21.4 dB average PSNR using only 8 views, outperforming prior methods. A generalization gap of 5.7-6.2 dB is consistently observed and analyzed, revealing fundamental limitations of sparse-view reconstruction. PhysicsNeRF enables physically consistent, generalizable 3D representations for agent interaction and simulation, and clarifies the expressiveness-generalization trade-off in constrained NeRF models.
>
---
#### [replaced 082] MDAA-Diff: CT-Guided Multi-Dose Adaptive Attention Diffusion Model for PET Denoising
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05112v2](http://arxiv.org/pdf/2505.05112v2)**

> **作者:** Xiaolong Niu; Zanting Ye; Xu Han; Yanchao Huang; Hao Sun; Hubing Wu; Lijun Lu
>
> **摘要:** Acquiring high-quality Positron Emission Tomography (PET) images requires administering high-dose radiotracers, which increases radiation exposure risks. Generating standard-dose PET (SPET) from low-dose PET (LPET) has become a potential solution. However, previous studies have primarily focused on single low-dose PET denoising, neglecting two critical factors: discrepancies in dose response caused by inter-patient variability, and complementary anatomical constraints derived from CT images. In this work, we propose a novel CT-Guided Multi-dose Adaptive Attention Denoising Diffusion Model (MDAA-Diff) for multi-dose PET denoising. Our approach integrates anatomical guidance and dose-level adaptation to achieve superior denoising performance under low-dose conditions. Specifically, this approach incorporates a CT-Guided High-frequency Wavelet Attention (HWA) module, which uses wavelet transforms to separate high-frequency anatomical boundary features from CT images. These extracted features are then incorporated into PET imaging through an adaptive weighted fusion mechanism to enhance edge details. Additionally, we propose the Dose-Adaptive Attention (DAA) module, a dose-conditioned enhancement mechanism that dynamically integrates dose levels into channel-spatial attention weight calculation. Extensive experiments on 18F-FDG and 68Ga-FAPI datasets demonstrate that MDAA-Diff outperforms state-of-the-art approaches in preserving diagnostic quality under reduced-dose conditions. Our code is publicly available.
>
---
#### [replaced 083] Directional Gradient Projection for Robust Fine-Tuning of Foundation Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15895v2](http://arxiv.org/pdf/2502.15895v2)**

> **作者:** Chengyue Huang; Junjiao Tian; Brisa Maneechotesuwan; Shivang Chopra; Zsolt Kira
>
> **备注:** Accepted to ICLR 2025
>
> **摘要:** Robust fine-tuning aims to adapt large foundation models to downstream tasks while preserving their robustness to distribution shifts. Existing methods primarily focus on constraining and projecting current model towards the pre-trained initialization based on the magnitudes between fine-tuned and pre-trained weights, which often require extensive hyper-parameter tuning and can sometimes result in underfitting. In this work, we propose Directional Gradient Projection (DiGraP), a novel layer-wise trainable method that incorporates directional information from gradients to bridge regularization and multi-objective optimization. Besides demonstrating our method on image classification, as another contribution we generalize this area to the multi-modal evaluation settings for robust fine-tuning. Specifically, we first bridge the uni-modal and multi-modal gap by performing analysis on Image Classification reformulated Visual Question Answering (VQA) benchmarks and further categorize ten out-of-distribution (OOD) VQA datasets by distribution shift types and degree (i.e. near versus far OOD). Experimental results show that DiGraP consistently outperforms existing baselines across Image Classfication and VQA tasks with discriminative and generative backbones, improving both in-distribution (ID) generalization and OOD robustness.
>
---
#### [replaced 084] Uncertainty-aware Efficient Subgraph Isomorphism using Graph Topology
- **分类: stat.ML; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2209.09090v3](http://arxiv.org/pdf/2209.09090v3)**

> **作者:** Arpan Kusari; Wenbo Sun
>
> **摘要:** Subgraph isomorphism, also known as subgraph matching, is typically regarded as an NP-complete problem. This complexity is further compounded in practical applications where edge weights are real-valued and may be affected by measurement noise and potential missing data. Such graph matching routinely arises in applications such as image matching and map matching. Most subgraph matching methods fail to perform node-to-node matching under presence of such corruptions. We propose a method for identifying the node correspondence between a subgraph and a full graph in the inexact case without node labels in two steps - (a) extract the minimal unique topology preserving subset from the subgraph and find its feasible matching in the full graph, and (b) implement a consensus-based algorithm to expand the matched node set by pairing unique paths based on boundary commutativity. To demonstrate the effectiveness of the proposed method, a simulation is performed on the Erdos-Renyi random graphs and two case studies are performed on the image-based affine covariant features dataset and KITTI stereo dataset respectively. Going beyond the existing subgraph matching approaches, the proposed method is shown to have realistically sub-linear computational efficiency, robustness to random measurement noise, and good statistical properties. Our method is also readily applicable to the exact matching case without loss of generality.
>
---
#### [replaced 085] MIRAGE: A Multi-modal Benchmark for Spatial Perception, Reasoning, and Intelligence
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.10604v2](http://arxiv.org/pdf/2505.10604v2)**

> **作者:** Chonghan Liu; Haoran Wang; Felix Henry; Pu Miao; Yajie Zhang; Yu Zhao; Peiran Wu
>
> **摘要:** Spatial perception and reasoning are core components of human cognition, encompassing object recognition, spatial relational understanding, and dynamic reasoning. Despite progress in computer vision, existing benchmarks reveal significant gaps in models' abilities to accurately recognize object attributes and reason about spatial relationships, both essential for dynamic reasoning. To address these limitations, we propose MIRAGE, a multi-modal benchmark designed to evaluate models' capabilities in Counting (object attribute recognition), Relation (spatial relational reasoning), and Counting with Relation. Through diverse and complex scenarios requiring fine-grained recognition and reasoning, MIRAGE highlights critical limitations in state-of-the-art models, underscoring the need for improved representations and reasoning frameworks. By targeting these foundational abilities, MIRAGE provides a pathway toward spatiotemporal reasoning in future research.
>
---
#### [replaced 086] Disentangle and Regularize: Sign Language Production with Articulator-Based Disentanglement and Channel-Aware Regularization
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06610v2](http://arxiv.org/pdf/2504.06610v2)**

> **作者:** Sumeyye Meryem Tasyurek; Tugce Kiziltepe; Hacer Yalim Keles
>
> **备注:** 12 pages, 5 figures, 6 table
>
> **摘要:** In this work, we propose DARSLP, a simple gloss-free, transformer-based sign language production (SLP) framework that directly maps spoken-language text to sign pose sequences. We first train a pose autoencoder that encodes sign poses into a compact latent space using an articulator-based disentanglement strategy, where features corresponding to the face, right hand, left hand, and body are modeled separately to promote structured and interpretable representation learning. Next, a non-autoregressive transformer decoder is trained to predict these latent representations from sentence-level text embeddings. To guide this process, we apply channel-aware regularization by aligning predicted latent distributions with priors extracted from the ground-truth encodings using a KL-divergence loss. The contribution of each channel to the loss is weighted according to its associated articulator region, enabling the model to account for the relative importance of different articulators during training. Our approach does not rely on gloss supervision or pretrained models, and achieves state-of-the-art results on the PHOENIX14T and CSL-Daily datasets.
>
---
#### [replaced 087] TextBraTS: Text-Guided Volumetric Brain Tumor Segmentation with Innovative Dataset Development and Fusion Module Exploration
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.16784v2](http://arxiv.org/pdf/2506.16784v2)**

> **作者:** Xiaoyu Shi; Rahul Kumar Jain; Yinhao Li; Ruibo Hou; Jingliang Cheng; Jie Bai; Guohua Zhao; Lanfen Lin; Rui Xu; Yen-wei Chen
>
> **摘要:** Deep learning has demonstrated remarkable success in medical image segmentation and computer-aided diagnosis. In particular, numerous advanced methods have achieved state-of-the-art performance in brain tumor segmentation from MRI scans. While recent studies in other medical imaging domains have revealed that integrating textual reports with visual data can enhance segmentation accuracy, the field of brain tumor analysis lacks a comprehensive dataset that combines radiological images with corresponding textual annotations. This limitation has hindered the exploration of multimodal approaches that leverage both imaging and textual data. To bridge this critical gap, we introduce the TextBraTS dataset, the first publicly available volume-level multimodal dataset that contains paired MRI volumes and rich textual annotations, derived from the widely adopted BraTS2020 benchmark. Building upon this novel dataset, we propose a novel baseline framework and sequential cross-attention method for text-guided volumetric medical image segmentation. Through extensive experiments with various text-image fusion strategies and templated text formulations, our approach demonstrates significant improvements in brain tumor segmentation accuracy, offering valuable insights into effective multimodal integration techniques. Our dataset, implementation code, and pre-trained models are publicly available at https://github.com/Jupitern52/TextBraTS.
>
---
#### [replaced 088] GmNet: Revisiting Gating Mechanisms From A Frequency View
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22841v2](http://arxiv.org/pdf/2503.22841v2)**

> **作者:** Yifan Wang; Xu Ma; Yitian Zhang; Zhongruo Wang; Sung-Cheol Kim; Vahid Mirjalili; Vidya Renganathan; Yun Fu
>
> **摘要:** Gating mechanisms have emerged as an effective strategy integrated into model designs beyond recurrent neural networks for addressing long-range dependency problems. In a broad understanding, it provides adaptive control over the information flow while maintaining computational efficiency. However, there is a lack of theoretical analysis on how the gating mechanism works in neural networks. In this paper, inspired by the \textit{convolution theorem}, we systematically explore the effect of gating mechanisms on the training dynamics of neural networks from a frequency perspective. We investigate the interact between the element-wise product and activation functions in managing the responses to different frequency components. Leveraging these insights, we propose a Gating Mechanism Network (GmNet), a lightweight model designed to efficiently utilize the information of various frequency components. It minimizes the low-frequency bias present in existing lightweight models. GmNet achieves impressive performance in terms of both effectiveness and efficiency in the image classification task.
>
---
#### [replaced 089] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05546v2](http://arxiv.org/pdf/2506.05546v2)**

> **作者:** Vadim Tschernezki; Diane Larlus; Iro Laina; Andrea Vedaldi
>
> **备注:** Camera-ready for CVPR25
>
> **摘要:** Computer vision is largely based on 2D techniques, with 3D vision still relegated to a relatively narrow subset of applications. However, by building on recent advances in 3D models such as neural radiance fields, some authors have shown that 3D techniques can at last improve outputs extracted from independent 2D views, by fusing them into 3D and denoising them. This is particularly helpful in egocentric videos, where the camera motion is significant, but only under the assumption that the scene itself is static. In fact, as shown in the recent analysis conducted by EPIC Fields, 3D techniques are ineffective when it comes to studying dynamic phenomena, and, in particular, when segmenting moving objects. In this paper, we look into this issue in more detail. First, we propose to improve dynamic segmentation in 3D by fusing motion segmentation predictions from a 2D-based model into layered radiance fields (Layered Motion Fusion). However, the high complexity of long, dynamic videos makes it challenging to capture the underlying geometric structure, and, as a result, hinders the fusion of motion cues into the (incomplete) scene geometry. We address this issue through test-time refinement, which helps the model to focus on specific frames, thereby reducing the data complexity. This results in a synergy between motion fusion and the refinement, and in turn leads to segmentation predictions of the 3D model that surpass the 2D baseline by a large margin. This demonstrates that 3D techniques can enhance 2D analysis even for dynamic phenomena in a challenging and realistic setting.
>
---
#### [replaced 090] FLARE: Toward Universal Dataset Purification against Backdoor Attacks
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19479v3](http://arxiv.org/pdf/2411.19479v3)**

> **作者:** Linshan Hou; Wei Luo; Zhongyun Hua; Songhua Chen; Leo Yu Zhang; Yiming Li
>
> **备注:** 15 pages, This paper is accepted and will appear in TIFS (CCF-A)
>
> **摘要:** Deep neural networks (DNNs) are susceptible to backdoor attacks, where adversaries poison datasets with adversary-specified triggers to implant hidden backdoors, enabling malicious manipulation of model predictions. Dataset purification serves as a proactive defense by removing malicious training samples to prevent backdoor injection at its source. We first reveal that the current advanced purification methods rely on a latent assumption that the backdoor connections between triggers and target labels in backdoor attacks are simpler to learn than the benign features. We demonstrate that this assumption, however, does not always hold, especially in all-to-all (A2A) and untargeted (UT) attacks. As a result, purification methods that analyze the separation between the poisoned and benign samples in the input-output space or the final hidden layer space are less effective. We observe that this separability is not confined to a single layer but varies across different hidden layers. Motivated by this understanding, we propose FLARE, a universal purification method to counter various backdoor attacks. FLARE aggregates abnormal activations from all hidden layers to construct representations for clustering. To enhance separation, FLARE develops an adaptive subspace selection algorithm to isolate the optimal space for dividing an entire dataset into two clusters. FLARE assesses the stability of each cluster and identifies the cluster with higher stability as poisoned. Extensive evaluations on benchmark datasets demonstrate the effectiveness of FLARE against 22 representative backdoor attacks, including all-to-one (A2O), all-to-all (A2A), and untargeted (UT) attacks, and its robustness to adaptive attacks. Codes are available at \href{https://github.com/THUYimingLi/BackdoorBox}{BackdoorBox} and \href{https://github.com/vtu81/backdoor-toolbox}{backdoor-toolbox}.
>
---
#### [replaced 091] How Visual Representations Map to Language Feature Space in Multimodal LLMs
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.11976v2](http://arxiv.org/pdf/2506.11976v2)**

> **作者:** Constantin Venhoff; Ashkan Khakzar; Sonia Joseph; Philip Torr; Neel Nanda
>
> **摘要:** Effective multimodal reasoning depends on the alignment of visual and linguistic representations, yet the mechanisms by which vision-language models (VLMs) achieve this alignment remain poorly understood. Following the LiMBeR framework, we deliberately maintain a frozen large language model (LLM) and a frozen vision transformer (ViT), connected solely by training a linear adapter during visual instruction tuning. By keeping the language model frozen, we ensure it maintains its original language representations without adaptation to visual data. Consequently, the linear adapter must map visual features directly into the LLM's existing representational space rather than allowing the language model to develop specialized visual understanding through fine-tuning. Our experimental design uniquely enables the use of pre-trained sparse autoencoders (SAEs) of the LLM as analytical probes. These SAEs remain perfectly aligned with the unchanged language model and serve as a snapshot of the learned language feature-representations. Through systematic analysis of SAE reconstruction error, sparsity patterns, and feature SAE descriptions, we reveal the layer-wise progression through which visual representations gradually align with language feature representations, converging in middle-to-later layers. This suggests a fundamental misalignment between ViT outputs and early LLM layers, raising important questions about whether current adapter-based architectures optimally facilitate cross-modal representation learning.
>
---
#### [replaced 092] LAPIG: Language Guided Projector Image Generation with Surface Adaptation and Stylization
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.12173v2](http://arxiv.org/pdf/2503.12173v2)**

> **作者:** Yuchen Deng; Haibin Ling; Bingyao Huang
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** We propose LAPIG, a language guided projector image generation method with surface adaptation and stylization. LAPIG consists of a projector-camera system and a target textured projection surface. LAPIG takes the user text prompt as input and aims to transform the surface style using the projector. LAPIG's key challenge is that due to the projector's physical brightness limitation and the surface texture, the viewer's perceived projection may suffer from color saturation and artifacts in both dark and bright regions, such that even with the state-of-the-art projector compensation techniques, the viewer may see clear surface texture-related artifacts. Therefore, how to generate a projector image that follows the user's instruction while also displaying minimum surface artifacts is an open problem. To address this issue, we propose projection surface adaptation (PSA) that can generate compensable surface stylization. We first train two networks to simulate the projector compensation and project-and-capture processes, this allows us to find a satisfactory projector image without real project-and-capture and utilize gradient descent for fast convergence. Then, we design content and saturation losses to guide the projector image generation, such that the generated image shows no clearly perceivable artifacts when projected. Finally, the generated image is projected for visually pleasing surface style morphing effects. The source code and video are available on the project page: https://Yu-chen-Deng.github.io/LAPIG/.
>
---
#### [replaced 093] Indeterminate Probability Theory
- **分类: cs.LG; cs.AI; cs.CV; math.ST; stat.ML; stat.TH**

- **链接: [http://arxiv.org/pdf/2303.11536v2](http://arxiv.org/pdf/2303.11536v2)**

> **作者:** Tao Yang; Chuang Liu; Xiaofeng Ma; Weijia Lu; Ning Wu; Bingyang Li; Zhifei Yang; Peng Liu; Lin Sun; Xiaodong Zhang; Can Zhang
>
> **备注:** 25 pages
>
> **摘要:** Complex continuous or mixed joint distributions (e.g., P(Y | z_1, z_2, ..., z_N)) generally lack closed-form solutions, often necessitating approximations such as MCMC. This paper proposes Indeterminate Probability Theory (IPT), which makes the following contributions: (1) An observer-centered framework in which experimental outcomes are represented as distributions combining ground truth with observation error; (2) The introduction of three independence candidate axioms that enable a two-phase probabilistic inference framework; (3) The derivation of closed-form solutions for arbitrary complex joint distributions under this framework. Both the Indeterminate Probability Neural Network (IPNN) model and the non-neural multivariate time series forecasting application demonstrate IPT's effectiveness in modeling high-dimensional distributions, with successful validation up to 1000 dimensions. Importantly, IPT is consistent with classical probability theory and subsumes the frequentist equation in the limit of vanishing observation error.
>
---
#### [replaced 094] G3Flow: Generative 3D Semantic Flow for Pose-aware and Generalizable Object Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2411.18369v3](http://arxiv.org/pdf/2411.18369v3)**

> **作者:** Tianxing Chen; Yao Mu; Zhixuan Liang; Zanxin Chen; Shijia Peng; Qiangyu Chen; Mingkun Xu; Ruizhen Hu; Hongyuan Zhang; Xuelong Li; Ping Luo
>
> **备注:** Webpage: https://tianxingchen.github.io/G3Flow/, accepted to CVPR 2025
>
> **摘要:** Recent advances in imitation learning for 3D robotic manipulation have shown promising results with diffusion-based policies. However, achieving human-level dexterity requires seamless integration of geometric precision and semantic understanding. We present G3Flow, a novel framework that constructs real-time semantic flow, a dynamic, object-centric 3D semantic representation by leveraging foundation models. Our approach uniquely combines 3D generative models for digital twin creation, vision foundation models for semantic feature extraction, and robust pose tracking for continuous semantic flow updates. This integration enables complete semantic understanding even under occlusions while eliminating manual annotation requirements. By incorporating semantic flow into diffusion policies, we demonstrate significant improvements in both terminal-constrained manipulation and cross-object generalization. Extensive experiments across five simulation tasks show that G3Flow consistently outperforms existing approaches, achieving up to 68.3% and 50.1% average success rates on terminal-constrained manipulation and cross-object generalization tasks respectively. Our results demonstrate the effectiveness of G3Flow in enhancing real-time dynamic semantic feature understanding for robotic manipulation policies.
>
---
#### [replaced 095] SALT: A Flexible Semi-Automatic Labeling Tool for General LiDAR Point Clouds with Cross-Scene Adaptability and 4D Consistency
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.23980v2](http://arxiv.org/pdf/2503.23980v2)**

> **作者:** Yanbo Wang; Yongtao Chen; Chuan Cao; Tianchen Deng; Wentao Zhao; Jingchuan Wang; Weidong Chen
>
> **摘要:** We propose a flexible Semi-Automatic Labeling Tool (SALT) for general LiDAR point clouds with cross-scene adaptability and 4D consistency. Unlike recent approaches that rely on camera distillation, SALT operates directly on raw LiDAR data, automatically generating pre-segmentation results. To achieve this, we propose a novel zero-shot learning paradigm, termed data alignment, which transforms LiDAR data into pseudo-images by aligning with the training distribution of vision foundation models. Additionally, we design a 4D-consistent prompting strategy and 4D non-maximum suppression module to enhance SAM2, ensuring high-quality, temporally consistent presegmentation. SALT surpasses the latest zero-shot methods by 18.4% PQ on SemanticKITTI and achieves nearly 40-50% of human annotator performance on our newly collected low-resolution LiDAR data and on combined data from three LiDAR types, significantly boosting annotation efficiency. We anticipate that SALT's open-sourcing will catalyze substantial expansion of current LiDAR datasets and lay the groundwork for the future development of LiDAR foundation models. Code is available at https://github.com/Cavendish518/SALT.
>
---
#### [replaced 096] Interpreting Global Perturbation Robustness of Image Models using Axiomatic Spectral Importance Decomposition
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.01139v3](http://arxiv.org/pdf/2408.01139v3)**

> **作者:** Róisín Luo; James McDermott; Colm O'Riordan
>
> **备注:** Accepted by Transactions on Machine Learning Research (TMLR 2024)
>
> **摘要:** Perturbation robustness evaluates the vulnerabilities of models, arising from a variety of perturbations, such as data corruptions and adversarial attacks. Understanding the mechanisms of perturbation robustness is critical for global interpretability. We present a model-agnostic, global mechanistic interpretability method to interpret the perturbation robustness of image models. This research is motivated by two key aspects. First, previous global interpretability works, in tandem with robustness benchmarks, e.g. mean corruption error (mCE), are not designed to directly interpret the mechanisms of perturbation robustness within image models. Second, we notice that the spectral signal-to-noise ratios (SNR) of perturbed natural images exponentially decay over the frequency. This power-law-like decay implies that: Low-frequency signals are generally more robust than high-frequency signals -- yet high classification accuracy can not be achieved by low-frequency signals alone. By applying Shapley value theory, our method axiomatically quantifies the predictive powers of robust features and non-robust features within an information theory framework. Our method, dubbed as \textbf{I-ASIDE} (\textbf{I}mage \textbf{A}xiomatic \textbf{S}pectral \textbf{I}mportance \textbf{D}ecomposition \textbf{E}xplanation), provides a unique insight into model robustness mechanisms. We conduct extensive experiments over a variety of vision models pre-trained on ImageNet to show that \textbf{I-ASIDE} can not only \textbf{measure} the perturbation robustness but also \textbf{provide interpretations} of its mechanisms.
>
---
#### [replaced 097] MDeRainNet: An Efficient Macro-pixel Image Rain Removal Network
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.10652v2](http://arxiv.org/pdf/2406.10652v2)**

> **作者:** Tao Yan; Weijiang He; Chenglong Wang; Xiangjie Zhu; Yinghui Wang; Rynson W. H. Lau
>
> **备注:** 14 pages, 13 figures, 4 tables
>
> **摘要:** Since rainy weather always degrades image quality and poses significant challenges to most computer vision-based intelligent systems, image de-raining has been a hot research topic. Fortunately, in a rainy light field (LF) image, background obscured by rain streaks in one sub-view may be visible in the other sub-views, and implicit depth information and recorded 4D structural information may benefit rain streak detection and removal. However, existing LF image rain removal methods either do not fully exploit the global correlations of 4D LF data or only utilize partial sub-views, resulting in sub-optimal rain removal performance and no-equally good quality for all de-rained sub-views. In this paper, we propose an efficient network, called MDeRainNet, for rain streak removal from LF images. The proposed network adopts a multi-scale encoder-decoder architecture, which directly works on Macro-pixel images (MPIs) to improve the rain removal performance. To fully model the global correlation between the spatial and the angular information, we propose an Extended Spatial-Angular Interaction (ESAI) module to merge them, in which a simple and effective Transformer-based Spatial-Angular Interaction Attention (SAIA) block is also proposed for modeling long-range geometric correlations and making full use of the angular information. Furthermore, to improve the generalization performance of our network on real-world rainy scenes, we propose a novel semi-supervised learning framework for our MDeRainNet, which utilizes multi-level KL loss to bridge the domain gap between features of synthetic and real-world rain streaks and introduces colored-residue image guided contrastive regularization to reconstruct rain-free images. Extensive experiments conducted on synthetic and real-world LFIs demonstrate that our method outperforms the state-of-the-art methods both quantitatively and qualitatively.
>
---
#### [replaced 098] Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model
- **分类: cs.AI; cs.CL; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.13642v2](http://arxiv.org/pdf/2506.13642v2)**

> **作者:** Shaolei Zhang; Shoutao Guo; Qingkai Fang; Yan Zhou; Yang Feng
>
> **备注:** Code: https://github.com/ictnlp/Stream-Omni , Model: https://huggingface.co/ICTNLP/stream-omni-8b
>
> **摘要:** The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience.
>
---
#### [replaced 099] EDA-DM: Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2401.04585v3](http://arxiv.org/pdf/2401.04585v3)**

> **作者:** Xuewen Liu; Zhikai Li; Junrui Xiao; Mengjuan Chen; Jianquan Li; Qingyi Gu
>
> **备注:** Code: http://github.com/BienLuky/EDA-DM
>
> **摘要:** Diffusion models have achieved great success in image generation tasks. However, the lengthy denoising process and complex neural networks hinder their low-latency applications in real-world scenarios. Quantization can effectively reduce model complexity, and post-training quantization (PTQ), which does not require fine-tuning, is highly promising for compressing and accelerating diffusion models. Unfortunately, we find that due to the highly dynamic activations, existing PTQ methods suffer from distribution mismatch issues at both calibration sample level and reconstruction output level, which makes the performance far from satisfactory. In this paper, we propose EDA-DM, a standardized PTQ method that efficiently addresses the above issues. Specifically, at the calibration sample level, we extract information from the density and diversity of latent space feature maps, which guides the selection of calibration samples to align with the overall sample distribution; and at the reconstruction output level, we theoretically analyze the reasons for previous reconstruction failures and, based on this insight, optimize block reconstruction using the Hessian loss of layers, aligning the outputs of quantized model and full-precision model at different network granularity. Extensive experiments demonstrate that EDA-DM significantly outperforms the existing PTQ methods across various models and datasets. Our method achieves a 1.83 times speedup and 4 times compression for the popular Stable-Diffusion on MS-COCO, with only a 0.05 loss in CLIP score. Code is available at http://github.com/BienLuky/EDA-DM .
>
---
#### [replaced 100] Steerable Transformers for Volumetric Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.15932v3](http://arxiv.org/pdf/2405.15932v3)**

> **作者:** Soumyabrata Kundu; Risi Kondor
>
> **摘要:** We introduce Steerable Transformers, an extension of the Vision Transformer mechanism that maintains equivariance to the special Euclidean group $\mathrm{SE}(d)$. We propose an equivariant attention mechanism that operates on features extracted by steerable convolutions. Operating in Fourier space, our network utilizes Fourier space non-linearities. Our experiments in both two and three dimensions show that adding steerable transformer layers to steerable convolutional networks enhances performance.
>
---
#### [replaced 101] Reasoning Limitations of Multimodal Large Language Models. A Case Study of Bongard Problems
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.01173v2](http://arxiv.org/pdf/2411.01173v2)**

> **作者:** Mikołaj Małkiński; Szymon Pawlonka; Jacek Mańdziuk
>
> **备注:** Accepted to The Forty-Second International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Abstract visual reasoning (AVR) involves discovering shared concepts across images through analogy, akin to solving IQ test problems. Bongard Problems (BPs) remain a key challenge in AVR, requiring both visual reasoning and verbal description. We investigate whether multimodal large language models (MLLMs) can solve BPs by formulating a set of diverse MLLM-suited solution strategies and testing $4$ proprietary and $4$ open-access models on $3$ BP datasets featuring synthetic (classic BPs) and real-world (Bongard HOI and Bongard-OpenWorld) images. Despite some successes on real-world datasets, MLLMs struggle with synthetic BPs. To explore this gap, we introduce Bongard-RWR, a dataset representing synthetic BP concepts using real-world images. Our findings suggest that weak MLLM performance on classical BPs is not due to the domain specificity, but rather comes from their general AVR limitations. Code and dataset are available at: https://github.com/pavonism/bongard-rwr
>
---
#### [replaced 102] Auto-Lesion Segmentation with a Novel Intensity Dark Channel Prior for COVID-19 Detection
- **分类: eess.IV; cs.CV; math.MG; math.OC**

- **链接: [http://arxiv.org/pdf/2309.12638v2](http://arxiv.org/pdf/2309.12638v2)**

> **作者:** Basma Jumaa Saleh; Zaid Omar; Vikrant Bhateja; Lila Iznita Izhar
>
> **备注:** The study requires withdrawal due to technical inconsistencies in the reported data that affect the conclusions. We apologize for any inconvenience
>
> **摘要:** During the COVID-19 pandemic, medical imaging techniques like computed tomography (CT) scans have demonstrated effectiveness in combating the rapid spread of the virus. Therefore, it is crucial to conduct research on computerized models for the detection of COVID-19 using CT imaging. A novel processing method has been developed, utilizing radiomic features, to assist in the CT-based diagnosis of COVID-19. Given the lower specificity of traditional features in distinguishing between different causes of pulmonary diseases, the objective of this study is to develop a CT-based radiomics framework for the differentiation of COVID-19 from other lung diseases. The model is designed to focus on outlining COVID-19 lesions, as traditional features often lack specificity in this aspect. The model categorizes images into three classes: COVID-19, non-COVID-19, or normal. It employs enhancement auto-segmentation principles using intensity dark channel prior (IDCP) and deep neural networks (ALS-IDCP-DNN) within a defined range of analysis thresholds. A publicly available dataset comprising COVID-19, normal, and non-COVID-19 classes was utilized to validate the proposed model's effectiveness. The best performing classification model, Residual Neural Network with 50 layers (Resnet-50), attained an average accuracy, precision, recall, and F1-score of 98.8%, 99%, 98%, and 98% respectively. These results demonstrate the capability of our model to accurately classify COVID-19 images, which could aid radiologists in diagnosing suspected COVID-19 patients. Furthermore, our model's performance surpasses that of more than 10 current state-of-the-art studies conducted on the same dataset.
>
---
#### [replaced 103] Disentangling representations of retinal images with generative models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.19186v3](http://arxiv.org/pdf/2402.19186v3)**

> **作者:** Sarah Müller; Lisa M. Koch; Hendrik P. A. Lensch; Philipp Berens
>
> **备注:** Final journal paper version for Medical Image Analysis (MedIA)
>
> **摘要:** Retinal fundus images play a crucial role in the early detection of eye diseases. However, the impact of technical factors on these images can pose challenges for reliable AI applications in ophthalmology. For example, large fundus cohorts are often confounded by factors like camera type, bearing the risk of learning shortcuts rather than the causal relationships behind the image generation process. Here, we introduce a population model for retinal fundus images that effectively disentangles patient attributes from camera effects, enabling controllable and highly realistic image generation. To achieve this, we propose a disentanglement loss based on distance correlation. Through qualitative and quantitative analyses, we show that our models encode desired information in disentangled subspaces and enable controllable image generation based on the learned subspaces, demonstrating the effectiveness of our disentanglement loss. The project's code is publicly available: https://github.com/berenslab/disentangling-retinal-images.
>
---
#### [replaced 104] Segment Anything for Satellite Imagery: A Strong Baseline and a Regional Dataset for Automatic Field Delineation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.16318v2](http://arxiv.org/pdf/2506.16318v2)**

> **作者:** Carmelo Scribano; Elena Govi; Paolo Bertellini; Simone Parisi; Giorgia Franchini; Marko Bertogna
>
> **备注:** Acceptet at ICIAP 2025
>
> **摘要:** Accurate mapping of agricultural field boundaries is essential for the efficient operation of agriculture. Automatic extraction from high-resolution satellite imagery, supported by computer vision techniques, can avoid costly ground surveys. In this paper, we present a pipeline for field delineation based on the Segment Anything Model (SAM), introducing a fine-tuning strategy to adapt SAM to this task. In addition to using published datasets, we describe a method for acquiring a complementary regional dataset that covers areas beyond current sources. Extensive experiments assess segmentation accuracy and evaluate the generalization capabilities. Our approach provides a robust baseline for automated field delineation. The new regional dataset, known as ERAS, is now publicly available.
>
---
#### [replaced 105] Navigating Conflicting Views: Harnessing Trust for Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.00958v4](http://arxiv.org/pdf/2406.00958v4)**

> **作者:** Jueqing Lu; Wray Buntine; Yuanyuan Qi; Joanna Dipnall; Belinda Gabbe; Lan Du
>
> **摘要:** Resolving conflicts is critical for improving the reliability of multi-view classification. While prior work focuses on learning consistent and informative representations across views, it often assumes perfect alignment and equal importance of all views, an assumption rarely met in real-world scenarios, as some views may express distinct information. To address this, we develop a computational trust-based discounting method that enhances the Evidential Multi-view framework by accounting for the instance-wise reliability of each view through a probability-sensitive trust mechanism. We evaluate our method on six real-world datasets using Top-1 Accuracy, Fleiss' Kappa, and a new metric, Multi-View Agreement with Ground Truth, to assess prediction reliability. We also assess the effectiveness of uncertainty in indicating prediction correctness via AUROC. Additionally, we test the scalability of our method through end-to-end training on a large-scale dataset. The experimental results show that computational trust can effectively resolve conflicts, paving the way for more reliable multi-view classification models in real-world applications. Codes available at: https://github.com/OverfitFlow/Trust4Conflict
>
---
#### [replaced 106] GAF: Gaussian Action Field as a Dynamic World Model for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14135v2](http://arxiv.org/pdf/2506.14135v2)**

> **作者:** Ying Chai; Litao Deng; Ruizhi Shao; Jiajun Zhang; Liangjun Xing; Hongwen Zhang; Yebin Liu
>
> **备注:** http://chaiying1.github.io/GAF.github.io/project_page/
>
> **摘要:** Accurate action inference is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we propose a Vision-to-4D-to-Action (V-4D-A) framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing simultaneous modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF supports three key query types: reconstruction of the current scene, prediction of future frames, and estimation of initial action via robot motion. Furthermore, the high-quality current and future frames generated by GAF facilitate manipulation action refinement through a GAF-guided diffusion model. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average success rate in robotic manipulation tasks by 10.33% over state-of-the-art methods. Project page: http://chaiying1.github.io/GAF.github.io/project_page/
>
---
#### [replaced 107] Thermal Vision: Pioneering Non-Invasive Temperature Tracking in Congested Spaces
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.00863v2](http://arxiv.org/pdf/2412.00863v2)**

> **作者:** Arijit Samal; Haroon R Lone
>
> **摘要:** Non-invasive temperature monitoring of individuals plays a crucial role in identifying and isolating symptomatic individuals. Temperature monitoring becomes particularly vital in settings characterized by close human proximity, often referred to as dense settings. However, existing research on non-invasive temperature estimation using thermal cameras has predominantly focused on sparse settings. Unfortunately, the risk of disease transmission is significantly higher in dense settings like movie theaters or classrooms. Consequently, there is an urgent need to develop robust temperature estimation methods tailored explicitly for dense settings. Our study proposes a non-invasive temperature estimation system that combines a thermal camera with an edge device. Our system employs YOLO models for face detection and utilizes a regression framework for temperature estimation. We evaluated the system on a diverse dataset collected in dense and sparse settings. Our proposed face detection model achieves an impressive mAP score of over 84 in both in-dataset and cross-dataset evaluations. Furthermore, the regression framework demonstrates remarkable performance with a mean square error of 0.18$^{\circ}$C and an impressive $R^2$ score of 0.96. Our experiments' results highlight the developed system's effectiveness, positioning it as a promising solution for continuous temperature monitoring in real-world applications. With this paper, we release our dataset and programming code publicly.
>
---
#### [replaced 108] Accurate early detection of Parkinson's disease from SPECT imaging through Convolutional Neural Networks
- **分类: eess.IV; cs.CV; cs.LG; stat.AP**

- **链接: [http://arxiv.org/pdf/2412.05348v2](http://arxiv.org/pdf/2412.05348v2)**

> **作者:** R. Prashanth
>
> **备注:** This article is accepted and published with revisions to the Artificial Intelligence in Health journal (2025). The accepted article can be accessed at https://doi.org/10.36922/AIH025040005
>
> **摘要:** Early and accurate detection of Parkinson's disease (PD) is a crucial diagnostic challenge carrying immense clinical significance, for effective treatment regimens and patient management. For instance, a group of subjects termed SWEDD who are clinically diagnosed as PD, but show normal Single Photon Emission Computed Tomography (SPECT) scans, change their diagnosis as non-PD after few years of follow up, and in the meantime, they are treated with PD medications which do more harm than good. In this work, machine learning models are developed using features from SPECT images to detect early PD and SWEDD subjects from normal. These models were observed to perform with high accuracy. It is inferred from the study that these diagnostic models carry potential to help PD clinicians in the diagnostic process
>
---
#### [replaced 109] CAD-GPT: Synthesising CAD Construction Sequence with Spatial Reasoning-Enhanced Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2412.19663v2](http://arxiv.org/pdf/2412.19663v2)**

> **作者:** Siyu Wang; Cailian Chen; Xinyi Le; Qimin Xu; Lei Xu; Yanzhou Zhang; Jie Yang
>
> **备注:** Accepted at AAAI 2025 (Vol. 39, No. 8), pages 7880-7888. DOI: 10.1609/aaai.v39i8.32849
>
> **摘要:** Computer-aided design (CAD) significantly enhances the efficiency, accuracy, and innovation of design processes by enabling precise 2D and 3D modeling, extensive analysis, and optimization. Existing methods for creating CAD models rely on latent vectors or point clouds, which are difficult to obtain, and storage costs are substantial. Recent advances in Multimodal Large Language Models (MLLMs) have inspired researchers to use natural language instructions and images for CAD model construction. However, these models still struggle with inferring accurate 3D spatial location and orientation, leading to inaccuracies in determining the spatial 3D starting points and extrusion directions for constructing geometries. This work introduces CAD-GPT, a CAD synthesis method with spatial reasoning-enhanced MLLM that takes either a single image or a textual description as input. To achieve precise spatial inference, our approach introduces a 3D Modeling Spatial Mechanism. This method maps 3D spatial positions and 3D sketch plane rotation angles into a 1D linguistic feature space using a specialized spatial unfolding mechanism, while discretizing 2D sketch coordinates into an appropriate planar space to enable precise determination of spatial starting position, sketch orientation, and 2D sketch coordinate translations. Extensive experiments demonstrate that CAD-GPT consistently outperforms existing state-of-the-art methods in CAD model synthesis, both quantitatively and qualitatively.
>
---
#### [replaced 110] Human Action CLIPs: Detecting AI-generated Human Motion
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2412.00526v2](http://arxiv.org/pdf/2412.00526v2)**

> **作者:** Matyas Bohacek; Hany Farid
>
> **摘要:** AI-generated video generation continues its journey through the uncanny valley to produce content that is increasingly perceptually indistinguishable from reality. To better protect individuals, organizations, and societies from its malicious applications, we describe an effective and robust technique for distinguishing real from AI-generated human motion using multi-modal semantic embeddings. Our method is robust to the types of laundering that typically confound more low- to mid-level approaches, including resolution and compression attacks. This method is evaluated against DeepAction, a custom-built, open-sourced dataset of video clips with human actions generated by seven text-to-video AI models and matching real footage. The dataset is available under an academic license at https://www.huggingface.co/datasets/faridlab/deepaction_v1.
>
---
#### [replaced 111] DART: An Automated End-to-End Object Detection Pipeline with Data Diversification, Open-Vocabulary Bounding Box Annotation, Pseudo-Label Review, and Model Training
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.09174v4](http://arxiv.org/pdf/2407.09174v4)**

> **作者:** Chen Xin; Andreas Hartel; Enkelejda Kasneci
>
> **备注:** Corrected minor typos; no changes to results or conclusions
>
> **摘要:** Accurate real-time object detection is vital across numerous industrial applications, from safety monitoring to quality control. Traditional approaches, however, are hindered by arduous manual annotation and data collection, struggling to adapt to ever-changing environments and novel target objects. To address these limitations, this paper presents DART, an innovative automated end-to-end pipeline that revolutionizes object detection workflows from data collection to model evaluation. It eliminates the need for laborious human labeling and extensive data collection while achieving outstanding accuracy across diverse scenarios. DART encompasses four key stages: (1) Data Diversification using subject-driven image generation (DreamBooth with SDXL), (2) Annotation via open-vocabulary object detection (Grounding DINO) to generate bounding box and class labels, (3) Review of generated images and pseudo-labels by large multimodal models (InternVL-1.5 and GPT-4o) to guarantee credibility, and (4) Training of real-time object detectors (YOLOv8 and YOLOv10) using the verified data. We apply DART to a self-collected dataset of construction machines named Liebherr Product, which contains over 15K high-quality images across 23 categories. The current instantiation of DART significantly increases average precision (AP) from 0.064 to 0.832. Its modular design ensures easy exchangeability and extensibility, allowing for future algorithm upgrades, seamless integration of new object categories, and adaptability to customized environments without manual labeling and additional data collection. The code and dataset are released at https://github.com/chen-xin-94/DART.
>
---
#### [replaced 112] Hallucination-Aware Multimodal Benchmark for Gastrointestinal Image Analysis with Large Vision-Language Models
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.07001v2](http://arxiv.org/pdf/2505.07001v2)**

> **作者:** Bidur Khanal; Sandesh Pokhrel; Sanjay Bhandari; Ramesh Rana; Nikesh Shrestha; Ram Bahadur Gurung; Cristian Linte; Angus Watson; Yash Raj Shrestha; Binod Bhattarai
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Vision-Language Models (VLMs) are becoming increasingly popular in the medical domain, bridging the gap between medical images and clinical language. Existing VLMs demonstrate an impressive ability to comprehend medical images and text queries to generate detailed, descriptive diagnostic medical reports. However, hallucination--the tendency to generate descriptions that are inconsistent with the visual content--remains a significant issue in VLMs, with particularly severe implications in the medical field. To facilitate VLM research on gastrointestinal (GI) image analysis and study hallucination, we curate a multimodal image-text GI dataset: Gut-VLM. This dataset is created using a two-stage pipeline: first, descriptive medical reports of Kvasir-v2 images are generated using ChatGPT, which introduces some hallucinated or incorrect texts. In the second stage, medical experts systematically review these reports, and identify and correct potential inaccuracies to ensure high-quality, clinically reliable annotations. Unlike traditional datasets that contain only descriptive texts, our dataset also features tags identifying hallucinated sentences and their corresponding corrections. A common approach to reducing hallucination in VLM is to finetune the model on a small-scale, problem-specific dataset. However, we take a different strategy using our dataset. Instead of finetuning the VLM solely for generating textual reports, we finetune it to detect and correct hallucinations, an approach we call hallucination-aware finetuning. Our results show that this approach is better than simply finetuning for descriptive report generation. Additionally, we conduct an extensive evaluation of state-of-the-art VLMs across several metrics, establishing a benchmark. GitHub Repo: https://github.com/bhattarailab/Hallucination-Aware-VLM.
>
---
#### [replaced 113] Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event-Based Eye Tracking with Motion-Aware Post-Processing
- **分类: cs.CV; cs.HC; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.12524v2](http://arxiv.org/pdf/2506.12524v2)**

> **作者:** Nuwan Bandara; Thivya Kandappu; Archan Misra
>
> **备注:** Accepted at IJCAI-W'25: Workshop for 4D Micro-Expression Recognition for Mind Reading, August 16--22, 2025, Montreal, Canada & Guangzhou, China
>
> **摘要:** Event-based eye tracking holds significant promise for fine-grained cognitive state inference, offering high temporal resolution and robustness to motion artifacts, critical features for decoding subtle mental states such as attention, confusion, or fatigue. In this work, we introduce a model-agnostic, inference-time refinement framework designed to enhance the output of existing event-based gaze estimation models without modifying their architecture or requiring retraining. Our method comprises two key post-processing modules: (i) Motion-Aware Median Filtering, which suppresses blink-induced spikes while preserving natural gaze dynamics, and (ii) Optical Flow-Based Local Refinement, which aligns gaze predictions with cumulative event motion to reduce spatial jitter and temporal discontinuities. To complement traditional spatial accuracy metrics, we propose a novel Jitter Metric that captures the temporal smoothness of predicted gaze trajectories based on velocity regularity and local signal complexity. Together, these contributions significantly improve the consistency of event-based gaze signals, making them better suited for downstream tasks such as micro-expression analysis and mind-state decoding. Our results demonstrate consistent improvements across multiple baseline models on controlled datasets, laying the groundwork for future integration with multimodal affect recognition systems in real-world environments.
>
---
