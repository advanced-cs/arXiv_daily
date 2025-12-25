# 计算机视觉 cs.CV

- **最新发布 77 篇**

- **更新 38 篇**

## 最新发布

#### [new 001] Beyond Pixel Simulation: Pathology Image Generation via Diagnostic Semantic Tokens and Prototype Control
- **分类: cs.CV**

- **简介: 提出UniPath框架，解决病理图像生成中语义控制弱、数据稀缺问题，通过多流控制实现高保真、细粒度可控生成，性能达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.21058v1](https://arxiv.org/pdf/2512.21058v1)**

> **作者:** Minghao Han; YiChen Liu; Yizhou Liu; Zizhi Chen; Jingqun Tang; Xuecheng Wu; Dingkang Yang; Lihua Zhang
>
> **备注:** 32 pages, 17 figures, and 6 tables
>
> **摘要:** In computational pathology, understanding and generation have evolved along disparate paths: advanced understanding models already exhibit diagnostic-level competence, whereas generative models largely simulate pixels. Progress remains hindered by three coupled factors: the scarcity of large, high-quality image-text corpora; the lack of precise, fine-grained semantic control, which forces reliance on non-semantic cues; and terminological heterogeneity, where diverse phrasings for the same diagnostic concept impede reliable text conditioning. We introduce UniPath, a semantics-driven pathology image generation framework that leverages mature diagnostic understanding to enable controllable generation. UniPath implements Multi-Stream Control: a Raw-Text stream; a High-Level Semantics stream that uses learnable queries to a frozen pathology MLLM to distill paraphrase-robust Diagnostic Semantic Tokens and to expand prompts into diagnosis-aware attribute bundles; and a Prototype stream that affords component-level morphological control via a prototype bank. On the data front, we curate a 2.65M image-text corpus and a finely annotated, high-quality 68K subset to alleviate data scarcity. For a comprehensive assessment, we establish a four-tier evaluation hierarchy tailored to pathology. Extensive experiments demonstrate UniPath's SOTA performance, including a Patho-FID of 80.9 (51% better than the second-best) and fine-grained semantic control achieving 98.7% of the real-image. The meticulously curated datasets, complete source code, and pre-trained model weights developed in this study will be made openly accessible to the public.
>
---
#### [new 002] NULLBUS: Multimodal Mixed-Supervision for Breast Ultrasound Segmentation via Nullable Global-Local Prompts
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出NullBUS框架，解决乳腺超声分割中提示缺失问题，通过可空全局-局部提示实现有无文本提示的混合监督训练，提升模型鲁棒性与性能。**

- **链接: [https://arxiv.org/pdf/2512.20783v1](https://arxiv.org/pdf/2512.20783v1)**

> **作者:** Raja Mallina; Bryar Shareef
>
> **备注:** 5 pages, 2 figures, and 4 tables
>
> **摘要:** Breast ultrasound (BUS) segmentation provides lesion boundaries essential for computer-aided diagnosis and treatment planning. While promptable methods can improve segmentation performance and tumor delineation when text or spatial prompts are available, many public BUS datasets lack reliable metadata or reports, constraining training to small multimodal subsets and reducing robustness. We propose NullBUS, a multimodal mixed-supervision framework that learns from images with and without prompts in a single model. To handle missing text, we introduce nullable prompts, implemented as learnable null embeddings with presence masks, enabling fallback to image-only evidence when metadata are absent and the use of text when present. Evaluated on a unified pool of three public BUS datasets, NullBUS achieves a mean IoU of 0.8568 and a mean Dice of 0.9103, demonstrating state-of-the-art performance under mixed prompt availability.
>
---
#### [new 003] TGC-Net: A Structure-Aware and Semantically-Aligned Framework for Text-Guided Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 提出TGC-Net，基于CLIP改进文本引导医学图像分割，解决结构保留、语义对齐与临床描述建模问题，实现高效高精度分割。**

- **链接: [https://arxiv.org/pdf/2512.21135v1](https://arxiv.org/pdf/2512.21135v1)**

> **作者:** Gaoren Lin; Huangxuan Zhao; Yuan Xiong; Lefei Zhang; Bo Du; Wentao Zhu
>
> **摘要:** Text-guided medical segmentation enhances segmentation accuracy by utilizing clinical reports as auxiliary information. However, existing methods typically rely on unaligned image and text encoders, which necessitate complex interaction modules for multimodal fusion. While CLIP provides a pre-aligned multimodal feature space, its direct application to medical imaging is limited by three main issues: insufficient preservation of fine-grained anatomical structures, inadequate modeling of complex clinical descriptions, and domain-specific semantic misalignment. To tackle these challenges, we propose TGC-Net, a CLIP-based framework focusing on parameter-efficient, task-specific adaptations. Specifically, it incorporates a Semantic-Structural Synergy Encoder (SSE) that augments CLIP's ViT with a CNN branch for multi-scale structural refinement, a Domain-Augmented Text Encoder (DATE) that injects large-language-model-derived medical knowledge, and a Vision-Language Calibration Module (VLCM) that refines cross-modal correspondence in a unified feature space. Experiments on five datasets across chest X-ray and thoracic CT modalities demonstrate that TGC-Net achieves state-of-the-art performance with substantially fewer trainable parameters, including notable Dice gains on challenging benchmarks.
>
---
#### [new 004] OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective
- **分类: cs.CV**

- **简介: 提出OccuFly，首个基于相机的空中语义场景补全基准，解决无人机因无LiDAR导致的3D感知难题，支持多季节多场景，推动高空视角3D理解研究。**

- **链接: [https://arxiv.org/pdf/2512.20770v1](https://arxiv.org/pdf/2512.20770v1)**

> **作者:** Markus Gross; Sai B. Matha; Aya Fahmy; Rui Song; Daniel Cremers; Henri Meess
>
> **摘要:** Semantic Scene Completion (SSC) is crucial for 3D perception in mobile robotics, as it enables holistic scene understanding by jointly estimating dense volumetric occupancy and per-voxel semantics. Although SSC has been widely studied in terrestrial domains such as autonomous driving, aerial scenarios like autonomous flying remain largely unexplored, thereby limiting progress on downstream applications. Furthermore, LiDAR sensors represent the primary modality for SSC data generation, which poses challenges for most uncrewed aerial vehicles (UAVs) due to flight regulations, mass and energy constraints, and the sparsity of LiDAR-based point clouds from elevated viewpoints. To address these limitations, we introduce OccuFly, the first real-world, camera-based aerial SSC benchmark, captured at altitudes of 50m, 40m, and 30m during spring, summer, fall, and winter. OccuFly covers urban, industrial, and rural scenarios, provides 22 semantic classes, and the data format adheres to established conventions to facilitate seamless integration with existing research. Crucially, we propose a LiDAR-free data generation framework based on camera modality, which is ubiquitous on modern UAVs. By utilizing traditional 3D reconstruction, our framework automates label transfer by lifting a subset of annotated 2D masks into the reconstructed point cloud, thereby substantially minimizing manual 3D annotation effort. Finally, we benchmark the state-of-the-art on OccuFly and highlight challenges specific to elevated viewpoints, yielding a comprehensive vision benchmark for holistic aerial 3D scene understanding.
>
---
#### [new 005] VisRes Bench: On Evaluating the Visual Reasoning Capabilities of VLMs
- **分类: cs.CV**

- **简介: 论文提出VisRes Bench评测基准，评估VLM视觉推理能力，揭示其在感知、规则与组合推理上的局限，推动多模态抽象推理研究。**

- **链接: [https://arxiv.org/pdf/2512.21194v1](https://arxiv.org/pdf/2512.21194v1)**

> **作者:** Brigitta Malagurski Törtei; Yasser Dahou; Ngoc Dung Huynh; Wamiq Reyaz Para; Phúc H. Lê Khac; Ankit Singh; Sofian Chaybouti; Sanath Narayan
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress across tasks such as visual question answering and image captioning. Yet, the extent to which these models perform visual reasoning as opposed to relying on linguistic priors remains unclear. To address this, we introduce VisRes Bench, a benchmark designed to study visual reasoning in naturalistic settings without contextual language supervision. Analyzing model behavior across three levels of complexity, we uncover clear limitations in perceptual and relational visual reasoning capacities. VisRes isolates distinct reasoning abilities across its levels. Level 1 probes perceptual completion and global image matching under perturbations such as blur, texture changes, occlusion, and rotation; Level 2 tests rule-based inference over a single attribute (e.g., color, count, orientation); and Level 3 targets compositional reasoning that requires integrating multiple visual attributes. Across more than 19,000 controlled task images, we find that state-of-the-art VLMs perform near random under subtle perceptual perturbations, revealing limited abstraction beyond pattern recognition. We conclude by discussing how VisRes provides a unified framework for advancing abstract visual reasoning in multimodal research.
>
---
#### [new 006] Beyond Artifacts: Real-Centric Envelope Modeling for Reliable AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 论文提出REM方法，通过建模真实图像分布而非生成器伪影，提升AI生成图在真实退化场景下的检测鲁棒性，并构建RealChain评测基准。**

- **链接: [https://arxiv.org/pdf/2512.20937v1](https://arxiv.org/pdf/2512.20937v1)**

> **作者:** Ruiqi Liu; Yi Han; Zhengbo Zhang; Liwei Yao; Zhiyuan Yan; Jialiang Shen; ZhiJin Chen; Boyi Sun; Lubin Weng; Jing Dong; Yan Wang; Shu Wu
>
> **摘要:** The rapid progress of generative models has intensified the need for reliable and robust detection under real-world conditions. However, existing detectors often overfit to generator-specific artifacts and remain highly sensitive to real-world degradations. As generative architectures evolve and images undergo multi-round cross-platform sharing and post-processing (chain degradations), these artifact cues become obsolete and harder to detect. To address this, we propose Real-centric Envelope Modeling (REM), a new paradigm that shifts detection from learning generator artifacts to modeling the robust distribution of real images. REM introduces feature-level perturbations in self-reconstruction to generate near-real samples, and employs an envelope estimator with cross-domain consistency to learn a boundary enclosing the real image manifold. We further build RealChain, a comprehensive benchmark covering both open-source and commercial generators with simulated real-world degradation. Across eight benchmark evaluations, REM achieves an average improvement of 7.5% over state-of-the-art methods, and notably maintains exceptional generalization on the severely degraded RealChain benchmark, establishing a solid foundation for synthetic image detection under real-world conditions. The code and the RealChain benchmark will be made publicly available upon acceptance of the paper.
>
---
#### [new 007] Beyond Weight Adaptation: Feature-Space Domain Injection for Cross-Modal Ship Re-Identification
- **分类: cs.CV**

- **简介: 论文提出DRI方法，通过特征空间注入适配跨模态船舶重识别，解决模态差异问题，在冻结预训练模型下实现高效微调，达SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.20892v1](https://arxiv.org/pdf/2512.20892v1)**

> **作者:** Tingfeng Xian; Wenlve Zhou; Zhiheng Zhou; Zhelin Li
>
> **摘要:** Cross-Modality Ship Re-Identification (CMS Re-ID) is critical for achieving all-day and all-weather maritime target tracking, yet it is fundamentally challenged by significant modality discrepancies. Mainstream solutions typically rely on explicit modality alignment strategies; however, this paradigm heavily depends on constructing large-scale paired datasets for pre-training. To address this, grounded in the Platonic Representation Hypothesis, we explore the potential of Vision Foundation Models (VFMs) in bridging modality gaps. Recognizing the suboptimal performance of existing generic Parameter-Efficient Fine-Tuning (PEFT) methods that operate within the weight space, particularly on limited-capacity models, we shift the optimization perspective to the feature space and propose a novel PEFT strategy termed Domain Representation Injection (DRI). Specifically, while keeping the VFM fully frozen to maximize the preservation of general knowledge, we design a lightweight, learnable Offset Encoder to extract domain-specific representations rich in modality and identity attributes from raw inputs. Guided by the contextual information of intermediate features at different layers, a Modulator adaptively transforms these representations. Subsequently, they are injected into the intermediate layers via additive fusion, dynamically reshaping the feature distribution to adapt to the downstream task without altering the VFM's pre-trained weights. Extensive experimental results demonstrate the superiority of our method, achieving State-of-the-Art (SOTA) performance with minimal trainable parameters. For instance, on the HOSS-ReID dataset, we attain 57.9\% and 60.5\% mAP using only 1.54M and 7.05M parameters, respectively. The code is available at https://github.com/TingfengXian/DRI.
>
---
#### [new 008] Streaming Video Instruction Tuning
- **分类: cs.CV**

- **简介: 提出Streamo模型，专攻流式视频实时多任务理解，构建465K指令数据集，实现统一训练与强时序推理，迈向智能连续视频理解。**

- **链接: [https://arxiv.org/pdf/2512.21334v1](https://arxiv.org/pdf/2512.21334v1)**

> **作者:** Jiaer Xia; Peixian Chen; Mengdan Zhang; Xing Sun; Kaiyang Zhou
>
> **摘要:** We present Streamo, a real-time streaming video LLM that serves as a general-purpose interactive assistant. Unlike existing online video models that focus narrowly on question answering or captioning, Streamo performs a broad spectrum of streaming video tasks, including real-time narration, action understanding, event captioning, temporal event grounding, and time-sensitive question answering. To develop such versatility, we construct Streamo-Instruct-465K, a large-scale instruction-following dataset tailored for streaming video understanding. The dataset covers diverse temporal contexts and multi-task supervision, enabling unified training across heterogeneous streaming tasks. After training end-to-end on the instruction-following dataset through a streamlined pipeline, Streamo exhibits strong temporal reasoning, responsive interaction, and broad generalization across a variety of streaming benchmarks. Extensive experiments show that Streamo bridges the gap between offline video perception models and real-time multimodal assistants, making a step toward unified, intelligent video understanding in continuous video streams.
>
---
#### [new 009] SPOT!: Map-Guided LLM Agent for Unsupervised Multi-CCTV Dynamic Object Tracking
- **分类: cs.CV**

- **简介: 提出SPOT系统，用地图引导LLM代理无监督追踪多摄像头盲区车辆，解决轨迹断裂问题，提升跨摄像头连续跟踪精度。**

- **链接: [https://arxiv.org/pdf/2512.20975v1](https://arxiv.org/pdf/2512.20975v1)**

> **作者:** Yujin Noh; Inho Jake Park; Chigon Hwang
>
> **备注:** 33 pages, 27figures
>
> **摘要:** CCTV-based vehicle tracking systems face structural limitations in continuously connecting the trajectories of the same vehicle across multiple camera environments. In particular, blind spots occur due to the intervals between CCTVs and limited Fields of View (FOV), which leads to object ID switching and trajectory loss, thereby reducing the reliability of real-time path prediction. This paper proposes SPOT (Spatial Prediction Over Trajectories), a map-guided LLM agent capable of tracking vehicles even in blind spots of multi-CCTV environments without prior training. The proposed method represents road structures (Waypoints) and CCTV placement information as documents based on 2D spatial coordinates and organizes them through chunking techniques to enable real-time querying and inference. Furthermore, it transforms the vehicle's position into the actual world coordinate system using the relative position and FOV information of objects observed in CCTV images. By combining map spatial information with the vehicle's moving direction, speed, and driving patterns, a beam search is performed at the intersection level to derive candidate CCTV locations where the vehicle is most likely to enter after the blind spot. Experimental results based on the CARLA simulator in a virtual city environment confirmed that the proposed method accurately predicts the next appearing CCTV even in blind spot sections, maintaining continuous vehicle trajectories more effectively than existing techniques.
>
---
#### [new 010] AndroidLens: Long-latency Evaluation with Nested Sub-targets for Android GUI Agents
- **分类: cs.CV**

- **简介: 提出AndroidLens框架，评估移动GUI智能体长延迟任务能力，解决现有评测局限，涵盖多领域复杂任务，揭示当前模型性能瓶颈与关键挑战。**

- **链接: [https://arxiv.org/pdf/2512.21302v1](https://arxiv.org/pdf/2512.21302v1)**

> **作者:** Yue Cao; Yingyao Wang; Pi Bu; Jingxuan Xing; Wei Jiang; Zekun Zhu; Junpeng Ma; Sashuai Zhou; Tong Lu; Jun Song; Yu Cheng; Yuning Jiang; Bo Zheng
>
> **备注:** 23 pages, 13 figures, 8 tables
>
> **摘要:** Graphical user interface (GUI) agents can substantially improve productivity by automating frequently executed long-latency tasks on mobile devices. However, existing evaluation benchmarks are still constrained to limited applications, simple tasks, and coarse-grained metrics. To address this, we introduce AndroidLens, a challenging evaluation framework for mobile GUI agents, comprising 571 long-latency tasks in both Chinese and English environments, each requiring an average of more than 26 steps to complete. The framework features: (1) tasks derived from real-world user scenarios across 38 domains, covering complex types such as multi-constraint, multi-goal, and domain-specific tasks; (2) static evaluation that preserves real-world anomalies and allows multiple valid paths to reduce bias; and (3) dynamic evaluation that employs a milestone-based scheme for fine-grained progress measurement via Average Task Progress (ATP). Our evaluation indicates that even the best models reach only a 12.7% task success rate and 50.47% ATP. We also underscore key challenges in real-world environments, including environmental anomalies, adaptive exploration, and long-term memory retention.
>
---
#### [new 011] XGrid-Mapping: Explicit Implicit Hybrid Grid Submaps for Efficient Incremental Neural LiDAR Mapping
- **分类: cs.CV**

- **简介: 提出XGrid-Mapping，融合显隐式网格实现高效增量神经LiDAR建图，解决现有方法效率低、结构利用不足问题，提升大规模实时建图质量与一致性。**

- **链接: [https://arxiv.org/pdf/2512.20976v1](https://arxiv.org/pdf/2512.20976v1)**

> **作者:** Zeqing Song; Zhongmiao Yan; Junyuan Deng; Songpengcheng Xia; Xiang Mu; Jingyi Xu; Qi Wu; Ling Pei
>
> **摘要:** Large-scale incremental mapping is fundamental to the development of robust and reliable autonomous systems, as it underpins incremental environmental understanding with sequential inputs for navigation and decision-making. LiDAR is widely used for this purpose due to its accuracy and robustness. Recently, neural LiDAR mapping has shown impressive performance; however, most approaches rely on dense implicit representations and underutilize geometric structure, while existing voxel-guided methods struggle to achieve real-time performance. To address these challenges, we propose XGrid-Mapping, a hybrid grid framework that jointly exploits explicit and implicit representations for efficient neural LiDAR mapping. Specifically, the strategy combines a sparse grid, providing geometric priors and structural guidance, with an implicit dense grid that enriches scene representation. By coupling the VDB structure with a submap-based organization, the framework reduces computational load and enables efficient incremental mapping on a large scale. To mitigate discontinuities across submaps, we introduce a distillation-based overlap alignment strategy, in which preceding submaps supervise subsequent ones to ensure consistency in overlapping regions. To further enhance robustness and sampling efficiency, we incorporate a dynamic removal module. Extensive experiments show that our approach delivers superior mapping quality while overcoming the efficiency limitations of voxel-guided methods, thereby outperforming existing state-of-the-art mapping methods.
>
---
#### [new 012] Multi-Attribute guided Thermal Face Image Translation based on Latent Diffusion Model
- **分类: cs.CV**

- **简介: 论文提出基于潜扩散模型的多属性引导方法，解决红外人脸转可见光时的身份特征丢失问题，提升异质人脸识别性能。**

- **链接: [https://arxiv.org/pdf/2512.21032v1](https://arxiv.org/pdf/2512.21032v1)**

> **作者:** Mingshu Cai; Osamu Yoshie; Yuya Ieiri
>
> **备注:** Accepted by 2025 IEEE International Joint Conference on Biometrics (IJCB 2025)
>
> **摘要:** Modern surveillance systems increasingly rely on multi-wavelength sensors and deep neural networks to recognize faces in infrared images captured at night. However, most facial recognition models are trained on visible light datasets, leading to substantial performance degradation on infrared inputs due to significant domain shifts. Early feature-based methods for infrared face recognition proved ineffective, prompting researchers to adopt generative approaches that convert infrared images into visible light images for improved recognition. This paradigm, known as Heterogeneous Face Recognition (HFR), faces challenges such as model and modality discrepancies, leading to distortion and feature loss in generated images. To address these limitations, this paper introduces a novel latent diffusion-based model designed to generate high-quality visible face images from thermal inputs while preserving critical identity features. A multi-attribute classifier is incorporated to extract key facial attributes from visible images, mitigating feature loss during infrared-to-visible image restoration. Additionally, we propose the Self-attn Mamba module, which enhances global modeling of cross-modal features and significantly improves inference speed. Experimental results on two benchmark datasets demonstrate the superiority of our approach, achieving state-of-the-art performance in both image quality and identity preservation.
>
---
#### [new 013] DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **简介: 提出DexAvatar框架，从单目手语视频重建精准3D手部与身体姿态，解决现有方法因遮挡和噪声导致的精度低问题，在SGNify数据集上提升35.11%。**

- **链接: [https://arxiv.org/pdf/2512.21054v1](https://arxiv.org/pdf/2512.21054v1)**

> **作者:** Kaustubh Kundu; Hrishav Bakul Barua; Lucy Robertson-Bell; Zhixi Cai; Kalin Stefanov
>
> **备注:** Accepted in WACV 2026
>
> **摘要:** The trend in sign language generation is centered around data-driven generative methods that require vast amounts of precise 2D and 3D human pose data to achieve an acceptable generation quality. However, currently, most sign language datasets are video-based and limited to automatically reconstructed 2D human poses (i.e., keypoints) and lack accurate 3D information. Furthermore, existing state-of-the-art for automatic 3D human pose estimation from sign language videos is prone to self-occlusion, noise, and motion blur effects, resulting in poor reconstruction quality. In response to this, we introduce DexAvatar, a novel framework to reconstruct bio-mechanically accurate fine-grained hand articulations and body movements from in-the-wild monocular sign language videos, guided by learned 3D hand and body priors. DexAvatar achieves strong performance in the SGNify motion capture dataset, the only benchmark available for this task, reaching an improvement of 35.11% in the estimation of body and hand poses compared to the state-of-the-art. The official website of this work is: https://github.com/kaustesseract/DexAvatar.
>
---
#### [new 014] CHAMMI-75: pre-training multi-channel models with heterogeneous microscopy images
- **分类: cs.CV; cs.LG**

- **简介: 提出CHAMMI-75数据集，解决多通道显微图像模型泛化差问题，训练通道自适应模型，提升跨实验细胞形态量化性能。**

- **链接: [https://arxiv.org/pdf/2512.20833v1](https://arxiv.org/pdf/2512.20833v1)**

> **作者:** Vidit Agrawal; John Peters; Tyler N. Thompson; Mohammad Vali Sanian; Chau Pham; Nikita Moshkov; Arshad Kazi; Aditya Pillai; Jack Freeman; Byunguk Kang; Samouil L. Farhi; Ernest Fraenkel; Ron Stewart; Lassi Paavolainen; Bryan A. Plummer; Juan C. Caicedo
>
> **备注:** 47 Pages, 23 Figures, 26 Tables
>
> **摘要:** Quantifying cell morphology using images and machine learning has proven to be a powerful tool to study the response of cells to treatments. However, models used to quantify cellular morphology are typically trained with a single microscopy imaging type. This results in specialized models that cannot be reused across biological studies because the technical specifications do not match (e.g., different number of channels), or because the target experimental conditions are out of distribution. Here, we present CHAMMI-75, an open access dataset of heterogeneous, multi-channel microscopy images from 75 diverse biological studies. We curated this resource from publicly available sources to investigate cellular morphology models that are channel-adaptive and can process any microscopy image type. Our experiments show that training with CHAMMI-75 can improve performance in multi-channel bioimaging tasks primarily because of its high diversity in microscopy modalities. This work paves the way to create the next generation of cellular morphology models for biological studies.
>
---
#### [new 015] X-ray Insights Unleashed: Pioneering the Enhancement of Multi-Label Long-Tail Data
- **分类: cs.CV**

- **简介: 论文提出用正常X光片训练扩散模型，通过修复头部病变保留尾部病变数据，结合LLM知识引导与渐进学习，提升长尾肺部异常诊断性能。**

- **链接: [https://arxiv.org/pdf/2512.20980v1](https://arxiv.org/pdf/2512.20980v1)**

> **作者:** Xinquan Yang; Jinheng Xie; Yawen Huang; Yuexiang Li; Huimin Huang; Hao Zheng; Xian Wu; Yefeng Zheng; Linlin Shen
>
> **摘要:** Long-tailed pulmonary anomalies in chest radiography present formidable diagnostic challenges. Despite the recent strides in diffusion-based methods for enhancing the representation of tailed lesions, the paucity of rare lesion exemplars curtails the generative capabilities of these approaches, thereby leaving the diagnostic precision less than optimal. In this paper, we propose a novel data synthesis pipeline designed to augment tail lesions utilizing a copious supply of conventional normal X-rays. Specifically, a sufficient quantity of normal samples is amassed to train a diffusion model capable of generating normal X-ray images. This pre-trained diffusion model is subsequently utilized to inpaint the head lesions present in the diseased X-rays, thereby preserving the tail classes as augmented training data. Additionally, we propose the integration of a Large Language Model Knowledge Guidance (LKG) module alongside a Progressive Incremental Learning (PIL) strategy to stabilize the inpainting fine-tuning process. Comprehensive evaluations conducted on the public lung datasets MIMIC and CheXpert demonstrate that the proposed method sets a new benchmark in performance.
>
---
#### [new 016] MVInverse: Feed-forward Multi-view Inverse Rendering in Seconds
- **分类: cs.CV**

- **简介: 提出MVInverse，首个前馈式多视角逆渲染框架，秒级预测材质与光照，通过跨视角注意力和无标签视频微调，解决一致性差与泛化弱问题。**

- **链接: [https://arxiv.org/pdf/2512.21003v1](https://arxiv.org/pdf/2512.21003v1)**

> **作者:** Xiangzuo Wu; Chengwei Ren; Jun Zhou; Xiu Li; Yuan Liu
>
> **备注:** 21 pages, 17 figures, 5 tables
>
> **摘要:** Multi-view inverse rendering aims to recover geometry, materials, and illumination consistently across multiple viewpoints. When applied to multi-view images, existing single-view approaches often ignore cross-view relationships, leading to inconsistent results. In contrast, multi-view optimization methods rely on slow differentiable rendering and per-scene refinement, making them computationally expensive and hard to scale. To address these limitations, we introduce a feed-forward multi-view inverse rendering framework that directly predicts spatially varying albedo, metallic, roughness, diffuse shading, and surface normals from sequences of RGB images. By alternating attention across views, our model captures both intra-view long-range lighting interactions and inter-view material consistency, enabling coherent scene-level reasoning within a single forward pass. Due to the scarcity of real-world training data, models trained on existing synthetic datasets often struggle to generalize to real-world scenes. To overcome this limitation, we propose a consistency-based finetuning strategy that leverages unlabeled real-world videos to enhance both multi-view coherence and robustness under in-the-wild conditions. Extensive experiments on benchmark datasets demonstrate that our method achieves state-of-the-art performance in terms of multi-view consistency, material and normal estimation quality, and generalization to real-world imagery.
>
---
#### [new 017] Reasoning-Driven Amodal Completion: Collaborative Agents and Perceptual Evaluation
- **分类: cs.CV**

- **简介: 论文提出多智能体推理框架解决遮挡物补全的语义与结构一致性问题，引入自校验机制与多样假设生成，并设计新评估指标MAC-Score。**

- **链接: [https://arxiv.org/pdf/2512.20936v1](https://arxiv.org/pdf/2512.20936v1)**

> **作者:** Hongxing Fan; Shuyu Zhao; Jiayang Ao; Lu Sheng
>
> **摘要:** Amodal completion, the task of inferring invisible object parts, faces significant challenges in maintaining semantic consistency and structural integrity. Prior progressive approaches are inherently limited by inference instability and error accumulation. To tackle these limitations, we present a Collaborative Multi-Agent Reasoning Framework that explicitly decouples Semantic Planning from Visual Synthesis. By employing specialized agents for upfront reasoning, our method generates a structured, explicit plan before pixel generation, enabling visually and semantically coherent single-pass synthesis. We integrate this framework with two critical mechanisms: (1) a self-correcting Verification Agent that employs Chain-of-Thought reasoning to rectify visible region segmentation and identify residual occluders strictly within the Semantic Planning phase, and (2) a Diverse Hypothesis Generator that addresses the ambiguity of invisible regions by offering diverse, plausible semantic interpretations, surpassing the limited pixel-level variations of standard random seed sampling. Furthermore, addressing the limitations of traditional metrics in assessing inferred invisible content, we introduce the MAC-Score (MLLM Amodal Completion Score), a novel human-aligned evaluation metric. Validated against human judgment and ground truth, these metrics establish a robust standard for assessing structural completeness and semantic consistency with visible context. Extensive experiments demonstrate that our framework significantly outperforms state-of-the-art methods across multiple datasets. Our project is available at: https://fanhongxing.github.io/remac-page.
>
---
#### [new 018] TrashDet: Iterative Neural Architecture Search for Efficient Waste Detection
- **分类: cs.CV; cs.LG**

- **简介: 论文提出TrashDet，用硬件感知神经架构搜索优化垃圾检测模型，适配TinyML设备，在精度、能耗、延迟上显著优于现有方案。**

- **链接: [https://arxiv.org/pdf/2512.20746v1](https://arxiv.org/pdf/2512.20746v1)**

> **作者:** Tony Tran; Bin Hu
>
> **备注:** 10 pages. The paper has been accepted by the WACV 2026 workshop
>
> **摘要:** This paper addresses trash detection on the TACO dataset under strict TinyML constraints using an iterative hardware-aware neural architecture search framework targeting edge and IoT devices. The proposed method constructs a Once-for-All-style ResDets supernet and performs iterative evolutionary search that alternates between backbone and neck/head optimization, supported by a population passthrough mechanism and an accuracy predictor to reduce search cost and improve stability. This framework yields a family of deployment-ready detectors, termed TrashDets. On a five-class TACO subset (paper, plastic, bottle, can, cigarette), the strongest variant, TrashDet-l, achieves 19.5 mAP50 with 30.5M parameters, improving accuracy by up to 3.6 mAP50 over prior detectors while using substantially fewer parameters. The TrashDet family spans 1.2M to 30.5M parameters with mAP50 values between 11.4 and 19.5, providing scalable detector options for diverse TinyML deployment budgets on resource-constrained hardware. On the MAX78002 microcontroller with the TrashNet dataset, two specialized variants, TrashDet-ResNet and TrashDet-MBNet, jointly dominate the ai87-fpndetector baseline, with TrashDet-ResNet achieving 7525~$μ$J energy per inference at 26.7 ms latency and 37.45 FPS, and TrashDet-MBNet improving mAP50 by 10.2%; together they reduce energy consumption by up to 88%, latency by up to 78%, and average power by up to 53% compared to existing TinyML detectors.
>
---
#### [new 019] VL4Gaze: Unleashing Vision-Language Models for Gaze Following
- **分类: cs.CV**

- **简介: 提出VL4Gaze基准，首次系统评估并训练视觉语言模型理解人类视线，解决其在注视对象、方向、定位等任务上的能力缺失问题。**

- **链接: [https://arxiv.org/pdf/2512.20735v1](https://arxiv.org/pdf/2512.20735v1)**

> **作者:** Shijing Wang; Chaoqun Cui; Yaping Huang; Hyung Jin Chang; Yihua Cheng
>
> **摘要:** Human gaze provides essential cues for interpreting attention, intention, and social interaction in visual scenes, yet gaze understanding remains largely unexplored in current vision-language models (VLMs). While recent VLMs achieve strong scene-level reasoning across a range of visual tasks, there exists no benchmark that systematically evaluates or trains them for gaze interpretation, leaving open the question of whether gaze understanding can emerge from general-purpose vision-language pre-training. To address this gap, we introduce VL4Gaze, the first large-scale benchmark designed to investigate, evaluate, and unlock the potential of VLMs for gaze understanding. VL4Gaze contains 489K automatically generated question-answer pairs across 124K images and formulates gaze understanding as a unified VQA problem through four complementary tasks: (1) gaze object description, (2) gaze direction description, (3) gaze point location, and (4) ambiguous question recognition. We comprehensively evaluate both commercial and open-source VLMs under in-context learning and fine-tuning settings. The results show that even large-scale VLMs struggle to reliably infer gaze semantics and spatial localization without task-specific supervision. In contrast, training on VL4Gaze brings substantial and consistent improvements across all tasks, highlighting the importance of targeted multi-task supervision for developing gaze understanding capabilities in VLMs. We will release the dataset and code to support further research and development in this direction.
>
---
#### [new 020] A Turn Toward Better Alignment: Few-Shot Generative Adaptation with Equivariant Feature Rotation
- **分类: cs.CV**

- **简介: 论文提出EFR方法，通过李群旋转对齐源域与目标域特征，解决少样本图像生成中分布结构差异问题，提升跨域生成效果。**

- **链接: [https://arxiv.org/pdf/2512.21174v1](https://arxiv.org/pdf/2512.21174v1)**

> **作者:** Chenghao Xu; Qi Liu; Jiexi Yan; Muli Yang; Cheng Deng
>
> **摘要:** Few-shot image generation aims to effectively adapt a source generative model to a target domain using very few training images. Most existing approaches introduce consistency constraints-typically through instance-level or distribution-level loss functions-to directly align the distribution patterns of source and target domains within their respective latent spaces. However, these strategies often fall short: overly strict constraints can amplify the negative effects of the domain gap, leading to distorted or uninformative content, while overly relaxed constraints may fail to leverage the source domain effectively. This limitation primarily stems from the inherent discrepancy in the underlying distribution structures of the source and target domains. The scarcity of target samples further compounds this issue by hindering accurate estimation of the target domain's distribution. To overcome these limitations, we propose Equivariant Feature Rotation (EFR), a novel adaptation strategy that aligns source and target domains at two complementary levels within a self-rotated proxy feature space. Specifically, we perform adaptive rotations within a parameterized Lie Group to transform both source and target features into an equivariant proxy space, where alignment is conducted. These learnable rotation matrices serve to bridge the domain gap by preserving intra-domain structural information without distortion, while the alignment optimization facilitates effective knowledge transfer from the source to the target domain. Comprehensive experiments on a variety of commonly used datasets demonstrate that our method significantly enhances the generative performance within the targeted domain.
>
---
#### [new 021] Lightweight framework for underground pipeline recognition and spatial localization based on multi-view 2D GPR images
- **分类: cs.CV; cs.AI**

- **简介: 提出轻量级框架，融合多视图2D探地雷达图像，提升地下管线识别与三维定位精度，解决小目标识别弱、多视图关联差问题。**

- **链接: [https://arxiv.org/pdf/2512.20866v1](https://arxiv.org/pdf/2512.20866v1)**

> **作者:** Haotian Lv; Chao Li; Jiangbo Dai; Yuhui Zhang; Zepeng Fan; Yiqiu Tan; Dawei Wang; Binglei Xie
>
> **摘要:** To address the issues of weak correlation between multi-view features, low recognition accuracy of small-scale targets, and insufficient robustness in complex scenarios in underground pipeline detection using 3D GPR, this paper proposes a 3D pipeline intelligent detection framework. First, based on a B/C/D-Scan three-view joint analysis strategy, a three-dimensional pipeline three-view feature evaluation method is established by cross-validating forward simulation results obtained using FDTD methods with actual measurement data. Second, the DCO-YOLO framework is proposed, which integrates DySample, CGLU, and OutlookAttention cross-dimensional correlation mechanisms into the original YOLOv11 algorithm, significantly improving the small-scale pipeline edge feature extraction capability. Furthermore, a 3D-DIoU spatial feature matching algorithm is proposed, which integrates three-dimensional geometric constraints and center distance penalty terms to achieve automated association of multi-view annotations. The three-view fusion strategy resolves inherent ambiguities in single-view detection. Experiments based on real urban underground pipeline data show that the proposed method achieves accuracy, recall, and mean average precision of 96.2%, 93.3%, and 96.7%, respectively, in complex multi-pipeline scenarios, which are 2.0%, 2.1%, and 0.9% higher than the baseline model. Ablation experiments validated the synergistic optimization effect of the dynamic feature enhancement module and Grad-CAM++ heatmap visualization demonstrated that the improved model significantly enhanced its ability to focus on pipeline geometric features. This study integrates deep learning optimization strategies with the physical characteristics of 3D GPR, offering an efficient and reliable novel technical framework for the intelligent recognition and localization of underground pipelines.
>
---
#### [new 022] Efficient and Robust Video Defense Framework against 3D-field Personalized Talking Face
- **分类: cs.CV**

- **简介: 提出高效视频防御框架，抵御3D个性化说话脸生成攻击，通过扰动3D信息获取并保持画质，实现47倍加速与强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.21019v1](https://arxiv.org/pdf/2512.21019v1)**

> **作者:** Rui-qing Sun; Xingshan Yao; Tian Lan; Hui-Yang Zhao; Jia-Ling Shi; Chen-Hao Cui; Zhijing Wu; Chen Yang; Xian-Ling Mao
>
> **摘要:** State-of-the-art 3D-field video-referenced Talking Face Generation (TFG) methods synthesize high-fidelity personalized talking-face videos in real time by modeling 3D geometry and appearance from reference portrait video. This capability raises significant privacy concerns regarding malicious misuse of personal portraits. However, no efficient defense framework exists to protect such videos against 3D-field TFG methods. While image-based defenses could apply per-frame 2D perturbations, they incur prohibitive computational costs, severe video quality degradation, failing to disrupt 3D information for video protection. To address this, we propose a novel and efficient video defense framework against 3D-field TFG methods, which protects portrait video by perturbing the 3D information acquisition process while maintain high-fidelity video quality. Specifically, our method introduces: (1) a similarity-guided parameter sharing mechanism for computational efficiency, and (2) a multi-scale dual-domain attention module to jointly optimize spatial-frequency perturbations. Extensive experiments demonstrate that our proposed framework exhibits strong defense capability and achieves a 47x acceleration over the fastest baseline while maintaining high fidelity. Moreover, it remains robust against scaling operations and state-of-the-art purification attacks, and the effectiveness of our design choices is further validated through ablation studies. Our project is available at https://github.com/Richen7418/VDF.
>
---
#### [new 023] Self-supervised Multiplex Consensus Mamba for General Image Fusion
- **分类: cs.CV**

- **简介: 提出SMC-Mamba框架，通过自监督与多模态共识机制，实现通用图像融合，提升下游任务性能，兼顾细节保留与跨模态信息整合。**

- **链接: [https://arxiv.org/pdf/2512.20921v1](https://arxiv.org/pdf/2512.20921v1)**

> **作者:** Yingying Wang; Rongjin Zhuang; Hui Zheng; Xuanhua He; Ke Cao; Xiaotong Tu; Xinghao Ding
>
> **备注:** Accepted by AAAI 2026, 9 pages, 4 figures
>
> **摘要:** Image fusion integrates complementary information from different modalities to generate high-quality fused images, thereby enhancing downstream tasks such as object detection and semantic segmentation. Unlike task-specific techniques that primarily focus on consolidating inter-modal information, general image fusion needs to address a wide range of tasks while improving performance without increasing complexity. To achieve this, we propose SMC-Mamba, a Self-supervised Multiplex Consensus Mamba framework for general image fusion. Specifically, the Modality-Agnostic Feature Enhancement (MAFE) module preserves fine details through adaptive gating and enhances global representations via spatial-channel and frequency-rotational scanning. The Multiplex Consensus Cross-modal Mamba (MCCM) module enables dynamic collaboration among experts, reaching a consensus to efficiently integrate complementary information from multiple modalities. The cross-modal scanning within MCCM further strengthens feature interactions across modalities, facilitating seamless integration of critical information from both sources. Additionally, we introduce a Bi-level Self-supervised Contrastive Learning Loss (BSCL), which preserves high-frequency information without increasing computational overhead while simultaneously boosting performance in downstream tasks. Extensive experiments demonstrate that our approach outperforms state-of-the-art (SOTA) image fusion algorithms in tasks such as infrared-visible, medical, multi-focus, and multi-exposure fusion, as well as downstream visual tasks.
>
---
#### [new 024] Leveraging Lightweight Entity Extraction for Scalable Event-Based Image Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 论文提出轻量级两阶段图像检索方法，结合事件实体提取与BEiT-3模型，提升复杂查询下检索精度与效率，解决语义模糊与扩展性问题。**

- **链接: [https://arxiv.org/pdf/2512.21221v1](https://arxiv.org/pdf/2512.21221v1)**

> **作者:** Dao Sy Duy Minh; Huynh Trung Kiet; Nguyen Lam Phu Quy; Phu-Hoa Pham; Tran Chi Nguyen
>
> **备注:** System description paper for EVENTA Grand Challenge Track 2 at ACM Multimedia 2025 (MM '25). Ranked 4th place. 6 pages, 1 figure, 2 tables
>
> **摘要:** Retrieving images from natural language descriptions is a core task at the intersection of computer vision and natural language processing, with wide-ranging applications in search engines, media archiving, and digital content management. However, real-world image-text retrieval remains challenging due to vague or context-dependent queries, linguistic variability, and the need for scalable solutions. In this work, we propose a lightweight two-stage retrieval pipeline that leverages event-centric entity extraction to incorporate temporal and contextual signals from real-world captions. The first stage performs efficient candidate filtering using BM25 based on salient entities, while the second stage applies BEiT-3 models to capture deep multimodal semantics and rerank the results. Evaluated on the OpenEvents v1 benchmark, our method achieves a mean average precision of 0.559, substantially outperforming prior baselines. These results highlight the effectiveness of combining event-guided filtering with long-text vision-language modeling for accurate and efficient retrieval in complex, real-world scenarios. Our code is available at https://github.com/PhamPhuHoa-23/Event-Based-Image-Retrieval
>
---
#### [new 025] Surgical Scene Segmentation using a Spike-Driven Video Transformer with Real-Time Potential
- **分类: cs.CV**

- **简介: 提出SpikeSurgSeg，首个脉冲驱动视频Transformer，解决手术场景分割实时性难题，兼顾精度与低延迟，适配非GPU平台。**

- **链接: [https://arxiv.org/pdf/2512.21284v1](https://arxiv.org/pdf/2512.21284v1)**

> **作者:** Shihao Zou; Jingjing Li; Wei Ji; Jincai Huang; Kai Wang; Guo Dan; Weixin Si; Yi Pan
>
> **摘要:** Modern surgical systems increasingly rely on intelligent scene understanding to provide timely situational awareness for enhanced intra-operative safety. Within this pipeline, surgical scene segmentation plays a central role in accurately perceiving operative events. Although recent deep learning models, particularly large-scale foundation models, achieve remarkable segmentation accuracy, their substantial computational demands and power consumption hinder real-time deployment in resource-constrained surgical environments. To address this limitation, we explore the emerging SNN as a promising paradigm for highly efficient surgical intelligence. However, their performance is still constrained by the scarcity of labeled surgical data and the inherently sparse nature of surgical video representations. To this end, we propose \textit{SpikeSurgSeg}, the first spike-driven video Transformer framework tailored for surgical scene segmentation with real-time potential on non-GPU platforms. To address the limited availability of surgical annotations, we introduce a surgical-scene masked autoencoding pretraining strategy for SNNs that enables robust spatiotemporal representation learning via layer-wise tube masking. Building on this pretrained backbone, we further adopt a lightweight spike-driven segmentation head that produces temporally consistent predictions while preserving the low-latency characteristics of SNNs. Extensive experiments on EndoVis18 and our in-house SurgBleed dataset demonstrate that SpikeSurgSeg achieves mIoU comparable to SOTA ANN-based models while reducing inference latency by at least $8\times$. Notably, it delivers over $20\times$ acceleration relative to most foundation-model baselines, underscoring its potential for time-critical surgical scene segmentation.
>
---
#### [new 026] Learning to Sense for Driving: Joint Optics-Sensor-Model Co-Design for Semantic Segmentation
- **分类: cs.CV**

- **简介: 论文提出光学-传感器-模型联合设计框架，优化RAW到语义分割的端到端流程，提升自动驾驶感知效率与鲁棒性，尤其在低光和边缘部署场景。**

- **链接: [https://arxiv.org/pdf/2512.20815v1](https://arxiv.org/pdf/2512.20815v1)**

> **作者:** Reeshad Khan amd John Gauch
>
> **摘要:** Traditional autonomous driving pipelines decouple camera design from downstream perception, relying on fixed optics and handcrafted ISPs that prioritize human viewable imagery rather than machine semantics. This separation discards information during demosaicing, denoising, or quantization, while forcing models to adapt to sensor artifacts. We present a task-driven co-design framework that unifies optics, sensor modeling, and lightweight semantic segmentation networks into a single end-to-end RAW-to-task pipeline. Building on DeepLens[19], our system integrates realistic cellphone-scale lens models, learnable color filter arrays, Poisson-Gaussian noise processes, and quantization, all optimized directly for segmentation objectives. Evaluations on KITTI-360 show consistent mIoU improvements over fixed pipelines, with optics modeling and CFA learning providing the largest gains, especially for thin or low-light-sensitive classes. Importantly, these robustness gains are achieved with a compact ~1M-parameter model running at ~28 FPS, demonstrating edge deployability. Visual and quantitative analyses further highlight how co-designed sensors adapt acquisition to semantic structure, sharpening boundaries and maintaining accuracy under blur, noise, and low bit-depth. Together, these findings establish full-stack co-optimization of optics, sensors, and networks as a principled path toward efficient, reliable, and deployable perception in autonomous systems.
>
---
#### [new 027] Post-Processing Mask-Based Table Segmentation for Structural Coordinate Extraction
- **分类: cs.CV**

- **简介: 论文提出多尺度信号处理法，从表掩码中鲁棒提取行列边界，提升低质图像表格结构识别精度，优化下游分析。**

- **链接: [https://arxiv.org/pdf/2512.21287v1](https://arxiv.org/pdf/2512.21287v1)**

> **作者:** Suren Bandara
>
> **摘要:** Structured data extraction from tables plays a crucial role in document image analysis for scanned documents and digital archives. Although many methods have been proposed to detect table structures and extract cell contents, accurately identifying table segment boundaries (rows and columns) remains challenging, particularly in low-resolution or noisy images. In many real-world scenarios, table data are incomplete or degraded, limiting the adaptability of transformer-based methods to noisy inputs. Mask-based edge detection techniques have shown greater robustness under such conditions, as their sensitivity can be adjusted through threshold tuning; however, existing approaches typically apply masks directly to images, leading to noise sensitivity, resolution loss, or high computational cost. This paper proposes a novel multi-scale signal-processing method for detecting table edges from table masks. Row and column transitions are modeled as one-dimensional signals and processed using Gaussian convolution with progressively increasing variances, followed by statistical thresholding to suppress noise while preserving stable structural edges. Detected signal peaks are mapped back to image coordinates to obtain accurate segment boundaries. Experimental results show that applying the proposed approach to column edge detection improves Cell-Aware Segmentation Accuracy (CASA) a layout-aware metric evaluating both textual correctness and correct cell placement from 67% to 76% on the PubLayNet-1M benchmark when using TableNet with PyTesseract OCR. The method is robust to resolution variations through zero-padding and scaling strategies and produces optimized structured tabular outputs suitable for downstream analysis.
>
---
#### [new 028] AnyAD: Unified Any-Modality Anomaly Detection in Incomplete Multi-Sequence MRI
- **分类: cs.CV**

- **简介: 提出AnyAD框架，解决多模态MRI中任意模态缺失下的异常检测问题，通过特征对齐与正常原型引导，实现无需重训的强泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.21264v1](https://arxiv.org/pdf/2512.21264v1)**

> **作者:** Changwei Wu; Yifei Chen; Yuxin Du; Mingxuan Liu; Jinying Zong; Beining Wu; Jie Dong; Feiwei Qin; Yunkang Cao; Qiyuan Tian
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Reliable anomaly detection in brain MRI remains challenging due to the scarcity of annotated abnormal cases and the frequent absence of key imaging modalities in real clinical workflows. Existing single-class or multi-class anomaly detection (AD) models typically rely on fixed modality configurations, require repetitive training, or fail to generalize to unseen modality combinations, limiting their clinical scalability. In this work, we present a unified Any-Modality AD framework that performs robust anomaly detection and localization under arbitrary MRI modality availability. The framework integrates a dual-pathway DINOv2 encoder with a feature distribution alignment mechanism that statistically aligns incomplete-modality features with full-modality representations, enabling stable inference even with severe modality dropout. To further enhance semantic consistency, we introduce an Intrinsic Normal Prototypes (INPs) extractor and an INP-guided decoder that reconstruct only normal anatomical patterns while naturally amplifying abnormal deviations. Through randomized modality masking and indirect feature completion during training, the model learns to adapt to all modality configurations without re-training. Extensive experiments on BraTS2018, MU-Glioma-Post, and Pretreat-MetsToBrain-Masks demonstrate that our approach consistently surpasses state-of-the-art industrial and medical AD baselines across 7 modality combinations, achieving superior generalization. This study establishes a scalable paradigm for multimodal medical AD under real-world, imperfect modality conditions. Our source code is available at https://github.com/wuchangw/AnyAD.
>
---
#### [new 029] ALIVE: An Avatar-Lecture Interactive Video Engine with Content-Aware Retrieval for Real-Time Interaction
- **分类: cs.CV**

- **简介: 提出ALIVE系统，本地化实现讲座视频实时交互，融合语音识别、LLM与神经头像，支持内容感知检索与多模态问答，提升学习体验。**

- **链接: [https://arxiv.org/pdf/2512.20858v1](https://arxiv.org/pdf/2512.20858v1)**

> **作者:** Md Zabirul Islam; Md Motaleb Hossen Manik; Ge Wang
>
> **摘要:** Traditional lecture videos offer flexibility but lack mechanisms for real-time clarification, forcing learners to search externally when confusion arises. Recent advances in large language models and neural avatars provide new opportunities for interactive learning, yet existing systems typically lack lecture awareness, rely on cloud-based services, or fail to integrate retrieval and avatar-delivered explanations in a unified, privacy-preserving pipeline. We present ALIVE, an Avatar-Lecture Interactive Video Engine that transforms passive lecture viewing into a dynamic, real-time learning experience. ALIVE operates fully on local hardware and integrates (1) Avatar-delivered lecture generated through ASR transcription, LLM refinement, and neural talking-head synthesis; (2) A content-aware retrieval mechanism that combines semantic similarity with timestamp alignment to surface contextually relevant lecture segments; and (3) Real-time multimodal interaction, enabling students to pause the lecture, ask questions through text or voice, and receive grounded explanations either as text or as avatar-delivered responses. To maintain responsiveness, ALIVE employs lightweight embedding models, FAISS-based retrieval, and segmented avatar synthesis with progressive preloading. We demonstrate the system on a complete medical imaging course, evaluate its retrieval accuracy, latency characteristics, and user experience, and show that ALIVE provides accurate, content-aware, and engaging real-time support. ALIVE illustrates how multimodal AI-when combined with content-aware retrieval and local deployment-can significantly enhance the pedagogical value of recorded lectures, offering an extensible pathway toward next-generation interactive learning environments.
>
---
#### [new 030] Quantile Rendering: Efficiently Embedding High-dimensional Feature on 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 提出Q-Render方法，高效渲染3D高维特征，解决3D-GS中开放词汇分割的信息损失与速度瓶颈，实现实时高质量分割。**

- **链接: [https://arxiv.org/pdf/2512.20927v1](https://arxiv.org/pdf/2512.20927v1)**

> **作者:** Yoonwoo Jeong; Cheng Sun; Frank Wang; Minsu Cho; Jaesung Choe
>
> **备注:** Will be updated
>
> **摘要:** Recent advancements in computer vision have successfully extended Open-vocabulary segmentation (OVS) to the 3D domain by leveraging 3D Gaussian Splatting (3D-GS). Despite this progress, efficiently rendering the high-dimensional features required for open-vocabulary queries poses a significant challenge. Existing methods employ codebooks or feature compression, causing information loss, thereby degrading segmentation quality. To address this limitation, we introduce Quantile Rendering (Q-Render), a novel rendering strategy for 3D Gaussians that efficiently handles high-dimensional features while maintaining high fidelity. Unlike conventional volume rendering, which densely samples all 3D Gaussians intersecting each ray, Q-Render sparsely samples only those with dominant influence along the ray. By integrating Q-Render into a generalizable 3D neural network, we also propose Gaussian Splatting Network (GS-Net), which predicts Gaussian features in a generalizable manner. Extensive experiments on ScanNet and LeRF demonstrate that our framework outperforms state-of-the-art methods, while enabling real-time rendering with an approximate ~43.7x speedup on 512-D feature maps. Code will be made publicly available.
>
---
#### [new 031] TICON: A Slide-Level Tile Contextualizer for Histopathology Representation Learning
- **分类: cs.CV**

- **简介: 提出TICON，用Transformer为病理切片小块嵌入添加全局上下文，统一不同基础模型输出，提升多种任务性能，仅用11K切片即超越现有大模型。**

- **链接: [https://arxiv.org/pdf/2512.21331v1](https://arxiv.org/pdf/2512.21331v1)**

> **作者:** Varun Belagali; Saarthak Kapse; Pierre Marza; Srijan Das; Zilinghan Li; Sofiène Boutaj; Pushpak Pati; Srikar Yellapragada; Tarak Nath Nandi; Ravi K Madduri; Joel Saltz; Prateek Prasanna; Stergios Christodoulidis Maria Vakalopoulou; Dimitris Samaras
>
> **摘要:** The interpretation of small tiles in large whole slide images (WSI) often needs a larger image context. We introduce TICON, a transformer-based tile representation contextualizer that produces rich, contextualized embeddings for ''any'' application in computational pathology. Standard tile encoder-based pipelines, which extract embeddings of tiles stripped from their context, fail to model the rich slide-level information essential for both local and global tasks. Furthermore, different tile-encoders excel at different downstream tasks. Therefore, a unified model is needed to contextualize embeddings derived from ''any'' tile-level foundation model. TICON addresses this need with a single, shared encoder, pretrained using a masked modeling objective to simultaneously unify and contextualize representations from diverse tile-level pathology foundation models. Our experiments demonstrate that TICON-contextualized embeddings significantly improve performance across many different tasks, establishing new state-of-the-art results on tile-level benchmarks (i.e., HEST-Bench, THUNDER, CATCH) and slide-level benchmarks (i.e., Patho-Bench). Finally, we pretrain an aggregator on TICON to form a slide-level foundation model, using only 11K WSIs, outperforming SoTA slide-level foundation models pretrained with up to 350K WSIs.
>
---
#### [new 032] Hierarchical Modeling Approach to Fast and Accurate Table Recognition
- **分类: cs.CV; cs.LG**

- **简介: 论文提出快速准确的表格识别方法，用非因果注意力建模全局结构，配合并行推理加速内容识别，解决现有模型慢且机理不清问题。**

- **链接: [https://arxiv.org/pdf/2512.21083v1](https://arxiv.org/pdf/2512.21083v1)**

> **作者:** Takaya Kawakatsu
>
> **摘要:** The extraction and use of diverse knowledge from numerous documents is a pressing challenge in intelligent information retrieval. Documents contain elements that require different recognition methods. Table recognition typically consists of three subtasks, namely table structure, cell position and cell content recognition. Recent models have achieved excellent recognition with a combination of multi-task learning, local attention, and mutual learning. However, their effectiveness has not been fully explained, and they require a long period of time for inference. This paper presents a novel multi-task model that utilizes non-causal attention to capture the entire table structure, and a parallel inference algorithm for faster cell content inference. The superiority is demonstrated both visually and statistically on two large public datasets.
>
---
#### [new 033] Transductive Visual Programming: Evolving Tool Libraries from Experience for Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.MA**

- **简介: 提出TVP框架，通过经验自动生成工具库，提升3D空间推理能力，优于现有方法，工具复用率高且泛化性强。**

- **链接: [https://arxiv.org/pdf/2512.20934v1](https://arxiv.org/pdf/2512.20934v1)**

> **作者:** Shengguang Wu; Xiaohan Wang; Yuhui Zhang; Hao Zhu; Serena Yeung-Levy
>
> **备注:** Project Website: https://transductive-visualprogram.github.io/
>
> **摘要:** Spatial reasoning in 3D scenes requires precise geometric calculations that challenge vision-language models. Visual programming addresses this by decomposing problems into steps calling specialized tools, yet existing methods rely on either fixed toolsets or speculative tool induction before solving problems, resulting in suboptimal programs and poor utilization of induced tools. We present Transductive Visual Programming (TVP), a novel framework that builds new tools from its own experience rather than speculation. TVP first solves problems using basic tools while accumulating experiential solutions into an Example Library, then abstracts recurring patterns from these programs into reusable higher-level tools for an evolving Tool Library. This allows TVP to tackle new problems with increasingly powerful tools learned from experience. On Omni3D-Bench, TVP achieves state-of-the-art performance, outperforming GPT-4o by 22% and the previous best visual programming system by 11%. Our transductively learned tools are used 5x more frequently as core program dependency than inductively created ones, demonstrating more effective tool discovery and reuse. The evolved tools also show strong generalization to unseen spatial tasks, achieving superior performance on benchmarks from SpatialScore-Hard collection without any testset-specific modification. Our work establishes experience-driven transductive tool creation as a powerful paradigm for building self-evolving visual programming agents that effectively tackle challenging spatial reasoning tasks. We release our code at https://transductive-visualprogram.github.io/.
>
---
#### [new 034] DGSAN: Dual-Graph Spatiotemporal Attention Network for Pulmonary Nodule Malignancy Prediction
- **分类: cs.CV; cs.AI**

- **简介: 论文提出DGSAN网络，融合多模态时序数据，通过双图注意力机制提升肺结节良恶性预测精度，解决现有融合方法低效问题。**

- **链接: [https://arxiv.org/pdf/2512.20898v1](https://arxiv.org/pdf/2512.20898v1)**

> **作者:** Xiao Yu; Zhaojie Fang; Guanyu Zhou; Yin Shen; Huoling Luo; Ye Li; Ahmed Elazab; Xiang Wan; Ruiquan Ge; Changmiao Wang
>
> **摘要:** Lung cancer continues to be the leading cause of cancer-related deaths globally. Early detection and diagnosis of pulmonary nodules are essential for improving patient survival rates. Although previous research has integrated multimodal and multi-temporal information, outperforming single modality and single time point, the fusion methods are limited to inefficient vector concatenation and simple mutual attention, highlighting the need for more effective multimodal information fusion. To address these challenges, we introduce a Dual-Graph Spatiotemporal Attention Network, which leverages temporal variations and multimodal data to enhance the accuracy of predictions. Our methodology involves developing a Global-Local Feature Encoder to better capture the local, global, and fused characteristics of pulmonary nodules. Additionally, a Dual-Graph Construction method organizes multimodal features into inter-modal and intra-modal graphs. Furthermore, a Hierarchical Cross-Modal Graph Fusion Module is introduced to refine feature integration. We also compiled a novel multimodal dataset named the NLST-cmst dataset as a comprehensive source of support for related research. Our extensive experiments, conducted on both the NLST-cmst and curated CSTL-derived datasets, demonstrate that our DGSAN significantly outperforms state-of-the-art methods in classifying pulmonary nodules with exceptional computational efficiency.
>
---
#### [new 035] Towards Arbitrary Motion Completing via Hierarchical Continuous Representation
- **分类: cs.CV**

- **简介: 论文提出NAME框架，用分层连续隐式表示建模人体运动，支持任意帧率插值/外推，解决运动序列连续性重建问题。**

- **链接: [https://arxiv.org/pdf/2512.21183v1](https://arxiv.org/pdf/2512.21183v1)**

> **作者:** Chenghao Xu; Guangtao Lyu; Qi Liu; Jiexi Yan; Muli Yang; Cheng Deng
>
> **摘要:** Physical motions are inherently continuous, and higher camera frame rates typically contribute to improved smoothness and temporal coherence. For the first time, we explore continuous representations of human motion sequences, featuring the ability to interpolate, inbetween, and even extrapolate any input motion sequences at arbitrary frame rates. To achieve this, we propose a novel parametric activation-induced hierarchical implicit representation framework, referred to as NAME, based on Implicit Neural Representations (INRs). Our method introduces a hierarchical temporal encoding mechanism that extracts features from motion sequences at multiple temporal scales, enabling effective capture of intricate temporal patterns. Additionally, we integrate a custom parametric activation function, powered by Fourier transformations, into the MLP-based decoder to enhance the expressiveness of the continuous representation. This parametric formulation significantly augments the model's ability to represent complex motion behaviors with high accuracy. Extensive evaluations across several benchmark datasets demonstrate the effectiveness and robustness of our proposed approach.
>
---
#### [new 036] Granular-ball Guided Masking: Structure-aware Data Augmentation
- **分类: cs.CV**

- **简介: 提出GBGM方法，通过粒球计算引导结构感知掩码增强，解决数据不足时模型过拟合问题，提升分类与重建性能，适配CNN与ViT。**

- **链接: [https://arxiv.org/pdf/2512.21011v1](https://arxiv.org/pdf/2512.21011v1)**

> **作者:** Shuyin Xia; Fan Chen; Dawei Dai; Meng Yang; Junwei Han; Xinbo Gao; Guoyin Wang
>
> **摘要:** Deep learning models have achieved remarkable success in computer vision, but they still rely heavily on large-scale labeled data and tend to overfit when data are limited or distributions shift. Data augmentation, particularly mask-based information dropping, can enhance robustness by forcing models to explore complementary cues; however, existing approaches often lack structural awareness and may discard essential semantics. We propose Granular-ball Guided Masking (GBGM), a structure-aware augmentation strategy guided by Granular-ball Computing (GBC). GBGM adaptively preserves semantically rich, structurally important regions while suppressing redundant areas through a coarse-to-fine hierarchical masking process, producing augmentations that are both representative and discriminative. Extensive experiments on multiple benchmarks demonstrate consistent improvements in classification accuracy and masked image reconstruction, confirming the effectiveness and broad applicability of the proposed method. Simple and model-agnostic, it integrates seamlessly into CNNs and Vision Transformers and provides a new paradigm for structure-aware data augmentation.
>
---
#### [new 037] Benchmarking and Enhancing VLM for Compressed Image Understanding
- **分类: cs.CV**

- **简介: 论文提出首个压缩图像理解评测基准，分析VLM性能下降原因，并设计通用适配器提升其在多种压缩图像上的表现。**

- **链接: [https://arxiv.org/pdf/2512.20901v1](https://arxiv.org/pdf/2512.20901v1)**

> **作者:** Zifu Zhang; Tongda Xu; Siqi Li; Shengxi Li; Yue Zhang; Mai Xu; Yan Wang
>
> **摘要:** With the rapid development of Vision-Language Models (VLMs) and the growing demand for their applications, efficient compression of the image inputs has become increasingly important. Existing VLMs predominantly digest and understand high-bitrate compressed images, while their ability to interpret low-bitrate compressed images has yet to be explored by far. In this paper, we introduce the first comprehensive benchmark to evaluate the ability of VLM against compressed images, varying existing widely used image codecs and diverse set of tasks, encompassing over one million compressed images in our benchmark. Next, we analyse the source of performance gap, by categorising the gap from a) the information loss during compression and b) generalisation failure of VLM. We visualize these gaps with concrete examples and identify that for compressed images, only the generalization gap can be mitigated. Finally, we propose a universal VLM adaptor to enhance model performance on images compressed by existing codecs. Consequently, we demonstrate that a single adaptor can improve VLM performance across images with varying codecs and bitrates by 10%-30%. We believe that our benchmark and enhancement method provide valuable insights and contribute toward bridging the gap between VLMs and compressed images.
>
---
#### [new 038] Optical Flow-Guided 6DoF Object Pose Tracking with an Event Camera
- **分类: cs.CV**

- **简介: 提出基于事件相机的6DoF物体位姿跟踪方法，利用光流引导2D-3D特征匹配，解决传统相机在动态场景下的跟踪难题，提升精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.21053v1](https://arxiv.org/pdf/2512.21053v1)**

> **作者:** Zibin Liu; Banglei Guan; Yang Shang; Shunkun Liang; Zhenbao Yu; Qifeng Yu
>
> **备注:** 9 pages, 5 figures. In Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)
>
> **摘要:** Object pose tracking is one of the pivotal technologies in multimedia, attracting ever-growing attention in recent years. Existing methods employing traditional cameras encounter numerous challenges such as motion blur, sensor noise, partial occlusion, and changing lighting conditions. The emerging bio-inspired sensors, particularly event cameras, possess advantages such as high dynamic range and low latency, which hold the potential to address the aforementioned challenges. In this work, we present an optical flow-guided 6DoF object pose tracking method with an event camera. A 2D-3D hybrid feature extraction strategy is firstly utilized to detect corners and edges from events and object models, which characterizes object motion precisely. Then, we search for the optical flow of corners by maximizing the event-associated probability within a spatio-temporal window, and establish the correlation between corners and edges guided by optical flow. Furthermore, by minimizing the distances between corners and edges, the 6DoF object pose is iteratively optimized to achieve continuous pose tracking. Experimental results of both simulated and real events demonstrate that our methods outperform event-based state-of-the-art methods in terms of both accuracy and robustness.
>
---
#### [new 039] Next-Scale Prediction: A Self-Supervised Approach for Real-World Image Denoising
- **分类: cs.CV**

- **简介: 提出Next-Scale Prediction自监督方法，解耦噪声去相关与细节保留，提升真实图像去噪效果，并支持无需重训的超分。**

- **链接: [https://arxiv.org/pdf/2512.21038v1](https://arxiv.org/pdf/2512.21038v1)**

> **作者:** Yiwen Shan; Haiyu Zhao; Peng Hu; Xi Peng; Yuanbiao Gou
>
> **摘要:** Self-supervised real-world image denoising remains a fundamental challenge, arising from the antagonistic trade-off between decorrelating spatially structured noise and preserving high-frequency details. Existing blind-spot network (BSN) methods rely on pixel-shuffle downsampling (PD) to decorrelate noise, but aggressive downsampling fragments fine structures, while milder downsampling fails to remove correlated noise. To address this, we introduce Next-Scale Prediction (NSP), a novel self-supervised paradigm that decouples noise decorrelation from detail preservation. NSP constructs cross-scale training pairs, where BSN takes low-resolution, fully decorrelated sub-images as input to predict high-resolution targets that retain fine details. As a by-product, NSP naturally supports super-resolution of noisy images without retraining or modification. Extensive experiments demonstrate that NSP achieves state-of-the-art self-supervised denoising performance on real-world benchmarks, significantly alleviating the long-standing conflict between noise decorrelation and detail preservation.
>
---
#### [new 040] ACD: Direct Conditional Control for Video Diffusion Models via Attention Supervision
- **分类: cs.CV**

- **简介: 提出ACD框架，通过注意力监督实现视频扩散模型的直接条件控制，提升条件对齐精度，同时保持时序连贯与画质，解决现有方法控制力不足问题。**

- **链接: [https://arxiv.org/pdf/2512.21268v1](https://arxiv.org/pdf/2512.21268v1)**

> **作者:** Weiqi Li; Zehao Zhang; Liang Lin; Guangrun Wang
>
> **摘要:** Controllability is a fundamental requirement in video synthesis, where accurate alignment with conditioning signals is essential. Existing classifier-free guidance methods typically achieve conditioning indirectly by modeling the joint distribution of data and conditions, which often results in limited controllability over the specified conditions. Classifier-based guidance enforces conditions through an external classifier, but the model may exploit this mechanism to raise the classifier score without genuinely satisfying the intended condition, resulting in adversarial artifacts and limited effective controllability. In this paper, we propose Attention-Conditional Diffusion (ACD), a novel framework for direct conditional control in video diffusion models via attention supervision. By aligning the model's attention maps with external control signals, ACD achieves better controllability. To support this, we introduce a sparse 3D-aware object layout as an efficient conditioning signal, along with a dedicated Layout ControlNet and an automated annotation pipeline for scalable layout integration. Extensive experiments on benchmark video generation datasets demonstrate that ACD delivers superior alignment with conditioning inputs while preserving temporal coherence and visual fidelity, establishing an effective paradigm for conditional video synthesis.
>
---
#### [new 041] Latent Implicit Visual Reasoning
- **分类: cs.CV**

- **简介: 提出无监督视觉推理机制，让多模态模型自适应提取视觉信息，解决其过度依赖文本、难处理纯视觉任务的问题，提升跨任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.21218v1](https://arxiv.org/pdf/2512.21218v1)**

> **作者:** Kelvin Li; Chuyi Shang; Leonid Karlinsky; Rogerio Feris; Trevor Darrell; Roei Herzig
>
> **摘要:** While Large Multimodal Models (LMMs) have made significant progress, they remain largely text-centric, relying on language as their core reasoning modality. As a result, they are limited in their ability to handle reasoning tasks that are predominantly visual. Recent approaches have sought to address this by supervising intermediate visual steps with helper images, depth maps, or image crops. However, these strategies impose restrictive priors on what "useful" visual abstractions look like, add heavy annotation costs, and struggle to generalize across tasks. To address this critical limitation, we propose a task-agnostic mechanism that trains LMMs to discover and use visual reasoning tokens without explicit supervision. These tokens attend globally and re-encode the image in a task-adaptive way, enabling the model to extract relevant visual information without hand-crafted supervision. Our approach outperforms direct fine-tuning and achieves state-of-the-art results on a diverse range of vision-centric tasks -- including those where intermediate abstractions are hard to specify -- while also generalizing to multi-task instruction tuning.
>
---
#### [new 042] NeRV360: Neural Representation for 360-Degree Videos with a Viewport Decoder
- **分类: cs.CV; cs.MM; eess.IV**

- **简介: 提出NeRV360，专为360视频压缩与实时解码设计，仅解码用户视口，降低内存7倍、提速2.5倍，提升画质。**

- **链接: [https://arxiv.org/pdf/2512.20871v1](https://arxiv.org/pdf/2512.20871v1)**

> **作者:** Daichi Arai; Kyohei Unno; Yasuko Sugito; Yuichi Kusakabe
>
> **备注:** 2026 IIEEJ International Conference on Image Electronics and Visual Computing (IEVC)
>
> **摘要:** Implicit neural representations for videos (NeRV) have shown strong potential for video compression. However, applying NeRV to high-resolution 360-degree videos causes high memory usage and slow decoding, making real-time applications impractical. We propose NeRV360, an end-to-end framework that decodes only the user-selected viewport instead of reconstructing the entire panoramic frame. Unlike conventional pipelines, NeRV360 integrates viewport extraction into decoding and introduces a spatial-temporal affine transform module for conditional decoding based on viewpoint and time. Experiments on 6K-resolution videos show that NeRV360 achieves a 7-fold reduction in memory consumption and a 2.5-fold increase in decoding speed compared to HNeRV, a representative prior work, while delivering better image quality in terms of objective metrics.
>
---
#### [new 043] DreaMontage: Arbitrary Frame-Guided One-Shot Video Generation
- **分类: cs.CV**

- **简介: 提出DreaMontage框架，实现任意帧引导的单镜头视频生成，解决现有方法连贯性差问题，通过条件控制、数据优化与分段推理提升视觉流畅性与生成效率。**

- **链接: [https://arxiv.org/pdf/2512.21252v1](https://arxiv.org/pdf/2512.21252v1)**

> **作者:** Jiawei Liu; Junqiao Li; Jiangfan Deng; Gen Li; Siyu Zhou; Zetao Fang; Shanshan Lao; Zengde Deng; Jianing Zhu; Tingting Ma; Jiayi Li; Yunqiu Wang; Qian He; Xinglong Wu
>
> **备注:** Project Page: https://dreamontage.github.io/DreaMontage/
>
> **摘要:** The "one-shot" technique represents a distinct and sophisticated aesthetic in filmmaking. However, its practical realization is often hindered by prohibitive costs and complex real-world constraints. Although emerging video generation models offer a virtual alternative, existing approaches typically rely on naive clip concatenation, which frequently fails to maintain visual smoothness and temporal coherence. In this paper, we introduce DreaMontage, a comprehensive framework designed for arbitrary frame-guided generation, capable of synthesizing seamless, expressive, and long-duration one-shot videos from diverse user-provided inputs. To achieve this, we address the challenge through three primary dimensions. (i) We integrate a lightweight intermediate-conditioning mechanism into the DiT architecture. By employing an Adaptive Tuning strategy that effectively leverages base training data, we unlock robust arbitrary-frame control capabilities. (ii) To enhance visual fidelity and cinematic expressiveness, we curate a high-quality dataset and implement a Visual Expression SFT stage. In addressing critical issues such as subject motion rationality and transition smoothness, we apply a Tailored DPO scheme, which significantly improves the success rate and usability of the generated content. (iii) To facilitate the production of extended sequences, we design a Segment-wise Auto-Regressive (SAR) inference strategy that operates in a memory-efficient manner. Extensive experiments demonstrate that our approach achieves visually striking and seamlessly coherent one-shot effects while maintaining computational efficiency, empowering users to transform fragmented visual materials into vivid, cohesive one-shot cinematic experiences.
>
---
#### [new 044] Input-Adaptive Visual Preprocessing for Efficient Fast Vision-Language Model Inference
- **分类: cs.CV**

- **简介: 提出自适应视觉预处理方法，动态调整图像分辨率与裁剪，减少冗余计算，提升VLM推理效率，无需修改模型或重训练。**

- **链接: [https://arxiv.org/pdf/2512.20839v1](https://arxiv.org/pdf/2512.20839v1)**

> **作者:** Putu Indah Githa Cahyani; Komang David Dananjaya Suartana; Novanto Yudistira
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated strong performance on multimodal reasoning tasks, but their deployment remains challenging due to high inference latency and computational cost, particularly when processing high-resolution visual inputs. While recent architectures such as FastVLM improve efficiency through optimized vision encoders, existing pipelines still rely on static visual preprocessing, leading to redundant computation for visually simple inputs. In this work, we propose an adaptive visual preprocessing method that dynamically adjusts input resolution and spatial coverage based on image content characteristics. The proposed approach combines content-aware image analysis, adaptive resolution selection, and content-aware cropping to reduce visual redundancy prior to vision encoding. Importantly, the method is integrated with FastVLM without modifying its architecture or requiring retraining. We evaluate the proposed method on a subset of the DocVQA dataset in an inference-only setting, focusing on efficiency-oriented metrics. Experimental results show that adaptive preprocessing reduces per-image inference time by over 50\%, lowers mean full generation time, and achieves a consistent reduction of more than 55\% in visual token count compared to the baseline pipeline. These findings demonstrate that input-aware preprocessing is an effective and lightweight strategy for improving deployment-oriented efficiency of vision-language models. To facilitate reproducibility, our implementation is provided as a fork of the FastVLM repository, incorporating the files for the proposed method, and is available at https://github.com/kmdavidds/mlfastlm.
>
---
#### [new 045] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation
- **分类: cs.CV**

- **简介: 提出T2AV-Compass基准，统一评估文本生成音视频模型，解决现有评估碎片化问题，涵盖客观指标与主观评测，揭示当前模型在跨模态对齐与真实性上的不足。**

- **链接: [https://arxiv.org/pdf/2512.21094v1](https://arxiv.org/pdf/2512.21094v1)**

> **作者:** Zhe Cao; Tao Wang; Jiaming Wang; Yanghai Wang; Yuanxing Zhang; Jialu Chen; Miao Deng; Jiahao Wang; Yubin Guo; Chenxi Liao; Yize Zhang; Zhaoxiang Zhang; Jiaheng Liu
>
> **摘要:** Text-to-Audio-Video (T2AV) generation aims to synthesize temporally coherent video and semantically synchronized audio from natural language, yet its evaluation remains fragmented, often relying on unimodal metrics or narrowly scoped benchmarks that fail to capture cross-modal alignment, instruction following, and perceptual realism under complex prompts. To address this limitation, we present T2AV-Compass, a unified benchmark for comprehensive evaluation of T2AV systems, consisting of 500 diverse and complex prompts constructed via a taxonomy-driven pipeline to ensure semantic richness and physical plausibility. Besides, T2AV-Compass introduces a dual-level evaluation framework that integrates objective signal-level metrics for video quality, audio quality, and cross-modal alignment with a subjective MLLM-as-a-Judge protocol for instruction following and realism assessment. Extensive evaluation of 11 representative T2AVsystems reveals that even the strongest models fall substantially short of human-level realism and cross-modal consistency, with persistent failures in audio realism, fine-grained synchronization, instruction following, etc. These results indicate significant improvement room for future models and highlight the value of T2AV-Compass as a challenging and diagnostic testbed for advancing text-to-audio-video generation.
>
---
#### [new 046] Human Motion Estimation with Everyday Wearables
- **分类: cs.CV**

- **简介: 论文提出EveryWear，用日常穿戴设备无标定实现人体运动估计，构建真实数据集Ego-Elec，采用多模态师生框架，解决穿戴性差、成本高、需标定等问题。**

- **链接: [https://arxiv.org/pdf/2512.21209v1](https://arxiv.org/pdf/2512.21209v1)**

> **作者:** Siqi Zhu; Yixuan Li; Junfu Li; Qi Wu; Zan Wang; Haozhe Ma; Wei Liang
>
> **摘要:** While on-body device-based human motion estimation is crucial for applications such as XR interaction, existing methods often suffer from poor wearability, expensive hardware, and cumbersome calibration, which hinder their adoption in daily life. To address these challenges, we present EveryWear, a lightweight and practical human motion capture approach based entirely on everyday wearables: a smartphone, smartwatch, earbuds, and smart glasses equipped with one forward-facing and two downward-facing cameras, requiring no explicit calibration before use. We introduce Ego-Elec, a 9-hour real-world dataset covering 56 daily activities across 17 diverse indoor and outdoor environments, with ground-truth 3D annotations provided by the motion capture (MoCap), to facilitate robust research and benchmarking in this direction. Our approach employs a multimodal teacher-student framework that integrates visual cues from egocentric cameras with inertial signals from consumer devices. By training directly on real-world data rather than synthetic data, our model effectively eliminates the sim-to-real gap that constrains prior work. Experiments demonstrate that our method outperforms baseline models, validating its effectiveness for practical full-body motion estimation.
>
---
#### [new 047] Beyond Memorization: A Multi-Modal Ordinal Regression Benchmark to Expose Popularity Bias in Vision-Language Models
- **分类: cs.CV**

- **简介: 论文揭示视觉语言模型存在流行度偏见，依赖记忆而非泛化理解。构建YearGuessr数据集与新评估指标，推动模型改进。**

- **链接: [https://arxiv.org/pdf/2512.21337v1](https://arxiv.org/pdf/2512.21337v1)**

> **作者:** Li-Zhong Szu-Tu; Ting-Lin Wu; Chia-Jui Chang; He Syu; Yu-Lun Liu
>
> **备注:** Project page: https://sytwu.github.io/BeyondMemo/
>
> **摘要:** We expose a significant popularity bias in state-of-the-art vision-language models (VLMs), which achieve up to 34% higher accuracy on famous buildings compared to ordinary ones, indicating a reliance on memorization over generalizable understanding. To systematically investigate this, we introduce the largest open benchmark for this task: the YearGuessr dataset, a collection of 55,546 building images with multi-modal attributes from 157 countries, annotated with continuous ordinal labels of their construction year (1001-2024), GPS data, and page-view counts as a proxy for popularity. Using this dataset, we frame the construction year prediction task as ordinal regression and introduce popularity-aware interval accuracy metrics to quantify this bias. Our resulting benchmark of 30+ models, including our YearCLIP model, confirms that VLMs excel on popular, memorized items but struggle significantly with unrecognized subjects, exposing a critical flaw in their reasoning capabilities. Project page: https://sytwu.github.io/BeyondMemo/
>
---
#### [new 048] FreeInpaint: Tuning-free Prompt Alignment and Visual Rationality Enhancement in Image Inpainting
- **分类: cs.CV**

- **简介: 论文提出FreeInpaint，解决图像修复中提示对齐与视觉合理性兼顾难题，通过优化扩散潜变量实现免调参、即插即用的高质量修复。**

- **链接: [https://arxiv.org/pdf/2512.21104v1](https://arxiv.org/pdf/2512.21104v1)**

> **作者:** Chao Gong; Dong Li; Yingwei Pan; Jingjing Chen; Ting Yao; Tao Mei
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Text-guided image inpainting endeavors to generate new content within specified regions of images using textual prompts from users. The primary challenge is to accurately align the inpainted areas with the user-provided prompts while maintaining a high degree of visual fidelity. While existing inpainting methods have produced visually convincing results by leveraging the pre-trained text-to-image diffusion models, they still struggle to uphold both prompt alignment and visual rationality simultaneously. In this work, we introduce FreeInpaint, a plug-and-play tuning-free approach that directly optimizes the diffusion latents on the fly during inference to improve the faithfulness of the generated images. Technically, we introduce a prior-guided noise optimization method that steers model attention towards valid inpainting regions by optimizing the initial noise. Furthermore, we meticulously design a composite guidance objective tailored specifically for the inpainting task. This objective efficiently directs the denoising process, enhancing prompt alignment and visual rationality by optimizing intermediate latents at each step. Through extensive experiments involving various inpainting diffusion models and evaluation metrics, we demonstrate the effectiveness and robustness of our proposed FreeInpaint.
>
---
#### [new 049] MarineEval: Assessing the Marine Intelligence of Vision-Language Models
- **分类: cs.CV; cs.DB**

- **简介: 论文提出MarineEval基准，评估现有视觉语言模型在海洋领域专业问答能力，揭示其不足，推动领域专用模型发展。**

- **链接: [https://arxiv.org/pdf/2512.21126v1](https://arxiv.org/pdf/2512.21126v1)**

> **作者:** YuK-Kwan Wong; Tuan-An To; Jipeng Zhang; Ziqiang Zheng; Sai-Kit Yeung
>
> **备注:** Accepted by The IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026
>
> **摘要:** We have witnessed promising progress led by large language models (LLMs) and further vision language models (VLMs) in handling various queries as a general-purpose assistant. VLMs, as a bridge to connect the visual world and language corpus, receive both visual content and various text-only user instructions to generate corresponding responses. Though great success has been achieved by VLMs in various fields, in this work, we ask whether the existing VLMs can act as domain experts, accurately answering marine questions, which require significant domain expertise and address special domain challenges/requirements. To comprehensively evaluate the effectiveness and explore the boundary of existing VLMs, we construct the first large-scale marine VLM dataset and benchmark called MarineEval, with 2,000 image-based question-answering pairs. During our dataset construction, we ensure the diversity and coverage of the constructed data: 7 task dimensions and 20 capacity dimensions. The domain requirements are specially integrated into the data construction and further verified by the corresponding marine domain experts. We comprehensively benchmark 17 existing VLMs on our MarineEval and also investigate the limitations of existing models in answering marine research questions. The experimental results reveal that existing VLMs cannot effectively answer the domain-specific questions, and there is still a large room for further performance improvements. We hope our new benchmark and observations will facilitate future research. Project Page: http://marineeval.hkustvgd.com/
>
---
#### [new 050] SegMo: Segment-aligned Text to 3D Human Motion Generation
- **分类: cs.CV**

- **简介: 提出SegMo框架，通过分段对齐文本与3D人体动作，实现细粒度生成与检索，提升动作生成质量及跨模态对齐效果。**

- **链接: [https://arxiv.org/pdf/2512.21237v1](https://arxiv.org/pdf/2512.21237v1)**

> **作者:** Bowen Dang; Lin Wu; Xiaohang Yang; Zheng Yuan; Zhixiang Chen
>
> **备注:** The IEEE/CVF Winter Conference on Applications of Computer Vision 2026
>
> **摘要:** Generating 3D human motions from textual descriptions is an important research problem with broad applications in video games, virtual reality, and augmented reality. Recent methods align the textual description with human motion at the sequence level, neglecting the internal semantic structure of modalities. However, both motion descriptions and motion sequences can be naturally decomposed into smaller and semantically coherent segments, which can serve as atomic alignment units to achieve finer-grained correspondence. Motivated by this, we propose SegMo, a novel Segment-aligned text-conditioned human Motion generation framework to achieve fine-grained text-motion alignment. Our framework consists of three modules: (1) Text Segment Extraction, which decomposes complex textual descriptions into temporally ordered phrases, each representing a simple atomic action; (2) Motion Segment Extraction, which partitions complete motion sequences into corresponding motion segments; and (3) Fine-grained Text-Motion Alignment, which aligns text and motion segments with contrastive learning. Extensive experiments demonstrate that SegMo improves the strong baseline on two widely used datasets, achieving an improved TOP 1 score of 0.553 on the HumanML3D test set. Moreover, thanks to the learned shared embedding space for text and motion segments, SegMo can also be applied to retrieval-style tasks such as motion grounding and motion-to-text retrieval.
>
---
#### [new 051] UniPR-3D: Towards Universal Visual Place Recognition with Visual Geometry Grounded Transformer
- **分类: cs.CV**

- **简介: 提出UniPR-3D，首个融合多视角3D信息的视觉地点识别架构，利用几何引导Transformer提升跨场景泛化能力，实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.21078v1](https://arxiv.org/pdf/2512.21078v1)**

> **作者:** Tianchen Deng; Xun Chen; Ziming Li; Hongming Shen; Danwei Wang; Javier Civera; Hesheng Wang
>
> **摘要:** Visual Place Recognition (VPR) has been traditionally formulated as a single-image retrieval task. Using multiple views offers clear advantages, yet this setting remains relatively underexplored and existing methods often struggle to generalize across diverse environments. In this work we introduce UniPR-3D, the first VPR architecture that effectively integrates information from multiple views. UniPR-3D builds on a VGGT backbone capable of encoding multi-view 3D representations, which we adapt by designing feature aggregators and fine-tune for the place recognition task. To construct our descriptor, we jointly leverage the 3D tokens and intermediate 2D tokens produced by VGGT. Based on their distinct characteristics, we design dedicated aggregation modules for 2D and 3D features, allowing our descriptor to capture fine-grained texture cues while also reasoning across viewpoints. To further enhance generalization, we incorporate both single- and multi-frame aggregation schemes, along with a variable-length sequence retrieval strategy. Our experiments show that UniPR-3D sets a new state of the art, outperforming both single- and multi-view baselines and highlighting the effectiveness of geometry-grounded tokens for VPR. Our code and models will be made publicly available on Github https://github.com/dtc111111/UniPR-3D.
>
---
#### [new 052] Fast SAM2 with Text-Driven Token Pruning
- **分类: cs.CV**

- **简介: 提出文本引导的视觉token剪枝方法，提升SAM2视频分割效率，降低计算与内存开销，保持精度，适用于实时、资源受限场景。**

- **链接: [https://arxiv.org/pdf/2512.21333v1](https://arxiv.org/pdf/2512.21333v1)**

> **作者:** Avilasha Mandal; Chaoning Zhang; Fachrina Dewi Puspitasari; Xudong Wang; Jiaquan Zhang; Caiyan Qin; Guoqing Wang; Yang Yang; Heng Tao Shen
>
> **备注:** 28 pages, 9 figures
>
> **摘要:** Segment Anything Model 2 (SAM2), a vision foundation model has significantly advanced in prompt-driven video object segmentation, yet their practical deployment remains limited by the high computational and memory cost of processing dense visual tokens across time. The SAM2 pipelines typically propagate all visual tokens produced by the image encoder through downstream temporal reasoning modules, regardless of their relevance to the target object, resulting in reduced scalability due to quadratic memory attention overhead. In this work, we introduce a text-guided token pruning framework that improves inference efficiency by selectively reducing token density prior to temporal propagation, without modifying the underlying segmentation architecture. Operating after visual encoding and before memory based propagation, our method ranks tokens using a lightweight routing mechanism that integrates local visual context, semantic relevance derived from object-centric textual descriptions (either user-provided or automatically generated), and uncertainty cues that help preserve ambiguous or boundary critical regions. By retaining only the most informative tokens for downstream processing, the proposed approach reduces redundant computation while maintaining segmentation fidelity. Extensive experiments across multiple challenging video segmentation benchmarks demonstrate that post-encoder token pruning provides a practical and effective pathway to efficient, prompt-aware video segmentation, achieving up to 42.50 percent faster inference and 37.41 percent lower GPU memory usage compared to the unpruned baseline SAM2, while preserving competitive J and F performance. These results highlight the potential of early token selection to improve the scalability of transformer-based video segmentation systems for real-time and resource-constrained applications.
>
---
#### [new 053] PUFM++: Point Cloud Upsampling via Enhanced Flow Matching
- **分类: cs.CV**

- **简介: 论文提出PUFM++，通过增强流匹配实现点云上采样，解决稀疏、噪声、残缺点云的高质量重建问题，提升几何保真度与下游任务兼容性。**

- **链接: [https://arxiv.org/pdf/2512.20988v1](https://arxiv.org/pdf/2512.20988v1)**

> **作者:** Zhi-Song Liu; Chenhang He; Roland Maier; Andreas Rupp
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** Recent advances in generative modeling have demonstrated strong promise for high-quality point cloud upsampling. In this work, we present PUFM++, an enhanced flow-matching framework for reconstructing dense and accurate point clouds from sparse, noisy, and partial observations. PUFM++ improves flow matching along three key axes: (i) geometric fidelity, (ii) robustness to imperfect input, and (iii) consistency with downstream surface-based tasks. We introduce a two-stage flow-matching strategy that first learns a direct, straight-path flow from sparse inputs to dense targets, and then refines it using noise-perturbed samples to approximate the terminal marginal distribution better. To accelerate and stabilize inference, we propose a data-driven adaptive time scheduler that improves sampling efficiency based on interpolation behavior. We further impose on-manifold constraints during sampling to ensure that generated points remain aligned with the underlying surface. Finally, we incorporate a recurrent interface network~(RIN) to strengthen hierarchical feature interactions and boost reconstruction quality. Extensive experiments on synthetic benchmarks and real-world scans show that PUFM++ sets a new state of the art in point cloud upsampling, delivering superior visual fidelity and quantitative accuracy across a wide range of tasks. Code and pretrained models are publicly available at https://github.com/Holmes-Alan/Enhanced_PUFM.
>
---
#### [new 054] PanoGrounder: Bridging 2D and 3D with Panoramic Scene Representations for VLM-based 3D Visual Grounding
- **分类: cs.CV**

- **简介: 提出PanoGrounder框架，用全景图衔接2D VLM与3D场景，提升3D视觉定位的泛化能力，在ScanRefer等数据集达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.20907v1](https://arxiv.org/pdf/2512.20907v1)**

> **作者:** Seongmin Jung; Seongho Choi; Gunwoo Jeon; Minsu Cho; Jongwoo Lim
>
> **摘要:** 3D Visual Grounding (3DVG) is a critical bridge from vision-language perception to robotics, requiring both language understanding and 3D scene reasoning. Traditional supervised models leverage explicit 3D geometry but exhibit limited generalization, owing to the scarcity of 3D vision-language datasets and the limited reasoning capabilities compared to modern vision-language models (VLMs). We propose PanoGrounder, a generalizable 3DVG framework that couples multi-modal panoramic representation with pretrained 2D VLMs for strong vision-language reasoning. Panoramic renderings, augmented with 3D semantic and geometric features, serve as an intermediate representation between 2D and 3D, and offer two major benefits: (i) they can be directly fed to VLMs with minimal adaptation and (ii) they retain long-range object-to-object relations thanks to their 360-degree field of view. We devise a three-stage pipeline that places a compact set of panoramic viewpoints considering the scene layout and geometry, grounds a text query on each panoramic rendering with a VLM, and fuses per-view predictions into a single 3D bounding box via lifting. Our approach achieves state-of-the-art results on ScanRefer and Nr3D, and demonstrates superior generalization to unseen 3D datasets and text rephrasings.
>
---
#### [new 055] UltraShape 1.0: High-Fidelity 3D Shape Generation via Scalable Geometric Refinement
- **分类: cs.CV; cs.GR**

- **简介: 提出UltraShape 1.0，通过两阶段扩散框架生成高保真3D形状，改进数据预处理并解耦空间定位与细节合成，提升几何质量。**

- **链接: [https://arxiv.org/pdf/2512.21185v1](https://arxiv.org/pdf/2512.21185v1)**

> **作者:** Tanghui Jia; Dongyu Yan; Dehao Hao; Yang Li; Kaiyi Zhang; Xianyi He; Lanjiong Li; Jinnan Chen; Lutao Jiang; Qishen Yin; Long Quan; Ying-Cong Chen; Li Yuan
>
> **备注:** 14 pages, 10 figures, Technical Report,
>
> **摘要:** In this report, we introduce UltraShape 1.0, a scalable 3D diffusion framework for high-fidelity 3D geometry generation. The proposed approach adopts a two-stage generation pipeline: a coarse global structure is first synthesized and then refined to produce detailed, high-quality geometry. To support reliable 3D generation, we develop a comprehensive data processing pipeline that includes a novel watertight processing method and high-quality data filtering. This pipeline improves the geometric quality of publicly available 3D datasets by removing low-quality samples, filling holes, and thickening thin structures, while preserving fine-grained geometric details. To enable fine-grained geometry refinement, we decouple spatial localization from geometric detail synthesis in the diffusion process. We achieve this by performing voxel-based refinement at fixed spatial locations, where voxel queries derived from coarse geometry provide explicit positional anchors encoded via RoPE, allowing the diffusion model to focus on synthesizing local geometric details within a reduced, structured solution space. Our model is trained exclusively on publicly available 3D datasets, achieving strong geometric quality despite limited training resources. Extensive evaluations demonstrate that UltraShape 1.0 performs competitively with existing open-source methods in both data processing quality and geometry generation. All code and trained models will be released to support future research.
>
---
#### [new 056] UniRec-0.1B: Unified Text and Formula Recognition with 0.1B Parameters
- **分类: cs.CV**

- **简介: 提出轻量级UniRec-0.1B模型，统一识别文档中文本与公式，解决大模型计算昂贵问题，通过新数据集、分层监督和语义解耦tokenizer实现高效高精度。**

- **链接: [https://arxiv.org/pdf/2512.21095v1](https://arxiv.org/pdf/2512.21095v1)**

> **作者:** Yongkun Du; Zhineng Chen; Yazhen Xie; Weikang Baiand Hao Feng; Wei Shi; Yuchen Su; Can Huang; Yu-Gang Jiang
>
> **摘要:** Text and formulas constitute the core informational components of many documents. Accurately and efficiently recognizing both is crucial for developing robust and generalizable document parsing systems. Recently, vision-language models (VLMs) have achieved impressive unified recognition of text and formulas. However, they are large-sized and computationally demanding, restricting their usage in many applications. In this paper, we propose UniRec-0.1B, a unified recognition model with only 0.1B parameters. It is capable of performing text and formula recognition at multiple levels, including characters, words, lines, paragraphs, and documents. To implement this task, we first establish UniRec40M, a large-scale dataset comprises 40 million text, formula and their mix samples, enabling the training of a powerful yet lightweight model. Secondly, we identify two challenges when building such a lightweight but unified expert model. They are: structural variability across hierarchies and semantic entanglement between textual and formulaic content. To tackle these, we introduce a hierarchical supervision training that explicitly guides structural comprehension, and a semantic-decoupled tokenizer that separates text and formula representations. Finally, we develop a comprehensive evaluation benchmark covering Chinese and English documents from multiple domains and with multiple levels. Experimental results on this and public benchmarks demonstrate that UniRec-0.1B outperforms both general-purpose VLMs and leading document parsing expert models, while achieving a 2-9$\times$ speedup, validating its effectiveness and efficiency. Codebase and Dataset: https://github.com/Topdu/OpenOCR.
>
---
#### [new 057] ORCA: Object Recognition and Comprehension for Archiving Marine Species
- **分类: cs.CV**

- **简介: 提出ORCA基准，解决海洋物种识别数据少、任务不明确问题，含14K图像与多模态标注，评测18模型，推动海洋视觉理解研究。**

- **链接: [https://arxiv.org/pdf/2512.21150v1](https://arxiv.org/pdf/2512.21150v1)**

> **作者:** Yuk-Kwan Wong; Haixin Liang; Zeyu Ma; Yiwei Chen; Ziqiang Zheng; Rinaldi Gotama; Pascal Sebastian; Lauren D. Sparks; Sai-Kit Yeung
>
> **备注:** Accepted by The IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2026
>
> **摘要:** Marine visual understanding is essential for monitoring and protecting marine ecosystems, enabling automatic and scalable biological surveys. However, progress is hindered by limited training data and the lack of a systematic task formulation that aligns domain-specific marine challenges with well-defined computer vision tasks, thereby limiting effective model application. To address this gap, we present ORCA, a multi-modal benchmark for marine research comprising 14,647 images from 478 species, with 42,217 bounding box annotations and 22,321 expert-verified instance captions. The dataset provides fine-grained visual and textual annotations that capture morphology-oriented attributes across diverse marine species. To catalyze methodological advances, we evaluate 18 state-of-the-art models on three tasks: object detection (closed-set and open-vocabulary), instance captioning, and visual grounding. Results highlight key challenges, including species diversity, morphological overlap, and specialized domain demands, underscoring the difficulty of marine understanding. ORCA thus establishes a comprehensive benchmark to advance research in marine domain. Project Page: http://orca.hkustvgd.com/.
>
---
#### [new 058] Learning from Next-Frame Prediction: Autoregressive Video Modeling Encodes Effective Representations
- **分类: cs.CV**

- **简介: 提出NExT-Vid框架，用自回归掩码下一帧预测预训练视频模型，解决现有方法语义定位不准、生成质量差问题，提升下游分类性能。**

- **链接: [https://arxiv.org/pdf/2512.21004v1](https://arxiv.org/pdf/2512.21004v1)**

> **作者:** Jinghan Li; Yang Jin; Hao Jiang; Yadong Mu; Yang Song; Kun Xu
>
> **摘要:** Recent advances in pretraining general foundation models have significantly improved performance across diverse downstream tasks. While autoregressive (AR) generative models like GPT have revolutionized NLP, most visual generative pretraining methods still rely on BERT-style masked modeling, which often disregards the temporal information essential for video analysis. The few existing autoregressive visual pretraining methods suffer from issues such as inaccurate semantic localization and poor generation quality, leading to poor semantics. In this work, we propose NExT-Vid, a novel autoregressive visual generative pretraining framework that utilizes masked next-frame prediction to jointly model images and videos. NExT-Vid introduces a context-isolated autoregressive predictor to decouple semantic representation from target decoding, and a conditioned flow-matching decoder to enhance generation quality and diversity. Through context-isolated flow-matching pretraining, our approach achieves strong representations. Extensive experiments on large-scale pretrained models demonstrate that our proposed method consistently outperforms previous generative pretraining methods for visual representation learning via attentive probing in downstream classification.
>
---
#### [new 059] GriDiT: Factorized Grid-Based Diffusion for Efficient Long Image Sequence Generation
- **分类: cs.CV**

- **简介: 提出GriDiT方法，通过低分辨序列生成+单帧超分，高效生成长图像序列，解决现有模型效率低、连贯性差问题，显著提升质量与速度。**

- **链接: [https://arxiv.org/pdf/2512.21276v1](https://arxiv.org/pdf/2512.21276v1)**

> **作者:** Snehal Singh Tomar; Alexandros Graikos; Arjun Krishna; Dimitris Samaras; Klaus Mueller
>
> **摘要:** Modern deep learning methods typically treat image sequences as large tensors of sequentially stacked frames. However, is this straightforward representation ideal given the current state-of-the-art (SoTA)? In this work, we address this question in the context of generative models and aim to devise a more effective way of modeling image sequence data. Observing the inefficiencies and bottlenecks of current SoTA image sequence generation methods, we showcase that rather than working with large tensors, we can improve the generation process by factorizing it into first generating the coarse sequence at low resolution and then refining the individual frames at high resolution. We train a generative model solely on grid images comprising subsampled frames. Yet, we learn to generate image sequences, using the strong self-attention mechanism of the Diffusion Transformer (DiT) to capture correlations between frames. In effect, our formulation extends a 2D image generator to operate as a low-resolution 3D image-sequence generator without introducing any architectural modifications. Subsequently, we super-resolve each frame individually to add the sequence-independent high-resolution details. This approach offers several advantages and can overcome key limitations of the SoTA in this domain. Compared to existing image sequence generation models, our method achieves superior synthesis quality and improved coherence across sequences. It also delivers high-fidelity generation of arbitrary-length sequences and increased efficiency in inference time and training data usage. Furthermore, our straightforward formulation enables our method to generalize effectively across diverse data domains, which typically require additional priors and supervision to model in a generative context. Our method consistently outperforms SoTA in quality and inference speed (at least twice-as-fast) across datasets.
>
---
#### [new 060] HiStream: Efficient High-Resolution Video Generation via Redundancy-Eliminated Streaming
- **分类: cs.CV**

- **简介: 论文提出HiStream框架，解决高分辨率视频生成计算瓶颈问题，通过空间、时间、步数三轴压缩冗余，大幅提升推理速度并保持高质量。**

- **链接: [https://arxiv.org/pdf/2512.21338v1](https://arxiv.org/pdf/2512.21338v1)**

> **作者:** Haonan Qiu; Shikun Liu; Zijian Zhou; Zhaochong An; Weiming Ren; Zhiheng Liu; Jonas Schult; Sen He; Shoufa Chen; Yuren Cong; Tao Xiang; Ziwei Liu; Juan-Manuel Perez-Rua
>
> **备注:** Project Page: http://haonanqiu.com/projects/HiStream.html
>
> **摘要:** High-resolution video generation, while crucial for digital media and film, is computationally bottlenecked by the quadratic complexity of diffusion models, making practical inference infeasible. To address this, we introduce HiStream, an efficient autoregressive framework that systematically reduces redundancy across three axes: i) Spatial Compression: denoising at low resolution before refining at high resolution with cached features; ii) Temporal Compression: a chunk-by-chunk strategy with a fixed-size anchor cache, ensuring stable inference speed; and iii) Timestep Compression: applying fewer denoising steps to subsequent, cache-conditioned chunks. On 1080p benchmarks, our primary HiStream model (i+ii) achieves state-of-the-art visual quality while demonstrating up to 76.2x faster denoising compared to the Wan2.1 baseline and negligible quality loss. Our faster variant, HiStream+, applies all three optimizations (i+ii+iii), achieving a 107.5x acceleration over the baseline, offering a compelling trade-off between speed and quality, thereby making high-resolution video generation both practical and scalable.
>
---
#### [new 061] A Large-Depth-Range Layer-Based Hologram Dataset for Machine Learning-Based 3D Computer-Generated Holography
- **分类: cs.CV; physics.optics**

- **简介: 提出KOREATECH-CGH数据集与幅度投影法，解决ML-CGH缺乏高质量大数据问题，提升大景深全息重建质量，支持模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2512.21040v1](https://arxiv.org/pdf/2512.21040v1)**

> **作者:** Jaehong Lee; You Chan No; YoungWoo Kim; Duksu Kim
>
> **摘要:** Machine learning-based computer-generated holography (ML-CGH) has advanced rapidly in recent years, yet progress is constrained by the limited availability of high-quality, large-scale hologram datasets. To address this, we present KOREATECH-CGH, a publicly available dataset comprising 6,000 pairs of RGB-D images and complex holograms across resolutions ranging from 256*256 to 2048*2048, with depth ranges extending to the theoretical limits of the angular spectrum method for wide 3D scene coverage. To improve hologram quality at large depth ranges, we introduce amplitude projection, a post-processing technique that replaces amplitude components of hologram wavefields at each depth layer while preserving phase. This approach enhances reconstruction fidelity, achieving 27.01 dB PSNR and 0.87 SSIM, surpassing a recent optimized silhouette-masking layer-based method by 2.03 dB and 0.04 SSIM, respectively. We further validate the utility of KOREATECH-CGH through experiments on hologram generation and super-resolution using state-of-the-art ML models, confirming its applicability for training and evaluating next-generation ML-CGH systems.
>
---
#### [new 062] FluencyVE: Marrying Temporal-Aware Mamba with Bypass Attention for Video Editing
- **分类: cs.CV**

- **简介: 论文提出FluencyVE，用Mamba替代时序注意力，结合低秩近似与加权平均，高效解决视频编辑中的时序不一致与高计算开销问题。**

- **链接: [https://arxiv.org/pdf/2512.21015v1](https://arxiv.org/pdf/2512.21015v1)**

> **作者:** Mingshu Cai; Yixuan Li; Osamu Yoshie; Yuya Ieiri
>
> **备注:** Accepted by IEEE Transactions on Multimedia (TMM)
>
> **摘要:** Large-scale text-to-image diffusion models have achieved unprecedented success in image generation and editing. However, extending this success to video editing remains challenging. Recent video editing efforts have adapted pretrained text-to-image models by adding temporal attention mechanisms to handle video tasks. Unfortunately, these methods continue to suffer from temporal inconsistency issues and high computational overheads. In this study, we propose FluencyVE, which is a simple yet effective one-shot video editing approach. FluencyVE integrates the linear time-series module, Mamba, into a video editing model based on pretrained Stable Diffusion models, replacing the temporal attention layer. This enables global frame-level attention while reducing the computational costs. In addition, we employ low-rank approximation matrices to replace the query and key weight matrices in the causal attention, and use a weighted averaging technique during training to update the attention scores. This approach significantly preserves the generative power of the text-to-image model while effectively reducing the computational burden. Experiments and analyses demonstrate promising results in editing various attributes, subjects, and locations in real-world videos.
>
---
#### [new 063] Multimodal Skeleton-Based Action Representation Learning via Decomposition and Composition
- **分类: cs.CV**

- **简介: 论文提出分解-组合框架，自监督学习多模态骨架动作表征，平衡效率与性能，解决现有方法计算开销大或效果差问题。**

- **链接: [https://arxiv.org/pdf/2512.21064v1](https://arxiv.org/pdf/2512.21064v1)**

> **作者:** Hongsong Wang; Heng Fei; Bingxuan Dai; Jie Gui
>
> **备注:** Accepted by Machine Intelligence Research (Journal Impact Factor 8.7, 2024)
>
> **摘要:** Multimodal human action understanding is a significant problem in computer vision, with the central challenge being the effective utilization of the complementarity among diverse modalities while maintaining model efficiency. However, most existing methods rely on simple late fusion to enhance performance, which results in substantial computational overhead. Although early fusion with a shared backbone for all modalities is efficient, it struggles to achieve excellent performance. To address the dilemma of balancing efficiency and effectiveness, we introduce a self-supervised multimodal skeleton-based action representation learning framework, named Decomposition and Composition. The Decomposition strategy meticulously decomposes the fused multimodal features into distinct unimodal features, subsequently aligning them with their respective ground truth unimodal counterparts. On the other hand, the Composition strategy integrates multiple unimodal features, leveraging them as self-supervised guidance to enhance the learning of multimodal representations. Extensive experiments on the NTU RGB+D 60, NTU RGB+D 120, and PKU-MMD II datasets demonstrate that the proposed method strikes an excellent balance between computational cost and model performance.
>
---
#### [new 064] Matrix Completion Via Reweighted Logarithmic Norm Minimization
- **分类: cs.CV**

- **简介: 提出重加权对数范数替代核范数，更优逼近矩阵秩，用ADMM求解，在图像修复中优于现有低秩矩阵补全方法。**

- **链接: [https://arxiv.org/pdf/2512.21050v1](https://arxiv.org/pdf/2512.21050v1)**

> **作者:** Zhijie Wang; Liangtian He; Qinghua Zhang; Jifei Miao; Liang-Jian Deng; Jun Liu
>
> **摘要:** Low-rank matrix completion (LRMC) has demonstrated remarkable success in a wide range of applications. To address the NP-hard nature of the rank minimization problem, the nuclear norm is commonly used as a convex and computationally tractable surrogate for the rank function. However, this approach often yields suboptimal solutions due to the excessive shrinkage of singular values. In this letter, we propose a novel reweighted logarithmic norm as a more effective nonconvex surrogate, which provides a closer approximation than many existing alternatives. We efficiently solve the resulting optimization problem by employing the alternating direction method of multipliers (ADMM). Experimental results on image inpainting demonstrate that the proposed method achieves superior performance compared to state-of-the-art LRMC approaches, both in terms of visual quality and quantitative metrics.
>
---
#### [new 065] Does the Data Processing Inequality Reflect Practice? On the Utility of Low-Level Tasks
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 论文探讨低层预处理对分类任务的实际效用，理论证明有限样本下预处理可提升准确率，并通过实验验证其与噪声、数据量等因素的关系。**

- **链接: [https://arxiv.org/pdf/2512.21315v1](https://arxiv.org/pdf/2512.21315v1)**

> **作者:** Roy Turgeman; Tom Tirer
>
> **摘要:** The data processing inequality is an information-theoretic principle stating that the information content of a signal cannot be increased by processing the observations. In particular, it suggests that there is no benefit in enhancing the signal or encoding it before addressing a classification problem. This assertion can be proven to be true for the case of the optimal Bayes classifier. However, in practice, it is common to perform "low-level" tasks before "high-level" downstream tasks despite the overwhelming capabilities of modern deep neural networks. In this paper, we aim to understand when and why low-level processing can be beneficial for classification. We present a comprehensive theoretical study of a binary classification setup, where we consider a classifier that is tightly connected to the optimal Bayes classifier and converges to it as the number of training samples increases. We prove that for any finite number of training samples, there exists a pre-classification processing that improves the classification accuracy. We also explore the effect of class separation, training set size, and class balance on the relative gain from this procedure. We support our theory with an empirical investigation of the theoretical setup. Finally, we conduct an empirical study where we investigate the effect of denoising and encoding on the performance of practical deep classifiers on benchmark datasets. Specifically, we vary the size and class distribution of the training set, and the noise level, and demonstrate trends that are consistent with our theoretical results.
>
---
#### [new 066] Equivariant Multiscale Learned Invertible Reconstruction for Cone Beam CT: From Simulated to Real Data
- **分类: physics.med-ph; cs.CV**

- **简介: 提出LIRE++，一种旋转等变多尺度可逆重建网络，用于提升CBCT图像质量，解决真实数据缺失、内存与速度瓶颈，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.21180v1](https://arxiv.org/pdf/2512.21180v1)**

> **作者:** Nikita Moriakov; Efstratios Gavves; Jonathan H. Mason; Carmen Seller-Oria; Jonas Teuwen; Jan-Jakob Sonke
>
> **备注:** 29 pages. arXiv admin note: substantial text overlap with arXiv:2401.11256
>
> **摘要:** Cone Beam CT (CBCT) is an important imaging modality nowadays, however lower image quality of CBCT compared to more conventional Computed Tomography (CT) remains a limiting factor in CBCT applications. Deep learning reconstruction methods are a promising alternative to classical analytical and iterative reconstruction methods, but applying such methods to CBCT is often difficult due to the lack of ground truth data, memory limitations and the need for fast inference at clinically-relevant resolutions. In this work we propose LIRE++, an end-to-end rotationally-equivariant multiscale learned invertible primal-dual scheme for fast and memory-efficient CBCT reconstruction. Memory optimizations and multiscale reconstruction allow for fast training and inference, while rotational equivariance improves parameter efficiency. LIRE++ was trained on simulated projection data from a fast quasi-Monte Carlo CBCT projection simulator that we developed as well. Evaluated on synthetic data, LIRE++ gave an average improvement of 1 dB in Peak Signal-to-Noise Ratio over alternative deep learning baselines. On real clinical data, LIRE++ improved the average Mean Absolute Error between the reconstruction and the corresponding planning CT by 10 Hounsfield Units with respect to current proprietary state-of-the-art hybrid deep-learning/iterative method.
>
---
#### [new 067] Generalization of Diffusion Models Arises with a Balanced Representation Space
- **分类: cs.LG; cs.CV**

- **简介: 论文研究扩散模型泛化机制，区分记忆与泛化表征，提出检测记忆方法及无训练编辑技术，强调表征学习对生成质量的关键作用。**

- **链接: [https://arxiv.org/pdf/2512.20963v1](https://arxiv.org/pdf/2512.20963v1)**

> **作者:** Zekai Zhang; Xiao Li; Xiang Li; Lianghe Shi; Meng Wu; Molei Tao; Qing Qu
>
> **备注:** 40 pages, 19 figures. The first two authors contributed equally
>
> **摘要:** Diffusion models excel at generating high-quality, diverse samples, yet they risk memorizing training data when overfit to the training objective. We analyze the distinctions between memorization and generalization in diffusion models through the lens of representation learning. By investigating a two-layer ReLU denoising autoencoder (DAE), we prove that (i) memorization corresponds to the model storing raw training samples in the learned weights for encoding and decoding, yielding localized "spiky" representations, whereas (ii) generalization arises when the model captures local data statistics, producing "balanced" representations. Furthermore, we validate these theoretical findings on real-world unconditional and text-to-image diffusion models, demonstrating that the same representation structures emerge in deep generative models with significant practical implications. Building on these insights, we propose a representation-based method for detecting memorization and a training-free editing technique that allows precise control via representation steering. Together, our results highlight that learning good representations is central to novel and meaningful generative modeling.
>
---
#### [new 068] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation
- **分类: cs.AI; cs.CL; cs.CV; cs.IR**

- **简介: 提出MegaRAG，融合多模态知识图谱增强RAG，解决长文档与跨模态理解难题，提升图文问答推理能力。**

- **链接: [https://arxiv.org/pdf/2512.20626v1](https://arxiv.org/pdf/2512.20626v1)**

> **作者:** Chi-Hsiang Hsiao; Yi-Cheng Wang; Tzung-Sheng Lin; Yi-Ren Yeh; Chu-Song Chen
>
> **摘要:** Retrieval-augmented generation (RAG) enables large language models (LLMs) to dynamically access external information, which is powerful for answering questions over previously unseen documents. Nonetheless, they struggle with high-level conceptual understanding and holistic comprehension due to limited context windows, which constrain their ability to perform deep reasoning over long-form, domain-specific content such as full-length books. To solve this problem, knowledge graphs (KGs) have been leveraged to provide entity-centric structure and hierarchical summaries, offering more structured support for reasoning. However, existing KG-based RAG solutions remain restricted to text-only inputs and fail to leverage the complementary insights provided by other modalities such as vision. On the other hand, reasoning from visual documents requires textual, visual, and spatial cues into structured, hierarchical concepts. To address this issue, we introduce a multimodal knowledge graph-based RAG that enables cross-modal reasoning for better content understanding. Our method incorporates visual cues into the construction of knowledge graphs, the retrieval phase, and the answer generation process. Experimental results across both global and fine-grained question answering tasks show that our approach consistently outperforms existing RAG-based approaches on both textual and multimodal corpora.
>
---
#### [new 069] HyDRA: Hierarchical and Dynamic Rank Adaptation for Mobile Vision Language Model
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 提出HyDRA框架，通过分层动态调整LoRA秩，高效微调移动端视觉语言模型，在不增参数下提升性能4.7%，优于全参微调。**

- **链接: [https://arxiv.org/pdf/2512.20674v1](https://arxiv.org/pdf/2512.20674v1)**

> **作者:** Yuanhao Xi; Xiaohuan Bing; Ramin Yahyapour
>
> **摘要:** Vision Language Models (VLMs) have undergone significant advancements, particularly with the emergence of mobile-oriented VLMs, which offer a wide range of application scenarios. However, the substantial computational requirements for training these models present a significant obstacle to their practical application. To address this issue, Low-Rank Adaptation (LoRA) has been proposed. Nevertheless, the standard LoRA with a fixed rank lacks sufficient capability for training mobile VLMs that process both text and image modalities. In this work, we introduce HyDRA, a parameter-efficient fine-tuning framework designed to implement hierarchical and dynamic rank scheduling for mobile VLMs. This framework incorporates two essential optimization strategies: (1) hierarchical optimization, which involves a coarse-grained approach that assigns different ranks to various layers, as well as a fine-grained method that adjusts ranks within individual layers, and (2) dynamic adjustment, which employs an end-to-end automatic optimization using a lightweight performance model to determine and adjust ranks during the fine-tuning process. Comprehensive experiments conducted on popular benchmarks demonstrate that HyDRA consistently outperforms the baseline, achieving a 4.7\% improvement across various model sizes without increasing the number of trainable parameters. In some tasks, it even surpasses full-parameter fine-tuning.
>
---
#### [new 070] Flow Gym
- **分类: physics.flu-dyn; cs.CV; cs.SE; physics.comp-ph**

- **简介: Flow Gym是用于流场量化算法研究与部署的工具包，统一接口支持测试、训练与部署，集成现有算法并提供JAX实现，解决多帧粒子图像流场分析问题。**

- **链接: [https://arxiv.org/pdf/2512.20642v1](https://arxiv.org/pdf/2512.20642v1)**

> **作者:** Francesco Banelli; Antonio Terpin; Alan Bonomi; Raffaello D'Andrea
>
> **备注:** Code: https://github.com/antonioterpin/flowgym
>
> **摘要:** Flow Gym is a toolkit for research and deployment of flow-field quantification methods inspired by OpenAI Gym and Stable-Baselines3. It uses SynthPix as synthetic image generation engine and provides a unified interface for the testing, deployment and training of (learning-based) algorithms for flow-field quantification from a number of consecutive images of tracer particles. It also contains a growing number of integrations of existing algorithms and stable (re-)implementations in JAX.
>
---
#### [new 071] RoboSafe: Safeguarding Embodied Agents via Executable Safety Logic
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 提出RoboSafe，通过可执行安全逻辑动态防护具身智能体，解决其执行中隐式风险问题，结合回溯与前瞻推理，显著降险保效。**

- **链接: [https://arxiv.org/pdf/2512.21220v1](https://arxiv.org/pdf/2512.21220v1)**

> **作者:** Le Wang; Zonghao Ying; Xiao Yang; Quanchen Zou; Zhenfei Yin; Tianlin Li; Jian Yang; Yaodong Yang; Aishan Liu; Xianglong Liu
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Embodied agents powered by vision-language models (VLMs) are increasingly capable of executing complex real-world tasks, yet they remain vulnerable to hazardous instructions that may trigger unsafe behaviors. Runtime safety guardrails, which intercept hazardous actions during task execution, offer a promising solution due to their flexibility. However, existing defenses often rely on static rule filters or prompt-level control, which struggle to address implicit risks arising in dynamic, temporally dependent, and context-rich environments. To address this, we propose RoboSafe, a hybrid reasoning runtime safeguard for embodied agents through executable predicate-based safety logic. RoboSafe integrates two complementary reasoning processes on a Hybrid Long-Short Safety Memory. We first propose a Backward Reflective Reasoning module that continuously revisits recent trajectories in short-term memory to infer temporal safety predicates and proactively triggers replanning when violations are detected. We then propose a Forward Predictive Reasoning module that anticipates upcoming risks by generating context-aware safety predicates from the long-term safety memory and the agent's multimodal observations. Together, these components form an adaptive, verifiable safety logic that is both interpretable and executable as code. Extensive experiments across multiple agents demonstrate that RoboSafe substantially reduces hazardous actions (-36.8% risk occurrence) compared with leading baselines, while maintaining near-original task performance. Real-world evaluations on physical robotic arms further confirm its practicality. Code will be released upon acceptance.
>
---
#### [new 072] STLDM: Spatio-Temporal Latent Diffusion Model for Precipitation Nowcasting
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 论文提出STLDM模型，用于降水临近预报，结合确定性预测与扩散模型增强，解决模糊与精度低问题，提升效果与效率。**

- **链接: [https://arxiv.org/pdf/2512.21118v1](https://arxiv.org/pdf/2512.21118v1)**

> **作者:** Shi Quan Foo; Chi-Ho Wong; Zhihan Gao; Dit-Yan Yeung; Ka-Hing Wong; Wai-Kin Wong
>
> **备注:** Accepted by TMLR. Camera-ready submission
>
> **摘要:** Precipitation nowcasting is a critical spatio-temporal prediction task for society to prevent severe damage owing to extreme weather events. Despite the advances in this field, the complex and stochastic nature of this task still poses challenges to existing approaches. Specifically, deterministic models tend to produce blurry predictions while generative models often struggle with poor accuracy. In this paper, we present a simple yet effective model architecture termed STLDM, a diffusion-based model that learns the latent representation from end to end alongside both the Variational Autoencoder and the conditioning network. STLDM decomposes this task into two stages: a deterministic forecasting stage handled by the conditioning network, and an enhancement stage performed by the latent diffusion model. Experimental results on multiple radar datasets demonstrate that STLDM achieves superior performance compared to the state of the art, while also improving inference efficiency. The code is available in https://github.com/sqfoo/stldm_official.
>
---
#### [new 073] Language-Guided Grasp Detection with Coarse-to-Fine Learning for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 提出LGGD方法，用粗到精学习实现语言引导抓取，解决语义对齐弱问题，提升机器人按指令抓物的准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.21065v1](https://arxiv.org/pdf/2512.21065v1)**

> **作者:** Zebin Jiang; Tianle Jin; Xiangtong Yao; Alois Knoll; Hu Cao
>
> **备注:** Submitted to IEEE Journal
>
> **摘要:** Grasping is one of the most fundamental challenging capabilities in robotic manipulation, especially in unstructured, cluttered, and semantically diverse environments. Recent researches have increasingly explored language-guided manipulation, where robots not only perceive the scene but also interpret task-relevant natural language instructions. However, existing language-conditioned grasping methods typically rely on shallow fusion strategies, leading to limited semantic grounding and weak alignment between linguistic intent and visual grasp reasoning.In this work, we propose Language-Guided Grasp Detection (LGGD) with a coarse-to-fine learning paradigm for robotic manipulation. LGGD leverages CLIP-based visual and textual embeddings within a hierarchical cross-modal fusion pipeline, progressively injecting linguistic cues into the visual feature reconstruction process. This design enables fine-grained visual-semantic alignment and improves the feasibility of the predicted grasps with respect to task instructions. In addition, we introduce a language-conditioned dynamic convolution head (LDCH) that mixes multiple convolution experts based on sentence-level features, enabling instruction-adaptive coarse mask and grasp predictions. A final refinement module further enhances grasp consistency and robustness in complex scenes.Experiments on the OCID-VLG and Grasp-Anything++ datasets show that LGGD surpasses existing language-guided grasping methods, exhibiting strong generalization to unseen objects and diverse language queries. Moreover, deployment on a real robotic platform demonstrates the practical effectiveness of our approach in executing accurate, instruction-conditioned grasp actions. The code will be released publicly upon acceptance.
>
---
#### [new 074] TexAvatars : Hybrid Texel-3D Representations for Stable Rigging of Photorealistic Gaussian Head Avatars
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 提出TexAvatars，融合解析绑定与纹理空间，提升高斯头像在极端表情姿态下的驱动稳定性与几何一致性，实现更逼真、泛化更强的3D头像建模。**

- **链接: [https://arxiv.org/pdf/2512.21099v1](https://arxiv.org/pdf/2512.21099v1)**

> **作者:** Jaeseong Lee; Junyeong Ahn; Taewoong Kang; Jaegul Choo
>
> **备注:** 3DV 2026, Project page with videos: https://summertight.github.io/TexAvatars/
>
> **摘要:** Constructing drivable and photorealistic 3D head avatars has become a central task in AR/XR, enabling immersive and expressive user experiences. With the emergence of high-fidelity and efficient representations such as 3D Gaussians, recent works have pushed toward ultra-detailed head avatars. Existing approaches typically fall into two categories: rule-based analytic rigging or neural network-based deformation fields. While effective in constrained settings, both approaches often fail to generalize to unseen expressions and poses, particularly in extreme reenactment scenarios. Other methods constrain Gaussians to the global texel space of 3DMMs to reduce rendering complexity. However, these texel-based avatars tend to underutilize the underlying mesh structure. They apply minimal analytic deformation and rely heavily on neural regressors and heuristic regularization in UV space, which weakens geometric consistency and limits extrapolation to complex, out-of-distribution deformations. To address these limitations, we introduce TexAvatars, a hybrid avatar representation that combines the explicit geometric grounding of analytic rigging with the spatial continuity of texel space. Our approach predicts local geometric attributes in UV space via CNNs, but drives 3D deformation through mesh-aware Jacobians, enabling smooth and semantically meaningful transitions across triangle boundaries. This hybrid design separates semantic modeling from geometric control, resulting in improved generalization, interpretability, and stability. Furthermore, TexAvatars captures fine-grained expression effects, including muscle-induced wrinkles, glabellar lines, and realistic mouth cavity geometry, with high fidelity. Our method achieves state-of-the-art performance under extreme pose and expression variations, demonstrating strong generalization in challenging head reenactment settings.
>
---
#### [new 075] MaskOpt: A Large-Scale Mask Optimization Dataset to Advance AI in Integrated Circuit Manufacturing
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 提出MaskOpt数据集，推动AI在IC制造中实现单元与上下文感知的掩模优化，解决现有方法忽略实际布局结构与邻近效应的问题。**

- **链接: [https://arxiv.org/pdf/2512.20655v1](https://arxiv.org/pdf/2512.20655v1)**

> **作者:** Yuting Hu; Lei Zhuang; Hua Xiang; Jinjun Xiong; Gi-Joon Nam
>
> **摘要:** As integrated circuit (IC) dimensions shrink below the lithographic wavelength, optical lithography faces growing challenges from diffraction and process variability. Model-based optical proximity correction (OPC) and inverse lithography technique (ILT) remain indispensable but computationally expensive, requiring repeated simulations that limit scalability. Although deep learning has been applied to mask optimization, existing datasets often rely on synthetic layouts, disregard standard-cell hierarchy, and neglect the surrounding contexts around the mask optimization targets, thereby constraining their applicability to practical mask optimization. To advance deep learning for cell- and context-aware mask optimization, we present MaskOpt, a large-scale benchmark dataset constructed from real IC designs at the 45$\mathrm{nm}$ node. MaskOpt includes 104,714 metal-layer tiles and 121,952 via-layer tiles. Each tile is clipped at a standard-cell placement to preserve cell information, exploiting repeated logic gate occurrences. Different context window sizes are supported in MaskOpt to capture the influence of neighboring shapes from optical proximity effects. We evaluate state-of-the-art deep learning models for IC mask optimization to build up benchmarks, and the evaluation results expose distinct trade-offs across baseline models. Further context size analysis and input ablation studies confirm the importance of both surrounding geometries and cell-aware inputs in achieving accurate mask generation.
>
---
#### [new 076] Schrödinger's Navigator: Imagining an Ensemble of Futures for Zero-Shot Object Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 提出Schrödinger's Navigator框架，通过轨迹条件3D想象应对遮挡与动态目标，提升零样本物体导航在未知环境中的成功率。**

- **链接: [https://arxiv.org/pdf/2512.21201v1](https://arxiv.org/pdf/2512.21201v1)**

> **作者:** Yu He; Da Huang; Zhenyang Liu; Zixiao Gu; Qiang Sun; Guangnan Ye; Yanwei Fu
>
> **摘要:** Zero-shot object navigation (ZSON) requires a robot to locate a target object in a previously unseen environment without relying on pre-built maps or task-specific training. However, existing ZSON methods often struggle in realistic and cluttered environments, particularly when the scene contains heavy occlusions, unknown risks, or dynamically moving target objects. To address these challenges, we propose \textbf{Schrödinger's Navigator}, a navigation framework inspired by Schrödinger's thought experiment on uncertainty. The framework treats unobserved space as a set of plausible future worlds and reasons over them before acting. Conditioned on egocentric visual inputs and three candidate trajectories, a trajectory-conditioned 3D world model imagines future observations along each path. This enables the agent to see beyond occlusions and anticipate risks in unseen regions without requiring extra detours or dense global mapping. The imagined 3D observations are fused into the navigation map and used to update a value map. These updates guide the policy toward trajectories that avoid occlusions, reduce exposure to uncertain space, and better track moving targets. Experiments on a Go2 quadruped robot across three challenging scenarios, including severe static occlusions, unknown risks, and dynamically moving targets, show that Schrödinger's Navigator consistently outperforms strong ZSON baselines in self-localization, object localization, and overall Success Rate in occlusion-heavy environments. These results demonstrate the effectiveness of trajectory-conditioned 3D imagination in enabling robust zero-shot object navigation.
>
---
#### [new 077] Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks
- **分类: cs.LG; cs.AI; cs.CR; cs.CV**

- **简介: 论文提出ARS-OPT与PARS-OPT算法，加速硬标签黑盒攻击中射线搜索的收敛，降低查询成本，理论与实验验证其高效性。**

- **链接: [https://arxiv.org/pdf/2512.21241v1](https://arxiv.org/pdf/2512.21241v1)**

> **作者:** Xinjie Xu; Shuyu Cheng; Dongwei Xu; Qi Xuan; Chen Ma
>
> **备注:** Published at AAAI 2026 (Oral). This version corresponds to the conference proceedings; v2 will include the appendix
>
> **摘要:** In hard-label black-box adversarial attacks, where only the top-1 predicted label is accessible, the prohibitive query complexity poses a major obstacle to practical deployment. In this paper, we focus on optimizing a representative class of attacks that search for the optimal ray direction yielding the minimum $\ell_2$-norm perturbation required to move a benign image into the adversarial region. Inspired by Nesterov's Accelerated Gradient (NAG), we propose a momentum-based algorithm, ARS-OPT, which proactively estimates the gradient with respect to a future ray direction inferred from accumulated momentum. We provide a theoretical analysis of its convergence behavior, showing that ARS-OPT enables more accurate directional updates and achieves faster, more stable optimization. To further accelerate convergence, we incorporate surrogate-model priors into ARS-OPT's gradient estimation, resulting in PARS-OPT with enhanced performance. The superiority of our approach is supported by theoretical guarantees under standard assumptions. Extensive experiments on ImageNet and CIFAR-10 demonstrate that our method surpasses 13 state-of-the-art approaches in query efficiency.
>
---
## 更新

#### [replaced 001] Let Androids Dream of Electric Sheep: A Human-Inspired Image Implication Understanding and Reasoning Framework
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [https://arxiv.org/pdf/2505.17019v2](https://arxiv.org/pdf/2505.17019v2)**

> **作者:** Chenhao Zhang; Yazhe Niu
>
> **备注:** 19 pages, 9 figures, 7 tables. Code & Dataset: https://github.com/MING-ZCH/Let-Androids-Dream-of-Electric-Sheep
>
> **摘要:** Metaphorical comprehension in images remains a critical challenge for AI systems, as existing models struggle to grasp the nuanced cultural, emotional, and contextual implications embedded in visual content. While multimodal large language models (MLLMs) excel in general Visual Question Answer (VQA) tasks, they struggle with a fundamental limitation on image implication tasks: contextual gaps that obscure the relationships between different visual elements and their abstract meanings. Inspired by the human cognitive process, we propose Let Androids Dream (LAD), a novel framework for image implication understanding and reasoning. LAD addresses contextual missing through the three-stage framework: (1) Perception: converting visual information into rich and multi-level textual representations, (2) Search: iteratively searching and integrating cross-domain knowledge to resolve ambiguity, and (3) Reasoning: generating context-alignment image implication via explicit reasoning. Our framework with the lightweight GPT-4o-mini model achieves SOTA performance compared to 15+ MLLMs on English image implication benchmark and a huge improvement on Chinese benchmark, performing comparable with the Gemini-3.0-pro model on Multiple-Choice Question (MCQ) and outperforms the GPT-4o model 36.7% on Open-Style Question (OSQ). Generalization experiments also show that our framework can effectively benefit general VQA and visual reasoning tasks. Additionally, our work provides new insights into how AI can more effectively interpret image implications, advancing the field of vision-language reasoning and human-AI interaction. Our project is publicly available at https://github.com/MING-ZCH/Let-Androids-Dream-of-Electric-Sheep.
>
---
#### [replaced 002] Rethinking Direct Preference Optimization in Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.18736v2](https://arxiv.org/pdf/2505.18736v2)**

> **作者:** Junyong Kang; Seohyun Lim; Kyungjune Baek; Hyunjung Shim
>
> **备注:** Accepted by SPIGM@NeurIPS 2025 and AAAI-26 (Oral)
>
> **摘要:** Aligning text-to-image (T2I) diffusion models with human preferences has emerged as a critical research challenge. While recent advances in this area have extended preference optimization techniques from large language models (LLMs) to the diffusion setting, they often struggle with limited exploration. In this work, we propose a novel and orthogonal approach to enhancing diffusion-based preference optimization. First, we introduce a stable reference model update strategy that relaxes the frozen reference model, encouraging exploration while maintaining a stable optimization anchor through reference model regularization. Second, we present a timestep-aware training strategy that mitigates the reward scale imbalance problem across timesteps. Our method can be integrated into various preference optimization algorithms. Experimental results show that our approach improves the performance of state-of-the-art methods on human preference evaluation benchmarks. The code is available at the Github: https://github.com/kaist-cvml/RethinkingDPO_Diffusion_Models.
>
---
#### [replaced 003] SemanticGen: Video Generation in Semantic Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20619v2](https://arxiv.org/pdf/2512.20619v2)**

> **作者:** Jianhong Bai; Xiaoshi Wu; Xintao Wang; Xiao Fu; Yuanxing Zhang; Qinghe Wang; Xiaoyu Shi; Menghan Xia; Zuozhu Liu; Haoji Hu; Pengfei Wan; Kun Gai
>
> **备注:** Project page: https://jianhongbai.github.io/SemanticGen/
>
> **摘要:** State-of-the-art video generative models typically learn the distribution of video latents in the VAE space and map them to pixels using a VAE decoder. While this approach can generate high-quality videos, it suffers from slow convergence and is computationally expensive when generating long videos. In this paper, we introduce SemanticGen, a novel solution to address these limitations by generating videos in the semantic space. Our main insight is that, due to the inherent redundancy in videos, the generation process should begin in a compact, high-level semantic space for global planning, followed by the addition of high-frequency details, rather than directly modeling a vast set of low-level video tokens using bi-directional attention. SemanticGen adopts a two-stage generation process. In the first stage, a diffusion model generates compact semantic video features, which define the global layout of the video. In the second stage, another diffusion model generates VAE latents conditioned on these semantic features to produce the final output. We observe that generation in the semantic space leads to faster convergence compared to the VAE latent space. Our method is also effective and computationally efficient when extended to long video generation. Extensive experiments demonstrate that SemanticGen produces high-quality videos and outperforms state-of-the-art approaches and strong baselines.
>
---
#### [replaced 004] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 论文提出O3SLM模型与SketchVCL数据集，解决LVLM理解手绘草图能力弱的问题，在多项草图任务上实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2511.14368v2](https://arxiv.org/pdf/2511.14368v2)**

> **作者:** Rishi Gupta; Mukilan Karuppasamy; Shyam Marjit; Aditay Tripathi; Anirban Chakraborty
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** While Large Vision Language Models (LVLMs) are increasingly deployed in real-world applications, their ability to interpret abstract visual inputs remains limited. Specifically, they struggle to comprehend hand-drawn sketches, a modality that offers an intuitive means of expressing concepts that are difficult to describe textually. We identify the primary bottleneck as the absence of a large-scale dataset that jointly models sketches, photorealistic images, and corresponding natural language instructions. To address this, we present two key contributions: (1) a new, large-scale dataset of image-sketch-instruction triplets designed to facilitate both pretraining and instruction tuning, and (2) O3SLM, an LVLM trained on this dataset. Comprehensive evaluations on multiple sketch-based tasks: (a) object localization, (b) counting, (c) image retrieval i.e., (SBIR and fine-grained SBIR), and (d) visual question answering (VQA); while incorporating the three existing sketch datasets, namely QuickDraw!, Sketchy, and Tu Berlin, along with our generated SketchVCL dataset, show that O3SLM achieves state-of-the-art performance, substantially outperforming existing LVLMs in sketch comprehension and reasoning.
>
---
#### [replaced 005] DEAR: Dataset for Evaluating the Aesthetics of Rendering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.05209v3](https://arxiv.org/pdf/2512.05209v3)**

> **作者:** Vsevolod Plohotnuk; Artyom Panshin; Nikola Banić; Simone Bianco; Michael Freeman; Egor Ershov
>
> **摘要:** Traditional Image Quality Assessment~(IQA) focuses on quantifying technical degradations such as noise, blur, or compression artifacts, using both full-reference and no-reference objective metrics. However, evaluation of rendering aesthetics, a growing domain relevant to photographic editing, content creation, and AI-generated imagery, remains underexplored due to the lack of datasets that reflect the inherently subjective nature of style preference. In this work, a novel benchmark dataset designed to model human aesthetic judgments of image rendering styles is introduced: the Dataset for Evaluating the Aesthetics of Rendering (DEAR). Built upon the MIT-Adobe FiveK dataset, DEAR incorporates pairwise human preference scores collected via large-scale crowdsourcing, with each image pair evaluated by 25 distinct human evaluators with a total of 13,648 of them participating overall. These annotations capture nuanced, context-sensitive aesthetic preferences, enabling the development and evaluation of models that go beyond traditional distortion-based IQA, focusing on a new task: Evaluation of Aesthetics of Rendering (EAR). The data collection pipeline is described, human voting patterns are analyzed, and multiple use cases are outlined, including style preference prediction, aesthetic benchmarking, and personalized aesthetic modeling. To the best of the authors' knowledge, DEAR is the first dataset to systematically address image aesthetics of rendering assessment grounded in subjective human preferences. A subset of 100 images with markup for them is published on HuggingFace (huggingface.co/datasets/vsevolodpl/DEAR).
>
---
#### [replaced 006] Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17514v2](https://arxiv.org/pdf/2512.17514v2)**

> **作者:** Sairam VCR; Rishabh Lalla; Aveen Dayal; Tejal Kulkarni; Anuj Lalla; Vineeth N Balasubramanian; Muhammad Haris Khan
>
> **摘要:** Current state-of-the-art approaches in Source-Free Object Detection (SFOD) typically rely on Mean-Teacher self-labeling. However, domain shift often reduces the detector's ability to maintain strong object-focused representations, causing high-confidence activations over background clutter. This weak object focus results in unreliable pseudo-labels from the detection head. While prior works mainly refine these pseudo-labels, they overlook the underlying need to strengthen the feature space itself. We propose FALCON-SFOD (Foundation-Aligned Learning with Clutter suppression and Noise robustness), a framework designed to enhance object-focused adaptation under domain shift. It consists of two complementary components. SPAR (Spatial Prior-Aware Regularization) leverages the generalization strength of vision foundation models to regularize the detector's feature space. Using class-agnostic binary masks derived from OV-SAM, SPAR promotes structured and foreground-focused activations by guiding the network toward object regions. IRPL (Imbalance-aware Noise Robust Pseudo-Labeling) complements SPAR by promoting balanced and noise-tolerant learning under severe foreground-background imbalance. Guided by a theoretical analysis that connects these designs to tighter localization and classification error bounds, FALCON-SFOD achieves competitive performance across SFOD benchmarks.
>
---
#### [replaced 007] Unbiased Region-Language Alignment for Open-Vocabulary Dense Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.06244v4](https://arxiv.org/pdf/2412.06244v4)**

> **作者:** Yunheng Li; Yuxuan Li; Quansheng Zeng; Wenhai Wang; Qibin Hou; Ming-Ming Cheng
>
> **备注:** Accepted at ICCV 2025. The code is available at https://github.com/HVision-NKU/DenseVLM
>
> **摘要:** Pre-trained vision-language models (VLMs), such as CLIP, have demonstrated impressive zero-shot recognition capability, but still underperform in dense prediction tasks. Self-distillation recently is emerging as a promising approach for fine-tuning VLMs to better adapt to local regions without requiring extensive annotations. However, previous state-of-the-art approaches often suffer from significant `foreground bias', where models tend to wrongly identify background regions as foreground objects. To alleviate this issue, we propose DenseVLM, a framework designed to learn unbiased region-language alignment from powerful pre-trained VLM representations. To alleviate this issue, we propose DenseVLM, a framework designed to learn unbiased region-language alignment from powerful pre-trained VLM representations. DenseVLM leverages the pre-trained VLM to retrieve categories for unlabeled regions and then decouples the interference between foreground and background features. We show that DenseVLM can directly replace the original VLM in open-vocabulary object detection and image segmentation methods, leading to notable performance improvements. Furthermore, it exhibits promising zero-shot scalability when training on more extensive and diverse datasets. Our code is available at https://github.com/HVision-NKU/DenseVLM.
>
---
#### [replaced 008] Intersectional Fairness in Vision-Language Models for Medical Image Disease Classification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.15249v2](https://arxiv.org/pdf/2512.15249v2)**

> **作者:** Yupeng Zhang; Adam G. Dunn; Usman Naseem; Jinman Kim
>
> **摘要:** Medical artificial intelligence (AI) systems, particularly multimodal vision-language models (VLM), often exhibit intersectional biases where models are systematically less confident in diagnosing marginalised patient subgroups. Such bias can lead to higher rates of inaccurate and missed diagnoses due to demographically skewed data and divergent distributions of diagnostic certainty. Current fairness interventions frequently fail to address these gaps or compromise overall diagnostic performance to achieve statistical parity among the subgroups. In this study, we developed Cross-Modal Alignment Consistency (CMAC-MMD), a training framework that standardises diagnostic certainty across intersectional patient subgroups. Unlike traditional debiasing methods, this approach equalises the model's decision confidence without requiring sensitive demographic data during clinical inference. We evaluated this approach using 10,015 skin lesion images (HAM10000) with external validation on 12,000 images (BCN20000), and 10,000 fundus images for glaucoma detection (Harvard-FairVLMed), stratifying performance by intersectional age, gender, and race attributes. In the dermatology cohort, the proposed method reduced the overall intersectional missed diagnosis gap (difference in True Positive Rate, $Δ$TPR) from 0.50 to 0.26 while improving the overall Area Under the Curve (AUC) from 0.94 to 0.97 compared to standard training. Similarly, for glaucoma screening, the method reduced $Δ$TPR from 0.41 to 0.31, achieving a better AUC of 0.72 (vs. 0.71 baseline). This establishes a scalable framework for developing high-stakes clinical decision support systems that are both accurate and can perform equitably across diverse patient subgroups, ensuring reliable performance without increasing privacy risks.
>
---
#### [replaced 009] TrackNetV5: Residual-Driven Spatio-Temporal Refinement and Motion Direction Decoupling for Fast Object Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02789v3](https://arxiv.org/pdf/2512.02789v3)**

> **作者:** Haonan Tang; Yanjun Chen; Lezhi Jiang; Qianfei Li; Xinyu Guo
>
> **摘要:** The TrackNet series has established a strong baseline for fast-moving small object tracking in sports. However, existing iterations face significant limitations: V1-V3 struggle with occlusions due to a reliance on purely visual cues, while TrackNetV4, despite introducing motion inputs, suffers from directional ambiguity as its absolute difference method discards motion polarity. To overcome these bottlenecks, we propose TrackNetV5, a robust architecture integrating two novel mechanisms. First, to recover lost directional priors, we introduce the Motion Direction Decoupling (MDD) module. Unlike V4, MDD decomposes temporal dynamics into signed polarity fields, explicitly encoding both movement occurrence and trajectory direction. Second, we propose the Residual-Driven Spatio-Temporal Refinement (R-STR) head. Operating on a coarse-to-fine paradigm, this Transformer-based module leverages factorized spatio-temporal contexts to estimate a corrective residual, effectively recovering occluded targets. Extensive experiments on the TrackNetV2 dataset demonstrate that TrackNetV5 achieves a new state-of-the-art F1-score of 0.9859 and an accuracy of 0.9733, significantly outperforming previous versions. Notably, this performance leap is achieved with a marginal 3.7% increase in FLOPs compared to V4, maintaining real-time inference capabilities while delivering superior tracking precision.
>
---
#### [replaced 010] Anatomy-R1: Enhancing Anatomy Reasoning in Multimodal Large Language Models via Anatomical Similarity Curriculum and Group Diversity Augmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.19512v2](https://arxiv.org/pdf/2512.19512v2)**

> **作者:** Ziyang Song; Zelin Zang; Zuyao Chen; Xusheng Liang; Dong Yi; Jinlin Wu; Hongbin Liu; Jiebo Luo; Zhen. Lei
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved impressive progress in natural image reasoning, yet their potential in medical imaging remains underexplored, especially in clinical anatomical surgical images. Anatomy understanding tasks demand precise understanding and clinically coherent answers, which are difficult to achieve due to the complexity of medical data and the scarcity of high-quality expert annotations. These challenges limit the effectiveness of conventional Supervised Fine-Tuning (SFT) strategies. While recent work has demonstrated that Group Relative Policy Optimization (GRPO) can enhance reasoning in MLLMs without relying on large amounts of data, we find two weaknesses that hinder GRPO's reasoning performance in anatomy recognition: 1) knowledge cannot be effectively shared between different anatomical structures, resulting in uneven information gain and preventing the model from converging, and 2) the model quickly converges to a single reasoning path, suppressing the exploration of diverse strategies. To overcome these challenges, we propose two novel methods. First, we implement a progressive learning strategy called Anatomical Similarity Curriculum Learning by controlling question difficulty via the similarity of answer choices, enabling the model to master complex problems incrementally. Second, we utilize question augmentation referred to as Group Diversity Question Augmentation to expand the model's search space for difficult queries, mitigating the tendency to produce uniform responses. Comprehensive experiments on the SGG-VQA and OmniMedVQA benchmarks show our method achieves a significant improvement across the two benchmarks, demonstrating its effectiveness in enhancing the medical reasoning capabilities of MLLMs. The code can be found in https://github.com/tomato996/Anatomy-R1
>
---
#### [replaced 011] Diagnose Like A REAL Pathologist: An Uncertainty-Focused Approach for Trustworthy Multi-Resolution Multiple Instance Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06433v2](https://arxiv.org/pdf/2511.06433v2)**

> **作者:** Sungrae Hong; Sol Lee; Jisu Shin; Jiwon Jeong; Mun Yong Yi
>
> **备注:** Accepted by IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** With the increasing demand for histopathological specimen examination and diagnostic reporting, Multiple Instance Learning (MIL) has received heightened research focus as a viable solution for AI-centric diagnostic aid. Recently, to improve its performance and make it work more like a pathologist, several MIL approaches based on the use of multiple-resolution images have been proposed, delivering often higher performance than those that use single-resolution images. Despite impressive recent developments of multiple-resolution MIL, previous approaches only focus on improving performance, thereby lacking research on well-calibrated MIL that clinical experts can rely on for trustworthy diagnostic results. In this study, we propose Uncertainty-Focused Calibrated MIL (UFC-MIL), which more closely mimics the pathologists' examination behaviors while providing calibrated diagnostic predictions, using multiple images with different resolutions. UFC-MIL includes a novel patch-wise loss that learns the latent patterns of instances and expresses their uncertainty for classification. Also, the attention-based architecture with a neighbor patch aggregation module collects features for the classifier. In addition, aggregated predictions are calibrated through patch-level uncertainty without requiring multiple iterative inferences, which is a key practical advantage. Against challenging public datasets, UFC-MIL shows superior performance in model calibration while achieving classification accuracy comparable to that of state-of-the-art methods.
>
---
#### [replaced 012] View-aware Cross-modal Distillation for Multi-view Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12870v2](https://arxiv.org/pdf/2511.12870v2)**

> **作者:** Trung Thanh Nguyen; Yasutomo Kawanishi; Vijay John; Takahiro Komamizu; Ichiro Ide
>
> **备注:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** The widespread use of multi-sensor systems has increased research in multi-view action recognition. While existing approaches in multi-view setups with fully overlapping sensors benefit from consistent view coverage, partially overlapping settings where actions are visible in only a subset of views remain underexplored. This challenge becomes more severe in real-world scenarios, as many systems provide only limited input modalities and rely on sequence-level annotations instead of dense frame-level labels. In this study, we propose View-aware Cross-modal Knowledge Distillation (ViCoKD), a framework that distills knowledge from a fully supervised multi-modal teacher to a modality- and annotation-limited student. ViCoKD employs a cross-modal adapter with cross-modal attention, allowing the student to exploit multi-modal correlations while operating with incomplete modalities. Moreover, we propose a View-aware Consistency module to address view misalignment, where the same action may appear differently or only partially across viewpoints. It enforces prediction alignment when the action is co-visible across views, guided by human-detection masks and confidence-weighted Jensen-Shannon divergence between their predicted class distributions. Experiments on the real-world MultiSensor-Home dataset show that ViCoKD consistently outperforms competitive distillation methods across multiple backbones and environments, delivering significant gains and surpassing the teacher model under limited conditions.
>
---
#### [replaced 013] Pointmap-Conditioned Diffusion for Consistent Novel View Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.02913v2](https://arxiv.org/pdf/2501.02913v2)**

> **作者:** Thang-Anh-Quan Nguyen; Nathan Piasco; Luis Roldão; Moussab Bennehar; Dzmitry Tsishkou; Laurent Caraffa; Jean-Philippe Tarel; Roland Brémond
>
> **备注:** WACV 2026. Project page: https://ntaquan0125.github.io/pointmap-conditioned-diffusion
>
> **摘要:** Synthesizing extrapolated views remains a difficult task, especially in urban driving scenes, where the only reliable sources of data are limited RGB captures and sparse LiDAR points. To address this problem, we present PointmapDiff, a framework for novel view synthesis that utilizes pre-trained 2D diffusion models. Our method leverages point maps (i.e., rasterized 3D scene coordinates) as a conditioning signal, capturing geometric and photometric priors from the reference images to guide the image generation process. With the proposed reference attention layers and ControlNet for point map features, PointmapDiff can generate accurate and consistent results across varying viewpoints while respecting geometric fidelity. Experiments on real-life driving data demonstrate that our method achieves high-quality generation with flexibility over point map conditioning signals (e.g., dense depth map or even sparse LiDAR points) and can be used to distill to 3D representations such as 3D Gaussian Splatting for improving view extrapolation.
>
---
#### [replaced 014] FedPOD: the deployable units of training for federated learning
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.20610v2](https://arxiv.org/pdf/2512.20610v2)**

> **作者:** Daewoon Kim; Si Young Yie; Jae Sung Lee
>
> **备注:** 12 pages, 12 figures, MICCAI
>
> **摘要:** This paper proposes FedPOD, which ranked first in the 2024 Federated Tumor Segmentation (FeTS) Challenge, for optimizing learning efficiency and communication cost in federated learning among multiple clients. Inspired by FedPIDAvg, we define a round-wise task for FedPOD to enhance training efficiency. FedPIDAvg achieved performance improvement by incorporating the training loss reduction for prediction entropy as weights using differential terms. Furthermore, by modeling data distribution with a Poisson distribution and using a PID controller, it reduced communication costs even in skewed data distribution. However, excluding participants classified as outliers based on the Poisson distribution can limit data utilization. Additionally, PID controller requires the same participants to be maintained throughout the federated learning process as it uses previous rounds' learning information in the current round. In our approach, FedPOD addresses these issues by including participants excluded as outliers, eliminating dependency on previous rounds' learning information, and applying a method for calculating validation loss at each round. In this challenge, FedPOD presents comparable performance to FedPIDAvg in metrics of Dice score, 0.78, 0.71 and 0.72 for WT, ET and TC in average, and projected convergence score, 0.74 in average. Furthermore, the concept of FedPOD draws inspiration from Kubernetes' smallest computing unit, POD, designed to be compatible with Kubernetes auto-scaling. Extending round-wise tasks of FedPOD to POD units allows flexible design by applying scale-out similar to Kubernetes' auto-scaling. This work demonstrated the potentials of FedPOD to enhance federated learning by improving efficiency, flexibility, and performance in metrics.
>
---
#### [replaced 015] SPOC: Spatially-Progressing Object State Change Segmentation in Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.11953v2](https://arxiv.org/pdf/2503.11953v2)**

> **作者:** Priyanka Mandikal; Tushar Nagarajan; Alex Stoken; Zihui Xue; Kristen Grauman
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Object state changes in video reveal critical cues about human and agent activity. However, existing methods are limited to temporal localization of when the object is in its initial state (e.g., cheese block) versus when it has completed a state change (e.g., grated cheese), offering no insight into where the change is unfolding. We propose to deepen the problem by introducing the spatially-progressing object state change segmentation task. The goal is to segment at the pixel-level those regions of an object that are actionable and those that are transformed. We show that state-of-the-art VLMs and video segmentation methods struggle at this task, underscoring its difficulty and novelty. As an initial baseline, we design a VLM-based pseudo-labeling approach, state-change dynamics constraints, and a novel WhereToChange benchmark built on in-the-wild Internet videos. Experiments on two datasets validate both the challenge of the new task as well as the promise of our model for localizing exactly where and how fast objects are changing in video. We further demonstrate useful implications for tracking activity progress to benefit robotic agents. Overall, our work positions spatial OSC segmentation as a new frontier task for video understanding: one that challenges current SOTA methods and invites the community to build more robust, state-change-sensitive representations. Project page: https://vision.cs.utexas.edu/projects/spoc-spatially-progressing-osc
>
---
#### [replaced 016] Towards Arbitrary-Scale Spacecraft Image Super-Resolution via Salient Region-Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.18127v2](https://arxiv.org/pdf/2504.18127v2)**

> **作者:** Jingfan Yang; Hu Gao; Ying Zhang; Depeng Dang
>
> **备注:** Accepted by Pattern Recognition
>
> **摘要:** Spacecraft image super-resolution seeks to enhance low-resolution spacecraft images into high-resolution ones. Although existing arbitrary-scale super-resolution methods perform well on general images, they tend to overlook the difference in features between the spacecraft core region and the large black space background, introducing irrelevant noise. In this paper, we propose a salient region-guided spacecraft image arbitrary-scale super-resolution network (SGSASR), which uses features from the spacecraft core salient regions to guide latent modulation and achieve arbitrary-scale super-resolution. Specifically, we design a spacecraft core region recognition block (SCRRB) that identifies the core salient regions in spacecraft images using a pre-trained saliency detection model. Furthermore, we present an adaptive-weighted feature fusion enhancement mechanism (AFFEM) to selectively aggregate the spacecraft core region features with general image features by dynamic weight parameter to enhance the response of the core salient regions. Experimental results demonstrate that the proposed SGSASR outperforms state-of-the-art approaches.
>
---
#### [replaced 017] Learning to Refocus with Video Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19823v2](https://arxiv.org/pdf/2512.19823v2)**

> **作者:** SaiKiran Tedla; Zhoutong Zhang; Xuaner Zhang; Shumian Xin
>
> **备注:** Code and data are available at https://learn2refocus.github.io . SIGGRAPH Asia 2025, Dec. 2025
>
> **摘要:** Focus is a cornerstone of photography, yet autofocus systems often fail to capture the intended subject, and users frequently wish to adjust focus after capture. We introduce a novel method for realistic post-capture refocusing using video diffusion models. From a single defocused image, our approach generates a perceptually accurate focal stack, represented as a video sequence, enabling interactive refocusing and unlocking a range of downstream applications. We release a large-scale focal stack dataset acquired under diverse real-world smartphone conditions to support this work and future research. Our method consistently outperforms existing approaches in both perceptual quality and robustness across challenging scenarios, paving the way for more advanced focus-editing capabilities in everyday photography. Code and data are available at https://learn2refocus.github.io
>
---
#### [replaced 018] A Multicore and Edge TPU-Accelerated Multimodal TinyML System for Livestock Behavior Recognition
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2504.11467v3](https://arxiv.org/pdf/2504.11467v3)**

> **作者:** Qianxue Zhang; Eiman Kanjo
>
> **备注:** 12 pages, 10 figures
>
> **摘要:** The advancement of technology has revolutionized the agricultural industry, transitioning it from labor-intensive farming practices to automated, AI-powered management systems. In recent years, more intelligent livestock monitoring solutions have been proposed to enhance farming efficiency and productivity. This work presents a novel approach to animal activity recognition and movement tracking, leveraging tiny machine learning (TinyML) techniques, wireless communication framework, and microcontroller platforms to develop an efficient, cost-effective livestock sensing system. It collects and fuses accelerometer data and vision inputs to build a multimodal network for three tasks: image classification, object detection, and behavior recognition. The system is deployed and evaluated on commercial microcontrollers for real-time inference using embedded applications, demonstrating up to 270$\times$ model size reduction, less than 80ms response latency, and on-par performance comparable to existing methods. The incorporation of the wireless communication technique allows for seamless data transmission between devices, benefiting use cases in remote locations with poor Internet connectivity. This work delivers a robust, scalable IoT-edge livestock monitoring solution adaptable to diverse farming needs, offering flexibility for future extensions.
>
---
#### [replaced 019] Interpretable Plant Leaf Disease Detection Using Attention-Enhanced CNN
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.17864v2](https://arxiv.org/pdf/2512.17864v2)**

> **作者:** Balram Singh; Ram Prakash Sharma; Somnath Dey
>
> **备注:** 27 pages, 12 figures
>
> **摘要:** Plant diseases pose a significant threat to global food security, necessitating accurate and interpretable disease detection methods. This study introduces an interpretable attention-guided Convolutional Neural Network (CNN), CBAM-VGG16, for plant leaf disease detection. By integrating Convolution Block Attention Module (CBAM) at each convolutional stage, the model enhances feature extraction and disease localization. Trained on five diverse plant disease datasets, our approach outperforms recent techniques, achieving high accuracy (up to 98.87%) and demonstrating robust generalization. Here, we show the effectiveness of our method through comprehensive evaluation and interpretability analysis using CBAM attention maps, Grad-CAM, Grad-CAM++, and Layer-wise Relevance Propagation (LRP). This study advances the application of explainable AI in agricultural diagnostics, offering a transparent and reliable system for smart farming. The code of our proposed work is available at https://github.com/BS0111/PlantAttentionCBAM.
>
---
#### [replaced 020] A European Multi-Center Breast Cancer MRI Dataset
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.00474v2](https://arxiv.org/pdf/2506.00474v2)**

> **作者:** Gustav Müller-Franzes; Lorena Escudero Sánchez; Nicholas Payne; Alexandra Athanasiou; Michael Kalogeropoulos; Aitor Lopez; Alfredo Miguel Soro Busto; Julia Camps Herrero; Nika Rasoolzadeh; Tianyu Zhang; Ritse Mann; Debora Jutz; Maike Bode; Christiane Kuhl; Yuan Gao; Wouter Veldhuis; Oliver Lester Saldanha; JieFu Zhu; Jakob Nikolas Kather; Daniel Truhn; Fiona J. Gilbert
>
> **摘要:** Early detection of breast cancer is critical for improving patient outcomes. While mammography remains the primary screening modality, magnetic resonance imaging (MRI) is increasingly recommended as a supplemental tool for women with dense breast tissue and those at elevated risk. However, the acquisition and interpretation of multiparametric breast MRI are time-consuming and require specialized expertise, limiting scalability in clinical practice. Artificial intelligence (AI) methods have shown promise in supporting breast MRI interpretation, but their development is hindered by the limited availability of large, diverse, and publicly accessible datasets. To address this gap, we present a publicly available, multi-center breast MRI dataset collected across six clinical institutions in five European countries. The dataset comprises 741 examinations from women undergoing screening or diagnostic breast MRI and includes malignant, benign, and non-lesion cases. Data were acquired using heterogeneous scanners, field strengths, and acquisition protocols, reflecting real-world clinical variability. In addition, we report baseline benchmark experiments using a transformer-based model to illustrate potential use cases of the dataset and to provide reference performance for future methodological comparisons.
>
---
#### [replaced 021] SmokeSeer: 3D Gaussian Splatting for Smoke Removal and Scene Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17329v3](https://arxiv.org/pdf/2509.17329v3)**

> **作者:** Neham Jain; Andrew Jong; Sebastian Scherer; Ioannis Gkioulekas
>
> **备注:** Project website: https://imaging.cs.cmu.edu/smokeseer
>
> **摘要:** Smoke in real-world scenes can severely degrade image quality and hamper visibility. Recent image restoration methods either rely on data-driven priors that are susceptible to hallucinations, or are limited to static low-density smoke. We introduce SmokeSeer, a method for simultaneous 3D scene reconstruction and smoke removal from multi-view video sequences. Our method uses thermal and RGB images, leveraging the reduced scattering in thermal images to see through smoke. We build upon 3D Gaussian splatting to fuse information from the two image modalities, and decompose the scene into smoke and non-smoke components. Unlike prior work, SmokeSeer handles a broad range of smoke densities and adapts to temporally varying smoke. We validate our method on synthetic data and a new real-world smoke dataset with RGB and thermal images. We provide an open-source implementation and data on the project website.
>
---
#### [replaced 022] V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval
- **分类: eess.IV; cs.AI; cs.AR; cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2512.12284v3](https://arxiv.org/pdf/2512.12284v3)**

> **作者:** Donghyuk Kim; Sejeong Yang; Wonjin Shin; Joo-Young Kim
>
> **备注:** 14 pages, 20 figures, conference, accepted by HPCA 2026
>
> **摘要:** Streaming video large language models (LLMs) are increasingly used for real-time multimodal tasks such as video captioning, question answering, conversational agents, and augmented reality. However, these models face fundamental memory and computational challenges because their key-value (KV) caches grow substantially with continuous streaming video input. This process requires an iterative prefill stage, which is a unique feature of streaming video LLMs. Due to its iterative prefill stage, it suffers from significant limitations, including extensive computation, substantial data transfer, and degradation in accuracy. Crucially, this issue is exacerbated for edge deployment, which is the primary target for these models. In this work, we propose V-Rex, the first software-hardware co-designed accelerator that comprehensively addresses both algorithmic and hardware bottlenecks in streaming video LLM inference. At its core, V-Rex introduces ReSV, a training-free dynamic KV cache retrieval algorithm. ReSV exploits temporal and spatial similarity-based token clustering to reduce excessive KV cache memory across video frames. To fully realize these algorithmic benefits, V-Rex offers a compact, low-latency hardware accelerator with a dynamic KV cache retrieval engine (DRE), featuring bit-level and early-exit based computing units. V-Rex achieves unprecedented real-time of 3.9-8.3 FPS and energy-efficient streaming video LLM inference on edge deployment with negligible accuracy loss. While DRE only accounts for 2.2% power and 2.0% area, the system delivers 1.9-19.7x speedup and 3.1-18.5x energy efficiency improvements over AGX Orin GPU. This work is the first to comprehensively tackle KV cache retrieval across algorithms and hardware, enabling real-time streaming video LLM inference on resource-constrained edge devices.
>
---
#### [replaced 023] SSL4RL: Revisiting Self-supervised Learning as Intrinsic Reward for Visual-Language Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.16416v3](https://arxiv.org/pdf/2510.16416v3)**

> **作者:** Xiaojun Guo; Runyu Zhou; Yifei Wang; Qi Zhang; Chenheng Zhang; Stefanie Jegelka; Xiaohan Wang; Jiajun Chai; Guojun Yin; Wei Lin; Yisen Wang
>
> **摘要:** Vision-language models (VLMs) have shown remarkable abilities by integrating large language models with visual inputs. However, they often fail to utilize visual evidence adequately, either depending on linguistic priors in vision-centric tasks or resorting to textual shortcuts during reasoning. Although reinforcement learning (RL) can align models with desired behaviors, its application to VLMs has been hindered by the lack of scalable and reliable reward mechanisms. To overcome this challenge, we propose SSL4RL, a novel framework that leverages self-supervised learning (SSL) tasks as a source of verifiable rewards for RL-based fine-tuning. Our approach reformulates SSL objectives-such as predicting image rotation or reconstructing masked patches-into dense, automatic reward signals, eliminating the need for human preference data or unreliable AI evaluators. Experiments show that SSL4RL substantially improves performance on both vision-centric and vision-language reasoning benchmarks. Furthermore, through systematic ablations, we identify key factors-such as task difficulty, model scale, and semantic alignment with the target domain-that influence the effectiveness of SSL4RL tasks, offering new design principles for future work. We also demonstrate the framework's generality by applying it to graph learning, where it yields significant gains. SSL4RL establishes a versatile and effective paradigm for aligning multimodal models using verifiable, self-supervised objectives.
>
---
#### [replaced 024] Knowledge Augmentation via Synthetic Data: A Framework for Real-World ECG Image Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.21968v2](https://arxiv.org/pdf/2507.21968v2)**

> **作者:** Xiaoyu Wang; Ramesh Nadarajah; Zhiqiang Zhang; David Wong
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** In real-world clinical practice, electrocardiograms (ECGs) are often captured and shared as photographs. However, publicly available ECG data, and thus most related research, relies on digital signals. This has led to a disconnect in which computer assisted interpretation of ECG cannot easily be applied to ECG images. The emergence of high-fidelity synthetic data generators has introduced practical alternatives by producing realistic, photo-like, ECG images derived from the digital signal that could help narrow this divide. To address this, we propose a novel knowledge augmentation framework that uses synthetic data generated from multiple sources to provide generalisable and accurate interpretation of ECG photographs. Our framework features two key contributions. First, we introduce a robust pre-processing pipeline designed to remove background artifacts and reduces visual differences between images. Second, we implement a two-stage training strategy: a Morphology Learning Stage, where the model captures broad morphological features from visually different, scan-like synthetic data, followed by a Task-Specific Adaptation Stage, where the model is fine-tuned on the photo-like target data. We tested the model on the British Heart Foundation Challenge dataset, to classify five common ECG findings: myocardial infarction (MI), atrial fibrillation, hypertrophy, conduction disturbance, and ST/T changes. Our approach, built upon the ConvNeXt backbone, outperforms a single-source training baseline and achieved \textbf{1st} place in the challenge with an macro-AUROC of \textbf{0.9677}. These results suggest that incorporating morphology learning from heterogeneous sources offers a more robust and generalizable paradigm than conventional single-source training.
>
---
#### [replaced 025] Learning to Generate Human-Human-Object Interactions from Textual Descriptions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20446v2](https://arxiv.org/pdf/2511.20446v2)**

> **作者:** Jeonghyeon Na; Sangwon Baik; Inhee Lee; Junyoung Lee; Hanbyul Joo
>
> **备注:** Project Page: https://tlb-miss.github.io/hhoi/
>
> **摘要:** The way humans interact with each other, including interpersonal distances, spatial configuration, and motion, varies significantly across different situations. To enable machines to understand such complex, context-dependent behaviors, it is essential to model multiple people in relation to the surrounding scene context. In this paper, we present a novel research problem to model the correlations between two people engaged in a shared interaction involving an object. We refer to this formulation as Human-Human-Object Interactions (HHOIs). To overcome the lack of dedicated datasets for HHOIs, we present a newly captured HHOIs dataset and a method to synthesize HHOI data by leveraging image generative models. As an intermediary, we obtain individual human-object interaction (HOIs) and human-human interaction (HHIs) from the HHOIs, and with these data, we train an text-to-HOI and text-to-HHI model using score-based diffusion model. Finally, we present a unified generative framework that integrates the two individual model, capable of synthesizing complete HHOIs in a single advanced sampling process. Our method extends HHOI generation to multi-human settings, enabling interactions involving more than two individuals. Experimental results show that our method generates realistic HHOIs conditioned on textual descriptions, outperforming previous approaches that focus only on single-human HOIs. Furthermore, we introduce multi-human motion generation involving objects as an application of our framework.
>
---
#### [replaced 026] GaussianVision: Vision-Language Alignment from Compressed Image Representations using 2D Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 提出GaussianVision，用2D高斯泼溅压缩图像表征，替代RGB像素输入，降低传输与计算开销，适配CLIP架构，实现高效视觉-语言对齐。**

- **链接: [https://arxiv.org/pdf/2509.22615v2](https://arxiv.org/pdf/2509.22615v2)**

> **作者:** Yasmine Omri; Connor Ding; Tsachy Weissman; Thierry Tambe
>
> **摘要:** Modern vision language pipelines are driven by RGB vision encoders trained on massive image text corpora. While these pipelines have enabled impressive zero-shot capabilities and strong transfer across tasks, they still inherit two structural inefficiencies from the pixel domain: (i) transmitting dense RGB images from edge devices to the cloud is energy-intensive and costly, and (ii) patch-based tokenization explodes sequence length, stressing attention budgets and context limits. We explore 2D Gaussian Splatting (2DGS) as an alternative visual substrate for alignment: a compact, spatially adaptive representation that parameterizes images by a set of colored anisotropic Gaussians. We develop a scalable 2DGS pipeline with structured initialization, luminance-aware pruning, and batched CUDA kernels, achieving over 90x faster fitting and about 97% GPU utilization compared to prior implementations. We further adapt contrastive language-image pre-training (CLIP) to 2DGS by reusing a frozen RGB-based transformer backbone with a lightweight splat-aware input stem and a perceiver resampler, training only 9.7% to 13.8% of the total parameters. On a 12.8M dataset from DataComp, GS encoders yield competitive zero-shot performance on 38 datasets from the CLIP benchmark while compressing inputs 3x to 23.5x relative to pixels. Our results establish 2DGS as a viable multimodal substrate, pinpoint architectural bottlenecks, and open a path toward representations that are both semantically powerful and transmission-efficient for edge-cloud learning.
>
---
#### [replaced 027] PIS3R: Very Large Parallax Image Stitching via Deep 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04236v3](https://arxiv.org/pdf/2508.04236v3)**

> **作者:** Muhua Zhu; Xinhao Jin; Chengbo Wang; Yongcong Zhang; Yifei Xue; Tie Ji; Yizhen Lao
>
> **摘要:** Image stitching aim to align two images taken from different viewpoints into one seamless, wider image. However, when the 3D scene contains depth variations and the camera baseline is significant, noticeable parallax occurs-meaning the relative positions of scene elements differ substantially between views. Most existing stitching methods struggle to handle such images with large parallax effectively. To address this challenge, in this paper, we propose an image stitching solution called PIS3R that is robust to very large parallax based on the novel concept of deep 3D reconstruction. First, we apply visual geometry grounded transformer to two input images with very large parallax to obtain both intrinsic and extrinsic parameters, as well as the dense 3D scene reconstruction. Subsequently, we reproject reconstructed dense point cloud onto a designated reference view using the recovered camera parameters, achieving pixel-wise alignment and generating an initial stitched image. Finally, to further address potential artifacts such as holes or noise in the initial stitching, we propose a point-conditioned image diffusion module to obtain the refined result.Compared with existing methods, our solution is very large parallax tolerant and also provides results that fully preserve the geometric integrity of all pixels in the 3D photogrammetric context, enabling direct applicability to downstream 3D vision tasks such as SfM. Experimental results demonstrate that the proposed algorithm provides accurate stitching results for images with very large parallax, and outperforms the existing methods qualitatively and quantitatively.
>
---
#### [replaced 028] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events
- **分类: cs.CV; cs.CL**

- **简介: 提出RSCC数据集，含6万+灾前/后图像对及描述文本，解决遥感灾害监测中时序与语义缺失问题，推动视觉-语言模型在双时相分析中的应用。**

- **链接: [https://arxiv.org/pdf/2509.01907v5](https://arxiv.org/pdf/2509.01907v5)**

> **作者:** Zhenyuan Chen; Chenxi Wang; Ningyu Zhang; Feng Zhang
>
> **备注:** Accepted by NeurIPS 2025 Dataset and Benchmark Track
>
> **摘要:** Remote sensing is critical for disaster monitoring, yet existing datasets lack temporal image pairs and detailed textual annotations. While single-snapshot imagery dominates current resources, it fails to capture dynamic disaster impacts over time. To address this gap, we introduce the Remote Sensing Change Caption (RSCC) dataset, a large-scale benchmark comprising 62,351 pre-/post-disaster image pairs (spanning earthquakes, floods, wildfires, and more) paired with rich, human-like change captions. By bridging the temporal and semantic divide in remote sensing data, RSCC enables robust training and evaluation of vision-language models for disaster-aware bi-temporal understanding. Our results highlight RSCC's ability to facilitate detailed disaster-related analysis, paving the way for more accurate, interpretable, and scalable vision-language applications in remote sensing. Code and dataset are available at https://github.com/Bili-Sakura/RSCC.
>
---
#### [replaced 029] Seeing Structural Failure Before it Happens: An Image-Based Physics-Informed Neural Network (PINN) for Spaghetti Bridge Load Prediction
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.23117v4](https://arxiv.org/pdf/2510.23117v4)**

> **作者:** Omer Jauhar Khan; Sudais Khan; Hafeez Anwar; Shahzeb Khan; Shams Ul Arifeen; Farman Ullah
>
> **备注:** 14 pages, 21 figures. Preprint
>
> **摘要:** Physics Informed Neural Networks (PINNs) are gaining attention for their ability to embed physical laws into deep learning models, which is particularly useful in structural engineering tasks with limited data. This paper aims to explore the use of PINNs to predict the weight of small scale spaghetti bridges, a task relevant to understanding load limits and potential failure modes in simplified structural models. Our proposed framework incorporates physics-based constraints to the prediction model for improved performance. In addition to standard PINNs, we introduce a novel architecture named Physics Informed Kolmogorov Arnold Network (PIKAN), which blends universal function approximation theory with physical insights. The structural parameters provided as input to the model are collected either manually or through computer vision methods. Our dataset includes 15 real bridges, augmented to 100 samples, and our best model achieves an $R^2$ score of 0.9603 and a mean absolute error (MAE) of 10.50 units. From applied perspective, we also provide a web based interface for parameter entry and prediction. These results show that PINNs can offer reliable estimates of structural weight, even with limited data, and may help inform early stage failure analysis in lightweight bridge designs. The complete data and code are available at https://github.com/OmerJauhar/PINNS-For-Spaghetti-Bridges.
>
---
#### [replaced 030] Steering Vision-Language Pre-trained Models for Incremental Face Presentation Attack Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19022v2](https://arxiv.org/pdf/2512.19022v2)**

> **作者:** Haoze Li; Jie Zhang; Guoying Zhao; Stephen Lin; Shiguang Shan
>
> **摘要:** Face Presentation Attack Detection (PAD) demands incremental learning (IL) to combat evolving spoofing tactics and domains. Privacy regulations, however, forbid retaining past data, necessitating rehearsal-free IL (RF-IL). Vision-Language Pre-trained (VLP) models, with their prompt-tunable cross-modal representations, enable efficient adaptation to new spoofing styles and domains. Capitalizing on this strength, we propose \textbf{SVLP-IL}, a VLP-based RF-IL framework that balances stability and plasticity via \textit{Multi-Aspect Prompting} (MAP) and \textit{Selective Elastic Weight Consolidation} (SEWC). MAP isolates domain dependencies, enhances distribution-shift sensitivity, and mitigates forgetting by jointly exploiting universal and domain-specific cues. SEWC selectively preserves critical weights from previous tasks, retaining essential knowledge while allowing flexibility for new adaptations. Comprehensive experiments across multiple PAD benchmarks show that SVLP-IL significantly reduces catastrophic forgetting and enhances performance on unseen domains. SVLP-IL offers a privacy-compliant, practical solution for robust lifelong PAD deployment in RF-IL settings.
>
---
#### [replaced 031] BevSplat: Resolving Height Ambiguity via Feature-Based Gaussian Primitives for Weakly-Supervised Cross-View Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.09080v4](https://arxiv.org/pdf/2502.09080v4)**

> **作者:** Qiwei Wang; Shaoxun Wu; Yujiao Shi
>
> **摘要:** This paper addresses the problem of weakly supervised cross-view localization, where the goal is to estimate the pose of a ground camera relative to a satellite image with noisy ground truth annotations. A common approach to bridge the cross-view domain gap for pose estimation is Bird's-Eye View (BEV) synthesis. However, existing methods struggle with height ambiguity due to the lack of depth information in ground images and satellite height maps. Previous solutions either assume a flat ground plane or rely on complex models, such as cross-view transformers. We propose BevSplat, a novel method that resolves height ambiguity by using feature-based Gaussian primitives. Each pixel in the ground image is represented by a 3D Gaussian with semantic and spatial features, which are synthesized into a BEV feature map for relative pose estimation. Additionally, to address challenges with panoramic query images, we introduce an icosphere-based supervision strategy for the Gaussian primitives. We validate our method on the widely used KITTI and VIGOR datasets, which include both pinhole and panoramic query images. Experimental results show that BevSplat significantly improves localization accuracy over prior approaches.
>
---
#### [replaced 032] DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15713v2](https://arxiv.org/pdf/2512.15713v2)**

> **作者:** Lunbin Zeng; Jingfeng Yao; Bencheng Liao; Hongyuan Tao; Wenyu Liu; Xinggang Wang
>
> **备注:** 11 pages, 5 figures, conference or other essential info
>
> **摘要:** In recent multimodal research, the diffusion paradigm has emerged as a promising alternative to the autoregressive paradigm (AR), owing to its unique decoding advantages. However, due to the capability limitations of the base diffusion language model, the performance of the diffusion vision language model (dVLM) still lags significantly behind that of mainstream models. This leads to a simple yet fundamental question: Is it possible to construct dVLMs based on existing powerful AR models? In response, we propose DiffusionVL, a dVLM family that could be translated from any powerful AR models. Through simple fine-tuning, we successfully adapt AR pre-trained models into the diffusion paradigm. This approach yields two key observations: (1) The paradigm shift from AR-based multimodal models to diffusion is remarkably effective. (2) Direct conversion of an AR language model to a dVLM is also feasible, achieving performance competitive with LLaVA-style visual-instruction-tuning. Further, we introduce a block-decoding design into dVLMs that supports arbitrary-length generation and KV cache reuse, achieving a significant inference speedup. We conduct a large number of experiments. Despite training with less than 5% of the data required by prior methods, DiffusionVL achieves a comprehensive performance improvement-a 34.4% gain on the MMMU-Pro (vision) bench and 37.5% gain on the MME (Cog.) bench-alongside a 2x inference speedup. The model and code are released at https://github.com/hustvl/DiffusionVL.
>
---
#### [replaced 033] Parameter Efficient Continual Learning with Dynamic Low-Rank Adaptation
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.11998v3](https://arxiv.org/pdf/2505.11998v3)**

> **作者:** Prashant Shivaram Bhat; Shakib Yazdani; Elahe Arani; Bahram Zonooz
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** Catastrophic forgetting has remained a critical challenge for deep neural networks in Continual Learning (CL) as it undermines consolidated knowledge when learning new tasks. Parameter efficient fine tuning CL techniques are gaining traction for their effectiveness in addressing catastrophic forgetting with a lightweight training schedule while avoiding degradation of consolidated knowledge in pre-trained models. However, low rank adapters (LoRA) in these approaches are highly sensitive to rank selection which can lead to sub-optimal resource allocation and performance. To this end, we introduce PEARL, a rehearsal-free CL framework that entails dynamic rank allocation for LoRA components during CL training. Specifically, PEARL leverages reference task weights and adaptively determines the rank of task-specific LoRA components based on the current tasks' proximity to reference task weights in parameter space. To demonstrate the versatility of PEARL, we evaluate it across three vision architectures (ResNet, Separable Convolutional Network and Vision Transformer) and a multitude of CL scenarios, and show that PEARL outperforms all considered baselines by a large margin.
>
---
#### [replaced 034] On the Design of One-step Diffusion via Shortcutting Flow Paths
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.11831v3](https://arxiv.org/pdf/2512.11831v3)**

> **作者:** Haitao Lin; Peiyan Hu; Minsi Ren; Zhifeng Gao; Zhi-Ming Ma; Guolin ke; Tailin Wu; Stan Z. Li
>
> **备注:** 10 pages of main body, conference paper
>
> **摘要:** Recent advances in few-step diffusion models have demonstrated their efficiency and effectiveness by shortcutting the probabilistic paths of diffusion models, especially in training one-step diffusion models from scratch (\emph{a.k.a.} shortcut models). However, their theoretical derivation and practical implementation are often closely coupled, which obscures the design space. To address this, we propose a common design framework for representative shortcut models. This framework provides theoretical justification for their validity and disentangles concrete component-level choices, thereby enabling systematic identification of improvements. With our proposed improvements, the resulting one-step model achieves a new state-of-the-art FID50k of 2.85 on ImageNet-256x256 under the classifier-free guidance setting with one step generation, and further reaches FID50k of 2.53 with 2x training steps. Remarkably, the model requires no pre-training, distillation, or curriculum learning. We believe our work lowers the barrier to component-level innovation in shortcut models and facilitates principled exploration of their design space.
>
---
#### [replaced 035] ChainReaction: Causal Chain-Guided Reasoning for Modular and Explainable Causal-Why Video Question Answering
- **分类: cs.CV; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 提出ChainReaction模型，用因果链解耦视频问答中的推理与答案生成，提升可解释性与性能，支持跨领域复用。**

- **链接: [https://arxiv.org/pdf/2508.21010v2](https://arxiv.org/pdf/2508.21010v2)**

> **作者:** Paritosh Parmar; Eric Peh; Basura Fernando
>
> **备注:** Project page: https://paritoshparmar.github.io/chainreaction/
>
> **摘要:** Existing Causal-Why Video Question Answering (VideoQA) models often struggle with higher-order reasoning, relying on opaque, monolithic pipelines that entangle video understanding, causal inference, and answer generation. These black-box approaches offer limited interpretability and tend to depend on shallow heuristics. We propose a novel, modular paradigm that explicitly decouples causal reasoning from answer generation, introducing natural language causal chains as interpretable intermediate representations. Inspired by human cognitive models, these structured cause-effect sequences bridge low-level video content with high-level causal reasoning, enabling transparent and logically coherent inference. Our two-stage architecture comprises a Causal Chain Extractor (CCE) that generates causal chains from video-question pairs, and a Causal Chain-Driven Answerer (CCDA) that derives answers grounded in these chains. To address the lack of annotated reasoning traces, we introduce a scalable method for generating accurate causal chains from existing datasets. We construct human verified causal chains for 46K samples. We also propose CauCo, a new evaluation metric for causality-oriented captioning. Experiments on three large-scale benchmarks demonstrate that our approach not only outperforms state-of-the-art models, but also yields substantial gains in explainability, user trust, and generalization -- positioning the CCE as a reusable causal reasoning engine across diverse domains. Project page: https://paritoshparmar.github.io/chainreaction/
>
---
#### [replaced 036] AD-R1: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving with Impartial World Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20325v2](https://arxiv.org/pdf/2511.20325v2)**

> **作者:** Tianyi Yan; Tao Tang; Xingtai Gui; Yongkang Li; Jiasen Zhesng; Weiyao Huang; Lingdong Kong; Wencheng Han; Xia Zhou; Xueyang Zhang; Yifei Zhan; Kun Zhan; Cheng-zhong Xu; Jianbing Shen
>
> **摘要:** End-to-end models for autonomous driving hold the promise of learning complex behaviors directly from sensor data, but face critical challenges in safety and handling long-tail events. Reinforcement Learning (RL) offers a promising path to overcome these limitations, yet its success in autonomous driving has been elusive. We identify a fundamental flaw hindering this progress: a deep seated optimistic bias in the world models used for RL. To address this, we introduce a framework for post-training policy refinement built around an Impartial World Model. Our primary contribution is to teach this model to be honest about danger. We achieve this with a novel data synthesis pipeline, Counterfactual Synthesis, which systematically generates a rich curriculum of plausible collisions and off-road events. This transforms the model from a passive scene completer into a veridical forecaster that remains faithful to the causal link between actions and outcomes. We then integrate this Impartial World Model into our closed-loop RL framework, where it serves as an internal critic. During refinement, the agent queries the critic to ``dream" of the outcomes for candidate actions. We demonstrate through extensive experiments, including on a new Risk Foreseeing Benchmark, that our model significantly outperforms baselines in predicting failures. Consequently, when used as a critic, it enables a substantial reduction in safety violations in challenging simulations, proving that teaching a model to dream of danger is a critical step towards building truly safe and intelligent autonomous agents.
>
---
#### [replaced 037] AGENet: Adaptive Edge-aware Geodesic Distance Learning for Few-Shot Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11662v2](https://arxiv.org/pdf/2511.11662v2)**

> **作者:** Ziyuan Gao
>
> **备注:** Accepted for publication in WACV 2026 (Round 2)
>
> **摘要:** Medical image segmentation requires large annotated datasets, creating a significant bottleneck for clinical applications. While few-shot segmentation methods can learn from minimal examples, existing approaches demonstrate suboptimal performance in precise boundary delineation for medical images, particularly when anatomically similar regions appear without sufficient spatial context. We propose AGENet (Adaptive Geodesic Edge-aware Network), a novel framework that incorporates spatial relationships through edge-aware geodesic distance learning. Our key insight is that medical structures follow predictable geometric patterns that can guide prototype extraction even with limited training data. Unlike methods relying on complex architectural components or heavy neural networks, our approach leverages computationally lightweight geometric modeling. The framework combines three main components: (1) An edge-aware geodesic distance learning module that respects anatomical boundaries through iterative Fast Marching refinement, (2) adaptive prototype extraction that captures both global structure and local boundary details via spatially-weighted aggregation, and (3) adaptive parameter learning that automatically adjusts to different organ characteristics. Extensive experiments across diverse medical imaging datasets demonstrate improvements over state-of-the-art methods. Notably, our method reduces boundary errors compared to existing approaches while maintaining computational efficiency, making it highly suitable for clinical applications requiring precise segmentation with limited annotated data.
>
---
#### [replaced 038] Deep Kronecker Network
- **分类: stat.ML; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2210.13327v2](https://arxiv.org/pdf/2210.13327v2)**

> **作者:** Long Feng; Guang Yang
>
> **摘要:** We propose Deep Kronecker Network (DKN), a novel framework designed for analyzing medical imaging data, such as MRI, fMRI, CT, etc. Medical imaging data is different from general images in at least two aspects: i) sample size is usually much more limited, ii) model interpretation is more of a concern compared to outcome prediction. Due to its unique nature, general methods, such as convolutional neural network (CNN), are difficult to be directly applied. As such, we propose DKN, that is able to i) adapt to low sample size limitation, ii) provide desired model interpretation, and iii) achieve the prediction power as CNN. The DKN is general in the sense that it not only works for both matrix and (high-order) tensor represented image data, but also could be applied to both discrete and continuous outcomes. The DKN is built on a Kronecker product structure and implicitly imposes a piecewise smooth property on coefficients. Moreover, the Kronecker structure can be written into a convolutional form, so DKN also resembles a CNN, particularly, a fully convolutional network (FCN). Furthermore, we prove that with an alternating minimization algorithm, the solutions of DKN are guaranteed to converge to the truth geometrically even if the objective function is highly nonconvex. Interestingly, the DKN is also highly connected to the tensor regression framework proposed by Zhou et al. (2010), where a CANDECOMP/PARAFAC (CP) low-rank structure is imposed on tensor coefficients. Finally, we conduct both classification and regression analyses using real MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) to demonstrate the effectiveness of DKN.
>
---
