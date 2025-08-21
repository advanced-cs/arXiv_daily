# 计算机视觉 cs.CV

- **最新发布 97 篇**

- **更新 60 篇**

## 最新发布

#### [new 001] 6-DoF Object Tracking with Event-based Optical Flow and Frames
- **分类: cs.CV**

- **简介: 本文提出结合事件流光流与RGB全局姿态估计器，实现高速下6-DoF物体跟踪，解决传统相机因帧率及运动模糊导致的跟踪难题。**

- **链接: [http://arxiv.org/pdf/2508.14776v1](http://arxiv.org/pdf/2508.14776v1)**

> **作者:** Zhichao Li; Arren Glover; Chiara Bartolozzi; Lorenzo Natale
>
> **摘要:** Tracking the position and orientation of objects in space (i.e., in 6-DoF) in real time is a fundamental problem in robotics for environment interaction. It becomes more challenging when objects move at high-speed due to frame rate limitations in conventional cameras and motion blur. Event cameras are characterized by high temporal resolution, low latency and high dynamic range, that can potentially overcome the impacts of motion blur. Traditional RGB cameras provide rich visual information that is more suitable for the challenging task of single-shot object pose estimation. In this work, we propose using event-based optical flow combined with an RGB based global object pose estimator for 6-DoF pose tracking of objects at high-speed, exploiting the core advantages of both types of vision sensors. Specifically, we propose an event-based optical flow algorithm for object motion measurement to implement an object 6-DoF velocity tracker. By integrating the tracked object 6-DoF velocity with low frequency estimated pose from the global pose estimator, the method can track pose when objects move at high-speed. The proposed algorithm is tested and validated on both synthetic and real world data, demonstrating its effectiveness, especially in high-speed motion scenarios.
>
---
#### [new 002] Tooth-Diffusion: Guided 3D CBCT Synthesis with Fine-Grained Tooth Conditioning
- **分类: cs.CV; cs.AI**

- **简介: 本文提出条件扩散框架，通过牙齿级二进制属性实现精细控制的3D CBCT生成，解决解剖真实性和局部修改难题，用于手术规划与数据增强。**

- **链接: [http://arxiv.org/pdf/2508.14276v1](http://arxiv.org/pdf/2508.14276v1)**

> **作者:** Said Djafar Said; Torkan Gholamalizadeh; Mostafa Mehdipour Ghazi
>
> **备注:** MICCAI 2025 Workshop on Oral and Dental Image Analysis (ODIN)
>
> **摘要:** Despite the growing importance of dental CBCT scans for diagnosis and treatment planning, generating anatomically realistic scans with fine-grained control remains a challenge in medical image synthesis. In this work, we propose a novel conditional diffusion framework for 3D dental volume generation, guided by tooth-level binary attributes that allow precise control over tooth presence and configuration. Our approach integrates wavelet-based denoising diffusion, FiLM conditioning, and masked loss functions to focus learning on relevant anatomical structures. We evaluate the model across diverse tasks, such as tooth addition, removal, and full dentition synthesis, using both paired and distributional similarity metrics. Results show strong fidelity and generalization with low FID scores, robust inpainting performance, and SSIM values above 0.91 even on unseen scans. By enabling realistic, localized modification of dentition without rescanning, this work opens opportunities for surgical planning, patient communication, and targeted data augmentation in dental AI workflows. The codes are available at: https://github.com/djafar1/tooth-diffusion.
>
---
#### [new 003] Seeing Further on the Shoulders of Giants: Knowledge Inheritance for Vision Foundation Models
- **分类: cs.CV**

- **简介: 该论文旨在构建通用视觉基础模型（VFM），解决传统数据驱动方法需大量标注数据与硬件资源的瓶颈。通过联合知识迁移与保留，统一多预训练模型于共享空间，减少分布差异影响，无需大规模数据即可继承教师知识，提升跨任务泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.14707v1](http://arxiv.org/pdf/2508.14707v1)**

> **作者:** Jiabo Huang; Chen Chen; Lingjuan Lyu
>
> **备注:** Technical report
>
> **摘要:** Vision foundation models (VFMs) are predominantly developed using data-centric methods. These methods require training on vast amounts of data usually with high-quality labels, which poses a bottleneck for most institutions that lack both large-scale data and high-end GPUs. On the other hand, many open-source vision models have been pretrained on domain-specific data, enabling them to distill and represent core knowledge in a form that is transferable across diverse applications. Even though these models are highly valuable assets, they remain largely under-explored in empowering the development of a general-purpose VFM. In this paper, we presents a new model-driven approach for training VFMs through joint knowledge transfer and preservation. Our method unifies multiple pre-trained teacher models in a shared latent space to mitigate the ``imbalanced transfer'' issue caused by their distributional gaps. Besides, we introduce a knowledge preservation strategy to take a general-purpose teacher as a knowledge base for integrating knowledge from the remaining purpose-specific teachers using an adapter module. By unifying and aggregating existing models, we build a powerful VFM to inherit teachers' expertise without needing to train on a large amount of labeled data. Our model not only provides generalizable visual features, but also inherently supports multiple downstream tasks. Extensive experiments demonstrate that our VFM outperforms existing data-centric models across four fundamental vision tasks, including image classification, object detection, semantic and instance segmentation.
>
---
#### [new 004] Tinker: Diffusion's Gift to 3D--Multi-View Consistent Editing From Sparse Inputs without Per-Scene Optimization
- **分类: cs.CV**

- **简介: 论文提出Tinker框架，解决多视角一致的3D编辑问题，无需场景优化即可从稀疏输入（如1-2张图）生成高质量多视角编辑结果。通过预训练扩散模型和新组件（引用编辑器、视频合成器）实现状态-of-the-art性能。**

- **链接: [http://arxiv.org/pdf/2508.14811v1](http://arxiv.org/pdf/2508.14811v1)**

> **作者:** Canyu Zhao; Xiaoman Li; Tianjian Feng; Zhiyue Zhao; Hao Chen; Chunhua Shen
>
> **备注:** Project webpage: https://aim-uofa.github.io/Tinker
>
> **摘要:** We introduce Tinker, a versatile framework for high-fidelity 3D editing that operates in both one-shot and few-shot regimes without any per-scene finetuning. Unlike prior techniques that demand extensive per-scene optimization to ensure multi-view consistency or to produce dozens of consistent edited input views, Tinker delivers robust, multi-view consistent edits from as few as one or two images. This capability stems from repurposing pretrained diffusion models, which unlocks their latent 3D awareness. To drive research in this space, we curate the first large-scale multi-view editing dataset and data pipeline, spanning diverse scenes and styles. Building on this dataset, we develop our framework capable of generating multi-view consistent edited views without per-scene training, which consists of two novel components: (1) Referring multi-view editor: Enables precise, reference-driven edits that remain coherent across all viewpoints. (2) Any-view-to-video synthesizer: Leverages spatial-temporal priors from video diffusion to perform high-quality scene completion and novel-view generation even from sparse inputs. Through extensive experiments, Tinker significantly reduces the barrier to generalizable 3D content creation, achieving state-of-the-art performance on editing, novel-view synthesis, and rendering enhancement tasks. We believe that Tinker represents a key step towards truly scalable, zero-shot 3D editing. Project webpage: https://aim-uofa.github.io/Tinker
>
---
#### [new 005] Controllable Latent Space Augmentation for Digital Pathology
- **分类: cs.CV**

- **简介: 论文解决数字病理学全切片图像分析中的高分辨率与数据稀缺问题，提出可控潜在空间增强方法HistAug，提升多实例学习性能。**

- **链接: [http://arxiv.org/pdf/2508.14588v1](http://arxiv.org/pdf/2508.14588v1)**

> **作者:** Sofiène Boutaj; Marin Scalbert; Pierre Marza; Florent Couzinie-Devy; Maria Vakalopoulou; Stergios Christodoulidis
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Whole slide image (WSI) analysis in digital pathology presents unique challenges due to the gigapixel resolution of WSIs and the scarcity of dense supervision signals. While Multiple Instance Learning (MIL) is a natural fit for slide-level tasks, training robust models requires large and diverse datasets. Even though image augmentation techniques could be utilized to increase data variability and reduce overfitting, implementing them effectively is not a trivial task. Traditional patch-level augmentation is prohibitively expensive due to the large number of patches extracted from each WSI, and existing feature-level augmentation methods lack control over transformation semantics. We introduce HistAug, a fast and efficient generative model for controllable augmentations in the latent space for digital pathology. By conditioning on explicit patch-level transformations (e.g., hue, erosion), HistAug generates realistic augmented embeddings while preserving initial semantic information. Our method allows the processing of a large number of patches in a single forward pass efficiently, while at the same time consistently improving MIL model performance. Experiments across multiple slide-level tasks and diverse organs show that HistAug outperforms existing methods, particularly in low-data regimes. Ablation studies confirm the benefits of learned transformations over noise-based perturbations and highlight the importance of uniform WSI-wise augmentation. Code is available at https://github.com/MICS-Lab/HistAug.
>
---
#### [new 006] SMTrack: End-to-End Trained Spiking Neural Networks for Multi-Object Tracking in RGB Videos
- **分类: cs.CV**

- **简介: 论文提出SMTrack，用于RGB视频的多目标跟踪，解决SNN在复杂场景下的应用难题，通过自适应损失和TrackTrack模块提升性能，实验表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.14607v1](http://arxiv.org/pdf/2508.14607v1)**

> **作者:** Pengzhi Zhong; Xinzhe Wang; Dan Zeng; Qihua Zhou; Feixiang He; Shuiwang Li
>
> **摘要:** Brain-inspired Spiking Neural Networks (SNNs) exhibit significant potential for low-power computation, yet their application in visual tasks remains largely confined to image classification, object detection, and event-based tracking. In contrast, real-world vision systems still widely use conventional RGB video streams, where the potential of directly-trained SNNs for complex temporal tasks such as multi-object tracking (MOT) remains underexplored. To address this challenge, we propose SMTrack-the first directly trained deep SNN framework for end-to-end multi-object tracking on standard RGB videos. SMTrack introduces an adaptive and scale-aware Normalized Wasserstein Distance loss (Asa-NWDLoss) to improve detection and localization performance under varying object scales and densities. Specifically, the method computes the average object size within each training batch and dynamically adjusts the normalization factor, thereby enhancing sensitivity to small objects. For the association stage, we incorporate the TrackTrack identity module to maintain robust and consistent object trajectories. Extensive evaluations on BEE24, MOT17, MOT20, and DanceTrack show that SMTrack achieves performance on par with leading ANN-based MOT methods, advancing robust and accurate SNN-based tracking in complex scenarios.
>
---
#### [new 007] A Survey on Video Anomaly Detection via Deep Learning: Human, Vehicle, and Environment
- **分类: cs.CV; cs.AI**

- **简介: 本文综述视频异常检测（VAD）在人类、车辆和环境场景中的研究进展，系统分析了深度学习方法在不同监督级别和自适应学习中的应用，总结了当前方法的贡献与局限，为理论与实际应用提供参考。**

- **链接: [http://arxiv.org/pdf/2508.14203v1](http://arxiv.org/pdf/2508.14203v1)**

> **作者:** Ghazal Alinezhad Noghre; Armin Danesh Pazho; Hamed Tabkhi
>
> **摘要:** Video Anomaly Detection (VAD) has emerged as a pivotal task in computer vision, with broad relevance across multiple fields. Recent advances in deep learning have driven significant progress in this area, yet the field remains fragmented across domains and learning paradigms. This survey offers a comprehensive perspective on VAD, systematically organizing the literature across various supervision levels, as well as adaptive learning methods such as online, active, and continual learning. We examine the state of VAD across three major application categories: human-centric, vehicle-centric, and environment-centric scenarios, each with distinct challenges and design considerations. In doing so, we identify fundamental contributions and limitations of current methodologies. By consolidating insights from subfields, we aim to provide the community with a structured foundation for advancing both theoretical understanding and real-world applicability of VAD systems. This survey aims to support researchers by providing a useful reference, while also drawing attention to the broader set of open challenges in anomaly detection, including both fundamental research questions and practical obstacles to real-world deployment.
>
---
#### [new 008] Inter-Class Relational Loss for Small Object Detection: A Case Study on License Plates
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对小目标检测中的梯度更新问题，提出基于类间关系的损失函数，利用车牌与车辆的空间关系增强小目标学习，构建新数据集并提升YOLOv12-T和UAV-DETR的mAP性能。**

- **链接: [http://arxiv.org/pdf/2508.14343v1](http://arxiv.org/pdf/2508.14343v1)**

> **作者:** Dian Ning; Dong Seog Han
>
> **摘要:** In one-stage multi-object detection tasks, various intersection over union (IoU)-based solutions aim at smooth and stable convergence near the targets during training. However, IoU-based losses fail to correctly update the gradient of small objects due to an extremely flat gradient. During the update of multiple objects, the learning of small objects' gradients suffers more because of insufficient gradient updates. Therefore, we propose an inter-class relational loss to efficiently update the gradient of small objects while not sacrificing the learning efficiency of other objects based on the simple fact that an object has a spatial relationship to another object (e.g., a car plate is attached to a car in a similar position). When the predicted car plate's bounding box is not within its car, a loss punishment is added to guide the learning, which is inversely proportional to the overlapped area of the car's and predicted car plate's bounding box. By leveraging the spatial relationship at the inter-class level, the loss guides small object predictions using larger objects and enhances latent information in deeper feature maps. In this paper, we present twofold contributions using license plate detection as a case study: (1) a new small vehicle multi-license plate dataset (SVMLP), featuring diverse real-world scenarios with high-quality annotations; and (2) a novel inter-class relational loss function designed to promote effective detection performance. We highlight the proposed ICR loss penalty can be easily added to existing IoU-based losses and enhance the performance. These contributions improve the standard mean Average Precision (mAP) metric, achieving gains of 10.3% and 1.6% in mAP$^{\text{test}}_{50}$ for YOLOv12-T and UAV-DETR, respectively, without any additional hyperparameter tuning. Code and dataset will be available soon.
>
---
#### [new 009] MF-LPR$^2$: Multi-Frame License Plate Image Restoration and Recognition using Optical Flow
- **分类: cs.CV; cs.AI**

- **简介: 论文针对多帧车牌图像因低分辨率、运动模糊等导致的识别难题，提出MF-LPR²框架，通过光流对齐与自研算法修正帧间误差，构建RLPR数据集，显著提升修复质量与识别准确率（86.44%）。**

- **链接: [http://arxiv.org/pdf/2508.14797v1](http://arxiv.org/pdf/2508.14797v1)**

> **作者:** Kihyun Na; Junseok Oh; Youngkwan Cho; Bumjin Kim; Sungmin Cho; Jinyoung Choi; Injung Kim
>
> **备注:** Accepted for publication in Computer Vision and Image Understanding (CVIU), 2025
>
> **摘要:** License plate recognition (LPR) is important for traffic law enforcement, crime investigation, and surveillance. However, license plate areas in dash cam images often suffer from low resolution, motion blur, and glare, which make accurate recognition challenging. Existing generative models that rely on pretrained priors cannot reliably restore such poor-quality images, frequently introducing severe artifacts and distortions. To address this issue, we propose a novel multi-frame license plate restoration and recognition framework, MF-LPR$^2$, which addresses ambiguities in poor-quality images by aligning and aggregating neighboring frames instead of relying on pretrained knowledge. To achieve accurate frame alignment, we employ a state-of-the-art optical flow estimator in conjunction with carefully designed algorithms that detect and correct erroneous optical flow estimations by leveraging the spatio-temporal consistency inherent in license plate image sequences. Our approach enhances both image quality and recognition accuracy while preserving the evidential content of the input images. In addition, we constructed a novel Realistic LPR (RLPR) dataset to evaluate MF-LPR$^2$. The RLPR dataset contains 200 pairs of low-quality license plate image sequences and high-quality pseudo ground-truth images, reflecting the complexities of real-world scenarios. In experiments, MF-LPR$^2$ outperformed eight recent restoration models in terms of PSNR, SSIM, and LPIPS by significant margins. In recognition, MF-LPR$^2$ achieved an accuracy of 86.44%, outperforming both the best single-frame LPR (14.04%) and the multi-frame LPR (82.55%) among the eleven baseline models. The results of ablation studies confirm that our filtering and refinement algorithms significantly contribute to these improvements.
>
---
#### [new 010] GALA: Guided Attention with Language Alignment for Open Vocabulary Gaussian Splatting
- **分类: cs.CV**

- **简介: 论文面向3D场景重建中的开放词汇理解任务，解决从2D图像中提取语言感知的细粒度3D表示问题。提出GALA框架，通过自监督对比学习生成场景特征场，并引入跨注意力模块与代码本实现语言对齐，提升2D/3D开放词汇查询性能。**

- **链接: [http://arxiv.org/pdf/2508.14278v1](http://arxiv.org/pdf/2508.14278v1)**

> **作者:** Elena Alegret Regalado; Kunyi Li; Sen Wang; Siyun Liang; Michael Niemeyer; Stefano Gasperini; Nassir Navab; Federico Tombari
>
> **摘要:** 3D scene reconstruction and understanding have gained increasing popularity, yet existing methods still struggle to capture fine-grained, language-aware 3D representations from 2D images. In this paper, we present GALA, a novel framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). GALA distills a scene-specific 3D instance feature field via self-supervised contrastive learning. To extend to generalized language feature fields, we introduce the core contribution of GALA, a cross-attention module with two learnable codebooks that encode view-independent semantic embeddings. This design not only ensures intra-instance feature similarity but also supports seamless 2D and 3D open-vocabulary queries. It reduces memory consumption by avoiding per-Gaussian high-dimensional feature learning. Extensive experiments on real-world datasets demonstrate GALA's remarkable open-vocabulary performance on both 2D and 3D.
>
---
#### [new 011] WISE-FUSE: Efficient Whole Slide Image Encoding via Coarse-to-Fine Patch Selection with VLM and LLM Knowledge Fusion
- **分类: cs.CV**

- **简介: 该论文针对计算病理学中WSI编码的高计算成本问题，提出WISE-FUSE框架，通过VLM与LLM融合，基于粗细层次补丁选择机制，高效提取诊断相关区域，显著缩短编码时间且保持诊断性能。**

- **链接: [http://arxiv.org/pdf/2508.14537v1](http://arxiv.org/pdf/2508.14537v1)**

> **作者:** Yonghan Shin; SeungKyu Kim; Won-Ki Jeong
>
> **摘要:** Whole slide images (WSIs) in computational pathology (CPath) pose a major computational challenge due to their gigapixel scale, often requiring the processing of tens to hundreds of thousands of high-resolution patches per slide. This results in prohibitive encoding costs, with preprocessing and training times extending to days or even weeks-making WSI encoding the most significant bottleneck in real-world deployment. In this work, we propose WISE-FUSE, an adaptive WSI encoding framework that leverages pathology-domain vision-language models and large language models to address this challenge by selectively processing diagnostically relevant regions. WISE-FUSE first computes similarity scores between low-resolution patches and class-specific textual descriptions using a knowledge distillation mechanism that preserves fine-grained diagnostic features. Based on these similarity scores, we select a small subset of informative regions for the target task, which quickly eliminates irrelevant patches at the coarse level. The corresponding high-resolution patches are then selectively encoded and fused with textual embeddings to reinforce diagnostic context. Extensive experiments demonstrate that WISE-FUSE reduces WSI encoding time by over threefold while achieving diagnostic performance comparable to or surpassing that of exhaustive patch processing, offering a scalable and practical solution for CPath.
>
---
#### [new 012] HyperDiff: Hypergraph Guided Diffusion Model for 3D Human Pose Estimation
- **分类: cs.CV**

- **简介: 该论文针对单目3D人体姿态估计中的深度模糊、遮挡及多尺度特征忽略问题，提出HyperDiff方法，融合扩散模型与HyperGCN，通过建模高阶关节相关性提升去噪能力，在两个数据集上取得最优性能并具备计算适应性。**

- **链接: [http://arxiv.org/pdf/2508.14431v1](http://arxiv.org/pdf/2508.14431v1)**

> **作者:** Bing Han; Yuhua Huang; Pan Gao
>
> **摘要:** Monocular 3D human pose estimation (HPE) often encounters challenges such as depth ambiguity and occlusion during the 2D-to-3D lifting process. Additionally, traditional methods may overlook multi-scale skeleton features when utilizing skeleton structure information, which can negatively impact the accuracy of pose estimation. To address these challenges, this paper introduces a novel 3D pose estimation method, HyperDiff, which integrates diffusion models with HyperGCN. The diffusion model effectively captures data uncertainty, alleviating depth ambiguity and occlusion. Meanwhile, HyperGCN, serving as a denoiser, employs multi-granularity structures to accurately model high-order correlations between joints. This improves the model's denoising capability especially for complex poses. Experimental results demonstrate that HyperDiff achieves state-of-the-art performance on the Human3.6M and MPI-INF-3DHP datasets and can flexibly adapt to varying computational resources to balance performance and efficiency.
>
---
#### [new 013] Vivid-VR: Distilling Concepts from Text-to-Video Diffusion Transformer for Photorealistic Video Restoration
- **分类: cs.CV**

- **简介: 该论文针对视频修复中的分布漂移问题，提出Vivid-VR方法，通过概念蒸馏与改进控制架构，提升纹理真实性和时间一致性。**

- **链接: [http://arxiv.org/pdf/2508.14483v1](http://arxiv.org/pdf/2508.14483v1)**

> **作者:** Haoran Bai; Xiaoxu Chen; Canqian Yang; Zongyao He; Sibin Deng; Ying Chen
>
> **摘要:** We present Vivid-VR, a DiT-based generative video restoration method built upon an advanced T2V foundation model, where ControlNet is leveraged to control the generation process, ensuring content consistency. However, conventional fine-tuning of such controllable pipelines frequently suffers from distribution drift due to limitations in imperfect multimodal alignment, resulting in compromised texture realism and temporal coherence. To tackle this challenge, we propose a concept distillation training strategy that utilizes the pretrained T2V model to synthesize training samples with embedded textual concepts, thereby distilling its conceptual understanding to preserve texture and temporal quality. To enhance generation controllability, we redesign the control architecture with two key components: 1) a control feature projector that filters degradation artifacts from input video latents to minimize their propagation through the generation pipeline, and 2) a new ControlNet connector employing a dual-branch design. This connector synergistically combines MLP-based feature mapping with cross-attention mechanism for dynamic control feature retrieval, enabling both content preservation and adaptive control signal modulation. Extensive experiments show that Vivid-VR performs favorably against existing approaches on both synthetic and real-world benchmarks, as well as AIGC videos, achieving impressive texture realism, visual vividness, and temporal consistency. The codes and checkpoints are publicly available at https://github.com/csbhr/Vivid-VR.
>
---
#### [new 014] Directed-Tokens: A Robust Multi-Modality Alignment Approach to Large Language-Vision Models
- **分类: cs.CV**

- **简介: 该论文针对大语言-视觉模型的多模态对齐问题，提出通过解决洗牌问题引入重建图像/文本顺序任务及定向token方法，结合新损失函数提升鲁棒性与跨模态对齐，达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2508.14264v1](http://arxiv.org/pdf/2508.14264v1)**

> **作者:** Thanh-Dat Truong; Huu-Thien Tran; Tran Thai Son; Bhiksha Raj; Khoa Luu
>
> **摘要:** Large multimodal models (LMMs) have gained impressive performance due to their outstanding capability in various understanding tasks. However, these models still suffer from some fundamental limitations related to robustness and generalization due to the alignment and correlation between visual and textual features. In this paper, we introduce a simple but efficient learning mechanism for improving the robust alignment between visual and textual modalities by solving shuffling problems. In particular, the proposed approach can improve reasoning capability, visual understanding, and cross-modality alignment by introducing two new tasks: reconstructing the image order and the text order into the LMM's pre-training and fine-tuning phases. In addition, we propose a new directed-token approach to capture visual and textual knowledge, enabling the capability to reconstruct the correct order of visual inputs. Then, we introduce a new Image-to-Response Guided loss to further improve the visual understanding of the LMM in its responses. The proposed approach consistently achieves state-of-the-art (SoTA) performance compared with prior LMMs on academic task-oriented and instruction-following LMM benchmarks.
>
---
#### [new 015] MoVieDrive: Multi-Modal Multi-View Urban Scene Video Generation
- **分类: cs.CV**

- **简介: 论文任务为多模态多视角城市场景视频生成，解决现有方法仅支持RGB视频且缺乏多模态数据生成的问题。提出统一扩散Transformer模型，融合共享与专用组件，通过多条件输入实现多模态视频生成，实验表明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.14327v1](http://arxiv.org/pdf/2508.14327v1)**

> **作者:** Guile Wu; David Huang; Dongfeng Bai; Bingbing Liu
>
> **备注:** Technical Report
>
> **摘要:** Video generation has recently shown superiority in urban scene synthesis for autonomous driving. Existing video generation approaches to autonomous driving primarily focus on RGB video generation and lack the ability to support multi-modal video generation. However, multi-modal data, such as depth maps and semantic maps, are crucial for holistic urban scene understanding in autonomous driving. Although it is feasible to use multiple models to generate different modalities, this increases the difficulty of model deployment and does not leverage complementary cues for multi-modal data generation. To address this problem, in this work, we propose a novel multi-modal multi-view video generation approach to autonomous driving. Specifically, we construct a unified diffusion transformer model composed of modal-shared components and modal-specific components. Then, we leverage diverse conditioning inputs to encode controllable scene structure and content cues into the unified diffusion model for multi-modal multi-view video generation. In this way, our approach is capable of generating multi-modal multi-view driving scene videos in a unified framework. Our experiments on the challenging real-world autonomous driving dataset, nuScenes, show that our approach can generate multi-modal multi-view urban scene videos with high fidelity and controllability, surpassing the state-of-the-art methods.
>
---
#### [new 016] Locality-aware Concept Bottleneck Model
- **分类: cs.CV**

- **简介: 该论文针对图像概念识别与定位任务，解决无标注CBMs定位不准确的问题，提出LCBM框架，通过原型学习与基础模型结合，提升概念空间定位精度。**

- **链接: [http://arxiv.org/pdf/2508.14562v1](http://arxiv.org/pdf/2508.14562v1)**

> **作者:** Sujin Jeon; Hyundo Lee; Eungseo Kim; Sanghack Lee; Byoung-Tak Zhang; Inwoo Hwang
>
> **备注:** 34 pages, 25 figures
>
> **摘要:** Concept bottleneck models (CBMs) are inherently interpretable models that make predictions based on human-understandable visual cues, referred to as concepts. As obtaining dense concept annotations with human labeling is demanding and costly, recent approaches utilize foundation models to determine the concepts existing in the images. However, such label-free CBMs often fail to localize concepts in relevant regions, attending to visually unrelated regions when predicting concept presence. To this end, we propose a framework, coined Locality-aware Concept Bottleneck Model (LCBM), which utilizes rich information from foundation models and adopts prototype learning to ensure accurate spatial localization of the concepts. Specifically, we assign one prototype to each concept, promoted to represent a prototypical image feature of that concept. These prototypes are learned by encouraging them to encode similar local regions, leveraging foundation models to assure the relevance of each prototype to its associated concept. Then we use the prototypes to facilitate the learning process of identifying the proper local region from which each concept should be predicted. Experimental results demonstrate that LCBM effectively identifies present concepts in the images and exhibits improved localization while maintaining comparable classification performance.
>
---
#### [new 017] FOCUS: Frequency-Optimized Conditioning of DiffUSion Models for mitigating catastrophic forgetting during Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文针对测试时适应中的灾难性遗忘问题，提出FOCUS方法，通过频率优化条件策略和轻量级Y-FPN网络，在扩散模型中保留语义信息，提升跨域适应性能并生成伪标签辅助其他方法。**

- **链接: [http://arxiv.org/pdf/2508.14437v1](http://arxiv.org/pdf/2508.14437v1)**

> **作者:** Gabriel Tjio; Jie Zhang; Xulei Yang; Yun Xing; Nhat Chung; Xiaofeng Cao; Ivor W. Tsang; Chee Keong Kwoh; Qing Guo
>
> **摘要:** Test-time adaptation enables models to adapt to evolving domains. However, balancing the tradeoff between preserving knowledge and adapting to domain shifts remains challenging for model adaptation methods, since adapting to domain shifts can induce forgetting of task-relevant knowledge. To address this problem, we propose FOCUS, a novel frequency-based conditioning approach within a diffusion-driven input-adaptation framework. Utilising learned, spatially adaptive frequency priors, our approach conditions the reverse steps during diffusion-driven denoising to preserve task-relevant semantic information for dense prediction. FOCUS leverages a trained, lightweight, Y-shaped Frequency Prediction Network (Y-FPN) that disentangles high and low frequency information from noisy images. This minimizes the computational costs involved in implementing our approach in a diffusion-driven framework. We train Y-FPN with FrequencyMix, a novel data augmentation method that perturbs the images across diverse frequency bands, which improves the robustness of our approach to diverse corruptions. We demonstrate the effectiveness of FOCUS for semantic segmentation and monocular depth estimation across 15 corruption types and three datasets, achieving state-of-the-art averaged performance. In addition to improving standalone performance, FOCUS complements existing model adaptation methods since we can derive pseudo labels from FOCUS-denoised images for additional supervision. Even under limited, intermittent supervision with the pseudo labels derived from the FOCUS denoised images, we show that FOCUS mitigates catastrophic forgetting for recent model adaptation methods.
>
---
#### [new 018] TCFNet: Bidirectional face-bone transformation via a Transformer-based coarse-to-fine point movement network
- **分类: cs.CV**

- **简介: 该论文针对正颌手术中面部骨骼点云双向变换任务，解决传统方法效率低、精度差及深度学习方法处理能力不足的问题，提出基于Transformer的TCFNet，分阶段处理并引入辅助损失提升性能。**

- **链接: [http://arxiv.org/pdf/2508.14373v1](http://arxiv.org/pdf/2508.14373v1)**

> **作者:** Runshi Zhang; Bimeng Jie; Yang He; Junchen Wang
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Computer-aided surgical simulation is a critical component of orthognathic surgical planning, where accurately simulating face-bone shape transformations is significant. The traditional biomechanical simulation methods are limited by their computational time consumption levels, labor-intensive data processing strategies and low accuracy. Recently, deep learning-based simulation methods have been proposed to view this problem as a point-to-point transformation between skeletal and facial point clouds. However, these approaches cannot process large-scale points, have limited receptive fields that lead to noisy points, and employ complex preprocessing and postprocessing operations based on registration. These shortcomings limit the performance and widespread applicability of such methods. Therefore, we propose a Transformer-based coarse-to-fine point movement network (TCFNet) to learn unique, complicated correspondences at the patch and point levels for dense face-bone point cloud transformations. This end-to-end framework adopts a Transformer-based network and a local information aggregation network (LIA-Net) in the first and second stages, respectively, which reinforce each other to generate precise point movement paths. LIA-Net can effectively compensate for the neighborhood precision loss of the Transformer-based network by modeling local geometric structures (edges, orientations and relative position features). The previous global features are employed to guide the local displacement using a gated recurrent unit. Inspired by deformable medical image registration, we propose an auxiliary loss that can utilize expert knowledge for reconstructing critical organs.Compared with the existing state-of-the-art (SOTA) methods on gathered datasets, TCFNet achieves outstanding evaluation metrics and visualization results. The code is available at https://github.com/Runshi-Zhang/TCFNet.
>
---
#### [new 019] SATURN: Autoregressive Image Generation Guided by Scene Graphs
- **分类: cs.CV**

- **简介: 该论文针对文本到图像生成中结构信息捕捉不足的问题，提出SATURN框架，通过场景图指导自回归模型生成，结合CLIP-VQ-VAE骨干与VAR变压器，提升图像质量和结构准确性。**

- **链接: [http://arxiv.org/pdf/2508.14502v1](http://arxiv.org/pdf/2508.14502v1)**

> **作者:** Thanh-Nhan Vo; Trong-Thuan Nguyen; Tam V. Nguyen; Minh-Triet Tran
>
> **备注:** Accepted to MAPR 2025
>
> **摘要:** State-of-the-art text-to-image models excel at photorealistic rendering but often struggle to capture the layout and object relationships implied by complex prompts. Scene graphs provide a natural structural prior, yet previous graph-guided approaches have typically relied on heavy GAN or diffusion pipelines, which lag behind modern autoregressive architectures in both speed and fidelity. We introduce SATURN (Structured Arrangement of Triplets for Unified Rendering Networks), a lightweight extension to VAR-CLIP that translates a scene graph into a salience-ordered token sequence, enabling a frozen CLIP-VQ-VAE backbone to interpret graph structure while fine-tuning only the VAR transformer. On the Visual Genome dataset, SATURN reduces FID from 56.45% to 21.62% and increases the Inception Score from 16.03 to 24.78, outperforming prior methods such as SG2IM and SGDiff without requiring extra modules or multi-stage training. Qualitative results further confirm improvements in object count fidelity and spatial relation accuracy, showing that SATURN effectively combines structural awareness with state-of-the-art autoregressive fidelity.
>
---
#### [new 020] Multi-Rationale Explainable Object Recognition via Contrastive Conditional Inference
- **分类: cs.CV**

- **简介: 该论文针对多理由可解释对象识别任务，提出对比条件推理框架，通过多理由基准提升分类准确性和理由质量，解决传统方法条件化不足与理由多样性的缺陷。**

- **链接: [http://arxiv.org/pdf/2508.14280v1](http://arxiv.org/pdf/2508.14280v1)**

> **作者:** Ali Rasekh; Sepehr Kazemi Ranjbar; Simon Gottschalk
>
> **摘要:** Explainable object recognition using vision-language models such as CLIP involves predicting accurate category labels supported by rationales that justify the decision-making process. Existing methods typically rely on prompt-based conditioning, which suffers from limitations in CLIP's text encoder and provides weak conditioning on explanatory structures. Additionally, prior datasets are often restricted to single, and frequently noisy, rationales that fail to capture the full diversity of discriminative image features. In this work, we introduce a multi-rationale explainable object recognition benchmark comprising datasets in which each image is annotated with multiple ground-truth rationales, along with evaluation metrics designed to offer a more comprehensive representation of the task. To overcome the limitations of previous approaches, we propose a contrastive conditional inference (CCI) framework that explicitly models the probabilistic relationships among image embeddings, category labels, and rationales. Without requiring any training, our framework enables more effective conditioning on rationales to predict accurate object categories. Our approach achieves state-of-the-art results on the multi-rationale explainable object recognition benchmark, including strong zero-shot performance, and sets a new standard for both classification accuracy and rationale quality. Together with the benchmark, this work provides a more complete framework for evaluating future models in explainable object recognition. The code will be made available online.
>
---
#### [new 021] Reliable Smoke Detection via Optical Flow-Guided Feature Fusion and Transformer-Based Uncertainty Modeling
- **分类: cs.CV**

- **简介: 该论文旨在解决复杂环境下烟雾检测可靠性问题，提出融合光学流与Transformer的不确定性建模方法，通过多阶段学习提升检测精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.14597v1](http://arxiv.org/pdf/2508.14597v1)**

> **作者:** Nitish Kumar Mahala; Muzammil Khan; Pushpendra Kumar
>
> **摘要:** Fire outbreaks pose critical threats to human life and infrastructure, necessitating high-fidelity early-warning systems that detect combustion precursors such as smoke. However, smoke plumes exhibit complex spatiotemporal dynamics influenced by illumination variability, flow kinematics, and environmental noise, undermining the reliability of traditional detectors. To address these challenges without the logistical complexity of multi-sensor arrays, we propose an information-fusion framework by integrating smoke feature representations extracted from monocular imagery. Specifically, a Two-Phase Uncertainty-Aware Shifted Windows Transformer for robust and reliable smoke detection, leveraging a novel smoke segmentation dataset, constructed via optical flow-based motion encoding, is proposed. The optical flow estimation is performed with a four-color-theorem-inspired dual-phase level-set fractional-order variational model, which preserves motion discontinuities. The resulting color-encoded optical flow maps are fused with appearance cues via a Gaussian Mixture Model to generate binary segmentation masks of the smoke regions. These fused representations are fed into the novel Shifted-Windows Transformer, which is augmented with a multi-scale uncertainty estimation head and trained under a two-phase learning regimen. First learning phase optimizes smoke detection accuracy, while during the second phase, the model learns to estimate plausibility confidence in its predictions by jointly modeling aleatoric and epistemic uncertainties. Extensive experiments using multiple evaluation metrics and comparative analysis with state-of-the-art approaches demonstrate superior generalization and robustness, offering a reliable solution for early fire detection in surveillance, industrial safety, and autonomous monitoring applications.
>
---
#### [new 022] Generalizable Engagement Estimation in Conversation via Domain Prompting and Parallel Attention
- **分类: cs.CV**

- **简介: 论文针对对话中跨领域参与度估计任务，解决泛化能力与复杂互动建模问题，提出DAPA框架，通过域提示和并行注意力模块提升模型性能，在多文化多语言基准中取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2508.14448v1](http://arxiv.org/pdf/2508.14448v1)**

> **作者:** Yangche Yu; Yin Chen; Jia Li; Peng Jia; Yu Zhang; Li Dai; Zhenzhen Hu; Meng Wang; Richang Hong
>
> **备注:** 1st Place in the Engagement Estimation Task held by MultiMediate 25
>
> **摘要:** Accurate engagement estimation is essential for adaptive human-computer interaction systems, yet robust deployment is hindered by poor generalizability across diverse domains and challenges in modeling complex interaction dynamics.To tackle these issues, we propose DAPA (Domain-Adaptive Parallel Attention), a novel framework for generalizable conversational engagement modeling. DAPA introduces a Domain Prompting mechanism by prepending learnable domain-specific vectors to the input, explicitly conditioning the model on the data's origin to facilitate domain-aware adaptation while preserving generalizable engagement representations. To capture interactional synchrony, the framework also incorporates a Parallel Cross-Attention module that explicitly aligns reactive (forward BiLSTM) and anticipatory (backward BiLSTM) states between participants.Extensive experiments demonstrate that DAPA establishes a new state-of-the-art performance on several cross-cultural and cross-linguistic benchmarks, notably achieving an absolute improvement of 0.45 in Concordance Correlation Coefficient (CCC) over a strong baseline on the NoXi-J test set. The superiority of our method was also confirmed by winning the first place in the Multi-Domain Engagement Estimation Challenge at MultiMediate'25.
>
---
#### [new 023] MoCHA-former: Moiré-Conditioned Hybrid Adaptive Transformer for Video Demoiréing
- **分类: cs.CV**

- **简介: 该论文针对视频摩尔纹退化问题，提出MoCHA-former，结合解耦与时空适应的Transformer，有效处理空间异质性、大尺度结构及时间波动，无需显式对齐。**

- **链接: [http://arxiv.org/pdf/2508.14423v1](http://arxiv.org/pdf/2508.14423v1)**

> **作者:** Jeahun Sung; Changhyun Roh; Chanho Eom; Jihyong Oh
>
> **备注:** Please visit our project page at [this http URL link](https://cmlab-korea.github.io/MoCHAformer-Demo/)
>
> **摘要:** Recent advances in portable imaging have made camera-based screen capture ubiquitous. Unfortunately, frequency aliasing between the camera's color filter array (CFA) and the display's sub-pixels induces moir\'e patterns that severely degrade captured photos and videos. Although various demoir\'eing models have been proposed to remove such moir\'e patterns, these approaches still suffer from several limitations: (i) spatially varying artifact strength within a frame, (ii) large-scale and globally spreading structures, (iii) channel-dependent statistics and (iv) rapid temporal fluctuations across frames. We address these issues with the Moir\'e Conditioned Hybrid Adaptive Transformer (MoCHA-former), which comprises two key components: Decoupled Moir\'e Adaptive Demoir\'eing (DMAD) and Spatio-Temporal Adaptive Demoir\'eing (STAD). DMAD separates moir\'e and content via a Moir\'e Decoupling Block (MDB) and a Detail Decoupling Block (DDB), then produces moir\'e-adaptive features using a Moir\'e Conditioning Block (MCB) for targeted restoration. STAD introduces a Spatial Fusion Block (SFB) with window attention to capture large-scale structures, and a Feature Channel Attention (FCA) to model channel dependence in RAW frames. To ensure temporal consistency, MoCHA-former performs implicit frame alignment without any explicit alignment module. We analyze moir\'e characteristics through qualitative and quantitative studies, and evaluate on two video datasets covering RAW and sRGB domains. MoCHA-former consistently surpasses prior methods across PSNR, SSIM, and LPIPS.
>
---
#### [new 024] Accelerating Image Classification with Graph Convolutional Neural Networks using Voronoi Diagrams
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对图像分类任务，通过结合图卷积网络（GCN）与Voronoi图，提出NVGCN模型，优化图像表示并提升分类效率与精度。**

- **链接: [http://arxiv.org/pdf/2508.14218v1](http://arxiv.org/pdf/2508.14218v1)**

> **作者:** Mustafa Mohammadi Gharasuie; Luis Rueda
>
> **备注:** 14 pages, 13 figures
>
> **摘要:** Recent advances in image classification have been significantly propelled by the integration of Graph Convolutional Networks (GCNs), offering a novel paradigm for handling complex data structures. This study introduces an innovative framework that employs GCNs in conjunction with Voronoi diagrams to peform image classification, leveraging their exceptional capability to model relational data. Unlike conventional convolutional neural networks, our approach utilizes a graph-based representation of images, where pixels or regions are treated as vertices of a graph, which are then simplified in the form of the corresponding Delaunay triangulations. Our model yields significant improvement in pre-processing time and classification accuracy on several benchmark datasets, surpassing existing state-of-the-art models, especially in scenarios that involve complex scenes and fine-grained categories. The experimental results, validated via cross-validation, underscore the potential of integrating GCNs with Voronoi diagrams in advancing image classification tasks. This research contributes to the field by introducing a novel approach to image classification, while opening new avenues for developing graph-based learning paradigms in other domains of computer vision and non-structured data. In particular, we have proposed a new version of the GCN in this paper, namely normalized Voronoi Graph Convolution Network (NVGCN), which is faster than the regular GCN.
>
---
#### [new 025] Incremental Object Detection with Prompt-based Methods
- **分类: cs.CV**

- **简介: 论文研究增量物体检测中的提示方法，解决其泛化性问题，通过分析不同方法并实验，发现结合提示与数据重放的方法有效。**

- **链接: [http://arxiv.org/pdf/2508.14599v1](http://arxiv.org/pdf/2508.14599v1)**

> **作者:** Matthias Neuwirth-Trapp; Maarten Bieshaar; Danda Pani Paudel; Luc Van Gool
>
> **备注:** Accepted to ICCV Workshops 2025
>
> **摘要:** Visual prompt-based methods have seen growing interest in incremental learning (IL) for image classification. These approaches learn additional embedding vectors while keeping the model frozen, making them efficient to train. However, no prior work has applied such methods to incremental object detection (IOD), leaving their generalizability unclear. In this paper, we analyze three different prompt-based methods under a complex domain-incremental learning setting. We additionally provide a wide range of reference baselines for comparison. Empirically, we show that the prompt-based approaches we tested underperform in this setting. However, a strong yet practical method, combining visual prompts with replaying a small portion of previous data, achieves the best results. Together with additional experiments on prompt length and initialization, our findings offer valuable insights for advancing prompt-based IL in IOD.
>
---
#### [new 026] FastTracker: Real-Time and Accurate Visual Tracking
- **分类: cs.CV**

- **简介: 该论文针对多目标跟踪中的泛化问题，提出通用框架处理车辆跟踪。通过遮挡感知重识别与道路结构优化，构建新数据集，实现复杂场景下高精度实时跟踪。**

- **链接: [http://arxiv.org/pdf/2508.14370v1](http://arxiv.org/pdf/2508.14370v1)**

> **作者:** Hamidreza Hashempoor; Yu Dong Hwang
>
> **摘要:** Conventional multi-object tracking (MOT) systems are predominantly designed for pedestrian tracking and often exhibit limited generalization to other object categories. This paper presents a generalized tracking framework capable of handling multiple object types, with a particular emphasis on vehicle tracking in complex traffic scenes. The proposed method incorporates two key components: (1) an occlusion-aware re-identification mechanism that enhances identity preservation for heavily occluded objects, and (2) a road-structure-aware tracklet refinement strategy that utilizes semantic scene priors such as lane directions, crosswalks, and road boundaries to improve trajectory continuity and accuracy. In addition, we introduce a new benchmark dataset comprising diverse vehicle classes with frame-level tracking annotations, specifically curated to support evaluation of vehicle-focused tracking methods. Extensive experimental results demonstrate that the proposed approach achieves robust performance on both the newly introduced dataset and several public benchmarks, highlighting its effectiveness in general-purpose object tracking. While our framework is designed for generalized multi-class tracking, it also achieves strong performance on conventional benchmarks, with HOTA scores of 66.4 on MOT17 and 65.7 on MOT20 test sets. Code and Benchmark are available: github.com/Hamidreza-Hashempoor/FastTracker, huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark.
>
---
#### [new 027] Deep Learning for Taxol Exposure Analysis: A New Cell Image Dataset and Attention-Based Baseline Model
- **分类: cs.CV**

- **简介: 该论文任务为基于细胞图像的Taxol浓度分类，解决缺乏公开数据集与有效模型的问题，通过构建新数据集及提出ResAttention-KNN模型实现自动化分析。**

- **链接: [http://arxiv.org/pdf/2508.14349v1](http://arxiv.org/pdf/2508.14349v1)**

> **作者:** Sean Fletcher; Gabby Scott; Douglas Currie; Xin Zhang; Yuqi Song; Bruce MacLeod
>
> **备注:** Accepted to the 2025 IEEE International Workshop on Foundations of Machine Learning for Drug Safety (FMLDS), to appear in November 2025
>
> **摘要:** Monitoring the effects of the chemotherapeutic agent Taxol at the cellular level is critical for both clinical evaluation and biomedical research. However, existing detection methods require specialized equipment, skilled personnel, and extensive sample preparation, making them expensive, labor-intensive, and unsuitable for high-throughput or real-time analysis. Deep learning approaches have shown great promise in medical and biological image analysis, enabling automated, high-throughput assessment of cellular morphology. Yet, no publicly available dataset currently exists for automated morphological analysis of cellular responses to Taxol exposure. To address this gap, we introduce a new microscopy image dataset capturing C6 glioma cells treated with varying concentrations of Taxol. To provide an effective solution for Taxol concentration classification and establish a benchmark for future studies on this dataset, we propose a baseline model named ResAttention-KNN, which combines a ResNet-50 with Convolutional Block Attention Modules and uses a k-Nearest Neighbors classifier in the learned embedding space. This model integrates attention-based refinement and non-parametric classification to enhance robustness and interpretability. Both the dataset and implementation are publicly released to support reproducibility and facilitate future research in vision-based biomedical analysis.
>
---
#### [new 028] Learning Point Cloud Representations with Pose Continuity for Depth-Based Category-Level 6D Object Pose Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文针对类别级6D姿态估计中姿态不连续问题，提出HRC-Pose框架，通过对比学习学习点云表示，分解姿态为旋转和翻译，提升泛化能力和实时性能。**

- **链接: [http://arxiv.org/pdf/2508.14358v1](http://arxiv.org/pdf/2508.14358v1)**

> **作者:** Zhujun Li; Shuo Zhang; Ioannis Stamos
>
> **备注:** Accepted by ICCV 2025 Workshop on Recovering 6D Object Pose (R6D)
>
> **摘要:** Category-level object pose estimation aims to predict the 6D pose and 3D size of objects within given categories. Existing approaches for this task rely solely on 6D poses as supervisory signals without explicitly capturing the intrinsic continuity of poses, leading to inconsistencies in predictions and reduced generalization to unseen poses. To address this limitation, we propose HRC-Pose, a novel depth-only framework for category-level object pose estimation, which leverages contrastive learning to learn point cloud representations that preserve the continuity of 6D poses. HRC-Pose decouples object pose into rotation and translation components, which are separately encoded and leveraged throughout the network. Specifically, we introduce a contrastive learning strategy for multi-task, multi-category scenarios based on our 6D pose-aware hierarchical ranking scheme, which contrasts point clouds from multiple categories by considering rotational and translational differences as well as categorical information. We further design pose estimation modules that separately process the learned rotation-aware and translation-aware embeddings. Our experiments demonstrate that HRC-Pose successfully learns continuous feature spaces. Results on REAL275 and CAMERA25 benchmarks show that our method consistently outperforms existing depth-only state-of-the-art methods and runs in real-time, demonstrating its effectiveness and potential for real-world applications. Our code is at https://github.com/zhujunli1993/HRC-Pose.
>
---
#### [new 029] WeedSense: Multi-Task Learning for Weed Segmentation, Height Estimation, and Growth Stage Classification
- **分类: cs.CV**

- **简介: 论文提出多任务学习框架，解决杂草监测问题，通过联合分割、高度估计和生长阶段分类，使用创新架构和数据集提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2508.14486v1](http://arxiv.org/pdf/2508.14486v1)**

> **作者:** Toqi Tahamid Sarker; Khaled R Ahmed; Taminul Islam; Cristiana Bernardi Rankrape; Karla Gage
>
> **备注:** This paper has been submitted and accepted for publication at ICCVW 2025
>
> **摘要:** Weed management represents a critical challenge in agriculture, significantly impacting crop yields and requiring substantial resources for control. Effective weed monitoring and analysis strategies are crucial for implementing sustainable agricultural practices and site-specific management approaches. We introduce WeedSense, a novel multi-task learning architecture for comprehensive weed analysis that jointly performs semantic segmentation, height estimation, and growth stage classification. We present a unique dataset capturing 16 weed species over an 11-week growth cycle with pixel-level annotations, height measurements, and temporal labels. WeedSense leverages a dual-path encoder incorporating Universal Inverted Bottleneck blocks and a Multi-Task Bifurcated Decoder with transformer-based feature fusion to generate multi-scale features and enable simultaneous prediction across multiple tasks. WeedSense outperforms other state-of-the-art models on our comprehensive evaluation. On our multi-task dataset, WeedSense achieves mIoU of 89.78% for segmentation, 1.67cm MAE for height estimation, and 99.99% accuracy for growth stage classification while maintaining real-time inference at 160 FPS. Our multitask approach achieves 3$\times$ faster inference than sequential single-task execution and uses 32.4% fewer parameters. Please see our project page at weedsense.github.io.
>
---
#### [new 030] Improving OCR using internal document redundancy
- **分类: cs.CV; cs.LG; eess.IV**

- **简介: 论文针对OCR在低质文档中的识别问题，提出利用文档内部冗余的无监督方法，通过扩展GMM模型结合EM算法和聚类调整，提升识别准确率，实验验证有效。**

- **链接: [http://arxiv.org/pdf/2508.14557v1](http://arxiv.org/pdf/2508.14557v1)**

> **作者:** Diego Belzarena; Seginus Mowlavi; Aitor Artola; Camilo Mariño; Marina Gardella; Ignacio Ramírez; Antoine Tadros; Roy He; Natalia Bottaioli; Boshra Rajaei; Gregory Randall; Jean-Michel Morel
>
> **备注:** 28 pages, 10 figures, including supplementary material. Code: https://github.com/seginusmowlavi/ocr-using-shape-redundancy. Dataset: https://github.com/camilomarino/ocr_berrutti_dataset
>
> **摘要:** Current OCR systems are based on deep learning models trained on large amounts of data. Although they have shown some ability to generalize to unseen data, especially in detection tasks, they can struggle with recognizing low-quality data. This is particularly evident for printed documents, where intra-domain data variability is typically low, but inter-domain data variability is high. In that context, current OCR methods do not fully exploit each document's redundancy. We propose an unsupervised method by leveraging the redundancy of character shapes within a document to correct imperfect outputs of a given OCR system and suggest better clustering. To this aim, we introduce an extended Gaussian Mixture Model (GMM) by alternating an Expectation-Maximization (EM) algorithm with an intra-cluster realignment process and normality statistical testing. We demonstrate improvements in documents with various levels of degradation, including recovered Uruguayan military archives and 17th to mid-20th century European newspapers.
>
---
#### [new 031] Fusing Monocular RGB Images with AIS Data to Create a 6D Pose Estimation Dataset for Marine Vessels
- **分类: cs.CV; cs.RO**

- **简介: 该论文旨在通过融合单目RGB图像与AIS数据，解决纯AIS定位的可靠性问题，提出基于YOLOX-X和PnP方法生成无需人工标注的海洋船舶6D姿态估计数据集BONK-pose。**

- **链接: [http://arxiv.org/pdf/2508.14767v1](http://arxiv.org/pdf/2508.14767v1)**

> **作者:** Fabian Holst; Emre Gülsoylu; Simone Frintrop
>
> **备注:** Author version of the submission to the IEEE Journal of Oceanic Engineering
>
> **摘要:** The paper presents a novel technique for creating a 6D pose estimation dataset for marine vessels by fusing monocular RGB images with Automatic Identification System (AIS) data. The proposed technique addresses the limitations of relying purely on AIS for location information, caused by issues like equipment reliability, data manipulation, and transmission delays. By combining vessel detections from monocular RGB images, obtained using an object detection network (YOLOX-X), with AIS messages, the technique generates 3D bounding boxes that represent the vessels' 6D poses, i.e. spatial and rotational dimensions. The paper evaluates different object detection models to locate vessels in image space. We also compare two transformation methods (homography and Perspective-n-Point) for aligning AIS data with image coordinates. The results of our work demonstrate that the Perspective-n-Point (PnP) method achieves a significantly lower projection error compared to homography-based approaches used before, and the YOLOX-X model achieves a mean Average Precision (mAP) of 0.80 at an Intersection over Union (IoU) threshold of 0.5 for relevant vessel classes. We show indication that our approach allows the creation of a 6D pose estimation dataset without needing manual annotation. Additionally, we introduce the Boats on Nordelbe Kehrwieder (BONK-pose), a publicly available dataset comprising 3753 images with 3D bounding box annotations for pose estimation, created by our data fusion approach. This dataset can be used for training and evaluating 6D pose estimation networks. In addition we introduce a set of 1000 images with 2D bounding box annotations for ship detection from the same scene.
>
---
#### [new 032] Safety-Critical Learning for Long-Tail Events: The TUM Traffic Accident Dataset
- **分类: cs.CV**

- **简介: 论文针对罕见交通事故检测任务，提出TUMTraf-A数据集及Accid3nD模型，结合规则与学习方法提升检测鲁棒性，用于高危场景下的事故识别与分析。**

- **链接: [http://arxiv.org/pdf/2508.14567v1](http://arxiv.org/pdf/2508.14567v1)**

> **作者:** Walter Zimmer; Ross Greer; Xingcheng Zhou; Rui Song; Marc Pavel; Daniel Lehmberg; Ahmed Ghita; Akshay Gopalkrishnan; Mohan Trivedi; Alois Knoll
>
> **备注:** Accepted for ICRA 40 Year Anniversary (ICRA40)
>
> **摘要:** Even though a significant amount of work has been done to increase the safety of transportation networks, accidents still occur regularly. They must be understood as an unavoidable and sporadic outcome of traffic networks. We present the TUM Traffic Accident (TUMTraf-A) dataset, a collection of real-world highway accidents. It contains ten sequences of vehicle crashes at high-speed driving with 294,924 labeled 2D and 93,012 labeled 3D boxes and track IDs within 48,144 labeled frames recorded from four roadside cameras and LiDARs at 10 Hz. The dataset contains ten object classes and is provided in the OpenLABEL format. We propose Accid3nD, an accident detection model that combines a rule-based approach with a learning-based one. Experiments and ablation studies on our dataset show the robustness of our proposed method. The dataset, model, and code are available on our project website: https://tum-traffic-dataset.github.io/tumtraf-a.
>
---
#### [new 033] LENS: Learning to Segment Anything with Unified Reinforced Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 论文针对文本提示图像分割任务，解决现有方法忽略推理链导致泛化能力差的问题，提出LENS框架通过统一强化学习优化推理与分割，提升模型在RefCOCO等基准上的性能。**

- **链接: [http://arxiv.org/pdf/2508.14153v1](http://arxiv.org/pdf/2508.14153v1)**

> **作者:** Lianghui Zhu; Bin Ouyang; Yuxuan Zhang; Tianheng Cheng; Rui Hu; Haocheng Shen; Longjin Ran; Xiaoxin Chen; Li Yu; Wenyu Liu; Xinggang Wang
>
> **备注:** Code is released at https://github.com/hustvl/LENS
>
> **摘要:** Text-prompted image segmentation enables fine-grained visual understanding and is critical for applications such as human-computer interaction and robotics. However, existing supervised fine-tuning methods typically ignore explicit chain-of-thought (CoT) reasoning at test time, which limits their ability to generalize to unseen prompts and domains. To address this issue, we introduce LENS, a scalable reinforcement-learning framework that jointly optimizes the reasoning process and segmentation in an end-to-end manner. We propose unified reinforcement-learning rewards that span sentence-, box-, and segment-level cues, encouraging the model to generate informative CoT rationales while refining mask quality. Using a publicly available 3-billion-parameter vision-language model, i.e., Qwen2.5-VL-3B-Instruct, LENS achieves an average cIoU of 81.2% on the RefCOCO, RefCOCO+, and RefCOCOg benchmarks, outperforming the strong fine-tuned method, i.e., GLaMM, by up to 5.6%. These results demonstrate that RL-driven CoT reasoning serves as a robust prior for text-prompted segmentation and offers a practical path toward more generalizable Segment Anything models. Code is available at https://github.com/hustvl/LENS.
>
---
#### [new 034] A comparative study of some wavelet and sampling operators on various features of an image
- **分类: cs.CV; math.FA; 41A25, 41A35, 46E30, 47A58, 47B38, 94A12**

- **简介: 该论文比较小波与采样算子在图像特征上的表现，分析其逼近性质及误差指标，通过实例验证理论，探讨算子在非理想条件下的适用性。**

- **链接: [http://arxiv.org/pdf/2508.14043v1](http://arxiv.org/pdf/2508.14043v1)**

> **作者:** Digvijay Singh; Rahul Shukla; Karunesh Kumar Singh
>
> **备注:** 15 pages
>
> **摘要:** This research includes the study of some positive sampling Kantorovich operators (SK operators) and their convergence properties. A comprehensive analysis of both local and global approximation properties is presented using sampling Kantorovich (SK), Gaussian, Bilateral and the thresholding wavelet-based operators in the framework of SK-operators. Explicitly, we start the article by introducing the basic terminology and state the fundamental theorem of approximation (FTA) by imposing the various required conditions corresponding to the various defined operators. We measure the error and study the other mathematical parameters such as the mean square error (MSE), the speckle index (SI), the speckle suppression index (SSI), the speckle mean preservation index (SMPI), and the equivalent number of looks (ENL) at various levels of resolution parameters. The nature of these operators are demonstrated via an example under ideal conditions in tabulated form at a certain level of samples. Eventually, another numerical example is illustrated to discuss the region of interest (ROI) via SI, SSI and SMPI of 2D Shepp-Logan Phantom taken slice from the 3D image, which gives the justification of the fundamental theorem of approximation (FTA). At the end of the derivation and illustrations we observe that the various operators have their own significance while studying the various features of the image because of the uneven nature of an image (non-ideal condition). Therefore, to some extent, some operators work well and some do not for some specific features of the image.
>
---
#### [new 035] CLIPSym: Delving into Symmetry Detection with CLIP
- **分类: cs.CV**

- **简介: 该论文针对对称性检测任务，解决旋转/反射对称识别问题，提出CLIPSym框架，融合CLIP语言编码与旋转等变解码器，并设计语义感知提示分组技术，在三个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2508.14197v1](http://arxiv.org/pdf/2508.14197v1)**

> **作者:** Tinghan Yang; Md Ashiqur Rahman; Raymond A. Yeh
>
> **摘要:** Symmetry is one of the most fundamental geometric cues in computer vision, and detecting it has been an ongoing challenge. With the recent advances in vision-language models,~i.e., CLIP, we investigate whether a pre-trained CLIP model can aid symmetry detection by leveraging the additional symmetry cues found in the natural image descriptions. We propose CLIPSym, which leverages CLIP's image and language encoders and a rotation-equivariant decoder based on a hybrid of Transformer and $G$-Convolution to detect rotation and reflection symmetries. To fully utilize CLIP's language encoder, we have developed a novel prompting technique called Semantic-Aware Prompt Grouping (SAPG), which aggregates a diverse set of frequent object-based prompts to better integrate the semantic cues for symmetry detection. Empirically, we show that CLIPSym outperforms the current state-of-the-art on three standard symmetry detection datasets (DENDI, SDRW, and LDRS). Finally, we conduct detailed ablations verifying the benefits of CLIP's pre-training, the proposed equivariant decoder, and the SAPG technique. The code is available at https://github.com/timyoung2333/CLIPSym.
>
---
#### [new 036] Multiscale Video Transformers for Class Agnostic Segmentation in Autonomous Driving
- **分类: cs.CV**

- **简介: 论文提出多尺度视频Transformer，用于自动驾驶中的类无关分割，解决未知物体检测问题，通过高效设计提升精度与效率。**

- **链接: [http://arxiv.org/pdf/2508.14729v1](http://arxiv.org/pdf/2508.14729v1)**

> **作者:** Leila Cheshmi; Mennatullah Siam
>
> **备注:** 6 pages, 2 figures, 1 table
>
> **摘要:** Ensuring safety in autonomous driving is a complex challenge requiring handling unknown objects and unforeseen driving scenarios. We develop multiscale video transformers capable of detecting unknown objects using only motion cues. Video semantic and panoptic segmentation often relies on known classes seen during training, overlooking novel categories. Recent visual grounding with large language models is computationally expensive, especially for pixel-level output. We propose an efficient video transformer trained end-to-end for class-agnostic segmentation without optical flow. Our method uses multi-stage multiscale query-memory decoding and a scale-specific random drop-token to ensure efficiency and accuracy, maintaining detailed spatiotemporal features with a shared, learnable memory module. Unlike conventional decoders that compress features, our memory-centric design preserves high-resolution information at multiple scales. We evaluate on DAVIS'16, KITTI, and Cityscapes. Our method consistently outperforms multiscale baselines while being efficient in GPU memory and run-time, demonstrating a promising direction for real-time, robust dense prediction in safety-critical robotics.
>
---
#### [new 037] Lifespan Pancreas Morphology for Control vs Type 2 Diabetes using AI on Largescale Clinical Imaging
- **分类: cs.CV**

- **简介: 该论文使用AI分析临床影像数据，研究2型糖尿病对胰腺形态的影响，通过比较CT/MRI数据，建立年龄趋势模型，发现糖尿病患者胰腺形态异常。**

- **链接: [http://arxiv.org/pdf/2508.14878v1](http://arxiv.org/pdf/2508.14878v1)**

> **作者:** Lucas W. Remedios; Chloe Cho; Trent M. Schwartz; Dingjie Su; Gaurav Rudravaram; Chenyu Gao; Aravind R. Krishnan; Adam M. Saunders; Michael E. Kim; Shunxing Bao; Thomas A. Lasko; Alvin C. Powers; Bennett A. Landman; John Virostko
>
> **摘要:** Purpose: Understanding how the pancreas changes is critical for detecting deviations in type 2 diabetes and other pancreatic disease. We measure pancreas size and shape using morphological measurements from ages 0 to 90. Our goals are to 1) identify reliable clinical imaging modalities for AI-based pancreas measurement, 2) establish normative morphological aging trends, and 3) detect potential deviations in type 2 diabetes. Approach: We analyzed a clinically acquired dataset of 2533 patients imaged with abdominal CT or MRI. We resampled the scans to 3mm isotropic resolution, segmented the pancreas using automated methods, and extracted 13 morphological pancreas features across the lifespan. First, we assessed CT and MRI measurements to determine which modalities provide consistent lifespan trends. Second, we characterized distributions of normative morphological patterns stratified by age group and sex. Third, we used GAMLSS regression to model pancreas morphology trends in 1350 patients matched for age, sex, and type 2 diabetes status to identify any deviations from normative aging associated with type 2 diabetes. Results: When adjusting for confounders, the aging trends for 10 of 13 morphological features were significantly different between patients with type 2 diabetes and non-diabetic controls (p < 0.05 after multiple comparisons corrections). Additionally, MRI appeared to yield different pancreas measurements than CT using our AI-based method. Conclusions: We provide lifespan trends demonstrating that the size and shape of the pancreas is altered in type 2 diabetes using 675 control patients and 675 diabetes patients. Moreover, our findings reinforce that the pancreas is smaller in type 2 diabetes. Additionally, we contribute a reference of lifespan pancreas morphology from a large cohort of non-diabetic control patients in a clinical setting.
>
---
#### [new 038] Local Scale Equivariance with Latent Deep Equilibrium Canonicalizer
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文针对计算机视觉中的尺度变化问题，提出深度均衡归一化器（DEC）以提升模型局部尺度等变性，通过整合至现有网络并优化预训练模型，在ImageNet基准上提高性能与一致性。**

- **链接: [http://arxiv.org/pdf/2508.14187v1](http://arxiv.org/pdf/2508.14187v1)**

> **作者:** Md Ashiqur Rahman; Chiao-An Yang; Michael N. Cheng; Lim Jun Hao; Jeremiah Jiang; Teck-Yian Lim; Raymond A. Yeh
>
> **摘要:** Scale variation is a fundamental challenge in computer vision. Objects of the same class can have different sizes, and their perceived size is further affected by the distance from the camera. These variations are local to the objects, i.e., different object sizes may change differently within the same image. To effectively handle scale variations, we present a deep equilibrium canonicalizer (DEC) to improve the local scale equivariance of a model. DEC can be easily incorporated into existing network architectures and can be adapted to a pre-trained model. Notably, we show that on the competitive ImageNet benchmark, DEC improves both model performance and local scale consistency across four popular pre-trained deep-nets, e.g., ViT, DeiT, Swin, and BEiT. Our code is available at https://github.com/ashiq24/local-scale-equivariance.
>
---
#### [new 039] GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting
- **分类: cs.CV**

- **简介: 论文针对3D高斯点渲染中极端视角或部分观察区域的高质量生成问题，提出GSFix3D框架，结合扩散模型知识蒸馏与自定义微调，利用网格和3D高斯点进行修复，并引入随机掩码增强策略，实现鲁棒的视图修复。**

- **链接: [http://arxiv.org/pdf/2508.14717v1](http://arxiv.org/pdf/2508.14717v1)**

> **作者:** Jiaxin Wei; Stefan Leutenegger; Simon Schaefer
>
> **摘要:** Recent developments in 3D Gaussian Splatting have significantly enhanced novel view synthesis, yet generating high-quality renderings from extreme novel viewpoints or partially observed regions remains challenging. Meanwhile, diffusion models exhibit strong generative capabilities, but their reliance on text prompts and lack of awareness of specific scene information hinder accurate 3D reconstruction tasks. To address these limitations, we introduce GSFix3D, a novel framework that improves the visual fidelity in under-constrained regions by distilling prior knowledge from diffusion models into 3D representations, while preserving consistency with observed scene details. At its core is GSFixer, a latent diffusion model obtained via our customized fine-tuning protocol that can leverage both mesh and 3D Gaussians to adapt pretrained generative models to a variety of environments and artifact types from different reconstruction methods, enabling robust novel view repair for unseen camera poses. Moreover, we propose a random mask augmentation strategy that empowers GSFixer to plausibly inpaint missing regions. Experiments on challenging benchmarks demonstrate that our GSFix3D and GSFixer achieve state-of-the-art performance, requiring only minimal scene-specific fine-tuning on captured data. Real-world test further confirms its resilience to potential pose errors. Our code and data will be made publicly available. Project page: https://gsfix3d.github.io.
>
---
#### [new 040] OccluNet: Spatio-Temporal Deep Learning for Occlusion Detection on DSA
- **分类: cs.CV; cs.AI**

- **简介: 论文提出OccluNet，结合YOLOX与Transformer时空注意力机制，用于DSA序列血管栓塞检测，解决解剖复杂性及时间约束问题，实现89.02%精度和74.87%召回率。**

- **链接: [http://arxiv.org/pdf/2508.14286v1](http://arxiv.org/pdf/2508.14286v1)**

> **作者:** Anushka A. Kore; Frank G. te Nijenhuis; Matthijs van der Sluijs; Wim van Zwam; Charles Majoie; Geert Lycklama à Nijeholt; Danny Ruijters; Frans Vos; Sandra Cornelissen; Ruisheng Su; Theo van Walsum
>
> **备注:** To be published in Proceedings of the SWITCH Workshop at MICCAI 2025, Lecture Notes in Computer Science (LNCS), Springer
>
> **摘要:** Accurate detection of vascular occlusions during endovascular thrombectomy (EVT) is critical in acute ischemic stroke (AIS). Interpretation of digital subtraction angiography (DSA) sequences poses challenges due to anatomical complexity and time constraints. This work proposes OccluNet, a spatio-temporal deep learning model that integrates YOLOX, a single-stage object detector, with transformer-based temporal attention mechanisms to automate occlusion detection in DSA sequences. We compared OccluNet with a YOLOv11 baseline trained on either individual DSA frames or minimum intensity projections. Two spatio-temporal variants were explored for OccluNet: pure temporal attention and divided space-time attention. Evaluation on DSA images from the MR CLEAN Registry revealed the model's capability to capture temporally consistent features, achieving precision and recall of 89.02% and 74.87%, respectively. OccluNet significantly outperformed the baseline models, and both attention variants attained similar performance. Source code is available at https://github.com/anushka-kore/OccluNet.git
>
---
#### [new 041] Reconstruction Using the Invisible: Intuition from NIR and Metadata for Enhanced 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文针对农业场景的3D重建难题，提出NIRPlant多模态数据集及NIRSplat模型，结合NIR与元数据提升3DGS性能，有效应对光照、遮挡等挑战。**

- **链接: [http://arxiv.org/pdf/2508.14443v1](http://arxiv.org/pdf/2508.14443v1)**

> **作者:** Gyusam Chang; Tuan-Anh Vu; Vivek Alumootil; Harris Song; Deanna Pham; Sangpil Kim; M. Khalid Jawed
>
> **摘要:** While 3D Gaussian Splatting (3DGS) has rapidly advanced, its application in agriculture remains underexplored. Agricultural scenes present unique challenges for 3D reconstruction methods, particularly due to uneven illumination, occlusions, and a limited field of view. To address these limitations, we introduce \textbf{NIRPlant}, a novel multimodal dataset encompassing Near-Infrared (NIR) imagery, RGB imagery, textual metadata, Depth, and LiDAR data collected under varied indoor and outdoor lighting conditions. By integrating NIR data, our approach enhances robustness and provides crucial botanical insights that extend beyond the visible spectrum. Additionally, we leverage text-based metadata derived from vegetation indices, such as NDVI, NDWI, and the chlorophyll index, which significantly enriches the contextual understanding of complex agricultural environments. To fully exploit these modalities, we propose \textbf{NIRSplat}, an effective multimodal Gaussian splatting architecture employing a cross-attention mechanism combined with 3D point-based positional encoding, providing robust geometric priors. Comprehensive experiments demonstrate that \textbf{NIRSplat} outperforms existing landmark methods, including 3DGS, CoR-GS, and InstantSplat, highlighting its effectiveness in challenging agricultural scenarios. The code and dataset are publicly available at: https://github.com/StructuresComp/3D-Reconstruction-NIR
>
---
#### [new 042] DreamSwapV: Mask-guided Subject Swapping for Any Customized Video Editing
- **分类: cs.CV**

- **简介: 论文提出DreamSwapV，用于视频中任意主体交换。解决现有方法领域局限与提示模糊问题，通过mask引导、多条件融合及自适应策略实现高保真定制编辑。**

- **链接: [http://arxiv.org/pdf/2508.14465v1](http://arxiv.org/pdf/2508.14465v1)**

> **作者:** Weitao Wang; Zichen Wang; Hongdeng Shen; Yulei Lu; Xirui Fan; Suhui Wu; Jun Zhang; Haoqian Wang; Hao Zhang
>
> **摘要:** With the rapid progress of video generation, demand for customized video editing is surging, where subject swapping constitutes a key component yet remains under-explored. Prevailing swapping approaches either specialize in narrow domains--such as human-body animation or hand-object interaction--or rely on some indirect editing paradigm or ambiguous text prompts that compromise final fidelity. In this paper, we propose DreamSwapV, a mask-guided, subject-agnostic, end-to-end framework that swaps any subject in any video for customization with a user-specified mask and reference image. To inject fine-grained guidance, we introduce multiple conditions and a dedicated condition fusion module that integrates them efficiently. In addition, an adaptive mask strategy is designed to accommodate subjects of varying scales and attributes, further improving interactions between the swapped subject and its surrounding context. Through our elaborate two-phase dataset construction and training scheme, our DreamSwapV outperforms existing methods, as validated by comprehensive experiments on VBench indicators and our first introduced DreamSwapV-Benchmark.
>
---
#### [new 043] Federated Action Recognition for Smart Worker Assistance Using FastPose
- **分类: cs.CV; cs.AI; cs.DC; cs.HC**

- **简介: 论文提出联邦学习框架用于工业动作识别，解决隐私与跨用户泛化问题，采用自定义数据集和FastPose模型，验证FL在提升准确率和隐私保护中的优势。**

- **链接: [http://arxiv.org/pdf/2508.14113v1](http://arxiv.org/pdf/2508.14113v1)**

> **作者:** Vinit Hegiste; Vidit Goyal; Tatjana Legler; Martin Ruskowski
>
> **备注:** 8 pages and submitted to FLTA2025 conference
>
> **摘要:** In smart manufacturing environments, accurate and real-time recognition of worker actions is essential for productivity, safety, and human-machine collaboration. While skeleton-based human activity recognition (HAR) offers robustness to lighting, viewpoint, and background variations, most existing approaches rely on centralized datasets, which are impractical in privacy-sensitive industrial scenarios. This paper presents a federated learning (FL) framework for pose-based HAR using a custom skeletal dataset of eight industrially relevant upper-body gestures, captured from five participants and processed using a modified FastPose model. Two temporal backbones, an LSTM and a Transformer encoder, are trained and evaluated under four paradigms: centralized, local (per-client), FL with weighted federated averaging (FedAvg), and federated ensemble learning (FedEnsemble). On the global test set, the FL Transformer improves over centralized training by +12.4 percentage points, with FedEnsemble delivering a +16.3 percentage points gain. On an unseen external client, FL and FedEnsemble exceed centralized accuracy by +52.6 and +58.3 percentage points, respectively. These results demonstrate that FL not only preserves privacy but also substantially enhances cross-user generalization, establishing it as a practical solution for scalable, privacy-aware HAR in heterogeneous industrial settings.
>
---
#### [new 044] MS-CLR: Multi-Skeleton Contrastive Learning for Human Action Recognition
- **分类: cs.CV**

- **简介: 论文针对基于骨骼的人类动作识别中单一骨骼结构限制泛化的问题，提出MS-CLR框架，通过多骨骼对比学习和改进的ST-GCN架构，提升模型对多样化骨骼结构的适应能力，并在NTU数据集上取得新状态。**

- **链接: [http://arxiv.org/pdf/2508.14889v1](http://arxiv.org/pdf/2508.14889v1)**

> **作者:** Mert Kiray; Alvaro Ritter; Nassir Navab; Benjamin Busam
>
> **摘要:** Contrastive learning has gained significant attention in skeleton-based action recognition for its ability to learn robust representations from unlabeled data. However, existing methods rely on a single skeleton convention, which limits their ability to generalize across datasets with diverse joint structures and anatomical coverage. We propose Multi-Skeleton Contrastive Learning (MS-CLR), a general self-supervised framework that aligns pose representations across multiple skeleton conventions extracted from the same sequence. This encourages the model to learn structural invariances and capture diverse anatomical cues, resulting in more expressive and generalizable features. To support this, we adapt the ST-GCN architecture to handle skeletons with varying joint layouts and scales through a unified representation scheme. Experiments on the NTU RGB+D 60 and 120 datasets demonstrate that MS-CLR consistently improves performance over strong single-skeleton contrastive learning baselines. A multi-skeleton ensemble further boosts performance, setting new state-of-the-art results on both datasets.
>
---
#### [new 045] GaussianArt: Unified Modeling of Geometry and Motion for Articulated Objects
- **分类: cs.CV**

- **简介: 论文针对关节物体的几何与运动分离重建问题，提出统一的3D高斯模型及MPArt-90基准，实现高效、可扩展的关节物体建模，适用于数字孪生与物理模拟。**

- **链接: [http://arxiv.org/pdf/2508.14891v1](http://arxiv.org/pdf/2508.14891v1)**

> **作者:** Licheng Shen; Saining Zhang; Honghan Li; Peilin Yang; Zihao Huang; Zongzheng Zhang; Hao Zhao
>
> **备注:** Project Page: https://sainingzhang.github.io/project/gaussianart/
>
> **摘要:** Reconstructing articulated objects is essential for building digital twins of interactive environments. However, prior methods typically decouple geometry and motion by first reconstructing object shape in distinct states and then estimating articulation through post-hoc alignment. This separation complicates the reconstruction pipeline and restricts scalability, especially for objects with complex, multi-part articulation. We introduce a unified representation that jointly models geometry and motion using articulated 3D Gaussians. This formulation improves robustness in motion decomposition and supports articulated objects with up to 20 parts, significantly outperforming prior approaches that often struggle beyond 2--3 parts due to brittle initialization. To systematically assess scalability and generalization, we propose MPArt-90, a new benchmark consisting of 90 articulated objects across 20 categories, each with diverse part counts and motion configurations. Extensive experiments show that our method consistently achieves superior accuracy in part-level geometry reconstruction and motion estimation across a broad range of object types. We further demonstrate applicability to downstream tasks such as robotic simulation and human-scene interaction modeling, highlighting the potential of unified articulated representations in scalable physical modeling.
>
---
#### [new 046] QuadINR: Hardware-Efficient Implicit Neural Representations Through Quadratic Activation
- **分类: cs.CV**

- **简介: 该论文针对隐式神经表示（INR）的硬件高效实现问题，提出QuadINR通过分段二次激活函数和统一N-stage流水线框架，在FPGA/ASIC上实现高能效，显著降低硬件资源与功耗，提升性能。**

- **链接: [http://arxiv.org/pdf/2508.14374v1](http://arxiv.org/pdf/2508.14374v1)**

> **作者:** Wenyong Zhou; Boyu Li; Jiachen Ren; Taiqiang Wu; Zhilin Ai; Zhengwu Liu; Ngai Wong
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Implicit Neural Representations (INRs) encode discrete signals continuously while addressing spectral bias through activation functions (AFs). Previous approaches mitigate this bias by employing complex AFs, which often incur significant hardware overhead. To tackle this challenge, we introduce QuadINR, a hardware-efficient INR that utilizes piecewise quadratic AFs to achieve superior performance with dramatic reductions in hardware consumption. The quadratic functions encompass rich harmonic content in their Fourier series, delivering enhanced expressivity for high-frequency signals, as verified through Neural Tangent Kernel (NTK) analysis. We develop a unified $N$-stage pipeline framework that facilitates efficient hardware implementation of various AFs in INRs. We demonstrate FPGA implementations on the VCU128 platform and an ASIC implementation in a 28nm process. Experiments across images and videos show that QuadINR achieves up to 2.06dB PSNR improvement over prior work, with an area of only 1914$\mu$m$^2$ and a dynamic power of 6.14mW, reducing resource and power consumption by up to 97\% and improving latency by up to 93\% vs existing baselines.
>
---
#### [new 047] AnchorSync: Global Consistency Optimization for Long Video Editing
- **分类: cs.CV**

- **简介: 该论文针对长视频编辑中的全局一致性与时间连贯性问题，提出AnchorSync框架，通过稀疏锚定帧编辑与多模态引导实现高保真长期编辑，提升视觉质量与稳定性。**

- **链接: [http://arxiv.org/pdf/2508.14609v1](http://arxiv.org/pdf/2508.14609v1)**

> **作者:** Zichi Liu; Yinggui Wang; Tao Wei; Chao Ma
>
> **备注:** ACM MM 2025; Code is released at https://github.com/VISION-SJTU/AnchorSync
>
> **摘要:** Editing long videos remains a challenging task due to the need for maintaining both global consistency and temporal coherence across thousands of frames. Existing methods often suffer from structural drift or temporal artifacts, particularly in minute-long sequences. We introduce AnchorSync, a novel diffusion-based framework that enables high-quality, long-term video editing by decoupling the task into sparse anchor frame editing and smooth intermediate frame interpolation. Our approach enforces structural consistency through a progressive denoising process and preserves temporal dynamics via multimodal guidance. Extensive experiments show that AnchorSync produces coherent, high-fidelity edits, surpassing prior methods in visual quality and temporal stability.
>
---
#### [new 048] RynnEC: Bringing MLLMs into Embodied World
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出RynnEC，一种用于具身认知的视频多模态大模型，解决区域级视频交互与数据稀缺问题，通过区域编码器、掩码解码器及自研数据管道提升物体理解与空间推理能力，并构建RynnEC-Bench基准。**

- **链接: [http://arxiv.org/pdf/2508.14160v1](http://arxiv.org/pdf/2508.14160v1)**

> **作者:** Ronghao Dang; Yuqian Yuan; Yunxuan Mao; Kehan Li; Jiangpin Liu; Zhikai Wang; Xin Li; Fan Wang; Deli Zhao
>
> **备注:** The technical report of RynnEC, an embodied cognition MLLM
>
> **摘要:** We introduce RynnEC, a video multimodal large language model designed for embodied cognition. Built upon a general-purpose vision-language foundation model, RynnEC incorporates a region encoder and a mask decoder, enabling flexible region-level video interaction. Despite its compact architecture, RynnEC achieves state-of-the-art performance in object property understanding, object segmentation, and spatial reasoning. Conceptually, it offers a region-centric video paradigm for the brain of embodied agents, providing fine-grained perception of the physical world and enabling more precise interactions. To mitigate the scarcity of annotated 3D datasets, we propose an egocentric video based pipeline for generating embodied cognition data. Furthermore, we introduce RynnEC-Bench, a region-centered benchmark for evaluating embodied cognitive capabilities. We anticipate that RynnEC will advance the development of general-purpose cognitive cores for embodied agents and facilitate generalization across diverse embodied tasks. The code, model checkpoints, and benchmark are available at: https://github.com/alibaba-damo-academy/RynnEC
>
---
#### [new 049] D^3-Talker: Dual-Branch Decoupled Deformation Fields for Few-Shot 3D Talking Head Synthesis
- **分类: cs.CV**

- **简介: 论文任务为少样本3D说话人脸合成，解决训练数据少导致的嘴唇同步与图像质量问题，提出双分支解耦变形场、对比损失及Coarse-to-Fine模块提升效果。**

- **链接: [http://arxiv.org/pdf/2508.14449v1](http://arxiv.org/pdf/2508.14449v1)**

> **作者:** Yuhang Guo; Kaijun Deng; Siyang Song; Jindong Xie; Wenhui Ma; Linlin Shen
>
> **摘要:** A key challenge in 3D talking head synthesis lies in the reliance on a long-duration talking head video to train a new model for each target identity from scratch. Recent methods have attempted to address this issue by extracting general features from audio through pre-training models. However, since audio contains information irrelevant to lip motion, existing approaches typically struggle to map the given audio to realistic lip behaviors in the target face when trained on only a few frames, causing poor lip synchronization and talking head image quality. This paper proposes D^3-Talker, a novel approach that constructs a static 3D Gaussian attribute field and employs audio and Facial Motion signals to independently control two distinct Gaussian attribute deformation fields, effectively decoupling the predictions of general and personalized deformations. We design a novel similarity contrastive loss function during pre-training to achieve more thorough decoupling. Furthermore, we integrate a Coarse-to-Fine module to refine the rendered images, alleviating blurriness caused by head movements and enhancing overall image quality. Extensive experiments demonstrate that D^3-Talker outperforms state-of-the-art methods in both high-fidelity rendering and accurate audio-lip synchronization with limited training data. Our code will be provided upon acceptance.
>
---
#### [new 050] CTA-Flux: Integrating Chinese Cultural Semantics into High-Quality English Text-to-Image Communities
- **分类: cs.CV**

- **简介: 该论文针对英文文本到图像模型处理中文提示的文化语义偏差问题，提出CTA-Flux方法，通过MMDiT直接控制Flux模型，减少参数并提升中文语义理解，保持与现有插件兼容，实现高质量图像生成。**

- **链接: [http://arxiv.org/pdf/2508.14405v1](http://arxiv.org/pdf/2508.14405v1)**

> **作者:** Yue Gong; Shanyuan Liu; Liuzhuozheng Li; Jian Zhu; Bo Cheng; Liebucha Wu; Xiaoyu Wu; Yuhang Ma; Dawei Leng; Yuhui Yin
>
> **摘要:** We proposed the Chinese Text Adapter-Flux (CTA-Flux). An adaptation method fits the Chinese text inputs to Flux, a powerful text-to-image (TTI) generative model initially trained on the English corpus. Despite the notable image generation ability conditioned on English text inputs, Flux performs poorly when processing non-English prompts, particularly due to linguistic and cultural biases inherent in predominantly English-centric training datasets. Existing approaches, such as translating non-English prompts into English or finetuning models for bilingual mappings, inadequately address culturally specific semantics, compromising image authenticity and quality. To address this issue, we introduce a novel method to bridge Chinese semantic understanding with compatibility in English-centric TTI model communities. Existing approaches relying on ControlNet-like architectures typically require a massive parameter scale and lack direct control over Chinese semantics. In comparison, CTA-flux leverages MultiModal Diffusion Transformer (MMDiT) to control the Flux backbone directly, significantly reducing the number of parameters while enhancing the model's understanding of Chinese semantics. This integration significantly improves the generation quality and cultural authenticity without extensive retraining of the entire model, thus maintaining compatibility with existing text-to-image plugins such as LoRA, IP-Adapter, and ControlNet. Empirical evaluations demonstrate that CTA-flux supports Chinese and English prompts and achieves superior image generation quality, visual realism, and faithful depiction of Chinese semantics.
>
---
#### [new 051] Taming Transformer for Emotion-Controllable Talking Face Generation
- **分类: cs.CV**

- **简介: 论文旨在实现情感可控的说话人脸生成任务，解决多模态情感建模与身份保持难题。通过预训练分离音频、量化视频，引入情绪锚点表示，结合自回归Transformer生成情感视频，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.14359v1](http://arxiv.org/pdf/2508.14359v1)**

> **作者:** Ziqi Zhang; Cheng Deng
>
> **摘要:** Talking face generation is a novel and challenging generation task, aiming at synthesizing a vivid speaking-face video given a specific audio. To fulfill emotion-controllable talking face generation, current methods need to overcome two challenges: One is how to effectively model the multimodal relationship related to the specific emotion, and the other is how to leverage this relationship to synthesize identity preserving emotional videos. In this paper, we propose a novel method to tackle the emotion-controllable talking face generation task discretely. Specifically, we employ two pre-training strategies to disentangle audio into independent components and quantize videos into combinations of visual tokens. Subsequently, we propose the emotion-anchor (EA) representation that integrates the emotional information into visual tokens. Finally, we introduce an autoregressive transformer to model the global distribution of the visual tokens under the given conditions and further predict the index sequence for synthesizing the manipulated videos. We conduct experiments on the MEAD dataset that controls the emotion of videos conditioned on multiple emotional audios. Extensive experiments demonstrate the superiorities of our method both qualitatively and quantitatively.
>
---
#### [new 052] Adversarial Hospital-Invariant Feature Learning for WSI Patch Classification
- **分类: cs.CV; eess.IV**

- **简介: 该论文针对WSI切片分类中的医院差异问题，提出对抗框架去除医院特定特征，提升跨医院分类性能。**

- **链接: [http://arxiv.org/pdf/2508.14779v1](http://arxiv.org/pdf/2508.14779v1)**

> **作者:** Mengliang Zhang; Jacob M. Luber
>
> **备注:** 8 pages,6 figures
>
> **摘要:** Pathology foundation models (PFMs) have demonstrated remarkable potential in whole-slide image (WSI) diagnosis. However, pathology images from different hospitals often vary due to differences in scanning hardware and preprocessing styles, which may lead PFMs to inadvertently learn hospital-specific features, posing risks for clinical deployment. In this work, we present the first systematic study of domain bias in PFMs arising from hospital source characteristics. Specifically, we (1) construct a pipeline for quantifying domain bias in PFMs, (2) evaluate and compare the performance of multiple models, and (3) propose a lightweight adversarial framework that removes latent hospital-specific features from frozen representations without modifying the encoder itself. By introducing a trainable adapter and a domain classifier connected through a gradient reversal layer (GRL), our method learns task-discriminative yet domain-invariant representations. Experiments on multi-center histopathology datasets demonstrate that our approach substantially reduces domain predictability while maintaining or even improving disease classification performance, particularly in out-of-domain (unseen hospital) scenarios. Further analyses, including hospital detection and feature space visualization, confirm the effectiveness of our method in mitigating hospital bias. We will provide our code based on acceptance.
>
---
#### [new 053] A Comprehensive Review of Agricultural Parcel and Boundary Delineation from Remote Sensing Images: Recent Progress and Future Perspectives
- **分类: cs.CV; eess.IV**

- **简介: 该论文综述农业地块与边界分割（APBD）的遥感图像处理方法，分类讨论传统与深度学习方法，分析多传感器与算法比较，提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2508.14558v1](http://arxiv.org/pdf/2508.14558v1)**

> **作者:** Juepeng Zheng; Zi Ye; Yibin Wen; Jianxi Huang; Zhiwei Zhang; Qingmei Li; Qiong Hu; Baodong Xu; Lingyuan Zhao; Haohuan Fu
>
> **摘要:** Powered by advances in multiple remote sensing sensors, the production of high spatial resolution images provides great potential to achieve cost-efficient and high-accuracy agricultural inventory and analysis in an automated way. Lots of studies that aim at providing an inventory of the level of each agricultural parcel have generated many methods for Agricultural Parcel and Boundary Delineation (APBD). This review covers APBD methods for detecting and delineating agricultural parcels and systematically reviews the past and present of APBD-related research applied to remote sensing images. With the goal to provide a clear knowledge map of existing APBD efforts, we conduct a comprehensive review of recent APBD papers to build a meta-data analysis, including the algorithm, the study site, the crop type, the sensor type, the evaluation method, etc. We categorize the methods into three classes: (1) traditional image processing methods (including pixel-based, edge-based and region-based); (2) traditional machine learning methods (such as random forest, decision tree); and (3) deep learning-based methods. With deep learning-oriented approaches contributing to a majority, we further discuss deep learning-based methods like semantic segmentation-based, object detection-based and Transformer-based methods. In addition, we discuss five APBD-related issues to further comprehend the APBD domain using remote sensing data, such as multi-sensor data in APBD task, comparisons between single-task learning and multi-task learning in the APBD domain, comparisons among different algorithms and different APBD tasks, etc. Finally, this review proposes some APBD-related applications and a few exciting prospects and potential hot topics in future APBD research. We hope this review help researchers who involved in APBD domain to keep track of its development and tendency.
>
---
#### [new 054] GeMS: Efficient Gaussian Splatting for Extreme Motion Blur
- **分类: cs.CV**

- **简介: 论文提出GeMS框架，通过直接从极端模糊图像进行3DGS重建，解决传统方法需清晰图像的局限，集成深度学习结构光流、概率分布初始化及事件数据精修。**

- **链接: [http://arxiv.org/pdf/2508.14682v1](http://arxiv.org/pdf/2508.14682v1)**

> **作者:** Gopi Raju Matta; Trisha Reddypalli; Vemunuri Divya Madhuri; Kaushik Mitra
>
> **摘要:** We introduce GeMS, a framework for 3D Gaussian Splatting (3DGS) designed to handle severely motion-blurred images. State-of-the-art deblurring methods for extreme blur, such as ExBluRF, as well as Gaussian Splatting-based approaches like Deblur-GS, typically assume access to sharp images for camera pose estimation and point cloud generation, an unrealistic assumption. Methods relying on COLMAP initialization, such as BAD-Gaussians, also fail due to unreliable feature correspondences under severe blur. To address these challenges, we propose GeMS, a 3DGS framework that reconstructs scenes directly from extremely blurred images. GeMS integrates: (1) VGGSfM, a deep learning-based Structure-from-Motion pipeline that estimates poses and generates point clouds directly from blurred inputs; (2) 3DGS-MCMC, which enables robust scene initialization by treating Gaussians as samples from a probability distribution, eliminating heuristic densification and pruning; and (3) joint optimization of camera trajectories and Gaussian parameters for stable reconstruction. While this pipeline produces strong results, inaccuracies may remain when all inputs are severely blurred. To mitigate this, we propose GeMS-E, which integrates a progressive refinement step using events: (4) Event-based Double Integral (EDI) deblurring restores sharper images that are then fed into GeMS, improving pose estimation, point cloud generation, and overall reconstruction. Both GeMS and GeMS-E achieve state-of-the-art performance on synthetic and real-world datasets. To our knowledge, this is the first framework to address extreme motion blur within 3DGS directly from severely blurred inputs.
>
---
#### [new 055] Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering
- **分类: cs.CV**

- **简介: 该论文针对正逆渲染的循环不一致性问题，提出Ouroboros框架，通过双单步扩散模型相互强化，实现室内外场景的高效、高质渲染，并支持无训练视频分解。**

- **链接: [http://arxiv.org/pdf/2508.14461v1](http://arxiv.org/pdf/2508.14461v1)**

> **作者:** Shanlin Sun; Yifan Wang; Hanwen Zhang; Yifeng Xiong; Qin Ren; Ruogu Fang; Xiaohui Xie; Chenyu You
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** While multi-step diffusion models have advanced both forward and inverse rendering, existing approaches often treat these problems independently, leading to cycle inconsistency and slow inference speed. In this work, we present Ouroboros, a framework composed of two single-step diffusion models that handle forward and inverse rendering with mutual reinforcement. Our approach extends intrinsic decomposition to both indoor and outdoor scenes and introduces a cycle consistency mechanism that ensures coherence between forward and inverse rendering outputs. Experimental results demonstrate state-of-the-art performance across diverse scenes while achieving substantially faster inference speed compared to other diffusion-based methods. We also demonstrate that Ouroboros can transfer to video decomposition in a training-free manner, reducing temporal inconsistency in video sequences while maintaining high-quality per-frame inverse rendering.
>
---
#### [new 056] EventSSEG: Event-driven Self-Supervised Segmentation with Probabilistic Attention
- **分类: cs.CV**

- **简介: 该论文针对事件相机道路分割任务，解决传统帧相机低延迟与高计算成本问题，提出EventSSEG方法，通过事件仅计算与概率注意力机制结合自监督学习，实现无需大量标注数据的高效分割。**

- **链接: [http://arxiv.org/pdf/2508.14856v1](http://arxiv.org/pdf/2508.14856v1)**

> **作者:** Lakshmi Annamalai; Chetan Singh Thakur
>
> **摘要:** Road segmentation is pivotal for autonomous vehicles, yet achieving low latency and low compute solutions using frame based cameras remains a challenge. Event cameras offer a promising alternative. To leverage their low power sensing, we introduce EventSSEG, a method for road segmentation that uses event only computing and a probabilistic attention mechanism. Event only computing poses a challenge in transferring pretrained weights from the conventional camera domain, requiring abundant labeled data, which is scarce. To overcome this, EventSSEG employs event-based self supervised learning, eliminating the need for extensive labeled data. Experiments on DSEC-Semantic and DDD17 show that EventSSEG achieves state of the art performance with minimal labeled events. This approach maximizes event cameras capabilities and addresses the lack of labeled events.
>
---
#### [new 057] Towards PerSense++: Advancing Training-Free Personalized Instance Segmentation in Dense Images
- **分类: cs.CV**

- **简介: 该论文提出PerSense++框架，解决密集图像中遮挡、杂乱背景和尺度变化导致的个性化实例分割难题，通过密度图、点提示筛选及改进模块提升分割鲁棒性，并构建专用基准PerSense-D。**

- **链接: [http://arxiv.org/pdf/2508.14660v1](http://arxiv.org/pdf/2508.14660v1)**

> **作者:** Muhammad Ibraheem Siddiqui; Muhammad Umer Sheikh; Hassan Abid; Kevin Henry; Muhammad Haris Khan
>
> **备注:** arXiv admin note: text overlap with arXiv:2405.13518
>
> **摘要:** Segmentation in dense visual scenes poses significant challenges due to occlusions, background clutter, and scale variations. To address this, we introduce PerSense, an end-to-end, training-free, and model-agnostic one-shot framework for Personalized instance Segmentation in dense images. PerSense employs a novel Instance Detection Module (IDM) that leverages density maps (DMs) to generate instance-level candidate point prompts, followed by a Point Prompt Selection Module (PPSM) that filters false positives via adaptive thresholding and spatial gating. A feedback mechanism further enhances segmentation by automatically selecting effective exemplars to improve DM quality. We additionally present PerSense++, an enhanced variant that incorporates three additional components to improve robustness in cluttered scenes: (i) a diversity-aware exemplar selection strategy that leverages feature and scale diversity for better DM generation; (ii) a hybrid IDM combining contour and peak-based prompt generation for improved instance separation within complex density patterns; and (iii) an Irrelevant Mask Rejection Module (IMRM) that discards spatially inconsistent masks using outlier analysis. Finally, to support this underexplored task, we introduce PerSense-D, a dedicated benchmark for personalized segmentation in dense images. Extensive experiments across multiple benchmarks demonstrate that PerSense++ outperforms existing methods in dense settings.
>
---
#### [new 058] Virtual Community: An Open World for Humans, Robots, and Society
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 论文构建开放世界平台Virtual Community，研究人机协作与社会智能。通过模拟人类与机器人共存场景，设计社区规划与机器人协作挑战，评估多智能体在开放环境中的合作与规划能力。**

- **链接: [http://arxiv.org/pdf/2508.14893v1](http://arxiv.org/pdf/2508.14893v1)**

> **作者:** Qinhong Zhou; Hongxin Zhang; Xiangye Lin; Zheyuan Zhang; Yutian Chen; Wenjun Liu; Zunzhe Zhang; Sunli Chen; Lixing Fang; Qiushi Lyu; Xinyu Sun; Jincheng Yang; Zeyuan Wang; Bao Chi Dang; Zhehuan Chen; Daksha Ladia; Jiageng Liu; Chuang Gan
>
> **备注:** website https://virtual-community-ai.github.io/
>
> **摘要:** The rapid progress in AI and Robotics may lead to a profound societal transformation, as humans and robots begin to coexist within shared communities, introducing both opportunities and challenges. To explore this future, we present Virtual Community-an open-world platform for humans, robots, and society-built on a universal physics engine and grounded in real-world 3D scenes. With Virtual Community, we aim to study embodied social intelligence at scale: 1) How robots can intelligently cooperate or compete; 2) How humans develop social relations and build community; 3) More importantly, how intelligent robots and humans can co-exist in an open world. To support these, Virtual Community features: 1) An open-source multi-agent physics simulator that supports robots, humans, and their interactions within a society; 2) A large-scale, real-world aligned community generation pipeline, including vast outdoor space, diverse indoor scenes, and a community of grounded agents with rich characters and appearances. Leveraging Virtual Community, we propose two novel challenges. The Community Planning Challenge evaluates multi-agent reasoning and planning ability in open-world settings, such as cooperating to help agents with daily activities and efficiently connecting other agents. The Community Robot Challenge requires multiple heterogeneous robots to collaborate in solving complex open-world tasks. We evaluate various baselines on these tasks and demonstrate the challenges in both high-level open-world task planning and low-level cooperation controls. We hope that Virtual Community will unlock further study of human-robot coexistence within open-world environments.
>
---
#### [new 059] HandCraft: Dynamic Sign Generation for Synthetic Data Augmentation
- **分类: cs.CV; cs.LG**

- **简介: 本论文针对手语识别中数据不足的问题，提出基于CMLPe的轻量生成模型与合成数据预训练方法，提升识别准确率，在LSFB和DiSPLaY数据集上取得新SOTA。**

- **链接: [http://arxiv.org/pdf/2508.14345v1](http://arxiv.org/pdf/2508.14345v1)**

> **作者:** Gaston Gustavo Rios
>
> **备注:** 26 pages, 4 figures, 9 tables, code available at https://github.com/okason97/HandCraft
>
> **摘要:** Sign Language Recognition (SLR) models face significant performance limitations due to insufficient training data availability. In this article, we address the challenge of limited data in SLR by introducing a novel and lightweight sign generation model based on CMLPe. This model, coupled with a synthetic data pretraining approach, consistently improves recognition accuracy, establishing new state-of-the-art results for the LSFB and DiSPLaY datasets using our Mamba-SL and Transformer-SL classifiers. Our findings reveal that synthetic data pretraining outperforms traditional augmentation methods in some cases and yields complementary benefits when implemented alongside them. Our approach democratizes sign generation and synthetic data pretraining for SLR by providing computationally efficient methods that achieve significant performance improvements across diverse datasets.
>
---
#### [new 060] TransLight: Image-Guided Customized Lighting Control with Generative Decoupling
- **分类: cs.CV; cs.AI**

- **简介: 本文提出TransLight，解决跨图像光照效果转移中内容完整性与定制控制问题，通过生成解耦分离图像内容与光照，生成百万级数据集并训练模型，实现高保真、灵活的光照控制。**

- **链接: [http://arxiv.org/pdf/2508.14814v1](http://arxiv.org/pdf/2508.14814v1)**

> **作者:** Zongming Li; Lianghui Zhu; Haocheng Shen; Longjin Ran; Wenyu Liu; Xinggang Wang
>
> **备注:** 15 pages, 9 figures
>
> **摘要:** Most existing illumination-editing approaches fail to simultaneously provide customized control of light effects and preserve content integrity. This makes them less effective for practical lighting stylization requirements, especially in the challenging task of transferring complex light effects from a reference image to a user-specified target image. To address this problem, we propose TransLight, a novel framework that enables high-fidelity and high-freedom transfer of light effects. Extracting the light effect from the reference image is the most critical and challenging step in our method. The difficulty lies in the complex geometric structure features embedded in light effects that are highly coupled with content in real-world scenarios. To achieve this, we first present Generative Decoupling, where two fine-tuned diffusion models are used to accurately separate image content and light effects, generating a newly curated, million-scale dataset of image-content-light triplets. Then, we employ IC-Light as the generative model and train our model with our triplets, injecting the reference lighting image as an additional conditioning signal. The resulting TransLight model enables customized and natural transfer of diverse light effects. Notably, by thoroughly disentangling light effects from reference images, our generative decoupling strategy endows TransLight with highly flexible illumination control. Experimental results establish TransLight as the first method to successfully transfer light effects across disparate images, delivering more customized illumination control than existing techniques and charting new directions for research in illumination harmonization and editing.
>
---
#### [new 061] PB-IAD: Utilizing multimodal foundation models for semantic industrial anomaly detection in dynamic manufacturing environments
- **分类: cs.CV; cs.AI**

- **简介: 论文提出PB-IAD框架，利用多模态基础模型进行动态制造环境下的工业异常检测，解决数据稀疏与适应性问题，通过提示模板和预处理模块实现用户定制化，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2508.14504v1](http://arxiv.org/pdf/2508.14504v1)**

> **作者:** Bernd Hofmann; Albert Scheck; Joerg Franke; Patrick Bruendl
>
> **摘要:** The detection of anomalies in manufacturing processes is crucial to ensure product quality and identify process deviations. Statistical and data-driven approaches remain the standard in industrial anomaly detection, yet their adaptability and usability are constrained by the dependence on extensive annotated datasets and limited flexibility under dynamic production conditions. Recent advances in the perception capabilities of foundation models provide promising opportunities for their adaptation to this downstream task. This paper presents PB-IAD (Prompt-based Industrial Anomaly Detection), a novel framework that leverages the multimodal and reasoning capabilities of foundation models for industrial anomaly detection. Specifically, PB-IAD addresses three key requirements of dynamic production environments: data sparsity, agile adaptability, and domain user centricity. In addition to the anomaly detection, the framework includes a prompt template that is specifically designed for iteratively implementing domain-specific process knowledge, as well as a pre-processing module that translates domain user inputs into effective system prompts. This user-centric design allows domain experts to customise the system flexibly without requiring data science expertise. The proposed framework is evaluated by utilizing GPT-4.1 across three distinct manufacturing scenarios, two data modalities, and an ablation study to systematically assess the contribution of semantic instructions. Furthermore, PB-IAD is benchmarked to state-of-the-art methods for anomaly detection such as PatchCore. The results demonstrate superior performance, particularly in data-sparse scenarios and low-shot settings, achieved solely through semantic instructions.
>
---
#### [new 062] Repeating Words for Video-Language Retrieval with Coarse-to-Fine Objectives
- **分类: cs.CV**

- **简介: 论文针对视频-语言检索中的高准确率与低训练成本问题，提出基于粗到细目标的框架，利用细粒度特征和关键词重复策略，结合新指标提升检索性能。**

- **链接: [http://arxiv.org/pdf/2508.14812v1](http://arxiv.org/pdf/2508.14812v1)**

> **作者:** Haoyu Zhao; Jiaxi Gu; Shicong Wang; Xing Zhang; Hang Xu; Zuxuan Wu; Yu-Gang Jiang
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** The explosive growth of video streaming presents challenges in achieving high accuracy and low training costs for video-language retrieval. However, existing methods rely on large-scale pre-training to improve video retrieval performance, resulting in significant computational demands. Additionally, the fine-grained information in videos and texts remains underexplored. To alleviate these problems, we propose a novel framework to learn fine-grained features for better alignment and introduce an inference pipeline to improve performance without additional training. Specifically, we employ coarse-to-fine objectives to understand the semantic information of video-text pairs, including contrastive and matching learning. The fine-grained data used for training is obtained through the Granularity-Aware Representation module, which is designed based on similarity analysis between video frames and words in captions. Furthermore, we observe that the repetition of keywords in the original captions, referred to as "Repetition", can enhance retrieval performance and improve alignment between video and text. Based on this insight, we propose a novel and effective inference pipeline that incorporates a voting mechanism and a new Matching Entropy metric to achieve better retrieval performance without requiring additional pre-training. Experimental results on four benchmarks demonstrate that the proposed method outperforms previous approaches. Additionally, our inference pipeline achieves significant performance improvements, with a 2.1% increase in Recall@1 on the MSR-VTT dataset and a 1.6% increase on the DiDeMo dataset.
>
---
#### [new 063] LookOut: Real-World Humanoid Egocentric Navigation
- **分类: cs.CV**

- **简介: 该论文任务是预测人类自中心视角下的未来6D头部姿态，解决真实环境中基于视觉的导航问题。提出框架融合时空3D特征建模，并构建Aria导航数据集，训练模型学习人类导航行为如避让、绕行等。**

- **链接: [http://arxiv.org/pdf/2508.14466v1](http://arxiv.org/pdf/2508.14466v1)**

> **作者:** Boxiao Pan; Adam W. Harley; C. Karen Liu; Leonidas J. Guibas
>
> **摘要:** The ability to predict collision-free future trajectories from egocentric observations is crucial in applications such as humanoid robotics, VR / AR, and assistive navigation. In this work, we introduce the challenging problem of predicting a sequence of future 6D head poses from an egocentric video. In particular, we predict both head translations and rotations to learn the active information-gathering behavior expressed through head-turning events. To solve this task, we propose a framework that reasons over temporally aggregated 3D latent features, which models the geometric and semantic constraints for both the static and dynamic parts of the environment. Motivated by the lack of training data in this space, we further contribute a data collection pipeline using the Project Aria glasses, and present a dataset collected through this approach. Our dataset, dubbed Aria Navigation Dataset (AND), consists of 4 hours of recording of users navigating in real-world scenarios. It includes diverse situations and navigation behaviors, providing a valuable resource for learning real-world egocentric navigation policies. Extensive experiments show that our model learns human-like navigation behaviors such as waiting / slowing down, rerouting, and looking around for traffic while generalizing to unseen environments. Check out our project webpage at https://sites.google.com/stanford.edu/lookout.
>
---
#### [new 064] Img2ST-Net: Efficient High-Resolution Spatial Omics Prediction from Whole Slide Histology Images via Fully Convolutional Image-to-Image Learning
- **分类: cs.CV**

- **简介: 论文提出Img2ST-Net，通过全卷积网络从组织切片图像高效预测高分辨率空间转录组数据，解决传统方法在高分辨率下的效率与稳定性问题，引入SSIM-ST评估指标。**

- **链接: [http://arxiv.org/pdf/2508.14393v1](http://arxiv.org/pdf/2508.14393v1)**

> **作者:** Junchao Zhu; Ruining Deng; Junlin Guo; Tianyuan Yao; Juming Xiong; Chongyu Qu; Mengmeng Yin; Yu Wang; Shilin Zhao; Haichun Yang; Daguang Xu; Yucheng Tang; Yuankai Huo
>
> **摘要:** Recent advances in multi-modal AI have demonstrated promising potential for generating the currently expensive spatial transcriptomics (ST) data directly from routine histology images, offering a means to reduce the high cost and time-intensive nature of ST data acquisition. However, the increasing resolution of ST, particularly with platforms such as Visium HD achieving 8um or finer, introduces significant computational and modeling challenges. Conventional spot-by-spot sequential regression frameworks become inefficient and unstable at this scale, while the inherent extreme sparsity and low expression levels of high-resolution ST further complicate both prediction and evaluation. To address these limitations, we propose Img2ST-Net, a novel histology-to-ST generation framework for efficient and parallel high-resolution ST prediction. Unlike conventional spot-by-spot inference methods, Img2ST-Net employs a fully convolutional architecture to generate dense, HD gene expression maps in a parallelized manner. By modeling HD ST data as super-pixel representations, the task is reformulated from image-to-omics inference into a super-content image generation problem with hundreds or thousands of output channels. This design not only improves computational efficiency but also better preserves the spatial organization intrinsic to spatial omics data. To enhance robustness under sparse expression patterns, we further introduce SSIM-ST, a structural-similarity-based evaluation metric tailored for high-resolution ST analysis. We present a scalable, biologically coherent framework for high-resolution ST prediction. Img2ST-Net offers a principled solution for efficient and accurate ST inference at scale. Our contributions lay the groundwork for next-generation ST modeling that is robust and resolution-aware. The source code has been made publicly available at https://github.com/hrlblab/Img2ST-Net.
>
---
#### [new 065] DINOv3 with Test-Time Training for Medical Image Registration
- **分类: cs.CV; cs.AI**

- **简介: 论文针对医学图像配准中训练数据不足的问题，提出基于DINOv3的测试时训练方法，无需额外训练即可实现高精度配准，实验表明在Abdomen MR-CT和ACDC数据集上均取得优异性能。**

- **链接: [http://arxiv.org/pdf/2508.14809v1](http://arxiv.org/pdf/2508.14809v1)**

> **作者:** Shansong Wang; Mojtaba Safari; Mingzhe Hu; Qiang Li; Chih-Wei Chang; Richard LJ Qiu; Xiaofeng Yang
>
> **摘要:** Prior medical image registration approaches, particularly learning-based methods, often require large amounts of training data, which constrains clinical adoption. To overcome this limitation, we propose a training-free pipeline that relies on a frozen DINOv3 encoder and test-time optimization of the deformation field in feature space. Across two representative benchmarks, the method is accurate and yields regular deformations. On Abdomen MR-CT, it attained the best mean Dice score (DSC) of 0.790 together with the lowest 95th percentile Hausdorff Distance (HD95) of 4.9+-5.0 and the lowest standard deviation of Log-Jacobian (SDLogJ) of 0.08+-0.02. On ACDC cardiac MRI, it improves mean DSC to 0.769 and reduces SDLogJ to 0.11 and HD95 to 4.8, a marked gain over the initial alignment. The results indicate that operating in a compact foundation feature space at test time offers a practical and general solution for clinical registration without additional training.
>
---
#### [new 066] Improved Mapping Between Illuminations and Sensors for RAW Images
- **分类: cs.CV**

- **简介: 论文旨在改进RAW图像的光照-传感器映射，解决跨传感器和光照条件的数据采集难题。通过构建包含390种光照、四台相机的定制数据集，提出轻量神经网络方法，实现高效映射并提升神经ISP训练效果。**

- **链接: [http://arxiv.org/pdf/2508.14730v1](http://arxiv.org/pdf/2508.14730v1)**

> **作者:** Abhijith Punnappurath; Luxi Zhao; Hoang Le; Abdelrahman Abdelhamed; SaiKiran Kumar Tedla; Michael S. Brown
>
> **摘要:** RAW images are unprocessed camera sensor output with sensor-specific RGB values based on the sensor's color filter spectral sensitivities. RAW images also incur strong color casts due to the sensor's response to the spectral properties of scene illumination. The sensor- and illumination-specific nature of RAW images makes it challenging to capture RAW datasets for deep learning methods, as scenes need to be captured for each sensor and under a wide range of illumination. Methods for illumination augmentation for a given sensor and the ability to map RAW images between sensors are important for reducing the burden of data capture. To explore this problem, we introduce the first-of-its-kind dataset comprising carefully captured scenes under a wide range of illumination. Specifically, we use a customized lightbox with tunable illumination spectra to capture several scenes with different cameras. Our illumination and sensor mapping dataset has 390 illuminations, four cameras, and 18 scenes. Using this dataset, we introduce a lightweight neural network approach for illumination and sensor mapping that outperforms competing methods. We demonstrate the utility of our approach on the downstream task of training a neural ISP. Link to project page: https://github.com/SamsungLabs/illum-sensor-mapping.
>
---
#### [new 067] Effect of Data Augmentation on Conformal Prediction for Diabetic Retinopathy
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究数据增强对糖尿病视网膜病变（DR）分级中置信预测（CP）可靠性的影响，旨在提升模型不确定性量化能力，通过对比不同增强策略验证其对置信度指标的作用。**

- **链接: [http://arxiv.org/pdf/2508.14266v1](http://arxiv.org/pdf/2508.14266v1)**

> **作者:** Rizwan Ahamed; Annahita Amireskandari; Joel Palko; Carol Laxson; Binod Bhattarai; Prashnna Gyawali
>
> **备注:** 3rd Workshop in Data Engineering in Medical Imaging (DEMI), MICCAI-2025 Workshop
>
> **摘要:** The clinical deployment of deep learning models for high-stakes tasks such as diabetic retinopathy (DR) grading requires demonstrable reliability. While models achieve high accuracy, their clinical utility is limited by a lack of robust uncertainty quantification. Conformal prediction (CP) offers a distribution-free framework to generate prediction sets with statistical guarantees of coverage. However, the interaction between standard training practices like data augmentation and the validity of these guarantees is not well understood. In this study, we systematically investigate how different data augmentation strategies affect the performance of conformal predictors for DR grading. Using the DDR dataset, we evaluate two backbone architectures -- ResNet-50 and a Co-Scale Conv-Attentional Transformer (CoaT) -- trained under five augmentation regimes: no augmentation, standard geometric transforms, CLAHE, Mixup, and CutMix. We analyze the downstream effects on conformal metrics, including empirical coverage, average prediction set size, and correct efficiency. Our results demonstrate that sample-mixing strategies like Mixup and CutMix not only improve predictive accuracy but also yield more reliable and efficient uncertainty estimates. Conversely, methods like CLAHE can negatively impact model certainty. These findings highlight the need to co-design augmentation strategies with downstream uncertainty quantification in mind to build genuinely trustworthy AI systems for medical imaging.
>
---
#### [new 068] Making Pose Representations More Expressive and Disentangled via Residual Vector Quantization
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对文本到动作生成中姿态代码表达不足的问题，提出残差向量量化方法，增强姿态表示的表达性与解耦性，提升运动细节捕捉能力。**

- **链接: [http://arxiv.org/pdf/2508.14561v1](http://arxiv.org/pdf/2508.14561v1)**

> **作者:** Sukhyun Jeong; Hong-Gi Shin; Yong-Hoon Choi
>
> **摘要:** Recent progress in text-to-motion has advanced both 3D human motion generation and text-based motion control. Controllable motion generation (CoMo), which enables intuitive control, typically relies on pose code representations, but discrete pose codes alone cannot capture fine-grained motion details, limiting expressiveness. To overcome this, we propose a method that augments pose code-based latent representations with continuous motion features using residual vector quantization (RVQ). This design preserves the interpretability and manipulability of pose codes while effectively capturing subtle motion characteristics such as high-frequency details. Experiments on the HumanML3D dataset show that our model reduces Frechet inception distance (FID) from 0.041 to 0.015 and improves Top-1 R-Precision from 0.508 to 0.510. Qualitative analysis of pairwise direction similarity between pose codes further confirms the model's controllability for motion editing.
>
---
#### [new 069] MUSE: Multi-Subject Unified Synthesis via Explicit Layout Semantic Expansion
- **分类: cs.CV**

- **简介: 论文针对布局可控的多主体合成任务，解决空间精确控制与主体身份保持的矛盾，提出MUSE框架，通过拼接交叉注意力（CCA）与分阶段训练实现统一合成，提升空间精度和身份一致性。**

- **链接: [http://arxiv.org/pdf/2508.14440v1](http://arxiv.org/pdf/2508.14440v1)**

> **作者:** Fei Peng; Junqiang Wu; Yan Li; Tingting Gao; Di Zhang; Huiyuan Fu
>
> **备注:** This paper is accepted by ICCV 2025
>
> **摘要:** Existing text-to-image diffusion models have demonstrated remarkable capabilities in generating high-quality images guided by textual prompts. However, achieving multi-subject compositional synthesis with precise spatial control remains a significant challenge. In this work, we address the task of layout-controllable multi-subject synthesis (LMS), which requires both faithful reconstruction of reference subjects and their accurate placement in specified regions within a unified image. While recent advancements have separately improved layout control and subject synthesis, existing approaches struggle to simultaneously satisfy the dual requirements of spatial precision and identity preservation in this composite task. To bridge this gap, we propose MUSE, a unified synthesis framework that employs concatenated cross-attention (CCA) to seamlessly integrate layout specifications with textual guidance through explicit semantic space expansion. The proposed CCA mechanism enables bidirectional modality alignment between spatial constraints and textual descriptions without interference. Furthermore, we design a progressive two-stage training strategy that decomposes the LMS task into learnable sub-objectives for effective optimization. Extensive experiments demonstrate that MUSE achieves zero-shot end-to-end generation with superior spatial accuracy and identity consistency compared to existing solutions, advancing the frontier of controllable image synthesis. Our code and model are available at https://github.com/pf0607/MUSE.
>
---
#### [new 070] Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles
- **分类: cs.CV**

- **简介: 该论文旨在生成安全关键场景以提升自动驾驶安全性。针对现有方法依赖预设威胁模式的局限，提出ScenGE框架，通过大语言模型生成对抗性案例并协同演化交通流，生成多样化威胁场景，实验证明其能发现更多碰撞案例，验证场景有效性。**

- **链接: [http://arxiv.org/pdf/2508.14527v1](http://arxiv.org/pdf/2508.14527v1)**

> **作者:** Jiangfan Liu; Yongkang Guo; Fangzhi Zhong; Tianyuan Zhang; Zonglei Jing; Siyuan Liang; Jiakai Wang; Mingchuan Zhang; Aishan Liu; Xianglong Liu
>
> **摘要:** The generation of safety-critical scenarios in simulation has become increasingly crucial for safety evaluation in autonomous vehicles prior to road deployment in society. However, current approaches largely rely on predefined threat patterns or rule-based strategies, which limit their ability to expose diverse and unforeseen failure modes. To overcome these, we propose ScenGE, a framework that can generate plentiful safety-critical scenarios by reasoning novel adversarial cases and then amplifying them with complex traffic flows. Given a simple prompt of a benign scene, it first performs Meta-Scenario Generation, where a large language model, grounded in structured driving knowledge, infers an adversarial agent whose behavior poses a threat that is both plausible and deliberately challenging. This meta-scenario is then specified in executable code for precise in-simulator control. Subsequently, Complex Scenario Evolution uses background vehicles to amplify the core threat introduced by Meta-Scenario. It builds an adversarial collaborator graph to identify key agent trajectories for optimization. These perturbations are designed to simultaneously reduce the ego vehicle's maneuvering space and create critical occlusions. Extensive experiments conducted on multiple reinforcement learning based AV models show that ScenGE uncovers more severe collision cases (+31.96%) on average than SoTA baselines. Additionally, our ScenGE can be applied to large model based AV systems and deployed on different simulators; we further observe that adversarial training on our scenarios improves the model robustness. Finally, we validate our framework through real-world vehicle tests and human evaluation, confirming that the generated scenarios are both plausible and critical. We hope our paper can build up a critical step towards building public trust and ensuring their safe deployment.
>
---
#### [new 071] GOGS: High-Fidelity Geometry and Relighting for Glossy Objects via Gaussian Surfels
- **分类: cs.CV**

- **简介: 论文针对光泽物体的逆渲染问题，提出GOGS框架，通过两阶段方法结合物理渲染与高斯surfels，解决计算成本及光照重构难题，实现高保真几何重建与真实 relighting。**

- **链接: [http://arxiv.org/pdf/2508.14563v1](http://arxiv.org/pdf/2508.14563v1)**

> **作者:** Xingyuan Yang; Min Wei
>
> **备注:** 13 pages, 13 figures
>
> **摘要:** Inverse rendering of glossy objects from RGB imagery remains fundamentally limited by inherent ambiguity. Although NeRF-based methods achieve high-fidelity reconstruction via dense-ray sampling, their computational cost is prohibitive. Recent 3D Gaussian Splatting achieves high reconstruction efficiency but exhibits limitations under specular reflections. Multi-view inconsistencies introduce high-frequency surface noise and structural artifacts, while simplified rendering equations obscure material properties, leading to implausible relighting results. To address these issues, we propose GOGS, a novel two-stage framework based on 2D Gaussian surfels. First, we establish robust surface reconstruction through physics-based rendering with split-sum approximation, enhanced by geometric priors from foundation models. Second, we perform material decomposition by leveraging Monte Carlo importance sampling of the full rendering equation, modeling indirect illumination via differentiable 2D Gaussian ray tracing and refining high-frequency specular details through spherical mipmap-based directional encoding that captures anisotropic highlights. Extensive experiments demonstrate state-of-the-art performance in geometry reconstruction, material separation, and photorealistic relighting under novel illuminations, outperforming existing inverse rendering approaches.
>
---
#### [new 072] Pixels to Play: A Foundation Model for 3D Gameplay
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出Pixels2Play-0.1，通过端到端行为克隆学习3D游戏玩法，利用人类演示与公开视频训练，采用解码器Transformer处理大动作空间，实现跨游戏泛化，解决AI代理需依赖原始像素流并适应新游戏的挑战。**

- **链接: [http://arxiv.org/pdf/2508.14295v1](http://arxiv.org/pdf/2508.14295v1)**

> **作者:** Yuguang Yue; Chris Green; Samuel Hunt; Irakli Salia; Wenzhe Shi; Jonathan J Hunt
>
> **摘要:** We introduce Pixels2Play-0.1 (P2P0.1), a foundation model that learns to play a wide range of 3D video games with recognizable human-like behavior. Motivated by emerging consumer and developer use cases - AI teammates, controllable NPCs, personalized live-streamers, assistive testers - we argue that an agent must rely on the same pixel stream available to players and generalize to new titles with minimal game-specific engineering. P2P0.1 is trained end-to-end with behavior cloning: labeled demonstrations collected from instrumented human game-play are complemented by unlabeled public videos, to which we impute actions via an inverse-dynamics model. A decoder-only transformer with auto-regressive action output handles the large action space while remaining latency-friendly on a single consumer GPU. We report qualitative results showing competent play across simple Roblox and classic MS-DOS titles, ablations on unlabeled data, and outline the scaling and evaluation steps required to reach expert-level, text-conditioned control.
>
---
#### [new 073] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对点云视频时空混乱问题，提出UST-SSM模型，通过空间-时间重构与特征聚合，实现对动态3D动作的有效建模与识别。**

- **链接: [http://arxiv.org/pdf/2508.14604v1](http://arxiv.org/pdf/2508.14604v1)**

> **作者:** Peiming Li; Ziyi Wang; Yulin Yuan; Hong Liu; Xiangming Meng; Junsong Yuan; Mengyuan Liu
>
> **备注:** 8 pages, 5 figures, Accepted to ICCV2025
>
> **摘要:** Point cloud videos capture dynamic 3D motion while reducing the effects of lighting and viewpoint variations, making them highly effective for recognizing subtle and continuous human actions. Although Selective State Space Models (SSMs) have shown good performance in sequence modeling with linear complexity, the spatio-temporal disorder of point cloud videos hinders their unidirectional modeling when directly unfolding the point cloud video into a 1D sequence through temporally sequential scanning. To address this challenge, we propose the Unified Spatio-Temporal State Space Model (UST-SSM), which extends the latest advancements in SSMs to point cloud videos. Specifically, we introduce Spatial-Temporal Selection Scanning (STSS), which reorganizes unordered points into semantic-aware sequences through prompt-guided clustering, thereby enabling the effective utilization of points that are spatially and temporally distant yet similar within the sequence. For missing 4D geometric and motion details, Spatio-Temporal Structure Aggregation (STSA) aggregates spatio-temporal features and compensates. To improve temporal interaction within the sampled sequence, Temporal Interaction Sampling (TIS) enhances fine-grained temporal dependencies through non-anchor frame utilization and expanded receptive fields. Experimental results on the MSR-Action3D, NTU RGB+D, and Synthia 4D datasets validate the effectiveness of our method. Our code is available at https://github.com/wangzy01/UST-SSM.
>
---
#### [new 074] 3D Cardiac Anatomy Generation Using Mesh Latent Diffusion Models
- **分类: eess.IV; cs.CV; cs.LG; q-bio.TO**

- **简介: 论文任务为生成3D心脏解剖结构，解决医学影像中扩散模型应用不足的问题，提出MeshLDM模型并验证其在急性心肌梗死患者数据集上的有效性。**

- **链接: [http://arxiv.org/pdf/2508.14122v1](http://arxiv.org/pdf/2508.14122v1)**

> **作者:** Jolanta Mozyrska; Marcel Beetz; Luke Melas-Kyriazi; Vicente Grau; Abhirup Banerjee; Alfonso Bueno-Orovio
>
> **摘要:** Diffusion models have recently gained immense interest for their generative capabilities, specifically the high quality and diversity of the synthesized data. However, examples of their applications in 3D medical imaging are still scarce, especially in cardiology. Generating diverse realistic cardiac anatomies is crucial for applications such as in silico trials, electromechanical computer simulations, or data augmentations for machine learning models. In this work, we investigate the application of Latent Diffusion Models (LDMs) for generating 3D meshes of human cardiac anatomies. To this end, we propose a novel LDM architecture -- MeshLDM. We apply the proposed model on a dataset of 3D meshes of left ventricular cardiac anatomies from patients with acute myocardial infarction and evaluate its performance in terms of both qualitative and quantitative clinical and 3D mesh reconstruction metrics. The proposed MeshLDM successfully captures characteristics of the cardiac shapes at end-diastolic (relaxation) and end-systolic (contraction) cardiac phases, generating meshes with a 2.4% difference in population mean compared to the gold standard.
>
---
#### [new 075] Activity Coefficient-based Channel Selection for Electroencephalogram: A Task-Independent Approach
- **分类: q-bio.NC; cs.CV; cs.HC; cs.LG; eess.SP**

- **简介: 论文针对脑机接口中高密度EEG通道选择问题，提出任务无关的ACCS方法，通过Channel Activity Coefficient量化通道效用，选择前16通道提升多类分类准确率至34.97%，适应多样应用。**

- **链接: [http://arxiv.org/pdf/2508.14060v1](http://arxiv.org/pdf/2508.14060v1)**

> **作者:** Kartik Pandey; Arun Balasubramanian; Debasis Samanta
>
> **摘要:** Electroencephalogram (EEG) signals have gained widespread adoption in brain-computer interface (BCI) applications due to their non-invasive, low-cost, and relatively simple acquisition process. The demand for higher spatial resolution, particularly in clinical settings, has led to the development of high-density electrode arrays. However, increasing the number of channels introduces challenges such as cross-channel interference and computational overhead. To address these issues, modern BCI systems often employ channel selection algorithms. Existing methods, however, are typically task-specific and require re-optimization for each new application. This work proposes a task-agnostic channel selection method, Activity Coefficient-based Channel Selection (ACCS), which uses a novel metric called the Channel Activity Coefficient (CAC) to quantify channel utility based on activity levels. By selecting the top 16 channels ranked by CAC, ACCS achieves up to 34.97% improvement in multi-class classification accuracy. Unlike traditional approaches, ACCS identifies a reusable set of informative channels independent of the downstream task or model, making it highly adaptable for diverse EEG-based applications.
>
---
#### [new 076] Deep Skin Lesion Segmentation with Transformer-CNN Fusion: Toward Intelligent Skin Cancer Analysis
- **分类: eess.IV; cs.CV**

- **简介: 该论文解决皮肤病变图像分割中的复杂结构、模糊边界和尺度变化问题，提出融合Transformer与CNN的改进TransUNet模型，通过全局语义建模和多尺度上采样提升分割精度，实验验证其在多个指标上优于现有方法，适用于自动化皮肤癌分析。**

- **链接: [http://arxiv.org/pdf/2508.14509v1](http://arxiv.org/pdf/2508.14509v1)**

> **作者:** Xin Wang; Xiaopei Zhang; Xingang Wang
>
> **摘要:** This paper proposes a high-precision semantic segmentation method based on an improved TransUNet architecture to address the challenges of complex lesion structures, blurred boundaries, and significant scale variations in skin lesion images. The method integrates a transformer module into the traditional encoder-decoder framework to model global semantic information, while retaining a convolutional branch to preserve local texture and edge features. This enhances the model's ability to perceive fine-grained structures. A boundary-guided attention mechanism and multi-scale upsampling path are also designed to improve lesion boundary localization and segmentation consistency. To verify the effectiveness of the approach, a series of experiments were conducted, including comparative studies, hyperparameter sensitivity analysis, data augmentation effects, input resolution variation, and training data split ratio tests. Experimental results show that the proposed model outperforms existing representative methods in mIoU, mDice, and mAcc, demonstrating stronger lesion recognition accuracy and robustness. In particular, the model achieves better boundary reconstruction and structural recovery in complex scenarios, making it well-suited for the key demands of automated segmentation tasks in skin lesion analysis.
>
---
#### [new 077] Fracture Detection and Localisation in Wrist and Hand Radiographs using Detection Transformer Variants
- **分类: eess.IV; cs.AI; cs.CV; 68T45; I.2.10**

- **简介: 该论文旨在通过检测变压器变体实现手腕和手部X光片中骨折的检测与定位，解决手动解读效率低的问题，采用RT-DETR/Co-DETR模型微调及对比学习，实现实时高精度诊断。**

- **链接: [http://arxiv.org/pdf/2508.14129v1](http://arxiv.org/pdf/2508.14129v1)**

> **作者:** Aditya Bagri; Vasanthakumar Venugopal; Anandakumar D; Revathi Ezhumalai; Kalyan Sivasailam; Bargava Subramanian; VarshiniPriya; Meenakumari K S; Abi M; Renita S
>
> **备注:** 18 pages, 21 figures
>
> **摘要:** Background: Accurate diagnosis of wrist and hand fractures using radiographs is essential in emergency care, but manual interpretation is slow and prone to errors. Transformer-based models show promise in improving medical image analysis, but their application to extremity fractures is limited. This study addresses this gap by applying object detection transformers to wrist and hand X-rays. Methods: We fine-tuned the RT-DETR and Co-DETR models, pre-trained on COCO, using over 26,000 annotated X-rays from a proprietary clinical dataset. Each image was labeled for fracture presence with bounding boxes. A ResNet-50 classifier was trained on cropped regions to refine abnormality classification. Supervised contrastive learning was used to enhance embedding quality. Performance was evaluated using AP@50, precision, and recall metrics, with additional testing on real-world X-rays. Results: RT-DETR showed moderate results (AP@50 = 0.39), while Co-DETR outperformed it with an AP@50 of 0.615 and faster convergence. The integrated pipeline achieved 83.1% accuracy, 85.1% precision, and 96.4% recall on real-world X-rays, demonstrating strong generalization across 13 fracture types. Visual inspection confirmed accurate localization. Conclusion: Our Co-DETR-based pipeline demonstrated high accuracy and clinical relevance in wrist and hand fracture detection, offering reliable localization and differentiation of fracture types. It is scalable, efficient, and suitable for real-time deployment in hospital workflows, improving diagnostic speed and reliability in musculoskeletal radiology.
>
---
#### [new 078] ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 该论文提出ShizhenGPT，旨在解决中医多模态诊断数据不足与多感官融合难题，通过构建包含100GB文本和200GB多模态数据的超大规模 dataset，实现中医视觉理解与跨模态推理，推动中医智能化诊断。**

- **链接: [http://arxiv.org/pdf/2508.14706v1](http://arxiv.org/pdf/2508.14706v1)**

> **作者:** Junying Chen; Zhenyang Cai; Zhiheng Liu; Yunjin Yang; Rongsheng Wang; Qingying Xiao; Xiangyi Feng; Zhan Su; Jing Guo; Xiang Wan; Guangjun Yu; Haizhou Li; Benyou Wang
>
> **摘要:** Despite the success of large language models (LLMs) in various domains, their potential in Traditional Chinese Medicine (TCM) remains largely underexplored due to two critical barriers: (1) the scarcity of high-quality TCM data and (2) the inherently multimodal nature of TCM diagnostics, which involve looking, listening, smelling, and pulse-taking. These sensory-rich modalities are beyond the scope of conventional LLMs. To address these challenges, we present ShizhenGPT, the first multimodal LLM tailored for TCM. To overcome data scarcity, we curate the largest TCM dataset to date, comprising 100GB+ of text and 200GB+ of multimodal data, including 1.2M images, 200 hours of audio, and physiological signals. ShizhenGPT is pretrained and instruction-tuned to achieve deep TCM knowledge and multimodal reasoning. For evaluation, we collect recent national TCM qualification exams and build a visual benchmark for Medicinal Recognition and Visual Diagnosis. Experiments demonstrate that ShizhenGPT outperforms comparable-scale LLMs and competes with larger proprietary models. Moreover, it leads in TCM visual understanding among existing multimodal LLMs and demonstrates unified perception across modalities like sound, pulse, smell, and vision, paving the way toward holistic multimodal perception and diagnosis in TCM. Datasets, models, and code are publicly available. We hope this work will inspire further exploration in this field.
>
---
#### [new 079] Disentanglement in T-space for Faster and Distributed Training of Diffusion Models with Fewer Latent-states
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对扩散模型训练中的潜在状态数量问题，提出通过T空间解耦技术，用更少的潜在状态（如T=32或1）实现高效训练，结合多单状态模型生成高质量样本，实验显示收敛速度提升4-6倍。**

- **链接: [http://arxiv.org/pdf/2508.14413v1](http://arxiv.org/pdf/2508.14413v1)**

> **作者:** Samarth Gupta; Raghudeep Gadde; Rui Chen; Aleix M. Martinez
>
> **摘要:** We challenge a fundamental assumption of diffusion models, namely, that a large number of latent-states or time-steps is required for training so that the reverse generative process is close to a Gaussian. We first show that with careful selection of a noise schedule, diffusion models trained over a small number of latent states (i.e. $T \sim 32$) match the performance of models trained over a much large number of latent states ($T \sim 1,000$). Second, we push this limit (on the minimum number of latent states required) to a single latent-state, which we refer to as complete disentanglement in T-space. We show that high quality samples can be easily generated by the disentangled model obtained by combining several independently trained single latent-state models. We provide extensive experiments to show that the proposed disentangled model provides 4-6$\times$ faster convergence measured across a variety of metrics on two different datasets.
>
---
#### [new 080] From Slices to Structures: Unsupervised 3D Reconstruction of Female Pelvic Anatomy from Freehand Transvaginal Ultrasound
- **分类: eess.IV; cs.CV**

- **简介: 该论文任务是无监督的3D重建，解决从自由手2D经阴道超声图像重建女性盆腔解剖结构的问题。通过借鉴高斯溅射原理，设计切片感知光栅化器，直接优化各向异性3D高斯参数，实现无需外部设备的高效体积重建。**

- **链接: [http://arxiv.org/pdf/2508.14552v1](http://arxiv.org/pdf/2508.14552v1)**

> **作者:** Max Krähenmann; Sergio Tascon-Morales; Fabian Laumer; Julia E. Vogt; Ece Ozkan
>
> **摘要:** Volumetric ultrasound has the potential to significantly improve diagnostic accuracy and clinical decision-making, yet its widespread adoption remains limited by dependence on specialized hardware and restrictive acquisition protocols. In this work, we present a novel unsupervised framework for reconstructing 3D anatomical structures from freehand 2D transvaginal ultrasound (TVS) sweeps, without requiring external tracking or learned pose estimators. Our method adapts the principles of Gaussian Splatting to the domain of ultrasound, introducing a slice-aware, differentiable rasterizer tailored to the unique physics and geometry of ultrasound imaging. We model anatomy as a collection of anisotropic 3D Gaussians and optimize their parameters directly from image-level supervision, leveraging sensorless probe motion estimation and domain-specific geometric priors. The result is a compact, flexible, and memory-efficient volumetric representation that captures anatomical detail with high spatial fidelity. This work demonstrates that accurate 3D reconstruction from 2D ultrasound images can be achieved through purely computational means, offering a scalable alternative to conventional 3D systems and enabling new opportunities for AI-assisted analysis and diagnosis.
>
---
#### [new 081] Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model
- **分类: eess.IV; cs.CV**

- **简介: 论文通过条件扩散模型生成多色染色图像，解决多色数据获取成本高、H&E与多色图像不匹配的问题，基于H&E图像生成多色染色，提升准确性和效率。**

- **链接: [http://arxiv.org/pdf/2508.14681v1](http://arxiv.org/pdf/2508.14681v1)**

> **作者:** Hyun-Jic Oh; Junsik Kim; Zhiyi Shi; Yichen Wu; Yu-An Chen; Peter K. Sorger; Hanspeter Pfister; Won-Ki Jeong
>
> **摘要:** Multiplex imaging is revolutionizing pathology by enabling the simultaneous visualization of multiple biomarkers within tissue samples, providing molecular-level insights that traditional hematoxylin and eosin (H&E) staining cannot provide. However, the complexity and cost of multiplex data acquisition have hindered its widespread adoption. Additionally, most existing large repositories of H&E images lack corresponding multiplex images, limiting opportunities for multimodal analysis. To address these challenges, we leverage recent advances in latent diffusion models (LDMs), which excel at modeling complex data distributions utilizing their powerful priors for fine-tuning to a target domain. In this paper, we introduce a novel framework for virtual multiplex staining that utilizes pretrained LDM parameters to generate multiplex images from H&E images using a conditional diffusion model. Our approach enables marker-by-marker generation by conditioning the diffusion model on each marker, while sharing the same architecture across all markers. To tackle the challenge of varying pixel value distributions across different marker stains and to improve inference speed, we fine-tune the model for single-step sampling, enhancing both color contrast fidelity and inference efficiency through pixel-level loss functions. We validate our framework on two publicly available datasets, notably demonstrating its effectiveness in generating up to 18 different marker types with improved accuracy, a substantial increase over the 2-3 marker types achieved in previous approaches. This validation highlights the potential of our framework, pioneering virtual multiplex staining. Finally, this paper bridges the gap between H&E and multiplex imaging, potentially enabling retrospective studies and large-scale analyses of existing H&E image repositories.
>
---
#### [new 082] MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds
- **分类: cs.GR; cs.CV**

- **简介: 论文任务是通过LLM将点云转化为可编辑的Blender脚本，解决现有方法受限于DSL和小数据集的问题，构建大规模代码数据集并训练模型实现复杂几何重建与编辑。**

- **链接: [http://arxiv.org/pdf/2508.14879v1](http://arxiv.org/pdf/2508.14879v1)**

> **作者:** Bingquan Dai; Li Ray Luo; Qihong Tang; Jie Wang; Xinyu Lian; Hao Xu; Minghan Qin; Xudong Xu; Bo Dai; Haoqian Wang; Zhaoyang Lyu; Jiangmiao Pang
>
> **摘要:** Reconstructing 3D objects into editable programs is pivotal for applications like reverse engineering and shape editing. However, existing methods often rely on limited domain-specific languages (DSLs) and small-scale datasets, restricting their ability to model complex geometries and structures. To address these challenges, we introduce MeshCoder, a novel framework that reconstructs complex 3D objects from point clouds into editable Blender Python scripts. We develop a comprehensive set of expressive Blender Python APIs capable of synthesizing intricate geometries. Leveraging these APIs, we construct a large-scale paired object-code dataset, where the code for each object is decomposed into distinct semantic parts. Subsequently, we train a multimodal large language model (LLM) that translates 3D point cloud into executable Blender Python scripts. Our approach not only achieves superior performance in shape-to-code reconstruction tasks but also facilitates intuitive geometric and topological editing through convenient code modifications. Furthermore, our code-based representation enhances the reasoning capabilities of LLMs in 3D shape understanding tasks. Together, these contributions establish MeshCoder as a powerful and flexible solution for programmatic 3D shape reconstruction and understanding.
>
---
#### [new 083] Physics-Constrained Diffusion Reconstruction with Posterior Correction for Quantitative and Fast PET Imaging
- **分类: physics.med-ph; cs.CV**

- **简介: 该论文提出一种结合物理约束的扩散模型（PET-DPC），用于快速且定量准确的PET图像重建。通过后验校正修正散射、衰减等物理效应，优于传统方法和深度学习模型，在速度与精度上均取得显著提升。**

- **链接: [http://arxiv.org/pdf/2508.14364v1](http://arxiv.org/pdf/2508.14364v1)**

> **作者:** Yucun Hou; Fenglin Zhan; Chenxi Li; Ziquan Yuan; Haoyu Lu; Yue Chen; Yihao Chen; Kexin Wang; Runze Liao; Haoqi Wen; Ganxi Du; Jiaru Ni; Taoran Chen; Jinyue Zhang; Jigang Yang; Jianyong Jiang
>
> **摘要:** Deep learning-based reconstruction of positron emission tomography(PET) data has gained increasing attention in recent years. While these methods achieve fast reconstruction,concerns remain regarding quantitative accuracy and the presence of artifacts,stemming from limited model interpretability,data driven dependence, and overfitting risks.These challenges have hindered clinical adoption.To address them,we propose a conditional diffusion model with posterior physical correction (PET-DPC) for PET image reconstruction. An innovative normalization procedure generates the input Geometric TOF Probabilistic Image (GTP-image),while physical information is incorporated during the diffusion sampling process to perform posterior scatter,attenuation,and random corrections. The model was trained and validated on 300 brain and 50 whole-body PET datasets,a physical phantom,and 20 simulated brain datasets. PET-DPC produced reconstructions closely aligned with fully corrected OSEM images,outperforming end-to-end deep learning models in quantitative metrics and,in some cases, surpassing traditional iterative methods. The model also generalized well to out-of-distribution(OOD) data. Compared to iterative methods,PET-DPC reduced reconstruction time by 50% for brain scans and 85% for whole-body scans. Ablation studies confirmed the critical role of posterior correction in implementing scatter and attenuation corrections,enhancing reconstruction accuracy. Experiments with physical phantoms further demonstrated PET-DPC's ability to preserve background uniformity and accurately reproduce tumor-to-background intensity ratios. Overall,these results highlight PET-DPC as a promising approach for rapid, quantitatively accurate PET reconstruction,with strong potential to improve clinical imaging workflows.
>
---
#### [new 084] Snap-Snap: Taking Two Images to Reconstruct 3D Human Gaussians in Milliseconds
- **分类: cs.GR; cs.CV**

- **简介: 论文任务是通过两张图像（前视与背视）快速重建3D人体高斯分布。解决从稀疏视角下保持3D一致性与补全缺失颜色信息的难题，通过优化模型与增强算法实现毫秒级高效重建。**

- **链接: [http://arxiv.org/pdf/2508.14892v1](http://arxiv.org/pdf/2508.14892v1)**

> **作者:** Jia Lu; Taoran Yi; Jiemin Fang; Chen Yang; Chuiyun Wu; Wei Shen; Wenyu Liu; Qi Tian; Xinggang Wang
>
> **备注:** Project page: https://hustvl.github.io/Snap-Snap/
>
> **摘要:** Reconstructing 3D human bodies from sparse views has been an appealing topic, which is crucial to broader the related applications. In this paper, we propose a quite challenging but valuable task to reconstruct the human body from only two images, i.e., the front and back view, which can largely lower the barrier for users to create their own 3D digital humans. The main challenges lie in the difficulty of building 3D consistency and recovering missing information from the highly sparse input. We redesign a geometry reconstruction model based on foundation reconstruction models to predict consistent point clouds even input images have scarce overlaps with extensive human data training. Furthermore, an enhancement algorithm is applied to supplement the missing color information, and then the complete human point clouds with colors can be obtained, which are directly transformed into 3D Gaussians for better rendering quality. Experiments show that our method can reconstruct the entire human in 190 ms on a single NVIDIA RTX 4090, with two images at a resolution of 1024x1024, demonstrating state-of-the-art performance on the THuman2.0 and cross-domain datasets. Additionally, our method can complete human reconstruction even with images captured by low-cost mobile devices, reducing the requirements for data collection. Demos and code are available at https://hustvl.github.io/Snap-Snap/.
>
---
#### [new 085] High-Throughput Low-Cost Segmentation of Brightfield Microscopy Live Cell Images
- **分类: q-bio.QM; cs.AI; cs.CV; eess.IV**

- **简介: 论文针对亮场显微镜下活细胞图像的高通量分割问题，提出低成本CNN模型，结合注意力机制与自适应优化，实现93%准确率，适用于实际实验室部署。**

- **链接: [http://arxiv.org/pdf/2508.14106v1](http://arxiv.org/pdf/2508.14106v1)**

> **作者:** Surajit Das; Gourav Roy; Pavel Zun
>
> **摘要:** Live cell culture is crucial in biomedical studies for analyzing cell properties and dynamics in vitro. This study focuses on segmenting unstained live cells imaged with bright-field microscopy. While many segmentation approaches exist for microscopic images, none consistently address the challenges of bright-field live-cell imaging with high throughput, where temporal phenotype changes, low contrast, noise, and motion-induced blur from cellular movement remain major obstacles. We developed a low-cost CNN-based pipeline incorporating comparative analysis of frozen encoders within a unified U-Net architecture enhanced with attention mechanisms, instance-aware systems, adaptive loss functions, hard instance retraining, dynamic learning rates, progressive mechanisms to mitigate overfitting, and an ensemble technique. The model was validated on a public dataset featuring diverse live cell variants, showing consistent competitiveness with state-of-the-art methods, achieving 93% test accuracy and an average F1-score of 89% (std. 0.07) on low-contrast, noisy, and blurry images. Notably, the model was trained primarily on bright-field images with limited exposure to phase-contrast microscopy (<10%), yet it generalized effectively to the phase-contrast LIVECell dataset, demonstrating modality, robustness and strong performance. This highlights its potential for real-world laboratory deployment across imaging conditions. The model requires minimal compute power and is adaptable using basic deep learning setups such as Google Colab, making it practical for training on other cell variants. Our pipeline outperforms existing methods in robustness and precision for bright-field microscopy segmentation. The code and dataset are available for reproducibility
>
---
#### [new 086] A Systematic Study of Deep Learning Models and xAI Methods for Region-of-Interest Detection in MRI Scans
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文旨在通过深度学习与可解释AI方法提升MRI膝关节ROI检测的自动化水平，解决人工解读效率低和主观差异问题。研究对比了ResNet50、ViT等模型及Grad-CAM等解释方法，发现ResNet50在分类和ROI识别中表现最佳，CNN迁移学习效果优于Transformer。**

- **链接: [http://arxiv.org/pdf/2508.14151v1](http://arxiv.org/pdf/2508.14151v1)**

> **作者:** Justin Yiu; Kushank Arora; Daniel Steinberg; Rohit Ghiya
>
> **摘要:** Magnetic Resonance Imaging (MRI) is an essential diagnostic tool for assessing knee injuries. However, manual interpretation of MRI slices remains time-consuming and prone to inter-observer variability. This study presents a systematic evaluation of various deep learning architectures combined with explainable AI (xAI) techniques for automated region of interest (ROI) detection in knee MRI scans. We investigate both supervised and self-supervised approaches, including ResNet50, InceptionV3, Vision Transformers (ViT), and multiple U-Net variants augmented with multi-layer perceptron (MLP) classifiers. To enhance interpretability and clinical relevance, we integrate xAI methods such as Grad-CAM and Saliency Maps. Model performance is assessed using AUC for classification and PSNR/SSIM for reconstruction quality, along with qualitative ROI visualizations. Our results demonstrate that ResNet50 consistently excels in classification and ROI identification, outperforming transformer-based models under the constraints of the MRNet dataset. While hybrid U-Net + MLP approaches show potential for leveraging spatial features in reconstruction and interpretability, their classification performance remains lower. Grad-CAM consistently provided the most clinically meaningful explanations across architectures. Overall, CNN-based transfer learning emerges as the most effective approach for this dataset, while future work with larger-scale pretraining may better unlock the potential of transformer models.
>
---
#### [new 087] Rule-based Key-Point Extraction for MR-Guided Biomechanical Digital Twins of the Spine
- **分类: eess.IV; cs.CV**

- **简介: 论文提出基于规则的MR图像关键点提取方法，解决脊柱数字孪生中精准解剖建模问题，通过图像对齐与椎体定向估计生成解剖标志点，用于生物力学模拟，实现无辐射的个性化建模。**

- **链接: [http://arxiv.org/pdf/2508.14708v1](http://arxiv.org/pdf/2508.14708v1)**

> **作者:** Robert Graf; Tanja Lerchl; Kati Nispel; Hendrik Möller; Matan Atad; Julian McGinnis; Julius Maria Watrinet; Johannes Paetzold; Daniel Rueckert; Jan S. Kirschke
>
> **摘要:** Digital twins offer a powerful framework for subject-specific simulation and clinical decision support, yet their development often hinges on accurate, individualized anatomical modeling. In this work, we present a rule-based approach for subpixel-accurate key-point extraction from MRI, adapted from prior CT-based methods. Our approach incorporates robust image alignment and vertebra-specific orientation estimation to generate anatomically meaningful landmarks that serve as boundary conditions and force application points, like muscle and ligament insertions in biomechanical models. These models enable the simulation of spinal mechanics considering the subject's individual anatomy, and thus support the development of tailored approaches in clinical diagnostics and treatment planning. By leveraging MR imaging, our method is radiation-free and well-suited for large-scale studies and use in underrepresented populations. This work contributes to the digital twin ecosystem by bridging the gap between precise medical image analysis with biomechanical simulation, and aligns with key themes in personalized modeling for healthcare.
>
---
#### [new 088] Understanding Data Influence with Differential Approximation
- **分类: cs.LG; cs.CV**

- **简介: 论文针对数据影响分析任务，解决现有工具准确性不足问题，提出Diff-In方法通过二阶近似高效计算样本影响，适用于大规模数据，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.14648v1](http://arxiv.org/pdf/2508.14648v1)**

> **作者:** Haoru Tan; Sitong Wu; Xiuzhe Wu; Wang Wang; Bo Zhao; Zeke Xie; Gui-Song Xia; Xiaojuan Qi
>
> **摘要:** Data plays a pivotal role in the groundbreaking advancements in artificial intelligence. The quantitative analysis of data significantly contributes to model training, enhancing both the efficiency and quality of data utilization. However, existing data analysis tools often lag in accuracy. For instance, many of these tools even assume that the loss function of neural networks is convex. These limitations make it challenging to implement current methods effectively. In this paper, we introduce a new formulation to approximate a sample's influence by accumulating the differences in influence between consecutive learning steps, which we term Diff-In. Specifically, we formulate the sample-wise influence as the cumulative sum of its changes/differences across successive training iterations. By employing second-order approximations, we approximate these difference terms with high accuracy while eliminating the need for model convexity required by existing methods. Despite being a second-order method, Diff-In maintains computational complexity comparable to that of first-order methods and remains scalable. This efficiency is achieved by computing the product of the Hessian and gradient, which can be efficiently approximated using finite differences of first-order gradients. We assess the approximation accuracy of Diff-In both theoretically and empirically. Our theoretical analysis demonstrates that Diff-In achieves significantly lower approximation error compared to existing influence estimators. Extensive experiments further confirm its superior performance across multiple benchmark datasets in three data-centric tasks: data cleaning, data deletion, and coreset selection. Notably, our experiments on data pruning for large-scale vision-language pre-training show that Diff-In can scale to millions of data points and outperforms strong baselines.
>
---
#### [new 089] From Image Captioning to Visual Storytelling
- **分类: cs.CL; cs.CV**

- **简介: 论文研究视觉叙事任务，解决生成连贯且基于图像序列的故事问题，通过结合图像描述与语言模型，提出新指标并优化框架。**

- **链接: [http://arxiv.org/pdf/2508.14045v1](http://arxiv.org/pdf/2508.14045v1)**

> **作者:** Admitos Passadakis; Yingjin Song; Albert Gatt
>
> **备注:** 16 pages (including references), 5 figures and 6 tables
>
> **摘要:** Visual Storytelling is a challenging multimodal task between Vision & Language, where the purpose is to generate a story for a stream of images. Its difficulty lies on the fact that the story should be both grounded to the image sequence but also narrative and coherent. The aim of this work is to balance between these aspects, by treating Visual Storytelling as a superset of Image Captioning, an approach quite different compared to most of prior relevant studies. This means that we firstly employ a vision-to-language model for obtaining captions of the input images, and then, these captions are transformed into coherent narratives using language-to-language methods. Our multifarious evaluation shows that integrating captioning and storytelling under a unified framework, has a positive impact on the quality of the produced stories. In addition, compared to numerous previous studies, this approach accelerates training time and makes our framework readily reusable and reproducible by anyone interested. Lastly, we propose a new metric/tool, named ideality, that can be used to simulate how far some results are from an oracle model, and we apply it to emulate human-likeness in visual storytelling.
>
---
#### [new 090] OmniSense: Towards Edge-Assisted Online Analytics for 360-Degree Videos
- **分类: cs.NI; cs.CV; cs.MM; eess.IV**

- **简介: 论文提出OmniSense框架，针对360度视频在线分析中的高计算与网络资源挑战，通过边缘计算、轻量SRoI预测和动态模型调优，实现低延迟高精度分析，实验表明其准确率提升19.8%-114.6%，效率提升2.0-2.4倍。**

- **链接: [http://arxiv.org/pdf/2508.14237v1](http://arxiv.org/pdf/2508.14237v1)**

> **作者:** Miao Zhang; Yifei Zhu; Linfeng Shen; Fangxin Wang; Jiangchuan Liu
>
> **备注:** 10 pages; Accepted by INFOCOM'23
>
> **摘要:** With the reduced hardware costs of omnidirectional cameras and the proliferation of various extended reality applications, more and more $360^\circ$ videos are being captured. To fully unleash their potential, advanced video analytics is expected to extract actionable insights and situational knowledge without blind spots from the videos. In this paper, we present OmniSense, a novel edge-assisted framework for online immersive video analytics. OmniSense achieves both low latency and high accuracy, combating the significant computation and network resource challenges of analyzing $360^\circ$ videos. Motivated by our measurement insights into $360^\circ$ videos, OmniSense introduces a lightweight spherical region of interest (SRoI) prediction algorithm to prune redundant information in $360^\circ$ frames. Incorporating the video content and network dynamics, it then smartly scales vision models to analyze the predicted SRoIs with optimized resource utilization. We implement a prototype of OmniSense with commodity devices and evaluate it on diverse real-world collected $360^\circ$ videos. Extensive evaluation results show that compared to resource-agnostic baselines, it improves the accuracy by $19.8\%$ -- $114.6\%$ with similar end-to-end latencies. Meanwhile, it hits $2.0\times$ -- $2.4\times$ speedups while keeping the accuracy on par with the highest accuracy of baselines.
>
---
#### [new 091] Automated surgical planning with nnU-Net: delineation of the anatomy in hepatobiliary phase MRI
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文任务为肝部解剖结构的自动分割，旨在解决术前规划中手动分割效率低的问题。采用nnU-Net模型训练与测试，验证其分割精度，并在临床中发现额外肿瘤，提升手术规划效率。**

- **链接: [http://arxiv.org/pdf/2508.14133v1](http://arxiv.org/pdf/2508.14133v1)**

> **作者:** Karin A. Olthof; Matteo Fusagli; Bianca Güttner; Tiziano Natali; Bram Westerink; Stefanie Speidel; Theo J. M. Ruers; Koert F. D. Kuhlmann; Andrey Zhylka
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Background: The aim of this study was to develop and evaluate a deep learning-based automated segmentation method for hepatic anatomy (i.e., parenchyma, tumors, portal vein, hepatic vein and biliary tree) from the hepatobiliary phase of gadoxetic acid-enhanced MRI. This method should ease the clinical workflow of preoperative planning. Methods: Manual segmentation was performed on hepatobiliary phase MRI scans from 90 consecutive patients who underwent liver surgery between January 2020 and October 2023. A deep learning network (nnU-Net v1) was trained on 72 patients with an extra focus on thin structures and topography preservation. Performance was evaluated on an 18-patient test set by comparing automated and manual segmentations using Dice similarity coefficient (DSC). Following clinical integration, 10 segmentations (assessment dataset) were generated using the network and manually refined for clinical use to quantify required adjustments using DSC. Results: In the test set, DSCs were 0.97+/-0.01 for liver parenchyma, 0.80+/-0.04 for hepatic vein, 0.79+/-0.07 for biliary tree, 0.77+/-0.17 for tumors, and 0.74+/-0.06 for portal vein. Average tumor detection rate was 76.6+/-24.1%, with a median of one false-positive per patient. The assessment dataset showed minor adjustments were required for clinical use of the 3D models, with high DSCs for parenchyma (1.00+/-0.00), portal vein (0.98+/-0.01) and hepatic vein (0.95+/-0.07). Tumor segmentation exhibited greater variability (DSC 0.80+/-0.27). During prospective clinical use, the model detected three additional tumors initially missed by radiologists. Conclusions: The proposed nnU-Net-based segmentation method enables accurate and automated delineation of hepatic anatomy. This enables 3D planning to be applied efficiently as a standard-of-care for every patient undergoing liver surgery.
>
---
#### [new 092] Organ-Agents: Virtual Human Physiology Simulator via LLMs
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出基于LLMs的多智能体框架Organ-Agents，用于模拟人体多系统生理过程，通过微调与强化学习训练，验证其在重症监护中的准确性与实用性。**

- **链接: [http://arxiv.org/pdf/2508.14357v1](http://arxiv.org/pdf/2508.14357v1)**

> **作者:** Rihao Chang; He Jiao; Weizhi Nie; Honglin Guo; Keliang Xie; Zhenhua Wu; Lina Zhao; Yunpeng Bai; Yongtao Ma; Lanjun Wang; Yuting Su; Xi Gao; Weijie Wang; Nicu Sebe; Bruno Lepri; Bingwei Sun
>
> **摘要:** Recent advances in large language models (LLMs) have enabled new possibilities in simulating complex physiological systems. We introduce Organ-Agents, a multi-agent framework that simulates human physiology via LLM-driven agents. Each Simulator models a specific system (e.g., cardiovascular, renal, immune). Training consists of supervised fine-tuning on system-specific time-series data, followed by reinforcement-guided coordination using dynamic reference selection and error correction. We curated data from 7,134 sepsis patients and 7,895 controls, generating high-resolution trajectories across 9 systems and 125 variables. Organ-Agents achieved high simulation accuracy on 4,509 held-out patients, with per-system MSEs <0.16 and robustness across SOFA-based severity strata. External validation on 22,689 ICU patients from two hospitals showed moderate degradation under distribution shifts with stable simulation. Organ-Agents faithfully reproduces critical multi-system events (e.g., hypotension, hyperlactatemia, hypoxemia) with coherent timing and phase progression. Evaluation by 15 critical care physicians confirmed realism and physiological plausibility (mean Likert ratings 3.9 and 3.7). Organ-Agents also enables counterfactual simulations under alternative sepsis treatment strategies, generating trajectories and APACHE II scores aligned with matched real-world patients. In downstream early warning tasks, classifiers trained on synthetic data showed minimal AUROC drops (<0.04), indicating preserved decision-relevant patterns. These results position Organ-Agents as a credible, interpretable, and generalizable digital twin for precision diagnosis, treatment simulation, and hypothesis testing in critical care.
>
---
#### [new 093] STAS: Spatio-Temporal Adaptive Computation Time for Spiking Transformers
- **分类: cs.LG; cs.AI; cs.CV; cs.NE**

- **简介: 该论文针对SNN-based Vision Transformer的高延迟与计算开销问题，提出STAS框架，通过时空自适应计算时间机制，结合动态计算策略与静态架构设计，实现能量效率提升与精度优化。**

- **链接: [http://arxiv.org/pdf/2508.14138v1](http://arxiv.org/pdf/2508.14138v1)**

> **作者:** Donghwa Kang; Doohyun Kim; Sang-Ki Ko; Jinkyu Lee; Brent ByungHoon Kang; Hyeongboo Baek
>
> **备注:** 8 pages
>
> **摘要:** Spiking neural networks (SNNs) offer energy efficiency over artificial neural networks (ANNs) but suffer from high latency and computational overhead due to their multi-timestep operational nature. While various dynamic computation methods have been developed to mitigate this by targeting spatial, temporal, or architecture-specific redundancies, they remain fragmented. While the principles of adaptive computation time (ACT) offer a robust foundation for a unified approach, its application to SNN-based vision Transformers (ViTs) is hindered by two core issues: the violation of its temporal similarity prerequisite and a static architecture fundamentally unsuited for its principles. To address these challenges, we propose STAS (Spatio-Temporal Adaptive computation time for Spiking transformers), a framework that co-designs the static architecture and dynamic computation policy. STAS introduces an integrated spike patch splitting (I-SPS) module to establish temporal stability by creating a unified input representation, thereby solving the architectural problem of temporal dissimilarity. This stability, in turn, allows our adaptive spiking self-attention (A-SSA) module to perform two-dimensional token pruning across both spatial and temporal axes. Implemented on spiking Transformer architectures and validated on CIFAR-10, CIFAR-100, and ImageNet, STAS reduces energy consumption by up to 45.9%, 43.8%, and 30.1%, respectively, while simultaneously improving accuracy over SOTA models.
>
---
#### [new 094] Fine-grained Image Quality Assessment for Perceptual Image Restoration
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 论文针对感知图像修复中的细粒度质量评估问题，构建了FGRestore数据集并提出FGResQ模型，提升评估精度。**

- **链接: [http://arxiv.org/pdf/2508.14475v1](http://arxiv.org/pdf/2508.14475v1)**

> **作者:** Xiangfei Sheng; Xiaofeng Pan; Zhichao Yang; Pengfei Chen; Leida Li
>
> **备注:** 9 pages,6 figures
>
> **摘要:** Recent years have witnessed remarkable achievements in perceptual image restoration (IR), creating an urgent demand for accurate image quality assessment (IQA), which is essential for both performance comparison and algorithm optimization. Unfortunately, the existing IQA metrics exhibit inherent weakness for IR task, particularly when distinguishing fine-grained quality differences among restored images. To address this dilemma, we contribute the first-of-its-kind fine-grained image quality assessment dataset for image restoration, termed FGRestore, comprising 18,408 restored images across six common IR tasks. Beyond conventional scalar quality scores, FGRestore was also annotated with 30,886 fine-grained pairwise preferences. Based on FGRestore, a comprehensive benchmark was conducted on the existing IQA metrics, which reveal significant inconsistencies between score-based IQA evaluations and the fine-grained restoration quality. Motivated by these findings, we further propose FGResQ, a new IQA model specifically designed for image restoration, which features both coarse-grained score regression and fine-grained quality ranking. Extensive experiments and comparisons demonstrate that FGResQ significantly outperforms state-of-the-art IQA metrics. Codes and model weights have been released in https://pxf0429.github.io/FGResQ/
>
---
#### [new 095] Hallucinations in medical devices
- **分类: eess.IV; cs.CV**

- **简介: 该论文旨在定义医疗设备中因深度学习导致的错误输出（幻觉），提出统一评估标准，并通过影像等案例探讨其影响及缓解方法。**

- **链接: [http://arxiv.org/pdf/2508.14118v1](http://arxiv.org/pdf/2508.14118v1)**

> **作者:** Jason Granstedt; Prabhat Kc; Rucha Deshpande; Victor Garcia; Aldo Badano
>
> **备注:** 19 pages, 2 figures
>
> **摘要:** Computer methods in medical devices are frequently imperfect and are known to produce errors in clinical or diagnostic tasks. However, when deep learning and data-based approaches yield output that exhibit errors, the devices are frequently said to hallucinate. Drawing from theoretical developments and empirical studies in multiple medical device areas, we introduce a practical and universal definition that denotes hallucinations as a type of error that is plausible and can be either impactful or benign to the task at hand. The definition aims at facilitating the evaluation of medical devices that suffer from hallucinations across product areas. Using examples from imaging and non-imaging applications, we explore how the proposed definition relates to evaluation methodologies and discuss existing approaches for minimizing the prevalence of hallucinations.
>
---
#### [new 096] Squeezed Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对扩散模型中各向同性噪声的问题，提出Squeezed Diffusion Models（SDM），通过非对称缩放噪声（尤其在主成分方向）提升图像生成质量，实验显示在CIFAR和CelebA数据集上显著提高FID分数和召回率。**

- **链接: [http://arxiv.org/pdf/2508.14871v1](http://arxiv.org/pdf/2508.14871v1)**

> **作者:** Jyotirmai Singh; Samar Khanna; James Burgess
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Diffusion models typically inject isotropic Gaussian noise, disregarding structure in the data. Motivated by the way quantum squeezed states redistribute uncertainty according to the Heisenberg uncertainty principle, we introduce Squeezed Diffusion Models (SDM), which scale noise anisotropically along the principal component of the training distribution. As squeezing enhances the signal-to-noise ratio in physics, we hypothesize that scaling noise in a data-dependent manner can better assist diffusion models in learning important data features. We study two configurations: (i) a Heisenberg diffusion model that compensates the scaling on the principal axis with inverse scaling on orthogonal directions and (ii) a standard SDM variant that scales only the principal axis. Counterintuitively, on CIFAR-10/100 and CelebA-64, mild antisqueezing - i.e. increasing variance on the principal axis - consistently improves FID by up to 15% and shifts the precision-recall frontier toward higher recall. Our results demonstrate that simple, data-aware noise shaping can deliver robust generative gains without architectural changes.
>
---
#### [new 097] A Real-world Display Inverse Rendering Dataset
- **分类: cs.GR; cs.CV**

- **简介: 本文提出首个真实世界显示逆向渲染数据集，用于从图像重建几何与反射率。针对显示相机系统缺乏公开数据的问题，构建LCD+偏振相机系统，采集多物体多材质样本并提供高精度地面真值，评估现有方法并提出更优基线。**

- **链接: [http://arxiv.org/pdf/2508.14411v1](http://arxiv.org/pdf/2508.14411v1)**

> **作者:** Seokjun Choi; Hoon-Gyu Chung; Yujin Jeon; Giljoo Nam; Seung-Hwan Baek
>
> **摘要:** Inverse rendering aims to reconstruct geometry and reflectance from captured images. Display-camera imaging systems offer unique advantages for this task: each pixel can easily function as a programmable point light source, and the polarized light emitted by LCD displays facilitates diffuse-specular separation. Despite these benefits, there is currently no public real-world dataset captured using display-camera systems, unlike other setups such as light stages. This absence hinders the development and evaluation of display-based inverse rendering methods. In this paper, we introduce the first real-world dataset for display-based inverse rendering. To achieve this, we construct and calibrate an imaging system comprising an LCD display and stereo polarization cameras. We then capture a diverse set of objects with diverse geometry and reflectance under one-light-at-a-time (OLAT) display patterns. We also provide high-quality ground-truth geometry. Our dataset enables the synthesis of captured images under arbitrary display patterns and different noise levels. Using this dataset, we evaluate the performance of existing photometric stereo and inverse rendering methods, and provide a simple, yet effective baseline for display inverse rendering, outperforming state-of-the-art inverse rendering methods. Code and dataset are available on our project page at https://michaelcsj.github.io/DIR/
>
---
## 更新

#### [replaced 001] Is Contrastive Distillation Enough for Learning Comprehensive 3D Representations?
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.08973v2](http://arxiv.org/pdf/2412.08973v2)**

> **作者:** Yifan Zhang; Junhui Hou
>
> **备注:** 21 pages, 10 figures
>
> **摘要:** Cross-modal contrastive distillation has recently been explored for learning effective 3D representations. However, existing methods focus primarily on modality-shared features, neglecting the modality-specific features during the pre-training process, which leads to suboptimal representations. In this paper, we theoretically analyze the limitations of current contrastive methods for 3D representation learning and propose a new framework, namely CMCR, to address these shortcomings. Our approach improves upon traditional methods by better integrating both modality-shared and modality-specific features. Specifically, we introduce masked image modeling and occupancy estimation tasks to guide the network in learning more comprehensive modality-specific features. Furthermore, we propose a novel multi-modal unified codebook that learns an embedding space shared across different modalities. Besides, we introduce geometry-enhanced masked image modeling to further boost 3D representation learning. Extensive experiments demonstrate that our method mitigates the challenges faced by traditional approaches and consistently outperforms existing image-to-LiDAR contrastive distillation methods in downstream tasks. Code will be available at https://github.com/Eaphan/CMCR.
>
---
#### [replaced 002] MAViS: A Multi-Agent Framework for Long-Sequence Video Storytelling
- **分类: cs.CV; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.08487v3](http://arxiv.org/pdf/2508.08487v3)**

> **作者:** Qian Wang; Ziqi Huang; Ruoxi Jia; Paul Debevec; Ning Yu
>
> **备注:** Video Generation Agent
>
> **摘要:** Despite recent advances, long-sequence video generation frameworks still suffer from significant limitations: poor assistive capability, suboptimal visual quality, and limited expressiveness. To mitigate these limitations, we propose MAViS, an end-to-end multi-agent collaborative framework for long-sequence video storytelling. MAViS orchestrates specialized agents across multiple stages, including script writing, shot designing, character modeling, keyframe generation, video animation, and audio generation. In each stage, agents operate under the 3E Principle -- Explore, Examine, and Enhance -- to ensure the completeness of intermediate outputs. Considering the capability limitations of current generative models, we propose the Script Writing Guidelines to optimize compatibility between scripts and generative tools. Experimental results demonstrate that MAViS achieves state-of-the-art performance in assistive capability, visual quality, and video expressiveness. Its modular framework further enables scalability with diverse generative models and tools. With just a brief user prompt, MAViS is capable of producing high-quality, expressive long-sequence video storytelling, enriching inspirations and creativity for users. To the best of our knowledge, MAViS is the only framework that provides multimodal design output -- videos with narratives and background music.
>
---
#### [replaced 003] Interpreting the linear structure of vision-language model embedding spaces
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.11695v4](http://arxiv.org/pdf/2504.11695v4)**

> **作者:** Isabel Papadimitriou; Huangyuan Su; Thomas Fel; Sham Kakade; Stephanie Gil
>
> **备注:** COLM 2025
>
> **摘要:** Vision-language models encode images and text in a joint space, minimizing the distance between corresponding image and text pairs. How are language and images organized in this joint space, and how do the models encode meaning and modality? To investigate this, we train and release sparse autoencoders (SAEs) on the embedding spaces of four vision-language models (CLIP, SigLIP, SigLIP2, and AIMv2). SAEs approximate model embeddings as sparse linear combinations of learned directions, or "concepts". We find that, compared to other methods of linear feature learning, SAEs are better at reconstructing the real embeddings, while also able to retain the most sparsity. Retraining SAEs with different seeds or different data diet leads to two findings: the rare, specific concepts captured by the SAEs are liable to change drastically, but we also show that commonly-activating concepts are remarkably stable across runs. Interestingly, while most concepts activate primarily for one modality, we find they are not merely encoding modality per se. Many are almost orthogonal to the subspace that defines modality, and the concept directions do not function as good modality classifiers, suggesting that they encode cross-modal semantics. To quantify this bridging behavior, we introduce the Bridge Score, a metric that identifies concept pairs which are both co-activated across aligned image-text inputs and geometrically aligned in the shared space. This reveals that even single-modality concepts can collaborate to support cross-modal integration. We release interactive demos of the SAEs for all models, allowing researchers to explore the organization of the concept spaces. Overall, our findings uncover a sparse linear structure within VLM embedding spaces that is shaped by modality, yet stitched together through latent bridges, offering new insight into how multimodal meaning is constructed.
>
---
#### [replaced 004] Natural Language Generation from Visual Events: State-of-the-Art and Key Open Questions
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13034v3](http://arxiv.org/pdf/2502.13034v3)**

> **作者:** Aditya K Surikuchi; Raquel Fernández; Sandro Pezzelle
>
> **摘要:** In recent years, a substantial body of work in visually grounded natural language processing has focused on real-life multimodal scenarios such as describing content depicted in images or videos. However, comparatively less attention has been devoted to study the nature and degree of interaction between the different modalities in these scenarios. In this paper, we argue that any task dealing with natural language generation from sequences of images or frames is an instance of the broader, more general problem of modeling the intricate relationships between visual events unfolding over time and the features of the language used to interpret, describe, or narrate them. Therefore, solving these tasks requires models to be capable of identifying and managing such intricacies. We consider five seemingly different tasks, which we argue are compelling instances of this broader multimodal problem. Subsequently, we survey the modeling and evaluation approaches adopted for these tasks in recent years and examine the common set of challenges these tasks pose. Building on this perspective, we identify key open questions and propose several research directions for future investigation.
>
---
#### [replaced 005] Fluorescence molecular optomic signatures improve identification of tumors in head and neck specimens
- **分类: cs.LG; cs.CV; 68T10 (Primary) 68U10 (Secondary); I.5.2; I.4.6**

- **链接: [http://arxiv.org/pdf/2208.13314v2](http://arxiv.org/pdf/2208.13314v2)**

> **作者:** Yao Chen; Samuel S. Streeter; Brady Hunt; Hira S. Sardar; Jason R. Gunn; Laura J. Tafe; Joseph A. Paydarfar; Brian W. Pogue; Keith D. Paulsen; Kimberley S. Samkoe
>
> **备注:** 21 pages, 8 figures, 1 table, submitted as a manuscript at Frontiers in Medical Technology
>
> **摘要:** In this study, a radiomics approach was extended to optical fluorescence molecular imaging data for tissue classification, termed 'optomics'. Fluorescence molecular imaging is emerging for precise surgical guidance during head and neck squamous cell carcinoma (HNSCC) resection. However, the tumor-to-normal tissue contrast is confounded by intrinsic physiological limitations of heterogeneous expression of the target molecule, epidermal growth factor receptor (EGFR). Optomics seek to improve tumor identification by probing textural pattern differences in EGFR expression conveyed by fluorescence. A total of 1,472 standardized optomic features were extracted from fluorescence image samples. A supervised machine learning pipeline involving a support vector machine classifier was trained with 25 top-ranked features selected by minimum redundancy maximum relevance criterion. Model predictive performance was compared to fluorescence intensity thresholding method by classifying testing set image patches of resected tissue with histologically confirmed malignancy status. The optomics approach provided consistent improvement in prediction accuracy on all test set samples, irrespective of dose, compared to fluorescence intensity thresholding method (mean accuracies of 89% vs. 81%; P = 0.0072). The improved performance demonstrates that extending the radiomics approach to fluorescence molecular imaging data offers a promising image analysis technique for cancer detection in fluorescence-guided surgery.
>
---
#### [replaced 006] Enhanced Anomaly Detection for Capsule Endoscopy Using Ensemble Learning Strategies
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06039v2](http://arxiv.org/pdf/2504.06039v2)**

> **作者:** Julia Werner; Christoph Gerum; Jorg Nick; Maxime Le Floch; Franz Brinkmann; Jochen Hampe; Oliver Bringmann
>
> **备注:** Accepted at the 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBS EMBC)
>
> **摘要:** Capsule endoscopy is a method to capture images of the gastrointestinal tract and screen for diseases which might remain hidden if investigated with standard endoscopes. Due to the limited size of a video capsule, embedding AI models directly into the capsule demands careful consideration of the model size and thus complicates anomaly detection in this field. Furthermore, the scarcity of available data in this domain poses an ongoing challenge to achieving effective anomaly detection. Thus, this work introduces an ensemble strategy to address this challenge in anomaly detection tasks in video capsule endoscopies, requiring only a small number of individual neural networks during both the training and inference phases. Ensemble learning combines the predictions of multiple independently trained neural networks. This has shown to be highly effective in enhancing both the accuracy and robustness of machine learning models. However, this comes at the cost of higher memory usage and increased computational effort, which quickly becomes prohibitive in many real-world applications. Instead of applying the same training algorithm to each individual network, we propose using various loss functions, drawn from the anomaly detection field, to train each network. The methods are validated on the two largest publicly available datasets for video capsule endoscopy images, the Galar and the Kvasir-Capsule dataset. We achieve an AUC score of 76.86% on the Kvasir-Capsule and an AUC score of 76.98% on the Galar dataset. Our approach outperforms current baselines with significantly fewer parameters across all models, which is a crucial step towards incorporating artificial intelligence into capsule endoscopies.
>
---
#### [replaced 007] MetaWild: A Multimodal Dataset for Animal Re-Identification with Environmental Metadata
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.13368v2](http://arxiv.org/pdf/2501.13368v2)**

> **作者:** Yuzhuo Li; Di Zhao; Tingrui Qiao; Yihao Wu; Bo Pang; Yun Sing Koh
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Identifying individual animals within large wildlife populations is essential for effective wildlife monitoring and conservation efforts. Recent advancements in computer vision have shown promise in animal re-identification (Animal ReID) by leveraging data from camera traps. However, existing Animal ReID datasets rely exclusively on visual data, overlooking environmental metadata that ecologists have identified as highly correlated with animal behavior and identity, such as temperature and circadian rhythms. Moreover, the emergence of multimodal models capable of jointly processing visual and textual data presents new opportunities for Animal ReID, but existing datasets fail to leverage these models' text-processing capabilities, limiting their full potential. Additionally, to facilitate the use of metadata in existing ReID methods, we propose the Meta-Feature Adapter (MFA), a lightweight module that can be incorporated into existing vision-language model (VLM)-based Animal ReID methods, allowing ReID models to leverage both environmental metadata and visual information to improve ReID performance. Experiments on MetaWild show that combining baseline ReID models with MFA to incorporate metadata consistently improves performance compared to using visual information alone, validating the effectiveness of incorporating metadata in re-identification. We hope that our proposed dataset can inspire further exploration of multimodal approaches for Animal ReID.
>
---
#### [replaced 008] VSF: Simple, Efficient, and Effective Negative Guidance in Few-Step Image Generation Models By Value Sign Flip
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10931v3](http://arxiv.org/pdf/2508.10931v3)**

> **作者:** Wenqi Guo; Shan Du
>
> **摘要:** We introduce Value Sign Flip (VSF), a simple and efficient method for incorporating negative prompt guidance in few-step diffusion and flow-matching image generation models. Unlike existing approaches such as classifier-free guidance (CFG), NASA, and NAG, VSF dynamically suppresses undesired content by flipping the sign of attention values from negative prompts. Our method requires only small computational overhead and integrates effectively with MMDiT-style architectures such as Stable Diffusion 3.5 Turbo, as well as cross-attention-based models like Wan. We validate VSF on challenging datasets with complex prompt pairs and demonstrate superior performance in both static image and video generation tasks. Experimental results show that VSF significantly improves negative prompt adherence compared to prior methods in few-step models, and even CFG in non-few-step models, while maintaining competitive image quality. Code and ComfyUI node are available in https://github.com/weathon/VSF/tree/main.
>
---
#### [replaced 009] Explicit Context Reasoning with Supervision for Visual Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16191v2](http://arxiv.org/pdf/2507.16191v2)**

> **作者:** Fansheng Zeng; Bineng Zhong; Haiying Xia; Yufei Tan; Xiantao Hu; Liangtao Shi; Shuxiang Song
>
> **摘要:** Contextual reasoning with constraints is crucial for enhancing temporal consistency in cross-frame modeling for visual tracking. However, mainstream tracking algorithms typically associate context by merely stacking historical information without explicitly supervising the association process, making it difficult to effectively model the target's evolving dynamics. To alleviate this problem, we propose RSTrack, which explicitly models and supervises context reasoning via three core mechanisms. \textit{1) Context Reasoning Mechanism}: Constructs a target state reasoning pipeline, converting unconstrained contextual associations into a temporal reasoning process that predicts the current representation based on historical target states, thereby enhancing temporal consistency. \textit{2) Forward Supervision Strategy}: Utilizes true target features as anchors to constrain the reasoning pipeline, guiding the predicted output toward the true target distribution and suppressing drift in the context reasoning process. \textit{3) Efficient State Modeling}: Employs a compression-reconstruction mechanism to extract the core features of the target, removing redundant information across frames and preventing ineffective contextual associations. These three mechanisms collaborate to effectively alleviate the issue of contextual association divergence in traditional temporal modeling. Experimental results show that RSTrack achieves state-of-the-art performance on multiple benchmark datasets while maintaining real-time running speeds. Our code is available at https://github.com/GXNU-ZhongLab/RSTrack.
>
---
#### [replaced 010] MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.08452v3](http://arxiv.org/pdf/2404.08452v3)**

> **作者:** Chenqi Kong; Anwei Luo; Peijun Bao; Yi Yu; Haoliang Li; Zengwei Zheng; Shiqi Wang; Alex C. Kot
>
> **摘要:** Deepfakes have recently raised significant trust issues and security concerns among the public. Compared to CNN face forgery detectors, ViT-based methods take advantage of the expressivity of transformers, achieving superior detection performance. However, these approaches still exhibit the following limitations: (1) Fully fine-tuning ViT-based models from ImageNet weights demands substantial computational and storage resources; (2) ViT-based methods struggle to capture local forgery clues, leading to model bias; (3) These methods limit their scope on only one or few face forgery features, resulting in limited generalizability. To tackle these challenges, this work introduces Mixture-of-Experts modules for Face Forgery Detection (MoE-FFD), a generalized yet parameter-efficient ViT-based approach. MoE-FFD only updates lightweight Low-Rank Adaptation (LoRA) and Adapter layers while keeping the ViT backbone frozen, thereby achieving parameter-efficient training. Moreover, MoE-FFD leverages the expressivity of transformers and local priors of CNNs to simultaneously extract global and local forgery clues. Additionally, novel MoE modules are designed to scale the model's capacity and smartly select optimal forgery experts, further enhancing forgery detection performance. Our proposed learning scheme can be seamlessly adapted to various transformer backbones in a plug-and-play manner. Extensive experimental results demonstrate that the proposed method achieves state-of-the-art face forgery detection performance with significantly reduced parameter overhead. The code is released at: https://github.com/LoveSiameseCat/MoE-FFD.
>
---
#### [replaced 011] ETA: Energy-based Test-time Adaptation for Depth Completion
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.05989v2](http://arxiv.org/pdf/2508.05989v2)**

> **作者:** Younjoon Chung; Hyoungseob Park; Patrick Rim; Xiaoran Zhang; Jihe He; Ziyao Zeng; Safa Cicek; Byung-Woo Hong; James S. Duncan; Alex Wong
>
> **摘要:** We propose a method for test-time adaptation of pretrained depth completion models. Depth completion models, trained on some ``source'' data, often predict erroneous outputs when transferred to ``target'' data captured in novel environmental conditions due to a covariate shift. The crux of our method lies in quantifying the likelihood of depth predictions belonging to the source data distribution. The challenge is in the lack of access to out-of-distribution (target) data prior to deployment. Hence, rather than making assumptions regarding the target distribution, we utilize adversarial perturbations as a mechanism to explore the data space. This enables us to train an energy model that scores local regions of depth predictions as in- or out-of-distribution. We update the parameters of pretrained depth completion models at test time to minimize energy, effectively aligning test-time predictions to those of the source distribution. We call our method ``Energy-based Test-time Adaptation'', or ETA for short. We evaluate our method across three indoor and three outdoor datasets, where ETA improve over the previous state-of-the-art method by an average of 6.94% for outdoors and 10.23% for indoors. Project Page: https://fuzzythecat.github.io/eta.
>
---
#### [replaced 012] MMHMER:Multi-viewer and Multi-task for Handwritten Mathematical Expression Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.05557v3](http://arxiv.org/pdf/2502.05557v3)**

> **作者:** Kehua Chen; Haoyang Shen; Lifan Zhong; Mingyi Chen
>
> **备注:** 7 pages;2 figures
>
> **摘要:** Handwritten Mathematical Expression Recognition (HMER) methods have made remarkable progress, with most existing HMER approaches based on either a hybrid CNN/RNN-based with GRU architecture or Transformer architectures. Each of these has its strengths and weaknesses. Leveraging different model structures as viewers and effectively integrating their diverse capabilities presents an intriguing avenue for exploration. This involves addressing two key challenges: 1) How to fuse these two methods effectively, and 2) How to achieve higher performance under an appropriate level of complexity. This paper proposes an efficient CNN-Transformer multi-viewer, multi-task approach to enhance the model's recognition performance. Our MMHMER model achieves 63.96%, 62.51%, and 65.46% ExpRate on CROHME14, CROHME16, and CROHME19, outperforming Posformer with an absolute gain of 1.28%, 1.48%, and 0.58%. The main contribution of our approach is that we propose a new multi-view, multi-task framework that can effectively integrate the strengths of CNN and Transformer. By leveraging the feature extraction capabilities of CNN and the sequence modeling capabilities of Transformer, our model can better handle the complexity of handwritten mathematical expressions.
>
---
#### [replaced 013] UnZipLoRA: Separating Content and Style from a Single Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.04465v2](http://arxiv.org/pdf/2412.04465v2)**

> **作者:** Chang Liu; Viraj Shah; Aiyu Cui; Svetlana Lazebnik
>
> **备注:** Project page: https://unziplora.github.io
>
> **摘要:** This paper introduces UnZipLoRA, a method for decomposing an image into its constituent subject and style, represented as two distinct LoRAs (Low-Rank Adaptations). Unlike existing personalization techniques that focus on either subject or style in isolation, or require separate training sets for each, UnZipLoRA disentangles these elements from a single image by training both the LoRAs simultaneously. UnZipLoRA ensures that the resulting LoRAs are compatible, i.e., they can be seamlessly combined using direct addition. UnZipLoRA enables independent manipulation and recontextualization of subject and style, including generating variations of each, applying the extracted style to new subjects, and recombining them to reconstruct the original image or create novel variations. To address the challenge of subject and style entanglement, UnZipLoRA employs a novel prompt separation technique, as well as column and block separation strategies to accurately preserve the characteristics of subject and style, and ensure compatibility between the learned LoRAs. Evaluation with human studies and quantitative metrics demonstrates UnZipLoRA's effectiveness compared to other state-of-the-art methods, including DreamBooth-LoRA, Inspiration Tree, and B-LoRA.
>
---
#### [replaced 014] Superpixel-informed Continuous Low-Rank Tensor Representation for Multi-Dimensional Data Recovery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.12261v2](http://arxiv.org/pdf/2508.12261v2)**

> **作者:** Zhizhou Wang; Jianli Wang; Ruijing Zheng; Zhenyu Wu
>
> **备注:** Under review in AAAI2026
>
> **摘要:** Low-rank tensor representation (LRTR) has emerged as a powerful tool for multi-dimensional data processing. However, classical LRTR-based methods face two critical limitations: (1) they typically assume that the holistic data is low-rank, this assumption is often violated in real-world scenarios with significant spatial variations; and (2) they are constrained to discrete meshgrid data, limiting their flexibility and applicability. To overcome these limitations, we propose a Superpixel-informed Continuous low-rank Tensor Representation (SCTR) framework, which enables continuous and flexible modeling of multi-dimensional data beyond traditional grid-based constraints. Our approach introduces two main innovations: First, motivated by the observation that semantically coherent regions exhibit stronger low-rank characteristics than holistic data, we employ superpixels as the basic modeling units. This design not only encodes rich semantic information, but also enhances adaptability to diverse forms of data streams. Second, we propose a novel asymmetric low-rank tensor factorization (ALTF) where superpixel-specific factor matrices are parameterized by a shared neural network with specialized heads. By strategically separating global pattern learning from local adaptation, this framework efficiently captures both cross-superpixel commonalities and within-superpixel variations. This yields a representation that is both highly expressive and compact, balancing model efficiency with adaptability. Extensive experiments on several benchmark datasets demonstrate that SCTR achieves 3-5 dB PSNR improvements over existing LRTR-based methods across multispectral images, videos, and color images.
>
---
#### [replaced 015] UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00288v2](http://arxiv.org/pdf/2508.00288v2)**

> **作者:** Jianqiang Xiao; Yuexuan Sun; Yixin Shao; Boxi Gan; Rongqiang Liu; Yanjing Wu; Weili Gua; Xiang Deng
>
> **备注:** Accepted to ACM MM Dataset Track 2025
>
> **摘要:** Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments.
>
---
#### [replaced 016] Unsupervised Urban Tree Biodiversity Mapping from Street-Level Imagery Using Spatially-Aware Visual Clustering
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.13814v2](http://arxiv.org/pdf/2508.13814v2)**

> **作者:** Diaa Addeen Abuhani; Marco Seccaroni; Martina Mazzarello; Imran Zualkernan; Fabio Duarte; Carlo Ratti
>
> **备注:** 27 pages, 7 figures, Nature Format
>
> **摘要:** Urban tree biodiversity is critical for climate resilience, ecological stability, and livability in cities, yet most municipalities lack detailed knowledge of their canopies. Field-based inventories provide reliable estimates of Shannon and Simpson diversity but are costly and time-consuming, while supervised AI methods require labeled data that often fail to generalize across regions. We introduce an unsupervised clustering framework that integrates visual embeddings from street-level imagery with spatial planting patterns to estimate biodiversity without labels. Applied to eight North American cities, the method recovers genus-level diversity patterns with high fidelity, achieving low Wasserstein distances to ground truth for Shannon and Simpson indices and preserving spatial autocorrelation. This scalable, fine-grained approach enables biodiversity mapping in cities lacking detailed inventories and offers a pathway for continuous, low-cost monitoring to support equitable access to greenery and adaptive management of urban ecosystems.
>
---
#### [replaced 017] Seeing More with Less: Video Capsule Endoscopy with Multi-Task Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.23479v2](http://arxiv.org/pdf/2507.23479v2)**

> **作者:** Julia Werner; Oliver Bause; Julius Oexle; Maxime Le Floch; Franz Brinkmann; Jochen Hampe; Oliver Bringmann
>
> **备注:** Accepted at Applications of Medical AI (AMAI workshop) at MICCAI 2025 (submitted version)
>
> **摘要:** Video capsule endoscopy has become increasingly important for investigating the small intestine within the gastrointestinal tract. However, a persistent challenge remains the short battery lifetime of such compact sensor edge devices. Integrating artificial intelligence can help overcome this limitation by enabling intelligent real-time decision-making, thereby reducing the energy consumption and prolonging the battery life. However, this remains challenging due to data sparsity and the limited resources of the device restricting the overall model size. In this work, we introduce a multi-task neural network that combines the functionalities of precise self-localization within the gastrointestinal tract with the ability to detect anomalies in the small intestine within a single model. Throughout the development process, we consistently restricted the total number of parameters to ensure the feasibility to deploy such model in a small capsule. We report the first multi-task results using the recently published Galar dataset, integrating established multi-task methods and Viterbi decoding for subsequent time-series analysis. This outperforms current single-task models and represents a significant advance in AI-based approaches in this field. Our model achieves an accuracy of 93.63% on the localization task and an accuracy of 87.48% on the anomaly detection task. The approach requires only 1 million parameters while surpassing the current baselines.
>
---
#### [replaced 018] Six-CD: Benchmarking Concept Removals for Benign Text-to-image Diffusion Models
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2406.14855v3](http://arxiv.org/pdf/2406.14855v3)**

> **作者:** Jie Ren; Kangrui Chen; Yingqian Cui; Shenglai Zeng; Hui Liu; Yue Xing; Jiliang Tang; Lingjuan Lyu
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Text-to-image (T2I) diffusion models have shown exceptional capabilities in generating images that closely correspond to textual prompts. However, the advancement of T2I diffusion models presents significant risks, as the models could be exploited for malicious purposes, such as generating images with violence or nudity, or creating unauthorized portraits of public figures in inappropriate contexts. To mitigate these risks, concept removal methods have been proposed. These methods aim to modify diffusion models to prevent the generation of malicious and unwanted concepts. Despite these efforts, existing research faces several challenges: (1) a lack of consistent comparisons on a comprehensive dataset, (2) ineffective prompts in harmful and nudity concepts, (3) overlooked evaluation of the ability to generate the benign part within prompts containing malicious concepts. To address these gaps, we propose to benchmark the concept removal methods by introducing a new dataset, Six-CD, along with a novel evaluation metric. In this benchmark, we conduct a thorough evaluation of concept removals, with the experimental observations and discussions offering valuable insights in the field.
>
---
#### [replaced 019] RNDiff: Rainfall nowcasting with Condition Diffusion Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.13737v2](http://arxiv.org/pdf/2402.13737v2)**

> **作者:** Xudong Ling; Chaorong Li; Fengqing Qin; Peng Yang; Yuanyuan Huang
>
> **摘要:** Diffusion models are widely used in image generation because they can generate high-quality and realistic samples. This is in contrast to generative adversarial networks (GANs) and variational autoencoders (VAEs), which have some limitations in terms of image quality.We introduce the diffusion model to the precipitation forecasting task and propose a short-term precipitation nowcasting with condition diffusion model based on historical observational data, which is referred to as SRNDiff. By incorporating an additional conditional decoder module in the denoising process, SRNDiff achieves end-to-end conditional rainfall prediction. SRNDiff is composed of two networks: a denoising network and a conditional Encoder network. The conditional network is composed of multiple independent UNet networks. These networks extract conditional feature maps at different resolutions, providing accurate conditional information that guides the diffusion model for conditional generation.SRNDiff surpasses GANs in terms of prediction accuracy, although it requires more computational resources.The SRNDiff model exhibits higher stability and efficiency during training than GANs-based approaches, and generates high-quality precipitation distribution samples that better reflect future actual precipitation conditions. This fully validates the advantages and potential of diffusion models in precipitation forecasting, providing new insights for enhancing rainfall prediction.
>
---
#### [replaced 020] Toward Errorless Training ImageNet-1k
- **分类: cs.CV; cs.LG; 68T07**

- **链接: [http://arxiv.org/pdf/2508.04941v3](http://arxiv.org/pdf/2508.04941v3)**

> **作者:** Bo Deng; Levi Heath
>
> **备注:** 14 pages, 2 figures, 5 tables
>
> **摘要:** In this paper, we describe a feedforward artificial neural network trained on the ImageNet 2012 contest dataset [7] with the new method of [5] to an accuracy rate of 98.3% with a 99.69 Top-1 rate, and an average of 285.9 labels that are perfectly classified over the 10 batch partitions of the dataset. The best performing model uses 322,430,160 parameters, with 4 decimal places precision. We conjecture that the reason our model does not achieve a 100% accuracy rate is due to a double-labeling problem, by which there are duplicate images in the dataset with different labels.
>
---
#### [replaced 021] Consistent and Optimal Solution to Camera Motion Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.01174v2](http://arxiv.org/pdf/2403.01174v2)**

> **作者:** Guangyang Zeng; Qingcheng Zeng; Xinghan Li; Biqiang Mu; Jiming Chen; Ling Shi; Junfeng Wu
>
> **备注:** 18 pages, 13 figures
>
> **摘要:** Given 2D point correspondences between an image pair, inferring the camera motion is a fundamental issue in the computer vision community. The existing works generally set out from the epipolar constraint and estimate the essential matrix, which is not optimal in the maximum likelihood (ML) sense. In this paper, we dive into the original measurement model with respect to the rotation matrix and normalized translation vector and formulate the ML problem. We then propose a two-step algorithm to solve it: In the first step, we estimate the variance of measurement noises and devise a consistent estimator based on bias elimination; In the second step, we execute a one-step Gauss-Newton iteration on manifold to refine the consistent estimate. We prove that the proposed estimate owns the same asymptotic statistical properties as the ML estimate: The first is consistency, i.e., the estimate converges to the ground truth as the point number increases; The second is asymptotic efficiency, i.e., the mean squared error of the estimate converges to the theoretical lower bound -- Cramer-Rao bound. In addition, we show that our algorithm has linear time complexity. These appealing characteristics endow our estimator with a great advantage in the case of dense point correspondences. Experiments on both synthetic data and real images demonstrate that when the point number reaches the order of hundreds, our estimator outperforms the state-of-the-art ones in terms of estimation accuracy and CPU time.
>
---
#### [replaced 022] Dark Miner: Defend against undesirable generation for text-to-image diffusion models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.17682v3](http://arxiv.org/pdf/2409.17682v3)**

> **作者:** Zheling Meng; Bo Peng; Xiaochuan Jin; Yue Jiang; Wei Wang; Jing Dong; Tieniu Tan
>
> **摘要:** Text-to-image diffusion models have been demonstrated with undesired generation due to unfiltered large-scale training data, such as sexual images and copyrights, necessitating the erasure of undesired concepts. Most existing methods focus on modifying the generation probabilities conditioned on the texts containing target concepts. However, they fail to guarantee the desired generation of texts unseen in the training phase, especially for the adversarial texts from malicious attacks. In this paper, we analyze the erasure task and point out that existing methods cannot guarantee the minimization of the total probabilities of undesired generation. To tackle this problem, we propose Dark Miner. It entails a recurring three-stage process that comprises mining, verifying, and circumventing. This method greedily mines embeddings with maximum generation probabilities of target concepts and more effectively reduces their generation. In the experiments, we evaluate its performance on the inappropriateness, object, and style concepts. Compared with the previous methods, our method achieves better erasure and defense results, especially under multiple adversarial attacks, while preserving the native generation capability of the models. Our code will be available on GitHub.
>
---
#### [replaced 023] AdaRing: Towards Ultra-Light Vision-Language Adaptation via Cross-Layer Tensor Ring Decomposition
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.11870v2](http://arxiv.org/pdf/2508.11870v2)**

> **作者:** Ying Huang; Yuanbin Man; Wenqi Jia; Zhengzhong Tu; Junzhou Huang; Miao Yin
>
> **摘要:** Adapter-based fine-tuning has gained remarkable attention in adapting large pre-trained vision language models (VLMs) for a wide range of downstream tasks efficiently. In this paradigm, only the inserted adapters are fine-tuned, without the need for training the original VLM backbone. Existing works scale adapters by integrating them into every layer of VLMs to increase the capacity of adapters. However, these methods face two primary limitations: 1) limited compression rate due to ignoring cross-layer redundancy, and 2) limited representational capacity across homogeneous adapters. In this paper, we propose a novel vision-language fine-tuning framework based on cross-layer tensor ring decomposition (TRD) with the integration and collaboration of diverse adapters, called AdaRing, achieving ultra-light parameter-efficient adaptation of VLMs on various tasks. To remove the high redundancy that exists among adapters across layers, we exploit the tensor-level low-rankness to formulate adapters as layer-shared tensor cores and layer-specific slices. Moreover, guided by generalization-aware fine-tuning, diverse rank-driven adapters cooperate to handle tasks that require different representations. Our experiments show that the proposed AdaRing achieves the state-of-the-art performance while reducing average training parameters by 90%.
>
---
#### [replaced 024] MMAD: Multi-label Micro-Action Detection in Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.05311v3](http://arxiv.org/pdf/2407.05311v3)**

> **作者:** Kun Li; Pengyu Liu; Dan Guo; Fei Wang; Zhiliang Wu; Hehe Fan; Meng Wang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Human body actions are an important form of non-verbal communication in social interactions. This paper specifically focuses on a subset of body actions known as micro-actions, which are subtle, low-intensity body movements with promising applications in human emotion analysis. In real-world scenarios, human micro-actions often temporally co-occur, with multiple micro-actions overlapping in time, such as concurrent head and hand movements. However, current research primarily focuses on recognizing individual micro-actions while overlooking their co-occurring nature. To address this gap, we propose a new task named Multi-label Micro-Action Detection (MMAD), which involves identifying all micro-actions in a given short video, determining their start and end times, and categorizing them. Accomplishing this requires a model capable of accurately capturing both long-term and short-term action relationships to detect multiple overlapping micro-actions. To facilitate the MMAD task, we introduce a new dataset named Multi-label Micro-Action-52 (MMA-52) and propose a baseline method equipped with a dual-path spatial-temporal adapter to address the challenges of subtle visual change in MMAD. We hope that MMA-52 can stimulate research on micro-action analysis in videos and prompt the development of spatio-temporal modeling in human-centric video understanding. The proposed MMA-52 dataset is available at: https://github.com/VUT-HFUT/Micro-Action.
>
---
#### [replaced 025] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21755v2](http://arxiv.org/pdf/2503.21755v2)**

> **作者:** Dian Zheng; Ziqi Huang; Hongbo Liu; Kai Zou; Yinan He; Fan Zhang; Lulu Gu; Yuanhan Zhang; Jingwen He; Wei-Shi Zheng; Yu Qiao; Ziwei Liu
>
> **备注:** Equal contributions from first two authors. Project page: https://vchitect.github.io/VBench-2.0-project/ Code: https://github.com/Vchitect/VBench
>
> **摘要:** Video generation has advanced significantly, evolving from producing unrealistic outputs to generating videos that appear visually convincing and temporally coherent. To evaluate these video generative models, benchmarks such as VBench have been developed to assess their faithfulness, measuring factors like per-frame aesthetics, temporal consistency, and basic prompt adherence. However, these aspects mainly represent superficial faithfulness, which focus on whether the video appears visually convincing rather than whether it adheres to real-world principles. While recent models perform increasingly well on these metrics, they still struggle to generate videos that are not just visually plausible but fundamentally realistic. To achieve real "world models" through video generation, the next frontier lies in intrinsic faithfulness to ensure that generated videos adhere to physical laws, commonsense reasoning, anatomical correctness, and compositional integrity. Achieving this level of realism is essential for applications such as AI-assisted filmmaking and simulated world modeling. To bridge this gap, we introduce VBench-2.0, a next-generation benchmark designed to automatically evaluate video generative models for their intrinsic faithfulness. VBench-2.0 assesses five key dimensions: Human Fidelity, Controllability, Creativity, Physics, and Commonsense, each further broken down into fine-grained capabilities. Tailored to individual dimensions, our evaluation framework integrates generalists such as SOTA VLMs and LLMs, and specialists, including anomaly detection methods proposed for video generation. We conduct extensive human annotations to ensure evaluation alignment with human judgment. By pushing beyond superficial faithfulness toward intrinsic faithfulness, VBench-2.0 aims to set a new standard for the next generation of video generative models in pursuit of intrinsic faithfulness.
>
---
#### [replaced 026] DuCos: Duality Constrained Depth Super-Resolution via Foundation Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.04171v2](http://arxiv.org/pdf/2503.04171v2)**

> **作者:** Zhiqiang Yan; Zhengxue Wang; Haoye Dong; Jun Li; Jian Yang; Gim Hee Lee
>
> **备注:** ICCV 2025
>
> **摘要:** We introduce DuCos, a novel depth super-resolution framework grounded in Lagrangian duality theory, offering a flexible integration of multiple constraints and reconstruction objectives to enhance accuracy and robustness. Our DuCos is the first to significantly improve generalization across diverse scenarios with foundation models as prompts. The prompt design consists of two key components: Correlative Fusion (CF) and Gradient Regulation (GR). CF facilitates precise geometric alignment and effective fusion between prompt and depth features, while GR refines depth predictions by enforcing consistency with sharp-edged depth maps derived from foundation models. Crucially, these prompts are seamlessly embedded into the Lagrangian constraint term, forming a synergistic and principled framework. Extensive experiments demonstrate that DuCos outperforms existing state-of-the-art methods, achieving superior accuracy, robustness, and generalization.
>
---
#### [replaced 027] What Makes for Good Image Captions?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.00485v3](http://arxiv.org/pdf/2405.00485v3)**

> **作者:** Delong Chen; Samuel Cahyawijaya; Etsuko Ishii; Ho Shu Chan; Yejin Bang; Pascale Fung
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** This paper establishes a formal information-theoretic framework for image captioning, conceptualizing captions as compressed linguistic representations that selectively encode semantic units in images. Our framework posits that good image captions should balance three key aspects: informationally sufficient, minimally redundant, and readily comprehensible by humans. By formulating these aspects as quantitative measures with adjustable weights, our framework provides a flexible foundation for analyzing and optimizing image captioning systems across diverse task requirements. To demonstrate its applicability, we introduce the Pyramid of Captions (PoCa) method, which generates enriched captions by integrating local and global visual information. We present both theoretical proof that PoCa improves caption quality under certain assumptions, and empirical validation of its effectiveness across various image captioning models and datasets.
>
---
#### [replaced 028] ViT-FIQA: Assessing Face Image Quality using Vision Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13957v2](http://arxiv.org/pdf/2508.13957v2)**

> **作者:** Andrea Atzori; Fadi Boutros; Naser Damer
>
> **备注:** Accepted at the IEEE/CVF International Conference on Computer Vision Workshops 2025 (ICCVW 2025)
>
> **摘要:** Face Image Quality Assessment (FIQA) aims to predict the utility of a face image for face recognition (FR) systems. State-of-the-art FIQA methods mainly rely on convolutional neural networks (CNNs), leaving the potential of Vision Transformer (ViT) architectures underexplored. This work proposes ViT-FIQA, a novel approach that extends standard ViT backbones, originally optimized for FR, through a learnable quality token designed to predict a scalar utility score for any given face image. The learnable quality token is concatenated with the standard image patch tokens, and the whole sequence is processed via global self-attention by the ViT encoders to aggregate contextual information across all patches. At the output of the backbone, ViT-FIQA branches into two heads: (1) the patch tokens are passed through a fully connected layer to learn discriminative face representations via a margin-penalty softmax loss, and (2) the quality token is fed into a regression head to learn to predict the face sample's utility. Extensive experiments on challenging benchmarks and several FR models, including both CNN- and ViT-based architectures, demonstrate that ViT-FIQA consistently achieves top-tier performance. These results underscore the effectiveness of transformer-based architectures in modeling face image utility and highlight the potential of ViTs as a scalable foundation for future FIQA research https://cutt.ly/irHlzXUC.
>
---
#### [replaced 029] Endo-FASt3r: Endoscopic Foundation model Adaptation for Structure from motion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07204v4](http://arxiv.org/pdf/2503.07204v4)**

> **作者:** Mona Sheikh Zeinoddin; Mobarak I. Hoque; Zafer Tandogdu; Greg Shaw; Matthew J. Clarkson; Evangelos Mazomenos; Danail Stoyanov
>
> **摘要:** Accurate depth and camera pose estimation is essential for achieving high-quality 3D visualisations in robotic-assisted surgery. Despite recent advancements in foundation model adaptation to monocular depth estimation of endoscopic scenes via self-supervised learning (SSL), no prior work has explored their use for pose estimation. These methods rely on low rank-based adaptation approaches, which constrain model updates to a low-rank space. We propose Endo-FASt3r, the first monocular SSL depth and pose estimation framework that uses foundation models for both tasks. We extend the Reloc3r relative pose estimation foundation model by designing Reloc3rX, introducing modifications necessary for convergence in SSL. We also present DoMoRA, a novel adaptation technique that enables higher-rank updates and faster convergence. Experiments on the SCARED dataset show that Endo-FASt3r achieves a substantial $10\%$ improvement in pose estimation and a $2\%$ improvement in depth estimation over prior work. Similar performance gains on the Hamlyn and StereoMIS datasets reinforce the generalisability of Endo-FASt3r across different datasets.
>
---
#### [replaced 030] Cherenkov Imaged Bio-morphological Features Verify Patient Positioning with Deformable Tissue Translocation in Breast Radiotherapy
- **分类: physics.med-ph; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.05680v2](http://arxiv.org/pdf/2409.05680v2)**

> **作者:** Yao Chen; Savannah M. Decker; Petr Bruza; David J. Gladstone; Lesley A. Jarvis; Brian W. Pogue; Kimberley S. Samkoe; Rongxiao Zhang
>
> **备注:** 25 pages, 4 figures, 1 table, journal under review
>
> **摘要:** Accurate patient positioning is critical for precise radiotherapy dose delivery, as positioning errors can significantly affect treatment outcomes. This study introduces a novel method for tracking loco-regional tissue deformation through Cherenkov image analysis during fractionated breast cancer radiotherapy. The primary goal was to develop and test an algorithm for Cherenkov-based regional position accuracy quantification, specifically for loco-regional deformations, which lack ideal quantification methods in radiotherapy. Blood vessel detection and segmentation were developed in Cherenkov images using a tissue phantom with incremental movements, and later applied to images from fractionated whole breast radiotherapy in human patients (n=10). A combined rigid and non-rigid registration technique was used to detect inter- and intra-fractional positioning variations. This approach quantified positioning variations in two parts: a global shift from rigid registration and a two-dimensional variation map of loco-regional deformation from non-rigid registration. The methodology was validated using an anthropomorphic chest phantom experiment, where known treatment couch translations and respiratory motion were simulated to assess inter- and intra-fractional uncertainties, yielding an average accuracy of 0.83 mm for couch translations up to 20 mm. Analysis of clinical Cherenkov data from ten breast cancer patients showed an inter-fraction setup variation of 3.7 plus minus 2.4 mm relative to the first fraction and loco-regional deformations (95th percentile) of up to 3.3 plus minus 1.9 mm. This study presents a Cherenkov-based approach to quantify global and local positioning variations, demonstrating feasibility in addressing loco-regional deformations that conventional imaging techniques fail to capture.
>
---
#### [replaced 031] Neural Restoration of Greening Defects in Historical Autochrome Photographs Based on Purely Synthetic Data
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.22291v2](http://arxiv.org/pdf/2505.22291v2)**

> **作者:** Saptarshi Neil Sinha; P. Julius Kuehn; Johannes Koppe; Arjan Kuijper; Michael Weinmann
>
> **摘要:** The preservation of early visual arts, particularly color photographs, is challenged by deterioration caused by aging and improper storage, leading to issues like blurring, scratches, color bleeding, and fading defects. Despite great advances in image restoration and enhancement in recent years, such systematic defects often cannot be restored by current state-of-the-art software features as available e.g. in Adobe Photoshop, but would require the incorporation of defect-aware priors into the underlying machine learning techniques. However, there are no publicly available datasets of autochromes with defect annotations. In this paper, we address these limitations and present the first approach that allows the automatic removal of greening color defects in digitized autochrome photographs. For this purpose, we introduce an approach for accurately simulating respective defects and use the respectively obtained synthesized data with its ground truth defect annotations to train a generative AI model with a carefully designed loss function that accounts for color imbalances between defected and non-defected areas. As demonstrated in our evaluation, our approach allows for the efficient and effective restoration of the considered defects, thereby overcoming limitations of alternative techniques that struggle with accurately reproducing original colors and may require significant manual effort.
>
---
#### [replaced 032] Hands-On: Segmenting Individual Signs from Continuous Sequences
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.08593v4](http://arxiv.org/pdf/2504.08593v4)**

> **作者:** JianHe Low; Harry Walsh; Ozge Mercanoglu Sincan; Richard Bowden
>
> **备注:** Accepted in the 19th IEEE International Conference on Automatic Face and Gesture Recognition
>
> **摘要:** This work tackles the challenge of continuous sign language segmentation, a key task with huge implications for sign language translation and data annotation. We propose a transformer-based architecture that models the temporal dynamics of signing and frames segmentation as a sequence labeling problem using the Begin-In-Out (BIO) tagging scheme. Our method leverages the HaMeR hand features, and is complemented with 3D Angles. Extensive experiments show that our model achieves state-of-the-art results on the DGS Corpus, while our features surpass prior benchmarks on BSLCorpus.
>
---
#### [replaced 033] Self-supervised Learning of LiDAR 3D Point Clouds via 2D-3D Neural Calibration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.12452v5](http://arxiv.org/pdf/2401.12452v5)**

> **作者:** Yifan Zhang; Junhui Hou; Siyu Ren; Jinjian Wu; Yixuan Yuan; Guangming Shi
>
> **备注:** Accepted to TPAMI2025
>
> **摘要:** This paper introduces a novel self-supervised learning framework for enhancing 3D perception in autonomous driving scenes. Specifically, our approach, namely NCLR, focuses on 2D-3D neural calibration, a novel pretext task that estimates the rigid pose aligning camera and LiDAR coordinate systems. First, we propose the learnable transformation alignment to bridge the domain gap between image and point cloud data, converting features into a unified representation space for effective comparison and matching. Second, we identify the overlapping area between the image and point cloud with the fused features. Third, we establish dense 2D-3D correspondences to estimate the rigid pose. The framework not only learns fine-grained matching from points to pixels but also achieves alignment of the image and point cloud at a holistic level, understanding the LiDAR-to-camera extrinsic parameters. We demonstrate the efficacy of NCLR by applying the pre-trained backbone to downstream tasks, such as LiDAR-based 3D semantic segmentation, object detection, and panoptic segmentation. Comprehensive experiments on various datasets illustrate the superiority of NCLR over existing self-supervised methods. The results confirm that joint learning from different modalities significantly enhances the network's understanding abilities and effectiveness of learned representation. The code is publicly available at https://github.com/Eaphan/NCLR.
>
---
#### [replaced 034] Latent Interpolation Learning Using Diffusion Models for Cardiac Volume Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13826v2](http://arxiv.org/pdf/2508.13826v2)**

> **作者:** Niklas Bubeck; Suprosanna Shit; Chen Chen; Can Zhao; Pengfei Guo; Dong Yang; Georg Zitzlsberger; Daguang Xu; Bernhard Kainz; Daniel Rueckert; Jiazhen Pan
>
> **摘要:** Cardiac Magnetic Resonance (CMR) imaging is a critical tool for diagnosing and managing cardiovascular disease, yet its utility is often limited by the sparse acquisition of 2D short-axis slices, resulting in incomplete volumetric information. Accurate 3D reconstruction from these sparse slices is essential for comprehensive cardiac assessment, but existing methods face challenges, including reliance on predefined interpolation schemes (e.g., linear or spherical), computational inefficiency, and dependence on additional semantic inputs such as segmentation labels or motion data. To address these limitations, we propose a novel \textbf{Ca}rdiac \textbf{L}atent \textbf{I}nterpolation \textbf{D}iffusion (CaLID) framework that introduces three key innovations. First, we present a data-driven interpolation scheme based on diffusion models, which can capture complex, non-linear relationships between sparse slices and improves reconstruction accuracy. Second, we design a computationally efficient method that operates in the latent space and speeds up 3D whole-heart upsampling time by a factor of 24, reducing computational overhead compared to previous methods. Third, with only sparse 2D CMR images as input, our method achieves SOTA performance against baseline methods, eliminating the need for auxiliary input such as morphological guidance, thus simplifying workflows. We further extend our method to 2D+T data, enabling the effective modeling of spatiotemporal dynamics and ensuring temporal coherence. Extensive volumetric evaluations and downstream segmentation tasks demonstrate that CaLID achieves superior reconstruction quality and efficiency. By addressing the fundamental limitations of existing approaches, our framework advances the state of the art for spatio and spatiotemporal whole-heart reconstruction, offering a robust and clinically practical solution for cardiovascular imaging.
>
---
#### [replaced 035] Bench2ADVLM: A Closed-Loop Benchmark for Vision-language Models in Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02028v2](http://arxiv.org/pdf/2508.02028v2)**

> **作者:** Tianyuan Zhang; Ting Jin; Lu Wang; Jiangfan Liu; Siyuan Liang; Mingchuan Zhang; Aishan Liu; Xianglong Liu
>
> **摘要:** Vision-Language Models (VLMs) have recently emerged as a promising paradigm in autonomous driving (AD). However, current performance evaluation protocols for VLM-based AD systems (ADVLMs) are predominantly confined to open-loop settings with static inputs, neglecting the more realistic and informative closed-loop setting that captures interactive behavior, feedback resilience, and real-world safety. To address this, we introduce Bench2ADVLM, a unified hierarchical closed-loop evaluation framework for real-time, interactive assessment of ADVLMs across both simulation and physical platforms. Inspired by dual-process theories of cognition, we first adapt diverse ADVLMs to simulation environments via a dual-system adaptation architecture. In this design, heterogeneous high-level driving commands generated by target ADVLMs (fast system) are interpreted by a general-purpose VLM (slow system) into standardized mid-level control actions suitable for execution in simulation. To bridge the gap between simulation and reality, we design a physical control abstraction layer that translates these mid-level actions into low-level actuation signals, enabling, for the first time, closed-loop testing of ADVLMs on physical vehicles. To enable more comprehensive evaluation, Bench2ADVLM introduces a self-reflective scenario generation module that automatically explores model behavior and uncovers potential failure modes for safety-critical scenario generation. Overall, Bench2ADVLM establishes a hierarchical evaluation pipeline that seamlessly integrates high-level abstract reasoning, mid-level simulation actions, and low-level real-world execution. Experiments on diverse scenarios across multiple state-of-the-art ADVLMs and physical platforms validate the diagnostic strength of our framework, revealing that existing ADVLMs still exhibit limited performance under closed-loop conditions.
>
---
#### [replaced 036] Extending Foundational Monocular Depth Estimators to Fisheye Cameras with Calibration Tokens
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.04928v3](http://arxiv.org/pdf/2508.04928v3)**

> **作者:** Suchisrit Gangopadhyay; Jung-Hee Kim; Xien Chen; Patrick Rim; Hyoungseob Park; Alex Wong
>
> **摘要:** We propose a method to extend foundational monocular depth estimators (FMDEs), trained on perspective images, to fisheye images. Despite being trained on tens of millions of images, FMDEs are susceptible to the covariate shift introduced by changes in camera calibration (intrinsic, distortion) parameters, leading to erroneous depth estimates. Our method aligns the distribution of latent embeddings encoding fisheye images to those of perspective images, enabling the reuse of FMDEs for fisheye cameras without retraining or finetuning. To this end, we introduce a set of Calibration Tokens as a light-weight adaptation mechanism that modulates the latent embeddings for alignment. By exploiting the already expressive latent space of FMDEs, we posit that modulating their embeddings avoids the negative impact of artifacts and loss introduced in conventional recalibration or map projection to a canonical reference frame in the image space. Our method is self-supervised and does not require fisheye images but leverages publicly available large-scale perspective image datasets. This is done by recalibrating perspective images to fisheye images, and enforcing consistency between their estimates during training. We evaluate our approach with several FMDEs, on both indoors and outdoors, where we consistently improve over state-of-the-art methods using a single set of tokens for both. Code available at: https://github.com/JungHeeKim29/calibration-token.
>
---
#### [replaced 037] SketchDNN: Joint Continuous-Discrete Diffusion for CAD Sketch Generation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11579v2](http://arxiv.org/pdf/2507.11579v2)**

> **作者:** Sathvik Chereddy; John Femiani
>
> **备注:** 17 pages, 63 figures, Proceedings of the 42nd International Conference on Machine Learning (ICML2025)
>
> **摘要:** We present SketchDNN, a generative model for synthesizing CAD sketches that jointly models both continuous parameters and discrete class labels through a unified continuous-discrete diffusion process. Our core innovation is Gaussian-Softmax diffusion, where logits perturbed with Gaussian noise are projected onto the probability simplex via a softmax transformation, facilitating blended class labels for discrete variables. This formulation addresses 2 key challenges, namely, the heterogeneity of primitive parameterizations and the permutation invariance of primitives in CAD sketches. Our approach significantly improves generation quality, reducing Fr\'echet Inception Distance (FID) from 16.04 to 7.80 and negative log-likelihood (NLL) from 84.8 to 81.33, establishing a new state-of-the-art in CAD sketch generation on the SketchGraphs dataset.
>
---
#### [replaced 038] Impact of Clinical Image Quality on Efficient Foundation Model Finetuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.11864v2](http://arxiv.org/pdf/2508.11864v2)**

> **作者:** Yucheng Tang; Pawel Rajwa; Alexander Ng; Yipei Wang; Wen Yan; Natasha Thorley; Aqua Asif; Clare Allen; Louise Dickinson; Francesco Giganti; Shonit Punwani; Daniel C. Alexander; Veeru Kasivisvanathan; Yipeng Hu
>
> **备注:** This paper was accepted to the 1st MICCAI Workshop on Efficient Medical AI (EMA4MICCAI2025) and selected for oral presentation
>
> **摘要:** Foundation models in medical imaging have shown promising label efficiency, achieving high performance on downstream tasks using only a fraction of the annotated data otherwise required. In this study, we evaluate this potential in the context of prostate multiparametric MRI using ProFound, a recently developed domain-specific vision foundation model pretrained on large-scale prostate MRI datasets. We investigate the impact of variable image quality on the label-efficient finetuning, by quantifying the generalisability of the finetuned models. We conduct a comprehensive set of experiments by systematically varying the ratios of high- and low-quality images in the finetuning and evaluation sets. Our findings indicate that image quality distribution and its finetune-and-test mismatch significantly affect model performance. In particular: a) Varying the ratio of high- to low-quality images between finetuning and test sets leads to notable differences in downstream performance; and b) The presence of sufficient high-quality images in the finetuning set is critical for maintaining strong performance, whilst the importance of matched finetuning and testing distribution varies between different downstream tasks, such as automated radiology reporting and prostate cancer detection. Importantly, experimental results also show that, although finetuning requires significantly less labeled data compared to training from scratch when the quality ratio is consistent, this label efficiency is not independent of the image quality distribution. For example, we show cases that, without sufficient high-quality images in finetuning, finetuned models may fail to outperform those without pretraining.
>
---
#### [replaced 039] Identity Preserving 3D Head Stylization with Multiview Score Distillation
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.13536v3](http://arxiv.org/pdf/2411.13536v3)**

> **作者:** Bahri Batuhan Bilecen; Ahmet Berke Gokmen; Furkan Guzelant; Aysegul Dundar
>
> **备注:** https://three-bee.github.io/head_stylization
>
> **摘要:** 3D head stylization transforms realistic facial features into artistic representations, enhancing user engagement across gaming and virtual reality applications. While 3D-aware generators have made significant advancements, many 3D stylization methods primarily provide near-frontal views and struggle to preserve the unique identities of original subjects, often resulting in outputs that lack diversity and individuality. This paper addresses these challenges by leveraging the PanoHead model, synthesizing images from a comprehensive 360-degree perspective. We propose a novel framework that employs negative log-likelihood distillation (LD) to enhance identity preservation and improve stylization quality. By integrating multi-view grid score and mirror gradients within the 3D GAN architecture and introducing a score rank weighing technique, our approach achieves substantial qualitative and quantitative improvements. Our findings not only advance the state of 3D head stylization but also provide valuable insights into effective distillation processes between diffusion models and GANs, focusing on the critical issue of identity preservation. Please visit the https://three-bee.github.io/head_stylization for more visuals.
>
---
#### [replaced 040] RotBench: Evaluating Multimodal Large Language Models on Identifying Image Rotation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.13968v2](http://arxiv.org/pdf/2508.13968v2)**

> **作者:** Tianyi Niu; Jaemin Cho; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** 20 pages. Code and data: https://github.com/tianyiniu/RotBench
>
> **摘要:** We investigate to what extent Multimodal Large Language Models (MLLMs) can accurately identify the orientation of input images rotated 0{\deg}, 90{\deg}, 180{\deg}, and 270{\deg}. This task demands robust visual reasoning capabilities to detect rotational cues and contextualize spatial relationships within images, regardless of their orientation. To evaluate MLLMs on these abilities, we introduce RotBench -- a 350-image manually-filtered benchmark comprising lifestyle, portrait, and landscape images. Despite the relatively simple nature of this task, we show that several state-of-the-art open and proprietary MLLMs, including GPT-5, o3, and Gemini-2.5-Pro, do not reliably identify rotation in input images. Providing models with auxiliary information -- including captions, depth maps, and more -- or using chain-of-thought prompting offers only small and inconsistent improvements. Our results indicate that most models are able to reliably identify right-side-up (0{\deg}) images, while certain models are able to identify upside-down (180{\deg}) images. None can reliably distinguish between 90{\deg} and 270{\deg}. Simultaneously showing the image rotated in different orientations leads to moderate performance gains for reasoning models, while a modified setup using voting improves the performance of weaker models. We further show that fine-tuning does not improve models' ability to distinguish 90{\deg} and 270{\deg} rotations, despite substantially improving the identification of 180{\deg} images. Together, these results reveal a significant gap between MLLMs' spatial reasoning capabilities and human perception in identifying rotation.
>
---
#### [replaced 041] Diffusion MRI with Machine Learning
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.00019v4](http://arxiv.org/pdf/2402.00019v4)**

> **作者:** Davood Karimi; Simon K. Warfield
>
> **摘要:** \hspace{2mm} Diffusion-weighted magnetic resonance imaging (dMRI) of the brain offers unique capabilities including noninvasive probing of tissue microstructure and structural connectivity. It is widely used for clinical assessment of disease and injury, and for neuroscience research. Analyzing the dMRI data to extract useful information for medical and scientific purposes can be challenging. The dMRI measurements may suffer from strong noise and artifacts, and may exhibit high inter-session and inter-scanner variability in the data, as well as inter-subject heterogeneity in brain structure. Moreover, the relationship between measurements and the phenomena of interest can be highly complex. Recent years have witnessed increasing use of machine learning methods for dMRI analysis. This manuscript aims to assess these efforts, with a focus on methods that have addressed data preprocessing and harmonization, microstructure mapping, tractography, and white matter tract analysis. We study the main findings, strengths, and weaknesses of the existing methods and suggest topics for future research. We find that machine learning may be exceptionally suited to tackle some of the difficult tasks in dMRI analysis. However, for this to happen, several shortcomings of existing methods and critical unresolved issues need to be addressed. There is a pressing need to improve evaluation practices, to increase the availability of rich training datasets and validation benchmarks, as well as model generalizability, reliability, and explainability concerns.
>
---
#### [replaced 042] Reconstruction-Free Anomaly Detection with Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.05662v2](http://arxiv.org/pdf/2504.05662v2)**

> **作者:** Shunsuke Sakai; Xiangteng He; Chunzhi Gu; Leonid Sigal; Tatsuhito Hasegawa
>
> **备注:** Code is available at https://github.com/SkyShunsuke/InversionAD
>
> **摘要:** Despite the remarkable success, recent reconstruction-based anomaly detection (AD) methods via diffusion modeling still involve fine-grained noise-strength tuning and computationally expensive multi-step denoising, leading to a fundamental tension between fidelity and efficiency. In this paper, we propose a novel inversion-based AD approach - detection via noising in latent space - which circumvents explicit reconstruction. Importantly, we contend that the limitations in prior reconstruction-based methods originate from the prevailing detection via denoising in RGB space paradigm. To address this, we model AD under a reconstruction-free formulation, which directly infers the final latent variable corresponding to the input image via DDIM inversion, and then measures the deviation based on the known prior distribution for anomaly scoring. Specifically, in approximating the original probability flow ODE using the Euler method, we only enforce very few inversion steps to noise the clean image to pursue inference efficiency. As the added noise is adaptively derived with the learned diffusion model, the original features for the clean testing image can still be leveraged to yield high detection accuracy. We perform extensive experiments and detailed analysis across three widely used image AD datasets under the unsupervised unified setting to demonstrate the effectiveness of our model, regarding state-of-the-art AD performance, and about 2 times inference time speedup without diffusion distillation.
>
---
#### [replaced 043] JRDB-Reasoning: A Difficulty-Graded Benchmark for Visual Reasoning in Robotics
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.10287v2](http://arxiv.org/pdf/2508.10287v2)**

> **作者:** Simindokht Jahangard; Mehrzad Mohammadi; Yi Shen; Zhixi Cai; Hamid Rezatofighi
>
> **摘要:** Recent advances in Vision-Language Models (VLMs) and large language models (LLMs) have greatly enhanced visual reasoning, a key capability for embodied AI agents like robots. However, existing visual reasoning benchmarks often suffer from several limitations: they lack a clear definition of reasoning complexity, offer have no control to generate questions over varying difficulty and task customization, and fail to provide structured, step-by-step reasoning annotations (workflows). To bridge these gaps, we formalize reasoning complexity, introduce an adaptive query engine that generates customizable questions of varying complexity with detailed intermediate annotations, and extend the JRDB dataset with human-object interaction and geometric relationship annotations to create JRDB-Reasoning, a benchmark tailored for visual reasoning in human-crowded environments. Our engine and benchmark enable fine-grained evaluation of visual reasoning frameworks and dynamic assessment of visual-language models across reasoning levels.
>
---
#### [replaced 044] AtmosMJ: Revisiting Gating Mechanism for AI Weather Forecasting Beyond the Year Scale
- **分类: cs.LG; cs.AI; cs.CV; physics.ao-ph**

- **链接: [http://arxiv.org/pdf/2506.09733v3](http://arxiv.org/pdf/2506.09733v3)**

> **作者:** Minjong Cheon
>
> **备注:** All authors of this manuscript have not reached a consensus on its submission to arXiv. Since at least one co-author does not agree with the current version being publicly available, we respectfully request the withdrawal of this preprint in accordance with the authors' collective decision
>
> **摘要:** The advent of Large Weather Models (LWMs) has marked a turning point in data-driven forecasting, with many models now outperforming traditional numerical systems in the medium range. However, achieving stable, long-range autoregressive forecasts beyond a few weeks remains a significant challenge. Prevailing state-of-the-art models that achieve year-long stability, such as SFNO and DLWP-HPX, have relied on transforming input data onto non-standard spatial domains like spherical harmonics or HEALPix meshes. This has led to the prevailing assumption that such representations are necessary to enforce physical consistency and long-term stability. This paper challenges that assumption by investigating whether comparable long-range performance can be achieved on the standard latitude-longitude grid. We introduce AtmosMJ, a deep convolutional network that operates directly on ERA5 data without any spherical remapping. The model's stability is enabled by a novel Gated Residual Fusion (GRF) mechanism, which adaptively moderates feature updates to prevent error accumulation over long recursive simulations. Our results demonstrate that AtmosMJ produces stable and physically plausible forecasts for about 500 days. In quantitative evaluations, it achieves competitive 10-day forecast accuracy against models like Pangu-Weather and GraphCast, all while requiring a remarkably low training budget of 5.7 days on a V100 GPU. Our findings suggest that efficient architectural design, rather than non-standard data representation, can be the key to unlocking stable and computationally efficient long-range weather prediction.
>
---
#### [replaced 045] CoT-Segmenter: Enhancing OOD Detection in Dense Road Scenes via Chain-of-Thought Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03984v2](http://arxiv.org/pdf/2507.03984v2)**

> **作者:** Jeonghyo Song; Kimin Yun; DaeUng Jo; Jinyoung Kim; Youngjoon Yoo
>
> **备注:** 6 pages, 3 figures. Accepted at IEEE International Conference on Advanced Visual and Signal-Based Systems 2025
>
> **摘要:** Effective Out-of-Distribution (OOD) detection is criti-cal for ensuring the reliability of semantic segmentation models, particularly in complex road environments where safety and accuracy are paramount. Despite recent advancements in large language models (LLMs), notably GPT-4, which significantly enhanced multimodal reasoning through Chain-of-Thought (CoT) prompting, the application of CoT-based visual reasoning for OOD semantic segmentation remains largely unexplored. In this paper, through extensive analyses of the road scene anomalies, we identify three challenging scenarios where current state-of-the-art OOD segmentation methods consistently struggle: (1) densely packed and overlapping objects, (2) distant scenes with small objects, and (3) large foreground-dominant objects. To address the presented challenges, we propose a novel CoT-based framework targeting OOD detection in road anomaly scenes. Our method leverages the extensive knowledge and reasoning capabilities of foundation models, such as GPT-4, to enhance OOD detection through improved image understanding and prompt-based reasoning aligned with observed problematic scene attributes. Extensive experiments show that our framework consistently outperforms state-of-the-art methods on both standard benchmarks and our newly defined challenging subset of the RoadAnomaly dataset, offering a robust and interpretable solution for OOD semantic segmentation in complex driving environments.
>
---
#### [replaced 046] Real-time Neural Rendering of LiDAR Point Clouds
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2502.11618v2](http://arxiv.org/pdf/2502.11618v2)**

> **作者:** Joni Vanherck; Brent Zoomers; Tom Mertens; Lode Jorissen; Nick Michiels
>
> **备注:** Accepted at Eurographics 2025
>
> **摘要:** Static LiDAR scanners produce accurate, dense, colored point clouds, but often contain obtrusive artifacts which makes them ill-suited for direct display. We propose an efficient method to render photorealistic images of such scans without any expensive preprocessing or training of a scene-specific model. A naive projection of the point cloud to the output view using 1x1 pixels is fast and retains the available detail, but also results in unintelligible renderings as background points leak in between the foreground pixels. The key insight is that these projections can be transformed into a realistic result using a deep convolutional model in the form of a U-Net, and a depth-based heuristic that prefilters the data. The U-Net also handles LiDAR-specific problems such as missing parts due to occlusion, color inconsistencies and varying point densities. We also describe a method to generate synthetic training data to deal with imperfectly-aligned ground truth images. Our method achieves real-time rendering rates using an off-the-shelf GPU and outperforms the state-of-the-art in both speed and quality.
>
---
#### [replaced 047] Dynamic watermarks in images generated by diffusion models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.08927v2](http://arxiv.org/pdf/2502.08927v2)**

> **作者:** Yunzhuo Chen; Naveed Akhtar; Nur Al Hasan Haldar; Ajmal Mian
>
> **摘要:** High-fidelity text-to-image diffusion models have revolutionized visual content generation, but their widespread use raises significant ethical concerns, including intellectual property protection and the misuse of synthetic media. To address these challenges, we propose a novel multi-stage watermarking framework for diffusion models, designed to establish copyright and trace generated images back to their source. Our multi-stage watermarking technique involves embedding: (i) a fixed watermark that is localized in the diffusion model's learned noise distribution and, (ii) a human-imperceptible, dynamic watermark in generates images, leveraging a fine-tuned decoder. By leveraging the Structural Similarity Index Measure (SSIM) and cosine similarity, we adapt the watermark's shape and color to the generated content while maintaining robustness. We demonstrate that our method enables reliable source verification through watermark classification, even when the dynamic watermark is adjusted for content-specific variations. Source model verification is enabled through watermark classification. o support further research, we generate a dataset of watermarked images and introduce a methodology to evaluate the statistical impact of watermarking on generated content.Additionally, we rigorously test our framework against various attack scenarios, demonstrating its robustness and minimal impact on image quality. Our work advances the field of AI-generated content security by providing a scalable solution for model ownership verification and misuse prevention.
>
---
#### [replaced 048] 3D-Generalist: Self-Improving Vision-Language-Action Models for Crafting 3D Worlds
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06484v2](http://arxiv.org/pdf/2507.06484v2)**

> **作者:** Fan-Yun Sun; Shengguang Wu; Christian Jacobsen; Thomas Yim; Haoming Zou; Alex Zook; Shangru Li; Yu-Hsin Chou; Ethem Can; Xunlei Wu; Clemens Eppner; Valts Blukis; Jonathan Tremblay; Jiajun Wu; Stan Birchfield; Nick Haber
>
> **备注:** project website: https://ai.stanford.edu/~sunfanyun/3d-generalist/
>
> **摘要:** Despite large-scale pretraining endowing models with language and vision reasoning capabilities, improving their spatial reasoning capability remains challenging due to the lack of data grounded in the 3D world. While it is possible for humans to manually create immersive and interactive worlds through 3D graphics, as seen in applications such as VR, gaming, and robotics, this process remains highly labor-intensive. In this paper, we propose a scalable method for generating high-quality 3D environments that can serve as training data for foundation models. We recast 3D environment building as a sequential decision-making problem, employing Vision-Language-Models (VLMs) as policies that output actions to jointly craft a 3D environment's layout, materials, lighting, and assets. Our proposed framework, 3D-Generalist, trains VLMs to generate more prompt-aligned 3D environments via self-improvement fine-tuning. We demonstrate the effectiveness of 3D-Generalist and the proposed training strategy in generating simulation-ready 3D environments. Furthermore, we demonstrate its quality and scalability in synthetic data generation by pretraining a vision foundation model on the generated data. After fine-tuning the pre-trained model on downstream tasks, we show that it surpasses models pre-trained on meticulously human-crafted synthetic data and approaches results achieved with real data orders of magnitude larger.
>
---
#### [replaced 049] FlightPatchNet: Multi-Scale Patch Network with Differential Coding for Flight Trajectory Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.16200v3](http://arxiv.org/pdf/2405.16200v3)**

> **作者:** Lan Wu; Xuebin Wang; Ruijuan Chu; Guangyi Liu; Jing Zhang; Linyu Wang
>
> **备注:** Accepted by UAI 2025. Code is available at https://github.com/graceLan1994/FlightPatchNet
>
> **摘要:** Accurate multi-step flight trajectory prediction plays an important role in Air Traffic Control, which can ensure the safety of air transportation. Two main issues limit the flight trajectory prediction performance of existing works. The first issue is the negative impact on prediction accuracy caused by the significant differences in data range. The second issue is that real-world flight trajectories involve underlying temporal dependencies, and most existing methods fail to reveal the hidden complex temporal variations and extract features from one single time scale. To address the above issues, we propose FlightPatchNet, a multi-scale patch network with differential coding for flight trajectory prediction. Specifically, FlightPatchNet first utilizes differential coding to encode the original values of longitude and latitude into first-order differences and generates embeddings for all variables at each time step. Then, global temporal attention is introduced to explore the dependencies between different time steps. To fully explore the diverse temporal patterns in flight trajectories, a multi-scale patch network is delicately designed to serve as the backbone. The multi-scale patch network exploits stacked patch mixer blocks to capture inter- and intra-patch dependencies under different time scales, and further integrates multi-scale temporal features across different scales and variables. Finally, FlightPatchNet ensembles multiple predictors to make direct multi-step prediction. Extensive experiments on ADS-B datasets demonstrate that our model outperforms the competitive baselines.
>
---
#### [replaced 050] A Novel Image Similarity Metric for Scene Composition Structure
- **分类: cs.CV; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2508.05037v2](http://arxiv.org/pdf/2508.05037v2)**

> **作者:** Md Redwanul Haque; Manzur Murshed; Manoranjan Paul; Tsz-Kwan Lee
>
> **备注:** 2025 IEEE ICIPW (Generative AI for World Simulations and Communications)
>
> **摘要:** The rapid advancement of generative AI models necessitates novel methods for evaluating image quality that extend beyond human perception. A critical concern for these models is the preservation of an image's underlying Scene Composition Structure (SCS), which defines the geometric relationships among objects and the background, their relative positions, sizes, orientations, etc. Maintaining SCS integrity is paramount for ensuring faithful and structurally accurate GenAI outputs. Traditional image similarity metrics often fall short in assessing SCS. Pixel-level approaches are overly sensitive to minor visual noise, while perception-based metrics prioritize human aesthetic appeal, neither adequately capturing structural fidelity. Furthermore, recent neural-network-based metrics introduce training overheads and potential generalization issues. We introduce the SCS Similarity Index Measure (SCSSIM), a novel, analytical, and training-free metric that quantifies SCS preservation by exploiting statistical measures derived from the Cuboidal hierarchical partitioning of images, robustly capturing non-object-based structural relationships. Our experiments demonstrate SCSSIM's high invariance to non-compositional distortions, accurately reflecting unchanged SCS. Conversely, it shows a strong monotonic decrease for compositional distortions, precisely indicating when SCS has been altered. Compared to existing metrics, SCSSIM exhibits superior properties for structural evaluation, making it an invaluable tool for developing and evaluating generative models, ensuring the integrity of scene composition.
>
---
#### [replaced 051] Improving Token-based Object Detection with Video
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22562v2](http://arxiv.org/pdf/2506.22562v2)**

> **作者:** Abhineet Singh; Nilanjan Ray
>
> **备注:** Published in IEEE Access
>
> **摘要:** This paper improves upon the Pix2Seq object detector by extending it for videos. In the process, it introduces a new way to perform end-to-end video object detection that improves upon existing video detectors in two key ways. First, by representing objects as variable-length sequences of discrete tokens, we can succinctly represent widely varying numbers of video objects, with diverse shapes and locations, without having to inject any localization cues in the training process. This eliminates the need to sample the space of all possible boxes that constrains conventional detectors and thus solves the dual problems of loss sparsity during training and heuristics-based postprocessing during inference. Second, it conceptualizes and outputs the video objects as fully integrated and indivisible 3D boxes or tracklets instead of generating image-specific 2D boxes and linking these boxes together to construct the video object, as done in most conventional detectors. This allows it to scale effortlessly with available computational resources by simply increasing the length of the video subsequence that the network takes as input, even generalizing to multi-object tracking if the subsequence can span the entire video. We compare our video detector with the baseline Pix2Seq static detector on several datasets and demonstrate consistent improvement, although with strong signs of being bottlenecked by our limited computational resources. We also compare it with several video detectors on UA-DETRAC to show that it is competitive with the current state of the art even with the computational bottleneck. We make our code and models publicly available.
>
---
#### [replaced 052] ExpVG: Investigating the Design Space of Visual Grounding in Multimodal Large Language Model
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.08066v2](http://arxiv.org/pdf/2508.08066v2)**

> **作者:** Weitai Kang; Weiming Zhuang; Zhizhong Li; Yan Yan; Lingjuan Lyu
>
> **备注:** 8 pages for the main paper
>
> **摘要:** Fine-grained multimodal capability in Multimodal Large Language Models (MLLMs) has emerged as a critical research direction, particularly for tackling the visual grounding (VG) problem. Despite the strong performance achieved by existing approaches, they often employ disparate design choices when fine-tuning MLLMs for VG, lacking systematic verification to support these designs. To bridge this gap, this paper presents a comprehensive study of various design choices that impact the VG performance of MLLMs. We conduct our analysis using LLaVA-1.5, which has been widely adopted in prior empirical studies of MLLMs. While more recent models exist, we follow this convention to ensure our findings remain broadly applicable and extendable to other architectures. We cover two key aspects: (1) exploring different visual grounding paradigms in MLLMs, identifying the most effective design, and providing our insights; and (2) conducting ablation studies on the design of grounding data to optimize MLLMs' fine-tuning for the VG task. Finally, our findings contribute to a stronger MLLM for VG, achieving improvements of +5.6% / +6.9% / +7.0% on RefCOCO/+/g over the LLaVA-1.5.
>
---
#### [replaced 053] CoMatcher: Multi-View Collaborative Feature Matching
- **分类: cs.CV; I.4.8; I.2.10; I.5.4**

- **链接: [http://arxiv.org/pdf/2504.01872v2](http://arxiv.org/pdf/2504.01872v2)**

> **作者:** Jintao Zhang; Zimin Xia; Mingyue Dong; Shuhan Shen; Linwei Yue; Xianwei Zheng
>
> **备注:** 15 pages, 7 figures, to be published in CVPR 2025
>
> **摘要:** This paper proposes a multi-view collaborative matching strategy for reliable track construction in complex scenarios. We observe that the pairwise matching paradigms applied to image set matching often result in ambiguous estimation when the selected independent pairs exhibit significant occlusions or extreme viewpoint changes. This challenge primarily stems from the inherent uncertainty in interpreting intricate 3D structures based on limited two-view observations, as the 3D-to-2D projection leads to significant information loss. To address this, we introduce CoMatcher, a deep multi-view matcher to (i) leverage complementary context cues from different views to form a holistic 3D scene understanding and (ii) utilize cross-view projection consistency to infer a reliable global solution. Building on CoMatcher, we develop a groupwise framework that fully exploits cross-view relationships for large-scale matching tasks. Extensive experiments on various complex scenarios demonstrate the superiority of our method over the mainstream two-view matching paradigm.
>
---
#### [replaced 054] DiffIER: Optimizing Diffusion Models with Iterative Error Reduction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13628v2](http://arxiv.org/pdf/2508.13628v2)**

> **作者:** Ao Chen; Lihe Ding; Tianfan Xue
>
> **摘要:** Diffusion models have demonstrated remarkable capabilities in generating high-quality samples and enhancing performance across diverse domains through Classifier-Free Guidance (CFG). However, the quality of generated samples is highly sensitive to the selection of the guidance weight. In this work, we identify a critical ``training-inference gap'' and we argue that it is the presence of this gap that undermines the performance of conditional generation and renders outputs highly sensitive to the guidance weight. We quantify this gap by measuring the accumulated error during the inference stage and establish a correlation between the selection of guidance weight and minimizing this gap. Furthermore, to mitigate this gap, we propose DiffIER, an optimization-based method for high-quality generation. We demonstrate that the accumulated error can be effectively reduced by an iterative error minimization at each step during inference. By introducing this novel plug-and-play optimization framework, we enable the optimization of errors at every single inference step and enhance generation quality. Empirical results demonstrate that our proposed method outperforms baseline approaches in conditional generation tasks. Furthermore, the method achieves consistent success in text-to-image generation, image super-resolution, and text-to-speech generation, underscoring its versatility and potential for broad applications in future research.
>
---
#### [replaced 055] BadBlocks: Low-Cost and Stealthy Backdoor Attacks Tailored for Text-to-Image Diffusion Models
- **分类: cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.03221v3](http://arxiv.org/pdf/2508.03221v3)**

> **作者:** Yu Pan; Jiahao Chen; Lin Wang; Bingrong Dai; Yi Du
>
> **摘要:** In recent years, Diffusion models have achieved remarkable progress in the field of image generation. However, recent studies have shown that diffusion models are susceptible to backdoor attacks, in which attackers can manipulate the output by injecting covert triggers such as specific visual patterns or textual phrases into the training dataset. Fortunately, with the continuous advancement of defense techniques, defenders have become increasingly capable of identifying and mitigating most backdoor attacks using visual inspection and neural network-based detection methods. However, in this paper, we identify a novel type of backdoor threat that is more lightweight and covert than existing approaches, which we name BadBlocks, requires only about 30% of the computational resources and 20% GPU time typically needed by previous backdoor attacks, yet it successfully injects backdoors and evades the most advanced defense frameworks. BadBlocks enables attackers to selectively contaminate specific blocks within the UNet architecture of diffusion models while maintaining normal functionality in the remaining components. Experimental results demonstrate that BadBlocks achieves a high attack success rate and low perceptual quality loss , even under extremely constrained computational resources and GPU time. Moreover, BadBlocks is able to bypass existing defense frameworks, especially the attention-based backdoor detection method, highlighting it as a novel and noteworthy threat. Ablation studies further demonstrate that effective backdoor injection does not require fine-tuning the entire network and highlight the pivotal role of certain neural network layers in backdoor mapping. Overall, BadBlocks significantly reduces the barrier to conducting backdoor attacks in all aspects. It enables attackers to inject backdoors into large-scale diffusion models even using consumer-grade GPUs.
>
---
#### [replaced 056] WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.05599v2](http://arxiv.org/pdf/2508.05599v2)**

> **作者:** Shaobin Zhuang; Yiwei Guo; Canmiao Fu; Zhipeng Huang; Zeyue Tian; Fangyikang Wang; Ying Zhang; Chen Li; Yali Wang
>
> **备注:** 23 pages, 10 figures, 37 tables
>
> **摘要:** Visual tokenizer is a critical component for vision generation. However, the existing tokenizers often face unsatisfactory trade-off between compression ratios and reconstruction fidelity. To fill this gap, we introduce a powerful and concise WeTok tokenizer, which surpasses the previous leading tokenizers via two core innovations. (1) Group-wise lookup-free Quantization (GQ). We partition the latent features into groups, and perform lookup-free quantization for each group. As a result, GQ can efficiently overcome memory and computation limitations of prior tokenizers, while achieving a reconstruction breakthrough with more scalable codebooks. (2) Generative Decoding (GD). Different from prior tokenizers, we introduce a generative decoder with a prior of extra noise variable. In this case, GD can probabilistically model the distribution of visual data conditioned on discrete tokens, allowing WeTok to reconstruct visual details, especially at high compression ratios. Extensive experiments on mainstream benchmarks show superior performance of our WeTok. On the ImageNet 50k validation set, WeTok achieves a record-low zero-shot rFID (WeTok: 0.12 vs. FLUX-VAE: 0.18 vs. SD-VAE 3.5: 0.19) with a 400% compression ratio. Furthermore, our highest compression model achieves a zero-shot rFID of 3.49 with a compression ratio of 768, outperforming Cosmos (384) 4.57 which has only 50% compression rate of ours. Code and models are available: https://github.com/zhuangshaobin/WeTok.
>
---
#### [replaced 057] VisioPhysioENet: Visual Physiological Engagement Detection Network
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.16126v3](http://arxiv.org/pdf/2409.16126v3)**

> **作者:** Alakhsimar Singh; Kanav Goyal; Nischay Verma; Puneet Kumar; Xiaobai Li; Amritpal Singh
>
> **备注:** 35 Pages, 4 figures, 5 Tables
>
> **摘要:** This paper presents VisioPhysioENet, a novel multimodal system that leverages visual and physiological signals to detect learner engagement. It employs a two-level approach for extracting both visual and physiological features. For visual feature extraction, Dlib is used to detect facial landmarks, while OpenCV provides additional estimations. The face recognition library, built on Dlib, is used to identify the facial region of interest specifically for physiological signal extraction. Physiological signals are then extracted using the plane-orthogonal-toskin method to assess cardiovascular activity. These features are integrated using advanced machine learning classifiers, enhancing the detection of various levels of engagement. We thoroughly tested VisioPhysioENet on the DAiSEE dataset. It achieved an accuracy of 63.09%. This shows it can better identify different levels of engagement compared to many existing methods. It performed 8.6% better than the only other model that uses both physiological and visual features.
>
---
#### [replaced 058] Efficient Long-duration Talking Video Synthesis with Linear Diffusion Transformer under Multimodal Guidance
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16748v3](http://arxiv.org/pdf/2411.16748v3)**

> **作者:** Haojie Zhang; Zhihao Liang; Ruibo Fu; Bingyan Liu; Zhengqi Wen; Xuefei Liu; Jianhua Tao; Yaling Liang
>
> **备注:** 13 pages, 11 figures
>
> **摘要:** Long-duration talking video synthesis faces persistent challenges in simultaneously achieving high video quality, portrait and temporal consistency, and computational efficiency. As video length increases, issues such as visual degradation, loss of identity consistency, temporal incoherence, and error accumulation become increasingly prominent, severely impacting the realism and reliability of generated results. To address these issues, we present LetsTalk, a diffusion transformer framework that incorporates multimodal guidance and a novel memory bank mechanism, explicitly maintaining contextual continuity and enabling robust, high-quality, and efficient long-duration talking video generation. Specifically, LetsTalk introduces a memory bank combined with a noise-regularized training strategy to mitigate error accumulation and sampling artifacts during long video generation. To further enhance efficiency and spatiotemporal consistency, LetsTalk employs a deep compression autoencoder and a spatiotemporal-aware transformer with linear attention for effective multimodal fusion. Furthermore, we systematically analyze three multimodal fusion schemes, adopting deep (Symbiotic Fusion) for portrait features to ensure visual consistency, and shallow (Direct Fusion) for audio to synchronize animation with speech while preserving motion diversity. Extensive experiments demonstrate that LetsTalk achieves state-of-the-art generation quality, producing temporally coherent and realistic talking videos with enhanced diversity and liveliness, while maintaining remarkable efficiency with 8 fewer parameters than previous approaches.
>
---
#### [replaced 059] 2D Gaussians Meet Visual Tokenizer
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13515v2](http://arxiv.org/pdf/2508.13515v2)**

> **作者:** Yiang Shi; Xiaoyang Guo; Wei Yin; Mingkai Jia; Qian Zhang; Xiaolin Hu; Wenyu Liu; Xinggang Wang
>
> **摘要:** The image tokenizer is a critical component in AR image generation, as it determines how rich and structured visual content is encoded into compact representations. Existing quantization-based tokenizers such as VQ-GAN primarily focus on appearance features like texture and color, often neglecting geometric structures due to their patch-based design. In this work, we explored how to incorporate more visual information into the tokenizer and proposed a new framework named Visual Gaussian Quantization (VGQ), a novel tokenizer paradigm that explicitly enhances structural modeling by integrating 2D Gaussians into traditional visual codebook quantization frameworks. Our approach addresses the inherent limitations of naive quantization methods such as VQ-GAN, which struggle to model structured visual information due to their patch-based design and emphasis on texture and color. In contrast, VGQ encodes image latents as 2D Gaussian distributions, effectively capturing geometric and spatial structures by directly modeling structure-related parameters such as position, rotation and scale. We further demonstrate that increasing the density of 2D Gaussians within the tokens leads to significant gains in reconstruction fidelity, providing a flexible trade-off between token efficiency and visual richness. On the ImageNet 256x256 benchmark, VGQ achieves strong reconstruction quality with an rFID score of 1.00. Furthermore, by increasing the density of 2D Gaussians within the tokens, VGQ gains a significant boost in reconstruction capability and achieves a state-of-the-art reconstruction rFID score of 0.556 and a PSNR of 24.93, substantially outperforming existing methods. Codes will be released soon.
>
---
#### [replaced 060] Marrying Autoregressive Transformer and Diffusion with Multi-Reference Autoregression
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09482v3](http://arxiv.org/pdf/2506.09482v3)**

> **作者:** Dingcheng Zhen; Qian Qiao; Xu Zheng; Tan Yu; Kangxi Wu; Ziwei Zhang; Siyuan Liu; Shunshun Yin; Ming Tao
>
> **摘要:** We introduce TransDiff, the first image generation model that marries Autoregressive (AR) Transformer with diffusion models. In this joint modeling framework, TransDiff encodes labels and images into high-level semantic features and employs a diffusion model to estimate the distribution of image samples. On the ImageNet 256x256 benchmark, TransDiff significantly outperforms other image generation models based on standalone AR Transformer or diffusion models. Specifically, TransDiff achieves a Frechet Inception Distance (FID) of 1.61 and an Inception Score (IS) of 293.4, and further provides x2 faster inference latency compared to state-of-the-art methods based on AR Transformer and x112 faster inference compared to diffusion-only models. Furthermore, building on the TransDiff model, we introduce a novel image generation paradigm called Multi-Reference Autoregression (MRAR), which performs autoregressive generation by predicting the next image. MRAR enables the model to reference multiple previously generated images, thereby facilitating the learning of more diverse representations and improving the quality of generated images in subsequent iterations. By applying MRAR, the performance of TransDiff is improved, with the FID reduced from 1.61 to 1.42. We expect TransDiff to open up a new frontier in the field of image generation.
>
---
