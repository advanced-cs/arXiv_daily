# 计算机视觉 cs.CV

- **最新发布 94 篇**

- **更新 68 篇**

## 最新发布

#### [new 001] Conscious Gaze: Adaptive Attention Mechanisms for Hallucination Mitigation in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉-语言模型中的物体幻觉问题，提出无需训练的推理框架CG-VLM。通过博弈解释机制感知视觉与文本协同状态，在关键时刻引导注意力聚焦视觉信息，抑制语言先验导致的幻觉，提升生成准确性。**

- **链接: [https://arxiv.org/pdf/2512.05546v1](https://arxiv.org/pdf/2512.05546v1)**

> **作者:** Weijue Bu; Guan Yuan; Guixian Zhang
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** Large Vision-Language Models (VLMs) often exhibit text inertia, where attention drifts from visual evidence toward linguistic priors, resulting in object hallucinations. Existing decoding strategies intervene only at the output logits and thus cannot correct internal reasoning drift, while recent internal-control methods based on heuristic head suppression or global steering vectors lack principled grounding. We introduce Conscious Gaze (CG-VLM), a training-free, inference-time framework that converts game-theoretic interpretability into actionable decoding control. A Cognitive Demand Sensor built on Harsanyi interactions estimates instantaneous vision-text synergy and identifies moments when visual grounding is necessary. Conditioned on this signal, a Focused Consensus Induction module selectively reorients mid-layer attention toward visual tokens before collapse into text priors. CG-VLM achieves state-of-the-art results on POPE and CHAIR across InstructBLIP, LLaVA, Qwen-VL, and mPLUG, while preserving general capabilities, demonstrating that token-level sensing enables precise, context-aware intervention without compromising foundational knowledge.
>
---
#### [new 002] See in Depth: Training-Free Surgical Scene Segmentation with Monocular Depth Priors
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究手术场景的像素级分割任务，旨在降低对密集标注的依赖。提出DepSeg框架，利用单目深度估计作为几何先验，指导点提示生成并结合SAM2与模板匹配实现无需训练的分割，在少样本模板下仍表现良好。**

- **链接: [https://arxiv.org/pdf/2512.05529v1](https://arxiv.org/pdf/2512.05529v1)**

> **作者:** Kunyi Yang; Qingyu Wang; Cheng Yuan; Yutong Ban
>
> **备注:** The first two authors contributed equally
>
> **摘要:** Pixel-wise segmentation of laparoscopic scenes is essential for computer-assisted surgery but difficult to scale due to the high cost of dense annotations. We propose depth-guided surgical scene segmentation (DepSeg), a training-free framework that utilizes monocular depth as a geometric prior together with pretrained vision foundation models. DepSeg first estimates a relative depth map with a pretrained monocular depth estimation network and proposes depth-guided point prompts, which SAM2 converts into class-agnostic masks. Each mask is then described by a pooled pretrained visual feature and classified via template matching against a template bank built from annotated frames. On the CholecSeg8k dataset, DepSeg improves over a direct SAM2 auto segmentation baseline (35.9% vs. 14.7% mIoU) and maintains competitive performance even when using only 10--20% of the object templates. These results show that depth-guided prompting and template-based classification offer an annotation-efficient segmentation approach.
>
---
#### [new 003] Manifold-Aware Point Cloud Completion via Geodesic-Attentive Hierarchical Feature Learning
- **分类: cs.CV**

- **简介: 该论文研究点云补全任务，旨在解决现有方法忽略点云内在非线性几何结构导致的重建不准确问题。提出一种流形感知框架，通过测地距离估计和测地关系注意力机制，增强特征学习的几何一致性和语义连贯性。**

- **链接: [https://arxiv.org/pdf/2512.05710v1](https://arxiv.org/pdf/2512.05710v1)**

> **作者:** Jianan Sun; Dongzhihan Wang; Mingyu Fan
>
> **摘要:** Point cloud completion seeks to recover geometrically consistent shapes from partial or sparse 3D observations. Although recent methods have achieved reasonable global shape reconstruction, they often rely on Euclidean proximity and overlook the intrinsic nonlinear geometric structure of point clouds, resulting in suboptimal geometric consistency and semantic ambiguity. In this paper, we present a manifold-aware point cloud completion framework that explicitly incorporates nonlinear geometry information throughout the feature learning pipeline. Our approach introduces two key modules: a Geodesic Distance Approximator (GDA), which estimates geodesic distances between points to capture the latent manifold topology, and a Manifold-Aware Feature Extractor (MAFE), which utilizes geodesic-based $k$-NN groupings and a geodesic-relational attention mechanism to guide the hierarchical feature extraction process. By integrating geodesic-aware relational attention, our method promotes semantic coherence and structural fidelity in the reconstructed point clouds. Extensive experiments on benchmark datasets demonstrate that our approach consistently outperforms state-of-the-art methods in reconstruction quality.
>
---
#### [new 004] Delving into Latent Spectral Biasing of Video VAEs for Superior Diffusability
- **分类: cs.CV**

- **简介: 该论文研究视频生成任务中VAE隐空间结构对扩散模型训练的影响，提出两种正则化方法改善隐空间频谱特性，提升扩散效率。所提SSVAE显著加快收敛并提高生成质量。**

- **链接: [https://arxiv.org/pdf/2512.05394v1](https://arxiv.org/pdf/2512.05394v1)**

> **作者:** Shizhan Liu; Xinran Deng; Zhuoyi Yang; Jiayan Teng; Xiaotao Gu; Jie Tang
>
> **摘要:** Latent diffusion models pair VAEs with diffusion backbones, and the structure of VAE latents strongly influences the difficulty of diffusion training. However, existing video VAEs typically focus on reconstruction fidelity, overlooking latent structure. We present a statistical analysis of video VAE latent spaces and identify two spectral properties essential for diffusion training: a spatio-temporal frequency spectrum biased toward low frequencies, and a channel-wise eigenspectrum dominated by a few modes. To induce these properties, we propose two lightweight, backbone-agnostic regularizers: Local Correlation Regularization and Latent Masked Reconstruction. Experiments show that our Spectral-Structured VAE (SSVAE) achieves a $3\times$ speedup in text-to-video generation convergence and a 10\% gain in video reward, outperforming strong open-source VAEs. The code is available at https://github.com/zai-org/SSVAE.
>
---
#### [new 005] LeAD-M3D: Leveraging Asymmetric Distillation for Real-time Monocular 3D Detection
- **分类: cs.CV**

- **简介: 该论文研究单目3D目标检测，旨在解决深度模糊与效率难题。作者提出LeAD-M3D方法，通过非对称蒸馏、3D匹配优化和置信门控推理，在无LiDAR情况下实现高精度与实时性，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.05663v1](https://arxiv.org/pdf/2512.05663v1)**

> **作者:** Johannes Meier; Jonathan Michel; Oussema Dhaouadi; Yung-Hsu Yang; Christoph Reich; Zuria Bauer; Stefan Roth; Marc Pollefeys; Jacques Kaiser; Daniel Cremers
>
> **摘要:** Real-time monocular 3D object detection remains challenging due to severe depth ambiguity, viewpoint shifts, and the high computational cost of 3D reasoning. Existing approaches either rely on LiDAR or geometric priors to compensate for missing depth, or sacrifice efficiency to achieve competitive accuracy. We introduce LeAD-M3D, a monocular 3D detector that achieves state-of-the-art accuracy and real-time inference without extra modalities. Our method is powered by three key components. Asymmetric Augmentation Denoising Distillation (A2D2) transfers geometric knowledge from a clean-image teacher to a mixup-noised student via a quality- and importance-weighted depth-feature loss, enabling stronger depth reasoning without LiDAR supervision. 3D-aware Consistent Matching (CM3D) improves prediction-to-ground truth assignment by integrating 3D MGIoU into the matching score, yielding more stable and precise supervision. Finally, Confidence-Gated 3D Inference (CGI3D) accelerates detection by restricting expensive 3D regression to top-confidence regions. Together, these components set a new Pareto frontier for monocular 3D detection: LeAD-M3D achieves state-of-the-art accuracy on KITTI and Waymo, and the best reported car AP on Rope3D, while running up to 3.6x faster than prior high-accuracy methods. Our results demonstrate that high fidelity and real-time efficiency in monocular 3D detection are simultaneously attainable - without LiDAR, stereo, or geometric assumptions.
>
---
#### [new 006] NICE: Neural Implicit Craniofacial Model for Orthognathic Surgery Prediction
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出NICE，一种基于神经隐式表示的颅面建模方法，用于正颌手术预测。针对现有模型在精度与效率上的不足，其通过区域特异性隐式SDF与形变解码器，结合共享手术隐编码，精确建模骨骼运动对软组织的非线性影响，提升术后面部外观预测准确性。**

- **链接: [https://arxiv.org/pdf/2512.05920v1](https://arxiv.org/pdf/2512.05920v1)**

> **作者:** Jiawen Yang; Yihui Cao; Xuanyu Tian; Yuyao Zhang; Hongjiang Wei
>
> **摘要:** Orthognathic surgery is a crucial intervention for correcting dentofacial skeletal deformities to enhance occlusal functionality and facial aesthetics. Accurate postoperative facial appearance prediction remains challenging due to the complex nonlinear interactions between skeletal movements and facial soft tissue. Existing biomechanical, parametric models and deep-learning approaches either lack computational efficiency or fail to fully capture these intricate interactions. To address these limitations, we propose Neural Implicit Craniofacial Model (NICE) which employs implicit neural representations for accurate anatomical reconstruction and surgical outcome prediction. NICE comprises a shape module, which employs region-specific implicit Signed Distance Function (SDF) decoders to reconstruct the facial surface, maxilla, and mandible, and a surgery module, which employs region-specific deformation decoders. These deformation decoders are driven by a shared surgical latent code to effectively model the complex, nonlinear biomechanical response of the facial surface to skeletal movements, incorporating anatomical prior knowledge. The deformation decoders output point-wise displacement fields, enabling precise modeling of surgical outcomes. Extensive experiments demonstrate that NICE outperforms current state-of-the-art methods, notably improving prediction accuracy in critical facial regions such as lips and chin, while robustly preserving anatomical integrity. This work provides a clinically viable tool for enhanced surgical planning and patient consultation in orthognathic procedures.
>
---
#### [new 007] USV: Unified Sparsification for Accelerating Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对视频扩散模型推理效率低的问题，提出统一稀疏化框架USV，协同优化模型内部计算与采样过程，通过动态稀疏策略实现显著加速，同时保持生成质量。**

- **链接: [https://arxiv.org/pdf/2512.05754v1](https://arxiv.org/pdf/2512.05754v1)**

> **作者:** Xinjian Wu; Hongmei Wang; Yuan Zhou; Qinglin Lu
>
> **摘要:** The scalability of high-fidelity video diffusion models (VDMs) is constrained by two key sources of redundancy: the quadratic complexity of global spatio-temporal attention and the computational overhead of long iterative denoising trajectories. Existing accelerators -- such as sparse attention and step-distilled samplers -- typically target a single dimension in isolation and quickly encounter diminishing returns, as the remaining bottlenecks become dominant. In this work, we introduce USV (Unified Sparsification for Video diffusion models), an end-to-end trainable framework that overcomes this limitation by jointly orchestrating sparsification across both the model's internal computation and its sampling process. USV learns a dynamic, data- and timestep-dependent sparsification policy that prunes redundant attention connections, adaptively merges semantically similar tokens, and reduces denoising steps, treating them not as independent tricks but as coordinated actions within a single optimization objective. This multi-dimensional co-design enables strong mutual reinforcement among previously disjoint acceleration strategies. Extensive experiments on large-scale video generation benchmarks demonstrate that USV achieves up to 83.3% speedup in the denoising process and 22.7% end-to-end acceleration, while maintaining high visual fidelity. Our results highlight unified, dynamic sparsification as a practical path toward efficient, high-quality video generation.
>
---
#### [new 008] SplatPainter: Interactive Authoring of 3D Gaussians from 2D Edits via Test-Time Training
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D内容编辑任务，旨在解决3D高斯泼溅模型难以交互式精细编辑的问题。作者提出SplatPainter，通过测试时训练实现基于2D视图输入的快速、非破坏性、状态感知的3D高斯属性更新，支持局部细节优化与全局重着色等操作。**

- **链接: [https://arxiv.org/pdf/2512.05354v1](https://arxiv.org/pdf/2512.05354v1)**

> **作者:** Yang Zheng; Hao Tan; Kai Zhang; Peng Wang; Leonidas Guibas; Gordon Wetzstein; Wang Yifan
>
> **备注:** project page https://y-zheng18.github.io/SplatPainter/
>
> **摘要:** The rise of 3D Gaussian Splatting has revolutionized photorealistic 3D asset creation, yet a critical gap remains for their interactive refinement and editing. Existing approaches based on diffusion or optimization are ill-suited for this task, as they are often prohibitively slow, destructive to the original asset's identity, or lack the precision for fine-grained control. To address this, we introduce \ourmethod, a state-aware feedforward model that enables continuous editing of 3D Gaussian assets from user-provided 2D view(s). Our method directly predicts updates to the attributes of a compact, feature-rich Gaussian representation and leverages Test-Time Training to create a state-aware, iterative workflow. The versatility of our approach allows a single architecture to perform diverse tasks, including high-fidelity local detail refinement, local paint-over, and consistent global recoloring, all at interactive speeds, paving the way for fluid and intuitive 3D content authoring.
>
---
#### [new 009] Learning High-Fidelity Cloth Animation via Skinning-Free Image Transfer
- **分类: cs.CV**

- **简介: 该论文研究3D服装动画生成，旨在解决传统基于蒙皮的方法因缺乏监督导致形变错位、细节失真的问题。提出无需蒙皮的双频解耦方法，分别估计顶点位置与法线，并通过图像迁移和多模态融合提升褶皱细节与动画质量。**

- **链接: [https://arxiv.org/pdf/2512.05593v1](https://arxiv.org/pdf/2512.05593v1)**

> **作者:** Rong Wang; Wei Mao; Changsheng Lu; Hongdong Li
>
> **备注:** Accepted to 3DV 2026
>
> **摘要:** We present a novel method for generating 3D garment deformations from given body poses, which is key to a wide range of applications, including virtual try-on and extended reality. To simplify the cloth dynamics, existing methods mostly rely on linear blend skinning to obtain low-frequency posed garment shape and only regress high-frequency wrinkles. However, due to the lack of explicit skinning supervision, such skinning-based approach often produces misaligned shapes when posing the garment, consequently corrupts the high-frequency signals and fails to recover high-fidelity wrinkles. To tackle this issue, we propose a skinning-free approach by independently estimating posed (i) vertex position for low-frequency posed garment shape, and (ii) vertex normal for high-frequency local wrinkle details. In this way, each frequency modality can be effectively decoupled and directly supervised by the geometry of the deformed garment. To further improve the visual quality of animation, we propose to encode both vertex attributes as rendered texture images, so that 3D garment deformation can be equivalently achieved via 2D image transfer. This enables us to leverage powerful pretrained image models to recover fine-grained visual details in wrinkles, while maintaining superior scalability for garments of diverse topologies without relying on manual UV partition. Finally, we propose a multimodal fusion to incorporate constraints from both frequency modalities and robustly recover deformed 3D garments from transferred images. Extensive experiments show that our method significantly improves animation quality on various garment types and recovers finer wrinkles than state-of-the-art methods.
>
---
#### [new 010] Age-Inclusive 3D Human Mesh Recovery for Action-Preserving Data Anonymization
- **分类: cs.CV**

- **简介: 该论文属于3D人体重建任务，旨在解决现有方法在儿童和婴儿上泛化差的问题。提出AionHMR框架，结合SMPL-A模型生成伪标签，训练专用Transformer模型，实现成人与儿童兼顾的实时3D重建，并用于动作保留的数据匿名化。**

- **链接: [https://arxiv.org/pdf/2512.05259v1](https://arxiv.org/pdf/2512.05259v1)**

> **作者:** Georgios Chatzichristodoulou; Niki Efthymiou; Panagiotis Filntisis; Georgios Pavlakos; Petros Maragos
>
> **摘要:** While three-dimensional (3D) shape and pose estimation is a highly researched area that has yielded significant advances, the resulting methods, despite performing well for the adult population, generally fail to generalize effectively to children and infants. This paper addresses this challenge by introducing AionHMR, a comprehensive framework designed to bridge this domain gap. We propose an optimization-based method that extends a top-performing model by incorporating the SMPL-A body model, enabling the concurrent and accurate modeling of adults, children, and infants. Leveraging this approach, we generated pseudo-ground-truth annotations for publicly available child and infant image databases. Using these new training data, we then developed and trained a specialized transformer-based deep learning model capable of real-time 3D age-inclusive human reconstruction. Extensive experiments demonstrate that our methods significantly improve shape and pose estimation for children and infants without compromising accuracy on adults. Importantly, our reconstructed meshes serve as privacy-preserving substitutes for raw images, retaining essential action, pose, and geometry information while enabling anonymized datasets release. As a demonstration, we introduce the 3D-BabyRobot dataset, a collection of action-preserving 3D reconstructions of children interacting with robots. This work bridges a crucial domain gap and establishes a foundation for inclusive, privacy-aware, and age-diverse 3D human modeling.
>
---
#### [new 011] ChromouVQA: Benchmarking Vision-Language Models under Chromatic Camouflaged Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ChromouVQA，一种基于色觉伪装图像的视觉-语言模型评测基准，旨在解决复杂背景下目标识别与图文理解难题。通过多任务设计评估模型在色彩混淆、几何干扰下的表现，并提出对比学习方法提升形状恢复能力，推动可复现的多模态研究。**

- **链接: [https://arxiv.org/pdf/2512.05137v1](https://arxiv.org/pdf/2512.05137v1)**

> **作者:** Yunfei Zhang; Yizhuo He; Yuanxun Shao; Zhengtao Yao; Haoyan Xu; Junhao Dong; Zhen Yao; Zhikang Dong
>
> **摘要:** Vision-Language Models (VLMs) have advanced multimodal understanding, yet still struggle when targets are embedded in cluttered backgrounds requiring figure-ground segregation. To address this, we introduce ChromouVQA, a large-scale, multi-task benchmark based on Ishihara-style chromatic camouflaged images. We extend classic dot plates with multiple fill geometries and vary chromatic separation, density, size, occlusion, and rotation, recording full metadata for reproducibility. The benchmark covers nine vision-question-answering tasks, including recognition, counting, comparison, and spatial reasoning. Evaluations of humans and VLMs reveal large gaps, especially under subtle chromatic contrast or disruptive geometric fills. We also propose a model-agnostic contrastive recipe aligning silhouettes with their camouflaged renderings, improving recovery of global shapes. ChromouVQA provides a compact, controlled benchmark for reproducible evaluation and extension. Code and dataset are available at https://github.com/Chromou-VQA-Benchmark/Chromou-VQA.
>
---
#### [new 012] Label-Efficient Point Cloud Segmentation with Active Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究3D点云语义分割中的标签高效标注问题，提出一种基于2D网格划分和模型集成不确定性的主动学习方法，有效减少标注成本，在多个数据集上性能优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.05759v1](https://arxiv.org/pdf/2512.05759v1)**

> **作者:** Johannes Meyer; Jasper Hoffmann; Felix Schulz; Dominik Merkle; Daniel Buescher; Alexander Reiterer; Joschka Boedecker; Wolfram Burgard
>
> **摘要:** Semantic segmentation of 3D point cloud data often comes with high annotation costs. Active learning automates the process of selecting which data to annotate, reducing the total amount of annotation needed to achieve satisfactory performance. Recent approaches to active learning for 3D point clouds are often based on sophisticated heuristics for both, splitting point clouds into annotatable regions and selecting the most beneficial for further neural network training. In this work, we propose a novel and easy-to-implement strategy to separate the point cloud into annotatable regions. In our approach, we utilize a 2D grid to subdivide the point cloud into columns. To identify the next data to be annotated, we employ a network ensemble to estimate the uncertainty in the network output. We evaluate our method on the S3DIS dataset, the Toronto-3D dataset, and a large-scale urban 3D point cloud of the city of Freiburg, which we labeled in parts manually. The extensive evaluation shows that our method yields performance on par with, or even better than, complex state-of-the-art methods on all datasets. Furthermore, we provide results suggesting that in the context of point clouds the annotated area can be a more meaningful measure for active learning algorithms than the number of annotated points.
>
---
#### [new 013] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究零样本超分辨率时空预测，指出模型在低分辨率训练后难以处理高分辨率中的高频信号，导致误差被“尺度锚定”。为此提出频率表示学习，通过分辨率对齐的频域表征和谱一致性训练，使误差随分辨率提升而降低，提升高分辨率推理精度。**

- **链接: [https://arxiv.org/pdf/2512.05132v1](https://arxiv.org/pdf/2512.05132v1)**

> **作者:** Wenshuo Wang; Fan Zhang
>
> **摘要:** Zero-Shot Super-Resolution Spatiotemporal Forecasting requires a deep learning model to be trained on low-resolution data and deployed for inference on high-resolution. Existing studies consider maintaining similar error across different resolutions as indicative of successful multi-resolution generalization. However, deep learning models serving as alternatives to numerical solvers should reduce error as resolution increases. The fundamental limitation is, the upper bound of physical law frequencies that low-resolution data can represent is constrained by its Nyquist frequency, making it difficult for models to process signals containing unseen frequency components during high-resolution inference. This results in errors being anchored at low resolution, incorrectly interpreted as successful generalization. We define this fundamental phenomenon as a new problem distinct from existing issues: Scale Anchoring. Therefore, we propose architecture-agnostic Frequency Representation Learning. It alleviates Scale Anchoring through resolution-aligned frequency representations and spectral consistency training: on grids with higher Nyquist frequencies, the frequency response in high-frequency bands of FRL-enhanced variants is more stable. This allows errors to decrease with resolution and significantly outperform baselines within our task and resolution range, while incurring only modest computational overhead.
>
---
#### [new 014] Concept-based Explainable Data Mining with VLM for 3D Detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决自动驾驶中稀有物体检测难的问题。作者提出一种基于视觉语言模型和概念挖掘的可解释数据筛选方法，通过2D语义信息指导3D稀有物体识别，提升检测性能并减少标注成本。**

- **链接: [https://arxiv.org/pdf/2512.05482v1](https://arxiv.org/pdf/2512.05482v1)**

> **作者:** Mai Tsujimoto
>
> **备注:** 28 pages including appendix. Code: https://github.com/mm1129/concept_based_rare_detector_2025
>
> **摘要:** Rare-object detection remains a challenging task in autonomous driving systems, particularly when relying solely on point cloud data. Although Vision-Language Models (VLMs) exhibit strong capabilities in image understanding, their potential to enhance 3D object detection through intelligent data mining has not been fully explored. This paper proposes a novel cross-modal framework that leverages 2D VLMs to identify and mine rare objects from driving scenes, thereby improving 3D object detection performance. Our approach synthesizes complementary techniques such as object detection, semantic feature extraction, dimensionality reduction, and multi-faceted outlier detection into a cohesive, explainable pipeline that systematically identifies rare but critical objects in driving scenes. By combining Isolation Forest and t-SNE-based outlier detection methods with concept-based filtering, the framework effectively identifies semantically meaningful rare objects. A key strength of this approach lies in its ability to extract and annotate targeted rare object concepts such as construction vehicles, motorcycles, and barriers. This substantially reduces the annotation burden and focuses only on the most valuable training samples. Experiments on the nuScenes dataset demonstrate that this concept-guided data mining strategy enhances the performance of 3D object detection models while utilizing only a fraction of the training data, with particularly notable improvements for challenging object categories such as trailers and bicycles compared with the same amount of random data. This finding has substantial implications for the efficient curation of datasets in safety-critical autonomous systems.
>
---
#### [new 015] YOLO and SGBM Integration for Autonomous Tree Branch Detection and Depth Estimation in Radiata Pine Pruning Applications
- **分类: cs.CV**

- **简介: 该论文研究无人机自主修剪中的树枝检测与深度估计任务，旨在解决人工修剪高危问题。提出YOLO与SGBM融合方法，仅用双目相机实现精确分支识别与定位，替代昂贵LiDAR，提升林业作业安全与效率。**

- **链接: [https://arxiv.org/pdf/2512.05412v1](https://arxiv.org/pdf/2512.05412v1)**

> **作者:** Yida Lin; Bing Xue; Mengjie Zhang; Sam Schofield; Richard Green
>
> **摘要:** Manual pruning of radiata pine trees poses significant safety risks due to extreme working heights and challenging terrain. This paper presents a computer vision framework that integrates YOLO object detection with Semi-Global Block Matching (SGBM) stereo vision for autonomous drone-based pruning operations. Our system achieves precise branch detection and depth estimation using only stereo camera input, eliminating the need for expensive LiDAR sensors. Experimental evaluation demonstrates YOLO's superior performance over Mask R-CNN, achieving 82.0% mAPmask50-95 for branch segmentation. The integrated system accurately localizes branches within a 2 m operational range, with processing times under one second per frame. These results establish the feasibility of cost-effective autonomous pruning systems that enhance worker safety and operational efficiency in commercial forestry.
>
---
#### [new 016] Semore: VLM-guided Enhanced Semantic Motion Representations for Visual Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉强化学习任务，旨在解决现有LLM引导方法中表征能力不足的问题。作者提出Semore框架，利用VLM提取语义与运动双路径表征，并通过特征级对齐和分离监督实现高效融合，提升策略决策的适应性与性能。**

- **链接: [https://arxiv.org/pdf/2512.05172v1](https://arxiv.org/pdf/2512.05172v1)**

> **作者:** Wentao Wang; Chunyang Liu; Kehua Sheng; Bo Zhang; Yan Wang
>
> **摘要:** The growing exploration of Large Language Models (LLM) and Vision-Language Models (VLM) has opened avenues for enhancing the effectiveness of reinforcement learning (RL). However, existing LLM-based RL methods often focus on the guidance of control policy and encounter the challenge of limited representations of the backbone networks. To tackle this problem, we introduce Enhanced Semantic Motion Representations (Semore), a new VLM-based framework for visual RL, which can simultaneously extract semantic and motion representations through a dual-path backbone from the RGB flows. Semore utilizes VLM with common-sense knowledge to retrieve key information from observations, while using the pre-trained clip to achieve the text-image alignment, thereby embedding the ground-truth representations into the backbone. To efficiently fuse semantic and motion representations for decision-making, our method adopts a separately supervised approach to simultaneously guide the extraction of semantics and motion, while allowing them to interact spontaneously. Extensive experiments demonstrate that, under the guidance of VLM at the feature level, our method exhibits efficient and adaptive ability compared to state-of-art methods. All codes are released.
>
---
#### [new 017] TED-4DGS: Temporally Activated and Embedding-based Deformation for 4DGS Compression
- **分类: cs.CV**

- **简介: 该论文属于动态3D场景压缩任务，旨在解决4DGS表示中冗余高、压缩效率低的问题。提出TED-4DGS方法，结合时序激活与嵌入变形机制，并引入率失真优化压缩框架，实现高效紧凑的动态3D高斯溅射表示。**

- **链接: [https://arxiv.org/pdf/2512.05446v1](https://arxiv.org/pdf/2512.05446v1)**

> **作者:** Cheng-Yuan Ho; He-Bi Yang; Jui-Chiu Chiang; Yu-Lun Liu; Wen-Hsiao Peng
>
> **摘要:** Building on the success of 3D Gaussian Splatting (3DGS) in static 3D scene representation, its extension to dynamic scenes, commonly referred to as 4DGS or dynamic 3DGS, has attracted increasing attention. However, designing more compact and efficient deformation schemes together with rate-distortion-optimized compression strategies for dynamic 3DGS representations remains an underexplored area. Prior methods either rely on space-time 4DGS with overspecified, short-lived Gaussian primitives or on canonical 3DGS with deformation that lacks explicit temporal control. To address this, we present TED-4DGS, a temporally activated and embedding-based deformation scheme for rate-distortion-optimized 4DGS compression that unifies the strengths of both families. TED-4DGS is built on a sparse anchor-based 3DGS representation. Each canonical anchor is assigned learnable temporal-activation parameters to specify its appearance and disappearance transitions over time, while a lightweight per-anchor temporal embedding queries a shared deformation bank to produce anchor-specific deformation. For rate-distortion compression, we incorporate an implicit neural representation (INR)-based hyperprior to model anchor attribute distributions, along with a channel-wise autoregressive model to capture intra-anchor correlations. With these novel elements, our scheme achieves state-of-the-art rate-distortion performance on several real-world datasets. To the best of our knowledge, this work represents one of the first attempts to pursue a rate-distortion-optimized compression framework for dynamic 3DGS representations.
>
---
#### [new 018] PoolNet: Deep Learning for 2D to 3D Video Process Validation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出PoolNet，属于2D到3D视频处理任务，旨在解决野外图像数据因相机姿态不足、遮挡和噪声导致的SfM处理困难问题。作者构建了深度学习框架，实现对可处理场景的快速帧级与场景级验证，显著提升效率。**

- **链接: [https://arxiv.org/pdf/2512.05362v1](https://arxiv.org/pdf/2512.05362v1)**

> **作者:** Sanchit Kaul; Joseph Luna; Shray Arora
>
> **备注:** All code related to this paper can be found at https://github.com/sanchitkaul/PoolNet.git
>
> **摘要:** Lifting Structure-from-Motion (SfM) information from sequential and non-sequential image data is a time-consuming and computationally expensive task. In addition to this, the majority of publicly available data is unfit for processing due to inadequate camera pose variation, obscuring scene elements, and noisy data. To solve this problem, we introduce PoolNet, a versatile deep learning framework for frame-level and scene-level validation of in-the-wild data. We demonstrate that our model successfully differentiates SfM ready scenes from those unfit for processing while significantly undercutting the amount of time state of the art algorithms take to obtain structure-from-motion data.
>
---
#### [new 019] The Dynamic Prior: Understanding 3D Structures for Casual Dynamic Videos
- **分类: cs.CV**

- **简介: 该论文针对动态场景中3D结构理解难题，提出无需专门训练的Dynamic Prior方法，结合视觉语言模型与SAM2实现动态物体识别，提升相机位姿、深度与运动轨迹估计精度。**

- **链接: [https://arxiv.org/pdf/2512.05398v1](https://arxiv.org/pdf/2512.05398v1)**

> **作者:** Zhuoyuan Wu; Xurui Yang; Jiahui Huang; Yue Wang; Jun Gao
>
> **备注:** Code is available at https://github.com/wuzy2115/DYNAPO
>
> **摘要:** Estimating accurate camera poses, 3D scene geometry, and object motion from in-the-wild videos is a long-standing challenge for classical structure from motion pipelines due to the presence of dynamic objects. Recent learning-based methods attempt to overcome this challenge by training motion estimators to filter dynamic objects and focus on the static background. However, their performance is largely limited by the availability of large-scale motion segmentation datasets, resulting in inaccurate segmentation and, therefore, inferior structural 3D understanding. In this work, we introduce the Dynamic Prior (\ourmodel) to robustly identify dynamic objects without task-specific training, leveraging the powerful reasoning capabilities of Vision-Language Models (VLMs) and the fine-grained spatial segmentation capacity of SAM2. \ourmodel can be seamlessly integrated into state-of-the-art pipelines for camera pose optimization, depth reconstruction, and 4D trajectory estimation. Extensive experiments on both synthetic and real-world videos demonstrate that \ourmodel not only achieves state-of-the-art performance on motion segmentation, but also significantly improves accuracy and robustness for structural 3D understanding.
>
---
#### [new 020] ParaUni: Enhance Generation in Unified Multimodal Model with Reinforcement-driven Hierarchical Parallel Information Interaction
- **分类: cs.CV**

- **简介: 该论文聚焦统一多模态生成任务，旨在解决视觉语言模型与扩散模型间因表征差异导致的交互不足问题。提出ParaUni，通过并行融合VLM多层特征及设计分层动态调整机制，增强生成质量与强化学习中的多奖励优化。**

- **链接: [https://arxiv.org/pdf/2512.05422v1](https://arxiv.org/pdf/2512.05422v1)**

> **作者:** Jiangtong Tan; Lin Liu; Jie Huanng; Xiaopeng Zhang; Qi Tian; Feng Zhao
>
> **摘要:** Unified multimodal models significantly improve visual generation by combining vision-language models (VLMs) with diffusion models. However, existing methods struggle to fully balance sufficient interaction and flexible implementation due to vast representation difference. Considering abundant and hierarchical information in VLM's layers from low-level details to high-level semantics, we propose \textbf{ParaUni}. It extracts features from variants VLM's layers in a \textbf{Para}llel way for comprehensive information interaction and retains a flexible separation architecture to enhance generation in \textbf{Uni}fied multimodal model. Concretely, visual features from all VLM's layers are fed in parallel into a Layer Integration Module (LIM), which efficiently integrates fine-grained details and semantic abstractions and provides the fused representation as a condition to the diffusion model. To further enhance performance, we reveal that these hierarchical layers respond unequally to different rewards in Reinforcement Learning (RL). Crucially, we design a Layer-wise Dynamic Adjustment Mechanism (LDAM) to facilitate multiple reward improvements that aligns the hierarchical properties of these layers using RL. Extensive experiments show ParaUni leverages complementary multi-layer features to substantially improve generation quality and shows strong potential for multiple reward advances during RL stages. Code is available at https://github.com/JosephTiTan/ParaUni.
>
---
#### [new 021] Self-Improving VLM Judges Without Human Annotations
- **分类: cs.CV**

- **简介: 该论文研究无需人类标注的视觉语言模型（VLM）评判方法，旨在降低依赖昂贵且易过时的人类偏好数据。提出一种自迭代框架，通过自生成多模态数据、推理轨迹和判断来训练VLM裁判，在多个评测基准上显著提升性能，实现媲美甚至超越大模型的评判效果。**

- **链接: [https://arxiv.org/pdf/2512.05145v1](https://arxiv.org/pdf/2512.05145v1)**

> **作者:** Inna Wanyin Lin; Yushi Hu; Shuyue Stella Li; Scott Geng; Pang Wei Koh; Luke Zettlemoyer; Tim Althoff; Marjan Ghazvininejad
>
> **摘要:** Effective judges of Vision-Language Models (VLMs) are crucial for model development. Current methods for training VLM judges mainly rely on large-scale human preference annotations. However, such an approach is costly, and the annotations easily become obsolete as models rapidly improve. In this work, we present a framework to self-train a VLM judge model without any human preference annotations, using only self-synthesized data. Our method is iterative and has three stages: (1) generate diverse multimodal instruction-response pairs at varying quality levels, (2) generate reasoning traces and judgments for each pair, removing the ones that do not match our expected quality levels, and (3) training on correct judge answers and their reasoning traces. We evaluate the resulting judge on Multimodal RewardBench and VL-RewardBench across domains: correctness, preference, reasoning, safety, and visual question-answering. Our method improves a Llama-3.2-11B multimodal judge from 0.38 to 0.51 in overall accuracy on VL-RewardBench, often outperforming much larger models including Llama-3.2-90B, GPT-4o, and Claude 3.5 Sonnet, with particularly strong gains in general, hallucination, and reasoning dimensions. The overall strength of these human-annotation-free results suggest the potential for a future self-judge that evolves alongside rapidly improving VLM capabilities.
>
---
#### [new 022] CARD: Correlation Aware Restoration with Diffusion
- **分类: cs.CV**

- **简介: 该论文研究图像恢复任务，针对真实传感器中空间相关噪声导致现有扩散模型性能下降的问题，提出无需训练的CARD方法，通过噪声白化处理实现对相关噪声的有效恢复，并构建真实相关噪声数据集CIN-D验证其优越性。**

- **链接: [https://arxiv.org/pdf/2512.05268v1](https://arxiv.org/pdf/2512.05268v1)**

> **作者:** Niki Nezakati; Arnab Ghosh; Amit Roy-Chowdhury; Vishwanath Saragadam
>
> **摘要:** Denoising diffusion models have achieved state-of-the-art performance in image restoration by modeling the process as sequential denoising steps. However, most approaches assume independent and identically distributed (i.i.d.) Gaussian noise, while real-world sensors often exhibit spatially correlated noise due to readout mechanisms, limiting their practical effectiveness. We introduce Correlation Aware Restoration with Diffusion (CARD), a training-free extension of DDRM that explicitly handles correlated Gaussian noise. CARD first whitens the noisy observation, which converts the noise into an i.i.d. form. Then, the diffusion restoration steps are replaced with noise-whitened updates, which inherits DDRM's closed-form sampling efficiency while now being able to handle correlated noise. To emphasize the importance of addressing correlated noise, we contribute CIN-D, a novel correlated noise dataset captured across diverse illumination conditions to evaluate restoration methods on real rolling-shutter sensor noise. This dataset fills a critical gap in the literature for experimental evaluation with real-world correlated noise. Experiments on standard benchmarks with synthetic correlated noise and on CIN-D demonstrate that CARD consistently outperforms existing methods across denoising, deblurring, and super-resolution tasks.
>
---
#### [new 023] Moving object detection from multi-depth images with an attention-enhanced CNN
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对太阳系移动天体检测中易受噪声干扰、依赖人工验证的问题，提出一种融合注意力机制的多输入卷积神经网络。通过同时处理多帧图像并增强关键特征关注，显著提升检测准确率，降低99%以上人工工作量。**

- **链接: [https://arxiv.org/pdf/2512.05415v1](https://arxiv.org/pdf/2512.05415v1)**

> **作者:** Masato Shibukawa; Fumi Yoshida; Toshifumi Yanagisawa; Takashi Ito; Hirohisa Kurosaki; Makoto Yoshikawa; Kohki Kamiya; Ji-an Jiang; Wesley Fraser; JJ Kavelaars; Susan Benecchi; Anne Verbiscer; Akira Hatakeyama; Hosei O; Naoya Ozaki
>
> **备注:** 14 pages, 22 figures, submitted to PASJ
>
> **摘要:** One of the greatest challenges for detecting moving objects in the solar system from wide-field survey data is determining whether a signal indicates a true object or is due to some other source, like noise. Object verification has relied heavily on human eyes, which usually results in significant labor costs. In order to address this limitation and reduce the reliance on manual intervention, we propose a multi-input convolutional neural network integrated with a convolutional block attention module. This method is specifically tailored to enhance the moving object detection system that we have developed and used previously. The current method introduces two innovations. This first one is a multi-input architecture that processes multiple stacked images simultaneously. The second is the incorporation of the convolutional block attention module which enables the model to focus on essential features in both spatial and channel dimensions. These advancements facilitate efficient learning from multiple inputs, leading to more robust detection of moving objects. The performance of the model is evaluated on a dataset consisting of approximately 2,000 observational images. We achieved an accuracy of nearly 99% with AUC (an Area Under the Curve) of >0.99. These metrics indicate that the proposed model achieves excellent classification performance. By adjusting the threshold for object detection, the new model reduces the human workload by more than 99% compared to manual verification.
>
---
#### [new 024] NormalView: sensor-agnostic tree species classification from backpack and aerial lidar data using geometric projections
- **分类: cs.CV**

- **简介: 该论文研究树种分类任务，旨在解决不同传感器点云数据的跨设备兼容性问题。作者提出NormalView方法，将点云几何信息投影为二维图像，结合YOLOv11进行分类，并验证多光谱强度信息的增益效果，实现传感器无关的高精度树种识别。**

- **链接: [https://arxiv.org/pdf/2512.05610v1](https://arxiv.org/pdf/2512.05610v1)**

> **作者:** Juho Korkeala; Jesse Muhojoki; Josef Taher; Klaara Salolahti; Matti Hyyppä; Antero Kukko; Juha Hyyppä
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Laser scanning has proven to be an invaluable tool in assessing the decomposition of forest environments. Mobile laser scanning (MLS) has shown to be highly promising for extremely accurate, tree level inventory. In this study, we present NormalView, a sensor-agnostic projection-based deep learning method for classifying tree species from point cloud data. NormalView embeds local geometric information into two-dimensional projections, in the form of normal vector estimates, and uses the projections as inputs to an image classification network, YOLOv11. In addition, we inspected the effect of multispectral radiometric intensity information on classification performance. We trained and tested our model on high-density MLS data (7 species, ~5000 pts/m^2), as well as high-density airborne laser scanning (ALS) data (9 species, >1000 pts/m^2). On the MLS data, NormalView achieves an overall accuracy (macro-average accuracy) of 95.5 % (94.8 %), and 91.8 % (79.1 %) on the ALS data. We found that having intensity information from multiple scanners provides benefits in tree species classification, and the best model on the multispectral ALS dataset was a model using intensity information from all three channels of the multispectral ALS. This study demonstrates that projection-based methods, when enhanced with geometric information and coupled with state-of-the-art image classification backbones, can achieve exceptional results. Crucially, these methods are sensor-agnostic, relying only on geometric information. Additionally, we publically release the MLS dataset used in the study.
>
---
#### [new 025] EditThinker: Unlocking Iterative Reasoning for Any Image Editor
- **分类: cs.CV**

- **简介: 该论文研究指令引导的图像编辑任务，旨在提升模型对指令的遵循能力。针对单步编辑效果有限的问题，提出迭代推理框架EditThinker，通过“编辑- critique - refinement”循环，利用MLLM生成批评与改进指令，并用强化学习优化，显著提升编辑效果。**

- **链接: [https://arxiv.org/pdf/2512.05965v1](https://arxiv.org/pdf/2512.05965v1)**

> **作者:** Hongyu Li; Manyuan Zhang; Dian Zheng; Ziyu Guo; Yimeng Jia; Kaituo Feng; Hao Yu; Yexin Liu; Yan Feng; Peng Pei; Xunliang Cai; Linjiang Huang; Hongsheng Li; Si Liu
>
> **备注:** Project page: https://appletea233.github.io/think-while-edit
>
> **摘要:** Instruction-based image editing has emerged as a prominent research area, which, benefiting from image generation foundation models, have achieved high aesthetic quality, making instruction-following capability the primary challenge. Existing approaches improve instruction adherence via supervised or reinforcement learning, yet single-turn success rates remain limited due to inherent stochasticity and a lack of deliberation. In this work, we propose a deliberative editing framework to 'think' while they edit, which simulates the human cognitive loop by iteratively executing a Think-while-Edit cycle: Critiquing results and Refining instructions , followed by Repeating the generation until satisfactory. Specifically, we train a single MLLM, EditThinker, to act as the reasoning engine of this framework, which jointly produce the critique score, reasoning process, and refined instructions. We employ reinforcement learning to align the EditThinker's thinking with its editing, thereby generating more targeted instruction improvements. Extensive experiments on four benchmarks demonstrate that our approach significantly improves the instruction-following capability of any image editing model by a large margin. We will release our data construction framework, datasets, and models to benefit the community.
>
---
#### [new 026] From Segments to Scenes: Temporal Understanding in Autonomous Driving via Vision-Language Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦自动驾驶中的时序理解任务，旨在解决现有VLM在理解自车视角下动态场景关系时表现不足的问题。作者构建了专属基准TAD，设计7项任务共6000 QA对，并提出两种无需训练的增强方法，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.05277v1](https://arxiv.org/pdf/2512.05277v1)**

> **作者:** Kevin Cannons; Saeed Ranjbar Alvar; Mohammad Asiful Hossain; Ahmad Rezaei; Mohsen Gholami; Alireza Heidarikhazaei; Zhou Weimin; Yong Zhang; Mohammad Akbari
>
> **摘要:** Temporal understanding in autonomous driving (AD) remains a significant challenge, even for recent state-of-the-art (SoTA) Vision-Language Models (VLMs). Prior work has introduced datasets and benchmarks aimed at improving temporal reasoning, but these have emphasized other video content, including sports, cooking, and movies. No existing benchmark focuses exclusively on the unique challenges of temporal understanding in ego-centric AD footage. To fill this gap, the Temporal Understanding in Autonomous Driving (TAD) benchmark is presented, which evaluates VLMs' ability to capture the dynamic relationships between actions in AD. TAD comprises nearly 6,000 question-answer (QA) pairs, spanning 7 human-designed tasks. In addition, an evaluation is performed that consists of 9 closed- and open-source generalist models as well as SoTA AD specialist models. When applied to TAD, current SoTA models demonstrated substandard accuracies, largely due to imperfect fine-grained motion understanding. To improve motion understanding and overall accuracy on TAD, two novel training-free solutions are proposed: Scene-CoT, that leverages Chain-of-Thought (CoT) and TCogMap, which incorporates an ego-centric temporal cognitive map. The proposed approaches are integrated with existing VLMs and improve average accuracy on TAD by up to 17.72%. By introducing TAD, benchmarking multiple SoTA models, and proposing effective enhancements, this work aims to catalyze future research on temporal understanding in AD. The benchmark and evaluation code are available at \href{https://huggingface.co/datasets/vbdai/TAD}{Hugging Face} and \href{https://github.com/vbdi/tad_bench}{Github}, respectively.
>
---
#### [new 027] Physics-Informed Graph Neural Network with Frequency-Aware Learning for Optical Aberration Correction
- **分类: cs.CV; physics.optics**

- **简介: 该论文针对显微成像中复杂光学像差导致图像退化的问题，提出ZRNet框架，属于物理引导的图像恢复任务。通过Zernike图模块建模泽尼克多项式物理关系，并设计频域对齐损失，联合优化像差系数预测与图像复原，提升大像差下的复原性能。**

- **链接: [https://arxiv.org/pdf/2512.05683v1](https://arxiv.org/pdf/2512.05683v1)**

> **作者:** Yong En Kok; Bowen Deng; Alexander Bentley; Andrew J. Parkes; Michael G. Somekh; Amanda J. Wright; Michael P. Pound
>
> **摘要:** Optical aberrations significantly degrade image quality in microscopy, particularly when imaging deeper into samples. These aberrations arise from distortions in the optical wavefront and can be mathematically represented using Zernike polynomials. Existing methods often address only mild aberrations on limited sample types and modalities, typically treating the problem as a black-box mapping without leveraging the underlying optical physics of wavefront distortions. We propose ZRNet, a physics-informed framework that jointly performs Zernike coefficient prediction and optical image Restoration. We contribute a Zernike Graph module that explicitly models physical relationships between Zernike polynomials based on their azimuthal degrees-ensuring that learned corrections align with fundamental optical principles. To further enforce physical consistency between image restoration and Zernike prediction, we introduce a Frequency-Aware Alignment (FAA) loss, which better aligns Zernike coefficient prediction and image features in the Fourier domain. Extensive experiments on CytoImageNet demonstrates that our approach achieves state-of-the-art performance in both image restoration and Zernike coefficient prediction across diverse microscopy modalities and biological samples with complex, large-amplitude aberrations. Code is available at https://github.com/janetkok/ZRNet.
>
---
#### [new 028] AREA3D: Active Reconstruction Agent with Unified Feed-Forward 3D Perception and Vision-Language Guidance
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究主动3D重建任务，旨在解决传统方法依赖手工几何规则导致视图冗余的问题。作者提出AREA3D，结合前馈3D感知与视觉-语言引导，解耦不确定性建模并引入语义指导，提升稀疏视角下的重建精度。**

- **链接: [https://arxiv.org/pdf/2512.05131v1](https://arxiv.org/pdf/2512.05131v1)**

> **作者:** Tianling Xu; Shengzhe Gan; Leslie Gu; Yuelei Li; Fangneng Zhan; Hanspeter Pfister
>
> **备注:** Under review
>
> **摘要:** Active 3D reconstruction enables an agent to autonomously select viewpoints to efficiently obtain accurate and complete scene geometry, rather than passively reconstructing scenes from pre-collected images. However, existing active reconstruction methods often rely on hand-crafted geometric heuristics, which can lead to redundant observations without substantially improving reconstruction quality. To address this limitation, we propose AREA3D, an active reconstruction agent that leverages feed-forward 3D reconstruction models and vision-language guidance. Our framework decouples view-uncertainty modeling from the underlying feed-forward reconstructor, enabling precise uncertainty estimation without expensive online optimization. In addition, an integrated vision-language model provides high-level semantic guidance, encouraging informative and diverse viewpoints beyond purely geometric cues. Extensive experiments on both scene-level and object-level benchmarks demonstrate that AREA3D achieves state-of-the-art reconstruction accuracy, particularly in the sparse-view regime. Code will be made available at: https://github.com/TianlingXu/AREA3D .
>
---
#### [new 029] 2K-Characters-10K-Stories: A Quality-Gated Stylized Narrative Dataset with Disentangled Control and Sequence Consistency
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉叙事生成任务，旨在解决角色身份与动态属性耦合导致的序列不一致问题。作者构建了包含2000个独特角色和1万故事的大规模数据集，提出人机协同流程与解耦控制机制，实现高质量、可控且连贯的视觉叙事生成。**

- **链接: [https://arxiv.org/pdf/2512.05557v1](https://arxiv.org/pdf/2512.05557v1)**

> **作者:** Xingxi Yin; Yicheng Li; Gong Yan; Chenglin Li; Jian Zhao; Cong Huang; Yue Deng; Yin Zhang
>
> **摘要:** Sequential identity consistency under precise transient attribute control remains a long-standing challenge in controllable visual storytelling. Existing datasets lack sufficient fidelity and fail to disentangle stable identities from transient attributes, limiting structured control over pose, expression, and scene composition and thus constraining reliable sequential synthesis. To address this gap, we introduce \textbf{2K-Characters-10K-Stories}, a multi-modal stylized narrative dataset of \textbf{2{,}000} uniquely stylized characters appearing across \textbf{10{,}000} illustration stories. It is the first dataset that pairs large-scale unique identities with explicit, decoupled control signals for sequential identity consistency. We introduce a \textbf{Human-in-the-Loop pipeline (HiL)} that leverages expert-verified character templates and LLM-guided narrative planning to generate highly-aligned structured data. A \textbf{decoupled control} scheme separates persistent identity from transient attributes -- pose and expression -- while a \textbf{Quality-Gated loop} integrating MMLM evaluation, Auto-Prompt Tuning, and Local Image Editing enforces pixel-level consistency. Extensive experiments demonstrate that models fine-tuned on our dataset achieves performance comparable to closed-source models in generating visual narratives.
>
---
#### [new 030] UniFS: Unified Multi-Contrast MRI Reconstruction via Frequency-Spatial Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究多对比度MRI重建任务，旨在解决现有方法对不同k空间欠采样模式泛化能力差的问题。提出UniFS模型，通过频域-空域融合框架，实现无需重训练即可适应多种欠采样模式，并充分挖掘跨模态频域互补信息，提升重建性能。**

- **链接: [https://arxiv.org/pdf/2512.05481v1](https://arxiv.org/pdf/2512.05481v1)**

> **作者:** Jialin Li; Yiwei Ren; Kai Pan; Dong Wei; Pujin Cheng; Xian Wu; Xiaoying Tang
>
> **摘要:** Recently, Multi-Contrast MR Reconstruction (MCMR) has emerged as a hot research topic that leverages high-quality auxiliary modalities to reconstruct undersampled target modalities of interest. However, existing methods often struggle to generalize across different k-space undersampling patterns, requiring the training of a separate model for each specific pattern, which limits their practical applicability. To address this challenge, we propose UniFS, a Unified Frequency-Spatial Fusion model designed to handle multiple k-space undersampling patterns for MCMR tasks without any need for retraining. UniFS integrates three key modules: a Cross-Modal Frequency Fusion module, an Adaptive Mask-Based Prompt Learning module, and a Dual-Branch Complementary Refinement module. These modules work together to extract domain-invariant features from diverse k-space undersampling patterns while dynamically adapt to their own variations. Another limitation of existing MCMR methods is their tendency to focus solely on spatial information while neglect frequency characteristics, or extract only shallow frequency features, thus failing to fully leverage complementary cross-modal frequency information. To relieve this issue, UniFS introduces an adaptive prompt-guided frequency fusion module for k-space learning, significantly enhancing the model's generalization performance. We evaluate our model on the BraTS and HCP datasets with various k-space undersampling patterns and acceleration factors, including previously unseen patterns, to comprehensively assess UniFS's generalizability. Experimental results across multiple scenarios demonstrate that UniFS achieves state-of-the-art performance. Our code is available at https://github.com/LIKP0/UniFS.
>
---
#### [new 031] Measuring the Effect of Background on Classification and Feature Importance in Deep Learning for AV Perception
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究自动驾驶感知中深度学习模型的可解释性，旨在量化背景特征对分类和特征重要性的影响。通过构建六种不同背景相关性和相机变化的合成交通标志数据集，分析模型是否依赖物体或背景进行分类，揭示训练域变化下背景特征的重要性变化。**

- **链接: [https://arxiv.org/pdf/2512.05937v1](https://arxiv.org/pdf/2512.05937v1)**

> **作者:** Anne Sielemann; Valentin Barner; Stefan Wolf; Masoud Roschani; Jens Ziehn; Juergen Beyerer
>
> **备注:** 8 pages, 2 figures, 7 tables
>
> **摘要:** Common approaches to explainable AI (XAI) for deep learning focus on analyzing the importance of input features on the classification task in a given model: saliency methods like SHAP and GradCAM are used to measure the impact of spatial regions of the input image on the classification result. Combined with ground truth information about the location of the object in the input image (e.g., a binary mask), it is determined whether object pixels had a high impact on the classification result, or whether the classification focused on background pixels. The former is considered to be a sign of a healthy classifier, whereas the latter is assumed to suggest overfitting on spurious correlations. A major challenge, however, is that these intuitive interpretations are difficult to test quantitatively, and hence the output of such explanations lacks an explanation itself. One particular reason is that correlations in real-world data are difficult to avoid, and whether they are spurious or legitimate is debatable. Synthetic data in turn can facilitate to actively enable or disable correlations where desired but often lack a sufficient quantification of realism and stochastic properties. [...] Therefore, we systematically generate six synthetic datasets for the task of traffic sign recognition, which differ only in their degree of camera variation and background correlation [...] to quantify the isolated influence of background correlation, different levels of camera variation, and considered traffic sign shapes on the classification performance, as well as background feature importance. [...] Results include a quantification of when and how much background features gain importance to support the classification task based on changes in the training domain [...]. Download: synset.de/datasets/synset-signset-ger/background-effect
>
---
#### [new 032] Bring Your Dreams to Life: Continual Text-to-Video Customization
- **分类: cs.CV**

- **简介: 该论文研究持续文本到视频定制生成任务，解决现有方法在增量学习中出现的灾难性遗忘和概念忽略问题。提出CCVD模型，通过属性保留模块、任务感知聚合策略和区域注意力引导的条件合成，实现新旧概念的持续学习与高效生成。**

- **链接: [https://arxiv.org/pdf/2512.05802v1](https://arxiv.org/pdf/2512.05802v1)**

> **作者:** Jiahua Dong; Xudong Wang; Wenqi Liang; Zongyan Han; Meng Cao; Duzhen Zhang; Hanbin Zhao; Zhi Han; Salman Khan; Fahad Shahbaz Khan
>
> **备注:** Accepted to AAAI2026
>
> **摘要:** Customized text-to-video generation (CTVG) has recently witnessed great progress in generating tailored videos from user-specific text. However, most CTVG methods assume that personalized concepts remain static and do not expand incrementally over time. Additionally, they struggle with forgetting and concept neglect when continuously learning new concepts, including subjects and motions. To resolve the above challenges, we develop a novel Continual Customized Video Diffusion (CCVD) model, which can continuously learn new concepts to generate videos across various text-to-video generation tasks by tackling forgetting and concept neglect. To address catastrophic forgetting, we introduce a concept-specific attribute retention module and a task-aware concept aggregation strategy. They can capture the unique characteristics and identities of old concepts during training, while combining all subject and motion adapters of old concepts based on their relevance during testing. Besides, to tackle concept neglect, we develop a controllable conditional synthesis to enhance regional features and align video contexts with user conditions, by incorporating layer-specific region attention-guided noise estimation. Extensive experimental comparisons demonstrate that our CCVD outperforms existing CTVG models. The code is available at https://github.com/JiahuaDong/CCVD.
>
---
#### [new 033] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking
- **分类: cs.CV**

- **简介: 该论文研究RGB-T跟踪中的参数高效微调，旨在解决低秩适应中秩空间冗余导致表达能力不足的问题。提出分组正交低秩适应框架，通过重要性分析冻结关键秩，并对冗余秩分组施加正交约束，促进互补特征学习，提升模型适应性与性能。**

- **链接: [https://arxiv.org/pdf/2512.05359v1](https://arxiv.org/pdf/2512.05359v1)**

> **作者:** Zekai Shao; Yufan Hu; Jingyuan Liu; Bin Fan; Hongmin Liu
>
> **备注:** 13 pages, 8 figures. Accepted by AAAI 2026. Extended version
>
> **摘要:** Parameter-efficient fine-tuning has emerged as a promising paradigm in RGB-T tracking, enabling downstream task adaptation by freezing pretrained parameters and fine-tuning only a small set of parameters. This set forms a rank space made up of multiple individual ranks, whose expressiveness directly shapes the model's adaptability. However, quantitative analysis reveals low-rank adaptation exhibits significant redundancy in the rank space, with many ranks contributing almost no practical information. This hinders the model's ability to learn more diverse knowledge to address the various challenges in RGB-T tracking. To address this issue, we propose the Group Orthogonal Low-Rank Adaptation (GOLA) framework for RGB-T tracking, which effectively leverages the rank space through structured parameter learning. Specifically, we adopt a rank decomposition partitioning strategy utilizing singular value decomposition to quantify rank importance, freeze crucial ranks to preserve the pretrained priors, and cluster the redundant ranks into groups to prepare for subsequent orthogonal constraints. We further design an inter-group orthogonal constraint strategy. This constraint enforces orthogonality between rank groups, compelling them to learn complementary features that target diverse challenges, thereby alleviating information redundancy. Experimental results demonstrate that GOLA effectively reduces parameter redundancy and enhances feature representation capabilities, significantly outperforming state-of-the-art methods across four benchmark datasets and validating its effectiveness in RGB-T tracking tasks.
>
---
#### [new 034] FlowEO: Generative Unsupervised Domain Adaptation for Earth Observation
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究地球观测图像的无监督域适应（UDA）任务，旨在解决跨传感器、地域及时空分布差异导致模型泛化能力差的问题。提出FlowEO框架，利用流匹配学习源到目标域的语义保持图像转换，提升分类与分割性能。**

- **链接: [https://arxiv.org/pdf/2512.05140v1](https://arxiv.org/pdf/2512.05140v1)**

> **作者:** Georges Le Bellier; Nicolas Audebert
>
> **备注:** 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Mar 2026, Tucson (AZ), United States
>
> **摘要:** The increasing availability of Earth observation data offers unprecedented opportunities for large-scale environmental monitoring and analysis. However, these datasets are inherently heterogeneous, stemming from diverse sensors, geographical regions, acquisition times, and atmospheric conditions. Distribution shifts between training and deployment domains severely limit the generalization of pretrained remote sensing models, making unsupervised domain adaptation (UDA) crucial for real-world applications. We introduce FlowEO, a novel framework that leverages generative models for image-space UDA in Earth observation. We leverage flow matching to learn a semantically preserving mapping that transports from the source to the target image distribution. This allows us to tackle challenging domain adaptation configurations for classification and semantic segmentation of Earth observation images. We conduct extensive experiments across four datasets covering adaptation scenarios such as SAR to optical translation and temporal and semantic shifts caused by natural disasters. Experimental results demonstrate that FlowEO outperforms existing image translation approaches for domain adaptation while achieving on-par or better perceptual image quality, highlighting the potential of flow-matching-based UDA for remote sensing.
>
---
#### [new 035] Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦长视频理解任务，旨在解决现有方法因被动感知导致的计算冗余和信息模糊问题。作者提出Active Video Perception（AVP）框架，通过MLLM代理迭代执行“规划-观察-反思”流程，主动获取与查询相关的时空证据，显著提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.05774v1](https://arxiv.org/pdf/2512.05774v1)**

> **作者:** Ziyang Wang; Honglu Zhou; Shijie Wang; Junnan Li; Caiming Xiong; Silvio Savarese; Mohit Bansal; Michael S. Ryoo; Juan Carlos Niebles
>
> **备注:** Website: https://activevideoperception.github.io/
>
> **摘要:** Long video understanding (LVU) is challenging because answering real-world queries often depends on sparse, temporally dispersed cues buried in hours of mostly redundant and irrelevant content. While agentic pipelines improve video reasoning capabilities, prevailing frameworks rely on a query-agnostic captioner to perceive video information, which wastes computation on irrelevant content and blurs fine-grained temporal and spatial information. Motivated by active perception theory, we argue that LVU agents should actively decide what, when, and where to observe, and continuously assess whether the current observation is sufficient to answer the query. We present Active Video Perception (AVP), an evidence-seeking framework that treats the video as an interactive environment and acquires compact, queryrelevant evidence directly from pixels. Concretely, AVP runs an iterative plan-observe-reflect process with MLLM agents. In each round, a planner proposes targeted video interactions, an observer executes them to extract time-stamped evidence, and a reflector evaluates the sufficiency of the evidence for the query, either halting with an answer or triggering further observation. Across five LVU benchmarks, AVP achieves highest performance with significant improvements. Notably, AVP outperforms the best agentic method by 5.7% in average accuracy while only requires 18.4% inference time and 12.4% input tokens.
>
---
#### [new 036] Deep Learning-Based Real-Time Sequential Facial Expression Analysis Using Geometric Features
- **分类: cs.CV**

- **简介: 该论文属于面部表情识别任务，旨在实现实时、准确的序列化表情分析。通过MediaPipe提取几何特征，结合ConvLSTM1D建模时序动态，实现高帧率下多阶段表情识别，提升了人机交互中情绪感知的效率与精度。**

- **链接: [https://arxiv.org/pdf/2512.05669v1](https://arxiv.org/pdf/2512.05669v1)**

> **作者:** Talha Enes Koksal; Abdurrahman Gumus
>
> **摘要:** Facial expression recognition is a crucial component in enhancing human-computer interaction and developing emotion-aware systems. Real-time detection and interpretation of facial expressions have become increasingly important for various applications, from user experience personalization to intelligent surveillance systems. This study presents a novel approach to real-time sequential facial expression recognition using deep learning and geometric features. The proposed method utilizes MediaPipe FaceMesh for rapid and accurate facial landmark detection. Geometric features, including Euclidean distances and angles, are extracted from these landmarks. Temporal dynamics are incorporated by analyzing feature differences between consecutive frames, enabling the detection of onset, apex, and offset phases of expressions. For classification, a ConvLSTM1D network followed by multilayer perceptron blocks is employed. The method's performance was evaluated on multiple publicly available datasets, including CK+, Oulu-CASIA (VIS and NIR), and MMI. Accuracies of 93%, 79%, 77%, and 68% were achieved respectively. Experiments with composite datasets were also conducted to assess the model's generalization capabilities. The approach demonstrated real-time applicability, processing approximately 165 frames per second on consumer-grade hardware. This research contributes to the field of facial expression analysis by providing a fast, accurate, and adaptable solution. The findings highlight the potential for further advancements in emotion-aware technologies and personalized user experiences, paving the way for more sophisticated human-computer interaction systems. To facilitate further research in this field, the complete source code for this study has been made publicly available on GitHub: https://github.com/miralab-ai/facial-expression-analysis.
>
---
#### [new 037] Inferring Compositional 4D Scenes without Ever Seeing One
- **分类: cs.CV**

- **简介: 该论文属4D场景重建任务，旨在解决真实世界多物体动态场景建模中缺乏4D组合数据的问题。提出COM4D方法，通过解耦空间与时间注意力训练，仅用静态多物体或单动态物体监督，实现无需4D组合数据的完整4D场景推断。**

- **链接: [https://arxiv.org/pdf/2512.05272v1](https://arxiv.org/pdf/2512.05272v1)**

> **作者:** Ahmet Berke Gokmen; Ajad Chhatkuli; Luc Van Gool; Danda Pani Paudel
>
> **备注:** Project page: https://github.com/insait-institute/COM4D
>
> **摘要:** Scenes in the real world are often composed of several static and dynamic objects. Capturing their 4-dimensional structures, composition and spatio-temporal configuration in-the-wild, though extremely interesting, is equally hard. Therefore, existing works often focus on one object at a time, while relying on some category-specific parametric shape model for dynamic objects. This can lead to inconsistent scene configurations, in addition to being limited to the modeled object categories. We propose COM4D (Compositional 4D), a method that consistently and jointly predicts the structure and spatio-temporal configuration of 4D/3D objects using only static multi-object or dynamic single object supervision. We achieve this by a carefully designed training of spatial and temporal attentions on 2D video input. The training is disentangled into learning from object compositions on the one hand, and single object dynamics throughout the video on the other, thus completely avoiding reliance on 4D compositional training data. At inference time, our proposed attention mixing mechanism combines these independently learned attentions, without requiring any 4D composition examples. By alternating between spatial and temporal reasoning, COM4D reconstructs complete and persistent 4D scenes with multiple interacting objects directly from monocular videos. Furthermore, COM4D provides state-of-the-art results in existing separate problems of 4D object and composed 3D reconstruction despite being purely data-driven.
>
---
#### [new 038] HQ-DM: Single Hadamard Transformation-Based Quantization-Aware Training for Low-Bit Diffusion Models
- **分类: cs.CV**

- **简介: 该论文研究低比特扩散模型的量化训练，旨在缓解激活矩阵中的异常值问题。提出HQ-DM框架，采用单Hadamard变换抑制异常值，支持INT卷积并避免权重异常放大，显著提升低比特量化下的生成性能。**

- **链接: [https://arxiv.org/pdf/2512.05746v1](https://arxiv.org/pdf/2512.05746v1)**

> **作者:** Shizhuo Mao; Hongtao Zou; Qihu Xie; Song Chen; Yi Kang
>
> **摘要:** Diffusion models have demonstrated significant applications in the field of image generation. However, their high computational and memory costs pose challenges for deployment. Model quantization has emerged as a promising solution to reduce storage overhead and accelerate inference. Nevertheless, existing quantization methods for diffusion models struggle to mitigate outliers in activation matrices during inference, leading to substantial performance degradation under low-bit quantization scenarios. To address this, we propose HQ-DM, a novel Quantization-Aware Training framework that applies Single Hadamard Transformation to activation matrices. This approach effectively reduces activation outliers while preserving model performance under quantization. Compared to traditional Double Hadamard Transformation, our proposed scheme offers distinct advantages by seamlessly supporting INT convolution operations while preventing the amplification of weight outliers. For conditional generation on the ImageNet 256x256 dataset using the LDM-4 model, our W4A4 and W4A3 quantization schemes improve the Inception Score by 12.8% and 467.73%, respectively, over the existing state-of-the-art method.
>
---
#### [new 039] Genetic Algorithms For Parameter Optimization for Disparity Map Generation of Radiata Pine Branch Images
- **分类: cs.CV**

- **简介: 该论文研究无人机林木分支距离测量中的视差图生成，针对传统立体匹配算法参数调优困难的问题，提出基于遗传算法的自动优化框架，提升精度与鲁棒性，适用于资源受限系统。**

- **链接: [https://arxiv.org/pdf/2512.05410v1](https://arxiv.org/pdf/2512.05410v1)**

> **作者:** Yida Lin; Bing Xue; Mengjie Zhang; Sam Schofield; Richard Green
>
> **摘要:** Traditional stereo matching algorithms like Semi-Global Block Matching (SGBM) with Weighted Least Squares (WLS) filtering offer speed advantages over neural networks for UAV applications, generating disparity maps in approximately 0.5 seconds per frame. However, these algorithms require meticulous parameter tuning. We propose a Genetic Algorithm (GA) based parameter optimization framework that systematically searches for optimal parameter configurations for SGBM and WLS, enabling UAVs to measure distances to tree branches with enhanced precision while maintaining processing efficiency. Our contributions include: (1) a novel GA-based parameter optimization framework that eliminates manual tuning; (2) a comprehensive evaluation methodology using multiple image quality metrics; and (3) a practical solution for resource-constrained UAV systems. Experimental results demonstrate that our GA-optimized approach reduces Mean Squared Error by 42.86% while increasing Peak Signal-to-Noise Ratio and Structural Similarity by 8.47% and 28.52%, respectively, compared with baseline configurations. Furthermore, our approach demonstrates superior generalization performance across varied imaging conditions, which is critcal for real-world forestry applications.
>
---
#### [new 040] DistillFSS: Synthesizing Few-Shot Knowledge into a Lightweight Segmentation Model
- **分类: cs.CV**

- **简介: 该论文研究跨域少样本语义分割（CD-FSS），旨在解决分布偏移、标签空间不重叠和标注样本稀缺下的分割难题。作者提出DistillFSS，通过师生蒸馏将少样本知识嵌入模型参数，实现无需支持图像的高效轻量推理，并支持快速扩展新类。**

- **链接: [https://arxiv.org/pdf/2512.05613v1](https://arxiv.org/pdf/2512.05613v1)**

> **作者:** Pasquale De Marinis; Pieter M. Blok; Uzay Kaymak; Rogier Brussee; Gennaro Vessio; Giovanna Castellano
>
> **摘要:** Cross-Domain Few-Shot Semantic Segmentation (CD-FSS) seeks to segment unknown classes in unseen domains using only a few annotated examples. This setting is inherently challenging: source and target domains exhibit substantial distribution shifts, label spaces are disjoint, and support images are scarce--making standard episodic methods unreliable and computationally demanding at test time. To address these constraints, we propose DistillFSS, a framework that embeds support-set knowledge directly into a model's parameters through a teacher--student distillation process. By internalizing few-shot reasoning into a dedicated layer within the student network, DistillFSS eliminates the need for support images at test time, enabling fast, lightweight inference, while allowing efficient extension to novel classes in unseen domains through rapid teacher-driven specialization. Combined with fine-tuning, the approach scales efficiently to large support sets and significantly reduces computational overhead. To evaluate the framework under realistic conditions, we introduce a new CD-FSS benchmark spanning medical imaging, industrial inspection, and remote sensing, with disjoint label spaces and variable support sizes. Experiments show that DistillFSS matches or surpasses state-of-the-art baselines, particularly in multi-class and multi-shot scenarios, while offering substantial efficiency gains. The code is available at https://github.com/pasqualedem/DistillFSS.
>
---
#### [new 041] EFDiT: Efficient Fine-grained Image Generation Using Diffusion Transformer Models
- **分类: cs.CV**

- **简介: 该论文研究细粒度图像生成任务，旨在解决语义信息纠缠和细节不足问题。提出分层嵌入器融合超类与子类语义，引入超分辨率增强细节，并设计高效ProAttention机制，在公共基准上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.05152v1](https://arxiv.org/pdf/2512.05152v1)**

> **作者:** Kun Wang; Donglin Di; Tonghua Su; Lei Fan
>
> **备注:** 6pages, 5figures, published to 2025 IEEE International Conference on Multimedia and Expo (ICME), Nantes, France, 2025
>
> **摘要:** Diffusion models are highly regarded for their controllability and the diversity of images they generate. However, class-conditional generation methods based on diffusion models often focus on more common categories. In large-scale fine-grained image generation, issues of semantic information entanglement and insufficient detail in the generated images still persist. This paper attempts to introduce a concept of a tiered embedder in fine-grained image generation, which integrates semantic information from both super and child classes, allowing the diffusion model to better incorporate semantic information and address the issue of semantic entanglement. To address the issue of insufficient detail in fine-grained images, we introduce the concept of super-resolution during the perceptual information generation stage, enhancing the detailed features of fine-grained images through enhancement and degradation models. Furthermore, we propose an efficient ProAttention mechanism that can be effectively implemented in the diffusion model. We evaluate our method through extensive experiments on public benchmarks, demonstrating that our approach outperforms other state-of-the-art fine-tuning methods in terms of performance.
>
---
#### [new 042] InverseCrafter: Efficient Video ReCapture as a Latent Domain Inverse Problem
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于4D视频生成任务，旨在解决现有方法计算成本高、易遗忘原始生成先验的问题。作者提出InverseCrafter，将视频重生成建模为潜在空间的修复逆问题，通过设计连续多通道掩码编码退化操作，实现高效、低开销的可控视频生成与编辑。**

- **链接: [https://arxiv.org/pdf/2512.05672v1](https://arxiv.org/pdf/2512.05672v1)**

> **作者:** Yeobin Hong; Suhyeon Lee; Hyungjin Chung; Jong Chul Ye
>
> **摘要:** Recent approaches to controllable 4D video generation often rely on fine-tuning pre-trained Video Diffusion Models (VDMs). This dominant paradigm is computationally expensive, requiring large-scale datasets and architectural modifications, and frequently suffers from catastrophic forgetting of the model's original generative priors. Here, we propose InverseCrafter, an efficient inpainting inverse solver that reformulates the 4D generation task as an inpainting problem solved in the latent space. The core of our method is a principled mechanism to encode the pixel space degradation operator into a continuous, multi-channel latent mask, thereby bypassing the costly bottleneck of repeated VAE operations and backpropagation. InverseCrafter not only achieves comparable novel view generation and superior measurement consistency in camera control tasks with near-zero computational overhead, but also excels at general-purpose video inpainting with editing. Code is available at https://github.com/yeobinhong/InverseCrafter.
>
---
#### [new 043] Hyperspectral Unmixing with 3D Convolutional Sparse Coding and Projected Simplex Volume Maximization
- **分类: cs.CV**

- **简介: 该论文研究高光谱解混任务，旨在分解像素为端元及其丰度。提出3D-CSCNet网络，基于3D卷积稀疏编码与算法展开，并引入PSVM算法初始化端元，结合自编码器框架联合学习光谱与空间特征，提升解混精度。**

- **链接: [https://arxiv.org/pdf/2512.05674v1](https://arxiv.org/pdf/2512.05674v1)**

> **作者:** Gargi Panda; Soumitra Kundu; Saumik Bhattacharya; Aurobinda Routray
>
> **摘要:** Hyperspectral unmixing (HSU) aims to separate each pixel into its constituent endmembers and estimate their corresponding abundance fractions. This work presents an algorithm-unrolling-based network for the HSU task, named the 3D Convolutional Sparse Coding Network (3D-CSCNet), built upon a 3D CSC model. Unlike existing unrolling-based networks, our 3D-CSCNet is designed within the powerful autoencoder (AE) framework. Specifically, to solve the 3D CSC problem, we propose a 3D CSC block (3D-CSCB) derived through deep algorithm unrolling. Given a hyperspectral image (HSI), 3D-CSCNet employs the 3D-CSCB to estimate the abundance matrix. The use of 3D CSC enables joint learning of spectral and spatial relationships in the 3D HSI data cube. The estimated abundance matrix is then passed to the AE decoder to reconstruct the HSI, and the decoder weights are extracted as the endmember matrix. Additionally, we propose a projected simplex volume maximization (PSVM) algorithm for endmember estimation, and the resulting endmembers are used to initialize the decoder weights of 3D-CSCNet. Extensive experiments on three real datasets and one simulated dataset with three different signal-to-noise ratio (SNR) levels demonstrate that our 3D-CSCNet outperforms state-of-the-art methods.
>
---
#### [new 044] Know-Show: Benchmarking Video-Language Models on Spatio-Temporal Grounded Reasoning
- **分类: cs.CV**

- **简介: 该论文聚焦视频-语言模型的时空 grounded 推理任务，旨在解决模型推理与视觉时空证据脱节的问题。提出 Know-Show 基准和无需训练的 GRAM 插件，评估并增强模型在动作语义与时空定位上的联合理解能力。**

- **链接: [https://arxiv.org/pdf/2512.05513v1](https://arxiv.org/pdf/2512.05513v1)**

> **作者:** Chinthani Sugandhika; Chen Li; Deepu Rajan; Basura Fernando
>
> **摘要:** Large Video-Language Models (Video-LMs) have achieved impressive progress in multimodal understanding, yet their reasoning remains weakly grounded in space and time. We present Know-Show, a new benchmark designed to evaluate spatio-temporal grounded reasoning, the ability of a model to reason about actions and their semantics while simultaneously grounding its inferences in visual and temporal evidence. Know-Show unifies reasoning and localization within a single evaluation framework consisting of five complementary scenarios across spatial (person, object, person-object, and hand-object) and temporal dimensions. Built from Charades, Action Genome, and Ego4D with 2.5K human-authored questions, the benchmark exposes significant gaps between current Video-LMs and human reasoning. To bridge this gap, we propose GRAM, a training-free plug-in that augments Video-LMs with fine-grained grounding through attention-based video token selection and explicit timestamp encoding. Extensive experiments across open and closed Video-LMs (Qwen, VideoLLaVA, GPT-4o, and Gemini, etc.) reveal that existing models struggle to "show what they know" and vice versa, especially in fine-grained hand-object interactions. Know-Show establishes a unified standard for assessing grounded reasoning in video-language understanding and provides insights toward developing interpretable and reliable multimodal reasoning systems. We will release the code at https://github.com/LUNAProject22/Know-Show.
>
---
#### [new 045] Ideal Observer for Segmentation of Dead Leaves Images
- **分类: cs.CV; math.ST; stat.ME**

- **简介: 该论文研究图像分割任务，旨在解决基于“死叶模型”生成图像的像素区域划分问题。作者提出贝叶斯理想观测器，计算后验概率，为分割性能提供理论上限，用于评估人类视觉与算法表现。**

- **链接: [https://arxiv.org/pdf/2512.05539v1](https://arxiv.org/pdf/2512.05539v1)**

> **作者:** Swantje Mahncke; Malte Ott
>
> **备注:** 41 pages, 16 figures
>
> **摘要:** The human visual environment is comprised of different surfaces that are distributed in space. The parts of a scene that are visible at any one time are governed by the occlusion of overlapping objects. In this work we consider "dead leaves" models, which replicate these occlusions when generating images by layering objects on top of each other. A dead leaves model is a generative model comprised of distributions for object position, shape, color and texture. An image is generated from a dead leaves model by sampling objects ("leaves") from these distributions until a stopping criterion is reached, usually when the image is fully covered or until a given number of leaves was sampled. Here, we describe a theoretical approach, based on previous work, to derive a Bayesian ideal observer for the partition of a given set of pixels based on independent dead leaves model distributions. Extending previous work, we provide step-by-step explanations for the computation of the posterior probability as well as describe factors that determine the feasibility of practically applying this computation. The dead leaves image model and the associated ideal observer can be applied to study segmentation decisions in a limited number of pixels, providing a principled upper-bound on performance, to which humans and vision algorithms could be compared.
>
---
#### [new 046] DashFusion: Dual-stream Alignment with Hierarchical Bottleneck Fusion for Multimodal Sentiment Analysis
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对多模态情感分析中的对齐与融合问题，提出DashFusion框架。通过双流对齐实现时序与语义同步，结合分层瓶颈融合提升效率与性能，在多个数据集上取得最优结果。**

- **链接: [https://arxiv.org/pdf/2512.05515v1](https://arxiv.org/pdf/2512.05515v1)**

> **作者:** Yuhua Wen; Qifei Li; Yingying Zhou; Yingming Gao; Zhengqi Wen; Jianhua Tao; Ya Li
>
> **备注:** Accepted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2025
>
> **摘要:** Multimodal sentiment analysis (MSA) integrates various modalities, such as text, image, and audio, to provide a more comprehensive understanding of sentiment. However, effective MSA is challenged by alignment and fusion issues. Alignment requires synchronizing both temporal and semantic information across modalities, while fusion involves integrating these aligned features into a unified representation. Existing methods often address alignment or fusion in isolation, leading to limitations in performance and efficiency. To tackle these issues, we propose a novel framework called Dual-stream Alignment with Hierarchical Bottleneck Fusion (DashFusion). Firstly, dual-stream alignment module synchronizes multimodal features through temporal and semantic alignment. Temporal alignment employs cross-modal attention to establish frame-level correspondences among multimodal sequences. Semantic alignment ensures consistency across the feature space through contrastive learning. Secondly, supervised contrastive learning leverages label information to refine the modality features. Finally, hierarchical bottleneck fusion progressively integrates multimodal information through compressed bottleneck tokens, which achieves a balance between performance and computational efficiency. We evaluate DashFusion on three datasets: CMU-MOSI, CMU-MOSEI, and CH-SIMS. Experimental results demonstrate that DashFusion achieves state-of-the-art performance across various metrics, and ablation studies confirm the effectiveness of our alignment and fusion techniques. The codes for our experiments are available at https://github.com/ultramarineX/DashFusion.
>
---
#### [new 047] Curvature-Regularized Variational Autoencoder for 3D Scene Reconstruction from Sparse Depth
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究从稀疏深度数据重建完整3D场景，解决因测量不足导致的几何误差问题。提出基于离散拉普拉斯算子的曲率正则化方法，显著提升重建精度，仅增加15%训练开销且不增加推理成本。**

- **链接: [https://arxiv.org/pdf/2512.05783v1](https://arxiv.org/pdf/2512.05783v1)**

> **作者:** Maryam Yousefi; Soodeh Bakhshandeh
>
> **摘要:** When depth sensors provide only 5% of needed measurements, reconstructing complete 3D scenes becomes difficult. Autonomous vehicles and robots cannot tolerate the geometric errors that sparse reconstruction introduces. We propose curvature regularization through a discrete Laplacian operator, achieving 18.1% better reconstruction accuracy than standard variational autoencoders. Our contribution challenges an implicit assumption in geometric deep learning: that combining multiple geometric constraints improves performance. A single well-designed regularization term not only matches but exceeds the effectiveness of complex multi-term formulations. The discrete Laplacian offers stable gradients and noise suppression with just 15% training overhead and zero inference cost. Code and models are available at https://github.com/Maryousefi/GeoVAE-3D.
>
---
#### [new 048] TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows
- **分类: cs.CV**

- **简介: 该论文属于多模态生成任务，旨在解决大模型生成效率低的问题。提出TwinFlow框架，实现无需预训练教师和对抗网络的一步生成，显著提升推理速度并保持生成质量。**

- **链接: [https://arxiv.org/pdf/2512.05150v1](https://arxiv.org/pdf/2512.05150v1)**

> **作者:** Zhenglin Cheng; Peng Sun; Jianguo Li; Tao Lin
>
> **备注:** arxiv v0
>
> **摘要:** Recent advances in large multi-modal generative models have demonstrated impressive capabilities in multi-modal generation, including image and video generation. These models are typically built upon multi-step frameworks like diffusion and flow matching, which inherently limits their inference efficiency (requiring 40-100 Number of Function Evaluations (NFEs)). While various few-step methods aim to accelerate the inference, existing solutions have clear limitations. Prominent distillation-based methods, such as progressive and consistency distillation, either require an iterative distillation procedure or show significant degradation at very few steps (< 4-NFE). Meanwhile, integrating adversarial training into distillation (e.g., DMD/DMD2 and SANA-Sprint) to enhance performance introduces training instability, added complexity, and high GPU memory overhead due to the auxiliary trained models. To this end, we propose TwinFlow, a simple yet effective framework for training 1-step generative models that bypasses the need of fixed pretrained teacher models and avoids standard adversarial networks during training, making it ideal for building large-scale, efficient models. On text-to-image tasks, our method achieves a GenEval score of 0.83 in 1-NFE, outperforming strong baselines like SANA-Sprint (a GAN loss-based framework) and RCGM (a consistency-based framework). Notably, we demonstrate the scalability of TwinFlow by full-parameter training on Qwen-Image-20B and transform it into an efficient few-step generator. With just 1-NFE, our approach matches the performance of the original 100-NFE model on both the GenEval and DPG-Bench benchmarks, reducing computational cost by $100\times$ with minor quality degradation. Project page is available at https://zhenglin-cheng.com/twinflow.
>
---
#### [new 049] LPD: Learnable Prototypes with Diversity Regularization for Weakly Supervised Histopathology Segmentation
- **分类: cs.CV**

- **简介: 该论文研究弱监督病理图像分割，旨在缓解类间相似、类内差异大及CAM区域收缩问题。提出一种端到端的可学习原型框架，引入多样性正则化，提升类内形态覆盖，实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.05922v1](https://arxiv.org/pdf/2512.05922v1)**

> **作者:** Khang Le; Anh Mai Vu; Thi Kim Trang Vo; Ha Thach; Ngoc Bui Lam Quang; Thanh-Huy Nguyen; Minh H. N. Le; Zhu Han; Chandra Mohan; Hien Van Nguyen
>
> **备注:** Note: Khang Le and Anh Mai Vu contributed equally
>
> **摘要:** Weakly supervised semantic segmentation (WSSS) in histopathology reduces pixel-level labeling by learning from image-level labels, but it is hindered by inter-class homogeneity, intra-class heterogeneity, and CAM-induced region shrinkage (global pooling-based class activation maps whose activations highlight only the most distinctive areas and miss nearby class regions). Recent works address these challenges by constructing a clustering prototype bank and then refining masks in a separate stage; however, such two-stage pipelines are costly, sensitive to hyperparameters, and decouple prototype discovery from segmentation learning, limiting their effectiveness and efficiency. We propose a cluster-free, one-stage learnable-prototype framework with diversity regularization to enhance morphological intra-class heterogeneity coverage. Our approach achieves state-of-the-art (SOTA) performance on BCSS-WSSS, outperforming prior methods in mIoU and mDice. Qualitative segmentation maps show sharper boundaries and fewer mislabels, and activation heatmaps further reveal that, compared with clustering-based prototypes, our learnable prototypes cover more diverse and complementary regions within each class, providing consistent qualitative evidence for their effectiveness.
>
---
#### [new 050] SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations
- **分类: cs.CV**

- **简介: 该论文聚焦角色动画生成任务，旨在解决复杂场景下动作迁移中结构失真与时序不一致的问题。提出SCAIL框架，通过3D姿态表示与扩散Transformer中的全上下文姿态注入机制，实现高质量、时序连贯的动画生成。**

- **链接: [https://arxiv.org/pdf/2512.05905v1](https://arxiv.org/pdf/2512.05905v1)**

> **作者:** Wenhao Yan; Sheng Ye; Zhuoyi Yang; Jiayan Teng; ZhenHui Dong; Kairui Wen; Xiaotao Gu; Yong-Jin Liu; Jie Tang
>
> **摘要:** Achieving character animation that meets studio-grade production standards remains challenging despite recent progress. Existing approaches can transfer motion from a driving video to a reference image, but often fail to preserve structural fidelity and temporal consistency in wild scenarios involving complex motion and cross-identity animations. In this work, we present \textbf{SCAIL} (\textbf{S}tudio-grade \textbf{C}haracter \textbf{A}nimation via \textbf{I}n-context \textbf{L}earning), a framework designed to address these challenges from two key innovations. First, we propose a novel 3D pose representation, providing a more robust and flexible motion signal. Second, we introduce a full-context pose injection mechanism within a diffusion-transformer architecture, enabling effective spatio-temporal reasoning over full motion sequences. To align with studio-level requirements, we develop a curated data pipeline ensuring both diversity and quality, and establish a comprehensive benchmark for systematic evaluation. Experiments show that \textbf{SCAIL} achieves state-of-the-art performance and advances character animation toward studio-grade reliability and realism.
>
---
#### [new 051] MedDIFT: Multi-Scale Diffusion-Based Correspondence in 3D Medical Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决传统方法在低对比度区域易错配的问题。作者提出MedDIFT，利用预训练扩散模型的多尺度中间特征作为体素描述符，通过余弦相似度实现无需训练的3D医学图像对应关系匹配，性能优于传统方法，媲美最新学习模型。**

- **链接: [https://arxiv.org/pdf/2512.05571v1](https://arxiv.org/pdf/2512.05571v1)**

> **作者:** Xingyu Zhang; Anna Reithmeir; Fryderyk Kögl; Rickmer Braren; Julia A. Schnabel; Daniel M. Lang
>
> **摘要:** Accurate spatial correspondence between medical images is essential for longitudinal analysis, lesion tracking, and image-guided interventions. Medical image registration methods rely on local intensity-based similarity measures, which fail to capture global semantic structure and often yield mismatches in low-contrast or anatomically variable regions. Recent advances in diffusion models suggest that their intermediate representations encode rich geometric and semantic information. We present MedDIFT, a training-free 3D correspondence framework that leverages multi-scale features from a pretrained latent medical diffusion model as voxel descriptors. MedDIFT fuses diffusion activations into rich voxel-wise descriptors and matches them via cosine similarity, with an optional local-search prior. On a publicly available lung CT dataset, MedDIFT achieves correspondence accuracy comparable to the state-of-the-art learning-based UniGradICON model and surpasses conventional B-spline-based registration, without requiring any task-specific model training. Ablation experiments confirm that multi-level feature fusion and modest diffusion noise improve performance.
>
---
#### [new 052] Edit-aware RAW Reconstruction
- **分类: cs.CV**

- **简介: 该论文研究RAW图像重建任务，旨在解决现有方法在多样化编辑和渲染下性能下降的问题。作者提出一种即插即用的编辑感知损失函数，结合可微ISP模块模拟真实成像流程，提升重建RAW图像在不同编辑下的保真度与灵活性。**

- **链接: [https://arxiv.org/pdf/2512.05859v1](https://arxiv.org/pdf/2512.05859v1)**

> **作者:** Abhijith Punnappurath; Luxi Zhao; Ke Zhao; Hue Nguyen; Radek Grzeszczuk; Michael S. Brown
>
> **摘要:** Users frequently edit camera images post-capture to achieve their preferred photofinishing style. While editing in the RAW domain provides greater accuracy and flexibility, most edits are performed on the camera's display-referred output (e.g., 8-bit sRGB JPEG) since RAW images are rarely stored. Existing RAW reconstruction methods can recover RAW data from sRGB images, but these approaches are typically optimized for pixel-wise RAW reconstruction fidelity and tend to degrade under diverse rendering styles and editing operations. We introduce a plug-and-play, edit-aware loss function that can be integrated into any existing RAW reconstruction framework to make the recovered RAWs more robust to different rendering styles and edits. Our loss formulation incorporates a modular, differentiable image signal processor (ISP) that simulates realistic photofinishing pipelines with tunable parameters. During training, parameters for each ISP module are randomly sampled from carefully designed distributions that model practical variations in real camera processing. The loss is then computed in sRGB space between ground-truth and reconstructed RAWs rendered through this differentiable ISP. Incorporating our loss improves sRGB reconstruction quality by up to 1.5-2 dB PSNR across various editing conditions. Moreover, when applied to metadata-assisted RAW reconstruction methods, our approach enables fine-tuning for target edits, yielding further gains. Since photographic editing is the primary motivation for RAW reconstruction in consumer imaging, our simple yet effective loss function provides a general mechanism for enhancing edit fidelity and rendering flexibility across existing methods.
>
---
#### [new 053] SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D生成模型中几何控制难的问题，提出SpaceControl方法，实现无需训练的测试时空间控制。支持多种几何输入，兼顾形状保真与视觉质量，结合交互界面实现实时编辑与生成。**

- **链接: [https://arxiv.org/pdf/2512.05343v1](https://arxiv.org/pdf/2512.05343v1)**

> **作者:** Elisabetta Fedele; Francis Engelmann; Ian Huang; Or Litany; Marc Pollefeys; Leonidas Guibas
>
> **备注:** Project page: https://spacecontrol3d.github.io/
>
> **摘要:** Generative methods for 3D assets have recently achieved remarkable progress, yet providing intuitive and precise control over the object geometry remains a key challenge. Existing approaches predominantly rely on text or image prompts, which often fall short in geometric specificity: language can be ambiguous, and images are cumbersome to edit. In this work, we introduce SpaceControl, a training-free test-time method for explicit spatial control of 3D generation. Our approach accepts a wide range of geometric inputs, from coarse primitives to detailed meshes, and integrates seamlessly with modern pre-trained generative models without requiring any additional training. A controllable parameter lets users trade off between geometric fidelity and output realism. Extensive quantitative evaluation and user studies demonstrate that SpaceControl outperforms both training-based and optimization-based baselines in geometric faithfulness while preserving high visual quality. Finally, we present an interactive user interface that enables online editing of superquadrics for direct conversion into textured 3D assets, facilitating practical deployment in creative workflows. Find our project page at https://spacecontrol3d.github.io/
>
---
#### [new 054] VOST-SGG: VLM-Aided One-Stage Spatio-Temporal Scene Graph Generation
- **分类: cs.CV**

- **简介: 该论文研究视频中的时空场景图生成（ST-SGG），旨在建模对象及其时序关系。针对现有单阶段模型查询初始化无语义、仅用视觉特征的问题，提出VOST-SGG，引入双源查询初始化和多模态特征融合，结合视觉语言模型增强语义推理，提升性能。**

- **链接: [https://arxiv.org/pdf/2512.05524v1](https://arxiv.org/pdf/2512.05524v1)**

> **作者:** Chinthani Sugandhika; Chen Li; Deepu Rajan; Basura Fernando
>
> **摘要:** Spatio-temporal scene graph generation (ST-SGG) aims to model objects and their evolving relationships across video frames, enabling interpretable representations for downstream reasoning tasks such as video captioning and visual question answering. Despite recent advancements in DETR-style single-stage ST-SGG models, they still suffer from several key limitations. First, while these models rely on attention-based learnable queries as a core component, these learnable queries are semantically uninformed and instance-agnostically initialized. Second, these models rely exclusively on unimodal visual features for predicate classification. To address these challenges, we propose VOST-SGG, a VLM-aided one-stage ST-SGG framework that integrates the common sense reasoning capabilities of vision-language models (VLMs) into the ST-SGG pipeline. First, we introduce the dual-source query initialization strategy that disentangles what to attend to from where to attend, enabling semantically grounded what-where reasoning. Furthermore, we propose a multi-modal feature bank that fuses visual, textual, and spatial cues derived from VLMs for improved predicate classification. Extensive experiments on the Action Genome dataset demonstrate that our approach achieves state-of-the-art performance, validating the effectiveness of integrating VLM-aided semantic priors and multi-modal features for ST-SGG. We will release the code at https://github.com/LUNAProject22/VOST.
>
---
#### [new 055] Self-Supervised AI-Generated Image Detection: A Camera Metadata Perspective
- **分类: cs.CV**

- **简介: 该论文属AI生成图像检测任务，旨在提升跨模型泛化性。提出自监督方法，利用真实照片的EXIF元数据构建预训练任务，学习摄影内在特征，进而通过单类或二分类模型检测AI图像，增强对未知生成器和图像扰动的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.05651v1](https://arxiv.org/pdf/2512.05651v1)**

> **作者:** Nan Zhong; Mian Zou; Yiran Xu; Zhenxing Qian; Xinpeng Zhang; Baoyuan Wu; Kede Ma
>
> **摘要:** The proliferation of AI-generated imagery poses escalating challenges for multimedia forensics, yet many existing detectors depend on assumptions about the internals of specific generative models, limiting their cross-model applicability. We introduce a self-supervised approach for detecting AI-generated images that leverages camera metadata -- specifically exchangeable image file format (EXIF) tags -- to learn features intrinsic to digital photography. Our pretext task trains a feature extractor solely on camera-captured photographs by classifying categorical EXIF tags (\eg, camera model and scene type) and pairwise-ranking ordinal and continuous EXIF tags (\eg, focal length and aperture value). Using these EXIF-induced features, we first perform one-class detection by modeling the distribution of photographic images with a Gaussian mixture model and flagging low-likelihood samples as AI-generated. We then extend to binary detection that treats the learned extractor as a strong regularizer for a classifier of the same architecture, operating on high-frequency residuals from spatially scrambled patches. Extensive experiments across various generative models demonstrate that our EXIF-induced detectors substantially advance the state of the art, delivering strong generalization to in-the-wild samples and robustness to common benign image perturbations.
>
---
#### [new 056] Your Latent Mask is Wrong: Pixel-Equivalent Latent Compositing for Diffusion Models
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文针对扩散模型中潜在空间修复的掩码不一致问题，提出像素等效潜在融合（PELC）原则。通过DecFormer网络实现高质量掩码融合，支持全分辨率控制与软边缘合成，显著减少边界误差，提升修复效果，无需微调主干模型。**

- **链接: [https://arxiv.org/pdf/2512.05198v1](https://arxiv.org/pdf/2512.05198v1)**

> **作者:** Rowan Bradbury; Dazhi Zhong
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Latent inpainting in diffusion models still relies almost universally on linearly interpolating VAE latents under a downsampled mask. We propose a key principle for compositing image latents: Pixel-Equivalent Latent Compositing (PELC). An equivalent latent compositor should be the same as compositing in pixel space. This principle enables full-resolution mask control and true soft-edge alpha compositing, even though VAEs compress images 8x spatially. Modern VAEs capture global context beyond patch-aligned local structure, so linear latent blending cannot be pixel-equivalent: it produces large artifacts at mask seams and global degradation and color shifts. We introduce DecFormer, a 7.7M-parameter transformer that predicts per-channel blend weights and an off-manifold residual correction to realize mask-consistent latent fusion. DecFormer is trained so that decoding after fusion matches pixel-space alpha compositing, is plug-compatible with existing diffusion pipelines, requires no backbone finetuning and adds only 0.07% of FLUX.1-Dev's parameters and 3.5% FLOP overhead. On the FLUX.1 family, DecFormer restores global color consistency, soft-mask support, sharp boundaries, and high-fidelity masking, reducing error metrics around edges by up to 53% over standard mask interpolation. Used as an inpainting prior, a lightweight LoRA on FLUX.1-Dev with DecFormer achieves fidelity comparable to FLUX.1-Fill, a fully finetuned inpainting model. While we focus on inpainting, PELC is a general recipe for pixel-equivalent latent editing, as we demonstrate on a complex color-correction task.
>
---
#### [new 057] World Models That Know When They Don't Know: Controllable Video Generation with Calibrated Uncertainty
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究可控视频生成中的不确定性量化，旨在解决模型幻觉问题。提出C3方法，通过评分规则、潜在空间不确定性估计与像素级映射，实现细粒度、校准的置信度预测，支持分布内校准与异常检测。**

- **链接: [https://arxiv.org/pdf/2512.05927v1](https://arxiv.org/pdf/2512.05927v1)**

> **作者:** Zhiting Mei; Tenny Yin; Micah Baker; Ola Shorinwa; Anirudha Majumdar
>
> **摘要:** Recent advances in generative video models have led to significant breakthroughs in high-fidelity video synthesis, specifically in controllable video generation where the generated video is conditioned on text and action inputs, e.g., in instruction-guided video editing and world modeling in robotics. Despite these exceptional capabilities, controllable video models often hallucinate - generating future video frames that are misaligned with physical reality - which raises serious concerns in many tasks such as robot policy evaluation and planning. However, state-of-the-art video models lack the ability to assess and express their confidence, impeding hallucination mitigation. To rigorously address this challenge, we propose C3, an uncertainty quantification (UQ) method for training continuous-scale calibrated controllable video models for dense confidence estimation at the subpatch level, precisely localizing the uncertainty in each generated video frame. Our UQ method introduces three core innovations to empower video models to estimate their uncertainty. First, our method develops a novel framework that trains video models for correctness and calibration via strictly proper scoring rules. Second, we estimate the video model's uncertainty in latent space, avoiding training instability and prohibitive training costs associated with pixel-space approaches. Third, we map the dense latent-space uncertainty to interpretable pixel-level uncertainty in the RGB space for intuitive visualization, providing high-resolution uncertainty heatmaps that identify untrustworthy regions. Through extensive experiments on large-scale robot learning datasets (Bridge and DROID) and real-world evaluations, we demonstrate that our method not only provides calibrated uncertainty estimates within the training distribution, but also enables effective out-of-distribution detection.
>
---
#### [new 058] IE2Video: Adapting Pretrained Diffusion Models for Event-Based Video Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出IE2Video任务，旨在利用稀疏RGB关键帧和事件相机数据重建连续RGB视频，解决传统相机功耗高、事件流难直接用于视频应用的问题。作者探索两种方法，发现基于预训练扩散模型的方案显著提升重建质量。**

- **链接: [https://arxiv.org/pdf/2512.05240v1](https://arxiv.org/pdf/2512.05240v1)**

> **作者:** Dmitrii Torbunov; Onur Okuducu; Yi Huang; Odera Dim; Rebecca Coles; Yonggang Cui; Yihui Ren
>
> **摘要:** Continuous video monitoring in surveillance, robotics, and wearable systems faces a fundamental power constraint: conventional RGB cameras consume substantial energy through fixed-rate capture. Event cameras offer sparse, motion-driven sensing with low power consumption, but produce asynchronous event streams rather than RGB video. We propose a hybrid capture paradigm that records sparse RGB keyframes alongside continuous event streams, then reconstructs full RGB video offline -- reducing capture power consumption while maintaining standard video output for downstream applications. We introduce the Image and Event to Video (IE2Video) task: reconstructing RGB video sequences from a single initial frame and subsequent event camera data. We investigate two architectural strategies: adapting an autoregressive model (HyperE2VID) for RGB generation, and injecting event representations into a pretrained text-to-video diffusion model (LTX) via learned encoders and low-rank adaptation. Our experiments demonstrate that the diffusion-based approach achieves 33\% better perceptual quality than the autoregressive baseline (0.283 vs 0.422 LPIPS). We validate our approach across three event camera datasets (BS-ERGB, HS-ERGB far/close) at varying sequence lengths (32-128 frames), demonstrating robust cross-dataset generalization with strong performance on unseen capture configurations.
>
---
#### [new 059] AQUA-Net: Adaptive Frequency Fusion and Illumination Aware Network for Underwater Image Enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究水下图像增强，旨在解决颜色失真、低对比度和计算复杂度高的问题。提出AQUA-Net模型，融合频域与光照信息，结合残差编解码结构，在少参数下实现高效增强，并发布真实深海水下视频数据集用于验证。**

- **链接: [https://arxiv.org/pdf/2512.05960v1](https://arxiv.org/pdf/2512.05960v1)**

> **作者:** Munsif Ali; Najmul Hassan; Lucia Ventura; Davide Di Bari; Simonepietro Canese
>
> **摘要:** Underwater images often suffer from severe color distortion, low contrast, and a hazy appearance due to wavelength-dependent light absorption and scattering. Simultaneously, existing deep learning models exhibit high computational complexity, which limits their practical deployment for real-time underwater applications. To address these challenges, this paper presents a novel underwater image enhancement model, called Adaptive Frequency Fusion and Illumination Aware Network (AQUA-Net). It integrates a residual encoder decoder with dual auxiliary branches, which operate in the frequency and illumination domains. The frequency fusion encoder enriches spatial representations with frequency cues from the Fourier domain and preserves fine textures and structural details. Inspired by Retinex, the illumination-aware decoder performs adaptive exposure correction through a learned illumination map that separates reflectance from lighting effects. This joint spatial, frequency, and illumination design enables the model to restore color balance, visual contrast, and perceptual realism under diverse underwater conditions. Additionally, we present a high-resolution, real-world underwater video-derived dataset from the Mediterranean Sea, which captures challenging deep-sea conditions with realistic visual degradations to enable robust evaluation and development of deep learning models. Extensive experiments on multiple benchmark datasets show that AQUA-Net performs on par with SOTA in both qualitative and quantitative evaluations while using less number of parameters. Ablation studies further confirm that the frequency and illumination branches provide complementary contributions that improve visibility and color representation. Overall, the proposed model shows strong generalization capability and robustness, and it provides an effective solution for real-world underwater imaging applications.
>
---
#### [new 060] Performance Evaluation of Deep Learning for Tree Branch Segmentation in Autonomous Forestry Systems
- **分类: cs.CV**

- **简介: 该论文研究无人机林业中树杈分割任务，解决多分辨率下精度与效率平衡问题。作者评估22种深度学习模型，提出TS-IoU和CPR等指标，建立多分辨率基准，开源代码供复现。**

- **链接: [https://arxiv.org/pdf/2512.05418v1](https://arxiv.org/pdf/2512.05418v1)**

> **作者:** Yida Lin; Bing Xue; Mengjie Zhang; Sam Schofield; Richard Green
>
> **摘要:** UAV-based autonomous forestry operations require rapid and precise tree branch segmentation for safe navigation and automated pruning across varying pixel resolutions and operational conditions. We evaluate different deep learning methods at three resolutions (256x256, 512x512, 1024x1024) using the Urban Street Tree Dataset, employing standard metrics (IoU, Dice) and specialized measures including Thin Structure IoU (TS-IoU) and Connectivity Preservation Rate (CPR). Among 22 configurations tested, U-Net with MiT-B4 backbone achieves strong performance at 256x256. At 512x512, MiT-B4 leads in IoU, Dice, TS-IoU, and Boundary-F1. At 1024x1024, U-Net+MiT-B3 shows the best validation performance for IoU/Dice and precision, while U-Net++ excels in boundary quality. PSPNet provides the most efficient option (2.36/9.43/37.74 GFLOPs) with 25.7/19.6/11.8 percentage point IoU reductions compared to top performers at respective resolutions. These results establish multi-resolution benchmarks for accuracy-efficiency trade-offs in embedded forestry systems. Implementation is available at https://github.com/BennyLinntu/PerformanceTreeBranchSegmentation.
>
---
#### [new 061] University Building Recognition Dataset in Thailand for the mission-oriented IoT sensor system
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对边缘设备上的协作学习任务，提出构建特定于任务的数据集。为支持无线联邦学习在校园建筑识别中的应用，作者创建了泰国朱拉隆功大学建筑识别数据集（CUBR），并验证了其在WAFL-ViT框架下优于独立训练的性能。**

- **链接: [https://arxiv.org/pdf/2512.05468v1](https://arxiv.org/pdf/2512.05468v1)**

> **作者:** Takara Taniguchi; Yudai Ueda; Atsuya Muramatsu; Kohki Hashimoto; Ryo Yagi; Hideya Ochiai; Chaodit Aswakul
>
> **摘要:** Many industrial sectors have been using of machine learning at inference mode on edge devices. Future directions show that training on edge devices is promising due to improvements in semiconductor performance. Wireless Ad Hoc Federated Learning (WAFL) has been proposed as a promising approach for collaborative learning with device-to-device communication among edges. In particular, WAFL with Vision Transformer (WAFL-ViT) has been tested on image recognition tasks with the UTokyo Building Recognition Dataset (UTBR). Since WAFL-ViT is a mission-oriented sensor system, it is essential to construct specific datasets by each mission. In our work, we have developed the Chulalongkorn University Building Recognition Dataset (CUBR), which is specialized for Chulalongkorn University as a case study in Thailand. Additionally, our results also demonstrate that training on WAFL scenarios achieves better accuracy than self-training scenarios. Dataset is available in https://github.com/jo2lxq/wafl/.
>
---
#### [new 062] Probing the effectiveness of World Models for Spatial Reasoning through Test-time Scaling
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉语言模型在空间推理中的局限，分析测试时扩展方法中验证器的有效性，发现现有验证器存在偏差问题。为此提出ViSA框架，通过可验证的细粒度声明提升多视角推理效果，揭示了当前世界模型在精细推理上的瓶颈。**

- **链接: [https://arxiv.org/pdf/2512.05809v1](https://arxiv.org/pdf/2512.05809v1)**

> **作者:** Saurav Jha; M. Jehanzeb Mirza; Wei Lin; Shiqi Yang; Sarath Chandar
>
> **备注:** Extended abstract at World Modeling Workshop 2026
>
> **摘要:** Vision-Language Models (VLMs) remain limited in spatial reasoning tasks that require multi-view understanding and embodied perspective shifts. Recent approaches such as MindJourney attempt to mitigate this gap through test-time scaling where a world model imagines action-conditioned trajectories and a heuristic verifier selects helpful views from such trajectories. In this work, we systematically examine how such test-time verifiers behave across benchmarks, uncovering both their promise and their pitfalls. Our uncertainty-based analyses show that MindJourney's verifier provides little meaningful calibration, and that random scoring often reduces answer entropy equally well, thus exposing systematic action biases and unreliable reward signals. To mitigate these, we introduce a Verification through Spatial Assertions (ViSA) framework that grounds the test-time reward in verifiable, frame-anchored micro-claims. This principled verifier consistently improves spatial reasoning on the SAT-Real benchmark and corrects trajectory-selection biases through more balanced exploratory behavior. However, on the challenging MMSI-Bench, none of the verifiers, including ours, achieve consistent scaling, suggesting that the current world models form an information bottleneck where imagined views fail to enrich fine-grained reasoning. Together, these findings chart the bad, good, and ugly aspects of test-time verification for world-model-based reasoning. Our code is available at https://github.com/chandar-lab/visa-for-mindjourney.
>
---
#### [new 063] Spatiotemporal Satellite Image Downscaling with Transfer Encoders and Autoregressive Generative Models
- **分类: cs.CV; cs.LG; stat.ML**

- **简介: 该论文研究卫星图像时空超分辨率重建，旨在从低分辨率遥感数据生成高分辨率图像。提出一种结合迁移编码器与扩散生成模型的框架，利用预训练U-Net提取时空特征，提升降尺度精度与物理一致性，实现长期环境监测中的高效影像重建。**

- **链接: [https://arxiv.org/pdf/2512.05139v1](https://arxiv.org/pdf/2512.05139v1)**

> **作者:** Yang Xiang; Jingwen Zhong; Yige Yan; Petros Koutrakis; Eric Garshick; Meredith Franklin
>
> **摘要:** We present a transfer-learning generative downscaling framework to reconstruct fine resolution satellite images from coarse scale inputs. Our approach combines a lightweight U-Net transfer encoder with a diffusion-based generative model. The simpler U-Net is first pretrained on a long time series of coarse resolution data to learn spatiotemporal representations; its encoder is then frozen and transferred to a larger downscaling model as physically meaningful latent features. Our application uses NASA's MERRA-2 reanalysis as the low resolution source domain (50 km) and the GEOS-5 Nature Run (G5NR) as the high resolution target (7 km). Our study area included a large area in Asia, which was made computationally tractable by splitting into two subregions and four seasons. We conducted domain similarity analysis using Wasserstein distances confirmed minimal distributional shift between MERRA-2 and G5NR, validating the safety of parameter frozen transfer. Across seasonal regional splits, our model achieved excellent performance (R2 = 0.65 to 0.94), outperforming comparison models including deterministic U-Nets, variational autoencoders, and prior transfer learning baselines. Out of data evaluations using semivariograms, ACF/PACF, and lag-based RMSE/R2 demonstrated that the predicted downscaled images preserved physically consistent spatial variability and temporal autocorrelation, enabling stable autoregressive reconstruction beyond the G5NR record. These results show that transfer enhanced diffusion models provide a robust and physically coherent solution for downscaling a long time series of coarse resolution images with limited training periods. This advancement has significant implications for improving environmental exposure assessment and long term environmental monitoring.
>
---
#### [new 064] DEAR: Dataset for Evaluating the Aesthetics of RenderingDEAR: Dataset for Evaluating the Aesthetics of Rendering
- **分类: cs.CV**

- **简介: 该论文提出DEAR数据集，旨在解决渲染美学评估问题。针对现有图像质量评估忽略主观风格偏好的局限，构建基于人类偏好的大规模标注数据集，支持渲染美学评价、风格偏好预测等任务。**

- **链接: [https://arxiv.org/pdf/2512.05209v1](https://arxiv.org/pdf/2512.05209v1)**

> **作者:** Vsevolod Plohotnuk; Artyom Panshin; Nikola Banić; Simone Bianco; Michael Freeman; Egor Ershov
>
> **摘要:** Traditional Image Quality Assessment~(IQA) focuses on quantifying technical degradations such as noise, blur, or compression artifacts, using both full-reference and no-reference objective metrics. However, evaluation of rendering aesthetics, a growing domain relevant to photographic editing, content creation, and AI-generated imagery, remains underexplored due to the lack of datasets that reflect the inherently subjective nature of style preference. In this work, a novel benchmark dataset designed to model human aesthetic judgments of image rendering styles is introduced: the Dataset for Evaluating the Aesthetics of Rendering (DEAR). Built upon the MIT-Adobe FiveK dataset, DEAR incorporates pairwise human preference scores collected via large-scale crowdsourcing, with each image pair evaluated by 25 distinct human evaluators with a total of 13,648 of them participating overall. These annotations capture nuanced, context-sensitive aesthetic preferences, enabling the development and evaluation of models that go beyond traditional distortion-based IQA, focusing on a new task: Evaluation of Aesthetics of Rendering (EAR). The data collection pipeline is described, human voting patterns are analyzed, and multiple use cases are outlined, including style preference prediction, aesthetic benchmarking, and personalized aesthetic modeling. To the best of the authors' knowledge, DEAR is the first dataset to systematically address image aesthetics of rendering assessment grounded in subjective human preferences. A subset of 100 images with markup for them is published on HuggingFace (huggingface.co/datasets/vsevolodpl/DEAR).
>
---
#### [new 065] Fine-tuning an ECG Foundation Model to Predict Coronary CT Angiography Outcomes
- **分类: cs.CV; cs.AI**

- **简介: 该论文旨在通过微调ECG基础模型，预测冠状动脉CT血管造影（CCTA）中的严重狭窄情况。针对冠心病筛查中CCTA的局限性，提出一种可解释的AI-ECG方法，实现对四支主要冠状动脉病变的无创、精准识别，并验证其稳定性和临床适用性。**

- **链接: [https://arxiv.org/pdf/2512.05136v1](https://arxiv.org/pdf/2512.05136v1)**

> **作者:** Yujie Xiao; Gongzhen Tang; Deyun Zhang; Jun Li; Guangkun Nie; Haoyu Wang; Shun Huang; Tong Liu; Qinghao Zhao; Kangyin Chen; Shenda Hong
>
> **摘要:** Coronary artery disease (CAD) remains a major global health burden. Accurate identification of the culprit vessel and assessment of stenosis severity are essential for guiding individualized therapy. Although coronary CT angiography (CCTA) is the first-line non-invasive modality for CAD diagnosis, its dependence on high-end equipment, radiation exposure, and strict patient cooperation limits large-scale use. With advances in artificial intelligence (AI) and the widespread availability of electrocardiography (ECG), AI-ECG offers a promising alternative for CAD screening. In this study, we developed an interpretable AI-ECG model to predict severe or complete stenosis of the four major coronary arteries on CCTA. On the internal validation set, the model's AUCs for the right coronary artery (RCA), left main coronary artery (LM), left anterior descending artery (LAD), and left circumflex artery (LCX) were 0.794, 0.818, 0.744, and 0.755, respectively; on the external validation set, the AUCs reached 0.749, 0.971, 0.667, and 0.727, respectively. Performance remained stable in a clinically normal-ECG subset, indicating robustness beyond overt ECG abnormalities. Subgroup analyses across demographic and acquisition-time strata further confirmed model stability. Risk stratification based on vessel-specific incidence thresholds showed consistent separation on calibration and cumulative event curves. Interpretability analyses revealed distinct waveform differences between high- and low-risk groups, highlighting key electrophysiological regions contributing to model decisions and offering new insights into the ECG correlates of coronary stenosis.
>
---
#### [new 066] A Comparative Study on Synthetic Facial Data Generation Techniques for Face Recognition
- **分类: cs.CV**

- **简介: 该论文属人脸识别任务，旨在解决数据隐私、偏见和数据不足问题。通过比较扩散模型、GANs和3D模型生成的合成人脸数据在多个指标上的表现，评估其在人脸识别中的有效性，推动合成数据应用研究。**

- **链接: [https://arxiv.org/pdf/2512.05928v1](https://arxiv.org/pdf/2512.05928v1)**

> **作者:** Pedro Vidal; Bernardo Biesseck; Luiz E. L. Coelho; Roger Granada; David Menotti
>
> **备注:** 18 pages, 17 figures
>
> **摘要:** Facial recognition has become a widely used method for authentication and identification, with applications for secure access and locating missing persons. Its success is largely attributed to deep learning, which leverages large datasets and effective loss functions to learn discriminative features. Despite these advances, facial recognition still faces challenges in explainability, demographic bias, privacy, and robustness to aging, pose variations, lighting changes, occlusions, and facial expressions. Privacy regulations have also led to the degradation of several datasets, raising legal, ethical, and privacy concerns. Synthetic facial data generation has been proposed as a promising solution. It mitigates privacy issues, enables experimentation with controlled facial attributes, alleviates demographic bias, and provides supplementary data to improve models trained on real data. This study compares the effectiveness of synthetic facial datasets generated using different techniques in facial recognition tasks. We evaluate accuracy, rank-1, rank-5, and the true positive rate at a false positive rate of 0.01% on eight leading datasets, offering a comparative analysis not extensively explored in the literature. Results demonstrate the ability of synthetic data to capture realistic variations while emphasizing the need for further research to close the performance gap with real data. Techniques such as diffusion models, GANs, and 3D models show substantial progress; however, challenges remain.
>
---
#### [new 067] LoC-Path: Learning to Compress for Pathology Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文研究病理全切片图像（WSI）的多模态大语言模型高效建模。针对现有方法计算开销大的问题，提出LoC-Path框架，通过压缩冗余图像块、保留关键信息，实现高效推理，兼顾性能与计算成本。**

- **链接: [https://arxiv.org/pdf/2512.05391v1](https://arxiv.org/pdf/2512.05391v1)**

> **作者:** Qingqiao Hu; Weimin Lyu; Meilong Xu; Kehan Qi; Xiaoling Hu; Saumya Gupta; Jiawei Zhou; Chao Chen
>
> **备注:** 20 pages
>
> **摘要:** Whole Slide Image (WSI) understanding is fundamentally challenging due to its gigapixel scale and the extreme sparsity of diagnostically relevant regions. Unlike human experts who primarily rely on key areas to arrive at a diagnosis, existing slide-level multimodal large language models (MLLMs) for pathology rely on heavy slide-level encoders that process thousands of patch features in a brute-force manner, resulting in excessive computational cost. In this work, we revisit the WSI-language modeling paradigm and show that tile-level features exhibit strong global and local redundancy, whereas only a small subset of tiles are truly task-relevant. Motivated by this observation, we introduce an efficient MLLM framework, called LoC-Path, that replaces the expensive slide-level encoder with redundancy-reducing modules. We first design a Sparse Token Merger (STM) and an MAE-pretrained resampler to remove local redundancy and compress globally redundant tile tokens into a compact slide-level representation set. We then propose a Cross-Attention Routing Adapter (CARA) and a Token Importance Scorer (TIS) to integrate the compressed visual representation with the language model in a computation-efficient manner. Extensive experiments demonstrate that our approach achieves performance comparable to existing state-of-the-art whole-slide MLLMs, while requiring significantly lower computation and memory.
>
---
#### [new 068] Fast SceneScript: Accurate and Efficient Structured Language Model via Multi-Token Prediction
- **分类: cs.CV**

- **简介: 该论文聚焦3D场景布局估计任务，旨在解决现有语言模型方法因自回归逐token生成导致的推理慢问题。作者提出Fast SceneScript，采用多token预测加速推理，结合改进的自推测解码与置信度引导机制提升准确性，并设计轻量级结构降低参数开销。**

- **链接: [https://arxiv.org/pdf/2512.05597v1](https://arxiv.org/pdf/2512.05597v1)**

> **作者:** Ruihong Yin; Xuepeng Shi; Oleksandr Bailo; Marco Manfredi; Theo Gevers
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Recent perception-generalist approaches based on language models have achieved state-of-the-art results across diverse tasks, including 3D scene layout estimation, via unified architecture and interface. However, these approaches rely on autoregressive next-token prediction, which is inherently slow. In this work, we introduce Fast SceneScript, a novel structured language model for accurate and efficient 3D scene layout estimation. Our method employs multi-token prediction (MTP) to reduce the number of autoregressive iterations and significantly accelerate inference. While MTP improves speed, unreliable token predictions can significantly reduce accuracy. To filter out unreliable tokens, we adapt self-speculative decoding (SSD) for structured language models and introduce confidence-guided decoding (CGD) with an improved scoring mechanism for token reliability. Furthermore, we design a parameter-efficient mechanism that reduces the parameter overhead of MTP. Extensive experiments on the ASE and Structured3D benchmarks demonstrate that Fast SceneScript can generate up to 9 tokens per decoder inference step without compromising accuracy, while adding only $\sim7.5\%$ additional parameters.
>
---
#### [new 069] Synset Signset Germany: a Synthetic Dataset for German Traffic Sign Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对交通标志识别任务，提出合成数据集Synset Signset Germany。结合GAN纹理生成与解析式场景建模，生成含真实磨损和光照变化的德国交通标志图像，支持XAI与鲁棒性测试，覆盖211类标志并提供丰富标注与元数据。**

- **链接: [https://arxiv.org/pdf/2512.05936v1](https://arxiv.org/pdf/2512.05936v1)**

> **作者:** Anne Sielemann; Lena Loercher; Max-Lion Schumacher; Stefan Wolf; Masoud Roschani; Jens Ziehn
>
> **备注:** 8 pages, 8 figures, 3 tables
>
> **摘要:** In this paper, we present a synthesis pipeline and dataset for training / testing data in the task of traffic sign recognition that combines the advantages of data-driven and analytical modeling: GAN-based texture generation enables data-driven dirt and wear artifacts, rendering unique and realistic traffic sign surfaces, while the analytical scene modulation achieves physically correct lighting and allows detailed parameterization. In particular, the latter opens up applications in the context of explainable AI (XAI) and robustness tests due to the possibility of evaluating the sensitivity to parameter changes, which we demonstrate with experiments. Our resulting synthetic traffic sign recognition dataset Synset Signset Germany contains a total of 105500 images of 211 different German traffic sign classes, including newly published (2020) and thus comparatively rare traffic signs. In addition to a mask and a segmentation image, we also provide extensive metadata including the stochastically selected environment and imaging effect parameters for each image. We evaluate the degree of realism of Synset Signset Germany on the real-world German Traffic Sign Recognition Benchmark (GTSRB) and in comparison to CATERED, a state-of-the-art synthetic traffic sign recognition dataset.
>
---
#### [new 070] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割中Transformer解码器对边缘细节和空间连续性建模不足的问题，提出一种新型解码框架，融合方向、频域-空间与结构注意力机制，提升肿瘤分割与器官边界提取的精度和泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.05494v1](https://arxiv.org/pdf/2512.05494v1)**

> **作者:** Fan Zhang; Zhiwei Gu; Hua Wang
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** To address the limitations of Transformer decoders in capturing edge details, recognizing local textures and modeling spatial continuity, this paper proposes a novel decoder framework specifically designed for medical image segmentation, comprising three core modules. First, the Adaptive Cross-Fusion Attention (ACFA) module integrates channel feature enhancement with spatial attention mechanisms and introduces learnable guidance in three directions (planar, horizontal, and vertical) to enhance responsiveness to key regions and structural orientations. Second, the Triple Feature Fusion Attention (TFFA) module fuses features from Spatial, Fourier and Wavelet domains, achieving joint frequency-spatial representation that strengthens global dependency and structural modeling while preserving local information such as edges and textures, making it particularly effective in complex and blurred boundary scenarios. Finally, the Structural-aware Multi-scale Masking Module (SMMM) optimizes the skip connections between encoder and decoder by leveraging multi-scale context and structural saliency filtering, effectively reducing feature redundancy and improving semantic interaction quality. Working synergistically, these modules not only address the shortcomings of traditional decoders but also significantly enhance performance in high-precision tasks such as tumor segmentation and organ boundary extraction, improving both segmentation accuracy and model generalization. Experimental results demonstrate that this framework provides an efficient and practical solution for medical image segmentation.
>
---
#### [new 071] ShaRP: SHAllow-LayeR Pruning for Video Large Language Models Acceleration
- **分类: cs.CV**

- **简介: 该论文针对视频大语言模型（VLLM）推理中视觉token计算量大的问题，提出在浅层解码器进行有效剪枝的ShaRP框架，通过因果掩码、去偏置和去重策略提升注意力机制下的token选择，实现高效压缩与稳定性能。**

- **链接: [https://arxiv.org/pdf/2512.05385v1](https://arxiv.org/pdf/2512.05385v1)**

> **作者:** Yingjie Xia; Tao Liu; Jinglei Shi; Qingsong Xie; Heng Guo; Jian Yang; Xi Wang
>
> **摘要:** Video Large Language Models (VLLMs) face the challenge of high computational load during the pre-filling stage due to the processing of an enormous number of visual tokens. Although attention-based pruning methods are widely used to accelerate inference, trials at early decoder layers often result in significant performance degradation, especially under high compression rates. We argue that while attention-based pruning inherently holds the potential to identify the most relevant visual tokens, its effectiveness in shallow decoder layers is limited by factors such as positional encoding bias and insufficient information interaction. In this paper, we propose an improved attention-based pruning framework, termed ShaRP, that integrates segment-aware causal masking, positional debiasing, and token deduplication for enhanced token selection. It enables effective pruning at shallow layers while maintaining stable performance under high compression rates without retraining. Extensive experiments demonstrate that ShaRP achieves competitive performance across multiple video understanding benchmarks, establishing a new paradigm for accelerating VLLM inference.
>
---
#### [new 072] FNOPT: Resolution-Agnostic, Self-Supervised Cloth Simulation using Meta-Optimization with Fourier Neural Operators
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出FNOpt，用于解决布料模拟中跨分辨率泛化差、依赖大量标注数据的问题。其将时间积分建模为优化问题，采用傅里叶神经算子构建自监督、分辨率无关的神经优化器，实现稳定、高精度的布料动态模拟。**

- **链接: [https://arxiv.org/pdf/2512.05762v1](https://arxiv.org/pdf/2512.05762v1)**

> **作者:** Ruochen Chen; Thuy Tran; Shaifali Parashar
>
> **备注:** Accepted for WACV
>
> **摘要:** We present FNOpt, a self-supervised cloth simulation framework that formulates time integration as an optimization problem and trains a resolution-agnostic neural optimizer parameterized by a Fourier neural operator (FNO). Prior neural simulators often rely on extensive ground truth data or sacrifice fine-scale detail, and generalize poorly across resolutions and motion patterns. In contrast, FNOpt learns to simulate physically plausible cloth dynamics and achieves stable and accurate rollouts across diverse mesh resolutions and motion patterns without retraining. Trained only on a coarse grid with physics-based losses, FNOpt generalizes to finer resolutions, capturing fine-scale wrinkles and preserving rollout stability. Extensive evaluations on a benchmark cloth simulation dataset demonstrate that FNOpt outperforms prior learning-based approaches in out-of-distribution settings in both accuracy and robustness. These results position FNO-based meta-optimization as a compelling alternative to previous neural simulators for cloth, thus reducing the need for curated data and improving cross-resolution reliability.
>
---
#### [new 073] UG-FedDA: Uncertainty-Guided Federated Domain Adaptation for Multi-Center Alzheimer's Disease Detection
- **分类: cs.CV**

- **简介: 该论文提出UG-FedDA，用于多中心阿尔茨海默病检测，属医学图像分类任务。旨在解决跨站点异质性和隐私保护下的模型泛化问题，结合不确定性量化与联邦域自适应，提升多中心MRI数据分类性能。**

- **链接: [https://arxiv.org/pdf/2512.05814v1](https://arxiv.org/pdf/2512.05814v1)**

> **作者:** Fubao Zhu; Zhanyuan Jia; Zhiguo Wang; Huan Huang; Danyang Sun; Chuang Han; Yanting Li; Jiaofen Nan; Chen Zhao; Weihua Zhou
>
> **备注:** The code is already available on GitHub: https://github.com/chenzhao2023/UG_FADDA_AlzhemiersClassification
>
> **摘要:** Alzheimer's disease (AD) is an irreversible neurodegenerative disorder, and early diagnosis is critical for timely intervention. However, most existing classification frameworks face challenges in multicenter studies, as they often neglect inter-site heterogeneity and lack mechanisms to quantify uncertainty, which limits their robustness and clinical applicability. To address these issues, we proposed Uncertainty-Guided Federated Domain Adaptation (UG-FedDA), a novel multicenter AD classification framework that integrates uncertainty quantification (UQ) with federated domain adaptation to handle cross-site structure magnetic resonance imaging (MRI) heterogeneity under privacy constraints. Our approach extracts multi-template region-of-interest (RoI) features using a self-attention transformer, capturing both regional representations and their interactions. UQ is integrated to guide feature alignment, mitigating source-target distribution shifts by down-weighting uncertain samples. Experiments are conducted on three public datasets: the Alzheimer's Disease Neuroimaging Initiative (ADNI), the Australian Imaging, Biomarkers and Lifestyle study (AIBL), and the Open Access Series of Imaging Studies (OASIS). UG-FedDA achieved consistent cross-domain improvements in accuracy, sensitivity, and area under the ROC curve across three classification tasks: AD vs. normal controls (NC), mild cognitive impairment (MCI) vs. AD, and NC vs. MCI. For NC vs. AD, UG-FedDA achieves accuracies of 90.54%, 89.04%, and 77.78% on ADNI, AIBL and OASIS datasets, respectively. For MCI vs. AD, accuracies are 80.20% (ADNI), 71.91% (AIBL), and 79.73% (OASIS). For NC vs. MCI, results are 76.87% (ADNI), 73.91% (AIBL), and 83.73% (OASIS). These results demonstrate that the proposed framework not only adapts efficiently across multiple sites but also preserves strict privacy.
>
---
#### [new 074] OWL: Unsupervised 3D Object Detection by Occupancy Guided Warm-up and Large Model Priors Reasoning
- **分类: cs.CV**

- **简介: 该论文研究无监督3D目标检测，旨在解决伪标签初始错误导致训练偏差的问题。提出OWL方法，通过占用引导预热、大模型先验推理和自适应重加权策略，提升伪标签质量与检测性能。**

- **链接: [https://arxiv.org/pdf/2512.05698v1](https://arxiv.org/pdf/2512.05698v1)**

> **作者:** Xusheng Guo; Wanfa Zhang; Shijia Zhao; Qiming Xia; Xiaolong Xie; Mingming Wang; Hai Wu; Chenglu Wen
>
> **备注:** The 40th Annual AAAI Conference on Artificial Intelligence
>
> **摘要:** Unsupervised 3D object detection leverages heuristic algorithms to discover potential objects, offering a promising route to reduce annotation costs in autonomous driving. Existing approaches mainly generate pseudo labels and refine them through self-training iterations. However, these pseudo-labels are often incorrect at the beginning of training, resulting in misleading the optimization process. Moreover, effectively filtering and refining them remains a critical challenge. In this paper, we propose OWL for unsupervised 3D object detection by occupancy guided warm-up and large-model priors reasoning. OWL first employs an Occupancy Guided Warm-up (OGW) strategy to initialize the backbone weight with spatial perception capabilities, mitigating the interference of incorrect pseudo-labels on network convergence. Furthermore, OWL introduces an Instance-Cued Reasoning (ICR) module that leverages the prior knowledge of large models to assess pseudo-label quality, enabling precise filtering and refinement. Finally, we design a Weight-adapted Self-training (WAS) strategy to dynamically re-weight pseudo-labels, improving the performance through self-training. Extensive experiments on Waymo Open Dataset (WOD) and KITTI demonstrate that OWL outperforms state-of-the-art unsupervised methods by over 15.0% mAP, revealing the effectiveness of our method.
>
---
#### [new 075] Underwater Image Reconstruction Using a Swin Transformer-Based Generator and PatchGAN Discriminator
- **分类: cs.CV**

- **简介: 该论文针对水下图像存在的颜色失真、低对比度和雾化问题，提出一种基于Swin Transformer生成器与PatchGAN判别器的重建方法。属于图像增强任务，利用Transformer捕获全局依赖，实现更优的颜色校正与细节恢复，显著提升水下图像质量。**

- **链接: [https://arxiv.org/pdf/2512.05866v1](https://arxiv.org/pdf/2512.05866v1)**

> **作者:** Md. Mahbub Hasan Akash; Aria Tasnim Mridula; Sheekar Banerjee; Ishtiak Al Mamoon
>
> **备注:** This paper has been accepted for presentation at the IEEE 28th International Conference on Computer and Information Technology (ICCIT), December 2025
>
> **摘要:** Underwater imaging is essential for marine exploration, environmental monitoring, and infrastructure inspection. However, water causes severe image degradation through wavelength-dependent absorption and scattering, resulting in color distortion, low contrast, and haze effects. Traditional reconstruction methods and convolutional neural network-based approaches often fail to adequately address these challenges due to limited receptive fields and inability to model global dependencies. This paper presented a novel deep learning framework that integrated a Swin Transformer architecture within a generative adversarial network (GAN) for underwater image reconstruction. Our generator employed a U-Net structure with Swin Transformer blocks to capture both local features and long-range dependencies crucial for color correction across entire images. A PatchGAN discriminator provided adversarial training to ensure high-frequency detail preservation. We trained and evaluated our model on the EUVP dataset, which contains paired underwater images of varying quality. Quantitative results demonstrate stateof-the-art performance with PSNR of 24.76 dB and SSIM of 0.89, representing significant improvements over existing methods. Visual results showed effective color balance restoration, contrast improvement, and haze reduction. An ablation study confirms the superiority of our Swin Transformer designed over convolutional alternatives. The proposed method offers robust underwater image reconstruction suitable for various marine applications.
>
---
#### [new 076] VRSA: Jailbreaking Multimodal Large Language Models through Visual Reasoning Sequential Attack
- **分类: cs.CV**

- **简介: 该论文属于多模态安全任务，旨在解决多模态大模型的视觉越狱攻击问题。提出VRSA方法，通过分解有害文本为语义连贯的序列图像，结合场景优化与图文一致性对齐，诱导模型逐步暴露有害意图，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2512.05853v1](https://arxiv.org/pdf/2512.05853v1)**

> **作者:** Shiji Zhao; Shukun Xiong; Yao Huang; Yan Jin; Zhenyu Wu; Jiyang Guan; Ranjie Duan; Jialing Tao; Hui Xue; Xingxing Wei
>
> **摘要:** Multimodal Large Language Models (MLLMs) are widely used in various fields due to their powerful cross-modal comprehension and generation capabilities. However, more modalities bring more vulnerabilities to being utilized for jailbreak attacks, which induces MLLMs to output harmful content. Due to the strong reasoning ability of MLLMs, previous jailbreak attacks try to explore reasoning safety risk in text modal, while similar threats have been largely overlooked in the visual modal. To fully evaluate potential safety risks in the visual reasoning task, we propose Visual Reasoning Sequential Attack (VRSA), which induces MLLMs to gradually externalize and aggregate complete harmful intent by decomposing the original harmful text into several sequentially related sub-images. In particular, to enhance the rationality of the scene in the image sequence, we propose Adaptive Scene Refinement to optimize the scene most relevant to the original harmful query. To ensure the semantic continuity of the generated image, we propose Semantic Coherent Completion to iteratively rewrite each sub-text combined with contextual information in this scene. In addition, we propose Text-Image Consistency Alignment to keep the semantical consistency. A series of experiments demonstrates that the VRSA can achieve a higher attack success rate compared with the state-of-the-art jailbreak attack methods on both the open-source and closed-source MLLMs such as GPT-4o and Claude-4.5-Sonnet.
>
---
#### [new 077] Phase-OTDR Event Detection Using Image-Based Data Transformation and Deep Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究光纤中六类事件的检测任务，旨在提升Phase-OTDR系统数据分析能力。通过将一维信号转换为图像，结合深度学习模型实现高效分类，准确率超98%，并公开数据与代码以促进后续研究。**

- **链接: [https://arxiv.org/pdf/2512.05830v1](https://arxiv.org/pdf/2512.05830v1)**

> **作者:** Muhammet Cagri Yeke; Samil Sirin; Kivilcim Yuksel; Abdurrahman Gumus
>
> **备注:** 22 pages, 11 figures, 5 tables
>
> **摘要:** This study focuses on event detection in optical fibers, specifically classifying six events using the Phase-OTDR system. A novel approach is introduced to enhance Phase-OTDR data analysis by transforming 1D data into grayscale images through techniques such as Gramian Angular Difference Field, Gramian Angular Summation Field, and Recurrence Plot. These grayscale images are combined into a multi-channel RGB representation, enabling more robust and adaptable analysis using transfer learning models. The proposed methodology achieves high classification accuracies of 98.84% and 98.24% with the EfficientNetB0 and DenseNet121 models, respectively. A 5-fold cross-validation process confirms the reliability of these models, with test accuracy rates of 99.07% and 98.68%. Using a publicly available Phase-OTDR dataset, the study demonstrates an efficient approach to understanding optical fiber events while reducing dataset size and improving analysis efficiency. The results highlight the transformative potential of image-based analysis in interpreting complex fiber optic sensing data, offering significant advancements in the accuracy and reliability of fiber optic monitoring systems. The codes and the corresponding image-based dataset are made publicly available on GitHub to support further research: https://github.com/miralab-ai/Phase-OTDR-event-detection.
>
---
#### [new 078] Experts-Guided Unbalanced Optimal Transport for ISP Learning from Unpaired and/or Paired Data
- **分类: cs.CV**

- **简介: 该论文研究图像信号处理（ISP）学习，旨在解决依赖成对数据的问题。提出基于非平衡最优传输的框架，支持成对与非成对训练，并引入专家委员会判别器提升色彩、结构与频域质量，实现优于或媲美有监督方法的效果。**

- **链接: [https://arxiv.org/pdf/2512.05635v1](https://arxiv.org/pdf/2512.05635v1)**

> **作者:** Georgy Perevozchikov; Nancy Mehta; Egor Ershov; Radu Timofte
>
> **摘要:** Learned Image Signal Processing (ISP) pipelines offer powerful end-to-end performance but are critically dependent on large-scale paired raw-to-sRGB datasets. This reliance on costly-to-acquire paired data remains a significant bottleneck. To address this challenge, we introduce a novel, unsupervised training framework based on Optimal Transport capable of training arbitrary ISP architectures in both unpaired and paired modes. We are the first to successfully apply Unbalanced Optimal Transport (UOT) for this complex, cross-domain translation task. Our UOT-based framework provides robustness to outliers in the target sRGB data, allowing it to discount atypical samples that would be prohibitively costly to map. A key component of our framework is a novel ``committee of expert discriminators,'' a hybrid adversarial regularizer. This committee guides the optimal transport mapping by providing specialized, targeted gradients to correct specific ISP failure modes, including color fidelity, structural artifacts, and frequency-domain realism. To demonstrate the superiority of our approach, we retrained existing state-of-the-art ISP architectures using our paired and unpaired setups. Our experiments show that while our framework, when trained in paired mode, exceeds the performance of the original paired methods across all metrics, our unpaired mode concurrently achieves quantitative and qualitative performance that rivals, and in some cases surpasses, the original paired-trained counterparts. The code and pre-trained models are available at: https://github.com/gosha20777/EGUOT-ISP.git.
>
---
#### [new 079] InvarDiff: Cross-Scale Invariance Caching for Accelerated Diffusion Models
- **分类: cs.CV; cs.DC; cs.LG**

- **简介: 该论文针对扩散模型推理慢的问题，提出InvarDiff方法，利用跨时间步和网络层的特征不变性进行缓存加速。通过构建二值缓存矩阵指导推理时的跨步与层间复用，减少冗余计算，在保持生成质量的同时实现2-3倍加速。**

- **链接: [https://arxiv.org/pdf/2512.05134v1](https://arxiv.org/pdf/2512.05134v1)**

> **作者:** Zihao Wu
>
> **备注:** 8 pages main, 8 pages appendix, 16 figures, 5 tables. Code: https://github.com/zihaowu25/InvarDiff
>
> **摘要:** Diffusion models deliver high-fidelity synthesis but remain slow due to iterative sampling. We empirically observe there exists feature invariance in deterministic sampling, and present InvarDiff, a training-free acceleration method that exploits the relative temporal invariance across timestep-scale and layer-scale. From a few deterministic runs, we compute a per-timestep, per-layer, per-module binary cache plan matrix and use a re-sampling correction to avoid drift when consecutive caches occur. Using quantile-based change metrics, this matrix specifies which module at which step is reused rather than recomputed. The same invariance criterion is applied at the step scale to enable cross-timestep caching, deciding whether an entire step can reuse cached results. During inference, InvarDiff performs step-first and layer-wise caching guided by this matrix. When applied to DiT and FLUX, our approach reduces redundant compute while preserving fidelity. Experiments show that InvarDiff achieves $2$-$3\times$ end-to-end speed-ups with minimal impact on standard quality metrics. Qualitatively, we observe almost no degradation in visual quality compared with full computations.
>
---
#### [new 080] WaterWave: Bridging Underwater Image Enhancement into Video Streams via Wavelet-based Temporal Consistency Field
- **分类: cs.CV**

- **简介: 该论文研究水下视频增强，解决单帧增强导致的时序不一致问题。提出WaterWave方法，通过小波域时序一致性场和光流校正模块，在无配对数据下实现流畅、细节保留的增强视频，提升下游跟踪任务性能。**

- **链接: [https://arxiv.org/pdf/2512.05492v1](https://arxiv.org/pdf/2512.05492v1)**

> **作者:** Qi Zhu; Jingyi Zhang; Naishan Zheng; Wei Yu; Jinghao Zhang; Deyi Ji; Feng Zhao
>
> **摘要:** Underwater video pairs are fairly difficult to obtain due to the complex underwater imaging. In this case, most existing video underwater enhancement methods are performed by directly applying the single-image enhancement model frame by frame, but a natural issue is lacking temporal consistency. To relieve the problem, we rethink the temporal manifold inherent in natural videos and observe a temporal consistency prior in dynamic scenes from the local temporal frequency perspective. Building upon the specific prior and no paired-data condition, we propose an implicit representation manner for enhanced video signals, which is conducted in the wavelet-based temporal consistency field, WaterWave. Specifically, under the constraints of the prior, we progressively filter and attenuate the inconsistent components while preserving motion details and scenes, achieving a natural-flowing video. Furthermore, to represent temporal frequency bands more accurately, an underwater flow correction module is designed to rectify estimated flows considering the transmission in underwater scenes. Extensive experiments demonstrate that WaterWave significantly enhances the quality of videos generated using single-image underwater enhancements. Additionally, our method demonstrates high potential in downstream underwater tracking tasks, such as UOSTrack and MAT, outperforming the original video by a large margin, i.e., 19.7% and 9.7% on precise respectively.
>
---
#### [new 081] Zoom in, Click out: Unlocking and Evaluating the Potential of Zooming for GUI Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究GUI界面元素定位任务，提出无需训练的ZoomClick方法，利用缩放操作的先验特性实现精准元素定位，并构建GUIZoom-Bench基准测试，提升模型在不同场景下的适应性与定位精度。**

- **链接: [https://arxiv.org/pdf/2512.05941v1](https://arxiv.org/pdf/2512.05941v1)**

> **作者:** Zhiyuan Jiang; Shenghao Xie; Wenyi Li; Wenqiang Zu; Peihang Li; Jiahao Qiu; Siqi Pei; Lei Ma; Tiejun Huang; Mengdi Wang; Shilong Liu
>
> **备注:** Code is available at https://github.com/Princeton-AI2-Lab/ZoomClick
>
> **摘要:** Grounding is a fundamental capability for building graphical user interface (GUI) agents. Although existing approaches rely on large-scale bounding box supervision, they still face various challenges, such as cross-platform generalization, complex layout analysis, and fine-grained element localization. In this paper, we investigate zoom as a strong yet underexplored prior for GUI grounding, and propose a training-free method, ZoomClick. By characterizing four key properties of zoom (i.e., pre-zoom, depth, shrink size, minimal crop size), we unlock its full capabilities for dynamic spatial focusing and adaptive context switching. Experiments demonstrate that our method significantly boosts the performance of both general vision-language and specialized GUI grounding models, achieving state-of-the-art results on several mainstream benchmarks; for example, UI-Venus-72B attains a 73.1% success rate on ScreenSpot-Pro. Furthermore, we present GUIZoom-Bench, a benchmark for evaluating model adaptability to zoom, aiming to inspire future research on improving zoom for further training and test-time scaling in GUI grounding tasks.
>
---
#### [new 082] Rethinking Infrared Small Target Detection: A Foundation-Driven Efficient Paradigm
- **分类: cs.CV**

- **简介: 该论文研究红外小目标检测任务，旨在探索视觉基础模型在该任务中的潜力。提出基础驱动高效范式FDEP，通过语义对齐融合模块和隐式自蒸馏策略，提升检测精度且无推理开销，并构建统一评估指标HSE，实现多模型公平比较。**

- **链接: [https://arxiv.org/pdf/2512.05511v1](https://arxiv.org/pdf/2512.05511v1)**

> **作者:** Chuang Yu; Jinmiao Zhao; Yunpeng Liu; Yaokun Li; Xiujun Shu; Yuanhao Feng; Bo Wang; Yimian Dai; Xiangyu Yue
>
> **摘要:** While large-scale visual foundation models (VFMs) exhibit strong generalization across diverse visual domains, their potential for single-frame infrared small target (SIRST) detection remains largely unexplored. To fill this gap, we systematically introduce the frozen representations from VFMs into the SIRST task for the first time and propose a Foundation-Driven Efficient Paradigm (FDEP), which can seamlessly adapt to existing encoder-decoder-based methods and significantly improve accuracy without additional inference overhead. Specifically, a Semantic Alignment Modulation Fusion (SAMF) module is designed to achieve dynamic alignment and deep fusion of the global semantic priors from VFMs with task-specific features. Meanwhile, to avoid the inference time burden introduced by VFMs, we propose a Collaborative Optimization-based Implicit Self-Distillation (CO-ISD) strategy, which enables implicit semantic transfer between the main and lightweight branches through parameter sharing and synchronized backpropagation. In addition, to unify the fragmented evaluation system, we construct a Holistic SIRST Evaluation (HSE) metric that performs multi-threshold integral evaluation at both pixel-level confidence and target-level robustness, providing a stable and comprehensive basis for fair model comparison. Extensive experiments demonstrate that the SIRST detection networks equipped with our FDEP framework achieve state-of-the-art (SOTA) performance on multiple public datasets. Our code is available at https://github.com/YuChuang1205/FDEP-Framework
>
---
#### [new 083] EmoStyle: Emotion-Driven Image Stylization
- **分类: cs.CV**

- **简介: 该论文提出情感驱动的图像风格化任务（AIS），旨在通过艺术风格激发特定情绪。为解决数据缺失和情绪-风格映射问题，构建了EmoStyleSet数据集，设计了情绪内容推理器与风格量化模块。实验表明方法能有效提升情感表达并保持内容一致。**

- **链接: [https://arxiv.org/pdf/2512.05478v1](https://arxiv.org/pdf/2512.05478v1)**

> **作者:** Jingyuan Yang; Zihuan Bai; Hui Huang
>
> **摘要:** Art has long been a profound medium for expressing emotions. While existing image stylization methods effectively transform visual appearance, they often overlook the emotional impact carried by styles. To bridge this gap, we introduce Affective Image Stylization (AIS), a task that applies artistic styles to evoke specific emotions while preserving content. We present EmoStyle, a framework designed to address key challenges in AIS, including the lack of training data and the emotion-style mapping. First, we construct EmoStyleSet, a content-emotion-stylized image triplet dataset derived from ArtEmis to support AIS. We then propose an Emotion-Content Reasoner that adaptively integrates emotional cues with content to learn coherent style queries. Given the discrete nature of artistic styles, we further develop a Style Quantizer that converts continuous style features into emotion-related codebook entries. Extensive qualitative and quantitative evaluations, including user studies, demonstrate that EmoStyle enhances emotional expressiveness while maintaining content consistency. Moreover, the learned emotion-aware style dictionary is adaptable to other generative tasks, highlighting its potential for broader applications. Our work establishes a foundation for emotion-driven image stylization, expanding the creative potential of AI-generated art.
>
---
#### [new 084] Distilling Expert Surgical Knowledge: How to train local surgical VLMs for anatomy explanation in Complete Mesocolic Excision
- **分类: cs.CV**

- **简介: 该论文致力于提升视觉大模型在结肠癌手术场景中的解剖理解能力。为解决现有模型领域知识不足和数据隐私问题，提出一种无需敏感图像的知识蒸馏框架，利用教师大模型生成专家级文本-掩码数据，用于本地小模型的监督微调与偏好优化，实现高效、隐私安全的手术视觉理解模型训练。**

- **链接: [https://arxiv.org/pdf/2512.05740v1](https://arxiv.org/pdf/2512.05740v1)**

> **作者:** Lennart Maack; Julia-Kristin Graß; Lisa-Marie Toscha; Nathaniel Melling; Alexander Schlaefer
>
> **摘要:** Recently, Vision Large Language Models (VLMs) have demonstrated high potential in computer-aided diagnosis and decision-support. However, current VLMs show deficits in domain specific surgical scene understanding, such as identifying and explaining anatomical landmarks during Complete Mesocolic Excision. Additionally, there is a need for locally deployable models to avoid patient data leakage to large VLMs, hosted outside the clinic. We propose a privacy-preserving framework to distill knowledge from large, general-purpose LLMs into an efficient, local VLM. We generate an expert-supervised dataset by prompting a teacher LLM without sensitive images, using only textual context and binary segmentation masks for spatial information. This dataset is used for Supervised Fine-Tuning (SFT) and subsequent Direct Preference Optimization (DPO) of the locally deployable VLM. Our evaluation confirms that finetuning VLMs with our generated datasets increases surgical domain knowledge compared to its base VLM by a large margin. Overall, this work validates a data-efficient and privacy-conforming way to train a surgical domain optimized, locally deployable VLM for surgical scene understanding.
>
---
#### [new 085] ProPhy: Progressive Physical Alignment for Dynamic World Simulation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决现有模型在复杂物理动态下生成结果物理不一致的问题。提出ProPhy框架，通过两阶段MoPE机制和物理对齐策略，实现细粒度的物理感知生成，提升动态真实性和物理合理性。**

- **链接: [https://arxiv.org/pdf/2512.05564v1](https://arxiv.org/pdf/2512.05564v1)**

> **作者:** Zijun Wang; Panwen Hu; Jing Wang; Terry Jingchen Zhang; Yuhao Cheng; Long Chen; Yiqiang Yan; Zutao Jiang; Hanhui Li; Xiaodan Liang
>
> **摘要:** Recent advances in video generation have shown remarkable potential for constructing world simulators. However, current models still struggle to produce physically consistent results, particularly when handling large-scale or complex dynamics. This limitation arises primarily because existing approaches respond isotropically to physical prompts and neglect the fine-grained alignment between generated content and localized physical cues. To address these challenges, we propose ProPhy, a Progressive Physical Alignment Framework that enables explicit physics-aware conditioning and anisotropic generation. ProPhy employs a two-stage Mixture-of-Physics-Experts (MoPE) mechanism for discriminative physical prior extraction, where Semantic Experts infer semantic-level physical principles from textual descriptions, and Refinement Experts capture token-level physical dynamics. This mechanism allows the model to learn fine-grained, physics-aware video representations that better reflect underlying physical laws. Furthermore, we introduce a physical alignment strategy that transfers the physical reasoning capabilities of vision-language models (VLMs) into the Refinement Experts, facilitating a more accurate representation of dynamic physical phenomena. Extensive experiments on physics-aware video generation benchmarks demonstrate that ProPhy produces more realistic, dynamic, and physically coherent results than existing state-of-the-art methods.
>
---
#### [new 086] SyncVoice: Towards Video Dubbing with Vision-Augmented Pretrained TTS Model
- **分类: eess.AS; cs.AI; cs.CL; cs.CV; cs.MM; cs.SD**

- **简介: 该论文研究视频配音任务，旨在解决现有方法在语音自然度、音画同步及多语言支持上的不足。作者提出SyncVoice框架，基于预训练TTS模型引入视觉信息并设计双说话人编码器，提升跨语言合成效果与音画一致性。**

- **链接: [https://arxiv.org/pdf/2512.05126v1](https://arxiv.org/pdf/2512.05126v1)**

> **作者:** Kaidi Wang; Yi He; Wenhao Guan; Weijie Wu; Hongwu Ding; Xiong Zhang; Di Wu; Meng Meng; Jian Luan; Lin Li; Qingyang Hong
>
> **摘要:** Video dubbing aims to generate high-fidelity speech that is precisely temporally aligned with the visual content. Existing methods still suffer from limitations in speech naturalness and audio-visual synchronization, and are limited to monolingual settings. To address these challenges, we propose SyncVoice, a vision-augmented video dubbing framework built upon a pretrained text-to-speech (TTS) model. By fine-tuning the TTS model on audio-visual data, we achieve strong audiovisual consistency. We propose a Dual Speaker Encoder to effectively mitigate inter-language interference in cross-lingual speech synthesis and explore the application of video dubbing in video translation scenarios. Experimental results show that SyncVoice achieves high-fidelity speech generation with strong synchronization performance, demonstrating its potential in video dubbing tasks.
>
---
#### [new 087] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文聚焦多语言多模态检索增强生成（RAG）任务，旨在解决现有VLMs因静态训练数据导致的知识局限问题。作者构建了大规模基准M4-RAG，涵盖42种语言、56种方言及8万余图文对，评估跨语言文化场景下的RAG性能，发现当前RAG难以有效扩展至大模型，揭示模型规模与检索效果间的不匹配问题。**

- **链接: [https://arxiv.org/pdf/2512.05959v1](https://arxiv.org/pdf/2512.05959v1)**

> **作者:** David Anugraha; Patrick Amadeus Irawan; Anshul Singh; En-Shiun Annie Lee; Genta Indra Winata
>
> **备注:** Preprint
>
> **摘要:** Vision-language models (VLMs) have achieved strong performance in visual question answering (VQA), yet they remain constrained by static training data. Retrieval-Augmented Generation (RAG) mitigates this limitation by enabling access to up-to-date, culturally grounded, and multilingual information; however, multilingual multimodal RAG remains largely underexplored. We introduce M4-RAG, a massive-scale benchmark covering 42 languages and 56 regional dialects and registers, comprising over 80,000 culturally diverse image-question pairs for evaluating retrieval-augmented VQA across languages and modalities. To balance realism with reproducibility, we build a controlled retrieval environment containing millions of carefully curated multilingual documents relevant to the query domains, approximating real-world retrieval conditions while ensuring consistent experimentation. Our systematic evaluation reveals that although RAG consistently benefits smaller VLMs, it fails to scale to larger models and often even degrades their performance, exposing a critical mismatch between model size and current retrieval effectiveness. M4-RAG provides a foundation for advancing next-generation RAG systems capable of reasoning seamlessly across languages, modalities, and cultural contexts.
>
---
#### [new 088] Multimodal Oncology Agent for IDH1 Mutation Prediction in Low-Grade Glioma
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出多模态肿瘤代理（MOA）用于预测低级别胶质瘤的IDH1突变。融合组织学与临床基因数据，结合外部知识源推理，提升预测准确率，F1达0.912，优于基线方法。**

- **链接: [https://arxiv.org/pdf/2512.05824v1](https://arxiv.org/pdf/2512.05824v1)**

> **作者:** Hafsa Akebli; Adam Shephard; Vincenzo Della Mea; Nasir Rajpoot
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Low-grade gliomas frequently present IDH1 mutations that define clinically distinct subgroups with specific prognostic and therapeutic implications. This work introduces a Multimodal Oncology Agent (MOA) integrating a histology tool based on the TITAN foundation model for IDH1 mutation prediction in low-grade glioma, combined with reasoning over structured clinical and genomic inputs through PubMed, Google Search, and OncoKB. MOA reports were quantitatively evaluated on 488 patients from the TCGA-LGG cohort against clinical and histology baselines. MOA without the histology tool outperformed the clinical baseline, achieving an F1-score of 0.826 compared to 0.798. When fused with histology features, MOA reached the highest performance with an F1-score of 0.912, exceeding both the histology baseline at 0.894 and the fused histology-clinical baseline at 0.897. These results demonstrate that the proposed agent captures complementary mutation-relevant information enriched through external biomedical sources, enabling accurate IDH1 mutation prediction.
>
---
#### [new 089] Physically-Based Simulation of Automotive LiDAR
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究自动驾驶激光雷达的物理仿真，旨在准确模拟飞行时间激光雷达的光学特性。通过结合近红外物理渲染与实验室测量，建模光束扩展、回波脉宽、环境光等效应，并针对实际系统进行参数标定与验证。**

- **链接: [https://arxiv.org/pdf/2512.05932v1](https://arxiv.org/pdf/2512.05932v1)**

> **作者:** L. Dudzik; M. Roschani; A. Sielemann; K. Trampert; J. Ziehn; J. Beyerer; C. Neumann
>
> **摘要:** We present an analytic model for simulating automotive time-of-flight (ToF) LiDAR that includes blooming, echo pulse width, and ambient light, along with steps to determine model parameters systematically through optical laboratory measurements. The model uses physically based rendering (PBR) in the near-infrared domain. It assumes single-bounce reflections and retroreflections over rasterized rendered images from shading or ray tracing, including light emitted from the sensor as well as stray light from other, non-correlated sources such as sunlight. Beams from the sensor and sensitivity of the receiving diodes are modeled with flexible beam steering patterns and with non-vanishing diameter. Different (all non-real time) computational approaches can be chosen based on system properties, computing capabilities, and desired output properties. Model parameters include system-specific properties, namely the physical spread of the LiDAR beam, combined with the sensitivity of the receiving diode; the intensity of the emitted light; the conversion between the intensity of reflected light and the echo pulse width; and scenario parameters such as environment lighting, positioning, and surface properties of the target(s) in the relevant infrared domain. System-specific properties of the model are determined from laboratory measurements of the photometric luminance on different target surfaces aligned with a goniometer at 0.01° resolution, which marks the best available resolution for measuring the beam pattern. The approach is calibrated for and tested on two automotive LiDAR systems, the Valeo Scala Gen. 2 and the Blickfeld Cube 1. Both systems differ notably in their properties and available interfaces, but the relevant model parameters could be extracted successfully.
>
---
#### [new 090] ARCAS: An Augmented Reality Collision Avoidance System with SLAM-Based Tracking for Enhancing VRU Safety
- **分类: eess.SY; cs.AR; cs.CV; cs.ET; cs.RO; eess.IV**

- **简介: 该论文提出ARCAS，一种基于SLAM和LiDAR的增强现实碰撞预警系统，旨在提升弱势道路使用者（VRU）安全性。通过AR头显实时叠加3D警示信息，解决混合交通中VRU碰撞风险问题，实验证明显著提升了反应时间与安全裕度。**

- **链接: [https://arxiv.org/pdf/2512.05299v1](https://arxiv.org/pdf/2512.05299v1)**

> **作者:** Ahmad Yehia; Jiseop Byeon; Tianyi Wang; Huihai Wang; Yiming Xu; Junfeng Jiao; Christian Claudel
>
> **备注:** 8 pages, 3 figures, 1 table
>
> **摘要:** Vulnerable road users (VRUs) face high collision risks in mixed traffic, yet most existing safety systems prioritize driver or vehicle assistance over direct VRU support. This paper presents ARCAS, a real-time augmented reality collision avoidance system that provides personalized spatial alerts to VRUs via wearable AR headsets. By fusing roadside 360-degree 3D LiDAR with SLAM-based headset tracking and an automatic 3D calibration procedure, ARCAS accurately overlays world-locked 3D bounding boxes and directional arrows onto approaching hazards in the user's passthrough view. The system also enables multi-headset coordination through shared world anchoring. Evaluated in real-world pedestrian interactions with e-scooters and vehicles (180 trials), ARCAS nearly doubled pedestrians' time-to-collision and increased counterparts' reaction margins by up to 4x compared to unaided-eye conditions. Results validate the feasibility and effectiveness of LiDAR-driven AR guidance and highlight the potential of wearable AR as a promising next-generation safety tool for urban mobility.
>
---
#### [new 091] Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多智能体驾驶仿真任务，旨在解决行为模型的效率与真实性平衡问题。作者提出实例中心的场景表示和对称上下文编码器，结合自适应奖励的逆强化学习方法，提升仿真效率、准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.05812v1](https://arxiv.org/pdf/2512.05812v1)**

> **作者:** Fabian Konstantinidis; Moritz Sackmann; Ulrich Hofmann; Christoph Stiller
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.
>
---
#### [new 092] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉语言模型缺乏物理动态理解的问题，提出SIMPACT框架，通过在测试时引入仿真增强VLM的物理推理能力，实现细粒度机器人操作任务的行动规划，提升在真实世界复杂操作任务中的表现。**

- **链接: [https://arxiv.org/pdf/2512.05955v1](https://arxiv.org/pdf/2512.05955v1)**

> **作者:** Haowen Liu; Shaoxiong Yao; Haonan Chen; Jiawei Gao; Jiayuan Mao; Jia-Bin Huang; Yilun Du
>
> **摘要:** Vision-Language Models (VLMs) exhibit remarkable common-sense and semantic reasoning capabilities. However, they lack a grounded understanding of physical dynamics. This limitation arises from training VLMs on static internet-scale visual-language data that contain no causal interactions or action-conditioned changes. Consequently, it remains challenging to leverage VLMs for fine-grained robotic manipulation tasks that require physical understanding, reasoning, and corresponding action planning. To overcome this, we present SIMPACT, a test-time, SIMulation-enabled ACTion Planning framework that equips VLMs with physical reasoning through simulation-in-the-loop world modeling, without requiring any additional training. From a single RGB-D observation, SIMPACT efficiently constructs physics simulations, enabling the VLM to propose informed actions, observe simulated rollouts, and iteratively refine its reasoning. By integrating language reasoning with physics prediction, our simulation-enabled VLM can understand contact dynamics and action outcomes in a physically grounded way. Our method demonstrates state-of-the-art performance on five challenging, real-world rigid-body and deformable manipulation tasks that require fine-grained physical reasoning, outperforming existing general-purpose robotic manipulation models. Our results demonstrate that embedding physics understanding via efficient simulation into VLM reasoning at test time offers a promising path towards generalizable embodied intelligence. Project webpage can be found at https://simpact-bot.github.io
>
---
#### [new 093] Interleaved Latent Visual Reasoning with Selective Perceptual Modeling
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究多模态大模型中的视觉推理任务，旨在解决现有方法在计算成本、感知精度与动态推理间的权衡问题。作者提出ILVR框架，通过交错文本生成与潜变量视觉表征，结合选择性自监督机制，实现高效且精细的多模态推理。**

- **链接: [https://arxiv.org/pdf/2512.05665v1](https://arxiv.org/pdf/2512.05665v1)**

> **作者:** Shuai Dong; Siyuan Wang; Xingyu Liu; Zhongyu Wei
>
> **备注:** 11 pages, 6 figures. Code available at https://github.com/XD111ds/ILVR
>
> **摘要:** Interleaved reasoning paradigms enhance Multimodal Large Language Models (MLLMs) with visual feedback but are hindered by the prohibitive computational cost of repeatedly re-encoding pixel-dense images. A promising alternative, latent visual reasoning, circumvents this bottleneck yet currently forces a critical trade-off: methods either sacrifice precise perceptual modeling by over-compressing features or fail to model dynamic problems due to static, non-interleaved structures. We introduce Interleaved Latent Visual Reasoning (ILVR), a framework that unifies dynamic state evolution with precise perceptual modeling. ILVR interleaves textual generation with latent visual representations that act as specific, evolving cues for subsequent reasoning. To enable this, we employ a self-supervision strategy where a Momentum Teacher Model selectively distills relevant features from helper images into sparse supervision targets. This adaptive selection mechanism guides the model to autonomously generate context-aware visual signals. Extensive experiments on multimodal reasoning benchmarks demonstrate that ILVR significantly outperforms existing approaches, effectively bridging the gap between fine-grained perception and sequential multimodal reasoning.
>
---
#### [new 094] EXR: An Interactive Immersive EHR Visualization in Extended Reality
- **分类: cs.HC; cs.CV; cs.LG; cs.MM**

- **简介: 该论文提出一种基于扩展现实（XR）的电子健康记录（EHR）可视化平台，旨在解决传统2D界面难以直观展示复杂医疗数据的问题。通过融合FHIR数据、三维医学影像与AI分割结果，构建可交互、沉浸式的共享3D环境，支持临床决策与协作分析。**

- **链接: [https://arxiv.org/pdf/2512.05438v1](https://arxiv.org/pdf/2512.05438v1)**

> **作者:** Benoit Marteau; Shaun Q. Y. Tan; Jieru Li; Andrew Hornback; Yishan Zhong; Shaunna Wang; Christian Lowson; Jason Woloff; Joshua M. Pahys; Steven W. Hwang; Coleman Hilton; May D. Wang
>
> **备注:** 11 pages, 6 figures. Preprint version. This paper has been accepted to IEEE ICIR 2025. This is the author-prepared version and not the final published version. The final version will appear in IEEE Xplo
>
> **摘要:** This paper presents the design and implementation of an Extended Reality (XR) platform for immersive, interactive visualization of Electronic Health Records (EHRs). The system extends beyond conventional 2D interfaces by visualizing both structured and unstructured patient data into a shared 3D environment, enabling intuitive exploration and real-time collaboration. The modular infrastructure integrates FHIR-based EHR data with volumetric medical imaging and AI-generated segmentation, ensuring interoperability with modern healthcare systems. The platform's capabilities are demonstrated using synthetic EHR datasets and computed tomography (CT)-derived spine models processed through an AI-powered segmentation pipeline. This work suggests that such integrated XR solutions could form the foundation for next-generation clinical decision-support tools, where advanced data infrastructures are directly accessible in an interactive and spatially rich environment.
>
---
## 更新

#### [replaced 001] Enabling Validation for Robust Few-Shot Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.04713v2](https://arxiv.org/pdf/2506.04713v2)**

> **作者:** Hanxin Wang; Tian Liu; Shu Kong
>
> **备注:** Project website: https://hannawang09.github.io/projects/vest/
>
> **摘要:** Few-Shot Recognition (FSR) tackles classification tasks by training with minimal task-specific labeled data. Prevailing methods adapt or finetune a pretrained Vision-Language Model (VLM) and augment the scarce training data by retrieving task-relevant but noisy samples from open data sources. The finetuned VLM generalizes decently well to the task-specific in-distribution (ID) test data but struggles with out-of-distribution (OOD) test data. This motivates our study of robust FSR with VLM finetuning. The core challenge of FSR is data scarcity, extending beyond limited training data to a complete lack of validation data. We identify a key paradox as a potential solution: repurposing the retrieved open data for validation. As such retrieved data are inherently OOD compared with the task-specific ID training data, finetuned VLMs yield degraded performance on the retrieved data. This causes the validation logic to favor the pretrained model without any finetuning, hindering improvements w.r.t generalization. To resolve this dilemma, we introduce a novel validation strategy that harmonizes performance gain and degradation on the few-shot ID data and the retrieved data, respectively. Our validation enables parameter selection for partial finetuning and checkpoint selection, mitigating overfitting and improving test-data generalization. We unify this strategy with robust learning into a cohesive framework: Validation-Enabled Stage-wise Tuning (VEST). Extensive experiments on the established ImageNet OOD benchmarks show that VEST significantly outperforms existing VLM adaptation methods, achieving state-of-the-art FSR performance on both ID and OOD data.
>
---
#### [replaced 002] Multi-Modal Data-Efficient 3D Scene Understanding for Autonomous Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究自动驾驶中的多模态半监督3D场景理解，旨在减少对标注LiDAR数据的依赖。提出LaserMix++框架，融合激光雷达与相机数据，通过跨模态交互、特征蒸馏和语言引导生成辅助监督，提升数据利用效率，在少标签下显著优于全监督方法。**

- **链接: [https://arxiv.org/pdf/2405.05258v3](https://arxiv.org/pdf/2405.05258v3)**

> **作者:** Lingdong Kong; Xiang Xu; Jiawei Ren; Wenwei Zhang; Liang Pan; Kai Chen; Wei Tsang Ooi; Ziwei Liu
>
> **备注:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** Efficient data utilization is crucial for advancing 3D scene understanding in autonomous driving, where reliance on heavily human-annotated LiDAR point clouds challenges fully supervised methods. Addressing this, our study extends into semi-supervised learning for LiDAR semantic segmentation, leveraging the intrinsic spatial priors of driving scenes and multi-sensor complements to augment the efficacy of unlabeled datasets. We introduce LaserMix++, an evolved framework that integrates laser beam manipulations from disparate LiDAR scans and incorporates LiDAR-camera correspondences to further assist data-efficient learning. Our framework is tailored to enhance 3D scene consistency regularization by incorporating multi-modality, including 1) multi-modal LaserMix operation for fine-grained cross-sensor interactions; 2) camera-to-LiDAR feature distillation that enhances LiDAR feature learning; and 3) language-driven knowledge guidance generating auxiliary supervisions using open-vocabulary models. The versatility of LaserMix++ enables applications across LiDAR representations, establishing it as a universally applicable solution. Our framework is rigorously validated through theoretical analysis and extensive experiments on popular driving perception datasets. Results demonstrate that LaserMix++ markedly outperforms fully supervised alternatives, achieving comparable accuracy with five times fewer annotations and significantly improving the supervised-only baselines. This substantial advancement underscores the potential of semi-supervised approaches in reducing the reliance on extensive labeled data in LiDAR-based 3D scene understanding systems.
>
---
#### [replaced 003] CookAnything: A Framework for Flexible and Consistent Multi-Step Recipe Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.03540v2](https://arxiv.org/pdf/2512.03540v2)**

> **作者:** Ruoxuan Zhang; Bin Wen; Hongxia Xie; Yi Yao; Songhan Zuo; Jian-Yu Jiang-Lin; Hong-Han Shuai; Wen-Huang Cheng
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** Cooking is a sequential and visually grounded activity, where each step such as chopping, mixing, or frying carries both procedural logic and visual semantics. While recent diffusion models have shown strong capabilities in text-to-image generation, they struggle to handle structured multi-step scenarios like recipe illustration. Additionally, current recipe illustration methods are unable to adjust to the natural variability in recipe length, generating a fixed number of images regardless of the actual instructions structure. To address these limitations, we present CookAnything, a flexible and consistent diffusion-based framework that generates coherent, semantically distinct image sequences from textual cooking instructions of arbitrary length. The framework introduces three key components: (1) Step-wise Regional Control (SRC), which aligns textual steps with corresponding image regions within a single denoising process; (2) Flexible RoPE, a step-aware positional encoding mechanism that enhances both temporal coherence and spatial diversity; and (3) Cross-Step Consistency Control (CSCC), which maintains fine-grained ingredient consistency across steps. Experimental results on recipe illustration benchmarks show that CookAnything performs better than existing methods in training-based and training-free settings. The proposed framework supports scalable, high-quality visual synthesis of complex multi-step instructions and holds significant potential for broad applications in instructional media, and procedural content creation.
>
---
#### [replaced 004] Structure is Supervision: Multiview Masked Autoencoders for Radiology
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.22294v3](https://arxiv.org/pdf/2511.22294v3)**

> **作者:** Sonia Laguna; Andrea Agostini; Alain Ryser; Samuel Ruiperez-Campillo; Irene Cannistraci; Moritz Vandenhirtz; Stephan Mandt; Nicolas Deperrois; Farhad Nooralahzadeh; Michael Krauthammer; Thomas M. Sutter; Julia E. Vogt
>
> **摘要:** Building robust medical machine learning systems requires pretraining strategies that exploit the intrinsic structure present in clinical data. We introduce Multiview Masked Autoencoder (MVMAE), a self-supervised framework that leverages the natural multi-view organization of radiology studies to learn view-invariant and disease-relevant representations. MVMAE combines masked image reconstruction with cross-view alignment, transforming clinical redundancy across projections into a powerful self-supervisory signal. We further extend this approach with MVMAE-V2T, which incorporates radiology reports as an auxiliary text-based learning signal to enhance semantic grounding while preserving fully vision-based inference. Evaluated on a downstream disease classification task on three large-scale public datasets, MIMIC-CXR, CheXpert, and PadChest, MVMAE consistently outperforms supervised and vision-language baselines. Furthermore, MVMAE-V2T provides additional gains, particularly in low-label regimes where structured textual supervision is most beneficial. Together, these results establish the importance of structural and textual supervision as complementary paths toward scalable, clinically grounded medical foundation models.
>
---
#### [replaced 005] ReSem3D: Refinable 3D Spatial Constraints via Fine-Grained Semantic Grounding for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.HC; cs.LG**

- **简介: 该论文研究机器人操作任务，旨在解决语义驱动3D空间约束中粒度粗、缺乏闭环规划和鲁棒性差的问题。提出ReSem3D框架，结合MLLM与VFM实现细粒度语义接地，分阶段构建层次化3D约束，并实时优化控制，提升泛化性与适应性。**

- **链接: [https://arxiv.org/pdf/2507.18262v3](https://arxiv.org/pdf/2507.18262v3)**

> **作者:** Chenyu Su; Weiwei Shang; Chen Qian; Fei Zhang; Shuang Cong
>
> **备注:** 12 pages,9 figures
>
> **摘要:** Semantics-driven 3D spatial constraints align highlevel semantic representations with low-level action spaces, facilitating the unification of task understanding and execution in robotic manipulation. The synergistic reasoning of Multimodal Large Language Models (MLLMs) and Vision Foundation Models (VFMs) enables cross-modal 3D spatial constraint construction. Nevertheless, existing methods have three key limitations: (1) coarse semantic granularity in constraint modeling, (2) lack of real-time closed-loop planning, (3) compromised robustness in semantically diverse environments. To address these challenges, we propose ReSem3D, a unified manipulation framework for semantically diverse environments, leveraging the synergy between VFMs and MLLMs to achieve fine-grained visual grounding and dynamically constructs hierarchical 3D spatial constraints for real-time manipulation. Specifically, the framework is driven by hierarchical recursive reasoning in MLLMs, which interact with VFMs to automatically construct 3D spatial constraints from natural language instructions and RGB-D observations in two stages: part-level extraction and region-level refinement. Subsequently, these constraints are encoded as real-time optimization objectives in joint space, enabling reactive behavior to dynamic disturbances. Extensive simulation and real-world experiments are conducted in semantically rich household and sparse chemical lab environments. The results demonstrate that ReSem3D performs diverse manipulation tasks under zero-shot conditions, exhibiting strong adaptability and generalization. Code and videos are available at https://github.com/scy-v/ReSem3D and https://resem3d.github.io.
>
---
#### [replaced 006] Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01030v2](https://arxiv.org/pdf/2512.01030v2)**

> **作者:** Jing He; Haodong Li; Mingzhi Sheng; Ying-Cong Chen
>
> **备注:** Work done at the Hong Kong University of Science and Technology (Guangzhou). Project page: https://lotus-2.github.io/
>
> **摘要:** Recovering pixel-wise geometric properties from a single image is fundamentally ill-posed due to appearance ambiguity and non-injective mappings between 2D observations and 3D structures. While discriminative regression models achieve strong performance through large-scale supervision, their success is bounded by the scale, quality and diversity of available data and limited physical reasoning. Recent diffusion models exhibit powerful world priors that encode geometry and semantics learned from massive image-text data, yet directly reusing their stochastic generative formulation is suboptimal for deterministic geometric inference: the former is optimized for diverse and high-fidelity image generation, whereas the latter requires stable and accurate predictions. In this work, we propose Lotus-2, a two-stage deterministic framework for stable, accurate and fine-grained geometric dense prediction, aiming to provide an optimal adaption protocol to fully exploit the pre-trained generative priors. Specifically, in the first stage, the core predictor employs a single-step deterministic formulation with a clean-data objective and a lightweight local continuity module (LCM) to generate globally coherent structures without grid artifacts. In the second stage, the detail sharpener performs a constrained multi-step rectified-flow refinement within the manifold defined by the core predictor, enhancing fine-grained geometry through noise-free deterministic flow matching. Using only 59K training samples, less than 1% of existing large-scale datasets, Lotus-2 establishes new state-of-the-art results in monocular depth estimation and highly competitive surface normal prediction. These results demonstrate that diffusion models can serve as deterministic world priors, enabling high-quality geometric reasoning beyond traditional discriminative and generative paradigms.
>
---
#### [replaced 007] LM-CartSeg: Automated Segmentation of Lateral and Medial Cartilage and Subchondral Bone for Radiomics Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03449v2](https://arxiv.org/pdf/2512.03449v2)**

> **作者:** Tongxu Zhang
>
> **备注:** The manuscript represents only a preliminary and substantially incompleted exploration. The author has decided not to stand by these results, and a thoroughly revised and significantly different version will be developed separately. Therefore this version is withdrawn and should not be cited
>
> **摘要:** Background and Objective: Radiomics of knee MRI requires robust, anatomically meaningful regions of interest (ROIs) that jointly capture cartilage and subchondral bone. Most existing work relies on manual ROIs and rarely reports quality control (QC). We present LM-CartSeg, a fully automatic pipeline for cartilage/bone segmentation, geometric lateral/medial (L/M) compartmentalisation and radiomics analysis. Methods: Two 3D nnU-Net models were trained on SKM-TEA (138 knees) and OAIZIB-CM (404 knees). At test time, zero-shot predictions were fused and refined by simple geometric rules: connected-component cleaning, construction of 10 mm subchondral bone bands in physical space, and a data-driven tibial L/M split based on PCA and k-means. Segmentation was evaluated on an OAIZIB-CM test set (103 knees) and on SKI-10 (100 knees). QC used volume and thickness signatures. From 10 ROIs we extracted 4 650 non-shape radiomic features to study inter-compartment similarity, dependence on ROI size, and OA vs. non-OA classification on OAIZIB-CM Results: Post-processing improved macro ASSD on OAIZIB-CM from 2.63 to 0.36 mm and HD95 from 25.2 to 3.35 mm, with DSC 0.91; zero-shot DSC on SKI-10 was 0.80. The geometric L/M rule produced stable compartments across datasets, whereas a direct L/M nnU-Net showed domain-dependent side swaps. Only 6 to 12 percent of features per ROI were strongly correlated with volume or thickness. Radiomics-based models models restricted to size-linked features. Conclusions: LM-CartSeg yields automatic, QCd ROIs and radiomic features that carry discriminative information beyond simple morphometry, providing a practical foundation for multi-centre knee OA radiomics studies.
>
---
#### [replaced 008] Neural Eulerian Scene Flow Fields
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.02031v3](https://arxiv.org/pdf/2410.02031v3)**

> **作者:** Kyle Vedder; Neehar Peri; Ishan Khatri; Siyi Li; Eric Eaton; Mehmet Kocamaz; Yue Wang; Zhiding Yu; Deva Ramanan; Joachim Pehserl
>
> **备注:** Accepted to ICLR 2025. Winner of CVPR 2024 WoD Argoverse Scene Flow Challenge, Unsupervised Track. Project page at https://vedder.io/eulerflow
>
> **摘要:** We reframe scene flow as the task of estimating a continuous space-time ODE that describes motion for an entire observation sequence, represented with a neural prior. Our method, EulerFlow, optimizes this neural prior estimate against several multi-observation reconstruction objectives, enabling high quality scene flow estimation via pure self-supervision on real-world data. EulerFlow works out-of-the-box without tuning across multiple domains, including large-scale autonomous driving scenes and dynamic tabletop settings. Remarkably, EulerFlow produces high quality flow estimates on small, fast moving objects like birds and tennis balls, and exhibits emergent 3D point tracking behavior by solving its estimated ODE over long-time horizons. On the Argoverse 2 2024 Scene Flow Challenge, EulerFlow outperforms all prior art, surpassing the next-best unsupervised method by more than 2.5x, and even exceeding the next-best supervised method by over 10%.
>
---
#### [replaced 009] Edge-Only Universal Adversarial Attacks in Distributed Learning
- **分类: cs.CR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.10500v2](https://arxiv.org/pdf/2411.10500v2)**

> **作者:** Giulio Rossolini; Tommaso Baldi; Alessandro Biondi; Giorgio Buttazzo
>
> **摘要:** Distributed learning frameworks, which partition neural network models across multiple computing nodes, enhance efficiency in collaborative edge-cloud systems, but may also introduce new vulnerabilities to evasion attacks, often in the form of adversarial perturbations. In this work, we present a new threat model that explores the feasibility of generating universal adversarial perturbations (UAPs) when the attacker has access only to the edge portion of the model, consisting of its initial network layers. Unlike traditional attacks that require full model knowledge, our approach shows that adversaries can induce effective mispredictions in the unknown cloud component by manipulating key feature representations at the edge. Following the proposed threat model, we introduce both edge-only untargeted and targeted formulations of UAPs designed to control intermediate features before the split point. Our results on ImageNet demonstrate strong attack transferability to the unknown cloud part, and we compare the proposed method with classical white-box and black-box techniques, highlighting its effectiveness. Additionally, we analyze the capability of an attacker to achieve targeted adversarial effects with edge-only knowledge, revealing intriguing behaviors across multiple networks. By introducing the first adversarial attacks with edge-only knowledge in split inference, this work underscores the importance of addressing partial model access in adversarial robustness, encouraging further research in this area.
>
---
#### [replaced 010] IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.RO**

- **简介: 该论文聚焦VLM驱动具身智能体在家庭任务中的交互安全评估。针对现有静态评测无法捕捉动态风险的问题，提出IS-Bench，首个支持多模态、过程导向的交互安全评测基准，包含161个场景与388种安全风险，揭示当前模型缺乏交互安全意识，并推动更安全AI系统发展。**

- **链接: [https://arxiv.org/pdf/2506.16402v3](https://arxiv.org/pdf/2506.16402v3)**

> **作者:** Xiaoya Lu; Zeren Chen; Xuhao Hu; Yijin Zhou; Weichen Zhang; Dongrui Liu; Lu Sheng; Jing Shao
>
> **摘要:** Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. Code and data are released under https://github.com/AI45Lab/IS-Bench.
>
---
#### [replaced 011] ChartQA-X: Generating Explanations for Visual Chart Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.13275v4](https://arxiv.org/pdf/2504.13275v4)**

> **作者:** Shamanthak Hegde; Pooyan Fazli; Hasti Seifi
>
> **备注:** WACV 2026. Project Page: https://teal-lab.github.io/chartqa-x
>
> **摘要:** The ability to explain complex information from chart images is vital for effective data-driven decision-making. In this work, we address the challenge of generating detailed explanations alongside answering questions about charts. We present ChartQA-X, a comprehensive dataset comprising 30,799 chart samples across four chart types, each paired with contextually relevant questions, answers, and explanations. Explanations are generated and selected based on metrics such as faithfulness, informativeness, coherence, and perplexity. Our human evaluation with 245 participants shows that model-generated explanations in ChartQA-X surpass human-written explanations in accuracy and logic and are comparable in terms of clarity and overall quality. Moreover, models fine-tuned on ChartQA-X show substantial improvements across various metrics, including absolute gains of up to 24.57 points in explanation quality, 18.96 percentage points in question-answering accuracy, and 14.75 percentage points on unseen benchmarks for the same task. By integrating explanatory narratives with answers, our approach enables agents to convey complex visual information more effectively, improving comprehension and greater trust in the generated responses.
>
---
#### [replaced 012] REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13026v2](https://arxiv.org/pdf/2511.13026v2)**

> **作者:** Jiaze Li; Hao Yin; Wenhui Tan; Jingyang Chen; Boshen Xu; Yuxun Qu; Yijing Chen; Jianzhong Ju; Zhenbo Luo; Jian Luan
>
> **摘要:** Self-reflection mechanisms that rely on purely text-based rethinking processes perform well in most multimodal tasks. However, when directly applied to long-form video understanding scenarios, they exhibit clear limitations. The fundamental reasons for this lie in two points: (1)long-form video understanding involves richer and more dynamic visual input, meaning rethinking only the text information is insufficient and necessitates a further rethinking process specifically targeting visual information; (2) purely text-based reflection mechanisms lack cross-modal interaction capabilities, preventing them from fully integrating visual information during reflection. Motivated by these insights, we propose REVISOR (REflective VIsual Segment Oriented Reasoning), a novel framework for tool-augmented multimodal reflection. REVISOR enables MLLMs to collaboratively construct introspective reflection processes across textual and visual modalities, significantly enhancing their reasoning capability for long-form video understanding. To ensure that REVISOR can learn to accurately review video segments highly relevant to the question during reinforcement learning, we designed the Dual Attribution Decoupled Reward (DADR) mechanism. Integrated into the GRPO training strategy, this mechanism enforces causal alignment between the model's reasoning and the selected video evidence. Notably, the REVISOR framework significantly enhances long-form video understanding capability of MLLMs without requiring supplementary supervised fine-tuning or external models, achieving impressive results on four benchmarks including VideoMME, LongVideoBench, MLVU, and LVBench.
>
---
#### [replaced 013] LymphAtlas- A Unified Multimodal Lymphoma Imaging Repository Delivering AI-Enhanced Diagnostic Insight
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.20454v2](https://arxiv.org/pdf/2504.20454v2)**

> **作者:** Jiajun Ding; Beiyao Zhu; Xiaosheng Liu; Lishen Zhang; Zhao Liu
>
> **备注:** 12 pages,3 figures
>
> **摘要:** This study integrates PET metabolic information with CT anatomical structures to establish a 3D multimodal segmentation dataset for lymphoma based on whole-body FDG PET/CT examinations, which bridges the gap of the lack of standardised multimodal segmentation datasets in the field of haematological malignancies. We retrospectively collected 483 examination datasets acquired between March 2011 and May 2024, involving 220 patients (106 non-Hodgkin lymphoma, 42 Hodgkin lymphoma); all data underwent ethical review and were rigorously de-identified. Complete 3D structural information was preserved during data acquisition, preprocessing and annotation, and a high-quality dataset was constructed based on the nnUNet format. By systematic technical validation and evaluation of the preprocessing process, annotation quality and automatic segmentation algorithm, the deep learning model trained based on this dataset is verified to achieve accurate segmentation of lymphoma lesions in PET/CT images with high accuracy, good robustness and reproducibility, which proves the applicability and stability of this dataset in accurate segmentation and quantitative analysis. The deep fusion of PET/CT images achieved with this dataset not only significantly improves the accurate portrayal of the morphology, location and metabolic features of tumour lesions, but also provides solid data support for early diagnosis, clinical staging and personalized treatment, and promotes the development of automated image segmentation and precision medicine based on deep learning. The dataset and related resources are available at https://github.com/SuperD0122/LymphAtlas-.
>
---
#### [replaced 014] PLANesT-3D: A new annotated dataset for segmentation of 3D plant point clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.21150v2](https://arxiv.org/pdf/2407.21150v2)**

> **作者:** Kerem Mertoğlu; Yusuf Şalk; Server Karahan Sarıkaya; Kaya Turgut; Yasemin Evrenesoğlu; Hakan Çevikalp; Ömer Nezih Gerek; Helin Dutağacı; David Rousseau
>
> **摘要:** Creation of new annotated public datasets is crucial in helping advances in 3D computer vision and machine learning meet their full potential for automatic interpretation of 3D plant models. Despite the proliferation of deep neural network architectures for segmentation and phenotyping of 3D plant models in the last decade, the amount of data, and diversity in terms of species and data acquisition modalities are far from sufficient for evaluation of such tools for their generalization ability. To contribute to closing this gap, we introduce PLANesT-3D; a new annotated dataset of 3D color point clouds of plants. PLANesT-3D is composed of 34 point cloud models representing 34 real plants from three different plant species: \textit{Capsicum annuum}, \textit{Rosa kordana}, and \textit{Ribes rubrum}. Both semantic labels in terms of "leaf" and "stem", and organ instance labels were manually annotated for the full point clouds. PLANesT-3D introduces diversity to existing datasets by adding point clouds of two new species and providing 3D data acquired with the low-cost SfM/MVS technique as opposed to laser scanning or expensive setups. Point clouds reconstructed with SfM/MVS modality exhibit challenges such as missing data, variable density, and illumination variations. As an additional contribution, SP-LSCnet, a novel semantic segmentation method that is a combination of unsupervised superpoint extraction and a 3D point-based deep learning approach is introduced and evaluated on the new dataset. The advantages of SP-LSCnet over other deep learning methods are its modular structure and increased interpretability. Two existing deep neural network architectures, PointNet++ and RoseSegNet, were also tested on the point clouds of PLANesT-3D for semantic segmentation.
>
---
#### [replaced 015] Multi-Scale Direction-Aware Network for Infrared Small Target Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.02037v3](https://arxiv.org/pdf/2406.02037v3)**

> **作者:** Jinmiao Zhao; Zelin Shi; Chuang Yu; Yunpeng Liu; Xinyi Ying; Yimian Dai
>
> **摘要:** Infrared small target detection faces the problem that it is difficult to effectively separate the background and the target. Existing deep learning-based methods focus on edge and shape features, but ignore the richer structural differences and detailed information embedded in high-frequency components from different directions, thereby failing to fully exploit the value of high-frequency directional features in target perception. To address this limitation, we propose a multi-scale direction-aware network (MSDA-Net), which is the first attempt to integrate the high-frequency directional features of infrared small targets as domain prior knowledge into neural networks. Specifically, to fully mine the high-frequency directional features, on the one hand, a high-frequency direction injection (HFDI) module without trainable parameters is constructed to inject the high-frequency directional information of the original image into the network. On the other hand, a multi-scale direction-aware (MSDA) module is constructed, which promotes the full extraction of local relations at different scales and the full perception of key features in different directions. In addition, considering the characteristics of infrared small targets, we construct a feature aggregation (FA) structure to address target disappearance in high-level feature maps, and a feature calibration fusion (FCF) module to alleviate feature bias during cross-layer feature fusion. Extensive experimental results show that our MSDA-Net achieves state-of-the-art (SOTA) results on multiple public datasets. The code can be available at https://github.com/YuChuang1205/MSDA-Net
>
---
#### [replaced 016] Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04677v2](https://arxiv.org/pdf/2512.04677v2)**

> **作者:** Yubo Huang; Hailong Guo; Fangtai Wu; Shifeng Zhang; Shijie Huang; Qijun Gan; Lin Liu; Sirui Zhao; Enhong Chen; Jiaming Liu; Steven Hoi
>
> **摘要:** Existing diffusion-based video generation methods are fundamentally constrained by sequential computation and long-horizon inconsistency, limiting their practical adoption in real-time, streaming audio-driven avatar synthesis. We present Live Avatar, an algorithm-system co-designed framework that enables efficient, high-fidelity, and infinite-length avatar generation using a 14-billion-parameter diffusion model. Our approach introduces Timestep-forcing Pipeline Parallelism (TPP), a distributed inference paradigm that pipelines denoising steps across multiple GPUs, effectively breaking the autoregressive bottleneck and ensuring stable, low-latency real-time streaming. To further enhance temporal consistency and mitigate identity drift and color artifacts, we propose the Rolling Sink Frame Mechanism (RSFM), which maintains sequence fidelity by dynamically recalibrating appearance using a cached reference image. Additionally, we leverage Self-Forcing Distribution Matching Distillation to facilitate causal, streamable adaptation of large-scale models without sacrificing visual quality. Live Avatar demonstrates state-of-the-art performance, reaching 20 FPS end-to-end generation on 5 H800 GPUs, and, to the best of our knowledge, is the first to achieve practical, real-time, high-fidelity avatar generation at this scale. Our work establishes a new paradigm for deploying advanced diffusion models in industrial long-form video synthesis applications.
>
---
#### [replaced 017] Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究第一视角下手部运动预测任务，旨在解决现有方法预测目标单一、模态差异大、手头运动耦合等问题。提出Uni-Hand框架，通过多模态融合、双分支扩散模型和目标指示机制，实现2D/3D手部关键点与交互状态的多目标预测，并验证其在下游任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.12878v3](https://arxiv.org/pdf/2511.12878v3)**

> **作者:** Junyi Ma; Wentao Bao; Jingyi Xu; Guanzhong Sun; Yu Zheng; Erhang Zhang; Xieyuanli Chen; Hesheng Wang
>
> **备注:** Extended journal version of MMTwin (IROS'25). Code and data: https://github.com/IRMVLab/UniHand
>
> **摘要:** Forecasting how human hands move in egocentric views is critical for applications like augmented reality and human-robot policy transfer. Recently, several hand trajectory prediction (HTP) methods have been developed to generate future possible hand waypoints, which still suffer from insufficient prediction targets, inherent modality gaps, entangled hand-head motion, and limited validation in downstream tasks. To address these limitations, we present a universal hand motion forecasting framework considering multi-modal input, multi-dimensional and multi-target prediction patterns, and multi-task affordances for downstream applications. We harmonize multiple modalities by vision-language fusion, global context incorporation, and task-aware text embedding injection, to forecast hand waypoints in both 2D and 3D spaces. A novel dual-branch diffusion is proposed to concurrently predict human head and hand movements, capturing their motion synergy in egocentric vision. By introducing target indicators, the prediction model can forecast the specific joint waypoints of the wrist or the fingers, besides the widely studied hand center points. In addition, we enable Uni-Hand to additionally predict hand-object interaction states (contact/separation) to facilitate downstream tasks better. As the first work to incorporate downstream task evaluation in the literature, we build novel benchmarks to assess the real-world applicability of hand motion forecasting algorithms. The experimental results on multiple publicly available datasets and our newly proposed benchmarks demonstrate that Uni-Hand achieves the state-of-the-art performance in multi-dimensional and multi-target hand motion forecasting. Extensive validation in multiple downstream tasks also presents its impressive human-robot policy transfer to enable robotic manipulation, and effective feature enhancement for action anticipation/recognition.
>
---
#### [replaced 018] TeleEgo: Benchmarking Egocentric AI Assistants in the Wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.23981v3](https://arxiv.org/pdf/2510.23981v3)**

> **作者:** Jiaqi Yan; Ruilong Ren; Jingren Liu; Shuning Xu; Ling Wang; Yiheng Wang; Xinlin Zhong; Yun Wang; Long Zhang; Xiangyu Chen; Changzhi Sun; Jixiang Luo; Dell Zhang; Hao Sun; Chi Zhang; Xuelong Li
>
> **摘要:** Egocentric AI assistants in real-world settings must process multi-modal inputs (video, audio, text), respond in real time, and retain evolving long-term memory. However, existing benchmarks typically evaluate these abilities in isolation, lack realistic streaming scenarios, or support only short-term tasks. We introduce \textbf{TeleEgo}, a long-duration, streaming, omni-modal benchmark for evaluating egocentric AI assistants in realistic daily contexts. The dataset features over 14 hours per participant of synchronized egocentric video, audio, and text across four domains: work \& study, lifestyle \& routines, social activities, and outings \& culture. All data is aligned on a unified global timeline and includes high-quality visual narrations and speech transcripts, curated through human refinement.TeleEgo defines 12 diagnostic subtasks across three core capabilities: Memory (recalling past events), Understanding (interpreting the current moment), and Cross-Memory Reasoning (linking distant events). It contains 3,291 human-verified QA items spanning multiple question formats (single-choice, binary, multi-choice, and open-ended), evaluated strictly in a streaming setting. We propose Real-Time Accuracy (RTA) to jointly capture correctness and responsiveness under tight decision windows, and Memory Persistence Time (MPT) as a forward-looking metric for long-term retention in continuous streams. In this work, we report RTA results for current models and release TeleEgo, together with an MPT evaluation framework, as a realistic and extensible benchmark for future egocentric assistants with stronger streaming memory, enabling systematic study of both real-time behavior and long-horizon memory.
>
---
#### [replaced 019] ZQBA: Zero Query Black-box Adversarial Attack
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00769v2](https://arxiv.org/pdf/2510.00769v2)**

> **作者:** Joana C. Costa; Tiago Roxo; Hugo Proença; Pedro R. M. Inácio
>
> **备注:** Accepted in ICAART 2026 Conference
>
> **摘要:** Current black-box adversarial attacks either require multiple queries or diffusion models to produce adversarial samples that can impair the target model performance. However, these methods require training a surrogate loss or diffusion models to produce adversarial samples, which limits their applicability in real-world settings. Thus, we propose a Zero Query Black-box Adversarial (ZQBA) attack that exploits the representations of Deep Neural Networks (DNNs) to fool other networks. Instead of requiring thousands of queries to produce deceiving adversarial samples, we use the feature maps obtained from a DNN and add them to clean images to impair the classification of a target model. The results suggest that ZQBA can transfer the adversarial samples to different models and across various datasets, namely CIFAR and Tiny ImageNet. The experiments also show that ZQBA is more effective than state-of-the-art black-box attacks with a single query, while maintaining the imperceptibility of perturbations, evaluated both quantitatively (SSIM) and qualitatively, emphasizing the vulnerabilities of employing DNNs in real-world contexts. All the source code is available at https://github.com/Joana-Cabral/ZQBA.
>
---
#### [replaced 020] GLDiTalker: Speech-Driven 3D Facial Animation with Graph Latent Diffusion Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.01826v5](https://arxiv.org/pdf/2408.01826v5)**

> **作者:** Yihong Lin; Zhaoxin Fan; Xianjia Wu; Lingyu Xiong; Liang Peng; Xiandong Li; Wenxiong Kang; Songju Lei; Huang Xu
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Speech-driven talking head generation is a critical yet challenging task with applications in augmented reality and virtual human modeling. While recent approaches using autoregressive and diffusion-based models have achieved notable progress, they often suffer from modality inconsistencies, particularly misalignment between audio and mesh, leading to reduced motion diversity and lip-sync accuracy. To address this, we propose GLDiTalker, a novel speech-driven 3D facial animation model based on a Graph Latent Diffusion Transformer. GLDiTalker resolves modality misalignment by diffusing signals within a quantized spatiotemporal latent space. It employs a two-stage training pipeline: the Graph-Enhanced Quantized Space Learning Stage ensures lip-sync accuracy, while the Space-Time Powered Latent Diffusion Stage enhances motion diversity. Together, these stages enable GLDiTalker to generate realistic, temporally stable 3D facial animations. Extensive evaluations on standard benchmarks demonstrate that GLDiTalker outperforms existing methods, achieving superior results in both lip-sync accuracy and motion diversity.
>
---
#### [replaced 021] Language Integration in Fine-Tuning Multimodal Large Language Models for Image-Based Regression
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.14997v2](https://arxiv.org/pdf/2507.14997v2)**

> **作者:** Roy H. Jennings; Genady Paikin; Roy Shaul; Evgeny Soloveichik
>
> **备注:** WACV 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) show promise for image-based regression tasks, but current approaches face key limitations. Recent methods fine-tune MLLMs using preset output vocabularies and generic task-level prompts (e.g., "How would you rate this image?"), assuming this mimics human rating behavior. \textbf{Our analysis reveals that these approaches provide no benefit over image-only training}. Models using preset vocabularies and generic prompts perform equivalently to image-only models, failing to leverage semantic understanding from textual input. We propose \textbf{Regression via Transformer-Based Classification} (RvTC), which replaces vocabulary-constrained classification with a flexible bin-based approach. Unlike approaches that address discretization errors through complex distributional modeling, RvTC eliminates manual vocabulary crafting through straightforward bin increase, achieving state-of-the-art performance on four image assessment datasets using only images. \textbf{More importantly, we demonstrate that data-specific prompts dramatically improve performance}. Unlike generic task descriptions, prompts containing semantic information about specific images enable MLLMs to leverage cross-modal understanding. On the AVA dataset, adding challenge titles to prompts substantially improves our already state-of-the-art image-only baseline. We demonstrate through empirical evidence from the AVA and AGIQA-3k datasets that MLLMs benefit from semantic prompt information, surpassing mere statistical biases. We validate RvTC across two different MLLM architectures, demonstrating consistent improvements and method generalizability.
>
---
#### [replaced 022] V-CECE: Visual Counterfactual Explanations via Conceptual Edits
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.16567v2](https://arxiv.org/pdf/2509.16567v2)**

> **作者:** Nikolaos Spanos; Maria Lymperaiou; Giorgos Filandrianos; Konstantinos Thomas; Athanasios Voulodimos; Giorgos Stamou
>
> **备注:** Accepted in NeurIPS 2025
>
> **摘要:** Recent black-box counterfactual generation frameworks fail to take into account the semantic content of the proposed edits, while relying heavily on training to guide the generation process. We propose a novel, plug-and-play black-box counterfactual generation framework, which suggests step-by-step edits based on theoretical guarantees of optimal edits to produce human-level counterfactual explanations with zero training. Our framework utilizes a pre-trained image editing diffusion model, and operates without access to the internals of the classifier, leading to an explainable counterfactual generation process. Throughout our experimentation, we showcase the explanatory gap between human reasoning and neural model behavior by utilizing both Convolutional Neural Network (CNN), Vision Transformer (ViT) and Large Vision Language Model (LVLM) classifiers, substantiated through a comprehensive human evaluation.
>
---
#### [replaced 023] MedDiff-FM: A Diffusion-based Foundation Model for Versatile Medical Image Applications
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.15432v3](https://arxiv.org/pdf/2410.15432v3)**

> **作者:** Yongrui Yu; Yannian Gu; Shaoting Zhang; Xiaofan Zhang
>
> **摘要:** Diffusion models have achieved significant success in both natural image and medical image domains, encompassing a wide range of applications. Previous investigations in medical images have often been constrained to specific anatomical regions, particular applications, and limited datasets, resulting in isolated diffusion models. This paper introduces a diffusion-based foundation model to address a diverse range of medical image tasks, namely MedDiff-FM. MedDiff-FM leverages 3D CT images from multiple publicly available datasets, covering anatomical regions from head to abdomen, to pre-train a diffusion foundation model, and explores the capabilities of the diffusion foundation model across a variety of application scenarios. The diffusion foundation model handles multi-level integrated image processing both at the image-level and patch-level, utilizes position embedding to establish multi-level spatial relationships, and leverages region classes and anatomical structures to capture certain anatomical regions. MedDiff-FM manages several downstream tasks seamlessly, including image denoising, anomaly detection, and image synthesis. MedDiff-FM is also capable of performing super-resolution, lesion generation, and lesion inpainting by rapidly fine-tuning the diffusion foundation model using ControlNet with task-specific conditions. The experimental results demonstrate the effectiveness of MedDiff-FM in addressing diverse downstream medical image tasks.
>
---
#### [replaced 024] SAT: Dynamic Spatial Aptitude Training for Multimodal Language Models
- **分类: cs.CV; cs.AI; cs.GR; cs.RO**

- **简介: 该论文聚焦多模态语言模型的空间推理能力，旨在提升其对静态与动态空间关系的理解。作者构建了基于3D仿真的SAT数据集，通过模拟生成高质量标注的静态和动态空间问答数据，有效增强了模型在真实场景中的空间认知表现。**

- **链接: [https://arxiv.org/pdf/2412.07755v3](https://arxiv.org/pdf/2412.07755v3)**

> **作者:** Arijit Ray; Jiafei Duan; Ellis Brown; Reuben Tan; Dina Bashkirova; Rose Hendrix; Kiana Ehsani; Aniruddha Kembhavi; Bryan A. Plummer; Ranjay Krishna; Kuo-Hao Zeng; Kate Saenko
>
> **备注:** Accepted to COLM 2025. Project webpage: https://arijitray.com/SAT/
>
> **摘要:** Reasoning about motion and space is a fundamental cognitive capability that is required by multiple real-world applications. While many studies highlight that large multimodal language models (MLMs) struggle to reason about space, they only focus on static spatial relationships, and not dynamic awareness of motion and space, i.e., reasoning about the effect of egocentric and object motions on spatial relationships. Manually annotating such object and camera movements is expensive. Hence, we introduce SAT, a simulated spatial aptitude training dataset utilizing 3D simulators, comprising both static and dynamic spatial reasoning across 175K question-answer (QA) pairs and 20K scenes. Complementing this, we also construct a small (150 image-QAs) yet challenging dynamic spatial test set using real-world images. Leveraging our SAT datasets and 6 existing static spatial benchmarks, we systematically investigate what improves both static and dynamic spatial awareness. Our results reveal that simulations are surprisingly effective at imparting spatial aptitude to MLMs that translate to real images. We show that perfect annotations in simulation are more effective than existing approaches of pseudo-annotating real images. For instance, SAT training improves a LLaVA-13B model by an average 11% and a LLaVA-Video-7B model by an average 8% on multiple spatial benchmarks, including our real-image dynamic test set and spatial reasoning on long videos -- even outperforming some large proprietary models. While reasoning over static relationships improves with synthetic training data, there is still considerable room for improvement for dynamic reasoning questions.
>
---
#### [replaced 025] SAGE: Saliency-Guided Contrastive Embeddings
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12744v2](https://arxiv.org/pdf/2511.12744v2)**

> **作者:** Colton R. Crum; Christopher Sweet; Adam Czajka
>
> **备注:** 11 pages, 2 figures, 5 tables
>
> **摘要:** Integrating human perceptual priors into the training of neural networks has been shown to raise model generalization, serve as an effective regularizer, and align models with human expertise for applications in high-risk domains. Existing approaches to integrate saliency into model training often rely on internal model mechanisms, which recent research suggests may be unreliable. Our insight is that many challenges associated with saliency-guided training stem from the placement of the guidance approaches solely within the image space. Instead, we move away from the image space, use the model's latent space embeddings to steer human guidance during training, and we propose SAGE (Saliency-Guided Contrastive Embeddings): a loss function that integrates human saliency into network training using contrastive embeddings. We apply salient-preserving and saliency-degrading signal augmentations to the input and capture the changes in embeddings and model logits. We guide the model towards salient features and away from non-salient features using a contrastive triplet loss. Additionally, we perform a sanity check on the logit distributions to ensure that the model outputs match the saliency-based augmentations. We demonstrate a boost in classification performance across both open- and closed-set scenarios against SOTA saliency-based methods, showing SAGE's effectiveness across various backbones, and include experiments to suggest its wide generalization across tasks.
>
---
#### [replaced 026] A Fractional Variational Approach to Spectral Filtering Using the Fourier Transform
- **分类: eess.IV; cs.CV; math-ph**

- **链接: [https://arxiv.org/pdf/2511.20675v2](https://arxiv.org/pdf/2511.20675v2)**

> **作者:** Nelson H. T. Lemes; José Claudinei Ferreira; Higor V. M. Ferreira
>
> **备注:** 31 pages, 3 figures, 2 tables
>
> **摘要:** The interference of fluorescence signals and noise remains a significant challenge in Raman spectrum analysis, often obscuring subtle spectral features that are critical for accurate analysis. Inspired by variational methods similar to those used in image denoising, our approach minimizes a functional involving fractional derivatives to balance noise suppression with the preservation of essential chemical features of the signal, such as peak position, intensity, and area. The original problem is reformulated in the frequency domain through the Fourier transform, making the implementation simple and fast. In this work, we discuss the theoretical framework, practical implementation, and the advantages and limitations of this method in the context of {simulated} Raman data, as well as in image processing. The main contribution of this article is the combination of a variational approach in the frequency domain, the use of fractional derivatives, and the optimization of the {regularization parameter and} derivative order through the concept of Shannon entropy. This work explores how the fractional order, combined with the regularization parameter, affects noise removal and preserves the essential features of the spectrum {and image}. Finally, the study shows that the combination of the proposed strategies produces an efficient, robust, and easily implementable filter.
>
---
#### [replaced 027] Test-Time 3D Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08485v3](https://arxiv.org/pdf/2503.08485v3)**

> **作者:** Fengyi Zhang; Xiangyu Sun; Huitong Yang; Zheng Zhang; Zi Huang; Yadan Luo
>
> **摘要:** Self-supervised 3D occupancy prediction offers a promising solution for understanding complex driving scenes without requiring costly 3D annotations. However, training dense occupancy decoders to capture fine-grained geometry and semantics can demand hundreds of GPU hours, and once trained, such models struggle to adapt to varying voxel resolutions or novel object categories without extensive retraining. To overcome these limitations, we propose a practical and flexible test-time occupancy prediction framework termed TT-Occ. Our method incrementally constructs, optimizes and voxelizes time-aware 3D Gaussians from raw sensor streams by integrating vision foundation models (VFMs) at runtime. The flexible nature of 3D Gaussians allows voxelization at arbitrary user-specified resolutions, while the generalization ability of VFMs enables accurate perception and open-vocabulary recognition, without any network training or fine-tuning. Specifically, TT-Occ operates in a lift-track-voxelize symphony: We first lift the geometry and semantics of surrounding-view extracted from VFMs to instantiate Gaussians at 3D space; Next, we track dynamic Gaussians while accumulating static ones to complete the scene and enforce temporal consistency; Finally, we voxelize the optimized Gaussians to generate occupancy prediction. Optionally, inherent noise in VFM predictions and tracking is mitigated by periodically smoothing neighboring Gaussians during optimization. To validate the generality and effectiveness of our framework, we offer two variants: one LiDAR-based and one vision-centric, and conduct extensive experiments on Occ3D and nuCraft benchmarks with varying voxel resolutions.
>
---
#### [replaced 028] From Correlation to Causation: Max-Pooling-Based Multi-Instance Learning Leads to More Robust Whole Slide Image Classification
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2408.09449v3](https://arxiv.org/pdf/2408.09449v3)**

> **作者:** Xin Liu; Weijia Zhang; Wei Tang; Thuc Duy Le; Jiuyong Li; Lin Liu; Min-Ling Zhang
>
> **摘要:** In whole slide images (WSIs) analysis, attention-based multi-instance learning (MIL) models are susceptible to spurious correlations and degrade under domain shift. These methods may assign high attention weights to non-tumor regions, such as staining biases or artifacts, leading to unreliable tumor region localization. In this paper, we revisit max-pooling-based MIL methods from a causal perspective. Under mild assumptions, our theoretical results demonstrate that max-pooling encourages the model to focus on causal factors while ignoring bias-related factors. Furthermore, we discover that existing max-pooling-based methods may overfit the training set through rote memorization of instance features and fail to learn meaningful patterns. To address these issues, we propose FocusMIL, which couples max-pooling with an instance-level variational information bottleneck (VIB) to learn compact, predictive latent representations, and employs a multi-bag mini-batch scheme to stabilize optimization. We conduct comprehensive experiments on three real-world datasets and one semi-synthetic dataset. The results show that, by capturing causal factors, FocusMIL exhibits significant advantages in out-of-distribution scenarios and instance-level tumor region localization tasks.
>
---
#### [replaced 029] You Only Train Once (YOTO): A Retraining-Free Object Detection Framework
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04888v2](https://arxiv.org/pdf/2512.04888v2)**

> **作者:** Priyanto Hidayatullah; Nurjannah Syakrani; Yudi Widhiyasana; Muhammad Rizqi Sholahuddin; Refdinal Tubagus; Zahri Al Adzani Hidayat; Hanri Fajar Ramadhan; Dafa Alfarizki Pratama; Farhan Muhammad Yasin
>
> **备注:** This manuscript was first submitted to the Engineering (Elsevier Journal). The preprint version was posted to arXiv afterwards to facilitate open access and community feedback
>
> **摘要:** Object detection constitutes the primary task within the domain of computer vision. It is utilized in numerous domains. Nonetheless, object detection continues to encounter the issue of catastrophic forgetting. The model must be retrained whenever new products are introduced, utilizing not only the new products dataset but also the entirety of the previous dataset. The outcome is obvious: increasing model training expenses and significant time consumption. In numerous sectors, particularly retail checkout, the frequent introduction of new products presents a great challenge. This study introduces You Only Train Once (YOTO), a methodology designed to address the issue of catastrophic forgetting by integrating YOLO11n for object localization with DeIT and Proxy Anchor Loss for feature extraction and metric learning. For classification, we utilize cosine similarity between the embedding features of the target product and those in the Qdrant vector database. In a case study conducted in a retail store with 140 products, the experimental results demonstrate that our proposed framework achieves encouraging accuracy, whether for detecting new or existing products. Furthermore, without retraining, the training duration difference is significant. We achieve almost 3 times the training time efficiency compared to classical object detection approaches. This efficiency escalates as additional new products are added to the product database. The average inference time is 580 ms per image containing multiple products, on an edge device, validating the proposed framework's feasibility for practical use.
>
---
#### [replaced 030] COOPER: A Unified Model for Cooperative Perception and Reasoning in Spatial Intelligence
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04563v2](https://arxiv.org/pdf/2512.04563v2)**

> **作者:** Zefeng Zhang; Xiangzhao Hao; Hengzhu Tang; Zhenyu Zhang; Jiawei Sheng; Xiaodong Li; Zhenyang Li; Li Gao; Daiting Shi; Dawei Yin; Tingwen Liu
>
> **摘要:** Visual Spatial Reasoning is crucial for enabling Multimodal Large Language Models (MLLMs) to understand object properties and spatial relationships, yet current models still struggle with 3D-aware reasoning. Existing approaches typically enhance either perception, by augmenting RGB inputs with auxiliary modalities such as depth and segmentation, or reasoning, by training on spatial VQA datasets and applying reinforcement learning, and thus treat these two aspects in isolation. In this work, we investigate whether a unified MLLM can develop an intrinsic ability to enhance spatial perception and, through adaptive interleaved reasoning, achieve stronger spatial intelligence. We propose \textbf{COOPER}, a unified MLLM that leverages depth and segmentation as auxiliary modalities and is trained in two stages to acquire auxiliary modality generation and adaptive, interleaved reasoning capabilities. COOPER achieves an average \textbf{6.91\%} improvement in spatial reasoning while maintaining general performance. Moreover, even a variant trained only for auxiliary modality generation attains a \textbf{7.92\%} gain on distance and size estimation, suggesting that learning to generate auxiliary modalities helps internalize spatial knowledge and strengthen spatial understanding.
>
---
#### [replaced 031] Dressing the Imagination: A Dataset for AI-Powered Translation of Text into Fashion Outfits and A Novel KAN Adapter for Enhanced Feature Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.13901v4](https://arxiv.org/pdf/2411.13901v4)**

> **作者:** Gayatri Deshmukh; Somsubhra De; Chirag Sehgal; Jishu Sen Gupta; Sparsh Mittal
>
> **备注:** Accepted as a Conference Paper at WACV 2026 (USA)
>
> **摘要:** Specialized datasets that capture the fashion industry's rich language and styling elements can boost progress in AI-driven fashion design. We present FLORA, (Fashion Language Outfit Representation for Apparel Generation), the first comprehensive dataset containing 4,330 curated pairs of fashion outfits and corresponding textual descriptions. Each description utilizes industry-specific terminology and jargon commonly used by professional fashion designers, providing precise and detailed insights into the outfits. Hence, the dataset captures the delicate features and subtle stylistic elements necessary to create high-fidelity fashion designs. We demonstrate that fine-tuning generative models on the FLORA dataset significantly enhances their capability to generate accurate and stylistically rich images from textual descriptions of fashion sketches. FLORA will catalyze the creation of advanced AI models capable of comprehending and producing subtle, stylistically rich fashion designs. It will also help fashion designers and end-users to bring their ideas to life. As a second orthogonal contribution, we introduce NeRA (Nonlinear low-rank Expressive Representation Adapter), a novel adapter architecture based on Kolmogorov-Arnold Networks (KAN). Unlike traditional PEFT techniques such as LoRA, LoKR, DoRA, and LoHA that use MLP adapters, NeRA uses learnable spline-based nonlinear transformations, enabling superior modeling of complex semantic relationships, achieving strong fidelity, faster convergence and semantic alignment. Extensive experiments on our proposed FLORA and LAION-5B datasets validate the superiority of NeRA over existing adapters. We will open-source both the FLORA dataset and our implementation code.
>
---
#### [replaced 032] A Scene-aware Models Adaptation Scheme for Cross-scene Online Inference on Mobile Devices
- **分类: cs.CV; cs.AI; cs.DC**

- **链接: [https://arxiv.org/pdf/2407.03331v2](https://arxiv.org/pdf/2407.03331v2)**

> **作者:** Yunzhe Li; Hongzi Zhu; Zhuohong Deng; Yunlong Cheng; Zimu Zheng; Liang Zhang; Shan Chang; Minyi Guo
>
> **备注:** This version presents the extended and revised journal version of our 2024 conference paper, incorporating new datasets, expanded evaluations, and improved methodological details. The manuscript has been accepted for publication in IEEE Transactions on Mobile Computing
>
> **摘要:** Emerging Artificial Intelligence of Things (AIoT) applications desire online prediction using deep neural network (DNN) models on mobile devices. However, due to the movement of devices, unfamiliar test samples constantly appear, significantly affecting the prediction accuracy of a pre-trained DNN. In addition, unstable network connection calls for local model inference. In this paper, we propose a light-weight scheme, called Anole, to cope with the local DNN model inference on mobile devices. The core idea of Anole is to first establish an army of compact DNN models, and then adaptively select the model fitting the current test sample best for online inference. The key is to automatically identify model-friendly scenes for training scene-specific DNN models. To this end, we design a weakly-supervised scene representation learning algorithm by combining both human heuristics and feature similarity in separating scenes. Moreover, we further train a model classifier to predict the best-fit scene-specific DNN model for each test sample. We implement Anole on different types of mobile devices and conduct extensive trace-driven and real-world experiments based on unmanned aerial vehicles (UAVs). The results demonstrate that Anole outwits the method of using a versatile large DNN in terms of prediction accuracy (4.5% higher), response time (33.1% faster) and power consumption (45.1% lower).
>
---
#### [replaced 033] TempoControl: Temporal Attention Guidance for Text-to-Video Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.02226v2](https://arxiv.org/pdf/2510.02226v2)**

> **作者:** Shira Schiber; Ofir Lindenbaum; Idan Schwartz
>
> **备注:** Under Review
>
> **摘要:** Recent advances in generative video models have enabled the creation of high-quality videos based on natural language prompts. However, these models frequently lack fine-grained temporal control, meaning they do not allow users to specify when particular visual elements should appear within a generated sequence. In this work, we introduce TempoControl, a method that allows for temporal alignment of visual concepts during inference, without requiring retraining or additional supervision. TempoControl utilizes cross-attention maps, a key component of text-to-video diffusion models, to guide the timing of concepts through a novel optimization approach. Our method steers attention using three complementary principles: aligning its temporal pattern with a control signal (correlation), adjusting its strength where visibility is required (magnitude), and preserving semantic consistency (entropy). TempoControl provides precise temporal control while maintaining high video quality and diversity. We demonstrate its effectiveness across various applications, including temporal reordering of single and multiple objects, action timing, and audio-aligned video generation. Please see our project page for more details: https://shira-schiber.github.io/TempoControl/.
>
---
#### [replaced 034] Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in Dynamic Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究动态环境中可扩展的3D场景重建，旨在解决传统关键帧选择方法在实时感知中的数据瓶颈问题。作者提出一种自适应关键帧选择方法，结合误差评估与动量更新机制，提升重建质量，适用于机器人在复杂动态场景中的应用。**

- **链接: [https://arxiv.org/pdf/2510.23928v2](https://arxiv.org/pdf/2510.23928v2)**

> **作者:** Raman Jha; Yang Zhou; Giuseppe Loianno
>
> **备注:** Accepted at ROBOVIS 2026
>
> **摘要:** In this paper, we propose an adaptive keyframe selection method for improved 3D scene reconstruction in dynamic environments. The proposed method integrates two complementary modules: an error-based selection module utilizing photometric and structural similarity (SSIM) errors, and a momentum-based update module that dynamically adjusts keyframe selection thresholds according to scene motion dynamics. By dynamically curating the most informative frames, our approach addresses a key data bottleneck in real-time perception. This allows for the creation of high-quality 3D world representations from a compressed data stream, a critical step towards scalable robot learning and deployment in complex, dynamic environments. Experimental results demonstrate significant improvements over traditional static keyframe selection strategies, such as fixed temporal intervals or uniform frame skipping. These findings highlight a meaningful advancement toward adaptive perception systems that can dynamically respond to complex and evolving visual scenes. We evaluate our proposed adaptive keyframe selection module on two recent state-of-the-art 3D reconstruction networks, Spann3r and CUT3R, and observe consistent improvements in reconstruction quality across both frameworks. Furthermore, an extensive ablation study confirms the effectiveness of each individual component in our method, underlining their contribution to the overall performance gains.
>
---
#### [replaced 035] Language-Instructed Reasoning for Group Activity Detection via Multimodal Large Language Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.16054v2](https://arxiv.org/pdf/2509.16054v2)**

> **作者:** Jihua Peng; Qianxiong Xu; Yichen Liu; Chenxi Liu; Cheng Long; Rui Zhao; Ziyue Li
>
> **备注:** This work is being incorporated into a larger study
>
> **摘要:** Group activity detection (GAD) aims to simultaneously identify group members and categorize their collective activities within video sequences. Existing deep learning-based methods develop specialized architectures (e.g., transformer networks) to model the dynamics of individual roles and semantic dependencies between individuals and groups. However, they rely solely on implicit pattern recognition from visual features and struggle with contextual reasoning and explainability. In this work, we propose LIR-GAD, a novel framework of language-instructed reasoning for GAD via Multimodal Large Language Model (MLLM). Our approach expand the original vocabulary of MLLM by introducing an activity-level <ACT> token and multiple cluster-specific <GROUP> tokens. We process video frames alongside two specially designed tokens and language instructions, which are then integrated into the MLLM. The pretrained commonsense knowledge embedded in the MLLM enables the <ACT> token and <GROUP> tokens to effectively capture the semantic information of collective activities and learn distinct representational features of different groups, respectively. Also, we introduce a multi-label classification loss to further enhance the <ACT> token's ability to learn discriminative semantic representations. Then, we design a Multimodal Dual-Alignment Fusion (MDAF) module that integrates MLLM's hidden embeddings corresponding to the designed tokens with visual features, significantly enhancing the performance of GAD. Both quantitative and qualitative experiments demonstrate the superior performance of our proposed method in GAD taks.
>
---
#### [replaced 036] SOAP: Enhancing Spatio-Temporal Relation and Motion Information Capturing for Few-Shot Action Recognition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2407.16344v5](https://arxiv.org/pdf/2407.16344v5)**

> **作者:** Wenbo Huang; Jinghui Zhang; Xuwei Qian; Zhen Wu; Meng Wang; Lei Zhang
>
> **备注:** Accepted by ACM MM 2024
>
> **摘要:** High frame-rate (HFR) videos of action recognition improve fine-grained expression while reducing the spatio-temporal relation and motion information density. Thus, large amounts of video samples are continuously required for traditional data-driven training. However, samples are not always sufficient in real-world scenarios, promoting few-shot action recognition (FSAR) research. We observe that most recent FSAR works build spatio-temporal relation of video samples via temporal alignment after spatial feature extraction, cutting apart spatial and temporal features within samples. They also capture motion information via narrow perspectives between adjacent frames without considering density, leading to insufficient motion information capturing. Therefore, we propose a novel plug-and-play architecture for FSAR called Spatio-tempOral frAme tuPle enhancer (SOAP) in this paper. The model we designed with such architecture refers to SOAP-Net. Temporal connections between different feature channels and spatio-temporal relation of features are considered instead of simple feature extraction. Comprehensive motion information is also captured, using frame tuples with multiple frames containing more motion information than adjacent frames. Combining frame tuples of diverse frame counts further provides a broader perspective. SOAP-Net achieves new state-of-the-art performance across well-known benchmarks such as SthSthV2, Kinetics, UCF101, and HMDB51. Extensive empirical evaluations underscore the competitiveness, pluggability, generalization, and robustness of SOAP. The code is released at https://github.com/wenbohuang1002/SOAP.
>
---
#### [replaced 037] InfiniBench: Infinite Benchmarking for Visual Spatial Reasoning with Customizable Scene Complexity
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18200v2](https://arxiv.org/pdf/2511.18200v2)**

> **作者:** Haoming Wang; Qiyao Xue; Wei Gao
>
> **摘要:** Modern vision-language models (VLMs) are expected to have abilities of spatial reasoning with diverse scene complexities, but evaluating such abilities is difficult due to the lack of benchmarks that are not only diverse and scalable but also fully customizable. Existing benchmarks offer limited customizability over the scene complexity and are incapable of isolating and analyzing specific VLM failure modes under distinct spatial conditions. To address this gap, instead of individually presenting benchmarks for different scene complexities, in this paper we present InfiniBench, a fully automated, customizable and user-friendly benchmark generator that can synthesize a theoretically infinite variety of 3D scenes with parameterized control on scene complexity. InfiniBench uniquely translates scene descriptions in natural language into photo-realistic videos with complex and physically plausible 3D layouts. This is achieved through three key innovations: 1) a LLM-based agentic framework that iteratively refines procedural scene constraints from scene descriptions; 2) a flexible cluster-based layout optimizer that generates dense and cluttered scenes previously intractable for procedural methods; and 3) a task-aware camera trajectory optimization method that renders scenes into videos with full object coverage as VLM input. Experiments demonstrate that InfiniBench outperforms state-of-the-art procedural and LLM-based 3D generation methods in prompt fidelity and physical plausibility, especially in high-complexity scenarios. We further showcased the usefulness of InfiniBench, by generating benchmarks for representative spatial reasoning tasks including measurement, perspective-taking and spatiotemporal tracking.
>
---
#### [replaced 038] Perspective-Invariant 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究LiDAR-based 3D目标检测，针对非车载平台数据缺失和跨平台适应问题，构建了多平台基准Pi3DET，提出几何与特征级对齐的跨平台自适应框架，实现视角不变的3D检测，推动通用3D感知系统发展。**

- **链接: [https://arxiv.org/pdf/2507.17665v2](https://arxiv.org/pdf/2507.17665v2)**

> **作者:** Ao Liang; Lingdong Kong; Dongyue Lu; Youquan Liu; Jian Fang; Huaici Zhao; Wei Tsang Ooi
>
> **备注:** ICCV 2025; 54 pages, 18 figures, 22 tables; Project Page at https://pi3det.github.io
>
> **摘要:** With the rise of robotics, LiDAR-based 3D object detection has garnered significant attention in both academia and industry. However, existing datasets and methods predominantly focus on vehicle-mounted platforms, leaving other autonomous platforms underexplored. To bridge this gap, we introduce Pi3DET, the first benchmark featuring LiDAR data and 3D bounding box annotations collected from multiple platforms: vehicle, quadruped, and drone, thereby facilitating research in 3D object detection for non-vehicle platforms as well as cross-platform 3D detection. Based on Pi3DET, we propose a novel cross-platform adaptation framework that transfers knowledge from the well-studied vehicle platform to other platforms. This framework achieves perspective-invariant 3D detection through robust alignment at both geometric and feature levels. Additionally, we establish a benchmark to evaluate the resilience and robustness of current 3D detectors in cross-platform scenarios, providing valuable insights for developing adaptive 3D perception systems. Extensive experiments validate the effectiveness of our approach on challenging cross-platform tasks, demonstrating substantial gains over existing adaptation methods. We hope this work paves the way for generalizable and unified 3D perception systems across diverse and complex environments. Our Pi3DET dataset, cross-platform benchmark suite, and annotation toolkit have been made publicly available.
>
---
#### [replaced 039] Joint Self-Supervised Video Alignment and Action Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.16832v3](https://arxiv.org/pdf/2503.16832v3)**

> **作者:** Ali Shah Ali; Syed Ahmed Mahmood; Mubin Saeed; Andrey Konin; M. Zeeshan Zia; Quoc-Huy Tran
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** We introduce a novel approach for simultaneous self-supervised video alignment and action segmentation based on a unified optimal transport framework. In particular, we first tackle self-supervised video alignment by developing a fused Gromov-Wasserstein optimal transport formulation with a structural prior, which trains efficiently on GPUs and needs only a few iterations for solving the optimal transport problem. Our single-task method achieves the state-of-the-art performance on multiple video alignment benchmarks and outperforms VAVA, which relies on a traditional Kantorovich optimal transport formulation with an optimality prior. Furthermore, we extend our approach by proposing a unified optimal transport framework for joint self-supervised video alignment and action segmentation, which requires training and storing a single model and saves both time and memory consumption as compared to two different single-task models. Extensive evaluations on several video alignment and action segmentation datasets demonstrate that our multi-task method achieves comparable video alignment yet superior action segmentation results over previous methods in video alignment and action segmentation respectively. Finally, to the best of our knowledge, this is the first work to unify video alignment and action segmentation into a single model. Our code is available on our research website: https://retrocausal.ai/research/.
>
---
#### [replaced 040] TabletopGen: Instance-Level Interactive 3D Tabletop Scene Generation from Text or Single Image
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01204v3](https://arxiv.org/pdf/2512.01204v3)**

> **作者:** Ziqian Wang; Yonghao He; Licheng Yang; Wei Zou; Hongxuan Ma; Liu Liu; Wei Sui; Yuxin Guo; Hu Su
>
> **备注:** Project page: https://d-robotics-ai-lab.github.io/TabletopGen.project/
>
> **摘要:** Generating high-fidelity, physically interactive 3D simulated tabletop scenes is essential for embodied AI--especially for robotic manipulation policy learning and data synthesis. However, current text- or image-driven 3D scene generation methods mainly focus on large-scale scenes, struggling to capture the high-density layouts and complex spatial relations that characterize tabletop scenes. To address these challenges, we propose TabletopGen, a training-free, fully automatic framework that generates diverse, instance-level interactive 3D tabletop scenes. TabletopGen accepts a reference image as input, which can be synthesized by a text-to-image model to enhance scene diversity. We then perform instance segmentation and completion on the reference to obtain per-instance images. Each instance is reconstructed into a 3D model followed by canonical coordinate alignment. The aligned 3D models then undergo pose and scale estimation before being assembled into a collision-free, simulation-ready tabletop scene. A key component of our framework is a novel pose and scale alignment approach that decouples the complex spatial reasoning into two stages: a Differentiable Rotation Optimizer for precise rotation recovery and a Top-view Spatial Alignment mechanism for robust translation and scale estimation, enabling accurate 3D reconstruction from 2D reference. Extensive experiments and user studies show that TabletopGen achieves state-of-the-art performance, markedly surpassing existing methods in visual fidelity, layout accuracy, and physical plausibility, capable of generating realistic tabletop scenes with rich stylistic and spatial diversity. Our code will be publicly available.
>
---
#### [replaced 041] Exploring Ordinal Bias in Action Recognition for Instructional Videos
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.06580v2](https://arxiv.org/pdf/2504.06580v2)**

> **作者:** Joochan Kim; Minjoon Jung; Byoung-Tak Zhang
>
> **备注:** Accepted at SCSL @ ICLR 2025
>
> **摘要:** Action recognition models have achieved promising results in understanding instructional videos. However, they often rely on dominant, dataset-specific action sequences rather than true video comprehension, a problem that we define as ordinal bias. To address this issue, we propose two effective video manipulation methods: Action Masking, which masks frames of frequently co-occurring actions, and Sequence Shuffling, which randomizes the order of action segments. Through comprehensive experiments, we demonstrate that current models exhibit significant performance drops when confronted with nonstandard action sequences, underscoring their vulnerability to ordinal bias. Our findings emphasize the importance of rethinking evaluation strategies and developing models capable of generalizing beyond fixed action patterns in diverse instructional videos.
>
---
#### [replaced 042] HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.16848v3](https://arxiv.org/pdf/2503.16848v3)**

> **作者:** Hou In Derek Pun; Hou In Ivan Tam; Austin T. Wang; Xiaoliang Huo; Angel X. Chang; Manolis Savva
>
> **备注:** Accepted at 3DV 2026; 29 pages with 11 figures and 6 tables; Camera-ready with additional discussion
>
> **摘要:** Despite advances in indoor 3D scene layout generation, synthesizing scenes with dense object arrangements remains challenging. Existing methods focus on large furniture while neglecting smaller objects, resulting in unrealistically empty scenes. Those that place small objects typically do not honor arrangement specifications, resulting in largely random placement not following the text description. We present Hierarchical Scene Motifs (HSM): a hierarchical framework for indoor scene generation with dense object arrangements across spatial scales. Indoor scenes are inherently hierarchical, with surfaces supporting objects at different scales, from large furniture on floors to smaller objects on tables and shelves. HSM embraces this hierarchy and exploits recurring cross-scale spatial patterns to generate complex and realistic scenes in a unified manner. Our experiments show that HSM outperforms existing methods by generating scenes that better conform to user input across room types and spatial configurations. Project website is available at https://3dlg-hcvc.github.io/hsm .
>
---
#### [replaced 043] EMMA: Efficient Multimodal Understanding, Generation, and Editing with a Unified Architecture
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04810v2](https://arxiv.org/pdf/2512.04810v2)**

> **作者:** Xin He; Longhui Wei; Jianbo Ouyang; Lingxi Xie; Qi Tian
>
> **备注:** Project Page: https://emma-umm.github.io/emma/
>
> **摘要:** We propose EMMA, an efficient and unified architecture for multimodal understanding, generation and editing. Specifically, EMMA primarily consists of 1) An efficient autoencoder with a 32x compression ratio, which significantly reduces the number of tokens required for generation. This also ensures the training balance between understanding and generation tasks by applying the same compression ratio to images. 2) Channel-wise concatenation instead of token-wise concatenation among visual understanding and generation tokens, which further reduces the visual tokens in unified architectures. 3) A shared-and-decoupled network that enables mutual improvements across tasks while meeting the task-specific modeling requirements. 4) A mixture-of-experts mechanism adopted for visual understanding encoder, which substantially improves perceptual capabilities with a few parameters increase. Extensive experiments have shown that EMMA-4B can significantly outperform state-of-the-art unified multimodal approaches (e.g., BAGEL-7B) in both efficiency and performance, while also achieving competitive results compared to recent multimodal understanding and generation experts (e.g., Qwen3-VL and Qwen-Image). We believe that EMMA lays a solid foundation for the future development of unified multimodal architectures.
>
---
#### [replaced 044] Adaptive Chain-of-Focus Reasoning via Dynamic Visual Search and Zooming for Efficient VLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15436v3](https://arxiv.org/pdf/2505.15436v3)**

> **作者:** Xintong Zhang; Zhi Gao; Bofei Zhang; Pengxiang Li; Xiaowen Zhang; Yang Liu; Tao Yuan; Yuwei Wu; Yunde Jia; Song-Chun Zhu; Qing Li
>
> **备注:** https://github.com/xtong-zhang/Chain-of-Focus
>
> **摘要:** Vision language models (VLMs) have achieved impressive performance across a variety of computer vision tasks. However, the multimodal reasoning capability has not been fully explored in existing models. In this paper, we propose a Chain-of-Focus (CoF) method that allows VLMs to perform adaptive focusing and zooming in on key image regions based on obtained visual cues and the given questions, achieving efficient multimodal reasoning. To enable this CoF capability, we present a two-stage training pipeline, including supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we construct the MM-CoF dataset, comprising 3K samples derived from a visual agent designed to adaptively identify key regions to solve visual tasks with different image resolutions and questions. We use MM-CoF to fine-tune the Qwen2.5-VL model for cold start. In the RL stage, we leverage the outcome accuracies and formats as rewards to update the Qwen2.5-VL model, enabling further refining the search and reasoning strategy of models without human priors. Our model achieves significant improvements on multiple benchmarks. On the V* benchmark that requires strong visual reasoning capability, our model outperforms existing VLMs by 5% among 8 image resolutions ranging from 224 to 4K, demonstrating the effectiveness of the proposed CoF method and facilitating the more efficient deployment of VLMs in practical applications.
>
---
#### [replaced 045] AortaDiff: A Unified Multitask Diffusion Framework For Contrast-Free AAA Imaging
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.01498v2](https://arxiv.org/pdf/2510.01498v2)**

> **作者:** Yuxuan Ou; Ning Bi; Jiazhen Pan; Jiancheng Yang; Boliang Yu; Usama Zidan; Regent Lee; Vicente Grau
>
> **备注:** WACV 2026
>
> **摘要:** While contrast-enhanced CT (CECT) is standard for assessing abdominal aortic aneurysms (AAA), the required iodinated contrast agents pose significant risks, including nephrotoxicity, patient allergies, and environmental harm. To reduce contrast agent use, recent deep learning methods have focused on generating synthetic CECT from non-contrast CT (NCCT) scans. However, most adopt a multi-stage pipeline that first generates images and then performs segmentation, which leads to error accumulation and fails to leverage shared semantic and anatomical structures. To address this, we propose a unified deep learning framework that generates synthetic CECT images from NCCT scans while simultaneously segmenting the aortic lumen and thrombus. Our approach integrates conditional diffusion models (CDM) with multi-task learning, enabling end-to-end joint optimization of image synthesis and anatomical segmentation. Unlike previous multitask diffusion models, our approach requires no initial predictions (e.g., a coarse segmentation mask), shares both encoder and decoder parameters across tasks, and employs a semi-supervised training strategy to learn from scans with missing segmentation labels, a common constraint in real-world clinical data. We evaluated our method on a cohort of 264 patients, where it consistently outperformed state-of-the-art single-task and multi-stage models. For image synthesis, our model achieved a PSNR of 25.61 dB, compared to 23.80 dB from a single-task CDM. For anatomical segmentation, it improved the lumen Dice score to 0.89 from 0.87 and the challenging thrombus Dice score to 0.53 from 0.48 (nnU-Net). These segmentation enhancements led to more accurate clinical measurements, reducing the lumen diameter MAE to 4.19 mm from 5.78 mm and the thrombus area error to 33.85% from 41.45% when compared to nnU-Net. Code is available at https://github.com/yuxuanou623/AortaDiff.git.
>
---
#### [replaced 046] Collaborative Face Experts Fusion in Video Generation: Boosting Identity Consistency Across Large Face Poses
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.09476v4](https://arxiv.org/pdf/2508.09476v4)**

> **作者:** Yuji Wang; Moran Li; Xiaobin Hu; Ran Yi; Jiangning Zhang; Chengming Xu; Weijian Cao; Yabiao Wang; Chengjie Wang; Lizhuang Ma
>
> **备注:** Project page: https://rain152.github.io/CoFE/
>
> **摘要:** Current video generation models struggle with identity preservation under large face poses, primarily facing two challenges: the difficulty in exploring an effective mechanism to integrate identity features into DiT architectures, and the lack of targeted coverage of large face poses in existing open-source video datasets. To address these, we present two key innovations. First, we propose Collaborative Face Experts Fusion (CoFE), which dynamically fuses complementary signals from three specialized experts within the DiT backbone: an identity expert that captures cross-pose invariant features, a semantic expert that encodes high-level visual context, and a detail expert that preserves pixel-level attributes such as skin texture and color gradients. Second, we introduce a data curation pipeline comprising three key components: Face Constraints to ensure diverse large-pose coverage, Identity Consistency to maintain stable identity across frames, and Speech Disambiguation to align textual captions with actual speaking behavior. This pipeline yields LaFID-180K, a large-scale dataset of pose-annotated video clips designed for identity-preserving video generation. Experimental results on several benchmarks demonstrate that our approach significantly outperforms state-of-the-art methods in face similarity, FID, and CLIP semantic alignment. Project page: https://rain152.github.io/CoFE/.
>
---
#### [replaced 047] A Strong View-Free Baseline Approach for Single-View Image Guided Point Cloud Completion
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2506.15747v2](https://arxiv.org/pdf/2506.15747v2)**

> **作者:** Fangzhou Lin; Zilin Dai; Rigved Sanku; Songlin Hou; Kazunori D Yamada; Haichong K. Zhang; Ziming Zhang
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** The single-view image guided point cloud completion (SVIPC) task aims to reconstruct a complete point cloud from a partial input with the help of a single-view image. While previous works have demonstrated the effectiveness of this multimodal approach, the fundamental necessity of image guidance remains largely unexamined. To explore this, we propose a strong baseline approach for SVIPC based on an attention-based multi-branch encoder-decoder network that only takes partial point clouds as input, view-free. Our hierarchical self-fusion mechanism, driven by cross-attention and self-attention layers, effectively integrates information across multiple streams, enriching feature representations and strengthening the networks ability to capture geometric structures. Extensive experiments and ablation studies on the ShapeNet-ViPC dataset demonstrate that our view-free framework performs superiorly to state-of-the-art SVIPC methods. We hope our findings provide new insights into the development of multimodal learning in SVIPC. Our demo code will be available at https://github.com/Zhang-VISLab.
>
---
#### [replaced 048] Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04926v2](https://arxiv.org/pdf/2512.04926v2)**

> **作者:** Yueming Pan; Ruoyu Feng; Qi Dai; Yuqi Wang; Wenfeng Lin; Mingyu Guo; Chong Luo; Nanning Zheng
>
> **摘要:** Latent Diffusion Models (LDMs) inherently follow a coarse-to-fine generation process, where high-level semantic structure is generated slightly earlier than fine-grained texture. This indicates the preceding semantics potentially benefit texture generation by providing a semantic anchor. Recent advances have integrated semantic priors from pretrained visual encoders to further enhance LDMs, yet they still denoise semantic and VAE-encoded texture synchronously, neglecting such ordering. Observing these, we propose Semantic-First Diffusion (SFD), a latent diffusion paradigm that explicitly prioritizes semantic formation. SFD first constructs composite latents by combining a compact semantic latent, which is extracted from a pretrained visual encoder via a dedicated Semantic VAE, with the texture latent. The core of SFD is to denoise the semantic and texture latents asynchronously using separate noise schedules: semantics precede textures by a temporal offset, providing clearer high-level guidance for texture refinement and enabling natural coarse-to-fine generation. On ImageNet 256x256 with guidance, SFD achieves FID 1.06 (LightningDiT-XL) and FID 1.04 (1.0B LightningDiT-XXL), while achieving up to 100x faster convergence than the original DiT. SFD also improves existing methods like ReDi and VA-VAE, demonstrating the effectiveness of asynchronous, semantics-led modeling. Project page and code: https://yuemingpan.github.io/SFD.github.io/.
>
---
#### [replaced 049] SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control
- **分类: cs.RO; cs.AI; cs.CV; cs.GR; eess.SY**

- **简介: 该论文聚焦 humanoid 控制任务，旨在解决现有控制器泛化性差、依赖人工奖励的问题。提出 SONIC，通过扩大模型、数据与计算规模，利用动作捕捉数据进行运动跟踪，学习通用运动先验，并支持多种输入接口与实际任务迁移。**

- **链接: [https://arxiv.org/pdf/2511.07820v2](https://arxiv.org/pdf/2511.07820v2)**

> **作者:** Zhengyi Luo; Ye Yuan; Tingwu Wang; Chenran Li; Sirui Chen; Fernando Castañeda; Zi-Ang Cao; Jiefeng Li; David Minor; Qingwei Ben; Xingye Da; Runyu Ding; Cyrus Hogg; Lina Song; Edy Lim; Eugene Jeong; Tairan He; Haoru Xue; Wenli Xiao; Zi Wang; Simon Yuen; Jan Kautz; Yan Chang; Umar Iqbal; Linxi "Jim" Fan; Yuke Zhu
>
> **备注:** Project page: https://nvlabs.github.io/SONIC/
>
> **摘要:** Despite the rise of billion-parameter foundation models trained across thousands of GPUs, similar scaling gains have not been shown for humanoid control. Current neural controllers for humanoids remain modest in size, target a limited set of behaviors, and are trained on a handful of GPUs over several days. We show that scaling up model capacity, data, and compute yields a generalist humanoid controller capable of creating natural and robust whole-body movements. Specifically, we posit motion tracking as a natural and scalable task for humanoid control, leveraging dense supervision from diverse motion-capture data to acquire human motion priors without manual reward engineering. We build a foundation model for motion tracking by scaling along three axes: network size (from 1.2M to 42M parameters), dataset volume (over 100M frames, 700 hours of high-quality motion data), and compute (9k GPU hours). Beyond demonstrating the benefits of scale, we show the practical utility of our model through two mechanisms: (1) a real-time universal kinematic planner that bridges motion tracking to downstream task execution, enabling natural and interactive control, and (2) a unified token space that supports various motion input interfaces, such as VR teleoperation devices, human videos, and vision-language-action (VLA) models, all using the same policy. Scaling motion tracking exhibits favorable properties: performance improves steadily with increased compute and data diversity, and learned representations generalize to unseen motions, establishing motion tracking at scale as a practical foundation for humanoid control.
>
---
#### [replaced 050] Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉-语言-动作（VLA）模型，旨在解决现有模型参数多、依赖大量机器人数据、泛化差的问题。作者提出轻量级模型Evo-1，通过新架构和两阶段训练，在不依赖机器人预训练的情况下保持语义对齐，提升性能与部署效率。**

- **链接: [https://arxiv.org/pdf/2511.04555v2](https://arxiv.org/pdf/2511.04555v2)**

> **作者:** Tao Lin; Yilei Zhong; Yuxin Du; Jingjing Zhang; Jiting Liu; Yinxinyu Chen; Encheng Gu; Ziyan Liu; Hongyi Cai; Yanwen Zou; Lixing Zou; Zhaoye Zhou; Gen Li; Bo Zhao
>
> **备注:** Github: https://github.com/MINT-SJTU/Evo-1
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a powerful framework that unifies perception, language, and control, enabling robots to perform diverse tasks through multimodal understanding. However, current VLA models typically contain massive parameters and rely heavily on large-scale robot data pretraining, leading to high computational costs during training, as well as limited deployability for real-time inference. Moreover, most training paradigms often degrade the perceptual representations of the vision-language backbone, resulting in overfitting and poor generalization to downstream tasks. In this work, we present Evo-1, a lightweight VLA model that reduces computation and improves deployment efficiency, while maintaining strong performance without pretraining on robot data. Evo-1 builds on a native multimodal Vision-Language model (VLM), incorporating a novel cross-modulated diffusion transformer along with an optimized integration module, together forming an effective architecture. We further introduce a two-stage training paradigm that progressively aligns action with perception, preserving the representations of the VLM. Notably, with only 0.77 billion parameters, Evo-1 achieves state-of-the-art results on the Meta-World and RoboTwin suite, surpassing the previous best models by 12.4% and 6.9%, respectively, and also attains a competitive result of 94.8% on LIBERO. In real-world evaluations, Evo-1 attains a 78% success rate with high inference frequency and low memory overhead, outperforming all baseline methods. We release code, data, and model weights to facilitate future research on lightweight and efficient VLA models.
>
---
#### [replaced 051] Martian World Model: Controllable Video Synthesis with Physically Accurate 3D Reconstructions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.07978v2](https://arxiv.org/pdf/2507.07978v2)**

> **作者:** Longfei Li; Zhiwen Fan; Wenyan Cong; Xinhang Liu; Yuyang Yin; Matt Foutter; Panwang Pan; Chenyu You; Yue Wang; Zhangyang Wang; Yao Zhao; Marco Pavone; Yunchao Wei
>
> **备注:** Project Page: https://marsgenai.github.io
>
> **摘要:** Synthesizing realistic Martian landscape videos is crucial for mission rehearsal and robotic simulation. However, this task poses unique challenges due to the scarcity of high-quality Martian data and the significant domain gap between Martian and terrestrial imagery. To address these challenges, we propose a holistic solution composed of two key components: 1) A data curation pipeline Multimodal Mars Synthesis (M3arsSynth), which reconstructs 3D Martian environments from real stereo navigation images, sourced from NASA's Planetary Data System (PDS), and renders high-fidelity multiview 3D video sequences. 2) A Martian terrain video generator, MarsGen, which synthesizes novel videos visually realistic and geometrically consistent with the 3D structure encoded in the data. Our M3arsSynth engine spans a wide range of Martian terrains and acquisition dates, enabling the generation of physically accurate 3D surface models at metric-scale resolution. MarsGen, fine-tuned on M3arsSynth data, synthesizes videos conditioned on an initial image frame and, optionally, camera trajectories or textual prompts, allowing for video generation in novel environments. Experimental results show that our approach outperforms video synthesis models trained on terrestrial datasets, achieving superior visual fidelity and 3D structural consistency.
>
---
#### [replaced 052] 3D Question Answering via only 2D Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.22143v2](https://arxiv.org/pdf/2505.22143v2)**

> **作者:** Fengyun Wang; Sicheng Yu; Jiawei Wu; Jinhui Tang; Hanwang Zhang; Qianru Sun
>
> **备注:** ICML2025
>
> **摘要:** Large vision-language models (LVLMs) have significantly advanced numerous fields. In this work, we explore how to harness their potential to address 3D scene understanding tasks, using 3D question answering (3D-QA) as a representative example. Due to the limited training data in 3D, we do not train LVLMs but infer in a zero-shot manner. Specifically, we sample 2D views from a 3D point cloud and feed them into 2D models to answer a given question. When the 2D model is chosen, e.g., LLAVA-OV, the quality of sampled views matters the most. We propose cdViews, a novel approach to automatically selecting critical and diverse Views for 3D-QA. cdViews consists of two key components: viewSelector prioritizing critical views based on their potential to provide answer-specific information, and viewNMS enhancing diversity by removing redundant views based on spatial overlap. We evaluate cdViews on the widely-used ScanQA and SQA benchmarks, demonstrating that it achieves state-of-the-art performance in 3D-QA while relying solely on 2D models without fine-tuning. These findings support our belief that 2D LVLMs are currently the most effective alternative (of the resource-intensive 3D LVLMs) for addressing 3D tasks.
>
---
#### [replaced 053] TextureSplat: Per-Primitive Texture Mapping for Reflective Gaussian Splatting
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.13348v2](https://arxiv.org/pdf/2506.13348v2)**

> **作者:** Mae Younes; Adnane Boukhayma
>
> **备注:** 3DV 2026
>
> **摘要:** Gaussian Splatting have demonstrated remarkable novel view synthesis performance at high rendering frame rates. Optimization-based inverse rendering within complex capture scenarios remains however a challenging problem. A particular case is modelling complex surface light interactions for highly reflective scenes, which results in intricate high frequency specular radiance components. We hypothesize that such challenging settings can benefit from increased representation power. We hence propose a method that tackles this issue through a geometrically and physically grounded Gaussian Splatting borne radiance field, where normals and material properties are spatially variable in the primitive's local space. Using per-primitive texture maps for this purpose, we also propose to harness the GPU hardware to accelerate rendering at test time via unified material texture atlas. Code will be available at https://github.com/maeyounes/TextureSplat
>
---
#### [replaced 054] Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots from Video
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文研究软连续体机器人动力学建模，旨在提升数据驱动方法的物理可解释性。提出注意力广播解码器（ABCD）生成像素级注意力图，结合2D振荡器网络实现动态过程的可视化，无需先验知识即可学习紧凑、可解释的模型，并显著提高预测精度。**

- **链接: [https://arxiv.org/pdf/2511.18322v2](https://arxiv.org/pdf/2511.18322v2)**

> **作者:** Henrik Krauss; Johann Licher; Naoya Takeishi; Annika Raatz; Takehisa Yairi
>
> **备注:** Dataset available at: https://zenodo.org/records/17812071
>
> **摘要:** Data-driven learning of soft continuum robot (SCR) dynamics from high-dimensional observations offers flexibility but often lacks physical interpretability, while model-based approaches require prior knowledge and can be computationally expensive. We bridge this gap by introducing (1) the Attention Broadcast Decoder (ABCD), a plug-and-play module for autoencoder-based latent dynamics learning that generates pixel-accurate attention maps localizing each latent dimension's contribution while filtering static backgrounds. (2) By coupling these attention maps to 2D oscillator networks, we enable direct on-image visualization of learned dynamics (masses, stiffness, and forces) without prior knowledge. We validate our approach on single- and double-segment SCRs, demonstrating that ABCD-based models significantly improve multi-step prediction accuracy: 5.7x error reduction for Koopman operators and 3.5x for oscillator networks on the two-segment robot. The learned oscillator network autonomously discovers a chain structure of oscillators. Unlike standard methods, ABCD models enable smooth latent space extrapolation beyond training data. This fully data-driven approach yields compact, physically interpretable models suitable for control applications.
>
---
#### [replaced 055] Open-PMC-18M: A High-Fidelity Large Scale Medical Dataset for Multimodal Representation Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.02738v3](https://arxiv.org/pdf/2506.02738v3)**

> **作者:** Negin Baghbanzadeh; Mohammed Saidul Islam; Sajad Ashkezari; Elham Dolatabadi; Arash Afkanpour
>
> **备注:** 21 pages
>
> **摘要:** In biomedical vision-language modeling, datasets are typically mined from scientific literature, pairing compound figures with captions that are short, context-dependent, and oftern partially informative. Prior work on subfigure extraction has been limited in both dataset size and generalizability. In addition, no existing effort has incorporated rich medical context in image-text pairs. We revisit data curation as a foundational component of effective biomedical representation learning. Our data curation process integrates transformer-based subfigure detection, subcaption extraction, and contextual text enrichment derived from inline references. Our subfigure extraction model, trained on a corpus of 500,000 compound figures, achieves state-of-the-art performance on real and synthetic benchmarks. Using this process, we curate and release Open-PMC-18M, a large-scale high-fidelity biomedical dataset comprising 18 million image-text pairs, spanning radiology, microscopy, and visible light photography. We train vision-language models on our dataset and perform extensive evaluation on 6 retrieval and 19 zero-shot classification tasks across three major modalities. The models trained on our dataset set a new state-of-the-art results in medical representation learning. We release our dataset, models, and code to support reproducible benchmarks and further study into biomedical vision-language modeling and representation learning.
>
---
#### [replaced 056] Variational Supervised Contrastive Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.07413v3](https://arxiv.org/pdf/2506.07413v3)**

> **作者:** Ziwen Wang; Jiajun Fan; Thao Nguyen; Heng Ji; Ge Liu
>
> **摘要:** Contrastive learning has proven to be highly efficient and adaptable in shaping representation spaces across diverse modalities by pulling similar samples together and pushing dissimilar ones apart. However, two key limitations persist: (1) Without explicit regulation of the embedding distribution, semantically related instances can inadvertently be pushed apart unless complementary signals guide pair selection, and (2) excessive reliance on large in-batch negatives and tailored augmentations hinders generalization. To address these limitations, we propose Variational Supervised Contrastive Learning (VarCon), which reformulates supervised contrastive learning as variational inference over latent class variables and maximizes a posterior-weighted evidence lower bound (ELBO) that replaces exhaustive pair-wise comparisons for efficient class-aware matching and grants fine-grained control over intra-class dispersion in the embedding space. Trained exclusively on image data, our experiments on CIFAR-10, CIFAR-100, ImageNet-100, and ImageNet-1K show that VarCon (1) achieves state-of-the-art performance for contrastive learning frameworks, reaching 79.36% Top-1 accuracy on ImageNet-1K and 78.29% on CIFAR-100 with a ResNet-50 encoder while converging in just 200 epochs; (2) yields substantially clearer decision boundaries and semantic organization in the embedding space, as evidenced by KNN classification, hierarchical clustering results, and transfer-learning assessments; and (3) demonstrates superior performance in few-shot learning than supervised baseline and superior robustness across various augmentation strategies. Our code is available at https://github.com/ziwenwang28/VarContrast.
>
---
#### [replaced 057] VirDA: Reusing Backbone for Unsupervised Domain Adaptation with Visual Reprogramming
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.01660v4](https://arxiv.org/pdf/2510.01660v4)**

> **作者:** Duy Nguyen; Dat Nguyen
>
> **备注:** To be published in TMLR
>
> **摘要:** Existing UDA pipelines fine-tune already well-trained backbone parameters for every new source-and-target pair, resulting in the number of training parameters and storage memory growing linearly with each new pair, and also preventing the reuse of these well-trained backbone parameters. Inspired by recent implications that existing backbones have textural biases, we propose making use of domain-specific textural bias for domain adaptation via visual reprogramming, namely VirDA. Instead of fine-tuning the full backbone, VirDA prepends a domain-specific visual reprogramming layer to the backbone. This layer produces visual prompts that act as an added textural bias to the input image, adapting its "style" to a target domain. To optimize these visual reprogramming layers, we use multiple objective functions that optimize the intra- and inter-domain distribution differences when domain-adapting visual prompts are applied. This process does not require modifying the backbone parameters, allowing the same backbone to be reused across different domains. We evaluate VirDA on Office-31 and obtain 92.8% mean accuracy with only 1.5M trainable parameters. VirDA surpasses PDA, the state-of-the-art parameter-efficient UDA baseline, by +1.6% accuracy while using just 46% of its parameters. Compared with full-backbone fine-tuning, VirDA outperforms CDTrans and FixBi by +0.2% and +1.4%, respectively, while requiring only 1.7% and 2.8% of their trainable parameters. Relative to the strongest current methods (PMTrans and TVT), VirDA uses ~1.7% of their parameters and trades off only 2.2% and 1.1% accuracy, respectively.
>
---
#### [replaced 058] Uncovering Grounding IDs: How External Cues Shape Multimodal Binding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.24072v3](https://arxiv.org/pdf/2509.24072v3)**

> **作者:** Hosein Hasani; Amirmohammad Izadi; Fatemeh Askari; Mobin Bagherian; Sadegh Mohammadian; Mohammad Izadi; Mahdieh Soleymani Baghshah
>
> **备注:** Under review as a conference paper at ICLR 2026
>
> **摘要:** Large vision-language models (LVLMs) show strong performance across multimodal benchmarks but remain limited in structured reasoning and precise grounding. Recent work has demonstrated that adding simple visual structures, such as partitions and annotations, improves accuracy, yet the internal mechanisms underlying these gains remain unclear. We investigate this phenomenon and propose the concept of Grounding IDs, latent identifiers induced by external cues that bind objects to their designated partitions across modalities. Through representation analysis, we find that these identifiers emerge as consistent within-partition alignment in embedding space and reduce the modality gap between image and text. Causal interventions further confirm that these identifiers mediate binding between objects and symbolic cues. We show that Grounding IDs strengthen attention between related components, which in turn improves cross-modal grounding and reduces hallucinations. Taken together, our results identify Grounding IDs as a key symbolic mechanism that explains how external cues enhance multimodal binding and offer both interpretability and practical improvements.
>
---
#### [replaced 059] M3DHMR: Monocular 3D Hand Mesh Recovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.20058v2](https://arxiv.org/pdf/2505.20058v2)**

> **作者:** Yihong Lin; Xianjia Wu; Xilai Wang; Jianqiao Hu; Songju Lei; Xiandong Li; Wenxiong Kang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Monocular 3D hand mesh recovery is challenging due to high degrees of freedom of hands, 2D-to-3D ambiguity and self-occlusion. Most existing methods are either inefficient or less straightforward for predicting the position of 3D mesh vertices. Thus, we propose a new pipeline called Monocular 3D Hand Mesh Recovery (M3DHMR) to directly estimate the positions of hand mesh vertices. M3DHMR provides 2D cues for 3D tasks from a single image and uses a new spiral decoder consist of several Dynamic Spiral Convolution (DSC) Layers and a Region of Interest (ROI) Layer. On the one hand, DSC Layers adaptively adjust the weights based on the vertex positions and extract the vertex features in both spatial and channel dimensions. On the other hand, ROI Layer utilizes the physical information and refines mesh vertices in each predefined hand region separately. Extensive experiments on popular dataset FreiHAND demonstrate that M3DHMR significantly outperforms state-of-the-art real-time methods.
>
---
#### [replaced 060] iMotion-LLM: Instruction-Conditioned Trajectory Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.06211v3](https://arxiv.org/pdf/2406.06211v3)**

> **作者:** Abdulwahab Felemban; Nussair Hroub; Jian Ding; Eslam Abdelrahman; Xiaoqian Shen; Abduallah Mohamed; Mohamed Elhoseiny
>
> **摘要:** We introduce iMotion-LLM, a large language model (LLM) integrated with trajectory prediction modules for interactive motion generation. Unlike conventional approaches, it generates feasible, safety-aligned trajectories based on textual instructions, enabling adaptable and context-aware driving behavior. It combines an encoder-decoder multimodal trajectory prediction model with a pre-trained LLM fine-tuned using LoRA, projecting scene features into the LLM input space and mapping special tokens to a trajectory decoder for text-based interaction and interpretable driving. To support this framework, we introduce two datasets: 1) InstructWaymo, an extension of the Waymo Open Motion Dataset with direction-based motion instructions, and 2) Open-Vocabulary InstructNuPlan, which features safety-aligned instruction-caption pairs and corresponding safe trajectory scenarios. Our experiments validate that instruction conditioning enables trajectory generation that follows the intended condition. iMotion-LLM demonstrates strong contextual comprehension, achieving 84% average accuracy in direction feasibility detection and 96% average accuracy in safety evaluation of open-vocabulary instructions. This work lays the foundation for text-guided motion generation in autonomous driving, supporting simulated data generation, model interpretability, and robust safety alignment testing for trajectory generation models. Our code, pre-trained model, and datasets are available at: https://vision-cair.github.io/iMotion-LLM/.
>
---
#### [replaced 061] Vision-centric Token Compression in Large Language Model
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于大语言模型高效推理任务，旨在解决长上下文带来的高计算与内存开销问题。提出Vision-centric Token Compression（Vist），通过慢-快路径将低显著性上下文转为图像压缩表示，提升效率。**

- **链接: [https://arxiv.org/pdf/2502.00791v4](https://arxiv.org/pdf/2502.00791v4)**

> **作者:** Ling Xing; Alex Jinpeng Wang; Rui Yan; Xiangbo Shu; Jinhui Tang
>
> **备注:** NeurIPS 2025 spotlight
>
> **摘要:** Real-world applications are stretching context windows to hundreds of thousand of tokens while Large Language Models (LLMs) swell from billions to trillions of parameters. This dual expansion send compute and memory costs skyrocketing, making token compression indispensable. We introduce Vision Centric Token Compression (Vist), a slow-fast compression framework that mirrors human reading: the fast path renders distant tokens into images, letting a frozen, lightweight vision encoder skim the low-salience context; the slow path feeds the proximal window into the LLM for fine-grained reasoning. A Probability-Informed Visual Enhancement (PVE) objective masks high-frequency tokens during training, steering the Resampler to concentrate on semantically rich regions-just as skilled reader gloss over function words. On eleven in-context learning benchmarks, Vist achieves the same accuracy with 2.3 times fewer tokens, cutting FLOPs by 16% and memory by 50%. This method delivers remarkable results, outperforming the strongest text encoder-based compression method CEPE by 7.6% on average over benchmarks like TriviaQA, NQ, PopQA, NLUI, and CLIN, setting a new standard for token efficiency in LLMs. The project is at https://github.com/CSU-JPG/VIST.
>
---
#### [replaced 062] AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.04504v4](https://arxiv.org/pdf/2503.04504v4)**

> **作者:** Sunghyun Ahn; Youngwan Jo; Kijung Lee; Sein Kwon; Inpyo Hong; Sanghyun Park
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Video anomaly detection (VAD) is crucial for video analysis and surveillance in computer vision. However, existing VAD models rely on learned normal patterns, which makes them difficult to apply to diverse environments. Consequently, users should retrain models or develop separate AI models for new environments, which requires expertise in machine learning, high-performance hardware, and extensive data collection, limiting the practical usability of VAD. To address these challenges, this study proposes customizable video anomaly detection (C-VAD) technique and the AnyAnomaly model. C-VAD considers user-defined text as an abnormal event and detects frames containing a specified event in a video. We effectively implemented AnyAnomaly using a context-aware visual question answering without fine-tuning the large vision language model. To validate the effectiveness of the proposed model, we constructed C-VAD datasets and demonstrated the superiority of AnyAnomaly. Furthermore, our approach showed competitive results on VAD benchmarks, achieving state-of-the-art performance on UBnormal and UCF-Crime and surpassing other methods in generalization across all datasets. Our code is available online at github.com/SkiddieAhn/Paper-AnyAnomaly.
>
---
#### [replaced 063] MHB: Multimodal Handshape-aware Boundary Detection for Continuous Sign Language Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19907v2](https://arxiv.org/pdf/2511.19907v2)**

> **作者:** Mingyu Zhao; Zhanfu Yang; Yang Zhou; Zhaoyang Xia; Can Jin; Xiaoxiao He; Dimitris N. Metaxas
>
> **摘要:** This paper employs a multimodal approach for continuous sign recognition by first using ML for detecting the start and end frames of signs in videos of American Sign Language (ASL) sentences, and then by recognizing the segmented signs. For improved robustness we use 3D skeletal features extracted from sign language videos to take into account the convergence of sign properties and their dynamics that tend to cluster at sign boundaries. Another focus of this paper is the incorporation of information from 3D handshape for boundary detection. To detect handshapes normally expected at the beginning and end of signs, we pretrain a handshape classifier for detection of 87 linguistically defined canonical handshape categories using a dataset that we created by integrating and normalizing several existing datasets. A multimodal fusion module is then used to unify the pretrained sign video segmentation framework and handshape classification models. Finally, the estimated boundaries are used for sign recognition, where the recognition model is trained on a large database containing both citation-form isolated signs and signs pre-segmented (based on manual annotations) from continuous signing-as such signs often differ a bit in certain respects. We evaluate our method on the ASLLRP corpus and demonstrate significant improvements over previous work.
>
---
#### [replaced 064] A Real-Time System for Egocentric Hand-Object Interaction Detection in Industrial Domains
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.13326v2](https://arxiv.org/pdf/2507.13326v2)**

> **作者:** Antonio Finocchiaro; Alessandro Sebastiano Catinello; Michele Mazzamuto; Rosario Leonardi; Antonino Furnari; Giovanni Maria Farinella
>
> **备注:** 12 pages, 4 figures, In International Conference on Image Analysis and Processing
>
> **摘要:** Hand-object interaction detection remains an open challenge in real-time applications, where intuitive user experiences depend on fast and accurate detection of interactions with surrounding objects. We propose an efficient approach for detecting hand-objects interactions from streaming egocentric vision that operates in real time. Our approach consists of an action recognition module and an object detection module for identifying active objects upon confirmed interaction. Our Mamba model with EfficientNetV2 as backbone for action recognition achieves 38.52% p-AP on the ENIGMA-51 benchmark at 30fps, while our fine-tuned YOLOWorld reaches 85.13% AP for hand and object. We implement our models in a cascaded architecture where the action recognition and object detection modules operate sequentially. When the action recognition predicts a contact state, it activates the object detection module, which in turn performs inference on the relevant frame to detect and classify the active object.
>
---
#### [replaced 065] Image-Guided Semantic Pseudo-LiDAR Point Generation for 3D Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2409.14985v3](https://arxiv.org/pdf/2409.14985v3)**

> **作者:** Minseung Lee; Seokha Moon; Seung Joon Lee; Reza Mahjourian; Jinkyu Kim
>
> **备注:** WACV 2026
>
> **摘要:** In autonomous driving scenarios, accurate perception is becoming an even more critical task for safe navigation. While LiDAR provides precise spatial data, its inherent sparsity makes it difficult to detect small or distant objects. Existing methods try to address this by generating additional points within a Region of Interest (RoI), but relying on LiDAR alone often leads to false positives and a failure to recover meaningful structures. To address these limitations, we propose Image-Guided Semantic Pseudo-LiDAR Point Generation model, called ImagePG, a novel framework that leverages rich RGB image features to generate dense and semantically meaningful 3D points. Our framework includes an Image-Guided RoI Points Generation (IG-RPG) module, which creates pseudo-points guided by image features, and an Image-Aware Occupancy Prediction Network (I-OPN), which provides spatial priors to guide point placement. A multi-stage refinement (MR) module further enhances point quality and detection robustness. To the best of our knowledge, ImagePG is the first method to directly leverage image features for point generation. Extensive experiments on the KITTI and Waymo datasets demonstrate that ImagePG significantly improves the detection of small and distant objects like pedestrians and cyclists, reducing false positives by nearly 50%. On the KITTI benchmark, our framework improves mAP by +1.38%p (car), +7.91%p (pedestrian), and +5.21%p (cyclist) on the test set over the baseline, achieving state-of-the-art cyclist performance on the KITTI leaderboard. The code is available at: https://github.com/MS-LIMA/ImagePG
>
---
#### [replaced 066] Point-PNG: Conditional Pseudo-Negatives Generation for Point Cloud Pre-Training
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云自监督学习任务，旨在解决现有方法因不变性坍塌导致变换信息丢失的问题。作者提出Point-PNG框架，通过条件伪负样本生成和COPE网络，显式惩罚不变性坍塌，提升表示的判别性和变换敏感性。**

- **链接: [https://arxiv.org/pdf/2409.15832v3](https://arxiv.org/pdf/2409.15832v3)**

> **作者:** Sutharsan Mahendren; Saimunur Rahman; Piotr Koniusz; Tharindu Fernando; Sridha Sridharan; Clinton Fookes; Peyman Moghadam
>
> **备注:** Accepted for publication in IEEE ACCESS
>
> **摘要:** We propose Point-PNG, a novel self-supervised learning framework that generates conditional pseudo-negatives in the latent space to learn point cloud representations that are both discriminative and transformation-sensitive. Conventional self-supervised learning methods focus on achieving invariance, discarding transformation-specific information. Recent approaches incorporate transformation sensitivity by explicitly modeling relationships between original and transformed inputs. However, they often suffer from an invariant-collapse phenomenon, where the predictor degenerates into identity mappings, resulting in latent representations with limited variation across transformations. To address this, we propose Point-PNG that explicitly penalizes invariant collapse through pseudo-negatives generation, enabling the network to capture richer transformation cues while preserving discriminative representations. To this end, we introduce a parametric network, COnditional Pseudo-Negatives Embedding (COPE), which learns localized displacements induced by transformations within the latent space. A key challenge arises when jointly training COPE with the MAE, as it tends to converge to trivial identity mappings. To overcome this, we design a loss function based on pseudo-negatives conditioned on the transformation, which penalizes such trivial invariant solutions and enforces meaningful representation learning. We validate Point-PNG on shape classification and relative pose estimation tasks, showing competitive performance on ModelNet40 and ScanObjectNN under challenging evaluation protocols, and achieving superior accuracy in relative pose estimation compared to supervised baselines.
>
---
#### [replaced 067] iFinder: Structured Zero-Shot Vision-Based LLM Grounding for Dash-Cam Video Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.19552v3](https://arxiv.org/pdf/2509.19552v3)**

> **作者:** Manyi Yao; Bingbing Zhuang; Sparsh Garg; Amit Roy-Chowdhury; Christian Shelton; Manmohan Chandraker; Abhishek Aich
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Grounding large language models (LLMs) in domain-specific tasks like post-hoc dash-cam driving video analysis is challenging due to their general-purpose training and lack of structured inductive biases. As vision is often the sole modality available for such analysis (i.e., no LiDAR, GPS, etc.), existing video-based vision-language models (V-VLMs) struggle with spatial reasoning, causal inference, and explainability of events in the input video. To this end, we introduce iFinder, a structured semantic grounding framework that decouples perception from reasoning by translating dash-cam videos into a hierarchical, interpretable data structure for LLMs. iFinder operates as a modular, training-free pipeline that employs pretrained vision models to extract critical cues -- object pose, lane positions, and object trajectories -- which are hierarchically organized into frame- and video-level structures. Combined with a three-block prompting strategy, it enables step-wise, grounded reasoning for the LLM to refine a peer V-VLM's outputs and provide accurate reasoning. Evaluations on four public dash-cam video benchmarks show that iFinder's proposed grounding with domain-specific cues, especially object orientation and global context, significantly outperforms end-to-end V-VLMs on four zero-shot driving benchmarks, with up to 39% gains in accident reasoning accuracy. By grounding LLMs with driving domain-specific representations, iFinder offers a zero-shot, interpretable, and reliable alternative to end-to-end V-VLMs for post-hoc driving video understanding.
>
---
#### [replaced 068] SegAssess: Panoramic quality mapping for robust and transferable unsupervised segmentation assessment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.01183v2](https://arxiv.org/pdf/2509.01183v2)**

> **作者:** Bingnan Yang; Mi Zhang; Zhili Zhang; Zhan Zhang; Yuanxin Zhao; Xiangyun Hu; Jianya Gong
>
> **摘要:** High-quality image segmentation is fundamental to pixel-level geospatial analysis in remote sensing, necessitating robust segmentation quality assessment (SQA), particularly in unsupervised settings lacking ground truth. Although recent deep learning (DL) based unsupervised SQA methods show potential, they often suffer from coarse evaluation granularity, incomplete assessments, and poor transferability. To overcome these limitations, this paper introduces Panoramic Quality Mapping (PQM) as a new paradigm for comprehensive, pixel-wise SQA, and presents SegAssess, a novel deep learning framework realizing this approach. SegAssess distinctively formulates SQA as a fine-grained, four-class panoramic segmentation task, classifying pixels within a segmentation mask under evaluation into true positive (TP), false positive (FP), true negative (TN), and false negative (FN) categories, thereby generating a complete quality map. Leveraging an enhanced Segment Anything Model (SAM) architecture, SegAssess uniquely employs the input mask as a prompt for effective feature integration via cross-attention. Key innovations include an Edge Guided Compaction (EGC) branch with an Aggregated Semantic Filter (ASF) module to refine predictions near challenging object edges, and an Augmented Mixup Sampling (AMS) training strategy integrating multi-source masks to significantly boost cross-domain robustness and zero-shot transferability. Comprehensive experiments demonstrate that SegAssess achieves state-of-the-art (SOTA) performance and exhibits remarkable zero-shot transferability to unseen masks. The code is available at https://github.com/Yangbn97/SegAssess.
>
---
