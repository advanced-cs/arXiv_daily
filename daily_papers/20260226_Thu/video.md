# 计算机视觉 cs.CV

- **最新发布 123 篇**

- **更新 69 篇**

## 最新发布

#### [new 001] AHAN: Asymmetric Hierarchical Attention Network for Identical Twin Face Verification
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，旨在解决同卵双胞胎识别难题。通过提出AHAN网络，利用多粒度注意力机制提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2602.21503v1](https://arxiv.org/pdf/2602.21503v1)**

> **作者:** Hoang-Nhat Nguyen
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Identical twin face verification represents an extreme fine-grained recognition challenge where even state-of-the-art systems fail due to overwhelming genetic similarity. Current face recognition methods achieve over 99.8% accuracy on standard benchmarks but drop dramatically to 88.9% when distinguishing identical twins, exposing critical vulnerabilities in biometric security systems. The difficulty lies in learning features that capture subtle, non-genetic variations that uniquely identify individuals. We propose the Asymmetric Hierarchical Attention Network (AHAN), a novel architecture specifically designed for this challenge through multi-granularity facial analysis. AHAN introduces a Hierarchical Cross-Attention (HCA) module that performs multi-scale analysis on semantic facial regions, enabling specialized processing at optimal resolutions. We further propose a Facial Asymmetry Attention Module (FAAM) that learns unique biometric signatures by computing cross-attention between left and right facial halves, capturing subtle asymmetric patterns that differ even between twins. To ensure the network learns truly individuating features, we introduce Twin-Aware Pair-Wise Cross-Attention (TA-PWCA), a training-only regularization strategy that uses each subject's own twin as the hardest possible distractor. Extensive experiments on the ND_TWIN dataset demonstrate that AHAN achieves 92.3% twin verification accuracy, representing a 3.4% improvement over state-of-the-art methods.
>
---
#### [new 002] CoLoGen: Progressive Learning of Concept`-`Localization Duality for Unified Image Generation
- **分类: cs.CV**

- **简介: 该论文提出CoLoGen，解决统一图像生成中概念与定位表示冲突的问题。通过渐进学习和PRW模块，实现概念与定位的协同，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2602.22150v1](https://arxiv.org/pdf/2602.22150v1)**

> **作者:** YuXin Song; Yu Lu; Haoyuan Sun; Huanjin Yao; Fanglong Liu; Yifan Sun; Haocheng Feng; Hang Zhou; Jingdong Wang
>
> **备注:** Accepted by CVPR2026. 15 pages, 8 figures
>
> **摘要:** Unified conditional image generation remains difficult because different tasks depend on fundamentally different internal representations. Some require conceptual understanding for semantic synthesis, while others rely on localization cues for spatial precision. Forcing these heterogeneous tasks to share a single representation leads to concept`-`localization representational conflict. To address this issue, we propose CoLoGen, a unified diffusion framework that progressively learns and reconciles this concept`-`localization duality. CoLoGen uses a staged curriculum that first builds core conceptual and localization abilities, then adapts them to diverse visual conditions, and finally refines their synergy for complex instruction`-`driven tasks. Central to this process is the Progressive Representation Weaving (PRW) module, which dynamically routes features to specialized experts and stably integrates their outputs across stages. Experiments on editing, controllable generation, and customized generation show that CoLoGen achieves competitive or superior performance, offering a principled representational perspective for unified image generation.
>
---
#### [new 003] Overview of the CXR-LT 2026 Challenge: Multi-Center Long-Tailed and Zero Shot Chest X-ray Classification
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决胸片中罕见疾病和未知病种的识别问题。通过构建多中心数据集，提出两个核心任务并评估模型性能。**

- **链接: [https://arxiv.org/pdf/2602.22092v1](https://arxiv.org/pdf/2602.22092v1)**

> **作者:** Hexin Dong; Yi Lin; Pengyu Zhou; Fengnian Zhao; Alan Clint Legasto; Mingquan Lin; Hao Chen; Yuzhe Yang; George Shih; Yifan Peng
>
> **摘要:** Chest X-ray (CXR) interpretation is hindered by the long-tailed distribution of pathologies and the open-world nature of clinical environments. Existing benchmarks often rely on closed-set classes from single institutions, failing to capture the prevalence of rare diseases or the appearance of novel findings. To address this, we present the CXR-LT 2026 challenge. This third iteration of the benchmark introduces a multi-center dataset comprising over 145,000 images from PadChest and NIH Chest X-ray datasets. The challenge defines two core tasks: (1) Robust Multi-Label Classification on 30 known classes and (2) Open-World Generalization to 6 unseen (out-of-distribution) rare disease classes. We report the results of the top-performing teams, evaluating them via mean Average Precision (mAP), AUROC, and F1-score. The winning solutions achieved an mAP of 0.5854 on Task 1 and 0.4315 on Task 2, demonstrating that large-scale vision-language pre-training significantly mitigates the performance drop typically associated with zero-shot diagnosis.
>
---
#### [new 004] RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations
- **分类: cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决视觉退化下VisRAG模型性能下降的问题。提出RobustVisRAG框架，通过因果路径分离语义与退化因素，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.22013v1](https://arxiv.org/pdf/2602.22013v1)**

> **作者:** I-Hsiang Chen; Yu-Wei Liu; Tse-Yu Wu; Yu-Chien Chiang; Jen-Chien Yang; Wei-Ting Chen
>
> **备注:** Accepted by CVPR2026; Project Page: https://robustvisrag.github.io
>
> **摘要:** Vision-based Retrieval-Augmented Generation (VisRAG) leverages vision-language models (VLMs) to jointly retrieve relevant visual documents and generate grounded answers based on multimodal evidence. However, existing VisRAG models degrade in performance when visual inputs suffer from distortions such as blur, noise, low light, or shadow, where semantic and degradation factors become entangled within pretrained visual encoders, leading to errors in both retrieval and generation stages. To address this limitation, we introduce RobustVisRAG, a causality-guided dual-path framework that improves VisRAG robustness while preserving efficiency and zero-shot generalization. RobustVisRAG uses a non-causal path to capture degradation signals through unidirectional attention and a causal path to learn purified semantics guided by these signals. Together with the proposed Non-Causal Distortion Modeling and Causal Semantic Alignment objectives, the framework enforces a clear separation between semantics and degradations, enabling stable retrieval and generation under challenging visual conditions. To evaluate robustness under realistic conditions, we introduce the Distortion-VisRAG dataset, a large-scale benchmark containing both synthetic and real-world degraded documents across seven domains, with 12 synthetic and 5 real distortion types that comprehensively reflect practical visual degradations. Experimental results show that RobustVisRAG improves retrieval, generation, and end-to-end performance by 7.35%, 6.35%, and 12.40%, respectively, on real-world degradations, while maintaining comparable accuracy on clean inputs.
>
---
#### [new 005] Directed Ordinal Diffusion Regularization for Progression-Aware Diabetic Retinopathy Grading
- **分类: cs.CV**

- **简介: 该论文属于糖尿病视网膜病变分级任务，旨在解决现有方法忽略疾病进展方向性的问题。通过构建有向图并引入扩散正则化，确保特征表示符合疾病自然发展路径。**

- **链接: [https://arxiv.org/pdf/2602.21942v1](https://arxiv.org/pdf/2602.21942v1)**

> **作者:** Huangwei Chen; Junhao Jia; Ruocheng Li; Cunyuan Yang; Wu Li; Xiaotao Pang; Yifei Chen; Haishuai Wang; Jiajun Bu; Lei Wu
>
> **备注:** 3 figures
>
> **摘要:** Diabetic Retinopathy (DR) progresses as a continuous and irreversible deterioration of the retina, following a well-defined clinical trajectory from mild to severe stages. However, most existing ordinal regression approaches model DR severity as a set of static, symmetric ranks, capturing relative order while ignoring the inherent unidirectional nature of disease progression. As a result, the learned feature representations may violate biological plausibility, allowing implausible proximity between non-consecutive stages or even reverse transitions. To bridge this gap, we propose Directed Ordinal Diffusion Regularization (D-ODR), which explicitly models the feature space as a directed flow by constructing a progression-constrained directed graph that strictly enforces forward disease evolution. By performing multi-scale diffusion on this directed structure, D-ODR imposes penalties on score inversions along valid progression paths, thereby effectively preventing the model from learning biologically inconsistent reverse transitions. This mechanism aligns the feature representation with the natural trajectory of DR worsening. Extensive experiments demonstrate that D-ODR yields superior grading performance compared to state-of-the-art ordinal regression and DR-specific grading methods, offering a more clinically reliable assessment of disease severity. Our code is available on https://github.com/HovChen/D-ODR.
>
---
#### [new 006] PanoEnv: Exploring 3D Spatial Intelligence in Panoramic Environments with Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决VLM在3D空间推理上的不足。通过构建PanoEnv基准和RL训练框架，提升模型对全景图像的3D理解能力。**

- **链接: [https://arxiv.org/pdf/2602.21992v1](https://arxiv.org/pdf/2602.21992v1)**

> **作者:** Zekai Lin; Xu Zheng
>
> **摘要:** 360 panoramic images are increasingly used in virtual reality, autonomous driving, and robotics for holistic scene understanding. However, current Vision-Language Models (VLMs) struggle with 3D spatial reasoning on Equirectangular Projection (ERP) images due to geometric distortion and limited 3D supervision. We introduce PanoEnv, a large-scale VQA benchmark built from synthetic 3D environments, containing 14.8K questions across five categories (e.g., relative position, volume comparison) grounded in accurate 3D annotations including depth, segmentation, and bounding boxes. Benchmarking 14 state-of-the-art VLMs reveals limited 3D understanding, achieving only 49.34% overall accuracy and 8.36% on open-ended (OE) questions. To enhance 3D reasoning, we propose a reinforcement learning post-training framework based on Group Relative Policy Optimization (GRPO) with a ground-truth-guided reward that incorporates five geometry-aware strategies such as distance tolerance and spatial consistency. A two-stage curriculum further mitigates catastrophic forgetting: Stage 1 trains on structured tasks (true/false and multiple choice), and Stage 2 fine-tunes on mixed open-ended data to improve generalization. Our 7B model achieves new state-of-the-art performance, improving overall accuracy to 52.93% (+3.59%) and open-ended accuracy to 14.83% while maintaining structured-task performance. It also achieves top semantic evaluation scores (Q-Score 6.24, P-Score 5.95), surpassing 32B models. These results demonstrate that PanoEnv-QA and our curriculum-based RL framework effectively instill 3D spatial intelligence in VLMs for omnidirectional perception.
>
---
#### [new 007] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments
- **分类: cs.CV**

- **简介: 该论文提出Olbedo数据集，解决户外场景的固有图像分解问题。通过收集多视角、多光照条件的无人机图像及详细标注，支持真实户外图像的材质与阴影分离，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.22025v1](https://arxiv.org/pdf/2602.22025v1)**

> **作者:** Shuang Song; Debao Huang; Deyan Deng; Haolin Xiong; Yang Tang; Yajie Zhao; Rongjun Qin
>
> **备注:** CVPR 2026
>
> **摘要:** Intrinsic image decomposition (IID) of outdoor scenes is crucial for relighting, editing, and understanding large-scale environments, but progress has been limited by the lack of real-world datasets with reliable albedo and shading supervision. We introduce Olbedo, a large-scale aerial dataset for outdoor albedo--shading decomposition in the wild. Olbedo contains 5,664 UAV images captured across four landscape types, multiple years, and diverse illumination conditions. Each view is accompanied by multi-view consistent albedo and shading maps, metric depth, surface normals, sun and sky shading components, camera poses, and, for recent flights, measured HDR sky domes. These annotations are derived from an inverse-rendering refinement pipeline over multi-view stereo reconstructions and calibrated sky illumination, together with per-pixel confidence masks. We demonstrate that Olbedo enables state-of-the-art diffusion-based IID models, originally trained on synthetic indoor data, to generalize to real outdoor imagery: fine-tuning on Olbedo significantly improves single-view outdoor albedo prediction on the MatrixCity benchmark. We further illustrate applications of Olbedo-trained models to multi-view consistent relighting of 3D assets, material editing, and scene change analysis for urban digital twins. We release the dataset, baseline models, and an evaluation protocol to support future research in outdoor intrinsic decomposition and illumination-aware aerial vision.
>
---
#### [new 008] SemVideo: Reconstructs What You Watch from Brain Activity via Hierarchical Semantic Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于fMRI到视频的重建任务，旨在解决视频重建中视觉表示不一致和时间不连贯的问题。提出SemVideo框架，利用层次语义引导提升重建效果。**

- **链接: [https://arxiv.org/pdf/2602.21819v1](https://arxiv.org/pdf/2602.21819v1)**

> **作者:** Minghan Yang; Lan Yang; Ke Li; Honggang Zhang; Kaiyue Pang; Yizhe Song
>
> **摘要:** Reconstructing dynamic visual experiences from brain activity provides a compelling avenue for exploring the neural mechanisms of human visual perception. While recent progress in fMRI-based image reconstruction has been notable, extending this success to video reconstruction remains a significant challenge. Current fMRI-to-video reconstruction approaches consistently encounter two major shortcomings: (i) inconsistent visual representations of salient objects across frames, leading to appearance mismatches; (ii) poor temporal coherence, resulting in motion misalignment or abrupt frame transitions. To address these limitations, we introduce SemVideo, a novel fMRI-to-video reconstruction framework guided by hierarchical semantic information. At the core of SemVideo is SemMiner, a hierarchical guidance module that constructs three levels of semantic cues from the original video stimulus: static anchor descriptions, motion-oriented narratives, and holistic summaries. Leveraging this semantic guidance, SemVideo comprises three key components: a Semantic Alignment Decoder that aligns fMRI signals with CLIP-style embeddings derived from SemMiner, a Motion Adaptation Decoder that reconstructs dynamic motion patterns using a novel tripartite attention fusion architecture, and a Conditional Video Render that leverages hierarchical semantic guidance for video reconstruction. Experiments conducted on the CC2017 and HCP datasets demonstrate that SemVideo achieves superior performance in both semantic alignment and temporal consistency, setting a new state-of-the-art in fMRI-to-video reconstruction.
>
---
#### [new 009] Exploring Vision-Language Models for Open-Vocabulary Zero-Shot Action Segmentation
- **分类: cs.CV**

- **简介: 该论文研究开放词汇零样本时序动作分割任务，解决传统方法受限于封闭词表的问题。通过引入无需训练的框架，利用视觉语言模型实现无监督动作分割。**

- **链接: [https://arxiv.org/pdf/2602.21406v1](https://arxiv.org/pdf/2602.21406v1)**

> **作者:** Asim Unmesh; Kaki Ramesh; Mayank Patel; Rahul Jain; Karthik Ramani
>
> **备注:** ICRA 2026
>
> **摘要:** Temporal Action Segmentation (TAS) requires dividing videos into action segments, yet the vast space of activities and alternative breakdowns makes collecting comprehensive datasets infeasible. Existing methods remain limited to closed vocabularies and fixed label sets. In this work, we explore the largely unexplored problem of Open-Vocabulary Zero-Shot Temporal Action Segmentation (OVTAS) by leveraging the strong zero-shot capabilities of Vision-Language Models (VLMs). We introduce a training-free pipeline that follows a segmentation-by-classification design: Frame-Action Embedding Similarity (FAES) matches video frames to candidate action labels, and Similarity-Matrix Temporal Segmentation (SMTS) enforces temporal consistency. Beyond proposing OVTAS, we present a systematic study across 14 diverse VLMs, providing the first broad analysis of their suitability for open-vocabulary action segmentation. Experiments on standard benchmarks show that OVTAS achieves strong results without task-specific supervision, underscoring the potential of VLMs for structured temporal understanding.
>
---
#### [new 010] SigVLP: Sigmoid Volume-Language Pre-Training for Self-Supervised CT-Volume Adaptive Representation Learning
- **分类: cs.CV**

- **简介: 该论文提出SigVLP模型，解决CT影像因尺寸不一致导致的信息损失问题，通过旋转位置编码实现自监督学习。**

- **链接: [https://arxiv.org/pdf/2602.21735v1](https://arxiv.org/pdf/2602.21735v1)**

> **作者:** Jiayi Wang; Hadrien Reynaud; Ibrahim Ethem Hamamci; Sezgin Er; Suprosanna Shit; Bjoern Menze; Bernhard Kainz
>
> **摘要:** Large-scale, volumetric medical imaging datasets typically aggregate scans from different vendors and devices, resulting in highly variable resolution, slice thicknesses, and numbers of slices per study. Consequently, training representation models usually requires cropping or interpolating along the z-axis to obtain fixed-size blocks, which inevitably causes information loss. We propose a new training approach to overcome this limitation. Instead of absolute position embeddings, we interpret volumes as sequences of 3D chunks and adopt Rotary Position Embeddings, allowing us to treat the z-axis as an unconstrained temporal dimensions. Building on this idea, we introduce a new vision-language model: SigVLP. In SigVLP, we implement Rotary Position Embedding as the positional encoding method, which is applied directly within the attention operation, generating input-conditioned sine and cosine weights on the fly. This design ensures consistent alignment between query and key projections and adapts to any input sizes. To allow for variable input size during training, we sample Computed Tomography volumes in chunks and pair them with localized organ-wise textual observations. Compared to using entire reports for conditioning, chunkwise alignment provides finer-grained supervision, enabling the model to establish stronger correlations between the text and volume representations, thereby improving the precision of text-to-volume alignment. Our models are trained with the Muon optimizer and evaluated on a diverse set of downstream tasks, including zero-shot abnormality and organ classification, segmentation, and retrieval tasks.
>
---
#### [new 011] Protein Graph Neural Networks for Heterogeneous Cryo-EM Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于冷冻电镜重构任务，旨在解决异质单粒子重构问题。通过图神经网络结合蛋白质结构先验，预测原子骨架构象，提升重构精度。**

- **链接: [https://arxiv.org/pdf/2602.21915v1](https://arxiv.org/pdf/2602.21915v1)**

> **作者:** Jonathan Krook; Axel Janson; Joakim andén; Melanie Weber; Ozan Öktem
>
> **摘要:** We present a geometry-aware method for heterogeneous single-particle cryogenic electron microscopy (cryo-EM) reconstruction that predicts atomic backbone conformations. To incorporate protein-structure priors, we represent the backbone as a graph and use a graph neural network (GNN) autodecoder that maps per-image latent variables to 3D displacements of a template conformation. The objective combines a data-discrepancy term based on a differentiable cryo-EM forward model with geometric regularization, and it supports unknown orientations via ellipsoidal support lifting (ESL) pose estimation. On synthetic datasets derived from molecular dynamics trajectories, the proposed GNN achieves higher accuracy compared to a multilayer perceptron (MLP) of comparable size, highlighting the benefits of a geometry-informed inductive bias.
>
---
#### [new 012] Virtual Biopsy for Intracranial Tumors Diagnosis on MRI
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像分析任务，旨在解决颅内肿瘤诊断中侵入性活检的风险与局限性。通过构建ICT-MRI数据集并提出Virtual Biopsy框架，实现非侵入性MRI病理预测。**

- **链接: [https://arxiv.org/pdf/2602.21613v1](https://arxiv.org/pdf/2602.21613v1)**

> **作者:** Xinzhe Luo; Shuai Shao; Yan Wang; Jiangtao Wang; Yutong Bai; Jianguo Zhang
>
> **摘要:** Deep intracranial tumors situated in eloquent brain regions controlling vital functions present critical diagnostic challenges. Clinical practice has shifted toward stereotactic biopsy for pathological confirmation before treatment. Yet biopsy carries inherent risks of hemorrhage and neurological deficits and struggles with sampling bias due to tumor spatial heterogeneity, because pathological changes are typically region-selective rather than tumor-wide. Therefore, advancing non-invasive MRI-based pathology prediction is essential for holistic tumor assessment and modern clinical decision-making. The primary challenge lies in data scarcity: low tumor incidence requires long collection cycles, and annotation demands biopsy-verified pathology from neurosurgical experts. Additionally, tiny lesion volumes lacking segmentation masks cause critical features to be overwhelmed by background noise. To address these challenges, we construct the ICT-MRI dataset - the first public biopsy-verified benchmark with 249 cases across four categories. We propose a Virtual Biopsy framework comprising: MRI-Processor for standardization; Tumor-Localizer employing vision-language models for coarse-to-fine localization via weak supervision; and Adaptive-Diagnoser with a Masked Channel Attention mechanism fusing local discriminative features with global contexts. Experiments demonstrate over 90% accuracy, outperforming baselines by more than 20%.
>
---
#### [new 013] Axial-Centric Cross-Plane Attention for 3D Medical Image Classification
- **分类: cs.CV**

- **简介: 该论文属于3D医学图像分类任务，旨在解决现有方法未能反映临床中以轴向平面为主的工作流程的问题。提出一种轴向中心的跨平面注意力架构，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2602.21636v1](https://arxiv.org/pdf/2602.21636v1)**

> **作者:** Doyoung Park; Jinsoo Kim; Lohendran Baskaran
>
> **备注:** Submitted to MICCAI 2026
>
> **摘要:** Clinicians commonly interpret three-dimensional (3D) medical images, such as computed tomography (CT) scans, using multiple anatomical planes rather than as a single volumetric representation. In this multi-planar approach, the axial plane typically serves as the primary acquisition and diagnostic reference, while the coronal and sagittal planes provide complementary spatial information to increase diagnostic confidence. However, many existing 3D deep learning methods either process volumetric data holistically or assign equal importance to all planes, failing to reflect the axial-centric clinical interpretation workflow. To address this gap, we propose an axial-centric cross-plane attention architecture for 3D medical image classification that captures the inherent asymmetric dependencies between different anatomical planes. Our architecture incorporates MedDINOv3, a medical vision foundation model pretrained via self-supervised learning on large-scale axial CT images, as a frozen feature extractor for the axial, coronal, and sagittal planes. RICA blocks and intra-plane transformer encoders capture plane-specific positional and contextual information within each anatomical plane, while axial-centric cross-plane transformer encoders condition axial features on complementary information from auxiliary planes. Experimental results on six datasets from the MedMNIST3D benchmark demonstrate that the proposed architecture consistently outperforms existing 3D and multi-plane models in terms of accuracy and AUC. Ablation studies further confirm the importance of axial-centric query-key-value allocation and directional cross-plane fusion. These results highlight the importance of aligning architectural design with clinical interpretation workflows for robust and data-efficient 3D medical image analysis.
>
---
#### [new 014] Mobile-Ready Automated Triage of Diabetic Retinopathy Using Digital Fundus Images
- **分类: cs.CV**

- **简介: 该论文属于糖尿病视网膜病变自动分诊任务，旨在解决人工诊断效率低、易出错的问题。通过轻量级深度学习框架，实现对眼底图像的高效准确评估。**

- **链接: [https://arxiv.org/pdf/2602.21943v1](https://arxiv.org/pdf/2602.21943v1)**

> **作者:** Aadi Joshi; Manav S. Sharma; Vijay Uttam Rathod; Ashlesha Sawant; Prajakta Musale; Asmita B. Kalamkar
>
> **备注:** Presented at ICCI 2025. 11 pages, 2 figures. MobileNetV3 + CORAL-based lightweight model for diabetic retinopathy severity classification with mobile deployment
>
> **摘要:** Diabetic Retinopathy (DR) is a major cause of vision impairment worldwide. However, manual diagnosis is often time-consuming and prone to errors, leading to delays in screening. This paper presents a lightweight automated deep learning framework for efficient assessment of DR severity from digital fundus images. We use a MobileNetV3 architecture with a Consistent Rank Logits (CORAL) head to model the ordered progression of disease while maintaining computational efficiency for resource-constrained environments. The model is trained and validated on a combined dataset of APTOS 2019 and IDRiD images using a preprocessing pipeline including circular cropping and illumination normalization. Extensive experiments including 3-fold cross-validation and ablation studies demonstrate strong performance. The model achieves a Quadratic Weighted Kappa (QWK) score of 0.9019 and an accuracy of 80.03 percent. Additionally, we address real-world deployment challenges through model calibration to reduce overconfidence and optimization for mobile devices. The proposed system provides a scalable and practical tool for early-stage diabetic retinopathy screening.
>
---
#### [new 015] RGB-Event HyperGraph Prompt for Kilometer Marker Recognition based on Pre-trained Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对地铁定位中的里程标识别任务，解决在复杂环境下传统RGB相机感知不足的问题，通过融合事件相机数据提升识别性能。**

- **链接: [https://arxiv.org/pdf/2602.22026v1](https://arxiv.org/pdf/2602.22026v1)**

> **作者:** Xiaoyu Xian; Shiao Wang; Xiao Wang; Daxin Tian; Yan Tian
>
> **备注:** Accepted by IEEE Transactions on Cognitive and Developmental Systems (IEEE TCDS) 2026
>
> **摘要:** Metro trains often operate in highly complex environments, characterized by illumination variations, high-speed motion, and adverse weather conditions. These factors pose significant challenges for visual perception systems, especially those relying solely on conventional RGB cameras. To tackle these difficulties, we explore the integration of event cameras into the perception system, leveraging their advantages in low-light conditions, high-speed scenarios, and low power consumption. Specifically, we focus on Kilometer Marker Recognition (KMR), a critical task for autonomous metro localization under GNSS-denied conditions. In this context, we propose a robust baseline method based on a pre-trained RGB OCR foundation model, enhanced through multi-modal adaptation. Furthermore, we construct the first large-scale RGB-Event dataset, EvMetro5K, containing 5,599 pairs of synchronized RGB-Event samples, split into 4,479 training and 1,120 testing samples. Extensive experiments on EvMetro5K and other widely used benchmarks demonstrate the effectiveness of our approach for KMR. Both the dataset and source code will be released on https://github.com/Event-AHU/EvMetro5K_benchmark
>
---
#### [new 016] RT-RMOT: A Dataset and Framework for RGB-Thermal Referring Multi-Object Tracking
- **分类: cs.CV**

- **简介: 该论文提出RT-RMOT任务，解决低可见性下的多目标跟踪问题。构建了RefRT数据集，并提出RTrack框架及优化策略，提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2602.22033v1](https://arxiv.org/pdf/2602.22033v1)**

> **作者:** Yanqiu Yu; Zhifan Jin; Sijia Chen; Tongfei Chu; En Yu; Liman Liu; Wenbing Tao
>
> **摘要:** Referring Multi-Object Tracking has attracted increasing attention due to its human-friendly interactive characteristics, yet it exhibits limitations in low-visibility conditions, such as nighttime, smoke, and other challenging scenarios. To overcome this limitation, we propose a new RGB-Thermal RMOT task, named RT-RMOT, which aims to fuse RGB appearance features with the illumination robustness of the thermal modality to enable all-day referring multi-object tracking. To promote research on RT-RMOT, we construct the first Referring Multi-Object Tracking dataset under RGB-Thermal modality, named RefRT. It contains 388 language descriptions, 1,250 tracked targets, and 166,147 Language-RGB-Thermal (L-RGB-T) triplets. Furthermore, we propose RTrack, a framework built upon a multimodal large language model (MLLM) that integrates RGB, thermal, and textual features. Since the initial framework still leaves room for improvement, we introduce a Group Sequence Policy Optimization (GSPO) strategy to further exploit the model's potential. To alleviate training instability during RL fine-tuning, we introduce a Clipped Advantage Scaling (CAS) strategy to suppress gradient explosion. In addition, we design Structured Output Reward and Comprehensive Detection Reward to balance exploration and exploitation, thereby improving the completeness and accuracy of target perception. Extensive experiments on the RefRT dataset demonstrate the effectiveness of the proposed RTrack framework.
>
---
#### [new 017] E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought
- **分类: cs.CV**

- **简介: 该论文属于电商海报质量评估任务，解决现有方法无法准确评估中文电商海报的问题。构建了E-comIQ-ZH数据集和模型，实现更贴近人类判断的自动化评估。**

- **链接: [https://arxiv.org/pdf/2602.21698v1](https://arxiv.org/pdf/2602.21698v1)**

> **作者:** Meiqi Sun; Mingyu Li; Junxiong Zhu
>
> **备注:** 21pages, 19figures, accepted by CVPR 2026
>
> **摘要:** Generative AI is widely used to create commercial posters. However, rapid advances in generation have outpaced automated quality assessment. Existing models emphasize generic esthetics or low level distortions and lack the functional criteria required for e-commerce design. It is especially challenging for Chinese content, where complex characters often produce subtle but critical textual artifacts that are overlooked by existing methods. To address this, we introduce E-comIQ-ZH, a framework for evaluating Chinese e-commerce posters. We build the first dataset E-comIQ-18k to feature multi dimensional scores and expert calibrated Chain of Thought (CoT) rationales. Using this dataset, we train E-comIQ-M, a specialized evaluation model that aligns with human expert judgment. Our framework enables E-comIQ-Bench, the first automated and scalable benchmark for the generation of Chinese e-commerce posters. Extensive experiments show our E-comIQ-M aligns more closely with expert standards and enables scalable automated assessment of e-commerce posters. All datasets, models, and evaluation tools will be released to support future research in this area.Code will be available at https://github.com/4mm7/E-comIQ-ZH.
>
---
#### [new 018] IHF-Harmony: Multi-Modality Magnetic Resonance Images Harmonization using Invertible Hierarchy Flow Model
- **分类: cs.CV**

- **简介: 该论文属于多模态MRI图像谐波化任务，解决跨模态可扩展性和依赖配对数据的问题。提出IHF-Harmony框架，通过可逆层次流实现无损重建和准确特征迁移。**

- **链接: [https://arxiv.org/pdf/2602.21536v1](https://arxiv.org/pdf/2602.21536v1)**

> **作者:** Pengli Zhu; Yitao Zhu; Haowen Pang; Anqi Qiu
>
> **摘要:** Retrospective MRI harmonization is limited by poor scalability across modalities and reliance on traveling subject datasets. To address these challenges, we introduce IHF-Harmony, a unified invertible hierarchy flow framework for multi-modality harmonization using unpaired data. By decomposing the translation process into reversible feature transformations, IHF-Harmony guarantees bijective mapping and lossless reconstruction to prevent anatomical distortion. Specifically, an invertible hierarchy flow (IHF) performs hierarchical subtractive coupling to progressively remove artefact-related features, while an artefact-aware normalization (AAN) employs anatomy-fixed feature modulation to accurately transfer target characteristics. Combined with anatomy and artefact consistency loss objectives, IHF-Harmony achieves high-fidelity harmonization that retains source anatomy. Experiments across multiple MRI modalities demonstrate that IHF-Harmony outperforms existing methods in both anatomical fidelity and downstream task performance, facilitating robust harmonization for large-scale multi-site imaging studies. Code will be released upon acceptance.
>
---
#### [new 019] Scan Clusters, Not Pixels: A Cluster-Centric Paradigm for Efficient Ultra-high-definition Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，解决UHD图像处理计算成本高的问题。通过聚类中心替代像素扫描，提升效率并取得最佳效果。**

- **链接: [https://arxiv.org/pdf/2602.21917v1](https://arxiv.org/pdf/2602.21917v1)**

> **作者:** Chen Wu; Ling Wang; Zhuoran Zheng; Yuning Cui; Zhixiong Yang; Xiangyu Chen; Yue Zhang; Weidong Jiang; Jingyuan Xia
>
> **备注:** Aceepted by CVPR26
>
> **摘要:** Ultra-High-Definition (UHD) image restoration is trapped in a scalability crisis: existing models, bound to pixel-wise operations, demand unsustainable computation. While state space models (SSMs) like Mamba promise linear complexity, their pixel-serial scanning remains a fundamental bottleneck for the millions of pixels in UHD content. We ask: must we process every pixel to understand the image? This paper introduces C$^2$SSM, a visual state space model that breaks this taboo by shifting from pixel-serial to cluster-serial scanning. Our core discovery is that the rich feature distribution of a UHD image can be distilled into a sparse set of semantic centroids via a neural-parameterized mixture model. C$^2$SSM leverages this to reformulate global modeling into a novel dual-path process: it scans and reasons over a handful of cluster centers, then diffuses the global context back to all pixels through a principled similarity distribution, all while a lightweight modulator preserves fine details. This cluster-centric paradigm achieves a decisive leap in efficiency, slashing computational costs while establishing new state-of-the-art results across five UHD restoration tasks. More than a solution, C$^2$SSM charts a new course for efficient large-scale vision: scan clusters, not pixels.
>
---
#### [new 020] SF3D-RGB: Scene Flow Estimation from Monocular Camera and Sparse LiDAR
- **分类: cs.CV**

- **简介: 该论文属于场景流估计任务，旨在通过融合单目图像和稀疏LiDAR点云提升场景流的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.21699v1](https://arxiv.org/pdf/2602.21699v1)**

> **作者:** Rajai Alhimdiat; Ramy Battrawy; René Schuster; Didier Stricker; Wesam Ashour
>
> **备注:** Accepted in Computer Vision Conference (CVC) 2026
>
> **摘要:** Scene flow estimation is an extremely important task in computer vision to support the perception of dynamic changes in the scene. For robust scene flow, learning-based approaches have recently achieved impressive results using either image-based or LiDAR-based modalities. However, these methods have tended to focus on the use of a single modality. To tackle these problems, we present a deep learning architecture, SF3D-RGB, that enables sparse scene flow estimation using 2D monocular images and 3D point clouds (e.g., acquired by LiDAR) as inputs. Our architecture is an end-to-end model that first encodes information from each modality into features and fuses them together. Then, the fused features enhance a graph matching module for better and more robust mapping matrix computation to generate an initial scene flow. Finally, a residual scene flow module further refines the initial scene flow. Our model is designed to strike a balance between accuracy and efficiency. Furthermore, experiments show that our proposed method outperforms single-modality methods and achieves better scene flow accuracy on real-world datasets while using fewer parameters compared to other state-of-the-art methods with fusion.
>
---
#### [new 021] Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps
- **分类: cs.CV**

- **简介: 该论文提出LGI地图，解决阴影生成与再照明任务中的物理一致性问题，通过结合几何与光照信息提升生成模型的 realism 和 consistency。**

- **链接: [https://arxiv.org/pdf/2602.21820v1](https://arxiv.org/pdf/2602.21820v1)**

> **作者:** Shan Wang; Peixia Li; Chenchen Xu; Ziang Cheng; Jiayu Yang; Hongdong Li; Pulak Purkait
>
> **备注:** ICRL 2026
>
> **摘要:** We propose Light-Geometry Interaction (LGI) maps, a novel representation that encodes light-aware occlusion from monocular depth. Unlike ray tracing, which requires full 3D reconstruction, LGI captures essential light-shadow interactions reliably and accurately, computed from off-the-shelf 2.5D depth map predictions. LGI explicitly ties illumination direction to geometry, providing a physics-inspired prior that constrains generative models. Without such prior, these models often produce floating shadows, inconsistent illumination, and implausible shadow geometry. Building on this representation, we propose a unified pipeline for joint shadow generation and relighting - unlike prior methods that treat them as disjoint tasks - capturing the intrinsic coupling of illumination and shadowing essential for modeling indirect effects. By embedding LGI into a bridge-matching generative backbone, we reduce ambiguity and enforce physically consistent light-shadow reasoning. To enable effective training, we curated the first large-scale benchmark dataset for joint shadow and relighting, covering reflections, transparency, and complex interreflections. Experiments show significant gains in realism and consistency across synthetic and real images. LGI thus bridges geometry-inspired rendering with generative modeling, enabling efficient, physically consistent shadow generation and relighting.
>
---
#### [new 022] Solaris: Building a Multiplayer Video World Model in Minecraft
- **分类: cs.CV**

- **简介: 该论文提出Solaris，解决多智能体视频生成问题，通过构建多人游戏数据系统，实现多视角一致观测与训练，提升模型在多人环境中的表现。**

- **链接: [https://arxiv.org/pdf/2602.22208v1](https://arxiv.org/pdf/2602.22208v1)**

> **作者:** Georgy Savva; Oscar Michel; Daohan Lu; Suppakit Waiwitlikhit; Timothy Meehan; Dhairya Mishra; Srivats Poddar; Jack Lu; Saining Xie
>
> **备注:** Project website: https://solaris-wm.github.io/
>
> **摘要:** Existing action-conditioned video generation models (video world models) are limited to single-agent perspectives, failing to capture the multi-agent interactions of real-world environments. We introduce Solaris, a multiplayer video world model that simulates consistent multi-view observations. To enable this, we develop a multiplayer data system designed for robust, continuous, and automated data collection on video games such as Minecraft. Unlike prior platforms built for single-player settings, our system supports coordinated multi-agent interaction and synchronized videos + actions capture. Using this system, we collect 12.64 million multiplayer frames and propose an evaluation framework for multiplayer movement, memory, grounding, building, and view consistency. We train Solaris using a staged pipeline that progressively transitions from single-player to multiplayer modeling, combining bidirectional, causal, and Self Forcing training. In the final stage, we introduce Checkpointed Self Forcing, a memory-efficient Self Forcing variant that enables a longer-horizon teacher. Results show our architecture and training design outperform existing baselines. Through open-sourcing our system and models, we hope to lay the groundwork for a new generation of multi-agent world models.
>
---
#### [new 023] Pseudo-View Enhancement via Confidence Fusion for Unposed Sparse-View Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，解决稀疏视角下重建质量差的问题。通过伪视图增强和高斯管理策略，提升重建完整性与几何一致性。**

- **链接: [https://arxiv.org/pdf/2602.21535v1](https://arxiv.org/pdf/2602.21535v1)**

> **作者:** Beizhen Zhao; Sicheng Yu; Guanzhi Ding; Yu Hu; Hao Wang
>
> **备注:** 14 pages
>
> **摘要:** 3D scene reconstruction under unposed sparse viewpoints is a highly challenging yet practically important problem, especially in outdoor scenes due to complex lighting and scale variation. With extremely limited input views, directly utilizing diffusion model to synthesize pseudo frames will introduce unreasonable geometry, which will harm the final reconstruction quality. To address these issues, we propose a novel framework for sparse-view outdoor reconstruction that achieves high-quality results through bidirectional pseudo frame restoration and scene perception Gaussian management. Specifically, we introduce a bidirectional pseudo frame restoration method that restores missing content by diffusion-based synthesis guided by adjacent frames with a lightweight pseudo-view deblur model and confidence mask inference algorithm. Then we propose a scene perception Gaussian management strategy that optimize Gaussians based on joint depth-density information. These designs significantly enhance reconstruction completeness, suppress floating artifacts and improve overall geometric consistency under extreme view sparsity. Experiments on outdoor benchmarks demonstrate substantial gains over existing methods in both fidelity and stability.
>
---
#### [new 024] Which Tool Response Should I Trust? Tool-Expertise-Aware Chest X-ray Agent with Multimodal Agentic Learning
- **分类: cs.CV**

- **简介: 该论文属于医疗AI任务，旨在解决多工具冲突与信任问题。通过构建TEA-CXA框架，实现多模态下工具可信度的动态学习与决策。**

- **链接: [https://arxiv.org/pdf/2602.21517v1](https://arxiv.org/pdf/2602.21517v1)**

> **作者:** Zheang Huai; Honglong Yang; Xiaomeng Li
>
> **备注:** 11 pages
>
> **摘要:** AI agents with tool-use capabilities show promise for integrating the domain expertise of various tools. In the medical field, however, tools are usually AI models that are inherently error-prone and can produce contradictory responses. Existing research on medical agents lacks sufficient understanding of the tools' realistic reliability and thus cannot effectively resolve tool conflicts. To address this gap, this paper introduces a framework that enables an agent to interact with tools and empirically learn their practical trustworthiness across different types of multimodal queries via agentic learning. As a concrete instantiation, we focus on chest X-ray analysis and present a tool-expertise-aware chest X-ray agent (TEA-CXA). When tool outputs disagree, the agent experimentally accepts or rejects multimodal tool results, receives rewards, and learns which tool to trust for each query type. Importantly, TEA-CXA extends existing codebases for reinforcement learning with multi-turn tool-calling that focus on textual inputs, to support multimodal contexts effectively. In addition, we enhance the codebase for medical use scenarios by supporting multiple tool calls in one turn, parallel tool inference, and multi-image accommodation within a single user query. Our code framework is applicable to general medical research on multi-turn tool-calling reinforcement learning in multimodal settings. Experiments show that TEA-CXA outperforms the state-of-the-art methods and a comprehensive set of baselines. Code will be released.
>
---
#### [new 025] Towards Controllable Video Synthesis of Routine and Rare OR Events
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于视频生成任务，旨在解决OR场景中罕见事件数据不足的问题。通过构建扩散框架，实现可控视频合成，提升安全事件检测能力。**

- **链接: [https://arxiv.org/pdf/2602.21365v1](https://arxiv.org/pdf/2602.21365v1)**

> **作者:** Dominik Schneider; Lalithkumar Seenivasan; Sampath Rapuri; Vishalroshan Anil; Aiza Maksutova; Yiqing Shen; Jan Emily Mangulabnan; Hao Ding; Jose L. Porras; Masaru Ishii; Mathias Unberath
>
> **备注:** Accepted to IPCAI 2026 and submitted to IJCARs
>
> **摘要:** Purpose: Curating large-scale datasets of operating room (OR) workflow, encompassing rare, safety-critical, or atypical events, remains operationally and ethically challenging. This data bottleneck complicates the development of ambient intelligence for detecting, understanding, and mitigating rare or safety-critical events in the OR. Methods: This work presents an OR video diffusion framework that enables controlled synthesis of rare and safety-critical events. The framework integrates a geometric abstraction module, a conditioning module, and a fine-tuned diffusion model to first transform OR scenes into abstract geometric representations, then condition the synthesis process, and finally generate realistic OR event videos. Using this framework, we also curate a synthetic dataset to train and validate AI models for detecting near-misses of sterile-field violations. Results: In synthesizing routine OR events, our method outperforms off-the-shelf video diffusion baselines, achieving lower FVD/LPIPS and higher SSIM/PSNR in both in- and out-of-domain datasets. Through qualitative results, we illustrate its ability for controlled video synthesis of counterfactual events. An AI model trained and validated on the generated synthetic data achieved a RECALL of 70.13% in detecting near safety-critical events. Finally, we conduct an ablation study to quantify performance gains from key design choices. Conclusion: Our solution enables controlled synthesis of routine and rare OR events from abstract geometric representations. Beyond demonstrating its capability to generate rare and safety-critical scenarios, we show its potential to support the development of ambient intelligence models.
>
---
#### [new 026] Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频深度伪造检测任务，旨在解决VLMs对动态时间不一致性的识别不足问题。提出FAQ基准，通过多级评估提升模型的时序推理能力。**

- **链接: [https://arxiv.org/pdf/2602.21779v1](https://arxiv.org/pdf/2602.21779v1)**

> **作者:** Zheyuan Gu; Qingsong Zhao; Yusong Wang; Zhaohong Huang; Xinqi Li; Cheng Yuan; Jiaowei Shao; Chi Zhang; Xuelong Li
>
> **备注:** 16 pages, 9 figures. Submitted to CVPR 2026
>
> **摘要:** Current Vision-Language Models (VLMs) for deepfake detection excel at identifying spatial artifacts but overlook a critical dimension: temporal inconsistencies in video forgeries. Adapting VLMs to reason about these dynamic cues remains a distinct challenge. To bridge this gap, we propose Forensic Answer-Questioning (FAQ), a large-scale benchmark that formulates temporal deepfake analysis as a multiple-choice task. FAQ introduces a three-level hierarchy to progressively evaluate and equip VLMs with forensic capabilities: (1) Facial Perception, testing the ability to identify static visual artifacts; (2) Temporal Deepfake Grounding, requiring the localization of dynamic forgery artifacts across frames; and (3) Forensic Reasoning, challenging models to synthesize evidence for final authenticity verdicts. We evaluate a range of VLMs on FAQ and generate a corresponding instruction-tuning set, FAQ-IT. Extensive experiments show that models fine-tuned on FAQ-IT achieve advanced performance on both in-domain and cross-dataset detection benchmarks. Ablation studies further validate the impact of our key design choices, confirming that FAQ is the driving force behind the temporal reasoning capabilities of these VLMs.
>
---
#### [new 027] WildSVG: Towards Reliable SVG Generation Under Real-Word Conditions
- **分类: cs.CV**

- **简介: 该论文提出SVG提取任务，解决真实场景下生成可靠SVG的问题。构建了WildSVG基准，包含真实和合成数据集，评估现有模型并探索改进方法。**

- **链接: [https://arxiv.org/pdf/2602.21416v1](https://arxiv.org/pdf/2602.21416v1)**

> **作者:** Marco Terral; Haotian Zhang; Tianyang Zhang; Meng Lin; Xiaoqing Xie; Haoran Dai; Darsh Kaushik; Pai Peng; Nicklas Scharpff; David Vazquez; Joan Rodriguez
>
> **备注:** 10 pages, 6 pages of additional material
>
> **摘要:** We introduce the task of SVG extraction, which consists in translating specific visual inputs from an image into scalable vector graphics. Existing multimodal models achieve strong results when generating SVGs from clean renderings or textual descriptions, but they fall short in real-world scenarios where natural images introduce noise, clutter, and domain shifts. A central challenge in this direction is the lack of suitable benchmarks. To address this need, we introduce the WildSVG Benchmark, formed by two complementary datasets: Natural WildSVG, built from real images containing company logos paired with their SVG annotations, and Synthetic WildSVG, which blends complex SVG renderings into real scenes to simulate difficult conditions. Together, these resources provide the first foundation for systematic benchmarking SVG extraction. We benchmark state-of-the-art multimodal models and find that current approaches perform well below what is needed for reliable SVG extraction in real scenarios. Nonetheless, iterative refinement methods point to a promising path forward, and model capabilities are steadily improving
>
---
#### [new 028] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型幻觉问题。通过分析激活模式，提出动态多模态激活引导方法，提升模型真实性。**

- **链接: [https://arxiv.org/pdf/2602.21704v1](https://arxiv.org/pdf/2602.21704v1)**

> **作者:** Jianghao Yin; Qin Chen; Kedi Chen; Jie Zhou; Xingjiao Wu; Liang He
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) exhibit outstanding performance on vision-language tasks but struggle with hallucination problems. Through in-depth analysis of LVLM activation patterns, we reveal two key findings: 1) truthfulness and visual perception capabilities predominantly engage different subsets of attention heads within the model architecture; and 2) truthfulness steering vectors vary significantly across different semantic contexts. Based on these observations, we propose Dynamic Multimodal Activation Steering, a training-free approach for hallucination mitigation. Our method constructs a semantic-based truthfulness steering vector database and computes visual perception steering vectors, enabling context-aware interventions during inference by dynamically selecting the most relevant steering vectors based on input semantic similarity and applying them to the most influential attention heads. We conduct comprehensive experiments across multiple models and datasets, demonstrating that our approach significantly enhances model performance, outperforming existing state-of-the-art methods.
>
---
#### [new 029] Automatic Map Density Selection for Locally-Performant Visual Place Recognition
- **分类: cs.CV**

- **简介: 该论文属于视觉位置识别任务，解决长期部署中局部性能不一致问题。通过动态选择地图密度，确保满足用户指定的局部召回率和覆盖比例要求。**

- **链接: [https://arxiv.org/pdf/2602.21473v1](https://arxiv.org/pdf/2602.21473v1)**

> **作者:** Somayeh Hussaini; Tobias Fischer; Michael Milford
>
> **备注:** Under Review
>
> **摘要:** A key challenge in translating Visual Place Recognition (VPR) from the lab to long-term deployment is ensuring a priori that a system can meet user-specified performance requirements across different parts of an environment, rather than just on average globally. A critical mechanism for controlling local VPR performance is the density of the reference mapping database, yet this factor is largely neglected in existing work, where benchmark datasets with fixed, engineering-driven (sensors, storage, GPS frequency) sampling densities are typically used. In this paper, we propose a dynamic VPR mapping approach that uses pairs of reference traverses from the target environment to automatically select an appropriate map density to satisfy two user-defined requirements: (1) a target Local Recall@1 level, and (2) the proportion of the operational environment over which this requirement must be met or exceeded, which we term the Recall Achievement Rate (RAR). Our approach is based on the hypothesis that match patterns between multiple reference traverses, evaluated across different map densities, can be modelled to predict the density required to meet these performance targets on unseen deployment data. Through extensive experiments across multiple VPR methods and the Nordland and Oxford RobotCar benchmarks, we show that our system consistently achieves or exceeds the specified local recall level over at least the user-specified proportion of the environment. Comparisons with alternative baselines demonstrate that our approach reliably selects the correct operating point in map density, avoiding unnecessary over-densification. Finally, ablation studies and analysis evaluate sensitivity to reference map choice and local space definitions, and reveal that conventional global Recall@1 is a poor predictor of the often more operationally meaningful RAR metric.
>
---
#### [new 030] Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow
- **分类: cs.CV**

- **简介: 该论文属于3D模型编辑任务，解决单视角编辑与多视角不一致问题。提出Easy3E框架，通过稀疏体素流实现全局一致变形，并利用法线引导恢复细节。**

- **链接: [https://arxiv.org/pdf/2602.21499v1](https://arxiv.org/pdf/2602.21499v1)**

> **作者:** Shimin Hu; Yuanyi Wei; Fei Zha; Yudong Guo; Juyong Zhang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Existing 3D editing methods rely on computationally intensive scene-by-scene iterative optimization and suffer from multi-view inconsistency. We propose an effective and fully feedforward 3D editing framework based on the TRELLIS generative backbone, capable of modifying 3D models from a single editing view. Our framework addresses two key issues: adapting training-free 2D editing to structured 3D representations, and overcoming the bottleneck of appearance fidelity in compressed 3D features. To ensure geometric consistency, we introduce Voxel FlowEdit, an edit-driven flow in the sparse voxel latent space that achieves globally consistent 3D deformation in a single pass. To restore high-fidelity details, we develop a normal-guided single to multi-view generation module as an external appearance prior, successfully recovering high-frequency textures. Experiments demonstrate that our method enables fast, globally consistent, and high-fidelity 3D model editing.
>
---
#### [new 031] Meta-FC: Meta-Learning with Feature Consistency for Robust and Generalizable Watermarking
- **分类: cs.CV**

- **简介: 该论文属于水印任务，旨在提升水印模型的鲁棒性和泛化能力。针对现有方法在训练中忽略干扰关系的问题，提出Meta-FC策略，通过元学习和特征一致性增强模型性能。**

- **链接: [https://arxiv.org/pdf/2602.21849v1](https://arxiv.org/pdf/2602.21849v1)**

> **作者:** Yuheng Li; Weitong Chen; Chengcheng Zhu; Jiale Zhang; Chunpeng Ge; Di Wu; Guodong Long
>
> **摘要:** Deep learning-based watermarking has made remarkable progress in recent years. To achieve robustness against various distortions, current methods commonly adopt a training strategy where a \underline{\textbf{s}}ingle \underline{\textbf{r}}andom \underline{\textbf{d}}istortion (SRD) is chosen as the noise layer in each training batch. However, the SRD strategy treats distortions independently within each batch, neglecting the inherent relationships among different types of distortions and causing optimization conflicts across batches. As a result, the robustness and generalizability of the watermarking model are limited. To address this issue, we propose a novel training strategy that enhances robustness and generalization via \underline{\textbf{meta}}-learning with \underline{\textbf{f}}eature \underline{\textbf{c}}onsistency (Meta-FC). Specifically, we randomly sample multiple distortions from the noise pool to construct a meta-training task, while holding out one distortion as a simulated ``unknown'' distortion for the meta-testing phase. Through meta-learning, the model is encouraged to identify and utilize neurons that exhibit stable activations across different types of distortions, mitigating the optimization conflicts caused by the random sampling of diverse distortions in each batch. To further promote the transformation of stable activations into distortion-invariant representations, we introduce a feature consistency loss that constrains the decoded features of the same image subjected to different distortions to remain consistent. Extensive experiments demonstrate that, compared to the SRD training strategy, Meta-FC improves the robustness and generalization of various watermarking models by an average of 1.59\%, 4.71\%, and 2.38\% under high-intensity, combined, and unknown distortions.
>
---
#### [new 032] GeoDiv: Framework For Measuring Geographical Diversity In Text-To-Image Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型输出缺乏地理多样性的问题。提出GeoDiv框架，通过SEVI和VDI评估模型的地理偏见，揭示其社会经济和视觉多样性不足的问题。**

- **链接: [https://arxiv.org/pdf/2602.22120v1](https://arxiv.org/pdf/2602.22120v1)**

> **作者:** Abhipsa Basu; Mohana Singh; Shashank Agnihotri; Margret Keuper; R. Venkatesh Babu
>
> **备注:** ICLR 2026
>
> **摘要:** Text-to-image (T2I) models are rapidly gaining popularity, yet their outputs often lack geographical diversity, reinforce stereotypes, and misrepresent regions. Given their broad reach, it is critical to rigorously evaluate how these models portray the world. Existing diversity metrics either rely on curated datasets or focus on surface-level visual similarity, limiting interpretability. We introduce GeoDiv, a framework leveraging large language and vision-language models to assess geographical diversity along two complementary axes: the Socio-Economic Visual Index (SEVI), capturing economic and condition-related cues, and the Visual Diversity Index (VDI), measuring variation in primary entities and backgrounds. Applied to images generated by models such as Stable Diffusion and FLUX.1-dev across $10$ entities and $16$ countries, GeoDiv reveals a consistent lack of diversity and identifies fine-grained attributes where models default to biased portrayals. Strikingly, depictions of countries like India, Nigeria, and Colombia are disproportionately impoverished and worn, reflecting underlying socio-economic biases. These results highlight the need for greater geographical nuance in generative models. GeoDiv provides the first systematic, interpretable framework for measuring such biases, marking a step toward fairer and more inclusive generative systems. Project page: https://abhipsabasu.github.io/geodiv
>
---
#### [new 033] AdaSpot: Spend Resolution Where It Matters for Precise Event Spotting
- **分类: cs.CV**

- **简介: 该论文提出AdaSpot，用于精确事件定位任务，解决视频中快速动作定位的高精度与效率问题，通过自适应选择关键区域提升性能。**

- **链接: [https://arxiv.org/pdf/2602.22073v1](https://arxiv.org/pdf/2602.22073v1)**

> **作者:** Artur Xarles; Sergio Escalera; Thomas B. Moeslund; Albert Clapés
>
> **摘要:** Precise Event Spotting aims to localize fast-paced actions or events in videos with high temporal precision, a key task for applications in sports analytics, robotics, and autonomous systems. Existing methods typically process all frames uniformly, overlooking the inherent spatio-temporal redundancy in video data. This leads to redundant computation on non-informative regions while limiting overall efficiency. To remain tractable, they often spatially downsample inputs, losing fine-grained details crucial for precise localization. To address these limitations, we propose \textbf{AdaSpot}, a simple yet effective framework that processes low-resolution videos to extract global task-relevant features while adaptively selecting the most informative region-of-interest in each frame for high-resolution processing. The selection is performed via an unsupervised, task-aware strategy that maintains spatio-temporal consistency across frames and avoids the training instability of learnable alternatives. This design preserves essential fine-grained visual cues with a marginal computational overhead compared to low-resolution-only baselines, while remaining far more efficient than uniform high-resolution processing. Experiments on standard PES benchmarks demonstrate that \textbf{AdaSpot} achieves state-of-the-art performance under strict evaluation metrics (\eg, $+3.96$ and $+2.26$ mAP$@0$ frames on Tennis and FineDiving), while also maintaining strong results under looser metrics. Code is available at: \href{https://github.com/arturxe2/AdaSpot}{https://github.com/arturxe2/AdaSpot}.
>
---
#### [new 034] HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles
- **分类: cs.CV**

- **简介: 该论文提出HorizonForge，解决自动驾驶场景生成中的可控性与真实感问题。通过Gaussian Splats和Mesh表示，实现精细3D编辑与语言驱动车辆插入，提升模拟效果。**

- **链接: [https://arxiv.org/pdf/2602.21333v1](https://arxiv.org/pdf/2602.21333v1)**

> **作者:** Yifan Wang; Francesco Pittaluga; Zaid Tasneem; Chenyu You; Manmohan Chandraker; Ziyu Jiang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Controllable driving scene generation is critical for realistic and scalable autonomous driving simulation, yet existing approaches struggle to jointly achieve photorealism and precise control. We introduce HorizonForge, a unified framework that reconstructs scenes as editable Gaussian Splats and Meshes, enabling fine-grained 3D manipulation and language-driven vehicle insertion. Edits are rendered through a noise-aware video diffusion process that enforces spatial and temporal consistency, producing diverse scene variations in a single feed-forward pass without per-trajectory optimization. To standardize evaluation, we further propose HorizonSuite, a comprehensive benchmark spanning ego- and agent-level editing tasks such as trajectory modifications and object manipulation. Extensive experiments show that Gaussian-Mesh representation delivers substantially higher fidelity than alternative 3D representations, and that temporal priors from video diffusion are essential for coherent synthesis. Combining these findings, HorizonForge establishes a simple yet powerful paradigm for photorealistic, controllable driving simulation, achieving an 83.4% user-preference gain and a 25.19% FID improvement over the second best state-of-the-art method. Project page: https://horizonforge.github.io/ .
>
---
#### [new 035] UniVBench: Towards Unified Evaluation for Video Foundation Models
- **分类: cs.CV**

- **简介: 该论文提出UniVBench，用于评估视频基础模型的统一能力，解决现有基准碎片化问题，涵盖视频理解、生成、编辑和重建任务。**

- **链接: [https://arxiv.org/pdf/2602.21835v1](https://arxiv.org/pdf/2602.21835v1)**

> **作者:** Jianhui Wei; Xiaotian Zhang; Yichen Li; Yuan Wang; Yan Zhang; Ziyi Chen; Zhihang Tang; Wei Xu; Zuozhu Liu
>
> **摘要:** Video foundation models aim to integrate video understanding, generation, editing, and instruction following within a single framework, making them a central direction for next-generation multimodal systems. However, existing evaluation benchmarks remain fragmented and limited in scope, as they each target a single task, rely on task-specific metrics, and typically use short or simple video clips. As a result, they do not capture the unified capabilities that these models are designed to deliver. To address this gap, we introduce UniVBench, a benchmark purpose-built for evaluating video foundation models across four core abilities: video understanding, video generation, video editing, and a newly proposed task, video reconstruction, which assesses how faithfully a model can reproduce video content it has encountered. Our benchmark substantially expands the complexity of evaluation by incorporating 200 high-quality, diverse and multi-shot videos, each paired with detailed captions, multi-format editing instructions, and reference images. All videos are human-created and carefully validated, offering richer cinematic information than prior benchmarks. In addition, we develop a unified agentic evaluation system (UniV-Eval) that standardizes prompting, instruction parsing, and scoring across all tasks, enabling fair, scalable, and reproducible comparisons of unified video models. By grounding evaluation in instruction-based multi-shot video tasks, UniVBench provides the first framework for measuring the integrated capabilities that video foundation models aim to achieve. Extensive human annotations ensure our evaluation aligns with human judgment, enabling rigorous assessment and accelerating progress toward robust video intelligence.
>
---
#### [new 036] Brain3D: Brain Report Automation via Inflated Vision Transformers in 3D
- **分类: cs.CV**

- **简介: 该论文属于医学影像报告生成任务，解决3D脑肿瘤MRI自动报告生成问题。提出Brain3D框架，通过3D视觉-语言模型提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2602.22098v1](https://arxiv.org/pdf/2602.22098v1)**

> **作者:** Mariano Barone; Francesco Di Serio; Giuseppe Riccio; Antonio Romano; Marco Postiglione; Antonino Ferraro; Vincenzo Moscato
>
> **摘要:** Current medical vision-language models (VLMs) process volumetric brain MRI using 2D slice-based approximations, fragmenting the spatial context required for accurate neuroradiological interpretation. We developed \textbf{Brain3D}, a staged vision-language framework for automated radiology report generation from 3D brain tumor MRI. Our approach inflates a pretrained 2D medical encoder into a native 3D architecture and progressively aligns it with a causal language model through three stages: contrastive grounding, supervised projector warmup, and LoRA-based linguistic specialization. Unlike generalist 3D medical VLMs, \textbf{Brain3D} is tailored to neuroradiology, where hemispheric laterality, tumor infiltration patterns, and anatomical localization are critical. Evaluated on 468 subjects (BraTS pathological cases plus healthy controls), our model achieves a Clinical Pathology F1 of 0.951 versus 0.413 for a strong 2D baseline while maintaining perfect specificity on healthy scans. The staged alignment proves essential: contrastive grounding establishes visual-textual correspondence, projector warmup stabilizes conditioning, and LoRA adaptation shifts output from verbose captions to structured clinical reports\footnote{Our code is publicly available for transparency and reproducibility
>
---
#### [new 037] MindDriver: Introducing Progressive Multimodal Reasoning for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决VLM在轨迹规划中的语义与物理空间对齐问题。提出MindDriver框架，通过多模态推理实现从语义到轨迹的渐进式规划。**

- **链接: [https://arxiv.org/pdf/2602.21952v1](https://arxiv.org/pdf/2602.21952v1)**

> **作者:** Lingjun Zhang; Yujian Yuan; Changjie Wu; Xinyuan Chang; Xin Cai; Shuang Zeng; Linzhe Shi; Sijin Wang; Hang Zhang; Mu Xu
>
> **备注:** CVPR2026; Yujian Yuan and Lingjun Zhang contributed equally with random order
>
> **摘要:** Vision-Language Models (VLM) exhibit strong reasoning capabilities, showing promise for end-to-end autonomous driving systems. Chain-of-Thought (CoT), as VLM's widely used reasoning strategy, is facing critical challenges. Existing textual CoT has a large gap between text semantic space and trajectory physical space. Although the recent approach utilizes future image to replace text as CoT process, it lacks clear planning-oriented objective guidance to generate images with accurate scene evolution. To address these, we innovatively propose MindDriver, a progressive multimodal reasoning framework that enables VLM to imitate human-like progressive thinking for autonomous driving. MindDriver presents semantic understanding, semantic-to-physical space imagination, and physical-space trajectory planning. To achieve aligned reasoning processes in MindDriver, we develop a feedback-guided automatic data annotation pipeline to generate aligned multimodal reasoning training data. Furthermore, we develop a progressive reinforcement fine-tuning method to optimize the alignment through progressive high- level reward-based learning. MindDriver demonstrates superior performance in both nuScences open-loop and Bench2Drive closed-loop evaluation. Codes are available at https://github.com/hotdogcheesewhite/MindDriver.
>
---
#### [new 038] GFPL: Generative Federated Prototype Learning for Resource-Constrained and Data-Imbalanced Vision Task
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉任务，解决资源受限和数据不平衡下的联邦学习问题。提出GFPL框架，通过原型生成与聚合提升模型性能，降低通信成本。**

- **链接: [https://arxiv.org/pdf/2602.21873v1](https://arxiv.org/pdf/2602.21873v1)**

> **作者:** Shiwei Lu; Yuhang He; Jiashuo Li; Qiang Wang; Yihong Gong
>
> **摘要:** Federated learning (FL) facilitates the secure utilization of decentralized images, advancing applications in medical image recognition and autonomous driving. However, conventional FL faces two critical challenges in real-world deployment: ineffective knowledge fusion caused by model updates biased toward majority-class features, and prohibitive communication overhead due to frequent transmissions of high-dimensional model parameters. Inspired by the human brain's efficiency in knowledge integration, we propose a novel Generative Federated Prototype Learning (GFPL) framework to address these issues. Within this framework, a prototype generation method based on Gaussian Mixture Model (GMM) captures the statistical information of class-wise features, while a prototype aggregation strategy using Bhattacharyya distance effectively fuses semantically similar knowledge across clients. In addition, these fused prototypes are leveraged to generate pseudo-features, thereby mitigating feature distribution imbalance across clients. To further enhance feature alignment during local training, we devise a dual-classifier architecture, optimized via a hybrid loss combining Dot Regression and Cross-Entropy. Extensive experiments on benchmarks show that GFPL improves model accuracy by 3.6% under imbalanced data settings while maintaining low communication cost.
>
---
#### [new 039] Momentum Memory for Knowledge Distillation in Computational Pathology
- **分类: cs.CV**

- **简介: 该论文属于计算病理学中的知识蒸馏任务，旨在解决多模态数据不足导致的模型性能问题。通过引入动量记忆机制，提升模型泛化能力与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.21395v1](https://arxiv.org/pdf/2602.21395v1)**

> **作者:** Yongxin Guo; Hao Lu; Onur C. Koyun; Zhengjie Zhu; Muhammet Fatih Demir; Metin Nafi Gurcan
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Multimodal learning that integrates genomics and histopathology has shown strong potential in cancer diagnosis, yet its clinical translation is hindered by the limited availability of paired histology-genomics data. Knowledge distillation (KD) offers a practical solution by transferring genomic supervision into histopathology models, enabling accurate inference using histology alone. However, existing KD methods rely on batch-local alignment, which introduces instability due to limited within-batch comparisons and ultimately degrades performance. To address these limitations, we propose Momentum Memory Knowledge Distillation (MoMKD), a cross-modal distillation framework driven by a momentum-updated memory. This memory aggregates genomic and histopathology information across batches, effectively enlarging the supervisory context available to each mini-batch. Furthermore, we decouple the gradients of the genomics and histology branches, preventing genomic signals from dominating histology feature learning during training and eliminating the modality-gap issue at inference time. Extensive experiments on the TCGA-BRCA benchmark (HER2, PR, and ODX classification tasks) and an independent in-house testing dataset demonstrate that MoMKD consistently outperforms state-of-the-art MIL and multimodal KD baselines, delivering strong performance and generalization under histology-only inference. Overall, MoMKD establishes a robust and generalizable knowledge distillation paradigm for computational pathology.
>
---
#### [new 040] WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs
- **分类: cs.CV**

- **简介: 该论文提出WeaveTime，解决视频大模型在流式场景下的时间感知问题。针对时间无关性导致的顺序混淆和历史混淆，通过引入时序重建目标和动态关注缓存，提升模型对时间序列的理解与推理能力。**

- **链接: [https://arxiv.org/pdf/2602.22142v1](https://arxiv.org/pdf/2602.22142v1)**

> **作者:** Yulin Zhang; Cheng Shi; Sibei Yang
>
> **备注:** Accepted at CVPR 2026 (preview; camera-ready in preparation)
>
> **摘要:** Recent advances in Multimodal Large Language Models have greatly improved visual understanding and reasoning, yet their quadratic attention and offline training protocols make them ill-suited for streaming settings where frames arrive sequentially and future observations are inaccessible. We diagnose a core limitation of current Video-LLMs, namely Time-Agnosticism, in which videos are treated as an unordered bag of evidence rather than a causally ordered sequence, yielding two failures in streams: temporal order ambiguity, in which the model cannot follow or reason over the correct chronological order, and past-current focus blindness where it fails to distinguish present observations from accumulated history. We present WeaveTime, a simple, efficient, and model agnostic framework that first teaches order and then uses order. We introduce a lightweight Temporal Reconstruction objective-our Streaming Order Perception enhancement-that instills order aware representations with minimal finetuning and no specialized streaming data. At inference, a Past-Current Dynamic Focus Cache performs uncertainty triggered, coarse-to-fine retrieval, expanding history only when needed. Plugged into exsiting Video-LLM without architectural changes, WeaveTime delivers consistent gains on representative streaming benchmarks, improving accuracy while reducing latency. These results establish WeaveTime as a practical path toward time aware stream Video-LLMs under strict online, time causal constraints. Code and weights will be made publicly available. Project Page: https://zhangyl4.github.io/publications/weavetime/
>
---
#### [new 041] TIRAuxCloud: A Thermal Infrared Dataset for Day and Night Cloud Detection
- **分类: cs.CV**

- **简介: 该论文提出TIRAuxCloud数据集，用于昼夜云检测任务，解决夜间云识别困难的问题，通过多模态热红外数据和辅助信息提升检测精度。**

- **链接: [https://arxiv.org/pdf/2602.21905v1](https://arxiv.org/pdf/2602.21905v1)**

> **作者:** Alexis Apostolakis; Vasileios Botsos; Niklas Wölki; Andrea Spichtinger; Nikolaos Ioannis Bountos; Ioannis Papoutsis; Panayiotis Tsanakas
>
> **摘要:** Clouds are a major obstacle in Earth observation, limiting the usability and reliability of critical remote sensing applications such as fire disaster response, urban heat island monitoring, and snow and ice cover mapping. Therefore, the ability to detect clouds 24/7 is of paramount importance. While visible and near-infrared bands are effective for daytime cloud detection, their dependence on solar illumination makes them unsuitable for nighttime monitoring. In contrast, thermal infrared (TIR) imagery plays a crucial role in detecting clouds at night, when sunlight is absent. Due to their generally lower temperatures, clouds emit distinct thermal signatures that are detectable in TIR bands. Despite this, accurate nighttime cloud detection remains challenging due to limited spectral information and the typically lower spatial resolution of TIR imagery. To address these challenges, we present TIRAuxCloud, a multi-modal dataset centered around thermal spectral data to facilitate cloud segmentation under both daytime and nighttime conditions. The dataset comprises a unique combination of multispectral data (TIR, optical, and near-infrared bands) from Landsat and VIIRS, aligned with auxiliary information layers. Elevation, land cover, meteorological variables, and cloud-free reference images are included to help reduce surface-cloud ambiguity and cloud formation uncertainty. To overcome the scarcity of manual cloud labels, we include a large set of samples with automated cloud masks and a smaller manually annotated subset to further evaluate and improve models. Comprehensive benchmarks are presented to establish performance baselines through supervised and transfer learning, demonstrating the dataset's value in advancing the development of innovative methods for day and night time cloud detection.
>
---
#### [new 042] Automating Timed Up and Go Phase Segmentation and Gait Analysis via the tugturn Markerless 3D Pipeline
- **分类: cs.CV**

- **简介: 该论文属于运动分析任务，旨在解决无标记TUG分析的自动化问题。提出tugturn.py工具，实现TUG阶段分割与步态分析。**

- **链接: [https://arxiv.org/pdf/2602.21425v1](https://arxiv.org/pdf/2602.21425v1)**

> **作者:** Abel Gonçalves Chinaglia; Guilherme Manna Cesar; Paulo Roberto Pereira Santiago
>
> **备注:** 16 pages, 2 figures, 1 pdf report, submitted to arXiv under cs.CV
>
> **摘要:** Instrumented Timed Up and Go (TUG) analysis can support clinical and research decision-making, but robust and reproducible markerless pipelines are still limited. We present \textit{tugturn.py}, a Python-based workflow for 3D markerless TUG processing that combines phase segmentation, gait-event detection, spatiotemporal metrics, intersegmental coordination, and dynamic stability analysis. The pipeline uses spatial thresholds to segment each trial into stand, first gait, turning, second gait, and sit phases, and applies a relative-distance strategy to detect heel-strike and toe-off events within valid gait windows. In addition to conventional kinematics, \textit{tugturn} provides Vector Coding outputs and Extrapolated Center of Mass (XCoM)-based metrics. The software is configured through TOML files and produces reproducible artifacts, including HTML reports, CSV tables, and quality-assurance visual outputs. A complete runnable example is provided with test data and command-line instructions. This manuscript describes the implementation, outputs, and reproducibility workflow of \textit{tugturn} as a focused software contribution for markerless biomechanical TUG analysis.
>
---
#### [new 043] VasGuideNet: Vascular Topology-Guided Couinaud Liver Segmentation with Structural Contrastive Loss
- **分类: cs.CV**

- **简介: 该论文属于肝脏分割任务，旨在解决血管附近边界不清晰和泛化能力差的问题。通过引入血管拓扑信息和结构对比损失，提升分割准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2602.21539v1](https://arxiv.org/pdf/2602.21539v1)**

> **作者:** Chaojie Shen; Jingjun Gu; Zihao Zhao; Ruocheng Li; Cunyuan Yang; Jiajun Bu; Lei Wu
>
> **摘要:** Accurate Couinaud liver segmentation is critical for preoperative surgical planning and tumor localization.However, existing methods primarily rely on image intensity and spatial location cues, without explicitly modeling vascular topology. As a result, they often produce indistinct boundaries near vessels and show limited generalization under anatomical variability.We propose VasGuideNet, the first Couinaud segmentation framework explicitly guided by vascular topology. Specifically, skeletonized vessels, Euclidean distance transform (EDT)--derived geometry, and k-nearest neighbor (kNN) connectivity are encoded into topology features using Graph Convolutional Networks (GCNs). These features are then injected into a 3D encoder--decoder backbone via a cross-attention fusion module. To further improve inter-class separability and anatomical consistency, we introduce a Structural Contrastive Loss (SCL) with a global memory bank.On Task08_HepaticVessel and our private LASSD dataset, VasGuideNet achieves Dice scores of 83.68% and 76.65% with RVDs of 1.68 and 7.08, respectively. It consistently outperforms representative baselines including UNETR, Swin UNETR, and G-UNETR++, delivering higher Dice/mIoU and lower RVD across datasets, demonstrating its effectiveness for anatomically consistent segmentation. Code is available at https://github.com/Qacket/VasGuideNet.git.
>
---
#### [new 044] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决视觉幻觉传播问题。提出一种无需训练的迭代框架，通过视觉证据监督推理过程，提升准确性并减少错误。**

- **链接: [https://arxiv.org/pdf/2602.21497v1](https://arxiv.org/pdf/2602.21497v1)**

> **作者:** Yongchang Zhang; Xianzheng Ma; Tianyi Liu; Guangquan Zhou; Yang Chen
>
> **备注:** CVPR2026 Accepted
>
> **摘要:** Recent large vision-language models (LVLMs) have demonstrated impressive reasoning ability by generating long chain-of-thought (CoT) responses. However, CoT reasoning in multimodal contexts is highly vulnerable to visual hallucination propagation: once an intermediate reasoning step becomes inconsistent with the visual evidence, subsequent steps-even if logically valid-can still lead to incorrect final answers. Existing solutions attempt to mitigate this issue by training models to "think with images" via reinforcement learning (RL). While effective, these methods are costly, model-specific, and difficult to generalize across architectures. Differently, we present a lightweight method that bypasses RL training and provides an iterative, training-free, plug-and-play framework for visually-grounded multimodal reasoning. Our key idea is to supervise each reasoning step at test time with visual evidence, ensuring that every decoded token is justified by corresponding visual cues. Concretely, we construct a textual visual-evidence pool that guides the model's reasoning generation. When existing evidence is insufficient, a visual decider module dynamically extracts additional relevant evidence from the image based on the ongoing reasoning context, expanding the pool until the model achieves sufficient visual certainty to terminate reasoning and produce the final answer. Extensive experiments on multiple LVLM backbones and benchmarks demonstrate the effectiveness of our approach. Our method achieves 16.5%-29.5% improvements on TreeBench and 13.7% RH-AUC gains on RH-Bench, substantially reducing hallucination rates while improving reasoning accuracy without additional training.
>
---
#### [new 045] AutoSew: A Geometric Approach to Stitching Prediction with Graph Neural Networks
- **分类: cs.CV**

- **简介: 该论文属于服装自动缝合任务，旨在解决无标注和语义线索下的缝合预测问题。通过几何方法与图神经网络，实现无需人工输入的自动缝合。**

- **链接: [https://arxiv.org/pdf/2602.22052v1](https://arxiv.org/pdf/2602.22052v1)**

> **作者:** Pablo Ríos-Navarro; Elena Garces; Jorge Lopez-Moreno
>
> **备注:** WACV 2026
>
> **摘要:** Automating garment assembly from sewing patterns remains a significant challenge due to the lack of standardized annotation protocols and the frequent absence of semantic cues. Existing methods often rely on panel labels or handcrafted heuristics, which limit their applicability to real-world, non-conforming patterns. We present AutoSew, a fully automatic, geometry-based approach for predicting stitch correspondences directly from 2D pattern contours. AutoSew formulates the problem as a graph matching task, leveraging a Graph Neural Network to capture local and global geometric context, and employing a differentiable optimal transport solver to infer stitching relationships-including multi-edge connections. To support this task, we update the GarmentCodeData dataset modifying over 18k patterns with realistic multi-edge annotations, reflecting industrial assembly scenarios. AutoSew achieves 96% F1-score and successfully assembles 73.3% of test garments without error, outperforming existing methods while relying solely on geometric input. Our results demonstrate that geometry alone can robustly guide stitching prediction, enabling scalable garment assembly without manual input.
>
---
#### [new 046] From Statics to Dynamics: Physics-Aware Image Editing with Latent Transition Priors
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决物理合理性不足的问题。通过构建物理过渡数据集并提出PhysicEdit框架，提升编辑结果的物理真实性和知识一致性。**

- **链接: [https://arxiv.org/pdf/2602.21778v1](https://arxiv.org/pdf/2602.21778v1)**

> **作者:** Liangbing Zhao; Le Zhuo; Sayak Paul; Hongsheng Li; Mohamed Elhoseiny
>
> **备注:** All code, checkpoints, and datasets are available at https://liangbingzhao.github.io/statics2dynamics/
>
> **摘要:** Instruction-based image editing has achieved remarkable success in semantic alignment, yet state-of-the-art models frequently fail to render physically plausible results when editing involves complex causal dynamics, such as refraction or material deformation. We attribute this limitation to the dominant paradigm that treats editing as a discrete mapping between image pairs, which provides only boundary conditions and leaves transition dynamics underspecified. To address this, we reformulate physics-aware editing as predictive physical state transitions and introduce PhysicTran38K, a large-scale video-based dataset comprising 38K transition trajectories across five physical domains, constructed via a two-stage filtering and constraint-aware annotation pipeline. Building on this supervision, we propose PhysicEdit, an end-to-end framework equipped with a textual-visual dual-thinking mechanism. It combines a frozen Qwen2.5-VL for physically grounded reasoning with learnable transition queries that provide timestep-adaptive visual guidance to a diffusion backbone. Experiments show that PhysicEdit improves over Qwen-Image-Edit by 5.9% in physical realism and 10.1% in knowledge-grounded editing, setting a new state-of-the-art for open-source methods, while remaining competitive with leading proprietary models.
>
---
#### [new 047] StoryTailor:A Zero-Shot Pipeline for Action-Rich Multi-Subject Visual Narratives
- **分类: cs.CV**

- **简介: 该论文提出StoryTailor，解决多主体视觉叙事生成中的动作忠实性、身份一致性和背景连贯性问题，通过三个模块实现高质量图像序列生成。**

- **链接: [https://arxiv.org/pdf/2602.21273v1](https://arxiv.org/pdf/2602.21273v1)**

> **作者:** Jinghao Hu; Yuhe Zhang; GuoHua Geng; Kang Li; Han Zhang
>
> **备注:** 24 pages,19 figures,accepted by CVPR2026
>
> **摘要:** Generating multi-frame, action-rich visual narratives without fine-tuning faces a threefold tension: action text faithfulness, subject identity fidelity, and cross-frame background continuity. We propose StoryTailor, a zero-shot pipeline that runs on a single RTX 4090 (24 GB) and produces temporally coherent, identity-preserving image sequences from a long narrative prompt, per-subject references, and grounding boxes. Three synergistic modules drive the system: Gaussian-Centered Attention (GCA) to dynamically focus on each subject core and ease grounding-box overlaps; Action-Boost Singular Value Reweighting (AB-SVR) to amplify action-related directions in the text embedding space; and Selective Forgetting Cache (SFC) that retains transferable background cues, forgets nonessential history, and selectively surfaces retained cues to build cross-scene semantic ties. Compared with baseline methods, experiments show that CLIP-T improves by up to 10-15%, with DreamSim lower than strong baselines, while CLIP-I stays in a visually acceptable, competitive range. With matched resolution and steps on a 24 GB GPU, inference is faster than FluxKontext. Qualitatively, StoryTailor delivers expressive interactions and evolving yet stable scenes.
>
---
#### [new 048] Learning to Fuse and Reconstruct Multi-View Graphs for Diabetic Retinopathy Grading
- **分类: cs.CV**

- **简介: 该论文属于糖尿病视网膜病变分级任务，旨在解决多视角眼底图像融合中的视图相关性问题。提出MVGFDR框架，通过图融合与重建提升分级性能。**

- **链接: [https://arxiv.org/pdf/2602.21944v1](https://arxiv.org/pdf/2602.21944v1)**

> **作者:** Haoran Li; Yuxin Lin; Huan Wang; Xiaoling Luo; Qi Zhu; Jiahua Shi; Huaming Chen; Bo Du; Johan Barthelemy; Zongyan Xue; Jun Shen; Yong Xu
>
> **摘要:** Diabetic retinopathy (DR) is one of the leading causes of vision loss worldwide, making early and accurate DR grading critical for timely intervention. Recent clinical practices leverage multi-view fundus images for DR detection with a wide coverage of the field of view (FOV), motivating deep learning methods to explore the potential of multi-view learning for DR grading. However, existing methods often overlook the inter-view correlations when fusing multi-view fundus images, failing to fully exploit the inherent consistency across views originating from the same patient. In this work, we present MVGFDR, an end-to-end Multi-View Graph Fusion framework for DR grading. Different from existing methods that directly fuse visual features from multiple views, MVGFDR is equipped with a novel Multi-View Graph Fusion (MVGF) module to explicitly disentangle the shared and view-specific visual features. Specifically, MVGF comprises three key components: (1) Multi-view Graph Initialization, which constructs visual graphs via residual-guided connections and employs Discrete Cosine Transform (DCT) coefficients as frequency-domain anchors; (2) Multi-view Graph Fusion, which integrates selective nodes across multi-view graphs based on frequency-domain relevance to capture complementary view-specific information; and (3) Masked Cross-view Reconstruction, which leverages masked reconstruction of shared information across views to facilitate view-invariant representation learning. Extensive experimental results on MFIDDR, by far the largest multi-view fundus image dataset, demonstrate the superiority of our proposed approach over existing state-of-the-art approaches in diabetic retinopathy grading.
>
---
#### [new 049] HybridINR-PCGC: Hybrid Lossless Point Cloud Geometry Compression Bridging Pretrained Model and Implicit Neural Representation
- **分类: cs.CV**

- **简介: 该论文属于点云几何压缩任务，旨在解决传统方法依赖训练数据和INR方法效率低的问题。提出HybridINR-PCGC框架，结合预训练模型与INR，提升压缩率和效率。**

- **链接: [https://arxiv.org/pdf/2602.21662v1](https://arxiv.org/pdf/2602.21662v1)**

> **作者:** Wenjie Huang; Qi Yang; Shuting Xia; He Huang; Zhu Li; Yiling Xu
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** Learning-based point cloud compression presents superior performance to handcrafted codecs. However, pretrained-based methods, which are based on end-to-end training and expected to generalize to all the potential samples, suffer from training data dependency. Implicit neural representation (INR) based methods are distribution-agnostic and more robust, but they require time-consuming online training and suffer from the bitstream overhead from the overfitted model. To address these limitations, we propose HybridINR-PCGC, a novel hybrid framework that bridges the pretrained model and INR. Our framework retains distribution-agnostic properties while leveraging a pretrained network to accelerate convergence and reduce model overhead, which consists of two parts: the Pretrained Prior Network (PPN) and the Distribution Agnostic Refiner (DAR). We leverage the PPN, designed for fast inference and stable performance, to generate a robust prior for accelerating the DAR's convergence. The DAR is decomposed into a base layer and an enhancement layer, and only the enhancement layer needed to be packed into the bitstream. Finally, we propose a supervised model compression module to further supervise and minimize the bitrate of the enhancement layer parameters. Based on experiment results, HybridINR-PCGC achieves a significantly improved compression rate and encoding efficiency. Specifically, our method achieves a Bpp reduction of approximately 20.43% compared to G-PCC on 8iVFB. In the challenging out-of-distribution scenario Cat1B, our method achieves a Bpp reduction of approximately 57.85% compared to UniPCGC. And our method exhibits a superior time-rate trade-off, achieving an average Bpp reduction of 15.193% relative to the LINR-PCGC on 8iVFB.
>
---
#### [new 050] A Framework for Cross-Domain Generalization in Coronary Artery Calcium Scoring Across Gated and Non-Gated Computed Tomography
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决冠状动脉钙化评分在有门控和无门控CT间的跨域泛化问题。通过引入CARD-ViT框架，实现无需额外数据的准确评分。**

- **链接: [https://arxiv.org/pdf/2602.21935v1](https://arxiv.org/pdf/2602.21935v1)**

> **作者:** Mahmut S. Gokmen; Moneera N. Haque; Steve W. Leung; Caroline N. Leach; Seth Parker; Stephen B. Hobbs; Vincent L. Sorrell; W. Brent Seales; V. K. Cody Bumgardner
>
> **摘要:** Coronary artery calcium (CAC) scoring is a key predictor of cardiovascular risk, but it relies on ECG-gated CT scans, restricting its use to specialized cardiac imaging settings. We introduce an automated framework for CAC detection and lesion-specific Agatston scoring that operates across both gated and non-gated CT scans. At its core is CARD-ViT, a self-supervised Vision Transformer trained exclusively on gated CT data using DINO. Without any non-gated training data, our framework achieves 0.707 accuracy and a Cohen's kappa of 0.528 on the Stanford non-gated dataset, matching models trained directly on non-gated scans. On gated test sets, the framework achieves 0.910 accuracy with Cohen's kappa scores of 0.871 and 0.874 across independent datasets, demonstrating robust risk stratification. These results demonstrate the feasibility of cross-domain CAC scoring from gated to non-gated domains, supporting scalable cardiovascular screening in routine chest imaging without additional scans or annotations.
>
---
#### [new 051] Assessing airborne laser scanning and aerial photogrammetry for deep learning-based stand delineation
- **分类: cs.CV**

- **简介: 该论文属于森林样地分割任务，旨在解决传统手动分割的主观性问题。通过对比不同数据源的深度学习方法，验证了DAP-CHM的可行性及模型对输入数据的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.21709v1](https://arxiv.org/pdf/2602.21709v1)**

> **作者:** Håkon Næss Sandum; Hans Ole Ørka; Oliver Tomic; Terje Gobakken
>
> **备注:** 20 pages, 4 figures, 4 tables
>
> **摘要:** Accurate forest stand delineation is essential for forest inventory and management but remains a largely manual and subjective process. A recent study has shown that deep learning can produce stand delineations comparable to expert interpreters when combining aerial imagery and airborne laser scanning (ALS) data. However, temporal misalignment between data sources limits operational scalability. Canopy height models (CHMs) derived from digital photogrammetry (DAP) offer better temporal alignment but may smoothen canopy surface and canopy gaps, raising the question of whether they can reliably replace ALS-derived CHMs. Similarly, the inclusion of a digital terrain model (DTM) has been suggested to improve delineation performance, but has remained untested in published literature. Using expert-delineated forest stands as reference data, we assessed a U-Net-based semantic segmentation framework with municipality-level cross-validation across six municipalities in southeastern Norway. We compared multispectral aerial imagery combined with (i) an ALS-derived CHM, (ii) a DAP-derived CHM, and (iii) a DAP-derived CHM in combination with a DTM. Results showed comparable performance across all data combinations, reaching overall accuracy values between 0.90-0.91. Agreement between model predictions was substantially larger than agreement with the reference data, highlighting both model consistency and the inherent subjectivity of stand delineation. The similar performance of DAP-CHMs, despite the reduced structural detail, and the lack of improvements of the DTM indicate that the framework is resilient to variations in input data. These findings indicate that large datasets for deep learning-based stand delineations can be assembled using projects including temporally aligned ALS data and DAP point clouds.
>
---
#### [new 052] WHOLE: World-Grounded Hand-Object Lifted from Egocentric Videos
- **分类: cs.CV**

- **简介: 该论文属于视觉感知任务，解决egocentric视频中手与物体交互的联合重建问题。提出WHOLE方法，通过学习手-物运动先验，实现更准确的运动和姿态估计。**

- **链接: [https://arxiv.org/pdf/2602.22209v1](https://arxiv.org/pdf/2602.22209v1)**

> **作者:** Yufei Ye; Jiaman Li; Ryan Rong; C. Karen Liu
>
> **备注:** Project website: https://judyye.github.io/whole-www
>
> **摘要:** Egocentric manipulation videos are highly challenging due to severe occlusions during interactions and frequent object entries and exits from the camera view as the person moves. Current methods typically focus on recovering either hand or object pose in isolation, but both struggle during interactions and fail to handle out-of-sight cases. Moreover, their independent predictions often lead to inconsistent hand-object relations. We introduce WHOLE, a method that holistically reconstructs hand and object motion in world space from egocentric videos given object templates. Our key insight is to learn a generative prior over hand-object motion to jointly reason about their interactions. At test time, the pretrained prior is guided to generate trajectories that conform to the video observations. This joint generative reconstruction substantially outperforms approaches that process hands and objects separately followed by post-processing. WHOLE achieves state-of-the-art performance on hand motion estimation, 6D object pose estimation, and their relative interaction reconstruction. Project website: https://judyye.github.io/whole-www
>
---
#### [new 053] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis
- **分类: cs.CV**

- **简介: 该论文提出CARE模型，用于全切片图像分析，解决病理区域异质性问题，通过自监督和分子信息引导的区域建模提升性能。**

- **链接: [https://arxiv.org/pdf/2602.21637v1](https://arxiv.org/pdf/2602.21637v1)**

> **作者:** Di Zhang; Zhangpeng Gong; Xiaobo Pang; Jiashuai Liu; Junbo Lu; Hao Cui; Jiusong Ge; Zhi Zeng; Kai Yi; Yinghua Li; Si Liu; Tingsong Yu; Haoran Wang; Mireia Crispin-Ortuzar; eimiao Yu; Chen Li; Zeyu Gao
>
> **摘要:** Foundation models have recently achieved impressive success in computational pathology, demonstrating strong generalization across diverse histopathology tasks. However, existing models overlook the heterogeneous and non-uniform organization of pathological regions of interest (ROIs) because they rely on natural image backbones not tailored for tissue morphology. Consequently, they often fail to capture the coherent tissue architecture beyond isolated patches, limiting interpretability and clinical relevance. To address these challenges, we present Cross-modal Adaptive Region Encoder (CARE), a foundation model for pathology that automatically partitions WSIs into several morphologically relevant regions. Specifically, CARE employs a two-stage pretraining strategy: (1) a self-supervised unimodal pretraining stage that learns morphological representations from 34,277 whole-slide images (WSIs) without segmentation annotations, and (2) a cross-modal alignment stage that leverages RNA and protein profiles to refine the construction and representation of adaptive regions. This molecular guidance enables CARE to identify biologically relevant patterns and generate irregular yet coherent tissue regions, selecting the most representative area as ROI. CARE supports a broad range of pathology-related tasks, using either the ROI feature or the slide-level feature obtained by aggregating adaptive regions. Based on only one-tenth of the pretraining data typically used by mainstream foundation models, CARE achieves superior average performance across 33 downstream benchmarks, including morphological classification, molecular prediction, and survival analysis, and outperforms other foundation model baselines overall.
>
---
#### [new 054] EndoDDC: Learning Sparse to Dense Reconstruction for Endoscopic Robotic Navigation via Diffusion Depth Completion
- **分类: cs.CV**

- **简介: 该论文属于深度估计任务，旨在解决内镜环境下的稀疏深度重建问题。通过融合图像、稀疏深度和梯度信息，利用扩散模型优化深度图，提升重建精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.21893v1](https://arxiv.org/pdf/2602.21893v1)**

> **作者:** Yinheng Lin; Yiming Huang; Beilei Cui; Long Bai; Huxin Gao; Hongliang Ren; Jiewen Lai
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Accurate depth estimation plays a critical role in the navigation of endoscopic surgical robots, forming the foundation for 3D reconstruction and safe instrument guidance. Fine-tuning pretrained models heavily relies on endoscopic surgical datasets with precise depth annotations. While existing self-supervised depth estimation techniques eliminate the need for accurate depth annotations, their performance degrades in environments with weak textures and variable lighting, leading to sparse reconstruction with invalid depth estimation. Depth completion using sparse depth maps can mitigate these issues and improve accuracy. Despite the advances in depth completion techniques in general fields, their application in endoscopy remains limited. To overcome these limitations, we propose EndoDDC, an endoscopy depth completion method that integrates images, sparse depth information with depth gradient features, and optimizes depth maps through a diffusion model, addressing the issues of weak texture and light reflection in endoscopic environments. Extensive experiments on two publicly available endoscopy datasets show that our approach outperforms state-of-the-art models in both depth accuracy and robustness. This demonstrates the potential of our method to reduce visual errors in complex endoscopic environments. Our code will be released at https://github.com/yinheng-lin/EndoDDC.
>
---
#### [new 055] Space-Time Forecasting of Dynamic Scenes with Motion-aware Gaussian Grouping
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于动态场景预测任务，旨在解决有限观测下运动一致性与长期演化问题。提出MoGaF框架，通过运动感知的高斯分组实现时空一致的动态表示。**

- **链接: [https://arxiv.org/pdf/2602.21668v1](https://arxiv.org/pdf/2602.21668v1)**

> **作者:** Junmyeong Lee; Hoseung Choi; Minsu Cho
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** Forecasting dynamic scenes remains a fundamental challenge in computer vision, as limited observations make it difficult to capture coherent object-level motion and long-term temporal evolution. We present Motion Group-aware Gaussian Forecasting (MoGaF), a framework for long-term scene extrapolation built upon the 4D Gaussian Splatting representation. MoGaF introduces motion-aware Gaussian grouping and group-wise optimization to enforce physically consistent motion across both rigid and non-rigid regions, yielding spatially coherent dynamic representations. Leveraging this structured space-time representation, a lightweight forecasting module predicts future motion, enabling realistic and temporally stable scene evolution. Experiments on synthetic and real-world datasets demonstrate that MoGaF consistently outperforms existing baselines in rendering quality, motion plausibility, and long-term forecasting stability. Our project page is available at https://slime0519.github.io/mogaf
>
---
#### [new 056] CADC: Content Adaptive Diffusion-Based Generative Image Compression
- **分类: cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决扩散模型在低比特率下内容适应性不足的问题。通过三项创新技术提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2602.21591v1](https://arxiv.org/pdf/2602.21591v1)**

> **作者:** Xihua Sheng; Lingyu Zhu; Tianyu Zhang; Dong Liu; Shiqi Wang; Jing Wang
>
> **备注:** CVPR2026
>
> **摘要:** Diffusion-based generative image compression has demonstrated remarkable potential for achieving realistic reconstruction at ultra-low bitrates. The key to unlocking this potential lies in making the entire compression process content-adaptive, ensuring that the encoder's representation and the decoder's generative prior are dynamically aligned with the semantic and structural characteristics of the input image. However, existing methods suffer from three critical limitations that prevent effective content adaptation. First, isotropic quantization applies a uniform quantization step, failing to adapt to the spatially varying complexity of image content and creating a misalignment with the diffusion model's noise-dependent prior. Second, the information concentration bottleneck -- arising from the dimensional mismatch between the high-dimensional noisy latent and the diffusion decoder's fixed input -- prevents the model from adaptively preserving essential semantic information in the primary channels. Third, existing textual conditioning strategies either need significant textual bitrate overhead or rely on generic, content-agnostic textual prompts, thereby failing to provide adaptive semantic guidance efficiently. To overcome these limitations, we propose a content-adaptive diffusion-based image codec with three technical innovations: 1) an Uncertainty-Guided Adaptive Quantization method that learns spatial uncertainty maps to adaptively align quantization distortion with content characteristics; 2) an Auxiliary Decoder-Guided Information Concentration method that uses a lightweight auxiliary decoder to enforce content-aware information preservation in the primary latent channels; and 3) a Bitrate-Free Adaptive Textual Conditioning method that derives content-aware textual descriptions from the auxiliary reconstructed image, enabling semantic guidance without bitrate cost.
>
---
#### [new 057] FlowFixer: Towards Detail-Preserving Subject-Driven Generation
- **分类: cs.CV**

- **简介: 论文提出FlowFixer，用于解决主体驱动生成中细节丢失的问题。通过图像到图像的翻译和关键点匹配评估，提升生成质量，属于高保真图像生成任务。**

- **链接: [https://arxiv.org/pdf/2602.21402v1](https://arxiv.org/pdf/2602.21402v1)**

> **作者:** Jinyoung Jun; Won-Dong Jang; Wenbin Ouyang; Raghudeep Gadde; Jungbeom Lee
>
> **摘要:** We present FlowFixer, a refinement framework for subject-driven generation (SDG) that restores fine details lost during generation caused by changes in scale and perspective of a subject. FlowFixer proposes direct image-to-image translation from visual references, avoiding ambiguities in language prompts. To enable image-to-image training, we introduce a one-step denoising scheme to generate self-supervised training data, which automatically removes high-frequency details while preserving global structure, effectively simulating real-world SDG errors. We further propose a keypoint matching-based metric to properly assess fidelity in details beyond semantic similarities usually measured by CLIP or DINO. Experimental results demonstrate that FlowFixer outperforms state-of-the-art SDG methods in both qualitative and quantitative evaluations, setting a new benchmark for high-fidelity subject-driven generation.
>
---
#### [new 058] Following the Diagnostic Trace: Visual Cognition-guided Cooperative Network for Chest X-Ray Diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像诊断任务，旨在解决CAD系统与临床流程脱节、缺乏可解释性的问题。提出VCC-Net模型，通过视觉认知引导实现人机协作，提升诊断可靠性与透明度。**

- **链接: [https://arxiv.org/pdf/2602.21657v1](https://arxiv.org/pdf/2602.21657v1)**

> **作者:** Shaoxuan Wu; Jingkun Chen; Chong Ma; Cong Shen; Xiao Zhang; Jun Feng
>
> **摘要:** Computer-aided diagnosis (CAD) has significantly advanced automated chest X-ray diagnosis but remains isolated from clinical workflows and lacks reliable decision support and interpretability. Human-AI collaboration seeks to enhance the reliability of diagnostic models by integrating the behaviors of controllable radiologists. However, the absence of interactive tools seamlessly embedded within diagnostic routines impedes collaboration, while the semantic gap between radiologists' decision-making patterns and model representations further limits clinical adoption. To overcome these limitations, we propose a visual cognition-guided collaborative network (VCC-Net) to achieve the cooperative diagnostic paradigm. VCC-Net centers on visual cognition (VC) and employs clinically compatible interfaces, such as eye-tracking or the mouse, to capture radiologists' visual search traces and attention patterns during diagnosis. VCC-Net employs VC as a spatial cognition guide, learning hierarchical visual search strategies to localize diagnostically key regions. A cognition-graph co-editing module subsequently integrates radiologist VC with model inference to construct a disease-aware graph. The module captures dependencies among anatomical regions and aligns model representations with VC-driven features, mitigating radiologist bias and facilitating complementary, transparent decision-making. Experiments on the public datasets SIIM-ACR, EGD-CXR, and self-constructed TB-Mouse dataset achieved classification accuracies of 88.40%, 85.05%, and 92.41%, respectively. The attention maps produced by VCC-Net exhibit strong concordance with radiologists' gaze distributions, demonstrating a mutual reinforcement of radiologist and model inference. The code is available at https://github.com/IPMI-NWU/VCC-Net.
>
---
#### [new 059] Adversarial Robustness of Deep Learning-Based Thyroid Nodule Segmentation in Ultrasound
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究深度学习在超声甲状腺结节分割中的对抗鲁棒性，针对对抗攻击与防御方法进行评估，旨在提升模型的可靠性。**

- **链接: [https://arxiv.org/pdf/2602.21452v1](https://arxiv.org/pdf/2602.21452v1)**

> **作者:** Nicholas Dietrich; David McShannon
>
> **备注:** 14 pages, 3 figures, 3 tables
>
> **摘要:** Introduction: Deep learning-based segmentation models are increasingly integrated into clinical imaging workflows, yet their robustness to adversarial perturbations remains incompletely characterized, particularly for ultrasound images. We evaluated adversarial attacks and inference-time defenses for thyroid nodule segmentation in B-mode ultrasound. Methods: Two black-box adversarial attacks were developed: (1) Structured Speckle Amplification Attack (SSAA), which injects boundary-targeted noise, and (2) Frequency-Domain Ultrasound Attack (FDUA), which applies bandpass-filtered phase perturbations in the Fourier domain. Three inference-time mitigations were evaluated on adversarial images: randomized preprocessing with test-time augmentation, deterministic input denoising, and stochastic ensemble inference with consistency-aware aggregation. Experiments were conducted on a U-Net segmentation model trained on cine-clips from a database of 192 thyroid nodules. Results: The baseline model achieved a mean Dice similarity coefficient (DSC) of 0.76 (SD 0.20) on unperturbed images. SSAA reduced DSC by 0.29 (SD 0.20) while maintaining high visual similarity (SSIM = 0.94). FDUA resulted in a smaller DSC reduction of 0.11 (SD 0.09) with lower visual fidelity (SSIM = 0.82). Against SSAA, all three defenses significantly improved DSC after correction, with deterministic denoising showing the largest recovery (+0.10, p < 0.001), followed by randomized preprocessing (+0.09, p < 0.001), and stochastic ensemble inference (+0.08, p = 0.002). No defense achieved statistically significant improvement against FDUA. Conclusion: Spatial-domain adversarial perturbations in ultrasound segmentation showed partial mitigation with input preprocessing, whereas frequency-domain perturbations were not mitigated by the defenses, highlighting modality-specific challenges in adversarial robustness evaluation.
>
---
#### [new 060] When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters
- **分类: cs.CV**

- **简介: 论文提出MasqLoRA，利用LoRA模块对文本到图像模型进行后门攻击，解决模型安全问题。通过少量样本训练，实现隐蔽攻击，攻击成功率高。**

- **链接: [https://arxiv.org/pdf/2602.21977v1](https://arxiv.org/pdf/2602.21977v1)**

> **作者:** Liangwei Lyu; Jiaqi Xu; Jianwei Ding; Qiyao Deng
>
> **摘要:** Low-Rank Adaptation (LoRA) has emerged as a leading technique for efficiently fine-tuning text-to-image diffusion models, and its widespread adoption on open-source platforms has fostered a vibrant culture of model sharing and customization. However, the same modular and plug-and-play flexibility that makes LoRA appealing also introduces a broader attack surface. To highlight this risk, we propose Masquerade-LoRA (MasqLoRA), the first systematic attack framework that leverages an independent LoRA module as the attack vehicle to stealthily inject malicious behavior into text-to-image diffusion models. MasqLoRA operates by freezing the base model parameters and updating only the low-rank adapter weights using a small number of "trigger word-target image" pairs. This enables the attacker to train a standalone backdoor LoRA module that embeds a hidden cross-modal mapping: when the module is loaded and a specific textual trigger is provided, the model produces a predefined visual output; otherwise, it behaves indistinguishably from the benign model, ensuring the stealthiness of the attack. Experimental results demonstrate that MasqLoRA can be trained with minimal resource overhead and achieves a high attack success rate of 99.8%. MasqLoRA reveals a severe and unique threat in the AI supply chain, underscoring the urgent need for dedicated defense mechanisms for the LoRA-centric sharing ecosystem.
>
---
#### [new 061] SAPNet++: Evolving Point-Prompted Instance Segmentation with Semantic and Spatial Awareness
- **分类: cs.CV**

- **简介: 该论文属于点提示实例分割任务，解决单点标注带来的细粒度模糊和边界不确定问题。提出SAPNet++，融合语义与空间感知模块提升分割精度。**

- **链接: [https://arxiv.org/pdf/2602.21762v1](https://arxiv.org/pdf/2602.21762v1)**

> **作者:** Zhaoyang Wei; Xumeng Han; Xuehui Yu; Xue Yang; Guorong Li; Zhenjun Han; Jianbin Jiao
>
> **备注:** 18 pages
>
> **摘要:** Single-point annotation is increasingly prominent in visual tasks for labeling cost reduction. However, it challenges tasks requiring high precision, such as the point-prompted instance segmentation (PPIS) task, which aims to estimate precise masks using single-point prompts to train a segmentation network. Due to the constraints of point annotations, granularity ambiguity and boundary uncertainty arise the difficulty distinguishing between different levels of detail (eg. whole object vs. parts) and the challenge of precisely delineating object boundaries. Previous works have usually inherited the paradigm of mask generation along with proposal selection to achieve PPIS. However, proposal selection relies solely on category information, failing to resolve the ambiguity of different granularity. Furthermore, mask generators offer only finite discrete solutions that often deviate from actual masks, particularly at boundaries. To address these issues, we propose the Semantic-Aware Point-Prompted Instance Segmentation Network (SAPNet). It integrates Point Distance Guidance and Box Mining Strategy to tackle group and local issues caused by the point's granularity ambiguity. Additionally, we incorporate completeness scores within proposals to add spatial granularity awareness, enhancing multiple instance learning (MIL) in proposal selection termed S-MIL. The Multi-level Affinity Refinement conveys pixel and semantic clues, narrowing boundary uncertainty during mask refinement. These modules culminate in SAPNet++, mitigating point prompt's granularity ambiguity and boundary uncertainty and significantly improving segmentation performance. Extensive experiments on four challenging datasets validate the effectiveness of our methods, highlighting the potential to advance PPIS.
>
---
#### [new 062] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context
- **分类: cs.CV**

- **简介: 该论文属于场景一致性视频生成任务，旨在解决传统方法因误差累积和非微分过程导致的不一致问题。提出“几何作为上下文”机制，结合相机控制与多任务框架提升生成效果。**

- **链接: [https://arxiv.org/pdf/2602.21929v1](https://arxiv.org/pdf/2602.21929v1)**

> **作者:** JiaKui Hu; Jialun Liu; Liying Yang; Xinliang Zhang; Kaiwen Li; Shuang Zeng; Yuanwei Li; Haibin Huang; Chi Zhang; Yanye Lu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Scene-consistent video generation aims to create videos that explore 3D scenes based on a camera trajectory. Previous methods rely on video generation models with external memory for consistency, or iterative 3D reconstruction and inpainting, which accumulate errors during inference due to incorrect intermediary outputs, non-differentiable processes, and separate models. To overcome these limitations, we introduce ``geometry-as-context". It iteratively completes the following steps using an autoregressive camera-controlled video generation model: (1) estimates the geometry of the current view necessary for 3D reconstruction, and (2) simulates and restores novel view images rendered by the 3D scene. Under this multi-task framework, we develop the camera gated attention module to enhance the model's capability to effectively leverage camera poses. During the training phase, text contexts are utilized to ascertain whether geometric or RGB images should be generated. To ensure that the model can generate RGB-only outputs during inference, the geometry context is randomly dropped from the interleaved text-image-geometry training sequence. The method has been tested on scene video generation with one-direction and forth-and-back trajectories. The results show its superiority over previous approaches in maintaining scene consistency and camera control.
>
---
#### [new 063] Synergizing Understanding and Generation with Interleaved Analyzing-Drafting Thinking
- **分类: cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决统一视觉语言模型中理解与生成能力协同不足的问题。提出AD-Loop机制，通过交替分析与草稿过程实现两者有效融合。**

- **链接: [https://arxiv.org/pdf/2602.21435v1](https://arxiv.org/pdf/2602.21435v1)**

> **作者:** Shengqiong Wu; Bobo Li; Xinkai Wang; Xiangtai Li; Lei Cui; Furu Wei; Shuicheng Yan; Hao Fei; Tat-seng Chua
>
> **备注:** 28 pages, 17 figures, 6 tables, ICLR conference
>
> **摘要:** Unified Vision-Language Models (UVLMs) aim to advance multimodal learning by supporting both understanding and generation within a single framework. However, existing approaches largely focus on architectural unification while overlooking the need for explicit interaction between the two capabilities during task solving. As a result, current models treat understanding and generation as parallel skills rather than synergistic processes. To achieve real synergy, we introduce the interleaved Analyzing-Drafting problem-solving loop (AD-Loop), a new think paradigm that dynamically alternates between analytic and drafting operations. By interleaving textual thoughts with visual thoughts, AD-Loop enables models to iteratively refine both comprehension and outputs, fostering genuine synergy. To train this mechanism, we design a two-stage strategy: supervised learning on interleaved thought data to initialize alternation, followed by reinforcement learning to promote adaptive and autonomous control. Extensive experiments demonstrate that AD-Loop consistently improves performance across standard benchmarks for both understanding and generation, with strong transferability to various UVLMs architectures. Visual analyses further validate the effectiveness of implicit visual thoughts. These results highlight AD-Loop as a principled and broadly applicable strategy for synergizing comprehension and creation. The project page is at https://sqwu.top/AD-Loop.
>
---
#### [new 064] NoLan: Mitigating Object Hallucinations in Large Vision-Language Models via Dynamic Suppression of Language Priors
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决对象幻觉问题。通过实验发现语言解码器的先验知识是主要原因，并提出NoLan框架动态抑制语言先验以减少幻觉。**

- **链接: [https://arxiv.org/pdf/2602.22144v1](https://arxiv.org/pdf/2602.22144v1)**

> **作者:** Lingfeng Ren; Weihao Yu; Runpeng Yu; Xinchao Wang
>
> **备注:** Code: https://github.com/lingfengren/NoLan
>
> **摘要:** Object hallucination is a critical issue in Large Vision-Language Models (LVLMs), where outputs include objects that do not appear in the input image. A natural question arises from this phenomenon: Which component of the LVLM pipeline primarily contributes to object hallucinations? The vision encoder to perceive visual information, or the language decoder to generate text responses? In this work, we strive to answer this question through designing a systematic experiment to analyze the roles of the vision encoder and the language decoder in hallucination generation. Our observations reveal that object hallucinations are predominantly associated with the strong priors from the language decoder. Based on this finding, we propose a simple and training-free framework, No-Language-Hallucination Decoding, NoLan, which refines the output distribution by dynamically suppressing language priors, modulated based on the output distribution difference between multimodal and text-only inputs. Experimental results demonstrate that NoLan effectively reduces object hallucinations across various LVLMs on different tasks. For instance, NoLan achieves substantial improvements on POPE, enhancing the accuracy of LLaVA-1.5 7B and Qwen-VL 7B by up to 6.45 and 7.21, respectively. The code is publicly available at: https://github.com/lingfengren/NoLan.
>
---
#### [new 065] CASR: A Robust Cyclic Framework for Arbitrary Large-Scale Super-Resolution with Distribution Alignment and Self-Similarity Awareness
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，解决跨尺度分布偏移问题。提出CASR框架，通过分布对齐和自相似性感知，实现任意尺度的稳定超分。**

- **链接: [https://arxiv.org/pdf/2602.22159v1](https://arxiv.org/pdf/2602.22159v1)**

> **作者:** Wenhao Guo; Zhaoran Zhao; Peng Lu; Sheng Li; Qian Qiao; RuiDe Li
>
> **摘要:** Arbitrary-Scale SR (ASISR) remains fundamentally limited by cross-scale distribution shift: once the inference scale leaves the training range, noise, blur, and artifacts accumulate sharply. We revisit this challenge from a cross-scale distribution transition perspective and propose CASR, a simple yet highly efficient cyclic SR framework that reformulates ultra-magnification as a sequence of in-distribution scale transitions. This design ensures stable inference at arbitrary scales while requiring only a single model. CASR tackles two major bottlenecks: distribution drift across iterations and patch-wise diffusion inconsistencies. The proposed SDAM module aligns structural distributions via superpixel aggregation, preventing error accumulation, while SARM module restores high-frequency textures by enforcing autocorrelation and embedding LR self-similarity priors. Despite using only a single model, our approach significantly reduces distribution drift, preserves long-range texture consistency, and achieves superior generalization even at extreme magnification.
>
---
#### [new 066] How to Take a Memorable Picture? Empowering Users with Actionable Feedback
- **分类: cs.CV**

- **简介: 该论文提出MemFeed任务，解决图像可记忆性提升问题。通过MemCoach模型提供自然语言反馈，增强用户拍摄记忆点，属于图像记忆增强领域。**

- **链接: [https://arxiv.org/pdf/2602.21877v1](https://arxiv.org/pdf/2602.21877v1)**

> **作者:** Francesco Laiti; Davide Talon; Jacopo Staiano; Elisa Ricci
>
> **备注:** Accepted @ CVPR 2026. Project page: https://laitifranz.github.io/MemCoach/
>
> **摘要:** Image memorability, i.e., how likely an image is to be remembered, has traditionally been studied in computer vision either as a passive prediction task, with models regressing a scalar score, or with generative methods altering the visual input to boost the image likelihood of being remembered. Yet, none of these paradigms supports users at capture time, when the crucial question is how to improve a photo memorability. We introduce the task of Memorability Feedback (MemFeed), where an automated model should provide actionable, human-interpretable guidance to users with the goal to enhance an image future recall. We also present MemCoach, the first approach designed to provide concrete suggestions in natural language for memorability improvement (e.g., "emphasize facial expression," "bring the subject forward"). Our method, based on Multimodal Large Language Models (MLLMs), is training-free and employs a teacher-student steering strategy, aligning the model internal activations toward more memorable patterns learned from a teacher model progressing along least-to-most memorable samples. To enable systematic evaluation on this novel task, we further introduce MemBench, a new benchmark featuring sequence-aligned photoshoots with annotated memorability scores. Our experiments, considering multiple MLLMs, demonstrate the effectiveness of MemCoach, showing consistently improved performance over several zero-shot models. The results indicate that memorability can not only be predicted but also taught and instructed, shifting the focus from mere prediction to actionable feedback for human creators.
>
---
#### [new 067] Send Less, Perceive More: Masked Quantized Point Cloud Communication for Loss-Tolerant Collaborative Perception
- **分类: cs.CV**

- **简介: 该论文属于协同感知任务，旨在解决带宽受限和数据包丢失问题。提出QPoint2Comm框架，通过量化点云索引实现高效通信与鲁棒感知。**

- **链接: [https://arxiv.org/pdf/2602.21667v1](https://arxiv.org/pdf/2602.21667v1)**

> **作者:** Sheng Xu; Enshu Wang; Hongfei Xue; Jian Teng; Bingyi Liu; Yi Zhu; Pu Wang; Libing Wu; Chunming Qiao
>
> **摘要:** Collaborative perception allows connected vehicles to overcome occlusions and limited viewpoints by sharing sensory information. However, existing approaches struggle to achieve high accuracy under strict bandwidth constraints and remain highly vulnerable to random transmission packet loss. We introduce QPoint2Comm, a quantized point-cloud communication framework that dramatically reduces bandwidth while preserving high-fidelity 3D information. Instead of transmitting intermediate features, QPoint2Comm directly communicates quantized point-cloud indices using a shared codebook, enabling efficient reconstruction with lower bandwidth than feature-based methods. To ensure robustness to possible communication packet loss, we employ a masked training strategy that simulates random packet loss, allowing the model to maintain strong performance even under severe transmission failures. In addition, a cascade attention fusion module is proposed to enhance multi-vehicle information integration. Extensive experiments on both simulated and real-world datasets demonstrate that QPoint2Comm sets a new state of the art in accuracy, communication efficiency, and resilience to packet loss.
>
---
#### [new 068] DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs
- **分类: cs.CV; cs.AI; cs.CL; cs.GR**

- **简介: 该论文属于视觉-语言模型在图问答任务中的研究，旨在解决VLM对结构化图理解不足的问题。提出DynamicGTR框架，动态选择最优图拓扑表示，提升问答准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.21864v1](https://arxiv.org/pdf/2602.21864v1)**

> **作者:** Yanbin Wei; Jiangyue Yan; Chun Kang; Yang Chen; Hua Liu; James Kwok; Yu Zhang
>
> **备注:** CVPR 2026
>
> **摘要:** Vision-Language Models (VLMs) have emerged as versatile solutions for zero-shot question answering (QA) across various domains. However, enabling VLMs to effectively comprehend structured graphs and perform accurate, efficient QA remains challenging. Existing approaches typically rely on one single graph topology representation (GTR), such as fixed-style visual images or unified text descriptions. This ``one-size-fits-all'' strategy often neglects model-specific and task-specific preferences, resulting in inaccurate or over-lengthy responses to graph-related queries. To address this, we propose the $\mbox{DynamicGTR}$ framework, which dynamically selects the optimal GTR for each query during inference, thereby enhancing the zero-shot graph QA capabilities of VLMs with a customizable accuracy and brevity trade-off. Extensive experiments show that DynamicGTR not only improves VLM-based graph algorithm QA performance but also successfully transfers the experience trained from synthetic graph algorithm tasks to real-world applications like link prediction and node classification, without any additional training. Additionally, DynamicGTR demonstrates strong transferability across tasks, domains, and models, suggesting its potential as a flexible solution for broad graph scenarios.
>
---
#### [new 069] Understanding Annotation Error Propagation and Learning an Adaptive Policy for Expert Intervention in Barrett's Video Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视频分割任务，解决标注误差传播问题。通过研究不同提示类型下的误差传播，提出L2RP框架，优化专家干预策略，提升分割准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.21855v1](https://arxiv.org/pdf/2602.21855v1)**

> **作者:** Lokesha Rasanjalee; Jin Lin Tan; Dileepa Pitawela; Rajvinder Singh; Hsiang-Ting Chen
>
> **备注:** Accepted at IEEE ISBI 2026
>
> **摘要:** Accurate annotation of endoscopic videos is essential yet time-consuming, particularly for challenging datasets such as dysplasia in Barrett's esophagus, where the affected regions are irregular and lack clear boundaries. Semi-automatic tools like Segment Anything Model 2 (SAM2) can ease this process by propagating annotations across frames, but small errors often accumulate and reduce accuracy, requiring expert review and correction. To address this, we systematically study how annotation errors propagate across different prompt types, namely masks, boxes, and points, and propose Learning-to-Re-Prompt (L2RP), a cost-aware framework that learns when and where to seek expert input. By tuning a human-cost parameter, our method balances annotation effort and segmentation accuracy. Experiments on a private Barrett's dysplasia dataset and the public SUN-SEG benchmark demonstrate improved temporal consistency and superior performance over baseline strategies.
>
---
#### [new 070] MultiAnimate: Pose-Guided Image Animation Made Extensible
- **分类: cs.CV**

- **简介: 该论文属于多角色图像动画任务，解决多角色身份混淆和遮挡问题。提出包含标识分配器和适配器的框架，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21581v1](https://arxiv.org/pdf/2602.21581v1)**

> **作者:** Yingcheng Hu; Haowen Gong; Chuanguang Yang; Zhulin An; Yongjun Xu; Songhua Liu
>
> **备注:** Project page at https://hyc001.github.io/MultiAnimate/
>
> **摘要:** Pose-guided human image animation aims to synthesize realistic videos of a reference character driven by a sequence of poses. While diffusion-based methods have achieved remarkable success, most existing approaches are limited to single-character animation. We observe that naively extending these methods to multi-character scenarios often leads to identity confusion and implausible occlusions between characters. To address these challenges, in this paper, we propose an extensible multi-character image animation framework built upon modern Diffusion Transformers (DiTs) for video generation. At its core, our framework introduces two novel components-Identifier Assigner and Identifier Adapter - which collaboratively capture per-person positional cues and inter-person spatial relationships. This mask-driven scheme, along with a scalable training strategy, not only enhances flexibility but also enables generalization to scenarios with more characters than those seen during training. Remarkably, trained on only a two-character dataset, our model generalizes to multi-character animation while maintaining compatibility with single-character cases. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in multi-character image animation, surpassing existing diffusion-based baselines.
>
---
#### [new 071] Innovative Tooth Segmentation Using Hierarchical Features and Bidirectional Sequence Modeling
- **分类: cs.CV**

- **简介: 该论文属于牙科图像分割任务，旨在解决传统方法分割不连续和计算效率低的问题。通过引入分层特征和双向序列建模，提升分割精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.21712v1](https://arxiv.org/pdf/2602.21712v1)**

> **作者:** Xinxin Zhao; Jian Jiang; Yan Tian; Liqin Wu; Zhaocheng Xu; Teddy Yang; Yunuo Zou; Xun Wang
>
> **备注:** Accepted by Pattern Recognition
>
> **摘要:** Tooth image segmentation is a cornerstone of dental digitization. However, traditional image encoders relying on fixed-resolution feature maps often lead to discontinuous segmentation and poor discrimination between target regions and background, due to insufficient modeling of environmental and global context. Moreover, transformer-based self-attention introduces substantial computational overhead because of its quadratic complexity (O(n^2)), making it inefficient for high-resolution dental images. To address these challenges, we introduce a three-stage encoder with hierarchical feature representation to capture scale-adaptive information in dental images. By jointly leveraging low-level details and high-level semantics through cross-scale feature fusion, the model effectively preserves fine structural information while maintaining strong contextual awareness. Furthermore, a bidirectional sequence modeling strategy is incorporated to enhance global spatial context understanding without incurring high computational cost. We validate our method on two dental datasets, with experimental results demonstrating its superiority over existing approaches. On the OralVision dataset, our model achieves a 1.1% improvement in mean intersection over union (mIoU).
>
---
#### [new 072] TranX-Adapter: Bridging Artifacts and Semantics within MLLMs for Robust AI-generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决MLLM中语义与纹理特征融合不畅的问题。提出TranX-Adapter，通过优化融合机制提升检测准确率。**

- **链接: [https://arxiv.org/pdf/2602.21716v1](https://arxiv.org/pdf/2602.21716v1)**

> **作者:** Wenbin Wang; Yuge Huang; Jianqing Xu; Yue Yu; Jiangtao Yan; Shouhong Ding; Pan Zhou; Yong Luo
>
> **摘要:** Rapid advances in AI-generated image (AIGI) technology enable highly realistic synthesis, threatening public information integrity and security. Recent studies have demonstrated that incorporating texture-level artifact features alongside semantic features into multimodal large language models (MLLMs) can enhance their AIGI detection capability. However, our preliminary analyses reveal that artifact features exhibit high intra-feature similarity, leading to an almost uniform attention map after the softmax operation. This phenomenon causes attention dilution, thereby hindering effective fusion between semantic and artifact features. To overcome this limitation, we propose a lightweight fusion adapter, TranX-Adapter, which integrates a Task-aware Optimal-Transport Fusion that leverages the Jensen-Shannon divergence between artifact and semantic prediction probabilities as a cost matrix to transfer artifact information into semantic features, and an X-Fusion that employs cross-attention to transfer semantic information into artifact features. Experiments on standard AIGI detection benchmarks upon several advanced MLLMs, show that our TranX-Adapter brings consistent and significant improvements (up to +6% accuracy).
>
---
#### [new 073] Scaling View Synthesis Transformers
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于新颖视角合成任务，研究视图合成Transformer的扩展规律，提出计算最优的编码器-解码器架构SVSM，提升性能与计算效率。**

- **链接: [https://arxiv.org/pdf/2602.21341v1](https://arxiv.org/pdf/2602.21341v1)**

> **作者:** Evan Kim; Hyunwoo Ryu; Thomas W. Mitchel; Vincent Sitzmann
>
> **备注:** Project page: https://www.evn.kim/research/svsm
>
> **摘要:** Geometry-free view synthesis transformers have recently achieved state-of-the-art performance in Novel View Synthesis (NVS), outperforming traditional approaches that rely on explicit geometry modeling. Yet the factors governing their scaling with compute remain unclear. We present a systematic study of scaling laws for view synthesis transformers and derive design principles for training compute-optimal NVS models. Contrary to prior findings, we show that encoder-decoder architectures can be compute-optimal; we trace earlier negative results to suboptimal architectural choices and comparisons across unequal training compute budgets. Across several compute levels, we demonstrate that our encoder-decoder architecture, which we call the Scalable View Synthesis Model (SVSM), scales as effectively as decoder-only models, achieves a superior performance-compute Pareto frontier, and surpasses the previous state-of-the-art on real-world NVS benchmarks with substantially reduced training compute.
>
---
#### [new 074] ECHOSAT: Estimating Canopy Height Over Space And Time
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ECHOSAT，解决森林树高动态监测问题，利用多源卫星数据和视觉Transformer模型，生成高分辨率、时间一致的全球树高地图。**

- **链接: [https://arxiv.org/pdf/2602.21421v1](https://arxiv.org/pdf/2602.21421v1)**

> **作者:** Jan Pauls; Karsten Schrödter; Sven Ligensa; Martin Schwartz; Berkant Turan; Max Zimmer; Sassan Saatchi; Sebastian Pokutta; Philippe Ciais; Fabian Gieseke
>
> **备注:** 19 pages, 12 figures, 6 tables
>
> **摘要:** Forest monitoring is critical for climate change mitigation. However, existing global tree height maps provide only static snapshots and do not capture temporal forest dynamics, which are essential for accurate carbon accounting. We introduce ECHOSAT, a global and temporally consistent tree height map at 10 m resolution spanning multiple years. To this end, we resort to multi-sensor satellite data to train a specialized vision transformer model, which performs pixel-level temporal regression. A self-supervised growth loss regularizes the predictions to follow growth curves that are in line with natural tree development, including gradual height increases over time, but also abrupt declines due to forest loss events such as fires. Our experimental evaluation shows that our model improves state-of-the-art accuracies in the context of single-year predictions. We also provide the first global-scale height map that accurately quantifies tree growth and disturbances over time. We expect ECHOSAT to advance global efforts in carbon monitoring and disturbance assessment. The maps can be accessed at https://github.com/ai4forest/echosat.
>
---
#### [new 075] Unified Unsupervised and Sparsely-Supervised 3D Object Detection by Semantic Pseudo-Labeling and Prototype Learning
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决依赖大量人工标注数据的问题。通过语义伪标签和原型学习，提出SPL框架，在无监督和稀疏监督下提升检测性能。**

- **链接: [https://arxiv.org/pdf/2602.21484v1](https://arxiv.org/pdf/2602.21484v1)**

> **作者:** Yushen He
>
> **摘要:** 3D object detection is essential for autonomous driving and robotic perception, yet its reliance on large-scale manually annotated data limits scalability and adaptability. To reduce annotation dependency, unsupervised and sparsely-supervised paradigms have emerged. However, they face intertwined challenges: low-quality pseudo-labels, unstable feature mining, and a lack of a unified training framework. This paper proposes SPL, a unified training framework for both Unsupervised and Sparsely-Supervised 3D Object Detection via Semantic Pseudo-labeling and prototype Learning. SPL first generates high-quality pseudo-labels by integrating image semantics, point cloud geometry, and temporal cues, producing both 3D bounding boxes for dense objects and 3D point labels for sparse ones. These pseudo-labels are not used directly but as probabilistic priors within a novel, multi-stage prototype learning strategy. This strategy stabilizes feature representation learning through memory-based initialization and momentum-based prototype updating, effectively mining features from both labeled and unlabeled data. Extensive experiments on KITTI and nuScenes datasets demonstrate that SPL significantly outperforms state-of-the-art methods in both settings. Our work provides a robust and generalizable solution for learning 3D object detectors with minimal or no manual annotations.
>
---
#### [new 076] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration
- **分类: cs.CV**

- **简介: 该论文属于多传感器标定任务，旨在解决无目标场景下的高精度传感器对齐问题。提出LiREC-Net，实现LiDAR、RGB和事件数据的联合标定。**

- **链接: [https://arxiv.org/pdf/2602.21754v1](https://arxiv.org/pdf/2602.21754v1)**

> **作者:** Aditya Ranjan Dash; Ramy Battrawy; René Schuster; Didier Stricker
>
> **备注:** Accepted in CVPR 2026
>
> **摘要:** Advanced autonomous systems rely on multi-sensor fusion for safer and more robust perception. To enable effective fusion, calibrating directly from natural driving scenes (i.e., target-free) with high accuracy is crucial for precise multi-sensor alignment. Existing learning-based calibration methods are typically designed for only a single pair of sensor modalities (i.e., a bi-modal setup). Unlike these methods, we propose LiREC-Net, a target-free, learning-based calibration network that jointly calibrates multiple sensor modality pairs, including LiDAR, RGB, and event data, within a unified framework. To reduce redundant computation and improve efficiency, we introduce a shared LiDAR representation that leverages features from both its 3D nature and projected depth map, ensuring better consistency across modalities. Trained and evaluated on established datasets, such as KITTI and DSEC, our LiREC-Net achieves competitive performance to bi-modal models and sets a new strong baseline for the tri-modal use case.
>
---
#### [new 077] Global-Aware Edge Prioritization for Pose Graph Initialization
- **分类: cs.CV**

- **简介: 该论文属于SfM任务，解决 pose graph 初始化问题。通过全局感知的边优先级方法，提升图的可靠性与紧凑性，提高重建精度。**

- **链接: [https://arxiv.org/pdf/2602.21963v1](https://arxiv.org/pdf/2602.21963v1)**

> **作者:** Tong Wei; Giorgos Tolias; Jiri Matas; Daniel Barath
>
> **备注:** accepted to CVPR 2026
>
> **摘要:** The pose graph is a core component of Structure-from-Motion (SfM), where images act as nodes and edges encode relative poses. Since geometric verification is expensive, SfM pipelines restrict the pose graph to a sparse set of candidate edges, making initialization critical. Existing methods rely on image retrieval to connect each image to its $k$ nearest neighbors, treating pairs independently and ignoring global consistency. We address this limitation through the concept of edge prioritization, ranking candidate edges by their utility for SfM. Our approach has three components: (1) a GNN trained with SfM-derived supervision to predict globally consistent edge reliability; (2) multi-minimal-spanning-tree-based pose graph construction guided by these ranks; and (3) connectivity-aware score modulation that reinforces weak regions and reduces graph diameter. This globally informed initialization yields more reliable and compact pose graphs, improving reconstruction accuracy in sparse and high-speed settings and outperforming SOTA retrieval methods on ambiguous scenes. The ode and trained models are available at https://github.com/weitong8591/global_edge_prior.
>
---
#### [new 078] Neu-PiG: Neural Preconditioned Grids for Fast Dynamic Surface Reconstruction on Long Sequences
- **分类: cs.CV**

- **简介: 该论文提出Neu-PiG，用于快速动态表面重建任务，解决长序列中一致性差和计算效率低的问题，通过预条件隐式网格编码实现高效变形优化。**

- **链接: [https://arxiv.org/pdf/2602.22212v1](https://arxiv.org/pdf/2602.22212v1)**

> **作者:** Julian Kaltheuner; Hannah Dröge; Markus Plack; Patrick Stotko; Reinhard Klein
>
> **备注:** CVPR 2026, Code: https://github.com/vc-bonn/neu-pig
>
> **摘要:** Temporally consistent surface reconstruction of dynamic 3D objects from unstructured point cloud data remains challenging, especially for very long sequences. Existing methods either optimize deformations incrementally, risking drift and requiring long runtimes, or rely on complex learned models that demand category-specific training. We present Neu-PiG, a fast deformation optimization method based on a novel preconditioned latent-grid encoding that distributes spatial features parameterized on the position and normal direction of a keyframe surface. Our method encodes entire deformations across all time steps at various spatial scales into a multi-resolution latent grid, parameterized by the position and normal direction of a reference surface from a single keyframe. This latent representation is then augmented for time modulation and decoded into per-frame 6-DoF deformations via a lightweight multilayer perceptron (MLP). To achieve high-fidelity, drift-free surface reconstructions in seconds, we employ Sobolev preconditioning during gradient-based training of the latent space, completely avoiding the need for any explicit correspondences or further priors. Experiments across diverse human and animal datasets demonstrate that Neu-PiG outperforms state-the-art approaches, offering both superior accuracy and scalability to long sequences while running at least 60x faster than existing training-free methods and achieving inference speeds on the same order as heavy pretrained models.
>
---
#### [new 079] Lie Flow: Video Dynamic Fields Modeling and Predicting with Lie Algebra as Geometric Physics Principle
- **分类: cs.CV**

- **简介: 该论文属于4D场景建模任务，解决动态场景中物理一致运动表示的问题。通过SE(3)李群建模，统一处理平移与旋转，提升视图合成精度与物理合理性。**

- **链接: [https://arxiv.org/pdf/2602.21645v1](https://arxiv.org/pdf/2602.21645v1)**

> **作者:** Weidong Qiao; Wangmeng Zuo; Hui Li
>
> **备注:** 10pages,5 figures
>
> **摘要:** Modeling 4D scenes requires capturing both spatial structure and temporal motion, which is challenging due to the need for physically consistent representations of complex rigid and non-rigid motions. Existing approaches mainly rely on translational displacements, which struggle to represent rotations, articulated transformations, often leading to spatial inconsistency and physically implausible motion. LieFlow, a dynamic radiance representation framework that explicitly models motion within the SE(3) Lie group, enabling coherent learning of translation and rotation in a unified geometric space. The SE(3) transformation field enforces physically inspired constraints to maintain motion continuity and geometric consistency. The evaluation includes a synthetic dataset with rigid-body trajectories and two real-world datasets capturing complex motion under natural lighting and occlusions. Across all datasets, LieFlow consistently improves view-synthesis fidelity, temporal coherence, and physical realism over NeRF-based baselines. These results confirm that SE(3)-based motion modeling offers a robust and physically grounded framework for representing dynamic 4D scenes.
>
---
#### [new 080] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出NESTOR模型，解决神经算子在大规模PDE预训练中的泛化与迁移问题，通过嵌套MoE框架提升对复杂系统依赖的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2602.22059v1](https://arxiv.org/pdf/2602.22059v1)**

> **作者:** Dengdi Sun; Xiaoya Zhou; Xiao Wang; Hao Si; Wanli Lyu; Jin Tang; Bin Luo
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Neural operators have emerged as an efficient paradigm for solving PDEs, overcoming the limitations of traditional numerical methods and significantly improving computational efficiency. However, due to the diversity and complexity of PDE systems, existing neural operators typically rely on a single network architecture, which limits their capacity to fully capture heterogeneous features and complex system dependencies. This constraint poses a bottleneck for large-scale PDE pre-training based on neural operators. To address these challenges, we propose a large-scale PDE pre-trained neural operator based on a nested Mixture-of-Experts (MoE) framework. In particular, the image-level MoE is designed to capture global dependencies, while the token-level Sub-MoE focuses on local dependencies. Our model can selectively activate the most suitable expert networks for a given input, thereby enhancing generalization and transferability. We conduct large-scale pre-training on twelve PDE datasets from diverse sources and successfully transfer the model to downstream tasks. Extensive experiments demonstrate the effectiveness of our approach.
>
---
#### [new 081] PSF-Med: Measuring and Explaining Paraphrase Sensitivity in Medical Vision Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医疗视觉语言模型研究，解决模型对问题重述敏感的问题。通过构建基准数据集，分析模型稳定性与图像依赖性，提出改进方法降低翻转率。**

- **链接: [https://arxiv.org/pdf/2602.21428v1](https://arxiv.org/pdf/2602.21428v1)**

> **作者:** Binesh Sadanandan; Vahid Behzadan
>
> **摘要:** Medical Vision Language Models (VLMs) can change their answers when clinicians rephrase the same question, which raises deployment risks. We introduce Paraphrase Sensitivity Failure (PSF)-Med, a benchmark of 19,748 chest Xray questions paired with about 92,000 meaningpreserving paraphrases across MIMIC-CXR and PadChest. Across six medical VLMs, we measure yes/no flips for the same image and find flip rates from 8% to 58%. However, low flip rate does not imply visual grounding: text-only baselines show that some models stay consistent even when the image is removed, suggesting they rely on language priors. To study mechanisms in one model, we apply GemmaScope 2 Sparse Autoencoders (SAEs) to MedGemma 4B and analyze FlipBank, a curated set of 158 flip cases. We identify a sparse feature at layer 17 that correlates with prompt framing and predicts decision margin shifts. In causal patching, removing this feature's contribution recovers 45% of the yesminus-no logit margin on average and fully reverses 15% of flips. Acting on this finding, we show that clamping the identified feature at inference reduces flip rates by 31% relative with only a 1.3 percentage-point accuracy cost, while also decreasing text-prior reliance. These results suggest that flip rate alone is not enough; robustness evaluations should test both paraphrase stability and image reliance.
>
---
#### [new 082] GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry
- **分类: cs.CV**

- **简介: 该论文属于运动分割任务，旨在解决动态场景中运动目标分割的问题。通过学习潜在4D几何特征，直接推理运动物体，避免复杂预处理和迭代优化，提升分割效率与精度。**

- **链接: [https://arxiv.org/pdf/2602.21810v1](https://arxiv.org/pdf/2602.21810v1)**

> **作者:** Xiankang He; Peile Lin; Ying Cui; Dongyan Guo; Chunhua Shen; Xiaoqin Zhang
>
> **摘要:** Motion segmentation in dynamic scenes is highly challenging, as conventional methods heavily rely on estimating camera poses and point correspondences from inherently noisy motion cues. Existing statistical inference or iterative optimization techniques that struggle to mitigate the cumulative errors in multi-stage pipelines often lead to limited performance or high computational cost. In contrast, we propose a fully learning-based approach that directly infers moving objects from latent feature representations via attention mechanisms, thus enabling end-to-end feed-forward motion segmentation. Our key insight is to bypass explicit correspondence estimation and instead let the model learn to implicitly disentangle object and camera motion. Supported by recent advances in 4D scene geometry reconstruction (e.g., $π^3$), the proposed method leverages reliable camera poses and rich spatial-temporal priors, which ensure stable training and robust inference for the model. Extensive experiments demonstrate that by eliminating complex pre-processing and iterative refinement, our approach achieves state-of-the-art motion segmentation performance with high efficiency. The code is available at:https://github.com/zjutcvg/GeoMotion.
>
---
#### [new 083] Tokenizing Semantic Segmentation with RLE
- **分类: cs.CV**

- **简介: 该论文属于图像和视频语义分割任务，通过语言模型生成RLE编码的分割掩码，解决分割序列过长的问题，并引入实例信息实现全景分割。**

- **链接: [https://arxiv.org/pdf/2602.21627v1](https://arxiv.org/pdf/2602.21627v1)**

> **作者:** Abhineet Singh; Justin Rozeboom; Nilanjan Ray
>
> **摘要:** This paper presents a new unified approach to semantic segmentation in both images and videos by using language modeling to output the masks as sequences of discrete tokens. We use run length encoding (RLE) to discretize the segmentation masks and then train a modified version of Pix2Seq \cite{p2s} to output these RLE tokens through autoregression. We propose novel tokenization strategies to compress the length of the token sequence to make it practicable to extend this approach to videos. We also show how instance information can be incorporated into the tokenization process to perform panoptic segmentation. We evaluate our proposed models on two datasets to show that they are competitive with the state of the art in spite of being bottlenecked by our limited computational resources.
>
---
#### [new 084] Enhancing Multi-Modal LLMs Reasoning via Difficulty-Aware Group Normalization
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型推理任务，解决std归一化在极端样本下的不稳定性问题。通过引入难度感知的分组归一化方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.21743v1](https://arxiv.org/pdf/2602.21743v1)**

> **作者:** Jinghan Li; Junfeng Fang; Jinda Lu; Yuan Wang; Xiaoyan Guo; Tianyu Zhang; Xiang Wang; Xiangnan He
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) and Group Relative Policy Optimization (GRPO) have significantly advanced the reasoning capabilities of large language models. Extending these methods to multimodal settings, however, faces a critical challenge: the instability of std-based normalization, which is easily distorted by extreme samples with nearly positive or negative rewards. Unlike pure-text LLMs, multimodal models are particularly sensitive to such distortions, as both perceptual and reasoning errors influence their responses. To address this, we characterize each sample by its difficulty, defined through perceptual complexity (measured via visual entropy) and reasoning uncertainty (captured by model confidence). Building on this characterization, we propose difficulty-aware group normalization (Durian), which re-groups samples by difficulty levels and shares the std within each group. Our approach preserves GRPO's intra-group distinctions while eliminating sensitivity to extreme cases, yielding significant performance gains across multiple multimodal reasoning benchmarks.
>
---
#### [new 085] UNet-Based Keypoint Regression for 3D Cone Localization in Autonomous Racing
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D锥体定位任务，旨在提升自动驾驶赛车中锥体的精确定位。通过UNet模型进行关键点检测，解决传统方法在环境变化下的敏感性问题，并实现实时高效定位。**

- **链接: [https://arxiv.org/pdf/2602.21904v1](https://arxiv.org/pdf/2602.21904v1)**

> **作者:** Mariia Baidachna; James Carty; Aidan Ferguson; Joseph Agrane; Varad Kulkarni; Aubrey Agub; Michael Baxendale; Aaron David; Rachel Horton; Elliott Atkinson
>
> **备注:** 8 pages, 9 figures. Accepted to ICCV End-to-End 3D Learning Workshop 2025 and presented as a poster; not included in the final proceedings due to a conference administrative error
>
> **摘要:** Accurate cone localization in 3D space is essential in autonomous racing for precise navigation around the track. Approaches that rely on traditional computer vision algorithms are sensitive to environmental variations, and neural networks are often trained on limited data and are infeasible to run in real time. We present a UNet-based neural network for keypoint detection on cones, leveraging the largest custom-labeled dataset we have assembled. Our approach enables accurate cone position estimation and the potential for color prediction. Our model achieves substantial improvements in keypoint accuracy over conventional methods. Furthermore, we leverage our predicted keypoints in the perception pipeline and evaluate the end-to-end autonomous system. Our results show high-quality performance across all metrics, highlighting the effectiveness of this approach and its potential for adoption in competitive autonomous racing systems.
>
---
#### [new 086] SPGen: Stochastic scanpath generation for paintings using unsupervised domain adaptation
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出SPGen模型，用于生成绘画作品的注视路径。属于视觉注意力预测任务，解决从照片到艺术品的领域差异问题，通过无监督域适应和随机噪声采样提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2602.22049v1](https://arxiv.org/pdf/2602.22049v1)**

> **作者:** Mohamed Amine Kerkouri; Marouane Tliba; Aladine Chetouani; Alessandro Bruno
>
> **备注:** Under Review
>
> **摘要:** Understanding human visual attention is key to preserving cultural heritage We introduce SPGen a novel deep learning model to predict scanpaths the sequence of eye movementswhen viewers observe paintings. Our architecture uses a Fully Convolutional Neural Network FCNN with differentiable fixation selection and learnable Gaussian priors to simulate natural viewing biases To address the domain gap between photographs and artworks we employ unsupervised domain adaptation via a gradient reversal layer allowing the model to transfer knowledge from natural scenes to paintings Furthermore a random noise sampler models the inherent stochasticity of eyetracking data. Extensive testing shows SPGen outperforms existing methods offering a powerful tool to analyze gaze behavior and advance the preservation and appreciation of artistic treasures.
>
---
#### [new 087] Off-The-Shelf Image-to-Image Models Are All You Need To Defeat Image Protection Schemes
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像安全领域，研究如何利用现成的图像生成模型破解图像保护机制。工作包括验证通用模型可有效去除多种防护扰动，揭示现有保护方案的脆弱性。**

- **链接: [https://arxiv.org/pdf/2602.22197v1](https://arxiv.org/pdf/2602.22197v1)**

> **作者:** Xavier Pleimling; Sifat Muhammad Abdullah; Gunjan Balde; Peng Gao; Mainack Mondal; Murtuza Jadliwala; Bimal Viswanath
>
> **备注:** This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). The final version will be available on IEEE Xplore. To IEEE SaTML 2026
>
> **摘要:** Advances in Generative AI (GenAI) have led to the development of various protection strategies to prevent the unauthorized use of images. These methods rely on adding imperceptible protective perturbations to images to thwart misuse such as style mimicry or deepfake manipulations. Although previous attacks on these protections required specialized, purpose-built methods, we demonstrate that this is no longer necessary. We show that off-the-shelf image-to-image GenAI models can be repurposed as generic ``denoisers" using a simple text prompt, effectively removing a wide range of protective perturbations. Across 8 case studies spanning 6 diverse protection schemes, our general-purpose attack not only circumvents these defenses but also outperforms existing specialized attacks while preserving the image's utility for the adversary. Our findings reveal a critical and widespread vulnerability in the current landscape of image protection, indicating that many schemes provide a false sense of security. We stress the urgent need to develop robust defenses and establish that any future protection mechanism must be benchmarked against attacks from off-the-shelf GenAI models. Code is available in this repository: https://github.com/mlsecviswanath/img2imgdenoiser
>
---
#### [new 088] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos
- **分类: cs.CV**

- **简介: 该论文属于自主驾驶任务，解决无标注视频中学习驾驶表示的问题。通过标签无关框架，从原始视频中联合预测点云、位姿等信息，提升驾驶感知性能。**

- **链接: [https://arxiv.org/pdf/2602.22091v1](https://arxiv.org/pdf/2602.22091v1)**

> **作者:** Matthew Strong; Wei-Jer Chang; Quentin Herau; Jiezhi Yang; Yihan Hu; Chensheng Peng; Wei Zhan
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Ego-centric driving videos available online provide an abundant source of visual data for autonomous driving, yet their lack of annotations makes it difficult to learn representations that capture both semantic structure and 3D geometry. Recent advances in large feedforward spatial models demonstrate that point maps and ego-motion can be inferred in a single forward pass, suggesting a promising direction for scalable driving perception. We therefore propose a label-free, teacher-guided framework for learning autonomous driving representations directly from unposed videos. Unlike prior self-supervised approaches that focus primarily on frame-to-frame consistency, we posit that safe and reactive driving depends critically on temporal context. To this end, we leverage a feedforward architecture equipped with a lightweight autoregressive module, trained using multi-modal supervisory signals that guide the model to jointly predict current and future point maps, camera poses, semantic segmentation, and motion masks. Multi-modal teachers provide sequence-level pseudo-supervision, enabling LFG to learn a unified pseudo-4D representation from raw YouTube videos without poses, labels, or LiDAR. The resulting encoder not only transfers effectively to downstream autonomous driving planning on the NAVSIM benchmark, surpassing multi-camera and LiDAR baselines with only a single monocular camera, but also yields strong performance when evaluated on a range of semantic, geometric, and qualitative motion prediction tasks. These geometry and motion-aware features position LFG as a compelling video-centric foundation model for autonomous driving.
>
---
#### [new 089] SEF-MAP: Subspace-Decomposed Expert Fusion for Robust Multimodal HD Map Prediction
- **分类: cs.CV**

- **简介: 该论文属于多模态高精度地图预测任务，解决相机与激光雷达数据不一致导致的性能下降问题，提出SEF-MAP框架，通过子空间融合提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.21589v1](https://arxiv.org/pdf/2602.21589v1)**

> **作者:** Haoxiang Fu; Lingfeng Zhang; Hao Li; Ruibing Hu; Zhengrong Li; Guanjing Liu; Zimu Tan; Long Chen; Hangjun Ye; Xiaoshuai Hao
>
> **摘要:** High-definition (HD) maps are essential for autonomous driving, yet multi-modal fusion often suffers from inconsistency between camera and LiDAR modalities, leading to performance degradation under low-light conditions, occlusions, or sparse point clouds. To address this, we propose SEFMAP, a Subspace-Expert Fusion framework for robust multimodal HD map prediction. The key idea is to explicitly disentangle BEV features into four semantic subspaces: LiDAR-private, Image-private, Shared, and Interaction. Each subspace is assigned a dedicated expert, thereby preserving modality-specific cues while capturing cross-modal consensus. To adaptively combine expert outputs, we introduce an uncertainty-aware gating mechanism at the BEV-cell level, where unreliable experts are down-weighted based on predictive variance, complemented by a usage balance regularizer to prevent expert collapse. To enhance robustness in degraded conditions and promote role specialization, we further propose distribution-aware masking: during training, modality-drop scenarios are simulated using EMA-statistical surrogate features, and a specialization loss enforces distinct behaviors of private, shared, and interaction experts across complete and masked inputs. Experiments on nuScenes and Argoverse2 benchmarks demonstrate that SEFMAP achieves state-of-the-art performance, surpassing prior methods by +4.2% and +4.8% in mAP, respectively. SEF-MAPprovides a robust and effective solution for multi-modal HD map prediction under diverse and degraded conditions.
>
---
#### [new 090] Brain Tumor Segmentation with Special Emphasis on the Non-Enhancing Brain Tumor Compartment
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于脑肿瘤分割任务，旨在解决非增强肿瘤区域的自动分割问题，以评估患者生存期和肿瘤生长潜力。**

- **链接: [https://arxiv.org/pdf/2602.21703v1](https://arxiv.org/pdf/2602.21703v1)**

> **作者:** T. Schaffer; A. Brawanski; S. Wein; A. M. Tomé; E. W. Lang
>
> **摘要:** A U-Net based deep learning architecture is designed to segment brain tumors as they appear on various MRI modalities. Special emphasis is lent to the non-enhancing tumor compartment. The latter has not been considered anymore in recent brain tumor segmentation challenges like the MICCAI challenges. However, it is considered to be indicative of the survival time of the patient as well as of areas of further tumor growth. Hence it deems essential to have means to automatically delineate its extension within the tumor.
>
---
#### [new 091] Mixed Magnification Aggregation for Generalizable Region-Level Representations in Computational Pathology
- **分类: cs.CV**

- **简介: 该论文属于计算病理学任务，旨在解决单一放大倍数导致的特征信息不足问题。通过混合放大倍数的区域聚合方法，提升多尺度特征表达，优化癌症生物标志物预测。**

- **链接: [https://arxiv.org/pdf/2602.22176v1](https://arxiv.org/pdf/2602.22176v1)**

> **作者:** Eric Zimmermann; Julian Viret; Michal Zelechowski; James Brian Hall; Neil Tenenholtz; Adam Casson; George Shaikovski; Eugene Vorontsov; Siqi Liu; Kristen A Severson
>
> **摘要:** In recent years, a standard computational pathology workflow has emerged where whole slide images are cropped into tiles, these tiles are processed using a foundation model, and task-specific models are built using the resulting representations. At least 15 different foundation models have been proposed, and the vast majority are trained exclusively with tiles using the 20$\times$ magnification. However, it is well known that certain histologic features can only be discerned with larger context windows and requires a pathologist to zoom in and out when analyzing a whole slide image. Furthermore, creating 224$\times$224 pixel crops at 20$\times$ leads to a large number of tiles per slide, which can be gigapixel in size. To more accurately capture multi-resolution features and investigate the possibility of reducing the number of representations per slide, we propose a region-level mixing encoder. Our approach jointly fuses image tile representations of a mixed magnification foundation model using a masked embedding modeling pretraining step. We explore a design space for pretraining the proposed mixed-magnification region aggregators and evaluate our models on transfer to biomarker prediction tasks representing various cancer types. Results demonstrate cancer dependent improvements in predictive performance, highlighting the importance of spatial context and understanding.
>
---
#### [new 092] Global-Local Dual Perception for MLLMs in High-Resolution Text-Rich Image Translation
- **分类: cs.CV**

- **简介: 该论文属于文本图像机器翻译任务，旨在解决高分辨率文本丰富图像中的翻译不完整和语义偏差问题。提出GLoTran框架，结合全局与局部视觉感知，提升翻译准确性和上下文一致性。**

- **链接: [https://arxiv.org/pdf/2602.21956v1](https://arxiv.org/pdf/2602.21956v1)**

> **作者:** Junxin Lu; Tengfei Song; Zhanglin Wu; Pengfei Li; Xiaowei Liang; Hui Yang; Kun Chen; Ning Xie; Yunfei Lu; Jing Zhao; Shiliang Sun; Daimeng Wei
>
> **摘要:** Text Image Machine Translation (TIMT) aims to translate text embedded in images in the source-language into target-language, requiring synergistic integration of visual perception and linguistic understanding. Existing TIMT methods, whether cascaded pipelines or end-to-end multimodal large language models (MLLMs),struggle with high-resolution text-rich images due to cluttered layouts, diverse fonts, and non-textual distractions, resulting in text omission, semantic drift, and contextual inconsistency. To address these challenges, we propose GLoTran, a global-local dual visual perception framework for MLLM-based TIMT. GLoTran integrates a low-resolution global image with multi-scale region-level text image slices under an instruction-guided alignment strategy, conditioning MLLMs to maintain scene-level contextual consistency while faithfully capturing fine-grained textual details. Moreover, to realize this dual-perception paradigm, we construct GLoD, a large-scale text-rich TIMT dataset comprising 510K high-resolution global-local image-text pairs covering diverse real-world scenarios. Extensive experiments demonstrate that GLoTran substantially improves translation completeness and accuracy over state-of-the-art MLLMs, offering a new paradigm for fine-grained TIMT under high-resolution and text-rich conditions.
>
---
#### [new 093] SurGo-R1: Benchmarking and Modeling Contextual Reasoning for Operative Zone in Surgical Video
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉任务，解决手术视频中操作区域识别问题。通过构建基准数据集和提出SurGo-R1模型，提升手术阶段与操作区域的上下文推理能力。**

- **链接: [https://arxiv.org/pdf/2602.21706v1](https://arxiv.org/pdf/2602.21706v1)**

> **作者:** Guanyi Qin; Xiaozhen Wang; Zhu Zhuo; Chang Han Low; Yuancan Xiao; Yibing Fu; Haofeng Liu; Kai Wang; Chunjiang Li; Yueming Jin
>
> **摘要:** Minimally invasive surgery has dramatically improved patient operative outcomes, yet identifying safe operative zones remains challenging in critical phases, requiring surgeons to integrate visual cues, procedural phase, and anatomical context under high cognitive load. Existing AI systems offer binary safety verification or static detection, ignoring the phase-dependent nature of intraoperative reasoning. We introduce ResGo, a benchmark of laparoscopic frames annotated with Go Zone bounding boxes and clinician-authored rationales covering phase, exposure quality reasoning, next action and risk reminder. We introduce evaluation metrics that treat correct grounding under incorrect phase as failures, revealing that most vision-language models cannot handle such tasks and perform poorly. We then present SurGo-R1, a model optimized via RLHF with a multi-turn phase-then-go architecture where the model first identifies the surgical phase, then generates reasoning and Go Zone coordinates conditioned on that context. On unseen procedures, SurGo-R1 achieves 76.6% phase accuracy, 32.7 mIoU, and 54.8% hardcore accuracy, a 6.6$\times$ improvement over the mainstream generalist VLMs. Code, model and benchmark will be available at https://github.com/jinlab-imvr/SurGo-R1
>
---
#### [new 094] UniHand: A Unified Model for Diverse Controlled 4D Hand Motion Modeling
- **分类: cs.CV**

- **简介: 该论文提出UniHand，解决4D手部运动建模问题，整合估计与生成任务，通过统一框架处理多源条件信号，提升模型鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2602.21631v1](https://arxiv.org/pdf/2602.21631v1)**

> **作者:** Zhihao Sun; Tong Wu; Ruirui Tu; Daoguo Dong; Zuxuan Wu
>
> **摘要:** Hand motion plays a central role in human interaction, yet modeling realistic 4D hand motion (i.e., 3D hand pose sequences over time) remains challenging. Research in this area is typically divided into two tasks: (1) Estimation approaches reconstruct precise motion from visual observations, but often fail under hand occlusion or absence; (2) Generation approaches focus on synthesizing hand poses by exploiting generative priors under multi-modal structured inputs and infilling motion from incomplete sequences. However, this separation not only limits the effective use of heterogeneous condition signals that frequently arise in practice, but also prevents knowledge transfer between the two tasks. We present UniHand, a unified diffusion-based framework that formulates both estimation and generation as conditional motion synthesis. UniHand integrates heterogeneous inputs by embedding structured signals into a shared latent space through a joint variational autoencoder, which aligns conditions such as MANO parameters and 2D skeletons. Visual observations are encoded with a frozen vision backbone, while a dedicated hand perceptron extracts hand-specific cues directly from image features, removing the need for complex detection and cropping pipelines. A latent diffusion model then synthesizes consistent motion sequences from these diverse conditions. Extensive experiments across multiple benchmarks demonstrate that UniHand delivers robust and accurate hand motion modeling, maintaining performance under severe occlusions and temporally incomplete inputs.
>
---
#### [new 095] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决单目占用预测问题。通过引入可泛化的视觉几何先验，提出GPOcc框架，提升预测精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.21552v1](https://arxiv.org/pdf/2602.21552v1)**

> **作者:** Changqing Zhou; Yueru Luo; Changhao Chen
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Accurate 3D scene understanding is essential for embodied intelligence, with occupancy prediction emerging as a key task for reasoning about both objects and free space. Existing approaches largely rely on depth priors (e.g., DepthAnything) but make only limited use of 3D cues, restricting performance and generalization. Recently, visual geometry models such as VGGT have shown strong capability in providing rich 3D priors, but similar to monocular depth foundation models, they still operate at the level of visible surfaces rather than volumetric interiors, motivating us to explore how to more effectively leverage these increasingly powerful geometry priors for 3D occupancy prediction. We present GPOcc, a framework that leverages generalizable visual geometry priors (GPs) for monocular occupancy prediction. Our method extends surface points inward along camera rays to generate volumetric samples, which are represented as Gaussian primitives for probabilistic occupancy inference. To handle streaming input, we further design a training-free incremental update strategy that fuses per-frame Gaussians into a unified global representation. Experiments on Occ-ScanNet and EmbodiedOcc-ScanNet demonstrate significant gains: GPOcc improves mIoU by +9.99 in the monocular setting and +11.79 in the streaming setting over prior state of the art. Under the same depth prior, it achieves +6.73 mIoU while running 2.65$\times$ faster. These results highlight that GPOcc leverages geometry priors more effectively and efficiently. Code will be released at https://github.com/JuIvyy/GPOcc.
>
---
#### [new 096] WeatherCity: Urban Scene Reconstruction with Controllable Multi-Weather Transformation
- **分类: cs.CV**

- **简介: 该论文属于4D城市场景重建任务，旨在解决现有方法无法灵活控制天气效果的问题。提出WeatherCity框架，结合文本编辑和物理模拟，实现高保真、可控的多天气场景生成。**

- **链接: [https://arxiv.org/pdf/2602.22096v1](https://arxiv.org/pdf/2602.22096v1)**

> **作者:** Wenhua Wu; Huai Guan; Zhe Liu; Hesheng Wang
>
> **摘要:** Editable high-fidelity 4D scenes are crucial for autonomous driving, as they can be applied to end-to-end training and closed-loop simulation. However, existing reconstruction methods are primarily limited to replicating observed scenes and lack the capability for diverse weather simulation. While image-level weather editing methods tend to introduce scene artifacts and offer poor controllability over the weather effects. To address these limitations, we propose WeatherCity, a novel framework for 4D urban scene reconstruction and weather editing. Specifically, we leverage a text-guided image editing model to achieve flexible editing of image weather backgrounds. To tackle the challenge of multi-weather modeling, we introduce a novel weather Gaussian representation based on shared scene features and dedicated weather-specific decoders. This representation is further enhanced with a content consistency optimization, ensuring coherent modeling across different weather conditions. Additionally, we design a physics-driven model that simulates dynamic weather effects through particles and motion patterns. Extensive experiments on multiple datasets and various scenes demonstrate that WeatherCity achieves flexible controllability, high fidelity, and temporal consistency in 4D reconstruction and weather editing. Our framework not only enables fine-grained control over weather conditions (e.g., light rain and heavy snow) but also supports object-level manipulation within the scene.
>
---
#### [new 097] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像描述生成任务，旨在解决现有模型依赖不完整或错误的人类标注的问题。提出CCCaption框架，通过双奖励机制优化描述的完整性和正确性。**

- **链接: [https://arxiv.org/pdf/2602.21655v1](https://arxiv.org/pdf/2602.21655v1)**

> **作者:** Zhijiang Tang; Linhua Wang; Jiaxin Qi; Weihao Jiang; Peng Hou; Anxiang Zeng; Jianqiang Huang
>
> **备注:** Accept by CVPR 2026
>
> **摘要:** Image captioning remains a fundamental task for vision language understanding, yet ground-truth supervision still relies predominantly on human-annotated references. Because human annotations reflect subjective preferences and expertise, ground-truth captions are often incomplete or even incorrect, which in turn limits caption models. We argue that caption quality should be assessed by two objective aspects: completeness (does the caption cover all salient visual facts?) and correctness (are the descriptions true with respect to the image?). To this end, we introduce CCCaption: a dual-reward reinforcement learning framework with a dedicated fine-tuning corpus that explicitly optimizes these properties to generate \textbf{C}omplete and \textbf{C}orrect \textbf{Captions}. For completeness, we use diverse LVLMs to disentangle the image into a set of visual queries, and reward captions that answer more of these queries, with a dynamic query sampling strategy to improve training efficiency. For correctness, we penalize captions that contain hallucinations by validating the authenticity of sub-caption queries, which are derived from the caption decomposition. Our symmetric dual-reward optimization jointly maximizes completeness and correctness, guiding models toward captions that better satisfy these objective criteria. Extensive experiments across standard captioning benchmarks show consistent improvements, offering a principled path to training caption models beyond human-annotation imitation.
>
---
#### [new 098] XStreamVGGT: Extremely Memory-Efficient Streaming Vision Geometry Grounded Transformer with KV Cache Compression
- **分类: cs.CV**

- **简介: 该论文提出XStreamVGGT，解决3D视觉重建中KV缓存内存过大的问题，通过剪枝和量化实现高效内存压缩，提升长时序应用的可扩展性。**

- **链接: [https://arxiv.org/pdf/2602.21780v1](https://arxiv.org/pdf/2602.21780v1)**

> **作者:** Zunhai Su; Weihao Ye; Hansen Feng; Keyu Fan; Jing Zhang; Dahai Yu; Zhengwu Liu; Ngai Wong
>
> **备注:** Submission to the Journal of the Society for Information Display
>
> **摘要:** Learning-based 3D visual geometry models have significantly advanced with the advent of large-scale transformers. Among these, StreamVGGT leverages frame-wise causal attention to deliver robust and efficient streaming 3D reconstruction. However, it suffers from unbounded growth in the Key-Value (KV) cache due to the massive influx of vision tokens from multi-image and long-video inputs, leading to increased memory consumption and inference latency as input frames accumulate. This ultimately limits its scalability for long-horizon applications. To address this gap, we propose XStreamVGGT, a tuning-free approach that seamlessly integrates pruning and quantization to systematically compress the KV cache, enabling extremely memory-efficient streaming inference. Specifically, redundant KVs generated from multi-frame inputs are initially pruned to conform to a fixed KV memory budget using an efficient token-importance identification mechanism that maintains full compatibility with high-performance attention kernels (e.g., FlashAttention). Additionally, leveraging the inherent distribution patterns of KV tensors, we apply dimension-adaptive KV quantization within the pruning pipeline to further minimize memory overhead while preserving numerical accuracy. Extensive evaluations show that XStreamVGGT achieves mostly negligible performance degradation while substantially reducing memory usage by 4.42$\times$ and accelerating inference by 5.48$\times$, enabling practical and scalable streaming 3D applications. The code is available at https://github.com/ywh187/XStreamVGGT/.
>
---
#### [new 099] Structure-to-Image: Zero-Shot Depth Estimation in Colonoscopy via High-Fidelity Sim-to-Real Adaptation
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理任务，解决结肠镜检查中单目深度估计的域差距问题。通过结构到图像的生成方法，提升模拟到真实数据的适应性。**

- **链接: [https://arxiv.org/pdf/2602.21740v1](https://arxiv.org/pdf/2602.21740v1)**

> **作者:** Juan Yang; Yuyan Zhang; Han Jia; Bing Hu; Wanzhong Song
>
> **备注:** \c{opyright} 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Monocular depth estimation (MDE) for colonoscopy is hampered by the domain gap between simulated and real-world images. Existing image-to-image translation methods, which use depth as a posterior constraint, often produce structural distortions and specular highlights by failing to balance realism with structure consistency. To address this, we propose a Structure-to-Image paradigm that transforms the depth map from a passive constraint into an active generative foundation. We are the first to introduce phase congruency to colonoscopic domain adaptation and design a cross-level structure constraint to co-optimize geometric structures and fine-grained details like vascular textures. In zero-shot evaluations conducted on a publicly available phantom dataset, the MDE model that was fine-tuned on our generated data achieved a maximum reduction of 44.18% in RMSE compared to competing methods. Our code is available at https://github.com/YyangJJuan/PC-S2I.git.
>
---
#### [new 100] SkyReels-V4: Multi-modal Video-Audio Generation, Inpainting and Editing model
- **分类: cs.CV**

- **简介: 该论文提出SkyReels-V4，属于视频生成与编辑任务，解决多模态输入下的视频音频联合生成与编辑问题，采用双流架构实现高效高质量视频生成。**

- **链接: [https://arxiv.org/pdf/2602.21818v1](https://arxiv.org/pdf/2602.21818v1)**

> **作者:** Guibin Chen; Dixuan Lin; Jiangping Yang; Youqiang Zhang; Zhengcong Fei; Debang Li; Sheng Chen; Chaofeng Ao; Nuo Pang; Yiming Wang; Yikun Dou; Zheng Chen; Mingyuan Fan; Tuanhui Li; Mingshan Chang; Hao Zhang; Xiaopeng Sun; Jingtao Xu; Yuqiang Xie; Jiahua Wang; Zhiheng Xu; Weiming Xiong; Yuzhe Jin; Baoxuan Gu; Binjie Mao; Yunjie Yu; Jujie He; Yuhao Feng; Shiwen Tu; Chaojie Wang; Rui Yan; Wei Shen; Jingchen Wu; Peng Zhao; Xuanyue Zhong; Zhuangzhuang Liu; Kaifei Wang; Fuxiang Zhang; Weikai Xu; Wenyan Liu; Binglu Zhang; Yu Shen; Tianhui Xiong; Bin Peng; Liang Zeng; Xuchen Song; Haoxiang Guo; Peiyu Wang; Yahui Zhou
>
> **摘要:** SkyReels V4 is a unified multi modal video foundation model for joint video audio generation, inpainting, and editing. The model adopts a dual stream Multimodal Diffusion Transformer (MMDiT) architecture, where one branch synthesizes video and the other generates temporally aligned audio, while sharing a powerful text encoder based on the Multimodal Large Language Models (MMLM). SkyReels V4 accepts rich multi modal instructions, including text, images, video clips, masks, and audio references. By combining the MMLMs multi modal instruction following capability with in context learning in the video branch MMDiT, the model can inject fine grained visual guidance under complex conditioning, while the audio branch MMDiT simultaneously leverages audio references to guide sound generation. On the video side, we adopt a channel concatenation formulation that unifies a wide range of inpainting style tasks, such as image to video, video extension, and video editing under a single interface, and naturally extends to vision referenced inpainting and editing via multi modal prompts. SkyReels V4 supports up to 1080p resolution, 32 FPS, and 15 second duration, enabling high fidelity, multi shot, cinema level video generation with synchronized audio. To make such high resolution, long-duration generation computationally feasible, we introduce an efficiency strategy: Joint generation of low resolution full sequences and high-resolution keyframes, followed by dedicated super-resolution and frame interpolation models. To our knowledge, SkyReels V4 is the first video foundation model that simultaneously supports multi-modal input, joint video audio generation, and a unified treatment of generation, inpainting, and editing, while maintaining strong efficiency and quality at cinematic resolutions and durations.
>
---
#### [new 101] MedTri: A Platform for Structured Medical Report Normalization to Enhance Vision-Language Pretraining
- **分类: cs.CV**

- **简介: 该论文属于医学视觉-语言预训练任务，旨在解决原始报告风格不一、信息冗余的问题。提出MedTri框架，将报告结构化为解剖实体-影像描述-诊断类别三元组，提升预训练效果。**

- **链接: [https://arxiv.org/pdf/2602.22143v1](https://arxiv.org/pdf/2602.22143v1)**

> **作者:** Yuetan Chu; Xinhua Ma; Xinran Jin; Gongning Luo; Xin Gao
>
> **摘要:** Medical vision-language pretraining increasingly relies on medical reports as large-scale supervisory signals; however, raw reports often exhibit substantial stylistic heterogeneity, variable length, and a considerable amount of image-irrelevant content. Although text normalization is frequently adopted as a preprocessing step in prior work, its design principles and empirical impact on vision-language pretraining remain insufficiently and systematically examined. In this study, we present MedTri, a deployable normalization framework for medical vision-language pretraining that converts free-text reports into a unified [Anatomical Entity: Radiologic Description + Diagnosis Category] triplet. This structured, anatomy-grounded normalization preserves essential morphological and spatial information while removing stylistic noise and image-irrelevant content, providing consistent and image-grounded textual supervision at scale. Across multiple datasets spanning both X-ray and computed tomography (CT) modalities, we demonstrate that structured, anatomy-grounded text normalization is an important factor in medical vision-language pretraining quality, yielding consistent improvements over raw reports and existing normalization baselines. In addition, we illustrate how this normalization can easily support modular text-level augmentation strategies, including knowledge enrichment and anatomy-grounded counterfactual supervision, which provide complementary gains in robustness and generalization without altering the core normalization process. Together, our results position structured text normalization as a critical and generalizable preprocessing component for medical vision-language learning, while MedTri provides this normalization platform. Code and data will be released at https://github.com/Arturia-Pendragon-Iris/MedTri.
>
---
#### [new 102] PatchDenoiser: Parameter-efficient multi-scale patch learning and fusion denoiser for medical images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像去噪任务，旨在解决传统方法细节丢失或模型复杂的问题。提出轻量级多尺度块学习框架PatchDenoiser，有效去噪并保留结构细节。**

- **链接: [https://arxiv.org/pdf/2602.21987v1](https://arxiv.org/pdf/2602.21987v1)**

> **作者:** Jitindra Fartiyal; Pedro Freire; Sergei K. Turitsyn; Sergei G. Solovski
>
> **备注:** Under review in Medical Image Analysis journal
>
> **摘要:** Medical images are essential for diagnosis, treatment planning, and research, but their quality is often degraded by noise from low-dose acquisition, patient motion, or scanner limitations, affecting both clinical interpretation and downstream analysis. Traditional filtering approaches often over-smooth and lose fine anatomical details, while deep learning methods, including CNNs, GANs, and transformers, may struggle to preserve such details or require large, computationally expensive models, limiting clinical practicality. We propose PatchDenoiser, a lightweight, energy-efficient multi-scale patch-based denoising framework. It decomposes denoising into local texture extraction and global context aggregation, fused via a spatially aware patch fusion strategy. This design enables effective noise suppression while preserving fine structural and anatomical details. PatchDenoiser is ultra-lightweight, with far fewer parameters and lower computational complexity than CNN-, GAN-, and transformer-based denoisers. On the 2016 Mayo Low-Dose CT dataset, PatchDenoiser consistently outperforms state-of-the-art CNN- and GAN-based methods in PSNR and SSIM. It is robust to variations in slice thickness, reconstruction kernels, and HU windows, generalizes across scanners without fine-tuning, and reduces parameters by ~9x and energy consumption per inference by ~27x compared with conventional CNN denoisers. PatchDenoiser thus provides a practical, scalable, and computationally efficient solution for medical image denoising, balancing performance, robustness, and clinical deployability.
>
---
#### [new 103] Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling
- **分类: cs.CV**

- **简介: 该论文属于扩散模型加速任务，旨在解决分布式并行方法生成质量下降和加速效果有限的问题。通过混合数据与流水线并行策略，提升推理速度并保持图像质量。**

- **链接: [https://arxiv.org/pdf/2602.21760v1](https://arxiv.org/pdf/2602.21760v1)**

> **作者:** Euisoo Jung; Byunghyun Kim; Hyunjin Kim; Seonghye Cho; Jae-Gil Lee
>
> **摘要:** Diffusion models have achieved remarkable progress in high-fidelity image, video, and audio generation, yet inference remains computationally expensive. Nevertheless, current diffusion acceleration methods based on distributed parallelism suffer from noticeable generation artifacts and fail to achieve substantial acceleration proportional to the number of GPUs. Therefore, we propose a hybrid parallelism framework that combines a novel data parallel strategy, condition-based partitioning, with an optimal pipeline scheduling method, adaptive parallelism switching, to reduce generation latency and achieve high generation quality in conditional diffusion models. The key ideas are to (i) leverage the conditional and unconditional denoising paths as a new data-partitioning perspective and (ii) adaptively enable optimal pipeline parallelism according to the denoising discrepancy between these two paths. Our framework achieves $2.31\times$ and $2.07\times$ latency reductions on SDXL and SD3, respectively, using two NVIDIA RTX~3090 GPUs, while preserving image quality. This result confirms the generality of our approach across U-Net-based diffusion models and DiT-based flow-matching architectures. Our approach also outperforms existing methods in acceleration under high-resolution synthesis settings. Code is available at https://github.com/kaist-dmlab/Hybridiff.
>
---
#### [new 104] MMLoP: Multi-Modal Low-Rank Prompting for Efficient Vision-Language Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出MMLoP，解决视觉-语言模型参数效率问题，通过低秩提示实现高效多模态适配。**

- **链接: [https://arxiv.org/pdf/2602.21397v1](https://arxiv.org/pdf/2602.21397v1)**

> **作者:** Sajjad Ghiasvand; Haniyeh Ehsani Oskouie; Mahnoosh Alizadeh; Ramtin Pedarsani
>
> **摘要:** Prompt learning has become a dominant paradigm for adapting vision-language models (VLMs) such as CLIP to downstream tasks without modifying pretrained weights. While extending prompts to both vision and text encoders across multiple transformer layers significantly boosts performance, it dramatically increases the number of trainable parameters, with state-of-the-art methods requiring millions of parameters and abandoning the parameter efficiency that makes prompt tuning attractive. In this work, we propose \textbf{MMLoP} (\textbf{M}ulti-\textbf{M}odal \textbf{Lo}w-Rank \textbf{P}rompting), a framework that achieves deep multi-modal prompting with only \textbf{11.5K trainable parameters}, comparable to early text-only methods like CoOp. MMLoP parameterizes vision and text prompts at each transformer layer through a low-rank factorization, which serves as an implicit regularizer against overfitting on few-shot training data. To further close the accuracy gap with state-of-the-art methods, we introduce three complementary components: a self-regulating consistency loss that anchors prompted representations to frozen zero-shot CLIP features at both the feature and logit levels, a uniform drift correction that removes the global embedding shift induced by prompt tuning to preserve class-discriminative structure, and a shared up-projection that couples vision and text prompts through a common low-rank factor to enforce cross-modal alignment. Extensive experiments across three benchmarks and 11 diverse datasets demonstrate that MMLoP achieves a highly favorable accuracy-efficiency tradeoff, outperforming the majority of existing methods including those with orders of magnitude more parameters, while achieving a harmonic mean of 79.70\% on base-to-novel generalization.
>
---
#### [new 105] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文研究扩散Transformer的条件嵌入，揭示其语义瓶颈。针对生成任务中的嵌入冗余问题，通过分析发现语义信息集中于少数维度，并验证剪枝有效性，为更高效条件机制提供新思路。**

- **链接: [https://arxiv.org/pdf/2602.21596v1](https://arxiv.org/pdf/2602.21596v1)**

> **作者:** Trung X. Pham; Kang Zhang; Ji Woo Hong; Chang D. Yoo
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Diffusion Transformers have achieved state-of-the-art performance in class-conditional and multimodal generation, yet the structure of their learned conditional embeddings remains poorly understood. In this work, we present the first systematic study of these embeddings and uncover a notable redundancy: class-conditioned embeddings exhibit extreme angular similarity, exceeding 99\% on ImageNet-1K, while continuous-condition tasks such as pose-guided image generation and video-to-audio generation reach over 99.9\%. We further find that semantic information is concentrated in a small subset of dimensions, with head dimensions carrying the dominant signal and tail dimensions contributing minimally. By pruning low-magnitude dimensions--removing up to two-thirds of the embedding space--we show that generation quality and fidelity remain largely unaffected, and in some cases improve. These results reveal a semantic bottleneck in Transformer-based diffusion models, providing new insights into how semantics are encoded and suggesting opportunities for more efficient conditioning mechanisms.
>
---
#### [new 106] StoryMovie: A Dataset for Semantic Alignment of Visual Stories with Movie Scripts and Subtitles
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出StoryMovie数据集，用于视觉故事与电影剧本、字幕的语义对齐。解决对话归属和角色关系生成中的错误问题，通过脚本与字幕的时间同步实现更准确的对话 attribution。**

- **链接: [https://arxiv.org/pdf/2602.21829v1](https://arxiv.org/pdf/2602.21829v1)**

> **作者:** Daniel Oliveira; David Martins de Matos
>
> **备注:** 15 pages, submitted to Journal of Visual Communication and Image Representation
>
> **摘要:** Visual storytelling models that correctly ground entities in images may still hallucinate semantic relationships, generating incorrect dialogue attribution, character interactions, or emotional states. We introduce StoryMovie, a dataset of 1,757 stories aligned with movie scripts and subtitles through LCS matching. Our alignment pipeline synchronizes screenplay dialogue with subtitle timestamps, enabling dialogue attribution by linking character names from scripts to temporal positions from subtitles. Using this aligned content, we generate stories that maintain visual grounding tags while incorporating authentic character names, dialogue, and relationship dynamics. We fine-tune Qwen Storyteller3 on this dataset, building on prior work in visual grounding and entity re-identification. Evaluation using DeepSeek V3 as judge shows that Storyteller3 achieves an 89.9% win rate against base Qwen2.5-VL 7B on subtitle alignment. Compared to Storyteller, trained without script grounding, Storyteller3 achieves 48.5% versus 38.0%, confirming that semantic alignment progressively improves dialogue attribution beyond visual grounding alone.
>
---
#### [new 107] RelA-Diffusion: Relativistic Adversarial Diffusion for Multi-Tracer PET Synthesis from Multi-Sequence MRI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于多模态图像合成任务，旨在解决多示踪剂PET图像生成难题。通过引入RelA-Diffusion框架，提升合成图像的准确性与真实性。**

- **链接: [https://arxiv.org/pdf/2602.21345v1](https://arxiv.org/pdf/2602.21345v1)**

> **作者:** Minhui Yu; Yongheng Sun; David S. Lalush; Jason P Mihalik; Pew-Thian Yap; Mingxia Liu
>
> **摘要:** Multi-tracer positron emission tomography (PET) provides critical insights into diverse neuropathological processes such as tau accumulation, neuroinflammation, and $β$-amyloid deposition in the brain, making it indispensable for comprehensive neurological assessment. However, routine acquisition of multi-tracer PET is limited by high costs, radiation exposure, and restricted tracer availability. Recent efforts have explored deep learning approaches for synthesizing PET images from structural MRI. While some methods rely solely on T1-weighted MRI, others incorporate additional sequences such as T2-FLAIR to improve pathological sensitivity. However, existing methods often struggle to capture fine-grained anatomical and pathological details, resulting in artifacts and unrealistic outputs. To this end, we propose RelA-Diffusion, a Relativistic Adversarial Diffusion framework for multi-tracer PET synthesis from multi-sequence MRI. By leveraging both T1-weighted and T2-FLAIR scans as complementary inputs, RelA-Diffusion captures richer structural information to guide image generation. To improve synthesis fidelity, we introduce a gradient-penalized relativistic adversarial loss to the intermediate clean predictions of the diffusion model. This loss compares real and generated images in a relative manner, encouraging the synthesis of more realistic local structures. Both the relativistic formulation and the gradient penalty contribute to stabilizing the training, while adversarial feedback at each diffusion timestep enables consistent refinement throughout the generation process. Extensive experiments on two datasets demonstrate that RelA-Diffusion outperforms existing methods in both visual fidelity and quantitative metrics, highlighting its potential for accurate synthesis of multi-tracer PET.
>
---
#### [new 108] Perceptual Quality Optimization of Image Super-Resolution
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属于图像超分辨率任务，旨在解决重建质量与视觉效果之间的权衡问题。提出Efficient-PBAN网络，通过感知质量优化提升图像视觉效果。**

- **链接: [https://arxiv.org/pdf/2602.21482v1](https://arxiv.org/pdf/2602.21482v1)**

> **作者:** Wei Zhou; Yixiao Li; Hadi Amirpour; Xiaoshuai Hao; Jiang Liu; Peng Wang; Hantao Liu
>
> **备注:** 6 pages, 2 figures, accepted in ICASSP 26
>
> **摘要:** Single-image super-resolution (SR) has achieved remarkable progress with deep learning, yet most approaches rely on distortion-oriented losses or heuristic perceptual priors, which often lead to a trade-off between fidelity and visual quality. To address this issue, we propose an \textit{Efficient Perceptual Bi-directional Attention Network (Efficient-PBAN)} that explicitly optimizes SR towards human-preferred quality. Unlike patch-based quality models, Efficient-PBAN avoids extensive patch sampling and enables efficient image-level perception. The proposed framework is trained on our self-constructed SR quality dataset that covers a wide range of state-of-the-art SR methods with corresponding human opinion scores. Using this dataset, Efficient-PBAN learns to predict perceptual quality in a way that correlates strongly with subjective judgments. The learned metric is further integrated into SR training as a differentiable perceptual loss, enabling closed-loop alignment between reconstruction and perceptual assessment. Extensive experiments demonstrate that our approach delivers superior perceptual quality. Code is publicly available at https://github.com/Lighting-YXLI/Efficient-PBAN.
>
---
#### [new 109] LiLo-VLA: Compositional Long-Horizon Manipulation via Linked Object-Centric Policies
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; eess.SY**

- **简介: 该论文提出LiLo-VLA，解决长时序操作任务中的组合复杂性和环境敏感性问题，通过模块化设计提升鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21531v1](https://arxiv.org/pdf/2602.21531v1)**

> **作者:** Yue Yang; Shuo Cheng; Yu Fang; Homanga Bharadhwaj; Mingyu Ding; Gedas Bertasius; Daniel Szafir
>
> **摘要:** General-purpose robots must master long-horizon manipulation, defined as tasks involving multiple kinematic structure changes (e.g., attaching or detaching objects) in unstructured environments. While Vision-Language-Action (VLA) models offer the potential to master diverse atomic skills, they struggle with the combinatorial complexity of sequencing them and are prone to cascading failures due to environmental sensitivity. To address these challenges, we propose LiLo-VLA (Linked Local VLA), a modular framework capable of zero-shot generalization to novel long-horizon tasks without ever being trained on them. Our approach decouples transport from interaction: a Reaching Module handles global motion, while an Interaction Module employs an object-centric VLA to process isolated objects of interest, ensuring robustness against irrelevant visual features and invariance to spatial configurations. Crucially, this modularity facilitates robust failure recovery through dynamic replanning and skill reuse, effectively mitigating the cascading errors common in end-to-end approaches. We introduce a 21-task simulation benchmark consisting of two challenging suites: LIBERO-Long++ and Ultra-Long. In these simulations, LiLo-VLA achieves a 69% average success rate, outperforming Pi0.5 by 41% and OpenVLA-OFT by 67%. Furthermore, real-world evaluations across 8 long-horizon tasks demonstrate an average success rate of 85%. Project page: https://yy-gx.github.io/LiLo-VLA/.
>
---
#### [new 110] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于机器学习中的模型遗忘任务，旨在解决模型因偏差而难以正确遗忘的问题。通过分析“快捷遗忘”现象，提出CUPID框架实现更有效的遗忘。**

- **链接: [https://arxiv.org/pdf/2602.21773v1](https://arxiv.org/pdf/2602.21773v1)**

> **作者:** JuneHyoung Kwon; MiHyeon Kim; Eunju Lee; Yoonji Lee; Seunghoon Lee; YoungBin Kim
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Machine unlearning, which enables a model to forget specific data, is crucial for ensuring data privacy and model reliability. However, its effectiveness can be severely undermined in real-world scenarios where models learn unintended biases from spurious correlations within the data. This paper investigates the unique challenges of unlearning from such biased models. We identify a novel phenomenon we term ``shortcut unlearning," where models exhibit an ``easy to learn, yet hard to forget" tendency. Specifically, models struggle to forget easily-learned, bias-aligned samples; instead of forgetting the class attribute, they unlearn the bias attribute, which can paradoxically improve accuracy on the class intended to be forgotten. To address this, we propose CUPID, a new unlearning framework inspired by the observation that samples with different biases exhibit distinct loss landscape sharpness. Our method first partitions the forget set into causal- and bias-approximated subsets based on sample sharpness, then disentangles model parameters into causal and bias pathways, and finally performs a targeted update by routing refined causal and bias gradients to their respective pathways. Extensive experiments on biased datasets including Waterbirds, BAR, and Biased NICO++ demonstrate that our method achieves state-of-the-art forgetting performance and effectively mitigates the shortcut unlearning problem.
>
---
#### [new 111] World Guidance: World Modeling in Condition Space for Action Generation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，解决未来预测与动作生成的平衡问题。提出WoG框架，通过条件空间建模提升动作生成精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.22010v1](https://arxiv.org/pdf/2602.22010v1)**

> **作者:** Yue Su; Sijin Chen; Haixin Shi; Mingyu Liu; Zhengshen Zhang; Ningyuan Huang; Weiheng Zhong; Zhengbang Zhu; Yuxiao Liu; Xihui Liu
>
> **备注:** Project Page: https://selen-suyue.github.io/WoGNet/
>
> **摘要:** Leveraging future observation modeling to facilitate action generation presents a promising avenue for enhancing the capabilities of Vision-Language-Action (VLA) models. However, existing approaches struggle to strike a balance between maintaining efficient, predictable future representations and preserving sufficient fine-grained information to guide precise action generation. To address this limitation, we propose WoG (World Guidance), a framework that maps future observations into compact conditions by injecting them into the action inference pipeline. The VLA is then trained to simultaneously predict these compressed conditions alongside future actions, thereby achieving effective world modeling within the condition space for action inference. We demonstrate that modeling and predicting this condition space not only facilitates fine-grained action generation but also exhibits superior generalization capabilities. Moreover, it learns effectively from substantial human manipulation videos. Extensive experiments across both simulation and real-world environments validate that our method significantly outperforms existing methods based on future prediction. Project page is available at: https://selen-suyue.github.io/WoGNet/
>
---
#### [new 112] Learning spatially adaptive sparsity level maps for arbitrary convolutional dictionaries
- **分类: eess.IV; cs.CV; cs.LG; math.OC**

- **简介: 该论文属于图像重建任务，旨在提升深度学习方法的可解释性和鲁棒性。通过引入空间自适应稀疏度图，优化卷积字典的使用，增强模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21707v1](https://arxiv.org/pdf/2602.21707v1)**

> **作者:** Joshua Schulz; David Schote; Christoph Kolbitsch; Kostas Papafitsoros; Andreas Kofler
>
> **摘要:** State-of-the-art learned reconstruction methods often rely on black-box modules that, despite their strong performance, raise questions about their interpretability and robustness. Here, we build on a recently proposed image reconstruction method, which is based on embedding data-driven information into a model-based convolutional dictionary regularization via neural network-inferred spatially adaptive sparsity level maps. By means of improved network design and dedicated training strategies, we extend the method to achieve filter-permutation invariance as well as the possibility to change the convolutional dictionary at inference time. We apply our method to low-field MRI and compare it to several other recent deep learning-based methods, also on in vivo data, in which the benefit for the use of a different dictionary is showcased. We further assess the method's robustness when tested on in- and out-of-distribution data. When tested on the latter, the proposed method suffers less from the data distribution shift compared to the other learned methods, which we attribute to its reduced reliance on training data due to its underlying model-based reconstruction component.
>
---
#### [new 113] Self-Correcting VLA: Online Action Refinement via Sparse World Imagination
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出SC-VLA，解决VLA模型对物理动态理解不足和缺乏自改进机制的问题。通过稀疏世界想象和在线动作优化，提升机器人操作任务性能。**

- **链接: [https://arxiv.org/pdf/2602.21633v1](https://arxiv.org/pdf/2602.21633v1)**

> **作者:** Chenyv Liu; Wentao Tan; Lei Zhu; Fengling Li; Jingjing Li; Guoli Yang; Heng Tao Shen
>
> **摘要:** Standard vision-language-action (VLA) models rely on fitting statistical data priors, limiting their robust understanding of underlying physical dynamics. Reinforcement learning enhances physical grounding through exploration yet typically relies on external reward signals that remain isolated from the agent's internal states. World action models have emerged as a promising paradigm that integrates imagination and control to enable predictive planning. However, they rely on implicit context modeling, lacking explicit mechanisms for self-improvement. To solve these problems, we propose Self-Correcting VLA (SC-VLA), which achieve self-improvement by intrinsically guiding action refinement through sparse imagination. We first design sparse world imagination by integrating auxiliary predictive heads to forecast current task progress and future trajectory trends, thereby constraining the policy to encode short-term physical evolution. Then we introduce the online action refinement module to reshape progress-dependent dense rewards, adjusting trajectory orientation based on the predicted sparse future states. Evaluations on challenging robot manipulation tasks from simulation benchmarks and real-world settings demonstrate that SC-VLA achieve state-of-the-art performance, yielding the highest task throughput with 16% fewer steps and a 9% higher success rate than the best-performing baselines, alongside a 14% gain in real-world experiments. Code is available at https://github.com/Kisaragi0/SC-VLA.
>
---
#### [new 114] Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出Lumosaic系统，解决动态场景下高光谱视频捕获问题。通过主动照明与编码曝光像素结合，实现高精度、实时的光谱视频重建。**

- **链接: [https://arxiv.org/pdf/2602.22140v1](https://arxiv.org/pdf/2602.22140v1)**

> **作者:** Dhruv Verma; Andrew Qiu; Roberto Rangel; Ayandev Barman; Hao Yang; Chenjia Hu; Fengqi Zhang; Roman Genov; David B. Lindell; Kiriakos N. Kutulakos; Alex Mariakakis
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We present Lumosaic, a compact active hyperspectral video system designed for real-time capture of dynamic scenes. Our approach combines a narrowband LED array with a coded-exposure-pixel (CEP) camera capable of high-speed, per-pixel exposure control, enabling joint encoding of scene information across space, time, and wavelength within each video frame. Unlike passive snapshot systems that divide light across multiple spectral channels simultaneously and assume no motion during a frame's exposure, Lumosaic actively synchronizes illumination and pixel-wise exposure, improving photon utilization and preserving spectral fidelity under motion. A learning-based reconstruction pipeline then recovers 31-channel hyperspectral (400-700 nm) video at 30 fps and VGA resolution, producing temporally coherent and spectrally accurate reconstructions. Experiments on synthetic and real data demonstrate that Lumosaic significantly improves reconstruction fidelity and temporal stability over existing snapshot hyperspectral imaging systems, enabling robust hyperspectral video across diverse materials and motion conditions.
>
---
#### [new 115] Towards single-shot coherent imaging via overlap-free ptychography
- **分类: physics.optics; cs.AI; cs.CV; cs.LG; physics.comp-ph**

- **简介: 该论文属于相干成像任务，旨在解决同步辐射和XFEL源中重叠扫描限制吞吐量和增加剂量的问题。通过扩展PtychoPINN框架，实现无重叠、单次曝光的成像，并提升传统多帧相位层析成像的速度与效果。**

- **链接: [https://arxiv.org/pdf/2602.21361v1](https://arxiv.org/pdf/2602.21361v1)**

> **作者:** Oliver Hoidn; Aashwin Mishra; Steven Henke; Albert Vong; Matthew Seaberg
>
> **摘要:** Ptychographic imaging at synchrotron and XFEL sources requires dense overlapping scans, limiting throughput and increasing dose. Extending coherent diffractive imaging to overlap-free operation on extended samples remains an open problem. Here, we extend PtychoPINN (O. Hoidn \emph{et al.}, \emph{Scientific Reports} \textbf{13}, 22789, 2023) to deliver \emph{overlap-free, single-shot} reconstructions in a Fresnel coherent diffraction imaging (CDI) geometry while also accelerating conventional multi-shot ptychography. The framework couples a differentiable forward model of coherent scattering with a Poisson photon-counting likelihood; real-space overlap enters as a tunable parameter via coordinate-based grouping rather than a hard requirement. On synthetic benchmarks, reconstructions remain accurate at low counts ($\sim\!10^4$ photons/frame), and overlap-free single-shot reconstruction with an experimental probe reaches amplitude structural similarity (SSIM) 0.904, compared with 0.968 for overlap-constrained reconstruction. Against a data-saturated supervised model with the same backbone (16,384 training images), PtychoPINN achieves higher SSIM with only 1,024 images and generalizes to unseen illumination profiles. Per-graphics processing unit (GPU) throughput is approximately $40\times$ that of least-squares maximum-likelihood (LSQ-ML) reconstruction at matched $128\times128$ resolution. These results, validated on experimental data from the Advanced Photon Source and the Linac Coherent Light Source, unify single-exposure Fresnel CDI and overlapped ptychography within one framework, supporting dose-efficient, high-throughput imaging at modern light sources.
>
---
#### [new 116] Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of Humanoid Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，旨在解决 humanoid 控制中数据分布固定和获取成本高的问题。提出闭环自动化数据生成框架，提升控制策略性能与可扩展性。**

- **链接: [https://arxiv.org/pdf/2602.21599v1](https://arxiv.org/pdf/2602.21599v1)**

> **作者:** Weisheng Xu; Qiwei Wu; Jiaxi Zhang; Tan Jing; Yangfan Li; Yuetong Fang; Jiaqi Xiong; Kai Wu; Rong Ou; Renjing Xu
>
> **摘要:** Physics-based humanoid control relies on training with motion datasets that have diverse data distributions. However, the fixed difficulty distribution of datasets limits the performance ceiling of the trained control policies. Additionally, the method of acquiring high-quality data through professional motion capture systems is constrained by costs, making it difficult to achieve large-scale scalability. To address these issues, we propose a closed-loop automated motion data generation and iterative framework. It can generate high-quality motion data with rich action semantics, including martial arts, dance, combat, sports, gymnastics, and more. Furthermore, our framework enables difficulty iteration of policies and data through physical metrics and objective evaluations, allowing the trained tracker to break through its original difficulty limits. On the PHC single-primitive tracker, using only approximately 1/10 of the AMASS dataset size, the average failure rate on the test set (2201 clips) is reduced by 45\% compared to the baseline. Finally, we conduct comprehensive ablation and comparative experiments to highlight the rationality and advantages of our framework.
>
---
#### [new 117] Learning in the Null Space: Small Singular Values for Continual Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决灾难性遗忘问题。通过利用小奇异值构建近似零空间，提出NESS方法，在权重空间直接应用正交性，实现高效且稳定的持续学习。**

- **链接: [https://arxiv.org/pdf/2602.21919v1](https://arxiv.org/pdf/2602.21919v1)**

> **作者:** Cuong Anh Pham; Praneeth Vepakomma; Samuel Horváth
>
> **备注:** 17 pages, accepted as Oral presentation at the Third Conference on Parsimony and Learning (CPAL 2026)
>
> **摘要:** Alleviating catastrophic forgetting while enabling further learning is a primary challenge in continual learning (CL). Orthogonal-based training methods have gained attention for their efficiency and strong theoretical properties, and many existing approaches enforce orthogonality through gradient projection. In this paper, we revisit orthogonality and exploit the fact that small singular values correspond to directions that are nearly orthogonal to the input space of previous tasks. Building on this principle, we introduce NESS (Null-space Estimated from Small Singular values), a CL method that applies orthogonality directly in the weight space rather than through gradient manipulation. Specifically, NESS constructs an approximate null space using the smallest singular values of each layer's input representation and parameterizes task-specific updates via a compact low-rank adaptation (LoRA-style) formulation constrained to this subspace. The subspace basis is fixed to preserve the null-space constraint, and only a single trainable matrix is learned for each task. This design ensures that the resulting updates remain approximately in the null space of previous inputs while enabling adaptation to new tasks. Our theoretical analysis and experiments on three benchmark datasets demonstrate competitive performance, low forgetting, and stable accuracy across tasks, highlighting the role of small singular values in continual learning. The code is available at https://github.com/pacman-ctm/NESS.
>
---
#### [new 118] Dream-SLAM: Dreaming the Unseen for Active SLAM in Dynamic Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于主动SLAM任务，旨在解决动态环境中定位、建图和探索效率的问题。提出Dream-SLAM方法，通过生成跨时空图像提升场景理解与长期规划能力。**

- **链接: [https://arxiv.org/pdf/2602.21967v1](https://arxiv.org/pdf/2602.21967v1)**

> **作者:** Xiangqi Meng; Pengxu Hou; Zhenjun Zhao; Javier Civera; Daniel Cremers; Hesheng Wang; Haoang Li
>
> **摘要:** In addition to the core tasks of simultaneous localization and mapping (SLAM), active SLAM additionally in- volves generating robot actions that enable effective and efficient exploration of unknown environments. However, existing active SLAM pipelines are limited by three main factors. First, they inherit the restrictions of the underlying SLAM modules that they may be using. Second, their motion planning strategies are typically shortsighted and lack long-term vision. Third, most approaches struggle to handle dynamic scenes. To address these limitations, we propose a novel monocular active SLAM method, Dream-SLAM, which is based on dreaming cross-spatio-temporal images and semantically plausible structures of partially observed dynamic environments. The generated cross-spatio-temporal im- ages are fused with real observations to mitigate noise and data incompleteness, leading to more accurate camera pose estimation and a more coherent 3D scene representation. Furthermore, we integrate dreamed and observed scene structures to enable long- horizon planning, producing farsighted trajectories that promote efficient and thorough exploration. Extensive experiments on both public and self-collected datasets demonstrate that Dream-SLAM outperforms state-of-the-art methods in localization accuracy, mapping quality, and exploration efficiency. Source code will be publicly available upon paper acceptance.
>
---
#### [new 119] Breaking Semantic-Aware Watermarks via LLM-Guided Coherence-Preserving Semantic Injection
- **分类: cs.LG; cs.CR; cs.CV**

- **简介: 该论文属于图像水印安全任务，旨在解决语义水印被LLM攻击的问题。通过提出CSI攻击，实现语义扰动以破坏水印检测。**

- **链接: [https://arxiv.org/pdf/2602.21593v1](https://arxiv.org/pdf/2602.21593v1)**

> **作者:** Zheng Gao; Xiaoyu Li; Zhicheng Bao; Xiaoyan Feng; Jiaojiao Jiang
>
> **备注:** Accepted by The Web Conference 2026 (Short Paper Track)
>
> **摘要:** Generative images have proliferated on Web platforms in social media and online copyright distribution scenarios, and semantic watermarking has increasingly been integrated into diffusion models to support reliable provenance tracking and forgery prevention for web content. Traditional noise-layer-based watermarking, however, remains vulnerable to inversion attacks that can recover embedded signals. To mitigate this, recent content-aware semantic watermarking schemes bind watermark signals to high-level image semantics, constraining local edits that would otherwise disrupt global coherence. Yet, large language models (LLMs) possess structured reasoning capabilities that enable targeted exploration of semantic spaces, allowing locally fine-grained but globally coherent semantic alterations that invalidate such bindings. To expose this overlooked vulnerability, we introduce a Coherence-Preserving Semantic Injection (CSI) attack that leverages LLM-guided semantic manipulation under embedding-space similarity constraints. This alignment enforces visual-semantic consistency while selectively perturbing watermark-relevant semantics, ultimately inducing detector misclassification. Extensive empirical results show that CSI consistently outperforms prevailing attack baselines against content-aware semantic watermarking, revealing a fundamental security weakness of current semantic watermark designs when confronted with LLM-driven semantic perturbations.
>
---
#### [new 120] WaterVIB: Learning Minimal Sufficient Watermark Representations via Variational Information Bottleneck
- **分类: cs.LG; cs.CR; cs.CV**

- **简介: 该论文属于水印任务，解决AIGC攻击下水印鲁棒性不足的问题。提出WaterVIB框架，通过信息瓶颈学习最小充分统计量，提升水印抗生成扰动能力。**

- **链接: [https://arxiv.org/pdf/2602.21508v1](https://arxiv.org/pdf/2602.21508v1)**

> **作者:** Haoyuan He; Yu Zheng; Jie Zhou; Jiwen Lu
>
> **备注:** 22 pages, 7 figures. Preprint
>
> **摘要:** Robust watermarking is critical for intellectual property protection, whereas existing methods face a severe vulnerability against regeneration-based AIGC attacks. We identify that existing methods fail because they entangle the watermark with high-frequency cover texture, which is susceptible to being rewritten during generative purification. To address this, we propose WaterVIB, a theoretically grounded framework that reformulates the encoder as an information sieve via the Variational Information Bottleneck. Instead of overfitting to fragile cover details, our approach forces the model to learn a Minimal Sufficient Statistic of the message. This effectively filters out redundant cover nuances prone to generative shifts, retaining only the essential signal invariant to regeneration. We theoretically prove that optimizing this bottleneck is a necessary condition for robustness against distribution-shifting attacks. Extensive experiments demonstrate that WaterVIB significantly outperforms state-of-the-art methods, achieving superior zero-shot resilience against unknown diffusion-based editing.
>
---
#### [new 121] Causal Decoding for Hallucination-Resistant Multimodal Large Language Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决对象幻觉问题。通过因果解码框架减少错误对象提及，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2602.21441v1](https://arxiv.org/pdf/2602.21441v1)**

> **作者:** Shiwei Tan; Hengyi Wang; Weiyi Qin; Qi Xu; Zhigang Hua; Hao Wang
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR), 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) deliver detailed responses on vision-language tasks, yet remain susceptible to object hallucination (introducing objects not present in the image), undermining reliability in practice. Prior efforts often rely on heuristic penalties, post-hoc correction, or generic decoding tweaks, which do not directly intervene in the mechanisms that trigger object hallucination and thus yield limited gains. To address this challenge, we propose a causal decoding framework that applies targeted causal interventions during generation to curb spurious object mentions. By reshaping the decoding dynamics to attenuate spurious dependencies, our approach reduces false object tokens while maintaining descriptive quality. Across captioning and QA benchmarks, our framework substantially lowers object-hallucination rates and achieves state-of-the-art faithfulness without degrading overall output quality.
>
---
#### [new 122] FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于联邦学习任务，旨在解决数据异质性导致的客户端漂移问题。提出FedVG框架，通过全局验证集引导优化，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21399v1](https://arxiv.org/pdf/2602.21399v1)**

> **作者:** Alina Devkota; Jacob Thrasher; Donald Adjeroh; Binod Bhattarai; Prashnna K. Gyawali
>
> **备注:** Accepted to CVPR 2026 (Findings Track)
>
> **摘要:** Federated Learning (FL) enables collaborative model training across multiple clients without sharing their private data. However, data heterogeneity across clients leads to client drift, which degrades the overall generalization performance of the model. This effect is further compounded by overemphasis on poorly performing clients. To address this problem, we propose FedVG, a novel gradient-based federated aggregation framework that leverages a global validation set to guide the optimization process. Such a global validation set can be established using readily available public datasets, ensuring accessibility and consistency across clients without compromising privacy. In contrast to conventional approaches that prioritize client dataset volume, FedVG assesses the generalization ability of client models by measuring the magnitude of validation gradients across layers. Specifically, we compute layerwise gradient norms to derive a client-specific score that reflects how much each client needs to adjust for improved generalization on the global validation set, thereby enabling more informed and adaptive federated aggregation. Extensive experiments on both natural and medical image benchmarking datasets, across diverse model architectures, demonstrate that FedVG consistently improves performance, particularly in highly heterogeneous settings. Moreover, FedVG is modular and can be seamlessly integrated with various state-of-the-art FL algorithms, often further improving their results. Our code is available at https://github.com/alinadevkota/FedVG.
>
---
#### [new 123] Uncertainty-Aware Diffusion Model for Multimodal Highway Trajectory Prediction via DDIM Sampling
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的轨迹预测任务，旨在解决多模态未来轨迹预测的准确性与效率问题。提出cVMDx框架，提升预测效率和不确定性感知能力。**

- **链接: [https://arxiv.org/pdf/2602.21319v1](https://arxiv.org/pdf/2602.21319v1)**

> **作者:** Marion Neumeier; Niklas Roßberg; Michael Botsch; Wolfgang Utschick
>
> **备注:** Accepted as a conference paper in IEEE Intelligent Vehicles Symposium (IV) 2026, Detroit, MI, United States
>
> **摘要:** Accurate and uncertainty-aware trajectory prediction remains a core challenge for autonomous driving, driven by complex multi-agent interactions, diverse scene contexts and the inherently stochastic nature of future motion. Diffusion-based generative models have recently shown strong potential for capturing multimodal futures, yet existing approaches such as cVMD suffer from slow sampling, limited exploitation of generative diversity and brittle scenario encodings. This work introduces cVMDx, an enhanced diffusion-based trajectory prediction framework that improves efficiency, robustness and multimodal predictive capability. Through DDIM sampling, cVMDx achieves up to a 100x reduction in inference time, enabling practical multi-sample generation for uncertainty estimation. A fitted Gaussian Mixture Model further provides tractable multimodal predictions from the generated trajectories. In addition, a CVQ-VAE variant is evaluated for scenario encoding. Experiments on the publicly available highD dataset show that cVMDx achieves higher accuracy and significantly improved efficiency over cVMD, enabling fully stochastic, multimodal trajectory prediction.
>
---
## 更新

#### [replaced 001] Dual-Channel Attention Guidance for Training-Free Image Editing Control in Diffusion Transformers
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.18022v2](https://arxiv.org/pdf/2602.18022v2)**

> **作者:** Guandong Li
>
> **摘要:** Training-free control over editing intensity is a critical requirement for diffusion-based image editing models built on the Diffusion Transformer (DiT) architecture. Existing attention manipulation methods focus exclusively on the Key space to modulate attention routing, leaving the Value space -- which governs feature aggregation -- entirely unexploited. In this paper, we first reveal that both Key and Value projections in DiT's multi-modal attention layers exhibit a pronounced bias-delta structure, where token embeddings cluster tightly around a layer-specific bias vector. Building on this observation, we propose Dual-Channel Attention Guidance (DCAG), a training-free framework that simultaneously manipulates both the Key channel (controlling where to attend) and the Value channel (controlling what to aggregate). We provide a theoretical analysis showing that the Key channel operates through the nonlinear softmax function, acting as a coarse control knob, while the Value channel operates through linear weighted summation, serving as a fine-grained complement. Together, the two-dimensional parameter space $(δ_k, δ_v)$ enables more precise editing-fidelity trade-offs than any single-channel method. Extensive experiments on the PIE-Bench benchmark (700 images, 10 editing categories) demonstrate that DCAG consistently outperforms Key-only guidance across all fidelity metrics, with the most significant improvements observed in localized editing tasks such as object deletion (4.9% LPIPS reduction) and object addition (3.2% LPIPS reduction).
>
---
#### [replaced 002] Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11910v3](https://arxiv.org/pdf/2511.11910v3)**

> **作者:** Siyou Li; Huanan Wu; Juexi Shao; Yinghao Ma; Yujian Gan; Yihao Luo; Yuwei Wang; Dong Nie; Lu Wang; Wenqing Wu; Le Zhang; Massimo Poesio; Juntao Yu
>
> **摘要:** Despite the recent advances in the video understanding ability of multimodal large language models (MLLMs), long video understanding remains a challenge. One of the main issues is that the number of vision tokens grows linearly with video length, which causes an explosion in attention cost, memory, and latency. To solve this challenge, we present Query-aware Token Selector (\textbf{QTSplus}), a lightweight yet powerful visual token selection module that serves as an information gate between the vision encoder and LLMs. Given a text query and video tokens, QTSplus dynamically selects the most important visual evidence for the input text query by (i) scoring visual tokens via cross-attention, (ii) \emph{predicting} an instance-specific retention budget based on the complexity of the query, and (iii) \emph{selecting} Top-$n$ tokens with a differentiable straight-through estimator during training and a hard gate at inference. Furthermore, a small re-encoder preserves temporal order using absolute time information, enabling second-level localization while maintaining global coverage. Integrated into Qwen2.5-VL, QTSplus compresses the vision stream by up to \textbf{89\%} and reduces end-to-end latency by \textbf{28\%} on long videos. The evaluation on eight long video understanding benchmarks shows near-parity accuracy overall when compared with the original Qwen models and outperforms the original model by \textbf{+20.5} and \textbf{+5.6} points respectively on TempCompass direction and order accuracies. These results show that QTSplus is an effective, general mechanism for scaling MLLMs to real-world long-video scenarios while preserving task-relevant evidence.
>
---
#### [replaced 003] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.20218v2](https://arxiv.org/pdf/2601.20218v2)**

> **作者:** Haoyou Deng; Keyu Yan; Chaojie Mao; Xiang Wang; Yu Liu; Changxin Gao; Nong Sang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Recent GRPO-based approaches built on flow matching models have shown remarkable improvements in human preference alignment for text-to-image generation. Nevertheless, they still suffer from the sparse reward problem: the terminal reward of the entire denoising trajectory is applied to all intermediate steps, resulting in a mismatch between the global feedback signals and the exact fine-grained contributions at intermediate denoising steps. To address this issue, we introduce \textbf{DenseGRPO}, a novel framework that aligns human preference with dense rewards, which evaluates the fine-grained contribution of each denoising step. Specifically, our approach includes two key components: (1) we propose to predict the step-wise reward gain as dense reward of each denoising step, which applies a reward model on the intermediate clean images via an ODE-based approach. This manner ensures an alignment between feedback signals and the contributions of individual steps, facilitating effective training; and (2) based on the estimated dense rewards, a mismatch drawback between the uniform exploration setting and the time-varying noise intensity in existing GRPO-based methods is revealed, leading to an inappropriate exploration space. Thus, we propose a reward-aware scheme to calibrate the exploration space by adaptively adjusting a timestep-specific stochasticity injection in the SDE sampler, ensuring a suitable exploration space at all timesteps. Extensive experiments on multiple standard benchmarks demonstrate the effectiveness of the proposed DenseGRPO and highlight the critical role of the valid dense rewards in flow matching model alignment.
>
---
#### [replaced 004] RobustGait: Robustness Analysis for Appearance Based Gait Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13065v2](https://arxiv.org/pdf/2511.13065v2)**

> **作者:** Reeshoon Sayera; Akash Kumar; Sirshapan Mitra; Prudvi Kamtam; Yogesh S Rawat
>
> **备注:** IEEE WACV'26 Main Conference
>
> **摘要:** Appearance-based gait recognition have achieved strong performance on controlled datasets, yet systematic evaluation of its robustness to real-world corruptions and silhouette variability remains lacking. We present RobustGait, a framework for fine-grained robustness evaluation of appearance-based gait recognition systems. RobustGait evaluation spans four dimensions: the type of perturbation (digital, environmental, temporal, occlusion), the silhouette extraction method (segmentation and parsing networks), the architectural capacities of gait recognition models, and various deployment scenarios. The benchmark introduces 15 corruption types at 5 severity levels across CASIA-B, CCPG, and SUSTech1K, with in-the-wild validation on MEVID, and evaluates six state-of-the-art gait systems. We came across several exciting insights. First, applying noise at the RGB level better reflects real-world degradation, and reveal how distortions propagate through silhouette extraction to the downstream gait recognition systems. Second, gait accuracy is highly sensitive to silhouette extractor biases, revealing an overlooked source of benchmark bias. Third, robustness is dependent on both the type of perturbation and the architectural design. Finally, we explore robustness-enhancing strategies, showing that noise-aware training and knowledge distillation improve performance and move toward deployment-ready systems. Code is available at https://reeshoon.github.io/robustgaitbenchmark
>
---
#### [replaced 005] Monocular Normal Estimation via Shading Sequence Estimation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.09929v3](https://arxiv.org/pdf/2602.09929v3)**

> **作者:** Zongrui Li; Xinhua Ma; Minghui Hu; Yunqing Zhao; Yingchen Yu; Qian Zheng; Chang Liu; Xudong Jiang; Song Bai
>
> **备注:** Accepted by ICLR 2026 (Oral)
>
> **摘要:** Monocular normal estimation aims to estimate the normal map from a single RGB image of an object under arbitrary lights. Existing methods rely on deep models to directly predict normal maps. However, they often suffer from 3D misalignment: while the estimated normal maps may appear to have a correct appearance, the reconstructed surfaces often fail to align with the geometric details. We argue that this misalignment stems from the current paradigm: the model struggles to distinguish and reconstruct varying geometry represented in normal maps, as the differences in underlying geometry are reflected only through relatively subtle color variations. To address this issue, we propose a new paradigm that reformulates normal estimation as shading sequence estimation, where shading sequences are more sensitive to various geometric information. Building on this paradigm, we present RoSE, a method that leverages image-to-video generative models to predict shading sequences. The predicted shading sequences are then converted into normal maps by solving a simple ordinary least-squares problem. To enhance robustness and better handle complex objects, RoSE is trained on a synthetic dataset, MultiShade, with diverse shapes, materials, and light conditions. Experiments demonstrate that RoSE achieves state-of-the-art performance on real-world benchmark datasets for object-based monocular normal estimation.
>
---
#### [replaced 006] ImpMIA: Leveraging Implicit Bias for Membership Inference Attack
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.10625v3](https://arxiv.org/pdf/2510.10625v3)**

> **作者:** Yuval Golbari; Navve Wasserman; Gal Vardi; Michal Irani
>
> **摘要:** Determining which data samples were used to train a model, known as Membership Inference Attack (MIA), is a well-studied and important problem with implications on data privacy. SotA methods (which are black-box attacks) rely on training many auxiliary reference models to imitate the behavior of the attacked model. As such, they rely on assumptions which rarely hold in real-world settings: (i) the attacker knows the training hyperparameters; (ii) all available non-training samples come from the same distribution as the training data; and (iii) the fraction of training data in the evaluation set is known. We show that removing these assumptions significantly harms the performance of black-box attacks. We introduce ImpMIA, a Membership Inference Attack that exploits the Implicit Bias of neural networks. Building on the maximum-margin implicit bias theory, ImpMIA uses the Karush-Kuhn-Tucker (KKT) optimality conditions to identify training samples -- those whose gradients most strongly reconstruct the trained model's parameters. Our approach is optimization-based, and requires NO training of reference-models, thus removing the need for any knowledge/assumptions regarding the attacked model's training procedure. While ImpMIA is a white-box attack (a setting which assumes access to model weights), this is becoming increasingly realistic given that many models are publicly available (e.g., via Hugging Face). ImpMIA achieves SotA performance compared to both black and white box attacks in settings where only the model weights are known, and a superset of the training data is available.
>
---
#### [replaced 007] LLaDA-MedV: Exploring Large Language Diffusion Models for Biomedical Image Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01617v2](https://arxiv.org/pdf/2508.01617v2)**

> **作者:** Xuanzhao Dong; Wenhui Zhu; Xiwen Chen; Zhipeng Wang; Peijie Qiu; Shao Tang; Xin Li; Yalin Wang
>
> **摘要:** Autoregressive models (ARMs) have long dominated the landscape of biomedical vision-language models (VLMs). Recently, masked diffusion models such as LLaDA have emerged as promising alternatives, yet their application in the biomedical domain remains largely underexplored. To bridge this gap, we introduce LLaDA-MedV, the first large language diffusion model tailored for biomedical image understanding through vision instruction tuning. LLaDA-MedV achieves relative performance gains of 7.855% over LLaVA-Med and 1.867% over LLaDA-V in the open-ended biomedical visual conversation task, and sets new state-of-the-art accuracy on the closed-form subset of three VQA benchmarks: 84.93% on VQA-RAD, 92.31% on SLAKE, and 95.15% on PathVQA. Furthermore, a detailed comparison with LLaVA-Med suggests that LLaDA-MedV is capable of generating reasonably longer responses by explicitly controlling response length, which can lead to more informative outputs. We also conduct an in-depth analysis of both the training and inference stages, highlighting the critical roles of initialization weight selection, fine-tuning strategies, and the interplay between sampling steps and response repetition. The code and model weight is released at https://github.com/LLM-VLM-GSL/LLaDA-MedV.
>
---
#### [replaced 008] TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20903v2](https://arxiv.org/pdf/2602.20903v2)**

> **作者:** Hanshen Zhu; Yuliang Liu; Xuecheng Wu; An-Lan Wang; Hao Feng; Dingkang Yang; Chao Feng; Can Huang; Jingqun Tang; Xiang Bai
>
> **备注:** Accepted by CVPR 2026; Code: https://github.com/CIawevy/TextPecker
>
> **摘要:** Visual Text Rendering (VTR) remains a critical challenge in text-to-image generation, where even advanced models frequently produce text with structural anomalies such as distortion, blurriness, and misalignment. However, we find that leading MLLMs and specialist OCR models largely fail to perceive these structural anomalies, creating a critical bottleneck for both VTR evaluation and RL-based optimization. As a result, even state-of-the-art generators (e.g., SeedDream4.0, Qwen-Image) still struggle to render structurally faithful text. To address this, we propose TextPecker, a plug-and-play structural anomaly perceptive RL strategy that mitigates noisy reward signals and works with any textto-image generator. To enable this capability, we construct a recognition dataset with character-level structural-anomaly annotations and develop a stroke-editing synthesis engine to expand structural-error coverage. Experiments show that TextPecker consistently improves diverse text-to-image models; even on the well-optimized Qwen-Image, it significantly yields average gains of 4% in structural fidelity and 8.7% in semantic alignment for Chinese text rendering, establishing a new state-of-the-art in high-fidelity VTR. Our work fills a gap in VTR optimization, providing a foundational step towards reliable and structural faithful visual text generation.
>
---
#### [replaced 009] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19206v2](https://arxiv.org/pdf/2602.19206v2)**

> **作者:** Zehao Deng; An Liu; Yan Wang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Zero-shot 3D Anomaly Detection is an emerging task that aims to detect anomalies in a target dataset without any target training data, which is particularly important in scenarios constrained by sample scarcity and data privacy concerns. While current methods adapt CLIP by projecting 3D point clouds into 2D representations, they face challenges. The projection inherently loses some geometric details, and the reliance on a single 2D modality provides an incomplete visual understanding, limiting their ability to detect diverse anomaly types. To address these limitations, we propose the Geometry-Aware Prompt and Synergistic View Representation Learning (GS-CLIP) framework, which enables the model to identify geometric anomalies through a two-stage learning process. In stage 1, we dynamically generate text prompts embedded with 3D geometric priors. These prompts contain global shape context and local defect information distilled by our Geometric Defect Distillation Module (GDDM). In stage 2, we introduce Synergistic View Representation Learning architecture that processes rendered and depth images in parallel. A Synergistic Refinement Module (SRM) subsequently fuses the features of both streams, capitalizing on their complementary strengths. Comprehensive experimental results on four large-scale public datasets show that GS-CLIP achieves superior performance in detection. Code can be available at https://github.com/zhushengxinyue/GS-CLIP.
>
---
#### [replaced 010] TIPS Over Tricks: Simple Prompts for Effective Zero-shot Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03594v2](https://arxiv.org/pdf/2602.03594v2)**

> **作者:** Alireza Salehi; Ehsan Karami; Sepehr Noey; Sahand Noey; Makoto Yamada; Reshad Hosseini; Mohammad Sabokrou
>
> **备注:** This is the extended version of the paper accepted in ICASSP'26, which will be publicly available in May. Authors' contributions may vary among the versions
>
> **摘要:** Anomaly detection identifies departures from expected behavior in safety-critical settings. When target-domain normal data are unavailable, zero-shot anomaly detection (ZSAD) leverages vision-language models (VLMs). However, CLIP's coarse image-text alignment limits both localization and detection due to (i) spatial misalignment and (ii) weak sensitivity to fine-grained anomalies; prior works compensate with complex auxiliary modules yet largely overlook the choice of backbone. We revisit the backbone and use TIPS-a VLM trained with spatially aware objectives. While TIPS alleviates CLIP's issues, it exposes a distributional gap between global and local features. We address this with decoupled prompts-fixed for image-level detection and learnable for pixel-level localization-and by injecting local evidence into the global score. Without CLIP-specific tricks, our TIPS-based pipeline improves image-level performance by 1.1-3.9% and pixel-level by 1.5-6.9% across seven industrial datasets, delivering strong generalization with a lean architecture. Code is available at github.com/AlirezaSalehy/Tipsomaly.
>
---
#### [replaced 011] Identifying Memorization of Diffusion Models through $p$-Laplace Analysis: Estimators, Bounds and Applications
- **分类: cs.CV; math.NA**

- **链接: [https://arxiv.org/pdf/2505.08246v2](https://arxiv.org/pdf/2505.08246v2)**

> **作者:** Jonathan Brokman; Itay Gershon; Amit Giloni; Omer Hofman; Roman Vainshtein; Hisashi Kojima; Guy Gilboa
>
> **备注:** This manuscript is a substantially extended version of our SSVM 2025 paper, including significant new theoretical results and additional experiments. It is currently under review as a journal submission
>
> **摘要:** Diffusion models, today's leading image generative models, estimate the score function, i.e. the gradient of the log probability of (perturbed) data samples, without direct access to the underlying probability distribution. This work investigates whether the estimated score function can be leveraged to compute higher-order differentials, namely the p-Laplace operators. We show that these operators can be employed to identify memorized training data. We propose a numerical p-Laplace approximation based on the learned score functions, showing its effectiveness in identifying key features of the probability landscape. Furthermore, theoretical error-bounds to these estimators are proven and demonstrated numerically. We analyze the structured case of Gaussian mixture models, and demonstrate that the results carry-over to text-conditioned image generative models (text-to-image), where memorization identification based on the p-Laplace operator is performed for the first time, showing its advantage on 500 memorized prompts ($\sim$3000 generated images) in a post-generation regime, especially when the conditioning text is unavailable.
>
---
#### [replaced 012] PoseAdapt: Sustainable Human Pose Estimation via Continual Learning Benchmarks and Toolkit
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.20469v3](https://arxiv.org/pdf/2409.20469v3)**

> **作者:** Muhammad Saif Ullah Khan; Didier Stricker
>
> **备注:** Accepted in WACV 2026 Applications Track
>
> **摘要:** Human pose estimators are typically retrained from scratch or naively fine-tuned whenever keypoint sets, sensing modalities, or deployment domains change--an inefficient, compute-intensive practice that rarely matches field constraints. We present PoseAdapt, an open-source framework and benchmark suite for continual pose model adaptation. PoseAdapt defines domain-incremental and class-incremental tracks that simulate realistic changes in density, lighting, and sensing modality, as well as skeleton growth. The toolkit supports two workflows: (i) Strategy Benchmarking, which lets researchers implement continual learning (CL) methods as plugins and evaluate them under standardized protocols; and (ii) Model Adaptation, which allows practitioners to adapt strong pretrained models to new tasks with minimal supervision. We evaluate representative regularization-based methods in single-step and sequential settings. Benchmarks enforce a fixed lightweight backbone, no access to past data, and tight per-step budgets. This isolates adaptation strategy effects, highlighting the difficulty of maintaining accuracy under strict resource limits. PoseAdapt connects modern CL techniques with practical pose estimation needs, enabling adaptable models that improve over time without repeated full retraining.
>
---
#### [replaced 013] Unified Reward Model for Multimodal Understanding and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.05236v2](https://arxiv.org/pdf/2503.05236v2)**

> **作者:** Yibin Wang; Yuhang Zang; Hao Li; Cheng Jin; Jiaqi Wang
>
> **备注:** project page: https://codegoat24.github.io/UnifiedReward/
>
> **摘要:** Recent advances in human preference alignment have significantly improved multimodal generation and understanding. A key approach is to train reward models that provide supervision signals for preference optimization. However, existing reward models are often task-specific, limiting their adaptability across diverse visual applications. We also argue that a reward model that jointly learning to assess multiple vision tasks may foster a synergistic effect, where improved image understanding enhances image generation assessment, and refined image evaluation benefits video assessment through better frame analysis. To this end, this paper proposes UnifiedReward, the first unified reward model for multimodal understanding and generation assessment. It supports both pairwise ranking and pointwise scoring, providing effective reward signals for vision model preference alignment. Specifically, (1) we first train UnifiedReward on our constructed large-scale human preference dataset, which covers both image and video generation/understanding tasks. (2) Then, we leverage it to automatically construct high-quality pairwise preference data from vision models by progressively filtering their outputs through our two-stage strategy, i.e., pair ranking and point sifting. (3) Finally, we use these data to align vision models with human preferences via Direct Preference Optimization (DPO). Experimental results show that jointly learning to assess diverse visual tasks yields substantial mutual benefits. We further apply our pipeline to both vision understanding and generation, achieving consistent improvements across each domain.
>
---
#### [replaced 014] Measuring the Measurers: Quality Evaluation of Hallucination Benchmarks for Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2406.17115v3](https://arxiv.org/pdf/2406.17115v3)**

> **作者:** Bei Yan; Jie Zhang; Zheng Yuan; Shiguang Shan; Xilin Chen
>
> **摘要:** Despite the outstanding performance in multimodal tasks, Large Vision-Language Models (LVLMs) have been plagued by the issue of hallucination, i.e., generating content that is inconsistent with the corresponding visual inputs. While previous works have proposed various benchmarks to evaluate this issue, the quality of these evaluations remains unverified. We observe that some of these benchmarks may produce inconsistent evaluation results across repeated tests or fail to align with human evaluation. To address this, we propose a Hallucination benchmark Quality Measurement framework (HQM), which leverages specific indicators to assess both reliability and validity. Our empirical analysis using HQM reveals and pinpoints potential evaluation issues in existing benchmarks, exposing a critical gap in current hallucination evaluation. To bridge this gap, we propose HQH, a High-Quality Hallucination benchmark, which demonstrates superior reliability and validity under HQM, serving as a credible evaluation tool. Our large-scale evaluation of popular LVLMs on HQH reveals severe hallucination problems, which occur not only in the models' main answer to a question but also in additional analysis. This highlights the necessity for future model improvements to effectively mitigate hallucinations and reduce the associated security risks in real-world applications. Our benchmark is publicly available at https://github.com/HQHBench/HQHBench.
>
---
#### [replaced 015] Grounding-IQA: Grounding Multimodal Language Model for Image Quality Assessment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.17237v4](https://arxiv.org/pdf/2411.17237v4)**

> **作者:** Zheng Chen; Xun Zhang; Wenbo Li; Renjing Pei; Fenglong Song; Xiongkuo Min; Xiaohong Liu; Xin Yuan; Yong Guo; Yulun Zhang
>
> **备注:** Accepted to ICLR 2026. Code is available at: https://github.com/zhengchen1999/Grounding-IQA
>
> **摘要:** The development of multimodal large language models (MLLMs) enables the evaluation of image quality through natural language descriptions. This advancement allows for more detailed assessments. However, these MLLM-based IQA methods primarily rely on general contextual descriptions, sometimes limiting fine-grained quality assessment. To address this limitation, we introduce a new image quality assessment (IQA) task paradigm, **grounding-IQA**. This paradigm integrates multimodal referring and grounding with IQA to realize more fine-grained quality perception, thereby extending existing IQA. Specifically, grounding-IQA comprises two subtasks: grounding-IQA-description (GIQA-DES) and visual question answering (GIQA-VQA). GIQA-DES involves detailed descriptions with precise locations (e.g., bounding boxes), while GIQA-VQA focuses on quality QA for local regions. To realize grounding-IQA, we construct a corresponding dataset, GIQA-160K, through our proposed automated annotation pipeline. Furthermore, we develop a well-designed benchmark, GIQA-Bench. The benchmark evaluates the grounding-IQA performance from three perspectives: description quality, VQA accuracy, and grounding precision. Experiments demonstrate that our proposed method facilitates the more fine-grained IQA application. Code: https://github.com/zhengchen1999/Grounding-IQA.
>
---
#### [replaced 016] Capturing Stable HDR Videos Using a Dual-Camera System
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.06593v3](https://arxiv.org/pdf/2507.06593v3)**

> **作者:** Qianyu Zhang; Bolun Zheng; Lingyu Zhu; Hangjia Pan; Zunjie Zhu; Zongpeng Li; Shiqi Wang
>
> **摘要:** High Dynamic Range (HDR) video acquisition using the alternating exposure (AE) paradigm has garnered significant attention due to its cost-effectiveness with a single consumer camera. However, despite progress driven by deep neural networks, these methods remain prone to temporal flicker in real-world applications due to inter-frame exposure inconsistencies. To address this challenge while maintaining the cost-effectiveness of the AE paradigm, we propose a novel learning-based HDR video generation solution. Specifically, we propose a dual-stream HDR video generation paradigm that decouples temporal luminance anchoring from exposure-variant detail reconstruction, overcoming the inherent limitations of the AE paradigm. To support this, we design an asynchronous dual-camera system (DCS), which enables independent exposure control across two cameras, eliminating the need for synchronization typically required in traditional multi-camera setups. Furthermore, an exposure-adaptive fusion network (EAFNet) is formulated for the DCS system. EAFNet integrates a pre-alignment subnetwork that aligns features across varying exposures, ensuring robust feature extraction for subsequent fusion, an asymmetric cross-feature fusion subnetwork that emphasizes reference-based attention to effectively merge these features across exposures, and a reconstruction subnetwork to mitigate ghosting artifacts and preserve fine details. Extensive experimental evaluations demonstrate that the proposed method achieves state-of-the-art performance across various datasets, showing the remarkable potential of our solution in HDR video reconstruction. The codes and data captured by DCS will be available at https://zqqqyu.github.io/DCS-HDR/.
>
---
#### [replaced 017] MALLVI: A Multi-Agent Framework for Integrated Generalized Robotics Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出MALLVI框架，解决机器人操作中依赖单一模型、缺乏闭环反馈的问题。通过多智能体协作，提升动态环境下的任务成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.16898v3](https://arxiv.org/pdf/2602.16898v3)**

> **作者:** Iman Ahmadi; Mehrshad Taji; Arad Mahdinezhad Kashani; AmirHossein Jadidi; Saina Kashani; Babak Khalaj
>
> **摘要:** Task planning for robotic manipulation with large language models (LLMs) is an emerging area. Prior approaches rely on specialized models, fine tuning, or prompt tuning, and often operate in an open loop manner without robust environmental feedback, making them fragile in dynamic settings. MALLVI presents a Multi Agent Large Language and Vision framework that enables closed-loop feedback driven robotic manipulation. Given a natural language instruction and an image of the environment, MALLVI generates executable atomic actions for a robot manipulator. After action execution, a Vision Language Model (VLM) evaluates environmental feedback and decides whether to repeat the process or proceed to the next step. Rather than using a single model, MALLVI coordinates specialized agents, Decomposer, Localizer, Thinker, and Reflector, to manage perception, localization, reasoning, and high level planning. An optional Descriptor agent provides visual memory of the initial state. The Reflector supports targeted error detection and recovery by reactivating only relevant agents, avoiding full replanning. Experiments in simulation and real-world settings show that iterative closed loop multi agent coordination improves generalization and increases success rates in zero shot manipulation tasks. Code available at https://github.com/iman1234ahmadi/MALLVI .
>
---
#### [replaced 018] PRISM: Programmatic Reasoning with Image Sequence Manipulation for LVLM Jailbreaking
- **分类: cs.CR; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.21540v2](https://arxiv.org/pdf/2507.21540v2)**

> **作者:** Quanchen Zou; Zonghao Ying; Moyang Chen; Wenzhuo Xu; Yisong Xiao; Yakai Li; Deyue Zhang; Dongdong Yang; Zhao Liu; Xiangzheng Zhang
>
> **备注:** There is an error happening in Figure 1, because Figure 1 did not perfectly show the exact overview of the PRISM pipeline
>
> **摘要:** The increasing sophistication of large vision-language models (LVLMs) has been accompanied by advances in safety alignment mechanisms designed to prevent harmful content generation. However, these defenses remain vulnerable to sophisticated adversarial attacks. Existing jailbreak methods typically rely on direct and semantically explicit prompts, overlooking subtle vulnerabilities in how LVLMs compose information over multiple reasoning steps. In this paper, we propose a novel and effective jailbreak framework inspired by Return-Oriented Programming (ROP) techniques from software security. Our approach decomposes a harmful instruction into a sequence of individually benign visual gadgets. A carefully engineered textual prompt directs the sequence of inputs, prompting the model to integrate the benign visual gadgets through its reasoning process to produce a coherent and harmful output. This makes the malicious intent emergent and difficult to detect from any single component. We validate our method through extensive experiments on established benchmarks including SafeBench and MM-SafetyBench, targeting popular LVLMs. Results show that our approach consistently and substantially outperforms existing baselines on state-of-the-art models, achieving near-perfect attack success rates (over 0.90 on SafeBench) and improving ASR by up to 0.39. Our findings reveal a critical and underexplored vulnerability that exploits the compositional reasoning abilities of LVLMs, highlighting the urgent need for defenses that secure the entire reasoning process.
>
---
#### [replaced 019] Transformer-based cardiac substructure segmentation from contrast and non-contrast computed tomography for radiotherapy planning
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.10855v3](https://arxiv.org/pdf/2505.10855v3)**

> **作者:** Aneesh Rangnekar; Nikhil Mankuzhy; Jonas Willmann; Chloe Min Seo Choi; Abraham Wu; Maria Thor; Andreas Rimner; Harini Veeraraghavan
>
> **摘要:** Accurate segmentation of cardiac substructures on computed tomography (CT) scans is essential for radiotherapy planning but typically requires large annotated datasets and often generalizes poorly across imaging protocols and patient variations. This study evaluated whether pretrained transformers enable data-efficient training using a fixed architecture with balanced curriculum learning. A hybrid pretrained transformer-convolutional network (SMIT) was fine-tuned on lung cancer patients (Cohort I, N $=$ 180) imaged in the supine position and validated on 60 held-out Cohort I patients and 65 breast cancer patients (Cohort II) imaged in both supine and prone positions. Two configurations were evaluated: SMIT-Balanced (32 contrast-enhanced CTs and 32 non-contrast CTs) and SMIT-Oracle (180 CTs). Performance was compared with nnU-Net and TotalSegmentator. Segmentation accuracy was assessed primarily using the 95th percentile Hausdorff distance (HD95), with radiation dose and overlap-based metrics evaluated as secondary endpoints. SMIT-Balanced achieved accuracy comparable to SMIT-Oracle despite using 64$\%$ fewer training scans. On Cohort I, HD95 was 6.6 $\pm$ 4.3 mm versus 5.4 $\pm$ 2.6 mm, and on Cohort II, 10.0 $\pm$ 9.4 mm versus 9.4 $\pm$ 9.8 mm, respectively, demonstrating robustness to patient, imaging, and data variations. Radiation dose metrics derived from SMIT segmentations were equivalent to those from manual delineations. Although nnU-Net improved over the publicly trained TotalSegmentator, it showed reduced cross-domain robustness compared to SMIT. Balanced curriculum training reduced labeled data requirements without compromising accuracy relative to the oracle model and maintained robustness across patient and imaging variations. Pretraining reduced dependence on data domain and obviated the need for data-specific architectural reconfiguration required by nnU-Net.
>
---
#### [replaced 020] KD-OCT: Efficient Knowledge Distillation for Clinical-Grade Retinal OCT Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.09069v2](https://arxiv.org/pdf/2512.09069v2)**

> **作者:** Erfan Nourbakhsh; Nasrin Sanjari; Ali Nourbakhsh
>
> **备注:** 7 pages, 5 figures (Accepted at ICSPIS 2025)
>
> **摘要:** Age-related macular degeneration (AMD) and choroidal neovascularization (CNV)-related conditions are leading causes of vision loss worldwide, with optical coherence tomography (OCT) serving as a cornerstone for early detection and management. However, deploying state-of-the-art deep learning models like ConvNeXtV2-Large in clinical settings is hindered by their computational demands. Therefore, it is desirable to develop efficient models that maintain high diagnostic performance while enabling real-time deployment. In this study, a novel knowledge distillation framework, termed KD-OCT, is proposed to compress a high-performance ConvNeXtV2-Large teacher model, enhanced with advanced augmentations, stochastic weight averaging, and focal loss, into a lightweight EfficientNet-B2 student for classifying normal, drusen, and CNV cases. KD-OCT employs real-time distillation with a combined loss balancing soft teacher knowledge transfer and hard ground-truth supervision. The effectiveness of the proposed method is evaluated on the Noor Eye Hospital (NEH) dataset using patient-level cross-validation. Experimental results demonstrate that KD-OCT outperforms comparable multi-scale or feature-fusion OCT classifiers in efficiency-accuracy balance, achieving near-teacher performance with substantial reductions in model size and inference time. Despite the compression, the student model exceeds most existing frameworks, facilitating edge deployment for AMD screening. Code is available at https://github.com/erfan-nourbakhsh/KD-OCT.
>
---
#### [replaced 021] Uncovering Grounding IDs: How External Cues Shape Multimodal Binding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.24072v4](https://arxiv.org/pdf/2509.24072v4)**

> **作者:** Hosein Hasani; Amirmohammad Izadi; Fatemeh Askari; Mobin Bagherian; Sadegh Mohammadian; Mohammad Izadi; Mahdieh Soleymani Baghshah
>
> **备注:** Under review as a conference paper at ICLR 2026
>
> **摘要:** Large vision-language models (LVLMs) show strong performance across multimodal benchmarks but remain limited in structured reasoning and precise grounding. Recent work has demonstrated that adding simple visual structures, such as partitions and annotations, improves accuracy, yet the internal mechanisms underlying these gains remain unclear. We investigate this phenomenon and propose the concept of Grounding IDs, latent identifiers induced by external cues that bind objects to their designated partitions across modalities. Through representation analysis, we find that these identifiers emerge as consistent within-partition alignment in embedding space and reduce the modality gap between image and text. Causal interventions further confirm that these identifiers mediate binding between objects and symbolic cues. We show that Grounding IDs strengthen attention between related components, which in turn improves cross-modal grounding and reduces hallucinations. Taken together, our results identify Grounding IDs as a key symbolic mechanism that explains how external cues enhance multimodal binding and offer both interpretability and practical improvements.
>
---
#### [replaced 022] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20880v2](https://arxiv.org/pdf/2602.20880v2)**

> **作者:** Yongli Xiang; Ziming Hong; Zhaoqing Wang; Xiangyu Zhao; Bo Han; Tongliang Liu
>
> **备注:** CVPR 2026; Code is released at https://github.com/tmllab/2026_CVPR_CASG
>
> **摘要:** Text-to-Image (T2I) diffusion models have demonstrated significant advancements in generating high-quality images, while raising potential safety concerns regarding harmful content generation. Safety-guidance-based methods have been proposed to mitigate harmful outputs by steering generation away from harmful zones, where the zones are averaged across multiple harmful categories based on predefined keywords. However, these approaches fail to capture the complex interplay among different harm categories, leading to "harmful conflicts" where mitigating one type of harm may inadvertently amplify another, thus increasing overall harmful rate. To address this issue, we propose Conflict-aware Adaptive Safety Guidance (CASG), a training-free framework that dynamically identifies and applies the category-aligned safety direction during generation. CASG is composed of two components: (i) Conflict-aware Category Identification (CaCI), which identifies the harmful category most aligned with the model's evolving generative state, and (ii) Conflict-resolving Guidance Application (CrGA), which applies safety steering solely along the identified category to avoid multi-category interference. CASG can be applied to both latent-space and text-space safeguards. Experiments on T2I safety benchmarks demonstrate CASG's state-of-the-art performance, reducing the harmful rate by up to 15.4% compared to existing methods.
>
---
#### [replaced 023] XtraLight-MedMamba for Classification of Neoplastic Tubular Adenomas
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.04819v3](https://arxiv.org/pdf/2602.04819v3)**

> **作者:** Aqsa Sultana; Rayan Afsar; Ahmed Rahu; Surendra P. Singh; Brian Shula; Brandon Combs; Derrick Forchetti; Vijayan K. Asari
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Accurate risk stratification of precancerous polyps during routine colonoscopy screenings is essential for lowering the risk of developing colorectal cancer (CRC). However, assessment of low-grade dysplasia remains limited by subjective histopathologic interpretation. Advancements in digital pathology and deep learning provide new opportunities to identify subtle and fine morphologic patterns associated with malignant progression that may be imperceptible to the human eye. In this work, we propose XtraLight-MedMamba, an ultra-lightweight state-space-based deep learning framework for classifying neoplastic tubular adenomas from whole-slide images (WSIs). The architecture is a blend of ConvNext based shallow feature extractor with parallel vision mamba to efficiently model both long- and short-range dependencies and image generalization. An integration of Spatial and Channel Attention Bridge (SCAB) module enhances multiscale feature extraction, while Fixed Non-Negative Orthogonal Classifier (FNOClassifier) enables substantial parameter reduction and improved generalization. The model was evaluated on a curated dataset acquired from patients with low-grade tubular adenomas, stratified into case and control cohorts based on subsequent CRC development. XtraLight-MedMamba achieved an accuracy of 97.18% and an F1-score of 0.9767 using approximately 32,000 parameters, outperforming transformer-based and conventional Mamba architectures with significantly higher model complexity.
>
---
#### [replaced 024] Tracing Copied Pixels and Regularizing Patch Affinity in Copy Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.17484v2](https://arxiv.org/pdf/2602.17484v2)**

> **作者:** Yichen Lu; Siwei Nie; Minlong Lu; Xudong Yang; Xiaobo Zhang; Peng Zhang
>
> **备注:** Accepted by ICCV2025 Github: https://github.com/eddielyc/CopyNCE
>
> **摘要:** Image Copy Detection (ICD) aims to identify manipulated content between image pairs through robust feature representation learning. While self-supervised learning (SSL) has advanced ICD systems, existing view-level contrastive methods struggle with sophisticated edits due to insufficient fine-grained correspondence learning. We address this limitation by exploiting the inherent geometric traceability in edited content through two key innovations. First, we propose PixTrace - a pixel coordinate tracking module that maintains explicit spatial mappings across editing transformations. Second, we introduce CopyNCE, a geometrically-guided contrastive loss that regularizes patch affinity using overlap ratios derived from PixTrace's verified mappings. Our method bridges pixel-level traceability with patch-level similarity learning, suppressing supervision noise in SSL training. Extensive experiments demonstrate not only state-of-the-art performance (88.7% uAP / 83.9% RP90 for matcher, 72.6% uAP / 68.4% RP90 for descriptor on DISC21 dataset) but also better interpretability over existing methods.
>
---
#### [replaced 025] Learning What Matters: Prioritized Concept Learning via Relative Error-driven Sample Selection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.01085v2](https://arxiv.org/pdf/2506.01085v2)**

> **作者:** Shivam Chandhok; Qian Yang; Oscar Manas; Kanishk Jain; Leonid Sigal; Aishwarya Agrawal
>
> **备注:** CVPR 2026
>
> **摘要:** Instruction tuning has been central to the success of recent vision-language models (VLMs), but it remains expensive-requiring large-scale datasets, high-quality annotations, and large compute budgets. We propose PRioritized cOncept learninG via Relative Error-driven Sample Selection (PROGRESS), a data- and compute-efficient framework that enables VLMs to dynamically select what to learn next based on their evolving needs during training. At each stage, the model tracks its learning progress across skills and selects the most informative samples-those it has not already mastered and that are not too difficult to learn at the current stage of training. This strategy effectively controls skill acquisition and the order in which skills are learned. Specifically, we sample from skills showing the highest learning progress, prioritizing those with the most rapid improvement. Unlike prior methods, PROGRESS requires no upfront answer annotations, queries answers only on a need basis, avoids reliance on additional supervision from auxiliary VLMs, and does not require compute-heavy gradient computations for data selection. Experiments across multiple instruction-tuning datasets of varying scales demonstrate that PROGRESS consistently outperforms state-of-the-art baselines with much less data and supervision. Additionally, we show strong cross-architecture generalization and transferability to larger models, validating PROGRESS as a scalable solution for efficient learning.
>
---
#### [replaced 026] WebGym: Scaling Training Environments for Visual Web Agents with Realistic Tasks
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.02439v4](https://arxiv.org/pdf/2601.02439v4)**

> **作者:** Hao Bai; Alexey Taymanov; Tong Zhang; Aviral Kumar; Spencer Whitehead
>
> **备注:** Added link to tasks on HF
>
> **摘要:** We present WebGym, the largest-to-date open-source environment for training realistic visual web agents. Real websites are non-stationary and diverse, making artificial or small-scale task sets insufficient for robust policy learning. WebGym contains nearly 300,000 tasks with rubric-based evaluations across diverse, real-world websites and difficulty levels. We train agents with a simple reinforcement learning (RL) recipe, which trains on the agent's own interaction traces (rollouts), using task rewards as feedback to guide learning. To enable scaling RL, we speed up sampling of trajectories in WebGym by developing a high-throughput asynchronous rollout system, designed specifically for web agents. Our system achieves a 4-5x rollout speedup compared to naive implementations. Second, we scale the task set breadth, depth, and size, which results in continued performance improvement. Fine-tuning a strong base vision-language model, Qwen-3-VL-8B-Instruct, on WebGym results in an improvement in success rate on an out-of-distribution test set from 26.2% to 42.9%, significantly outperforming agents based on proprietary models such as GPT-4o and GPT-5-Thinking that achieve 27.1% and 29.8%, respectively. This improvement is substantial because our test set consists only of tasks on websites never seen during training, unlike many other prior works on training visual web agents.
>
---
#### [replaced 027] Extracting and Analyzing Rail Crossing Behavior Signatures from Videos using Tensor Methods
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.16057v2](https://arxiv.org/pdf/2602.16057v2)**

> **作者:** Dawon Ahn; Het Patel; Aemal Khattak; Jia Chen; Evangelos E. Papalexakis
>
> **备注:** 6 pages, 10 figures. Accepted at InnovaRail 2026
>
> **摘要:** Railway crossings present complex safety challenges where driver behavior varies by location, time, and conditions. Traditional approaches analyze crossings individually, limiting the ability to identify shared behavioral patterns across locations. We propose a multi-view tensor decomposition framework that captures behavioral similarities across three temporal phases: Approach (warning activation to gate lowering), Waiting (gates down to train passage), and Clearance (train passage to gate raising). We analyze railway crossing videos from multiple locations using TimeSformer embeddings to represent each phase. By constructing phase-specific similarity matrices and applying non-negative symmetric CP decomposition, we discover latent behavioral components with distinct temporal signatures. Our tensor analysis reveals that crossing location appears to be a stronger determinant of behavior patterns than time of day, and that approach-phase behavior provides particularly discriminative signatures. Visualization of the learned component space confirms location-based clustering, with certain crossings forming distinct behavioral clusters. This automated framework enables scalable pattern discovery across multiple crossings, providing a foundation for grouping locations by behavioral similarity to inform targeted safety interventions.
>
---
#### [replaced 028] Hallucination Filtering in Radiology Vision-Language Models Using Discrete Semantic Entropy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.09256v2](https://arxiv.org/pdf/2510.09256v2)**

> **作者:** Patrick Wienholt; Sophie Caselitz; Robert Siepmann; Philipp Bruners; Keno Bressem; Christiane Kuhl; Jakob Nikolas Kather; Sven Nebelung; Daniel Truhn
>
> **备注:** Code is available: https://github.com/TruhnLab/VisionSemanticEntropy
>
> **摘要:** To determine whether using discrete semantic entropy (DSE) to reject questions likely to generate hallucinations can improve the accuracy of black-box vision-language models (VLMs) in radiologic image based visual question answering (VQA). This retrospective study evaluated DSE using two publicly available, de-identified datasets: the VQA-Med 2019 benchmark (500 images with clinical questions and short-text answers) and a diagnostic radiology dataset (206 cases: 60 computed tomography scans, 60 magnetic resonance images, 60 radiographs, 26 angiograms) with corresponding ground-truth diagnoses. GPT-4o and GPT-4.1 (Generative Pretrained Transformer; OpenAI) answered each question 15 times using a temperature of 1.0. Baseline accuracy was determined using low-temperature answers (temperature 0.1). Meaning-equivalent responses were grouped using bidirectional entailment checks, and DSE was computed from the relative frequencies of the resulting semantic clusters. Accuracy was recalculated after excluding questions with DSE > 0.6 or > 0.3. p-values and 95% confidence intervals were obtained using bootstrap resampling and a Bonferroni-corrected threshold of p < .004 for statistical significance. Across 706 image-question pairs, baseline accuracy was 51.7% for GPT-4o and 54.8% for GPT-4.1. After filtering out high-entropy questions (DSE > 0.3), accuracy on the remaining questions was 76.3% (retained questions: 334/706) for GPT-4o and 63.8% (retained questions: 499/706) for GPT-4.1 (both p < .001). Accuracy gains were observed across both datasets and largely remained statistically significant after Bonferroni correction. DSE enables reliable hallucination detection in black-box VLMs by quantifying semantic inconsistency. This method significantly improves diagnostic answer accuracy and offers a filtering strategy for clinical VLM applications.
>
---
#### [replaced 029] Lang2Lift: A Language-Guided Autonomous Forklift System for Outdoor Industrial Pallet Handling
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Lang2Lift系统，解决户外物流中基于语言指令的自动叉车取货问题。通过语言引导的视觉感知与运动控制，实现精准 pallet 捕捉与搬运。**

- **链接: [https://arxiv.org/pdf/2508.15427v2](https://arxiv.org/pdf/2508.15427v2)**

> **作者:** Huy Hoang Nguyen; Johannes Huemer; Markus Murschitz; Tobias Glueck; Minh Nhat Vu; Andreas Kugi
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Automating pallet handling in outdoor logistics and construction environments remains challenging due to unstructured scenes, variable pallet configurations, and changing environmental conditions. In this paper, we present Lang2Lift, an end-to-end language-guided autonomous forklift system designed to support practical pallet pick-up operations in real-world outdoor settings. The system enables operators to specify target pallets using natural language instructions, allowing flexible selection among multiple pallets with different loads and spatial arrangements. Lang2Lift integrates foundation-model-based perception modules with motion planning and control in a closed-loop autonomy pipeline. Language-grounded visual perception is used to identify and segment target pallets, followed by 6D pose estimation and geometric refinement to generate manipulation-feasible insertion poses. The resulting pose estimates are directly coupled with the forklift planning and control modules to execute fully autonomous pallet pick-up maneuvers. We deploy and evaluate the proposed system on the ADAPT autonomous outdoor forklift platform across diverse real-world scenarios, including cluttered scenes, variable lighting, and different payload configurations. Tolerance-based pose evaluation further indicates accuracy sufficient for successful fork insertion. Timing and failure analyses highlight key deployment trade-offs and practical limitations, providing insights into integrating language-guided perception within industrial automation systems. Video demonstrations are available at https://eric-nguyen1402.github.io/lang2lift.github.io/
>
---
#### [replaced 030] Hyperbolic Busemann Neural Networks
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18858v2](https://arxiv.org/pdf/2602.18858v2)**

> **作者:** Ziheng Chen; Bernhard Schölkopf; Nicu Sebe
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Hyperbolic spaces provide a natural geometry for representing hierarchical and tree-structured data due to their exponential volume growth. To leverage these benefits, neural networks require intrinsic and efficient components that operate directly in hyperbolic space. In this work, we lift two core components of neural networks, Multinomial Logistic Regression (MLR) and Fully Connected (FC) layers, into hyperbolic space via Busemann functions, resulting in Busemann MLR (BMLR) and Busemann FC (BFC) layers with a unified mathematical interpretation. BMLR provides compact parameters, a point-to-horosphere distance interpretation, batch-efficient computation, and a Euclidean limit, while BFC generalizes FC and activation layers with comparable complexity. Experiments on image classification, genome sequence learning, node classification, and link prediction demonstrate improvements in effectiveness and efficiency over prior hyperbolic layers. The code is available at https://github.com/GitZH-Chen/HBNN.
>
---
#### [replaced 031] Compression then Matching: An Efficient Pre-training Paradigm for Multimodal Embedding
- **分类: cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2511.08480v2](https://arxiv.org/pdf/2511.08480v2)**

> **作者:** Da Li; Yuxiao Luo; Keping Bi; Jiafeng Guo; Wei Yuan; Biao Yang; Yan Wang; Fan Yang; Tingting Gao; Guorui Zhou
>
> **备注:** Multimodal Embedding
>
> **摘要:** Multimodal large language models advance multimodal representation learning by acquiring transferable semantic embeddings, thereby substantially enhancing performance across a range of vision-language tasks, including cross-modal retrieval, clustering, and classification. An effective embedding is expected to comprehensively preserve the semantic content of the input while simultaneously emphasizing features that are discriminative for downstream tasks. Recent approaches demonstrate that MLLMs can be adapted into competitive embedding models via large-scale contrastive learning, enabling the simultaneous optimization of two complementary objectives. We argue that the two aforementioned objectives can be decoupled: a comprehensive understanding of the input facilitates the embedding model in achieving superior performance in downstream tasks via contrastive learning. In this paper, we propose CoMa, a compressed pre-training phase, which serves as a warm-up stage for contrastive learning. Experiments demonstrate that with only a small amount of pre-training data, we can transform an MLLM into a competitive embedding model. CoMa achieves new state-of-the-art results among MLLMs of comparable size on the MMEB, realizing optimization in both efficiency and effectiveness.
>
---
#### [replaced 032] World Simulation with Video Foundation Models for Physical AI
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出Cosmos-Predict2.5，用于物理AI的世界模拟，解决视频生成与控制问题，通过统一多模态生成和强化学习提升模拟质量。**

- **链接: [https://arxiv.org/pdf/2511.00062v2](https://arxiv.org/pdf/2511.00062v2)**

> **作者:** NVIDIA; :; Arslan Ali; Junjie Bai; Maciej Bala; Yogesh Balaji; Aaron Blakeman; Tiffany Cai; Jiaxin Cao; Tianshi Cao; Elizabeth Cha; Yu-Wei Chao; Prithvijit Chattopadhyay; Mike Chen; Yongxin Chen; Yu Chen; Shuai Cheng; Yin Cui; Jenna Diamond; Yifan Ding; Jiaojiao Fan; Linxi Fan; Liang Feng; Francesco Ferroni; Sanja Fidler; Xiao Fu; Ruiyuan Gao; Yunhao Ge; Jinwei Gu; Aryaman Gupta; Siddharth Gururani; Imad El Hanafi; Ali Hassani; Zekun Hao; Jacob Huffman; Joel Jang; Pooya Jannaty; Jan Kautz; Grace Lam; Xuan Li; Zhaoshuo Li; Maosheng Liao; Chen-Hsuan Lin; Tsung-Yi Lin; Yen-Chen Lin; Huan Ling; Ming-Yu Liu; Xian Liu; Yifan Lu; Alice Luo; Qianli Ma; Hanzi Mao; Kaichun Mo; Seungjun Nah; Yashraj Narang; Abhijeet Panaskar; Lindsey Pavao; Trung Pham; Morteza Ramezanali; Fitsum Reda; Scott Reed; Xuanchi Ren; Haonan Shao; Yue Shen; Stella Shi; Shuran Song; Bartosz Stefaniak; Shangkun Sun; Shitao Tang; Sameena Tasmeen; Lyne Tchapmi; Wei-Cheng Tseng; Jibin Varghese; Andrew Z. Wang; Hao Wang; Haoxiang Wang; Heng Wang; Ting-Chun Wang; Fangyin Wei; Jiashu Xu; Dinghao Yang; Xiaodong Yang; Haotian Ye; Seonghyeon Ye; Xiaohui Zeng; Jing Zhang; Qinsheng Zhang; Kaiwen Zheng; Andrew Zhu; Yuke Zhu
>
> **摘要:** We introduce [Cosmos-Predict2.5], the latest generation of the Cosmos World Foundation Models for Physical AI. Built on a flow-based architecture, [Cosmos-Predict2.5] unifies Text2World, Image2World, and Video2World generation in a single model and leverages [Cosmos-Reason1], a Physical AI vision-language model, to provide richer text grounding and finer control of world simulation. Trained on 200M curated video clips and refined with reinforcement learning-based post-training, [Cosmos-Predict2.5] achieves substantial improvements over [Cosmos-Predict1] in video quality and instruction alignment, with models released at 2B and 14B scales. These capabilities enable more reliable synthetic data generation, policy evaluation, and closed-loop simulation for robotics and autonomous systems. We further extend the family with [Cosmos-Transfer2.5], a control-net style framework for Sim2Real and Real2Real world translation. Despite being 3.5$\times$ smaller than [Cosmos-Transfer1], it delivers higher fidelity and robust long-horizon video generation. Together, these advances establish [Cosmos-Predict2.5] and [Cosmos-Transfer2.5] as versatile tools for scaling embodied intelligence. To accelerate research and deployment in Physical AI, we release source code, pretrained checkpoints, and curated benchmarks under the NVIDIA Open Model License at https://github.com/nvidia-cosmos/cosmos-predict2.5 and https://github.com/nvidia-cosmos/cosmos-transfer2.5. We hope these open resources lower the barrier to adoption and foster innovation in building the next generation of embodied intelligence.
>
---
#### [replaced 033] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出VULCA-Bench，属于视觉-语言模型的文化理解评估任务，旨在解决现有基准对文化深层解读不足的问题。工作包括构建多文化图像-评论数据集及评估框架。**

- **链接: [https://arxiv.org/pdf/2601.07986v3](https://arxiv.org/pdf/2601.07986v3)**

> **作者:** Haorui Yu; Diji Yang; Hang He; Fengrui Zhang; Qiufeng Yi
>
> **备注:** 8 pages, 4 figures, submitted to ACL 2026 Dataset Track
>
> **摘要:** We introduce VULCA-Bench, a multicultural art-critique benchmark for evaluating Vision-Language Models' (VLMs) cultural understanding beyond surface-level visual perception. Existing VLM benchmarks predominantly measure L1-L2 capabilities (object recognition, scene description, and factual question answering) while under-evaluate higher-order cultural interpretation. VULCA-Bench contains 7,410 matched image-critique pairs spanning eight cultural traditions, with Chinese-English bilingual coverage. We operationalise cultural understanding using a five-layer framework (L1-L5, from Visual Perception to Philosophical Aesthetics), instantiated as 225 culture-specific dimensions and supported by expert-written bilingual critiques. Our pilot results indicate that higher-layer reasoning (L3-L5) is consistently more challenging than visual and technical analysis (L1-L2). The dataset, evaluation scripts, and annotation tools are available under CC BY 4.0 at https://github.com/yha9806/VULCA-Bench.
>
---
#### [replaced 034] MIRA: Multimodal Iterative Reasoning Agent for Image Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21087v2](https://arxiv.org/pdf/2511.21087v2)**

> **作者:** Ziyun Zeng; Hang Hua; Jiebo Luo
>
> **摘要:** Instruction-guided image editing offers an intuitive way for users to edit images with natural language. However, diffusion-based editing models often struggle to accurately interpret complex user instructions, especially those involving compositional relationships, contextual cues, or referring expressions, leading to edits that drift semantically or fail to reflect the intended changes. We tackle this problem by proposing MIRA (Multimodal Iterative Reasoning Agent), a lightweight, plug-and-play multimodal reasoning agent that performs editing through an iterative perception-reasoning-action loop, effectively simulating multi-turn human-model interaction processes. Instead of issuing a single prompt or static plan, MIRA predicts atomic edit instructions step by step, using visual feedback to make its decisions. Our 150K multimodal tool-use dataset, MIRA-Editing, combined with a two-stage SFT + GRPO training pipeline, enables MIRA to perform reasoning and editing over complex editing instructions. When paired with open-source image editing models such as Flux.1-Kontext, Step1X-Edit, and Qwen-Image-Edit, MIRA significantly improves both semantic consistency and perceptual quality, achieving performance comparable to or exceeding proprietary systems such as GPT-Image and Nano-Banana.
>
---
#### [replaced 035] Training-free Mixed-Resolution Latent Upsampling for Spatially Accelerated Diffusion Transformers
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.08422v3](https://arxiv.org/pdf/2507.08422v3)**

> **作者:** Wongi Jeong; Kyungryeol Lee; Hoigi Seo; Se Young Chun
>
> **摘要:** Diffusion transformers (DiTs) offer excellent scalability for high-fidelity generation, but their computational overhead poses a great challenge for practical deployment. Existing acceleration methods primarily exploit the temporal dimension, whereas spatial acceleration remains underexplored. In this work, we investigate spatial acceleration for DiTs via latent upsampling. We found that naïve latent upsampling for spatial acceleration introduces artifacts, primarily due to aliasing in high-frequency edge regions and mismatching from noise-timestep discrepancies. Then, based on these findings and analyses, we propose a training-free spatial acceleration framework, dubbed Region-Adaptive Latent Upsampling (RALU), to mitigate those artifacts while achieving spatial acceleration of DiTs by our mixed-resolution latent upsampling. RALU achieves artifact-free, efficient acceleration with early upsampling only on artifact-prone edge regions and noise-timestep matching for different latent resolutions, leading to up to 7.0$\times$ speedup on FLUX-1.dev and 3.0$\times$ on Stable Diffusion 3 with negligible quality degradation. Furthermore, our RALU is complementarily applicable to existing temporal acceleration methods and timestep-distilled models, leading to up to 15.9$\times$ speedup.
>
---
#### [replaced 036] V-Retrver: Evidence-Driven Agentic Reasoning for Universal Multimodal Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.06034v2](https://arxiv.org/pdf/2602.06034v2)**

> **作者:** Dongyang Chen; Chaoyang Wang; Dezhao Su; Xi Xiao; Zeyu Zhang; Jing Xiong; Qing Li; Yuzhang Shang; Shichao Kan
>
> **备注:** Project page: https://github.com/chendy25/V-Retrver
>
> **摘要:** Multimodal Large Language Models (MLLMs) have recently been applied to universal multimodal retrieval, where Chain-of-Thought (CoT) reasoning improves candidate reranking. However, existing approaches remain largely language-driven, relying on static visual encodings and lacking the ability to actively verify fine-grained visual evidence, which often leads to speculative reasoning in visually ambiguous cases. We propose V-Retrver, an evidence-driven retrieval framework that reformulates multimodal retrieval as an agentic reasoning process grounded in visual inspection. V-Retrver enables an MLLM to selectively acquire visual evidence during reasoning via external visual tools, performing a multimodal interleaved reasoning process that alternates between hypothesis generation and targeted visual verification.To train such an evidence-gathering retrieval agent, we adopt a curriculum-based learning strategy combining supervised reasoning activation, rejection-based refinement, and reinforcement learning with an evidence-aligned objective. Experiments across multiple multimodal retrieval benchmarks demonstrate consistent improvements in retrieval accuracy (with 23.0% improvements on average), perception-driven reasoning reliability, and generalization.
>
---
#### [replaced 037] LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.00462v3](https://arxiv.org/pdf/2602.00462v3)**

> **作者:** Benno Krojer; Shravan Nayak; Oscar Mañas; Vaibhav Adlakha; Desmond Elliott; Siva Reddy; Marius Mosbach
>
> **备注:** Updates: small change in interpretability percentage for Qwen-based variants we trained (pre-processing fix), clarification in Section 3 on our method (after feedback from readers), additional appendix section
>
> **摘要:** Transforming a large language model (LLM) into a Vision-Language Model (VLM) can be achieved by mapping the visual tokens from a vision encoder into the embedding space of an LLM. Intriguingly, this mapping can be as simple as a shallow MLP transformation. To understand why LLMs can so readily process visual tokens, we need interpretability methods that reveal what is encoded in the visual token representations at every layer of LLM processing. In this work, we introduce LatentLens, a novel approach for mapping latent representations to descriptions in natural language. LatentLens works by encoding a large text corpus and storing contextualized token representations for each token in that corpus. Visual token representations are then compared to their contextualized textual representations, with the top-k nearest neighbor representations providing descriptions of the visual token. We evaluate this method on 10 different VLMs, showing that commonly used methods, such as LogitLens, substantially underestimate the interpretability of visual tokens. With LatentLens instead, the majority of visual tokens are interpretable across all studied models and all layers. Qualitatively, we show that the descriptions produced by LatentLens are semantically meaningful and provide more fine-grained interpretations for humans compared to individual tokens. More broadly, our findings contribute new evidence on the alignment between vision and language representations, opening up new directions for analyzing latent representations.
>
---
#### [replaced 038] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.20981v2](https://arxiv.org/pdf/2602.20981v2)**

> **作者:** Christian Simon; Masato Ishii; Wei-Yao Wang; Koichi Saito; Akio Hayakawa; Dongseok Shim; Zhi Zhong; Shuyang Cui; Shusuke Takahashi; Takashi Shibuya; Yuki Mitsufuji
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Scaling multimodal alignment between video and audio is challenging, particularly due to limited data and the mismatch between text descriptions and frame-level video information. In this work, we tackle the scaling challenge in multimodal-to-audio generation, examining whether models trained on short instances can generalize to longer ones during testing. To tackle this challenge, we present multimodal hierarchical networks so-called MMHNet, an enhanced extension of state-of-the-art video-to-audio models. Our approach integrates a hierarchical method and non-causal Mamba to support long-form audio generation. Our proposed method significantly improves long audio generation up to more than 5 minutes. We also prove that training short and testing long is possible in the video-to-audio generation tasks without training on the longer durations. We show in our experiments that our proposed method could achieve remarkable results on long-video to audio benchmarks, beating prior works in video-to-audio tasks. Moreover, we showcase our model capability in generating more than 5 minutes, while prior video-to-audio methods fall short in generating with long durations.
>
---
#### [replaced 039] Uni-MMMU: A Massive Multi-discipline Multimodal Unified Benchmark
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13759v2](https://arxiv.org/pdf/2510.13759v2)**

> **作者:** Kai Zou; Ziqi Huang; Yuhao Dong; Shulin Tian; Dian Zheng; Hongbo Liu; Jingwen He; Bin Liu; Yu Qiao; Ziwei Liu
>
> **备注:** Equal contributions from frst three authors. Project page: https://vchitect.github.io/Uni-MMMU-Project/ Code: https://github.com/vchitect/Uni-MMMU
>
> **摘要:** Unified multimodal models aim to jointly enable visual understanding and generation, yet current benchmarks rarely examine their true integration. Existing evaluations either treat the two abilities in isolation or overlook tasks that inherently couple them. To address this gap, we present Uni-MMMU, a comprehensive and discipline-aware benchmark that systematically unfolds the bidirectional synergy between generation and understanding across eight reasoning-centric domains, including science, coding, mathematics, and puzzles. Each task is bidirectionally coupled, demanding models to (i) leverage conceptual understanding to guide precise visual synthesis, or (ii) utilize generation as a cognitive scaffold for analytical reasoning. Uni-MMMU incorporates verifiable intermediate reasoning steps, unique ground truths, and a reproducible scoring protocol for both textual and visual outputs. Through extensive evaluation of state-of-the-art unified, generation-only, and understanding-only models, we reveal substantial performance disparities and cross-modal dependencies, offering new insights into when and how these abilities reinforce one another, and establishing a reliable foundation for advancing unified models.
>
---
#### [replaced 040] Twin Co-Adaptive Dialogue for Progressive Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.14868v2](https://arxiv.org/pdf/2504.14868v2)**

> **作者:** Jianhui Wang; Yangfan He; Yan Zhong; Xinyuan Song; Jiayi Su; Yuheng Feng; Ruoyu Wang; Hongyang He; Wenyu Zhu; Xinhang Yuan; Miao Zhang; Keqin Li; Jiaqi Chen; Tianyu Shi; Xueqian Wang
>
> **摘要:** Modern text-to-image generation systems have enabled the creation of remarkably realistic and high-quality visuals, yet they often falter when handling the inherent ambiguities in user prompts. In this work, we present Twin-Co, a framework that leverages synchronized, co-adaptive dialogue to progressively refine image generation. Instead of a static generation process, Twin-Co employs a dynamic, iterative workflow where an intelligent dialogue agent continuously interacts with the user. Initially, a base image is generated from the user's prompt. Then, through a series of synchronized dialogue exchanges, the system adapts and optimizes the image according to evolving user feedback. The co-adaptive process allows the system to progressively narrow down ambiguities and better align with user intent. Experiments demonstrate that Twin-Co not only enhances user experience by reducing trial-and-error iterations but also improves the quality of the generated images, streamlining creative process across various applications.
>
---
#### [replaced 041] From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20630v2](https://arxiv.org/pdf/2602.20630v2)**

> **作者:** Yepeng Liu; Hao Li; Liwen Yang; Fangzhen Li; Xudi Ge; Yuliang Gu; kuang Gao; Bing Wang; Guang Chen; Hangjun Ye; Yongchao Xu
>
> **备注:** There are unresolved issues regarding authorship and manuscript details. We withdraw this submission to make necessary corrections
>
> **摘要:** Keypoint-based matching is a fundamental component of modern 3D vision systems, such as Structure-from-Motion (SfM) and SLAM. Most existing learning-based methods are trained on image pairs, a paradigm that fails to explicitly optimize for the long-term trackability of keypoints across sequences under challenging viewpoint and illumination changes. In this paper, we reframe keypoint detection as a sequential decision-making problem. We introduce TraqPoint, a novel, end-to-end Reinforcement Learning (RL) framework designed to optimize the \textbf{Tra}ck-\textbf{q}uality (Traq) of keypoints directly on image sequences. Our core innovation is a track-aware reward mechanism that jointly encourages the consistency and distinctiveness of keypoints across multiple views, guided by a policy gradient method. Extensive evaluations on sparse matching benchmarks, including relative pose estimation and 3D reconstruction, demonstrate that TraqPoint significantly outperforms some state-of-the-art (SOTA) keypoint detection and description methods.
>
---
#### [replaced 042] Improving Denoising Diffusion Models via Simultaneous Estimation of Image and Noise
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2310.17167v2](https://arxiv.org/pdf/2310.17167v2)**

> **作者:** Zhenkai Zhang; Krista A. Ehinger; Tom Drummond
>
> **备注:** Published in Proceedings of the 15th Asian Conference on Machine Learning, PMLR 222:1638-1653, 2024
>
> **摘要:** This paper introduces two key contributions aimed at improving the speed and quality of images generated through inverse diffusion processes. The first contribution involves reparameterizing the diffusion process in terms of the angle on a quarter-circular arc between the image and noise, specifically setting the conventional $\displaystyle \sqrt{\barα}=\cos(η)$. This reparameterization eliminates two singularities and allows for the expression of diffusion evolution as a well-behaved ordinary differential equation (ODE). In turn, this allows higher order ODE solvers such as Runge-Kutta methods to be used effectively. The second contribution is to directly estimate both the image ($\mathbf{x}_0$) and noise ($\mathbfε$) using our network, which enables more stable calculations of the update step in the inverse diffusion steps, as accurate estimation of both the image and noise are crucial at different stages of the process. Together with these changes, our model achieves faster generation, with the ability to converge on high-quality images more quickly, and higher quality of the generated images, as measured by metrics such as Frechet Inception Distance (FID), spatial Frechet Inception Distance (sFID), precision, and recall.
>
---
#### [replaced 043] MUGSQA: Novel Multi-Uncertainty-Based Gaussian Splatting Quality Assessment Method, Dataset, and Benchmarks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06830v2](https://arxiv.org/pdf/2511.06830v2)**

> **作者:** Tianang Chen; Jian Jin; Shilv Cai; Zhuangzi Li; Weisi Lin
>
> **备注:** ICASSP 2026
>
> **摘要:** Gaussian Splatting (GS) has recently emerged as a promising technique for 3D object reconstruction, delivering high-quality rendering results with significantly improved reconstruction speed. As variants continue to appear, assessing the perceptual quality of 3D objects reconstructed with different GS-based methods remains an open challenge. To address this issue, we first propose a unified multi-distance subjective quality assessment method that closely mimics human viewing behavior for objects reconstructed with GS-based methods in actual applications, thereby better collecting perceptual experiences. Based on it, we also construct a novel GS quality assessment dataset named MUGSQA, which is constructed considering multiple uncertainties of the input data. These uncertainties include the quantity and resolution of input views, the view distance, and the accuracy of the initial point cloud. Moreover, we construct two benchmarks: one to evaluate the robustness of various GS-based reconstruction methods under multiple uncertainties, and the other to evaluate the performance of existing quality assessment metrics. Our dataset and benchmark code will be released soon.
>
---
#### [replaced 044] Aerial Vision-Language Navigation with a Unified Framework for Spatial, Temporal and Embodied Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.08639v2](https://arxiv.org/pdf/2512.08639v2)**

> **作者:** Huilin Xu; Zhuoyang Liu; Yixiang Luomei; Feng Xu
>
> **备注:** Under Review, 15 pages, 11 figures
>
> **摘要:** Aerial Vision-and-Language Navigation (VLN) aims to enable unmanned aerial vehicles (UAVs) to interpret natural language instructions and navigate complex urban environments using onboard visual observation. This task holds promise for real-world applications such as low-altitude inspection, search-and-rescue, and autonomous aerial delivery. Existing methods often rely on panoramic images, depth inputs, or odometry to support spatial reasoning and action planning. These requirements increase system cost and integration complexity, thus hindering practical deployment for lightweight UAVs. We present a unified aerial VLN framework that operates solely on egocentric monocular RGB observations and natural language instructions. The model formulates navigation as a next-token prediction problem, jointly optimizing spatial perception, trajectory reasoning, and action prediction through prompt-guided multi-task learning. Moreover, we propose a keyframe selection strategy to reduce visual redundancy by retaining semantically informative frames, along with an action merging and label reweighting mechanism that mitigates long-tailed supervision imbalance and facilitates stable multi-task co-training. Extensive experiments on the AerialVLN and OpenFly benchmark validate the effectiveness of our method. Under the challenging monocular RGB-only setting, our model achieves strong results across both seen and unseen environments. It significantly outperforms existing RGB-only baselines and narrows the performance gap with state-of-the-art panoramic RGB-D counterparts. Comprehensive ablation studies further demonstrate the contribution of our task design and architectural choices.
>
---
#### [replaced 045] HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出HetroD数据集和基准，用于解决异构交通中自动驾驶的挑战，聚焦于脆弱道路使用者的行为建模与预测。**

- **链接: [https://arxiv.org/pdf/2602.03447v2](https://arxiv.org/pdf/2602.03447v2)**

> **作者:** Yu-Hsiang Chen; Wei-Jer Chang; Christian Kotulla; Thomas Keutgens; Steffen Runde; Tobias Moers; Christoph Klas; Wei Zhan; Masayoshi Tomizuka; Yi-Ting Chen
>
> **备注:** IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** We present HetroD, a dataset and benchmark for developing autonomous driving systems in heterogeneous environments. HetroD targets the critical challenge of navi- gating real-world heterogeneous traffic dominated by vulner- able road users (VRUs), including pedestrians, cyclists, and motorcyclists that interact with vehicles. These mixed agent types exhibit complex behaviors such as hook turns, lane splitting, and informal right-of-way negotiation. Such behaviors pose significant challenges for autonomous vehicles but remain underrepresented in existing datasets focused on structured, lane-disciplined traffic. To bridge the gap, we collect a large- scale drone-based dataset to provide a holistic observation of traffic scenes with centimeter-accurate annotations, HD maps, and traffic signal states. We further develop a modular toolkit for extracting per-agent scenarios to support downstream task development. In total, the dataset comprises over 65.4k high- fidelity agent trajectories, 70% of which are from VRUs. HetroD supports modeling of VRU behaviors in dense, het- erogeneous traffic and provides standardized benchmarks for forecasting, planning, and simulation tasks. Evaluation results reveal that state-of-the-art prediction and planning models struggle with the challenges presented by our dataset: they fail to predict lateral VRU movements, cannot handle unstructured maneuvers, and exhibit limited performance in dense and multi-agent scenarios, highlighting the need for more robust approaches to heterogeneous traffic. See our project page for more examples: https://hetroddata.github.io/HetroD/
>
---
#### [replaced 046] Any Image Restoration via Efficient Spatial-Frequency Degradation Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.14249v2](https://arxiv.org/pdf/2504.14249v2)**

> **作者:** Bin Ren; Eduard Zamfir; Zongwei Wu; Yawei Li; Yidi Li; Danda Pani Paudel; Radu Timofte; Ming-Hsuan Yang; Luc Van Gool; Nicu Sebe
>
> **备注:** Efficient All-in-One Image Restoration, Accepted by TMLR in 2026
>
> **摘要:** Restoring multiple degradations efficiently via just one model has become increasingly significant and impactful, especially with the proliferation of mobile devices. Traditional solutions typically involve training dedicated models per degradation, resulting in inefficiency and redundancy. More recent approaches either introduce additional modules to learn visual prompts, significantly increasing the size of the model, or incorporate cross modal transfer from large language models trained on vast datasets, adding complexity to the system architecture. In contrast, our approach, termed AnyIR, takes a unified path that leverages inherent similarity across various degradations to enable both efficient and comprehensive restoration through a joint embedding mechanism, without scaling up the model or relying on large language models. Specifically, we examine the sublatent space of each input, identifying key components and reweighting them first in a gated manner. To unify intrinsic degradation awareness with contextualized attention, we propose a spatial frequency parallel fusion strategy that strengthens spatially informed local global interactions and enriches restoration fidelity from the frequency domain. Comprehensive evaluations across four all-in-one restoration benchmarks demonstrate that AnyIR attains SOTA performance while reducing model parameters by 84% and FLOPs by 80% relative to the baseline. These results highlight the potential of AnyIR as an effective and lightweight solution for further all in one image restoration. Our code is available at: https://github.com/Amazingren/AnyIR.
>
---
#### [replaced 047] FigEx2: Visual-Conditioned Panel Detection and Captioning for Scientific Compound Figures
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出FigEx2，解决科学复合图中面板检测与描述生成问题，通过视觉条件框架实现精准定位和高质量 captions 生成。**

- **链接: [https://arxiv.org/pdf/2601.08026v3](https://arxiv.org/pdf/2601.08026v3)**

> **作者:** Jifeng Song; Arun Das; Pan Wang; Hui Ji; Kun Zhao; Yufei Huang
>
> **摘要:** Scientific compound figures combine multiple labeled panels into a single image, but captions in real pipelines are often missing or only provide figure-level summaries, making panel-level understanding difficult. In this paper, we propose FigEx2, visual-conditioned framework that localizes panels and generates panel-wise captions directly from the compound figure. To mitigate the impact of diverse phrasing in open-ended captioning, we introduce a noise-aware gated fusion module that adaptively filters token-level features to stabilize the detection query space. Furthermore, we employ a staged optimization strategy combining supervised learning with reinforcement learning (RL), utilizing CLIP-based alignment and BERTScore-based semantic rewards to enforce strict multimodal consistency. To support high-quality supervision, we curate BioSci-Fig-Cap, a refined benchmark for panel-level grounding, alongside cross-disciplinary test suites in physics and chemistry. Experimental results demonstrate that FigEx2 achieves a superior 0.726 mAP@0.5:0.95 for detection and significantly outperforms Qwen3-VL-8B by 0.51 in METEOR and 0.24 in BERTScore. Notably, FigEx2 exhibits remarkable zero-shot transferability to out-of-distribution scientific domains without any fine-tuning.
>
---
#### [replaced 048] NTK-Guided Implicit Neural Teaching
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15487v2](https://arxiv.org/pdf/2511.15487v2)**

> **作者:** Chen Zhang; Wei Zuo; Bingyang Cheng; Yikun Wang; Wei-Bin Kou; Yik Chung WU; Ngai Wong
>
> **备注:** CVPR 2026 (18 pages, 10 figures)
>
> **摘要:** Implicit Neural Representations (INRs) parameterize continuous signals via multilayer perceptrons (MLPs), enabling compact, resolution-independent modeling for tasks like image, audio, and 3D reconstruction. However, fitting high-resolution signals demands optimizing over millions of coordinates, incurring prohibitive computational costs. To address it, we propose NTK-Guided Implicit Neural Teaching (NINT), which accelerates training by dynamically selecting coordinates that maximize global functional updates. Leveraging the Neural Tangent Kernel (NTK), NINT scores examples by the norm of their NTK-augmented loss gradients, capturing both fitting errors and heterogeneous leverage (self-influence and cross-coordinate coupling). This dual consideration enables faster convergence compared to existing methods. Through extensive experiments, we demonstrate that NINT significantly reduces training time by nearly half while maintaining or improving representation quality, establishing state-of-the-art acceleration among recent sampling-based strategies.
>
---
#### [replaced 049] Rectifying Geometry-Induced Similarity Distortions for Real-World Aerial-Ground Person Re-Identification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.21405v2](https://arxiv.org/pdf/2601.21405v2)**

> **作者:** Kailash A. Hambarde; Hugo Proença
>
> **摘要:** Aerial-ground person re-identification (AG-ReID) is fundamentally challenged by extreme viewpoint and distance discrepancies between aerial and ground cameras, which induce severe geometric distortions and invalidate the assumption of a shared similarity space across views. Existing methods primarily rely on geometry-aware feature learning or appearance-conditioned prompting, while implicitly assuming that the geometry-invariant dot-product similarity used in attention mechanisms remains reliable under large viewpoint and scale variations. We argue that this assumption does not hold. Extreme camera geometry systematically distorts the query-key similarity space and degrades attention-based matching, even when feature representations are partially aligned. To address this issue, we introduce Geometry-Induced Query-Key Transformation (GIQT), a lightweight low-rank module that explicitly rectifies the similarity space by conditioning query-key interactions on camera geometry. Rather than modifying feature representations or the attention formulation itself, GIQT adapts the similarity computation to compensate for dominant geometry-induced anisotropic distortions. Building on this local similarity rectification, we further incorporate a geometry-conditioned prompt generation mechanism that provides global, view-adaptive representation priors derived directly from camera geometry.Experiments on four aerial-ground person re-identification benchmarks demonstrate that the proposed framework consistently improves robustness under extreme and previously unseen geometric conditions, while introducing minimal computational overhead compared to state-of-the-art methods.
>
---
#### [replaced 050] Renaissance: Investigating the Pretraining of Vision-Language Encoders
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言任务，旨在研究VL编码器的预训练方法。通过引入Renaissance框架，探索模型冻结和结构设计对性能的影响。**

- **链接: [https://arxiv.org/pdf/2411.06657v2](https://arxiv.org/pdf/2411.06657v2)**

> **作者:** Clayton Fields; Casey Kennington
>
> **备注:** 9 pages
>
> **摘要:** In the past several years there has been an explosion of available models for vision-language (VL) tasks. Unfortunately, the literature still leaves open a number of questions related to best practices in designing and training such models. Additionally, the limited programming tools available for modeling make conducting VL research more difficult than necessary. In this paper, we seek to answer several questions related to the pretraining of VL encoders through meta-analysis. To conduct these experiments, we introduce a VL evaluation framework called Renaissance. In our first set of experiments, we show that we can save significant compute at little to no cost to downstream performance, by freezing large parts of VL models during pretraining. In our second set of experiments, we examine the effect of basing a VL transformer on a vision model versus a text model. Renaissance offers a great deal of flexibility in creating, training and evaluating transformer encoders for VL modeling. Its source code will be made publicly available upon publication. The source code for Renaissance can be found at https://github.com/bsu-slim/renaissance.
>
---
#### [replaced 051] Progressive Checkerboards for Autoregressive Multiscale Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03811v2](https://arxiv.org/pdf/2602.03811v2)**

> **作者:** David Eigen
>
> **摘要:** A key challenge in autoregressive image generation is to efficiently sample independent locations in parallel, while still modeling mutual dependencies with serial conditioning. Some recent works have addressed this by conditioning between scales in a multiscale pyramid. Others have looked at parallelizing samples in a single image using regular partitions or randomized orders. In this work we examine a flexible, fixed ordering based on progressive checkerboards for multiscale autoregressive image generation. Our ordering draws samples in parallel from evenly spaced regions at each scale, maintaining full balance in all levels of a quadtree subdivision at each step. This enables effective conditioning both between and within scales. Intriguingly, we find evidence that in our balanced setting, a wide range of scale-up factors lead to similar results, so long as the total number of serial steps is constant. On class-conditional ImageNet, our method achieves competitive performance compared to recent state-of-the-art autoregressive systems with like model capacity, using fewer sampling steps.
>
---
#### [replaced 052] Beyond Calibration: Confounding Pathology Limits Foundation Model Specificity in Abdominal Trauma CT
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.10359v2](https://arxiv.org/pdf/2602.10359v2)**

> **作者:** Jineel H Raythatha; Shuchang Ye; Jeremy Hsu; Jinman Kim
>
> **备注:** 26 pages, 4 figures, 4 tables
>
> **摘要:** Purpose: Translating foundation models into clinical practice requires evaluating their performance under compound distribution shift, where severe class imbalance coexists with heterogeneous imaging appearances. This challenge is relevant for traumatic bowel injury, a rare but high-mortality diagnosis. We investigated whether specificity deficits in foundation models are associated with heterogeneity in the negative class. Methods: This retrospective study used the multi-institutional, RSNA Abdominal Traumatic Injury CT dataset (2019-2023), comprising scans from 23 centres. Two foundation models (MedCLIP, zero-shot; RadDINO, linear probe) were compared against three task-specific approaches (CNN, Transformer, Ensemble). Models were trained on 3,147 patients (2.3% bowel injury prevalence) and evaluated on an enriched 100-patient test set. To isolate negative-class effects, specificity was assessed in patients without bowel injury who had concurrent solid organ injury (n=58) versus no abdominal pathology (n=50). Results: Foundation models achieved equivalent discrimination to task-specific models (AUC, 0.64-0.68 versus 0.58-0.64) with higher sensitivity (79-91% vs 41-74%) but lower specificity (33-50% vs 50-88%). All models demonstrated high specificity in patients without abdominal pathology (84-100%). When solid organ injuries were present, specificity declined substantially for foundation models (50-51 percentage points) compared with smaller reductions of 12-41 percentage points for task-specific models. Conclusion: Foundation models matched task-specific discrimination without task-specific training, but their specificity deficits were driven primarily by confounding negative-class heterogeneity rather than prevalence alone. Susceptibility to negative-class heterogeneity decreased progressively with labelled training, suggesting adaptation is required before clinical implementation.
>
---
#### [replaced 053] Chain-of-Thought Compression Should Not Be Blind: V-Skip for Efficient Multimodal Reasoning via Dual-Path Anchoring
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，解决CoT推理延迟高问题。通过V-Skip方法，结合语言和视觉信息优化token压缩，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2601.13879v3](https://arxiv.org/pdf/2601.13879v3)**

> **作者:** Dongxu Zhang; Yiding Sun; Cheng Tan; Wenbiao Yan; Ning Yang; Jihua Zhu; Haijun Zhang
>
> **摘要:** While Chain-of-Thought (CoT) reasoning significantly enhances the performance of Multimodal Large Language Models (MLLMs), its autoregressive nature incurs prohibitive latency constraints. Current efforts to mitigate this via token compression often fail by blindly applying text-centric metrics to multimodal contexts. We identify a critical failure mode termed Visual Amnesia, where linguistically redundant tokens are erroneously pruned, leading to hallucinations. To address this, we introduce V-Skip that reformulates token pruning as a Visual-Anchored Information Bottleneck (VA-IB) optimization problem. V-Skip employs a dual-path gating mechanism that weighs token importance through both linguistic surprisal and cross-modal attention flow, effectively rescuing visually salient anchors. Extensive experiments on Qwen2-VL and Llama-3.2 families demonstrate that V-Skip achieves a $2.9\times$ speedup with negligible accuracy loss. Specifically, it preserves fine-grained visual details, outperforming other baselines over 30\% on the DocVQA.
>
---
#### [replaced 054] TRACE: Your Diffusion Model is Secretly an Instance Edge Detector
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.07982v4](https://arxiv.org/pdf/2503.07982v4)**

> **作者:** Sanghyun Jo; Ziseok Lee; Wooyeol Lee; Jonghyun Choi; Jaesik Park; Kyungsu Kim
>
> **备注:** Accepted to ICLR 2026 (Oral)
>
> **摘要:** High-quality instance and panoptic segmentation has traditionally relied on dense instance-level annotations such as masks, boxes, or points, which are costly, inconsistent, and difficult to scale. Unsupervised and weakly-supervised approaches reduce this burden but remain constrained by semantic backbone constraints and human bias, often producing merged or fragmented outputs. We present TRACE (TRAnsforming diffusion Cues to instance Edges), showing that text-to-image diffusion models secretly function as instance edge annotators. TRACE identifies the Instance Emergence Point (IEP) where object boundaries first appear in self-attention maps, extracts boundaries through Attention Boundary Divergence (ABDiv), and distills them into a lightweight one-step edge decoder. This design removes the need for per-image diffusion inversion, achieving 81x faster inference while producing sharper and more connected boundaries. On the COCO benchmark, TRACE improves unsupervised instance segmentation by +5.1 AP, and in tag-supervised panoptic segmentation it outperforms point-supervised baselines by +1.7 PQ without using any instance-level labels. These results reveal that diffusion models encode hidden instance boundary priors, and that decoding these signals offers a practical and scalable alternative to costly manual annotation. Project Page: https://shjo-april.github.io/TRACE/
>
---
#### [replaced 055] TimeBlind: A Spatio-Temporal Compositionality Benchmark for Video LLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.00288v3](https://arxiv.org/pdf/2602.00288v3)**

> **作者:** Baiqi Li; Kangyi Zhao; Ce Zhang; Chancharik Mitra; Jean de Dieu Nyandwi; Gedas Bertasius
>
> **备注:** For code and data, see https://baiqi-li.github.io/timeblind_project/
>
> **摘要:** Fine-grained spatio-temporal understanding is essential for video reasoning and embodied AI. Yet, while Multimodal Large Language Models (MLLMs) master static semantics, their grasp of temporal dynamics remains brittle. We present TimeBlind, a diagnostic benchmark for compositional spatio-temporal understanding. Inspired by cognitive science, TimeBlind categorizes fine-grained temporal understanding into three levels: recognizing atomic events, characterizing event properties, and reasoning about event interdependencies. Unlike benchmarks that conflate recognition with temporal reasoning, TimeBlind leverages a minimal-pairs paradigm: video pairs share identical static visual content but differ solely in temporal structure, utilizing complementary questions to neutralize language priors. Evaluating over 20 state-of-the-art MLLMs (e.g., GPT-5, Gemini 3 Pro) on 600 curated instances (2400 video-question pairs), reveals that the Instance Accuracy (correctly distinguishing both videos in a pair) of the best performing MLLM is only 48.2%, far below the human performance (98.2%). These results demonstrate that even frontier models rely heavily on static visual shortcuts rather than genuine temporal logic, positioning TimeBlind as a vital diagnostic tool for next-generation video understanding. Dataset and code are available at https://baiqi-li.github.io/timeblind_project/ .
>
---
#### [replaced 056] InsightX Agent: An LMM-based Agentic Framework with Integrated Tools for Reliable X-ray NDT Analysis
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.14899v3](https://arxiv.org/pdf/2507.14899v3)**

> **作者:** Jiale Liu; Huan Wang; Yue Zhang; Xiaoyu Luo; Jiaxiang Hu; Zhiliang Liu; Min Xie
>
> **摘要:** Non-destructive testing (NDT), particularly X-ray inspection, is vital for industrial quality assurance, yet existing deep-learning-based approaches often lack interactivity, interpretability, and the capacity for critical self-assessment, limiting their reliability and operator trust. To address these shortcomings, this paper proposes InsightX Agent, a novel LMM-based agentic framework designed to deliver reliable, interpretable, and interactive X-ray NDT analysis. Unlike typical sequential pipelines, InsightX Agent positions a Large Multimodal Model (LMM) as a central orchestrator, coordinating between the Sparse Deformable Multi-Scale Detector (SDMSD) and the Evidence-Grounded Reflection (EGR) tool. The SDMSD generates dense defect region proposals from multi-scale feature maps and sparsifies them through Non-Maximum Suppression (NMS), optimizing detection of small, dense targets in X-ray images while maintaining computational efficiency. The EGR tool guides the LMM agent through a chain-of-thought-inspired review process, incorporating context assessment, individual defect analysis, false positive elimination, confidence recalibration and quality assurance to validate and refine the SDMSD's initial proposals. By strategically employing and intelligently using tools, InsightX Agent moves beyond passive data processing to active reasoning, enhancing diagnostic reliability and providing interpretations that integrate diverse information sources. Experimental evaluations on the GDXray+ dataset demonstrate that InsightX Agent not only achieves a high object detection F1-score of 96.54\% but also offers significantly improved interpretability and trustworthiness in its analyses, highlighting the transformative potential of LMM-based agentic frameworks for industrial inspection tasks.
>
---
#### [replaced 057] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决传统方法中空间信息丢失和计算冗余的问题。通过引入双隐式神经记忆，提升导航效率与性能。**

- **链接: [https://arxiv.org/pdf/2509.22548v2](https://arxiv.org/pdf/2509.22548v2)**

> **作者:** Shuang Zeng; Dekang Qi; Xinyuan Chang; Feng Xiong; Shichao Xie; Xiaolong Wu; Shiyi Liang; Mu Xu; Xing Wei; Ning Guo
>
> **备注:** Accepted to ICLR 2026. Project page: https://miv-xjtu.github.io/JanusVLN.github.io/
>
> **摘要:** Vision-and-Language Navigation requires an embodied agent to navigate through unseen environments, guided by natural language instructions and a continuous video stream. Recent advances in VLN have been driven by the powerful semantic understanding of Multimodal Large Language Models. However, these methods typically rely on explicit semantic memory, such as building textual cognitive maps or storing historical visual frames. This type of method suffers from spatial information loss, computational redundancy, and memory bloat, which impede efficient navigation. Inspired by the implicit scene representation in human navigation, analogous to the left brain's semantic understanding and the right brain's spatial cognition, we propose JanusVLN, a novel VLN framework featuring a dual implicit neural memory that models spatial-geometric and visual-semantic memory as separate, compact, and fixed-size neural representations. This framework first extends the MLLM to incorporate 3D prior knowledge from the spatial-geometric encoder, thereby enhancing the spatial reasoning capabilities of models based solely on RGB input. Then, the historical key-value caches from the spatial-geometric and visual-semantic encoders are constructed into a dual implicit memory. By retaining only the KVs of tokens in the initial and sliding window, redundant computation is avoided, enabling efficient incremental updates. Extensive experiments demonstrate that JanusVLN outperforms over 20 recent methods to achieve SOTA performance. For example, the success rate improves by 10.5-35.5 compared to methods using multiple data types as input and by 3.6-10.8 compared to methods using more RGB training data. This indicates that the proposed dual implicit neural memory, as a novel paradigm, explores promising new directions for future VLN research. Ours project page: https://miv-xjtu.github.io/JanusVLN.github.io/.
>
---
#### [replaced 058] PD-VLA: Accelerating Vision-Language-Action Model Integrated with Action Chunking via Parallel Decoding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在引入动作分块后推理效率低的问题。提出PD-VLA框架，通过并行解码提升效率，保持性能。**

- **链接: [https://arxiv.org/pdf/2503.02310v2](https://arxiv.org/pdf/2503.02310v2)**

> **作者:** Wenxuan Song; Jiayi Chen; Pengxiang Ding; Han Zhao; Wei Zhao; Zhide Zhong; Zongyuan Ge; Zhijun Li; Donglin Wang; Jun Ma; Lujia Wang; Haoang Li
>
> **备注:** Accepted by IROS 2025, updated results on LIBERO
>
> **摘要:** Vision-Language-Action (VLA) models demonstrate remarkable potential for generalizable robotic manipulation. The performance of VLA models can be improved by integrating with action chunking, a critical technique for effective control. However, action chunking linearly scales up action dimensions in VLA models with increased chunking sizes. This reduces the inference efficiency. To tackle this problem, we propose PD-VLA, the first parallel decoding framework for VLA models integrated with action chunking. Our framework reformulates autoregressive decoding as a nonlinear system solved by parallel fixed-point iterations. This approach preserves model performance with mathematical guarantees while significantly improving decoding speed. In addition, it enables training-free acceleration without architectural changes, as well as seamless synergy with existing acceleration techniques. Extensive simulations validate that our PD-VLA maintains competitive success rates while achieving 2.52 times execution frequency on manipulators (with 7 degrees of freedom) compared with the fundamental VLA model. Furthermore, we experimentally identify the most effective settings for acceleration. Finally, real-world experiments validate its high applicability across different tasks.
>
---
#### [replaced 059] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20205v2](https://arxiv.org/pdf/2602.20205v2)**

> **作者:** Xiwen Chen; Wenhui Zhu; Gen Li; Xuanzhao Dong; Yujian Xiong; Hao Wang; Peijie Qiu; Qingquan Song; Zhipeng Wang; Shao Tang; Yalin Wang; Abolfazl Razi
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Multi-modal large language models (MLLMs) achieve strong visual-language reasoning but suffer from high inference cost due to redundant visual tokens. Recent work explores visual token pruning to accelerate inference, while existing pruning methods overlook the underlying distributional structure of visual representations. We propose OTPrune, a training-free framework that formulates pruning as distribution alignment via optimal transport (OT). By minimizing the 2-Wasserstein distance between the full and pruned token distributions, OTPrune preserves both local diversity and global representativeness while reducing inference cost. Moreover, we derive a tractable submodular objective that enables efficient optimization, and theoretically prove its monotonicity and submodularity, providing a principled foundation for stable and efficient pruning. We further provide a comprehensive analysis that explains how distributional alignment contributes to stable and semantically faithful pruning. Comprehensive experiments on wider benchmarks demonstrate that OTPrune achieves superior performance-efficiency tradeoffs compared to state-of-the-art methods. The code is available at https://github.com/xiwenc1/OTPrune.
>
---
#### [replaced 060] A Comprehensive Survey on Underwater Image Enhancement Based on Deep Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.19684v4](https://arxiv.org/pdf/2405.19684v4)**

> **作者:** Xiaofeng Cong; Yu Zhao; Jie Gui; Junming Hou; Dacheng Tao
>
> **备注:** This article has been accepted for publication in IEEE Transactions on Emerging Topics in Computational Intelligence
>
> **摘要:** Underwater image enhancement (UIE) presents a significant challenge within computer vision research. Despite the development of numerous UIE algorithms, a thorough and systematic review is still absent. To foster future advancements, we provide a detailed overview of the UIE task from several perspectives. Firstly, we introduce the physical models, data construction processes, evaluation metrics, and loss functions. Secondly, we categorize and discuss recent algorithms based on their contributions, considering six aspects: network architecture, learning strategy, learning stage, auxiliary tasks, domain perspective, and disentanglement fusion. Thirdly, due to the varying experimental setups in the existing literature, a comprehensive and unbiased comparison is currently unavailable. To address this, we perform both quantitative and qualitative evaluations of state-of-the-art algorithms across multiple benchmark datasets. Lastly, we identify key areas for future research in UIE. A collection of resources for UIE can be found at {https://github.com/YuZhao1999/UIE}.
>
---
#### [replaced 061] MedicalPatchNet: A Patch-Based Self-Explainable AI Architecture for Chest X-ray Classification
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.07477v2](https://arxiv.org/pdf/2509.07477v2)**

> **作者:** Patrick Wienholt; Christiane Kuhl; Jakob Nikolas Kather; Sven Nebelung; Daniel Truhn
>
> **备注:** 28 pages, 12 figures
>
> **摘要:** Deep neural networks excel in radiological image classification but frequently suffer from poor interpretability, limiting clinical acceptance. We present MedicalPatchNet, an inherently self-explainable architecture for chest X-ray classification that transparently attributes decisions to distinct image regions. MedicalPatchNet splits images into non-overlapping patches, independently classifies each patch, and aggregates predictions, enabling intuitive visualization of each patch's diagnostic contribution without post-hoc techniques. Trained on the CheXpert dataset (223,414 images), MedicalPatchNet matches the classification performance (AUROC 0.907 vs. 0.908) of EfficientNetV2-S, while improving interpretability: MedicalPatchNet demonstrates improved interpretability with higher pathology localization accuracy (mean hit-rate 0.485 vs. 0.376 with Grad-CAM) on the CheXlocalize dataset. By providing explicit, reliable explanations accessible even to non-AI experts, MedicalPatchNet mitigates risks associated with shortcut learning, thus improving clinical trust. Our model is publicly available with reproducible training and inference scripts and contributes to safer, explainable AI-assisted diagnostics across medical imaging domains. We make the code publicly available: https://github.com/TruhnLab/MedicalPatchNet
>
---
#### [replaced 062] Enhancing Multi-Image Understanding through Delimiter Token Scaling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01984v2](https://arxiv.org/pdf/2602.01984v2)**

> **作者:** Minyoung Lee; Yeji Park; Dongjun Hwang; Yejin Kim; Seong Joon Oh; Junsuk Choe
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) achieve strong performance on single-image tasks, but their performance declines when multiple images are provided as input. One major reason is the cross-image information leakage, where the model struggles to distinguish information across different images. Existing LVLMs already employ delimiter tokens to mark the start and end of each image, yet our analysis reveals that these tokens fail to effectively block cross-image information leakage. To enhance their effectiveness, we propose a method that scales the hidden states of delimiter tokens. This enhances the model's ability to preserve image-specific information by reinforcing intra-image interaction and limiting undesired cross-image interactions. Consequently, the model is better able to distinguish between images and reason over them more accurately. Experiments show performance gains on multi-image benchmarks such as Mantis, MuirBench, MIRB, and QBench2. We further evaluate our method on text-only tasks that require clear distinction. The method improves performance on multi-document and multi-table understanding benchmarks, including TQABench, MultiNews, and WCEP-10. Notably, our method requires no additional training or inference cost.
>
---
#### [replaced 063] TherA: Thermal-Aware Visual-Language Prompting for Controllable RGB-to-Thermal Infrared Translation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19430v2](https://arxiv.org/pdf/2602.19430v2)**

> **作者:** Dong-Guw Lee; Tai Hyoung Rhee; Hyunsoo Jang; Young-Sik Shin; Ukcheol Shin; Ayoung Kim
>
> **摘要:** Despite the inherent advantages of thermal infrared(TIR) imaging, large-scale data collection and annotation remain a major bottleneck for TIR-based perception. A practical alternative is to synthesize pseudo TIR data via image translation; however, most RGB-to-TIR approaches heavily rely on RGB-centric priors that overlook thermal physics, yielding implausible heat distributions. In this paper, we introduce TherA, a controllable RGB-to-TIR translation framework that produces diverse and thermally plausible images at both scene and object level. TherA couples TherA-VLM with a latent-diffusion-based translator. Given a single RGB image and a user-prompted condition pair, TherA-VLM yields a thermal-aware embedding that encodes scene, object, material, and heat-emission context reflecting the input scene-condition pair. Conditioning the diffusion model on this embedding enables realistic TIR synthesis and fine-grained control across time of day, weather, and object state. Compared to other baselines, TherA achieves state-of-the-art translation performance, demonstrating improved zero-shot translation performance up to 33% increase averaged across all metrics.
>
---
#### [replaced 064] RAYNOVA: Scale-Temporal Autoregressive World Modeling in Ray Space
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20685v2](https://arxiv.org/pdf/2602.20685v2)**

> **作者:** Yichen Xie; Chensheng Peng; Mazen Abdelfattah; Yihan Hu; Jiezhi Yang; Eric Higgins; Ryan Brigden; Masayoshi Tomizuka; Wei Zhan
>
> **备注:** Accepted by CVPR 2026; Project website: https://raynova-ai.github.io/
>
> **摘要:** World foundation models aim to simulate the evolution of the real world with physically plausible behavior. Unlike prior methods that handle spatial and temporal correlations separately, we propose RAYNOVA, a geometry-agonistic multiview world model for driving scenarios that employs a dual-causal autoregressive framework. It follows both scale-wise and temporal topological orders in the autoregressive process, and leverages global attention for unified 4D spatio-temporal reasoning. Different from existing works that impose strong 3D geometric priors, RAYNOVA constructs an isotropic spatio-temporal representation across views, frames, and scales based on relative Plücker-ray positional encoding, enabling robust generalization to diverse camera setups and ego motions. We further introduce a recurrent training paradigm to alleviate distribution drift in long-horizon video generation. RAYNOVA achieves state-of-the-art multi-view video generation results on nuScenes, while offering higher throughput and strong controllability under diverse input conditions, generalizing to novel views and camera configurations without explicit 3D scene representation. Our code will be released at https://raynova-ai.github.io/.
>
---
#### [replaced 065] Voxel Densification for Serialized 3D Object Detection: Mitigating Sparsity via Pre-serialization Expansion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.16069v3](https://arxiv.org/pdf/2508.16069v3)**

> **作者:** Qifeng Liu; Dawei Zhao; Yabo Dong; Linzhi Shang; Liang Xiao; Juan Wang; Kunkong Zhao; Dongming Lu; Qi Zhu
>
> **备注:** Under review
>
> **摘要:** Recent advances in point cloud object detection have increasingly adopted Transformer-based and State Space Models (SSMs) to capture long-range dependencies. However, these serialized frameworks strictly maintain the consistency of input and output voxel dimensions, inherently lacking the capability for voxel expansion. This limitation hinders performance, as expanding the voxel set is known to significantly enhance detection accuracy, particularly for sparse foreground objects. To bridge this gap, we propose a novel Voxel Densification Module (VDM). Unlike standard convolutional stems, VDM is explicitly designed to promote pre-serialization spatial expansion. It leverages sparse 3D convolutions to propagate foreground semantics to neighboring empty voxels, effectively densifying the feature representation before it is flattened into a sequence. Simultaneously, VDM incorporates residual sparse blocks to aggregate fine-grained local context, ensuring rich geometric feature extraction. To balance the computational overhead of increased voxel density, we introduce a strategic cascaded downsampling mechanism. We integrate VDM into both Transformer-based (DSVT) and SSM-based (LION) detectors. Extensive experiments demonstrate that VDM consistently improves detection accuracy across multiple benchmarks. Specifically, our method achieves 74.8 mAPH (L2) on the Waymo validation set and 70.5 mAP on the nuScenes test set. Furthermore, it attains 42.6 mAP on the Argoverse 2 validation set and 67.6 mAP on the ONCE validation set, consistently outperforming the baseline models. The source code will be made publicly available at https://github.com/qifeng22/VDM.
>
---
#### [replaced 066] Variation-aware Vision Token Dropping for Faster Large Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.01552v2](https://arxiv.org/pdf/2509.01552v2)**

> **作者:** Junjie Chen; Xuyang Liu; Zichen Wen; Yiyu Wang; Siteng Huang; Honggang Chen
>
> **备注:** Accepted by CVPR 2026. Code is available at \url{https://github.com/xuyang-liu16/V2Drop}
>
> **摘要:** Large vision-language models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding tasks. However, the increasing demand for high-resolution image and long-video understanding results in substantial token counts, consequently leading to reduced inference efficiency. Token compression offers a direct solution by reducing the number of tokens to be processed, thereby improving computational efficiency without architectural changes. Through extensive analysis, we identify two critical limitations in existing inner-LLM token compression methods: positional bias and incompatibility with efficient operators, which critically hinder their practical deployment for LVLM acceleration. This paper presents the first approach from a dynamic token variation perspective, revealing that visual token variations within LLMs exhibit task-agnostic properties. We propose Variation-aware Vision Token Dropping (\textit{i.e.}, \textbf{V$^2$Drop}), which progressively removes visual tokens with minimal variation during LVLM inference, thereby enhancing computational efficiency. Extensive experiments across multiple models and benchmarks consistently demonstrate that V$^2$Drop maintains \textbf{94.0\%} and \textbf{98.6\%} of the original performance for image and video understanding tasks respectively, while reducing LLM generation latency by \textbf{31.5\%} and \textbf{74.2\%}.
>
---
#### [replaced 067] JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19610v3](https://arxiv.org/pdf/2505.19610v3)**

> **作者:** Jiaxin Song; Yixu Wang; Jie Li; Rui Yu; Yan Teng; Xingjun Ma; Yingchun Wang
>
> **备注:** The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Vision-Language Models (VLMs) exhibit impressive performance, yet the integration of powerful vision encoders has significantly broadened their attack surface, rendering them increasingly susceptible to jailbreak attacks. However, lacking well-defined attack objectives, existing jailbreak methods often struggle with gradient-based strategies prone to local optima and lacking precise directional guidance, and typically decouple visual and textual modalities, thereby limiting their effectiveness by neglecting crucial cross-modal interactions. Inspired by the Eliciting Latent Knowledge (ELK) framework, we posit that VLMs encode safety-relevant information within their internal fusion-layer representations, revealing an implicit safety decision boundary in the latent space. This motivates exploiting boundary to steer model behavior. Accordingly, we propose JailBound, a novel latent space jailbreak framework comprising two stages: (1) Safety Boundary Probing, which addresses the guidance issue by approximating decision boundary within fusion layer's latent space, thereby identifying optimal perturbation directions towards the target region; and (2) Safety Boundary Crossing, which overcomes the limitations of decoupled approaches by jointly optimizing adversarial perturbations across both image and text inputs. This latter stage employs an innovative mechanism to steer the model's internal state towards policy-violating outputs while maintaining cross-modal semantic consistency. Extensive experiments on six diverse VLMs demonstrate JailBound's efficacy, achieves 94.32% white-box and 67.28% black-box attack success averagely, which are 6.17% and 21.13% higher than SOTA methods, respectively. Our findings expose a overlooked safety risk in VLMs and highlight the urgent need for more robust defenses. Warning: This paper contains potentially sensitive, harmful and offensive content.
>
---
#### [replaced 068] Pay Attention to Where You Looked
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.18970v2](https://arxiv.org/pdf/2601.18970v2)**

> **作者:** Alex Berian; JhihYang Wu; Daniel Brignac; Natnael Daba; Abhijit Mahalanobis
>
> **备注:** ICIP 2025 Workshop on Generative AI for World Simulations and Communications
>
> **摘要:** Novel view synthesis (NVS) has advanced with generative modeling, enabling photorealistic image generation. In few-shot NVS, where only a few input views are available, existing methods often assume equal importance for all input views relative to the target, leading to suboptimal results. We address this limitation by introducing a camera-weighting mechanism that adjusts the importance of source views based on their relevance to the target. We propose two approaches: a deterministic weighting scheme leveraging geometric properties like Euclidean distance and angular differences, and a cross-attention-based learning scheme that optimizes view weighting. Additionally, models can be further trained with our camera-weighting scheme to refine their understanding of view relevance and enhance synthesis quality. This mechanism is adaptable and can be integrated into various NVS algorithms, improving their ability to synthesize high-quality novel views. Our results demonstrate that adaptive view weighting enhances accuracy and realism, offering a promising direction for improving NVS.
>
---
#### [replaced 069] Object-Centric World Models from Few-Shot Annotations for Sample-Efficient Reinforcement Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2501.16443v2](https://arxiv.org/pdf/2501.16443v2)**

> **作者:** Weipu Zhang; Adam Jelley; Trevor McInroe; Amos Storkey; Gang Wang
>
> **摘要:** While deep reinforcement learning (RL) from pixels has achieved remarkable success, its sample inefficiency remains a critical limitation for real-world applications. Model-based RL (MBRL) addresses this by learning a world model to generate simulated experience, but standard approaches that rely on pixel-level reconstruction losses often fail to capture small, task-critical objects in complex, dynamic scenes. We posit that an object-centric (OC) representation can direct model capacity toward semantically meaningful entities, improving dynamics prediction and sample efficiency. In this work, we introduce OC-STORM, an object-centric MBRL framework that enhances a learned world model with object representations extracted by a pretrained segmentation network. By conditioning on a minimal number of annotated frames, OC-STORM learns to track decision-relevant object dynamics and inter-object interactions without extensive labeling or access to privileged information. Empirical results demonstrate that OC-STORM significantly outperforms the STORM baseline on the Atari 100k benchmark and achieves state-of-the-art sample efficiency on challenging boss fights in the visually complex game Hollow Knight. Our findings underscore the potential of integrating OC priors into MBRL for complex visual domains. Project page: https://oc-storm.weipuzhang.com
>
---
