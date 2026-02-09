# 计算机视觉 cs.CV

- **最新发布 97 篇**

- **更新 63 篇**

## 最新发布

#### [new 001] Learning Human Visual Attention on 3D Surfaces through Geometry-Queried Semantic Priors
- **分类: cs.CV**

- **简介: 该论文属于3D视觉注意力建模任务，解决传统方法缺乏语义感知的问题。提出SemGeo-AttentionNet，结合几何与语义信息，提升对3D表面显著区域的预测能力。**

- **链接: [https://arxiv.org/pdf/2602.06419v1](https://arxiv.org/pdf/2602.06419v1)**

> **作者:** Soham Pahari; Sandeep C. Kumain
>
> **摘要:** Human visual attention on three-dimensional objects emerges from the interplay between bottom-up geometric processing and top-down semantic recognition. Existing 3D saliency methods rely on hand-crafted geometric features or learning-based approaches that lack semantic awareness, failing to explain why humans fixate on semantically meaningful but geometrically unremarkable regions. We introduce SemGeo-AttentionNet, a dual-stream architecture that explicitly formalizes this dichotomy through asymmetric cross-modal fusion, leveraging diffusion-based semantic priors from geometry-conditioned multi-view rendering and point cloud transformers for geometric processing. Cross-attention ensures geometric features query semantic content, enabling bottom-up distinctiveness to guide top-down retrieval. We extend our framework to temporal scanpath generation through reinforcement learning, introducing the first formulation respecting 3D mesh topology with inhibition-of-return dynamics. Evaluation on SAL3D, NUS3D and 3DVA datasets demonstrates substantial improvements, validating how cognitively motivated architectures effectively model human visual attention on three-dimensional surfaces.
>
---
#### [new 002] POPL-KF: A Pose-Only Geometric Representation-Based Kalman Filter for Point-Line-Based Visual-Inertial Odometry
- **分类: cs.CV**

- **简介: 该论文属于视觉惯性里程计任务，旨在提升复杂场景下的定位精度。针对传统方法的线性化误差和延迟更新问题，提出POPL-KF系统，采用纯位姿几何表示，优化点线特征融合与更新机制。**

- **链接: [https://arxiv.org/pdf/2602.06425v1](https://arxiv.org/pdf/2602.06425v1)**

> **作者:** Aiping Wang; Zhaolong Yang; Shuwen Chen; Hai Zhang
>
> **摘要:** Mainstream Visual-inertial odometry (VIO) systems rely on point features for motion estimation and localization. However, their performance degrades in challenging scenarios. Moreover, the localization accuracy of multi-state constraint Kalman filter (MSCKF)-based VIO systems suffers from linearization errors associated with feature 3D coordinates and delayed measurement updates. To improve the performance of VIO in challenging scenes, we first propose a pose-only geometric representation for line features. Building on this, we develop POPL-KF, a Kalman filter-based VIO system that employs a pose-only geometric representation for both point and line features. POPL-KF mitigates linearization errors by explicitly eliminating both point and line feature coordinates from the measurement equations, while enabling immediate update of visual measurements. We also design a unified base-frames selection algorithm for both point and line features to ensure optimal constraints on camera poses within the pose-only measurement model. To further improve line feature quality, a line feature filter based on image grid segmentation and bidirectional optical flow consistency is proposed. Our system is evaluated on public datasets and real-world experiments, demonstrating that POPL-KF outperforms the state-of-the-art (SOTA) filter-based methods (OpenVINS, PO-KF) and optimization-based methods (PL-VINS, EPLF-VINS), while maintaining real-time performance.
>
---
#### [new 003] CineScene: Implicit 3D as Effective Scene Representation for Cinematic Video Generation
- **分类: cs.CV**

- **简介: 该论文属于影视视频生成任务，旨在解决静态场景与动态主体分离生成的问题。提出CineScene框架，利用隐式3D场景表示实现场景一致的视频生成。**

- **链接: [https://arxiv.org/pdf/2602.06959v1](https://arxiv.org/pdf/2602.06959v1)**

> **作者:** Kaiyi Huang; Yukun Huang; Yu Li; Jianhong Bai; Xintao Wang; Zinan Lin; Xuefei Ning; Jiwen Yu; Pengfei Wan; Yu Wang; Xihui Liu
>
> **备注:** Project website: https://karine-huang.github.io/CineScene/
>
> **摘要:** Cinematic video production requires control over scene-subject composition and camera movement, but live-action shooting remains costly due to the need for constructing physical sets. To address this, we introduce the task of cinematic video generation with decoupled scene context: given multiple images of a static environment, the goal is to synthesize high-quality videos featuring dynamic subject while preserving the underlying scene consistency and following a user-specified camera trajectory. We present CineScene, a framework that leverages implicit 3D-aware scene representation for cinematic video generation. Our key innovation is a novel context conditioning mechanism that injects 3D-aware features in an implicit way: By encoding scene images into visual representations through VGGT, CineScene injects spatial priors into a pretrained text-to-video generation model by additional context concatenation, enabling camera-controlled video synthesis with consistent scenes and dynamic subjects. To further enhance the model's robustness, we introduce a simple yet effective random-shuffling strategy for the input scene images during training. To address the lack of training data, we construct a scene-decoupled dataset with Unreal Engine 5, containing paired videos of scenes with and without dynamic subjects, panoramic images representing the underlying static scene, along with their camera trajectories. Experiments show that CineScene achieves state-of-the-art performance in scene-consistent cinematic video generation, handling large camera movements and demonstrating generalization across diverse environments.
>
---
#### [new 004] Adaptive and Balanced Re-initialization for Long-timescale Continual Test-time Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于持续测试时域适应任务，旨在解决模型在长期非平稳环境中性能下降的问题。提出ABR方法通过自适应重初始化提升模型长期表现。**

- **链接: [https://arxiv.org/pdf/2602.06328v1](https://arxiv.org/pdf/2602.06328v1)**

> **作者:** Yanshuo Wang; Jinguang Tong; Jun Lan; Weiqiang Wang; Huijia Zhu; Haoxing Chen; Xuesong Li; Jie Hong
>
> **备注:** Accepted in ICASSP 2026
>
> **摘要:** Continual test-time domain adaptation (CTTA) aims to adjust models so that they can perform well over time across non-stationary environments. While previous methods have made considerable efforts to optimize the adaptation process, a crucial question remains: Can the model adapt to continually changing environments over a long time? In this work, we explore facilitating better CTTA in the long run using a re-initialization (or reset) based method. First, we observe that the long-term performance is associated with the trajectory pattern in label flip. Based on this observed correlation, we propose a simple yet effective policy, Adaptive-and-Balanced Re-initialization (ABR), towards preserving the model's long-term performance. In particular, ABR performs weight re-initialization using adaptive intervals. The adaptive interval is determined based on the change in label flip. The proposed method is validated on extensive CTTA benchmarks, achieving superior performance.
>
---
#### [new 005] Exploring Specular Reflection Inconsistency for Generalizable Face Forgery Detection
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测任务，旨在解决高质伪造人脸难以检测的问题。通过分析镜面反射不一致性，提出SRI-Net方法提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.06452v1](https://arxiv.org/pdf/2602.06452v1)**

> **作者:** Hongyan Fei; Zexi Jia; Chuanwei Huang; Jinchao Zhang; Jie Zhou
>
> **摘要:** Detecting deepfakes has become increasingly challenging as forgery faces synthesized by AI-generated methods, particularly diffusion models, achieve unprecedented quality and resolution. Existing forgery detection approaches relying on spatial and frequency features demonstrate limited efficacy against high-quality, entirely synthesized forgeries. In this paper, we propose a novel detection method grounded in the observation that facial attributes governed by complex physical laws and multiple parameters are inherently difficult to replicate. Specifically, we focus on illumination, particularly the specular reflection component in the Phong illumination model, which poses the greatest replication challenge due to its parametric complexity and nonlinear formulation. We introduce a fast and accurate face texture estimation method based on Retinex theory to enable precise specular reflection separation. Furthermore, drawing from the mathematical formulation of specular reflection, we posit that forgery evidence manifests not only in the specular reflection itself but also in its relationship with corresponding face texture and direct light. To address this issue, we design the Specular-Reflection-Inconsistency-Network (SRI-Net), incorporating a two-stage cross-attention mechanism to capture these correlations and integrate specular reflection related features with image features for robust forgery detection. Experimental results demonstrate that our method achieves superior performance on both traditional deepfake datasets and generative deepfake datasets, particularly those containing diffusion-generated forgery faces.
>
---
#### [new 006] Rebenchmarking Unsupervised Monocular 3D Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文属于单目3D占用预测任务，旨在解决无监督方法在训练与评估不一致及遮挡区域不确定性问题。通过改进评估协议和引入遮挡感知机制，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.06488v1](https://arxiv.org/pdf/2602.06488v1)**

> **作者:** Zizhan Guo; Yi Feng; Mengtan Zhang; Haoran Zhang; Wei Ye; Rui Fan
>
> **摘要:** Inferring the 3D structure from a single image, particularly in occluded regions, remains a fundamental yet unsolved challenge in vision-centric autonomous driving. Existing unsupervised approaches typically train a neural radiance field and treat the network outputs as occupancy probabilities during evaluation, overlooking the inconsistency between training and evaluation protocols. Moreover, the prevalent use of 2D ground truth fails to reveal the inherent ambiguity in occluded regions caused by insufficient geometric constraints. To address these issues, this paper presents a reformulated benchmark for unsupervised monocular 3D occupancy prediction. We first interpret the variables involved in the volume rendering process and identify the most physically consistent representation of the occupancy probability. Building on these analyses, we improve existing evaluation protocols by aligning the newly identified representation with voxel-wise 3D occupancy ground truth, thereby enabling unsupervised methods to be evaluated in a manner consistent with that of supervised approaches. Additionally, to impose explicit constraints in occluded regions, we introduce an occlusion-aware polarization mechanism that incorporates multi-view visual cues to enhance discrimination between occupied and free spaces in these regions. Extensive experiments demonstrate that our approach not only significantly outperforms existing unsupervised approaches but also matches the performance of supervised ones. Our source code and evaluation protocol will be made available upon publication.
>
---
#### [new 007] MeDocVL: A Visual Language Model for Medical Document Understanding and Parsing
- **分类: cs.CV**

- **简介: 该论文提出MeDocVL，用于医疗文档理解与解析。解决医疗OCR中布局复杂、术语专业、标注噪声等问题。通过标签优化和混合后训练策略，提升提取精度。**

- **链接: [https://arxiv.org/pdf/2602.06402v1](https://arxiv.org/pdf/2602.06402v1)**

> **作者:** Wenjie Wang; Wei Wu; Ying Liu; Yuan Zhao; Xiaole Lv; Liang Diao; Zengjian Fan; Wenfeng Xie; Ziling Lin; De Shi; Lin Huang; Kaihe Xu; Hong Li
>
> **备注:** 20 pages, 8 figures. Technical report
>
> **摘要:** Medical document OCR is challenging due to complex layouts, domain-specific terminology, and noisy annotations, while requiring strict field-level exact matching. Existing OCR systems and general-purpose vision-language models often fail to reliably parse such documents. We propose MeDocVL, a post-trained vision-language model for query-driven medical document parsing. Our framework combines Training-driven Label Refinement to construct high-quality supervision from noisy annotations, with a Noise-aware Hybrid Post-training strategy that integrates reinforcement learning and supervised fine-tuning to achieve robust and precise extraction. Experiments on medical invoice benchmarks show that MeDocVL consistently outperforms conventional OCR systems and strong VLM baselines, achieving state-of-the-art performance under noisy supervision.
>
---
#### [new 008] LAB-Det: Language as a Domain-Invariant Bridge for Training-Free One-Shot Domain Generalization in Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测领域的域泛化任务，解决在数据稀缺场景下模型适应问题。提出LAB-Det，通过语言描述引导冻结检测器，无需参数更新即可提升检测性能。**

- **链接: [https://arxiv.org/pdf/2602.06474v1](https://arxiv.org/pdf/2602.06474v1)**

> **作者:** Xu Zhang; Zhe Chen; Jing Zhang; Dacheng Tao
>
> **摘要:** Foundation object detectors such as GLIP and Grounding DINO excel on general-domain data but often degrade in specialized and data-scarce settings like underwater imagery or industrial defects. Typical cross-domain few-shot approaches rely on fine-tuning scarce target data, incurring cost and overfitting risks. We instead ask: Can a frozen detector adapt with only one exemplar per class without training? To answer this, we introduce training-free one-shot domain generalization for object detection, where detectors must adapt to specialized domains with only one annotated exemplar per class and no weight updates. To tackle this task, we propose LAB-Det, which exploits Language As a domain-invariant Bridge. Instead of adapting visual features, we project each exemplar into a descriptive text that conditions and guides a frozen detector. This linguistic conditioning replaces gradient-based adaptation, enabling robust generalization in data-scarce domains. We evaluate on UODD (underwater) and NEU-DET (industrial defects), two widely adopted benchmarks for data-scarce detection, where object boundaries are often ambiguous, and LAB-Det achieves up to 5.4 mAP improvement over state-of-the-art fine-tuned baselines without updating a single parameter. These results establish linguistic adaptation as an efficient and interpretable alternative to fine-tuning in specialized detection settings.
>
---
#### [new 009] Di3PO -- Diptych Diffusion DPO for Targeted Improvements in Image
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有方法在偏好调优中生成对效率低、差异不明显的问题。提出Di3PO方法，通过隔离特定区域提升训练效率。**

- **链接: [https://arxiv.org/pdf/2602.06355v1](https://arxiv.org/pdf/2602.06355v1)**

> **作者:** Sanjana Reddy; Ishaan Malhi; Sally Ma; Praneet Dutta
>
> **摘要:** Existing methods for preference tuning of text-to-image (T2I) diffusion models often rely on computationally expensive generation steps to create positive and negative pairs of images. These approaches frequently yield training pairs that either lack meaningful differences, are expensive to sample and filter, or exhibit significant variance in irrelevant pixel regions, thereby degrading training efficiency. To address these limitations, we introduce "Di3PO", a novel method for constructing positive and negative pairs that isolates specific regions targeted for improvement during preference tuning, while keeping the surrounding context in the image stable. We demonstrate the efficacy of our approach by applying it to the challenging task of text rendering in diffusion models, showcasing improvements over baseline methods of SFT and DPO.
>
---
#### [new 010] Halt the Hallucination: Decoupling Signal and Semantic OOD Detection Based on Cascaded Early Rejection
- **分类: cs.CV**

- **简介: 该论文属于OOD检测任务，旨在解决深度学习模型在分布外数据中产生语义幻觉的问题。提出CER框架，通过结构和语义分析实现高效异常检测。**

- **链接: [https://arxiv.org/pdf/2602.06330v1](https://arxiv.org/pdf/2602.06330v1)**

> **作者:** Ningkang Peng; Chuanjie Cheng; Jingyang Mao; Xiaoqian Peng; Feng Xing; Bo Zhang; Chao Tan; Zhichao Zheng; Peiheng Li; Yanhui Gu
>
> **摘要:** Efficient and robust Out-of-Distribution (OOD) detection is paramount for safety-critical applications.However, existing methods still execute full-scale inference on low-level statistical noise. This computational mismatch not only incurs resource waste but also induces semantic hallucination, where deep networks forcefully interpret physical anomalies as high-confidence semantic features.To address this, we propose the Cascaded Early Rejection (CER) framework, which realizes hierarchical filtering for anomaly detection via a coarse-to-fine logic.CER comprises two core modules: 1)Structural Energy Sieve (SES), which establishes a non-parametric barrier at the network entry using the Laplacian operator to efficiently intercept physical signal anomalies; and 2) the Semantically-aware Hyperspherical Energy (SHE) detector, which decouples feature magnitude from direction in intermediate layers to identify fine-grained semantic deviations. Experimental results demonstrate that CER not only reduces computational overhead by 32% but also achieves a significant performance leap on the CIFAR-100 benchmark:the average FPR95 drastically decreases from 33.58% to 22.84%, and AUROC improves to 93.97%. Crucially, in real-world scenarios simulating sensor failures, CER exhibits performance far exceeding state-of-the-art methods. As a universal plugin, CER can be seamlessly integrated into various SOTA models to provide performance gains.
>
---
#### [new 011] A neuromorphic model of the insect visual system for natural image processing
- **分类: cs.CV; cs.NE**

- **简介: 该论文提出一种仿生昆虫视觉模型，用于自然图像处理。旨在解决生物启发视觉建模问题，通过自监督学习生成稀疏特征，提升任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06405v1](https://arxiv.org/pdf/2602.06405v1)**

> **作者:** Adam D. Hines; Karin Nordström; Andrew B. Barron
>
> **备注:** 21 pages, 7 figures, under review
>
> **摘要:** Insect vision supports complex behaviors including associative learning, navigation, and object detection, and has long motivated computational models for understanding biological visual processing. However, many contemporary models prioritize task performance while neglecting biologically grounded processing pathways. Here, we introduce a bio-inspired vision model that captures principles of the insect visual system to transform dense visual input into sparse, discriminative codes. The model is trained using a fully self-supervised contrastive objective, enabling representation learning without labeled data and supporting reuse across tasks without reliance on domain-specific classifiers. We evaluated the resulting representations on flower recognition tasks and natural image benchmarks. The model consistently produced reliable sparse codes that distinguish visually similar inputs. To support different modelling and deployment uses, we have implemented the model as both an artificial neural network and a spiking neural network. In a simulated localization setting, our approach outperformed a simple image downsampling comparison baseline, highlighting the functional benefit of incorporating neuromorphic visual processing pathways. Collectively, these results advance insect computational modelling by providing a generalizable bio-inspired vision model capable of sparse computation across diverse tasks.
>
---
#### [new 012] Bridging the Indoor-Outdoor Gap: Vision-Centric Instruction-Guided Embodied Navigation for the Last Meters
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于室内外导航任务，解决室外到室内无缝过渡问题。提出无需外部先验的视觉引导导航框架，并构建首个相关数据集，提升导航精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.06427v1](https://arxiv.org/pdf/2602.06427v1)**

> **作者:** Yuxiang Zhao; Yirong Yang; Yanqing Zhu; Yanfen Shen; Chiyu Wang; Zhining Gu; Pei Shi; Wei Guo; Mu Xu
>
> **摘要:** Embodied navigation holds significant promise for real-world applications such as last-mile delivery. However, most existing approaches are confined to either indoor or outdoor environments and rely heavily on strong assumptions, such as access to precise coordinate systems. While current outdoor methods can guide agents to the vicinity of a target using coarse-grained localization, they fail to enable fine-grained entry through specific building entrances, critically limiting their utility in practical deployment scenarios that require seamless outdoor-to-indoor transitions. To bridge this gap, we introduce a novel task: out-to-in prior-free instruction-driven embodied navigation. This formulation explicitly eliminates reliance on accurate external priors, requiring agents to navigate solely based on egocentric visual observations guided by instructions. To tackle this task, we propose a vision-centric embodied navigation framework that leverages image-based prompts to drive decision-making. Additionally, we present the first open-source dataset for this task, featuring a pipeline that integrates trajectory-conditioned video synthesis into the data generation process. Through extensive experiments, we demonstrate that our proposed method consistently outperforms state-of-the-art baselines across key metrics including success rate and path efficiency.
>
---
#### [new 013] MMEarth-Bench: Global Model Adaptation via Multimodal Test-Time Training
- **分类: cs.CV**

- **简介: 该论文提出MMEarth-Bench，解决地理环境下多模态模型泛化能力不足的问题，通过引入多模态任务和测试时训练方法提升模型适应性。**

- **链接: [https://arxiv.org/pdf/2602.06285v1](https://arxiv.org/pdf/2602.06285v1)**

> **作者:** Lucia Gordon; Serge Belongie; Christian Igel; Nico Lang
>
> **摘要:** Recent research in geospatial machine learning has demonstrated that models pretrained with self-supervised learning on Earth observation data can perform well on downstream tasks with limited training data. However, most of the existing geospatial benchmark datasets have few data modalities and poor global representation, limiting the ability to evaluate multimodal pretrained models at global scales. To fill this gap, we introduce MMEarth-Bench, a collection of five new multimodal environmental tasks with 12 modalities, globally distributed data, and both in- and out-of-distribution test splits. We benchmark a diverse set of pretrained models and find that while (multimodal) pretraining tends to improve model robustness in limited data settings, geographic generalization abilities remain poor. In order to facilitate model adaptation to new downstream tasks and geographic domains, we propose a model-agnostic method for test-time training with multimodal reconstruction (TTT-MMR) that uses all the modalities available at test time as auxiliary tasks, regardless of whether a pretrained model accepts them as input. Our method improves model performance on both the random and geographic test splits, and geographic batching leads to a good trade-off between regularization and specialization during TTT. Our dataset, code, and visualization tool are linked from the project page at lgordon99.github.io/mmearth-bench.
>
---
#### [new 014] DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DriveWorld-VLA，解决自动驾驶中场景演化与动作规划统一的问题。通过融合视觉-语言-动作与世界模型，在潜在空间实现联合建模与决策，提升感知与规划效果。**

- **链接: [https://arxiv.org/pdf/2602.06521v1](https://arxiv.org/pdf/2602.06521v1)**

> **作者:** Feiyang jia; Lin Liu; Ziying Song; Caiyan Jia; Hangjun Ye; Xiaoshuai Hao; Long Chen
>
> **备注:** 20 pages, 7 tables, 12 figures
>
> **摘要:** End-to-end (E2E) autonomous driving has recently attracted increasing interest in unifying Vision-Language-Action (VLA) with World Models to enhance decision-making and forward-looking imagination. However, existing methods fail to effectively unify future scene evolution and action planning within a single architecture due to inadequate sharing of latent states, limiting the impact of visual imagination on action decisions. To address this limitation, we propose DriveWorld-VLA, a novel framework that unifies world modeling and planning within a latent space by tightly integrating VLA and world models at the representation level, which enables the VLA planner to benefit directly from holistic scene-evolution modeling and reducing reliance on dense annotated supervision. Additionally, DriveWorld-VLA incorporates the latent states of the world model as core decision-making states for the VLA planner, facilitating the planner to assess how candidate actions impact future scene evolution. By conducting world modeling entirely in the latent space, DriveWorld-VLA supports controllable, action-conditioned imagination at the feature level, avoiding expensive pixel-level rollouts. Extensive open-loop and closed-loop evaluations demonstrate the effectiveness of DriveWorld-VLA, which achieves state-of-the-art performance with 91.3 PDMS on NAVSIMv1, 86.8 EPDMS on NAVSIMv2, and 0.16 3-second average collision rate on nuScenes. Code and models will be released in https://github.com/liulin815/DriveWorld-VLA.git.
>
---
#### [new 015] CytoCrowd: A Multi-Annotator Benchmark Dataset for Cytology Image Analysis
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 该论文提出CytoCrowd数据集，用于细胞图像分析。解决医学图像中专家标注不一致的问题，通过提供多标注和黄金标准，支持模型训练与标注聚合算法评估。**

- **链接: [https://arxiv.org/pdf/2602.06674v1](https://arxiv.org/pdf/2602.06674v1)**

> **作者:** Yonghao Si; Xingyuan Zeng; Zhao Chen; Libin Zheng; Caleb Chen Cao; Lei Chen; Jian Yin
>
> **摘要:** High-quality annotated datasets are crucial for advancing machine learning in medical image analysis. However, a critical gap exists: most datasets either offer a single, clean ground truth, which hides real-world expert disagreement, or they provide multiple annotations without a separate gold standard for objective evaluation. To bridge this gap, we introduce CytoCrowd, a new public benchmark for cytology analysis. The dataset features 446 high-resolution images, each with two key components: (1) raw, conflicting annotations from four independent pathologists, and (2) a separate, high-quality gold-standard ground truth established by a senior expert. This dual structure makes CytoCrowd a versatile resource. It serves as a benchmark for standard computer vision tasks, such as object detection and classification, using the ground truth. Simultaneously, it provides a realistic testbed for evaluating annotation aggregation algorithms that must resolve expert disagreements. We provide comprehensive baseline results for both tasks. Our experiments demonstrate the challenges presented by CytoCrowd and establish its value as a resource for developing the next generation of models for medical image analysis.
>
---
#### [new 016] EgoAVU: Egocentric Audio-Visual Understanding
- **分类: cs.CV**

- **简介: 该论文提出EgoAVU，解决egocentric视频中多模态理解问题，通过生成高质量数据提升MLLMs对音频和视觉信息的联合理解能力。**

- **链接: [https://arxiv.org/pdf/2602.06139v1](https://arxiv.org/pdf/2602.06139v1)**

> **作者:** Ashish Seth; Xinhao Mei; Changsheng Zhao; Varun Nagaraja; Ernie Chang; Gregory P. Meyer; Gael Le Lan; Yunyang Xiong; Vikas Chandra; Yangyang Shi; Dinesh Manocha; Zhipeng Cai
>
> **摘要:** Understanding egocentric videos plays a vital role for embodied intelligence. Recent multi-modal large language models (MLLMs) can accept both visual and audio inputs. However, due to the challenge of obtaining text labels with coherent joint-modality information, whether MLLMs can jointly understand both modalities in egocentric videos remains under-explored. To address this problem, we introduce EgoAVU, a scalable data engine to automatically generate egocentric audio-visual narrations, questions, and answers. EgoAVU enriches human narrations with multimodal context and generates audio-visual narrations through cross-modal correlation modeling. Token-based video filtering and modular, graph-based curation ensure both data diversity and quality. Leveraging EgoAVU, we construct EgoAVU-Instruct, a large-scale training dataset of 3M samples, and EgoAVU-Bench, a manually verified evaluation split covering diverse tasks. EgoAVU-Bench clearly reveals the limitations of existing MLLMs: they bias heavily toward visual signals, often neglecting audio cues or failing to correspond audio with the visual source. Finetuning MLLMs on EgoAVU-Instruct effectively addresses this issue, enabling up to 113% performance improvement on EgoAVU-Bench. Such benefits also transfer to other benchmarks such as EgoTempo and EgoIllusion, achieving up to 28% relative performance gain. Code will be released to the community.
>
---
#### [new 017] DeDPO: Debiased Direct Preference Optimization for Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于扩散模型对齐任务，解决DPO依赖高成本人工标注的问题。通过引入去偏技术与合成数据，提出DeDPO方法，实现高效、鲁棒的模型训练。**

- **链接: [https://arxiv.org/pdf/2602.06195v1](https://arxiv.org/pdf/2602.06195v1)**

> **作者:** Khiem Pham; Quang Nguyen; Tung Nguyen; Jingsen Zhu; Michele Santacatterina; Dimitris Metaxas; Ramin Zabih
>
> **摘要:** Direct Preference Optimization (DPO) has emerged as a predominant alignment method for diffusion models, facilitating off-policy training without explicit reward modeling. However, its reliance on large-scale, high-quality human preference labels presents a severe cost and scalability bottleneck. To overcome this, We propose a semi-supervised framework augmenting limited human data with a large corpus of unlabeled pairs annotated via cost-effective synthetic AI feedback. Our paper introduces Debiased DPO (DeDPO), which uniquely integrates a debiased estimation technique from causal inference into the DPO objective. By explicitly identifying and correcting the systematic bias and noise inherent in synthetic annotators, DeDPO ensures robust learning from imperfect feedback sources, including self-training and Vision-Language Models (VLMs). Experiments demonstrate that DeDPO is robust to the variations in synthetic labeling methods, achieving performance that matches and occasionally exceeds the theoretical upper bound of models trained on fully human-labeled data. This establishes DeDPO as a scalable solution for human-AI alignment using inexpensive synthetic supervision.
>
---
#### [new 018] MicroBi-ConvLSTM: An Ultra-Lightweight Efficient Model for Human Activity Recognition on Resource Constrained Devices
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于人体活动识别任务，旨在解决资源受限设备上的模型轻量化问题。提出MicroBi-ConvLSTM模型，在保持高精度的同时显著减少参数量和内存占用。**

- **链接: [https://arxiv.org/pdf/2602.06523v1](https://arxiv.org/pdf/2602.06523v1)**

> **作者:** Mridankan Mandal
>
> **摘要:** Human Activity Recognition (HAR) on resource constrained wearables requires models that balance accuracy against strict memory and computational budgets. State of the art lightweight architectures such as TinierHAR (34K parameters) and TinyHAR (55K parameters) achieve strong accuracy, but exceed memory budgets of microcontrollers with limited SRAM once operating system overhead is considered. We present MicroBi-ConvLSTM, an ultra-lightweight convolutional-recurrent architecture achieving 11.4K parameters on average through two stage convolutional feature extraction with 4x temporal pooling and a single bidirectional LSTM layer. This represents 2.9x parameter reduction versus TinierHAR and 11.9x versus DeepConvLSTM while preserving linear O(N) complexity. Evaluation across eight diverse HAR benchmarks shows that MicroBi-ConvLSTM maintains competitive performance within the ultra-lightweight regime: 93.41% macro F1 on UCI-HAR, 94.46% on SKODA assembly gestures, and 88.98% on Daphnet gait freeze detection. Systematic ablation reveals task dependent component contributions where bidirectionality benefits episodic event detection, but provides marginal gains on periodic locomotion. INT8 post training quantization incurs only 0.21% average F1-score degradation, yielding a 23.0 KB average deployment footprint suitable for memory constrained edge devices.
>
---
#### [new 019] RFDM: Residual Flow Diffusion Model for Efficient Causal Video Editing
- **分类: cs.CV**

- **简介: 该论文提出RFDM模型，用于高效因果视频编辑。解决视频编辑中输入固定长度和计算成本高的问题，通过残差流扩散模型实现逐帧编辑。**

- **链接: [https://arxiv.org/pdf/2602.06871v1](https://arxiv.org/pdf/2602.06871v1)**

> **作者:** Mohammadreza Salehi; Mehdi Noroozi; Luca Morreale; Ruchika Chavhan; Malcolm Chadwick; Alberto Gil Ramos; Abhinav Mehrotra
>
> **摘要:** Instructional video editing applies edits to an input video using only text prompts, enabling intuitive natural-language control. Despite rapid progress, most methods still require fixed-length inputs and substantial compute. Meanwhile, autoregressive video generation enables efficient variable-length synthesis, yet remains under-explored for video editing. We introduce a causal, efficient video editing model that edits variable-length videos frame by frame. For efficiency, we start from a 2D image-to-image (I2I) diffusion model and adapt it to video-to-video (V2V) editing by conditioning the edit at time step t on the model's prediction at t-1. To leverage videos' temporal redundancy, we propose a new I2I diffusion forward process formulation that encourages the model to predict the residual between the target output and the previous prediction. We call this Residual Flow Diffusion Model (RFDM), which focuses the denoising process on changes between consecutive frames. Moreover, we propose a new benchmark that better ranks state-of-the-art methods for editing tasks. Trained on paired video data for global/local style transfer and object removal, RFDM surpasses I2I-based methods and competes with fully spatiotemporal (3D) V2V models, while matching the compute of image models and scaling independently of input video length. More content can be found in: https://smsd75.github.io/RFDM_page/
>
---
#### [new 020] What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution
- **分类: cs.CV**

- **简介: 该论文属于场景文本识别任务，旨在解决合成数据与真实数据之间的域差距问题。通过构建更真实的合成数据集UnionST-S和自进化学习框架，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.06450v1](https://arxiv.org/pdf/2602.06450v1)**

> **作者:** Xingsong Ye; Yongkun Du; JiaXin Zhang; Chen Li; Jing LYU; Zhineng Chen
>
> **摘要:** Large-scale and categorical-balanced text data is essential for training effective Scene Text Recognition (STR) models, which is hard to achieve when collecting real data. Synthetic data offers a cost-effective and perfectly labeled alternative. However, its performance often lags behind, revealing a significant domain gap between real and current synthetic data. In this work, we systematically analyze mainstream rendering-based synthetic datasets and identify their key limitations: insufficient diversity in corpus, font, and layout, which restricts their realism in complex scenarios. To address these issues, we introduce UnionST, a strong data engine synthesizes text covering a union of challenging samples and better aligns with the complexity observed in the wild. We then construct UnionST-S, a large-scale synthetic dataset with improved simulations in challenging scenarios. Furthermore, we develop a self-evolution learning (SEL) framework for effective real data annotation. Experiments show that models trained on UnionST-S achieve significant improvements over existing synthetic datasets. They even surpass real-data performance in certain scenarios. Moreover, when using SEL, the trained models achieve competitive performance by only seeing 9% of real data labels.
>
---
#### [new 021] FlowConsist: Make Your Flow Consistent with Real Trajectory
- **分类: cs.CV**

- **简介: 该论文属于生成模型任务，解决快速流模型轨迹不一致问题。通过引入边际速度和轨迹校正策略，提升生成样本的一致性与质量。**

- **链接: [https://arxiv.org/pdf/2602.06346v1](https://arxiv.org/pdf/2602.06346v1)**

> **作者:** Tianyi Zhang; Chengcheng Liu; Jinwei Chen; Chun-Le Guo; Chongyi Li; Ming-Ming Cheng; Bo Li; Peng-Tao Jiang
>
> **摘要:** Fast flow models accelerate the iterative sampling process by learning to directly predict ODE path integrals, enabling one-step or few-step generation. However, we argue that current fast-flow training paradigms suffer from two fundamental issues. First, conditional velocities constructed from randomly paired noise-data samples introduce systematic trajectory drift, preventing models from following a consistent ODE path. Second, the model's approximation errors accumulate over time steps, leading to severe deviations across long time intervals. To address these issues, we propose FlowConsist, a training framework designed to enforce trajectory consistency in fast flows. We propose a principled alternative that replaces conditional velocities with the marginal velocities predicted by the model itself, aligning optimization with the true trajectory. To further address error accumulation over time steps, we introduce a trajectory rectification strategy that aligns the marginal distributions of generated and real samples at every time step along the trajectory. Our method establishes a new state-of-the-art on ImageNet 256$\times$256, achieving an FID of 1.52 with only 1 sampling step.
>
---
#### [new 022] AnyThermal: Towards Learning Universal Representations for Thermal Perception
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出AnyThermal，解决热成像特征提取任务，通过多环境数据训练，提升模型泛化能力，适用于多种下游任务。**

- **链接: [https://arxiv.org/pdf/2602.06203v1](https://arxiv.org/pdf/2602.06203v1)**

> **作者:** Parv Maheshwari; Jay Karhade; Yogesh Chawla; Isaiah Adu; Florian Heisen; Andrew Porco; Andrew Jong; Yifei Liu; Santosh Pitla; Sebastian Scherer; Wenshan Wang
>
> **备注:** Accepted at IEEE ICRA (International Conference on Robotics & Automation) 2026
>
> **摘要:** We present AnyThermal, a thermal backbone that captures robust task-agnostic thermal features suitable for a variety of tasks such as cross-modal place recognition, thermal segmentation, and monocular depth estimation using thermal images. Existing thermal backbones that follow task-specific training from small-scale data result in utility limited to a specific environment and task. Unlike prior methods, AnyThermal can be used for a wide range of environments (indoor, aerial, off-road, urban) and tasks, all without task-specific training. Our key insight is to distill the feature representations from visual foundation models such as DINOv2 into a thermal encoder using thermal data from these multiple environments. To bridge the diversity gap of the existing RGB-Thermal datasets, we introduce the TartanRGBT platform, the first open-source data collection platform with synced RGB-Thermal image acquisition. We use this payload to collect the TartanRGBT dataset - a diverse and balanced dataset collected in 4 environments. We demonstrate the efficacy of AnyThermal and TartanRGBT, achieving state-of-the-art results with improvements of up to 36% across diverse environments and downstream tasks on existing datasets.
>
---
#### [new 023] Addressing the Waypoint-Action Gap in End-to-End Autonomous Driving via Vehicle Motion Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决waypoint与action之间的差距问题。通过引入可微车辆模型框架，使基于动作的策略能在基于waypoint的基准中训练和评估。**

- **链接: [https://arxiv.org/pdf/2602.06214v1](https://arxiv.org/pdf/2602.06214v1)**

> **作者:** Jorge Daniel Rodríguez-Vidal; Gabriel Villalonga; Diego Porres; Antonio M. López Peña
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** End-to-End Autonomous Driving (E2E-AD) systems are typically grouped by the nature of their outputs: (i) waypoint-based models that predict a future trajectory, and (ii) action-based models that directly output throttle, steer and brake. Most recent benchmark protocols and training pipelines are waypoint-based, which makes action-based policies harder to train and compare, slowing their progress. To bridge this waypoint-action gap, we propose a novel, differentiable vehicle-model framework that rolls out predicted action sequences to their corresponding ego-frame waypoint trajectories while supervising in waypoint space. Our approach enables action-based architectures to be trained and evaluated, for the first time, within waypoint-based benchmarks without modifying the underlying evaluation protocol. We extensively evaluate our framework across multiple challenging benchmarks and observe consistent improvements over the baselines. In particular, on NAVSIM \texttt{navhard} our approach achieves state-of-the-art performance. Our code will be made publicly available upon acceptance.
>
---
#### [new 024] Accelerating Vision Transformers on Brain Processing Unit
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决Vision Transformer在Brain Processing Unit上加速困难的问题。通过重构模型结构，使其适配BPU硬件，实现高效推理。**

- **链接: [https://arxiv.org/pdf/2602.06300v1](https://arxiv.org/pdf/2602.06300v1)**

> **作者:** Jinchi Tang; Yan Guo
>
> **摘要:** With the advancement of deep learning technologies, specialized neural processing hardware such as Brain Processing Units (BPUs) have emerged as dedicated platforms for CNN acceleration, offering optimized INT8 computation capabilities for convolutional operations. Meanwhile, Vision Transformer (ViT) models, such as the Data-efficient Image Transformer (DeiT), have demonstrated superior performance and play increasingly crucial roles in computer vision tasks. However, due to the architectural mismatch between CNN-optimized hardware and Vision Transformer computation characteristics--namely, that linear layers in Transformers operate on three-dimensional data while BPU acceleration is designed for four-dimensional convolution operations-it is difficult or even impossible to leverage BPU's advantages when deploying Vision Transformers. To address this challenge, we propose a novel approach that restructures the Vision Transformer by replacing linear layers and layer normalization operations with carefully designed convolutional operators. This enables DeiT to fully utilize the acceleration capabilities of BPUs, while allowing the original weight parameters to be inherited by the restructured models without retraining or fine-tuning. To the best of our knowledge, this is the first successful deployment of Vision Transformers that fully leverages BPU classification datasets demonstrate the effectiveness of our approach. Specifically, the quantized DeiT-Base model achieves 80.4% accuracy on ImageNet, compared to the original 81.8%, while obtaining up to a 3.8* inference speedup. Our finetuned DeiT model on the flower classification dataset also achieves excellent performance, with only a 0.5% accuracy drop for the DeiT-Base model, further demonstrating the effectiveness of our method.
>
---
#### [new 025] Revisiting Emotions Representation for Recognition in the Wild
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于面部情绪识别任务，旨在解决传统单标签分类无法准确描述复杂情绪的问题。通过将情绪建模为概率分布，实现更丰富的多情绪混合描述。**

- **链接: [https://arxiv.org/pdf/2602.06778v1](https://arxiv.org/pdf/2602.06778v1)**

> **作者:** Joao Baptista Cardia Neto; Claudio Ferrari; Stefano Berretti
>
> **摘要:** Facial emotion recognition has been typically cast as a single-label classification problem of one out of six prototypical emotions. However, that is an oversimplification that is unsuitable for representing the multifaceted spectrum of spontaneous emotional states, which are most often the result of a combination of multiple emotions contributing at different intensities. Building on this, a promising direction that was explored recently is to cast emotion recognition as a distribution learning problem. Still, such approaches are limited in that research datasets are typically annotated with a single emotion class. In this paper, we contribute a novel approach to describe complex emotional states as probability distributions over a set of emotion classes. To do so, we propose a solution to automatically re-label existing datasets by exploiting the result of a study in which a large set of both basic and compound emotions is mapped to probability distributions in the Valence-Arousal-Dominance (VAD) space. In this way, given a face image annotated with VAD values, we can estimate the likelihood of it belonging to each of the distributions, so that emotional states can be described as a mixture of emotions, enriching their description, while also accounting for the ambiguous nature of their perception. In a preliminary set of experiments, we illustrate the advantages of this solution and a new possible direction of investigation. Data annotations are available at https://github.com/jbcnrlz/affectnet-b-annotation.
>
---
#### [new 026] MedMO: Grounding and Understanding Multimodal Large Language Model for Medical Images
- **分类: cs.CV**

- **简介: 该论文提出MedMO，解决医学图像与文本的多模态理解问题，通过跨模态预训练、指令调优和强化学习提升医学场景下的视觉-语言对齐与推理能力。**

- **链接: [https://arxiv.org/pdf/2602.06965v1](https://arxiv.org/pdf/2602.06965v1)**

> **作者:** Ankan Deria; Komal Kumar; Adinath Madhavrao Dukre; Eran Segal; Salman Khan; Imran Razzak
>
> **备注:** 21 pages, 6 figures and 4 tables
>
> **摘要:** Multimodal large language models (MLLMs) have rapidly advanced, yet their adoption in medicine remains limited by gaps in domain coverage, modality alignment, and grounded reasoning. In this work, we introduce MedMO, a medical foundation model built upon a generalized MLLM architecture and trained exclusively on large-scale, domain-specific data. MedMO follows a multi-stage training recipe: (i) cross-modal pretraining to align heterogeneous visual encoders with a medical language backbone; (ii) instruction tuning on multi-task supervision that spans captioning, VQA, report generation, retrieval, and grounded disease localization with bounding boxes; and (iii) reinforcement learning with verifiable rewards that combine factuality checks with a box-level GIoU reward to strengthen spatial grounding and step-by-step reasoning in complex clinical scenarios. MedMO consistently outperforms strong open-source medical MLLMs across multiple modalities and tasks. On VQA benchmarks, MedMO achieves an average accuracy improvement of +13.7% over the baseline and performs within 1.9% of the SOTA Fleming-VL. For text-based QA, it attains +6.9% over the baseline and +14.5% over Fleming-VL. In medical report generation, MedMO delivers significant gains in both semantic and clinical accuracy. Moreover, it exhibits strong grounding capability, achieving an IoU improvement of +40.4 over the baseline and +37.0% over Fleming-VL, underscoring its robust spatial reasoning and localization performance. Evaluations across radiology, ophthalmology, and pathology-microscopy confirm MedMO's broad cross-modality generalization. We release two versions of MedMO: 4B and 8B. Project is available at https://genmilab.github.io/MedMO-Page
>
---
#### [new 027] MGP-KAD: Multimodal Geometric Priors and Kolmogorov-Arnold Decoder for Single-View 3D Reconstruction in Complex Scenes
- **分类: cs.CV**

- **简介: 该论文属于单视角3D重建任务，解决复杂场景下的重建难题。通过融合RGB与几何先验及改进解码器，提升重建精度与细节保持。**

- **链接: [https://arxiv.org/pdf/2602.06158v1](https://arxiv.org/pdf/2602.06158v1)**

> **作者:** Luoxi Zhang; Chun Xie; Itaru Kitahara
>
> **备注:** 6 pages. Published in IEEE International Conference on Image Processing (ICIP) 2025
>
> **摘要:** Single-view 3D reconstruction in complex real-world scenes is challenging due to noise, object diversity, and limited dataset availability. To address these challenges, we propose MGP-KAD, a novel multimodal feature fusion framework that integrates RGB and geometric prior to enhance reconstruction accuracy. The geometric prior is generated by sampling and clustering ground-truth object data, producing class-level features that dynamically adjust during training to improve geometric understanding. Additionally, we introduce a hybrid decoder based on Kolmogorov-Arnold Networks (KAN) to overcome the limitations of traditional linear decoders in processing complex multimodal inputs. Extensive experiments on the Pix3D dataset demonstrate that MGP-KAD achieves state-of-the-art (SOTA) performance, significantly improving geometric integrity, smoothness, and detail preservation. Our work provides a robust and effective solution for advancing single-view 3D reconstruction in complex scenes.
>
---
#### [new 028] LIBERO-X: Robustness Litmus for Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出LIBERO-X基准，用于评估视觉-语言-动作模型的鲁棒性。针对现有基准评估不足的问题，设计分层评估协议和多样化数据集，提升模型测试可靠性。**

- **链接: [https://arxiv.org/pdf/2602.06556v1](https://arxiv.org/pdf/2602.06556v1)**

> **作者:** Guodong Wang; Chenkai Zhang; Qingjie Liu; Jinjin Zhang; Jiancheng Cai; Junjie Liu; Xinmin Liu
>
> **备注:** 19 pages, 14 figures and 8 tables
>
> **摘要:** Reliable benchmarking is critical for advancing Vision-Language-Action (VLA) models, as it reveals their generalization, robustness, and alignment of perception with language-driven manipulation tasks. However, existing benchmarks often provide limited or misleading assessments due to insufficient evaluation protocols that inadequately capture real-world distribution shifts. This work systematically rethinks VLA benchmarking from both evaluation and data perspectives, introducing LIBERO-X, a benchmark featuring: 1) A hierarchical evaluation protocol with progressive difficulty levels targeting three core capabilities: spatial generalization, object recognition, and task instruction understanding. This design enables fine-grained analysis of performance degradation under increasing environmental and task complexity; 2) A high-diversity training dataset collected via human teleoperation, where each scene supports multiple fine-grained manipulation objectives to bridge the train-evaluation distribution gap. Experiments with representative VLA models reveal significant performance drops under cumulative perturbations, exposing persistent limitations in scene comprehension and instruction grounding. By integrating hierarchical evaluation with diverse training data, LIBERO-X offers a more reliable foundation for assessing and advancing VLA development.
>
---
#### [new 029] Prompt Reinjection: Alleviating Prompt Forgetting in Multimodal Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决MMDiTs中提示遗忘问题。通过引入提示重注入方法，在不增加训练成本的情况下提升生成质量与指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2602.06886v1](https://arxiv.org/pdf/2602.06886v1)**

> **作者:** Yuxuan Yao; Yuxuan Chen; Hui Li; Kaihui Cheng; Qipeng Guo; Yuwei Sun; Zilong Dong; Jingdong Wang; Siyu Zhu
>
> **备注:** 18 pages
>
> **摘要:** Multimodal Diffusion Transformers (MMDiTs) for text-to-image generation maintain separate text and image branches, with bidirectional information flow between text tokens and visual latents throughout denoising. In this setting, we observe a prompt forgetting phenomenon: the semantics of the prompt representation in the text branch is progressively forgotten as depth increases. We further verify this effect on three representative MMDiTs--SD3, SD3.5, and FLUX.1 by probing linguistic attributes of the representations over the layers in the text branch. Motivated by these findings, we introduce a training-free approach, prompt reinjection, which reinjects prompt representations from early layers into later layers to alleviate this forgetting. Experiments on GenEval, DPG, and T2I-CompBench++ show consistent gains in instruction-following capability, along with improvements on metrics capturing preference, aesthetics, and overall text--image generation quality.
>
---
#### [new 030] POINTS-GUI-G: GUI-Grounding Journey
- **分类: cs.CV**

- **简介: 该论文属于GUI接地任务，旨在提升模型精准定位界面元素的能力。通过数据工程、训练策略优化和强化学习，提出新模型POINT-GUI-G-8B，显著提升多项指标表现。**

- **链接: [https://arxiv.org/pdf/2602.06391v1](https://arxiv.org/pdf/2602.06391v1)**

> **作者:** Zhongyin Zhao; Yuan Liu; Yikun Liu; Haicheng Wang; Le Tian; Xiao Zhou; Yangxiu You; Zilin Yu; Yang Yu; Jie Zhou
>
> **摘要:** The rapid advancement of vision-language models has catalyzed the emergence of GUI agents, which hold immense potential for automating complex tasks, from online shopping to flight booking, thereby alleviating the burden of repetitive digital workflows. As a foundational capability, GUI grounding is typically established as a prerequisite for end-to-end task execution. It enables models to precisely locate interface elements, such as text and icons, to perform accurate operations like clicking and typing. Unlike prior works that fine-tune models already possessing strong spatial awareness (e.g., Qwen3-VL), we aim to master the full technical pipeline by starting from a base model with minimal grounding ability, such as POINTS-1.5. We introduce POINTS-GUI-G-8B, which achieves state-of-the-art performance with scores of 59.9 on ScreenSpot-Pro, 66.0 on OSWorld-G, 95.7 on ScreenSpot-v2, and 49.9 on UI-Vision. Our model's success is driven by three key factors: (1) Refined Data Engineering, involving the unification of diverse open-source datasets format alongside sophisticated strategies for augmentation, filtering, and difficulty grading; (2) Improved Training Strategies, including continuous fine-tuning of the vision encoder to enhance perceptual accuracy and maintaining resolution consistency between training and inference; and (3) Reinforcement Learning (RL) with Verifiable Rewards. While RL is traditionally used to bolster reasoning, we demonstrate that it significantly improves precision in the perception-intensive GUI grounding task. Furthermore, GUI grounding provides a natural advantage for RL, as rewards are easily verifiable and highly accurate.
>
---
#### [new 031] Unsupervised Anomaly Detection of Diseases in the Female Pelvis for Real-Time MR Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像异常检测任务，旨在解决女性骨盆MRI中疾病诊断延迟问题。通过无监督方法构建实时异常检测框架，利用健康数据训练模型，实现病灶区域的自动识别。**

- **链接: [https://arxiv.org/pdf/2602.06179v1](https://arxiv.org/pdf/2602.06179v1)**

> **作者:** Anika Knupfer; Johanna P. Müller; Jordina A. Verdera; Martin Fenske; Claudius S. Mathy; Smiti Tripathy; Sebastian Arndt; Matthias May; Michael Uder; Matthias W. Beckmann; Stefanie Burghaus; Jana Hutter
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Pelvic diseases in women of reproductive age represent a major global health burden, with diagnosis frequently delayed due to high anatomical variability, complicating MRI interpretation. Existing AI approaches are largely disease-specific and lack real-time compatibility, limiting generalizability and clinical integration. To address these challenges, we establish a benchmark framework for disease- and parameter-agnostic, real-time-compatible unsupervised anomaly detection in pelvic MRI. The method uses a residual variational autoencoder trained exclusively on healthy sagittal T2-weighted scans acquired across diverse imaging protocols to model normal pelvic anatomy. During inference, reconstruction error heatmaps indicate deviations from learned healthy structure, enabling detection of pathological regions without labeled abnormal data. The model is trained on 294 healthy scans and augmented with diffusion-generated synthetic data to improve robustness. Quantitative evaluation on the publicly available Uterine Myoma MRI Dataset yields an average area-under-the-curve (AUC) value of 0.736, with 0.828 sensitivity and 0.692 specificity. Additional inter-observer clinical evaluation extends analysis to endometrial cancer, endometriosis, and adenomyosis, revealing the influence of anatomical heterogeneity and inter-observer variability on performance interpretation. With a reconstruction time of approximately 92.6 frames per second, the proposed framework establishes a baseline for unsupervised anomaly detection in the female pelvis and supports future integration into real-time MRI. Code is available upon request (https://github.com/AniKnu/UADPelvis), prospective data sets are available for academic collaboration.
>
---
#### [new 032] PANC: Prior-Aware Normalized Cut for Object Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出PANC方法，解决弱监督下的对象分割问题。通过引入先验信息优化图谱结构，提升分割的稳定性与可控性，实现高质量、可重复的分割结果。**

- **链接: [https://arxiv.org/pdf/2602.06912v1](https://arxiv.org/pdf/2602.06912v1)**

> **作者:** Juan Gutiérrez; Victor Gutiérrez-Garcia; José Luis Blanco-Murillo
>
> **摘要:** Fully unsupervised segmentation pipelines naively seek the most salient object, should this be present. As a result, most of the methods reported in the literature deliver non-deterministic partitions that are sensitive to initialization, seed order, and threshold heuristics. We propose PANC, a weakly supervised spectral segmentation framework that uses a minimal set of annotated visual tokens to produce stable, controllable, and reproducible object masks. From the TokenCut approach, we augment the token-token affinity graph with a handful of priors coupled to anchor nodes. By manipulating the graph topology, we bias the spectral eigenspace toward partitions that are consistent with the annotations. Our approach preserves the global grouping enforced by dense self-supervised visual features, trading annotated tokens for significant gains in reproducibility, user control, and segmentation quality. Using 5 to 30 annotations per dataset, our training-free method achieves state-of-the-art performance among weakly and unsupervised approaches on standard benchmarks (e.g., DUTS-TE, ECSSD, MS COCO). Contrarily, it excels in domains where dense labels are costly or intra-class differences are subtle. We report strong and reliable results on homogeneous, fine-grained, and texture-limited domains, achieving 96.8% (+14.43% over SotA), 78.0% (+0.2%), and 78.8% (+0.37%) average mean intersection-over-union (mIoU) on CrackForest (CFD), CUB-200-2011, and HAM10000 datasets, respectively. For multi-object benchmarks, the framework showcases explicit, user-controllable semantic segmentation.
>
---
#### [new 033] Rethinking Multi-Condition DiTs: Eliminating Redundant Attention via Position-Alignment and Keyword-Scoping
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于文本到图像生成任务，旨在解决多条件控制中的计算冗余问题。提出PKA框架，通过位置对齐和关键词作用域注意力提高效率，减少计算开销。**

- **链接: [https://arxiv.org/pdf/2602.06850v1](https://arxiv.org/pdf/2602.06850v1)**

> **作者:** Chao Zhou; Tianyi Wei; Yiling Chen; Wenbo Zhou; Nenghai Yu
>
> **摘要:** While modern text-to-image models excel at prompt-based generation, they often lack the fine-grained control necessary for specific user requirements like spatial layouts or subject appearances. Multi-condition control addresses this, yet its integration into Diffusion Transformers (DiTs) is bottlenecked by the conventional ``concatenate-and-attend'' strategy, which suffers from quadratic computational and memory overhead as the number of conditions scales. Our analysis reveals that much of this cross-modal interaction is spatially or semantically redundant. To this end, we propose Position-aligned and Keyword-scoped Attention (PKA), a highly efficient framework designed to eliminate these redundancies. Specifically, Position-Aligned Attention (PAA) linearizes spatial control by enforcing localized patch alignment, while Keyword-Scoped Attention (KSA) prunes irrelevant subject-driven interactions via semantic-aware masking. To facilitate efficient learning, we further introduce a Conditional Sensitivity-Aware Sampling (CSAS) strategy that reweights the training objective towards critical denoising phases, drastically accelerating convergence and enhancing conditional fidelity. Empirically, PKA delivers a 10.0$\times$ inference speedup and a 5.1$\times$ VRAM saving, providing a scalable and resource-friendly solution for high-fidelity multi-conditioned generation.
>
---
#### [new 034] Clinical-Prior Guided Multi-Modal Learning with Latent Attention Pooling for Gait-Based Scoliosis Screening
- **分类: cs.CV**

- **简介: 该论文属于脊柱侧弯筛查任务，旨在解决传统方法主观性强、难以扩展的问题。提出ScoliGait数据集和多模态框架，结合临床先验与注意力机制，提升检测准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2602.06743v1](https://arxiv.org/pdf/2602.06743v1)**

> **作者:** Dong Chen; Zizhuang Wei; Jialei Xu; Xinyang Sun; Zonglin He; Meiru An; Huili Peng; Yong Hu; Kenneth MC Cheung
>
> **摘要:** Adolescent Idiopathic Scoliosis (AIS) is a prevalent spinal deformity whose progression can be mitigated through early detection. Conventional screening methods are often subjective, difficult to scale, and reliant on specialized clinical expertise. Video-based gait analysis offers a promising alternative, but current datasets and methods frequently suffer from data leakage, where performance is inflated by repeated clips from the same individual, or employ oversimplified models that lack clinical interpretability. To address these limitations, we introduce ScoliGait, a new benchmark dataset comprising 1,572 gait video clips for training and 300 fully independent clips for testing. Each clip is annotated with radiographic Cobb angles and descriptive text based on clinical kinematic priors. We propose a multi-modal framework that integrates a clinical-prior-guided kinematic knowledge map for interpretable feature representation, alongside a latent attention pooling mechanism to fuse video, text, and knowledge map modalities. Our method establishes a new state-of-the-art, demonstrating a significant performance gap on a realistic, non-repeating subject benchmark. Our approach establishes a new state of the art, showing a significant performance gain on a realistic, subject-independent benchmark. This work provides a robust, interpretable, and clinically grounded foundation for scalable, non-invasive AIS assessment.
>
---
#### [new 035] SPDA-SAM: A Self-prompted Depth-Aware Segment Anything Model for Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于实例分割任务，旨在解决SAM依赖人工提示和缺乏深度信息的问题。提出SPDA-SAM，结合自提示模块和RGB-D融合，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2602.06335v1](https://arxiv.org/pdf/2602.06335v1)**

> **作者:** Yihan Shang; Wei Wang; Chao Huang; Xinghui Dong
>
> **摘要:** Recently, Segment Anything Model (SAM) has demonstrated strong generalizability in various instance segmentation tasks. However, its performance is severely dependent on the quality of manual prompts. In addition, the RGB images that instance segmentation methods normally use inherently lack depth information. As a result, the ability of these methods to perceive spatial structures and delineate object boundaries is hindered. To address these challenges, we propose a Self-prompted Depth-Aware SAM (SPDA-SAM) for instance segmentation. Specifically, we design a Semantic-Spatial Self-prompt Module (SSSPM) which extracts the semantic and spatial prompts from the image encoder and the mask decoder of SAM, respectively. Furthermore, we introduce a Coarse-to-Fine RGB-D Fusion Module (C2FFM), in which the features extracted from a monocular RGB image and the depth map estimated from it are fused. In particular, the structural information in the depth map is used to provide coarse-grained guidance to feature fusion, while local variations in depth are encoded in order to fuse fine-grained feature representations. To our knowledge, SAM has not been explored in such self-prompted and depth-aware manners. Experimental results demonstrate that our SPDA-SAM outperforms its state-of-the-art counterparts across twelve different data sets. These promising results should be due to the guidance of the self-prompts and the compensation for the spatial information loss by the coarse-to-fine RGB-D fusion operation.
>
---
#### [new 036] NECromancer: Breathing Life into Skeletons via BVH Animation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出NECromancer，解决跨物种运动建模问题，通过通用运动分词器实现不同骨骼结构的运动迁移与生成。**

- **链接: [https://arxiv.org/pdf/2602.06548v1](https://arxiv.org/pdf/2602.06548v1)**

> **作者:** Mingxi Xu; Qi Wang; Zhengyu Wen; Phong Dao Thien; Zhengyu Li; Ning Zhang; Xiaoyu He; Wei Zhao; Kehong Gong; Mingyuan Zhang
>
> **摘要:** Motion tokenization is a key component of generalizable motion models, yet most existing approaches are restricted to species-specific skeletons, limiting their applicability across diverse morphologies. We propose NECromancer (NEC), a universal motion tokenizer that operates directly on arbitrary BVH skeletons. NEC consists of three components: (1) an Ontology-aware Skeletal Graph Encoder (OwO) that encodes structural priors from BVH files, including joint semantics, rest-pose offsets, and skeletal topology, into skeletal embeddings; (2) a Topology-Agnostic Tokenizer (TAT) that compresses motion sequences into a universal, topology-invariant discrete representation; and (3) the Unified BVH Universe (UvU), a large-scale dataset aggregating BVH motions across heterogeneous skeletons. Experiments show that NEC achieves high-fidelity reconstruction under substantial compression and effectively disentangles motion from skeletal structure. The resulting token space supports cross-species motion transfer, composition, denoising, generation with token-based models, and text-motion retrieval, establishing a unified framework for motion analysis and synthesis across diverse morphologies. Demo page: https://animotionlab.github.io/NECromancer/
>
---
#### [new 037] Universal Anti-forensics Attack against Image Forgery Detection via Multi-modal Guidance
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于图像伪造检测任务，旨在解决AIGC检测器易受反取证攻击的问题。提出ForgeryEraser框架，通过多模态引导损失降低检测性能，有效隐藏伪造痕迹。**

- **链接: [https://arxiv.org/pdf/2602.06530v1](https://arxiv.org/pdf/2602.06530v1)**

> **作者:** Haipeng Li; Rongxuan Peng; Anwei Luo; Shunquan Tan; Changsheng Chen; Anastasia Antsiferova
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** The rapid advancement of AI-Generated Content (AIGC) technologies poses significant challenges for authenticity assessment. However, existing evaluation protocols largely overlook anti-forensics attack, failing to ensure the comprehensive robustness of state-of-the-art AIGC detectors in real-world applications. To bridge this gap, we propose ForgeryEraser, a framework designed to execute universal anti-forensics attack without access to the target AIGC detectors. We reveal an adversarial vulnerability stemming from the systemic reliance on Vision-Language Models (VLMs) as shared backbones (e.g., CLIP), where downstream AIGC detectors inherit the feature space of these publicly accessible models. Instead of traditional logit-based optimization, we design a multi-modal guidance loss to drive forged image embeddings within the VLM feature space toward text-derived authentic anchors to erase forgery traces, while repelling them from forgery anchors. Extensive experiments demonstrate that ForgeryEraser causes substantial performance degradation to advanced AIGC detectors on both global synthesis and local editing benchmarks. Moreover, ForgeryEraser induces explainable forensic models to generate explanations consistent with authentic images for forged images. Our code will be made publicly available.
>
---
#### [new 038] From Blurry to Believable: Enhancing Low-quality Talking Heads with 3D Generative Priors
- **分类: cs.CV**

- **简介: 该论文属于3D头像重建任务，旨在提升低质量视频中头像的细节与动画一致性。通过引入SuperHead框架，利用3D生成模型先验，优化生成模型的潜在表示，以获得高质量的3D头像模型。**

- **链接: [https://arxiv.org/pdf/2602.06122v1](https://arxiv.org/pdf/2602.06122v1)**

> **作者:** Ding-Jiun Huang; Yuanhao Wang; Shao-Ji Yuan; Albert Mosella-Montoro; Francisco Vicente Carrasco; Cheng Zhang; Fernando De la Torre
>
> **备注:** Accepted to 3DV 2026. Project Page: https://humansensinglab.github.io/super-head/
>
> **摘要:** Creating high-fidelity, animatable 3D talking heads is crucial for immersive applications, yet often hindered by the prevalence of low-quality image or video sources, which yield poor 3D reconstructions. In this paper, we introduce SuperHead, a novel framework for enhancing low-resolution, animatable 3D head avatars. The core challenge lies in synthesizing high-quality geometry and textures, while ensuring both 3D and temporal consistency during animation and preserving subject identity. Despite recent progress in image, video and 3D-based super-resolution (SR), existing SR techniques are ill-equipped to handle dynamic 3D inputs. To address this, SuperHead leverages the rich priors from pre-trained 3D generative models via a novel dynamics-aware 3D inversion scheme. This process optimizes the latent representation of the generative model to produce a super-resolved 3D Gaussian Splatting (3DGS) head model, which is subsequently rigged to an underlying parametric head model (e.g., FLAME) for animation. The inversion is jointly supervised using a sparse collection of upscaled 2D face renderings and corresponding depth maps, captured from diverse facial expressions and camera viewpoints, to ensure realism under dynamic facial motions. Experiments demonstrate that SuperHead generates avatars with fine-grained facial details under dynamic motions, significantly outperforming baseline methods in visual quality.
>
---
#### [new 039] ASMa: Asymmetric Spatio-temporal Masking for Skeleton Action Representation Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于骨架动作表示学习任务，旨在解决现有自监督学习方法在动作识别中特征表示不均衡的问题。提出ASMa方法，通过不对称时空掩码提升表示的全面性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06251v1](https://arxiv.org/pdf/2602.06251v1)**

> **作者:** Aman Anand; Amir Eskandari; Elyas Rahsno; Farhana Zulkernine
>
> **摘要:** Self-supervised learning (SSL) has shown remarkable success in skeleton-based action recognition by leveraging data augmentations to learn meaningful representations. However, existing SSL methods rely on data augmentations that predominantly focus on masking high-motion frames and high-degree joints such as joints with degree 3 or 4. This results in biased and incomplete feature representations that struggle to generalize across varied motion patterns. To address this, we propose Asymmetric Spatio-temporal Masking (ASMa) for Skeleton Action Representation Learning, a novel combination of masking to learn a full spectrum of spatio-temporal dynamics inherent in human actions. ASMa employs two complementary masking strategies: one that selectively masks high-degree joints and low-motion, and another that masks low-degree joints and high-motion frames. These masking strategies ensure a more balanced and comprehensive skeleton representation learning. Furthermore, we introduce a learnable feature alignment module to effectively align the representations learned from both masked views. To facilitate deployment in resource-constrained settings and on low-resource devices, we compress the learned and aligned representation into a lightweight model using knowledge distillation. Extensive experiments on NTU RGB+D 60, NTU RGB+D 120, and PKU-MMD datasets demonstrate that our approach outperforms existing SSL methods with an average improvement of 2.7-4.4% in fine-tuning and up to 5.9% in transfer learning to noisy datasets and achieves competitive performance compared to fully supervised baselines. Our distilled model achieves 91.4% parameter reduction and 3x faster inference on edge devices while maintaining competitive accuracy, enabling practical deployment in resource-constrained scenarios.
>
---
#### [new 040] Parameters as Experts: Adapting Vision Models with Dynamic Parameter Routing
- **分类: cs.CV**

- **简介: 该论文属于视觉模型适应任务，解决参数高效微调的挑战。提出AdaRoute方法，通过动态参数路由实现低秩适配，提升特征表示能力。**

- **链接: [https://arxiv.org/pdf/2602.06862v1](https://arxiv.org/pdf/2602.06862v1)**

> **作者:** Meng Lou; Stanley Yu; Yizhou Yu
>
> **摘要:** Adapting pre-trained vision models using parameter-efficient fine-tuning (PEFT) remains challenging, as it aims to achieve performance comparable to full fine-tuning using a minimal number of trainable parameters. When applied to complex dense prediction tasks, existing methods exhibit limitations, including input-agnostic modeling and redundant cross-layer representations. To this end, we propose AdaRoute, a new adapter-style method featuring a simple mixture-of-experts (MoE) architecture. Specifically, we introduce shared expert centers, where each expert is a trainable parameter matrix. During a feedforward pass, each AdaRoute module in the network dynamically generates weight matrices tailored for the current module via a simple dynamic parameter routing mechanism, which selectively aggregates parameter matrices in the corresponding expert center. Dynamic weight matrices in AdaRoute modules facilitate low-rank adaptation in an input-dependent manner, thus generating more customized and powerful feature representations. Moreover, since AdaRoute modules across multiple network layers share the same expert center, they improve feature diversity by promoting implicit cross-layer feature interaction. Extensive experiments demonstrate the superiority of AdaRoute on diverse vision tasks, including semantic segmentation, object detection and instance segmentation, and panoptic segmentation. Code will be available at: https://bit.ly/3NZcr0H.
>
---
#### [new 041] Instance-Free Domain Adaptive Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测领域，解决无目标实例的域适应问题。提出RSCN网络，通过背景特征原型实现域对齐，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2602.06484v1](https://arxiv.org/pdf/2602.06484v1)**

> **作者:** Hengfu Yu; Jinhong Deng; Lixin Duan; Wen Li
>
> **备注:** 14 pages, 12 figures
>
> **摘要:** While Domain Adaptive Object Detection (DAOD) has made significant strides, most methods rely on unlabeled target data that is assumed to contain sufficient foreground instances. However, in many practical scenarios (e.g., wildlife monitoring, lesion detection), collecting target domain data with objects of interest is prohibitively costly, whereas background-only data is abundant. This common practical constraint introduces a significant technical challenge: the difficulty of achieving domain alignment when target instances are unavailable, forcing adaptation to rely solely on the target background information. We formulate this challenge as the novel problem of Instance-Free Domain Adaptive Object Detection. To tackle this, we propose the Relational and Structural Consistency Network (RSCN) which pioneers an alignment strategy based on background feature prototypes while simultaneously encouraging consistency in the relationship between the source foreground features and the background features within each domain, enabling robust adaptation even without target instances. To facilitate research, we further curate three specialized benchmarks, including simulative auto-driving detection, wildlife detection, and lung nodule detection. Extensive experiments show that RSCN significantly outperforms existing DAOD methods across all three benchmarks in the instance-free scenario. The code and benchmarks will be released soon.
>
---
#### [new 042] Unsupervised MRI-US Multimodal Image Registration with Multilevel Correlation Pyramidal Optimization
- **分类: cs.CV**

- **简介: 该论文属于多模态医学图像配准任务，旨在解决术前与术中图像因模态差异和变形带来的配准难题。提出MCPO方法，通过多级相关金字塔优化实现精准配准。**

- **链接: [https://arxiv.org/pdf/2602.06288v1](https://arxiv.org/pdf/2602.06288v1)**

> **作者:** Jiazheng Wang; Zeyu Liu; Min Liu; Xiang Chen; Hang Zhang
>
> **备注:** first-place method of ReMIND2Reg Learn2Reg 2025 (in MICCAI 2025)
>
> **摘要:** Surgical navigation based on multimodal image registration has played a significant role in providing intraoperative guidance to surgeons by showing the relative position of the target area to critical anatomical structures during surgery. However, due to the differences between multimodal images and intraoperative image deformation caused by tissue displacement and removal during the surgery, effective registration of preoperative and intraoperative multimodal images faces significant challenges. To address the multimodal image registration challenges in Learn2Reg 2025, an unsupervised multimodal medical image registration method based on multilevel correlation pyramidal optimization (MCPO) is designed to solve these problems. First, the features of each modality are extracted based on the modality independent neighborhood descriptor, and the multimodal images is mapped to the feature space. Second, a multilevel pyramidal fusion optimization mechanism is designed to achieve global optimization and local detail complementation of the displacement field through dense correlation analysis and weight-balanced coupled convex optimization for input features at different scales. Our method focuses on the ReMIND2Reg task in Learn2Reg 2025. Based on the results, our method achieved the first place in the validation phase and test phase of ReMIND2Reg. The MCPO is also validated on the Resect dataset, achieving an average TRE of 1.798 mm. This demonstrates the broad applicability of our method in preoperative-to-intraoperative image registration. The code is avaliable at https://github.com/wjiazheng/MCPO.
>
---
#### [new 043] SPARC: Separating Perception And Reasoning Circuits for Test-time Scaling of VLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出SPARC框架，解决视觉语言模型测试时扩展的稳定性问题。通过分离感知与推理模块，提升模型在复杂任务中的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.06566v1](https://arxiv.org/pdf/2602.06566v1)**

> **作者:** Niccolo Avogaro; Nayanika Debnath; Li Mi; Thomas Frick; Junling Wang; Zexue He; Hang Hua; Konrad Schindler; Mattia Rigotti
>
> **摘要:** Despite recent successes, test-time scaling - i.e., dynamically expanding the token budget during inference as needed - remains brittle for vision-language models (VLMs): unstructured chains-of-thought about images entangle perception and reasoning, leading to long, disorganized contexts where small perceptual mistakes may cascade into completely wrong answers. Moreover, expensive reinforcement learning with hand-crafted rewards is required to achieve good performance. Here, we introduce SPARC (Separating Perception And Reasoning Circuits), a modular framework that explicitly decouples visual perception from reasoning. Inspired by sequential sensory-to-cognitive processing in the brain, SPARC implements a two-stage pipeline where the model first performs explicit visual search to localize question-relevant regions, then conditions its reasoning on those regions to produce the final answer. This separation enables independent test-time scaling with asymmetric compute allocation (e.g., prioritizing perceptual processing under distribution shift), supports selective optimization (e.g., improving the perceptual stage alone when it is the bottleneck for end-to-end performance), and accommodates compressed contexts by running global search at lower image resolutions and allocating high-resolution processing only to selected regions, thereby reducing total visual tokens count and compute. Across challenging visual reasoning benchmarks, SPARC outperforms monolithic baselines and strong visual-grounding approaches. For instance, SPARC improves the accuracy of Qwen3VL-4B on the $V^*$ VQA benchmark by 6.7 percentage points, and it surpasses "thinking with images" by 4.6 points on a challenging OOD task despite requiring a 200$\times$ lower token budget.
>
---
#### [new 044] Uncertainty-Aware 4D Gaussian Splatting for Monocular Occluded Human Rendering
- **分类: cs.CV**

- **简介: 该论文属于单目人体渲染任务，解决遮挡下渲染质量下降问题。提出U-4DGS框架，结合概率变形网络和双光栅化管道，生成不确定性图以提升渲染精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06343v1](https://arxiv.org/pdf/2602.06343v1)**

> **作者:** Weiquan Wang; Feifei Shao; Lin Li; Zhen Wang; Jun Xiao; Long Chen
>
> **摘要:** High-fidelity rendering of dynamic humans from monocular videos typically degrades catastrophically under occlusions. Existing solutions incorporate external priors-either hallucinating missing content via generative models, which induces severe temporal flickering, or imposing rigid geometric heuristics that fail to capture diverse appearances. To this end, we reformulate the task as a Maximum A Posteriori estimation problem under heteroscedastic observation noise. In this paper, we propose U-4DGS, a framework integrating a Probabilistic Deformation Network and a Double Rasterization pipeline. This architecture renders pixel-aligned uncertainty maps that act as an adaptive gradient modulator, automatically attenuating artifacts from unreliable observations. Furthermore, to prevent geometric drift in regions lacking reliable visual cues, we enforce Confidence-Aware Regularizations, which leverage the learned uncertainty to selectively propagate spatial-temporal validity. Extensive experiments on ZJU-MoCap and OcMotion demonstrate that U-4DGS achieves SOTA rendering fidelity and robustness.
>
---
#### [new 045] Taming SAM3 in the Wild: A Concept Bank for Open-Vocabulary Segmentation
- **分类: cs.CV**

- **简介: 该论文属于开放词汇分割任务，解决模型在分布漂移下的性能下降问题。提出ConceptBank框架，通过动态构建概念库来恢复视觉与提示的对齐。**

- **链接: [https://arxiv.org/pdf/2602.06333v1](https://arxiv.org/pdf/2602.06333v1)**

> **作者:** Gensheng Pei; Xiruo Jiang; Yazhou Yao; Xiangbo Shu; Fumin Shen; Byeungwoo Jeon
>
> **摘要:** The recent introduction of \texttt{SAM3} has revolutionized Open-Vocabulary Segmentation (OVS) through \textit{promptable concept segmentation}, which grounds pixel predictions in flexible concept prompts. However, this reliance on pre-defined concepts makes the model vulnerable: when visual distributions shift (\textit{data drift}) or conditional label distributions evolve (\textit{concept drift}) in the target domain, the alignment between visual evidence and prompts breaks down. In this work, we present \textsc{ConceptBank}, a parameter-free calibration framework to restore this alignment on the fly. Instead of adhering to static prompts, we construct a dataset-specific concept bank from the target statistics. Our approach (\textit{i}) anchors target-domain evidence via class-wise visual prototypes, (\textit{ii}) mines representative supports to suppress outliers under data drift, and (\textit{iii}) fuses candidate concepts to rectify concept drift. We demonstrate that \textsc{ConceptBank} effectively adapts \texttt{SAM3} to distribution drifts, including challenging natural-scene and remote-sensing scenarios, establishing a new baseline for robustness and efficiency in OVS. Code and model are available at https://github.com/pgsmall/ConceptBank.
>
---
#### [new 046] ProtoQuant: Quantization of Prototypical Parts For General and Fine-Grained Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出ProtoQuant，解决图像分类中的可解释性与泛化问题。通过原型量化实现稳定且可解释的模型，无需微调主干网络，适用于大规模数据集。**

- **链接: [https://arxiv.org/pdf/2602.06592v1](https://arxiv.org/pdf/2602.06592v1)**

> **作者:** Mikołaj Janusz; Adam Wróbel; Bartosz Zieliński; Dawid Rymarczyk
>
> **备注:** Work under review. Code will be released upon acceptance
>
> **摘要:** Prototypical parts-based models offer a "this looks like that" paradigm for intrinsic interpretability, yet they typically struggle with ImageNet-scale generalization and often require computationally expensive backbone finetuning. Furthermore, existing methods frequently suffer from "prototype drift," where learned prototypes lack tangible grounding in the training distribution and change their activation under small perturbations. We present ProtoQuant, a novel architecture that achieves prototype stability and grounded interpretability through latent vector quantization. By constraining prototypes to a discrete learned codebook within the latent space, we ensure they remain faithful representations of the training data without the need to update the backbone. This design allows ProtoQuant to function as an efficient, interpretable head that scales to large-scale datasets. We evaluate ProtoQuant on ImageNet and several fine-grained benchmarks (CUB-200, Cars-196). Our results demonstrate that ProtoQuant achieves competitive classification accuracy while generalizing to ImageNet and comparable interpretability metrics to other prototypical-parts-based methods.
>
---
#### [new 047] Efficient-LVSM: Faster, Cheaper, and Better Large View Synthesis Model via Decoupled Co-Refinement Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Efficient-LVSM，解决视图合成任务中的计算效率与性能问题。通过双流结构和解耦注意力机制，提升速度与效果。**

- **链接: [https://arxiv.org/pdf/2602.06478v1](https://arxiv.org/pdf/2602.06478v1)**

> **作者:** Xiaosong Jia; Yihang Sun; Junqi You; Songbur Wong; Zichen Zou; Junchi Yan; Zuxuan Wu; Yu-Gang Jiang
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Feedforward models for novel view synthesis (NVS) have recently advanced by transformer-based methods like LVSM, using attention among all input and target views. In this work, we argue that its full self-attention design is suboptimal, suffering from quadratic complexity with respect to the number of input views and rigid parameter sharing among heterogeneous tokens. We propose Efficient-LVSM, a dual-stream architecture that avoids these issues with a decoupled co-refinement mechanism. It applies intra-view self-attention for input views and self-then-cross attention for target views, eliminating unnecessary computation. Efficient-LVSM achieves 29.86 dB PSNR on RealEstate10K with 2 input views, surpassing LVSM by 0.2 dB, with 2x faster training convergence and 4.4x faster inference speed. Efficient-LVSM achieves state-of-the-art performance on multiple benchmarks, exhibits strong zero-shot generalization to unseen view counts, and enables incremental inference with KV-cache, thanks to its decoupled designs.
>
---
#### [new 048] GaussianPOP: Principled Simplification Framework for Compact 3D Gaussian Splatting via Error Quantification
- **分类: cs.CV**

- **简介: 该论文属于3D高斯点云简化任务，解决现有方法依赖不准确重要性评分导致的压缩与质量不平衡问题。提出GaussianPOP框架，通过分析误差量化实现高效简化。**

- **链接: [https://arxiv.org/pdf/2602.06830v1](https://arxiv.org/pdf/2602.06830v1)**

> **作者:** Soonbin Lee; Yeong-Gyu Kim; Simon Sasse; Tomas M. Borges; Yago Sanchez; Eun-Seok Ryu; Thomas Schierl; Cornelius Hellge
>
> **摘要:** Existing 3D Gaussian Splatting simplification methods commonly use importance scores, such as blending weights or sensitivity, to identify redundant Gaussians. However, these scores are not driven by visual error metrics, often leading to suboptimal trade-offs between compactness and rendering fidelity. We present GaussianPOP, a principled simplification framework based on analytical Gaussian error quantification. Our key contribution is a novel error criterion, derived directly from the 3DGS rendering equation, that precisely measures each Gaussian's contribution to the rendered image. By introducing a highly efficient algorithm, our framework enables practical error calculation in a single forward pass. The framework is both accurate and flexible, supporting on-training pruning as well as post-training simplification via iterative error re-quantification for improved stability. Experimental results show that our method consistently outperforms existing state-of-the-art pruning methods across both application scenarios, achieving a superior trade-off between model compactness and high rendering quality.
>
---
#### [new 049] An Interpretable Vision Transformer as a Fingerprint-Based Diagnostic Aid for Kabuki and Wiedemann-Steiner Syndromes
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于医学诊断任务，旨在解决KS和WSS早期诊断困难的问题。通过构建视觉Transformer模型，利用指纹图像进行分类，并提升模型可解释性。**

- **链接: [https://arxiv.org/pdf/2602.06282v1](https://arxiv.org/pdf/2602.06282v1)**

> **作者:** Marilyn Lionts; Arnhildur Tomasdottir; Viktor I. Agustsson; Yuankai Huo; Hans T. Bjornsson; Lotta M. Ellingsen
>
> **摘要:** Kabuki syndrome (KS) and Wiedemann-Steiner syndrome (WSS) are rare but distinct developmental disorders that share overlapping clinical features, including neurodevelopmental delay, growth restriction, and persistent fetal fingertip pads. While genetic testing remains the diagnostic gold standard, many individuals with KS or WSS remain undiagnosed due to barriers in access to both genetic testing and expertise. Dermatoglyphic anomalies, despite being established hallmarks of several genetic syndromes, remain an underutilized diagnostic signal in the era of molecular testing. This study presents a vision transformer-based deep learning model that leverages fingerprint images to distinguish individuals with KS and WSS from unaffected controls and from one another. We evaluate model performance across three binary classification tasks. Across the three classification tasks, the model achieved AUC scores of 0.80 (control vs. KS), 0.73 (control vs. WSS), and 0.85 (KS vs. WSS), with corresponding F1 scores of 0.71, 0.72, and 0.83, respectively. Beyond classification, we apply attention-based visualizations to identify fingerprint regions most salient to model predictions, enhancing interpretability. Together, these findings suggest the presence of syndrome-specific fingerprint features, demonstrating the feasibility of a fingerprint-based artificial intelligence (AI) tool as a noninvasive, interpretable, and accessible future diagnostic aid for the early diagnosis of underdiagnosed genetic syndromes.
>
---
#### [new 050] ChatUMM: Robust Context Tracking for Conversational Interleaved Generation
- **分类: cs.CV**

- **简介: 该论文提出ChatUMM，解决对话中多轮上下文跟踪问题，通过创新训练策略和数据合成方法，提升多模态生成的连贯性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06442v1](https://arxiv.org/pdf/2602.06442v1)**

> **作者:** Wenxun Dai; Zhiyuan Zhao; Yule Zhong; Yiji Cheng; Jianwei Zhang; Linqing Wang; Shiyi Zhang; Yunlong Lin; Runze He; Fellix Song; Wayne Zhuang; Yong Liu; Haoji Zhang; Yansong Tang; Qinglin Lu; Chunyu Wang
>
> **备注:** ChatUMM Project
>
> **摘要:** Unified multimodal models (UMMs) have achieved remarkable progress yet remain constrained by a single-turn interaction paradigm, effectively functioning as solvers for independent requests rather than assistants in continuous dialogue. To bridge this gap, we present ChatUMM. As a conversational unified model, it excels at robust context tracking to sustain interleaved multimodal generation. ChatUMM derives its capabilities from two key innovations: an interleaved multi-turn training strategy that models serialized text-image streams as a continuous conversational flow, and a systematic conversational data synthesis pipeline. This pipeline transforms a diverse set of standard single-turn datasets into fluid dialogues through three progressive stages: constructing basic stateful dialogues, enforcing long-range dependency resolution via ``distractor'' turns with history-dependent query rewriting, and synthesizing naturally interleaved multimodal responses. Extensive evaluations demonstrate that ChatUMM achieves state-of-the-art performance among open-source unified models on visual understanding and instruction-guided editing benchmarks, while maintaining competitive fidelity in text-to-image generation. Notably, ChatUMM exhibits superior robustness in complex multi-turn scenarios, ensuring fluid, context-aware dialogues.
>
---
#### [new 051] Alleviating Sparse Rewards by Modeling Step-Wise and Long-Term Sampling Effects in Flow-Based GRPO
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决稀疏奖励问题。通过引入步骤级奖励和转折点机制，提升模型对长期影响的建模能力，优化生成效果。**

- **链接: [https://arxiv.org/pdf/2602.06422v1](https://arxiv.org/pdf/2602.06422v1)**

> **作者:** Yunze Tong; Mushui Liu; Canyu Zhao; Wanggui He; Shiyi Zhang; Hongwei Zhang; Peng Zhang; Jinlong Liu; Ju Huang; Jiamang Wang; Hao Jiang; Pipei Huang
>
> **备注:** 18 pages, in submission
>
> **摘要:** Deploying GRPO on Flow Matching models has proven effective for text-to-image generation. However, existing paradigms typically propagate an outcome-based reward to all preceding denoising steps without distinguishing the local effect of each step. Moreover, current group-wise ranking mainly compares trajectories at matched timesteps and ignores within-trajectory dependencies, where certain early denoising actions can affect later states via delayed, implicit interactions. We propose TurningPoint-GRPO (TP-GRPO), a GRPO framework that alleviates step-wise reward sparsity and explicitly models long-term effects within the denoising trajectory. TP-GRPO makes two key innovations: (i) it replaces outcome-based rewards with step-level incremental rewards, providing a dense, step-aware learning signal that better isolates each denoising action's "pure" effect, and (ii) it identifies turning points-steps that flip the local reward trend and make subsequent reward evolution consistent with the overall trajectory trend-and assigns these actions an aggregated long-term reward to capture their delayed impact. Turning points are detected solely via sign changes in incremental rewards, making TP-GRPO efficient and hyperparameter-free. Extensive experiments also demonstrate that TP-GRPO exploits reward signals more effectively and consistently improves generation. Demo code is available at https://github.com/YunzeTong/TurningPoint-GRPO.
>
---
#### [new 052] An Integer Linear Programming Approach to Geometrically Consistent Partial-Partial Shape Matching
- **分类: cs.CV**

- **简介: 该论文属于3D形状匹配任务，解决部分-部分匹配问题，提出一种整数线性规划方法，利用几何一致性提升匹配精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06590v1](https://arxiv.org/pdf/2602.06590v1)**

> **作者:** Viktoria Ehm; Paul Roetzer; Florian Bernard; Daniel Cremers
>
> **摘要:** The task of establishing correspondences between two 3D shapes is a long-standing challenge in computer vision. While numerous studies address full-full and partial-full 3D shape matching, only a limited number of works have explored the partial-partial setting, very likely due to its unique challenges: we must compute accurate correspondences while at the same time find the unknown overlapping region. Nevertheless, partial-partial 3D shape matching reflects the most realistic setting, as in many real-world cases, such as 3D scanning, shapes are only partially observable. In this work, we introduce the first integer linear programming approach specifically designed to address the distinctive challenges of partial-partial shape matching. Our method leverages geometric consistency as a strong prior, enabling both robust estimation of the overlapping region and computation of neighbourhood-preserving correspondences. We empirically demonstrate that our approach achieves high-quality matching results both in terms of matching error and smoothness. Moreover, we show that our method is more scalable than previous formalisms.
>
---
#### [new 053] FloorplanVLM: A Vision-Language Model for Floorplan Vectorization
- **分类: cs.CV**

- **简介: 该论文提出FloorplanVLM，解决将位图户型图转换为工程级矢量图形的问题，通过图像条件序列建模实现精确几何约束满足。**

- **链接: [https://arxiv.org/pdf/2602.06507v1](https://arxiv.org/pdf/2602.06507v1)**

> **作者:** Yuanqing Liu; Ziming Yang; Yulong Li; Yue Yang
>
> **摘要:** Converting raster floorplans into engineering-grade vector graphics is challenging due to complex topology and strict geometric constraints. To address this, we present FloorplanVLM, a unified framework that reformulates floorplan vectorization as an image-conditioned sequence modeling task. Unlike pixel-based methods that rely on fragile heuristics or query-based transformers that generate fragmented rooms, our model directly outputs structured JSON sequences representing the global topology. This 'pixels-to-sequence' paradigm enables the precise and holistic constraint satisfaction of complex geometries, such as slanted walls and curved arcs. To support this data-hungry approach, we introduce a scalable data engine: we construct a large-scale dataset (Floorplan-2M) and a high-fidelity subset (Floorplan-HQ-300K) to balance geometric diversity and pixel-level precision. We then employ a progressive training strategy, using Supervised Fine-Tuning (SFT) for structural grounding and quality annealing, followed by Group Relative Policy Optimization (GRPO) for strict geometric alignment. To standardize evaluation on complex layouts, we establish and open-source FPBench-2K. Evaluated on this rigorous benchmark, FloorplanVLM demonstrates exceptional structural validity, achieving $\textbf{92.52%}$ external-wall IoU and robust generalization across non-Manhattan architectures.
>
---
#### [new 054] M3: High-fidelity Text-to-Image Generation via Multi-Modal, Multi-Agent and Multi-Round Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂组合提示下的生成质量问题。提出M3框架，通过多模态、多代理和多轮推理提升生成质量。**

- **链接: [https://arxiv.org/pdf/2602.06166v1](https://arxiv.org/pdf/2602.06166v1)**

> **作者:** Bangji Yang; Ruihan Guo; Jiajun Fan; Chaoran Cheng; Ge Liu
>
> **摘要:** Generative models have achieved impressive fidelity in text-to-image synthesis, yet struggle with complex compositional prompts involving multiple constraints. We introduce \textbf{M3 (Multi-Modal, Multi-Agent, Multi-Round)}, a training-free framework that systematically resolves these failures through iterative inference-time refinement. M3 orchestrates off-the-shelf foundation models in a robust multi-agent loop: a Planner decomposes prompts into verifiable checklists, while specialized Checker, Refiner, and Editor agents surgically correct constraints one at a time, with a Verifier ensuring monotonic improvement. Applied to open-source models, M3 achieves remarkable results on the challenging OneIG-EN benchmark, with our Qwen-Image+M3 surpassing commercial flagship systems including Imagen4 (0.515) and Seedream 3.0 (0.530), reaching state-of-the-art performance (0.532 overall). This demonstrates that intelligent multi-agent reasoning can elevate open-source models beyond proprietary alternatives. M3 also substantially improves GenEval compositional metrics, effectively doubling spatial reasoning performance on hardened test sets. As a plug-and-play module compatible with any pre-trained T2I model, M3 establishes a new paradigm for compositional generation without costly retraining.
>
---
#### [new 055] CauCLIP: Bridging the Sim-to-Real Gap in Surgical Video Understanding via Causality-Inspired Vision-Language Modeling
- **分类: cs.CV**

- **简介: 该论文属于手术视频理解任务，旨在解决仿真与真实数据间的域差距问题。提出CauCLIP框架，通过因果视觉语言建模和增强策略，提升模型的域泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06619v1](https://arxiv.org/pdf/2602.06619v1)**

> **作者:** Yuxin He; An Li; Cheng Xue
>
> **摘要:** Surgical phase recognition is a critical component for context-aware decision support in intelligent operating rooms, yet training robust models is hindered by limited annotated clinical videos and large domain gaps between synthetic and real surgical data. To address this, we propose CauCLIP, a causality-inspired vision-language framework that leverages CLIP to learn domain-invariant representations for surgical phase recognition without access to target domain data. Our approach integrates a frequency-based augmentation strategy to perturb domain-specific attributes while preserving semantic structures, and a causal suppression loss that mitigates non-causal biases and reinforces causal surgical features. These components are combined in a unified training framework that enables the model to focus on stable causal factors underlying surgical workflows. Experiments on the SurgVisDom hard adaptation benchmark demonstrate that our method substantially outperforms all competing approaches, highlighting the effectiveness of causality-guided vision-language models for domain-generalizable surgical video understanding.
>
---
#### [new 056] Seeing Beyond Redundancy: Task Complexity's Role in Vision Token Specialization in VLLMs
- **分类: cs.CV**

- **简介: 该论文研究视觉大语言模型（VLLMs）在复杂视觉任务中的表现问题，探讨任务复杂度与视觉信息压缩的关系，提出合成数据集和评估指标以分析视觉冗余，旨在提升VLLMs的视觉能力。**

- **链接: [https://arxiv.org/pdf/2602.06914v1](https://arxiv.org/pdf/2602.06914v1)**

> **作者:** Darryl Hannan; John Cooper; Dylan White; Yijing Watkins
>
> **备注:** 25 pages
>
> **摘要:** Vision capabilities in vision large language models (VLLMs) have consistently lagged behind their linguistic capabilities. In particular, numerous benchmark studies have demonstrated that VLLMs struggle when fine-grained visual information or spatial reasoning is required. However, we do not yet understand exactly why VLLMs struggle so much with these tasks relative to others. Some works have focused on visual redundancy as an explanation, where high-level visual information is uniformly spread across numerous tokens and specific, fine-grained visual information is discarded. In this work, we investigate this premise in greater detail, seeking to better understand exactly how various types of visual information are processed by the model and what types of visual information are discarded. To do so, we introduce a simple synthetic benchmark dataset that is specifically constructed to probe various visual features, along with a set of metrics for measuring visual redundancy, allowing us to better understand the nuances of their relationship. Then, we explore fine-tuning VLLMs on a number of complex visual tasks to better understand how redundancy and compression change based upon the complexity of the data that a model is trained on. We find that there is a connection between task complexity and visual compression, implying that having a sufficient ratio of high complexity visual data is crucial for altering the way that VLLMs distribute their visual representation and consequently improving their performance on complex visual tasks. We hope that this work will provide valuable insights for training the next generation of VLLMs.
>
---
#### [new 057] NanoFLUX: Distillation-Driven Compression of Large Text-to-Image Generation Models for Mobile Devices
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决大模型难以在移动设备上运行的问题。通过模型压缩和优化，提出NanoFLUX，在保持质量的同时提升运行效率。**

- **链接: [https://arxiv.org/pdf/2602.06879v1](https://arxiv.org/pdf/2602.06879v1)**

> **作者:** Ruchika Chavhan; Malcolm Chadwick; Alberto Gil Couto Pimentel Ramos; Luca Morreale; Mehdi Noroozi; Abhinav Mehrotra
>
> **摘要:** While large-scale text-to-image diffusion models continue to improve in visual quality, their increasing scale has widened the gap between state-of-the-art models and on-device solutions. To address this gap, we introduce NanoFLUX, a 2.4B text-to-image flow-matching model distilled from 17B FLUX.1-Schnell using a progressive compression pipeline designed to preserve generation quality. Our contributions include: (1) A model compression strategy driven by pruning redundant components in the diffusion transformer, reducing its size from 12B to 2B; (2) A ResNet-based token downsampling mechanism that reduces latency by allowing intermediate blocks to operate on lower-resolution tokens while preserving high-resolution processing elsewhere; (3) A novel text encoder distillation approach that leverages visual signals from early layers of the denoiser during sampling. Empirically, NanoFLUX generates 512 x 512 images in approximately 2.5 seconds on mobile devices, demonstrating the feasibility of high-quality on-device text-to-image generation.
>
---
#### [new 058] A Unified Formula for Affine Transformations between Calibrated Cameras
- **分类: cs.CV**

- **简介: 该论文属于视觉几何任务，解决 calibrated 相机间图像块的仿射变换问题，提出了一种基于相对相机姿态、坐标和表面法线的统一公式。**

- **链接: [https://arxiv.org/pdf/2602.06805v1](https://arxiv.org/pdf/2602.06805v1)**

> **作者:** Levente Hajder
>
> **摘要:** In this technical note, we derive a closed-form expression for the affine transformation mapping local image patches between two calibrated views. We show that the transformation is a function of the relative camera pose, the image coordinates, and the local surface normal.
>
---
#### [new 059] DAVE: Distribution-aware Attribution via ViT Gradient Decomposition
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于视觉Transformer的可解释性任务，旨在解决其 attribution map 生成不稳定的问题，提出DAVE方法通过梯度分解实现更准确的像素级解释。**

- **链接: [https://arxiv.org/pdf/2602.06613v1](https://arxiv.org/pdf/2602.06613v1)**

> **作者:** Adam Wróbel; Siddhartha Gairola; Jacek Tabor; Bernt Schiele; Bartosz Zieliński; Dawid Rymarczyk
>
> **备注:** work under review. Code will be released upon acceptance
>
> **摘要:** Vision Transformers (ViTs) have become a dominant architecture in computer vision, yet producing stable and high-resolution attribution maps for these models remains challenging. Architectural components such as patch embeddings and attention routing often introduce structured artifacts in pixel-level explanations, causing many existing methods to rely on coarse patch-level attributions. We introduce DAVE \textit{(\underline{D}istribution-aware \underline{A}ttribution via \underline{V}iT Gradient D\underline{E}composition)}, a mathematically grounded attribution method for ViTs based on a structured decomposition of the input gradient. By exploiting architectural properties of ViTs, DAVE isolates locally equivariant and stable components of the effective input--output mapping. It separates these from architecture-induced artifacts and other sources of instability.
>
---
#### [new 060] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言模型研究，旨在理解其嵌入空间的几何结构。通过引入跨模态冗余假设，提出SAE方法，揭示了模型中的对齐信号和模态差异来源。**

- **链接: [https://arxiv.org/pdf/2602.06218v1](https://arxiv.org/pdf/2602.06218v1)**

> **作者:** Grégoire Dhimoïla; Thomas Fel; Victor Boutin; Agustin Picard
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Vision-language models (VLMs) align images and text with remarkable success, yet the geometry of their shared embedding space remains poorly understood. To probe this geometry, we begin from the Iso-Energy Assumption, which exploits cross-modal redundancy: a concept that is truly shared should exhibit the same average energy across modalities. We operationalize this assumption with an Aligned Sparse Autoencoder (SAE) that encourages energy consistency during training while preserving reconstruction. We find that this inductive bias changes the SAE solution without harming reconstruction, giving us a representation that serves as a tool for geometric analysis. Sanity checks on controlled data with known ground truth confirm that alignment improves when Iso-Energy holds and remains neutral when it does not. Applied to foundational VLMs, our framework reveals a clear structure with practical consequences: (i) sparse bimodal atoms carry the entire cross-modal alignment signal; (ii) unimodal atoms act as modality-specific biases and fully explain the modality gap; (iii) removing unimodal atoms collapses the gap without harming performance; (iv) restricting vector arithmetic to the bimodal subspace yields in-distribution edits and improved retrieval. These findings suggest that the right inductive bias can both preserve model fidelity and render the latent geometry interpretable and actionable.
>
---
#### [new 061] RAIGen: Rare Attribute Identification in Text-to-Image Generative Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出RAIGen，用于发现扩散模型中罕见属性的无监督框架，解决模型偏见问题，通过激活分析识别未被充分表示的语义特征。**

- **链接: [https://arxiv.org/pdf/2602.06806v1](https://arxiv.org/pdf/2602.06806v1)**

> **作者:** Silpa Vadakkeeveetil Sreelatha; Dan Wang; Serge Belongie; Muhammad Awais; Anjan Dutta
>
> **摘要:** Text-to-image diffusion models achieve impressive generation quality but inherit and amplify training-data biases, skewing coverage of semantic attributes. Prior work addresses this in two ways. Closed-set approaches mitigate biases in predefined fairness categories (e.g., gender, race), assuming socially salient minority attributes are known a priori. Open-set approaches frame the task as bias identification, highlighting majority attributes that dominate outputs. Both overlook a complementary task: uncovering rare or minority features underrepresented in the data distribution (social, cultural, or stylistic) yet still encoded in model representations. We introduce RAIGen, the first framework, to our knowledge, for un-supervised rare-attribute discovery in diffusion models. RAIGen leverages Matryoshka Sparse Autoencoders and a novel minority metric combining neuron activation frequency with semantic distinctiveness to identify interpretable neurons whose top-activating images reveal underrepresented attributes. Experiments show RAIGen discovers attributes beyond fixed fairness categories in Stable Diffusion, scales to larger models such as SDXL, supports systematic auditing across architectures, and enables targeted amplification of rare attributes during generation.
>
---
#### [new 062] DreamHome-Pano: Design-Aware and Conflict-Free Panoramic Interior Generation
- **分类: cs.CV**

- **简介: 该论文属于室内设计生成任务，解决布局与风格冲突问题。提出DreamHome-Pano框架，通过语义桥梁和结构感知机制，实现风格与布局的和谐统一。**

- **链接: [https://arxiv.org/pdf/2602.06494v1](https://arxiv.org/pdf/2602.06494v1)**

> **作者:** Lulu Chen; Yijiang Hu; Yuanqing Liu; Yulong Li; Yue Yang
>
> **摘要:** In modern interior design, the generation of personalized spaces frequently necessitates a delicate balance between rigid architectural structural constraints and specific stylistic preferences. However, existing multi-condition generative frameworks often struggle to harmonize these inputs, leading to "condition conflicts" where stylistic attributes inadvertently compromise the geometric precision of the layout. To address this challenge, we present DreamHome-Pano, a controllable panoramic generation framework designed for high-fidelity interior synthesis. Our approach introduces a Prompt-LLM that serves as a semantic bridge, effectively translating layout constraints and style references into professional descriptive prompts to achieve precise cross-modal alignment. To safeguard architectural integrity during the generative process, we develop a Conflict-Free Control architecture that incorporates structural-aware geometric priors and a multi-condition decoupling strategy, effectively suppressing stylistic interference from eroding the spatial layout. Furthermore, we establish a comprehensive panoramic interior benchmark alongside a multi-stage training pipeline, encompassing progressive Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). Experimental results demonstrate that DreamHome-Pano achieves a superior balance between aesthetic quality and structural consistency, offering a robust and professional-grade solution for panoramic interior visualization.
>
---
#### [new 063] AdaptOVCD: Training-Free Open-Vocabulary Remote Sensing Change Detection via Adaptive Information Fusion
- **分类: cs.CV**

- **简介: 该论文属于遥感变化检测任务，解决传统方法依赖预定义类别和大量标注的问题。提出AdaptOVCD框架，通过多层级信息融合实现无训练的开放词汇变化检测。**

- **链接: [https://arxiv.org/pdf/2602.06529v1](https://arxiv.org/pdf/2602.06529v1)**

> **作者:** Mingyu Dou; Shi Qiu; Ming Hu; Yifan Chen; Huping Ye; Xiaohan Liao; Zhe Sun
>
> **摘要:** Remote sensing change detection plays a pivotal role in domains such as environmental monitoring, urban planning, and disaster assessment. However, existing methods typically rely on predefined categories and large-scale pixel-level annotations, which limit their generalization and applicability in open-world scenarios. To address these limitations, this paper proposes AdaptOVCD, a training-free Open-Vocabulary Change Detection (OVCD) architecture based on dual-dimensional multi-level information fusion. The framework integrates multi-level information fusion across data, feature, and decision levels vertically while incorporating targeted adaptive designs horizontally, achieving deep synergy among heterogeneous pre-trained models to effectively mitigate error propagation. Specifically, (1) at the data level, Adaptive Radiometric Alignment (ARA) fuses radiometric statistics with original texture features and synergizes with SAM-HQ to achieve radiometrically consistent segmentation; (2) at the feature level, Adaptive Change Thresholding (ACT) combines global difference distributions with edge structure priors and leverages DINOv3 to achieve robust change detection; (3) at the decision level, Adaptive Confidence Filtering (ACF) integrates semantic confidence with spatial constraints and collaborates with DGTRS-CLIP to achieve high-confidence semantic identification. Comprehensive evaluations across nine scenarios demonstrate that AdaptOVCD detects arbitrary category changes in a zero-shot manner, significantly outperforming existing training-free methods. Meanwhile, it achieves 84.89\% of the fully-supervised performance upper bound in cross-dataset evaluations and exhibits superior generalization capabilities. The code is available at https://github.com/Dmygithub/AdaptOVCD.
>
---
#### [new 064] MetaSSP: Enhancing Semi-supervised Implicit 3D Reconstruction through Meta-adaptive EMA and SDF-aware Pseudo-label Evaluation
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决单视角下依赖大量标注数据的问题。提出MetaSSP框架，利用未标注图像和伪标签优化重建效果。**

- **链接: [https://arxiv.org/pdf/2602.06163v1](https://arxiv.org/pdf/2602.06163v1)**

> **作者:** Luoxi Zhang; Chun Xie; Itaru Kitahara
>
> **摘要:** Implicit SDF-based methods for single-view 3D reconstruction achieve high-quality surfaces but require large labeled datasets, limiting their scalability. We propose MetaSSP, a novel semi-supervised framework that exploits abundant unlabeled images. Our approach introduces gradient-based parameter importance estimation to regularize adaptive EMA updates and an SDF-aware pseudo-label weighting mechanism combining augmentation consistency with SDF variance. Beginning with a 10% supervised warm-up, the unified pipeline jointly refines labeled and unlabeled data. On the Pix3D benchmark, our method reduces Chamfer Distance by approximately 20.61% and increases IoU by around 24.09% compared to existing semi-supervised baselines, setting a new state of the art.
>
---
#### [new 065] Point Virtual Transformer
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决远距离点云稀疏导致的检测难题。通过引入虚拟点并结合Transformer框架，提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.06406v1](https://arxiv.org/pdf/2602.06406v1)**

> **作者:** Veerain Sood; Bnalin; Gaurav Pandey
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** LiDAR-based 3D object detectors often struggle to detect far-field objects due to the sparsity of point clouds at long ranges, which limits the availability of reliable geometric cues. To address this, prior approaches augment LiDAR data with depth-completed virtual points derived from RGB images; however, directly incorporating all virtual points leads to increased computational cost and introduces challenges in effectively fusing real and virtual information. We present Point Virtual Transformer (PointViT), a transformer-based 3D object detection framework that jointly reasons over raw LiDAR points and selectively sampled virtual points. The framework examines multiple fusion strategies, ranging from early point-level fusion to BEV-based gated fusion, and analyses their trade-offs in terms of accuracy and efficiency. The fused point cloud is voxelized and encoded using sparse convolutions to form a BEV representation, from which a compact set of high-confidence object queries is initialised and refined through a transformer-based context aggregation module. Experiments on the KITTI benchmark report 91.16% 3D AP, 95.94% BEV AP, and 99.36% AP on the KITTI 2D detection benchmark for the Car class.
>
---
#### [new 066] ForeHOI: Feed-forward 3D Object Reconstruction from Daily Hand-Object Interaction Videos
- **分类: cs.CV**

- **简介: 该论文提出ForeHOI，解决单目手-物体交互视频中的3D物体重建问题，通过联合2D掩码修复与3D形状补全，提升重建精度与速度。**

- **链接: [https://arxiv.org/pdf/2602.06226v1](https://arxiv.org/pdf/2602.06226v1)**

> **作者:** Yuantao Chen; Jiahao Chang; Chongjie Ye; Chaoran Zhang; Zhaojie Fang; Chenghong Li; Xiaoguang Han
>
> **备注:** 14 pages, 7 figures, Page: https://tao-11-chen.github.io/project_pages/ForeHOI/
>
> **摘要:** The ubiquity of monocular videos capturing daily hand-object interactions presents a valuable resource for embodied intelligence. While 3D hand reconstruction from in-the-wild videos has seen significant progress, reconstructing the involved objects remains challenging due to severe occlusions and the complex, coupled motion of the camera, hands, and object. In this paper, we introduce ForeHOI, a novel feed-forward model that directly reconstructs 3D object geometry from monocular hand-object interaction videos within one minute of inference time, eliminating the need for any pre-processing steps. Our key insight is that, the joint prediction of 2D mask inpainting and 3D shape completion in a feed-forward framework can effectively address the problem of severe occlusion in monocular hand-held object videos, thereby achieving results that outperform the performance of optimization-based methods. The information exchanges between the 2D and 3D shape completion boosts the overall reconstruction quality, enabling the framework to effectively handle severe hand-object occlusion. Furthermore, to support the training of our model, we contribute the first large-scale, high-fidelity synthetic dataset of hand-object interactions with comprehensive annotations. Extensive experiments demonstrate that ForeHOI achieves state-of-the-art performance in object reconstruction, significantly outperforming previous methods with around a 100x speedup. Code and data are available at: https://github.com/Tao-11-chen/ForeHOI.
>
---
#### [new 067] Can We Build a Monolithic Model for Fake Image Detection? SICA: Semantic-Induced Constrained Adaptation for Unified-Yet-Discriminative Artifact Feature Space Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于虚假图像检测任务，旨在解决单模型在多个子领域中性能不佳的问题。通过提出SICA方法，重建统一且区分的特征空间，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.06676v1](https://arxiv.org/pdf/2602.06676v1)**

> **作者:** Bo Du; Xiaochen Ma; Xuekang Zhu; Zhe Yang; Chaogun Niu; Jian Liu; Ji-Zhe Zhou
>
> **摘要:** Fake Image Detection (FID), aiming at unified detection across four image forensic subdomains, is critical in real-world forensic scenarios. Compared with ensemble approaches, monolithic FID models are theoretically more promising, but to date, consistently yield inferior performance in practice. In this work, by discovering the ``heterogeneous phenomenon'', which is the intrinsic distinctness of artifacts across subdomains, we diagnose the cause of this underperformance for the first time: the collapse of the artifact feature space driven by such phenomenon. The core challenge for developing a practical monolithic FID model thus boils down to the ``unified-yet-discriminative" reconstruction of the artifact feature space. To address this paradoxical challenge, we hypothesize that high-level semantics can serve as a structural prior for the reconstruction, and further propose Semantic-Induced Constrained Adaptation (SICA), the first monolithic FID paradigm. Extensive experiments on our OpenMMSec dataset demonstrate that SICA outperforms 15 state-of-the-art methods and reconstructs the target unified-yet-discriminative artifact feature space in a near-orthogonal manner, thus firmly validating our hypothesis. The code and dataset are available at:https: //github.com/scu-zjz/SICA_OpenMMSec.
>
---
#### [new 068] PhenoLIP: Integrating Phenotype Ontology Knowledge into Medical Vision-Language Pretraining
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医疗视觉-语言预训练任务，旨在解决现有模型未能有效利用医学表型本体知识的问题。通过构建PhenoKG和提出PhenoLIP框架，提升医学图像理解的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.06184v1](https://arxiv.org/pdf/2602.06184v1)**

> **作者:** Cheng Liang; Chaoyi Wu; Weike Zhao; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **摘要:** Recent progress in large-scale CLIP-like vision-language models(VLMs) has greatly advanced medical image analysis. However, most existing medical VLMs still rely on coarse image-text contrastive objectives and fail to capture the systematic visual knowledge encoded in well-defined medical phenotype ontologies. To address this gap, we construct PhenoKG, the first large-scale, phenotype-centric multimodal knowledge graph that encompasses over 520K high-quality image-text pairs linked to more than 3,000 phenotypes. Building upon PhenoKG, we propose PhenoLIP, a novel pretraining framework that explicitly incorporates structured phenotype knowledge into medical VLMs through a two-stage process. We first learn a knowledge-enhanced phenotype embedding space from textual ontology data and then distill this structured knowledge into multimodal pretraining via a teacher-guided knowledge distillation objective. To support evaluation, we further introduce PhenoBench, an expert-verified benchmark designed for phenotype recognition, comprising over 7,800 image--caption pairs covering more than 1,000 phenotypes. Extensive experiments demonstrate that PhenoLIP outperforms previous state-of-the-art baselines, improving upon BiomedCLIP in phenotype classification accuracy by 8.85\% and BIOMEDICA in cross-modal retrieval by 15.03%, underscoring the value of integrating phenotype-centric priors into medical VLMs for structured and interpretable medical image understanding.
>
---
#### [new 069] Revisiting Salient Object Detection from an Observer-Centric Perspective
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分割任务，旨在解决传统方法忽略观察者主观差异的问题。通过引入观者中心视角，提出OC-SOD框架及相应数据集，提升个性化与上下文感知的显著性检测效果。**

- **链接: [https://arxiv.org/pdf/2602.06369v1](https://arxiv.org/pdf/2602.06369v1)**

> **作者:** Fuxi Zhang; Yifan Wang; Hengrun Zhao; Zhuohan Sun; Changxing Xia; Lijun Wang; Huchuan Lu; Yangrui Shao; Chen Yang; Long Teng
>
> **摘要:** Salient object detection is inherently a subjective problem, as observers with different priors may perceive different objects as salient. However, existing methods predominantly formulate it as an objective prediction task with a single groundtruth segmentation map for each image, which renders the problem under-determined and fundamentally ill-posed. To address this issue, we propose Observer-Centric Salient Object Detection (OC-SOD), where salient regions are predicted by considering not only the visual cues but also the observer-specific factors such as their preferences or intents. As a result, this formulation captures the intrinsic ambiguity and diversity of human perception, enabling personalized and context-aware saliency prediction. By leveraging multi-modal large language models, we develop an efficient data annotation pipeline and construct the first OC-SOD dataset named OC-SODBench, comprising 33k training, validation and test images with 152k textual prompts and object pairs. Built upon this new dataset, we further design OC-SODAgent, an agentic baseline which performs OC-SOD via a human-like "Perceive-Reflect-Adjust" process. Extensive experiments on our proposed OC-SODBench have justified the effectiveness of our contribution. Through this observer-centric perspective, we aim to bridge the gap between human perception and computational modeling, offering a more realistic and flexible understanding of what makes an object truly "salient." Code and dataset are publicly available at: https://github.com/Dustzx/OC_SOD
>
---
#### [new 070] Machine Learning for Detection and Severity Estimation of Sweetpotato Weevil Damage in Field and Lab Conditions
- **分类: cs.CV**

- **简介: 该论文属于农业病虫害检测任务，旨在解决传统人工评估甜瓜象甲损伤效率低、主观性强的问题。通过计算机视觉技术实现田间和实验室中损伤的自动检测与严重程度估计。**

- **链接: [https://arxiv.org/pdf/2602.06786v1](https://arxiv.org/pdf/2602.06786v1)**

> **作者:** Doreen M. Chelangat; Sudi Murindanyi; Bruce Mugizi; Paul Musana; Benard Yada; Milton A. Otema; Florence Osaru; Andrew Katumba; Joyce Nakatumba-Nabende
>
> **摘要:** Sweetpotato weevils (Cylas spp.) are considered among the most destructive pests impacting sweetpotato production, particularly in sub-Saharan Africa. Traditional methods for assessing weevil damage, predominantly relying on manual scoring, are labour-intensive, subjective, and often yield inconsistent results. These challenges significantly hinder breeding programs aimed at developing resilient sweetpotato varieties. This study introduces a computer vision-based approach for the automated evaluation of weevil damage in both field and laboratory contexts. In the field settings, we collected data to train classification models to predict root-damage severity levels, achieving a test accuracy of 71.43%. Additionally, we established a laboratory dataset and designed an object detection pipeline employing YOLO12, a leading real-time detection model. This methodology incorporated a two-stage laboratory pipeline that combined root segmentation with a tiling strategy to improve the detectability of small objects. The resulting model demonstrated a mean average precision of 77.7% in identifying minute weevil feeding holes. Our findings indicate that computer vision technologies can provide efficient, objective, and scalable assessment tools that align seamlessly with contemporary breeding workflows. These advancements represent a significant improvement in enhancing phenotyping efficiency within sweetpotato breeding programs and play a crucial role in mitigating the detrimental effects of weevils on food security.
>
---
#### [new 071] TFusionOcc: Student's t-Distribution Based Object-Centric Multi-Sensor Fusion Framework for 3D Occupancy Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，旨在解决多传感器融合中几何细节捕捉不足的问题。提出TFusionOcc框架，利用t分布和可变形超二次曲面提升预测精度。**

- **链接: [https://arxiv.org/pdf/2602.06400v1](https://arxiv.org/pdf/2602.06400v1)**

> **作者:** Zhenxing Ming; Julie Stephany Berrio; Mao Shan; Stewart Worrall
>
> **摘要:** 3D semantic occupancy prediction enables autonomous vehicles (AVs) to perceive fine-grained geometric and semantic structure of their surroundings from onboard sensors, which is essential for safe decision-making and navigation. Recent models for 3D semantic occupancy prediction have successfully addressed the challenge of describing real-world objects with varied shapes and classes. However, the intermediate representations used by existing methods for 3D semantic occupancy prediction rely heavily on 3D voxel volumes or a set of 3D Gaussians, hindering the model's ability to efficiently and effectively capture fine-grained geometric details in the 3D driving environment. This paper introduces TFusionOcc, a novel object-centric multi-sensor fusion framework for predicting 3D semantic occupancy. By leveraging multi-stage multi-sensor fusion, Student's t-distribution, and the T-Mixture model (TMM), together with more geometrically flexible primitives, such as the deformable superquadric (superquadric with inverse warp), the proposed method achieved state-of-the-art (SOTA) performance on the nuScenes benchmark. In addition, extensive experiments were conducted on the nuScenes-C dataset to demonstrate the robustness of the proposed method in different camera and lidar corruption scenarios. The code will be available at: https://github.com/DanielMing123/TFusionOcc
>
---
#### [new 072] Robust Pedestrian Detection with Uncertain Modality
- **分类: cs.CV**

- **简介: 该论文属于跨模态行人检测任务，解决多模态数据缺失导致的检测性能下降问题。构建了TRNT数据集，并提出AUNet模型以适应不确定输入，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06363v1](https://arxiv.org/pdf/2602.06363v1)**

> **作者:** Qian Bie; Xiao Wang; Bin Yang; Zhixi Yu; Jun Chen; Xin Xu
>
> **备注:** Due to the limitation "The abstract field cannot be longer than 1,920 characters", the abstract here is shorter than that in the PDF file
>
> **摘要:** Existing cross-modal pedestrian detection (CMPD) employs complementary information from RGB and thermal-infrared (TIR) modalities to detect pedestrians in 24h-surveillance systems.RGB captures rich pedestrian details under daylight, while TIR excels at night. However, TIR focuses primarily on the person's silhouette, neglecting critical texture details essential for detection. While the near-infrared (NIR) captures texture under low-light conditions, which effectively alleviates performance issues of RGB and detail loss in TIR, thereby reducing missed detections. To this end, we construct a new Triplet RGB-NIR-TIR (TRNT) dataset, comprising 8,281 pixel-aligned image triplets, establishing a comprehensive foundation for algorithmic research. However, due to the variable nature of real-world scenarios, imaging devices may not always capture all three modalities simultaneously. This results in input data with unpredictable combinations of modal types, which challenge existing CMPD methods that fail to extract robust pedestrian information under arbitrary input combinations, leading to significant performance degradation. To address these challenges, we propose the Adaptive Uncertainty-aware Network (AUNet) for accurately discriminating modal availability and fully utilizing the available information under uncertain inputs. Specifically, we introduce Unified Modality Validation Refinement (UMVR), which includes an uncertainty-aware router to validate modal availability and a semantic refinement to ensure the reliability of information within the modality. Furthermore, we design a Modality-Aware Interaction (MAI) module to adaptively activate or deactivate its internal interaction mechanisms per UMVR output, enabling effective complementary information fusion from available modalities.
>
---
#### [new 073] Forest canopy height estimation from satellite RGB imagery using large-scale airborne LiDAR-derived training data and monocular depth estimation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于森林冠层高度估计任务，旨在利用卫星RGB影像和机载LiDAR数据训练模型，解决高分辨率、连续冠层高度映射的问题。**

- **链接: [https://arxiv.org/pdf/2602.06503v1](https://arxiv.org/pdf/2602.06503v1)**

> **作者:** Yongkang Lai; Xihan Mu; Tim R. McVicar; Dasheng Fan; Donghui Xie; Shanxin Guo; Wenli Huang; Tianjie Zhao; Guangjian Yan
>
> **摘要:** Large-scale, high-resolution forest canopy height mapping plays a crucial role in understanding regional and global carbon and water cycles. Spaceborne LiDAR missions, including the Ice, Cloud, and Land Elevation Satellite-2 (ICESat-2) and the Global Ecosystem Dynamics Investigation (GEDI), provide global observations of forest structure but are spatially sparse and subject to inherent uncertainties. In contrast, near-surface LiDAR platforms, such as airborne and unmanned aerial vehicle (UAV) LiDAR systems, offer much finer measurements of forest canopy structure, and a growing number of countries have made these datasets openly available. In this study, a state-of-the-art monocular depth estimation model, Depth Anything V2, was trained using approximately 16,000 km2 of canopy height models (CHMs) derived from publicly available airborne LiDAR point clouds and related products across multiple countries, together with 3 m resolution PlanetScope and airborne RGB imagery. The trained model, referred to as Depth2CHM, enables the estimation of spatially continuous CHMs directly from PlanetScope RGB imagery. Independent validation was conducted at sites in China (approximately 1 km2) and the United States (approximately 116 km2). The results showed that Depth2CHM could accurately estimate canopy height, with biases of 0.59 m and 0.41 m and root mean square errors (RMSEs) of 2.54 m and 5.75 m for these two sites, respectively. Compared with an existing global meter-resolution CHM product, the mean absolute error is reduced by approximately 1.5 m and the RMSE by approximately 2 m. These results demonstrated that monocular depth estimation networks trained with large-scale airborne LiDAR-derived canopy height data provide a promising and scalable pathway for high-resolution, spatially continuous forest canopy height estimation from satellite RGB imagery.
>
---
#### [new 074] Driving with DINO: Vision Foundation Features as a Unified Bridge for Sim-to-Real Generation in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶视频生成任务，解决sim-to-real域差距问题。通过DINO特征构建统一桥梁，提升生成视频的现实感与控制一致性。**

- **链接: [https://arxiv.org/pdf/2602.06159v1](https://arxiv.org/pdf/2602.06159v1)**

> **作者:** Xuyang Chen; Conglang Zhang; Chuanheng Fu; Zihao Yang; Kaixuan Zhou; Yizhi Zhang; Jianan He; Yanfeng Zhang; Mingwei Sun; Zengmao Wang; Zhen Dong; Xiaoxiao Long; Liqiu Meng
>
> **备注:** Project website https://albertchen98.github.io/DwD-project/
>
> **摘要:** Driven by the emergence of Controllable Video Diffusion, existing Sim2Real methods for autonomous driving video generation typically rely on explicit intermediate representations to bridge the domain gap. However, these modalities face a fundamental Consistency-Realism Dilemma. Low-level signals (e.g., edges, blurred images) ensure precise control but compromise realism by "baking in" synthetic artifacts, whereas high-level priors (e.g., depth, semantics, HDMaps) facilitate photorealism but lack the structural detail required for consistent guidance. In this work, we present Driving with DINO (DwD), a novel framework that leverages Vision Foundation Module (VFM) features as a unified bridge between the simulation and real-world domains. We first identify that these features encode a spectrum of information, from high-level semantics to fine-grained structure. To effectively utilize this, we employ Principal Subspace Projection to discard the high-frequency elements responsible for "texture baking," while concurrently introducing Random Channel Tail Drop to mitigate the structural loss inherent in rigid dimensionality reduction, thereby reconciling realism with control consistency. Furthermore, to fully leverage DINOv3's high-resolution capabilities for enhancing control precision, we introduce a learnable Spatial Alignment Module that adapts these high-resolution features to the diffusion backbone. Finally, we propose a Causal Temporal Aggregator employing causal convolutions to explicitly preserve historical motion context when integrating frame-wise DINO features, which effectively mitigates motion blur and guarantees temporal stability. Project page: https://albertchen98.github.io/DwD-project/
>
---
#### [new 075] Reliable Mislabel Detection for Video Capsule Endoscopy Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医疗图像分类任务，旨在解决医学数据标注错误的问题。通过提出一种误标检测框架，提升数据准确性，从而改善异常检测性能。**

- **链接: [https://arxiv.org/pdf/2602.06938v1](https://arxiv.org/pdf/2602.06938v1)**

> **作者:** Julia Werner; Julius Oexle; Oliver Bause; Maxime Le Floch; Franz Brinkmann; Hannah Tolle; Jochen Hampe; Oliver Bringmann
>
> **摘要:** The classification performance of deep neural networks relies strongly on access to large, accurately annotated datasets. In medical imaging, however, obtaining such datasets is particularly challenging since annotations must be provided by specialized physicians, which severely limits the pool of annotators. Furthermore, class boundaries can often be ambiguous or difficult to define which further complicates machine learning-based classification. In this paper, we want to address this problem and introduce a framework for mislabel detection in medical datasets. This is validated on the two largest, publicly available datasets for Video Capsule Endoscopy, an important imaging procedure for examining the gastrointestinal tract based on a video stream of lowresolution images. In addition, potentially mislabeled samples identified by our pipeline were reviewed and re-annotated by three experienced gastroenterologists. Our results show that the proposed framework successfully detects incorrectly labeled data and results in an improved anomaly detection performance after cleaning the datasets compared to current baselines.
>
---
#### [new 076] Gold Exploration using Representations from a Multispectral Autoencoder
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于矿产勘探任务，旨在通过卫星影像识别金矿区域。利用多光谱自动编码器生成特征表示，提升勘探效率与准确性。**

- **链接: [https://arxiv.org/pdf/2602.06748v1](https://arxiv.org/pdf/2602.06748v1)**

> **作者:** Argyro Tsandalidou; Konstantinos Dogeas; Eleftheria Tetoula Tsonga; Elisavet Parselia; Georgios Tsimiklis; George Arvanitakis
>
> **备注:** Presented in Eurips2025, 1st Workshop: Advances in Representation Learning for Earth Observation
>
> **摘要:** Satellite imagery is employed for large-scale prospectivity mapping due to the high cost and typically limited availability of on-site mineral exploration data. In this work, we present a proof-of-concept framework that leverages generative representations learned from multispectral Sentinel-2 imagery to identify gold-bearing regions from space. An autoencoder foundation model, called Isometric, which is pretrained on the large-scale FalconSpace-S2 v1.0 dataset, produces information-dense spectral-spatial representations that serve as inputs to a lightweight XGBoost classifier. We compare this representation-based approach with a raw spectral input baseline using a dataset of 63 Sentinel-2 images from known gold and non-gold locations. The proposed method improves patch-level accuracy from 0.51 to 0.68 and image-level accuracy from 0.55 to 0.73, demonstrating that generative embeddings capture transferable mineralogical patterns even with limited labeled data. These results highlight the potential of foundation-model representations to make mineral exploration more efficient, scalable, and globally applicable.
>
---
#### [new 077] PlanViz: Evaluating Planning-Oriented Image Generation and Editing for Computer-Use Tasks
- **分类: cs.CV**

- **简介: 该论文提出PlanViz，用于评估统一多模态模型在计算机使用任务中的图像生成与编辑能力，解决其在空间推理和流程理解方面的不足。**

- **链接: [https://arxiv.org/pdf/2602.06663v1](https://arxiv.org/pdf/2602.06663v1)**

> **作者:** Junxian Li; Kai Liu; Leyang Chen; Weida Wang; Zhixin Wang; Jiaqi Xu; Fan Li; Renjing Pei; Linghe Kong; Yulun Zhang
>
> **备注:** The main part of our paper: PlanViz Code is at: https://github.com/lijunxian111/PlanViz Supplementary material is at: https://github.com/lijunxian111/PlanViz/releases/tag/v1
>
> **摘要:** Unified multimodal models (UMMs) have shown impressive capabilities in generating natural images and supporting multimodal reasoning. However, their potential in supporting computer-use planning tasks, which are closely related to our lives, remain underexplored. Image generation and editing in computer-use tasks require capabilities like spatial reasoning and procedural understanding, and it is still unknown whether UMMs have these capabilities to finish these tasks or not. Therefore, we propose PlanViz, a new benchmark designed to evaluate image generation and editing for computer-use tasks. To achieve the goal of our evaluation, we focus on sub-tasks which frequently involve in daily life and require planning steps. Specifically, three new sub-tasks are designed: route planning, work diagramming, and web&UI displaying. We address challenges in data quality ensuring by curating human-annotated questions and reference images, and a quality control process. For challenges of comprehensive and exact evaluation, a task-adaptive score, PlanScore, is proposed. The score helps understanding the correctness, visual quality and efficiency of generated images. Through experiments, we highlight key limitations and opportunities for future research on this topic.
>
---
#### [new 078] DroneKey++: A Size Prior-free Method and New Benchmark for Drone 3D Pose Estimation from Sequential Images
- **分类: cs.CV**

- **简介: 该论文属于无人机3D姿态估计任务，解决现有方法依赖先验信息及数据集不足的问题。提出DroneKey++框架，实现无需先验信息的准确姿态估计，并构建大规模合成数据集6DroneSyn。**

- **链接: [https://arxiv.org/pdf/2602.06211v1](https://arxiv.org/pdf/2602.06211v1)**

> **作者:** Seo-Bin Hwang; Yeong-Jun Cho
>
> **备注:** 8 page, 5 figures, 6 tables, Accepted to ICRA 2026 (to appear)
>
> **摘要:** Accurate 3D pose estimation of drones is essential for security and surveillance systems. However, existing methods often rely on prior drone information such as physical sizes or 3D meshes. At the same time, current datasets are small-scale, limited to single models, and collected under constrained environments, which makes reliable validation of generalization difficult. We present DroneKey++, a prior-free framework that jointly performs keypoint detection, drone classification, and 3D pose estimation. The framework employs a keypoint encoder for simultaneous keypoint detection and classification, and a pose decoder that estimates 3D pose using ray-based geometric reasoning and class embeddings. To address dataset limitations, we construct 6DroneSyn, a large-scale synthetic benchmark with over 50K images covering 7 drone models and 88 outdoor backgrounds, generated using 360-degree panoramic synthesis. Experiments show that DroneKey++ achieves MAE 17.34 deg and MedAE 17.1 deg for rotation, MAE 0.135 m and MedAE 0.242 m for translation, with inference speeds of 19.25 FPS (CPU) and 414.07 FPS (GPU), demonstrating both strong generalization across drone models and suitability for real-time applications. The dataset is publicly available.
>
---
#### [new 079] Tempora: Characterising the Time-Contingent Utility of Online Test-Time Adaptation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于测试时自适应（TTA）任务，解决传统评估忽略时间约束的问题。提出Tempora框架，评估不同时间压力下的模型性能，揭示排名变化原因。**

- **链接: [https://arxiv.org/pdf/2602.06136v1](https://arxiv.org/pdf/2602.06136v1)**

> **作者:** Sudarshan Sreeram; Young D. Kwon; Cecilia Mascolo
>
> **备注:** Preprint. Under review. Code available upon acceptance
>
> **摘要:** Test-time adaptation (TTA) offers a compelling remedy for machine learning (ML) models that degrade under domain shifts, improving generalisation on-the-fly with only unlabelled samples. This flexibility suits real deployments, yet conventional evaluations unrealistically assume unbounded processing time, overlooking the accuracy-latency trade-off. As ML increasingly underpins latency-sensitive and user-facing use-cases, temporal pressure constrains the viability of adaptable inference; predictions arriving too late to act on are futile. We introduce Tempora, a framework for evaluating TTA under this pressure. It consists of temporal scenarios that model deployment constraints, evaluation protocols that operationalise measurement, and time-contingent utility metrics that quantify the accuracy-latency trade-off. We instantiate the framework with three such metrics: (1) discrete utility for asynchronous streams with hard deadlines, (2) continuous utility for interactive settings where value decays with latency, and (3) amortised utility for budget-constrained deployments. Applying Tempora to seven TTA methods on ImageNet-C across 240 temporal evaluations reveals rank instability: conventional rankings do not predict rankings under temporal pressure; ETA, a state-of-the-art method in the conventional setting, falls short in 41.2% of evaluations. The highest-utility method varies with corruption type and temporal pressure, with no clear winner. By enabling systematic evaluation across diverse temporal constraints for the first time, Tempora reveals when and why rankings invert, offering practitioners a lens for method selection and researchers a target for deployable adaptation.
>
---
#### [new 080] AEGPO: Adaptive Entropy-Guided Policy Optimization for Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于强化学习任务，旨在解决扩散模型中策略优化效率低的问题。通过分析注意力熵，提出AEGPO方法，动态分配资源并聚焦关键步骤，提升优化效果。**

- **链接: [https://arxiv.org/pdf/2602.06825v1](https://arxiv.org/pdf/2602.06825v1)**

> **作者:** Yuming Li; Qingyu Li; Chengyu Bai; Xiangyang Luo; Zeyue Xue; Wenyu Qin; Meng Wang; Yikai Wang; Shanghang Zhang
>
> **摘要:** Reinforcement learning from human feedback (RLHF) shows promise for aligning diffusion and flow models, yet policy optimization methods such as GRPO suffer from inefficient and static sampling strategies. These methods treat all prompts and denoising steps uniformly, ignoring substantial variations in sample learning value as well as the dynamic nature of critical exploration moments. To address this issue, we conduct a detailed analysis of the internal attention dynamics during GRPO training and uncover a key insight: attention entropy can serve as a powerful dual-signal proxy. First, across different samples, the relative change in attention entropy (ΔEntropy), which reflects the divergence between the current policy and the base policy, acts as a robust indicator of sample learning value. Second, during the denoising process, the peaks of absolute attention entropy (Entropy(t)), which quantify attention dispersion, effectively identify critical timesteps where high-value exploration occurs. Building on this observation, we propose Adaptive Entropy-Guided Policy Optimization (AEGPO), a novel dual-signal, dual-level adaptive optimization strategy. At the global level, AEGPO uses ΔEntropy to dynamically allocate rollout budgets, prioritizing prompts with higher learning value. At the local level, it exploits the peaks of Entropy(t) to guide exploration selectively at critical high-dispersion timesteps rather than uniformly across all denoising steps. By focusing computation on the most informative samples and the most critical moments, AEGPO enables more efficient and effective policy optimization. Experiments on text-to-image generation tasks demonstrate that AEGPO significantly accelerates convergence and achieves superior alignment performance compared to standard GRPO variants.
>
---
#### [new 081] Diffeomorphism-Equivariant Neural Networks
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究如何将微分同胚等变性引入神经网络，解决传统方法在处理无限维群时的局限。通过能量优化实现等变性，提升模型对未知变换的泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06695v1](https://arxiv.org/pdf/2602.06695v1)**

> **作者:** Josephine Elisabeth Oettinger; Zakhar Shumaylov; Johannes Bostelmann; Jan Lellmann; Carola-Bibiane Schönlieb
>
> **摘要:** Incorporating group symmetries via equivariance into neural networks has emerged as a robust approach for overcoming the efficiency and data demands of modern deep learning. While most existing approaches, such as group convolutions and averaging-based methods, focus on compact, finite, or low-dimensional groups with linear actions, this work explores how equivariance can be extended to infinite-dimensional groups. We propose a strategy designed to induce diffeomorphism equivariance in pre-trained neural networks via energy-based canonicalisation. Formulating equivariance as an optimisation problem allows us to access the rich toolbox of already established differentiable image registration methods. Empirical results on segmentation and classification tasks confirm that our approach achieves approximate equivariance and generalises to unseen transformations without relying on extensive data augmentation or retraining.
>
---
#### [new 082] MultiGraspNet: A Multitask 3D Vision Model for Multi-gripper Robotic Grasping
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MultiGraspNet，解决多夹爪机器人抓取问题。通过统一框架同时预测平行与吸力夹爪的抓取位姿，提升抓取效率与适应性。**

- **链接: [https://arxiv.org/pdf/2602.06504v1](https://arxiv.org/pdf/2602.06504v1)**

> **作者:** Stephany Ortuno-Chanelo; Paolo Rabino; Enrico Civitelli; Tatiana Tommasi; Raffaello Camoriano
>
> **摘要:** Vision-based models for robotic grasping automate critical, repetitive, and draining industrial tasks. Existing approaches are typically limited in two ways: they either target a single gripper and are potentially applied on costly dual-arm setups, or rely on custom hybrid grippers that require ad-hoc learning procedures with logic that cannot be transferred across tasks, restricting their general applicability. In this work, we present MultiGraspNet, a novel multitask 3D deep learning method that predicts feasible poses simultaneously for parallel and vacuum grippers within a unified framework, enabling a single robot to handle multiple end effectors. The model is trained on the richly annotated GraspNet-1Billion and SuctionNet-1Billion datasets, which have been aligned for the purpose, and generates graspability masks quantifying the suitability of each scene point for successful grasps. By sharing early-stage features while maintaining gripper-specific refiners, MultiGraspNet effectively leverages complementary information across grasping modalities, enhancing robustness and adaptability in cluttered scenes. We characterize MultiGraspNet's performance with an extensive experimental analysis, demonstrating its competitiveness with single-task models on relevant benchmarks. We run real-world experiments on a single-arm multi-gripper robotic setup showing that our approach outperforms the vacuum baseline, grasping 16% percent more seen objects and 32% more of the novel ones, while obtaining competitive results for the parallel task.
>
---
#### [new 083] Analyzing Diffusion and Autoregressive Vision Language Models in Multimodal Embedding Space
- **分类: cs.MM; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究多模态扩散模型作为嵌入模型的性能，对比其与自回归模型在分类、视觉问答和检索任务中的表现，发现扩散模型在图像-文本对齐上存在不足。**

- **链接: [https://arxiv.org/pdf/2602.06056v1](https://arxiv.org/pdf/2602.06056v1)**

> **作者:** Zihang Wang; Siyue Zhang; Yilun Zhao; Jingyi Yang; Tingyu Song; Anh Tuan Luu; Chen Zhao
>
> **摘要:** Embedding models are a fundamental component of modern AI systems such as semantic search and retrieval-augmented generation. Recent advances in large foundation models have substantially accelerated the development of embedding models, including those based on Large Language Models (LLMs), Vision Language Models (VLMs), and Multimodal LLMs. More recently, Large Diffusion Language Models (dLLMs) and Multimodal dLLMs have emerged as competitive alternatives to autoregressive models, offering advantages such as bidirectional attention and parallel generation. This progress naturally raises a critical yet unexplored question: can Multimodal dLLMs serve as effective multimodal embedding models? To answer this, we present the first systematic study of converting Multimodal dLLMs into embedding models. We evaluate state-of-the-art Multimodal dLLMs and Autoregressive VLMs across three categories of embedding tasks: classification, visual question answering, and information retrieval. Our results show that Multimodal dLLM embeddings generally underperform their autoregressive VLM counterparts. The stronger diffusion-based model, LaViDa, lags by only 3.5 points on classification, 2.5 points on VQA, and 4.4 points on retrieval tasks, whereas the other diffusion-based model, MMaDA, exhibits substantially larger performance gaps, exceeding 20 points across all tasks. Further analysis reveals insufficient image-text alignment in diffusion-based models, accounting for the observed limitations in their embedding performance.
>
---
#### [new 084] SVRepair: Structured Visual Reasoning for Automated Program Repair
- **分类: cs.SE; cs.AI; cs.CV**

- **简介: 该论文属于程序修复任务，解决传统方法未充分利用视觉信息的问题。提出SVRepair框架，通过结构化视觉表示提升多模态程序修复效果。**

- **链接: [https://arxiv.org/pdf/2602.06090v1](https://arxiv.org/pdf/2602.06090v1)**

> **作者:** Xiaoxuan Tang; Jincheng Wang; Liwei Luo; Jingxuan Xu; Sheng Zhou; Dajun Chen; Wei Jiang; Yong Li
>
> **备注:** 16 pages, 3 figures
>
> **摘要:** Large language models (LLMs) have recently shown strong potential for Automated Program Repair (APR), yet most existing approaches remain unimodal and fail to leverage the rich diagnostic signals contained in visual artifacts such as screenshots and control-flow graphs. In practice, many bug reports convey critical information visually (e.g., layout breakage or missing widgets), but directly using such dense visual inputs often causes context loss and noise, making it difficult for MLLMs to ground visual observations into precise fault localization and executable patches. To bridge this semantic gap, we propose \textbf{SVRepair}, a multimodal APR framework with structured visual representation. SVRepair first fine-tunes a vision-language model, \textbf{Structured Visual Representation (SVR)}, to uniformly transform heterogeneous visual artifacts into a \emph{semantic scene graph} that captures GUI elements and their structural relations (e.g., hierarchy), providing normalized, code-relevant context for downstream repair. Building on the graph, SVRepair drives a coding agent to localize faults and synthesize patches, and further introduces an iterative visual-artifact segmentation strategy that progressively narrows the input to bug-centered regions to suppress irrelevant context and reduce hallucinations. Extensive experiments across multiple benchmarks demonstrate state-of-the-art performance: SVRepair achieves \textbf{36.47\%} accuracy on SWE-Bench M, \textbf{38.02\%} on MMCode, and \textbf{95.12\%} on CodeVision, validating the effectiveness of SVRepair for multimodal program repair.
>
---
#### [new 085] Zero-shot Multi-Contrast Brain MRI Registration by Intensity Randomizing T1-weighted MRI (LUMIR25)
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于多对比脑部MRI配准任务，解决在域偏移下的零样本配准问题。通过强度随机化、多模态损失和实例优化提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06292v1](https://arxiv.org/pdf/2602.06292v1)**

> **作者:** Hengjie Liu; Yimeng Dou; Di Xu; Xinyi Fu; Dan Ruan; Ke Sheng
>
> **备注:** Submitted to and reviewed by Learn2Reg MICCAI 2025
>
> **摘要:** In this paper, we summarize the methods and results of our submission to the LUMIR25 challenge in Learn2Reg 2025, which achieved 1st place overall on the test set. Extended from LUMIR24, this year's task focuses on zero-shot registration under domain shifts (high-field MRI, pathological brains, and various MRI contrasts), while the training data comprise only in-domain T1-weighted brain MRI. We start with a meticulous analysis of LUMIR24 winners to identify the main contributors to good monomodal registration performance. To achieve good generalization with diverse contrasts from a model trained with T1-weighted MRI only, we employ three simple but effective strategies: (i) a multimodal loss based on the modality-independent neighborhood descriptor (MIND), (ii) intensity randomization for appearance augmentation, and (iii) lightweight instance-specific optimization (ISO) on feature encoders at inference time. On the validation set, our approach achieves reasonable T1-T2 registration accuracy while maintaining good deformation regularity.
>
---
#### [new 086] Vision Transformer Finetuning Benefits from Non-Smooth Components
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文属于视觉Transformer研究任务，旨在解决微调性能优化问题。通过分析组件的可塑性，发现高可塑性模块能提升微调效果。**

- **链接: [https://arxiv.org/pdf/2602.06883v1](https://arxiv.org/pdf/2602.06883v1)**

> **作者:** Ambroise Odonnat; Laetitia Chapel; Romain Tavenard; Ievgen Redko
>
> **摘要:** The smoothness of the transformer architecture has been extensively studied in the context of generalization, training stability, and adversarial robustness. However, its role in transfer learning remains poorly understood. In this paper, we analyze the ability of vision transformer components to adapt their outputs to changes in inputs, or, in other words, their plasticity. Defined as an average rate of change, it captures the sensitivity to input perturbation; in particular, a high plasticity implies low smoothness. We demonstrate through theoretical analysis and comprehensive experiments that this perspective provides principled guidance in choosing the components to prioritize during adaptation. A key takeaway for practitioners is that the high plasticity of the attention modules and feedforward layers consistently leads to better finetuning performance. Our findings depart from the prevailing assumption that smoothness is desirable, offering a novel perspective on the functional properties of transformers. The code is available at https://github.com/ambroiseodt/vit-plasticity.
>
---
#### [new 087] ALIEN: Analytic Latent Watermarking for Controllable Generation
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属于生成模型水印任务，旨在解决水印鲁棒性与生成质量平衡问题。提出ALIEN框架，通过分析方法实现可控水印嵌入，提升性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06101v1](https://arxiv.org/pdf/2602.06101v1)**

> **作者:** Liangqi Lei; Keke Gai; Jing Yu; Qi Wu
>
> **摘要:** Watermarking is a technical alternative to safeguarding intellectual property and reducing misuse. Existing methods focus on optimizing watermarked latent variables to balance watermark robustness and fidelity, as Latent diffusion models (LDMs) are considered a powerful tool for generative tasks. However, reliance on computationally intensive heuristic optimization for iterative signal refinement results in high training overhead and local optima entrapment.To address these issues, we propose an \underline{A}na\underline{l}ytical Watermark\underline{i}ng Framework for Controllabl\underline{e} Generatio\underline{n} (ALIEN). We develop the first analytical derivation of the time-dependent modulation coefficient that guides the diffusion of watermark residuals to achieve controllable watermark embedding pattern.Experimental results show that ALIEN-Q outperforms the state-of-the-art by 33.1\% across 5 quality metrics, and ALIEN-R demonstrates 14.0\% improved robustness against generative variant and stability threats compared to the state-of-the-art across 15 distinct conditions. Code can be available at https://anonymous.4open.science/r/ALIEN/.
>
---
#### [new 088] CORP: Closed-Form One-shot Representation-Preserving Structured Pruning for Vision Transformers
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决Vision Transformers推理成本高的问题。提出CORP框架，在无需微调的情况下，通过结构化剪枝大幅降低计算和内存消耗，同时保持模型精度。**

- **链接: [https://arxiv.org/pdf/2602.05243v1](https://arxiv.org/pdf/2602.05243v1)**

> **作者:** Boxiang Zhang; Baijian Yang
>
> **摘要:** Vision Transformers achieve strong accuracy but incur high compute and memory cost. Structured pruning can reduce inference cost, but most methods rely on retraining or multi-stage optimization. These requirements limit post-training deployment. We propose \textbf{CORP}, a closed-form one-shot structured pruning framework for Vision Transformers. CORP removes entire MLP hidden dimensions and attention substructures without labels, gradients, or fine-tuning. It operates under strict post-training constraints using only a small unlabeled calibration set. CORP formulates structured pruning as a representation recovery problem. It models removed activations and attention logits as affine functions of retained components and derives closed-form ridge regression solutions that fold compensation into model weights. This minimizes expected representation error under the calibration distribution. Experiments on ImageNet with DeiT models show strong redundancy in MLP and attention representations. Without compensation, one-shot structured pruning causes severe accuracy degradation. With CORP, models preserve accuracy under aggressive sparsity. On DeiT-Huge, CORP retains 82.8\% Top-1 accuracy after pruning 50\% of both MLP and attention structures. CORP completes pruning in under 20 minutes on a single GPU and delivers substantial real-world efficiency gains.
>
---
#### [new 089] Relevance-aware Multi-context Contrastive Decoding for Retrieval-augmented Visual Question Answering
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决LVLMs缺乏实体细节知识的问题。提出RMCD方法，通过多上下文对比解码提升RAG效果，有效融合相关上下文并抑制无关影响。**

- **链接: [https://arxiv.org/pdf/2602.06050v1](https://arxiv.org/pdf/2602.06050v1)**

> **作者:** Jongha Kim; Byungoh Ko; Jeehye Na; Jinsung Yoon; Hyunwoo J. Kim
>
> **备注:** WACV 2026
>
> **摘要:** Despite the remarkable capabilities of Large Vision Language Models (LVLMs), they still lack detailed knowledge about specific entities. Retrieval-augmented Generation (RAG) is a widely adopted solution that enhances LVLMs by providing additional contexts from an external Knowledge Base. However, we observe that previous decoding methods for RAG are sub-optimal as they fail to sufficiently leverage multiple relevant contexts and suppress the negative effects of irrelevant contexts. To this end, we propose Relevance-aware Multi-context Contrastive Decoding (RMCD), a novel decoding method for RAG. RMCD outputs a final prediction by combining outputs predicted with each context, where each output is weighted based on its relevance to the question. By doing so, RMCD effectively aggregates useful information from multiple relevant contexts while also counteracting the negative effects of irrelevant ones. Experiments show that RMCD consistently outperforms other decoding methods across multiple LVLMs, achieving the best performance on three knowledge-intensive visual question-answering benchmarks. Also, RMCD can be simply applied by replacing the decoding method of LVLMs without additional training. Analyses also show that RMCD is robust to the retrieval results, consistently performing the best across the weakest to the strongest retrieval results. Code is available at https://github.com/mlvlab/RMCD.
>
---
#### [new 090] DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出DreamDojo，一个基于大规模人类视频的通用机器人世界模型，解决少数据和缺乏动作标签的问题。通过学习多样化交互与精细控制，提升机器人在复杂环境中的模拟能力。**

- **链接: [https://arxiv.org/pdf/2602.06949v1](https://arxiv.org/pdf/2602.06949v1)**

> **作者:** Shenyuan Gao; William Liang; Kaiyuan Zheng; Ayaan Malik; Seonghyeon Ye; Sihyun Yu; Wei-Cheng Tseng; Yuzhu Dong; Kaichun Mo; Chen-Hsuan Lin; Qianli Ma; Seungjun Nah; Loic Magne; Jiannan Xiang; Yuqi Xie; Ruijie Zheng; Dantong Niu; You Liang Tan; K. R. Zentner; George Kurian; Suneel Indupuru; Pooya Jannaty; Jinwei Gu; Jun Zhang; Jitendra Malik; Pieter Abbeel; Ming-Yu Liu; Yuke Zhu; Joel Jang; Linxi "Jim" Fan
>
> **备注:** Project page: https://dreamdojo-world.github.io/
>
> **摘要:** Being able to simulate the outcomes of actions in varied environments will revolutionize the development of generalist agents at scale. However, modeling these world dynamics, especially for dexterous robotics tasks, poses significant challenges due to limited data coverage and scarce action labels. As an endeavor towards this end, we introduce DreamDojo, a foundation world model that learns diverse interactions and dexterous controls from 44k hours of egocentric human videos. Our data mixture represents the largest video dataset to date for world model pretraining, spanning a wide range of daily scenarios with diverse objects and skills. To address the scarcity of action labels, we introduce continuous latent actions as unified proxy actions, enhancing interaction knowledge transfer from unlabeled videos. After post-training on small-scale target robot data, DreamDojo demonstrates a strong understanding of physics and precise action controllability. We also devise a distillation pipeline that accelerates DreamDojo to a real-time speed of 10.81 FPS and further improves context consistency. Our work enables several important applications based on generative world models, including live teleoperation, policy evaluation, and model-based planning. Systematic evaluation on multiple challenging out-of-distribution (OOD) benchmarks verifies the significance of our method for simulating open-world, contact-rich tasks, paving the way for general-purpose robot world models.
>
---
#### [new 091] Think Proprioceptively: Embodied Visual Reasoning for VLA Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决机器人如何有效利用本体感知提升操作性能。通过将本体信息转化为文本标记并早期融合，优化视觉推理与动作选择。**

- **链接: [https://arxiv.org/pdf/2602.06575v1](https://arxiv.org/pdf/2602.06575v1)**

> **作者:** Fangyuan Wang; Peng Zhou; Jiaming Qi; Shipeng Lyu; David Navarro-Alarcon; Guodong Guo
>
> **摘要:** Vision-language-action (VLA) models typically inject proprioception only as a late conditioning signal, which prevents robot state from shaping instruction understanding and from influencing which visual tokens are attended throughout the policy. We introduce ThinkProprio, which converts proprioception into a sequence of text tokens in the VLM embedding space and fuses them with the task instruction at the input. This early fusion lets embodied state participate in subsequent visual reasoning and token selection, biasing computation toward action-critical evidence while suppressing redundant visual tokens. In a systematic ablation over proprioception encoding, state entry point, and action-head conditioning, we find that text tokenization is more effective than learned projectors, and that retaining roughly 15% of visual tokens can match the performance of using the full token set. Across CALVIN, LIBERO, and real-world manipulation, ThinkProprio matches or improves over strong baselines while reducing end-to-end inference latency over 50%.
>
---
#### [new 092] COSMOS: Coherent Supergaussian Modeling with Spatial Priors for Sparse-View 3D Splatting
- **分类: eess.IV; cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，解决稀疏视角下3D高斯点云的过拟合与结构退化问题。提出COSMOS方法，引入结构先验和空间注意力机制，提升重建一致性与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.06044v1](https://arxiv.org/pdf/2602.06044v1)**

> **作者:** Chaeyoung Jeong; Kwangsu Kim
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently emerged as a promising approach for 3D reconstruction, providing explicit, point-based representations and enabling high-quality real time rendering. However, when trained with sparse input views, 3DGS suffers from overfitting and structural degradation, leading to poor generalization on novel views. This limitation arises from its optimization relying solely on photometric loss without incorporating any 3D structure priors. To address this issue, we propose Coherent supergaussian Modeling with Spatial Priors (COSMOS). Inspired by the concept of superpoints from 3D segmentation, COSMOS introduces 3D structure priors by newly defining supergaussian groupings of Gaussians based on local geometric cues and appearance features. To this end, COSMOS applies inter group global self-attention across supergaussian groups and sparse local attention among individual Gaussians, enabling the integration of global and local spatial information. These structure-aware features are then used for predicting Gaussian attributes, facilitating more consistent 3D reconstructions. Furthermore, by leveraging supergaussian-based grouping, COSMOS enforces an intra-group positional regularization to maintain structural coherence and suppress floaters, thereby enhancing training stability under sparse-view conditions. Our experiments on Blender and DTU show that COSMOS surpasses state-of-the-art methods in sparse-view settings without any external depth supervision.
>
---
#### [new 093] Orientation-Robust Latent Motion Trajectory Learning for Annotation-free Cardiac Phase Detection in Fetal Echocardiography
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于心脏相位检测任务，旨在无需人工标注的情况下，准确识别胎儿超声中的舒张末期和收缩末期帧。通过自监督学习方法，模型在不同心脏方位下实现鲁棒的相位定位。**

- **链接: [https://arxiv.org/pdf/2602.06761v1](https://arxiv.org/pdf/2602.06761v1)**

> **作者:** Yingyu Yang; Qianye Yang; Can Peng; Elena D'Alberti; Olga Patey; Aris T. Papageorghiou; J. Alison Noble
>
> **备注:** Preprint, Submitted to a journal
>
> **摘要:** Fetal echocardiography is essential for detecting congenital heart disease (CHD), facilitating pregnancy management, optimized delivery planning, and timely postnatal interventions. Among standard imaging planes, the four-chamber (4CH) view provides comprehensive information for CHD diagnosis, where clinicians carefully inspect the end-diastolic (ED) and end-systolic (ES) phases to evaluate cardiac structure and motion. Automated detection of these cardiac phases is thus a critical component toward fully automated CHD analysis. Yet, in the absence of fetal electrocardiography (ECG), manual identification of ED and ES frames remains a labor-intensive bottleneck. We present ORBIT (Orientation-Robust Beat Inference from Trajectories), a self-supervised framework that identifies cardiac phases without manual annotations under various fetal heart orientation. ORBIT employs registration as self-supervision task and learns a latent motion trajectory of cardiac deformation, whose turning points capture transitions between cardiac relaxation and contraction, enabling accurate and orientation-robust localization of ED and ES frames across diverse fetal positions. Trained exclusively on normal fetal echocardiography videos, ORBIT achieves consistent performance on both normal (MAE = 1.9 frames for ED and 1.6 for ES) and CHD cases (MAE = 2.4 frames for ED and 2.1 for ES), outperforming existing annotation-free approaches constrained by fixed orientation assumptions. These results highlight the potential of ORBIT to facilitate robust cardiac phase detection directly from 4CH fetal echocardiography.
>
---
#### [new 094] Same Answer, Different Representations: Hidden instability in VLMs
- **分类: cs.AI; cs.CV**

- **简介: 该论文研究视觉语言模型（VLM）的稳定性问题，揭示其输出不变但内部表示变化的隐性不稳定性，提出新评估框架并分析不同任务下的失败模式。**

- **链接: [https://arxiv.org/pdf/2602.06652v1](https://arxiv.org/pdf/2602.06652v1)**

> **作者:** Farooq Ahmad Wani; Alessandro Suglia; Rohit Saxena; Aryo Pradipta Gema; Wai-Chung Kwan; Fazl Barez; Maria Sofia Bucarelli; Fabrizio Silvestri; Pasquale Minervini
>
> **摘要:** The robustness of Vision Language Models (VLMs) is commonly assessed through output-level invariance, implicitly assuming that stable predictions reflect stable multimodal processing. In this work, we argue that this assumption is insufficient. We introduce a representation-aware and frequency-aware evaluation framework that measures internal embedding drift, spectral sensitivity, and structural smoothness (spatial consistency of vision tokens), alongside standard label-based metrics. Applying this framework to modern VLMs across the SEEDBench, MMMU, and POPE datasets reveals three distinct failure modes. First, models frequently preserve predicted answers while undergoing substantial internal representation drift; for perturbations such as text overlays, this drift approaches the magnitude of inter-image variability, indicating that representations move to regions typically occupied by unrelated inputs despite unchanged outputs. Second, robustness does not improve with scale; larger models achieve higher accuracy but exhibit equal or greater sensitivity, consistent with sharper yet more fragile decision boundaries. Third, we find that perturbations affect tasks differently: they harm reasoning when they disrupt how models combine coarse and fine visual cues, but on the hallucination benchmarks, they can reduce false positives by making models generate more conservative answers.
>
---
#### [new 095] Trifuse: Enhancing Attention-Based GUI Grounding via Multimodal Fusion
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于GUI接地任务，解决现有方法依赖大量数据和泛化能力差的问题。提出Trifuse框架，通过多模态融合提升注意力机制的可靠性。**

- **链接: [https://arxiv.org/pdf/2602.06351v1](https://arxiv.org/pdf/2602.06351v1)**

> **作者:** Longhui Ma; Di Zhao; Siwei Wang; Zhao Lv; Miao Wang
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** GUI grounding maps natural language instructions to the correct interface elements, serving as the perception foundation for GUI agents. Existing approaches predominantly rely on fine-tuning multimodal large language models (MLLMs) using large-scale GUI datasets to predict target element coordinates, which is data-intensive and generalizes poorly to unseen interfaces. Recent attention-based alternatives exploit localization signals in MLLMs attention mechanisms without task-specific fine-tuning, but suffer from low reliability due to the lack of explicit and complementary spatial anchors in GUI images. To address this limitation, we propose Trifuse, an attention-based grounding framework that explicitly integrates complementary spatial anchors. Trifuse integrates attention, OCR-derived textual cues, and icon-level caption semantics via a Consensus-SinglePeak (CS) fusion strategy that enforces cross-modal agreement while retaining sharp localization peaks. Extensive evaluations on four grounding benchmarks demonstrate that Trifuse achieves strong performance without task-specific fine-tuning, substantially reducing the reliance on expensive annotated data. Moreover, ablation studies reveal that incorporating OCR and caption cues consistently improves attention-based grounding performance across different backbones, highlighting its effectiveness as a general framework for GUI grounding.
>
---
#### [new 096] OmniVideo-R1: Reinforcing Audio-visual Reasoning with Query Intention and Modality Attention
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态视频理解任务，旨在提升音频-视觉联合推理能力。通过引入查询强化和模态注意力机制，增强模型对多模态信息的融合与理解。**

- **链接: [https://arxiv.org/pdf/2602.05847v1](https://arxiv.org/pdf/2602.05847v1)**

> **作者:** Zhangquan Chen; Jiale Tao; Ruihuang Li; Yihao Hu; Ruitao Chen; Zhantao Yang; Xinlei Yu; Haodong Jing; Manyuan Zhang; Shuai Shao; Biao Wang; Qinglin Lu; Ruqi Huang
>
> **备注:** 19 pages, 12 figures
>
> **摘要:** While humans perceive the world through diverse modalities that operate synergistically to support a holistic understanding of their surroundings, existing omnivideo models still face substantial challenges on audio-visual understanding tasks. In this paper, we propose OmniVideo-R1, a novel reinforced framework that improves mixed-modality reasoning. OmniVideo-R1 empowers models to "think with omnimodal cues" by two key strategies: (1) query-intensive grounding based on self-supervised learning paradigms; and (2) modality-attentive fusion built upon contrastive learning paradigms. Extensive experiments on multiple benchmarks demonstrate that OmniVideo-R1 consistently outperforms strong baselines, highlighting its effectiveness and robust generalization capabilities.
>
---
#### [new 097] AS-Mamba: Asymmetric Self-Guided Mamba Decoupled Iterative Network for Metal Artifact Reduction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于金属伪影去除任务，旨在提升CT图像质量。针对现有方法无法有效捕捉伪影方向特征的问题，提出AS-Mamba网络，结合状态空间模型与频域修正，提升结构保真度。**

- **链接: [https://arxiv.org/pdf/2602.06350v1](https://arxiv.org/pdf/2602.06350v1)**

> **作者:** Bowen Ning; Zekun Zhou; Xinyi Zhong; Zhongzhen Wang; HongXin Wu; HaiTao Wang; Liu Shi; Qiegen Liu
>
> **备注:** 10 pages,10 figures
>
> **摘要:** Metal artifact significantly degrades Computed Tomography (CT) image quality, impeding accurate clinical diagnosis. However, existing deep learning approaches, such as CNN and Transformer, often fail to explicitly capture the directional geometric features of artifacts, leading to compromised structural restoration. To address these limitations, we propose the Asymmetric Self-Guided Mamba (AS-Mamba) for metal artifact reduction. Specifically, the linear propagation of metal-induced streak artifacts aligns well with the sequential modeling capability of State Space Models (SSMs). Consequently, the Mamba architecture is leveraged to explicitly capture and suppress these directional artifacts. Simultaneously, a frequency domain correction mechanism is incorporated to rectify the global amplitude spectrum, thereby mitigating intensity inhomogeneity caused by beam hardening. Furthermore, to bridge the distribution gap across diverse clinical scenarios, we introduce a self-guided contrastive regularization strategy. Extensive experiments on public andclinical dental CBCT datasets demonstrate that AS-Mamba achieves superior performance in suppressing directional streaks and preserving structural details, validating the effectiveness of integrating physical geometric priors into deep network design.
>
---
## 更新

#### [replaced 001] A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.03604v2](https://arxiv.org/pdf/2602.03604v2)**

> **作者:** Basile Terver; Randall Balestriero; Megi Dervishi; David Fan; Quentin Garrido; Tushar Nagarajan; Koustuv Sinha; Wancong Zhang; Mike Rabbat; Yann LeCun; Amir Bar
>
> **备注:** v2: clarify confusion in definition of JEPAs vs. regularization-based JEPAs
>
> **摘要:** We present EB-JEPA, an open-source library for learning representations and world models using Joint-Embedding Predictive Architectures (JEPAs). JEPAs learn to predict in representation space rather than pixel space, avoiding the pitfalls of generative modeling while capturing semantically meaningful features suitable for downstream tasks. Our library provides modular, self-contained implementations that illustrate how representation learning techniques developed for image-level self-supervised learning can transfer to video, where temporal dynamics add complexity, and ultimately to action-conditioned world models, where the model must additionally learn to predict the effects of control inputs. Each example is designed for single-GPU training within a few hours, making energy-based self-supervised learning accessible for research and education. We provide ablations of JEA components on CIFAR-10. Probing these representations yields 91% accuracy, indicating that the model learns useful features. Extending to video, we include a multi-step prediction example on Moving MNIST that demonstrates how the same principles scale to temporal modeling. Finally, we show how these representations can drive action-conditioned world models, achieving a 97% planning success rate on the Two Rooms navigation task. Comprehensive ablations reveal the critical importance of each regularization component for preventing representation collapse. Code is available at https://github.com/facebookresearch/eb_jepa.
>
---
#### [replaced 002] WAFT: Warping-Alone Field Transforms for Optical Flow
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.21526v3](https://arxiv.org/pdf/2506.21526v3)**

> **作者:** Yihan Wang; Jia Deng
>
> **摘要:** We introduce Warping-Alone Field Transforms (WAFT), a simple and effective method for optical flow. WAFT is similar to RAFT but replaces cost volume with high-resolution warping, achieving better accuracy with lower memory cost. This design challenges the conventional wisdom that constructing cost volumes is necessary for strong performance. WAFT is a simple and flexible meta-architecture with minimal inductive biases and reliance on custom designs. Compared with existing methods, WAFT ranks 1st on Spring, Sintel, and KITTI benchmarks, achieves the best zero-shot generalization on KITTI, while being 1.3-4.1x faster than existing methods that have competitive accuracy (e.g., 1.3x than Flowformer++, 4.1x than CCMR+). Code and model weights are available at \href{https://github.com/princeton-vl/WAFT}{https://github.com/princeton-vl/WAFT}.
>
---
#### [replaced 003] Generative Modeling via Drifting
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.04770v2](https://arxiv.org/pdf/2602.04770v2)**

> **作者:** Mingyang Deng; He Li; Tianhong Li; Yilun Du; Kaiming He
>
> **备注:** Project page: https://lambertae.github.io/projects/drifting/
>
> **摘要:** Generative modeling can be formulated as learning a mapping f such that its pushforward distribution matches the data distribution. The pushforward behavior can be carried out iteratively at inference time, for example in diffusion and flow-based models. In this paper, we propose a new paradigm called Drifting Models, which evolve the pushforward distribution during training and naturally admit one-step inference. We introduce a drifting field that governs the sample movement and achieves equilibrium when the distributions match. This leads to a training objective that allows the neural network optimizer to evolve the distribution. In experiments, our one-step generator achieves state-of-the-art results on ImageNet at 256 x 256 resolution, with an FID of 1.54 in latent space and 1.61 in pixel space. We hope that our work opens up new opportunities for high-quality one-step generation.
>
---
#### [replaced 004] Probing Perceptual Constancy in Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.10273v3](https://arxiv.org/pdf/2502.10273v3)**

> **作者:** Haoran Sun; Bingyang Wang; Suyang Yu; Yijiang Li; Qingying Gao; Haiyun Lyu; Lianyu Huang; Zelong Hong; Jiahui Ge; Qianli Ma; Hang He; Yifan Zhou; Lingzi Guo; Lantao Mei; Maijunxian Wang; Dezhi Luo; Hokin Deng
>
> **备注:** Under Review
>
> **摘要:** Perceptual constancy is the ability to maintain stable perceptions of objects despite changes in sensory input, such as variations in distance, angle, or lighting. This ability is crucial for visual understanding in a dynamic world. Here, we explored such ability in current Vision Language Models (VLMs). In this study, we evaluated 155 VLMs using 236 experiments across three domains: color, size, and shape constancy. The experiments included single-image and video adaptations of classic cognitive tasks, along with novel tasks in in-the-wild conditions. We found significant variability in VLM performance across these domains, with model performance in shape constancy clearly dissociated from that of color and size constancy.
>
---
#### [replaced 005] Preserving Spectral Structure and Statistics in Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17873v2](https://arxiv.org/pdf/2512.17873v2)**

> **作者:** Baohua Yan; Jennifer Kava; Qingyuan Liu; Xuan Di
>
> **摘要:** Standard diffusion models (DMs) rely on the total destruction of data into non-informative white noise, forcing the backward process to denoise from a fully unstructured noise state. While ensuring diversity, this results in a cumbersome and computationally intensive image generation task. We address this challenge by proposing new forward and backward process within a mathematically tractable spectral space. Unlike pixel-based DMs, our forward process converges towards an informative Gaussian prior N(mu_hat,Sigma_hat) rather than white noise. Our method, termed Preserving Spectral Structure and Statistics (PreSS) in diffusion models, guides spectral components toward this informative prior while ensuring that corresponding structural signals remain intact at terminal time. This provides a principled starting point for the backward process, enabling high-quality image reconstruction that builds upon preserved spectral structure while maintaining high generative diversity. Experimental results on CIFAR-10, CelebA and CelebA-HQ demonstrate significant reductions in computational complexity, improved visual diversity, less drift, and a smoother diffusion process compared to pixel-based DMs.
>
---
#### [replaced 006] Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02214v2](https://arxiv.org/pdf/2602.02214v2)**

> **作者:** Hongzhou Zhu; Min Zhao; Guande He; Hang Su; Chongxuan Li; Jun Zhu
>
> **备注:** Project page and the code: \href{https://thu-ml.github.io/CausalForcing.github.io/}{https://thu-ml.github.io/CausalForcing.github.io/}
>
> **摘要:** To achieve real-time interactive video generation, current methods distill pretrained bidirectional video diffusion models into few-step autoregressive (AR) models, facing an architectural gap when full attention is replaced by causal attention. However, existing approaches do not bridge this gap theoretically. They initialize the AR student via ODE distillation, which requires frame-level injectivity, where each noisy frame must map to a unique clean frame under the PF-ODE of an AR teacher. Distilling an AR student from a bidirectional teacher violates this condition, preventing recovery of the teacher's flow map and instead inducing a conditional-expectation solution, which degrades performance. To address this issue, we propose Causal Forcing that uses an AR teacher for ODE initialization, thereby bridging the architectural gap. Empirical results show that our method outperforms all baselines across all metrics, surpassing the SOTA Self Forcing by 19.3\% in Dynamic Degree, 8.7\% in VisionReward, and 16.7\% in Instruction Following. Project page and the code: \href{https://thu-ml.github.io/CausalForcing.github.io/}{https://thu-ml.github.io/CausalForcing.github.io/}
>
---
#### [replaced 007] Adaptive Attention Distillation for Robust Few-Shot Segmentation under Environmental Perturbations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.03596v2](https://arxiv.org/pdf/2601.03596v2)**

> **作者:** Qianyu Guo; Jingrong Wu; Jieji Ren; Weifeng Ge; Wenqiang Zhang
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Few-shot segmentation (FSS) aims to rapidly learn novel class concepts from limited examples to segment specific targets in unseen images, and has been widely applied in areas such as medical diagnosis and industrial inspection. However, existing studies largely overlook the complex environmental factors encountered in real world scenarios-such as illumination, background, and camera viewpoint-which can substantially increase the difficulty of test images. As a result, models trained under laboratory conditions often fall short of practical deployment requirements. To bridge this gap, in this paper, an environment-robust FSS setting is introduced that explicitly incorporates challenging test cases arising from complex environments-such as motion blur, small objects, and camouflaged targets-to enhance model's robustness under realistic, dynamic conditions. An environment robust FSS benchmark (ER-FSS) is established, covering eight datasets across multiple real world scenarios. In addition, an Adaptive Attention Distillation (AAD) method is proposed, which repeatedly contrasts and distills key shared semantics between known (support) and unknown (query) images to derive class-specific attention for novel categories. This strengthens the model's ability to focus on the correct targets in complex environments, thereby improving environmental robustness. Comparative experiments show that AAD improves mIoU by 3.3% - 8.5% across all datasets and settings, demonstrating superior performance and strong generalization. The source code and dataset are available at: https://github.com/guoqianyu-alberta/Adaptive-Attention-Distillation-for-FSS.
>
---
#### [replaced 008] Multimodal Iterative RAG for Knowledge-Intensive Visual Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.00798v5](https://arxiv.org/pdf/2509.00798v5)**

> **作者:** Changin Choi; Wonseok Lee; Jungmin Ko; Wonjong Rhee
>
> **摘要:** Knowledge-intensive visual question answering (VQA) requires external knowledge beyond image content, demanding precise visual grounding and coherent integration of visual and textual information. Although multimodal retrieval-augmented generation has achieved notable advances by incorporating external knowledge bases, existing approaches largely adopt single-pass frameworks that often fail to acquire sufficient knowledge and lack mechanisms to revise misdirected reasoning. We propose PMSR (Progressive Multimodal Search and Reasoning), a framework that progressively constructs a structured reasoning trajectory to enhance both knowledge acquisition and synthesis. PMSR uses dual-scope queries conditioned on both the latest record and the trajectory to retrieve diverse knowledge from heterogeneous knowledge bases. The retrieved evidence is then synthesized into compact records via compositional reasoning. This design facilitates controlled iterative refinement, which supports more stable reasoning trajectories with reduced error propagation. Extensive experiments across six diverse benchmarks (Encyclopedic-VQA, InfoSeek, MMSearch, LiveVQA, FVQA, and OK-VQA) demonstrate that PMSR consistently improves both retrieval recall and end-to-end answer accuracy.
>
---
#### [replaced 009] Concepts in Motion: Temporal Bottlenecks for Interpretable Video Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.20899v2](https://arxiv.org/pdf/2509.20899v2)**

> **作者:** Patrick Knab; Sascha Marton; Philipp J. Schubert; Drago Guggiana; Christian Bartelt
>
> **摘要:** Concept Bottleneck Models (CBMs) enable interpretable image classification by structuring predictions around human-understandable concepts, but extending this paradigm to video remains challenging due to the difficulty of extracting concepts and modeling them over time. In this paper, we introduce $\textbf{MoTIF}$ (Moving Temporal Interpretable Framework), a transformer-based concept architecture that operates on sequences of temporally grounded concept activations, by employing per-concept temporal self-attention to model when individual concepts recur and how their temporal patterns contribute to predictions. Central to the framework is an agentic concept discovery module to automatically extract object- and action-centric textual concepts from videos, yielding temporally expressive concept sets without manual supervision. Across multiple video benchmarks, this combination substantially narrows the performance gap between interpretable and black-box video models while maintaining faithful and temporally grounded concept explanations. Code available at $\href{https://github.com/patrick-knab/MoTIF}{github.com/patrick-knab/MoTIF}$.
>
---
#### [replaced 010] DoRAN: Stabilizing Weight-Decomposed Low-Rank Adaptation via Noise Injection and Auxiliary Networks
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04331v2](https://arxiv.org/pdf/2510.04331v2)**

> **作者:** Nghiem T. Diep; Hien Dang; Tuan Truong; Tan Dinh; Huy Nguyen; Nhat Ho
>
> **备注:** Nghiem T. Diep, Hien Dang, and Tuan Truong contributed equally to this work
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) methods have become the standard paradigm for adapting large-scale models. Among these techniques, Weight-Decomposed Low-Rank Adaptation (DoRA) has been shown to improve both the learning capacity and training stability of the Low-Rank Adaptation (LoRA) method by explicitly decomposing pre-trained weights into magnitude and directional components. In this work, we propose DoRAN, a new technique designed to stabilize training and boost the sample efficiency of DoRA. Our framework introduces two key components: (i) the injection of learnable noise into the denominator of DoRA weight decomposition, which serves as an adaptive regularizer to mitigate instabilities and improve the estimation rate of low-rank matrices; and (ii) the replacement of static low-rank matrices with auxiliary networks that generate them dynamically, enabling parameter coupling between the query and value projection matrices, leading to improved sample efficiency both theoretically and empirically. Comprehensive experiments on vision and language benchmarks show that DoRAN consistently outperforms LoRA, DoRA, and other PEFT baselines, underscoring the effectiveness of combining noise-based regularization with network-based parameter generation.
>
---
#### [replaced 011] A Data-driven Typology of Vision Models from Integrated Representational Metrics
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.21628v3](https://arxiv.org/pdf/2509.21628v3)**

> **作者:** Jialin Wu; Shreya Saha; Yiqing Bo; Meenakshi Khosla
>
> **备注:** Update the main text format
>
> **摘要:** Large vision models differ widely in architecture and training paradigm, yet we lack principled methods to determine which aspects of their representations are shared across families and which reflect distinctive computational strategies. We leverage a suite of representational similarity metrics, each capturing a different facet-geometry, unit tuning, or linear decodability-and assess family separability using multiple complementary measures. Metrics preserving geometry or tuning (e.g., RSA, Soft Matching) yield strong family discrimination, whereas flexible mappings such as Linear Predictivity show weaker separation. These findings indicate that geometry and tuning carry family-specific signatures, while linearly decodable information is more broadly shared. To integrate these complementary facets, we adapt Similarity Network Fusion (SNF), a method inspired by multi-omics integration. SNF achieves substantially sharper family separation than any individual metric and produces robust composite signatures. Clustering of the fused similarity matrix recovers both expected and surprising patterns: supervised ResNets and ViTs form distinct clusters, yet all self-supervised models group together across architectural boundaries. Hybrid architectures (ConvNeXt, Swin) cluster with masked autoencoders, suggesting convergence between architectural modernization and reconstruction-based training. This biology-inspired framework provides a principled typology of vision models, showing that emergent computational strategies-shaped jointly by architecture and training objective-define representational structure beyond surface design categories.
>
---
#### [replaced 012] Robust Detection of Retinal Neovascularization in Widefield Optical Coherence Tomography
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17744v2](https://arxiv.org/pdf/2511.17744v2)**

> **作者:** Jinyi Hao; Jie Wang; Liqin Gao; Tristan T. Hormel; Yukun Guo; An-Lun Wu; Christina J. Flaxel; Steven T. Bailey; Kotaro Tsuboi; Thomas S. Hwang; Yali Jia
>
> **备注:** 21 pages, 12 figures. Submitted to Optica. Corresponding author: Yali Jia
>
> **摘要:** Retinal neovascularization (RNV) is a vision threatening development in diabetic retinopathy (DR). Vision loss associated with RNV is preventable with timely intervention, making RNV clinical screening and monitoring a priority. Optical coherence tomography (OCT) angiography (OCTA) provides high-resolution imaging and high-sensitivity detection of RNV lesions. With recent commercial devices introducing widefield OCTA imaging to the clinic, the technology stands to improve early detection of RNV pathology. However, to meet clinical requirements these imaging capabilities must be combined with effective RNV detection and quantification, but existing algorithms for OCTA images are optimized for conventional, i.e. narrow, fields of view. Here, we present a novel approach for RNV diagnosis and staging on widefield OCT/OCTA. Unlike conventional methods dependent on multi-layer retinal segmentation, our model reframes RNV identification as a direct binary localization task. Our fully automated approach was trained and validated on 589 widefield scans (17x17-mm to 26x21-mm) collected from multiple devices at multiple clinics. Our method achieved a device-dependent area under curve (AUC) ranging from 0.96 to 0.99 for RNV diagnosis, and mean intersection over union (IOU) ranging from 0.76 to 0.88 for segmentation. We also demonstrate our method's ability to monitor lesion growth longitudinally. Our results indicate that deep learning-based analysis for widefield OCTA images could offer a valuable means for improving RNV screening and management.
>
---
#### [replaced 013] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.06363v4](https://arxiv.org/pdf/2507.06363v4)**

> **作者:** Szymon Płotka; Gizem Mert; Maciej Chrabaszcz; Ewa Szczurek; Arkadiusz Sitek
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** In recent years, artificial intelligence has significantly advanced medical image segmentation. Nonetheless, challenges remain, including efficient 3D medical image processing across diverse modalities and handling data variability. In this work, we introduce Hierarchical Soft Mixture-of-Experts (HoME), a two-level token-routing layer for efficient long-context modeling, specifically designed for 3D medical image segmentation. Built on the Mamba Selective State Space Model (SSM) backbone, HoME enhances sequential modeling through adaptive expert routing. In the first level, a Soft Mixture-of-Experts (SMoE) layer partitions input sequences into local groups, routing tokens to specialized per-group experts for localized feature extraction. The second level aggregates these outputs through a global SMoE layer, enabling cross-group information fusion and global context refinement. This hierarchical design, combining local expert routing with global expert refinement, enhances generalizability and segmentation performance, surpassing state-of-the-art results across datasets from the three most widely used 3D medical imaging modalities and varying data qualities. The code is publicly available at https://github.com/gmum/MambaHoME.
>
---
#### [replaced 014] Inverse problems with diffusion models: MAP estimation via mode-seeking loss
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10524v2](https://arxiv.org/pdf/2512.10524v2)**

> **作者:** Sai Bharath Chandra Gutha; Ricardo Vinuesa; Hossein Azizpour
>
> **摘要:** A pre-trained unconditional diffusion model, combined with posterior sampling or maximum a posteriori (MAP) estimation techniques, can solve arbitrary inverse problems without task-specific training or fine-tuning. However, existing posterior sampling and MAP estimation methods often rely on modeling approximations and can also be computationally demanding. In this work, we propose a new MAP estimation strategy for solving inverse problems with a pre-trained unconditional diffusion model. Specifically, we introduce the variational mode-seeking loss (VML) and show that its minimization at each reverse diffusion step guides the generated sample towards the MAP estimate (modes in practice). VML arises from a novel perspective of minimizing the Kullback-Leibler (KL) divergence between the diffusion posterior $p(\mathbf{x}_0|\mathbf{x}_t)$ and the measurement posterior $p(\mathbf{x}_0|\mathbf{y})$, where $\mathbf{y}$ denotes the measurement. Importantly, for linear inverse problems, VML can be analytically derived without any modeling approximations. Based on further theoretical insights, we propose VML-MAP, an empirically effective algorithm for solving inverse problems via VML minimization, and validate its efficacy in both performance and computational time through extensive experiments on diverse image-restoration tasks across multiple datasets.
>
---
#### [replaced 015] XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.22726v3](https://arxiv.org/pdf/2506.22726v3)**

> **作者:** Yu Zhang; Xi Zhang; Hualin Zhou; Xinyuan Chen; Shang Gao; Hong Jia; Jianfei Yang; Yuankai Qi; Tao Gu
>
> **摘要:** Deep learning for human sensing on edge systems presents significant potential for smart applications. However, its training and development are hindered by the limited availability of sensor data and resource constraints of edge systems. While transferring pre-trained models to different sensing applications is promising, existing methods often require extensive sensor data and computational resources, resulting in high costs and limited transferability. In this paper, we propose XTransfer, a first-of-its-kind method enabling modality-agnostic, few-shot model transfer with resource-efficient design. XTransfer flexibly uses pre-trained models and transfers knowledge across different modalities by (i) model repairing that safely mitigates modality shift by adapting pre-trained layers with only few sensor data, and (ii) layer recombining that efficiently searches and recombines layers of interest from source models in a layer-wise manner to restructure models. We benchmark various baselines across diverse human sensing datasets spanning different modalities. The results show that XTransfer achieves state-of-the-art performance while significantly reducing the costs of sensor data collection, model training, and edge deployment.
>
---
#### [replaced 016] T$^3$-S2S: Training-free Triplet Tuning for Sketch to Scene Synthesis in Controllable Concept Art Generation
- **分类: cs.CV; cs.CL; cs.GR**

- **简介: 该论文属于2D概念艺术生成任务，解决3D场景生成中多实例和结构化地形布局的问题。提出T3-S2S方法，通过三模块优化生成更精确的场景图像。**

- **链接: [https://arxiv.org/pdf/2412.13486v2](https://arxiv.org/pdf/2412.13486v2)**

> **作者:** Zhenhong Sun; Yifu Wang; Yonhon Ng; Yongzhi Xu; Daoyi Dong; Hongdong Li; Pan Ji
>
> **备注:** https://openreview.net/forum?id=lyn2BgKQ8F
>
> **摘要:** 2D concept art generation for 3D scenes is a crucial yet challenging task in computer graphics, as creating natural intuitive environments still demands extensive manual effort in concept design. While generative AI has simplified 2D concept design via text-to-image synthesis, it struggles with complex multi-instance scenes and offers limited support for structured terrain layout. In this paper, we propose a Training-free Triplet Tuning for Sketch-to-Scene (T3-S2S) generation after reviewing the entire cross-attention mechanism. This scheme revitalizes the ControlNet model for detailed multi-instance generation via three key modules: Prompt Balance ensures keyword representation and minimizes the risk of missing critical instances; Characteristic Priority emphasizes sketch-based features by highlighting TopK indices in feature channels; and Dense Tuning refines contour details within instance-related regions of the attention map. Leveraging the controllability of T3-S2S, we also introduce a feature-sharing strategy with dual prompt sets to generate layer-aware isometric and terrain-view representations for the terrain layout. Experiments show that our sketch-to-scene workflow consistently produces multi-instance 2D scenes with details aligned with input prompts.
>
---
#### [replaced 017] Detecting Latin in Historical Books with Large Language Models: A Multimodal Benchmark
- **分类: cs.CL; cs.AI; cs.CV; cs.DL**

- **简介: 该论文属于历史文献处理任务，旨在从混合语言文档中检测拉丁语片段。通过构建多模态数据集，评估大模型的检测能力，建立处理拉丁语的基准。**

- **链接: [https://arxiv.org/pdf/2510.19585v3](https://arxiv.org/pdf/2510.19585v3)**

> **作者:** Yu Wu; Ke Shu; Jonas Fischer; Lidia Pivovarova; David Rosson; Eetu Mäkelä; Mikko Tolonen
>
> **备注:** Accepted by the EACL 2026 main conference. Code and data available at https://github.com/COMHIS/EACL26-detect-latin
>
> **摘要:** This paper presents a novel task of extracting low-resourced and noisy Latin fragments from mixed-language historical documents with varied layouts. We benchmark and evaluate the performance of large foundation models against a multimodal dataset of 724 annotated pages. The results demonstrate that reliable Latin detection with contemporary zero-shot models is achievable, yet these models lack a functional comprehension of Latin. This study establishes a comprehensive baseline for processing Latin within mixed-language corpora, supporting quantitative analysis in intellectual history and historical linguistics. Both the dataset and code are available at https://github.com/COMHIS/EACL26-detect-latin.
>
---
#### [replaced 018] SyncAnyone: Implicit Disentanglement via Progressive Self-Correction for Lip-Syncing in the wild
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.21736v3](https://arxiv.org/pdf/2512.21736v3)**

> **作者:** Xindi Zhang; Dechao Meng; Steven Xiao; Qi Wang; Peng Zhang; Bang Zhang
>
> **备注:** Project page: https://humanaigc.github.io/sync_anyone_demo_page/
>
> **摘要:** High-quality AI-powered video dubbing demands precise audio-lip synchronization, high-fidelity visual generation, and faithful preservation of identity and background. Most existing methods rely on a mask-based training strategy, where the mouth region is masked in talking-head videos, and the model learns to synthesize lip movements from corrupted inputs and target audios. While this facilitates lip-sync accuracy, it disrupts spatiotemporal context, impairing performance on dynamic facial motions and causing instability in facial structure and background consistency. To overcome this limitation, we propose SyncAnyone, a novel two-stage learning framework that achieves accurate motion modeling and high visual fidelity simultaneously. In Stage 1, we train a diffusion-based video transformer for masked mouth inpainting, leveraging its strong spatiotemporal modeling to generate accurate, audio-driven lip movements. However, due to input corruption, minor artifacts may arise in the surrounding facial regions and the background. In Stage 2, we develop a mask-free tuning pipeline to address mask-induced artifacts. Specifically, on the basis of the Stage 1 model, we develop a data generation pipeline that creates pseudo-paired training samples by synthesizing lip-synced videos from the source video and random sampled audio. We further tune the stage 2 model on this synthetic data, achieving precise lip editing and better background consistency. Extensive experiments show that our method achieves state-of-the-art results in visual quality, temporal coherence, and identity preservation under in-the wild lip-syncing scenarios.
>
---
#### [replaced 019] ReflexFlow: Rethinking Learning Objective for Exposure Bias Alleviation in Flow Matching
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.04904v2](https://arxiv.org/pdf/2512.04904v2)**

> **作者:** Guanbo Huang; Jingjia Mao; Fanding Huang; Fengkai Liu; Xiangyang Luo; Yaoyuan Liang; Jiasheng Lu; Xiaoe Wang; Pei Liu; Ruiliu Fu; Shao-Lun Huang
>
> **备注:** After careful consideration, we have decided to withdraw our submission for substantial revisions. We plan to significantly improve Section 4 and include more comprehensive experiments. These changes are necessary to ensure the paper's quality and rigor. We believe the revisions will strengthen the contribution and provide a more solid foundation for the results
>
> **摘要:** Despite tremendous recent progress, Flow Matching methods still suffer from exposure bias due to discrepancies in training and inference. This paper investigates the root causes of exposure bias in Flow Matching, including: (1) the model lacks generalization to biased inputs during training, and (2) insufficient low-frequency content captured during early denoising, leading to accumulated bias. Based on these insights, we propose ReflexFlow, a simple and effective reflexive refinement of the Flow Matching learning objective that dynamically corrects exposure bias. ReflexFlow consists of two components: (1) Anti-Drift Rectification (ADR), which reflexively adjusts prediction targets for biased inputs utilizing a redesigned loss under training-time scheduled sampling; and (2) Frequency Compensation (FC), which reflects on missing low-frequency components and compensates them by reweighting the loss using exposure bias. ReflexFlow is model-agnostic, compatible with all Flow Matching frameworks, and improves generation quality across datasets. Experiments on CIFAR-10, CelebA-64, and ImageNet-256 show that ReflexFlow outperforms prior approaches in mitigating exposure bias, achieving a 35.65% reduction in FID on CelebA-64.
>
---
#### [replaced 020] A Comparative Study of 3D Person Detection: Sensor Modalities and Robustness in Diverse Indoor and Outdoor Environments
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05538v2](https://arxiv.org/pdf/2602.05538v2)**

> **作者:** Malaz Tamim; Andrea Matic-Flierl; Karsten Roscher
>
> **备注:** Accepted for VISAPP 2026
>
> **摘要:** Accurate 3D person detection is critical for safety in applications such as robotics, industrial monitoring, and surveillance. This work presents a systematic evaluation of 3D person detection using camera-only, LiDAR-only, and camera-LiDAR fusion. While most existing research focuses on autonomous driving, we explore detection performance and robustness in diverse indoor and outdoor scenes using the JRDB dataset. We compare three representative models - BEVDepth (camera), PointPillars (LiDAR), and DAL (camera-LiDAR fusion) - and analyze their behavior under varying occlusion and distance levels. Our results show that the fusion-based approach consistently outperforms single-modality models, particularly in challenging scenarios. We further investigate robustness against sensor corruptions and misalignments, revealing that while DAL offers improved resilience, it remains sensitive to sensor misalignment and certain LiDAR-based corruptions. In contrast, the camera-based BEVDepth model showed the lowest performance and was most affected by occlusion, distance, and noise. Our findings highlight the importance of utilizing sensor fusion for enhanced 3D person detection, while also underscoring the need for ongoing research to address the vulnerabilities inherent in these systems.
>
---
#### [replaced 021] CompEvent: Complex-valued Event-RGB Fusion for Low-light Video Enhancement and Deblurring
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14469v3](https://arxiv.org/pdf/2511.14469v3)**

> **作者:** Mingchen Zhong; Xin Lu; Dong Li; Senyan Xu; Ruixuan Jiang; Xueyang Fu; Baocai Yin
>
> **摘要:** Low-light video deblurring poses significant challenges in applications like nighttime surveillance and autonomous driving due to dim lighting and long exposures. While event cameras offer potential solutions with superior low-light sensitivity and high temporal resolution, existing fusion methods typically employ staged strategies, limiting their effectiveness against combined low-light and motion blur degradations. To overcome this, we propose CompEvent, a complex neural network framework enabling holistic full-process fusion of event data and RGB frames for enhanced joint restoration. CompEvent features two core components: 1) Complex Temporal Alignment GRU, which utilizes complex-valued convolutions and processes video and event streams iteratively via GRU to achieve temporal alignment and continuous fusion; and 2) Complex Space-Frequency Learning module, which performs unified complex-valued signal processing in both spatial and frequency domains, facilitating deep fusion through spatial structures and system-level characteristics. By leveraging the holistic representation capability of complex-valued neural networks, CompEvent achieves full-process spatiotemporal fusion, maximizes complementary learning between modalities, and significantly strengthens low-light video deblurring capability. Extensive experiments demonstrate that CompEvent outperforms SOTA methods in addressing this challenging task.
>
---
#### [replaced 022] Self-Supervised Video Representation Learning in a Heuristic Decoupled Perspective
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.14069v2](https://arxiv.org/pdf/2407.14069v2)**

> **作者:** Zeen Song; Wenwen Qiang; Changwen Zheng; Hui Xiong; Gang Hua
>
> **摘要:** Video contrastive learning (V-CL) has emerged as a popular framework for unsupervised video representation learning, demonstrating strong results in tasks such as action classification and detection. Yet, to harness these benefits, it is critical for the learned representations to fully capture both static and dynamic semantics. However, our experiments show that existing V-CL methods fail to effectively learn either type of feature. Through a rigorous theoretical analysis based on the Structural Causal Model and gradient update, we find that in a given dataset, certain static semantics consistently co-occur with specific dynamic semantics. This phenomenon creates spurious correlations between static and dynamic semantics in the dataset. However, existing V-CL methods do not differentiate static and dynamic similarities when computing sample similarity. As a result, learning only one type of semantics is sufficient for the model to minimize the contrastive loss. Ultimately, this causes the V-CL pre-training process to prioritize learning the easier-to-learn semantics. To address this limitation, we propose Bi-level Optimization with Decoupling for Video Contrastive Learning. (BOD-VCL). In BOD-VCL, we model videos as linear dynamical systems based on Koopman theory. In this system, all frame-to-frame transitions are represented by a linear Koopman operator. By performing eigen-decomposition on this operator, we can separate time-variant and time-invariant components of semantics, which allows us to explicitly separate the static and dynamic semantics in the video. By modeling static and dynamic similarity separately, both types of semantics can be fully exploited during the V-CL training process. BOD-VCL can be seamlessly integrated into existing V-CL frameworks, and experimental results highlight the significant improvements achieved by our method.
>
---
#### [replaced 023] PromptSplit: Revealing Prompt-Level Disagreement in Generative Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.04009v2](https://arxiv.org/pdf/2602.04009v2)**

> **作者:** Mehdi Lotfian; Mohammad Jalali; Farzan Farnia
>
> **摘要:** Prompt-guided generative AI models have rapidly expanded across vision and language domains, producing realistic and diverse outputs from textual inputs. The growing variety of such models, trained with different data and architectures, calls for principled methods to identify which types of prompts lead to distinct model behaviors. In this work, we propose PromptSplit, a kernel-based framework for detecting and analyzing prompt-dependent disagreement between generative models. For each compared model pair, PromptSplit constructs a joint prompt--output representation by forming tensor-product embeddings of the prompt and image (or text) features, and then computes the corresponding kernel covariance matrix. We utilize the eigenspace of the weighted difference between these matrices to identify the main directions of behavioral difference across prompts. To ensure scalability, we employ a random-projection approximation that reduces computational complexity to $O(nr^2 + r^3)$ for projection dimension $r$. We further provide a theoretical analysis showing that this approximation yields an eigenstructure estimate whose expected deviation from the full-dimensional result is bounded by $O(1/r^2)$. Experiments across text-to-image, text-to-text, and image-captioning settings demonstrate that PromptSplit accurately detects ground-truth behavioral differences and isolates the prompts responsible, offering an interpretable tool for detecting where generative models disagree.
>
---
#### [replaced 024] MATTER: Multiscale Attention for Registration Error Regression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.12924v3](https://arxiv.org/pdf/2509.12924v3)**

> **作者:** Shipeng Liu; Ziliang Xiong; Khac-Hoang Ngo; Per-Erik Forssén
>
> **摘要:** Point cloud registration (PCR) is crucial for many downstream tasks, such as simultaneous localization and mapping (SLAM) and object tracking. This makes detecting and quantifying registration misalignment, i.e., PCR quality validation, an important task. All existing methods treat validation as a classification task, aiming to assign the PCR quality to a few classes. In this work, we instead use regression for PCR validation, allowing for a more fine-grained quantification of the registration quality. We also extend previously used misalignment-related features by using multiscale extraction and attention-based aggregation. This leads to accurate and robust registration error estimation on diverse datasets, especially for point clouds with heterogeneous spatial densities. Furthermore, when used to guide a mapping downstream task, our method significantly improves the mapping quality for a given amount of re-registered frames, compared to the state-of-the-art classification-based method.
>
---
#### [replaced 025] Investigating Disability Representations in Text-to-Image Models
- **分类: cs.CL; cs.CV; cs.CY; cs.HC**

- **简介: 该论文属于AI生成图像的伦理研究任务，旨在解决残疾群体在文本到图像模型中的代表性问题。通过分析不同提示下的图像，评估模型的偏见并提出改进策略。**

- **链接: [https://arxiv.org/pdf/2602.04687v2](https://arxiv.org/pdf/2602.04687v2)**

> **作者:** Yang Yian; Yu Fan; Liudmila Zavolokina; Sarah Ebling
>
> **备注:** 21 pages, 9 figures. References included
>
> **摘要:** Text-to-image generative models have made remarkable progress in producing high-quality visual content from textual descriptions, yet concerns remain about how they represent social groups. While characteristics like gender and race have received increasing attention, disability representations remain underexplored. This study investigates how people with disabilities are represented in AI-generated images by analyzing outputs from Stable Diffusion XL and DALL-E 3 using a structured prompt design. We analyze disability representations by comparing image similarities between generic disability prompts and prompts referring to specific disability categories. Moreover, we evaluate how mitigation strategies influence disability portrayals, with a focus on assessing affective framing through sentiment polarity analysis, combining both automatic and human evaluation. Our findings reveal persistent representational imbalances and highlight the need for continuous evaluation and refinement of generative models to foster more diverse and inclusive portrayals of disability.
>
---
#### [replaced 026] AR as an Evaluation Playground: Bridging Metrics and Visual Perception of Computer Vision Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04102v2](https://arxiv.org/pdf/2508.04102v2)**

> **作者:** Ashkan Ganj; Yiqin Zhao; Tian Guo
>
> **备注:** Accepted at MMSys 2026
>
> **摘要:** Quantitative metrics are central to evaluating computer vision (CV) models, but they often fail to capture real-world performance due to protocol inconsistencies and ground-truth noise. While visual perception studies can complement these metrics, they often require end-to-end systems that are time-consuming to implement and setups that are difficult to reproduce. We systematically summarize key challenges in evaluating CV models and present the design of ARCADE, an evaluation platform that leverages augmented reality (AR) to enable easy, reproducible, and human-centered CV evaluation. ARCADE uses a modular architecture that provides cross-platform data collection, pluggable model inference, and interactive AR tasks, supporting both metric and visual perception evaluation. We demonstrate ARCADE through a user study with 15 participants and case studies on two representative CV tasks, depth and lighting estimation, showing that ARCADE can reveal perceptual flaws in model quality that are often missed by traditional metrics. We also evaluate ARCADE's usability and performance, showing its flexibility as a reliable real-time platform.
>
---
#### [replaced 027] High-Precision Edge Detection via Task-Adaptive Texture Handling and Ideal-Prior Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.19992v5](https://arxiv.org/pdf/2407.19992v5)**

> **作者:** Hao Shu
>
> **备注:** 30 pages
>
> **摘要:** Image edge detection (ED) requires specialized architectures, reliable supervision, and rigorous evaluation criteria to ensure accurate localization. In this work, we present a framework for high-precision ED that jointly addresses architectural design, data supervision, and evaluation consistency. We propose SDPED, a compact ED model built upon Cascaded Skipping Density Blocks (CSDB), motivated by a task-adaptive architectural transfer from image super-resolution. By re-engineering texture-oriented structures for ED, SDPED effectively differentiates textures from edges while preserving fine spatial precision. Extensive experiments on four benchmark datasets (BRIND, UDED, MDBD, and BIPED2) demonstrate consistent performance improvements, particularly in Average Precision (AP), with gains of up to 22.5% on MDBD and 11.8% on BIPED2. In addition, we introduce an ideal-prior guidance strategy that incorporates noiseless data into training by treating labels as noise-free samples, providing a practical means to mitigate the subjectivity and noise inherent in human annotations. To enable fair and resolution-independent evaluation, we further adopt a fixed-pixel criterion for assessing localization accuracy. Overall, this work offers a coherent solution for high-precision ED and provides insights applicable to precision-oriented modeling in low-level and soft-computing-based vision tasks. Codes can be found on https://github.com/Hao-B-Shu/SDPED.
>
---
#### [replaced 028] Multi-Sensor Attention Networks for Automated Subsurface Delamination Detection in Concrete Bridge Decks
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2512.20113v3](https://arxiv.org/pdf/2512.20113v3)**

> **作者:** Alireza Moayedikia; Amirhossein Moayedikia
>
> **摘要:** Subsurface delaminations in concrete bridge decks remain undetectable through conventional visual inspection, necessitating automated non-destructive evaluation methods. This work introduces a deep learning framework that integrates Ground Penetrating Radar (GPR) and Infrared Thermography (IRT) through hierarchical attention mechanisms. Our architecture employs temporal self-attention to process GPR electromagnetic signals, spatial attention to analyze thermal imagery, and cross-modal attention with learnable embeddings to model inter-sensor correspondences. We integrate Monte Carlo dropout-based uncertainty quantification, decomposing prediction confidence into model uncertainty and data-driven uncertainty components. Testing across five real-world bridge datasets from the SDNET2021 benchmark reveals that our approach delivers substantial performance gains over single-sensor and concatenation-based baselines when applied to balanced or moderately imbalanced data distributions. Comprehensive ablation analysis confirms that cross-modal attention mechanisms contribute meaningful improvements beyond unimodal attention alone. Critically, we identify and characterize specific failure modes: under extreme class imbalance, attention-based architectures demonstrate susceptibility to majority class bias, indicating scenarios where simpler architectural choices may prove more robust. Our findings equip practitioners with empirically-grounded criteria for selecting appropriate fusion strategies based on dataset characteristics, rather than promoting universal architectural superiority.
>
---
#### [replaced 029] Extreme Weather Nowcasting via Local Precipitation Pattern Prediction
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05204v2](https://arxiv.org/pdf/2602.05204v2)**

> **作者:** Changhoon Song; Teng Yuan Chang; Youngjoon Hong
>
> **备注:** 10pages, 20 figures, The Fourteenth International Conference on Learning Representations, see https://github.com/tony890048/exPreCast
>
> **摘要:** Accurate forecasting of extreme weather events such as heavy rainfall or storms is critical for risk management and disaster mitigation. Although high-resolution radar observations have spurred extensive research on nowcasting models, precipitation nowcasting remains particularly challenging due to pronounced spatial locality, intricate fine-scale rainfall structures, and variability in forecasting horizons. While recent diffusion-based generative ensembles show promising results, they are computationally expensive and unsuitable for real-time applications. In contrast, deterministic models are computationally efficient but remain biased toward normal rainfall. Furthermore, the benchmark datasets commonly used in prior studies are themselves skewed--either dominated by ordinary rainfall events or restricted to extreme rainfall episodes--thereby hindering general applicability in real-world settings. In this paper, we propose exPreCast, an efficient deterministic framework for generating finely detailed radar forecasts, and introduce a newly constructed balanced radar dataset from the Korea Meteorological Administration (KMA), which encompasses both ordinary precipitation and extreme events. Our model integrates local spatiotemporal attention, a texture-preserving cubic dual upsampling decoder, and a temporal extractor to flexibly adjust forecasting horizons. Experiments on established benchmarks (SEVIR and MeteoNet) as well as on the balanced KMA dataset demonstrate that our approach achieves state-of-the-art performance, delivering accurate and reliable nowcasts across both normal and extreme rainfall regimes.
>
---
#### [replaced 030] Enhancing Features in Long-tailed Data Using Large Vision Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.10852v3](https://arxiv.org/pdf/2504.10852v3)**

> **作者:** Pengxiao Han; Changkun Ye; Jinguang Tong; Cuicui Jiang; Jie Hong; Li Fang; Xuesong Li
>
> **摘要:** Language-based foundation models, such as large language models (LLMs) or large vision-language models (LVLMs), have been widely studied in long-tailed recognition. However, the need for linguistic data is not applicable to all practical tasks. In this study, we aim to explore using large vision models (LVMs) or visual foundation models (VFMs) to enhance long-tailed data features without any language information. Specifically, we extract features from the LVM and fuse them with features in the baseline network's map and latent space to obtain the augmented features. Moreover, we design several prototype-based losses in the latent space to further exploit the potential of the augmented features. In the experimental section, we validate our approach on two benchmark datasets: ImageNet-LT and iNaturalist2018.
>
---
#### [replaced 031] EchoJEPA: A Latent Predictive Foundation Model for Echocardiography
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02603v3](https://arxiv.org/pdf/2602.02603v3)**

> **作者:** Alif Munim; Adibvafa Fallahpour; Teodora Szasz; Ahmadreza Attarpour; River Jiang; Brana Sooriyakanthan; Maala Sooriyakanthan; Heather Whitney; Jeremy Slivnick; Barry Rubin; Wendy Tsang; Bo Wang
>
> **摘要:** Foundation models for echocardiography often struggle to disentangle anatomical signal from the stochastic speckle and acquisition artifacts inherent to ultrasound. We present EchoJEPA, a foundation model trained on 18 million echocardiograms across 300K patients, representing the largest pretraining corpus for this modality to date. By leveraging a latent predictive objective, EchoJEPA learns robust anatomical representations that ignore speckle noise. We validate this using a novel multi-view probing framework with frozen backbones, where EchoJEPA outperforms leading baselines by approximately 20% in left ventricular ejection fraction (LVEF) estimation and 17% in right ventricular systolic pressure (RVSP) estimation. The model also exhibits remarkable sample efficiency, reaching 79% view classification accuracy with only 1% of labeled data versus 42% for the best baseline trained on 100%. Crucially, EchoJEPA demonstrates superior generalization, degrading by only 2% under physics-informed acoustic perturbations compared to 17% for competitors. Most remarkably, its zero-shot performance on pediatric patients surpasses fully fine-tuned baselines, establishing latent prediction as a superior paradigm for robust, generalizable medical AI.
>
---
#### [replaced 032] An Example for Domain Adaptation Using CycleGAN
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.08776v3](https://arxiv.org/pdf/2601.08776v3)**

> **作者:** Yanhua Zhao
>
> **备注:** 3 pages, 2 figures
>
> **摘要:** Cycle-Consistent Adversarial Network (CycleGAN) is very promising in domain adaptation. In this report, an example in medical domain will be explained. We present struecture of a CycleGAN model for unpaired image-to-image translation from microscopy to pseudo H\&E stained histopathology images.
>
---
#### [replaced 033] Visual Autoregressive Modeling for Instruction-Guided Image Editing
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2508.15772v2](https://arxiv.org/pdf/2508.15772v2)**

> **作者:** Qingyang Mao; Qi Cai; Yehao Li; Yingwei Pan; Mingyue Cheng; Ting Yao; Qi Liu; Tao Mei
>
> **备注:** ICLR 2026; Source codes and models are available at https://github.com/HiDream-ai/VAREdit
>
> **摘要:** Recent advances in diffusion models have brought remarkable visual fidelity to instruction-guided image editing. However, their global denoising process inherently entangles the edited region with the entire image context, leading to unintended spurious modifications and compromised adherence to editing instructions. In contrast, autoregressive models offer a distinct paradigm by formulating image synthesis as a sequential process over discrete visual tokens. Their causal and compositional mechanism naturally circumvents the adherence challenges of diffusion-based methods. In this paper, we present VAREdit, a visual autoregressive (VAR) framework that reframes image editing as a next-scale prediction problem. Conditioned on source image features and text instructions, VAREdit generates multi-scale target features to achieve precise edits. A core challenge in this paradigm is how to effectively condition the source image tokens. We observe that finest-scale source features cannot effectively guide the prediction of coarser target features. To bridge this gap, we introduce a Scale-Aligned Reference (SAR) module, which injects scale-matched conditioning information into the first self-attention layer. VAREdit demonstrates significant advancements in both editing adherence and efficiency. On EMU-Edit and PIE-Bench benchmarks, VAREdit outperforms leading diffusion-based methods by a substantial margin in terms of both CLIP and GPT scores. Moreover, VAREdit completes a 512$\times$512 editing in 1.2 seconds, making it 2.2$\times$ faster than the similarly sized UltraEdit. Code is available at: https://github.com/HiDream-ai/VAREdit.
>
---
#### [replaced 034] DRMOT: A Dataset and Framework for RGBD Referring Multi-Object Tracking
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.04692v2](https://arxiv.org/pdf/2602.04692v2)**

> **作者:** Sijia Chen; Lijuan Ma; Yanqiu Yu; En Yu; Liman Liu; Wenbing Tao
>
> **备注:** https://github.com/chen-si-jia/DRMOT
>
> **摘要:** Referring Multi-Object Tracking (RMOT) aims to track specific targets based on language descriptions and is vital for interactive AI systems such as robotics and autonomous driving. However, existing RMOT models rely solely on 2D RGB data, making it challenging to accurately detect and associate targets characterized by complex spatial semantics (e.g., ``the person closest to the camera'') and to maintain reliable identities under severe occlusion, due to the absence of explicit 3D spatial information. In this work, we propose a novel task, RGBD Referring Multi-Object Tracking (DRMOT), which explicitly requires models to fuse RGB, Depth (D), and Language (L) modalities to achieve 3D-aware tracking. To advance research on the DRMOT task, we construct a tailored RGBD referring multi-object tracking dataset, named DRSet, designed to evaluate models' spatial-semantic grounding and tracking capabilities. Specifically, DRSet contains RGB images and depth maps from 187 scenes, along with 240 language descriptions, among which 56 descriptions incorporate depth-related information. Furthermore, we propose DRTrack, a MLLM-guided depth-referring tracking framework. DRTrack performs depth-aware target grounding from joint RGB-D-L inputs and enforces robust trajectory association by incorporating depth cues. Extensive experiments on the DRSet dataset demonstrate the effectiveness of our framework.
>
---
#### [replaced 035] HSG-12M: A Large-Scale Benchmark of Spatial Multigraphs from the Energy Spectra of Non-Hermitian Crystals
- **分类: cs.LG; cond-mat.mes-hall; cond-mat.other; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.08618v2](https://arxiv.org/pdf/2506.08618v2)**

> **作者:** Xianquan Yan; Hakan Akgün; Kenji Kawaguchi; N. Duane Loh; Ching Hua Lee
>
> **备注:** 48 pages, 13 figures, 14 tables. Code & pipeline: [https://github.com/sarinstein-yan/Poly2Graph] Dataset: [https://github.com/sarinstein-yan/HSG-12M] Dataset released under CC BY 4.0. Benchmark scripts and data loaders included
>
> **摘要:** AI is transforming scientific research by revealing new ways to understand complex physical systems, but its impact remains constrained by the lack of large, high-quality domain-specific datasets. A rich, largely untapped resource lies in non-Hermitian quantum physics, where the energy spectra of crystals form intricate geometries on the complex plane -- termed as Hamiltonian spectral graphs. Despite their significance as fingerprints for electronic behavior, their systematic study has been intractable due to the reliance on manual extraction. To unlock this potential, we introduce Poly2Graph: a high-performance, open-source pipeline that automates the mapping of 1-D crystal Hamiltonians to spectral graphs. Using this tool, we present HSG-12M: a dataset containing 11.6 million static and 5.1 million dynamic Hamiltonian spectral graphs across 1401 characteristic-polynomial classes, distilled from 177 TB of spectral potential data. Crucially, HSG-12M is the first large-scale dataset of spatial multigraphs -- graphs embedded in a metric space where multiple geometrically distinct trajectories between two nodes are retained as separate edges. This simultaneously addresses a critical gap, as existing graph benchmarks overwhelmingly assume simple, non-spatial edges, discarding vital geometric information. Benchmarks with popular GNNs expose new challenges in learning spatial multi-edges at scale. Beyond its practical utility, we show that spectral graphs serve as universal topological fingerprints of polynomials, vectors, and matrices, forging a new algebra-to-graph link. HSG-12M lays the groundwork for data-driven scientific discovery in condensed matter physics, new opportunities in geometry-aware graph learning and beyond.
>
---
#### [replaced 036] Anonymization Prompt Learning for Facial Privacy-Preserving Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.16895v3](https://arxiv.org/pdf/2405.16895v3)**

> **作者:** Liang Shi; Jie Zhang; Shiguang Shan
>
> **备注:** Accepted by IJCV
>
> **摘要:** Text-to-image diffusion models, such as Stable Diffusion, generate highly realistic images from text descriptions. However, the generation of certain content at such high quality raises concerns. A prominent issue is the accurate depiction of identifiable facial images, which could lead to malicious deepfake generation and privacy violations. In this paper, we propose Anonymization Prompt Learning (APL) to address this problem. Specifically, we train a learnable prompt prefix for text-to-image diffusion models, which forces the model to generate anonymized facial identities, even when prompted to produce images of specific individuals. Extensive quantitative and qualitative experiments demonstrate the successful anonymization performance of APL, which anonymizes any specific individuals without compromising the quality of non-identity-specific image generation. Furthermore, we reveal the plug-and-play property of the learned prompt prefix, enabling its effective application across different pretrained text-to-image models for transferrable privacy and security protection against the risks of deepfakes.
>
---
#### [replaced 037] Generalization of Self-Supervised Vision Transformers for Protein Localization Across Microscopy Domains
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05527v2](https://arxiv.org/pdf/2602.05527v2)**

> **作者:** Ben Isselmann; Dilara Göksu; Andreas Weinmann
>
> **备注:** Preprint; not yet peer reviewed. AMEE Conference Proceeding 2025, 11 pages, 2 figures
>
> **摘要:** Task-specific microscopy datasets are often too small to train deep learning models that learn robust feature representations. Self-supervised learning (SSL) can mitigate this by pretraining on large unlabeled datasets, but it remains unclear how well such representations transfer across microscopy domains with different staining protocols and channel configurations. We investigate the cross-domain transferability of DINO-pretrained Vision Transformers for protein localization on the OpenCell dataset. We generate image embeddings using three DINO backbones pretrained on ImageNet-1k, the Human Protein Atlas (HPA), and OpenCell, and evaluate them by training a supervised classification head on OpenCell labels. All pretrained models transfer well, with the microscopy-specific HPA-pretrained model achieving the best performance (mean macro $F_1$-score = 0.8221 $\pm$ 0.0062), slightly outperforming a DINO model trained directly on OpenCell (0.8057 $\pm$ 0.0090). These results highlight the value of large-scale pretraining and indicate that domain-relevant SSL representations can generalize effectively to related but distinct microscopy datasets, enabling strong downstream performance even when task-specific labeled data are limited.
>
---
#### [replaced 038] An Evaluation of Hybrid Annotation Workflows on High-Ambiguity Spatiotemporal Video Footage
- **分类: cs.CV; cs.HC; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.21798v2](https://arxiv.org/pdf/2510.21798v2)**

> **作者:** Juan Gutiérrez; Victor Gutiérrez; Ángel Mora; Silvia Rodriguez; José Luis Blanco
>
> **摘要:** Manual annotation remains the gold standard for high-quality, dense temporal video datasets, yet it is inherently time-consuming. Vision-language models can aid human annotators and expedite this process. We report on the impact of automatic Pre-Annotations from a tuned encoder on a Human-in-the-Loop labeling workflow for video footage. Quantitative analysis in a study of a single-iteration test involving 18 volunteers demonstrates that our workflow reduced annotation time by 35% for the majority (72%) of the participants. Beyond efficiency, we provide a rigorous framework for benchmarking AI-assisted workflows that quantifies trade-offs between algorithmic speed and the integrity of human verification.
>
---
#### [replaced 039] Sketch2Scene: Automatic Generation of Interactive 3D Game Scenes from User's Casual Sketches
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2408.04567v2](https://arxiv.org/pdf/2408.04567v2)**

> **作者:** Yongzhi Xu; Yonhon Ng; Yifu Wang; Inkyu Sa; Yunfei Duan; Zhenhong Sun; Yang Li; Pan Ji; Hongdong Li
>
> **备注:** Project Page: https://xrvisionlabs.github.io/Sketch2Scene/ Code: https://github.com/Tencent/Triplet_Tuning
>
> **摘要:** 3D Content Generation is at the heart of many computer graphics applications, including video gaming, film-making, virtual and augmented reality, etc. This paper proposes a novel deep-learning based approach for automatically generating interactive and playable 3D game scenes, all from the user's casual prompts such as a hand-drawn sketch. Sketch-based input offers a natural, and convenient way to convey the user's design intention in the content creation process. To circumvent the data-deficient challenge in learning (i.e. the lack of large training data of 3D scenes), our method leverages a pre-trained 2D denoising diffusion model to generate a 2D image of the scene as the conceptual guidance. In this process, we adopt the isometric projection mode to factor out unknown camera poses while obtaining the scene layout. From the generated isometric image, we use a pre-trained image understanding method to segment the image into meaningful parts, such as off-ground objects, trees, and buildings, and extract the 2D scene layout. These segments and layouts are subsequently fed into a procedural content generation (PCG) engine, such as a 3D video game engine like Unity or Unreal, to create the 3D scene. The resulting 3D scene can be seamlessly integrated into a game development environment and is readily playable. Extensive tests demonstrate that our method can efficiently generate high-quality and interactive 3D game scenes with layouts that closely follow the user's intention.
>
---
#### [replaced 040] MRD: Using Physically Based Differentiable Rendering to Probe Vision Models for 3D Scene Understanding
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2512.12307v3](https://arxiv.org/pdf/2512.12307v3)**

> **作者:** Benjamin Beilharz; Thomas S. A. Wallis
>
> **备注:** 23 pages, 11 figures. Added appendix with more figure results. Code will be available here: https://github.com/ag-perception-wallis-lab/MRD
>
> **摘要:** While deep learning methods have achieved impressive success in many vision benchmarks, it remains difficult to understand and explain the representations and decisions of these models. Though vision models are typically trained on 2D inputs, they are often assumed to develop an implicit representation of the underlying 3D scene (for example, showing tolerance to partial occlusion, or the ability to reason about relative depth). Here, we introduce MRD (metamers rendered differentiably), an approach that uses physically based differentiable rendering to probe vision models' implicit understanding of generative 3D scene properties, by finding 3D scene parameters that are physically different but produce the same model activation (i.e. are model metamers). Unlike previous pixel-based methods for evaluating model representations, these reconstruction results are always grounded in physical scene descriptions. This means we can, for example, probe a model's sensitivity to object shape while holding material and lighting constant. As a proof-of-principle, we assess multiple models in their ability to recover scene parameters of geometry (shape) and bidirectional reflectance distribution function (material). The results show high similarity in model activation between target and optimized scenes, with varying visual results. Qualitatively, these reconstructions help investigate the physical scene attributes to which models are sensitive or invariant. MRD holds promise for advancing our understanding of both computer and human vision by enabling analysis of how physical scene parameters drive changes in model responses.
>
---
#### [replaced 041] Dataset Distillation as Pushforward Optimal Quantization
- **分类: cs.LG; cs.CV; math.OC; stat.ML**

- **链接: [https://arxiv.org/pdf/2501.07681v3](https://arxiv.org/pdf/2501.07681v3)**

> **作者:** Hong Ye Tan; Emma Slade
>
> **备注:** ICLR 2026, https://openreview.net/forum?id=FMSp8AUF3m
>
> **摘要:** Dataset distillation aims to find a synthetic training set such that training on the synthetic data achieves similar performance to training on real data, with orders of magnitude less computational requirements. Existing methods can be broadly categorized as either bi-level optimization problems that have neural network training heuristics as the lower level problem, or disentangled methods that bypass the bi-level optimization by matching distributions of data. The latter method has the major advantages of speed and scalability in terms of size of both training and distilled datasets. We demonstrate that when equipped with an encoder-decoder structure, the empirically successful disentangled methods can be reformulated as an optimal quantization problem, where a finite set of points is found to approximate the underlying probability measure by minimizing the expected projection distance. In particular, we link existing disentangled dataset distillation methods to the classical optimal quantization and Wasserstein barycenter problems, demonstrating consistency of distilled datasets for diffusion-based generative priors. We propose Dataset Distillation by Optimal Quantization, based on clustering in a latent space. Compared to the previous SOTA method D\textsuperscript{4}M, we achieve better performance and inter-model generalization on the ImageNet-1K dataset with trivial additional computation, and SOTA performance in higher image-per-class settings. Using the distilled noise initializations in a stronger diffusion transformer model, we obtain SOTA distillation performance on ImageNet-1K and its subsets, outperforming diffusion guidance methods.
>
---
#### [replaced 042] Predicting Camera Pose from Perspective Descriptions for Spatial Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.06041v2](https://arxiv.org/pdf/2602.06041v2)**

> **作者:** Xuejun Zhang; Aditi Tiwari; Zhenhailong Wang; Heng Ji
>
> **摘要:** Multi-image spatial reasoning remains challenging for current multimodal large language models (MLLMs). While single-view perception is inherently 2D, reasoning over multiple views requires building a coherent scene understanding across viewpoints. In particular, we study perspective taking, where a model must build a coherent 3D understanding from multi-view observations and use it to reason from a new, language-specified viewpoint. We introduce CAMCUE, a pose-aware multi-image framework that uses camera pose as an explicit geometric anchor for cross-view fusion and novel-view reasoning. CAMCUE injects per-view pose into visual tokens, grounds natural-language viewpoint descriptions to a target camera pose, and synthesizes a pose-conditioned imagined target view to support answering. To support this setting, we curate CAMCUE-DATA with 27,668 training and 508 test instances pairing multi-view images and poses with diverse target-viewpoint descriptions and perspective-shift questions. We also include human-annotated viewpoint descriptions in the test split to evaluate generalization to human language. CAMCUE improves overall accuracy by 9.06% and predicts target poses from natural-language viewpoint descriptions with over 90% rotation accuracy within 20° and translation accuracy within a 0.5 error threshold. This direct grounding avoids expensive test-time search-and-match, reducing inference time from 256.6s to 1.45s per example and enabling fast, interactive use in real-world scenarios.
>
---
#### [replaced 043] Continual-MEGA: A Large-scale Benchmark for Generalizable Continual Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.00956v3](https://arxiv.org/pdf/2506.00956v3)**

> **作者:** Geonu Lee; Yujeong Oh; Geonhui Jang; Soyoung Lee; Jeonghyo Song; Sungmin Cha; YoungJoon Yoo
>
> **摘要:** In this paper, we introduce a new benchmark for continual learning in anomaly detection, aimed at better reflecting real-world deployment scenarios. Our benchmark, Continual-MEGA, includes a large and diverse dataset that significantly expands existing evaluation settings by combining carefully curated existing datasets with our newly proposed dataset, ContinualAD. In addition to standard continual learning with expanded quantity, we propose a novel scenario that measures zero-shot generalization to unseen classes, those not observed during continual adaptation. This setting poses a new problem setting that continual adaptation also enhances zero-shot performance. We also present a unified baseline algorithm that improves robustness in few-shot detection and maintains strong generalization. Through extensive evaluations, we report three key findings: (1) existing methods show substantial room for improvement, particularly in pixel-level defect localization; (2) our proposed method consistently outperforms prior approaches; and (3) the newly introduced ContinualAD dataset enhances the performance of strong anomaly detection models. We release the benchmark and code in https://github.com/Continual-Mega/Continual-Mega.
>
---
#### [replaced 044] Learning a distance measure from the information-estimation geometry of data
- **分类: eess.IV; cs.CV; cs.IT; eess.SP; stat.ML**

- **链接: [https://arxiv.org/pdf/2510.02514v2](https://arxiv.org/pdf/2510.02514v2)**

> **作者:** Guy Ohayon; Pierre-Etienne H. Fiquet; Florentin Guth; Jona Ballé; Eero P. Simoncelli
>
> **备注:** ICLR 2026. Code is available at https://github.com/ohayonguy/information-estimation-metric
>
> **摘要:** We introduce the Information-Estimation Metric (IEM), a novel form of distance function derived from an underlying continuous probability density over a domain of signals. The IEM is rooted in a fundamental relationship between information theory and estimation theory, which links the log-probability of a signal with the errors of an optimal denoiser, applied to noisy observations of the signal. In particular, the IEM between a pair of signals is obtained by comparing their denoising error vectors over a range of noise amplitudes. Geometrically, this amounts to comparing the score vector fields of the blurred density around the signals over a range of blur levels. We prove that the IEM is a valid global distance metric and derive a closed-form expression for its local second-order approximation, which yields a Riemannian metric. For Gaussian-distributed signals, the IEM coincides with the Mahalanobis distance. But for more complex distributions, it adapts, both locally and globally, to the geometry of the distribution. In practice, the IEM can be computed using a learned denoiser (analogous to generative diffusion models) and solving a one-dimensional integral. To demonstrate the value of our framework, we learn an IEM on the ImageNet database. Experiments show that this IEM is competitive with or outperforms state-of-the-art supervised image quality metrics in predicting human perceptual judgments.
>
---
#### [replaced 045] M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.10931v2](https://arxiv.org/pdf/2505.10931v2)**

> **作者:** Chao Wang; Wei Lu; Xiang Li; Jian Yang; Lei Luo
>
> **摘要:** Single-source remote sensing object detection using optical or SAR images struggles in complex environments. Optical images offer rich textural details but are often affected by low-light, cloud-obscured, or low-resolution conditions, reducing the detection performance. SAR images are robust to weather, but suffer from speckle noise and limited semantic expressiveness. Optical and SAR images provide complementary advantages, and fusing them can significantly improve the detection accuracy. However, progress in this field is hindered by the lack of large-scale, standardized datasets. To address these challenges, we propose the first comprehensive dataset for optical-SAR fusion object detection, named Multi-resolution, Multi-polarization, Multi-scene, Multi-source SAR dataset (M4-SAR). It contains 112,184 precisely aligned image pairs and nearly one million labeled instances with arbitrary orientations, spanning six key categories. To enable standardized evaluation, we develop a unified benchmarking toolkit that integrates six state-of-the-art multi-source fusion methods. Furthermore, we propose E2E-OSDet, a novel end-to-end multi-source fusion detection framework that mitigates cross-domain discrepancies and establishes a robust baseline for future studies. Extensive experiments on M4-SAR demonstrate that fusing optical and SAR data can improve $mAP$ by 5.7\% over single-source inputs, with particularly significant gains in complex environments. The dataset and code are publicly available at https://github.com/wchao0601/M4-SAR.
>
---
#### [replaced 046] SPIDER: Scalable Physics-Informed Dexterous Retargeting
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SPIDER框架，解决机器人控制中数据稀缺与人类演示转换的问题。通过物理引导的重定向方法，将人类运动数据转化为动态可行的机器人轨迹，提升任务成功率并加速策略学习。**

- **链接: [https://arxiv.org/pdf/2511.09484v2](https://arxiv.org/pdf/2511.09484v2)**

> **作者:** Chaoyi Pan; Changhao Wang; Haozhi Qi; Zixi Liu; Homanga Bharadhwaj; Akash Sharma; Tingfan Wu; Guanya Shi; Jitendra Malik; Francois Hogan
>
> **备注:** Project website: https://jc-bao.github.io/spider-project/
>
> **摘要:** Learning dexterous and agile policy for humanoid and dexterous hand control requires large-scale demonstrations, but collecting robot-specific data is prohibitively expensive. In contrast, abundant human motion data is readily available from motion capture, videos, and virtual reality, which could help address the data scarcity problem. However, due to the embodiment gap and missing dynamic information like force and torque, these demonstrations cannot be directly executed on robots. To bridge this gap, we propose Scalable Physics-Informed DExterous Retargeting (SPIDER), a physics-based retargeting framework to transform and augment kinematic-only human demonstrations to dynamically feasible robot trajectories at scale. Our key insight is that human demonstrations should provide global task structure and objective, while large-scale physics-based sampling with curriculum-style virtual contact guidance should refine trajectories to ensure dynamical feasibility and correct contact sequences. SPIDER scales across diverse 9 humanoid/dexterous hand embodiments and 6 datasets, improving success rates by 18% compared to standard sampling, while being 10X faster than reinforcement learning (RL) baselines, and enabling the generation of a 2.4M frames dynamic-feasible robot dataset for policy learning. As a universal physics-based retargeting method, SPIDER can work with diverse quality data and generate diverse and high-quality data to enable efficient policy learning with methods like RL.
>
---
#### [replaced 047] SPARK: Scalable Real-Time Point Cloud Aggregation with Multi-View Self-Calibration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.08414v3](https://arxiv.org/pdf/2601.08414v3)**

> **作者:** Chentian Sun
>
> **备注:** 10 pages, 1 figure, submitted to IEEE Transactions on Image Processing (TIP). Version 3: Minor revision; several experimental results have been removed and supplemented after further verification
>
> **摘要:** Real-time multi-camera 3D reconstruction is crucial for 3D perception, immersive interaction, and robotics. Existing methods struggle with multi-view fusion, camera extrinsic uncertainty, and scalability for large camera setups. We propose SPARK, a self-calibrating real-time multi-camera point cloud reconstruction framework that jointly handles point cloud fusion and extrinsic uncertainty. SPARK consists of: (1) a geometry-aware online extrinsic estimation module leveraging multi-view priors and enforcing cross-view and temporal consistency for stable self-calibration, and (2) a confidence-driven point cloud fusion strategy modeling depth reliability and visibility at pixel and point levels to suppress noise and view-dependent inconsistencies. By performing frame-wise fusion without accumulation, SPARK produces stable point clouds in dynamic scenes while scaling linearly with the number of cameras. Extensive experiments on real-world multi-camera systems show that SPARK outperforms existing approaches in extrinsic accuracy, geometric consistency, temporal stability, and real-time performance, demonstrating its effectiveness and scalability for large-scale multi-camera 3D reconstruction.
>
---
#### [replaced 048] SoliReward: Mitigating Susceptibility to Reward Hacking and Annotation Noise in Video Generation Reward Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.22170v2](https://arxiv.org/pdf/2512.22170v2)**

> **作者:** Jiesong Lian; Ruizhe Zhong; Zixiang Zhou; Xiaoyue Mi; Yixue Hao; Yuan Zhou; Qinglin Lu; Long Hu; Junchi Yan
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Post-training alignment of video generation models with human preferences is a critical goal. Developing effective Reward Models (RMs) for this process faces significant methodological hurdles. Current data collection paradigms, reliant on in-prompt pairwise annotations, suffer from labeling noise. Concurrently, the architectural design of VLM-based RMs, particularly their output mechanisms, remains underexplored. Furthermore, RM is susceptible to reward hacking in post-training. To mitigate these limitations, we propose SoliReward, a systematic framework for video RM training. Our framework first sources high-quality, cost-efficient data via single-item binary annotations, then constructs preference pairs using a cross-prompt pairing strategy. Architecturally, we employ a Hierarchical Progressive Query Attention mechanism to enhance feature aggregation. Finally, we introduce a modified BT loss that explicitly accommodates win-tie scenarios. This approach regularizes the RM's score distribution for positive samples, providing more nuanced preference signals to alleviate over-focus on a small number of top-scoring samples. Our approach is validated on benchmarks evaluating physical plausibility, subject deformity, and semantic alignment, demonstrating improvements in direct RM evaluation metrics and in the efficacy of post-training on video generation models. Code and benchmark are available at https://github.com/lian700/SoliReward
>
---
#### [replaced 049] FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于生成模型任务，解决长上下文生成中的计算效率问题。通过引入FlashBlock机制，减少注意力计算冗余，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2602.05305v2](https://arxiv.org/pdf/2602.05305v2)**

> **作者:** Zhuokun Chen; Jianfei Cai; Bohan Zhuang
>
> **摘要:** Generating long-form content, such as minute-long videos and extended texts, is increasingly important for modern generative models. Block diffusion improves inference efficiency via KV caching and block-wise causal inference and has been widely adopted in diffusion language models and video generation. However, in long-context settings, block diffusion still incurs substantial overhead from repeatedly computing attention over a growing KV cache. We identify an underexplored property of block diffusion: cross-step redundancy of attention within a block. Our analysis shows that attention outputs from tokens outside the current block remain largely stable across diffusion steps, while block-internal attention varies significantly. Based on this observation, we propose FlashBlock, a cached block-external attention mechanism that reuses stable attention output, reducing attention computation and KV cache access without modifying the diffusion process. Moreover, FlashBlock is orthogonal to sparse attention and can be combined as a complementary residual reuse strategy, substantially improving model accuracy under aggressive sparsification. Experiments on diffusion language models and video generation demonstrate up to 1.44$\times$ higher token throughput and up to 1.6$\times$ reduction in attention time, with negligible impact on generation quality. Project page: https://caesarhhh.github.io/FlashBlock/.
>
---
#### [replaced 050] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.05449v2](https://arxiv.org/pdf/2602.05449v2)**

> **作者:** Chang Zou; Changlin Li; Yang Li; Patrol Li; Jianbing Wu; Xiao He; Songtao Liu; Zhao Zhong; Kailin Huang; Linfeng Zhang
>
> **备注:** 17 pages, 7 figures; cvpr2026 submission
>
> **摘要:** While diffusion models have achieved great success in the field of video generation, this progress is accompanied by a rapidly escalating computational burden. Among the existing acceleration methods, Feature Caching is popular due to its training-free property and considerable speedup performance, but it inevitably faces semantic and detail drop with further compression. Another widely adopted method, training-aware step-distillation, though successful in image generation, also faces drastic degradation in video generation with a few steps. Furthermore, the quality loss becomes more severe when simply applying training-free feature caching to the step-distilled models, due to the sparser sampling steps. This paper novelly introduces a distillation-compatible learnable feature caching mechanism for the first time. We employ a lightweight learnable neural predictor instead of traditional training-free heuristics for diffusion models, enabling a more accurate capture of the high-dimensional feature evolution process. Furthermore, we explore the challenges of highly compressed distillation on large-scale video models and propose a conservative Restricted MeanFlow approach to achieve more stable and lossless distillation. By undertaking these initiatives, we further push the acceleration boundaries to $11.8\times$ while preserving generation quality. Extensive experiments demonstrate the effectiveness of our method. The code will be made publicly available soon.
>
---
#### [replaced 051] STAG: Structural Test-time Alignment of Gradients for Online Adaptation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2402.09004v2](https://arxiv.org/pdf/2402.09004v2)**

> **作者:** Juhyeon Shin; Yujin Oh; Jonghyun Lee; Saehyung Lee; Minjun Park; Dongjun Lee; Uiwon Hwang; Sungroh Yoon
>
> **摘要:** Test-Time Adaptation (TTA) adapts pre-trained models using only unlabeled test streams, requiring real-time inference and update without access to source data. We propose StructuralTest-time Alignment of Gradients (STAG), a lightweight plug-in enhancer that exploits an always-available structural signal: the classifier's intrinsic geometry. STAG derives class-wise structural anchors from classifier weights via self-structural entropy, and during adaptation analytically computes the predicted-class entropy gradient from forward-pass quantities, aligning it to the corresponding anchor with a cosine-similarity loss. This closed-form design incurs near-zero memory and latency overhead and requires no additional backpropagation beyond the underlying baseline. Across corrupted image classification and continual semantic segmentation, STAG provides broadly applicable performance gains for strong TTA baselines on both CNN and Transformer architectures regardless of the underlying normalization scheme, with particularly large gains under challenging online regimes such as imbalanced label shifts, single-sample adaptation, mixed corruption streams and long-horizon continual TTA.
>
---
#### [replaced 052] Refer-Agent: A Collaborative Multi-Agent System with Reasoning and Reflection for Referring Video Object Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03595v2](https://arxiv.org/pdf/2602.03595v2)**

> **作者:** Haichao Jiang; Tianming Liang; Wei-Shi Zheng; Jian-Fang Hu
>
> **摘要:** Referring Video Object Segmentation (RVOS) aims to segment objects in videos based on textual queries. Current methods mainly rely on large-scale supervised fine-tuning (SFT) of Multi-modal Large Language Models (MLLMs). However, this paradigm suffers from heavy data dependence and limited scalability against the rapid evolution of MLLMs. Although recent zero-shot approaches offer a flexible alternative, their performance remains significantly behind SFT-based methods, due to the straightforward workflow designs. To address these limitations, we propose \textbf{Refer-Agent}, a collaborative multi-agent system with alternating reasoning-reflection mechanisms. This system decomposes RVOS into step-by-step reasoning process. During reasoning, we introduce a Coarse-to-Fine frame selection strategy to ensure the frame diversity and textual relevance, along with a Dynamic Focus Layout that adaptively adjusts the agent's visual focus. Furthermore, we propose a Chain-of-Reflection mechanism, which employs a Questioner-Responder pair to generate a self-reflection chain, enabling the system to verify intermediate results and generates feedback for next-round reasoning refinement. Extensive experiments on five challenging benchmarks demonstrate that Refer-Agent significantly outperforms state-of-the-art methods, including both SFT-based models and zero-shot approaches. Moreover, Refer-Agent is flexible and enables fast integration of new MLLMs without any additional fine-tuning costs. Code will be released at https://github.com/iSEE-Laboratory/Refer-Agent.
>
---
#### [replaced 053] DarkEQA: Benchmarking Vision-Language Models for Embodied Question Answering in Low-Light Indoor Environments
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉语言模型在低光环境下的问答任务，旨在解决现有基准未覆盖低光条件的问题。工作包括构建DarkEQA基准，模拟真实低光场景，评估模型性能。**

- **链接: [https://arxiv.org/pdf/2512.24985v3](https://arxiv.org/pdf/2512.24985v3)**

> **作者:** Yohan Park; Hyunwoo Ha; Wonjun Jo; Tae-Hyun Oh
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Vision Language Models (VLMs) are increasingly adopted as central reasoning modules for embodied agents. Existing benchmarks evaluate their capabilities under ideal, well-lit conditions, yet robust 24/7 operation demands performance under a wide range of visual degradations, including low-light conditions at night or in dark environments--a core necessity that has been largely overlooked. To address this underexplored challenge, we present DarkEQA, an open-source benchmark for evaluating EQA-relevant perceptual primitives under multi-level low-light conditions. DarkEQA isolates the perception bottleneck by evaluating question answering from egocentric observations under controlled degradations, enabling attributable robustness analysis. A key design feature of DarkEQA is its physical fidelity: visual degradations are modeled in linear RAW space, simulating physics-based illumination drop and sensor noise followed by an ISP-inspired rendering pipeline. We demonstrate the utility of DarkEQA by evaluating a wide range of state-of-the-art VLMs and Low-Light Image Enhancement (LLIE) models. Our analysis systematically reveals VLMs' limitations when operating under these challenging visual conditions. Project website: https://darkeqa-benchmark.github.io/
>
---
#### [replaced 054] ConsisDrive: Identity-Preserving Driving World Models for Video Generation by Instance Mask
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03213v2](https://arxiv.org/pdf/2602.03213v2)**

> **作者:** Zhuoran Yang; Yanyong Zhang
>
> **摘要:** Autonomous driving relies on robust models trained on large-scale, high-quality multi-view driving videos. Although world models provide a cost-effective solution for generating realistic driving data, they often suffer from identity drift, where the same object changes its appearance or category across frames due to the absence of instance-level temporal constraints. We introduce ConsisDrive, an identity-preserving driving world model designed to enforce temporal consistency at the instance level. Our framework incorporates two key components: (1) Instance-Masked Attention, which applies instance identity masks and trajectory masks within attention blocks to ensure that visual tokens interact only with their corresponding instance features across spatial and temporal dimensions, thereby preserving object identity consistency; and (2) Instance-Masked Loss, which adaptively emphasizes foreground regions with probabilistic instance masking, reducing background noise while maintaining overall scene fidelity. By integrating these mechanisms, ConsisDrive achieves state-of-the-art driving video generation quality and demonstrates significant improvements in downstream autonomous driving tasks on the nuScenes dataset. Our project page is https://shanpoyang654.github.io/ConsisDrive/page.html.
>
---
#### [replaced 055] DiMo: Discrete Diffusion Modeling for Motion Generation and Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.04188v2](https://arxiv.org/pdf/2602.04188v2)**

> **作者:** Ning Zhang; Zhengyu Li; Kwong Weng Loh; Mingxi Xu; Qi Wang; Zhengyu Wen; Xiaoyu He; Wei Zhao; Kehong Gong; Mingyuan Zhang
>
> **摘要:** Prior masked modeling motion generation methods predominantly study text-to-motion. We present DiMo, a discrete diffusion-style framework, which extends masked modeling to bidirectional text--motion understanding and generation. Unlike GPT-style autoregressive approaches that tokenize motion and decode sequentially, DiMo performs iterative masked token refinement, unifying Text-to-Motion (T2M), Motion-to-Text (M2T), and text-free Motion-to-Motion (M2M) within a single model. This decoding paradigm naturally enables a quality-latency trade-off at inference via the number of refinement steps. We further improve motion token fidelity with residual vector quantization (RVQ) and enhance alignment and controllability with Group Relative Policy Optimization (GRPO). Experiments on HumanML3D and KIT-ML show strong motion quality and competitive bidirectional understanding under a unified framework. In addition, we demonstrate model ability in text-free motion completion, text-guided motion prediction and motion caption correction without architectural change. Additional qualitative results are available on our project page: https://animotionlab.github.io/DiMo/.
>
---
#### [replaced 056] Aligned Novel View Image and Geometry Synthesis via Cross-modal Attention Instillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.11924v3](https://arxiv.org/pdf/2506.11924v3)**

> **作者:** Min-Seop Kwak; Junho Kim; Sangdoo Yun; Dongyoon Han; Taekyung Kim; Seungryong Kim; Jin-Hwa Kim
>
> **备注:** Project page at https://cvlab-kaist.github.io/MoAI
>
> **摘要:** We introduce a diffusion-based framework that performs aligned novel view image and geometry generation via a warping-and-inpainting methodology. Unlike prior methods that require dense posed images or pose-embedded generative models limited to in-domain views, our method leverages off-the-shelf geometry predictors to predict partial geometries viewed from reference images, and formulates novel-view synthesis as an inpainting task for both image and geometry. To ensure accurate alignment between generated images and geometry, we propose cross-modal attention distillation, where attention maps from the image diffusion branch are injected into a parallel geometry diffusion branch during both training and inference. This multi-task approach achieves synergistic effects, facilitating geometrically robust image synthesis as well as well-defined geometry prediction. We further introduce proximity-based mesh conditioning to integrate depth and normal cues, interpolating between point cloud and filtering erroneously predicted geometry from influencing the generation process. Empirically, our method achieves high-fidelity extrapolative view synthesis on both image and geometry across a range of unseen scenes, delivers competitive reconstruction quality under interpolation settings, and produces geometrically aligned colored point clouds for comprehensive 3D completion. Project page is available at https://cvlab-kaist.github.io/MoAI.
>
---
#### [replaced 057] FloodDiffusion: Tailored Diffusion Forcing for Streaming Motion Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03520v2](https://arxiv.org/pdf/2512.03520v2)**

> **作者:** Yiyi Cai; Yuhan Wu; Kunhang Li; You Zhou; Bo Zheng; Haiyang Liu
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** We present FloodDiffusion, a new framework for text-driven, streaming human motion generation. Given time-varying text prompts, FloodDiffusion generates text-aligned, seamless motion sequences with real-time latency. Unlike existing methods that rely on chunk-by-chunk or auto-regressive model with diffusion head, we adopt a diffusion forcing framework to model this time-series generation task under time-varying control events. We find that a straightforward implementation of vanilla diffusion forcing (as proposed for video models) fails to model real motion distributions. We demonstrate that to guarantee modeling the output distribution, the vanilla diffusion forcing must be tailored to: (i) train with a bi-directional attention instead of casual attention; (ii) implement a lower triangular time scheduler instead of a random one; (iii) utilize a continues time-varying way to introduce text conditioning. With these improvements, we demonstrate in the first time that the diffusion forcing-based framework achieves state-of-the-art performance on the streaming motion generation task, reaching an FID of 0.057 on the HumanML3D benchmark. Models, code, and weights are available. https://shandaai.github.io/FloodDiffusion/
>
---
#### [replaced 058] Extracting Manifold Information from Point Clouds
- **分类: cs.CV; cs.CG; math.NA**

- **链接: [https://arxiv.org/pdf/2404.00427v3](https://arxiv.org/pdf/2404.00427v3)**

> **作者:** Patrick Guidotti
>
> **备注:** 21 pages, 11 figures, 4 tables
>
> **摘要:** A kernel based method is proposed for the construction of signature (defining) functions of subsets of $\mathbb{R}^d$. The subsets can range from full dimensional manifolds (open subsets) to point clouds (a finite number of points) and include bounded (closed) smooth manifolds of any codimension. The interpolation and analysis of point clouds are the main application. Two extreme cases in terms of regularity are considered, where the data set is interpolated by an analytic surface, at the one extreme, and by a Hölder continuous surface, at the other. The signature function can be computed as a combination of translated kernels, the coefficients of which are the solution of a Fredholm integral equation (matrix equation in the finite dimensional case). Once it is obtained, it can be used to estimate the dimension as well as the normal and the curvatures of the interpolated manifold. The method is global and does not require the data set to be organized or structured in any particular way. It admits a variational formulation with a natural regularized counterpart, that proves useful in dealing with data sets corrupted by numerical error or noise. The underlying analytical structure of the approach is presented in general before it is applied to the case of point clouds.
>
---
#### [replaced 059] 3D Object Detection for Autonomous Driving: A Survey
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2106.10823v4](https://arxiv.org/pdf/2106.10823v4)**

> **作者:** Rui Qian; Xin Lai; Xirong Li
>
> **备注:** The manuscript is accepted by Pattern Recognition on 14 May 2022
>
> **摘要:** Autonomous driving is regarded as one of the most promising remedies to shield human beings from severe crashes. To this end, 3D object detection serves as the core basis of perception stack especially for the sake of path planning, motion prediction, and collision avoidance etc. Taking a quick glance at the progress we have made, we attribute challenges to visual appearance recovery in the absence of depth information from images, representation learning from partially occluded unstructured point clouds, and semantic alignments over heterogeneous features from cross modalities. Despite existing efforts, 3D object detection for autonomous driving is still in its infancy. Recently, a large body of literature have been investigated to address this 3D vision task. Nevertheless, few investigations have looked into collecting and structuring this growing knowledge. We therefore aim to fill this gap in a comprehensive survey, encompassing all the main concerns including sensors, datasets, performance metrics and the recent state-of-the-art detection methods, together with their pros and cons. Furthermore, we provide quantitative comparisons with the state of the art. A case study on fifteen selected representative methods is presented, involved with runtime analysis, error analysis, and robustness analysis. Finally, we provide concluding remarks after an in-depth analysis of the surveyed works and identify promising directions for future work.
>
---
#### [replaced 060] Spectral Compressive Imaging via Chromaticity-Intensity Decomposition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.16690v2](https://arxiv.org/pdf/2509.16690v2)**

> **作者:** Xiaodong Wang; Zijun He; Ping Wang; Lishun Wang; Yanan Hu; Xin Yuan
>
> **摘要:** In coded aperture snapshot spectral imaging (CASSI), the captured measurement entangles spatial and spectral information, posing a severely ill-posed inverse problem for hyperspectral images (HSIs) reconstruction. Moreover, the captured radiance inherently depends on scene illumination, making it difficult to recover the intrinsic spectral reflectance that remains invariant to lighting conditions. To address these challenges, we propose a chromaticity-intensity decomposition framework, which disentangles an HSI into a spatially smooth intensity map and a spectrally variant chromaticity cube. The chromaticity encodes lighting-invariant reflectance, enriched with high-frequency spatial details and local spectral sparsity. Building on this decomposition, we develop CIDNet, a Chromaticity-Intensity Decomposition unfolding network within a dual-camera CASSI system. CIDNet integrates a hybrid spatial-spectral Transformer tailored to reconstruct fine-grained and sparse spectral chromaticity and a degradation-aware, spatially-adaptive noise estimation module that captures anisotropic noise across iterative stages. Extensive experiments on both synthetic and real-world CASSI datasets demonstrate that our method achieves superior performance in both spectral and chromaticity fidelity. Code and models will be publicly available.
>
---
#### [replaced 061] Synthetic Data Guided Feature Selection for Robust Activity Recognition in Older Adults
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.17053v2](https://arxiv.org/pdf/2601.17053v2)**

> **作者:** Shuhao Que; Dieuwke van Dartel; Ilse Heeringa; Han Hegeman; Miriam Vollenbroek-Hutten; Ying Wang
>
> **备注:** This paper has been submitted to Nordic Conference on Digital Health and Wireless Solutions 2026, currently under review
>
> **摘要:** Physical activity during hip fracture rehabilitation is essential for mitigating long-term functional decline in geriatric patients. However, it is rarely quantified in clinical practice. Existing continuous monitoring systems with commercially available wearable activity trackers are typically developed in middle-aged adults and therefore perform unreliably in older adults with slower and more variable gait patterns. This study aimed to develop a robust human activity recognition (HAR) system to improve continuous physical activity recognition in the context of hip fracture rehabilitation. 24 healthy older adults aged over 80 years were included to perform activities of daily living (walking, standing, sitting, lying down, and postural transfers) under simulated free-living conditions for 75 minutes while wearing two accelerometers positioned on the lower back and anterior upper thigh. Model robustness was evaluated using leave-one-subject-out cross-validation. The synthetic data demonstrated potential to improve generalization across participants. The resulting feature intervention model (FIM), aided by synthetic data guidance, achieved reliable activity recognition with mean F1-scores of 0.896 for walking, 0.927 for standing, 0.997 for sitting, 0.937 for lying down, and 0.816 for postural transfers. Compared with a control condition model without synthetic data, the FIM significantly improved the postural transfer detection, i.e., an activity class of high clinical relevance that is often overlooked in existing HAR literature. In conclusion, these preliminary results demonstrate the feasibility of robust activity recognition in older adults. Further validation in hip fracture patient populations is required to assess the clinical utility of the proposed monitoring system.
>
---
#### [replaced 062] Simulating the Visual World with Artificial Intelligence: A Roadmap
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08585v4](https://arxiv.org/pdf/2511.08585v4)**

> **作者:** Jingtong Yue; Ziqi Huang; Zhaoxi Chen; Xintao Wang; Pengfei Wan; Ziwei Liu
>
> **备注:** Project page: https://world-model-roadmap.github.io/ Github Repo: https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model
>
> **摘要:** The landscape of video generation is shifting, from a focus on generating visually appealing clips to building virtual environments that support interaction and maintain physical plausibility. These developments point toward the emergence of video foundation models that function not only as visual generators but also as implicit world models, models that simulate the physical dynamics, agent-environment interactions, and task planning that govern real or imagined worlds. This survey provides a systematic overview of this evolution, conceptualizing modern video foundation models as the combination of two core components: an implicit world model and a video renderer. The world model encodes structured knowledge about the world, including physical laws, interaction dynamics, and agent behavior. It serves as a latent simulation engine that enables coherent visual reasoning, long-term temporal consistency, and goal-driven planning. The video renderer transforms this latent simulation into realistic visual observations, effectively producing videos as a "window" into the simulated world. We trace the progression of video generation through four generations, in which the core capabilities advance step by step, ultimately culminating in a world model, built upon a video generation model, that embodies intrinsic physical plausibility, real-time multimodal interaction, and planning capabilities spanning multiple spatiotemporal scales. For each generation, we define its core characteristics, highlight representative works, and examine their application domains such as robotics, autonomous driving, and interactive gaming. Finally, we discuss open challenges and design principles for next-generation world models, including the role of agent intelligence in shaping and evaluating these systems. An up-to-date list of related works is maintained at this link.
>
---
#### [replaced 063] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning
- **分类: cs.LG; cs.CG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.23484v2](https://arxiv.org/pdf/2510.23484v2)**

> **作者:** Julie Mordacq; David Loiseaux; Vicky Kalogeiton; Steve Oudot
>
> **备注:** NeurIPS 2025
>
> **摘要:** Self-supervised learning (SSL) has emerged as a powerful paradigm for learning representations without labeled data, often by enforcing invariance to input transformations such as rotations or blurring. Recent studies have highlighted two pivotal properties for effective representations: (i) avoiding dimensional collapse-where the learned features occupy only a low-dimensional subspace, and (ii) enhancing uniformity of the induced distribution. In this work, we introduce T-REGS, a simple regularization framework for SSL based on the length of the Minimum Spanning Tree (MST) over the learned representation. We provide theoretical analysis demonstrating that T-REGS simultaneously mitigates dimensional collapse and promotes distribution uniformity on arbitrary compact Riemannian manifolds. Several experiments on synthetic data and on classical SSL benchmarks validate the effectiveness of our approach at enhancing representation quality.
>
---
