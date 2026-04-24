# 计算机视觉 cs.CV

- **最新发布 102 篇**

- **更新 70 篇**

## 最新发布

#### [new 001] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval
- **分类: cs.CV**

- **简介: 该论文属于多修改图像检索任务，旨在解决实体覆盖不足和文本-实体错位问题。构建了两个多修改数据集，并提出TEMA框架，提升检索效果与效率。**

- **链接: [https://arxiv.org/pdf/2604.21806](https://arxiv.org/pdf/2604.21806)**

> **作者:** Zixu Li; Yupeng Hu; Zhiheng Fu; Zhiwei Chen; Yongqi Li; Liqiang Nie
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Composed Image Retrieval (CIR) is an important image retrieval paradigm that enables users to retrieve a target image using a multimodal query that consists of a reference image and modification text. Although research on CIR has made significant progress, prevailing setups still rely simple modification texts that typically cover only a limited range of salient changes, which induces two limitations highly relevant to practical applications, namely Insufficient Entity Coverage and Clause-Entity Misalignment. In order to address these issues and bring CIR closer to real-world use, we construct two instruction-rich multi-modification datasets, M-FashionIQ and M-CIRR. In addition, we propose TEMA, the Text-oriented Entity Mapping Architecture, which is the first CIR framework designed for multi-modification while also accommodating simple modifications. Extensive experiments on four benchmark datasets demonstrate that TEMA's superiority in both original and multi-modification scenarios, while maintaining an optimal balance between retrieval accuracy and computational efficiency. Our codes and constructed multi-modification dataset (M-FashionIQ and M-CIRR) are available at this https URL.
>
---
#### [new 002] StyleVAR: Controllable Image Style Transfer via Visual Autoregressive Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出StyleVAR，用于可控图像风格迁移。解决风格与内容平衡问题，通过视觉自回归建模实现。**

- **链接: [https://arxiv.org/pdf/2604.21052](https://arxiv.org/pdf/2604.21052)**

> **作者:** Liqi Jing; Dingming Zhang; Peinian Li; Lichen Zhu
>
> **摘要:** We build on the Visual Autoregressive Modeling (VAR) framework and formulate style transfer as conditional discrete sequence modeling in a learned latent space. Images are decomposed into multi-scale representations and tokenized into discrete codes by a VQ-VAE; a transformer then autoregressively models the distribution of target tokens conditioned on style and content tokens. To inject style and content information, we introduce a blended cross-attention mechanism in which the evolving target representation attends to its own history, while style and content features act as queries that decide which aspects of this history to emphasize. A scale-dependent blending coefficient controls the relative influence of style and content at each stage, encouraging the synthesized representation to align with both the content structure and the style texture without breaking the autoregressive continuity of VAR. We train StyleVAR in two stages from a pretrained VAR checkpoint: supervised fine-tuning on a large triplet dataset of content--style--target images, followed by reinforcement fine-tuning with Group Relative Policy Optimization (GRPO) against a DreamSim-based perceptual reward, with per-action normalization weighting to rebalance credit across VAR's multi-scale hierarchy. Across three benchmarks spanning in-, near-, and out-of-distribution regimes, StyleVAR consistently outperforms an AdaIN baseline on Style Loss, Content Loss, LPIPS, SSIM, DreamSim, and CLIP similarity, and the GRPO stage yields further gains over the SFT checkpoint, most notably on the reward-aligned perceptual metrics. Qualitatively, the method transfers texture while maintaining semantic structure, especially for landscapes and architectural scenes, while a generalization gap on internet images and difficulty with human faces highlight the need for better content diversity and stronger structural priors.
>
---
#### [new 003] Divide-then-Diagnose: Weaving Clinician-Inspired Contexts for Ultra-Long Capsule Endoscopy Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于胶囊内镜视频分析任务，旨在解决超长视频中关键病灶帧提取与诊断问题。提出DiCE框架，通过上下文组织和证据聚合实现高效诊断摘要。**

- **链接: [https://arxiv.org/pdf/2604.21814](https://arxiv.org/pdf/2604.21814)**

> **作者:** Bowen Liu; Li Yang; Shanshan Song; Mingyu Tang; Zhifang Gao; Qifeng Chen; Yangqiu Song; Huimin Chen; Xiaomeng Li
>
> **摘要:** Capsule endoscopy (CE) enables non-invasive gastrointestinal screening, but current CE research remains largely limited to frame-level classification and detection, leaving video-level analysis underexplored. To bridge this gap, we introduce and formally define a new task, diagnosis-driven CE video summarization, which requires extracting key evidence frames that covers clinically meaningful findings and making accurate diagnoses from those evidence frames. This setting is challenging because diagnostically relevant events are extremely sparse and can be overwhelmed by tens of thousands of redundant normal frames, while individual observations are often ambiguous due to motion blur, debris, specular highlights, and rapid viewpoint changes. To facilitate research in this direction, we introduce VideoCAP, the first CE dataset with diagnosis-driven annotations derived from real clinical reports. VideoCAP comprises 240 full-length videos and provides realistic supervision for both key evidence frame extraction and diagnosis. To address this task, we further propose DiCE, a clinician-inspired framework that mirrors the standard CE reading workflow. DiCE first performs efficient candidate screening over the raw video, then uses a Context Weaver to organize candidates into coherent diagnostic contexts that preserve distinct lesion events, and an Evidence Converger to aggregate multi-frame evidence within each context into robust clip-level judgments. Experiments show that DiCE consistently outperforms state-of-the-art methods, producing concise and clinically reliable diagnostic summaries. These results highlight diagnosis-driven contextual reasoning as a promising paradigm for ultra-long CE video summarization.
>
---
#### [new 004] VFM$^{4}$SDG: Unveiling the Power of VFMs for Single-Domain Generalized Object Detection
- **分类: cs.CV**

- **简介: 该论文属于单域广义目标检测任务，旨在解决域变化导致的检测性能下降问题。通过引入冻结视觉基础模型，提升检测器在跨域场景下的稳定性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.21502](https://arxiv.org/pdf/2604.21502)**

> **作者:** Yupeng Zhang; Ruize Han; Ningnan Guo; Wei Feng; Song Wang; Liang Wan
>
> **摘要:** In real-world scenarios, continual changes in weather, illumination, and imaging conditions cause significant domain shifts, leading detectors trained on a single source domain to degrade severely in unseen environments. Existing single-domain generalized object detection (SDGOD) methods mainly rely on data augmentation or domain-invariant representation learning, but pay limited attention to detector mechanisms, leaving clear limitations under complex domain shifts. Through analytical experiments, we find that performance degradation is dominated by increasing missed detections, which fundamentally arises from reduced cross-domain stability of the detector: object-background and inter-instance relations become less stable in the encoding stage, while semantic-spatial alignment of query representations also becomes harder to maintain in the decoding stage. To this end, we propose VFM$^{4}$SDG, a dual-prior learning framework for SDGOD, which introduces a frozen vision foundation model (VFM) as a transferable cross-domain stability prior into detector representation learning and query modeling. In the encoding stage, we propose Cross-domain Stable Relational Prior Distillation to enhance the robustness of object-background and inter-instance relational modeling. In the decoding stage, we propose Semantic-Contextual Prior-based Query Enhancement, which injects category-level semantic prototypes and global visual context into queries to improve their semantic recognition and spatial localization stability in unseen domains. Extensive experiments show that the proposed method consistently outperforms existing SOTA methods on standard SDGOD benchmarks and two mainstream DETR-based detectors, demonstrating its effectiveness, robustness, and generality.
>
---
#### [new 005] Projected Gradient Unlearning for Text-to-Image Diffusion Models: Defending Against Concept Revival Attacks
- **分类: cs.CV**

- **简介: 该论文属于机器学习中的模型去偏任务，解决扩散模型中概念复兴问题。通过PGU方法，在微调后仍能有效消除不良概念。**

- **链接: [https://arxiv.org/pdf/2604.21041](https://arxiv.org/pdf/2604.21041)**

> **作者:** Aljalila Aladawi; Mohammed Talha Alam; Fakhri Karray
>
> **摘要:** Machine unlearning for text-to-image diffusion models aims to selectively remove undesirable concepts from pre-trained models without costly retraining. Current unlearning methods share a common weakness: erased concepts return when the model is fine-tuned on downstream data, even when that data is entirely unrelated. We adapt Projected Gradient Unlearning (PGU) from classification to the diffusion domain as a post-hoc hardening step. By constructing a Core Gradient Space (CGS) from the retain concept activations and projecting gradient updates into its orthogonal complement, PGU ensures that subsequent fine-tuning cannot undo the achieved erasure. Applied on top of existing methods (ESD, UCE, Receler), the approach eliminates revival for style concepts and substantially delays it for object concepts, running in roughly 6 minutes versus the ~2 hours required by Meta-Unlearning. PGU and Meta-Unlearning turn out to be complementary: which performs better depends on how the concept is encoded, and retain concept selection should follow visual feature similarity rather than semantic grouping.
>
---
#### [new 006] Reinforcing 3D Understanding in Point-VLMs via Geometric Reward Credit Assignment
- **分类: cs.CV**

- **简介: 该论文属于3D视觉语言模型任务，解决几何幻觉问题。通过几何奖励分配和重投影一致性约束，提升3D结构的准确性与物理合理性。**

- **链接: [https://arxiv.org/pdf/2604.21160](https://arxiv.org/pdf/2604.21160)**

> **作者:** Jingkun Chen; Ruoshi Xu; Mingqi Gao; Shengda Luo; Jungong Han
>
> **备注:** 10 pages, 3 figures, 5 tables
>
> **摘要:** Point-Vision-Language Models promise to empower embodied agents with executable spatial reasoning, yet they frequently succumb to geometric hallucination where predicted 3D structures contradict the observed 2D reality. We identify a key cause of this failure not as a representation bottleneck but as a structural misalignment in reinforcement learning, where sparse geometric tokens are drowned out by noisy and broadcasted sequence-level rewards. To resolve this causal dilution, we propose Geometric Reward Credit Assignment, a framework that disentangles holistic supervision into field-specific signals and routes them exclusively to their responsible token spans. This mechanism transforms vague feedback into precise gradient updates and effectively turns generic policy optimization into targeted structural alignment. Furthermore, we internalize physical constraints via a Reprojection-Consistency term which serves as a cross-modal verifier to penalize physically impossible geometries. Validated on a calibrated benchmark derived from ShapeNetCore, our approach bridges the reliability gap by boosting 3D KPA from 0.64 to 0.93, increasing 3D bounding box intersection over union to 0.686, and raising reprojection consistency scores to 0.852. Crucially, these gains are achieved while maintaining robust 2D localization performance, marking a meaningful step from plausible textual outputs toward physically verifiable spatial predictions.
>
---
#### [new 007] Unlocking Multi-Spectral Data for Multi-Modal Models with Guided Inputs and Chain-of-Thought Reasoning
- **分类: cs.CV**

- **简介: 该论文属于多模态模型任务，旨在解决LMMs仅适应RGB图像的问题。通过在推理阶段引入多光谱数据，提升模型在遥感任务中的性能。**

- **链接: [https://arxiv.org/pdf/2604.21032](https://arxiv.org/pdf/2604.21032)**

> **作者:** Dahun Kim; Ganesh Satish Mallya; Anelia Angelova
>
> **备注:** Accepted to IGARSS 2026
>
> **摘要:** Multi-spectral imagery is a valuable input signal for Remote Sensing applications, such as land-use and land-cover classification and environmental monitoring. However, generalist Large Multi-modal Models (LMMs) are typically trained on RGB images, limiting their applicability to the RGB domain. At the same time, training multi-spectral multi-modal models is expensive and produces uniquely specialized models. To address this, we propose a novel training-free approach that introduces multi-spectral data within the inference pipeline of standard RGB-only LMMs, allowing large gains in performance. Our approach leverages the LMMs' understanding of the visual space by adapting non-RGB inputs to that space and injecting domain-specific information and Chain-of-Thought reasoning as instructions. We demonstrate this with the Gemini 2.5 model and observe strong Zero-Shot performance gains on popular Remote Sensing benchmarks. These results highlight the potential for geospatial professionals to leverage powerful generalist models for specialized sensor inputs, benefiting from rich reasoning capabilities grounded in specialized data.
>
---
#### [new 008] Temporal Prototyping and Hierarchical Alignment for Unsupervised Video-based Visible-Infrared Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于视频可见-红外行人重识别任务，解决无监督视频VI-ReID问题，提出HiTPro框架提升跨模态匹配性能。**

- **链接: [https://arxiv.org/pdf/2604.21324](https://arxiv.org/pdf/2604.21324)**

> **作者:** Zhiyong Li; Wei Jiang; Haojie Liu; Mingyu Wang; Wanchong Xu; Weijie Mao
>
> **摘要:** Visible-infrared person re-identification (VI-ReID) enables cross-modality identity matching for all-day surveillance, yet existing methods predominantly focus on the image level or rely heavily on costly identity annotations. While video-based VI-ReID has recently emerged to exploit temporal dynamics for improved robustness, existing studies remain limited to supervised settings. Crucially, the unsupervised video VI-ReID problem, where models must learn from RGB and infrared tracklets without identity labels, remains largely unexplored despite its practical importance in real-world deployment. To bridge this gap, we propose HiTPro (Hierarchical Temporal Prototyping), a prototype-driven framework without explicit hard pseudo-label assignment for unsupervised video-based VI-ReID. HiTPro begins with an efficient Temporal-aware Feature Encoder that first extracts discriminative frame-level features and then aggregates them into a robust tracklet-level representation. Building upon these features, HiTPro first constructs reliable intra-camera prototypes via Intra-Camera Tracklet Prototyping by aggregating features from temporally partitioned sub-tracklets. Through Hierarchical Cross-Prototype Alignment, we perform a two-stage positive mining process: progressing from within-modality associations to cross-modality matching, enhanced by Dynamic Threshold Strategy and Soft Weight Assignment. Finally, {Hierarchical Contrastive Learning} progressively optimizes feature-prototype alignment across three levels: intra-camera discrimination, cross-camera same-modality consistency, and cross-modality invariance. Extensive experiments on HITSZ-VCM and BUPTCampus demonstrate that HiTPro achieves state-of-the-art performance under fully unsupervised settings, significantly outperforming adapted baselines and establishes a strong baseline for future research.
>
---
#### [new 009] Unlocking the Power of Critical Factors for 3D Visual Geometry Estimation
- **分类: cs.CV**

- **简介: 该论文属于视觉几何估计任务，旨在解决多帧模型与单帧方法在性能上的矛盾。通过分析关键因素，提出改进方法，提升模型准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.21713](https://arxiv.org/pdf/2604.21713)**

> **作者:** Guangkai Xu; Hua Geng; Huanyi Zheng; Songyi Yin; Yanlong Sun; Hao Chen; Chunhua Shen
>
> **备注:** Accepted to CVPR 2026. GitHub Page: this https URL
>
> **摘要:** Feed-forward visual geometry estimation has recently made rapid progress. However, an important gap remains: multi-frame models usually produce better cross-frame consistency, yet they often underperform strong per-frame methods on single-frame accuracy. This observation motivates our systematic investigation into the critical factors driving model performance through rigorous ablation studies, which reveals several key insights: 1) Scaling up data diversity and quality unlocks further performance gains even in state-of-the-art visual geometry estimation methods; 2) Commonly adopted confidence-aware loss and gradient-based loss mechanisms may unintentionally hinder performance; 3) Joint supervision through both per-sequence and per-frame alignment improves results, while local region alignment surprisingly degrades performance. Furthermore, we introduce two enhancements to integrate the advantages of optimization-based methods and high-resolution inputs: a consistency loss function that enforces alignment between depth maps, camera parameters, and point maps, and an efficient architectural design that leverages high-resolution information. We integrate these designs into CARVE, a resolution-enhanced model for feed-forward visual geometry estimation. Experiments on point cloud reconstruction, video depth estimation, and camera pose/intrinsic estimation show that CARVE achieves strong and robust performance across diverse benchmarks.
>
---
#### [new 010] Rethinking Cross-Domain Evaluation for Face Forgery Detection with Semantic Fine-grained Alignment and Mixture-of-Experts
- **分类: cs.CV**

- **简介: 该论文属于人脸伪造检测任务，解决跨数据集泛化能力不足的问题。提出Cross-AUC评估指标和SFAM框架，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.21478](https://arxiv.org/pdf/2604.21478)**

> **作者:** Yuhan Luo; Tao Chen; Decheng Liu
>
> **备注:** The source code is available at this https URL
>
> **摘要:** Nowadays, visual data forgery detection plays an increasingly important role in social and economic security with the rapid development of generative models. Existing face forgery detectors still can't achieve satisfactory performance because of poor generalization ability across datasets. The key factor that led to this phenomenon is the lack of suitable metrics: the commonly used cross-dataset AUC metric fails to reveal an important issue where detection scores may shift significantly across data domains. To explicitly evaluate cross-domain score comparability, we propose \textbf{Cross-AUC}, an evaluation metric that can compute AUC across dataset pairs by contrasting real samples from one dataset with fake samples from another (and vice versa). It is interesting to find that evaluating representative detectors under the Cross-AUC metric reveals substantial performance drops, exposing an overlooked robustness problem. Besides, we also propose the novel framework \textbf{S}emantic \textbf{F}ine-grained \textbf{A}lignment and \textbf{M}ixture-of-Experts (\textbf{SFAM}), consisting of a patch-level image-text alignment module that enhances CLIP's sensitivity to manipulation artifacts, and the facial region mixture-of-experts module, which routes features from different facial regions to specialized experts for region-aware forgery analysis. Extensive qualitative and quantitative experiments on the public datasets prove that the proposed method achieves superior performance compared with the state-of-the-art methods with various suitable metrics.
>
---
#### [new 011] Foveated Reasoning: Stateful, Action-based Visual Focusing for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出Foveated Reasoner，解决视觉语言模型中高分辨率图像带来的计算开销问题。通过模拟人类视觉聚焦机制，提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.21079](https://arxiv.org/pdf/2604.21079)**

> **作者:** Juhong Min; Lazar Valkov; Vitali Petsiuk; Hossein Souri; Deen Dayal Mohan
>
> **摘要:** Vision-language models benefit from high-resolution images, but the increase in visual-token count incurs high compute overhead. Humans resolve this tension via foveation: a coarse view guides "where to look", while selectively acquired high-acuity evidence refines "what to think". We introduce Foveated Reasoner, an autoregressive vision-language framework that unifies foveation and reasoning within a single decoding trajectory. Starting from a low-resolution view, the model triggers foveation only when needed, retrieves high-resolution evidence from selected regions, and injects it back into the same decoding trajectory. We train the method with a two-stage pipeline: coldstart supervision to bootstrap foveation behavior, followed by reinforcement learning to jointly improve evidence acquisition and task accuracy while discouraging trivial "see-everything" solutions. Experiments show that the method learns effective foveation policies and achieves stronger accuracy under tight visual-token budgets across multiple vision-language benchmarks.
>
---
#### [new 012] 2L-LSH: A Locality-Sensitive Hash Function-Based Method For Rapid Point Cloud Indexing
- **分类: cs.CV**

- **简介: 该论文属于点云处理任务，解决快速邻近点搜索问题。提出2L-LSH算法，通过两步哈希策略提升搜索效率，实验表明其优于Kd-tree和Octree。**

- **链接: [https://arxiv.org/pdf/2604.21442](https://arxiv.org/pdf/2604.21442)**

> **作者:** Shurui Wang; Yuhe Zhang; Ruizhe Guo; Yaning Zhang; Yifei Xie; Xinyu Zhou
>
> **备注:** 13 pages, 13 figures. Published in The Computer Journal
>
> **摘要:** The development of 3D scanning technology has enabled the acquisition of massive point cloud models with diverse structures and large scales, thereby presenting significant challenges in point cloud processing. Fast neighboring points search is one of the most common problems, which is frequently used in model reconstruction, classification, retrieval and feature visualization. Hash function is well known for its high-speed and accurate performance in searching high-dimensional data, which is also the core of the proposed 2L-LSH. Specifically, the 2L-LSH algorithm adopts a two-step hash function strategy, in which the popular step divides the bounding box of the point cloud model and the second step constructs a generalized table-based data structure. The proposed 2L-LSH offers a highly efficient and accurate solution for fast neighboring points search in large-scale 3D point cloud models, making it a promising technique for various applications in the field. The proposed algorithm is compared with the well-known methods including Kd-tree and Octree; the obtained results demonstrated that the proposed method outperforms Kd-tree and Octree in terms of speed, i.e. the time consumption of kNN search can be 51.111% and 94.159% lower than Kd-tree and Octree, respectively. And the RN search time can be 54.519% and 41.840% lower than Kd-tree and Octree, respectively.
>
---
#### [new 013] Back to Source: Open-Set Continual Test-Time Adaptation via Domain Compensation
- **分类: cs.CV**

- **简介: 该论文属于测试时自适应任务，解决持续变化域与未知类别共存的挑战。提出DOCO框架，实现领域补偿与异常检测的协同优化。**

- **链接: [https://arxiv.org/pdf/2604.21772](https://arxiv.org/pdf/2604.21772)**

> **作者:** Yingkai Yang; Chaoqi Chen; Hui Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Test-Time Adaptation (TTA) aims to mitigate distributional shifts between training and test domains during inference time. However, existing TTA methods fall short in the realistic scenario where models face both continually changing domains and the simultaneous emergence of unknown semantic classes, a challenging setting we term Open-set Continual Test-Time Adaptation (OCTTA). The coupling of domain and semantic shifts often collapses the feature space, severely degrading both classification and out-of-distribution detection. To tackle this, we propose DOmain COmpensation (DOCO), a lightweight and effective framework that robustly performs domain adaptation and OOD detection in a synergistic, closed loop. DOCO first performs dynamic, adaptation-conditioned sample splitting to separate likely ID from OOD samples. Then, using only the ID samples, it learns a domain compensation prompt by aligning feature statistics with the source domain, guided by a structural preservation regularizer that prevents semantic distortion. This learned prompt is then propagated to the OOD samples within the same batch, effectively isolating their semantic novelty for more reliable detection. Extensive experiments on multiple challenging benchmarks demonstrate that DOCO outperforms prior CTTA and OSTTA methods, establishing a new state-of-the-art for the demanding OCTTA setting.
>
---
#### [new 014] Encoder-Free Human Motion Understanding via Structured Motion Descriptions
- **分类: cs.CV**

- **简介: 该论文属于人体动作理解任务，旨在解决传统方法依赖编码器的问题。提出SMD将动作序列转为结构化文本，使LLM直接利用预训练知识进行动作推理，提升问答与描述性能。**

- **链接: [https://arxiv.org/pdf/2604.21668](https://arxiv.org/pdf/2604.21668)**

> **作者:** Yao Zhang; Zhuchenyang Liu; Thomas Ploetz; Yu Xiao
>
> **摘要:** The world knowledge and reasoning capabilities of text-based large language models (LLMs) are advancing rapidly, yet current approaches to human motion understanding, including motion question answering and captioning, have not fully exploited these capabilities. Existing LLM-based methods typically learn motion-language alignment through dedicated encoders that project motion features into the LLM's embedding space, remaining constrained by cross-modal representation and alignment. Inspired by biomechanical analysis, where joint angles and body-part kinematics have long served as a precise descriptive language for human movement, we propose \textbf{Structured Motion Description (SMD)}, a rule-based, deterministic approach that converts joint position sequences into structured natural language descriptions of joint angles, body part movements, and global trajectory. By representing motion as text, SMD enables LLMs to apply their pretrained knowledge of body parts, spatial directions, and movement semantics directly to motion reasoning, without requiring learned encoders or alignment modules. We show that this approach goes beyond state-of-the-art results on both motion question answering (66.7\% on BABEL-QA, 90.1\% on HuMMan-QA) and motion captioning (R@1 of 0.584, CIDEr of 53.16 on HumanML3D), surpassing all prior methods. SMD additionally offers practical benefits: the same text input works across different LLMs with only lightweight LoRA adaptation (validated on 8 LLMs from 6 model families), and its human-readable representation enables interpretable attention analysis over motion descriptions. Code, data, and pretrained LoRA adapters are available at this https URL.
>
---
#### [new 015] Trust-SSL: Additive-Residual Selective Invariance for Robust Aerial Self-Supervised Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.NE**

- **简介: 该论文属于自监督学习任务，旨在提升航拍图像在退化情况下的表示鲁棒性。通过引入信任权重和残差损失，增强模型对噪声的抵抗能力。**

- **链接: [https://arxiv.org/pdf/2604.21349](https://arxiv.org/pdf/2604.21349)**

> **作者:** Wadii Boulila; Adel Ammar; Bilel Benjdira; Maha Driss
>
> **备注:** 17 pages
>
> **摘要:** Self-supervised learning (SSL) is a standard approach for representation learning in aerial imagery. Existing methods enforce invariance between augmented views, which works well when augmentations preserve semantic content. However, aerial images are frequently degraded by haze, motion blur, rain, and occlusion that remove critical evidence. Enforcing alignment between a clean and a severely degraded view can introduce spurious structure into the latent space. This study proposes a training strategy and architectural modification to enhance SSL robustness to such corruptions. It introduces a per-sample, per-factor trust weight into the alignment objective, combined with the base contrastive loss as an additive residual. A stop-gradient is applied to the trust weight instead of a multiplicative gate. While a multiplicative gate is a natural choice, experiments show it impairs the backbone, whereas our additive-residual approach improves it. Using a 200-epoch protocol on a 210,000-image corpus, the method achieves the highest mean linear-probe accuracy among six backbones on EuroSAT, AID, and NWPU-RESISC45 (90.20% compared to 88.46% for SimCLR and 89.82% for VICReg). It yields the largest improvements under severe information-erasing corruptions on EuroSAT (+19.9 points on haze at s=5 over SimCLR). The method also demonstrates consistent gains of +1 to +3 points in Mahalanobis AUROC on a zero-shot cross-domain stress test using BDD100K weather splits. Two ablations (scalar uncertainty and cosine gate) indicate the additive-residual formulation is the primary source of these improvements. An evidential variant using Dempster-Shafer fusion introduces interpretable signals of conflict and ignorance. These findings offer a concrete design principle for uncertainty-aware SSL. Code is publicly available at this https URL.
>
---
#### [new 016] Linear Image Generation by Synthesizing Exposure Brackets
- **分类: cs.CV**

- **简介: 该论文属于文本到线性图像生成任务，旨在解决生成高质量、保留动态范围的线性图像问题。通过曝光 bracket 表示和 DiT 架构实现高效生成。**

- **链接: [https://arxiv.org/pdf/2604.21008](https://arxiv.org/pdf/2604.21008)**

> **作者:** Yuekun Dai; Zhoutong Zhang; Shangchen Zhou; Nanxuan Zhao
>
> **备注:** accepted by CVPR2026
>
> **摘要:** The life of a photo begins with photons striking the sensor, whose signals are passed through a sophisticated image signal processing (ISP) pipeline to produce a display-referred image. However, such images are no longer faithful to the incident light, being compressed in dynamic range and stylized by subjective preferences. In contrast, RAW images record direct sensor signals before non-linear tone mapping. After camera response curve correction and demosaicing, they can be converted into linear images, which are scene-referred representations that directly reflect true irradiance and are invariant to sensor-specific factors. Since image sensors have better dynamic range and bit depth, linear images contain richer information than display-referred ones, leaving users more room for editing during post-processing. Despite this advantage, current generative models mainly synthesize display-referred images, which inherently limits downstream editing. In this paper, we address the task of text-to-linear-image generation: synthesizing a high-quality, scene-referred linear image that preserves full dynamic range, conditioned on a text prompt, for professional post-processing. Generating linear images is challenging, as pre-trained VAEs in latent diffusion models struggle to simultaneously preserve extreme highlights and shadows due to the higher dynamic range and bit depth. To this end, we represent a linear image as a sequence of exposure brackets, each capturing a specific portion of the dynamic range, and propose a DiT-based flow-matching architecture for text-conditioned exposure bracket generation. We further demonstrate downstream applications including text-guided linear image editing and structure-conditioned generation via ControlNet.
>
---
#### [new 017] Seeing Without Eyes: 4D Human-Scene Understanding from Wearable IMUs
- **分类: cs.CV**

- **简介: 该论文属于4D人体-场景理解任务，解决视觉感知的隐私与效率问题。通过可穿戴IMU传感器，实现人体运动与场景结构的非视觉重建。**

- **链接: [https://arxiv.org/pdf/2604.21926](https://arxiv.org/pdf/2604.21926)**

> **作者:** Hao-Yu Hsu; Tianhang Cheng; Jing Wen; Alexander G. Schwing; Shenlong Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** Understanding human activities and their surrounding environments typically relies on visual perception, yet cameras pose persistent challenges in privacy, safety, energy efficiency, and scalability. We explore an alternative: 4D perception without vision. Its goal is to reconstruct human motion and 3D scene layouts purely from everyday wearable sensors. For this we introduce IMU-to-4D, a framework that repurposes large language models for non-visual spatiotemporal understanding of human-scene dynamics. IMU-to-4D uses data from a few inertial sensors from earbuds, watches, or smartphones and predicts detailed 4D human motion together with coarse scene structure. Experiments across diverse human-scene datasets show that IMU-to-4D yields more coherent and temporally stable results than SoTA cascaded pipelines, suggesting wearable motion sensors alone can support rich 4D understanding.
>
---
#### [new 018] GraphLeap: Decoupling Graph Construction and Convolution for Vision GNN Acceleration on FPGA
- **分类: cs.CV; cs.DC**

- **简介: 该论文提出GraphLeap，解决Vision GNN中图构建与卷积的依赖问题，通过解耦实现加速。属于视觉图神经网络加速任务。**

- **链接: [https://arxiv.org/pdf/2604.21290](https://arxiv.org/pdf/2604.21290)**

> **作者:** Anvitha Ramachandran; Dhruv Parikh; Viktor Prasanna
>
> **备注:** FCCM 2026
>
> **摘要:** Vision Graph Neural Networks (ViGs) represent an image as a graph of patch tokens, enabling adaptive, feature-driven neighborhoods. Unlike CNNs with fixed grid biases or Vision Transformers with global token interactions, ViGs rely on dynamic graph convolution: at each layer, a feature-dependent graph is built via k-nearest-neighbor (kNN) search on current patch features, followed by message passing. This per-layer graph construction is the main bottleneck, consuming 50--95\% of graph convolution time on CPUs and GPUs, scaling as $O(N^2)$ with the number of patches $N$, and creating a sequential dependency between graph construction and feature updates. We introduce GraphLeap, a simple reformulation that removes this dependency by decoupling graph construction from feature update across layers. GraphLeap performs the feature update at layer $\ell$ using a graph built from the previous layer's features, while simultaneously using the current layer's features to construct the graph for layer $\ell+1$. This one-layer-lookahead graph construction enables concurrent graph construction and message passing. Although using prior-layer features can introduce minor accuracy degradation, lightweight fine-tuning for a few epochs is sufficient to recover the original accuracy. Building on GraphLeap, we present the first end-to-end FPGA accelerator for Vision GNNs. Our streaming, layer-pipelined design overlaps a kNN graph construction engine with a feature update engine, exploits node- and channel-level parallelism, and enables efficient on-chip dataflow without explicit edge-feature materialization. Evaluated on isotropic and pyramidal ViG models on an Alveo U280 FPGA, GraphLeap achieves up to $95.7\times$ speedup over CPU and $8.5\times$ speedup over GPU baselines, demonstrating the feasibility of real-time Vision GNN inference.
>
---
#### [new 019] PLAS-Net: Pixel-Level Area Segmentation for UAV-Based Beach Litter Monitoring
- **分类: cs.CV; cs.CY**

- **简介: 该论文提出PLAS-Net，解决UAV监测中海滩垃圾面积估算不准确的问题，通过像素级分割提高生态风险评估精度。**

- **链接: [https://arxiv.org/pdf/2604.21313](https://arxiv.org/pdf/2604.21313)**

> **作者:** Yongying Liu; Jiaqi Wang; Jian Song; Xinlei Shao; Yijia Chen; Nan Xu; Katsunori Mizuno; Shigeru Tabeta; Fan Zhao
>
> **备注:** 30 pages, 12 figures
>
> **摘要:** Accurate quantification of the physical exposure area of beach litter, rather than simple item counts, is essential for credible ecological risk assessment of marine debris. However, automated UAV-based monitoring predominantly relies on bounding-box detection, which systematically overestimates the planar area of irregular litter objects. To address this geometric limitation, we develop PLAS-Net (Pixel-level Litter Area Segmentor), an instance segmentation framework that extracts pixel-accurate physical footprints of coastal debris. Evaluated on UAV imagery from a monsoon-driven pocket beach in Koh Tao, Thailand, PLAS-Net achieves a mAP_50 of 58.7% with higher precision than eleven baseline models, demonstrating improved mask fidelity under complex coastal conditions. To illustrate how the accuracy of the masking affects the conclusions of environmental analysis, we conducted three downstream demonstrations: (i) power-law fitting of normalized plastic density (NPD) to characterize fragmentation dynamics; (ii) area-weighted ecological risk index (ERI) to map spatial pollution hotspots; and (iii) source composition analysis revealing the abundance-area paradox: fishing gear constitutes a small proportion of the total number of items, but has the largest physical area per unit item. Pixel-level area extraction can provide more valuable information for coastal monitoring compared to methods based solely on counting.
>
---
#### [new 020] DCMorph: Face Morphing via Dual-Stream Cross-Attention Diffusion
- **分类: cs.CV**

- **简介: 该论文属于人脸伪造任务，旨在提升人脸伪装攻击效果。提出DCMorph框架，通过双流注意力机制和扩散模型实现高质量人脸融合，有效规避检测。**

- **链接: [https://arxiv.org/pdf/2604.21627](https://arxiv.org/pdf/2604.21627)**

> **作者:** Tahar Chettaoui; Eduarda Caldeira; Guray Ozgur; Raghavendra Ramachandra; Fadi Boutros; Naser Damer
>
> **备注:** Accepted At CVPR-W 2026
>
> **摘要:** Advancing face morphing attack techniques is crucial to anticipate evolving threats and develop robust defensive mechanisms for identity verification systems. This work introduces DCMorph, a dual-stream diffusion-based morphing framework that simultaneously operates at both identity conditioning and latent space levels. Unlike image-level methods suffering from blending artifacts or GAN-based approaches with limited reconstruction fidelity, DCMorph leverages identity-conditioned latent diffusion models through two mechanisms: (1) decoupled cross-attention interpolation that injects identity-specific features from both source faces into the denoising process, enabling explicit dual-identity conditioning absent in existing diffusion-based methods, and (2) DDIM inversion with spherical interpolation between inverted latent representations from both source faces, providing geometrically consistent initial latent representation that preserves structural attributes. Vulnerability analyses across four state-of-the-art face recognition systems demonstrate that DCMorph achieves the highest attack success rates compared to existing methods at both operational thresholds, while remaining challenging to detect by current morphing attack detection solutions.
>
---
#### [new 021] Discriminative-Generative Synergy for Occlusion Robust 3D Human Mesh Recovery
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于3D人体网格恢复任务，旨在解决遮挡下重建不准确的问题。融合视觉Transformer与扩散模型，提升重建的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.21712](https://arxiv.org/pdf/2604.21712)**

> **作者:** Yang Liu; Zhiyong Zhang
>
> **摘要:** 3D human mesh recovery from monocular RGB images aims to estimate anatomically plausible 3D human models for downstream applications, but remains challenging under partial or severe occlusions. Regression-based methods are efficient yet often produce implausible or inaccurate results in unconstrained scenarios, while diffusion-based methods provide strong generative priors for occluded regions but may weaken fidelity to rare poses due to over-reliance on generation. To address these limitations, we propose a brain-inspired synergistic framework that integrates the discriminative power of vision transformers with the generative capability of conditional diffusion models. Specifically, the ViT-based pathway extracts deterministic visual cues from visible regions, while the diffusion-based pathway synthesizes structurally coherent human body representations. To effectively bridge the two pathways, we design a diverse-consistent feature learning module to align discriminative features with generative priors, and a cross-attention multi-level fusion mechanism to enable bidirectional interaction across semantic levels. Experiments on standard benchmarks demonstrate that our method achieves superior performance on key metrics and shows strong robustness in complex real-world scenarios.
>
---
#### [new 022] Frozen LLMs as Map-Aware Spatio-Temporal Reasoners for Vehicle Trajectory Prediction
- **分类: cs.CV**

- **简介: 该论文属于车辆轨迹预测任务，旨在解决LLMs在自动驾驶中对交通行为和路网拓扑理解不足的问题。通过融合地图信息与轨迹数据，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.21479](https://arxiv.org/pdf/2604.21479)**

> **作者:** Yanjiao Liu; Jiawei Liu; Xun Gong; Zifei Nie
>
> **摘要:** Large language models (LLMs) have recently demonstrated strong reasoning capabilities and attracted increasing research attention in the field of autonomous driving (AD). However, safe application of LLMs on AD perception and prediction still requires a thorough understanding of both the dynamic traffic agents and the static road infrastructure. To this end, this study introduces a framework to evaluate the capability of LLMs in understanding the behaviors of dynamic traffic agents and the topology of road networks. The framework leverages frozen LLMs as the reasoning engine, employing a traffic encoder to extract spatial-level scene features from observed trajectories of agents, while a lightweight Convolutional Neural Network (CNN) encodes the local high-definition (HD) maps. To assess the intrinsic reasoning ability of LLMs, the extracted scene features are then transformed into LLM-compatible tokens via a reprogramming adapter. By residing the prediction burden with the LLMs, a simpler linear decoder is applied to output future trajectories. The framework enables a quantitative analysis of the influence of multi-modal information, especially the impact of map semantics on trajectory prediction accuracy, and allows seamless integration of frozen LLMs with minimal adaptation, thereby demonstrating strong generalizability across diverse LLM architectures and providing a unified platform for model evaluation.
>
---
#### [new 023] SyMTRS: Benchmark Multi-Task Synthetic Dataset for Depth, Domain Adaptation and Super-Resolution in Aerial Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SyMTRS数据集，解决遥感中深度估计、域适应和超分辨率的问题，提供多任务合成数据支持研究。**

- **链接: [https://arxiv.org/pdf/2604.21801](https://arxiv.org/pdf/2604.21801)**

> **作者:** Safouane El Ghazouali; Nicola Venturi; Michael Rueegsegger; Umberto Michelucci
>
> **摘要:** Recent advances in deep learning for remote sensing rely heavily on large annotated datasets, yet acquiring high-quality ground truth for geometric, radiometric, and multi-domain tasks remains costly and often infeasible. In particular, the lack of accurate depth annotations, controlled illumination variations, and multi-scale paired imagery limits progress in monocular depth estimation, domain adaptation, and super-resolution for aerial scenes. We present SyMTRS, a large-scale synthetic dataset generated using a high-fidelity urban simulation pipeline. The dataset provides high-resolution RGB aerial imagery (2048 x 2048), pixel-perfect depth maps, night-time counterparts for domain adaptation, and aligned low-resolution variants for super-resolution at x2, x4, and x8 scales. Unlike existing remote sensing datasets that focus on a single task or modality, SyMTRS is designed as a unified multi-task benchmark enabling joint research in geometric understanding, cross-domain robustness, and resolution enhancement. We describe the dataset generation process, its statistical properties, and its positioning relative to existing benchmarks. SyMTRS aims to bridge critical gaps in remote sensing research by enabling controlled experiments with perfect geometric ground truth and consistent multi-domain supervision. The results obtained in this work can be reproduced from this Github repository: this https URL.
>
---
#### [new 024] A Probabilistic Framework for Improving Dense Object Detection in Underwater Image Data via Annealing-Based Data Augmentation
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在提升水下密集场景中的检测性能。通过引入基于模拟退火的数据增强方法，提高模型在复杂水下环境中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.21198](https://arxiv.org/pdf/2604.21198)**

> **作者:** Eleanor Wiesler; Trace Baxley
>
> **摘要:** Object detection models typically perform well on images captured in controlled environments with stable lighting, water clarity, and viewpoint, but their performance degrades substantially in real-world underwater settings characterized by high variability and frequent occlusions. In this work, we address these challenges by introducing a novel data augmentation framework designed to improve robustness in dense and unconstrained underwater scenes. Using the DeepFish dataset, which contains images of fish in natural environments, we first generate bounding box annotations from provided segmentation masks to construct a custom detection dataset. We then propose a pseudo-simulated annealing-based augmentation algorithm, inspired by the copy-paste strategy of Deng et al. [1], to synthesize realistic crowded fish scenarios. Our approach improves spatial diversity and object density during training, enabling better generalization to complex scenes. Experimental results show that our method significantly outperforms a baseline YOLOv10 model, particularly on a challenging test set of manually annotated images collected from live-stream footage in the Florida Keys. These results demonstrate the effectiveness of our augmentation strategy for improving detection performance in dense, real-world underwater environments.
>
---
#### [new 025] Sapiens2
- **分类: cs.CV**

- **简介: 该论文提出Sapiens2，一个用于人像视觉的高分辨率Transformer模型，解决通用性、多样性和高保真输出问题，通过改进预训练和架构提升性能。**

- **链接: [https://arxiv.org/pdf/2604.21681](https://arxiv.org/pdf/2604.21681)**

> **作者:** Rawal Khirodkar; He Wen; Julieta Martinez; Yuan Dong; Su Zhaoen; Shunsuke Saito
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** We present Sapiens2, a model family of high-resolution transformers for human-centric vision focused on generalization, versatility, and high-fidelity outputs. Our model sizes range from 0.4 to 5 billion parameters, with native 1K resolution and hierarchical variants that support 4K. Sapiens2 substantially improves over its predecessor in both pretraining and post-training. First, to learn features that capture low-level details (for dense prediction) and high-level semantics (for zero-shot or few-label settings), we combine masked image reconstruction with self-distilled contrastive objectives. Our evaluations show that this unified pretraining objective is better suited for a wider range of downstream tasks. Second, along the data axis, we pretrain on a curated dataset of 1 billion high-quality human images and improve the quality and quantity of task annotations. Third, architecturally, we incorporate advances from frontier models that enable longer training schedules with improved stability. Our 4K models adopt windowed attention to reason over longer spatial context and are pretrained with 2K output resolution. Sapiens2 sets a new state-of-the-art and improves over the first generation on pose (+4 mAP), body-part segmentation (+24.3 mIoU), normal estimation (45.6% lower angular error) and extends to new tasks such as pointmap and albedo estimation. Code: this https URL
>
---
#### [new 026] When Prompts Override Vision: Prompt-Induced Hallucinations in LVLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型 hallucinations 问题。通过分析发现，文本先验是主要原因，提出 HalluVL-DPO 方法进行优化，提升模型的视觉一致性。**

- **链接: [https://arxiv.org/pdf/2604.21911](https://arxiv.org/pdf/2604.21911)**

> **作者:** Pegah Khayatan; Jayneel Parekh; Arnaud Dapogny; Mustafa Shukor; Alasdair Newson; Matthieu Cord
>
> **摘要:** Despite impressive progress in capabilities of large vision-language models (LVLMs), these systems remain vulnerable to hallucinations, i.e., outputs that are not grounded in the visual input. Prior work has attributed hallucinations in LVLMs to factors such as limitations of the vision backbone or the dominance of the language component, yet the relative importance of these factors remains unclear. To resolve this ambiguity, We propose HalluScope, a benchmark to better understand the extent to which different factors induce hallucinations. Our analysis indicates that hallucinations largely stem from excessive reliance on textual priors and background knowledge, especially information introduced through textual instructions. To mitigate hallucinations induced by textual instruction priors, we propose HalluVL-DPO, a framework for fine-tuning off-the-shelf LVLMs towards more visually grounded responses. HalluVL-DPO leverages preference optimization using a curated training dataset that we construct, guiding the model to prefer grounded responses over hallucinated ones. We demonstrate that our optimized model effectively mitigates the targeted hallucination failure mode, while preserving or improving performance on other hallucination benchmarks and visual capability evaluations. To support reproducibility and further research, we will publicly release our evaluation benchmark, preference training dataset, and code at this https URL .
>
---
#### [new 027] Directional Confusions Reveal Divergent Inductive Biases Through Rate-Distortion Geometry in Human and Machine Vision
- **分类: cs.CV; cs.IT; q-bio.NC**

- **简介: 论文研究人类与机器视觉在分类任务中的错误差异，通过分析混淆矩阵的不对称性，揭示不同的归纳偏置。任务为图像分类，解决错误模式差异问题，利用率失真框架量化几何特征。**

- **链接: [https://arxiv.org/pdf/2604.21909](https://arxiv.org/pdf/2604.21909)**

> **作者:** Leyla Roksan Caglar; Pedro A.M. Mediano; Baihan Lin
>
> **摘要:** Humans and modern vision models can reach similar classification accuracy while making systematically different kinds of mistakes - differing not in how often they err, but in who gets mistaken for whom, and in which direction. We show that these directional confusions reveal distinct inductive biases that are invisible to accuracy alone. Using matched human and deep vision model responses on a natural-image categorization task under 12 perturbation types, we quantify asymmetry in confusion matrices and link it to generalization geometry through a Rate-Distortion (RD) framework, summarized by three geometric signatures (slope (beta), curvature (kappa)) and efficiency (AUC). We find that humans exhibit broad but weak asymmetries, whereas deep vision models show sparser, stronger directional collapses. Robustness training reduces global asymmetry but fails to recover the human-like breadth-strength profile of graded similarity. Mechanistic simulations further show that different asymmetry organizations shift the RD frontier in opposite directions, even when matched for performance. Together, these results position directional confusions and RD geometry as compact, interpretable signatures of inductive bias under distribution shift.
>
---
#### [new 028] DualSplat: Robust 3D Gaussian Splatting via Pseudo-Mask Bootstrapping from Reconstruction Failures
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决多视图一致性破坏下的重建问题。通过利用重建失败生成伪掩码，提升动态物体场景的重建效果。**

- **链接: [https://arxiv.org/pdf/2604.21631](https://arxiv.org/pdf/2604.21631)**

> **作者:** Xu Wang; Zhiru Wang; Shiyun Xie; Chengwei Pan; Yisong Chen
>
> **备注:** 10 pages,6 figures, accepted to Computer Vision and Pattern Recognition Conference 2026
>
> **摘要:** While 3D Gaussian Splatting (3DGS) achieves real-time photorealistic rendering, its performance degrades significantly when training images contain transient objects that violate multi-view consistency. Existing methods face a circular dependency: accurate transient detection requires a well-reconstructed static scene, while clean reconstruction itself depends on reliable transient masks. We address this challenge with DualSplat, a Failure-to-Prior framework that converts first-pass reconstruction failures into explicit priors for a second reconstruction stage. We observe that transients, which appear in only a subset of views, often manifest as incomplete fragments during conservative initial training. We exploit these failures to construct object-level pseudo-masks by combining photometric residuals, feature mismatches, and SAM2 instance boundaries. These pseudo-masks then guide a clean second-pass 3DGS optimization, while a lightweight MLP refines them online by gradually shifting from prior supervision to self-consistency. Experiments on RobustNeRF and NeRF On-the-go show that DualSplat outperforms existing baselines, demonstrating particularly clear advantages in transient-heavy scenes and transient regions.
>
---
#### [new 029] Leveraging Multimodal LLMs for Built Environment and Housing Attribute Assessment from Street-View Imagery
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于建筑环境评估任务，旨在通过街景图像自动评估房屋状况。工作包括使用多模态LLM进行模型训练与优化，提升评估效率和准确性。**

- **链接: [https://arxiv.org/pdf/2604.21102](https://arxiv.org/pdf/2604.21102)**

> **作者:** Siyuan Yao; Siavash Ghorbany; Kuangshi Ai; Arnav Cherukuthota; Meghan Forstchen; Alexis Korotasz; Matthew Sisk; Ming Hu; Chaoli Wang
>
> **摘要:** We present a novel framework for automatically evaluating building conditions nationwide in the United States by leveraging large language models (LLMs) and Google Street View (GSV) imagery. By fine-tuning Gemma 3 27B on a modest human-labeled dataset, our approach achieves strong alignment with human mean opinion scores (MOS), outperforming even individual raters on SRCC and PLCC relative to the MOS benchmark. To enhance efficiency, we apply knowledge distillation, transferring the capabilities of Gemma 3 27B to a smaller Gemma 3 4B model that achieves comparable performance with a 3x speedup. Further, we distill the knowledge into a CNN-based model (EfficientNetV2-M) and a transformer (SwinV2-B), delivering close performance while achieving a 30x speed gain. Furthermore, we investigate LLMs' capabilities for assessing an extensive list of built environment and housing attributes through a human-AI alignment study and develop a visualization dashboard that integrates LLM assessment outcomes for downstream analysis by homeowners. Our framework offers a flexible and efficient solution for large-scale building condition assessment, enabling high accuracy with minimal human labeling effort.
>
---
#### [new 030] SpatiO: Adaptive Test-Time Orchestration of Vision-Language Agents for Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言空间推理任务，旨在解决现有方法在空间适应性上的不足。提出SpatiO框架和TTO机制，通过多代理协作提升推理性能。**

- **链接: [https://arxiv.org/pdf/2604.21190](https://arxiv.org/pdf/2604.21190)**

> **作者:** Chan Yeong Hwang; Miso Choi; Sunghyun On; Jinkyu Kim; Jungbeom Lee
>
> **备注:** Technical report
>
> **摘要:** Understanding visual scenes requires not only recognizing objects but also reasoning about their spatial relationships. Unlike general vision-language tasks, spatial reasoning requires integrating multiple inductive biases, such as 2D appearance cues, depth signals, and geometric constraints, whose reliability varies across contexts. This suggests that effective spatial reasoning requires \emph{spatial adaptability}: the ability to flexibly coordinate different reasoning strategies depending on the input. However, most existing approaches rely on a single reasoning pipeline that implicitly learns a fixed spatial prior, limiting their ability to adapt under distribution changes. Multi-agent systems offer a promising alternative by aggregating diverse reasoning trajectories, but prior attempts in spatial reasoning primarily employ homogeneous agents, restricting the diversity of inductive biases they can leverage. In this work, we introduce \textbf{\textsc{SpatiO}}, a heterogeneous multi-agent framework for spatial reasoning that coordinates multiple vision-language specialists with complementary inductive biases. To enable effective collaboration, we propose \textbf{Test-Time Orchestration (TTO)}, an optimization mechanism that dynamically evaluates and reweights agents based on their observed reliability during inference, without modifying model parameters. Extensive experiments on diverse spatial reasoning benchmarks, including 3DSRBench, STVQA-7k, CV-Bench, and Omni3D-Bench, demonstrate that \textsc{SpatiO} consistently improves spatial reasoning performance over both closed-source and open-source baselines.
>
---
#### [new 031] VARestorer: One-Step VAR Distillation for Real-World Image Super-Resolution
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像超分辨率任务，解决VAR模型在真实场景下生成模糊、不一致的问题。通过单步蒸馏和跨尺度注意力机制提升效果与效率。**

- **链接: [https://arxiv.org/pdf/2604.21450](https://arxiv.org/pdf/2604.21450)**

> **作者:** Yixuan Zhu; Shilin Ma; Haolin Wang; Ao Li; Yanzhe Jing; Yansong Tang; Lei Chen; Jiwen Lu; Jie Zhou
>
> **备注:** Accepted in ICLR 2026. Code is available at this https URL
>
> **摘要:** Recent advancements in visual autoregressive models (VAR) have demonstrated their effectiveness in image generation, highlighting their potential for real-world image super-resolution (Real-ISR). However, adapting VAR for ISR presents critical challenges. The next-scale prediction mechanism, constrained by causal attention, fails to fully exploit global low-quality (LQ) context, resulting in blurry and inconsistent high-quality (HQ) outputs. Additionally, error accumulation in the iterative prediction severely degrades coherence in ISR task. To address these issues, we propose VARestorer, a simple yet effective distillation framework that transforms a pre-trained text-to-image VAR model into a one-step ISR model. By leveraging distribution matching, our method eliminates the need for iterative refinement, significantly reducing error propagation and inference time. Furthermore, we introduce pyramid image conditioning with cross-scale attention, which enables bidirectional scale-wise interactions and fully utilizes the input image information while adapting to the autoregressive mechanism. This prevents later LQ tokens from being overlooked in the transformer. By fine-tuning only 1.2\% of the model parameters through parameter-efficient adapters, our method maintains the expressive power of the original VAR model while significantly enhancing efficiency. Extensive experiments show that VARestorer achieves state-of-the-art performance with 72.32 MUSIQ and 0.7669 CLIPIQA on DIV2K dataset, while accelerating inference by 10 times compared to conventional VAR inference.
>
---
#### [new 032] UHR-DETR: Efficient End-to-End Small Object Detection for Ultra-High-Resolution Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文属于小目标检测任务，针对超高清遥感图像中因分辨率高导致的内存瓶颈问题，提出UHR-DETR模型，实现高效端到端检测。**

- **链接: [https://arxiv.org/pdf/2604.21435](https://arxiv.org/pdf/2604.21435)**

> **作者:** Jingfang Li; Haoran Zhu; Wen Yang; Jinrui Zhang; Fang Xu; Haijian Zhang; Gui-Song Xia
>
> **摘要:** Ultra-High-Resolution (UHR) imagery has become essential for modern remote sensing, offering unprecedented spatial coverage. However, detecting small objects in such vast scenes presents a critical dilemma: retaining the original resolution for small objects causes prohibitive memory bottlenecks. Conversely, conventional compromises like image downsampling or patch cropping either erase small objects or destroy context. To break this dilemma, we propose UHR-DETR, an efficient end-to-end transformer-based detector designed for UHR imagery. First, we introduce a Coverage-Maximizing Sparse Encoder that dynamically allocates finite computational resources to informative high-resolution regions, ensuring maximum object coverage with minimal spatial redundancy. Second, we design a Global-Local Decoupled Decoder. By integrating macroscopic scene awareness with microscopic object details, this module resolves semantic ambiguities and prevents scene fragmentation. Extensive experiments on the UHR imagery datasets (e.g., STAR and SODA-A) demonstrate the superiority of UHR-DETR under strict hardware constraints (e.g., a single 24GB RTX 3090). It achieves a 2.8\% mAP improvement while delivering a 10$\times$ inference speedup compared to standard sliding-window baselines on the STAR dataset. Our codes and models will be available at this https URL.
>
---
#### [new 033] WildSplatter: Feed-forward 3D Gaussian Splatting with Appearance Control from Unconstrained Images
- **分类: cs.CV**

- **简介: 该论文提出WildSplatter，属于3D重建任务，解决无约束图像下的3D高斯溅射问题，通过联合学习3D高斯和外观嵌入，实现快速且可控的渲染。**

- **链接: [https://arxiv.org/pdf/2604.21182](https://arxiv.org/pdf/2604.21182)**

> **作者:** Yuki Fujimura; Takahiro Kushida; Kazuya Kitano; Takuya Funatomi; Yasuhiro Mukaigawa
>
> **备注:** Project page: this https URL
>
> **摘要:** We propose WildSplatter, a feed-forward 3D Gaussian Splatting (3DGS) model for unconstrained images with unknown camera parameters and varying lighting conditions. 3DGS is an effective scene representation that enables high-quality, real-time rendering; however, it typically requires iterative optimization and multi-view images captured under consistent lighting with known camera parameters. WildSplatter is trained on unconstrained photo collections and jointly learns 3D Gaussians and appearance embeddings conditioned on input images. This design enables flexible modulation of Gaussian colors to represent significant variations in lighting and appearance. Our method reconstructs 3D Gaussians from sparse input views in under one second, while also enabling appearance control under diverse lighting conditions. Experimental results demonstrate that our approach outperforms existing pose-free 3DGS methods on challenging real-world datasets with varying illumination.
>
---
#### [new 034] Gmd: Gaussian mixture descriptor for pair matching of 3D fragments
- **分类: cs.CV**

- **简介: 该论文属于3D碎片配准任务，解决碎片表面匹配问题。提出GMD描述子，通过GMM建模点云分布，实现碎片的高效匹配与重建。**

- **链接: [https://arxiv.org/pdf/2604.21519](https://arxiv.org/pdf/2604.21519)**

> **作者:** Meijun Xiong; Zhenguo Shi; Xinyu Zhou; Yuhe Zhang; Shunli Zhang
>
> **备注:** 24 pages, 10 figures. Published in Multimedia Systems
>
> **摘要:** In the automatic reassembly of fragments acquired using laser scanners to reconstruct objects, a crucial step is the matching of fractured surfaces. In this paper, we propose a novel local descriptor that uses the Gaussian Mixture Model (GMM) to fit the distribution of points, allowing for the description and matching of fractured surfaces of fragments. Our method involves dividing a local surface patch into concave and convex regions for estimating the k value of GMM. Then the final Gaussian Mixture Descriptor (GMD) of the fractured surface is formed by merging the regional GMDs. To measure the similarities between GMDs for determining adjacent fragments, we employ the L2 distance and align the fragments using Random Sample Consensus (RANSAC) and Iterative Closest Point (ICP). The extensive experiments on real-scanned public datasets and Terracotta datasets demonstrate the effectiveness of our approach; furthermore, the comparisons with several existing methods also validate the advantage of the proposed method.
>
---
#### [new 035] Optimizing Diffusion Priors with a Single Observation
- **分类: cs.CV; cs.LG; stat.ME**

- **简介: 该论文属于图像逆问题任务，解决扩散先验在少量观测下过拟合的问题。通过结合多个先验并优化指数权重，提升后验样本的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.21066](https://arxiv.org/pdf/2604.21066)**

> **作者:** Frederic Wang; Katherine L. Bouman
>
> **摘要:** While diffusion priors generate high-quality posterior samples across many inverse problems, they are often trained on limited training sets or purely simulated data, thus inheriting the errors and biases of these underlying sources. Current approaches to finetuning diffusion models rely on a large number of observations with varying forward operators, which can be difficult to collect for many applications, and thus lead to overfitting when the measurement set is small. We propose a method for tuning a prior from only a single observation by combining existing diffusion priors into a single product-of-experts prior and identifying the exponents that maximize the Bayesian evidence. We validate our method on real-world inverse problems, including black hole imaging, where the true prior is unknown a priori, and image deblurring with text-conditioned priors. We find that the evidence is often maximized by priors that extend beyond those trained on a single dataset. By generalizing the prior through exponent weighting, our approach enables posterior sampling from both tempered and combined diffusion models, yielding more flexible priors that improve the trustworthiness of the resulting posterior image distribution.
>
---
#### [new 036] AttDiff-GAN: A Hybrid Diffusion-GAN Framework for Facial Attribute Editing
- **分类: cs.CV**

- **简介: 该论文属于人脸属性编辑任务，旨在解决GAN与扩散模型在属性对齐和编辑精度上的不足。提出AttDiff-GAN框架，结合两者优势，提升编辑准确性和图像质量。**

- **链接: [https://arxiv.org/pdf/2604.21289](https://arxiv.org/pdf/2604.21289)**

> **作者:** Wenmin Huang; Weiqi Luo; Xiaochun Cao; Jiwu Huang
>
> **摘要:** Facial attribute editing aims to modify target attributes while preserving attribute-irrelevant content and overall image fidelity. Existing GAN-based methods provide favorable controllability, but often suffer from weak alignment between style codes and attribute semantics. Diffusion-based methods can synthesize highly realistic images; however, their editing precision is limited by the entanglement of semantic directions among different attributes. In this paper, we propose AttDiff-GAN, a hybrid framework that combines GAN-based attribute manipulation with diffusion-based image generation. A key challenge in such integration lies in the inconsistency between one-step adversarial learning and multi-step diffusion denoising, which makes effective optimization difficult. To address this issue, we decouple attribute editing from image synthesis by introducing a feature-level adversarial learning scheme to learn explicit attribute manipulation, and then using the manipulated features to guide the diffusion process for image generation, while also removing the reliance on semantic direction-based editing. Moreover, we enhance style-attribute alignment by introducing PriorMapper, which incorporates facial priors into style generation, and RefineExtractor, which captures global semantic relationships through a Transformer for more precise style extraction. Experimental results on CelebA-HQ show that the proposed method achieves more accurate facial attribute editing and better preservation of non-target attributes than state-of-the-art methods in both qualitative and quantitative evaluations.
>
---
#### [new 037] Clinically-Informed Modeling for Pediatric Brain Tumor Classification from Whole-Slide Histopathology Images
- **分类: cs.CV**

- **简介: 该论文属于儿科脑肿瘤分类任务，解决数据稀缺、类别不平衡和细粒度区分问题。通过引入专家引导的对比微调框架，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.21060](https://arxiv.org/pdf/2604.21060)**

> **作者:** Joakim Nguyen; Jian Yu; Jinrui Fang; Nicholas Konz; Tianlong Chen; Sanjay Krishnan; Chandra Krishnan; Ying Ding; Hairong Wang; Ankita Shukla
>
> **备注:** Accepted at the IEEE International Conference on Healthcare Informatics (ICHI), 2026
>
> **摘要:** Accurate diagnosis of pediatric brain tumors, starting with histopathology, presents unique challenges for deep learning, including severe data scarcity, class imbalance, and fine-grained morphologic overlap across diagnostically distinct subtypes. While pathology foundation models have advanced patch-level representation learning, their effective adaptation to weakly supervised pediatric brain tumor classification under limited data remains underexplored. In this work, we introduce an expert-guided contrastive fine-tuning framework for pediatric brain tumor diagnosis from whole-slide images (WSI). Our approach integrates contrastive learning into slide-level multiple instance learning (MIL) to explicitly regularize the geometry of slide-level representations during downstream fine-tuning. We propose both a general supervised contrastive setting and an expert-guided variant that incorporates clinically informed hard negatives targeting diagnostically confusable subtypes. Through comprehensive experiments on pediatric brain tumor WSI classification under realistic low-sample and class-imbalanced conditions, we demonstrate that contrastive fine-tuning yields measurable improvements in fine-grained diagnostic distinctions. Our experimental analyses reveal complementary strengths across different contrastive strategies, with expert-guided hard negatives promoting more compact intra-class representations and improved inter-class separation. This work highlights the importance of explicitly shaping slide-level representations for robust fine-grained classification in data-scarce pediatric pathology settings.
>
---
#### [new 038] Reshoot-Anything: A Self-Supervised Model for In-the-Wild Video Reshooting
- **分类: cs.CV**

- **简介: 该论文属于视频重拍摄任务，解决非刚性场景中多视角数据稀缺问题。通过自监督框架生成伪多视角数据，提升视频重拍摄的时空一致性与视图合成质量。**

- **链接: [https://arxiv.org/pdf/2604.21776](https://arxiv.org/pdf/2604.21776)**

> **作者:** Avinash Paliwal; Adithya Iyer; Shivin Yadav; Muhammad Ali Afridi; Midhun Harikumar
>
> **摘要:** Precise camera control for reshooting dynamic videos is bottlenecked by the severe scarcity of paired multi-view data for non-rigid scenes. We overcome this limitation with a highly scalable self-supervised framework capable of leveraging internet-scale monocular videos. Our core contribution is the generation of pseudo multi-view training triplets, consisting of a source video, a geometric anchor, and a target video. We achieve this by extracting distinct smooth random-walk crop trajectories from a single input video to serve as the source and target views. The anchor is synthetically generated by forward-warping the first frame of the source with a dense tracking field, which effectively simulates the distorted point-cloud inputs expected at inference. Because our independent cropping strategy introduces spatial misalignment and artificial occlusions, the model cannot simply copy information from the current source frame. Instead, it is forced to implicitly learn 4D spatiotemporal structures by actively routing and re-projecting missing high-fidelity textures across distinct times and viewpoints from the source video to reconstruct the target. At inference, our minimally adapted diffusion transformer utilizes a 4D point-cloud derived anchor to achieve state-of-the-art temporal consistency, robust camera control, and high-fidelity novel view synthesis on complex dynamic scenes.
>
---
#### [new 039] VG-CoT: Towards Trustworthy Visual Reasoning via Grounded Chain-of-Thought
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VG-CoT数据集，解决LVLMs在视觉推理中的可信度问题，通过自动化流程连接推理步骤与视觉证据。**

- **链接: [https://arxiv.org/pdf/2604.21396](https://arxiv.org/pdf/2604.21396)**

> **作者:** Byeonggeuk Lim; Kyeonghyun Kim; JungMin Yun; YoungBin Kim
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** The advancement of Large Vision-Language Models (LVLMs) requires precise local region-based reasoning that faithfully grounds the model's logic in actual visual evidence. However, existing datasets face limitations in scalability due to extensive manual annotation and lack of explicit alignment between multi-step reasoning and corresponding image regions, which constrains the evaluation of model trustworthiness. To address these challenges, we propose the Visual Grounding Chain-of-Thought (VG-CoT) dataset, which explicitly links each reasoning step to real visual evidence within the image through a fully automated three-stage pipeline. The pipeline first extracts object- and text-level visual evidence using state-of-the-art detection and OCR models, then generates step-by-step grounded reasoning with GPT-4o, and finally refines the grounding through a rationale-driven open-set detection process. In addition, we introduce a new benchmark that comprehensively evaluates LVLMs reasoning across three complementary dimensions: Rationale Quality, Answer Accuracy, and Reasoning-Answer Alignment. Experiments with representative LVLMs, including LLaVA-1.5 and Qwen2-VL, demonstrate consistent improvements on most evaluation metrics, confirming that VG-CoT effectively enhances trustworthy, evidence-based reasoning while maintaining scalable and cost-efficient dataset construction. The dataset and code will be released publicly upon acceptance to facilitate further research.
>
---
#### [new 040] Pretrain Where? Investigating How Pretraining Data Diversity Impacts Geospatial Foundation Model Performance
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究预训练数据多样性对地理基础模型性能的影响，旨在解决数据多样性与模型表现关系的问题。通过构建不同区域的预训练数据集并分析其效果，发现光谱多样性对性能影响显著。**

- **链接: [https://arxiv.org/pdf/2604.21104](https://arxiv.org/pdf/2604.21104)**

> **作者:** Amandeep Kaur; Mirali Purohit; Gedeon Muhawenayo; Esther Rolf; Hannah Kerner
>
> **备注:** Accepted at EarthVision workshop, CVPR 2026
>
> **摘要:** New geospatial foundation models introduce a new model architecture and pretraining dataset, often sampled using different notions of data diversity. Performance differences are largely attributed to the model architecture or input modalities, while the role of the pretraining dataset is rarely studied. To address this research gap, we conducted a systematic study on how the geographic composition of pretraining data affects a model's downstream performance. We created global and per-continent pretraining datasets and evaluated them on global and per-continent downstream datasets. We found that the pretraining dataset from Europe outperformed global and continent-specific pretraining datasets on both global and local downstream evaluations. To investigate the factors influencing a pretraining dataset's downstream performance, we analysed 10 pretraining datasets using diversity across continents, biomes, landcover and spectral values. We found that only spectral diversity was strongly correlated with performance, while others were weakly correlated. This finding establishes a new dimension of diversity to be accounted for when creating a high-performing pretraining dataset. We open-sourced 7 new pretraining datasets, pretrained models, and our experimental framework at this https URL.
>
---
#### [new 041] From Codebooks to VLMs: Evaluating Automated Visual Discourse Analysis for Climate Change on Social Media
- **分类: cs.CV**

- **简介: 该论文属于视觉话语分析任务，旨在评估计算机视觉方法在社交媒体气候传播分析中的应用。通过模型对比与提示工程，提升对图像内容的识别与趋势分析能力。**

- **链接: [https://arxiv.org/pdf/2604.21786](https://arxiv.org/pdf/2604.21786)**

> **作者:** Katharina Prasse; Steffen Jung; Isaac Bravo; Stefanie Walter; Patrick Knab; Christian Bartelt; Margret Keuper
>
> **摘要:** Social media platforms have become primary arenas for climate communication, generating millions of images and posts that - if systematically analysed - can reveal which communication strategies mobilise public concern and which fall flat. We aim to facilitate such research by analysing how computer vision methods can be used for social media discourse analysis. This analysis includes application-based taxonomy design, model selection, prompt engineering, and validation. We benchmark six promptable vision-language models and 15 zero-shot CLIP-like models on two datasets from X (formerly Twitter) - a 1,038-image expert-annotated set and a larger corpus of over 1.2 million images, with 50,000 labels manually validated - spanning five annotation dimensions: animal content, climate change consequences, climate action, image setting, and image type. Among the models benchmarked, Gemini-3.1-flash-lite outperforms all others across all super-categories and both datasets, while the gap to open-weight models of moderate size remains relatively small. Beyond instance-level metrics, we advocate for distributional evaluation: VLM predictions can reliably recover population level trends even when per-image accuracy is moderate, making them a viable starting point for discourse analysis at scale. We find that chain-of-thought reasoning reduces rather than improves performance, and that annotation dimension specific prompt design improves performance. We release tweet IDs and labels along with our code at this https URL.
>
---
#### [new 042] WorldMark: A Unified Benchmark Suite for Interactive Video World Models
- **分类: cs.CV**

- **简介: 该论文提出WorldMark，一个统一的交互视频世界模型基准测试套件，解决跨模型公平比较问题，通过标准化场景、动作序列和控制接口，支持多模型评估与对比。**

- **链接: [https://arxiv.org/pdf/2604.21686](https://arxiv.org/pdf/2604.21686)**

> **作者:** Xiaojie Xu; Zhengyuan Lin; Kang He; Yukang Feng; Xiaofeng Mao; Yuanyang Yin; Kaipeng Zhang; Yongtao Ge
>
> **摘要:** Interactive video generation models such as Genie, YUME, HY-World, and Matrix-Game are advancing rapidly, yet every model is evaluated on its own benchmark with private scenes and trajectories, making fair cross-model comparison impossible. Existing public benchmarks offer useful metrics such as trajectory error, aesthetic scores, and VLM-based judgments, but none supplies the standardized test conditions -- identical scenes, identical action sequences, and a unified control interface -- needed to make those metrics comparable across models with heterogeneous inputs. We introduce WorldMark, the first benchmark that provides such a common playing field for interactive Image-to-Video world models. WorldMark contributes: (1) a unified action-mapping layer that translates a shared WASD-style action vocabulary into each model's native control format, enabling apples-to-apples comparison across six major models on identical scenes and trajectories; (2) a hierarchical test suite of 500 evaluation cases covering first- and third-person viewpoints, photorealistic and stylized scenes, and three difficulty tiers from Easy to Hard spanning 20-60s; and (3) a modular evaluation toolkit for Visual Quality, Control Alignment, and World Consistency, designed so that researchers can reuse our standardized inputs while plugging in their own metrics as the field evolves. We will release all data, evaluation code, and model outputs to facilitate future research. Beyond offline metrics, we launch World Model Arena (this http URL), an online platform where anyone can pit leading world models against each other in side-by-side battles and watch the live leaderboard.
>
---
#### [new 043] Local Neighborhood Instability in Parametric Projections: Quantitative and Visual Analysis
- **分类: cs.CV**

- **简介: 该论文研究参数化投影的局部稳定性问题，通过量化分析与可视化方法评估投影的稳定性，解决因数据变化导致的布局不稳定问题。**

- **链接: [https://arxiv.org/pdf/2604.21617](https://arxiv.org/pdf/2604.21617)**

> **作者:** Frederik L. Dennig; Daniel A. Keim
>
> **备注:** 6 pages, 3 figures, LaTeX, to appear at the 17th International EuroVis Workshop on Visual Analytics
>
> **摘要:** Parametric projections let analysts embed new points in real time, but input variations from measurement noise or data drift can produce unpredictable shifts in the 2D layout. Whether and where a projection is locally stable remains largely unexamined. In this paper, we present a stability evaluation framework that probes parametric projections with Gaussian perturbations around selected anchor points and assesses how neighborhoods deform in the 2D embedding. Our approach combines quantitative measures of mean displacement, bias, and nearest-anchor assignment error with per-anchor visualizations of displacement vectors, local PCA ellipsoids, and Voronoi misassignment for detailed inspection. We demonstrate the framework's effectiveness on UMAP- and t-SNE-based neural projectors of varying network sizes and study the effect of Jacobian regularization as a gradient-based robustness strategy. We apply our framework to the MNIST and Fashion-MNIST datasets. The results show that our framework identifies unstable projection regions invisible to reconstruction error or neighborhood-preservation metrics.
>
---
#### [new 044] Context Unrolling in Omni Models
- **分类: cs.CV**

- **简介: 该论文提出Omni模型，解决多模态任务中的信息融合问题。通过联合训练多种模态，实现跨模态推理，提升生成与理解性能。**

- **链接: [https://arxiv.org/pdf/2604.21921](https://arxiv.org/pdf/2604.21921)**

> **作者:** Ceyuan Yang; Zhijie Lin; Yang Zhao; Fei Xiao; Hao He; Qi Zhao; Chaorui Deng; Kunchang Li; Zihan Ding; Yuwei Guo; Fuyun Wang; Fangqi Zhu; Xiaonan Nie; Shenhan Zhu; Shanchuan Lin; Hongsheng Li; Weilin Huang; Guang Shi; Haoqi Fan
>
> **备注:** Report
>
> **摘要:** We present Omni, a unified multimodal model natively trained on diverse modalities, including text, images, videos, 3D geometry, and hidden representations. We find that such training enables Context Unrolling, where the model explicitly reasons across multiple modal representations before producing predictions. This process enables the model to aggregate complementary information across heterogeneous modalities, facilitating a more faithful approximation of the shared multimodal knowledge manifold and improving downstream reasoning fidelity. As a result, Omni achieves strong performance on both multimodal generation and understanding benchmarks, while demonstrating advanced multimodal reasoning capabilities, including in-context generation of text, image, video, and 3D geometry.
>
---
#### [new 045] Prototype-Based Test-Time Adaptation of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的测试时自适应任务，解决传统方法效率低和性能不佳的问题，提出基于原型的TTA方法PTA，提升模型在多个基准上的表现。**

- **链接: [https://arxiv.org/pdf/2604.21360](https://arxiv.org/pdf/2604.21360)**

> **作者:** Zhaohong Huang; Yuxin Zhang; Wenjing Liu; Fei Chao; Rongrong Ji
>
> **摘要:** Test-time adaptation (TTA) has emerged as a promising paradigm for vision-language models (VLMs) to bridge the distribution gap between pre-training and test data. Recent works have focused on backpropagation-free TTA methods that rely on cache-based designs, but these introduce two key limitations. First, inference latency increases as the cache grows with the number of classes, leading to inefficiencies in large-scale settings. Second, suboptimal performance occurs when the cache contains insufficient or incorrect samples. In this paper, we present Prototype-Based Test-Time Adaptation (PTA), an efficient and effective TTA paradigm that uses a set of class-specific knowledge prototypes to accumulate knowledge from test samples. Particularly, knowledge prototypes are adaptively weighted based on the zero-shot class confidence of each test sample, incorporating the sample's visual features into the corresponding class-specific prototype. It is worth highlighting that the knowledge from past test samples is integrated and utilized solely in the prototypes, eliminating the overhead of cache population and retrieval that hinders the efficiency of existing TTA methods. This endows PTA with extremely high efficiency while achieving state-of-the-art performance on 15 image recognition benchmarks and 4 robust point cloud analysis benchmarks. For example, PTA improves CLIP's accuracy from 65.64% to 69.38% on 10 cross-domain benchmarks, while retaining 92% of CLIP's inference speed on large-scale ImageNet-1K. In contrast, the cache-based TDA achieves a lower accuracy of 67.97% and operates at only 50% of CLIP's inference speed.
>
---
#### [new 046] Ramen: Robust Test-Time Adaptation of Vision-Language Models with Active Sample Selection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言模型的测试时适应任务，解决混合领域下模型性能下降的问题。提出Ramen框架，通过主动样本选择提升适应效果。**

- **链接: [https://arxiv.org/pdf/2604.21728](https://arxiv.org/pdf/2604.21728)**

> **作者:** Wenxuan Bao; Yanjun Zhao; Xiyuan Yang; Jingrui He
>
> **备注:** Accepted by CVPR 2026 (Findings Track)
>
> **摘要:** Pretrained vision-language models such as CLIP exhibit strong zero-shot generalization but remain sensitive to distribution shifts. Test-time adaptation adapts models during inference without access to source data or target labels, offering a practical way to handle such shifts. However, existing methods typically assume that test samples come from a single, consistent domain, while in practice, test data often include samples from mixed domains with distinct characteristics. Consequently, their performance degrades under mixed-domain settings. To address this, we present Ramen, a framework for robust test-time adaptation through active sample selection. For each incoming test sample, Ramen retrieves a customized batch of relevant samples from previously seen data based on two criteria: domain consistency, which ensures that adaptation focuses on data from similar domains, and prediction balance, which mitigates adaptation bias caused by skewed predictions. To improve efficiency, Ramen employs an embedding-gradient cache that stores the embeddings and sample-level gradients of past test images. The stored embeddings are used to retrieve relevant samples, and the corresponding gradients are aggregated for model updates, eliminating the need for any additional forward or backward passes. Our theoretical analysis provides insight into why the proposed adaptation mechanism is effective under mixed-domain shifts. Experiments on multiple image corruption and domain-shift benchmarks demonstrate that Ramen achieves strong and consistent performance, offering robust and efficient adaptation in complex mixed-domain scenarios. Our code is available at this https URL .
>
---
#### [new 047] WFM: 3D Wavelet Flow Matching for Ultrafast Multi-Modal MRI Synthesis
- **分类: cs.CV**

- **简介: 该论文提出WFM方法，用于快速多模态MRI合成。针对扩散模型计算成本高、效率低的问题，通过波形空间的直接流匹配实现高效生成，仅需1-2步即可完成高质量合成。**

- **链接: [https://arxiv.org/pdf/2604.21146](https://arxiv.org/pdf/2604.21146)**

> **作者:** Yalcin Tur; Mihajlo Stojkovic; Ulas Bagci
>
> **备注:** 17 pages, 4 figures, 3 tables. Accepted at MIDL 2026 (Poster)
>
> **摘要:** Diffusion models have achieved remarkable quality in multi-modal MRI synthesis, but their computational cost (hundreds of sampling steps and separate models per modality) limits clinical deployment. We observe that this inefficiency stems from an unnecessary starting point: diffusion begins from pure noise, discarding the structural information already present in available MRI sequences. We propose WFM (Wavelet Flow Matching), which instead learns a direct flow from an informed prior, the mean of conditioning modalities in wavelet space, to the target distribution. Because the source and target share underlying anatomy and differ primarily in contrast, this formulation enables accurate synthesis in just 1-2 integration steps. A single 82M-parameter model with class conditioning synthesizes all four BraTS modalities (T1, T1c, T2, FLAIR), replacing four separate diffusion models totaling 326M parameters. On BraTS 2024, WFM achieves 26.8 dB PSNR and 0.94 SSIM, within 1-2 dB of diffusion baselines, while running 250-1000x faster (0.16-0.64s vs. 160s per volume). This speed-quality trade-off makes real-time MRI synthesis practical for clinical workflows. Code is available at this https URL.
>
---
#### [new 048] Sparse Forcing: Native Trainable Sparse Attention for Real-time Autoregressive Diffusion Video Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Sparse Forcing，用于实时自回归扩散视频生成任务，解决长序列生成质量与计算效率问题，通过引入可训练的稀疏注意力机制提升效果并加速解码。**

- **链接: [https://arxiv.org/pdf/2604.21221](https://arxiv.org/pdf/2604.21221)**

> **作者:** Boxun Xu; Yuming Du; Zichang Liu; Siyu Yang; Ziyang Jiang; Siqi Yan; Rajasi Saha; Albert Pumarola; Wenchen Wang; Peng Li
>
> **摘要:** We introduce Sparse Forcing, a training-and-inference paradigm for autoregressive video diffusion models that improves long-horizon generation quality while reducing decoding latency. Sparse Forcing is motivated by an empirical observation in autoregressive diffusion rollouts: attention concentrates on a persistent subset of salient visual blocks, forming an implicit spatiotemporal memory in the KV cache, and exhibits a locally structured block-sparse pattern within sliding windows. Building on this observation, we propose a trainable native sparsity mechanism that learns to compress, preserve, and update these persistent blocks while restricting computation within each local window to a dynamically selected local neighborhood. To make the approach practical at scale for both training and inference, we further propose Persistent Block-Sparse Attention (PBSA), an efficient GPU kernel that accelerates sparse attention and memory updates for low-latency, memory-efficient decoding. Experiments show that Sparse Forcing improves the VBench score by +0.26 over Self-Forcing on 5-second text-to-video generation while delivering a 1.11-1.17x decoding speedup and 42% lower peak KV-cache footprint. The gains are more pronounced on longer-horizon rollouts, delivering improved visual quality with +0.68 and +2.74 VBench improvements, and 1.22x and 1.27x speedups on 20-second and 1-minute generations, respectively.
>
---
#### [new 049] EdgeFormer: local patch-based edge detection transformer on point clouds
- **分类: cs.CV**

- **简介: 该论文属于点云边缘检测任务，旨在解决细粒度边缘特征难以有效检测的问题。通过构建局部块特征描述符并进行分类，提出EdgeFormer网络提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.21387](https://arxiv.org/pdf/2604.21387)**

> **作者:** Yifei Xie; Zhikun Tu; Tong Yang; Yuhe Zhang; Xinyu Zhou
>
> **备注:** 22 pages, 9 figures. Published in Pattern Analysis and Applications
>
> **摘要:** Edge points on 3D point clouds can clearly convey 3D geometry and surface characteristics, therefore, edge detection is widely used in many vision applications with high industrial and commercial demands. However, the fine-grained edge features are difficult to detect effectively as they are generally densely distributed or exhibit small-scale surface gradients. To address this issue, we present a learning-based edge detection network, named EdgeFormer, which mainly consists of two stages. Based on the observation that spatially neighboring points tend to exhibit high correlation, forming the local underlying surface, we convert the edge detection of the entire point cloud into a point classification based on local patches. Therefore, in the first stage, we construct local patch feature descriptors that describe the local neighborhood around each point. In the second stage, we classify each point by analyzing the local patch feature descriptors generated in the first stage. Due to the conversion of the point cloud into local patches, the proposed method can effectively extract the finer details. The experimental results show that our model demonstrates competitive performance compared to six baselines.
>
---
#### [new 050] Latent Denoising Improves Visual Alignment in Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决LMMs视觉表示弱、对分布变化敏感的问题。通过引入潜在去噪框架，提升视觉特征对齐和多模态理解能力。**

- **链接: [https://arxiv.org/pdf/2604.21343](https://arxiv.org/pdf/2604.21343)**

> **作者:** Dhruv Parikh; Jacob Fein-Ashley; Rajgopal Kannan; Viktor Prasanna
>
> **备注:** Technical Report
>
> **摘要:** Large Multimodal Models (LMMs) such as LLaVA are typically trained with an autoregressive language modeling objective, providing only indirect supervision to visual tokens. This often yields weak internal visual representations and brittle behavior under distribution shift. Inspired by recent progress on latent denoising for learning high-quality visual tokenizers, we show that the same principle provides an effective form of visual supervision for improving internal visual feature alignment and multimodal understanding in LMMs. We propose a latent denoising framework that corrupts projected visual tokens using a saliency-aware mixture of masking and Gaussian noising. The LMM is trained to denoise these corrupted tokens by recovering clean teacher patch features from hidden states at a selected intermediate LLM layer using a decoder. To prevent representation collapse, our framework also preserves the teacher's intra-image similarity structure and applies intra-image contrastive patch distillation. During inference, corruption and auxiliary heads are disabled, introducing no additional inference-time overhead. Across a broad suite of standard multimodal benchmarks, our method consistently improves visual understanding and reasoning over strong baselines, and yields clear gains on compositional robustness benchmarks (e.g., NaturalBench). Moreover, under ImageNet-C-style non-adversarial common corruptions applied to benchmark images, our method maintains higher accuracy and exhibits reduced degradation at both moderate and severe corruption levels. Our code is available at this https URL.
>
---
#### [new 051] Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于多模态视觉语言任务，旨在解决egocentric视角下指指点点的指代推理问题。针对MLLMs在空间语义理解上的不足，提出EgoPoint-Bench基准，并通过合成数据提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.21461](https://arxiv.org/pdf/2604.21461)**

> **作者:** Chentao Li; Zirui Gao; Mingze Gao; Yinglian Ren; Jianjiang Feng; Jie Zhou
>
> **备注:** 20 pages, 14 figures. Committed to ACL 2026
>
> **摘要:** Egocentric AI agents, such as smart glasses, rely on pointing gestures to resolve referential ambiguities in natural language commands. However, despite advancements in Multimodal Large Language Models (MLLMs), current systems often fail to precisely ground the spatial semantics of pointing. Instead, they rely on spurious correlations with visual proximity or object saliency, a phenomenon we term "Referential Hallucination." To address this gap, we introduce EgoPoint-Bench, a comprehensive question-answering benchmark designed to evaluate and enhance multimodal pointing reasoning in egocentric views. Comprising over 11k high-fidelity simulated and real-world samples, the benchmark spans five evaluation dimensions and three levels of referential complexity. Extensive experiments demonstrate that while state-of-the-art proprietary and open-source models struggle with egocentric pointing, models fine-tuned on our synthetic data achieve significant performance gains and robust sim-to-real generalization. This work highlights the importance of spatially aware supervision and offers a scalable path toward precise egocentric AI assistants. Project page: this https URL
>
---
#### [new 052] CHRep: Cross-modal Histology Representation and Post-hoc Calibration for Spatial Gene Expression Prediction
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于空间转录组预测任务，解决H&E图像到基因表达预测中的稳定性与准确性问题，提出CHRep框架提升预测性能。**

- **链接: [https://arxiv.org/pdf/2604.21573](https://arxiv.org/pdf/2604.21573)**

> **作者:** Changfan Wang; Xinran Wang; Donghai Liu; Fei Su; Lulu Sun; Zhicheng Zhao; Zhu Meng
>
> **摘要:** Spatial transcriptomics (ST) enables spatially resolved gene profiling but remains expensive and low-throughput, limiting large-cohort studies and routine clinical use. Predicting spatial gene expression from routine hematoxylin and eosin (H&E) slides is a promising alternative, yet under realistic leave-one-slide-out evaluation, existing models often suffer from slide-level appearance shifts and regression-driven over-smoothing that suppress biologically meaningful variation. CHRep is a two-phase framework for robust histology-to-expression prediction. In the training phase, CHRep learns a structure-aware representation by jointly optimizing correlation-aware regression, symmetric image-expression alignment, and coordinate-induced spatial topology regularization. In the inference phase, cross-slide robustness is improved without backbone fine-tuning through a lightweight calibration module trained on the training slides, which combines a non-parametric estimate from a training gallery with a magnitude-regularized correction module. Unlike prior embedding-alignment or retrieval-based transfer methods that rely on a single prediction route, CHRep couples topology-preserving representation learning with post-hoc calibration, enabling stable neighborhood retrieval and controlled bias correction under slide-level shifts. Across the three cohorts, CHRep consistently improves gene-wise correlation under leave-one-slide-out evaluation, with the largest gains observed on Alex+10x. Relative to HAGE, the Pearson correlation coefficient on all considered genes [PCC(ACG)] increases by 4.0% on cSCC and 9.8% on HER2+. Relative to mclSTExp, PCC(ACG) further improves by 39.5% on Alex+10x, together with 9.7% and 9.0% reductions in mean squared error (MSE) and mean absolute error (MAE), respectively.
>
---
#### [new 053] ImageHD: Energy-Efficient On-Device Continual Learning of Visual Representations via Hyperdimensional Computing
- **分类: cs.CV**

- **简介: 该论文属于边缘AI中的持续学习任务，解决设备端高效学习问题。提出ImageHD系统，结合超维计算与轻量CNN，实现低功耗、低延迟的视觉表示学习。**

- **链接: [https://arxiv.org/pdf/2604.21280](https://arxiv.org/pdf/2604.21280)**

> **作者:** Jebacyril Arockiaraj; Dhruv Parikh; Viktor Prasanna
>
> **备注:** FCCM 2026
>
> **摘要:** On-device continual learning (CL) is critical for edge AI systems operating on non-stationary data streams, but most existing methods rely on backpropagation or exemplar-heavy classifiers, incurring substantial compute, memory, and latency overheads. Hyperdimensional computing (HDC) offers a lightweight alternative through fast, non-iterative online updates. Combined with a compact convolutional neural network (CNN) feature extractor, HDC enables efficient on-device adaptation with strong visual representations. However, prior HDC-based CL systems often depend on multi-tier memory hierarchies and complex cluster management, limiting deployability on resource-constrained hardware. We present ImageHD, an FPGA accelerator for on-device continual learning of visual data based on HDC. ImageHD targets streaming CL under strict latency and on-chip memory constraints, avoiding costly iterative optimization. At the algorithmic level, we introduce a hardware-aware CL method that bounds class exemplars through a unified exemplar memory and a hardware-efficient cluster merging strategy, while incorporating a quantized CNN front-end to reduce deployment overhead without sacrificing accuracy. At the system level, ImageHD is implemented as a streaming dataflow architecture on the AMD Zynq ZCU104 FPGA, integrating HDC encoding, similarity search, and bounded cluster management using word-packed binary hypervectors for massively parallel bitwise computation within tight on-chip resource budgets. On CORe50, ImageHD achieves up to 40.4x (4.84x) speedup and 383x (105.1x) energy efficiency over optimized CPU (GPU) baselines, demonstrating the practicality of HDC-enabled continual learning for real-time edge AI.
>
---
#### [new 054] Component-Based Out-of-Distribution Detection
- **分类: cs.CV**

- **简介: 该论文属于OOD检测任务，旨在解决传统方法对局部异常敏感度不足及组合异常检测无效的问题。提出CoOD框架，通过分解输入并计算CSS和CCS实现更准确的检测。**

- **链接: [https://arxiv.org/pdf/2604.21546](https://arxiv.org/pdf/2604.21546)**

> **作者:** Wenrui Liu; Hong Chang; Ruibing Hou; Shiguang Shan; Xilin Chen
>
> **摘要:** Out-of-Distribution (OOD) detection requires sensitivity to subtle shifts without overreacting to natural In-Distribution (ID) diversity. However, from the viewpoint of detection granularity, global representation inevitably suppress local OOD cues, while patch-based methods are unstable due to entangled spurious-correlation and noise. And neither them is effective in detecting compositional OODs composed of valid ID components. Inspired by recognition-by-components theory, we present a training-free Component-Based OOD Detection (CoOD) framework that addresses the existing limitations by decomposing inputs into functional components. To instantiate CoOD, we derive Component Shift Score (CSS) to detect local appearance shifts, and Compositional Consistency Score (CCS) to identify cross-component compositional inconsistencies. Empirically, CoOD achieves consistent improvements on both coarse- and fine-grained OOD detection.
>
---
#### [new 055] Interpretable facial dynamics as behavioral and perceptual traces of deepfakes
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于深度伪造检测任务，旨在解决如何解释深度伪造与真实面部行为的差异。通过分析面部动态的可解释特征，提升检测准确性并揭示人类感知与模型判断的关系。**

- **链接: [https://arxiv.org/pdf/2604.21760](https://arxiv.org/pdf/2604.21760)**

> **作者:** Timothy Joseph Murphy; Jennifer Cook; Hélio Clemente José Cuve
>
> **备注:** Main paper: 19 pages, 5 figures, 4 tables. SI Appendix: 11 pages, 3 figures, 6 tables
>
> **摘要:** Deepfake detection research has largely converged on deep learning approaches that, despite strong benchmark performance, offer limited insight into what distinguishes real from manipulated facial behavior. This study presents an interpretable alternative grounded in bio-behavioral features of facial dynamics and evaluates how computational detection strategies relate to human perceptual judgments. We identify core low-dimensional patterns of facial movement, from which temporal features characterizing spatiotemporal structure were derived. Traditional machine learning classifiers trained on these features achieved modest but significant above-chance deepfake classification, driven by higher-order temporal irregularities that were more pronounced in manipulated than real facial dynamics. Notably, detection was substantially more accurate for videos containing emotive expressions than those without. An emotional valence classification analysis further indicated that emotive signals are systematically degraded in deepfakes, explaining the differential impact of emotive dynamics on detection. Furthermore, we provide an additional and often overlooked dimension of explainability by assessing the relationship between model decisions and human perceptual detection. Model and human judgments converged for emotive but diverged for non-emotive videos, and even where outputs aligned, underlying detection strategies differed. These findings demonstrate that face-swapped deepfakes carry a measurable behavioral fingerprint, most salient during emotional expression. Additionally, model-human comparisons suggest that interpretable computational features and human perception may offer complementary rather than redundant routes to detection.
>
---
#### [new 056] Multiscale Super Resolution without Image Priors
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于图像超分辨率任务，解决多尺度下图像重建的模糊问题。通过不同像素尺寸的图像组合，使问题可解，并利用傅里叶域和迭代方法实现高效重建。**

- **链接: [https://arxiv.org/pdf/2604.21810](https://arxiv.org/pdf/2604.21810)**

> **作者:** Daniel Fu; Gabby Litterio; Pedro Felzenszwalb; Rashid Zia
>
> **摘要:** We address the ambiguities in the super-resolution problem under translation. We demonstrate that combinations of low-resolution images at different scales can be used to make the super-resolution problem well posed. Such differences in scale can be achieved using sensors with different pixel sizes (as demonstrated here) or by varying the effective pixel size through changes in optical magnification (e.g., using a zoom lens). We show that images acquired with pairwise coprime pixel sizes lead to a system with a stable inverse, and furthermore, that super-resolution images can be reconstructed efficiently using Fourier domain techniques or iterative least squares methods. Our mathematical analysis provides an expression for the expected error of the least squares reconstruction for large signals assuming i.i.d. noise that elucidates the noise-resolution tradeoff. These results are validated through both one- and two-dimensional experiments that leverage charge-coupled device (CCD) hardware binning to explore reconstructions over a large range of effective pixel sizes. Finally, two-dimensional reconstructions for a series of targets are used to demonstrate the advantages of multiscale super-resolution, and implications of these results for common imaging systems are discussed.
>
---
#### [new 057] Pre-process for segmentation task with nonlinear diffusion filters
- **分类: cs.CV**

- **简介: 该论文属于图像分割预处理任务，旨在通过非线性扩散滤波获得分段常数图像。工作包括提出新的扩散系数，实现边缘保持的图像分割。**

- **链接: [https://arxiv.org/pdf/2604.21422](https://arxiv.org/pdf/2604.21422)**

> **作者:** Javier Sanguino; Carlos Platero; Olga Velasco
>
> **备注:** Manuscript from 2017, previously unpublished, 37 pages
>
> **摘要:** This paper deals with the case of using nonlinear diffusion filters to obtain piecewise constant images as a previous process for segmentation techniques. We first show an intrinsic formulation for the nonlinear diffusion equation to provide some design conditions on the diffusion filters. According to this theoretical framework, we propose a new family of diffusivities; they are obtained from nonlinear diffusion techniques and are related with backward diffusion. Their goal is to split the image in closed contours with a homogenized grey intensity inside and with no blurred edges. We also prove that our filters satisfy the well-posedness semi-discrete and full discrete scale-space requirements. This shows that by using semi-implicit schemes, a forward nonlinear diffusion equation is solved, instead of a backward nonlinear diffusion equation, connecting with an edge-preserving process. Under the conditions established for the diffusivity and using a stopping criterion for the diffusion time, we get piecewise constant images with a low computational effort. Finally, we test our filter with real images and we illustrate the effects of our diffusivity function as a method to get piecewise constant images. The code is available at this https URL.
>
---
#### [new 058] Seeing Isn't Believing: Uncovering Blind Spots in Evaluator Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决评估模型的可靠性问题。通过引入扰动测试，发现当前评估模型存在显著盲点，无法有效检测错误。**

- **链接: [https://arxiv.org/pdf/2604.21523](https://arxiv.org/pdf/2604.21523)**

> **作者:** Mohammed Safi Ur Rahman Khan; Sanjay Suryanarayanan; Tushar Anand; Mitesh M. Khapra
>
> **摘要:** Large Vision-Language Models (VLMs) are increasingly used to evaluate outputs of other models, for image-to-text (I2T) tasks such as visual question answering, and text-to-image (T2I) generation tasks. Despite this growing reliance, the reliability of these Evaluator VLMs remains under explored. In this work, we systematically evaluate the reliability of Evaluator VLMs across both I2T and T2I tasks. We introduce targeted perturbations that degrade output quality along key error dimensions, including object hallucinations, spatial reasoning, factual grounding, and visual fidelity. These perturbations test whether Evaluator VLMs can reliably account for these quality degrading errors in their evaluations. Using a comprehensive benchmark of over 4000 perturbed instances spanning 40 perturbation dimensions, we evaluate 4 prominent VLMs using single-answer scoring, pairwise comparison, and reference-guided paradigms. Our findings reveal that current VLM evaluators exhibit substantial blind spots: they often fail to detect perturbed outputs - in some cases exceeding 50%, struggle particularly with fine-grained compositional and spatial errors, and are often insensitive to hallucinated content that contradicts the input image. Pairwise comparison proves more reliable, though failure rates persist. These results highlight the unreliable nature of current Evaluator VLMs and urge caution in their deployment for benchmarking and development decisions. Code and data have been made publicly available.
>
---
#### [new 059] Deep kernel video approximation for unsupervised action segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频动作分割任务，解决无监督下视频动作划分问题。通过深度核空间学习，利用MMD优化视频分布近似，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2604.21572](https://arxiv.org/pdf/2604.21572)**

> **作者:** Silvia L. Pintea; Jouke Dijkstra
>
> **备注:** Accepted at ICPR 2026
>
> **摘要:** This work focuses on per-video unsupervised action segmentation, which is of interest to applications where storing large datasets is either not possible, or nor permitted. We propose to segment videos by learning in deep kernel space, to approximate the underlying frame distribution, as closely as possible. To define this closeness metric between the original video distribution and its approximation, we rely on maximum mean discrepancy (MMD) which is a geometry-preserving metric in distribution space, and thus gives more reliable estimates. Moreover, unlike the commonly used optimal transport metric, MMD is both easier to optimize, and faster. We choose to use neural tangent kernels (NTKs) to define the kernel space where MMD operates, because of their improved descriptive power as opposed to fixed kernels. And, also, because NTKs sidestep the trivial solution, when jointly learning the inputs (video approximation) and the kernel function. Finally, we show competitive results when compared to state-of-the-art per-video methods, on six standard benchmarks. Additionally, our method has higher F1 scores than prior agglomerative work, when the number of segments is unknown.
>
---
#### [new 060] Sculpt4D: Generating 4D Shapes via Sparse-Attention Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于4D生成任务，解决动态形状生成中的时间伪影和计算成本高的问题。提出Sculpt4D框架，结合稀疏注意力机制，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2604.21592](https://arxiv.org/pdf/2604.21592)**

> **作者:** Minghao Yin; Wenbo Hu; Jiale Xu; Ying Shan; Kai Han
>
> **摘要:** Recent breakthroughs in 3D generative modeling have yielded remarkable progress in static shape synthesis, yet high-fidelity dynamic 4D generation remains elusive, hindered by temporal artifacts and prohibitive computational demand. We present Sculpt4D, a native 4D generative framework that seamlessly integrates efficient temporal modeling into a pretrained 3D Diffusion Transformer (Hunyuan3D 2.1), thereby mitigating the scarcity of 4D training data. At its core lies a Block Sparse Attention mechanism that preserves object identity by anchoring to the initial frame while capturing rich motion dynamics via a time-decaying sparse mask. This design faithfully models complex spatiotemporal dependencies with high fidelity, while sidestepping the quadratic overhead of full attention and reducing network total computation by 56%. Consequently, Sculpt4D establishes a new state-of-the-art in temporally coherent 4D synthesis and charts a path toward efficient and scalable 4D generation.
>
---
#### [new 061] Building a Precise Video Language with Human-AI Oversight
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文聚焦视频语言建模任务，解决视频精准描述问题。通过结构化规范和人机协作框架CHAI，提升视频字幕的准确性与专业性。**

- **链接: [https://arxiv.org/pdf/2604.21718](https://arxiv.org/pdf/2604.21718)**

> **作者:** Zhiqiu Lin; Chancharik Mitra; Siyuan Cen; Isaac Li; Yuhan Huang; Yu Tong Tiffany Ling; Hewei Wang; Irene Pi; Shihang Zhu; Ryan Rao; George Liu; Jiaxi Li; Ruojin Li; Yili Han; Yilun Du; Deva Ramanan
>
> **备注:** CVPR 2026 Highlight. Project page: this https URL
>
> **摘要:** Video-language models (VLMs) learn to reason about the dynamic visual world through natural language. We introduce a suite of open datasets, benchmarks, and recipes for scalable oversight that enable precise video captioning. First, we define a structured specification for describing subjects, scenes, motion, spatial, and camera dynamics, grounded by hundreds of carefully defined visual primitives developed with professional video creators such as filmmakers. Next, to curate high-quality captions, we introduce CHAI (Critique-based Human-AI Oversight), a framework where trained experts critique and revise model-generated pre-captions into improved post-captions. This division of labor improves annotation accuracy and efficiency by offloading text generation to models, allowing humans to better focus on verification. Additionally, these critiques and preferences between pre- and post-captions provide rich supervision for improving open-source models (Qwen3-VL) on caption generation, reward modeling, and critique generation through SFT, DPO, and inference-time scaling. Our ablations show that critique quality in precision, recall, and constructiveness, ensured by our oversight framework, directly governs downstream performance. With modest expert supervision, the resulting model outperforms closed-source models such as Gemini-3.1-Pro. Finally, we apply our approach to re-caption large-scale professional videos (e.g., films, commercials, games) and fine-tune video generation models such as Wan to better follow detailed prompts of up to 400 words, achieving finer control over cinematography including camera motion, angle, lens, focus, point of view, and framing. Our results show that precise specification and human-AI oversight are key to professional-level video understanding and generation. Data and code are available on our project page: this https URL
>
---
#### [new 062] UAU-Net: Uncertainty-aware Representation Learning and Evidential Classification for Facial Action Unit Detection
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于面部动作单元检测任务，旨在解决不确定性建模问题。通过引入概率特征提取和证据分类方法，提升模型的鲁棒性和可靠性。**

- **链接: [https://arxiv.org/pdf/2604.21227](https://arxiv.org/pdf/2604.21227)**

> **作者:** Yuze Li; Zhilei Liu
>
> **备注:** Accepted by ICMR 2026
>
> **摘要:** Facial action unit (AU) detection remains challenging because it involves heterogeneous, AU-specific uncertainties arising at both the representation and decision stages. Recent methods have improved discriminative feature learning, but they often treat the AU representations as deterministic, overlooking uncertainty caused by visual noise, subject-dependent appearance variations, and ambiguous inter-AU relationships, all of which can substantially degrade robustness. Meanwhile, conventional point-estimation classifiers often provide poorly calibrated confidence, producing overconfident predictions, especially under the severe label imbalance typical of AU datasets. We propose UAU-Net, an Uncertainty-aware AU detection framework that explicitly models uncertainty at both stages. At the representation stage, we introduce CV-AFE, a conditional VAE (CVAE)-based AU feature extraction module that learns probabilistic AU representations by jointly estimating feature means and variances across multiple spatio-temporal scales; conditioning on AU labels further enables CV-AFE to capture uncertainty associated with inter-AU dependencies. At the decision stage, we design AB-ENN, an Asymmetric Beta Evidential Neural Network for multi-label AU detection, which parameterizes predictive uncertainty with Beta distributions and mitigates overconfidence via an asymmetric loss tailored to highly imbalanced binary labels. Extensive experiments on BP4D and DISFA show that UAU-Net achieves strong AU detection performance, and further analyses indicate that modeling uncertainty in both representation learning and evidential prediction improves robustness and reliability.
>
---
#### [new 063] The First Challenge on Remote Sensing Infrared Image Super-Resolution at NTIRE 2026: Benchmark Results and Method Overview
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于红外图像超分辨率任务，旨在通过x4缩放因子从低分辨率图像恢复高分辨率红外图像。论文介绍了挑战赛的设计、数据集及结果分析。**

- **链接: [https://arxiv.org/pdf/2604.21312](https://arxiv.org/pdf/2604.21312)**

> **作者:** Kai Liu; Haoyang Yue; Zeli Lin; Zheng Chen; Jingkai Wang; Jue Gong; Jiatong Li; Xianglong Yan; Libo Zhu; Jianze Li; Ziqing Zhang; Zihan Zhou; Xiaoyang Liu; Radu Timofte; Yulun Zhang; Junye Chen; Zhenming Yan; Yucong Hong; Ruize Han; Song Wang; Li Pang; Heng Zhao; Xinqiao Wu; Deyu Meng; Xiangyong Cao; Weijun Yuan; Zhan Li; Zhanglu Chen; Boyang Yao; Yihang Chen; Yifan Deng; Zengyuan Zuo; Junjun Jiang; Saiprasad Meesiyawar; Sulocha Yatageri; Nikhil Akalwadi; Ramesh Ashok Tabib; Uma Mudenagudi; Jiachen Tu; Yaokun Shi; Guoyi Xu; Yaoxin Jiang; Cici Liu; Tongyao Mu; Qiong Cao; Yifan Wang; Kosuke Shigematsu; Hiroto Shirono; Asuka Shin; Wei Zhou; Linfeng Li; Lingdong Kong; Ce Wang; Xingwei Zhong; Wanjie Sun; Dafeng Zhang; Hongxin Lan; Qisheng Xu; Mingyue He; Hui Geng; Tianjiao Wan; Kele Xu; Changjian Wang; Antoine Carreaud; Nicola Santacroce; Shanci Li; Jan Skaloud; Adrien Gressin
>
> **备注:** Github Repo: this https URL
>
> **摘要:** This paper presents the NTIRE 2026 Remote Sensing Infrared Image Super-Resolution (x4) Challenge, one of the associated challenges of NTIRE 2026. The challenge aims to recover high-resolution (HR) infrared images from low-resolution (LR) inputs generated through bicubic downsampling with a x4 scaling factor. The objective is to develop effective models or solutions that achieve state-of-the-art performance for infrared image SR in remote sensing scenarios. To reflect the characteristics of infrared data and practical application needs, the challenge adopts a single-track setting. A total of 115 participants registered for the competition, with 13 teams submitting valid entries. This report summarizes the challenge design, dataset, evaluation protocol, main results, and the representative methods of each team. The challenge serves as a benchmark to advance research in infrared image super-resolution and promote the development of effective solutions for real-world remote sensing applications.
>
---
#### [new 064] Instance-level Visual Active Tracking with Occlusion-Aware Planning
- **分类: cs.CV**

- **简介: 该论文属于视觉跟踪任务，解决目标混淆和遮挡问题。提出OA-VAT框架，包含实例原型初始化、在线跟踪增强和遮挡感知路径规划，提升跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.21453](https://arxiv.org/pdf/2604.21453)**

> **作者:** Haowei Sun; Kai Zhou; Hao Gao; Shiteng Zhang; Jinwu Hu; Xutao Wen; Qixiang Ye; Mingkui Tan
>
> **备注:** CVPR 2026 Poster
>
> **摘要:** Visual Active Tracking (VAT) aims to control cameras to follow a target in 3D space, which is critical for applications like drone navigation and security surveillance. However, it faces two key bottlenecks in real-world deployment: confusion from visually similar distractors caused by insufficient instance-level discrimination and severe failure under occlusions due to the absence of active planning. To address these, we propose OA-VAT, a unified pipeline with three complementary modules. First, a training-free Instance-Aware Offline Prototype Initialization aggregates multi-view augmented features via DINOv3 to construct discriminative instance prototypes, mitigating distractor confusion. Second, an Online Prototype Enhancement Tracker enhances prototypes online and integrates a confidence-aware Kalman filter for stable tracking under appearance and motion changes. Third, an Occlusion-Aware Trajectory Planner, trained on our new Planning-20k dataset, uses conditional diffusion to generate obstacle-avoiding paths for occlusion recovery. Experiments demonstrate OA-VAT achieves 0.93 average SR on UnrealCV (+2.2% vs. SOTA TrackVLA), 90.8% average CAR on real-world datasets (+12.1% vs. SOTA GC-VAT), and 81.6% TSR on a DJI Tello drone. Running at 35 FPS on an RTX 3090, it delivers robust, real-time performance for practical deployment.
>
---
#### [new 065] OmniFit: Multi-modal 3D Body Fitting via Scale-agnostic Dense Landmark Prediction
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出OmniFit，解决3D人体拟合任务中的多模态输入与尺度无关问题。通过密集关键点预测和尺度预测，提升拟合精度。**

- **链接: [https://arxiv.org/pdf/2604.21575](https://arxiv.org/pdf/2604.21575)**

> **作者:** Zeyu Cai; Yuliang Xiu; Renke Wang; Zhijing Shao; Xiaoben Li; Siyuan Yu; Chao Xu; Yang Liu; Baigui Sun; Jian Yang; Zhenyu Zhang
>
> **备注:** Project Page: this https URL
>
> **摘要:** Fitting an underlying body model to 3D clothed human assets has been extensively studied, yet most approaches focus on either single-modal inputs such as point clouds or multi-view images alone, often requiring a known metric scale. This constraint is frequently impractical, especially for AI-generated assets where scale distortion is common. We propose OmniFit, a method that can seamlessly handle diverse multi-modal inputs, including full scans, partial depth observations, and image captures, while remaining scale-agnostic for both real and synthetic assets. Our key innovation is a simple yet effective conditional transformer decoder that directly maps surface points to dense body landmarks, which are then used for SMPL-X parameter fitting. In addition, an optional plug-and-play image adapter incorporates visual cues to compensate for missing geometric information. We further introduce a dedicated scale predictor that rescales subjects to canonical body proportions. OmniFit substantially outperforms state-of-the-art methods by 57.1 to 80.9 percent across daily and loose clothing scenarios. To the best of our knowledge, it is the first body fitting method to surpass multi-view optimization baselines and the first to achieve millimeter-level accuracy on the CAPE and 4D-DRESS benchmarks.
>
---
#### [new 066] Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉问答任务，旨在解决多步、意图驱动的植物病理诊断问题。提出PlantInquiryVQA基准，通过结构化提问提升模型诊断准确性与推理效率。**

- **链接: [https://arxiv.org/pdf/2604.20983](https://arxiv.org/pdf/2604.20983)**

> **作者:** Syed Nazmus Sakib; Nafiul Haque; Shahrear Bin Amin; Hasan Muhammad Abdullah; Md. Mehedi Hasan; Mohammad Zabed Hossain; Shifat E. Arman
>
> **备注:** Accepted at ACL 2026 Findings
>
> **摘要:** Vision evaluations are typically done through multi-step processes. In most contemporary fields, experts analyze images using structured, evidence-based adaptive questioning. In plant pathology, botanists inspect leaf images, identify visual cues, infer diagnostic intent, and probe further with targeted questions that adapt to species, symptoms, and severity. This structured probing is crucial for accurate disease diagnosis and treatment formulation. Yet current vision-language models are evaluated on single-turn question answering. To address this gap, we introduce PlantInquiryVQA, a benchmark for studying multi-step, intent-driven visual reasoning in botanical diagnosis. We formalize a Chain of Inquiry framework modeling diagnostic trajectories as ordered question-answer sequences conditioned on grounded visual cues and explicit epistemic intent. We release a dataset of 24,950 expert-curated plant images and 138,068 question-answer pairs annotated with visual grounding, severity labels, and domain-specific reasoning templates. Evaluations on top-tier Multimodal Large Language Models reveal that while they describe visual symptoms adequately, they struggle with safe clinical reasoning and accurate diagnosis. Importantly, structured question-guided inquiry significantly improves diagnostic correctness, reduces hallucination, and increases reasoning efficiency. We hope PlantInquiryVQA serves as a foundational benchmark in advancing research to train diagnostic agents to reason like expert botanists rather than static classifiers.
>
---
#### [new 067] KD-CVG: A Knowledge-Driven Approach for Creative Video Generation
- **分类: cs.CV**

- **简介: 该论文属于创意视频生成任务，解决T2V模型在语义对齐和运动适应性上的不足。提出KD-CVG方法，通过知识驱动提升生成效果。**

- **链接: [https://arxiv.org/pdf/2604.21362](https://arxiv.org/pdf/2604.21362)**

> **作者:** Linkai Liu; Wei Feng; Xi Zhao; Shen Zhang; Xingye Chen; Zheng Zhang; Jingjing Lv; Junjie Shen; Ching Law; Yuchen Zhou; Zipeng Guo; Chao Gou
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Creative Generation (CG) leverages generative models to automatically produce advertising content that highlights product features, and it has been a significant focus of recent research. However, while CG has advanced considerably, most efforts have concentrated on generating advertising text and images, leaving Creative Video Generation (CVG) relatively underexplored. This gap is largely due to two major challenges faced by Text-to-Video (T2V) models: (a) \textbf{ambiguous semantic alignment}, where models struggle to accurately correlate product selling points with creative video content, and (b) \textbf{inadequate motion adaptability}, resulting in unrealistic movements and distortions. To address these challenges, we develop a comprehensive Advertising Creative Knowledge Base (ACKB) as a foundational resource and propose a knowledge-driven approach (KD-CVG) to overcome the knowledge limitations of existing models. KD-CVG consists of two primary modules: Semantic-Aware Retrieval (SAR) and Multimodal Knowledge Reference (MKR). SAR utilizes the semantic awareness of graph attention networks and reinforcement learning feedback to enhance the model's comprehension of the connections between selling points and creative videos. Building on this, MKR incorporates semantic and motion priors into the T2V model to address existing knowledge gaps. Extensive experiments have demonstrated KD-CVG's superior performance in achieving semantic alignment and motion adaptability, validating its effectiveness over other state-of-the-art methods. The code and dataset will be open source at this https URL.
>
---
#### [new 068] an interpretable vision transformer framework for automated brain tumor classification
- **分类: cs.CV**

- **简介: 该论文属于脑肿瘤分类任务，旨在解决手动诊断耗时、易出错的问题。提出一种基于Vision Transformer的可解释框架，提升分类准确率与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.21311](https://arxiv.org/pdf/2604.21311)**

> **作者:** Chinedu Emmanuel Mbonu; Tochukwu Sunday Belonwu; Okwuchukwu Ejike Chukwuogo; Kenechukwu Sylvanus Anigbogu
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Brain tumors represent one of the most critical neurological conditions, where early and accurate diagnosis is directly correlated with patient survival rates. Manual interpretation of Magnetic Resonance Imaging (MRI) scans is time-intensive, subject to inter-observer variability, and demands significant specialist expertise. This paper proposes a deep learning framework for automated four-class brain tumor classification distinguishing glioma, meningioma, pituitary tumor, and healthy brain tissue from a dataset of 7,023 MRI scans. The proposed system employs a Vision Transformer (ViT-B/16) pretrained on ImageNet-21k as the backbone, augmented with a clinically motivated preprocessing and training pipeline. Contrast Limited Adaptive Histogram Equalization (CLAHE) is applied to enhance local contrast and accentuate tumor boundaries invisible to standard normalization. A two-stage fine-tuning strategy is adopted: the classification head is warmed up with the backbone frozen, followed by full fine-tuning with discriminative learning rates. MixUp and CutMix augmentation is applied per batch to improve generalization. Exponential Moving Average (EMA) of weights and Test-Time Augmentation (TTA) further stabilize and boost performance. Attention Rollout visualization provides clinically interpretable heatmaps of the brain regions driving each prediction. The proposed model achieves a test accuracy of 99.29%, macro F1-score of 99.25%, and perfect recall on both healthy and meningioma classes, outperforming all CNN-based baselines
>
---
#### [new 069] UniGenDet: A Unified Generative-Discriminative Framework for Co-Evolutionary Image Generation and Generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于图像生成与检测任务，旨在解决两者架构差异带来的协同困难。提出UniGenDet框架，通过联合优化提升生成质量与检测效果。**

- **链接: [https://arxiv.org/pdf/2604.21904](https://arxiv.org/pdf/2604.21904)**

> **作者:** Yanran Zhang; Wenzhao Zheng; Yifei Li; Bingyao Yu; Yu Zheng; Lei Chen; Jiwen Lu; Jie Zhou
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** In recent years, significant progress has been made in both image generation and generated image detection. Despite their rapid, yet largely independent, development, these two fields have evolved distinct architectural paradigms: the former predominantly relies on generative networks, while the latter favors discriminative frameworks. A recent trend in both domains is the use of adversarial information to enhance performance, revealing potential for synergy. However, the significant architectural divergence between them presents considerable challenges. Departing from previous approaches, we propose UniGenDet: a Unified generative-discriminative framework for co-evolutionary image Generation and generated image Detection. To bridge the task gap, we design a symbiotic multimodal self-attention mechanism and a unified fine-tuning algorithm. This synergy allows the generation task to improve the interpretability of authenticity identification, while authenticity criteria guide the creation of higher-fidelity images. Furthermore, we introduce a detector-informed generative alignment mechanism to facilitate seamless information exchange. Extensive experiments on multiple datasets demonstrate that our method achieves state-of-the-art performance. Code: \href{this https URL}{this https URL}.
>
---
#### [new 070] MiMIC: Mitigating Visual Modality Collapse in Universal Multimodal Retrieval While Avoiding Semantic Misalignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态检索任务，旨在解决视觉模态崩溃和语义错位问题。提出MiMIC模型，通过融合解码器架构和鲁棒训练方法提升性能。**

- **链接: [https://arxiv.org/pdf/2604.21326](https://arxiv.org/pdf/2604.21326)**

> **作者:** Juan Li; Chuanghao Ding; Xujie Zhang; Cam-Tu Nguyen
>
> **摘要:** Universal Multimodal Retrieval (UMR) aims to map different modalities (e.g., visual and textual) into a shared embedding space for multi-modal retrieval. Existing UMR methods can be broadly divided into two categories: early-fusion approaches, such as Marvel, which projects visual features into the language model (LM) space for integrating with text modality, and late-fusion approaches, such as UniVL-DR, which encode visual and textual inputs using separate encoders and obtain fused embeddings through addition. Our pilot study reveals that Marvel exhibits visual modality collapse, which is characterized by the model's tendency to disregard visual features while depending excessively on textual cues. In contrast, although UniVL-DR is less affected by this issue, it is more susceptible to semantic misalignment, where semantically related content is positioned far apart in the embedding space. To address these challenges, we propose MiMIC, which introduces two key innovations: (1) a fusion-in-decoder architecture for effective multimodal integration, and (2) robust training through single modality mixin and random caption dropout. Experiments on the WebQA+ and EVQA+ datasets, where image in documents or queries might lack captions, indicate that MiMIC consistently outperforms both early- and late-fusion baselines.
>
---
#### [new 071] Seeing Fast and Slow: Learning the Flow of Time in Videos
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文研究视频中时间流动的感知与控制，解决视频速度变化检测与生成问题。通过自监督学习提取时间特征，构建慢动作视频数据集，并实现速度可控的视频生成与时间超分辨率。**

- **链接: [https://arxiv.org/pdf/2604.21931](https://arxiv.org/pdf/2604.21931)**

> **作者:** Yen-Siang Wu; Rundong Luo; Jingsen Zhu; Tao Tu; Ali Farhadi; Matthew Wallingford; Yu-Chiang Frank Wang; Steve Marschner; Wei-Chiu Ma
>
> **备注:** Project page: this https URL
>
> **摘要:** How can we tell whether a video has been sped up or slowed down? How can we generate videos at different speeds? Although videos have been central to modern computer vision research, little attention has been paid to perceiving and controlling the passage of time. In this paper, we study time as a learnable visual concept and develop models for reasoning about and manipulating the flow of time in videos. We first exploit the multimodal cues and temporal structure naturally present in videos to learn, in a self-supervised manner, to detect speed changes and estimate playback speed. We then show that these learned temporal reasoning models enable us to curate the largest slow-motion video dataset to date from noisy in-the-wild sources. Such slow-motion footage, typically filmed by high-speed cameras, contains substantially richer temporal detail than standard videos. Using this data, we further develop models capable of temporal control, including speed-conditioned video generation, which produces motion at specified playback speed, and temporal super-resolution, which tranforms low-FPS, blurry videos into high-FPS sequences with fine-grained temporal details. Our findings highlight time as a manipulable, perceptual dimension in video learning, opening doors to temporally controllable video generation, temporal forensics detection, and potentially richer world-models that understand how events unfold over time.
>
---
#### [new 072] Grounding Video Reasoning in Physical Signals
- **分类: cs.CV**

- **简介: 该论文属于物理视频理解任务，旨在解决视频问答中事件定位不足的问题。提出一个扩展的基准，涵盖多源视频和多种提示类型，评估模型在物理和语义层面的推理能力。**

- **链接: [https://arxiv.org/pdf/2604.21873](https://arxiv.org/pdf/2604.21873)**

> **作者:** Alibay Osmanli; Zixu Cheng; Shaogang Gong
>
> **备注:** Benchmark for Grounding Video Reasoning in Physical Signals
>
> **摘要:** Physical video understanding requires more than naming an event correctly. A model can answer a question about pouring, sliding, or collision from textual regularities while still failing to localize the event in time or space. We introduce a grounded benchmark for physical video understanding that extends the what--when--where evaluation structure of V-STaR to four video sources, six physics domains, three prompt families (physics, vstar_like, and neutral_rstr), and four input conditions (original, shuffled, ablated, and frame-masked). The benchmark contains 1,560 base video clips from SSV2, YouCook2, HoloAssist, and Roundabout-TAU. Each clip is first converted into a shared grounded event record, and the three query families are derived from that record. Temporal and spatial targets are shared across prompt families, while the non-physics families use deterministic family-appropriate semantic a_what targets derived from the same record. Across models and prompt families, physics remains the strongest regime overall, vstar_like is the clearest non-physics semantic comparison, and neutral_rstr behaves as a harder templated control. Prompt-family robustness is selective rather than universal, perturbation gains cluster in weak original cases, and spatial grounding is the weakest across settings. These results suggest that video Q&A reasoning benchmarks shall report physically grounded, prompt-aware, and perturbation-aware diagnostics alongside aggregate accuracy.
>
---
#### [new 073] Vista4D: Video Reshooting with 4D Point Clouds
- **分类: cs.CV**

- **简介: 该论文提出Vista4D，解决视频重拍摄问题，通过4D点云实现更精确的相机控制和场景一致性。**

- **链接: [https://arxiv.org/pdf/2604.21915](https://arxiv.org/pdf/2604.21915)**

> **作者:** Kuan Heng Lin; Zhizheng Liu; Pablo Salamanca; Yash Kant; Ryan Burgert; Yuancheng Xu; Koichi Namekata; Yiwei Zhao; Bolei Zhou; Micah Goldblum; Paul Debevec; Ning Yu
>
> **备注:** 24 pages, 20 figures, CVPR 2026, see project page at this https URL
>
> **摘要:** We present Vista4D, a robust and flexible video reshooting framework that grounds the input video and target cameras in a 4D point cloud. Specifically, given an input video, our method re-synthesizes the scene with the same dynamics from a different camera trajectory and viewpoint. Existing video reshooting methods often struggle with depth estimation artifacts of real-world dynamic videos, while also failing to preserve content appearance and failing to maintain precise camera control for challenging new trajectories. We build a 4D-grounded point cloud representation with static pixel segmentation and 4D reconstruction to explicitly preserve seen content and provide rich camera signals, and we train with reconstructed multiview dynamic data for robustness against point cloud artifacts during real-world inference. Our results demonstrate improved 4D consistency, camera control, and visual quality compared to state-of-the-art baselines under a variety of videos and camera paths. Moreover, our method generalizes to real-world applications such as dynamic scene expansion and 4D scene recomposition. See our project page for results, code, and models: this https URL
>
---
#### [new 074] HyperFM: An Efficient Hyperspectral Foundation Model with Spectral Grouping
- **分类: cs.CV**

- **简介: 该论文提出HyperFM，解决 hyperspectral 数据处理难题，通过 spectral grouping 提升模型效率与性能，用于大气云属性反演任务。**

- **链接: [https://arxiv.org/pdf/2604.21127](https://arxiv.org/pdf/2604.21127)**

> **作者:** Zahid Hassan Tushar; Sanjay Purushotham
>
> **备注:** 15 pages, 8 figures, to be published in CVPR 2026 findings, Code and data are publicly available on this https URL
>
> **摘要:** The NASA PACE mission provides unprecedented hyperspectral observations of ocean color, aerosols, and clouds, offering new insights into how these components interact and influence Earth's climate and air quality. Its Ocean Color Instrument measures light across hundreds of finely spaced wavelength bands, enabling detailed characterization of features such as phytoplankton composition, aerosol properties, and cloud microphysics. However, hyperspectral data of this scale is large, complex, and difficult to label, requiring specialized processing and analysis techniques. Existing foundation models, which have transformed computer vision and natural language processing, are generally trained on standard RGB imagery and therefore struggle to interpret the continuous spectral signatures captured by PACE. While recent advances have introduced hyperspectral foundation models, they are typically trained on cloud-free observations and often remain limited to single-sensor datasets due to spectral inconsistencies across instruments. Moreover, existing models tend to be parameter-heavy and computationally expensive, limiting scalability and adoption in operational settings. To address these challenges, we introduce HyperFM, a parameter-efficient hyperspectral foundation model that leverages intra-group and inter-group spectral attention along with hybrid parameter decomposition to better capture spectral spatial relationships while reducing computational cost. HyperFM demonstrates consistent performance improvements over existing hyperspectral foundation models and task-specific state-of-the-art methods across four benchmark downstream atmospheric cloud property retrieval tasks. To support further research, we additionally release HyperFM250K, a large-scale hyperspectral dataset from the PACE mission that includes both clear and cloudy scenes.
>
---
#### [new 075] Causal Disentanglement for Full-Reference Image Quality Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像质量评估任务，旨在解决传统方法依赖特征比较的问题。通过因果推理和解耦表示学习，提出新方法提升评估性能与跨域泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.21654](https://arxiv.org/pdf/2604.21654)**

> **作者:** Zhen Zhang; Jielei Chu; Tian Zhang; Weide Liu; Fengmao Lv; Tianrui Li; Jun Cheng; Yuming Fang
>
> **摘要:** Existing deep network-based full-reference image quality assessment (FR-IQA) models typically work by performing pairwise comparisons of deep features from the reference and distorted images. In this paper, we approach this problem from a different perspective and propose a novel FR-IQA paradigm based on causal inference and decoupled representation learning. Unlike typical feature comparison-based FR-IQA models, our approach formulates degradation estimation as a causal disentanglement process guided by intervention on latent representations. We first decouple degradation and content representations by exploiting the content invariance between the reference and distorted images. Second, inspired by the human visual masking effect, we design a masking module to model the causal relationship between image content and degradation features, thereby extracting content-influenced degradation features from distorted images. Finally, quality scores are predicted from these degradation features using either supervised regression or label-free dimensionality reduction. Extensive experiments demonstrate that our method achieves highly competitive performance on standard IQA benchmarks across fully supervised, few-label, and label-free settings. Furthermore, we evaluate the approach on diverse non-standard natural image domains with scarce data, including underwater, radiographic, medical, neutron, and screen-content images. Benefiting from its ability to perform scenario-specific training and prediction without labeled IQA data, our method exhibits superior cross-domain generalization compared to existing training-free FR-IQA models.
>
---
#### [new 076] SparseGF: A Height-Aware Sparse Segmentation Framework with Context Compression for Robust Ground Filtering Across Urban to Natural Scenes
- **分类: cs.CV**

- **简介: 该论文属于点云地面过滤任务，旨在解决深度学习方法在跨场景泛化中的上下文-细节矛盾和高物误分类问题。提出SparseGF框架，通过上下文压缩和高度感知损失提升分割效果。**

- **链接: [https://arxiv.org/pdf/2604.21356](https://arxiv.org/pdf/2604.21356)**

> **作者:** Nannan Qin; Pengjie Tao; Haiyan Guan; Zhizhong Kang; Lingfei Ma; Xiangyun Hu; Jonathan Li
>
> **摘要:** High-quality digital terrain models derived from airborne laser scanning (ALS) data are essential for a wide range of geospatial analyses, and their generation typically relies on robust ground filtering (GF) to separate point clouds across diverse landscapes into ground and non-ground parts. Although current deep-learning-based GF methods have demonstrated impressive performance, especially in specific challenging terrains, their cross-scene generalization remains limited by two persistent issues: the context-detail dilemma in large-scale processing due to limited computational resources, and the random misclassification of tall objects arising from classification-only optimization. To overcome these limitations, we propose SparseGF, a height-aware sparse segmentation framework enhanced with context compression. It is built upon three key innovations: (1) a convex-mirror-inspired context compression module that condenses expansive contexts into compact representations while preserving central details; (2) a hybrid sparse voxel-point network architecture that effectively interprets compressed representations while mitigating compression-induced geometric distortion; and (3) a height-aware loss function that explicitly enforces topographic elevation priors during training to suppress random misclassification of tall objects. Extensive evaluations on two large-scale ALS benchmark datasets demonstrate that SparseGF delivers robust GF across urban to natural terrains, achieving leading performance in complex urban scenes, competitive results on mixed terrains, and moderate yet non-catastrophic accuracy in densely forested steep areas. This work offers new insights into deep-learning-based GF research and encourages further exploration toward truly cross-scene generalization for large-scale environmental monitoring.
>
---
#### [new 077] S1-VL: Scientific Multimodal Reasoning Model with Thinking-with-Images
- **分类: cs.CV**

- **简介: 该论文提出S1-VL模型，解决科学领域多模态推理问题，支持结构化思维与图像操作结合，提升复杂科学任务的处理能力。**

- **链接: [https://arxiv.org/pdf/2604.21409](https://arxiv.org/pdf/2604.21409)**

> **作者:** Qingxiao Li; Lifeng Xu; QingLi Wang; Yudong Bai; Mingwei Ou; Shu Hu; Nan Xu
>
> **备注:** 29 pages, 13 figures
>
> **摘要:** We present S1-VL, a multimodal reasoning model for scientific domains that natively supports two complementary reasoning paradigms: Scientific Reasoning, which relies on structured chain-of-thought, and Thinking-with-Images, which enables the model to actively manipulate images through Python code execution during reasoning. In the Thinking-with-Images mode, the model generates and executes image-processing code in a sandbox environment, obtains intermediate visual results, and continues reasoning in a multi-turn iterative manner. This design is particularly effective for challenging scenarios such as high-resolution scientific chart interpretation, microscopic image understanding, and geometry-assisted reasoning. To construct the training data, we collect scientific multimodal datasets spanning six disciplines: mathematics, physics, chemistry, astronomy, geography, and biology. We further develop a six-dimensional quality filtering framework for reasoning trajectories. To mitigate redundant, ineffective, and erroneous visual operations commonly found in existing datasets, we propose a multi-stage filtering pipeline together with an adaptive data routing strategy. This strategy converts samples with low visual information gain into pure Reasoning-mode data, enabling the model to learn when image operations are truly necessary. S1-VL is trained through a four-stage progressive pipeline: scientific multimodal SFT, Thinking-with-Images cold-start SFT, and two stages of reinforcement learning with SAPO. We build S1-VL-32B on top of Qwen3-VL-32B-Thinking and evaluate it on 13 benchmarks. Experimental results show that S1-VL-32B achieves state-of-the-art performance on all five Thinking-with-Images benchmarks, including HRBench-4K, HRBench-8K, MME-RealWorld-CN, MME-RealWorld-Lite, and V*, and outperforms compared systems on scientific reasoning benchmarks such as Physics and VRSBench.
>
---
#### [new 078] LatRef-Diff: Latent and Reference-Guided Diffusion for Facial Attribute Editing and Style Manipulation
- **分类: cs.CV**

- **简介: 该论文属于人脸属性编辑与风格操控任务，解决现有方法控制不精准、训练不稳定的问题。提出LatRef-Diff框架，利用风格代码和双引导机制实现高效图像生成与编辑。**

- **链接: [https://arxiv.org/pdf/2604.21279](https://arxiv.org/pdf/2604.21279)**

> **作者:** Wenmin Huang; Weiqi Luo; Xiaochun Cao; Jiwu Huang
>
> **摘要:** Facial attribute editing and style manipulation are crucial for applications like virtual avatars and photo editing. However, achieving precise control over facial attributes without altering unrelated features is challenging due to the complexity of facial structures and the strong correlations between attributes. While conditional GANs have shown progress, they are limited by accuracy issues and training instability. Diffusion models, though promising, face challenges in style manipulation due to the limited expressiveness of semantic directions. In this paper, we propose LatRef-Diff, a novel diffusion-based framework that addresses these limitations. We replace the traditional semantic directions in diffusion models with style codes and propose two methods for generating them: latent and reference guidance. Based on these style codes, we design a style modulation module that integrates them into the target image, enabling both random and customized style manipulation. This module incorporates learnable vectors, cross-attention mechanisms, and a hierarchical design to improve accuracy and image quality. Additionally, to enhance training stability while eliminating the need for paired images (e.g., before and after editing), we propose a forward-backward consistency training strategy. This strategy first removes the target attribute approximately using image-specific semantic directions and then restores it via style modulation, guided by perceptual and classification losses. Extensive experiments on CelebA-HQ demonstrate that LatRef-Diff achieves state-of-the-art performance in both qualitative and quantitative evaluations. Ablation studies validate the effectiveness of our model's design choices.
>
---
#### [new 079] Attention-based multiple instance learning for predominant growth pattern prediction in lung adenocarcinoma wsi using foundation models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于肺腺癌生长模式预测任务，旨在减少标注需求。通过注意力机制整合预训练模型特征，提升全切片水平的预测性能。**

- **链接: [https://arxiv.org/pdf/2604.21530](https://arxiv.org/pdf/2604.21530)**

> **作者:** Laura Valeria Perez-Herrera; M.J. Garcia-Gonzalez; Karen Lopez-Linares
>
> **摘要:** Lung adenocarcinoma (LUAD) grading depends on accurately identifying growth patterns, which are indicators of prognosis and can influence treatment decisions. Common deep learning approaches to determine the predominant pattern rely on patch-level classification or segmentation, requiring extensive annotations. This study proposes an attention-based multiple instance learning (ABMIL) framework to predict the predominant LUAD growth pattern at the whole slide level to reduce annotation burden. Our approach integrates pretrained pathology foundation models as patch encoders, used either frozen or fine-tuned on annotated patches, to extract discriminative features that are aggregated through attention mechanisms. Experiments show that fine-tuned encoders improve performance, with Prov-GigaPath achieving the highest agreement (\k{appa} = 0.699) under ABMIL. Compared to simple patch-aggregation baselines, ABMIL yields more robust predictions by leveraging slide-level supervision and spatial attention. Future work will extend this framework to estimate the full distribution of growth patterns and validate performance on external cohorts.
>
---
#### [new 080] You Only Gaussian Once: Controllable 3D Gaussian Splatting for Ultra-Densely Sampled Scenes
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决3DGS在工业应用中的资源不可控、数据污染等问题。提出YOGO框架和Immersion数据集，提升重建精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.21400](https://arxiv.org/pdf/2604.21400)**

> **作者:** Jinrang Jia; Zhenjia Li; Yifeng Shi
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** 3D Gaussian Splatting (3DGS) has revolutionized neural rendering, yet existing methods remain predominantly research prototypes ill-suited for production-level deployment. We identify a critical "Industry-Academia Gap" hindering real-world application: unpredictable resource consumption from heuristic Gaussian growth, the "sparsity shield" of current benchmarks that rewards hallucination over physical fidelity, and severe multi-sensor data pollution. To bridge this gap, we propose YOGO (You Only Gaussian Once), a system-level framework that reformulates the stochastic growth process into a deterministic, budget-aware equilibrium. YOGO integrates a novel budget controller for hardware-constrained resource allocation and an availability-registration protocol for robust multi-sensor fusion. To push the boundaries of reconstruction fidelity, we introduce Immersion v1.0, the first ultra-dense indoor dataset specifically designed to break the "sparsity shield." By providing saturated viewpoint coverage, Immersion v1.0 forces algorithms to focus on extreme physical fidelity rather than viewpoint interpolation, and enables the community to focus on the upper limits of high-fidelity reconstruction. Extensive experiments demonstrate that YOGO achieves state-of-the-art visual quality while maintaining a strictly deterministic profile, establishing a new standard for production-grade 3DGS. To facilitate reproducibility, part scenes of Immersion v1.0 dataset and source code of YOGO has been publicly released. The project link is this https URL.
>
---
#### [new 081] Materialistic RIR: Material Conditioned Realistic RIR Generation
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文属于声学建模任务，旨在解决空间与材料影响纠缠的问题。通过分离空间与材料模块，实现材料可控的房间脉冲响应生成，提升音质真实性和可控制性。**

- **链接: [https://arxiv.org/pdf/2604.21119](https://arxiv.org/pdf/2604.21119)**

> **作者:** Mahnoor Fatima Saad; Sagnik Majumder; Kristen Grauman; Ziad Al-Halah
>
> **备注:** Accepted to CVPR 2026 Findings. Project page: this https URL
>
> **摘要:** Rings like gold, thuds like wood! The sound we hear in a scene is shaped not only by the spatial layout of the environment but also by the materials of the objects and surfaces within it. For instance, a room with wooden walls will produce a different acoustic experience from a room with the same spatial layout but concrete walls. Accurately modeling these effects is essential for applications such as virtual reality, robotics, architectural design, and audio engineering. Yet, existing methods for acoustic modeling often entangle spatial and material influences in correlated representations, which limits user control and reduces the realism of the generated acoustics. In this work, we present a novel approach for material-controlled Room Impulse Response (RIR) generation that explicitly disentangles the effects of spatial and material cues in a scene. Our approach models the RIR using two modules: a spatial module that captures the influence of the spatial layout of the scene, and a material module that modulates this spatial RIR according to a user-specified material configuration. This explicitly disentangled design allows users to easily modify the material configuration of a scene and observe its impact on acoustics without altering the spatial structure or scene content. Our model provides significant improvements over prior approaches on both acoustic-based metrics (up to +16% on RTE) and material-based metrics (up to +70%). Furthermore, through a human perceptual study, we demonstrate the improved realism and material sensitivity of our model compared to the strongest baselines.
>
---
#### [new 082] ID-Eraser: Proactive Defense Against Face Swapping via Identity Perturbation
- **分类: cs.CV**

- **简介: 该论文属于深度伪造防御任务，旨在解决面部交换带来的隐私安全问题。通过身份扰动和图像重建，提出ID-Eraser有效阻止恶意换脸。**

- **链接: [https://arxiv.org/pdf/2604.21465](https://arxiv.org/pdf/2604.21465)**

> **作者:** Junyan Luo; Peipeng Yu; Jianwei Fei; Shiya Zeng; Xiaoyu Zhou; Zhihua Xia; Xiang Liu
>
> **摘要:** Deepfake technologies have rapidly advanced with modern generative AI, and face swapping in particular poses serious threats to privacy and digital security. Existing proactive defenses mostly rely on pixel-level perturbations, which are ineffective against contemporary swapping models that extract robust high-level identity embeddings. We propose ID-Eraser, a feature-space proactive defense that removes identifiable facial information to prevent malicious face swapping. By injecting learnable perturbations into identity embeddings and reconstructing natural-looking protection images through a Face Revive Generator (FRG), ID-Eraser produces visually realistic results for humans while rendering the protected identities unusable for Deepfake models. Experiments show that ID-Eraser substantially disrupts identity recognition across diverse face recognition and swapping systems under strict black-box settings, achieving the lowest Top-1 accuracy (0.30) with the best FID (1.64) and LPIPS (0.020). Compared with swaps generated from clean inputs, the identity similarity of protected swaps drops sharply to an average of 0.504 across five representative face swapping models. ID-Eraser further demonstrates strong cross-dataset generalization, robustness to common distortions, and practical effectiveness on commercial APIs, reducing Tencent API similarity from 0.76 to 0.36.
>
---
#### [new 083] Addressing Image Authenticity When Cameras Use Generative AI
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像真实性任务，解决相机生成AI导致的图像内容失真问题，通过优化编码器和MLP解码器恢复未失真图像。**

- **链接: [https://arxiv.org/pdf/2604.21879](https://arxiv.org/pdf/2604.21879)**

> **作者:** Umar Masud; Abhijith Punnappurath; Luxi Zhao; David B. Lindell; Michael S. Brown
>
> **备注:** To appear in CVPR 2026 Workshop on Authenticity and Provenance in the Age of Generative AI
>
> **摘要:** The ability of generative AI (GenAI) methods to photorealistically alter camera images has raised awareness about the authenticity of images shared online. Interestingly, images captured directly by our cameras are considered authentic and faithful. However, with the increasing integration of deep-learning modules into cameras' capture-time hardware -- namely, the image signal processor (ISP) -- there is now a potential for hallucinated content in images directly output by our cameras. Hallucinated capture-time image content is typically benign, such as enhanced edges or texture, but in certain operations, such as AI-based digital zoom or low-light image enhancement, hallucinations can potentially alter the semantics and interpretation of the image content. As a result, users may not realize that the content in their camera images is not authentic. This paper addresses this issue by enabling users to recover the 'unhallucinated' version of the camera image to avoid misinterpretation of the image content. Our approach works by optimizing an image-specific multi-layer perceptron (MLP) decoder together with a modality-specific encoder so that, given the camera image, we can recover the image before hallucinated content was added. The encoder and MLP are self-contained and can be applied post-capture to the image without requiring access to the camera ISP. Moreover, the encoder and MLP decoder require only 180 KB of storage and can be readily saved as metadata within standard image formats such as JPEG and HEIC.
>
---
#### [new 084] Efficient Logic Gate Networks for Video Copy Detection
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文属于视频拷贝检测任务，旨在解决大规模场景下计算成本高、效率低的问题。提出基于逻辑门网络的框架，采用紧凑的逻辑表示替代传统方法，实现高效且准确的检测。**

- **链接: [https://arxiv.org/pdf/2604.21694](https://arxiv.org/pdf/2604.21694)**

> **作者:** Katarzyna Fojcik
>
> **摘要:** Video copy detection requires robust similarity estimation under diverse visual distortions while operating at very large scale. Although deep neural networks achieve strong performance, their computational cost and descriptor size limit practical deployment in high-throughput systems. In this work, we propose a video copy detection framework based on differentiable Logic Gate Networks (LGNs), which replace conventional floating-point feature extractors with compact, logic-based representations. Our approach combines aggressive frame miniaturization, binary preprocessing, and a trainable LGN embedding model that learns both logical operations and interconnections. After training, the model can be discretized into a purely Boolean circuit, enabling extremely fast and memory-efficient inference. We systematically evaluate different similarity strategies, binarization schemes, and LGN architectures across multiple dataset folds and difficulty levels. Experimental results demonstrate that LGN-based models achieve competitive or superior accuracy and ranking performance compared to prior models, while producing descriptors several orders of magnitude smaller and delivering inference speeds exceeding 11k samples per second. These findings indicate that logic-based models offer a promising alternative for scalable and resource-efficient video copy detection.
>
---
#### [new 085] Exploring the Role of Synthetic Data Augmentation in Controllable Human-Centric Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于可控人体视频生成任务，旨在解决真实人体视频数据稀缺的问题。通过研究合成数据的作用，提出一种扩散框架，探索其与真实数据的互补性，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2604.21291](https://arxiv.org/pdf/2604.21291)**

> **作者:** Yuanchen Fei; Yude Zou; Zejian Kang; Ming Li; Jiaying Zhou; Xiangru Huang
>
> **摘要:** Controllable human video generation aims to produce realistic videos of humans with explicitly guided motions and appearances,serving as a foundation for digital humans, animation, and embodied this http URL, the scarcity of largescale, diverse, and privacy safe human video datasets poses a major bottleneck, especially for rare identities and complex this http URL data provides a scalable and controllable alternative,yet its actual contribution to generative modeling remains underexplored due to the persistent Sim2Real this http URL this work,we systematically investigate the impact of synthetic data on controllable human video generation. We propose a diffusion-based framework that enables fine-grained control over appearance and motion while providing a unfied testbed to analyze how synthetic data interacts with real world data during training. Through extensive experiments, we reveal the complementary roles of synthetic and real data and demonstrate possible methods for efficiently selecting synthetic samples to enhance motion realism,temporal consistency,and identity this http URL study offers the first comprehensive exploration of synthetic data's role in human-centric video synthesis and provides practical insights for building data-efficient and generalizable generative models.
>
---
#### [new 086] Teacher-Guided Routing for Sparse Vision Mixture-of-Experts
- **分类: cs.CV**

- **简介: 该论文属于视觉任务，解决稀疏MoE训练中路由不稳定的问题。通过教师指导的路由方法，提升专家选择的稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2604.21330](https://arxiv.org/pdf/2604.21330)**

> **作者:** Masahiro Kada; Ryota Yoshihashi; Satoshi Ikehata; Rei Kawakami; Ikuro Sato
>
> **摘要:** Recent progress in deep learning has been driven by increasingly large-scale models, but the resulting computational cost has become a critical bottleneck. Sparse Mixture of Experts (MoE) offers an effective solution by activating only a small subset of experts for each input, achieving high scalability without sacrificing inference speed. Although effective, sparse MoE training exhibits characteristic optimization difficulties. Because the router receives informative gradients only through the experts selected in the forward pass, it suffers from gradient blocking and obtains little information from unselected routes. This limited, highly localized feedback makes it difficult for the router to learn appropriate expert-selection scores and often leads to unstable routing dynamics, such as fluctuating expert assignments during training. To address this issue, we propose TGR-MoE: Teacher-Guided Routing for Sparse Vision Mixture-of-Experts, a simple yet effective method that stabilizes router learning using supervision derived from a pretrained dense teacher model. TGR-MoE constructs a teacher router from the teacher's intermediate representations and uses its routing outputs as pseudo-supervision for the student router, suppressing frequent routing fluctuations during training and enabling knowledge-guided expert selection from the early stages of training. Extensive experiments on ImageNet-1K and CIFAR-100 demonstrate that TGR consistently improves both accuracy and routing consistency, while maintaining stable training even under highly sparse configurations.
>
---
#### [new 087] Micro-DualNet: Dual-Path Spatio-Temporal Network for Micro-Action Recognition
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文属于微动作识别任务，解决微动作多样化的时空特征问题。提出双路径网络，通过空间-时间与时间-空间路径协同处理，提升细粒度视频理解性能。**

- **链接: [https://arxiv.org/pdf/2604.21011](https://arxiv.org/pdf/2604.21011)**

> **作者:** Naga VS Raviteja Chappa; Evangelos Sariyanidi; Lisa Yankowitz; Gokul Nair; Casey J. Zampella; Robert T. Schultz; Birkan Tunç
>
> **备注:** Accepted to International Conference on Automatic Face and Gesture Recognition (FG)
>
> **摘要:** Micro-actions are subtle, localized movements lasting 1-3 seconds such as scratching one's head or tapping fingers. Such subtle actions are essential for social communication, ubiquitously used in natural interactions, and thus critical for fine-grained video understanding, yet remain poorly understood by current computer vision systems. We identify a fundamental challenge: micro-actions exhibit diverse spatio-temporal characteristics where some are defined by spatial configurations while others manifest through temporal dynamics. Existing methods that commit to a single spatio-temporal decomposition cannot accommodate this diversity. We propose a dual-path network that processes anatomically-grounded spatial entities through parallel Spatial-Temporal (ST) and Temporal-Spatial (TS) pathways. The ST path captures spatial configurations before modeling temporal dynamics, while the TS path inverts this order to prioritize temporal dynamics. Rather than fixed fusion, we introduce entity-level adaptive routing where each body part learns its optimal processing preference, complemented by Mutual Action Consistency (MAC) loss that enforces cross-path coherence. Extensive experiments demonstrate competitive performance on MA-52 dataset and state-of-the-art results on iMiGUE dataset. Our work reveals that architectural adaptation to the inherent complexity of micro-actions is essential for advancing fine-grained video understanding.
>
---
#### [new 088] FryNet: Dual-Stream Adversarial Fusion for Non-Destructive Frying Oil Oxidation Assessment
- **分类: cs.CV**

- **简介: 该论文属于食品质量检测任务，解决传统化学检测方法无法实时评估油品氧化的问题。提出FryNet模型，通过双流网络实现油品氧化状态的快速准确评估。**

- **链接: [https://arxiv.org/pdf/2604.21321](https://arxiv.org/pdf/2604.21321)**

> **作者:** Khaled R Ahmed; Toqi Tahamid Sarker; Taminul Islam; Tamany M Alanezi; Amer AbuGhazaleh
>
> **备注:** 10 pages, 7 figures, this paper has been submitted and accepted for publication at CVPRW 2026
>
> **摘要:** Monitoring frying oil degradation is critical for food safety, yet current practice relies on destructive wet-chemistry assays that provide no spatial information and are unsuitable for real-time use. We identify a fundamental obstacle in thermal-image-based inspection, the camera-fingerprint shortcut, whereby models memorize sensor-specific noise and thermal bias instead of learning oxidation chemistry, collapsing under video-disjoint evaluation. We propose FryNet, a dual-stream RGB-thermal framework that jointly performs oil-region segmentation, serviceability classification, and regression of four chemical oxidation indices (PV, p-AV, Totox, temperature) in a single forward pass. A ThermalMiT-B2 backbone with channel and spatial attention extracts thermal features, while an RGB-MAE Encoder learns chemically grounded representations via masked autoencoding and chemical alignment. Dual-Encoder DANN adversarially regularizes both streams against video identity via Gradient Reversal Layers, and FiLM fusion bridges thermal structure with RGB chemical context. On 7,226 paired frames across 28 frying videos, FryNet achieves 98.97% mIoU, 100% classification accuracy, and 2.32 mean regression MAE, outperforming all seven baselines.
>
---
#### [new 089] ARFBench: Benchmarking Time Series Question Answering Ability for Software Incident Response
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于时间序列问答任务，旨在评估模型对软件异常时间序列的理解能力。提出ARFBench基准，包含大量真实数据，并验证了多种模型效果，探索了混合方法与人类专家的互补性。**

- **链接: [https://arxiv.org/pdf/2604.21199](https://arxiv.org/pdf/2604.21199)**

> **作者:** Stephan Xie; Ben Cohen; Mononito Goswami; Junhong Shen; Emaad Khwaja; Chenghao Liu; David Asker; Othmane Abou-Amal; Ameet Talwalkar
>
> **摘要:** Time series question-answering (TSQA), in which we ask natural language questions to infer and reason about properties of time series, is a promising yet underexplored capability of foundation models. In this work, we present ARFBench, a TSQA benchmark that evaluates the understanding of multimodal foundation models (FMs) on time series anomalies prevalent in software incident data. ARFBench consists of 750 questions across 142 time series and 5.38M data points from 63 production incidents sourced exclusively from internal telemetry at Datadog. We evaluate leading proprietary and open-source LLMs, VLMs, and time series FMs and observe that frontier VLMs perform markedly better than existing baselines; the leading model (GPT-5) achieves a 62.7% accuracy and 51.9% F1. We next demonstrate the promise of specialized multimodal approaches. We develop a novel TSFM + VLM hybrid prototype which we post-train on a small set of synthetic and real data that yields comparable overall F1 and accuracy with frontier models. Lastly, we find models and human domain experts exhibit complementary strengths. We define a model-expert oracle, a best-of-2 oracle selector over model and expert answers, yielding 82.8% F1 and 87.2% accuracy and establishing a new superhuman frontier for future TSQA models. The benchmark is available at this https URL.
>
---
#### [new 090] Supervised Learning Has a Necessary Geometric Blind Spot: Theory, Consequences, and Minimal Repair
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于机器学习领域，研究监督学习中的几何盲区问题。通过理论分析揭示监督学习固有缺陷，并提出修复方法。**

- **链接: [https://arxiv.org/pdf/2604.21395](https://arxiv.org/pdf/2604.21395)**

> **作者:** Vishal Rajput
>
> **备注:** 29 pages. Code: this https URL. Preprint, not peer-reviewed. Affiliation: KU Leuven, Belgium
>
> **摘要:** We prove that empirical risk minimisation (ERM) imposes a necessary geometric constraint on learned representations: any encoder that minimises supervised loss must retain non-zero Jacobian sensitivity in directions that are label-correlated in training data but nuisance at test time. This is not a contingent failure of current methods; it is a mathematical consequence of the supervised objective itself. We call this the geometric blind spot of supervised learning (Theorem 1), and show it holds across proper scoring rules, architectures, and dataset sizes. This single theorem unifies four lines of prior empirical work that were previously treated separately: non-robust predictive features, texture bias, corruption fragility, and the robustness-accuracy tradeoff. In this framing, adversarial vulnerability is one consequence of a broader structural fact about supervised learning geometry. We introduce Trajectory Deviation Index (TDI), a diagnostic that measures the theorem's bounded quantity directly, and show why common alternatives miss the key failure mode. PGD adversarial training reaches Jacobian Frobenius 2.91 yet has the worst clean-input geometry (TDI 1.336), while PMH achieves TDI 0.904. TDI is the only metric that detects this dissociation because it measures isotropic path-length distortion -- the exact quantity Theorem 1 bounds. Across seven vision tasks, BERT/SST-2, and ImageNet ViT-B/16 backbones used by CLIP, DINO, and SAM, the blind spot is measurable and repairable. It is present at foundation-model scale, worsens monotonically across language-model sizes (blind-spot ratio 0.860 to 0.765 to 0.742 from 66M to 340M), and is amplified by task-specific ERM fine-tuning (+54%), while PMH repairs it by 11x with one additional training term whose Gaussian form Proposition 5 proves is the unique perturbation law that uniformly penalises the encoder Jacobian.
>
---
#### [new 091] A Deep U-Net Framework for Flood Hazard Mapping Using Hydraulic Simulations of the Wupper Catchment
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于洪水风险制图任务，旨在解决传统水力模拟计算成本高的问题。通过构建深度学习代理模型，实现高效准确的水位预测。**

- **链接: [https://arxiv.org/pdf/2604.21028](https://arxiv.org/pdf/2604.21028)**

> **作者:** Christian Lammers; Fernando Arévalo; Leonie Märker-Neuhaus; Daniel Heinenberg; Christian Förster; Karl-Heinz Spies
>
> **备注:** 18 Pages, 9 Figures
>
> **摘要:** The increasing frequency and severity of global flood events highlights the need for the development of rapid and reliable flood prediction tools. This process traditionally relies on computationally expensive hydraulic simulations. This research presents a prediction tool by developing a deep-learning based surrogate model to accurately and efficiently predict the maximum water level across a grid. This was achieved by conducting a series of experiments to optimize a U-Net architecture, patch generation, and data handling for approximating a hydraulic model. This research demonstrates that a deep learning surrogate model can serve as a computationally efficient alternative to traditional hydraulic simulations. The framework was tested using hydraulic simulations of the Wupper catchment in the North-Rhein Westphalia region (Germany), obtaining comparable results.
>
---
#### [new 092] AITP: Traffic Accident Responsibility Allocation via Multimodal Large Language Models
- **分类: cs.CL; cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于交通事故责任分配任务，旨在解决事故中因果推理与法律知识整合问题。提出AITP模型，结合多模态思维链和检索增强生成技术，提升责任判定准确性。**

- **链接: [https://arxiv.org/pdf/2604.20878](https://arxiv.org/pdf/2604.20878)**

> **作者:** Zijin Zhou; Songan Zhang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable progress in Traffic Accident Detection (TAD) and Traffic Accident Understanding (TAU). However, existing studies mainly focus on describing and interpreting accident videos, leaving room for deeper causal reasoning and integration of legal knowledge. Traffic Accident Responsibility Allocation (TARA) is a more challenging task that requires multi-step reasoning grounded in traffic regulations. To address this, we introduce AITP (Artificial Intelligence Traffic Police), a multimodal large language model for responsibility reasoning and allocation. AITP enhances reasoning via a Multimodal Chain-of-Thought (MCoT) mechanism and integrates legal knowledge through Retrieval-Augmented Generation (RAG). We further present DecaTARA, a decathlon-style benchmark unifying ten interrelated traffic accident reasoning tasks with 67,941 annotated videos and 195,821 question-answer pairs. Extensive experiments show that AITP achieves state-of-the-art performance across responsibility allocation, TAD, and TAU tasks, establishing a new paradigm for reasoning-driven multimodal traffic analysis.
>
---
#### [new 093] AttentionBender: Manipulating Cross-Attention in Video Diffusion Transformers as a Creative Probe
- **分类: cs.MM; cs.CV; cs.HC**

- **简介: 该论文提出AttentionBender，用于操控视频扩散Transformer中的交叉注意力机制，解决艺术家难以理解模型内部运作的问题。通过变换注意力图，实现对生成过程的创造性干预。**

- **链接: [https://arxiv.org/pdf/2604.20936](https://arxiv.org/pdf/2604.20936)**

> **作者:** Adam Cole; Mick Grierson
>
> **备注:** To appear in the Proceedings of the 2026 ACM Creativity and Cognition (C&C '26). 15 pages, 19 figures
>
> **摘要:** We present AttentionBender, a tool that manipulates cross-attention in Video Diffusion Transformers to help artists probe the internal mechanics of black-box video generation. While generative outputs are increasingly realistic, prompt-only control limits artists' ability to build intuition for the model's material process or to work beyond its default tendencies. Using an autobiographical research-through-design approach, we built on Network Bending to design AttentionBender, which applies 2D transforms (rotation, scaling, translation, etc.) to cross-attention maps to modulate generation. We assess AttentionBender by visualizing 4,500+ video generations across prompts, operations, and layer targets. Our results suggest that cross-attention is highly entangled: targeted manipulations often resist clean, localized control, producing distributed distortions and glitch aesthetics over linear edits. AttentionBender contributes a tool that functions both as an Explainable AI style probe of transformer attention mechanisms, and as a creative technique for producing novel aesthetics beyond the model's learned representational space.
>
---
#### [new 094] Symbolic Grounding Reveals Representational Bottlenecks in Abstract Visual Reasoning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究抽象视觉推理任务，解决VLM在Bongard问题上的瓶颈问题。通过对比视觉模型与基于符号输入的LLM，发现表示是主要瓶颈，符号输入可作为诊断上限。**

- **链接: [https://arxiv.org/pdf/2604.21346](https://arxiv.org/pdf/2604.21346)**

> **作者:** Mohit Vaishnav; Tanel Tammet
>
> **摘要:** Vision--language models (VLMs) often fail on abstract visual reasoning benchmarks such as Bongard problems, raising the question of whether the main bottleneck lies in reasoning or representation. We study this on Bongard-LOGO, a synthetic benchmark of abstract concept learning with ground-truth generative programs, by comparing end-to-end VLMs on raw images with large language models (LLMs) given symbolic inputs derived from those images. Using symbolic inputs as a diagnostic probe rather than a practical multimodal architecture, our \emph{Componential--Grammatical (C--G)} paradigm reformulates Bongard-LOGO as a symbolic reasoning task based on LOGO-style action programs or structured descriptions. LLMs achieve large and consistent gains, reaching mid--90s accuracy on Free-form problems, while a strong visual baseline remains near chance under matched task definitions. Ablations on input format, explicit concept prompts, and minimal visual grounding show that these factors matter much less than the shift from pixels to symbolic structure. These results identify representation as a key bottleneck in abstract visual reasoning and show how symbolic input can serve as a controlled diagnostic upper bound.
>
---
#### [new 095] Measure Twice, Click Once: Co-evolving Proposer and Visual Critic via Reinforcement Learning for GUI Grounding
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于GUI接地任务，解决自然语言到像素坐标精准映射的问题。提出协同进化框架，通过强化学习提升定位准确性和批评可靠性。**

- **链接: [https://arxiv.org/pdf/2604.21268](https://arxiv.org/pdf/2604.21268)**

> **作者:** Wenkai Wang; Xiyun Li; Hongcan Guo; Wenhao Yu; Tianqing Fang; Haitao Mi; Dong Yu; Shengyu Zhang
>
> **摘要:** Graphical User Interface (GUI) grounding requires mapping natural language instructions to precise pixel coordinates. However, due to visually homogeneous elements and dense layouts, models typically grasp semantic intent yet struggle with achieving precise localization. While scaling sampling attempts (Pass@k) reveals potential gains, static self-consistency strategies derived from geometric clustering often yield limited improvements, as the model's predictions tend to be spatially dispersed. In this paper, we propose replacing static consistency strategies with a learnable selection mechanism that selects the optimal target by critiquing its own proposals rendered on the screenshot. Given the significant disparity between the model's grounding and critiquing capabilities, we propose a co-evolving Propose-then-Critic framework. To jointly optimize these, we introduce a maturity-aware adaptive co-evolutionary reinforcement learning paradigm. This approach dynamically balances the training objectives of proposer and critic, where the diversity of the proposer's outputs enhances critic robustness, while the critic's maturing discrimination capability conversely unlocks the proposer's potential for extensive spatial exploration, fostering the mutual reinforcement and co-evolution of both capabilities, thereby ensuring generalizability to adapt to diverse and complex interface layouts. Extensive experiments over 6 benchmarks show that our method significantly enhances both grounding accuracy and critic reliability.
>
---
#### [new 096] Bridging the Training-Deployment Gap: Gated Encoding and Multi-Scale Refinement for Efficient Quantization-Aware Image Enhancement
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于图像增强任务，旨在解决模型训练与移动端部署间的性能差异问题。通过引入门控编码和多尺度优化，结合量化感知训练，提升模型在低精度下的表现。**

- **链接: [https://arxiv.org/pdf/2604.21743](https://arxiv.org/pdf/2604.21743)**

> **作者:** Dat To-Thanh; Nghia Nguyen-Trong; Hoang Vo; Hieu Bui-Minh; Tinh-Anh Nguyen-Nhu
>
> **备注:** 10 pages, 3 figures. Accepted at the Mobile AI (MAI) 2026 Workshop at CVPR 2026
>
> **摘要:** Image enhancement models for mobile devices often struggle to balance high output quality with the fast processing speeds required by mobile hardware. While recent deep learning models can enhance low-quality mobile photos into high-quality images, their performance is often degraded when converted to lower-precision formats for actual use on mobile phones. To address this training-deployment mismatch, we propose an efficient image enhancement model designed specifically for mobile deployment. Our approach uses a hierarchical network architecture with gated encoder blocks and multiscale refinement to preserve fine-grained visual features. Moreover, we incorporate Quantization-Aware Training (QAT) to simulate the effects of low-precision representation during the training process. This allows the network to adapt and prevents the typical drop in quality seen with standard post-training quantization (PTQ). Experimental results demonstrate that the proposed method produces high-fidelity visual output while maintaining the low computational overhead needed for practical use on standard mobile devices. The code will be available at this https URL.
>
---
#### [new 097] PanGuide3D: Cohort-Robust Pancreas Tumor Segmentation via Probabilistic Pancreas Conditioning and a Transformer Bottleneck
- **分类: q-bio.QM; cs.CV; cs.LG**

- **简介: 该论文属于胰腺肿瘤分割任务，旨在提升模型在不同数据集间的泛化能力。通过引入概率胰腺条件和Transformer瓶颈，提高分割准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.20981](https://arxiv.org/pdf/2604.20981)**

> **作者:** Sunny Joy Ma; Xiang Ma
>
> **摘要:** Pancreatic tumor segmentation in contrast-enhanced computed tomography (CT) is clinically important yet technically challenging: lesions are often small, heterogeneous, and easily confused with surrounding soft tissue, and models that perform well on one cohort frequently degrade under cohort shift. Our goal is to improve cross-cohort generalization while keeping the model architecture simple, efficient, and practical for 3D CT segmentation. We introduce PanGuide3D, a cohort-robust architecture with a shared 3D encoder, a pancreas decoder that predicts a probabilistic pancreas map, and a tumor decoder that is explicitly conditioned on this pancreas probability at multiple scales via differentiable soft gating. To capture long-range context under distribution shift, we further add a lightweight Transformer bottleneck in the U-Net bottleneck representation. We evaluate cohort transfer by training on the PanTS (Pancreatic Tumor Segmentation) cohort and testing both in-cohort (PanTS) and out-of-cohort on MSD (Medical Segmentation Decathlon) Task07 Pancreas, using matched preprocessing and training protocols across strong baselines. We collect voxel-level segmentation metrics, patient-level tumor detection, subgroup analyses by tumor size and anatomical location, volume-conditioned performance analyses, and calibration measurements to assess reliability. Across the evaluated models, PanGuide3D achieves the best overall tumor performance and shows improved cross-cohort generalization, particularly for small tumors and challenging anatomical locations, while reducing anatomically implausible false positives. These findings support probabilistic anatomical conditioning as a practical strategy for improving cross-cohort robustness in an end-to-end model and suggest potential utility for contouring support, treatment planning, and multi-institutional studies.
>
---
#### [new 098] Beyond Single Plots: A Benchmark for Question Answering on Multi-Charts
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于多图表问答任务，旨在解决多图表信息理解问题。构建了PolyChartQA数据集，并评估了多种模型的表现。**

- **链接: [https://arxiv.org/pdf/2604.21344](https://arxiv.org/pdf/2604.21344)**

> **作者:** Azher Ahmed Efat; Seok Hwan Song; Wallapak Tavanapong
>
> **摘要:** Charts are widely used to present complex information. Deriving meaningful insights in real-world contexts often requires interpreting multiple related charts together. Research on understanding multi-chart images has not been extensively explored. We introduce PolyChartQA, a mid-scale dataset specifically designed for question answering over multi-chart images. PolyChartQA comprises 534 multi-chart images (with a total of 2,297 sub-charts) sourced from peer-reviewed computer science research publications and 2,694 QA pairs. We evaluate the performance of nine state-of-the-art Multimodal Language Models (MLMs) on PolyChartQA across question type, difficulty, question source, and key structural characteristics of multi-charts. Our results show a 27.4% LLM-based accuracy (L-Accuracy) drop on human-authored questions compared to MLM-generated questions, and a 5.39% L-accuracy gain with our proposed prompting method.
>
---
#### [new 099] StyleID: A Perception-Aware Dataset and Metric for Stylization-Agnostic Facial Identity Recognition
- **分类: cs.GR; cs.CV; cs.HC; cs.MM**

- **简介: 该论文属于人脸识别任务，旨在解决stylization下身份识别不一致的问题。提出StyleID数据集和评估框架，提升模型对不同风格图像的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.21689](https://arxiv.org/pdf/2604.21689)**

> **作者:** Kwan Yun; Changmin Lee; Ayeong Jeong; Youngseo Kim; Seungmi Lee; Junyong Noh
>
> **备注:** SIGGRAPH 2026 / ACM TOG. Project page at this https URL
>
> **摘要:** Creative face stylization aims to render portraits in diverse visual idioms such as cartoons, sketches, and paintings while retaining recognizable identity. However, current identity encoders, which are typically trained and calibrated on natural photographs, exhibit severe brittleness under stylization. They often mistake changes in texture or color palette for identity drift or fail to detect geometric exaggerations. This reveals the lack of a style-agnostic framework to evaluate and supervise identity consistency across varying styles and strengths. To address this gap, we introduce StyleID, a human perception-aware dataset and evaluation framework for facial identity under stylization. StyleID comprises two datasets: (i) StyleBench-H, a benchmark that captures human same-different verification judgments across diffusion- and flow-matching-based stylization at multiple style strengths, and (ii) StyleBench-S, a supervision set derived from psychometric recognition-strength curves obtained through controlled two-alternative forced-choice (2AFC) experiments. Leveraging StyleBench-S, we fine-tune existing semantic encoders to align their similarity orderings with human perception across styles and strengths. Experiments demonstrate that our calibrated models yield significantly higher correlation with human judgments and enhanced robustness for out-of-domain, artist drawn portraits. All of our datasets, code, and pretrained models are publicly available at this https URL
>
---
#### [new 100] DiffNR: Diffusion-Enhanced Neural Representation Optimization for Sparse-View 3D Tomographic Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于3D重建任务，解决稀疏视角CT重建中的伪影问题。提出DiffNR框架，结合扩散模型优化神经表示，提升重建质量与效率。**

- **链接: [https://arxiv.org/pdf/2604.21518](https://arxiv.org/pdf/2604.21518)**

> **作者:** Shiyan Su; Ruyi Zha; Danli Shi; Hongdong Li; Xuelian Cheng
>
> **备注:** Accepted to AAAI 2026. Project page: this https URL
>
> **摘要:** Neural representations (NRs), such as neural fields and 3D Gaussians, effectively model volumetric data in computed tomography (CT) but suffer from severe artifacts under sparse-view settings. To address this, we propose DiffNR, a novel framework that enhances NR optimization with diffusion priors. At its core is SliceFixer, a single-step diffusion model designed to correct artifacts in degraded slices. We integrate specialized conditioning layers into the network and develop tailored data curation strategies to support model finetuning. During reconstruction, SliceFixer periodically generates pseudo-reference volumes, providing auxiliary 3D perceptual supervision to fix underconstrained regions. Compared to prior methods that embed CT solvers into time-consuming iterative denoising, our repair-and-augment strategy avoids frequent diffusion model queries, leading to better runtime performance. Extensive experiments show that DiffNR improves PSNR by 3.99 dB on average, generalizes well across domains, and maintains efficient optimization.
>
---
#### [new 101] Robust Test-time Video-Text Retrieval: Benchmarking and Adapting for Query Shifts
- **分类: cs.IR; cs.AI; cs.CV**

- **简介: 该论文属于视频-文本检索任务，解决模型在真实查询分布变化下的性能下降问题。通过构建基准和提出HAT-VTR框架，提升模型的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.20851](https://arxiv.org/pdf/2604.20851)**

> **作者:** Bingqing Zhang; Zhuo Cao; Heming Du; Yang Li; Xue Li; Jiajun Liu; Sen Wang
>
> **备注:** Accepted to ICLR2026
>
> **摘要:** Modern video-text retrieval (VTR) models excel on in-distribution benchmarks but are highly vulnerable to real-world query shifts, where the distribution of query data deviates from the training domain, leading to a sharp performance drop. Existing image-focused robustness solutions are inadequate to handle this vulnerability in video, as they fail to address the complex spatio-temporal dynamics inherent in these shifts. To systematically evaluate this vulnerability, we first introduce a comprehensive benchmark featuring 12 distinct types of video perturbations across five severity degrees. Analysis on this benchmark reveals that query shifts amplify the hubness phenomenon, where a few gallery items become dominant "hubs" that attract a disproportionate number of queries. To mitigate this, we then propose HAT-VTR (Hubness Alleviation for Test-time Video-Text Retrieval), as our baseline test-time adaptation framework designed to directly counteract hubness in VTR. It leverages two key components: a Hubness Suppression Memory to refine similarity scores, and multi-granular losses to enforce temporal feature consistency. Extensive experiments demonstrate that HAT-VTR substantially improves robustness, consistently outperforming prior methods across diverse query shift scenarios, and enhancing model reliability for real-world applications.
>
---
#### [new 102] Neuro-Symbolic Manipulation Understanding with Enriched Semantic Event Chains
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作理解任务，旨在提升动作推理的准确性与鲁棒性。通过改进的eSEC-LAM框架，将语义事件链转化为更丰富的符号状态，增强决策与解释能力。**

- **链接: [https://arxiv.org/pdf/2604.21053](https://arxiv.org/pdf/2604.21053)**

> **作者:** Fatemeh Ziaeetabar
>
> **摘要:** Robotic systems operating in human environments must reason about how object interactions evolve over time, which actions are currently being performed, and what manipulation step is likely to follow. Classical enriched Semantic Event Chains (eSECs) provide an interpretable relational description of manipulation, but remain primarily descriptive and do not directly support uncertainty-aware decision making. In this paper, we propose eSEC-LAM, a neuro-symbolic framework that transforms eSECs into an explicit event-level symbolic state for manipulation understanding. The proposed formulation augments classical eSECs with confidence-aware predicates, functional object roles, affordance priors, primitive-level abstraction, and saliency-guided explanation cues. These enriched symbolic states are derived from a foundation-model-based perception front-end through deterministic predicate extraction, while current-action inference and next-primitive prediction are performed using lightweight symbolic reasoning over primitive pre- and post-conditions. We evaluate the proposed framework on EPIC-KITCHENS-100, EPIC-KITCHENS VISOR, and Assembly101 across action recognition, next-primitive prediction, robustness to perception noise, and explanation consistency. Experimental results show that eSEC-LAM achieves competitive action recognition, substantially improves next-primitive prediction, remains more robust under degraded perceptual conditions than both classical symbolic and end-to-end video baselines, and provides temporally consistent explanation traces grounded in explicit relational evidence. These findings demonstrate that enriched Semantic Event Chains can serve not only as interpretable descriptors of manipulation, but also as effective internal states for neuro-symbolic action reasoning.
>
---
## 更新

#### [replaced 001] LRDUN: A Low-Rank Deep Unfolding Network for Efficient Spectral Compressive Imaging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18513](https://arxiv.org/pdf/2511.18513)**

> **作者:** He Huang; Yujun Guo; Wei He
>
> **备注:** 17 pages, 16 figures,
>
> **摘要:** Deep unfolding networks (DUNs) have achieved remarkable success and become the mainstream paradigm for spectral compressive imaging (SCI) reconstruction. Existing DUNs are derived from full-HSI imaging models, where each stage operates directly on the high-dimensional HSI, refining the entire data cube based on the single 2D coded measurement. However, this paradigm leads to computational redundancy and suffers from the ill-posed nature of mapping 2D residuals back to 3D space of HSI. In this paper, we propose two novel imaging models corresponding to the spectral basis and subspace image by explicitly integrating low-rank (LR) decomposition with the sensing model. Compared to recovering the full HSI, estimating these compact low-dimensional components significantly mitigates the ill-posedness. Building upon these novel models, we develop the Low-Rank Deep Unfolding Network (LRDUN), which jointly solves the two subproblems within an unfolded proximal gradient descent (PGD) framework. Furthermore, we introduce a Generalized Feature Unfolding Mechanism (GFUM) that decouples the physical rank in the data-fidelity term from the feature dimensionality in the prior module, enhancing the representational capacity and flexibility of the network. Extensive experiments on simulated and real datasets demonstrate that the proposed LRDUN achieves state-of-the-art (SOTA) reconstruction quality with significantly reduced computational cost.
>
---
#### [replaced 002] Dehaze-then-Splat: Generative Dehazing with Physics-Informed 3D Gaussian Splatting for Smoke-Free Novel View Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.13589](https://arxiv.org/pdf/2604.13589)**

> **作者:** Boss Chen; Hanqing Wang
>
> **摘要:** We present Dehaze-then-Splat, a two-stage pipeline for multi-view smoke removal and novel view synthesis developed for Track~2 of the NTIRE 2026 3D Restoration and Reconstruction Challenge. In the first stage, we produce pseudo-clean training images via per-frame generative dehazing using Nano Banana Pro, followed by brightness normalization. In the second stage, we train 3D Gaussian Splatting (3DGS) with physics-informed auxiliary losses -- depth supervision via Pearson correlation with pseudo-depth, dark channel prior regularization, and dual-source gradient matching -- that compensate for cross-view inconsistencies inherent in frame-wise generative processing. We identify a fundamental tension in dehaze-then-reconstruct pipelines: per-image restoration quality does not guarantee multi-view consistency, and such inconsistency manifests as blurred renders and structural instability in downstream 3D this http URL analysis shows that MCMC-based densification with early stopping, combined with depth and haze-suppression priors, effectively mitigates these artifacts. On the Akikaze validation scene, our pipeline achieves 20.98\,dB PSNR and 0.683 SSIM for novel view synthesis, a +1.50\,dB improvement over the unregularized baseline.
>
---
#### [replaced 003] RefAerial: A Benchmark and Approach for Referring Detection in Aerial Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.20543](https://arxiv.org/pdf/2604.20543)**

> **作者:** Guyue Hu; Hao Song; Yuxing Tong; Duzhi Yuan; Dengdi Sun; Aihua Zheng; Chenglong Li; Jin Tang
>
> **摘要:** Referring detection refers to locate the target referred by natural languages, which has recently attracted growing research interests. However, existing datasets are limited to ground images with large object centered in relative small scenes. This paper introduces a large-scale challenging dataset for referring detection in aerial images, termed as RefAerial. It distinguishes from conventional ground referring detection datasets by 4 characteristics: (1) low but diverse object-to-scene ratios, (2) numerous targets and distractors, (3)complex and fine-grained referring descriptions, (4) diverse and broad scenes in the aerial view. We also develop a human-in-the-loop referring expansion and annotation engine (REA-Engine) for efficient semi-automated referring pair annotation. Besides, we observe that existing ground referring detection approaches exhibiting serious performance degradation on our aerial dataset since the intrinsic scale variety issue within or across aerial images. Therefore, we further propose a novel scale-comprehensive and sensitive (SCS) framework for referring detection in aerial images. It consists of a mixture-of-granularity (MoG) attention and a two-stage comprehensive-to-sensitive (CtS) decoding strategy. Specifically, the mixture-of-granularity attention is developed for scale-comprehensive target understanding. In addition, the two-stage comprehensive-to-sensitive decoding strategy is designed for coarse-to-fine referring target decoding. Eventually, the proposed SCS framework achieves remarkable performance on our aerial referring detection dataset and even promising performance boost on conventional ground referring detection datasets.
>
---
#### [replaced 004] ViPS: Video-informed Pose Spaces for Auto-Rigged Meshes
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2604.17623](https://arxiv.org/pdf/2604.17623)**

> **作者:** Honglin Chen; Karran Pandey; Rundi Wu; Matheus Gadelha; Yannick Hold-Geoffroy; Ayush Tewari; Niloy J. Mitra; Changxi Zheng; Paul Guerrero
>
> **备注:** Project page: this https URL
>
> **摘要:** Kinematic rigs provide a structured interface for articulating 3D meshes, but they lack an inherent representation of the plausible manifold of joint configurations for a given asset. Without such a pose space, stochastic sampling or manual manipulation of raw rig parameters often leads to semantic or geometric violations, such as anatomical hyperextension and non-physical self-intersections. We propose Video-informed Pose Spaces (ViPS), a feed-forward framework that discovers the latent distribution of valid articulations for auto-rigged meshes by distilling motion priors from a pretrained video diffusion model. Unlike existing methods that rely on scarce artist-authored 4D datasets, ViPS transfers generative video priors into a universal distribution over a given rig parameterization. Differentiable geometric validators applied to the skinned mesh enforce asset-specific validity without requiring manual regularizers. Our model learns a smooth, compact, and controllable pose space that supports diverse sampling, manifold projection for inverse kinematics, and temporally coherent trajectories for keyframing. Furthermore, the distilled 3D pose samples serve as precise semantic proxies for guiding video diffusion, effectively closing the loop between generative 2D priors and structured 3D kinematic control. Our evaluations show that ViPS, trained solely on video priors, matches the performance of state-of-the-art methods trained on synthetic artist-created 4D data in both plausibility and diversity. Most importantly, as a universal model, ViPS demonstrates robust zero-shot generalization to out-of-distribution species and unseen skeletal topologies.
>
---
#### [replaced 005] Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction for Sparse-View CT
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.13576](https://arxiv.org/pdf/2509.13576)**

> **作者:** Haodong Li; Shuo Han; Haiyang Mao; Yu Shi; Changsheng Fang; Jianjia Zhang; Weiwen Wu; Hengyong Yu
>
> **备注:** 17 pages, 15 figures, accepted by IEEE Transactions on Medical Imaging
>
> **摘要:** Sparse-View CT (SVCT) reconstruction enhances temporal resolution and reduces radiation dose, yet its clinical use is hindered by artifacts due to view reduction and domain shifts from scanner, protocol, or anatomical variations, leading to performance degradation in out-of-distribution (OOD) scenarios. In this work, we propose a Cross-Distribution Diffusion Priors-Driven Iterative Reconstruction (CDPIR) framework to tackle the OOD problem in SVCT. CDPIR integrates cross-distribution diffusion priors, derived from a Scalable Interpolant Transformer (SiT), with model-based iterative reconstruction methods. Specifically, we train a SiT backbone, an extension of the Diffusion Transformer (DiT) architecture, to establish a unified stochastic interpolant framework, leveraging Classifier-Free Guidance (CFG) across multiple datasets. By randomly dropping the conditioning with a null embedding during training, the model learns both domain-specific and domain-invariant priors, enhancing generalizability. During sampling, the globally sensitive transformer-based diffusion model exploits the cross-distribution prior within the unified stochastic interpolant framework, enabling flexible and stable control over multi-distribution-to-noise interpolation paths and decoupled sampling strategies, thereby improving adaptation to OOD reconstruction. By alternating between data fidelity and sampling updates, our model achieves state-of-the-art performance with superior detail preservation in SVCT reconstructions. Extensive experiments demonstrate that CDPIR significantly outperforms existing approaches, particularly under OOD conditions, highlighting its robustness and potential clinical value in challenging imaging scenarios.
>
---
#### [replaced 006] TV Subgradient-Guided Multi-Source Fusion for Spectral Imaging in Dual-Camera CASSI Systems
- **分类: cs.CV; physics.optics**

- **链接: [https://arxiv.org/pdf/2509.10897](https://arxiv.org/pdf/2509.10897)**

> **作者:** Weiqiang Zhao; Tianzhu Liu; Yuzhe Gui; Wei Bian; Yanfeng Gu
>
> **备注:** Main text: 14 pages, 12 figures; Supplementary material: 8 pages, 3 figures
>
> **摘要:** Balancing spectral, spatial, and temporal resolutions is a key challenge in spectral imaging. The Dual-Camera Coded Aperture Snapshot Spectral Imaging (DC-CASSI) system alleviates this trade-off but suffers from severely ill-posed reconstruction problems due to its high compression ratio. Existing methods are constrained by scene-specific tuning or excessive reliance on paired training data. To address these issues, we propose a Total Variation (TV) subgradient-guided multi-source fusion framework for DC-CASSI reconstruction, comprising three core components: (1) An end-to-end Single-Disperser CASSI (SD-CASSI) observation model based on the tensor-form Kronecker $\delta$, which establishes a rigorous mathematical foundation for physical constraints while enabling efficient adjoint operator implementation; (2) An adaptive spatial reference generator that integrates SD-CASSI's physical model and RGB subspace constraint, generating the reference image as reliable spatial prior; (3) A TV subgradient-guided regularization term that encodes local structural directions from the reference image into spectral reconstruction, achieving high-quality fused results. The framework is validated on simulated datasets and real-world datasets. Experimental results demonstrate that it achieves state-of-the-art reconstruction performance and robust noise resilience. This work not only establishes an interpretable theoretical foundation for subgradient-guided fusion but also provides a practical fusion-based paradigm for high-fidelity spectral image reconstruction in DC-CASSI systems. Source code: this https URL.
>
---
#### [replaced 007] Human Presence Detection via Wi-Fi Range-Filtered Doppler Spectrum on Commodity Laptops
- **分类: eess.SP; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10845](https://arxiv.org/pdf/2603.10845)**

> **作者:** Jessica Sanson; Rahul C. Shah; Valerio Frascolla
>
> **备注:** 6 pages, Conference
>
> **摘要:** Human Presence Detection (HPD) is key to enable intelligent power management and security features in everyday devices. In this paper we propose the first HPD solution that leverages monostatic Wi-Fi sensing and detects user position using only the built-in Wi-Fi hardware of a device, with no need for external devices, access points, or additional sensors. In contrast, existing HPD solutions for laptops require external dedicated sensors which add cost and complexity, or rely on camera-based approaches that introduce significant privacy concerns. We herewith introduce the Range-Filtered Doppler Spectrum (RF-DS), a novel Wi-Fi sensing technique for presence estimation that enables both range-selective and temporally windowed detection of user presence. By applying targeted range-area filtering in the Channel Impulse Response (CIR) domain before Doppler analysis, our method focuses processing on task-relevant spatial zones, significantly reducing computational complexity. In addition, the use of temporal windows in the spectrum domain provides greater estimator stability compared to conventional 2D Range-Doppler detectors. Furthermore, we propose an adaptive multi-rate processing framework that dynamically adjusts Channel State Information (CSI) sampling rates-operating at low frame rates (10Hz) during idle periods and high rates (100Hz) only when motion is detected. To our knowledge, this is the first low-complexity solution for occupancy detection using monostatic Wi-Fi sensing on a built-in Wi-Fi network interface controller (NIC) of a commercial off-the-shelf laptop that requires no external network infrastructure or specialized sensors. Our solution can scale across different environments and devices without calibration or retraining.
>
---
#### [replaced 008] From Image to Music Language: A Two-Stage Structure Decoding Approach for Complex Polyphonic OMR
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于光学音乐识别（OMR）任务，解决复杂多声部乐谱的结构解码问题。通过两阶段流程，提出BeadSolver方法提升音符结构准确性。**

- **链接: [https://arxiv.org/pdf/2604.20522](https://arxiv.org/pdf/2604.20522)**

> **作者:** Nan Xu; Shiheng Li; Shengchao Hou
>
> **备注:** 49 pages, 16 figures, 16 tables
>
> **摘要:** We propose a new approach for a practical two-stage Optical Music Recognition (OMR) pipeline, with a particular focus on its second stage. Given symbol and event candidates from the visual pipeline, we decode them into an editable, verifiable, and exportable score structure. We focus on complex polyphonic staff notation, especially piano scores, where voice separation and intra-measure timing are the main bottlenecks. Our approach formulates second-stage decoding as a structure decoding problem and uses topology recognition with probability-guided search (BeadSolver) as its core method. We also describe a data strategy that combines procedural generation with recognition-feedback annotations. The result is a practical decoding component for real OMR systems and a path to accumulate structured score data for future end-to-end, multimodal, and RL-style methods.
>
---
#### [replaced 009] Demystifying Action Space Design for Robotic Manipulation Policies
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操控策略学习任务，旨在解决动作空间设计对策略学习的影响问题。通过大量实验，分析了不同动作表示方式的优劣，提出了优化动作空间设计的方法。**

- **链接: [https://arxiv.org/pdf/2602.23408](https://arxiv.org/pdf/2602.23408)**

> **作者:** Yuchun Feng; Jinliang Zheng; Zhihao Wang; Dongxiu Liu; Jianxiong Li; Jiangmiao Pang; Tai Wang; Xianyuan Zhan
>
> **摘要:** The specification of the action space plays a pivotal role in imitation-based robotic manipulation policy learning, fundamentally shaping the optimization landscape of policy learning. While recent advances have focused heavily on scaling training data and model capacity, the choice of action space remains guided by ad-hoc heuristics or legacy designs, leading to an ambiguous understanding of robotic policy design philosophies. To address this ambiguity, we conducted a large-scale and systematic empirical study, confirming that the action space does have significant and complex impacts on robotic policy learning. We dissect the action design space along temporal and spatial axes, facilitating a structured analysis of how these choices govern both policy learnability and control stability. Based on 13,000+ real-world rollouts on a bimanual robot and evaluation on 500+ trained models over four scenarios, we examine the trade-offs between absolute vs. delta representations, and joint-space vs. task-space parameterizations. Our large-scale results suggest that properly designing the policy to predict delta actions consistently improves performance, while joint-space and task-space representations offer complementary strengths, favoring control stability and generalization, respectively.
>
---
#### [replaced 010] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.03956](https://arxiv.org/pdf/2604.03956)**

> **作者:** Ravi Ranjan; Agoritsa Polyzou
>
> **备注:** 18 pages, 9 figures, Accepted to ACL-2026, KnowFM
>
> **摘要:** Vision-language-action (VLA) models are emerging as embodied foundation models for robotic manipulation, but their deployment introduces a new unlearning challenge: removing unsafe, spurious, or privacy-sensitive behaviors without degrading perception, language grounding, and action control. In OpenVLA-style policies, behavior is produced through a fused visual encoder, a cross-modal projector, and a language backbone that predicts tokenized robot actions, so undesirable knowledge can be distributed across perception, alignment, and reasoning/action layers rather than confined to a single module. Consequently, partial unlearning applied only to the vision stack or only to the language backbone is often insufficient, while conventional unlearning baselines designed for standalone vision or language models may leave residual forgetting or incur unnecessary utility loss in embodied settings. We propose VLA-Forget, a hybrid unlearning framework that combines ratio-aware selective editing for perception and cross-modal specificity with layer-selective reasoning/action unlearning for utility-preserving forgetting. VLA-Forget jointly optimizes three objectives: targeted forgetting, perceptual preservation, and reasoning retention, through staged updates over the visual encoder, projector, and upper action-generating transformer blocks. Across forget-set behavior probes and retain-task evaluations, VLA-Forget improves forgetting efficacy by 10%, preserves perceptual specificity by 22%, retains reasoning and task success by 9%, and reduces post-quantization recovery by 55% relative to strong unlearning baselines.
>
---
#### [replaced 011] Counterfactual Segmentation Reasoning: Diagnosing and Mitigating Pixel-Grounding Hallucination
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦于视觉语言模型中的分割幻觉问题，提出Counterfactual Segmentation Reasoning任务，通过构建基准和改进模型以减少错误分割。**

- **链接: [https://arxiv.org/pdf/2506.21546](https://arxiv.org/pdf/2506.21546)**

> **作者:** Xinzhuo Li; Adheesh Juvekar; Jiaxun Zhang; Xingyou Liu; Muntasir Wahed; Kiet A. Nguyen; Yifan Shen; Tianjiao Yu; Ismini Lourentzou
>
> **备注:** Project webpage: this https URL
>
> **摘要:** Segmentation Vision-Language Models (VLMs) have significantly advanced grounded visual understanding, yet they remain prone to pixel-grounding hallucinations, producing masks for incorrect objects or for objects that are entirely absent. Existing evaluations rely almost entirely on text- or label-based perturbations, which check only whether the predicted mask matches the queried label. Such evaluations overlook the spatial footprint and severity of hallucination and therefore fail to reveal vision-driven hallucinations, which are more challenging and more prevalent. To address this gap, we formalize the task of Counterfactual Segmentation Reasoning (CSR), where a model must segment the referenced object in the factual image and abstain in its counterfactual counterpart. To support this task, we curate HalluSegBench, the first large-scale benchmark to diagnose referring and reasoning expression segmentation hallucinations using controlled visual counterfactuals, alongside new evaluation metrics that measure hallucination severity and disentangle vision- and language-driven failure modes. We further introduce RobustSeg, a segmentation VLM trained with counterfactual fine-tuning (CFT) to learn when to segment and when to abstain. Experimental results confirm RobustSeg reduces hallucinations by 30%, while improving segmentation performance on FP-RefCOCO(+/g).
>
---
#### [replaced 012] Federated Learning for Surgical Vision in Appendicitis Classification: Results of the FedSurg EndoVis 2024 Challenge
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.04772](https://arxiv.org/pdf/2510.04772)**

> **作者:** Max Kirchner; Hanna Hoffmann; Alexander C. Jenke; Oliver L. Saldanha; Kevin Pfeiffer; Weam Kanjo; Julia Alekseenko; Claas de Boer; Santhi Raj Kolamuri; Lorenzo Mazza; Nicolas Padoy; Sophia Bano; Annika Reinke; Lena Maier-Hein; Danail Stoyanov; Jakob N. Kather; Fiona R. Kolbinger; Sebastian Bodenstedt; Stefanie Speidel
>
> **备注:** A challenge report pre-print (31 pages), including 7 tables and 8 figures
>
> **摘要:** Developing generalizable surgical AI requires multi-institutional data, yet patient privacy constraints preclude direct data sharing, making Federated Learning (FL) a natural candidate solution. The application of FL to complex, spatiotemporal surgical video data remains largely unbenchmarked. We present the FedSurg Challenge, the first international benchmarking initiative dedicated to FL in surgical vision, evaluated as a proof-of-concept on a multi-center laparoscopic appendectomy dataset (preliminary subset of Appendix300). Three submissions were evaluated on generalization to an unseen center and center-specific adaptation. Centralized and Swarm Learning baselines isolate the contributions of task difficulty and decentralization to observed performance. Even with all data pooled centrally, the task achieved only 26.31\% F1-score on the unseen center, while decentralized training introduced an additional, separable performance penalty. Temporal modeling emerges as the dominant architectural factor: video-level spatiotemporal models consistently outperformed frame-level approaches regardless of aggregation strategy. Naive local fine-tuning leads to classifier collapse on imbalanced local data; structured personalized FL with parameter-efficient fine-tuning represents a more principled path toward center-specific adaptation. By characterizing current FL limitations through rigorous statistical analysis, this work establishes a methodological reference point for robust, privacy-preserving AI systems in surgical video analysis.
>
---
#### [replaced 013] Adaptive Moments are Surprisingly Effective for Plug-and-Play Diffusion Sampling
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16797](https://arxiv.org/pdf/2603.16797)**

> **作者:** Christian Belardi; Justin Lovelace; Kilian Q. Weinberger; Carla P. Gomes
>
> **摘要:** Guided diffusion sampling relies on approximating often intractable likelihood scores, which introduces significant noise into the sampling dynamics. We propose using adaptive moment estimation to stabilize these noisy likelihood scores during sampling. Despite its simplicity, our approach achieves state-of-the-art results on image restoration and class-conditional generation tasks, outperforming more complicated methods, which are often computationally more expensive. We provide empirical analysis of our method on both synthetic and real data, demonstrating that mitigating gradient noise through adaptive moments offers an effective way to improve alignment.
>
---
#### [replaced 014] ATATA: One Algorithm to Align Them All
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11194](https://arxiv.org/pdf/2601.11194)**

> **作者:** Boyi Pang; Savva Ignatyev; Vladimir Ippolitov; Ramil Khafizov; Yurii Melnik; Oleg Voynov; Maksim Nakhodnov; Aibek Alanov; Xiaopeng Fan; Peter Wonka; Evgeny Burnaev
>
> **摘要:** We suggest a new multi-modal algorithm for joint inference of paired structurally aligned samples with Rectified Flow models. While some existing methods propose a codependent generation process, they do not view the problem of joint generation from a structural alignment perspective. Recent work uses Score Distillation Sampling to generate aligned 3D models, but SDS is known to be time-consuming, prone to mode collapse, and often provides cartoonish results. By contrast, our suggested approach relies on the joint transport of a segment in the sample space, yielding faster computation at inference time. Our approach can be built on top of an arbitrary Rectified Flow model operating on the structured latent space. We show the applicability of our method to the domains of image, video, and 3D shape generation using state-of-the-art baselines and evaluate it against both editing-based and joint inference-based competing approaches. We demonstrate a high degree of structural alignment for the sample pairs obtained with our method and a high visual quality of the samples. Our method improves the state-of-the-art for image and video generation pipelines. For 3D generation, it is able to show comparable quality while working orders of magnitude faster.
>
---
#### [replaced 015] Tumor-anchored deep feature random forests for out-of-distribution detection in lung cancer segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.08216](https://arxiv.org/pdf/2512.08216)**

> **作者:** Aneesh Rangnekar; Harini Veeraraghavan
>
> **备注:** Accepted for publication in Transactions on Machine Learning Research (TMLR), 2026. Code available at: this https URL
>
> **摘要:** Accurate segmentation of lung tumors from 3D computed tomography (CT) scans is essential for automated treatment planning and response assessment. Despite self-supervised pretraining on numerous datasets, state-of-the-art transformer backbones remain susceptible to out-of-distribution (OOD) inputs, often producing confidently incorrect segmentations with potential for risk in clinical deployment. Hence, we introduce RF-Deep, a lightweight post-hoc random forests-based framework that leverages deep features trained with limited outlier exposure, requiring as few as 40 labeled scans (20 in-distribution and 20 OOD), to improve scan-level OOD detection. RF-Deep repurposes the hierarchical features from the pretrained-then-finetuned segmentation backbones, aggregating features from multiple regions-of-interest anchored to predicted tumor regions to capture OOD likelihood. We evaluated RF-Deep on 2,232 CT volumes spanning near-OOD (pulmonary embolism, COVID-19 negative) and far-OOD (kidney cancer, healthy pancreas) datasets. RF-Deep achieved AUROC >~93 on the challenging near-OOD datasets, where it outperformed the next best method by 4--7 percentage points, and produced near-perfect detection (AUROC >~99) on far-OOD datasets. The approach also showed transferability to two blinded validation datasets under the ensemble configuration (COVID-19 positive and breast cancer; AUROC >~94). RF-Deep maintained consistent performance across backbones of different depths and pretraining strategies, demonstrating applicability of post-hoc detectors as a safety filter for clinical deployment of tumor segmentation pipelines.
>
---
#### [replaced 016] Find the Differences: Differential Morphing Attack Detection vs Face Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.14734](https://arxiv.org/pdf/2604.14734)**

> **作者:** Una M. Kelly; Luuk J. Spreeuwers; Raymond N.J. Veldhuis
>
> **摘要:** Morphing is a challenge to face recognition (FR) for which several morphing attack detection solutions have been proposed. We argue that face recognition and differential morphing attack detection (D-MAD) in principle perform very similar tasks, which we support by comparing an FR system with two existing D-MAD approaches. We also show that currently used decision thresholds inherently lead to FR systems being vulnerable to morphing attacks and that this explains the tradeoff between performance on normal images and vulnerability to morphing attacks. We propose using FR systems that are already in place for morphing detection and introduce a new evaluation threshold that guarantees an upper limit to the vulnerability to morphing attacks - even of unknown types.
>
---
#### [replaced 017] Bridging Supervision Gaps: A Unified Framework for Remote Sensing Change Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.17747](https://arxiv.org/pdf/2601.17747)**

> **作者:** Kaixuan Jiang; Chen Wu; Zhenghui Zhao; Chengxi Han; Haonan Guo; Hongruixuan Chen
>
> **摘要:** Change detection (CD) aims to identify surface changes from multi-temporal remote sensing imagery. In real-world scenarios, Pixel-level change labels are expensive to acquire, and existing models struggle to adapt to scenarios with diverse annotation availability. To tackle this challenge, we propose a unified change detection framework (UniCD), which collaboratively handles supervised, weakly-supervised, and unsupervised tasks through a coupled architecture. UniCD eliminates architectural barriers through a shared encoder and multi-branch collaborative learning mechanism, achieving deep coupling of heterogeneous supervision signals. Specifically, UniCD consists of three supervision-specific branches. In the supervision branch, UniCD introduces the spatial-temporal awareness module (STAM), achieving efficient synergistic fusion of bi-temporal features. In the weakly-supervised branch, we construct change representation regularization (CRR), which steers model convergence from coarse-grained activations toward coherent and separable change modeling. In the unsupervised branch, we propose semantic prior-driven change inference (SPCI), which transforms unsupervised tasks into controlled weakly-supervised path optimization. Experiments on mainstream datasets demonstrate that UniCD achieves optimal performance across three tasks. It exhibits significant accuracy improvements in weakly and unsupervised scenarios, surpassing current state-of-the-art by 12.72% and 12.37% on LEVIR-CD, respectively.
>
---
#### [replaced 018] What's Left Unsaid? Detecting and Correcting Misleading Omissions in Multimodal News Previews
- **分类: cs.CV; cs.SI**

- **链接: [https://arxiv.org/pdf/2601.05563](https://arxiv.org/pdf/2601.05563)**

> **作者:** Fanxiao Li; Jiaying Wu; Tingchao Fu; Dayang Li; Herun Wan; Wei Zhou; Min-Yen Kan
>
> **摘要:** Even when factually correct, social-media news previews (image-headline pairs) can induce interpretation drift: by selectively omitting crucial context, they lead readers to form judgments that diverge from what the full article supports. This covert harm is subtler than explicit misinformation, yet remains underexplored. To address this gap, we develop a multi-stage pipeline that simulates preview-based and context-based understanding, enabling construction of the MM-Misleading benchmark. Using MM-Misleading, we systematically evaluate open-source LVLMs and uncover pronounced blind spots in omission-based misleadingness detection. We further propose OMGuard, which combines (1) Interpretation-Aware Fine-Tuning for misleadingness detection and (2) Rationale-Guided Misleading Content Correction, where explicit rationales guide headline rewriting to reduce misleading impressions. Experiments show that OMGuard lifts an 8B model's detection accuracy to the level of a 235B LVLM while delivering markedly stronger end-to-end correction. Further analysis shows that misleadingness usually arises from local narrative shifts, such as missing background, instead of global frame changes, and identifies image-driven cases where text-only correction fails, underscoring the need for visual interventions.
>
---
#### [replaced 019] PLAF: Pixel-wise Language-Aligned Feature Extraction for Efficient 3D Scene Understanding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景理解任务，旨在解决语义表示在2D与3D间语言对齐及冗余问题。提出PLAF框架，实现像素级语义对齐与高效存储查询。**

- **链接: [https://arxiv.org/pdf/2604.15770](https://arxiv.org/pdf/2604.15770)**

> **作者:** Junjie Wen; Junlin He; Fei Ma; Jinqiang Cui
>
> **备注:** Accepted by ICCA 2026
>
> **摘要:** Accurate open-vocabulary 3D scene understanding requires semantic representations that are both language-aligned and spatially precise at the pixel level, while remaining scalable when lifted to 3D space. However, existing representations struggle to jointly satisfy these requirements, and densely propagating pixel-wise semantics to 3D often results in substantial redundancy, leading to inefficient storage and querying in large-scale scenes. To address these challenges, we present \emph{PLAF}, a Pixel-wise Language-Aligned Feature extraction framework that enables dense and accurate semantic alignment in 2D without sacrificing open-vocabulary expressiveness. Building upon this representation, we further design an efficient semantic storage and querying scheme that significantly reduces redundancy across both 2D and 3D domains. Experimental results show that \emph{PLAF} provides a strong semantic foundation for accurate and efficient open-vocabulary 3D scene understanding. The codes are publicly available at this https URL.
>
---
#### [replaced 020] Fake or Real, Can Robots Tell? Evaluating VLM Robustness to Domain Shift in Single-View Robotic Scene Understanding
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文研究机器人场景理解任务，解决VLM在领域迁移下的鲁棒性问题。通过对比真实与3D打印物体，评估模型性能及评价指标的可靠性。**

- **链接: [https://arxiv.org/pdf/2506.19579](https://arxiv.org/pdf/2506.19579)**

> **作者:** Federico Tavella; Amber Drinkwater; Angelo Cangelosi
>
> **摘要:** Robotic scene understanding increasingly relies on Vision-Language Models (VLMs) to generate natural language descriptions of the environment. In this work, we systematically evaluate single-view object captioning for tabletop scenes captured by a robotic manipulator, introducing a controlled physical domain shift that contrasts real-world tools with geometrically similar 3D-printed counterparts that differ in texture, colour, and material. We benchmark a suite of state-of-the-art, locally deployable VLMs across multiple metrics to assess semantic alignment and factual grounding. Our results demonstrate that while VLMs describe common real-world objects effectively, performance degrades markedly on 3D-printed items despite their structurally familiar forms. We further expose critical vulnerabilities in standard evaluation metrics, showing that some fail to detect domain shifts entirely or reward fluent but factually incorrect captions. These findings highlight the limitations of deploying foundation models for embodied agents and the need for more robust architectures and evaluation protocols in physical robotic applications.
>
---
#### [replaced 021] Semantic-Fast-SAM: Efficient Semantic Segmenter
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.20169](https://arxiv.org/pdf/2604.20169)**

> **作者:** Byunghyun Kim
>
> **备注:** APSIPA ASC 2025
>
> **摘要:** We propose Semantic-Fast-SAM (SFS), a semantic segmentation framework that combines the Fast Segment Anything model with a semantic labeling pipeline to achieve real-time performance without sacrificing accuracy. FastSAM is an efficient CNN-based re-implementation of the Segment Anything Model (SAM) that runs much faster than the original transformer-based SAM. Building upon FastSAM's rapid mask generation, we integrate a Semantic-Segment-Anything (SSA) labeling strategy to assign meaningful categories to each mask. The resulting SFS model produces high-quality semantic segmentation maps at a fraction of the computational cost and memory footprint of the original SAM-based approach. Experiments on Cityscapes and ADE20K benchmarks demonstrate that SFS matches the accuracy of prior SAM-based methods (mIoU ~ 70.33 on Cityscapes and 48.01 on ADE20K) while achieving approximately 20x faster inference than SSA in the closed-set setting. We also show that SFS effectively handles open-vocabulary segmentation by leveraging CLIP-based semantic heads, outperforming recent open-vocabulary models on broad class labeling. This work enables practical real-time semantic segmentation with the "segment-anything" capability, broadening the applicability of foundation segmentation models in robotics scenarios. The implementation is available at this https URL.
>
---
#### [replaced 022] A Lightweight Transformer for Pain Recognition from Brain Activity
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.16491](https://arxiv.org/pdf/2604.16491)**

> **作者:** Stefanos Gkikas; Christian Arzate Cruz; Yu Fang; Lu Cao; Muhammad Umar Khan; Thomas Kassiotis; Giorgos Giannakakis; Raul Fernandez Rojas; Randy Gomez
>
> **摘要:** Pain is a multifaceted and widespread phenomenon with substantial clinical and societal burden, making reliable automated assessment a critical objective. This paper presents a lightweight transformer architecture that fuses multiple fNIRS representations through a unified tokenization mechanism, enabling joint modeling of complementary signal views without requiring modality-specific adaptations or increasing architectural complexity. The proposed token-mixing strategy preserves spatial, temporal, and time-frequency characteristics by projecting heterogeneous inputs onto a shared latent representation, using a structured segmentation scheme to control the granularity of local aggregation and global interaction. The model is evaluated on the AI4Pain dataset using stacked raw waveform and power spectral density representations of fNIRS inputs. Experimental results demonstrate competitive pain recognition performance while remaining computationally compact, making the approach suitable for real-time inference on both GPU and CPU hardware.
>
---
#### [replaced 023] Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.12845](https://arxiv.org/pdf/2603.12845)**

> **作者:** Fei Wang; Xinye Zheng; Kun Li; Yanyan Wei; Yuxin Liu; Ganpeng Hu; Tong Bao; Jingwen Yang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Predicting enzyme kinetic parameters quantifies how efficiently an enzyme catalyzes a specific substrate under defined biochemical conditions. Canonical parameters such as the turnover number ($k_\text{cat}$), Michaelis constant ($K_\text{m}$), and inhibition constant ($K_\text{i}$) depend jointly on the enzyme sequence, the substrate chemistry, and the conformational adaptation of the active site during binding. Many learning pipelines simplify this process to a static compatibility problem between the enzyme and substrate, fusing their representations through shallow operations and regressing a single value. Such formulations overlook the staged nature of catalysis, which involves both substrate recognition and conformational adaptation. In this regard, we reformulate kinetic prediction as a staged multimodal conditional modeling problem and introduce the Enzyme-Reaction Bridging Adapter (ERBA), which injects cross-modal information via fine-tuning into Protein Language Models (PLMs) while preserving their biochemical priors. ERBA performs conditioning in two stages: Molecular Recognition Cross-Attention (MRCA) first injects substrate information into the enzyme representation to capture specificity; Geometry-aware Mixture-of-Experts (G-MoE) then integrates active-site structure and routes samples to pocket-specialized experts to reflect induced fit. To maintain semantic fidelity, Enzyme-Substrate Distribution Alignment (ESDA) enforces distributional consistency within the PLM manifold in a reproducing kernel Hilbert space. Experiments across three kinetic endpoints and multiple PLM backbones, ERBA delivers consistent gains and stronger out-of-distribution performance compared with sequence-only and shallow-fusion baselines, offering a biologically grounded route to scalable kinetic prediction and a foundation for adding cofactors, mutations, and time-resolved structural cues.
>
---
#### [replaced 024] DAVIS: OOD Detection via Dominant Activations and Variance for Increased Separation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.22703](https://arxiv.org/pdf/2601.22703)**

> **作者:** Abid Hassan; Tuan Ngo; Saad Shafiq; Nenad Medvidovic
>
> **摘要:** Detecting out-of-distribution (OOD) inputs is a critical safeguard for deploying machine learning models in the real world. However, most post-hoc detection methods operate on penultimate feature representations derived from global average pooling (GAP) -- a lossy operation that discards valuable distributional statistics from activation maps prior to global average pooling. We contend that these overlooked statistics, particularly channel-wise variance and dominant (maximum) activations, are highly discriminative for OOD detection. We introduce DAVIS, a simple and broadly applicable post-hoc technique that enriches feature vectors by incorporating these crucial statistics, directly addressing the information loss from GAP. Extensive evaluations show DAVIS sets a new benchmark across diverse architectures, including ResNet, DenseNet, and EfficientNet. It achieves significant reductions in the false positive rate (FPR95), with improvements of 48.26\% on CIFAR-10 using ResNet-18, 38.13\% on CIFAR-100 using ResNet-34, and 26.83\% on ImageNet-1k benchmarks using MobileNet-v2. Our analysis reveals the underlying mechanism for this improvement, providing a principled basis for moving beyond the mean in OOD detection.
>
---
#### [replaced 025] Automated Annotation of Shearographic Measurements Enabling Weakly Supervised Defect Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.06171](https://arxiv.org/pdf/2512.06171)**

> **作者:** Jessica Plassmann; Nicolas Schuler; Michael Schuth; Georg von Freymann
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Shearography is an interferometric technique sensitive to surface displacement gradients, providing high sensitivity for detecting subsurface defects in safety-critical components. A key limitation to industrial adoption is the lack of high-quality annotated datasets, since manual labeling remains labor-intensive, subjective, and difficult to standardize. We present an automated labeling pipeline that generates candidate defect bounding boxes with Grounded DINO, refines them using SAM masks, and exports YOLO-format labels for downstream detector training. Quantitative evaluation shows the generated boxes are suitable for weakly supervised learning, while high-resolution masks provide qualitative visualization. This approach reduces manual effort and supports scalable dataset creation for robust industrial defect detection.
>
---
#### [replaced 026] PAT3D: Physics-Augmented Text-to-3D Scene Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21978](https://arxiv.org/pdf/2511.21978)**

> **作者:** Guying Lin; Kemeng Huang; Michael Liu; Ruihan Gao; Hanke Chen; Lyuhao Chen; Beijia Lu; Taku Komura; Yuan Liu; Jun-Yan Zhu; Minchen Li
>
> **备注:** 19 pages, 12 figures
>
> **摘要:** We introduce PAT3D, the first physics-augmented text-to-3D scene generation framework that integrates vision-language models with physics-based simulation to produce physically plausible, simulation-ready, and intersection-free 3D scenes. Given a text prompt, PAT3D generates 3D objects, infers their spatial relations, and organizes them into a hierarchical scene tree, which is then converted into initial conditions for simulation. A differentiable rigid-body simulator ensures realistic object interactions under gravity, driving the scene toward static equilibrium without interpenetrations. To further enhance scene quality, we introduce a simulation-in-the-loop optimization procedure that guarantees physical stability and non-intersection, while improving semantic consistency with the input prompt. Experiments demonstrate that PAT3D substantially outperforms prior approaches in physical plausibility, semantic consistency, and visual quality. Beyond high-quality generation, PAT3D uniquely enables simulation-ready 3D scenes for downstream tasks such as scene editing and robotic manipulation. Code and data are available at: this https URL.
>
---
#### [replaced 027] VFM-VAE: Vision Foundation Models Can Be Good Tokenizers for Latent Diffusion Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.18457](https://arxiv.org/pdf/2510.18457)**

> **作者:** Tianci Bi; Xiaoyi Zhang; Yan Lu; Nanning Zheng
>
> **备注:** Accepted at CVPR 2026. Code and models available at: this https URL
>
> **摘要:** The performance of Latent Diffusion Models (LDMs) is critically dependent on the quality of their visual tokenizers. While recent works have explored incorporating Vision Foundation Models (VFMs) into the tokenizers training via distillation, we empirically find this approach inevitably weakens the robustness of learnt representation from original VFM. In this paper, we bypass the distillation by proposing a more direct approach by leveraging the frozen VFM for the LDMs tokenizer, named VFM Variational Autoencoder (VFM-VAE).To fully exploit the potential to leverage frozen VFM for the LDMs tokenizer, we design a new decoder to reconstruct realistic images from the semantic-rich representation of VFM. With the proposed VFM-VAE, we conduct a systematic study on how the representation from different tokenizers impact the representation learning process throughout diffusion training, enabling synergistic benefits of dual-side alignment on both tokenizers and diffusion models. Our effort in tokenizer design and training strategy lead to superior performance and efficiency: our system reaches a gFID (w/o CFG) of 2.22 in merely 80 epochs (a 10$\times$ speedup over prior tokenizers). With continued training to 640 epochs, it further attains a gFID (w/o CFG) of 1.62. These results offer solid evidence for the substantial potential of VFMs to serve as visual tokenizers to accelerate the LDM training progress.
>
---
#### [replaced 028] SatSAM2: Motion-Constrained Video Object Tracking in Satellite Imagery using Promptable SAM2 and Kalman Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18264](https://arxiv.org/pdf/2511.18264)**

> **作者:** Ruijie Fan; Junyan Ye; Huan Chen; Zilong Huang; Xiaolei Wang; Weijia Li
>
> **备注:** 14 pages, 12 figures
>
> **摘要:** Existing satellite video tracking methods often struggle with generalization, requiring scenario-specific training to achieve satisfactory performance, and are prone to track loss in the presence of occlusion. To address these challenges, we propose SatSAM2, a zero-shot satellite video tracker built on SAM2, designed to adapt foundation models to the remote sensing domain. SatSAM2 introduces two core modules: a Kalman Filter-based Constrained Motion Module (KFCMM) to exploit temporal motion cues and suppress drift, and a Motion-Constrained State Machine (MCSM) to regulate tracking states based on motion dynamics and reliability. To support large-scale evaluation, we propose MatrixCity Video Object Tracking (MVOT), a synthetic benchmark containing 1,500+ sequences and 157K annotated frames with diverse viewpoints, illumination, and occlusion conditions. Extensive experiments on two satellite tracking benchmarks and MVOT show that SatSAM2 outperforms both traditional and foundation model-based trackers, including SAM2 and its variants. Notably, on the OOTB dataset, SatSAM2 achieves a 5.84% AUC improvement over state-of-the-art methods. Our code and dataset will be publicly released to encourage further research.
>
---
#### [replaced 029] FunduSegmenter: Leveraging the RETFound Foundation Model for Joint Optic Disc and Optic Cup Segmentation in Retinal Fundus Images
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.11354](https://arxiv.org/pdf/2508.11354)**

> **作者:** Zhenyi Zhao; Muthu Rama Krishnan Mookiah; Emanuele Trucco
>
> **摘要:** Purpose: This study introduces the first adaptation of RETFound for joint optic disc (OD) and optic cup (OC) segmentation. RETFound is a well-known foundation model developed for fundus camera and optical coherence tomography images, which has shown promising performance in disease diagnosis. Methods: We propose FunduSegmenter, a model integrating a series of novel modules with RETFound, including a Pre-adapter, a Decoder, a Post-adapter, skip connections with Convolutional Block Attention Module and a Vision Transformer block adapter. The model is evaluated on a proprietary dataset, GoDARTS, and four public datasets, IDRiD, Drishti-GS, RIM-ONE-r3, and REFUGE, through internal verification, external verification and domain generalization experiments. Results: An average Dice similarity coefficient of 90.51% was achieved in internal verification, which outperformed all baselines, some substantially (nnU-Net: 82.91%; DUNet: 89.17%; TransUNet: 87.91%). In all external verification experiments, the average results were about 3% higher than those of the best baseline, and our model was also competitive in domain generalization. Conclusions: This study explored the potential of the latent general representations learned by RETFound for OD and OC segmentation in fundus camera images. Our FunduSegmenter generally outperformed state-of-the-art baseline methods. The proposed modules are general and can be extended to fine-tuning other foundation models. Translational Relevance: The model shows strong stability and generalization on both in-distribution and out-of-distribution data, providing stable OD and OC segmentation. This is an essential step for many automated tasks, from setting the accurate retinal coordinate to biomarker discovery. The code and trained weights are available at: this https URL.
>
---
#### [replaced 030] VidHal: Benchmarking Temporal Hallucinations in Vision LLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.16771](https://arxiv.org/pdf/2411.16771)**

> **作者:** Wey Yeh Choong; Yangyang Guo; Mohan Kankanhalli
>
> **备注:** To appear in TMLR 2026. Code available at this https URL
>
> **摘要:** Vision Large Language Models (VLLMs) are widely acknowledged to be prone to hallucinations. Existing research addressing this problem has primarily been confined to image inputs, with limited exploration of video-based hallucinations. Furthermore, current evaluation methods fail to capture nuanced errors in generated responses, which are often exacerbated by the rich spatiotemporal dynamics of videos. To address this, we introduce VidHal, a benchmark specially designed to evaluate video-based hallucinations in VLLMs. VidHal is constructed by bootstrapping video instances across a wide range of common temporal aspects. A defining feature of our benchmark lies in the careful creation of captions which represent varying levels of hallucination associated with each video. To enable fine-grained evaluation, we propose a novel caption ordering task requiring VLLMs to rank captions by hallucinatory extent. We conduct extensive experiments on VidHal and comprehensively evaluate a broad selection of models. Our results uncover significant limitations in existing VLLMs regarding hallucination generation. Through our benchmark, we aim to inspire further research on 1) holistic understanding of VLLM capabilities, particularly regarding hallucination, and 2) extensive development of advanced VLLMs to alleviate this problem.
>
---
#### [replaced 031] Accelerating Vision Transformers with Adaptive Patch Sizes
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.18091](https://arxiv.org/pdf/2510.18091)**

> **作者:** Rohan Choudhury; JungEun Kim; Jinhyung Park; Eunho Yang; László A. Jeni; Kris M. Kitani
>
> **备注:** Accepted to ICLR 2026. Project page at this https URL
>
> **摘要:** Vision Transformers (ViTs) partition input images into uniformly sized patches regardless of their content, resulting in long input sequence lengths for high-resolution images. We present Adaptive Patch Transformers (APT), which addresses this by using multiple different patch sizes within the same image. APT reduces the total number of input tokens by allocating larger patch sizes in more homogeneous areas and smaller patches in more complex ones. APT achieves a drastic speedup in ViT inference and training, increasing throughput by 40% on ViT-L and 50% on ViT-H while maintaining downstream performance, and can be applied to a previously fine-tuned ViT, converging in as little as 1 epoch. It also significantly reduces training and inference time without loss of performance in high-resolution dense visual tasks, achieving up to 30\% faster training and inference in visual QA, object detection, and semantic segmentation.
>
---
#### [replaced 032] RailVQA: A Benchmark and Framework for Efficient Interpretable Visual Cognition in Automatic Train Operation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27112](https://arxiv.org/pdf/2603.27112)**

> **作者:** Sen Zhang; Runmei Li; Shizhuang Deng; Zhichao Zheng; Yuhe Zhang; Jiani Li; Kailun Zhang; Tao Zhang; Wenjun Wu; Qunbo Wang
>
> **摘要:** As Automatic Train Operation (ATO) advances toward GoA4 and beyond, it increasingly depends on efficient, reliable cab-view visual perception and decision-oriented inference to ensure safe operation in complex and dynamic railway environments. However, existing approaches focus primarily on basic perception and often generalize poorly to rare yet safety-critical corner cases. They also lack the high-level reasoning and planning capabilities required for operational decision-making. Although recent Large Multi-modal Models (LMMs) show strong generalization and cognitive capabilities, their use in safety-critical ATO is hindered by high computational cost and hallucination risk. Meanwhile, reliable domain-specific benchmarks for systematically evaluating cognitive capabilities are still lacking. To address these gaps, we introduce RailVQA-bench, the first VQA benchmark for cab-view visual cognition in ATO, comprising 20,000 single-frame and 1,168 video based QA pairs to evaluate cognitive generalization and interpretability in both static and dynamic scenarios. Furthermore, we propose RailVQA-CoM, a collaborative large-small model framework that combines small-model efficiency with large-model cognition via a transparent three-module architecture and adaptive temporal sampling, improving perceptual generalization and enabling more efficient reasoning and planning. Experiments demonstrate that the proposed approach substantially improves performance, enhances interpretability, improves efficiency, and strengthens cross-domain generalization in autonomous driving systems. Code and datasets will be available at this https URL.
>
---
#### [replaced 033] PC2Model: ISPRS benchmark on 3D point cloud to model registration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.19596](https://arxiv.org/pdf/2604.19596)**

> **作者:** Mehdi Maboudi; Said Harb; Jackson Ferrao; Kourosh Khoshelham; Yelda Turkan; Karam Mawas
>
> **备注:** ISPRS Congress 2026, Toronto
>
> **摘要:** Point cloud registration involves aligning one point cloud with another or with a three-dimensional (3D) model, enabling the integration of multimodal data into a unified representation. This is essential in applications such as construction monitoring, autonomous driving, robotics, and virtual or augmented reality (VR/AR). With the increasing accessibility of point cloud acquisition technologies, such as Light Detection and Ranging (LiDAR) and structured light scanning, along with recent advances in deep learning, the research focus has increasingly shifted towards downstream tasks, particularly point cloud-to-model (PC2Model) registration. While data-driven methods aim to automate this process, they struggle with sparsity, noise, clutter, and occlusions in real-world scans, which limit their performance. To address these challenges, this paper introduces the PC2Model benchmark, a publicly available dataset designed to support the training and evaluation of both classical and data-driven methods. Developed under the leadership of ICWG II/Ib, the PC2Model benchmark adopts a hybrid design that combines simulated point clouds with, in some cases, real-world scans and their corresponding 3D models. Simulated data provide precise ground truth and controlled conditions, while real-world data introduce sensor and environmental artefacts. This design supports robust training and evaluation across domains and enables the systematic analysis of model transferability from simulated to real-world scenarios. The dataset is publicly accessible at: \href{this https URL}{this https URL}
>
---
#### [replaced 034] PercHead: Perceptual Head Model for Single-Image 3D Head Reconstruction & Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.02777](https://arxiv.org/pdf/2511.02777)**

> **作者:** Antonio Oroz; Matthias Nießner; Tobias Kirschstein
>
> **备注:** Project Page: this https URL Video: this https URL
>
> **摘要:** We present PercHead, a model for single-image 3D head reconstruction and disentangled 3D editing - two tasks that are inherently challenging due to ambiguity in plausible explanations for the same input. At the heart of our approach lies our novel perceptual loss based on DINOv2 and SAM 2.1. Unlike widely-adopted low-level losses like LPIPS, SSIM or L1, we rely on deep visual understanding of images and the resulting generalized supervision signals. We show that our new loss can be a drop-in replacement for standard losses and used to improve visual quality in high-frequency areas. We base our model architecture on Vision Transformers (ViTs), allowing us to decouple the 3D representation from the 2D input. We train our method on multi-view images for view-consistency and in-the-wild images for strong transferability to new environments. Our model achieves state-of-the-art performance in novel-view synthesis and, furthermore, exhibits exceptional robustness to extreme viewing angles. We also extend our base model to disentangled 3D editing by swapping the encoder and fine-tuning the network. A segmentation map controls geometry and either a text prompt or a reference image specifies appearance. We highlight the intuitive and powerful 3D editing capabilities through an interactive GUI. Project Page: this https URL Video: this https URL
>
---
#### [replaced 035] Fusion Complexity Inversion: Why Simpler Cross View Modules Outperform SSMs and Cross View Attention Transformers for Pasture Biomass Regression
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.07819](https://arxiv.org/pdf/2603.07819)**

> **作者:** Mridankan Mandal
>
> **备注:** Accepted to CVPR: Vision for Agriculture Workshop 2026
>
> **摘要:** Accurate estimation of pasture biomass from agricultural imagery is critical for sustainable livestock management, yet existing methods are limited by the small, imbalanced, and sparsely annotated datasets typical of real world monitoring. In this study, adaptation of vision foundation models to agricultural regression is systematically evaluated on the CSIRO Pasture Biomass benchmark, a 357 image dual view dataset with laboratory validated, component wise ground truth for five biomass targets, through 17 configurations spanning four backbones (EfficientNet-B3 to DINOv3-ViT-L), five cross view fusion mechanisms, and a 4x2 metadata factorial. A counterintuitive principle, termed "fusion complexity inversion", is uncovered: on scarce agricultural data, a two layer gated depthwise convolution (R^2 = 0.903) outperforms cross view attention transformers (0.833), bidirectional SSMs (0.819), and full Mamba (0.793, below the no fusion baseline). Backbone pretraining scale is found to monotonically dominate all architectural choices, with the DINOv2 -> DINOv3 upgrade alone yielding +5.0 R^2 points. Training only metadata (species, state, and NDVI) is shown to create a universal ceiling at R^2 ~ 0.829, collapsing an 8.4 point fusion spread to 0.1 points. Actionable guidelines for sparse agricultural benchmarks are established: backbone quality should be prioritized over fusion complexity, local modules preferred over global alternatives, and features unavailable at inference excluded.
>
---
#### [replaced 036] StreamMeCo: Long-Term Agent Memory Compression for Efficient Streaming Video Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.09000](https://arxiv.org/pdf/2604.09000)**

> **作者:** Junxi Wang; Te Sun; Jiayi Zhu; Junxian Li; Haowen Xu; Zichen Wen; Xuming Hu; Zhiyu Li; Linfeng Zhang
>
> **备注:** 2026ACL Findings
>
> **摘要:** Vision agent memory has shown remarkable effectiveness in streaming video understanding. However, storing such memory for videos incurs substantial memory overhead, leading to high costs in both storage and computation. To address this issue, we propose StreamMeCo, an efficient Stream Agent Memory Compression framework. Specifically, based on the connectivity of the memory graph, StreamMeCo introduces edge-free minmax sampling for the isolated nodes and an edge-aware weight pruning for connected nodes, evicting the redundant memory nodes while maintaining the accuracy. In addition, we introduce a time-decay memory retrieval mechanism to further eliminate the performance degradation caused by memory compression. Extensive experiments on three challenging benchmark datasets (M3-Bench-robot, M3-Bench-web and Video-MME-Long) demonstrate that under 70% memory graph compression, StreamMeCo achieves a 1.87* speedup in memory retrieval while delivering an average accuracy improvement of 1.0%. Our code is available at this https URL.
>
---
#### [replaced 037] Catalyst: Out-of-Distribution Detection via Elastic Scaling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02409](https://arxiv.org/pdf/2602.02409)**

> **作者:** Abid Hassan; Tuan Ngo; Saad Shafiq; Nenad Medvidovic
>
> **备注:** Accepted at Conference on Computer Vision and Pattern Recognition (CVPR) 2026. arXiv admin note: text overlap with arXiv:2601.22703
>
> **摘要:** Out-of-distribution (OOD) detection is critical for the safe deployment of deep neural networks. State-of-the-art post-hoc methods typically derive OOD scores from the output logits or penultimate feature vector obtained via global average pooling (GAP). We contend that this exclusive reliance on the logit or feature vector discards a rich, complementary signal: the raw channel-wise statistics of the pre-pooling feature map lost in GAP. In this paper, we introduce Catalyst, a post-hoc framework that exploits these under-explored signals. Catalyst computes an input-dependent scaling factor ($\gamma$) on-the-fly from these raw statistics (e.g., mean, standard deviation, and maximum activation). This $\gamma$ is then fused with the existing baseline score, multiplicatively modulating it -- an $\textit{elastic scaling}$ -- to push the ID and OOD distributions further apart. We demonstrate Catalyst is a generalizable framework: it seamlessly integrates with logit-based methods (e.g., Energy, ReAct, SCALE) and also provides a significant boost to distance-based detectors like KNN. As a result, Catalyst achieves substantial and consistent performance gains, reducing the average False Positive Rate by 32.87 on CIFAR-10 (ResNet-18), 27.94% on CIFAR-100 (ResNet-18), and 22.25% on ImageNet (ResNet-50). Our results highlight the untapped potential of pre-pooling statistics and demonstrate that Catalyst is complementary to existing OOD detection approaches. Our code is available here: this https URL
>
---
#### [replaced 038] E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3D Gaussian Splatting Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.17969](https://arxiv.org/pdf/2604.17969)**

> **作者:** Koya Sakamoto; Taiki Miyanishi; Daichi Azuma; Shuhei Kurita; Shu Morikuni; Naoya Chiba; Motoaki Kawanabe; Yusuke Iwasawa; Yutaka Matsuo
>
> **备注:** Project page: this https URL
>
> **摘要:** Visual search in 3D environments requires embodied agents to actively explore their surroundings and acquire task-relevant evidence. However, existing visual search and embodied AI benchmarks, including EQA, typically rely on static observations or constrained egocentric motion, and thus do not explicitly evaluate fine-grained viewpoint-dependent phenomena that arise under unrestricted 5-DoF viewpoint control in real-world 3D environments, such as visibility changes caused by vertical viewpoint shifts, revealing contents inside containers, and disambiguating object attributes that are only observable from specific angles. To address this limitation, we introduce {E3VS-Bench}, a benchmark for embodied 3D visual search where agents must control their viewpoints in 5-DoF to gather viewpoint-dependent evidence for question answering. E3VS-Bench consists of 99 high-fidelity 3D scenes reconstructed using 3D Gaussian Splatting and 2,014 question-driven episodes. 3D Gaussian Splatting enables photorealistic free-viewpoint rendering that preserves fine-grained visual details (e.g., small text and subtle attributes) often degraded in mesh-based simulators, thereby allowing the construction of questions that cannot be answered from a single view and instead require active inspection across viewpoints in 5-DoF. We evaluate multiple state-of-the-art VLMs and compare their performance with humans. Despite strong 2D reasoning ability, all models exhibit a substantial gap from humans, highlighting limitations in active perception and coherent viewpoint planning specifically under full 5-DoF viewpoint changes.
>
---
#### [replaced 039] BiTDiff: Fine-Grained 3D Conducting Motion Generation via BiMamba-Transformer Diffusion
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2604.04395](https://arxiv.org/pdf/2604.04395)**

> **作者:** Tianzhi Jia; Kaixing Yang; Xiaole Yang; Xulong Tang; Ke Qiu; Shikui Wei; Yao Zhao
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** 3D conducting motion generation aims to synthesize fine-grained conductor motions from music, with broad potential in music education, virtual performance, digital human animation, and human-AI co-creation. However, this task remains underexplored due to two major challenges: (1) the lack of large-scale fine-grained 3D conducting datasets and (2) the absence of effective methods that can jointly support long-sequence generation with high quality and efficiency. To address the data limitation, we develop a quality-oriented 3D conducting motion collection pipeline and construct CM-Data, a fine-grained SMPL-X dataset with about 10 hours of conducting motion data. To the best of our knowledge, CM-Data is the first and largest public dataset for 3D conducting motion generation. To address the methodological limitation, we propose BiTDiff, a novel framework for 3D conducting motion generation, built upon a BiMamba-Transformer hybrid model architecture for efficient long-sequence modeling and a Diffusion-based generative strategy with human-kinematic decomposition for high-quality motion synthesis. Specifically, BiTDiff introduces auxiliary physical-consistency losses and a hand-/body-specific forward-kinematics design for better fine-grained motion modeling, while leveraging BiMamba for memory-efficient long-sequence temporal modeling and Transformer for cross-modal semantic alignment. In addition, BiTDiff supports training-free joint-level motion editing, enabling downstream human-AI interaction design. Extensive quantitative and qualitative experiments demonstrate that BiTDiff achieves state-of-the-art (SOTA) performance for 3D conducting motion generation on the CM-Data dataset. Code will be available upon acceptance.
>
---
#### [replaced 040] Video-Robin: Autoregressive Diffusion Planning for Intent-Grounded Video-to-Music Generation
- **分类: cs.SD; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于视频到音乐生成任务，旨在解决现有模型在语义控制和音频质量上的不足。提出Video-Robin，结合自回归规划与扩散合成，提升生成音乐的语义对齐和质量。**

- **链接: [https://arxiv.org/pdf/2604.17656](https://arxiv.org/pdf/2604.17656)**

> **作者:** Vaibhavi Lokegaonkar; Aryan Vijay Bhosale; Vishnu Raj; Gouthaman KV; Ramani Duraiswami; Lie Lu; Sreyan Ghosh; Dinesh Manocha
>
> **摘要:** Video-to-music (V2M) is the fundamental task of creating background music for an input video. Recent V2M models achieve audiovisual alignment by typically relying on visual conditioning alone and provide limited semantic and stylistic controllability to the end user. In this paper, we present Video-Robin, a novel text-conditioned video-to-music generation model that enables fast, high-quality, semantically aligned music generation for video content. To balance musical fidelity and semantic understanding, Video-Robin integrates autoregressive planning with diffusion-based synthesis. Specifically, an autoregressive module models global structure by semantically aligning visual and textual inputs to produce high-level music latents. These latents are subsequently refined into coherent, high-fidelity music using local Diffusion Transformers. By factoring semantically driven planning into diffusion-based synthesis, Video-Robin enables fine-grained creator control without sacrificing audio realism. Our proposed model outperforms baselines that solely accept video input and additional feature conditioned baselines on both in-distribution and out-of-distribution benchmarks with a 2.21x speed in inference compared to SOTA. We will open-source everything upon paper acceptance.
>
---
#### [replaced 041] CrackForward: Context-Aware Severity Stage Crack Synthesis for Data Augmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.19941](https://arxiv.org/pdf/2604.19941)**

> **作者:** Nassim Sadallah; Mohand Saïd Allili
>
> **备注:** 6
>
> **摘要:** Reliable crack detection and segmentation are vital for structural health monitoring, yet the scarcity of well-annotated data constitutes a major challenge. To address this limitation, we propose a novel context-aware generative framework designed to synthesize realistic crack growth patterns for data augmentation. Unlike existing methods that primarily manipulate textures or background content, CrackForward explicitly models crack morphology by combining directional crack elongation with learned thickening and branching. Our framework integrates two key innovations: (i) a contextually guided crack expansion module, which uses local directional cues and adaptive random walk to simulate realistic propagation paths; and (ii) a two-stage U-Net-style generator that learns to reproduce spatially varying crack characteristics such as thickness, branching, and growth. Experimental results show that the generated samples preserve target-stage saturation and thickness characteristics and improve the performance of several crack segmentation architectures. These results indicate that structure-aware synthetic crack generation can provide more informative training data than conventional augmentation alone.
>
---
#### [replaced 042] Flow Matching for Conditional MRI-CT and CBCT-CT Image Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04823](https://arxiv.org/pdf/2510.04823)**

> **作者:** Arnela Hadzic; Simon Johannes Joham; Martin Urschler
>
> **备注:** Published in the Proceedings of the Third Austrian Symposium on AI, Robotics, and Vision (AIRoV 2026)
>
> **摘要:** Generating synthetic CT (sCT) from MRI or CBCT plays a crucial role in enabling MRI-only and CBCT-based adaptive radiotherapy, improving treatment precision while reducing patient radiation exposure. To address this task, we adopt a fully 3D Flow Matching (FM) framework, motivated by recent work demonstrating FM's efficiency in producing high-quality images. In our approach, a Gaussian noise volume is transformed into an sCT image by integrating a learned FM velocity field, conditioned on features extracted from the input MRI or CBCT using a lightweight 3D encoder. We evaluated the method on the SynthRAD2025 Challenge benchmark, training separate models for MRI to sCT and CBCT to sCT across three anatomical regions: abdomen, head and neck, and thorax. Validation and testing were performed through the challenge submission system. The results indicate that the method accurately reconstructs global anatomical structures; however, preservation of fine details was limited, primarily due to the relatively low training resolution imposed by memory and runtime constraints. Future work will explore patch-based training and latent-space flow models to improve resolution and local structural fidelity.
>
---
#### [replaced 043] Beyond the Frame: Generating 360 Panoramic Videos from Perspective Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.07940](https://arxiv.org/pdf/2504.07940)**

> **作者:** Rundong Luo; Matthew Wallingford; Ali Farhadi; Noah Snavely; Wei-Chiu Ma
>
> **备注:** Project page: this https URL
>
> **摘要:** 360° videos have emerged as a promising medium to represent our dynamic visual world. Compared to the "tunnel vision" of standard cameras, their borderless field of view offers a more complete perspective of our surroundings. While existing video models excel at producing standard videos, their ability to generate full panoramic videos remains elusive. In this paper, we investigate the task of video-to-360° generation: given a perspective video as input, our goal is to generate a full panoramic video that is consistent with the original video. Unlike conventional video generation tasks, the output's field of view is significantly larger, and the model is required to have a deep understanding of both the spatial layout of the scene and the dynamics of objects to maintain spatio-temporal consistency. To address these challenges, we first leverage the abundant 360° videos available online and develop a high-quality data filtering pipeline to curate pairwise training data. We then carefully design a series of geometry- and motion-aware operations to facilitate the learning process and improve the quality of 360° video generation. Experimental results demonstrate that our model can generate realistic and coherent 360° videos from in-the-wild perspective video. In addition, we showcase its potential applications, including video stabilization, camera viewpoint control, and interactive visual question answering.
>
---
#### [replaced 044] Transformer-Progressive Mamba Network for Lightweight Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.03232](https://arxiv.org/pdf/2511.03232)**

> **作者:** Sichen Guo; Wenjie Li; Yuanyang Liu; Guangwei Gao; Jian Yang; Chia-Wen Lin
>
> **备注:** 14 pages, 12 figures, 9 tables
>
> **摘要:** Recently, Mamba-based super-resolution (SR) methods have demonstrated the ability to capture global receptive fields with linear complexity, addressing the quadratic computational cost of Transformer-based SR approaches. However, existing Mamba-based methods lack fine-grained transitions across different modeling scales, which limits the efficiency of feature representation. In this paper, we propose T-PMambaSR, a lightweight SR framework that integrates window-based self-attention with Progressive Mamba. By enabling interactions among receptive fields of different scales, our method establishes a fine-grained modeling paradigm that progressively enhances feature representation without introducing additional computational cost. Furthermore, we introduce an Adaptive High-Frequency Refinement Module (AHFRM) to recover high-frequency details lost during Transformer and Mamba processing. Extensive experiments demonstrate that T-PMambaSR progressively enhances the model's receptive field and expressiveness, yielding better performance than recent Transformer- or Mamba-based methods while incurring lower computational cost.
>
---
#### [replaced 045] Geo-R1: Improving Few-Shot Geospatial Referring Expression Understanding with Reinforcement Fine-Tuning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.21976](https://arxiv.org/pdf/2509.21976)**

> **作者:** Zilun Zhang; Zian Guan; Tiancheng Zhao; Haozhan Shen; Tianyu Li; Yuxiang Cai; Zhonggen Su; Zhaojun Liu; Jianwei Yin; Xiang Li
>
> **备注:** Accepted by ISPRS
>
> **摘要:** Referring expression understanding in remote sensing poses unique challenges, as it requires reasoning over complex object-context relationships. While supervised fine-tuning (SFT) on multimodal large language models achieves strong performance with massive labeled datasets, they struggle in data-scarce scenarios, leading to poor generalization. To address this limitation, we propose Geo-R1, a reasoning-centric reinforcement fine-tuning (RFT) paradigm for few-shot geospatial referring. Geo-R1 enforces the model to first generate explicit, interpretable reasoning chains that decompose referring expressions, and then leverage these rationales to localize target objects. This "reason first, then act" process enables the model to make more effective use of limited annotations, enhances generalization, and provides interpretability. We validate Geo-R1 on three carefully designed few-shot geospatial referring benchmarks, where our model consistently and substantially outperforms SFT baselines. It also demonstrates strong cross-dataset generalization, highlighting its robustness. Code and data will be released at: this https URL.
>
---
#### [replaced 046] AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security
- **分类: cs.AI; cs.CC; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于AI安全任务，旨在解决AI代理的复杂安全与风险问题。提出AgentDoG框架，实现细粒度风险诊断与透明监控。**

- **链接: [https://arxiv.org/pdf/2601.18491](https://arxiv.org/pdf/2601.18491)**

> **作者:** Dongrui Liu; Qihan Ren; Chen Qian; Shuai Shao; Yuejin Xie; Yu Li; Zhonghao Yang; Haoyu Luo; Peng Wang; Qingyu Liu; Binxin Hu; Ling Tang; Jilin Mei; Dadi Guo; Leitao Yuan; Junyao Yang; Guanxu Chen; Qihao Lin; Yi Yu; Bo Zhang; Jiaxuan Guo; Jie Zhang; Wenqi Shao; Huiqi Deng; Zhiheng Xi; Wenjie Wang; Wenxuan Wang; Wen Shen; Zhikai Chen; Haoyu Xie; Jialing Tao; Juntao Dai; Jiaming Ji; Zhongjie Ba; Linfeng Zhang; Yong Liu; Quanshi Zhang; Lei Zhu; Zhihua Wei; Hui Xue; Chaochao Lu; Jing Shao; Xia Hu
>
> **备注:** 40 pages, 26 figures
>
> **摘要:** The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy that orthogonally categorizes agentic risks by their source (where), failure mode (how), and consequence (what). Guided by this structured and hierarchical taxonomy, we introduce a new fine-grained agentic safety benchmark (ATBench) and a Diagnostic Guardrail framework for agent safety and security (AgentDoG). AgentDoG provides fine-grained and contextual monitoring across agent trajectories. More Crucially, AgentDoG can diagnose the root causes of unsafe actions and seemingly safe but unreasonable actions, offering provenance and transparency beyond binary labels to facilitate effective agent alignment. AgentDoG variants are available in three sizes (4B, 7B, and 8B parameters) across Qwen and Llama model families. Extensive experimental results demonstrate that AgentDoG achieves state-of-the-art performance in agentic safety moderation in diverse and complex interactive scenarios. All models and datasets are openly released.
>
---
#### [replaced 047] MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于评估任务，旨在解决MLLM-as-a-Judge中的组合性偏差问题。通过构建MM-JudgeBias基准，分析模型在多模态输入下的可靠性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.18164](https://arxiv.org/pdf/2604.18164)**

> **作者:** Sua Lee; Sanghee Park; Jinbae Im
>
> **备注:** ACL 2026 Main
>
> **摘要:** Multimodal Large Language Models (MLLMs) have been increasingly used as automatic evaluators-a paradigm known as MLLM-as-a-Judge. However, their reliability and vulnerabilities to biases remain underexplored. We find that many MLLM judges fail to reliably integrate key visual or textual cues, yielding unreliable evaluations when evidence is missing or mismatched, and exhibiting instability under semantically irrelevant perturbations. To address this, we systematically define Compositional Bias in MLLM-as-a-Judge systems and introduce MM-JudgeBias, a benchmark for evaluating it. MM-JudgeBias introduces controlled perturbations across Query, Image, and Response, and evaluates model behavior via two complementary metrics: Bias-Deviation (BD) for sensitivity and Bias-Conformity (BC) for stability. Our dataset of over 1,800 curated and refined multimodal samples, drawn from 29 source benchmarks, enables a fine-grained diagnosis of nine bias types across diverse tasks and domains. Experiments on 26 state-of-the-art MLLMs reveal systematic modality neglect and asymmetric evaluation tendencies, underscoring the need for more reliable judges.
>
---
#### [replaced 048] When to Trust the Answer: Question-Aligned Semantic Nearest Neighbor Entropy for Safer Surgical VQA
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.01458](https://arxiv.org/pdf/2511.01458)**

> **作者:** Luca Carlini; Dennis Pierantozzi; Mauro Orazio Drago; Chiara Lena; Cesare Hassan; Elena De Momi; Danail Stoyanov; Sophia Bano; Mobarak I. Hoque
>
> **摘要:** Safety and reliability are critical for deploying visual question answering (VQA) systems in surgery, where incorrect or ambiguous responses can cause patient harm. A key limitation of existing uncertainty estimation methods, such as Semantic Nearest Neighbor Entropy (SNNE), is that they do not explicitly account for the conditioning question. As a result, they may assign high confidence to answers that are semantically consistent yet misaligned with the clinical question, especially under variation in question phrasing. We propose Question-Aligned Semantic Nearest Neighbor Entropy (QA-SNNE), a black-box uncertainty estimator that incorporates question-answer alignment into semantic entropy through bilateral gating. QA-SNNE measures uncertainty by weighting pairwise semantic similarities among sampled answers according to their relevance to the question, using embedding-based, entailment-based, or cross-encoder alignment strategies. To assess robustness to language variation, we construct an out-of-template rephrased version of a benchmark surgical VQA dataset, where only the question wording is modified while images and ground-truth answers remain unchanged. We evaluate QA-SNNE on five VQA models across two benchmark surgical VQA datasets in both zero-shot and parameter-efficient fine-tuned (PEFT) settings, including out-of-template questions. QA-SNNE improves AUROC on EndoVis18-VQA for two of three zero-shot models in-template (e.g., +15% for Llama3.2 and +21% for Qwen2.5) and achieves up to +8% AUROC improvement under out-of-template rephrasing, with mixed results on external validation. Overall, QA-SNNE provides a practical, model-agnostic safeguard for surgical VQA by linking semantic uncertainty to question relevance.
>
---
#### [replaced 049] Fourier Series Coder: A Novel Perspective on Angle Boundary Discontinuity Problem for Oriented Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.20281](https://arxiv.org/pdf/2604.20281)**

> **作者:** Minghong Wei; Pu Cao; Zhihao Chen; Zhiyuan Zang; Lu Yang; Qing Song
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** With the rapid advancement of intelligent driving and remote sensing, oriented object detection has gained widespread attention. However, achieving high-precision performance is fundamentally constrained by the Angle Boundary Discontinuity (ABD) and Cyclic Ambiguity (CA) problems, which typically cause significant angle fluctuations near periodic boundaries. Although recent studies propose continuous angle coders to alleviate these issues, our theoretical and empirical analyses reveal that state-of-the-art methods still suffer from substantial cyclic errors. We attribute this instability to the structural noise amplification within their non-orthogonal decoding mechanisms. This mathematical vulnerability significantly exacerbates angular deviations, particularly for square-like objects. To resolve this fundamentally, we propose the Fourier Series Coder (FSC), a lightweight plug-and-play component that establishes a continuous, reversible, and mathematically robust angle encoding-decoding paradigm. By rigorously mapping angles onto a minimal orthogonal Fourier basis and explicitly enforcing a geometric manifold constraint, FSC effectively prevents feature modulus collapse. This structurally stabilized representation ensures highly robust phase unwrapping, intrinsically eliminating the need for heuristic truncations while achieving strict boundary continuity and superior noise immunity. Extensive experiments across three large-scale datasets demonstrate that FSC achieves highly competitive overall performance, yielding substantial improvements in high-precision detection. The code will be available at this https URL.
>
---
#### [replaced 050] SCASeg: Strip Cross-Attention for Efficient Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.17061](https://arxiv.org/pdf/2411.17061)**

> **作者:** Guoan Xu; Jiaming Chen; Wenfeng Huang; Wenjing Jia; Guangwei Gao; Guo-Jun Qi
>
> **备注:** TIP
>
> **摘要:** The Vision Transformer (ViT) has achieved notable success in computer vision, with its variants widely validated across various downstream tasks, including semantic segmentation. However, as general-purpose visual encoders, ViT backbones often do not fully address the specific requirements of task decoders, highlighting opportunities for designing decoders optimized for efficient semantic segmentation. This paper proposes Strip Cross-Attention (SCASeg), an innovative decoder head specifically designed for semantic segmentation. Instead of relying on the conventional skip connections, we utilize lateral connections between encoder and decoder stages, leveraging encoder features as Queries in cross-attention modules. Additionally, we introduce a Cross-Layer Block (CLB) that integrates hierarchical feature maps from various encoder and decoder stages to form a unified representation for Keys and Values. The CLB also incorporates the local perceptual strengths of convolution, enabling SCASeg to capture both global and local context dependencies across multiple layers, thus enhancing feature interaction at different scales and improving overall efficiency. To further optimize computational efficiency, SCASeg compresses the channels of queries and keys into one dimension, creating strip-like patterns that reduce memory usage and increase inference speed compared to traditional vanilla cross-attention. Experiments show that SCASeg's adaptable decoder delivers competitive performance across various setups, outperforming leading segmentation architectures on benchmark datasets, including ADE20K, Cityscapes, COCO-Stuff 164k, and Pascal VOC2012, even under diverse computational constraints.
>
---
#### [replaced 051] Information Bottleneck-Guided Heterogeneous Graph Learning for Interpretable Neurodevelopmental Disorder Diagnosis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.20769](https://arxiv.org/pdf/2502.20769)**

> **作者:** Yueyang Li; Lei Chen; Wenhao Dong; Shengyu Gong; Zijian Kang; Boyang Wei; Weiming Zeng; Hongjie Yan; Lingbin Bian; Zhiguo Zhang; Wai Ting Siok; Nizhuan Wang
>
> **摘要:** Developing interpretable models for neurodevelopmental disorders (NDDs) diagnosis presents significant challenges in effectively encoding, decoding, and integrating multimodal neuroimaging data. While many existing machine learning approaches have shown promise in brain network analysis, they typically suffer from limited interpretability, particularly in extracting meaningful biomarkers from functional magnetic resonance imaging (fMRI) data and establishing clear relationships between imaging features and demographic characteristics. Besides, current graph neural network methodologies face limitations in capturing both local and global functional connectivity patterns while simultaneously achieving theoretically principled multimodal data fusion. To address these challenges, we propose the Interpretable Information Bottleneck Heterogeneous Graph Neural Network (I2B-HGNN), a unified framework that applies information bottleneck principles to guide both brain connectivity modeling and cross-modal feature integration. This framework comprises two complementary components. The first is the Information Bottleneck Graph Transformer (IBGraphFormer), which combines transformer-based global attention mechanisms with graph neural networks through information bottleneck-guided pooling to identify sufficient biomarkers. The second is the Information Bottleneck Heterogeneous Graph Attention Network (IB-HGAN), which employs meta-path-based heterogeneous graph learning with structural consistency constraints to achieve interpretable fusion of neuroimaging and demographic data. The experimental results demonstrate that I2B-HGNN achieves superior performance in diagnosing NDDs, exhibiting both high classification accuracy and the ability to provide interpretable biomarker identification while effectively analyzing non-imaging data.
>
---
#### [replaced 052] GeCo: Evaluating Geometric Consistency for Video Generation via Motion and Structure
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.22274](https://arxiv.org/pdf/2512.22274)**

> **作者:** Leslie Gu; Junhwa Hur; Charles Herrmann; Fangneng Zhan; Todd Zickler; Deqing Sun; Hanspeter Pfister
>
> **摘要:** We introduce GeCo, a geometry-grounded metric for jointly detecting geometric deformation and occlusion-inconsistency artifacts in static scenes. By fusing residual motion and depth priors, GeCo produces interpretable, dense consistency maps that reveal these artifacts. We use GeCo to systematically benchmark recent video generation models, uncovering common failure modes, and further employ it as a training-free guidance loss to reduce deformation artifacts during video generation.
>
---
#### [replaced 053] Render-in-the-Loop: Vector Graphics Generation via Visual Self-Feedback
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.20730](https://arxiv.org/pdf/2604.20730)**

> **作者:** Guotao Liang; Zhangcheng Wang; Juncheng Hu; Haitao Zhou; Ziteng Xue; Jing Zhang; Dong Xu; Qian Yu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown promising capabilities in generating Scalable Vector Graphics (SVG) via direct code synthesis. However, existing paradigms typically adopt an open-loop "blind drawing" approach, where models generate symbolic code sequences without perceiving intermediate visual outcomes. This methodology severely underutilizes the powerful visual priors embedded in MLLMs vision encoders, treating SVG generation as a disjointed textual sequence modeling task rather than an integrated visuo-spatial one. Consequently, models struggle to reason about partial canvas states and implicit occlusion relationships, which are visually explicit but textually ambiguous. To bridge this gap, we propose Render-in-the-Loop, a novel generation paradigm that reformulates SVG synthesis as a step-wise, visual-context-aware process. By rendering intermediate code states into a cumulative canvas, the model explicitly observes the evolving visual context at each step, leveraging on-the-fly feedback to guide subsequent generation. However, we demonstrate that applying this visual loop naively to off-the-shelf models is suboptimal due to their inability to leverage incremental visual-code mappings. To address this, we first utilize fine-grained path decomposition to construct dense multi-step visual trajectories, and then introduce a Visual Self-Feedback (VSF) training strategy to condition the next primitive generation on intermediate visual states. Furthermore, a Render-and-Verify (RaV) inference mechanism is proposed to effectively filter degenerate and redundant primitives. Our framework, instantiated on a multimodal foundation model, outperforms strong open-weight baselines on the standard MMSVGBench. This result highlights the remarkable data efficiency and generalization capability of our Render-in-the-Loop paradigm for both Text-to-SVG and Image-to-SVG tasks.
>
---
#### [replaced 054] TimePre: Bridging Accuracy, Efficiency, and Stability in Probabilistic Time-Series Forecasting
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18539](https://arxiv.org/pdf/2511.18539)**

> **作者:** Lingyu Jiang; Lingyu Xu; Peiran Li; Dengzhe Hou; Qianwen Ge; Dingyi Zhuang; Shuo Xing; Wenjing Chen; Xiangbo Gao; Ting-Hsuan Chen; Xueying Zhan; Xin Zhang; Ziming Zhang; Zhengzhong Tu; Michael Zielewski; Kazunori Yamada; Fangzhou Lin
>
> **备注:** 15 pages, 5 figures, 6 tables
>
> **摘要:** We propose TimePre, a simple framework that unifies the efficiency of Multilayer Perceptron (MLP)-based models with the distributional flexibility of Multiple Choice Learning (MCL) for Probabilistic Time-Series Forecasting (PTSF). Stabilized Instance Normalization (SIN), the core of TimePre, is a normalization layer that explicitly addresses the trade-off among accuracy, efficiency, and stability. SIN stabilizes the hybrid architecture by correcting channel-wise statistical shifts, thereby resolving the catastrophic hypothesis collapse. Extensive experiments on six benchmark datasets demonstrate that TimePre achieves state-of-the-art (SOTA) accuracy on key probabilistic metrics. Critically, TimePre achieves inference speeds that are orders of magnitude faster than sampling-based models, and is more stable than prior MCL approaches.
>
---
#### [replaced 055] ImVideoEdit: Image-learning Video Editing via 2D Spatial Difference Attention Blocks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.07958](https://arxiv.org/pdf/2604.07958)**

> **作者:** Jiayang Xu; Fan Zhuo; Majun Zhang; Changhao Pan; Zehan Wang; Siyu Chen; Xiaoda Yang; Tao Jin; Zhou Zhao
>
> **摘要:** Current video editing models often rely on expensive paired video data, which limits their practical scalability. In essence, most video editing tasks can be formulated as a decoupled spatiotemporal process, where the temporal dynamics of the pretrained model are preserved while spatial content is selectively and precisely modified. Based on this insight, we propose ImVideoEdit, an efficient framework that learns video editing capabilities entirely from image pairs. By freezing the pre-trained 3D attention modules and treating images as single-frame videos, we decouple the 2D spatial learning process to help preserve the original temporal dynamics. The core of our approach is a Predict-Update Spatial Difference Attention module that progressively extracts and injects spatial differences. Rather than relying on rigid external masks, we incorporate a Text-Guided Dynamic Semantic Gating mechanism for adaptive and implicit text-driven modifications. Despite training on only 13K image pairs for 5 epochs with exceptionally low computational overhead, ImVideoEdit achieves editing fidelity and temporal consistency comparable to larger models trained on extensive video datasets.
>
---
#### [replaced 056] VVS: Accelerating Speculative Decoding for Visual Autoregressive Generation via Partial Verification Skipping
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.13587](https://arxiv.org/pdf/2511.13587)**

> **作者:** Haotian Dong; Ye Li; Rongwei Lu; Chen Tang; Shu-Tao Xia; Zhi Wang
>
> **备注:** CVPR 2026
>
> **摘要:** Visual autoregressive (AR) generation models have demonstrated strong potential for image generation, yet their next-token-prediction paradigm introduces considerable inference latency. Although speculative decoding (SD) has been proven effective for accelerating visual AR models, its "draft one step, then verify one step" paradigm prevents a direct reduction in the number of forward passes, limiting its acceleration potential. Motivated by the interchangeability of visual tokens, we explore verification skipping in the SD process for the first time to explicitly cut the number of target model forward passes, thereby reducing inference latency. By analyzing the characteristics of the drafting stage, we observe that verification redundancy and stale feature reusability are key factors to maintain generation quality while improving speed for verification-free steps. Inspired by these two observations, we propose a novel SD framework VVS to accelerate visual AR model via partial verification skipping, which integrates three complementary modules: (1) a verification-free token selector with dynamic truncation, (2) token-level feature caching and reuse, and (3) fine-grained skipped step scheduling. Consequently, VVS reduces the number of target model forward passes by $2.8\times$ relative to vanilla AR decoding while maintaining competitive generation quality, offering a superior speed-quality trade-off over conventional SD frameworks and revealing strong potential to reshape the SD paradigm. Our code is available at this https URL.
>
---
#### [replaced 057] Anatomy-Aware Text-Visual Fusion with Dual-Perspective Prompts for Fine-Grained Lumbar Spine Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.03476](https://arxiv.org/pdf/2504.03476)**

> **作者:** Sheng Lian; Jianlong Cai; Dengfeng Pan; Guang-Yong Chen; Hao Xu; Fan Zhang; Guodong Fan; Shuo Li
>
> **摘要:** Accurate lumbar spine segmentation is crucial for diagnosing spinal disorders. Existing methods typically use coarse-grained segmentation strategies that lack the fine detail needed for precise diagnosis. Additionally, their reliance on visual-only models hinders the capture of anatomical semantics, leading to misclassified categories and poor segmentation details. To address these limitations, we present ATM-Net, an innovative framework that employs an anatomy-aware, text-guided, multi-modal fusion mechanism for fine-grained segmentation of lumbar substructures, i.e., vertebrae (VBs), intervertebral discs (IDs), and spinal canal (SC). ATM-Net adopts the Anatomy-aware Text Prompt Generator (ATPG) to adaptively convert image annotations into anatomy-aware prompts in different views. These insights are further integrated with image features via the Holistic Anatomy-aware Semantic Fusion (HASF) module, building a comprehensive anatomical context. The Channel-wise Contrastive Anatomy-Aware Enhancement (CCAE) module further enhances class discrimination and refines segmentation through class-wise channel-level multi-modal contrastive learning. Extensive experiments on the MRSpineSeg and SPIDER datasets demonstrate that ATM-Net significantly outperforms state-of-the-art methods, with consistent improvements regarding class discrimination and segmentation details. For example, ATM-Net achieves Dice of 79.39% and HD95 of 9.91 pixels on SPIDER, outperforming the competitive SpineParseNet by 8.31% and 4.14 pixels, respectively.
>
---
#### [replaced 058] Geometry-aided Vision-based Localization of Future Mars Helicopters in Challenging Illumination Conditions
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，解决火星直升机在光照变化下的定位问题。提出Geo-LoFTR模型，提升图像匹配鲁棒性。**

- **链接: [https://arxiv.org/pdf/2502.09795](https://arxiv.org/pdf/2502.09795)**

> **作者:** Dario Pisanti; Robert Hewitt; Roland Brockers; Georgios Georgakis
>
> **摘要:** Planetary exploration using aerial assets has the potential for unprecedented scientific discoveries on Mars. While NASA's Mars helicopter Ingenuity proved flight in Martian atmosphere is possible, future Mars rotorcraft will require advanced navigation capabilities for long-range flights. One such critical capability is Map-based Localization (MbL) which registers an onboard image to a reference map during flight to mitigate cumulative drift from visual odometry. However, significant illumination differences between rotorcraft observations and a reference map prove challenging for traditional MbL systems, restricting the operational window of the vehicle. In this work, we investigate a new MbL system and propose Geo-LoFTR, a geometry-aided deep learning model for image registration that is more robust under large illumination differences than prior models. The system is supported by a custom simulation framework that uses real orbital maps to produce large amounts of realistic images of the Martian terrain. Comprehensive evaluations show that our proposed system outperforms prior MbL efforts in terms of localization accuracy under significant lighting and scale variations. Furthermore, we demonstrate the validity of our approach across a simulated Martian day and on real Mars imagery. Code and datasets are available at: this https URL.
>
---
#### [replaced 059] SurgViVQA: Temporally-Grounded Video Question Answering for Surgical Scene Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.03325](https://arxiv.org/pdf/2511.03325)**

> **作者:** Mauro Orazio Drago; Luca Carlini; Pelinsu Celebi Balyemez; Dennis Pierantozzi; Chiara Lena; Cesare Hassan; Danail Stoyanov; Elena De Momi; Sophia Bano; Mobarak I. Hoque
>
> **摘要:** Video Question Answering (VideoQA) in the surgical domain aims to enhance intraoperative understanding by enabling AI models to reason over temporally coherent events rather than isolated frames. Current approaches are limited to static image features, and available datasets often lack temporal annotations, ignoring the dynamics critical for accurate procedural interpretation. We propose SurgViVQA, a surgical VideoQA model that extends visual reasoning from static images to dynamic surgical scenes. It uses a Masked Video--Text Encoder to fuse video and question features, capturing temporal cues such as motion and tool--tissue interactions, which a fine-tuned large language model (LLM) then decodes into coherent answers. To evaluate its performance, we curated REAL-Colon-VQA, a colonoscopic video dataset that includes motion-related questions and diagnostic attributes, as well as out-of-template questions with rephrased or semantically altered formulations to assess model robustness. Experimental validation on REAL-Colon-VQA and the public EndoVis18-VQA dataset shows that SurgViVQA outperforms existing image-based VQA benchmark models, particularly in keyword accuracy, improving over PitVQA by +11\% on REAL-Colon-VQA and +9\% on EndoVis18-VQA. A perturbation study on the questions further confirms improved generalizability and robustness to variations in question phrasing. SurgViVQA and the REAL-Colon-VQA dataset provide a framework for temporally-aware understanding in surgical VideoQA, enabling AI models to interpret dynamic procedural contexts more effectively. Code and dataset available at this https URL.
>
---
#### [replaced 060] LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15269](https://arxiv.org/pdf/2505.15269)**

> **作者:** Zhenyu Ning; Guangda Liu; Qihao Jin; Chengwei Li; Wenchao Ding; Minyi Guo; Jieru Zhao
>
> **备注:** Accepted by DAC'26
>
> **摘要:** Recent developments in Video Large Language Models (Video LLMs) have enabled models to process hour-long videos and exhibit exceptional performance. Nonetheless, the Key-Value (KV) cache expands linearly over time, leading to substantial memory overhead and response delay--critical challenges in various real-world online applications, such as Deepseek services, autonomous driving and robotics. To mitigate these issues, we propose $\textbf{LiveVLM}$, a training-free and query-agnostic framework specifically designed for online video understanding and real-time interaction. LiveVLM employs a Vision Sink Bucketing (VSB) mechanism to process video streams in real time, retain long-term video details and eliminate redundant KVs. This mechanism utilizes vision-to-vision attention scores as the metric and seeks to maximize the coverage of contextual information during compression. Noting that KV cache compressed in a query-agnostic manner inevitably retains irrelevant information for specific queries, LiveVLM incorporates a Position-agnostic KV Retrieval (PaR) mechanism to reduce interference from redundant context. The keypoint of PaR lies in decoupling positional embeddings to enhance the similarity between key tensors, thereby supporting efficient retrieval at the granularity of pages. Extensive experiments demonstrate that LiveVLM enables the foundation LLaVA-OneVision model to achieve state-of-the-art accuracy among both training-free query-agnostic methods and training-based online models.
>
---
#### [replaced 061] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18792](https://arxiv.org/pdf/2602.18792)**

> **作者:** Changlu Guo; Anders Nymark Christensen; Anders Bjorholm Dahl; Morten Rieger Hannemose
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Visual counterfactual explanations aim to reveal the minimal semantic modifications that can alter a model's prediction, providing causal and interpretable insights into deep neural networks. However, existing diffusion-based counterfactual generation methods are often computationally expensive, slow to sample, and imprecise in localizing the modified regions. To address these limitations, we propose MaskDiME, a simple, fast, yet effective diffusion framework that unifies semantic consistency and spatial precision through localized sampling. Our approach adaptively focuses on decision-relevant regions to achieve localized and semantically consistent counterfactual generation while preserving high image fidelity. Our training-free framework, MaskDiME, performs inference over 30x faster than the baseline and achieves comparable or state-of-the-art performance across five benchmark datasets spanning diverse visual domains, establishing a practical and generalizable solution for efficient counterfactual explanation.
>
---
#### [replaced 062] SGG-R$^{\rm 3}$: From Next-Token Prediction to End-to-End Unbiased Scene Graph Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07961](https://arxiv.org/pdf/2603.07961)**

> **作者:** Jiaye Feng; Qixiang Yin; Yuankun Liu; Tong Mo; Weiping Li
>
> **摘要:** Scene Graph Generation (SGG) structures visual scenes as graphs of objects and their relations. While Multimodal Large Language Models (MLLMs) have advanced end-to-end SGG, current methods are hindered by both a lack of task-specific structured reasoning and the challenges of sparse, long-tailed relation distributions, resulting in incomplete scene graphs characterized by low recall and biased predictions. To address these issues, we introduce SGG-R$^{\rm 3}$, a structured reasoning framework that integrates task-specific chain-of-thought (CoT)-guided supervised fine-tuning (SFT) and reinforcement learning (RL) with group sequence policy optimization (GSPO), designed to engage in three sequential stages to achieve end-to-end unbiased scene graph generation. During the SFT phase, we propose a relation augmentation strategy by leveraging an MLLM and refined via embedding similarity filtering to alleviate relation sparsity. Subsequently, a stage-aligned reward scheme optimizes the procedural reasoning during RL. Specifically, we propose a novel dual-granularity reward which integrates fine-grained and coarse-grained relation rewards, simultaneously mitigating the long-tail issue via frequency-based adaptive weighting of predicates and improving relation coverage through semantic clustering. Experiments on two benchmarks show that SGG-R$^{\rm 3}$ achieves superior performance compared to existing methods, demonstrating the effectiveness and generalization of the framework.
>
---
#### [replaced 063] DepthMaster: Taming Diffusion Models for Monocular Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.02576](https://arxiv.org/pdf/2501.02576)**

> **作者:** Ziyang Song; Zerong Wang; Bo Li; Hao Zhang; Ruijie Zhu; Li Liu; Peng-Tao Jiang; Tianzhu Zhang
>
> **备注:** 11 pages, 6 figures, 6 tables
>
> **摘要:** Monocular depth estimation within the diffusion-denoising paradigm demonstrates impressive generalization ability but suffers from low inference speed. Recent methods adopt a single-step deterministic paradigm to improve inference efficiency while maintaining comparable performance. However, they overlook the gap between generative and discriminative features, leading to suboptimal results. In this work, we propose DepthMaster, a single-step diffusion model designed to adapt generative features for the discriminative depth estimation task. First, to mitigate overfitting to texture details introduced by generative features, we propose a Feature Alignment module, which incorporates high-quality semantic features to enhance the denoising network's representation capability. Second, to address the lack of fine-grained details in the single-step deterministic framework, we propose a Fourier Enhancement module to adaptively balance low-frequency structure and high-frequency details. We adopt a two-stage training strategy to fully leverage the potential of the two modules. In the first stage, we focus on learning the global scene structure with the Feature Alignment module, while in the second stage, we exploit the Fourier Enhancement module to improve the visual quality. Through these efforts, our model achieves state-of-the-art performance in terms of generalization and detail preservation, outperforming other diffusion-based methods across various datasets. Our project page can be found at this https URL.
>
---
#### [replaced 064] FastSHADE: Fast Self-augmented Hierarchical Asymmetric Denoising for Efficient inference on mobile devices
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10275](https://arxiv.org/pdf/2604.10275)**

> **作者:** Nikolay Falaleev
>
> **备注:** To appear in the Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) 2026
>
> **摘要:** Real-time image denoising is essential for modern mobile photography but remains challenging due to the strict latency and power constraints of edge devices. This paper presents FastSHADE (Fast Self-augmented Hierarchical Asymmetric Denoising), a lightweight U-Net-style network tailored for real-time, high-fidelity restoration on mobile GPUs. Our method features a multi-stage architecture incorporating a novel Asymmetric Frequency Denoising Block (AFDB) that decouples spatial structure extraction from high-frequency noise suppression to maximize efficiency, and a Spatially Gated Upsampler (SGU) that optimizes high-resolution skip connection fusion. To address generalization, we introduce an efficient Noise Shifting Self-Augmentation strategy that enhances data diversity without inducing domain shifts. Evaluations on the MAI2021 benchmark demonstrate that our scalable model family establishes a highly efficient speed-fidelity trade-off. Our base FastSHADE-M variant maintains real-time latency (<50 ms on an Adreno 840 GPU) while preserving structural integrity, and our scaled-up FastSHADE-XL establishes a new state-of-the-art for overall image quality, achieving 37.94 dB PSNR.
>
---
#### [replaced 065] Efficient Multi-Source Knowledge Transfer by Model Merging
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.19353](https://arxiv.org/pdf/2508.19353)**

> **作者:** Marcin Osial; Bartosz Wójcik; Bartosz Zieliński; Sebastian Cygert
>
> **摘要:** While transfer learning is an effective strategy, it often overlooks the opportunity to leverage knowledge from numerous available models online. Addressing this multi-source transfer learning problem is a promising path to boost adaptability and cut re-training costs. However, existing methods remain inherently coarse-grained: they lack the precision needed for fine-grained knowledge extraction as well as the scalability required to aggregate knowledge from either large numbers of source models or models with high parameter counts. We address these limitations by leveraging Singular Value Decomposition (SVD) to first decompose each source model into its elementary, rank-one components. A subsequent aggregation stage then selects only the most salient components from all sources, thereby overcoming the previous efficiency and precision limitations. To best preserve and leverage the synthesized knowledge base, our method adapts to the target task by fine-tuning only the principal singular values of the merged matrix. In essence, this process recalibrates the importance of top SVD components. The proposed framework allows for efficient and scalable multi-source transfer learning in both vision and language domains, while remaining robust to perturbations in both the input space and the parameter space.
>
---
#### [replaced 066] Preserving Knowledge in Large Language Model with Model-Agnostic Self-Decompression
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于自然语言处理任务，解决LLM在微调中的灾难性遗忘问题及多模态模型性能下降问题，提出TG-SFT方法生成训练数据以减少遗忘。**

- **链接: [https://arxiv.org/pdf/2406.11354](https://arxiv.org/pdf/2406.11354)**

> **作者:** Zilun Zhang; Yutao Sun; Tiancheng Zhao; Leigang Sha; Ruochen Xu; Kyusong Lee; Jianwei Yin
>
> **备注:** Accepted by ICASSP 2026 (Oral)
>
> **摘要:** Humans can retain old knowledge while learning new information, but Large Language Models (LLMs) often suffer from catastrophic forgetting when post-pretrained or supervised fine-tuned (SFT) on domain-specific data. Moreover, for Multimodal Large Language Models (MLLMs) which are composed of the LLM base and visual projector (e.g. LLaVA), a significant decline in performance on language benchmarks was observed compared to their single-modality counterparts. To address these challenges, we introduce a novel model-agnostic self-decompression method, Tree Generation (TG), that decompresses knowledge within LLMs into the training corpus. This paper focuses on TG-SFT, which can synthetically generate SFT data for the instruction tuning steps. By incorporating the dumped corpus during SFT for MLLMs, we significantly reduce the forgetting problem.
>
---
#### [replaced 067] Unsharp Measurement with Adaptive Gaussian POVMs for Quantum-Inspired Image Processing
- **分类: quant-ph; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.04685](https://arxiv.org/pdf/2604.04685)**

> **作者:** Debashis Saikia; Bikash K. Behera; Mayukha Pal; Prasanta K. Panigrahi
>
> **摘要:** We propose a data-adaptive probabilistic intensity remapping framework for structure-preserving transformation of grayscale images. The suggested method formulates intensity transformation as a continuous, data-driven remapping process, in contrast to traditional histogram-based techniques that rely on hard thresholding and generate piecewise-constant mappings. The image statistics yield representative intensity values, and Gaussian-based weighting methods probabilistically allocate each pixel to several components. Smooth transitions while preserving structural features are achieved by computing the output intensity as an expectation over these components. A smooth transition from soft probabilistic remapping to hard assignment is made possible by the introduction of a nonlinear sharpening parameter $\gamma$ to regulate the degree of localization. This offers clear control over the trade-off between intensity discrimination and smoothing. Furthermore, the resolution of the remapping function is determined by the number of components $k$. When compared to thresholding-based methods, experimental results on standard benchmark images show that the suggested method achieves better structural fidelity and controlled information reduction as measured by PSNR, SSIM, and entropy. Overall, by allowing continuous, probabilistic intensity modifications, the framework provides a robust and efficient substitute for discrete thresholding.
>
---
#### [replaced 068] APCoTTA: Continual Test-Time Adaptation for Semantic Segmentation of Airborne LiDAR Point Clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.09971](https://arxiv.org/pdf/2505.09971)**

> **作者:** Yuan Gao; Shaobo Xia; Sheng Nie; Cheng Wang; Xiaohuan Xi; Bisheng Yang
>
> **备注:** 18 pages,12 figures
>
> **摘要:** Airborne laser scanning (ALS) point cloud semantic segmentation is a fundamental task for large-scale 3D scene understanding. Fixed models deployed in real-world scenarios often suffer from performance degradation due to continuous domain shifts caused by environmental and sensor changes. Continuous Test-Time Adaptation (CTTA) enables adaptation to evolving unlabeled domains, but its application to ALS point clouds remains underexplored, hindered by the lack of benchmarks and the risks of catastrophic forgetting and error accumulation. To address these challenges, we propose APCoTTA (ALS Point cloud Continuous Test-Time Adaptation), a novel CTTA framework tailored for ALS point cloud semantic segmentation. APCoTTA consists of three key components. First, we adapt a gradient-driven layer selection mechanism for ALS point clouds, selectively updating low-confidence layers while freezing stable ones to preserve source knowledge and mitigate catastrophic forgetting. Second, an entropy-based consistency loss discards unreliable samples and enforces consistency regularization solely on reliable ones, effectively reducing error accumulation and improving adaptation stability. Third, a random parameter interpolation mechanism stochastically blends adapted parameters with source model parameters, further balancing target adaptation and source knowledge retention. Finally, we construct two benchmarks, ISPRSC and H3DC, to address the lack of CTTA benchmarks for ALS point cloud segmentation. Extensive experiments demonstrate that APCoTTA achieves superior performance on both benchmarks, improving mIoU by approximately 9\% and 14\% over direct inference. The new benchmarks and code are available at this https URL.
>
---
#### [replaced 069] Low Cost, High Efficiency: LiDAR Place Recognition in Vineyards with Matryoshka Representation Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于机器人定位任务，解决农业环境中基于LiDAR的场景识别问题。提出MinkUNeXt-VINE方法，采用多损失结构提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2601.18714](https://arxiv.org/pdf/2601.18714)**

> **作者:** Judith Vilella-Cantos; Mauro Martini; Marcello Chiaberge; Mónica Ballesta; David Valiente
>
> **摘要:** Localization in agricultural environments is challenging due to their unstructured nature and lack of distinctive landmarks. Although agricultural settings have been studied in the context of object classification and segmentation, the place recognition task for mobile robots is not trivial in the current state of the art. In this study, we propose MinkUNeXt-VINE, a lightweight, deep-learning-based method that surpasses state-of-the-art methods in vineyard environments thanks to its pre-processing and Matryoshka Representation Learning multi-loss approach. Our method prioritizes enhanced performance with low-cost, sparse LiDAR inputs and lower-dimensionality outputs to ensure high efficiency in real-time scenarios. Additionally, we present a comprehensive ablation study of the results on various evaluation cases and two extensive long-term vineyard datasets employing different LiDAR sensors. The results demonstrate the efficiency of the trade-off output produced by this approach, as well as its robust performance on low-cost and low-resolution input data. The code is publicly available for reproduction.
>
---
#### [replaced 070] UbiQVision: Quantifying Uncertainty in XAI for Image Recognition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.20288](https://arxiv.org/pdf/2512.20288)**

> **作者:** Akshat Dubey; Aleksandar Anžel; Bahar İlgen; Georges Hattab
>
> **备注:** Under Review. Updated manuscript. Feedback from reviewers incorporated
>
> **摘要:** Recent advances in deep learning have led to its widespread adoption across diverse domains, including medical imaging. This progress is driven by increasingly sophisticated model architectures, such as ResNets, Vision Transformers, and Hybrid Convolutional Neural Networks, that offer enhanced performance at the cost of greater complexity. This complexity often compromises model explainability and interpretability. SHAP has emerged as a prominent method for providing interpretable visualizations that aid domain experts in understanding model predictions. However, SHAP explanations can be unstable and unreliable in the presence of epistemic and aleatoric uncertainty. In this study, we address this challenge by using Dirichlet posterior sampling and Dempster-Shafer theory to quantify the uncertainty that arises from these unstable explanations in medical imaging applications. The framework uses a belief, plausible, and fusion map approach alongside statistical quantitative analysis to produce quantification of uncertainty in SHAP. Furthermore, we evaluated our framework on three medical imaging datasets with varying class distributions, image qualities, and modality types which introduces noise due to varying image resolutions and modality-specific aspect covering the examples from pathology, ophthalmology, and radiology, introducing significant epistemic uncertainty.
>
---
