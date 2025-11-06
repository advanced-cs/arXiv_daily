# 计算机视觉 cs.CV

- **最新发布 58 篇**

- **更新 59 篇**

## 最新发布

#### [new 001] SCALE-VLP: Soft-Weighted Contrastive Volumetric Vision-Language Pre-training with Spatial-Knowledge Semantics
- **分类: cs.CV**

- **简介: SCALE-VLP提出一种面向三维医学影像（如CT）的视觉-语言预训练框架，解决传统方法忽略空间结构与语义连续性的问题，通过软加权对比学习融合解剖结构与医学本体知识，提升跨任务与跨域的泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.02996v1](http://arxiv.org/pdf/2511.02996v1)**

> **作者:** Ailar Mahdizadeh; Puria Azadi Moghadam; Xiangteng He; Shahriar Mirabbasi; Panos Nasiopoulos; Leonid Sigal
>
> **摘要:** Vision-language models (VLMs) have demonstrated strong cross-modal capabilities, yet most work remains limited to 2D data and assumes binary supervision (i.e., positive vs. negative pairs), overlooking the continuous and structured dependencies present in volumetric data such as CT. Existing approaches often treat volumetric scans as independent 2D slices, compromising spatial coherence and underutilizing rich clinical semantics. We propose SCALE-VLP, a soft-weighted contrastive vision-language pre-training framework that integrates (i) volumetric spatial semantics to preserve anatomical structure and (ii) domain-aware, knowledge-infused semantics (e.g., radiological ontologies) to guide alignment. This yields structurally consistent and semantically grounded representations under limited supervision, demonstrating strong cross-task transferability (retrieval, report generation, and classification), and cross-domain generalizability with consistent gains without further fine-tuning. In particular, compared to the previous state of the art, SCALE-VLP achieves up to 4.3x higher top-1 CT-report retrieval, improves abnormality classification by 10 points, and reaches ROUGE-L 0.44 and BERT-F1 0.89 for report generation. Further, in zero-shot evaluation on an out-of-domain external dataset, we observe consistent gains, indicating the cross-task and cross-domain generalization ability of SCALE-VLP.
>
---
#### [new 002] From Propagation to Prediction: Point-level Uncertainty Evaluation of MLS Point Clouds under Limited Ground Truth
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出一种无地面真值的点云不确定性评估框架，通过几何特征与XGBoost模型预测MLS点云的逐点不确定性，解决传统方法依赖高成本GT的问题，验证了不确定性可学习性。**

- **链接: [http://arxiv.org/pdf/2511.03053v1](http://arxiv.org/pdf/2511.03053v1)**

> **作者:** Ziyang Xu; Olaf Wysocki; Christoph Holst
>
> **摘要:** Evaluating uncertainty is critical for reliable use of Mobile Laser Scanning (MLS) point clouds in many high-precision applications such as Scan-to-BIM, deformation analysis, and 3D modeling. However, obtaining the ground truth (GT) for evaluation is often costly and infeasible in many real-world applications. To reduce this long-standing reliance on GT in uncertainty evaluation research, this study presents a learning-based framework for MLS point clouds that integrates optimal neighborhood estimation with geometric feature extraction. Experiments on a real-world dataset show that the proposed framework is feasible and the XGBoost model delivers fully comparable accuracy to Random Forest while achieving substantially higher efficiency (about 3 times faster), providing initial evidence that geometric features can be used to predict point-level uncertainty quantified by the C2C distance. In summary, this study shows that MLS point clouds' uncertainty is learnable, offering a novel learning-based viewpoint towards uncertainty evaluation research.
>
---
#### [new 003] Accelerating Physical Property Reasoning for Augmented Visual Cognition
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出\sysname，加速视觉引导的物理属性推理，解决传统方法延迟高（10-20分钟）的问题。通过几何重建、特征融合与并行编码等优化，将延迟降至6秒内，提升精度与效率，并在智能眼镜上实现实时应用。**

- **链接: [http://arxiv.org/pdf/2511.03126v1](http://arxiv.org/pdf/2511.03126v1)**

> **作者:** Hongbo Lan; Zhenlin An; Haoyu Li; Vaibhav Singh; Longfei Shangguan
>
> **摘要:** This paper introduces \sysname, a system that accelerates vision-guided physical property reasoning to enable augmented visual cognition. \sysname minimizes the run-time latency of this reasoning pipeline through a combination of both algorithmic and systematic optimizations, including rapid geometric 3D reconstruction, efficient semantic feature fusion, and parallel view encoding. Through these simple yet effective optimizations, \sysname reduces the end-to-end latency of this reasoning pipeline from 10--20 minutes to less than 6 seconds. A head-to-head comparison on the ABO dataset shows that \sysname achieves this 62.9$\times$--287.2$\times$ speedup while not only reaching on-par (and sometimes slightly better) object-level physical property estimation accuracy(e.g. mass), but also demonstrating superior performance in material segmentation and voxel-level inference than two SOTA baselines. We further combine gaze-tracking with \sysname to localize the object of interest in cluttered, real-world environments, streamlining the physical property reasoning on smart glasses. The case study with Meta Aria Glasses conducted at an IKEA furniture store demonstrates that \sysname achives consistently high performance compared to controlled captures, providing robust property estimations even with fewer views in real-world scenarios.
>
---
#### [new 004] SurgViVQA: Temporally-Grounded Video Question Answering for Surgical Scene Understanding
- **分类: cs.CV**

- **简介: 论文提出SurgViVQA，面向手术视频的时序问答任务，解决传统方法忽略动态时序信息的问题。通过掩码视频-文本编码器建模运动与工具-组织交互，并构建REAL-Colon-VQA数据集，显著提升问答准确率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.03325v1](http://arxiv.org/pdf/2511.03325v1)**

> **作者:** Mauro Orazio Drago; Luca Carlini; Pelinsu Celebi Balyemez; Dennis Pierantozzi; Chiara Lena; Cesare Hassan; Danail Stoyanov; Elena De Momi; Sophia Bano; Mobarak I. Hoque
>
> **摘要:** Video Question Answering (VideoQA) in the surgical domain aims to enhance intraoperative understanding by enabling AI models to reason over temporally coherent events rather than isolated frames. Current approaches are limited to static image features, and available datasets often lack temporal annotations, ignoring the dynamics critical for accurate procedural interpretation. We propose SurgViVQA, a surgical VideoQA model that extends visual reasoning from static images to dynamic surgical scenes. It uses a Masked Video--Text Encoder to fuse video and question features, capturing temporal cues such as motion and tool--tissue interactions, which a fine-tuned large language model (LLM) then decodes into coherent answers. To evaluate its performance, we curated REAL-Colon-VQA, a colonoscopic video dataset that includes motion-related questions and diagnostic attributes, as well as out-of-template questions with rephrased or semantically altered formulations to assess model robustness. Experimental validation on REAL-Colon-VQA and the public EndoVis18-VQA dataset shows that SurgViVQA outperforms existing image-based VQA benchmark models, particularly in keyword accuracy, improving over PitVQA by +11\% on REAL-Colon-VQA and +9\% on EndoVis18-VQA. A perturbation study on the questions further confirms improved generalizability and robustness to variations in question phrasing. SurgViVQA and the REAL-Colon-VQA dataset provide a framework for temporally-aware understanding in surgical VideoQA, enabling AI models to interpret dynamic procedural contexts more effectively. Code and dataset available at https://github.com/madratak/SurgViVQA.
>
---
#### [new 005] Robust Alignment of the Human Embryo in 3D Ultrasound using PCA and an Ensemble of Heuristic, Atlas-based and Learning-based Classifiers Evaluated on the Rotterdam Periconceptional Cohort
- **分类: cs.CV; I.4**

- **简介: 该论文提出一种基于PCA与多分类器融合的3D超声胚胎自动对齐方法，解决胎儿发育监测中标准切面难以一致获取的问题，在千余例数据上实现98.5%对齐准确率。**

- **链接: [http://arxiv.org/pdf/2511.03416v1](http://arxiv.org/pdf/2511.03416v1)**

> **作者:** Nikolai Herrmann; Marcella C. Zijta; Stefan Klein; Régine P. M. Steegers-Theunissen; Rene M. H. Wijnen; Bernadette S. de Bakker; Melek Rousian; Wietske A. P. Bastiaansen
>
> **备注:** Submitted version of paper accepted at International Workshop on Preterm, Perinatal and Paediatric Image Analysis 2025
>
> **摘要:** Standardized alignment of the embryo in three-dimensional (3D) ultrasound images aids prenatal growth monitoring by facilitating standard plane detection, improving visualization of landmarks and accentuating differences between different scans. In this work, we propose an automated method for standardizing this alignment. Given a segmentation mask of the embryo, Principal Component Analysis (PCA) is applied to the mask extracting the embryo's principal axes, from which four candidate orientations are derived. The candidate in standard orientation is selected using one of three strategies: a heuristic based on Pearson's correlation assessing shape, image matching to an atlas through normalized cross-correlation, and a Random Forest classifier. We tested our method on 2166 images longitudinally acquired 3D ultrasound scans from 1043 pregnancies from the Rotterdam Periconceptional Cohort, ranging from 7+0 to 12+6 weeks of gestational age. In 99.0% of images, PCA correctly extracted the principal axes of the embryo. The correct candidate was selected by the Pearson Heuristic, Atlas-based and Random Forest in 97.4%, 95.8%, and 98.4% of images, respectively. A Majority Vote of these selection methods resulted in an accuracy of 98.5%. The high accuracy of this pipeline enables consistent embryonic alignment in the first trimester, enabling scalable analysis in both clinical and research settings. The code is publicly available at: https://gitlab.com/radiology/prenatal-image-analysis/pca-3d-alignment.
>
---
#### [new 006] Disentangled Concepts Speak Louder Than Words:Explainable Video Action Recognition
- **分类: cs.CV**

- **简介: 该论文提出DANCE框架，用于可解释视频动作识别，解决现有方法无法区分运动与空间上下文的问题。通过解耦运动、物体、场景三类概念，结合大语言模型与概念瓶颈设计，提升解释清晰度与模型可调试性。**

- **链接: [http://arxiv.org/pdf/2511.03725v1](http://arxiv.org/pdf/2511.03725v1)**

> **作者:** Jongseo Lee; Wooil Lee; Gyeong-Moon Park; Seong Tae Kim; Jinwoo Choi
>
> **备注:** NeurIPS 2025 Spotlight paper. Project page: https://jong980812.github.io/DANCE/
>
> **摘要:** Effective explanations of video action recognition models should disentangle how movements unfold over time from the surrounding spatial context. However, existing methods based on saliency produce entangled explanations, making it unclear whether predictions rely on motion or spatial context. Language-based approaches offer structure but often fail to explain motions due to their tacit nature -- intuitively understood but difficult to verbalize. To address these challenges, we propose Disentangled Action aNd Context concept-based Explainable (DANCE) video action recognition, a framework that predicts actions through disentangled concept types: motion dynamics, objects, and scenes. We define motion dynamics concepts as human pose sequences. We employ a large language model to automatically extract object and scene concepts. Built on an ante-hoc concept bottleneck design, DANCE enforces prediction through these concepts. Experiments on four datasets -- KTH, Penn Action, HAA500, and UCF-101 -- demonstrate that DANCE significantly improves explanation clarity with competitive performance. We validate the superior interpretability of DANCE through a user study. Experimental results also show that DANCE is beneficial for model debugging, editing, and failure analysis.
>
---
#### [new 007] Signal Intensity-weighted coordinate channels improve learning stability and generalisation in 1D and 2D CNNs in localisation tasks on biomedical signals
- **分类: cs.CV**

- **简介: 该论文针对生物医学信号定位任务，提出一种信号强度加权的坐标通道方法，替代传统纯坐标通道，通过引入强度-位置耦合先验，提升1D/2D CNN在ECG时序定位和细胞核坐标回归中的收敛速度与泛化性能。**

- **链接: [http://arxiv.org/pdf/2511.03645v1](http://arxiv.org/pdf/2511.03645v1)**

> **作者:** Vittal L. Rao
>
> **摘要:** Localisation tasks in biomedical data often require models to learn meaningful spatial or temporal relationships from signals with complex intensity distributions. A common strategy, exemplified by CoordConv layers, is to append coordinate channels to convolutional inputs, enabling networks to learn absolute positions. In this work, we propose a signal intensity-weighted coordinate representation that replaces the pure coordinate channels with channels scaled by local signal intensity. This modification embeds an intensity-position coupling directly in the input representation, introducing a simple and modality-agnostic inductive bias. We evaluate the approach on two distinct localisation problems: (i) predicting the time of morphological transition in 20-second, two-lead ECG signals, and (ii) regressing the coordinates of nuclear centres in cytological images from the SiPaKMeD dataset. In both cases, the proposed representation yields faster convergence and higher generalisation performance relative to conventional coordinate-channel approaches, demonstrating its effectiveness across both one-dimensional and two-dimensional biomedical signals.
>
---
#### [new 008] Generalizing Shape-from-Template to Topological Changes
- **分类: cs.CV**

- **简介: 该论文拓展了Shape-from-Template（SfT）任务，解决其无法处理拓扑变化（如撕裂、切割）的问题。通过动态分区模板并优化能量函数，首次实现支持拓扑变化的鲁棒三维表面重建。**

- **链接: [http://arxiv.org/pdf/2511.03459v1](http://arxiv.org/pdf/2511.03459v1)**

> **作者:** Kevin Manogue; Tomasz M Schang; Dilara Kuş; Jonas Müller; Stefan Zachow; Agniva Sengupta
>
> **备注:** Accepted for publication at Smart Tools and Applications in Graphics (STAG), Genoa, Italy (2025)
>
> **摘要:** Reconstructing the surfaces of deformable objects from correspondences between a 3D template and a 2D image is well studied under Shape-from-Template (SfT) methods; however, existing approaches break down when topological changes accompany the deformation. We propose a principled extension of SfT that enables reconstruction in the presence of such changes. Our approach is initialized with a classical SfT solution and iteratively adapts the template by partitioning its spatial domain so as to minimize an energy functional that jointly encodes physical plausibility and reprojection consistency. We demonstrate that the method robustly captures a wide range of practically relevant topological events including tears and cuts on bounded 2D surfaces, thereby establishing the first general framework for topological-change-aware SfT. Experiments on both synthetic and real data confirm that our approach consistently outperforms baseline methods.
>
---
#### [new 009] SurgAnt-ViVQA: Learning to Anticipate Surgical Events through GRU-Driven Temporal Cross-Attention
- **分类: cs.CV**

- **简介: 论文提出SurgAnt-ViVQA模型，面向手术事件预测任务，解决传统VQA仅描述当前场景、缺乏前瞻能力的问题。构建首个手术前瞻数据集PitVQA-Anticipation，通过GRU时序交叉注意力实现未来阶段、器械与耗时的精准预测。**

- **链接: [http://arxiv.org/pdf/2511.03178v1](http://arxiv.org/pdf/2511.03178v1)**

> **作者:** Shreyas C. Dhake; Jiayuan Huang; Runlong He; Danyal Z. Khan; Evangelos B. Mazomenos; Sophia Bano; Hani J. Marcus; Danail Stoyanov; Matthew J. Clarkson; Mobarak I. Hoque
>
> **备注:** 12 pages
>
> **摘要:** Anticipating forthcoming surgical events is vital for real-time assistance in endonasal transsphenoidal pituitary surgery, where visibility is limited and workflow changes rapidly. Most visual question answering (VQA) systems reason on isolated frames with static vision language alignment, providing little support for forecasting next steps or instrument needs. Existing surgical VQA datasets likewise center on the current scene rather than the near future. We introduce PitVQA-Anticipation, the first VQA dataset designed for forward looking surgical reasoning. It comprises 33.5 hours of operative video and 734,769 question answer pairs built from temporally grouped clips and expert annotations across four tasks: predicting the future phase, next step, upcoming instrument, and remaining duration. We further propose SurgAnt-ViVQA, a video language model that adapts a large language model using a GRU Gated Temporal Cross-Attention module. A bidirectional GRU encodes frame to frame dynamics, while an adaptive gate injects visual context into the language stream at the token level. Parameter efficient fine tuning customizes the language backbone to the surgical domain. SurgAnt-ViVQA tested upon on PitVQA-Anticipation and EndoVis datasets, surpassing strong image and video based baselines. Ablations show that temporal recurrence and gated fusion drive most of the gains. A frame budget study indicates a trade-off: 8 frames maximize fluency, whereas 32 frames slightly reduce BLEU but improve numeric time estimation. By pairing a temporally aware encoder with fine grained gated cross-attention, SurgAnt-ViVQA advances surgical VQA from retrospective description to proactive anticipation. PitVQA-Anticipation offers a comprehensive benchmark for this setting and highlights the importance of targeted temporal modeling for reliable, future aware surgical assistance.
>
---
#### [new 010] Human Mesh Modeling for Anny Body
- **分类: cs.CV**

- **简介: 论文提出Anny，一种无需3D扫描、基于人体测量学的可微人体建模方法，通过语义参数（如年龄、性别）生成多样化的逼真人体形状，解决现有模型封闭、样本偏窄问题，并开源支持HMR与合成数据生成。**

- **链接: [http://arxiv.org/pdf/2511.03589v1](http://arxiv.org/pdf/2511.03589v1)**

> **作者:** Romain Brégier; Guénolé Fiche; Laura Bravo-Sánchez; Thomas Lucas; Matthieu Armando; Philippe Weinzaepfel; Grégory Rogez; Fabien Baradel
>
> **备注:** We release our model and code at https://github.com/naver/anny
>
> **摘要:** Parametric body models are central to many human-centric tasks, yet existing models often rely on costly 3D scans and learned shape spaces that are proprietary and demographically narrow. We introduce Anny, a simple, fully differentiable, and scan-free human body model grounded in anthropometric knowledge from the MakeHuman community. Anny defines a continuous, interpretable shape space, where phenotype parameters (e.g. gender, age, height, weight) control blendshapes spanning a wide range of human forms -- across ages (from infants to elders), body types, and proportions. Calibrated using WHO population statistics, it provides realistic and demographically grounded human shape variation within a single unified model. Thanks to its openness and semantic control, Anny serves as a versatile foundation for 3D human modeling -- supporting millimeter-accurate scan fitting, controlled synthetic data generation, and Human Mesh Recovery (HMR). We further introduce Anny-One, a collection of 800k photorealistic humans generated with Anny, showing that despite its simplicity, HMR models trained with Anny can match the performance of those trained with scan-based body models, while remaining interpretable and broadly representative. The Anny body model and its code are released under the Apache 2.0 license, making Anny an accessible foundation for human-centric 3D modeling.
>
---
#### [new 011] EvtSlowTV - A Large and Diverse Dataset for Event-Based Depth Estimation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出EvtSlowTV，首个大规模事件相机数据集，用于无监督深度估计。解决现有数据集规模小、场景受限问题，利用YouTube视频构建超13B事件数据，提升模型在复杂动态场景中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.02953v1](http://arxiv.org/pdf/2511.02953v1)**

> **作者:** Sadiq Layi Macaulay; Nimet Kaygusuz; Simon Hadfield
>
> **摘要:** Event cameras, with their high dynamic range (HDR) and low latency, offer a promising alternative for robust depth estimation in challenging environments. However, many event-based depth estimation approaches are constrained by small-scale annotated datasets, limiting their generalizability to real-world scenarios. To bridge this gap, we introduce EvtSlowTV, a large-scale event camera dataset curated from publicly available YouTube footage, which contains more than 13B events across various environmental conditions and motions, including seasonal hiking, flying, scenic driving, and underwater exploration. EvtSlowTV is an order of magnitude larger than existing event datasets, providing an unconstrained, naturalistic setting for event-based depth learning. This work shows the suitability of EvtSlowTV for a self-supervised learning framework to capitalise on the HDR potential of raw event streams. We further demonstrate that training with EvtSlowTV enhances the model's ability to generalise to complex scenes and motions. Our approach removes the need for frame-based annotations and preserves the asynchronous nature of event data.
>
---
#### [new 012] IEC3D-AD: A 3D Dataset of Industrial Equipment Components for Unsupervised Point Cloud Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文针对工业设备组件的无监督点云异常检测任务，构建了高精度真实场景数据集IEC3D-AD，并提出GMANet方法，通过几何形态生成与空间差异优化提升异常检测性能。**

- **链接: [http://arxiv.org/pdf/2511.03267v1](http://arxiv.org/pdf/2511.03267v1)**

> **作者:** Bingyang Guo; Hongjie Li; Ruiyun Yu; Hanzhe Liang; Jinbao Wang
>
> **摘要:** 3D anomaly detection (3D-AD) plays a critical role in industrial manufacturing, particularly in ensuring the reliability and safety of core equipment components. Although existing 3D datasets like Real3D-AD and MVTec 3D-AD offer broad application support, they fall short in capturing the complexities and subtle defects found in real industrial environments. This limitation hampers precise anomaly detection research, especially for industrial equipment components (IEC) such as bearings, rings, and bolts. To address this challenge, we have developed a point cloud anomaly detection dataset (IEC3D-AD) specific to real industrial scenarios. This dataset is directly collected from actual production lines, ensuring high fidelity and relevance. Compared to existing datasets, IEC3D-AD features significantly improved point cloud resolution and defect annotation granularity, facilitating more demanding anomaly detection tasks. Furthermore, inspired by generative 2D-AD methods, we introduce a novel 3D-AD paradigm (GMANet) on IEC3D-AD. This paradigm generates synthetic point cloud samples based on geometric morphological analysis, then reduces the margin and increases the overlap between normal and abnormal point-level features through spatial discrepancy optimization. Extensive experiments demonstrate the effectiveness of our method on both IEC3D-AD and other datasets.
>
---
#### [new 013] Hybrid Convolution and Vision Transformer NAS Search Space for TinyML Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文面向TinyML图像分类任务，旨在解决混合CNN-ViT架构参数大、计算成本高的问题，提出一种可搜索的混合CNN-ViT NAS空间，包含新型池化模块，在CIFAR10上实现了更小模型下更高的精度与推理速度。**

- **链接: [http://arxiv.org/pdf/2511.02992v1](http://arxiv.org/pdf/2511.02992v1)**

> **作者:** Mikhael Djajapermana; Moritz Reiber; Daniel Mueller-Gritschneder; Ulf Schlichtmann
>
> **备注:** Presented at ITEM workshop co-located with ECML PKDD 2024, Vilnius LT
>
> **摘要:** Hybrids of Convolutional Neural Network (CNN) and Vision Transformer (ViT) have outperformed pure CNN or ViT architecture. However, since these architectures require large parameters and incur large computational costs, they are unsuitable for tinyML deployment. This paper introduces a new hybrid CNN-ViT search space for Neural Architecture Search (NAS) to find efficient hybrid architectures for image classification. The search space covers hybrid CNN and ViT blocks to learn local and global information, as well as the novel Pooling block of searchable pooling layers for efficient feature map reduction. Experimental results on the CIFAR10 dataset show that our proposed search space can produce hybrid CNN-ViT architectures with superior accuracy and inference speed to ResNet-based tinyML models under tight model size constraints.
>
---
#### [new 014] Image-Intrinsic Priors for Integrated Circuit Defect Detection and Novel Class Discovery via Self-Supervised Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出IC DefectNCD框架，利用图像内在先验实现集成电路缺陷检测与未知缺陷发现，无需标注。通过自监督学习聚合正常特征、自适应分割缺陷区域，并引入注意力机制提升对新缺陷的分类能力。**

- **链接: [http://arxiv.org/pdf/2511.03120v1](http://arxiv.org/pdf/2511.03120v1)**

> **作者:** Botong. Zhao; Xubin. Wang; Shujing. Lyu; Yue. Lu
>
> **摘要:** Integrated circuit manufacturing is highly complex, comprising hundreds of process steps. Defects can arise at any stage, causing yield loss and ultimately degrading product reliability. Supervised methods require extensive human annotation and struggle with emergent categories and rare, data scarce defects. Clustering-based unsupervised methods often exhibit unstable performance due to missing priors. We propose IC DefectNCD, a support set free framework that leverages Image Intrinsic Priors in IC SEM images for defect detection and novel class discovery. We first develop Self Normal Information Guided IC Defect Detection, aggregating representative normal features via a learnable normal information extractor and using reconstruction residuals to coarsely localize defect regions. To handle saliency variations across defects, we introduce an adaptive binarization strategy that produces stable subimages focused on core defective areas. Finally, we design Self Defect Information Guided IC Defect Classification, which incorporates a soft mask guided attention mechanism to inject spatial defect priors into the teacher student model. This enhances sensitivity to defective regions, suppresses background interference, and enables recognition and classification of unseen defects. We validate the approach on a real world dataset spanning three key fabrication stages and covering 15 defect types. Experiments demonstrate robust performance on both defect detection and unseen defect classification.
>
---
#### [new 015] Subsampled Randomized Fourier GaLore for Adapting Foundation Models in Depth-Driven Liver Landmark Segmentation
- **分类: cs.CV**

- **简介: 该论文针对腹腔镜肝手术中深度感知受限导致的解剖标志分割难题，提出融合RGB与深度信息的双编码器框架，创新引入SRFT-GaLore高效微调大模型，显著提升分割精度与跨数据集泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.03163v1](http://arxiv.org/pdf/2511.03163v1)**

> **作者:** Yun-Chen Lin; Jiayuan Huang; Hanyuan Zhang; Sergi Kavtaradze; Matthew J. Clarkson; Mobarak I. Hoque
>
> **备注:** 12 pages
>
> **摘要:** Accurate detection and delineation of anatomical structures in medical imaging are critical for computer-assisted interventions, particularly in laparoscopic liver surgery where 2D video streams limit depth perception and complicate landmark localization. While recent works have leveraged monocular depth cues for enhanced landmark detection, challenges remain in fusing RGB and depth features and in efficiently adapting large-scale vision models to surgical domains. We propose a depth-guided liver landmark segmentation framework integrating semantic and geometric cues via vision foundation encoders. We employ Segment Anything Model V2 (SAM2) encoder to extract RGB features and Depth Anything V2 (DA2) encoder to extract depth-aware features. To efficiently adapt SAM2, we introduce SRFT-GaLore, a novel low-rank gradient projection method that replaces the computationally expensive SVD with a Subsampled Randomized Fourier Transform (SRFT). This enables efficient fine-tuning of high-dimensional attention layers without sacrificing representational power. A cross-attention fusion module further integrates RGB and depth cues. To assess cross-dataset generalization, we also construct a new Laparoscopic Liver Surgical Dataset (LLSD) as an external validation benchmark. On the public L3D dataset, our method achieves a 4.85% improvement in Dice Similarity Coefficient and a 11.78-point reduction in Average Symmetric Surface Distance compared to the D2GPLand. To further assess generalization capability, we evaluate our model on LLSD dataset. Our model maintains competitive performance and significantly outperforms SAM-based baselines, demonstrating strong cross-dataset robustness and adaptability to unseen surgical environments. These results demonstrate that our SRFT-GaLore-enhanced dual-encoder framework enables scalable and precise segmentation under real-time, depth-constrained surgical settings.
>
---
#### [new 016] Decoupled Multi-Predictor Optimization for Inference-Efficient Model Tuning
- **分类: cs.CV**

- **简介: 该论文针对高效推理的模型微调任务，提出解耦多预测器优化（DMPO），解决早期层难以同时提供基础特征与判别特征的问题。通过轻量旁路模块与统计预测器架构，及两阶段解耦训练，实现计算成本降低下的性能提升。**

- **链接: [http://arxiv.org/pdf/2511.03245v1](http://arxiv.org/pdf/2511.03245v1)**

> **作者:** Liwei Luo; Shuaitengyuan Li; Dongwei Ren; Qilong Wang; Pengfei Zhu; Qinghua Hu
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Recently, remarkable progress has been made in large-scale pre-trained model tuning, and inference efficiency is becoming more crucial for practical deployment. Early exiting in conjunction with multi-stage predictors, when cooperated with a parameter-efficient fine-tuning strategy, offers a straightforward way to achieve an inference-efficient model. However, a key challenge remains unresolved: How can early stages provide low-level fundamental features to deep stages while simultaneously supplying high-level discriminative features to early-stage predictors? To address this problem, we propose a Decoupled Multi-Predictor Optimization (DMPO) method to effectively decouple the low-level representative ability and high-level discriminative ability in early stages. First, in terms of architecture, we introduce a lightweight bypass module into multi-stage predictors for functional decomposition of shallow features from early stages, while a high-order statistics-based predictor is developed for early stages to effectively enhance their discriminative ability. To reasonably train our multi-predictor architecture, a decoupled optimization is proposed to allocate two-phase loss weights for multi-stage predictors during model tuning, where the initial training phase enables the model to prioritize the acquisition of discriminative ability of deep stages via emphasizing representative ability of early stages, and the latter training phase drives discriminative ability towards earlier stages as much as possible. As such, our DMPO can effectively decouple representative and discriminative abilities in early stages in terms of architecture design and model optimization. Experiments across various datasets and pre-trained backbones demonstrate that DMPO clearly outperforms its counterparts when reducing computational cost.
>
---
#### [new 017] Multi-Object Tracking Retrieval with LLaVA-Video: A Training-Free Solution to MOT25-StAG Challenge
- **分类: cs.CV**

- **简介: 该论文针对MOT25-StAG挑战，提出一种无训练的多目标跟踪检索方法，结合FastTracker与LLaVA-Video，将任务建模为视频检索，实现对自由文本查询的时空目标定位与跟踪，取得亚军成绩。**

- **链接: [http://arxiv.org/pdf/2511.03332v1](http://arxiv.org/pdf/2511.03332v1)**

> **作者:** Yi Yang; Yiming Xu; Timo Kaiser; Hao Cheng; Bodo Rosenhahn; Michael Ying Yang
>
> **摘要:** In this report, we present our solution to the MOT25-Spatiotemporal Action Grounding (MOT25-StAG) Challenge. The aim of this challenge is to accurately localize and track multiple objects that match specific and free-form language queries, using video data of complex real-world scenes as input. We model the underlying task as a video retrieval problem and present a two-stage, zero-shot approach, combining the advantages of the SOTA tracking model FastTracker and Multi-modal Large Language Model LLaVA-Video. On the MOT25-StAG test set, our method achieves m-HIoU and HOTA scores of 20.68 and 10.73 respectively, which won second place in the challenge.
>
---
#### [new 018] Deploying Rapid Damage Assessments from sUAS Imagery for Disaster Response
- **分类: cs.CV; cs.AI; cs.CY**

- **简介: 该论文提出首个用于sUAS影像的AI/ML建筑损毁评估系统，解决灾后影像数据过载导致响应延迟问题，基于2.1万+标注图像训练模型，并在飓风响应中成功部署，实现18分钟评估415栋建筑。**

- **链接: [http://arxiv.org/pdf/2511.03132v1](http://arxiv.org/pdf/2511.03132v1)**

> **作者:** Thomas Manzini; Priyankari Perali; Robin R. Murphy
>
> **备注:** 6 pages, 4 figures, 1 table. Accepted - In Press, IAAI'26
>
> **摘要:** This paper presents the first AI/ML system for automating building damage assessment in uncrewed aerial systems (sUAS) imagery to be deployed operationally during federally declared disasters (Hurricanes Debby and Helene). In response to major disasters, sUAS teams are dispatched to collect imagery of the affected areas to assess damage; however, at recent disasters, teams collectively delivered between 47GB and 369GB of imagery per day, representing more imagery than can reasonably be transmitted or interpreted by subject matter experts in the disaster scene, thus delaying response efforts. To alleviate this data avalanche encountered in practice, computer vision and machine learning techniques are necessary. While prior work has been deployed to automatically assess damage in satellite imagery, there is no current state of practice for sUAS-based damage assessment systems, as all known work has been confined to academic settings. This work establishes the state of practice via the development and deployment of models for building damage assessment with sUAS imagery. The model development involved training on the largest known dataset of post-disaster sUAS aerial imagery, containing 21,716 building damage labels, and the operational training of 91 disaster practitioners. The best performing model was deployed during the responses to Hurricanes Debby and Helene, where it assessed a combined 415 buildings in approximately 18 minutes. This work contributes documentation of the actual use of AI/ML for damage assessment during a disaster and lessons learned to the benefit of the AI/ML research and user communities.
>
---
#### [new 019] QG-CoC: Question-Guided Chain-of-Captions for Large Multimodal Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对多图像场景下多模态大模型感知细粒度信息与推理能力不足的问题，提出QG-CoC——一种问题引导的连贯描述提示方法，提升模型在多图像任务中的跨图感知与信息合成能力，实验验证其在零样本设置下的优越性。**

- **链接: [http://arxiv.org/pdf/2511.03206v1](http://arxiv.org/pdf/2511.03206v1)**

> **作者:** Kuei-Chun Kao; Hsu Tzu-Yin; Yunqi Hong; Ruochen Wang; Cho-Jui Hsieh
>
> **备注:** 16 pages
>
> **摘要:** Recently, Multimodal Large Language Models (MLLMs) encounter two key issues in multi-image contexts: (1) a lack of fine-grained perception across disparate images, and (2) a diminished capability to effectively reason over and synthesize information from multiple visual inputs. However, while various prompting methods aim to describe visual content, many existing studies focus primarily on single-image settings or specific, constrained scenarios. This leaves a critical gap in understanding and addressing how MLLMs tackle more general and complex multi-image reasoning tasks. Thus, we first extensively investigate how current prompting methods perceive fine-grained visual details and process visual information when dealing with multiple images. Our findings reveal that existing prompting methods fall short in attending to needed clues and seamlessly integrating perception and reasoning. Inspired by the findings, we propose a new zero-shot prompting method, Question-Guided Chain-of-Captions (QG-CoC), a generalized prompting approach that effectively handles problems with an arbitrary number of images. We evaluate our method on various open-source and closed-source MLLMs for multi-image and single-image benchmarks. Experimental results indicate that QG-CoC demonstrates competitive performance across tasks and exhibits robust improvements in the challenging scenarios where existing prompting methods fail.
>
---
#### [new 020] Diffusion-SDPO: Safeguarded Direct Preference Optimization for Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对文本到图像扩散模型的偏好对齐问题，发现标准DPO会恶化优劣样本的重建质量，提出Diffusion-SDPO，通过自适应缩放劣样本梯度，保障优样本质量不降，提升对齐效果。**

- **链接: [http://arxiv.org/pdf/2511.03317v1](http://arxiv.org/pdf/2511.03317v1)**

> **作者:** Minghao Fu; Guo-Hua Wang; Tianyu Cui; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** The code is publicly available at https://github.com/AIDC-AI/Diffusion-SDPO
>
> **摘要:** Text-to-image diffusion models deliver high-quality images, yet aligning them with human preferences remains challenging. We revisit diffusion-based Direct Preference Optimization (DPO) for these models and identify a critical pathology: enlarging the preference margin does not necessarily improve generation quality. In particular, the standard Diffusion-DPO objective can increase the reconstruction error of both winner and loser branches. Consequently, degradation of the less-preferred outputs can become sufficiently severe that the preferred branch is also adversely affected even as the margin grows. To address this, we introduce Diffusion-SDPO, a safeguarded update rule that preserves the winner by adaptively scaling the loser gradient according to its alignment with the winner gradient. A first-order analysis yields a closed-form scaling coefficient that guarantees the error of the preferred output is non-increasing at each optimization step. Our method is simple, model-agnostic, broadly compatible with existing DPO-style alignment frameworks and adds only marginal computational overhead. Across standard text-to-image benchmarks, Diffusion-SDPO delivers consistent gains over preference-learning baselines on automated preference, aesthetic, and prompt alignment metrics. Code is publicly available at https://github.com/AIDC-AI/Diffusion-SDPO.
>
---
#### [new 021] Generative deep learning for foundational video translation in ultrasound
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种生成式深度学习方法，实现超声CFD与灰度视频间的跨模态翻译，解决数据不平衡问题。模型在数万视频上训练，生成视频在视觉质量与临床任务性能上接近真实数据，具备跨领域泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.03255v1](http://arxiv.org/pdf/2511.03255v1)**

> **作者:** Nikolina Tomic Roshni Bhatnagar; Sarthak Jain; Connor Lau; Tien-Yu Liu; Laura Gambini; Rima Arnaout
>
> **摘要:** Deep learning (DL) has the potential to revolutionize image acquisition and interpretation across medicine, however, attention to data imbalance and missingness is required. Ultrasound data presents a particular challenge because in addition to different views and structures, it includes several sub-modalities-such as greyscale and color flow doppler (CFD)-that are often imbalanced in clinical studies. Image translation can help balance datasets but is challenging for ultrasound sub-modalities to date. Here, we present a generative method for ultrasound CFD-greyscale video translation, trained on 54,975 videos and tested on 8,368. The method developed leveraged pixel-wise, adversarial, and perceptual loses and utilized two networks: one for reconstructing anatomic structures and one for denoising to achieve realistic ultrasound imaging. Average pairwise SSIM between synthetic videos and ground truth was 0.91+/-0.04. Synthetic videos performed indistinguishably from real ones in DL classification and segmentation tasks and when evaluated by blinded clinical experts: F1 score was 0.9 for real and 0.89 for synthetic videos; Dice score between real and synthetic segmentation was 0.97. Overall clinician accuracy in distinguishing real vs synthetic videos was 54+/-6% (42-61%), indicating realistic synthetic videos. Although trained only on heart videos, the model worked well on ultrasound spanning several clinical domains (average SSIM 0.91+/-0.05), demonstrating foundational abilities. Together, these data expand the utility of retrospectively collected imaging and augment the dataset design toolbox for medical imaging.
>
---
#### [new 022] Finetuning-Free Personalization of Text to Image Generation via Hypernetworks
- **分类: cs.CV**

- **简介: 该论文提出一种无微调的文本到图像个性化方法，利用超网络直接从主体图像预测LoRA权重，结合HM-CFG提升泛化性，实现高效、高保真个性化生成，解决传统方法依赖微调、计算开销大的问题。**

- **链接: [http://arxiv.org/pdf/2511.03156v1](http://arxiv.org/pdf/2511.03156v1)**

> **作者:** Sagar Shrestha; Gopal Sharma; Luowei Zhou; Suren Kumar
>
> **摘要:** Personalizing text-to-image diffusion models has traditionally relied on subject-specific fine-tuning approaches such as DreamBooth~\cite{ruiz2023dreambooth}, which are computationally expensive and slow at inference. Recent adapter- and encoder-based methods attempt to reduce this overhead but still depend on additional fine-tuning or large backbone models for satisfactory results. In this work, we revisit an orthogonal direction: fine-tuning-free personalization via Hypernetworks that predict LoRA-adapted weights directly from subject images. Prior hypernetwork-based approaches, however, suffer from costly data generation or unstable attempts to mimic base model optimization trajectories. We address these limitations with an end-to-end training objective, stabilized by a simple output regularization, yielding reliable and effective hypernetworks. Our method removes the need for per-subject optimization at test time while preserving both subject fidelity and prompt alignment. To further enhance compositional generalization at inference time, we introduce Hybrid-Model Classifier-Free Guidance (HM-CFG), which combines the compositional strengths of the base diffusion model with the subject fidelity of personalized models during sampling. Extensive experiments on CelebA-HQ, AFHQ-v2, and DreamBench demonstrate that our approach achieves strong personalization performance and highlights the promise of hypernetworks as a scalable and effective direction for open-category personalization.
>
---
#### [new 023] Enhancing Medical Image Segmentation via Heat Conduction Equation
- **分类: cs.CV**

- **简介: 该论文针对医学图像分割中全局上下文建模效率低的问题，提出融合Mamba与热传导方程的混合模型，通过热扩散算子增强语义抽象，在CT/MRI数据上显著超越基线方法。**

- **链接: [http://arxiv.org/pdf/2511.03260v1](http://arxiv.org/pdf/2511.03260v1)**

> **作者:** Rong Wu; Yim-Sang Yu
>
> **摘要:** Medical image segmentation has been significantly advanced by deep learning architectures, notably U-Net variants. However, existing models struggle to achieve efficient global context modeling and long-range dependency reasoning under practical computational budgets simultaneously. In this work, we propose a novel hybrid architecture utilizing U-Mamba with Heat Conduction Equation. Our model combines Mamba-based state-space modules for efficient long-range reasoning with Heat Conduction Operators (HCOs) in the bottleneck layers, simulating frequency-domain thermal diffusion for enhanced semantic abstraction. Experimental results on multimodal abdominal CT and MRI datasets demonstrate that the proposed model consistently outperforms strong baselines, validating its effectiveness and generalizability. It suggest that blending state-space dynamics with heat-based global diffusion offers a scalable and interpretable solution for medical segmentation tasks.
>
---
#### [new 024] DentalSplat: Dental Occlusion Novel View Synthesis from Sparse Intra-Oral Photographs
- **分类: cs.CV**

- **简介: 论文提出DentalSplat，解决从极稀疏（仅3张）口腔照片中重建牙合三维模型并合成新视角的难题，通过先验引导初始化与光流约束优化3D高斯泼溅，显著提升重建质量与效率。**

- **链接: [http://arxiv.org/pdf/2511.03099v1](http://arxiv.org/pdf/2511.03099v1)**

> **作者:** Yiyi Miao; Taoyu Wu; Tong Chen; Sihao Li; Ji Jiang; Youpeng Yang; Angelos Stefanidis; Limin Yu; Jionglong Su
>
> **摘要:** In orthodontic treatment, particularly within telemedicine contexts, observing patients' dental occlusion from multiple viewpoints facilitates timely clinical decision-making. Recent advances in 3D Gaussian Splatting (3DGS) have shown strong potential in 3D reconstruction and novel view synthesis. However, conventional 3DGS pipelines typically rely on densely captured multi-view inputs and precisely initialized camera poses, limiting their practicality. Orthodontic cases, in contrast, often comprise only three sparse images, specifically, the anterior view and bilateral buccal views, rendering the reconstruction task especially challenging. The extreme sparsity of input views severely degrades reconstruction quality, while the absence of camera pose information further complicates the process. To overcome these limitations, we propose DentalSplat, an effective framework for 3D reconstruction from sparse orthodontic imagery. Our method leverages a prior-guided dense stereo reconstruction model to initialize the point cloud, followed by a scale-adaptive pruning strategy to improve the training efficiency and reconstruction quality of 3DGS. In scenarios with extremely sparse viewpoints, we further incorporate optical flow as a geometric constraint, coupled with gradient regularization, to enhance rendering fidelity. We validate our approach on a large-scale dataset comprising 950 clinical cases and an additional video-based test set of 195 cases designed to simulate real-world remote orthodontic imaging conditions. Experimental results demonstrate that our method effectively handles sparse input scenarios and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art techniques.
>
---
#### [new 025] Generative Hints
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出生成提示（Generative Hints），用于视觉分类任务，解决数据增强无法充分学习输入空间不变性的问题。通过生成虚拟样本并引入hint约束，实现半监督训练，显著提升模型泛化能力与准确率。**

- **链接: [http://arxiv.org/pdf/2511.02933v1](http://arxiv.org/pdf/2511.02933v1)**

> **作者:** Andy Dimnaku; Abdullah Yusuf Kavranoğlu; Yaser Abu-Mostafa
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Data augmentation is widely used in vision to introduce variation and mitigate overfitting, through enabling models to learn invariant properties, such as spatial invariance. However, these properties are not fully captured by data augmentation alone, since it attempts to learn the property on transformations of the training data only. We propose generative hints, a training methodology that directly enforces known invariances in the entire input space. Our approach leverages a generative model trained on the training set to approximate the input distribution and generate unlabeled images, which we refer to as virtual examples. These virtual examples are used to enforce functional properties known as hints. In generative hints, although the training dataset is fully labeled, the model is trained in a semi-supervised manner on both the classification and hint objectives, using the unlabeled virtual examples to guide the model in learning the desired hint. Across datasets, architectures, and loss functions, generative hints consistently outperform standard data augmentation when learning the same property. On popular fine-grained visual classification benchmarks, we achieved up to 1.78% top-1 accuracy improvement (0.63% on average) over fine-tuned models with data augmentation and an average performance boost of 1.286% on the CheXpert X-ray dataset.
>
---
#### [new 026] ISC-Perception: A Hybrid Computer Vision Dataset for Object Detection in Novel Steel Assembly
- **分类: cs.CV; eess.IV**

- **简介: 论文提出ISC-Perception，首个用于钢连接件检测的混合视觉数据集，解决施工场景数据稀缺问题，融合合成与真实图像，自动标注，大幅降低人力成本，显著提升检测精度。**

- **链接: [http://arxiv.org/pdf/2511.03098v1](http://arxiv.org/pdf/2511.03098v1)**

> **作者:** Miftahur Rahman; Samuel Adebayo; Dorian A. Acevedo-Mejia; David Hester; Daniel McPolin; Karen Rafferty; Debra F. Laefer
>
> **摘要:** The Intermeshed Steel Connection (ISC) system, when paired with robotic manipulators, can accelerate steel-frame assembly and improve worker safety by eliminating manual assembly. Dependable perception is one of the initial stages for ISC-aware robots. However, this is hampered by the absence of a dedicated image corpus, as collecting photographs on active construction sites is logistically difficult and raises safety and privacy concerns. In response, we introduce ISC-Perception, the first hybrid dataset expressly designed for ISC component detection. It blends procedurally rendered CAD images, game-engine photorealistic scenes, and a limited, curated set of real photographs, enabling fully automatic labelling of the synthetic portion. We explicitly account for all human effort to produce the dataset, including simulation engine and scene setup, asset preparation, post-processing scripts and quality checks; our total human time to generate a 10,000-image dataset was 30.5,h versus 166.7,h for manual labelling at 60,s per image (-81.7%). A manual pilot on a representative image with five instances of ISC members took 60,s (maximum 80,s), anchoring the manual baseline. Detectors trained on ISC-Perception achieved a mean Average Precision at IoU 0.50 of 0.756, substantially surpassing models trained on synthetic-only or photorealistic-only data. On a 1,200-frame bench test, we report mAP@0.50/mAP@[0.50:0.95] of 0.943/0.823. By bridging the data gap for construction-robotics perception, ISC-Perception facilitates rapid development of custom object detectors and is freely available for research and industrial use upon request.
>
---
#### [new 027] Decoupling Augmentation Bias in Prompt Learning for Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文面向视觉-语言模型的零样本学习任务，解决提示学习中图像增强引入的伪相关偏差问题，提出AAPL方法，通过对抗token解耦增强噪声与语义特征，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.03367v1](http://arxiv.org/pdf/2511.03367v1)**

> **作者:** Gahyeon Kim; Sohee Kim; Seokju Lee
>
> **备注:** Accepted in Pattern Recognition
>
> **摘要:** Recent advances in large-scale vision and language models have led to significant progress in zero-shot learning tasks. Methods such as CoOp and CoCoOp have shown that replacing handcrafted prompts with learnable vectors, known as prompt learning, can result in improved performance. However, these models often struggle to generalize to entirely unseen categories. While traditional zero-shot learning techniques benefit from various data augmentation strategies, prompt learning has primarily focused on text-based modifications, leaving the potential of image-based augmentation largely unexplored. In this work, we explore how image-level augmentations, particularly those that introduce attribute-specific variations, can support and enhance prompt learning. Our analysis examines the interaction between these augmentations and soft prompt frameworks, revealing their potential to improve generalization. We also identify a limitation in existing methods, such as CoCoOp, which do not provide explicit guidance for learning prompts that focus on semantically meaningful visual features. To address this, we propose Adding Attributes to Prompt Learning, AAPL, a novel method that introduces adversarial token embeddings to decouple superficial visual variations introduced by augmentation from class-relevant semantic representations. This decoupling enables the learned prompts to concentrate on visually discriminative features that align with the target categories. We conduct comprehensive experiments on eleven benchmark datasets, and AAPL consistently outperforms existing methods across few-shot, zero-shot, cross-dataset, and domain generalization settings. Our source code is publicly available at: https://github.com/Gahyeonkim09/AAPL
>
---
#### [new 028] A Lightweight 3D-CNN for Event-Based Human Action Recognition with Privacy-Preserving Potential
- **分类: cs.CV**

- **简介: 该论文提出一种轻量级3D-CNN，用于基于事件相机的人体动作识别，解决传统摄像头隐私泄露问题。通过高效时空建模、focal loss与数据增强，在隐私保护前提下实现94.17%准确率，优于主流模型。**

- **链接: [http://arxiv.org/pdf/2511.03665v1](http://arxiv.org/pdf/2511.03665v1)**

> **作者:** Mehdi Sefidgar Dilmaghani; Francis Fowley; Peter Corcoran
>
> **摘要:** This paper presents a lightweight three-dimensional convolutional neural network (3DCNN) for human activity recognition (HAR) using event-based vision data. Privacy preservation is a key challenge in human monitoring systems, as conventional frame-based cameras capture identifiable personal information. In contrast, event cameras record only changes in pixel intensity, providing an inherently privacy-preserving sensing modality. The proposed network effectively models both spatial and temporal dynamics while maintaining a compact design suitable for edge deployment. To address class imbalance and enhance generalization, focal loss with class reweighting and targeted data augmentation strategies are employed. The model is trained and evaluated on a composite dataset derived from the Toyota Smart Home and ETRI datasets. Experimental results demonstrate an F1-score of 0.9415 and an overall accuracy of 94.17%, outperforming benchmark 3D-CNN architectures such as C3D, ResNet3D, and MC3_18 by up to 3%. These results highlight the potential of event-based deep learning for developing accurate, efficient, and privacy-aware human action recognition systems suitable for real-world edge applications.
>
---
#### [new 029] Diffusion-Guided Mask-Consistent Paired Mixing for Endoscopic Image Segmentation
- **分类: cs.CV**

- **简介: 该论文针对内镜图像分割任务，解决混合增强中掩码错位与合成域偏移问题，提出扩散引导的掩码一致配对混合方法（MCPMix）与自适应锚定策略（RLA），在保留几何结构的同时增强外观多样性，显著提升分割性能。**

- **链接: [http://arxiv.org/pdf/2511.03219v1](http://arxiv.org/pdf/2511.03219v1)**

> **作者:** Pengyu Jie; Wanquan Liu; Rui He; Yihui Wen; Deyu Meng; Chenqiang Gao
>
> **摘要:** Augmentation for dense prediction typically relies on either sample mixing or generative synthesis. Mixing improves robustness but misaligned masks yield soft label ambiguity. Diffusion synthesis increases apparent diversity but, when trained as common samples, overlooks the structural benefit of mask conditioning and introduces synthetic-real domain shift. We propose a paired, diffusion-guided paradigm that fuses the strengths of both. For each real image, a synthetic counterpart is generated under the same mask and the pair is used as a controllable input for Mask-Consistent Paired Mixing (MCPMix), which mixes only image appearance while supervision always uses the original hard mask. This produces a continuous family of intermediate samples that smoothly bridges synthetic and real appearances under shared geometry, enlarging diversity without compromising pixel-level semantics. To keep learning aligned with real data, Real-Anchored Learnable Annealing (RLA) adaptively adjusts the mixing strength and the loss weight of mixed samples over training, gradually re-anchoring optimization to real data and mitigating distributional bias. Across Kvasir-SEG, PICCOLO, CVC-ClinicDB, a private NPC-LES cohort, and ISIC 2017, the approach achieves state-of-the-art segmentation performance and consistent gains over baselines. The results show that combining label-preserving mixing with diffusion-driven diversity, together with adaptive re-anchoring, yields robust and generalizable endoscopic segmentation.
>
---
#### [new 030] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions
- **分类: cs.CV**

- **简介: UniAVGen提出统一音频视频生成框架，解决跨模态同步与语义一致性差的问题，通过非对称跨模态交互与人脸感知调制，实现高效联合生成，仅用1.3M样本即超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.03334v1](http://arxiv.org/pdf/2511.03334v1)**

> **作者:** Guozhen Zhang; Zixiang Zhou; Teng Hu; Ziqiao Peng; Youliang Zhang; Yi Chen; Yuan Zhou; Qinglin Lu; Limin Wang
>
> **摘要:** Due to the lack of effective cross-modal modeling, existing open-source audio-video generation methods often exhibit compromised lip synchronization and insufficient semantic consistency. To mitigate these drawbacks, we propose UniAVGen, a unified framework for joint audio and video generation. UniAVGen is anchored in a dual-branch joint synthesis architecture, incorporating two parallel Diffusion Transformers (DiTs) to build a cohesive cross-modal latent space. At its heart lies an Asymmetric Cross-Modal Interaction mechanism, which enables bidirectional, temporally aligned cross-attention, thus ensuring precise spatiotemporal synchronization and semantic consistency. Furthermore, this cross-modal interaction is augmented by a Face-Aware Modulation module, which dynamically prioritizes salient regions in the interaction process. To enhance generative fidelity during inference, we additionally introduce Modality-Aware Classifier-Free Guidance, a novel strategy that explicitly amplifies cross-modal correlation signals. Notably, UniAVGen's robust joint synthesis design enables seamless unification of pivotal audio-video tasks within a single model, such as joint audio-video generation and continuation, video-to-audio dubbing, and audio-driven video synthesis. Comprehensive experiments validate that, with far fewer training samples (1.3M vs. 30.1M), UniAVGen delivers overall advantages in audio-video synchronization, timbre consistency, and emotion consistency.
>
---
#### [new 031] Unified Long Video Inpainting and Outpainting via Overlapping High-Order Co-Denoising
- **分类: cs.CV**

- **简介: 该论文提出一种统一的长视频修复与外推方法，基于扩散模型，通过LoRA高效微调与重叠高阶去噪策略，实现无拼接伪影的超长视频编辑，显著提升时序一致性与生成质量。**

- **链接: [http://arxiv.org/pdf/2511.03272v1](http://arxiv.org/pdf/2511.03272v1)**

> **作者:** Shuangquan Lyu; Steven Mao; Yue Ma
>
> **摘要:** Generating long videos remains a fundamental challenge, and achieving high controllability in video inpainting and outpainting is particularly demanding. To address both of these challenges simultaneously and achieve controllable video inpainting and outpainting for long video clips, we introduce a novel and unified approach for long video inpainting and outpainting that extends text-to-video diffusion models to generate arbitrarily long, spatially edited videos with high fidelity. Our method leverages LoRA to efficiently fine-tune a large pre-trained video diffusion model like Alibaba's Wan 2.1 for masked region video synthesis, and employs an overlap-and-blend temporal co-denoising strategy with high-order solvers to maintain consistency across long sequences. In contrast to prior work that struggles with fixed-length clips or exhibits stitching artifacts, our system enables arbitrarily long video generation and editing without noticeable seams or drift. We validate our approach on challenging inpainting/outpainting tasks including editing or adding objects over hundreds of frames and demonstrate superior performance to baseline methods like Wan 2.1 model and VACE in terms of quality (PSNR/SSIM), and perceptual realism (LPIPS). Our method enables practical long-range video editing with minimal overhead, achieved a balance between parameter efficient and superior performance.
>
---
#### [new 032] A Foundation Model for Brain MRI with Dynamic Modality Integration
- **分类: cs.CV**

- **简介: 该论文提出一种脑部MRI基础模型，通过可学习模态嵌入与条件归一化，实现多模态动态整合与缺失模态自适应重建，解决传统模型需独立训练各模态的问题，支持自监督学习与跨中心泛化。**

- **链接: [http://arxiv.org/pdf/2511.03014v1](http://arxiv.org/pdf/2511.03014v1)**

> **作者:** Minh Sao Khue Luu; Bair N. Tuchinov
>
> **备注:** Preliminary work; results ongoing
>
> **摘要:** We present a foundation model for brain MRI that can work with different combinations of imaging sequences. The model uses one encoder with learnable modality embeddings, conditional layer normalization, and a masked autoencoding objective that accounts for missing modalities. A variance-covariance regularizer is applied to stabilize feature learning and improve representation diversity. This design removes the need for separate models for each modality and allows the network to adapt when some sequences are missing or unseen. It is trained on about 60,000 multi-center MRIs using self-supervised reconstruction and modality imputation to learn flexible representations. A learnable modality embedding guides feature extraction so the encoder can adjust to different inputs. We describe our planned evaluation on brain tumor and multiple sclerosis segmentation, as well as lesion classification, under various modality settings. Preliminary results show that the method works feasibly, and further experiments are planned to study its performance in more detail. All code and pretrained models are available at https://github.com/BrainFM/brainfm
>
---
#### [new 033] A Plug-and-Play Framework for Volumetric Light-Sheet Image Reconstruction
- **分类: cs.CV; cs.NA; math.NA**

- **简介: 该论文提出一种即插即用的压缩感知框架，用于光片显微镜的三维心肌动态成像，解决时空分辨率权衡问题。通过DMD编码与ADMM优化，融合多类去噪器及时间正则化，实现高压缩比下清晰、低光毒性的细胞结构重建。**

- **链接: [http://arxiv.org/pdf/2511.03093v1](http://arxiv.org/pdf/2511.03093v1)**

> **作者:** Yi Gong; Xinyuan Zhang; Jichen Chai; Yichen Ding; Yifei Lou
>
> **摘要:** Cardiac contraction is a rapid, coordinated process that unfolds across three-dimensional tissue on millisecond timescales. Traditional optical imaging is often inadequate for capturing dynamic cellular structure in the beating heart because of a fundamental trade-off between spatial and temporal resolution. To overcome these limitations, we propose a high-performance computational imaging framework that integrates Compressive Sensing (CS) with Light-Sheet Microscopy (LSM) for efficient, low-phototoxic cardiac imaging. The system performs compressed acquisition of fluorescence signals via random binary mask coding using a Digital Micromirror Device (DMD). We propose a Plug-and-Play (PnP) framework, solved using the alternating direction method of multipliers (ADMM), which flexibly incorporates advanced denoisers, including Tikhonov, Total Variation (TV), and BM3D. To preserve structural continuity in dynamic imaging, we further introduce temporal regularization enforcing smoothness between adjacent z-slices. Experimental results on zebrafish heart imaging under high compression ratios demonstrate that the proposed method successfully reconstructs cellular structures with excellent denoising performance and image clarity, validating the effectiveness and robustness of our algorithm in real-world high-speed, low-light biological imaging scenarios.
>
---
#### [new 034] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection
- **分类: cs.CV**

- **简介: 该论文面向细粒度社交互动检测任务，解决现有方法忽视个体局部线索与交互建模的问题。提出部分感知的自底向上分组推理框架，利用身体部位特征与人际关系精准推断社交群体。**

- **链接: [http://arxiv.org/pdf/2511.03666v1](http://arxiv.org/pdf/2511.03666v1)**

> **作者:** Dongkeun Kim; Minsu Cho; Suha Kwak
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Social interactions often emerge from subtle, fine-grained cues such as facial expressions, gaze, and gestures. However, existing methods for social interaction detection overlook such nuanced cues and primarily rely on holistic representations of individuals. Moreover, they directly detect social groups without explicitly modeling the underlying interactions between individuals. These drawbacks limit their ability to capture localized social signals and introduce ambiguity when group configurations should be inferred from social interactions grounded in nuanced cues. In this work, we propose a part-aware bottom-up group reasoning framework for fine-grained social interaction detection. The proposed method infers social groups and their interactions using body part features and their interpersonal relations. Our model first detects individuals and enhances their features using part-aware cues, and then infers group configuration by associating individuals via similarity-based reasoning, which considers not only spatial relations but also subtle social cues that signal interactions, leading to more accurate group inference. Experiments on the NVI dataset demonstrate that our method outperforms prior methods, achieving the new state of the art.
>
---
#### [new 035] OmniVLA: Unifiying Multi-Sensor Perception for Physically-Grounded Multimodal VLA
- **分类: cs.CV; cs.RO**

- **简介: 论文提出OmniVLA，一种融合红外、毫米波雷达和麦克风阵列的多模态视觉-语言-动作模型，通过传感器掩码图像统一多传感器数据，突破RGB-only限制，显著提升物理交互任务的成功率与学习效率。**

- **链接: [http://arxiv.org/pdf/2511.01210v1](http://arxiv.org/pdf/2511.01210v1)**

> **作者:** Heyu Guo; Shanmu Wang; Ruichun Ma; Shiqi Jiang; Yasaman Ghasempour; Omid Abari; Baining Guo; Lili Qi
>
> **摘要:** Vision-language-action (VLA) models have shown strong generalization for action prediction through large-scale vision-language pretraining. However, most existing models rely solely on RGB cameras, limiting their perception and, consequently, manipulation capabilities. We present OmniVLA, an omni-modality VLA model that integrates novel sensing modalities for physically-grounded spatial intelligence beyond RGB perception. The core of our approach is the sensor-masked image, a unified representation that overlays spatially grounded and physically meaningful masks onto the RGB images, derived from sensors including an infrared camera, a mmWave radar, and a microphone array. This image-native unification keeps sensor input close to RGB statistics to facilitate training, provides a uniform interface across sensor hardware, and enables data-efficient learning with lightweight per-sensor projectors. Built on this, we present a multisensory vision-language-action model architecture and train the model based on an RGB-pretrained VLA backbone. We evaluate OmniVLA on challenging real-world tasks where sensor-modality perception is needed to guide the manipulation. OmniVLA achieves an average task success rate of 84%, significantly outperforms both RGB-only and raw-sensor-input baseline models by 59% and 28% respectively, meanwhile showing higher learning efficiency and stronger generalization capability.
>
---
#### [new 036] Transformer-Progressive Mamba Network for Lightweight Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文面向轻量级图像超分辨率任务，提出T-PMambaSR，融合窗口自注意力与渐进式Mamba，实现多尺度特征细粒度建模，并引入自适应高频细化模块恢复细节，在降低计算成本下超越现有Transformer与Mamba方法。**

- **链接: [http://arxiv.org/pdf/2511.03232v1](http://arxiv.org/pdf/2511.03232v1)**

> **作者:** Sichen Guo; Wenjie Li; Yuanyang Liu; Guangwei Gao; Jian Yang; Chia-Wen Lin
>
> **备注:** 12 pages, 10 figures, 7 tables
>
> **摘要:** Recently, Mamba-based super-resolution (SR) methods have demonstrated the ability to capture global receptive fields with linear complexity, addressing the quadratic computational cost of Transformer-based SR approaches. However, existing Mamba-based methods lack fine-grained transitions across different modeling scales, which limits the efficiency of feature representation. In this paper, we propose T-PMambaSR, a lightweight SR framework that integrates window-based self-attention with Progressive Mamba. By enabling interactions among receptive fields of different scales, our method establishes a fine-grained modeling paradigm that progressively enhances feature representation with linear complexity. Furthermore, we introduce an Adaptive High-Frequency Refinement Module (AHFRM) to recover high-frequency details lost during Transformer and Mamba processing. Extensive experiments demonstrate that T-PMambaSR progressively enhances the model's receptive field and expressiveness, yielding better performance than recent Transformer- or Mamba-based methods while incurring lower computational cost. Our codes will be released after acceptance.
>
---
#### [new 037] MvBody: Multi-View-Based Hybrid Transformer Using Optical 3D Body Scan for Explainable Cesarean Section Prediction
- **分类: cs.CV; 68T10, 68T45**

- **简介: 该论文提出MvBody模型，利用孕期3D体扫与自报数据预测剖宫产风险，解决资源受限地区预测难题。结合多视角Transformer与度量学习，实现84.62%准确率，并通过积分梯度解释关键因素。**

- **链接: [http://arxiv.org/pdf/2511.03212v1](http://arxiv.org/pdf/2511.03212v1)**

> **作者:** Ruting Cheng; Boyuan Feng; Yijiang Zheng; Chuhui Qiu; Aizierjiang Aiersilan; Joaquin A. Calderon; Wentao Zhao; Qing Pan; James K. Hahn
>
> **备注:** 19 pages, 4 figures
>
> **摘要:** Accurately assessing the risk of cesarean section (CS) delivery is critical, especially in settings with limited medical resources, where access to healthcare is often restricted. Early and reliable risk prediction allows better-informed prenatal care decisions and can improve maternal and neonatal outcomes. However, most existing predictive models are tailored for in-hospital use during labor and rely on parameters that are often unavailable in resource-limited or home-based settings. In this study, we conduct a pilot investigation to examine the feasibility of using 3D body shape for CS risk assessment for future applications with more affordable general devices. We propose a novel multi-view-based Transformer network, MvBody, which predicts CS risk using only self-reported medical data and 3D optical body scans obtained between the 31st and 38th weeks of gestation. To enhance training efficiency and model generalizability in data-scarce environments, we incorporate a metric learning loss into the network. Compared to widely used machine learning models and the latest advanced 3D analysis methods, our method demonstrates superior performance, achieving an accuracy of 84.62% and an Area Under the Receiver Operating Characteristic Curve (AUC-ROC) of 0.724 on the independent test set. To improve transparency and trust in the model's predictions, we apply the Integrated Gradients algorithm to provide theoretically grounded explanations of the model's decision-making process. Our results indicate that pre-pregnancy weight, maternal age, obstetric history, previous CS history, and body shape, particularly around the head and shoulders, are key contributors to CS risk prediction.
>
---
#### [new 038] Cropland Mapping using Geospatial Embeddings
- **分类: cs.CV**

- **简介: 该论文属于遥感土地覆盖分类任务，旨在解决传统方法效率低的问题。研究利用Presto和AlphaEarth的地理空间嵌入，在多哥实现高精度农田制图，简化流程并提升土地利用变化与气候影响评估能力。**

- **链接: [http://arxiv.org/pdf/2511.02923v1](http://arxiv.org/pdf/2511.02923v1)**

> **作者:** Ivan Zvonkov; Gabriel Tseng; Inbal Becker-Reshef; Hannah Kerner
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** Accurate and up-to-date land cover maps are essential for understanding land use change, a key driver of climate change. Geospatial embeddings offer a more efficient and accessible way to map landscape features, yet their use in real-world mapping applications remains underexplored. In this work, we evaluated the utility of geospatial embeddings for cropland mapping in Togo. We produced cropland maps using embeddings from Presto and AlphaEarth. Our findings show that geospatial embeddings can simplify workflows, achieve high-accuracy cropland classification and ultimately support better assessments of land use change and its climate impacts.
>
---
#### [new 039] SLIP: Structural-aware Language-Image Pretraining for Vision-Language Alignment
- **分类: cs.CV; cs.AI**

- **简介: 论文提出SLIP，一种结构感知的视觉-语言预训练方法，解决传统方法忽略图像-文本间结构关系的问题。通过构建电商商品共购图并引入结构对比损失，提升跨模态对齐性能，显著优于CLIP。**

- **链接: [http://arxiv.org/pdf/2511.03019v1](http://arxiv.org/pdf/2511.03019v1)**

> **作者:** Wenbo Lu
>
> **备注:** Capstone Paper
>
> **摘要:** Vision-Language Pretraining (VLP) has achieved remarkable success across various downstream tasks, but such gains are largely driven by scaling up on training data. Yet, literature methods treat image-text pairs as isolated training examples; this neglects the rich relational structure naturally present in many domains, such as e-commerce product co-purchase graphs and social recommendation networks. Inspired by neuroscientific evidence that human encodes knowledge as relationship cognitive maps, we introduce Structure-aware Language-Image Pretraining (SLIP). SLIP integrates a structural contrastive loss to align modalities while also modeling relationships between neighboring entities in a structured graph. To support this paradigm, we construct a large-scale Amazon Product Co-purchase Multimodal Graph Dataset, enabling structured cross-modality supervision at scale. Experiment results show that SLIP consistently outperforms CLIP on cross-modal retrieval and classification tasks in both zero-shot and few-shot settings, showing the value of relational supervision for cross-modal alignment.
>
---
#### [new 040] PETWB-REP: A Multi-Cancer Whole-Body FDG PET/CT and Radiology Report Dataset for Medical Imaging Research
- **分类: cs.CV**

- **简介: 该论文发布PETWB-REP数据集，解决多癌种功能与解剖影像结合的标注数据稀缺问题，构建了490例FDG PET/CT影像与脱敏放射报告的多模态数据集，支持医学影像AI与放射组学研究。**

- **链接: [http://arxiv.org/pdf/2511.03194v1](http://arxiv.org/pdf/2511.03194v1)**

> **作者:** Le Xue; Gang Feng; Wenbo Zhang; Yichi Zhang; Lanlan Li; Shuqi Wang; Liling Peng; Sisi Peng; Xin Gao
>
> **摘要:** Publicly available, large-scale medical imaging datasets are crucial for developing and validating artificial intelligence models and conducting retrospective clinical research. However, datasets that combine functional and anatomical imaging with detailed clinical reports across multiple cancer types remain scarce. Here, we present PETWB-REP, a curated dataset comprising whole-body 18F-Fluorodeoxyglucose (FDG) Positron Emission Tomography/Computed Tomography (PET/CT) scans and corresponding radiology reports from 490 patients diagnosed with various malignancies. The dataset primarily includes common cancers such as lung cancer, liver cancer, breast cancer, prostate cancer, and ovarian cancer. This dataset includes paired PET and CT images, de-identified textual reports, and structured clinical metadata. It is designed to support research in medical imaging, radiomics, artificial intelligence, and multi-modal learning.
>
---
#### [new 041] Learning with less: label-efficient land cover classification at very high spatial resolution using self-supervised deep learning
- **分类: cs.CV**

- **简介: 该论文针对高分辨率土地覆盖分类标注数据稀缺问题，提出基于自监督学习的标签高效方法，仅用1000个标注样本，通过无标签图像预训练ResNet-101，迁移至多种分割模型，实现州级1米分辨率、8类土地覆盖的高精度制图。**

- **链接: [http://arxiv.org/pdf/2511.03004v1](http://arxiv.org/pdf/2511.03004v1)**

> **作者:** Dakota Hester; Vitor S. Martins; Lucas B. Ferreira; Thainara M. A. Lima
>
> **备注:** 25 pages, 11 figures. Submitted in Science of Remote Sensing
>
> **摘要:** Deep learning semantic segmentation methods have shown promising performance for very high 1-m resolution land cover classification, but the challenge of collecting large volumes of representative training data creates a significant barrier to widespread adoption of such models for meter-scale land cover mapping over large areas. In this study, we present a novel label-efficient approach for statewide 1-m land cover classification using only 1,000 annotated reference image patches with self-supervised deep learning. We use the "Bootstrap Your Own Latent" pre-training strategy with a large amount of unlabeled color-infrared aerial images (377,921 256x256 1-m pixel patches) to pre-train a ResNet-101 convolutional encoder. The learned encoder weights were subsequently transferred into multiple deep semantic segmentation architectures (FCN, U-Net, Attention U-Net, DeepLabV3+, UPerNet, PAN), which were then fine-tuned using very small training dataset sizes with cross-validation (250, 500, 750 patches). Among the fine-tuned models, we obtained the 87.14% overall accuracy and 75.58% macro F1 score using an ensemble of the best performing U-Net models for comprehensive 1-m, 8-class land cover mapping, covering more than 123 billion pixels over the state of Mississippi, USA. Detailed qualitative and quantitative analysis revealed accurate mapping of open water and forested areas, while highlighting challenges in accurate delineation between cropland, herbaceous, and barren land cover types. These results show that self-supervised learning is an effective strategy for reducing the need for large volumes of manually annotated data, directly addressing a major limitation to high spatial resolution land cover mapping at scale.
>
---
#### [new 042] ProM3E: Probabilistic Masked MultiModal Embedding Model for Ecology
- **分类: cs.CV**

- **简介: ProM3E提出一种概率性多模态嵌入模型，用于生态学中的任意模态间生成与融合，通过掩码重建学习模态间关联，支持模态反转与跨模态检索，并验证了其优越的表征学习能力。**

- **链接: [http://arxiv.org/pdf/2511.02946v1](http://arxiv.org/pdf/2511.02946v1)**

> **作者:** Srikumar Sastry; Subash Khanal; Aayush Dhakal; Jiayu Lin; Dan Cher; Phoenix Jarosz; Nathan Jacobs
>
> **备注:** 21 pages, 16 figures
>
> **摘要:** We introduce ProM3E, a probabilistic masked multimodal embedding model for any-to-any generation of multimodal representations for ecology. ProM3E is based on masked modality reconstruction in the embedding space, learning to infer missing modalities given a few context modalities. By design, our model supports modality inversion in the embedding space. The probabilistic nature of our model allows us to analyse the feasibility of fusing various modalities for given downstream tasks, essentially learning what to fuse. Using these features of our model, we propose a novel cross-modal retrieval approach that mixes inter-modal and intra-modal similarities to achieve superior performance across all retrieval tasks. We further leverage the hidden representation from our model to perform linear probing tasks and demonstrate the superior representation learning capability of our model. All our code, datasets and model will be released at https://vishu26.github.io/prom3e.
>
---
#### [new 043] Comprehensive Assessment of LiDAR Evaluation Metrics: A Comparative Study Using Simulated and Real Data
- **分类: cs.RO; cs.CV**

- **简介: 该论文旨在评估LiDAR仿真与实测数据的相似性度量方法，解决虚拟测试环境真实性验证问题。通过对比多种指标，发现密度感知切比雪夫距离（DCD）最有效，并在真实数据构建的仿真环境中验证了其与感知性能的高度相关性。**

- **链接: [http://arxiv.org/pdf/2511.02994v1](http://arxiv.org/pdf/2511.02994v1)**

> **作者:** Syed Mostaquim Ali; Taufiq Rahman; Ghazal Farhani; Mohamed H. Zaki; Benoit Anctil; Dominique Charlebois
>
> **摘要:** For developing safe Autonomous Driving Systems (ADS), rigorous testing is required before they are deemed safe for road deployments. Since comprehensive conventional physical testing is impractical due to cost and safety concerns, Virtual Testing Environments (VTE) can be adopted as an alternative. Comparing VTE-generated sensor outputs against their real-world analogues can be a strong indication that the VTE accurately represents reality. Correspondingly, this work explores a comprehensive experimental approach to finding evaluation metrics suitable for comparing real-world and simulated LiDAR scans. The metrics were tested in terms of sensitivity and accuracy with different noise, density, distortion, sensor orientation, and channel settings. From comparing the metrics, we found that Density Aware Chamfer Distance (DCD) works best across all cases. In the second step of the research, a Virtual Testing Environment was generated using real LiDAR scan data. The data was collected in a controlled environment with only static objects using an instrumented vehicle equipped with LiDAR, IMU and cameras. Simulated LiDAR scans were generated from the VTEs using the same pose as real LiDAR scans. The simulated and LiDAR scans were compared in terms of model perception and geometric similarity. Actual and simulated LiDAR scans have a similar semantic segmentation output with a mIoU of 21\% with corrected intensity and an average density aware chamfer distance (DCD) of 0.63. This indicates a slight difference in the geometric properties of simulated and real LiDAR scans and a significant difference between model outputs. During the comparison, density-aware chamfer distance was found to be the most correlated among the metrics with perception methods.
>
---
#### [new 044] Flying Robotics Art: ROS-based Drone Draws the Record-Breaking Mural
- **分类: cs.RO; cs.CV; cs.SY; eess.SY; I.2.9; J.5**

- **简介: 该论文提出一种基于ROS的自主无人机系统，用于在户外恶劣条件下精准绘制超大壁画。通过融合IR与LiDAR定位、方向解耦控制与抗湍流喷漆机构，解决艺术精度与环境鲁棒性难题，实现高精度自主绘画。**

- **链接: [http://arxiv.org/pdf/2511.03651v1](http://arxiv.org/pdf/2511.03651v1)**

> **作者:** Andrei A. Korigodskii; Oleg D. Kalachev; Artem E. Vasiunik; Matvei V. Urvantsev; Georgii E. Bondar
>
> **摘要:** This paper presents the innovative design and successful deployment of a pioneering autonomous unmanned aerial system developed for executing the world's largest mural painted by a drone. Addressing the dual challenges of maintaining artistic precision and operational reliability under adverse outdoor conditions such as wind and direct sunlight, our work introduces a robust system capable of navigating and painting outdoors with unprecedented accuracy. Key to our approach is a novel navigation system that combines an infrared (IR) motion capture camera and LiDAR technology, enabling precise location tracking tailored specifically for largescale artistic applications. We employ a unique control architecture that uses different regulation in tangential and normal directions relative to the planned path, enabling precise trajectory tracking and stable line rendering. We also present algorithms for trajectory planning and path optimization, allowing for complex curve drawing and area filling. The system includes a custom-designed paint spraying mechanism, specifically engineered to function effectively amidst the turbulent airflow generated by the drone's propellers, which also protects the drone's critical components from paint-related damage, ensuring longevity and consistent performance. Experimental results demonstrate the system's robustness and precision in varied conditions, showcasing its potential for autonomous large-scale art creation and expanding the functional applications of robotics in creative fields.
>
---
#### [new 045] Test Time Adaptation Using Adaptive Quantile Recalibration
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出自适应分位数重校准（AQR），用于无监督测试时域自适应，解决传统方法依赖先验知识或重训练的问题。AQR通过通道级分位数对齐捕获激活分布全貌，兼容多种归一化层，显著提升模型在分布偏移下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.03148v1](http://arxiv.org/pdf/2511.03148v1)**

> **作者:** Paria Mehrbod; Pedro Vianna; Geraldin Nanfack; Guy Wolf; Eugene Belilovsky
>
> **摘要:** Domain adaptation is a key strategy for enhancing the generalizability of deep learning models in real-world scenarios, where test distributions often diverge significantly from the training domain. However, conventional approaches typically rely on prior knowledge of the target domain or require model retraining, limiting their practicality in dynamic or resource-constrained environments. Recent test-time adaptation methods based on batch normalization statistic updates allow for unsupervised adaptation, but they often fail to capture complex activation distributions and are constrained to specific normalization layers. We propose Adaptive Quantile Recalibration (AQR), a test-time adaptation technique that modifies pre-activation distributions by aligning quantiles on a channel-wise basis. AQR captures the full shape of activation distributions and generalizes across architectures employing BatchNorm, GroupNorm, or LayerNorm. To address the challenge of estimating distribution tails under varying batch sizes, AQR incorporates a robust tail calibration strategy that improves stability and precision. Our method leverages source-domain statistics computed at training time, enabling unsupervised adaptation without retraining models. Experiments on CIFAR-10-C, CIFAR-100-C, and ImageNet-C across multiple architectures demonstrate that AQR achieves robust adaptation across diverse settings, outperforming existing test-time adaptation baselines. These results highlight AQR's potential for deployment in real-world scenarios with dynamic and unpredictable data distributions.
>
---
#### [new 046] A Probabilistic U-Net Approach to Downscaling Climate Simulations
- **分类: cs.LG; cs.CV; physics.ao-ph**

- **简介: 该论文将概率U-Net用于气候模拟的统计降尺度任务，解决粗分辨率模型无法满足精细研究需求的问题。通过对比四种损失函数，发现WMSE-MS-SSIM更优处理极端值，afCRPS更佳捕捉多尺度空间变异性。**

- **链接: [http://arxiv.org/pdf/2511.03197v1](http://arxiv.org/pdf/2511.03197v1)**

> **作者:** Maryam Alipourhajiagha; Pierre-Louis Lemaire; Youssef Diouane; Julie Carreau
>
> **备注:** NeurIPS 2025 AI4Science
>
> **摘要:** Climate models are limited by heavy computational costs, often producing outputs at coarse spatial resolutions, while many climate change impact studies require finer scales. Statistical downscaling bridges this gap, and we adapt the probabilistic U-Net for this task, combining a deterministic U-Net backbone with a variational latent space to capture aleatoric uncertainty. We evaluate four training objectives, afCRPS and WMSE-MS-SSIM with three settings for downscaling precipitation and temperature from $16\times$ coarser resolution. Our main finding is that WMSE-MS-SSIM performs well for extremes under certain settings, whereas afCRPS better captures spatial variability across scales.
>
---
#### [new 047] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI
- **分类: eess.IV; cs.CV; I.2.10; I.4.8; J.3**

- **简介: 该论文针对撒哈拉以南非洲MRI数据稀缺与域偏移问题，提出SegFormer3D-plus，结合直方图匹配、放射组学采样与频率感知Transformer，实现低资源环境下脑胶质瘤的高效精准分割。**

- **链接: [http://arxiv.org/pdf/2511.02928v1](http://arxiv.org/pdf/2511.02928v1)**

> **作者:** Ilerioluwakiiye Abolade; Aniekan Udo; Augustine Ojo; Abdulbasit Oyetunji; Hammed Ajigbotosho; Aondana Iorumbur; Confidence Raymond; Maruf Adewole
>
> **备注:** 4 pages, 2 figures. Accepted as an abstract at the Women in Machine Learning (WiML) Workshop at NeurIPS 2025
>
> **摘要:** Glioma segmentation is critical for diagnosis and treatment planning, yet remains challenging in Sub-Saharan Africa due to limited MRI infrastructure and heterogeneous acquisition protocols that induce severe domain shift. We propose SegFormer3D-plus, a radiomics-guided transformer architecture designed for robust segmentation under domain variability. Our method combines: (1) histogram matching for intensity harmonization across scanners, (2) radiomic feature extraction with PCA-reduced k-means for domain-aware stratified sampling, (3) a dual-pathway encoder with frequency-aware feature extraction and spatial-channel attention, and (4) composite Dice-Cross-Entropy loss for boundary refinement. Pretrained on BraTS 2023 and fine-tuned on BraTS-Africa data, SegFormer3D-plus demonstrates improved tumor subregion delineation and boundary localization across heterogeneous African clinical scans, highlighting the value of radiomics-guided domain adaptation for resource-limited settings.
>
---
#### [new 048] Scheduling the Off-Diagonal Weingarten Loss of Neural SDFs for CAD Models
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文针对神经SDF的CAD重建任务，解决固定曲率正则化权重抑制细节恢复的问题，提出多种ODW损失调度策略，动态调整正则化强度，显著提升重建精度，较基线提升35% Chamfer Distance。**

- **链接: [http://arxiv.org/pdf/2511.03147v1](http://arxiv.org/pdf/2511.03147v1)**

> **作者:** Haotian Yin; Przemyslaw Musialski
>
> **备注:** Lecture Notes in Computer Science (LNCS), 20th International Symposium on Visual Computing 2025, 12 pages, 4 figures, preprint
>
> **摘要:** Neural signed distance functions (SDFs) have become a powerful representation for geometric reconstruction from point clouds, yet they often require both gradient- and curvature-based regularization to suppress spurious warp and preserve structural fidelity. FlatCAD introduced the Off-Diagonal Weingarten (ODW) loss as an efficient second-order prior for CAD surfaces, approximating full-Hessian regularization at roughly half the computational cost. However, FlatCAD applies a fixed ODW weight throughout training, which is suboptimal: strong regularization stabilizes early optimization but suppresses detail recovery in later stages. We present scheduling strategies for the ODW loss that assign a high initial weight to stabilize optimization and progressively decay it to permit fine-scale refinement. We investigate constant, linear, quintic, and step interpolation schedules, as well as an increasing warm-up variant. Experiments on the ABC CAD dataset demonstrate that time-varying schedules consistently outperform fixed weights. Our method achieves up to a 35% improvement in Chamfer Distance over the FlatCAD baseline, establishing scheduling as a simple yet effective extension of curvature regularization for robust CAD reconstruction.
>
---
#### [new 049] Optimizing the nnU-Net model for brain tumor (Glioma) segmentation Using a BraTS Sub-Saharan Africa (SSA) dataset
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对脑胶质瘤分割任务，利用BraTS SSA数据集优化nnU-Net模型，发现原始60例数据结合在线增强优于360例离线增强数据，揭示数据质量与真实变异对模型泛化的重要性，Dice分数达0.84。**

- **链接: [http://arxiv.org/pdf/2511.02893v1](http://arxiv.org/pdf/2511.02893v1)**

> **作者:** Chukwuemeka Arua Kalu; Adaobi Chiazor Emegoakor; Fortune Okafor; Augustine Okoh Uchenna; Chijioke Kelvin Ukpai; Godsent Erere Onyeugbo
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Medical image segmentation is a critical achievement in modern medical science, developed over decades of research. It allows for the exact delineation of anatomical and pathological features in two- or three-dimensional pictures by utilizing notions like pixel intensity, texture, and anatomical context. With the advent of automated segmentation, physicians and radiologists may now concentrate on diagnosis and treatment planning while intelligent computers perform routine image processing tasks. This study used the BraTS Sub-Saharan Africa dataset, a selected subset of the BraTS dataset that included 60 multimodal MRI cases from patients with glioma. Surprisingly, the nnU Net model trained on the initial 60 instances performed better than the network trained on an offline-augmented dataset of 360 cases. Hypothetically, the offline augmentations introduced artificial anatomical variances or intensity distributions, reducing generalization. In contrast, the original dataset, when paired with nnU Net's robust online augmentation procedures, maintained realistic variability and produced better results. The study achieved a Dice score of 0.84 for whole tumor segmentation. These findings highlight the significance of data quality and proper augmentation approaches in constructing accurate, generalizable medical picture segmentation models, particularly for under-represented locations.
>
---
#### [new 050] Decoupled Entropy Minimization
- **分类: cs.LG; cs.CV; cs.IT; math.IT; math.ST; stat.ML; stat.TH**

- **简介: 该论文针对熵最小化（EM）在弱监督学习中的耦合缺陷，提出解耦框架AdaDEM，分离聚类驱动与梯度校准机制，缓解奖励坍缩与易类偏差，提升噪声与动态环境下的模型性能。**

- **链接: [http://arxiv.org/pdf/2511.03256v1](http://arxiv.org/pdf/2511.03256v1)**

> **作者:** Jing Ma; Hanlin Li; Xiang Xiang
>
> **备注:** To appear at NeurIPS 2025 (main conference), San Diego, CA, USA. Codes available at https://github.com/HAIV-Lab/DEM/
>
> **摘要:** Entropy Minimization (EM) is beneficial to reducing class overlap, bridging domain gap, and restricting uncertainty for various tasks in machine learning, yet its potential is limited. To study the internal mechanism of EM, we reformulate and decouple the classical EM into two parts with opposite effects: cluster aggregation driving factor (CADF) rewards dominant classes and prompts a peaked output distribution, while gradient mitigation calibrator (GMC) penalizes high-confidence classes based on predicted probabilities. Furthermore, we reveal the limitations of classical EM caused by its coupled formulation: 1) reward collapse impedes the contribution of high-certainty samples in the learning process, and 2) easy-class bias induces misalignment between output distribution and label distribution. To address these issues, we propose Adaptive Decoupled Entropy Minimization (AdaDEM), which normalizes the reward brought from CADF and employs a marginal entropy calibrator (MEC) to replace GMC. AdaDEM outperforms DEM*, an upper-bound variant of classical EM, and achieves superior performance across various imperfectly supervised learning tasks in noisy and dynamic environments.
>
---
#### [new 051] Morpho-Genomic Deep Learning for Ovarian Cancer Subtype and Gene Mutation Prediction from Histopathology
- **分类: eess.IV; cs.CV; q-bio.QM**

- **简介: 该论文提出一种融合ResNet-50与ViT的深度学习模型，从H&E病理图像中预测卵巢癌亚型和关键基因突变（如TP53、BRCA1），实现无基因检测的精准分型，解决传统诊断无法揭示基因变异的问题。**

- **链接: [http://arxiv.org/pdf/2511.03365v1](http://arxiv.org/pdf/2511.03365v1)**

> **作者:** Gabriela Fernandes
>
> **摘要:** Ovarian cancer remains one of the most lethal gynecological malignancies, largely due to late diagnosis and extensive heterogeneity across subtypes. Current diagnostic methods are limited in their ability to reveal underlying genomic variations essential for precision oncology. This study introduces a novel hybrid deep learning pipeline that integrates quantitative nuclear morphometry with deep convolutional image features to perform ovarian cancer subtype classification and gene mutation inference directly from Hematoxylin and Eosin (H&E) histopathological images. Using $\sim45,000$ image patches sourced from The Cancer Genome Atlas (TCGA) and public datasets, a fusion model combining a ResNet-50 Convolutional Neural Network (CNN) encoder and a Vision Transformer (ViT) was developed. This model successfully captured both local morphological texture and global tissue context. The pipeline achieved a robust overall subtype classification accuracy of $84.2\%$ (Macro AUC of $0.87 \pm 0.03$). Crucially, the model demonstrated the capacity for gene mutation inference with moderate-to-high accuracy: $AUC_{TP53} = 0.82 \pm 0.02$, $AUC_{BRCA1} = 0.76 \pm 0.04$, and $AUC_{ARID1A} = 0.73 \pm 0.05$. Feature importance analysis established direct quantitative links, revealing that nuclear solidity and eccentricity were the dominant predictors for TP53 mutation. These findings validate that quantifiable histological phenotypes encode measurable genomic signals, paving the way for cost-effective, precision histopathology in ovarian cancer triage and diagnosis.
>
---
#### [new 052] NEF-NET+: Adapting Electrocardio panorama in the wild
- **分类: eess.SP; cs.AI; cs.CV; eess.IV**

- **简介: 论文提出NEF-NET+，解决传统ECG视角受限问题，实现任意视角、长时程、跨设备的心电全景合成，通过新架构与校准流程提升鲁棒性，并构建Panobench基准验证性能，显著优于前作。**

- **链接: [http://arxiv.org/pdf/2511.02880v1](http://arxiv.org/pdf/2511.02880v1)**

> **作者:** Zehui Zhan; Yaojun Hu; Jiajing Zhan; Wanchen Lian; Wanqing Wu; Jintai Chen
>
> **摘要:** Conventional multi-lead electrocardiogram (ECG) systems capture cardiac signals from a fixed set of anatomical viewpoints defined by lead placement. However, certain cardiac conditions (e.g., Brugada syndrome) require additional, non-standard viewpoints to reveal diagnostically critical patterns that may be absent in standard leads. To systematically overcome this limitation, Nef-Net was recently introduced to reconstruct a continuous electrocardiac field, enabling virtual observation of ECG signals from arbitrary views (termed Electrocardio Panorama). Despite its promise, Nef-Net operates under idealized assumptions and faces in-the-wild challenges, such as long-duration ECG modeling, robustness to device-specific signal artifacts, and suboptimal lead placement calibration. This paper presents NEF-NET+, an enhanced framework for realistic panoramic ECG synthesis that supports arbitrary-length signal synthesis from any desired view, generalizes across ECG devices, and com- pensates for operator-induced deviations in electrode placement. These capabilities are enabled by a newly designed model architecture that performs direct view transformation, incorporating a workflow comprising offline pretraining, device calibration tuning steps as well as an on-the-fly calibration step for patient-specific adaptation. To rigorously evaluate panoramic ECG synthesis, we construct a new Electrocardio Panorama benchmark, called Panobench, comprising 5367 recordings with 48-view per subject, capturing the full spatial variability of cardiac electrical activity. Experimental results show that NEF-NET+ delivers substantial improvements over Nef-Net, yielding an increase of around 6 dB in PSNR in real-world setting. The code and Panobench will be released in a subsequent publication.
>
---
#### [new 053] Seeing What You Say: Expressive Image Generation from Speech
- **分类: eess.AS; cs.CV; cs.MM**

- **简介: 论文提出VoxStudio，首个端到端语音到图像生成模型，直接从语音生成富有表现力的图像，通过语音信息瓶颈模块保留语调与情感，避免依赖文本中间表示，并构建了VoxEmoset情感语音图像数据集。**

- **链接: [http://arxiv.org/pdf/2511.03423v1](http://arxiv.org/pdf/2511.03423v1)**

> **作者:** Jiyoung Lee; Song Park; Sanghyuk Chun; Soo-Whan Chung
>
> **备注:** In progress
>
> **摘要:** This paper proposes VoxStudio, the first unified and end-to-end speech-to-image model that generates expressive images directly from spoken descriptions by jointly aligning linguistic and paralinguistic information. At its core is a speech information bottleneck (SIB) module, which compresses raw speech into compact semantic tokens, preserving prosody and emotional nuance. By operating directly on these tokens, VoxStudio eliminates the need for an additional speech-to-text system, which often ignores the hidden details beyond text, e.g., tone or emotion. We also release VoxEmoset, a large-scale paired emotional speech-image dataset built via an advanced TTS engine to affordably generate richly expressive utterances. Comprehensive experiments on the SpokenCOCO, Flickr8kAudio, and VoxEmoset benchmarks demonstrate the feasibility of our method and highlight key challenges, including emotional consistency and linguistic ambiguity, paving the way for future research.
>
---
#### [new 054] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 论文提出OneOcc，面向腿足机器人，仅用全景相机实现360°语义占据预测，解决运动抖动与视角连续性难题，创新融合双投影、双网格、动态解码与步态补偿模块，并发布两个全景基准数据集，达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.03571v1](http://arxiv.org/pdf/2511.03571v1)**

> **作者:** Hao Shi; Ze Wang; Shangwei Guo; Mengfei Duan; Song Wang; Teng Chen; Kailun Yang; Lin Wang; Kaiwei Wang
>
> **备注:** Datasets and code will be publicly available at https://github.com/MasterHow/OneOcc
>
> **摘要:** Robust 3D semantic occupancy is crucial for legged/humanoid robots, yet most semantic scene completion (SSC) systems target wheeled platforms with forward-facing sensors. We present OneOcc, a vision-only panoramic SSC framework designed for gait-introduced body jitter and 360{\deg} continuity. OneOcc combines: (i) Dual-Projection fusion (DP-ER) to exploit the annular panorama and its equirectangular unfolding, preserving 360{\deg} continuity and grid alignment; (ii) Bi-Grid Voxelization (BGV) to reason in Cartesian and cylindrical-polar spaces, reducing discretization bias and sharpening free/occupied boundaries; (iii) a lightweight decoder with Hierarchical AMoE-3D for dynamic multi-scale fusion and better long-range/occlusion reasoning; and (iv) plug-and-play Gait Displacement Compensation (GDC) learning feature-level motion correction without extra sensors. We also release two panoramic occupancy benchmarks: QuadOcc (real quadruped, first-person 360{\deg}) and Human360Occ (H3O) (CARLA human-ego 360{\deg} with RGB, Depth, semantic occupancy; standardized within-/cross-city splits). OneOcc sets new state-of-the-art (SOTA): on QuadOcc it beats strong vision baselines and popular LiDAR ones; on H3O it gains +3.83 mIoU (within-city) and +8.08 (cross-city). Modules are lightweight, enabling deployable full-surround perception for legged/humanoid robots. Datasets and code will be publicly available at https://github.com/MasterHow/OneOcc.
>
---
#### [new 055] Benchmarking ResNet for Short-Term Hypoglycemia Classification with DiaData
- **分类: eess.SP; cs.CV; eess.IV**

- **简介: 该论文面向1型糖尿病短期低血糖预测任务，通过提升DiaData数据质量（去噪、插补），并基于ResNet构建分类模型，实现提前2小时预测低血糖，验证了数据清洗与数据量对模型性能的提升作用。**

- **链接: [http://arxiv.org/pdf/2511.02849v1](http://arxiv.org/pdf/2511.02849v1)**

> **作者:** Beyza Cinar; Maria Maleshkova
>
> **备注:** 11 pages, 5 Tables, 4 Figures, BHI 2025 conference (JBHI special issue)
>
> **摘要:** Individualized therapy is driven forward by medical data analysis, which provides insight into the patient's context. In particular, for Type 1 Diabetes (T1D), which is an autoimmune disease, relationships between demographics, sensor data, and context can be analyzed. However, outliers, noisy data, and small data volumes cannot provide a reliable analysis. Hence, the research domain requires large volumes of high-quality data. Moreover, missing values can lead to information loss. To address this limitation, this study improves the data quality of DiaData, an integration of 15 separate datasets containing glucose values from 2510 subjects with T1D. Notably, we make the following contributions: 1) Outliers are identified with the interquartile range (IQR) approach and treated by replacing them with missing values. 2) Small gaps ($\le$ 25 min) are imputed with linear interpolation and larger gaps ($\ge$ 30 and $<$ 120 min) with Stineman interpolation. Based on a visual comparison, Stineman interpolation provides more realistic glucose estimates than linear interpolation for larger gaps. 3) After data cleaning, the correlation between glucose and heart rate is analyzed, yielding a moderate relation between 15 and 60 minutes before hypoglycemia ($\le$ 70 mg/dL). 4) Finally, a benchmark for hypoglycemia classification is provided with a state-of-the-art ResNet model. The model is trained with the Maindatabase and Subdatabase II of DiaData to classify hypoglycemia onset up to 2 hours in advance. Training with more data improves performance by 7% while using quality-refined data yields a 2-3% gain compared to raw data.
>
---
#### [new 056] A Feedback-Control Framework for Efficient Dataset Collection from In-Vehicle Data Streams
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出FCDC框架，将车载数据采集建模为闭环反馈控制问题，通过在线概率模型动态调节样本保留，解决传统开环采集的冗余与低效问题，实现数据多样性提升与存储减少。**

- **链接: [http://arxiv.org/pdf/2511.03239v1](http://arxiv.org/pdf/2511.03239v1)**

> **作者:** Philipp Reis; Philipp Rigoll; Christian Steinhauser; Jacob Langner; Eric Sax
>
> **摘要:** Modern AI systems are increasingly constrained not by model capacity but by the quality and diversity of their data. Despite growing emphasis on data-centric AI, most datasets are still gathered in an open-loop manner which accumulates redundant samples without feedback from the current coverage. This results in inefficient storage, costly labeling, and limited generalization. To address this, this paper introduces \ac{FCDC}, a paradigm that formulates data collection as a closed-loop control problem. \ac{FCDC} continuously approximates the state of the collected data distribution using an online probabilistic model and adaptively regulates sample retention using based on feedback signals such as likelihood and Mahalanobis distance. Through this feedback mechanism, the system dynamically balances exploration and exploitation, maintains dataset diversity, and prevents redundancy from accumulating over time. Besides showcasing the controllability of \ac{FCDC} on a synthetic dataset, experiments on a real data stream show that \ac{FCDC} produces more balanced datasets by $\SI{25.9}{\percent}$ while reducing data storage by $\SI{39.8}{\percent}$. These results demonstrate that data collection itself can be actively controlled, transforming collection from a passive pipeline stage into a self-regulating, feedback-driven process at the core of data-centric AI.
>
---
#### [new 057] Benchmarking the Thinking Mode of Multimodal Large Language Models in Clinical Tasks
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文评估多模态大模型在临床任务中“思考模式”对性能的影响，对比其与“非思考模式”在医学视觉问答和图像解读上的表现，发现提升有限，凸显医疗专用数据与知识整合方法的迫切需求。**

- **链接: [http://arxiv.org/pdf/2511.03328v1](http://arxiv.org/pdf/2511.03328v1)**

> **作者:** Jindong Hong; Tianjie Chen; Lingjie Luo; Chuanyang Zheng; Ting Xu; Haibao Yu; Jianing Qiu; Qianzhong Chen; Suning Huang; Yan Xu; Yong Gui; Yijun He; Jiankai Sun
>
> **摘要:** A recent advancement in Multimodal Large Language Models (MLLMs) research is the emergence of "reasoning MLLMs" that offer explicit control over their internal thinking processes (normally referred as the "thinking mode") alongside the standard "non-thinking mode". This capability allows these models to engage in a step-by-step process of internal deliberation before generating a final response. With the rapid transition to and adoption of these "dual-state" MLLMs, this work rigorously evaluated how the enhanced reasoning processes of these MLLMs impact model performance and reliability in clinical tasks. This paper evaluates the active "thinking mode" capabilities of two leading MLLMs, Seed1.5-VL and Gemini-2.5-Flash, for medical applications. We assessed their performance on four visual medical tasks using VQA-RAD and ROCOv2 datasets. Our findings reveal that the improvement from activating the thinking mode remains marginal compared to the standard non-thinking mode for the majority of the tasks. Their performance on complex medical tasks such as open-ended VQA and medical image interpretation remains suboptimal, highlighting the need for domain-specific medical data and more advanced methods for medical knowledge integration.
>
---
#### [new 058] Data-Efficient Realized Volatility Forecasting with Vision Transformers
- **分类: cs.LG; cs.CV; I.4**

- **简介: 该论文将Vision Transformer用于期权隐含波动率面预测未来30天实现波动率，首次探索Transformer在期权数据上的应用，揭示其捕捉非线性与季节性模式的潜力，属金融时间序列预测任务。**

- **链接: [http://arxiv.org/pdf/2511.03046v1](http://arxiv.org/pdf/2511.03046v1)**

> **作者:** Emi Soroka; Artem Arzyn
>
> **备注:** NeurIPS Generative AI in Finance
>
> **摘要:** Recent work in financial machine learning has shown the virtue of complexity: the phenomenon by which deep learning methods capable of learning highly nonlinear relationships outperform simpler approaches in financial forecasting. While transformer architectures like Informer have shown promise for financial time series forecasting, the application of transformer models for options data remains largely unexplored. We conduct preliminary studies towards the development of a transformer model for options data by training the Vision Transformer (ViT) architecture, typically used in modern image recognition and classification systems, to predict the realized volatility of an asset over the next 30 days from its implied volatility surface (augmented with date information) for a single day. We show that the ViT can learn seasonal patterns and nonlinear features from the IV surface, suggesting a promising direction for model development.
>
---
## 更新

#### [replaced 001] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing
- **分类: eess.AS; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.21448v3](http://arxiv.org/pdf/2506.21448v3)**

> **作者:** Huadai Liu; Kaicheng Luo; Jialei Wang; Wen Wang; Qian Chen; Zhou Zhao; Wei Xue
>
> **备注:** Accepted by NeurIPS 2025 Main
>
> **摘要:** While end-to-end video-to-audio generation has greatly improved, producing high-fidelity audio that authentically captures the nuances of visual content remains challenging. Like professionals in the creative industries, this generation requires sophisticated reasoning about items such as visual dynamics, acoustic environments, and temporal relationships. We present ThinkSound, a novel framework that leverages Chain-of-Thought (CoT) reasoning to enable stepwise, interactive audio generation and editing for videos. Our approach decomposes the process into three complementary stages: foundational foley generation that creates semantically coherent soundscapes, interactive object-centric refinement through precise user interactions, and targeted editing guided by natural language instructions. At each stage, a multimodal large language model generates contextually aligned CoT reasoning that guides a unified audio foundation model. Furthermore, we introduce AudioCoT, a comprehensive dataset with structured reasoning annotations that establishes connections between visual content, textual descriptions, and sound synthesis. Experiments demonstrate that ThinkSound achieves state-of-the-art performance in video-to-audio generation across both audio metrics and CoT metrics, and excels in the out-of-distribution Movie Gen Audio benchmark. The project page is available at https://ThinkSound-Project.github.io.
>
---
#### [replaced 002] Seal2Real: Prompt Prior Learning on Diffusion Model for Unsupervised Document Seal Data Generation and Realisation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2310.00546v2](http://arxiv.org/pdf/2310.00546v2)**

> **作者:** Mingfu Yan; Jiancheng Huang; Shifeng Chen
>
> **摘要:** Seal-related tasks in document processing-such as seal segmentation, authenticity verification, seal removal, and text recognition under seals-hold substantial commercial importance. However, progress in these areas has been hindered by the scarcity of labeled document seal datasets, which are essential for supervised learning. To address this limitation, we propose Seal2Real, a novel generative framework designed to synthesize large-scale labeled document seal data. As part of this work, we also present Seal-DB, a comprehensive dataset containing 20,000 labeled images to support seal-related research. Seal2Real introduces a prompt prior learning architecture built upon a pre-trained Stable Diffusion model, effectively transferring its generative capability to the unsupervised domain of seal image synthesis. By producing highly realistic synthetic seal images, Seal2Real significantly enhances the performance of downstream seal-related tasks on real-world data. Experimental evaluations on the Seal-DB dataset demonstrate the effectiveness and practical value of the proposed framework.
>
---
#### [replaced 003] A Survey on Text-Driven 360-Degree Panorama Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14799v3](http://arxiv.org/pdf/2502.14799v3)**

> **作者:** Hai Wang; Xiaoyu Xiang; Weihao Xia; Jing-Hao Xue
>
> **备注:** Accepted by IEEE TCSVT, Code: https://github.com/littlewhitesea/Text-Driven-Pano-Gen
>
> **摘要:** The advent of text-driven 360-degree panorama generation, enabling the synthesis of 360-degree panoramic images directly from textual descriptions, marks a transformative advancement in immersive visual content creation. This innovation significantly simplifies the traditionally complex process of producing such content. Recent progress in text-to-image diffusion models has accelerated the rapid development in this emerging field. This survey presents a comprehensive review of text-driven 360-degree panorama generation, offering an in-depth analysis of state-of-the-art algorithms. We extend our analysis to two closely related domains: text-driven 360-degree 3D scene generation and text-driven 360-degree panoramic video generation. Furthermore, we critically examine current limitations and propose promising directions for future research. A curated project page with relevant resources and research papers is available at https://littlewhitesea.github.io/Text-Driven-Pano-Gen/.
>
---
#### [replaced 004] Benchmarking Foundation Models and Parameter-Efficient Fine-Tuning for Prognosis Prediction in Medical Imaging
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18434v2](http://arxiv.org/pdf/2506.18434v2)**

> **作者:** Filippo Ruffini; Elena Mulero Ayllon; Linlin Shen; Paolo Soda; Valerio Guarrasi
>
> **摘要:** Despite the significant potential of Foundation Models (FMs) in medical imaging, their application to prognosis prediction remains challenging due to data scarcity, class imbalance, and task complexity, which limit their clinical adoption. This study introduces the first structured benchmark to assess the robustness and efficiency of transfer learning strategies for FMs compared with convolutional neural networks (CNNs) in predicting COVID-19 patient outcomes from chest X-rays. The goal is to systematically compare finetuning strategies, both classical and parameter efficient, under realistic clinical constraints related to data scarcity and class imbalance, offering empirical guidance for AI deployment in clinical workflows. Four publicly available COVID-19 chest X-ray datasets were used, covering mortality, severity, and ICU admission, with varying sample sizes and class imbalances. CNNs pretrained on ImageNet and FMs pretrained on general or biomedical datasets were adapted using full finetuning, linear probing, and parameter-efficient methods. Models were evaluated under full data and few shot regimes using the Matthews Correlation Coefficient (MCC) and Precision Recall AUC (PR-AUC), with cross validation and class weighted losses. CNNs with full fine-tuning performed robustly on small, imbalanced datasets, while FMs with Parameter-Efficient Fine-Tuning (PEFT), particularly LoRA and BitFit, achieved competitive results on larger datasets. Severe class imbalance degraded PEFT performance, whereas balanced data mitigated this effect. In few-shot settings, FMs showed limited generalization, with linear probing yielding the most stable results. No single fine-tuning strategy proved universally optimal: CNNs remain dependable for low-resource scenarios, whereas FMs benefit from parameter-efficient methods when data are sufficient.
>
---
#### [replaced 005] DA$^2$: Depth Anything in Any Direction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.26618v4](http://arxiv.org/pdf/2509.26618v4)**

> **作者:** Haodong Li; Wangguangdong Zheng; Jing He; Yuhao Liu; Xin Lin; Xin Yang; Ying-Cong Chen; Chunchao Guo
>
> **备注:** Work primarily done during an internship at Tencent Hunyuan. Project page: https://depth-any-in-any-dir.github.io/
>
> **摘要:** Panorama has a full FoV (360$^\circ\times$180$^\circ$), offering a more complete visual description than perspective images. Thanks to this characteristic, panoramic depth estimation is gaining increasing traction in 3D vision. However, due to the scarcity of panoramic data, previous methods are often restricted to in-domain settings, leading to poor zero-shot generalization. Furthermore, due to the spherical distortions inherent in panoramas, many approaches rely on perspective splitting (e.g., cubemaps), which leads to suboptimal efficiency. To address these challenges, we propose $\textbf{DA}$$^{\textbf{2}}$: $\textbf{D}$epth $\textbf{A}$nything in $\textbf{A}$ny $\textbf{D}$irection, an accurate, zero-shot generalizable, and fully end-to-end panoramic depth estimator. Specifically, for scaling up panoramic data, we introduce a data curation engine for generating high-quality panoramic depth data from perspective, and create $\sim$543K panoramic RGB-depth pairs, bringing the total to $\sim$607K. To further mitigate the spherical distortions, we present SphereViT, which explicitly leverages spherical coordinates to enforce the spherical geometric consistency in panoramic image features, yielding improved performance. A comprehensive benchmark on multiple datasets clearly demonstrates DA$^{2}$'s SoTA performance, with an average 38% improvement on AbsRel over the strongest zero-shot baseline. Surprisingly, DA$^{2}$ even outperforms prior in-domain methods, highlighting its superior zero-shot generalization. Moreover, as an end-to-end solution, DA$^{2}$ exhibits much higher efficiency over fusion-based approaches. Both the code and the curated panoramic data has be released. Project page: https://depth-any-in-any-dir.github.io/.
>
---
#### [replaced 006] OLATverse: A Large-scale Real-world Object Dataset with Precise Lighting Control
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2511.02483v2](http://arxiv.org/pdf/2511.02483v2)**

> **作者:** Xilong Zhou; Jianchun Chen; Pramod Rao; Timo Teufel; Linjie Lyu; Tigran Minasian; Oleksandr Sotnychenko; Xiao-Xiao Long; Marc Habermann; Christian Theobalt
>
> **摘要:** We introduce OLATverse, a large-scale dataset comprising around 9M images of 765 real-world objects, captured from multiple viewpoints under a diverse set of precisely controlled lighting conditions. While recent advances in object-centric inverse rendering, novel view synthesis and relighting have shown promising results, most techniques still heavily rely on the synthetic datasets for training and small-scale real-world datasets for benchmarking, which limits their realism and generalization. To address this gap, OLATverse offers two key advantages over existing datasets: large-scale coverage of real objects and high-fidelity appearance under precisely controlled illuminations. Specifically, OLATverse contains 765 common and uncommon real-world objects, spanning a wide range of material categories. Each object is captured using 35 DSLR cameras and 331 individually controlled light sources, enabling the simulation of diverse illumination conditions. In addition, for each object, we provide well-calibrated camera parameters, accurate object masks, photometric surface normals, and diffuse albedo as auxiliary resources. We also construct an extensive evaluation set, establishing the first comprehensive real-world object-centric benchmark for inverse rendering and normal estimation. We believe that OLATverse represents a pivotal step toward integrating the next generation of inverse rendering and relighting methods with real-world data. The full dataset, along with all post-processing workflows, will be publicly released at https://vcai.mpi-inf.mpg.de/projects/OLATverse/.
>
---
#### [replaced 007] Balancing Tails when Comparing Distributions: Comprehensive Equity Index (CEI) with Application to Bias Evaluation in Operational Face Biometrics
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.10564v2](http://arxiv.org/pdf/2506.10564v2)**

> **作者:** Imanol Solano; Julian Fierrez; Aythami Morales; Alejandro Peña; Ruben Tolosana; Francisco Zamora-Martinez; Javier San Agustin
>
> **摘要:** Demographic bias in high-performance face recognition (FR) systems often eludes detection by existing metrics, especially with respect to subtle disparities in the tails of the score distribution. We introduce the Comprehensive Equity Index (CEI), a novel metric designed to address this limitation. CEI uniquely analyzes genuine and impostor score distributions separately, enabling a configurable focus on tail probabilities while also considering overall distribution shapes. Our extensive experiments (evaluating state-of-the-art FR systems, intentionally biased models, and diverse datasets) confirm CEI's superior ability to detect nuanced biases where previous methods fall short. Furthermore, we present CEI^A, an automated version of the metric that enhances objectivity and simplifies practical application. CEI provides a robust and sensitive tool for operational FR fairness assessment. The proposed methods have been developed particularly for bias evaluation in face biometrics but, in general, they are applicable for comparing statistical distributions in any problem where one is interested in analyzing the distribution tails.
>
---
#### [replaced 008] SmartWilds: Multimodal Wildlife Monitoring Dataset
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.18894v2](http://arxiv.org/pdf/2509.18894v2)**

> **作者:** Jenna Kline; Anirudh Potlapally; Bharath Pillai; Tanishka Wani; Rugved Katole; Vedant Patil; Penelope Covey; Hari Subramoni; Tanya Berger-Wolf; Christopher Stewart
>
> **备注:** Accepted to Imageomics Workshop at Neurips 2025
>
> **摘要:** We present the first release of SmartWilds, a multimodal wildlife monitoring dataset. SmartWilds is a synchronized collection of drone imagery, camera trap photographs and videos, and bioacoustic recordings collected during summer 2025 at The Wilds safari park in Ohio. This dataset supports multimodal AI research for comprehensive environmental monitoring, addressing critical needs in endangered species research, conservation ecology, and habitat management. Our pilot deployment captured four days of synchronized monitoring across three modalities in a 220-acre pasture containing Pere David's deer, Sichuan takin, Przewalski's horses, as well as species native to Ohio. We provide a comparative analysis of sensor modality performance, demonstrating complementary strengths for landuse patterns, species detection, behavioral analysis, and habitat monitoring. This work establishes reproducible protocols for multimodal wildlife monitoring while contributing open datasets to advance conservation computer vision research. Future releases will include synchronized GPS tracking data from tagged individuals, citizen science data, and expanded temporal coverage across multiple seasons.
>
---
#### [replaced 009] BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14318v4](http://arxiv.org/pdf/2506.14318v4)**

> **作者:** Amirreza Fateh; Yasin Rezvani; Sara Moayedi; Sadjad Rezvani; Fatemeh Fateh; Mansoor Fateh; Vahid Abolghasemi
>
> **摘要:** Accurate segmentation and classification of brain tumors from Magnetic Resonance Imaging (MRI) remain key challenges in medical image analysis, primarily due to the lack of high-quality, balanced, and diverse datasets with expert annotations. In this work, we address this gap by introducing BRISC, a dataset designed for brain tumor segmentation and classification tasks, featuring high-resolution segmentation masks. The dataset comprises 6,000 contrast-enhanced T1-weighted MRI scans, which were collated from multiple public datasets that lacked segmentation labels. Our primary contribution is the subsequent expert annotation of these images, performed by certified radiologists and physicians. It includes three major tumor types, namely glioma, meningioma, and pituitary, as well as non-tumorous cases. Each sample includes high-resolution labels and is categorized across axial, sagittal, and coronal imaging planes to facilitate robust model development and cross-view generalization. To demonstrate the utility of the dataset, we provide benchmark results for both tasks using standard deep learning models. The BRISC dataset is made publicly available. datasetlink: Kaggle (https://www.kaggle.com/datasets/briscdataset/brisc2025/), Figshare (https://doi.org/10.6084/m9.figshare.30533120), Zenodo (https://doi.org/10.5281/zenodo.17524350)
>
---
#### [replaced 010] P3P Made Easy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.01312v2](http://arxiv.org/pdf/2508.01312v2)**

> **作者:** Seong Hun Lee; Patrick Vandewalle; Javier Civera
>
> **摘要:** We revisit the classical Perspective-Three-Point (P3P) problem, which aims to recover the absolute pose of a calibrated camera from three 2D-3D correspondences. It has long been known that P3P can be reduced to a quartic polynomial with analytically simple and computationally efficient coefficients. However, this elegant formulation has been largely overlooked in modern literature. Building on the theoretical foundation that traces back to Grunert's work in 1841, we propose a compact algebraic solver that achieves accuracy and runtime comparable to state-of-the-art methods. Our results show that this classical formulation remains highly competitive when implemented with modern insights, offering an excellent balance between simplicity, efficiency, and accuracy.
>
---
#### [replaced 011] SpatialLM: Training Large Language Models for Structured Indoor Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07491v2](http://arxiv.org/pdf/2506.07491v2)**

> **作者:** Yongsen Mao; Junhao Zhong; Chuan Fang; Jia Zheng; Rui Tang; Hao Zhu; Ping Tan; Zihan Zhou
>
> **摘要:** SpatialLM is a large language model designed to process 3D point cloud data and generate structured 3D scene understanding outputs. These outputs include architectural elements like walls, doors, windows, and oriented object boxes with their semantic categories. Unlike previous methods which exploit task-specific network designs, our model adheres to the standard multimodal LLM architecture and is fine-tuned directly from open-source LLMs. To train SpatialLM, we collect a large-scale, high-quality synthetic dataset consisting of the point clouds of 12,328 indoor scenes (54,778 rooms) with ground-truth 3D annotations, and conduct a careful study on various modeling and training decisions. On public benchmarks, our model gives state-of-the-art performance in layout estimation and competitive results in 3D object detection. With that, we show a feasible path for enhancing the spatial understanding capabilities of modern LLMs for applications in augmented reality, embodied robotics, and more.
>
---
#### [replaced 012] Towards Interpretable and Efficient Attention: Compressing All by Contracting a Few
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.16875v3](http://arxiv.org/pdf/2509.16875v3)**

> **作者:** Qishuai Wen; Zhiyuan Huang; Chun-Guang Li
>
> **备注:** NeurIPS2025 Spotlight; Code is available at https://github.com/QishuaiWen/CBSA
>
> **摘要:** Attention mechanisms have achieved significant empirical success in multiple fields, but their underlying optimization objectives remain unclear yet. Moreover, the quadratic complexity of self-attention has become increasingly prohibitive. Although interpretability and efficiency are two mutually reinforcing pursuits, prior work typically investigates them separately. In this paper, we propose a unified optimization objective that derives inherently interpretable and efficient attention mechanisms through algorithm unrolling. Precisely, we construct a gradient step of the proposed objective with a set of forward-pass operations of our \emph{Contract-and-Broadcast Self-Attention} (CBSA), which compresses input tokens towards low-dimensional structures by contracting a few representatives of them. This novel mechanism can not only scale linearly by fixing the number of representatives, but also covers the instantiations of varied attention mechanisms when using different sets of representatives. We conduct extensive experiments to demonstrate comparable performance and superior advantages over black-box attention mechanisms on visual tasks. Our work sheds light on the integration of interpretability and efficiency, as well as the unified formula of attention mechanisms.
>
---
#### [replaced 013] Text-guided Fine-Grained Video Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2511.00524v2](http://arxiv.org/pdf/2511.00524v2)**

> **作者:** Jihao Gu; Kun Li; He Wang; Kaan Akşit
>
> **摘要:** Video Anomaly Detection (VAD) aims to identify anomalous events within video segments. In scenarios such as surveillance or industrial process monitoring, anomaly detection is of critical importance. While existing approaches are semi-automated, requiring human assessment for anomaly detection, traditional VADs offer limited output as either normal or anomalous. We propose Text-guided Fine-Grained Video Anomaly Detection (T-VAD), a framework built upon Large Vision-Language Model (LVLM). T-VAD introduces an Anomaly Heatmap Decoder (AHD) that performs pixel-wise visual-textual feature alignment to generate fine-grained anomaly heatmaps. Furthermore, we design a Region-aware Anomaly Encoder (RAE) that transforms the heatmaps into learnable textual embeddings, guiding the LVLM to accurately identify and localize anomalous events in videos. This significantly enhances both the granularity and interactivity of anomaly detection. The proposed method achieving SOTA performance by demonstrating 94.8% Area Under the Curve (AUC, specifically micro-AUC) and 67.8%/76.7% accuracy in anomaly heatmaps (RBDC/TBDC) on the UBnormal dataset, and subjectively verified more preferable textual description on the ShanghaiTech-based dataset (BLEU-4: 62.67 for targets, 88.84 for trajectories; Yes/No accuracy: 97.67%), and on the UBnormal dataset (BLEU-4: 50.32 for targets, 78.10 for trajectories; Yes/No accuracy: 89.73%).
>
---
#### [replaced 014] ALTo: Adaptive-Length Tokenizer for Autoregressive Mask Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16495v2](http://arxiv.org/pdf/2505.16495v2)**

> **作者:** Lingfeng Wang; Hualing Lin; Senda Chen; Tao Wang; Changxu Cheng; Yangyang Zhong; Dong Zheng; Wuyue Zhao
>
> **摘要:** While humans effortlessly draw visual objects and shapes by adaptively allocating attention based on their complexity, existing multimodal large language models (MLLMs) remain constrained by rigid token representations. Bridging this gap, we propose ALTo, an adaptive length tokenizer for autoregressive mask generation. To achieve this, a novel token length predictor is designed, along with a length regularization term and a differentiable token chunking strategy. We further build ALToLLM that seamlessly integrates ALTo into MLLM. Preferences on the trade-offs between mask quality and efficiency is implemented by group relative policy optimization (GRPO). Experiments demonstrate that ALToLLM achieves state-of-the-art performance with adaptive token cost on popular segmentation benchmarks. Code and models are released at https://github.com/yayafengzi/ALToLLM.
>
---
#### [replaced 015] Towards Fine-Grained Text-to-3D Quality Assessment: A Benchmark and A Two-Stage Rank-Learning Metric
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23841v2](http://arxiv.org/pdf/2509.23841v2)**

> **作者:** Bingyang Cui; Yujie Zhang; Qi Yang; Zhu Li; Yiling Xu
>
> **摘要:** Recent advances in Text-to-3D (T23D) generative models have enabled the synthesis of diverse, high-fidelity 3D assets from textual prompts. However, existing challenges restrict the development of reliable T23D quality assessment (T23DQA). First, existing benchmarks are outdated, fragmented, and coarse-grained, making fine-grained metric training infeasible. Moreover, current objective metrics exhibit inherent design limitations, resulting in non-representative feature extraction and diminished metric robustness. To address these limitations, we introduce T23D-CompBench, a comprehensive benchmark for compositional T23D generation. We define five components with twelve sub-components for compositional prompts, which are used to generate 3,600 textured meshes from ten state-of-the-art generative models. A large-scale subjective experiment is conducted to collect 129,600 reliable human ratings across different perspectives. Based on T23D-CompBench, we further propose Rank2Score, an effective evaluator with two-stage training for T23DQA. Rank2Score enhances pairwise training via supervised contrastive regression and curriculum learning in the first stage, and subsequently refines predictions using mean opinion scores to achieve closer alignment with human judgments in the second stage. Extensive experiments and downstream applications demonstrate that Rank2Score consistently outperforms existing metrics across multiple dimensions and can additionally serve as a reward function to optimize generative models. The project is available at https://cbysjtu.github.io/Rank2Score/.
>
---
#### [replaced 016] PLUTO-4: Frontier Pathology Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2511.02826v2](http://arxiv.org/pdf/2511.02826v2)**

> **作者:** Harshith Padigela; Shima Nofallah; Atchuth Naveen Chilaparasetti; Ryun Han; Andrew Walker; Judy Shen; Chintan Shah; Blake Martin; Aashish Sood; Elliot Miller; Ben Glass; Andy Beck; Harsha Pokkalla; Syed Ashar Javed
>
> **摘要:** Foundation models trained on large-scale pathology image corpora have demonstrated strong transfer capabilities across diverse histopathology tasks. Building on this progress, we introduce PLUTO-4, our next generation of pathology foundation models that extend the Pathology-Universal Transformer (PLUTO) to frontier scale. We share two complementary Vision Transformer architectures in the PLUTO-4 family: a compact and efficient PLUTO-4S model optimized for multi-scale deployment using a FlexiViT setup with 2D-RoPE embeddings, and a frontier-scale PLUTO-4G model trained with a single patch size to maximize representation capacity and stability. Both models are pretrained using a self-supervised objective derived from DINOv2 on a large multi-institutional corpus containing 551,164 WSIs from 137,144 patients across over 50 institutions, spanning over 60 disease types and over 100 stains. Comprehensive evaluation across public and internal benchmarks demonstrates that PLUTO-4 achieves state-of-the-art performance on tasks requiring varying spatial and biological context, including patch-level classification, segmentation, and slide-level diagnosis. The compact PLUTO-4S provides high-throughput and robust performance for practical deployment, while PLUTO-4G establishes new performance frontiers across multiple pathology benchmarks, including an 11% improvement in dermatopathology diagnosis. These diverse improvements underscore PLUTO-4's potential to transform real-world applications as a backbone for translational research and diagnostic use cases.
>
---
#### [replaced 017] A Label Propagation Strategy for CutMix in Multi-Label Remote Sensing Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.13451v3](http://arxiv.org/pdf/2405.13451v3)**

> **作者:** Tom Burgert; Kai Norman Clasen; Jonas Klotz; Tim Siebert; Begüm Demir
>
> **备注:** Accepted at IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
>
> **摘要:** The development of supervised deep learning-based methods for multi-label scene classification (MLC) is one of the prominent research directions in remote sensing (RS). However, collecting annotations for large RS image archives is time-consuming and costly. To address this issue, several data augmentation methods have been introduced in RS. Among others, the CutMix data augmentation technique, which combines parts of two existing training images to generate an augmented image, stands out as a particularly effective approach. However, the direct application of CutMix in RS MLC can lead to the erasure or addition of class labels (i.e., label noise) in the augmented (i.e., combined) training image. To address this problem, we introduce a label propagation (LP) strategy that allows the effective application of CutMix in the context of MLC problems in RS without being affected by label noise. To this end, our proposed LP strategy exploits pixel-level class positional information to update the multi-label of the augmented training image. We propose to access such class positional information from reference maps (e.g., thematic products) associated with each training image or from class explanation masks provided by an explanation method if no reference maps are available. Similarly to pairing two training images, our LP strategy carries out a pairing operation on the associated pixel-level class positional information to derive the updated multi-label for the augmented image. Experimental results show the effectiveness of our LP strategy in general (e.g., an improvement of 2% to 4% mAP macro compared to standard CutMix) and its robustness in the case of various simulated and real scenarios with noisy class positional information in particular. Code is available at https://git.tu-berlin.de/rsim/cutmix_lp.
>
---
#### [replaced 018] Breaking Down Monocular Ambiguity: Exploiting Temporal Evolution for 3D Lane Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20525v3](http://arxiv.org/pdf/2504.20525v3)**

> **作者:** Huan Zheng; Wencheng Han; Tianyi Yan; Cheng-zhong Xu; Jianbing Shen
>
> **摘要:** Monocular 3D lane detection aims to estimate the 3D position of lanes from frontal-view (FV) images. However, existing methods are fundamentally constrained by the inherent ambiguity of single-frame input, which leads to inaccurate geometric predictions and poor lane integrity, especially for distant lanes. To overcome this, we propose to unlock the rich information embedded in the temporal evolution of the scene as the vehicle moves. Our proposed Geometry-aware Temporal Aggregation Network (GTA-Net) systematically leverages the temporal information from complementary perspectives. First, Temporal Geometry Enhancement Module (TGEM) learns geometric consistency across consecutive frames, effectively recovering depth information from motion to build a reliable 3D scene representation. Second, to enhance lane integrity, Temporal Instance-aware Query Generation (TIQG) module aggregates instance cues from past and present frames. Crucially, for lanes that are ambiguous in the current view, TIQG innovatively synthesizes a pseudo future perspective to generate queries that reveal lanes which would otherwise be missed. The experiments demonstrate that GTA-Net achieves new SoTA results, significantly outperforming existing monocular 3D lane detection solutions.
>
---
#### [replaced 019] Alleviating Hyperparameter-Tuning Burden in SVM Classifiers for Pulmonary Nodules Diagnosis with Multi-Task Bayesian Optimization
- **分类: eess.IV; cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2411.06184v2](http://arxiv.org/pdf/2411.06184v2)**

> **作者:** Wenhao Chi; Haiping Liu; Hongqiao Dong; Wenhua Liang; Bo Liu
>
> **备注:** 12 pages, 4 figures, 37 references
>
> **摘要:** In the field of non-invasive medical imaging, radiomic features are utilized to measure tumor characteristics. However, these features can be affected by the techniques used to discretize the images, ultimately impacting the accuracy of diagnosis. To investigate the influence of various image discretization methods on diagnosis, it is common practice to evaluate multiple discretization strategies individually. This approach often leads to redundant and time-consuming tasks such as training predictive models and fine-tuning hyperparameters separately. This study examines the feasibility of employing multi-task Bayesian optimization to accelerate the hyperparameters search for classifying benign and malignant pulmonary nodules using RBF SVM. Our findings suggest that multi-task Bayesian optimization significantly accelerates the search for hyperparameters in comparison to a single-task approach. To the best of our knowledge, this is the first investigation to utilize multi-task Bayesian optimization in a critical medical context.
>
---
#### [replaced 020] Exploring Typographic Visual Prompts Injection Threats in Cross-Modality Generation Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11519v4](http://arxiv.org/pdf/2503.11519v4)**

> **作者:** Hao Cheng; Erjia Xiao; Yichi Wang; Lingfeng Zhang; Qiang Zhang; Jiahang Cao; Kaidi Xu; Mengshu Sun; Xiaoshuai Hao; Jindong Gu; Renjing Xu
>
> **备注:** This paper is accepted by IJCAI2025 Workshop on Deepfake Detection, Localization, and Interpretability as Best Student Paper
>
> **摘要:** Current Cross-Modality Generation Models (GMs) demonstrate remarkable capabilities in various generative tasks. Given the ubiquity and information richness of vision modality inputs in real-world scenarios, Cross-Vision tasks, encompassing Vision-Language Perception (VLP) and Image-to-Image (I2I), have attracted significant attention. Large Vision Language Models (LVLMs) and I2I Generation Models (GMs) are employed to handle VLP and I2I tasks, respectively. Previous research indicates that printing typographic words into input images significantly induces LVLMs and I2I GMs to produce disruptive outputs that are semantically aligned with those words. Additionally, visual prompts, as a more sophisticated form of typography, are also revealed to pose security risks to various applications of cross-vision tasks. However, the specific characteristics of the threats posed by visual prompts remain underexplored. In this paper, to comprehensively investigate the performance impact induced by Typographic Visual Prompt Injection (TVPI) in various LVLMs and I2I GMs, we propose the Typographic Visual Prompts Injection Dataset and thoroughly evaluate the TVPI security risks on various open-source and closed-source LVLMs and I2I GMs under visual prompts with different target semantics, deepening the understanding of TVPI threats.
>
---
#### [replaced 021] Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04789v3](http://arxiv.org/pdf/2506.04789v3)**

> **作者:** Gaia Di Lorenzo; Federico Tombari; Marc Pollefeys; Daniel Barath
>
> **摘要:** Learning effective multi-modal 3D representations of objects is essential for numerous applications, such as augmented reality and robotics. Existing methods often rely on task-specific embeddings that are tailored either for semantic understanding or geometric reconstruction. As a result, these embeddings typically cannot be decoded into explicit geometry and simultaneously reused across tasks. In this paper, we propose Object-X, a versatile multi-modal object representation framework capable of encoding rich object embeddings (e.g. images, point cloud, text) and decoding them back into detailed geometric and visual reconstructions. Object-X operates by geometrically grounding the captured modalities in a 3D voxel grid and learning an unstructured embedding fusing the information from the voxels with the object attributes. The learned embedding enables 3D Gaussian Splatting-based object reconstruction, while also supporting a range of downstream tasks, including scene alignment, single-image 3D object reconstruction, and localization. Evaluations on two challenging real-world datasets demonstrate that Object-X produces high-fidelity novel-view synthesis comparable to standard 3D Gaussian Splatting, while significantly improving geometric accuracy. Moreover, Object-X achieves competitive performance with specialized methods in scene alignment and localization. Critically, our object-centric descriptors require 3-4 orders of magnitude less storage compared to traditional image- or point cloud-based approaches, establishing Object-X as a scalable and highly practical solution for multi-modal 3D scene representation.
>
---
#### [replaced 022] MobileGeo: Exploring Hierarchical Knowledge Distillation for Resource-Efficient Cross-view Drone Geo-Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.22582v2](http://arxiv.org/pdf/2510.22582v2)**

> **作者:** Jian Sun; Kangdao Liu; Chi Zhang; Chuangquan Chen; Junge Shen; Chi-Man Vong
>
> **摘要:** Cross-view geo-localization (CVGL) enables drone localization by matching aerial images to geo-tagged satellite databases, which is critical for autonomous navigation in GNSS-denied environments. However, existing methods rely on resource-intensive feature alignment and multi-branch architectures, incurring high inference costs that limit their deployment on mobile edge devices. We propose MobileGeo, a mobile-friendly framework designed for efficient on-device CVGL. MobileGeo achieves its efficiency through two key components: 1) During training, a Hierarchical Distillation (HD-CVGL) paradigm, coupled with Uncertainty-Aware Prediction Alignment (UAPA), distills essential information into a compact model without incurring inference overhead. 2) During inference, an efficient Multi-view Selection Refinement Module (MSRM) leverages mutual information to filter redundant views and reduce computational load. Extensive experiments demonstrate that MobileGeo outperforms previous state-of-the-art methods, achieving a 4.19\% improvement in AP on University-1652 dataset while being over 5$\times$ more efficient in FLOPs and 3$\times$ faster. Crucially, MobileGeo runs at 251.5 FPS on an NVIDIA AGX Orin edge device, demonstrating its practical viability for real-time on-device drone geo-localization.
>
---
#### [replaced 023] Depth Matters: Multimodal RGB-D Perception for Robust Autonomous Agents
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16711v2](http://arxiv.org/pdf/2503.16711v2)**

> **作者:** Mihaela-Larisa Clement; Mónika Farsang; Felix Resch; Mihai-Teodor Stanusoiu; Radu Grosu
>
> **备注:** Submitted to ICRA 2025
>
> **摘要:** Autonomous agents that rely purely on perception to make real-time control decisions require efficient and robust architectures. In this work, we demonstrate that augmenting RGB input with depth information significantly enhances our agents' ability to predict steering commands compared to using RGB alone. We benchmark lightweight recurrent controllers that leverage the fused RGB-D features for sequential decision-making. To train our models, we collect high-quality data using a small-scale autonomous car controlled by an expert driver via a physical steering wheel, capturing varying levels of steering difficulty. Our models were successfully deployed on real hardware and inherently avoided dynamic and static obstacles, under out-of-distribution conditions. Specifically, our findings reveal that the early fusion of depth data results in a highly robust controller, which remains effective even with frame drops and increased noise levels, without compromising the network's focus on the task.
>
---
#### [replaced 024] Revisiting semi-supervised learning in the era of foundation models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09707v4](http://arxiv.org/pdf/2503.09707v4)**

> **作者:** Ping Zhang; Zheda Mai; Quang-Huy Nguyen; Wei-Lun Chao
>
> **备注:** The paper has been accepted to NeurIPS 2025. Ping Zhang and Zheda Mai contributed equally to this work
>
> **摘要:** Semi-supervised learning (SSL) leverages abundant unlabeled data alongside limited labeled data to enhance learning. As vision foundation models (VFMs) increasingly serve as the backbone of vision applications, it remains unclear how SSL interacts with these pre-trained models. To address this gap, we develop new SSL benchmark datasets where frozen VFMs underperform and systematically evaluate representative SSL methods. We make a surprising observation: parameter-efficient fine-tuning (PEFT) using only labeled data often matches SSL performance, even without leveraging unlabeled data. This motivates us to revisit self-training, a conceptually simple SSL baseline, where we use the supervised PEFT model to pseudo-label unlabeled data for further training. To overcome the notorious issue of noisy pseudo-labels, we propose ensembling multiple PEFT approaches and VFM backbones to produce more robust pseudo-labels. Empirical results validate the effectiveness of this simple yet powerful approach, providing actionable insights into SSL with VFMs and paving the way for more scalable and practical semi-supervised learning in the era of foundation models.
>
---
#### [replaced 025] ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.07661v3](http://arxiv.org/pdf/2406.07661v3)**

> **作者:** Anurag Ghosh; Shen Zheng; Robert Tamburo; Khiem Vuong; Juan Alvarez-Padilla; Hailiang Zhu; Michael Cardei; Nicholas Dunn; Christoph Mertz; Srinivasa G. Narasimhan
>
> **备注:** ICCV 2025 Accepted Paper
>
> **摘要:** Perceiving and autonomously navigating through work zones is a challenging and underexplored problem. Open datasets for this long-tailed scenario are scarce. We propose the ROADWork dataset to learn to recognize, observe, analyze, and drive through work zones. State-of-the-art foundation models fail when applied to work zones. Fine-tuning models on our dataset significantly improves perception and navigation in work zones. With ROADWork dataset, we discover new work zone images with higher precision (+32.5%) at a much higher rate (12.8$\times$) around the world. Open-vocabulary methods fail too, whereas fine-tuned detectors improve performance (+32.2 AP). Vision-Language Models (VLMs) struggle to describe work zones, but fine-tuning substantially improves performance (+36.7 SPICE). Beyond fine-tuning, we show the value of simple techniques. Video label propagation provides additional gains (+2.6 AP) for instance segmentation. While reading work zone signs, composing a detector and text spotter via crop-scaling improves performance +14.2% 1-NED). Composing work zone detections to provide context further reduces hallucinations (+3.9 SPICE) in VLMs. We predict navigational goals and compute drivable paths from work zone videos. Incorporating road work semantics ensures 53.6% goals have angular error (AE) < 0.5 (+9.9 %) and 75.3% pathways have AE < 0.5 (+8.1 %).
>
---
#### [replaced 026] Transfer Learning-based Real-time Handgun Detection
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2311.13559v3](http://arxiv.org/pdf/2311.13559v3)**

> **作者:** Youssef Elmir
>
> **备注:** 16 pages, 9 figures, and 3 tables. published at The Iraqi Journal of Science, issued by College of Science at University of Baghdad
>
> **摘要:** Traditional surveillance systems rely on human attention, limiting their effectiveness. This study employs convolutional neural networks and transfer learning to develop a real-time computer vision system for automatic handgun detection. Comprehensive analysis of online handgun detection methods is conducted, emphasizing reducing false positives and learning time. Transfer learning is demonstrated as an effective approach. Despite technical challenges, the proposed system achieves a precision rate of 84.74%, demonstrating promising performance comparable to related works, enabling faster learning and accurate automatic handgun detection for enhanced security. This research advances security measures by reducing human monitoring dependence, showcasing the potential of transfer learning-based approaches for efficient and reliable handgun detection.
>
---
#### [replaced 027] Hulu-Med: A Transparent Generalist Model towards Holistic Medical Vision-Language Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.08668v2](http://arxiv.org/pdf/2510.08668v2)**

> **作者:** Songtao Jiang; Yuan Wang; Sibo Song; Tianxiang Hu; Chenyi Zhou; Bin Pu; Yan Zhang; Zhibo Yang; Yang Feng; Joey Tianyi Zhou; Jin Hao; Zijian Chen; Ruijia Wu; Tao Tang; Junhui Lv; Hongxia Xu; Hongwei Wang; Jun Xiao; Bin Feng; Fudong Zhu; Kenli Li; Weidi Xie; Jimeng Sun; Jian Wu; Zuozhu Liu
>
> **摘要:** Real-world clinical decision-making requires integrating heterogeneous data, including medical text, 2D images, 3D volumes, and videos, while existing AI systems fail to unify all these signals, limiting their utility. In this paper, we introduce Hulu-Med, a transparent, generalist medical Vision-Language Model (VLM) designed to unify language-only, 2D/3D vision-language, and video understanding within a single architecture. Hulu-Med is trained on a curated corpus of 16.7 million samples, comprising exclusively public or synthetic data, spanning 12 major anatomical systems and 14 medical imaging modalities. Hulu-Med employs a medical-aware token-reduction strategy that prunes redundant visual tokens, achieving up to a 55% reduction for 3D and video inputs, improving cross-modal efficiency, and enabling training at 7B-32B parameter scales in approximately 4,000-40,000 GPU hours. Across 30 public in-domain and out-of-domain medical benchmarks-covering text reasoning, visual question answering, report generation, multilingual dialogue, video understanding, and rare disease diagnosis-Hulu-Med surpasses existing open-source models on 27 of 30 benchmarks and outperforms proprietary systems such as GPT-4o on 16 benchmarks. Despite being a VLM, Hulu-Med outperforms GPT-4o and matches GPT-o1 on the text-only HealthBench. For the first time in the community, we provide a fully transparent, reproducible and cost-effective pipeline for holistic medical vision-language understanding by releasing our end-to-end data curation, training procedures, and model parameters. Code and models are available at https://github.com/ZJUI-AI4H/Hulu-Med.
>
---
#### [replaced 028] Automatic Road Subsurface Distress Recognition from Ground Penetrating Radar Images using Deep Learning-based Cross-verification
- **分类: cs.CV; cs.AI; I.4.9; I.5.4; J.2**

- **链接: [http://arxiv.org/pdf/2507.11081v2](http://arxiv.org/pdf/2507.11081v2)**

> **作者:** Chang Peng; Bao Yang; Meiqi Li; Ge Zhang; Hui Sun; Zhenyu Jiang
>
> **摘要:** Ground penetrating radar (GPR) has become a rapid and non-destructive solution for road subsurface distress (RSD) detection. Deep learning-based automatic RSD recognition, though ameliorating the burden of data processing, suffers from data scarcity and insufficient capability to recognize defects. In this study, a rigorously validated 3D GPR dataset containing 2134 samples of diverse types was constructed through field scanning. A novel cross-verification strategy was proposed to fully exploit the complementary abilities of region proposal networks in object recognition from different views of GPR images. The method achieves outstanding accuracy with a recall over 98.6% in field tests. The approach, integrated into an online RSD detection system, can reduce the human labor of inspection by around 90%.
>
---
#### [replaced 029] ESA: Energy-Based Shot Assembly Optimization for Automatic Video Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2511.02505v2](http://arxiv.org/pdf/2511.02505v2)**

> **作者:** Yaosen Chen; Wei Wang; Tianheng Zheng; Xuming Wen; Han Yang; Yanru Zhang
>
> **摘要:** Shot assembly is a crucial step in film production and video editing, involving the sequencing and arrangement of shots to construct a narrative, convey information, or evoke emotions. Traditionally, this process has been manually executed by experienced editors. While current intelligent video editing technologies can handle some automated video editing tasks, they often fail to capture the creator's unique artistic expression in shot assembly. To address this challenge, we propose an energy-based optimization method for video shot assembly. Specifically, we first perform visual-semantic matching between the script generated by a large language model and a video library to obtain subsets of candidate shots aligned with the script semantics. Next, we segment and label the shots from reference videos, extracting attributes such as shot size, camera motion, and semantics. We then employ energy-based models to learn from these attributes, scoring candidate shot sequences based on their alignment with reference styles. Finally, we achieve shot assembly optimization by combining multiple syntax rules, producing videos that align with the assembly style of the reference videos. Our method not only automates the arrangement and combination of independent shots according to specific logic, narrative requirements, or artistic styles but also learns the assembly style of reference videos, creating a coherent visual sequence or holistic visual expression. With our system, even users with no prior video editing experience can create visually compelling videos. Project page: https://sobeymil.github.io/esa.com
>
---
#### [replaced 030] Erasing 'Ugly' from the Internet: Propagation of the Beauty Myth in Text-Image Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2511.00749v2](http://arxiv.org/pdf/2511.00749v2)**

> **作者:** Tanvi Dinkar; Aiqi Jiang; Gavin Abercrombie; Ioannis Konstas
>
> **备注:** This is a preprint under review
>
> **摘要:** Social media has exacerbated the promotion of Western beauty norms, leading to negative self-image, particularly in women and girls, and causing harm such as body dysmorphia. Increasingly content on the internet has been artificially generated, leading to concerns that these norms are being exaggerated. The aim of this work is to study how generative AI models may encode 'beauty' and erase 'ugliness', and discuss the implications of this for society. To investigate these aims, we create two image generation pipelines: a text-to-image model and a text-to-language model-to image model. We develop a structured beauty taxonomy which we use to prompt three language models (LMs) and two text-to-image models to cumulatively generate 5984 images using our two pipelines. We then recruit women and non-binary social media users to evaluate 1200 of the images through a Likert-scale within-subjects study. Participants show high agreement in their ratings. Our results show that 86.5% of generated images depicted people with lighter skin tones, 22% contained explicit content despite Safe for Work (SFW) training, and 74% were rated as being in a younger age demographic. In particular, the images of non-binary individuals were rated as both younger and more hypersexualised, indicating troubling intersectional effects. Notably, prompts encoded with 'negative' or 'ugly' beauty traits (such as "a wide nose") consistently produced higher Not SFW (NSFW) ratings regardless of gender. This work sheds light on the pervasive demographic biases related to beauty standards present in generative AI models -- biases that are actively perpetuated by model developers, such as via negative prompting. We conclude by discussing the implications of this on society, which include pollution of the data streams and active erasure of features that do not fall inside the stereotype of what is considered beautiful by developers.
>
---
#### [replaced 031] MAROON: A Framework for the Joint Characterization of Near-Field High-Resolution Radar and Optical Depth Imaging Techniques
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.00527v3](http://arxiv.org/pdf/2411.00527v3)**

> **作者:** Vanessa Wirth; Johanna Bräunig; Nikolai Hofmann; Martin Vossiek; Tim Weyrich; Marc Stamminger
>
> **摘要:** Utilizing the complementary strengths of wavelength-specific range or depth sensors is crucial for robust computer-assisted tasks such as autonomous driving. Despite this, there is still little research done at the intersection of optical depth sensors and radars operating close range, where the target is decimeters away from the sensors. Together with a growing interest in high-resolution imaging radars operating in the near field, the question arises how these sensors behave in comparison to their traditional optical counterparts. In this work, we take on the unique challenge of jointly characterizing depth imagers from both, the optical and radio-frequency domain using a multimodal spatial calibration. We collect data from four depth imagers, with three optical sensors of varying operation principle and an imaging radar. We provide a comprehensive evaluation of their depth measurements with respect to distinct object materials, geometries, and object-to-sensor distances. Specifically, we reveal scattering effects of partially transmissive materials and investigate the response of radio-frequency signals. All object measurements will be made public in form of a multimodal dataset, called MAROON.
>
---
#### [replaced 032] FUSAR-KLIP: Towards Multimodal Foundation Models for Remote Sensing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23927v2](http://arxiv.org/pdf/2509.23927v2)**

> **作者:** Yi Yang; Xiaokun Zhang; Qingchen Fang; Jing Liu; Ziqi Ye; Rui Li; Li Liu; Haipeng Wang
>
> **摘要:** Cross-modal artificial intelligence has garnered widespread attention in recent years, achieving significant progress in the study of natural images. However, existing methods are mostly designed for RGB imagery, leaving a significant gap in modeling synthetic aperture radar (SAR) imagery. SAR, with its all-day, all-weather imaging capabilities, plays an irreplaceable role in remote sensing scene understanding. To address this gap, this paper proposes FUSAR-KLIP, the first universal SAR multimodal foundational model, along with reusable data and evaluation baselines. Specifically: (1) This work introduces the critical yet long-overlooked attribute of geographic information into remote sensing research, constructing FUSAR-GEOVL-1M (the first large-scale SAR dataset with complete geographic projection properties), covering multiple satellite platforms, 120,000 images, and 135 cities. (2) Aligned structured text is generated through a hierarchical cognitive chain-of-thought (HCoT), providing more than one million multi-dimensional semantic annotations of landforms, regional functions, target attributes, and spatial relationships. (3) We design a Self-Consistent Iterative Optimization mechanism that continuously enhances cross-modal alignment through a self-supervised closed loop of contrastive, matching, and reconstruction learning on a transferable multimodal encoder. (4) A unified evaluation benchmark is established across 11 representative downstream vision and vision-language tasks, with comparisons against 14 leading foundation models, where FUSAR-KLIP demonstrates leading performance, particularly in object counting and land-cover classification. We expect that FUSAR-KLIP's large-scale multimodal data, transferable model architecture, and comprehensive experimental benchmark will significantly advance the development of SAR multimodal baseline models.
>
---
#### [replaced 033] Interpretable Tile-Based Classification of Paclitaxel Exposure
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23363v2](http://arxiv.org/pdf/2510.23363v2)**

> **作者:** Sean Fletcher; Gabby Scott; Douglas Currie; Xin Zhang; Yuqi Song; Bruce MacLeod
>
> **摘要:** Medical image analysis is central to drug discovery and preclinical evaluation, where scalable, objective readouts can accelerate decision-making. We address classification of paclitaxel (Taxol) exposure from phase-contrast microscopy of C6 glioma cells -- a task with subtle dose differences that challenges full-image models. We propose a simple tiling-and-aggregation pipeline that operates on local patches and combines tile outputs into an image label, achieving state-of-the-art accuracy on the benchmark dataset and improving over the published baseline by around 20 percentage points, with trends confirmed by cross-validation. To understand why tiling is effective, we further apply Grad-CAM and Score-CAM and attention analyses, which enhance model interpretability and point toward robustness-oriented directions for future medical image research. Code is released to facilitate reproduction and extension.
>
---
#### [replaced 034] Revisiting Multimodal Positional Encoding in Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23095v2](http://arxiv.org/pdf/2510.23095v2)**

> **作者:** Jie Huang; Xuejing Liu; Sibo Song; Ruibing Hou; Hong Chang; Junyang Lin; Shuai Bai
>
> **备注:** 16 pages
>
> **摘要:** Multimodal position encoding is essential for vision-language models, yet there has been little systematic investigation into multimodal position encoding. We conduct a comprehensive analysis of multimodal Rotary Positional Embedding (RoPE) by examining its two core components: position design and frequency allocation. Through extensive experiments, we identify three key guidelines: positional coherence, full frequency utilization, and preservation of textual priors-ensuring unambiguous layout, rich representation, and faithful transfer from the pre-trained LLM. Based on these insights, we propose Multi-Head RoPE (MHRoPE) and MRoPE-Interleave (MRoPE-I), two simple and plug-and-play variants that require no architectural changes. Our methods consistently outperform existing approaches across diverse benchmarks, with significant improvements in both general and fine-grained multimodal understanding. Code will be avaliable at https://github.com/JJJYmmm/Multimodal-RoPEs.
>
---
#### [replaced 035] ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.23734v3](http://arxiv.org/pdf/2505.23734v3)**

> **作者:** Weijie Wang; Donny Y. Chen; Zeyu Zhang; Duochao Shi; Akide Liu; Bohan Zhuang
>
> **备注:** NeurIPS 2025, Project Page: https://lhmd.top/zpressor, Code: https://github.com/ziplab/ZPressor
>
> **摘要:** Feed-forward 3D Gaussian Splatting (3DGS) models have recently emerged as a promising solution for novel view synthesis, enabling one-pass inference without the need for per-scene 3DGS optimization. However, their scalability is fundamentally constrained by the limited capacity of their models, leading to degraded performance or excessive memory consumption as the number of input views increases. In this work, we analyze feed-forward 3DGS frameworks through the lens of the Information Bottleneck principle and introduce ZPressor, a lightweight architecture-agnostic module that enables efficient compression of multi-view inputs into a compact latent state $Z$ that retains essential scene information while discarding redundancy. Concretely, ZPressor enables existing feed-forward 3DGS models to scale to over 100 input views at 480P resolution on an 80GB GPU, by partitioning the views into anchor and support sets and using cross attention to compress the information from the support views into anchor views, forming the compressed latent state $Z$. We show that integrating ZPressor into several state-of-the-art feed-forward 3DGS models consistently improves performance under moderate input views and enhances robustness under dense view settings on two large-scale benchmarks DL3DV-10K and RealEstate10K. The video results, code and trained models are available on our project page: https://lhmd.top/zpressor.
>
---
#### [replaced 036] Harmonious Color Pairings: Insights from Human Preference and Natural Hue Statistics
- **分类: cs.HC; cs.CV; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2508.15777v2](http://arxiv.org/pdf/2508.15777v2)**

> **作者:** Ortensia Forni; Alexandre Darmon; Michael Benzaquen
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** While color harmony has long been studied in art and design, a clear consensus remains elusive, as most models are grounded in qualitative insights or limited datasets. In this work, we present a quantitative, data-driven study of color pairing preferences using controlled hue-based palettes in the HSL color space. Participants evaluated combinations of thirteen distinct hues, enabling us to construct a preference matrix and define a combinability index for each color. Our results reveal that preferences are highly hue dependent, challenging the assumption of universal harmony rules proposed in the literature. Yet, when averaged over hues, statistically meaningful patterns of aesthetic preference emerge, with certain hue separations perceived as more harmonious. Strikingly, these patterns align with hue distributions found in natural landscapes, pointing to a statistical correspondence between human color preferences and the structure of color in nature. Finally, we analyze our color-pairing score matrix through principal component analysis, which uncovers two complementary hue groups whose interplay underlies the global structure of color-pairing preferences. Together, these findings offer a quantitative framework for studying color harmony and its potential perceptual and ecological underpinnings.
>
---
#### [replaced 037] Towards 1000-fold Electron Microscopy Image Compression for Connectomics via VQ-VAE with Transformer Prior
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2511.00231v2](http://arxiv.org/pdf/2511.00231v2)**

> **作者:** Fuming Yang; Yicong Li; Hanspeter Pfister; Jeff W. Lichtman; Yaron Meirovitch
>
> **摘要:** Petascale electron microscopy (EM) datasets push storage, transfer, and downstream analysis toward their current limits. We present a vector-quantized variational autoencoder-based (VQ-VAE) compression framework for EM that spans 16x to 1024x and enables pay-as-you-decode usage: top-only decoding for extreme compression, with an optional Transformer prior that predicts bottom tokens (without changing the compression ratio) to restore texture via feature-wise linear modulation (FiLM) and concatenation; we further introduce an ROI-driven workflow that performs selective high-resolution reconstruction from 1024x-compressed latents only where needed.
>
---
#### [replaced 038] CLIP Meets Diffusion: A Synergistic Approach to Anomaly Detection
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.11772v3](http://arxiv.org/pdf/2506.11772v3)**

> **作者:** Byeongchan Lee; John Won; Seunghyun Lee; Jinwoo Shin
>
> **备注:** Accepted at TMLR 2025
>
> **摘要:** Anomaly detection is a complex problem due to the ambiguity in defining anomalies, the diversity of anomaly types (e.g., local and global defect), and the scarcity of training data. As such, it necessitates a comprehensive model capable of capturing both low-level and high-level features, even with limited data. To address this, we propose CLIPFUSION, a method that leverages both discriminative and generative foundation models. Specifically, the CLIP-based discriminative model excels at capturing global features, while the diffusion-based generative model effectively captures local details, creating a synergistic and complementary approach. Notably, we introduce a methodology for utilizing cross-attention maps and feature maps extracted from diffusion models specifically for anomaly detection. Experimental results on benchmark datasets (MVTec-AD, VisA) demonstrate that CLIPFUSION consistently outperforms baseline methods, achieving outstanding performance in both anomaly segmentation and classification. We believe that our method underscores the effectiveness of multi-modal and multi-model fusion in tackling the multifaceted challenges of anomaly detection, providing a scalable solution for real-world applications.
>
---
#### [replaced 039] WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.26125v2](http://arxiv.org/pdf/2510.26125v2)**

> **作者:** Runsheng Xu; Hubert Lin; Wonseok Jeon; Hao Feng; Yuliang Zou; Liting Sun; John Gorman; Kate Tolstaya; Sarah Tang; Brandyn White; Ben Sapp; Mingxing Tan; Jyh-Jing Hwang; Dragomir Anguelov
>
> **摘要:** Vision-based end-to-end (E2E) driving has garnered significant interest in the research community due to its scalability and synergy with multimodal large language models (MLLMs). However, current E2E driving benchmarks primarily feature nominal scenarios, failing to adequately test the true potential of these systems. Furthermore, existing open-loop evaluation metrics often fall short in capturing the multi-modal nature of driving or effectively evaluating performance in long-tail scenarios. To address these gaps, we introduce the Waymo Open Dataset for End-to-End Driving (WOD-E2E). WOD-E2E contains 4,021 driving segments (approximately 12 hours), specifically curated for challenging long-tail scenarios that that are rare in daily life with an occurring frequency of less than 0.03%. Concretely, each segment in WOD-E2E includes the high-level routing information, ego states, and 360-degree camera views from 8 surrounding cameras. To evaluate the E2E driving performance on these long-tail situations, we propose a novel open-loop evaluation metric: Rater Feedback Score (RFS). Unlike conventional metrics that measure the distance between predicted way points and the logs, RFS measures how closely the predicted trajectory matches rater-annotated trajectory preference labels. We have released rater preference labels for all WOD-E2E validation set segments, while the held out test set labels have been used for the 2025 WOD-E2E Challenge. Through our work, we aim to foster state of the art research into generalizable, robust, and safe end-to-end autonomous driving agents capable of handling complex real-world situations.
>
---
#### [replaced 040] ViFP: A Framework for Visual False Positive Detection to Enhance Reasoning Reliability in VLMs
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.04201v2](http://arxiv.org/pdf/2508.04201v2)**

> **作者:** Ben Zhang; LuLu Yu; Lei Gao; QuanJiang Guo; Jing Liu; Hui Gao
>
> **摘要:** During reasoning in vision-language models (VLMs), false positive (FP) reasoning occurs when a model produces the correct answer but follows an incorrect reasoning path, resulting in undermined reasoning reliability. Existing approaches mainly rely on prompt engineering, knowledge distillation or reinforcement learning to improve reasoning reliability, both of which require large amounts of high-quality data and thus limit practical applicability. Few approaches have focused on directly detecting and correcting FPs. To address these issues, we propose ViFP, a framework for Visual False Positive Detection to Enhance Reasoning Reliability in VLMs. ViFP builds effective reasoning paths through multi-turn QA and dynamically analyzes the consistency of the reasoning path to identify potential FPs. It also introduces a targeted reasoning chain correction mechanism to modify FP reasoning, thereby improving logical consistency and accuracy. Finally, we introduce a reliability evaluation metric, VoC, which integrates answer accuracy and the FP rate, providing a quantitative tool to assess whether a VLM not only answers correctly but also reasons reliably. Our experiments on closed-source VLMs show that ViFP consistently improves performance across three datasets: A-OKVQA, OK-VQA, and FVQA. On A-OKVQA, ViFP improves accuracy by up to 5.4%, surpassing the previous state-of-the-art by 4.3%, and significantly reduces the number of FPs, validating its benefits in enhancing reasoning reliability.
>
---
#### [replaced 041] Human Perception-Inspired Grain Segmentation Refinement Using Conditional Random Fields
- **分类: cond-mat.mtrl-sci; cs.CV**

- **链接: [http://arxiv.org/pdf/2312.09968v3](http://arxiv.org/pdf/2312.09968v3)**

> **作者:** Doruk Aksoy; Huolin L. Xin; Timothy J. Rupert; William J. Bowman
>
> **备注:** v3 = published version (OA, CC BY 4.0)
>
> **摘要:** Automated detection of grain boundaries (GBs) in electron microscope images of polycrystalline materials could help accelerate the nanoscale characterization of myriad engineering materials and novel materials under scientific research. Accurate segmentation of interconnected line networks, such as GBs in polycrystalline material microstructures, poses a significant challenge due to the fragmented masks produced by conventional computer vision (CV) algorithms, including convolutional neural networks. These algorithms struggle with thin masks, often necessitating post-processing for effective contour closure and continuity. Previous approaches in this domain have typically relied on custom post-processing techniques that are problem-specific and heavily dependent on the quality of the mask obtained from a CV algorithm. Addressing this issue, this paper introduces a fast, high-fidelity post-processing technique that is universally applicable to segmentation masks of interconnected line networks. Leveraging domain knowledge about grain boundary connectivity, this method employs conditional random fields and perceptual grouping rules to refine segmentation masks of any image with a discernible grain structure. This approach significantly enhances segmentation mask accuracy by correctly reconstructing fragmented GBs in electron microscopy images of a polycrystalline oxide. The refinement improves the statistical representation of the microstructure, reflected by a 51 % improvement in a grain alignment metric that provides a more physically meaningful assessment of complex microstructures than conventional metrics. This method enables rapid and accurate characterization, facilitating an unprecedented level of data analysis and improving the understanding of GB networks, making it suitable for a range of disciplines where precise segmentation of interconnected line networks is essential.
>
---
#### [replaced 042] Generative View Stitching
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.24718v2](http://arxiv.org/pdf/2510.24718v2)**

> **作者:** Chonghyuk Song; Michal Stary; Boyuan Chen; George Kopanas; Vincent Sitzmann
>
> **备注:** Updated acknowledgements and fixed figure visibility issue on Safari. Project website: https://andrewsonga.github.io/gvs
>
> **摘要:** Autoregressive video diffusion models are capable of long rollouts that are stable and consistent with history, but they are unable to guide the current generation with conditioning from the future. In camera-guided video generation with a predefined camera trajectory, this limitation leads to collisions with the generated scene, after which autoregression quickly collapses. To address this, we propose Generative View Stitching (GVS), which samples the entire sequence in parallel such that the generated scene is faithful to every part of the predefined camera trajectory. Our main contribution is a sampling algorithm that extends prior work on diffusion stitching for robot planning to video generation. While such stitching methods usually require a specially trained model, GVS is compatible with any off-the-shelf video model trained with Diffusion Forcing, a prevalent sequence diffusion framework that we show already provides the affordances necessary for stitching. We then introduce Omni Guidance, a technique that enhances the temporal consistency in stitching by conditioning on both the past and future, and that enables our proposed loop-closing mechanism for delivering long-range coherence. Overall, GVS achieves camera-guided video generation that is stable, collision-free, frame-to-frame consistent, and closes loops for a variety of predefined camera paths, including Oscar Reutersv\"ard's Impossible Staircase. Results are best viewed as videos at https://andrewsonga.github.io/gvs.
>
---
#### [replaced 043] Voost: A Unified and Scalable Diffusion Transformer for Bidirectional Virtual Try-On and Try-Off
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.04825v2](http://arxiv.org/pdf/2508.04825v2)**

> **作者:** Seungyong Lee; Jeong-gi Kwak
>
> **备注:** Accepted to SIGGRAPH Asia 2025, project page: https://nxnai.github.io/Voost/
>
> **摘要:** Virtual try-on aims to synthesize a realistic image of a person wearing a target garment, but accurately modeling garment-body correspondence remains a persistent challenge, especially under pose and appearance variation. In this paper, we propose Voost - a unified and scalable framework that jointly learns virtual try-on and try-off with a single diffusion transformer. By modeling both tasks jointly, Voost enables each garment-person pair to supervise both directions and supports flexible conditioning over generation direction and garment category, enhancing garment-body relational reasoning without task-specific networks, auxiliary losses, or additional labels. In addition, we introduce two inference-time techniques: attention temperature scaling for robustness to resolution or mask variation, and self-corrective sampling that leverages bidirectional consistency between tasks. Extensive experiments demonstrate that Voost achieves state-of-the-art results on both try-on and try-off benchmarks, consistently outperforming strong baselines in alignment accuracy, visual fidelity, and generalization.
>
---
#### [replaced 044] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04220v3](http://arxiv.org/pdf/2506.04220v3)**

> **作者:** Fangrui Zhu; Hanhui Wang; Yiming Xie; Jing Gu; Tianye Ding; Jianwei Yang; Huaizu Jiang
>
> **备注:** NeurIPS 2025, code link: https://github.com/neu-vi/struct2d
>
> **摘要:** Unlocking spatial reasoning in Multimodal Large Language Models (MLLMs) is crucial for enabling intelligent interaction with 3D environments. While prior efforts often rely on explicit 3D inputs or specialized model architectures, we ask: can MLLMs reason about 3D space using only structured 2D representations derived from perception? We introduce Struct2D, a perception-guided prompting framework that combines bird's-eye-view (BEV) images with object marks and object-centric metadata, optionally incorporating egocentric keyframes when needed. Using Struct2D, we conduct an in-depth zero-shot analysis of closed-source MLLMs (e.g., GPT-o3) and find that they exhibit surprisingly strong spatial reasoning abilities when provided with structured 2D inputs, effectively handling tasks such as relative direction estimation and route planning. Building on these insights, we construct Struct2D-Set, a large-scale instruction tuning dataset with 200K fine-grained QA pairs across eight spatial reasoning categories, generated automatically from 3D indoor scenes. We fine-tune an open-source MLLM (Qwen2.5VL) on Struct2D-Set, achieving competitive performance on multiple benchmarks, including 3D question answering, dense captioning, and object grounding. Our approach demonstrates that structured 2D inputs can effectively bridge perception and language reasoning in MLLMs-without requiring explicit 3D representations as input. We will release both our code and dataset to support future research.
>
---
#### [replaced 045] Disentanglement with Factor Quantized Variational Autoencoders
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.14851v3](http://arxiv.org/pdf/2409.14851v3)**

> **作者:** Gulcin Baykal; Melih Kandemir; Gozde Unal
>
> **备注:** Accepted to Neurocomputing
>
> **摘要:** Disentangled representation learning aims to represent the underlying generative factors of a dataset in a latent representation independently of one another. In our work, we propose a discrete variational autoencoder (VAE) based model where the ground truth information about the generative factors are not provided to the model. We demonstrate the advantages of learning discrete representations over learning continuous representations in facilitating disentanglement. Furthermore, we propose incorporating an inductive bias into the model to further enhance disentanglement. Precisely, we propose scalar quantization of the latent variables in a latent representation with scalar values from a global codebook, and we add a total correlation term to the optimization as an inductive bias. Our method called FactorQVAE combines optimization based disentanglement approaches with discrete representation learning, and it outperforms the former disentanglement methods in terms of two disentanglement metrics (DCI and InfoMEC) while improving the reconstruction performance. Our code can be found at https://github.com/ituvisionlab/FactorQVAE.
>
---
#### [replaced 046] FusionRF: High-Fidelity Satellite Neural Radiance Fields from Multispectral and Panchromatic Acquisitions
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2409.15132v3](http://arxiv.org/pdf/2409.15132v3)**

> **作者:** Michael Sprintson; Rama Chellappa; Cheng Peng
>
> **摘要:** We introduce FusionRF, a novel framework for digital surface reconstruction from satellite multispectral and panchromatic images. Current work has demonstrated the increased accuracy of neural photogrammetry for surface reconstruction from optical satellite images compared to algorithmic methods. Common satellites produce both a panchromatic and multispectral image, which contain high spatial and spectral information respectively. Current neural reconstruction methods require multispectral images to be upsampled with a pansharpening method using the spatial data in the panchromatic image. However, these methods may introduce biases and hallucinations due to domain gaps. FusionRF introduces joint image fusion during optimization through a novel cross-resolution kernel that learns to resolve spatial resolution loss present in multispectral images. As input, FusionRF accepts the original multispectral and panchromatic data, eliminating the need for image preprocessing. FusionRF also leverages multimodal appearance embeddings that encode the image characteristics of each modality and view within a uniform representation. By optimizing on both modalities, FusionRF learns to fuse image modalities while performing reconstruction tasks and eliminates the need for a pansharpening preprocessing step. We evaluate our method on multispectral and panchromatic satellite images from the WorldView-3 satellite in various locations, and show that FusionRF provides an average of 17% reduction in depth reconstruction error, and renders sharp training and novel views.
>
---
#### [replaced 047] A New Comprehensive Framework for Multi-Exposure Stereo Coding Utilizing Low Rank Tucker-ALS and 3D-HEVC Techniques
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2104.04726v2](http://arxiv.org/pdf/2104.04726v2)**

> **作者:** Mansi Sharma; Jyotsana Grover
>
> **摘要:** Display technology must offer high dynamic range (HDR) contrast-based depth induction and 3D personalization simultaneously. Efficient algorithms to compress HDR stereo data is critical. Direct capturing of HDR content is complicated due to the high expense and scarcity of HDR cameras. The HDR 3D images could be generated in low-cost by fusing low-dynamic-range (LDR) images acquired using a stereo camera with various exposure settings. In this paper, an efficient scheme for coding multi-exposure stereo images is proposed based on a tensor low-rank approximation scheme. The multi-exposure fusion can be realized to generate HDR stereo output at the decoder for increased realism and exaggerated binocular 3D depth cues. For exploiting spatial redundancy in LDR stereo images, the stack of multi-exposure stereo images is decomposed into a set of projection matrices and a core tensor following an alternating least squares Tucker decomposition model. The compact, low-rank representation of the scene, thus, generated is further processed by 3D extension of High Efficiency Video Coding standard. The encoding with 3D-HEVC enhance the proposed scheme efficiency by exploiting intra-frame, inter-view and the inter-component redundancies in low-rank approximated representation. We consider constant luminance property of IPT and Y'CbCr color space to precisely approximate intensity prediction and perceptually minimize the encoding distortion. Besides, the proposed scheme gives flexibility to adjust the bitrate of tensor latent components by changing the rank of core tensor and its quantization. Extensive experiments on natural scenes demonstrate that the proposed scheme outperforms state-of-the-art JPEG-XT and 3D-HEVC range coding standards.
>
---
#### [replaced 048] Med-Banana-50K: A Cross-modality Large-Scale Dataset for Text-guided Medical Image Editing
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2511.00801v2](http://arxiv.org/pdf/2511.00801v2)**

> **作者:** Zhihui Chen; Mengling Feng
>
> **摘要:** Recent advances in multimodal large language models have enabled remarkable medical image editing capabilities. However, the research community's progress remains constrained by the absence of large-scale, high-quality, and openly accessible datasets built specifically for medical image editing with strict anatomical and clinical constraints. We introduce Med-Banana-50K, a comprehensive 50K-image dataset for instruction-based medical image editing spanning three modalities (chest X-ray, brain MRI, fundus photography) and 23 disease types. Our dataset is constructed by leveraging Gemini-2.5-Flash-Image to generate bidirectional edits (lesion addition and removal) from real medical images. What distinguishes Med-Banana-50K from general-domain editing datasets is our systematic approach to medical quality control: we employ LLM-as-Judge with a medically grounded rubric (instruction compliance, structural plausibility, realism, and fidelity preservation) and history-aware iterative refinement up to five rounds. Beyond single-turn editing, Med-Banana-50K includes 37K failed attempts with full conversation logs for preference learning and alignment research. By providing this large-scale, medically validated, and fully documented resource, Med-Banana-50K establishes a foundation for training and evaluating the next generation of medical image editing models.Our dataset and code are publicly available at [https://github.com/richardChenzhihui/med-banana-50k].
>
---
#### [replaced 049] MagCache: Fast Video Generation with Magnitude-Aware Cache
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09045v2](http://arxiv.org/pdf/2506.09045v2)**

> **作者:** Zehong Ma; Longhui Wei; Feng Wang; Shiliang Zhang; Qi Tian
>
> **备注:** Project Page: https://zehong-ma.github.io/MagCache Accepted by NeurIPS 2025
>
> **摘要:** Existing acceleration techniques for video diffusion models often rely on uniform heuristics or time-embedding variants to skip timesteps and reuse cached features. These approaches typically require extensive calibration with curated prompts and risk inconsistent outputs due to prompt-specific overfitting. In this paper, we introduce a novel and robust discovery: a unified magnitude law observed across different models and prompts. Specifically, the magnitude ratio of successive residual outputs decreases monotonically, steadily in most timesteps while rapidly in the last several steps. Leveraging this insight, we introduce a Magnitude-aware Cache (MagCache) that adaptively skips unimportant timesteps using an error modeling mechanism and adaptive caching strategy. Unlike existing methods requiring dozens of curated samples for calibration, MagCache only requires a single sample for calibration. Experimental results show that MagCache achieves 2.10x-2.68x speedups on Open-Sora, CogVideoX, Wan 2.1, and HunyuanVideo, while preserving superior visual fidelity. It significantly outperforms existing methods in LPIPS, SSIM, and PSNR, under similar computational budgets.
>
---
#### [replaced 050] Stable Part Diffusion 4D: Multi-View RGB and Kinematic Parts Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.10687v2](http://arxiv.org/pdf/2509.10687v2)**

> **作者:** Hao Zhang; Chun-Han Yao; Simon Donné; Narendra Ahuja; Varun Jampani
>
> **备注:** Page: https://stablepartdiffusion4d.github.io/
>
> **摘要:** We present Stable Part Diffusion 4D (SP4D), a framework for generating paired RGB and kinematic part videos from monocular inputs. Unlike conventional part segmentation methods that rely on appearance-based semantic cues, SP4D learns to produce kinematic parts - structural components aligned with object articulation and consistent across views and time. SP4D adopts a dual-branch diffusion model that jointly synthesizes RGB frames and corresponding part segmentation maps. To simplify the architecture and flexibly enable different part counts, we introduce a spatial color encoding scheme that maps part masks to continuous RGB-like images. This encoding allows the segmentation branch to share the latent VAE from the RGB branch, while enabling part segmentation to be recovered via straightforward post-processing. A Bidirectional Diffusion Fusion (BiDiFuse) module enhances cross-branch consistency, supported by a contrastive part consistency loss to promote spatial and temporal alignment of part predictions. We demonstrate that the generated 2D part maps can be lifted to 3D to derive skeletal structures and harmonic skinning weights with few manual adjustments. To train and evaluate SP4D, we construct KinematicParts20K, a curated dataset of over 20K rigged objects selected and processed from Objaverse XL (Deitke et al., 2023), each paired with multi-view RGB and part video sequences. Experiments show that SP4D generalizes strongly to diverse scenarios, including real-world videos, novel generated objects, and rare articulated poses, producing kinematic-aware outputs suitable for downstream animation and motion-related tasks.
>
---
#### [replaced 051] Reg-DPO: SFT-Regularized Direct Preference Optimization with GT-Pair for Improving Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2511.01450v2](http://arxiv.org/pdf/2511.01450v2)**

> **作者:** Jie Du; Xinyu Gong; Qingshan Tan; Wen Li; Yangming Cheng; Weitao Wang; Chenlu Zhan; Suhui Wu; Hao Zhang; Jun Zhang
>
> **摘要:** Recent studies have identified Direct Preference Optimization (DPO) as an efficient and reward-free approach to improving video generation quality. However, existing methods largely follow image-domain paradigms and are mainly developed on small-scale models (approximately 2B parameters), limiting their ability to address the unique challenges of video tasks, such as costly data construction, unstable training, and heavy memory consumption. To overcome these limitations, we introduce a GT-Pair that automatically builds high-quality preference pairs by using real videos as positives and model-generated videos as negatives, eliminating the need for any external annotation. We further present Reg-DPO, which incorporates the SFT loss as a regularization term into the DPO loss to enhance training stability and generation fidelity. Additionally, by combining the FSDP framework with multiple memory optimization techniques, our approach achieves nearly three times higher training capacity than using FSDP alone. Extensive experiments on both I2V and T2V tasks across multiple datasets demonstrate that our method consistently outperforms existing approaches, delivering superior video generation quality.
>
---
#### [replaced 052] Which Way Does Time Flow? A Psychophysics-Grounded Evaluation for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.26241v2](http://arxiv.org/pdf/2510.26241v2)**

> **作者:** Shiho Matta; Lis Kanashiro Pereira; Peitao Han; Fei Cheng; Shigeru Kitazawa
>
> **备注:** 10 pages
>
> **摘要:** Modern vision-language models (VLMs) excel at many multimodal tasks, yet their grasp of temporal information in video remains weak and, crucially, under-evaluated. We probe this gap with a deceptively simple but revealing challenge: judging the arrow of time (AoT)-whether a short clip is played forward or backward. We introduce AoT-PsyPhyBENCH, a psychophysically validated benchmark that tests whether VLMs can infer temporal direction in natural videos using the same stimuli and behavioral baselines established for humans. Our comprehensive evaluation of open-weight and proprietary, reasoning and non-reasoning VLMs reveals that most models perform near chance, and even the best lag far behind human accuracy on physically irreversible processes (e.g., free fall, diffusion/explosion) and causal manual actions (division/addition) that humans recognize almost instantly. These results highlight a fundamental gap in current multimodal systems: while they capture rich visual-semantic correlations, they lack the inductive biases required for temporal continuity and causal understanding. We release the code and data for AoT-PsyPhyBENCH to encourage further progress in the physical and temporal reasoning capabilities of VLMs.
>
---
#### [replaced 053] TABLET: A Large-Scale Dataset for Robust Visual Table Understanding
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21205v2](http://arxiv.org/pdf/2509.21205v2)**

> **作者:** Iñigo Alonso; Imanol Miranda; Eneko Agirre; Mirella Lapata
>
> **摘要:** While table understanding increasingly relies on pixel-only settings where tables are processed as visual representations, current benchmarks predominantly use synthetic renderings that lack the complexity and visual diversity of real-world tables. Additionally, existing visual table understanding (VTU) datasets offer fixed examples with single visualizations and pre-defined instructions, providing no access to underlying serialized data for reformulation. We introduce TABLET, a large-scale VTU dataset with 4 million examples across 20 tasks, grounded in 2 million unique tables where 88% preserve original visualizations. Each example includes paired image-HTML representations, comprehensive metadata, and provenance information linking back to the source datasets. Fine-tuning vision-language models like Qwen2.5-VL-7B on TABLET improves performance on seen and unseen VTU tasks while increasing robustness on real-world table visualizations. By preserving original visualizations and maintaining example traceability in a unified large-scale collection, TABLET establishes a foundation for robust training and extensible evaluation of future VTU models.
>
---
#### [replaced 054] SAM-EM: Real-Time Segmentation for Automated Liquid Phase Transmission Electron Microscopy
- **分类: cs.CV; physics.data-an**

- **链接: [http://arxiv.org/pdf/2501.03153v2](http://arxiv.org/pdf/2501.03153v2)**

> **作者:** Alexander Wang; Max Xu; Risha Goel; Zain Shabeeb; Isabel Panicker; Vida Jamali
>
> **摘要:** The absence of robust segmentation frameworks for noisy liquid phase transmission electron microscopy (LPTEM) videos prevents reliable extraction of particle trajectories, creating a major barrier to quantitative analysis and to connecting observed dynamics with materials characterization and design. To address this challenge, we present Segment Anything Model for Electron Microscopy (SAM-EM), a domain-adapted foundation model that unifies segmentation, tracking, and statistical analysis for LPTEM data. Built on Segment Anything Model 2 (SAM~2), SAM-EM is derived through full-model fine-tuning on 46,600 curated LPTEM synthetic video frames, substantially improving mask quality and temporal identity stability compared to zero-shot SAM~2 and existing baselines. Beyond segmentation, SAM-EM integrates particle tracking with statistical tools, including mean-squared displacement and particle displacement distribution analysis, providing an end-to-end framework for extracting and interpreting nanoscale dynamics. Crucially, full fine-tuning allows SAM-EM to remain robust under low signal-to-noise conditions, such as those caused by increased liquid sample thickness in LPTEM experiments. By establishing a reliable analysis pipeline, SAM-EM transforms LPTEM into a quantitative single-particle tracking platform and accelerates its integration into data-driven materials discovery and design. Project page: \href{https://github.com/JamaliLab/SAM-EM}{github.com/JamaliLab/SAM-EM}.
>
---
#### [replaced 055] MSDNet: Multi-Scale Decoder for Few-Shot Semantic Segmentation via Transformer-Guided Prototyping
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.11316v5](http://arxiv.org/pdf/2409.11316v5)**

> **作者:** Amirreza Fateh; Mohammad Reza Mohammadi; Mohammad Reza Jahed Motlagh
>
> **摘要:** Few-shot Semantic Segmentation addresses the challenge of segmenting objects in query images with only a handful of annotated examples. However, many previous state-of-the-art methods either have to discard intricate local semantic features or suffer from high computational complexity. To address these challenges, we propose a new Few-shot Semantic Segmentation framework based on the Transformer architecture. Our approach introduces the spatial transformer decoder and the contextual mask generation module to improve the relational understanding between support and query images. Moreover, we introduce a multi scale decoder to refine the segmentation mask by incorporating features from different resolutions in a hierarchical manner. Additionally, our approach integrates global features from intermediate encoder stages to improve contextual understanding, while maintaining a lightweight structure to reduce complexity. This balance between performance and efficiency enables our method to achieve competitive results on benchmark datasets such as PASCAL-5^i and COCO-20^i in both 1-shot and 5-shot settings. Notably, our model with only 1.5 million parameters demonstrates competitive performance while overcoming limitations of existing methodologies. https://github.com/amirrezafateh/MSDNet
>
---
#### [replaced 056] Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.13174v4](http://arxiv.org/pdf/2409.13174v4)**

> **作者:** Hao Cheng; Erjia Xiao; Yichi Wang; Chengyuan Yu; Mengshu Sun; Qiang Zhang; Jiahang Cao; Yijie Guo; Ning Liu; Kaidi Xu; Jize Zhang; Chao Shen; Philip Torr; Jindong Gu; Renjing Xu
>
> **摘要:** Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.
>
---
#### [replaced 057] BoxCell: Leveraging SAM for Cell Segmentation with Box Supervision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.17960v2](http://arxiv.org/pdf/2311.17960v2)**

> **作者:** Aayush Kumar Tyagi; Vaibhav Mishra; Prathosh A. P.; Mausam
>
> **摘要:** Cell segmentation in histopathological images is vital for diagnosis, and treatment of several diseases. Annotating data is tedious, and requires medical expertise, making it difficult to employ supervised learning. Instead, we study a weakly supervised setting, where only bounding box supervision is available, and present the use of Segment Anything (SAM) for this without any finetuning, i.e., directly utilizing the pre-trained model. We propose BoxCell, a cell segmentation framework that utilizes SAM's capability to interpret bounding boxes as prompts, \emph{both} at train and test times. At train time, gold bounding boxes given to SAM produce (pseudo-)masks, which are used to train a standalone segmenter. At test time, BoxCell generates two segmentation masks: (1) generated by this standalone segmenter, and (2) a trained object detector outputs bounding boxes, which are given as prompts to SAM to produce another mask. Recognizing complementary strengths, we reconcile the two segmentation masks using a novel integer programming formulation with intensity and spatial constraints. We experiment on three publicly available cell segmentation datasets namely, CoNSep, MoNuSeg, and TNBC, and find that BoxCell significantly outperforms existing box supervised image segmentation models, obtaining 6-10 point Dice gains.
>
---
#### [replaced 058] EraseFlow: Learning Concept Erasure Policies via GFlowNet-Driven Alignment
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2511.00804v2](http://arxiv.org/pdf/2511.00804v2)**

> **作者:** Abhiram Kusumba; Maitreya Patel; Kyle Min; Changhoon Kim; Chitta Baral; Yezhou Yang
>
> **备注:** NeurIPS'25 Spotlight | Project page: https://eraseflow.github.io/
>
> **摘要:** Erasing harmful or proprietary concepts from powerful text to image generators is an emerging safety requirement, yet current "concept erasure" techniques either collapse image quality, rely on brittle adversarial losses, or demand prohibitive retraining cycles. We trace these limitations to a myopic view of the denoising trajectories that govern diffusion based generation. We introduce EraseFlow, the first framework that casts concept unlearning as exploration in the space of denoising paths and optimizes it with GFlowNets equipped with the trajectory balance objective. By sampling entire trajectories rather than single end states, EraseFlow learns a stochastic policy that steers generation away from target concepts while preserving the model's prior. EraseFlow eliminates the need for carefully crafted reward models and by doing this, it generalizes effectively to unseen concepts and avoids hackable rewards while improving the performance. Extensive empirical results demonstrate that EraseFlow outperforms existing baselines and achieves an optimal trade off between performance and prior preservation.
>
---
#### [replaced 059] Automated Segmentation of Coronal Brain Tissue Slabs for 3D Neuropathology
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09805v2](http://arxiv.org/pdf/2508.09805v2)**

> **作者:** Jonathan Williams Ramirez; Dina Zemlyanker; Lucas Deden-Binder; Rogeny Herisse; Erendira Garcia Pallares; Karthik Gopinath; Harshvardhan Gazula; Christopher Mount; Liana N. Kozanno; Michael S. Marshall; Theresa R. Connors; Matthew P. Frosch; Mark Montine; Derek H. Oakley; Christine L. Mac Donald; C. Dirk Keene; Bradley T. Hyman; Juan Eugenio Iglesias
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Advances in image registration and machine learning have recently enabled volumetric analysis of postmortem brain tissue from conventional photographs of coronal slabs, which are routinely collected in brain banks and neuropathology laboratories worldwide. One caveat of this methodology is the requirement of segmentation of the tissue from photographs, which currently requires costly manual intervention. In this article, we present a deep learning model to automate this process. The automatic segmentation tool relies on a U-Net architecture that was trained with a combination of 1,414 manually segmented images of both fixed and fresh tissue, from specimens with varying diagnoses, photographed at two different sites. Automated model predictions on a subset of photographs not seen in training were analyzed to estimate performance compared to manual labels, including both inter- and intra-rater variability. Our model achieved a median Dice score over 0.98, mean surface distance under 0.4mm, and 95\% Hausdorff distance under 1.60mm, which approaches inter-/intra-rater levels. Our tool is publicly available at surfer.nmr.mgh.harvard.edu/fswiki/PhotoTools.
>
---
