# 计算机视觉 cs.CV

- **最新发布 86 篇**

- **更新 48 篇**

## 最新发布

#### [new 001] Enhancing annotations for 5D apple pose estimation through 3D Gaussian Splatting (3DGS)
- **分类: cs.CV; cs.RO**

- **简介: 该论文属农业机器人中的苹果位姿估计任务，旨在解决因遮挡导致的标注难、不一致问题。提出基于3D高斯泼溅（3DGS）的标注增强管线：重建果园场景→简化人工标注→自动投影生成海量训练标签→训练评估位姿模型。显著减少人工标注量（99.6%），但发现模型仍难学习苹果朝向。**

- **链接: [https://arxiv.org/pdf/2512.20148v1](https://arxiv.org/pdf/2512.20148v1)**

> **作者:** Robert van de Ven; Trim Bresilla; Bram Nelissen; Ard Nieuwenhuizen; Eldert J. van Henten; Gert Kootstra
>
> **备注:** 33 pages, excluding appendices. 17 figures
>
> **摘要:** Automating tasks in orchards is challenging because of the large amount of variation in the environment and occlusions. One of the challenges is apple pose estimation, where key points, such as the calyx, are often occluded. Recently developed pose estimation methods no longer rely on these key points, but still require them for annotations, making annotating challenging and time-consuming. Due to the abovementioned occlusions, there can be conflicting and missing annotations of the same fruit between different images. Novel 3D reconstruction methods can be used to simplify annotating and enlarge datasets. We propose a novel pipeline consisting of 3D Gaussian Splatting to reconstruct an orchard scene, simplified annotations, automated projection of the annotations to images, and the training and evaluation of a pose estimation method. Using our pipeline, 105 manual annotations were required to obtain 28,191 training labels, a reduction of 99.6%. Experimental results indicated that training with labels of fruits that are $\leq95\%$ occluded resulted in the best performance, with a neutral F1 score of 0.927 on the original images and 0.970 on the rendered images. Adjusting the size of the training dataset had small effects on the model performance in terms of F1 score and pose estimation accuracy. It was found that the least occluded fruits had the best position estimation, which worsened as the fruits became more occluded. It was also found that the tested pose estimation method was unable to correctly learn the orientation estimation of apples.
>
---
#### [new 002] Towards Natural Language-Based Document Image Retrieval: New Dataset and Benchmark
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文提出自然语言驱动的文档图像检索（NL-DIR）任务，解决现有方法仅支持图像查询、难以处理细粒度文本查询的问题。构建含41K文档图像及205K人工校验文本查询的新基准数据集，评估主流视觉语言模型，并设计高效两阶段检索方法。**

- **链接: [https://arxiv.org/pdf/2512.20174v1](https://arxiv.org/pdf/2512.20174v1)**

> **作者:** Hao Guo; Xugong Qin; Jun Jie Ou Yang; Peng Zhang; Gangyan Zeng; Yubo Li; Hailun Lin
>
> **备注:** CVPR 2025
>
> **摘要:** Document image retrieval (DIR) aims to retrieve document images from a gallery according to a given query. Existing DIR methods are primarily based on image queries that retrieve documents within the same coarse semantic category, e.g., newspapers or receipts. However, these methods struggle to effectively retrieve document images in real-world scenarios where textual queries with fine-grained semantics are usually provided. To bridge this gap, we introduce a new Natural Language-based Document Image Retrieval (NL-DIR) benchmark with corresponding evaluation metrics. In this work, natural language descriptions serve as semantically rich queries for the DIR task. The NL-DIR dataset contains 41K authentic document images, each paired with five high-quality, fine-grained semantic queries generated and evaluated through large language models in conjunction with manual verification. We perform zero-shot and fine-tuning evaluations of existing mainstream contrastive vision-language models and OCR-free visual document understanding (VDU) models. A two-stage retrieval method is further investigated for performance improvement while achieving both time and space efficiency. We hope the proposed NL-DIR benchmark can bring new opportunities and facilitate research for the VDU community. Datasets and codes will be publicly available at huggingface.co/datasets/nianbing/NL-DIR.
>
---
#### [new 003] AMoE: Agglomerative Mixture-of-Experts Vision Foundation Model
- **分类: cs.CV**

- **简介: 该论文属视觉基础模型训练任务，旨在提升多教师知识蒸馏的计算效率与数据效率。提出AMoE框架，含非对称关系蒸馏损失、令牌均衡批处理和分层数据采样，并构建OpenLVD200M数据集，实现高效多教师蒸馏。**

- **链接: [https://arxiv.org/pdf/2512.20157v1](https://arxiv.org/pdf/2512.20157v1)**

> **作者:** Sofian Chaybouti; Sanath Narayan; Yasser Dahou; Phúc H. Lê Khac; Ankit Singh; Ngoc Dung Huynh; Wamiq Reyaz Para; Hilde Kuehne; Hakim Hacid
>
> **备注:** 17 pages, 8 figures, 11 tables
>
> **摘要:** Vision foundation models trained via multi-teacher distillation offer a promising path toward unified visual representations, yet the learning dynamics and data efficiency of such approaches remain underexplored. In this paper, we systematically study multi-teacher distillation for vision foundation models and identify key factors that enable training at lower computational cost. We introduce Agglomerative Mixture-of-Experts Vision Foundation Models (AMoE), which distill knowledge from SigLIP2 and DINOv3 simultaneously into a Mixture-of-Experts student. We show that (1) our Asymmetric Relation-Knowledge Distillation loss preserves the geometric properties of each teacher while enabling effective knowledge transfer, (2) token-balanced batching that packs varying-resolution images into sequences with uniform token budgets stabilizes representation learning across resolutions without sacrificing performance, and (3) hierarchical clustering and sampling of training data--typically reserved for self-supervised learning--substantially improves sample efficiency over random sampling for multi-teacher distillation. By combining these findings, we curate OpenLVD200M, a 200M-image corpus that demonstrates superior efficiency for multi-teacher distillation. Instantiated in a Mixture-of-Experts. We release OpenLVD200M and distilled models.
>
---
#### [new 004] Few-Shot-Based Modular Image-to-Video Adapter for Diffusion Models
- **分类: cs.CV**

- **简介: 该论文面向图像动画任务，解决扩散模型在少样本下难以泛化新运动模式、提示控制不精准的问题。提出轻量可组合的模块化图像到视频适配器（MIVA），单模块学一种运动，仅需约10样本即可训练，支持多模块组合实现灵活运动控制。**

- **链接: [https://arxiv.org/pdf/2512.20000v1](https://arxiv.org/pdf/2512.20000v1)**

> **作者:** Zhenhao Li; Shaohan Yi; Zheng Liu; Leonartinus Gao; Minh Ngoc Le; Ambrose Ling; Zhuoran Wang; Md Amirul Islam; Zhixiang Chi; Yuanhao Yu
>
> **摘要:** Diffusion models (DMs) have recently achieved impressive photorealism in image and video generation. However, their application to image animation remains limited, even when trained on large-scale datasets. Two primary challenges contribute to this: the high dimensionality of video signals leads to a scarcity of training data, causing DMs to favor memorization over prompt compliance when generating motion; moreover, DMs struggle to generalize to novel motion patterns not present in the training set, and fine-tuning them to learn such patterns, especially using limited training data, is still under-explored. To address these limitations, we propose Modular Image-to-Video Adapter (MIVA), a lightweight sub-network attachable to a pre-trained DM, each designed to capture a single motion pattern and scalable via parallelization. MIVAs can be efficiently trained on approximately ten samples using a single consumer-grade GPU. At inference time, users can specify motion by selecting one or multiple MIVAs, eliminating the need for prompt engineering. Extensive experiments demonstrate that MIVA enables more precise motion control while maintaining, or even surpassing, the generation quality of models trained on significantly larger datasets.
>
---
#### [new 005] A Novel CNN Gradient Boosting Ensemble for Guava Disease Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属图像分类任务，旨在解决孟加拉国番石榴因炭疽病和果蝇感染导致的产量与品质下降问题。作者构建CNN-梯度提升集成模型，在GFDD24数据集上实现99.99%准确率，支持实时病害检测。**

- **链接: [https://arxiv.org/pdf/2512.19989v1](https://arxiv.org/pdf/2512.19989v1)**

> **作者:** Tamim Ahasan Rijon; Yeasin Arafath
>
> **备注:** Accepted at IEEE ICCIT 2025. This is the author accepted manuscript
>
> **摘要:** As a significant agricultural country, Bangladesh utilizes its fertile land for guava cultivation and dedicated labor to boost its economic development. In a nation like Bangladesh, enhancing guava production and agricultural practices plays a crucial role in its economy. Anthracnose and fruit fly infection can lower the quality and productivity of guava, a crucial tropical fruit. Expert systems that detect diseases early can reduce losses and safeguard the harvest. Images of guava fruits classified into the Healthy, Fruit Flies, and Anthracnose classes are included in the Guava Fruit Disease Dataset 2024 (GFDD24), which comes from plantations in Rajshahi and Pabna, Bangladesh. This study aims to create models using CNN alongside traditional machine learning techniques that can effectively identify guava diseases in locally cultivated varieties in Bangladesh. In order to achieve the highest classification accuracy of approximately 99.99% for the guava dataset, we propose utilizing ensemble models that combine CNNML with Gradient Boosting Machine. In general, the CNN-ML cascade framework exhibits strong, high-accuracy guava disease detection that is appropriate for real-time agricultural monitoring systems.
>
---
#### [new 006] Unified Brain Surface and Volume Registration
- **分类: cs.CV; cs.AI**

- **简介: 该论文属医学图像配准任务，旨在解决脑MRI中皮层表面与内部体积联合配准不一致的问题。作者提出NeurAlign深度学习框架，通过球面坐标空间统一建模表面与体素，实现几何一致、高精度、快速的端到端联合配准。**

- **链接: [https://arxiv.org/pdf/2512.19928v1](https://arxiv.org/pdf/2512.19928v1)**

> **作者:** S. Mazdak Abulnaga; Andrew Hoopes; Malte Hoffmann; Robin Magnet; Maks Ovsjanikov; Lilla Zöllei; John Guttag; Bruce Fischl; Adrian Dalca
>
> **摘要:** Accurate registration of brain MRI scans is fundamental for cross-subject analysis in neuroscientific studies. This involves aligning both the cortical surface of the brain and the interior volume. Traditional methods treat volumetric and surface-based registration separately, which often leads to inconsistencies that limit downstream analyses. We propose a deep learning framework, NeurAlign, that registers $3$D brain MRI images by jointly aligning both cortical and subcortical regions through a unified volume-and-surface-based representation. Our approach leverages an intermediate spherical coordinate space to bridge anatomical surface topology with volumetric anatomy, enabling consistent and anatomically accurate alignment. By integrating spherical registration into the learning, our method ensures geometric coherence between volume and surface domains. In a series of experiments on both in-domain and out-of-domain datasets, our method consistently outperforms both classical and machine learning-based registration methods -- improving the Dice score by up to 7 points while maintaining regular deformation fields. Additionally, it is orders of magnitude faster than the standard method for this task, and is simpler to use because it requires no additional inputs beyond an MRI scan. With its superior accuracy, fast inference, and ease of use, NeurAlign sets a new standard for joint cortical and subcortical registration.
>
---
#### [new 007] LiteFusion: Taming 3D Object Detectors from Vision-Based to Multi-Modal with Minimal Adaptation
- **分类: cs.CV**

- **简介: 该论文面向多模态3D目标检测任务，旨在解决现有方法对LiDAR强依赖、部署难（受限于3D稀疏卷积）及鲁棒性差的问题。提出LiteFusion：将LiDAR仅作为几何辅助信息，在四元数空间融合至图像特征，摒弃LiDAR专用骨干网络，提升精度、鲁棒性与硬件兼容性。**

- **链接: [https://arxiv.org/pdf/2512.20217v1](https://arxiv.org/pdf/2512.20217v1)**

> **作者:** Xiangxuan Ren; Zhongdao Wang; Pin Tang; Guoqing Wang; Jilai Zheng; Chao Ma
>
> **备注:** 13 pages, 9 figures, 8 tables
>
> **摘要:** 3D object detection is fundamental for safe and robust intelligent transportation systems. Current multi-modal 3D object detectors often rely on complex architectures and training strategies to achieve higher detection accuracy. However, these methods heavily rely on the LiDAR sensor so that they suffer from large performance drops when LiDAR is absent, which compromises the robustness and safety of autonomous systems in practical scenarios. Moreover, existing multi-modal detectors face difficulties in deployment on diverse hardware platforms, such as NPUs and FPGAs, due to their reliance on 3D sparse convolution operators, which are primarily optimized for NVIDIA GPUs. To address these challenges, we reconsider the role of LiDAR in the camera-LiDAR fusion paradigm and introduce a novel multi-modal 3D detector, LiteFusion. Instead of treating LiDAR point clouds as an independent modality with a separate feature extraction backbone, LiteFusion utilizes LiDAR data as a complementary source of geometric information to enhance camera-based detection. This straightforward approach completely eliminates the reliance on a 3D backbone, making the method highly deployment-friendly. Specifically, LiteFusion integrates complementary features from LiDAR points into image features within a quaternion space, where the orthogonal constraints are well-preserved during network training. This helps model domain-specific relations across modalities, yielding a compact cross-modal embedding. Experiments on the nuScenes dataset show that LiteFusion improves the baseline vision-based detector by +20.4% mAP and +19.7% NDS with a minimal increase in parameters (1.1%) without using dedicated LiDAR encoders. Notably, even in the absence of LiDAR input, LiteFusion maintains strong results , highlighting its favorable robustness and effectiveness across diverse fusion paradigms and deployment scenarios.
>
---
#### [new 008] SE360: Semantic Edit in 360$^\circ$ Panoramas via Hierarchical Data Construction
- **分类: cs.CV**

- **简介: 该论文属360°全景图像编辑任务，旨在解决现有方法在ERP与透视视图中编辑结果不真实、语义不准的问题。提出SE360框架：构建无监督分层数据生成 pipeline，结合VLM与自适应投影；设计两阶段数据精炼策略；训练Transformer扩散模型，支持文本/掩码/参考图多条件编辑。**

- **链接: [https://arxiv.org/pdf/2512.19943v1](https://arxiv.org/pdf/2512.19943v1)**

> **作者:** Haoyi Zhong; Fang-Lue Zhang; Andrew Chalmers; Taehyun Rhee
>
> **摘要:** While instruction-based image editing is emerging, extending it to 360$^\circ$ panoramas introduces additional challenges. Existing methods often produce implausible results in both equirectangular projections (ERP) and perspective views. To address these limitations, we propose SE360, a novel framework for multi-condition guided object editing in 360$^\circ$ panoramas. At its core is a novel coarse-to-fine autonomous data generation pipeline without manual intervention. This pipeline leverages a Vision-Language Model (VLM) and adaptive projection adjustment for hierarchical analysis, ensuring the holistic segmentation of objects and their physical context. The resulting data pairs are both semantically meaningful and geometrically consistent, even when sourced from unlabeled panoramas. Furthermore, we introduce a cost-effective, two-stage data refinement strategy to improve data realism and mitigate model overfitting to erase artifacts. Based on the constructed dataset, we train a Transformer-based diffusion model to allow flexible object editing guided by text, mask, or reference image in 360$^\circ$ panoramas. Our experiments demonstrate that our method outperforms existing methods in both visual quality and semantic accuracy.
>
---
#### [new 009] HistoWAS: A Pathomics Framework for Large-Scale Feature-Wide Association Studies of Tissue Topology and Patient Outcomes
- **分类: cs.CV**

- **简介: 该论文提出HistoWAS框架，属病理图像空间组学任务，旨在解决组织微结构空间特征与临床结局关联分析缺乏有效工具的问题。工作包括：构建含30个GIS启发的拓扑/空间特征的新特征空间，并设计类PheWAS的批量关联分析引擎，在KPMP肾组织WSIs上验证。**

- **链接: [https://arxiv.org/pdf/2512.19954v1](https://arxiv.org/pdf/2512.19954v1)**

> **作者:** Yuechen Yang; Junlin Guo; Yanfan Zhu; Jialin Yue; Junchao Zhu; Yu Wang; Shilin Zhao; Haichun Yang; Xingyi Guo; Jovan Tanevski; Laura Barisoni; Avi Z. Rosenberg; Yuankai Huo
>
> **摘要:** High-throughput "pathomic" analysis of Whole Slide Images (WSIs) offers new opportunities to study tissue characteristics and for biomarker discovery. However, the clinical relevance of the tissue characteristics at the micro- and macro-environment level is limited by the lack of tools that facilitate the measurement of the spatial interaction of individual structure characteristics and their association with clinical parameters. To address these challenges, we introduce HistoWAS (Histology-Wide Association Study), a computational framework designed to link tissue spatial organization to clinical outcomes. Specifically, HistoWAS implements (1) a feature space that augments conventional metrics with 30 topological and spatial features, adapted from Geographic Information Systems (GIS) point pattern analysis, to quantify tissue micro-architecture; and (2) an association study engine, inspired by Phenome-Wide Association Studies (PheWAS), that performs mass univariate regression for each feature with statistical correction. As a proof of concept, we applied HistoWAS to analyze a total of 102 features (72 conventional object-level features and our 30 spatial features) using 385 PAS-stained WSIs from 206 participants in the Kidney Precision Medicine Project (KPMP). The code and data have been released to https://github.com/hrlblab/histoWAS.
>
---
#### [new 010] HEART-VIT: Hessian-Guided Efficient Dynamic Attention and Token Pruning in Vision Transformer
- **分类: cs.CV**

- **简介: 该论文属模型压缩任务，旨在解决ViT因二次注意力计算和冗余导致的高延迟、高资源消耗问题。提出HEART-ViT框架，首次统一采用Hessian二阶敏感度指导动态注意力头与token联合剪枝，实现高精度保持下的显著FLOPs降低与边缘端加速。**

- **链接: [https://arxiv.org/pdf/2512.20120v1](https://arxiv.org/pdf/2512.20120v1)**

> **作者:** Mohammad Helal Uddin; Liam Seymour; Sabur Baidya
>
> **摘要:** Vision Transformers (ViTs) deliver state-of-the-art accuracy but their quadratic attention cost and redundant computations severely hinder deployment on latency and resource-constrained platforms. Existing pruning approaches treat either tokens or heads in isolation, relying on heuristics or first-order signals, which often sacrifice accuracy or fail to generalize across inputs. We introduce HEART-ViT, a Hessian-guided efficient dynamic attention and token pruning framework for vision transformers, which to the best of our knowledge is the first unified, second-order, input-adaptive framework for ViT optimization. HEART-ViT estimates curvature-weighted sensitivities of both tokens and attention heads using efficient Hessian-vector products, enabling principled pruning decisions under explicit loss budgets.This dual-view sensitivity reveals an important structural insight: token pruning dominates computational savings, while head pruning provides fine-grained redundancy removal, and their combination achieves a superior trade-off. On ImageNet-100 and ImageNet-1K with ViT-B/16 and DeiT-B/16, HEART-ViT achieves up to 49.4 percent FLOPs reduction, 36 percent lower latency, and 46 percent higher throughput, while consistently matching or even surpassing baseline accuracy after fine-tuning, for example 4.7 percent recovery at 40 percent token pruning. Beyond theoretical benchmarks, we deploy HEART-ViT on different edge devices such as AGX Orin, demonstrating that our reductions in FLOPs and latency translate directly into real-world gains in inference speed and energy efficiency. HEART-ViT bridges the gap between theory and practice, delivering the first unified, curvature-driven pruning framework that is both accuracy-preserving and edge-efficient.
>
---
#### [new 011] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis
- **分类: cs.CV**

- **简介: 该论文面向多模态医学诊断任务，解决传统GNN依赖单一静态图、难以建模患者特异性病理关系的问题。提出MAPI-GNN框架，通过多维判别器挖掘图感知模式，动态构建多激活平面图，并经关系融合引擎实现鲁棒诊断。**

- **链接: [https://arxiv.org/pdf/2512.20026v1](https://arxiv.org/pdf/2512.20026v1)**

> **作者:** Ziwei Qin; Xuhui Song; Deqing Huang; Na Qin; Jun Li
>
> **备注:** Accepted by Proceedings of the AAAI Conference on Artificial Intelligence 40 (AAAI-26)
>
> **摘要:** Graph neural networks are increasingly applied to multimodal medical diagnosis for their inherent relational modeling capabilities. However, their efficacy is often compromised by the prevailing reliance on a single, static graph built from indiscriminate features, hindering the ability to model patient-specific pathological relationships. To this end, the proposed Multi-Activation Plane Interaction Graph Neural Network (MAPI-GNN) reconstructs this single-graph paradigm by learning a multifaceted graph profile from semantically disentangled feature subspaces. The framework first uncovers latent graph-aware patterns via a multi-dimensional discriminator; these patterns then guide the dynamic construction of a stack of activation graphs; and this multifaceted profile is finally aggregated and contextualized by a relational fusion engine for a robust diagnosis. Extensive experiments on two diverse tasks, comprising over 1300 patient samples, demonstrate that MAPI-GNN significantly outperforms state-of-the-art methods.
>
---
#### [new 012] LiDARDraft: Generating LiDAR Point Cloud from Versatile Inputs
- **分类: cs.CV**

- **简介: 该论文属生成式AI任务，旨在解决LiDAR点云生成中高质量与多模态可控性难以兼顾的问题。提出LiDARDraft框架，将文本、图像等输入统一映射为3D布局，提取语义与深度控制信号，再通过rangemap-based ControlNet实现像素级对齐的可控点云生成。**

- **链接: [https://arxiv.org/pdf/2512.20105v1](https://arxiv.org/pdf/2512.20105v1)**

> **作者:** Haiyun Wei; Fan Lu; Yunwei Zhu; Zehan Zheng; Weiyi Xue; Lin Shao; Xudong Zhang; Ya Wu; Rong Fu; Guang Chen
>
> **摘要:** Generating realistic and diverse LiDAR point clouds is crucial for autonomous driving simulation. Although previous methods achieve LiDAR point cloud generation from user inputs, they struggle to attain high-quality results while enabling versatile controllability, due to the imbalance between the complex distribution of LiDAR point clouds and the simple control signals. To address the limitation, we propose LiDARDraft, which utilizes the 3D layout to build a bridge between versatile conditional signals and LiDAR point clouds. The 3D layout can be trivially generated from various user inputs such as textual descriptions and images. Specifically, we represent text, images, and point clouds as unified 3D layouts, which are further transformed into semantic and depth control signals. Then, we employ a rangemap-based ControlNet to guide LiDAR point cloud generation. This pixel-level alignment approach demonstrates excellent performance in controllable LiDAR point clouds generation, enabling "simulation from scratch", allowing self-driving environments to be created from arbitrary textual descriptions, images and sketches.
>
---
#### [new 013] FlashVLM: Text-Guided Visual Token Selection for Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属多模态模型效率优化任务，旨在解决VLM中视觉令牌冗余导致的高计算成本问题。提出FlashVLM框架，通过文本引导的跨模态相似性计算、显式相关性融合与多样性保留机制，动态精简视觉令牌，在大幅压缩（最高94.4%）下保持甚至超越原模型性能。**

- **链接: [https://arxiv.org/pdf/2512.20561v1](https://arxiv.org/pdf/2512.20561v1)**

> **作者:** Kaitong Cai; Jusheng Zhang; Jing Yang; Yijia Fan; Pengtao Xie; Jian Wang; Keze Wang
>
> **备注:** Under submission
>
> **摘要:** Large vision-language models (VLMs) typically process hundreds or thousands of visual tokens per image or video frame, incurring quadratic attention cost and substantial redundancy. Existing token reduction methods often ignore the textual query or rely on deep attention maps, whose instability under aggressive pruning leads to degraded semantic alignment. We propose FlashVLM, a text guided visual token selection framework that dynamically adapts visual inputs to the query. Instead of relying on noisy attention weights, FlashVLM computes an explicit cross modal similarity between projected image tokens and normalized text embeddings in the language model space. This extrinsic relevance is fused with intrinsic visual saliency using log domain weighting and temperature controlled sharpening. In addition, a diversity preserving partition retains a minimal yet representative set of background tokens to maintain global context. Under identical token budgets and evaluation protocols, FlashVLM achieves beyond lossless compression, slightly surpassing the unpruned baseline while pruning up to 77.8 percent of visual tokens on LLaVA 1.5, and maintaining 92.8 percent accuracy even under 94.4 percent compression. Extensive experiments on 14 image and video benchmarks demonstrate that FlashVLM delivers state of the art efficiency performance trade offs while maintaining strong robustness and generalization across mainstream VLMs.
>
---
#### [new 014] DETACH : Decomposed Spatio-Temporal Alignment for Exocentric Video and Ambient Sensors with Staged Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向**多模态时序对齐任务**，解决**外源视频与环境传感器数据对齐困难**问题：全局对齐无法捕获局部动作细节且易混淆语义相似动作。提出DETACH框架，通过**空间-时间解耦建模、在线聚类发现传感器空间特征、两阶段对比对齐**，显著提升下游动作识别性能。**

- **链接: [https://arxiv.org/pdf/2512.20409v1](https://arxiv.org/pdf/2512.20409v1)**

> **作者:** Junho Yoon; Jaemo Jung; Hyunju Kim; Dongman Lee
>
> **摘要:** Aligning egocentric video with wearable sensors have shown promise for human action recognition, but face practical limitations in user discomfort, privacy concerns, and scalability. We explore exocentric video with ambient sensors as a non-intrusive, scalable alternative. While prior egocentric-wearable works predominantly adopt Global Alignment by encoding entire sequences into unified representations, this approach fails in exocentric-ambient settings due to two problems: (P1) inability to capture local details such as subtle motions, and (P2) over-reliance on modality-invariant temporal patterns, causing misalignment between actions sharing similar temporal patterns with different spatio-semantic contexts. To resolve these problems, we propose DETACH, a decomposed spatio-temporal framework. This explicit decomposition preserves local details, while our novel sensor-spatial features discovered via online clustering provide semantic grounding for context-aware alignment. To align the decomposed features, our two-stage approach establishes spatial correspondence through mutual supervision, then performs temporal alignment via a spatial-temporal weighted contrastive loss that adaptively handles easy negatives, hard negatives, and false negatives. Comprehensive experiments with downstream tasks on Opportunity++ and HWU-USP datasets demonstrate substantial improvements over adapted egocentric-wearable baselines.
>
---
#### [new 015] PHANTOM: PHysical ANamorphic Threats Obstructing Connected Vehicle Mobility
- **分类: cs.CV; cs.AI; cs.CR; cs.LG**

- **简介: 该论文提出PHANTOM框架，针对网联自动驾驶汽车（CAV）的物理对抗攻击任务，解决其视觉感知与V2X通信层的安全漏洞问题。通过设计人眼自然、模型误判的变形艺术式对抗样本，实现黑盒、跨模型攻击，并验证其在真实仿真环境中对感知与网络通信的双重破坏效果。**

- **链接: [https://arxiv.org/pdf/2512.19711v1](https://arxiv.org/pdf/2512.19711v1)**

> **作者:** Md Nahid Hasan Shuvo; Moinul Hossain
>
> **摘要:** Connected autonomous vehicles (CAVs) rely on vision-based deep neural networks (DNNs) and low-latency (Vehicle-to-Everything) V2X communication to navigate safely and efficiently. Despite their advances, these systems remain vulnerable to physical adversarial attacks. In this paper, we introduce PHANTOM (PHysical ANamorphic Threats Obstructing connected vehicle Mobility), a novel framework for crafting and deploying perspective-dependent adversarial examples using \textit{anamorphic art}. PHANTOM exploits geometric distortions that appear natural to humans but are misclassified with high confidence by state-of-the-art object detectors. Unlike conventional attacks, PHANTOM operates in black-box settings without model access and demonstrates strong transferability across four diverse detector architectures (YOLOv5, SSD, Faster R-CNN, and RetinaNet). Comprehensive evaluation in CARLA across varying speeds, weather conditions, and lighting scenarios shows that PHANTOM achieves over 90\% attack success rate under optimal conditions and maintains 60-80\% effectiveness even in degraded environments. The attack activates within 6-10 meters of the target, providing insufficient time for safe maneuvering. Beyond individual vehicle deception, PHANTOM triggers network-wide disruption in CAV systems: SUMO-OMNeT++ co-simulation demonstrates that false emergency messages propagate through V2X links, increasing Peak Age of Information by 68-89\% and degrading safety-critical communication. These findings expose critical vulnerabilities in both perception and communication layers of CAV ecosystems.
>
---
#### [new 016] Learning to Refocus with Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属图像编辑任务，旨在解决单张模糊图像的交互式后处理重聚焦问题。提出基于视频扩散模型的新方法，从单张失焦图生成高质量焦点堆栈视频，并构建大规模真实手机拍摄数据集，显著提升感知质量与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.19823v1](https://arxiv.org/pdf/2512.19823v1)**

> **作者:** SaiKiran Tedla; Zhoutong Zhang; Xuaner Zhang; Shumian Xin
>
> **备注:** Code and data are available at https://www.learn2refocus.github.io . SIGGRAPH Asia 2025, Dec. 2025
>
> **摘要:** Focus is a cornerstone of photography, yet autofocus systems often fail to capture the intended subject, and users frequently wish to adjust focus after capture. We introduce a novel method for realistic post-capture refocusing using video diffusion models. From a single defocused image, our approach generates a perceptually accurate focal stack, represented as a video sequence, enabling interactive refocusing and unlocking a range of downstream applications. We release a large-scale focal stack dataset acquired under diverse real-world smartphone conditions to support this work and future research. Our method consistently outperforms existing approaches in both perceptual quality and robustness across challenging scenarios, paving the way for more advanced focus-editing capabilities in everyday photography. Code and data are available at www.learn2refocus.github.io
>
---
#### [new 017] PaveSync: A Unified and Comprehensive Dataset for Pavement Distress Analysis and Classification
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的目标检测任务，旨在解决路面病害检测因数据集不统一导致泛化性差的问题。作者构建了标准化、多国来源的PaveSync数据集（52747张图像、13类病害、135277个标注框），并基于其开展SOTA模型基准测试，支持公平评估与跨环境迁移。**

- **链接: [https://arxiv.org/pdf/2512.20011v1](https://arxiv.org/pdf/2512.20011v1)**

> **作者:** Blessing Agyei Kyem; Joshua Kofi Asamoah; Anthony Dontoh; Andrews Danyo; Eugene Denteh; Armstrong Aboah
>
> **摘要:** Automated pavement defect detection often struggles to generalize across diverse real-world conditions due to the lack of standardized datasets. Existing datasets differ in annotation styles, distress type definitions, and formats, limiting their integration for unified training. To address this gap, we introduce a comprehensive benchmark dataset that consolidates multiple publicly available sources into a standardized collection of 52747 images from seven countries, with 135277 bounding box annotations covering 13 distinct distress types. The dataset captures broad real-world variation in image quality, resolution, viewing angles, and weather conditions, offering a unique resource for consistent training and evaluation. Its effectiveness was demonstrated through benchmarking with state-of-the-art object detection models including YOLOv8-YOLOv12, Faster R-CNN, and DETR, which achieved competitive performance across diverse scenarios. By standardizing class definitions and annotation formats, this dataset provides the first globally representative benchmark for pavement defect detection and enables fair comparison of models, including zero-shot transfer to new environments.
>
---
#### [new 018] LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属端到端自动驾驶任务，旨在解决模仿学习中专家（高可见性、低不确定性、明确导航意图）与学生（传感器受限、意图模糊）间的感知与认知不对称问题。作者提出LEAD方法，通过缩小专家-学生差距，显著提升CARLA及真实世界基准性能。**

- **链接: [https://arxiv.org/pdf/2512.20563v1](https://arxiv.org/pdf/2512.20563v1)**

> **作者:** Long Nguyen; Micha Fauth; Bernhard Jaeger; Daniel Dauner; Maximilian Igl; Andreas Geiger; Kashyap Chitta
>
> **摘要:** Simulators can generate virtually unlimited driving data, yet imitation learning policies in simulation still struggle to achieve robust closed-loop performance. Motivated by this gap, we empirically study how misalignment between privileged expert demonstrations and sensor-based student observations can limit the effectiveness of imitation learning. More precisely, experts have significantly higher visibility (e.g., ignoring occlusions) and far lower uncertainty (e.g., knowing other vehicles' actions), making them difficult to imitate reliably. Furthermore, navigational intent (i.e., the route to follow) is under-specified in student models at test time via only a single target point. We demonstrate that these asymmetries can measurably limit driving performance in CARLA and offer practical interventions to address them. After careful modifications to narrow the gaps between expert and student, our TransFuser v6 (TFv6) student policy achieves a new state of the art on all major publicly available CARLA closed-loop benchmarks, reaching 95 DS on Bench2Drive and more than doubling prior performances on Longest6~v2 and Town13. Additionally, by integrating perception supervision from our dataset into a shared sim-to-real pipeline, we show consistent gains on the NAVSIM and Waymo Vision-Based End-to-End driving benchmarks. Our code, data, and models are publicly available at https://github.com/autonomousvision/lead.
>
---
#### [new 019] $\text{H}^2$em: Learning Hierarchical Hyperbolic Embeddings for Compositional Zero-Shot Learning
- **分类: cs.CV**

- **简介: 该论文面向组合式零样本学习（CZSL）任务，解决现有方法难以建模大规模语义与概念层次结构的问题。提出H2em框架，在双曲空间中学习分层嵌入，设计双层次蕴含损失、判别对齐损失及双曲跨模态注意力，显著提升泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.20029v1](https://arxiv.org/pdf/2512.20029v1)**

> **作者:** Lin Li; Jiahui Li; Jiaming Lei; Jun Xiao; Feifei Shao; Long Chen
>
> **摘要:** Compositional zero-shot learning (CZSL) aims to recognize unseen state-object compositions by generalizing from a training set of their primitives (state and object). Current methods often overlook the rich hierarchical structures, such as the semantic hierarchy of primitives (e.g., apple fruit) and the conceptual hierarchy between primitives and compositions (e.g, sliced apple apple). A few recent efforts have shown effectiveness in modeling these hierarchies through loss regularization within Euclidean space. In this paper, we argue that they fail to scale to the large-scale taxonomies required for real-world CZSL: the space's polynomial volume growth in flat geometry cannot match the exponential structure, impairing generalization capacity. To this end, we propose H2em, a new framework that learns Hierarchical Hyperbolic EMbeddings for CZSL. H2em leverages the unique properties of hyperbolic geometry, a space naturally suited for embedding tree-like structures with low distortion. However, a naive hyperbolic mapping may suffer from hierarchical collapse and poor fine-grained discrimination. We further design two learning objectives to structure this space: a Dual-Hierarchical Entailment Loss that uses hyperbolic entailment cones to enforce the predefined hierarchies, and a Discriminative Alignment Loss with hard negative mining to establish a large geodesic distance between semantically similar compositions. Furthermore, we devise Hyperbolic Cross-Modal Attention to realize instance-aware cross-modal infusion within hyperbolic geometry. Extensive ablations on three benchmarks demonstrate that H2em establishes a new state-of-the-art in both closed-world and open-world scenarios. Our codes will be released.
>
---
#### [new 020] Item Region-based Style Classification Network (IRSN): A Fashion Style Classifier Based on Domain Knowledge of Fashion Experts
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向时尚风格分类任务，旨在解决同类风格内视觉差异大、不同风格间外观相似导致的分类难问题。提出基于商品区域的风格分类网络（IRSN），融合局部商品特征与全局特征，并采用双主干结构增强表征能力，在多个数据集上显著提升准确率。**

- **链接: [https://arxiv.org/pdf/2512.20088v1](https://arxiv.org/pdf/2512.20088v1)**

> **作者:** Jinyoung Choi; Youngchae Kwon; Injung Kim
>
> **备注:** This is a pre-print of an article published in Applied Intelligence. The final authenticated version is available online at: https://doi.org/10.1007/s10489-024-05683-9
>
> **摘要:** Fashion style classification is a challenging task because of the large visual variation within the same style and the existence of visually similar styles. Styles are expressed not only by the global appearance, but also by the attributes of individual items and their combinations. In this study, we propose an item region-based fashion style classification network (IRSN) to effectively classify fashion styles by analyzing item-specific features and their combinations in addition to global features. IRSN extracts features of each item region using item region pooling (IRP), analyzes them separately, and combines them using gated feature fusion (GFF). In addition, we improve the feature extractor by applying a dual-backbone architecture that combines a domain-specific feature extractor and a general feature extractor pre-trained with a large-scale image-text dataset. In experiments, applying IRSN to six widely-used backbones, including EfficientNet, ConvNeXt, and Swin Transformer, improved style classification accuracy by an average of 6.9% and a maximum of 14.5% on the FashionStyle14 dataset and by an average of 7.6% and a maximum of 15.1% on the ShowniqV3 dataset. Visualization analysis also supports that the IRSN models are better than the baseline models at capturing differences between similar style classes.
>
---
#### [new 021] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs
- **分类: cs.CV**

- **简介: 论文提出Widget2Code任务，解决从图像化UI小部件生成高保真、紧凑代码的难题。针对 widget 缺乏标注、空间约束严、结构密集等特点，构建图像-only 基准与多维评估指标；提出 WidgetFactory 系统，含 WidgetDSL、编译器和自适应渲染模块，显著提升视觉一致性与代码可靠性。**

- **链接: [https://arxiv.org/pdf/2512.19918v1](https://arxiv.org/pdf/2512.19918v1)**

> **作者:** Houston H. Zhang; Tao Zhang; Baoze Lin; Yuanqi Xue; Yincheng Zhu; Huan Liu; Li Gu; Linfeng Ye; Ziqiang Wang; Xinxin Zuo; Yang Wang; Yuanhao Yu; Zhixiang Chi
>
> **备注:** Code: https://github.com/Djanghao/widget2code
>
> **摘要:** User interface to code (UI2Code) aims to generate executable code that can faithfully reconstruct a given input UI. Prior work focuses largely on web pages and mobile screens, leaving app widgets underexplored. Unlike web or mobile UIs with rich hierarchical context, widgets are compact, context-free micro-interfaces that summarize key information through dense layouts and iconography under strict spatial constraints. Moreover, while (image, code) pairs are widely available for web or mobile UIs, widget designs are proprietary and lack accessible markup. We formalize this setting as the Widget-to-Code (Widget2Code) and introduce an image-only widget benchmark with fine-grained, multi-dimensional evaluation metrics. Benchmarking shows that although generalized multimodal large language models (MLLMs) outperform specialized UI2Code methods, they still produce unreliable and visually inconsistent code. To address these limitations, we develop a baseline that jointly advances perceptual understanding and structured code generation. At the perceptual level, we follow widget design principles to assemble atomic components into complete layouts, equipped with icon retrieval and reusable visualization modules. At the system level, we design an end-to-end infrastructure, WidgetFactory, which includes a framework-agnostic widget-tailored domain-specific language (WidgetDSL) and a compiler that translates it into multiple front-end implementations (e.g., React, HTML/CSS). An adaptive rendering module further refines spatial dimensions to satisfy compactness constraints. Together, these contributions substantially enhance visual fidelity, establishing a strong baseline and unified infrastructure for future Widget2Code research.
>
---
#### [new 022] Generating the Past, Present and Future from a Motion-Blurred Image
- **分类: cs.CV; cs.GR**

- **简介: 该论文属图像/视频生成任务，旨在从单张运动模糊图像恢复过去、当前与未来的动态视频。它利用预训练的大规模视频扩散模型，无需手工先验，直接生成反映复杂场景运动的视频，支持相机轨迹、物体运动及3D结构等下游分析。**

- **链接: [https://arxiv.org/pdf/2512.19817v1](https://arxiv.org/pdf/2512.19817v1)**

> **作者:** SaiKiran Tedla; Kelly Zhu; Trevor Canham; Felix Taubner; Michael S. Brown; Kiriakos N. Kutulakos; David B. Lindell
>
> **备注:** Code and data are available at https://blur2vid.github.io
>
> **摘要:** We seek to answer the question: what can a motion-blurred image reveal about a scene's past, present, and future? Although motion blur obscures image details and degrades visual quality, it also encodes information about scene and camera motion during an exposure. Previous techniques leverage this information to estimate a sharp image from an input blurry one, or to predict a sequence of video frames showing what might have occurred at the moment of image capture. However, they rely on handcrafted priors or network architectures to resolve ambiguities in this inverse problem, and do not incorporate image and video priors on large-scale datasets. As such, existing methods struggle to reproduce complex scene dynamics and do not attempt to recover what occurred before or after an image was taken. Here, we introduce a new technique that repurposes a pre-trained video diffusion model trained on internet-scale datasets to recover videos revealing complex scene dynamics during the moment of capture and what might have occurred immediately into the past or future. Our approach is robust and versatile; it outperforms previous methods for this task, generalizes to challenging in-the-wild images, and supports downstream tasks such as recovering camera trajectories, object motion, and dynamic 3D scene structure. Code and data are available at https://blur2vid.github.io
>
---
#### [new 023] FedPOD: the deployable units of training for federated learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出FedPOD，面向联邦学习任务，解决FedPIDAvg中 outlier剔除导致数据利用不足、依赖历史轮次信息限制灵活性的问题；通过纳入被剔除客户端、去历史依赖、每轮计算验证损失，并借鉴K8s POD设计实现弹性扩缩容，提升效率、灵活性与性能。**

- **链接: [https://arxiv.org/pdf/2512.20610v1](https://arxiv.org/pdf/2512.20610v1)**

> **作者:** Daewoon Kim; Si Young Yie; Jae Sung Lee
>
> **备注:** 12 pages, 12 figures, MICCAI
>
> **摘要:** This paper proposes FedPOD (Proportionally Orchestrated Derivative) for optimizing learning efficiency and communication cost in federated learning among multiple clients. Inspired by FedPIDAvg, we define a round-wise task for FedPOD to enhance training efficiency. FedPIDAvg achieved performance improvement by incorporating the training loss reduction for prediction entropy as weights using differential terms. Furthermore, by modeling data distribution with a Poisson distribution and using a PID controller, it reduced communication costs even in skewed data distribution. However, excluding participants classified as outliers based on the Poisson distribution can limit data utilization. Additionally, PID controller requires the same participants to be maintained throughout the federated learning process as it uses previous rounds' learning information in the current round. In our approach, FedPOD addresses these issues by including participants excluded as outliers, eliminating dependency on previous rounds' learning information, and applying a method for calculating validation loss at each round. In this challenge, FedPOD presents comparable performance to FedPIDAvg in metrics of Dice score, 0.78, 0.71 and 0.72 for WT, ET and TC in average, and projected convergence score, 0.74 in average. Furthermore, the concept of FedPOD draws inspiration from Kubernetes' smallest computing unit, POD, designed to be compatible with Kubernetes auto-scaling. Extending round-wise tasks of FedPOD to POD units allows flexible design by applying scale-out similar to Kubernetes' auto-scaling. This work demonstrated the potentials of FedPOD to enhance federated learning by improving efficiency, flexibility, and performance in metrics.
>
---
#### [new 024] Multi Modal Attention Networks with Uncertainty Quantification for Automated Concrete Bridge Deck Delamination Detection
- **分类: cs.CV; eess.IV**

- **简介: 该论文面向桥梁甲板脱粘自动检测任务，解决单模态方法（雷达/热成像）各自局限问题。提出多模态注意力网络，融合雷达时序与热图空间特征，并引入不确定性量化，提升检测精度、鲁棒性与决策安全性。**

- **链接: [https://arxiv.org/pdf/2512.20113v1](https://arxiv.org/pdf/2512.20113v1)**

> **作者:** Alireza Moayedikia; Sattar Dorafshan
>
> **摘要:** Deteriorating civil infrastructure requires automated inspection techniques overcoming limitations of visual assessment. While Ground Penetrating Radar and Infrared Thermography enable subsurface defect detection, single modal approaches face complementary constraints radar struggles with moisture and shallow defects, while thermography exhibits weather dependency and limited depth. This paper presents a multi modal attention network fusing radar temporal patterns with thermal spatial signatures for bridge deck delamination detection. Our architecture introduces temporal attention for radar processing, spatial attention for thermal features, and cross modal fusion with learnable embeddings discovering complementary defect patterns invisible to individual sensors. We incorporate uncertainty quantification through Monte Carlo dropout and learned variance estimation, decomposing uncertainty into epistemic and aleatoric components for safety critical decisions. Experiments on five bridge datasets reveal that on balanced to moderately imbalanced data, our approach substantially outperforms baselines in accuracy and AUC representing meaningful improvements over single modal and concatenation based fusion. Ablation studies demonstrate cross modal attention provides critical gains beyond within modality attention, while multi head mechanisms achieve improved calibration. Uncertainty quantification reduces calibration error, enabling selective prediction by rejecting uncertain cases. However, under extreme class imbalance, attention mechanisms show vulnerability to majority class collapse. These findings provide actionable guidance: attention based architecture performs well across typical scenarios, while extreme imbalance requires specialized techniques. Our system maintains deployment efficiency, enabling real time inspection with characterized capabilities and limitations.
>
---
#### [new 025] Beyond Vision: Contextually Enriched Image Captioning with Multi-Modal Retrieva
- **分类: cs.CV; cs.AI**

- **简介: 该论文属图像描述生成任务，旨在解决传统方法缺乏事件背景、时间线索等非视觉上下文的问题。提出多模态检索增强框架：用BEIT-3/SigLIP检索相似图，ORB/SIFT几何重排序，语义搜索获取关联文章，再经QLoRA微调Qwen3融合上下文生成丰富描述。**

- **链接: [https://arxiv.org/pdf/2512.20042v1](https://arxiv.org/pdf/2512.20042v1)**

> **作者:** Nguyen Lam Phu Quy; Pham Phu Hoa; Tran Chi Nguyen; Dao Sy Duy Minh; Nguyen Hoang Minh Ngoc; Huynh Trung Kiet
>
> **备注:** 7 pages, 5 figures. System description for the EVENTA Grand Challenge (Track 1) at ACM MM'25
>
> **摘要:** Real-world image captions often lack contextual depth, omitting crucial details such as event background, temporal cues, outcomes, and named entities that are not visually discernible. This gap limits the effectiveness of image understanding in domains like journalism, education, and digital archives, where richer, more informative descriptions are essential. To address this, we propose a multimodal pipeline that augments visual input with external textual knowledge. Our system retrieves semantically similar images using BEIT-3 (Flickr30k-384 and COCO-384) and SigLIP So-384, reranks them using ORB and SIFT for geometric alignment, and extracts contextual information from related articles via semantic search. A fine-tuned Qwen3 model with QLoRA then integrates this context with base captions generated by Instruct BLIP (Vicuna-7B) to produce event-enriched, context-aware descriptions. Evaluated on the OpenEvents v1 dataset, our approach generates significantly more informative captions compared to traditional methods, showing strong potential for real-world applications requiring deeper visual-textual understanding
>
---
#### [new 026] Repurposing Video Diffusion Transformers for Robust Point Tracking
- **分类: cs.CV**

- **简介: 该论文属点跟踪任务，旨在解决视频中对应点跨帧定位不鲁棒、缺乏时序一致性的问题。提出DiTracker方法：利用预训练视频Diffusion Transformer（DiT）的时空特征，通过查询-键匹配、LoRA微调及与ResNet成本融合，实现高效鲁棒跟踪。**

- **链接: [https://arxiv.org/pdf/2512.20606v1](https://arxiv.org/pdf/2512.20606v1)**

> **作者:** Soowon Son; Honggyu An; Chaehyun Kim; Hyunah Ko; Jisu Nam; Dahyun Chung; Siyoon Jin; Jung Yi; Jaewon Min; Junhwa Hur; Seungryong Kim
>
> **备注:** Project Page: https://cvlab-kaist.github.io/DiTracker/
>
> **摘要:** Point tracking aims to localize corresponding points across video frames, serving as a fundamental task for 4D reconstruction, robotics, and video editing. Existing methods commonly rely on shallow convolutional backbones such as ResNet that process frames independently, lacking temporal coherence and producing unreliable matching costs under challenging conditions. Through systematic analysis, we find that video Diffusion Transformers (DiTs), pre-trained on large-scale real-world videos with spatio-temporal attention, inherently exhibit strong point tracking capability and robustly handle dynamic motions and frequent occlusions. We propose DiTracker, which adapts video DiTs through: (1) query-key attention matching, (2) lightweight LoRA tuning, and (3) cost fusion with a ResNet backbone. Despite training with 8 times smaller batch size, DiTracker achieves state-of-the-art performance on challenging ITTO benchmark and matches or outperforms state-of-the-art models on TAP-Vid benchmarks. Our work validates video DiT features as an effective and efficient foundation for point tracking.
>
---
#### [new 027] Chain-of-Anomaly Thoughts with Large Vision-Language Models
- **分类: cs.CV; cs.MA**

- **简介: 该论文属视频异常检测任务，旨在解决大视觉语言模型因“正常性偏置”而漏检犯罪事件的问题。提出Chain-of-Anomaly-Thoughts（CoAT）多智能体推理框架，通过引入归纳式犯罪偏差与异常聚焦分类层，显著提升低/高分辨率视频中的异常检测与分类性能。**

- **链接: [https://arxiv.org/pdf/2512.20417v1](https://arxiv.org/pdf/2512.20417v1)**

> **作者:** Pedro Domingos; João Pereira; Vasco Lopes; João Neves; David Semedo
>
> **备注:** 2 pages, 3 figures, 1 table. Accepted for RECPAD 2025
>
> **摘要:** Automated video surveillance with Large Vision-Language Models is limited by their inherent bias towards normality, often failing to detect crimes. While Chain-of-Thought reasoning strategies show significant potential for improving performance in language tasks, the lack of inductive anomaly biases in their reasoning further steers the models towards normal interpretations. To address this, we propose Chain-of-Anomaly-Thoughts (CoAT), a multi-agent reasoning framework that introduces inductive criminal bias in the reasoning process through a final, anomaly-focused classification layer. Our method significantly improves Anomaly Detection, boosting F1-score by 11.8 p.p. on challenging low-resolution footage and Anomaly Classification by 3.78 p.p. in high-resolution videos.
>
---
#### [new 028] DDAVS: Disentangled Audio Semantics and Delayed Bidirectional Alignment for Audio-Visual Segmentation
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文面向音频-视觉分割（AVS）任务，旨在解决多声源纠缠与音视频时序/语义错位问题。提出DDAVS框架：通过可学习查询与音频原型记忆库解耦音频语义，并引入延迟双向交叉注意力增强音视频对齐。在多源、多实例场景下性能领先。**

- **链接: [https://arxiv.org/pdf/2512.20117v1](https://arxiv.org/pdf/2512.20117v1)**

> **作者:** Jingqi Tian; Yiheng Du; Haoji Zhang; Yuji Wang; Isaac Ning Lee; Xulong Bai; Tianrui Zhu; Jingxuan Niu; Yansong Tang
>
> **备注:** https://trilarflagz.github.io/DDAVS-page/
>
> **摘要:** Audio-Visual Segmentation (AVS) aims to localize sound-producing objects at the pixel level by jointly leveraging auditory and visual information. However, existing methods often suffer from multi-source entanglement and audio-visual misalignment, which lead to biases toward louder or larger objects while overlooking weaker, smaller, or co-occurring sources. To address these challenges, we propose DDAVS, a Disentangled Audio Semantics and Delayed Bidirectional Alignment framework. To mitigate multi-source entanglement, DDAVS employs learnable queries to extract audio semantics and anchor them within a structured semantic space derived from an audio prototype memory bank. This is further optimized through contrastive learning to enhance discriminability and robustness. To alleviate audio-visual misalignment, DDAVS introduces dual cross-attention with delayed modality interaction, improving the robustness of multimodal alignment. Extensive experiments on the AVS-Objects and VPO benchmarks demonstrate that DDAVS consistently outperforms existing approaches, exhibiting strong performance across single-source, multi-source, and multi-instance scenarios. These results validate the effectiveness and generalization ability of our framework under challenging real-world audio-visual segmentation conditions. Project page: https://trilarflagz.github.io/DDAVS-page/
>
---
#### [new 029] Linking Faces and Voices Across Languages: Insights from the FAME 2026 Challenge
- **分类: cs.CV**

- **简介: 该论文介绍FAME 2026挑战赛，属跨模态（人脸-语音）关联任务，旨在解决多语言环境下训练与测试语言不一致时的鲁棒性问题。工作为组织并总结该ICASSP 2026国际挑战赛。**

- **链接: [https://arxiv.org/pdf/2512.20376v1](https://arxiv.org/pdf/2512.20376v1)**

> **作者:** Marta Moscati; Ahmed Abdullah; Muhammad Saad Saeed; Shah Nawaz; Rohan Kumar Das; Muhammad Zaigham Zaheer; Junaid Mir; Muhammad Haroon Yousaf; Khalid Mahmood Malik; Markus Schedl
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Over half of the world's population is bilingual and people often communicate under multilingual scenarios. The Face-Voice Association in Multilingual Environments (FAME) 2026 Challenge, held at ICASSP 2026, focuses on developing methods for face-voice association that are effective when the language at test-time is different than the training one. This report provides a brief summary of the challenge.
>
---
#### [new 030] FlashLips: 100-FPS Mask-Free Latent Lip-Sync using Reconstruction Instead of Diffusion or GANs
- **分类: cs.CV**

- **简介: 该论文提出FlashLips，解决高保真、实时唇形同步任务。它摒弃GAN/扩散模型，采用两阶段重建式方法：第一阶段用自监督掩码无关编辑器在潜在空间重构图像；第二阶段用流匹配音频-姿态Transformer预测唇部姿态，实现>100 FPS、无需显式掩码的高质量唇动合成。**

- **链接: [https://arxiv.org/pdf/2512.20033v1](https://arxiv.org/pdf/2512.20033v1)**

> **作者:** Andreas Zinonos; Michał Stypułkowski; Antoni Bigata; Stavros Petridis; Maja Pantic; Nikita Drobyshev
>
> **摘要:** We present FlashLips, a two-stage, mask-free lip-sync system that decouples lips control from rendering and achieves real-time performance running at over 100 FPS on a single GPU, while matching the visual quality of larger state-of-the-art models. Stage 1 is a compact, one-step latent-space editor that reconstructs an image using a reference identity, a masked target frame, and a low-dimensional lips-pose vector, trained purely with reconstruction losses - no GANs or diffusion. To remove explicit masks at inference, we use self-supervision: we generate mouth-altered variants of the target image, that serve as pseudo ground truth for fine-tuning, teaching the network to localize edits to the lips while preserving the rest. Stage 2 is an audio-to-pose transformer trained with a flow-matching objective to predict lips-poses vectors from speech. Together, these stages form a simple and stable pipeline that combines deterministic reconstruction with robust audio control, delivering high perceptual quality and faster-than-real-time speed.
>
---
#### [new 031] UbiQVision: Quantifying Uncertainty in XAI for Image Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属可解释人工智能（XAI）任务，旨在解决SHAP在医疗图像识别中解释不稳定、不可靠的问题。作者提出UbiQVision框架，结合Dirichlet后验采样与Dempster-Shafer理论，量化SHAP解释中的认知与偶然不确定性，并在多模态医学影像数据上验证。**

- **链接: [https://arxiv.org/pdf/2512.20288v1](https://arxiv.org/pdf/2512.20288v1)**

> **作者:** Akshat Dubey; Aleksandar Anžel; Bahar İlgen; Georges Hattab
>
> **摘要:** Recent advances in deep learning have led to its widespread adoption across diverse domains, including medical imaging. This progress is driven by increasingly sophisticated model architectures, such as ResNets, Vision Transformers, and Hybrid Convolutional Neural Networks, that offer enhanced performance at the cost of greater complexity. This complexity often compromises model explainability and interpretability. SHAP has emerged as a prominent method for providing interpretable visualizations that aid domain experts in understanding model predictions. However, SHAP explanations can be unstable and unreliable in the presence of epistemic and aleatoric uncertainty. In this study, we address this challenge by using Dirichlet posterior sampling and Dempster-Shafer theory to quantify the uncertainty that arises from these unstable explanations in medical imaging applications. The framework uses a belief, plausible, and fusion map approach alongside statistical quantitative analysis to produce quantification of uncertainty in SHAP. Furthermore, we evaluated our framework on three medical imaging datasets with varying class distributions, image qualities, and modality types which introduces noise due to varying image resolutions and modality-specific aspect covering the examples from pathology, ophthalmology, and radiology, introducing significant epistemic uncertainty.
>
---
#### [new 032] Degradation-Aware Metric Prompting for Hyperspectral Image Restoration
- **分类: cs.CV; eess.IV**

- **简介: 该论文面向高光谱图像（HSI）统一恢复任务，解决真实场景中 degradation 先验难获取的问题。提出 Degradation-Aware Metric Prompting（DAMP）框架：设计空间-光谱退化度量生成退化提示（DP），并结合空间-光谱自适应模块（SSAM）与 MoE 架构，实现无需显式退化标签的鲁棒、泛化恢复。**

- **链接: [https://arxiv.org/pdf/2512.20251v1](https://arxiv.org/pdf/2512.20251v1)**

> **作者:** Binfeng Wang; Di Wang; Haonan Guo; Ying Fu; Jing Zhang
>
> **摘要:** Unified hyperspectral image (HSI) restoration aims to recover various degraded HSIs using a single model, offering great practical value. However, existing methods often depend on explicit degradation priors (e.g., degradation labels) as prompts to guide restoration, which are difficult to obtain due to complex and mixed degradations in real-world scenarios. To address this challenge, we propose a Degradation-Aware Metric Prompting (DAMP) framework. Instead of relying on predefined degradation priors, we design spatial-spectral degradation metrics to continuously quantify multi-dimensional degradations, serving as Degradation Prompts (DP). These DP enable the model to capture cross-task similarities in degradation distributions and enhance shared feature learning. Furthermore, we introduce a Spatial-Spectral Adaptive Module (SSAM) that dynamically modulates spatial and spectral feature extraction through learnable parameters. By integrating SSAM as experts within a Mixture-of-Experts architecture, and using DP as the gating router, the framework enables adaptive, efficient, and robust restoration under diverse, mixed, or unseen degradations. Extensive experiments on natural and remote sensing HSI datasets show that DAMP achieves state-of-the-art performance and demonstrates exceptional generalization capability. Code is publicly available at https://github.com/MiliLab/DAMP.
>
---
#### [new 033] Skin Lesion Classification Using a Soft Voting Ensemble of Convolutional Neural Networks
- **分类: cs.CV**

- **简介: 该论文属皮肤病变分类任务，旨在提升早期皮肤癌诊断准确率。提出基于MobileNetV2、VGG19和InceptionV3的软投票CNN集成方法，结合图像增强、重采样及双编码器分割预处理，在HAM10000、ISIC 2016/2019上分别达96.32%、90.86%、93.92%准确率。**

- **链接: [https://arxiv.org/pdf/2512.20431v1](https://arxiv.org/pdf/2512.20431v1)**

> **作者:** Abdullah Al Shafi; Abdul Muntakim; Pintu Chandra Shill; Rowzatul Zannat; Abdullah Al-Amin
>
> **备注:** Authors' version of the paper published in proceedings of ECCE, DOI: https://doi.org/10.1109/ECCE64574.2025.11013422
>
> **摘要:** Skin cancer can be identified by dermoscopic examination and ocular inspection, but early detection significantly increases survival chances. Artificial intelligence (AI), using annotated skin images and Convolutional Neural Networks (CNNs), improves diagnostic accuracy. This paper presents an early skin cancer classification method using a soft voting ensemble of CNNs. In this investigation, three benchmark datasets, namely HAM10000, ISIC 2016, and ISIC 2019, were used. The process involved rebalancing, image augmentation, and filtering techniques, followed by a hybrid dual encoder for segmentation via transfer learning. Accurate segmentation focused classification models on clinically significant features, reducing background artifacts and improving accuracy. Classification was performed through an ensemble of MobileNetV2, VGG19, and InceptionV3, balancing accuracy and speed for real-world deployment. The method achieved lesion recognition accuracies of 96.32\%, 90.86\%, and 93.92\% for the three datasets. The system performance was evaluated using established skin lesion detection metrics, yielding impressive results.
>
---
#### [new 034] IndicDLP: A Foundational Dataset for Multi-Lingual and Multi-Domain Document Layout Parsing
- **分类: cs.CV**

- **简介: 该论文面向文档布局解析（DLP）任务，解决现有数据集缺乏印地语系多语言、多领域及细粒度标注的问题。作者构建了大规模基础数据集IndicDLP（覆盖11种印度语言+英语、12个领域），并推出UED-mini辅助预训练，实验证明其显著提升模型性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.20236v1](https://arxiv.org/pdf/2512.20236v1)**

> **作者:** Oikantik Nath; Sahithi Kukkala; Mitesh Khapra; Ravi Kiran Sarvadevabhatla
>
> **备注:** Accepted in ICDAR 2025 (Oral Presentation) - Best Student Paper Runner-Up Award
>
> **摘要:** Document layout analysis is essential for downstream tasks such as information retrieval, extraction, OCR, and digitization. However, existing large-scale datasets like PubLayNet and DocBank lack fine-grained region labels and multilingual diversity, making them insufficient for representing complex document layouts. In contrast, human-annotated datasets such as M6Doc and D4LA offer richer labels and greater domain diversity, but are too small to train robust models and lack adequate multilingual coverage. This gap is especially pronounced for Indic documents, which encompass diverse scripts yet remain underrepresented in current datasets, further limiting progress in this space. To address these shortcomings, we introduce IndicDLP, a large-scale foundational document layout dataset spanning 11 representative Indic languages alongside English and 12 common document domains. Additionally, we curate UED-mini, a dataset derived from DocLayNet and M6Doc, to enhance pretraining and provide a solid foundation for Indic layout models. Our experiments demonstrate that fine-tuning existing English models on IndicDLP significantly boosts performance, validating its effectiveness. Moreover, models trained on IndicDLP generalize well beyond Indic layouts, making it a valuable resource for document digitization. This work bridges gaps in scale, diversity, and annotation granularity, driving inclusive and efficient document understanding.
>
---
#### [new 035] AlignPose: Generalizable 6D Pose Estimation via Multi-view Feature-metric Alignment
- **分类: cs.CV**

- **简介: 该论文提出AlignPose，解决单视图6D位姿估计因深度模糊、遮挡等导致的泛化局限。通过多视角特征-度量对齐，无需对象特训或对称标注，实现强泛化多视图6D位姿估计，在多个工业数据集上达到SOTA。**

- **链接: [https://arxiv.org/pdf/2512.20538v1](https://arxiv.org/pdf/2512.20538v1)**

> **作者:** Anna Šárová Mikeštíková; Médéric Fourmy; Martin Cífka; Josef Sivic; Vladimir Petrik
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Single-view RGB model-based object pose estimation methods achieve strong generalization but are fundamentally limited by depth ambiguity, clutter, and occlusions. Multi-view pose estimation methods have the potential to solve these issues, but existing works rely on precise single-view pose estimates or lack generalization to unseen objects. We address these challenges via the following three contributions. First, we introduce AlignPose, a 6D object pose estimation method that aggregates information from multiple extrinsically calibrated RGB views and does not require any object-specific training or symmetry annotation. Second, the key component of this approach is a new multi-view feature-metric refinement specifically designed for object pose. It optimizes a single, consistent world-frame object pose minimizing the feature discrepancy between on-the-fly rendered object features and observed image features across all views simultaneously. Third, we report extensive experiments on four datasets (YCB-V, T-LESS, ITODD-MV, HouseCat6D) using the BOP benchmark evaluation and show that AlignPose outperforms other published methods, especially on challenging industrial datasets where multiple views are readily available in practice.
>
---
#### [new 036] milliMamba: Specular-Aware Human Pose Estimation via Dual mmWave Radar with Multi-Frame Mamba Fusion
- **分类: cs.CV**

- **简介: 该论文面向毫米波雷达2D人体姿态估计任务，解决雷达信号因镜面反射导致的稀疏性与关键点缺失问题。提出milliMamba框架：采用Cross-View Fusion Mamba编码器高效建模长序列时空特征，结合时空交叉注意力解码器预测关节坐标，并引入速度损失增强运动平滑性。**

- **链接: [https://arxiv.org/pdf/2512.20128v1](https://arxiv.org/pdf/2512.20128v1)**

> **作者:** Niraj Prakash Kini; Shiau-Rung Tsai; Guan-Hsun Lin; Wen-Hsiao Peng; Ching-Wen Ma; Jenq-Neng Hwang
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Millimeter-wave radar offers a privacy-preserving and lighting-invariant alternative to RGB sensors for Human Pose Estimation (HPE) task. However, the radar signals are often sparse due to specular reflection, making the extraction of robust features from radar signals highly challenging. To address this, we present milliMamba, a radar-based 2D human pose estimation framework that jointly models spatio-temporal dependencies across both the feature extraction and decoding stages. Specifically, given the high dimensionality of radar inputs, we adopt a Cross-View Fusion Mamba encoder to efficiently extract spatio-temporal features from longer sequences with linear complexity. A Spatio-Temporal-Cross Attention decoder then predicts joint coordinates across multiple frames. Together, this spatio-temporal modeling pipeline enables the model to leverage contextual cues from neighboring frames and joints to infer missing joints caused by specular reflections. To reinforce motion smoothness, we incorporate a velocity loss alongside the standard keypoint loss during training. Experiments on the TransHuPR and HuPR datasets demonstrate that our method achieves significant performance improvements, exceeding the baselines by 11.0 AP and 14.6 AP, respectively, while maintaining reasonable complexity. Code: https://github.com/NYCU-MAPL/milliMamba
>
---
#### [new 037] High Dimensional Data Decomposition for Anomaly Detection of Textured Images
- **分类: cs.CV; stat.ML**

- **简介: 该论文面向纹理图像异常检测任务，解决传统方法在纹理缺陷图像中误检率高、鲁棒性差、依赖大数据等问题。提出纹理基集成平滑分解（TBSD）方法，通过学习纹理基函数建模准周期性，并利用其先验知识精准分离稀疏异常。**

- **链接: [https://arxiv.org/pdf/2512.20432v1](https://arxiv.org/pdf/2512.20432v1)**

> **作者:** Ji Song; Xing Wang; Jianguo Wu; Xiaowei Yue
>
> **摘要:** In the realm of diverse high-dimensional data, images play a significant role across various processes of manufacturing systems where efficient image anomaly detection has emerged as a core technology of utmost importance. However, when applied to textured defect images, conventional anomaly detection methods have limitations including non-negligible misidentification, low robustness, and excessive reliance on large-scale and structured datasets. This paper proposes a texture basis integrated smooth decomposition (TBSD) approach, which is targeted at efficient anomaly detection in textured images with smooth backgrounds and sparse anomalies. Mathematical formulation of quasi-periodicity and its theoretical properties are investigated for image texture estimation. TBSD method consists of two principal processes: the first process learns the texture basis functions to effectively extract quasi-periodic texture patterns; the subsequent anomaly detection process utilizes that texture basis as prior knowledge to prevent texture misidentification and capture potential anomalies with high accuracy.The proposed method surpasses benchmarks with less misidentification, smaller training dataset requirement, and superior anomaly detection performance on both simulation and real-world datasets.
>
---
#### [new 038] Learning to Reason in 4D: Dynamic Spatial Understanding for Vision Language Models
- **分类: cs.CV**

- **简介: 该论文面向动态空间推理（DSR）任务，解决VLM在4D（3D+时间）空间演化理解上的短板。提出DSR Suite：构建基于真实视频的4D多选数据集（DSR-Train/Bench），并设计轻量几何选择模块（GSM）将4D几何先验融入VLM，显著提升其动态空间推理能力。**

- **链接: [https://arxiv.org/pdf/2512.20557v1](https://arxiv.org/pdf/2512.20557v1)**

> **作者:** Shengchao Zhou; Yuxin Chen; Yuying Ge; Wei Huang; Jiehong Lin; Ying Shan; Xiaojuan Qi
>
> **摘要:** Vision-language models (VLM) excel at general understanding yet remain weak at dynamic spatial reasoning (DSR), i.e., reasoning about the evolvement of object geometry and relationship in 3D space over time, largely due to the scarcity of scalable 4D-aware training resources. To bridge this gap across aspects of dataset, benchmark and model, we introduce DSR Suite. First, we propose an automated pipeline that generates multiple-choice question-answer pairs from in-the-wild videos for DSR. By leveraging modern vision foundation models, the pipeline extracts rich geometric and motion information, including camera poses, local point clouds, object masks, orientations, and 3D trajectories. These geometric cues enable the construction of DSR-Train for learning and further human-refined DSR-Bench for evaluation. Compared with previous works, our data emphasize (i) in-the-wild video sources, (ii) object- and scene-level 3D requirements, (iii) viewpoint transformations, (iv) multi-object interactions, and (v) fine-grained, procedural answers. Beyond data, we propose a lightweight Geometry Selection Module (GSM) to seamlessly integrate geometric priors into VLMs, which condenses question semantics and extracts question-relevant knowledge from pretrained 4D reconstruction priors into a compact set of geometry tokens. This targeted extraction avoids overwhelming the model with irrelevant knowledge. Experiments show that integrating DSR-Train and GSM into Qwen2.5-VL-7B significantly enhances its dynamic spatial reasoning capability, while maintaining accuracy on general video understanding benchmarks.
>
---
#### [new 039] Multi-temporal Adaptive Red-Green-Blue and Long-Wave Infrared Fusion for You Only Look Once-Based Landmine Detection from Unmanned Aerial Systems
- **分类: cs.CV**

- **简介: 该论文属遥感目标检测任务，旨在解决无人机航拍下地表地雷的自动识别问题。通过自适应RGB与长波红外（LWIR）多时序融合，结合YOLO系列模型（尤以YOLOv11最优），提升热对比度特征提取能力，并系统评估了模型精度、训练效率及环境适应性。**

- **链接: [https://arxiv.org/pdf/2512.20487v1](https://arxiv.org/pdf/2512.20487v1)**

> **作者:** James E. Gallagher; Edward J. Oughton; Jana Kosecka
>
> **备注:** 21 pages with 6 figures
>
> **摘要:** Landmines remain a persistent humanitarian threat, with 110 million actively deployed mines across 60 countries, claiming 26,000 casualties annually. This research evaluates adaptive Red-Green-Blue (RGB) and Long-Wave Infrared (LWIR) fusion for Unmanned Aerial Systems (UAS)-based detection of surface-laid landmines, leveraging the thermal contrast between the ordnance and the surrounding soil to enhance feature extraction. Using You Only Look Once (YOLO) architectures (v8, v10, v11) across 114 test images, generating 35,640 model-condition evaluations, YOLOv11 achieved optimal performance (86.8% mAP), with 10 to 30% thermal fusion at 5 to 10m altitude identified as the optimal detection parameters. A complementary architectural comparison revealed that while RF-DETR achieved the highest accuracy (69.2% mAP), followed by Faster R-CNN (67.6%), YOLOv11 (64.2%), and RetinaNet (50.2%), YOLOv11 trained 17.7 times faster than the transformer-based RF-DETR (41 minutes versus 12 hours), presenting a critical accuracy-efficiency tradeoff for operational deployment. Aggregated multi-temporal training datasets outperformed season-specific approaches by 1.8 to 9.6%, suggesting that models benefit from exposure to diverse thermal conditions. Anti-Tank (AT) mines achieved 61.9% detection accuracy, compared with 19.2% for Anti-Personnel (AP) mines, reflecting both the size differential and thermal-mass differences between these ordnance classes. As this research examined surface-laid mines where thermal contrast is maximized, future research should quantify thermal contrast effects for mines buried at varying depths across heterogeneous soil types.
>
---
#### [new 040] Active Intelligence in Video Avatars via Closed-loop World Modeling
- **分类: cs.CV**

- **简介: 该论文面向视频虚拟人主动智能任务，解决现有方法缺乏自主目标导向行为的问题。提出L-IVA基准与ORCA框架，通过闭合OTAR循环和双系统架构构建内部世界模型，实现基于POMDP的在线推理与信念更新，使虚拟人具备长程、自适应、目标驱动的交互能力。**

- **链接: [https://arxiv.org/pdf/2512.20615v1](https://arxiv.org/pdf/2512.20615v1)**

> **作者:** Xuanhua He; Tianyu Yang; Ke Cao; Ruiqi Wu; Cheng Meng; Yong Zhang; Zhuoliang Kang; Xiaoming Wei; Qifeng Chen
>
> **备注:** Project Page: https://xuanhuahe.github.io/ORCA/
>
> **摘要:** Current video avatar generation methods excel at identity preservation and motion alignment but lack genuine agency, they cannot autonomously pursue long-term goals through adaptive environmental interaction. We address this by introducing L-IVA (Long-horizon Interactive Visual Avatar), a task and benchmark for evaluating goal-directed planning in stochastic generative environments, and ORCA (Online Reasoning and Cognitive Architecture), the first framework enabling active intelligence in video avatars. ORCA embodies Internal World Model (IWM) capabilities through two key innovations: (1) a closed-loop OTAR cycle (Observe-Think-Act-Reflect) that maintains robust state tracking under generative uncertainty by continuously verifying predicted outcomes against actual generations, and (2) a hierarchical dual-system architecture where System 2 performs strategic reasoning with state prediction while System 1 translates abstract plans into precise, model-specific action captions. By formulating avatar control as a POMDP and implementing continuous belief updating with outcome verification, ORCA enables autonomous multi-step task completion in open-domain scenarios. Extensive experiments demonstrate that ORCA significantly outperforms open-loop and non-reflective baselines in task success rate and behavioral coherence, validating our IWM-inspired design for advancing video avatar intelligence from passive animation to active, goal-oriented behavior.
>
---
#### [new 041] Bridging Modalities and Transferring Knowledge: Enhanced Multimodal Understanding and Recognition
- **分类: cs.CV**

- **简介: 该论文属多模态机器学习任务，旨在提升跨模态理解与知识迁移能力。针对空间语言解析、医学文本定位、知识图谱构建、动作识别等挑战，提出Spatial-Reasoning BERT、解剖位置映射、文本到事实对齐、视频-检测融合及RGB知识蒸馏等方法。**

- **链接: [https://arxiv.org/pdf/2512.20501v1](https://arxiv.org/pdf/2512.20501v1)**

> **作者:** Gorjan Radevski
>
> **备注:** Ph.D. manuscript; Supervisors/Mentors: Marie-Francine Moens and Tinne Tuytelaars
>
> **摘要:** This manuscript explores multimodal alignment, translation, fusion, and transference to enhance machine understanding of complex inputs. We organize the work into five chapters, each addressing unique challenges in multimodal machine learning. Chapter 3 introduces Spatial-Reasoning Bert for translating text-based spatial relations into 2D arrangements between clip-arts. This enables effective decoding of spatial language into visual representations, paving the way for automated scene generation aligned with human spatial understanding. Chapter 4 presents a method for translating medical texts into specific 3D locations within an anatomical atlas. We introduce a loss function leveraging spatial co-occurrences of medical terms to create interpretable mappings, significantly enhancing medical text navigability. Chapter 5 tackles translating structured text into canonical facts within knowledge graphs. We develop a benchmark for linking natural language to entities and predicates, addressing ambiguities in text extraction to provide clearer, actionable insights. Chapter 6 explores multimodal fusion methods for compositional action recognition. We propose a method fusing video frames and object detection representations, improving recognition robustness and accuracy. Chapter 7 investigates multimodal knowledge transference for egocentric action recognition. We demonstrate how multimodal knowledge distillation enables RGB-only models to mimic multimodal fusion-based capabilities, reducing computational requirements while maintaining performance. These contributions advance methodologies for spatial language understanding, medical text interpretation, knowledge graph enrichment, and action recognition, enhancing computational systems' ability to process complex, multimodal inputs across diverse applications.
>
---
#### [new 042] Block-Recurrent Dynamics in Vision Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出“块递归假说”（BRH），揭示ViT深层计算存在可复用的少块递归结构。通过分析表征相似性、构建递归代理模型Raptor，并验证其高效复现性能，论文旨在建立ViT的动力学解释框架，解决其深层机制不明的问题。**

- **链接: [https://arxiv.org/pdf/2512.19941v1](https://arxiv.org/pdf/2512.19941v1)**

> **作者:** Mozes Jacobs; Thomas Fel; Richard Hakim; Alessandra Brondetta; Demba Ba; T. Andy Keller
>
> **备注:** 25 pages, 15 figures
>
> **摘要:** As Vision Transformers (ViTs) become standard vision backbones, a mechanistic account of their computational phenomenology is essential. Despite architectural cues that hint at dynamical structure, there is no settled framework that interprets Transformer depth as a well-characterized flow. In this work, we introduce the Block-Recurrent Hypothesis (BRH), arguing that trained ViTs admit a block-recurrent depth structure such that the computation of the original $L$ blocks can be accurately rewritten using only $k \ll L$ distinct blocks applied recurrently. Across diverse ViTs, between-layer representational similarity matrices suggest few contiguous phases. To determine whether these phases reflect genuinely reusable computation, we train block-recurrent surrogates of pretrained ViTs: Recurrent Approximations to Phase-structured TransfORmers (Raptor). In small-scale, we demonstrate that stochastic depth and training promote recurrent structure and subsequently correlate with our ability to accurately fit Raptor. We then provide an empirical existence proof for BRH by training a Raptor model to recover $96\%$ of DINOv2 ImageNet-1k linear probe accuracy in only 2 blocks at equivalent computational cost. Finally, we leverage our hypothesis to develop a program of Dynamical Interpretability. We find i) directional convergence into class-dependent angular basins with self-correcting trajectories under small perturbations, ii) token-specific dynamics, where cls executes sharp late reorientations while patch tokens exhibit strong late-stage coherence toward their mean direction, and iii) a collapse to low rank updates in late depth, consistent with convergence to low-dimensional attractors. Altogether, we find a compact recurrent program emerges along ViT depth, pointing to a low-complexity normative solution that enables these models to be studied through principled dynamical systems analysis.
>
---
#### [new 043] A Contextual Analysis of Driver-Facing and Dual-View Video Inputs for Distraction Detection in Naturalistic Driving Environments
- **分类: cs.CV**

- **简介: 该论文属驾驶分心检测任务，旨在解决单视角（仅驾驶员）建模忽略道路环境上下文的问题。作者在真实自然驾驶数据上，对比三种时空模型在单视图与双视图（驾驶员+道路）输入下的性能，发现融合效果高度依赖架构设计，强调需融合感知的多视图建模。**

- **链接: [https://arxiv.org/pdf/2512.20025v1](https://arxiv.org/pdf/2512.20025v1)**

> **作者:** Anthony Dontoh; Stephanie Ivey; Armstrong Aboah
>
> **摘要:** Despite increasing interest in computer vision-based distracted driving detection, most existing models rely exclusively on driver-facing views and overlook crucial environmental context that influences driving behavior. This study investigates whether incorporating road-facing views alongside driver-facing footage improves distraction detection accuracy in naturalistic driving conditions. Using synchronized dual-camera recordings from real-world driving, we benchmark three leading spatiotemporal action recognition architectures: SlowFast-R50, X3D-M, and SlowOnly-R50. Each model is evaluated under two input configurations: driver-only and stacked dual-view. Results show that while contextual inputs can improve detection in certain models, performance gains depend strongly on the underlying architecture. The single-pathway SlowOnly model achieved a 9.8 percent improvement with dual-view inputs, while the dual-pathway SlowFast model experienced a 7.2 percent drop in accuracy due to representational conflicts. These findings suggest that simply adding visual context is not sufficient and may lead to interference unless the architecture is specifically designed to support multi-view integration. This study presents one of the first systematic comparisons of single- and dual-view distraction detection models using naturalistic driving data and underscores the importance of fusion-aware design for future multimodal driver monitoring systems.
>
---
#### [new 044] BiCoR-Seg: Bidirectional Co-Refinement Framework for High-Resolution Remote Sensing Image Segmentation
- **分类: cs.CV**

- **简介: 该论文面向高分辨率遥感图像语义分割任务，旨在解决类间相似性高、类内差异大导致的边界模糊与误分问题。提出BiCoR-Seg框架，含热图驱动的双向信息协同模块（HBIS）、分层监督策略及跨层类嵌入Fisher判别损失，提升特征判别力与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.20255v1](https://arxiv.org/pdf/2512.20255v1)**

> **作者:** Jinghao Shi; Jianing Song
>
> **摘要:** High-resolution remote sensing image semantic segmentation (HRSS) is a fundamental yet critical task in the field of Earth observation. However, it has long faced the challenges of high inter-class similarity and large intra-class variability. Existing approaches often struggle to effectively inject abstract yet strongly discriminative semantic knowledge into pixel-level feature learning, leading to blurred boundaries and class confusion in complex scenes. To address these challenges, we propose Bidirectional Co-Refinement Framework for HRSS (BiCoR-Seg). Specifically, we design a Heatmap-driven Bidirectional Information Synergy Module (HBIS), which establishes a bidirectional information flow between feature maps and class embeddings by generating class-level heatmaps. Based on HBIS, we further introduce a hierarchical supervision strategy, where the interpretable heatmaps generated by each HBIS module are directly utilized as low-resolution segmentation predictions for supervision, thereby enhancing the discriminative capacity of shallow features. In addition, to further improve the discriminability of the embedding representations, we propose a cross-layer class embedding Fisher Discriminative Loss to enforce intra-class compactness and enlarge inter-class separability. Extensive experiments on the LoveDA, Vaihingen, and Potsdam datasets demonstrate that BiCoR-Seg achieves outstanding segmentation performance while offering stronger interpretability. The released code is available at https://github.com/ShiJinghao566/BiCoR-Seg.
>
---
#### [new 045] Generative Latent Coding for Ultra-Low Bitrate Image Compression
- **分类: cs.CV; eess.IV**

- **简介: 该论文属图像压缩任务，旨在解决超低码率下高保真与高真实感难以兼顾的问题。提出生成式潜在编码（GLC）架构，在VQ-VAE的语义丰富、感知对齐的潜在空间中进行变换编码，并引入类别化超模块与码预测监督以降码增质。**

- **链接: [https://arxiv.org/pdf/2512.20194v1](https://arxiv.org/pdf/2512.20194v1)**

> **作者:** Zhaoyang Jia; Jiahao Li; Bin Li; Houqiang Li; Yan Lu
>
> **备注:** Accepted at CVPR 2024
>
> **摘要:** Most existing image compression approaches perform transform coding in the pixel space to reduce its spatial redundancy. However, they encounter difficulties in achieving both high-realism and high-fidelity at low bitrate, as the pixel-space distortion may not align with human perception. To address this issue, we introduce a Generative Latent Coding (GLC) architecture, which performs transform coding in the latent space of a generative vector-quantized variational auto-encoder (VQ-VAE), instead of in the pixel space. The generative latent space is characterized by greater sparsity, richer semantic and better alignment with human perception, rendering it advantageous for achieving high-realism and high-fidelity compression. Additionally, we introduce a categorical hyper module to reduce the bit cost of hyper-information, and a code-prediction-based supervision to enhance the semantic consistency. Experiments demonstrate that our GLC maintains high visual quality with less than 0.04 bpp on natural images and less than 0.01 bpp on facial images. On the CLIC2020 test set, we achieve the same FID as MS-ILLM with 45% fewer bits. Furthermore, the powerful generative latent space enables various applications built on our GLC pipeline, such as image restoration and style transfer. The code is available at https://github.com/jzyustc/GLC.
>
---
#### [new 046] HyGE-Occ: Hybrid View-Transformation with 3D Gaussian and Edge Priors for 3D Panoptic Occupancy Prediction
- **分类: cs.CV**

- **简介: 该论文面向3D全景占据预测任务，旨在解决现有方法几何精度低、实例边界模糊的问题。提出HyGE-Occ框架：融合高斯深度与离散深度的混合视图变换提升几何一致性，并引入BEV边缘先验增强边界感知。**

- **链接: [https://arxiv.org/pdf/2512.19871v1](https://arxiv.org/pdf/2512.19871v1)**

> **作者:** Jong Wook Kim; Wonseok Roh; Ha Dam Baek; Pilhyeon Lee; Jonghyun Choi; Sangpil Kim
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** 3D Panoptic Occupancy Prediction aims to reconstruct a dense volumetric scene map by predicting the semantic class and instance identity of every occupied region in 3D space. Achieving such fine-grained 3D understanding requires precise geometric reasoning and spatially consistent scene representation across complex environments. However, existing approaches often struggle to maintain precise geometry and capture the precise spatial range of 3D instances critical for robust panoptic separation. To overcome these limitations, we introduce HyGE-Occ, a novel framework that leverages a hybrid view-transformation branch with 3D Gaussian and edge priors to enhance both geometric consistency and boundary awareness in 3D panoptic occupancy prediction. HyGE-Occ employs a hybrid view-transformation branch that fuses a continuous Gaussian-based depth representation with a discretized depth-bin formulation, producing BEV features with improved geometric consistency and structural coherence. In parallel, we extract edge maps from BEV features and use them as auxiliary information to learn edge cues. In our extensive experiments on the Occ3D-nuScenes dataset, HyGE-Occ outperforms existing work, demonstrating superior 3D geometric reasoning.
>
---
#### [new 047] LADLE-MM: Limited Annotation based Detector with Learned Ensembles for Multimodal Misinformation
- **分类: cs.CV**

- **简介: 该论文面向多模态虚假信息检测任务，解决标注数据稀缺与计算资源受限下的检测难题。提出LADLE-MM模型：采用双单模态分支加一融合BLIP多模态嵌入的分支，参数减少60.3%，在DGM4和VERITE上均超越现有方法。**

- **链接: [https://arxiv.org/pdf/2512.20257v1](https://arxiv.org/pdf/2512.20257v1)**

> **作者:** Daniele Cardullo; Simone Teglia; Irene Amerini
>
> **摘要:** With the rise of easily accessible tools for generating and manipulating multimedia content, realistic synthetic alterations to digital media have become a widespread threat, often involving manipulations across multiple modalities simultaneously. Recently, such techniques have been increasingly employed to distort narratives of important events and to spread misinformation on social media, prompting the development of misinformation detectors. In the context of misinformation conveyed through image-text pairs, several detection methods have been proposed. However, these approaches typically rely on computationally intensive architectures or require large amounts of annotated data. In this work we introduce LADLE-MM: Limited Annotation based Detector with Learned Ensembles for Multimodal Misinformation, a model-soup initialized multimodal misinformation detector designed to operate under a limited annotation setup and constrained training resources. LADLE-MM is composed of two unimodal branches and a third multimodal one that enhances image and text representations with additional multimodal embeddings extracted from BLIP, serving as fixed reference space. Despite using 60.3% fewer trainable parameters than previous state-of-the-art models, LADLE-MM achieves competitive performance on both binary and multi-label classification tasks on the DGM4 benchmark, outperforming existing methods when trained without grounding annotations. Moreover, when evaluated on the VERITE dataset, LADLE-MM outperforms current state-of-the-art approaches that utilize more complex architectures involving Large Vision-Language-Models, demonstrating the effective generalization ability in an open-set setting and strong robustness to unimodal bias.
>
---
#### [new 048] RANSAC Scoring Functions: Analysis and Reality Check
- **分类: cs.CV; stat.AP**

- **简介: 该论文属鲁棒几何模型拟合任务，聚焦RANSAC中评分函数的设计与评估。它分析现有方法（如MAGSAC++）的理论缺陷，统一建模噪声与异常值，提出新评估方法，并实验证明主流评分函数性能无显著差异。**

- **链接: [https://arxiv.org/pdf/2512.19850v1](https://arxiv.org/pdf/2512.19850v1)**

> **作者:** A. Shekhovtsov
>
> **摘要:** We revisit the problem of assigning a score (a quality of fit) to candidate geometric models -- one of the key components of RANSAC for robust geometric fitting. In a non-robust setting, the ``gold standard'' scoring function, known as the geometric error, follows from a probabilistic model with Gaussian noises. We extend it to spherical noises. In a robust setting, we consider a mixture with uniformly distributed outliers and show that a threshold-based parameterization leads to a unified view of likelihood-based and robust M-estimators and associated local optimization schemes. Next we analyze MAGSAC++ which stands out for two reasons. First, it achieves the best results according to existing benchmarks. Second, it makes quite different modeling assumptions and derivation steps. We discovered, however that the derivation does not correspond to sound principles and the resulting score function is in fact numerically equivalent to a simple Gaussian-uniform likelihood, a basic model within the proposed framework. Finally, we propose an experimental methodology for evaluating scoring functions: assuming either a large validation set, or a small random validation set in expectation. We find that all scoring functions, including using a learned inlier distribution, perform identically. In particular, MAGSAC++ score is found to be neither better performing than simple contenders nor less sensitive to the choice of the threshold hyperparameter. Our theoretical and experimental analysis thus comprehensively revisit the state-of-the-art, which is critical for any future research seeking to improve the methods or apply them to other robust fitting problems.
>
---
#### [new 049] SegEarth-R2: Towards Comprehensive Language-guided Segmentation for Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文面向遥感图像的语言引导分割任务，解决现有模型难以处理复杂地理语义（如多目标、多粒度、隐含意图）的问题。作者构建首个大规模基准LaSeRS，并提出新模型SegEarth-R2，引入空间注意力监督与灵活分割查询机制，显著提升复杂指令下的像素级定位能力。**

- **链接: [https://arxiv.org/pdf/2512.20013v1](https://arxiv.org/pdf/2512.20013v1)**

> **作者:** Zepeng Xin; Kaiyu Li; Luodi Chen; Wanchen Li; Yuchen Xiao; Hui Qiao; Weizhan Zhang; Deyu Meng; Xiangyong Cao
>
> **摘要:** Effectively grounding complex language to pixels in remote sensing (RS) images is a critical challenge for applications like disaster response and environmental monitoring. Current models can parse simple, single-target commands but fail when presented with complex geospatial scenarios, e.g., segmenting objects at various granularities, executing multi-target instructions, and interpreting implicit user intent. To drive progress against these failures, we present LaSeRS, the first large-scale dataset built for comprehensive training and evaluation across four critical dimensions of language-guided segmentation: hierarchical granularity, target multiplicity, reasoning requirements, and linguistic variability. By capturing these dimensions, LaSeRS moves beyond simple commands, providing a benchmark for complex geospatial reasoning. This addresses a critical gap: existing datasets oversimplify, leading to sensitivity-prone real-world models. We also propose SegEarth-R2, an MLLM architecture designed for comprehensive language-guided segmentation in RS, which directly confronts these challenges. The model's effectiveness stems from two key improvements: (1) a spatial attention supervision mechanism specifically handles the localization of small objects and their components, and (2) a flexible and efficient segmentation query mechanism that handles both single-target and multi-target scenarios. Experimental results demonstrate that our SegEarth-R2 achieves outstanding performance on LaSeRS and other benchmarks, establishing a powerful baseline for the next generation of geospatial segmentation. All data and code will be released at https://github.com/earth-insights/SegEarth-R2.
>
---
#### [new 050] CRAFT: Continuous Reasoning and Agentic Feedback Tuning for Multimodal Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属多模态文本到图像生成任务，旨在解决现有推理式优化方法不可控、难解释、难终止的问题。提出无需训练、模型无关的CRAFT框架：将提示分解为结构化视觉问题，用VLM验证图像，LLM代理针对性编辑失败约束，并设明确停止条件，提升准确性与可控性。**

- **链接: [https://arxiv.org/pdf/2512.20362v1](https://arxiv.org/pdf/2512.20362v1)**

> **作者:** V. Kovalev; A. Kuvshinov; A. Buzovkin; D. Pokidov; D. Timonin
>
> **备注:** 37 pages, 42 figures
>
> **摘要:** Recent work has shown that inference-time reasoning and reflection can improve text-to-image generation without retraining. However, existing approaches often rely on implicit, holistic critiques or unconstrained prompt rewrites, making their behavior difficult to interpret, control, or stop reliably. In contrast, large language models have benefited from explicit, structured forms of **thinking** based on verification, targeted correction, and early stopping. We introduce CRAFT (Continuous Reasoning and Agentic Feedback Tuning), a training-free, model-agnostic framework that brings this structured reasoning paradigm to multimodal image generation. CRAFT decomposes a prompt into dependency-structured visual questions, veries generated images using a vision-language model, and applies targeted prompt edits through an LLM agent only where constraints fail. The process iterates with an explicit stopping criterion once all constraints are satised, yielding an interpretable and controllable inference-time renement loop. Across multiple model families and challenging benchmarks, CRAFT consistently improves compositional accuracy, text rendering, and preference-based evaluations, with particularly strong gains for lightweight generators. Importantly, these improvements incur only a negligible inference-time overhead, allowing smaller or cheaper models to approach the quality of substantially more expensive systems. Our results suggest that explicitly structured, constraint-driven inference-time reasoning is a key ingredient for improving the reliability of multimodal generative models.
>
---
#### [new 051] JDPNet: A Network Based on Joint Degradation Processing for Underwater Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于水下图像增强任务，旨在解决多种非线性耦合退化（如模糊、偏色、低对比）难以联合建模的问题。提出JDPNet网络，通过联合特征挖掘模块、概率引导策略和AquaBalanceLoss损失函数，统一建模并协同优化耦合退化，在多数据集上实现SOTA性能与效率平衡。**

- **链接: [https://arxiv.org/pdf/2512.20213v1](https://arxiv.org/pdf/2512.20213v1)**

> **作者:** Tao Ye; Hongbin Ren; Chongbing Zhang; Haoran Chen; Xiaosong Li
>
> **摘要:** Given the complexity of underwater environments and the variability of water as a medium, underwater images are inevitably subject to various types of degradation. The degradations present nonlinear coupling rather than simple superposition, which renders the effective processing of such coupled degradations particularly challenging. Most existing methods focus on designing specific branches, modules, or strategies for specific degradations, with little attention paid to the potential information embedded in their coupling. Consequently, they struggle to effectively capture and process the nonlinear interactions of multiple degradations from a bottom-up perspective. To address this issue, we propose JDPNet, a joint degradation processing network, that mines and unifies the potential information inherent in coupled degradations within a unified framework. Specifically, we introduce a joint feature-mining module, along with a probabilistic bootstrap distribution strategy, to facilitate effective mining and unified adjustment of coupled degradation features. Furthermore, to balance color, clarity, and contrast, we design a novel AquaBalanceLoss to guide the network in learning from multiple coupled degradation losses. Experiments on six publicly available underwater datasets, as well as two new datasets constructed in this study, show that JDPNet exhibits state-of-the-art performance while offering a better tradeoff between performance, parameter size, and computational cost.
>
---
#### [new 052] CoDi -- an exemplar-conditioned diffusion model for low-shot counting
- **分类: cs.CV**

- **简介: 该论文面向低样本目标计数任务，解决密集小目标下定位不准、计数难的问题。提出首个基于扩散模型的CoDi方法，通过 exemplar 条件化模块引导密度图生成，兼顾精度与定位能力，在多个基准上显著超越现有方法。**

- **链接: [https://arxiv.org/pdf/2512.20153v1](https://arxiv.org/pdf/2512.20153v1)**

> **作者:** Grega Šuštar; Jer Pelhan; Alan Lukežič; Matej Kristan
>
> **摘要:** Low-shot object counting addresses estimating the number of previously unobserved objects in an image using only few or no annotated test-time exemplars. A considerable challenge for modern low-shot counters are dense regions with small objects. While total counts in such situations are typically well addressed by density-based counters, their usefulness is limited by poor localization capabilities. This is better addressed by point-detection-based counters, which are based on query-based detectors. However, due to limited number of pre-trained queries, they underperform on images with very large numbers of objects, and resort to ad-hoc techniques like upsampling and tiling. We propose CoDi, the first latent diffusion-based low-shot counter that produces high-quality density maps on which object locations can be determined by non-maxima suppression. Our core contribution is the new exemplar-based conditioning module that extracts and adjusts the object prototypes to the intermediate layers of the denoising network, leading to accurate object location estimation. On FSC benchmark, CoDi outperforms state-of-the-art by 15% MAE, 13% MAE and 10% MAE in the few-shot, one-shot, and reference-less scenarios, respectively, and sets a new state-of-the-art on MCAC benchmark by outperforming the top method by 44% MAE. The code is available at https://github.com/gsustar/CoDi.
>
---
#### [new 053] A Dual-Branch Local-Global Framework for Cross-Resolution Land Cover Mapping
- **分类: cs.CV**

- **简介: 该论文面向跨分辨率土地覆盖制图任务，解决粗标签监督下难以学习细粒度空间结构的问题。提出双分支框架DDTM：扩散分支细化局部语义，Transformer分支建模全局上下文，并设计置信度评估模块抑制伪标签噪声。**

- **链接: [https://arxiv.org/pdf/2512.19990v1](https://arxiv.org/pdf/2512.19990v1)**

> **作者:** Peng Gao; Ke Li; Di Wang; Yongshan Zhu; Yiming Zhang; Xuemei Luo; Yifeng Wang
>
> **摘要:** Cross-resolution land cover mapping aims to produce high-resolution semantic predictions from coarse or low-resolution supervision, yet the severe resolution mismatch makes effective learning highly challenging. Existing weakly supervised approaches often struggle to align fine-grained spatial structures with coarse labels, leading to noisy supervision and degraded mapping accuracy. To tackle this problem, we propose DDTM, a dual-branch weakly supervised framework that explicitly decouples local semantic refinement from global contextual reasoning. Specifically, DDTM introduces a diffusion-based branch to progressively refine fine-scale local semantics under coarse supervision, while a transformer-based branch enforces long-range contextual consistency across large spatial extents. In addition, we design a pseudo-label confidence evaluation module to mitigate noise induced by cross-resolution inconsistencies and to selectively exploit reliable supervisory signals. Extensive experiments demonstrate that DDTM establishes a new state-of-the-art on the Chesapeake Bay benchmark, achieving 66.52\% mIoU and substantially outperforming prior weakly supervised methods. The code is available at https://github.com/gpgpgp123/DDTM.
>
---
#### [new 054] UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis
- **分类: cs.CV**

- **简介: 该论文面向新型视图合成（NVS）任务，旨在解决稀疏输入下快速生成高质量、3D一致新视角图像的难题。提出UMAMI框架：用双向Transformer统一编码多视图与光线信息，分叉出轻量回归头（保真渲染可见区）和掩码自回归扩散头（补全遮挡/未见区），端到端训练，兼顾速度与质量。**

- **链接: [https://arxiv.org/pdf/2512.20107v1](https://arxiv.org/pdf/2512.20107v1)**

> **作者:** Thanh-Tung Le; Tuan Pham; Tung Nguyen; Deying Kong; Xiaohui Xie; Stephan Mandt
>
> **备注:** Accepted to NeurIPS 2025. The first two authors contributed equally
>
> **摘要:** Novel view synthesis (NVS) seeks to render photorealistic, 3D-consistent images of a scene from unseen camera poses given only a sparse set of posed views. Existing deterministic networks render observed regions quickly but blur unobserved areas, whereas stochastic diffusion-based methods hallucinate plausible content yet incur heavy training- and inference-time costs. In this paper, we propose a hybrid framework that unifies the strengths of both paradigms. A bidirectional transformer encodes multi-view image tokens and Plucker-ray embeddings, producing a shared latent representation. Two lightweight heads then act on this representation: (i) a feed-forward regression head that renders pixels where geometry is well constrained, and (ii) a masked autoregressive diffusion head that completes occluded or unseen regions. The entire model is trained end-to-end with joint photometric and diffusion losses, without handcrafted 3D inductive biases, enabling scalability across diverse scenes. Experiments demonstrate that our method attains state-of-the-art image quality while reducing rendering time by an order of magnitude compared with fully generative baselines.
>
---
#### [new 055] Progressive Learned Image Compression for Machine Perception
- **分类: cs.CV; eess.IV**

- **简介: 该论文属机器感知导向的图像压缩任务，旨在解决现有学习型编解码器缺乏细粒度可伸缩渐进压缩的问题。提出PICM-Net，基于三元平面编码实现渐进压缩，并设计自适应解码控制器，动态调整解码层级以保障下游分类置信度。**

- **链接: [https://arxiv.org/pdf/2512.20070v1](https://arxiv.org/pdf/2512.20070v1)**

> **作者:** Jungwoo Kim; Jun-Hyuk Kim; Jong-Seok Lee
>
> **摘要:** Recent advances in learned image codecs have been extended from human perception toward machine perception. However, progressive image compression with fine granular scalability (FGS)-which enables decoding a single bitstream at multiple quality levels-remains unexplored for machine-oriented codecs. In this work, we propose a novel progressive learned image compression codec for machine perception, PICM-Net, based on trit-plane coding. By analyzing the difference between human- and machine-oriented rate-distortion priorities, we systematically examine the latent prioritization strategies in terms of machine-oriented codecs. To further enhance real-world adaptability, we design an adaptive decoding controller, which dynamically determines the necessary decoding level during inference time to maintain the desired confidence of downstream machine prediction. Extensive experiments demonstrate that our approach enables efficient and adaptive progressive transmission while maintaining high performance in the downstream classification task, establishing a new paradigm for machine-aware progressive image compression.
>
---
#### [new 056] How Much 3D Do Video Foundation Models Encode?
- **分类: cs.CV; cs.AI**

- **简介: 该论文属模型分析任务，旨在探究视频基础模型（VidFMs）是否自发习得3D理解。作者提出首个模型无关的评估框架，通过轻量读出头量化VidFMs对多种3D属性的感知能力，发现先进视频生成模型在无3D训练数据下仍具备强3D理解，甚至超越专用3D模型。**

- **链接: [https://arxiv.org/pdf/2512.19949v1](https://arxiv.org/pdf/2512.19949v1)**

> **作者:** Zixuan Huang; Xiang Li; Zhaoyang Lv; James M. Rehg
>
> **备注:** Project Page: https://vidfm-3d-probe.github.io
>
> **摘要:** Videos are continuous 2D projections of 3D worlds. After training on large video data, will global 3D understanding naturally emerge? We study this by quantifying the 3D understanding of existing Video Foundation Models (VidFMs) pretrained on vast video data. We propose the first model-agnostic framework that measures the 3D awareness of various VidFMs by estimating multiple 3D properties from their features via shallow read-outs. Our study presents meaningful findings regarding the 3D awareness of VidFMs on multiple axes. In particular, we show that state-of-the-art video generation models exhibit a strong understanding of 3D objects and scenes, despite not being trained on any 3D data. Such understanding can even surpass that of large expert models specifically trained for 3D tasks. Our findings, together with the 3D benchmarking of major VidFMs, provide valuable observations for building scalable 3D models.
>
---
#### [new 057] Vehicle-centric Perception via Multimodal Structured Pre-training
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属车辆中心感知任务，旨在解决现有预训练模型缺乏车辆知识导致表征能力弱的问题。提出VehicleMAE-V2模型，通过引入对称性、轮廓和语义三类结构先验（SMM/CRM/SRM模块），结合多模态结构化预训练，并构建Autobot4M数据集，显著提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2512.19934v1](https://arxiv.org/pdf/2512.19934v1)**

> **作者:** Wentao Wu; Xiao Wang; Chenglong Li; Jin Tang; Bin Luo
>
> **备注:** Journal extension of VehicleMAE (AAAI 2024)
>
> **摘要:** Vehicle-centric perception plays a crucial role in many intelligent systems, including large-scale surveillance systems, intelligent transportation, and autonomous driving. Existing approaches lack effective learning of vehicle-related knowledge during pre-training, resulting in poor capability for modeling general vehicle perception representations. To handle this problem, we propose VehicleMAE-V2, a novel vehicle-centric pre-trained large model. By exploring and exploiting vehicle-related multimodal structured priors to guide the masked token reconstruction process, our approach can significantly enhance the model's capability to learn generalizable representations for vehicle-centric perception. Specifically, we design the Symmetry-guided Mask Module (SMM), Contour-guided Representation Module (CRM) and Semantics-guided Representation Module (SRM) to incorporate three kinds of structured priors into token reconstruction including symmetry, contour and semantics of vehicles respectively. SMM utilizes the vehicle symmetry constraints to avoid retaining symmetric patches and can thus select high-quality masked image patches and reduce information redundancy. CRM minimizes the probability distribution divergence between contour features and reconstructed features and can thus preserve holistic vehicle structure information during pixel-level reconstruction. SRM aligns image-text features through contrastive learning and cross-modal distillation to address the feature confusion caused by insufficient semantic understanding during masked reconstruction. To support the pre-training of VehicleMAE-V2, we construct Autobot4M, a large-scale dataset comprising approximately 4 million vehicle images and 12,693 text descriptions. Extensive experiments on five downstream tasks demonstrate the superior performance of VehicleMAE-V2.
>
---
#### [new 058] ${D}^{3}${ETOR}: ${D}$ebate-Enhanced Pseudo Labeling and Frequency-Aware Progressive ${D}$ebiasing for Weakly-Supervised Camouflaged Object ${D}$etection with Scribble Annotations
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向弱监督伪装目标检测（WSCOD）任务，解决 scribble 标注下伪标签不可靠与标注偏差两大问题。提出 D³ETOR 框架：第一阶段通过多智能体辩论增强 SAM 生成高质量伪标签；第二阶段设计频率感知的渐进去偏网络 FADeNet 缓解 scribble 偏差。**

- **链接: [https://arxiv.org/pdf/2512.20260v1](https://arxiv.org/pdf/2512.20260v1)**

> **作者:** Jiawei Ge; Jiuxin Cao; Xinyi Li; Xuelin Zhu; Chang Liu; Bo Liu; Chen Feng; Ioannis Patras
>
> **摘要:** Weakly-Supervised Camouflaged Object Detection (WSCOD) aims to locate and segment objects that are visually concealed within their surrounding scenes, relying solely on sparse supervision such as scribble annotations. Despite recent progress, existing WSCOD methods still lag far behind fully supervised ones due to two major limitations: (1) the pseudo masks generated by general-purpose segmentation models (e.g., SAM) and filtered via rules are often unreliable, as these models lack the task-specific semantic understanding required for effective pseudo labeling in COD; and (2) the neglect of inherent annotation bias in scribbles, which hinders the model from capturing the global structure of camouflaged objects. To overcome these challenges, we propose ${D}^{3}$ETOR, a two-stage WSCOD framework consisting of Debate-Enhanced Pseudo Labeling and Frequency-Aware Progressive Debiasing. In the first stage, we introduce an adaptive entropy-driven point sampling method and a multi-agent debate mechanism to enhance the capability of SAM for COD, improving the interpretability and precision of pseudo masks. In the second stage, we design FADeNet, which progressively fuses multi-level frequency-aware features to balance global semantic understanding with local detail modeling, while dynamically reweighting supervision strength across regions to alleviate scribble bias. By jointly exploiting the supervision signals from both the pseudo masks and scribble semantics, ${D}^{3}$ETOR significantly narrows the gap between weakly and fully supervised COD, achieving state-of-the-art performance on multiple benchmarks.
>
---
#### [new 059] Multi-Grained Text-Guided Image Fusion for Multi-Exposure and Multi-Focus Scenarios
- **分类: cs.CV**

- **简介: 该论文属图像融合任务，旨在解决多曝光/多焦点图像融合中动态范围与焦深差异导致的细节丢失问题。提出MTIF方法：引入多粒度文本描述、分层跨模态调制、多级监督对齐及显著性驱动语义增强，提升融合质量。**

- **链接: [https://arxiv.org/pdf/2512.20556v1](https://arxiv.org/pdf/2512.20556v1)**

> **作者:** Mingwei Tang; Jiahao Nie; Guang Yang; Ziqing Cui; Jie Li
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Image fusion aims to synthesize a single high-quality image from a pair of inputs captured under challenging conditions, such as differing exposure levels or focal depths. A core challenge lies in effectively handling disparities in dynamic range and focus depth between the inputs. With the advent of vision-language models, recent methods incorporate textual descriptions as auxiliary guidance to enhance fusion quality. However, simply incorporating coarse-grained descriptions hampers the understanding of fine-grained details and poses challenges for precise cross-modal alignment. To address these limitations, we propose Multi-grained Text-guided Image Fusion (MTIF), a novel fusion paradigm with three key designs. First, it introduces multi-grained textual descriptions that separately capture fine details, structural cues, and semantic content, guiding image fusion through a hierarchical cross-modal modulation module. Second, it involves supervision signals at each granularity to facilitate alignment between visual and textual features and enhance the utility of auxiliary text. Third, it adopts a saliency-driven enrichment module to augment training data with dense semantic content, further strengthening the cross-modal modulation and alignment. Extensive experiments show that MTIF consistently outperforms previous methods on both multi-exposure and multi-focus image fusion tasks.
>
---
#### [new 060] SemanticGen: Video Generation in Semantic Space
- **分类: cs.CV**

- **简介: 该论文属视频生成任务，旨在解决现有模型在VAE隐空间生成视频时收敛慢、计算贵、难扩展至长视频的问题。提出SemanticGen：先用扩散模型生成紧凑语义特征（全局规划），再条件生成VAE隐变量以还原像素，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2512.20619v1](https://arxiv.org/pdf/2512.20619v1)**

> **作者:** Jianhong Bai; Xiaoshi Wu; Xintao Wang; Fu Xiao; Yuanxing Zhang; Qinghe Wang; Xiaoyu Shi; Menghan Xia; Zuozhu Liu; Haoji Hu; Pengfei Wan; Kun Gai
>
> **备注:** Project page: https://jianhongbai.github.io/SemanticGen/
>
> **摘要:** State-of-the-art video generative models typically learn the distribution of video latents in the VAE space and map them to pixels using a VAE decoder. While this approach can generate high-quality videos, it suffers from slow convergence and is computationally expensive when generating long videos. In this paper, we introduce SemanticGen, a novel solution to address these limitations by generating videos in the semantic space. Our main insight is that, due to the inherent redundancy in videos, the generation process should begin in a compact, high-level semantic space for global planning, followed by the addition of high-frequency details, rather than directly modeling a vast set of low-level video tokens using bi-directional attention. SemanticGen adopts a two-stage generation process. In the first stage, a diffusion model generates compact semantic video features, which define the global layout of the video. In the second stage, another diffusion model generates VAE latents conditioned on these semantic features to produce the final output. We observe that generation in the semantic space leads to faster convergence compared to the VAE latent space. Our method is also effective and computationally efficient when extended to long video generation. Extensive experiments demonstrate that SemanticGen produces high-quality videos and outperforms state-of-the-art approaches and strong baselines.
>
---
#### [new 061] SmartSplat: Feature-Smart Gaussians for Scalable Compression of Ultra-High-Resolution Images
- **分类: cs.CV**

- **简介: 该论文属图像压缩任务，旨在解决超高清图像高效压缩与高保真重建的难题。提出SmartSplat框架，融合梯度-颜色引导采样、排他性均匀采样和尺度自适应颜色初始化，提升高斯基元的空间覆盖与色彩表征能力，在极高压缩比下保持高质量重建。**

- **链接: [https://arxiv.org/pdf/2512.20377v1](https://arxiv.org/pdf/2512.20377v1)**

> **作者:** Linfei Li; Lin Zhang; Zhong Wang; Ying Shen
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Recent advances in generative AI have accelerated the production of ultra-high-resolution visual content, posing significant challenges for efficient compression and real-time decoding on end-user devices. Inspired by 3D Gaussian Splatting, recent 2D Gaussian image models improve representation efficiency, yet existing methods struggle to balance compression ratio and reconstruction fidelity in ultra-high-resolution scenarios. To address this issue, we propose SmartSplat, a highly adaptive and feature-aware GS-based image compression framework that supports arbitrary image resolutions and compression ratios. SmartSplat leverages image-aware features such as gradients and color variances, introducing a Gradient-Color Guided Variational Sampling strategy together with an Exclusion-based Uniform Sampling scheme to improve the non-overlapping coverage of Gaussian primitives in pixel space. In addition, we propose a Scale-Adaptive Gaussian Color Sampling method to enhance color initialization across scales. Through joint optimization of spatial layout, scale, and color initialization, SmartSplat efficiently captures both local structures and global textures using a limited number of Gaussians, achieving high reconstruction quality under strong compression. Extensive experiments on DIV8K and a newly constructed 16K dataset demonstrate that SmartSplat consistently outperforms state-of-the-art methods at comparable compression ratios and exceeds their compression limits, showing strong scalability and practical applicability. The code is publicly available at https://github.com/lif314/SmartSplat.
>
---
#### [new 062] UTDesign: A Unified Framework for Stylized Text Editing and Generation in Graphic Design Images
- **分类: cs.CV**

- **简介: 该论文提出UTDesign框架，解决AI图形设计中文字渲染精度低（尤其小字号、中文）问题。任务属 stylized text editing & generation。工作包括：1）DiT文本风格迁移模型生成RGBA文字；2）多模态条件编码器实现背景/布局/提示驱动的文本生成；3）集成T2I与MLLM构建端到端文本到设计流水线。**

- **链接: [https://arxiv.org/pdf/2512.20479v1](https://arxiv.org/pdf/2512.20479v1)**

> **作者:** Yiming Zhao; Yuanpeng Gao; Yuxuan Luo; Jiwei Duan; Shisong Lin; Longfei Xiong; Zhouhui Lian
>
> **备注:** 22 pages, 25 figures, SIGGRAPH Asia 2025, Conference Paper
>
> **摘要:** AI-assisted graphic design has emerged as a powerful tool for automating the creation and editing of design elements such as posters, banners, and advertisements. While diffusion-based text-to-image models have demonstrated strong capabilities in visual content generation, their text rendering performance, particularly for small-scale typography and non-Latin scripts, remains limited. In this paper, we propose UTDesign, a unified framework for high-precision stylized text editing and conditional text generation in design images, supporting both English and Chinese scripts. Our framework introduces a novel DiT-based text style transfer model trained from scratch on a synthetic dataset, capable of generating transparent RGBA text foregrounds that preserve the style of reference glyphs. We further extend this model into a conditional text generation framework by training a multi-modal condition encoder on a curated dataset with detailed text annotations, enabling accurate, style-consistent text synthesis conditioned on background images, prompts, and layout specifications. Finally, we integrate our approach into a fully automated text-to-design (T2D) pipeline by incorporating pre-trained text-to-image (T2I) models and an MLLM-based layout planner. Extensive experiments demonstrate that UTDesign achieves state-of-the-art performance among open-source methods in terms of stylistic consistency and text accuracy, and also exhibits unique advantages compared to proprietary commercial approaches. Code and data for this paper are available at https://github.com/ZYM-PKU/UTDesign.
>
---
#### [new 063] WSD-MIL: Window Scale Decay Multiple Instance Learning for Whole Slide Image Classification
- **分类: cs.CV**

- **简介: 该论文属计算病理学中的全切片图像（WSI）分类任务，旨在解决Transformer-MIL方法因固定尺度注意力和高计算复杂度导致的多尺度肿瘤区域建模不足问题。提出WSD-MIL：含窗尺度衰减注意力模块（集群采样+渐进缩窗）和SE区域门控模块，提升多尺度建模能力并降内存62%。**

- **链接: [https://arxiv.org/pdf/2512.19982v1](https://arxiv.org/pdf/2512.19982v1)**

> **作者:** Le Feng; Li Xiao
>
> **摘要:** In recent years, the integration of pre-trained foundational models with multiple instance learning (MIL) has improved diagnostic accuracy in computational pathology. However, existing MIL methods focus on optimizing feature extractors and aggregation strategies while overlooking the complex semantic relationships among instances within whole slide image (WSI). Although Transformer-based MIL approaches aiming to model instance dependencies, the quadratic computational complexity limits their scalability to large-scale WSIs. Moreover, due to the pronounced variations in tumor region scales across different WSIs, existing Transformer-based methods employing fixed-scale attention mechanisms face significant challenges in precisely capturing local instance correlations and fail to account for the distance-based decay effect of patch relevance. To address these challenges, we propose window scale decay MIL (WSD-MIL), designed to enhance the capacity to model tumor regions of varying scales while improving computational efficiency. WSD-MIL comprises: 1) a window scale decay based attention module, which employs a cluster-based sampling strategy to reduce computational costs while progressively decaying attention window-scale to capture local instance relationships at varying scales; and 2) a squeeze-and-excitation based region gate module, which dynamically adjusts window weights to enhance global information modeling. Experimental results demonstrate that WSD-MIL achieves state-of-the-art performance on the CAMELYON16 and TCGA-BRCA datasets while reducing 62% of the computational memory. The code will be publicly available.
>
---
#### [new 064] The devil is in the details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection
- **分类: cs.CV**

- **简介: 该论文面向视频虚拟试衣（VVT）任务，旨在解决现有DiT方法在细粒度服饰动态建模、背景一致性保持及计算开销大等问题。提出KeyTailor框架与ViT-HD数据集，通过关键帧驱动的细节注入策略，在不修改DiT结构下提升服饰真实感与背景完整性。**

- **链接: [https://arxiv.org/pdf/2512.20340v1](https://arxiv.org/pdf/2512.20340v1)**

> **作者:** Qingdong He; Xueqin Chen; Yanjie Pan; Peng Tang; Pengcheng Xu; Zhenye Gan; Chengjie Wang; Xiaobin Hu; Jiangning Zhang; Yabiao Wang
>
> **摘要:** Although diffusion transformer (DiT)-based video virtual try-on (VVT) has made significant progress in synthesizing realistic videos, existing methods still struggle to capture fine-grained garment dynamics and preserve background integrity across video frames. They also incur high computational costs due to additional interaction modules introduced into DiTs, while the limited scale and quality of existing public datasets also restrict model generalization and effective training. To address these challenges, we propose a novel framework, KeyTailor, along with a large-scale, high-definition dataset, ViT-HD. The core idea of KeyTailor is a keyframe-driven details injection strategy, motivated by the fact that keyframes inherently contain both foreground dynamics and background consistency. Specifically, KeyTailor adopts an instruction-guided keyframe sampling strategy to filter informative frames from the input video. Subsequently,two tailored keyframe-driven modules, the garment details enhancement module and the collaborative background optimization module, are employed to distill garment dynamics into garment-related latents and to optimize the integrity of background latents, both guided by keyframes.These enriched details are then injected into standard DiT blocks together with pose, mask, and noise latents, enabling efficient and realistic try-on video synthesis. This design ensures consistency without explicitly modifying the DiT architecture, while simultaneously avoiding additional complexity. In addition, our dataset ViT-HD comprises 15, 070 high-quality video samples at a resolution of 810*1080, covering diverse garments. Extensive experiments demonstrate that KeyTailor outperforms state-of-the-art baselines in terms of garment fidelity and background integrity across both dynamic and static scenarios.
>
---
#### [new 065] VALLR-Pin: Dual-Decoding Visual Speech Recognition for Mandarin with Pinyin-Guided LLM Refinement
- **分类: cs.CV**

- **简介: 该论文面向中文唇读任务，解决 Mandarin 视觉语音识别中因视位模糊和同音字多导致的识别不准问题。提出 VALLR-Pin 框架：双解码器联合预测汉字与拼音，再用拼音引导大模型对候选文本进行同音纠错，并通过合成噪声数据微调 LLM 以适配模型错误模式。**

- **链接: [https://arxiv.org/pdf/2512.20032v1](https://arxiv.org/pdf/2512.20032v1)**

> **作者:** Chang Sun; Dongliang Xie; Bo Qin; Hong Yang
>
> **摘要:** Visual Speech Recognition aims to transcribe spoken words from silent lip-motion videos. This task is particularly challenging for Mandarin, as visemes are highly ambiguous and homophones are prevalent. We propose VALLR-Pin, a novel two-stage framework that extends the recent VALLR architecture from English to Mandarin. First, a shared video encoder feeds into dual decoders, which jointly predict both Chinese character sequences and their standard Pinyin romanization. The multi-task learning of character and phonetic outputs fosters robust visual-semantic representations. During inference, the text decoder generates multiple candidate transcripts. We construct a prompt by concatenating the Pinyin output with these candidate Chinese sequences and feed it to a large language model to resolve ambiguities and refine the transcription. This provides the LLM with explicit phonetic context to correct homophone-induced errors. Finally, we fine-tune the LLM on synthetic noisy examples: we generate imperfect Pinyin-text pairs from intermediate VALLR-Pin checkpoints using the training data, creating instruction-response pairs for error correction. This endows the LLM with awareness of our model's specific error patterns. In summary, VALLR-Pin synergizes visual features with phonetic and linguistic context to improve Mandarin lip-reading performance.
>
---
#### [new 066] Effect of Activation Function and Model Optimizer on the Performance of Human Activity Recognition System Using Various Deep Learning Models
- **分类: cs.CV**

- **简介: 该论文属人类活动识别（HAR）任务，旨在探究激活函数与优化器组合对深度学习模型性能的影响。作者在BiLSTM和ConvLSTM上系统实验ReLU/Sigmoid/Tanh与SGD/Adam/RMSprop/Adagrad的组合，在HMDB51与UCF101子集上验证，发现ConvLSTM+Adam/RMSprop最优，达99%精度。**

- **链接: [https://arxiv.org/pdf/2512.20104v1](https://arxiv.org/pdf/2512.20104v1)**

> **作者:** Subrata Kumer Paula; Dewan Nafiul Islam Noora; Rakhi Rani Paula; Md. Ekramul Hamidb; Fahmid Al Faridc; Hezerul Abdul Karimd; Md. Maruf Al Hossain Princee; Abu Saleh Musa Miahb
>
> **摘要:** Human Activity Recognition (HAR) plays a vital role in healthcare, surveillance, and innovative environments, where reliable action recognition supports timely decision-making and automation. Although deep learning-based HAR systems are widely adopted, the impact of Activation Functions (AFs) and Model Optimizers (MOs) on performance has not been sufficiently analyzed, particularly regarding how their combinations influence model behavior in practical scenarios. Most existing studies focus on architecture design, while the interaction between AF and MO choices remains relatively unexplored. In this work, we investigate the effect of three commonly used activation functions (ReLU, Sigmoid, and Tanh) combined with four optimization algorithms (SGD, Adam, RMSprop, and Adagrad) using two recurrent deep learning architectures, namely BiLSTM and ConvLSTM. Experiments are conducted on six medically relevant activity classes selected from the HMDB51 and UCF101 datasets, considering their suitability for healthcare-oriented HAR applications. Our experimental results show that ConvLSTM consistently outperforms BiLSTM across both datasets. ConvLSTM, combined with Adam or RMSprop, achieves an accuracy of up to 99.00%, demonstrating strong spatio-temporal learning capabilities and stable performance. While BiLSTM performs reasonably well on UCF101, with accuracy approaching 98.00%, its performance drops to approximately 60.00% on HMDB51, indicating limited robustness across datasets and weaker sensitivity to AF and MO variations. This study provides practical insights for optimizing HAR systems, particularly for real-world healthcare environments where fast and precise activity detection is critical.
>
---
#### [new 067] SpatialTree: How Spatial Abilities Branch Out in MLLMs
- **分类: cs.CV**

- **简介: 该论文提出SpatialTree，一种认知科学启发的空间能力四层分类法（L1–L4），构建首个能力中心化分层基准，评估27种子能力；揭示能力间相关性与迁移规律；提出auto-think策略提升RL训练效果，系统推进MLLM空间能力建模与提升。**

- **链接: [https://arxiv.org/pdf/2512.20617v1](https://arxiv.org/pdf/2512.20617v1)**

> **作者:** Yuxi Xiao; Longfei Li; Shen Yan; Xinhang Liu; Sida Peng; Yunchao Wei; Xiaowei Zhou; Bingyi Kang
>
> **备注:** webpage: https://spatialtree.github.io/
>
> **摘要:** Cognitive science suggests that spatial ability develops progressively-from perception to reasoning and interaction. Yet in multimodal LLMs (MLLMs), this hierarchy remains poorly understood, as most studies focus on a narrow set of tasks. We introduce SpatialTree, a cognitive-science-inspired hierarchy that organizes spatial abilities into four levels: low-level perception (L1), mental mapping (L2), simulation (L3), and agentic competence (L4). Based on this taxonomy, we construct the first capability-centric hierarchical benchmark, thoroughly evaluating mainstream MLLMs across 27 sub-abilities. The evaluation results reveal a clear structure: L1 skills are largely orthogonal, whereas higher-level skills are strongly correlated, indicating increasing interdependency. Through targeted supervised fine-tuning, we uncover a surprising transfer dynamic-negative transfer within L1, but strong cross-level transfer from low- to high-level abilities with notable synergy. Finally, we explore how to improve the entire hierarchy. We find that naive RL that encourages extensive "thinking" is unreliable: it helps complex reasoning but hurts intuitive perception. We propose a simple auto-think strategy that suppresses unnecessary deliberation, enabling RL to consistently improve performance across all levels. By building SpatialTree, we provide a proof-of-concept framework for understanding and systematically scaling spatial abilities in MLLMs.
>
---
#### [new 068] SirenPose: Dynamic Scene Reconstruction via Geometric Supervision
- **分类: cs.CV**

- **简介: 该论文提出SirenPose，面向单目视频动态3D场景重建任务，解决运动失真、时序不一致与几何不准问题。通过融合SIREN的周期性激活与关键点几何监督，引入物理约束和图神经网络，提升重建精度、时序连贯性与动态保真度。**

- **链接: [https://arxiv.org/pdf/2512.20531v1](https://arxiv.org/pdf/2512.20531v1)**

> **作者:** Kaitong Cai; Jensen Zhang; Jing Yang; Keze Wang
>
> **备注:** Under submission
>
> **摘要:** We introduce SirenPose, a geometry-aware loss formulation that integrates the periodic activation properties of sinusoidal representation networks with keypoint-based geometric supervision, enabling accurate and temporally consistent reconstruction of dynamic 3D scenes from monocular videos. Existing approaches often struggle with motion fidelity and spatiotemporal coherence in challenging settings involving fast motion, multi-object interaction, occlusion, and rapid scene changes. SirenPose incorporates physics inspired constraints to enforce coherent keypoint predictions across both spatial and temporal dimensions, while leveraging high frequency signal modeling to capture fine grained geometric details. We further expand the UniKPT dataset to 600,000 annotated instances and integrate graph neural networks to model keypoint relationships and structural correlations. Extensive experiments on benchmarks including Sintel, Bonn, and DAVIS demonstrate that SirenPose consistently outperforms state-of-the-art methods. On DAVIS, SirenPose achieves a 17.8 percent reduction in FVD, a 28.7 percent reduction in FID, and a 6.0 percent improvement in LPIPS compared to MoSCA. It also improves temporal consistency, geometric accuracy, user score, and motion smoothness. In pose estimation, SirenPose outperforms Monst3R with lower absolute trajectory error as well as reduced translational and rotational relative pose error, highlighting its effectiveness in handling rapid motion, complex dynamics, and physically plausible reconstruction.
>
---
#### [new 069] Beyond Motion Pattern: An Empirical Study of Physical Forces for Human Motion Understanding
- **分类: cs.CV**

- **简介: 该论文属人体动作理解任务，旨在探究物理力（如关节驱动力）能否提升模型性能。作者将力特征融入现有模型，在步态识别、动作识别和视频描述三个任务的8个基准上实证验证，发现力信息在遮挡、视角变化等挑战场景下显著提升准确率与语义质量。**

- **链接: [https://arxiv.org/pdf/2512.20451v1](https://arxiv.org/pdf/2512.20451v1)**

> **作者:** Anh Dao; Manh Tran; Yufei Zhang; Xiaoming Liu; Zijun Cui
>
> **摘要:** Human motion understanding has advanced rapidly through vision-based progress in recognition, tracking, and captioning. However, most existing methods overlook physical cues such as joint actuation forces that are fundamental in biomechanics. This gap motivates our study: if and when do physically inferred forces enhance motion understanding? By incorporating forces into established motion understanding pipelines, we systematically evaluate their impact across baseline models on 3 major tasks: gait recognition, action recognition, and fine-grained video captioning. Across 8 benchmarks, incorporating forces yields consistent performance gains; for example, on CASIA-B, Rank-1 gait recognition accuracy improved from 89.52% to 90.39% (+0.87), with larger gain observed under challenging conditions: +2.7% when wearing a coat and +3.0% at the side view. On Gait3D, performance also increases from 46.0% to 47.3% (+1.3). In action recognition, CTR-GCN achieved +2.00% on Penn Action, while high-exertion classes like punching/slapping improved by +6.96%. Even in video captioning, Qwen2.5-VL's ROUGE-L score rose from 0.310 to 0.339 (+0.029), indicating that physics-inferred forces enhance temporal grounding and semantic richness. These results demonstrate that force cues can substantially complement visual and kinematic features under dynamic, occluded, or appearance-varying conditions.
>
---
#### [new 070] TAVID: Text-Driven Audio-Visual Interactive Dialogue Generation
- **分类: cs.CV; cs.AI; eess.AS; eess.IV**

- **简介: 该论文提出TAVID框架，解决文本驱动的音视频协同对话生成任务，旨在同步生成逼真互动人脸与自然对话语音。通过双向跨模态映射器（运动映射器与说话人映射器）融合视听信息，统一建模对话中的视听交互，提升交互真实性与流畅性。**

- **链接: [https://arxiv.org/pdf/2512.20296v1](https://arxiv.org/pdf/2512.20296v1)**

> **作者:** Ji-Hoon Kim; Junseok Ahn; Doyeop Kwak; Joon Son Chung; Shinji Watanabe
>
> **备注:** Project page: https://mm.kaist.ac.kr/projects/TAVID
>
> **摘要:** The objective of this paper is to jointly synthesize interactive videos and conversational speech from text and reference images. With the ultimate goal of building human-like conversational systems, recent studies have explored talking or listening head generation as well as conversational speech generation. However, these works are typically studied in isolation, overlooking the multimodal nature of human conversation, which involves tightly coupled audio-visual interactions. In this paper, we introduce TAVID, a unified framework that generates both interactive faces and conversational speech in a synchronized manner. TAVID integrates face and speech generation pipelines through two cross-modal mappers (i.e., a motion mapper and a speaker mapper), which enable bidirectional exchange of complementary information between the audio and visual modalities. We evaluate our system across four dimensions: talking face realism, listening head responsiveness, dyadic interaction fluency, and speech quality. Extensive experiments demonstrate the effectiveness of our approach across all these aspects.
>
---
#### [new 071] Simplifying Multi-Task Architectures Through Task-Specific Normalization
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属多任务学习（MTL）领域，旨在解决MTL中任务干扰与资源分配失衡问题。作者发现任务特定归一化已足够有效，进而提出轻量级TSσBN方法，通过可学习门控实现软容量分配，在保持共享主干的同时提升性能、稳定性和可解释性。**

- **链接: [https://arxiv.org/pdf/2512.20420v1](https://arxiv.org/pdf/2512.20420v1)**

> **作者:** Mihai Suteu; Ovidiu Serban
>
> **摘要:** Multi-task learning (MTL) aims to leverage shared knowledge across tasks to improve generalization and parameter efficiency, yet balancing resources and mitigating interference remain open challenges. Architectural solutions often introduce elaborate task-specific modules or routing schemes, increasing complexity and overhead. In this work, we show that normalization layers alone are sufficient to address many of these challenges. Simply replacing shared normalization with task-specific variants already yields competitive performance, questioning the need for complex designs. Building on this insight, we propose Task-Specific Sigmoid Batch Normalization (TS$σ$BN), a lightweight mechanism that enables tasks to softly allocate network capacity while fully sharing feature extractors. TS$σ$BN improves stability across CNNs and Transformers, matching or exceeding performance on NYUv2, Cityscapes, CelebA, and PascalContext, while remaining highly parameter-efficient. Moreover, its learned gates provide a natural framework for analyzing MTL dynamics, offering interpretable insights into capacity allocation, filter specialization, and task relationships. Our findings suggest that complex MTL architectures may be unnecessary and that task-specific normalization offers a simple, interpretable, and efficient alternative.
>
---
#### [new 072] Cube Bench: A Benchmark for Spatial Visual Reasoning in MLLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Cube Bench基准，用于评估多模态大语言模型（MLLMs）的空间与序列推理能力。针对Rubik's Cube任务，分解为五项技能，统一评测框架下对比七种模型，揭示其在复杂度提升时性能骤降及开源/闭源差距，并验证自校正的有限增益。**

- **链接: [https://arxiv.org/pdf/2512.20595v1](https://arxiv.org/pdf/2512.20595v1)**

> **作者:** Dhruv Anand; Ehsan Shareghi
>
> **备注:** 27 pages, 5 figures, 9 tables. Cube available at https://github.com/dana-23/cube-bench
>
> **摘要:** We introduce Cube Bench, a Rubik's-cube benchmark for evaluating spatial and sequential reasoning in multimodal large language models (MLLMs). The benchmark decomposes performance into five skills: (i) reconstructing cube faces from images and text, (ii) choosing the optimal next move, (iii) predicting the outcome of a candidate move without applying it, (iv) executing multi-step plans while recovering from mistakes, and (v) detecting and revising one's own errors. Using a shared set of scrambled cube states, identical prompts and parsers, and a single distance-to-solved metric, we compare recent MLLMs side by side as a function of scramble depth. Across seven MLLMs, accuracy drops sharply with depth; once a trajectory stalls or diverges, models rarely recover, and high face-reconstruction accuracy does not guarantee competent action selection or multi-step execution. A pronounced closed- vs open-source gap emerges: the strongest closed model leads on both single-step perception tasks and multi-step control tasks, while open-weight models cluster near chance on the hardest settings; yet even the best MLLM degrades at higher cube complexity. A simple self-correction via reflective thinking yields modest gains but can also introduce overthinking. Cube Bench offers a compact, reproducible probe of sequential spatial reasoning in MLLMs.
>
---
#### [new 073] How I Met Your Bias: Investigating Bias Amplification in Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究扩散模型中的偏见放大问题，属公平性与生成模型交叉任务。它首次揭示采样算法及超参数会显著影响偏见放大程度，而非仅由模型本身决定；通过多数据集实验，证明固定模型下调整采样参数可实现偏见抑制或加剧。**

- **链接: [https://arxiv.org/pdf/2512.20233v1](https://arxiv.org/pdf/2512.20233v1)**

> **作者:** Nathan Roos; Ekaterina Iakovleva; Ani Gjergji; Vito Paolo Pastore; Enzo Tartaglione
>
> **摘要:** Diffusion-based generative models demonstrate state-of-the-art performance across various image synthesis tasks, yet their tendency to replicate and amplify dataset biases remains poorly understood. Although previous research has viewed bias amplification as an inherent characteristic of diffusion models, this work provides the first analysis of how sampling algorithms and their hyperparameters influence bias amplification. We empirically demonstrate that samplers for diffusion models -- commonly optimized for sample quality and speed -- have a significant and measurable effect on bias amplification. Through controlled studies with models trained on Biased MNIST, Multi-Color MNIST and BFFHQ, and with Stable Diffusion, we show that sampling hyperparameters can induce both bias reduction and amplification, even when the trained model is fixed. Source code is available at https://github.com/How-I-met-your-bias/how_i_met_your_bias.
>
---
#### [new 074] LongVideoAgent: Multi-Agent Reasoning with Long Videos
- **分类: cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文面向长视频问答任务，旨在解决现有方法因内容压缩或工具受限导致的时序定位不准、细节丢失问题。提出多智能体框架：主LLM协调定位（grounding）与视觉理解（vision）代理，结合强化学习优化协作效率与准确性，在自建LongTVQA数据集上显著优于基线。**

- **链接: [https://arxiv.org/pdf/2512.20618v1](https://arxiv.org/pdf/2512.20618v1)**

> **作者:** Runtao Liu; Ziyi Liu; Jiaqi Tang; Yue Ma; Renjie Pi; Jipeng Zhang; Qifeng Chen
>
> **摘要:** Recent advances in multimodal LLMs and systems that use tools for long-video QA point to the promise of reasoning over hour-long episodes. However, many methods still compress content into lossy summaries or rely on limited toolsets, weakening temporal grounding and missing fine-grained cues. We propose a multi-agent framework in which a master LLM coordinates a grounding agent to localize question-relevant segments and a vision agent to extract targeted textual observations. The master agent plans with a step limit, and is trained with reinforcement learning to encourage concise, correct, and efficient multi-agent cooperation. This design helps the master agent focus on relevant clips via grounding, complements subtitles with visual detail, and yields interpretable trajectories. On our proposed LongTVQA and LongTVQA+ which are episode-level datasets aggregated from TVQA/TVQA+, our multi-agent system significantly outperforms strong non-agent baselines. Experiments also show reinforcement learning further strengthens reasoning and planning for the trained agent. Code and data will be shared at https://longvideoagent.github.io/.
>
---
#### [new 075] Towards Generative Location Awareness for Disaster Response: A Probabilistic Cross-view Geolocalization Approach
- **分类: cs.AI; cs.CV**

- **简介: 该论文面向灾害响应中的定位难题，提出ProbGLC方法，实现跨视角（如卫星-地面）概率化地理定位。它融合概率与确定性模型，提升定位精度（Acc@1km达0.86）与可解释性（提供不确定性分布和可定位性评分），并在多灾种数据集上验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.20056v1](https://arxiv.org/pdf/2512.20056v1)**

> **作者:** Hao Li; Fabian Deuser; Wenping Yin; Steffen Knoblauch; Wufan Zhao; Filip Biljecki; Yong Xue; Wei Huang
>
> **摘要:** As Earth's climate changes, it is impacting disasters and extreme weather events across the planet. Record-breaking heat waves, drenching rainfalls, extreme wildfires, and widespread flooding during hurricanes are all becoming more frequent and more intense. Rapid and efficient response to disaster events is essential for climate resilience and sustainability. A key challenge in disaster response is to accurately and quickly identify disaster locations to support decision-making and resources allocation. In this paper, we propose a Probabilistic Cross-view Geolocalization approach, called ProbGLC, exploring new pathways towards generative location awareness for rapid disaster response. Herein, we combine probabilistic and deterministic geolocalization models into a unified framework to simultaneously enhance model explainability (via uncertainty quantification) and achieve state-of-the-art geolocalization performance. Designed for rapid diaster response, the ProbGLC is able to address cross-view geolocalization across multiple disaster events as well as to offer unique features of probabilistic distribution and localizability score. To evaluate the ProbGLC, we conduct extensive experiments on two cross-view disaster datasets (i.e., MultiIAN and SAGAINDisaster), consisting diverse cross-view imagery pairs of multiple disaster types (e.g., hurricanes, wildfires, floods, to tornadoes). Preliminary results confirms the superior geolocalization accuracy (i.e., 0.86 in Acc@1km and 0.97 in Acc@25km) and model explainability (i.e., via probabilistic distributions and localizability scores) of the proposed ProbGLC approach, highlighting the great potential of leveraging generative cross-view approach to facilitate location awareness for better and faster disaster response. The data and code is publicly available at https://github.com/bobleegogogo/ProbGLC
>
---
#### [new 076] Exploring Deep-to-Shallow Transformable Neural Networks for Intelligent Embedded Systems
- **分类: cs.LG; cs.CV**

- **简介: 该论文属神经架构搜索（NAS）任务，旨在解决嵌入式系统中深度CNN精度高但硬件效率低、浅层网络效率高但精度差的矛盾。提出Double-Win NAS框架，自动搜索高性能深网并等价转换为高效浅网，并引入混合可变训练与任意分辨率弹性训练技术。**

- **链接: [https://arxiv.org/pdf/2512.19731v1](https://arxiv.org/pdf/2512.19731v1)**

> **作者:** Xiangzhong Luo; Weichen Liu
>
> **备注:** Accepted by IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems
>
> **摘要:** Thanks to the evolving network depth, convolutional neural networks (CNNs) have achieved remarkable success across various embedded scenarios, paving the way for ubiquitous embedded intelligence. Despite its promise, the evolving network depth comes at the cost of degraded hardware efficiency. In contrast to deep networks, shallow networks can deliver superior hardware efficiency but often suffer from inferior accuracy. To address this dilemma, we propose Double-Win NAS, a novel deep-to-shallow transformable neural architecture search (NAS) paradigm tailored for resource-constrained intelligent embedded systems. Specifically, Double-Win NAS strives to automatically explore deep networks to first win strong accuracy, which are then equivalently transformed into their shallow counterparts to further win strong hardware efficiency. In addition to search, we also propose two enhanced training techniques, including hybrid transformable training towards better training accuracy and arbitrary-resolution elastic training towards enabling natural network elasticity across arbitrary input resolutions. Extensive experimental results on two popular intelligent embedded systems (i.e., NVIDIA Jetson AGX Xavier and NVIDIA Jetson Nano) and two representative large-scale datasets (i.e., ImageNet and ImageNet-100) clearly demonstrate the superiority of Double-Win NAS over previous state-of-the-art NAS approaches.
>
---
#### [new 077] Snapshot 3D image projection using a diffractive decoder
- **分类: physics.optics; cs.CV; cs.NE; physics.app-ph**

- **简介: 该论文提出一种基于衍射解码器的快照式3D投影方法，旨在解决密集轴向层间衍射串扰导致的深度复用难题。通过多层衍射波前解码与端到端深度学习优化，实现单次曝光下高保真、亚波长级轴向分辨的多平面3D图像投影。**

- **链接: [https://arxiv.org/pdf/2512.20464v1](https://arxiv.org/pdf/2512.20464v1)**

> **作者:** Cagatay Isil; Alexander Chen; Yuhang Li; F. Onuralp Ardic; Shiqi Chen; Che-Yung Shen; Aydogan Ozcan
>
> **备注:** 22 Pages, 8 Figures
>
> **摘要:** 3D image display is essential for next-generation volumetric imaging; however, dense depth multiplexing for 3D image projection remains challenging because diffraction-induced cross-talk rapidly increases as the axial image planes get closer. Here, we introduce a 3D display system comprising a digital encoder and a diffractive optical decoder, which simultaneously projects different images onto multiple target axial planes with high axial resolution. By leveraging multi-layer diffractive wavefront decoding and deep learning-based end-to-end optimization, the system achieves high-fidelity depth-resolved 3D image projection in a snapshot, enabling axial plane separations on the order of a wavelength. The digital encoder leverages a Fourier encoder network to capture multi-scale spatial and frequency-domain features from input images, integrates axial position encoding, and generates a unified phase representation that simultaneously encodes all images to be axially projected in a single snapshot through a jointly-optimized diffractive decoder. We characterized the impact of diffractive decoder depth, output diffraction efficiency, spatial light modulator resolution, and axial encoding density, revealing trade-offs that govern axial separation and 3D image projection quality. We further demonstrated the capability to display volumetric images containing 28 axial slices, as well as the ability to dynamically reconfigure the axial locations of the image planes, performed on demand. Finally, we experimentally validated the presented approach, demonstrating close agreement between the measured results and the target images. These results establish the diffractive 3D display system as a compact and scalable framework for depth-resolved snapshot 3D image projection, with potential applications in holographic displays, AR/VR interfaces, and volumetric optical computing.
>
---
#### [new 078] Unified Multimodal Brain Decoding via Cross-Subject Soft-ROI Fusion
- **分类: cs.LG; cs.CV; eess.IV**

- **简介: 该论文属跨被试多模态脑解码任务，旨在从fMRI信号重建视觉刺激对应的自然语言描述。针对跨被试泛化差与提示不可解释两大挑战，提出BrainROI模型：设计软ROI融合fMRI编码器提升泛化性，并引入可审计的闭环提示优化机制增强稳定性与透明度。**

- **链接: [https://arxiv.org/pdf/2512.20249v1](https://arxiv.org/pdf/2512.20249v1)**

> **作者:** Xuanyu Hu
>
> **备注:** 15 pages, 2 figures, 4 tables. Submitted to ICPR 2026
>
> **摘要:** Multimodal brain decoding aims to reconstruct semantic information that is consistent with visual stimuli from brain activity signals such as fMRI, and then generate readable natural language descriptions. However, multimodal brain decoding still faces key challenges in cross-subject generalization and interpretability. We propose a BrainROI model and achieve leading-level results in brain-captioning evaluation on the NSD dataset. Under the cross-subject setting, compared with recent state-of-the-art methods and representative baselines, metrics such as BLEU-4 and CIDEr show clear improvements. Firstly, to address the heterogeneity of functional brain topology across subjects, we design a new fMRI encoder. We use multi-atlas soft functional parcellations (soft-ROI) as a shared space. We extend the discrete ROI Concatenation strategy in MINDLLM to a voxel-wise gated fusion mechanism (Voxel-gate). We also ensure consistent ROI mapping through global label alignment, which enhances cross-subject transferability. Secondly, to overcome the limitations of manual and black-box prompting methods in stability and transparency, we introduce an interpretable prompt optimization process. In a small-sample closed loop, we use a locally deployed Qwen model to iteratively generate and select human-readable prompts. This process improves the stability of prompt design and preserves an auditable optimization trajectory. Finally, we impose parameterized decoding constraints during inference to further improve the stability and quality of the generated descriptions.
>
---
#### [new 079] Dreamcrafter: Immersive Editing of 3D Radiance Fields Through Flexible, Generative Inputs and Outputs
- **分类: cs.HC; cs.CV**

- **简介: 该论文提出Dreamcrafter系统，属3D场景编辑任务，旨在解决现有AI生成式编辑高延迟与沉浸式交互难兼顾的问题。工作包括：构建VR驱动的模块化架构，融合自然语言与直接操作控制，并引入代理表示支持高延迟下的实时交互。**

- **链接: [https://arxiv.org/pdf/2512.20129v1](https://arxiv.org/pdf/2512.20129v1)**

> **作者:** Cyrus Vachha; Yixiao Kang; Zach Dive; Ashwat Chidambaram; Anik Gupta; Eunice Jun; Bjoern Hartmann
>
> **备注:** CHI 2025, Project page: https://dream-crafter.github.io/
>
> **摘要:** Authoring 3D scenes is a central task for spatial computing applications. Competing visions for lowering existing barriers are (1) focus on immersive, direct manipulation of 3D content or (2) leverage AI techniques that capture real scenes (3D Radiance Fields such as, NeRFs, 3D Gaussian Splatting) and modify them at a higher level of abstraction, at the cost of high latency. We unify the complementary strengths of these approaches and investigate how to integrate generative AI advances into real-time, immersive 3D Radiance Field editing. We introduce Dreamcrafter, a VR-based 3D scene editing system that: (1) provides a modular architecture to integrate generative AI algorithms; (2) combines different levels of control for creating objects, including natural language and direct manipulation; and (3) introduces proxy representations that support interaction during high-latency operations. We contribute empirical findings on control preferences and discuss how generative AI interfaces beyond text input enhance creativity in scene editing and world building.
>
---
#### [new 080] Dual-Encoder Transformer-Based Multimodal Learning for Ischemic Stroke Lesion Segmentation Using Diffusion MRI
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文面向医学图像分割任务，旨在解决缺血性脑卒中病灶在扩散MRI（DWI/ADC）中的自动精准分割难题。作者基于ISLES 2022数据集，对比多种模型，提出双编码器TransUNet架构，融合双模态与相邻切片信息，最终以85.4% Dice得分实现最优性能。**

- **链接: [https://arxiv.org/pdf/2512.20436v1](https://arxiv.org/pdf/2512.20436v1)**

> **作者:** Muhammad Usman; Azka Rehman; Muhammad Mutti Ur Rehman; Abd Ur Rehman; Muhammad Umar Farooq
>
> **摘要:** Accurate segmentation of ischemic stroke lesions from diffusion magnetic resonance imaging (MRI) is essential for clinical decision-making and outcome assessment. Diffusion-Weighted Imaging (DWI) and Apparent Diffusion Coefficient (ADC) scans provide complementary information on acute and sub-acute ischemic changes; however, automated lesion delineation remains challenging due to variability in lesion appearance. In this work, we study ischemic stroke lesion segmentation using multimodal diffusion MRI from the ISLES 2022 dataset. Several state-of-the-art convolutional and transformer-based architectures, including U-Net variants, Swin-UNet, and TransUNet, are benchmarked. Based on performance, a dual-encoder TransUNet architecture is proposed to learn modality-specific representations from DWI and ADC inputs. To incorporate spatial context, adjacent slice information is integrated using a three-slice input configuration. All models are trained under a unified framework and evaluated using the Dice Similarity Coefficient (DSC). Results show that transformer-based models outperform convolutional baselines, and the proposed dual-encoder TransUNet achieves the best performance, reaching a Dice score of 85.4% on the test set. The proposed framework offers a robust solution for automated ischemic stroke lesion segmentation from diffusion MRI.
>
---
#### [new 081] CLIP Based Region-Aware Feature Fusion for Automated BBPS Scoring in Colonoscopy Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文属医学图像分析任务，旨在解决BBPS评分主观性强、一致性差的问题。作者构建了2240张结肠镜图像新数据集，提出基于CLIP的区域感知特征融合方法，结合视觉特征与粪便相关文本先验，实现无需分割的自动BBPS评分。**

- **链接: [https://arxiv.org/pdf/2512.20374v1](https://arxiv.org/pdf/2512.20374v1)**

> **作者:** Yujia Fu; Zhiyu Dong; Tianwen Qian; Chenye Zheng; Danian Ji; Linhai Zhuo
>
> **备注:** 12 pages, 9 figures, BMVC 2025 submission
>
> **摘要:** Accurate assessment of bowel cleanliness is essential for effective colonoscopy procedures. The Boston Bowel Preparation Scale (BBPS) offers a standardized scoring system but suffers from subjectivity and inter-observer variability when performed manually. In this paper, to support robust training and evaluation, we construct a high-quality colonoscopy dataset comprising 2,240 images from 517 subjects, annotated with expert-agreed BBPS scores. We propose a novel automated BBPS scoring framework that leverages the CLIP model with adapter-based transfer learning and a dedicated fecal-feature extraction branch. Our method fuses global visual features with stool-related textual priors to improve the accuracy of bowel cleanliness evaluation without requiring explicit segmentation. Extensive experiments on both our dataset and the public NERTHU dataset demonstrate the superiority of our approach over existing baselines, highlighting its potential for clinical deployment in computer-aided colonoscopy analysis.
>
---
#### [new 082] Generative Digital Twins: Vision-Language Simulation Models for Executable Industrial Systems
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出视觉-语言仿真模型（VLSM），解决工业数字孪生中从草图和自然语言生成可执行FlexScript代码的问题。构建首个12万+三元组数据集，设计SVR、PMR、ESR三项新指标，并通过模型消融验证其高结构准确率与执行鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.20387v1](https://arxiv.org/pdf/2512.20387v1)**

> **作者:** YuChe Hsu; AnJui Wang; TsaiChing Ni; YuanFu Yang
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** We propose a Vision-Language Simulation Model (VLSM) that unifies visual and textual understanding to synthesize executable FlexScript from layout sketches and natural-language prompts, enabling cross-modal reasoning for industrial simulation systems. To support this new paradigm, the study constructs the first large-scale dataset for generative digital twins, comprising over 120,000 prompt-sketch-code triplets that enable multimodal learning between textual descriptions, spatial structures, and simulation logic. In parallel, three novel evaluation metrics, Structural Validity Rate (SVR), Parameter Match Rate (PMR), and Execution Success Rate (ESR), are proposed specifically for this task to comprehensively evaluate structural integrity, parameter fidelity, and simulator executability. Through systematic ablation across vision encoders, connectors, and code-pretrained language backbones, the proposed models achieve near-perfect structural accuracy and high execution robustness. This work establishes a foundation for generative digital twins that integrate visual reasoning and language understanding into executable industrial simulation systems.
>
---
#### [new 083] Retrieval-augmented Prompt Learning for Pre-trained Foundation Models
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **简介: 该论文提出RetroPrompt，属提示学习任务，旨在解决预训练基础模型在少样本下过拟合、依赖死记硬背的问题。其通过引入基于训练数据构建的公开知识库与全程检索机制，解耦知识与记忆，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.20145v1](https://arxiv.org/pdf/2512.20145v1)**

> **作者:** Xiang Chen; Yixin Ou; Quan Feng; Lei Li; Piji Li; Haibo Ye; Sheng-Jun Huang; Shuofei Qiao; Shumin Deng; Huajun Chen; Ningyu Zhang
>
> **备注:** IEEE/ACM Transactions on Audio, Speech and Language Processing
>
> **摘要:** The pre-trained foundation models (PFMs) have become essential for facilitating large-scale multimodal learning. Researchers have effectively employed the ``pre-train, prompt, and predict'' paradigm through prompt learning to induce improved few-shot performance. However, prompt learning approaches for PFMs still follow a parametric learning paradigm. As such, the stability of generalization in memorization and rote learning can be compromised. More specifically, conventional prompt learning might face difficulties in fully utilizing atypical instances and avoiding overfitting to shallow patterns with limited data during the process of fully-supervised training. To overcome these constraints, we present our approach, named RetroPrompt, which aims to achieve a balance between memorization and generalization by decoupling knowledge from mere memorization. Unlike traditional prompting methods, RetroPrompt leverages a publicly accessible knowledge base generated from the training data and incorporates a retrieval mechanism throughout the input, training, and inference stages. This enables the model to actively retrieve relevant contextual information from the corpus, thereby enhancing the available cues. We conduct comprehensive experiments on a variety of datasets across natural language processing and computer vision tasks to demonstrate the superior performance of our proposed approach, RetroPrompt, in both zero-shot and few-shot scenarios. Through detailed analysis of memorization patterns, we observe that RetroPrompt effectively reduces the reliance on rote memorization, leading to enhanced generalization.
>
---
#### [new 084] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出KnowVal系统，面向自动驾驶任务，解决现有数据驱动方法难以建模复杂逻辑与价值对齐的问题。工作包括：构建驾驶知识图谱、设计LLM驱动的知识检索机制、建立人类偏好数据集并训练价值模型，以实现知识增强、语言推理与价值引导的协同规划。**

- **链接: [https://arxiv.org/pdf/2512.20299v1](https://arxiv.org/pdf/2512.20299v1)**

> **作者:** Zhongyu Xia; Wenhao Chen; Yongtao Wang; Ming-Hsuan Yang
>
> **摘要:** Visual-language reasoning, driving knowledge, and value alignment are essential for advanced autonomous driving systems. However, existing approaches largely rely on data-driven learning, making it difficult to capture the complex logic underlying decision-making through imitation or limited reinforcement rewards. To address this, we propose KnowVal, a new autonomous driving system that enables visual-language reasoning through the synergistic integration of open-world perception and knowledge retrieval. Specifically, we construct a comprehensive driving knowledge graph that encodes traffic laws, defensive driving principles, and ethical norms, complemented by an efficient LLM-based retrieval mechanism tailored for driving scenarios. Furthermore, we develop a human-preference dataset and train a Value Model to guide interpretable, value-aligned trajectory assessment. Experimental results show that our method substantially improves planning performance while remaining compatible with existing architectures. Notably, KnowVal achieves the lowest collision rate on nuScenes and state-of-the-art results on Bench2Drive.
>
---
#### [new 085] SAM Audio: Segment Anything in Audio
- **分类: eess.AS; cs.CV**

- **简介: 该论文提出SAM Audio，面向通用音频源分离任务，解决现有模型领域受限、提示模态单一的问题。工作包括：构建支持文本/视觉/时间跨度多模态提示的扩散Transformer基础模型，用流匹配在大规模多类型音频上训练，并建立新基准与无参考评估模型。**

- **链接: [https://arxiv.org/pdf/2512.18099v1](https://arxiv.org/pdf/2512.18099v1)**

> **作者:** Bowen Shi; Andros Tjandra; John Hoffman; Helin Wang; Yi-Chiao Wu; Luya Gao; Julius Richter; Matt Le; Apoorv Vyas; Sanyuan Chen; Christoph Feichtenhofer; Piotr Dollár; Wei-Ning Hsu; Ann Lee
>
> **摘要:** General audio source separation is a key capability for multimodal AI systems that can perceive and reason about sound. Despite substantial progress in recent years, existing separation models are either domain-specific, designed for fixed categories such as speech or music, or limited in controllability, supporting only a single prompting modality such as text. In this work, we present SAM Audio, a foundation model for general audio separation that unifies text, visual, and temporal span prompting within a single framework. Built on a diffusion transformer architecture, SAM Audio is trained with flow matching on large-scale audio data spanning speech, music, and general sounds, and can flexibly separate target sources described by language, visual masks, or temporal spans. The model achieves state-of-the-art performance across a diverse suite of benchmarks, including general sound, speech, music, and musical instrument separation in both in-the-wild and professionally produced audios, substantially outperforming prior general-purpose and specialized systems. Furthermore, we introduce a new real-world separation benchmark with human-labeled multimodal prompts and a reference-free evaluation model that correlates strongly with human judgment.
>
---
#### [new 086] Field-Space Attention for Structure-Preserving Earth System Transformers
- **分类: cs.LG; cs.CV; math-ph**

- **简介: 该论文提出Field-Space Attention，用于地球系统建模任务。旨在解决传统Transformer在物理场建模中结构失真、优化不稳定、可解释性差等问题。工作包括：在物理域（球面连续场）计算注意力，固定多尺度分解，学习结构保持形变，提升超分辨率性能与物理一致性。**

- **链接: [https://arxiv.org/pdf/2512.20350v1](https://arxiv.org/pdf/2512.20350v1)**

> **作者:** Maximilian Witte; Johannes Meuer; Étienne Plésiat; Christopher Kadow
>
> **摘要:** Accurate and physically consistent modeling of Earth system dynamics requires machine-learning architectures that operate directly on continuous geophysical fields and preserve their underlying geometric structure. Here we introduce Field-Space attention, a mechanism for Earth system Transformers that computes attention in the physical domain rather than in a learned latent space. By maintaining all intermediate representations as continuous fields on the sphere, the architecture enables interpretable internal states and facilitates the enforcement of scientific constraints. The model employs a fixed, non-learned multiscale decomposition and learns structure-preserving deformations of the input field, allowing coherent integration of coarse and fine-scale information while avoiding the optimization instabilities characteristic of standard single-scale Vision Transformers. Applied to global temperature super-resolution on a HEALPix grid, Field-Space Transformers converge more rapidly and stably than conventional Vision Transformers and U-Net baselines, while requiring substantially fewer parameters. The explicit preservation of field structure throughout the network allows physical and statistical priors to be embedded directly into the architecture, yielding improved fidelity and reliability in data-driven Earth system modeling. These results position Field-Space Attention as a compact, interpretable, and physically grounded building block for next-generation Earth system prediction and generative modeling frameworks.
>
---
## 更新

#### [replaced 001] RemoteReasoner: Towards Unifying Geospatial Reasoning Workflow
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.19280v3](https://arxiv.org/pdf/2507.19280v3)**

> **作者:** Liang Yao; Fan Liu; Hongbo Lu; Chuanyi Zhang; Rui Min; Shengxiang Xu; Shimin Di; Pai Peng
>
> **摘要:** Remote sensing imagery presents vast, inherently unstructured spatial data, necessitating sophisticated reasoning to interpret complex user intents and contextual relationships beyond simple recognition tasks. In this paper, we aim to construct an Earth observation workflow to handle complex queries by reasoning about spatial context and user intent. As a reasoning workflow, it should autonomously explore and construct its own inference paths, rather than being confined to predefined ground-truth sequences. Ideally, its architecture ought to be unified yet generalized, possessing capabilities to perform diverse reasoning tasks through one model without requiring additional fine-tuning. Existing remote sensing approaches rely on supervised fine-tuning paradigms and task-specific heads, limiting both autonomous reasoning and unified generalization. To this end, we propose RemoteReasoner, a unified workflow for geospatial reasoning. The design of RemoteReasoner integrates a multi-modal large language model (MLLM) for interpreting user instructions and localizing targets, together with task transformation strategies that enable multi-granularity tasks, including object-, region-, and pixel-level. In contrast to existing methods, our framework is trained with reinforcement learning (RL) to endow the MLLM sufficient reasoning autonomy. At the inference stage, our transformation strategies enable diverse task output formats without requiring task-specific decoders or further fine-tuning. Experiments demonstrated that RemoteReasoner achieves state-of-the-art (SOTA) performance across multi-granularity reasoning tasks. Furthermore, it retains the MLLM's inherent generalization capability, demonstrating robust performance on unseen tasks and out-of-distribution categories.
>
---
#### [replaced 002] GradMix: Gradient-based Selective Mixup for Robust Data Augmentation in Class-Incremental Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.08528v2](https://arxiv.org/pdf/2505.08528v2)**

> **作者:** Minsu Kim; Seong-Hyeon Hwang; Steven Euijong Whang
>
> **备注:** Accepted to KDD 2026
>
> **摘要:** In the context of continual learning, acquiring new knowledge while maintaining previous knowledge presents a significant challenge. Existing methods often use experience replay techniques that store a small portion of previous task data for training. In experience replay approaches, data augmentation has emerged as a promising strategy to further improve the model performance by mixing limited previous task data with sufficient current task data. However, we theoretically and empirically analyze that training with mixed samples from random sample pairs may harm the knowledge of previous tasks and cause greater catastrophic forgetting. We then propose GradMix, a robust data augmentation method specifically designed for mitigating catastrophic forgetting in class-incremental learning. GradMix performs gradient-based selective mixup using a class-based criterion that mixes only samples from helpful class pairs and not from detrimental class pairs for reducing catastrophic forgetting. Our experiments on various real datasets show that GradMix outperforms data augmentation baselines in accuracy by minimizing the forgetting of previous knowledge.
>
---
#### [replaced 003] Vision Language Models are Confused Tourists
- **分类: cs.CV; cs.CL**

- **简介: 该论文属多模态鲁棒性评估任务，旨在解决VLMs在多元文化混合输入下性能骤降的问题。作者提出ConfusedTourist基准，通过图像堆叠等扰动测试模型对地理文化线索的稳定性，并发现模型因注意力被干扰而失效，揭示其文化鲁棒性缺陷。**

- **链接: [https://arxiv.org/pdf/2511.17004v3](https://arxiv.org/pdf/2511.17004v3)**

> **作者:** Patrick Amadeus Irawan; Ikhlasul Akmal Hanif; Muhammad Dehan Al Kautsar; Genta Indra Winata; Fajri Koto; Alham Fikri Aji
>
> **摘要:** Although the cultural dimension has been one of the key aspects in evaluating Vision-Language Models (VLMs), their ability to remain stable across diverse cultural inputs remains largely untested, despite being crucial to support diversity and multicultural societies. Existing evaluations often rely on benchmarks featuring only a singular cultural concept per image, overlooking scenarios where multiple, potentially unrelated cultural cues coexist. To address this gap, we introduce ConfusedTourist, a novel cultural adversarial robustness suite designed to assess VLMs' stability against perturbed geographical cues. Our experiments reveal a critical vulnerability, where accuracy drops heavily under simple image-stacking perturbations and even worsens with its image-generation-based variant. Interpretability analyses further show that these failures stem from systematic attention shifts toward distracting cues, diverting the model from its intended focus. These findings highlight a critical challenge: visual cultural concept mixing can substantially impair even state-of-the-art VLMs, underscoring the urgent need for more culturally robust multimodal understanding.
>
---
#### [replaced 004] Fine-Grained Instruction-Guided Graph Reasoning for Vision-and-Language Navigation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.11006v2](https://arxiv.org/pdf/2503.11006v2)**

> **作者:** Yaohua Liu; Xinyuan Song; Yunfu Deng; Yifan Xie; Binkai Ou; Yan Zhong
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Vision-and-Language Navigation (VLN) requires an embodied agent to traverse complex environments by following natural language instructions, demanding accurate alignment between visual observations and linguistic guidance. Despite recent progress, existing methods typically encode visual and directional cues in a coupled manner, and process instructions without explicitly extracting navigation-critical semantics, which often leads to imprecise spatial reasoning and suboptimal cross-modal alignment. To address these challenges, we propose a fine-grained instruction-guided graph reasoning framework (OIKG) that enhances both spatial representation and instruction understanding during navigation. Specifically, an observation-graph interaction mechanism is introduced to disentangle angular and visual cues while strengthening directed edge representations through geometric embedding, enabling more reliable spatial reasoning within the navigation graph. In addition, a fine-grained instruction guidance module is designed to explicitly extract and leverage location-specific and object-centric information from language instructions, facilitating more precise cross-modal alignment between linguistic semantics and navigable trajectories. By jointly integrating structured graph reasoning with instruction-critical semantic cues, the proposed approach significantly improves the agent's ability to follow complex navigation instructions. Extensive experiments on the R2R and RxR benchmarks demonstrate that our method consistently achieves state-of-the-art performance across multiple evaluation metrics, validating the effectiveness of fine-grained instruction-guided graph reasoning for vision-and-language navigation.
>
---
#### [replaced 005] Spectral Bottleneck in Sinusoidal Representation Networks: Noise is All You Need
- **分类: eess.AS; cs.CV; cs.LG; cs.SD; eess.IV**

- **简介: 该论文研究隐式神经表示（SIREN）的频谱瓶颈问题：因初始化不当导致高频拟合失败。提出目标感知的WINNER初始化方法，通过调控激活频谱与NTK特性提升音频/图像拟合精度。**

- **链接: [https://arxiv.org/pdf/2509.09719v2](https://arxiv.org/pdf/2509.09719v2)**

> **作者:** Hemanth Chandravamsi; Dhanush V. Shenoy; Itay Zinn; Ziv Chen; Shimon Pisnoy; Steven H. Frankel
>
> **摘要:** This work identifies and attempts to address a fundamental limitation of implicit neural representations with sinusoidal activation. The fitting error of SIRENs is highly sensitive to the target frequency content and to the choice of initialization. In extreme cases, this sensitivity leads to a spectral bottleneck that can result in a zero-valued output. This phenomenon is characterized by analyzing the evolution of activation spectra and the empirical neural tangent kernel (NTK) during the training process. An unfavorable distribution of energy across frequency modes was noted to give rise to this failure mode. Furthermore, the effect of Gaussian perturbations applied to the baseline uniformly initialized weights is examined, showing how these perturbations influence activation spectra and the NTK eigenbasis of SIREN. Overall, initialization emerges as a central factor governing the evolution of SIRENs, indicating the need for adaptive, target-aware strategies as the target length increases and fine-scale detail becomes essential. The proposed weight initialization scheme (WINNER) represents a simple ad hoc step in this direction and demonstrates that fitting accuracy can be significantly improved by modifying the spectral profile of network activations through a target-aware initialization. The approach achieves state-of-the-art performance on audio fitting tasks and yields notable improvements in image fitting tasks.
>
---
#### [replaced 006] SPECIAL: Zero-shot Hyperspectral Image Classification With CLIP
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.16222v3](https://arxiv.org/pdf/2501.16222v3)**

> **作者:** Li Pang; Jing Yao; Kaiyu Li; Jun Zhou; Deyu Meng; Xiangyong Cao
>
> **摘要:** Hyperspectral image (HSI) classification aims to categorize each pixel in an HSI into a specific land cover class, which is crucial for applications such as remote sensing, environmental monitoring, and agriculture. Although deep learning-based HSI classification methods have achieved significant advancements, existing methods still rely on manually labeled data for training, which is both time-consuming and labor-intensive. To address this limitation, we introduce a novel zero-shot hyperspectral image classification framework based on CLIP (SPECIAL), aiming to eliminate the need for manual annotations. The SPECIAL framework consists of two main stages: (1) CLIP-based pseudo-label generation, and (2) noisy label learning. In the first stage, HSI is spectrally interpolated to produce RGB bands. These bands are subsequently classified using CLIP, resulting in noisy pseudo-labels that are accompanied by confidence scores. To improve the quality of these labels, we propose a scaling strategy that fuses predictions from multiple spatial scales. In the second stage, spectral information and a label refinement technique are incorporated to mitigate label noise and further enhance classification accuracy. Experimental results on three benchmark datasets demonstrate that our SPECIAL outperforms existing methods in zero-shot HSI classification, showing its potential for more practical applications. The code is available at https://github.com/LiPang/SPECIAL.
>
---
#### [replaced 007] LiteGE: Lightweight Geodesic Embedding for Efficient Geodesics Computation and Non-Isometric Shape Correspondence
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2512.17781v2](https://arxiv.org/pdf/2512.17781v2)**

> **作者:** Yohanes Yudhi Adikusuma; Qixing Huang; Ying He
>
> **摘要:** Computing geodesic distances on 3D surfaces is fundamental to many tasks in 3D vision and geometry processing, with deep connections to tasks such as shape correspondence. Recent learning-based methods achieve strong performance but rely on large 3D backbones, leading to high memory usage and latency, which limit their use in interactive or resource-constrained settings. We introduce LiteGE, a lightweight approach that constructs compact, category-aware shape descriptors by applying Principal Component Analysis (PCA) to unsigned distance field (UDFs) samples at informative voxels. This descriptor is efficient to compute and removes the need for high-capacity networks. LiteGE remains robust on sparse point clouds, supporting inputs with as few as 300 points, where prior methods fail. Extensive experiments show that LiteGE reduces memory usage and inference time by up to 300$\times$ compared to existing neural approaches. In addition, by exploiting the intrinsic relationship between geodesic distance and shape correspondence, LiteGE enables fast and accurate shape matching. Our method achieves up to 1000$\times$ speedup over state-of-the-art mesh-based approaches while maintaining comparable accuracy on non-isometric shape pairs, including evaluations on point-cloud inputs.
>
---
#### [replaced 008] Towards Dataset Copyright Evasion Attack against Personalized Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [https://arxiv.org/pdf/2505.02824v2](https://arxiv.org/pdf/2505.02824v2)**

> **作者:** Kuofeng Gao; Yufei Zhu; Yiming Li; Jiawang Bai; Yong Yang; Zhifeng Li; Shu-Tao Xia
>
> **备注:** Accepted by IEEE Transactions on Information Forensics and Security
>
> **摘要:** Text-to-image (T2I) diffusion models enable high-quality image generation conditioned on textual prompts. However, fine-tuning these pre-trained models for personalization raises concerns about unauthorized dataset usage. To address this issue, dataset ownership verification (DOV) has recently been proposed, which embeds watermarks into fine-tuning datasets via backdoor techniques. These watermarks remain dormant on benign samples but produce owner-specified outputs when triggered. Despite its promise, the robustness of DOV against copyright evasion attacks (CEA) remains unexplored. In this paper, we investigate how adversaries can circumvent these mechanisms, enabling models trained on watermarked datasets to bypass ownership verification. We begin by analyzing the limitations of potential attacks achieved by backdoor removal, including TPD and T2IShield. In practice, TPD suffers from inconsistent effectiveness due to randomness, while T2IShield fails when watermarks are embedded as local image patches. To this end, we introduce CEAT2I, the first CEA specifically targeting DOV in T2I diffusion models. CEAT2I consists of three stages: (1) motivated by the observation that T2I models converge faster on watermarked samples with respect to intermediate features rather than training loss, we reliably detect watermarked samples; (2) we iteratively ablate tokens from the prompts of detected samples and monitor feature shifts to identify trigger tokens; and (3) we apply a closed-form concept erasure method to remove the injected watermarks. Extensive experiments demonstrate that CEAT2I effectively evades state-of-the-art DOV mechanisms while preserving model performance. The code is available at https://github.com/csyufei/CEAT2I.
>
---
#### [replaced 009] Weakly Supervised Ephemeral Gully Detection In Remote Sensing Images Using Vision Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13891v2](https://arxiv.org/pdf/2511.13891v2)**

> **作者:** Seyed Mohamad Ali Tousi; Ramy Farag; John A. Lory; G. N. DeSouza
>
> **摘要:** Among soil erosion problems, Ephemeral Gullies are one of the most concerning phenomena occurring in agricultural fields. Their short temporal cycles increase the difficulty in automatically detecting them using classical computer vision approaches and remote sensing. Also, due to scarcity of and the difficulty in producing accurate labeled data, automatic detection of ephemeral gullies using Machine Learning is limited to zero-shot approaches which are hard to implement. To overcome these challenges, we present the first weakly supervised pipeline for detection of ephemeral gullies. Our method relies on remote sensing and uses Vision Language Models (VLMs) to drastically reduce the labor-intensive task of manual labeling. In order to achieve that, the method exploits: 1) the knowledge embedded in the VLM's pretraining; 2) a teacher-student model where the teacher learns from noisy labels coming from the VLMs, and the student learns by weak supervision using teacher-generate labels and a noise-aware loss function. We also make available the first-of-its-kind dataset for semi-supervised detection of ephemeral gully from remote-sensed images. The dataset consists of a number of locations labeled by a group of soil and plant scientists, as well as a large number of unlabeled locations. The dataset represent more than 18,000 high-resolution remote-sensing images obtained over the course of 13 years. Our experimental results demonstrate the validity of our approach by showing superior performances compared to VLMs and the label model itself when using weak supervision to train an student model. The code and dataset for this work are made publicly available.
>
---
#### [replaced 010] Multiscale Corrections by Continuous Super-Resolution
- **分类: math.NA; cs.CE; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2411.07576v2](https://arxiv.org/pdf/2411.07576v2)**

> **作者:** Zhi-Song Liu; Roland Maier; Andreas Rupp
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** Finite element methods typically require a high resolution to satisfactorily approximate micro and even macro patterns of an underlying physical model. This issue can be circumvented by appropriate multiscale strategies that are able to obtain reasonable approximations on under-resolved scales. In this paper, we study the implicit neural representation and propose a continuous super-resolution network as a correction strategy for multiscale effects. It can take coarse finite element data to learn both in-distribution and out-of-distribution high-resolution finite element predictions. Our highlight is the design of a local implicit transformer, which is able to learn multiscale features. We also propose Gabor wavelet-based coordinate encodings, which can overcome the bias of neural networks learning low-frequency features. Finally, perception is often preferred over distortion, so scientists can recognize the visual pattern for further investigation. However, implicit neural representation is known for its lack of local pattern supervision. We propose to use stochastic cosine similarities to compare the local feature differences between prediction and ground truth. It shows better performance on structural alignments. Our experiments show that our proposed strategy achieves superior performance as an in-distribution and out-of-distribution super-resolution strategy.
>
---
#### [replaced 011] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2512.10652v2](https://arxiv.org/pdf/2512.10652v2)**

> **作者:** Jian-Yu Jiang-Lin; Kang-Yang Huang; Ling Zou; Ling Lo; Sheng-Ping Yang; Yu-Wen Tseng; Kun-Hsiang Lin; Chia-Ling Chen; Yu-Ting Ta; Yan-Tsung Wang; Po-Ching Chen; Hongxia Xie; Hong-Han Shuai; Wen-Huang Cheng
>
> **摘要:** Advances in generative modeling have made it increasingly easy to fabricate realistic portrayals of individuals, creating serious risks for security, communication, and public trust. Detecting such person-driven manipulations requires systems that not only distinguish altered content from authentic media but also provide clear and reliable reasoning. In this paper, we introduce TriDF, a comprehensive benchmark for interpretable DeepFake detection. TriDF contains high-quality forgeries from advanced synthesis models, covering 16 DeepFake types across image, video, and audio modalities. The benchmark evaluates three key aspects: Perception, which measures the ability of a model to identify fine-grained manipulation artifacts using human-annotated evidence; Detection, which assesses classification performance across diverse forgery families and generators; and Hallucination, which quantifies the reliability of model-generated explanations. Experiments on state-of-the-art multimodal large language models show that accurate perception is essential for reliable detection, but hallucination can severely disrupt decision-making, revealing the interdependence of these three aspects. TriDF provides a unified framework for understanding the interaction between detection accuracy, evidence identification, and explanation reliability, offering a foundation for building trustworthy systems that address real-world synthetic media threats.
>
---
#### [replaced 012] Portable Biomechanics Laboratory: Clinically Accessible Movement Analysis from a Handheld Smartphone
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.08268v2](https://arxiv.org/pdf/2507.08268v2)**

> **作者:** J. D. Peiffer; Kunal Shah; Irina Djuraskovic; Shawana Anarwala; Kayan Abdou; Rujvee Patel; Prakash Jayabalan; Brenton Pennicooke; R. James Cotton
>
> **备注:** 24 pages, 10 figures
>
> **摘要:** Movement directly reflects neurological and musculoskeletal health, yet objective biomechanical assessment is rarely available in routine care. We introduce Portable Biomechanics Laboratory (PBL), a secure platform for fitting biomechanical models to video collected with a handheld, moving, smartphone. We validate this approach on over 15 hours of data synchronized to ground truth motion capture, finding mean joint-angle errors < 3$°$ and pelvis-translation errors of a few centimeters across patients with neurological-injury, lower-limb prosthesis users, pediatric in-patients, and controls. In > 5 hours of prospective deployments to neurosurgery and sports-medicine clinics, PBL was easy to setup, yielded highly reliable gait metrics (ICC > 0.9), and detected clinically relevant differences. For cervical-myelopathy patients, its measurement of gait quality correlated with modified Japanese Orthopedic Association (mJOA) scores and were responsive to clinical intervention. Handheld smartphone video can therefore deliver accurate, scalable, and low-burden biomechanical measurement, enabling greatly increased monitoring of movement impairments. We release the first clinically-validated method for measuring whole-body kinematics from handheld smartphone video at https://IntelligentSensingAndRehabilitation.github.io/MonocularBiomechanics/.
>
---
#### [replaced 013] VibrantLeaves: A principled parametric image generator for training deep restoration models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.10201v2](https://arxiv.org/pdf/2504.10201v2)**

> **作者:** Raphael Achddou; Yann Gousseau; Saïd Ladjal; Sabine Süsstrunk
>
> **摘要:** Even though Deep Neural Networks are extremely powerful for image restoration tasks, they have several limitations. They are poorly understood and suffer from strong biases inherited from the training sets. One way to address these shortcomings is to have a better control over the training sets, in particular by using synthetic sets. In this paper, we propose a synthetic image generator relying on a few simple principles. In particular, we focus on geometric modeling, textures, and a simple modeling of image acquisition. These properties, integrated in a classical Dead Leaves model, enable the creation of efficient training sets. Standard image denoising and super-resolution networks can be trained on such datasets, reaching performance almost on par with training on natural image datasets. As a first step towards explainability, we provide a careful analysis of the considered principles, identifying which image properties are necessary to obtain good performances. Besides, such training also yields better robustness to various geometric and radiometric perturbations of the test sets.
>
---
#### [replaced 014] On Structured State-Space Duality
- **分类: cs.LG; cs.CL; cs.CV; stat.ML**

- **简介: 该论文研究结构化状态空间模型（SSM）与注意力机制的等价性问题，属模型理论分析任务。它将SSD从标量恒等矩阵推广至一般对角SSM，给出等价于1-半可分掩码注意力的充要条件，并证明其不适用于标准Softmax注意力。**

- **链接: [https://arxiv.org/pdf/2510.04944v2](https://arxiv.org/pdf/2510.04944v2)**

> **作者:** Jerry Yao-Chieh Hu; Xiwen Zhang; Ali ElSheikh; Weimin Wu; Han Liu
>
> **备注:** v2 fixed typos and added numerical results (Appendix B)
>
> **摘要:** Structured State-Space Duality (SSD) [Dao & Gu, ICML 2024] is an equivalence between a simple Structured State-Space Model (SSM) and a masked attention mechanism. In particular, a state-space model with a scalar-times-identity state matrix is equivalent to a masked self-attention with a $1$-semiseparable causal mask. Consequently, the same sequence transformation (model) has two algorithmic realizations: as a linear-time $O(T)$ recurrence or as a quadratic-time $O(T^2)$ attention. In this note, we formalize and generalize this duality: (i) we extend SSD from the scalar-identity case to general diagonal SSMs (diagonal state matrices); (ii) we show that these diagonal SSMs match the scalar case's training complexity lower bounds while supporting richer dynamics; (iii) we establish a necessary and sufficient condition under which an SSM is equivalent to $1$-semiseparable masked attention; and (iv) we show that such duality fails to extend to standard softmax attention due to rank explosion. Together, these results tighten bridge between recurrent SSMs and Transformers, and widen the design space for expressive yet efficient sequence models.
>
---
#### [replaced 015] Machine Unlearning in the Era of Quantum Machine Learning: An Empirical Study
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19253v2](https://arxiv.org/pdf/2512.19253v2)**

> **作者:** Carla Crivoi; Radu Tudor Ionescu
>
> **摘要:** We present the first comprehensive empirical study of machine unlearning (MU) in hybrid quantum-classical neural networks. While MU has been extensively explored in classical deep learning, its behavior within variational quantum circuits (VQCs) and quantum-augmented architectures remains largely unexplored. First, we adapt a broad suite of unlearning methods to quantum settings, including gradient-based, distillation-based, regularization-based and certified techniques. Second, we introduce two new unlearning strategies tailored to hybrid models. Experiments across Iris, MNIST, and Fashion-MNIST, under both subset removal and full-class deletion, reveal that quantum models can support effective unlearning, but outcomes depend strongly on circuit depth, entanglement structure, and task complexity. Shallow VQCs display high intrinsic stability with minimal memorization, whereas deeper hybrid models exhibit stronger trade-offs between utility, forgetting strength, and alignment with retrain oracle. We find that certain methods, e.g. EU-k, LCA, and Certified Unlearning, consistently provide the best balance across metrics. These findings establish baseline empirical insights into quantum machine unlearning and highlight the need for quantum-aware algorithms and theoretical guarantees, as quantum machine learning systems continue to expand in scale and capability. We publicly release our code at: https://github.com/CrivoiCarla/HQML.
>
---
#### [replaced 016] Drifting Away from Truth: GenAI-Driven News Diversity Challenges LVLM-Based Misinformation Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12711v4](https://arxiv.org/pdf/2508.12711v4)**

> **作者:** Fanxiao Li; Jiaying Wu; Tingchao Fu; Yunyun Dong; Bingbing Song; Wei Zhou
>
> **摘要:** The proliferation of multimodal misinformation poses growing threats to public discourse and societal trust. While Large Vision-Language Models (LVLMs) have enabled recent progress in multimodal misinformation detection (MMD), the rise of generative AI (GenAI) tools introduces a new challenge: GenAI-driven news diversity, characterized by highly varied and complex content. We show that this diversity induces multi-level drift, comprising (1) model-level misperception drift, where stylistic variations disrupt a model's internal reasoning, and (2) evidence-level drift, where expression diversity degrades the quality or relevance of retrieved external evidence. These drifts significantly degrade the robustness of current LVLM-based MMD systems. To systematically study this problem, we introduce DriftBench, a large-scale benchmark comprising 16,000 news instances across six categories of diversification. We design three evaluation tasks: (1) robustness of truth verification under multi-level drift; (2) susceptibility to adversarial evidence contamination generated by GenAI; and (3) analysis of reasoning consistency across diverse inputs. Experiments with six state-of-the-art LVLM-based detectors show substantial performance drops (average F1 -14.8%) and increasingly unstable reasoning traces, with even more severe failures under adversarial evidence injection. Our findings uncover fundamental vulnerabilities in existing MMD systems and suggest an urgent need for more resilient approaches in the GenAI era.
>
---
#### [replaced 017] Categorical Equivariant Deep Learning: Category-Equivariant Neural Networks and Universal Approximation Theorems
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出范畴等变深度学习（CENNs），统一各类等变网络（群、偏序集、图、层等）。旨在拓展等变学习 beyond 群作用，建模几何、上下文与组合对称性。工作包括构建范畴化等变框架、定义线性/非线性层、证明通用逼近定理，并实例化多类结构。**

- **链接: [https://arxiv.org/pdf/2511.18417v2](https://arxiv.org/pdf/2511.18417v2)**

> **作者:** Yoshihiro Maruyama
>
> **摘要:** We develop a theory of category-equivariant neural networks (CENNs) that unifies group/groupoid-equivariant networks, poset/lattice-equivariant networks, graph and sheaf neural networks. Equivariance is formulated as naturality in a topological category with Radon measures. Formulating linear and nonlinear layers in the categorical setup, we prove the equivariant universal approximation theorem in the general setting: the class of finite-depth CENNs is dense in the space of continuous equivariant transformations. We instantiate the framework for groups/groupoids, posets/lattices, graphs and cellular sheaves, deriving universal approximation theorems for them in a systematic manner. Categorical equivariant deep learning thus allows us to expand the horizons of equivariant deep learning beyond group actions, encompassing not only geometric symmetries but also contextual and compositional symmetries.
>
---
#### [replaced 018] TropNNC: Structured Neural Network Compression Using Tropical Geometry
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.03945v3](https://arxiv.org/pdf/2409.03945v3)**

> **作者:** Konstantinos Fotopoulos; Petros Maragos; Panagiotis Misiakos
>
> **备注:** v3: restructured the paper, formalized some heuristic improvements to the algorithm, and added acknowledgments
>
> **摘要:** We present TropNNC, a framework for compressing neural networks with linear and convolutional layers and ReLU activations using tropical geometry. By representing a network's output as a tropical rational function, TropNNC enables structured compression via reduction of the corresponding tropical polynomials. Our method refines the geometric approximation of previous work by adaptively selecting the weights of retained neurons. Key contributions include the first application of tropical geometry to convolutional layers and the tightest known theoretical compression bound. TropNNC requires only access to network weights - no training data - and achieves competitive performance on MNIST, CIFAR, and ImageNet, matching strong baselines such as ThiNet and CUP.
>
---
#### [replaced 019] Deep Learning for Spatio-Temporal Fusion in Land Surface Temperature Estimation: A Comprehensive Survey, Experimental Analysis, and Future Trends
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.16631v2](https://arxiv.org/pdf/2412.16631v2)**

> **作者:** Sofiane Bouaziz; Adel Hafiane; Raphael Canals; Rachid Nedjai
>
> **摘要:** Land Surface Temperature (LST) plays a key role in climate monitoring, urban heat assessment, and land-atmosphere interactions. However, current thermal infrared satellite sensors cannot simultaneously achieve high spatial and temporal resolution. Spatio-temporal fusion (STF) techniques address this limitation by combining complementary satellite data, one with high spatial but low temporal resolution, and another with high temporal but low spatial resolution. Existing STF techniques, from classical models to modern deep learning (DL) architectures, were primarily developed for surface reflectance (SR). Their application to thermal data remains limited and often overlooks LST-specific spatial and temporal variability. This study provides a focused review of DL-based STF methods for LST. We present a formal mathematical definition of the thermal fusion task, propose a refined taxonomy of relevant DL methods, and analyze the modifications required when adapting SR-oriented models to LST. To support reproducibility and benchmarking, we introduce a new dataset comprising 51 Terra MODIS-Landsat LST pairs from 2013 to 2024, and evaluate representative models to explore their behavior on thermal data. The analysis highlights performance gaps, architecture sensitivities, and open research challenges. The dataset and accompanying resources are publicly available at https://github.com/Sofianebouaziz1/STF-LST.
>
---
#### [replaced 020] I Want It That Way! Specifying Nuanced Camera Motions in Video Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.09472v2](https://arxiv.org/pdf/2504.09472v2)**

> **作者:** Pooja Guhan; Divya Kothandaraman; Geonsun Lee; Tsung-Wei Huang; Guan-Ming Su; Dinesh Manocha
>
> **摘要:** Specifying nuanced and compelling camera motion remains a major hurdle for non-expert creators using generative tools, creating an ``expressive gap" where generic text prompts fail to capture cinematic vision. To address this, we present a novel zero-shot diffusion-based system that enables personalized camera motion transfer from a single reference video onto a user-provided static image. Our technical contribution introduces an intuitive interaction paradigm that bypasses the need for 3D data, predefined trajectories, or complex graphical interfaces. The core pipeline leverages a text-to-video diffusion model, employing a two-phase strategy: 1) a multi-concept learning method using LoRA layers and an orthogonality loss to distinctly capture spatial-temporal characteristics and scene features, and 2) a homography-based refinement strategy to enhance temporal and spatial alignment of the generated video. Extensive evaluation demonstrates the efficacy of our method. In a comparative study with 72 participants, our system was significantly preferred over prior work for both motion accuracy (90.45\%) and scene preservation (70.31\%). A second study confirmed our interface significantly improves usability and creative control for video direction. Our work contributes a robust technical solution and a novel human-centered design, significantly expanding cinematic video editing for diverse users.
>
---
#### [replaced 021] Neural Implicit Heart Coordinates: 3D cardiac shape reconstruction from sparse segmentations
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2512.19316v2](https://arxiv.org/pdf/2512.19316v2)**

> **作者:** Marica Muffoletto; Uxio Hermida; Charlène Mauger; Avan Suinesiaputra; Yiyang Xu; Richard Burns; Lisa Pankewitz; Andrew D McCulloch; Steffen E Petersen; Daniel Rueckert; Alistair A Young
>
> **备注:** 42 pages, 8 figures
>
> **摘要:** Accurate reconstruction of cardiac anatomy from sparse clinical images remains a major challenge in patient-specific modeling. While neural implicit functions have previously been applied to this task, their application to mapping anatomical consistency across subjects has been limited. In this work, we introduce Neural Implicit Heart Coordinates (NIHCs), a standardized implicit coordinate system, based on universal ventricular coordinates, that provides a common anatomical reference frame for the human heart. Our method predicts NIHCs directly from a limited number of 2D segmentations (sparse acquisition) and subsequently decodes them into dense 3D segmentations and high-resolution meshes at arbitrary output resolution. Trained on a large dataset of 5,000 cardiac meshes, the model achieves high reconstruction accuracy on clinical contours, with mean Euclidean surface errors of 2.51$\pm$0.33 mm in a diseased cohort (n=4549) and 2.3$\pm$0.36 mm in a healthy cohort (n=5576). The NIHC representation enables anatomically coherent reconstruction even under severe slice sparsity and segmentation noise, faithfully recovering complex structures such as the valve planes. Compared with traditional pipelines, inference time is reduced from over 60 s to 5-15 s. These results demonstrate that NIHCs constitute a robust and efficient anatomical representation for patient-specific 3D cardiac reconstruction from minimal input data.
>
---
#### [replaced 022] COMPACT: COMPositional Atomic-to-Complex Visual Capability Tuning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.21850v2](https://arxiv.org/pdf/2504.21850v2)**

> **作者:** Xindi Wu; Hee Seung Hwang; Polina Kirichenko; Esin Tureci; Olga Russakovsky
>
> **摘要:** Visual instruction tuning (VIT) datasets are constructed from randomly sampled image-question pairs, without regard to the informativeness of each pair. Recent dataset selection methods have shown that a small fraction of such datasets enriched with informative samples can lead to efficient finetuning of Multimodal Large Language Models. In this work, we explore the impact of sample complexity on informative data curation and introduce COMPACT (COMPositional Atomic-to-complex Visual Capability Tuning), a VIT data recipe that scales training sample complexity by combining multiple atomic visual capabilities in a single training example. Concretely, we synthesize rich and informative text questions for each image, allowing us to significantly reduce the number of training examples required for effective visual instruction tuning. COMPACT demonstrates superior data efficiency compared to existing data reduction methods. When applied to the LLAVA-665K VIT dataset, COMPACT reduces the data budget by 90% while still achieving 100.2% of the full VIT performance (compared to only 97.5% by the state-of-the-art method) across eight multimodal benchmarks. Further, training on the COMPACT data outperforms training on the full-scale data on particularly complex benchmarks such as MM-Vet (+8.6%) and MMStar (+2.9%). COMPACT offers a scalable and efficient synthetic data generation recipe to improve on visual language tasks.
>
---
#### [replaced 023] Reinforcement Learning for Large Model: A Survey
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.08189v3](https://arxiv.org/pdf/2508.08189v3)**

> **作者:** Weijia Wu; Chen Gao; Joya Chen; Kevin Qinghong Lin; Qingwei Meng; Yiming Zhang; Yuke Qiu; Hong Zhou; Mike Zheng Shou
>
> **备注:** 22 pages
>
> **摘要:** Recent advances at the intersection of reinforcement learning (RL) and visual intelligence have enabled agents that not only perceive complex visual scenes but also reason, generate, and act within them. This survey offers a critical and up-to-date synthesis of the field. We first formalize visual RL problems and trace the evolution of policy-optimization strategies from RLHF to verifiable reward paradigms, and from Proximal Policy Optimization to Group Relative Policy Optimization. We then organize more than 200 representative works into four thematic pillars: multi-modal large language models, visual generation, unified model frameworks, and vision-language-action models. For each pillar we examine algorithmic design, reward engineering, benchmark progress, and we distill trends such as curriculum-driven training, preference-aligned diffusion, and unified reward modeling. Finally, we review evaluation protocols spanning set-level fidelity, sample-level preference, and state-level stability, and we identify open challenges that include sample efficiency, generalization, and safe deployment. Our goal is to provide researchers and practitioners with a coherent map of the rapidly expanding landscape of visual RL and to highlight promising directions for future inquiry. Resources are available at: https://github.com/weijiawu/Awesome-Visual-Reinforcement-Learning.
>
---
#### [replaced 024] From Binary to Semantic: Utilizing Large-Scale Binary Occupancy Data for 3D Semantic Occupancy Prediction
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.13387v3](https://arxiv.org/pdf/2507.13387v3)**

> **作者:** Chihiro Noguchi; Takaki Yamamoto
>
> **备注:** Accepted to ICCV Workshop 2025
>
> **摘要:** Accurate perception of the surrounding environment is essential for safe autonomous driving. 3D occupancy prediction, which estimates detailed 3D structures of roads, buildings, and other objects, is particularly important for vision-centric autonomous driving systems that do not rely on LiDAR sensors. However, in 3D semantic occupancy prediction -- where each voxel is assigned a semantic label -- annotated LiDAR point clouds are required, making data acquisition costly. In contrast, large-scale binary occupancy data, which only indicate occupied or free space without semantic labels, can be collected at a lower cost. Despite their availability, the potential of leveraging such data remains unexplored. In this study, we investigate the utilization of large-scale binary occupancy data from two perspectives: (1) pre-training and (2) learning-based auto-labeling. We propose a novel binary occupancy-based framework that decomposes the prediction process into binary and semantic occupancy modules, enabling effective use of binary occupancy data. Our experimental results demonstrate that the proposed framework outperforms existing methods in both pre-training and auto-labeling tasks, highlighting its effectiveness in enhancing 3D semantic occupancy prediction. The code will be available at https://github.com/ToyotaInfoTech/b2s-occupancy
>
---
#### [replaced 025] SLIM: Semantic-based Low-bitrate Image compression for Machines by leveraging diffusion
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18200v2](https://arxiv.org/pdf/2512.18200v2)**

> **作者:** Hyeonjin Lee; Jun-Hyuk Kim; Jong-Seok Lee
>
> **摘要:** In recent years, the demand of image compression models for machine vision has increased dramatically. However, the training frameworks of image compression still focus on the vision of human, maintaining the excessive perceptual details, thus have limitations in optimally reducing the bits per pixel in the case of performing machine vision tasks. In this paper, we propose Semantic-based Low-bitrate Image compression for Machines by leveraging diffusion, termed SLIM. This is a new effective training framework of image compression for machine vision, using a pretrained latent diffusion model.The compressor model of our method focuses only on the Region-of-Interest (RoI) areas for machine vision in the image latent, to compress it compactly. Then the pretrained Unet model enhances the decompressed latent, utilizing a RoI-focused text caption which containing semantic information of the image. Therefore, SLIM is able to focus on RoI areas of the image without any guide mask at the inference stage, achieving low bitrate when compressing. And SLIM is also able to enhance a decompressed latent by denoising steps, so the final reconstructed image from the enhanced latent can be optimized for the machine vision task while still containing perceptual details for human vision. Experimental results show that SLIM achieves a higher classification accuracy in the same bits per pixel condition, compared to conventional image compression models for machines.
>
---
#### [replaced 026] Learning Informative Attention Weights for Person Re-Identification
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.08961v2](https://arxiv.org/pdf/2505.08961v2)**

> **作者:** Yancheng Wang; Nebojsa Jojic; Yingzhen Yang
>
> **摘要:** Attention mechanisms have been widely used in deep learning, and recent efforts have been devoted to incorporating attention modules into deep neural networks (DNNs) for person Re-Identification (Re-ID) to enhance their discriminative feature learning capabilities. Existing attention modules, including self-attention and channel attention, learn attention weights that quantify the importance of feature tokens or feature channels. However, existing attention methods do not explicitly ensure that the attention weights are informative for predicting the identity of the person in the input image, and may consequently introduce noisy information from the input image. To address this issue, we propose a novel method termed Reduction of Information Bottleneck loss (RIB), motivated by the principle of the Information Bottleneck (IB). A novel distribution-free and efficient variational upper bound for the IB loss (IBB), which can be optimized by standard SGD, is derived and incorporated into the training loss of the RIB models. RIB is applied to DNNs with self-attention modules through a novel Differentiable Channel Selection Attention module, or DCS-Attention, that selects the most informative channels for computing attention weights, leading to competitive models termed RIB-DCS. RIB is also incorporated into DNNs with existing channel attention modules to promote the learning of informative channel attention weights, leading to models termed RIB-CA. Both RIB-DCS and RIB-CA are applied to fixed neural network backbones and learnable backbones with Differentiable Neural Architecture Search (DNAS). Extensive experiments on multiple person Re-ID benchmarks show that RIB significantly enhances the prediction accuracy of DNNs for person Re-ID, even for the occluded person Re-ID.
>
---
#### [replaced 027] HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17601v2](https://arxiv.org/pdf/2512.17601v2)**

> **作者:** Zhaolin Cai; Fan Li; Ziwei Zheng; Haixia Bi; Lijun He
>
> **备注:** AAAI 2026 Oral
>
> **摘要:** Video Anomaly Detection (VAD) aims to locate events that deviate from normal patterns in videos. Traditional approaches often rely on extensive labeled data and incur high computational costs. Recent tuning-free methods based on Multimodal Large Language Models (MLLMs) offer a promising alternative by leveraging their rich world knowledge. However, these methods typically rely on textual outputs, which introduces information loss, exhibits normalcy bias, and suffers from prompt sensitivity, making them insufficient for capturing subtle anomalous cues. To address these constraints, we propose HeadHunt-VAD, a novel tuning-free VAD paradigm that bypasses textual generation by directly hunting robust anomaly-sensitive internal attention heads within the frozen MLLM. Central to our method is a Robust Head Identification module that systematically evaluates all attention heads using a multi-criteria analysis of saliency and stability, identifying a sparse subset of heads that are consistently discriminative across diverse prompts. Features from these expert heads are then fed into a lightweight anomaly scorer and a temporal locator, enabling efficient and accurate anomaly detection with interpretable outputs. Extensive experiments show that HeadHunt-VAD achieves state-of-the-art performance among tuning-free methods on two major VAD benchmarks while maintaining high efficiency, validating head-level probing in MLLMs as a powerful and practical solution for real-world anomaly detection.
>
---
#### [replaced 028] Binarization-Aware Adjuster for Discrete Decision Learning with an Application to Edge Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12460v3](https://arxiv.org/pdf/2506.12460v3)**

> **作者:** Hao Shu
>
> **备注:** 28 pages
>
> **摘要:** Discrete decision tasks in machine learning exhibit a fundamental misalignment between training and inference: models are optimized with continuous-valued outputs but evaluated using discrete predictions. This misalignment arises from the discontinuity of discretization operations, which prevents decision behavior from being directly incorporated into gradient-based optimization. To address this issue, we propose a theoretically grounded framework termed the Binarization-Aware Adjuster (BAA), which embeds binarization characteristics into continuous optimization. The framework is built upon the Distance Weight Function (DWF), which modulates loss contributions according to prediction correctness and proximity to the decision threshold, thereby aligning optimization emphasis with decision-critical regions while remaining compatible with standard learning pipelines. We apply the proposed BAA framework to the edge detection (ED) task, a representative binary decision problem. Experimental results on representative models and datasets show that incorporating BAA into optimization leads to consistent performance improvements, supporting its effectiveness. Overall, this work establishes a principled approach for aligning continuous optimization with discrete decision behavior, with its effectiveness demonstrated in a concrete application setting.
>
---
#### [replaced 029] Simulated Ensemble Attack: Transferring Jailbreaks Across Fine-tuned Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01741v2](https://arxiv.org/pdf/2508.01741v2)**

> **作者:** Ruofan Wang; Xin Wang; Yang Yao; Xuan Tong; Xingjun Ma
>
> **摘要:** Fine-tuning open-source Vision-Language Models (VLMs) creates a critical yet underexplored attack surface: vulnerabilities in the base VLM could be retained in fine-tuned variants, rendering them susceptible to transferable jailbreak attacks. To demonstrate this risk, we introduce the Simulated Ensemble Attack (SEA), a novel grey-box jailbreak method in which the adversary has full access to the base VLM but no knowledge of the fine-tuned target's weights or training configuration. To improve jailbreak transferability across fine-tuned VLMs, SEA combines two key techniques: Fine-tuning Trajectory Simulation (FTS) and Targeted Prompt Guidance (TPG). FTS generates transferable adversarial images by simulating the vision encoder's parameter shifts, while TPG is a textual strategy that steers the language decoder toward adversarially optimized outputs. Experiments on the Qwen2-VL family (2B and 7B) demonstrate that SEA achieves high transfer attack success rates exceeding 86.5% and toxicity rates near 49.5% across diverse fine-tuned variants, even those specifically fine-tuned to improve safety behaviors. Notably, while direct PGD-based image jailbreaks rarely transfer across fine-tuned VLMs, SEA reliably exploits inherited vulnerabilities from the base model, significantly enhancing transferability. These findings highlight an urgent need to safeguard fine-tuned proprietary VLMs against transferable vulnerabilities inherited from open-source foundations, motivating the development of holistic defenses across the entire model lifecycle.
>
---
#### [replaced 030] Image Matching Filtering and Refinement by Planes and Beyond
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.09484v4](https://arxiv.org/pdf/2411.09484v4)**

> **作者:** Fabio Bellavia; Zhenjun Zhao; Luca Morelli; Fabio Remondino
>
> **备注:** project page: https://github.com/fb82/MiHo
>
> **摘要:** This paper introduces a modular, non-deep learning method for filtering and refining sparse correspondences in image matching. Assuming that motion flow within the scene can be approximated by local homography transformations, matches are aggregated into overlapping clusters corresponding to virtual planes using an iterative RANSAC-based approach discarding incompatible correspondences. Moreover, the underlying planar structural design provides an explicit map between local patches associated with the matches, by which optionally refine the keypoint positions through cross-correlation template matching after the patch reprojection. Finally, to enhance robustness and fault-tolerance against violations of the piece-wise planar approximation assumption, a further strategy is designed in order to minimize the relative patch distortion in the plane reprojection by introducing an intermediate homography that projects both patches into a common plane. The proposed method is extensively evaluated on standard datasets and image matching pipelines, and compared with state-of-the-art approaches. Unlike other current comparisons, the proposed benchmark also takes into account the more general, real, and practical cases where camera intrinsics are unavailable. Experimental results demonstrate that our proposed non-deep learning, geometry-based filter is effective in presence of outliers and the optional cross-correlation refinement step is valid in the case of corner-like keypoints. Finally, this study suggests that there is still significant development potential in practical image matching solutions in the considered research direction, which could be in the future incorporated in novel deep image matching architectures.
>
---
#### [replaced 031] Compression for Better: A General and Stable Lossless Compression Framework
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2412.06868v2](https://arxiv.org/pdf/2412.06868v2)**

> **作者:** Boyang Zhang; Daning Cheng; Yunquan Zhang; Fangming Liu; Wenguang Chen
>
> **备注:** Under Review
>
> **摘要:** This work focus on how to stabilize and lossless model compression, aiming to reduce model complexity and enhance efficiency without sacrificing performance due to compression errors. A key challenge is effectively leveraging compression errors and defining the boundaries for lossless compression to minimize model loss. i.e., compression for better. Currently, there is no systematic approach to determining this error boundary or understanding its specific impact on model performance. We propose a general \textbf{L}oss\textbf{L}ess \textbf{C}ompression theoretical framework (\textbf{LLC}), which further delineates the compression neighborhood and higher-order analysis boundaries through the total differential, thereby specifying the error range within which a model can be compressed without loss. To verify the effectiveness of LLC, we apply various compression techniques, including quantization and decomposition. Specifically, for quantization, we reformulate the classic quantization search problem as a grouped knapsack problem within the lossless neighborhood, achieving lossless quantization while improving computational efficiency. For decomposition, LLC addresses the approximation problem under low-rank constraints, automatically determining the rank for each layer and producing lossless low-rank models. We conduct extensive experiments on multiple neural network architectures on different datasets. The results show that without fancy tricks, LLC can effectively achieve lossless model compression. Our code will be made publicly.
>
---
#### [replaced 032] Regressor-Guided Generative Image Editing Balances User Emotions to Reduce Time Spent Online
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [https://arxiv.org/pdf/2501.12289v2](https://arxiv.org/pdf/2501.12289v2)**

> **作者:** Christoph Gebhardt; Robin Willardt; Seyedmorteza Sadat; Chih-Wei Ning; Andreas Brombach; Jie Song; Otmar Hilliges; Christian Holz
>
> **备注:** 44 pages, 22 figures
>
> **摘要:** Internet overuse is a widespread phenomenon in today's digital society. Existing interventions, such as time limits or grayscaling, often rely on restrictive controls that provoke psychological reactance and are frequently circumvented. Building on prior work showing that emotional responses mediate the relationship between content consumption and online engagement, we investigate whether regulating the emotional impact of images can reduce online use in a non-coercive manner. We introduce and systematically analyze three regressor-guided image-editing approaches: (i) global optimization of emotion-related image attributes, (ii) optimization in a style latent space, and (iii) a diffusion-based method using classifier and classifier-free guidance. While the first two approaches modify low-level visual features (e.g., contrast, color), the diffusion-based method enables higher-level changes (e.g., adjusting clothing, facial features). Results from a controlled image-rating study and a social media experiment show that diffusion-based edits balance emotional responses and are associated with lower usage duration while preserving visual quality.
>
---
#### [replaced 033] UniMPR: A Unified Framework for Multimodal Place Recognition with Heterogeneous Sensor Configurations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18279v2](https://arxiv.org/pdf/2512.18279v2)**

> **作者:** Zhangshuo Qi; Jingyi Xu; Luqi Cheng; Shichen Wen; Yiming Ma; Guangming Xiong
>
> **备注:** 14 pages, 9 figures
>
> **摘要:** Place recognition is a critical component of autonomous vehicles and robotics, enabling global localization in GPS-denied environments. Recent advances have spurred significant interest in multimodal place recognition (MPR), which leverages complementary strengths of multiple modalities. Despite its potential, most existing MPR methods still face three key challenges: (1) dynamically adapting to various modality inputs within a unified framework, (2) maintaining robustness with missing or degraded modalities, and (3) generalizing across diverse sensor configurations and setups. In this paper, we propose UniMPR, a unified framework for multimodal place recognition. Using only one trained model, it can seamlessly adapt to any combination of common perceptual modalities (e.g., camera, LiDAR, radar). To tackle the data heterogeneity, we unify all inputs within a polar BEV feature space. Subsequently, the polar BEVs are fed into a multi-branch network to exploit discriminative intra-model and inter-modal features from any modality combinations. To fully exploit the network's generalization capability and robustness, we construct a large-scale training set from multiple datasets and introduce an adaptive label assignment strategy for extensive pre-training. Experiments on seven datasets demonstrate that UniMPR achieves state-of-the-art performance under varying sensor configurations, modality combinations, and environmental conditions. Our code will be released at https://github.com/QiZS-BIT/UniMPR.
>
---
#### [replaced 034] Memorize-and-Generate: Towards Long-Term Consistency in Real-Time Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18741v2](https://arxiv.org/pdf/2512.18741v2)**

> **作者:** Tianrui Zhu; Shiyi Zhang; Zhirui Sun; Jingqi Tian; Yansong Tang
>
> **备注:** Code will be released at https://github.com/Xilluill/MAG
>
> **摘要:** Frame-level autoregressive (frame-AR) models have achieved significant progress, enabling real-time video generation comparable to bidirectional diffusion models and serving as a foundation for interactive world models and game engines. However, current approaches in long video generation typically rely on window attention, which naively discards historical context outside the window, leading to catastrophic forgetting and scene inconsistency; conversely, retaining full history incurs prohibitive memory costs. To address this trade-off, we propose Memorize-and-Generate (MAG), a framework that decouples memory compression and frame generation into distinct tasks. Specifically, we train a memory model to compress historical information into a compact KV cache, and a separate generator model to synthesize subsequent frames utilizing this compressed representation. Furthermore, we introduce MAG-Bench to strictly evaluate historical memory retention. Extensive experiments demonstrate that MAG achieves superior historical scene consistency while maintaining competitive performance on standard video generation benchmarks.
>
---
#### [replaced 035] EmoCAST: Emotional Talking Portrait via Emotive Text Description
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.20615v2](https://arxiv.org/pdf/2508.20615v2)**

> **作者:** Yiguo Jiang; Xiaodong Cun; Yong Zhang; Yudian Zheng; Fan Tang; Chi-Man Pun
>
> **摘要:** Emotional talking head synthesis aims to generate talking portrait videos with vivid expressions. Existing methods still exhibit limitations in control flexibility, motion naturalness, and expression quality. Moreover, currently available datasets are mainly collected in lab settings, further exacerbating these shortcomings and hindering real-world deployment. To address these challenges, we propose EmoCAST, a diffusion-based talking head framework for precise, text-driven emotional synthesis. Its contributions are threefold: (1) architectural modules that enable effective text control; (2) an emotional talking-head dataset that expands the framework's ability; and (3) training strategies that further improve performance. Specifically, for appearance modeling, emotional prompts are integrated through a text-guided emotive attention module, enhancing spatial knowledge to improve emotion understanding. To strengthen audio-emotion alignment, we introduce an emotive audio attention module to capture the interplay between controlled emotion and driving audio, generating emotion-aware features to guide precise facial motion synthesis. Additionally, we construct a large-scale, in-the-wild emotional talking head dataset with emotive text descriptions to optimize the framework's performance. Based on this dataset, we propose an emotion-aware sampling strategy and a progressive functional training strategy that improve the model's ability to capture nuanced expressive features and achieve accurate lip-sync. Overall, EmoCAST achieves state-of-the-art performance in generating realistic, emotionally expressive, and audio-synchronized talking-head videos. Project Page: https://github.com/GVCLab/EmoCAST
>
---
#### [replaced 036] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文面向点云配准任务，解决现有方法在新场景中需重训练或调参的泛化瓶颈。提出零样本配准框架BUFFER-X：自适应体素/搜索半径、用最远点采样替代学习型关键点检测、分块尺度归一化，并构建多尺度描述符与层级内点搜索，显著提升跨场景鲁棒性。**

- **链接: [https://arxiv.org/pdf/2503.07940v3](https://arxiv.org/pdf/2503.07940v3)**

> **作者:** Minkyun Seo; Hyungtae Lim; Kanghee Lee; Luca Carlone; Jaesik Park
>
> **备注:** 20 pages, 14 figures. Accepted as a highlight paper at ICCV 2025
>
> **摘要:** Recent advances in deep learning-based point cloud registration have improved generalization, yet most methods still require retraining or manual parameter tuning for each new environment. In this paper, we identify three key factors limiting generalization: (a) reliance on environment-specific voxel size and search radius, (b) poor out-of-domain robustness of learning-based keypoint detectors, and (c) raw coordinate usage, which exacerbates scale discrepancies. To address these issues, we present a zero-shot registration pipeline called BUFFER-X by (a) adaptively determining voxel size/search radii, (b) using farthest point sampling to bypass learned detectors, and (c) leveraging patch-wise scale normalization for consistent coordinate bounds. In particular, we present a multi-scale patch-based descriptor generation and a hierarchical inlier search across scales to improve robustness in diverse scenes. We also propose a novel generalizability benchmark using 11 datasets that cover various indoor/outdoor scenarios and sensor modalities, demonstrating that BUFFER-X achieves substantial generalization without prior information or manual parameter tuning for the test datasets. Our code is available at https://github.com/MIT-SPARK/BUFFER-X.
>
---
#### [replaced 037] Seedance 1.5 pro: A Native Audio-Visual Joint Generation Foundation Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13507v3](https://arxiv.org/pdf/2512.13507v3)**

> **作者:** Team Seedance; Heyi Chen; Siyan Chen; Xin Chen; Yanfei Chen; Ying Chen; Zhuo Chen; Feng Cheng; Tianheng Cheng; Xinqi Cheng; Xuyan Chi; Jian Cong; Jing Cui; Qinpeng Cui; Qide Dong; Junliang Fan; Jing Fang; Zetao Fang; Chengjian Feng; Han Feng; Mingyuan Gao; Yu Gao; Dong Guo; Qiushan Guo; Boyang Hao; Qingkai Hao; Bibo He; Qian He; Tuyen Hoang; Ruoqing Hu; Xi Hu; Weilin Huang; Zhaoyang Huang; Zhongyi Huang; Donglei Ji; Siqi Jiang; Wei Jiang; Yunpu Jiang; Zhuo Jiang; Ashley Kim; Jianan Kong; Zhichao Lai; Shanshan Lao; Yichong Leng; Ai Li; Feiya Li; Gen Li; Huixia Li; JiaShi Li; Liang Li; Ming Li; Shanshan Li; Tao Li; Xian Li; Xiaojie Li; Xiaoyang Li; Xingxing Li; Yameng Li; Yifu Li; Yiying Li; Chao Liang; Han Liang; Jianzhong Liang; Ying Liang; Zhiqiang Liang; Wang Liao; Yalin Liao; Heng Lin; Kengyu Lin; Shanchuan Lin; Xi Lin; Zhijie Lin; Feng Ling; Fangfang Liu; Gaohong Liu; Jiawei Liu; Jie Liu; Jihao Liu; Shouda Liu; Shu Liu; Sichao Liu; Songwei Liu; Xin Liu; Xue Liu; Yibo Liu; Zikun Liu; Zuxi Liu; Junlin Lyu; Lecheng Lyu; Qian Lyu; Han Mu; Xiaonan Nie; Jingzhe Ning; Xitong Pan; Yanghua Peng; Lianke Qin; Xueqiong Qu; Yuxi Ren; Kai Shen; Guang Shi; Lei Shi; Yan Song; Yinglong Song; Fan Sun; Li Sun; Renfei Sun; Yan Sun; Zeyu Sun; Wenjing Tang; Yaxue Tang; Zirui Tao; Feng Wang; Furui Wang; Jinran Wang; Junkai Wang; Ke Wang; Kexin Wang; Qingyi Wang; Rui Wang; Sen Wang; Shuai Wang; Tingru Wang; Weichen Wang; Xin Wang; Yanhui Wang; Yue Wang; Yuping Wang; Yuxuan Wang; Ziyu Wang; Guoqiang Wei; Wanru Wei; Di Wu; Guohong Wu; Hanjie Wu; Jian Wu; Jie Wu; Ruolan Wu; Xinglong Wu; Yonghui Wu; Ruiqi Xia; Liang Xiang; Fei Xiao; XueFeng Xiao; Pan Xie; Shuangyi Xie; Shuang Xu; Jinlan Xue; Shen Yan; Bangbang Yang; Ceyuan Yang; Jiaqi Yang; Runkai Yang; Tao Yang; Yang Yang; Yihang Yang; ZhiXian Yang; Ziyan Yang; Songting Yao; Yifan Yao; Zilyu Ye; Bowen Yu; Jian Yu; Chujie Yuan; Linxiao Yuan; Sichun Zeng; Weihong Zeng; Xuejiao Zeng; Yan Zeng; Chuntao Zhang; Heng Zhang; Jingjie Zhang; Kuo Zhang; Liang Zhang; Liying Zhang; Manlin Zhang; Ting Zhang; Weida Zhang; Xiaohe Zhang; Xinyan Zhang; Yan Zhang; Yuan Zhang; Zixiang Zhang; Fengxuan Zhao; Huating Zhao; Yang Zhao; Hao Zheng; Jianbin Zheng; Xiaozheng Zheng; Yangyang Zheng; Yijie Zheng; Jiexin Zhou; Jiahui Zhu; Kuan Zhu; Shenhan Zhu; Wenjia Zhu; Benhui Zou; Feilong Zuo
>
> **备注:** Seedance 1.5 pro Technical Report
>
> **摘要:** Recent strides in video generation have paved the way for unified audio-visual generation. In this work, we present Seedance 1.5 pro, a foundational model engineered specifically for native, joint audio-video generation. Leveraging a dual-branch Diffusion Transformer architecture, the model integrates a cross-modal joint module with a specialized multi-stage data pipeline, achieving exceptional audio-visual synchronization and superior generation quality. To ensure practical utility, we implement meticulous post-training optimizations, including Supervised Fine-Tuning (SFT) on high-quality datasets and Reinforcement Learning from Human Feedback (RLHF) with multi-dimensional reward models. Furthermore, we introduce an acceleration framework that boosts inference speed by over 10X. Seedance 1.5 pro distinguishes itself through precise multilingual and dialect lip-syncing, dynamic cinematic camera control, and enhanced narrative coherence, positioning it as a robust engine for professional-grade content creation. Seedance 1.5 pro is now accessible on Volcano Engine at https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?type=GenVideo.
>
---
#### [replaced 038] VTCBench: Can Vision-Language Models Understand Long Context with Vision-Text Compression?
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属多模态长上下文理解任务，旨在探究视觉-文本压缩（VTC）对VLM长上下文能力的影响。作者构建首个VTC专用基准VTCBench，含检索、推理、记忆三类子任务，并评估主流模型，发现VTC虽提升效率，却严重损害VLM的长依赖建模能力。**

- **链接: [https://arxiv.org/pdf/2512.15649v2](https://arxiv.org/pdf/2512.15649v2)**

> **作者:** Hongbo Zhao; Meng Wang; Fei Zhu; Wenzhuo Liu; Bolin Ni; Fanhu Zeng; Gaofeng Meng; Zhaoxiang Zhang
>
> **摘要:** The computational and memory overheads associated with expanding the context window of LLMs severely limit their scalability. A noteworthy solution is vision-text compression (VTC), exemplified by frameworks like DeepSeek-OCR and Glyph, which convert long texts into dense 2D visual representations, thereby achieving token compression ratios of 3x-20x. However, the impact of this high information density on the core long-context capabilities of vision-language models (VLMs) remains under-investigated. To address this gap, we introduce the first benchmark for VTC and systematically assess the performance of VLMs across three long-context understanding settings: VTC-Retrieval, which evaluates the model's ability to retrieve and aggregate information; VTC-Reasoning, which requires models to infer latent associations to locate facts with minimal lexical overlap; and VTC-Memory, which measures comprehensive question answering within long-term dialogue memory. Furthermore, we establish the VTCBench-Wild to simulate diverse input scenarios.We comprehensively evaluate leading open-source and proprietary models on our benchmarks. The results indicate that, despite being able to decode textual information (e.g., OCR) well, most VLMs exhibit a surprisingly poor long-context understanding ability with VTC-processed information, failing to capture long associations or dependencies in the context.This study provides a deep understanding of VTC and serves as a foundation for designing more efficient and scalable VLMs.
>
---
#### [replaced 039] Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属行为克隆（BC）任务，旨在解决语言条件操控中动作序列的累积误差、物理不连续及语义-物理错位问题。提出CCoL框架，通过视觉-语言-本体感知的连续协同学习与双向交叉注意力对齐语义与物理表征，提升动作执行的连贯性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.14396v5](https://arxiv.org/pdf/2511.14396v5)**

> **作者:** Xiuxiu Qi; Yu Yang; Jiannong Cao; Luyao Bai; Chongshan Fan; Chengtai Cao; Hongpeng Wang
>
> **备注:** Accepted at AAAI 2026, the Project website is available at https://qhemu.github.io/CCoL/
>
> **摘要:** Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states.
>
---
#### [replaced 040] Video Generation Models Are Good Latent Reward Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21541v2](https://arxiv.org/pdf/2511.21541v2)**

> **作者:** Xiaoyue Mi; Wenqing Yu; Jiesong Lian; Shibo Jie; Ruizhe Zhong; Zijun Liu; Guozhen Zhang; Zixiang Zhou; Zhiyong Xu; Yuan Zhou; Qinglin Lu; Fan Tang
>
> **摘要:** Reward feedback learning (ReFL) has proven effective for aligning image generation with human preferences. However, its extension to video generation faces significant challenges. Existing video reward models rely on vision-language models designed for pixel-space inputs, confining ReFL optimization to near-complete denoising steps after computationally expensive VAE decoding. This pixel-space approach incurs substantial memory overhead and increased training time, and its late-stage optimization lacks early-stage supervision, refining only visual quality rather than fundamental motion dynamics and structural coherence. In this work, we show that pre-trained video generation models are naturally suited for reward modeling in the noisy latent space, as they are explicitly designed to process noisy latent representations at arbitrary timesteps and inherently preserve temporal information through their sequential modeling capabilities. Accordingly, we propose Process Reward Feedback Learning~(PRFL), a framework that conducts preference optimization entirely in latent space, enabling efficient gradient backpropagation throughout the full denoising chain without VAE decoding. Extensive experiments demonstrate that PRFL significantly improves alignment with human preferences, while achieving substantial reductions in memory consumption and training time compared to RGB ReFL.
>
---
#### [replaced 041] WaveletGaussian: Wavelet-domain Diffusion for Sparse-view 3D Gaussian Object Reconstruction
- **分类: cs.CV; eess.IV; eess.SP**

- **链接: [https://arxiv.org/pdf/2509.19073v2](https://arxiv.org/pdf/2509.19073v2)**

> **作者:** Hung Nguyen; Runfa Li; An Le; Truong Nguyen
>
> **摘要:** 3D Gaussian Splatting (3DGS) has become a powerful representation for image-based object reconstruction, yet its performance drops sharply in sparse-view settings. Prior works address this limitation by employing diffusion models to repair corrupted renders, subsequently using them as pseudo ground truths for later optimization. While effective, such approaches incur heavy computation from the diffusion fine-tuning and repair steps. We present WaveletGaussian, a framework for more efficient sparse-view 3D Gaussian object reconstruction. Our key idea is to shift diffusion into the wavelet domain: diffusion is applied only to the low-resolution LL subband, while high-frequency subbands are refined with a lightweight network. We further propose an efficient online random masking strategy to curate training pairs for diffusion fine-tuning, replacing the commonly used, but inefficient, leave-one-out strategy. Experiments across two benchmark datasets, Mip-NeRF 360 and OmniObject3D, show WaveletGaussian achieves competitive rendering quality while substantially reducing training time.
>
---
#### [replaced 042] Reinforcement Learning for Unsupervised Video Summarization with Reward Generator Training
- **分类: cs.MM; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2407.04258v2](https://arxiv.org/pdf/2407.04258v2)**

> **作者:** Mehryar Abbasi; Hadi Hadizadeh; Parvaneh Saeedi
>
> **备注:** in IEEE Transactions on Circuits and Systems for Video Technology
>
> **摘要:** This paper presents a novel approach for unsupervised video summarization using reinforcement learning (RL), addressing limitations like unstable adversarial training and reliance on heuristic-based reward functions. The method operates on the principle that reconstruction fidelity serves as a proxy for informativeness, correlating summary quality with reconstruction ability. The summarizer model assigns importance scores to frames to generate the final summary. For training, RL is coupled with a unique reward generation pipeline that incentivizes improved reconstructions. This pipeline uses a generator model to reconstruct the full video from the selected summary frames; the similarity between the original and reconstructed video provides the reward signal. The generator itself is pre-trained self-supervisedly to reconstruct randomly masked frames. This two-stage training process enhances stability compared to adversarial architectures. Experimental results show strong alignment with human judgments and promising F-scores, validating the reconstruction objective.
>
---
#### [replaced 043] GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.11340v2](https://arxiv.org/pdf/2501.11340v2)**

> **作者:** Zhenliang Ni; Qiangyu Yan; Mouxiao Huang; Tianning Yuan; Yehui Tang; Hailin Hu; Xinghao Chen; Yunhe Wang
>
> **备注:** AAAI 2026
>
> **摘要:** The rapid advancement of video generation models has made it increasingly challenging to distinguish AI-generated videos from real ones. This issue underscores the urgent need for effective AI-generated video detectors to prevent the dissemination of false information via such videos. However, the development of high-performance AI-generated video detectors is currently impeded by the lack of large-scale, high-quality datasets specifically designed for generative video detection. To this end, we introduce GenVidBench, a challenging AI-generated video detection dataset with several key advantages: 1) Large-scale video collection: The dataset contains 6.78 million videos and is currently the largest dataset for AI-generated video detection. 2) Cross-Source and Cross-Generator: The cross-source generation reduces the interference of video content on the detection. The cross-generator ensures diversity in video attributes between the training and test sets, preventing them from being overly similar. 3) State-of-the-Art Video Generators: The dataset includes videos from 11 state-of-the-art AI video generators, ensuring that it covers the latest advancements in the field of video generation. These generators ensure that the datasets are not only large in scale but also diverse, aiding in the development of generalized and effective detection models. Additionally, we present extensive experimental results with advanced video classification models. With GenVidBench, researchers can efficiently develop and evaluate AI-generated video detection models.. Datasets and code are available at https://genvidbench.github.io.
>
---
#### [replaced 044] LAMIC: Layout-Aware Multi-Image Composition via Scalability of Multimodal Diffusion Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.00477v2](https://arxiv.org/pdf/2508.00477v2)**

> **作者:** Yuzhuo Chen; Zehua Ma; Jianhua Wang; Kai Kang; Shunyu Yao; Weiming Zhang
>
> **备注:** 8 pages, 5 figures, 3 tables
>
> **摘要:** In controllable image synthesis, generating coherent and consistent images from multiple references with spatial layout awareness remains an open challenge. We present LAMIC, a Layout-Aware Multi-Image Composition framework that, for the first time, extends single-reference diffusion models to multi-reference scenarios in a training-free manner. Built upon the MMDiT model, LAMIC introduces two plug-and-play attention mechanisms: 1) Group Isolation Attention (GIA) to enhance entity disentanglement; and 2) Region-Modulated Attention (RMA) to enable layout-aware generation. To comprehensively evaluate model capabilities, we further introduce three metrics: 1) Inclusion Ratio (IN-R) and Fill Ratio (FI-R) for assessing layout control; and 2) Background Similarity (BG-S) for measuring background consistency. Extensive experiments show that LAMIC achieves state-of-the-art performance across most major metrics: it consistently outperforms existing multi-reference baselines in ID-S, BG-S, IN-R and AVG scores across all settings, and achieves the best DPG in complex composition tasks. These results demonstrate LAMIC's superior abilities in identity keeping, background preservation, layout control, and prompt-following, all achieved without any training or fine-tuning, showcasing strong zero-shot generalization ability. By inheriting the strengths of advanced single-reference models and enabling seamless extension to multi-image scenarios, LAMIC establishes a new training-free paradigm for controllable multi-image composition. As foundation models continue to evolve, LAMIC's performance is expected to scale accordingly. Our implementation is available at: https://github.com/Suchenl/LAMIC.
>
---
#### [replaced 045] Resolution scaling governs DINOv3 transfer performance in chest radiograph classification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.07191v2](https://arxiv.org/pdf/2510.07191v2)**

> **作者:** Soroosh Tayebi Arasteh; Mina Shaigan; Christiane Kuhl; Jakob Nikolas Kather; Sven Nebelung; Daniel Truhn
>
> **摘要:** Self-supervised learning (SSL) has advanced visual representation learning, but its value in chest radiography, a high-volume imaging modality with fine-grained findings, remains unclear. Meta's DINOv3 extends earlier SSL models through Gram-anchored self-distillation. Whether these design choices improve transfer learning for chest radiography has not been systematically tested. We benchmarked DINOv3 against DINOv2 and ImageNet initialization across seven datasets (n>814,000). Two representative backbones were evaluated: ViT-B/16 and ConvNeXt-B. Images were analyzed at 224x224, 512x512, and 1024x1024 pixels. We additionally assessed frozen features from a 7B model. The primary outcome was mean AUROC across labels. At 224x224, DINOv3 and DINOv2 achieved comparable performance on adult datasets. Increasing resolution to 512x512 yielded consistent improvements for DINOv3 over both DINOv2 and ImageNet. In contrast, results in pediatric cohort showed no differences across initializations. Across all settings, ConvNeXt-B outperformed ViT-B/16. Models using frozen DINOv3-7B features underperformed relative to fully finetuned 86-89M-parameter backbones, highlighting the importance of domain adaptation. Scaling to 1024x1024 did not further improve accuracy. Resolution-related gains were most evident for boundary-dependent and small focal abnormalities. In chest radiography, higher input resolution is critical for leveraging the benefits of modern self-supervised models. 512x512 pixels represent a practical upper limit where DINOv3-initialized ConvNeXt-B networks provide the strongest performance, while larger inputs offer minimal return on cost. Clinically, these findings support use of finetuned, mid-sized backbones at 512x512 for chest radiograph interpretation, with the greatest gains expected in detecting subtle or boundary-centered lesions relevant to emergency and critical care settings.
>
---
#### [replaced 046] Next-Embedding Prediction Makes Strong Vision Learners
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.16922v2](https://arxiv.org/pdf/2512.16922v2)**

> **作者:** Sihan Xu; Ziqiao Ma; Wenhao Chai; Xuweiyi Chen; Weiyang Jin; Joyce Chai; Saining Xie; Stella X. Yu
>
> **备注:** Project Page: https://sihanxu.me/nepa
>
> **摘要:** Inspired by the success of generative pretraining in natural language, we ask whether the same principles can yield strong self-supervised visual learners. Instead of training models to output features for downstream use, we train them to generate embeddings to perform predictive tasks directly. This work explores such a shift from learning representations to learning models. Specifically, models learn to predict future patch embeddings conditioned on past ones, using causal masking and stop gradient, which we refer to as Next-Embedding Predictive Autoregression (NEPA). We demonstrate that a simple Transformer pretrained on ImageNet-1k with next embedding prediction as its sole learning objective is effective - no pixel reconstruction, discrete tokens, contrastive loss, or task-specific heads. This formulation retains architectural simplicity and scalability, without requiring additional design complexity. NEPA achieves strong results across tasks, attaining 83.8% and 85.3% top-1 accuracy on ImageNet-1K with ViT-B and ViT-L backbones after fine-tuning, and transferring effectively to semantic segmentation on ADE20K. We believe generative pretraining from embeddings provides a simple, scalable, and potentially modality-agnostic alternative to visual self-supervised learning.
>
---
#### [replaced 047] LoGoPlanner: Localization Grounded Navigation Policy with Metric-aware Visual Geometry
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LoGoPlanner，面向移动机器人在非结构化环境中的端到端导航任务，旨在解决传统模块化方法误差累积与现有端到端方法依赖外部定位、泛化性差的问题。其通过度量感知视觉几何骨干网络实现隐式定位、历史几何重建与几何条件策略学习，提升规划一致性与跨平台泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.19629v2](https://arxiv.org/pdf/2512.19629v2)**

> **作者:** Jiaqi Peng; Wenzhe Cai; Yuqiang Yang; Tai Wang; Yuan Shen; Jiangmiao Pang
>
> **备注:** Project page:https://steinate.github.io/logoplanner.github.io/
>
> **摘要:** Trajectory planning in unstructured environments is a fundamental and challenging capability for mobile robots. Traditional modular pipelines suffer from latency and cascading errors across perception, localization, mapping, and planning modules. Recent end-to-end learning methods map raw visual observations directly to control signals or trajectories, promising greater performance and efficiency in open-world settings. However, most prior end-to-end approaches still rely on separate localization modules that depend on accurate sensor extrinsic calibration for self-state estimation, thereby limiting generalization across embodiments and environments. We introduce LoGoPlanner, a localization-grounded, end-to-end navigation framework that addresses these limitations by: (1) finetuning a long-horizon visual-geometry backbone to ground predictions with absolute metric scale, thereby providing implicit state estimation for accurate localization; (2) reconstructing surrounding scene geometry from historical observations to supply dense, fine-grained environmental awareness for reliable obstacle avoidance; and (3) conditioning the policy on implicit geometry bootstrapped by the aforementioned auxiliary tasks, thereby reducing error propagation. We evaluate LoGoPlanner in both simulation and real-world settings, where its fully end-to-end design reduces cumulative error while metric-aware geometry memory enhances planning consistency and obstacle avoidance, leading to more than a 27.3\% improvement over oracle-localization baselines and strong generalization across embodiments and environments. The code and models have been made publicly available on the https://steinate.github.io/logoplanner.github.io.
>
---
#### [replaced 048] FiGO: Fine-Grained Object Counting without Annotations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.11705v4](https://arxiv.org/pdf/2504.11705v4)**

> **作者:** Adriano D'Alessandro; Ali Mahdavi-Amiri; Ghassan Hamarneh
>
> **备注:** data - https://dalessandro.dev/datasets/lookalikes/
>
> **摘要:** Class-agnostic counting (CAC) methods reduce annotation costs by letting users define what to count at test-time through text or visual exemplars. However, current open-vocabulary approaches work well for broad categories but fail when fine-grained category distinctions are needed, such as telling apart waterfowl species or pepper cultivars. We present FiGO, a new annotation-free method that adapts existing counting models to fine-grained categories using only the category name. Our approach uses a text-to-image diffusion model to create synthetic examples and a joint positive/hard-negative loss to learn a compact concept embedding that conditions a specialization module to convert outputs from any frozen counter into accurate, fine-grained estimates. To evaluate fine-grained counting, we introduce LOOKALIKES, a dataset of 37 subcategories across 14 parent categories with many visually similar objects per image. Our method substantially outperforms strong open-vocabulary baselines, moving counting systems from "count all the peppers" to "count only the habaneros."
>
---
