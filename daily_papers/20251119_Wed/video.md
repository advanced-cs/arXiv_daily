# 计算机视觉 cs.CV

- **最新发布 154 篇**

- **更新 117 篇**

## 最新发布

#### [new 001] iGaussian: Real-Time Camera Pose Estimation via Feed-Forward 3D Gaussian Splatting Inversion
- **分类: cs.CV**

- **简介: 论文提出iGaussian，用于实时相机位姿估计任务。针对传统迭代方法计算开销大、无法实时的问题，设计两阶段前馈框架：先用基于高斯场景先验的网络粗估位姿，再通过特征匹配与多视角融合精调。核心创新为无需可微渲染的跨相关模块和加权多视图预测器，显著提升速度与精度。**

- **链接: [https://arxiv.org/pdf/2511.14149v1](https://arxiv.org/pdf/2511.14149v1)**

> **作者:** Hao Wang; Linqing Zhao; Xiuwei Xu; Jiwen Lu; Haibin Yan
>
> **备注:** IROS 2025
>
> **摘要:** Recent trends in SLAM and visual navigation have embraced 3D Gaussians as the preferred scene representation, highlighting the importance of estimating camera poses from a single image using a pre-built Gaussian model. However, existing approaches typically rely on an iterative \textit{render-compare-refine} loop, where candidate views are first rendered using NeRF or Gaussian Splatting, then compared against the target image, and finally, discrepancies are used to update the pose. This multi-round process incurs significant computational overhead, hindering real-time performance in robotics. In this paper, we propose iGaussian, a two-stage feed-forward framework that achieves real-time camera pose estimation through direct 3D Gaussian inversion. Our method first regresses a coarse 6DoF pose using a Gaussian Scene Prior-based Pose Regression Network with spatial uniform sampling and guided attention mechanisms, then refines it through feature matching and multi-model fusion. The key contribution lies in our cross-correlation module that aligns image embeddings with 3D Gaussian attributes without differentiable rendering, coupled with a Weighted Multiview Predictor that fuses features from Multiple strategically sampled viewpoints. Experimental results on the NeRF Synthetic, Mip-NeRF 360, and T\&T+DB datasets demonstrate a significant performance improvement over previous methods, reducing median rotation errors to 0.2° while achieving 2.87 FPS tracking on mobile robots, which is an impressive 10 times speedup compared to optimization-based approaches. Code: https://github.com/pythongod-exe/iGaussian
>
---
#### [new 002] Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于动态场景重建任务，旨在无物体先验条件下重建手与物体交互的几何与外观。提出交互感知的4D高斯点绘制方法，通过优化参数建模相互遮挡和边缘模糊，并引入手信息增强物体变形场，结合渐进式优化策略提升重建质量。**

- **链接: [https://arxiv.org/pdf/2511.14540v1](https://arxiv.org/pdf/2511.14540v1)**

> **作者:** Hao Tian; Chenyangguang Zhang; Rui Liu; Wen Shen; Xiaolin Qin
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** This paper focuses on a challenging setting of simultaneously modeling geometry and appearance of hand-object interaction scenes without any object priors. We follow the trend of dynamic 3D Gaussian Splatting based methods, and address several significant challenges. To model complex hand-object interaction with mutual occlusion and edge blur, we present interaction-aware hand-object Gaussians with newly introduced optimizable parameters aiming to adopt piecewise linear hypothesis for clearer structural representation. Moreover, considering the complementarity and tightness of hand shape and object shape during interaction dynamics, we incorporate hand information into object deformation field, constructing interaction-aware dynamic fields to model flexible motions. To further address difficulties in the optimization process, we propose a progressive strategy that handles dynamic regions and static background step by step. Correspondingly, explicit regularizations are designed to stabilize the hand-object representations for smooth motion transition, physical interaction reality, and coherent lighting. Experiments show that our approach surpasses existing dynamic 3D-GS-based methods and achieves state-of-the-art performance in reconstructing dynamic hand-object interaction.
>
---
#### [new 003] 3D-Guided Scalable Flow Matching for Generating Volumetric Tissue Spatial Transcriptomics from Serial Histology
- **分类: cs.CV**

- **简介: 该论文提出HoloTea，一种3D-aware流匹配框架，用于从连续组织切片生成体素级空间转录组数据。解决现有方法忽略三维结构或无法扩展的问题，通过跨切面特征对齐和3D一致性先验提升表达精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.14613v1](https://arxiv.org/pdf/2511.14613v1)**

> **作者:** Mohammad Vali Sanian; Arshia Hemmat; Amirhossein Vahidi; Jonas Maaskola; Jimmy Tsz Hang Lee; Stanislaw Makarchuk; Yeliz Demirci; Nana-Jane Chipampe; Omer Bayraktar; Lassi Paavolainen; Mohammad Lotfollahi
>
> **备注:** 11 pages
>
> **摘要:** A scalable and robust 3D tissue transcriptomics profile can enable a holistic understanding of tissue organization and provide deeper insights into human biology and disease. Most predictive algorithms that infer ST directly from histology treat each section independently and ignore 3D structure, while existing 3D-aware approaches are not generative and do not scale well. We present Holographic Tissue Expression Inpainting and Analysis (HoloTea), a 3D-aware flow-matching framework that imputes spot-level gene expression from H&E while explicitly using information from adjacent sections. Our key idea is to retrieve morphologically corresponding spots on neighboring slides in a shared feature space and fuse this cross section context into a lightweight ControlNet, allowing conditioning to follow anatomical continuity. To better capture the count nature of the data, we introduce a 3D-consistent prior for flow matching that combines a learned zero-inflated negative binomial (ZINB) prior with a spatial-empirical prior constructed from neighboring sections. A global attention block introduces 3D H&E scaling linearly with the number of spots in the slide, enabling training and inference on large 3D ST datasets. Across three spatial transcriptomics datasets spanning different tissue types and resolutions, HoloTea consistently improves 3D expression accuracy and generalization compared to 2D and 3D baselines. We envision HoloTea advancing the creation of accurate 3D virtual tissues, ultimately accelerating biomarker discovery and deepening our understanding of disease.
>
---
#### [new 004] DIR-TIR: Dialog-Iterative Refinement for Text-to-Image Retrieval
- **分类: cs.CV**

- **简介: 该论文提出DIR-TIR框架，用于对话式文本到图像检索任务，旨在通过多轮对话逐步精炼用户意图与图像特征，解决单次查询精度低、交互性差的问题。工作包括设计对话精炼和图像精炼模块，显著提升检索准确率与用户体验。**

- **链接: [https://arxiv.org/pdf/2511.14449v1](https://arxiv.org/pdf/2511.14449v1)**

> **作者:** Zongwei Zhen; Biqing Zeng
>
> **摘要:** This paper addresses the task of interactive, conversational text-to-image retrieval. Our DIR-TIR framework progressively refines the target image search through two specialized modules: the Dialog Refiner Module and the Image Refiner Module. The Dialog Refiner actively queries users to extract essential information and generate increasingly precise descriptions of the target image. Complementarily, the Image Refiner identifies perceptual gaps between generated images and user intentions, strategically reducing the visual-semantic discrepancy. By leveraging multi-turn dialogues, DIR-TIR provides superior controllability and fault tolerance compared to conventional single-query methods, significantly improving target image hit accuracy. Comprehensive experiments across diverse image datasets demonstrate our dialogue-based approach substantially outperforms initial-description-only baselines, while the synergistic module integration achieves both higher retrieval precision and enhanced interactive experience.
>
---
#### [new 005] SMART: Shot-Aware Multimodal Video Moment Retrieval with Audio-Enhanced MLLM
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SMART框架，用于视频片段检索任务，旨在通过融合音频与视觉特征并利用镜头级结构提升细粒度时间定位精度。解决现有方法依赖单一模态和粗粒度时序理解的问题，引入Shot-aware Token Compression和优化提示设计，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.14143v1](https://arxiv.org/pdf/2511.14143v1)**

> **作者:** An Yu; Weiheng Lu; Jian Li; Zhenfei Zhang; Yunhang Shen; Felix X. -F. Ye; Ming-Ching Chang
>
> **摘要:** Video Moment Retrieval is a task in video understanding that aims to localize a specific temporal segment in an untrimmed video based on a natural language query. Despite recent progress in moment retrieval from videos using both traditional techniques and Multimodal Large Language Models (MLLM), most existing methods still rely on coarse temporal understanding and a single visual modality, limiting performance on complex videos. To address this, we introduce \textit{S}hot-aware \textit{M}ultimodal \textit{A}udio-enhanced \textit{R}etrieval of \textit{T}emporal \textit{S}egments (SMART), an MLLM-based framework that integrates audio cues and leverages shot-level temporal structure. SMART enriches multimodal representations by combining audio and visual features while applying \textbf{Shot-aware Token Compression}, which selectively retains high-information tokens within each shot to reduce redundancy and preserve fine-grained temporal details. We also refine prompt design to better utilize audio-visual cues. Evaluations on Charades-STA and QVHighlights show that SMART achieves significant improvements over state-of-the-art methods, including a 1.61\% increase in R1@0.5 and 2.59\% gain in R1@0.7 on Charades-STA.
>
---
#### [new 006] Synergizing Multigrid Algorithms with Vision Transformer: A Novel Approach to Enhance the Seismic Foundation Model
- **分类: cs.CV; cs.AI; math.NA**

- **简介: 论文提出ADATG方法，通过谱分解与分层希尔伯特编码，结合自适应训练策略，提升视觉Transformer在地震数据预训练中对高低频特征的捕捉能力，解决传统ViT忽略地震数据层次结构的问题。**

- **链接: [https://arxiv.org/pdf/2511.13800v1](https://arxiv.org/pdf/2511.13800v1)**

> **作者:** Huiwen Wu; Shuo Zhang; Yi Liu; Hongbin Ye
>
> **摘要:** Due to the emergency and homogenization of Artificial Intelligence (AI) technology development, transformer-based foundation models have revolutionized scientific applications, such as drug discovery, materials research, and astronomy. However, seismic data presents unique characteristics that require specialized processing techniques for pretraining foundation models in seismic contexts with high- and low-frequency features playing crucial roles. Existing vision transformers (ViTs) with sequential tokenization ignore the intrinsic pattern and fail to grasp both the high- and low-frequency seismic information efficiently and effectively. This work introduces a novel adaptive two-grid foundation model training strategy (ADATG) with Hilbert encoding specifically tailored for seismogram data, leveraging the hierarchical structures inherent in seismic data. Specifically, our approach employs spectrum decomposition to separate high- and low-frequency components and utilizes hierarchical Hilbert encoding to represent the data effectively. Moreover, observing the frequency principle observed in ViTs, we propose an adaptive training strategy that initially emphasizes coarse-level information and then progressively refines the model's focus on fine-level features. Our extensive experiments demonstrate the effectiveness and efficiency of our training methods. This research highlights the importance of data encoding and training strategies informed by the distinct characteristics of high- and low-frequency features in seismic images, ultimately contributing to the enhancement of visual seismic foundation models pretraining.
>
---
#### [new 007] Dental3R: Geometry-Aware Pairing for Intraoral 3D Reconstruction from Sparse-View Photographs
- **分类: cs.CV**

- **简介: 该论文提出Dental3R，用于从稀疏口腔照片中重建高保真三维牙齿模型。针对 pose 不稳定和细节丢失问题，采用几何感知配对策略与小波正则化3DGS，提升重建质量和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.14315v1](https://arxiv.org/pdf/2511.14315v1)**

> **作者:** Yiyi Miao; Taoyu Wu; Tong Chen; Ji Jiang; Zhe Tang; Zhengyong Jiang; Angelos Stefanidis; Limin Yu; Jionglong Su
>
> **摘要:** Intraoral 3D reconstruction is fundamental to digital orthodontics, yet conventional methods like intraoral scanning are inaccessible for remote tele-orthodontics, which typically relies on sparse smartphone imagery. While 3D Gaussian Splatting (3DGS) shows promise for novel view synthesis, its application to the standard clinical triad of unposed anterior and bilateral buccal photographs is challenging. The large view baselines, inconsistent illumination, and specular surfaces common in intraoral settings can destabilize simultaneous pose and geometry estimation. Furthermore, sparse-view photometric supervision often induces a frequency bias, leading to over-smoothed reconstructions that lose critical diagnostic details. To address these limitations, we propose \textbf{Dental3R}, a pose-free, graph-guided pipeline for robust, high-fidelity reconstruction from sparse intraoral photographs. Our method first constructs a Geometry-Aware Pairing Strategy (GAPS) to intelligently select a compact subgraph of high-value image pairs. The GAPS focuses on correspondence matching, thereby improving the stability of the geometry initialization and reducing memory usage. Building on the recovered poses and point cloud, we train the 3DGS model with a wavelet-regularized objective. By enforcing band-limited fidelity using a discrete wavelet transform, our approach preserves fine enamel boundaries and interproximal edges while suppressing high-frequency artifacts. We validate our approach on a large-scale dataset of 950 clinical cases and an additional video-based test set of 195 cases. Experimental results demonstrate that Dental3R effectively handles sparse, unposed inputs and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art methods.
>
---
#### [new 008] ManipShield: A Unified Framework for Image Manipulation Detection, Localization and Explanation
- **分类: cs.CV**

- **简介: 论文提出ManipShield框架，解决图像篡改检测、定位与解释难题。基于新构建的ManipBench基准，利用多模态大模型实现统一检测、定位与可解释分析，性能优于现有方法且泛化能力强。**

- **链接: [https://arxiv.org/pdf/2511.14259v1](https://arxiv.org/pdf/2511.14259v1)**

> **作者:** Zitong Xu; Huiyu Duan; Xiaoyu Wang; Zhaolin Cai; Kaiwei Zhang; Qiang Hu; Jing Liu; Xiongkuo Min; Guangtao Zhai
>
> **摘要:** With the rapid advancement of generative models, powerful image editing methods now enable diverse and highly realistic image manipulations that far surpass traditional deepfake techniques, posing new challenges for manipulation detection. Existing image manipulation detection and localization (IMDL) benchmarks suffer from limited content diversity, narrow generative-model coverage, and insufficient interpretability, which hinders the generalization and explanation capabilities of current manipulation detection methods. To address these limitations, we introduce \textbf{ManipBench}, a large-scale benchmark for image manipulation detection and localization focusing on AI-edited images. ManipBench contains over 450K manipulated images produced by 25 state-of-the-art image editing models across 12 manipulation categories, among which 100K images are further annotated with bounding boxes, judgment cues, and textual explanations to support interpretable detection. Building upon ManipBench, we propose \textbf{ManipShield}, an all-in-one model based on a Multimodal Large Language Model (MLLM) that leverages contrastive LoRA fine-tuning and task-specific decoders to achieve unified image manipulation detection, localization, and explanation. Extensive experiments on ManipBench and several public datasets demonstrate that ManipShield achieves state-of-the-art performance and exhibits strong generality to unseen manipulation models. Both ManipBench and ManipShield will be released upon publication.
>
---
#### [new 009] Cranio-ID: Graph-Based Craniofacial Identification via Automatic Landmark Annotation in 2D Multi-View X-rays
- **分类: cs.CV**

- **简介: 论文提出Cranio-ID框架，解决2D多视角X光片与面部图像间颅面特征自动匹配问题。通过YOLO-pose模型自动标注关键点，并用图结构与交叉注意力机制实现跨模态对应，提升法医颅面识别的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2511.14411v1](https://arxiv.org/pdf/2511.14411v1)**

> **作者:** Ravi Shankar Prasad; Nandani Sharma; Dinesh Singh
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** In forensic craniofacial identification and in many biomedical applications, craniometric landmarks are important. Traditional methods for locating landmarks are time-consuming and require specialized knowledge and expertise. Current methods utilize superimposition and deep learning-based methods that employ automatic annotation of landmarks. However, these methods are not reliable due to insufficient large-scale validation studies. In this paper, we proposed a novel framework Cranio-ID: First, an automatic annotation of landmarks on 2D skulls (which are X-ray scans of faces) with their respective optical images using our trained YOLO-pose models. Second, cross-modal matching by formulating these landmarks into graph representations and then finding semantic correspondence between graphs of these two modalities using cross-attention and optimal transport framework. Our proposed framework is validated on the S2F and CUHK datasets (CUHK dataset resembles with S2F dataset). Extensive experiments have been conducted to evaluate the performance of our proposed framework, which demonstrates significant improvements in both reliability and accuracy, as well as its effectiveness in cross-domain skull-to-face and sketch-to-face matching in forensic science.
>
---
#### [new 010] FAPE-IR: Frequency-Aware Planning and Execution Framework for All-in-One Image Restoration
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FAPE-IR框架，用于统一图像修复任务。针对多退化场景下模型适应性差的问题，利用冻结的多模态大语言模型生成频率感知修复计划，并通过LoRA-MoE扩散执行器动态选择高频或低频专家，结合对抗训练与频率正则化提升修复质量与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.14099v1](https://arxiv.org/pdf/2511.14099v1)**

> **作者:** Jingren Liu; Shuning Xu; Qirui Yang; Yun Wang; Xiangyu Chen; Zhong Ji
>
> **摘要:** All-in-One Image Restoration (AIO-IR) aims to develop a unified model that can handle multiple degradations under complex conditions. However, existing methods often rely on task-specific designs or latent routing strategies, making it hard to adapt to real-world scenarios with various degradations. We propose FAPE-IR, a Frequency-Aware Planning and Execution framework for image restoration. It uses a frozen Multimodal Large Language Model (MLLM) as a planner to analyze degraded images and generate concise, frequency-aware restoration plans. These plans guide a LoRA-based Mixture-of-Experts (LoRA-MoE) module within a diffusion-based executor, which dynamically selects high- or low-frequency experts, complemented by frequency features of the input image. To further improve restoration quality and reduce artifacts, we introduce adversarial training and a frequency regularization loss. By coupling semantic planning with frequency-based restoration, FAPE-IR offers a unified and interpretable solution for all-in-one image restoration. Extensive experiments show that FAPE-IR achieves state-of-the-art performance across seven restoration tasks and exhibits strong zero-shot generalization under mixed degradations.
>
---
#### [new 011] Diffusion As Self-Distillation: End-to-End Latent Diffusion In One Model
- **分类: cs.CV**

- **简介: 论文提出Diffusion as Self-Distillation（DSD），将编码器、解码器和扩散网络统一为单个可端到端训练的模型，解决传统三阶段架构效率低、性能差的问题。通过稳定潜在空间避免“潜变量坍塌”，在ImageNet上实现高生成质量。**

- **链接: [https://arxiv.org/pdf/2511.14716v1](https://arxiv.org/pdf/2511.14716v1)**

> **作者:** Xiyuan Wang; Muhan Zhang
>
> **备注:** Tech Report. 10 pages
>
> **摘要:** Standard Latent Diffusion Models rely on a complex, three-part architecture consisting of a separate encoder, decoder, and diffusion network, which are trained in multiple stages. This modular design is computationally inefficient, leads to suboptimal performance, and prevents the unification of diffusion with the single-network architectures common in vision foundation models. Our goal is to unify these three components into a single, end-to-end trainable network. We first demonstrate that a naive joint training approach fails catastrophically due to ``latent collapse'', where the diffusion training objective interferes with the network's ability to learn a good latent representation. We identify the root causes of this instability by drawing a novel analogy between diffusion and self-distillation based unsupervised learning method. Based on this insight, we propose Diffusion as Self-Distillation (DSD), a new framework with key modifications to the training objective that stabilize the latent space. This approach enables, for the first time, the stable end-to-end training of a single network that simultaneously learns to encode, decode, and perform diffusion. DSD achieves outstanding performance on the ImageNet $256\times 256$ conditional generation task: FID=13.44/6.38/4.25 with only 42M/118M/205M parameters and 50 training epochs on ImageNet, without using classifier-free-guidance.
>
---
#### [new 012] Segmenting Collision Sound Sources in Egocentric Videos
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 论文提出碰撞声音源分割（CS3）任务，旨在根据音频在第一人称视频中定位引发碰撞声的物体。针对视觉 clutter 和短时交互挑战，作者提出弱监督方法，结合CLIP、SAM2及手部物体线索，显著优于基线。**

- **链接: [https://arxiv.org/pdf/2511.13863v1](https://arxiv.org/pdf/2511.13863v1)**

> **作者:** Kranti Kumar Parida; Omar Emara; Hazel Doughty; Dima Damen
>
> **备注:** Under Review. Webpage: https://krantiparida.github.io/projects/cs3.html
>
> **摘要:** Humans excel at multisensory perception and can often recognise object properties from the sound of their interactions. Inspired by this, we propose the novel task of Collision Sound Source Segmentation (CS3), where we aim to segment the objects responsible for a collision sound in visual input (i.e. video frames from the collision clip), conditioned on the audio. This task presents unique challenges. Unlike isolated sound events, a collision sound arises from interactions between two objects, and the acoustic signature of the collision depends on both. We focus on egocentric video, where sounds are often clear, but the visual scene is cluttered, objects are small, and interactions are brief. To address these challenges, we propose a weakly-supervised method for audio-conditioned segmentation, utilising foundation models (CLIP and SAM2). We also incorporate egocentric cues, i.e. objects in hands, to find acting objects that can potentially be collision sound sources. Our approach outperforms competitive baselines by $3\times$ and $4.7\times$ in mIoU on two benchmarks we introduce for the CS3 task: EPIC-CS3 and Ego4D-CS3.
>
---
#### [new 013] Enhancing End-to-End Autonomous Driving with Risk Semantic Distillaion from VLM
- **分类: cs.CV; cs.RO**

- **简介: 论文提出Risk Semantic Distillation（RSD）框架，用于增强端到端自动驾驶系统的泛化能力。通过从视觉语言模型中蒸馏风险语义信息到BEV特征，提升对复杂场景中风险物体的感知与规划能力，解决现有方法在未见场景下表现不佳的问题。**

- **链接: [https://arxiv.org/pdf/2511.14499v1](https://arxiv.org/pdf/2511.14499v1)**

> **作者:** Jack Qin; Zhitao Wang; Yinan Zheng; Keyu Chen; Yang Zhou; Yuanxin Zhong; Siyuan Cheng
>
> **摘要:** The autonomous driving (AD) system has exhibited remarkable performance in complex driving scenarios. However, generalization is still a key limitation for the current system, which refers to the ability to handle unseen scenarios or unfamiliar sensor configurations.Related works have explored the use of Vision-Language Models (VLMs) to address few-shot or zero-shot tasks. While promising, these methods introduce a new challenge: the emergence of a hybrid AD system, where two distinct systems are used to plan a trajectory, leading to potential inconsistencies. Alternative research directions have explored Vision-Language-Action (VLA) frameworks that generate control actions from VLM directly. However, these end-to-end solutions demonstrate prohibitive computational demands. To overcome these challenges, we introduce Risk Semantic Distillation (RSD), a novel framework that leverages VLMs to enhance the training of End-to-End (E2E) AD backbones. By providing risk attention for key objects, RSD addresses the issue of generalization. Specifically, we introduce RiskHead, a plug-in module that distills causal risk estimates from Vision-Language Models into Bird's-Eye-View (BEV) features, yielding interpretable risk-attention maps.This approach allows BEV features to learn richer and more nuanced risk attention representations, which directly enhance the model's ability to handle spatial boundaries and risky objects.By focusing on risk attention, RSD aligns better with human-like driving behavior, which is essential to navigate in complex and dynamic environments. Our experiments on the Bench2Drive benchmark demonstrate the effectiveness of RSD in managing complex and unpredictable driving conditions. Due to the enhanced BEV representations enabled by RSD, we observed a significant improvement in both perception and planning capabilities.
>
---
#### [new 014] BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition
- **分类: cs.CV**

- **简介: 该论文针对长尾识别任务，提出BCE3S方法，通过三重协同学习优化特征紧凑性和可分性，缓解类别不平衡问题，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.14097v1](https://arxiv.org/pdf/2511.14097v1)**

> **作者:** Weijia Fan; Qiufu Li; Jiajun Wen; Xiaoyang Peng
>
> **备注:** [AAAI-2026] code: https://github.com/wakinghours-github/BCE3S
>
> **摘要:** For long-tailed recognition (LTR) tasks, high intra-class compactness and inter-class separability in both head and tail classes, as well as balanced separability among all the classifier vectors, are preferred. The existing LTR methods based on cross-entropy (CE) loss not only struggle to learn features with desirable properties but also couple imbalanced classifier vectors in the denominator of its Softmax, amplifying the imbalance effects in LTR. In this paper, for the LTR, we propose a binary cross-entropy (BCE)-based tripartite synergistic learning, termed BCE3S, which consists of three components: (1) BCE-based joint learning optimizes both the classifier and sample features, which achieves better compactness and separability among features than the CE-based joint learning, by decoupling the metrics between feature and the imbalanced classifier vectors in multiple Sigmoid; (2) BCE-based contrastive learning further improves the intra-class compactness of features; (3) BCE-based uniform learning balances the separability among classifier vectors and interactively enhances the feature properties by combining with the joint learning. The extensive experiments show that the LTR model trained by BCE3S not only achieves higher compactness and separability among sample features, but also balances the classifier's separability, achieving SOTA performance on various long-tailed datasets such as CIFAR10-LT, CIFAR100-LT, ImageNet-LT, and iNaturalist2018.
>
---
#### [new 015] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对少样本视频目标检测（FSVOD）任务，解决因标注数据少导致的时序一致性差与泛化能力弱的问题。提出一种基于视觉Transformer的对象感知时序建模方法，通过筛选高置信度特征实现高效传播，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2511.13784v1](https://arxiv.org/pdf/2511.13784v1)**

> **作者:** Yogesh Kumar; Anand Mishra
>
> **备注:** Accepted at AAAI 2026 Main Track
>
> **摘要:** Few-shot Video Object Detection (FSVOD) addresses the challenge of detecting novel objects in videos with limited labeled examples, overcoming the constraints of traditional detection methods that require extensive training data. This task presents key challenges, including maintaining temporal consistency across frames affected by occlusion and appearance variations, and achieving novel object generalization without relying on complex region proposals, which are often computationally expensive and require task-specific training. Our novel object-aware temporal modeling approach addresses these challenges by incorporating a filtering mechanism that selectively propagates high-confidence object features across frames. This enables efficient feature progression, reduces noise accumulation, and enhances detection accuracy in a few-shot setting. By utilizing few-shot trained detection and classification heads with focused feature propagation, we achieve robust temporal consistency without depending on explicit object tube proposals. Our approach achieves performance gains, with AP improvements of 3.7% (FSVOD-500), 5.3% (FSYTV-40), 4.3% (VidOR), and 4.5 (VidVRD) in the 5-shot setting. Further results demonstrate improvements in 1-shot, 3-shot, and 10-shot configurations. We make the code public at: https://github.com/yogesh-iitj/fs-video-vit
>
---
#### [new 016] RSPose: Ranking Based Losses for Human Pose Estimation
- **分类: cs.CV**

- **简介: 该论文针对人体姿态估计任务，解决heatmap损失函数与评估指标不一致的问题。提出基于排序的损失函数RSPose，提升关键点定位精度和置信度相关性，显著改善mAP性能。**

- **链接: [https://arxiv.org/pdf/2511.13857v1](https://arxiv.org/pdf/2511.13857v1)**

> **作者:** Muhammed Can Keles; Bedrettin Cetinkaya; Sinan Kalkan; Emre Akbas
>
> **摘要:** While heatmap-based human pose estimation methods have shown strong performance, they suffer from three main problems: (P1) "Commonly used Mean Squared Error (MSE)" Loss may not always improve joint localization because it penalizes all pixel deviations equally, without focusing explicitly on sharpening and correctly localizing the peak corresponding to the joint; (P2) heatmaps are spatially and class-wise imbalanced; and, (P3) there is a discrepancy between the evaluation metric (i.e., mAP) and the loss functions. We propose ranking-based losses to address these issues. Both theoretically and empirically, we show that our proposed losses are superior to commonly used heatmap losses (MSE, KL-Divergence). Our losses considerably increase the correlation between confidence scores and localization qualities, which is desirable because higher correlation leads to more accurate instance selection during Non-Maximum Suppression (NMS) and better Average Precision (mAP) performance. We refer to the models trained with our losses as RSPose. We show the effectiveness of RSPose across two different modes: one-dimensional and two-dimensional heatmaps, on three different datasets (COCO, CrowdPose, MPII). To the best of our knowledge, we are the first to propose losses that align with the evaluation metric (mAP) for human pose estimation. RSPose outperforms the previous state of the art on the COCO-val set and achieves an mAP score of 79.9 with ViTPose-H, a vision transformer model for human pose estimation. We also improve SimCC Resnet-50, a coordinate classification-based pose estimation method, by 1.5 AP on the COCO-val set, achieving 73.6 AP.
>
---
#### [new 017] Single Tensor Cell Segmentation using Scalar Field Representations
- **分类: cs.CV; cs.LG**

- **简介: 论文提出基于标量场表示的单张量细胞分割方法，通过求解泊松方程或热扩散方程构建连续标量场，利用分水岭法实现细胞实例分割，无需正则化即可鲁棒处理噪声并保持边界清晰，简化模型结构，提升效率，适用于边缘计算。**

- **链接: [https://arxiv.org/pdf/2511.13947v1](https://arxiv.org/pdf/2511.13947v1)**

> **作者:** Kevin I. Ruiz Vargas; Gabriel G. Galdino; Tsang Ing Ren; Alexandre L. Cunha
>
> **备注:** Submitted to IEEE ISBI 2026
>
> **摘要:** We investigate image segmentation of cells under the lens of scalar fields. Our goal is to learn a continuous scalar field on image domains such that its segmentation produces robust instances for cells present in images. This field is a function parameterized by the trained network, and its segmentation is realized by the watershed method. The fields we experiment with are solutions to the Poisson partial differential equation and a diffusion mimicking the steady-state solution of the heat equation. These solutions are obtained by minimizing just the field residuals, no regularization is needed, providing a robust regression capable of diminishing the adverse impacts of outliers in the training data and allowing for sharp cell boundaries. A single tensor is all that is needed to train a \unet\ thus simplifying implementation, lowering training and inference times, hence reducing energy consumption, and requiring a small memory footprint, all attractive features in edge computing. We present competitive results on public datasets from the literature and show that our novel, simple yet geometrically insightful approach can achieve excellent cell segmentation results.
>
---
#### [new 018] CCSD: Cross-Modal Compositional Self-Distillation for Robust Brain Tumor Segmentation with Missing Modalities
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对脑肿瘤分割任务，解决多模态MRI中常见模态缺失问题。提出CCSD框架，通过共享-特定编码器和两种自蒸馏策略，提升模型在任意模态组合下的鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.14599v1](https://arxiv.org/pdf/2511.14599v1)**

> **作者:** Dongqing Xie; Yonghuang Wu; Zisheng Ai; Jun Min; Zhencun Jiang; Shaojin Geng; Lei Wang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** The accurate segmentation of brain tumors from multi-modal MRI is critical for clinical diagnosis and treatment planning. While integrating complementary information from various MRI sequences is a common practice, the frequent absence of one or more modalities in real-world clinical settings poses a significant challenge, severely compromising the performance and generalizability of deep learning-based segmentation models. To address this challenge, we propose a novel Cross-Modal Compositional Self-Distillation (CCSD) framework that can flexibly handle arbitrary combinations of input modalities. CCSD adopts a shared-specific encoder-decoder architecture and incorporates two self-distillation strategies: (i) a hierarchical modality self-distillation mechanism that transfers knowledge across modality hierarchies to reduce semantic discrepancies, and (ii) a progressive modality combination distillation approach that enhances robustness to missing modalities by simulating gradual modality dropout during training. Extensive experiments on public brain tumor segmentation benchmarks demonstrate that CCSD achieves state-of-the-art performance across various missing-modality scenarios, with strong generalization and stability.
>
---
#### [new 019] Step by Step Network
- **分类: cs.CV**

- **简介: 该论文提出Step by Step Network（StepsNet），旨在解决深层神经网络因shortcut degradation和宽度受限导致性能提升停滞的问题。通过分通道逐步学习机制，提升模型深度利用效率，在图像分类、目标检测等任务上优于传统残差网络。**

- **链接: [https://arxiv.org/pdf/2511.14329v1](https://arxiv.org/pdf/2511.14329v1)**

> **作者:** Dongchen Han; Tianzhu Ye; Zhuofan Xia; Kaiyi Chen; Yulin Wang; Hanting Chen; Gao Huang
>
> **摘要:** Scaling up network depth is a fundamental pursuit in neural architecture design, as theory suggests that deeper models offer exponentially greater capability. Benefiting from the residual connections, modern neural networks can scale up to more than one hundred layers and enjoy wide success. However, as networks continue to deepen, current architectures often struggle to realize their theoretical capacity improvements, calling for more advanced designs to further unleash the potential of deeper networks. In this paper, we identify two key barriers that obstruct residual models from scaling deeper: shortcut degradation and limited width. Shortcut degradation hinders deep-layer learning, while the inherent depth-width trade-off imposes limited width. To mitigate these issues, we propose a generalized residual architecture dubbed Step by Step Network (StepsNet) to bridge the gap between theoretical potential and practical performance of deep models. Specifically, we separate features along the channel dimension and let the model learn progressively via stacking blocks with increasing width. The resulting method mitigates the two identified problems and serves as a versatile macro design applicable to various models. Extensive experiments show that our method consistently outperforms residual models across diverse tasks, including image classification, object detection, semantic segmentation, and language modeling. These results position StepsNet as a superior generalization of the widely adopted residual architecture.
>
---
#### [new 020] A Neural Field-Based Approach for View Computation & Data Exploration in 3D Urban Environments
- **分类: cs.CV; cs.GR**

- **简介: 论文提出基于神经场的视图计算与数据探索方法，解决3D城市环境中因遮挡和手动调整视角导致的大规模探索效率低问题。通过构建隐式环境表示，支持快速正向和反向查询，提升可见性分析、日照评估等城市分析任务的效果。**

- **链接: [https://arxiv.org/pdf/2511.14742v1](https://arxiv.org/pdf/2511.14742v1)**

> **作者:** Stefan Cobeli; Kazi Shahrukh Omar; Rodrigo Valença; Nivan Ferreira; Fabio Miranda
>
> **备注:** Accepted at IEEE Transactions on Visualization and Computer Graphics. Code and data are publicly available at https://urbantk.org/neural-3d
>
> **摘要:** Despite the growing availability of 3D urban datasets, extracting insights remains challenging due to computational bottlenecks and the complexity of interacting with data. In fact, the intricate geometry of 3D urban environments results in high degrees of occlusion and requires extensive manual viewpoint adjustments that make large-scale exploration inefficient. To address this, we propose a view-based approach for 3D data exploration, where a vector field encodes views from the environment. To support this approach, we introduce a neural field-based method that constructs an efficient implicit representation of 3D environments. This representation enables both faster direct queries, which consist of the computation of view assessment indices, and inverse queries, which help avoid occlusion and facilitate the search for views that match desired data patterns. Our approach supports key urban analysis tasks such as visibility assessments, solar exposure evaluation, and assessing the visual impact of new developments. We validate our method through quantitative experiments, case studies informed by real-world urban challenges, and feedback from domain experts. Results show its effectiveness in finding desirable viewpoints, analyzing building facade visibility, and evaluating views from outdoor spaces. Code and data are publicly available at https://urbantk.org/neural-3d.
>
---
#### [new 021] Breaking the Passive Learning Trap: An Active Perception Strategy for Human Motion Prediction
- **分类: cs.CV**

- **简介: 论文提出主动感知策略（APS）解决人体运动预测中依赖隐式建模导致的信息冗余问题。通过商空间表示解耦运动几何与坐标冗余，并引入辅助学习机制增强时空建模，提升预测精度。**

- **链接: [https://arxiv.org/pdf/2511.14237v1](https://arxiv.org/pdf/2511.14237v1)**

> **作者:** Juncheng Hu; Zijian Zhang; Zeyu Wang; Guoyu Wang; Yingji Li; Kedi Lyu
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Forecasting 3D human motion is an important embodiment of fine-grained understanding and cognition of human behavior by artificial agents. Current approaches excessively rely on implicit network modeling of spatiotemporal relationships and motion characteristics, falling into the passive learning trap that results in redundant and monotonous 3D coordinate information acquisition while lacking actively guided explicit learning mechanisms. To overcome these issues, we propose an Active Perceptual Strategy (APS) for human motion prediction, leveraging quotient space representations to explicitly encode motion properties while introducing auxiliary learning objectives to strengthen spatio-temporal modeling. Specifically, we first design a data perception module that projects poses into the quotient space, decoupling motion geometry from coordinate redundancy. By jointly encoding tangent vectors and Grassmann projections, this module simultaneously achieves geometric dimension reduction, semantic decoupling, and dynamic constraint enforcement for effective motion pose characterization. Furthermore, we introduce a network perception module that actively learns spatio-temporal dependencies through restorative learning. This module deliberately masks specific joints or injects noise to construct auxiliary supervision signals. A dedicated auxiliary learning network is designed to actively adapt and learn from perturbed information. Notably, APS is model agnostic and can be integrated with different prediction models to enhance active perceptual. The experimental results demonstrate that our method achieves the new state-of-the-art, outperforming existing methods by large margins: 16.3% on H3.6M, 13.9% on CMU Mocap, and 10.1% on 3DPW.
>
---
#### [new 022] Multi-Scale Correlation-Aware Transformer for Maritime Vessel Re-Identification
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对海上船舶重识别任务，解决因船体外观变化大和局部缺失导致的异常样本干扰问题。提出MCFormer网络，通过全局与局部相关性模块协同建模多尺度特征，提升识别鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.14203v1](https://arxiv.org/pdf/2511.14203v1)**

> **作者:** Yunhe Liu
>
> **摘要:** Maritime vessel re-identification (Re-ID) plays a crucial role in advancing maritime monitoring and intelligent situational awareness systems. However, some existing vessel Re-ID methods are directly adapted from pedestrian-focused algorithms, making them ill-suited for mitigating the unique problems present in vessel images, particularly the greater intra-identity variations and more severe missing of local parts, which lead to the emergence of outlier samples within the same identity. To address these challenges, we propose the Multi-scale Correlation-aware Transformer Network (MCFormer), which explicitly models multi-scale correlations across the entire input set to suppress the adverse effects of outlier samples with intra-identity variations or local missing, incorporating two novel modules, the Global Correlation Module (GCM), and the Local Correlation Module (LCM). Specifically, GCM constructs a global similarity affinity matrix across all input images to model global correlations through feature aggregation based on inter-image consistency, rather than solely learning features from individual images as in most existing approaches. Simultaneously, LCM mines and aligns local features of positive samples with contextual similarity to extract local correlations by maintaining a dynamic memory bank, effectively compensating for missing or occluded regions in individual images. To further enhance feature robustness, MCFormer integrates global and local features that have been respectively correlated across multiple scales, effectively capturing latent relationships among image features. Experiments on three benchmarks demonstrate that MCFormer achieves state-of-the-art performance.
>
---
#### [new 023] Gaussian Splatting-based Low-Rank Tensor Representation for Multi-Dimensional Image Recovery
- **分类: cs.CV**

- **简介: 论文提出GSLR框架用于多维图像恢复，解决传统t-SVD方法在捕捉局部高频信息上的不足。通过2D和1D高斯点绘技术构建低秩张量表示，提升对局部细节的表达能力，并设计无监督恢复模型验证效果。**

- **链接: [https://arxiv.org/pdf/2511.14270v1](https://arxiv.org/pdf/2511.14270v1)**

> **作者:** Yiming Zeng; Xi-Le Zhao; Wei-Hao Wu; Teng-Yu Ji; Chao Wang
>
> **摘要:** Tensor singular value decomposition (t-SVD) is a promising tool for multi-dimensional image representation, which decomposes a multi-dimensional image into a latent tensor and an accompanying transform matrix. However, two critical limitations of t-SVD methods persist: (1) the approximation of the latent tensor (e.g., tensor factorizations) is coarse and fails to accurately capture spatial local high-frequency information; (2) The transform matrix is composed of fixed basis atoms (e.g., complex exponential atoms in DFT and cosine atoms in DCT) and cannot precisely capture local high-frequency information along the mode-3 fibers. To address these two limitations, we propose a Gaussian Splatting-based Low-rank tensor Representation (GSLR) framework, which compactly and continuously represents multi-dimensional images. Specifically, we leverage tailored 2D Gaussian splatting and 1D Gaussian splatting to generate the latent tensor and transform matrix, respectively. The 2D and 1D Gaussian splatting are indispensable and complementary under this representation framework, which enjoys a powerful representation capability, especially for local high-frequency information. To evaluate the representation ability of the proposed GSLR, we develop an unsupervised GSLR-based multi-dimensional image recovery model. Extensive experiments on multi-dimensional image recovery demonstrate that GSLR consistently outperforms state-of-the-art methods, particularly in capturing local high-frequency information.
>
---
#### [new 024] ARC-Chapter: Structuring Hour-Long Videos into Navigable Chapters and Hierarchical Summaries
- **分类: cs.CV**

- **简介: 论文提出ARC-Chapter，解决长视频（如讲座、纪录片）自动分章难题。通过构建百万级双语标注数据集和设计新评估指标GRACE，实现更精准的章节划分与层级摘要，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.14349v1](https://arxiv.org/pdf/2511.14349v1)**

> **作者:** Junfu Pu; Teng Wang; Yixiao Ge; Yuying Ge; Chen Li; Ying Shan
>
> **备注:** Project Page: https://arcchapter.github.io/index_en.html
>
> **摘要:** The proliferation of hour-long videos (e.g., lectures, podcasts, documentaries) has intensified demand for efficient content structuring. However, existing approaches are constrained by small-scale training with annotations that are typical short and coarse, restricting generalization to nuanced transitions in long videos. We introduce ARC-Chapter, the first large-scale video chaptering model trained on over million-level long video chapters, featuring bilingual, temporally grounded, and hierarchical chapter annotations. To achieve this goal, we curated a bilingual English-Chinese chapter dataset via a structured pipeline that unifies ASR transcripts, scene texts, visual captions into multi-level annotations, from short title to long summaries. We demonstrate clear performance improvements with data scaling, both in data volume and label intensity. Moreover, we design a new evaluation metric termed GRACE, which incorporates many-to-one segment overlaps and semantic similarity, better reflecting real-world chaptering flexibility. Extensive experiments demonstrate that ARC-Chapter establishes a new state-of-the-art by a significant margin, outperforming the previous best by 14.0% in F1 score and 11.3% in SODA score. Moreover, ARC-Chapter shows excellent transferability, improving the state-of-the-art on downstream tasks like dense video captioning on YouCook2.
>
---
#### [new 025] Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection
- **分类: cs.CV**

- **简介: 论文提出JFD3框架，解决红外无人机目标检测中运动模糊导致特征不清晰的问题。通过双分支结构和特征恢复机制，增强检测特征表示，提升鲁棒性与实时性。**

- **链接: [https://arxiv.org/pdf/2511.14371v1](https://arxiv.org/pdf/2511.14371v1)**

> **作者:** Xiaolin Wang; Houzhang Fang; Qingshan Li; Lu Wang; Yi Chang; Luxin Yan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Infrared unmanned aerial vehicle (UAV) target images often suffer from motion blur degradation caused by rapid sensor movement, significantly reducing contrast between target and background. Generally, detection performance heavily depends on the discriminative feature representation between target and background. Existing methods typically treat deblurring as a preprocessing step focused on visual quality, while neglecting the enhancement of task-relevant features crucial for detection. Improving feature representation for detection under blur conditions remains challenging. In this paper, we propose a novel Joint Feature-Domain Deblurring and Detection end-to-end framework, dubbed JFD3. We design a dual-branch architecture with shared weights, where the clear branch guides the blurred branch to enhance discriminative feature representation. Specifically, we first introduce a lightweight feature restoration network, where features from the clear branch serve as feature-level supervision to guide the blurred branch, thereby enhancing its distinctive capability for detection. We then propose a frequency structure guidance module that refines the structure prior from the restoration network and integrates it into shallow detection layers to enrich target structural information. Finally, a feature consistency self-supervised loss is imposed between the dual-branch detection backbones, driving the blurred branch to approximate the feature representations of the clear one. Wealso construct a benchmark, named IRBlurUAV, containing 30,000 simulated and 4,118 real infrared UAV target images with diverse motion blur. Extensive experiments on IRBlurUAV demonstrate that JFD3 achieves superior detection performance while maintaining real-time efficiency.
>
---
#### [new 026] Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出UMEG-Net解决少样本精准事件检测问题，通过统一人体骨骼与运动物体关键点构建图结构，并结合多模态知识蒸馏提升性能，在少量标注数据下显著优于基线模型。**

- **链接: [https://arxiv.org/pdf/2511.14186v1](https://arxiv.org/pdf/2511.14186v1)**

> **作者:** Zhaoyu Liu; Kan Jiang; Murong Ma; Zhe Hou; Yun Lin; Jin Song Dong
>
> **备注:** The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Precise event spotting (PES) aims to recognize fine-grained events at exact moments and has become a key component of sports analytics. This task is particularly challenging due to rapid succession, motion blur, and subtle visual differences. Consequently, most existing methods rely on domain-specific, end-to-end training with large labeled datasets and often struggle in few-shot conditions due to their dependence on pixel- or pose-based inputs alone. However, obtaining large labeled datasets is practically hard. We propose a Unified Multi-Entity Graph Network (UMEG-Net) for few-shot PES. UMEG-Net integrates human skeletons and sport-specific object keypoints into a unified graph and features an efficient spatio-temporal extraction module based on advanced GCN and multi-scale temporal shift. To further enhance performance, we employ multimodal distillation to transfer knowledge from keypoint-based graphs to visual representations. Our approach achieves robust performance with limited labeled data and significantly outperforms baseline models in few-shot settings, providing a scalable and effective solution for few-shot PES. Code is publicly available at https://github.com/LZYAndy/UMEG-Net.
>
---
#### [new 027] Vision Large Language Models Are Good Noise Handlers in Engagement Analysis
- **分类: cs.CV**

- **简介: 论文研究视频中参与度识别任务，针对标签主观性和噪声问题，提出利用视觉大语言模型（VLM）优化标注并结合课程学习策略训练模型，显著提升多个基准上的性能。**

- **链接: [https://arxiv.org/pdf/2511.14749v1](https://arxiv.org/pdf/2511.14749v1)**

> **作者:** Alexander Vedernikov; Puneet Kumar; Haoyu Chen; Tapio Seppänen; Xiaobai Li
>
> **摘要:** Engagement recognition in video datasets, unlike traditional image classification tasks, is particularly challenged by subjective labels and noise limiting model performance. To overcome the challenges of subjective and noisy engagement labels, we propose a framework leveraging Vision Large Language Models (VLMs) to refine annotations and guide the training process. Our framework uses a questionnaire to extract behavioral cues and split data into high- and low-reliability subsets. We also introduce a training strategy combining curriculum learning with soft label refinement, gradually incorporating ambiguous samples while adjusting supervision to reflect uncertainty. We demonstrate that classical computer vision models trained on refined high-reliability subsets and enhanced with our curriculum strategy show improvements, highlighting benefits of addressing label subjectivity with VLMs. This method surpasses prior state of the art across engagement benchmarks such as EngageNet (three of six feature settings, maximum improvement of +1.21%), and DREAMS / PAFE with F1 gains of +0.22 / +0.06.
>
---
#### [new 028] Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding
- **分类: cs.CV**

- **简介: 论文针对视觉定位任务中强化学习微调CoT推理性能下降的问题，提出基于课程学习的相对策略优化方法CuRPO，通过复杂度指标逐步训练，提升模型在长文本和复杂场景下的定位准确率。**

- **链接: [https://arxiv.org/pdf/2511.13924v1](https://arxiv.org/pdf/2511.13924v1)**

> **作者:** Qingyang Yan; Guangyao Chen; Yixiong Zou
>
> **备注:** AAAI 2026 (Oral)
>
> **摘要:** Chain-of-Thought (CoT) prompting has recently shown significant promise across various NLP and computer vision tasks by explicitly generating intermediate reasoning steps. However, we find that reinforcement learning (RL)-based fine-tuned CoT reasoning can paradoxically degrade performance in Visual Grounding tasks, particularly as CoT outputs become lengthy or complex. Additionally, our analysis reveals that increased dataset size does not always enhance performance due to varying data complexities. Motivated by these findings, we propose Curriculum-based Relative Policy Optimization (CuRPO), a novel training strategy that leverages CoT length and generalized Intersection over Union (gIoU) rewards as complexity indicators to progressively structure training data from simpler to more challenging examples. Extensive experiments on RefCOCO, RefCOCO+, RefCOCOg, and LISA datasets demonstrate the effectiveness of our approach. CuRPO consistently outperforms existing methods, including Visual-RFT, with notable improvements of up to +12.52 mAP on RefCOCO. Moreover, CuRPO exhibits exceptional efficiency and robustness, delivering strong localization performance even in few-shot learning scenarios, particularly benefiting tasks characterized by ambiguous and intricate textual descriptions.The code is released on https://github.com/qyoung-yan/CuRPO.
>
---
#### [new 029] Clinically-Validated Innovative Mobile Application for Assessing Blinking and Eyelid Movements
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出并验证了Bapp应用，用于客观评估眨眼和眼睑运动。针对传统方法复杂、昂贵且临床适用性差的问题，研究利用Flutter和Google ML Kit开发移动端实时分析工具，并基于45例患者视频数据进行验证，准确率达98.3%，证明其在眼科临床中的可靠性与实用性。**

- **链接: [https://arxiv.org/pdf/2511.14361v1](https://arxiv.org/pdf/2511.14361v1)**

> **作者:** Gustavo Adolpho Bonesso; Carlos Marcelo Gurjão de Godoy; Tammy Hentona Osaki; Midori Hentona Osaki; Bárbara Moreira Ribeiro Trindade dos Santos; Regina Célia Coelho
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Blinking is a vital physiological process that protects and maintains the health of the ocular surface. Objective assessment of eyelid movements remains challenging due to the complexity, cost, and limited clinical applicability of existing tools. This study presents the clinical validation of Bapp (Blink Application), a mobile application developed using the Flutter framework and integrated with Google ML Kit for on-device, real-time analysis of eyelid movements. The validation occurred using 45 videos from real patients, whose blinks were manually annotated by ophthalmology specialists from the Paulista School of Medicine of the Federal University of Sao Paulo (EPM-UNIFESP) to serve as the ground truth. Bapp's performance was evaluated using standard metrics, including Precision, Recall, and F1-Score, with results demonstrating 98.4% precision, 96.9% recall, and an overall accuracy of 98.3%. These outcomes confirm the reliability of Bapp as a portable, accessible, and objective tool for monitoring both normal and abnormal eyelid movements. The application offers a promising alternative to traditional manual blink counting, supporting continuous ocular health monitoring and postoperative evaluation in clinical environments.
>
---
#### [new 030] MRI Plane Orientation Detection using a Context-Aware 2.5D Model
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文解决MRI图像中平面方向识别问题，提出2.5D上下文模型提升准确性，减少误判，并验证其在脑肿瘤检测中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.14021v1](https://arxiv.org/pdf/2511.14021v1)**

> **作者:** SangHyuk Kim; Daniel Haehn; Sumientra Rampersad
>
> **备注:** 5 pages, 5 figures, 2 tables
>
> **摘要:** Humans can easily identify anatomical planes (axial, coronal, and sagittal) on a 2D MRI slice, but automated systems struggle with this task. Missing plane orientation metadata can complicate analysis, increase domain shift when merging heterogeneous datasets, and reduce accuracy of diagnostic classifiers. This study develops a classifier that accurately generates plane orientation metadata. We adopt a 2.5D context-aware model that leverages multi-slice information to avoid ambiguity from isolated slices and enable robust feature learning. We train the 2.5D model on both 3D slice sequences and static 2D images. While our 2D reference model achieves 98.74% accuracy, our 2.5D method raises this to 99.49%, reducing errors by 60%, highlighting the importance of 2.5D context. We validate the utility of our generated metadata in a brain tumor detection task. A gated strategy selectively uses metadata-enhanced predictions based on uncertainty scores, boosting accuracy from 97.0% with an image-only model to 98.0%, reducing misdiagnoses by 33.3%. We integrate our plane orientation model into an interactive web application and provide it open-source.
>
---
#### [new 031] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D表面重建任务，针对稀疏视角下高斯点绘优化易过拟合的问题，提出SparseSurf方法。通过引入立体几何-纹理对齐和伪特征增强的几何一致性约束，提升重建精度与新视角合成质量。**

- **链接: [https://arxiv.org/pdf/2511.14633v1](https://arxiv.org/pdf/2511.14633v1)**

> **作者:** Meiying Gu; Jiawei Zhang; Jiahe Li; Xiaohan Yu; Haonan Luo; Jin Zheng; Xiao Bai
>
> **备注:** Accepted at AAAI 2026. Project page: https://miya-oi.github.io/SparseSurf-project
>
> **摘要:** Recent advances in optimizing Gaussian Splatting for scene geometry have enabled efficient reconstruction of detailed surfaces from images. However, when input views are sparse, such optimization is prone to overfitting, leading to suboptimal reconstruction quality. Existing approaches address this challenge by employing flattened Gaussian primitives to better fit surface geometry, combined with depth regularization to alleviate geometric ambiguities under limited viewpoints. Nevertheless, the increased anisotropy inherent in flattened Gaussians exacerbates overfitting in sparse-view scenarios, hindering accurate surface fitting and degrading novel view synthesis performance. In this paper, we propose \net{}, a method that reconstructs more accurate and detailed surfaces while preserving high-quality novel view rendering. Our key insight is to introduce Stereo Geometry-Texture Alignment, which bridges rendering quality and geometry estimation, thereby jointly enhancing both surface reconstruction and view synthesis. In addition, we present a Pseudo-Feature Enhanced Geometry Consistency that enforces multi-view geometric consistency by incorporating both training and unseen views, effectively mitigating overfitting caused by sparse supervision. Extensive experiments on the DTU, BlendedMVS, and Mip-NeRF360 datasets demonstrate that our method achieves the state-of-the-art performance.
>
---
#### [new 032] FreeSwim: Revisiting Sliding-Window Attention Mechanisms for Training-Free Ultra-High-Resolution Video Generation
- **分类: cs.CV**

- **简介: 论文提出FreeSwim，解决视频生成中高分辨率下Transformer注意力机制计算成本高的问题。通过内向滑动窗口注意力和双路径结构，在无需训练的情况下实现高效、高质量的超高清视频生成。**

- **链接: [https://arxiv.org/pdf/2511.14712v1](https://arxiv.org/pdf/2511.14712v1)**

> **作者:** Yunfeng Wu; Jiayi Song; Zhenxiong Tan; Zihao He; Songhua Liu
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** The quadratic time and memory complexity of the attention mechanism in modern Transformer based video generators makes end-to-end training for ultra high resolution videos prohibitively expensive. Motivated by this limitation, we introduce a training-free approach that leverages video Diffusion Transformers pretrained at their native scale to synthesize higher resolution videos without any additional training or adaptation. At the core of our method lies an inward sliding window attention mechanism, which originates from a key observation: maintaining each query token's training scale receptive field is crucial for preserving visual fidelity and detail. However, naive local window attention, unfortunately, often leads to repetitive content and exhibits a lack of global coherence in the generated results. To overcome this challenge, we devise a dual-path pipeline that backs up window attention with a novel cross-attention override strategy, enabling the semantic content produced by local attention to be guided by another branch with a full receptive field and, therefore, ensuring holistic consistency. Furthermore, to improve efficiency, we incorporate a cross-attention caching strategy for this branch to avoid the frequent computation of full 3D attention. Extensive experiments demonstrate that our method delivers ultra-high-resolution videos with fine-grained visual details and high efficiency in a training-free paradigm. Meanwhile, it achieves superior performance on VBench, even compared to training-based alternatives, with competitive or improved efficiency. Codes are available at: https://github.com/WillWu111/FreeSwim
>
---
#### [new 033] Learning Subglacial Bed Topography from Sparse Radar with Physics-Guided Residuals
- **分类: cs.CV**

- **简介: 论文提出物理引导的残差学习框架，解决冰下地形重建中雷达数据稀疏问题。通过融合物理约束与数据驱动方法，提升床面高程预测精度与结构一致性，适用于冰盖模型和跨区域推广。**

- **链接: [https://arxiv.org/pdf/2511.14473v1](https://arxiv.org/pdf/2511.14473v1)**

> **作者:** Bayu Adhi Tama; Jianwu Wang; Vandana Janeja; Mostafa Cham
>
> **摘要:** Accurate subglacial bed topography is essential for ice sheet modeling, yet radar observations are sparse and uneven. We propose a physics-guided residual learning framework that predicts bed thickness residuals over a BedMachine prior and reconstructs bed from the observed surface. A DeepLabV3+ decoder over a standard encoder (e.g.,ResNet-50) is trained with lightweight physics and data terms: multi-scale mass conservation, flow-aligned total variation, Laplacian damping, non-negativity of thickness, a ramped prior-consistency term, and a masked Huber fit to radar picks modulated by a confidence map. To measure real-world generalization, we adopt leakage-safe blockwise hold-outs (vertical/horizontal) with safety buffers and report metrics only on held-out cores. Across two Greenland sub-regions, our approach achieves strong test-core accuracy and high structural fidelity, outperforming U-Net, Attention U-Net, FPN, and a plain CNN. The residual-over-prior design, combined with physics, yields spatially coherent, physically plausible beds suitable for operational mapping under domain shift.
>
---
#### [new 034] SLAM-AGS: Slide-Label Aware Multi-Task Pretraining Using Adaptive Gradient Surgery in Computational Cytology
- **分类: cs.CV**

- **简介: 该论文针对计算细胞学中标签不可靠、罕见事件检测难的问题，提出SLAM-AGS框架，通过滑片标签感知的多任务预训练与自适应梯度手术稳定学习，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2511.14639v1](https://arxiv.org/pdf/2511.14639v1)**

> **作者:** Marco Acerbis; Swarnadip Chatterjee; Christophe Avenel; Joakim Lindblad
>
> **备注:** 5 pages, 2 figures, Submitted to ISBI2026
>
> **摘要:** Computational cytology faces two major challenges: i) instance-level labels are unreliable and prohibitively costly to obtain, ii) witness rates are extremely low. We propose SLAM-AGS, a Slide-Label-Aware Multitask pretraining framework that jointly optimizes (i) a weakly supervised similarity objective on slide-negative patches and (ii) a self-supervised contrastive objective on slide-positive patches, yielding stronger performance on downstream tasks. To stabilize learning, we apply Adaptive Gradient Surgery to tackle conflicting task gradients and prevent model collapse. We integrate the pretrained encoder into an attention-based Multiple Instance Learning aggregator for bag-level prediction and attention-guided retrieval of the most abnormal instances in a bag. On a publicly available bone-marrow cytology dataset, with simulated witness rates from 10% down to 0.5%, SLAM-AGS improves bag-level F1-Score and Top 400 positive cell retrieval over other pretraining methods, with the largest gains at low witness rates, showing that resolving gradient interference enables stable pretraining and better performance on downstream tasks. To facilitate reproducibility, we share our complete implementation and evaluation framework as open source: https://github.com/Ace95/SLAM-AGS.
>
---
#### [new 035] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出O3SLM模型，解决LVLMs难以理解手绘草图的问题。通过构建大规模图像-草图-指令三元组数据集，训练出能高效处理草图任务的视觉语言模型，在对象定位、计数、检索和视觉问答等任务上达到最优效果。**

- **链接: [https://arxiv.org/pdf/2511.14368v1](https://arxiv.org/pdf/2511.14368v1)**

> **作者:** Rishi Gupta; Mukilan Karuppasamy; Shyam Marjit; Aditay Tripathi; Anirban Chakraborty
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** While Large Vision Language Models (LVLMs) are increasingly deployed in real-world applications, their ability to interpret abstract visual inputs remains limited. Specifically, they struggle to comprehend hand-drawn sketches, a modality that offers an intuitive means of expressing concepts that are difficult to describe textually. We identify the primary bottleneck as the absence of a large-scale dataset that jointly models sketches, photorealistic images, and corresponding natural language instructions. To address this, we present two key contributions: (1) a new, large-scale dataset of image-sketch-instruction triplets designed to facilitate both pretraining and instruction tuning, and (2) O3SLM, an LVLM trained on this dataset. Comprehensive evaluations on multiple sketch-based tasks: (a) object localization, (b) counting, (c) image retrieval i.e., (SBIR and fine-grained SBIR), and (d) visual question answering (VQA); while incorporating the three existing sketch datasets, namely QuickDraw!, Sketchy, and Tu Berlin, along with our generated SketchVCL dataset, show that O3SLM achieves state-of-the-art performance, substantially outperforming existing LVLMs in sketch comprehension and reasoning.
>
---
#### [new 036] Impact of Image Resolution on Age Estimation with DeepFace and InsightFace
- **分类: cs.CV; cs.AI**

- **简介: 论文研究图像分辨率对深度学习年龄估计的影响，任务为自动年龄估算。通过在7种分辨率下测试DeepFace和InsightFace模型，发现224x224像素时准确率最高，低或过高分辨率均降低性能，且InsightFace更快。**

- **链接: [https://arxiv.org/pdf/2511.14689v1](https://arxiv.org/pdf/2511.14689v1)**

> **作者:** Shiyar Jamo
>
> **备注:** 6 pages, 7 figures, 7 tables. Evaluation of DeepFace and InsightFace age estimation across seven image resolutions (64 to 1080 px)
>
> **摘要:** Automatic age estimation is widely used for age verification, where input images often vary considerably in resolution. This study evaluates the effect of image resolution on age estimation accuracy using DeepFace and InsightFace. A total of 1000 images from the IMDB-Clean dataset were processed in seven resolutions, resulting in 7000 test samples. Performance was evaluated using Mean Absolute Error (MAE), Standard Deviation (SD), and Median Absolute Error (MedAE). Based on this study, we conclude that input image resolution has a clear and consistent impact on the accuracy of age estimation in both DeepFace and InsightFace. Both frameworks achieve optimal performance at 224x224 pixels, with an MAE of 10.83 years (DeepFace) and 7.46 years (InsightFace). At low resolutions, MAE increases substantially, while very high resolutions also degrade accuracy. InsightFace is consistently faster than DeepFace across all resolutions.
>
---
#### [new 037] Wave-Former: Through-Occlusion 3D Reconstruction via Wireless Shape Completion
- **分类: cs.CV**

- **简介: 论文提出Wave-Former，用于完全遮挡物体的高精度3D重建任务。针对毫米波信号噪声大、覆盖有限的问题，设计三阶段物理感知模型，结合Transformer和熵引导选择，实现从无线信号到完整几何形状的推断，仅用合成数据训练即在真实场景中表现优异。**

- **链接: [https://arxiv.org/pdf/2511.14152v1](https://arxiv.org/pdf/2511.14152v1)**

> **作者:** Laura Dodds; Maisy Lam; Waleed Akbar; Yibo Cheng; Fadel Adib
>
> **摘要:** We present Wave-Former, a novel method capable of high-accuracy 3D shape reconstruction for completely occluded, diverse, everyday objects. This capability can open new applications spanning robotics, augmented reality, and logistics. Our approach leverages millimeter-wave (mmWave) wireless signals, which can penetrate common occlusions and reflect off hidden objects. In contrast to past mmWave reconstruction methods, which suffer from limited coverage and high noise, Wave-Former introduces a physics-aware shape completion model capable of inferring full 3D geometry. At the heart of Wave-Former's design is a novel three-stage pipeline which bridges raw wireless signals with recent advancements in vision-based shape completion by incorporating physical properties of mmWave signals. The pipeline proposes candidate geometric surfaces, employs a transformer-based shape completion model designed specifically for mmWave signals, and finally performs entropy-guided surface selection. This enables Wave-Former to be trained using entirely synthetic point-clouds, while demonstrating impressive generalization to real-world data.In head-to-head comparisons with state-of-the-art baselines, Wave-Former raises recall from 54% to 72% while maintaining a high precision of 85%.
>
---
#### [new 038] IBGS: Image-Based Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于新颖视图合成任务，旨在解决3DGS因低阶球谐函数无法捕捉高频细节和视角依赖效应的问题。提出Image-Based Gaussian Splatting，利用高分辨率源图建模像素颜色残差，提升渲染质量且不增加存储开销。**

- **链接: [https://arxiv.org/pdf/2511.14357v1](https://arxiv.org/pdf/2511.14357v1)**

> **作者:** Hoang Chuong Nguyen; Wei Mao; Jose M. Alvarez; Miaomiao Liu
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently emerged as a fast, high-quality method for novel view synthesis (NVS). However, its use of low-degree spherical harmonics limits its ability to capture spatially varying color and view-dependent effects such as specular highlights. Existing works augment Gaussians with either a global texture map, which struggles with complex scenes, or per-Gaussian texture maps, which introduces high storage overhead. We propose Image-Based Gaussian Splatting, an efficient alternative that leverages high-resolution source images for fine details and view-specific color modeling. Specifically, we model each pixel color as a combination of a base color from standard 3DGS rendering and a learned residual inferred from neighboring training images. This promotes accurate surface alignment and enables rendering images of high-frequency details and accurate view-dependent effects. Experiments on standard NVS benchmarks show that our method significantly outperforms prior Gaussian Splatting approaches in rendering quality, without increasing the storage footprint.
>
---
#### [new 039] Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization
- **分类: cs.CV; cs.LG**

- **简介: 论文针对单目深度估计任务，提出WeSTAR框架，通过弱监督自训练与正则化提升基础模型在未见域的泛化能力，解决下游数据下性能优化问题。**

- **链接: [https://arxiv.org/pdf/2511.14238v1](https://arxiv.org/pdf/2511.14238v1)**

> **作者:** Yan Huang; Yongyi Su; Xin Lin; Le Zhang; Xun Xu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** The emergence of foundation models has substantially advanced zero-shot generalization in monocular depth estimation (MDE), as exemplified by the Depth Anything series. However, given access to some data from downstream tasks, a natural question arises: can the performance of these models be further improved? To this end, we propose WeSTAR, a parameter-efficient framework that performs Weakly supervised Self-Training Adaptation with Regularization, designed to enhance the robustness of MDE foundation models in unseen and diverse domains. We first adopt a dense self-training objective as the primary source of structural self-supervision. To further improve robustness, we introduce semantically-aware hierarchical normalization, which exploits instance-level segmentation maps to perform more stable and multi-scale structural normalization. Beyond dense supervision, we introduce a cost-efficient weak supervision in the form of pairwise ordinal depth annotations to further guide the adaptation process, which enforces informative ordinal constraints to mitigate local topological errors. Finally, a weight regularization loss is employed to anchor the LoRA updates, ensuring training stability and preserving the model's generalizable knowledge. Extensive experiments on both realistic and corrupted out-of-distribution datasets under diverse and challenging scenarios demonstrate that WeSTAR consistently improves generalization and achieves state-of-the-art performance across a wide range of benchmarks.
>
---
#### [new 040] Passive Dementia Screening via Facial Temporal Micro-Dynamics Analysis of In-the-Wild Talking-Head Video
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种无需语言的被动痴呆筛查方法，通过分析自然状态下视频中的面部微动态（如眨眼、嘴部运动等），实现无干预、跨设备、跨文化的早期神经认知变化检测。在新构建的数据集YT DemTalk上，模型达到高准确率。**

- **链接: [https://arxiv.org/pdf/2511.13802v1](https://arxiv.org/pdf/2511.13802v1)**

> **作者:** Filippo Cenacchi. Longbing Cao; Mitchell McEwan; Deborah Richards
>
> **摘要:** We target passive dementia screening from short camera-facing talking head video, developing a facial temporal micro dynamics analysis for language free detection of early neuro cognitive change. This enables unscripted, in the wild video analysis at scale to capture natural facial behaviors, transferrable across devices, topics, and cultures without active intervention by clinicians or researchers during recording. Most existing resources prioritize speech or scripted interviews, limiting use outside clinics and coupling predictions to language and transcription. In contrast, we identify and analyze whether temporal facial kinematics, including blink dynamics, small mouth jaw motions, gaze variability, and subtle head adjustments, are sufficient for dementia screening without speech or text. By stabilizing facial signals, we convert these micro movements into interpretable facial microdynamic time series, smooth them, and summarize short windows into compact clip level statistics for screening. Each window is encoded by its activity mix (the relative share of motion across streams), thus the predictor analyzes the distribution of motion across streams rather than its magnitude, making per channel effects transparent. We also introduce YT DemTalk, a new dataset curated from publicly available, in the wild camera facing videos. It contains 300 clips (150 with self reported dementia, 150 controls) to test our model and offer a first benchmarking of the corpus. On YT DemTalk, ablations identify gaze lability and mouth/jaw dynamics as the most informative cues, and light weighted shallow classifiers could attain a dementia prediction performance of (AUROC) 0.953, 0.961 Average Precision (AP), 0.851 F1-score, and 0.857 accuracy.
>
---
#### [new 041] Multi-view Phase-aware Pedestrian-Vehicle Incident Reasoning Framework with Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MP-PVIR框架，解决行人-车辆事故中行为阶段不明、多视角信息未融合的问题。通过四阶段流程实现事故行为分段、多视角分析与因果推理，提升交通安全管理的智能化水平。**

- **链接: [https://arxiv.org/pdf/2511.14120v1](https://arxiv.org/pdf/2511.14120v1)**

> **作者:** Hao Zhen; Yunxiang Yang; Jidong J. Yang
>
> **备注:** 23 pages, 4 figures, 3 tables
>
> **摘要:** Pedestrian-vehicle incidents remain a critical urban safety challenge, with pedestrians accounting for over 20% of global traffic fatalities. Although existing video-based systems can detect when incidents occur, they provide little insight into how these events unfold across the distinct cognitive phases of pedestrian behavior. Recent vision-language models (VLMs) have shown strong potential for video understanding, but they remain limited in that they typically process videos in isolation, without explicit temporal structuring or multi-view integration. This paper introduces Multi-view Phase-aware Pedestrian-Vehicle Incident Reasoning (MP-PVIR), a unified framework that systematically processes multi-view video streams into structured diagnostic reports through four stages: (1) event-triggered multi-view video acquisition, (2) pedestrian behavior phase segmentation, (3) phase-specific multi-view reasoning, and (4) hierarchical synthesis and diagnostic reasoning. The framework operationalizes behavioral theory by automatically segmenting incidents into cognitive phases, performing synchronized multi-view analysis within each phase, and synthesizing results into causal chains with targeted prevention strategies. Particularly, two specialized VLMs underpin the MP-PVIR pipeline: TG-VLM for behavioral phase segmentation (mIoU = 0.4881) and PhaVR-VLM for phase-aware multi-view analysis, achieving a captioning score of 33.063 and up to 64.70% accuracy on question answering. Finally, a designated large language model is used to generate comprehensive reports detailing scene understanding, behavior interpretation, causal reasoning, and prevention recommendations. Evaluation on the Woven Traffic Safety dataset shows that MP-PVIR effectively translates multi-view video data into actionable insights, advancing AI-driven traffic safety analytics for vehicle-infrastructure cooperative systems.
>
---
#### [new 042] GCA-ResUNet:Image segmentation in medical images using grouped coordinate attention
- **分类: cs.CV; cs.AI**

- **简介: 论文提出GCA-ResUNet用于医学图像分割，解决传统U-Net难以捕捉长距离依赖的问题。通过引入分组坐标注意力机制，在保持高效计算的前提下增强特征表示与边界精度。**

- **链接: [https://arxiv.org/pdf/2511.14087v1](https://arxiv.org/pdf/2511.14087v1)**

> **作者:** Jun Ding; Shang Gao
>
> **摘要:** Medical image segmentation underpins computer-aided diagnosis and therapy by supporting clinical diagnosis, preoperative planning, and disease monitoring. While U-Net style convolutional neural networks perform well due to their encoder-decoder structures with skip connections, they struggle to capture long-range dependencies. Transformer-based variants address global context but often require heavy computation and large training datasets. This paper proposes GCA-ResUNet, an efficient segmentation network that integrates Grouped Coordinate Attention (GCA) into ResNet-50 residual blocks. GCA uses grouped coordinate modeling to jointly encode global dependencies across channels and spatial locations, strengthening feature representation and boundary delineation while adding minimal parameter and FLOP overhead compared with self-attention. On the Synapse dataset, GCA-ResUNet achieves a Dice score of 86.11%, and on the ACDC dataset, it reaches 92.64%, surpassing several state-of-the-art baselines while maintaining fast inference and favorable computational efficiency. These results indicate that GCA offers a practical way to enhance convolutional architectures with global modeling capability, enabling high-accuracy and resource-efficient medical image segmentation.
>
---
#### [new 043] Attention Via Convolutional Nearest Neighbors
- **分类: cs.CV**

- **简介: 论文提出ConvNN框架，统一卷积与自注意力机制，将其视为k近邻聚合的特例。通过在视觉任务中插值二者，提升模型性能与可解释性，解决架构设计中的本质差异问题。**

- **链接: [https://arxiv.org/pdf/2511.14137v1](https://arxiv.org/pdf/2511.14137v1)**

> **作者:** Mingi Kang; Jeová Farias Sales Rocha Neto
>
> **摘要:** The shift from Convolutional Neural Networks to Transformers has reshaped computer vision, yet these two architectural families are typically viewed as fundamentally distinct. We argue that convolution and self-attention, despite their apparent differences, can be unified within a single k-nearest neighbor aggregation framework. The critical insight is that both operations are special cases of neighbor selection and aggregation; convolution selects neighbors by spatial proximity, while attention selects by feature similarity, revealing they exist on a continuous spectrum. We introduce Convolutional Nearest Neighbors (ConvNN), a unified framework that formalizes this connection. Crucially, ConvNN serves as a drop-in replacement for convolutional and attention layers, enabling systematic exploration of the intermediate spectrum between these two extremes. We validate the framework's coherence on CIFAR-10 and CIFAR-100 classification tasks across two complementary architectures: (1) Hybrid branching in VGG improves accuracy on both CIFAR datasets by combining spatial-proximity and feature-similarity selection; and (2) ConvNN in ViT outperforms standard attention and other attention variants on both datasets. Extensive ablations on $k$ values and architectural variants reveal that interpolating along this spectrum provides regularization benefits by balancing local and global receptive fields. Our work provides a unifying framework that dissolves the apparent distinction between convolution and attention, with implications for designing more principled and interpretable vision architectures.
>
---
#### [new 044] Learning Representation and Synergy Invariances: A Povable Framework for Generalized Multimodal Face Anti-Spoofing
- **分类: cs.CV**

- **简介: 该论文属于多模态人脸识别反欺骗任务，针对跨域性能下降问题，提出RiSe框架。通过AsyIRM解决模态表示不变性风险，通过MMSD缓解模态协同伪相关风险，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.14157v1](https://arxiv.org/pdf/2511.14157v1)**

> **作者:** Xun Lin; Shuai Wang; Yi Yu; Zitong Yu; Jiale Zhou; Yizhong Liu; Xiaochun Cao; Alex Kot; Yefeng Zheng
>
> **摘要:** Multimodal Face Anti-Spoofing (FAS) methods, which integrate multiple visual modalities, often suffer even more severe performance degradation than unimodal FAS when deployed in unseen domains. This is mainly due to two overlooked risks that affect cross-domain multimodal generalization. The first is the modal representation invariant risk, i.e., whether representations remain generalizable under domain shift. We theoretically show that the inherent class asymmetry in FAS (diverse spoofs vs. compact reals) enlarges the upper bound of generalization error, and this effect is further amplified in multimodal settings. The second is the modal synergy invariant risk, where models overfit to domain-specific inter-modal correlations. Such spurious synergy cannot generalize to unseen attacks in target domains, leading to performance drops. To solve these issues, we propose a provable framework, namely Multimodal Representation and Synergy Invariance Learning (RiSe). For representation risk, RiSe introduces Asymmetric Invariant Risk Minimization (AsyIRM), which learns an invariant spherical decision boundary in radial space to fit asymmetric distributions, while preserving domain cues in angular space. For synergy risk, RiSe employs Multimodal Synergy Disentanglement (MMSD), a self-supervised task enhancing intrinsic, generalizable modal features via cross-sample mixing and disentanglement. Theoretical analysis and experiments verify RiSe, which achieves state-of-the-art cross-domain performance.
>
---
#### [new 045] XAttn-BMD: Multimodal Deep Learning with Cross-Attention for Femoral Neck Bone Mineral Density Estimation
- **分类: cs.CV**

- **简介: 该论文提出XAttn-BMD模型，用于从髋部X光图像和临床数据中预测股骨颈骨密度（BMD），解决 osteoporosis 早期筛查难题。通过交叉注意力机制融合多模态特征，并设计加权损失函数提升模型性能，实验表明其在回归与分类任务上均优于基线方法。**

- **链接: [https://arxiv.org/pdf/2511.14604v1](https://arxiv.org/pdf/2511.14604v1)**

> **作者:** Yilin Zhang; Leo D. Westbury; Elaine M. Dennison; Nicholas C. Harvey; Nicholas R. Fuggle; Rahman Attar
>
> **备注:** 11 figures, 10 tables, 38 pages. Submitted to Artificial Intelligence in Medicine (currently with editor)
>
> **摘要:** Poor bone health is a significant public health concern, and low bone mineral density (BMD) leads to an increased fracture risk, a key feature of osteoporosis. We present XAttn-BMD (Cross-Attention BMD), a multimodal deep learning framework that predicts femoral neck BMD from hip X-ray images and structured clinical metadata. It utilizes a novel bidirectional cross-attention mechanism to dynamically integrate image and metadata features for cross-modal mutual reinforcement. A Weighted Smooth L1 loss is tailored to address BMD imbalance and prioritize clinically significant cases. Extensive experiments on the data from the Hertfordshire Cohort Study show that our model outperforms the baseline models in regression generalization and robustness. Ablation studies confirm the effectiveness of both cross-attention fusion and the customized loss function. Experimental results show that the integration of multimodal data via cross-attention outperforms naive feature concatenation without cross-attention, reducing MSE by 16.7%, MAE by 6.03%, and increasing the R2 score by 16.4%, highlighting the effectiveness of the approach for femoral neck BMD estimation. Furthermore, screening performance was evaluated using binary classification at clinically relevant femoral neck BMD thresholds, demonstrating the model's potential in real-world scenarios.
>
---
#### [new 046] Free Lunch to Meet the Gap: Intermediate Domain Reconstruction for Cross-Domain Few-Shot Learning
- **分类: cs.CV**

- **简介: 该论文属于跨域小样本学习任务，旨在解决语义不匹配、域差异大和数据稀缺问题。通过构建中间域代理并重建目标域特征，实现快速域对齐，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.14279v1](https://arxiv.org/pdf/2511.14279v1)**

> **作者:** Tong Zhang; Yifan Zhao; Liangyu Wang; Jia Li
>
> **备注:** Accepted to IJCV 2025
>
> **摘要:** Cross-Domain Few-Shot Learning (CDFSL) endeavors to transfer generalized knowledge from the source domain to target domains using only a minimal amount of training data, which faces a triplet of learning challenges in the meantime, i.e., semantic disjoint, large domain discrepancy, and data scarcity. Different from predominant CDFSL works focused on generalized representations, we make novel attempts to construct Intermediate Domain Proxies (IDP) with source feature embeddings as the codebook and reconstruct the target domain feature with this learned codebook. We then conduct an empirical study to explore the intrinsic attributes from perspectives of visual styles and semantic contents in intermediate domain proxies. Reaping benefits from these attributes of intermediate domains, we develop a fast domain alignment method to use these proxies as learning guidance for target domain feature transformation. With the collaborative learning of intermediate domain reconstruction and target feature transformation, our proposed model is able to surpass the state-of-the-art models by a margin on 8 cross-domain few-shot learning benchmarks.
>
---
#### [new 047] Saliency-Guided Deep Learning for Bridge Defect Detection in Drone Imagery
- **分类: cs.CV**

- **简介: 论文提出一种基于显著性引导的深度学习方法，用于无人机影像中桥梁缺陷的自动检测、定位与分类。该方法通过显著性提取缺陷区域，并结合YOLOX模型在增强图像上进行高效精准识别，解决桥梁缺陷智能检测难题。**

- **链接: [https://arxiv.org/pdf/2511.14040v1](https://arxiv.org/pdf/2511.14040v1)**

> **作者:** Loucif Hebbache; Dariush Amirkhani; Mohand Saïd Allili; Jean-François Lapointe
>
> **摘要:** Anomaly object detection and classification are one of the main challenging tasks in computer vision and pattern recognition. In this paper, we propose a new method to automatically detect, localize and classify defects in concrete bridge structures using drone imagery. This framework is constituted of two main stages. The first stage uses saliency for defect region proposals where defects often exhibit local discontinuities in the normal surface patterns with regard to their surrounding. The second stage employs a YOLOX-based deep learning detector that operates on saliency-enhanced images obtained by applying bounding-box level brightness augmentation to salient defect regions. Experimental results on standard datasets confirm the performance of our framework and its suitability in terms of accuracy and computational efficiency, which give a huge potential to be implemented in a self-powered inspection system.
>
---
#### [new 048] V2VLoc: Robust GNSS-Free Collaborative Perception via LiDAR Localization
- **分类: cs.CV**

- **简介: 论文提出V2VLoc框架，解决GNSS失效环境下多智能体协同感知的定位与特征对齐问题。通过轻量级姿态生成器和置信度感知时空对齐Transformer，实现鲁棒LiDAR定位与协作检测，在自建数据集上达到SOTA性能。**

- **链接: [https://arxiv.org/pdf/2511.14247v1](https://arxiv.org/pdf/2511.14247v1)**

> **作者:** Wenkai Lin; Qiming Xia; Wen Li; Xun Huang; Chenglu Wen
>
> **备注:** AAAI2026
>
> **摘要:** Multi-agents rely on accurate poses to share and align observations, enabling a collaborative perception of the environment. However, traditional GNSS-based localization often fails in GNSS-denied environments, making consistent feature alignment difficult in collaboration. To tackle this challenge, we propose a robust GNSS-free collaborative perception framework based on LiDAR localization. Specifically, we propose a lightweight Pose Generator with Confidence (PGC) to estimate compact pose and confidence representations. To alleviate the effects of localization errors, we further develop the Pose-Aware Spatio-Temporal Alignment Transformer (PASTAT), which performs confidence-aware spatial alignment while capturing essential temporal context. Additionally, we present a new simulation dataset, V2VLoc, which can be adapted for both LiDAR localization and collaborative detection tasks. V2VLoc comprises three subsets: Town1Loc, Town4Loc, and V2VDet. Town1Loc and Town4Loc offer multi-traversal sequences for training in localization tasks, whereas V2VDet is specifically intended for the collaborative detection task. Extensive experiments conducted on the V2VLoc dataset demonstrate that our approach achieves state-of-the-art performance under GNSS-denied conditions. We further conduct extended experiments on the real-world V2V4Real dataset to validate the effectiveness and generalizability of PASTAT.
>
---
#### [new 049] Zero-shot Synthetic Video Realism Enhancement via Structure-aware Denoising
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种零样本框架，通过结构感知去噪增强合成视频的真实感，保留时空结构一致性并提升视觉真实度，无需微调扩散视频模型。**

- **链接: [https://arxiv.org/pdf/2511.14719v1](https://arxiv.org/pdf/2511.14719v1)**

> **作者:** Yifan Wang; Liya Ji; Zhanghan Ke; Harry Yang; Ser-Nam Lim; Qifeng Chen
>
> **备注:** Project Page: https://wyf0824.github.io/Video_Realism_Enhancement/
>
> **摘要:** We propose an approach to enhancing synthetic video realism, which can re-render synthetic videos from a simulator in photorealistic fashion. Our realism enhancement approach is a zero-shot framework that focuses on preserving the multi-level structures from synthetic videos into the enhanced one in both spatial and temporal domains, built upon a diffusion video foundational model without further fine-tuning. Specifically, we incorporate an effective modification to have the generation/denoising process conditioned on estimated structure-aware information from the synthetic video, such as depth maps, semantic maps, and edge maps, by an auxiliary model, rather than extracting the information from a simulator. This guidance ensures that the enhanced videos are consistent with the original synthetic video at both the structural and semantic levels. Our approach is a simple yet general and powerful approach to enhancing synthetic video realism: we show that our approach outperforms existing baselines in structural consistency with the original video while maintaining state-of-the-art photorealism quality in our experiments.
>
---
#### [new 050] VLMs Guided Interpretable Decision Making for Autonomous Driving
- **分类: cs.CV**

- **简介: 论文研究自动驾驶中的可解释决策问题，针对VLM直接决策不稳定的问题，提出将其作为语义增强器，融合视觉与语言特征并引入后处理优化，提升决策准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2511.13881v1](https://arxiv.org/pdf/2511.13881v1)**

> **作者:** Xin Hu; Taotao Jing; Renran Tian; Zhengming Ding
>
> **备注:** Accepted by WACV 2026
>
> **摘要:** Recent advancements in autonomous driving (AD) have explored the use of vision-language models (VLMs) within visual question answering (VQA) frameworks for direct driving decision-making. However, these approaches often depend on handcrafted prompts and suffer from inconsistent performance, limiting their robustness and generalization in real-world scenarios. In this work, we evaluate state-of-the-art open-source VLMs on high-level decision-making tasks using ego-view visual inputs and identify critical limitations in their ability to deliver reliable, context-aware decisions. Motivated by these observations, we propose a new approach that shifts the role of VLMs from direct decision generators to semantic enhancers. Specifically, we leverage their strong general scene understanding to enrich existing vision-based benchmarks with structured, linguistically rich scene descriptions. Building on this enriched representation, we introduce a multi-modal interactive architecture that fuses visual and linguistic features for more accurate decision-making and interpretable textual explanations. Furthermore, we design a post-hoc refinement module that utilizes VLMs to enhance prediction reliability. Extensive experiments on two autonomous driving benchmarks demonstrate that our approach achieves state-of-the-art performance, offering a promising direction for integrating VLMs into reliable and interpretable AD systems.
>
---
#### [new 051] Co-Me: Confidence-Guided Token Merging for Visual Geometric Transformers
- **分类: cs.CV; cs.RO**

- **简介: 论文提出Co-Me，一种无需重训练的视觉几何Transformer加速方法。通过轻量级置信度预测器筛选并合并低置信度token，在保持空间覆盖的前提下显著降低计算量，提升实时3D感知与重建效率。**

- **链接: [https://arxiv.org/pdf/2511.14751v1](https://arxiv.org/pdf/2511.14751v1)**

> **作者:** Yutian Chen; Yuheng Qiu; Ruogu Li; Ali Agha; Shayegan Omidshafiei; Jay Patrikar; Sebastian Scherer
>
> **摘要:** We propose Confidence-Guided Token Merging (Co-Me), an acceleration mechanism for visual geometric transformers without retraining or finetuning the base model. Co-Me distilled a light-weight confidence predictor to rank tokens by uncertainty and selectively merge low-confidence ones, effectively reducing computation while maintaining spatial coverage. Compared to similarity-based merging or pruning, the confidence signal in Co-Me reliably indicates regions emphasized by the transformer, enabling substantial acceleration without degrading performance. Co-Me applies seamlessly to various multi-view and streaming visual geometric transformers, achieving speedups that scale with sequence length. When applied to VGGT and MapAnything, Co-Me achieves up to $11.3\times$ and $7.2\times$ speedup, making visual geometric transformers practical for real-time 3D perception and reconstruction.
>
---
#### [new 052] Text-Driven Reasoning Video Editing via Reinforcement Learning on Digital Twin Representations
- **分类: cs.CV**

- **简介: 论文提出推理视频编辑任务，解决用户用隐式文本查询修改视频的难题。RIVER模型通过数字孪生表示分离推理与生成，利用大语言模型多跳推理并指导扩散编辑器执行修改，结合强化学习训练，并构建RVEBenchmark验证性能。**

- **链接: [https://arxiv.org/pdf/2511.14100v1](https://arxiv.org/pdf/2511.14100v1)**

> **作者:** Yiqing Shen; Chenjia Li; Mathias Unberath
>
> **摘要:** Text-driven video editing enables users to modify video content only using text queries. While existing methods can modify video content if explicit descriptions of editing targets with precise spatial locations and temporal boundaries are provided, these requirements become impractical when users attempt to conceptualize edits through implicit queries referencing semantic properties or object relationships. We introduce reasoning video editing, a task where video editing models must interpret implicit queries through multi-hop reasoning to infer editing targets before executing modifications, and a first model attempting to solve this complex task, RIVER (Reasoning-based Implicit Video Editor). RIVER decouples reasoning from generation through digital twin representations of video content that preserve spatial relationships, temporal trajectories, and semantic attributes. A large language model then processes this representation jointly with the implicit query, performing multi-hop reasoning to determine modifications, then outputs structured instructions that guide a diffusion-based editor to execute pixel-level changes. RIVER training uses reinforcement learning with rewards that evaluate reasoning accuracy and generation quality. Finally, we introduce RVEBenchmark, a benchmark of 100 videos with 519 implicit queries spanning three levels and categories of reasoning complexity specifically for reasoning video editing. RIVER demonstrates best performance on the proposed RVEBenchmark and also achieves state-of-the-art performance on two additional video editing benchmarks (VegGIE and FiVE), where it surpasses six baseline methods.
>
---
#### [new 053] InstantViR: Real-Time Video Inverse Problem Solver with Distilled Diffusion Prior
- **分类: cs.CV**

- **简介: 论文提出InstantViR，用于实时视频逆问题求解，如修复、去模糊和超分辨率。通过蒸馏预训练视频扩散模型，将慢速迭代方法转化为单次前向传播，实现高质重建与低延迟（>35 FPS），无需配对数据。**

- **链接: [https://arxiv.org/pdf/2511.14208v1](https://arxiv.org/pdf/2511.14208v1)**

> **作者:** Weimin Bai; Suzhe Xu; Yiwei Ren; Jinhua Hao; Ming Sun; Wenzheng Chen; He Sun
>
> **摘要:** Video inverse problems are fundamental to streaming, telepresence, and AR/VR, where high perceptual quality must coexist with tight latency constraints. Diffusion-based priors currently deliver state-of-the-art reconstructions, but existing approaches either adapt image diffusion models with ad hoc temporal regularizers - leading to temporal artifacts - or rely on native video diffusion models whose iterative posterior sampling is far too slow for real-time use. We introduce InstantViR, an amortized inference framework for ultra-fast video reconstruction powered by a pre-trained video diffusion prior. We distill a powerful bidirectional video diffusion model (teacher) into a causal autoregressive student that maps a degraded video directly to its restored version in a single forward pass, inheriting the teacher's strong temporal modeling while completely removing iterative test-time optimization. The distillation is prior-driven: it only requires the teacher diffusion model and known degradation operators, and does not rely on externally paired clean/noisy video data. To further boost throughput, we replace the video-diffusion backbone VAE with a high-efficiency LeanVAE via an innovative teacher-space regularized distillation scheme, enabling low-latency latent-space processing. Across streaming random inpainting, Gaussian deblurring and super-resolution, InstantViR matches or surpasses the reconstruction quality of diffusion-based baselines while running at over 35 FPS on NVIDIA A100 GPUs, achieving up to 100 times speedups over iterative video diffusion solvers. These results show that diffusion-based video reconstruction is compatible with real-time, interactive, editable, streaming scenarios, turning high-quality video restoration into a practical component of modern vision systems.
>
---
#### [new 054] Training-free Detection of AI-generated images via Cropping Robustness
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出WaRPAD，一种无需训练的AI生成图像检测方法。利用自监督模型对裁剪鲁棒性的特性，通过Haar小波分解提取高频方向敏感性，结合多尺度分块平均得分实现高效检测，适用于多种生成模型和真实数据集。**

- **链接: [https://arxiv.org/pdf/2511.14030v1](https://arxiv.org/pdf/2511.14030v1)**

> **作者:** Sungik Choi; Hankook Lee; Moontae Lee
>
> **摘要:** AI-generated image detection has become crucial with the rapid advancement of vision-generative models. Instead of training detectors tailored to specific datasets, we study a training-free approach leveraging self-supervised models without requiring prior data knowledge. These models, pre-trained with augmentations like RandomResizedCrop, learn to produce consistent representations across varying resolutions. Motivated by this, we propose WaRPAD, a training-free AI-generated image detection algorithm based on self-supervised models. Since neighborhood pixel differences in images are highly sensitive to resizing operations, WaRPAD first defines a base score function that quantifies the sensitivity of image embeddings to perturbations along high-frequency directions extracted via Haar wavelet decomposition. To simulate robustness against cropping augmentation, we rescale each image to a multiple of the models input size, divide it into smaller patches, and compute the base score for each patch. The final detection score is then obtained by averaging the scores across all patches. We validate WaRPAD on real datasets of diverse resolutions and domains, and images generated by 23 different generative models. Our method consistently achieves competitive performance and demonstrates strong robustness to test-time corruptions. Furthermore, as invariance to RandomResizedCrop is a common training scheme across self-supervised models, we show that WaRPAD is applicable across self-supervised models.
>
---
#### [new 055] LINGUAL: Language-INtegrated GUidance in Active Learning for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出LINGUAL框架，用于医学图像分割中的主动学习。针对专家标注耗时且困难的问题，利用自然语言指令自动执行分割任务，减少80%标注时间，提升效率。**

- **链接: [https://arxiv.org/pdf/2511.14028v1](https://arxiv.org/pdf/2511.14028v1)**

> **作者:** Md Shazid Islam; Shreyangshu Bera; Sudipta Paul; Amit K. Roy-Chowdhury
>
> **摘要:** Although active learning (AL) in segmentation tasks enables experts to annotate selected regions of interest (ROIs) instead of entire images, it remains highly challenging, labor-intensive, and cognitively demanding due to the blurry and ambiguous boundaries commonly observed in medical images. Also, in conventional AL, annotation effort is a function of the ROI- larger regions make the task cognitively easier but incur higher annotation costs, whereas smaller regions demand finer precision and more attention from the expert. In this context, language guidance provides an effective alternative, requiring minimal expert effort while bypassing the cognitively demanding task of precise boundary delineation in segmentation. Towards this goal, we introduce LINGUAL: a framework that receives natural language instructions from an expert, translates them into executable programs through in-context learning, and automatically performs the corresponding sequence of sub-tasks without any human intervention. We demonstrate the effectiveness of LINGUAL in active domain adaptation (ADA) achieving comparable or superior performance to AL baselines while reducing estimated annotation time by approximately 80%.
>
---
#### [new 056] A Quantitative Method for Shoulder Presentation Evaluation in Biometric Identity Documents
- **分类: cs.CV**

- **简介: 该论文针对生物特征证件中肩部姿态合规性评估问题，提出一种轻量级定量算法SPE，利用2D姿态估计结果量化肩部偏航和翻滚角度，通过高相关性验证其有效性，可自动检测不合规样本。**

- **链接: [https://arxiv.org/pdf/2511.14376v1](https://arxiv.org/pdf/2511.14376v1)**

> **作者:** Alfonso Pedro Ridao
>
> **备注:** 13 pages, 4 figures, conference or journal submission. Course project from DTU Compute, Technical University of Denmark
>
> **摘要:** International standards for biometric identity documents mandate strict compliance with pose requirements, including the square presentation of a subject's shoulders. However, the literature on automated quality assessment offers few quantitative methods for evaluating this specific attribute. This paper proposes a Shoulder Presentation Evaluation (SPE) algorithm to address this gap. The method quantifies shoulder yaw and roll using only the 3D coordinates of two shoulder landmarks provided by common pose estimation frameworks. The algorithm was evaluated on a dataset of 121 portrait images. The resulting SPE scores demonstrated a strong Pearson correlation (r approx. 0.80) with human-assigned labels. An analysis of the metric's filtering performance, using an adapted Error-versus-Discard methodology, confirmed its utility in identifying non-compliant samples. The proposed algorithm is a viable lightweight tool for automated compliance checking in enrolment systems.
>
---
#### [new 057] Error-Driven Scene Editing for 3D Grounding in Large Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文提出DEER-3D框架，通过错误驱动的场景编辑提升大语言模型在3D环境中的空间定位能力。针对现有模型因训练数据偏重语言推理而缺乏空间理解的问题，该方法基于诊断错误并进行最小化场景修改，实现精准反事实监督，从而显著改善3D接地准确性。**

- **链接: [https://arxiv.org/pdf/2511.14086v1](https://arxiv.org/pdf/2511.14086v1)**

> **作者:** Yue Zhang; Zun Wang; Han Lin; Jialu Li; Jianing Yang; Yonatan Bitton; Idan Szpektor; Mohit Bansal
>
> **备注:** Code: https://github.com/zhangyuejoslin/Deer-3D
>
> **摘要:** Despite recent progress in 3D-LLMs, they remain limited in accurately grounding language to visual and spatial elements in 3D environments. This limitation stems in part from training data that focuses on language reasoning rather than spatial understanding due to scarce 3D resources, leaving inherent grounding biases unresolved. To address this, we propose 3D scene editing as a key mechanism to generate precise visual counterfactuals that mitigate these biases through fine-grained spatial manipulation, without requiring costly scene reconstruction or large-scale 3D data collection. Furthermore, to make these edits targeted and directly address the specific weaknesses of the model, we introduce DEER-3D, an error-driven framework following a structured "Decompose, Diagnostic Evaluation, Edit, and Re-train" workflow, rather than broadly or randomly augmenting data as in conventional approaches. Specifically, upon identifying a grounding failure of the 3D-LLM, our framework first diagnoses the exact predicate-level error (e.g., attribute or spatial relation). It then executes minimal, predicate-aligned 3D scene edits, such as recoloring or repositioning, to produce targeted counterfactual supervision for iterative model fine-tuning, significantly enhancing grounding accuracy. We evaluate our editing pipeline across multiple benchmarks for 3D grounding and scene understanding tasks, consistently demonstrating improvements across all evaluated datasets through iterative refinement. DEER-3D underscores the effectiveness of targeted, error-driven scene editing in bridging linguistic reasoning capabilities with spatial grounding in 3D LLMs.
>
---
#### [new 058] H-CNN-ViT: A Hierarchical Gated Attention Multi-Branch Model for Bladder Cancer Recurrence Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对膀胱癌术后复发预测任务，提出H-CNN-ViT模型，通过多模态MRI数据融合与层次化门控注意力机制，提升预测准确性，解决影像解读困难和缺乏专用数据集的问题。**

- **链接: [https://arxiv.org/pdf/2511.13869v1](https://arxiv.org/pdf/2511.13869v1)**

> **作者:** Xueyang Li; Zongren Wang; Yuliang Zhang; Zixuan Pan; Yu-Jen Chen; Nishchal Sapkota; Gelei Xu; Danny Z. Chen; Yiyu Shi
>
> **摘要:** Bladder cancer is one of the most prevalent malignancies worldwide, with a recurrence rate of up to 78%, necessitating accurate post-operative monitoring for effective patient management. Multi-sequence contrast-enhanced MRI is commonly used for recurrence detection; however, interpreting these scans remains challenging, even for experienced radiologists, due to post-surgical alterations such as scarring, swelling, and tissue remodeling. AI-assisted diagnostic tools have shown promise in improving bladder cancer recurrence prediction, yet progress in this field is hindered by the lack of dedicated multi-sequence MRI datasets for recurrence assessment study. In this work, we first introduce a curated multi-sequence, multi-modal MRI dataset specifically designed for bladder cancer recurrence prediction, establishing a valuable benchmark for future research. We then propose H-CNN-ViT, a new Hierarchical Gated Attention Multi-Branch model that enables selective weighting of features from the global (ViT) and local (CNN) paths based on contextual demands, achieving a balanced and targeted feature fusion. Our multi-branch architecture processes each modality independently, ensuring that the unique properties of each imaging channel are optimally captured and integrated. Evaluated on our dataset, H-CNN-ViT achieves an AUC of 78.6%, surpassing state-of-the-art models. Our model is publicly available at https://github.com/XLIAaron/H-CNN-ViT}.
>
---
#### [new 059] Coffee: Controllable Diffusion Fine-tuning
- **分类: cs.CV**

- **简介: 论文提出Coffee方法，用于可控微调文本到图像扩散模型，解决微调时学习 unwanted concepts 的问题。通过语言指定 undesired concepts，防止其与用户提示纠缠，无需额外训练，可灵活调整。**

- **链接: [https://arxiv.org/pdf/2511.14113v1](https://arxiv.org/pdf/2511.14113v1)**

> **作者:** Ziyao Zeng; Jingcheng Ni; Ruyi Liu; Alex Wong
>
> **摘要:** Text-to-image diffusion models can generate diverse content with flexible prompts, which makes them well-suited for customization through fine-tuning with a small amount of user-provided data. However, controllable fine-tuning that prevents models from learning undesired concepts present in the fine-tuning data, and from entangling those concepts with user prompts, remains an open challenge. It is crucial for downstream tasks like bias mitigation, preventing malicious adaptation, attribute disentanglement, and generalizable fine-tuning of diffusion policy. We propose Coffee that allows using language to specify undesired concepts to regularize the adaptation process. The crux of our method lies in keeping the embeddings of the user prompt from aligning with undesired concepts. Crucially, Coffee requires no additional training and enables flexible modification of undesired concepts by modifying textual descriptions. We evaluate Coffee by fine-tuning on images associated with user prompts paired with undesired concepts. Experimental results demonstrate that Coffee can prevent text-to-image models from learning specified undesired concepts during fine-tuning and outperforms existing methods. Code will be released upon acceptance.
>
---
#### [new 060] Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 论文针对自动驾驶中双目深度估计模型的物理对抗攻击问题，提出一种基于3D全局纹理的对抗样本方法，通过新渲染模块和融合攻击策略，实现视角不变的隐蔽攻击，有效误导立体匹配模型输出错误深度信息。**

- **链接: [https://arxiv.org/pdf/2511.14386v1](https://arxiv.org/pdf/2511.14386v1)**

> **作者:** Kangqiao Zhao; Shuo Huai; Xurui Song; Jun Luo
>
> **摘要:** Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.
>
---
#### [new 061] 2D Gaussians Spatial Transport for Point-supervised Density Regression
- **分类: cs.CV**

- **简介: 该论文提出Gaussian Spatial Transport（GST），用于点监督密度回归任务，解决传统最优传输计算效率低的问题。通过高斯斑点估计像素与标注对应关系，构建贝叶斯运输计划，并设计新损失函数提升模型优化效率。**

- **链接: [https://arxiv.org/pdf/2511.14477v1](https://arxiv.org/pdf/2511.14477v1)**

> **作者:** Miao Shang; Xiaopeng Hong
>
> **备注:** 9 pages, 5 figures, accepted by AAAI, 2026
>
> **摘要:** This paper introduces Gaussian Spatial Transport (GST), a novel framework that leverages Gaussian splatting to facilitate transport from the probability measure in the image coordinate space to the annotation map. We propose a Gaussian splatting-based method to estimate pixel-annotation correspondence, which is then used to compute a transport plan derived from Bayesian probability. To integrate the resulting transport plan into standard network optimization in typical computer vision tasks, we derive a loss function that measures discrepancy after transport. Extensive experiments on representative computer vision tasks, including crowd counting and landmark detection, validate the effectiveness of our approach. Compared to conventional optimal transport schemes, GST eliminates iterative transport plan computation during training, significantly improving efficiency. Code is available at https://github.com/infinite0522/GST.
>
---
#### [new 062] Mind the Gap: Evaluating LLM Understanding of Human-Taught Road Safety Principles
- **分类: cs.CV**

- **简介: 论文评估多模态大语言模型对道路安全原则的理解能力，针对其在零样本场景下识别交通标志与安全规范的不足，构建教材图像数据集并分析人机理解差异，旨在揭示AI在交通安全认知上的短板。**

- **链接: [https://arxiv.org/pdf/2511.13909v1](https://arxiv.org/pdf/2511.13909v1)**

> **作者:** Chalamalasetti Kranti
>
> **备注:** Preprint
>
> **摘要:** Following road safety norms is non-negotiable not only for humans but also for the AI systems that govern autonomous vehicles. In this work, we evaluate how well multi-modal large language models (LLMs) understand road safety concepts, specifically through schematic and illustrative representations. We curate a pilot dataset of images depicting traffic signs and road-safety norms sourced from school text books and use it to evaluate models capabilities in a zero-shot setting. Our preliminary results show that these models struggle with safety reasoning and reveal gaps between human learning and model interpretation. We further provide an analysis of these performance gaps for future research.
>
---
#### [new 063] ArchMap: Arch-Flattening and Knowledge-Guided Vision Language Model for Tooth Counting and Structured Dental Understanding
- **分类: cs.CV**

- **简介: 论文提出ArchMap框架，用于3D牙科扫描的结构化理解任务。针对现有方法依赖大量标注数据和特定设备的问题，该工作通过几何归一化与知识引导推理，实现无需训练的精准牙齿计数、分区及临床条件识别，提升跨设备稳定性与实用性。**

- **链接: [https://arxiv.org/pdf/2511.14336v1](https://arxiv.org/pdf/2511.14336v1)**

> **作者:** Bohan Zhang; Yiyi Miao; Taoyu Wu; Tong Chen; Ji Jiang; Zhuoxiao Li; Zhe Tang; Limin Yu; Jionglong Su
>
> **摘要:** A structured understanding of intraoral 3D scans is essential for digital orthodontics. However, existing deep-learning approaches rely heavily on modality-specific training, large annotated datasets, and controlled scanning conditions, which limit generalization across devices and hinder deployment in real clinical workflows. Moreover, raw intraoral meshes exhibit substantial variation in arch pose, incomplete geometry caused by occlusion or tooth contact, and a lack of texture cues, making unified semantic interpretation highly challenging. To address these limitations, we propose ArchMap, a training-free and knowledge-guided framework for robust structured dental understanding. ArchMap first introduces a geometry-aware arch-flattening module that standardizes raw 3D meshes into spatially aligned, continuity-preserving multi-view projections. We then construct a Dental Knowledge Base (DKB) encoding hierarchical tooth ontology, dentition-stage policies, and clinical semantics to constrain the symbolic reasoning space. We validate ArchMap on 1060 pre-/post-orthodontic cases, demonstrating robust performance in tooth counting, anatomical partitioning, dentition-stage classification, and the identification of clinical conditions such as crowding, missing teeth, prosthetics, and caries. Compared with supervised pipelines and prompted VLM baselines, ArchMap achieves higher accuracy, reduced semantic drift, and superior stability under sparse or artifact-prone conditions. As a fully training-free system, ArchMap demonstrates that combining geometric normalization with ontology-guided multimodal reasoning offers a practical and scalable solution for the structured analysis of 3D intraoral scans in modern digital orthodontics.
>
---
#### [new 064] Find the Leak, Fix the Split: Cluster-Based Method to Prevent Leakage in Video-Derived Datasets
- **分类: cs.CV**

- **简介: 该论文针对视频数据集中的信息泄露问题，提出基于聚类的帧选择策略，通过分组视觉相似帧来优化训练、验证和测试集的划分，提升数据集的代表性与平衡性。**

- **链接: [https://arxiv.org/pdf/2511.13944v1](https://arxiv.org/pdf/2511.13944v1)**

> **作者:** Noam Glazner; Noam Tsfaty; Sharon Shalev; Avishai Weizman
>
> **备注:** 1 figure, 1 table
>
> **摘要:** We propose a cluster-based frame selection strategy to mitigate information leakage in video-derived frames datasets. By grouping visually similar frames before splitting into training, validation, and test sets, the method produces more representative, balanced, and reliable dataset partitions.
>
---
#### [new 065] HyMAD: A Hybrid Multi-Activity Detection Approach for Border Surveillance and Monitoring
- **分类: cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于多活动检测任务，旨在解决地震信号中重叠活动（如人、动物、车辆）难以区分的问题。提出HyMAD框架，融合频谱特征与时间依赖性，通过自注意力和跨模态融合实现高精度多标签分类。**

- **链接: [https://arxiv.org/pdf/2511.14698v1](https://arxiv.org/pdf/2511.14698v1)**

> **作者:** Sriram Srinivasan; Srinivasan Aruchamy; Siva Ram Krisha Vadali
>
> **备注:** Multi-label seismic signal classification using novel attention-based feature fusion. Submitting to cs.CV due to relevance to general pattern recognition and time-frequency (spectrogram) analysis
>
> **摘要:** Seismic sensing has emerged as a promising solution for border surveillance and monitoring; the seismic sensors that are often buried underground are small and cannot be noticed easily, making them difficult for intruders to detect, avoid, or vandalize. This significantly enhances their effectiveness compared to highly visible cameras or fences. However, accurately detecting and distinguishing between overlapping activities that are happening simultaneously, such as human intrusions, animal movements, and vehicle rumbling, remains a major challenge due to the complex and noisy nature of seismic signals. Correctly identifying simultaneous activities is critical because failing to separate them can lead to misclassification, missed detections, and an incomplete understanding of the situation, thereby reducing the reliability of surveillance systems. To tackle this problem, we propose HyMAD (Hybrid Multi-Activity Detection), a deep neural architecture based on spatio-temporal feature fusion. The framework integrates spectral features extracted with SincNet and temporal dependencies modeled by a recurrent neural network (RNN). In addition, HyMAD employs self-attention layers to strengthen intra-modal representations and a cross-modal fusion module to achieve robust multi-label classification of seismic events. e evaluate our approach on a dataset constructed from real-world field recordings collected in the context of border surveillance and monitoring, demonstrating its ability to generalize to complex, simultaneous activity scenarios involving humans, animals, and vehicles. Our method achieves competitive performance and offers a modular framework for extending seismic-based activity recognition in real-world security applications.
>
---
#### [new 066] ARC Is a Vision Problem!
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文将ARC任务视为图像到图像的转换问题，提出基于视觉模型的VARC框架，通过Vision Transformer在无语言依赖下实现高精度推理，显著优于现有纯视觉方法并接近人类水平。**

- **链接: [https://arxiv.org/pdf/2511.14761v1](https://arxiv.org/pdf/2511.14761v1)**

> **作者:** Keya Hu; Ali Cy; Linlu Qiu; Xiaoman Delores Ding; Runqian Wang; Yeyin Eva Zhu; Jacob Andreas; Kaiming He
>
> **备注:** Technical Report. Project webpage: https://github.com/lillian039/VARC
>
> **摘要:** The Abstraction and Reasoning Corpus (ARC) is designed to promote research on abstract reasoning, a fundamental aspect of human intelligence. Common approaches to ARC treat it as a language-oriented problem, addressed by large language models (LLMs) or recurrent reasoning models. However, although the puzzle-like tasks in ARC are inherently visual, existing research has rarely approached the problem from a vision-centric perspective. In this work, we formulate ARC within a vision paradigm, framing it as an image-to-image translation problem. To incorporate visual priors, we represent the inputs on a "canvas" that can be processed like natural images. It is then natural for us to apply standard vision architectures, such as a vanilla Vision Transformer (ViT), to perform image-to-image mapping. Our model is trained from scratch solely on ARC data and generalizes to unseen tasks through test-time training. Our framework, termed Vision ARC (VARC), achieves 60.4% accuracy on the ARC-1 benchmark, substantially outperforming existing methods that are also trained from scratch. Our results are competitive with those of leading LLMs and close the gap to average human performance.
>
---
#### [new 067] CompEvent: Complex-valued Event-RGB Fusion for Low-light Video Enhancement and Deblurring
- **分类: cs.CV**

- **简介: 论文提出CompEvent框架，用于低光视频去模糊任务。针对现有方法在低光和运动模糊联合退化下效果不佳的问题，该工作通过复数神经网络实现事件数据与RGB帧的全流程时空融合，提升恢复效果。**

- **链接: [https://arxiv.org/pdf/2511.14469v1](https://arxiv.org/pdf/2511.14469v1)**

> **作者:** Mingchen Zhong; Xin Lu; Dong Li; Senyan Xu; Ruixuan Jiang; Xueyang Fu; Baocai Yin
>
> **摘要:** Low-light video deblurring poses significant challenges in applications like nighttime surveillance and autonomous driving due to dim lighting and long exposures. While event cameras offer potential solutions with superior low-light sensitivity and high temporal resolution, existing fusion methods typically employ staged strategies, limiting their effectiveness against combined low-light and motion blur degradations. To overcome this, we propose CompEvent, a complex neural network framework enabling holistic full-process fusion of event data and RGB frames for enhanced joint restoration. CompEvent features two core components: 1) Complex Temporal Alignment GRU, which utilizes complex-valued convolutions and processes video and event streams iteratively via GRU to achieve temporal alignment and continuous fusion; and 2) Complex Space-Frequency Learning module, which performs unified complex-valued signal processing in both spatial and frequency domains, facilitating deep fusion through spatial structures and system-level characteristics. By leveraging the holistic representation capability of complex-valued neural networks, CompEvent achieves full-process spatiotemporal fusion, maximizes complementary learning between modalities, and significantly strengthens low-light video deblurring capability. Extensive experiments demonstrate that CompEvent outperforms SOTA methods in addressing this challenging task. The code is available at https://github.com/YuXie1/CompEvent.
>
---
#### [new 068] BEDLAM2.0: Synthetic Humans and Cameras in Motion
- **分类: cs.CV**

- **简介: 该论文提出BEDLAM2.0数据集，用于提升视频中3D人体运动估计的准确性，尤其解决人体与相机同时运动时世界坐标系下估计难题。相比前代数据集，它增强了多样性与真实性，显著改善了模型性能。**

- **链接: [https://arxiv.org/pdf/2511.14394v1](https://arxiv.org/pdf/2511.14394v1)**

> **作者:** Joachim Tesch; Giorgio Becherini; Prerana Achar; Anastasios Yiannakidis; Muhammed Kocabas; Priyanka Patel; Michael J. Black
>
> **备注:** NeurIPS 2025 (Datasets and Benchmarks track, oral). Project website: https://bedlam2.is.tue.mpg.de
>
> **摘要:** Inferring 3D human motion from video remains a challenging problem with many applications. While traditional methods estimate the human in image coordinates, many applications require human motion to be estimated in world coordinates. This is particularly challenging when there is both human and camera motion. Progress on this topic has been limited by the lack of rich video data with ground truth human and camera movement. We address this with BEDLAM2.0, a new dataset that goes beyond the popular BEDLAM dataset in important ways. In addition to introducing more diverse and realistic cameras and camera motions, BEDLAM2.0 increases diversity and realism of body shape, motions, clothing, hair, and 3D environments. Additionally, it adds shoes, which were missing in BEDLAM. BEDLAM has become a key resource for training 3D human pose and motion regressors today and we show that BEDLAM2.0 is significantly better, particularly for training methods that estimate humans in world coordinates. We compare state-of-the art methods trained on BEDLAM and BEDLAM2.0, and find that BEDLAM2.0 significantly improves accuracy over BEDLAM. For research purposes, we provide the rendered videos, ground truth body parameters, and camera motions. We also provide the 3D assets to which we have rights and links to those from third parties.
>
---
#### [new 069] MRI Embeddings Complement Clinical Predictors for Cognitive Decline Modeling in Alzheimer's Disease Cohorts
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于阿尔茨海默病认知衰退预测任务，旨在解决单一模态特征难以全面捕捉病情变化的问题。作者提出基于3D Vision Transformer的MRI嵌入方法，结合临床表型数据，通过轨迹感知标签和无监督预训练提升模型性能，验证了多模态互补性。**

- **链接: [https://arxiv.org/pdf/2511.14601v1](https://arxiv.org/pdf/2511.14601v1)**

> **作者:** Nathaniel Putera; Daniel Vilet Rodríguez; Noah Videcrantz; Julia Machnio; Mostafa Mehdipour Ghazi
>
> **备注:** Accepted at SPIE - Medical Imaging Conference 2026
>
> **摘要:** Accurate modeling of cognitive decline in Alzheimer's disease is essential for early stratification and personalized management. While tabular predictors provide robust markers of global risk, their ability to capture subtle brain changes remains limited. In this study, we evaluate the predictive contributions of tabular and imaging-based representations, with a focus on transformer-derived Magnetic Resonance Imaging (MRI) embeddings. We introduce a trajectory-aware labeling strategy based on Dynamic Time Warping clustering to capture heterogeneous patterns of cognitive change, and train a 3D Vision Transformer (ViT) via unsupervised reconstruction on harmonized and augmented MRI data to obtain anatomy-preserving embeddings without progression labels. The pretrained encoder embeddings are subsequently assessed using both traditional machine learning classifiers and deep learning heads, and compared against tabular representations and convolutional network baselines. Results highlight complementary strengths across modalities. Clinical and volumetric features achieved the highest AUCs of around 0.70 for predicting mild and severe progression, underscoring their utility in capturing global decline trajectories. In contrast, MRI embeddings from the ViT model were most effective in distinguishing cognitively stable individuals with an AUC of 0.71. However, all approaches struggled in the heterogeneous moderate group. These findings indicate that clinical features excel in identifying high-risk extremes, whereas transformer-based MRI embeddings are more sensitive to subtle markers of stability, motivating multimodal fusion strategies for AD progression modeling.
>
---
#### [new 070] RTS-Mono: A Real-Time Self-Supervised Monocular Depth Estimation Method for Real-World Deployment
- **分类: cs.CV**

- **简介: 论文提出RTS-Mono，一种轻量级实时自监督单目深度估计方法，解决现有模型计算资源消耗大、性能差的问题。通过轻量编码器和多尺度稀疏融合解码器，在保持高精度的同时实现高速推理，适用于真实场景部署。**

- **链接: [https://arxiv.org/pdf/2511.14107v1](https://arxiv.org/pdf/2511.14107v1)**

> **作者:** Zeyu Cheng; Tongfei Liu; Tao Lei; Xiang Hua; Yi Zhang; Chengkai Tang
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Depth information is crucial for autonomous driving and intelligent robot navigation. The simplicity and flexibility of self-supervised monocular depth estimation are conducive to its role in these fields. However, most existing monocular depth estimation models consume many computing resources. Although some methods have reduced the model's size and improved computing efficiency, the performance deteriorates, seriously hindering the real-world deployment of self-supervised monocular depth estimation models in the real world. To address this problem, we proposed a real-time self-supervised monocular depth estimation method and implemented it in the real world. It is called RTS-Mono, which is a lightweight and efficient encoder-decoder architecture. The encoder is based on Lite-Encoder, and the decoder is designed with a multi-scale sparse fusion framework to minimize redundancy, ensure performance, and improve inference speed. RTS-Mono achieved state-of-the-art (SoTA) performance in high and low resolutions with extremely low parameter counts (3 M) in experiments based on the KITTI dataset. Compared with lightweight methods, RTS-Mono improved Abs Rel and Sq Rel by 5.6% and 9.8% at low resolution and improved Sq Rel and RMSE by 6.1% and 1.9% at high resolution. In real-world deployment experiments, RTS-Mono has extremely high accuracy and can perform real-time inference on Nvidia Jetson Orin at a speed of 49 FPS. Source code is available at https://github.com/ZYCheng777/RTS-Mono.
>
---
#### [new 071] GloTok: Global Perspective Tokenizer for Image Reconstruction and Generation
- **分类: cs.CV**

- **简介: 论文提出GloTok，一种用于图像重建与生成的全局视角分词器，解决现有方法因局部监督导致语义分布不均的问题。通过全局关系建模和残差学习，实现更均匀的潜在表示，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.14184v1](https://arxiv.org/pdf/2511.14184v1)**

> **作者:** Xuan Zhao; Zhongyu Zhang; Yuge Huang; Yuxi Mi; Guodong Mu; Shouhong Ding; Jun Wang; Rizen Guo; Shuigeng Zhou
>
> **摘要:** Existing state-of-the-art image tokenization methods leverage diverse semantic features from pre-trained vision models for additional supervision, to expand the distribution of latent representations and thereby improve the quality of image reconstruction and generation. These methods employ a locally supervised approach for semantic supervision, which limits the uniformity of semantic distribution. However, VA-VAE proves that a more uniform feature distribution yields better generation performance. In this work, we introduce a Global Perspective Tokenizer (GloTok), which utilizes global relational information to model a more uniform semantic distribution of tokenized features. Specifically, a codebook-wise histogram relation learning method is proposed to transfer the semantics, which are modeled by pre-trained models on the entire dataset, to the semantic codebook. Then, we design a residual learning module that recovers the fine-grained details to minimize the reconstruction error caused by quantization. Through the above design, GloTok delivers more uniformly distributed semantic latent representations, which facilitates the training of autoregressive (AR) models for generating high-quality images without requiring direct access to pre-trained models during the training process. Experiments on the standard ImageNet-1k benchmark clearly show that our proposed method achieves state-of-the-art reconstruction performance and generation quality.
>
---
#### [new 072] Improving segmentation of retinal arteries and veins using cardiac signal in doppler holograms
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种基于心脏信号的动脉静脉分割方法，解决传统方法忽略时序信息的问题。通过在U-Net中引入脉搏分析特征，提升Doppler全息图中血管分割精度，实现高效定量评估视网膜血流动力学。**

- **链接: [https://arxiv.org/pdf/2511.14654v1](https://arxiv.org/pdf/2511.14654v1)**

> **作者:** Marius Dubosc; Yann Fischer; Zacharie Auray; Nicolas Boutry; Edwin Carlinet; Michael Atlan; Thierry Geraud
>
> **备注:** 5 pages, 3 figures, 1 table. Submitted to ISBI2026
>
> **摘要:** Doppler holography is an emerging retinal imaging technique that captures the dynamic behavior of blood flow with high temporal resolution, enabling quantitative assessment of retinal hemodynamics. This requires accurate segmentation of retinal arteries and veins, but traditional segmentation methods focus solely on spatial information and overlook the temporal richness of holographic data. In this work, we propose a simple yet effective approach for artery-vein segmentation in temporal Doppler holograms using standard segmentation architectures. By incorporating features derived from a dedicated pulse analysis pipeline, our method allows conventional U-Nets to exploit temporal dynamics and achieve performance comparable to more complex attention- or iteration-based models. These findings demonstrate that time-resolved preprocessing can unlock the full potential of deep learning for Doppler holography, opening new perspectives for quantitative exploration of retinal hemodynamics. The dataset is publicly available at https://huggingface.co/datasets/DigitalHolography/
>
---
#### [new 073] Explaining Digital Pathology Models via Clustering Activations
- **分类: cs.CV**

- **简介: 论文提出一种基于聚类的可解释性方法，用于分析数字病理学中的CNN模型。该方法通过可视化激活聚类，揭示模型全局行为并提供细粒度信息，解决传统梯度类方法仅关注局部区域的问题，提升临床信任与采纳速度。**

- **链接: [https://arxiv.org/pdf/2511.14558v1](https://arxiv.org/pdf/2511.14558v1)**

> **作者:** Adam Bajger; Jan Obdržálek; Vojtěch Kůr; Rudolf Nenutil; Petr Holub; Vít Musil; Tomáš Brázdil
>
> **摘要:** We present a clustering-based explainability technique for digital pathology models based on convolutional neural networks. Unlike commonly used methods based on saliency maps, such as occlusion, GradCAM, or relevance propagation, which highlight regions that contribute the most to the prediction for a single slide, our method shows the global behaviour of the model under consideration, while also providing more fine-grained information. The result clusters can be visualised not only to understand the model, but also to increase confidence in its operation, leading to faster adoption in clinical practice. We also evaluate the performance of our technique on an existing model for detecting prostate cancer, demonstrating its usefulness.
>
---
#### [new 074] OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型中音视频token处理效率低的问题，提出无需训练的OmniZip框架。通过音频引导动态压缩视频token，实现加速推理与内存优化，同时保持性能不变。**

- **链接: [https://arxiv.org/pdf/2511.14582v1](https://arxiv.org/pdf/2511.14582v1)**

> **作者:** Keda Tao; Kele Shao; Bohan Yu; Weiqiang Wang; Jian liu; Huan Wang
>
> **备注:** Code Link: https://github.com/KD-TAO/OmniZip
>
> **摘要:** Omnimodal large language models (OmniLLMs) have attracted increasing research attention of late towards unified audio-video understanding, wherein processing audio-video token sequences creates a significant computational bottleneck, however. Existing token compression methods have yet to accommodate this emerging need of jointly compressing multimodal tokens. To bridge this gap, we present OmniZip, a training-free, audio-guided audio-visual token-compression framework that optimizes multimodal token representation and accelerates inference. Specifically, OmniZip first identifies salient audio tokens, then computes an audio retention score for each time group to capture information density, thereby dynamically guiding video token pruning and preserving cues from audio anchors enhanced by cross-modal similarity. For each time window, OmniZip compresses the video tokens using an interleaved spatio-temporal scheme. Extensive empirical results demonstrate the merits of OmniZip - it achieves 3.42X inference speedup and 1.4X memory reduction over other top-performing counterparts, while maintaining performance with no training.
>
---
#### [new 075] SMGeo: Cross-View Object Geo-Localization with Grid-Level Mixture-of-Experts
- **分类: cs.CV**

- **简介: 论文提出SMGeo模型，解决无人机与卫星图像间跨视图目标地理定位问题。通过端到端Transformer架构和网格级稀疏专家混合机制，实现高精度实时定位，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.14093v1](https://arxiv.org/pdf/2511.14093v1)**

> **作者:** Fan Zhang; Haoyuan Ren; Fei Ma; Qiang Yin; Yongsheng Zhou
>
> **摘要:** Cross-view object Geo-localization aims to precisely pinpoint the same object across large-scale satellite imagery based on drone images. Due to significant differences in viewpoint and scale, coupled with complex background interference, traditional multi-stage "retrieval-matching" pipelines are prone to cumulative errors. To address this, we present SMGeo, a promptable end-to-end transformer-based model for object Geo-localization. This model supports click prompting and can output object Geo-localization in real time when prompted to allow for interactive use. The model employs a fully transformer-based architecture, utilizing a Swin-Transformer for joint feature encoding of both drone and satellite imagery and an anchor-free transformer detection head for coordinate regression. In order to better capture both inter-modal and intra-view dependencies, we introduce a grid-level sparse Mixture-of-Experts (GMoE) into the cross-view encoder, allowing it to adaptively activate specialized experts according to the content, scale and source of each grid. We also employ an anchor-free detection head for coordinate regression, directly predicting object locations via heat-map supervision in the reference images. This approach avoids scale bias and matching complexity introduced by predefined anchor boxes. On the drone-to-satellite task, SMGeo achieves leading performance in accuracy at IoU=0.25 and mIoU metrics (e.g., 87.51%, 62.50%, and 61.45% in the test set, respectively), significantly outperforming representative methods such as DetGeo (61.97%, 57.66%, and 54.05%, respectively). Ablation studies demonstrate complementary gains from shared encoding, query-guided fusion, and grid-level sparse mixture-of-experts.
>
---
#### [new 076] UniGen-1.5: Enhancing Image Generation and Editing through Reward Unification in Reinforcement Learning
- **分类: cs.CV**

- **简介: 论文提出UniGen-1.5，一个用于图像生成与编辑的统一多模态大模型。通过强化学习和轻量编辑指令对齐，提升图像理解和生成能力，解决图像编辑指令理解不足的问题，性能优于现有模型。**

- **链接: [https://arxiv.org/pdf/2511.14760v1](https://arxiv.org/pdf/2511.14760v1)**

> **作者:** Rui Tian; Mingfei Gao; Haiming Gang; Jiasen Lu; Zhe Gan; Yinfei Yang; Zuxuan Wu; Afshin Dehghan
>
> **摘要:** We present UniGen-1.5, a unified multimodal large language model (MLLM) for advanced image understanding, generation and editing. Building upon UniGen, we comprehensively enhance the model architecture and training pipeline to strengthen the image understanding and generation capabilities while unlocking strong image editing ability. Especially, we propose a unified Reinforcement Learning (RL) strategy that improves both image generation and image editing jointly via shared reward models. To further enhance image editing performance, we propose a light Edit Instruction Alignment stage that significantly improves the editing instruction comprehension that is essential for the success of the RL training. Experimental results show that UniGen-1.5 demonstrates competitive understanding and generation performance. Specifically, UniGen-1.5 achieves 0.89 and 4.31 overall scores on GenEval and ImgEdit that surpass the state-of-the-art models such as BAGEL and reaching performance comparable to proprietary models such as GPT-Image-1.
>
---
#### [new 077] Zero-Training Task-Specific Model Synthesis for Few-Shot Medical Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 论文提出零训练任务特定模型合成方法（ZS-TMS），用于少样本医学图像分类。针对医疗数据稀缺难题，利用预训练生成模型根据单张图像和文本描述直接合成分类器参数，无需训练即可部署，显著提升1-shot和5-shot场景下的性能。**

- **链接: [https://arxiv.org/pdf/2511.14082v1](https://arxiv.org/pdf/2511.14082v1)**

> **作者:** Yao Qin; Yangyang Yan; YuanChao Yang; Jinhua Pang; Huanyong Bi; Yuan Liu; HaiHua Wang
>
> **摘要:** Deep learning models have achieved remarkable success in medical image analysis but are fundamentally constrained by the requirement for large-scale, meticulously annotated datasets. This dependency on "big data" is a critical bottleneck in the medical domain, where patient data is inherently difficult to acquire and expert annotation is expensive, particularly for rare diseases where samples are scarce by definition. To overcome this fundamental challenge, we propose a novel paradigm: Zero-Training Task-Specific Model Synthesis (ZS-TMS). Instead of adapting a pre-existing model or training a new one, our approach leverages a large-scale, pre-trained generative engine to directly synthesize the entire set of parameters for a task-specific classifier. Our framework, the Semantic-Guided Parameter Synthesizer (SGPS), takes as input minimal, multi-modal task information as little as a single example image (1-shot) and a corresponding clinical text description to directly synthesize the entire set of parameters for a task-specific classifier. The generative engine interprets these inputs to generate the weights for a lightweight, efficient classifier (e.g., an EfficientNet-V2), which can be deployed for inference immediately without any task-specific training or fine-tuning. We conduct extensive evaluations on challenging few-shot classification benchmarks derived from the ISIC 2018 skin lesion dataset and a custom rare disease dataset. Our results demonstrate that SGPS establishes a new state-of-the-art, significantly outperforming advanced few-shot and zero-shot learning methods, especially in the ultra-low data regimes of 1-shot and 5-shot classification. This work paves the way for the rapid development and deployment of AI-powered diagnostic tools, particularly for the long tail of rare diseases where data is critically limited.
>
---
#### [new 078] Enhancing LLM-based Autonomous Driving with Modular Traffic Light and Sign Recognition
- **分类: cs.CV**

- **简介: 论文提出TLS-Assist模块，增强LLM在自动驾驶中的交通灯和标志识别能力，解决其对关键物体检测不稳的问题。该模块将检测结果转为自然语言注入LLM输入，提升安全性与规则遵守，支持多视角相机，实验显示性能提升显著。**

- **链接: [https://arxiv.org/pdf/2511.14391v1](https://arxiv.org/pdf/2511.14391v1)**

> **作者:** Fabian Schmidt; Noushiq Mohammed Kayilan Abdul Nazar; Markus Enzweiler; Abhinav Valada
>
> **摘要:** Large Language Models (LLMs) are increasingly used for decision-making and planning in autonomous driving, showing promising reasoning capabilities and potential to generalize across diverse traffic situations. However, current LLM-based driving agents lack explicit mechanisms to enforce traffic rules and often struggle to reliably detect small, safety-critical objects such as traffic lights and signs. To address this limitation, we introduce TLS-Assist, a modular redundancy layer that augments LLM-based autonomous driving agents with explicit traffic light and sign recognition. TLS-Assist converts detections into structured natural language messages that are injected into the LLM input, enforcing explicit attention to safety-critical cues. The framework is plug-and-play, model-agnostic, and supports both single-view and multi-view camera setups. We evaluate TLS-Assist in a closed-loop setup on the LangAuto benchmark in CARLA. The results demonstrate relative driving performance improvements of up to 14% over LMDrive and 7% over BEVDriver, while consistently reducing traffic light and sign infractions. We publicly release the code and models on https://github.com/iis-esslingen/TLS-Assist.
>
---
#### [new 079] SAE-MCVT: A Real-Time and Scalable Multi-Camera Vehicle Tracking Framework Powered by Edge Computing
- **分类: cs.CV**

- **简介: 论文提出SAE-MCVT框架，解决多摄像头车辆追踪（MCVT）中实时性和可扩展性不足的问题。通过边缘计算与中心协同处理，实现城市规模下的高效轨迹跟踪，实验表明其在2K分辨率下保持实时运行并达到61.2 IDF1得分。**

- **链接: [https://arxiv.org/pdf/2511.13904v1](https://arxiv.org/pdf/2511.13904v1)**

> **作者:** Yuqiang Lin; Sam Lockyer; Florian Stanek; Markus Zarbock; Adrian Evans; Wenbin Li; Nic Zhang
>
> **摘要:** In modern Intelligent Transportation Systems (ITS), cameras are a key component due to their ability to provide valuable information for multiple stakeholders. A central task is Multi-Camera Vehicle Tracking (MCVT), which generates vehicle trajectories and enables applications such as anomaly detection, traffic density estimation, and suspect vehicle tracking. However, most existing studies on MCVT emphasize accuracy while overlooking real-time performance and scalability. These two aspects are essential for real-world deployment and become increasingly challenging in city-scale applications as the number of cameras grows. To address this issue, we propose SAE-MCVT, the first scalable real-time MCVT framework. The system includes several edge devices that interact with one central workstation separately. On the edge side, live RTSP video streams are serialized and processed through modules including object detection, object tracking, geo-mapping, and feature extraction. Only lightweight metadata -- vehicle locations and deep appearance features -- are transmitted to the central workstation. On the central side, cross-camera association is calculated under the constraint of spatial-temporal relations between adjacent cameras, which are learned through a self-supervised camera link model. Experiments on the RoundaboutHD dataset show that SAE-MCVT maintains real-time operation on 2K 15 FPS video streams and achieves an IDF1 score of 61.2. To the best of our knowledge, this is the first scalable real-time MCVT framework suitable for city-scale deployment.
>
---
#### [new 080] Silhouette-to-Contour Registration: Aligning Intraoral Scan Models with Cephalometric Radiographs
- **分类: cs.CV**

- **简介: 该论文属于医学影像配准任务，旨在解决口腔扫描模型与头颅侧位片间因几何畸变、低对比度等导致的3D-2D对齐难题。提出DentalSCR框架，通过构建统一坐标系和轮廓引导注册，实现稳定、高精度且可解释的对齐。**

- **链接: [https://arxiv.org/pdf/2511.14343v1](https://arxiv.org/pdf/2511.14343v1)**

> **作者:** Yiyi Miao; Taoyu Wu; Ji Jiang; Tong Chen; Zhe Tang; Zhengyong Jiang; Angelos Stefanidis; Limin Yu; Jionglong Su
>
> **摘要:** Reliable 3D-2D alignment between intraoral scan (IOS) models and lateral cephalometric radiographs is critical for orthodontic diagnosis, yet conventional intensity-driven registration methods struggle under real clinical conditions, where cephalograms exhibit projective magnification, geometric distortion, low-contrast dental crowns, and acquisition-dependent variation. These factors hinder the stability of appearance-based similarity metrics and often lead to convergence failures or anatomically implausible alignments. To address these limitations, we propose DentalSCR, a pose-stable, contour-guided framework for accurate and interpretable silhouette-to-contour registration. Our method first constructs a U-Midline Dental Axis (UMDA) to establish a unified cross-arch anatomical coordinate system, thereby stabilizing initialization and standardizing projection geometry across cases. Using this reference frame, we generate radiograph-like projections via a surface-based DRR formulation with coronal-axis perspective and Gaussian splatting, which preserves clinical source-object-detector magnification and emphasizes external silhouettes. Registration is then formulated as a 2D similarity transform optimized with a symmetric bidirectional Chamfer distance under a hierarchical coarse-to-fine schedule, enabling both large capture range and subpixel-level contour agreement. We evaluate DentalSCR on 34 expert-annotated clinical cases. Experimental results demonstrate substantial reductions in landmark error-particularly at posterior teeth-tighter dispersion on the lower jaw, and low Chamfer and controlled Hausdorff distances at the curve level. These findings indicate that DentalSCR robustly handles real-world cephalograms and delivers high-fidelity, clinically inspectable 3D--2D alignment, outperforming conventional baselines.
>
---
#### [new 081] A Generative Data Framework with Authentic Supervision for Underwater Image Restoration and Enhancement
- **分类: cs.CV**

- **简介: 论文针对水下图像恢复与增强任务，解决现有方法因缺乏高质量配对数据导致的色偏和泛化差问题。提出基于自然图像生成合成数据集的方法，构建6类退化场景的真值标签，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.14521v1](https://arxiv.org/pdf/2511.14521v1)**

> **作者:** Yufeng Tian; Yifan Chen; Zhe Sun; Libang Chen; Mingyu Dou; Jijun Lu; Ye Zheng; Xuelong Li
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Underwater image restoration and enhancement are crucial for correcting color distortion and restoring image details, thereby establishing a fundamental basis for subsequent underwater visual tasks. However, current deep learning methodologies in this area are frequently constrained by the scarcity of high-quality paired datasets. Since it is difficult to obtain pristine reference labels in underwater scenes, existing benchmarks often rely on manually selected results from enhancement algorithms, providing debatable reference images that lack globally consistent color and authentic supervision. This limits the model's capabilities in color restoration, image enhancement, and generalization. To overcome this limitation, we propose using in-air natural images as unambiguous reference targets and translating them into underwater-degraded versions, thereby constructing synthetic datasets that provide authentic supervision signals for model learning. Specifically, we establish a generative data framework based on unpaired image-to-image translation, producing a large-scale dataset that covers 6 representative underwater degradation types. The framework constructs synthetic datasets with precise ground-truth labels, which facilitate the learning of an accurate mapping from degraded underwater images to their pristine scene appearances. Extensive quantitative and qualitative experiments across 6 representative network architectures and 3 independent test sets show that models trained on our synthetic data achieve comparable or superior color restoration and generalization performance to those trained on existing benchmarks. This research provides a reliable and scalable data-driven solution for underwater image restoration and enhancement. The generated dataset is publicly available at: https://github.com/yftian2025/SynUIEDatasets.git.
>
---
#### [new 082] D-PerceptCT: Deep Perceptual Enhancement for Low-Dose CT Images
- **分类: cs.CV**

- **简介: 该论文针对低剂量CT图像质量差的问题，提出D-PerceptCT模型，通过模拟人类视觉系统增强图像中关键解剖结构和病理细节，提升诊断可用性。**

- **链接: [https://arxiv.org/pdf/2511.14518v1](https://arxiv.org/pdf/2511.14518v1)**

> **作者:** Taifour Yousra Nabila; Azeddine Beghdadi; Marie Luong; Zuheng Ming; Habib Zaidi; Faouzi Alaya Cheikh
>
> **摘要:** Low Dose Computed Tomography (LDCT) is widely used as an imaging solution to aid diagnosis and other clinical tasks. However, this comes at the price of a deterioration in image quality due to the low dose of radiation used to reduce the risk of secondary cancer development. While some efficient methods have been proposed to enhance LDCT quality, many overestimate noise and perform excessive smoothing, leading to a loss of critical details. In this paper, we introduce D-PerceptCT, a novel architecture inspired by key principles of the Human Visual System (HVS) to enhance LDCT images. The objective is to guide the model to enhance or preserve perceptually relevant features, thereby providing radiologists with CT images where critical anatomical structures and fine pathological details are perceptu- ally visible. D-PerceptCT consists of two main blocks: 1) a Visual Dual-path Extractor (ViDex), which integrates semantic priors from a pretrained DINOv2 model with local spatial features, allowing the network to incorporate semantic-awareness during enhancement; (2) a Global-Local State-Space block that captures long-range information and multiscale features to preserve the important structures and fine details for diagnosis. In addition, we propose a novel deep perceptual loss, designated as the Deep Perceptual Relevancy Loss Function (DPRLF), which is inspired by human contrast sensitivity, to further emphasize perceptually important features. Extensive experiments on the Mayo2016 dataset demonstrate the effectiveness of D-PerceptCT method for LDCT enhancement, showing better preservation of structural and textural information within LDCT images compared to SOTA methods.
>
---
#### [new 083] Fusing Biomechanical and Spatio-Temporal Features for Fall Prediction: Characterizing and Mitigating the Simulation-to-Reality Gap
- **分类: cs.CV**

- **简介: 该论文属于跌倒预测任务，旨在解决模拟数据与真实场景间的性能差距问题。提出BioST-GCN模型融合生物力学与时空特征，提升预测准确性，并通过个性化策略和隐私保护数据流程缓解仿真到现实的偏差。**

- **链接: [https://arxiv.org/pdf/2511.14620v1](https://arxiv.org/pdf/2511.14620v1)**

> **作者:** Md Fokhrul Islam; Sajeda Al-Hammouri; Christopher J. Arellano; Kavan Hazeli; Heman Shakeri
>
> **摘要:** Falls are a leading cause of injury and loss of independence among older adults. Vision-based fall prediction systems offer a non-invasive solution to anticipate falls seconds before impact, but their development is hindered by the scarcity of available fall data. Contributing to these efforts, this study proposes the Biomechanical Spatio-Temporal Graph Convolutional Network (BioST-GCN), a dual-stream model that combines both pose and biomechanical information using a cross-attention fusion mechanism. Our model outperforms the vanilla ST-GCN baseline by 5.32% and 2.91% F1-score on the simulated MCF-UA stunt-actor and MUVIM datasets, respectively. The spatio-temporal attention mechanisms in the ST-GCN stream also provide interpretability by identifying critical joints and temporal phases. However, a critical simulation-reality gap persists. While our model achieves an 89.0% F1-score with full supervision on simulated data, zero-shot generalization to unseen subjects drops to 35.9%. This performance decline is likely due to biases in simulated data, such as `intent-to-fall' cues. For older adults, particularly those with diabetes or frailty, this gap is exacerbated by their unique kinematic profiles. To address this, we propose personalization strategies and advocate for privacy-preserving data pipelines to enable real-world validation. Our findings underscore the urgent need to bridge the gap between simulated and real-world data to develop effective fall prediction systems for vulnerable elderly populations.
>
---
#### [new 084] nuCarla: A nuScenes-Style Bird's-Eye View Perception Dataset for CARLA Simulation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出nuCarla，一个用于CARLA仿真中鸟瞰图感知的大型数据集，解决闭环端到端自动驾驶训练缺乏标准数据的问题。该工作提供兼容nuScenes格式、类分布均衡的数据及高性能BEV模型，加速闭环模拟开发。**

- **链接: [https://arxiv.org/pdf/2511.13744v1](https://arxiv.org/pdf/2511.13744v1)**

> **作者:** Zhijie Qiao; Zhong Cao; Henry X. Liu
>
> **摘要:** End-to-end (E2E) autonomous driving heavily relies on closed-loop simulation, where perception, planning, and control are jointly trained and evaluated in interactive environments. Yet, most existing datasets are collected from the real world under non-interactive conditions, primarily supporting open-loop learning while offering limited value for closed-loop testing. Due to the lack of standardized, large-scale, and thoroughly verified datasets to facilitate learning of meaningful intermediate representations, such as bird's-eye-view (BEV) features, closed-loop E2E models remain far behind even simple rule-based baselines. To address this challenge, we introduce nuCarla, a large-scale, nuScenes-style BEV perception dataset built within the CARLA simulator. nuCarla features (1) full compatibility with the nuScenes format, enabling seamless transfer of real-world perception models; (2) a dataset scale comparable to nuScenes, but with more balanced class distributions; (3) direct usability for closed-loop simulation deployment; and (4) high-performance BEV backbones that achieve state-of-the-art detection results. By providing both data and models as open benchmarks, nuCarla substantially accelerates closed-loop E2E development, paving the way toward reliable and safety-aware research in autonomous driving.
>
---
#### [new 085] Hierarchical Semantic Learning for Multi-Class Aorta Segmentation
- **分类: cs.CV**

- **简介: 论文针对多类主动脉分割任务，解决现有方法忽视解剖层次关系和类别不平衡的问题。提出基于课程学习的分层语义损失与分形Softmax，提升分割精度与效率，实验表明Dice分数显著优于基线。**

- **链接: [https://arxiv.org/pdf/2511.14187v1](https://arxiv.org/pdf/2511.14187v1)**

> **作者:** Pengcheng Shi
>
> **备注:** Accepted by MICCAI 2024 Workshop AortaSeg
>
> **摘要:** The aorta, the body's largest artery, is prone to pathologies such as dissection, aneurysm, and atherosclerosis, which often require timely intervention. Minimally invasive repairs involving branch vessels necessitate detailed 3D anatomical analysis. Existing methods often overlook hierarchical anatomical relationships while struggling with severe class imbalance inherent in vascular structures. We address these challenges with a curriculum learning strategy that leverages a novel fractal softmax for hierarchical semantic learning. Inspired by human cognition, our approach progressively learns anatomical constraints by decomposing complex structures from simple to complex components. The curriculum learning framework naturally addresses class imbalance by first establishing robust feature representations for dominant classes before tackling rare but anatomically critical structures, significantly accelerating model convergence in multi-class scenarios. Our two-stage inference strategy achieves up to fivefold acceleration, enhancing clinical practicality. On the validation set at epoch 50, our hierarchical semantic loss improves the Dice score of nnU-Net ResEnc M by 11.65%. The proposed model demonstrates a 5.6% higher Dice score than baselines on the test set. Experimental results show significant improvements in segmentation accuracy and efficiency, making the framework suitable for real-time clinical applications. The implementation code for this challenge entry is publicly available at: https://github.com/PengchengShi1220/AortaSeg24. The code for fractal softmax will be available at https://github.com/PengchengShi1220/fractal-softmax.
>
---
#### [new 086] Segmentation-Aware Latent Diffusion for Satellite Image Super-Resolution: Enabling Smallholder Farm Boundary Delineation
- **分类: cs.CV**

- **简介: 论文提出SEED-SR方法，解决小农户农田边界分割难题。通过在分割感知的潜在空间中进行超分辨率重建，实现20倍放大，显著提升分割精度，优于现有参考图像超分方法。**

- **链接: [https://arxiv.org/pdf/2511.14481v1](https://arxiv.org/pdf/2511.14481v1)**

> **作者:** Aditi Agarwal; Anjali Jain; Nikita Saxena; Ishan Deshpande; Michal Kazmierski; Abigail Annkah; Nadav Sherman; Karthikeyan Shanmugam; Alok Talekar; Vaibhav Rajan
>
> **摘要:** Delineating farm boundaries through segmentation of satellite images is a fundamental step in many agricultural applications. The task is particularly challenging for smallholder farms, where accurate delineation requires the use of high resolution (HR) imagery which are available only at low revisit frequencies (e.g., annually). To support more frequent (sub-) seasonal monitoring, HR images could be combined as references (ref) with low resolution (LR) images -- having higher revisit frequency (e.g., weekly) -- using reference-based super-resolution (Ref-SR) methods. However, current Ref-SR methods optimize perceptual quality and smooth over crucial features needed for downstream tasks, and are unable to meet the large scale-factor requirements for this task. Further, previous two-step approaches of SR followed by segmentation do not effectively utilize diverse satellite sources as inputs. We address these problems through a new approach, $\textbf{SEED-SR}$, which uses a combination of conditional latent diffusion models and large-scale multi-spectral, multi-source geo-spatial foundation models. Our key innovation is to bypass the explicit SR task in the pixel space and instead perform SR in a segmentation-aware latent space. This unique approach enables us to generate segmentation maps at an unprecedented 20$\times$ scale factor, and rigorous experiments on two large, real datasets demonstrate up to $\textbf{25.5}$ and $\textbf{12.9}$ relative improvement in instance and semantic segmentation metrics respectively over approaches based on state-of-the-art Ref-SR methods.
>
---
#### [new 087] Deep Learning-Based Regional White Matter Hyperintensity Mapping as a Robust Biomarker for Alzheimer's Disease
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像分析任务，旨在解决阿尔茨海默病诊断中白质高信号（WMH）空间分布被忽视的问题。作者提出深度学习方法实现WMH精准分割与区域量化，结合脑萎缩指标提升分类性能，AUC达0.97，揭示前部白质束为AD敏感区域。**

- **链接: [https://arxiv.org/pdf/2511.14588v1](https://arxiv.org/pdf/2511.14588v1)**

> **作者:** Julia Machnio; Mads Nielsen; Mostafa Mehdipour Ghazi
>
> **备注:** Accepted at SPIE - Medical Imaging Conference 2026
>
> **摘要:** White matter hyperintensities (WMH) are key imaging markers in cognitive aging, Alzheimer's disease (AD), and related dementias. Although automated methods for WMH segmentation have advanced, most provide only global lesion load and overlook their spatial distribution across distinct white matter regions. We propose a deep learning framework for robust WMH segmentation and localization, evaluated across public datasets and an independent Alzheimer's Disease Neuroimaging Initiative (ADNI) cohort. Our results show that the predicted lesion loads are in line with the reference WMH estimates, confirming the robustness to variations in lesion load, acquisition, and demographics. Beyond accurate segmentation, we quantify WMH load within anatomically defined regions and combine these measures with brain structure volumes to assess diagnostic value. Regional WMH volumes consistently outperform global lesion burden for disease classification, and integration with brain atrophy metrics further improves performance, reaching area under the curve (AUC) values up to 0.97. Several spatially distinct regions, particularly within anterior white matter tracts, are reproducibly associated with diagnostic status, indicating localized vulnerability in AD. These results highlight the added value of regional WMH quantification. Incorporating localized lesion metrics alongside atrophy markers may enhance early diagnosis and stratification in neurodegenerative disorders.
>
---
#### [new 088] Uni-Hema: Unified Model for Digital Hematopathology
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Uni-Hema，一个统一的多任务、多模态模型，用于数字血液病理分析。它解决现有模型无法跨疾病类别进行联合推理的问题，整合检测、分类、分割等任务，在46个数据集上训练，实现单细胞级可解释分析。**

- **链接: [https://arxiv.org/pdf/2511.13889v1](https://arxiv.org/pdf/2511.13889v1)**

> **作者:** Abdul Rehman; Iqra Rasool; Ayesha Imran; Mohsen Ali; Waqas Sultani
>
> **摘要:** Digital hematopathology requires cell-level analysis across diverse disease categories, including malignant disorders (e.g., leukemia), infectious conditions (e.g., malaria), and non-malignant red blood cell disorders (e.g., sickle cell disease). Whether single-task, vision-language, WSI-optimized, or single-cell hematology models, these approaches share a key limitation, they cannot provide unified, multi-task, multi-modal reasoning across the complexities of digital hematopathology. To overcome these limitations, we propose Uni-Hema, a multi-task, unified model for digital hematopathology integrating detection, classification, segmentation, morphology prediction, and reasoning across multiple diseases. Uni-Hema leverages 46 publicly available datasets, encompassing over 700K images and 21K question-answer pairs, and is built upon Hema-Former, a multimodal module that bridges visual and textual representations at the hierarchy level for the different tasks (detection, classification, segmentation, morphology, mask language modeling and visual question answer) at different granularity. Extensive experiments demonstrate that Uni-Hema achieves comparable or superior performance to train on a single-task and single dataset models, across diverse hematological tasks, while providing interpretable, morphologically relevant insights at the single-cell level. Our framework establishes a new standard for multi-task and multi-modal digital hematopathology. The code will be made publicly available.
>
---
#### [new 089] RISE: Single Static Radar-based Indoor Scene Understanding
- **分类: cs.CV**

- **简介: 论文提出RISE系统，利用单静态毫米波雷达实现隐私保护的室内场景理解，解决低分辨率导致的几何推理难题。通过建模多路径反射增强观测，并结合仿真到现实的扩散框架，首次实现雷达室内布局重建与物体检测，显著提升精度。**

- **链接: [https://arxiv.org/pdf/2511.14019v1](https://arxiv.org/pdf/2511.14019v1)**

> **作者:** Kaichen Zhou; Laura Dodds; Sayed Saad Afzal; Fadel Adib
>
> **摘要:** Robust and privacy-preserving indoor scene understanding remains a fundamental open problem. While optical sensors such as RGB and LiDAR offer high spatial fidelity, they suffer from severe occlusions and introduce privacy risks in indoor environments. In contrast, millimeter-wave (mmWave) radar preserves privacy and penetrates obstacles, but its inherently low spatial resolution makes reliable geometric reasoning difficult. We introduce RISE, the first benchmark and system for single-static-radar indoor scene understanding, jointly targeting layout reconstruction and object detection. RISE is built upon the key insight that multipath reflections, traditionally treated as noise, encode rich geometric cues. To exploit this, we propose a Bi-Angular Multipath Enhancement that explicitly models Angle-of-Arrival and Angle-of-Departure to recover secondary (ghost) reflections and reveal invisible structures. On top of these enhanced observations, a simulation-to-reality Hierarchical Diffusion framework transforms fragmented radar responses into complete layout reconstruction and object detection. Our benchmark contains 50,000 frames collected across 100 real indoor trajectories, forming the first large-scale dataset dedicated to radar-based indoor scene understanding. Extensive experiments show that RISE reduces the Chamfer Distance by 60% (down to 16 cm) compared to the state of the art in layout reconstruction, and delivers the first mmWave-based object detection, achieving 58% IoU. These results establish RISE as a new foundation for geometry-aware and privacy-preserving indoor scene understanding using a single static radar.
>
---
#### [new 090] UniSER: A Foundation Model for Unified Soft Effects Removal
- **分类: cs.CV**

- **简介: 论文提出UniSER，一个统一的图像修复基础模型，解决镜头耀斑、雾霾、阴影和反射等软效应退化问题。通过构建380万对数据集并训练扩散Transformer，实现单一框架内高效高保真修复，优于专用与通用模型。**

- **链接: [https://arxiv.org/pdf/2511.14183v1](https://arxiv.org/pdf/2511.14183v1)**

> **作者:** Jingdong Zhang; Lingzhi Zhang; Qing Liu; Mang Tik Chiu; Connelly Barnes; Yizhou Wang; Haoran You; Xiaoyang Liu; Yuqian Zhou; Zhe Lin; Eli Shechtman; Sohrab Amirghodsi; Xin Li; Wenping Wang; Xiaohang Zhan
>
> **摘要:** Digital images are often degraded by soft effects such as lens flare, haze, shadows, and reflections, which reduce aesthetics even though the underlying pixels remain partially visible. The prevailing works address these degradations in isolation, developing highly specialized, specialist models that lack scalability and fail to exploit the shared underlying essences of these restoration problems. While specialist models are limited, recent large-scale pretrained generalist models offer powerful, text-driven image editing capabilities. while recent general-purpose systems (e.g., GPT-4o, Flux Kontext, Nano Banana) require detailed prompts and often fail to achieve robust removal on these fine-grained tasks or preserve identity of the scene. Leveraging the common essence of soft effects, i.e., semi-transparent occlusions, we introduce a foundational versatile model UniSER, capable of addressing diverse degradations caused by soft effects within a single framework. Our methodology centers on curating a massive 3.8M-pair dataset to ensure robustness and generalization, which includes novel, physically-plausible data to fill critical gaps in public benchmarks, and a tailored training pipeline that fine-tunes a Diffusion Transformer to learn robust restoration priors from this diverse data, integrating fine-grained mask and strength controls. This synergistic approach allows UniSER to significantly outperform both specialist and generalist models, achieving robust, high-fidelity restoration in the wild.
>
---
#### [new 091] Can World Simulators Reason? Gen-ViRe: A Generative Visual Reasoning Benchmark
- **分类: cs.CV**

- **简介: 论文提出Gen-ViRe基准，评估视频生成模型的视觉推理能力，解决现有方法无法衡量多步规划、逻辑推理等认知能力的问题。通过六维24子任务量化分析，揭示视觉质量与推理深度的差距。**

- **链接: [https://arxiv.org/pdf/2511.13853v1](https://arxiv.org/pdf/2511.13853v1)**

> **作者:** Xinxin Liu; Zhaopan Xu; Kai Wang; Yong Jae Lee; Yuzhang Shang
>
> **备注:** 10 pages
>
> **摘要:** While Chain-of-Thought (CoT) prompting enables sophisticated symbolic reasoning in LLMs, it remains confined to discrete text and cannot simulate the continuous, physics-governed dynamics of the real world. Recent video generation models have emerged as potential world simulators through Chain-of-Frames (CoF) reasoning -- materializing thought as frame-by-frame visual sequences, with each frame representing a physically-grounded reasoning step. Despite compelling demonstrations, a challenge persists: existing benchmarks, focusing on fidelity or alignment, do not assess CoF reasoning and thus cannot measure core cognitive abilities in multi-step planning, algorithmic logic, or abstract pattern extrapolation. This evaluation void prevents systematic understanding of model capabilities and principled guidance for improvement. We introduce Gen-ViRe (Generative Visual Reasoning Benchmark), a framework grounded in cognitive science and real-world AI applications, which decomposes CoF reasoning into six cognitive dimensions -- from perceptual logic to abstract planning -- and 24 subtasks. Through multi-source data curation, minimal prompting protocols, and hybrid VLM-assisted evaluation with detailed criteria, Gen-ViRe delivers the first quantitative assessment of video models as reasoners. Our experiments on SOTA systems reveal substantial discrepancies between impressive visual quality and actual reasoning depth, establishing baselines and diagnostic tools to advance genuine world simulators.
>
---
#### [new 092] $A^2$GC: $A$symmetric $A$ggregation with Geometric Constraints for Locally Aggregated Descriptors
- **分类: cs.CV**

- **简介: 该论文针对视觉地点识别（VPR）任务，解决现有方法在特征聚合时对源和目标分布差异处理不足的问题。提出A²GC-VPR方法，通过非对称聚合与几何约束增强局部描述符匹配精度和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.14109v1](https://arxiv.org/pdf/2511.14109v1)**

> **作者:** Zhenyu Li; Tianyi Shang
>
> **备注:** 8 pages, 4figures
>
> **摘要:** Visual Place Recognition (VPR) aims to match query images against a database using visual cues. State-of-the-art methods aggregate features from deep backbones to form global descriptors. Optimal transport-based aggregation methods reformulate feature-to-cluster assignment as a transport problem, but the standard Sinkhorn algorithm symmetrically treats source and target marginals, limiting effectiveness when image features and cluster centers exhibit substantially different distributions. We propose an asymmetric aggregation VPR method with geometric constraints for locally aggregated descriptors, called $A^2$GC-VPR. Our method employs row-column normalization averaging with separate marginal calibration, enabling asymmetric matching that adapts to distributional discrepancies in visual place recognition. Geometric constraints are incorporated through learnable coordinate embeddings, computing compatibility scores fused with feature similarities, thereby promoting spatially proximal features to the same cluster and enhancing spatial awareness. Experimental results on MSLS, NordLand, and Pittsburgh datasets demonstrate superior performance, validating the effectiveness of our approach in improving matching accuracy and robustness.
>
---
#### [new 093] Learning Skill-Attributes for Transferable Assessment in Video
- **分类: cs.CV**

- **简介: 论文提出CrossTrainer方法，通过学习跨运动的技能属性（如平衡、控制）来实现视频技能评估的迁移。解决专家标注稀缺问题，提升模型在多运动场景下的泛化能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.13993v1](https://arxiv.org/pdf/2511.13993v1)**

> **作者:** Kumar Ashutosh; Kristen Grauman
>
> **备注:** NeurIPS 2025, Project webpage: https://vision.cs.utexas.edu/projects/CrossTrainer/
>
> **摘要:** Skill assessment from video entails rating the quality of a person's physical performance and explaining what could be done better. Today's models specialize for an individual sport, and suffer from the high cost and scarcity of expert-level supervision across the long tail of sports. Towards closing that gap, we explore transferable video representations for skill assessment. Our CrossTrainer approach discovers skill-attributes, such as balance, control, and hand positioning -- whose meaning transcends the boundaries of any given sport, then trains a multimodal language model to generate actionable feedback for a novel video, e.g., "lift hands more to generate more power" as well as its proficiency level, e.g., early expert. We validate the new model on multiple datasets for both cross-sport (transfer) and intra-sport (in-domain) settings, where it achieves gains up to 60% relative to the state of the art. By abstracting out the shared behaviors indicative of human skill, the proposed video representation generalizes substantially better than an array of existing techniques, enriching today's multimodal large language models.
>
---
#### [new 094] DeCo-VAE: Learning Compact Latents for Video Reconstruction via Decoupled Representation
- **分类: cs.CV; cs.LG; cs.MM**

- **简介: 论文提出DeCo-VAE，用于视频重建任务，解决现有VAE因忽略帧间相似性导致的冗余潜变量建模问题。通过分解视频为关键帧、运动和残差三部分，分别学习潜变量，并采用解耦训练策略提升稳定性和准确性。**

- **链接: [https://arxiv.org/pdf/2511.14530v1](https://arxiv.org/pdf/2511.14530v1)**

> **作者:** Xiangchen Yin; Jiahui Yuan; Zhangchi Hu; Wenzhang Sun; Jie Chen; Xiaozhen Qiao; Hao Li; Xiaoyan Sun
>
> **摘要:** Existing video Variational Autoencoders (VAEs) generally overlook the similarity between frame contents, leading to redundant latent modeling. In this paper, we propose decoupled VAE (DeCo-VAE) to achieve compact latent representation. Instead of encoding RGB pixels directly, we decompose video content into distinct components via explicit decoupling: keyframe, motion and residual, and learn dedicated latent representation for each. To avoid cross-component interference, we design dedicated encoders for each decoupled component and adopt a shared 3D decoder to maintain spatiotemporal consistency during reconstruction. We further utilize a decoupled adaptation strategy that freezes partial encoders while training the others sequentially, ensuring stable training and accurate learning of both static and dynamic features. Extensive quantitative and qualitative experiments demonstrate that DeCo-VAE achieves superior video reconstruction performance.
>
---
#### [new 095] Measurement-Constrained Sampling for Text-Prompted Blind Face Restoration
- **分类: cs.CV**

- **简介: 该论文属于盲人脸复原任务，旨在解决低质量输入对应多个合理高清重建结果的“一对多”问题。作者提出测量约束采样方法，通过构建逆问题并结合文本提示引导扩散模型生成多样化且符合提示的复原结果。**

- **链接: [https://arxiv.org/pdf/2511.14213v1](https://arxiv.org/pdf/2511.14213v1)**

> **作者:** Wenjie Li; Yulun Zhang; Guangwei Gao; Heng Guo; Zhanyu Ma
>
> **摘要:** Blind face restoration (BFR) may correspond to multiple plausible high-quality (HQ) reconstructions under extremely low-quality (LQ) inputs. However, existing methods typically produce deterministic results, struggling to capture this one-to-many nature. In this paper, we propose a Measurement-Constrained Sampling (MCS) approach that enables diverse LQ face reconstructions conditioned on different textual prompts. Specifically, we formulate BFR as a measurement-constrained generative task by constructing an inverse problem through controlled degradations of coarse restorations, which allows posterior-guided sampling within text-to-image diffusion. Measurement constraints include both Forward Measurement, which ensures results align with input structures, and Reverse Measurement, which produces projection spaces, ensuring that the solution can align with various prompts. Experiments show that our MCS can generate prompt-aligned results and outperforms existing BFR methods. Codes will be released after acceptance.
>
---
#### [new 096] LSP-YOLO: A Lightweight Single-Stage Network for Sitting Posture Recognition on Embedded Devices
- **分类: cs.CV; cs.AI**

- **简介: 论文提出LSP-YOLO，一种轻量级单阶段网络用于嵌入式设备上的坐姿识别任务，解决现有方法计算复杂、实时性差的问题。通过改进模块设计与直接关键点分类，实现高精度与高效推理。**

- **链接: [https://arxiv.org/pdf/2511.14322v1](https://arxiv.org/pdf/2511.14322v1)**

> **作者:** Nanjun Li; Ziyue Hao; Quanqiang Wang; Xuanyin Wang
>
> **备注:** Submitted to Engineering Applications of Artificial Intelligence (EAAI)
>
> **摘要:** With the rise in sedentary behavior, health problems caused by poor sitting posture have drawn increasing attention. Most existing methods, whether using invasive sensors or computer vision, rely on two-stage pipelines, which result in high intrusiveness, intensive computation, and poor real-time performance on embedded edge devices. Inspired by YOLOv11-Pose, a lightweight single-stage network for sitting posture recognition on embedded edge devices termed LSP-YOLO was proposed. By integrating partial convolution(PConv) and Similarity-Aware Activation Module(SimAM), a lightweight module, Light-C3k2, was designed to reduce computational cost while maintaining feature extraction capability. In the recognition head, keypoints were directly mapped to posture classes through pointwise convolution, and intermediate supervision was employed to enable efficient fusion of pose estimation and classification. Furthermore, a dataset containing 5,000 images across six posture categories was constructed for model training and testing. The smallest trained model, LSP-YOLO-n, achieved 94.2% accuracy and 251 Fps on personal computer(PC) with a model size of only 1.9 MB. Meanwhile, real-time and high-accuracy inference under constrained computational resources was demonstrated on the SV830C + GC030A platform. The proposed approach is characterized by high efficiency, lightweight design and deployability, making it suitable for smart classrooms, rehabilitation, and human-computer interaction applications.
>
---
#### [new 097] RepAir: A Framework for Airway Segmentation and Discontinuity Correction in CT
- **分类: cs.CV**

- **简介: 该论文提出RepAir框架，用于CT图像中气道分割与断点修复。针对现有方法分割结果不连续的问题，结合深度学习与拓扑修正，提升气道完整性和解剖一致性，优于现有3D U-Net方法。**

- **链接: [https://arxiv.org/pdf/2511.14649v1](https://arxiv.org/pdf/2511.14649v1)**

> **作者:** John M. Oyer; Ali Namvar; Benjamin A. Hoff; Wassim W. Labaki; Ella A. Kazerooni; Charles R. Hatt; Fernando J. Martinez; MeiLan K. Han; Craig J. Galbán; Sundaresh Ram
>
> **备注:** 4 pages, 3 figures, 1 table. Preprint submitted to SSIAI 2026 Conference on November 17, 2025
>
> **摘要:** Accurate airway segmentation from chest computed tomography (CT) scans is essential for quantitative lung analysis, yet manual annotation is impractical and many automated U-Net-based methods yield disconnected components that hinder reliable biomarker extraction. We present RepAir, a three-stage framework for robust 3D airway segmentation that combines an nnU-Net-based network with anatomically informed topology correction. The segmentation network produces an initial airway mask, after which a skeleton-based algorithm identifies potential discontinuities and proposes reconnections. A 1D convolutional classifier then determines which candidate links correspond to true anatomical branches versus false or obstructed paths. We evaluate RepAir on two distinct datasets: ATM'22, comprising annotated CT scans from predominantly healthy subjects and AeroPath, encompassing annotated scans with severe airway pathology. Across both datasets, RepAir outperforms existing 3D U-Net-based approaches such as Bronchinet and NaviAirway on both voxel-level and topological metrics, and produces more complete and anatomically consistent airway trees while maintaining high segmentation accuracy.
>
---
#### [new 098] Hybrid Convolution Neural Network Integrated with Pseudo-Newton Boosting for Lumbar Spine Degeneration Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对腰椎退行性病变的自动检测任务，提出一种融合EfficientNet与VGG19的混合神经网络模型，引入伪牛顿提升层和稀疏特征压缩层，提升特征选择与表示能力，显著优于传统迁移学习方法。**

- **链接: [https://arxiv.org/pdf/2511.13877v1](https://arxiv.org/pdf/2511.13877v1)**

> **作者:** Pandiyaraju V; Abishek Karthik; Jaspin K; Kannan A; Jaime Lloret
>
> **摘要:** This paper proposes a new enhanced model architecture to perform classification of lumbar spine degeneration with DICOM images while using a hybrid approach, integrating EfficientNet and VGG19 together with custom-designed components. The proposed model is differentiated from traditional transfer learning methods as it incorporates a Pseudo-Newton Boosting layer along with a Sparsity-Induced Feature Reduction Layer that forms a multi-tiered framework, further improving feature selection and representation. The Pseudo-Newton Boosting layer makes smart variations of feature weights, with more detailed anatomical features, which are mostly left out in a transfer learning setup. In addition, the Sparsity-Induced Layer removes redundancy for learned features, producing lean yet robust representations for pathology in the lumbar spine. This architecture is novel as it overcomes the constraints in the traditional transfer learning approach, especially in the high-dimensional context of medical images, and achieves a significant performance boost, reaching a precision of 0.9, recall of 0.861, F1 score of 0.88, loss of 0.18, and an accuracy of 88.1%, compared to the baseline model, EfficientNet. This work will present the architectures, preprocessing pipeline, and experimental results. The results contribute to the development of automated diagnostic tools for medical images.
>
---
#### [new 099] Revisiting Data Scaling Law for Medical Segmentation
- **分类: cs.CV**

- **简介: 论文研究医学图像分割中的数据缩放规律，解决标注数据稀缺问题。通过分析15个任务和4种模态，提出基于图像配准的形变增强方法，提升数据利用效率，加速模型收敛，优于传统幂律趋势。**

- **链接: [https://arxiv.org/pdf/2511.13883v1](https://arxiv.org/pdf/2511.13883v1)**

> **作者:** Yuetan Chu; Zhongyi Han; Gongning Luo; Xin Gao
>
> **摘要:** The population loss of trained deep neural networks often exhibits power law scaling with the size of the training dataset, guiding significant performance advancements in deep learning applications. In this study, we focus on the scaling relationship with data size in the context of medical anatomical segmentation, a domain that remains underexplored. We analyze scaling laws for anatomical segmentation across 15 semantic tasks and 4 imaging modalities, demonstrating that larger datasets significantly improve segmentation performance, following similar scaling trends. Motivated by the topological isomorphism in images sharing anatomical structures, we evaluate the impact of deformation-guided augmentation strategies on data scaling laws, specifically random elastic deformation and registration-guided deformation. We also propose a novel, scalable image augmentation approach that generates diffeomorphic mappings from geodesic subspace based on image registration to introduce realistic deformation. Our experimental results demonstrate that both registered and generated deformation-based augmentation considerably enhance data utilization efficiency. The proposed generated deformation method notably achieves superior performance and accelerated convergence, surpassing standard power law scaling trends without requiring additional data. Overall, this work provides insights into the understanding of segmentation scalability and topological variation impact in medical imaging, thereby leading to more efficient model development with reduced annotation and computational costs.
>
---
#### [new 100] AdaTok: Adaptive Token Compression with Object-Aware Representations for Efficient Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **简介: 论文提出AdaTok，针对多模态大模型图像token冗余问题，通过对象感知的自适应压缩策略减少token数量，提升计算效率并更贴近人类视觉认知，实验表明仅用10% token即可达到96%性能。**

- **链接: [https://arxiv.org/pdf/2511.14169v1](https://arxiv.org/pdf/2511.14169v1)**

> **作者:** Xinliang Zhang; Lei Zhu; Hangzhou He; Shuang Zeng; Ourui Fu; Jiakui Hu; Zhengjian Yao; Yanye Lu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated substantial value in unified text-image understanding and reasoning, primarily by converting images into sequences of patch-level tokens that align with their architectural paradigm. However, patch-level tokenization leads to a quadratic growth in image tokens, burdening MLLMs' understanding and reasoning with enormous computation and memory. Additionally, the traditional patch-wise scanning tokenization workflow misaligns with the human vision cognition system, further leading to hallucination and computational redundancy. To address this issue, we propose an object-level token merging strategy for Adaptive Token compression, revealing the consistency with human vision system. The experiments are conducted on multiple comprehensive benchmarks, which show that our approach averagely, utilizes only 10% tokens while achieving almost 96% of the vanilla model's performance. More extensive experimental results in comparison with relevant works demonstrate the superiority of our method in balancing compression ratio and performance. Our code will be available.
>
---
#### [new 101] Agentic Video Intelligence: A Flexible Framework for Advanced Video Exploration and Understanding
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Agentic Video Intelligence（AVI），解决视频理解中推理能力不足与可解释性差的问题。通过三阶段推理流程、结构化知识库和开源模型集成，实现灵活、无需训练的视频分析，提升准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2511.14446v1](https://arxiv.org/pdf/2511.14446v1)**

> **作者:** Hong Gao; Yiming Bao; Xuezhen Tu; Yutong Xu; Yue Jin; Yiyang Mu; Bin Zhong; Linan Yue; Min-Ling Zhang
>
> **摘要:** Video understanding requires not only visual recognition but also complex reasoning. While Vision-Language Models (VLMs) demonstrate impressive capabilities, they typically process videos largely in a single-pass manner with limited support for evidence revisit and iterative refinement. While recently emerging agent-based methods enable long-horizon reasoning, they either depend heavily on expensive proprietary models or require extensive agentic RL training. To overcome these limitations, we propose Agentic Video Intelligence (AVI), a flexible and training-free framework that can mirror human video comprehension through system-level design and optimization. AVI introduces three key innovations: (1) a human-inspired three-phase reasoning process (Retrieve-Perceive-Review) that ensures both sufficient global exploration and focused local analysis, (2) a structured video knowledge base organized through entity graphs, along with multi-granularity integrated tools, constituting the agent's interaction environment, and (3) an open-source model ensemble combining reasoning LLMs with lightweight base CV models and VLM, eliminating dependence on proprietary APIs or RL training. Experiments on LVBench, VideoMME-Long, LongVideoBench, and Charades-STA demonstrate that AVI achieves competitive performance while offering superior interpretability.
>
---
#### [new 102] NeuralBoneReg: A Novel Self-Supervised Method for Robust and Accurate Multi-Modal Bone Surface Registration
- **分类: cs.CV**

- **简介: 论文提出NeuralBoneReg，一种自监督的多模态骨表面配准方法，用于解决术前与术中影像间因模态差异导致的配准难题。通过隐式神经距离场和MLP模块实现跨模态精准对齐，在多个数据集上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.14286v1](https://arxiv.org/pdf/2511.14286v1)**

> **作者:** Luohong Wu; Matthias Seibold; Nicola A. Cavalcanti; Yunke Ao; Roman Flepp; Aidana Massalimova; Lilian Calvet; Philipp Fürnstahl
>
> **摘要:** In computer- and robot-assisted orthopedic surgery (CAOS), patient-specific surgical plans derived from preoperative imaging define target locations and implant trajectories. During surgery, these plans must be accurately transferred, relying on precise cross-registration between preoperative and intraoperative data. However, substantial modality heterogeneity across imaging modalities makes this registration challenging and error-prone. Robust, automatic, and modality-agnostic bone surface registration is therefore clinically important. We propose NeuralBoneReg, a self-supervised, surface-based framework that registers bone surfaces using 3D point clouds as a modality-agnostic representation. NeuralBoneReg includes two modules: an implicit neural unsigned distance field (UDF) that learns the preoperative bone model, and an MLP-based registration module that performs global initialization and local refinement by generating transformation hypotheses to align the intraoperative point cloud with the neural UDF. Unlike SOTA supervised methods, NeuralBoneReg operates in a self-supervised manner, without requiring inter-subject training data. We evaluated NeuralBoneReg against baseline methods on two publicly available multi-modal datasets: a CT-ultrasound dataset of the fibula and tibia (UltraBones100k) and a CT-RGB-D dataset of spinal vertebrae (SpineDepth). The evaluation also includes a newly introduced CT--ultrasound dataset of cadaveric subjects containing femur and pelvis (UltraBones-Hip), which will be made publicly available. NeuralBoneReg matches or surpasses existing methods across all datasets, achieving mean RRE/RTE of 1.68°/1.86 mm on UltraBones100k, 1.88°/1.89 mm on UltraBones-Hip, and 3.79°/2.45 mm on SpineDepth. These results demonstrate strong generalizability across anatomies and modalities, providing robust and accurate cross-modal alignment for CAOS.
>
---
#### [new 103] Language as an Anchor: Preserving Relative Visual Geometry for Domain Incremental Learning
- **分类: cs.CV**

- **简介: 论文提出LAVA框架解决领域增量学习中的知识遗忘与干扰问题，通过文本锚定的相对视觉几何保持机制，实现跨域知识迁移与稳定学习。**

- **链接: [https://arxiv.org/pdf/2511.14401v1](https://arxiv.org/pdf/2511.14401v1)**

> **作者:** Shuyi Geng; Tao Zhou; Yi Zhou
>
> **摘要:** A key challenge in Domain Incremental Learning (DIL) is to continually learn under shifting distributions while preserving knowledge from previous domains. Existing methods face a fundamental dilemma. On one hand, projecting all domains into a single unified visual space leads to inter-domain interference and semantic distortion, as large shifts may vary with not only visual appearance but also underlying semantics. On the other hand, isolating domain-specific parameters causes knowledge fragmentation, creating "knowledge islands" that hamper knowledge reuse and exacerbate forgetting. To address this issue, we propose LAVA (Language-Anchored Visual Alignment), a novel DIL framework that replaces direct feature alignment with relative alignment driven by a text-based reference anchor. LAVA guides the visual representations of each incoming domain to preserve a consistent relative geometry, which is defined by mirroring the pairwise semantic similarities between the class names. This anchored geometric structure acts as a bridge across domains, enabling the retrieval of class-aware prior knowledge and facilitating robust feature aggregation. Extensive experiments on standard DIL benchmarks demonstrate that LAVA achieves significant performance improvements over state-of-the-arts. Code is available at https://github.com/ShuyiGeng/LAVA.
>
---
#### [new 104] A Trajectory-free Crash Detection Framework with Generative Approach and Segment Map Diffusion
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出一种无轨迹的碰撞检测框架，通过生成式模型和路段地图扩散技术，直接利用路段级交通动态数据实现实时碰撞检测，解决传统方法依赖轨迹获取的问题。**

- **链接: [https://arxiv.org/pdf/2511.13795v1](https://arxiv.org/pdf/2511.13795v1)**

> **作者:** Weiying Shen; Hao Yu; Yu Dong; Pan Liu; Yu Han; Xin Wen
>
> **备注:** To be presented at TRB 2026 (TRBAM-26-01711) and a revised version will be submitted to Transportation Research Part C: Emerging Technologies
>
> **摘要:** Real-time crash detection is essential for developing proactive safety management strategy and enhancing overall traffic efficiency. To address the limitations associated with trajectory acquisition and vehicle tracking, road segment maps recording the individual-level traffic dynamic data were directly served in crash detection. A novel two-stage trajectory-free crash detection framework, was present to generate the rational future road segment map and identify crashes. The first-stage diffusion-based segment map generation model, Mapfusion, conducts a noisy-to-normal process that progressively adds noise to the road segment map until the map is corrupted to pure Gaussian noise. The denoising process is guided by sequential embedding components capturing the temporal dynamics of segment map sequences. Furthermore, the generation model is designed to incorporate background context through ControlNet to enhance generation control. Crash detection is achieved by comparing the monitored segment map with the generations from diffusion model in second stage. Trained on non-crash vehicle motion data, Mapfusion successfully generates realistic road segment evolution maps based on learned motion patterns and remains robust across different sampling intervals. Experiments on real-world crashes indicate the effectiveness of the proposed two-stage method in accurately detecting crashes.
>
---
#### [new 105] Seeing Beyond the Image: ECG and Anatomical Knowledge-Guided Myocardial Scar Segmentation from Late Gadolinium-Enhanced Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在提高心肌瘢痕的分割精度。针对LGE-MRI图像中对比度不一和伪影导致的挑战，提出融合ECG电生理信息与解剖先验知识的多模态框架，并设计时序感知特征融合机制，显著提升分割性能。**

- **链接: [https://arxiv.org/pdf/2511.14702v1](https://arxiv.org/pdf/2511.14702v1)**

> **作者:** Farheen Ramzan; Yusuf Kiberu; Nikesh Jathanna; Meryem Jabrane; Vicente Grau; Shahnaz Jamil-Copley; Richard H. Clayton; Chen; Chen
>
> **摘要:** Accurate segmentation of myocardial scar from late gadolinium enhanced (LGE) cardiac MRI is essential for evaluating tissue viability, yet remains challenging due to variable contrast and imaging artifacts. Electrocardiogram (ECG) signals provide complementary physiological information, as conduction abnormalities can help localize or suggest scarred myocardial regions. In this work, we propose a novel multimodal framework that integrates ECG-derived electrophysiological information with anatomical priors from the AHA-17 atlas for physiologically consistent LGE-based scar segmentation. As ECGs and LGE-MRIs are not acquired simultaneously, we introduce a Temporal Aware Feature Fusion (TAFF) mechanism that dynamically weights and fuses features based on their acquisition time difference. Our method was evaluated on a clinical dataset and achieved substantial gains over the state-of-the-art image-only baseline (nnU-Net), increasing the average Dice score for scars from 0.6149 to 0.8463 and achieving high performance in both precision (0.9115) and sensitivity (0.9043). These results show that integrating physiological and anatomical knowledge allows the model to "see beyond the image", setting a new direction for robust and physiologically grounded cardiac scar segmentation.
>
---
#### [new 106] Online Data Curation for Object Detection via Marginal Contributions to Dataset-level Average Precision
- **分类: cs.CV**

- **简介: 论文提出DetGain，一种面向目标检测的在线数据筛选方法，通过估计每张图像对整体AP的边际贡献，动态选择高价值样本，提升模型性能并增强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.14197v1](https://arxiv.org/pdf/2511.14197v1)**

> **作者:** Zitang Sun; Masakazu Yoshimura; Junji Otsuka; Atsushi Irie; Takeshi Ohashi
>
> **备注:** preprint version, under review
>
> **摘要:** High-quality data has become a primary driver of progress under scale laws, with curated datasets often outperforming much larger unfiltered ones at lower cost. Online data curation extends this idea by dynamically selecting training samples based on the model's evolving state. While effective in classification and multimodal learning, existing online sampling strategies rarely extend to object detection because of its structural complexity and domain gaps. We introduce DetGain, an online data curation method specifically for object detection that estimates the marginal perturbation of each image to dataset-level Average Precision (AP) based on its prediction quality. By modeling global score distributions, DetGain efficiently estimates the global AP change and computes teacher-student contribution gaps to select informative samples at each iteration. The method is architecture-agnostic and minimally intrusive, enabling straightforward integration into diverse object detection architectures. Experiments on the COCO dataset with multiple representative detectors show consistent improvements in accuracy. DetGain also demonstrates strong robustness under low-quality data and can be effectively combined with knowledge distillation techniques to further enhance performance, highlighting its potential as a general and complementary strategy for data-efficient object detection.
>
---
#### [new 107] MVI-Bench: A Comprehensive Benchmark for Evaluating Robustness to Misleading Visual Inputs in LVLMs
- **分类: cs.CV**

- **简介: 该论文提出MVI-Bench，首个针对大视觉语言模型（LVLMs）在误导性视觉输入下鲁棒性的综合评测基准。解决现有评估忽略视觉误导问题的不足，通过三个层级的视觉误导分类构建1248个VQA任务，并引入MVI-Sensitivity指标，揭示LVLMs的脆弱性并提供改进方向。**

- **链接: [https://arxiv.org/pdf/2511.14159v1](https://arxiv.org/pdf/2511.14159v1)**

> **作者:** Huiyi Chen; Jiawei Peng; Dehai Min; Changchang Sun; Kaijie Chen; Yan Yan; Xu Yang; Lu Cheng
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Evaluating the robustness of Large Vision-Language Models (LVLMs) is essential for their continued development and responsible deployment in real-world applications. However, existing robustness benchmarks typically focus on hallucination or misleading textual inputs, while largely overlooking the equally critical challenge posed by misleading visual inputs in assessing visual understanding. To fill this important gap, we introduce MVI-Bench, the first comprehensive benchmark specially designed for evaluating how Misleading Visual Inputs undermine the robustness of LVLMs. Grounded in fundamental visual primitives, the design of MVI-Bench centers on three hierarchical levels of misleading visual inputs: Visual Concept, Visual Attribute, and Visual Relationship. Using this taxonomy, we curate six representative categories and compile 1,248 expertly annotated VQA instances. To facilitate fine-grained robustness evaluation, we further introduce MVI-Sensitivity, a novel metric that characterizes LVLM robustness at a granular level. Empirical results across 18 state-of-the-art LVLMs uncover pronounced vulnerabilities to misleading visual inputs, and our in-depth analyses on MVI-Bench provide actionable insights that can guide the development of more reliable and robust LVLMs. The benchmark and codebase can be accessed at https://github.com/chenyil6/MVI-Bench.
>
---
#### [new 108] Known Meets Unknown: Mitigating Overconfidence in Open Set Recognition
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文研究开放集识别任务，针对模型对语义相似未知样本产生过自信误判的问题，提出双模块框架：通过参数扰动估计不确定性，并用学习型分类器提升已知与未知类别的区分度，从而改善性能。**

- **链接: [https://arxiv.org/pdf/2511.13775v1](https://arxiv.org/pdf/2511.13775v1)**

> **作者:** Dongdong Zhao; Ranxin Fang; Changtian Song; Zhihui Liu; Jianwen Xiang
>
> **备注:** 8 pages, 5 figures, 2 tables
>
> **摘要:** Open Set Recognition (OSR) requires models not only to accurately classify known classes but also to effectively reject unknown samples. However, when unknown samples are semantically similar to known classes, inter-class overlap in the feature space often causes models to assign unjustifiably high confidence to them, leading to misclassification as known classes -- a phenomenon known as overconfidence. This overconfidence undermines OSR by blurring the decision boundary between known and unknown classes. To address this issue, we propose a framework that explicitly mitigates overconfidence caused by inter-class overlap. The framework consists of two components: a perturbation-based uncertainty estimation module, which applies controllable parameter perturbations to generate diverse predictions and quantify predictive uncertainty, and an unknown detection module with distinct learning-based classifiers, implemented as a two-stage procedure, which leverages the estimated uncertainty to improve discrimination between known and unknown classes, thereby enhancing OSR performance. Experimental results on three public datasets show that the proposed framework achieves superior performance over existing OSR methods.
>
---
#### [new 109] Stage Aware Diagnosis of Diabetic Retinopathy via Ordinal Regression
- **分类: cs.CV**

- **简介: 论文提出基于序数回归的糖尿病视网膜病变分期诊断方法，解决DR分级准确率低的问题。利用预处理增强图像特征，结合APTOS-2019数据集训练模型，QWK达0.8992，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.14398v1](https://arxiv.org/pdf/2511.14398v1)**

> **作者:** Saksham Kumar; D Sridhar Aditya; T Likhil Kumar; Thulasi Bikku; Srinivasarao Thota; Chandan Kumar
>
> **备注:** Submitted to Confluence 2026, Amity University
>
> **摘要:** Diabetic Retinopathy (DR) has emerged as a major cause of preventable blindness in recent times. With timely screening and intervention, the condition can be prevented from causing irreversible damage. The work introduces a state-of-the-art Ordinal Regression-based DR Detection framework that uses the APTOS-2019 fundus image dataset. A widely accepted combination of preprocessing methods: Green Channel (GC) Extraction, Noise Masking, and CLAHE, was used to isolate the most relevant features for DR classification. Model performance was evaluated using the Quadratic Weighted Kappa, with a focus on agreement between results and clinical grading. Our Ordinal Regression approach attained a QWK score of 0.8992, setting a new benchmark on the APTOS dataset.
>
---
#### [new 110] NeuralSSD: A Neural Solver for Signed Distance Surface Reconstruction
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 论文提出NeuralSSD，用于从点云数据重建高质量3D隐式表面。针对现有方法难以精确拟合输入点云的问题，设计新能量方程与卷积网络，提升重建精度与稳定性，在ShapeNet和Matterport上达SOTA效果。**

- **链接: [https://arxiv.org/pdf/2511.14283v1](https://arxiv.org/pdf/2511.14283v1)**

> **作者:** Zi-Chen Xi; Jiahui Huang; Hao-Xiang Chen; Francis Williams; Qun-Ce Xu; Tai-Jiang Mu; Shi-Min Hu
>
> **备注:** Under review
>
> **摘要:** We proposed a generalized method, NeuralSSD, for reconstructing a 3D implicit surface from the widely-available point cloud data. NeuralSSD is a solver-based on the neural Galerkin method, aimed at reconstructing higher-quality and accurate surfaces from input point clouds. Implicit method is preferred due to its ability to accurately represent shapes and its robustness in handling topological changes. However, existing parameterizations of implicit fields lack explicit mechanisms to ensure a tight fit between the surface and input data. To address this, we propose a novel energy equation that balances the reliability of point cloud information. Additionally, we introduce a new convolutional network that learns three-dimensional information to achieve superior optimization results. This approach ensures that the reconstructed surface closely adheres to the raw input points and infers valuable inductive biases from point clouds, resulting in a highly accurate and stable surface reconstruction. NeuralSSD is evaluated on a variety of challenging datasets, including the ShapeNet and Matterport datasets, and achieves state-of-the-art results in terms of both surface reconstruction accuracy and generalizability.
>
---
#### [new 111] DoGCLR: Dominance-Game Contrastive Learning Network for Skeleton-Based Action Recognition
- **分类: cs.CV**

- **简介: 该论文提出DoGCLR框架，用于骨架动作识别任务，解决现有方法均匀处理骨架区域和负样本选择不佳的问题。通过博弈论建模正负样本构建，结合时空定位与熵驱动的内存管理策略，提升运动信息保留与对比学习效果。**

- **链接: [https://arxiv.org/pdf/2511.14179v1](https://arxiv.org/pdf/2511.14179v1)**

> **作者:** Yanshan Li; Ke Ma; Miaomiao Wei; Linhui Dai
>
> **备注:** 14 pages, 7 figures, journal
>
> **摘要:** Existing self-supervised contrastive learning methods for skeleton-based action recognition often process all skeleton regions uniformly, and adopt a first-in-first-out (FIFO) queue to store negative samples, which leads to motion information loss and non-optimal negative sample selection. To address these challenges, this paper proposes Dominance-Game Contrastive Learning network for skeleton-based action Recognition (DoGCLR), a self-supervised framework based on game theory. DoGCLR models the construction of positive and negative samples as a dynamic Dominance Game, where both sample types interact to reach an equilibrium that balances semantic preservation and discriminative strength. Specifically, a spatio-temporal dual weight localization mechanism identifies key motion regions and guides region-wise augmentations to enhance motion diversity while maintaining semantics. In parallel, an entropy-driven dominance strategy manages the memory bank by retaining high entropy (hard) negatives and replacing low-entropy (weak) ones, ensuring consistent exposure to informative contrastive signals. Extensive experiments are conducted on NTU RGB+D and PKU-MMD datasets. On NTU RGB+D 60 X-Sub/X-View, DoGCLR achieves 81.1%/89.4% accuracy, and on NTU RGB+D 120 X-Sub/X-Set, DoGCLR achieves 71.2%/75.5% accuracy, surpassing state-of-the-art methods by 0.1%, 2.7%, 1.1%, and 2.3%, respectively. On PKU-MMD Part I/Part II, DoGCLR performs comparably to the state-of-the-art methods and achieves a 1.9% higher accuracy on Part II, highlighting its strong robustness on more challenging scenarios.
>
---
#### [new 112] SAM-Fed: SAM-Guided Federated Semi-Supervised Learning for Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出SAM-Fed框架，用于医疗图像分割任务，解决联邦半监督学习中伪标签可靠性低和客户端计算资源有限的问题。通过高容量模型指导轻量客户端，结合知识蒸馏与自适应一致机制提升分割精度。**

- **链接: [https://arxiv.org/pdf/2511.14302v1](https://arxiv.org/pdf/2511.14302v1)**

> **作者:** Sahar Nasirihaghighi; Negin Ghamsarian; Yiping Li; Marcel Breeuwer; Raphael Sznitman; Klaus Schoeffmann
>
> **摘要:** Medical image segmentation is clinically important, yet data privacy and the cost of expert annotation limit the availability of labeled data. Federated semi-supervised learning (FSSL) offers a solution but faces two challenges: pseudo-label reliability depends on the strength of local models, and client devices often require compact or heterogeneous architectures due to limited computational resources. These constraints reduce the quality and stability of pseudo-labels, while large models, though more accurate, cannot be trained or used for routine inference on client devices. We propose SAM-Fed, a federated semi-supervised framework that leverages a high-capacity segmentation foundation model to guide lightweight clients during training. SAM-Fed combines dual knowledge distillation with an adaptive agreement mechanism to refine pixel-level supervision. Experiments on skin lesion and polyp segmentation across homogeneous and heterogeneous settings show that SAM-Fed consistently outperforms state-of-the-art FSSL methods.
>
---
#### [new 113] Temporal Realism Evaluation of Generated Videos Using Compressed-Domain Motion Vectors
- **分类: cs.CV**

- **简介: 论文提出基于压缩域运动矢量的视频时序真实性评估方法，解决生成视频运动不真实的问题。通过计算真实与生成视频运动矢量统计差异，量化时序缺陷，并验证其在下游任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.13897v1](https://arxiv.org/pdf/2511.13897v1)**

> **作者:** Mert Onur Cakiroglu; Idil Bilge Altun; Zhihe Lu; Mehmet Dalkilic; Hasan Kurban
>
> **摘要:** Temporal realism remains a central weakness of current generative video models, as most evaluation metrics prioritize spatial appearance and offer limited sensitivity to motion. We introduce a scalable, model-agnostic framework that assesses temporal behavior using motion vectors (MVs) extracted directly from compressed video streams. Codec-generated MVs from standards such as H.264 and HEVC provide lightweight, resolution-consistent descriptors of motion dynamics. We quantify realism by computing Kullback-Leibler, Jensen-Shannon, and Wasserstein divergences between MV statistics of real and generated videos. Experiments on the GenVidBench dataset containing videos from eight state-of-the-art generators reveal systematic discrepancies from real motion: entropy-based divergences rank Pika and SVD as closest to real videos, MV-sum statistics favor VC2 and Text2Video-Zero, and CogVideo shows the largest deviations across both measures. Visualizations of MV fields and class-conditional motion heatmaps further reveal center bias, sparse and piecewise constant flows, and grid-like artifacts that frame-level metrics do not capture. Beyond evaluation, we investigate MV-RGB fusion through channel concatenation, cross-attention, joint embedding, and a motion-aware fusion module. Incorporating MVs improves downstream classification across ResNet, I3D, and TSN backbones, with ResNet-18 and ResNet-34 reaching up to 97.4% accuracy and I3D achieving 99.0% accuracy on real-versus-generated discrimination. These findings demonstrate that compressed-domain MVs provide an effective temporal signal for diagnosing motion defects in generative videos and for strengthening temporal reasoning in discriminative models. The implementation is available at: https://github.com/KurbanIntelligenceLab/Motion-Vector-Learning
>
---
#### [new 114] Parameter Aware Mamba Model for Multi-task Dense Prediction
- **分类: cs.CV**

- **简介: 论文提出PAMM框架，用于多任务密集预测任务，解决任务间交互建模问题。通过双状态空间参数专家和多方向希尔伯特扫描，增强任务关联与特征感知能力，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.14503v1](https://arxiv.org/pdf/2511.14503v1)**

> **作者:** Xinzhuo Yu; Yunzhi Zhuge; Sitong Gong; Lu Zhang; Pingping Zhang; Huchuan Lu
>
> **备注:** Accepted to IEEE Transactions on Cybernetics
>
> **摘要:** Understanding the inter-relations and interactions between tasks is crucial for multi-task dense prediction. Existing methods predominantly utilize convolutional layers and attention mechanisms to explore task-level interactions. In this work, we introduce a novel decoder-based framework, Parameter Aware Mamba Model (PAMM), specifically designed for dense prediction in multi-task learning setting. Distinct from approaches that employ Transformers to model holistic task relationships, PAMM leverages the rich, scalable parameters of state space models to enhance task interconnectivity. It features dual state space parameter experts that integrate and set task-specific parameter priors, capturing the intrinsic properties of each task. This approach not only facilitates precise multi-task interactions but also allows for the global integration of task priors through the structured state space sequence model (S4). Furthermore, we employ the Multi-Directional Hilbert Scanning method to construct multi-angle feature sequences, thereby enhancing the sequence model's perceptual capabilities for 2D data. Extensive experiments on the NYUD-v2 and PASCAL-Context benchmarks demonstrate the effectiveness of our proposed method. Our code is available at https://github.com/CQC-gogopro/PAMM.
>
---
#### [new 115] Flood-LDM: Generalizable Latent Diffusion Models for rapid and accurate zero-shot High-Resolution Flood Mapping
- **分类: cs.CV**

- **简介: 论文提出Flood-LDM，利用潜在扩散模型对粗网格洪水图进行超分辨率重建，解决传统方法计算慢、泛化能力差的问题，实现快速准确的高分辨率洪水制图，提升实时风险管控能力。**

- **链接: [https://arxiv.org/pdf/2511.14033v1](https://arxiv.org/pdf/2511.14033v1)**

> **作者:** Sun Han Neo; Sachith Seneviratne; Herath Mudiyanselage Viraj Vidura Herath; Abhishek Saha; Sanka Rasnayaka; Lucy Amanda Marshall
>
> **备注:** Accepted for publication at the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Flood prediction is critical for emergency planning and response to mitigate human and economic losses. Traditional physics-based hydrodynamic models generate high-resolution flood maps using numerical methods requiring fine-grid discretization; which are computationally intensive and impractical for real-time large-scale applications. While recent studies have applied convolutional neural networks for flood map super-resolution with good accuracy and speed, they suffer from limited generalizability to unseen areas. In this paper, we propose a novel approach that leverages latent diffusion models to perform super-resolution on coarse-grid flood maps, with the objective of achieving the accuracy of fine-grid flood maps while significantly reducing inference time. Experimental results demonstrate that latent diffusion models substantially decrease the computational time required to produce high-fidelity flood maps without compromising on accuracy, enabling their use in real-time flood risk management. Moreover, diffusion models exhibit superior generalizability across different physical locations, with transfer learning further accelerating adaptation to new geographic regions. Our approach also incorporates physics-informed inputs, addressing the common limitation of black-box behavior in machine learning, thereby enhancing interpretability. Code is available at https://github.com/neosunhan/flood-diff.
>
---
#### [new 116] ForensicFlow: A Tri-Modal Adaptive Network for Robust Deepfake Detection
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 该论文提出ForensicFlow，一种三模态自适应网络，用于视频深度伪造检测任务。针对单流CNN难以捕捉多尺度伪造痕迹的问题，融合RGB、纹理和频域特征，通过注意力机制动态加权，提升检测鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.14554v1](https://arxiv.org/pdf/2511.14554v1)**

> **作者:** Mohammad Romani
>
> **备注:** 11 pages, 4 figures, 2 tables. Preprint. Submitted on November 18, 2025
>
> **摘要:** Deepfakes generated by advanced GANs and autoencoders severely threaten information integrity and societal stability. Single-stream CNNs fail to capture multi-scale forgery artifacts across spatial, texture, and frequency domains, limiting robustness and generalization. We introduce the ForensicFlow, a tri-modal forensic framework that synergistically fuses RGB, texture, and frequency evidence for video Deepfake detection. The RGB branch (ConvNeXt-tiny) extracts global visual inconsistencies; the texture branch (Swin Transformer-tiny) detects fine-grained blending artifacts; the frequency branch (CNN + SE) identifies periodic spectral noise. Attention-based temporal pooling dynamically prioritizes high-evidence frames, while adaptive attention fusion balances branch contributions.Trained on Celeb-DF (v2) with Focal Loss, ForensicFlow achieves AUC 0.9752, F1-Score 0.9408, and accuracy 0.9208, outperforming single-stream baselines. Ablation validates branch synergy; Grad-CAM confirms forensic focus. This comprehensive feature fusion provides superior resilience against subtle forgeries.
>
---
#### [new 117] Let Language Constrain Geometry: Vision-Language Models as Semantic and Spatial Critics for 3D Generation
- **分类: cs.CV**

- **简介: 论文提出VLM3D框架，利用视觉语言模型（VLM）作为语义与空间评判器，解决文本到3D生成中的语义对齐差和几何不一致问题。通过双查询批评信号指导优化与前馈流水线，显著提升生成质量。**

- **链接: [https://arxiv.org/pdf/2511.14271v1](https://arxiv.org/pdf/2511.14271v1)**

> **作者:** Weimin Bai; Yubo Li; Weijian Luo; Zeqiang Lai; Yequan Wang; Wenzheng Chen; He Sun
>
> **摘要:** Text-to-3D generation has advanced rapidly, yet state-of-the-art models, encompassing both optimization-based and feed-forward architectures, still face two fundamental limitations. First, they struggle with coarse semantic alignment, often failing to capture fine-grained prompt details. Second, they lack robust 3D spatial understanding, leading to geometric inconsistencies and catastrophic failures in part assembly and spatial relationships. To address these challenges, we propose VLM3D, a general framework that repurposes large vision-language models (VLMs) as powerful, differentiable semantic and spatial critics. Our core contribution is a dual-query critic signal derived from the VLM's Yes or No log-odds, which assesses both semantic fidelity and geometric coherence. We demonstrate the generality of this guidance signal across two distinct paradigms: (1) As a reward objective for optimization-based pipelines, VLM3D significantly outperforms existing methods on standard benchmarks. (2) As a test-time guidance module for feed-forward pipelines, it actively steers the iterative sampling process of SOTA native 3D models to correct severe spatial errors. VLM3D establishes a principled and generalizable path to inject the VLM's rich, language-grounded understanding of both semantics and space into diverse 3D generative pipelines.
>
---
#### [new 118] Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details
- **分类: cs.CV**

- **简介: 论文研究多神经符号距离函数（SDF）的紧凑潜在空间表示问题，旨在提升高保真几何细节的恢复能力和表示紧凑性。通过结合泛化与过拟合学习策略及新型采样方法，实现高效训练与高质量重建。**

- **链接: [https://arxiv.org/pdf/2511.14539v1](https://arxiv.org/pdf/2511.14539v1)**

> **作者:** Qiang Bai; Bojian Wu; Xi Yang; Zhizhong Han
>
> **备注:** Accepted as an Poster paper at the AAAI Conference on Artificial Intelligence (AAAI-26)
>
> **摘要:** Neural signed distance functions (SDFs) have been a vital representation to represent 3D shapes or scenes with neural networks. An SDF is an implicit function that can query signed distances at specific coordinates for recovering a 3D surface. Although implicit functions work well on a single shape or scene, they pose obstacles when analyzing multiple SDFs with high-fidelity geometry details, due to the limited information encoded in the latent space for SDFs and the loss of geometry details. To overcome these obstacles, we introduce a method to represent multiple SDFs in a common space, aiming to recover more high-fidelity geometry details with more compact latent representations. Our key idea is to take full advantage of the benefits of generalization-based and overfitting-based learning strategies, which manage to preserve high-fidelity geometry details with compact latent codes. Based on this framework, we also introduce a novel sampling strategy to sample training queries. The sampling can improve the training efficiency and eliminate artifacts caused by the influence of other SDFs. We report numerical and visual evaluations on widely used benchmarks to validate our designs and show advantages over the latest methods in terms of the representative ability and compactness.
>
---
#### [new 119] CascadedViT: Cascaded Chunk-FeedForward and Cascaded Group Attention Vision Transformer
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CascadedViT（CViT），一种轻量级视觉Transformer架构，通过新型分块前馈网络（CCFFN）提升计算效率。解决ViT在资源受限设备上部署困难的问题，在ImageNet-1K上实现更高能效与准确率平衡。**

- **链接: [https://arxiv.org/pdf/2511.14111v1](https://arxiv.org/pdf/2511.14111v1)**

> **作者:** Srivathsan Sivakumar; Faisal Z. Qureshi
>
> **摘要:** Vision Transformers (ViTs) have demonstrated remarkable performance across a range of computer vision tasks; however, their high computational, memory, and energy demands hinder deployment on resource-constrained platforms. In this paper, we propose \emph{Cascaded-ViT (CViT)}, a lightweight and compute-efficient vision transformer architecture featuring a novel feedforward network design called \emph{Cascaded-Chunk Feed Forward Network (CCFFN)}. By splitting input features, CCFFN improves parameter and FLOP efficiency without sacrificing accuracy. Experiments on ImageNet-1K show that our \emph{CViT-XL} model achieves 75.5\% Top-1 accuracy while reducing FLOPs by 15\% and energy consumption by 3.3\% compared to EfficientViT-M5. Across various model sizes, the CViT family consistently exhibits the lowest energy consumption, making it suitable for deployment on battery-constrained devices such as mobile phones and drones. Furthermore, when evaluated using a new metric called \emph{Accuracy-Per-FLOP (APF)}, which quantifies compute efficiency relative to accuracy, CViT models consistently achieve top-ranking efficiency. Particularly, CViT-L is 2.2\% more accurate than EfficientViT-M2 while having comparable APF scores.
>
---
#### [new 120] GEN3D: Generating Domain-Free 3D Scenes from a Single Image
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Gen3D，解决单图生成高质量、通用3D场景的问题。通过RGBD图像初始化点云并优化高斯泼溅表示，实现广域、高保真场景重建与一致新视角合成。**

- **链接: [https://arxiv.org/pdf/2511.14291v1](https://arxiv.org/pdf/2511.14291v1)**

> **作者:** Yuxin Zhang; Ziyu Lu; Hongbo Duan; Keyu Fan; Pengting Luo; Peiyu Zhuang; Mengyu Yang; Houde Liu
>
> **备注:** 5 pages , 2 figures
>
> **摘要:** Despite recent advancements in neural 3D reconstruction, the dependence on dense multi-view captures restricts their broader applicability. Additionally, 3D scene generation is vital for advancing embodied AI and world models, which depend on diverse, high-quality scenes for learning and evaluation. In this work, we propose Gen3d, a novel method for generation of high-quality, wide-scope, and generic 3D scenes from a single image. After the initial point cloud is created by lifting the RGBD image, Gen3d maintains and expands its world model. The 3D scene is finalized through optimizing a Gaussian splatting representation. Extensive experiments on diverse datasets demonstrate the strong generalization capability and superior performance of our method in generating a world model and Synthesizing high-fidelity and consistent novel views.
>
---
#### [new 121] Learning to See Through a Baby's Eyes: Early Visual Diets Enable Robust Visual Intelligence in Humans and Machines
- **分类: cs.CV**

- **简介: 论文研究视觉智能的发育机制，提出CATDiet模拟婴儿视觉发展过程，通过自监督学习提升模型在多种场景下的鲁棒性，并验证其与生物发育规律的一致性。**

- **链接: [https://arxiv.org/pdf/2511.14440v1](https://arxiv.org/pdf/2511.14440v1)**

> **作者:** Yusen Cai; Bhargava Satya Nunna; Qing Lin; Mengmi Zhang
>
> **摘要:** Newborns perceive the world with low-acuity, color-degraded, and temporally continuous vision, which gradually sharpens as infants develop. To explore the ecological advantages of such staged "visual diets", we train self-supervised learning (SSL) models on object-centric videos under constraints that simulate infant vision: grayscale-to-color (C), blur-to-sharp (A), and preserved temporal continuity (T)-collectively termed CATDiet. For evaluation, we establish a comprehensive benchmark across ten datasets, covering clean and corrupted image recognition, texture-shape cue conflict tests, silhouette recognition, depth-order classification, and the visual cliff paradigm. All CATDiet variants demonstrate enhanced robustness in object recognition, despite being trained solely on object-centric videos. Remarkably, models also exhibit biologically aligned developmental patterns, including neural plasticity changes mirroring synaptic density in macaque V1 and behaviors resembling infants' visual cliff responses. Building on these insights, CombDiet initializes SSL with CATDiet before standard training while preserving temporal continuity. Trained on object-centric or head-mounted infant videos, CombDiet outperforms standard SSL on both in-domain and out-of-domain object recognition and depth perception. Together, these results suggest that the developmental progression of early infant visual experience offers a powerful reverse-engineering framework for understanding the emergence of robust visual intelligence in machines. All code, data, and models will be publicly released.
>
---
#### [new 122] Automated glenoid bone loss measurement and segmentation in CT scans for pre-operative planning in shoulder instability
- **分类: cs.CV; cs.AI; q-bio.QM**

- **简介: 该论文属于医学影像分析任务，旨在解决肩关节不稳术前骨缺损测量效率低、一致性差的问题。作者提出一个基于深度学习的全自动流程，包括分割、定位和几何计算三阶段，实现CT图像中肩胛盂骨缺损的精准测量，性能优于人工测量。**

- **链接: [https://arxiv.org/pdf/2511.14083v1](https://arxiv.org/pdf/2511.14083v1)**

> **作者:** Zhonghao Liu; Hanxue Gu; Qihang Li; Michael Fox; Jay M. Levin; Maciej A. Mazurowski; Brian C. Lau
>
> **摘要:** Reliable measurement of glenoid bone loss is essential for operative planning in shoulder instability, but current manual and semi-automated methods are time-consuming and often subject to interreader variability. We developed and validated a fully automated deep learning pipeline for measuring glenoid bone loss on three-dimensional computed tomography (CT) scans using a linear-based, en-face view, best-circle method. Shoulder CT images of 91 patients (average age, 40 years; range, 14-89 years; 65 men) were retrospectively collected along with manual labels including glenoid segmentation, landmarks, and bone loss measurements. The multi-stage algorithm has three main stages: (1) segmentation, where we developed a U-Net to automatically segment the glenoid and humerus; (2) anatomical landmark detection, where a second network predicts glenoid rim points; and (3) geometric fitting, where we applied principal component analysis (PCA), projection, and circle fitting to compute the percentage of bone loss. The automated measurements showed strong agreement with consensus readings and exceeded surgeon-to-surgeon consistency (intraclass correlation coefficient (ICC) 0.84 vs 0.78), including in low- and high-bone-loss subgroups (ICC 0.71 vs 0.63 and 0.83 vs 0.21, respectively; P < 0.001). For classifying patients into low, medium, and high bone-loss categories, the pipeline achieved a recall of 0.714 for low and 0.857 for high severity, with no low cases misclassified as high or vice versa. These results suggest that our method is a time-efficient and clinically reliable tool for preoperative planning in shoulder instability and for screening patients with substantial glenoid bone loss. Code and dataset are available at https://github.com/Edenliu1/Auto-Glenoid-Measurement-DL-Pipeline.
>
---
#### [new 123] PAVE: An End-to-End Dataset for Production Autonomous Vehicle Evaluation
- **分类: cs.CV**

- **简介: 论文提出PAVE数据集，用于评估自动驾驶车辆的真实行为安全性。解决现有数据集无法反映黑箱控制下AV真实表现的问题。工作包括收集多车型真实自动驾驶数据、标注关键帧信息及构建端到端评估框架。**

- **链接: [https://arxiv.org/pdf/2511.14185v1](https://arxiv.org/pdf/2511.14185v1)**

> **作者:** Xiangyu Li; Chen Wang; Yumao Liu; Dengbo He; Jiahao Zhang; Ke Ma
>
> **摘要:** Most existing autonomous-driving datasets (e.g., KITTI, nuScenes, and the Waymo Perception Dataset), collected by human-driving mode or unidentified driving mode, can only serve as early training for the perception and prediction of autonomous vehicles (AVs). To evaluate the real behavioral safety of AVs controlled in the black box, we present the first end-to-end benchmark dataset collected entirely by autonomous-driving mode in the real world. This dataset contains over 100 hours of naturalistic data from multiple production autonomous-driving vehicle models in the market. We segment the original data into 32,727 key frames, each consisting of four synchronized camera images and high-precision GNSS/IMU data (0.8 cm localization accuracy). For each key frame, 20 Hz vehicle trajectories spanning the past 6 s and future 5 s are provided, along with detailed 2D annotations of surrounding vehicles, pedestrians, traffic lights, and traffic signs. These key frames have rich scenario-level attributes, including driver intent, area type (covering highways, urban roads, and residential areas), lighting (day, night, or dusk), weather (clear or rain), road surface (paved or unpaved), traffic and vulnerable road users (VRU) density, traffic lights, and traffic signs (warning, prohibition, and indication). To evaluate the safety of AVs, we employ an end-to-end motion planning model that predicts vehicle trajectories with an Average Displacement Error (ADE) of 1.4 m on autonomous-driving frames. The dataset continues to expand by over 10 hours of new data weekly, thereby providing a sustainable foundation for research on AV driving behavior analysis and safety evaluation.
>
---
#### [new 124] Can You Learn to See Without Images? Procedural Warm-Up for Vision Transformers
- **分类: cs.CV**

- **简介: 论文研究如何通过程序生成数据预训练视觉Transformer，以提升其数据效率和性能。该方法绕过图像嵌入机制，引入抽象计算先验，在少量数据下显著改善模型收敛速度与准确率，解决视觉模型依赖大量图像数据的问题。**

- **链接: [https://arxiv.org/pdf/2511.13945v1](https://arxiv.org/pdf/2511.13945v1)**

> **作者:** Zachary Shinnick; Liangze Jiang; Hemanth Saratchandran; Damien Teney; Anton van den Hengel
>
> **摘要:** Transformers show remarkable versatility across domains, suggesting the existence of inductive biases beneficial across modalities. In this work, we explore a new way to instil such generic biases in vision transformers (ViTs) by pretraining on procedurally-generated data devoid of visual or semantic content. We generate this data with simple algorithms such as formal grammars, so the results bear no relationship to either natural or synthetic images. We use this procedurally-generated data to pretrain ViTs in a warm-up phase that bypasses their visual patch embedding mechanisms, thus encouraging the models to internalise abstract computational priors. When followed by standard image-based training, this warm-up significantly improves data efficiency, convergence speed, and downstream performance. On ImageNet-1k for example, allocating just 1% of the training budget to procedural data improves final accuracy by over 1.7%. In terms of its effect on performance, 1% procedurally generated data is thus equivalent to 28% of the ImageNet-1k data. These findings suggest a promising path toward new data-efficient and domain-agnostic pretraining strategies.
>
---
#### [new 125] GRLoc: Geometric Representation Regression for Visual Localization
- **分类: cs.CV**

- **简介: 论文提出GRLoc，通过回归3D几何表示（射线方向和点图）来估计相机位姿，解决传统绝对位姿回归模型缺乏几何先验、易过拟合的问题。该方法显式分离旋转与平移预测，提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.13864v1](https://arxiv.org/pdf/2511.13864v1)**

> **作者:** Changyang Li; Xuejian Ma; Lixiang Liu; Zhan Li; Qingan Yan; Yi Xu
>
> **摘要:** Absolute Pose Regression (APR) has emerged as a compelling paradigm for visual localization. However, APR models typically operate as black boxes, directly regressing a 6-DoF pose from a query image, which can lead to memorizing training views rather than understanding 3D scene geometry. In this work, we propose a geometrically-grounded alternative. Inspired by novel view synthesis, which renders images from intermediate geometric representations, we reformulate APR as its inverse that regresses the underlying 3D representations directly from the image, and we name this paradigm Geometric Representation Regression (GRR). Our model explicitly predicts two disentangled geometric representations in the world coordinate system: (1) a ray bundle's directions to estimate camera rotation, and (2) a corresponding pointmap to estimate camera translation. The final 6-DoF camera pose is then recovered from these geometric components using a differentiable deterministic solver. This disentangled approach, which separates the learned visual-to-geometry mapping from the final pose calculation, introduces a strong geometric prior into the network. We find that the explicit decoupling of rotation and translation predictions measurably boosts performance. We demonstrate state-of-the-art performance on 7-Scenes and Cambridge Landmarks datasets, validating that modeling the inverse rendering process is a more robust path toward generalizable absolute pose estimation.
>
---
#### [new 126] CORE: Compact Object-centric REpresentations as a New Paradigm for Token Merging in LVLMs
- **分类: cs.CV**

- **简介: 该论文针对大视觉语言模型（LVLMs）因图像分辨率导致的计算和内存开销问题，提出CORE方法。通过对象中心表示与掩码引导合并策略，实现高效视觉令牌压缩，同时保持语义完整性与空间顺序，显著提升压缩效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.14072v1](https://arxiv.org/pdf/2511.14072v1)**

> **作者:** Jingyu Lei; Gaoang Wang; Der-Horng Lee
>
> **摘要:** Large Vision-Language Models (LVLMs) usually suffer from prohibitive computational and memory costs due to the quadratic growth of visual tokens with image resolution. Existing token compression methods, while varied, often lack a high-level semantic understanding, leading to suboptimal merges, information redundancy, or context loss. To address these limitations, we introduce CORE (Compact Object-centric REpresentations), a new paradigm for visual token compression. CORE leverages an efficient segmentation decoder to generate object masks, which serve as a high-level semantic prior to guide the merging of visual tokens into a compact set of object-centric representations. Furthermore, a novel centroid-guided sorting mechanism restores a coherent spatial order to the merged tokens, preserving vital positional information. Extensive experiments show that CORE not only establishes a new state-of-the-art on six authoritative benchmarks for fixed-rate compression, but also achieves dramatic efficiency gains in adaptive-rate settings. Even under extreme compression, after aggressively retaining with only 2.2% of all visual tokens, CORE still maintains 97.4% of baseline performance. Our work demonstrates the superiority of object-centric representations for efficient and effective LVLM processing.
>
---
#### [new 127] StreamingTalker: Audio-driven 3D Facial Animation with Autoregressive Diffusion Model
- **分类: cs.CV**

- **简介: 论文提出StreamingTalker，用于语音驱动的3D人脸动画生成。针对现有方法处理长音频时延迟高、超出训练长度无法处理的问题，设计了自回归扩散模型，通过动态条件逐帧生成面部动作，实现低延迟实时合成。**

- **链接: [https://arxiv.org/pdf/2511.14223v1](https://arxiv.org/pdf/2511.14223v1)**

> **作者:** Yifan Yang; Zhi Cen; Sida Peng; Xiangwei Chen; Yifu Deng; Xinyu Zhu; Fan Jia; Xiaowei Zhou; Hujun Bao
>
> **摘要:** This paper focuses on the task of speech-driven 3D facial animation, which aims to generate realistic and synchronized facial motions driven by speech inputs.Recent methods have employed audio-conditioned diffusion models for 3D facial animation, achieving impressive results in generating expressive and natural animations.However, these methods process the whole audio sequences in a single pass, which poses two major challenges: they tend to perform poorly when handling audio sequences that exceed the training horizon and will suffer from significant latency when processing long audio inputs. To address these limitations, we propose a novel autoregressive diffusion model that processes input audio in a streaming manner. This design ensures flexibility with varying audio lengths and achieves low latency independent of audio duration. Specifically, we select a limited number of past frames as historical motion context and combine them with the audio input to create a dynamic condition. This condition guides the diffusion process to iteratively generate facial motion frames, enabling real-time synthesis with high-quality results. Additionally, we implemented a real-time interactive demo, highlighting the effectiveness and efficiency of our approach. We will release the code at https://zju3dv.github.io/StreamingTalker/.
>
---
#### [new 128] QwenCLIP: Boosting Medical Vision-Language Pretraining via LLM Embeddings and Prompt tuning
- **分类: cs.CV**

- **简介: 论文提出QwenCLIP，用于医学视觉-语言预训练，解决CLIP文本编码器输入长度短、语义理解浅的问题。通过引入LLM嵌入模块和可学习提示，提升长篇放射学报告的表示能力与跨模态对齐效果。**

- **链接: [https://arxiv.org/pdf/2511.13876v1](https://arxiv.org/pdf/2511.13876v1)**

> **作者:** Xiaoyang Wei; Camille Kurtz; Florence Cloppet
>
> **备注:** This work has been submitted to the IEEE ISBI for possible publication
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) has demonstrated strong generalization for vision-language tasks in computer vision and medical domains, yet its text encoder accepts only up to 77 tokens, which limits its ability to represent long and information-rich radiology reports. Recent adaptations using domain-specific encoders, such as PubMedBERT or ClinicalBERT, mitigate this issue by leveraging medical corpora, but remain constrained by their limited input length (typically 512 tokens) and relatively shallow semantic understanding. To address these limitations, we propose QwenCLIP, a vision-language framework that replaces CLIP's text encoder with a large language model (LLM)-based embedding module (e.g., Qwen3-Embedding) and introduces learnable prompts to enhance cross-modal alignment. By leveraging the extended context window and richer representations of LLMs, QwenCLIP captures comprehensive medical semantics from long-form clinical text, substantially improving medical image-text alignment and downstream performance on radiology benchmarks. Our code is publicly available at https://github.com/Wxy-24/QwenCLIP.
>
---
#### [new 129] Semantic Context Matters: Improving Conditioning for Autoregressive Models
- **分类: cs.CV**

- **简介: 论文提出SCAR方法，解决自回归图像生成中条件控制弱、指令遵循差的问题。通过压缩语义预填充和语义对齐引导，提升生成图像的语义一致性和可控性，适用于图像编辑与可控生成任务。**

- **链接: [https://arxiv.org/pdf/2511.14063v1](https://arxiv.org/pdf/2511.14063v1)**

> **作者:** Dongyang Jin; Ryan Xu; Jianhao Zeng; Rui Lan; Yancheng Bai; Lei Sun; Xiangxiang Chu
>
> **摘要:** Recently, autoregressive (AR) models have shown strong potential in image generation, offering better scalability and easier integration with unified multi-modal systems compared to diffusion-based methods. However, extending AR models to general image editing remains challenging due to weak and inefficient conditioning, often leading to poor instruction adherence and visual artifacts. To address this, we propose SCAR, a Semantic-Context-driven method for Autoregressive models. SCAR introduces two key components: Compressed Semantic Prefilling, which encodes high-level semantics into a compact and efficient prefix, and Semantic Alignment Guidance, which aligns the last visual hidden states with target semantics during autoregressive decoding to enhance instruction fidelity. Unlike decoding-stage injection methods, SCAR builds upon the flexibility and generality of vector-quantized-based prefilling while overcoming its semantic limitations and high cost. It generalizes across both next-token and next-set AR paradigms with minimal architectural changes. SCAR achieves superior visual fidelity and semantic alignment on both instruction editing and controllable generation benchmarks, outperforming prior AR-based methods while maintaining controllability. All code will be released.
>
---
#### [new 130] FashionMAC: Deformation-Free Fashion Image Generation with Fine-Grained Model Appearance Customization
- **分类: cs.CV**

- **简介: 该论文属于服装图像生成任务，旨在解决现有方法需变形导致纹理失真及难以控制模型细节的问题。提出FashionMAC框架，通过去变形生成和区域自适应注意力机制实现高质量、细粒度可控的时尚图像合成。**

- **链接: [https://arxiv.org/pdf/2511.14031v1](https://arxiv.org/pdf/2511.14031v1)**

> **作者:** Rong Zhang; Jinxiao Li; Jingnan Wang; Zhiwen Zuo; Jianfeng Dong; Wei Li; Chi Wang; Weiwei Xu; Xun Wang
>
> **摘要:** Garment-centric fashion image generation aims to synthesize realistic and controllable human models dressing a given garment, which has attracted growing interest due to its practical applications in e-commerce. The key challenges of the task lie in two aspects: (1) faithfully preserving the garment details, and (2) gaining fine-grained controllability over the model's appearance. Existing methods typically require performing garment deformation in the generation process, which often leads to garment texture distortions. Also, they fail to control the fine-grained attributes of the generated models, due to the lack of specifically designed mechanisms. To address these issues, we propose FashionMAC, a novel diffusion-based deformation-free framework that achieves high-quality and controllable fashion showcase image generation. The core idea of our framework is to eliminate the need for performing garment deformation and directly outpaint the garment segmented from a dressed person, which enables faithful preservation of the intricate garment details. Moreover, we propose a novel region-adaptive decoupled attention (RADA) mechanism along with a chained mask injection strategy to achieve fine-grained appearance controllability over the synthesized human models. Specifically, RADA adaptively predicts the generated regions for each fine-grained text attribute and enforces the text attribute to focus on the predicted regions by a chained mask injection strategy, significantly enhancing the visual fidelity and the controllability. Extensive experiments validate the superior performance of our framework compared to existing state-of-the-art methods.
>
---
#### [new 131] Weakly Supervised Ephemeral Gully Detection In Remote Sensing Images Using Vision Language Models
- **分类: cs.CV**

- **简介: 论文提出弱监督方法检测遥感图像中的临时沟壑，解决标注数据稀缺和传统方法难以捕捉短时变化的问题。利用视觉语言模型生成噪声标签，通过教师-学生框架与抗噪损失函数训练学生模型，显著优于纯零样本方法。**

- **链接: [https://arxiv.org/pdf/2511.13891v1](https://arxiv.org/pdf/2511.13891v1)**

> **作者:** Seyed Mohamad Ali Tousi; John A. Lory; G. N. DeSouza
>
> **摘要:** Among soil erosion problems, Ephemeral Gullies are one of the most concerning phenomena occurring in agricultural fields. Their short temporal cycles increase the difficulty in automatically detecting them using classical computer vision approaches and remote sensing. Also, due to scarcity of and the difficulty in producing accurate labeled data, automatic detection of ephemeral gullies using Machine Learning is limited to zero-shot approaches which are hard to implement. To overcome these challenges, we present the first weakly supervised pipeline for detection of ephemeral gullies. Our method relies on remote sensing and uses Vision Language Models (VLMs) to drastically reduce the labor-intensive task of manual labeling. In order to achieve that, the method exploits: 1) the knowledge embedded in the VLM's pretraining; 2) a teacher-student model where the teacher learns from noisy labels coming from the VLMs, and the student learns by weak supervision using teacher-generate labels and a noise-aware loss function. We also make available the first-of-its-kind dataset for semi-supervised detection of ephemeral gully from remote-sensed images. The dataset consists of a number of locations labeled by a group of soil and plant scientists, as well as a large number of unlabeled locations. The dataset represent more than 18,000 high-resolution remote-sensing images obtained over the course of 13 years. Our experimental results demonstrate the validity of our approach by showing superior performances compared to VLMs and the label model itself when using weak supervision to train an student model. The code and dataset for this work are made publicly available.
>
---
#### [new 132] Orion: A Unified Visual Agent for Multimodal Perception, Advanced Visual Reasoning and Execution
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出Orion框架，解决视觉智能从被动理解到主动执行的难题。通过多模态感知与工具调用，实现复杂视觉任务的自动化推理与执行，提升视觉AI的实用性与性能。**

- **链接: [https://arxiv.org/pdf/2511.14210v1](https://arxiv.org/pdf/2511.14210v1)**

> **作者:** N Dinesh Reddy; Sudeep Pillai
>
> **摘要:** We introduce Orion, a visual agent framework that can take in any modality and generate any modality. Using an agentic framework with multiple tool-calling capabilities, Orion is designed for visual AI tasks and achieves state-of-the-art results. Unlike traditional vision-language models that produce descriptive outputs, Orion orchestrates a suite of specialized computer vision tools, including object detection, keypoint localization, panoptic segmentation, Optical Character Recognition, and geometric analysis, to execute complex multi-step visual workflows. The system achieves competitive performance on MMMU, MMBench, DocVQA, and MMLongBench while extending monolithic vision-language models to production-grade visual intelligence. By combining neural perception with symbolic execution, Orion enables autonomous visual reasoning, marking a transition from passive visual understanding to active, tool-driven visual intelligence.
>
---
#### [new 133] CD-DPE: Dual-Prompt Expert Network based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于多对比MRI超分辨率任务，旨在解决不同模态间对比度差异导致的特征融合困难问题。提出CD-DPE网络，通过特征解耦和双提示融合机制，提升重建细节与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.14014v1](https://arxiv.org/pdf/2511.14014v1)**

> **作者:** Xianming Gu; Lihui Wang; Ying Cao; Zeyu Deng; Yingfeng Ou; Guodong Hu; Yi Chen
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multi-contrast magnetic resonance imaging (MRI) super-resolution intends to reconstruct high-resolution (HR) images from low-resolution (LR) scans by leveraging structural information present in HR reference images acquired with different contrasts. This technique enhances anatomical detail and soft tissue differentiation, which is vital for early diagnosis and clinical decision-making. However, inherent contrasts disparities between modalities pose fundamental challenges in effectively utilizing reference image textures to guide target image reconstruction, often resulting in suboptimal feature integration. To address this issue, we propose a dual-prompt expert network based on a convolutional dictionary feature decoupling (CD-DPE) strategy for multi-contrast MRI super-resolution. Specifically, we introduce an iterative convolutional dictionary feature decoupling module (CD-FDM) to separate features into cross-contrast and intra-contrast components, thereby reducing redundancy and interference. To fully integrate these features, a novel dual-prompt feature fusion expert module (DP-FFEM) is proposed. This module uses a frequency prompt to guide the selection of relevant reference features for incorporation into the target image, while an adaptive routing prompt determines the optimal method for fusing reference and target features to enhance reconstruction quality. Extensive experiments on public multi-contrast MRI datasets demonstrate that CD-DPE outperforms state-of-the-art methods in reconstructing fine details. Additionally, experiments on unseen datasets demonstrated that CD-DPE exhibits strong generalization capabilities.
>
---
#### [new 134] Iterative Diffusion-Refined Neural Attenuation Fields for Multi-Source Stationary CT Reconstruction: NAF Meets Diffusion Model
- **分类: cs.CV**

- **简介: 该论文针对多源静态CT在超稀疏视角下的重建质量差问题，提出Diff-NAF框架。通过神经衰减场与扩散模型结合，迭代生成并精炼投影数据，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2511.14310v1](https://arxiv.org/pdf/2511.14310v1)**

> **作者:** Jiancheng Fang; Shaoyu Wang; Junlin Wang; Weiwen Wu; Yikun Zhang; Qiegen Liu
>
> **摘要:** Multi-source stationary computed tomography (CT) has recently attracted attention for its ability to achieve rapid image reconstruction, making it suitable for time-sensitive clinical and industrial applications. However, practical systems are often constrained by ultra-sparse-view sampling, which significantly degrades reconstruction quality. Traditional methods struggle under ultra-sparse-view settings, where interpolation becomes inaccurate and the resulting reconstructions are unsatisfactory. To address this challenge, this study proposes Diffusion-Refined Neural Attenuation Fields (Diff-NAF), an iterative framework tailored for multi-source stationary CT under ultra-sparse-view conditions. Diff-NAF combines a Neural Attenuation Field representation with a dual-branch conditional diffusion model. The process begins by training an initial NAF using ultra-sparse-view projections. New projections are then generated through an Angle-Prior Guided Projection Synthesis strategy that exploits inter view priors, and are subsequently refined by a Diffusion-driven Reuse Projection Refinement Module. The refined projections are incorporated as pseudo-labels into the training set for the next iteration. Through iterative refinement, Diff-NAF progressively enhances projection completeness and reconstruction fidelity under ultra-sparse-view conditions, ultimately yielding high-quality CT reconstructions. Experimental results on multiple simulated 3D CT volumes and real projection data demonstrate that Diff-NAF achieves the best performance under ultra-sparse-view conditions.
>
---
#### [new 135] EchoAgent: Guideline-Centric Reasoning Agent for Echocardiography Measurement and Interpretation
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 论文提出EchoAgent，一种基于指南的超声心动图视频分析框架，解决现有模型无法进行视频级推理和规范测量的问题。通过LLM协调视觉工具实现时空定位、测量与解读，引入可行性预测模型优化工具选择，提升结果的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.13948v1](https://arxiv.org/pdf/2511.13948v1)**

> **作者:** Matin Daghyani; Lyuyang Wang; Nima Hashemi; Bassant Medhat; Baraa Abdelsamad; Eros Rojas Velez; XiaoXiao Li; Michael Y. C. Tsang; Christina Luong; Teresa S. M. Tsang; Purang Abolmaesumi
>
> **备注:** 12 pages, Under Review
>
> **摘要:** Purpose: Echocardiographic interpretation requires video-level reasoning and guideline-based measurement analysis, which current deep learning models for cardiac ultrasound do not support. We present EchoAgent, a framework that enables structured, interpretable automation for this domain. Methods: EchoAgent orchestrates specialized vision tools under Large Language Model (LLM) control to perform temporal localization, spatial measurement, and clinical interpretation. A key contribution is a measurement-feasibility prediction model that determines whether anatomical structures are reliably measurable in each frame, enabling autonomous tool selection. We curated a benchmark of diverse, clinically validated video-query pairs for evaluation. Results: EchoAgent achieves accurate, interpretable results despite added complexity of spatiotemporal video analysis. Outputs are grounded in visual evidence and clinical guidelines, supporting transparency and traceability. Conclusion: This work demonstrates the feasibility of agentic, guideline-aligned reasoning for echocardiographic video analysis, enabled by task-specific tools and full video-level automation. EchoAgent sets a new direction for trustworthy AI in cardiac ultrasound.
>
---
#### [new 136] FusionFM: All-in-One Multi-Modal Image Fusion with Flow Matching
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FusionFM，一种基于流匹配的统一多模态图像融合方法，解决现有方法依赖任务特定模型、训练成本高及推理慢的问题。通过伪标签选择与精炼机制提升融合质量，结合持续学习策略增强多任务性能。**

- **链接: [https://arxiv.org/pdf/2511.13794v1](https://arxiv.org/pdf/2511.13794v1)**

> **作者:** Huayi Zhu; Xiu Shu; Youqiang Xiong; Qiao Liu; Rui Chen; Di Yuan; Xiaojun Chang; Zhenyu He
>
> **摘要:** Current multi-modal image fusion methods typically rely on task-specific models, leading to high training costs and limited scalability. While generative methods provide a unified modeling perspective, they often suffer from slow inference due to the complex sampling trajectories from noise to image. To address this, we formulate image fusion as a direct probabilistic transport from source modalities to the fused image distribution, leveraging the flow matching paradigm to improve sampling efficiency and structural consistency. To mitigate the lack of high-quality fused images for supervision, we collect fusion results from multiple state-of-the-art models as priors, and employ a task-aware selection function to select the most reliable pseudo-labels for each task. We further introduce a Fusion Refiner module that employs a divide-and-conquer strategy to systematically identify, decompose, and enhance degraded components in selected pseudo-labels. For multi-task scenarios, we integrate elastic weight consolidation and experience replay mechanisms to preserve cross-task performance and enhance continual learning ability from both parameter stability and memory retention perspectives. Our approach achieves competitive performance across diverse fusion tasks, while significantly improving sampling efficiency and maintaining a lightweight model design. The code will be available at: https://github.com/Ist-Zhy/FusionFM.
>
---
#### [new 137] Exploring Transferability of Self-Supervised Learning by Task Conflict Calibration
- **分类: cs.LG; cs.CV**

- **简介: 论文研究自监督学习（SSL）的表示迁移能力，针对迁移受限于任务冲突的问题，提出TC²方法。通过多任务构造与双层优化框架，利用因果因子提取和权重分配提升迁移性能，在下游任务中验证有效性。**

- **链接: [https://arxiv.org/pdf/2511.13787v1](https://arxiv.org/pdf/2511.13787v1)**

> **作者:** Huijie Guo; Jingyao Wang; Peizheng Guo; Xingchen Shen; Changwen Zheng; Wenwen Qiang
>
> **摘要:** In this paper, we explore the transferability of SSL by addressing two central questions: (i) what is the representation transferability of SSL, and (ii) how can we effectively model this transferability? Transferability is defined as the ability of a representation learned from one task to support the objective of another. Inspired by the meta-learning paradigm, we construct multiple SSL tasks within each training batch to support explicitly modeling transferability. Based on empirical evidence and causal analysis, we find that although introducing task-level information improves transferability, it is still hindered by task conflict. To address this issue, we propose a Task Conflict Calibration (TC$^2$) method to alleviate the impact of task conflict. Specifically, it first splits batches to create multiple SSL tasks, infusing task-level information. Next, it uses a factor extraction network to produce causal generative factors for all tasks and a weight extraction network to assign dedicated weights to each sample, employing data reconstruction, orthogonality, and sparsity to ensure effectiveness. Finally, TC$^2$ calibrates sample representations during SSL training and integrates into the pipeline via a two-stage bi-level optimization framework to boost the transferability of learned representations. Experimental results on multiple downstream tasks demonstrate that our method consistently improves the transferability of SSL models.
>
---
#### [new 138] Certified but Fooled! Breaking Certified Defences with Ghost Certificates
- **分类: cs.LG; cs.CR; cs.CV**

- **简介: 论文研究认证防御的漏洞，提出通过伪造证书欺骗模型生成虚假鲁棒性保证。针对图像分类任务，设计不可察觉的扰动使模型误分类但输出大鲁棒半径，成功绕过DensePure等先进防御方法。**

- **链接: [https://arxiv.org/pdf/2511.14003v1](https://arxiv.org/pdf/2511.14003v1)**

> **作者:** Quoc Viet Vo; Tashreque M. Haq; Paul Montague; Tamas Abraham; Ehsan Abbasnejad; Damith C. Ranasinghe
>
> **备注:** Published as a conference paper at the Fortieth AAAI Conference on Artificial Intelligence (AAAI-26). Code available at: https://github.com/ghostcert/ghostcert
>
> **摘要:** Certified defenses promise provable robustness guarantees. We study the malicious exploitation of probabilistic certification frameworks to better understand the limits of guarantee provisions. Now, the objective is to not only mislead a classifier, but also manipulate the certification process to generate a robustness guarantee for an adversarial input certificate spoofing. A recent study in ICLR demonstrated that crafting large perturbations can shift inputs far into regions capable of generating a certificate for an incorrect class. Our study investigates if perturbations needed to cause a misclassification and yet coax a certified model into issuing a deceptive, large robustness radius for a target class can still be made small and imperceptible. We explore the idea of region-focused adversarial examples to craft imperceptible perturbations, spoof certificates and achieve certification radii larger than the source class ghost certificates. Extensive evaluations with the ImageNet demonstrate the ability to effectively bypass state-of-the-art certified defenses such as Densepure. Our work underscores the need to better understand the limits of robustness certification methods.
>
---
#### [new 139] Scene Graph-Guided Generative AI Framework for Synthesizing and Evaluating Industrial Hazard Scenarios
- **分类: cs.AI; cs.CV**

- **简介: 论文提出一种场景图引导的生成框架，用于合成工业危险场景图像并评估其真实性。解决真实危险数据难获取的问题，通过分析OSHA报告构建场景图指导生成，并引入VQA评分机制提升评估精度。**

- **链接: [https://arxiv.org/pdf/2511.13970v1](https://arxiv.org/pdf/2511.13970v1)**

> **作者:** Sanjay Acharjee; Abir Khan Ratul; Diego Patino; Md Nazmus Sakib
>
> **摘要:** Training vision models to detect workplace hazards accurately requires realistic images of unsafe conditions that could lead to accidents. However, acquiring such datasets is difficult because capturing accident-triggering scenarios as they occur is nearly impossible. To overcome this limitation, this study presents a novel scene graph-guided generative AI framework that synthesizes photorealistic images of hazardous scenarios grounded in historical Occupational Safety and Health Administration (OSHA) accident reports. OSHA narratives are analyzed using GPT-4o to extract structured hazard reasoning, which is converted into object-level scene graphs capturing spatial and contextual relationships essential for understanding risk. These graphs guide a text-to-image diffusion model to generate compositionally accurate hazard scenes. To evaluate the realism and semantic fidelity of the generated data, a visual question answering (VQA) framework is introduced. Across four state-of-the-art generative models, the proposed VQA Graph Score outperforms CLIP and BLIP metrics based on entropy-based validation, confirming its higher discriminative sensitivity.
>
---
#### [new 140] Can LLMs Create Legally Relevant Summaries and Analyses of Videos?
- **分类: cs.MM; cs.AI; cs.CV; cs.CY**

- **简介: 论文研究大语言模型（LLM）是否能从视频中提取法律相关事实并生成摘要和法律信函。任务是让LLM理解视频内容并生成高质量法律文本，以帮助普通人处理法律事务。作者用120个YouTube视频测试，71.7%的摘要获高或中质量评价。**

- **链接: [https://arxiv.org/pdf/2511.13772v1](https://arxiv.org/pdf/2511.13772v1)**

> **作者:** Lyra Hoeben-Kuil; Gijs van Dijck; Jaromir Savelka; Johanna Gunawan; Konrad Kollnig; Marta Kolacz; Mindy Duffourc; Shashank Chakravarthy; Hannes Westermann
>
> **备注:** Accepted for publication at JURIX 2025 Torino, Italy. This is the preprint version. Code and data available at: https://github.com/maastrichtlawtech/jurix2025_LLM_video_analysis
>
> **摘要:** Understanding the legally relevant factual basis of an event and conveying it through text is a key skill of legal professionals. This skill is important for preparing forms (e.g., insurance claims) or other legal documents (e.g., court claims), but often presents a challenge for laypeople. Current AI approaches aim to bridge this gap, but mostly rely on the user to articulate what has happened in text, which may be challenging for many. Here, we investigate the capability of large language models (LLMs) to understand and summarize events occurring in videos. We ask an LLM to summarize and draft legal letters, based on 120 YouTube videos showing legal issues in various domains. Overall, 71.7\% of the summaries were rated as of high or medium quality, which is a promising result, opening the door to a number of applications in e.g. access to justice.
>
---
#### [new 141] Self-Supervised Compression and Artifact Correction for Streaming Underwater Imaging Sonar
- **分类: eess.IV; cs.CV; cs.LG; cs.MM**

- **简介: 论文提出SCOPE框架，解决水下声呐图像压缩与伪影矫正难题。通过自监督学习联合优化，实现低比特率传输（≤0.0118 bpp）并提升图像质量（SSIM=0.77），显著减少带宽需求且支持实时检测。**

- **链接: [https://arxiv.org/pdf/2511.13922v1](https://arxiv.org/pdf/2511.13922v1)**

> **作者:** Rongsheng Qian; Chi Xu; Xiaoqiang Ma; Hao Fang; Yili Jin; William I. Atlas; Jiangchuan Liu
>
> **备注:** Accepted to WACV 2026
>
> **摘要:** Real-time imaging sonar has become an important tool for underwater monitoring in environments where optical sensing is unreliable. Its broader use is constrained by two coupled challenges: highly limited uplink bandwidth and severe sonar-specific artifacts (speckle, motion blur, reverberation, acoustic shadows) that affect up to 98% of frames. We present SCOPE, a self-supervised framework that jointly performs compression and artifact correction without clean-noise pairs or synthetic assumptions. SCOPE combines (i) Adaptive Codebook Compression (ACC), which learns frequency-encoded latent representations tailored to sonar, with (ii) Frequency-Aware Multiscale Segmentation (FAMS), which decomposes frames into low-frequency structure and sparse high-frequency dynamics while suppressing rapidly fluctuating artifacts. A hedging training strategy further guides frequency-aware learning using low-pass proxy pairs generated without labels. Evaluated on months of in-situ ARIS sonar data, SCOPE achieves a structural similarity index (SSIM) of 0.77, representing a 40% improvement over prior self-supervised denoising baselines, at bitrates down to <= 0.0118 bpp. It reduces uplink bandwidth by more than 80% while improving downstream detection. The system runs in real time, with 3.1 ms encoding on an embedded GPU and 97 ms full multi-layer decoding on the server end. SCOPE has been deployed for months in three Pacific Northwest rivers to support real-time salmon enumeration and environmental monitoring in the wild. Results demonstrate that learning frequency-structured latents enables practical, low-bitrate sonar streaming with preserved signal details under real-world deployment conditions.
>
---
#### [new 142] MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with MoE-LayerNorm
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出MoETTA框架，解决测试时适应（TTA）在混合分布偏移下的性能下降问题。通过引入Mixture-of-Experts结构，实现多方向参数更新，提升对复杂真实场景的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.13760v1](https://arxiv.org/pdf/2511.13760v1)**

> **作者:** Xiao Fan; Jingyan Jiang; Zhaoru Chen; Fanding Huang; Xiao Chen; Qinting Jiang; Bowen Zhang; Xing Tang; Zhi Wang
>
> **备注:** Accepted by AAAI 2026 Main Technical Track
>
> **摘要:** Test-Time adaptation (TTA) has proven effective in mitigating performance drops under single-domain distribution shifts by updating model parameters during inference. However, real-world deployments often involve mixed distribution shifts, where test samples are affected by diverse and potentially conflicting domain factors, posing significant challenges even for SOTA TTA methods. A key limitation in existing approaches is their reliance on a unified adaptation path, which fails to account for the fact that optimal gradient directions can vary significantly across different domains. Moreover, current benchmarks focus only on synthetic or homogeneous shifts, failing to capture the complexity of real-world heterogeneous mixed distribution shifts. To address this, we propose MoETTA, a novel entropy-based TTA framework that integrates the Mixture-of-Experts (MoE) architecture. Rather than enforcing a single parameter update rule for all test samples, MoETTA introduces a set of structurally decoupled experts, enabling adaptation along diverse gradient directions. This design allows the model to better accommodate heterogeneous shifts through flexible and disentangled parameter updates. To simulate realistic deployment conditions, we introduce two new benchmarks: potpourri and potpourri+. While classical settings focus solely on synthetic corruptions, potpourri encompasses a broader range of domain shifts--including natural, artistic, and adversarial distortions--capturing more realistic deployment challenges. Additionally, potpourri+ further includes source-domain samples to evaluate robustness against catastrophic forgetting. Extensive experiments across three mixed distribution shifts settings show that MoETTA consistently outperforms strong baselines, establishing SOTA performance and highlighting the benefit of modeling multiple adaptation directions via expert-level diversity.
>
---
#### [new 143] Going Places: Place Recognition in Artificial and Natural Systems
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究place recognition任务，旨在解决人工系统与生物系统在位置识别上的共性与差异。通过整合机器人、动物和人类的研究成果，提出统一概念框架，以提升人工系统的泛化能力与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.14341v1](https://arxiv.org/pdf/2511.14341v1)**

> **作者:** Michael Milford; Tobias Fischer
>
> **摘要:** Place recognition, the ability to identify previously visited locations, is critical for both biological navigation and autonomous systems. This review synthesizes findings from robotic systems, animal studies, and human research to explore how different systems encode and recall place. We examine the computational and representational strategies employed across artificial systems, animals, and humans, highlighting convergent solutions such as topological mapping, cue integration, and memory management. Animal systems reveal evolved mechanisms for multimodal navigation and environmental adaptation, while human studies provide unique insights into semantic place concepts, cultural influences, and introspective capabilities. Artificial systems showcase scalable architectures and data-driven models. We propose a unifying set of concepts by which to consider and develop place recognition mechanisms and identify key challenges such as generalization, robustness, and environmental variability. This review aims to foster innovations in artificial localization by connecting future developments in artificial place recognition systems to insights from both animal navigation research and human spatial cognition studies.
>
---
#### [new 144] The CHASM-SWPC Dataset for Coronal Hole Detection & Analysis
- **分类: astro-ph.IM; cs.CV; cs.LG**

- **简介: 论文提出CHASM-SWPC数据集，用于冠状暗区自动检测任务。通过半自动标注工具CHASM生成1111张二值掩膜，训练CHRONNOS神经网络模型，在准确率、TSS和IoU上均优于原模型，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2511.14044v1](https://arxiv.org/pdf/2511.14044v1)**

> **作者:** Cutter Beck; Evan Smith; Khagendra Katuwal; Rudra Kafle; Jacob Whitehill
>
> **摘要:** Coronal holes (CHs) are low-activity, low-density solar coronal regions with open magnetic field lines (Cranmer 2009). In the extreme ultraviolet (EUV) spectrum, CHs appear as dark patches. Using daily hand-drawn maps from the Space Weather Prediction Center (SWPC), we developed a semi-automated pipeline to digitize the SWPC maps into binary segmentation masks. The resulting masks constitute the CHASM-SWPC dataset, a high-quality dataset to train and test automated CH detection models, which is released with this paper. We developed CHASM (Coronal Hole Annotation using Semi-automatic Methods), a software tool for semi-automatic annotation that enables users to rapidly and accurately annotate SWPC maps. The CHASM tool enabled us to annotate 1,111 CH masks, comprising the CHASM-SWPC-1111 dataset. We then trained multiple CHRONNOS (Coronal Hole RecOgnition Neural Network Over multi-Spectral-data) architecture (Jarolim et al. 2021) neural networks using the CHASM-SWPC dataset and compared their performance. Training the CHRONNOS neural network on these data achieved an accuracy of 0.9805, a True Skill Statistic (TSS) of 0.6807, and an intersection-over-union (IoU) of 0.5668, which is higher than the original pretrained CHRONNOS model Jarolim et al. (2021) achieved an accuracy of 0.9708, a TSS of 0.6749, and an IoU of 0.4805, when evaluated on the CHASM-SWPC-1111 test set.
>
---
#### [new 145] MindCross: Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals
- **分类: cs.MM; cs.CV; cs.HC**

- **简介: 论文提出MindCross框架，用于从脑信号重建视频的跨被试任务。针对现有方法数据效率低、适应慢的问题，通过分离提取个体特异与共通信息，并引入Top-K协作模块，实现仅用少量数据快速适配新被试。**

- **链接: [https://arxiv.org/pdf/2511.14196v1](https://arxiv.org/pdf/2511.14196v1)**

> **作者:** Xuan-Hao Liu; Yan-Kai Liu; Tianyi Zhou; Bao-Liang Lu; Wei-Long Zheng
>
> **备注:** AAAI 2026, 16 pages
>
> **摘要:** Reconstructing video from brain signals is an important brain decoding task. Existing brain decoding frameworks are primarily built on a subject-dependent paradigm, which requires large amounts of brain data for each subject. However, the expensive cost of collecting brain-video data causes severe data scarcity. Although some cross-subject methods being introduced, they often overfocus with subject-invariant information while neglecting subject-specific information, resulting in slow fine-tune-based adaptation strategy. To achieve fast and data-efficient new subject adaptation, we propose MindCross, a novel cross-subject framework. MindCross's N specific encoders and one shared encoder are designed to extract subject-specific and subject-invariant information, respectively. Additionally, a Top-K collaboration module is adopted to enhance new subject decoding with the knowledge learned from previous subjects' encoders. Extensive experiments on fMRI/EEG-to-video benchmarks demonstrate MindCross's efficacy and efficiency of cross-subject decoding and new subject adaptation using only one model.
>
---
#### [new 146] Enhancing Agentic Autonomous Scientific Discovery with Vision-Language Model Capabilities
- **分类: cs.CL; cs.AI; cs.CV; cs.MA**

- **简介: 论文提出用视觉语言模型增强多智能体系统，实现自主科学发现。通过VLM评估图表并动态调整推理路径，解决传统方法易出错、难解释的问题，在天体物理任务中显著提升准确率并生成可审计的推理过程。**

- **链接: [https://arxiv.org/pdf/2511.14631v1](https://arxiv.org/pdf/2511.14631v1)**

> **作者:** Kahaan Gandhi; Boris Bolliet; Inigo Zubeldia
>
> **摘要:** We show that multi-agent systems guided by vision-language models (VLMs) improve end-to-end autonomous scientific discovery. By treating plots as verifiable checkpoints, a VLM-as-a-judge evaluates figures against dynamically generated domain-specific rubrics, enabling agents to correct their own errors and steer exploratory data analysis in real-time. Case studies in cosmology and astrochemistry demonstrate recovery from faulty reasoning paths and adaptation to new datasets without human intervention. On a 10-task benchmark for data-driven discovery, VLM-augmented systems achieve pass at 1 scores of 0.7-0.8, compared to 0.2-0.3 for code-only and 0.4-0.5 for code-and-text baselines, while also providing auditable reasoning traces that improve interpretability. Code available here: https://github.com/CMBAgents/cmbagent
>
---
#### [new 147] ELiC: Efficient LiDAR Geometry Compression via Cross-Bit-depth Feature Propagation and Bag-of-Encoders
- **分类: eess.IV; cs.CV**

- **简介: 论文提出ELiC框架，用于高效激光雷达几何压缩。针对现有方法独立处理各比特深度、效率低的问题，通过跨比特深度特征传播、编码器选择机制和Morton序层次结构，提升压缩效率与实时性。**

- **链接: [https://arxiv.org/pdf/2511.14070v1](https://arxiv.org/pdf/2511.14070v1)**

> **作者:** Junsik Kim; Gun Bang; Soowoong Kim
>
> **摘要:** Hierarchical LiDAR geometry compression encodes voxel occupancies from low to high bit-depths, yet prior methods treat each depth independently and re-estimate local context from coordinates at every level, limiting compression efficiency. We present ELiC, a real-time framework that combines cross-bit-depth feature propagation, a Bag-of-Encoders (BoE) selection scheme, and a Morton-order-preserving hierarchy. Cross-bit-depth propagation reuses features extracted at denser, lower depths to support prediction at sparser, higher depths. BoE selects, per depth, the most suitable coding network from a small pool, adapting capacity to observed occupancy statistics without training a separate model for each level. The Morton hierarchy maintains global Z-order across depth transitions, eliminating per-level sorting and reducing latency. Together these components improve entropy modeling and computation efficiency, yielding state-of-the-art compression at real-time throughput on Ford and SemanticKITTI. Code and models will be released upon publication.
>
---
#### [new 148] PoCGM: Poisson-Conditioned Generative Model for Sparse-View CT Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文针对稀疏视角CT重建任务，解决因投影视图减少导致的伪影和细节丢失问题。提出PoCGM模型，将PFGM++改造成条件生成框架，利用稀疏数据引导重建，有效抑制伪影并保留结构细节。**

- **链接: [https://arxiv.org/pdf/2511.13967v1](https://arxiv.org/pdf/2511.13967v1)**

> **作者:** Changsheng Fang; Yongtong Liu; Bahareh Morovati; Shuo Han; Li Zhou; Hengyong Yu
>
> **备注:** 18th International Meeting on Fully 3D Image Reconstruction in Radiology and Nuclear Medicine, Shanghai, CHINA, 2025
>
> **摘要:** In computed tomography (CT), reducing the number of projection views is an effective strategy to lower radiation exposure and/or improve temporal resolution. However, this often results in severe aliasing artifacts and loss of structural details in reconstructed images, posing significant challenges for clinical applications. Inspired by the success of the Poisson Flow Generative Model (PFGM++) in natural image generation, we propose a PoCGM (Poisson-Conditioned Generative Model) to address the challenges of sparse-view CT reconstruction. Since PFGM++ was originally designed for unconditional generation, it lacks direct applicability to medical imaging tasks that require integrating conditional inputs. To overcome this limitation, the PoCGM reformulates PFGM++ into a conditional generative framework by incorporating sparse-view data as guidance during both training and sampling phases. By modeling the posterior distribution of full-view reconstructions conditioned on sparse observations, PoCGM effectively suppresses artifacts while preserving fine structural details. Qualitative and quantitative evaluations demonstrate that PoCGM outperforms the baselines, achieving improved artifact suppression, enhanced detail preservation, and reliable performance in dose-sensitive and time-critical imaging scenarios.
>
---
#### [new 149] AnaCP: Toward Upper-Bound Continual Learning via Analytic Contrastive Projection
- **分类: cs.LG; cs.CV**

- **简介: 该论文研究类增量学习（CIL）任务，解决传统方法因灾难性遗忘导致性能下降的问题。提出AnaCP方法，在不使用梯度训练的情况下实现特征表示的增量适应，同时保持解析分类器的高效性，达到联合训练的准确率上限。**

- **链接: [https://arxiv.org/pdf/2511.13880v1](https://arxiv.org/pdf/2511.13880v1)**

> **作者:** Saleh Momeni; Changnan Xiao; Bing Liu
>
> **摘要:** This paper studies the problem of class-incremental learning (CIL), a core setting within continual learning where a model learns a sequence of tasks, each containing a distinct set of classes. Traditional CIL methods, which do not leverage pre-trained models (PTMs), suffer from catastrophic forgetting (CF) due to the need to incrementally learn both feature representations and the classifier. The integration of PTMs into CIL has recently led to efficient approaches that treat the PTM as a fixed feature extractor combined with analytic classifiers, achieving state-of-the-art performance. However, they still face a major limitation: the inability to continually adapt feature representations to best suit the CIL tasks, leading to suboptimal performance. To address this, we propose AnaCP (Analytic Contrastive Projection), a novel method that preserves the efficiency of analytic classifiers while enabling incremental feature adaptation without gradient-based training, thereby eliminating the CF caused by gradient updates. Our experiments show that AnaCP not only outperforms existing baselines but also achieves the accuracy level of joint training, which is regarded as the upper bound of CIL.
>
---
#### [new 150] RoboTidy : A 3D Gaussian Splatting Household Tidying Benchmark for Embodied Navigation and Action
- **分类: cs.RO; cs.CV**

- **简介: 论文提出RoboTidy，一个用于语言引导家庭整理的3D基准，解决现有基准缺乏用户偏好建模、移动支持和泛化能力差的问题。工作包括构建500个逼真场景、提供6.4k操作轨迹和1.5k导航轨迹，支持VLA和VLN训练与评估，并实现在真实世界中的端到端测试。**

- **链接: [https://arxiv.org/pdf/2511.14161v1](https://arxiv.org/pdf/2511.14161v1)**

> **作者:** Xiaoquan Sun; Ruijian Zhang; Kang Pang; Bingchen Miao; Yuxiang Tan; Zhen Yang; Ming Li; Jiayu Chen
>
> **摘要:** Household tidying is an important application area, yet current benchmarks neither model user preferences nor support mobility, and they generalize poorly, making it hard to comprehensively assess integrated language-to-action capabilities. To address this, we propose RoboTidy, a unified benchmark for language-guided household tidying that supports Vision-Language-Action (VLA) and Vision-Language-Navigation (VLN) training and evaluation. RoboTidy provides 500 photorealistic 3D Gaussian Splatting (3DGS) household scenes (covering 500 objects and containers) with collisions, formulates tidying as an "Action (Object, Container)" list, and supplies 6.4k high-quality manipulation demonstration trajectories and 1.5k naviagtion trajectories to support both few-shot and large-scale training. We also deploy RoboTidy in the real world for object tidying, establishing an end-to-end benchmark for household tidying. RoboTidy offers a scalable platform and bridges a key gap in embodied AI by enabling holistic and realistic evaluation of language-guided robots.
>
---
#### [new 151] KANGURA: Kolmogorov-Arnold Network-Based Geometry-Aware Learning with Unified Representation Attention for 3D Modeling of Complex Structures
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出KANGURA模型，用于3D几何建模任务，解决现有方法难以捕捉复杂结构几何依赖的问题。通过Kolmogorov-Arnold网络与统一注意力机制，实现几何感知的表示学习，在MFC电极结构优化中表现优异。**

- **链接: [https://arxiv.org/pdf/2511.13798v1](https://arxiv.org/pdf/2511.13798v1)**

> **作者:** Mohammad Reza Shafie; Morteza Hajiabadi; Hamed Khosravi; Mobina Noori; Imtiaz Ahmed
>
> **摘要:** Microbial Fuel Cells (MFCs) offer a promising pathway for sustainable energy generation by converting organic matter into electricity through microbial processes. A key factor influencing MFC performance is the anode structure, where design and material properties play a crucial role. Existing predictive models struggle to capture the complex geometric dependencies necessary to optimize these structures. To solve this problem, we propose KANGURA: Kolmogorov-Arnold Network-Based Geometry-Aware Learning with Unified Representation Attention. KANGURA introduces a new approach to three-dimensional (3D) machine learning modeling. It formulates prediction as a function decomposition problem, where Kolmogorov-Arnold Network (KAN)- based representation learning reconstructs geometric relationships without a conventional multi- layer perceptron (MLP). To refine spatial understanding, geometry-disentangled representation learning separates structural variations into interpretable components, while unified attention mechanisms dynamically enhance critical geometric regions. Experimental results demonstrate that KANGURA outperforms over 15 state-of-the-art (SOTA) models on the ModelNet40 benchmark dataset, achieving 92.7% accuracy, and excels in a real-world MFC anode structure problem with 97% accuracy. This establishes KANGURA as a robust framework for 3D geometric modeling, unlocking new possibilities for optimizing complex structures in advanced manufacturing and quality-driven engineering applications.
>
---
#### [new 152] IMSE: Efficient U-Net-based Speech Enhancement using Inception Depthwise Convolution and Amplitude-Aware Linear Attention
- **分类: cs.SD; cs.AI; cs.CV**

- **简介: 该论文属于语音增强任务，旨在解决资源受限设备上模型轻量化与高性能难以兼顾的问题。提出IMSE网络，通过引入Amplitude-Aware Linear Attention和Inception Depthwise Convolution，显著减少参数量（-16.8%）并保持优异性能（PESQ=3.373）。**

- **链接: [https://arxiv.org/pdf/2511.14515v1](https://arxiv.org/pdf/2511.14515v1)**

> **作者:** Xinxin Tang; Bin Qin; Yufang Li
>
> **摘要:** Achieving a balance between lightweight design and high performance remains a significant challenge for speech enhancement (SE) tasks on resource-constrained devices. Existing state-of-the-art methods, such as MUSE, have established a strong baseline with only 0.51M parameters by introducing a Multi-path Enhanced Taylor (MET) transformer and Deformable Embedding (DE). However, an in-depth analysis reveals that MUSE still suffers from efficiency bottlenecks: the MET module relies on a complex "approximate-compensate" mechanism to mitigate the limitations of Taylor-expansion-based attention, while the offset calculation for deformable embedding introduces additional computational burden. This paper proposes IMSE, a systematically optimized and ultra-lightweight network. We introduce two core innovations: 1) Replacing the MET module with Amplitude-Aware Linear Attention (MALA). MALA fundamentally rectifies the "amplitude-ignoring" problem in linear attention by explicitly preserving the norm information of query vectors in the attention calculation, achieving efficient global modeling without an auxiliary compensation branch. 2) Replacing the DE module with Inception Depthwise Convolution (IDConv). IDConv borrows the Inception concept, decomposing large-kernel operations into efficient parallel branches (square, horizontal, and vertical strips), thereby capturing spectrogram features with extremely low parameter redundancy. Extensive experiments on the VoiceBank+DEMAND dataset demonstrate that, compared to the MUSE baseline, IMSE significantly reduces the parameter count by 16.8\% (from 0.513M to 0.427M) while achieving competitive performance comparable to the state-of-the-art on the PESQ metric (3.373). This study sets a new benchmark for the trade-off between model size and speech quality in ultra-lightweight speech enhancement.
>
---
#### [new 153] Attention via Synaptic Plasticity is All You Need: A Biologically Inspired Spiking Neuromorphic Transformer
- **分类: cs.NE; cs.AI; cs.CV; cs.ET; stat.ML**

- **简介: 论文提出S²TDPT，一种基于脉冲神经网络的类脑注意力机制Transformer，解决传统Transformer能耗高、不适用于神经形态硬件的问题。通过STDP实现自注意力，提升能效与可解释性，在图像分类任务中表现优异。**

- **链接: [https://arxiv.org/pdf/2511.14691v1](https://arxiv.org/pdf/2511.14691v1)**

> **作者:** Kallol Mondal; Ankush Kumar
>
> **备注:** 21 Pages, 5 Figures, 3 Table
>
> **摘要:** Attention is the brain's ability to selectively focus on a few specific aspects while ignoring irrelevant ones. This biological principle inspired the attention mechanism in modern Transformers. Transformers now underpin large language models (LLMs) such as GPT, but at the cost of massive training and inference energy, leading to a large carbon footprint. While brain attention emerges from neural circuits, Transformer attention relies on dot-product similarity to weight elements in the input sequence. Neuromorphic computing, especially spiking neural networks (SNNs), offers a brain-inspired path to energy-efficient intelligence. Despite recent work on attention-based spiking Transformers, the core attention layer remains non-neuromorphic. Current spiking attention (i) relies on dot-product or element-wise similarity suited to floating-point operations, not event-driven spikes; (ii) keeps attention matrices that suffer from the von Neumann bottleneck, limiting in-memory computing; and (iii) still diverges from brain-like computation. To address these issues, we propose the Spiking STDP Transformer (S$^{2}$TDPT), a neuromorphic Transformer that implements self-attention through spike-timing-dependent plasticity (STDP), embedding query--key correlations in synaptic weights. STDP, a core mechanism of memory and learning in the brain and widely studied in neuromorphic devices, naturally enables in-memory computing and supports non-von Neumann hardware. On CIFAR-10 and CIFAR-100, our model achieves 94.35\% and 78.08\% accuracy with only four timesteps and 0.49 mJ on CIFAR-100, an 88.47\% energy reduction compared to a standard ANN Transformer. Grad-CAM shows that the model attends to semantically relevant regions, enhancing interpretability. Overall, S$^{2}$TDPT illustrates how biologically inspired attention can yield energy-efficient, hardware-friendly, and explainable neuromorphic models.
>
---
#### [new 154] Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出CCoL框架，用于行为克隆任务，解决序列动作决策中的误差累积问题。通过视觉、语言与本体感知的连续协同学习，实现语义-物理对齐，提升动作执行的连贯性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.14396v1](https://arxiv.org/pdf/2511.14396v1)**

> **作者:** Xiuxiu Qi; Yu Yang; Jiannong Cao; Luyao Bai; Chongshan Fan; Chengtai Cao; Hongpeng Wang
>
> **备注:** Accepted at AAAI 2026, the Project website is available at https://qhemu.github.io/CCoL/
>
> **摘要:** Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states.
>
---
## 更新

#### [replaced 001] Benchmarking Deep Learning-Based Object Detection Models on Feature Deficient Astrophotography Imagery Dataset
- **分类: cs.CV; astro-ph.IM**

- **链接: [https://arxiv.org/pdf/2508.06537v2](https://arxiv.org/pdf/2508.06537v2)**

> **作者:** Shantanusinh Parmar
>
> **摘要:** Object detection models are typically trained on datasets like ImageNet, COCO, and PASCAL VOC, which focus on everyday objects. However, these lack signal sparsity found in non-commercial domains. MobilTelesco, a smartphone-based astrophotography dataset, addresses this by providing sparse night-sky images. We benchmark several detection models on it, highlighting challenges under feature-deficient conditions.
>
---
#### [replaced 002] AdCare-VLM: Towards a Unified and Pre-aligned Latent Representation for Healthcare Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.00275v2](https://arxiv.org/pdf/2505.00275v2)**

> **作者:** Md Asaduzzaman Jabin; Hanqi Jiang; Yiwei Li; Patrick Kaggwa; Eugene Douglass; Juliet N. Sekandi; Tianming Liu
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: 7th International Workshop on Large Scale Holistic Video Understanding: Toward Video Foundation Models
>
> **摘要:** Chronic diseases, including diabetes, hypertension, asthma, HIV-AIDS, epilepsy, and tuberculosis, necessitate rigorous adherence to medication to avert disease progression, manage symptoms, and decrease mortality rates. Adherence is frequently undermined by factors including patient behavior, caregiver support, elevated medical costs, and insufficient healthcare infrastructure. We propose AdCare-VLM, a specialized LLaVA-based multimodal large vision language model (LVLM) by introducing a unified visual latent space with pre-alignment to facilitate visual question answering (VQA) concerning medication adherence through patient videos. We employ a private dataset comprising 806 custom-annotated tuberculosis (TB) medication monitoring videos, which have been labeled by clinical experts, to fine-tune the model for adherence pattern detection. We present LLM-TB-VQA, a detailed medical adherence VQA dataset that encompasses positive, negative, and ambiguous adherence cases. Our method identifies correlations between visual features, such as the clear visibility of the patient's face, medication, water intake, and the act of ingestion, and their associated medical concepts in captions. This facilitates the integration of aligned visual-linguistic representations and improves multimodal interactions. Experimental results indicate that our method surpasses parameter-efficient fine-tuning (PEFT) enabled VLM models, such as LLaVA-V1.5 and Chat-UniVi, with absolute improvements ranging from 3.1% to 3.54% across pre-trained, regular, and low-rank adaptation (LoRA) configurations. Comprehensive ablation studies and attention map visualizations substantiate our approach, enhancing interpretability.
>
---
#### [replaced 003] CARScenes: Semantic VLM Dataset for Safe Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.10701v2](https://arxiv.org/pdf/2511.10701v2)**

> **作者:** Yuankai He; Weisong Shi
>
> **备注:** 8 pages, 6 figures, 7 tables
>
> **摘要:** CAR-Scenes is a frame-level dataset for autonomous driving that enables training and evaluation of vision-language models (VLMs) for interpretable, scene-level understanding. We annotate 5,192 images drawn from Argoverse 1, Cityscapes, KITTI, and nuScenes using a 28-key category/sub-category knowledge base covering environment, road geometry, background-vehicle behavior, ego-vehicle behavior, vulnerable road users, sensor states, and a discrete severity scale (1-10), totaling 350+ leaf attributes. Labels are produced by a GPT-4o-assisted vision-language pipeline with human-in-the-loop verification; we release the exact prompts, post-processing rules, and per-field baseline model performance. CAR-Scenes also provides attribute co-occurrence graphs and JSONL records that support semantic retrieval, dataset triage, and risk-aware scenario mining across sources. To calibrate task difficulty, we include reproducible, non-benchmark baselines, notably a LoRA-tuned Qwen2-VL-2B with deterministic decoding, evaluated via scalar accuracy, micro-averaged F1 for list attributes, and severity MAE/RMSE on a fixed validation split. We publicly release the annotation and analysis scripts, including graph construction and evaluation scripts, to enable explainable, data-centric workflows for future intelligent vehicles. Dataset: https://github.com/Croquembouche/CAR-Scenes
>
---
#### [replaced 004] Equivariant neural networks and equivarification
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [https://arxiv.org/pdf/1906.07172v5](https://arxiv.org/pdf/1906.07172v5)**

> **作者:** Erkao Bao; Jingcheng Lu; Linqi Song; Nathan Hart-Hodgson; William Parson; Yanheng Zhou
>
> **备注:** More explanations and experiments were added; a theoretical comparison with G-CNN was added
>
> **摘要:** Equivariant neural networks are a class of neural networks designed to preserve symmetries inherent in the data. In this paper, we introduce a general method for modifying a neural network to enforce equivariance, a process we refer to as equivarification. We further show that group convolutional neural networks (G-CNNs) arise as a special case of our framework.
>
---
#### [replaced 005] Foundation Models in Medical Imaging: A Review and Outlook
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.09095v4](https://arxiv.org/pdf/2506.09095v4)**

> **作者:** Vivien van Veldhuizen; Vanessa Botha; Chunyao Lu; Melis Erdal Cesur; Kevin Groot Lipman; Edwin D. de Jong; Hugo Horlings; Clárisa I. Sanchez; Cees G. M. Snoek; Lodewyk Wessels; Ritse Mann; Eric Marcus; Jonas Teuwen
>
> **摘要:** Foundation models (FMs) are changing the way medical images are analyzed by learning from large collections of unlabeled data. Instead of relying on manually annotated examples, FMs are pre-trained to learn general-purpose visual features that can later be adapted to specific clinical tasks with little additional supervision. In this review, we examine how FMs are being developed and applied in pathology, radiology, and ophthalmology, drawing on evidence from over 150 studies. We explain the core components of FM pipelines, including model architectures, self-supervised learning methods, and strategies for downstream adaptation. We also review how FMs are being used in each imaging domain and compare design choices across applications. Finally, we discuss key challenges and open questions to guide future research.
>
---
#### [replaced 006] Fine-Grained Representation for Lane Topology Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.12590v2](https://arxiv.org/pdf/2511.12590v2)**

> **作者:** Guoqing Xu; Yiheng Li; Yang Yang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Precise modeling of lane topology is essential for autonomous driving, as it directly impacts navigation and control decisions. Existing methods typically represent each lane with a single query and infer topological connectivity based on the similarity between lane queries. However, this kind of design struggles to accurately model complex lane structures, leading to unreliable topology prediction. In this view, we propose a Fine-Grained lane topology reasoning framework (TopoFG). It divides the procedure from bird's-eye-view (BEV) features to topology prediction via fine-grained queries into three phases, i.e., Hierarchical Prior Extractor (HPE), Region-Focused Decoder (RFD), and Robust Boundary-Point Topology Reasoning (RBTR). Specifically, HPE extracts global spatial priors from the BEV mask and local sequential priors from in-lane keypoint sequences to guide subsequent fine-grained query modeling. RFD constructs fine-grained queries by integrating the spatial and sequential priors. It then samples reference points in RoI regions of the mask and applies cross-attention with BEV features to refine the query representations of each lane. RBTR models lane connectivity based on boundary-point query features and further employs a topological denoising strategy to reduce matching ambiguity. By integrating spatial and sequential priors into fine-grained queries and applying a denoising strategy to boundary-point topology reasoning, our method precisely models complex lane structures and delivers trustworthy topology predictions. Extensive experiments on the OpenLane-V2 benchmark demonstrate that TopoFG achieves new state-of-the-art performance, with an OLS of 48.0 on subsetA and 45.4 on subsetB.
>
---
#### [replaced 007] Context-Aware Multimodal Representation Learning for Spatio-Temporally Explicit Environmental Modelling
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11706v2](https://arxiv.org/pdf/2511.11706v2)**

> **作者:** Julia Peters; Karin Mora; Miguel D. Mahecha; Chaonan Ji; David Montero; Clemens Mosig; Guido Kraemer
>
> **备注:** 10 pages (incliding 2 pages of references), 7 figures
>
> **摘要:** Earth observation (EO) foundation models have emerged as an effective approach to derive latent representations of the Earth system from various remote sensing sensors. These models produce embeddings that can be used as analysis-ready datasets, enabling the modelling of ecosystem dynamics without extensive sensor-specific preprocessing. However, existing models typically operate at fixed spatial or temporal scales, limiting their use for ecological analyses that require both fine spatial detail and high temporal fidelity. To overcome these limitations, we propose a representation learning framework that integrates different EO modalities into a unified feature space at high spatio-temporal resolution. We introduce the framework using Sentinel-1 and Sentinel-2 data as representative modalities. Our approach produces a latent space at native 10 m resolution and the temporal frequency of cloud-free Sentinel-2 acquisitions. Each sensor is first modeled independently to capture its sensor-specific characteristics. Their representations are then combined into a shared model. This two-stage design enables modality-specific optimisation and easy extension to new sensors, retaining pretrained encoders while retraining only fusion layers. This enables the model to capture complementary remote sensing data and to preserve coherence across space and time. Qualitative analyses reveal that the learned embeddings exhibit high spatial and semantic consistency across heterogeneous landscapes. Quantitative evaluation in modelling Gross Primary Production reveals that they encode ecologically meaningful patterns and retain sufficient temporal fidelity to support fine-scale analyses. Overall, the proposed framework provides a flexible, analysis-ready representation learning approach for environmental applications requiring diverse spatial and temporal resolutions.
>
---
#### [replaced 008] Vision Transformers with Self-Distilled Registers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.21501v3](https://arxiv.org/pdf/2505.21501v3)**

> **作者:** Yinjie Chen; Zipeng Yan; Chong Zhou; Bo Dai; Andrew F. Luo
>
> **备注:** NeurIPS 2025 Spotlight. Website: https://github.com/0raiser0/PH-Reg
>
> **摘要:** Vision Transformers (ViTs) have emerged as the dominant architecture for visual processing tasks, demonstrating excellent scalability with increased training data and model size. However, recent work has identified the emergence of artifact tokens in ViTs that are incongruous with local semantics. These anomalous tokens degrade ViT performance in tasks that require fine-grained localization or structural coherence. An effective mitigation of this issue is the addition of register tokens to ViTs, which implicitly "absorb" the artifact term during training. Given the availability of existing large-scale pre-trained ViTs, in this paper we seek add register tokens to existing models without needing to re-train from scratch, which is infeasible considering their size. Specifically, we propose Post Hoc Registers (PH-Reg), an efficient self-distillation method that integrates registers into an existing ViT without requiring additional labeled data and full retraining. PH-Reg initializes both teacher and student networks from the same pre-trained ViT. The teacher remains frozen and unmodified, while the student is augmented with randomly initialized register tokens. By applying test-time augmentation to the teacher's inputs, we generate denoised dense embeddings free of artifacts, which are then used to optimize only a small subset of unlocked student weights. We show that our approach can effectively reduce the number of artifact tokens, improving the segmentation and depth prediction of the student ViT under zero-shot and linear probing.
>
---
#### [replaced 009] VisAidMath: Benchmarking Visual-Aided Mathematical Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2410.22995v2](https://arxiv.org/pdf/2410.22995v2)**

> **作者:** Jingkun Ma; Runzhe Zhan; Yang Li; Di Sun; Hou Pong Chan; Lidia S. Chao; Derek F. Wong
>
> **备注:** 58 pages, 28 figures
>
> **摘要:** A hallmark of advanced artificial intelligence is the capacity to progress from passive visual perception to the strategic modification of visual information to facilitate complex reasoning. This advanced capability, however, remains critically underdeveloped in current Large Multi-modal Models (LMMs). The deficiency is often masked by evaluation metrics that prioritize final-answer accuracy, creating an illusion of competence where genuine reasoning is absent. Using the domain of geometric problem-solving as a precise instrument, we probe this issue through tasks that require constructing visual aids. To this end, we introduce \textbf{VisAidMath}, a challenging benchmark, and our novel Three-Layered Funnel Evaluation Framework. This framework moves beyond simple accuracy (ACCU) to scrutinize the generation of valid visual aids (PVA) and the soundness of subsequent reasoning steps (SPRS). Our extensive experiments on state-of-the-art models, including Doubao-Seed-1.6 and o4, reveal a profound ``Reasoning Illusion''. We observe that high surface-level accuracy conceals a catastrophic failure in the models' ability to produce valid visual aids or to reason from them. Our findings expose a fundamental schism between visual perception and logical deduction in modern LMMs. We host an evaluation platform at CodaBench for testing publicly. Homepage: https://nlp2ct.github.io/VisAidMathHomepage/ Evaluation: https://www.codabench.org/competitions/7634/
>
---
#### [replaced 010] Mapping Reduced Accessibility to WASH Facilities in Rohingya Refugee Camps With Sub-Meter Imagery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07231v3](https://arxiv.org/pdf/2511.07231v3)**

> **作者:** Kyeongjin Ahn; YongHun Suh; Sungwon Han; Jeasurk Yang; Hannes Taubenböck; Meeyoung Cha
>
> **备注:** 23 pages, 13 figures, 2 tables
>
> **摘要:** Access to Water, Sanitation, and Hygiene (WASH) services remains a major public health concern in refugee camps. This study introduces a remote sensing-driven framework to quantify WASH accessibility-specifically to water pumps, latrines, and bathing cubicles-in the Rohingya camps of Cox's Bazar, one of the world's most densely populated displacement settings. Detecting refugee shelters in such emergent camps presents substantial challenges, primarily due to their dense spatial configuration and irregular geometric patterns. Using sub-meter satellite images, we develop a semi-supervised segmentation framework that achieves an F1-score of 76.4% in detecting individual refugee shelters. Applying the framework across multi-year data reveals declining WASH accessibility, driven by rapid refugee population growth and reduced facility availability, rising from 25 people per facility in 2022 to 29.4 in 2025. Gender-disaggregated analysis further shows that women and girls experience reduced accessibility, in scenarios with inadequate safety-related segregation in WASH facilities. These findings suggest the importance of demand-responsive allocation strategies that can identify areas with under-served populations-such as women and girls-and ensure that limited infrastructure serves the greatest number of people in settings with fixed or shrinking budgets. We also discuss the value of high-resolution remote sensing and machine learning to detect inequality and inform equitable resource planning in complex humanitarian environments.
>
---
#### [replaced 011] PFAvatar: Pose-Fusion 3D Personalized Avatar Reconstruction from Real-World Outfit-of-the-Day Photos
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [https://arxiv.org/pdf/2511.12935v2](https://arxiv.org/pdf/2511.12935v2)**

> **作者:** Dianbing Xi; Guoyuan An; Jingsen Zhu; Zhijian Liu; Yuan Liu; Ruiyuan Zhang; Jiayuan Lu; Yuchi Huo; Rui Wang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** We propose PFAvatar (Pose-Fusion Avatar), a new method that reconstructs high-quality 3D avatars from Outfit of the Day(OOTD) photos, which exhibit diverse poses, occlusions, and complex backgrounds. Our method consists of two stages: (1) fine-tuning a pose-aware diffusion model from few-shot OOTD examples and (2) distilling a 3D avatar represented by a neural radiance field (NeRF). In the first stage, unlike previous methods that segment images into assets (e.g., garments, accessories) for 3D assembly, which is prone to inconsistency, we avoid decomposition and directly model the full-body appearance. By integrating a pre-trained ControlNet for pose estimation and a novel Condition Prior Preservation Loss (CPPL), our method enables end-to-end learning of fine details while mitigating language drift in few-shot training. Our method completes personalization in just 5 minutes, achieving a 48x speed-up compared to previous approaches. In the second stage, we introduce a NeRF-based avatar representation optimized by canonical SMPL-X space sampling and Multi-Resolution 3D-SDS. Compared to mesh-based representations that suffer from resolution-dependent discretization and erroneous occluded geometry, our continuous radiance field can preserve high-frequency textures (e.g., hair) and handle occlusions correctly through transmittance. Experiments demonstrate that PFAvatar outperforms state-of-the-art methods in terms of reconstruction fidelity, detail preservation, and robustness to occlusions/truncations, advancing practical 3D avatar generation from real-world OOTD albums. In addition, the reconstructed 3D avatar supports downstream applications such as virtual try-on, animation, and human video reenactment, further demonstrating the versatility and practical value of our approach.
>
---
#### [replaced 012] Sa2VA-i: Improving Sa2VA Results with Consistent Training and Inference
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.19082v2](https://arxiv.org/pdf/2509.19082v2)**

> **作者:** Alexey Nekrasov; Ali Athar; Daan de Geus; Alexander Hermans; Bastian Leibe
>
> **摘要:** Sa2VA is a recent model for language-guided dense grounding in images and video that achieves state-of-the-art results on multiple segmentation benchmarks and that has become widely popular. However, we found that Sa2VA does not perform according to its full potential for referring video object segmentation tasks. We identify inconsistencies between training and inference procedures as the key factor holding it back. To mitigate this issue, we propose an improved version of Sa2VA, Sa2VA-i, that rectifies these issues and improves the results. In fact, Sa2VA-i sets a new state of the art for multiple video benchmarks and achieves improvements of up to +11.6 J&F on MeViS, +1.4 on Ref-YT-VOS, +3.3 on Ref-DAVIS and +4.1 on ReVOS using the same Sa2VA checkpoints. With our fixes, the Sa2VA-i-1B model even performs on par with the original Sa2VA-26B model on the MeViS benchmark. We hope that this work will show the importance of seemingly trivial implementation details and that it will provide valuable insights for the referring video segmentation field. We provide the code and updated models at https://github.com/kumuji/sa2va-i
>
---
#### [replaced 013] MonoDream: Monocular Vision-Language Navigation with Panoramic Dreaming
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2508.02549v3](https://arxiv.org/pdf/2508.02549v3)**

> **作者:** Shuo Wang; Yongcai Wang; Zhaoxin Fan; Yucheng Wang; Maiyue Chen; Kaihui Wang; Zhizhong Su; Wanting Li; Xudong Cai; Yeying Jin; Deying Li
>
> **摘要:** Vision-Language Navigation (VLN) tasks often leverage panoramic RGB and depth inputs to provide rich spatial cues for action planning, but these sensors can be costly or less accessible in real-world deployments. Recent approaches based on Vision-Language Action (VLA) models achieve strong results with monocular input, yet they still lag behind methods using panoramic RGB-D information. We present MonoDream, a lightweight VLA framework that enables monocular agents to learn a Unified Navigation Representation (UNR). This shared feature representation jointly aligns navigation-relevant visual semantics (e.g., global layout, depth, and future cues) and language-grounded action intent, enabling more reliable action prediction. MonoDream further introduces Latent Panoramic Dreaming (LPD) tasks to supervise the UNR, which train the model to predict latent features of panoramic RGB and depth observations at both current and future steps based on only monocular input. Experiments on multiple VLN benchmarks show that MonoDream consistently improves monocular navigation performance and significantly narrows the gap with panoramic-based agents.
>
---
#### [replaced 014] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11478v2](https://arxiv.org/pdf/2511.11478v2)**

> **作者:** Nhat Chung; Taisei Hanyu; Toan Nguyen; Huy Le; Frederick Bumgarner; Duy Minh Ho Nguyen; Khoa Vo; Kashu Yamazaki; Chase Rainwater; Tung Kieu; Anh Nguyen; Ngan Le
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** As embodied agents operate in increasingly complex environments, the ability to perceive, track, and reason about individual object instances over time becomes essential, especially in tasks requiring sequenced interactions with visually similar objects. In these non-Markovian settings, key decision cues are often hidden in object-specific histories rather than the current scene. Without persistent memory of prior interactions (what has been interacted with, where it has been, or how it has changed) visuomotor policies may fail, repeat past actions, or overlook completed ones. To surface this challenge, we introduce LIBERO-Mem, a non-Markovian task suite for stress-testing robotic manipulation under object-level partial observability. It combines short- and long-horizon object tracking with temporally sequenced subgoals, requiring reasoning beyond the current frame. However, vision-language-action (VLA) models often struggle in such settings, with token scaling quickly becoming intractable even for tasks spanning just a few hundred frames. We propose Embodied-SlotSSM, a slot-centric VLA framework built for temporal scalability. It maintains spatio-temporally consistent slot identities and leverages them through two mechanisms: (1) slot-state-space modeling for reconstructing short-term history, and (2) a relational encoder to align the input tokens with action decoding. Together, these components enable temporally grounded, context-aware action prediction. Experiments show Embodied-SlotSSM's baseline performance on LIBERO-Mem and general tasks, offering a scalable solution for non-Markovian reasoning in object-centric robotic policies.
>
---
#### [replaced 015] LED: Light Enhanced Depth Estimation at Night
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2409.08031v3](https://arxiv.org/pdf/2409.08031v3)**

> **作者:** Simon de Moreau; Yasser Almehio; Andrei Bursuc; Hafid El-Idrissi; Bogdan Stanciulescu; Fabien Moutarde
>
> **备注:** BMVC 2025 (Poster). Code and dataset available on the project page : https://simondemoreau.github.io/LED/ 21 pages, 13 figures
>
> **摘要:** Nighttime camera-based depth estimation is a highly challenging task, especially for autonomous driving applications, where accurate depth perception is essential for ensuring safe navigation. Models trained on daytime data often fail in the absence of precise but costly LiDAR. Even vision foundation models trained on large amounts of data are unreliable in low-light conditions. In this work, we aim to improve the reliability of perception systems at night time. To this end, we introduce Light Enhanced Depth (LED), a novel, cost-effective approach that significantly improves depth estimation in low-light environments by harnessing a pattern projected by high definition headlights available in modern vehicles. LED leads to significant performance boosts across multiple depth-estimation architectures (encoder-decoder, Adabins, DepthFormer, Depth Anything V2) both on synthetic and real datasets. Furthermore, increased performances beyond illuminated areas reveal a holistic enhancement in scene understanding. Finally, we release the Nighttime Synthetic Drive Dataset, a synthetic and photo-realistic nighttime dataset, which comprises 49,990 comprehensively annotated images.
>
---
#### [replaced 016] Segmentation-Driven Initialization for Sparse-view 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.11853v2](https://arxiv.org/pdf/2509.11853v2)**

> **作者:** Yi-Hsin Li; Thomas Sikora; Sebastian Knorr; Mårten Sjöström
>
> **摘要:** Sparse-view synthesis remains a challenging problem due to the difficulty of recovering accurate geometry and appearance from limited observations. While recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time rendering with competitive quality, existing pipelines often rely on Structure-from-Motion (SfM) for camera pose estimation, an approach that struggles in genuinely sparse-view settings. Moreover, several SfM-free methods replace SfM with multi-view stereo (MVS) models, but generate massive numbers of 3D Gaussians by back-projecting every pixel into 3D space, leading to high memory costs. We propose Segmentation-Driven Initialization for Gaussian Splatting (SDI-GS), a method that mitigates inefficiency by leveraging region-based segmentation to identify and retain only structurally significant regions. This enables selective downsampling of the dense point cloud, preserving scene fidelity while substantially reducing Gaussian count. Experiments across diverse benchmarks show that SDI-GS reduces Gaussian count by up to 50% and achieves comparable or superior rendering quality in PSNR and SSIM, with only marginal degradation in LPIPS. It further enables faster training and lower memory footprint, advancing the practicality of 3DGS for constrained-view scenarios.
>
---
#### [replaced 017] DepthVision: Enabling Robust Vision-Language Models with GAN-Based LiDAR-to-RGB Synthesis for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.07463v2](https://arxiv.org/pdf/2509.07463v2)**

> **作者:** Sven Kirchner; Nils Purschke; Ross Greer; Alois C. Knoll
>
> **摘要:** Ensuring reliable autonomous operation when visual input is degraded remains a key challenge in intelligent vehicles and robotics. We present DepthVision, a multimodal framework that enables Vision--Language Models (VLMs) to exploit LiDAR data without any architectural changes or retraining. DepthVision synthesizes dense, RGB-like images from sparse LiDAR point clouds using a conditional GAN with an integrated refiner, and feeds these into off-the-shelf VLMs through their standard visual interface. A Luminance-Aware Modality Adaptation (LAMA) module fuses synthesized and real camera images by dynamically weighting each modality based on ambient lighting, compensating for degradation such as darkness or motion blur. This design turns LiDAR into a drop-in visual surrogate when RGB becomes unreliable, effectively extending the operational envelope of existing VLMs. We evaluate DepthVision on real and simulated datasets across multiple VLMs and safety-critical tasks, including vehicle-in-the-loop experiments. The results show substantial improvements in low-light scene understanding over RGB-only baselines while preserving full compatibility with frozen VLM architectures. These findings demonstrate that LiDAR-guided RGB synthesis is a practical pathway for integrating range sensing into modern vision-language systems for autonomous driving.
>
---
#### [replaced 018] Cross-Domain Few-Shot Learning with Coalescent Projections and Latent Space Reservation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.15243v2](https://arxiv.org/pdf/2507.15243v2)**

> **作者:** Naeem Paeedeh; Mahardhika Pratama; Imam Mustafa Kamal; Wolfgang Mayer; Jimmy Cao; Ryszard Kowlczyk
>
> **摘要:** Despite the progress in cross-domain few-shot learning, a model pre-trained with DINO combined with a prototypical classifier outperforms the latest SOTA methods. A crucial limitation that needs to be overcome is that updating too many parameters of the transformers leads to overfitting due to the scarcity of labeled samples. To address this challenge, we propose a new concept, coalescent projection, as an effective successor to soft prompts. Additionally, we propose a novel pseudo-class generation method, combined with self-supervised transformations, that relies solely on the base domain to prepare the network to encounter unseen samples from different domains. The proposed method exhibits its effectiveness in comprehensive experiments on the extreme domain-shift problem of the BSCD-FSL benchmark. Our code is published at \href{https://github.com/Naeem-Paeedeh/CPLSR}{https://github.com/Naeem-Paeedeh/CPLSR}.
>
---
#### [replaced 019] Region-Wise Correspondence Prediction between Manga Line Art Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.09501v3](https://arxiv.org/pdf/2509.09501v3)**

> **作者:** Yingxuan Li; Jiafeng Mao; Qianru Qiu; Yusuke Matsui
>
> **摘要:** Understanding region-wise correspondences between manga line art images is fundamental for high-level manga processing, supporting downstream tasks such as line art colorization and in-between frame generation. Unlike natural images that contain rich visual cues, manga line art consists only of sparse black-and-white strokes, making it challenging to determine which regions correspond across images. In this work, we introduce a new task: predicting region-wise correspondence between raw manga line art images without any annotations. To address this problem, we propose a Transformer-based framework trained on large-scale, automatically generated region correspondences. The model learns to suppress noisy matches and strengthen consistent structural relationships, resulting in robust patch-level feature alignment within and across images. During inference, our method segments each line art and establishes coherent region-level correspondences through edge-aware clustering and region matching. We construct manually annotated benchmarks for evaluation, and experiments across multiple datasets demonstrate both high patch-level accuracy and strong region-level correspondence performance, achieving 78.4-84.4% region-level accuracy. These results highlight the potential of our method for real-world manga and animation applications.
>
---
#### [replaced 020] 4D-VLA: Spatiotemporal Vision-Language-Action Pretraining with Cross-Scene Calibration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.22242v2](https://arxiv.org/pdf/2506.22242v2)**

> **作者:** Jiahui Zhang; Yurui Chen; Yueming Xu; Ze Huang; Yanpeng Zhou; Yu-Jie Yuan; Xinyue Cai; Guowei Huang; Xingyue Quan; Hang Xu; Li Zhang
>
> **摘要:** Leveraging diverse robotic data for pretraining remains a critical challenge. Existing methods typically model the dataset's action distribution using simple observations as inputs. However, these inputs are often incomplete, resulting in a dispersed conditional action distribution-an issue we refer to as coordinate system chaos and state chaos. This inconsistency significantly hampers pretraining efficiency. To address this, we propose 4D-VLA, a novel approach that effectively integrates 4D information into the input to mitigate these sources of chaos. Our model introduces depth and temporal information into visual features with sequential RGB-D inputs, aligning the coordinate systems of the robot and the scene. This alignment endows the model with strong spatiotemporal reasoning capabilities while minimizing training overhead. Additionally, we introduce memory bank sampling, a frame sampling strategy designed to extract informative frames from historical images, further improving effectiveness and efficiency. Experimental results demonstrate that our pretraining method and architectural components substantially enhance model performance. In both simulated and real-world experiments, our model achieves a significant increase in success rate over OpenVLA. To further assess spatial perception and generalization to novel views, we introduce MV-Bench, a multi-view simulation benchmark. Our model consistently outperforms existing methods, demonstrating stronger spatial understanding and adaptability.
>
---
#### [replaced 021] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02051v2](https://arxiv.org/pdf/2508.02051v2)**

> **作者:** Junhao Cai; Taegun An; Chengjun Jin; Sung Il Choi; Juhyun Park; Changhee Joo
>
> **备注:** Accepted at AAAI 2026 as a Conference Paper (Oral Presentation)
>
> **摘要:** Distributed multi-stage image compression -- where visual content traverses multiple processing nodes under varying quality requirements -- poses challenges. Progressive methods enable bitstream truncation but underutilize available compute resources; successive compression repeats costly pixel-domain operations and suffers cumulative quality loss and inefficiency; fixed-parameter models lack post-encoding flexibility. In this work, we developed the Hierarchical Cascade Framework (HCF) that achieves high rate-distortion performance and better computational efficiency through direct latent-space transformations across network nodes in distributed multi-stage image compression systems. Under HCF, we introduced policy-driven quantization control to optimize rate-distortion trade-offs, and established the edge quantization principle through differential entropy analysis. The configuration based on this principle demonstrates up to 0.6dB PSNR gains over other configurations. When comprehensively evaluated on the Kodak, CLIC, and CLIC2020-mobile datasets, HCF outperforms successive-compression methods by up to 5.56% BD-Rate in PSNR on CLIC, while saving up to 97.8% FLOPs, 96.5% GPU memory, and 90.0% execution time. It also outperforms state-of-the-art progressive compression methods by up to 12.64% BD-Rate on Kodak and enables retraining-free cross-quality adaptation with 7.13-10.87% BD-Rate reductions on CLIC2020-mobile.
>
---
#### [replaced 022] SF-Loc: A Visual Mapping and Geo-Localization System based on Sparse Visual Structure Frames
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.01500v3](https://arxiv.org/pdf/2412.01500v3)**

> **作者:** Yuxuan Zhou; Xingxing Li; Shengyu Li; Chunxi Xia; Xuanbin Wang; Shaoquan Feng
>
> **摘要:** For high-level geo-spatial applications and intelligent robotics, accurate global pose information is of crucial importance. Map-aided localization is a universal approach to overcome the limitations of global navigation satellite system (GNSS) in challenging environments. However, current solutions face challenges in terms of mapping flexibility, storage burden and re-localization performance. In this work, we present SF-Loc, a lightweight visual mapping and map-aided localization system, whose core idea is the map representation based on sparse frames with dense but compact depth, termed as visual structure frames. In the mapping phase, multi-sensor dense bundle adjustment (MS-DBA) is applied to construct geo-referenced visual structure frames. The local co-visbility is checked to keep the map sparsity and achieve incremental mapping. In the localization phase, coarse-to-fine vision-based localization is performed, in which multi-frame information and the map distribution are fully integrated. To be specific, the concept of spatially smoothed similarity (SSS) is proposed to overcome the place ambiguity, and pairwise frame matching is applied for efficient and robust pose estimation. Experimental results on the cross-season dataset verify the effectiveness of the system. In complex urban road scenarios, the map size is down to 3 MB per kilometer and stable decimeter-level re-localization can be achieved. The code will be made open-source soon (https://github.com/GREAT-WHU/SF-Loc).
>
---
#### [replaced 023] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.03779v2](https://arxiv.org/pdf/2507.03779v2)**

> **作者:** Jiaqi Zhang; Juntuo Wang; Zhixin Sun; John Zou; Randall Balestriero
>
> **备注:** Accepted by 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Large-scale vision foundation models such as DINOv2 boast impressive performances by leveraging massive architectures and training datasets. But numerous scenarios require practitioners to reproduce those pre-training solutions, such as on private data, new modalities, or simply for scientific questioning--which is currently extremely demanding computation-wise. We thus propose a novel pre-training strategy for DINOv2 that simultaneously accelerates convergence--and strengthens robustness to common corruptions as a by-product. Our approach involves a frequency filtering curriculum--low-frequency being seen first--and the Gaussian noise patching augmentation. Applied to a ViT-B/16 backbone trained on ImageNet-1K, while pre-training time and FLOPs are reduced by 1.6x and 2.25x, our method still achieves matching robustness in corruption benchmarks (ImageNet-C) and maintains competitive linear probing performance compared with baseline. This dual benefit of efficiency and robustness makes large-scale self-supervised foundation modeling more attainable, while opening the door to novel exploration around data curriculum and augmentation as means to improve self-supervised learning models robustness. The code is available at https://github.com/KevinZ0217/fast_dinov2
>
---
#### [replaced 024] MoReFun: Past-Movement Guided Motion Representation Learning for Future Motion Prediction and Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.02091v2](https://arxiv.org/pdf/2408.02091v2)**

> **作者:** Junyu Shi; Haoting Wu; Zhiyuan Zhang; Lijiang Liu; Yong Sun; Qiang Nie
>
> **摘要:** 3D human motion prediction aims to generate coherent future motions from observed sequences, yet existing end-to-end regression frameworks often fail to capture complex dynamics and tend to produce temporally inconsistent or static predictions-a limitation rooted in representation shortcutting, where models rely on superficial cues rather than learning meaningful motion structure. We propose a two-stage self-supervised framework that decouples representation learning from prediction. In the pretraining stage, the model performs unified past-future self-reconstruction, reconstructing the past sequence while recovering masked joints in the future sequence under full historical guidance. A velocity-based masking strategy selects highly dynamic joints, forcing the model to focus on informative motion components and internalize the statistical dependencies between past and future states without regression interference. In the fine-tuning stage, the pretrained model predicts the entire future sequence, now treated as fully masked, and is further equipped with a lightweight future-text prediction head for joint optimization of low-level motion prediction and high-level motion understanding. Experiments on Human3.6M, 3DPW, and AMASS show that our method reduces average prediction errors by 8.8% over state-of-the-art methods while achieving competitive future-motion understanding performance compared to LLM-based models. Code is available at: https://github.com/JunyuShi02/MoReFun
>
---
#### [replaced 025] How does My Model Fail? Automatic Identification and Interpretation of Physical Plausibility Failure Modes with Matryoshka Transcoders
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10094v2](https://arxiv.org/pdf/2511.10094v2)**

> **作者:** Yiming Tang; Abhijeet Sinha; Dianbo Liu
>
> **摘要:** Although recent generative models are remarkably capable of producing instruction-following and realistic outputs, they remain prone to notable physical plausibility failures. Though critical in applications, these physical plausibility errors often escape detection by existing evaluation methods. Furthermore, no framework exists for automatically identifying and interpreting specific physical error patterns in natural language, preventing targeted model improvements. We introduce Matryoshka Transcoders, a novel framework for the automatic discovery and interpretation of physical plausibility features in generative models. Our approach extends the Matryoshka representation learning paradigm to transcoder architectures, enabling hierarchical sparse feature learning at multiple granularity levels. By training on intermediate representations from a physical plausibility classifier and leveraging large multimodal models for interpretation, our method identifies diverse physics-related failure modes without manual feature engineering, achieving superior feature relevance and feature accuracy compared to existing approaches. We utilize the discovered visual patterns to establish a benchmark for evaluating physical plausibility in generative models. Our analysis of eight state-of-the-art generative models provides valuable insights into how these models fail to follow physical constraints, paving the way for further model improvements.
>
---
#### [replaced 026] GeoMVD: Geometry-Enhanced Multi-View Generation Model Based on Geometric Information Extraction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12204v2](https://arxiv.org/pdf/2511.12204v2)**

> **作者:** Jiaqi Wu; Yaosen Chen; Shuyuan Zhu
>
> **摘要:** Multi-view image generation holds significant application value in computer vision, particularly in domains like 3D reconstruction, virtual reality, and augmented reality. Most existing methods, which rely on extending single images, face notable computational challenges in maintaining cross-view consistency and generating high-resolution outputs. To address these issues, we propose the Geometry-guided Multi-View Diffusion Model, which incorporates mechanisms for extracting multi-view geometric information and adjusting the intensity of geometric features to generate images that are both consistent across views and rich in detail. Specifically, we design a multi-view geometry information extraction module that leverages depth maps, normal maps, and foreground segmentation masks to construct a shared geometric structure, ensuring shape and structural consistency across different views. To enhance consistency and detail restoration during generation, we develop a decoupled geometry-enhanced attention mechanism that strengthens feature focus on key geometric details, thereby improving overall image quality and detail preservation. Furthermore, we apply an adaptive learning strategy that fine-tunes the model to better capture spatial relationships and visual coherence between the generated views, ensuring realistic results. Our model also incorporates an iterative refinement process that progressively improves the output quality through multiple stages of image generation. Finally, a dynamic geometry information intensity adjustment mechanism is proposed to adaptively regulate the influence of geometric data, optimizing overall quality while ensuring the naturalness of generated images. More details can be found on the project page: https://sobeymil.github.io/GeoMVD.com.
>
---
#### [replaced 027] Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13015v2](https://arxiv.org/pdf/2511.13015v2)**

> **作者:** King-Man Tam; Satoshi Ikehata; Yuta Asano; Zhaoyi An; Rei Kawakami
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** Universal Photometric Stereo is a promising approach for recovering surface normals without strict lighting assumptions. However, it struggles when multi-illumination cues are unreliable, such as under biased lighting or in shadows or self-occluded regions of complex in-the-wild scenes. We propose GeoUniPS, a universal photometric stereo network that integrates synthetic supervision with high-level geometric priors from large-scale 3D reconstruction models pretrained on massive in-the-wild data. Our key insight is that these 3D reconstruction models serve as visual-geometry foundation models, inherently encoding rich geometric knowledge of real scenes. To leverage this, we design a Light-Geometry Dual-Branch Encoder that extracts both multi-illumination cues and geometric priors from the frozen 3D reconstruction model. We also address the limitations of the conventional orthographic projection assumption by introducing the PS-Perp dataset with realistic perspective projection to enable learning of spatially varying view directions. Extensive experiments demonstrate that GeoUniPS delivers state-of-the-arts performance across multiple datasets, both quantitatively and qualitatively, especially in the complex in-the-wild scenes.
>
---
#### [replaced 028] Exploring Convolutional Neural Networks for Rice Grain Classification: An Explainable AI Approach
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.05513v3](https://arxiv.org/pdf/2505.05513v3)**

> **作者:** Muhammad Junaid Asif; Hamza Khan; Rabia Tehseen; Syed Tahir Hussain Rizvi; Mujtaba Asad; Rana Fayyaz Ahmad; Shazia Saqib
>
> **摘要:** Rice is an essential staple food worldwide that is important in promoting international trade, economic growth, and nutrition. Asian countries such as China, India, Pakistan, Thailand, Vietnam, and Indonesia are notable for their significant contribution to the cultivation and utilization of rice. These nations are also known for cultivating different rice grains, including short and long grains. These sizes are further classified as basmati, jasmine, kainat saila, ipsala, arborio, etc., catering to diverse culinary preferences and cultural traditions. For both local and international trade, inspecting and maintaining the quality of rice grains to satisfy customers and preserve a country's reputation is necessary. Manual quality check and classification is quite a laborious and time-consuming process. It is also highly prone to mistakes. Therefore, an automatic solution must be proposed for the effective and efficient classification of different varieties of rice grains. This research paper presents an automatic framework based on a convolutional neural network (CNN) for classifying different varieties of rice grains. We evaluated the proposed model based on performance metrics such as accuracy, recall, precision, and F1-Score. The CNN model underwent rigorous training and validation, achieving a remarkable accuracy rate and a perfect area under each class's Receiver Operating Characteristic (ROC) curve. The confusion matrix analysis confirmed the model's effectiveness in distinguishing between the different rice varieties, indicating minimal misclassifications. Additionally, the integration of explainability techniques such as LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) provided valuable insights into the model's decision-making process, revealing how specific features of the rice grains influenced classification outcomes.
>
---
#### [replaced 029] CVChess: A Deep Learning Framework for Converting Chessboard Images to Forsyth-Edwards Notation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.11522v3](https://arxiv.org/pdf/2511.11522v3)**

> **作者:** Luthira Abeykoon; Ved Patel; Gawthaman Senthilvelan; Darshan Kasundra
>
> **摘要:** Chess has experienced a large increase in viewership since the pandemic, driven largely by the accessibility of online learning platforms. However, no equivalent assistance exists for physical chess games, creating a divide between analog and digital chess experiences. This paper presents CVChess, a deep learning framework for converting chessboard images to Forsyth-Edwards Notation (FEN), which is later input into online chess engines to provide you with the best next move. Our approach employs a convolutional neural network (CNN) with residual layers to perform piece recognition from smartphone camera images. The system processes RGB images of a physical chess board through a multistep process: image preprocessing using the Hough Line Transform for edge detection, projective transform to achieve a top-down board alignment, segmentation into 64 individual squares, and piece classification into 13 classes (6 unique white pieces, 6 unique black pieces and an empty square) using the residual CNN. Residual connections help retain low-level visual features while enabling deeper feature extraction, improving accuracy and stability during training. We train and evaluate our model using the Chess Recognition Dataset (ChessReD), containing 10,800 annotated smartphone images captured under diverse lighting conditions and angles. The resulting classifications are encoded as an FEN string, which can be fed into a chess engine to generate the most optimal move
>
---
#### [replaced 030] Efficient Fourier Filtering Network with Contrastive Learning for AAV-based Unaligned Bimodal Salient Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.03728v3](https://arxiv.org/pdf/2411.03728v3)**

> **作者:** Pengfei Lyu; Pak-Hei Yeung; Xiaosheng Yu; Xiufei Cheng; Chengdong Wu; Jagath C. Rajapakse
>
> **备注:** Accepted by TGRS 2025
>
> **摘要:** Autonomous aerial vehicle (AAV)-based bi-modal salient object detection (BSOD) aims to segment salient objects in a scene utilizing complementary cues in unaligned RGB and thermal image pairs. However, the high computational expense of existing AAV-based BSOD models limits their applicability to real-world AAV devices. To address this problem, we propose an efficient Fourier filter network with contrastive learning that achieves both real-time and accurate performance. Specifically, we first design a semantic contrastive alignment loss to align the two modalities at the semantic level, which facilitates mutual refinement in a parameter-free way. Second, inspired by the fast Fourier transform that obtains global relevance in linear complexity, we propose synchronized alignment fusion, which aligns and fuses bi-modal features in the channel and spatial dimensions by a hierarchical filtering mechanism. Our proposed model, AlignSal, reduces the number of parameters by 70.0%, decreases the floating point operations by 49.4%, and increases the inference speed by 152.5% compared to the cutting-edge BSOD model (i.e., MROS). Extensive experiments on the AAV RGB-T 2400 and seven bi-modal dense prediction datasets demonstrate that AlignSal achieves both real-time inference speed and better performance and generalizability compared to nineteen state-of-the-art models across most evaluation metrics. In addition, our ablation studies further verify AlignSal's potential in boosting the performance of existing aligned BSOD models on AAV-based unaligned data. The code is available at: https://github.com/JoshuaLPF/AlignSal.
>
---
#### [replaced 031] ODE$_t$(ODE$_l$): Shortcutting the Time and the Length in Diffusion and Flow Models for Faster Sampling
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.21714v3](https://arxiv.org/pdf/2506.21714v3)**

> **作者:** Denis Gudovskiy; Wenzhao Zheng; Tomoyuki Okuno; Yohei Nakata; Kurt Keutzer
>
> **备注:** Accepted to WACV 2026. Preprint. Github page: github.com/gudovskiy/odelt
>
> **摘要:** Continuous normalizing flows (CNFs) and diffusion models (DMs) generate high-quality data from a noise distribution. However, their sampling process demands multiple iterations to solve an ordinary differential equation (ODE) with high computational complexity. State-of-the-art methods focus on reducing the number of discrete time steps during sampling to improve efficiency. In this work, we explore a complementary direction in which the quality-complexity tradeoff can also be controlled in terms of the neural network length. We achieve this by rewiring the blocks in the transformer-based architecture to solve an inner discretized ODE w.r.t. its depth. Then, we apply a length consistency term during flow matching training, and as a result, the sampling can be performed with an arbitrary number of time steps and transformer blocks. Unlike others, our ODE$_t$(ODE$_l$) approach is solver-agnostic in time dimension and reduces both latency and, importantly, memory usage. CelebA-HQ and ImageNet generation experiments show a latency reduction of up to $2\times$ in the most efficient sampling mode, and FID improvement of up to $2.8$ points for high-quality sampling when applied to prior methods. We open-source our code and checkpoints at github.com/gudovskiy/odelt.
>
---
#### [replaced 032] DINO-Detect: A Simple yet Effective Framework for Blur-Robust AI-Generated Image Detection
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.12511v2](https://arxiv.org/pdf/2511.12511v2)**

> **作者:** Jialiang Shen; Jiyang Zheng; Yunqi Xue; Huajie Chen; Yu Yao; Hui Kang; Ruiqi Liu; Helin Gong; Yang Yang; Dadong Wang; Tongliang Liu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** With growing concerns over image authenticity and digital safety, the field of AI-generated image (AIGI) detection has progressed rapidly. Yet, most AIGI detectors still struggle under real-world degradations, particularly motion blur, which frequently occurs in handheld photography, fast motion, and compressed video. Such blur distorts fine textures and suppresses high-frequency artifacts, causing severe performance drops in real-world settings. We address this limitation with a blur-robust AIGI detection framework based on teacher-student knowledge distillation. A high-capacity teacher (DINOv3), trained on clean (i.e., sharp) images, provides stable and semantically rich representations that serve as a reference for learning. By freezing the teacher to maintain its generalization ability, we distill its feature and logit responses from sharp images to a student trained on blurred counterparts, enabling the student to produce consistent representations under motion degradation. Extensive experiments benchmarks show that our method achieves state-of-the-art performance under both motion-blurred and clean conditions, demonstrating improved generalization and real-world applicability. Source codes will be released at: https://github.com/JiaLiangShen/Dino-Detect-for-blur-robust-AIGC-Detection.
>
---
#### [replaced 033] MOON Embedding: Multimodal Representation Learning for E-commerce Search Advertising
- **分类: cs.IR; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.11305v2](https://arxiv.org/pdf/2511.11305v2)**

> **作者:** Chenghan Fu; Daoze Zhang; Yukang Lin; Zhanheng Nie; Xiang Zhang; Jianyu Liu; Yueran Liu; Wanxian Guan; Pengjie Wang; Jian Xu; Bo Zheng
>
> **备注:** 31 pages, 12 figures
>
> **摘要:** We introduce MOON, our comprehensive set of sustainable iterative practices for multimodal representation learning for e-commerce applications. MOON has already been fully deployed across all stages of Taobao search advertising system, including retrieval, relevance, ranking, and so on. The performance gains are particularly significant on click-through rate (CTR) prediction task, which achieves an overall +20.00% online CTR improvement. Over the past three years, this project has delivered the largest improvement on CTR prediction task and undergone five full-scale iterations. Throughout the exploration and iteration of our MOON, we have accumulated valuable insights and practical experience that we believe will benefit the research community. MOON contains a three-stage training paradigm of "Pretraining, Post-training, and Application", allowing effective integration of multimodal representations with downstream tasks. Notably, to bridge the misalignment between the objectives of multimodal representation learning and downstream training, we define the exchange rate to quantify how effectively improvements in an intermediate metric can translate into downstream gains. Through this analysis, we identify the image-based search recall as a critical intermediate metric guiding the optimization of multimodal models. Over three years and five iterations, MOON has evolved along four critical dimensions: data processing, training strategy, model architecture, and downstream application. The lessons and insights gained through the iterative improvements will also be shared. As part of our exploration into scaling effects in the e-commerce field, we further conduct a systematic study of the scaling laws governing multimodal representation learning, examining multiple factors such as the number of training tokens, negative samples, and the length of user behavior sequences.
>
---
#### [replaced 034] Divide and Merge: Motion and Semantic Learning in End-to-End Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.07631v3](https://arxiv.org/pdf/2502.07631v3)**

> **作者:** Yinzhe Shen; Omer Sahin Tas; Kaiwen Wang; Royden Wagner; Christoph Stiller
>
> **摘要:** Perceiving the environment and its changes over time corresponds to two fundamental yet heterogeneous types of information: semantics and motion. Previous end-to-end autonomous driving works represent both types of information in a single feature vector. However, including motion related tasks, such as prediction and planning, impairs detection and tracking performance, a phenomenon known as negative transfer in multi-task learning. To address this issue, we propose Neural-Bayes motion decoding, a novel parallel detection, tracking, and prediction method that separates semantic and motion learning. Specifically, we employ a set of learned motion queries that operate in parallel with detection and tracking queries, sharing a unified set of recursively updated reference points. Moreover, we employ interactive semantic decoding to enhance information exchange in semantic tasks, promoting positive transfer. Experiments on the nuScenes dataset with UniAD and SparseDrive confirm the effectiveness of our divide and merge approach, resulting in performance improvements across perception, prediction, and planning. Our code is available at https://github.com/shenyinzhe/DMAD.
>
---
#### [replaced 035] A Style is Worth One Code: Unlocking Code-to-Style Image Generation with Discrete Style Space
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10555v3](https://arxiv.org/pdf/2511.10555v3)**

> **作者:** Huijie Liu; Shuhao Cui; Haoxiang Cao; Shuai Ma; Kai Wu; Guoliang Kang
>
> **备注:** Homepage: https://github.com/Kwai-Kolors.github.io/CoTyle Code: https://github.com/Kwai-Kolors/CoTyle Demo: https://huggingface.co/spaces/Kwai-Kolors/CoTyle
>
> **摘要:** Innovative visual stylization is a cornerstone of artistic creation, yet generating novel and consistent visual styles remains a significant challenge. Existing generative approaches typically rely on lengthy textual prompts, reference images, or parameter-efficient fine-tuning to guide style-aware image generation, but often struggle with style consistency, limited creativity, and complex style representations. In this paper, we affirm that a style is worth one numerical code by introducing the novel task, code-to-style image generation, which produces images with novel, consistent visual styles conditioned solely on a numerical style code. To date, this field has only been primarily explored by the industry (e.g., Midjourney), with no open-source research from the academic community. To fill this gap, we propose CoTyle, the first open-source method for this task. Specifically, we first train a discrete style codebook from a collection of images to extract style embeddings. These embeddings serve as conditions for a text-to-image diffusion model (T2I-DM) to generate stylistic images. Subsequently, we train an autoregressive style generator on the discrete style embeddings to model their distribution, allowing the synthesis of novel style embeddings. During inference, a numerical style code is mapped to a unique style embedding by the style generator, and this embedding guides the T2I-DM to generate images in the corresponding style. Unlike existing methods, our method offers unparalleled simplicity and diversity, unlocking a vast space of reproducible styles from minimal input. Extensive experiments validate that CoTyle effectively turns a numerical code into a style controller, demonstrating a style is worth one code.
>
---
#### [replaced 036] OG-VLA: Orthographic Image Generation for 3D-Aware Vision-Language Action Model
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.01196v2](https://arxiv.org/pdf/2506.01196v2)**

> **作者:** Ishika Singh; Ankit Goyal; Stan Birchfield; Dieter Fox; Animesh Garg; Valts Blukis
>
> **备注:** 13 pages
>
> **摘要:** We introduce OG-VLA, a novel architecture and learning framework that combines the generalization strengths of Vision Language Action models (VLAs) with the robustness of 3D-aware policies. We address the challenge of mapping natural language instructions and one or more RGBD observations to quasi-static robot actions. 3D-aware robot policies achieve state-of-the-art performance on precise robot manipulation tasks, but struggle with generalization to unseen instructions, scenes, and objects. On the other hand, VLAs excel at generalizing across instructions and scenes, but can be sensitive to camera and robot pose variations. We leverage prior knowledge embedded in language and vision foundation models to improve generalization of 3D-aware keyframe policies. OG-VLA unprojects input observations from diverse views into a point cloud which is then rendered from canonical orthographic views, ensuring input view invariance and consistency between input and output spaces. These canonical views are processed with a vision backbone, a Large Language Model (LLM), and an image diffusion model to generate images that encode the next position and orientation of the end-effector on the input scene. Evaluations on the Arnold and Colosseum benchmarks demonstrate state-of-the-art generalization to unseen environments, with over 40% relative improvements while maintaining robust performance in seen settings. We also show real-world adaption in 3 to 5 demonstrations along with strong generalization. Videos and resources at https://og-vla.github.io/
>
---
#### [replaced 037] RynnEC: Bringing MLLMs into Embodied World
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2508.14160v2](https://arxiv.org/pdf/2508.14160v2)**

> **作者:** Ronghao Dang; Yuqian Yuan; Yunxuan Mao; Kehan Li; Jiangpin Liu; Zhikai Wang; Xin Li; Fan Wang; Deli Zhao
>
> **备注:** The technical report of RynnEC, an embodied cognition MLLM
>
> **摘要:** We introduce RynnEC, a video multimodal large language model designed for embodied cognition. Built upon a general-purpose vision-language foundation model, RynnEC incorporates a region encoder and a mask decoder, enabling flexible region-level video interaction. Despite its compact architecture, RynnEC achieves state-of-the-art performance in object property understanding, object segmentation, and spatial reasoning. Conceptually, it offers a region-centric video paradigm for the brain of embodied agents, providing fine-grained perception of the physical world and enabling more precise interactions. To mitigate the scarcity of annotated 3D datasets, we propose an egocentric video based pipeline for generating embodied cognition data. Furthermore, we introduce RynnEC-Bench, a region-centered benchmark for evaluating embodied cognitive capabilities. We anticipate that RynnEC will advance the development of general-purpose cognitive cores for embodied agents and facilitate generalization across diverse embodied tasks. The code, model checkpoints, and benchmark are available at: https://github.com/alibaba-damo-academy/RynnEC
>
---
#### [replaced 038] SpeeDe3DGS: Speedy Deformable 3D Gaussian Splatting with Temporal Pruning and Motion Grouping
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.07917v2](https://arxiv.org/pdf/2506.07917v2)**

> **作者:** Allen Tu; Haiyang Ying; Alex Hanson; Yonghan Lee; Tom Goldstein; Matthias Zwicker
>
> **备注:** Project Page: https://speede3dgs.github.io/
>
> **摘要:** Dynamic extensions of 3D Gaussian Splatting (3DGS) achieve high-quality reconstructions through neural motion fields, but per-Gaussian neural inference makes these models computationally expensive. Building on DeformableGS, we introduce Speedy Deformable 3D Gaussian Splatting (SpeeDe3DGS), which bridges this efficiency-fidelity gap through three complementary modules: Temporal Sensitivity Pruning (TSP) removes low-impact Gaussians via temporally aggregated sensitivity analysis, Temporal Sensitivity Sampling (TSS) perturbs timestamps to suppress floaters and improve temporal coherence, and GroupFlow distills the learned deformation field into shared SE(3) transformations for efficient groupwise motion. On the 50 dynamic scenes in MonoDyGauBench, integrating TSP and TSS into DeformableGS accelerates rendering by 6.78$\times$ on average while maintaining neural-field fidelity and using 10$\times$ fewer primitives. Adding GroupFlow culminates in 13.71$\times$ faster rendering and 2.53$\times$ shorter training, surpassing all baselines in speed while preserving superior image quality.
>
---
#### [replaced 039] GMAT: Grounded Multi-Agent Clinical Description Generation for Text Encoder in Vision-Language MIL for Whole Slide Image Classification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.01293v2](https://arxiv.org/pdf/2508.01293v2)**

> **作者:** Ngoc Bui Lam Quang; Nam Le Nguyen Binh; Thanh-Huy Nguyen; Le Thien Phuc Nguyen; Quan Nguyen; Ulas Bagci
>
> **备注:** Acccepted in MICCAI Workshop 2025
>
> **摘要:** Multiple Instance Learning (MIL) is the leading approach for whole slide image (WSI) classification, enabling efficient analysis of gigapixel pathology slides. Recent work has introduced vision-language models (VLMs) into MIL pipelines to incorporate medical knowledge through text-based class descriptions rather than simple class names. However, when these methods rely on large language models (LLMs) to generate clinical descriptions or use fixed-length prompts to represent complex pathology concepts, the limited token capacity of VLMs often constrains the expressiveness and richness of the encoded class information. Additionally, descriptions generated solely by LLMs may lack domain grounding and fine-grained medical specificity, leading to suboptimal alignment with visual features. To address these challenges, we propose a vision-language MIL framework with two key contributions: (1) A grounded multi-agent description generation system that leverages curated pathology textbooks and agent specialization (e.g., morphology, spatial context) to produce accurate and diverse clinical descriptions; (2) A text encoding strategy using a list of descriptions rather than a single prompt, capturing fine-grained and complementary clinical signals for better alignment with visual features. Integrated into a VLM-MIL pipeline, our approach shows improved performance over single-prompt class baselines and achieves results comparable to state-of-the-art models, as demonstrated on renal and lung cancer datasets.
>
---
#### [replaced 040] Crossing Borders: A Multimodal Challenge for Indian Poetry Translation and Image Generation
- **分类: cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13689v2](https://arxiv.org/pdf/2511.13689v2)**

> **作者:** Sofia Jamil; Kotla Sai Charan; Sriparna Saha; Koustava Goswami; Joseph K J
>
> **摘要:** Indian poetry, known for its linguistic complexity and deep cultural resonance, has a rich and varied heritage spanning thousands of years. However, its layered meanings, cultural allusions, and sophisticated grammatical constructions often pose challenges for comprehension, especially for non-native speakers or readers unfamiliar with its context and language. Despite its cultural significance, existing works on poetry have largely overlooked Indian language poems. In this paper, we propose the Translation and Image Generation (TAI) framework, leveraging Large Language Models (LLMs) and Latent Diffusion Models through appropriate prompt tuning. Our framework supports the United Nations Sustainable Development Goals of Quality Education (SDG 4) and Reduced Inequalities (SDG 10) by enhancing the accessibility of culturally rich Indian-language poetry to a global audience. It includes (1) a translation module that uses an Odds Ratio Preference Alignment Algorithm to accurately translate morphologically rich poetry into English, and (2) an image generation module that employs a semantic graph to capture tokens, dependencies, and semantic relationships between metaphors and their meanings, to create visually meaningful representations of Indian poems. Our comprehensive experimental evaluation, including both human and quantitative assessments, demonstrates the superiority of TAI Diffusion in poem image generation tasks, outperforming strong baselines. To further address the scarcity of resources for Indian-language poetry, we introduce the Morphologically Rich Indian Language Poems MorphoVerse Dataset, comprising 1,570 poems across 21 low-resource Indian languages. By addressing the gap in poetry translation and visual comprehension, this work aims to broaden accessibility and enrich the reader's experience.
>
---
#### [replaced 041] SMOL-MapSeg: Show Me One Label as prompt
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.05501v2](https://arxiv.org/pdf/2508.05501v2)**

> **作者:** Yunshuang Yuan; Frank Thiemann; Thorsten Dahms; Monika Sester
>
> **摘要:** Historical maps offer valuable insights into changes on Earth's surface but pose challenges for modern segmentation models due to inconsistent visual styles and symbols. While deep learning models such as UNet and pre-trained foundation models perform well in domains like autonomous driving and medical imaging, they struggle with the variability of historical maps, where similar concepts appear in diverse forms. To address this issue, we propose On-Need Declarative (OND) knowledge-based prompting, a method that provides explicit image-label pair prompts to guide models in linking visual patterns with semantic concepts. This enables users to define and segment target concepts on demand, supporting flexible, concept-aware segmentation. Our approach replaces the prompt encoder of the Segment Anything Model (SAM) with the OND prompting mechanism and fine-tunes it on historical maps, creating SMOL-MapSeg (Show Me One Label). Unlike existing SAM-based fine-tuning methods that are class-agnostic or restricted to fixed classes, SMOL-MapSeg supports class-aware segmentation across arbitrary datasets. Experiments show that SMOL-MapSeg accurately segments user-defined classes and substantially outperforms baseline models. Furthermore, it demonstrates strong generalization even with minimal training data, highlighting its potential for scalable and adaptable historical map analysis.
>
---
#### [replaced 042] GAIS: Frame-Level Gated Audio-Visual Integration with Semantic Variance-Scaled Perturbation for Text-Video Retrieval
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.01711v2](https://arxiv.org/pdf/2508.01711v2)**

> **作者:** Bowen Yang; Yun Cao; Chen He; Xiaosu Su
>
> **备注:** 13 pages
>
> **摘要:** Text-to-video retrieval requires precise alignment between language and temporally rich audio-video signals. However, existing methods often emphasize visual cues while underutilizing audio semantics or relying on coarse fusion strategies, resulting in suboptimal multimodal representations. We introduce GAIS, a retrieval framework that strengthens multimodal alignment from both representation and regularization perspectives. First, a Frame-level Gated Fusion (FGF) module adaptively integrates audio-visual features under textual guidance, enabling fine-grained temporal selection of informative frames. Second, a Semantic Variance-Scaled Perturbation (SVSP) mechanism regularizes the text embedding space by controlling perturbation magnitude in a semantics-aware manner. These two modules are complementary: FGF minimizes modality gaps through selective fusion, while SVSP improves embedding stability and discrimination. Extensive experiments on MSR-VTT, DiDeMo, LSMDC, and VATEX demonstrate that GAIS consistently outperforms strong baselines across multiple retrieval metrics while maintaining notable computational efficiency.
>
---
#### [replaced 043] Improved Sample Complexity Bounds for Diffusion Model Training
- **分类: cs.LG; cs.CV; cs.IT; math.ST; stat.ML**

- **链接: [https://arxiv.org/pdf/2311.13745v4](https://arxiv.org/pdf/2311.13745v4)**

> **作者:** Shivam Gupta; Aditya Parulekar; Eric Price; Zhiyang Xun
>
> **备注:** Bugfix
>
> **摘要:** Diffusion models have become the most popular approach to deep generative modeling of images, largely due to their empirical performance and reliability. From a theoretical standpoint, a number of recent works have studied the iteration complexity of sampling, assuming access to an accurate diffusion model. In this work, we focus on understanding the sample complexity of training such a model; how many samples are needed to learn an accurate diffusion model using a sufficiently expressive neural network? Prior work showed bounds polynomial in the dimension, desired Total Variation error, and Wasserstein error. We show an exponential improvement in the dependence on Wasserstein error and depth, along with improved dependencies on other relevant parameters.
>
---
#### [replaced 044] Governance-Ready Small Language Models for Medical Imaging: Prompting, Abstention, and PACS Integration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13378v3](https://arxiv.org/pdf/2508.13378v3)**

> **作者:** Yiting Wang; Ziwei Wang; Di Zhu; Jiachen Zhong; Weiyi Li
>
> **备注:** Under Review
>
> **摘要:** Small Language Models (SLMs) are a practical option for narrow, workflow-relevant medical imaging utilities where privacy, latency, and cost dominate. We present a governance-ready recipe that combines prompt scaffolds, calibrated abstention, and standards-compliant integration into Picture Archiving and Communication Systems (PACS). Our focus is the assistive task of AP/PA view tagging for chest radiographs. Using four deployable SLMs (Qwen2.5-VL, MiniCPM-V, Gemma 7B, LLaVA 7B) on NIH Chest X-ray, we provide illustrative evidence: reflection-oriented prompts benefit lighter models, whereas stronger baselines are less sensitive. Beyond accuracy, we operationalize abstention, expected calibration error, and oversight burden, and we map outputs to DICOM tags, HL7 v2 messages, and FHIR ImagingStudy. The contribution is a prompt-first deployment framework, an operations playbook for calibration, logging, and change management, and a clear pathway from pilot utilities to reader studies without over-claiming clinical validation. We additionally specify a human-factors RACI, stratified calibration for dataset shift, and an auditable evidence pack to support local governance reviews.
>
---
#### [replaced 045] Automatic Intermodal Loading Unit Identification using Computer Vision: A Scoping Review
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17707v2](https://arxiv.org/pdf/2509.17707v2)**

> **作者:** Emre Gülsoylu; Alhassan Abdelhalim; Derya Kara Boztas; Ole Grasse; Carlos Jahn; Simone Frintrop; Janick Edinger
>
> **备注:** Submission to Transportation Research Part C: Emerging Technologies. 36 pages, 5 figures, 4 tables
>
> **摘要:** Background: The standardisation of Intermodal Loading Units (ILUs), including containers, semi-trailers, and swap bodies, has transformed global trade, yet efficient and robust identification remains an operational bottleneck in ports and terminals. Objective: To map Computer Vision (CV) methods for ILU identification, clarify terminology, summarise the evolution of proposed approaches, and highlight research gaps, future directions and their potential effects on terminal operations. Methods: Following PRISMA-ScR, we searched Google Scholar and dblp for English-language studies with quantitative results. After dual reviewer screening, the studies were charted across methods, datasets, and evaluation metrics. Results: 63 empirical studies on CV-based solutions for the ILU identification task, published between 1990 and 2025 were reviewed. Methodological evolution of ILU identification solutions, datasets, evaluation of the proposed methods and future research directions are summarised. A shift from static (e.g. OCR-gates) to vehicle mounted camera setups, which enables precise monitoring is observed. The reported results for end-to-end accuracy range from 5% to 96%. Conclusions: We propose standardised terminology, advocate for open-access datasets, codebases and model weights to enable fair evaluation and define future work directions. The shift from static to dynamic camera settings introduces new challenges that have transformative potential for transportation and logistics. However, the lack of public benchmark datasets, open-access code, and standardised terminology hinders the advancements in this field. As for the future work, we suggest addressing the new challenges emerged from vehicle mounted cameras, exploring synthetic data generation, refining the multi-stage methods into unified end-to-end models to reduce complexity, and focusing on contextless text recognition.
>
---
#### [replaced 046] StrokeFusion: Vector Sketch Generation via Joint Stroke-UDF Encoding and Latent Sequence Diffusion
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.23752v4](https://arxiv.org/pdf/2503.23752v4)**

> **作者:** Jin Zhou; Yi Zhou; Hongliang Yang; Pengfei Xu; Hui Huang
>
> **摘要:** In the field of sketch generation, raster-format trained models often produce non-stroke artifacts, while vector-format trained models typically lack a holistic understanding of sketches, leading to compromised recognizability. Moreover, existing methods struggle to extract common features from similar elements (e.g., eyes of animals) appearing at varying positions across sketches. To address these challenges, we propose StrokeFusion, a two-stage framework for vector sketch generation. It contains a dual-modal sketch feature learning network that maps strokes into a high-quality latent space. This network decomposes sketches into normalized strokes and jointly encodes stroke sequences with Unsigned Distance Function (UDF) maps, representing sketches as sets of stroke feature vectors. Building upon this representation, our framework exploits a stroke-level latent diffusion model that simultaneously adjusts stroke position, scale, and trajectory during generation. This enables high-fidelity sketch generation while supporting stroke interpolation editing. Extensive experiments on the QuickDraw dataset demonstrate that our framework outperforms state-of-the-art techniques, validating its effectiveness in preserving structural integrity and semantic features. Code and models will be made publicly available upon publication.
>
---
#### [replaced 047] MedGEN-Bench: Contextually entangled benchmark for open-ended multimodal medical generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13135v2](https://arxiv.org/pdf/2511.13135v2)**

> **作者:** Junjie Yang; Yuhao Yan; Gang Wu; Yuxuan Wang; Ruoyu Liang; Xinjie Jiang; Xiang Wan; Fenglei Fan; Yongquan Zhang; Feiwei Qin; Changmiao Wang
>
> **备注:** CVPR 2026 Under Review
>
> **摘要:** As Vision-Language Models (VLMs) increasingly gain traction in medical applications, clinicians are progressively expecting AI systems not only to generate textual diagnoses but also to produce corresponding medical images that integrate seamlessly into authentic clinical workflows. Despite the growing interest, existing medical visual benchmarks present notable limitations. They often rely on ambiguous queries that lack sufficient relevance to image content, oversimplify complex diagnostic reasoning into closed-ended shortcuts, and adopt a text-centric evaluation paradigm that overlooks the importance of image generation capabilities. To address these challenges, we introduce MedGEN-Bench, a comprehensive multimodal benchmark designed to advance medical AI research. MedGEN-Bench comprises 6,422 expert-validated image-text pairs spanning six imaging modalities, 16 clinical tasks, and 28 subtasks. It is structured into three distinct formats: Visual Question Answering, Image Editing, and Contextual Multimodal Generation. What sets MedGEN-Bench apart is its focus on contextually intertwined instructions that necessitate sophisticated cross-modal reasoning and open-ended generative outputs, moving beyond the constraints of multiple-choice formats. To evaluate the performance of existing systems, we employ a novel three-tier assessment framework that integrates pixel-level metrics, semantic text analysis, and expert-guided clinical relevance scoring. Using this framework, we systematically assess 10 compositional frameworks, 3 unified models, and 5 VLMs.
>
---
#### [replaced 048] LoG3D: Ultra-High-Resolution 3D Shape Modeling via Local-to-Global Partitioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10040v2](https://arxiv.org/pdf/2511.10040v2)**

> **作者:** Xinran Yang; Shuichang Lai; Jiangjing Lyu; Hongjie Li; Bowen Pan; Yuanqi Li; Jie Guo; Zhengkang Zhou; Yanwen Guo
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Generating high-fidelity 3D contents remains a fundamental challenge due to the complexity of representing arbitrary topologies-such as open surfaces and intricate internal structures-while preserving geometric details. Prevailing methods based on signed distance fields (SDFs) are hampered by costly watertight preprocessing and struggle with non-manifold geometries, while point-cloud representations often suffer from sampling artifacts and surface discontinuities. To overcome these limitations, we propose a novel 3D variational autoencoder (VAE) framework built upon unsigned distance fields (UDFs)-a more robust and computationally efficient representation that naturally handles complex and incomplete shapes. Our core innovation is a local-to-global (LoG) architecture that processes the UDF by partitioning it into uniform subvolumes, termed UBlocks. This architecture couples 3D convolutions for capturing local detail with sparse transformers for enforcing global coherence. A Pad-Average strategy further ensures smooth transitions at subvolume boundaries during reconstruction. This modular design enables seamless scaling to ultra-high resolutions up to $2048^3$-a regime previously unattainable for 3D VAEs. Experiments demonstrate state-of-the-art performance in both reconstruction accuracy and generative quality, yielding superior surface smoothness and geometric flexibility.
>
---
#### [replaced 049] YOLO Meets Mixture-of-Experts: Adaptive Expert Routing for Robust Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13344v2](https://arxiv.org/pdf/2511.13344v2)**

> **作者:** Ori Meiraz; Sharon Shalev; Avishai Weizman
>
> **备注:** 1 figure, 1 table
>
> **摘要:** This paper presents a novel Mixture-of-Experts framework for object detection, incorporating adaptive routing among multiple YOLOv9-T experts to enable dynamic feature specialization and achieve higher mean Average Precision (mAP) and Average Recall (AR) compared to a single YOLOv9-T model.
>
---
#### [replaced 050] Improving Greenland Bed Topography Mapping with Uncertainty-Aware Graph Learning on Sparse Radar Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.08571v2](https://arxiv.org/pdf/2509.08571v2)**

> **作者:** Bayu Adhi Tama; Homayra Alam; Mostafa Cham; Omar Faruque; Jianwu Wang; Vandana Janeja
>
> **摘要:** Accurate maps of Greenland's subglacial bed are essential for sea-level projections, but radar observations are sparse and uneven. We introduce GraphTopoNet, a graph-learning framework that fuses heterogeneous supervision and explicitly models uncertainty via Monte Carlo dropout. Spatial graphs built from surface observables (elevation, velocity, mass balance) are augmented with gradient features and polynomial trends to capture both local variability and broad structure. To handle data gaps, we employ a hybrid loss that combines confidence-weighted radar supervision with dynamically balanced regularization. Applied to three Greenland subregions, GraphTopoNet outperforms interpolation, convolutional, and graph-based baselines, reducing error by up to 60 percent while preserving fine-scale glacial features. The resulting bed maps improve reliability for operational modeling, supporting agencies engaged in climate forecasting and policy. More broadly, GraphTopoNet shows how graph machine learning can convert sparse, uncertain geophysical observations into actionable knowledge at continental scale.
>
---
#### [replaced 051] UVLM: Benchmarking Video Language Model for Underwater World Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.02373v2](https://arxiv.org/pdf/2507.02373v2)**

> **作者:** Xizhe Xue; Yang Zhou; Dawei Yan; Lijie Tao; Junjie Li; Ying Li; Haokui Zhang; Rong Xiao
>
> **备注:** 18 pages, 10 figures, 7 tables. Accepted to the Fortieth AAAI Conference on Artificial Intelligence (AAAI-26), 2026
>
> **摘要:** Recently, the remarkable success of large language models (LLMs) has achieved a profound impact on the field of artificial intelligence. Numerous advanced works based on LLMs have been proposed and applied in various scenarios. Among them, video language models (VidLMs) are particularly widely used. However, existing works primarily focus on terrestrial scenarios, overlooking the highly demanding application needs of underwater observation. To overcome this gap, we introduce UVLM, an under water observation benchmark which is build through a collaborative approach combining human expertise and AI models. To ensure data quality, we have conducted in-depth considerations from multiple perspectives. First, to address the unique challenges of underwater environments, we selected videos that represent typical underwater challenges including light variations, water turbidity, and diverse viewing angles to construct the dataset. Second, to ensure data diversity, the dataset covers a wide range of frame rates, resolutions, 419 classes of marine animals, and various static plants and terrains. Next, for task diversity, we adopted a structured design where observation targets are categorized into two major classes: biological and environmental. Each category includes content observation and change/action observation, totaling 20 distinct task types. Finally, we designed several challenging evaluation metrics to enable quantitative comparison and analysis of different methods. Experiments on two representative VidLMs demonstrate that fine-tuning VidLMs on UVLM significantly improves underwater world understanding while also showing potential for slight improvements on existing in-air VidLM benchmarks, such as VideoMME and Perception text. The dataset and prompt engineering will be released publicly.
>
---
#### [replaced 052] Measuring Train Driver Performance as Key to Approval of Driverless Trains
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.19735v3](https://arxiv.org/pdf/2504.19735v3)**

> **作者:** Rustam Tagiew; Prasannavenkatesh Balaji
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Points 2.1.4(b), 2.4.2(b) and 2.4.3(b) in Annex I of Implementing Regulation (EU) No. 402/2013 allow a simplified approach for the safety approval of computer vision systems for driverless trains, if they have 'similar' functions and interfaces as the replaced human driver. The human driver is not replaced one-to-one by a technical system - only a limited set of cognitive functions are replaced. However, performance in the most challenging function, obstacle detection, is difficult to quantify due to the deficiency of published measurement results. This article summarizes the data published so far. This article also goes a long way to remedy this situation by providing a new public and anonymized dataset of 711 train driver performance measurements from controlled experiments. The measurements are made for different speeds, obstacle sizes, train protection systems and obstacle color contrasts respectively. The measured values are reaction time and distance to the obstacle. The goal of this paper is an unbiased and exhaustive description of the presented dataset for research, standardization and regulation. The dataset with supplementing information and literature is published on https://data.fid-move.de/de/dataset/atosensedata
>
---
#### [replaced 053] Continual Learning for Image Captioning through Improved Image-Text Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.06009v2](https://arxiv.org/pdf/2510.06009v2)**

> **作者:** Bertram Taetz; Gal Bordelius
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Generating accurate and coherent image captions in a continual learning setting remains a major challenge due to catastrophic forgetting and the difficulty of aligning evolving visual concepts with language over time. In this work, we propose a novel multi-loss framework for continual image captioning that integrates semantic guidance through prompt-based continual learning and contrastive alignment. Built upon a pretrained ViT-GPT-2 backbone, our approach combines standard cross-entropy loss with three additional components: (1) a prompt-based cosine similarity loss that aligns image embeddings with synthetically constructed prompts encoding objects, attributes, and actions; (2) a CLIP-style loss that promotes alignment between image embeddings and target caption embedding; and (3) a language-guided contrastive loss that employs a triplet loss to enhance class-level discriminability between tasks. Notably, our approach introduces no additional overhead at inference time and requires no prompts during caption generation. We find that this approach mitigates catastrophic forgetting, while achieving better semantic caption alignment compared to state-of-the-art methods. The code can be found via the following link: https://github.com/Gepardius/Taetz_Bordelius_Continual_ImageCaptioning.
>
---
#### [replaced 054] Not All Regions Are Equal: Attention-Guided Perturbation Network for Industrial Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2408.07490v3](https://arxiv.org/pdf/2408.07490v3)**

> **作者:** Tingfeng Huang; Weijia Kong; Yuxuan Cheng; Jingbo Xia; Rui Yu; Jinhai Xiang; Xinwei He
>
> **摘要:** In unsupervised image anomaly detection, reconstruction methods aim to train models to capture normal patterns comprehensively for normal data reconstruction. Yet, these models sometimes retain unintended reconstruction capacity for anomalous regions during inference, leading to missed detections. To mitigate this issue, existing works perturb normal samples in a sample-agnostic manner, uniformly adding noise across spatial locations before reconstructing the original. Despite promising results, they disregard the fact that foreground locations are inherently more critical for robust reconstruction. Motivated by this, we present a novel reconstruction framework named Attention-Guided Perturbation Network (AGPNet) for industrial anomaly detection. Its core idea is to add perturbations guided by a sample-aware attention mask to improve the learning of invariant normal patterns at important locations. AGPNet consists of two branches, \ie, a reconstruction branch and an auxiliary attention-based perturbation one. The reconstruction branch learns to reconstruct normal samples, while the auxiliary one aims to produce attention masks to guide the noise perturbation process for normal samples. By perturbing more aggressively at those important regions, we encourage the reconstruction branch to learn inherent normal patterns both comprehensively and robustly. Extensive experiments are conducted on several popular benchmarks covering MVTec-AD, VisA, and MVTec-3D, and show that AGPNet consistently obtains leading anomaly detection performance across a variety of setups, including few-shot, one-class, and multi-class ones.
>
---
#### [replaced 055] Manifold Learning for Hyperspectral Images
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.15016v3](https://arxiv.org/pdf/2503.15016v3)**

> **作者:** Fethi Harkat; Guillaume Gey; Valérie Perrier; Kévin Polisano; Tiphaine Deuberet
>
> **摘要:** Traditional feature extraction and projection techniques, such as Principal Component Analysis, struggle to adequately represent X-Ray Transmission (XRT) Multi-Energy (ME) images, limiting the performance of neural networks in decision-making processes. To address this issue, we propose a method that approximates the dataset topology by constructing adjacency graphs using the Uniform Manifold Approximation and Projection. This approach captures nonlinear correlations within the data, significantly improving the performance of machine learning algorithms, particularly in processing Hyperspectral Images (HSI) from X-ray transmission spectroscopy. This technique not only preserves the global structure of the data but also enhances feature separability, leading to more accurate and robust classification results.
>
---
#### [replaced 056] Rethinking Token-wise Feature Caching: Accelerating Diffusion Transformers with Dual Feature Caching
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.18911v2](https://arxiv.org/pdf/2412.18911v2)**

> **作者:** Chang Zou; Evelyn Zhang; Runlin Guo; Haohang Xu; Conghui He; Xuming Hu; Linfeng Zhang
>
> **摘要:** Diffusion Transformers (DiT) have become the dominant methods in image and video generation yet still suffer substantial computational costs. As an effective approach for DiT acceleration, feature caching methods are designed to cache the features of DiT in previous timesteps and reuse them in the next timesteps, allowing us to skip the computation in the next timesteps. Among them, token-wise feature caching has been introduced to perform different caching ratios for different tokens in DiTs, aiming to skip the computation for unimportant tokens while still computing the important ones. In this paper, we propose to carefully check the effectiveness in token-wise feature caching with the following two questions: (1) Is it really necessary to compute the so-called "important" tokens in each step? (2) Are so-called important tokens really important? Surprisingly, this paper gives some counter-intuition answers, demonstrating that consistently computing the selected ``important tokens'' in all steps is not necessary. The selection of the so-called ``important tokens'' is often ineffective, and even sometimes shows inferior performance than random selection. Based on these observations, this paper introduces dual feature caching referred to as DuCa, which performs aggressive caching strategy and conservative caching strategy iteratively and selects the tokens for computing randomly. Extensive experimental results demonstrate the effectiveness of our method in DiT, PixArt, FLUX, and OpenSora, demonstrating significant improvements than the previous token-wise feature caching.
>
---
#### [replaced 057] Deploying Rapid Damage Assessments from sUAS Imagery for Disaster Response
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [https://arxiv.org/pdf/2511.03132v2](https://arxiv.org/pdf/2511.03132v2)**

> **作者:** Thomas Manzini; Priyankari Perali; Robin R. Murphy
>
> **备注:** 6 pages, 4 figures, 1 table. Accepted - In Press, IAAI'26
>
> **摘要:** This paper presents the first AI/ML system for automating building damage assessment in uncrewed aerial systems (sUAS) imagery to be deployed operationally during federally declared disasters (Hurricanes Debby and Helene). In response to major disasters, sUAS teams are dispatched to collect imagery of the affected areas to assess damage; however, at recent disasters, teams collectively delivered between 47GB and 369GB of imagery per day, representing more imagery than can reasonably be transmitted or interpreted by subject matter experts in the disaster scene, thus delaying response efforts. To alleviate this data avalanche encountered in practice, computer vision and machine learning techniques are necessary. While prior work has been deployed to automatically assess damage in satellite imagery, there is no current state of practice for sUAS-based damage assessment systems, as all known work has been confined to academic settings. This work establishes the state of practice via the development and deployment of models for building damage assessment with sUAS imagery. The model development involved training on the largest known dataset of post-disaster sUAS aerial imagery, containing 21,716 building damage labels, and the operational training of 91 disaster practitioners. The best performing model was deployed during the responses to Hurricanes Debby and Helene, where it assessed a combined 415 buildings in approximately 18 minutes. This work contributes documentation of the actual use of AI/ML for damage assessment during a disaster and lessons learned to the benefit of the AI/ML research and user communities.
>
---
#### [replaced 058] Towards Prospective Medical Image Reconstruction via Knowledge-Informed Dynamic Optimal Transport
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.17644v3](https://arxiv.org/pdf/2505.17644v3)**

> **作者:** Taoran Zheng; Yan Yang; Xing Li; Xiang Gu; Jian Sun; Zongben Xu
>
> **摘要:** Medical image reconstruction from measurement data is a vital but challenging inverse problem. Deep learning approaches have achieved promising results, but often requires paired measurement and high-quality images, which is typically simulated through a forward model, i.e., retrospective reconstruction. However, training on simulated pairs commonly leads to performance degradation on real prospective data due to the retrospective-to-prospective gap caused by incomplete imaging knowledge in simulation. To address this challenge, this paper introduces imaging Knowledge-Informed Dynamic Optimal Transport (KIDOT), a novel dynamic optimal transport framework with optimality in the sense of preserving consistency with imaging physics in transport, that conceptualizes reconstruction as finding a dynamic transport path. KIDOT learns from unpaired data by modeling reconstruction as a continuous evolution path from measurements to images, guided by an imaging knowledge-informed cost function and transport equation. This dynamic and knowledge-aware approach enhances robustness and better leverages unpaired data while respecting acquisition physics. Theoretically, we demonstrate that KIDOT naturally generalizes dynamic optimal transport, ensuring its mathematical rationale and solution existence. Extensive experiments on MRI and CT reconstruction demonstrate KIDOT's superior performance.
>
---
#### [replaced 059] DocSLM: A Small Vision-Language Model for Long Multimodal Document Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11313v2](https://arxiv.org/pdf/2511.11313v2)**

> **作者:** Tanveer Hannan; Dimitrios Mallios; Parth Pathak; Faegheh Sardari; Thomas Seidl; Gedas Bertasius; Mohsen Fayyaz; Sunando Sengupta
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated strong multimodal reasoning capabilities on long and complex documents. However, their high memory footprint makes them impractical for deployment on resource-constrained edge devices. We present DocSLM, an efficient Small Vision-Language Model designed for long-document understanding under constrained memory resources. DocSLM incorporates a Hierarchical Multimodal Compressor that jointly encodes visual, textual, and layout information from each page into a fixed-length sequence, greatly reducing memory consumption while preserving both local and global semantics. To enable scalable processing over arbitrarily long inputs, we introduce a Streaming Abstention mechanism that operates on document segments sequentially and filters low-confidence responses using an entropy-based uncertainty calibrator. Across multiple long multimodal document benchmarks, DocSLM matches or surpasses state-of-the-art methods while using 82\% fewer visual tokens, 75\% fewer parameters, and 71\% lower latency, delivering reliable multimodal document understanding on lightweight edge devices. Code is available in the supplementary material.
>
---
#### [replaced 060] Well-Conditioned Polynomial Representations for Mathematical Handwriting Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.10815v3](https://arxiv.org/pdf/2509.10815v3)**

> **作者:** Robert M. Corless; Deepak Singh Kalhan; Stephen M. Watt
>
> **摘要:** Previous work has made use of a parameterized plane curve polynomial representation for mathematical handwriting, with the polynomials represented in a Legendre or Legendre-Sobolev graded basis. This provides a compact geometric representation for the digital ink. Preliminary results have also been shown for Chebyshev and Chebyshev-Sobolev bases. This article explores the trade-offs between basis choice and polynomial degree to achieve accurate modeling with a low computational cost. To do this, we consider the condition number for polynomial evaluation in these bases and bound how the various inner products give norms for the variations between symbols.
>
---
#### [replaced 061] Decoupling Scene Perception and Ego Status: A Multi-Context Fusion Approach for Enhanced Generalization in End-to-End Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13079v2](https://arxiv.org/pdf/2511.13079v2)**

> **作者:** Jiacheng Tang; Mingyue Feng; Jiachao Liu; Yaonong Wang; Jian Pu
>
> **备注:** Accepted to AAAI 2026 (Oral)
>
> **摘要:** Modular design of planning-oriented autonomous driving has markedly advanced end-to-end systems. However, existing architectures remain constrained by an over-reliance on ego status, hindering generalization and robust scene understanding. We identify the root cause as an inherent design within these architectures that allows ego status to be easily leveraged as a shortcut. Specifically, the premature fusion of ego status in the upstream BEV encoder allows an information flow from this strong prior to dominate the downstream planning module. To address this challenge, we propose AdaptiveAD, an architectural-level solution based on a multi-context fusion strategy. Its core is a dual-branch structure that explicitly decouples scene perception and ego status. One branch performs scene-driven reasoning based on multi-task learning, but with ego status deliberately omitted from the BEV encoder, while the other conducts ego-driven reasoning based solely on the planning task. A scene-aware fusion module then adaptively integrates the complementary decisions from the two branches to form the final planning trajectory. To ensure this decoupling does not compromise multi-task learning, we introduce a path attention mechanism for ego-BEV interaction and add two targeted auxiliary tasks: BEV unidirectional distillation and autoregressive online mapping. Extensive evaluations on the nuScenes dataset demonstrate that AdaptiveAD achieves state-of-the-art open-loop planning performance. Crucially, it significantly mitigates the over-reliance on ego status and exhibits impressive generalization capabilities across diverse scenarios.
>
---
#### [replaced 062] UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.12988v2](https://arxiv.org/pdf/2511.12988v2)**

> **作者:** Furui Xu; Shaobo Wang; Jiajun Zhang; Chenghao Sun; Haixiang Tang; Linfeng Zhang
>
> **备注:** AAAI 2026, 13 pages, 9 figures, 5 tables
>
> **摘要:** The growing scale of datasets in deep learning has introduced significant computational challenges. Dataset pruning addresses this challenge by constructing a compact but informative coreset from the full dataset with comparable performance. Previous approaches typically establish scoring metrics based on specific criteria to identify representative samples. However, these methods predominantly rely on sample scores obtained from the model's performance during the training (i.e., fitting) phase. As scoring models achieve near-optimal performance on training data, such fitting-centric approaches induce a dense distribution of sample scores within a narrow numerical range. This concentration reduces the distinction between samples and hinders effective selection. To address this challenge, we conduct dataset pruning from the perspective of generalization, i.e., scoring samples based on models not exposed to them during training. We propose a plug-and-play framework, UNSEEN, which can be integrated into existing dataset pruning methods. Additionally, conventional score-based methods are single-step and rely on models trained solely on the complete dataset, providing limited perspective on the importance of samples. To address this limitation, we scale UNSEEN to multi-step scenarios and propose an incremental selection technique through scoring models trained on varying coresets, and optimize the quality of the coreset dynamically. Extensive experiments demonstrate that our method significantly outperforms existing state-of-the-art (SOTA) methods on CIFAR-10, CIFAR-100, and ImageNet-1K. Notably, on ImageNet-1K, UNSEEN achieves lossless performance while reducing training data by 30\%.
>
---
#### [replaced 063] Generalized Denoising Diffusion Codebook Models (gDDCM): Tokenizing images using a pre-trained diffusion model
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.13387v2](https://arxiv.org/pdf/2511.13387v2)**

> **作者:** Fei Kong
>
> **备注:** in Chinese language
>
> **摘要:** Recently, the Denoising Diffusion Codebook Models (DDCM) was proposed. DDCM leverages the Denoising Diffusion Probabilistic Model (DDPM) and replaces the random noise in the backward process with noise sampled from specific sets according to a predefined rule, thereby enabling image compression. However, DDCM cannot be applied to methods other than DDPM. In this paper, we propose the generalized Denoising Diffusion Compression Model (gDDCM), which extends DDCM to mainstream diffusion models and their variants, including DDPM, Score-Based Models, Consistency Models, and Rectified Flow. We evaluate our method on CIFAR-10 and LSUN Bedroom datasets. Experimental results demonstrate that our approach successfully generalizes DDCM to the aforementioned models and achieves improved performance.
>
---
#### [replaced 064] ArtiWorld: LLM-Driven Articulation of 3D Objects in Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12977v2](https://arxiv.org/pdf/2511.12977v2)**

> **作者:** Yixuan Yang; Luyang Xie; Zhen Luo; Zixiang Zhao; Tongsheng Ding; Mingqi Gao; Feng Zheng
>
> **摘要:** Building interactive simulators and scalable robot-learning environments requires a large number of articulated assets. However, most existing 3D assets in simulation are rigid, and manually converting them into articulated objects is extremely labor- and cost-intensive. This raises a natural question: can we automatically identify articulable objects in a scene and convert them into articulated assets directly? In this paper, we present ArtiWorld, a scene-aware pipeline that localizes candidate articulable objects from textual scene descriptions and reconstructs executable URDF models that preserve the original geometry. At the core of this pipeline is Arti4URDF, which leverages 3D point cloud, prior knowledge of a large language model (LLM), and a URDF-oriented prompt design to rapidly convert rigid objects into interactive URDF-based articulated objects while maintaining their 3D shape. We evaluate ArtiWorld at three levels: 3D simulated objects, full 3D simulated scenes, and real-world scan scenes. Across all three settings, our method consistently outperforms existing approaches and achieves state-of-the-art performance, while preserving object geometry and correctly capturing object interactivity to produce usable URDF-based articulated models. This provides a practical path toward building interactive, robot-ready simulation environments directly from existing 3D assets. Code and data will be released.
>
---
#### [replaced 065] Towards Reliable Human Evaluations in Gesture Generation: Insights from a Community-Driven State-of-the-Art Benchmark
- **分类: cs.CV; cs.GR; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.01233v2](https://arxiv.org/pdf/2511.01233v2)**

> **作者:** Rajmund Nagy; Hendric Voss; Thanh Hoang-Minh; Mihail Tsakov; Teodor Nikolov; Zeyi Zhang; Tenglong Ao; Sicheng Yang; Shaoli Huang; Yongkang Cheng; M. Hamza Mughal; Rishabh Dabral; Kiran Chhatre; Christian Theobalt; Libin Liu; Stefan Kopp; Rachel McDonnell; Michael Neff; Taras Kucherenko; Youngwoo Yoon; Gustav Eje Henter
>
> **备注:** 23 pages, 10 figures. The last two authors made equal contributions
>
> **摘要:** We review human evaluation practices in automated, speech-driven 3D gesture generation and find a lack of standardisation and frequent use of flawed experimental setups. This leads to a situation where it is impossible to know how different methods compare, or what the state of the art is. In order to address common shortcomings of evaluation design, and to standardise future user studies in gesture-generation works, we introduce a detailed human evaluation protocol for the widely-used BEAT2 motion-capture dataset. Using this protocol, we conduct large-scale crowdsourced evaluation to rank six recent gesture-generation models -- each trained by its original authors -- across two key evaluation dimensions: motion realism and speech-gesture alignment. Our results provide strong evidence that 1) newer models do not consistently outperform earlier approaches; 2) published claims of high motion realism or speech-gesture alignment may not hold up under rigorous evaluation; and 3) the field must adopt disentangled assessments of motion quality and multimodal alignment for accurate benchmarking in order to make progress. Finally, in order to drive standardisation and enable new evaluation research, we will release five hours of synthetic motion from the benchmarked models; over 750 rendered video stimuli from the user studies -- enabling new evaluations without model reimplementation required -- alongside our open-source rendering script, and the 16,000 pairwise human preference votes collected for our benchmark.
>
---
#### [replaced 066] Seeing and Knowing in the Wild: Open-domain Visual Entity Recognition with Large-scale Knowledge Graphs via Contrastive Learning
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.13675v2](https://arxiv.org/pdf/2510.13675v2)**

> **作者:** Hongkuan Zhou; Lavdim Halilaj; Sebastian Monka; Stefan Schmid; Yuqicheng Zhu; Jingcheng Wu; Nadeem Nazer; Steffen Staab
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Open-domain visual entity recognition aims to identify and link entities depicted in images to a vast and evolving set of real-world concepts, such as those found in Wikidata. Unlike conventional classification tasks with fixed label sets, it operates under open-set conditions, where most target entities are unseen during training and exhibit long-tail distributions. This makes the task inherently challenging due to limited supervision, high visual ambiguity, and the need for semantic disambiguation. We propose a Knowledge-guided Contrastive Learning (KnowCoL) framework that combines both images and text descriptions into a shared semantic space grounded by structured information from Wikidata. By abstracting visual and textual inputs to a conceptual level, the model leverages entity descriptions, type hierarchies, and relational context to support zero-shot entity recognition. We evaluate our approach on the OVEN benchmark, a large-scale open-domain visual recognition dataset with Wikidata IDs as the label space. Our experiments show that using visual, textual, and structured knowledge greatly improves accuracy, especially for rare and unseen entities. Our smallest model improves the accuracy on unseen entities by 10.5% compared to the state-of-the-art, despite being 35 times smaller.
>
---
#### [replaced 067] PALM: A Dataset and Baseline for Learning Multi-subject Hand Prior
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05403v2](https://arxiv.org/pdf/2511.05403v2)**

> **作者:** Zicong Fan; Edoardo Remelli; David Dimond; Fadime Sener; Liuhao Ge; Bugra Tekin; Cem Keskin; Shreyas Hampali
>
> **摘要:** The ability to grasp objects, signal with gestures, and share emotion through touch all stem from the unique capabilities of human hands. Yet creating high-quality personalized hand avatars from images remains challenging due to complex geometry, appearance, and articulation, particularly under unconstrained lighting and limited views. Progress has also been limited by the lack of datasets that jointly provide accurate 3D geometry, high-resolution multiview imagery, and a diverse population of subjects. To address this, we present PALM, a large-scale dataset comprising 13k high-quality hand scans from 263 subjects and 90k multi-view images, capturing rich variation in skin tone, age, and geometry. To show its utility, we present a baseline PALM-Net, a multi-subject prior over hand geometry and material properties learned via physically based inverse rendering, enabling realistic, relightable single-image hand avatar personalization. PALM's scale and diversity make it a valuable real-world resource for hand modeling and related research.
>
---
#### [replaced 068] MOON: Generative MLLM-based Multimodal Representation Learning for E-commerce Product Understanding
- **分类: cs.CV; cs.AI; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.11999v2](https://arxiv.org/pdf/2508.11999v2)**

> **作者:** Daoze Zhang; Zhanheng Nie; Jianyu Liu; Chenghan Fu; Wanxian Guan; Yuan Gao; Jun Song; Pengjie Wang; Jian Xu; Bo Zheng
>
> **备注:** Accepted by WSDM 2026. 11 pages, 9 figures
>
> **摘要:** With the rapid advancement of e-commerce, exploring general representations rather than task-specific ones has attracted increasing research attention. For product understanding, although existing discriminative dual-flow architectures drive progress in this field, they inherently struggle to model the many-to-one alignment between multiple images and texts of products. Therefore, we argue that generative Multimodal Large Language Models (MLLMs) hold significant potential for improving product representation learning. Nevertheless, achieving this goal still remains non-trivial due to several key challenges: the lack of multimodal and aspect-aware modeling modules in typical LLMs; the common presence of background noise in product images; and the absence of a standard benchmark for evaluation. To address these issues, we propose the first generative MLLM-based model named MOON for product representation learning. Our method (1) employs a guided Mixture-of-Experts (MoE) module for targeted modeling of multimodal and aspect-specific product content; (2) effectively detects core semantic regions in product images to mitigate the distraction and interference caused by background noise; and (3) introduces the specialized negative sampling strategy to increase the difficulty and diversity of negative samples. In addition, we release a large-scale multimodal benchmark MBE for various product understanding tasks. Experimentally, our model demonstrates competitive zero-shot performance on both our benchmark and the public dataset, showcasing strong generalization across various downstream tasks, including cross-modal retrieval, product classification, and attribute prediction. Furthermore, the case study and visualization illustrate the effectiveness of MOON for product understanding.
>
---
#### [replaced 069] Squeezed Diffusion Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.14871v2](https://arxiv.org/pdf/2508.14871v2)**

> **作者:** Jyotirmai Singh; Samar Khanna; James Burgess
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Diffusion models typically inject isotropic Gaussian noise, disregarding structure in the data. Motivated by the way quantum squeezed states redistribute uncertainty according to the Heisenberg uncertainty principle, we introduce Squeezed Diffusion Models (SDM), which scale noise anisotropically along the principal component of the training distribution. As squeezing enhances the signal-to-noise ratio in physics, we hypothesize that scaling noise in a data-dependent manner can better assist diffusion models in learning important data features. We study two configurations: (i) a Heisenberg diffusion model that compensates the scaling on the principal axis with inverse scaling on orthogonal directions and (ii) a standard SDM variant that scales only the principal axis. Counterintuitively, on CIFAR-10/100 and CelebA-64, mild antisqueezing - i.e. increasing variance on the principal axis - consistently improves FID by up to 15% and shifts the precision-recall frontier toward higher recall. Our results demonstrate that simple, data-aware noise shaping can deliver robust generative gains without architectural changes.
>
---
#### [replaced 070] Availability-aware Sensor Fusion via Unified Canonical Space
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.07029v2](https://arxiv.org/pdf/2503.07029v2)**

> **作者:** Dong-Hee Paek; Seung-Hyun Kong
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Sensor fusion of camera, LiDAR, and 4-dimensional (4D) Radar has brought a significant performance improvement in autonomous driving. However, there still exist fundamental challenges: deeply coupled fusion methods assume continuous sensor availability, making them vulnerable to sensor degradation and failure, whereas sensor-wise cross-attention fusion methods struggle with computational cost and unified feature representation. This paper presents availability-aware sensor fusion (ASF), a novel method that employs unified canonical projection (UCP) to enable consistency in all sensor features for fusion and cross-attention across sensors along patches (CASAP) to enhance robustness of sensor fusion against sensor degradation and failure. As a result, the proposed ASF shows a superior object detection performance to the existing state-of-the-art fusion methods under various weather and sensor degradation (or failure) conditions. Extensive experiments on the K-Radar dataset demonstrate that ASF achieves improvements of 9.7% in AP BEV (87.2%) and 20.1% in AP 3D (73.6%) in object detection at IoU=0.5, while requiring a low computational cost. All codes are available at https://github.com/kaist-avelab/k-radar.
>
---
#### [replaced 071] SemCo: Toward Semantic Coherent Visual Relationship Forecasting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2107.01181v2](https://arxiv.org/pdf/2107.01181v2)**

> **作者:** Yangjun Ou; Yao Liu; Li Mi; Zhenzhong Chen
>
> **摘要:** Visual Relationship Forecasting (VRF) aims to anticipate relations among objects without observing future visual content. The task relies on capturing and modeling the semantic coherence in object interactions, as it underpins the evolution of events and scenes in videos. However, existing VRF datasets offer limited support for learning such coherence due to noisy annotations in the datasets and weak correlations between different actions and relationship transitions in subject-object pair. Furthermore, existing methods struggle to distinguish similar relationships and overfit to unchanging relationships in consecutive frames. To address these challenges, we present SemCoBench, a benchmark that emphasizes semantic coherence for visual relationship forecasting. Based on action labels and short-term subject-object pairs, SemCoBench decomposes relationship categories and dynamics by cleaning and reorganizing video datasets to ensure predicting semantic coherence in object interactions. In addition, we also present Semantic Coherent Transformer method (SemCoFormer) to model the semantic coherence with a Relationship Augmented Module (RAM) and a Coherence Reasoning Module (CRM). RAM is designed to distinguish similar relationships, and CRM facilitates the model's focus on the dynamics in relationships. The experimental results on SemCoBench demonstrate that modeling the semantic coherence is a key step toward reasonable, fine-grained, and diverse visual relationship forecasting, contributing to a more comprehensive understanding of video scenes.
>
---
#### [replaced 072] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12409v2](https://arxiv.org/pdf/2506.12409v2)**

> **作者:** Ziwei Liu; Borui Kang; Wei Li; Hangjie Yuan; Yanbing Yang; Wenbin Li; Jun Luo; Yifan Zhu; Tao Feng
>
> **摘要:** Vision-Language Continual Learning (VLCL) has attracted significant research attention for its robust capabilities, and the adoption of Parameter-Efficient Fine-Tuning (PEFT) strategies is enabling these models to achieve competitive performance with substantially reduced resource consumption. However, dominated First-Order (FO) optimization is prone to trap models in suboptimal local minima, especially in limited exploration subspace within PEFT. To overcome this challenge, this paper pioneers a systematic exploration of adopting Zeroth-Order (ZO) optimization for PEFT-based VLCL. We first identify the incompatibility of naive full-ZO adoption in VLCL due to optimization process instability. We then investigate the application of ZO optimization from a modality branch-wise to a fine-grained layer-wise across various training units to identify an optimal strategy. Besides, a key theoretical insight reveals that vision modality exhibit higher variance than language counterparts in VLCL during the ZO optimization process, and we propose a modality-aware ZO strategy, which adopts gradient sign normalization in ZO and constrains vision modality perturbation to further improve performance. Benefiting from the adoption of ZO optimization, PEFT-based VLCL fulfills better ability to escape local minima during the optimization process, extensive experiments on four benchmarks demonstrate that our method achieves state-of-the-art results.
>
---
#### [replaced 073] Rasterized Steered Mixture of Experts for Efficient 2D Image Regression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.05814v2](https://arxiv.org/pdf/2510.05814v2)**

> **作者:** Yi-Hsin Li; Mårten Sjöström; Sebastian Knorr; Thomas Sikora
>
> **摘要:** The Steered Mixture of Experts regression framework has demonstrated strong performance in image reconstruction, compression, denoising, and super-resolution. However, its high computational cost limits practical applications. This work introduces a rasterization-based optimization strategy that combines the efficiency of rasterized Gaussian kernel rendering with the edge-aware gating mechanism of the Steered Mixture of Experts. The proposed method is designed to accelerate two-dimensional image regression while maintaining the model's inherent sparsity and reconstruction quality. By replacing global iterative optimization with a rasterized formulation, the method achieves significantly faster parameter updates and more memory-efficient model representations. In addition, the proposed framework supports applications such as native super-resolution and image denoising, which are not directly achievable with standard rasterized Gaussian kernel approaches. The combination of fast rasterized optimization with the edge-aware structure of the Steered Mixture of Experts provides a new balance between computational efficiency and reconstruction fidelity for two-dimensional image processing tasks.
>
---
#### [replaced 074] Subjective and Objective Quality Evaluation of Super-Resolution Enhanced Broadcast Images on a Novel SR-IQA Dataset
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2409.17451v3](https://arxiv.org/pdf/2409.17451v3)**

> **作者:** Yongrok Kim; Junha Shin; Juhyun Lee; Hyunsuk Ko
>
> **备注:** Accepted for publication in IEEE Access
>
> **摘要:** To display low-quality broadcast content on high-resolution screens in full-screen format, the application of Super-Resolution (SR), a key consumer technology, is essential. Recently, SR methods have been developed that not only increase resolution while preserving the original image information but also enhance the perceived quality. However, evaluating the quality of SR images generated from low-quality sources, such as SR-enhanced broadcast content, is challenging due to the need to consider both distortions and improvements. Additionally, assessing SR image quality without original high-quality sources presents another significant challenge. Unfortunately, there has been a dearth of research specifically addressing the Image Quality Assessment (IQA) of SR images under these conditions. In this work, we introduce a new IQA dataset for SR broadcast images in both 2K and 4K resolutions. We conducted a subjective quality evaluation to obtain the Mean Opinion Score (MOS) for these SR images and performed a comprehensive human study to identify the key factors influencing the perceived quality. Finally, we evaluated the performance of existing IQA metrics on our dataset. This study reveals the limitations of current metrics, highlighting the need for a more robust IQA metric that better correlates with the perceived quality of SR images.
>
---
#### [replaced 075] MMaDA-Parallel: Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09611v3](https://arxiv.org/pdf/2511.09611v3)**

> **作者:** Ye Tian; Ling Yang; Jiongfan Yang; Anran Wang; Yu Tian; Jiani Zheng; Haochen Wang; Zhiyang Teng; Zhuochen Wang; Yinjie Wang; Yunhai Tong; Mengdi Wang; Xiangtai Li
>
> **备注:** Project Page: https://tyfeld.github.io/mmadaparellel.github.io/
>
> **摘要:** While thinking-aware generation aims to improve performance on complex tasks, we identify a critical failure mode where existing sequential, autoregressive approaches can paradoxically degrade performance due to error propagation. To systematically analyze this issue, we propose ParaBench, a new benchmark designed to evaluate both text and image output modalities. Our analysis using ParaBench reveals that this performance degradation is strongly correlated with poor alignment between the generated reasoning and the final image. To resolve this, we propose a parallel multimodal diffusion framework, MMaDA-Parallel, that enables continuous, bidirectional interaction between text and images throughout the entire denoising trajectory. MMaDA-Parallel is trained with supervised finetuning and then further optimized by Parallel Reinforcement Learning (ParaRL), a novel strategy that applies semantic rewards along the trajectory to enforce cross-modal consistency. Experiments validate that our model significantly improves cross-modal alignment and semantic consistency, achieving a 6.9\% improvement in Output Alignment on ParaBench compared to the state-of-the-art model, Bagel, establishing a more robust paradigm for thinking-aware image synthesis. Our code is open-sourced at https://github.com/tyfeld/MMaDA-Parallel
>
---
#### [replaced 076] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13031v2](https://arxiv.org/pdf/2511.13031v2)**

> **作者:** Weihua Wang; Yubo Cui; Xiangru Lin; Zhiheng Li; Zheng Fang
>
> **备注:** Accepted to AAAI-2026
>
> **摘要:** Vision-based 3D Semantic Scene Completion (SSC) has received growing attention due to its potential in autonomous driving. While most existing approaches follow an ego-centric paradigm by aggregating and diffusing features over the entire scene, they often overlook fine-grained object-level details, leading to semantic and geometric ambiguities, especially in complex environments. To address this limitation, we propose Ocean, an object-centric prediction framework that decomposes the scene into individual object instances to enable more accurate semantic occupancy prediction. Specifically, we first employ a lightweight segmentation model, MobileSAM, to extract instance masks from the input image. Then, we introduce a 3D Semantic Group Attention module that leverages linear attention to aggregate object-centric features in 3D space. To handle segmentation errors and missing instances, we further design a Global Similarity-Guided Attention module that leverages segmentation features for global interaction. Finally, we propose an Instance-aware Local Diffusion module that improves instance features through a generative process and subsequently refines the scene representation in the BEV space. Extensive experiments on the SemanticKITTI and SSCBench-KITTI360 benchmarks demonstrate that Ocean achieves state-of-the-art performance, with mIoU scores of 17.40 and 20.28, respectively.
>
---
#### [replaced 077] Iterative Explainability for Weakly Supervised Segmentation in Medical PE Detection
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2412.07384v2](https://arxiv.org/pdf/2412.07384v2)**

> **作者:** Florin Condrea; Saikiran Rapaka; Marius Leordeanu
>
> **备注:** Paper accepted at MICAD2025 Previous title: "Label up: Learning pulmonary embolism segmentation from image level annotation through model explainability"
>
> **摘要:** Pulmonary Embolism (PE) are a leading cause of cardiovascular death. Computed tomographic pulmonary angiography (CTPA) is the gold standard for PE diagnosis, with growing interest in AI-based diagnostic assistance. However, these algorithms are limited by scarce fine-grained annotations of thromboembolic burden. We address this challenge with iExplain, a weakly supervised learning algorithm that transforms coarse image-level annotations into detailed pixel-level PE masks through iterative model explainability. Our approach generates soft segmentation maps used to mask detected regions, enabling the process to repeat and discover additional embolisms that would be missed in a single pass. This iterative refinement effectively captures complete PE regions and detects multiple distinct embolisms. Models trained on these automatically generated annotations achieve excellent PE detection performance, with significant improvements at each iteration. We demonstrate iExplain's effectiveness on the RSPECT augmented dataset, achieving results comparable to strongly supervised methods while outperforming existing weakly supervised methods.
>
---
#### [replaced 078] Deep Learning and Machine Learning -- Object Detection and Semantic Segmentation: From Theory to Applications
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2410.15584v3](https://arxiv.org/pdf/2410.15584v3)**

> **作者:** Jintao Ren; Ziqian Bi; Qian Niu; Xinyuan Song; Zekun Jiang; Junyu Liu; Benji Peng; Sen Zhang; Xuanhe Pan; Jinlang Wang; Keyu Chen; Caitlyn Heqi Yin; Pohsun Feng; Yizhu Wen; Tianyang Wang; Silin Chen; Ming Li; Jiawei Xu; Ming Liu
>
> **备注:** 167 pages
>
> **摘要:** An in-depth exploration of object detection and semantic segmentation is provided, combining theoretical foundations with practical applications. State-of-the-art advancements in machine learning and deep learning are reviewed, focusing on convolutional neural networks (CNNs), YOLO architectures, and transformer-based approaches such as DETR. The integration of artificial intelligence (AI) techniques and large language models for enhancing object detection in complex environments is examined. Additionally, a comprehensive analysis of big data processing is presented, with emphasis on model optimization and performance evaluation metrics. By bridging the gap between traditional methods and modern deep learning frameworks, valuable insights are offered for researchers, data scientists, and engineers aiming to apply AI-driven methodologies to large-scale object detection tasks.
>
---
#### [replaced 079] Clone Deterministic 3D Worlds
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.26782v2](https://arxiv.org/pdf/2510.26782v2)**

> **作者:** Zaishuo Xia; Yukuan Lu; Xinyi Li; Yifan Xu; Yubei Chen
>
> **摘要:** A world model is an internal model that simulates how the world evolves. Given past observations and actions, it predicts the future physical state of both the embodied agent and its environment. Accurate world models are essential for enabling agents to think, plan, and reason effectively in complex, dynamic settings. However, existing world models often focus on random generation of open worlds, but neglect the need for high-fidelity modeling of deterministic scenarios (such as fixed-map mazes and static space robot navigation). In this work, we take a step toward building a truly accurate world model by addressing a fundamental yet open problem: constructing a model that can fully clone a deterministic 3D world. 1) Through diagnostic experiment, we quantitatively demonstrate that high-fidelity cloning is feasible and the primary bottleneck for long-horizon fidelity is the geometric structure of the latent representation, not the dynamics model itself. 2) Building on this insight, we show that applying temporal contrastive learning principle as a geometric regularization can effectively curate a latent space that better reflects the underlying physical state manifold, demonstrating that contrastive constraints can serve as a powerful inductive bias for stable world modeling; we call this approach Geometrically-Regularized World Models (GRWM). At its core is a lightweight geometric regularization module that can be seamlessly integrated into standard autoencoders, reshaping their latent space to provide a stable foundation for effective dynamics modeling. By focusing on representation quality, GRWM offers a simple yet powerful pipeline for improving world model fidelity.
>
---
#### [replaced 080] Towards Sharper Object Boundaries in Self-Supervised Depth Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2509.15987v2](https://arxiv.org/pdf/2509.15987v2)**

> **作者:** Aurélien Cecille; Stefan Duffner; Franck Davoine; Rémi Agier; Thibault Neveu
>
> **备注:** BMVC 2025 Oral, 10 pages, 6 figures
>
> **摘要:** Accurate monocular depth estimation is crucial for 3D scene understanding, but existing methods often blur depth at object boundaries, introducing spurious intermediate 3D points. While achieving sharp edges usually requires very fine-grained supervision, our method produces crisp depth discontinuities using only self-supervision. Specifically, we model per-pixel depth as a mixture distribution, capturing multiple plausible depths and shifting uncertainty from direct regression to the mixture weights. This formulation integrates seamlessly into existing pipelines via variance-aware loss functions and uncertainty propagation. Extensive evaluations on KITTI and VKITTIv2 show that our method achieves up to 35% higher boundary sharpness and improves point cloud quality compared to state-of-the-art baselines.
>
---
#### [replaced 081] Is Noise Conditioning Necessary for Denoising Generative Models?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.13129v2](https://arxiv.org/pdf/2502.13129v2)**

> **作者:** Qiao Sun; Zhicheng Jiang; Hanhong Zhao; Kaiming He
>
> **备注:** Update ImageNet experiments (SiT with CFG). Update Appendix
>
> **摘要:** It is widely believed that noise conditioning is indispensable for denoising diffusion models to work successfully. This work challenges this belief. Motivated by research on blind image denoising, we investigate a variety of denoising-based generative models in the absence of noise conditioning. To our surprise, most models exhibit graceful degradation, and in some cases, they even perform better without noise conditioning. We provide a theoretical analysis of the error caused by removing noise conditioning and demonstrate that our analysis aligns with empirical observations. We further introduce a noise-unconditional model that achieves a competitive FID of 2.23 on CIFAR-10, significantly narrowing the gap to leading noise-conditional models. We hope our findings will inspire the community to revisit the foundations and formulations of denoising generative models.
>
---
#### [replaced 082] FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.08945v2](https://arxiv.org/pdf/2511.08945v2)**

> **作者:** Haowei Zhang; Yuanpei Zhao; Ji-Zhe Zhou; Mao Li
>
> **备注:** 12 pages, AAAI-26
>
> **摘要:** Improving the diversity of generated results while maintaining high visual quality remains a significant challenge in image generation tasks. Fractal Generative Models (FGMs) are efficient in generating high-quality images, but their inherent self-similarity limits the diversity of output images. To address this issue, we propose a novel approach based on the Hausdorff Dimension (HD), a widely recognized concept in fractal geometry used to quantify structural complexity, which aids in enhancing the diversity of generated outputs. To incorporate HD into FGM, we propose a learnable HD estimation method that predicts HD directly from image embeddings, addressing computational cost concerns. However, simply introducing HD into a hybrid loss is insufficient to enhance diversity in FGMs due to: 1) degradation of image quality, and 2) limited improvement in generation diversity. To this end, during training, we adopt an HD-based loss with a monotonic momentum-driven scheduling strategy to progressively optimize the hyperparameters, obtaining optimal diversity without sacrificing visual quality. Moreover, during inference, we employ HD-guided rejection sampling to select geometrically richer outputs. Extensive experiments on the ImageNet dataset demonstrate that our FGM-HD framework yields a 39\% improvement in output diversity compared to vanilla FGMs, while preserving comparable image quality. To our knowledge, this is the very first work introducing HD into FGM. Our method effectively enhances the diversity of generated outputs while offering a principled theoretical contribution to FGM development.
>
---
#### [replaced 083] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.05430v2](https://arxiv.org/pdf/2508.05430v2)**

> **作者:** Hubert Baniecki; Maximilian Muschalik; Fabian Fumagalli; Barbara Hammer; Eyke Hüllermeier; Przemyslaw Biecek
>
> **备注:** NeurIPS 2025. Code: https://github.com/hbaniecki/fixlip
>
> **摘要:** Language-image pre-training (LIP) enables the development of vision-language models capable of zero-shot classification, localization, multimodal retrieval, and semantic understanding. Various explanation methods have been proposed to visualize the importance of input image-text pairs on the model's similarity outputs. However, popular saliency maps are limited by capturing only first-order attributions, overlooking the complex cross-modal interactions intrinsic to such encoders. We introduce faithful interaction explanations of LIP models (FIxLIP) as a unified approach to decomposing the similarity in vision-language encoders. FIxLIP is rooted in game theory, where we analyze how using the weighted Banzhaf interaction index offers greater flexibility and improves computational efficiency over the Shapley interaction quantification framework. From a practical perspective, we propose how to naturally extend explanation evaluation metrics, such as the pointing game and area between the insertion/deletion curves, to second-order interaction explanations. Experiments on the MS COCO and ImageNet-1k benchmarks validate that second-order methods, such as FIxLIP, outperform first-order attribution methods. Beyond delivering high-quality explanations, we demonstrate the utility of FIxLIP in comparing different models, e.g. CLIP vs. SigLIP-2.
>
---
#### [replaced 084] Unlocking the Forgery Detection Potential of Vanilla MLLMs: A Novel Training-Free Pipeline
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.13442v2](https://arxiv.org/pdf/2511.13442v2)**

> **作者:** Rui Zuo; Qinyue Tong; Zhe-Ming Lu; Ziqian Lu
>
> **摘要:** With the rapid advancement of artificial intelligence-generated content (AIGC) technologies, including multimodal large language models (MLLMs) and diffusion models, image generation and manipulation have become remarkably effortless. Existing image forgery detection and localization (IFDL) methods often struggle to generalize across diverse datasets and offer limited interpretability. Nowadays, MLLMs demonstrate strong generalization potential across diverse vision-language tasks, and some studies introduce this capability to IFDL via large-scale training. However, such approaches cost considerable computational resources, while failing to reveal the inherent generalization potential of vanilla MLLMs to address this problem. Inspired by this observation, we propose Foresee, a training-free MLLM-based pipeline tailored for image forgery analysis. It eliminates the need for additional training and enables a lightweight inference process, while surpassing existing MLLM-based methods in both tamper localization accuracy and the richness of textual explanations. Foresee employs a type-prior-driven strategy and utilizes a Flexible Feature Detector (FFD) module to specifically handle copy-move manipulations, thereby effectively unleashing the potential of vanilla MLLMs in the forensic domain. Extensive experiments demonstrate that our approach simultaneously achieves superior localization accuracy and provides more comprehensive textual explanations. Moreover, Foresee exhibits stronger generalization capability, outperforming existing IFDL methods across various tampering types, including copy-move, splicing, removal, local enhancement, deepfake, and AIGC-based editing. The code will be released in the final version.
>
---
#### [replaced 085] Video Compression Commander: Plug-and-Play Inference Acceleration for Video Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.14454v2](https://arxiv.org/pdf/2505.14454v2)**

> **作者:** Xuyang Liu; Yiyu Wang; Junpeng Ma; Linfeng Zhang
>
> **备注:** EMNLP 2025 main
>
> **摘要:** Video large language models (VideoLLM) excel at video understanding, but face efficiency challenges due to the quadratic complexity of abundant visual tokens. Our systematic analysis of token compression methods for VideoLLMs reveals two critical issues: (i) overlooking distinctive visual signals across frames, leading to information loss; (ii) suffering from implementation constraints, causing incompatibility with modern architectures or efficient operators. To address these challenges, we distill three design principles for VideoLLM token compression and propose a plug-and-play inference acceleration framework "Video Compression Commander" (VidCom2). By quantifying each frame's uniqueness, VidCom2 adaptively adjusts compression intensity across frames, effectively preserving essential information while reducing redundancy in video sequences. Extensive experiments across various VideoLLMs and benchmarks demonstrate the superior performance and efficiency of our VidCom2. With only 25% visual tokens, VidCom2 achieves 99.6% of the original performance on LLaVA-OV while reducing 70.8% of the LLM generation latency. Notably, our Frame Compression Adjustment strategy is compatible with other token compression methods to further improve their performance. Our code is available at https://github.com/xuyang-liu16/VidCom2.
>
---
#### [replaced 086] Large Language Models and 3D Vision for Intelligent Robotic Perception and Autonomy
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11777v2](https://arxiv.org/pdf/2511.11777v2)**

> **作者:** Vinit Mehta; Charu Sharma; Karthick Thiyagarajan
>
> **备注:** 45 pages, 15 figures, MDPI Sensors Journal
>
> **摘要:** With the rapid advancement of artificial intelligence and robotics, the integration of Large Language Models (LLMs) with 3D vision is emerging as a transformative approach to enhancing robotic sensing technologies. This convergence enables machines to perceive, reason and interact with complex environments through natural language and spatial understanding, bridging the gap between linguistic intelligence and spatial perception. This review provides a comprehensive analysis of state-of-the-art methodologies, applications and challenges at the intersection of LLMs and 3D vision, with a focus on next-generation robotic sensing technologies. We first introduce the foundational principles of LLMs and 3D data representations, followed by an in-depth examination of 3D sensing technologies critical for robotics. The review then explores key advancements in scene understanding, text-to-3D generation, object grounding and embodied agents, highlighting cutting-edge techniques such as zero-shot 3D segmentation, dynamic scene synthesis and language-guided manipulation. Furthermore, we discuss multimodal LLMs that integrate 3D data with touch, auditory and thermal inputs, enhancing environmental comprehension and robotic decision-making. To support future research, we catalog benchmark datasets and evaluation metrics tailored for 3D-language and vision tasks. Finally, we identify key challenges and future research directions, including adaptive model architectures, enhanced cross-modal alignment and real-time processing capabilities, which pave the way for more intelligent, context-aware and autonomous robotic sensing systems.
>
---
#### [replaced 087] SlotMatch: Distilling Object-Centric Representations for Unsupervised Video Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.03411v3](https://arxiv.org/pdf/2508.03411v3)**

> **作者:** Diana-Nicoleta Grigore; Neelu Madan; Andreas Mogelmose; Thomas B. Moeslund; Radu Tudor Ionescu
>
> **摘要:** Unsupervised video segmentation is a challenging computer vision task, especially due to the lack of supervisory signals coupled with the complexity of visual scenes. To overcome this challenge, state-of-the-art models based on slot attention often have to rely on large and computationally expensive neural architectures. To this end, we propose a simple knowledge distillation framework that effectively transfers object-centric representations to a lightweight student. The proposed framework, called SlotMatch, aligns corresponding teacher and student slots via the cosine similarity, requiring no additional distillation objectives or auxiliary supervision. The simplicity of SlotMatch is confirmed via theoretical and empirical evidence, both indicating that integrating additional losses is redundant. We conduct experiments on three datasets to compare the state-of-the-art teacher model, SlotContrast, with our distilled student. The results show that our student based on SlotMatch matches and even outperforms its teacher, while using 3.6x less parameters and running up to 2.7x faster. Moreover, our student surpasses all other state-of-the-art unsupervised video segmentation models.
>
---
#### [replaced 088] Viper-F1: Fast and Fine-Grained Multimodal Understanding with Cross-Modal State-Space Modulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11177v3](https://arxiv.org/pdf/2511.11177v3)**

> **作者:** Quoc-Huy Trinh; Mustapha Abdullahi; Do Duy Hung Trinh; Bo Zhao; Debesh Jha
>
> **备注:** Need to enhance the method and benchmark to be better
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have enabled impressive progress in vision-language understanding, yet their high computational cost limits deployment in resource-constrained scenarios such as robotic manipulation, personal assistants, and smart cameras. Most existing methods rely on Transformer-based cross-attention, whose quadratic complexity hinders efficiency. Moreover, small vision-language models often struggle to precisely capture fine-grained, task-relevant visual regions, leading to degraded performance on fine-grained reasoning tasks that limit their effectiveness in the real world. To address these issues, we introduce Viper-F1, a Hybrid State-Space Vision-Language Model that replaces attention with efficient Liquid State-Space Dynamics. To further enhance visual grounding, we propose a Token-Grid Correlation Module, which computes lightweight correlations between text tokens and image patches and modulates the state-space dynamics via FiLM conditioning. This enables the model to selectively emphasize visual regions relevant to the textual prompt while maintaining linear-time inference. Experimental results across multiple benchmarks demonstrate that Viper-F1 achieves accurate, fine-grained understanding with significantly improved efficiency.
>
---
#### [replaced 089] Neural Atlas Graphs for Dynamic Scene Decomposition and Editing
- **分类: cs.GR; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.16336v3](https://arxiv.org/pdf/2509.16336v3)**

> **作者:** Jan Philipp Schneider; Pratik Singh Bisht; Ilya Chugunov; Andreas Kolb; Michael Moeller; Felix Heide
>
> **摘要:** Learning editable high-resolution scene representations for dynamic scenes is an open problem with applications across the domains from autonomous driving to creative editing - the most successful approaches today make a trade-off between editability and supporting scene complexity: neural atlases represent dynamic scenes as two deforming image layers, foreground and background, which are editable in 2D, but break down when multiple objects occlude and interact. In contrast, scene graph models make use of annotated data such as masks and bounding boxes from autonomous-driving datasets to capture complex 3D spatial relationships, but their implicit volumetric node representations are challenging to edit view-consistently. We propose Neural Atlas Graphs (NAGs), a hybrid high-resolution scene representation, where every graph node is a view-dependent neural atlas, facilitating both 2D appearance editing and 3D ordering and positioning of scene elements. Fit at test-time, NAGs achieve state-of-the-art quantitative results on the Waymo Open Dataset - by 5 dB PSNR increase compared to existing methods - and make environmental editing possible in high resolution and visual quality - creating counterfactual driving scenarios with new backgrounds and edited vehicle appearance. We find that the method also generalizes beyond driving scenes and compares favorably - by more than 7 dB in PSNR - to recent matting and video editing baselines on the DAVIS video dataset with a diverse set of human and animal-centric scenes. Project Page: https://princeton-computational-imaging.github.io/nag/
>
---
#### [replaced 090] LENS: Learning to Segment Anything with Unified Reinforced Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.14153v2](https://arxiv.org/pdf/2508.14153v2)**

> **作者:** Lianghui Zhu; Bin Ouyang; Yuxuan Zhang; Tianheng Cheng; Rui Hu; Haocheng Shen; Longjin Ran; Xiaoxin Chen; Li Yu; Wenyu Liu; Xinggang Wang
>
> **备注:** Code is released at https://github.com/hustvl/LENS
>
> **摘要:** Text-prompted image segmentation enables fine-grained visual understanding and is critical for applications such as human-computer interaction and robotics. However, existing supervised fine-tuning methods typically ignore explicit chain-of-thought (CoT) reasoning at test time, which limits their ability to generalize to unseen prompts and domains. To address this issue, we introduce LENS, a scalable reinforcement-learning framework that jointly optimizes the reasoning process and segmentation in an end-to-end manner. We propose unified reinforcement-learning rewards that span sentence-, box-, and segment-level cues, encouraging the model to generate informative CoT rationales while refining mask quality. Using a publicly available 3-billion-parameter vision-language model, i.e., Qwen2.5-VL-3B-Instruct, LENS achieves an average cIoU of 81.2% on the RefCOCO, RefCOCO+, and RefCOCOg benchmarks, outperforming the strong fine-tuned method, i.e., GLaMM, by up to 5.6%. These results demonstrate that RL-driven CoT reasoning significantly enhances text-prompted segmentation and offers a practical path toward more generalizable Segment Anything models (SAM). Code is available at https://github.com/hustvl/LENS.
>
---
#### [replaced 091] Towards Understanding 3D Vision: the Role of Gaussian Curvature
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11825v2](https://arxiv.org/pdf/2508.11825v2)**

> **作者:** Sherlon Almeida da Silva; Davi Geiger; Luiz Velho; Moacir Antonelli Ponti
>
> **摘要:** Recent advances in computer vision have predominantly relied on data-driven approaches that leverage deep learning and large-scale datasets. Deep neural networks have achieved remarkable success in tasks such as stereo matching and monocular depth reconstruction. However, these methods lack explicit models of 3D geometry that can be directly analyzed, transferred across modalities, or systematically modified for controlled experimentation. We investigate the role of Gaussian curvature in 3D surface modeling. Besides Gaussian curvature being an invariant quantity under change of observers or coordinate systems, we demonstrate using the Middlebury stereo dataset that it offers a sparse and compact description of 3D surfaces. Furthermore, we show a strong correlation between the performance rank of top state-of-the-art stereo and monocular methods and the low total absolute Gaussian curvature. We propose that this property can serve as a geometric prior to improve future 3D reconstruction algorithms.
>
---
#### [replaced 092] Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.12089v2](https://arxiv.org/pdf/2510.12089v2)**

> **作者:** Xingpei Ma; Shenneng Huang; Jiaran Cai; Yuansheng Guan; Shen Zheng; Hanfeng Zhao; Qiang Zhang; Shunsi Zhang
>
> **备注:** AAAI 2026
>
> **摘要:** Recent advances in diffusion models have significantly improved audio-driven human video generation, surpassing traditional methods in both quality and controllability. However, existing approaches still face challenges in lip-sync accuracy, temporal coherence for long video generation, and multi-character animation. In this work, we propose a diffusion transformer (DiT)-based framework for generating lifelike talking videos of arbitrary length, and introduce a training-free method for multi-character audio-driven animation. First, we employ a LoRA-based training strategy combined with a position shift inference approach, which enables efficient long video generation while preserving the capabilities of the foundation model. Moreover, we combine partial parameter updates with reward feedback to enhance both lip synchronization and natural body motion. Finally, we propose a training-free approach, Mask Classifier-Free Guidance (Mask-CFG), for multi-character animation, which requires no specialized datasets or model modifications and supports audio-driven animation for three or more characters. Experimental results demonstrate that our method outperforms existing state-of-the-art approaches, achieving high-quality, temporally coherent, and multi-character audio-driven video generation in a simple, efficient, and cost-effective manner.
>
---
#### [replaced 093] MoHoBench: Assessing Honesty of Multimodal Large Language Models via Unanswerable Visual Questions
- **分类: cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.21503v2](https://arxiv.org/pdf/2507.21503v2)**

> **作者:** Yanxu Zhu; Shitong Duan; Xiangxu Zhang; Jitao Sang; Peng Zhang; Tun Lu; Xiao Zhou; Jing Yao; Xiaoyuan Yi; Xing Xie
>
> **备注:** AAAI2026 Oral
>
> **摘要:** Recently Multimodal Large Language Models (MLLMs) have achieved considerable advancements in vision-language tasks, yet produce potentially harmful or untrustworthy content. Despite substantial work investigating the trustworthiness of language models, MMLMs' capability to act honestly, especially when faced with visually unanswerable questions, remains largely underexplored. This work presents the first systematic assessment of honesty behaviors across various MLLMs. We ground honesty in models' response behaviors to unanswerable visual questions, define four representative types of such questions, and construct MoHoBench, a large-scale MMLM honest benchmark, consisting of 12k+ visual question samples, whose quality is guaranteed by multi-stage filtering and human verification. Using MoHoBench, we benchmarked the honesty of 28 popular MMLMs and conducted a comprehensive analysis. Our findings show that: (1) most models fail to appropriately refuse to answer when necessary, and (2) MMLMs' honesty is not solely a language modeling issue, but is deeply influenced by visual information, necessitating the development of dedicated methods for multimodal honesty alignment. Therefore, we implemented initial alignment methods using supervised and preference learning to improve honesty behavior, providing a foundation for future work on trustworthy MLLMs. Our data and code can be found at https://github.com/yanxuzhu/MoHoBench.
>
---
#### [replaced 094] Real-Time Sign Language to text Translation using Deep Learning: A Comparative study of LSTM and 3D CNN
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13137v2](https://arxiv.org/pdf/2510.13137v2)**

> **作者:** Madhumati Pol; Anvay Anturkar; Anushka Khot; Ayush Andure; Aniruddha Ghosh; Anvit Magadum; Anvay Bahadur
>
> **摘要:** This study investigates the performance of 3D Convolutional Neural Networks (3D CNNs) and Long Short-Term Memory (LSTM) networks for real-time American Sign Language (ASL) recognition. Though 3D CNNs are good at spatiotemporal feature extraction from video sequences, LSTMs are optimized for modeling temporal dependencies in sequential data. We evaluate both architectures on a dataset containing 1,200 ASL signs across 50 classes, comparing their accuracy, computational efficiency, and latency under similar training conditions. Experimental results demonstrate that 3D CNNs achieve 92.4% recognition accuracy but require 3.2% more processing time per frame compared to LSTMs, which maintain 86.7% accuracy with significantly lower resource consumption. The hybrid 3D CNNLSTM model shows decent performance, which suggests that context-dependent architecture selection is crucial for practical implementation.This project provides professional benchmarks for developing assistive technologies, highlighting trade-offs between recognition precision and real-time operational requirements in edge computing environments.
>
---
#### [replaced 095] Learning few-step posterior samplers by unfolding and distillation of diffusion models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.02686v2](https://arxiv.org/pdf/2507.02686v2)**

> **作者:** Charlesquin Kemajou Mbakam; Jonathan Spence; Marcelo Pereyra
>
> **备注:** 34 pages, 18 figures, 11 tables
>
> **摘要:** Diffusion models (DMs) have emerged as powerful image priors in Bayesian computational imaging. Two primary strategies have been proposed for leveraging DMs in this context: Plug-and-Play methods, which are zero-shot and highly flexible but rely on approximations; and specialized conditional DMs, which achieve higher accuracy and faster inference for specific tasks through supervised training. In this work, we introduce a novel framework that integrates deep unfolding and model distillation to transform a DM image prior into a few-step conditional model for posterior sampling. A central innovation of our approach is the unfolding of a Markov chain Monte Carlo (MCMC) algorithm - specifically, the recently proposed LATINO Langevin sampler (Spagnoletti et al., 2025) - representing the first known instance of deep unfolding applied to a Monte Carlo sampling scheme. We demonstrate our proposed unfolded and distilled samplers through extensive experiments and comparisons with the state of the art, where they achieve excellent accuracy and computational efficiency, while retaining the flexibility to adapt to variations in the forward model at inference time.
>
---
#### [replaced 096] MAVias: Mitigate any Visual Bias
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.06632v2](https://arxiv.org/pdf/2412.06632v2)**

> **作者:** Ioannis Sarridis; Christos Koutlis; Symeon Papadopoulos; Christos Diou
>
> **摘要:** Mitigating biases in computer vision models is an essential step towards the trustworthiness of artificial intelligence models. Existing bias mitigation methods focus on a small set of predefined biases, limiting their applicability in visual datasets where multiple, possibly unknown biases exist. To address this limitation, we introduce MAVias, an open-set bias mitigation approach leveraging foundation models to discover spurious associations between visual attributes and target classes. MAVias first captures a wide variety of visual features in natural language via a foundation image tagging model, and then leverages a large language model to select those visual features defining the target class, resulting in a set of language-coded potential visual biases. We then translate this set of potential biases into vision-language embeddings and introduce an in-processing bias mitigation approach to prevent the model from encoding information related to them. Our experiments on diverse datasets, including CelebA, Waterbirds, ImageNet, and UrbanCars, show that MAVias effectively detects and mitigates a wide range of biases in visual recognition tasks outperforming current state-of-the-art.
>
---
#### [replaced 097] STONE: Pioneering the One-to-N Backdoor Threat in 3D Point Cloud
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11210v2](https://arxiv.org/pdf/2511.11210v2)**

> **作者:** Dongmei Shan; Wei Lian; Chongxia Wang
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Backdoor attacks pose a critical threat to deep learning, especially in safety-sensitive 3D domains such as autonomous driving and robotics. Despite their potency, existing attacks on 3D point clouds are limited to a static one-to-one paradigm, leaving the more flexible one-to-N backdoor threat largely unexplored and without a theoretical or practical foundation. We address this by introducing STONE (Spherical Trigger One-to-N Backdoor Enabling), the first framework that instantiates this threat through a configurable spherical trigger. Its parameterizable spatial properties create a dynamic key space, enabling a single trigger to control multiple output labels. Theoretically, we ground STONE through Neural Tangent Kernel (NTK) analysis, providing the first formal basis for one-to-N mappings in 3D models. Empirically, extensive evaluations show high attack success rate (up to 100\%) with no loss in clean-data accuracy. This work establishes a foundational benchmark for multi-target threats in 3D vision, crucial for securing future intelligent systems.
>
---
#### [replaced 098] UniVST: A Unified Framework for Training-free Localized Video Style Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.20084v5](https://arxiv.org/pdf/2410.20084v5)**

> **作者:** Quanjian Song; Mingbao Lin; Wengyi Zhan; Shuicheng Yan; Liujuan Cao; Rongrong Ji
>
> **备注:** Accepted by TPAMI 2025; Project Page: https://quanjiansong.github.io/projects/UniVST
>
> **摘要:** This paper presents UniVST, a unified framework for localized video style transfer based on diffusion models. It operates without the need for training, offering a distinct advantage over existing diffusion methods that transfer style across entire videos. The endeavors of this paper comprise: (1) A point-matching mask propagation strategy that leverages the feature maps from the DDIM inversion. This streamlines the model's architecture by obviating the need for tracking models. (2) A training-free AdaIN-guided localized video stylization mechanism that operates at both the latent and attention levels. This balances content fidelity and style richness, mitigating the loss of localized details commonly associated with direct video stylization. (3) A sliding-window consistent smoothing scheme that harnesses optical flow within the pixel representation and refines predicted noise to update the latent space. This significantly enhances temporal consistency and diminishes artifacts in stylized video. Our proposed UniVST has been validated to be superior to existing methods in quantitative and qualitative metrics. It adeptly addresses the challenges of preserving the primary object's style while ensuring temporal consistency and detail preservation. Our code is available at https://github.com/QuanjianSong/UniVST.
>
---
#### [replaced 099] On the Topological Foundation of Learning and Memory
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/1103.1587v3](https://arxiv.org/pdf/1103.1587v3)**

> **作者:** Xin Li
>
> **摘要:** We propose a formal foundation for cognition rooted in algebraic topology, built on a Homological Parity Principle. This posits that even-dimensional homology represents stable Structure/Context (e.g., generative models), while odd-dimensional homology represents dynamic Flow/Content (e.g., sensory/memory data). Cognition is governed by the Context-Content Uncertainty Principle (CCUP), a dynamical cycle aligning these parities. This framework distinguishes two modes: Inference (waking), where the scaffold predicts the flow (a Context-before-Content process); and Learning (sleep), an inverted Structure-before-Specificity process where memory traces sculpt the scaffold. This parity interpretation unifies cognitive functions like semantic and episodic memory and provides a structural generalization of existing theories, recasting Friston's Free Energy Principle and Tonini's Integrated Information in topological terms.
>
---
#### [replaced 100] Deep Equilibrium models for Poisson Imaging Inverse problems via Mirror Descent
- **分类: math.OC; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.11461v2](https://arxiv.org/pdf/2507.11461v2)**

> **作者:** Christian Daniele; Silvia Villa; Samuel Vaiter; Luca Calatroni
>
> **摘要:** Deep Equilibrium Models (DEQs) are implicit neural networks with fixed points, which have recently gained attention for learning image regularization functionals, particularly in settings involving Gaussian fidelities, where assumptions on the forward operator ensure contractiveness of standard (proximal) Gradient Descent operators. In this work, we extend the application of DEQs to Poisson inverse problems, where the data fidelity term is more appropriately modeled by the Kullback--Leibler divergence. To this end, we introduce a novel DEQ formulation based on Mirror Descent defined in terms of a tailored non-Euclidean geometry that naturally adapts with the structure of the data term. This enables the learning of neural regularizers within a principled training framework. We derive sufficient conditions and establish refined convergence results based on the Kurdyka--Lojasiewicz framework for subanalytic functions with non-closed domains to guarantee the convergence of the learned reconstruction scheme and propose computational strategies that enable both efficient training and parameter-free inference. Numerical experiments show that our method outperforms traditional model-based approaches and it is comparable to the performance of Bregman Plug-and-Play methods, while mitigating their typical drawbacks, such as time-consuming tuning of hyper-parameters. The code is publicly available at https://github.com/christiandaniele/DEQ-MD.
>
---
#### [replaced 101] DeSamba: Decoupled Spectral Adaptive Framework for 3D Multi-Sequence MRI Lesion Classification
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.15487v3](https://arxiv.org/pdf/2507.15487v3)**

> **作者:** Dezhen Wang; Sheng Miao; Rongxin Chai; Jiufa Cui
>
> **备注:** Our manuscript requires further experimental work, and the dataset cannot be made publicly available; therefore, we respectfully request withdrawal of the paper
>
> **摘要:** Magnetic Resonance Imaging (MRI) sequences provide rich spatial and frequency domain information, which is crucial for accurate lesion classification in medical imaging. However, effectively integrating multi-sequence MRI data for robust 3D lesion classification remains a challenge. In this paper, we propose DeSamba (Decoupled Spectral Adaptive Network and Mamba-Based Model), a novel framework designed to extract decoupled representations and adaptively fuse spatial and spectral features for lesion classification. DeSamba introduces a Decoupled Representation Learning Module (DRLM) that decouples features from different MRI sequences through self-reconstruction and cross-reconstruction, and a Spectral Adaptive Modulation Block (SAMB) within the proposed SAMNet, enabling dynamic fusion of spectral and spatial information based on lesion characteristics. We evaluate DeSamba on two clinically relevant 3D datasets. On a six-class spinal metastasis dataset (n=1,448), DeSamba achieves 62.10% Top-1 accuracy, 63.62% F1-score, 87.71% AUC, and 93.55% Top-3 accuracy on an external validation set (n=372), outperforming all state-of-the-art (SOTA) baselines. On a spondylitis dataset (n=251) involving a challenging binary classification task, DeSamba achieves 70.00%/64.52% accuracy and 74.75/73.88 AUC on internal and external validation sets, respectively. Ablation studies demonstrate that both DRLM and SAMB significantly contribute to overall performance, with over 10% relative improvement compared to the baseline. Our results highlight the potential of DeSamba as a generalizable and effective solution for 3D lesion classification in multi-sequence medical imaging.
>
---
#### [replaced 102] RelTopo: Multi-Level Relational Modeling for Driving Scene Topology Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.13553v3](https://arxiv.org/pdf/2506.13553v3)**

> **作者:** Yueru Luo; Changqing Zhou; Yiming Yang; Erlong Li; Chao Zheng; Shuqi Mei; Shuguang Cui; Zhen Li
>
> **备注:** Preprint. Under review
>
> **摘要:** Accurate road topology reasoning is critical for autonomous driving, as it requires both perceiving road elements and understanding how lanes connect to each other (L2L) and to traffic elements (L2T). Existing methods often focus on either perception or L2L reasoning, leaving L2T underexplored and fall short of jointly optimizing perception and reasoning. Moreover, although topology prediction inherently involves relations, relational modeling itself is seldom incorporated into feature extraction or supervision. As humans naturally leverage contextual relationships to recognize road element and infer their connectivity, we posit that relational modeling can likewise benefit both perception and reasoning, and that these two tasks should be mutually enhancing. To this end, we propose RelTopo, a multi-level relational modeling approach that systematically integrates relational cues across three levels: 1) perception-level: a relation-aware lane detector with geometry-biased self-attention and curve-guided cross-attention enriches lane representations; 2) reasoning-level: relation-enhanced topology heads, including a geometry-enhanced L2L head and a cross-view L2T head, enhance topology inference via relational cues; and 3) supervision-level: a contrastive InfoNCE strategy regularizes relational embeddings. This design enables perception and reasoning to be optimized jointly. Extensive experiments on OpenLane-V2 demonstrate that RelTopo significantly improves both detection and topology reasoning, with gains of +3.1 in DET$_l$, +5.3 in TOP$_{ll}$, +4.9 in TOP$_{lt}$, and +4.4 overall in OLS, setting a new state-of-the-art. Code will be released.
>
---
#### [replaced 103] Accuracy is Not Enough: Poisoning Interpretability in Federated Learning via Color Skew
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13535v2](https://arxiv.org/pdf/2511.13535v2)**

> **作者:** Farhin Farhad Riya; Shahinul Hoque; Jinyuan Stella Sun; Olivera Kotevska
>
> **摘要:** As machine learning models are increasingly deployed in safety-critical domains, visual explanation techniques have become essential tools for supporting transparency. In this work, we reveal a new class of attacks that compromise model interpretability without affecting accuracy. Specifically, we show that small color perturbations applied by adversarial clients in a federated learning setting can shift a model's saliency maps away from semantically meaningful regions while keeping the prediction unchanged. The proposed saliency-aware attack framework, called Chromatic Perturbation Module, systematically crafts adversarial examples by altering the color contrast between foreground and background in a way that disrupts explanation fidelity. These perturbations accumulate across training rounds, poisoning the global model's internal feature attributions in a stealthy and persistent manner. Our findings challenge a common assumption in model auditing that correct predictions imply faithful explanations and demonstrate that interpretability itself can be an attack surface. We evaluate this vulnerability across multiple datasets and show that standard training pipelines are insufficient to detect or mitigate explanation degradation, especially in the federated learning setting, where subtle color perturbations are harder to discern. Our attack reduces peak activation overlap in Grad-CAM explanations by up to 35% while preserving classification accuracy above 96% on all evaluated datasets.
>
---
#### [replaced 104] Spatial Policy: Guiding Visuomotor Robotic Manipulation with Spatial-Aware Modeling and Reasoning
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.15874v2](https://arxiv.org/pdf/2508.15874v2)**

> **作者:** Yijun Liu; Yuwei Liu; Yuan Meng; Jieheng Zhang; Yuwei Zhou; Ye Li; Jiacheng Jiang; Kangye Ji; Shijia Ge; Zhi Wang; Wenwu Zhu
>
> **摘要:** Vision-centric hierarchical embodied models have demonstrated strong potential. However, existing methods lack spatial awareness capabilities, limiting their effectiveness in bridging visual plans to actionable control in complex environments. To address this problem, we propose Spatial Policy (SP), a unified spatial-aware visuomotor robotic manipulation framework via explicit spatial modeling and reasoning. Specifically, we first design a spatial-conditioned embodied video generation module to model spatially guided predictions through the spatial plan table. Then, we propose a flow-based action prediction module to infer executable actions with coordination. Finally, we propose a spatial reasoning feedback policy to refine the spatial plan table via dual-stage replanning. Extensive experiments show that SP substantially outperforms state-of-the-art baselines, achieving over 33% improvement on Meta-World and over 25% improvement on iTHOR, demonstrating strong effectiveness across 23 embodied control tasks. We additionally evaluate SP in real-world robotic experiments to verify its practical viability. SP enhances the practicality of embodied models for robotic control applications. Code and checkpoints are maintained at https://plantpotatoonmoon.github.io/SpatialPolicy/.
>
---
#### [replaced 105] Logos as a Well-Tempered Pre-train for Sign Language Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.10481v2](https://arxiv.org/pdf/2505.10481v2)**

> **作者:** Ilya Ovodov; Petr Surovtsev; Karina Kvanchiani; Alexander Kapitanov; Alexander Nagaev
>
> **摘要:** This paper examines two aspects of the isolated sign language recognition (ISLR) task. First, although a certain number of datasets is available, the data for individual sign languages is limited. It poses the challenge of cross-language ISLR model training, including transfer learning. Second, similar signs can have different semantic meanings. It leads to ambiguity in dataset labeling and raises the question of the best policy for annotating such signs. To address these issues, this study presents Logos, a novel Russian Sign Language (RSL) dataset, the most extensive available ISLR dataset by the number of signers, one of the most extensive datasets in size and vocabulary, and the largest RSL dataset. It is shown that a model, pre-trained on the Logos dataset can be used as a universal encoder for other language SLR tasks, including few-shot learning. We explore cross-language transfer learning approaches and find that joint training using multiple classification heads benefits accuracy for the target low-resource datasets the most. The key feature of the Logos dataset is explicitly annotated visually similar sign groups. We show that explicitly labeling visually similar signs improves trained model quality as a visual encoder for downstream tasks. Based on the proposed contributions, we outperform current state-of-the-art results for the WLASL dataset and get competitive results for the AUTSL dataset, with a single stream model processing solely RGB video. The source code, dataset, and pre-trained models are publicly available.
>
---
#### [replaced 106] Beyond Flatlands: Unlocking Spatial Intelligence by Decoupling 3D Reasoning from Numerical Regression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11239v2](https://arxiv.org/pdf/2511.11239v2)**

> **作者:** Zhongbin Guo; Jiahe Liu; Yushan Li; Wenyu Gao; Zhen Yang; Chenzhi Li; Xinyue Zhang; Ping Jian
>
> **摘要:** Existing Vision Language Models (VLMs) architecturally rooted in "flatland" perception, fundamentally struggle to comprehend real-world 3D spatial intelligence. This failure stems from a dual-bottleneck: input-stage conflict between computationally exorbitant geometric-aware encoders and superficial 2D-only features, and output-stage misalignment where discrete tokenizers are structurally incapable of producing precise, continuous numerical values. To break this impasse, we introduce GEODE (Geometric-Output and Decoupled-Input Engine), a novel architecture that resolves this dual-bottleneck by decoupling 3D reasoning from numerical generation. GEODE augments main VLM with two specialized, plug-and-play modules: Decoupled Rationale Module (DRM) that acts as spatial co-processor, aligning explicit 3D data with 2D visual features via cross-attention and distilling spatial Chain-of-Thought (CoT) logic into injectable Rationale Tokens; and Direct Regression Head (DRH), an "Embedding-as-Value" paradigm which routes specialized control tokens to a lightweight MLP for precise, continuous regression of scalars and 3D bounding boxes. The synergy of these modules allows our 1.5B parameter model to function as a high-level semantic dispatcher, achieving state-of-the-art spatial reasoning performance that rivals 7B+ models.
>
---
#### [replaced 107] Rethinking Saliency Maps: A Cognitive Human Aligned Taxonomy and Evaluation Framework for Explanations
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.13081v2](https://arxiv.org/pdf/2511.13081v2)**

> **作者:** Yehonatan Elisha; Seffi Cohen; Oren Barkan; Noam Koenigstein
>
> **摘要:** Saliency maps are widely used for visual explanations in deep learning, but a fundamental lack of consensus persists regarding their intended purpose and alignment with diverse user queries. This ambiguity hinders the effective evaluation and practical utility of explanation methods. We address this gap by introducing the Reference-Frame $\times$ Granularity (RFxG) taxonomy, a principled conceptual framework that organizes saliency explanations along two essential axes:Reference-Frame: Distinguishing between pointwise ("Why this prediction?") and contrastive ("Why this and not an alternative?") explanations. Granularity: Ranging from fine-grained class-level (e.g., "Why Husky?") to coarse-grained group-level (e.g., "Why Dog?") interpretations. Using the RFxG lens, we demonstrate critical limitations in existing evaluation metrics, which overwhelmingly prioritize pointwise faithfulness while neglecting contrastive reasoning and semantic granularity. To systematically assess explanation quality across both RFxG dimensions, we propose four novel faithfulness metrics. Our comprehensive evaluation framework applies these metrics to ten state-of-the-art saliency methods, four model architectures, and three datasets. By advocating a shift toward user-intent-driven evaluation, our work provides both the conceptual foundation and the practical tools necessary to develop visual explanations that are not only faithful to the underlying model behavior but are also meaningfully aligned with the complexity of human understanding and inquiry.
>
---
#### [replaced 108] MicroEvoEval: A Systematic Evaluation Framework for Image-Based Microstructure Evolution Prediction
- **分类: cond-mat.mtrl-sci; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.08955v2](https://arxiv.org/pdf/2511.08955v2)**

> **作者:** Qinyi Zhang; Duanyu Feng; Ronghui Han; Yangshuai Wang; Hao Wang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Simulating microstructure evolution (MicroEvo) is vital for materials design but demands high numerical accuracy, efficiency, and physical fidelity. Although recent studies on deep learning (DL) offer a promising alternative to traditional solvers, the field lacks standardized benchmarks. Existing studies are flawed due to a lack of comparing specialized MicroEvo DL models with state-of-the-art spatio-temporal architectures, an overemphasis on numerical accuracy over physical fidelity, and a failure to analyze error propagation over time. To address these gaps, we introduce MicroEvoEval, the first comprehensive benchmark for image-based microstructure evolution prediction. We evaluate 14 models, encompassing both domain-specific and general-purpose architectures, across four representative MicroEvo tasks with datasets specifically structured for both short- and long-term assessment. Our multi-faceted evaluation framework goes beyond numerical accuracy and computational cost, incorporating a curated set of structure-preserving metrics to assess physical fidelity. Our extensive evaluations yield several key insights. Notably, we find that modern architectures (e.g., VMamba), not only achieve superior long-term stability and physical fidelity but also operate with an order-of-magnitude greater computational efficiency. The results highlight the necessity of holistic evaluation and identify these modern architectures as a highly promising direction for developing efficient and reliable surrogate models in data-driven materials science.
>
---
#### [replaced 109] From Perception to Reasoning: Deep Thinking Empowers Multimodal Large Language Models
- **分类: cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12861v2](https://arxiv.org/pdf/2511.12861v2)**

> **作者:** Wenxin Zhu; Andong Chen; Yuchen Song; Kehai Chen; Conghui Zhu; Ziyan Chen; Tiejun Zhao
>
> **备注:** Survey; 7 figures, 3 tables, 44 pages
>
> **摘要:** With the remarkable success of Multimodal Large Language Models (MLLMs) in perception tasks, enhancing their complex reasoning capabilities has emerged as a critical research focus. Existing models still suffer from challenges such as opaque reasoning paths and insufficient generalization ability. Chain-of-Thought (CoT) reasoning, which has demonstrated significant efficacy in language models by enhancing reasoning transparency and output interpretability, holds promise for improving model reasoning capabilities when extended to the multimodal domain. This paper provides a systematic review centered on "Multimodal Chain-of-Thought" (MCoT). First, it analyzes the background and theoretical motivations for its inception from the perspectives of technical evolution and task demands. Then, it introduces mainstream MCoT methods from three aspects: CoT paradigms, the post-training stage, and the inference stage, while also analyzing their underlying mechanisms. Furthermore, the paper summarizes existing evaluation benchmarks and metrics, and discusses the application scenarios of MCoT. Finally, it analyzes the challenges currently facing MCoT and provides an outlook on its future research directions.
>
---
#### [replaced 110] Learnable Total Variation with Lambda Mapping for Low-Dose CT Denoising
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10500v2](https://arxiv.org/pdf/2511.10500v2)**

> **作者:** Yusuf Talha Basak; Mehmet Ozan Unal; Metin Ertas; Isa Yildirim
>
> **摘要:** Although Total Variation (TV) performs well in noise reduction and edge preservation on images, its dependence on the lambda parameter limits its efficiency and makes it difficult to use effectively. In this study, we present a Learnable Total Variation (LTV) framework that couples an unrolled TV solver with a data-driven Lambda Mapping Network (LambdaNet) predicting a per-pixel regularization map. The pipeline is trained end-to-end so that reconstruction and regularization are optimized jointly, yielding spatially adaptive smoothing: strong in homogeneous regions, relaxed near anatomical boundaries. Experiments on the DeepLesion dataset, using a realistic noise model adapted from the LoDoPaB-CT methodology, show consistent gains over classical TV and FBP+U-Net: +2.9 dB PSNR and +6% SSIM on average. LTV provides an interpretable alternative to black-box CNNs and a basis for 3D and data-consistency-driven reconstruction.
>
---
#### [replaced 111] The Promise of RL for Autoregressive Image Editing
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.01119v3](https://arxiv.org/pdf/2508.01119v3)**

> **作者:** Saba Ahmadi; Rabiul Awal; Ankur Sikarwar; Amirhossein Kazemnejad; Ge Ya Luo; Juan A. Rodriguez; Sai Rajeswar; Siva Reddy; Christopher Pal; Benno Krojer; Aishwarya Agrawal
>
> **摘要:** While image generation techniques are now capable of producing high-quality images that respect prompts which span multiple sentences, the task of text-guided image editing remains a challenge. Even edit requests that consist of only a few words often fail to be executed correctly. We explore three strategies to enhance performance on a wide range of image editing tasks: supervised fine-tuning (SFT), reinforcement learning (RL), and Chain-of-Thought (CoT) reasoning. In order to study all these components in one consistent framework, we adopt an autoregressive multimodal model that processes textual and visual tokens in a unified manner. We find RL combined with a large multi-modal LLM verifier to be the most effective of these strategies. As a result, we release EARL: Editing with Autoregression and RL, a strong RL-based image editing model that performs competitively on a diverse range of edits compared to strong baselines, despite using much less training data. Thus, EARL pushes the frontier of autoregressive multimodal models on image editing. We release our code, training data, and trained models at https://github.com/mair-lab/EARL.
>
---
#### [replaced 112] SAM2MOT: A Novel Paradigm of Multi-Object Tracking by Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.04519v5](https://arxiv.org/pdf/2504.04519v5)**

> **作者:** Junjie Jiang; Zelin Wang; Manqi Zhao; Yin Li; DongSheng Jiang
>
> **摘要:** Inspired by Segment Anything 2, which generalizes segmentation from images to videos, we propose SAM2MOT--a novel segmentation-driven paradigm for multi-object tracking that breaks away from the conventional detection-association framework. In contrast to previous approaches that treat segmentation as auxiliary information, SAM2MOT places it at the heart of the tracking process, systematically tackling challenges like false positives and occlusions. Its effectiveness has been thoroughly validated on major MOT benchmarks. Furthermore, SAM2MOT integrates pre-trained detector, pre-trained segmentor with tracking logic into a zero-shot MOT system that requires no fine-tuning. This significantly reduces dependence on labeled data and paves the way for transitioning MOT research from task-specific solutions to general-purpose systems. Experiments on DanceTrack, UAVDT, and BDD100K show state-of-the-art results. Notably, SAM2MOT outperforms existing methods on DanceTrack by +2.1 HOTA and +4.5 IDF1, highlighting its effectiveness in MOT. Code is available at https://github.com/TripleJoy/SAM2MOT.
>
---
#### [replaced 113] Synthetic Geology: Structural Geology Meets Deep Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.11164v2](https://arxiv.org/pdf/2506.11164v2)**

> **作者:** Simon Ghyselincks; Valeriia Okhmak; Stefano Zampini; George Turkiyyah; David Keyes; Eldad Haber
>
> **备注:** 10 pages, 9 figures, geological simulation code at https://doi.org/10.5281/zenodo.15244035, generative AI code at https://github.com/chipnbits/flowtrain_stochastic_interpolation/releases/tag/v1.0.2
>
> **摘要:** Reconstructing the structural geology and mineral composition of the first few kilometers of the Earth's subsurface from sparse or indirect surface observations remains a long-standing challenge with critical applications in mineral exploration, geohazard assessment, and geotechnical engineering. This inherently ill-posed problem is often addressed by classical geophysical inversion methods, which typically yield a single maximum-likelihood model that fails to capture the full range of plausible geology. The adoption of modern deep learning methods has been limited by the lack of large 3D training datasets. We address this gap with \textit{StructuralGeo}, a geological simulation engine that mimics eons of tectonic, magmatic, and sedimentary processes to generate a virtually limitless supply of realistic synthetic 3D lithological models. Using this dataset, we train both unconditional and conditional generative flow-matching models with a 3D attention U-net architecture. The resulting foundation model can reconstruct multiple plausible 3D scenarios from surface topography and sparse borehole data, depicting structures such as layers, faults, folds, and dikes. By sampling many reconstructions from the same observations, we introduce a probabilistic framework for estimating the size and extent of subsurface features. While the realism of the output is bounded by the fidelity of the training data to true geology, this combination of simulation and generative AI functions offers a flexible prior for probabilistic modeling, regional fine-tuning, and use as an AI-based regularizer in traditional geophysical inversion workflows.
>
---
#### [replaced 114] Iris: Integrating Language into Diffusion-based Monocular Depth Estimation
- **分类: cs.CV; cs.CL; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2411.16750v4](https://arxiv.org/pdf/2411.16750v4)**

> **作者:** Ziyao Zeng; Jingcheng Ni; Daniel Wang; Patrick Rim; Younjoon Chung; Fengyu Yang; Byung-Woo Hong; Alex Wong
>
> **摘要:** Traditional monocular depth estimation suffers from inherent ambiguity and visual nuisances. We demonstrate that language can enhance monocular depth estimation by providing an additional condition (rather than images alone) aligned with plausible 3D scenes, thereby reducing the solution space for depth estimation. This conditional distribution is learned during the text-to-image pre-training of diffusion models. To generate images under various viewpoints and layouts that precisely reflect textual descriptions, the model implicitly models object sizes, shapes, and scales, their spatial relationships, and the overall scene structure. In this paper, Iris, we investigate the benefits of our strategy to integrate text descriptions into training and inference of diffusion-based depth estimation models. We experiment with three different diffusion-based monocular depth estimators (Marigold, Lotus, and E2E-FT) and their variants. By training on HyperSim and Virtual KITTI, and evaluating on NYUv2, KITTI, ETH3D, ScanNet, and DIODE, we find that our strategy improves the overall monocular depth estimation accuracy, especially in small areas. It also improves the model's depth perception of specific regions described in the text. We find that by providing more details in the text, the depth prediction can be iteratively refined. Simultaneously, we find that language can act as a constraint to accelerate the convergence of both training and the inference diffusion trajectory. Code and generated text data will be released upon acceptance.
>
---
#### [replaced 115] MMEdge: Accelerating On-device Multimodal Inference via Pipelined Sensing and Encoding
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.25327v5](https://arxiv.org/pdf/2510.25327v5)**

> **作者:** Runxi Huang; Mingxuan Yu; Mingyu Tsoi; Xiaomin Ouyang
>
> **备注:** Code available at: https://github.com/HKUST-MINSys-Lab/MMEdge. Accepted by SenSys 2026
>
> **摘要:** Real-time multimodal inference on resource-constrained edge devices is essential for applications such as autonomous driving, human-computer interaction, and mobile health. However, prior work often overlooks the tight coupling between sensing dynamics and model execution, as well as the complex inter-modality dependencies. In this paper, we propose MMEdge, an new on-device multi-modal inference framework based on pipelined sensing and encoding. Instead of waiting for complete sensor inputs, MMEdge decomposes the entire inference process into a sequence of fine-grained sensing and encoding units, allowing computation to proceed incrementally as data arrive. MMEdge also introduces a lightweight but effective temporal aggregation module that captures rich temporal dynamics across different pipelined units to maintain accuracy performance. Such pipelined design also opens up opportunities for fine-grained cross-modal optimization and early decision-making during inference. To further enhance system performance under resource variability and input data complexity, MMEdge incorporates an adaptive multimodal configuration optimizer that dynamically selects optimal sensing and model configurations for each modality under latency constraints, and a cross-modal speculative skipping mechanism that bypasses future units of slower modalities when early predictions reach sufficient confidence. We evaluate MMEdge using two public multimodal datasets and deploy it on a real-world unmanned aerial vehicle (UAV)-based multimodal testbed. The results show that MMEdge significantly reduces end-to-end latency while maintaining high task accuracy across various system and data dynamics.
>
---
#### [replaced 116] Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.12878v2](https://arxiv.org/pdf/2511.12878v2)**

> **作者:** Junyi Ma; Wentao Bao; Jingyi Xu; Guanzhong Sun; Yu Zheng; Erhang Zhang; Xieyuanli Chen; Hesheng Wang
>
> **备注:** Extended journal version of MMTwin (IROS'25)
>
> **摘要:** Forecasting how human hands move in egocentric views is critical for applications like augmented reality and human-robot policy transfer. Recently, several hand trajectory prediction (HTP) methods have been developed to generate future possible hand waypoints, which still suffer from insufficient prediction targets, inherent modality gaps, entangled hand-head motion, and limited validation in downstream tasks. To address these limitations, we present a universal hand motion forecasting framework considering multi-modal input, multi-dimensional and multi-target prediction patterns, and multi-task affordances for downstream applications. We harmonize multiple modalities by vision-language fusion, global context incorporation, and task-aware text embedding injection, to forecast hand waypoints in both 2D and 3D spaces. A novel dual-branch diffusion is proposed to concurrently predict human head and hand movements, capturing their motion synergy in egocentric vision. By introducing target indicators, the prediction model can forecast the specific joint waypoints of the wrist or the fingers, besides the widely studied hand center points. In addition, we enable Uni-Hand to additionally predict hand-object interaction states (contact/separation) to facilitate downstream tasks better. As the first work to incorporate downstream task evaluation in the literature, we build novel benchmarks to assess the real-world applicability of hand motion forecasting algorithms. The experimental results on multiple publicly available datasets and our newly proposed benchmarks demonstrate that Uni-Hand achieves the state-of-the-art performance in multi-dimensional and multi-target hand motion forecasting. Extensive validation in multiple downstream tasks also presents its impressive human-robot policy transfer to enable robotic manipulation, and effective feature enhancement for action anticipation/recognition.
>
---
#### [replaced 117] StyleDrive: Towards Driving-Style Aware Benchmarking of End-To-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2506.23982v3](https://arxiv.org/pdf/2506.23982v3)**

> **作者:** Ruiyang Hao; Bowen Jing; Haibao Yu; Zaiqing Nie
>
> **备注:** 25 pages, 7 figures, 5 tables
>
> **摘要:** Personalization, while extensively studied in conventional autonomous driving pipelines, has been largely overlooked in the context of end-to-end autonomous driving (E2EAD), despite its critical role in fostering user trust, safety perception, and real-world adoption. A primary bottleneck is the absence of large-scale real-world datasets that systematically capture driving preferences, severely limiting the development and evaluation of personalized E2EAD models. In this work, we introduce the first large-scale real-world dataset explicitly curated for personalized E2EAD, integrating comprehensive scene topology with rich dynamic context derived from agent dynamics and semantics inferred via a fine-tuned vision-language model (VLM). We propose a hybrid annotation pipeline that combines behavioral analysis, rule-and-distribution-based heuristics, and subjective semantic modeling guided by VLM reasoning, with final refinement through human-in-the-loop verification. Building upon this dataset, we introduce the first standardized benchmark for systematically evaluating personalized E2EAD models. Empirical evaluations on state-of-the-art architectures demonstrate that incorporating personalized driving preferences significantly improves behavioral alignment with human demonstrations.
>
---
