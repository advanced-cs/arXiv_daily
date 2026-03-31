# 计算机视觉 cs.CV

- **最新发布 336 篇**

- **更新 206 篇**

## 最新发布

#### [new 001] Wan-R1: Verifiable-Reinforcement Learning for Video Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视频推理任务，解决视频生成模型在空间推理和多步规划上的不足。通过设计可验证的奖励函数，提升强化学习的效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.27866](https://arxiv.org/pdf/2603.27866)**

> **作者:** Ming Liu; Yunbei Zhang; Shilong Liu; Liwen Wang; Wensheng Zhang
>
> **摘要:** Video generation models produce visually coherent content but struggle with tasks requiring spatial reasoning and multi-step planning. Reinforcement learning (RL) offers a path to improve generalization, but its effectiveness in video reasoning hinges on reward design -- a challenge that has received little systematic study. We investigate this problem by adapting Group Relative Policy Optimization (GRPO) to flow-based video models and training them on maze-solving and robotic navigation tasks. We first show that multimodal reward models fail catastrophically in this setting. To address this, we design verifiable reward functions grounded in objective task metrics. For structured game environments, we introduce a multi-component trajectory reward. For robotic navigation, we propose an embedding-level verifiable reward. Our experiments show that RL fine-tuning with verifiable rewards improves generalization. For example, on complex 3D mazes, our model improves exact match accuracy by 29.1\% over the SFT baseline, and on trap-avoidance tasks by 51.4\%. Our systematic reward analysis reveals that verifiable rewards are critical for stable training, while multimodal reward models could lead to degenerate solutions. These findings establish verifiable reward design as a key enabler for robust video reasoning. Code will be publicly available.
>
---
#### [new 002] ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains
- **分类: cs.CV**

- **简介: 该论文提出ELViS模型，解决跨领域图像相似性计算问题。通过局部描述符和最优传输机制，提升模型泛化能力，适用于多种场景。**

- **链接: [https://arxiv.org/pdf/2603.28603](https://arxiv.org/pdf/2603.28603)**

> **作者:** Pavel Suma; Giorgos Kordopatis-Zilos; Yannis Kalantidis; Giorgos Tolias
>
> **备注:** ICLR 2026
>
> **摘要:** Large-scale instance-level training data is scarce, so models are typically trained on domain-specific datasets. Yet in real-world retrieval, they must handle diverse domains, making generalization to unseen data critical. We introduce ELViS, an image-to-image similarity model that generalizes effectively to unseen domains. Unlike conventional approaches, our model operates in similarity space rather than representation space, promoting cross-domain transfer. It leverages local descriptor correspondences, refines their similarities through an optimal transport step with data-dependent gains that suppress uninformative descriptors, and aggregates strong correspondences via a voting process into an image-level similarity. This design injects strong inductive biases, yielding a simple, efficient, and interpretable model. To assess generalization, we compile a benchmark of eight datasets spanning landmarks, artworks, products, and multi-domain collections, and evaluate ELViS as a re-ranking method. Our experiments show that ELViS outperforms competing methods by a large margin in out-of-domain scenarios and on average, while requiring only a fraction of their computational cost. Code available at: this https URL
>
---
#### [new 003] ForestSim: A Synthetic Benchmark for Intelligent Vehicle Perception in Unstructured Forest Environments
- **分类: cs.CV**

- **简介: 该论文提出ForestSim，一个用于智能车辆在非结构化森林环境中的语义分割合成数据集，解决野外感知数据不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.27923](https://arxiv.org/pdf/2603.27923)**

> **作者:** Pragat Wagle; Zheng Chen; Lantao Liu
>
> **摘要:** Robust scene understanding is essential for intelligent vehicles operating in natural, unstructured environments. While semantic segmentation datasets for structured urban driving are abundant, the datasets for extremely unstructured wild environments remain scarce due to the difficulty and cost of generating pixel-accurate annotations. These limitations hinder the development of perception systems needed for intelligent ground vehicles tasked with forestry automation, agricultural robotics, disaster response, and all-terrain mobility. To address this gap, we present ForestSim, a high-fidelity synthetic dataset designed for training and evaluating semantic segmentation models for intelligent vehicles in forested off-road and no-road environments. ForestSim contains 2094 photorealistic images across 25 diverse environments, covering multiple seasons, terrain types, and foliage densities. Using Unreal Engine environments integrated with Microsoft AirSim, we generate consistent, pixel-accurate labels across 20 classes relevant to autonomous navigation. We benchmark ForestSim using state-of-the-art architectures and report strong performance despite the inherent challenges of unstructured scenes. ForestSim provides a scalable and accessible foundation for perception research supporting the next generation of intelligent off-road vehicles. The dataset and code are publicly available: Dataset: this https URL Code: this https URL
>
---
#### [new 004] HD-VGGT: High-Resolution Visual Geometry Transformer
- **分类: cs.CV**

- **简介: 该论文提出HD-VGGT，解决高分辨率3D重建问题。通过双分支结构和特征调制，提升重建精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.27222](https://arxiv.org/pdf/2603.27222)**

> **作者:** Tianrun Chen; Yuanqi Hu; Yidong Han; Hanjie Xu; Deyi Ji; Qi Zhu; Chunan Yu; Xin Zhang; Cheng Chen; Chaotao Ding; Ying Zang; Xuanfu Li; Jin Ma; Lanyun Zhu
>
> **摘要:** High-resolution imagery is essential for accurate 3D reconstruction, as many geometric details only emerge at fine spatial scales. Recent feed-forward approaches, such as the Visual Geometry Grounded Transformer (VGGT), have demonstrated the ability to infer scene geometry from large collections of images in a single forward pass. However, scaling these models to high-resolution inputs remains challenging: the number of tokens in transformer architectures grows rapidly with both image resolution and the number of views, leading to prohibitive computational and memory costs. Moreover, we observe that visually ambiguous regions, such as repetitive patterns, weak textures, or specular surfaces, often produce unstable feature tokens that degrade geometric inference, especially at higher resolutions. We introduce HD-VGGT, a dual-branch architecture for efficient and robust high-resolution 3D reconstruction. A low-resolution branch predicts a coarse, globally consistent geometry, while a high-resolution branch refines details via a learned feature upsampling module. To handle unstable tokens, we propose Feature Modulation, which suppresses unreliable features early in the transformer. HD-VGGT leverages high-resolution images and supervision without full-resolution transformer costs, achieving state-of-the-art reconstruction quality.
>
---
#### [new 005] Beyond Static Visual Tokens: Structured Sequential Visual Chain-of-Thought Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉推理任务，旨在解决多模态大模型中静态视觉编码与缺乏动态视觉访问的问题。提出SSV-CoT方法，通过结构化顺序视觉思维链实现更有效的视觉认知。**

- **链接: [https://arxiv.org/pdf/2603.26737](https://arxiv.org/pdf/2603.26737)**

> **作者:** Guangfu Guo; Xiaoqian Lu; Yue Feng; Mingming Sun
>
> **摘要:** Current multimodal LLMs encode images as static visual prefixes and rely on text-based reasoning, lacking goal-driven and adaptive visual access. Inspired by human visual perception-where attention is selectively and sequentially shifted from the most informative regions to secondary cues-we propose Structural Sequential Visual CoT SSV-CoT. First, a question-relevant saliency map identifies and organizes key visual regions, explicitly modeling the spatial distribution of visual importance. Second, reasoning is performed following this discriminative order, inducing a curriculum-like semantic progression from primary to secondary cues. This method is trained end-to-end, using text cot and answer supervision, without relying on region-level annotations or specialized external tools. Experiments on diverse visual reasoning benchmarks show gains, validating structured and sequential visual cognition.
>
---
#### [new 006] GUIDED: Granular Understanding via Identification, Detection, and Discrimination for Fine-Grained Open-Vocabulary Object Detection
- **分类: cs.CV**

- **简介: 该论文提出GUIDED框架，解决细粒度开放词汇目标检测中的语义纠缠问题。通过分离定位与识别路径，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.27014](https://arxiv.org/pdf/2603.27014)**

> **作者:** Jiaming Li; Zhijia Liang; Weikai Chen; Lin Ma; Guanbin Li
>
> **备注:** NIPS2025
>
> **摘要:** Fine-grained open-vocabulary object detection (FG-OVD) aims to detect novel object categories described by attribute-rich texts. While existing open-vocabulary detectors show promise at the base-category level, they underperform in fine-grained settings due to the semantic entanglement of subjects and attributes in pretrained vision-language model (VLM) embeddings -- leading to over-representation of attributes, mislocalization, and semantic drift in embedding space. We propose GUIDED, a decomposition framework specifically designed to address the semantic entanglement between subjects and attributes in fine-grained prompts. By separating object localization and fine-grained recognition into distinct pathways, HUIDED aligns each subtask with the module best suited for its respective roles. Specifically, given a fine-grained class name, we first use a language model to extract a coarse-grained subject and its descriptive attributes. Then the detector is guided solely by the subject embedding, ensuring stable localization unaffected by irrelevant or overrepresented attributes. To selectively retain helpful attributes, we introduce an attribute embedding fusion module that incorporates attribute information into detection queries in an attention-based manner. This mitigates over-representation while preserving discriminative power. Finally, a region-level attribute discrimination module compares each detected region against full fine-grained class names using a refined vision-language model with a projection head for improved alignment. Extensive experiments on FG-OVD and 3F-OVD benchmarks show that GUIDED achieves new state-of-the-art results, demonstrating the benefits of disentangled modeling and modular optimization. Our code will be released at this https URL.
>
---
#### [new 007] RAWIC: Bit-Depth Adaptive Lossless Raw Image Compression
- **分类: cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决高比特深度原始图像的无损压缩问题。提出RAWIC框架，通过自适应位深的熵模型实现不同相机原始图像的有效压缩。**

- **链接: [https://arxiv.org/pdf/2603.28105](https://arxiv.org/pdf/2603.28105)**

> **作者:** Chunhang Zheng; Tongda Xu; Mingli Xie; Yan Wang; Dou Li
>
> **备注:** Accepted by ICME 2026
>
> **摘要:** Raw images preserve linear sensor measurements and high bit-depth information crucial for advanced vision tasks and photography applications, yet their storage remains challenging due to large file sizes, varying bit depths, and sensor-dependent characteristics. Existing learned lossless compression methods mainly target 8-bit sRGB images, while raw reconstruction approaches are inherently lossy and rely on camera-specific assumptions. To address these challenges, we introduce RAWIC, a bit-depth-adaptive learned lossless compression framework for Bayer-pattern raw images. We first convert single-channel Bayer data into a four-channel RGGB format and partition it into patches. For each patch, we compute its bit depth and use it as auxiliary input to guide compression. A bit-depth-adaptive entropy model is then designed to estimate patch distributions conditioned on their bit depths. This architecture enables a single model to handle raw images from diverse cameras and bit depths. Experiments show that RAWIC consistently surpasses traditional lossless codecs, achieving an average 7.7% bitrate reduction over JPEG-XL. Our code is available at this https URL.
>
---
#### [new 008] OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决面板感知的上下文图像生成问题。通过引入正交面板相对算子，提升模型在面板内编辑中的表现。**

- **链接: [https://arxiv.org/pdf/2603.27637](https://arxiv.org/pdf/2603.27637)**

> **作者:** Sanghyeon Lee; Minwoo Lee; Euijin Shin; Kangyeol Kim; Seunghwan Choi; Jaegul Choo
>
> **备注:** Accepted to CVPR 2026. 16 pages, 9 figures. Includes Supplementary Material
>
> **摘要:** We introduce a parameter-efficient adaptation method for panel-aware in-context image generation with pre-trained diffusion transformers. The key idea is to compose learnable, panel-specific orthogonal operators onto the backbone's frozen positional encodings. This design provides two desirable properties: (1) isometry, which preserves the geometry of internal features, and (2) same-panel invariance, which maintains the model's pre-trained intra-panel synthesis behavior. Through controlled experiments, we demonstrate that the effectiveness of our adaptation method is not tied to a specific positional encoding design but generalizes across diverse positional encoding regimes. By enabling effective panel-relative conditioning, the proposed method consistently improves in-context image-based instructional editing pipelines, including state-of-the-art approaches.
>
---
#### [new 009] Real-time Appearance-based Gaze Estimation for Open Domains
- **分类: cs.CV**

- **简介: 该论文属于视觉任务中的眼动估计，解决模型在非受限场景下的泛化问题。通过数据增强和多任务学习提升模型性能，实现轻量级实时跟踪。**

- **链接: [https://arxiv.org/pdf/2603.26945](https://arxiv.org/pdf/2603.26945)**

> **作者:** Zhenhao Li; Zheng Liu; Seunghyun Lee; Amin Fadaeinejad; Yuanhao Yu
>
> **摘要:** Appearance-based gaze estimation (AGE) has achieved remarkable performance in constrained settings, yet we reveal a significant generalization gap where existing AGE models often fail in practical, unconstrained scenarios, particularly those involving facial wearables and poor lighting conditions. We attribute this failure to two core factors: limited image diversity and inconsistent label fidelity across different datasets, especially along the pitch axis. To address these, we propose a robust AGE framework that enhances generalization without requiring additional human-annotated data. First, we expand the image manifold via an ensemble of augmentation techniques, including synthesis of eyeglasses, masks, and varied lighting. Second, to mitigate the impact of anisotropic inter-dataset label deviation, we reformulate gaze regression as a multi-task learning problem, incorporating multi-view supervised contrastive (SupCon) learning, discretized label classification, and eye-region segmentation as auxiliary objectives. To rigorously validate our approach, we curate new benchmark datasets designed to evaluate gaze robustness under challenging conditions, a dimension largely overlooked by existing evaluation protocols. Our MobileNet-based lightweight model achieves generalization performance competitive with the state-of-the-art (SOTA) UniGaze-H, while utilizing less than 1\% of its parameters, enabling high-fidelity, real-time gaze tracking on mobile devices.
>
---
#### [new 010] Progressive Prompt-Guided Cross-Modal Reasoning for Referring Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于 referring image segmentation 任务，旨在将语言描述准确对应到图像中的目标对象。提出 PPCR 框架，通过语义与空间推理实现更精确的分割。**

- **链接: [https://arxiv.org/pdf/2603.27993](https://arxiv.org/pdf/2603.27993)**

> **作者:** Jiachen Li; Hongyun Wang; Jinyu Xu; Wenbo Jiang; Yanchun Ma; Yongjian Liu; Qing Xie; Bolong Zheng
>
> **摘要:** Referring image segmentation aims to localize and segment a target object in an image based on a free-form referring expression. The core challenge lies in effectively bridging linguistic descriptions with object-level visual representations, especially when referring expressions involve detailed attributes and complex inter-object relationships. Existing methods either rely on cross-modal alignment or employ Semantic Segmentation Prompts, but they often lack explicit reasoning mechanisms for grounding language descriptions to target regions in the image. To address these limitations, we propose PPCR, a Progressive Prompt-guided Cross-modal Reasoning framework for referring image segmentation. PPCR explicitly structures the reasoning process as a Semantic Understanding-Spatial Grounding-Instance Segmentation pipeline. Specifically, PPCR first employs multimodal large language models (MLLMs) to generate Semantic Segmentation Prompt that capture key semantic cues of the target object. Based on this semantic context, Spatial Segmentation Prompt are further generated to reason about object location and spatial extent, enabling a progressive transition from semantic understanding to spatial grounding. The Semantic and Spatial Segmentation prompts are then jointly integrated into the segmentation module to guide accurate target localization and segmentation. Extensive experiments on standard referring image segmentation benchmarks demonstrate that PPCR consistently outperforms existing methods. The code will be publicly released to facilitate reproducibility.
>
---
#### [new 011] Estimating the Impact of COVID-19 on Travel Demand in Houston Area Using Deep Learning and Satellite Imagery
- **分类: cs.CV; stat.AP**

- **简介: 该论文属于交通需求预测任务，旨在评估新冠疫情对休斯顿地区出行需求的影响。通过深度学习和卫星图像分析，检测不同地点车辆数量变化，提供出行趋势信息。**

- **链接: [https://arxiv.org/pdf/2603.27486](https://arxiv.org/pdf/2603.27486)**

> **作者:** Alekhya Pachika; Lu Gao; Lingguang Song; Pan Lu; Xingju Wang
>
> **摘要:** Considering recent advances in remote sensing satellite systems and computer vision algorithms, many satellite sensing platforms and sensors have been used to monitor the condition and usage of transportation infrastructure systems. The level of details that can be detected increases significantly with the increase of ground sample distance (GSD), which is around 15 cm - 30 cm for high-resolution satellite images. In this study, we analyzed data acquired from high-resolution satellite imagery to provide insights, predictive signals, and trend for travel demand estimation. More specifically, we estimate the impact of COVID-19 in the metropolitan area of Houston using satellite imagery from Google Earth Engine datasets. We developed a car-counting model through Detectron2 and Faster R-CNN to monitor the presence of cars within different locations (i.e., university, shopping mall, community plaza, restaurant, supermarket) before and during the COVID-19. The results show that the number of cars detected at these selected locations reduced on average 30% in 2020 compared with the previous year 2019. The results also show that satellite imagery provides rich information for travel demand and economic activity estimation. Together with advanced computer vision and deep learning algorithms, it can generate reliable and accurate information for transportation agency decision makers.
>
---
#### [new 012] Structural Graph Probing of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的可解释性研究，旨在探索模型内部结构与行为的关系。通过构建神经元相关图，分析模型不同层次的拓扑结构，揭示其对多模态行为的影响。**

- **链接: [https://arxiv.org/pdf/2603.27070](https://arxiv.org/pdf/2603.27070)**

> **作者:** Haoyu He; Yue Zhuo; Yu Zheng; Qi R. Wang
>
> **备注:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Vision-language models (VLMs) achieve strong multimodal performance, yet how computation is organized across populations of neurons remains poorly understood. In this work, we study VLMs through the lens of neural topology, representing each layer as a within-layer correlation graph derived from neuron-neuron co-activations. This view allows us to ask whether population-level structure is behaviorally meaningful, how it changes across modalities and depth, and whether it identifies causally influential internal components under intervention. We show that correlation topology carries recoverable behavioral signal; moreover, cross-modal structure progressively consolidates with depth around a compact set of recurrent hub neurons, whose targeted perturbation substantially alters model output. Neural topology thus emerges as a meaningful intermediate scale for VLM interpretability: richer than local attribution, more tractable than full circuit recovery, and empirically tied to multimodal behavior. Code is publicly available at this https URL.
>
---
#### [new 013] GeoHCC: Local Geometry-Aware Hierarchical Context Compression for 3D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D重建压缩任务，旨在解决3D高斯泼溅存储开销大的问题。提出GeoHCC框架，通过几何感知的锚点剪枝和分层熵编码提升压缩效率与结构保真度。**

- **链接: [https://arxiv.org/pdf/2603.28431](https://arxiv.org/pdf/2603.28431)**

> **作者:** Xuan Deng; Xiandong Meng; Hengyu Man; Qiang Zhu; Tiange Zhang; Debin Zhao; Xiaopeng Fan
>
> **备注:** 10
>
> **摘要:** Although 3D Gaussian Splatting (3DGS) enables high-fidelity real-time rendering, its prohibitive storage overhead severely hinders practical deployment. Recent anchor-based 3DGS compression schemes reduce redundancy through context modeling, yet overlook explicit geometric dependencies, leading to structural degradation and suboptimal rate-distortion performance. In this paper, we propose GeoHCC, a geometry-aware 3DGS compression framework that incorporates inter-anchor geometric correlations into anchor pruning and entropy coding for compact representation. We first introduce Neighborhood-Aware Anchor Pruning (NAAP), which evaluates anchor importance via weighted neighborhood feature aggregation and merges redundant anchors into salient neighbors, yielding a compact yet geometry-consistent anchor set. Building upon this optimized structure, we further develop a hierarchical entropy coding scheme, in which coarse-to-fine priors are exploited through a lightweight Geometry-Guided Convolution (GG-Conv) operator to enable spatially adaptive context modeling and rate-distortion optimization. Extensive experiments demonstrate that GeoHCC effectively resolves the structure preservation bottleneck, maintaining superior geometric integrity and rendering fidelity over state-of-the-art anchor-based approaches.
>
---
#### [new 014] Streamlined Open-Vocabulary Human-Object Interaction Detection
- **分类: cs.CV**

- **简介: 该论文属于开放词汇人-物交互检测任务，解决现有方法在跨模型特征融合中的难题。提出SL-HOI框架，仅使用DINOv3模型，通过优化注意力机制提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.27500](https://arxiv.org/pdf/2603.27500)**

> **作者:** Chang Sun; Dongliang Liao; Changxing Ding
>
> **摘要:** Open-vocabulary human-object interaction (HOI) detection aims to localize and recognize all human-object interactions in an image, including those unseen during training. Existing approaches usually rely on the collaboration between a conventional HOI detector and a Vision-Language Model (VLM) to recognize unseen HOI categories. However, feature fusion in this paradigm is challenging due to significant gaps in cross-model representations. To address this issue, we introduce SL-HOI, a StreamLined open-vocabulary HOI detection framework based solely on the powerful DINOv3 model. Our design leverages the complementary strengths of DINOv3's components: its backbone for fine-grained localization and its text-aligned vision head for open-vocabulary interaction classification. Moreover, to facilitate smooth cross-attention between the interaction queries and the vision head's output, we propose first feeding both the interaction queries and the backbone image tokens into the vision head, effectively bridging their representation gaps. All DINOv3 parameters in our approach are frozen, with only a small number of learnable parameters added, allowing a fast adaptation to the HOI detection task. Extensive experiments show that SL-HOI achieves state-of-the-art performance on both the SWiG-HOI and HICO-DET benchmarks, demonstrating the effectiveness of our streamlined model architecture. Code is available at this https URL.
>
---
#### [new 015] Data Organization Matters in Multimodal Instruction Tuning: A Controlled Study of Capability Trade-offs
- **分类: cs.CV**

- **简介: 该论文研究多模态模型训练中数据组织对能力平衡的影响，旨在优化视觉理解、结构推理和文档识别的协同。通过对比不同数据调度策略，发现课程训练效果最佳。**

- **链接: [https://arxiv.org/pdf/2603.27744](https://arxiv.org/pdf/2603.27744)**

> **作者:** Guowei Tang
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Recent multimodal large language models (MLLMs) perform strongly on general visual understanding, diagram and chart reasoning, and document-centric perception. However, these abilities are learned from heterogeneous supervision sources with very different task structures and learning demands, and the effect of their temporal organization during training remains underexplored. We study whether data organization affects the trade-off among general understanding, structured reasoning, and fine-grained OCR/document understanding in multimodal instruction tuning. To isolate this factor, we use a controlled three-stage training framework in which the backbone, trainable modules, and optimization pipeline are fixed across all runs, and only the temporal arrangement of post-alignment supervision is changed. We compare four strategies: direct mixture, curriculum training, balanced sampling, and reverse curriculum. Experiments on general visual instruction following, diagram reasoning, chart reasoning, scene-text question answering, and document question answering show that data organization is a first-order design variable in multimodal adaptation. Curriculum training gives the best overall trade-off and the strongest structured reasoning performance. Balanced sampling is better for OCR-oriented capability but weakens the broader capability balance. Reverse curriculum performs worst in both final performance and optimization stability. Training-dynamics analysis further suggests that building general understanding and reasoning before introducing OCR-intensive supervision leads to smoother optimization and faster convergence. These findings highlight data scheduling as an explicit design dimension for multimodal model adaptation.
>
---
#### [new 016] Event-Based Method for High-Speed 3D Deformation Measurement under Extreme Illumination Conditions
- **分类: cs.CV**

- **简介: 该论文属于高精度3D变形测量任务，解决极端光照下传统相机失效的问题。采用事件相机阵列，结合校准与坐标变换实现高精度变形测量。**

- **链接: [https://arxiv.org/pdf/2603.28159](https://arxiv.org/pdf/2603.28159)**

> **作者:** Banglei Guan; Yifei Bian; Zibin Liu; Haoyang Li; Xuanyu Bai; Taihang Lei; Bin Li; Yang Shang; Qifeng Yu
>
> **备注:** Exp Mech (2026)
>
> **摘要:** Background: Large engineering structures, such as space launch towers and suspension bridges, are subjected to extreme forces that cause high-speed 3D deformation and compromise safety. These structures typically operate under extreme illumination conditions. Traditional cameras often struggle to handle strong light intensity, leading to overexposure due to their limited dynamic range. Objective: Event cameras have emerged as a compelling alternative to traditional cameras in high dynamic range and low-latency applications. This paper presents an integrated method, from calibration to measurement, using a multi-event camera array for high-speed 3D deformation monitoring of structures in extreme illumination conditions. Methods: Firstly, the proposed method combines the characteristics of the asynchronous event stream and temporal correlation analysis to extract the corresponding marker center point. Subsequently, the method achieves rapid calibration by solving the Kruppa equations in conjunction with a parameter optimization framework. Finally, by employing a unified coordinate transformation and linear intersection, the method enables the measurement of 3D deformation of the target structure. Results: Experiments confirmed that the relative measurement error is below 0.08%. Field experiments under extreme illumination conditions, including self-calibration of a multi-event camera array and 3D deformation measurement, verified the performance of the proposed method. Conclusions: This paper addressed the critical limitation of traditional cameras in measuring high-speed 3D deformations under extreme illumination conditions. The experimental results demonstrate that, compared to other methods, the proposed method can accurately measure 3D deformations of structures under harsh lighting conditions, and the relative error of the measured deformation is less than 0.1%.
>
---
#### [new 017] ConceptWeaver: Weaving Disentangled Concepts with Flow
- **分类: cs.CV**

- **简介: 该论文提出ConceptWeaver，解决流模型中概念解耦与定制问题。通过分析生成过程的三个阶段，实现单图概念分离与编辑。**

- **链接: [https://arxiv.org/pdf/2603.28493](https://arxiv.org/pdf/2603.28493)**

> **作者:** Jintao Chen; Aiming Hao; Xiaoqing Chen; Chengyu Bai; Chubin Chen; Yanxun Li; Jiahong Wu; Xiangxiang Chu; Shanghang Zhang
>
> **摘要:** Pre-trained flow-based models excel at synthesizing complex scenes yet lack a direct mechanism for disentangling and customizing their underlying concepts from one-shot real-world sources. To demystify this process, we first introduce a novel differential probing technique to isolate and analyze the influence of individual concept tokens on the velocity field over time. This investigation yields a critical insight: the generative process is not monolithic but unfolds in three distinct stages. An initial \textbf{Blueprint Stage} establishes low-frequency structure, followed by a pivotal \textbf{Instantiation Stage} where content concepts emerge with peak intensity and become naturally disentangled, creating an optimal window for manipulation. A final concept-insensitive refinement stage then synthesizes fine-grained details. Guided by this discovery, we propose \textbf{ConceptWeaver}, a framework for one-shot concept disentanglement. ConceptWeaver learns concept-specific semantic offsets from a single reference image using a stage-aware optimization strategy that aligns with the three-stage framework. These learned offsets are then deployed during inference via our novel ConceptWeaver Guidance (CWG) mechanism, which strategically injects them at the appropriate generative stage. Extensive experiments validate that ConceptWeaver enables high-fidelity, compositional synthesis and editing, demonstrating that understanding and leveraging the intrinsic, staged nature of flow models is key to unlocking precise, multi-granularity content manipulation.
>
---
#### [new 018] LightMover: Generative Light Movement with Color and Intensity Controls
- **分类: cs.CV; cs.CL; cs.GR; cs.LG**

- **简介: 该论文提出LightMover，解决单图像中可控光照编辑问题。通过视频扩散先验生成物理合理光照变化，实现对光位、颜色、强度的精确控制。**

- **链接: [https://arxiv.org/pdf/2603.27209](https://arxiv.org/pdf/2603.27209)**

> **作者:** Gengze Zhou; Tianyu Wang; Soo Ye Kim; Zhixin Shu; Xin Yu; Yannick Hold-Geoffroy; Sumit Chaturvedi; Qi Wu; Zhe Lin; Scott Cohen
>
> **备注:** CVPR 2026. 10 pages, 5 figures, 6 tables in main paper; supplementary material included
>
> **摘要:** We present LightMover, a framework for controllable light manipulation in single images that leverages video diffusion priors to produce physically plausible illumination changes without re-rendering the scene. We formulate light editing as a sequence-to-sequence prediction problem in visual token space: given an image and light-control tokens, the model adjusts light position, color, and intensity together with resulting reflections, shadows, and falloff from a single view. This unified treatment of spatial (movement) and appearance (color, intensity) controls improves both manipulation and illumination understanding. We further introduce an adaptive token-pruning mechanism that preserves spatially informative tokens while compactly encoding non-spatial attributes, reducing control sequence length by 41% while maintaining editing fidelity. To train our framework, we construct a scalable rendering pipeline that generates large numbers of image pairs across varied light positions, colors, and intensities while keeping the scene content consistent with the original image. LightMover enables precise, independent control over light position, color, and intensity, and achieves high PSNR and strong semantic consistency (DINO, CLIP) across different tasks.
>
---
#### [new 019] LongCat-Next: Lexicalizing Modalities as Discrete Tokens
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态任务，旨在解决现有系统对非语言模态处理不融合的问题。提出DiNA框架和dNaViT模型，实现跨模态的统一离散建模。**

- **链接: [https://arxiv.org/pdf/2603.27538](https://arxiv.org/pdf/2603.27538)**

> **作者:** Meituan LongCat Team; Bin Xiao; Chao Wang; Chengjiang Li; Chi Zhang; Chong Peng; Hang Yu; Hao Yang; Haonan Yan; Haoze Sun; Haozhe Zhao; Hong Liu; Hui Su; Jiaqi Zhang; Jiawei Wang; Jing Li; Kefeng Zhang; Manyuan Zhang; Minhao Jing; Peng Pei; Quan Chen; Taofeng Xue; Tongxin Pan; Xiaotong Li; Xiaoyang Li; Xiaoyu Zhao; Xing Hu; Xinyang Lin; Xunliang Cai; Yan Bai; Yan Feng; Yanjie Li; Yao Qiu; Yerui Sun; Yifan Lu; Ying Luo; Yipeng Mei; Yitian Chen; Yuchen Xie; Yufang Liu; Yufei Chen; Yulei Qian; Yuqi Peng; Zhihang Yu; Zhixiong Han; Changran Wang; Chen Chen; Dian Zheng; Fengjiao Chen; Ge Yang; Haowei Guo; Haozhe Wang; Hongyu Li; Huicheng Jiang; Jiale Hong; Jialv Zou; Jiamu Li; Jianping Lin; Jiaxing Liu; Jie Yang; Jing Jin; Jun Kuang; Juncheng She; Kunming Luo; Kuofeng Gao; Lin Qiu; Linsen Guo; Mianqiu Huang; Qi Li; Qian Wang; Rumei Li; Siyu Ren; Wei Wang; Wenlong He; Xi Chen; Xiao Liu; Xiaoyu Li; Xu Huang; Xuanyu Zhu; Xuezhi Cao; Yaoming Zhu; Yifei Cao; Yimeng Jia; Yizhen Jiang; Yufei Gao; Zeyang Hu; Zhenlong Yuan; Zijian Zhang; Ziwen Wang
>
> **备注:** LongCat-Next Technical Report
>
> **摘要:** The prevailing Next-Token Prediction (NTP) paradigm has driven the success of large language models through discrete autoregressive modeling. However, contemporary multimodal systems remain language-centric, often treating non-linguistic modalities as external attachments, leading to fragmented architectures and suboptimal integration. To transcend this limitation, we introduce Discrete Native Autoregressive (DiNA), a unified framework that represents multimodal information within a shared discrete space, enabling a consistent and principled autoregressive modeling across modalities. A key innovation is the Discrete Native Any-resolution Visual Transformer (dNaViT), which performs tokenization and de-tokenization at arbitrary resolutions, transforming continuous visual signals into hierarchical discrete tokens. Building on this foundation, we develop LongCat-Next, a native multimodal model that processes text, vision, and audio under a single autoregressive objective with minimal modality-specific design. As an industrial-strength foundation model, it excels at seeing, painting, and talking within a single framework, achieving strong performance across a wide range of multimodal benchmarks. In particular, LongCat-Next addresses the long-standing performance ceiling of discrete vision modeling on understanding tasks and provides a unified approach to effectively reconcile the conflict between understanding and generation. As an attempt toward native multimodality, we open-source the LongCat-Next and its tokenizers, hoping to foster further research and development in the community. GitHub: this https URL
>
---
#### [new 020] RetinexDualV2: Physically-Grounded Dual Retinex for Generalized UHD Image Restoration
- **分类: cs.CV**

- **简介: 该论文提出RetinexDualV2，用于UHD图像修复任务，解决复杂退化问题。通过物理引导机制，实现高效反射与光照校正。**

- **链接: [https://arxiv.org/pdf/2603.27979](https://arxiv.org/pdf/2603.27979)**

> **作者:** Mohab Kishawy; Jun Chen
>
> **摘要:** We propose RetinexDualV2, a unified, physically grounded dual-branch framework for diverse Ultra-High-Definition (UHD) image restoration. Unlike generic models, our method employs a Task-Specific Physical Grounding Module (TS-PGM) to extract degradation-aware priors (e.g., rain masks and dark channels). These explicitly guide a Retinex decomposition network via a novel Physical-conditioned Multi-head Self-Attention (PC-MSA) mechanism, enabling robust reflection and illumination correction. This physical conditioning allows a single architecture to handle various complex degradations seamlessly, without task-specific structural modifications. RetinexDualV2 demonstrates exceptional generalizability, securing 4\textsuperscript{th} place in the NTIRE 2026 Day and Night Raindrop Removal Challenge and 5\textsuperscript{th} place in the Joint Noise Low-light Enhancement (JNLLIE) Challenge. Extensive experiments confirm the state-of-the-art performance and efficiency of our physically motivated approach.
>
---
#### [new 021] SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering
- **分类: cs.CV**

- **简介: 该论文提出SGS-Intrinsic，解决稀疏视角下室内逆渲染问题，通过构建语义一致的高斯场实现几何重建与材质光照解耦。**

- **链接: [https://arxiv.org/pdf/2603.27516](https://arxiv.org/pdf/2603.27516)**

> **作者:** Jiahao Niu; Rongjia Zheng; Wenju Xu; WeiShi Zheng; Qing Zhang
>
> **备注:** CVPR2026
>
> **摘要:** We present SGS-Intrinsic, an indoor inverse rendering framework that works well for sparse-view images. Unlike existing 3D Gaussian Splatting (3DGS) based methods that focus on object-centric reconstruction and fail to work under sparse view settings, our method allows to achieve high-quality geometry reconstruction and accurate disentanglement of material and illumination. The core idea is to construct a dense and geometry-consistent Gaussian semantic field guided by semantic and geometric priors, providing a reliable foundation for subsequent inverse rendering. Building upon this, we perform material-illumination disentanglement by combining a hybrid illumination model and material prior to effectively capture illumination-material interactions. To mitigate the impact of cast shadows and enhance the robustness of material recovery, we introduce illumination-invariant material constraint together with a deshadowing model. Extensive experiments on benchmark datasets show that our method consistently improves both reconstruction fidelity and inverse rendering quality over existing 3DGS-based inverse rendering approaches. Our code is available at this https URL.
>
---
#### [new 022] MultiLoc: Multi-view Guided Relative Pose Regression for Fast and Robust Visual Re-Localization
- **分类: cs.CV**

- **简介: 该论文属于视觉重定位任务，解决RPR在未见环境中的性能限制。提出MultiLoc模型，通过多视角融合和几何相关参考视图选择，提升姿态估计的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.27170](https://arxiv.org/pdf/2603.27170)**

> **作者:** Nobel Dang; Bing Li
>
> **摘要:** Relative Pose Regression (RPR) generalizes well to unseen environments, but its performance is often limited due to pairwise and local spatial views. To this end, we propose MultiLoc, a novel multi-view guided RPR model trained at scale, equipping relative pose regression with globally consistent spatial and geometric understanding. Specifically, our method jointly fuses multiple reference views and their associated camera poses in a single forward pass, enabling accurate zero-shot pose estimation with real-time efficiency. To reliably supply informative context, we further propose a co-visibility-driven retrieval strategy for geometrically relevant reference view selection. MultiLoc establishes a new benchmark in visual re-localization, consistently outperforming existing state-of-the-art (SOTA) relative pose regression (RPR) methods across diverse datasets, including WaySpots, Cambridge Landmarks, and Indoor6. Furthermore, MultiLoc's pose regressor exhibits SOTA performance in relative pose estimation, surpassing RPR, feature matching and non-regression-based techniques on the MegaDepth-1500, ScanNet-1500, and ACID benchmarks. These results demonstrate robust domain generalization of MultiLoc across indoor, outdoor and natural environments. Code will be made publicly available.
>
---
#### [new 023] EVA: Bridging Performance and Human Alignment in Hard-Attention Vision Models for Image Classification
- **分类: cs.CV**

- **简介: 该论文提出EVA模型，解决视觉模型性能与人类注意力对齐的平衡问题，通过机制设计提升图像分类中的人类可解释性。**

- **链接: [https://arxiv.org/pdf/2603.27340](https://arxiv.org/pdf/2603.27340)**

> **作者:** Pengcheng Pan; Yonekura Shogo; Kuniyoshi Yasuo
>
> **摘要:** Optimizing vision models purely for classification accuracy can impose an alignment tax, degrading human-like scanpaths and limiting interpretability. We introduce EVA, a neuroscience-inspired hard-attention mechanistic testbed that makes the performance-human-likeness trade-off explicit and adjustable. EVA samples a small number of sequential glimpses using a minimal fovea-periphery representation with CNN-based feature extractor and integrates variance control and adaptive gating to stabilize and regulate attention dynamics. EVA is trained with the standard classification objective without gaze supervision. On CIFAR-10 with dense human gaze annotations, EVA improves scanpath alignment under established metrics such as DTW, NSS, while maintaining competitive accuracy. Ablations show that CNN-based feature extraction drives accuracy but suppresses human-likeness, whereas variance control and gating restore human-aligned trajectories with minimal performance loss. We further validate EVA's scalability on ImageNet-100 and evaluate scanpath alignment on COCO-Search18 without COCO-Search18 gaze supervision or finetuning, where EVA yields human-like scanpaths on natural scenes without additional training. Overall, EVA provides a principled framework for trustworthy, human-interpretable active vision.
>
---
#### [new 024] Multimodal Deep Learning for Diabetic Foot Ulcer Staging Using Integrated RGB and Thermal Imaging
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于糖尿病足溃疡分期任务，旨在通过多模态图像提升分类准确性。研究结合RGB与热成像数据，构建多通道模型，提升诊断效果。**

- **链接: [https://arxiv.org/pdf/2603.26952](https://arxiv.org/pdf/2603.26952)**

> **作者:** Gulengul Mermer; Mustafa Furkan Aksu; Gozde Ozsezer; Sevki Cetinkalp; Orhan Er; Mehmet Kemal Gullu
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Diabetic foot ulcers (DFU) are one of the serious complications of diabetes that can lead to amputations and high healthcare costs. Regular monitoring and early diagnosis are critical for reducing the clinical burden and the risk of amputation. The aim of this study is to investigate the impact of using multimodal images on deep learning models for the classification of DFU stages. To this end, we developed a Raspberry Pi-based portable imaging system capable of simultaneously capturing RGB and thermal images. Using this prototype, a dataset consisting of 1,205 samples was collected in a hospital setting. The dataset was labeled by experts into six distinct stages. To evaluate the models performance, we prepared three different training sets: RGB-only, thermal-only, and RGB+Thermal (with the thermal image added as a fourth channel). We trained these training sets on the DenseNet121, EfficientNetV2, InceptionV3, ResNet50, and VGG16 models. The results show that the multimodal training dataset, in which RGB and thermal data are combined across four channels, outperforms single-modal approaches. The highest performance was observed in the VGG16 model trained on the RGB+Thermal dataset. The model achieved an accuracy of 93.25%, an F1-score of 92.53%, and an MCC of 91.03%. Grad-CAM heatmap visualizations demonstrated that the thermal channel helped the model focus on the correct location by highlighting temperature anomalies in the ulcer region, while the RGB channel supported the decision-making process with complementary structural and textural information.
>
---
#### [new 025] CNMBI: Determining the Number of Clusters Using Center Pairwise Matching and Boundary Filtering
- **分类: cs.CV**

- **简介: 该论文属于聚类任务，旨在解决无先验信息下确定最佳聚类数的问题。提出CNMBI方法，通过中心配对和边界过滤，提高聚类数量判断的准确性与灵活性。**

- **链接: [https://arxiv.org/pdf/2603.26744](https://arxiv.org/pdf/2603.26744)**

> **作者:** Ruilin Zhang; Haiyang Zheng; Hongpeng Wang
>
> **摘要:** One of the main challenges in data mining is choosing the optimal number of clusters without prior information. Notably, existing methods are usually in the philosophy of cluster validation and hence have underlying assumptions on data distribution, which prevents their application to complex data such as large-scale images and high-dimensional data from the real world. In this regard, we propose an approach named CNMBI. Leveraging the distribution information inherent in the data space, we map the target task as a dynamic comparison process between cluster centers regarding positional behavior, without relying on the complete clustering results and designing the complex validity index as before. Bipartite graph theory is then employed to efficiently model this process. Additionally, we find that different samples have different confidence levels and thereby actively remove low-confidence ones, which is, for the first time to our knowledge, considered in cluster number determination. CNMBI is robust and allows for more flexibility in the dimension and shape of the target data (e.g., CIFAR-10 and STL-10). Extensive comparison studies with state-of-the-art competitors on various challenging datasets demonstrate the superiority of our method.
>
---
#### [new 026] On-the-fly Repulsion in the Contextual Space for Rich Diversity in Diffusion Transformers
- **分类: cs.CV; cs.AI; cs.GR; cs.LG**

- **简介: 该论文属于文本到图像生成任务，旨在解决生成结果多样性不足的问题。通过在上下文空间中引入实时排斥机制，提升生成多样性且不牺牲质量。**

- **链接: [https://arxiv.org/pdf/2603.28762](https://arxiv.org/pdf/2603.28762)**

> **作者:** Omer Dahary; Benaya Koren; Daniel Garibi; Daniel Cohen-Or
>
> **备注:** Conditionally accepted to SIGGRAPH 2026. Project page: this https URL
>
> **摘要:** Modern Text-to-Image (T2I) diffusion models have achieved remarkable semantic alignment, yet they often suffer from a significant lack of variety, converging on a narrow set of visual solutions for any given prompt. This typicality bias presents a challenge for creative applications that require a wide range of generative outcomes. We identify a fundamental trade-off in current approaches to diversity: modifying model inputs requires costly optimization to incorporate feedback from the generative path. In contrast, acting on spatially-committed intermediate latents tends to disrupt the forming visual structure, leading to artifacts. In this work, we propose to apply repulsion in the Contextual Space as a novel framework for achieving rich diversity in Diffusion Transformers. By intervening in the multimodal attention channels, we apply on-the-fly repulsion during the transformer's forward pass, injecting the intervention between blocks where text conditioning is enriched with emergent image structure. This allows for redirecting the guidance trajectory after it is structurally informed but before the composition is fixed. Our results demonstrate that repulsion in the Contextual Space produces significantly richer diversity without sacrificing visual fidelity or semantic adherence. Furthermore, our method is uniquely efficient, imposing a small computational overhead while remaining effective even in modern "Turbo" and distilled models where traditional trajectory-based interventions typically fail.
>
---
#### [new 027] The Language of Touch: Translating Vibrations into Text with Dual-Branch Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于 vibrotactile captioning 任务，旨在将触觉振动信号转化为文本描述。针对振动数据的周期性与非周期性结构及缺乏空间语义的问题，提出 ViPAC 方法，通过双分支策略和动态融合机制提升生成效果。**

- **链接: [https://arxiv.org/pdf/2603.26804](https://arxiv.org/pdf/2603.26804)**

> **作者:** Jin Chen; Yifeng Lin; Chao Zeng; Si Wu; Tiesong Zhao
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** The standardization of vibrotactile data by IEEE P1918.1 workgroup has greatly advanced its applications in virtual reality, human-computer interaction and embodied artificial intelligence. Despite these efforts, the semantic interpretation and understanding of vibrotactile signals remain an unresolved challenge. In this paper, we make the first attempt to address vibrotactile captioning, {\it i.e.}, generating natural language descriptions from vibrotactile signals. We propose Vibrotactile Periodic-Aperiodic Captioning (ViPAC), a method designed to handle the intrinsic properties of vibrotactile data, including hybrid periodic-aperiodic structures and the lack of spatial semantics. Specifically, ViPAC employs a dual-branch strategy to disentangle periodic and aperiodic components, combined with a dynamic fusion mechanism that adaptively integrates signal features. It also introduces an orthogonality constraint and weighting regularization to ensure feature complementarity and fusion consistency. Additionally, we construct LMT108-CAP, the first vibrotactile-text paired dataset, using GPT-4o to generate five constrained captions per surface image from the popular LMT-108 dataset. Experiments show that ViPAC significantly outperforms the baseline methods adapted from audio and image captioning, achieving superior lexical fidelity and semantic alignment.
>
---
#### [new 028] TokenDial: Continuous Attribute Control in Text-to-Video via Spatiotemporal Token Offsets
- **分类: cs.CV**

- **简介: 该论文提出TokenDial，用于文本到视频生成中的连续属性控制。解决预训练模型对属性变化程度控制不足的问题，通过引入时空token偏移实现更精确的编辑。**

- **链接: [https://arxiv.org/pdf/2603.27520](https://arxiv.org/pdf/2603.27520)**

> **作者:** Zhixuan Liu; Peter Schaldenbrand; Yijun Li; Long Mai; Aniruddha Mahapatra; Cusuh Ham; Jean Oh; Jui-Hsien Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** We present TokenDial, a framework for continuous, slider-style attribute control in pretrained text-to-video generation models. While modern generators produce strong holistic videos, they offer limited control over how much an attribute changes (e.g., effect intensity or motion magnitude) without drifting identity, background, or temporal coherence. TokenDial is built on the observation: additive offsets in the intermediate spatiotemporal visual patch-token space form a semantic control direction, where adjusting the offset magnitude yields coherent, predictable edits for both appearance and motion dynamics. We learn attribute-specific token offsets without retraining the backbone, using pretrained understanding signals: semantic direction matching for appearance and motion-magnitude scaling for motion. We demonstrate TokenDial's effectiveness on diverse attributes and prompts, achieving stronger controllability and higher-quality edits than state-of-the-art baselines, supported by extensive quantitative evaluation and human studies.
>
---
#### [new 029] Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于安全增强任务，旨在解决LVLMs中不安全通道的问题。通过因果分析和双模态投影方法，提升模型安全性同时保持性能。**

- **链接: [https://arxiv.org/pdf/2603.27240](https://arxiv.org/pdf/2603.27240)**

> **作者:** Jinhu Fu; Yihang Lou; Qingyi Si; Shudong Zhang; Yan Bai; Sen Su
>
> **备注:** Accepted by CVPR 2026 main conference
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved impressive performance across multimodal understanding and reasoning tasks, yet their internal safety mechanisms remain opaque and poorly controlled. In this work, we present a comprehensive framework for diagnosing and repairing unsafe channels within LVLMs (CARE). We first perform causal mediation analysis to identify neurons and layers that are causally responsible for unsafe behaviors. Based on these findings, we introduce a dual-modal safety subspace projection method that learns generalized safety subspaces for both visual and textual modalities through generalized eigen-decomposition between benign and malicious activations. During inference, activations are dynamically projected toward these safety subspaces via a hybrid fusion mechanism that adaptively balances visual and textual corrections, effectively suppressing unsafe features while preserving semantic fidelity. Extensive experiments on multiple safety benchmarks demonstrate that our causal-subspace repair framework significantly enhances safety robustness without degrading general multimodal capabilities, outperforming prior activation steering and alignment-based baselines. Additionally, our method exhibits good transferability, defending against unseen attacks.
>
---
#### [new 030] RatSeizure: A Benchmark and Saliency-Context Transformer for Rat Seizure Localization
- **分类: cs.CV**

- **简介: 该论文属于癫痫行为分析任务，解决动物模型 seizure 定位问题。构建了首个公开数据集 RatSeizure，并提出 RaSeformer 模型提升定位性能。**

- **链接: [https://arxiv.org/pdf/2603.26780](https://arxiv.org/pdf/2603.26780)**

> **作者:** Ting Yu Tsai; An Yu; Lucy Lee; Felix X.-F. Ye; Damian S. Shin; Tzu-Jen Kao; Xin Li; Ming-Ching Chang
>
> **摘要:** Animal models, particularly rats, play a critical role in seizure research for studying epileptogenesis and treatment response. However, progress is limited by the lack of datasets with precise temporal annotations and standardized evaluation protocols. Existing animal behavior datasets often have limited accessibility, coarse labeling, and insufficient temporal localization of clinically meaningful events. To address these limitations, we introduce RatSeizure, the first publicly benchmark for fine-grained seizure behavior analysis. The dataset consists of recorded clips annotated with seizure-related action units and temporal boundaries, enabling both behavior classification and temporal localization. We further propose RaSeformer, a saliency-context Transformer for temporal action localization that highlights behavior-relevant context while suppressing redundant cues. Experiments on RatSeizure show that RaSeformer achieves strong performance and provides a competitive reference model for this challenging task. We also establish standardized dataset splits and evaluation protocols to support reproducible benchmarking.
>
---
#### [new 031] Industrial3D: A Terrestrial LiDAR Point Cloud Dataset and CrossParadigm Benchmark for Industrial Infrastructure
- **分类: cs.CV**

- **简介: 该论文提出Industrial3D数据集和跨范式基准，解决工业点云语义理解难题，针对MEP设施的几何模糊和类别不平衡问题进行研究。**

- **链接: [https://arxiv.org/pdf/2603.28660](https://arxiv.org/pdf/2603.28660)**

> **作者:** Chao Yin; Hongzhe Yue; Qing Han; Difeng Hu; Zhenyu Liang; Fangzhou Lin; Bing Sun; Boyu Wang; Mingkai Li; Wei Yao; Jack C.P. Cheng
>
> **备注:** 49 pages, 8 figure, 14 tables
>
> **摘要:** Automated semantic understanding of dense point clouds is a prerequisite for Scan-to-BIM pipelines, digital twin construction, and as-built verification--core tasks in the digital transformation of the construction industry. Yet for industrial mechanical, electrical, and plumbing (MEP) facilities, this challenge remains largely unsolved: TLS acquisitions of water treatment plants, chiller halls, and pumping stations exhibit extreme geometric ambiguity, severe occlusion, and extreme class imbalance that architectural benchmarks (e.g., S3DIS or ScanNet) cannot adequately represent. We present Industrial3D, a terrestrial LiDAR dataset comprising 612 million expertly labelled points at 6 mm resolution from 13 water treatment facilities. At 6.6x the scale of the closest comparable MEP dataset, Industrial3D provides the largest and most demanding testbed for industrial 3D scene understanding to date. We further establish the first industrial cross-paradigm benchmark, evaluating nine representative methods across fully supervised, weakly supervised, unsupervised, and foundation model settings under a unified benchmark protocol. The best supervised method achieves 55.74% mIoU, whereas zero-shot Point-SAM reaches only 15.79%--a 39.95 percentage-point gap that quantifies the unresolved domain-transfer challenge for industrial TLS data. Systematic analysis reveals that this gap originates from a dual crisis: statistical rarity (215:1 imbalance, 3.5x more severe than S3DIS) and geometric ambiguity (tail-class points share cylindrical primitives with head-class pipes) that frequency-based re-weighting alone cannot resolve. Industrial3D, along with benchmark code and pre-trained models, will be publicly available at this https URL.
>
---
#### [new 032] Towards Domain-Generalized Open-Vocabulary Object Detection: A Progressive Domain-invariant Cross-modal Alignment Method
- **分类: cs.CV**

- **简介: 该论文属于开放词汇目标检测任务，旨在解决领域泛化问题。针对领域分布变化导致的跨模态对齐失效，提出PICA方法，通过渐进式对齐提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27556](https://arxiv.org/pdf/2603.27556)**

> **作者:** Xiaoran Xu; Xiaoshan Yang; Jiangang Yang; Yifan Xu; Jian Liu; Changsheng Xu
>
> **摘要:** Open-Vocabulary Object Detection (OVOD) has achieved remarkable success in generalizing to novel categories. However, this success often rests on the implicit assumption of domain stationarity. In this work, we provide a principled revisit of the OVOD paradigm, uncovering a fundamental vulnerability: the fragile coupling between visual manifolds and textual embeddings when distribution shifts occur. We first systematically formalize Domain-Generalized Open-Vocabulary Object Detection (DG-OVOD). Through empirical analysis, we demonstrate that visual shifts do not merely add noise; they cause a collapse of the latent cross-modal space where novel category visual signals detach from their semantic anchors. Motivated by these insights, we propose Progressive Domain-invariant Cross-modal Alignment (PICA). PICA departs from uniform training by introducing a multi-level ambiguity and signal strength curriculum. It builds adaptive pseudo-word prototypes, refined via sample confidence and visual consistency, to enforce invariant cross-domain modality alignment. Our findings suggest that OVOD's robustness to domain shifts is intrinsically linked to the stability of the latent cross-modal alignment space. Our work provides both a challenging benchmark and a new perspective on building truly generalizable open-vocabulary systems that extend beyond static laboratory conditions.
>
---
#### [new 033] Physics-Aware Diffusion for LiDAR Point Cloud Densification
- **分类: cs.CV**

- **简介: 该论文属于点云补全任务，解决LiDAR距离远导致的点云稀疏问题。通过物理感知扩散模型，提升点云密度与质量。**

- **链接: [https://arxiv.org/pdf/2603.26759](https://arxiv.org/pdf/2603.26759)**

> **作者:** Zeping Zhang; Robert Laganière
>
> **摘要:** LiDAR perception is severely limited by the distance-dependent sparsity of distant objects. While diffusion models can recover dense geometry, they suffer from prohibitive latency and physical hallucinations manifesting as ghost points. We propose Scanline-Consistent Range-Aware Diffusion, a framework that treats densification as probabilistic refinement rather than generation. By leveraging Partial Diffusion (SDEdit) on a coarse prior, we achieve high-fidelity results in just 156ms. Our novel Ray-Consistency loss and Negative Ray Augmentation enforce sensor physics to suppress artifacts. Our method achieves state-of-the-art results on KITTI-360 and nuScenes, directly boosting off-the-shelf 3D detectors without retraining. Code will be made available.
>
---
#### [new 034] RAP: Retrieve, Adapt, and Prompt-Fit for Training-Free Few-Shot Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，解决少样本分割中依赖标注且忽略解剖结构的问题。提出RAP框架，通过检索、适配和提示生成实现无需训练的高效分割。**

- **链接: [https://arxiv.org/pdf/2603.27705](https://arxiv.org/pdf/2603.27705)**

> **作者:** Zhihao Mao; Bangpu Chen
>
> **备注:** This paper has been accepted by IJCNN 2026
>
> **摘要:** Few-shot medical image segmentation (FSMIS) has achieved notable progress, yet most existing methods mainly rely on semantic correspondences from scarce annotations while under-utilizing a key property of medical imagery: anatomical targets exhibit repeatable high-frequency morphology (e.g., boundary geometry and spatial layout) across patients and acquisitions. We propose RAP, a training-free framework that retrieves, adapts, and prompts Segment Anything Model 2 (SAM2) for FSMIS. First, RAP retrieves morphologically compatible supports from an archive using DINOv3 features to reduce brittleness in single-support choice. Second, it adapts the retrieved support mask to the query by fitting boundary-aware structural cues, yielding an anatomy-consistent pre-mask under domain shifts. Third, RAP converts the pre-mask into prompts by sampling positive points via Voronoi partitioning and negative points via sector-based sampling, and feeds them into SAM2 for final refinement without any fine-tuning. Extensive experiments on multiple medical segmentation benchmarks show that RAP consistently surpasses prior FSMIS baselines and achieves state-of-the-art performance. Overall, RAP demonstrates that explicit structural fitting combined with retrieval-augmented prompting offers a simple and effective route to robust training-free few-shot medical segmentation.
>
---
#### [new 035] LiDAR for Crowd Management: Applications, Benefits, and Future Directions
- **分类: cs.CV**

- **简介: 本文探讨LiDAR在人群管理中的应用，解决传统监控技术的不足。论文分析了人群检测、计数、跟踪与行为分类任务，提出LiDAR的优势及研究挑战。**

- **链接: [https://arxiv.org/pdf/2603.27663](https://arxiv.org/pdf/2603.27663)**

> **作者:** Abdullah Khanfor; Chaima Zaghouani; Hakim Ghazzai; Ahmad Alsharoa; Gianluca Setti
>
> **备注:** 8 pages, 5 figures, 1 table
>
> **摘要:** Light Detection and Ranging (LiDAR) technology offers significant advantages for effective crowd management. This article presents LiDAR technology and highlights its primary advantages over other monitoring technologies, including enhanced privacy, performance in various weather conditions, and precise 3D mapping. We present a general taxonomy of four key tasks in crowd management: crowd detection, counting, tracking, and behavior classification, with illustrative examples of LiDAR applications for each task. We identify challenges and open research directions, including the scarcity of dedicated datasets, sensor fusion requirements, artificial intelligence integration, and processing needs for LiDAR point clouds. This article offers actionable insights for developing crowd management solutions tailored to public safety applications.
>
---
#### [new 036] Sim-to-Real Fruit Detection Using Synthetic Data: Quantitative Evaluation and Embedded Deployment with Isaac Sim
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，研究合成数据在模拟到真实场景中的有效性，解决数据不足和部署限制问题。通过对比不同训练策略，验证了合成与真实数据结合的优越性。**

- **链接: [https://arxiv.org/pdf/2603.28670](https://arxiv.org/pdf/2603.28670)**

> **作者:** Martina Hutter-Mironovova
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** This study investigates the effectiveness of synthetic data for sim-to-real transfer in object detection under constrained data conditions and embedded deployment requirements. Synthetic datasets were generated in NVIDIA Isaac Sim and combined with limited real-world fruit images to train YOLO-based detection models under real-only, synthetic-only, and hybrid regimes. Performance was evaluated on two test datasets: an in-domain dataset with conditions matching the training data and a domain shift dataset containing real fruit and different background conditions. Results show that models trained exclusively on real data achieve the highest accuracy, while synthetic-only models exhibit reduced performance due to a domain gap. Hybrid training strategies significantly improve performance compared to synthetic-only approaches and achieve results close to real-only training while reducing the need for manual annotation. Under domain shift conditions, all models show performance degradation, with hybrid models providing improved robustness. The trained models were successfully deployed on a Jetson Orin NX using TensorRT optimization, achieving real-time inference performance. The findings highlight that synthetic data is most effective when used in combination with real data and that deployment constraints must be considered alongside detection accuracy.
>
---
#### [new 037] Test-Time Instance-Specific Parameter Composition: A New Paradigm for Adaptive Generative Modeling
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Composer，解决生成模型静态参数无法适应输入的问题，通过测试时实例特定参数组合实现动态适配，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.27665](https://arxiv.org/pdf/2603.27665)**

> **作者:** Minh-Tuan Tran; Xuan-May Le; Quan Hung Tran; Mehrtash Harandi; Dinh Phung; Trung Le
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Existing generative models, such as diffusion and auto-regressive networks, are inherently static, relying on a fixed set of pretrained parameters to handle all inputs. In contrast, humans flexibly adapt their internal generative representations to each perceptual or imaginative context. Inspired by this capability, we introduce Composer, a new paradigm for adaptive generative modeling based on test-time instance-specific parameter composition. Composer generates input-conditioned parameter adaptations at inference time, which are injected into the pretrained model's weights, enabling per-input specialization without fine-tuning or retraining. Adaptation occurs once prior to multi-step generation, yielding higher-quality, context-aware outputs with minimal computational and memory overhead. Experiments show that Composer substantially improves performance across diverse generative models and use cases, including lightweight/quantized models and test-time scaling. By leveraging input-aware parameter composition, Composer establishes a new paradigm for designing generative models that dynamically adapt to each input, moving beyond static parameterization.
>
---
#### [new 038] SonoWorld: From One Image to a 3D Audio-Visual Scene
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文提出Image2AVScene任务，旨在从单张图像生成3D音视频场景。工作包括生成全景图、构建3D场景、放置声音锚点并渲染空间音频，实现音视频同步的沉浸式体验。**

- **链接: [https://arxiv.org/pdf/2603.28757](https://arxiv.org/pdf/2603.28757)**

> **作者:** Derong Jin; Xiyi Chen; Ming C. Lin; Ruohan Gao
>
> **备注:** Accepted by CVPR 2026, project page: this https URL
>
> **摘要:** Tremendous progress in visual scene generation now turns a single image into an explorable 3D world, yet immersion remains incomplete without sound. We introduce Image2AVScene, the task of generating a 3D audio-visual scene from a single image, and present SonoWorld, the first framework to tackle this challenge. From one image, our pipeline outpaints a 360° panorama, lifts it into a navigable 3D scene, places language-guided sound anchors, and renders ambisonics for point, areal, and ambient sources, yielding spatial audio aligned with scene geometry and semantics. Quantitative evaluations on a newly curated real-world dataset and a controlled user study confirm the effectiveness of our approach. Beyond free-viewpoint audio-visual rendering, we also demonstrate applications to one-shot acoustic learning and audio-visual spatial source separation. Project website: this https URL
>
---
#### [new 039] Understanding Semantic Perturbations on In-Processing Generative Image Watermarks
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像水印任务，旨在解决生成模型水印在语义篡改下的鲁棒性问题。通过构建框架测试水印在语义编辑下的表现，发现现有方法在语义变化下易失效。**

- **链接: [https://arxiv.org/pdf/2603.27513](https://arxiv.org/pdf/2603.27513)**

> **作者:** Anirudh Nakra; Min Wu
>
> **摘要:** The widespread deployment of high-fidelity generative models has intensified the need for reliable mechanisms for provenance and content authentication. In-processing watermarking, embedding a signature into the generative model's synthesis procedure, has been advocated as a solution and is often reported to be robust to standard post-processing (such as geometric transforms and filtering). Yet robustness to semantic manipulations that alter high-level scene content while maintaining reasonable visual quality is not well studied or understood. We introduce a simple, multi-stage framework for systematically stress-testing in-processing generative watermarks under semantic drift. The framework utilizes off-the-shelf models for object detection, mask generation, and semantically guided inpainting or regeneration to produce controlled, meaning-altering edits with minimal perceptual degradation. Based on extensive experiments on representative schemes, we find that robustness varies significantly with the degree of semantic entanglement: methods by which watermarks remain detectable under a broad suite of conventional perturbations can fail under semantic edits, with watermark detectability in many cases dropping to near zero while image quality remains high. Overall, our results reveal a critical gap in current watermarking evaluations and suggest that watermark designs and benchmarking must explicitly account for robustness against semantic manipulation.
>
---
#### [new 040] SVGS: Single-View to 3D Object Editing via Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D场景编辑任务，解决单视图文本驱动编辑的效率与一致性问题。提出SVGS方法，基于3D高斯泼溅，提升编辑速度与效果。**

- **链接: [https://arxiv.org/pdf/2603.28126](https://arxiv.org/pdf/2603.28126)**

> **作者:** Pengcheng Xue; Yan Tian; Qiutao Song; Ziyi Wang; Linyang He; Weiping Ding; Mahmoud Hassaballah; Karen Egiazarian; Wei-Fa Yang; Leszek Rutkowski
>
> **摘要:** Text-driven 3D scene editing has attracted considerable interest due to its convenience and user-friendliness. However, methods that rely on implicit 3D representations, such as Neural Radiance Fields (NeRF), while effective in rendering complex scenes, are hindered by slow processing speeds and limited control over specific regions of the scene. Moreover, existing approaches, including Instruct-NeRF2NeRF and GaussianEditor, which utilize multi-view editing strategies, frequently produce inconsistent results across different views when executing text instructions. This inconsistency can adversely affect the overall performance of the model, complicating the task of balancing the consistency of editing results with editing efficiency. To address these challenges, we propose a novel method termed Single-View to 3D Object Editing via Gaussian Splatting (SVGS), which is a single-view text-driven editing technique based on 3D Gaussian Splatting (3DGS). Specifically, in response to text instructions, we introduce a single-view editing strategy grounded in multi-view diffusion models, which reconstructs 3D scenes by leveraging only those views that yield consistent editing results. Additionally, we employ sparse 3D Gaussian Splatting as the 3D representation, which significantly enhances editing efficiency. We conducted a comparative analysis of SVGS against existing baseline methods across various scene settings, and the results indicate that SVGS outperforms its counterparts in both editing capability and processing speed, representing a significant advancement in 3D editing technology. For further details, please visit our project page at: this https URL.
>
---
#### [new 041] From 3D Pose to Prose: Biomechanics-Grounded Vision--Language Coaching
- **分类: cs.CV**

- **简介: 该论文提出BioCoach，用于健身指导的视觉-语言框架，解决视频中动作分析与反馈的问题。通过融合视觉和3D骨骼信息，生成精准文本反馈。**

- **链接: [https://arxiv.org/pdf/2603.26938](https://arxiv.org/pdf/2603.26938)**

> **作者:** Yuyang Ji; Yixuan Shen; Shengjie Zhu; Yu Kong; Feng Liu
>
> **摘要:** We present BioCoach, a biomechanics-grounded vision--language framework for fitness coaching from streaming video. BioCoach fuses visual appearance and 3D skeletal kinematics, through a novel three-stage pipeline: an exercise-specific degree-of-freedom selector that focuses analysis on salient joints; a structured biomechanical context that pairs individualized morphometrics with cycle and constraint analysis; and a vision--biomechanics conditioned feedback module that applies cross-attention to generate precise, actionable text. Using parameter-efficient training that freezes the vision and language backbones, BioCoach yields transparent, personalized reasoning rather than pattern matching. To enable learning and fair evaluation, we augment QEVD-fit-coach with biomechanics-oriented feedback to create QEVD-bio-fit-coach, and we introduce a biomechanics-aware LLM judge metric. BioCoach delivers clear gains on QEVD-bio-fit-coach across lexical and judgment metrics while maintaining temporal triggering; on the original QEVD-fit-coach, it improves text quality and correctness with near-parity timing, demonstrating that explicit kinematics and constraints are key to accurate, phase-aware coaching.
>
---
#### [new 042] HandX: Scaling Bimanual Motion and Interaction Generation
- **分类: cs.CV**

- **简介: 该论文提出HandX，解决手部运动与双臂交互生成问题，通过构建高质量数据集和标注方法，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2603.28766](https://arxiv.org/pdf/2603.28766)**

> **作者:** Zimu Zhang; Yucheng Zhang; Xiyan Xu; Ziyin Wang; Sirui Xu; Kai Zhou; Bing Zhou; Chuan Guo; Jian Wang; Yu-Xiong Wang; Liang-Yan Gui
>
> **备注:** CVPR 2026. Project Page: this https URL. Code: this https URL
>
> **摘要:** Synthesizing human motion has advanced rapidly, yet realistic hand motion and bimanual interaction remain underexplored. Whole-body models often miss the fine-grained cues that drive dexterous behavior, finger articulation, contact timing, and inter-hand coordination, and existing resources lack high-fidelity bimanual sequences that capture nuanced finger dynamics and collaboration. To fill this gap, we present HandX, a unified foundation spanning data, annotation, and evaluation. We consolidate and filter existing datasets for quality, and collect a new motion-capture dataset targeting underrepresented bimanual interactions with detailed finger dynamics. For scalable annotation, we introduce a decoupled strategy that extracts representative motion features, e.g., contact events and finger flexion, and then leverages reasoning from large language models to produce fine-grained, semantically rich descriptions aligned with these features. Building on the resulting data and annotations, we benchmark diffusion and autoregressive models with versatile conditioning modes. Experiments demonstrate high-quality dexterous motion generation, supported by our newly proposed hand-focused metrics. We further observe clear scaling trends: larger models trained on larger, higher-quality datasets produce more semantically coherent bimanual motion. Our dataset is released to support future research.
>
---
#### [new 043] MEDIC-AD: Towards Medical Vision-Language Model's Clinical Intelligence
- **分类: cs.CV**

- **简介: 该论文属于医学视觉-语言模型任务，旨在提升病变检测、症状跟踪和可视化解释能力。通过分阶段框架增强模型的临床智能，实现更准确的医疗分析与解释。**

- **链接: [https://arxiv.org/pdf/2603.27176](https://arxiv.org/pdf/2603.27176)**

> **作者:** Woohyeon Park; Jaeik Kim; Sunghwan Steve Cho; Pa Hong; Wookyoung Jeong; Yoojin Nam; Namjoon Kim; Ginny Y. Wong; Ka Chun Cheung; Jaeyoung Do
>
> **摘要:** Lesion detection, symptom tracking, and visual explainability are central to real-world medical image analysis, yet current medical Vision-Language Models (VLMs) still lack mechanisms that translate their broad knowledge into clinically actionable outputs. To bridge this gap, we present MEDIC-AD, a clinically oriented VLM that strengthens these three capabilities through a stage-wise framework. First, learnable anomaly-aware tokens (<Ano>) encourage the model to focus on abnormal regions and build more discriminative lesion centered representations. Second, inter image difference tokens (<Diff>) explicitly encode temporal changes between studies, allowing the model to distinguish worsening, improvement, and stability in disease burden. Finally, a dedicated explainability stage trains the model to generate heatmaps that highlight lesion-related regions, offering clear visual evidence that is consistent with the model's reasoning. Through our staged design, MEDIC-AD steadily boosts performance across anomaly detection, symptom tracking, and anomaly segmentation, achieving state-of-the-art results compared with both closed source and medical-specialized baselines. Evaluations on real longitudinal clinical data collected from real hospital workflows further show that MEDIC-AD delivers stable predictions and clinically faithful explanations in practical patient-monitoring and decision-support workflows
>
---
#### [new 044] Ordinal Semantic Segmentation Applied to Medical and Odontological Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于语义分割任务，旨在解决类别序关系被忽略的问题。通过引入包含序关系的损失函数，提升分割的语义一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.26736](https://arxiv.org/pdf/2603.26736)**

> **作者:** Mariana Dória Prata Lima; Gilson Antonio Giraldi; Jaime S. Cardoso
>
> **备注:** 23 pages, 1 figure
>
> **摘要:** Semantic segmentation consists of assigning a semantic label to each pixel according to predefined classes. This process facilitates the understanding of object appearance and spatial relationships, playing an important role in the global interpretation of image content. Although modern deep learning approaches achieve high accuracy, they often ignore ordinal relationships among classes, which may encode important domain knowledge for scene interpretation. In this work, loss functions that incorporate ordinal relationships into deep neural networks are investigated to promote greater semantic consistency in semantic segmentation tasks. These loss functions are categorized as unimodal, quasi-unimodal, and spatial. Unimodal losses constrain the predicted probability distribution according to the class ordering, while quasi-unimodal losses relax this constraint by allowing small variations while preserving ordinal coherence. Spatial losses penalize semantic inconsistencies between neighboring pixels, encouraging smoother transitions in the image space. In particular, this study adapts loss functions originally proposed for ordinal classification to ordinal semantic segmentation. Among them, the Expanded Mean Squared Error (EXP_MSE), the Quasi-Unimodal Loss (QUL), and the spatial Contact Surface Loss using Signal Distance Function (CSSDF) are investigated. These approaches have shown promising results in medical imaging, improving robustness, generalization, and anatomical consistency.
>
---
#### [new 045] Domain-Guided YOLO26 with Composite BCE-Dice-Lovász Loss for Multi-Class Fetal Head Ultrasound Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决胎儿头部结构在超声图像中的多类别分割问题。通过改进YOLO26模型，引入复合损失函数和增强策略，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2603.26755](https://arxiv.org/pdf/2603.26755)**

> **作者:** M. Fazri Nizar
>
> **摘要:** Segmenting fetal head structures from prenatal ultrasound remains a practical bottleneck in obstetric imaging. The current state-of-the-art baseline, proposed alongside the published dataset, adapts the Segment Anything Model with per-class Dice and Lovász losses but still depends on bounding-box prompts at test time. We build a prompt-free pipeline on top of YOLO26-Seg that jointly detects and segments three structures, Brain, Cavum Septi Pellucidi (CSP), and Lateral Ventricles (LV), in a single forward pass. Three modifications are central to our approach: (i) a composite BCE-Dice-Lovász segmentation loss with inverse-frequency class weighting, injected into the YOLO26 training loop via runtime monkey-patching; (ii) domain-guided copy-paste augmentation that transplants minority-class structures while respecting their anatomical location relative to the brain boundary; and (iii) inter-patient stratified splitting to prevent data leakage. On 575 held-out test images, the composite loss variant reaches a mean Dice coefficient of 0.9253, exceeding the baseline (0.9012) by 2.68 percentage points, despite reporting over three foreground classes only, whereas the baseline's reported mean includes the easy background class. We further ablate each component and discuss annotation-quality and class-imbalance effects on CSP and LV performance.
>
---
#### [new 046] Gated Condition Injection without Multimodal Attention: Towards Controllable Linear-Attention Transformers
- **分类: cs.CV**

- **简介: 该论文属于可控图像生成任务，旨在解决现有模型在边缘设备上效率低、条件支持不足的问题。提出一种基于线性注意力的可控扩散框架，提升生成效果与灵活性。**

- **链接: [https://arxiv.org/pdf/2603.27666](https://arxiv.org/pdf/2603.27666)**

> **作者:** Yuhe Liu; Zhenxiong Tan; Yujia Hu; Songhua Liu; Xinchao Wang
>
> **摘要:** Recent advances in diffusion-based controllable visual generation have led to remarkable improvements in image quality. However, these powerful models are typically deployed on cloud servers due to their large computational demands, raising serious concerns about user data privacy. To enable secure and efficient on-device generation, we explore in this paper controllable diffusion models built upon linear attention architectures, which offer superior scalability and efficiency, even on edge devices. Yet, our experiments reveal that existing controllable generation frameworks, such as ControlNet and OminiControl, either lack the flexibility to support multiple heterogeneous condition types or suffer from slow convergence on such linear-attention models. To address these limitations, we propose a novel controllable diffusion framework tailored for linear attention backbones like SANA. The core of our method lies in a unified gated conditioning module working in a dual-path pipeline, which effectively integrates multi-type conditional inputs, such as spatially aligned and non-aligned cues. Extensive experiments on multiple tasks and benchmarks demonstrate that our approach achieves state-of-the-art controllable generation performance based on linear-attention models, surpassing existing methods in terms of fidelity and controllability.
>
---
#### [new 047] Detection of Adversarial Attacks in Robotic Perception
- **分类: cs.CV; cs.AI; cs.CR; cs.RO**

- **简介: 论文属于机器人感知任务，旨在检测深度神经网络在语义分割中面临的对抗攻击，解决其安全性问题，提出针对性的检测策略。**

- **链接: [https://arxiv.org/pdf/2603.28594](https://arxiv.org/pdf/2603.28594)**

> **作者:** Ziad Sharawy; Mohammad Nakshbandiand; Sorin Mihai Grigorescu
>
> **备注:** 9 pages, 6 figures. Accepted and presented at STE 2025, Transilvania University of Brasov, Romania
>
> **摘要:** Deep Neural Networks (DNNs) achieve strong performance in semantic segmentation for robotic perception but remain vulnerable to adversarial attacks, threatening safety-critical applications. While robustness has been studied for image classification, semantic segmentation in robotic contexts requires specialized architectures and detection strategies.
>
---
#### [new 048] Human-Centric Perception for Child Sexual Abuse Imagery
- **分类: cs.CV**

- **简介: 该论文属于儿童性虐待图像分类任务，旨在解决自动识别中的主观性和解释性问题。工作包括构建人体关键点数据集，提出两种联合姿态与检测方法，提升分类的客观性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.27290](https://arxiv.org/pdf/2603.27290)**

> **作者:** Camila Laranjeira; João Macedo; Sandra Avila; Fabrício Benevenuto; Jefersson A. dos Santos
>
> **备注:** submitted to IEEE Transactions on Information Forensics and Security (TIFS)
>
> **摘要:** Law enforcement agencies and non-gonvernmental organizations handling reports of Child Sexual Abuse Imagery (CSAI) are overwhelmed by large volumes of data, requiring the aid of automation tools. However, defining sexual abuse in images of children is inherently challenging, encompassing sexually explicit activities and hints of sexuality conveyed by the individual's pose, or their attire. CSAI classification methods often rely on black-box approaches, targeting broad and abstract concepts such as pornography. Thus, our work is an in-depth exploration of tasks from the literature on Human-Centric Perception, across the domains of safe images, adult pornography, and CSAI, focusing on targets that enable more objective and explainable pipelines for CSAI classification in the future. We introduce the Body-Keypoint-Part Dataset (BKPD), gathering images of people from varying age groups and sexual explicitness to approximate the domain of CSAI, along with manually curated hierarchically structured labels for skeletal keypoints and bounding boxes for person and body parts, including head, chest, hip, and hands. We propose two methods, namely BKP-Association and YOLO-BKP, for simultaneous pose estimation and detection, with targets associated per individual for a comprehensive decomposed representation of each person. Our methods are benchmarked on COCO-Keypoints and COCO-HumanParts, as well as our human-centric dataset, achieving competitive results with models that jointly perform all tasks. Cross-domain ablation studies on BKPD and a case study on RCPD highlight the challenges posed by sexually explicit domains. Our study addresses previously unexplored targets in the CSAI domain, paving the way for novel research opportunities.
>
---
#### [new 049] EuraGovExam: A Multilingual Multimodal Benchmark from Real-World Civil Service Exams
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出EuraGovExam，一个跨语言、多模态的基准数据集，用于评估视觉-语言模型在复杂公职考试场景中的表现。**

- **链接: [https://arxiv.org/pdf/2603.27223](https://arxiv.org/pdf/2603.27223)**

> **作者:** JaeSeong Kim; Chaehwan Lim; Sang Hyun Gil; Suan Lee
>
> **摘要:** We present EuraGovExam, a multilingual and multimodal benchmark sourced from real-world civil service examinations across five representative Eurasian regions: South Korea, Japan, Taiwan, India, and the European Union. Designed to reflect the authentic complexity of public-sector assessments, the dataset contains over 8,000 high-resolution scanned multiple-choice questions covering 17 diverse academic and administrative domains. Unlike existing benchmarks, EuraGovExam embeds all question content--including problem statements, answer choices, and visual elements--within a single image, providing only a minimal standardized instruction for answer formatting. This design demands that models perform layout-aware, cross-lingual reasoning directly from visual input. All items are drawn from real exam documents, preserving rich visual structures such as tables, multilingual typography, and form-like layouts. Evaluation results show that even state-of-the-art vision-language models (VLMs) achieve only 86% accuracy, underscoring the benchmark's difficulty and its power to diagnose the limitations of current models. By emphasizing cultural realism, visual complexity, and linguistic diversity, EuraGovExam establishes a new standard for evaluating VLMs in high-stakes, multilingual, image-grounded settings. It also supports practical applications in e-governance, public-sector document analysis, and equitable exam preparation.
>
---
#### [new 050] Amped: Adaptive Multi-stage Non-edge Pruning for Edge Detection
- **分类: cs.CV**

- **简介: 该论文属于边缘检测任务，旨在解决Transformer模型计算成本高、效率低的问题。提出Amped框架，通过非边缘剪枝提升效率，同时保持高精度。**

- **链接: [https://arxiv.org/pdf/2603.27661](https://arxiv.org/pdf/2603.27661)**

> **作者:** Yuhan Gao; Xinqing Li; Xin He; Bing Li; Xinzhong Zhu; Ming-Ming Cheng; Yun Liu
>
> **摘要:** Edge detection is a fundamental image analysis task that underpins numerous high-level vision applications. Recent advances in Transformer architectures have significantly improved edge quality by capturing long-range dependencies, but this often comes with computational overhead. Achieving higher pixel-level accuracy requires increased input resolution, further escalating computational cost and limiting practical deployment. Building on the strong representational capacity of recent Transformer-based edge detectors, we propose an Adaptive Multi-stage non-edge Pruning framework for Edge Detection(Amped). Amped identifies high-confidence non-edge tokens and removes them as early as possible to substantially reduce computation, thus retaining high accuracy while cutting GFLOPs and accelerating inference with minimal performance loss. Moreover, to mitigate the structural complexity of existing edge detection networks and facilitate their integration into real-world systems, we introduce a simple yet high-performance Transformer-based model, termed Streamline Edge Detector(SED). Applied to both existing detectors and our SED, the proposed pruning strategy provides a favorable balance between accuracy and efficiency-reducing GFLOPs by up to 40% with only a 0.4% drop in ODS F-measure. In addition, despite its simplicity, SED achieves a state-of-the-art ODS F-measure of 86.5%. The code will be released.
>
---
#### [new 051] You Only Erase Once: Erasing Anything without Bringing Unexpected Content
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决对象擦除时生成无关内容的问题。通过无配对数据训练，结合检测器和上下文损失，实现高质量擦除。**

- **链接: [https://arxiv.org/pdf/2603.27599](https://arxiv.org/pdf/2603.27599)**

> **作者:** Yixing Zhu; Qing Zhang; Wenju Xu; Wei-Shi Zheng
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** We present YOEO, an approach for object erasure. Unlike recent diffusion-based methods which struggle to erase target objects without generating unexpected content within the masked regions due to lack of sufficient paired training data and explicit constraint on content generation, our method allows to produce high-quality object erasure results free of unwanted objects or artifacts while faithfully preserving the overall context coherence to the surrounding content. We achieve this goal by training an object erasure diffusion model on unpaired data containing only large-scale real-world images, under the supervision of a sundries detector and a context coherence loss that are built upon an entity segmentation model. To enable more efficient training and inference, a diffusion distillation strategy is employed to train for a few-step erasure diffusion model. Extensive experiments show that our method outperforms the state-of-the-art object erasure methods. Code will be available at this https URL.
>
---
#### [new 052] FlashSign: Pose-Free Guidance for Efficient Sign Language Video Generation
- **分类: cs.CV**

- **简介: 该论文属于手语视频生成任务，旨在解决现有模型依赖复杂姿态表示导致效率低的问题。提出一种无需姿态的扩散方法和可训练注意力机制，提升生成速度与质量。**

- **链接: [https://arxiv.org/pdf/2603.27915](https://arxiv.org/pdf/2603.27915)**

> **作者:** Liuzhou Zhang; Zeyu Zhang; Biao Wu; Luyao Tang; Zirui Song; Hongyang He; Renda Han; Guangzhen Yao; Huacan Wang; Ronghao Chen; Xiuying Chen; Guan Huang; Zheng Zhu
>
> **摘要:** Sign language plays a crucial role in bridging communication gaps between the deaf and hard-of-hearing communities. However, existing sign language video generation models often rely on complex intermediate representations, which limits their flexibility and efficiency. In this work, we propose a novel pose-free framework for real-time sign language video generation. Our method eliminates the need for intermediate pose representations by directly mapping natural language text to sign language videos using a diffusion-based approach. We introduce two key innovations: (1) a pose-free generative model based on the a state-of-the-art diffusion backbone, which learns implicit text-to-gesture alignments without pose estimation, and (2) a Trainable Sliding Tile Attention (T-STA) mechanism that accelerates inference by exploiting spatio-temporal locality patterns. Unlike previous training-free sparsity approaches, T-STA integrates trainable sparsity into both training and inference, ensuring consistency and eliminating the train-test gap. This approach significantly reduces computational overhead while maintaining high generation quality, making real-time deployment feasible. Our method increases video generation speed by 3.07x without compromising video quality. Our contributions open new avenues for real-time, high-quality, pose-free sign language synthesis, with potential applications in inclusive communication tools for diverse communities. Code: this https URL.
>
---
#### [new 053] A Cross-Scale Decoder with Token Refinement for Off-Road Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于越野场景语义分割任务，解决边界模糊、结构稀疏等问题。提出跨尺度解码器，通过全局-局部token精修、门控细节桥和不确定性引导的点精修提升分割精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.27931](https://arxiv.org/pdf/2603.27931)**

> **作者:** Seongkyu Choi Jhonghyun An
>
> **摘要:** Off-road semantic segmentation is fundamentally challenged by irregular terrain, vegetation clutter, and inherent annotation ambiguity. Unlike urban scenes with crisp object boundaries, off-road environments exhibit strong class-level similarity among terrain categories, resulting in thick and uncertain transition regions that degrade boundary coherence and destabilize training. Rare or thin structures, such as narrow traversable gaps or isolated obstacles, further receive sparse and unreliable supervision and are easily overwhelmed by dominant background textures. Existing decoder designs either rely on low-scale bottlenecks that oversmooth fine structural details, or repeatedly fuse high-detail features, which tends to amplify annotation noise and incur substantial computational cost. We present a cross-scale decoder that explicitly addresses these challenges through three complementary mechanisms. First, a global--local token refinement module consolidates semantic context on a compact bottleneck lattice, guided by boundary-aware regularization to remain robust under ambiguous supervision. Second, a gated detail bridge selectively injects fine-scale structural cues only once through cross-scale attention, preserving boundary and texture information while avoiding noise accumulation. Third, an uncertainty-guided class-aware point refinement selectively updates the least reliable pixels, improving rare and ambiguous structures with minimal computational overhead. The resulting framework achieves noise-robust and boundary-preserving segmentation tailored to off-road environments, recovering fine structural details while maintaining deployment-friendly efficiency. Experimental results on standard off-road benchmarks demonstrate consistent improvements over prior approaches without resorting to heavy dense feature fusion.
>
---
#### [new 054] SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人手与物体交互的3D理解任务，旨在解决真实环境中手和物体3D标注不准确的问题。提出SHOW3D数据集和多相机系统，实现高精度3D标注与真实环境结合。**

- **链接: [https://arxiv.org/pdf/2603.28760](https://arxiv.org/pdf/2603.28760)**

> **作者:** Patrick Rim; Kevin Harris; Braden Copple; Shangchen Han; Xu Xie; Ivan Shugurov; Sizhe An; He Wen; Alex Wong; Tomas Hodan; Kun He
>
> **备注:** CVPR 2026
>
> **摘要:** Accurate 3D understanding of human hands and objects during manipulation remains a significant challenge for egocentric computer vision. Existing hand-object interaction datasets are predominantly captured in controlled studio settings, which limits both environmental diversity and the ability of models trained on such data to generalize to real-world scenarios. To address this challenge, we introduce a novel marker-less multi-camera system that allows for nearly unconstrained mobility in genuinely in-the-wild conditions, while still having the ability to generate precise 3D annotations of hands and objects. The capture system consists of a lightweight, back-mounted, multi-camera rig that is synchronized and calibrated with a user-worn VR headset. For 3D ground-truth annotation of hands and objects, we develop an ego-exo tracking pipeline and rigorously evaluate its quality. Finally, we present SHOW3D, the first large-scale dataset with 3D annotations that show hands interacting with objects in diverse real-world environments, including outdoor settings. Our approach significantly reduces the fundamental trade-off between environmental realism and accuracy of 3D annotations, which we validate with experiments on several downstream tasks. this http URL
>
---
#### [new 055] arg-VU: Affordance Reasoning with Physics-Aware 3D Geometry for Visual Understanding in Robotic Surgery
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉理解任务，解决手术中软组织变形带来的感知与动作关联问题。提出arg-VU框架，结合物理约束与3D几何，提升手术场景的 affordance 预测准确性。**

- **链接: [https://arxiv.org/pdf/2603.26814](https://arxiv.org/pdf/2603.26814)**

> **作者:** Nan Xiao; Yunxin Fan; Farong Wang; Fei Liu
>
> **摘要:** Affordance reasoning provides a principled link between perception and action, yet remains underexplored in surgical robotics, where tissues are highly deformable, compliant, and dynamically coupled with tool motion. We present arg-VU, a physics-aware affordance reasoning framework that integrates temporally consistent geometry tracking with constraint-induced mechanical modeling for surgical visual understanding. Surgical scenes are reconstructed using 3D Gaussian Splatting (3DGS) and converted into a temporally tracked surface representation. Extended Position-Based Dynamics (XPBD) embeds local deformation constraints and produces representative geometry points (RGPs) whose constraint sensitivities define anisotropic stiffness metrics capturing the local constraint-manifold geometry. Robotic tool poses in SE(3) are incorporated to compute rigidly induced displacements at RGPs, from which we derive two complementary measures: a physics-aware compliance energy that evaluates mechanical feasibility with respect to local deformation constraints, and a positional agreement score that captures motion alignment (as kinematic motion baseline). Experiments on surgical video datasets show that arg-VU yields more stable, physically consistent, and interpretable affordance predictions than kinematic baselines. These results demonstrate that physics-aware geometric representations enable reliable affordance reasoning for deformable surgical environments and support embodied robotic interaction.
>
---
#### [new 056] LACON: Training Text-to-Image Model from Uncurated Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决传统数据集筛选方法可能浪费潜在有用数据的问题。通过引入LACON框架，利用未筛选数据中的质量信号进行训练，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.26866](https://arxiv.org/pdf/2603.26866)**

> **作者:** Zhiyang Liang; Ziyu Wan; Hongyu Liu; Dong Chen; Qiu Shen; Hao Zhu; Dongdong Chen
>
> **摘要:** The success of modern text-to-image generation is largely attributed to massive, high-quality datasets. Currently, these datasets are curated through a filter-first paradigm that aggressively discards low-quality raw data based on the assumption that it is detrimental to model performance. Is the discarded bad data truly useless, or does it hold untapped potential? In this work, we critically re-examine this question. We propose LACON (Labeling-and-Conditioning), a novel training framework that exploits the underlying uncurated data distribution. Instead of filtering, LACON re-purposes quality signals, such as aesthetic scores and watermark probabilities as explicit, quantitative condition labels. The generative model is then trained to learn the full spectrum of data quality, from bad to good. By learning the explicit boundary between high- and low-quality content, LACON achieves superior generation quality compared to baselines trained only on filtered data using the same compute budget, proving the significant value of uncurated data.
>
---
#### [new 057] MolmoPoint: Better Pointing for VLMs with Grounding Tokens
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型的定位任务，旨在解决传统坐标生成方式复杂且效率低的问题。提出通过选择视觉token实现更直观的定位机制，提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2603.28069](https://arxiv.org/pdf/2603.28069)**

> **作者:** Christopher Clark; Yue Yang; Jae Sung Park; Zixian Ma; Jieyu Zhang; Rohun Tripathi; Mohammadreza Salehi; Sangho Lee; Taira Anderson; Winson Han; Ranjay Krishna
>
> **摘要:** Grounding has become a fundamental capability of vision-language models (VLMs). Most existing VLMs point by generating coordinates as part of their text output, which requires learning a complicated coordinate system and results in a high token count. Instead, we propose a more intuitive pointing mechanism that directly selects the visual tokens that contain the target concept. Our model generates a special pointing token that cross-attends to the input image or video tokens and selects the appropriate one. To make this model more fine-grained, we follow these pointing tokens with an additional special token that selects a fine-grained subpatch within the initially selected region, and then a third token that specifies a location within that subpatch. We further show that performance improves by generating points sequentially in a consistent order, encoding the relative position of the previously selected point, and including a special no-more-points class when selecting visual tokens. Using this method, we set a new state-of-the-art on image pointing (70.7% on PointBench), set a new state-of-the-art among fully open models on GUI pointing (61.1% on ScreenSpotPro), and improve video pointing (59.1% human preference win rate vs. a text coordinate baseline) and tracking (+6.3% gain on Molmo2Track). We additionally show that our method achieves much higher sample efficiency and discuss the qualitative differences that emerge from this design change.
>
---
#### [new 058] Beyond Scanpaths: Graph-Based Gaze Simulation in Dynamic Scenes
- **分类: cs.CV**

- **简介: 该论文属于视觉注意力建模任务，旨在解决动态场景中人类注视轨迹的模拟问题。通过构建图模型和预测机制，生成更自然的 gaze 轨迹与显著图。**

- **链接: [https://arxiv.org/pdf/2603.28319](https://arxiv.org/pdf/2603.28319)**

> **作者:** Luke Palmer; Petar Palasek; Hazem Abdelkawy
>
> **摘要:** Accurately modelling human attention is essential for numerous computer vision applications, particularly in the domain of automotive safety. Existing methods typically collapse gaze into saliency maps or scanpaths, treating gaze dynamics only implicitly. We instead formulate gaze modelling as an autoregressive dynamical system and explicitly unroll raw gaze trajectories over time, conditioned on both gaze history and the evolving environment. Driving scenes are represented as gaze-centric graphs processed by the Affinity Relation Transformer (ART), a heterogeneous graph transformer that models interactions between driver gaze, traffic objects, and road structure. We further introduce the Object Density Network (ODN) to predict next-step gaze distributions, capturing the stochastic and object-centric nature of attentional shifts in complex environments. We also release Focus100, a new dataset of raw gaze data from 30 participants viewing egocentric driving footage. Trained directly on raw gaze, without fixation filtering, our unified approach produces more natural gaze trajectories, scanpath dynamics, and saliency maps than existing attention models, offering valuable insights for the temporal modelling of human attention in dynamic environments.
>
---
#### [new 059] Limits of Imagery Reasoning in Frontier LLM Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言推理任务，探讨LLM在空间推理上的局限性。通过引入外部影像模块，发现模型仍无法有效处理3D模型旋转，揭示其缺乏基础的视觉空间能力。**

- **链接: [https://arxiv.org/pdf/2603.26779](https://arxiv.org/pdf/2603.26779)**

> **作者:** Sergio Y. Hayashi; Nina S. T. Hirata
>
> **备注:** 25 pages
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, yet they struggle with spatial tasks that require mental simulation, such as mental rotation. This paper investigates whether equipping an LLM with an external ``Imagery Module'' -- a tool capable of rendering and rotating 3D models -- can bridge this gap, functioning as a ``cognitive prosthetic.'' We conducted experiments using a dual-module architecture in which a reasoning module (an MLLM) interacts with an imagery module on 3D model rotation tasks. Performance was lower than expected, with accuracy reaching at most 62.5%. Further investigation suggests that even when the burden of maintaining and manipulating a holistic 3D state is outsourced, the system still fails. This reveals that current frontier models lack the foundational visual-spatial primitives required to interface with imagery. Specifically, they lack: (1) the low-level sensitivity to extract spatial signals such as (a) depth, (b) motion, and (c) short-horizon dynamic prediction; and (2) the capacity to reason contemplatively over images, dynamically shifting visual focus and balancing imagery with symbolic and associative information.
>
---
#### [new 060] A Near-Raw Talking-Head Video Dataset for Various Computer Vision Tasks
- **分类: cs.CV; cs.MM; eess.IV**

- **简介: 该论文提出一个近原始的说话头视频数据集，用于视频压缩和增强研究，解决现有数据集规模小、质量低的问题。**

- **链接: [https://arxiv.org/pdf/2603.26763](https://arxiv.org/pdf/2603.26763)**

> **作者:** Babak Naderi; Ross Cutler
>
> **摘要:** Talking-head videos constitute a predominant content type in real-time communication, yet publicly available datasets for video processing research in this domain remain scarce and limited in signal fidelity. In this paper, we open-source a near-raw dataset of 847 talking-head recordings (approximately 212 minutes), each 15\,s in duration, captured from 805 participants using 446 unique consumer webcam devices in their natural environments. All recordings are stored using the FFV1 lossless codec, preserving the camera-native signal -- uncompressed (24.4\%) or MJPEG-encoded (75.6\%) -- without additional lossy processing. Each recording is annotated with a Mean Opinion Score (MOS) and ten perceptual quality tokens that jointly explain 64.4\% of the MOS variance. From this corpus, we curate a stratified benchmarking subset of 120 clips in three content conditions: original, background blur, and background replacement. Codec efficiency evaluation across four datasets and four codecs, namely H.264, H.265, H.266, and AV1, yields VMAF BD-rate savings up to $-71.3\%$ (H.266) relative to H.264, with significant encoder$\times$dataset ($\eta_p^2 = .112$) and encoder$\times$content condition ($\eta_p^2 = .149$) interactions, demonstrating that both content type and background processing affect compression efficiency. The dataset offers 5$\times$ the scale of the largest prior talking-head webcam dataset (847 vs.\ 160 clips) with lossless signal fidelity, establishing a resource for training and benchmarking video compression and enhancement models in real-time communication.
>
---
#### [new 061] Post-hoc Self-explanation of CNNs
- **分类: cs.CV; stat.ML**

- **简介: 该论文属于模型解释任务，旨在提升CNN的可解释性。通过替换分类器并利用k-means和特征激活生成解释图，解决现有模型解释不准确的问题。**

- **链接: [https://arxiv.org/pdf/2603.28466](https://arxiv.org/pdf/2603.28466)**

> **作者:** Ahcène Boubekki; Line H. Clemmensen
>
> **摘要:** Although standard Convolutional Neural Networks (CNNs) can be mathematically reinterpreted as Self-Explainable Models (SEMs), their built-in prototypes do not on their own accurately represent the data. Replacing the final linear layer with a $k$-means-based classifier addresses this limitation without compromising performance. This work introduces a common formalization of $k$-means-based post-hoc explanations for the classifier, the encoder's final output (B4), and combinations of intermediate feature activations. The latter approach leverages the spatial consistency of convolutional receptive fields to generate concept-based explanation maps, which are supported by gradient-free feature attribution maps. Empirical evaluation with a ResNet34 shows that using shallower, less compressed feature activations, such as those from the last three blocks (B234), results in a trade-off between semantic fidelity and a slight reduction in predictive performance.
>
---
#### [new 062] JND-Guided Neural Watermarking with Spatial Transformer Decoding for Screen-Capture Robustness
- **分类: cs.CV**

- **简介: 该论文属于屏幕拍摄鲁棒水印任务，解决水印在复杂失真下提取困难的问题。通过深度学习框架，结合噪声模拟、感知损失和自动定位模块，提升水印的鲁棒性和视觉质量。**

- **链接: [https://arxiv.org/pdf/2603.26766](https://arxiv.org/pdf/2603.26766)**

> **作者:** Jiayi Qin; Jingwei Li; Chuan Wu
>
> **摘要:** Screen-shooting robust watermarking aims to imperceptibly embed extractable information into host images such that the watermark survives the complex distortion pipeline of screen display and camera recapture. However, achieving high extraction accuracy while maintaining satisfactory visual quality remains an open challenge, primarily because the screen-shooting channel introduces severe and entangled degradations including Moiré patterns, color-gamut shifts, perspective warping, and sensor noise. In this paper, we present an end-to-end deep learning framework that jointly optimizes watermark embedding and extraction for screen-shooting robustness. Our framework incorporates three key innovations: (i) a comprehensive noise simulation layer that faithfully models realistic screen-shooting distortions -- notably including a physically-motivated Moiré pattern generator -- enabling the network to learn robust representations against the full spectrum of capture-channel noise through adversarial training; (ii) a Just Noticeable Distortion (JND) perceptual loss function that adaptively modulates watermark embedding strength by supervising the perceptual discrepancy between the JND coefficient map and the watermark residual, thereby concentrating watermark energy in perceptually insensitive regions to maximize visual quality; and (iii) two complementary automatic localization modules -- a semantic-segmentation-based foreground extractor for captured image rectification and a symmetric noise template mechanism for anti-cropping region recovery -- that enable fully automated watermark decoding under realistic deployment conditions. Extensive experiments demonstrate that our method achieves an average PSNR of 30.94~dB and SSIM of 0.94 on watermarked images while embedding 127-bit payloads.
>
---
#### [new 063] Weakly Convex Ridge Regularization for 3D Non-Cartesian MRI Reconstruction
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D MRI重建任务，旨在解决加速扫描下的重建延迟和稳定性问题。提出弱凸岭正则化方法，提升重建效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27158](https://arxiv.org/pdf/2603.27158)**

> **作者:** German Shâma Wache; Chaithya G R; Asma Tanabene; Sebastian Neumayer
>
> **摘要:** While highly accelerated non-Cartesian acquisition protocols significantly reduce scan time, they often entail long reconstruction delays. Deep learning based reconstruction methods can alleviate this, but often lack stability and robustness to distribution shifts. As an alternative, we train a rotation invariant weakly convex ridge regularizer (WCRR). The resulting variational reconstruction approach is benchmarked against state of the art methods on retrospectively simulated data and (out of distribution) on prospective GoLF SPARKLING and CAIPIRINHA acquisitions. Our approach consistently outperforms widely used baselines and achieves performance comparable to Plug and Play reconstruction with a state of the art 3D DRUNet denoiser, while offering substantially improved computational efficiency and robustness to acquisition changes. In summary, WCRR unifies the strengths of principled variational methods and modern deep learning based approaches.
>
---
#### [new 064] A Multimodal Deep Learning Framework for Edema Classification Using HCT and Clinical Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在提升脑水肿检测的准确性。通过融合头颅CT与临床数据，提出AttentionMixer框架，实现高效、可解释的多模态融合。**

- **链接: [https://arxiv.org/pdf/2603.26726](https://arxiv.org/pdf/2603.26726)**

> **作者:** Aram Ansary Ogholbake; Hannah Choi; Spencer Brandenburg; Alyssa Antuna; Zahraa Al-Sharshahi; Makayla Cox; Haseeb Ahmed; Jacqueline Frank; Nathan Millson; Luke Bauerle; Jessica Lee; David Dornbos III; Qiang Cheng
>
> **摘要:** We propose AttentionMixer, a unified deep learning framework for multimodal detection of brain edema that combines structural head CT (HCT) with routine clinical metadata. While HCT provides rich spatial information, clinical variables such as age, laboratory values, and scan timing capture complementary context that might be ignored or naively concatenated. AttentionMixer is designed to fuse these heterogeneous sources in a principled and efficient manner. HCT volumes are first encoded using a self-supervised Vision Transformer Autoencoder (ViT-AE++), without requiring large labeled datasets. Clinical metadata are mapped into the same feature space and used as keys and values in a cross-attention module, where HCT-derived feature vector serves as queries. This cross-attention fusion allows the network to dynamically modulate imaging features based on patient-specific context and provides an interpretable mechanism for multimodal integration. A lightweight MLP-Mixer then refines the fused representation before final classification, enabling global dependency modeling with substantially reduced parameter overhead. Missing or incomplete metadata are handled via a learnable embedding, promoting robustness to real-world clinical data quality. We evaluate AttentionMixer on a curated brain HCT cohort with expert edema annotations using five-fold cross-validation. Compared with strong HCT-only, metadata-only, and prior multimodal baselines, AttentionMixer achieves superior performance (accuracy 87.32%, precision 92.10%, F1-score 85.37%, AUC 94.14%). Ablation studies confirm the benefit of both cross-attention and MLP-Mixer refinement, and permutation-based metadata importance analysis highlights clinically meaningful variables driving predictions. These results demonstrate that structured, interpretable multimodal fusion can substantially improve edema detection in clinical practice.
>
---
#### [new 065] RailVQA: A Benchmark and Framework for Efficient Interpretable Visual Cognition in Automatic Train Operation
- **分类: cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决ATO中视觉认知与决策问题。提出RailVQA-bench基准和RailVQA-CoM框架，提升感知泛化与可解释性，降低延迟。**

- **链接: [https://arxiv.org/pdf/2603.27112](https://arxiv.org/pdf/2603.27112)**

> **作者:** Sen Zhang; Runmei Li; Zhichao Zheng; Yuhe Zhang; Jiani Li; Kailun Zhang; Tao Zhang; Wenjun Wu; Qunbo Wang
>
> **摘要:** Automatic Train Operation (ATO) relies on low-latency, reliable cab-view visual perception and decision-oriented inference to ensure safe operation in complex and dynamic railway environments. However, existing approaches focus primarily on basic perception and often generalize poorly to rare yet safety-critical corner cases. They also lack the high-level reasoning and planning capabilities required for operational decision-making. Although recent Large Multi-modal Models (LMMs) show strong generalization and cognitive capabilities, their use in safety-critical ATO is hindered by high computational cost and hallucination risk. Meanwhile, reliable domain-specific benchmarks for systematically evaluating cognitive capabilities are still lacking. To address these gaps, we introduce RailVQA-bench, the first VQA benchmark for cab-view visual cognition in ATO, comprising 20,000 single-frame and 1,168 video based QA pairs to evaluate cognitive generalization and interpretability in both static and dynamic scenarios. Furthermore, we propose RailVQA-CoM, a collaborative large-small model framework that combines small-model efficiency with large-model cognition via a transparent three-module architecture and adaptive temporal sampling, improving perceptual generalization and enabling efficient reasoning and planning. Experiments demonstrate that the proposed approach substantially improves performance, enhances interpretability, reduces inference latency, and strengthens cross-domain generalization, while enabling plug-and-play deployment in autonomous driving systems. Code and datasets will be available at this https URL.
>
---
#### [new 066] Unsafe by Reciprocity: How Generation-Understanding Coupling Undermines Safety in Unified Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于安全研究任务，探讨统一多模态模型中的安全漏洞。通过提出RICE攻击框架，揭示理解与生成功能间的相互影响如何加剧安全风险。**

- **链接: [https://arxiv.org/pdf/2603.27332](https://arxiv.org/pdf/2603.27332)**

> **作者:** Kaishen Wang; Heng Huang
>
> **备注:** 7 figures, 3 tables
>
> **摘要:** Recent advances in Large Language Models (LLMs) and Text-to-Image (T2I) models have led to the emergence of Unified Multimodal Models (UMMs), where multimodal understanding and image generation are tightly integrated within a shared architecture. Prior studies suggest that such reciprocity enhances cross-functionality performance through shared representations and joint optimization. However, the safety implications of this tight coupling remain largely unexplored, as existing safety research predominantly analyzes understanding and generation functionalities in isolation. In this work, we investigate whether cross-functionality reciprocity itself constitutes a structural source of vulnerability in UMMs. We propose RICE: Reciprocal Interaction-based Cross-functionality Exploitation, a novel attack paradigm that explicitly exploits bidirectional interactions between understanding and generation. Using this framework, we systematically evaluate Generation-to-Understanding (G-U) and Understanding-to-Generation (U-G) attack pathways, demonstrating that unsafe intermediate signals can propagate across modalities and amplify safety risks. Extensive experiments show high Attack Success Rates (ASR) in both directions, revealing previously overlooked safety weaknesses inherent to UMMs.
>
---
#### [new 067] DinoDental: Benchmarking DINOv3 as a Unified Vision Encoder for Dental Image Analysis
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决牙科影像标注稀缺问题。通过构建DinoDental基准，评估DINOv3在牙科图像中的适用性。**

- **链接: [https://arxiv.org/pdf/2603.28297](https://arxiv.org/pdf/2603.28297)**

> **作者:** Kun Tang; Xinquan Yang; Mianjie Zheng; Xuefen Liu; Xuguang Li; Xiaoqi Guo; Ruihan Chen; Linlin Shen; He Meng
>
> **摘要:** The scarcity and high cost of expert annotations in dental imaging present a significant challenge for the development of AI in dentistry. DINOv3, a state-of-the-art, self-supervised vision foundation model pre-trained on 1.7 billion images, offers a promising pathway to mitigate this issue. However, its reliability when transferred to the dental domain, with its unique imaging characteristics and clinical subtleties, remains unclear. To address this, we introduce DinoDental, a unified benchmark designed to systematically evaluate whether DINOv3 can serve as a reliable, off-the-shelf encoder for comprehensive dental image analysis without requiring domain-specific pre-training. Constructed from multiple public datasets, DinoDental covers a wide range of tasks, including classification, detection, and instance segmentation on both panoramic radiographs and intraoral photographs. We further analyze the model's transfer performance by scaling its size and input resolution, and by comparing different adaptation strategies, including frozen features, full fine-tuning, and the parameter-efficient Low-Rank Adaptation (LoRA) method. Our experiments show that DINOv3 can serve as a strong unified encoder for dental image analysis across both panoramic radiographs and intraoral photographs, remaining competitive across tasks while showing particularly clear advantages for intraoral image understanding and boundary-sensitive dense prediction. Collectively, DinoDental provides a systematic framework for comprehensively evaluating DINOv3 in dental analysis, establishing a foundational benchmark to guide efficient and effective model selection and adaptation for the dental AI community.
>
---
#### [new 068] Follow Your Heart: Landmark-Guided Transducer Pose Scoring for Point-of-Care Echocardiography
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学影像分析任务，旨在解决点位护理超声中获取高质量四腔心图像的难题。通过多任务网络提供定位反馈和LVEF估计，提升图像采集效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.27143](https://arxiv.org/pdf/2603.27143)**

> **作者:** Zaiyang Guo; Jessie N. Dong; Filippos Bellos; Jilei Hao; Emily J. MacKay; Trevor Chan; Shir Goldfinger; Sethu Reddy; Steven Vance; Jason J. Corso; Alison M. Pouch
>
> **备注:** Accepted for oral presentation at the International Symposium on Biomedical Imaging 2026
>
> **摘要:** Point-of-care transthoracic echocardiography (TTE) makes it possible to assess a patient's cardiac function in almost any setting. A critical step in the TTE exam is acquisition of the apical 4-chamber (A4CH) view, which is used to evaluate clinically impactful measurements such as left ventricular ejection fraction (LVEF). However, optimizing transducer pose for high-quality image acquisition and subsequent measurement is a challenging task, particularly for novice users. In this work, we present a multi-task network that provides feedback cues for A4CH view acquisition and automatically estimates LVEF in high-quality A4CH images. The network cascades a transducer pose scoring module and an uncertainty-aware LV landmark detector with automated LVEF estimation. A strength is that network training and inference do not require cumbersome or costly setups for transducer position tracking. We evaluate performance on point-of-care TTE data acquired with a spatially dense "sweep" protocol around the optimal A4CH view. The results demonstrate the network's ability to determine when the transducer pose is on target, close to target, or far from target based on the images alone, while generating visual landmark cues that guide anatomical interpretation and orientation. In conclusion, we demonstrate a promising strategy to provide guidance for A4CH view acquisition, which may be useful when deploying point-of-care TTE in limited resource settings.
>
---
#### [new 069] Tiny-ViT: A Compact Vision Transformer for Efficient and Explainable Potato Leaf Disease Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于植物病害分类任务，旨在解决土豆叶片疾病高效准确识别问题。提出Tiny-ViT模型，具有高精度、低计算量和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.26761](https://arxiv.org/pdf/2603.26761)**

> **作者:** Shakil Mia; Umme Habiba; Urmi Akter; SK Rezwana Quadir Raisa; Jeba Maliha; Md. Iqbal Hossain; Md. Shakhauat Hossan Sumon
>
> **备注:** Accepted and Presented Paper at the 2026 IEEE International Conference on Electrical, Computer and Telecommunication Engineering, Rajshahi, Bangladesh
>
> **摘要:** Early and precise identification of plant diseases, especially in potato crops is important to ensure the health of the crops and ensure the maximum yield . Potato leaf diseases, such as Early Blight and Late Blight, pose significant challenges to farmers, often resulting in yield losses and increased pesticide use. Traditional methods of detection are not only time-consuming, but are also subject to human error, which is why automated and efficient methods are required. The paper introduces a new method of potato leaf disease classification Tiny-ViT model, which is a small and effective Vision Transformer (ViT) developed to be used in resource-limited systems. The model is tested on a dataset of three classes, namely Early Blight, Late Blight, and Healthy leaves, and the preprocessing procedures include resizing, CLAHE, and Gaussian blur to improve the quality of the image. Tiny-ViT model has an impressive test accuracy of 99.85% and a mean CV accuracy of 99.82% which is better than baseline models such as DEIT Small, SWIN Tiny, and MobileViT XS. In addition to this, the model has a Matthews Correlation Coefficient (MCC) of 0.9990 and narrow confidence intervals (CI) of [0.9980, 0.9995], which indicates high reliability and generalization. The training and testing inference time is competitive, and the model exhibits low computational expenses, thereby, making it applicable in real-time applications. Moreover, interpretability of the model is improved with the help of GRAD-CAM, which identifies diseased areas. Altogether, the proposed Tiny-ViT is a solution with a high level of robustness, efficiency, and explainability to the problem of plant disease classification.
>
---
#### [new 070] MedLoc-R1: Performance-Aware Curriculum Reward Scheduling for GRPO-Based Medical Visual Grounding
- **分类: cs.CV**

- **简介: 该论文针对医疗视觉定位任务，解决RL方法在医学图像中因奖励稀疏导致的训练困难问题。提出MedLoc-R1框架，通过动态调整奖励机制提升定位精度与训练稳定性。**

- **链接: [https://arxiv.org/pdf/2603.28120](https://arxiv.org/pdf/2603.28120)**

> **作者:** Guangjing Yang; Ziyuan Qin; Chaoran Zhang; Chenlin Du; Jinlin Wang; Wanran Sun; Zhenyu Zhang; Bing Ji; Qicheng Lao
>
> **备注:** 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
>
> **摘要:** Medical visual grounding serves as a crucial foundation for fine-grained multimodal reasoning and interpretable clinical decision support. Despite recent advances in reinforcement learning (RL) for grounding tasks, existing approaches such as Group Relative Policy Optimization~(GRPO) suffer from severe reward sparsity when directly applied to medical images, primarily due to the inherent difficulty of localizing small or ambiguous regions of interest, which is further exacerbated by the rigid and suboptimal nature of fixed IoU-based reward schemes in RL. This leads to vanishing policy gradients and stagnated optimization, particularly during early training. To address this challenge, we propose MedLoc-R1, a performance-aware reward scheduling framework that progressively tightens the reward criterion in accordance with model readiness. MedLoc-R1 introduces a sliding-window performance tracker and a multi-condition update rule that automatically adjust the reward schedule from dense, easily obtainable signals to stricter, fine-grained localization requirements, while preserving the favorable properties of GRPO without introducing auxiliary networks or additional gradient paths. Experiments on three medical visual grounding benchmarks demonstrate that MedLoc-R1 consistently improves both localization accuracy and training stability over GRPO-based baselines. Our framework offers a general, lightweight, and effective solution for RL-based grounding in high-stakes medical applications. Code \& checkpoints are available at \hyperlink{}{this https URL}.
>
---
#### [new 071] RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，旨在通过地面和卫星图像估计相机位姿。研究使用全景图和OSM，提出RHO模型解决全景畸变和提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.27758](https://arxiv.org/pdf/2603.27758)**

> **作者:** Junwei Zheng; Ruize Dai; Ruiping Liu; Zichao Zeng; Yufan Chen; Fangjinhua Wang; Kunyu Peng; Kailun Yang; Jiaming Zhang; Rainer Stiefelhagen
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** Metric Cross-View Geo-Localization (MCVGL) aims to estimate the 3-DoF camera pose (position and heading) by matching ground and satellite images. In this work, instead of pinhole and satellite images, we study robust MCVGL using holistic panoramas and OpenStreetMap (OSM). To this end, we establish a large-scale MCVGL benchmark dataset, CV-RHO, with over 2.7M images under different weather and lighting conditions, as well as sensor noise. Furthermore, we propose a model termed RHO with a two-branch Pin-Pan architecture for accurate visual localization. A Split-Undistort-Merge (SUM) module is introduced to address the panoramic distortion, and a Position-Orientation Fusion (POF) mechanism is designed to enhance the localization accuracy. Extensive experiments prove the value of our CV-RHO dataset and the effectiveness of the RHO model, with a significant performance gain up to 20% compared with the state-of-the-art baselines. Project page: this https URL.
>
---
#### [new 072] AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers
- **分类: cs.CV**

- **简介: 该论文属于场景理解任务，旨在解决3D场景中 affordance 学习的难题。通过构建大规模数据集并提出 AffordMatcher 方法，实现图像与点云间的语义匹配，提升 affordance 区域识别精度。**

- **链接: [https://arxiv.org/pdf/2603.27970](https://arxiv.org/pdf/2603.27970)**

> **作者:** Nghia Vu; Tuong Do; Khang Nguyen; Baoru Huang; Nhat Le; Binh Xuan Nguyen; Erman Tjiputra; Quang D. Tran; Ravi Prakash; Te-Chuan Chiu; Anh Nguyen
>
> **备注:** 14 pages. Accepted to CVPR 2026
>
> **摘要:** Affordance learning is a complex challenge in many applications, where existing approaches primarily focus on the geometric structures, visual knowledge, and affordance labels of objects to determine interactable regions. However, extending this learning capability to a scene is significantly more complicated, as incorporating object- and scene-level semantics is not straightforward. In this work, we introduce AffordBridge, a large-scale dataset with 291,637 functional interaction annotations across 685 high-resolution indoor scenes in the form of point clouds. Our affordance annotations are complemented by RGB images that are linked to the same instances within the scenes. Building upon our dataset, we propose AffordMatcher, an affordance learning method that establishes coherent semantic correspondences between image-based and point cloud-based instances for keypoint matching, enabling a more precise identification of affordance regions based on cues, so-called visual signifiers. Experimental results on our dataset demonstrate the effectiveness of our approach compared to other methods.
>
---
#### [new 073] Towards Emotion Recognition with 3D Pointclouds Obtained from Facial Expression Images
- **分类: cs.CV; cs.AI; cs.ET; cs.HC; eess.IV**

- **简介: 该论文属于情感识别任务，旨在解决传统2D图像方法的隐私问题。提出使用3D点云和HFWS技术实现连续、隐私友好的情感识别，并构建了AffectNet3D数据集进行验证。**

- **链接: [https://arxiv.org/pdf/2603.27798](https://arxiv.org/pdf/2603.27798)**

> **作者:** Laura Rayón Ropero; Jasper De Laet; Filip Lemic; Pau Sabater Nácher; Nabeel Nisar Bhat; Sergi Abadal; Jeroen Famaey; Eduard Alarcón; Xavier Costa-Pérez
>
> **备注:** 18 pages, 12 figures, 2 tables. Accepted for publication at IEEE Transactions on Affective Computing
>
> **摘要:** Facial Emotion Recognition is a critical research area within Affective Computing due to its wide-ranging applications in Human Computer Interaction, mental health assessment and fatigue monitoring. Current FER methods predominantly rely on Deep Learning techniques trained on 2D image data, which pose significant privacy concerns and are unsuitable for continuous, real-time monitoring. As an alternative, we propose High-Frequency Wireless Sensing (HFWS) as an enabler of continuous, privacy-aware FER, through the generation of detailed 3D facial pointclouds via on-person sensors embedded in wearables. We present arguments supporting the privacy advantages of HFWS over traditional 2D imaging, particularly under increasingly stringent data protection regulations. A major barrier to adopting HFWS for FER is the scarcity of labeled 3D FER datasets. Towards addressing this issue, we introduce a FLAME-based method to generate 3D facial pointclouds from existing public 2D datasets. Using this approach, we create AffectNet3D, a 3D version of the AffectNet database. To evaluate the quality and usability of the generated data, we design a pointcloud refinement pipeline focused on isolating the facial region, and train the popular PointNet++ model on the refined pointclouds. Fine-tuning the model on a small subset of the unseen 3D FER dataset BU-3DFE yields a classification accuracy exceeding 70%, comparable to oracle-level performance. To further investigate the potential of HFWS-based FER for continuous monitoring, we simulate wearable sensing conditions by masking portions of the generated pointclouds. Experimental results show that models trained on AffectNet3D and fine-tuned with just 25% of BU-3DFE outperform those trained solely on BU-3DFE. These findings highlight the viability of our pipeline and support the feasibility of continuous, privacy-aware FER via wearable HFWS systems.
>
---
#### [new 074] LVRPO: Language-Visual Alignment with GRPO for Multimodal Understanding and Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.MA; cs.MM**

- **简介: 该论文提出LVRPO框架，解决多模态理解与生成中的语言-视觉对齐问题，通过强化学习直接优化模型行为，提升细粒度推理和可控生成能力。**

- **链接: [https://arxiv.org/pdf/2603.27693](https://arxiv.org/pdf/2603.27693)**

> **作者:** Shentong Mo; Sukmin Yun
>
> **摘要:** Unified multimodal pretraining has emerged as a promising paradigm for jointly modeling language and vision within a single foundation model. However, existing approaches largely rely on implicit or indirect alignment signals and remain suboptimal for simultaneously supporting multimodal understanding and generation, particularly in settings that require fine-grained language-visual reasoning and controllable generation. In this work, we propose LVRPO, a language-visual reinforcement-based preference optimization framework that explicitly aligns language and visual representations using Group Relative Policy Optimization (GRPO). Instead of introducing additional alignment losses at the representation level, LVRPO directly optimizes multimodal model behaviors through preference-driven reinforcement signals, encouraging consistent and semantically grounded interactions between language and vision across both understanding and generation tasks. This formulation enables effective alignment without requiring auxiliary encoders or handcrafted cross-modal objectives, and naturally extends to diverse multimodal capabilities. Empirically, LVRPO consistently outperforms strong unified-pretraining baselines on a broad suite of benchmarks spanning multimodal understanding, generation, and reasoning.
>
---
#### [new 075] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，解决图像级监督下像素级定位与可解释性问题。提出ReAL和CGRO模块，实现无需像素标签的精准异常定位与推理。**

- **链接: [https://arxiv.org/pdf/2603.27179](https://arxiv.org/pdf/2603.27179)**

> **作者:** Yizhou Jin; Yuezhu Feng; Jinjin Zhang; Peng Wang; Qingjie Liu; Yunhong Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Multimodal large language models (MLLMs) have recently demonstrated remarkable reasoning and perceptual abilities for anomaly detection. However, most approaches remain confined to image-level anomaly detection and textual reasoning, while pixel-level localization still relies on external vision modules and dense annotations. In this work, we activate the intrinsic reasoning potential of MLLMs to perform anomaly detection, pixel-level localization, and interpretable reasoning solely from image-level supervision, without any auxiliary components or pixel-wise labels. Specifically, we propose Reasoning-Driven Anomaly Localization (ReAL), which extracts anomaly-related tokens from the autoregressive reasoning process and aggregates their attention responses to produce pixel-level anomaly maps. We further introduce a Consistency-Guided Reasoning Optimization (CGRO) module that leverages reinforcement learning to align reasoning tokens with visual attentions, resulting in more coherent reasoning and accurate anomaly localization. Extensive experiments on four public benchmarks demonstrate that our method significantly improves anomaly detection, localization, and interpretability. Remarkably, despite relying solely on image-level supervision, our approach achieves performance competitive with MLLM-based methods trained under dense pixel-level supervision. Code is available at this https URL.
>
---
#### [new 076] Unified Number-Free Text-to-Motion Generation Via Flow Matching
- **分类: cs.CV**

- **简介: 该论文属于文本到运动生成任务，解决多智能体运动合成中数量不固定导致的泛化问题。提出UMF模型，通过分解生成阶段提升效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.27040](https://arxiv.org/pdf/2603.27040)**

> **作者:** Guanhe Huang; Oya Celiktutan
>
> **摘要:** Generative models excel at motion synthesis for a fixed number of agents but struggle to generalize with variable agents. Based on limited, domain-specific data, existing methods employ autoregressive models to generate motion recursively, which suffer from inefficiency and error accumulation. We propose Unified Motion Flow (UMF), which consists of Pyramid Motion Flow (P-Flow) and Semi-Noise Motion Flow (S-Flow). UMF decomposes the number-free motion generation into a single-pass motion prior generation stage and multi-pass reaction generation stages. Specifically, UMF utilizes a unified latent space to bridge the distribution gap between heterogeneous motion datasets, enabling effective unified training. For motion prior generation, P-Flow operates on hierarchical resolutions conditioned on different noise levels, thereby mitigating computational overheads. For reaction generation, S-Flow learns a joint probabilistic path that adaptively performs reaction transformation and context reconstruction, alleviating error accumulation. Extensive results and user studies demonstrate UMF' s effectiveness as a generalist model for multi-person motion generation from text. Project page: this https URL.
>
---
#### [new 077] Efficient Domain Adaptation for Text Line Recognition via Decoupled Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文本行识别任务，解决OCR领域域适应计算成本高的问题。通过解耦视觉检测与语言修正，提升效率并降低资源需求。**

- **链接: [https://arxiv.org/pdf/2603.28028](https://arxiv.org/pdf/2603.28028)**

> **作者:** Arundhathi Dev; Justin Zhan
>
> **备注:** Accepted to the International Conference on Machine Intelligence Theory and Applications (MiTA 2026)
>
> **摘要:** Optical character recognition remains critical infrastructure for document digitization, yet state-of-the-art performance is often restricted to well-resourced institutions by prohibitive computational barriers. End-to-end transformer architectures achieve strong accuracy but demand hundreds of GPU hours for domain adaptation, limiting accessibility for practitioners and digital humanities scholars. We present a modular detection-and-correction framework that achieves near-SOTA accuracy with single-GPU training. Our approach decouples lightweight visual character detection (domain-agnostic) from domain-specific linguistic correction using pretrained sequence models including T5, ByT5, and BART. By training the correctors entirely on synthetic noise, we enable annotation-free domain adaptation without requiring labeled target images. Evaluating across modern clean handwriting, cursive script, and historical documents, we identify a critical "Pareto frontier" in architecture selection: T5-Base excels on modern text with standard vocabulary, whereas ByT5-Base dominates on historical documents by reconstructing archaic spellings at the byte level. Our results demonstrate that this decoupled paradigm matches end-to-end transformer accuracy while reducing compute by approximately 95%, establishing a viable, resource-efficient alternative to monolithic OCR architectures.
>
---
#### [new 078] Decompose, Mix, Adapt: A Unified Framework for Parameter-Efficient Neural Network Recombination and Compression
- **分类: cs.CV**

- **简介: 该论文提出CRISP框架，解决参数重组中的模型压缩与高效微调问题，通过分解权重和共享基矩阵实现两者统一。**

- **链接: [https://arxiv.org/pdf/2603.27383](https://arxiv.org/pdf/2603.27383)**

> **作者:** Nazia Tasnim; Shrimai Prabhumoye; Bryan A. Plummer
>
> **备注:** Accepted in CVPR, 2026 (Main Track)
>
> **摘要:** Parameter Recombination (PR) methods aim to efficiently compose the weights of a neural network for applications like Parameter-Efficient FineTuning (PEFT) and Model Compression (MC), among others. Most methods typically focus on one application of PR, which can make composing them challenging. For example, when deploying a large model you may wish to compress the model and also quickly adapt to new settings. However, PEFT methods often can still contain millions of parameters. This may be small compared to the original model size, but can be problematic in resource constrained deployments like edge devices, where they take a larger portion of the compressed model's parameters. To address this, we present Coefficient-gated weight Recombination by Interpolated Shared basis Projections (CRISP), a general approach that seamlessly integrates multiple PR tasks within the same framework. CRISP accomplishes this by factorizing pretrained weights into basis matrices and their component mixing projections. Sharing basis matrices across layers and adjusting its size enables us to perform MC, whereas the mixer weight's small size (fewer than 200 in some experiments) enables CRISP to support PEFT. Experiments show CRISP outperforms methods from prior work capable of dual-task applications by 4-5\% while also outperforming the state-of-the-art in PEFT by 1.5\% and PEFT+MC combinations by 1\%. Our code is available on the repository: this https URL.
>
---
#### [new 079] DiffAttn: Diffusion-Based Drivers' Visual Attention Prediction with LLM-Enhanced Semantic Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉注意力预测任务，旨在提升智能车辆对驾驶场景的理解。通过扩散模型与大语言模型结合，增强对关键安全信息的感知和解释能力。**

- **链接: [https://arxiv.org/pdf/2603.28251](https://arxiv.org/pdf/2603.28251)**

> **作者:** Weimin Liu; Qingkun Li; Jiyuan Qiu; Wenjun Wang; Joshua H. Meng
>
> **摘要:** Drivers' visual attention provides critical cues for anticipating latent hazards and directly shapes decision-making and control maneuvers, where its absence can compromise traffic safety. To emulate drivers' perception patterns and advance visual attention prediction for intelligent vehicles, we propose DiffAttn, a diffusion-based framework that formulates this task as a conditional diffusion-denoising process, enabling more accurate modeling of drivers' attention. To capture both local and global scene features, we adopt Swin Transformer as encoder and design a decoder that combines a Feature Fusion Pyramid for cross-layer interaction with dense, multi-scale conditional diffusion to jointly enhance denoising learning and model fine-grained local and global scene contexts. Additionally, a large language model (LLM) layer is incorporated to enhance top-down semantic reasoning and improve sensitivity to safety-critical cues. Extensive experiments on four public datasets demonstrate that DiffAttn achieves state-of-the-art (SoTA) performance, surpassing most video-based, top-down-feature-driven, and LLM-enhanced baselines. Our framework further supports interpretable driver-centric scene understanding and has the potential to improve in-cabin human-machine interaction, risk perception, and drivers' state measurement in intelligent vehicles.
>
---
#### [new 080] Demo-Pose: Depth-Monocular Modality Fusion For Object Pose Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D目标姿态估计任务，解决从RGB-D输入中准确估计物体6D位姿和3D尺寸的问题。提出DeMo-Pose方法融合RGB语义与深度信息，提升姿态估计性能。**

- **链接: [https://arxiv.org/pdf/2603.27533](https://arxiv.org/pdf/2603.27533)**

> **作者:** Rachit Agarwal; Abhishek Joshi; Sathish Chalasani; Woo Jin Kim
>
> **备注:** Accepted at ICASSP 2026, 5 pages, 3 figures, 3 tables
>
> **摘要:** Object pose estimation is a fundamental task in 3D vision with applications in robotics, AR/VR, and scene understanding. We address the challenge of category-level 9-DoF pose estimation (6D pose + 3Dsize) from RGB-D input, without relying on CAD models during inference. Existing depth-only methods achieve strong results but ignore semantic cues from RGB, while many RGB-D fusion models underperform due to suboptimal cross-modal fusion that fails to align semantic RGB cues with 3D geometric representations. We propose DeMo-Pose, a hybrid architecture that fuses monocular semantic features with depth-based graph convolutional representations via a novel multimodal fusion strategy. To further improve geometric reasoning, we introduce a novel Mesh-Point Loss (MPL) that leverages mesh structure during training without adding inference overhead. Our approach achieves real-time inference and significantly improves over state-of-the-art methods across object categories, outperforming the strong GPV-Pose baseline by 3.2\% on 3D IoU and 11.1\% on pose accuracy on the REAL275 benchmark. The results highlight the effectiveness of depth-RGB fusion and geometry-aware learning, enabling robust category-level 3D pose estimation for real-world applications.
>
---
#### [new 081] Learning Multi-View Spatial Reasoning from Cross-View Relations
- **分类: cs.CV**

- **简介: 该论文研究多视角空间推理任务，旨在提升视觉-语言模型在3D环境中的多视角理解与操作能力。通过构建XVR数据集，增强模型在对应、验证和定位任务上的表现。**

- **链接: [https://arxiv.org/pdf/2603.27967](https://arxiv.org/pdf/2603.27967)**

> **作者:** Suchae Jeong; Jaehwi Song; Haeone Lee; Hanna Kim; Jian Kim; Dongjun Lee; Dong Kyu Shin; Changyeon Kim; Dongyoon Hahm; Woogyeol Jin; Juheon Choi; Kimin Lee
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Vision-language models (VLMs) have achieved impressive results on single-view vision tasks, but lack the multi-view spatial reasoning capabilities essential for embodied AI systems to understand 3D environments and manipulate objects across different viewpoints. In this work, we introduce Cross-View Relations (XVR), a large-scale dataset designed to teach VLMs spatial reasoning across multiple views. XVR comprises 100K vision-question-answer samples derived from 18K diverse 3D scenes and 70K robotic manipulation trajectories, spanning three fundamental spatial reasoning tasks: Correspondence (matching objects across views), Verification (validating spatial relationships), and Localization (identifying object positions). VLMs fine-tuned on XVR achieve substantial improvements on established multi-view and robotic spatial reasoning benchmarks (MindCube and RoboSpatial). When integrated as backbones in Vision-Language-Action models, XVR-trained representations improve success rates on RoboCasa. Our results demonstrate that explicit training on cross-view spatial relations significantly enhances multi-view reasoning and transfers effectively to real-world robotic manipulation.
>
---
#### [new 082] TwinMixing: A Shuffle-Aware Feature Interaction Model for Multi-Task Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TwinMixing模型，用于自动驾驶中的多任务分割（可行驶区域和车道分割）。旨在提升分割精度与实时性，通过共享编码器和任务专用解码器实现高效特征交互。**

- **链接: [https://arxiv.org/pdf/2603.28233](https://arxiv.org/pdf/2603.28233)**

> **作者:** Minh-Khoi Do; Huy Che; Dinh-Duy Phan; Duc-Khai Lam; Duc-Lung Vu
>
> **摘要:** Accurate and efficient perception is essential for autonomous driving, where segmentation tasks such as drivable-area and lane segmentation provide critical cues for motion planning and control. However, achieving high segmentation accuracy while maintaining real-time performance on low-cost hardware remains a challenging problem. To address this issue, we introduce TwinMixing, a lightweight multi-task segmentation model designed explicitly for drivable-area and lane segmentation. The proposed network features a shared encoder and task-specific decoders, enabling both feature sharing and task specialization. Within the encoder, we propose an Efficient Pyramid Mixing (EPM) module that enhances multi-scale feature extraction through a combination of grouped convolutions, depthwise dilated convolutions and channel shuffle operations, effectively expanding the receptive field while minimizing computational cost. Each decoder adopts a Dual-Branch Upsampling (DBU) Block composed of a learnable transposed convolution-based Fine detailed branch and a parameter-free bilinear interpolation-based Coarse grained branch, achieving detailed yet spatially consistent feature reconstruction. Extensive experiments on the BDD100K dataset validate the effectiveness of TwinMixing across three configurations - tiny, base, and large. Among them, the base configuration achieves the best trade-off between accuracy and computational efficiency, reaching 92.0% mIoU for drivable-area segmentation and 32.3% IoU for lane segmentation with only 0.43M parameters and 3.95 GFLOPs. Moreover, TwinMixing consistently outperforms existing segmentation models on the same tasks, as illustrated in Fig. 1. Thanks to its compact and modular design, TwinMixing demonstrates strong potential for real-time deployment in autonomous driving and embedded perception systems. The source code: this https URL.
>
---
#### [new 083] TerraSky3D: Multi-View Reconstructions of European Landmarks in 4K
- **分类: cs.CV**

- **简介: 该论文提出TerraSky3D数据集，解决3D重建数据不足的问题，包含高分辨率欧洲地标多视角图像及配套信息，用于训练和评估3D重建算法。**

- **链接: [https://arxiv.org/pdf/2603.28287](https://arxiv.org/pdf/2603.28287)**

> **作者:** Mattia D'Urso; Yuxi Hu; Christian Sormann; Mattia Rossi; Friedrich Fraundorfer
>
> **备注:** Accepted at 3DMV at CVPR Workshop 2026
>
> **摘要:** Despite the growing need for data of more and more sophisticated 3D reconstruction pipelines, we can still observe a scarcity of suitable public datasets. Existing 3D datasets are either low resolution, limited to a small amount of scenes, based on images of varying quality because retrieved from the internet, or limited to specific capturing scenarios. Motivated by this lack of suitable 3D datasets, we captured TerraSky3D, a high-resolution large-scale 3D reconstruction dataset comprising 50,000 images divided into 150 ground, aerial, and mixed scenes. The dataset focuses on European landmarks and comes with curated calibration data, camera poses, and depth maps. TerraSky3D tries to answer the need for challenging dataset that can be used to train and evaluate 3D reconstruction-related pipelines.
>
---
#### [new 084] TDEC: Deep Embedded Image Clustering with Transformer and Distribution Information
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像聚类任务，解决深度聚类中特征表示与维度不友好、缺乏全局信息融合的问题。提出TDEC方法，结合Transformer和分布信息，提升聚类性能。**

- **链接: [https://arxiv.org/pdf/2603.26746](https://arxiv.org/pdf/2603.26746)**

> **作者:** Ruilin Zhang; Haiyang Zheng; Hongpeng Wang
>
> **摘要:** Image clustering is a crucial but challenging task in multimedia machine learning. Recently the combination of clustering with deep learning has achieved promising performance against conventional methods on high-dimensional image data. Unfortunately, existing deep clustering methods (DC) often ignore the importance of information fusion with a global perception field among different image regions on clustering images, especially complex ones. Additionally, the learned features are usually clustering-unfriendly in terms of dimensionality and are based only on simple distance information for the clustering. In this regard, we propose a deep embedded image clustering TDEC, which for the first time to our knowledge, jointly considers feature representation, dimensional preference, and robust assignment for image clustering. Specifically, we introduce the Transformer to form a novel module T-Encoder to learn discriminative features with global dependency while using the Dim-Reduction block to build a friendly low-dimensional space favoring clustering. Moreover, the distribution information of embedded features is considered in the clustering process to provide reliable supervised signals for joint training. Our method is robust and allows for more flexibility in data size, the number of clusters, and the context complexity. More importantly, the clustering performance of TDEC is much higher than recent competitors. Extensive experiments with state-of-the-art approaches on complex datasets show the superiority of TDEC.
>
---
#### [new 085] LOME: Learning Human-Object Manipulation with Action-Conditioned Egocentric World Model
- **分类: cs.CV**

- **简介: 该论文提出LOME，用于生成基于动作的人机交互视频，解决真实物理交互和泛化问题。属于人机操作生成任务。**

- **链接: [https://arxiv.org/pdf/2603.27449](https://arxiv.org/pdf/2603.27449)**

> **作者:** Quankai Gao; Jiawei Yang; Qiangeng Xu; Le Chen; Yue Wang
>
> **摘要:** Learning human-object manipulation presents significant challenges due to its fine-grained and contact-rich nature of the motions involved. Traditional physics-based animation requires extensive modeling and manual setup, and more importantly, it neither generalizes well across diverse object morphologies nor scales effectively to real-world environment. To address these limitations, we introduce LOME, an egocentric world model that can generate realistic human-object interactions as videos conditioned on an input image, a text prompt, and per-frame human actions, including both body poses and hand gestures. LOME injects strong and precise action guidance into object manipulation by jointly estimating spatial human actions and the environment contexts during training. After finetuning a pretrained video generative model on videos of diverse egocentric human-object interactions, LOME demonstrates not only high action-following accuracy and strong generalization to unseen scenarios, but also realistic physical consequences of hand-object interactions, e.g., liquid flowing from a bottle into a mug after executing a ``pouring'' action. Extensive experiments demonstrate that our video-based framework significantly outperforms state-of-the-art image based and video-based action-conditioned methods and Image/Text-to-Video (I/T2V) generative model in terms of both temporal consistency and motion control. LOME paves the way for photorealistic AR/VR experiences and scalable robotic training, without being limited to simulated environments or relying on explicit 3D/4D modeling.
>
---
#### [new 086] From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文提出NAS3R，解决无监督的3D重建问题，通过自监督学习联合学习3D几何与相机参数，无需标注数据。**

- **链接: [https://arxiv.org/pdf/2603.27455](https://arxiv.org/pdf/2603.27455)**

> **作者:** Ranran Huang; Weixun Luo; Ye Mao; Krystian Mikolajczyk
>
> **摘要:** In this paper, we introduce NAS3R, a self-supervised feed-forward framework that jointly learns explicit 3D geometry and camera parameters with no ground-truth annotations and no pretrained priors. During training, NAS3R reconstructs 3D Gaussians from uncalibrated and unposed context views and renders target views using its self-predicted camera parameters, enabling self-supervised training from 2D photometric supervision. To ensure stable convergence, NAS3R integrates reconstruction and camera prediction within a shared transformer backbone regulated by masked attention, and adopts a depth-based Gaussian formulation that facilitates well-conditioned optimization. The framework is compatible with state-of-the-art supervised 3D reconstruction architectures and can incorporate pretrained priors or intrinsic information when available. Extensive experiments show that NAS3R achieves superior results to other self-supervised methods, establishing a scalable and geometry-aware paradigm for 3D reconstruction from unconstrained data. Code and models are publicly available at this https URL.
>
---
#### [new 087] CLIP-AUTT: Test-Time Personalization with Action Unit Prompting for Fine-Grained Video Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文属于视频情感识别任务，解决细粒度情感识别中个性化不足的问题。通过引入动作单元作为提示，提出CLIP-AU和CLIP-AUTT方法，实现更精准的情感识别与个性化适应。**

- **链接: [https://arxiv.org/pdf/2603.27999](https://arxiv.org/pdf/2603.27999)**

> **作者:** Muhammad Osama Zeeshan; Masoumeh Sharafi; Benoît Savary; Alessandro Lameiras Koerich; Marco Pedersoli; Eric Granger
>
> **摘要:** Personalization in emotion recognition (ER) is essential for an accurate interpretation of subtle and subject-specific expressive patterns. Recent advances in vision-language models (VLMs) such as CLIP demonstrate strong potential for leveraging joint image-text representations in ER. However, CLIP-based methods either depend on CLIP's contrastive pretraining or on LLMs to generate descriptive text prompts, which are noisy, computationally expensive, and fail to capture fine-grained expressions, leading to degraded performance. In this work, we leverage Action Units (AUs) as structured textual prompts within CLIP to model fine-grained facial expressions. AUs encode the subtle muscle activations underlying expressions, providing localized and interpretable semantic cues for more robust ER. We introduce CLIP-AU, a lightweight AU-guided temporal learning method that integrates interpretable AU semantics into CLIP. It learns generic, subject-agnostic representations by aligning AU prompts with facial dynamics, enabling fine-grained ER without CLIP fine-tuning or LLM-generated text supervision. Although CLIP-AU models fine-grained AU semantics, it does not adapt to subject-specific variability in subtle expressions. To address this limitation, we propose CLIP-AUTT, a video-based test-time personalization method that dynamically adapts AU prompts to videos from unseen subjects. By combining entropy-guided temporal window selection with prompt tuning, CLIP-AUTT enables subject-specific adaptation while preserving temporal consistency. Our extensive experiments on three challenging video-based subtle ER datasets, BioVid, StressID, and BAH, indicate that CLIP-AU and CLIP-AUTT outperform state-of-the-art CLIP-based FER and TTA methods, achieving robust and personalized subtle ER.
>
---
#### [new 088] Beyond Textual Knowledge-Leveraging Multimodal Knowledge Bases for Enhancing Vision-and-Language Navigation
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于视觉语言导航任务，旨在解决语义线索提取与视觉对齐困难的问题。通过整合多模态知识库，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.26859](https://arxiv.org/pdf/2603.26859)**

> **作者:** Dongsheng Yang; Yinfeng Yu; Liejun Wang
>
> **备注:** Main paper (37 pages). Accepted for publication by the Information Processing and Management,Volume 63,Issue 6,September 2026,104766
>
> **摘要:** Vision-and-Language Navigation (VLN) requires an agent to navigate through complex unseen environments based on natural language instructions. However, existing methods often struggle to effectively capture key semantic cues and accurately align them with visual observations. To address this limitation, we propose Beyond Textual Knowledge (BTK), a VLN framework that synergistically integrates environment-specific textual knowledge with generative image knowledge bases. BTK employs Qwen3-4B to extract goal-related phrases and utilizes Flux-Schnell to construct two large-scale image knowledge bases: R2R-GP and REVERIE-GP. Additionally, we leverage BLIP-2 to construct a large-scale textual knowledge base derived from panoramic views, providing environment-specific semantic cues. These multimodal knowledge bases are effectively integrated via the Goal-Aware Augmentor and Knowledge Augmentor, significantly enhancing semantic grounding and cross-modal alignment. Extensive experiments on the R2R dataset with 7,189 trajectories and the REVERIE dataset with 21,702 instructions demonstrate that BTK significantly outperforms existing baselines. On the test unseen splits of R2R and REVERIE, SR increased by 5% and 2.07% respectively, and SPL increased by 4% and 3.69% respectively. The source code is available at this https URL.
>
---
#### [new 089] A training-free framework for high-fidelity appearance transfer via diffusion transformers
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决扩散模型在参考图像编辑中的结构破坏问题。提出无需训练的框架，通过分离结构与外观实现高保真迁移。**

- **链接: [https://arxiv.org/pdf/2603.26767](https://arxiv.org/pdf/2603.26767)**

> **作者:** Shengrong Gu; Ye Wang; Song Wu; Rui Ma; Qian Wang; Lanjun Wang; Zili Yi
>
> **摘要:** Diffusion Transformers (DiTs) excel at generation, but their global self-attention makes controllable, reference-image-based editing a distinct challenge. Unlike U-Nets, naively injecting local appearance into a DiT can disrupt its holistic scene structure. We address this by proposing the first training-free framework specifically designed to tame DiTs for high-fidelity appearance transfer. Our core is a synergistic system that disentangles structure and appearance. We leverage high-fidelity inversion to establish a rich content prior for the source image, capturing its lighting and micro-textures. A novel attention-sharing mechanism then dynamically fuses purified appearance features from a reference, guided by geometric priors. Our unified approach operates at 1024px and outperforms specialized methods on tasks ranging from semantic attribute transfer to fine-grained material application. Extensive experiments confirm our state-of-the-art performance in both structural preservation and appearance fidelity.
>
---
#### [new 090] SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文提出SpatialStack，解决3D空间推理问题，通过多层几何-语言融合提升视觉语言模型的3D理解能力。**

- **链接: [https://arxiv.org/pdf/2603.27437](https://arxiv.org/pdf/2603.27437)**

> **作者:** Jiang Zhang; Shijie Zhou; Bangya Liu; Achuta Kadambi; Zhiwen Fan
>
> **备注:** CVPR 2026, Project Website: this https URL
>
> **摘要:** Large vision-language models (VLMs) still struggle with reliable 3D spatial reasoning, a core capability for embodied and physical AI systems. This limitation arises from their inability to capture fine-grained 3D geometry and spatial relationships. While recent efforts have introduced multi-view geometry transformers into VLMs, they typically fuse only the deep-layer features from vision and geometry encoders, discarding rich hierarchical signals and creating a fundamental bottleneck for spatial understanding. To overcome this, we propose SpatialStack, a general hierarchical fusion framework that progressively aligns vision, geometry, and language representations across the model hierarchy. Moving beyond conventional late-stage vision-geometry fusion, SpatialStack stacks and synchronizes multi-level geometric features with the language backbone, enabling the model to capture both local geometric precision and global contextual semantics. Building upon this framework, we develop VLM-SpatialStack, a model that achieves state-of-the-art performance on multiple 3D spatial reasoning benchmarks. Extensive experiments and ablations demonstrate that our multi-level fusion strategy consistently enhances 3D understanding and generalizes robustly across diverse spatial reasoning tasks, establishing SpatialStack as an effective and extensible design paradigm for vision-language-geometry integration in next-generation multimodal physical AI systems.
>
---
#### [new 091] Envisioning global urban development with satellite imagery and generative AI
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于城市规划任务，旨在解决传统预测方法无法反映生成性发展的不足。通过构建多模态生成AI框架，生成高质量的全球城市卫星图像，支持可持续城市发展和场景化规划。**

- **链接: [https://arxiv.org/pdf/2603.26831](https://arxiv.org/pdf/2603.26831)**

> **作者:** Kailai Sun; Yuebing Liang; Mingyi He; Yunhan Zheng; Alok Prakash; Shenhao Wang; Jinhua Zhao; Alex "Sandy'' Pentland
>
> **摘要:** Urban development has been a defining force in human history, shaping cities for centuries. However, past studies mostly analyze such development as predictive tasks, failing to reflect its generative nature. Therefore, this study designs a multimodal generative AI framework to envision sustainable urban development at a global scale. By integrating prompts and geospatial controls, our framework can generate high-fidelity, diverse, and realistic urban satellite imagery across the 500 largest metropolitan areas worldwide. It enables users to specify urban development goals, creating new images that align with them while offering diverse scenarios whose appearance can be controlled with text prompts and geospatial constraints. It also facilitates urban redevelopment practices by learning from the surrounding environment. Beyond visual synthesis, we find that it encodes and interprets latent representations of urban form for global cross-city learning, successfully transferring styles of urban environments across a global spatial network. The latent representations can also enhance downstream prediction tasks such as carbon emission prediction. Further, human expert evaluation confirms that our generated urban images are comparable to real urban images. Overall, this study presents innovative approaches for accelerated urban planning and supports scenario-based planning processes for worldwide cities.
>
---
#### [new 092] PANDORA: Pixel-wise Attention Dissolution and Latent Guidance for Zero-Shot Object Removal
- **分类: cs.CV**

- **简介: 该论文提出PANDORA框架，用于零样本物体移除任务，解决背景一致性与多物体移除难题，通过注意力机制优化实现高效、精准的图像修复。**

- **链接: [https://arxiv.org/pdf/2603.27555](https://arxiv.org/pdf/2603.27555)**

> **作者:** Dinh-Khoi Vo; Van-Loc Nguyen; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **备注:** ICME 2026
>
> **摘要:** Removing objects from natural images is challenging due to difficulty of synthesizing semantically coherent content while preserving background integrity. Existing methods often rely on fine-tuning, prompt engineering, or inference-time optimization, yet still suffer from texture inconsistency, rigid artifacts, weak foreground-background disentanglement, and poor scalability for multi-object removal. We propose a novel zero-shot object removal framework, namely PANDORA, that operates directly on pre-trained text-to-image diffusion models, requiring no fine-tuning, prompts, or optimization. We propose Pixel-wise Attention Dissolution to remove object by nullifying the most correlated attention keys for masked pixels, effectively eliminating the object from self-attention flow and allowing background context to dominate reconstruction. We further introduce Localized Attentional Disentanglement Guidance to steer denoising toward latent manifolds favorable to clean object removal. Together, these components enable precise, non-rigid, prompt-free, and scalable multi-object erasure in a single pass. Experiments demonstrate superior visual fidelity and semantic plausibility compared to state-of-the-art methods. The project page is available at this https URL.
>
---
#### [new 093] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR
- **分类: cs.CV**

- **简介: 该论文属于LiDAR地面分割任务，解决传统方法依赖人工标注或特定传感器配置的问题。提出TerraSeg模型，利用自监督学习实现通用、实时的地面分割。**

- **链接: [https://arxiv.org/pdf/2603.27344](https://arxiv.org/pdf/2603.27344)**

> **作者:** Ted Lentsch; Santiago Montiel-Marín; Holger Caesar; Dariu M. Gavrila
>
> **备注:** CVPR 2026
>
> **摘要:** LiDAR perception is fundamental to robotics, enabling machines to understand their environment in 3D. A crucial task for LiDAR-based scene understanding and navigation is ground segmentation. However, existing methods are either handcrafted for specific sensor configurations or rely on costly per-point manual labels, severely limiting their generalization and scalability. To overcome this, we introduce TerraSeg, the first self-supervised, domain-agnostic model for LiDAR ground segmentation. We train TerraSeg on OmniLiDAR, a unified large-scale dataset that aggregates and standardizes data from 12 major public benchmarks. Spanning almost 22 million raw scans across 15 distinct sensor models, OmniLiDAR provides unprecedented diversity for learning a highly generalizable ground model. To supervise training without human annotations, we propose PseudoLabeler, a novel module that generates high-quality ground and non-ground labels through self-supervised per-scan runtime optimization. Extensive evaluations demonstrate that, despite using no manual labels, TerraSeg achieves state-of-the-art results on nuScenes, SemanticKITTI, and Waymo Perception while delivering real-time performance. Our code and model weights are publicly available.
>
---
#### [new 094] Attention Frequency Modulation: Training-Free Spectral Modulation of Diffusion Cross-Attention
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，解决扩散模型中跨注意力机制的控制问题。提出AFM方法，在不重新训练的情况下，通过频域调整注意力分布，实现对生成图像的可控编辑。**

- **链接: [https://arxiv.org/pdf/2603.28114](https://arxiv.org/pdf/2603.28114)**

> **作者:** Seunghun Oh; Unsang Park
>
> **备注:** 16 pages; preprint
>
> **摘要:** Cross-attention is the primary interface through which text conditions latent diffusion models, yet its step-wise multi-resolution dynamics remain under-characterized, limiting principled training-free control. We cast diffusion cross-attention as a spatiotemporal signal on the latent grid by summarizing token-softmax weights into token-agnostic concentration maps and tracking their radially binned Fourier power over denoising. Across prompts and seeds, encoder cross-attention exhibits a consistent coarse-to-fine spectral progression, yielding a stable time-frequency fingerprint of token competition. Building on this structure, we introduce Attention Frequency Modulation (AFM), a plug-and-play inference-time intervention that edits token-wise pre-softmax cross-attention logits in the Fourier domain: low- and high-frequency bands are reweighted with a progress-aligned schedule and can be adaptively gated by token-allocation entropy, before the token softmax. AFM provides a continuous handle to bias the spatial scale of token-competition patterns without retraining, prompt editing, or parameter updates. Experiments on Stable Diffusion show that AFM reliably redistributes attention spectra and produces substantial visual edits while largely preserving semantic alignment. Finally, we find that entropy mainly acts as an adaptive gain on the same frequency-based edit rather than an independent control axis.
>
---
#### [new 095] PhyDCM: A Reproducible Open-Source Framework for AI-Assisted Brain Tumor Classification from Multi-Sequence MRI
- **分类: cs.CV; cs.AI**

- **简介: 论文提出PhyDCM框架，用于脑肿瘤MRI分类。解决传统方法难以复现及扩展的问题，整合混合架构与标准化处理，提升诊断准确性与系统灵活性。**

- **链接: [https://arxiv.org/pdf/2603.26794](https://arxiv.org/pdf/2603.26794)**

> **作者:** Hayder Saad Abdulbaqi; Mohammed Hadi Rahim; Mohammed Hassan Hadi; Haider Ali Aboud; Ali Hussein Allawi
>
> **备注:** 18 pages, 9 figures, 6 tables
>
> **摘要:** MRI-based medical imaging has become indispensable in modern clinical diagnosis, particularly for brain tumor detection. However, the rapid growth in data volume poses challenges for conventional diagnostic approaches. Although deep learning has shown strong performance in automated classification, many existing solutions are confined to closed technical architectures, limiting reproducibility and further academic development. PhyDCM is introduced as an open-source software framework that integrates a hybrid classification architecture based on MedViT with standardized DICOM processing and an interactive desktop visualization interface. The system is designed as a modular digital library that separates computational logic from the graphical interface, allowing independent modification and extension of components. Standardized preprocessing, including intensity rescaling and limited data augmentation, ensures consistency across varying MRI acquisition settings. Experimental evaluation on MRI datasets from BRISC2025 and curated Kaggle collections (FigShare, SARTAJ, and Br35H) demonstrates stable diagnostic performance, achieving over 93% classification accuracy across categories. The framework supports structured, exportable outputs and multi-planar reconstruction of volumetric data. By emphasizing transparency, modularity, and accessibility, PhyDCM provides a practical foundation for reproducible AI-driven medical image analysis, with flexibility for future integration of additional imaging modalities.
>
---
#### [new 096] MathGen: Revealing the Illusion of Mathematical Competence through Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于数学视觉生成任务，研究生成模型在需要可视化表达数学解时的表现。工作包括构建基准测试集和验证工具，评估不同模型的数学准确性。**

- **链接: [https://arxiv.org/pdf/2603.27959](https://arxiv.org/pdf/2603.27959)**

> **作者:** Ruiyao Liu; Hui Shen; Ping Zhang; Yunta Hsieh; Yifan Zhang; Jing Xu; Sicheng Chen; Junchen Li; Jiawei Lu; Jianing Ma; Jiaqi Mo; Qi Han; Zhen Zhang; Zhongwei Wan; Jing Xiong; Xin Wang; Ziyuan Liu; Hangrui Cao; Ngai Wong
>
> **摘要:** Modern generative models have demonstrated the ability to solve challenging mathematical problems. In many real-world settings, however, mathematical solutions must be expressed visually through diagrams, plots, geometric constructions, and structured symbolic layouts, where correctness depends on precise visual composition. Can generative models still do so when the answer must be rendered visually rather than written in text? To study this problem, we introduce MathGen, a rigorous benchmark of 900 problems spanning seven core domains, each paired with an executable verifier under a Script-as-a-Judge protocol for deterministic and objective evaluation. Experiments on representative open-source and proprietary text-to-image models show that mathematical fidelity remains a major bottleneck: even the best closed-source model reaches only 42.0% overall accuracy, while open-source models achieve just ~ 1-11%, often near 0% on structured tasks. Overall, current T2I models remain far from competent at even elementary mathematical visual generation.
>
---
#### [new 097] ResAdapt: Adaptive Resolution for Efficient Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ResAdapt，解决多模态大模型中视觉分辨率与时间上下文难以兼顾的问题。通过输入侧自适应分配视觉预算，提升效率与准确率。属于多模态推理任务。**

- **链接: [https://arxiv.org/pdf/2603.28610](https://arxiv.org/pdf/2603.28610)**

> **作者:** Huanxuan Liao; Zhongtao Jiang; Yupu Hao; Yuqiao Tan; Shizhu He; Jun Zhao; Kun Xu; Kang Liu
>
> **备注:** work in progress
>
> **摘要:** Multimodal Large Language Models (MLLMs) achieve stronger visual understanding by scaling input fidelity, yet the resulting visual token growth makes jointly sustaining high spatial resolution and long temporal context prohibitive. We argue that the bottleneck lies not in how post-encoding representations are compressed but in the volume of pixels the encoder receives, and address it with ResAdapt, an Input-side adaptation framework that learns how much visual budget each frame should receive before encoding. ResAdapt couples a lightweight Allocator with an unchanged MLLM backbone, so the backbone retains its native visual-token interface while receiving an operator-transformed input. We formulate allocation as a contextual bandit and train the Allocator with Cost-Aware Policy Optimization (CAPO), which converts sparse rollout feedback into a stable accuracy-cost learning signal. Across budget-controlled video QA, temporal grounding, and image reasoning tasks, ResAdapt improves low-budget operating points and often lies on or near the efficiency-accuracy frontier, with the clearest gains on reasoning-intensive benchmarks under aggressive compression. Notably, ResAdapt supports up to 16x more frames at the same visual budget while delivering over 15% performance gain. Code is available at this https URL.
>
---
#### [new 098] Complet4R: Geometric Complete 4D Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出Complet4R，解决动态场景的4D几何重建问题，通过统一框架实现完整、时序一致的重建。**

- **链接: [https://arxiv.org/pdf/2603.27300](https://arxiv.org/pdf/2603.27300)**

> **作者:** Weibang Wang; Kenan Li; Zhuoguang Chen; Yijun Yuan; Hang Zhao
>
> **摘要:** We introduce Complet4R, a novel end-to-end framework for Geometric Complete 4D Reconstruction, which aims to recover temporally coherent and geometrically complete reconstruction for dynamic scenes. Our method formalizes the task of Geometric Complete 4D Reconstruction as a unified framework of reconstruction and completion, by directly accumulating full contexts onto each frame. Unlike previous approaches that rely on pairwise reconstruction or local motion estimation, Complet4R utilizes a decoder-only transformer to operate all context globally directly from sequential video input, reconstructing a complete geometry for every single timestamp, including occluded regions visible in other frames. Our method demonstrates the state-of-the-art performance on our proposed benchmark for Geometric Complete 4D Reconstruction and the 3D Point Tracking task. Code will be released to support future research.
>
---
#### [new 099] Incentivizing Temporal-Awareness in Egocentric Video Understanding Models
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决多模态大语言模型在第一人称视频中缺乏时间感知的问题。通过提出TGPO算法，增强模型的时间连贯性与因果推理能力。**

- **链接: [https://arxiv.org/pdf/2603.27184](https://arxiv.org/pdf/2603.27184)**

> **作者:** Zhiyang Xu; Tian Qin; Bowen Jin; Zhengfeng Lai; Meng Cao; Lifu Huang; Peng Zhang
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Multimodal large language models (MLLMs) have recently shown strong performance in visual understanding, yet they often lack temporal awareness, particularly in egocentric settings where reasoning depends on the correct ordering and evolution of events. This deficiency stems in part from training objectives that fail to explicitly reward temporal reasoning and instead rely on frame-level spatial shortcuts. To address this limitation, we propose Temporal Global Policy Optimization (TGPO), a reinforcement learning with verifiable rewards (RLVR) algorithm designed to incentivize temporal awareness in MLLMs. TGPO contrasts model outputs generated from temporally ordered versus shuffled video frames to derive calibrated, globally normalized reward signals that explicitly favor temporally coherent reasoning. Integrated with GRPO and GSPO, TGPO supports cold-start RL training and effectively suppresses spatial shortcut behaviors learned by existing MLLMs. Experiments across five egocentric video benchmarks demonstrate that TGPO consistently improves temporal grounding and causal coherence, outperforming prior RL-based video reasoning approaches. Our results suggest that TGPO offers a simple and scalable pathway toward temporally robust MLLMs for egocentric video understanding.
>
---
#### [new 100] TGIF2: Extended Text-Guided Inpainting Forgery Dataset & Benchmark
- **分类: cs.CV; cs.AI; cs.CR; cs.MM**

- **简介: 该论文属于图像伪造检测任务，旨在解决生成式图像修复带来的检测难题。工作包括构建TGIF2数据集，评估现有方法在新型修复上的表现，并揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2603.28613](https://arxiv.org/pdf/2603.28613)**

> **作者:** Hannes Mareen; Dimitrios Karageorgiou; Paschalis Giakoumoglou; Peter Lambert; Symeon Papadopoulos; Glenn Van Wallendael
>
> **备注:** 33 pages, accepted at Journal on Information Security
>
> **摘要:** Generative AI has made text-guided inpainting a powerful image editing tool, but at the same time a growing challenge for media forensics. Existing benchmarks, including our text-guided inpainting forgery (TGIF) dataset, show that image forgery localization (IFL) methods can localize manipulations in spliced images but struggle not in fully regenerated (FR) images, while synthetic image detection (SID) methods can detect fully regenerated images but cannot perform localization. With new generative inpainting models emerging and the open problem of localization in FR images remaining, updated datasets and benchmarks are needed. We introduce TGIF2, an extended version of TGIF, that captures recent advances in text-guided inpainting and enables a deeper analysis of forensic robustness. TGIF2 augments the original dataset with edits generated by FLUX.1 models, as well as with random non-semantic masks. Using the TGIF2 dataset, we conduct a forensic evaluation spanning IFL and SID, including fine-tuning IFL methods on FR images and generative super-resolution attacks. Our experiments show that both IFL and SID methods degrade on FLUX.1 manipulations, highlighting limited generalization. Additionally, while fine-tuning improves localization on FR images, evaluation with random non-semantic masks reveals object bias. Furthermore, generative super-resolution significantly weakens forensic traces, demonstrating that common image enhancement operations can undermine current forensic pipelines. In summary, TGIF2 provides an updated dataset and benchmark, which enables new insights into the challenges posed by modern inpainting and AI-based image enhancements. TGIF2 is available at this https URL.
>
---
#### [new 101] Towards Context-Aware Image Anonymization with Multi-Agent Reasoning
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于图像匿名化任务，旨在解决现有方法过处理或遗漏隐私信息的问题。提出CAIAMAR框架，结合多智能体推理与扩散模型，提升隐私保护效果并保持图像质量。**

- **链接: [https://arxiv.org/pdf/2603.27817](https://arxiv.org/pdf/2603.27817)**

> **作者:** Robert Aufschläger; Jakob Folz; Gautam Savaliya; Manjitha D Vidanalage; Michael Heigl; Martin Schramm
>
> **备注:** Accepted to IEEE CVPR 2026 GRAIL-V Workshop
>
> **摘要:** Street-level imagery contains personally identifiable information (PII), some of which is context-dependent. Existing anonymization methods either over-process images or miss subtle identifiers, while API-based solutions compromise data sovereignty. We present an agentic framework CAIAMAR (\underline{C}ontext-\underline{A}ware \underline{I}mage \underline{A}nonymization with \underline{M}ulti-\underline{A}gent \underline{R}easoning) for context-aware PII segmentation with diffusion-based anonymization, combining pre-defined processing for high-confidence cases with multi-agent reasoning for indirect identifiers. Three specialized agents coordinate via round-robin speaker selection in a Plan-Do-Check-Act (PDCA) cycle, enabling large vision-language models to classify PII based on spatial context (private vs. public property) rather than rigid category rules. The agents implement spatially-filtered coarse-to-fine detection where a scout-and-zoom strategy identifies candidates, open-vocabulary segmentation processes localized crops, and $IoU$-based deduplication ($30\%$ threshold) prevents redundant processing. Modal-specific diffusion guidance with appearance decorrelation substantially reduces re-identification (Re-ID) risks. On CUHK03-NP, our method reduces person Re-ID risk by $73\%$ ($R1$: $16.9\%$ vs. $62.4\%$ baseline). For image quality preservation on CityScapes, we achieve KID: $0.001$, and FID: $9.1$, significantly outperforming existing anonymization. The agentic workflow detects non-direct PII instances across object categories, and downstream semantic segmentation is preserved. Operating entirely on-premise with open-source models, the framework generates human-interpretable audit trails supporting EU's GDPR transparency requirements while flagging failed cases for human review.
>
---
#### [new 102] Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs
- **分类: cs.CV**

- **简介: 该论文属于图像到点云配准任务，解决跨模态特征难以通用的问题。通过构建异构图，增强特征交互并优化对应关系，提升配准性能。**

- **链接: [https://arxiv.org/pdf/2603.27969](https://arxiv.org/pdf/2603.27969)**

> **作者:** Pei An; Junfeng Ding; Jiaqi Yang; Yulong Wang; Jie Ma; Liangliang Nan
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Image-to-point-cloud (I2P) registration aims to align 2D images with 3D point clouds by establishing reliable 2D-3D correspondences. The drastic modality gap between images and point clouds makes it challenging to learn features that are both discriminative and generalizable, leading to severe performance drops in unseen scenarios. We address this challenge by introducing a heterogeneous graph that enables refining both cross-modal features and correspondences within a unified architecture. The proposed graph represents a mapping between segmented 2D and 3D regions, which enhances cross-modal feature interaction and thus improves feature discriminability. In addition, modeling the consistency among vertices and edges within the graph enables pruning of unreliable correspondences. Building on these insights, we propose a heterogeneous graph embedded I2P registration method, termed Hg-I2P. It learns a heterogeneous graph by mining multi-path feature relationships, adapts features under the guidance of heterogeneous edges, and prunes correspondences using graph-based projection consistency. Experiments on six indoor and outdoor benchmarks under cross-domain setups demonstrate that Hg-I2P significantly outperforms existing methods in both generalization and accuracy. Code is released on this https URL.
>
---
#### [new 103] OmniColor: A Unified Framework for Multi-modal Lineart Colorization
- **分类: cs.CV**

- **简介: 该论文属于线稿上色任务，旨在解决多模态控制下的精准与灵活上色问题。提出OmniColor框架，融合空间与语义约束，提升控制性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.27531](https://arxiv.org/pdf/2603.27531)**

> **作者:** Xulu Zhang; Haoqian Du; Xiaoyong Wei; Qing Li
>
> **摘要:** Lineart colorization is a critical stage in professional content creation, yet achieving precise and flexible results under diverse user constraints remains a significant challenge. To address this, we propose OmniColor, a unified framework for multi-modal lineart colorization that supports arbitrary combinations of control signals. Specifically, we systematically categorize guidance signals into two types: spatially-aligned conditions and semantic-reference conditions. For spatially-aligned inputs, we employ a dual-path encoding strategy paired with a Dense Feature Alignment loss to ensure rigorous boundary preservation and precise color restoration. For semantic-reference inputs, we utilize a VLM-only encoding scheme integrated with a Temporal Redundancy Elimination mechanism to filter repetitive information and enhance inference efficiency. To resolve potential input conflicts, we introduce an Adaptive Spatial-Semantic Gating module that dynamically balances multi-modal constraints. Experimental results demonstrate that OmniColor achieves superior controllability, visual quality, and temporal stability, providing a robust and practical solution for lineart colorization. The source code and dataset will be open at this https URL.
>
---
#### [new 104] Confidence Matters: Uncertainty Quantification and Precision Assessment of Deep Learning-based CMR Biomarker Estimates Using Scan-rescan Data
- **分类: cs.CV**

- **简介: 该论文属于心血管磁共振影像分析任务，旨在评估深度学习模型在心脏功能生物标志物估计中的精度。通过引入不确定性量化方法，发现点估计指标可能高估性能，需采用更全面的分布指标来评估扫描-再扫描一致性。**

- **链接: [https://arxiv.org/pdf/2603.26789](https://arxiv.org/pdf/2603.26789)**

> **作者:** Dewmini Hasara Wickremasinghe; Michelle Gibogwe; Andrew Bell; Esther Puyol-Antón; Muhummad Sohaib Nazir; Reza Razavi; Bruno Paun; Paul Aljabar; Andrew P. King
>
> **摘要:** The performance of deep learning (DL) methods for the analysis of cine cardiovascular magnetic resonance (CMR) is typically assessed in terms of accuracy, overlooking precision. In this work, uncertainty estimation techniques, namely deep ensemble, test-time augmentation, and Monte Carlo dropout, are applied to a state-of-the-art DL pipeline for cardiac functional biomarker estimation, and new distribution-based metrics are proposed for the assessment of biomarker precision. The model achieved high accuracy (average Dice 87%) and point estimate precision on two external validation scan-rescan CMR datasets. However, distribution-based metrics showed that the overlap between scan/rescan confidence intervals was >50% in less than 45% of the cases. Statistical similarity tests between scan and rescan biomarkers also resulted in significant differences for over 65% of the cases. We conclude that, while point estimate metrics might suggest good performance, distributional analyses reveal lower precision, highlighting the need to use more representative metrics to assess scan-rescan agreement.
>
---
#### [new 105] Can We Change the Stroke Size for Easier Diffusion?
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于扩散模型任务，旨在解决低信噪比下的像素级预测难题。通过控制笔触大小，调整目标与扰动的粗糙度，以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.26783](https://arxiv.org/pdf/2603.26783)**

> **作者:** Yunwei Bai; Ying Kiat Tan; Yao Shu; Tsuhan Chen
>
> **摘要:** Diffusion models can be challenged in the low signal-to-noise regime, where they have to make pixel-level predictions despite the presence of high noise. The geometric intuition is akin to using the finest stroke for oil painting throughout, which may be ineffective. We therefore study stroke-size control as a controlled intervention that changes the effective roughness of the supervised target, predictions and perturbations across timesteps, in an attempt to ease the low signal-to-noise challenge. We analyze the advantages and trade-offs of the intervention both theoretically and empirically. Code will be released.
>
---
#### [new 106] Energy-Aware Imitation Learning for Steering Prediction Using Events and Frames
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的转向预测任务，旨在解决传统摄像头在复杂环境下精度不足的问题。通过融合事件相机与帧图像，提出一种能量感知的模仿学习框架。**

- **链接: [https://arxiv.org/pdf/2603.28008](https://arxiv.org/pdf/2603.28008)**

> **作者:** Hu Cao; Jiong Liu; Xingzhuo Yan; Rui Song; Yan Xia; Walter Zimmer; Guang Chen; Alois Knoll
>
> **备注:** Submitted to the journal
>
> **摘要:** In autonomous driving, relying solely on frame-based cameras can lead to inaccuracies caused by factors like long exposure times, high-speed motion, and challenging lighting conditions. To address these issues, we introduce a bio-inspired vision sensor known as the event camera. Unlike conventional cameras, event cameras capture sparse, asynchronous events that provide a complementary modality to mitigate these challenges. In this work, we propose an energy-aware imitation learning framework for steering prediction that leverages both events and frames. Specifically, we design an Energy-driven Cross-modality Fusion Module (ECFM) and an energy-aware decoder to produce reliable and safe predictions. Extensive experiments on two public real-world datasets, DDD20 and DRFuser, demonstrate that our method outperforms existing state-of-the-art (SOTA) approaches. The codes and trained models will be released upon acceptance.
>
---
#### [new 107] Class-Distribution Guided Active Learning for 3D Occupancy Prediction in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文针对自动驾驶中的3D占用预测任务，解决类别不平衡和标注成本高的问题，提出一种基于类别分布的主动学习框架，有效提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.27294](https://arxiv.org/pdf/2603.27294)**

> **作者:** Wonjune Kim; In-Jae Lee; Sihwan Hwang; Sanmin Kim; Dongsuk Kum
>
> **备注:** IEEE RA-L 2026
>
> **摘要:** 3D occupancy prediction provides dense spatial understanding critical for safe autonomous driving. However, this task suffers from a severe class imbalance due to its volumetric representation, where safety-critical objects (bicycles, traffic cones, pedestrians) occupy minimal voxels compared to dominant backgrounds. Additionally, voxel-level annotation is costly, yet dedicating effort to dominant classes is inefficient. To address these challenges, we propose a class-distribution guided active learning framework for selecting training samples to annotate in autonomous driving datasets. Our approach combines three complementary criteria to select the training samples. Inter-sample diversity prioritizes samples whose predicted class distributions differ from those of the labeled set, intra-set diversity prevents redundant sampling within each acquisition cycle, and frequency-weighted uncertainty emphasizes rare classes by reweighting voxel-level entropy with inverse per-sample class proportions. We ensure evaluation validity by using a geographically disjoint train/validation split of Occ3D-nuScenes, which reduces train-validation overlap and mitigates potential map memorization. With only 42.4% labeled data, our framework reaches 26.62 mIoU, comparable to full supervision and outperforming active learning baselines at the same budget. We further validate generality on SemanticKITTI using a different architecture, demonstrating consistent effectiveness across datasets.
>
---
#### [new 108] CDH-Bench: A Commonsense-Driven Hallucination Benchmark for Evaluating Visual Fidelity in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决视觉证据与常识冲突时模型是否产生幻觉的问题。提出CDH-Bench基准，用于评估模型在该场景下的视觉保真度。**

- **链接: [https://arxiv.org/pdf/2603.27982](https://arxiv.org/pdf/2603.27982)**

> **作者:** Kesheng Chen; Yamin Hu; Qi Zhou; Zhenqian Zhu; Wenjian Luo
>
> **摘要:** Vision-language models (VLMs) achieve strong performance on many benchmarks, yet a basic reliability question remains underexplored: when visual evidence conflicts with commonsense, do models follow what is shown or what commonsense suggests? A characteristic failure in this setting is that the model overrides visual evidence and outputs the commonsense alternative. We term this phenomenon \textbf{commonsense-driven hallucination} (CDH). To evaluate it, we introduce \textbf{CDH-Bench}, a benchmark designed to create explicit \textbf{visual evidence--commonsense conflicts}. CDH-Bench covers three dimensions: \textit{counting anomalies}, \textit{relational anomalies}, and \textit{attribute anomalies}. We evaluate frontier VLMs under \textit{binary Question Answering (QA)} and \textit{multiple-choice QA}, and report metrics including \textit{Counterfactual Accuracy} (CF-Acc), \textit{Commonsense Accuracy} (CS-Acc), \textit{Counterfactual Accuracy Drop} (CFAD), \textit{Commonsense Collapse Rate} (CCR), and \textit{Relative Prior Dependency} (RPD). Results show that even strong models remain vulnerable to prior-driven normalization under visual evidence--commonsense conflict. CDH-Bench provides a controlled diagnostic of visual fidelity under visual evidence--commonsense conflict.
>
---
#### [new 109] The Nonverbal Gap: Toward Affective Computer Vision for Safer and More Equitable Online Dating
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉任务，旨在解决在线约会中非语言线索缺失导致的安全与公平问题，提出通过情感计算技术提升平台安全性。**

- **链接: [https://arxiv.org/pdf/2603.26727](https://arxiv.org/pdf/2603.26727)**

> **作者:** Ratna Kandala; Niva Manchanda; Akshata Kishore Moharir
>
> **摘要:** Online dating has become the dominant way romantic relationships begin, yet current platforms strip the nonverbal cues: gaze, facial expression, body posture, response timing, that humans rely on to signal comfort, disinterest, and consent, creating a communication gap with disproportionate safety consequences for women. We argue that this gap represents both a technical opportunity and a moral responsibility for the computer vision community, which has developed the affective tools, facial action unit detection, gaze estimation, engagement modeling, and multimodal affect recognition, needed to begin addressing it, yet has largely ignored the dating domain as a research context. We propose a fairness-first research agenda organized around four capability areas: real-time discomfort detection, engagement asymmetry modeling between partners, consent-aware interaction design, and longitudinal interaction summarization, each grounded in established CV methodology and motivated by the social psychology of romantic communication. We argue that responsible pursuit of this agenda requires purpose-built datasets collected under dyadic consent protocols, fairness evaluation disaggregated across race, gender identity, neurotype, and cultural background, and architectural commitments to on-device processing that prevent affective data from becoming platform surveillance infrastructure. This vision paper calls on the WICV community, whose members are uniquely positioned to understand both the technical opportunity and the human stakes, to establish online dating safety as a first-class research domain before commercial deployment outpaces ethical deliberation.
>
---
#### [new 110] Learning to See through Illumination Extremes with Event Streaming in Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文研究多模态大语言模型在极端光照下的视觉推理问题，提出Event-MLLM模型，通过融合事件流与RGB图像提升感知能力。**

- **链接: [https://arxiv.org/pdf/2603.27558](https://arxiv.org/pdf/2603.27558)**

> **作者:** Baoheng Zhang; Jiahui Liu; Gui Zhao; Weizhou Zhang; Yixuan Ma; Jun Jiang; Yingxian Chen; Wilton W.T. Fok; Xiaojuan Qi; Hayden Kwok-Hay So
>
> **备注:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) perform strong vision-language reasoning under standard conditions but fail in extreme illumination, where RGB inputs lose irrevocable structure and semantics. We propose Event-MLLM, an event-enhanced model that performs all-light visual reasoning by dynamically fusing event streams with RGB frames. Two key components drive our approach: an Illumination Indicator - a learnable signal derived from a DINOv2 branch that represents exposure degradation and adaptively modulates event-RGB fusion - and an Illumination Correction Loss that aligns fused features with non-degraded (normal-light) semantics in the latent space, compensating for information lost in extreme lighting. We curate the first multi-illumination event-instruction corpus for MLLMs, with 2,241 event-RGB samples (around 6 QA pairs each) across diverse scenes and 17 brightness rates (0.05x - 20x), plus an instruct-following benchmark for reasoning, counting, and fine-grained recognition under extreme lighting. Experiments show that Event-MLLM markedly outperforms general-purpose, illumination-adaptive, and event-only baselines, setting a new state of the art in robust multimodal perception and reasoning under challenging illumination.
>
---
#### [new 111] RecycleLoRA: Rank-Revealing QR-Based Dual-LoRA Subspace Adaptation for Domain Generalized Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于领域泛化语义分割任务，解决VFMs中子空间结构利用不足和LoRA表示多样性低的问题。通过RRQR分解提取子空间方向，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.28142](https://arxiv.org/pdf/2603.28142)**

> **作者:** Chanseul Cho; Seokju Yun; Jeaseong Jeon; Seungjae Moon; Youngmin Ro
>
> **备注:** Accepted to CVPR 2026 (Findings)
>
> **摘要:** Domain Generalized Semantic Segmentation (DGSS) aims to maintain robust performance across unseen target domains. Vision Foundation Models (VFMs) offer rich multi-domain knowledge that can enhance generalization. However, strategies for actively exploiting the rich subspace structures within VFMs remain under-explored, with many existing methods focusing primarily on preserving pre-trained knowledge. Furthermore, their LoRA components often suffer from limited representational diversity and inefficient parameter utilization. We propose RecycleLoRA, which addresses both challenges by employing Rank-Revealing QR Decomposition (RRQR) to systematically exploit VFM's subspace structures and enhance LoRA's representational richness. Our main adapter leverages minor subspace directions identified by RRQR to learn diverse and independent features, achieving competitive performance even when used alone. We further introduce a sub adapter that carefully refines major directions with minimal adjustments, providing complementary improvements to the main adapter's strong baseline performance. This design enables the dual adapters to learn distinct representations without requiring additional regularization losses. Our systematic exploitation of pre-trained subspace structures through RRQR-based initialization leads to superior domain generalization performance. RecycleLoRA achieves state-of-the-art performance on both synthetic-to-real generalization and real-to-real generalization tasks without complex architectures or additional inference latency.
>
---
#### [new 112] Language-Conditioned World Modeling for Visual Navigation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究语言条件视觉导航任务，解决在无目标图像情况下，根据语言指令进行导航的问题。构建了LCVN数据集，并提出两种框架实现语言理解、状态预测与动作生成的联合学习。**

- **链接: [https://arxiv.org/pdf/2603.26741](https://arxiv.org/pdf/2603.26741)**

> **作者:** Yifei Dong; Fengyi Wu; Yilong Dai; Lingdong Kong; Guangyu Chen; Xu Zhu; Qiyu Hu; Tianyu Wang; Johnalbert Garnica; Feng Liu; Siyu Huang; Qi Dai; Zhi-Qi Cheng
>
> **备注:** 19 pages, 6 figures, Code: this https URL
>
> **摘要:** We study language-conditioned visual navigation (LCVN), in which an embodied agent is asked to follow a natural language instruction based only on an initial egocentric observation. Without access to goal images, the agent must rely on language to shape its perception and continuous control, making the grounding problem particularly challenging. We formulate this problem as open-loop trajectory prediction conditioned on linguistic instructions and introduce the LCVN Dataset, a benchmark of 39,016 trajectories and 117,048 human-verified instructions that supports reproducible research across a range of environments and instruction styles. Using this dataset, we develop LCVN frameworks that link language grounding, future-state prediction, and action generation through two complementary model families. The first family combines LCVN-WM, a diffusion-based world model, with LCVN-AC, an actor-critic agent trained in the latent space of the world model. The second family, LCVN-Uni, adopts an autoregressive multimodal architecture that predicts both actions and future observations. Experiments show that these families offer different advantages: the former provides more temporally coherent rollouts, whereas the latter generalizes better to unseen environments. Taken together, these observations point to the value of jointly studying language grounding, imagination, and policy learning in a unified task setting, and LCVN provides a concrete basis for further investigation of language-conditioned world models. The code is available at this https URL.
>
---
#### [new 113] AutoCut: End-to-end advertisement video editing based on multimodal discretization and controllable generation
- **分类: cs.CV**

- **简介: 该论文提出AutoCut，解决广告视频制作效率低、成本高的问题。通过多模态离散化和可控生成，实现视频编辑的端到端自动化。**

- **链接: [https://arxiv.org/pdf/2603.28366](https://arxiv.org/pdf/2603.28366)**

> **作者:** Milton Zhou; Sizhong Qin; Yongzhi Li; Quan Chen; Peng Jiang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Short-form videos have become a primary medium for digital advertising, requiring scalable and efficient content creation. However, current workflows and AI tools remain disjoint and modality-specific, leading to high production costs and low overall efficiency. To address this issue, we propose AutoCut, an end-to-end advertisement video editing framework based on multimodal discretization and controllable editing. AutoCut employs dedicated encoders to extract video and audio features, then applies residual vector quantization to discretize them into unified tokens aligned with textual representations, constructing a shared video-audio-text token space. Built upon a foundation model, we further develop a multimodal large language model for video editing through combined multimodal alignment and supervised fine-tuning, supporting tasks covering video selection and ordering, script generation, and background music selection within a unified editing framework. Finally, a complete production pipeline converts the predicted token sequences into deployable long video outputs. Experiments on real-world advertisement datasets show that AutoCut reduces production cost and iteration time while substantially improving consistency and controllability, paving the way for scalable video creation.
>
---
#### [new 114] Can Unsupervised Segmentation Reduce Annotation Costs for Video Semantic Segmentation?
- **分类: cs.CV**

- **简介: 论文研究视频语义分割中的标注成本问题，探讨如何利用未标注帧和粗略标注来减少人工标注需求。属于降低标注成本的任务，通过自动化掩码生成实现高效数据集构建。**

- **链接: [https://arxiv.org/pdf/2603.27697](https://arxiv.org/pdf/2603.27697)**

> **作者:** Samik Some; Vinay P. Namboodiri
>
> **备注:** Published in ICVGIP 2025
>
> **摘要:** Present-day deep neural networks for video semantic segmentation require a large number of fine-grained pixel-level annotations to achieve the best possible results. Obtaining such annotations, however, is very expensive. On the other hand, raw, unannotated video frames are practically free to obtain. Similarly, coarse annotations, which do not require precise boundaries, are also much cheaper. This paper investigates approaches to reduce the annotation cost required for video segmentation datasets by utilising such resources. We show that using state-of-the-art segmentation foundation models, Segment Anything Model (SAM) and Segment Anything Model 2 (SAM 2), we can utilise both unannotated frames as well as coarse annotations to alleviate the effort required for manual annotation of video segmentation datasets by automating mask generation. Our investigation suggests that if used appropriately, we can reduce the need for annotation by a third with similar performance for video semantic segmentation. More significantly, our analysis suggests that the variety of frames in the dataset is more important than the number of frames for obtaining the best performance.
>
---
#### [new 115] Fully Spiking Neural Networks with Target Awareness for Energy-Efficient UAV Tracking
- **分类: cs.CV**

- **简介: 该论文属于无人机视觉跟踪任务，解决传统SNN依赖昂贵事件相机的问题，提出STATrack框架，使用RGB输入实现高效低能耗跟踪。**

- **链接: [https://arxiv.org/pdf/2603.27493](https://arxiv.org/pdf/2603.27493)**

> **作者:** Pengzhi Zhong; Jiwei Mo; Dan Zeng; Feixiang He; Shuiwang Li
>
> **摘要:** Spiking Neural Networks (SNNs), characterized by their event-driven computation and low power consumption, have shown great potential for energy-efficient visual tracking on unmanned aerial vehicles (UAVs). However, existing efficient SNN-based trackers heavily rely on costly event cameras, limiting their deployment on UAVs. To address this limitation, we propose STATrack, an efficient fully spiking neural network framework for UAV visual tracking using RGB inputs only. To the best of our knowledge, this work is the first to investigate spiking neural networks for UAV visual tracking tasks. To mitigate the weakening of target features by background tokens, we propose adaptively maximizing the mutual information between templates and features. Extensive experiments on four widely used UAV tracking benchmarks demonstrate that STATrack achieves competitive tracking performance while maintaining low energy consumption.
>
---
#### [new 116] Unblur-SLAM: Dense Neural SLAM for Blurry Inputs
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出Unblur-SLAM，解决模糊图像下的3D重建问题。通过去模糊和优化技术，提升位姿估计与几何纹理重建效果。**

- **链接: [https://arxiv.org/pdf/2603.26810](https://arxiv.org/pdf/2603.26810)**

> **作者:** Qi Zhang; Denis Rozumny; Francesco Girlanda; Sezer Karaoglu; Marc Pollefeys; Theo Gevers; Martin R. Oswald
>
> **备注:** 14 pages, 9 figures (based on the document's total length and the final Figure 9 ). Accepted By CVPR 2026
>
> **摘要:** We propose Unblur-SLAM, a novel RGB SLAM pipeline for sharp 3D reconstruction from blurred image inputs. In contrast to previous work, our approach is able to handle different types of blur and demonstrates state-of-the-art performance in the presence of both motion blur and defocus blur. Moreover, we adjust the computation effort with the amount of blur in the input image. As a first stage, our method uses a feed-forward image deblurring model for which we propose a suitable training scheme that can improve both tracking and mapping modules. Frames that are successfully deblurred by the feed-forward network obtain refined poses and depth through local-global multi-view optimization and loop closure. Frames that fail the first stage deblurring are directly modeled through the global 3DGS representation and an additional blur network to model multiple blurred sub-frames and simulate the blur formation process in 3D space, thereby learning sharp details and refined sub-frame poses. Experiments on several real-world datasets demonstrate consistent improvements in both pose estimation and sharp reconstruction results of geometry and texture.
>
---
#### [new 117] MD-RWKV-UNet: Scale-Aware Anatomical Encoding with Cross-Stage Fusion for Multi-Organ Segmentation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于多器官分割任务，旨在解决器官尺度变化大、结构复杂的问题。提出MD-RWKV-UNet模型，通过动态编码和跨阶段融合提升分割精度。**

- **链接: [https://arxiv.org/pdf/2603.27261](https://arxiv.org/pdf/2603.27261)**

> **作者:** Zhuoyi Fang
>
> **摘要:** Multi-organ segmentation in medical imaging remains challenging due to large anatomical variability, complex inter-organ dependencies, and diverse organ scales and shapes. Conventional encoder-decoder architectures often struggle to capture both fine-grained local details and long-range context, which are crucial for accurate delineation - especially for small or deformable organs. To address these limitations, we propose MD-RWKV-UNet, a dynamic encoder network that enables scale-aware representation and spatially adaptive context modeling. At its core is the MD-RWKV block, a dual-path module that integrates deformable spatial shifts with the Receptance Weighted Key Value mechanism, allowing the receptive field to adapt dynamically to local structural cues. We further incorporate Selective Kernel Attention to enable adaptive selection of convolutional kernels with varying receptive fields, enhancing multi-scale interaction and improving robustness to organ size and shape variation. In parallel, a cross-stage dual-attention fusion strategy aggregates multi-level features across the encoder, preserving low-level structure while enhancing semantic consistency. Unlike methods that stack static convolutions or rely heavily on global attention, our approach provides a lightweight yet expressive solution for dynamic organ modeling. Experiments on Synapse and ACDC demonstrate state-of-the-art performance, particularly in boundary precision and small-organ segmentation.
>
---
#### [new 118] The Geometry of Robustness: Optimizing Loss Landscape Curvature and Feature Manifold Alignment for Robust Finetuning of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的微调任务，旨在解决ID准确率、OOD泛化和对抗鲁棒性之间的三重权衡。提出GRACE框架，通过优化参数空间曲率和特征空间不变性，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27139](https://arxiv.org/pdf/2603.27139)**

> **作者:** Shivang Chopra; Shaunak Halbe; Chengyue Huan; Brisa Maneechotesuwan; Zsolt Kira
>
> **摘要:** Fine-tuning approaches for Vision-Language Models (VLMs) face a critical three-way trade-off between In-Distribution (ID) accuracy, Out-of-Distribution (OOD) generalization, and adversarial robustness. Existing robust fine-tuning strategies resolve at most two axes of this trade-off. Generalization-preserving methods retain ID/OOD performance but leave models vulnerable to adversarial attacks, while adversarial training improves robustness to targeted attacks but degrades ID/OOD accuracy. Our key insight is that the robustness trade-off stems from two geometric failures: sharp, anisotropic minima in parameter space and unstable feature representations that deform under perturbation. To address this, we propose GRACE (Gram-aligned Robustness via Adaptive Curvature Estimation), a unified fine-tuning framework that jointly regularizes the parameter-space curvature and feature-space invariance for VLMs. Grounded in Robust PAC-Bayes theory, GRACE employs adaptive weight perturbations scaled by local curvature to promote flatter minima, combined with a feature alignment loss that maintains representation consistency across clean, adversarial, and OOD inputs. On ImageNet fine-tuning of CLIP models, GRACE simultaneously improves ID accuracy by 10.8%, and adversarial accuracy by 13.5% while maintaining 57.0% OOD accuracy (vs. 57.4% zero-shot baseline). Geometric analysis confirms that GRACE converges to flatter minima without feature distortion across distribution shifts, providing a principled step toward generalized robustness in foundation VLMs.
>
---
#### [new 119] GradAttn: Replacing Fixed Residual Connections with Task-Modulated Attention Pathways
- **分类: cs.CV**

- **简介: 该论文提出GradAttn，解决深度卷积网络梯度退化问题，通过注意力机制替代固定残差连接，提升特征学习效果。属于图像分类任务。**

- **链接: [https://arxiv.org/pdf/2603.26756](https://arxiv.org/pdf/2603.26756)**

> **作者:** Soudeep Ghoshal; Himanshu Buckchash
>
> **备注:** 14 pages, 5 figures. Under review
>
> **摘要:** Deep ConvNets suffer from gradient signal degradation as network depth increases, limiting effective feature learning in complex architectures. ResNet addressed this through residual connections, but these fixed short-circuits cannot adapt to varying input complexity or selectively emphasize task relevant features across network hierarchies. This study introduces GradAttn, a hybrid CNN-transformer framework that replaces fixed residual connections with attention-controlled gradient flow. By extracting multi-scale CNN features at different depths and regulating them through self-attention, GradAttn dynamically weights shallow texture features and deep semantic representations. For representational analysis, we evaluated three GradAttn variants across eight diverse datasets, from natural images, medical imaging, to fashion recognition. Results demonstrate that GradAttn outperforms ResNet-18 on five of eight datasets, achieving up to +11.07% accuracy improvement on FashionMNIST while maintaining comparable network size. Gradient flow analysis reveals that controlled instabilities, introduced by attention, often coincide with improved generalization, challenging the assumption that perfect stability is optimal. Furthermore, positional encoding effectiveness proves dataset dependent, with CNN hierarchies frequently encoding sufficient spatial structure. These findings allow attention mechanisms as enablers of learnable gradient control, offering a new paradigm for adaptive representation learning in deep neural architectures.
>
---
#### [new 120] BHCast: Unlocking Black Hole Plasma Dynamics from a Single Blurry Image with Long-Term Forecasting
- **分类: cs.CV; astro-ph.IM; cs.LG**

- **简介: 该论文提出BHCast，用于从单张模糊图像中预测黑洞等离子体动态，解决黑洞性质推断问题。通过神经网络和梯度提升树，实现动态建模与参数恢复。**

- **链接: [https://arxiv.org/pdf/2603.26777](https://arxiv.org/pdf/2603.26777)**

> **作者:** Renbo Tu; Ali SaraerToosi; Nicholas S. Conroy; Gennady Pekhimenko; Aviad Levis
>
> **备注:** CVPR 2026
>
> **摘要:** The Event Horizon Telescope (EHT) delivered the first image of a black hole by capturing the light from its surrounding accretion flow, revealing structure but not dynamics. Simulations of black hole accretion dynamics are essential for interpreting EHT images but costly to generate and impractical for inference. Motivated by this bottleneck, BHCast presents a framework for forecasting black hole plasma dynamics from a single, blurry snapshot, such as those captured by the EHT. At its core, BHCast is a neural model that transforms a static image into forecasted future frames, revealing the underlying dynamics hidden within one snapshot. With a multi-scale pyramid loss, we demonstrate how autoregressive forecasting can simultaneously super-resolve and evolve a blurry frame into a coherent, high-resolution movie that remains stable over long time horizons. From forecasted dynamics, we can then extract interpretable spatio-temporal features, such as pattern speed (rotation rate) and pitch angle. Finally, BHCast uses gradient-boosting trees to recover black hole properties from these plasma features, including the spin and viewing inclination angle. The separation between forecasting and inference provides modular flexibility, interpretability, and robust uncertainty quantification. We demonstrate the effectiveness of BHCast on simulations of two distinct black hole accretion systems, Sagittarius A* and M87*, by testing on simulated frames blurred to EHT resolution and real EHT images of M87*. Ultimately, our methodology establishes a scalable paradigm for solving inverse problems, demonstrating the potential of learned dynamics to unlock insights from resolution-limited scientific data.
>
---
#### [new 121] RINO: Rotation-Invariant Non-Rigid Correspondences
- **分类: cs.CV**

- **简介: 该论文属于3D形状对应任务，解决非刚性变形下的密集对应问题。提出RINO框架，通过旋转不变特征提取实现鲁棒匹配。**

- **链接: [https://arxiv.org/pdf/2603.27773](https://arxiv.org/pdf/2603.27773)**

> **作者:** Maolin Gao; Shao Jie Hu-Chen; Congyue Deng; Riccardo Marin; Leonidas Guibas; Daniel Cremers
>
> **备注:** 17 pages, 36 Figures, Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Dense 3D shape correspondence remains a central challenge in computer vision and graphics as many deep learning approaches still rely on intermediate geometric features or handcrafted descriptors, limiting their effectiveness under non-isometric deformations, partial data, and non-manifold inputs. To overcome these issues, we introduce RINO, an unsupervised, rotation-invariant dense correspondence framework that effectively unifies rigid and non-rigid shape matching. The core of our method is the novel RINONet, a feature extractor that integrates vector-based SO(3)-invariant learning with orientation-aware complex functional maps to extract robust features directly from raw geometry. This allows for a fully end-to-end, data-driven approach that bypasses the need for shape pre-alignment or handcrafted features. Extensive experiments show unprecedented performance of RINO across challenging non-rigid matching tasks, including arbitrary poses, non-isometry, partiality, non-manifoldness, and noise.
>
---
#### [new 122] From Content to Audience: A Multimodal Annotation Framework for Broadcast Television Analytics
- **分类: cs.CV; cs.AI; cs.CY**

- **简介: 该论文属于广播电视内容分析任务，旨在解决多模态标注框架的构建与评估问题。通过对比不同模型和输入策略，提升新闻内容的语义标注效果，并实现与观众数据的关联分析。**

- **链接: [https://arxiv.org/pdf/2603.26772](https://arxiv.org/pdf/2603.26772)**

> **作者:** Paolo Cupini; Francesco Pierri
>
> **摘要:** Automated semantic annotation of broadcast television content presents distinctive challenges, combining structured audiovisual composition, domain-specific editorial patterns, and strict operational constraints. While multimodal large language models (MLLMs) have demonstrated strong general-purpose video understanding capabilities, their comparative effectiveness across pipeline architectures and input configurations in broadcast-specific settings remains empirically undercharacterized. This paper presents a systematic evaluation of multimodal annotation pipelines applied to broadcast television news in the Italian setting. We construct a domain-specific benchmark of clips labeled across four semantic dimensions: visual environment classification, topic classification, sensitive content detection, and named entity recognition. Two different pipeline architectures are evaluated across nine frontier models, including Gemini 3.0 Pro, LLaMA 4 Maverick, Qwen-VL variants, and Gemma 3, under progressively enriched input strategies combining visual signals, automatic speech recognition, speaker diarization, and metadata. Experimental results demonstrate that gains from video input are strongly model-dependent: larger models effectively leverage temporal continuity, while smaller models show performance degradation under extended multimodal context, likely due to token overload. Beyond benchmarking, the selected pipeline is deployed on 14 full broadcast episodes, with minute-level annotations integrated with normalized audience measurement data provided by an Italian media company. This integration enables correlational analysis of topic-level audience sensitivity and generational engagement divergence, demonstrating the operational viability of the proposed framework for content-based audience analytics.
>
---
#### [new 123] DreamLite: A Lightweight On-Device Unified Model for Image Generation and Editing
- **分类: cs.CV**

- **简介: 该论文提出DreamLite，一个轻量级统一的设备端扩散模型，解决图像生成与编辑任务。通过优化结构和训练策略，实现高效且高质量的图像处理。**

- **链接: [https://arxiv.org/pdf/2603.28713](https://arxiv.org/pdf/2603.28713)**

> **作者:** Kailai Feng; Yuxiang Wei; Bo Chen; Yang Pan; Hu Ye; Songwei Liu; Chenqian Yan; Yuan Gao
>
> **备注:** this https URL
>
> **摘要:** Diffusion models have made significant progress in both text-to-image (T2I) generation and text-guided image editing. However, these models are typically built with billions of parameters, leading to high latency and increased deployment challenges. While on-device diffusion models improve efficiency, they largely focus on T2I generation and lack support for image editing. In this paper, we propose DreamLite, a compact unified on-device diffusion model (0.39B) that supports both T2I generation and text-guided image editing within a single network. DreamLite is built on a pruned mobile U-Net backbone and unifies conditioning through in-context spatial concatenation in the latent space. It concatenates images horizontally as input, using a (target | blank) configuration for generation tasks and (target | source) for editing tasks. To stabilize the training of this compact model, we introduce a task-progressive joint pretraining strategy that sequentially targets T2I, editing, and joint tasks. After high-quality SFT and reinforcement learning, DreamLite achieves GenEval (0.72) for image generation and ImgEdit (4.11) for image editing, outperforming existing on-device models and remaining competitive with several server-side models. By employing step distillation, we further reduce denoising processing to just 4 steps, enabling our DreamLite could generate or edit a 1024 x 1024 image in less than 1s on a Xiaomi 14 smartphone. To the best of our knowledge, DreamLite is the first unified on-device diffusion model that supports both image generation and image editing.
>
---
#### [new 124] \textit{4DSurf}: High-Fidelity Dynamic Scene Surface Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于动态场景表面重建任务，解决大变形下时间一致性不足的问题。提出4DSurf框架，通过高斯变形和分段策略实现高效、一致的表面重建。**

- **链接: [https://arxiv.org/pdf/2603.28064](https://arxiv.org/pdf/2603.28064)**

> **作者:** Renjie Wu; Hongdong Li; Jose M. Alvarez; Miaomiao Liu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** This paper addresses the problem of dynamic scene surface reconstruction using Gaussian Splatting (GS), aiming to recover temporally consistent geometry. While existing GS-based dynamic surface reconstruction methods can yield superior reconstruction, they are typically limited to either a single object or objects with only small deformations, struggling to maintain temporally consistent surface reconstruction of large deformations over time. We propose ``\textit{4DSurf}'', a novel and unified framework for generic dynamic surface reconstruction that does not require specifying the number or types of objects in the scene, can handle large surface deformations and temporal inconsistency in reconstruction. The key innovation of our framework is the introduction of Gaussian deformations induced Signed Distance Function Flow Regularization that constrains the motion of Gaussians to align with the evolving surface. To handle large deformations, we introduce an Overlapping Segment Partitioning strategy that divides the sequence into overlapping segments with small deformations and incrementally passes geometric information across segments through the shared overlapping timestep. Experiments on two challenging dynamic scene datasets, Hi4D and CMU Panoptic, demonstrate that our method outperforms state-of-the-art surface reconstruction methods by 49\% and 19\% in Chamfer distance, respectively, and achieves superior temporal consistency under sparse-view settings.
>
---
#### [new 125] LogiStory: A Logic-Aware Framework for Multi-Image Story Visualization
- **分类: cs.CV; cs.MA**

- **简介: 该论文提出LogiStory框架，解决多图故事可视化中逻辑连贯性不足的问题。通过建模视觉逻辑，提升叙事清晰度和视觉质量。**

- **链接: [https://arxiv.org/pdf/2603.28082](https://arxiv.org/pdf/2603.28082)**

> **作者:** Chutian Meng; Fan Ma; Chi Zhang; Jiaxu Miao; Yi Yang; Yueting Zhuang
>
> **摘要:** Generating coherent and communicative visual sequences, such as image sequences and videos, remains a significant challenge for current multimodal systems. Despite advances in visual quality and the integration of world knowledge, existing models still struggle to maintain logical flow, often resulting in disjointed actions, fragmented narratives, and unclear storylines. We attribute these issues to the lack of attention to visual logic, a critical yet underexplored dimension of visual sequence generation that we define as the perceptual and causal coherence among characters, actions, and scenes over time. To bridge this gap, we propose a logic-aware multi-image story visualization framework, LogiStory. The framework is built around the central innovation of explicitly modeling visual logic in story visualization. To realize this idea, we design a multi-agent system that grounds roles, extracts causal chains, and verifies story-level consistency, transforming narrative coherence from an implicit byproduct of image generation into an explicit modeling objective. This design effectively bridges structured story planning with visual generation, enhancing both narrative clarity and visual quality in story visualization. Furthermore, to evaluate the generation capacity, we construct LogicTale, a benchmark comprising richly annotated stories, emphasizing causal reasoning, and visual logic interpretability. We establish comprehensive automatic and human evaluation protocols designed to measure both visual logic and perceptual quality. Experiments demonstrate that our approach significantly improves the narrative logic of generated visual stories. This work provides a foundational step towards modeling and enforcing visual logic in general image sequence and video generation tasks.
>
---
#### [new 126] Rényi Entropy: A New Token Pruning Metric for Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于视觉Transformer的优化任务，旨在解决自注意力机制计算复杂度高的问题。通过引入基于Rényi熵的Col-Ln指标，实现更可靠的token剪枝。**

- **链接: [https://arxiv.org/pdf/2603.27900](https://arxiv.org/pdf/2603.27900)**

> **作者:** Wei-Yuan Su; Ruijie Zhang; Zheng Zhang
>
> **摘要:** Vision Transformers (ViTs) achieve state-of-the-art performance but suffer from the $O(N^2)$ complexity of self-attention, making inference costly for high-resolution inputs. To address this bottleneck, token pruning has emerged as a critical technique to accelerate inference. Most existing methods rely on the [CLS] token to estimate patch importance. However, we argue that the [CLS] token can be unreliable in early layers where semantic representations are still immature. As a result, pruning in the early layer often leads to inaccurate importance estimation and unnecessary information loss. In this work, we propose a training-free token importance metric, namely Col-Ln, which is derived from Rényi entropy that enables the identification of informative tokens from the first layer of the network, thereby enabling more reliable pruning in token reduction. Extensive experiments on ViTs and Large Vision-Language Models (LVLMs) demonstrate that our approach consistently outperforms state-of-the-art pruning methods across diverse benchmarks.
>
---
#### [new 127] Inference-Time Structural Reasoning for Compositional Vision-Language Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言理解任务，解决 compositional reasoning 问题。通过结构化推理方法提升模型对关系结构的敏感性，提出文本图解析与图不对称评分机制，实验验证了方法有效性。**

- **链接: [https://arxiv.org/pdf/2603.27349](https://arxiv.org/pdf/2603.27349)**

> **作者:** Amartya Bhattacharya
>
> **摘要:** Vision-language models (VLMs) excel at image-text retrieval yet persistently fail at compositional reasoning, distinguishing captions that share the same words but differ in relational structure. We present, a unified evaluation and augmentation framework benchmarking four architecturally diverse VLMs,CLIP, BLIP, LLaVA, and Qwen3-VL-8B-Thinking,on the Winoground benchmark under plain and scene-graph-augmented regimes. We introduce a dependency-based TextSceneGraphParser (spaCy) extracting subject-relation-object triples, and a Graph Asymmetry Scorer using optimal bipartite matching to inject structural relational priors. Caption ablation experiments (subject-object masking and swapping) reveal that Qwen3-VL-8B-Thinking achieves a group score of 62.75, far above all encoder-based models, while a proposed multi-turn SG filtering strategy further lifts it to 66.0, surpassing prior open-source state-of-the-art. We analyze the capability augmentation tradeoff and find that SG augmentation benefits already capable models while providing negligible or negative gains for weaker baselines. Code: this https URL
>
---
#### [new 128] RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation
- **分类: cs.CV**

- **简介: 该论文属于事故预判任务，解决传统方法依赖主观标注导致风险估计不准的问题。提出RiskProp，通过自监督学习和时间风险传播，提升早期事故预测效果。**

- **链接: [https://arxiv.org/pdf/2603.27165](https://arxiv.org/pdf/2603.27165)**

> **作者:** Yiyang Zou; Tianhao Zhao; Peilun Xiao; Hongyu Jin; Longyu Qi; Yuxuan Li; Liyin Liang; Yifeng Qian; Chunbo Lai; Yutian Lin; Zhihui Li; Yu Wu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Accident anticipation aims to predict impending collisions from dashcam videos and trigger early alerts. Existing methods rely on binary supervision with manually annotated "anomaly onset" frames, which are subjective and inconsistent, leading to inaccurate risk estimation. In contrast, we propose RiskProp, a novel collision-anchored self-supervised risk propagation paradigm for early accident anticipation, which removes the need for anomaly onset annotations and leverages only the reliably annotated collision frame. RiskProp models temporal risk evolution through two observation-driven losses: first, since future frames contain more definitive evidence of an impending accident, we introduce a future-frame regularization loss that uses the model's next-frame prediction as a soft target to supervise the current frame, enabling backward propagation of risk signals; second, inspired by the empirical trend of rising risk before accidents, we design an adaptive monotonic constraint to encourage a non-decreasing progression over time. Experiments on CAP and Nexar demonstrate that RiskProp achieves state-of-the-art performance and produces smoother, more discriminative risk curves, improving both early anticipation and interpretability.
>
---
#### [new 129] ORSIFlow: Saliency-Guided Rectified Flow for Optical Remote Sensing Salient Object Detection
- **分类: cs.CV**

- **简介: 该论文提出ORSIFlow，解决光学遥感图像显著目标检测问题，通过重构流框架实现高效准确的显著性图生成。**

- **链接: [https://arxiv.org/pdf/2603.28584](https://arxiv.org/pdf/2603.28584)**

> **作者:** Haojing Chen; Yutong Li; Zhihang Liu; Tao Tan; Haoyu Bian; Qiuju Ma
>
> **备注:** Accepted by ICME 2026
>
> **摘要:** Optical Remote Sensing Image Salient Object Detection (ORSI-SOD) remains challenging due to complex backgrounds, low contrast, irregular object shapes, and large variations in object scale. Existing discriminative methods directly regress saliency maps, while recent diffusion-based generative approaches suffer from stochastic sampling and high computational cost. In this paper, we propose ORSIFlow, a saliency-guided rectified flow framework that reformulates ORSI-SOD as a deterministic latent flow generation problem. ORSIFlow performs saliency mask generation in a compact latent space constructed by a frozen variational autoencoder, enabling efficient inference with only a few steps. To enhance saliency awareness, we design a Salient Feature Discriminator for global semantic discrimination and a Salient Feature Calibrator for precise boundary refinement. Extensive experiments on multiple public benchmarks show that ORSIFlow achieves state-of-the-art performance with significantly improved efficiency. Codes are available at: this https URL.
>
---
#### [new 130] Generalizable Detection of AI Generated Images with Large Models and Fuzzy Decision Tree
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决现有方法泛化能力差的问题。通过结合轻量级检测器与MLLMs，利用模糊决策树实现有效融合，提升检测精度和泛化性。**

- **链接: [https://arxiv.org/pdf/2603.28508](https://arxiv.org/pdf/2603.28508)**

> **作者:** Fei Wu; Guanghao Ding; Zijian Niu; Zhenrui Wang; Lei Yang; Zhuosheng Zhang; Shilin Wang
>
> **摘要:** The malicious use and widespread dissemination of AI-generated images pose a serious threat to the authenticity of digital content. Existing detection methods exploit low-level artifacts left by common manipulation steps within the generation pipeline, but they often lack generalization due to model-specific overfitting. Recently, researchers have resorted to Multimodal Large Language Models (MLLMs) for AIGC detection, leveraging their high-level semantic reasoning and broad generalization capabilities. While promising, MLLMs lack the fine-grained perceptual sensitivity to subtle generation artifacts, making them inadequate as standalone detectors. To address this issue, we propose a novel AI-generated image detection framework that synergistically integrates lightweight artifact-aware detectors with MLLMs via a fuzzy decision tree. The decision tree treats the outputs of basic detectors as fuzzy membership values, enabling adaptive fusion of complementary cues from semantic and perceptual perspectives. Extensive experiments demonstrate that the proposed method achieves state-of-the-art accuracy and strong generalization across diverse generative models.
>
---
#### [new 131] Object Detection Based on Distributed Convolutional Neural Networks
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决多尺度对象检测问题。提出基于分布式卷积神经网络的检测方法，通过多尺度特征检测实现快速准确的物体定位。**

- **链接: [https://arxiv.org/pdf/2603.28050](https://arxiv.org/pdf/2603.28050)**

> **作者:** Liang Sun
>
> **摘要:** Based on the Distributed Convolutional Neural Network(DisCNN), a straightforward object detection method is proposed. The modules of the output vector of a DisCNN with respect to a specific positive class are positively monotonic with the presence probabilities of the positive features. So, by identifying all high-scoring patches across all possible scales, the positive object can be detected by overlapping them to form a bounding box. The essential idea is that the object is detected by detecting its features on multiple scales, ranging from specific sub-features to abstract features composed of these sub-features. Training DisCNN requires only object-centered image data with positive and negative class labels. The detection process for multiple positive classes can be conducted in parallel to significantly accelerate it, and also faster for single-object detection because of its lightweight model architecture.
>
---
#### [new 132] GEMS: Agent-Native Multimodal Generation with Memory and Skills
- **分类: cs.CV**

- **简介: 该论文提出GEMS框架，解决多模态生成模型在复杂指令和专用任务上的不足。通过引入代理循环、记忆和技能模块，提升模型性能。属于多模态生成任务。**

- **链接: [https://arxiv.org/pdf/2603.28088](https://arxiv.org/pdf/2603.28088)**

> **作者:** Zefeng He; Siyuan Huang; Xiaoye Qu; Yafu Li; Tong Zhu; Yu Cheng; Yang Yang
>
> **备注:** Project Page: this https URL
>
> **摘要:** Recent multimodal generation models have achieved remarkable progress on general-purpose generation tasks, yet continue to struggle with complex instructions and specialized downstream tasks. Inspired by the success of advanced agent frameworks such as Claude Code, we propose \textbf{GEMS} (Agent-Native Multimodal \textbf{GE}neration with \textbf{M}emory and \textbf{S}kills), a framework that pushes beyond the inherent limitations of foundational models on both general and downstream tasks. GEMS is built upon three core components. Agent Loop introduces a structured multi-agent framework that iteratively improves generation quality through closed-loop optimization. Agent Memory provides a persistent, trajectory-level memory that hierarchically stores both factual states and compressed experiential summaries, enabling a global view of the optimization process while reducing redundancy. Agent Skill offers an extensible collection of domain-specific expertise with on-demand loading, allowing the system to effectively handle diverse downstream applications. Across five mainstream tasks and four downstream tasks, evaluated on multiple generative backends, GEMS consistently achieves significant performance gains. Most notably, it enables the lightweight 6B model Z-Image-Turbo to surpass the state-of-the-art Nano Banana 2 on GenEval2, demonstrating the effectiveness of agent harness in extending model capabilities beyond their original limits.
>
---
#### [new 133] Ink Detection from Surface Topography of the Herculaneum Papyri
- **分类: cs.CV; cs.DL**

- **简介: 该论文属于古籍阅读任务，旨在解决碳化纸莎草纸墨迹检测难题。通过分析表面形貌，利用机器学习区分墨迹与纸张，验证高分辨率形貌对检测的有效性。**

- **链接: [https://arxiv.org/pdf/2603.27698](https://arxiv.org/pdf/2603.27698)**

> **作者:** Giorgio Angelotti; Federica Nicolardi; Paul Henderson; W. Brent Seales
>
> **备注:** 9 pages, 3 figures, 2 tables. Currently under review
>
> **摘要:** Reading the Herculaneum papyri is challenging because both the scrolls and the ink, which is carbon-based, are carbonized. In X-ray radiography and tomography, ink detection typically relies on density- or composition-driven contrast, but carbon ink on carbonized papyrus provides little attenuation contrast. Building on the morphological hypothesis, we show that the surface morphology of written regions contains enough signal to distinguish ink from papyrus. To this end, we train machine learning models on three-dimensional optical profilometry from mechanically opened Herculaneum papyri to separate inked and uninked areas. We further quantify how lateral sampling governs learnability and how a native-resolution model behaves on coarsened inputs. We show that high-resolution topography alone contains a usable signal for ink detection. Diminishing segmentation performance with decreasing lateral resolution provides insight into the characteristic spatial scales that must be resolved on our dataset to exploit the morphological signal. These findings inform spatial resolution targets for morphology-based reading of closed scrolls through X-ray tomography.
>
---
#### [new 134] SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于运动预测任务，解决动态环境中异构观测长度下的预测准确性问题。提出SHARP框架，通过增量处理和上下文流提升预测鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2603.28091](https://arxiv.org/pdf/2603.28091)**

> **作者:** Alexander Prutsch; Christian Fruhwirth-Reisinger; David Schinagl; Horst Possegger
>
> **备注:** CVPR 2026. Project page at this https URL
>
> **摘要:** In dynamic traffic environments, motion forecasting models must be able to accurately estimate future trajectories continuously. Streaming-based methods are a promising solution, but despite recent advances, their performance often degrades when exposed to heterogeneous observation lengths. To address this, we propose a novel streaming-based motion forecasting framework that explicitly focuses on evolving scenes. Our method incrementally processes incoming observation windows and leverages an instance-aware context streaming to maintain and update latent agent representations across inference steps. A dual training objective further enables consistent forecasting accuracy across diverse observation horizons. Extensive experiments on Argoverse 2, nuScenes, and Argoverse 1 demonstrate the robustness of our approach under evolving scene conditions and also on the single-agent benchmarks. Our model achieves state-of-the-art performance in streaming inference on the Argoverse 2 multi-agent benchmark, while maintaining minimal latency, highlighting its suitability for real-world deployment.
>
---
#### [new 135] Octree-based Learned Point Cloud Geometry Compression: A Lossy Perspective
- **分类: cs.CV**

- **简介: 该论文属于点云压缩任务，针对损失率压缩中因量化导致的严重失真问题，提出两种改进方法：一种是基于叶节点的有损压缩，另一种是LiDAR点云的速率控制方法。**

- **链接: [https://arxiv.org/pdf/2603.28095](https://arxiv.org/pdf/2603.28095)**

> **作者:** Kaiyu Zheng; Wei Gao; Huiming Zheng
>
> **摘要:** Octree-based context learning has recently become a leading method in point cloud compression. However, its potential on lossy compression remains undiscovered. The traditional lossy compression paradigm using lossless octree representation with quantization step adjustment may result in severe distortions due to massive missing points in quantization. Therefore, we analyze data characteristics of different point clouds and propose lossy approaches specifically. For object point clouds that suffer from quantization step adjustment, we propose a new leaf nodes lossy compression method, which achieves lossy compression by performing bit-wise coding and binary prediction on leaf nodes. For LiDAR point clouds, we explore variable rate approaches and propose a simple but effective rate control method. Experimental results demonstrate that the proposed leaf nodes lossy compression method significantly outperforms the previous octree-based method on object point clouds, and the proposed rate control method achieves about 1% bit error without finetuning on LiDAR point clouds.
>
---
#### [new 136] ToLL: Topological Layout Learning with Structural Multi-view Augmentation for 3D Scene Graph Pretraining
- **分类: cs.CV**

- **简介: 该论文属于3D场景图生成任务，解决数据稀缺导致的泛化能力不足问题。提出ToLL框架，通过拓扑布局学习和结构多视角增强提升表示质量。**

- **链接: [https://arxiv.org/pdf/2603.28178](https://arxiv.org/pdf/2603.28178)**

> **作者:** Yucheng Huang; Luping Ji; Xiangwei Jiang; Wen Li; Mao Ye
>
> **备注:** Under Reivew
>
> **摘要:** 3D Scene Graph (3DSG) generation plays a pivotal role in spatial understanding and semantic-affordance perception. However, its generalizability is often constrained by data scarcity. Current solutions primarily focus on cross-modal assisted representation learning and object-centric generation pre-training. The former relies heavily on predicate annotations, while the latter's predicate learning may be bypassed due to strong object priors. Consequently, they could not often provide a label-free and robust self-supervised proxy task for 3DSG fine-tuning. To bridge this gap, we propose a Topological Layout Learning (ToLL) for 3DSG pretraining framework. In detail, we design an Anchor-Conditioned Topological Geometry Reasoning, with a GNN to recover the global layout of zero-centered subgraphs by the spatial priors from sparse anchors. This process is strictly modulated by predicate features, thereby enforcing the predicate relation learning. Furthermore, we construct a Structural Multi-view Augmentation to avoid semantic corruption, and enhancing representations via self-distillation. The extensive experiments on 3DSSG dataset demonstrate that our ToLL could improve representation quality, outperforming state-of-the-art baselines.
>
---
#### [new 137] LightCtrl: Training-free Controllable Video Relighting
- **分类: cs.CV**

- **简介: 该论文属于视频重光照任务，解决现有方法无法有效控制光照的问题。提出LightCtrl，通过用户提供的光照轨迹实现视频光照的显式控制。**

- **链接: [https://arxiv.org/pdf/2603.27083](https://arxiv.org/pdf/2603.27083)**

> **作者:** Yizuo Peng; Xuelin Chen; Kai Zhang; Xiaodong Cun
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Recent diffusion models have achieved remarkable success in image relighting, and this success has quickly been extended to video relighting. However, existing methods offer limited explicit control over illumination in the relighted output. We present LightCtrl, the first controllable video relighting method that enables explicit control of video illumination through a user-supplied light trajectory in a training-free manner. Our approach combines pre-trained diffusion models: an image relighting model processes each frame individually, followed by a video diffusion prior to enhance temporal consistency. To achieve explicit control over dynamically varying lighting, we introduce two key components. First, a Light Map Injection module samples light trajectory-specific noise and injects it into the latent representation of the source video, improving illumination coherence with the conditional light trajectory. Second, a Geometry-Aware Relighting module dynamically combines RGB and normal map latents in the frequency domain to suppress the influence of the original lighting, further enhancing adherence to the input light trajectory. Experiments show that LightCtrl produces high-quality videos with diverse illumination changes that closely follow the specified light trajectory, demonstrating improved controllability over baseline methods. Code is available at: this https URL.
>
---
#### [new 138] Beyond Dataset Distillation: Lossless Dataset Concentration via Diffusion-Assisted Distribution Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉识别任务，解决大 dataset 的高成本和访问问题。提出 DsCo 框架，通过扩散模型生成小而具代表性的数据集，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2603.27987](https://arxiv.org/pdf/2603.27987)**

> **作者:** Tongfei Liu; Yufan Liu; Bing Li; Weiming Hu
>
> **摘要:** The high cost and accessibility problem associated with large datasets hinder the development of large-scale visual recognition systems. Dataset Distillation addresses these problems by synthesizing compact surrogate datasets for efficient training, storage, transfer, and privacy preservation. The existing state-of-the-art diffusion-based dataset distillation methods face three issues: lack of theoretical justification, poor efficiency in scaling to high data volumes, and failure in data-free scenarios. To address these issues, we establish a theoretical framework that justifies the use of diffusion models by proving the equivalence between dataset distillation and distribution matching, and reveals an inherent efficiency limit in the dataset distillation paradigm. We then propose a Dataset Concentration (DsCo) framework that uses a diffusion-based Noise-Optimization (NOpt) method to synthesize a small yet representative set of samples, and optionally augments the synthetic data via "Doping", which mixes selected samples from the original dataset with the synthetic samples to overcome the efficiency limit of dataset distillation. DsCo is applicable in both data-accessible and data-free scenarios, achieving SOTA performances for low data volumes, and it extends well to high data volumes, where it nearly reduces the dataset size by half with no performance degradation.
>
---
#### [new 139] YOLO Object Detectors for Robotics -- a Comparative Study
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于目标检测任务，研究YOLO模型在机器人工作空间物体检测中的适用性，通过实验验证不同版本的鲁棒性与效果。**

- **链接: [https://arxiv.org/pdf/2603.27029](https://arxiv.org/pdf/2603.27029)**

> **作者:** Patryk Niżeniec; Marcin Iwanowski; Marcin Gahbler
>
> **摘要:** YOLO object detectors recently became a key component of vision systems in many domains. The family of available YOLO models consists of multiple versions, each in various variants. The research reported in this paper aims to validate the applicability of members of this family to detect objects located within the robot workspace. In our experiments, we used our custom dataset and the COCO2017 dataset. To test the robustness of investigated detectors, the images of these datasets were subject to distortions. The results of our experiments, including variations of training/testing configurations and models, may support the choice of the appropriate YOLO version for robotic vision tasks.
>
---
#### [new 140] $R_{dm}$: Re-conceptualizing Distribution Matching as a Reward for Diffusion Distillation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于生成模型任务，旨在解决扩散模型采样慢的问题。通过将分布匹配重构为奖励，提出R_dm框架，提升生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2603.28460](https://arxiv.org/pdf/2603.28460)**

> **作者:** Linqian Fan; Peiqin Sun; Tiancheng Wen; Shun Lu; Chengru Song
>
> **摘要:** Diffusion models achieve state-of-the-art generative performance but are fundamentally bottlenecked by their slow iterative sampling process. While diffusion distillation techniques enable high-fidelity few-step generation, traditional objectives often restrict the student's performance by anchoring it solely to the teacher. Recent approaches have attempted to break this ceiling by integrating Reinforcement Learning (RL), typically through a simple summation of distillation and RL objectives. In this work, we propose a novel paradigm by reconceptualizing distribution matching as a reward, denoted as $R_{dm}$. This unified perspective bridges the algorithmic gap between Diffusion Matching Distillation (DMD) and RL, providing several key benefits. (1) Enhanced optimization stability: we introduce Group Normalized Distribution Matching (GNDM), which adapts standard RL group normalization to stabilize $R_{dm}$ estimation. By leveraging group-mean statistics, GNDM establishes a more robust and effective optimization direction. (2) Seamless reward integration: our reward-centric formulation inherently supports adaptive weighting mechanisms, allowing flexible combination of DMD with external reward models. (3) Improved sampling efficiency: by aligning with RL principles, the framework readily incorporates importance sampling (IS), leading to a significant boost in sampling efficiency. Extensive experiments demonstrate that GNDM outperforms vanilla DMD, reducing the FID by 1.87. Furthermore, our multi-reward variant, GNDMR, surpasses existing baselines by achieving a strong balance between aesthetic quality and fidelity, reaching a peak HPS of 30.37 and a low FID-SD of 12.21. Overall, $R_{dm}$ provides a flexible, stable, and efficient framework for real-time high-fidelity synthesis. Code will be released upon publication.
>
---
#### [new 141] SAGE: Sink-Aware Grounded Decoding for Multimodal Hallucination Mitigation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型中的幻觉检测任务，旨在减少生成内容与视觉输入不一致的问题。提出SAGE框架，通过动态调整注意力机制来抑制幻觉。**

- **链接: [https://arxiv.org/pdf/2603.27898](https://arxiv.org/pdf/2603.27898)**

> **作者:** Tripti Shukla; Zsolt Kira
>
> **备注:** 25 pages, 6 figures, 7 tables
>
> **摘要:** Large vision-language models (VLMs) frequently suffer from hallucinations, generating content that is inconsistent with visual inputs. Existing methods typically address this problem through post-hoc filtering, additional training objectives, or external verification, but they do not intervene during the decoding process when hallucinations arise. In this work, we introduce SAGE, a Sink-Aware Grounded Decoding framework that mitigates hallucinations by dynamically modulating self-attention during generation. Hallucinations are strongly correlated with attention sink tokens - punctuation or function tokens that accumulate disproportionate attention despite carrying limited semantic content. SAGE leverages these tokens as anchors to monitor grounding reliability in real time. At each sink trigger, the method extracts semantic concepts from the generated sequence, estimates their visual grounding using both self-attention maps and gradient-based attribution, and measures their spatial agreement. Based on this signal, self-attention distributions are adaptively sharpened or broadened to reinforce grounded regions or suppress unreliable ones. Extensive experiments across diverse hallucination benchmarks demonstrate that SAGE consistently outperforms existing decoding strategies, achieving substantial reductions in hallucination while preserving descriptive coverage, without requiring model retraining or architectural modifications. Our method achieves an average relative improvement of 10.65% on MSCOCO and 7.19% on AMBER across diverse VLM architectures, demonstrating consistent gains in hallucination mitigation.
>
---
#### [new 142] FlowIt: Global Matching for Optical Flow with Confidence-Guided Refinement
- **分类: cs.CV**

- **简介: 该论文提出FlowIt，用于光学流估计，解决大像素位移和局部匹配不足的问题，通过全局上下文建模和置信度引导的精修实现更准确的运动估计。**

- **链接: [https://arxiv.org/pdf/2603.28759](https://arxiv.org/pdf/2603.28759)**

> **作者:** Sadra Safadoust; Fabio Tosi; Matteo Poggi; Fatma Güney
>
> **摘要:** We present FlowIt, a novel architecture for optical flow estimation designed to robustly handle large pixel displacements. At its core, FlowIt leverages a hierarchical transformer architecture that captures extensive global context, enabling the model to effectively model long-range correspondences. To overcome the limitations of localized matching, we formulate the flow initialization as an optimal transport problem. This formulation yields a highly robust initial flow field, alongside explicitly derived occlusion and confidence maps. These cues are then seamlessly integrated into a guided refinement stage, where the network actively propagates reliable motion estimates from high-confidence regions into ambiguous, low-confidence areas. Extensive experiments across the Sintel, KITTI, Spring, and LayeredFlow datasets validate the efficacy of our approach. FlowIt achieves state-of-the-art results on the competitive Sintel and KITTI benchmarks, while simultaneously establishing new state-of-the-art cross-dataset zero-shot generalization performance on Sintel, Spring, and LayeredFlow.
>
---
#### [new 143] Seen2Scene: Completing Realistic 3D Scenes with Visibility-Guided Flow
- **分类: cs.CV**

- **简介: 该论文提出Seen2Scene，用于3D场景补全任务，解决从不完整真实扫描中生成逼真场景的问题。通过可见性引导的流匹配方法，直接在真实数据上训练，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.28548](https://arxiv.org/pdf/2603.28548)**

> **作者:** Quan Meng; Yujin Chen; Lei Li; Matthias Nießner; Angela Dai
>
> **备注:** Project page: this https URL Video: this https URL
>
> **摘要:** We present Seen2Scene, the first flow matching-based approach that trains directly on incomplete, real-world 3D scans for scene completion and generation. Unlike prior methods that rely on complete and hence synthetic 3D data, our approach introduces visibility-guided flow matching, which explicitly masks out unknown regions in real scans, enabling effective learning from real-world, partial observations. We represent 3D scenes using truncated signed distance field (TSDF) volumes encoded in sparse grids and employ a sparse transformer to efficiently model complex scene structures while masking unknown regions. We employ 3D layout boxes as an input conditioning signal, and our approach is flexibly adapted to various other inputs such as text or partial scans. By learning directly from real-world, incomplete 3D scans, Seen2Scene enables realistic 3D scene completion for complex, cluttered real environments. Experiments demonstrate that our model produces coherent, complete, and realistic 3D scenes, outperforming baselines in completion accuracy and generation quality.
>
---
#### [new 144] EFlow: Fast Few-Step Video Generator Training from Scratch via Efficient Solution Flow
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决视频扩散模型训练效率低的问题。提出EFlow框架，通过减少采样步骤和优化注意力机制，提升训练速度并降低延迟。**

- **链接: [https://arxiv.org/pdf/2603.27086](https://arxiv.org/pdf/2603.27086)**

> **作者:** Dogyun Park; Yanyu Li; Sergey Tulyakov; Anil Kag
>
> **摘要:** Scaling video diffusion transformers is fundamentally bottlenecked by two compounding costs: the expensive quadratic complexity of attention per step, and the iterative sampling steps. In this work, we propose EFlow, an efficient few-step training framework, that tackles these bottlenecks simultaneously. To reduce sampling steps, we build on a solution-flow objective that learns a function mapping a noised state at time t to time s. Making this formulation computationally feasible and high-quality at video scale, however, demands two complementary innovations. First, we propose Gated Local-Global Attention, a token-droppable hybrid block which is efficient, expressive, and remains highly stable under aggressive random token-dropping, substantially reducing per-step compute. Second, we develop an efficient few-step training recipe. We propose Path-Drop Guided training to replace the expensive guidance target with a computationally cheap, weak path. Furthermore, we augment this with a Mean-Velocity Additivity regularizer to ensure high fidelity at extremely low step counts. Together, our EFlow enables a practical from-scratch training pipeline, achieving up to 2.5x higher training throughput over standard solution-flow, and 45.3x lower inference latency than standard iterative models with competitive performance on Kinetics and large-scale text-to-video datasets.
>
---
#### [new 145] Adapting SAM to Nuclei Instance Segmentation and Classification via Cooperative Fine-Grained Refinement
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决SAM在核实例分割中局部特征感知不足和计算成本高的问题。通过提出协同细粒度优化框架提升分割精度。**

- **链接: [https://arxiv.org/pdf/2603.28027](https://arxiv.org/pdf/2603.28027)**

> **作者:** Jingze Su; Tianle Zhu; Jiaxin Cai; Zhiyi Wang; Qi Li; Xiao Zhang; Tong Tong; Shu Wang; Wenxi Liu
>
> **备注:** 18 pages, 10 figures, 12 tables
>
> **摘要:** Nuclei instance segmentation is critical in computational pathology for cancer diagnosis and prognosis. Recently, the Segment Anything Model has demonstrated exceptional performance in various segmentation tasks, leveraging its rich priors and powerful global context modeling capabilities derived from large-scale pre-training on natural images. However, directly applying SAM to the medical imaging domain faces significant limitations: it lacks sufficient perception of the local structural features that are crucial for nuclei segmentation, and full fine-tuning for downstream tasks requires substantial computational costs. To efficiently transfer SAM's robust prior knowledge to nuclei instance segmentation while supplementing its task-aware local perception, we propose a parameter-efficient fine-tuning framework, named Cooperative Fine-Grained Refinement of SAM, consisting of three core components: 1) a Multi-scale Adaptive Local-aware Adapter, which enables effective capability transfer by augmenting the frozen SAM backbone with minimal parameters and instilling a powerful perception of local structures through dynamically generated, multi-scale convolutional kernels; 2) a Hierarchical Modulated Fusion Module, which dynamically aggregates multi-level encoder features to preserve fine-grained spatial details; and 3) a Boundary-Guided Mask Refinement, which integrates multi-context boundary cues with semantic features through explicit supervision, producing a boundary-focused signal to refine initial mask predictions for sharper delineation. These three components work cooperatively to enhance local perception, preserve spatial details, and refine boundaries, enabling SAM to perform accurate nuclei instance segmentation directly.
>
---
#### [new 146] A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps
- **分类: cs.CV**

- **简介: 该论文属于少样本目标检测任务，旨在解决训练样本少导致的优化不稳定和泛化能力差问题。通过设计混合集成解码器和统一渐进微调框架，提升模型性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.28182](https://arxiv.org/pdf/2603.28182)**

> **作者:** Xuanlong Yu; Youyang Sha; Longfei Liu; Xi Shen; Di Yang
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Few-shot object detection (FSOD) is challenging due to unstable optimization and limited generalization arising from the scarcity of training samples. To address these issues, we propose a hybrid ensemble decoder that enhances generalization during fine-tuning. Inspired by ensemble learning, the decoder comprises a shared hierarchical layer followed by multiple parallel decoder branches, where each branch employs denoising queries either inherited from the shared layer or newly initialized to encourage prediction diversity. This design fully exploits pretrained weights without introducing additional parameters, and the resulting diverse predictions can be effectively ensembled to improve generalization. We further leverage a unified progressive fine-tuning framework with a plateau-aware learning rate schedule, which stabilizes optimization and achieves strong few-shot adaptation without complex data augmentations or extensive hyperparameter tuning. Extensive experiments on CD-FSOD, ODinW-13, and RF100-VL validate the effectiveness of our approach. Notably, on RF100-VL, which includes 100 datasets across diverse domains, our method achieves an average performance of 41.9 in the 10-shot setting, significantly outperforming the recent approach SAM3, which obtains 35.7. We further construct a mixed-domain test set from CD-FSOD to evaluate robustness to out-of-distribution (OOD) samples, showing that our proposed modules lead to clear improvement gains. These results highlight the effectiveness, generalization, and robustness of the proposed method. Code is available at: this https URL.
>
---
#### [new 147] MDPBench: A Benchmark for Multilingual Document Parsing in Real-World Scenarios
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MDPBench，一个用于多语言文档解析的基准，解决真实场景下多语言、多格式文档解析问题。工作包括构建数据集和评估模型性能。**

- **链接: [https://arxiv.org/pdf/2603.28130](https://arxiv.org/pdf/2603.28130)**

> **作者:** Zhang Li; Zhibo Lin; Qiang Liu; Ziyang Zhang; Shuo Zhang; Zidun Guo; Jiajun Song; Jiarui Zhang; Xiang Bai; Yuliang Liu
>
> **摘要:** We introduce Multilingual Document Parsing Benchmark, the first benchmark for multilingual digital and photographed document parsing. Document parsing has made remarkable strides, yet almost exclusively on clean, digital, well-formatted pages in a handful of dominant languages. No systematic benchmark exists to evaluate how models perform on digital and photographed documents across diverse scripts and low-resource languages. MDPBench comprises 3,400 document images spanning 17 languages, diverse scripts, and varied photographic conditions, with high-quality annotations produced through a rigorous pipeline of expert model labeling, manual correction, and human verification. To ensure fair comparison and prevent data leakage, we maintain separate public and private evaluation splits. Our comprehensive evaluation of both open-source and closed-source models uncovers a striking finding: while closed-source models (notably Gemini3-Pro) prove relatively robust, open-source alternatives suffer dramatic performance collapse, particularly on non-Latin scripts and real-world photographed documents, with an average drop of 17.8% on photographed documents and 14.0% on non-Latin scripts. These results reveal significant performance imbalances across languages and conditions, and point to concrete directions for building more inclusive, deployment-ready parsing systems. Source available at this https URL.
>
---
#### [new 148] Contextual inference from single objects in Vision-Language models
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究视觉语言模型如何从单个物体推断场景上下文，属于场景理解任务。解决模型是否能通过单个物体进行细粒度和粗粒度场景分类的问题，通过行为和机制分析验证其能力与机制。**

- **链接: [https://arxiv.org/pdf/2603.26731](https://arxiv.org/pdf/2603.26731)**

> **作者:** Martina G. Vilas; Timothy Schaumlöffel; Gemma Roig
>
> **摘要:** How much scene context a single object carries is a well-studied question in human scene perception, yet how this capacity is organized in vision-language models (VLMs) remains poorly understood, with direct implications for the robustness of these models. We investigate this question through a systematic behavioral and mechanistic analysis of contextual inference from single objects. Presenting VLMs with single objects on masked backgrounds, we probe their ability to infer both fine-grained scene category and coarse superordinate context (indoor vs. outdoor). We found that single objects support above-chance inference at both levels, with performance modulated by the same object properties that predict human scene categorization. Object identity, scene, and superordinate predictions are partially dissociable: accurate inference at one level neither requires nor guarantees accurate inference at the others, and the degree of coupling differs markedly across models. Mechanistically, object representations that remain stable when background context is removed are more predictive of successful contextual inference. Scene and superordinate schemas are grounded in fundamentally different ways: scene identity is encoded in image tokens throughout the network, while superordinate information emerges only late or not at all. Together, these results reveal that the organization of contextual inference in VLMs is more complex than accuracy alone suggests, with behavioral and mechanistic signatures
>
---
#### [new 149] GS3LAM: Gaussian Semantic Splatting SLAM
- **分类: cs.CV**

- **简介: 该论文提出GS3LAM，属于SLAM任务，解决语义地图构建中的效率与精度问题，通过高斯语义场和多模态优化实现实时、稠密、一致的语义地图。**

- **链接: [https://arxiv.org/pdf/2603.27781](https://arxiv.org/pdf/2603.27781)**

> **作者:** Linfei Li; Lin Zhang; Zhong Wang; Ying Shen
>
> **备注:** Accepted by ACM MM 2024
>
> **摘要:** Recently, the multi-modal fusion of RGB, depth, and semantics has shown great potential in dense Simultaneous Localization and Mapping (SLAM). However, a prerequisite for generating consistent semantic maps is the availability of dense, efficient, and scalable scene representations. Existing semantic SLAM systems based on explicit representations are often limited by resolution and an inability to predict unknown areas. Conversely, implicit representations typically rely on time-consuming ray tracing, failing to meet real-time requirements. Fortunately, 3D Gaussian Splatting (3DGS) has emerged as a promising representation that combines the efficiency of point-based methods with the continuity of geometric structures. To this end, we propose GS3LAM, a Gaussian Semantic Splatting SLAM framework that processes multimodal data to render consistent, dense semantic maps in real-time. GS3LAM models the scene as a Semantic Gaussian Field (SG-Field) and jointly optimizes camera poses and the field via multimodal error constraints. Furthermore, a Depth-adaptive Scale Regularization (DSR) scheme is introduced to resolve misalignments between scale-invariant Gaussians and geometric surfaces. To mitigate catastrophic forgetting, we propose a Random Sampling-based Keyframe Mapping (RSKM) strategy, which demonstrates superior performance over common local covisibility optimization methods. Extensive experiments on benchmark datasets show that GS3LAM achieves increased tracking robustness, superior rendering quality, and enhanced semantic precision compared to state-of-the-art methods. Source code is available at this https URL.
>
---
#### [new 150] Poppy: Polarization-based Plug-and-Play Guidance for Enhancing Monocular Normal Estimation
- **分类: cs.CV**

- **简介: 该论文属于单目法线估计任务，解决反射、无纹理和暗表面的估计难题。通过引入无需训练的Poppy框架，利用偏振信息优化法线估计。**

- **链接: [https://arxiv.org/pdf/2603.27891](https://arxiv.org/pdf/2603.27891)**

> **作者:** Irene Kim; Sai Tanmay Reddy Chakkera; Alexandros Graikos; Dimitris Samaras; Akshat Dave
>
> **备注:** project page: this https URL
>
> **摘要:** Monocular surface normal estimators trained on large-scale RGB-normal data often perform poorly in the edge cases of reflective, textureless, and dark surfaces. Polarization encodes surface orientation independently of texture and albedo, offering a physics-based complement for these cases. Existing polarization methods, however, require multi-view capture or specialized training data, limiting generalization. We introduce Poppy, a training-free framework that refines normals from any frozen RGB backbone using single-shot polarization measurements at test time. Keeping backbone weights frozen, Poppy optimizes per-pixel offsets to the input RGB and output normal along with a learned reflectance decomposition. A differentiable rendering layer converts the refined normals into polarization predictions and penalizes mismatches with the observed signal. Across seven benchmarks and three backbone architectures (diffusion, flow, and feed-forward), Poppy reduces mean angular error by 23-26% on synthetic data and 6-16% on real data. These results show that guiding learned RGB-based normal estimators with polarization cues at test time refines normals on challenging surfaces without retraining.
>
---
#### [new 151] Drift-AR: Single-Step Visual Autoregressive Generation via Anti-Symmetric Drifting
- **分类: cs.CV**

- **简介: 该论文提出Drift-AR，解决AR-Diffusion模型的双速瓶颈问题。通过熵信号统一加速AR和视觉解码阶段，实现单步高质生成。**

- **链接: [https://arxiv.org/pdf/2603.28049](https://arxiv.org/pdf/2603.28049)**

> **作者:** Zhen Zou; Xiaoxiao Ma; Mingde Yao; Jie Huang; LinJiang Huang; Feng Zhao
>
> **摘要:** Autoregressive (AR)-Diffusion hybrid paradigms combine AR's structured semantic modeling with diffusion's high-fidelity synthesis, yet suffer from a dual speed bottleneck: the sequential AR stage and the iterative multi-step denoising of the diffusion vision decode stage. Existing methods address each in isolation without a unified principle design. We observe that the per-position \emph{prediction entropy} of continuous-space AR models naturally encodes spatially varying generation uncertainty, which simultaneously governing draft prediction quality in the AR stage and reflecting the corrective effort required by vision decoding stage, which is not fully explored before. Since entropy is inherently tied to both bottlenecks, it serves as a natural unifying signal for joint acceleration. In this work, we propose \textbf{Drift-AR}, which leverages entropy signal to accelerate both stages: 1) for AR acceleration, we introduce Entropy-Informed Speculative Decoding that align draft--target entropy distributions via a causal-normalized entropy loss, resolving the entropy mismatch that causes excessive draft rejection; 2) for visual decoder acceleration, we reinterpret entropy as the \emph{physical variance} of the initial state for an anti-symmetric drifting field -- high-entropy positions activate stronger drift toward the data manifold while low-entropy positions yield vanishing drift -- enabling single-step (1-NFE) decoding without iterative denoising or distillation. Moreover, both stages share the same entropy signal, which is computed once with no extra cost. Experiments on MAR, TransDiff, and NextStep-1 demonstrate 3.8--5.5$\times$ speedup with genuine 1-NFE decoding, matching or surpassing original quality. Code will be available at this https URL.
>
---
#### [new 152] SegRGB-X: General RGB-X Semantic Segmentation Model
- **分类: cs.CV**

- **简介: 该论文属于跨模态语义分割任务，解决多传感器数据分割效率低的问题。提出SegRGB-X框架，通过三个创新模块实现通用多模态分割，取得最佳性能。**

- **链接: [https://arxiv.org/pdf/2603.28023](https://arxiv.org/pdf/2603.28023)**

> **作者:** Jiong Liu; Yingjie Xu; Xingcheng Zhou; Rui Song; Walter Zimmer; Alois Knoll; Hu Cao
>
> **备注:** Submitted to IEEE TITS
>
> **摘要:** Semantic segmentation across arbitrary sensor modalities faces significant challenges due to diverse sensor characteristics, and the traditional configurations for this task result in redundant development efforts. We address these challenges by introducing a universal arbitrary-modal semantic segmentation framework that unifies segmentation across multiple modalities. Our approach features three key innovations: (1) the Modality-aware CLIP (MA-CLIP), which provides modality-specific scene understanding guidance through LoRA fine-tuning; (2) Modality-aligned Embeddings for capturing fine-grained features; and (3) the Domain-specific Refinement Module (DSRM) for dynamic feature adjustment. Evaluated on five diverse datasets with different complementary modalities (event, thermal, depth, polarization, and light field), our model surpasses specialized multi-modal methods and achieves state-of-the-art performance with a mIoU of 65.03%. The codes will be released upon acceptance.
>
---
#### [new 153] Transferring Physical Priors into Remote Sensing Segmentation via Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于遥感图像语义分割任务，旨在解决传统方法依赖对齐数据和昂贵重训练的问题。通过构建物理知识图谱和新数据集，提出PriorSeg模型提升分割精度与物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.27504](https://arxiv.org/pdf/2603.27504)**

> **作者:** Yuxi Lu; Kunqi Li; Zhidong Li; Xiaohan Su; Biao Wu; Chenya Huang; Bin Liang
>
> **摘要:** Semantic segmentation of remote sensing imagery is fundamental to Earth observation. Achieving accurate results requires integrating not only optical images but also physical variables such as the Digital Elevation Model (DEM), Synthetic Aperture Radar (SAR) and Normalized Difference Vegetation Index (NDVI). Recent foundation models (FMs) leverage pre-training to exploit these variables but still depend on spatially aligned data and costly retraining when involving new sensors. To overcome these limitations, we introduce a novel paradigm for integrating domain-specific physical priors into segmentation models. We first construct a Physical-Centric Knowledge Graph (PCKG) by prompting large language models to extract physical priors from 1,763 vocabularies, and use it to build a heterogeneous, spatial-aligned dataset, Phy-Sky-SA. Building on this foundation, we develop PriorSeg, a physics-aware residual refinement model trained with a joint visual-physical strategy that incorporates a novel physics-consistency loss. Experiments on heterogeneous settings demonstrate that PriorSeg improves segmentation accuracy and physical plausibility without retraining the FMs. Ablation studies verify the effectiveness of the Phy-Sky-SA dataset, the PCKG, and the physics-consistency loss.
>
---
#### [new 154] Navigating the Mirage: A Dual-Path Agentic Framework for Robust Misleading Chart Question Answering
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于图表问答任务，旨在解决误导性图表的鲁棒回答问题。提出ChartCynics框架，通过双路径机制识别视觉欺骗，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2603.28583](https://arxiv.org/pdf/2603.28583)**

> **作者:** Yanjie Zhang; Yafei Li; Rui Sheng; Zixin Chen; Yanna Lin; Huamin Qu; Lei Chen; Yushi Sun
>
> **备注:** 10pages, 4 figures
>
> **摘要:** Despite the success of Vision-Language Models (VLMs), misleading charts remain a significant challenge due to their deceptive visual structures and distorted data representations. We present ChartCynics, an agentic dual-path framework designed to unmask visual deception via a "skeptical" reasoning paradigm. Unlike holistic models, ChartCynics decouples perception from verification: a Diagnostic Vision Path captures structural anomalies (e.g., inverted axes) through strategic ROI cropping, while an OCR-Driven Data Path ensures numerical grounding. To resolve cross-modal conflicts, we introduce an Agentic Summarizer optimized via a two-stage protocol: Oracle-Informed SFT for reasoning distillation and Deception-Aware GRPO for adversarial alignment. This pipeline effectively penalizes visual traps and enforces logical consistency. Evaluations on two benchmarks show that ChartCynics achieves 74.43% and 64.55% accuracy, providing an absolute performance boost of ~29% over the Qwen3-VL-8B backbone, outperforming state-of-the-art proprietary models. Our results demonstrate that specialized agentic workflows can grant smaller open-source models superior robustness, establishing a new foundation for trustworthy chart interpretation.
>
---
#### [new 155] AI-Powered Facial Mask Removal Is Not Suitable For Biometric Identification
- **分类: cs.CV; cs.AI**

- **简介: 论文探讨AI面部去遮挡技术在生物识别中的适用性，旨在解决误识别风险问题。研究评估了商业AI去遮挡效果及其可靠性。**

- **链接: [https://arxiv.org/pdf/2603.27747](https://arxiv.org/pdf/2603.27747)**

> **作者:** Emily A Cooper; Hany Farid
>
> **摘要:** Recently, crowd-sourced online criminal investigations have used generative-AI to enhance low-quality visual evidence. In one high-profile case, social-media users circulated an "AI-unmasked" image of a federal agent involved in a fatal shooting, fueling a wide-spread misidentification. In response to this and similar incidents, we conducted a large-scale analysis evaluating the efficacy and risks of commercial AI-powered facial unmasking, specifically assessing whether the resulting faces can be reliably matched to true identities.
>
---
#### [new 156] VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，解决自然语言描述下视频对象分割的问题。针对传统方法在动态和多步骤推理场景中的不足，提出VIRST框架，统一视频推理与像素分割，提升性能。**

- **链接: [https://arxiv.org/pdf/2603.27060](https://arxiv.org/pdf/2603.27060)**

> **作者:** Jihwan Hong; Jaeyoung Do
>
> **备注:** CVPR 2026
>
> **摘要:** Referring Video Object Segmentation (RVOS) aims to segment target objects in videos based on natural language descriptions. However, fixed keyframe-based approaches that couple a vision language model with a separate propagation module often fail to capture rapidly changing spatiotemporal dynamics and to handle queries requiring multi-step reasoning, leading to sharp performance drops on motion-intensive and reasoning-oriented videos beyond static RVOS benchmarks. To address these limitations, we propose VIRST (Video-Instructed Reasoning Assistant for Spatio-Temporal Segmentation), an end-to-end framework that unifies global video reasoning and pixel-level mask prediction within a single model. VIRST bridges semantic and segmentation representations through the Spatio-Temporal Fusion (STF), which fuses segmentation-aware video features into the vision-language backbone, and employs the Temporal Dynamic Anchor Updater to maintain temporally adjacent anchor frames that provide stable temporal cues under large motion, occlusion, and reappearance. This unified design achieves state-of-the-art results across diverse RVOS benchmarks under realistic and challenging conditions, demonstrating strong generalization to both referring and reasoning oriented settings. The code and checkpoints are available at this https URL.
>
---
#### [new 157] CiQi-Agent: Aligning Vision, Tools and Aesthetics in Multimodal Agent for Cultural Reasoning on Chinese Porcelains
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CiQi-Agent，用于中国瓷器文化推理任务，解决非专家难以理解瓷器鉴定的问题。通过多模态分析和知识检索，实现精准属性识别与解释性描述。**

- **链接: [https://arxiv.org/pdf/2603.28474](https://arxiv.org/pdf/2603.28474)**

> **作者:** Wenhan Wang; Zhixiang Zhou; Zhongtian Ma; Yanzhu Chen; Ziyu Lin; Hao Sheng; Pengfei Liu; Honglin Ma; Wenqi Shao; Qiaosheng Zhang; Yu Qiao
>
> **摘要:** The connoisseurship of antique Chinese porcelain demands extensive historical expertise, material understanding, and aesthetic sensitivity, making it difficult for non-specialists to engage. To democratize cultural-heritage understanding and assist expert connoisseurship, we introduce CiQi-Agent -- a domain-specific Porcelain Connoisseurship Agent for intelligent analysis of antique Chinese porcelain. CiQi-Agent supports multi-image porcelain inputs and enables vision tool invocation and multimodal retrieval-augmented generation, performing fine-grained connoisseurship analysis across six attributes: dynasty, reign period, kiln site, glaze color, decorative motif, and vessel shape. Beyond attribute classification, it captures subtle visual details, retrieves relevant domain knowledge, and integrates visual and textual evidence to produce coherent, explainable connoisseurship descriptions. To achieve this capability, we construct a large-scale, expert-annotated dataset CiQi-VQA, comprising 29,596 porcelain specimens, 51,553 images, and 557,940 visual question--answering pairs, and further establish a comprehensive benchmark CiQi-Bench aligned with the previously mentioned six attributes. CiQi-Agent is trained through supervised fine-tuning, reinforcement learning, and a tool-augmented reasoning framework that integrates two categories of tools: a vision tool and multimodal retrieval tools. Experimental results show that CiQi-Agent (7B) outperforms all competitive open- and closed-source models across all six attributes on CiQi-Bench, achieving on average 12.2\% higher accuracy than GPT-5. The model and dataset have been released and are publicly available at this https URL.
>
---
#### [new 158] To View Transform or Not to View Transform: NeRF-based Pre-training Perspective
- **分类: cs.CV**

- **简介: 该论文属于3D感知任务，旨在解决NeRF与视图变换结合导致的表示模糊问题。提出NeRP3D模型，保留预训练NeRF网络，提升3D场景理解效果。**

- **链接: [https://arxiv.org/pdf/2603.28090](https://arxiv.org/pdf/2603.28090)**

> **作者:** Hyeonjun Jeong; Juyeb Shin; Dongsuk Kum
>
> **备注:** The Fourteenth International Conference on Learning Representations (ICLR'26)
>
> **摘要:** Neural radiance fields (NeRFs) have emerged as a prominent pre-training paradigm for vision-centric autonomous driving, which enhances 3D geometry and appearance understanding in a fully self-supervised manner. To apply NeRF-based pretraining to 3D perception models, recent approaches have simply applied NeRFs to volumetric features obtained from view transformation. However, coupling NeRFs with view transformation inherits conflicting priors; view transformation imposes discrete and rigid representations, whereas radiance fields assume continuous and adaptive functions. When these opposing assumptions are forced into a single pipeline, the misalignment surfaces as blurry and ambiguous 3D representations that ultimately limit 3D scene understanding. Moreover, the NeRF network for pre-training is discarded during downstream tasks, resulting in inefficient utilization of enhanced 3D representations through NeRF. In this paper, we propose a novel NeRF-Resembled Point-based 3D detector that can learn continuous 3D representation and thus avoid the misaligned priors from view transformation. NeRP3D preserves the pre-trained NeRF network regardless of the tasks, inheriting the principle of continuous 3D representation learning and leading to greater potentials for both scene reconstruction and detection tasks. Experiments on nuScenes dataset demonstrate that our proposed approach significantly improves previous state-of-the-art methods, outperforming not only pretext scene reconstruction tasks but also downstream detection tasks.
>
---
#### [new 159] OpenDPR: Open-Vocabulary Change Detection via Vision-Centric Diffusion-Guided Prototype Retrieval for Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文属于开放词汇变化检测任务，旨在解决传统方法在类别泛化和变化定位上的不足。通过结合视觉基础模型与扩散生成原型，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2603.27645](https://arxiv.org/pdf/2603.27645)**

> **作者:** Qi Guo; Jue Wang; Yinhe Liu; Yanfei Zhong
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Open-vocabulary change detection (OVCD) seeks to recognize arbitrary changes of interest by enabling generalization beyond a fixed set of predefined classes. We reformulate OVCD as a two-stage pipeline: first generate class-agnostic change proposals using visual foundation models (VFMs) such as SAM and DINOv2, and then perform category identification with vision-language models (VLMs) such as CLIP. We reveal that category identification errors are the primary bottleneck of OVCD, mainly due to the limited ability of VLMs based on image-text matching to represent fine-grained land-cover categories. To address this, we propose OpenDPR, a training-free vision-centric diffusion-guided prototype retrieval framework. OpenDPR leverages diffusion models to construct diverse prototypes for target categories offline, and to perform similarity retrieval with change proposals in the visual space during inference. The secondary bottleneck lies in change localization, due to the inherent lack of change priors in VFMs. To bridge this gap, we design a spatial-to-change weakly supervised change detection module named S2C to adapt their strong spatial modeling capabilities for change localization. Integrating the pretrained S2C into OpenDPR leads to an optional weakly supervised variant named OpenDPR-W, which further improves OVCD with minimal supervision. Experimental results on four benchmark datasets demonstrate that the proposed methods achieve state-of-the-art performance under both supervision modes. Code is available at this https URL.
>
---
#### [new 160] Optimized Weighted Voting System for Brain Tumor Classification Using MRI Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于脑肿瘤分类任务，旨在提高MRI图像分类的准确性。通过集成多种深度学习与传统机器学习模型，结合加权投票机制，提升诊断效果。**

- **链接: [https://arxiv.org/pdf/2603.28357](https://arxiv.org/pdf/2603.28357)**

> **作者:** Ha Anh Vu
>
> **摘要:** The accurate classification of brain tumors from MRI scans is essential for effective diagnosis and treatment planning. This paper presents a weighted ensemble learning approach that combines deep learning and traditional machine learning models to improve classification performance. The proposed system integrates multiple classifiers, including ResNet101, DenseNet121, Xception, CNN-MRI, and ResNet50 with edge-enhanced images, SVM, and KNN with HOG features. A weighted voting mechanism assigns higher influence to models with better individual accuracy, ensuring robust decision-making. Image processing techniques such as Balance Contrast Enhancement, K-means clustering, and Canny edge detection are applied to enhance feature extraction. Experimental evaluations on the Figshare and Kaggle MRI datasets demonstrate that the proposed method achieves state-of-the-art accuracy, outperforming existing models. These findings highlight the potential of ensemble-based learning for improving brain tumor classification, offering a reliable and scalable framework for medical image analysis.
>
---
#### [new 161] Zero-shot Vision-Language Reranking for Cross-View Geolocalization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨视角地理定位任务，旨在提升Top-1准确率。通过引入零样本视觉-语言模型进行成对重排序，有效改善了检索精度。**

- **链接: [https://arxiv.org/pdf/2603.27251](https://arxiv.org/pdf/2603.27251)**

> **作者:** Yunus Talha Erzurumlu; John E. Anderson; William J. Shuart; Charles Toth; Alper Yilmaz
>
> **备注:** 7 pages, 4 figures. Accepted to XXV ISPRS Congress
>
> **摘要:** Cross-view geolocalization (CVGL) systems, while effective at retrieving a list of relevant candidates (high Recall@k), often fail to identify the single best match (low Top-1 accuracy). This work investigates the use of zero-shot Vision-Language Models (VLMs) as rerankers to address this gap. We propose a two-stage framework: state-of-the-art (SOTA) retrieval followed by VLM reranking. We systematically compare two strategies: (1) Pointwise (scoring candidates individually) and (2) Pairwise (comparing candidates relatively). Experiments on the VIGOR dataset show a clear divergence: all pointwise methods cause a catastrophic drop in performance or no change at all. In contrast, a pairwise comparison strategy using LLaVA improves Top-1 accuracy over the strong retrieval baseline. Our analysis concludes that, these VLMs are poorly calibrated for absolute relevance scoring but are effective at fine-grained relative visual judgment, making pairwise reranking a promising direction for enhancing CVGL precision.
>
---
#### [new 162] GEditBench v2: A Human-Aligned Benchmark for General Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决现有评估框架不足的问题。提出GEditBench v2基准和PVC-Judge模型，提升视觉一致性评估的准确性。**

- **链接: [https://arxiv.org/pdf/2603.28547](https://arxiv.org/pdf/2603.28547)**

> **作者:** Zhangqi Jiang; Zheng Sun; Xianfang Zeng; Yufeng Yang; Xuanyang Zhang; Yongliang Wu; Wei Cheng; Gang Yu; Xu Yang; Bihan Wen
>
> **备注:** 30 pages, 24 figures
>
> **摘要:** Recent advances in image editing have enabled models to handle complex instructions with impressive realism. However, existing evaluation frameworks lag behind: current benchmarks suffer from narrow task coverage, while standard metrics fail to adequately capture visual consistency, i.e., the preservation of identity, structure and semantic coherence between edited and original images. To address these limitations, we introduce GEditBench v2, a comprehensive benchmark with 1,200 real-world user queries spanning 23 tasks, including a dedicated open-set category for unconstrained, out-of-distribution editing instructions beyond predefined tasks. Furthermore, we propose PVC-Judge, an open-source pairwise assessment model for visual consistency, trained via two novel region-decoupled preference data synthesis pipelines. Besides, we construct VCReward-Bench using expert-annotated preference pairs to assess the alignment of PVC-Judge with human judgments on visual consistency evaluation. Experiments show that our PVC-Judge achieves state-of-the-art evaluation performance among open-source models and even surpasses GPT-5.1 on average. Finally, by benchmarking 16 frontier editing models, we show that GEditBench v2 enables more human-aligned evaluation, revealing critical limitations of current models, and providing a reliable foundation for advancing precise image editing.
>
---
#### [new 163] Distilled Large Language Model-Driven Dynamic Sparse Expert Activation Mechanism
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉识别任务，解决高类间相似性和计算资源限制问题。提出DS-MoE框架，通过动态稀疏专家激活实现高效准确的缺陷检测。**

- **链接: [https://arxiv.org/pdf/2603.26735](https://arxiv.org/pdf/2603.26735)**

> **作者:** Qinghui Chen; Zekai Zhang; Zaigui Zhang; Kai Zhang; Dagang Li; Wenmin Wang; Jinglin Zhang; Cong Liu
>
> **摘要:** High inter-class similarity, extreme scale variation, and limited computational budgets hinder reliable visual recognition across diverse real-world data. Existing vision-centric and cross-modal approaches often rely on rigid fusion mechanisms and heavy annotation pipelines, leading to sub-optimal generalization. We propose the Distilled Large Language Model (LLM)-Driven Sparse Mixture-of-Experts (DS-MoE) framework, which integrates text-guided dynamic routing and lightweight multi-scale comprehension. The DS-MoE framework dynamically aligns textual semantics with defect-specific visual patterns through a sparse MoE architecture, where task-relevant experts are adaptively activated based on semantic relevance, resolving inter-class ambiguity. A lightweight MobileSAM encoder enables real-time inference while preserving multi-scale defect details. Extensive experiments on PCB, aluminum foil, and mold defect datasets demonstrate that our framework achieves superior performance compared to existing pure vision models. \textbf{DS-MoE} surpasses YOLOv8/YOLOX with gains of +13.9, +1.4, and +2.0 pp mAP@ 0.5:0.95 on BBMP, aluminum, and PCB, respectively, while also improving precision and recall.
>
---
#### [new 164] A Benchmarking Methodology to Assess Open-Source Video Large Language Models in Automatic Captioning of News Videos
- **分类: cs.CV**

- **简介: 该论文属于新闻视频自动字幕生成任务，旨在评估开源视频大语言模型的性能。通过构建基准数据集并引入新指标，分析模型在主题和实体保持方面的表现，以提升评价的准确性。**

- **链接: [https://arxiv.org/pdf/2603.27662](https://arxiv.org/pdf/2603.27662)**

> **作者:** David Miranda Paredes; Jose M. Saavedra; Marcelo Pizarro
>
> **摘要:** News videos are among the most prevalent content types produced by television stations and online streaming platforms, yet generating textual descriptions to facilitate indexing and retrieval largely remains a manual process. Video Large Language Models (VidLLMs) offer significant potential to automate this task, but a comprehensive evaluation in the news domain is still lacking. This work presents a comparative study of eight state-of-the-art open-source VidLLMs for automatic news video captioning, evaluated on two complementary benchmark datasets: a Chilean TV news corpus (approximately 1,345 clips) and a BBC News corpus (9,838 clips). We employ lexical metrics (METEOR, ROUGE-L), semantic metrics (BERTScore, CLIPScore, Text Similarity, Mean Reciprocal Rank), and two novel fidelity metrics proposed in this work: the Thematic Fidelity Score (TFS) and Entity Fidelity Score (EFS). Our analysis reveals that standard metrics exhibit limited discriminative power for news video captioning due to surface-form dependence, static-frame insensitivity, and function-word inflation. TFS and EFS address these gaps by directly assessing thematic structure preservation and named-entity coverage in the generated captions. Results show that Gemma~3 achieves the highest overall performance across both datasets and most evaluation dimensions, with Qwen-VL as a consistent runner-up.
>
---
#### [new 165] Let Triggers Control: Frequency-Aware Dropout for Effective Token Control
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决触发词控制不足的问题。通过提出频率感知丢弃方法（FAD），提升提示的准确性和生成质量，无需额外参数。**

- **链接: [https://arxiv.org/pdf/2603.27199](https://arxiv.org/pdf/2603.27199)**

> **作者:** Junyoung Koh; Hoyeon Moon; Dongha Kim; Seungmin Lee; Sanghyun Park; Min Song
>
> **备注:** CVPR 2026 P13N: Personalization in Generative AI workshop
>
> **摘要:** Text-to-image models such as Stable Diffusion have achieved unprecedented levels of high-fidelity visual synthesis. As these models advance, personalization of generative models -- commonly facilitated through Low-Rank Adaptation (LoRA) with a dedicated trigger token -- has become a significant area of research. Previous works have naively assumed that fine-tuning with a single trigger token to represent new concepts. However, this often results in poor controllability, where the trigger token alone fails to reliably evoke the intended concept. We attribute this issue to the frequent co-occurrence of the trigger token with the surrounding context during fine-tuning, which entangles their representations and compromises the token's semantic distinctiveness. To disentangle this, we propose Frequency-Aware Dropout (FAD) -- a novel regularization technique that improves prompt controllability without adding new parameters. FAD consists of two key components: co-occurrence analysis and curriculum-inspired scheduling. Qualitative and quantitative analyses across token-based diffusion models (SD~1.5 and SDXL) and natural language--driven backbones (FLUX and Qwen-Image) demonstrate consistent gains in prompt fidelity, stylistic precision, and user-perceived quality. Our method provides a simple yet effective dropout strategy that enhances controllability and personalization in text-to-image generation. Notably, it achieves these improvements without introducing additional parameters or architectural modifications, making it readily applicable to existing models with minimal computational overhead.
>
---
#### [new 166] BlankSkip: Early-exit Object Detection onboard Nano-drones
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决纳米无人机上计算资源受限的问题。通过引入早期退出机制，提升推理效率，同时保持检测精度。**

- **链接: [https://arxiv.org/pdf/2603.28149](https://arxiv.org/pdf/2603.28149)**

> **作者:** Carlo Marra; Beatrice Alessandra Motetti; Alessio Burrello; Enrico Macii; Massimo Poncino; Daniele Jahier Pagliari
>
> **备注:** Accepted for publication in the Embedded Vision Workshop of the 2026 Computer Vision and Pattern Recognition (CVPR) conference
>
> **摘要:** Deploying tiny computer vision Deep Neural Networks (DNNs) on-board nano-sized drones is key for achieving autonomy, but is complicated by the extremely tight constraints of their computational platforms (approximately 10 MiB memory, 1 W power budget). Early-exit adaptive DNNs that dial down the computational effort for "easy-to-process" input frames represent a promising way to reduce the average inference latency. However, while this approach is extensively studied for classification, its application to dense tasks like object detection (OD) is not straightforward. In this paper, we propose BlankSkip, an adaptive network for on-device OD that leverages a simple auxiliary classification task for early exit, i.e., identifying frames with no objects of interest. With experiments using a real-world nano-drone platform, the Bitcraze Crazyflie 2.1, we achieve up to 24% average throughput improvement with a limited 0.015 mean Average Precision (mAP) drop compared to a static MobileNet-SSD detector, on a state-of-the-art nano-drones OD dataset.
>
---
#### [new 167] Decoupling Wavelet Sub-bands for Single Source Domain Generalization in Fundus Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决视网膜图像域泛化问题。通过引入WaveSDG网络，分离解剖结构与域特异性外观，提升模型在未知域的性能。**

- **链接: [https://arxiv.org/pdf/2603.28463](https://arxiv.org/pdf/2603.28463)**

> **作者:** Shramana Dey; Varun Ajith; Abhirup Banerjee; Sushmita Mitra
>
> **摘要:** Domain generalization in fundus imaging is challenging due to variations in acquisition conditions across devices and clinical settings. The inability to adapt to these variations causes performance degradation on unseen domains for deep learning models. Besides, obtaining annotated data across domains is often expensive and privacy constraints restricts their availability. Although single-source domain generalization (SDG) offers a realistic solution to this problem, the existing approaches frequently fail to capture anatomical topology or decouple appearance from anatomical features. This research introduces WaveSDG, a new wavelet-guided segmentation network for SDG. It decouples anatomical structure from domain-specific appearance through a wavelet sub-band decomposition. A novel Wavelet-based Invariant Structure Extraction and Refinement (WISER) module is proposed to process encoder features by leveraging distinct semantic roles of each wavelet sub-band. The module refines low-frequency components to anchor global anatomy, while selectively enhancing directional edges and suppressing noise within the high-frequency sub-bands. Extensive ablation studies validate the effectiveness of the WISER module and its decoupling strategy. Our evaluations on optic cup and optic disc segmentation across one source and five unseen target datasets show that WaveSDG consistently outperforms seven state-of-the-art methods. Notably, it achieves the best balanced Dice score and lowest 95th percentile Hausdorff distance with reduced variance, indicating improved accuracy, robustness, and cross-domain stability.
>
---
#### [new 168] Event6D: Event-based Novel Object 6D Pose Tracking
- **分类: cs.CV**

- **简介: 该论文属于6D物体位姿跟踪任务，解决传统方法在快速动态场景中因运动模糊而失效的问题。通过事件相机和深度信息，实现对新物体的高效、精准跟踪。**

- **链接: [https://arxiv.org/pdf/2603.28045](https://arxiv.org/pdf/2603.28045)**

> **作者:** Jae-Young Kang; Hoonehee Cho; Taeyeop Lee; Minjun Kang; Bowen Wen; Youngho Kim; Kuk-Jin Yoon
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Event cameras provide microsecond latency, making them suitable for 6D object pose tracking in fast, dynamic scenes where conventional RGB and depth pipelines suffer from motion blur and large pixel displacements. We introduce EventTrack6D, an event-depth tracking framework that generalizes to novel objects without object-specific training by reconstructing both intensity and depth at arbitrary timestamps between depth frames. Conditioned on the most recent depth measurement, our dual reconstruction recovers dense photometric and geometric cues from sparse event streams. Our EventTrack6D operates at over 120 FPS and maintains temporal consistency under rapid motion. To support training and evaluation, we introduce a comprehensive benchmark suite: a large-scale synthetic dataset for training and two complementary evaluation sets, including real and simulated event datasets. Trained exclusively on synthetic data, EventTrack6D generalizes effectively to real-world scenarios without fine-tuning, maintaining accurate tracking across diverse objects and motion patterns. Our method and datasets validate the effectiveness of event cameras for event-based 6D pose tracking of novel objects. Code and datasets are publicly available at this https URL.
>
---
#### [new 169] Falcon Perception
- **分类: cs.CV**

- **简介: 该论文提出Falcon Perception，解决视觉任务中感知与模型融合的问题，通过统一的Transformer架构实现高效密集预测。**

- **链接: [https://arxiv.org/pdf/2603.27365](https://arxiv.org/pdf/2603.27365)**

> **作者:** Aviraj Bevli; Sofian Chaybouti; Yasser Dahou; Hakim Hacid; Ngoc Dung Huynh; Phuc H. Le Khac; Sanath Narayan; Wamiq Reyaz Para; Ankit Singh
>
> **摘要:** Perception-centric systems are typically implemented with a modular encoder-decoder pipeline: a vision backbone for feature extraction and a separate decoder (or late-fusion module) for task prediction. This raises a central question: is this architectural separation essential or can a single early-fusion stack do both perception and task modeling at scale? We introduce Falcon Perception, a unified dense Transformer that processes image patches and text tokens in a shared parameter space from the first layer, using a hybrid attention pattern (bidirectional among image tokens, causal for prediction tokens) to combine global visual context with autoregressive, variable-length instance generation. To keep dense outputs practical, Falcon Perception retains a lightweight token interface and decodes continuous spatial outputs with specialized heads, enabling parallel high-resolution mask prediction. Our design promotes simplicity: we keep a single scalable backbone and shift complexity toward data and training signals, adding only small heads where outputs are continuous and dense. On SA-Co, Falcon Perception improves mask quality to 68.0 Macro-F$_1$ compared to 62.3 of SAM3. We also introduce PBench, a benchmark targeting compositional prompts (OCR, spatial constraints, relations) and dense long-context regimes, where the model shows better gains. Finally, we extend the same early-fusion recipe to Falcon OCR: a compact 300M-parameter model which attains 80.3% on olmOCR and 88.64 on OmniDocBench.
>
---
#### [new 170] TTE-CAM: Built-in Class Activation Maps for Test-Time Explainability in Pretrained Black-Box CNNs
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决CNN模型缺乏可解释性的问题。通过TTE-CAM框架，将预训练CNN转换为自解释模型，保持性能同时提供真实解释。**

- **链接: [https://arxiv.org/pdf/2603.26885](https://arxiv.org/pdf/2603.26885)**

> **作者:** Kerol Djoumessi; Philipp Berens
>
> **备注:** Unlocking Test-Time Explainability from Pretrained Black-Box CNNs
>
> **摘要:** Convolutional neural networks (CNNs) achieve state-of-the-art performance in medical image analysis yet remain opaque, limiting adoption in high-stakes clinical settings. Existing approaches face a fundamental trade-off: post-hoc methods provide unfaithful approximate explanations, while inherently interpretable architectures are faithful but often sacrifice predictive performance. We introduce TTE-CAM, a test-time framework that bridges this gap by converting pretrained black-box CNNs into self-explainable models via a convolution-based replacement of their classification head, initialized from the original weights. The resulting model preserves black-box predictive performance while delivering built-in faithful explanations competitive with post-hoc methods, both qualitatively and quantitatively. The code is available at this https URL
>
---
#### [new 171] Towards Intrinsic-Aware Monocular 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于单目3D目标检测任务，旨在解决相机内参变化导致的检测不稳定性问题。提出MonoIA框架，通过语义表示建模内参变化，提升检测鲁棒性与跨场景泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.27059](https://arxiv.org/pdf/2603.27059)**

> **作者:** Zhihao Zhang; Abhinav Kumar; Xiaoming Liu
>
> **备注:** This paper is accepted by CVPR 2026
>
> **摘要:** Monocular 3D object detection (Mono3D) aims to infer object locations and dimensions in 3D space from a single RGB image. Despite recent progress, existing methods remain highly sensitive to camera intrinsics and struggle to generalize across diverse settings, since intrinsics govern how 3D scenes are projected onto the image plane. We propose MonoIA, a unified intrinsic-aware framework that models and adapts to intrinsic variation through a language-grounded representation. The key insight is that intrinsic variation is not a numeric difference but a perceptual transformation that alters apparent scale, perspective, and spatial geometry. To capture this effect, MonoIA employs large language models and vision-language models to generate intrinsic embeddings that encode the visual and geometric implications of camera parameters. These embeddings are hierarchically integrated into the detection network via an Intrinsic Adaptation Module, allowing the model to modulate its feature representations according to camera-specific configurations and maintain consistent 3D detection across intrinsics. This shifts intrinsic modeling from numeric conditioning to semantic representation, enabling robust and unified perception across cameras. Extensive experiments show that MonoIA achieves new state-of-the-art results on standard benchmarks including KITTI, Waymo, and nuScenes (e.g., +1.18% on the KITTI leaderboard), and further improves performance under multi-dataset training (e.g., +4.46% on KITTI Val).
>
---
#### [new 172] Low Dose CT for Stroke Diagnosis: A Dual Pipeline Deep Learning Framework for Portable Neuroimaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决低剂量CT在卒中诊断中的可靠性问题。通过深度学习框架，比较直接分类与去噪后分类的效果，提升移动医疗环境下的诊断准确性。**

- **链接: [https://arxiv.org/pdf/2603.26764](https://arxiv.org/pdf/2603.26764)**

> **作者:** Rhea Ghosal; Ronok Ghosal; Eileen Lou
>
> **备注:** 13 pages, 4 figures, 3 tables. Includes dose-level evaluation and robustness stress tests (motion and ring artifacts). Code and dataset based on RSNA Intracranial Hemorrhage Detection
>
> **摘要:** Portable CT scanners enable early stroke detection in prehospital and low-resource settings but require reduced radiation doses, introducing noise that degrades diagnostic reliability. We present a deep learning framework for stroke classification from simulated low-dose CT (LDCT) brain scans for AI-assisted triage in mobile clinical environments. Controlled Poisson noise is applied to high-dose CT images to simulate realistic LDCT conditions. We compare two pipelines: (1) direct classification of noisy LDCT images and (2) denoising followed by classification. Performance is evaluated across multiple dose levels using accuracy, sensitivity, and AUC. While denoising improves perceptual image quality, it does not consistently improve classification. In several settings, direct classification yields higher sensitivity, revealing a trade-off between perceptual quality and diagnostic utility. The best denoise-then-classify pipeline achieves 0.94 AUC and 0.91 accuracy at moderate dose levels, outperforming direct classification by up to 6% in select cases. This work establishes a reproducible baseline for LDCT stroke triage using hemorrhagic stroke data (RSNA dataset) and highlights the need for validation on ischemic cohorts and real-world portable CT systems.
>
---
#### [new 173] Domain-Invariant Prompt Learning for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型领域，解决域适应问题。针对现有方法在跨域场景下的不足，提出DiCoOp，通过对抗训练学习域不变提示，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.28555](https://arxiv.org/pdf/2603.28555)**

> **作者:** Arsham Gholamzadeh Khoee; Yinan Yu; Robert Feldt
>
> **摘要:** Large pre-trained vision-language models like CLIP have transformed computer vision by aligning images and text in a shared feature space, enabling robust zero-shot transfer via prompting. Soft-prompting, such as Context Optimization (CoOp), effectively adapts these models for downstream recognition tasks by learning a set of context vectors. However, CoOp lacks explicit mechanisms for handling domain shifts across unseen distributions. To address this, we propose Domain-invariant Context Optimization (DiCoOp), an extension of CoOp optimized for domain generalization. By employing an adversarial training approach, DiCoOp forces the model to learn domain-invariant prompts while preserving discriminative power for classification. Experimental results show that DiCoOp consistently surpasses CoOp in domain generalization tasks across diverse visual domains.
>
---
#### [new 174] Improving Automated Wound Assessment Using Joint Boundary Segmentation and Multi-Class Classification Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决伤口分类与边界分割的临床应用问题。通过联合模型实现多类伤口的准确识别与分割，提升AI在医疗中的实用性。**

- **链接: [https://arxiv.org/pdf/2603.27325](https://arxiv.org/pdf/2603.27325)**

> **作者:** Mehedi Hasan Tusar; Fateme Fayyazbakhsh; Igor Melnychuk; Ming C. Leu
>
> **摘要:** Accurate wound classification and boundary segmentation are essential for guiding clinical decisions in both chronic and acute wound management. However, most existing AI models are limited, focusing on a narrow set of wound types or performing only a single task (segmentation or classification), which reduces their clinical applicability. This study presents a deep learning model based on YOLOv11 that simultaneously performs wound boundary segmentation (WBS) and wound classification (WC) across five clinically relevant wound types: burn injury (BI), pressure injury (PI), diabetic foot ulcer (DFU), vascular ulcer (VU), and surgical wound (SW). A wound-type balanced dataset of 2,963 annotated images was created to train the models for both tasks, with stratified five-fold cross-validation ensuring robust and unbiased evaluation. The models trained on the original non-augmented dataset achieved consistent performance across folds, though BI detection accuracy was relatively lower. Therefore, the dataset was augmented using rotation, flipping, and variations in brightness, saturation, and exposure to help the model learn more generalized and invariant features. This augmentation significantly improved model performance, particularly in detecting visually subtle BI cases. Among tested variants, YOLOv11x achieved the highest performance with F1-scores of 0.9341 (WBS) and 0.8736 (WC), while the lightweight YOLOv11n provided comparable accuracy at lower computational cost, making it suitable for resource-constrained deployments. Supported by confusion matrices and visual detection outputs, the results confirm the model's robustness against complex backgrounds and high intra-class variability, demonstrating the potential of YOLOv11-based architectures for accurate, real-time wound analysis in both clinical and remote care settings.
>
---
#### [new 175] Dual-Path Learning based on Frequency Structural Decoupling and Regional-Aware Fusion for Low-Light Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于低光图像超分辨率任务，解决传统方法串行处理导致的伪影放大、纹理抑制等问题，提出DTP框架，通过频率解耦与区域感知融合实现更优重建。**

- **链接: [https://arxiv.org/pdf/2603.27301](https://arxiv.org/pdf/2603.27301)**

> **作者:** Ji-Xuan He; Jia-Cheng Zhao; Feng-Qi Cui; Jinyang Huang; Yang Liu; Sirui Zhao; Meng Li; Zhi Liu
>
> **摘要:** Low-light image super-resolution (LLISR) is essential for restoring fine visual details and perceptual quality under insufficient illumination conditions with ubiquitous low-resolution devices. Although pioneer methods achieve high performance on single tasks, they solve both tasks in a serial manner, which inevitably leads to artifact amplification, texture suppression, and structural degradation. To address this, we propose Decoupling then Perceive (DTP), a novel frequency-aware framework that explicitly separates luminance and texture into semantically independent components, enabling specialized modeling and coherent reconstruction. Specifically, to adaptively separate the input into low-frequency luminance and high-frequency texture subspaces, we propose a Frequency-aware Structural Decoupling (FSD) mechanism, which lays a solid foundation for targeted representation learning and reconstruction. Based on the decoupled representation, a Semantics-specific Dual-path Representation (SDR) learning strategy that performs targeted enhancement and reconstruction for each frequency component is further designed, facilitating robust luminance adjustment and fine-grained texture recovery. To promote structural consistency and perceptual alignment in the reconstructed output, building upon this dual-path modeling, we further introduce a Cross-frequency Semantic Recomposition (CSR) module that selectively integrates the decoupled representations. Extensive experiments on the most widely used LLISR benchmarks demonstrate the superiority of our DTP framework, improving $+$1.6\% PSNR, $+$9.6\% SSIM, and $-$48\% LPIPS compared to the most state-of-the-art (SOTA) algorithm. Codes are released at this https URL.
>
---
#### [new 176] TIR-Agent: Training an Explorative and Efficient Agent for Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决现有方法依赖启发式调度导致效率低下的问题。提出TIR-Agent，通过两阶段训练实现高效工具调用策略。**

- **链接: [https://arxiv.org/pdf/2603.27742](https://arxiv.org/pdf/2603.27742)**

> **作者:** Yisheng Zhang; Guoli Jia; Haote Hu; Shanxu Zhao; Kaikai Zhao; Long Sun; Xinwei Long; Kai Tian; Che Jiang; Zhaoxiang Liu; Kai Wang; Shiguo Lian; Kaiyan Zhang; Bowen Zhou
>
> **摘要:** Vision-language agents that orchestrate specialized tools for image restoration (IR) have emerged as a promising method, yet most existing frameworks operate in a training-free manner. They rely on heuristic task scheduling and exhaustive tool traversal, resulting in sub-optimal restoration paths and prohibitive computational cost. We argue that the core bottleneck lies in the absence of a learned policy to make decision, as a vision-language model cannot efficiently handle degradation-aware task ordering and tool composition. To this end, we propose TIR-Agent, a trainable image restoration agent that performs a direct tool-calling policy through a two-stage training pipeline of supervised fine-tuning (SFT) followed by reinforcement learning (RL). Two key designs underpin effective RL training: (i) a random perturbation strategy applied to the SFT data, which broadens the policy's exploration over task schedules and tool compositions, and (ii) a multi-dimensional adaptive reward mechanism that dynamically re-weights heterogeneous image quality metrics to mitigate reward hacking. To support high-throughput, asynchronous GPU-based tool invocation during training, we further develop a globally shared model-call pool. Experiments on both in-domain and out-of-domain degradations show that TIR-Agent outperforms 12 baselines, including 6 all-in-one models, 3 training-free agents, and 3 proprietary models, and achieves over 2.5$\times$ inference speedup by eliminating redundant tool executions.
>
---
#### [new 177] A Robust Low-Rank Prior Model for Structured Cartoon-Texture Image Decomposition with Heavy-Tailed Noise
- **分类: cs.CV; math.OC**

- **简介: 该论文属于图像分解任务，旨在解决重尾噪声下的卡通-纹理分解问题。通过引入鲁棒的低秩先验模型和Huber损失函数，提升分解的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.27579](https://arxiv.org/pdf/2603.27579)**

> **作者:** Weihao Tang; Hongjin He
>
> **备注:** This paper introduces a robust model for cartoon-texture image decomposition with heavy-tailed noise. It has 11 figures and 4 tables
>
> **摘要:** Cartoon-texture image decomposition is a fundamental yet challenging problem in image processing. A significant hurdle in achieving accurate decomposition is the pervasive presence of noise in the observed images, which severely impedes robust results. To address the challenging problem of cartoon-texture decomposition in the presence of heavy-tailed noise, we in this paper propose a robust low-rank prior model. Our approach departs from conventional models by adopting the Huber loss function as the data-fidelity term, rather than the traditional $\ell_2$-norm, while retaining the total variation norm and nuclear norm to characterize the cartoon and texture components, respectively. Given the inherent structure, we employ two implementable operator splitting algorithms, tailored to different degradation operators. Extensive numerical experiments, particularly on image restoration tasks under high-intensity heavy-tailed noise, efficiently demonstrate the superior performance of our model.
>
---
#### [new 178] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ChartNet，一个大规模多模态数据集，用于提升图表理解能力。解决现有模型在图表视觉、数据与语言联合推理上的不足，通过生成多样化图表样本及精细对齐的多模态数据，推动图表解释与推理研究。**

- **链接: [https://arxiv.org/pdf/2603.27064](https://arxiv.org/pdf/2603.27064)**

> **作者:** Jovana Kondic; Pengyuan Li; Dhiraj Joshi; Isaac Sanchez; Ben Wiesel; Shafiq Abedin; Amit Alfassy; Eli Schwartz; Daniel Caraballo; Yagmur Gizem Cinar; Florian Scheidegger; Steven I. Ross; Daniel Karl I. Weidele; Hang Hua; Ekaterina Arutyunova; Roei Herzig; Zexue He; Zihan Wang; Xinyue Yu; Yunfei Zhao; Sicong Jiang; Minghao Liu; Qunshu Lin; Peter Staar; Luis Lastras; Aude Oliva; Rogerio Feris
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Understanding charts requires models to jointly reason over geometric visual patterns, structured numerical data, and natural language -- a capability where current vision-language models (VLMs) remain limited. We introduce ChartNet, a high-quality, million-scale multimodal dataset designed to advance chart interpretation and reasoning. ChartNet leverages a novel code-guided synthesis pipeline to generate 1.5 million diverse chart samples spanning 24 chart types and 6 plotting libraries. Each sample consists of five aligned components: plotting code, rendered chart image, data table, natural language summary, and question-answering with reasoning, providing fine-grained cross-modal alignment. To capture the full spectrum of chart comprehension, ChartNet additionally includes specialized subsets encompassing human annotated data, real-world data, safety, and grounding. Moreover, a rigorous quality-filtering pipeline ensures visual fidelity, semantic accuracy, and diversity across chart representations. Fine-tuning on ChartNet consistently improves results across benchmarks, demonstrating its utility as large-scale supervision for multimodal models. As the largest open-source dataset of its kind, ChartNet aims to support the development of foundation models with robust and generalizable capabilities for data visualization understanding. The dataset is publicly available at this https URL
>
---
#### [new 179] Chat-Scene++: Exploiting Context-Rich Object Identification for 3D LLM
- **分类: cs.CV**

- **简介: 该论文提出Chat-Scene++，解决3D场景中细粒度物体定位与上下文推理问题，通过构建上下文丰富的物体序列实现高效场景理解与交互。**

- **链接: [https://arxiv.org/pdf/2603.27507](https://arxiv.org/pdf/2603.27507)**

> **作者:** Haifeng Huang; Yilun Chen; Zehan Wang; Jiangmiao Pang; Zhou Zhao
>
> **摘要:** Recent advancements in multi-modal large language models (MLLMs) have shown strong potential for 3D scene understanding. However, existing methods struggle with fine-grained object grounding and contextual reasoning, limiting their ability to interpret and interact with complex 3D environments. In this paper, we present Chat-Scene++, an MLLM framework that represents 3D scenes as context-rich object sequences. By structuring scenes as sequences of objects with contextual semantics, Chat-Scene++ enables object-centric representation and interaction. It decomposes a 3D scene into object representations paired with identifier tokens, allowing LLMs to follow instructions across diverse 3D vision-language tasks. To capture inter-object relationships and global semantics, Chat-Scene++ extracts context-rich object features using large-scale pre-trained 3D scene-level and 2D image-level encoders, unlike the isolated per-object features in Chat-Scene. Its flexible object-centric design also supports grounded chain-of-thought (G-CoT) reasoning, enabling the model to distinguish objects at both category and spatial levels during multi-step inference. Without the need for additional task-specific heads or fine-tuning, Chat-Scene++ achieves state-of-the-art performance on five major 3D vision-language benchmarks: ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, and SQA3D. These results highlight its effectiveness in scene comprehension, object grounding, and spatial reasoning. Additionally, without reconstructing 3D worlds through computationally expensive processes, we demonstrate its applicability to real-world scenarios using only 2D inputs.
>
---
#### [new 180] SEA: Evaluating Sketch Abstraction Efficiency via Element-level Commonsense Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文属于图像理解任务，旨在解决 sketches 抽象效率评估问题。提出 SEA 指标和 CommonSketch 数据集，以量化 sketch 中语义元素的经济表达与可识别性。**

- **链接: [https://arxiv.org/pdf/2603.28363](https://arxiv.org/pdf/2603.28363)**

> **作者:** Jiho Park; Sieun Choi; Jaeyoon Seo; Minho Sohn; Yeana Kim; Jihie Kim
>
> **摘要:** A sketch is a distilled form of visual abstraction that conveys core concepts through simplified yet purposeful strokes while omitting extraneous detail. Despite its expressive power, quantifying the efficiency of semantic abstraction in sketches remains challenging. Existing evaluation methods that rely on reference images, low-level visual features, or recognition accuracy do not capture abstraction, the defining property of sketches. To address these limitations, we introduce SEA (Sketch Evaluation metric for Abstraction efficiency), a reference-free metric that assesses how economically a sketch represents class-defining visual elements while preserving semantic recognizability. These elements are derived per class from commonsense knowledge about features typically depicted in sketches. SEA leverages a visual question answering model to determine the presence of each element and returns a quantitative score that reflects semantic retention under visual economy. To support this metric, we present CommonSketch, the first semantically annotated sketch dataset, comprising 23,100 human-drawn sketches across 300 classes, each paired with a caption and element-level annotations. Experiments show that SEA aligns closely with human judgments and reliably discriminates levels of abstraction efficiency, while CommonSketch serves as a benchmark providing systematic evaluation of element-level sketch understanding across various vision-language models.
>
---
#### [new 181] UniDA3D: A Unified Domain-Adaptive Framework for Multi-View 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于多视角3D目标检测任务，旨在解决复杂环境下检测性能下降的问题。提出UniDA3D框架，通过域自适应提升不同天气条件下的检测效果。**

- **链接: [https://arxiv.org/pdf/2603.27995](https://arxiv.org/pdf/2603.27995)**

> **作者:** Hongjing Wu; Cheng Chi; Jinlin Wu; Yanzhao Su; Zhen Lei; Wenqi Ren
>
> **摘要:** Camera-only 3D object detection is critical for autonomous driving, offering a cost-effective alternative to LiDAR based methods. In particular, multi-view 3D object detection has emerged as a promising direction due to its balanced trade-off between performance and cost. However, existing methods often suffer significant performance degradation under complex environmental conditions such as nighttime, fog, and rain, primarily due to their reliance on training data collected mostly in ideal conditions. To address this challenge, we propose UniDA3D, a unified domain-adaptive multi-view 3D object detector designed for robust perception under diverse adverse conditions. UniDA3D formulates nighttime, rainy, and foggy scenes as a unified multi target domain adaptation problem and leverages a novel query guided domain discrepancy mitigation (QDDM) module to align object features between source and target domains at both batch and global levels via query-centric adversarial and contrastive learning. Furthermore, we introduce a domain-adaptive teacher student training pipeline with an exponential-moving-average teacher and dynamically updated high-quality pseudo labels to enhance consistency learning and suppress background noise in unlabeled target domains. In contrast to prior approaches that require separate training for each condition, UniDA3D performs a single unified training process across multiple domains, enabling robust all-weather 3D perception. On a synthesized multi-view 3D benchmark constructed by generating nighttime, rainy, and foggy counterparts from nuScenes (nuScenes-Night, nuScenes-Rain, and nuScenes-Haze), UniDA3D consistently outperforms state of-the-art camera-only multi-view 3D detectors under extreme conditions, achieving substantial gains in mAP and NDS while maintaining real-time inference efficiency.
>
---
#### [new 182] Explaining CLIP Zero-shot Predictions Through Concepts
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言理解任务，旨在解决CLIP模型预测不透明的问题。通过引入EZPC方法，将CLIP的嵌入投影到可解释的概念空间，实现零样本分类的可解释性。**

- **链接: [https://arxiv.org/pdf/2603.28211](https://arxiv.org/pdf/2603.28211)**

> **作者:** Onat Ozdemir; Anders Christensen; Stephan Alaniz; Zeynep Akata; Emre Akbas
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Large-scale vision-language models such as CLIP have achieved remarkable success in zero-shot image recognition, yet their predictions remain largely opaque to human understanding. In contrast, Concept Bottleneck Models provide interpretable intermediate representations by reasoning through human-defined concepts, but they rely on concept supervision and lack the ability to generalize to unseen classes. We introduce EZPC that bridges these two paradigms by explaining CLIP's zero-shot predictions through human-understandable concepts. Our method projects CLIP's joint image-text embeddings into a concept space learned from language descriptions, enabling faithful and transparent explanations without additional supervision. The model learns this projection via a combination of alignment and reconstruction objectives, ensuring that concept activations preserve CLIP's semantic structure while remaining interpretable. Extensive experiments on five benchmark datasets, CIFAR-100, CUB-200-2011, Places365, ImageNet-100, and ImageNet-1k, demonstrate that our approach maintains CLIP's strong zero-shot classification accuracy while providing meaningful concept-level explanations. By grounding open-vocabulary predictions in explicit semantic concepts, our method offers a principled step toward interpretable and trustworthy vision-language models. Code is available at this https URL.
>
---
#### [new 183] XSPA: Crafting Imperceptible X-Shaped Sparse Adversarial Perturbations for Transferable Attacks on VLMs
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型的鲁棒性问题，提出XSPA攻击方法，在稀疏且几何固定条件下生成扰动，验证模型对微小扰动的敏感性。**

- **链接: [https://arxiv.org/pdf/2603.28568](https://arxiv.org/pdf/2603.28568)**

> **作者:** Chengyin Hu; Jiaju Han; Xuemeng Sun; Qike Zhang; Yiwei Wei; Ang Li; Chunlei Meng; Xiang Chen; Jiahuan Long
>
> **摘要:** Vision-language models (VLMs) rely on a shared visual-textual representation space to perform tasks such as zero-shot classification, image captioning, and visual question answering (VQA). While this shared space enables strong cross-task generalization, it may also introduce a common vulnerability: small visual perturbations can propagate through the shared embedding space and cause correlated semantic failures across tasks. This risk is particularly important in interactive and decision-support settings, yet it remains unclear whether VLMs are robust to highly constrained, sparse, and geometrically fixed perturbations. To address this question, we propose X-shaped Sparse Pixel Attack (XSPA), an imperceptible structured attack that restricts perturbations to two intersecting diagonal lines. Compared with dense perturbations or flexible localized patches, XSPA operates under a much stricter attack budget and thus provides a more stringent test of VLM robustness. Within this sparse support, XSPA jointly optimizes a classification objective, cross-task semantic guidance, and regularization on perturbation magnitude and along-line smoothness, inducing transferable misclassification as well as semantic drift in captioning and VQA while preserving visual subtlety. Under the default setting, XSPA modifies only about 1.76% of image pixels. Experiments on the COCO dataset show that XSPA consistently degrades performance across all three tasks. Zero-shot accuracy drops by 52.33 points on OpenAI CLIP ViT-L/14 and 67.00 points on OpenCLIP ViT-B/16, while GPT-4-evaluated caption consistency decreases by up to 58.60 points and VQA correctness by up to 44.38 points. These results suggest that even highly sparse and visually subtle perturbations with fixed geometric priors can substantially disrupt cross-task semantics in VLMs, revealing a notable robustness gap in current multimodal systems.
>
---
#### [new 184] Benchmarking Multi-View BEV Object Detection with Mixed Pinhole and Fisheye Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D目标检测任务，解决混合相机（针孔与鱼眼）下BEV检测性能下降的问题。通过数据转换、视图变换模块和极坐标表示等方法，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27818](https://arxiv.org/pdf/2603.27818)**

> **作者:** Xiangzhong Liu; Hao Shen
>
> **备注:** 8 pages,5 figures, IEEE International Conference on Robotics and Automation (ICRA),Vienna, Austria, 1-5 June 2026
>
> **摘要:** Modern autonomous driving systems increasingly rely on mixed camera configurations with pinhole and fisheye cameras for full view perception. However, Bird's-Eye View (BEV) 3D object detection models are predominantly designed for pinhole cameras, leading to performance degradation under fisheye distortion. To bridge this gap, we introduce a multi-view BEV detection benchmark with mixed cameras by converting KITTI-360 into nuScenes format. Our study encompasses three adaptations: rectification for zero-shot evaluation and fine-tuning of nuScenes-trained models, distortion-aware view transformation modules (VTMs) via the MEI camera model, and polar coordinate representations to better align with radial distortion. We systematically evaluate three representative BEV architectures, BEVFormer, BEVDet and PETR, across these strategies. We demonstrate that projection-free architectures are inherently more robust and effective against fisheye distortion than other VTMs. This work establishes the first real-data 3D detection benchmark with fisheye and pinhole images and provides systematic adaptation and practical guidelines for designing robust and cost-effective 3D perception systems. The code is available at this https URL.
>
---
#### [new 185] NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather
- **分类: cs.CV**

- **简介: 该论文提出NimbusGS，用于在混合恶劣天气下进行统一的3D场景重建，解决多视角输入下的退化问题。**

- **链接: [https://arxiv.org/pdf/2603.27228](https://arxiv.org/pdf/2603.27228)**

> **作者:** Yanying Li; Jinyang Li; Shengfeng He; Yangyang Xu; Junyu Dong; Yong Du
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** We present NimbusGS, a unified framework for reconstructing high-quality 3D scenes from degraded multi-view inputs captured under diverse and mixed adverse weather conditions. Unlike existing methods that target specific weather types, NimbusGS addresses the broader challenge of generalization by modeling the dual nature of weather: a continuous, view-consistent medium that attenuates light, and dynamic, view-dependent particles that cause scattering and occlusion. To capture this structure, we decompose degradations into a global transmission field and per-view particulate residuals. The transmission field represents static atmospheric effects shared across views, while the residuals model transient disturbances unique to each input. To enable stable geometry learning under severe visibility degradation, we introduce a geometry-guided gradient scaling mechanism that mitigates gradient imbalance during the self-supervised optimization of 3D Gaussian representations. This physically grounded formulation allows NimbusGS to disentangle complex degradations while preserving scene structure, yielding superior geometry reconstruction and outperforming task-specific methods across diverse and challenging weather conditions. Code is available at this https URL.
>
---
#### [new 186] ExFusion: Efficient Transformer Training via Multi-Experts Fusion
- **分类: cs.CV**

- **简介: 该论文提出ExFusion方法，解决MoE模型训练成本高的问题，通过多专家融合提升Transformer效率，实现性能增强且计算开销小。**

- **链接: [https://arxiv.org/pdf/2603.27965](https://arxiv.org/pdf/2603.27965)**

> **作者:** Jiacheng Ruan; Daize Dong; Xiaoye Qu; Tong Zhu; Ting Liu; Yuzhuo Fu; Yu Cheng; Suncheng Xiang
>
> **备注:** Accepted by IEEE TMM2026
>
> **摘要:** Mixture-of-Experts (MoE) models substantially improve performance by increasing the capacity of dense architectures. However, directly training MoE models requires considerable computational resources and introduces extra overhead in parameter storage and deployment. Therefore, it is critical to develop an approach that leverages the multi-expert capability of MoE to enhance performance while incurring minimal additional cost. To this end, we propose a novel pre-training approach, termed ExFusion, which improves the efficiency of Transformer training through multi-expert fusion. Specifically, during the initialization phase, ExFusion upcycles the feed-forward network (FFN) of the Transformer into a multi-expert configuration, where each expert is assigned a weight for later parameter fusion. During training, these weights allow multiple experts to be fused into a single unified expert equivalent to the original FFN, which is subsequently used for forward computation. As a result, ExFusion introduces multi-expert characteristics into the training process while incurring only marginal computational cost compared to standard dense training. After training, the learned weights are used to integrate multi-experts into a single unified expert, thereby eliminating additional overhead in storage and deployment. Extensive experiments on a variety of computer vision and natural language processing tasks demonstrate the effectiveness of the proposed method.
>
---
#### [new 187] Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal
- **分类: cs.CV**

- **简介: 该论文属于LiDAR数据处理任务，旨在解决移动LiDAR中鬼影点干扰问题。通过构建大规模FWL数据集并提出新模型提升鬼影检测与移除效果，优化3D定位与目标检测性能。**

- **链接: [https://arxiv.org/pdf/2603.28224](https://arxiv.org/pdf/2603.28224)**

> **作者:** Kazuma Ikeda; Ryosei Hara; Rokuto Nagata; Ozora Sako.Zihao Ding; Takahiro Kado; Ibuki Fujioka; Taro Beppu; Mariko Isogawa; Kentaro Yoshioka
>
> **备注:** Accepted to CVPR 2026 (Main)
>
> **摘要:** LiDAR has become an essential sensing modality in autonomous driving, robotics, and smart-city applications. However, ghost points (or ghosts), which are false reflections caused by multi-path laser returns from glass and reflective surfaces, severely degrade 3D mapping and localization accuracy. Prior ghost removal relies on geometric consistency in dense point clouds, failing on mobile LiDAR's sparse, dynamic data. We address this by exploiting full-waveform LiDAR (FWL), which captures complete temporal intensity profiles rather than just peak distances, providing crucial cues for distinguishing ghosts from genuine reflections in mobile scenarios. As this is a new task, we present Ghost-FWL, the first and largest annotated mobile FWL dataset for ghost detection and removal. Ghost-FWL comprises 24K frames across 10 diverse scenes with 7.5 billion peak-level annotations, which is 100x larger than existing annotated FWL datasets. Benefiting from this large-scale dataset, we establish a FWL-based baseline model for ghost detection and propose FWL-MAE, a masked autoencoder for efficient self-supervised representation learning on FWL data. Experiments show that our baseline outperforms existing methods in ghost removal accuracy, and our ghost removal further enhances downstream tasks such as LiDAR-based SLAM (66% trajectory error reduction) and 3D object detection (50x false positive reduction). The dataset and code is publicly available and can be accessed via the project page: this https URL
>
---
#### [new 188] Structured Observation Language for Efficient and Generalizable Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决环境变化下的泛化能力不足问题。提出SOL-Nav框架，将视觉信息转化为结构化语言描述，提升导航效率与泛化性。**

- **链接: [https://arxiv.org/pdf/2603.27577](https://arxiv.org/pdf/2603.27577)**

> **作者:** Daojie Peng; Fulong Ma; Jun Ma
>
> **摘要:** Vision-Language Navigation (VLN) requires an embodied agent to navigate complex environments by following natural language instructions, which typically demands tight fusion of visual and language modalities. Existing VLN methods often convert raw images into visual tokens or implicit features, requiring large-scale visual pre-training and suffering from poor generalization under environmental variations (e.g., lighting, texture). To address these issues, we propose SOL-Nav (Structured Observation Language for Navigation), a novel framework that translates egocentric visual observations into compact structured language descriptions for efficient and generalizable navigation. Specifically, we divide RGB-D images into a N*N grid, extract representative semantic, color, and depth information for each grid cell to form structured text, and concatenate this with the language instruction as pure language input to a pre-trained language model (PLM). Experimental results on standard VLN benchmarks (R2R, RxR) and real-world deployments demonstrate that SOL-Nav significantly reduces the model size and training data dependency, fully leverages the reasoning and representation capabilities of PLMs, and achieves strong generalization to unseen environments.
>
---
#### [new 189] Motion Semantics Guided Normalizing Flow for Privacy-Preserving Video Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在解决骨架方法在建模人类活动时缺乏层次性的问题。通过分解运动语义，提升异常检测的准确性。**

- **链接: [https://arxiv.org/pdf/2603.26745](https://arxiv.org/pdf/2603.26745)**

> **作者:** Yang Liu; Boan Chen; Yuanyuan Meng; Jing Liu; Zhengliang Guo; Wei Zhou; Peng Sun; Hong Chen
>
> **备注:** Accepted to IEEE ICME 2026
>
> **摘要:** As embodied perception systems increasingly bridge digital and physical realms in interactive multimedia applications, the need for privacy-preserving approaches to understand human activities in physical environments has become paramount. Video anomaly detection is a critical task in such embodied multimedia systems for intelligent surveillance and forensic analysis. Skeleton-based approaches have emerged as a privacy-preserving alternative that processes physical world information through abstract human pose representations while discarding sensitive visual attributes such as identity and facial features. However, existing skeleton-based methods predominantly model continuous motion trajectories in a monolithic manner, failing to capture the hierarchical nature of human activities composed of discrete semantic primitives and fine-grained kinematic details, which leads to reduced discriminability when anomalies manifest at different abstraction levels. In this regard, we propose Motion Semantics Guided Normalizing Flow (MSG-Flow) that decomposes skeleton-based VAD into hierarchical motion semantics modeling. It employs vector quantized variational auto-encoder to discretize continuous motion into interpretable primitives, an autoregressive Transformer to model semantic-level temporal dependencies, and a conditional normalizing flow to capture detail-level pose variations. Extensive experiments on benchmarks (HR-ShanghaiTech & HR-UBnormal) demonstrate that MSG-Flow achieves state-of-the-art performance with 88.1% and 75.8% AUC respectively.
>
---
#### [new 190] TrendGen: An Outfit Recommendation and Display System
- **分类: cs.CV**

- **简介: 该论文提出TrendGen，一个用于时尚推荐和展示的AI系统。解决真实场景下图像质量差的问题，通过生成高质量服装视图和推荐搭配，提升电商购物体验。**

- **链接: [https://arxiv.org/pdf/2603.27264](https://arxiv.org/pdf/2603.27264)**

> **作者:** Theodoros Koukopoulos; Dimos Klimenof; Ioannis Xarchakos
>
> **摘要:** Recent advances in Computer Vision have significantly improved image understanding and generation, revolutionizing the fashion industry. However, challenges such as inconsistent lighting, non-ideal garment angles, complex backgrounds, and occlusions in raw images hinder their full potential. Overcoming these obstacles is crucial for developing robust fashion AI systems capable of real-world applications. In this paper, we introduce TrendGen, a Fashion AI system designed to enhance online shopping with intelligent outfit recommendations. Deployed on a major e-commerce platform, TrendGen leverages cloth images and product attributes to generate trend-aligned, cohesive outfit suggestions. Additionally, it employs Generative AI to transform raw images into high-quality lay-down views, offering a clear and structured presentation of garments. Our evaluation on production data demonstrates TrendGen's consistent high-quality outfits and lay-down images, marking a significant advancement in AI-driven solutions for fashion retail.
>
---
#### [new 191] RehearsalNeRF: Decoupling Intrinsic Neural Fields of Dynamic Illuminations for Scene Editing
- **分类: cs.CV**

- **简介: 该论文属于场景编辑任务，解决动态光照下神经辐射场的解耦问题。通过利用稳定光照下的场景数据，学习分离光照与场景辐射，实现更鲁棒的视图合成与编辑。**

- **链接: [https://arxiv.org/pdf/2603.27948](https://arxiv.org/pdf/2603.27948)**

> **作者:** Changyeon Won; Hyunjun Jung; Jungu Cho; Seonmi Park; Chi-Hoon Lee; Hae-Gon Jeon
>
> **备注:** Accepted to the International Journal of Computer Vision (IJCV). Changyeon Won and Hyunjun Jung contributed equally to this work
>
> **摘要:** Although there has been significant progress in neural radiance fields, an issue on dynamic illumination changes still remains unsolved. Different from relevant works that parameterize time-variant/-invariant components in scenes, subjects' radiance is highly entangled with their own emitted radiance and lighting colors in spatio-temporal domain. In this paper, we present a new effective method to learn disentangled neural fields under the severe illumination changes, named RehearsalNeRF. Our key idea is to leverage scenes captured under stable lighting like rehearsal stages, easily taken before dynamic illumination occurs, to enforce geometric consistency between the different lighting conditions. In particular, RehearsalNeRF employs a learnable vector for lighting effects which represents illumination colors in a temporal dimension and is used to disentangle projected light colors from scene radiance. Furthermore, our RehearsalNeRF is also able to reconstruct the neural fields of dynamic objects by simply adopting off-the-shelf interactive masks. To decouple the dynamic objects, we propose a new regularization leveraging optical flow, which provides coarse supervision for the color disentanglement. We demonstrate the effectiveness of RehearsalNeRF by showing robust performances on novel view synthesis and scene editing under dynamic illumination conditions. Our source code and video datasets will be publicly available.
>
---
#### [new 192] 3-D Representations for Hyperspectral Flame Tomography
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于火焰重建任务，旨在比较不同3-D表示方法在热化学重构中的效果。通过实验验证，发现带总变分正则化的体素网格方法在精度和效率上表现最佳。**

- **链接: [https://arxiv.org/pdf/2603.27832](https://arxiv.org/pdf/2603.27832)**

> **作者:** Nicolas Tricard; Zituo Chen; Sili Deng
>
> **备注:** 7 pages, 2 figures, 1 table
>
> **摘要:** Flame tomography is a compelling approach for extracting large amounts of data from experiments via 3-D thermochemical reconstruction. Recent efforts employing neural-network flame representations have suggested improved reconstruction quality compared with classical tomography approaches, but a rigorous quantitative comparison with the same algorithm using a voxel-grid representation has not been conducted. Here, we compare a classical voxel-grid representation with varying regularizers to a continuous neural representation for tomographic reconstruction of a simulated pool fire. The representations are constructed to give temperature and composition as a function of location, and a subsequent ray-tracing step is used to solve the radiative transfer equation to determine the spectral intensity incident on hyperspectral infrared cameras, which is then convolved with an instrument lineshape function. We demonstrate that the voxel-grid approach with a total-variation regularizer reproduces the ground-truth synthetic flame with the highest accuracy for reduced memory intensity and runtime. Future work will explore more representations and under experimental configurations.
>
---
#### [new 193] MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures
- **分类: cs.CV**

- **简介: 该论文属于化学结构识别任务，解决多模态Markush结构识别问题。提出MarkushGrapher-2模型，结合OCR、视觉与文本编码，提升识别精度。**

- **链接: [https://arxiv.org/pdf/2603.28550](https://arxiv.org/pdf/2603.28550)**

> **作者:** Tim Strohmeyer; Lucas Morin; Gerhard Ingmar Meijer; Valéry Weber; Ahmed Nassar; Peter Staar
>
> **备注:** 15 pages, to be published in CVPR 2026
>
> **摘要:** Automatically extracting chemical structures from documents is essential for the large-scale analysis of the literature in chemistry. Automatic pipelines have been developed to recognize molecules represented either in figures or in text independently. However, methods for recognizing chemical structures from multimodal descriptions (Markush structures) lag behind in precision and cannot be used for automatic large-scale processing. In this work, we present MarkushGrapher-2, an end-to-end approach for the multimodal recognition of chemical structures in documents. First, our method employs a dedicated OCR model to extract text from chemical images. Second, the text, image, and layout information are jointly encoded through a Vision-Text-Layout encoder and an Optical Chemical Structure Recognition vision encoder. Finally, the resulting encodings are effectively fused through a two-stage training strategy and used to auto-regressively generate a representation of the Markush structure. To address the lack of training data, we introduce an automatic pipeline for constructing a large-scale dataset of real-world Markush structures. In addition, we present IP5-M, a large manually-annotated benchmark of real-world Markush structures, designed to advance research on this challenging task. Extensive experiments show that our approach substantially outperforms state-of-the-art models in multimodal Markush structure recognition, while maintaining strong performance in molecule structure recognition. Code, models, and datasets are released publicly.
>
---
#### [new 194] Leveraging Avatar Fingerprinting: A Multi-Generator Photorealistic Talking-Head Public Database and Benchmark
- **分类: cs.CV**

- **简介: 该论文属于身份验证任务，旨在解决AI生成头像的指纹识别问题。构建了AVAPrintDB数据库，并定义了基准测试，以评估不同生成器下的指纹识别系统性能。**

- **链接: [https://arxiv.org/pdf/2603.26934](https://arxiv.org/pdf/2603.26934)**

> **作者:** Laura Pedrouzo-Rodriguez; Luis F. Gomez; Ruben Tolosana; Ruben Vera-Rodriguez; Roberto Daza; Aythami Morales; Julian Fierrez
>
> **摘要:** Recent advances in photorealistic avatar generation have enabled highly realistic talking-head avatars, raising security concerns regarding identity impersonation in AI-mediated communication. To advance in this challenging problem, the task of avatar fingerprinting aims to determine whether two avatar videos are driven by the same human operator or not. However, current public databases in the literature are scarce and based solely on old-fashioned talking-head avatar generators, not representing realistic scenarios for the current task of avatar fingerprinting. To overcome this situation, the present article introduces AVAPrintDB, a new publicly available multi-generator talking-head avatar database for avatar fingerprinting. AVAPrintDB is constructed from two audiovisual corpora and three state-of-the-art avatar generators (GAGAvatar, LivePortrait, HunyuanPortrait), representing different synthesis paradigms, and includes both self- and cross-reenactments to simulate legitimate usage and impersonation scenarios. Building on this database, we also define a standardized and reproducible benchmark for avatar fingerprinting, considering public state-of-the-art avatar fingerprinting systems and exploring novel methods based on Foundation Models (DINOv2 and CLIP). Also, we conduct a comprehensive analysis under generator and dataset shift. Our results show that, while identity-related motion cues persist across synthetic avatars, current avatar fingerprinting systems remain highly sensitive to changes in the synthesis pipeline and source domain. The AVAPrintDB, benchmark protocols, and avatar fingerprinting systems are publicly available to facilitate reproducible research.
>
---
#### [new 195] Generative Shape Reconstruction with Geometry-Guided Langevin Dynamics
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D形状重建任务，解决从不完整或噪声数据中重建完整形状的问题。提出GG-Langevin方法，结合扩散模型与测量一致性，提升重建精度和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27016](https://arxiv.org/pdf/2603.27016)**

> **作者:** Linus Härenstam-Nielsen; Dmitrii Pozdeev; Thomas Dagès; Nikita Araslanov; Daniel Cremers
>
> **摘要:** Reconstructing complete 3D shapes from incomplete or noisy observations is a fundamentally ill-posed problem that requires balancing measurement consistency with shape plausibility. Existing methods for shape reconstruction can achieve strong geometric fidelity in ideal conditions but fail under realistic conditions with incomplete measurements or noise. At the same time, recent generative models for 3D shapes can synthesize highly realistic and detailed shapes but fail to be consistent with observed measurements. In this work, we introduce GG-Langevin: Geometry-Guided Langevin dynamics, a probabilistic approach that unifies these complementary perspectives. By traversing the trajectories of Langevin dynamics induced by a diffusion model, while preserving measurement consistency at every step, we generatively reconstruct shapes that fit both the measurements and the data-informed prior. We demonstrate through extensive experiments that GG-Langevin achieves higher geometric accuracy and greater robustness to missing data than existing methods for surface reconstruction.
>
---
#### [new 196] SPROUT: A Scalable Diffusion Foundation Model for Agricultural Vision
- **分类: cs.CV**

- **简介: 该论文提出SPROUT，一个用于农业视觉的可扩展扩散基础模型，解决农业领域与通用视觉模型间的领域差异问题。通过无监督训练学习结构感知表示，提升农业任务性能。**

- **链接: [https://arxiv.org/pdf/2603.27519](https://arxiv.org/pdf/2603.27519)**

> **作者:** Shuai Xiang; Wei Guo; James Burridge; Shouyang Liu; Hao Lu; Tokihiro Fukatsu
>
> **摘要:** Vision Foundation Models (VFM) pre-trained on large-scale unlabeled data have achieved remarkable success on general computer vision tasks, yet typically suffer from significant domain gaps when applied to agriculture. In this context, we introduce $SPROUT$ ($S$calable $P$lant $R$epresentation model via $O$pen-field $U$nsupervised $T$raining), a multi-crop, multi-task agricultural foundation model trained via diffusion denoising. SPROUT leverages a VAE-free Pixel-space Diffusion Transformer to learn rich, structure-aware representations through denoising and enabling efficient end-to-end training. We pre-train SPROUT on a curated dataset of 2.6 million high-quality agricultural images spanning diverse crops, growth stages, and environments. Extensive experiments demonstrate that SPROUT consistently outperforms state-of-the-art web-pretrained and agricultural foundation models across a wide range of downstream tasks, while requiring substantially lower pre-training cost. The code and model are available at this https URL.
>
---
#### [new 197] Evaluating Large and Lightweight Vision Models for Irregular Component Segmentation in E-Waste Disassembly
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标分割任务，旨在解决电子垃圾拆解中不规则组件的精准分割问题。通过对比SAM2与YOLOv8模型，评估其在复杂场景下的性能。**

- **链接: [https://arxiv.org/pdf/2603.27441](https://arxiv.org/pdf/2603.27441)**

> **作者:** Xinyao Zhang; Chang Liu; Xiao Liang; Minghui Zheng; Sara Behdad
>
> **备注:** Accepted at ASME MSEC2026
>
> **摘要:** Precise segmentation of irregular and densely arranged components is essential for robotic disassembly and material recovery in electronic waste (e-waste) recycling. This study evaluates the impact of model architecture and scale on segmentation performance by comparing SAM2, a transformer-based vision model, with the lightweight YOLOv8 network. Both models were trained and tested on a newly collected dataset of 1,456 annotated RGB images of laptop components including logic boards, heat sinks, and fans, captured under varying illumination and orientation conditions. Data augmentation techniques, such as random rotation, flipping, and cropping, were applied to improve model robustness. YOLOv8 achieved higher segmentation accuracy (mAP50 = 98.8%, mAP50-95 = 85%) and stronger boundary precision than SAM2 (mAP50 = 8.4%). SAM2 demonstrated flexibility in representing diverse object structures but often produced overlapping masks and inconsistent contours. These findings show that large pre-trained models require task-specific optimization for industrial applications. The resulting dataset and benchmarking framework provide a foundation for developing scalable vision algorithms for robotic e-waste disassembly and circular manufacturing systems.
>
---
#### [new 198] UniDAC: Universal Metric Depth Estimation for Any Camera
- **分类: cs.CV**

- **简介: 该论文属于单目度量深度估计任务，解决不同相机类型间泛化能力不足的问题。提出UniDAC框架，通过解耦深度预测与尺度估计，实现跨相机的统一性能。**

- **链接: [https://arxiv.org/pdf/2603.27105](https://arxiv.org/pdf/2603.27105)**

> **作者:** Girish Chandar Ganesan; Yuliang Guo; Liu Ren; Xiaoming Liu
>
> **摘要:** Monocular metric depth estimation (MMDE) is a core challenge in computer vision, playing a pivotal role in real-world applications that demand accurate spatial understanding. Although prior works have shown promising zero-shot performance in MMDE, they often struggle with generalization across diverse camera types, such as fisheye and $360^\circ$ cameras. Recent advances have addressed this through unified camera representations or canonical representation spaces, but they require either including large-FoV camera data during training or separately trained models for different domains. We propose UniDAC, an MMDE framework that presents universal robustness in all domains and generalizes across diverse cameras using a single model. We achieve this by decoupling metric depth estimation into relative depth prediction and spatially varying scale estimation, enabling robust performance across different domains. We propose a lightweight Depth-Guided Scale Estimation module that upsamples a coarse scale map to high resolution using the relative depth map as guidance to account for local scale variations. Furthermore, we introduce RoPE-$\phi$, a distortion-aware positional embedding that respects the spatial warping in Equi-Rectangular Projections (ERP) via latitude-aware weighting. UniDAC achieves state of the art (SoTA) in cross-camera generalization by consistently outperforming prior methods across all datasets.
>
---
#### [new 199] SJD-VP: Speculative Jacobi Decoding with Verification Prediction for Autoregressive Image Generation
- **分类: cs.CV**

- **简介: 该论文属于自回归图像生成任务，旨在解决SJD方法中推测令牌接受率低的问题。通过分析令牌概率变化，提出SJD-VP提升接受率和生成质量。**

- **链接: [https://arxiv.org/pdf/2603.27115](https://arxiv.org/pdf/2603.27115)**

> **作者:** Bingqi Shan; Baoquan Zhang; Xiaochen Qi; Xutao Li; Yunming Ye; Liqiang Nie
>
> **摘要:** Speculative Jacobi Decoding (SJD) has emerged as a promising method for accelerating autoregressive image generation. Despite its potential, existing SJD approaches often suffer from the low acceptance rate issue of speculative tokens due to token selection ambiguity. Recent works attempt to mitigate this issue primarily from the relaxed token verification perspective but fail to fully exploit the iterative dynamics of decoding. In this paper, we conduct an in-depth analysis and make a novel observation that tokens whose probabilities increase are more likely to match the verification-accepted and correct token. Based on this, we propose a novel Speculative Jacobi Decoding with Verification Prediction (SJD-VP). The key idea is to leverage the change in token probabilities across iterations to guide sampling, favoring tokens whose probabilities increase. This effectively predicts which tokens are likely to pass subsequent verification, boosting the acceptance rate. In particular, our SJD-VP is plug-and-play and can be seamlessly integrated into existing SJD methods. Extensive experiments on standard benchmarks demonstrate that our SJD-VP method consistently accelerates autoregressive decoding while improving image generation quality.
>
---
#### [new 200] Diversity Matters: Dataset Diversification and Dual-Branch Network for Generalized AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像检测任务，旨在解决泛化性不足的问题。通过数据多样化和双分支网络提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.27800](https://arxiv.org/pdf/2603.27800)**

> **作者:** Nusrat Tasnim; Kutub Uddin; Khalid Malik
>
> **摘要:** The rapid proliferation of AI-generated images, powered by generative adversarial networks (GANs), diffusion models, and other synthesis techniques, has raised serious concerns about misinformation, copyright violations, and digital security. However, detecting such images in a generalized and robust manner remains a major challenge due to the vast diversity of generative models and data distributions. In this work, we present \textbf{Diversity Matters}, a novel framework that emphasizes data diversity and feature domain complementarity for AI-generated image detection. The proposed method introduces a feature-domain similarity filtering mechanism that discards redundant or highly similar samples across both inter-class and intra-class distributions, ensuring a more diverse and representative training set. Furthermore, we propose a dual-branch network that combines CLIP features from the pixel domain and the frequency domain to jointly capture semantic and structural cues, leading to improved generalization against unseen generative models and adversarial conditions. Extensive experiments on benchmark datasets demonstrate that the proposed approach significantly improves cross-model and cross-dataset performance compared to existing methods. \textbf{Diversity Matters} highlights the critical role of data and feature diversity in building reliable and robust detectors against the rapidly evolving landscape of synthetic content.
>
---
#### [new 201] A Provable Energy-Guided Test-Time Defense Boosting Adversarial Robustness of Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于对抗鲁棒性研究，旨在提升大视觉语言模型的可靠性。提出ET3方法，在不训练的情况下通过最小化输入能量增强模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.26984](https://arxiv.org/pdf/2603.26984)**

> **作者:** Mujtaba Hussain Mirza; Antonio D'Orazio; Odelia Melamed; Iacopo Masi
>
> **备注:** Accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026, Main Conference
>
> **摘要:** Despite the rapid progress in multimodal models and Large Visual-Language Models (LVLM), they remain highly susceptible to adversarial perturbations, raising serious concerns about their reliability in real-world use. While adversarial training has become the leading paradigm for building models that are robust to adversarial attacks, Test-Time Transformations (TTT) have emerged as a promising strategy to boost robustness at this http URL light of this, we propose Energy-Guided Test-Time Transformation (ET3), a lightweight, training-free defense that enhances the robustness by minimizing the energy of the input this http URL method is grounded in a theory that proves our transformation succeeds in classification under reasonable assumptions. We present extensive experiments demonstrating that ET3 provides a strong defense for classifiers, zero-shot classification with CLIP, and also for boosting the robustness of LVLMs in tasks such as Image Captioning and Visual Question Answering. Code is available at this http URL .
>
---
#### [new 202] STRIDE: When to Speak Meets Sequence Denoising for Streaming Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，解决在线流媒体中的何时响应问题。提出STRIDE方法，通过结构化时间建模和迭代去噪，提升响应的可靠性与时间一致性。**

- **链接: [https://arxiv.org/pdf/2603.27593](https://arxiv.org/pdf/2603.27593)**

> **作者:** Junho Kim; Hosu Lee; James M. Rehg; Minsu Kim; Yong Man Ro
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent progress in video large language models (Video-LLMs) has enabled strong offline reasoning over long and complex videos. However, real-world deployments increasingly require streaming perception and proactive interaction, where video frames arrive online and the system must decide not only what to respond, but also when to respond. In this work, we revisit proactive activation in streaming video as a structured sequence modeling problem, motivated by the observation that temporal transitions in streaming video naturally form span-structured activation patterns. To capture this span-level structure, we model activation signals jointly over a sliding temporal window and update them iteratively as new frames arrive. We propose STRIDE (Structured Temporal Refinement with Iterative DEnoising), which employs a lightweight masked diffusion module at the activation interface to jointly predict and progressively refine activation signals across the window. Extensive experiments on diverse streaming benchmarks and downstream models demonstrate that STRIDE shows more reliable and temporally coherent proactive responses, significantly improving when-to-speak decision quality in online streaming scenarios.
>
---
#### [new 203] AdaptToken: Entropy-based Adaptive Token Selection for MLLM Long Video Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态大模型的长视频理解任务，旨在解决高内存消耗和上下文长度限制问题。通过自适应选择关键帧/标记，提升模型准确率并减少推理时间。**

- **链接: [https://arxiv.org/pdf/2603.28696](https://arxiv.org/pdf/2603.28696)**

> **作者:** Haozhe Qi; Kevin Qu; Mahdi Rad; Rui Wang; Alexander Mathis; Marc Pollefeys
>
> **备注:** Project page: this https URL
>
> **摘要:** Long video understanding remains challenging for Multi-modal Large Language Models (MLLMs) due to high memory costs and context-length limits. Prior approaches mitigate this by scoring and selecting frames/tokens within short clips, but they lack a principled mechanism to (i) compare relevance across distant video clips and (ii) stop processing once sufficient evidence has been gathered. We propose AdaptToken, a training-free framework that turns an MLLM's self-uncertainty into a global control signal for long-video token selection. AdaptToken splits a video into groups, extracts cross-modal attention to rank tokens within each group, and uses the model's response entropy to estimate each group's prompt relevance. This entropy signal enables a global token budget allocation across groups and further supports early stopping (AdaptToken-Lite), skipping the remaining groups when the model becomes sufficiently certain. Across four long-video benchmarks (VideoMME, LongVideoBench, LVBench, and MLVU) and multiple base MLLMs (7B-72B), AdaptToken consistently improves accuracy (e.g., +6.7 on average over Qwen2.5-VL 7B) and continues to benefit from extremely long inputs (up to 10K frames), while AdaptToken-Lite reduces inference time by about half with comparable performance. Project page: this https URL
>
---
#### [new 204] Contour-Guided Query-Based Feature Fusion for Boundary-Aware and Generalizable Cardiac Ultrasound Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决心脏超声图像边界不清晰、结构不一致的问题。通过引入轮廓引导的特征融合方法，提升分割精度和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.28110](https://arxiv.org/pdf/2603.28110)**

> **作者:** Zahid Ullah; Sieun Choi; Jihie Kim
>
> **摘要:** Accurate cardiac ultrasound segmentation is essential for reliable assessment of ventricular function in intelligent healthcare systems. However, echocardiographic images are challenging due to low contrast, speckle noise, irregular boundaries, and domain shifts across devices and patient populations. Existing methods, largely based on appearance-driven learning, often fail to preserve boundary precision and structural consistency under these conditions. To address these issues, we propose a Contour-Guided Query Refinement Network (CGQR-Net) for boundary-aware cardiac ultrasound segmentation. The framework integrates multi-resolution feature representations with contour-derived structural priors. An HRNet backbone preserves high-resolution spatial details while capturing multi-scale context. A coarse segmentation is first generated, from which anatomical contours are extracted and encoded into learnable query embeddings. These contour-guided queries interact with fused feature maps via cross-attention, enabling structure-aware refinement that improves boundary delineation and reduces noise artifacts. A dual-head supervision strategy jointly optimizes segmentation and boundary prediction to enforce structural consistency. The proposed method is evaluated on the CAMUS dataset and further validated on the CardiacNet dataset to assess cross-dataset generalization. Experimental results demonstrate improved segmentation accuracy, enhanced boundary precision, and robust performance across varying imaging conditions. These results highlight the effectiveness of integrating contour-level structural information with feature-level representations for reliable cardiac ultrasound segmentation.
>
---
#### [new 205] Customized Visual Storytelling with Unified Multimodal LLMs
- **分类: cs.CV**

- **简介: 该论文属于多模态故事生成任务，旨在解决单一文本输入和缺乏多样化镜头控制的问题。工作包括提出VstoryGen框架，整合多模态信息并提升故事的一致性和电影感。**

- **链接: [https://arxiv.org/pdf/2603.27690](https://arxiv.org/pdf/2603.27690)**

> **作者:** Wei-Hua Li; Cheng Sun; Chu-Song Chen
>
> **备注:** Paper accepted to the CVPR 2026 Workshop on Generative AI for Storytelling (CVPRW)
>
> **摘要:** Multimodal story customization aims to generate coherent story flows conditioned on textual descriptions, reference identity images, and shot types. While recent progress in story generation has shown promising results, most approaches rely on text-only inputs. A few studies incorporate character identity cues (e.g., facial ID), but lack broader multimodal conditioning. In this work, we introduce VstoryGen, a multimodal framework that integrates descriptions with character and background references to enable customizable story generation. To enhance cinematic diversity, we introduce shot-type control via parameter-efficient prompt tuning on movie data, enabling the model to generate sequences that more faithfully reflect cinematic grammar. To evaluate our framework, we establish two new benchmarks that assess multimodal story customization from the perspectives of character and scene consistency, text-visual alignment, and shot-type control. Experiments demonstrate that VstoryGen achieves improved consistency and cinematic diversity compared to existing methods.
>
---
#### [new 206] VistaGEN: Consistent Driving Video Generation with Fine-Grained Control Using Multiview Visual-Language Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决驾驶视频中细粒度对象控制与时空一致性问题。提出VistaGEN模型，结合多视角视觉语言推理，提升生成视频的可控性与连贯性。**

- **链接: [https://arxiv.org/pdf/2603.28353](https://arxiv.org/pdf/2603.28353)**

> **作者:** Li-Heng Chen; Ke Cheng; Yahui Liu; Lei Shi; Shi-Sheng Huang; Hongbo Fu
>
> **摘要:** Driving video generation has achieved much progress in controllability, video resolution, and length, but fails to support fine-grained object-level controllability for diverse driving videos, while preserving the spatiotemporal consistency, especially in long video generation. In this paper, we present a new driving video generation technique, called VistaGEN, which enables fine-grained control of specific entities, including 3D objects, images, and text descriptions, while maintaining spatiotemporal consistency in long video sequences. Our key innovation is the incorporation of multiview visual-language reasoning into the long driving video generation. To this end, we inject visual-language features into a multiview video generator to enable fine-grained controllability. More importantly, we propose a multiview vision-language evaluator (MV-VLM) to intelligently and automatically evaluate spatiotemporal consistency of the generated content, thus formulating a novel generation-evaluation-regeneration closed-loop generation mechanism. This mechanism ensures high-quality, coherent outputs, facilitating the creation of complex and reliable driving scenarios. Besides, within the closed-loop generation, we introduce an object-level refinement module to refine the unsatisfied results evaluated from the MV-VLM and then feed them back to the video generator for regeneration. Extensive evaluation shows that our VistaGEN achieves diverse driving video generation results with fine-grained controllability, especially for long-tail objects, and much better spatiotemporal consistency than previous approaches.
>
---
#### [new 207] Learning to Focus and Precise Cropping: A Reinforcement Learning Framework with Information Gaps and Grounding Loss for MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态视觉问答任务，旨在解决MLLMs对裁剪区域关注不足的问题。通过两阶段强化学习框架提升模型对细节的感知与推理能力。**

- **链接: [https://arxiv.org/pdf/2603.27494](https://arxiv.org/pdf/2603.27494)**

> **作者:** Xuanpu Zhao; Zhentao Tan; Dianmo Sheng; Tianxiang Chen; Yao Liu; Yue Wu; Tao Gong; Qi Chu; Nenghai Yu
>
> **摘要:** To enhance the perception and reasoning capabilities of multimodal large language models in complex visual scenes, recent research has introduced agent-based workflows. In these works, MLLMs autonomously utilize image cropping tool to analyze regions of interest for question answering. While existing training strategies, such as those employing supervised fine-tuning and reinforcement learning, have made significant progress, our empirical analysis reveals a key limitation. We demonstrate the model's strong reliance on global input and its weak dependence on the details within the cropped region. To address this issue, we propose a novel two-stage reinforcement learning framework that does not require trajectory supervision. In the first stage, we introduce the ``Information Gap" mechanism by adjusting the granularity of the global image. This mechanism trains the model to answer questions by focusing on cropped key regions, driven by the information gain these regions provide. The second stage further enhances cropping precision by incorporating a grounding loss, using a small number of bounding box annotations. Experiments show that our method significantly enhances the model's attention to cropped regions, enabling it to achieve state-of-the-art performance on high-resolution visual question-answering benchmarks. Our method provides a more efficient approach for perceiving and reasoning fine-grained details in MLLMs. Code is available at: this https URL.
>
---
#### [new 208] An Intelligent Framework for Real-Time Yoga Pose Detection and Posture Correction
- **分类: cs.CV**

- **简介: 该论文属于实时瑜伽姿势检测与纠正任务，解决自训中姿势错误导致效果降低和受伤风险的问题。通过融合轻量模型与深度学习，实现姿势评估与实时反馈。**

- **链接: [https://arxiv.org/pdf/2603.26760](https://arxiv.org/pdf/2603.26760)**

> **作者:** Chandramouli Haldar
>
> **摘要:** Yoga is widely recognized for improving physical fitness, flexibility, and mental well being. However, these benefits depend strongly on correct posture execution. Improper alignment during yoga practice can reduce effectiveness and increase the risk of musculoskeletal injuries, especially in self guided or online training environments. This paper presents a hybrid Edge AI based framework for real time yoga pose detection and posture correction. The proposed system integrates lightweight human pose estimation models with biomechanical feature extraction and a CNN LSTM based temporal learning architecture to recognize yoga poses and analyze motion dynamics. Joint angles and skeletal features are computed from detected keypoints and compared with reference pose configurations to evaluate posture correctness. A quantitative scoring mechanism is introduced to measure alignment deviations and generate real time corrective feedback through visual, text based, and voice based guidance. In addition, Edge AI optimization techniques such as model quantization and pruning are applied to enable low latency performance on resource constrained devices. The proposed framework provides an intelligent and scalable digital yoga assistant that can improve user safety and training effectiveness in modern fitness applications.
>
---
#### [new 209] Difference Feedback: Generating Multimodal Process-Level Supervision for VLM Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型的强化学习任务，旨在解决多步骤推理中奖励稀疏导致的对齐问题。提出差分反馈方法，自动构建步骤级监督，提升视觉与推理过程的对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.27482](https://arxiv.org/pdf/2603.27482)**

> **作者:** Feiding; Yongkang Zhang; Yuhao Liao; Zijian Zeng; Chunzheng Zhu; Yaozong Zheng; Yafei Liu; Yeling Peng; Youwei Wang; Sibo Wang; Huiming Yang; Linglin Liao; Shunzhi Yang
>
> **摘要:** Vision--language models (VLMs) are increasingly aligned via Group Relative Policy Optimization (GRPO)-style training. However, relying solely on terminal outcome rewards yields sparse credit assignment in multi-step reasoning, weakening the linkage between visual evidence and intermediate steps and often causing unstable optimization and visual hallucinations. We propose Differential Feedback, which automatically constructs token/step-level supervision masks by repairing erroneous reasoning trajectories, explicitly marking the key positions that require correction. Without costly large-scale step-by-step human annotations, our method enables process-level visual alignment and can be seamlessly integrated into existing GRPO-like frameworks. Experiments on multimodal reasoning benchmarks including MMMStar and MathVista show an average 3% improvement under matched compute budgets. Our approach offers an effective, low-cost solution for accurate vision--reasoning process alignment.
>
---
#### [new 210] RealBirdID: Benchmarking Bird Species Identification in the Era of MLLMs
- **分类: cs.CV**

- **简介: 该论文属于细粒度鸟类识别任务，旨在解决模型在无法确定物种时盲目猜测的问题。提出RealBirdID基准，要求模型要么准确识别，要么给出合理理由拒绝回答。**

- **链接: [https://arxiv.org/pdf/2603.27033](https://arxiv.org/pdf/2603.27033)**

> **作者:** Logan Lawrence; Mustafa Chasmai; Rangel Daroya; Wuao Liu; Seoyun Jeong; Aaron Sun; Max Hamilton; Fabien Delattre; Oindrila Saha; Subhransu Maji; Grant Van Horn
>
> **备注:** Accepted to CVPR26. 23 pages, 23 figures, 5 tables
>
> **摘要:** Fine-grained bird species identification in the wild is frequently unanswerable from a single image: key cues may be non-visual (e.g. vocalization), or obscured due to occlusion, camera angle, or low resolution. Yet today's multimodal systems are typically judged on answerable, in-schema cases, encouraging confident guesses rather than principled abstention. We propose the RealBirdID benchmark: given an image of a bird, a system should either answer with a species or abstain with a concrete, evidence-based rationale: "requires vocalization," "low quality image," or "view obstructed". For each genus, the dataset includes a validation split composed of curated unanswerable examples with labeled rationales, paired with a companion set of clearly answerable instances. We find that (1) the species identification on the answerable set is challenging for a variety of open-source and proprietary models (less than 13% accuracy for MLLMs including GPT-5 and Gemini-2.5 Pro), (2) models with greater classification ability are not necessarily more calibrated to abstain from unanswerable examples, and (3) that MLLMs generally fail at providing correct reasons even when they do abstain. RealBirdID establishes a focused target for abstention-aware fine-grained recognition and a recipe for measuring progress.
>
---
#### [new 211] Multi-view Graph Convolutional Network with Fully Leveraging Consistency via Granular-ball-based Topology Construction, Feature Enhancement and Interactive Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多视图学习任务，旨在解决GCN方法中一致性利用不足的问题。提出MGCN-FLC模型，通过拓扑构建、特征增强和交互融合三个模块，全面提升节点、特征和视图间的一致性。**

- **链接: [https://arxiv.org/pdf/2603.26729](https://arxiv.org/pdf/2603.26729)**

> **作者:** Chengjie Cui; Taihua Xua; Shuyin Xia; Qinghua Zhang; Yun Cui; Shiping Wang
>
> **摘要:** The effective utilization of consistency is crucial for multi-view learning. GCNs leverage node connections to propagate information across the graph, facilitating the exploitation of consistency in multi-view data. However, most existing GCN-based multi-view methods suffer from several limitations. First, current approaches predominantly rely on KNN for topology construction, where the artificial selection of the k value significantly constrains the effective exploitation of inter-node consistency. Second, the inter-feature consistency within individual views is often overlooked, which adversely affects the quality of the final embedding representations. Moreover, these methods fail to fully utilize inter-view consistency as the fusion of embedded representations from multiple views is often implemented after the intra-view graph convolutional operation. Collectively, these issues limit the model's capacity to fully capture inter-node, inter-feature and inter-view consistency. To address these issues, this paper proposes the multi-view graph convolutional network with fully leveraging consistency via GB-based topology construction, feature enhancement and interactive fusion (MGCN-FLC). MGCN-FLC can fully utilize three types of consistency via the following three modules to enhance learning ability:The topology construction module based on the granular ball algorithm, which clusters nodes into granular balls with high internal similarity to capture inter-node consistency;The feature enhancement module that improves feature representations by capturing inter-feature consistency;The interactive fusion module that enables each view to deeply interact with all other views, thereby obtaining more comprehensive inter-view consistency. Experimental results on nine datasets show that the proposed MGCN-FLC outperforms state-of-the-art semi-supervised node classification methods.
>
---
#### [new 212] Project Imaging-X: A Survey of 1000+ Open-Access Medical Imaging Datasets for Foundation Model Development
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像数据集调研任务，旨在解决医疗数据分散、规模小的问题。通过系统整理1000+数据集，提出融合方法并构建统一资源平台。**

- **链接: [https://arxiv.org/pdf/2603.27460](https://arxiv.org/pdf/2603.27460)**

> **作者:** Zhongying Deng; Cheng Tang; Ziyan Huang; Jiashi Lin; Ying Chen; Junzhi Ning; Chenglong Ma; Jiyao Liu; Wei Li; Yinghao Zhu; Shujian Gao; Yanyan Huang; Sibo Ju; Yanzhou Su; Pengcheng Chen; Wenhao Tang; Tianbin Li; Haoyu Wang; Yuanfeng Ji; Hui Sun; Shaobo Min; Liang Peng; Feilong Tang; Haochen Xue; Rulin Zhou; Chaoyang Zhang; Wenjie Li; Shaohao Rui; Weijie Ma; Xingyue Zhao; Yibin Wang; Kun Yuan; Zhaohui Lu; Shujun Wang; Jinjie Wei; Lihao Liu; Dingkang Yang; Lin Wang; Yulong Li; Haolin Yang; Yiqing Shen; Lequan Yu; Xiaowei Hu; Yun Gu; Yicheng Wu; Benyou Wang; Minghui Zhang; Angelica I. Aviles-Rivero; Qi Gao; Hongming Shan; Xiaoyu Ren; Fang Yan; Hongyu Zhou; Haodong Duan; Maosong Cao; Shanshan Wang; Bin Fu; Xiaomeng Li; Zhi Hou; Chunfeng Song; Lei Bai; Yuan Cheng; Yuandong Pu; Xiang Li; Wenhai Wang; Hao Chen; Jiaxin Zhuang; Songyang Zhang; Huiguang He; Mengzhang Li; Bohan Zhuang; Zhian Bai; Rongshan Yu; Liansheng Wang; Yukun Zhou; Xiaosong Wang; Xin Guo; Guanbin Li; Xiangru Lin; Dakai Jin; Mianxin Liu; Wenlong Zhang; Qi Qin; Conghui He; Yuqiang Li; Ye Luo; Nanqing Dong; Jie Xu; Wenqi Shao; Bo Zhang; Qiujuan Yan; Yihao Liu; Jun Ma; Zhi Lu; Yuewen Cao; Zongwei Zhou; Jianming Liang; Shixiang Tang; Qi Duan; Dongzhan Zhou
>
> **备注:** 157 pages, 19 figures, 26 tables. Project repo: \url{this https URL}
>
> **摘要:** Foundation models have demonstrated remarkable success across diverse domains and tasks, primarily due to the thrive of large-scale, diverse, and high-quality datasets. However, in the field of medical imaging, the curation and assembling of such medical datasets are highly challenging due to the reliance on clinical expertise and strict ethical and privacy constraints, resulting in a scarcity of large-scale unified medical datasets and hindering the development of powerful medical foundation models. In this work, we present the largest survey to date of medical image datasets, covering over 1,000 open-access datasets with a systematic catalog of their modalities, tasks, anatomies, annotations, limitations, and potential for integration. Our analysis exposes a landscape that is modest in scale, fragmented across narrowly scoped tasks, and unevenly distributed across organs and modalities, which in turn limits the utility of existing medical image datasets for developing versatile and robust medical foundation models. To turn fragmentation into scale, we propose a metadata-driven fusion paradigm (MDFP) that integrates public datasets with shared modalities or tasks, thereby transforming multiple small data silos into larger, more coherent resources. Building on MDFP, we release an interactive discovery portal that enables end-to-end, automated medical image dataset integration, and compile all surveyed datasets into a unified, structured table that clearly summarizes their key characteristics and provides reference links, offering the community an accessible and comprehensive repository. By charting the current terrain and offering a principled path to dataset consolidation, our survey provides a practical roadmap for scaling medical imaging corpora, supporting faster data discovery, more principled dataset creation, and more capable medical foundation models.
>
---
#### [new 213] Seeing the Scene Matters: Revealing Forgetting in Video Understanding Models with a Scene-Aware Long-Video Benchmark
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频上下文记忆不足的问题。通过构建SceneBench基准和Scene-RAG方法，验证了现有模型在场景级推理中的显著遗忘现象。**

- **链接: [https://arxiv.org/pdf/2603.27259](https://arxiv.org/pdf/2603.27259)**

> **作者:** Seng Nam Chen; Hao Chen; Chenglam Ho; Xinyu Mao; Jinping Wang; Yu Zhang; Chao Li
>
> **摘要:** Long video understanding (LVU) remains a core challenge in multimodal learning. Although recent vision-language models (VLMs) have made notable progress, existing benchmarks mainly focus on either fine-grained perception or coarse summarization, offering limited insight into temporal understanding over long contexts. In this work, we define a scene as a coherent segment of a video in which both visual and semantic contexts remain consistent, aligning with human perception. This leads us to a key question: can current VLMs reason effectively over long, scene-level contexts? To answer this, we introduce a new benchmark, SceneBench, designed to provide scene-level challenges. Our evaluation reveals a sharp drop in accuracy when VLMs attempt to answer scene-level questions, indicating significant forgetting of long-range context. To further validate these findings, we propose Scene Retrieval-Augmented Generation (Scene-RAG), which constructs a dynamic scene memory by retrieving and integrating relevant context across scenes. This Scene-RAG improves VLM performance by +2.50%, confirming that current models still struggle with long-context retention. We hope SceneBench will encourage future research toward VLMs with more robust, human-like video comprehension.
>
---
#### [new 214] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition
- **分类: cs.CV**

- **简介: 该论文属于人体识别任务，解决传统融合方法静态且低效的问题，提出FusionAgent框架，通过动态模型选择和信心感知融合提升识别效果与效率。**

- **链接: [https://arxiv.org/pdf/2603.26908](https://arxiv.org/pdf/2603.26908)**

> **作者:** Jie Zhu; Xiao Guo; Yiyang Su; Anil Jain; Xiaoming Liu
>
> **备注:** CVPR 2026
>
> **摘要:** Model fusion is a key strategy for robust recognition in unconstrained scenarios, as different models provide complementary strengths. This is especially important for whole-body human recognition, where biometric cues such as face, gait, and body shape vary across samples and are typically integrated via score-fusion. However, existing score-fusion strategies are usually static, invoking all models for every test sample regardless of sample quality or modality reliability. To overcome these limitations, we propose \textbf{FusionAgent}, a novel agentic framework that leverages a Multimodal Large Language Model (MLLM) to perform dynamic, sample-specific model selection. Each expert model is treated as a tool, and through Reinforcement Fine-Tuning (RFT) with a metric-based reward, the agent learns to adaptively determine the optimal model combination for each test input. To address the model score misalignment and embedding heterogeneity, we introduce Anchor-based Confidence Top-k (ACT) score-fusion, which anchors on the most confident model and integrates complementary predictions in a confidence-aware manner. Extensive experiments on multiple whole-body biometric benchmarks demonstrate that FusionAgent significantly outperforms SoTA methods while achieving higher efficiency through fewer model invocations, underscoring the critical role of dynamic, explainable, and robust model fusion in real-world recognition systems. Project page: \href{this https URL}{FusionAgent}.
>
---
#### [new 215] Robust Remote Sensing Image-Text Retrieval with Noisy Correspondence
- **分类: cs.CV**

- **简介: 该论文属于遥感图像-文本检索任务，解决噪声对应问题。提出RRSITR方法，通过自适应学习策略提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.28134](https://arxiv.org/pdf/2603.28134)**

> **作者:** Qiya Song; Yiqiang Xie; Yuan Sun; Renwei Dian; Xudong Kang
>
> **摘要:** As a pivotal task that bridges remote visual and linguistic understanding, Remote Sensing Image-Text Retrieval (RSITR) has attracted considerable research interest in recent years. However, almost all RSITR methods implicitly assume that image-text pairs are matched perfectly. In practice, acquiring a large set of well-aligned data pairs is often prohibitively expensive or even infeasible. In addition, we also notice that the remote sensing datasets (e.g., RSITMD) truly contain some inaccurate or mismatched image text descriptions. Based on the above observations, we reveal an important but untouched problem in RSITR, i.e., Noisy Correspondence (NC). To overcome these challenges, we propose a novel Robust Remote Sensing Image-Text Retrieval (RRSITR) paradigm that designs a self-paced learning strategy to mimic human cognitive learning patterns, thereby learning from easy to hard from multi-modal data with NC. Specifically, we first divide all training sample pairs into three categories based on the loss magnitude of each pair, i.e., clean sample pairs, ambiguous sample pairs, and noisy sample pairs. Then, we respectively estimate the reliability of each training pair by assigning a weight to each pair based on the values of the loss. Further, we respectively design a new multi-modal self-paced function to dynamically regulate the training sequence and weights of the samples, thus establishing a progressive learning process. Finally, for noisy sample pairs, we present a robust triplet loss to dynamically adjust the soft margin based on semantic similarity, thereby enhancing the robustness against noise. Extensive experiments on three popular benchmark datasets demonstrate that the proposed RRSITR significantly outperforms the state-of-the-art methods, especially in high noise rates. The code is available at: this https URL
>
---
#### [new 216] Live Interactive Training for Video Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频分割任务，旨在减少用户交互次数。提出LIT框架，使模型在推理时在线学习用户修正，提升后续帧的分割效果。**

- **链接: [https://arxiv.org/pdf/2603.26929](https://arxiv.org/pdf/2603.26929)**

> **作者:** Xinyu Yang; Haozheng Yu; Yihong Sun; Bharath Hariharan; Jennifer J. Sun
>
> **备注:** CVPR 2026
>
> **摘要:** Interactive video segmentation often requires many user interventions for robust performance in challenging scenarios (e.g., occlusions, object separations, camouflage, etc.). Yet, even state-of-the-art models like SAM2 use corrections only for immediate fixes without learning from this feedback, leading to inefficient, repetitive user effort. To address this, we introduce Live Interactive Training (LIT), a novel framework for prompt-based visual systems where models also learn online from human corrections at inference time. Our primary instantiation, LIT-LoRA, implements this by continually updating a lightweight LoRA module on-the-fly. When a user provides a correction, this module is rapidly trained on that feedback, allowing the vision system to improve performance on subsequent frames of the same video. Leveraging the core principles of LIT, our LIT-LoRA implementation achieves an average 18-34% reduction in total corrections on challenging video segmentation benchmarks, with a negligible training overhead of ~0.5s per correction. We further demonstrate its generality by successfully adapting it to other segmentation models and extending it to CLIP-based fine-grained image classification. Our work highlights the promise of live adaptation to transform interactive tools and significantly reduce redundant human effort in complex visual tasks. Project: this https URL.
>
---
#### [new 217] MotionRFT: Unified Reinforcement Fine-Tuning for Text-to-Motion Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到动作生成任务，旨在解决模型与高层次目标对齐的问题。提出MotionRFT框架，通过统一语义表示和高效微调方法提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.27185](https://arxiv.org/pdf/2603.27185)**

> **作者:** Xiaofeng Tan; Wanjiang Weng; Hongsong Wang; Fang Zhao; Xin Geng; Liang Wang
>
> **摘要:** Text-to-motion generation has advanced with diffusion- and flow-based generative models, yet supervised pretraining remains insufficient to align models with high-level objectives such as semantic consistency, realism, and human preference. Existing post-training methods have key limitations: they (1) target a specific motion representation, such as joints, (2) optimize a particular aspect, such as text-motion alignment, and may compromise other factors; and (3) incur substantial computational overhead, data dependence, and coarse-grained optimization. We present a reinforcement fine-tuning framework that comprises a heterogeneous-representation, multi-dimensional reward model, MotionReward, and an efficient, fine-grained fine-tuning method, EasyTune. To obtain a unified semantics representation, MotionReward maps heterogeneous motions into a shared semantic space anchored by text, enabling multidimensional reward learning; Self-refinement Preference Learning further enhances semantics without additional annotations. For efficient and effective fine-tuning, we identify the recursive gradient dependence across denoising steps as the key bottleneck, and propose EasyTune, which optimizes step-wise rather than over the full trajectory, yielding dense, fine-grained, and memory-efficient updates. Extensive experiments validate the effectiveness of our framework, achieving FID 0.132 at 22.10 GB peak memory for MLD model and saving up to 15.22 GB over DRaFT. It reduces FID by 22.9% on joint-based ACMDM, and achieves a 12.6% R-Precision gain and 23.3% FID improvement on rotation-based HY Motion. Our project page with code is publicly available.
>
---
#### [new 218] Steering Sparse Autoencoder Latents to Control Dynamic Head Pruning in Vision Transformers (Student Abstract)
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉Transformer的动态头剪枝任务，旨在解决剪枝策略难以解释和控制的问题。通过集成稀疏自编码器，利用可解释的稀疏潜在变量调控剪枝决策，实现高效且可控的剪枝。**

- **链接: [https://arxiv.org/pdf/2603.26743](https://arxiv.org/pdf/2603.26743)**

> **作者:** Yousung Lee; Dongsoo Har
>
> **备注:** 3 pages, 5 figures. Accepted as AAAI 2026 Student Abstract. Includes additional appendix with extended analysis
>
> **摘要:** Dynamic head pruning in Vision Transformers (ViTs) improves efficiency by removing redundant attention heads, but existing pruning policies are often difficult to interpret and control. In this work, we propose a novel framework by integrating Sparse Autoencoders (SAEs) with dynamic pruning, leveraging their ability to disentangle dense embeddings into interpretable and controllable sparse latents. Specifically, we train an SAE on the final-layer residual embedding of the ViT and amplify the sparse latents with different strategies to alter pruning decisions. Among them, per-class steering reveals compact, class-specific head subsets that preserve accuracy. For example, bowl improves accuracy (76% to 82%) while reducing head usage (0.72 to 0.33) via heads h2 and h5. These results show that sparse latent features enable class-specific control of dynamic pruning, effectively bridging pruning efficiency and mechanistic interpretability in ViTs.
>
---
#### [new 219] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis
- **分类: cs.CV**

- **简介: 该论文属于HDR新视角合成任务，解决环境光照依赖性外观建模问题。通过引入物理启发的框架，结合图像曝光与高斯光照分支，提升HDR细节重建效果。**

- **链接: [https://arxiv.org/pdf/2603.28020](https://arxiv.org/pdf/2603.28020)**

> **作者:** Huimin Zeng; Yue Bai; Hailing Wang; Yun Fu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** High dynamic range novel view synthesis (HDR-NVS) reconstructs scenes with dynamic details by fusing multi-exposure low dynamic range (LDR) views, yet it struggles to capture ambient illumination-dependent appearance. Implicitly supervising HDR content by constraining tone-mapped results fails in correcting abnormal HDR values, and results in limited gradients for Gaussians in under/over-exposed regions. To this end, we introduce PhysHDR-GS, a physically inspired HDR-NVS framework that models scene appearance via intrinsic reflectance and adjustable ambient illumination. PhysHDR-GS employs a complementary image-exposure (IE) branch and Gaussian-illumination (GI) branch to faithfully reproduce standard camera observations and capture illumination-dependent appearance changes, respectively. During training, the proposed cross-branch HDR consistency loss provides explicit supervision for HDR content, while an illumination-guided gradient scaling strategy mitigates exposure-biased gradient starvation and reduces under-densified representations. Experimental results across realistic and synthetic datasets demonstrate our superiority in reconstructing HDR details (e.g., a PSNR gain of 2.04 dB over HDR-GS), while maintaining real-time rendering speed (up to 76 FPS). Code and models are available at this https URL.
>
---
#### [new 220] PRUE: A Practical Recipe for Field Boundary Segmentation at Scale
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于农业遥感中的田地边界分割任务，旨在解决现有方法对光照、尺度和地理位置敏感的问题。通过改进U-Net模型，提升分割性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27101](https://arxiv.org/pdf/2603.27101)**

> **作者:** Gedeon Muhawenayo; Caleb Robinson; Subash Khanal; Zhanpei Fang; Isaac Corley; Alexander Wollam; Tianyi Gao; Leonard Strnad; Ryan Avery; Lyndon Estes; Ana M. Tárano; Nathan Jacobs; Hannah Kerner
>
> **备注:** 12 pages, 3 figures, supplementary material. Accepted at CVPR 2026 (IEEE/CVF Conference on Computer Vision and Pattern Recognition)
>
> **摘要:** Large-scale maps of field boundaries are essential for agricultural monitoring tasks. Existing deep learning approaches for satellite-based field mapping are sensitive to illumination, spatial scale, and changes in geographic location. We conduct the first systematic evaluation of segmentation and geospatial foundation models (GFMs) for global field boundary delineation using the Fields of The World (FTW) benchmark. We evaluate 18 models under unified experimental settings, showing that a U-Net semantic segmentation model outperforms instance-based and GFM alternatives on a suite of performance and deployment metrics. We propose a new segmentation approach that combines a U-Net backbone, composite loss functions, and targeted data augmentations to enhance performance and robustness under real-world conditions. Our model achieves a 76\% IoU and 47\% object-F1 on FTW, an increase of 6\% and 9\% over the previous baseline. Our approach provides a practical framework for reliable, scalable, and reproducible field boundary delineation across model design, training, and inference. We release all models and model-derived field boundary datasets for five countries.
>
---
#### [new 221] SFDemorpher: Generalizable Face Demorphing for Operational Morphing Attack Detection
- **分类: cs.CV**

- **简介: 该论文属于人脸防活体攻击任务，解决morphing攻击检测问题。提出SFDemorpher框架，通过联合StyleGAN和高维特征空间进行身份解耦，提升检测泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.28322](https://arxiv.org/pdf/2603.28322)**

> **作者:** Raul Ismayilov; Luuk Spreeuwers
>
> **摘要:** Face morphing attacks compromise biometric security by creating document images that verify against multiple identities, posing significant risks from document issuance to border control. Differential Morphing Attack Detection (D-MAD) offers an effective countermeasure, particularly when employing face demorphing to disentangle identities blended in the morph. However, existing methods lack operational generalizability due to limited training data and the assumption that all document inputs are morphs. This paper presents SFDemorpher, a framework designed for the operational deployment of face demorphing for D-MAD that performs identity disentanglement within joint StyleGAN latent and high-dimensional feature spaces. We introduce a dual-pass training strategy handling both morphed and bona fide documents, leveraging a hybrid corpus with predominantly synthetic identities to enhance robustness against unseen distributions. Extensive evaluation confirms state-of-the-art generalizability across unseen identities, diverse capture conditions, and 13 morphing techniques, spanning both border verification and the challenging document enrollment stage. Our framework achieves superior D-MAD performance by widening the margin between the score distributions of bona fide and morphed samples while providing high-fidelity visual reconstructions facilitating explainability.
>
---
#### [new 222] Tracking without Seeing: Geospatial Inference using Encrypted Traffic from Distributed Nodes
- **分类: cs.CV; cs.LG; cs.NI**

- **简介: 该论文属于目标跟踪任务，解决在无原始数据情况下通过加密流量进行地理空间推断的问题。提出GraySense框架，利用包大小信息实现物体跟踪。**

- **链接: [https://arxiv.org/pdf/2603.27811](https://arxiv.org/pdf/2603.27811)**

> **作者:** Sadik Yagiz Yetim; Gaofeng Dong; Isaac-Neil Zanoria; Ronit Barman; Maggie Wigness; Tarek Abdelzaher; Mani Srivastava; Suhas Diggavi
>
> **摘要:** Accurate observation of dynamic environments traditionally relies on synthesizing raw, signal-level information from multiple distributed sensors. This work investigates an alternative approach: performing geospatial inference using only encrypted packet-level information, without access to the raw sensory data. We further explore how this indirect information can be fused with directly available sensory data to extend overall inference capabilities. We introduce GraySense, a learning-based framework that performs geospatial object tracking by analyzing encrypted wireless video transmission traffic, such as packet sizes, from cameras with inaccessible streams. GraySense leverages the inherent relationship between scene dynamics and transmitted packet sizes to infer object motion. The framework consists of two stages: (1) a Packet Grouping module that identifies frame boundaries and estimates frame sizes from encrypted network traffic, and (2) a Tracker module, based on a Transformer encoder with a recurrent state, which fuses indirect packet-based inputs with optional direct camera-based inputs to estimate the object's position. Extensive experiments with realistic videos from the CARLA simulator and emulated networks under varying conditions show that GraySense achieves 2.33 meters tracking error (Euclidean distance) without raw signal access, within the dimensions of tracked objects (4.61m x 1.93m). To our knowledge, this capability has not been previously demonstrated, expanding the use of latent signals for sensing.
>
---
#### [new 223] Annotation-Free Detection of Drivable Areas and Curbs Leveraging LiDAR Point Cloud Maps
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的感知任务，旨在解决无监督训练数据生成问题。通过LiDAR地图和检测结合，自动标注可行驶区域和路缘，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.27553](https://arxiv.org/pdf/2603.27553)**

> **作者:** Fulong Ma; Daojie Peng; Jun Ma
>
> **摘要:** Drivable areas and curbs are critical traffic elements for autonomous driving, forming essential components of the vehicle visual perception system and ensuring driving safety. Deep neural networks (DNNs) have significantly improved perception performance for drivable area and curb detection, but most DNN-based methods rely on large manually labeled datasets, which are costly, time-consuming, and expert-dependent, limiting their real-world application. Thus, we developed an automated training data generation module. Our previous work generated training labels using single-frame LiDAR and RGB data, suffering from occlusion and distant point cloud sparsity. In this paper, we propose a novel map-based automatic data labeler (MADL) module, combining LiDAR mapping/localization with curb detection to automatically generate training data for both tasks. MADL avoids occlusion and point cloud sparsity issues via LiDAR mapping, creating accurate large-scale datasets for DNN training. In addition, we construct a data review agent to filter the data generated by the MADL module, eliminating low-quality samples. Experiments on the KITTI, KITTI-CARLA and 3D-Curb datasets show that MADL achieves impressive performance compared to manual labeling, and outperforms traditional and state-of-the-art self-supervised methods in robustness and accuracy.
>
---
#### [new 224] Aesthetic Assessment of Chinese Handwritings Based on Vision Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于手写汉字美学评估任务，旨在解决传统评分反馈不足的问题。通过视觉语言模型生成多级反馈，提升学习者书写改进效果。**

- **链接: [https://arxiv.org/pdf/2603.26768](https://arxiv.org/pdf/2603.26768)**

> **作者:** Chen Zheng; Yuxuan Lai; Haoyang Lu; Wentao Ma; Jitao Yang; Jian Wang
>
> **备注:** Accepted by CCL2025
>
> **摘要:** The handwriting of Chinese characters is a fundamental aspect of learning the Chinese language. Previous automated assessment methods often framed scoring as a regression problem. However, this score-only feedback lacks actionable guidance, which limits its effectiveness in helping learners improve their handwriting skills. In this paper, we leverage vision-language models (VLMs) to analyze the quality of handwritten Chinese characters and generate multi-level feedback. Specifically, we investigate two feedback generation tasks: simple grade feedback (Task 1) and enriched, descriptive feedback (Task 2). We explore both low-rank adaptation (LoRA)-based fine-tuning strategies and in-context learning methods to integrate aesthetic assessment knowledge into VLMs. Experimental results show that our approach achieves state-of-the-art performances across multiple evaluation tracks in the CCL 2025 workshop on evaluation of handwritten Chinese character quality.
>
---
#### [new 225] MuSEAgent: A Multimodal Reasoning Agent with Stateful Experiences
- **分类: cs.CV**

- **简介: 该论文提出MuSEAgent，解决多模态推理任务中的决策问题。通过状态化经验学习，提升 agent 的多模态理解与动态检索能力。**

- **链接: [https://arxiv.org/pdf/2603.27813](https://arxiv.org/pdf/2603.27813)**

> **作者:** Shijian Wang; Jiarui Jin; Runhao Fu; Zexuan Yan; Xingjian Wang; Mengkang Hu; Eric Wang; Xiaoxi Li; Kangning Zhang; Li Yao; Wenxiang Jiao; Xuelian Cheng; Yuan Lu; Zongyuan Ge
>
> **摘要:** Research agents have recently achieved significant progress in information seeking and synthesis across heterogeneous textual and visual sources. In this paper, we introduce MuSEAgent, a multimodal reasoning agent that enhances decision-making by extending the capabilities of research agents to discover and leverage stateful experiences. Rather than relying on trajectory-level retrieval, we propose a stateful experience learning paradigm that abstracts interaction data into atomic decision experiences through hindsight reasoning. These experiences are organized into a quality-filtered experience bank that supports policy-driven experience retrieval at inference time. Specifically, MuSEAgent enables adaptive experience exploitation through complementary wide- and deep-search strategies, allowing the agent to dynamically retrieve multimodal guidance across diverse compositional semantic viewpoints. Extensive experiments demonstrate that MuSEAgent consistently outperforms strong trajectory-level experience retrieval baselines on both fine-grained visual perception and complex multimodal reasoning tasks. These results validate the effectiveness of stateful experience modeling in improving multimodal agent reasoning.
>
---
#### [new 226] BINO: Encoder Centric Self Supervised Stereo With Native Pair Input
- **分类: cs.CV**

- **简介: 该论文属于立体视觉任务，旨在提升特征对齐能力。通过输入级融合和位置编码，构建紧凑编码器，避免依赖外部模块，实现高效跨视图推理。**

- **链接: [https://arxiv.org/pdf/2603.27904](https://arxiv.org/pdf/2603.27904)**

> **作者:** Haokun Zhou
>
> **摘要:** Stereo needs features that preserve fine cross view correspondence rather than only semantic similarity. Recent self supervised vision models transfer well, but they are not built for this goal, and geometry focused methods often rely on a binocular decoder or another explicit linkage module during pretraining. BINO asks whether strong binocular structure can instead be learned inside a compact encoder. It does this by fusing the rectified pair at the input stage, forming stereo micro cell tokens, and using a row aware patch phase positional encoding. Training uses one view masked token only distillation together with occlusion and view specific appearance mismatch. In a strict low resource setting with pretraining only on KITTI object, BINO gives the best frozen descriptor results under a no linkage probe among all compared baselines on proxy dense stereo, hard negative retrieval, and KITTI Stereo~2012 disparity. With the same lightweight stereo head for every encoder, it stays near CroCo~v2 while using a much smaller encoder. Supplementary transfer experiments on KITTI Stereo~2015 show the same qualitative trend. These results suggest that much of the cross view reasoning often assigned to a separate linkage module can be learned inside a compact and reusable encoder.
>
---
#### [new 227] Bridging the Geometry Mismatch: Frequency-Aware Anisotropic Serialization for Thin-Structure SSMs
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，解决薄结构分割中几何不匹配问题。通过频率感知的非各向同性序列化方法，提升分割精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.28503](https://arxiv.org/pdf/2603.28503)**

> **作者:** Jin Bai; Huiyao Zhang; Qi Wen; Ningyang Li; Shengyang Li; Atta ur Rahman; Xiaolin Tian
>
> **摘要:** The segmentation of thin linear structures is inherently topology allowbreak-critical, where minor local errors can sever long-range connectivity. While recent State-Space Models (SSMs) offer efficient long-range modeling, their isotropic serialization (e.g., raster scanning) creates a geometry mismatch for anisotropic targets, causing state propagation across rather than along the structure trajectories. To address this, we propose FGOS-Net, a framework based on frequency allowbreak-geometric disentanglement. We first decompose features into a stable topology carrier and directional high-frequency bands, leveraging the latter to explicitly correct spatial misalignments induced by downsampling. Building on this calibrated topology, we introduce frequency-aligned scanning that elevates serialization to a geometry-conditioned decision, preserving direction-consistent traces. Coupled with an active probing strategy to selectively inject high-frequency details and suppress texture ambiguity, FGOS-Net consistently outperforms strong baselines across four challenging benchmarks. Notably, it achieves 91.3% mIoU and 97.1% clDice on DeepCrack while running at 80 FPS with only 7.87 GFLOPs.
>
---
#### [new 228] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于3D语义占用预测任务，旨在解决缺乏高质量3D数据和实例级标注的问题。作者构建了ADMesh和CarlaOcc数据集，提供高精度、实例级的3D occupancy标注，推动自动驾驶中的3D感知研究。**

- **链接: [https://arxiv.org/pdf/2603.27238](https://arxiv.org/pdf/2603.27238)**

> **作者:** Yi Feng; Junwu E; Zizhan Guo; Yu Ma; Hanli Wang; Rui Fan
>
> **备注:** Accepted to CVPR 2026. Code and dataset are available at this https URL
>
> **摘要:** Panoptic occupancy prediction aims to jointly infer voxel-wise semantics and instance identities within a unified 3D scene representation. Nevertheless, progress in this field remains constrained by the absence of high-quality 3D mesh resources, instance-level annotations, and physically consistent occupancy datasets. Existing benchmarks typically provide incomplete and low-resolution geometry without instance-level annotations, limiting the development of models capable of achieving precise geometric reconstruction, reliable occlusion reasoning, and holistic 3D understanding. To address these challenges, this paper presents an instance-centric benchmark for the 3D panoptic occupancy prediction task. Specifically, we introduce ADMesh, the first unified 3D mesh library tailored for autonomous driving, which integrates over 15K high-quality 3D models with diverse textures and rich semantic annotations. Building upon ADMesh, we further construct CarlaOcc, a large-scale, physically consistent panoptic occupancy dataset generated using the CARLA simulator. This dataset contains over 100K frames with fine-grained, instance-level occupancy ground truth at voxel resolutions as fine as 0.05 m. Furthermore, standardized evaluation metrics are introduced to quantify the quality of existing occupancy datasets. Finally, a systematic benchmark of representative models is established on the proposed dataset, which provides a unified platform for fair comparison and reproducible research in the field of 3D panoptic perception. Code and dataset are available at this https URL.
>
---
#### [new 229] Divide and Restore: A Modular Task-Decoupled Framework for Universal Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，解决多类型退化修复问题。提出一种模块化框架，通过任务解耦和动态路由，提高效率与扩展性。**

- **链接: [https://arxiv.org/pdf/2603.28658](https://arxiv.org/pdf/2603.28658)**

> **作者:** Joanna Wiekiera; Martyna Zur
>
> **摘要:** Restoring images affected by various types of degradation, such as noise, blur, or improper exposure, remains a significant challenge in computer vision. While recent trends favor complex monolithic all-in-one architectures, these models often suffer from negative task interference and require extensive joint training cycles on high-end computing clusters. In this paper, we propose a modular, task-decoupled image restoration framework based on an explicit diagnostic routing mechanism. The architecture consists of a lightweight Convolutional Neural Network (CNN) classifier that evaluates the input image and dynamically directs it to a specialized restoration node. A key advantage of this framework is its model-agnostic extensibility: while we demonstrate it using three independent U-Net experts, the system allows for the integration of any restoration method tailored to specific tasks. By isolating reconstruction paths, the framework prevents feature conflicts and significantly reduces training overhead. Unlike monolithic models, adding new degradation types in our framework only requires training a single expert and updating the router, rather than a full system retraining. Experimental results demonstrate that this computationally accessible approach offers a scalable and efficient solution for multi-degradation restoration on standard local hardware. The code will be published upon paper acceptance.
>
---
#### [new 230] Generating Synthetic Wildlife Health Data from Camera Trap Imagery: A Pipeline for Alopecia and Body Condition Training Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于野生动物健康检测任务，解决缺乏公开可用的机器学习数据集问题。通过生成合成图像，模拟皮毛脱落和体况下降，构建训练数据。**

- **链接: [https://arxiv.org/pdf/2603.26754](https://arxiv.org/pdf/2603.26754)**

> **作者:** David Brundage
>
> **摘要:** No publicly available, ML ready datasets exist for wildlife health conditions in camera trap imagery, creating a fundamental barrier to automated health screening. We present a pipeline for generating synthetic training images depicting alopecia and body condition deterioration in wildlife from real camera trap photographs. Our pipeline constructs a curated base image set from iWildCam using MegaDetector derived bounding boxes and center frame weighted stratified sampling across 8 North American species. A generative phenotype editing system produces controlled severity variants depicting hair loss consistent with mange and emaciation. An adaptive scene drift quality control system uses a sham prefilter and decoupled mask then score approach with complementary day or night metrics to reject images where the generative model altered the original scene. We frame the pipeline explicitly as a screening data source. From 201 base images across 4 species, we generate 553 QC passing synthetic variants with an overall pass rate of 83 percent. A sim to real transfer experiment training exclusively on synthetic data and testing on real camera trap images of suspected health conditions achieves 0.85 AUROC, demonstrating that the synthetic data captures visual features sufficient for screening.
>
---
#### [new 231] Edge Reliability Gap in Vision-Language Models: Quantifying Failure Modes of Compressed VLMs Under Visual Corruption
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究压缩视觉语言模型在视觉退化下的失效模式，属于模型可靠性分析任务。通过对比不同参数量模型，识别其失败特征，解决压缩模型安全性评估问题。**

- **链接: [https://arxiv.org/pdf/2603.26769](https://arxiv.org/pdf/2603.26769)**

> **作者:** Mehmet Kaan Erol
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** The rapid compression of large vision-language models (VLMs) for edge deployment raises an underexplored question: do compact models fail differently, not merely more often? This study compares a 7-billion-parameter quantised VLM (Qwen2.5-VL-7B, 4-bit NF4) against a 500-million-parameter FP16 model (SmolVLM2-500M) across 4,000 samples from VQAv2 and COCO Captions. A three-category error taxonomy (Object Blindness, Semantic Drift, Prior Bias) is applied as a diagnostic framework. A text-only GPT-4o judge reveals Semantic Drift (B) as the dominant failure mode on VQAv2 and on COCO for Qwen, with a mixed Object Blindness / Semantic Drift profile for SmolVLM2 on COCO; Prior Bias (C) is present on VQAv2 but absent on COCO for both models. Confidence calibration is measured via Expected Calibration Error (ECE) using geometric mean token probability, compositional reasoning is probed with structured negation probes across four templates, and a blur robustness experiment completes the evaluation. For this model pair, the compact model exhibits a qualitatively distinct failure signature: a 12.5pp larger negation collapse (-33.2pp vs. -20.8pp, Wald 95% CI [8.2, 16.8]pp, p < 10^-8), driven almost entirely by COCO while the VQAv2 gap is not statistically significant (4.5pp, p=0.19). The most discriminating template is false_yn: SMOLVLM2-500M responds "Yes" (incorrectly claiming a depicted object is absent) on 100% of COCO trials vs. 14% for Q WEN 2.5-VL-7B. Asymmetric dataset-dependent miscalibration and a blur experiment with two controlled ablations complete the analysis. The fully reproducible pipeline is released for systematic safety auditing of compressed VLMs prior to edge deployment.
>
---
#### [new 232] Hydra: Unifying Document Retrieval and Generation in a Single Vision-Language Model
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文提出Hydra模型，解决文档检索与生成需分开模型的问题。通过单视觉语言模型实现两者功能，降低系统复杂度并提升效率。**

- **链接: [https://arxiv.org/pdf/2603.28554](https://arxiv.org/pdf/2603.28554)**

> **作者:** Athos Georgiou
>
> **备注:** Comments: 17 pages, 2 figures, 7 tables. ## Model Cards - this https URL - this https URL - this https URL - this https URL ## Scripts & evals - this https URL
>
> **摘要:** Visual document understanding typically requires separate retrieval and generation models, doubling memory and system complexity. We present Hydra, a dual-head approach that provides both ColBERT-style late-interaction retrieval and autoregressive generation from a single vision-language model (VLM). A single LoRA adapter, trained only for retrieval, is toggled at inference: enabling it produces multi-vector embeddings; disabling it recovers the base model's generation quality -- byte-identical outputs in 100% of 10,500 greedy and stochastic samples, with max delta-ANLS = 0.0044 across 15,301 samples on four VQA benchmarks (three informative; ChartQA is near-zero for both models under greedy decoding) when compared against an independent base-model pipeline. We identify three engineering requirements (attention-mode restoration, lm_head preservation, KV-cache-aware decoding) whose omission silently breaks generation despite correct weight recovery. On ViDoRe V1, Hydra (4B) is within 1 percentage point of a controlled single-head baseline in a single training run, with higher aggregate scores on V2 and V3 that are concentrated on a subset of tasks; multi-seed experiments are needed to confirm these trends. The single-model design reduces peak GPU memory by 41%, though adapter switching introduces throughput overhead under concurrent serving loads. An ablation shows that GritLM-style joint training provides no benefit within the LoRA-based (r=16) training regime. A proof-of-concept extension to Qwen2.5-Omni-3B demonstrates that the mechanism generalizes to audio retrieval and video embedding, with speech generation.
>
---
#### [new 233] Survey on Remote Sensing Scene Classification: From Traditional Methods to Large Generative AI Models
- **分类: cs.CV**

- **简介: 本文综述了遥感场景分类任务，从传统方法到生成式AI模型的演进，探讨了技术发展、挑战与未来方向。**

- **链接: [https://arxiv.org/pdf/2603.26751](https://arxiv.org/pdf/2603.26751)**

> **作者:** Qionghao Huang; Can Hu
>
> **备注:** Accepted in Journal of King Saud University Computer and Information Sciences
>
> **摘要:** Remote sensing scene classification has experienced a paradigmatic transformation from traditional handcrafted feature methods to sophisticated artificial intelligence systems that now form the backbone of modern Earth observation applications. This comprehensive survey examines the complete methodological evolution, systematically tracing development from classical texture descriptors and machine learning classifiers through the deep learning revolution to current state-of-the-art foundation models and generative AI approaches. We chronicle the pivotal shift from manual feature engineering to automated hierarchical representation learning via convolutional neural networks, followed by advanced architectures including Vision Transformers, graph neural networks, and hybrid frameworks. The survey provides in-depth coverage of breakthrough developments in self-supervised foundation models and vision-language systems, highlighting exceptional performance in zero-shot and few-shot learning scenarios. Special emphasis is placed on generative AI innovations that tackle persistent challenges through synthetic data generation and advanced feature learning strategies. We analyze contemporary obstacles including annotation costs, multimodal data fusion complexities, interpretability demands, and ethical considerations, alongside current trends in edge computing deployment, federated learning frameworks, and sustainable AI practices. Based on comprehensive analysis of recent advances and gaps, we identify key future research priorities: advancing hyperspectral and multi-temporal analysis capabilities, developing robust cross-domain generalization methods, and establishing standardized evaluation protocols to accelerate scientific progress in remote sensing scene classification systems.
>
---
#### [new 234] From Diffusion To Flow: Efficient Motion Generation In MotionGPT3
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文本到动作生成任务，研究比较扩散与修正流目标在MotionGPT3中的效果，验证修正流在效率和性能上的优势。**

- **链接: [https://arxiv.org/pdf/2603.26747](https://arxiv.org/pdf/2603.26747)**

> **作者:** Jaymin Ban; JiHong Jeon; SangYeop Jeong
>
> **备注:** ReALM-GEN Workshop ICLR 2026
>
> **摘要:** Recent text-driven motion generation methods span both discrete token-based approaches and continuous-latent formulations. MotionGPT3 exemplifies the latter paradigm, combining a learned continuous motion latent space with a diffusion-based prior for text-conditioned synthesis. While rectified flow objectives have recently demonstrated favorable convergence and inference-time properties relative to diffusion in image and audio generation, it remains unclear whether these advantages transfer cleanly to the motion generation setting. In this work, we conduct a controlled empirical study comparing diffusion and rectified flow objectives within the MotionGPT3 framework. By holding the model architecture, training protocol, and evaluation setup fixed, we isolate the effect of the generative objective on training dynamics, final performance, and inference efficiency. Experiments on the HumanML3D dataset show that rectified flow converges in fewer training epochs, reaches strong test performance earlier, and matches or exceeds diffusion-based motion quality under identical conditions. Moreover, flow-based priors exhibit stable behavior across a wide range of inference step counts and achieve competitive quality with fewer sampling steps, yielding improved efficiency--quality trade-offs. Overall, our results suggest that several known benefits of rectified flow objectives do extend to continuous-latent text-to-motion generation, highlighting the importance of the training objective choice in motion priors.
>
---
#### [new 235] SleepVLM: Explainable and Rule-Grounded Sleep Staging via a Vision-Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于睡眠分期任务，旨在解决自动化睡眠分期缺乏可审计解释的问题。提出SleepVLM模型，结合规则与视觉语言模型，生成符合AASM标准的可读理由，提升临床可信度。**

- **链接: [https://arxiv.org/pdf/2603.26738](https://arxiv.org/pdf/2603.26738)**

> **作者:** Guifeng Deng; Pan Wang; Jiquan Wang; Shuying Rao; Junyi Xie; Wanjun Guo; Tao Li; Haiteng Jiang
>
> **备注:** Under review
>
> **摘要:** While automated sleep staging has achieved expert-level accuracy, its clinical adoption is hindered by a lack of auditable reasoning. We introduce SleepVLM, a rule-grounded vision-language model (VLM) designed to stage sleep from multi-channel polysomnography (PSG) waveform images while generating clinician-readable rationales based on American Academy of Sleep Medicine (AASM) scoring criteria. Utilizing waveform-perceptual pre-training and rule-grounded supervised fine-tuning, SleepVLM achieved Cohen's kappa scores of 0.767 on an held out test set (MASS-SS1) and 0.743 on an external cohort (ZUAMHCS), matching state-of-the-art performance. Expert evaluations further validated the quality of the model's reasoning, with mean scores exceeding 4.0/5.0 for factual accuracy, evidence comprehensiveness, and logical coherence. By coupling competitive performance with transparent, rule-based explanations, SleepVLM may improve the trustworthiness and auditability of automated sleep staging in clinical workflows. To facilitate further research in interpretable sleep medicine, we release MASS-EX, a novel expert-annotated dataset.
>
---
#### [new 236] Gen-Searcher: Reinforcing Agentic Search for Image Generation
- **分类: cs.CV**

- **简介: 该论文提出Gen-Searcher，解决图像生成中知识不足的问题，通过搜索增强的代理进行多跳推理和知识收集，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.28767](https://arxiv.org/pdf/2603.28767)**

> **作者:** Kaituo Feng; Manyuan Zhang; Shuang Chen; Yunlong Lin; Kaixuan Fan; Yilei Jiang; Hongyu Li; Dian Zheng; Chenyang Wang; Xiangyu Yue
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** Recent image generation models have shown strong capabilities in generating high-fidelity and photorealistic images. However, they are fundamentally constrained by frozen internal knowledge, thus often failing on real-world scenarios that are knowledge-intensive or require up-to-date information. In this paper, we present Gen-Searcher, as the first attempt to train a search-augmented image generation agent, which performs multi-hop reasoning and search to collect the textual knowledge and reference images needed for grounded generation. To achieve this, we construct a tailored data pipeline and curate two high-quality datasets, Gen-Searcher-SFT-10k and Gen-Searcher-RL-6k, containing diverse search-intensive prompts and corresponding ground-truth synthesis images. We further introduce KnowGen, a comprehensive benchmark that explicitly requires search-grounded external knowledge for image generation and evaluates models from multiple dimensions. Based on these resources, we train Gen-Searcher with SFT followed by agentic reinforcement learning with dual reward feedback, which combines text-based and image-based rewards to provide more stable and informative learning signals for GRPO training. Experiments show that Gen-Searcher brings substantial gains, improving Qwen-Image by around 16 points on KnowGen and 15 points on WISE. We hope this work can serve as an open foundation for search agents in image generation, and we fully open-source our data, models, and code.
>
---
#### [new 237] Inference-time Trajectory Optimization for Manga Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决预训练模型在漫画图像上表现不佳的问题。通过推理时轨迹优化，无需重新训练即可提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.27790](https://arxiv.org/pdf/2603.27790)**

> **作者:** Ryosuke Furuta
>
> **摘要:** We present an inference-time adaptation method that tailors a pretrained image editing model to each input manga image using only the input image itself. Despite recent progress in pretrained image editing, such models often underperform on manga because they are trained predominantly on natural-image data. Re-training or fine-tuning large-scale models on manga is, however, generally impractical due to both computational cost and copyright constraints. To address this issue, our method slightly corrects the generation trajectory at inference time so that the input image can be reconstructed more faithfully under an empty prompt. Experimental results show that our method consistently outperforms existing baselines while incurring only negligible computational overhead.
>
---
#### [new 238] AIBench: Evaluating Visual-Logical Consistency in Academic Illustration Generation
- **分类: cs.CV**

- **简介: 该论文属于学术插图生成评估任务，旨在解决视觉逻辑一致性评价难题。通过设计多级问题，利用VQA和VLM进行准确评估。**

- **链接: [https://arxiv.org/pdf/2603.28068](https://arxiv.org/pdf/2603.28068)**

> **作者:** Zhaohe Liao; Kaixun Jiang; Zhihang Liu; Yujie Wei; Junqiu Yu; Quanhao Li; Hong-Tao Yu; Pandeng Li; Yuzheng Wang; Zhen Xing; Shiwei Zhang; Chen-Wei Xie; Yun Zheng; Xihui Liu
>
> **摘要:** Although image generation has boosted various applications via its rapid evolution, whether the state-of-the-art models are able to produce ready-to-use academic illustrations for papers is still largely this http URL comparing or evaluating the illustration with VLM is native but requires oracle multi-modal understanding ability, which is unreliable for long and complex texts and illustrations. To address this, we propose AIBench, the first benchmark using VQA for evaluating logic correctness of the academic illustrations and VLMs for assessing aesthetics. In detail, we designed four levels of questions proposed from a logic diagram summarized from the method part of the paper, which query whether the generated illustration aligns with the paper on different scales. Our VQA-based approach raises more accurate and detailed evaluations on visual-logical consistency while relying less on the ability of the judger VLM. With our high-quality AIBench, we conduct extensive experiments and conclude that the performance gap between models on this task is significantly larger than general ones, reflecting their various complex reasoning and high-density generation ability. Further, the logic and aesthetics are hard to optimize simultaneously as in handcrafted illustrations. Additional experiments further state that test-time scaling on both abilities significantly boosts the performance on this task.
>
---
#### [new 239] MOOZY: A Patient-First Foundation Model for Computational Pathology
- **分类: cs.CV**

- **简介: 该论文提出MOOZY，一个以患者为中心的病理学基础模型，解决跨任务迁移和多切片关系建模问题，通过患者级预训练提升性能。**

- **链接: [https://arxiv.org/pdf/2603.27048](https://arxiv.org/pdf/2603.27048)**

> **作者:** Yousef Kotp; Vincent Quoc-Huy Trinh; Christopher Pal; Mahdi S. Hosseini
>
> **摘要:** Computational pathology needs whole-slide image (WSI) foundation models that transfer across diverse clinical tasks, yet current approaches remain largely slide-centric, often depend on private data and expensive paired-report supervision, and do not explicitly model relationships among multiple slides from the same patient. We present MOOZY, a patient-first pathology foundation model in which the patient case, not the individual slide, is the core unit of representation. MOOZY explicitly models dependencies across all slides from the same patient via a case transformer during pretraining, combining multi-stage open self-supervision with scaled low-cost task supervision. In Stage 1, we pretrain a vision-only slide encoder on 77,134 public slide feature grids using masked self-distillation. In Stage 2, we align these representations with clinical semantics using a case transformer and multi-task supervision over 333 tasks from 56 public datasets, including 205 classification and 128 survival tasks across four endpoints. Across eight held-out tasks with five-fold frozen-feature probe evaluation, MOOZY achieves best or tied-best performance on most metrics and improves macro averages over TITAN by +7.37%, +5.50%, and +7.83% and over PRISM by +8.83%, +10.70%, and +9.78% for weighted F1, weighted ROC-AUC, and balanced accuracy, respectively. MOOZY is also parameter efficient with 85.77M parameters, 14x smaller than GigaPath. These results demonstrate that open, reproducible patient-level pretraining yields transferable embeddings, providing a practical path toward scalable patient-first histopathology foundation models.
>
---
#### [new 240] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，解决扩散Transformer在边缘设备部署效率低的问题。通过结构优化，提出EdgeDiT，在保持性能的同时显著降低参数、FLOPs和延迟。**

- **链接: [https://arxiv.org/pdf/2603.28405](https://arxiv.org/pdf/2603.28405)**

> **作者:** Sravanth Kodavanti; Manjunath Arveti; Sowmya Vajrala; Srinivas Miriyala; Vikram N R
>
> **备注:** Accepted at the Mobile AI Workshop, CVPR 2026
>
> **摘要:** Diffusion Transformers (DiT) have established a new state-of-the-art in high-fidelity image synthesis; however, their massive computational complexity and memory requirements hinder local deployment on resource-constrained edge devices. In this paper, we introduce EdgeDiT, a family of hardware-efficient generative transformers specifically engineered for mobile Neural Processing Units (NPUs), such as the Qualcomm Hexagon and Apple Neural Engine (ANE). By leveraging a hardware-aware optimization framework, we systematically identify and prune structural redundancies within the DiT backbone that are particularly taxing for mobile data-flows. Our approach yields a series of lightweight models that achieve a 20-30% reduction in parameters, a 36-46% decrease in FLOPs, and a 1.65-fold reduction in on-device latency without sacrificing the scaling advantages or the expressive capacity of the original transformer architecture. Extensive benchmarking demonstrates that EdgeDiT offers a superior Pareto-optimal trade-off between Frechet Inception Distance (FID) and inference latency compared to both optimized mobile U-Nets and vanilla DiT variants. By enabling responsive, private, and offline generative AI directly on-device, EdgeDiT provides a scalable blueprint for transitioning large-scale foundation models from high-end GPUs to the palm of the user.
>
---
#### [new 241] MV-RoMa: From Pairwise Matching into Multi-View Track Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D视觉任务，解决多视图对应不一致问题。提出MV-RoMa模型，联合估计多视图密集对应，提升SfM的重建精度和稠密性。**

- **链接: [https://arxiv.org/pdf/2603.27542](https://arxiv.org/pdf/2603.27542)**

> **作者:** Jongmin Lee; Seungyeop Kang; Sungjoo Yoo
>
> **备注:** CVPR 2026 Accepted
>
> **摘要:** Establishing consistent correspondences across images is essential for 3D vision tasks such as structure-from-motion (SfM), yet most existing matchers operate in a pairwise manner, often producing fragmented and geometrically inconsistent tracks when their predictions are chained across views. We propose MV-RoMa, a multi-view dense matching model that jointly estimates dense correspondences from a source image to multiple co-visible targets. Specifically, we design an efficient model architecture which avoids high computational cost of full cross-attention for multi-view feature interaction: (i) multi-view encoder that leverages pair-wise matching results as a geometric prior, and (ii) multi-view matching refiner that refines correspondences using pixel-wise attention. Additionally, we propose a post-processing strategy that integrates our model's consistent multi-view correspondences as high-quality tracks for SfM. Across diverse and challenging benchmarks, MV-RoMa produces more reliable correspondences and substantially denser, more accurate 3D reconstructions than existing sparse and dense matching methods. Project page: this https URL.
>
---
#### [new 242] Prototype-Enhanced Multi-View Learning for Thyroid Nodule Ultrasound Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于甲状腺结节超声分类任务，旨在解决模型在不同设备和环境下泛化能力差的问题。通过多视角学习和原型增强方法提升模型的稳定性和准确性。**

- **链接: [https://arxiv.org/pdf/2603.28315](https://arxiv.org/pdf/2603.28315)**

> **作者:** Yangmei Chen; Zhongyuan Zhang; Xikun Zhang; Xinyu Hao; Mingliang Hou; Renqiang Luo; Ziqi Xu
>
> **备注:** 6 pages, IWCMC 2026 accepted
>
> **摘要:** Thyroid nodule classification using ultrasound imaging is essential for early diagnosis and clinical decision-making; however, despite promising performance on in-distribution data, existing deep learning methods often exhibit limited robustness and generalisation when deployed across different ultrasound devices or clinical environments. This limitation is mainly attributed to the pronounced heterogeneity of thyroid ultrasound images, which can lead models to capture spurious correlations rather than reliable diagnostic cues. To address this challenge, we propose PEMV-thyroid, a Prototype-Enhanced Multi-View learning framework that accounts for data heterogeneity by learning complementary representations from multiple feature perspectives and refining decision boundaries through a prototype-based correction mechanism with mixed prototype information. By integrating multi-view representations with prototype-level guidance, the proposed approach enables more stable representation learning under heterogeneous imaging conditions. Extensive experiments on multiple thyroid ultrasound datasets demonstrate that PEMV-thyroid consistently outperforms state-of-the-art methods, particularly in cross-device and cross-domain evaluation scenarios, leading to improved diagnostic accuracy and generalisation performance in real-world clinical settings. The source code is available at this https URL.
>
---
#### [new 243] Unified Restoration-Perception Learning: Maritime Infrared-Visible Image Fusion and Segmentation
- **分类: cs.CV**

- **简介: 该论文属于图像融合与分割任务，解决海洋场景中因雾和强反射导致的图像退化问题，提出新数据集和多任务框架以提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.28414](https://arxiv.org/pdf/2603.28414)**

> **作者:** Weichao Cai; Weiliang Huang; Biao Xue; Chao Huang; Fei Yuan; Bob Zhang
>
> **摘要:** Marine scene understanding and segmentation plays a vital role in maritime monitoring and navigation safety. However, prevalent factors like fog and strong reflections in maritime environments cause severe image degradation, significantly compromising the stability of semantic perception. Existing restoration and enhancement methods typically target specific degradations or focus solely on visual quality, lacking end-to-end collaborative mechanisms that simultaneously improve structural recovery and semantic effectiveness. Moreover, publicly available infrared-visible datasets are predominantly collected from urban scenes, failing to capture the authentic characteristics of coupled degradations in marine environments. To address these challenges, the Infrared-Visible Maritime Ship Dataset (IVMSD) is proposed to cover various maritime scenarios under diverse weather and illumination conditions. Building upon this dataset, a Multi-task Complementary Learning Framework (MCLF) is proposed to collaboratively perform image restoration, multimodal fusion, and semantic segmentation within a unified architecture. The framework includes a Frequency-Spatial Enhancement Complementary (FSEC) module for degradation suppression and structural enhancement, a Semantic-Visual Consistency Attention (SVCA) module for semantic-consistent guidance, and a cross-modality guided attention mechanism for selective fusion. Experimental results on IVMSD demonstrate that the proposed method achieves state-of-the-art segmentation performance, significantly enhancing robustness and perceptual quality under complex maritime conditions.
>
---
#### [new 244] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决MCoT模型中的幻觉问题。通过分析发现幻觉主要源于关联推理步骤中的发散思维，并提出有效策略进行干预。**

- **链接: [https://arxiv.org/pdf/2603.27201](https://arxiv.org/pdf/2603.27201)**

> **作者:** Ji Ma; Wei Suo; Peng Wang; Yanning Zhang
>
> **备注:** CVPR 2026
>
> **摘要:** Multimodal Chain-of-Thought (MCoT) models have demonstrated impressive capability in complex visual reasoning tasks. Unfortunately, recent studies reveal that they suffer from severe hallucination problems due to diminished visual attention during the generation process. However, visual attention decay is a well-studied problem in Large Vision-Language Models (LVLMs). Considering the fundamental differences in reasoning processes between MCoT models and traditional LVLMs, we raise a basic question: Whether MCoT models have unique causes of hallucinations? To answer this question, we systematically investigate the hallucination patterns of MCoT models and find that fabricated texts are primarily generated in associative reasoning steps, which we term divergent thinking. Leveraging these insights, we introduce a simple yet effective strategy that can effectively localize divergent thinking steps and intervene in the decoding process to mitigate hallucinations. Extensive experiments show that our method outperforms existing methods by a large margin. More importantly, our proposed method can be conveniently integrated with other hallucination mitigation methods and further boost their performance. The code is publicly available at this https URL.
>
---
#### [new 245] Synergizing Discriminative Exemplars and Self-Refined Experience for MLLM-based In-Context Learning in Medical Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医疗诊断任务，旨在提升MLLM在医学领域的表现。针对模型泛化能力不足问题，提出结合DECS与SRES的ICL框架，无需更新模型权重即可提高性能。**

- **链接: [https://arxiv.org/pdf/2603.27737](https://arxiv.org/pdf/2603.27737)**

> **作者:** Wenkai Zhao; Zipei Wang; Mengjie Fang; Di Dong; Jie Tian; Lingwei Zhang
>
> **摘要:** General Multimodal Large Language Models (MLLMs) often underperform in capturing domain-specific nuances in medical diagnosis, trailing behind fully supervised baselines. Although fine-tuning provides a remedy, the high costs of expert annotation and massive computational overhead limit its scalability. To bridge this gap without updating the weights of the pre-trained backbone of the MLLM, we propose a Clinician Mimetic Workflow. This is a novel In-Context Learning (ICL) framework designed to synergize Discriminative Exemplar Coreset Selection (DECS) and Self-Refined Experience Summarization (SRES). Specifically, DECS simulates a clinician's ability to reference "anchor cases" by selecting discriminative visual coresets from noisy data at the computational level; meanwhile, SRES mimics the cognition and reflection in clinical diagnosis by distilling diverse rollouts into a dynamic textual Experience Bank. Extensive evaluation across all 12 datasets of the MedMNIST 2D benchmark demonstrates that our method outperforms zero-shot general and medical MLLMs. Simultaneously, it achieves performance levels comparable to fully supervised vision models and domain-specific fine-tuned MLLMs, setting a new benchmark for parameter-efficient medical in-context learning. Our code is available at an anonymous repository: this https URL.
>
---
#### [new 246] Why Aggregate Accuracy is Inadequate for Evaluating Fairness in Law Enforcement Facial Recognition Systems
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文探讨了执法人脸识别系统中聚合准确率无法有效评估公平性的问题，指出需关注子群体的错误分布。任务为公平性评估，解决准确率指标不足的问题，通过分析FPR和FNR进行更全面的评估。**

- **链接: [https://arxiv.org/pdf/2603.28675](https://arxiv.org/pdf/2603.28675)**

> **作者:** Khalid Adnan Alsayed
>
> **备注:** 9 pages, 2 tables, 1 figure. Position paper with empirical subgroup analysis highlighting limitations of aggregate accuracy in fairness evaluation
>
> **摘要:** Facial recognition systems are increasingly deployed in law enforcement and security contexts, where algorithmic decisions can carry significant societal consequences. Despite high reported accuracy, growing evidence demonstrates that such systems often exhibit uneven performance across demographic groups, leading to disproportionate error rates and potential harm. This paper argues that aggregate accuracy is an insufficient metric for evaluating the fairness and reliability of facial recognition systems in high-stakes environments. Through analysis of subgroup-level error distribution, including false positive rate (FPR) and false negative rate (FNR), the paper demonstrates how aggregate performance metrics can obscure critical disparities across demographic groups. Empirical observations show that systems with similar overall accuracy can exhibit substantially different fairness profiles, with subgroup error rates varying significantly despite a single aggregate metric. The paper further examines the operational risks associated with accuracy-centric evaluation practices in law enforcement applications, where misclassification may result in wrongful suspicion or missed identification. It highlights the importance of fairness-aware evaluation approaches and model-agnostic auditing strategies that enable post-deployment assessment of real-world systems. The findings emphasise the need to move beyond accuracy as a primary metric and adopt more comprehensive evaluation frameworks for responsible AI deployment.
>
---
#### [new 247] Clore: Interactive Pathology Image Segmentation with Click-based Local Refinement
- **分类: cs.CV**

- **简介: 该论文属于病理图像分割任务，旨在解决交互式分割中精细结构捕捉不足和交互成本高的问题。提出Clore方法，通过点击引导的分层细化提升分割精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.27625](https://arxiv.org/pdf/2603.27625)**

> **作者:** Tiantong Wang; Minfan Zhao; Jun Shi; Hannan Wang; Yue Dai
>
> **摘要:** Recent advancements in deep learning-based interactive segmentation methods have significantly improved pathology image segmentation. Most existing approaches utilize user-provided positive and negative clicks to guide the segmentation process. However, these methods primarily rely on iterative global updates for refinement, which lead to redundant re-prediction and often fail to capture fine-grained structures or correct subtle errors during localized adjustments. To address this limitation, we propose the Click-based Local Refinement (Clore) pipeline, a simple yet efficient method designed to enhance interactive segmentation. The key innovation of Clore lies in its hierarchical interaction paradigm: the initial clicks drive global segmentation to rapidly outline large target regions, while subsequent clicks progressively refine local details to achieve precise boundaries. This approach not only improves the ability to handle fine-grained segmentation tasks but also achieves high-quality results with fewer interactions. Experimental results on four datasets demonstrate that Clore achieves the best balance between segmentation accuracy and interaction cost, making it an effective solution for efficient and accurate interactive pathology image segmentation.
>
---
#### [new 248] Make It Up: Fake Images, Real Gains in Generalized Few-shot Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于广义小样本语义分割任务，旨在解决标注稀缺导致的类别覆盖不足和伪标签质量差的问题。提出Syn4Seg框架，通过生成合成数据和优化伪标签提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.27206](https://arxiv.org/pdf/2603.27206)**

> **作者:** Guohuan Xie; Xin He; Dingying Fan; Le Zhang; Ming-Ming Cheng; Yun Liu
>
> **摘要:** Generalized few-shot semantic segmentation (GFSS) is fundamentally limited by the coverage of novel-class appearances under scarce annotations. While diffusion models can synthesize novel-class images at scale, practical gains are often hindered by insufficient coverage and noisy supervision when masks are unavailable or unreliable. We propose Syn4Seg, a generation-enhanced GFSS framework designed to expand novel-class coverage while improving pseudo-label quality. Syn4Seg first maximizes prompt-space coverage by constructing an embedding-deduplicated prompt bank for each novel class, yielding diverse yet class-consistent synthetic images. It then performs support-guided pseudo-label estimation via a two-stage refinement that i) filters low-consistency regions to obtain high-precision seeds and ii) relabels uncertain pixels with image-adaptive prototypes that combine global (support) and local (image) statistics. Finally, we refine only boundary-band and unlabeled pixels using a constrained SAM-based update to improve contour fidelity without overwriting high-confidence interiors. Extensive experiments on PASCAL-$5^i$ and COCO-$20^i$ demonstrate consistent improvements in both 1-shot and 5-shot settings, highlighting synthetic data as a scalable path for GFSS with reliable masks and precise boundaries.
>
---
#### [new 249] From Pixels to Reality: Physical-Digital Patch Attacks on Real-World Camera
- **分类: cs.CV**

- **简介: 该论文属于安全领域，针对摄像头认证系统提出物理-数字对抗攻击方法DiPA，解决真实场景下对抗样本的生成与传递问题，通过手机屏幕展示对抗补丁实现高效攻击。**

- **链接: [https://arxiv.org/pdf/2603.28425](https://arxiv.org/pdf/2603.28425)**

> **作者:** Victoria Leonenkova; Ekaterina Shumitskaya; Dmitriy Vatolin; Anastasia Antsiferova
>
> **备注:** Accepted to the PerCom 2026 Demo
>
> **摘要:** This demonstration presents Digital-Physical Adversarial Attacks (DiPA), a new class of practical adversarial attacks against pervasive camera-based authentication systems, where an attacker displays an adversarial patch directly on a smartphone screen instead of relying on printed artifacts. This digital-only physical presentation enables rapid deployment, removes the need for total-variation regularization, and improves patch transferability in black-box conditions. DiPA leverages an ensemble of state-of-the-art face-recognition models (ArcFace, MagFace, CosFace) to enhance transfer across unseen commercial systems. Our interactive demo shows a real-time dodging attack against a deployed face-recognition camera, preventing authorized users from being recognized while participants dynamically adjust patch patterns and observe immediate effects on the sensing pipeline. We further demonstrate DiPA's superiority over existing physical attacks in terms of success rate, feature-space distortion, and reductions in detection confidence, highlighting critical vulnerabilities at the intersection of mobile devices, pervasive vision, and sensor-driven authentication infrastructures.
>
---
#### [new 250] PoseDreamer: Scalable and Photorealistic Human Data Generation Pipeline with Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出PoseDreamer，用于生成高质量3D人体数据，解决标注困难和数据多样性不足的问题。通过扩散模型生成合成数据，提升图像质量和模型性能。**

- **链接: [https://arxiv.org/pdf/2603.28763](https://arxiv.org/pdf/2603.28763)**

> **作者:** Lorenza Prospero; Orest Kupyn; Ostap Viniavskyi; João F. Henriques; Christian Rupprecht
>
> **摘要:** Acquiring labeled datasets for 3D human mesh estimation is challenging due to depth ambiguities and the inherent difficulty of annotating 3D geometry from monocular images. Existing datasets are either real, with manually annotated 3D geometry and limited scale, or synthetic, rendered from 3D engines that provide precise labels but suffer from limited photorealism, low diversity, and high production costs. In this work, we explore a third path: generated data. We introduce PoseDreamer, a novel pipeline that leverages diffusion models to generate large-scale synthetic datasets with 3D mesh annotations. Our approach combines controllable image generation with Direct Preference Optimization for control alignment, curriculum-based hard sample mining, and multi-stage quality filtering. Together, these components naturally maintain correspondence between 3D labels and generated images, while prioritizing challenging samples to maximize dataset utility. Using PoseDreamer, we generate more than 500,000 high-quality synthetic samples, achieving a 76% improvement in image-quality metrics compared to rendering-based datasets. Models trained on PoseDreamer achieve performance comparable to or superior to those trained on real-world and traditional synthetic datasets. In addition, combining PoseDreamer with synthetic datasets results in better performance than combining real-world and synthetic datasets, demonstrating the complementary nature of our dataset. We will release the full dataset and generation code.
>
---
#### [new 251] Brain-Inspired Multimodal Spiking Neural Network for Image-Text Retrieval
- **分类: cs.CV**

- **简介: 该论文属于图像-文本检索任务，旨在解决多模态SNN在语义融合、能耗和速度上的挑战。提出CMSF网络，在低能耗下实现高效准确的多模态匹配。**

- **链接: [https://arxiv.org/pdf/2603.26787](https://arxiv.org/pdf/2603.26787)**

> **作者:** Xintao Zong; Xian Zhong; Wenxuan Liu; Jianhao Ding; Zhaofei Yu; Tiejun Huang
>
> **摘要:** Spiking neural networks (SNNs) have recently shown strong potential in unimodal visual and textual tasks, yet building a directly trained, low-energy, and high-performance SNN for multimodal applications such as image-text retrieval (ITR) remains highly challenging. Existing artificial neural network (ANN)-based methods often pursue richer unimodal semantics using deeper and more complex architectures, while overlooking cross-modal interaction, retrieval latency, and energy efficiency. To address these limitations, we present a brain-inspired Cross-Modal Spike Fusion network (CMSF) and apply it to ITR for the first time. The proposed spike fusion mechanism integrates unimodal features at the spike level, generating enhanced multimodal representations that act as soft supervisory signals to refine unimodal spike embeddings, effectively mitigating semantic loss within CMSF. Despite requiring only two time steps, CMSF achieves top-tier retrieval accuracy, surpassing state-of-the-art ANN counterparts while maintaining exceptionally low energy consumption and high retrieval speed. This work marks a significant step toward multimodal SNNs, offering a brain-inspired framework that unifies temporal dynamics with cross-modal alignment and provides new insights for future spiking-based multimodal research. The code is available at this https URL.
>
---
#### [new 252] K$α$LOS finds Consensus: A Meta-Algorithm for Evaluating Inter-Annotator Agreement in Complex Vision Tasks
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，解决标注一致性评估问题。针对现有指标无法处理空间对应问题，提出KαLOS算法，通过标准化数据集质量评估，提升基准测试可靠性。**

- **链接: [https://arxiv.org/pdf/2603.27197](https://arxiv.org/pdf/2603.27197)**

> **作者:** David Tschirschwitz; Volker Rodehorst
>
> **备注:** Accepted at CVPR 2026. Also known as KALOS
>
> **摘要:** Progress in object detection benchmarks is stagnating. It is limited not by architectures but by the inability to distinguish model improvements from label noise. To restore trust in benchmarking the field requires rigorous quantification of annotation consistency to ensure the reliability of evaluation data. However, standard statistical metrics fail to handle the instance correspondence problem inherent to vision tasks. Furthermore, validating new agreement metrics remains circular because no objective ground truth for agreement exists. This forces reliance on unverifiable heuristics. We propose K$\alpha$LOS (KALOS), a unified meta-algorithm that generalizes the "Localization First" principle to standardize dataset quality evaluation. By resolving spatial correspondence before assessing agreement, our framework transforms complex spatio-categorical problems into nominal reliability matrices. Unlike prior heuristic implementations, K$\alpha$LOS employs a principled, data-driven configuration; by statistically calibrating the localization parameters to the inherent agreement distribution, it generalizes to diverse tasks ranging from bounding boxes to volumetric segmentation or pose estimation. This standardization enables granular diagnostics beyond a single score. These include annotator vitality, collaboration clustering, and localization sensitivity. To validate this approach, we introduce a novel and empirically derived noise generator. Where prior validations relied on uniform error assumptions, our controllable testbed models complex and non-isotropic human variability. This provides evidence of the metric's properties and establishes K$\alpha$LOS as a robust standard for distinguishing signal from noise in modern computer vision benchmarks.
>
---
#### [new 253] ColorFLUX: A Structure-Color Decoupling Framework for Old Photo Colorization
- **分类: cs.CV**

- **简介: 论文提出ColorFLUX框架，解决老照片色彩还原问题。通过结构与颜色解耦及视觉语义提示，提升色彩准确性与真实性。**

- **链接: [https://arxiv.org/pdf/2603.28162](https://arxiv.org/pdf/2603.28162)**

> **作者:** Bingchen Li; Zhixin Wang; Fan Li; Jiaqi Xu; Jiaming Guo; Renjing Pei; Xin Li; Zhibo Chen
>
> **备注:** Accepted by CVPR26
>
> **摘要:** Old photos preserve invaluable historical memories, making their restoration and colorization highly desirable. While existing restoration models can address some degradation issues like denoising and scratch removal, they often struggle with accurate colorization. This limitation arises from the unique degradation inherent in old photos, such as faded brightness and altered color hues, which are different from modern photo distributions, creating a substantial domain gap during colorization. In this paper, we propose a novel old photo colorization framework based on the generative diffusion model FLUX. Our approach introduces a structure-color decoupling strategy that separates structure preservation from color restoration, enabling accurate colorization of old photos while maintaining structural consistency. We further enhance the model with a progressive Direct Preference Optimization (Pro-DPO) strategy, which allows the model to learn subtle color preferences through coarse-to-fine transitions in color augmentation. Additionally, we address the limitations of text-based prompts by introducing visual semantic prompts, which extract fine-grained semantic information directly from old photos, helping to eliminate the color bias inherent in old photos. Experimental results on both synthetic and real datasets demonstrate that our approach outperforms existing state-of-the-art colorization methods, including closed-source commercial models, producing high-quality and vivid colorization.
>
---
#### [new 254] Deep Learning Aided Vision System for Planetary Rovers
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出一种用于行星探测器的视觉系统，结合实时感知与离线地形重建，解决自主导航中的距离估计和目标检测问题。**

- **链接: [https://arxiv.org/pdf/2603.26802](https://arxiv.org/pdf/2603.26802)**

> **作者:** Lomash Relia; Jai G Singla; Amitabh; Nitant Dube
>
> **摘要:** This study presents a vision system for planetary rovers, combining real-time perception with offline terrain reconstruction. The real-time module integrates CLAHE enhanced stereo imagery, YOLOv11n based object detection, and a neural network to estimate object distances. The offline module uses the Depth Anything V2 metric monocular depth estimation model to generate depth maps from captured images, which are fused into dense point clouds using Open3D. Real world distance estimates from the real time pipeline provide reliable metric context alongside the qualitative reconstructions. Evaluation on Chandrayaan 3 NavCam stereo imagery, benchmarked against a CAHV based utility, shows that the neural network achieves a median depth error of 2.26 cm within a 1 to 10 meter range. The object detection model maintains a balanced precision recall tradeoff on grayscale lunar scenes. This architecture offers a scalable, compute-efficient vision solution for autonomous planetary exploration.
>
---
#### [new 255] Rethinking Structure Preservation in Text-Guided Image Editing with Visual Autoregressive Models
- **分类: cs.CV**

- **简介: 该论文属于文本引导的图像编辑任务，旨在解决VAR模型中编辑区域定位不准确和结构一致性差的问题。通过改进token定位、特征注入和强化学习优化，提升编辑质量和结构保持。**

- **链接: [https://arxiv.org/pdf/2603.28367](https://arxiv.org/pdf/2603.28367)**

> **作者:** Tao Xia; Jiawei Liu; Yukun Zhang; Ting Liu; Wei Wang; Lei Zhang
>
> **摘要:** Visual autoregressive (VAR) models have recently emerged as a promising family of generative models, enabling a wide range of downstream vision tasks such as text-guided image editing. By shifting the editing paradigm from noise manipulation in diffusion-based methods to token-level operations, VAR-based approaches achieve better background preservation and significantly faster inference. However, existing VAR-based editing methods still face two key challenges: accurately localizing editable tokens and maintaining structural consistency in the edited results. In this work, we propose a novel text-guided image editing framework rooted in an analysis of intermediate feature distributions within VAR models. First, we introduce a coarse-to-fine token localization strategy that can refine editable regions, balancing editing fidelity and background preservation. Second, we analyze the intermediate representations of VAR models and identify structure-related features, by which we design a simple yet effective feature injection mechanism to enhance structural consistency between the edited and source images. Third, we develop a reinforcement learning-based adaptive feature injection scheme that automatically learns scale- and layer-specific injection ratios to jointly optimize editing fidelity and structure preservation. Extensive experiments demonstrate that our method achieves superior structural consistency and editing quality compared with state-of-the-art approaches, across both local and global editing scenarios.
>
---
#### [new 256] Communicating about Space: Language-Mediated Spatial Integration Across Partial Views
- **分类: cs.CV**

- **简介: 该论文研究多模态大语言模型在空间协作沟通中的表现，旨在解决如何通过对话整合不同视角的空间信息。工作包括构建基准COSMIC，评估模型在空间任务中的能力，并与人类表现对比。**

- **链接: [https://arxiv.org/pdf/2603.27183](https://arxiv.org/pdf/2603.27183)**

> **作者:** Ankur Sikarwar; Debangan Mishra; Sudarshan Nikhil; Ponnurangam Kumaraguru; Aishwarya Agrawal
>
> **摘要:** Humans build shared spatial understanding by communicating partial, viewpoint-dependent observations. We ask whether Multimodal Large Language Models (MLLMs) can do the same, aligning distinct egocentric views through dialogue to form a coherent, allocentric mental model of a shared environment. To study this systematically, we introduce COSMIC, a benchmark for Collaborative Spatial Communication. In this setting, two static MLLM agents observe a 3D indoor environment from different viewpoints and exchange natural-language messages to solve spatial queries. COSMIC contains 899 diverse scenes and 1250 question-answer pairs spanning five tasks. We find a consistent capability hierarchy, MLLMs are most reliable at identifying shared anchor objects across views, perform worse on relational reasoning, and largely fail at building globally consistent maps, performing near chance, even for the frontier models. Moreover, we find thinking capability yields consistent gains in anchor grounding, but is insufficient for higher-level spatial communication. To contextualize model behavior, we additionally collect 250 human-human dialogues. Humans achieve 95% aggregate accuracy, leaving significant room for improvement for even the best performing model Gemini-3-Pro-Thinking which achieves 72% aggregate accuracy. Moreover, human conversations become increasingly specific as partners converge on a shared mental model, whereas model dialogues continue to explore new possibilities rather than converging, consistent with a limited ability to build and maintain a robust shared mental model. Our code and data is available at this https URL
>
---
#### [new 257] Implicit neural representations for larval zebrafish brain microscopy: a reproducible benchmark on the MapZebrain atlas
- **分类: cs.CV; cs.AI; cs.LG; q-bio.NC**

- **简介: 该论文属于神经影像分析任务，解决高分辨率斑马鱼脑显微图像的重建问题。通过对比不同神经表示方法，评估其在保持神经结构细节上的效果，为脑图谱工作流提供基准。**

- **链接: [https://arxiv.org/pdf/2603.26811](https://arxiv.org/pdf/2603.26811)**

> **作者:** Agnieszka Pregowska
>
> **摘要:** Implicit neural representations (INRs) offer continuous coordinate-based encodings for atlas registration, cross-modality resampling, sparse-view completion, and compact sharing of neuroanatomical data. Yet reproducible evaluation is lacking for high-resolution larval zebrafish microscopy, where preserving neuropil boundaries and fine neuronal processes is critical. We present a reproducible INR benchmark for the MapZebrain larval zebrafish brain atlas. Using a unified, seed-controlled protocol, we compare SIREN, Fourier features, Haar positional encoding, and a multi-resolution grid on 950 grayscale microscopy images, including atlas slices and single-neuron projections. Images are normalized with per-image (1,99) percentiles estimated from 10% of pixels in non-held-out columns, and spatial generalization is tested with a deterministic 40% column-wise hold-out along the X-axis. Haar and Fourier achieve the strongest macro-averaged reconstruction fidelity on held-out columns (about 26 dB), while the grid is moderately behind. SIREN performs worse in macro averages but remains competitive on area-weighted micro averages in the all-in-one regime. SSIM and edge-focused error further show that Haar and Fourier preserve boundaries more accurately. These results indicate that explicit spectral and multiscale encodings better capture high-frequency neuroanatomical detail than smoother-bias alternatives. For MapZebrain workflows, Haar and Fourier are best suited to boundary-sensitive tasks such as atlas registration, label transfer, and morphology-preserving sharing, while SIREN remains a lightweight baseline for background modelling or denoising.
>
---
#### [new 258] Spatial Orthogonal Refinement for Robust RGB-Event Visual Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉目标跟踪任务，解决高速运动下RGB图像模糊问题。提出SOR-Track框架，利用事件相机数据与RGB图像融合，提升跟踪鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27913](https://arxiv.org/pdf/2603.27913)**

> **作者:** Dexing Huang; Shiao Wang; Fan Zhang; Xiao Wang
>
> **备注:** Joint International Conference on Automation-Intelligence-Safety and International Symposium on Autonomous Systems 2026 (ICAIS and ISAS 2026)
>
> **摘要:** Robust visual object tracking (VOT) remains challenging in high-speed motion scenarios, where conventional RGB sensors suffer from severe motion blur and performance degradation. Event cameras, with microsecond temporal resolution and high dynamic range, provide complementary structural cues that can potentially compensate for these limitations. However, existing RGB-Event fusion methods typically treat event data as dense intensity representations and adopt black-box fusion strategies, failing to explicitly leverage the directional geometric priors inherently encoded in event streams to rectify degraded RGB features. To address this limitation, we propose SOR-Track, a streamlined framework for robust RGB-Event tracking based on Spatial Orthogonal Refinement (SOR). The core SOR module employs a set of orthogonal directional filters that are dynamically guided by local motion orientations to extract sharp and motion-consistent structural responses from event streams. These responses serve as geometric anchors to modulate and refine aliased RGB textures through an asymmetric structural modulation mechanism, thereby explicitly bridging structural discrepancies between two modalities. Extensive experiments on the large-scale FE108 benchmark demonstrate that SOR-Track consistently outperforms existing fusion-based trackers, particularly under motion blur and low-light conditions. Despite its simplicity, the proposed method offers a principled and physics-grounded approach to multi-modal feature alignment and texture rectification. The source code of this paper will be released on this https URL
>
---
#### [new 259] HighlightBench: Benchmarking Markup-Driven Table Reasoning in Scientific Documents
- **分类: cs.CV**

- **简介: 该论文属于表格理解任务，旨在解决模型对文档中视觉标记（如高亮）的逻辑推理问题。提出HighlightBench基准，分解评估任务以诊断模型在标记驱动表理解中的表现。**

- **链接: [https://arxiv.org/pdf/2603.26784](https://arxiv.org/pdf/2603.26784)**

> **作者:** Lexin Wang; Shenghua Liu; Yiwei Wang; Yujun Cai; Yuyao Ge; Jiayu Yao; Jiafeng Guo; Xueqi Cheng
>
> **摘要:** Visual markups such as highlights, underlines, and bold text are common in table-centric documents. Although multimodal large language models (MLLMs) have made substantial progress in document understanding, their ability to treat such cues as explicit logical directives remains under-explored. More importantly, existing evaluations cannot distinguish whether a model fails to see the markup or fails to reason with it. This creates a key blind spot in assessing markup-conditioned behavior over tables. To address this gap, we introduce HighlightBench, a diagnostic benchmark for markup-driven table understanding that decomposes evaluation into five task families: Markup Grounding, Constrained Retrieval, Local Relations, Aggregation \& Comparison, and Consistency \& Missingness. We further provide a reference pipeline that makes intermediate decisions explicit, enabling reproducible baselines and finer-grained attribution of errors along the perception-to-execution chain. Experiments show that even strong models remain unstable when visual cues must be consistently aligned with symbolic reasoning under structured output constraints.
>
---
#### [new 260] An Annotation-to-Detection Framework for Autonomous and Robust Vine Trunk Localization in the Field by Mobile Agricultural Robots
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于农业机器人领域的物体检测任务，旨在解决无标记数据下 vine trunk 的定位问题。通过多模态数据融合与增量标注方法，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.26724](https://arxiv.org/pdf/2603.26724)**

> **作者:** Dimitrios Chatziparaschis; Elia Scudiero; Brent Sams; Konstantinos Karydis
>
> **备注:** 7 pages, 6 figures, conference
>
> **摘要:** The dynamic and heterogeneous nature of agricultural fields presents significant challenges for object detection and localization, particularly for autonomous mobile robots that are tasked with surveying previously unseen unstructured environments. Concurrently, there is a growing need for real-time detection systems that do not depend on large-scale manually labeled real-world datasets. In this work, we introduce a comprehensive annotation-to-detection framework designed to train a robust multi-modal detector using limited and partially labeled training data. The proposed methodology incorporates cross-modal annotation transfer and an early-stage sensor fusion pipeline, which, in conjunction with a multi-stage detection architecture, effectively trains and enhances the system's multi-modal detection capabilities. The effectiveness of the framework was demonstrated through vine trunk detection in novel vineyard settings that featured diverse lighting conditions and varying crop densities to validate performance. When integrated with a customized multi-modal LiDAR and Odometry Mapping (LOAM) algorithm and a tree association module, the system demonstrated high-performance trunk localization, successfully identifying over 70% of trees in a single traversal with a mean distance error of less than 0.37m. The results reveal that by leveraging multi-modal, incremental-stage annotation and training, the proposed framework achieves robust detection performance regardless of limited starting annotations, showcasing its potential for real-world and near-ground agricultural applications.
>
---
#### [new 261] SceneExpander: Expanding 3D Scenes with Free-Form Inserted Views
- **分类: cs.CV**

- **简介: 该论文提出SceneExpander，解决3D场景扩展中的视图不一致问题，通过引入生成模型插入新视角并进行适应性优化，提升场景重建质量。**

- **链接: [https://arxiv.org/pdf/2603.27084](https://arxiv.org/pdf/2603.27084)**

> **作者:** Zijian He; enjie Liu; Yihao Wang; Weizhi Zhong; Huan Yuan; Kun Gai; Guangrun Wang; Guanbin Li
>
> **摘要:** World building with 3D scene representations is increasingly important for content creation, simulation, and interactive experiences, yet real workflows are inherently iterative: creators must repeatedly extend an existing scene under user control. Motivated by this research gap, we study 3D scene expansion in a user-centric workflow: starting from a real scene captured by multi-view images, we extend its coverage by inserting an additional view synthesized by a generative model. Unlike simple object editing or style transfer in a fixed scene, the inserted view is often 3D-misaligned with the original reconstruction, introducing geometry shifts, hallucinated content, or view-dependent artifacts that break global multi-view consistency. To address the challenge, we propose SceneExpander, which applies test-time adaptation to a parametric feed-forward 3D reconstruction model with two complementary distillation signals: anchor distillation stabilizes the original scene by distilling geometric cues from the captured views, while inserted-view self-distillation preserves observation-supported predictions yet adapts latent geometry and appearance to accommodate the misaligned inserted view. Experiments on ETH scenes and online data demonstrate improved expansion behavior and reconstruction quality under misalignment.
>
---
#### [new 262] MotiMem: Motion-Aware Approximate Memory for Energy-Efficient Neural Perception in Autonomous Vehicles
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶感知任务，解决高分辨率传感器带来的内存能耗问题。提出MotiMem，通过运动感知和稀疏编码降低内存动态能耗，同时保持检测精度。**

- **链接: [https://arxiv.org/pdf/2603.27108](https://arxiv.org/pdf/2603.27108)**

> **作者:** Haohua Que; Mingkai Liu; Jiayue Xie; Haojia Gao; Jiajun Sun; Hongyi Xu; Handong Yao; Fei Qiao
>
> **备注:** 8 pages,6 figures,conference
>
> **摘要:** High-resolution sensors are critical for robust autonomous perception but impose a severe memory wall on battery-constrained electric vehicles. In these systems, data movement energy often outweighs computation. Traditional image compression is ill-suited as it is semantically blind and optimizes for storage rather than bus switching activity. We propose MotiMem, a hardware-software co-designed interface. Exploiting temporal coherence,MotiMem uses lightweight 2D Motion Propagation to dynamically identify Regions of Interest (RoI). Complementing this, a Hybrid Sparsity-Aware Coding scheme leverages adaptive inversion and truncation to induce bitlevel sparsity. Extensive experiments across nuScenes, Waymo, and KITTI with 16 detection models demonstrate that MotiMem reduces memory-interface dynamic energy by approximately 43 percent while retaining approximately 93 percent of the object detection accuracy, establishing a new Pareto frontier significantly superior to standard codecs like JPEG and WebP.
>
---
#### [new 263] When Surfaces Lie: Exploiting Wrinkle-Induced Attention Shift to Attack Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型的鲁棒性问题，针对柔性表面褶皱等非刚性变形提出攻击方法，通过生成逼真扰动降低模型性能。**

- **链接: [https://arxiv.org/pdf/2603.27759](https://arxiv.org/pdf/2603.27759)**

> **作者:** Chengyin Hu; Xuemeng Sun; Jiajun Han; Qike Zhang; Xiang Chen; Xin Wang; Yiwei Wei; Jiahua Long
>
> **摘要:** Visual-Language Models (VLMs) have demonstrated exceptional cross-modal understanding across various tasks, including zero-shot classification, image captioning, and visual question answering. However, their robustness to physically plausible non-rigid deformations-such as wrinkles on flexible surfaces-remains poorly understood. In this work, we propose a parametric structural perturbation method inspired by the mechanics of three-dimensional fabric wrinkles. Specifically, our method generates photorealistic non-rigid perturbations by constructing multi-scale wrinkle fields and integrating displacement field distortion with surface-consistent appearance variations. To achieve an optimal balance between visual naturalness and adversarial effectiveness, we design a hierarchical fitness function in a low-dimensional parameter space and employ an optimization-based search strategy. We evaluate our approach using a two-stage framework: perturbations are first optimized on a zero-shot classification proxy task and subsequently assessed for transferability on generative tasks. Experimental results demonstrate that our method significantly degrades the performance of various state-of-the-art VLMs, consistently outperforming baselines in both image captioning and visual question-answering tasks.
>
---
#### [new 264] V-CAST: Video Curvature-Aware Spatio-Temporal Pruning for Efficient Video Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于视频大模型优化任务，旨在解决长序列推理中冗余视觉标记问题。提出V-CAST方法，通过时空剪枝提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.27650](https://arxiv.org/pdf/2603.27650)**

> **作者:** Xinying Lin; Xuyang Liu; Yiyu Wang; Teng Ma; Wenqi Ren
>
> **备注:** Code: \url{this https URL}
>
> **摘要:** Video large language models (VideoLLMs) show strong capability in video understanding, yet long-context inference is still dominated by massive redundant visual tokens in the prefill stage. We revisit token compression for VideoLLMs under a tight budget and identify a key bottleneck, namely insufficient spatio-temporal information coverage. Existing methods often introduce discontinuous coverage through coarse per-frame allocation or scene segmentation, and token merging can further misalign spatio-temporal coordinates under MRoPE-style discrete (t,h,w) bindings. To address these issues, we propose V-CAST (Video Curvature-Aware Spatio-Temporal Pruning), a training-free, plug-and-play pruning policy for long-context video inference. V-CAST casts token compression as a trajectory approximation problem and introduces a curvature-guided temporal allocation module that routes per-frame token budgets to semantic turns and event boundaries. It further adopts a dual-anchor spatial selection mechanism that preserves high-entropy visual evidence without attention intervention, while keeping retained tokens at their original coordinates to maintain positional alignment. Extensive experiments across multiple VideoLLMs of different architectures and scales demonstrate that V-CAST achieves 98.6% of the original performance, outperforms the second-best method by +1.1% on average, and reduces peak memory and total latency to 86.7% and 86.4% of vanilla Qwen3-VL-8B-Instruct.
>
---
#### [new 265] SVH-BD : Synthetic Vegetation Hyperspectral Benchmark Dataset for Emulation of Remote Sensing Images
- **分类: cs.CV; eess.SP**

- **简介: 该论文提出SVH-BD数据集，用于遥感图像模拟，解决植被特征反演与不确定性量化问题，通过合成高光谱数据支持机器学习和辐射传输模拟研究。**

- **链接: [https://arxiv.org/pdf/2603.28390](https://arxiv.org/pdf/2603.28390)**

> **作者:** Chedly Ben Azizi; Claire Guilloteau; Gilles Roussel; Matthieu Puigt
>
> **摘要:** This dataset provides a large collection of 10,915 synthetic hyperspectral image cubes paired with pixel-level vegetation trait maps, designed to support research in radiative transfer emulation, vegetation trait retrieval, and uncertainty quantification. Each hyperspectral cube contains 211 bands spanning 400--2500 nm at 10 nm resolution and a fixed spatial layout of 64 \times 64 pixels, offering continuous simulated surface reflectance spectra suitable for emulator development and machine-learning tasks requiring high spectral detail. Vegetation traits were derived by inverting Sentinel-2 Level-2A surface reflectance using a PROSAIL-based lookup-table approach, followed by forward PROSAIL simulations to generate hyperspectral reflectance under physically consistent canopy and illumination conditions. The dataset covers four ecologically diverse regions -- East Africa, Northern France, Eastern India, and Southern Spain -- and includes 5th and 95th percentile uncertainty maps as well as Sentinel-2 scene classification layers. This resource enables benchmarking of inversion methods, development of fast radiative transfer emulators, and studies of spectral--biophysical relationships under controlled yet realistic environmental variability.
>
---
#### [new 266] DipGuava: Disentangling Personalized Gaussian Features for 3D Head Avatars from Monocular Video
- **分类: cs.CV**

- **简介: 该论文属于3D头像生成任务，旨在解决个性化细节缺失问题。通过分解面部外观为结构和残差两部分，提升重建精度与真实感。**

- **链接: [https://arxiv.org/pdf/2603.28003](https://arxiv.org/pdf/2603.28003)**

> **作者:** Jeonghaeng Lee; Seok Keun Choi; Zhixuan Li; Weisi Lin; Sanghoon Lee
>
> **备注:** AAAI 2026
>
> **摘要:** While recent 3D head avatar creation methods attempt to animate facial dynamics, they often fail to capture personalized details, limiting realism and expressiveness. To fill this gap, we present DipGuava (Disentangled and Personalized Gaussian UV Avatar), a novel 3D Gaussian head avatar creation method that successfully generates avatars with personalized attributes from monocular video. DipGuava is the first method to explicitly disentangle facial appearance into two complementary components, trained in a structured two-stage pipeline that significantly reduces learning ambiguity and enhances reconstruction fidelity. In the first stage, we learn a stable geometry-driven base appearance that captures global facial structure and coarse expression-dependent variations. In the second stage, the personalized residual details not captured in the first stage are predicted, including high-frequency components and nonlinearly varying features such as wrinkles and subtle skin deformations. These components are fused via dynamic appearance fusion that integrates residual details after deformation, ensuring spatial and semantic alignment. This disentangled design enables DipGuava to generate photorealistic, identity-preserving avatars, consistently outperforming prior methods in both visual quality and quantitativeperformance, as demonstrated in extensive experiments.
>
---
#### [new 267] HMPDM: A Diffusion Model for Driving Video Prediction with Historical Motion Priors
- **分类: cs.CV**

- **简介: 该论文属于视频预测任务，旨在解决自动驾驶中场景演化预测的时空一致性与多样性问题。提出HMPDM模型，通过历史运动先验增强预测效果。**

- **链接: [https://arxiv.org/pdf/2603.27371](https://arxiv.org/pdf/2603.27371)**

> **作者:** Ke Li; Tianjia Yang; Kaidi Liang; Xianbiao Hu; Ruwen Qin
>
> **摘要:** Video prediction is a useful function for autonomous driving, enabling intelligent vehicles to reliably anticipate how driving scenes will evolve and thereby supporting reasoning and safer planning. However, existing models are constrained by multi-stage training pipelines and remain insufficient in modeling the diverse motion patterns in real driving scenes, leading to degraded temporal consistency and visual quality. To address these challenges, this paper introduces the historical motion priors-informed diffusion model (HMPDM), a video prediction model that leverages historical motion priors to enhance motion understanding and temporal coherence. The proposed deep learning system introduces three key designs: (i) a Temporal-aware Latent Conditioning (TaLC) module for implicit historical motion injection; (ii) a Motion-aware Pyramid Encoder (MaPE) for multi-scale motion representation; (iii) a Self-Conditioning (SC) strategy for stable iterative denoising. Extensive experiments on the Cityscapes and KITTI benchmarks demonstrate that HMPDM outperforms state-of-the-art video prediction methods with efficiency, achieving a 28.2% improvement in FVD on Cityscapes under the same monocular RGB input configuration setting. The implementation codes are publicly available at this https URL.
>
---
#### [new 268] Unsafe2Safe: Controllable Image Anonymization for Downstream Utility
- **分类: cs.CV; cs.CY; cs.LG**

- **简介: 该论文提出Unsafe2Safe，解决图像隐私保护问题。通过检测敏感区域并进行可控编辑，确保数据隐私同时保持下游任务性能。**

- **链接: [https://arxiv.org/pdf/2603.28605](https://arxiv.org/pdf/2603.28605)**

> **作者:** Mih Dinh; SouYoung Jin
>
> **备注:** Accepted at CVPR 2026 and CVPR 2026 Workshop on Machine Unlearning for Computer Vision
>
> **摘要:** Large-scale image datasets frequently contain identifiable or sensitive content, raising privacy risks when training models that may memorize and leak such information. We present Unsafe2Safe, a fully automated pipeline that detects privacy-prone images and rewrites only their sensitive regions using multimodally guided diffusion editing. Unsafe2Safe operates in two stages. Stage 1 uses a vision-language model to (i) inspect images for privacy risks, (ii) generate paired private and public captions that respectively include and omit sensitive attributes, and (iii) prompt a large language model to produce structured, identity-neutral edit instructions conditioned on the public caption. Stage 2 employs instruction-driven diffusion editors to apply these dual textual prompts, producing privacy-safe images that preserve global structure and task-relevant semantics while neutralizing private content. To measure anonymization quality, we introduce a unified evaluation suite covering Quality, Cheating, Privacy, and Utility dimensions. Across MS-COCO, Caltech101, and MIT Indoor67, Unsafe2Safe reduces face similarity, text similarity, and demographic predictability by large margins, while maintaining downstream model accuracy comparable to training on raw data. Fine-tuning diffusion editors on our automatically generated triplets (private caption, public caption, edit instruction) further improves both privacy protection and semantic fidelity. Unsafe2Safe provides a scalable, principled solution for constructing large, privacy-safe datasets without sacrificing visual consistency or downstream utility.
>
---
#### [new 269] Intelligent Road Condition Monitoring using 3D In-Air SONAR Sensing
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究利用3D声纳传感器监测道路状况，解决道路材料分类和损伤检测问题。通过实验验证了声纳在恶劣环境下的有效性，但损伤检测准确率仍需提升。**

- **链接: [https://arxiv.org/pdf/2603.28141](https://arxiv.org/pdf/2603.28141)**

> **作者:** Amber Cassimon; Robin Kerstens; Walter Daems; Jan Steckel
>
> **备注:** 10 pages, 9 figures, 2 tables
>
> **摘要:** In this paper, we investigate the capabilities of in-air 3D SONAR sensors for the monitoring of road surface conditions. Concretely, we consider two applications: Road material classification and Road damage detection and classification. While such tasks can be performed with other sensor modalities, such as camera sensors and LiDAR sensors, these sensor modalities tend to fail in harsh sensing conditions, such as heavy rain, smoke or fog. By using a sensing modality that is robust to such interference, we enable the creation of opportunistic sensing applications, where vehicles performing other tasks (garbage collection, mail delivery, etc.) can also be used to monitor the condition of the road. For these tasks, we use a single dataset, in which different types of damages are annotated, with labels including the material of the road surface. In the material classification task, we differentiate between three different road materials: Asphalt, Concrete and Element roads. In the damage detection and classification task, we determine if there is damage, and what type of damage (independent of material type), without localizing the damage. We are succesful in determining the road surface type from SONAR sensor data, with F1 scores approaching 90% on the test set, but find that for the detection of damages performace lags, with F1 score around 75%. From this, we conclude that SONAR sensing is a promising modality to include in opportunistic sensing-based pavement management systems, but that further research is needed to reach the desired accuracy.
>
---
#### [new 270] JaWildText: A Benchmark for Vision-Language Models on Japanese Scene Text Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出JaWildText，一个针对日语文本场景的视觉-语言模型基准。解决日语场景文本理解中的挑战，如多语种混杂、竖排书写等。包含三个任务，用于评估模型在日语文本理解方面的能力。**

- **链接: [https://arxiv.org/pdf/2603.27942](https://arxiv.org/pdf/2603.27942)**

> **作者:** Koki Maeda; Naoaki Okazaki
>
> **备注:** 18 pages
>
> **摘要:** Japanese scene text poses challenges that multilingual benchmarks often fail to capture, including mixed scripts, frequent vertical writing, and a character inventory far larger than the Latin alphabet. Although Japanese is included in several multilingual benchmarks, these resources do not adequately capture the language-specific complexities. Meanwhile, existing Japanese visual text datasets have primarily focused on scanned documents, leaving in-the-wild scene text underexplored. To fill this gap, we introduce JaWildText, a diagnostic benchmark for evaluating vision-language models (VLMs) on Japanese scene text understanding. JaWildText contains 3,241 instances from 2,961 images newly captured in Japan, with 1.12 million annotated characters spanning 3,643 unique character types. It comprises three complementary tasks that vary in visual organization, output format, and writing style: (i) Dense Scene Text Visual Question Answering (STVQA), which requires reasoning over multiple pieces of visual text evidence; (ii) Receipt Key Information Extraction (KIE), which tests layout-aware structured extraction from mobile-captured receipts; and (iii) Handwriting OCR, which evaluates page-level transcription across various media and writing directions. We evaluate 14 open-weight VLMs and find that the best model achieves an average score of 0.64 across the three tasks. Error analyses show recognition remains the dominant bottleneck, especially for kanji. JaWildText enables fine-grained, script-aware diagnosis of Japanese scene text capabilities, and will be released with evaluation code.
>
---
#### [new 271] Elucidating the Design Space of Flow Matching for Cellular Microscopy
- **分类: cs.CV**

- **简介: 该论文属于生成模型任务，旨在优化流匹配模型以模拟细胞对生物扰动的响应。通过系统分析设计空间，提出简洁有效的训练方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.26790](https://arxiv.org/pdf/2603.26790)**

> **作者:** Charles Jones; Emmanuel Noutahi; Jason Hartford; Cian Eastwood
>
> **摘要:** Flow-matching generative models are increasingly used to simulate cell responses to biological perturbations. However, the design space for building such models is large and underexplored. We systematically analyse the design space of flow matching models for cell-microscopy images, finding that many popular techniques are unnecessary and can even hurt performance. We develop a simple, stable, and scalable recipe which we use to train our foundation model. We scale our model to two orders of magnitude larger than prior methods, achieving a two-fold FID and ten-fold KID improvement over prior methods. We then fine-tune our model with pre-trained molecular embeddings to achieve state-of-the-art performance simulating responses to unseen molecules. Code is available at this https URL
>
---
#### [new 272] SaSaSaSa2VA: 2nd Place of the 5th PVUW MeViS-Text Track
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，解决运动中心的文本引用问题。通过引入目标存在感知验证机制，提升模型性能，获得第二名。**

- **链接: [https://arxiv.org/pdf/2603.27241](https://arxiv.org/pdf/2603.27241)**

> **作者:** Dengxian Gong; Quanzhu Niu; Shihao Chen; Yuanzheng Wu; Yikang Zhou; Tao Zhang; Haobo Yuan; Lu Qi; Shunping Ji
>
> **摘要:** Referring video object segmentation (RVOS) commonly grounds targets in videos based on static textual cues. MeViS benchmark extends this by incorporating motion-centric expressions (referring & reasoning motion expressions) and introducing no-target queries. Extending SaSaSa2VA, where increased input frames and [SEG] tokens already strengthen the Sa2VA backbone, we adopt a simple yet effective target existence-aware verification mechanism, leading to Still Awesome SaSaSa2VA (SaSaSaSa2VA). Despite its simplicity, the method achieves a final score of 89.19 in the 5th PVUW Challenge (MeViS-Text Track), securing 2nd place. Both quantitative results and ablations suggest that this existence-aware verification strategy is sufficient to unlock strong performance on motion-centric referring tasks.
>
---
#### [new 273] LLM Enhanced Action Recognition via Hierarchical Global-Local Skeleton-Language Model
- **分类: cs.CV**

- **简介: 该论文属于动作识别任务，旨在解决骨架动作识别中语义建模不足的问题。提出HocSLM模型，结合全局局部网络与视觉语言模型，提升动作语义理解和跨模态对齐能力。**

- **链接: [https://arxiv.org/pdf/2603.27103](https://arxiv.org/pdf/2603.27103)**

> **作者:** Ruosi Wang; Fangwei Zuo; Lei Li; Zhaoqiang Xia
>
> **摘要:** Skeleton-based human action recognition has achieved remarkable progress in recent years. However, most existing GCN-based methods rely on short-range motion topologies, which not only struggle to capture long-range joint dependencies and complex temporal dynamics but also limit cross-modal semantic alignment and understanding due to insufficient modeling of action semantics. To address these challenges, we propose a hierarchical global-local skeleton-language model (HocSLM), enabling the large action model be more representative of action semantics. First, we design a hierarchical global-local network (HGLNet) that consists of a composite-topology spatial module and a dual-path hierarchical temporal module. By synergistically integrating multi-level global and local modules, HGLNet achieves dynamically collaborative modeling at both global and local scales while preserving prior knowledge of human physical structure, significantly enhancing the model's representation of complex spatio-temporal relationships. Then, a large vision-language model (VLM) is employed to generate textual descriptions by passing the original RGB video sequences to this model, providing the rich action semantics for further training the skeleton-language model. Furthermore, we introduce a skeleton-language sequential fusion module by combining the features from HGLNet and the generated descriptions, which utilizes a skeleton-language model (SLM) for aligning skeletal spatio-temporal features and textual action descriptions precisely within a unified semantic space. The SLM model could significantly enhance the HGLNet's semantic discrimination capabilities and cross-modal understanding abilities. Extensive experiments demonstrate that the proposed HocSLM achieves the state-of-the-art performance on three mainstream benchmark datasets: NTU RGB+D 60, NTU RGB+D 120, and Northwestern-UCLA.
>
---
#### [new 274] Bridging Visual Representation and Reinforcement Learning from Verifiable Rewards in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态强化学习任务，解决LVLMs中视觉信息与强化学习结合不足的问题。提出KAWHI方法，将视觉信息有效融入奖励优化，提升多模态推理性能。**

- **链接: [https://arxiv.org/pdf/2603.27375](https://arxiv.org/pdf/2603.27375)**

> **作者:** Yuhang Han; Yuyang Wu; Zhengbo Jiao; Yiyu Wang; Xuyang Liu; Shaobo Wang; Hanlin Xu; Xuming Hu; Linfeng Zhang
>
> **备注:** Homepage: \url{this https URL}
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) has substantially enhanced the reasoning capabilities of large language models in abstract reasoning tasks. However, its application to Large Vision-Language Models (LVLMs) remains constrained by a structural representational bottleneck. Existing approaches generally lack explicit modeling and effective utilization of visual information, preventing visual representations from being tightly coupled with the reinforcement learning optimization process and thereby limiting further improvements in multimodal reasoning performance. To address this limitation, we propose KAWHI (Key-Region Aligned Weighted Harmonic Incentive), a plug-and-play reward reweighting mechanism that explicitly incorporates structured visual information into uniform reward policy optimization methods (e.g., GRPO and GSPO). The method adaptively localizes semantically salient regions through hierarchical geometric aggregation, identifies vision-critical attention heads via structured attribution, and performs paragraph-level credit reallocation to align spatial visual evidence with semantically decisive reasoning steps. Extensive empirical evaluations on diverse reasoning benchmarks substantiate KAWHI as a general-purpose enhancement module, consistently improving the performance of various uniform reward optimization methods. Project page: KAWHI (this https URL)
>
---
#### [new 275] ObjectMorpher: 3D-Aware Image Editing via Deformable 3DGS Models
- **分类: cs.CV**

- **简介: 该论文提出ObjectMorpher，解决3D-aware图像编辑任务中的对象级控制问题。通过3DGS模型实现高效、逼真的编辑，提升可控性与效果。**

- **链接: [https://arxiv.org/pdf/2603.28152](https://arxiv.org/pdf/2603.28152)**

> **作者:** Yuhuan Xie; Aoxuan Pan; Yi-Hua Huang; Chirui Chang; Peng Dai; Xin Yu; Xiaojuan Qi
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Achieving precise, object-level control in image editing remains challenging: 2D methods lack 3D awareness and often yield ambiguous or implausible results, while existing 3D-aware approaches rely on heavy optimization or incomplete monocular reconstructions. We present ObjectMorpher, a unified, interactive framework that converts ambiguous 2D edits into geometry-grounded operations. ObjectMorpher lifts target instances with an image-to-3D generator into editable 3D Gaussian Splatting (3DGS), enabling fast, identity-preserving manipulation. Users drag control points; a graph-based non-rigid deformation with as-rigid-as-possible (ARAP) constraints ensures physically sensible shape and pose changes. A composite diffusion module harmonizes lighting, color, and boundaries for seamless reintegration. Across diverse categories, ObjectMorpher delivers fine-grained, photorealistic edits with superior controllability and efficiency, outperforming 2D drag and 3D-aware baselines on KID, LPIPS, SIFID, and user preference.
>
---
#### [new 276] Integrating Multimodal Large Language Model Knowledge into Amodal Completion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像补全任务，解决遮挡物体的完整重建问题。通过引入多模态大语言模型知识，提出AmodalCG框架，提升补全准确性。**

- **链接: [https://arxiv.org/pdf/2603.28333](https://arxiv.org/pdf/2603.28333)**

> **作者:** Heecheol Yun; Eunho Yang
>
> **摘要:** With the widespread adoption of autonomous vehicles and robotics, amodal completion, which reconstructs the occluded parts of people and objects in an image, has become increasingly crucial. Just as humans infer hidden regions based on prior experience and common sense, this task inherently requires physical knowledge about real-world entities. However, existing approaches either depend solely on the image generation ability of visual generative models, which lack such knowledge, or leverage it only during the segmentation stage, preventing it from explicitly guiding the completion process. To address this, we propose AmodalCG, a novel framework that harnesses the real-world knowledge of Multimodal Large Language Models (MLLMs) to guide amodal completion. Our framework first assesses the extent of occlusion to selectively invoke MLLM guidance only when the target object is heavily occluded. If guidance is required, the framework further incorporates MLLMs to reason about both the (1) extent and (2) content of the missing regions. Finally, a visual generative model integrates these guidance and iteratively refines imperfect completions that may arise from inaccurate MLLM guidance. Experimental results on various real-world images show impressive improvements compared to all existing works, suggesting MLLMs as a promising direction for addressing challenging amodal completion.
>
---
#### [new 277] Effort-Based Criticality Metrics for Evaluating 3D Perception Errors in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶感知误差评估任务，旨在解决传统关键性指标无法区分误报与漏报问题。提出三种基于努力的新指标，量化感知错误的安全影响。**

- **链接: [https://arxiv.org/pdf/2603.28029](https://arxiv.org/pdf/2603.28029)**

> **作者:** Sharang Kaul; Simon Bultmann; Mario Berk; Abhinav Valada
>
> **摘要:** Criticality metrics such as time-to-collision (TTC) quantify collision urgency but conflate the consequences of false-positive (FP) and false-negative (FN) perception errors. We propose two novel effort-based metrics: False Speed Reduction (FSR), the cumulative velocity loss from persistent phantom detections, and Maximum Deceleration Rate (MDR), the peak braking demand from missed objects under a constant-acceleration model. These longitudinal metrics are complemented by Lateral Evasion Acceleration (LEA), adapted from prior lateral evasion kinematics and coupled with reachability-based collision timing to quantify the minimum steering effort to avoid a predicted collision. A reachability-based ellipsoidal collision filter ensures only dynamically plausible threats are scored, with frame-level matching and track-level aggregation. Evaluation of different perception pipelines on nuScenes and Argoverse~2 shows that 65-93% of errors are non-critical, and Spearman correlation analysis confirms that all three metrics capture safety-relevant information inaccessible to established time-based, deceleration-based, or normalized criticality measures, enabling targeted mining of the most critical perception failures.
>
---
#### [new 278] Curriculum-Guided Myocardial Scar Segmentation for Ischemic and Non-ischemic Cardiomyopathy
- **分类: cs.CV**

- **简介: 该论文属于心肌瘢痕分割任务，旨在解决LGE-CMR图像中瘢痕分割困难的问题。通过引入课程学习策略，提升模型在不确定标签和微小瘢痕上的分割性能。**

- **链接: [https://arxiv.org/pdf/2603.28560](https://arxiv.org/pdf/2603.28560)**

> **作者:** Nivetha Jayakumar; Jonathan Pan; Shuo Wang; Bishow Paudel; Nisha Hosadurg; Cristiane C. Singulane; Sivam Bhatt; Amit R. Patel; Miaomiao Zhang
>
> **摘要:** Identification and quantification of myocardial scar is important for diagnosis and prognosis of cardiovascular diseases. However, reliable scar segmentation from Late Gadolinium Enhancement Cardiac Magnetic Resonance (LGE-CMR) images remains a challenge due to variations in contrast enhancement across patients, suboptimal imaging conditions such as post contrast washout, and inconsistencies in ground truth annotations on diffuse scars caused by inter observer variability. In this work, we propose a curriculum learning-based framework designed to improve segmentation performance under these challenging conditions. The method introduces a progressive training strategy that guides the model from high-confidence, clearly defined scar regions to low confidence or visually ambiguous samples with limited scar burden. By structuring the learning process in this manner, the network develops robustness to uncertain labels and subtle scar appearances that are often underrepresented in conventional training pipelines. Experimental results show that the proposed approach enhances segmentation accuracy and consistency, particularly for cases with minimal or diffuse scar, outperforming standard training baselines. This strategy provides a principled way to leverage imperfect data for improved myocardial scar quantification in clinical applications. Our code is publicly available on GitHub.
>
---
#### [new 279] TrackMAE: Video Representation Learning via Track Mask and Predict
- **分类: cs.CV**

- **简介: 该论文提出TrackMAE，解决视频表征学习中运动信息编码不足的问题。通过显式利用运动轨迹作为重建信号，提升视频表示的判别性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.27268](https://arxiv.org/pdf/2603.27268)**

> **作者:** Renaud Vandeghen; Fida Mohammad Thoker; Marc Van Droogenbroeck; Bernard Ghanem
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Masked video modeling (MVM) has emerged as a simple and scalable self-supervised pretraining paradigm, but only encodes motion information implicitly, limiting the encoding of temporal dynamics in the learned representations. As a result, such models struggle on motion-centric tasks that require fine-grained motion awareness. To address this, we propose TrackMAE, a simple masked video modeling paradigm that explicitly uses motion information as a reconstruction signal. In TrackMAE, we use an off-the-shelf point tracker to sparsely track points in the input videos, generating motion trajectories. Furthermore, we exploit the extracted trajectories to improve random tube masking with a motion-aware masking strategy. We enhance video representations learned in both pixel and feature semantic reconstruction spaces by providing a complementary supervision signal in the form of motion targets. We evaluate on six datasets across diverse downstream settings and find that TrackMAE consistently outperforms state-of-the-art video self-supervised learning baselines, learning more discriminative and generalizable representations. Code available at this https URL
>
---
#### [new 280] Quantized Vision-Language Models for Damage Assessment: A Comparative Study of LLaVA-1.5-7B Quantization Levels
- **分类: cs.CV**

- **简介: 该论文属于桥梁损伤评估任务，旨在通过量化视觉-语言模型实现自动化损伤识别，解决描述质量、速度与资源消耗之间的平衡问题。**

- **链接: [https://arxiv.org/pdf/2603.26770](https://arxiv.org/pdf/2603.26770)**

> **作者:** Takato Yasuno
>
> **备注:** 16 pages, 4 figures, 8 tables
>
> **摘要:** Bridge infrastructure inspection is a critical but labor-intensive task requiring expert assessment of structural damage such as rebar exposure, cracking, and corrosion. This paper presents a comprehensive study of quantized Vision-Language Models (VLMs) for automated bridge damage assessment, focusing on the trade-offs between description quality, inference speed, and resource requirements. We develop an end-to-end pipeline combining LLaVA-1.5-7B for visual damage analysis, structured JSON extraction, and rule-based priority scoring. To enable deployment on consumer-grade GPUs, we conduct a systematic comparison of three quantization levels: Q4_K_M, Q5_K_M, and Q8\_0 across 254 rebar exposure images. We introduce a 5-point quality evaluation framework assessing damage type recognition, severity classification. Our results demonstrate that Q5_K_M achieves the optimal balance: quality score 3.18$\pm$1.35/5.0, inference time 5.67s/image, and 0.56 quality/sec efficiency -- 8.5% higher quality than Q4_K_M with only 4.5% speed reduction, while matching Q8_0's quality with 25% faster inference. Statistical analysis reveals Q5_K_M exhibits the weakest text-quality correlation (-0.148), indicating consistent performance regardless of description length.
>
---
#### [new 281] IP-SAM: Prompt-Space Conditioning for Prompt-Absent Camouflaged Object Detection
- **分类: cs.CV**

- **简介: 该论文提出IP-SAM，解决伪装目标检测中无提示输入的自动分割问题。通过提示空间条件化，提升模型性能，适用于医学图像分割。**

- **链接: [https://arxiv.org/pdf/2603.27250](https://arxiv.org/pdf/2603.27250)**

> **作者:** Huiyao Zhang; Jin Bai; Rui Guo; JianWen Tan; HongFei Wang; Ye Li
>
> **摘要:** Prompt-conditioned foundation segmenters have emerged as a dominant paradigm for image segmentation, where explicit spatial prompts (e.g., points, boxes, masks) guide mask decoding. However, many real-world deployments require fully automatic segmentation, creating a structural mismatch: the decoder expects prompts that are unavailable at inference. Existing adaptations typically modify intermediate features, inadvertently bypassing the model's native prompt interface and weakening prompt-conditioned decoding. We propose IP-SAM, which revisits adaptation from a prompt-space perspective through prompt-space conditioning. Specifically, a Self-Prompt Generator (SPG) distills image context into complementary intrinsic prompts that serve as coarse regional anchors. These cues are projected through SAM2's frozen prompt encoder, restoring prompt-guided decoding without external intervention. To suppress background-induced false positives, Prompt-Space Gating (PSG) leverages the intrinsic background prompt as an asymmetric suppressive constraint prior to decoding. Under a deterministic no-external-prompt protocol, IP-SAM achieves state-of-the-art performance across four camouflaged object detection benchmarks (e.g., MAE 0.017 on COD10K) with only 21.26M trainable parameters (optimizing SPG, PSG, and a task-specific mask decoder trained from scratch, alongside image-encoder LoRA while keeping the prompt encoder frozen). Furthermore, the proposed conditioning strategy generalizes beyond COD to medical polyp segmentation, where a model trained solely on Kvasir-SEG exhibits strong zero-shot transfer to both CVC-ClinicDB and ETIS.
>
---
#### [new 282] Dual-View Optical Flow for 4D Micro-Expression Recognition - A Multi-Stream Fusion Attention Approach
- **分类: cs.CV**

- **简介: 该论文属于微表情识别任务，旨在解决4D面部数据中微表情识别的挑战。通过双视角光流和多流融合注意力机制，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2603.26849](https://arxiv.org/pdf/2603.26849)**

> **作者:** Luu Tu Nguyen; Thi Bich Phuong Man; Vu Tram Anh Khuong; Thanh Ha Le; Thi Duyen Ngo
>
> **摘要:** Micro-expression recognition is vital for affective computing but remains challenging due to the extremely brief, low-intensity facial motions involved and the high-dimensional nature of 4D mesh data. To address these challenges, we introduce a dual-view optical flow approach that simplifies mesh processing by capturing each micro-expression sequence from two synchronized viewpoints and computing optical flow to represent motion. Our pipeline begins with view separation and sequence-wise face cropping to ensure spatial consistency, followed by automatic apex-frame detection based on peak motion intensity in both views. We decompose each sequence into onset-apex and apex-offset phases, extracting horizontal, vertical, and magnitude flow channels for each phase. These are fed into our Triple-Stream MicroAttNet, which employs a fusion attention module to adaptively weight modality-specific features and a squeeze-and-excitation block to enhance magnitude representations. Training uses focal loss to mitigate class imbalance and the Adam optimizer with early stopping. Evaluated on the multi-label 4DME dataset, comprising 24 subjects and five emotion categories, in the 4DMR IJCAI Workshop Challenge 2025, our method achieves a macro-UF1 score of 0.536, outperforming the official baseline by over 50\% and securing first place. Ablation studies confirm that both the fusion attention and SE components each contribute up to 3.6 points of UF1 gain. These results demonstrate that dual-view, phase-aware optical flow combined with multi-stream fusion yields a robust and interpretable solution for 4D micro-expression recognition.
>
---
#### [new 283] From Prediction to Diagnosis: Reasoning-Aware AI for Photovoltaic Defect Inspection
- **分类: cs.CV**

- **简介: 该论文属于光伏缺陷检测任务，旨在提升自动化检测的可解释性与诊断能力。提出REVL-PV框架，结合多模态图像进行推理感知学习，生成结构化诊断报告。**

- **链接: [https://arxiv.org/pdf/2603.26776](https://arxiv.org/pdf/2603.26776)**

> **作者:** Dev Mistry; Feng Qiu; Bo Chen; Feng Liu; Can Chen; Mohammad Shahidehpour; Ren Wang
>
> **备注:** 34 pages, 5 figures
>
> **摘要:** Reliable photovoltaic defect identification is essential for maintaining energy yield, ensuring warranty compliance, and enabling scalable inspection of rapidly expanding solar fleets. Although recent advances in computer vision have improved automated defect detection, most existing systems operate as opaque classifiers that provide limited diagnostic insight for high-stakes energy infrastructure. Here we introduce REVL-PV, a vision-language framework that embeds domain-specific diagnostic reasoning into multimodal learning across electroluminescence, thermal, and visible-light imagery. By requiring the model to link visual evidence to plausible defect mechanisms before classification, the framework produces structured diagnostic reports aligned with professional photovoltaic inspection practice. Evaluated on 1,927 real-world modules spanning eight defect categories, REVL-PV achieves 93\% classification accuracy while producing interpretable diagnostic rationales and maintaining strong robustness under realistic image corruptions. A blind concordance study with a certified solar inspection expert shows strong semantic alignment between model explanations and expert assessments across defect identification, root-cause attribution, and visual descriptions. These results demonstrate that reasoning-aware multimodal learning establishes a general paradigm for trustworthy AI-assisted inspection of photovoltaic energy infrastructure.
>
---
#### [new 284] INSID3: Training-Free In-Context Segmentation with DINOv3
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，解决无监督的上下文分割问题。提出INSID3方法，无需训练直接利用DINOv3特征实现多粒度分割，提升性能并减少参数。**

- **链接: [https://arxiv.org/pdf/2603.28480](https://arxiv.org/pdf/2603.28480)**

> **作者:** Claudia Cuttano; Gabriele Trivigno; Christoph Reich; Daniel Cremers; Carlo Masone; Stefan Roth
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** In-context segmentation (ICS) aims to segment arbitrary concepts, e.g., objects, parts, or personalized instances, given one annotated visual examples. Existing work relies on (i) fine-tuning vision foundation models (VFMs), which improves in-domain results but harms generalization, or (ii) combines multiple frozen VFMs, which preserves generalization but yields architectural complexity and fixed segmentation granularities. We revisit ICS from a minimalist perspective and ask: Can a single self-supervised backbone support both semantic matching and segmentation, without any supervision or auxiliary models? We show that scaled-up dense self-supervised features from DINOv3 exhibit strong spatial structure and semantic correspondence. We introduce INSID3, a training-free approach that segments concepts at varying granularities only from frozen DINOv3 features, given an in-context example. INSID3 achieves state-of-the-art results across one-shot semantic, part, and personalized segmentation, outperforming previous work by +7.5 % mIoU, while using 3x fewer parameters and without any mask or category-level supervision. Code is available at this https URL .
>
---
#### [new 285] Look, Compare and Draw: Differential Query Transformer for Automatic Oil Painting
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于自动油画生成任务，旨在解决重复笔触导致的审美下降问题。提出DQ-Transformer模型，通过差异分析和位置编码提升笔触的动态与细腻度。**

- **链接: [https://arxiv.org/pdf/2603.27720](https://arxiv.org/pdf/2603.27720)**

> **作者:** Lingyu Liu; Yaxiong Wang; Li Zhu; Lizi Liao; Zhedong Zheng
>
> **备注:** this https URL
>
> **摘要:** This work introduces a new approach to automatic oil painting that emphasizes the creation of dynamic and expressive brushstrokes. A pivotal challenge lies in mitigating the duplicate and common-place strokes, which often lead to less aesthetic outcomes. Inspired by the human painting process, \ie, observing, comparing, and drawing, we incorporate differential image analysis into a neural oil painting model, allowing the model to effectively concentrate on the incremental impact of successive brushstrokes. To operationalize this concept, we propose the Differential Query Transformer (DQ-Transformer), a new architecture that leverages differentially derived image representations enriched with positional encoding to guide the stroke prediction process. This integration enables the model to maintain heightened sensitivity to local details, resulting in more refined and nuanced stroke generation. Furthermore, we incorporate adversarial training into our framework, enhancing the accuracy of stroke prediction and thereby improving the overall realism and fidelity of the synthesized paintings. Extensive qualitative evaluations, complemented by a controlled user study, validate that our DQ-Transformer surpasses existing methods in both visual realism and artistic authenticity, typically achieving these results with fewer strokes. The stroke-by-stroke painting animations are available on our project website.
>
---
#### [new 286] Computer Vision with a Superpixelation Camera
- **分类: cs.CV**

- **简介: 该论文提出一种新型相机SuperCam，通过实时超像素分割减少数据冗余，解决资源受限设备中计算机视觉处理效率低的问题。**

- **链接: [https://arxiv.org/pdf/2603.26900](https://arxiv.org/pdf/2603.26900)**

> **作者:** Sasidharan Mahalingam; Rachel Brown; Atul Ingle
>
> **摘要:** Conventional cameras generate a lot of data that can be challenging to process in resource-constrained applications. Usually, cameras generate data streams on the order of the number of pixels in the image. However, most of this captured data is redundant for many downstream computer vision algorithms. We propose a novel camera design, which we call SuperCam, that adaptively processes captured data by performing superpixel segmentation on the fly. We show that SuperCam performs better than current state-of-the-art superpixel algorithms under memory-constrained situations. We also compare how well SuperCam performs when the compressed data is used for downstream computer vision tasks. Our results demonstrate that the proposed design provides superior output for image segmentation, object detection, and monocular depth estimation in situations where the available memory on the camera is limited. We posit that superpixel segmentation will play a crucial role as more computer vision inference models are deployed in edge devices. SuperCam would allow computer vision engineers to design more efficient systems for these applications.
>
---
#### [new 287] E-TIDE: Fast, Structure-Preserving Motion Forecasting from Event Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出E-TIDE，解决事件流中的运动预测任务，通过轻量结构实现高效、低功耗的未来事件表示预测。**

- **链接: [https://arxiv.org/pdf/2603.27757](https://arxiv.org/pdf/2603.27757)**

> **作者:** Biswadeep Sen; Benoit R. Cottereau; Nicolas Cuperlier; Terence Sim
>
> **摘要:** Event-based cameras capture visual information as asynchronous streams of per-pixel brightness changes, generating sparse, temporally precise data. Compared to conventional frame-based sensors, they offer significant advantages in capturing high-speed dynamics while consuming substantially less power. Predicting future event representations from past observations is an important problem, enabling downstream tasks such as future semantic segmentation or object tracking without requiring access to future sensor measurements. While recent state-of-the-art approaches achieve strong performance, they often rely on computationally heavy backbones and, in some cases, large-scale pretraining, limiting their applicability in resource-constrained scenarios. In this work, we introduce E-TIDE, a lightweight, end-to-end trainable architecture for event-tensor prediction that is designed to operate efficiently without large-scale pretraining. Our approach employs the TIDE module (Temporal Interaction for Dynamic Events), motivated by efficient spatiotemporal interaction design for sparse event tensors, to capture temporal dependencies via large-kernel mixing and activity-aware gating while maintaining low computational complexity. Experiments on standard event-based datasets demonstrate that our method achieves competitive performance with significantly reduced model size and training requirements, making it well-suited for real-time deployment under tight latency and memory budgets.
>
---
#### [new 288] Beyond Mortality: Advancements in Post-Mortem Iris Recognition through Data Collection and Computer-Aided Forensic Examination
- **分类: cs.CV**

- **简介: 该论文属于后死亡虹膜识别任务，旨在解决数据不足与识别准确性问题。通过构建大规模数据集、评估现有方法并提出检测模型，提升后死亡虹膜识别的可靠性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.26976](https://arxiv.org/pdf/2603.26976)**

> **作者:** Rasel Ahmed Bhuiyan; Parisa Farmanifard; Renu Sharma; Andrey Kuehlkamp; Aidan Boyd; Patrick J Flynn; Kevin W Bowyer; Arun Ross; Dennis Chute; Adam Czajka
>
> **摘要:** Post-mortem iris recognition brings both hope to the forensic community (a short-term but accurate and fast means of verifying identity) as well as concerns to society (its potential illicit use in post-mortem impersonation). These hopes and concerns have grown along with the volume of research in post-mortem iris recognition. Barriers to further progress in post-mortem iris recognition include the difficult nature of data collection, and the resulting small number of approaches designed specifically for comparing iris images of deceased subjects. This paper makes several unique contributions to mitigate these barriers. First, we have collected and we offer a new dataset of NIR (compliant with ISO/IEC 19794-6 where possible) and visible-light iris images collected after demise from 259 subjects, with the largest PMI (post-mortem interval) being 1,674 hours. For one subject, the data has been collected before and after death, the first such case ever published. Second, the collected dataset was combined with publicly-available post-mortem samples to assess the current state of the art in automatic forensic iris recognition with five iris recognition methods and data originating from 338 deceased subjects. These experiments include analyses of how selected demographic factors influence recognition performance. Thirdly, this study implements a model for detecting post-mortem iris images, which can be considered as presentation attacks. Finally, we offer an open-source forensic tool integrating three post-mortem iris recognition methods with explainability elements added to make the comparison process more human-interpretable.
>
---
#### [new 289] Mind the Shape Gap: A Benchmark and Baseline for Deformation-Aware 6D Pose Estimation of Agricultural Produce
- **分类: cs.CV**

- **简介: 该论文属于农业机器人6D位姿估计任务，解决因农产品形变和类内形状差异导致的定位不准问题。提出PEAR基准和SEED方法，实现单图像下位姿与形变联合预测。**

- **链接: [https://arxiv.org/pdf/2603.27429](https://arxiv.org/pdf/2603.27429)**

> **作者:** Nikolas Chatzis; Angeliki Tsinouka; Katerina Papadimitriou; Niki Efthymiou; Marios Glytsos; George Retsinas; Paris Oikonomou; Gerasimos Potamianos; Petros Maragos; Panagiotis Paraskevas Filntisis
>
> **摘要:** Accurate 6D pose estimation for robotic harvesting is fundamentally hindered by the biological deformability and high intra-class shape variability of agricultural produce. Instance-level methods fail in this setting, as obtaining exact 3D models for every unique piece of produce is practically infeasible, while category-level approaches that rely on a fixed template suffer significant accuracy degradation when the prior deviates from the true instance geometry. To bridge such lack of robustness to deformation, we introduce PEAR (Pose and dEformation of Agricultural pRoduce), the first benchmark providing joint 6D pose and per-instance 3D deformation ground truth across 8 produce categories, acquired via a robotic manipulator for high annotation accuracy. Using PEAR, we show that state-of-the-art methods suffer up to 6x performance degradation when faced with the inherent geometric deviations of real-world produce. Motivated by this finding, we propose SEED (Simultaneous Estimation of posE and Deformation), a unified RGB-only framework that jointly predicts 6D pose and explicit lattice deformations from a single image across multiple produce categories. Trained entirely on synthetic data with generative texture augmentation applied at the UV level, SEED outperforms MegaPose on 6 out of 8 categories under identical RGB-only conditions, demonstrating that explicit shape modeling is a critical step toward reliable pose estimation in agricultural robotics.
>
---
#### [new 290] FeDMRA: Federated Incremental Learning with Dynamic Memory Replay Allocation
- **分类: cs.LG; cs.AI; cs.CV; cs.DC; stat.ML**

- **简介: 该论文属于联邦持续学习任务，解决非独立同分布数据下的模型遗忘问题。提出动态记忆分配策略，提升模型性能与公平性。**

- **链接: [https://arxiv.org/pdf/2603.28455](https://arxiv.org/pdf/2603.28455)**

> **作者:** Tiantian Wang; Xiang Xiang; Simon S. Du
>
> **摘要:** In federated healthcare systems, Federated Class-Incremental Learning (FCIL) has emerged as a key paradigm, enabling continuous adaptive model learning among distributed clients while safeguarding data privacy. However, in practical applications, data across agent nodes within the distributed framework often exhibits non-independent and identically distributed (non-IID) characteristics, rendering traditional continual learning methods inapplicable. To address these challenges, this paper covers more comprehensive incremental task scenarios and proposes a dynamic memory allocation strategy for exemplar storage based on the data replay mechanism. This strategy fully taps into the inherent potential of data heterogeneity, while taking into account the performance fairness of all participating clients, thereby establishing a balanced and adaptive solution to mitigate catastrophic forgetting. Unlike the fixed allocation of client exemplar memory, the proposed scheme emphasizes the rational allocation of limited storage resources among clients to improve model performance. Furthermore, extensive experiments are conducted on three medical image datasets, and the results demonstrate significant performance improvements compared to existing baseline models.
>
---
#### [new 291] Stepwise Credit Assignment for GRPO on Flow-Matching Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于强化学习任务，解决流模型中信用分配不合理的问题。通过分步信用分配提升样本效率和收敛速度。**

- **链接: [https://arxiv.org/pdf/2603.28718](https://arxiv.org/pdf/2603.28718)**

> **作者:** Yash Savani; Branislav Kveton; Yuchen Liu; Yilin Wang; Jing Shi; Subhojyoti Mukherjee; Nikos Vlassis; Krishna Kumar Singh
>
> **备注:** Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026 Project page: this https URL
>
> **摘要:** Flow-GRPO successfully applies reinforcement learning to flow models, but uses uniform credit assignment across all steps. This ignores the temporal structure of diffusion generation: early steps determine composition and content (low-frequency structure), while late steps resolve details and textures (high-frequency details). Moreover, assigning uniform credit based solely on the final image can inadvertently reward suboptimal intermediate steps, especially when errors are corrected later in the diffusion trajectory. We propose Stepwise-Flow-GRPO, which assigns credit based on each step's reward improvement. By leveraging Tweedie's formula to obtain intermediate reward estimates and introducing gain-based advantages, our method achieves superior sample efficiency and faster convergence. We also introduce a DDIM-inspired SDE that improves reward quality while preserving stochasticity for policy gradients.
>
---
#### [new 292] SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出SOLE-R1，用于机器人强化学习的视频-语言推理模型，解决无监督任务学习问题。通过视频和自然语言目标生成密集奖励信号，实现零样本在线学习。**

- **链接: [https://arxiv.org/pdf/2603.28730](https://arxiv.org/pdf/2603.28730)**

> **作者:** Philip Schroeder; Thomas Weng; Karl Schmeckpeper; Eric Rosen; Stephen Hart; Ondrej Biza
>
> **摘要:** Vision-language models (VLMs) have shown impressive capabilities across diverse tasks, motivating efforts to leverage these models to supervise robot learning. However, when used as evaluators in reinforcement learning (RL), today's strongest models often fail under partial observability and distribution shift, enabling policies to exploit perceptual errors rather than solve the task. To address this limitation, we introduce SOLE-R1 (Self-Observing LEarner), a video-language reasoning model explicitly designed to serve as the sole reward signal for online RL. Given only raw video observations and a natural-language goal, SOLE-R1 performs per-timestep spatiotemporal chain-of-thought (CoT) reasoning and produces dense estimates of task progress that can be used directly as rewards. To train SOLE-R1, we develop a large-scale video trajectory and reasoning synthesis pipeline that generates temporally grounded CoT traces aligned with continuous progress supervision. This data is combined with foundational spatial and multi-frame temporal reasoning, and used to train the model with a hybrid framework that couples supervised fine-tuning with RL from verifiable rewards. Across four different simulation environments and a real-robot setting, SOLE-R1 enables zero-shot online RL from random initialization: robots learn previously unseen manipulation tasks without ground-truth rewards, success indicators, demonstrations, or task-specific tuning. SOLE-R1 succeeds on 24 unseen tasks and substantially outperforms strong vision-language rewarders, including GPT-5 and Gemini-3-Pro, while exhibiting markedly greater robustness to reward hacking.
>
---
#### [new 293] EMPD: An Event-based Multimodal Physiological Dataset for Remote Pulse Wave Detection
- **分类: eess.SP; cs.CV; cs.LG**

- **简介: 该论文提出EMPD数据集，用于解决非接触式生理监测中的运动伪影和时间分辨率问题。通过事件相机等多模态设备采集高精度生理信号。**

- **链接: [https://arxiv.org/pdf/2603.26699](https://arxiv.org/pdf/2603.26699)**

> **作者:** Qian Feng; Pengfei Li; Rongshan Gao; Jiale Xu; Rui Gong; Yidi Li
>
> **备注:** 12 pages, 4 figures, 2 tables
>
> **摘要:** Remote photoplethysmography (rPPG) based on traditional frame-based cameras often struggles with motion artifacts and limited temporal resolution. To address these limitations, we introduce EMPD (Event-based Multimodal Physiological Dataset), the first benchmark dataset specifically designed for non-contact physiological sensing via event cameras. The dataset leverages a laser-assisted acquisition system where a high-coherence laser modulates subtle skin vibrations from the radial artery into significant signals detectable by a neuromorphic sensor. The hardware platform integrates a high-resolution event camera to capture micro-motions and intensity transients, an industrial RGB camera to provide traditional rPPG benchmarks, and a clinical-grade pulse oximeter to record ground truth PPG waveforms. EMPD contains 193 valid records collected from 83 subjects, covering a wide heart rate range (40-110 BPM) under both resting and post-exercise conditions. By providing precisely synchronized multimodal data with microsecond-level temporal precision, EMPD serves as a crucial resource for developing robust algorithms in the field of neuromorphic physiological monitoring. The dataset is publicly available at: this https URL
>
---
#### [new 294] Beyond Benchmarks: A Framework for Post Deployment Validation of CT Lung Nodule Detection AI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像AI的后部署验证任务，旨在评估CT肺结节检测模型在不同扫描参数下的性能。通过模拟剂量和层厚变化，分析模型敏感性，提出可复现的验证框架。**

- **链接: [https://arxiv.org/pdf/2603.26785](https://arxiv.org/pdf/2603.26785)**

> **作者:** Daniel Soliman
>
> **摘要:** Background: Artificial intelligence (AI) assisted lung nodule detection systems are increasingly deployed in clinical settings without site-specific validation. Performance reported under benchmark conditions may not reflect real-world behavior when acquisition parameters differ from training data. Purpose: To propose and demonstrate a physics-guided framework for evaluating the sensitivity of a deployed lung nodule detection model to systematic variation in CT acquisition parameters. Methods: Twenty-one cases from the publicly available LIDC-IDRI dataset were evaluated using a MONAI RetinaNet model pretrained on LUNA16 (fold 0, no fine-tuning). Five imaging conditions were tested: baseline, 25% dose reduction, 50% dose reduction, 3 mm slice thickness, and 5 mm slice thickness. Dose reduction was simulated via image-domain Gaussian noise; slice thickness via moving average along the z-axis. Detection sensitivity was computed at a confidence threshold of 0.5 with a 15 mm matching criterion. Results: Baseline sensitivity was 45.2% (57/126 consensus nodules). Dose reduction produced slight degradation: 41.3% at 25% dose and 42.1% at 50% dose. The 5 mm slice thickness condition produced a marked drop to 26.2% - a 19 percentage point reduction representing a 42% relative decrease from baseline. This finding was consistent across confidence thresholds from 0.1 to 0.9. Per-case analysis revealed heterogeneous performance including two cases with complete detection failure at baseline. Conclusion: Slice thickness represents a more fundamental constraint on AI detection performance than image noise under the conditions tested. The proposed framework is reproducible, requires no proprietary scanner data, and is designed to serve as the basis for ongoing post-deployment QA in resource-constrained environment.
>
---
#### [new 295] Grounding Social Perception in Intuitive Physics
- **分类: q-bio.NC; cs.AI; cs.CV**

- **简介: 该论文属于社会感知任务，旨在理解物理世界中的社交行为。通过构建PHASE数据集和提出SIMPLE模型，解决如何从物理交互中推断代理目标与关系的问题。**

- **链接: [https://arxiv.org/pdf/2603.27410](https://arxiv.org/pdf/2603.27410)**

> **作者:** Lance Ying; Aydan Y. Huang; Aviv Netanyahu; Andrei Barbu; Boris Katz; Joshua B. Tenenbaum; Tianmin Shu
>
> **备注:** 26 pages, 11 figures
>
> **摘要:** People infer rich social information from others' actions. These inferences are often constrained by the physical world: what agents can do, what obstacles permit, and how the physical actions of agents causally change an environment and other agents' mental states and behavior. We propose that such rich social perception is more than visual pattern matching, but rather a reasoning process grounded in an integration of intuitive psychology with intuitive physics. To test this hypothesis, we introduced PHASE (PHysically grounded Abstract Social Events), a large dataset of procedurally generated animations, depicting physically simulated two-agent interactions on a 2D surface. Each animation follows the style of the Heider and Simmel movie, with systematic variation in environment geometry, object dynamics, agent capacities, goals, and relationships (friendly/adversarial/neutral). We then present a computational model, SIMPLE, a physics-grounded Bayesian inverse planning model that integrates planning, probabilistic planning, and physics simulation to infer agents' goals and relations from their trajectories. Our experimental results showed that SIMPLE achieved high accuracy and agreement with human judgments across diverse scenarios, while feedforward baseline models -- including strong vision-language models -- and physics-agnostic inverse planning failed to achieve human-level performance and did not align with human judgments. These results suggest that our model provides a computational account for how people understand physically grounded social scenes by inverting a generative model of physics and agents.
>
---
#### [new 296] ImagenWorld: Stress-Testing Image Generation Models with Explainable Human Evaluation on Open-ended Real-World Tasks
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出ImagenWorld，一个涵盖6类任务和6个领域的图像生成基准，解决现有评估不足的问题。通过人类标注和可解释评价，提升模型评估的准确性和诊断能力。**

- **链接: [https://arxiv.org/pdf/2603.27862](https://arxiv.org/pdf/2603.27862)**

> **作者:** Samin Mahdizadeh Sani; Max Ku; Nima Jamali; Matina Mahdizadeh Sani; Paria Khoshtab; Wei-Chieh Sun; Parnian Fazel; Zhi Rui Tam; Thomas Chong; Edisy Kin Wai Chan; Donald Wai Tong Tsang; Chiao-Wei Hsu; Ting Wai Lam; Ho Yin Sam Ng; Chiafeng Chu; Chak-Wing Mak; Keming Wu; Hiu Tung Wong; Yik Chun Ho; Chi Ruan; Zhuofeng Li; I-Sheng Fang; Shih-Ying Yeh; Ho Kei Cheng; Ping Nie; Wenhu Chen
>
> **备注:** Published in ICLR 2026
>
> **摘要:** Advances in diffusion, autoregressive, and hybrid models have enabled high-quality image synthesis for tasks such as text-to-image, editing, and reference-guided composition. Yet, existing benchmarks remain limited, either focus on isolated tasks, cover only narrow domains, or provide opaque scores without explaining failure modes. We introduce \textbf{ImagenWorld}, a benchmark of 3.6K condition sets spanning six core tasks (generation and editing, with single or multiple references) and six topical domains (artworks, photorealistic images, information graphics, textual graphics, computer graphics, and screenshots). The benchmark is supported by 20K fine-grained human annotations and an explainable evaluation schema that tags localized object-level and segment-level errors, complementing automated VLM-based metrics. Our large-scale evaluation of 14 models yields several insights: (1) models typically struggle more in editing tasks than in generation tasks, especially in local edits. (2) models excel in artistic and photorealistic settings but struggle with symbolic and text-heavy domains such as screenshots and information graphics. (3) closed-source systems lead overall, while targeted data curation (e.g., Qwen-Image) narrows the gap in text-heavy cases. (4) modern VLM-based metrics achieve Kendall accuracies up to 0.79, approximating human ranking, but fall short of fine-grained, explainable error attribution. ImagenWorld provides both a rigorous benchmark and a diagnostic tool to advance robust image generation.
>
---
#### [new 297] Engineering Mythology: A Digital-Physical Framework for Culturally-Inspired Public Art
- **分类: cs.GR; cs.CV; cs.CY; cs.RO**

- **简介: 该论文属于跨学科艺术与工程任务，旨在融合文化传统与现代技术。通过数字-物理流程，实现文化灵感公共艺术的创作与协作，解决文化传承与技术创新结合的问题。**

- **链接: [https://arxiv.org/pdf/2603.27801](https://arxiv.org/pdf/2603.27801)**

> **作者:** Jnaneshwar Das; Christopher Filkins; Rajesh Moharana; Ekadashi Barik; Bishweshwar Das; David Ayers; Christopher Skiba; Rodney Staggers Jr; Mark Dill; Swig Miller; Daniel Tulberg; Patrick Smith; Seth Brink; Kyle Breen; Harish Anand; Ramon Arrowsmith
>
> **备注:** 19 pages, 28 figures, 4 tables
>
> **摘要:** Navagunjara Reborn: The Phoenix of Odisha was built for Burning Man 2025 as both a sculpture and an experiment-a fusion of myth, craft, and computation. This paper describes the digital-physical workflow developed for the project: a pipeline that linked digital sculpting, distributed fabrication by artisans in Odisha (India), modular structural optimization in the U.S., iterative feedback through photogrammetry and digital twins, and finally, one-shot full assembly at the art site in Black Rock Desert, Nevada. The desert installation tested not just materials, but also systems of collaboration: between artisans and engineers, between myth and technology, between cultural specificity and global experimentation. We share the lessons learned in design, fabrication, and deployment and offer a framework for future interdisciplinary projects at the intersection of cultural heritage, STEAM education, and public art. In retrospect, this workflow can be read as a convergence of many knowledge systems-artisan practice, structural engineering, mythic narrative, and environmental constraint-rather than as execution of a single fixed blueprint.
>
---
#### [new 298] Segmenting Superbubbles in a Simulated Multiphase Interstellar Medium using Computer Vision
- **分类: astro-ph.GA; cs.CV**

- **简介: 该论文属于3D分割任务，旨在精确识别和跟踪超泡结构。通过计算机视觉方法，解决多相星际介质中复杂结构的分析问题。**

- **链接: [https://arxiv.org/pdf/2603.27741](https://arxiv.org/pdf/2603.27741)**

> **作者:** Jing-Wen Chen; Alex S. Hill; Anna Ordog; Rebecca A. Booth; Mohamed S. Shehata
>
> **摘要:** We developed a computer vision-based methodology to achieve precise 3D segmentation and tracking of superbubbles within magnetohydrodynamic simulations of the supernova-driven interstellar medium. Leveraging advanced 3D transformer models, our approach effectively captures the complex morphology and dynamic evolution of these astrophysical structures. To demonstrate the technique, we specifically focused on a superbubble exhibiting interesting interactions with its surrounding medium, driven by a series of successive supernova explosions. Our model successfully generated detailed 3D segmentation masks, enabling us to visualize and analyze the bubble's structural evolution over time. The results reveal insights into the superbubble's growth patterns, energy retention, and interactions with surrounding interstellar matter. This interdisciplinary approach not only enhances our understanding of superbubble dynamics but also offers a robust framework for investigating other complex phenomena in the cosmos.
>
---
#### [new 299] SpatialAnt: Autonomous Zero-Shot Robot Navigation via Active Scene Reconstruction and Visual Anticipation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉导航任务，解决零样本机器人导航中因自建场景不完整导致的性能下降问题。提出SpatialAnt框架，结合物理对齐和视觉前瞻机制，提升导航鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.26837](https://arxiv.org/pdf/2603.26837)**

> **作者:** Jiwen Zhang; Xiangyu Shi; Siyuan Wang; Zerui Li; Zhongyu Wei; Qi Wu
>
> **备注:** 10 pages, 4 figures, 5 tables. Homepage: this https URL
>
> **摘要:** Vision-and-Language Navigation (VLN) has recently benefited from Multimodal Large Language Models (MLLMs), enabling zero-shot navigation. While recent exploration-based zero-shot methods have shown promising results by leveraging global scene priors, they rely on high-quality human-crafted scene reconstructions, which are impractical for real-world robot deployment. When encountering an unseen environment, a robot should build its own priors through pre-exploration. However, these self-built reconstructions are inevitably incomplete and noisy, which severely degrade methods that depend on high-quality scene reconstructions. To address these issues, we propose SpatialAnt, a zero-shot navigation framework designed to bridge the gap between imperfect self-reconstructions and robust execution. SpatialAnt introduces a physical grounding strategy to recover the absolute metric scale for monocular-based reconstructions. Furthermore, rather than treating the noisy self-reconstructed scenes as absolute spatial references, we propose a novel visual anticipation mechanism. This mechanism leverages the noisy point clouds to render future observations, enabling the agent to perform counterfactual reasoning and prune paths that contradict human instructions. Extensive experiments in both simulated and real-world environments demonstrate that SpatialAnt significantly outperforms existing zero-shot methods. We achieve a 66% Success Rate (SR) on R2R-CE and 50.8% SR on RxR-CE benchmarks. Physical deployment on a Hello Robot further confirms the efficiency and efficacy of our framework, achieving a 52% SR in challenging real-world settings.
>
---
#### [new 300] VAN-AD: Visual Masked Autoencoder with Normalizing Flow For Time Series Anomaly Detection
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于时间序列异常检测任务，旨在解决现有方法泛化能力差的问题。通过改进视觉掩码自编码器，提出VAN-AD框架提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.26842](https://arxiv.org/pdf/2603.26842)**

> **作者:** PengYu Chen; Shang Wan; Xiaohou Shi; Yuan Chang; Yan Sun; Sajal K. Das
>
> **备注:** 13 pages, 20 figures
>
> **摘要:** Time series anomaly detection (TSAD) is essential for maintaining the reliability and security of IoT-enabled service systems. Existing methods require training one specific model for each dataset, which exhibits limited generalization capability across different target datasets, hindering anomaly detection performance in various scenarios with scarce training data. To address this limitation, foundation models have emerged as a promising direction. However, existing approaches either repurpose large language models (LLMs) or construct largescale time series datasets to develop general anomaly detection foundation models, and still face challenges caused by severe cross-modal gaps or in-domain heterogeneity. In this paper, we investigate the applicability of large-scale vision models to TSAD. Specifically, we adapt a visual Masked Autoencoder (MAE) pretrained on ImageNet to the TSAD task. However, directly transferring MAE to TSAD introduces two key challenges: overgeneralization and limited local perception. To address these challenges, we propose VAN-AD, a novel MAE-based framework for TSAD. To alleviate the over-generalization issue, we design an Adaptive Distribution Mapping Module (ADMM), which maps the reconstruction results before and after MAE into a unified statistical space to amplify discrepancies caused by abnormal patterns. To overcome the limitation of local perception, we further develop a Normalizing Flow Module (NFM), which combines MAE with normalizing flow to estimate the probability density of the current window under the global distribution. Extensive experiments on nine real-world datasets demonstrate that VAN-AD consistently outperforms existing state-of-the-art methods across multiple evaluation this http URL make our code and datasets available at this https URL.
>
---
#### [new 301] MRI-to-CT synthesis using drifting models
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于MRI到CT图像合成任务，旨在通过MRI生成高质量CT图像，避免辐射。工作对比多种模型，提出漂移模型，提升图像质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.28498](https://arxiv.org/pdf/2603.28498)**

> **作者:** Qing Lyu; Jianxu Wang; Jeremy Hudson; Ge Wang; Chirstopher T. Whitlow
>
> **摘要:** Accurate MRI-to-CT synthesis could enable MR-only pelvic workflows by providing CT-like images with bone details while avoiding additional ionizing radiation. In this work, we investigate recently proposed drifting models for synthesizing pelvis CT images from MRI and benchmark them against convolutional neural networks (UNet, VAE), a generative adversarial network (WGAN-GP), a physics-inspired probabilistic model (PPFM), and diffusion-based methods (FastDDPM, DDIM, DDPM). Experiments are performed on two complementary datasets: Gold Atlas Male Pelvis and the SynthRAD2023 pelvis subset. Image fidelity and structural consistency are evaluated with SSIM, PSNR, and RMSE, complemented by qualitative assessment of anatomically critical regions such as cortical bone and pelvic soft-tissue interfaces. Across both datasets, the proposed drifting model achieves high SSIM and PSNR and low RMSE, surpassing strong diffusion baselines and conventional CNN-, VAE-, GAN-, and PPFM-based methods. Visual inspection shows sharper cortical bone edges, improved depiction of sacral and femoral head geometry, and reduced artifacts or over-smoothing, particularly at bone-air-soft tissue boundaries. Moreover, the drifting model attains these gains with one-step inference and inference times on the order of milliseconds, yielding a more favorable accuracy-efficiency trade-off than iterative diffusion sampling while remaining competitive in image quality. These findings suggest that drifting models are a promising direction for fast, high-quality pelvic synthetic CT generation from MRI and warrant further investigation for downstream applications such as MRI-only radiotherapy planning and PET/MR attenuation correction.
>
---
#### [new 302] ReMemNav: A Rethinking and Memory-Augmented Framework for Zero-Shot Object Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于零样本目标导航任务，旨在解决未知环境中定位未见目标的问题。提出ReMemNav框架，结合视觉语言模型与记忆机制，提升导航成功率和效率。**

- **链接: [https://arxiv.org/pdf/2603.26788](https://arxiv.org/pdf/2603.26788)**

> **作者:** Feng Wu; Wei Zuo; Wenliang Yang; Jun Xiao; Yang Liu; Xinhua Zeng
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Zero-shot object navigation requires agents to locate unseen target objects in unfamiliar environments without prior maps or task-specific training which remains a significant challenge. Although recent advancements in vision-language models(VLMs) provide promising commonsense reasoning capabilities for this task, these models still suffer from spatial hallucinations, local exploration deadlocks, and a disconnect between high-level semantic intent and low-level control. In this regard, we propose a novel hierarchical navigation framework named ReMemNav, which seamlessly integrates panoramic semantic priors and episodic memory with VLMs. We introduce the Recognize Anything Model to anchor the spatial reasoning process of the VLM. We also design an adaptive dual-modal rethinking mechanism based on an episodic semantic buffer queue. The proposed mechanism actively verifies target visibility and corrects decisions using historical memory to prevent deadlocks. For low-level action execution, ReMemNav extracts a sequence of feasible actions using depth masks, allowing the VLM to select the optimal action for mapping into actual spatial movement. Extensive evaluations on HM3D and MP3D demonstrate that ReMemNav outperforms existing training-free zero-shot baselines in both success rate and exploration efficiency. Specifically, we achieve significant absolute performance improvements, with SR and SPL increasing by 1.7% and 7.0% on HM3D v0.1, 18.2% and 11.1% on HM3D v0.2, and 8.7% and 7.9% on MP3D.
>
---
#### [new 303] Uncertainty-Aware Mapping from 3D Keypoints to Anatomical Landmarks for Markerless Biomechanics
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于生物力学任务，解决3D关键点到解剖标志物映射中的不确定性问题，通过建模预测不确定性实现质量控制。**

- **链接: [https://arxiv.org/pdf/2603.26844](https://arxiv.org/pdf/2603.26844)**

> **作者:** Cesare Davide Pace; Alessandro Marco De Nunzio; Claudio De Stefano; Francesco Fontanella; Mario Molinara
>
> **备注:** 7 pages, 1 figure, submitted to Patter Recognition Letters, uncertainty-aware framework for 3D keypoint-to-landmark mapping in markerless biomechanics
>
> **摘要:** Markerless biomechanics increasingly relies on 3D skeletal keypoints extracted from video, yet downstream biomechanical mappings typically treat these estimates as deterministic, providing no principled mechanism for frame-wise quality control. In this work, we investigate predictive uncertainty as a quantitative measure of confidence for mapping 3D pose keypoints to 3D anatomical landmarks, a critical step preceding inverse kinematics and musculoskeletal analysis. Within a temporal learning framework, we model both uncertainty arising from observation noise and uncertainty related to model limitations. Using synchronized motion capture ground truth on AMASS, we evaluate uncertainty at frame and joint level through error--uncertainty rank correlation, risk--coverage analysis, and catastrophic outlier detection. Across experiments, uncertainty estimates, particularly those associated with model uncertainty, exhibit a strong monotonic association with landmark error (Spearman $\rho \approx 0.63$), enabling selective retention of reliable frames (error reduced to $\approx 16.8$ mm at 10% coverage) and accurate detection of severe failures (ROC-AUC $\approx 0.92$ for errors $>50$ mm). Reliability ranking remains stable under controlled input degradation, including Gaussian noise and simulated missing joints. In contrast, uncertainty attributable to observation noise provides limited additional benefit in this setting, suggesting that dominant failures in keypoint-to-landmark mapping are driven primarily by model uncertainty. Our results establish predictive uncertainty as a practical, frame-wise tool for automatic quality control in markerless biomechanical pipelines.
>
---
#### [new 304] Contextual Graph Representations for Task-Driven 3D Perception and Planning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究机器人任务规划与3D场景图的结合，旨在解决任务规划中状态空间过大的问题。通过构建基准和使用图神经网络优化表示。**

- **链接: [https://arxiv.org/pdf/2603.26685](https://arxiv.org/pdf/2603.26685)**

> **作者:** Christopher Agia
>
> **备注:** University of Toronto Undergraduate Thesis, 2021. 85 pages, 24 figures
>
> **摘要:** Recent advances in computer vision facilitate fully automatic extraction of object-centric relational representations from visual-inertial data. These state representations, dubbed 3D scene graphs, are a hierarchical decomposition of real-world scenes with a dense multiplex graph structure. While 3D scene graphs claim to promote efficient task planning for robot systems, they contain numerous objects and relations when only small subsets are required for a given task. This magnifies the state space that task planners must operate over and prohibits deployment in resource constrained settings. This thesis tests the suitability of existing embodied AI environments for research at the intersection of robot task planning and 3D scene graphs and constructs a benchmark for empirical comparison of state-of-the-art classical planners. Furthermore, we explore the use of graph neural networks to harness invariances in the relational structure of planning domains and learn representations that afford faster planning.
>
---
#### [new 305] External Benchmarking of Lung Ultrasound Models for Pneumothorax-Related Signs: A Manifest-Based Multi-Source Study
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于肺部超声AI评估任务，旨在解决二分类模型无法准确识别 pneumothorax 相关征象的问题，通过构建多源基准测试模型性能。**

- **链接: [https://arxiv.org/pdf/2603.26832](https://arxiv.org/pdf/2603.26832)**

> **作者:** Takehiro Ishikawa
>
> **摘要:** Background and Aims: Reproducible external benchmarks for pneumothorax-related lung ultrasound (LUS) AI are scarce, and binary lung-sliding classification may obscure clinically important signs. We therefore developed a manifest-based external benchmark and used it to test both cross-domain generalization and task validity. Methods: We curated 280 clips from 190 publicly accessible LUS source videos and released a reconstruction manifest containing URLs, timestamps, crop coordinates, labels, and probe shape. Labels were normal lung sliding, absent lung sliding, lung point, and lung pulse. A previously published single-site binary classifier was evaluated on this benchmark; challenge-state analysis examined lung point and lung pulse using the predicted probability of absent sliding, P(absent). Results: The single-site comparator achieved ROC-AUC 0.9625 in-domain but 0.7050 on the heterogeneous external benchmark; restricting external evaluation to linear clips still yielded ROC-AUC 0.7212. In challenge-state analysis, mean P(absent) ranked absent (0.504) > lung point (0.313) > normal (0.186) > lung pulse (0.143). Lung pulse differed from absent clips (p=0.000470) but not from normal clips (p=0.813), indicating that the binary model treated pulse as normal-like despite absent sliding. Lung point differed from both absent (p=0.000468) and normal (p=0.000026), supporting its interpretation as an intermediate ambiguity state rather than a clean binary class. Conclusion: A manifest-based, multi-source benchmark can support reproducible external evaluation without redistributing source videos. Binary lung-sliding classification is an incomplete proxy for pneumothorax reasoning because it obscures blind-spot and ambiguity states such as lung pulse and lung point.
>
---
#### [new 306] TokenDance: Token-to-Token Music-to-Dance Generation with Bidirectional Mamba
- **分类: cs.AI; cs.CV; cs.SD**

- **简介: 该论文属于音乐到舞蹈生成任务，旨在解决现有模型泛化能力差、生成舞蹈单一的问题。通过双模态分词和双向Mamba架构，提升生成质量和效率。**

- **链接: [https://arxiv.org/pdf/2603.27314](https://arxiv.org/pdf/2603.27314)**

> **作者:** Ziyue Yang; Kaixing Yang; Xulong Tang
>
> **备注:** CVPR2026 Workshop on HuMoGen
>
> **摘要:** Music-to-dance generation has broad applications in virtual reality, dance education, and digital character animation. However, the limited coverage of existing 3D dance datasets confines current models to a narrow subset of music styles and choreographic patterns, resulting in poor generalization to real-world music. Consequently, generated dances often become overly simplistic and repetitive, substantially degrading expressiveness and realism. To tackle this problem, we present TokenDance, a two-stage music-to-dance generation framework that explicitly addresses this limitation through dual-modality tokenization and efficient token-level generation. In the first stage, we discretize both dance and music using Finite Scalar Quantization, where dance motions are factorized into upper and lower-body components with kinematic-dynamic constraints, and music is decomposed into semantic and acoustic features with dedicated codebooks to capture choreography-specific structures. In the second stage, we introduce a Local-Global-Local token-to-token generator built on a Bidirectional Mamba backbone, enabling coherent motion synthesis, strong music-dance alignment, and efficient non-autoregressive inference. Extensive experiments demonstrate that TokenDance achieves overall state-of-the-art (SOTA) performance in both generation quality and inference speed, highlighting its effectiveness and practical value for real-world music-to-dance applications.
>
---
#### [new 307] CARLA-Air: Fly Drones Inside a CARLA World -- A Unified Infrastructure for Air-Ground Embodied Intelligence
- **分类: cs.RO; cs.AI; cs.CV; cs.HC**

- **简介: 该论文提出CARLA-Air，融合空中与地面模拟，解决多模态智能体协同仿真问题。它统一了高保真驾驶与飞行物理，支持多种任务的开发与测试。**

- **链接: [https://arxiv.org/pdf/2603.28032](https://arxiv.org/pdf/2603.28032)**

> **作者:** Tianle Zeng; Hanxuan Chen; Yanci Wen; Hong Zhang
>
> **备注:** Prebuilt binaries, project page, full source code, and community discussion group are all available at: this https URL
>
> **摘要:** The convergence of low-altitude economies, embodied intelligence, and air-ground cooperative systems creates growing demand for simulation infrastructure capable of jointly modeling aerial and ground agents within a single physically coherent environment. Existing open-source platforms remain domain-segregated: driving simulators lack aerial dynamics, while multirotor simulators lack realistic ground scenes. Bridge-based co-simulation introduces synchronization overhead and cannot guarantee strict spatial-temporal consistency. We present CARLA-Air, an open-source infrastructure that unifies high-fidelity urban driving and physics-accurate multirotor flight within a single Unreal Engine process. The platform preserves both CARLA and AirSim native Python APIs and ROS 2 interfaces, enabling zero-modification code reuse. Within a shared physics tick and rendering pipeline, CARLA-Air delivers photorealistic environments with rule-compliant traffic, socially-aware pedestrians, and aerodynamically consistent UAV dynamics, synchronously capturing up to 18 sensor modalities across all platforms at each tick. The platform supports representative air-ground embodied intelligence workloads spanning cooperation, embodied navigation and vision-language action, multi-modal perception and dataset construction, and reinforcement-learning-based policy training. An extensible asset pipeline allows integration of custom robot platforms into the shared world. By inheriting AirSim's aerial capabilities -- whose upstream development has been archived -- CARLA-Air ensures this widely adopted flight stack continues to evolve within a modern infrastructure. Released with prebuilt binaries and full source: this https URL
>
---
#### [new 308] From Pixels to BFS: High Maze Accuracy Does Not Imply Visual Planning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视觉空间任务研究，探讨多模态模型是否具备真实规划能力。通过MazeBench基准测试，发现高准确率源于文本搜索而非空间理解，揭示了模型依赖文本枚举而非真正推理的问题。**

- **链接: [https://arxiv.org/pdf/2603.26839](https://arxiv.org/pdf/2603.26839)**

> **作者:** Alberto G. Rodriguez Salgado
>
> **备注:** 15 pages, 10 figures. Code and mazes available at this https URL
>
> **摘要:** How do multimodal models solve visual spatial tasks -- through genuine planning, or through brute-force search in token space? We introduce \textsc{MazeBench}, a benchmark of 110 procedurally generated maze images across nine controlled groups, and evaluate 16 model configurations from OpenAI, Anthropic, Google, and Alibaba. GPT-5.4 solves 91\% and Gemini 3.1 Pro 79\%, but these scores are misleading: models typically translate images into text grids and then enumerate paths step by step, consuming 1,710--22,818 tokens per solve for a task humans do quickly. Without added reasoning budgets, all configurations score only 2--12\%; on 20$\times$20 ultra-hard mazes, they hit token limits and fail. Qualitative traces reveal a common two-stage strategy: image-to-grid translation followed by token-level search, effectively BFS in prose. A text-grid ablation shows Claude Sonnet 4.6 rising from 6\% on images to 80\% when given the correct grid, isolating weak visual extraction from downstream search. When explicitly instructed not to construct a grid or perform graph search, models still revert to the same enumeration strategy. \textsc{MazeBench} therefore shows that high accuracy on visual planning tasks does not imply human-like spatial understanding.
>
---
#### [new 309] Toward Actionable Digital Twins for Radiation-Based Imaging and Therapy: Mathematical Formulation, Modular Workflow, and an OpenKBP-Based Dose-Surrogate Prototype
- **分类: eess.IV; cs.CV; stat.AP; stat.CO**

- **简介: 该论文属于放射治疗中的数字孪生任务，旨在解决剂量预测与不确定性量化问题，构建模块化框架并实现可重复的原型系统。**

- **链接: [https://arxiv.org/pdf/2603.26820](https://arxiv.org/pdf/2603.26820)**

> **作者:** Hsin-Hsiung Huang; Bulent Soykan
>
> **摘要:** Digital twins for radiation-based imaging and therapy are most useful when they assimilate patient data, quantify predictive uncertainty, and support clinically constrained decisions. This paper presents a modular framework for actionable digital twins in radiation-based imaging and therapy and instantiates its reproducible open-data component using the \openkbpfull{} benchmark. The framework couples PatientData, Model, Solver, Calibration, and Decision modules and formalizes latent-state updating, uncertainty propagation, and chance-constrained action selection. As an initial implementation, we build a GPU-ready PyTorch/MONAI reimplementation of the \openkbp{} starter pipeline: an 11-channel, 19.2M-parameter 3D U-Net trained with a masked loss over the feasible region and equipped with Monte Carlo dropout for voxel-wise epistemic uncertainty. To emulate the update loop on a static benchmark, we introduce decoder-only proxy recalibration and illustrate uncertainty-aware virtual-therapy evaluation using DVH-based and biological utilities. A complete three-fraction loop including recalibration, Monte Carlo inference, and spatial optimization executes in 10.3~s. On the 100-patient test set, the model achieved mean dose and DVH scores of 2.65 and 1.82~Gy, respectively, with 0.58~s mean inference time per patient. The \openkbp{} case study thus serves as a reproducible test bed for dose prediction, uncertainty propagation, and proxy closed-loop adaptation, while future institutional studies will address longitudinal calibration with delivered-dose logs and repeat imaging.
>
---
#### [new 310] LITTA: Late-Interaction and Test-Time Alignment for Visually-Grounded Multimodal Retrieval
- **分类: cs.IR; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态文档检索任务，解决视觉丰富文档中证据检索困难的问题。提出LITTA框架，通过查询扩展和测试时对齐提升检索效果。**

- **链接: [https://arxiv.org/pdf/2603.26683](https://arxiv.org/pdf/2603.26683)**

> **作者:** Seonok Kim
>
> **摘要:** Retrieving relevant evidence from visually rich documents such as textbooks, technical reports, and manuals is challenging due to long context, complex layouts, and weak lexical overlap between user questions and supporting pages. We propose LITTA, a query-expansion-centric retrieval framework for evidence page retrieval that improves multimodal document retrieval without retriever retraining. Given a user query, LITTA generates complementary query variants using a large language model and retrieves candidate pages for each variant using a frozen vision retriever with late-interaction scoring. Candidates from expanded queries are then aggregated through reciprocal rank fusion to improve evidence coverage and reduce sensitivity to any single phrasing. This simple test-time strategy significantly improves retrieval robustness while remaining compatible with existing multimodal embedding indices. We evaluate LITTA on visually grounded document retrieval tasks across three domains: computer science, pharmaceuticals, and industrial manuals. Multi-query retrieval consistently improves top-k accuracy, recall, and MRR compared to single-query retrieval, with particularly large gains in domains with high visual and semantic variability. Moreover, the accuracy-efficiency trade-off is directly controllable by the number of query variants, making LITTA practical for deployment under latency constraints. These results demonstrate that query expansion provides a simple yet effective mechanism for improving visually grounded multimodal retrieval.
>
---
#### [new 311] $AutoDrive\text{-}P^3$: Unified Chain of Perception-Prediction-Planning Thought via Reinforcement Fine-Tuning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AutoDrive-P³框架，解决VLM在自动驾驶中缺乏连贯推理和模块协同的问题。通过整合感知、预测与规划，提升决策安全性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.28116](https://arxiv.org/pdf/2603.28116)**

> **作者:** Yuqi Ye; Zijian Zhang; Junhong Lin; Shangkun Sun; Changhao Peng; Wei Gao
>
> **备注:** Accepted at ICLR 2026 (International Conference on Learning Representations)
>
> **摘要:** Vision-language models (VLMs) are increasingly being adopted for end-to-end autonomous driving systems due to their exceptional performance in handling long-tail scenarios. However, current VLM-based approaches suffer from two major limitations: 1) Some VLMs directly output planning results without chain-of-thought (CoT) reasoning, bypassing crucial perception and prediction stages which creates a significant domain gap and compromises decision-making capability; 2) Other VLMs can generate outputs for perception, prediction, and planning tasks but employ a fragmented decision-making approach where these modules operate separately, leading to a significant lack of synergy that undermines true planning performance. To address these limitations, we propose ${AutoDrive\text{-}P^3}$, a novel framework that seamlessly integrates $\textbf{P}$erception, $\textbf{P}$rediction, and $\textbf{P}$lanning through structured reasoning. We introduce the ${P^3\text{-}CoT}$ dataset to facilitate coherent reasoning and propose ${P^3\text{-}GRPO}$, a hierarchical reinforcement learning algorithm that provides progressive supervision across all three tasks. Specifically, ${AutoDrive\text{-}P^3}$ progressively generates CoT reasoning and answers for perception, prediction, and planning, where perception provides essential information for subsequent prediction and planning, while both perception and prediction collectively contribute to the final planning decisions, enabling safer and more interpretable autonomous driving. Additionally, to balance inference efficiency with performance, we introduce dual thinking modes: detailed thinking and fast thinking. Extensive experiments on both open-loop (nuScenes) and closed-loop (NAVSIMv1/v2) benchmarks demonstrate that our approach achieves state-of-the-art performance in planning tasks. Code is available at this https URL.
>
---
#### [new 312] ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于视频插帧任务，解决移动设备上实时帧率翻倍的效率与精度问题。通过利用H.264解码器的运动矢量，减少计算负担，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2603.26835](https://arxiv.org/pdf/2603.26835)**

> **作者:** Shibo Liu
>
> **备注:** 12 pages, 4 figures, 9 tables
>
> **摘要:** Mobile displays refresh at 90-120 Hz, yet most video is encoded at 24-30 frames per second; real-time frame-rate doubling requires each synthesized frame within 33.3 ms on mobile neural processing units. We show that mainstream flow-based video frame interpolation faces three structural deployment barriers on mobile accelerators: spatial sampling operators exceed the frame budget or lack hardware support, iterative flow refinement collapses under 8-bit post-training quantization, and memory-bound operators dominate the inference graph. ANVIL addresses these barriers by reusing motion vectors already computed by the H.264 decoder to prealign input frames, removing learned optical flow, spatial sampling, and iterative accumulation from the accelerator graph. The remaining residual is refined by a convolution-dominated network whose inference graph is composed almost entirely of compute-bound operators. On a Snapdragon 8 Gen 3 device, ANVIL achieves 12.8 ms 1080p network inference in 8-bit integer precision; an open-source Android player sustains 28.4 ms median end-to-end latency per interpolated frame pair over 54,623 consecutively logged samples during 30-minute continuous playback. Per-operator causal analysis identifies quantized accumulation on recurrent flow states as a key mechanism behind integer quantization failure in iterative methods. The current design targets H.264 playback scenarios with decoder-exposed motion vectors.
>
---
#### [new 313] Reliability-Aware Weighted Multi-Scale Spatio-Temporal Maps for Heart Rate Monitoring
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于心率监测任务，解决rPPG信号在非受控环境下的质量下降问题。提出WMST地图和HHH波let map，提升信号可靠性与估计精度。**

- **链接: [https://arxiv.org/pdf/2603.26836](https://arxiv.org/pdf/2603.26836)**

> **作者:** Arpan Bairagi; Rakesh Dey; Siladittya Manna; Umapada Pal
>
> **备注:** 6 pages, 4 figures. Under review at ICIP 2026
>
> **摘要:** Remote photoplethysmography (rPPG) allows for the contactless estimation of physiological signals from facial videos by analyzing subtle skin color changes. However, rPPG signals are extremely susceptible to illumination changes, motion, shadows, and specular reflections, resulting in low-quality signals in unconstrained environments. To overcome these issues, we present a Reliability-Aware Weighted Multi-Scale Spatio-Temporal (WMST) map that models pixel reliability through the suppression of environmental noises. These noises are modeled using different weighting strategies to focus on more physiologically valid areas. Leveraging the WMST map, we develop an SSL contrastive learning approach based on Swin-Unet, where positive pairs are generated from conventional rPPG signals and temporally expanded WMST maps. Moreover, we introduce a new High-High-High (HHH) wavelet map as a negative example that maintains motion and structural details while filtering out physiological information. Here, our aim is to estimate heart rate (HR), and the experiments on public rPPG benchmarks show that our approach enhances motion and illumination robustness with lower HR estimation error and higher Pearson correlation than existing Self-Supervised Learning (SSL) based rPPG methods.
>
---
#### [new 314] Learning to Select Visual In-Context Demonstrations
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究视觉上下文学习中的演示选择问题，旨在提升多模态大语言模型的性能。通过强化学习方法，构建更优演示集，解决事实性回归任务中的冗余与覆盖不足问题。**

- **链接: [https://arxiv.org/pdf/2603.26775](https://arxiv.org/pdf/2603.26775)**

> **作者:** Eugene Lee; Yu-Chi Lin; Jiajie Diao
>
> **备注:** 21 pages, 12 figure, accepted to Computer Vision and Pattern Recognition Conference (CVPR) 2026 Findings Track
>
> **摘要:** Multimodal Large Language Models (MLLMs) adapt to visual tasks via in-context learning (ICL), which relies heavily on demonstration quality. The dominant demonstration selection strategy is unsupervised k-Nearest Neighbor (kNN) search. While simple, this similarity-first approach is sub-optimal for complex factual regression tasks; it selects redundant examples that fail to capture the task's full output range. We reframe selection as a sequential decision-making problem and introduce Learning to Select Demonstrations (LSD), training a Reinforcement Learning agent to construct optimal demonstration sets. Using a Dueling DQN with a query-centric Transformer Decoder, our agent learns a policy that maximizes MLLM downstream performance. Evaluating across five visual regression benchmarks, we uncover a crucial dichotomy: while kNN remains optimal for subjective preference tasks, LSD significantly outperforms baselines on objective, factual regression tasks. By balancing visual relevance with diversity, LSD better defines regression boundaries, illuminating when learned selection is strictly necessary for visual ICL.
>
---
#### [new 315] StreamingVLA: Streaming Vision-Language-Action Model with Action Flow Matching and Adaptive Early Observation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出StreamingVLA，解决VLA模型在边缘设备上的效率问题。通过动作流匹配和自适应观测机制，实现各阶段并行，提升执行速度与流畅性。**

- **链接: [https://arxiv.org/pdf/2603.28565](https://arxiv.org/pdf/2603.28565)**

> **作者:** Yiran Shi; Dongqi Guo; Tianchen Zhao; Feng Gao; Liangzhi Shi; Chao Yu; ZhiJian Mo; Qihua Xiao; XiaoShuai Peng; Qingmin Liao; Yu Wang
>
> **摘要:** Vision-language-action (VLA) models have demonstrated exceptional performance in natural language-driven perception and control. However, the high computational cost of VLA models poses significant efficiency challenges, particularly for resource-constrained edge platforms in real-world deployments. However, since different stages of VLA (observation, action generation and execution) must proceed sequentially, and wait for the completion of the preceding stage, the system suffers from frequent halting and high latency. To address this, We conduct a systematic analysis to identify the challenges for fast and fluent generation, and propose enabling VLAs with the ability to asynchronously parallelize across VLA stages in a "streaming" manner. First, we eliminate the reliance on action chunking and adopt action flow matching, which learns the trajectory of action flows rather than denoising chunk-wise actions. It overlaps the latency of action generation and execution. Second, we design an action saliency-aware adaptive observation mechanism, thereby overlapping the latency of execution and observation. Without sacrificing performance, StreamingVLA achieves substantial speedup and improves the fluency of execution. It achieves a 2.4 $\times$ latency speedup and reduces execution halting by 6.5 $\times$.
>
---
#### [new 316] Efficient Inference of Large Vision Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决LVLM推理效率低的问题。通过分类和分析现有优化方法，提出系统性框架以提升推理速度。**

- **链接: [https://arxiv.org/pdf/2603.27960](https://arxiv.org/pdf/2603.27960)**

> **作者:** Surendra Pathak
>
> **备注:** 12 pages
>
> **摘要:** Although Large Vision Language Models (LVLMs) have demonstrated impressive multimodal reasoning capabilities, their scalability and deployment are constrained by massive computational requirements. In particular, the massive amount of visual tokens from high-resolution input data aggravates the situation due to the quadratic complexity of attention mechanisms. To address these issues, the research community has developed several optimization frameworks. This paper presents a comprehensive survey of the current state-of-the-art techniques for accelerating LVLM inference. We introduce a systematic taxonomy that categorizes existing optimization frameworks into four primary dimensions: visual token compression, memory management and serving, efficient architectural design, and advanced decoding strategies. Furthermore, we critically examine the limitations of these current methodologies and identify critical open problems to inspire future research directions in efficient multimodal systems.
>
---
#### [new 317] Stress Classification from ECG Signals Using Vision Transformer
- **分类: eess.SP; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于心电图（ECG）应力分类任务，旨在解决个体间差异带来的挑战。通过将ECG转换为2D频谱图并使用Vision Transformer进行分类，取得了优于传统方法的结果。**

- **链接: [https://arxiv.org/pdf/2603.26721](https://arxiv.org/pdf/2603.26721)**

> **作者:** Zeeshan Ahmad; Naimul Khan
>
> **备注:** 10 pages
>
> **摘要:** Vision Transformers have shown tremendous success in numerous computer vision applications; however, they have not been exploited for stress assessment using physiological signals such as Electrocardiogram (ECG). In order to get the maximum benefit from the vision transformer for multilevel stress assessment, in this paper, we transform the raw ECG data into 2D spectrograms using short time Fourier transform (STFT). These spectrograms are divided into patches for feeding to the transformer encoder. We also perform experiments with 1D CNN and ResNet-18 (CNN model). We perform leave-onesubject-out cross validation (LOSOCV) experiments on WESAD and Ryerson Multimedia Lab (RML) dataset. One of the biggest challenges of LOSOCV based experiments is to tackle the problem of intersubject variability. In this research, we address the issue of intersubject variability and show our success using 2D spectrograms and the attention mechanism of transformer. Experiments show that vision transformer handles the effect of intersubject variability much better than CNN-based models and beats all previous state-of-the-art methods by a considerable margin. Moreover, our method is end-to-end, does not require handcrafted features, and can learn robust representations. The proposed method achieved 71.01% and 76.7% accuracies with RML dataset and WESAD dataset respectively for three class classification and 88.3% for binary classification on WESAD.
>
---
#### [new 318] RAD-LAD: Rule and Language Grounded Autonomous Driving in Real-Time
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出LAD和RAD两种方法，解决自动驾驶中的实时决策问题。LAD实现快速运动规划，RAD提升规则性，二者结合提升系统可靠性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.28522](https://arxiv.org/pdf/2603.28522)**

> **作者:** Anurag Ghosh; Srinivasa Narasimhan; Manmohan Chandraker; Francesco Pittaluga
>
> **摘要:** We present LAD, a real-time language--action planner with an interruptible architecture that produces a motion plan in a single forward pass (~20 Hz) or generates textual reasoning alongside a motion plan (~10 Hz). LAD is fast enough for real-time closed-loop deployment, achieving ~3x lower latency than prior driving language models while setting a new learning-based state of the art on nuPlan Test14-Hard and InterPlan. We also introduce RAD, a rule-based planner designed to address structural limitations of PDM-Closed. RAD achieves state-of-the-art performance among rule-based planners on nuPlan Test14-Hard and InterPlan. Finally, we show that combining RAD and LAD enables hybrid planning that captures the strengths of both approaches. This hybrid system demonstrates that rules and learning provide complementary capabilities: rules support reliable maneuvering, while language enables adaptive and explainable decision-making.
>
---
#### [new 319] Privacy-Preserving Iris Recognition: Performance Challenges and Outlook
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于隐私保护任务，旨在解决虹膜识别中的隐私安全问题。通过FHE技术实现隐私保护，提出一种符合标准的框架，并评估其性能与准确性。**

- **链接: [https://arxiv.org/pdf/2603.26890](https://arxiv.org/pdf/2603.26890)**

> **作者:** Christina Karakosta; Lian Alhedaithy; William J. Knottenbelt
>
> **摘要:** Iris-based biometric identification is increasingly recognized for its significant accuracy and long-term stability compared to other biometric modalities such as fingerprints or facial features. However, all biometric modalities are highly sensitive data that raise serious privacy and security concerns, particularly in decentralized and untrusted environments. While Fully Homomorphic Encryption (FHE) has emerged as a promising solution for protecting sensitive data during computation, existing privacy-preserving iris recognition systems face significant performance limitations that hinder their practical deployment. This paper investigates the performance challenges of the current landscape of privacy-preserving iris recognition systems using FHE. Based on these insights, we outline a scalable privacy-preserving framework that aligns with all the requirements specified in the ISO/IEC 24745 standard. Leveraging the Open Iris library, our approach starts with robust iris segmentation, followed by normalization and feature extraction using Gabor filters to generate iris codes. We then apply binary masking to filter out unreliable regions and perform matching using Hamming distance on encrypted iris codes. The accuracy and performance of our proposed privacy-preserving framework is evaluated on the CASIA-Iris-Thousand dataset. Results show that our privacy-preserving framework yields very similar accuracy to the cleartext equivalent, but a much higher computational overhead with respect to pairwise iris template comparisons, of $\sim 120\,000 \times$. This points towards the need for the deployment of two-level schemes in the context of scalable $1-N$ template comparisons.
>
---
#### [new 320] ContraMap: Contrastive Uncertainty Mapping for Robot Environment Representation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出ContraMap，用于机器人环境建模中的不确定性映射。解决感知可靠性问题，通过对比学习实现实时环境预测与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2603.27632](https://arxiv.org/pdf/2603.27632)**

> **作者:** Chi Cuong Le; Weiming Zhi
>
> **摘要:** Reliable robot perception requires not only predicting scene structure, but also identifying where predictions should be treated as unreliable due to sparse or missing observations. We present ContraMap, a contrastive continuous mapping method that augments kernel-based discriminative maps with an explicit uncertainty class trained using synthetic noise samples. This formulation treats unobserved regions as a contrastive class, enabling joint environment prediction and spatial uncertainty estimation in real time without Bayesian inference. Under a simple mixture-model view, we show that the probability assigned to the uncertainty class is a monotonic function of a distance-aware uncertainty surrogate. Experiments in 2D occupancy mapping, 3D semantic mapping, and tabletop scene reconstruction show that ContraMap preserves mapping quality, produces spatially coherent uncertainty estimates, and is substantially more efficient than Bayesian kernelmap baselines.
>
---
#### [new 321] DiffSoup: Direct Differentiable Rasterization of Triangle Soup for Extreme Radiance Field Simplification
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出DiffSoup，用于极端简化辐射场的直接可微三角形汤渲染。解决高效传输与跨平台渲染问题，通过少量三角形和神经纹理实现稳定训练与传统图形管线集成。**

- **链接: [https://arxiv.org/pdf/2603.27151](https://arxiv.org/pdf/2603.27151)**

> **作者:** Kenji Tojo; Bernd Bickel; Nobuyuki Umetani
>
> **摘要:** Radiance field reconstruction aims to recover high-quality 3D representations from multi-view RGB images. Recent advances, such as 3D Gaussian splatting, enable real-time rendering with high visual fidelity on sufficiently powerful graphics hardware. However, efficient online transmission and rendering across diverse platforms requires drastic model simplification, reducing the number of primitives by several orders of magnitude. We introduce DiffSoup, a radiance field representation that employs a soup (i.e., a highly unstructured set) of a small number of triangles with neural textures and binary opacity. We show that this binary opacity representation is directly differentiable via stochastic opacity masking, enabling stable training without a mollifier (i.e., smooth rasterization). DiffSoup can be rasterized using standard depth testing, enabling seamless integration into traditional graphics pipelines and interactive rendering on consumer-grade laptops and mobile devices. Code is available at this https URL.
>
---
#### [new 322] Central-to-Local Adaptive Generative Diffusion Framework for Improving Gene Expression Prediction in Data-Limited Spatial Transcriptomics
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于空间转录组学任务，解决数据稀缺问题。提出C2L-ST框架，结合形态学先验与少量分子信息，生成高质量图像-基因对，提升基因表达预测效果。**

- **链接: [https://arxiv.org/pdf/2603.26827](https://arxiv.org/pdf/2603.26827)**

> **作者:** Yaoyu Fang; Jiahe Qian; Xinkun Wang; Lee A. Cooper; Bo Zhou
>
> **备注:** 31 pages, 12 figures, under review
>
> **摘要:** Spatial Transcriptomics (ST) provides spatially resolved gene expression profiles within intact tissue architecture, enabling molecular analysis in histological context. However, the high cost, limited throughput, and restricted data sharing of ST experiments result in severe data scarcity, constraining the development of robust computational models. To address this limitation, we present a Central-to-Local adaptive generative diffusion framework for ST (C2L-ST) that integrates large-scale morphological priors with limited molecular guidance. A global central model is first pretrained on extensive histopathology datasets to learn transferable morphological representations, and institution-specific local models are then adapted through lightweight gene-conditioned modulation using a small number of paired image-gene spots. This strategy enables the synthesis of realistic and molecularly consistent histology patches under data-limited conditions. The generated images exhibit high visual and structural fidelity, reproduce cellular composition, and show strong embedding overlap with real data across multiple organs, reflecting both realism and diversity. When incorporated into downstream training, synthetic image-gene pairs improve gene expression prediction accuracy and spatial coherence, achieving performance comparable to real data while requiring only a fraction of sampled spots. C2L-ST provides a scalable and data-efficient framework for molecular-level data augmentation, offering a domain-adaptive and generalizable approach for integrating histology and transcriptomics in spatial biology and related fields.
>
---
#### [new 323] Deep Learning Multi-Horizon Irradiance Nowcasting: A Comparative Evaluation of Three Methods for Leveraging Sky Images
- **分类: eess.SY; cs.AI; cs.CV**

- **简介: 该论文属于光伏功率预测任务，旨在提升短时辐照度预报精度。通过比较三种利用天空图像的深度学习方法，验证了工程化特征融合的有效性。**

- **链接: [https://arxiv.org/pdf/2603.26704](https://arxiv.org/pdf/2603.26704)**

> **作者:** Erling W. Eriksen; Magnus M. Nygård; Niklas Erdmann; Heine N. Riise
>
> **摘要:** We investigate three distinct methods of incorporating all-sky imager (ASI) images into deep learning (DL) irradiance nowcasting. The first method relies on a convolutional neural network (CNN) to extract features directly from raw RGB images. The second method uses state-of-the-art algorithms to engineer 2D feature maps informed by domain knowledge, e.g., cloud segmentation, the cloud motion vector, solar position, and cloud base height. These feature maps are then passed to a CNN to extract compound features. The final method relies on aggregating the engineered 2D feature maps into time-series input. Each of the three methods were then used as part of a DL model trained on a high-frequency, 29-day dataset to generate multi-horizon forecasts of global horizontal irradiance up to 15 minutes ahead. The models were then evaluated using root mean squared error and skill score on 7 selected days of data. Aggregated engineered ASI features as model input yielded superior forecasting performance, demonstrating that integration of ASI images into DL nowcasting models is possible without complex spatially-ordered DL-architectures and inputs, underscoring opportunities for alternative image processing methods as well as the potential for improved spatial DL feature processing methods.
>
---
#### [new 324] MeshTailor: Cutting Seams via Generative Mesh Traversal
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出MeshTailor，解决3D表面缝合线生成任务。通过直接操作网格图，避免投影误差，生成更连贯的缝合线。**

- **链接: [https://arxiv.org/pdf/2603.27309](https://arxiv.org/pdf/2603.27309)**

> **作者:** Xueqi Ma; Xingguang Yan; Congyue Zhang; Hui Huang
>
> **摘要:** We present MeshTailor, the first mesh-native generative framework for synthesizing edge-aligned seams on 3D surfaces. Unlike prior optimization-based or extrinsic learning-based methods, MeshTailor operates directly on the mesh graph, eliminating projection artifacts and fragile snapping heuristics. We introduce ChainingSeams, a hierarchical serialization of the seam graph that prioritizes global structural cuts before local details in a coarse-to-fine manner, and a dual-stream encoder that fuses topological and geometric context. Leveraging this hierarchical representation and enriched vertex embeddings, our MeshTailor Transformer utilizes an autoregressive pointer layer to trace seams vertex-by-vertex within local neighborhoods, ensuring projection-free, edge-aligned seams. Extensive evaluations show that MeshTailor produces more coherent, professional-quality seam layouts compared to recent optimization-based and learning-based baselines.
>
---
#### [new 325] Exploring Student Perception on Gen AI Adoption in Higher Education: A Descriptive Study
- **分类: cs.CY; cs.CV**

- **简介: 该论文属于教育技术研究，旨在探讨学生对生成式AI在高等教育中应用的感知。通过问卷调查，分析学生使用习惯、态度及对机构支持的需求，提出应加强AI素养教育。**

- **链接: [https://arxiv.org/pdf/2603.27777](https://arxiv.org/pdf/2603.27777)**

> **作者:** Harpreet Singh; Jaspreet Singh; Satwant Singh; Rupinder Singh; Shamim Ibne Shahid; Mohammad Hassan; Tayarani Najaran
>
> **摘要:** The rapid proliferation of Generative Artificial Intelligence (GenAI) is reshaping pedagogical practices and assessment models in higher education. While institutional and educator perspectives on GenAI integration are increasingly documented, the student perspective remains comparatively underexplored. This study examines how students perceive, use, and evaluate GenAI within their academic practices, focusing on usage patterns, perceived benefits, and expectations for institutional support. Data were collected through a questionnaire administered to 436 postgraduate Computer Science students at the University of Hertfordshire and analysed using descriptive methods. The findings reveal a Confidence-Competence Paradox: although more than 60% of students report high familiarity with tools such as ChatGPT, daily academic use remains limited and confidence in effective application is only moderate. Students primarily employ GenAI for cognitive scaffolding tasks, including concept clarification and brainstorming, rather than fully automated content generation. At the same time, respondents express concerns regarding data privacy, reliability of AI-generated information, and the potential erosion of critical thinking skills. The results also indicate strong student support for integrating AI literacy into curricula and programme Knowledge, Skills, and Behaviours (KSBs). Overall, the study suggests that universities should move beyond a policing approach to GenAI and adopt a pedagogical framework that emphasises AI literacy, ethical guidance, and equitable access to AI tools.
>
---
#### [new 326] Quantitative measurements of biological/chemical concentrations using smartphone cameras
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于生物/化学浓度检测任务，旨在利用智能手机相机定量测量样本浓度。通过构建图像数据库并结合图像处理技术，实现对荧光物质和胶体混合物的准确检测。**

- **链接: [https://arxiv.org/pdf/2603.27118](https://arxiv.org/pdf/2603.27118)**

> **作者:** Zhendong Cao; Hongji Dai; Zhida Li; Ash Parameswaran
>
> **摘要:** This paper presents a smartphone-based imaging system capable of quantifying the concentration of an assortment of biological/chemical assay samples. The main objective is to construct an image database which characterizes the relationship between color information and concentrations of the biological/chemical assay sample. For this aim, a designated optical setup combined with image processing and data analyzing techniques was implemented. A series of experiments conducted on selected assays, including fluorescein, RNA Mango, homogenized milk and yeast have demonstrated that the proposed system estimates the concentration of fluorescent materials and colloidal mixtures comparable to currently used commercial and laboratory instruments. Furthermore, by utilizing the camera and computational power of smartphones, eventual development can be directed toward extremely compact, inexpensive and portable analysis and diagnostic systems which will allow experiments and tests to be conducted in remote or impoverished areas.
>
---
#### [new 327] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ManipArena，用于评估机器人推理导向的通用操作能力。解决真实场景下评估标准缺失的问题，通过20个任务和多维度测试，提升评估的公平性与真实性。**

- **链接: [https://arxiv.org/pdf/2603.28545](https://arxiv.org/pdf/2603.28545)**

> **作者:** Yu Sun; Meng Cao; Ping Yang; Rongtao Xu; Yunxiao Yan; Runze Xu; Liang Ma; Roy Gan; Andy Zhai; Qingxuan Chen; Zunnan Xu; Hao Wang; Jincheng Yu; Lucy Liang; Qian Wang; Ivan Laptev; Ian D Reid; Xiaodan Liang
>
> **备注:** Technical report for CVPR 2026 Challenge ManipArena
>
> **摘要:** Vision-Language-Action (VLA) models and world models have recently emerged as promising paradigms for general-purpose robotic intelligence, yet their progress is hindered by the lack of reliable evaluation protocols that reflect real-world deployment. Existing benchmarks are largely simulator-centric, which provide controllability but fail to capture the reality gap caused by perception noise, complex contact dynamics, hardware constraints, and system latency. Moreover, fragmented real-world evaluations across different robot platforms prevent fair and reproducible comparison. To address these challenges, we introduce ManipArena, a standardized evaluation framework designed to bridge simulation and real-world execution. ManipArena comprises 20 diverse tasks across 10,812 expert trajectories emphasizing reasoning-oriented manipulation tasks requiring semantic and spatial reasoning, supports multi-level generalization through controlled out-of-distribution settings, and incorporates long-horizon mobile manipulation beyond tabletop scenarios. The framework further provides rich sensory diagnostics, including low-level motor signals, and synchronized real-to-sim environments constructed via high-quality 3D scanning. Together, these features enable fair, realistic, and reproducible evaluation for both VLA and world model approaches, providing a scalable foundation for diagnosing and advancing embodied intelligence systems.
>
---
#### [new 328] FedFG: Privacy-Preserving and Robust Federated Learning via Flow-Matching Generation
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于联邦学习任务，旨在解决隐私泄露和中毒攻击问题。提出FedFG框架，通过流匹配生成保护隐私并增强模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27986](https://arxiv.org/pdf/2603.27986)**

> **作者:** Ruiyang Wang; Rong Pan; Zhengan Yao
>
> **摘要:** Federated learning (FL) enables distributed clients to collaboratively train a global model using local private data. Nevertheless, recent studies show that conventional FL algorithms still exhibit deficiencies in privacy protection, and the server lacks a reliable and stable aggregation rule for updating the global model. This situation creates opportunities for adversaries: on the one hand, they may eavesdrop on uploaded gradients or model parameters, potentially leaking benign clients' private data; on the other hand, they may compromise clients to launch poisoning attacks that corrupt the global model. To balance accuracy and security, we propose FedFG, a robust FL framework based on flow-matching generation that simultaneously preserves client privacy and resists sophisticated poisoning attacks. On the client side, each local network is decoupled into a private feature extractor and a public classifier. Each client is further equipped with a flow-matching generator that replaces the extractor when interacting with the server, thereby protecting private features while learning an approximation of the underlying data distribution. Complementing the client-side design, the server employs a client-update verification scheme and a novel robust aggregation mechanism driven by synthetic samples produced by the flow-matching generator. Experiments on MNIST, FMNIST, and CIFAR-10 demonstrate that, compared with prior work, our approach adapts to multiple attack strategies and achieves higher accuracy while maintaining strong privacy protection.
>
---
#### [new 329] A Comparative Study in Surgical AI: Datasets, Foundation Models, and Barriers to Med-AGI
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 论文探讨了AI在手术领域的应用，聚焦于工具检测任务。研究指出当前模型在神经外科工具检测中表现不足，且模型规模扩大效果有限，揭示数据与标注的局限性及技术挑战。**

- **链接: [https://arxiv.org/pdf/2603.27341](https://arxiv.org/pdf/2603.27341)**

> **作者:** Kirill Skobelev; Eric Fithian; Yegor Baranovski; Jack Cook; Sandeep Angara; Shauna Otto; Zhuang-Fang Yi; John Zhu; Daniel A. Donoho; X.Y. Han; Neeraj Mainkar; Margaux Masson-Forsythe
>
> **摘要:** Recent Artificial Intelligence (AI) models have matched or exceeded human experts in several benchmarks of biomedical task performance, but have lagged behind on surgical image-analysis benchmarks. Since surgery requires integrating disparate tasks -- including multimodal data integration, human interaction, and physical effects -- generally-capable AI models could be particularly attractive as a collaborative tool if performance could be improved. On the one hand, the canonical approach of scaling architecture size and training data is attractive, especially since there are millions of hours of surgical video data generated per year. On the other hand, preparing surgical data for AI training requires significantly higher levels of professional expertise, and training on that data requires expensive computational resources. These trade-offs paint an uncertain picture of whether and to-what-extent modern AI could aid surgical practice. In this paper, we explore this question through a case study of surgical tool detection using state-of-the-art AI methods available in 2026. We demonstrate that even with multi-billion parameter models and extensive training, current Vision Language Models fall short in the seemingly simple task of tool detection in neurosurgery. Additionally, we show scaling experiments indicating that increasing model size and training time only leads to diminishing improvements in relevant performance metrics. Thus, our experiments suggest that current models could still face significant obstacles in surgical use cases. Moreover, some obstacles cannot be simply ``scaled away'' with additional compute and persist across diverse model architectures, raising the question of whether data and label availability are the only limiting factors. We discuss the main contributors to these constraints and advance potential solutions.
>
---
#### [new 330] Dictionary-based Pathology Mining with Hard-instance-assisted Classifier Debiasing for Genetic Biomarker Prediction from WSIs
- **分类: q-bio.QM; cs.CV; cs.LG**

- **简介: 该论文属于基因生物标志物预测任务，解决病理特征表示不准确和模型过拟合问题。提出D2Bio框架，通过字典式病理挖掘和难例去偏分类器提升预测效果。**

- **链接: [https://arxiv.org/pdf/2603.26809](https://arxiv.org/pdf/2603.26809)**

> **作者:** Ling Zhang; Boxiang Yun; Ting Jin; Qingli Li; Xinxing Li; Yan Wang
>
> **备注:** 13 pages, 13 figures
>
> **摘要:** Prediction of genetic biomarkers, e.g., microsatellite instability in colorectal cancer is crucial for clinical decision making. But, two primary challenges hamper accurate prediction: (1) It is difficult to construct a pathology-aware representation involving the complex interconnections among pathological components. (2) WSIs contain a large proportion of areas unrelated to genetic biomarkers, which make the model easily overfit simple but irrelative instances. We hereby propose a Dictionary-based hierarchical pathology mining with hard-instance-assisted classifier Debiasing framework to address these challenges, dubbed as D2Bio. Our first module, dictionary-based hierarchical pathology mining, is able to mine diverse and very fine-grained pathological contextual interaction without the limit to the distances between patches. The second module, hard-instance-assisted classfier debiasing, learns a debiased classifier via focusing on hard but task-related features, without any additional annotations. Experimental results on five cohorts show the superiority of our method, with over 4% improvement in AUROC compared with the second best on the TCGA-CRC-MSI cohort. Our analysis further shows the clinical interpretability of D2Bio in genetic biomarker diagnosis and potential clinical utility in survival analysis. Code will be available at this https URL.
>
---
#### [new 331] Tele-Catch: Adaptive Teleoperation for Dexterous Dynamic 3D Object Catching
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于动态物体抓取任务，解决纯远程操作在动态物体捕捉中因时间、姿态和力误差导致的失败问题。提出Tele-Catch框架，结合人类输入与自主策略，提升抓取精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.28427](https://arxiv.org/pdf/2603.28427)**

> **作者:** Weiguang Zhao; Junting Dong; Rui Zhang; Kailin Li; Qin Zhao; Kaizhu Huang
>
> **摘要:** Teleoperation is a key paradigm for transferring human dexterity to robots, yet most prior work targets objects that are initially static, such as grasping or manipulation. Dynamic object catch, where objects move before contact, remains underexplored. Pure teleoperation in this task often fails due to timing, pose, and force errors, highlighting the need for shared autonomy that combines human input with autonomous policies. To this end, we present Tele-Catch, a systematic framework for dexterous hand teleoperation in dynamic object catching. At its core, we design DAIM, a dynamics-aware adaptive integration mechanism that realizes shared autonomy by fusing glove-based teleoperation signals into the diffusion policy denoising process. It adaptively modulates control based on the interaction object state. To improve policy robustness, we introduce DP-U3R, which integrates unsupervised geometric representations from point cloud observations into diffusion policy learning, enabling geometry-aware decision making. Extensive experiments demonstrate that Tele-Catch significantly improves accuracy and robustness in dynamic catching tasks, while also exhibiting consistent gains across distinct dexterous hand embodiments and previously unseen object categories.
>
---
#### [new 332] Uni-World VLA: Interleaved World Modeling and Planning for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Uni-World VLA模型，解决自动驾驶中环境建模与路径规划的协同问题。通过交替预测未来帧和规划动作，实现闭环控制，提升动态场景下的决策能力。**

- **链接: [https://arxiv.org/pdf/2603.27287](https://arxiv.org/pdf/2603.27287)**

> **作者:** Qiqi Liu; Huan Xu; Jingyu Li; Bin Sun; Zhihui Hao; Dangen She; Xiatian Zhu; Li Zhang
>
> **备注:** 22 pages, 8 figures. Submitted to ECCV 2026. Code will be released
>
> **摘要:** Autonomous driving requires reasoning about how the environment evolves and planning actions accordingly. Existing world-model-based approaches typically predict future scenes first and plan afterwards, resulting in open-loop imagination that may drift from the actual decision process. In this paper, we present Uni-World VLA, a unified vision-language-action (VLA) model that tightly interleaves future frame prediction and trajectory planning. Instead of generating a full world rollout before planning, our model alternates between predicting future frames and ego actions step by step, allowing planning decisions to be continuously conditioned on the imagined future observations. This interleaved generation forms a closed-loop interaction between world modeling and control, enabling more adaptive decision-making in dynamic traffic scenarios. In addition, we incorporate monocular depth information into frames to provide stronger geometric cues for world modeling, improving long-horizon scene prediction. Experiments on the NAVSIM benchmark show that our approach achieves competitive closed-loop planning performance while producing high-fidelity future frame predictions. These results demonstrate that tightly coupling world prediction and planning is a promising direction for scalable VLA driving systems.
>
---
#### [new 333] Guided Lensless Polarization Imaging
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于光学成像任务，旨在解决传统 polarization 相机成本高、体积大及 lensless 系统重建质量低的问题。通过结合 RGB 相机与 polarization 传感器，提升重建精度与实用性。**

- **链接: [https://arxiv.org/pdf/2603.27357](https://arxiv.org/pdf/2603.27357)**

> **作者:** Noa Kraicer; Erez Yosef; Raja Giryes
>
> **摘要:** Polarization imaging captures the polarization state of light, revealing information invisible to the human eye yet valuable in domains such as biomedical diagnostics, autonomous driving, and remote sensing. However, conventional polarization cameras are often expensive, bulky, or both, limiting their practical use. Lensless imaging offers a compact, low-cost alternative by replacing the lens with a simple optical element like a diffuser and performing computational reconstruction, but existing lensless polarization systems suffer from limited reconstruction quality. To overcome these limitations, we introduce a RGB-guided lensless polarization imaging system that combines a compact polarization-RGB sensor with an auxiliary, widely available conventional RGB camera providing structural guidance. We reconstruct multi-angle polarization images for each RGB color channel through a two-stage pipeline: a physics-based inversion recovers an initial polarization image, followed by a Transformer-based fusion network that refines this reconstruction using the RGB guidance image from the conventional RGB camera. Our two-stage method significantly improves reconstruction quality and fidelity over lensless-only baselines, generalizes across datasets and imaging conditions, and achieves high-quality real-world results on our physical prototype lensless camera without any fine-tuning.
>
---
#### [new 334] SpatialPoint: Spatial-aware Point Prediction for Embodied Localization
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出SpatialPoint，解决具身定位任务，通过融合深度信息提升视觉语言模型在3D空间中的定位能力。**

- **链接: [https://arxiv.org/pdf/2603.26690](https://arxiv.org/pdf/2603.26690)**

> **作者:** Qiming Zhu; Zhirui Fang; Tianming Zhang; Chuanxiu Liu; Xiaoke Jiang; Lei Zhang
>
> **备注:** 19 pages, 12 figures, supplementary material included
>
> **摘要:** Embodied intelligence fundamentally requires a capability to determine where to act in 3D space. We formalize this requirement as embodied localization -- the problem of predicting executable 3D points conditioned on visual observations and language instructions. We instantiate embodied localization with two complementary target types: touchable points, surface-grounded 3D points enabling direct physical interaction, and air points, free-space 3D points specifying placement and navigation goals, directional constraints, or geometric relations. Embodied localization is inherently a problem of embodied 3D spatial reasoning -- yet most existing vision-language systems rely predominantly on RGB inputs, necessitating implicit geometric reconstruction that limits cross-scene generalization, despite the widespread adoption of RGB-D sensors in robotics. To address this gap, we propose SpatialPoint, a spatial-aware vision-language framework with careful design that integrates structured depth into a vision-language model (VLM) and generates camera-frame 3D coordinates. We construct a 2.6M-sample RGB-D dataset covering both touchable and air points QA pairs for training and evaluation. Extensive experiments demonstrate that incorporating depth into VLMs significantly improves embodied localization performance. We further validate SpatialPoint through real-robot deployment across three representative tasks: language-guided robotic arm grasping at specified locations, object placement to target destinations, and mobile robot navigation to goal positions.
>
---
#### [new 335] Hybrid Diffusion Model for Breast Ultrasound Image Augmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像增强任务，旨在解决乳腺超声图像数据不足的问题。通过结合文本到图像生成与图像修复，提升合成图像质量，实现更有效的数据增强。**

- **链接: [https://arxiv.org/pdf/2603.26834](https://arxiv.org/pdf/2603.26834)**

> **作者:** Farhan Fuad Abir; Sanjeda Sara Jennifer; Niloofar Yousefi; Laura J. Brattain
>
> **备注:** Accepted at IEEE International Symposium on Biomedical Imaging (ISBI) 2026
>
> **摘要:** We propose a hybrid diffusion-based augmentation framework to overcome the critical challenge of ultrasound data augmentation in breast ultrasound (BUS) datasets. Unlike conventional diffusion-based augmentations, our approach improves visual fidelity and preserves ultrasound texture by combining text-to-image generation with image-to-image (img2img) refinement, as well as fine-tuning with low-rank adaptation (LoRA) and textual inversion (TI). Our method generated realistic, class-consistent images on an open-source Kaggle breast ultrasound image dataset (BUSI). Compared to the Stable Diffusion v1.5 baseline, incorporating TI and img2img refinement reduced the Frechet Inception Distance (FID) from 45.97 to 33.29, demonstrating a substantial gain in fidelity while maintaining comparable downstream classification performance. Overall, the proposed framework effectively mitigates the low-fidelity limitations of synthetic ultrasound images and enhances the quality of augmentation for robust diagnostic modeling.
>
---
#### [new 336] Pandora: Articulated 3D Scene Graphs from Egocentric Vision
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Pandora，解决机器人感知环境不完整的问题，通过人类第一视角数据构建可动3D场景图，提升机器人对物体动态和容器关系的理解，增强移动操作能力。**

- **链接: [https://arxiv.org/pdf/2603.28732](https://arxiv.org/pdf/2603.28732)**

> **作者:** Alan Yu; Yun Chang; Christopher Xie; Luca Carlone
>
> **备注:** 14 pages, 5 figures. Presented at the 2025 British Machine Vision Conference (BMVC) in Sheffield, UK
>
> **摘要:** Robotic mapping systems typically approach building metric-semantic scene representations from the robot's own sensors and cameras. However, these "first person" maps inherit the robot's own limitations due to its embodiment or skillset, which may leave many aspects of the environment unexplored. For example, the robot might not be able to open drawers or access wall cabinets. In this sense, the map representation is not as complete, and requires a more capable robot to fill in the gaps. We narrow these blind spots in current methods by leveraging egocentric data captured as a human naturally explores a scene wearing Project Aria glasses, giving a way to directly transfer knowledge about articulation from the human to any deployable robot. We demonstrate that, by using simple heuristics, we can leverage egocentric data to recover models of articulate object parts, with quality comparable to those of state-of-the-art methods based on other input modalities. We also show how to integrate these models into 3D scene graph representations, leading to a better understanding of object dynamics and object-container relationships. We finally demonstrate that these articulated 3D scene graphs enhance a robot's ability to perform mobile manipulation tasks, showcasing an application where a Boston Dynamics Spot is tasked with retrieving concealed target items, given only the 3D scene graph as input.
>
---
## 更新

#### [replaced 001] Fast SceneScript: Fast and Accurate Language-Based 3D Scene Understanding via Multi-Token Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.05597](https://arxiv.org/pdf/2512.05597)**

> **作者:** Ruihong Yin; Xuepeng Shi; Oleksandr Bailo; Marco Manfredi; Theo Gevers
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** Recent perception-generalist approaches based on language models have achieved state-of-the-art results across diverse tasks, including 3D scene layout estimation and 3D object detection, via unified architecture and interface. However, these approaches rely on autoregressive next-token prediction, which is inherently slow. In this work, we introduce Fast SceneScript, a novel structured language model for accurate and efficient 3D scene understanding. Our method employs multi-token prediction (MTP) to reduce the number of autoregressive iterations and significantly accelerate inference. While MTP improves speed, unreliable token predictions can significantly reduce accuracy. To filter out unreliable tokens, we adapt self-speculative decoding (SSD) for structured language models and introduce confidence-guided decoding (CGD) with an improved scoring mechanism for token reliability. Furthermore, we design a parameter-efficient mechanism that reduces the parameter overhead of MTP. Extensive experiments on synthetic and real-world benchmarks demonstrate that Fast SceneScript can generate up to 9 tokens per decoder inference step without compromising accuracy, while adding only $\sim7.5\%$ additional parameters.
>
---
#### [replaced 002] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03350](https://arxiv.org/pdf/2512.03350)**

> **作者:** Yu Yuan; Tharindu Wickremasinghe; Zeeshan Nadir; Xijun Wang; Yiheng Chi; Stanley H. Chan
>
> **备注:** Accepted by CVPR 2026. Camera-Ready Version. Project Page: this https URL
>
> **摘要:** Images and videos are discrete 2D projections of the 4D world (3D space + time). Most visual understanding, prediction, and generation operate directly on 2D observations, leading to suboptimal performance. We propose SeeU, a novel approach that learns the continuous 4D dynamics and generate the unseen visual contents. The principle behind SeeU is a new 2D$\to$4D$\to$2D learning framework. SeeU first reconstructs the 4D world from sparse and monocular 2D frames (2D$\to$4D). It then learns the continuous 4D dynamics on a low-rank representation and physical constraints (discrete 4D$\to$continuous 4D). Finally, SeeU rolls the world forward in time, re-projects it back to 2D at sampled times and viewpoints, and generates unseen regions based on spatial-temporal context awareness (4D$\to$2D). By modeling dynamics in 4D, SeeU achieves continuous and physically-consistent novel visual generation, demonstrating strong potentials in multiple tasks including unseen temporal generation, unseen spatial generation, and video editing. All data and code will be public at this https URL
>
---
#### [replaced 003] OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09326](https://arxiv.org/pdf/2603.09326)**

> **作者:** Tengjin Weng; Wenhao Jiang; Jingyi Wang; Ming Li; Lin Ma; Zhong Ming
>
> **备注:** accepted by CVPR 2026
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable performance across a wide range of vision language tasks. However, their ability in low-level visual perception, particularly in detecting fine-grained visual discrepancies, remains underexplored and lacks systematic analysis. In this work, we introduce OddGridBench, a controllable benchmark for evaluating the visual discrepancy sensitivity of MLLMs. OddGridBench comprises over 1,400 grid-based images, where a single element differs from all others by one or multiple visual attributes such as color, size, rotation, or position. Experiments reveal that all evaluated MLLMs, including open-source families such as Qwen3-VL and InternVL3.5, and proprietary systems like Gemini-2.5-Pro and GPT-5, perform far below human levels in visual discrepancy detection. We further propose OddGrid-GRPO, a reinforcement learning framework that integrates curriculum learning and distance-aware reward. By progressively controlling the difficulty of training samples and incorporating spatial proximity constraints into the reward design, OddGrid-GRPO significantly enhances the model's fine-grained visual discrimination ability. We hope OddGridBench and OddGrid-GRPO will lay the groundwork for advancing perceptual grounding and visual discrepancy sensitivity in multimodal intelligence. Code and dataset are available at this https URL.
>
---
#### [replaced 004] Scaling Spatial Intelligence with Multimodal Foundation Models
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **简介: 该论文属于多模态基础模型任务，旨在提升模型的空间智能。通过构建大规模数据集，增强模型在多个空间智能基准上的表现，并分析数据扩展与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.13719](https://arxiv.org/pdf/2511.13719)**

> **作者:** Zhongang Cai; Ruisi Wang; Chenyang Gu; Fanyi Pu; Junxiang Xu; Yubo Wang; Wanqi Yin; Zhitao Yang; Chen Wei; Qingping Sun; Tongxi Zhou; Jiaqi Li; Hui En Pang; Oscar Qian; Yukun Wei; Zhiqian Lin; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Xiangyu Fan; Hanming Deng; Lewei Lu; Liang Pan; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: this https URL ; Models: this https URL . This report is based on the v1.1 version of SenseNova-SI. Accepted to CVPR 2026
>
> **摘要:** Despite remarkable progress, multimodal foundation models still exhibit surprising deficiencies in spatial intelligence. In this work, we explore scaling up multimodal foundation models to cultivate spatial intelligence within the SenseNova-SI family, built upon established multimodal foundations including visual understanding models (i.e., Qwen3-VL and InternVL3) and unified understanding and generation models (i.e., Bagel). We take a principled approach to constructing high-performing and robust spatial intelligence by systematically curating SenseNova-SI-8M: eight million diverse data samples under a rigorous taxonomy of spatial capabilities. SenseNova-SI demonstrates unprecedented performance across a broad range of spatial intelligence benchmarks: 68.8% on VSI-Bench, 43.3% on MMSI, 85.7% on MindCube, 54.7% on ViewSpatial, 47.7% on SITE, 63.9% on BLINK, 55.5% on 3DSR, and 72.0% on EmbSpatial, while maintaining strong general multimodal understanding (e.g., 84.9% on MMBench-En). More importantly, we analyze the impact of data scaling, discuss early signs of emergent generalization capabilities enabled by diverse data training, analyze the risk of overfitting and language shortcuts, present a preliminary study on spatial chain-of-thought reasoning, and validate the potential downstream application. All newly trained multimodal foundation models are publicly released.
>
---
#### [replaced 005] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.25053](https://arxiv.org/pdf/2603.25053)**

> **作者:** Liyuan Zhu; Manjunath Narayana; Michal Stary; Will Hutchcroft; Gordon Wetzstein; Iro Armeni
>
> **备注:** CVPR 2026 main paper camera-ready. Project page: this http URL
>
> **摘要:** We present GaussFusion, a novel approach for improving 3D Gaussian splatting (3DGS) reconstructions in the wild through geometry-informed video generation. GaussFusion mitigates common 3DGS artifacts, including floaters, flickering, and blur caused by camera pose errors, incomplete coverage, and noisy geometry initialization. Unlike prior RGB-based approaches limited to a single reconstruction pipeline, our method introduces a geometry-informed video-to-video generator that refines 3DGS renderings across both optimization-based and feed-forward methods. Given an existing reconstruction, we render a Gaussian primitive video buffer encoding depth, normals, opacity, and covariance, which the generator refines to produce temporally coherent, artifact-free frames. We further introduce an artifact synthesis pipeline that simulates diverse degradation patterns, ensuring robustness and generalization. GaussFusion achieves state-of-the-art performance on novel-view synthesis benchmarks, and an efficient variant runs in real time at 15 FPS while maintaining similar performance, enabling interactive 3D applications.
>
---
#### [replaced 006] Source-Only Cross-Weather LiDAR via Geometry-Aware Point Drop
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.01250](https://arxiv.org/pdf/2511.01250)**

> **作者:** YoungJae Cheong; Jhonghyun An
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** LiDAR semantic segmentation degrades in adverse weather because refraction, scattering, and point dropouts corrupt geometry. Prior work in weather simulation, mixing-based augmentation, domain randomization, and uncertainty or boundary regularization improves robustness but still overlooks structural vulnerabilities near boundaries, corners, and sparse regions. We present a Light Geometry-aware adapter. The module aligns azimuth and applies horizontal circular padding to preserve neighbor continuity across the 0~360 degree wrap-around boundary. A local-window K-Nearest Neighbors gathers nearby points and computes simple local statistics, which are compressed into compact geometry-aware cues. During training, these cues drive region-aware regularization that stabilizes predictions in structurally fragile areas. The adapter is plug and play, complements augmentation, and can be enabled only during training with negligible inference cost. We adopt a source-only cross-weather setup where models train on SemanticKITTI and are evaluated on SemanticSTF without target labels or fine-tuning. The adapter improves mIoU by 7.9 percentage points over the data-centric augmentation baseline and by 0.6 points over the class-centric regularization baseline. These results indicate that geometry-driven regularization is a key direction for all-weather LiDAR segmentation.
>
---
#### [replaced 007] 3D CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出3D-CAVLA，解决机器人在3D环境中任务泛化问题。通过引入深度感知、思维链和区域检测，提升视觉语言动作模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.05800](https://arxiv.org/pdf/2505.05800)**

> **作者:** Vineet Bhat; Yu-Hsiang Lan; Prashanth Krishnamurthy; Ramesh Karri; Farshad Khorrami
>
> **备注:** Accepted at the 1st Workshop on 3D LLM/VLA, CVPR 2025. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Robotic manipulation in 3D requires effective computation of N degree-of-freedom joint-space trajectories that enable precise and robust control. To achieve this, robots must integrate semantic understanding with visual perception to transform real-world observations into low-level control for object interaction. Recent advances in Vision-Language-Action (VLA) models have shown promise by mapping RGB images and language instructions to task space velocities, typically trained on large datasets of teleoperated demonstrations. However, these models often struggle with generalization beyond their training distributions. In this work, we introduce 3D-CAVLA, a novel finetuning framework that enhances task generalization of VLA policies by incorporating three key components: (i) chain-of-thought reasoning for structured decision-making, (ii) depth-aware perception for 3D spatial understanding, and (iii) task-oriented region-of-interest detection for focused manipulation. Extensive experiments in the LIBERO simulation environment demonstrate that 3D-CAVLA achieves an average success rate of 98.1% across diverse in-domain task suites. On unseen tasks, 3D-CAVLA delivers an absolute improvement of 8.8% in success rate, underscoring the benefits of 3D scene awareness for robust generalization. We validate our approach on real-world tabletop experiments demonstrating that the proposed model translates effectively from simulation to physical robots. 3D-CAVLA achieves over a 3X faster training convergence and delivers a 25% gain in success rate on unseen real world tasks. We will open-source our code and the unseen tasks dataset to promote community-driven research here: this https URL
>
---
#### [replaced 008] LH2Face: Loss function for Hard High-quality Face
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.23555](https://arxiv.org/pdf/2506.23555)**

> **作者:** Fan Xie; Yang Wang; Yikang Jiao; Zhenyu Yuan; Congxi Chen; Chuanxin Zhao
>
> **摘要:** In current practical face authentication systems, most face recognition (FR) algorithms are based on cosine similarity with softmax classification. Despite its reliable classification performance, this method struggles with hard samples. A popular strategy to improve FR performance is incorporating angular or cosine margins. However, it does not take face quality or recognition hardness into account, simply increasing the margin value and thus causing an overly uniform training strategy. To address this problem, a novel loss function is proposed, named Loss function for Hard High-quality Face (LH2Face). Firstly, a similarity measure based on the von Mises-Fisher (vMF) distribution is stated, specifically focusing on the logarithm of the Probability Density Function (PDF), which represents the distance between a probability distribution and a vector. Then, an adaptive margin-based multi-classification method using softmax, called the Uncertainty-Aware Margin Function, is implemented in the article. Furthermore, proxy-based loss functions are used to apply extra constraints between the proxy and sample to optimize their representation space distribution. Finally, a renderer is constructed that optimizes FR through face reconstruction and vice versa. Our LH2Face is superior to similiar schemes on hard high-quality face datasets, achieving 49.39% accuracy on the IJB-B dataset, which surpasses the second-place method by 2.37%.
>
---
#### [replaced 009] Fair Benchmarking of Emerging One-Step Generative Models Against Multistep Diffusion and Flow Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14186](https://arxiv.org/pdf/2603.14186)**

> **作者:** Advaith Ravishankar; Serena Liu; Mingyang Wang; Todd Zhou; Jeffrey Zhou; Arnav Sharma; Ziling Hu; Léopold Das; Abdulaziz Sobirov; Faizaan Siddique; Freddy Yu; Seungjoo Baek; Yan Luo; Mengyu Wang
>
> **摘要:** State-of-the-art text-to-image models produce high-quality images, but inference remains expensive as generation requires several sequential ODE or denoising steps. Native one-step models aim to reduce this cost by mapping noise to an image in a single step, yet fair comparisons to multi-step systems are difficult because studies use mismatched sampling steps and different classifier-free guidance (CFG) settings, where CFG can shift FID, Inception Score, and CLIP-based alignment in opposing directions. It is also unclear how well one-step models scale to multi-step inference, and there is limited standardized out-of-distribution evaluation for label-ID-conditioned generators beyond ImageNet. To address this, We benchmark eight models spanning one-step flows (MeanFlow, Improved MeanFlow, SoFlow), multi-step baselines (RAE, Scale-RAE), and established systems (SiT, Stable Diffusion 3.5, FLUX.1) under a controlled class-conditional protocol on ImageNet validation, ImageNetV2, and reLAIONet, our new proofread out-of-distribution dataset aligned to ImageNet label IDs. Using FID, Inception Score, CLIP Score, and Pick Score, we show that FID-focused model development and CFG selection can be misleading in few-step regimes, where guidance changes can improve FID while degrading text-image alignment and human preference signals and worsening perceived quality. We further show that leading one-step models benefit from step scaling and become substantially more competitive under multi-step inference, although they still exhibit characteristic local distortions. To capture these tradeoffs, we introduce MinMax Harmonic Mean (MMHM), a composite proxy over all four metrics that stabilizes hyperparameter selection across guidance and step sweeps.
>
---
#### [replaced 010] Coarse-Guided Visual Generation via Weighted h-Transform Sampling
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.12057](https://arxiv.org/pdf/2603.12057)**

> **作者:** Yanghao Wang; Ziqi Jiang; Zhen Wang; Long Chen
>
> **摘要:** Coarse-guided visual generation, which synthesizes fine visual samples from degraded or low-fidelity coarse references, is essential for various real-world applications. While training-based approaches are effective, they are inherently limited by high training costs and restricted generalization due to paired data collection. Accordingly, recent training-free works propose to leverage pretrained diffusion models and incorporate guidance during the sampling process. However, these training-free methods either require knowing the forward (fine-to-coarse) transformation operator, e.g., bicubic downsampling, or are difficult to balance between guidance and synthetic quality. To address these challenges, we propose a novel guided method by using the h-transform, a tool that can constrain stochastic processes (e.g., sampling process) under desired conditions. Specifically, we modify the transition probability at each sampling timestep by adding to the original differential equation with a drift function, which approximately steers the generation toward the ideal fine sample. To address unavoidable approximation errors, we introduce a noise-level-aware schedule that gradually de-weights the term as the error increases, ensuring both guidance adherence and high-quality synthesis. Extensive experiments across diverse image and video generation tasks demonstrate the effectiveness and generalization of our method.
>
---
#### [replaced 011] A$^3$: Towards Advertising Aesthetic Assessment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24037](https://arxiv.org/pdf/2603.24037)**

> **作者:** Kaiyuan Ji; Yixuan Gao; Lu Sun; Yushuo Zheng; Zijian Chen; Jianbo Zhang; Xiangyang Zhu; Yuan Tian; Zicheng Zhang; Guangtao Zhai
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Advertising images significantly impact commercial conversion rates and brand equity, yet current evaluation methods rely on subjective judgments, lacking scalability, standardized criteria, and interpretability. To address these challenges, we present A^3 (Advertising Aesthetic Assessment), a comprehensive framework encompassing four components: a paradigm (A^3-Law), a dataset (A^3-Dataset), a multimodal large language model (A^3-Align), and a benchmark (A^3-Bench). Central to A^3 is a theory-driven paradigm, A^3-Law, comprising three hierarchical stages: (1) Perceptual Attention, evaluating perceptual image signals for their ability to attract attention; (2) Formal Interest, assessing formal composition of image color and spatial layout in evoking interest; and (3) Desire Impact, measuring desire evocation from images and their persuasive impact. Building on A^3-Law, we construct A^3-Dataset with 120K instruction-response pairs from 30K advertising images, each richly annotated with multi-dimensional labels and Chain-of-Thought (CoT) rationales. We further develop A^3-Align, trained under A^3-Law with CoT-guided learning on A^3-Dataset. Extensive experiments on A^3-Bench demonstrate that A^3-Align achieves superior alignment with A^3-Law compared to existing models, and this alignment generalizes well to quality advertisement selection and prescriptive advertisement critique, indicating its potential for broader deployment. Dataset, code, and models can be found at: this https URL.
>
---
#### [replaced 012] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出RobotSeg，解决机器人图像和视频分割任务。针对机器人结构复杂、形态多样的问题，改进分割模型，提升准确性与效率。**

- **链接: [https://arxiv.org/pdf/2511.22950](https://arxiv.org/pdf/2511.22950)**

> **作者:** Haiyang Mei; Qiming Huang; Hai Ci; Mike Zheng Shou
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Accurate robot segmentation is a fundamental capability for robotic perception. It enables precise visual servoing for VLA systems, scalable robot-centric data augmentation, accurate real-to-sim transfer, and reliable safety monitoring in dynamic human-robot environments. Despite the strong capabilities of modern segmentation models, surprisingly it remains challenging to segment robots. This is due to robot embodiment diversity, appearance ambiguity, structural complexity, and rapid shape changes. Embracing these challenges, we introduce RobotSeg, a foundation model for robot segmentation in image and video. RobotSeg is built upon the versatile SAM 2 foundation model but addresses its three limitations for robot segmentation, namely the lack of adaptation to articulated robots, reliance on manual prompts, and the need for per-frame training mask annotations, by introducing a structure-enhanced memory associator, a robot prompt generator, and a label-efficient training strategy. These innovations collectively enable a structure-aware, automatic, and label-efficient solution. We further construct the video robot segmentation (VRS) dataset comprising over 2.8k videos (138k frames) with diverse robot embodiments and environments. Extensive experiments demonstrate that RobotSeg achieves state-of-the-art performance on both images and videos, establishing a strong foundation for future advances in robot perception.
>
---
#### [replaced 013] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VideoARM，用于长视频理解任务，解决传统方法依赖预处理和高token消耗的问题，通过自适应推理与分层记忆机制提升效果并降低资源消耗。**

- **链接: [https://arxiv.org/pdf/2512.12360](https://arxiv.org/pdf/2512.12360)**

> **作者:** Yufei Yin; Qianke Meng; Minghao Chen; Jiajun Ding; Zhenwei Shao; Zhou Yu
>
> **备注:** Accepted to CVPR 2026, code available at this https URL
>
> **摘要:** Long-form video understanding remains challenging due to the extended temporal structure and dense multimodal cues. Despite recent progress, many existing approaches still rely on hand-crafted reasoning pipelines or employ token-consuming video preprocessing to guide MLLMs in autonomous reasoning. To overcome these limitations, we introduce VideoARM, an Agentic Reasoning-over-hierarchical-Memory paradigm for long-form video understanding. Instead of static, exhaustive preprocessing, VideoARM performs adaptive, on-the-fly agentic reasoning and memory construction. Specifically, VideoARM performs an adaptive and continuous loop of observing, thinking, acting, and memorizing, where a controller autonomously invokes tools to interpret the video in a coarse-to-fine manner, thereby substantially reducing token consumption. In parallel, a hierarchical multimodal memory continuously captures and updates multi-level clues throughout the operation of the agent, providing precise contextual information to support the controller in decision-making. Experiments on prevalent benchmarks demonstrate that VideoARM outperforms the state-of-the-art method, DVD, while significantly reducing token consumption for long-form videos.
>
---
#### [replaced 014] Synergizing Deep Learning and Biological Heuristics for Extreme Long-Tail White Blood Cell Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16249](https://arxiv.org/pdf/2603.16249)**

> **作者:** Duc T. Nguyen; Hoang-Long Nguyen; Huy-Hieu Pham
>
> **备注:** Accepted at IEEE ISBI 2026
>
> **摘要:** Automated white blood cell (WBC) classification is essential for leukemia screening but remains challenged by extreme class imbalance, long-tail distributions, and domain shift, leading deep models to overfit dominant classes and fail on rare subtypes. We propose a hybrid framework for rare-class generalization that integrates a generative Pix2Pix-based restoration module for artifact removal, a Swin Transformer ensemble with MedSigLIP contrastive embeddings for robust representation learning, and a biologically-inspired refinement step using geometric spikiness and Mahalanobis-based morphological constraints to recover out-of-distribution predictions. Evaluated on the WBCBench 2026 challenge, our method achieves a Macro-F1 of 0.77139 on the private leaderboard, demonstrating strong performance under severe imbalance and highlighting the value of incorporating biological priors into deep learning for hematological image analysis. The code is available at this https URL
>
---
#### [replaced 015] SAGE: Style-Adaptive Generalization for Privacy-Constrained Semantic Segmentation Across Domains
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02369](https://arxiv.org/pdf/2512.02369)**

> **作者:** Qingmei Li; Yang Zhang; Peifeng Zhang; Haohuan Fu; Juepeng Zheng
>
> **摘要:** Domain generalization for semantic segmentation aims to mitigate the degradation in model performance caused by domain shifts. However, in many real-world scenarios, we are unable to access the model parameters and architectural details due to privacy concerns and security constraints. Traditional fine-tuning or adaptation is hindered, leading to the demand for input-level strategies that can enhance generalization without modifying model weights. To this end, we propose a \textbf{S}tyle-\textbf{A}daptive \textbf{GE}neralization framework (\textbf{SAGE}), which improves the generalization of frozen models under privacy constraints. SAGE learns to synthesize visual prompts that implicitly align feature distributions across styles instead of directly fine-tuning the backbone. Specifically, we first utilize style transfer to construct a diverse style representation of the source domain, thereby learning a set of style characteristics that can cover a wide range of visual features. Then, the model adaptively fuses these style cues according to the visual context of each input, forming a dynamic prompt that harmonizes the image appearance without touching the interior of the model. Through this closed-loop design, SAGE effectively bridges the gap between frozen model invariance and the diversity of unseen domains. Extensive experiments on five benchmark datasets demonstrate that SAGE achieves competitive or superior performance compared to state-of-the-art methods under privacy constraints and outperforms full fine-tuning baselines in all settings.
>
---
#### [replaced 016] NARVis: Neural Accelerated Rendering for Real-Time Scientific Point Cloud Visualization
- **分类: cs.GR; cs.CV; cs.HC; cs.LG**

- **链接: [https://arxiv.org/pdf/2407.19097](https://arxiv.org/pdf/2407.19097)**

> **作者:** Srinidhi Hegde; Kaur Kullman; Thomas Grubb; Leslie Lait; Stephen Guimond; Matthias Zwicker
>
> **摘要:** Exploring scientific datasets with billions of samples in real-time visualization presents a challenge - balancing high-fidelity rendering with speed. This work introduces a neural accelerated renderer, NARVis, that uses the neural deferred rendering framework to visualize large-scale scientific point cloud data. NARVis augments a real-time point cloud rendering pipeline with high-quality neural post-processing, making the approach ideal for interactive visualization at scale. Specifically, we render the multi-attribute point cloud using a high-performance multi-attribute rasterizer and train a neural renderer to capture the desired post-processing effects from a conventional high-quality renderer. NARVis is effective in visualizing complex multidimensional Lagrangian flow fields and photometric scans of a large terrain as compared to the state-of-the-art high-quality renderers. Extensive evaluations demonstrate that NARVis prioritizes speed and scalability while retaining high visual fidelity. We achieve competitive frame rates of $>$126 fps for interactive rendering of $>$350M points (i.e., an effective throughput of $>$44 billion points per second) using ~12 GB of memory on RTX 2080 Ti GPU. Furthermore, NARVis is generalizable across different point clouds with similar visualization needs and the desired post-processing effects could be obtained with substantial high quality even at lower resolutions of the original point cloud, further reducing the memory requirements.
>
---
#### [replaced 017] Robust Ego-Exo Correspondence with Long-Term Memory
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.11417](https://arxiv.org/pdf/2510.11417)**

> **作者:** Yijun Hu; Bing Fan; Xin Gu; Haiqing Ren; Dongfang Liu; Heng Fan; Libo Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Establishing object-level correspondence between egocentric and exocentric views is essential for intelligent assistants to deliver precise and intuitive visual guidance. However, this task faces numerous challenges, including extreme viewpoint variations, occlusions, and the presence of small objects. Existing approaches usually borrow solutions from video object segmentation models, but still suffer from the aforementioned challenges. Recently, the Segment Anything Model 2 (SAM 2) has shown strong generalization capabilities and excellent performance in video object segmentation. Yet, when simply applied to the ego-exo correspondence (EEC) task, SAM 2 encounters severe difficulties due to ineffective ego-exo feature fusion and limited long-term memory capacity, especially for long videos. Addressing these problems, we propose a novel EEC framework based on SAM 2 with long-term memories by presenting a dual-memory architecture and an adaptive feature routing module inspired by Mixture-of-Experts (MoE). Compared to SAM 2, our approach features (i) a Memory-View MoE module which consists of a dual-branch routing mechanism to adaptively assign contribution weights to each expert feature along both channel and spatial dimensions, and (ii) a dual-memory bank system with a simple yet effective compression strategy to retain critical long-term information while eliminating redundancy. In the extensive experiments on the challenging EgoExo4D benchmark, our method, dubbed LM-EEC, achieves new state-of-the-art results and significantly outperforms existing methods and the SAM 2 baseline, showcasing its strong generalization across diverse scenarios. Our code and model are available at this https URL.
>
---
#### [replaced 018] Mind-of-Director: Multi-modal Agent-Driven Film Previsualization via Collaborative Decision-Making
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14790](https://arxiv.org/pdf/2603.14790)**

> **作者:** Shufeng Nan; Mengtian Li; Sixiao Zheng; Yuwei Lu; Han Zhang; Yanwei Fu
>
> **摘要:** We present Mind-of-Director, a multi-modal agent-driven framework for film previz that models the collaborative decision-making process of a film production team. Given a creative idea, Mind-of-Director orchestrates multiple specialized agents to produce previz sequences within the game engine. The framework consists of four cooperative modules: Script Development, where agents draft and refine the screenplay iteratively; Virtual Scene Design, which transforms text into semantically aligned 3D environments; Character Behaviour Control, which determines character blocking and motion; and Camera Planning, which optimizes framing, movement, and composition for cinematic camera effects. A real-time visual editing system built in the game engine further enables interactive inspection and synchronized timeline adjustment across scenes, behaviours, and cameras. Extensive experiments and human evaluations show that Mind-of-Director generates high-quality, semantically grounded previz sequences in approximately 25 minutes per idea, demonstrating the effectiveness of agent collaboration for both automated prototyping and human-in-the-loop filmmaking.
>
---
#### [replaced 019] M4Human: A Large-Scale Multimodal mmWave Radar Benchmark for Human Mesh Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12378](https://arxiv.org/pdf/2512.12378)**

> **作者:** Junqiao Fan; Yunjiao Zhou; Yizhuo Yang; Xinyuan Cui; Jiarui Zhang; Lihua Xie; Jianfei Yang; Chris Xiaoxuan Lu; Fangqiang Ding
>
> **摘要:** Human mesh reconstruction (HMR) provides direct insights into body-environment interaction, which enables various immersive applications. While existing large-scale HMR datasets rely heavily on line-of-sight RGB input, vision-based sensing is limited by occlusion, lighting variation, and privacy concerns. To overcome these limitations, recent efforts have explored radio-frequency (RF) mmWave radar for privacy-preserving indoor human sensing. However, current radar datasets are constrained by sparse skeleton labels, limited scale, and simple in-place actions. To advance the HMR research community, we introduce M4Human, the current largest-scale (661K-frame) ($9\times$ prior largest) multimodal benchmark, featuring high-resolution mmWave radar, RGB, and depth data. M4Human provides both raw radar tensors (RT) and processed radar point clouds (RPC) to enable research across different levels of RF signal granularity. M4Human includes high-quality motion capture (MoCap) annotations with 3D meshes and global trajectories, and spans 20 subjects and 50 diverse actions, including in-place, sit-in-place, and free-space sports or rehabilitation movements. We establish benchmarks on both RT and RPC modalities, as well as multimodal fusion with RGB-D modalities. Extensive results highlight the significance of M4Human for radar-based human modeling while revealing persistent challenges under fast, unconstrained motion. The dataset and code will be released after the paper publication.
>
---
#### [replaced 020] NeAR: Coupled Neural Asset-Renderer Stack
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18600](https://arxiv.org/pdf/2511.18600)**

> **作者:** Hong Li; Chongjie Ye; Houyuan Chen; Weiqing Xiao; Ziyang Yan; Lixing Xiao; Zhaoxi Chen; Jianfeng Xiang; Shaocong Xu; Xuhui Liu; Yikai Wang; Baochang Zhang; Xiaoguang Han; Jiaolong Yang; Hao Zhao
>
> **备注:** Accepted by CVPR 2026. The project page: this https URL
>
> **摘要:** Neural asset authoring and neural rendering have traditionally evolved as disjoint paradigms: one generates digital assets for fixed graphics pipelines, while the other maps conventional assets to images. However, treating them as independent entities limits the potential for end-to-end optimization in fidelity and consistency. In this paper, we bridge this gap with NeAR, a Coupled Neural Asset--Renderer Stack. We argue that co-designing the asset representation and the renderer creates a robust "contract" for superior generation. On the asset side, we introduce the Lighting-Homogenized SLAT (LH-SLAT). Leveraging a rectified-flow model, NeAR lifts casually lit single images into a canonical, illumination-invariant latent space, effectively suppressing baked-in shadows and highlights. On the renderer side, we design a lighting-aware neural decoder tailored to interpret these homogenized latents. Conditioned on HDR environment maps and camera views, it synthesizes relightable 3D Gaussian splats in real-time without per-object optimization. We validate NeAR on four tasks: (1) G-buffer-based forward rendering, (2) random-lit reconstruction, (3) unknown-lit relighting, and (4) novel-view relighting. Extensive experiments demonstrate that our coupled stack outperforms state-of-the-art baselines in both quantitative metrics and perceptual quality. We hope this coupled asset-renderer perspective inspires future graphics stacks that view neural assets and renderers as co-designed components instead of independent entities.
>
---
#### [replaced 021] Memory-Augmented Vision-Language Agents for Persistent and Semantically Consistent Object Captioning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24257](https://arxiv.org/pdf/2603.24257)**

> **作者:** Tommaso Galliena; Stefano Rosa; Tommaso Apicella; Pietro Morerio; Alessio Del Bue; Lorenzo Natale
>
> **备注:** 24 pages, 7 figures, 7 tables (including Supplementary Materials)
>
> **摘要:** Vision-Language Models (VLMs) often yield inconsistent descriptions of the same object across viewpoints, hindering the ability of embodied agents to construct consistent semantic representations over time. Previous methods resolved inconsistencies using offline multi-view aggregation or multi-stage pipelines that decouple exploration, data association, and caption learning, with limited capacity to reason over previously observed objects. In this paper, we introduce a unified, memory-augmented Vision-Language agent that simultaneously handles data association, object captioning, and exploration policy within a single autoregressive framework. The model processes the current RGB observation, a top-down explored map, and an object-level episodic memory serialized into object-level tokens, ensuring persistent object identity and semantic consistency across extended sequences. To train the model in a self-supervised manner, we collect a dataset in photorealistic 3D environments using a disagreement-based policy and a pseudo-captioning model that enforces consistency across multi-view caption histories. Extensive evaluation on a manually annotated object-level test set, demonstrate improvements of up to +11.86% in standard captioning scores and +7.39% in caption self-similarity over baseline models, while enabling scalable performance through a compact scene representation. Code, model weights, and data are available at this https URL.
>
---
#### [replaced 022] Vision-Language Agents for Interactive Forest Change Analysis
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于遥感图像变化分析任务，旨在解决森林动态的像素级变化检测与语义描述问题。提出一种基于大语言模型的视觉语言代理系统，并构建了Forest-Change数据集进行验证。**

- **链接: [https://arxiv.org/pdf/2601.04497](https://arxiv.org/pdf/2601.04497)**

> **作者:** James Brock; Ce Zhang; Nantheera Anantrasirichai
>
> **备注:** 5 pages, 4 figures, Accepted into IGARSS 2026
>
> **摘要:** Modern forest monitoring workflows increasingly benefit from the growing availability of high-resolution satellite imagery and advances in deep learning. Two persistent challenges in this context are accurate pixel-level change detection and meaningful semantic change captioning for complex forest dynamics. While large language models (LLMs) are being adapted for interactive data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored. To address this gap, we introduce an LLM-driven agent for integrated forest change analysis that supports natural language querying across multiple RSICI tasks. The proposed system builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration. To facilitate adaptation and evaluation in forest environments, we further introduce the Forest-Change dataset, which comprises bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated using a combination of human annotation and rule-based methods. Experimental results show that the proposed system achieves mIoU and BLEU-4 scores of 67.10% and 40.17% on the Forest-Change dataset, and 88.13% and 34.41% on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI benchmark for joint change detection and captioning. These results highlight the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and efficiency of forest change analysis. All data and code are publicly available at this https URL.
>
---
#### [replaced 023] ParaUni: Enhance Generation in Unified Multimodal Model with Reinforcement-driven Hierarchical Parallel Information Interaction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.05422](https://arxiv.org/pdf/2512.05422)**

> **作者:** Jiangtong Tan; Lin Liu; Jie Huanng; Xiaopeng Zhang; Qi Tian; Feng Zhao
>
> **摘要:** Unified multimodal models significantly improve visual generation by combining vision-language models (VLMs) with diffusion models. However, existing methods struggle to fully balance sufficient interaction and flexible implementation due to vast representation difference. Considering abundant and hierarchical information in VLM's layers from low-level details to high-level semantics, we propose \textbf{ParaUni}. It extracts features from variants VLM's layers in a \textbf{Para}llel way for comprehensive information interaction and retains a flexible separation architecture to enhance generation in \textbf{Uni}fied multimodal model. Concretely, visual features from all VLM's layers are fed in parallel into a Layer Integration Module (LIM), which efficiently integrates fine-grained details and semantic abstractions and provides the fused representation as a condition to the diffusion model. To further enhance performance, we reveal that these hierarchical layers respond unequally to different rewards in Reinforcement Learning (RL). Crucially, we design a Layer-wise Dynamic Adjustment Mechanism (LDAM) to facilitate multiple reward improvements that aligns the hierarchical properties of these layers using RL. Extensive experiments show ParaUni leverages complementary multi-layer features to substantially improve generation quality and shows strong potential for multiple reward advances during RL stages. Code is available at this https URL.
>
---
#### [replaced 024] SAGE: Training Smart Any-Horizon Agents for Long Video Reasoning with Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13874](https://arxiv.org/pdf/2512.13874)**

> **作者:** Jitesh Jain; Jialuo Li; Zixian Ma; Jieyu Zhang; Chris Dongjoo Kim; Sangho Lee; Rohun Tripathi; Tanmay Gupta; Christopher Clark; Humphrey Shi
>
> **备注:** Project Page: this https URL
>
> **摘要:** As humans, we are natural any-horizon reasoners, i.e., we can decide whether to iteratively skim long videos or watch short ones in full when necessary for a given task. With this in mind, one would expect video reasoning models to reason flexibly across different durations. However, SOTA models are still trained to predict answers in a single turn while processing a large number of frames, akin to watching an entire long video, requiring significant resources. This raises the question: Is it possible to develop performant any-horizon video reasoning systems? Inspired by human behavior, we first propose SAGE, an agent system that performs multi-turn reasoning on long videos while handling simpler problems in a single turn. Secondly, we introduce an easy synthetic data generation pipeline using Gemini-2.5-Flash to train the orchestrator, SAGE-MM, which lies at the core of SAGE. We further propose an effective RL post-training recipe essential for instilling any-horizon reasoning ability in SAGE-MM. Thirdly, we curate SAGE-Bench with an average duration of greater than 700 seconds for evaluating video reasoning ability in real-world entertainment use cases. Lastly, we empirically validate the effectiveness of our system, data, and RL recipe, observing notable improvements of up to 6.1% on open-ended video reasoning tasks, as well as an impressive 8.2% improvement on videos longer than 10 minutes.
>
---
#### [replaced 025] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言生成任务，旨在解决3D物体与文本指令间语义不一致的问题。提出AffordGrasp框架，结合扩散模型与语义标注，生成物理合理且语义准确的抓取姿态。**

- **链接: [https://arxiv.org/pdf/2603.08021](https://arxiv.org/pdf/2603.08021)**

> **作者:** Xiaofei Wu; Yi Zhang; Yumeng Liu; Yuexin Ma; Yujiao Shi; Xuming He
>
> **备注:** CVPR 2026
>
> **摘要:** Generating human grasping poses that accurately reflect both object geometry and user-specified interaction semantics is essential for natural hand-object interactions in AR/VR and embodied AI. However, existing semantic grasping approaches struggle with the large modality gap between 3D object representations and textual instructions, and often lack explicit spatial or semantic constraints, leading to physically invalid or semantically inconsistent grasps. In this work, we present AffordGrasp, a diffusion-based framework that produces physically stable and semantically faithful human grasps with high precision. We first introduce a scalable annotation pipeline that automatically enriches hand-object interaction datasets with fine-grained structured language labels capturing interaction intent. Building upon these annotations, AffordGrasp integrates an affordance-aware latent representation of hand poses with a dual-conditioning diffusion process, enabling the model to jointly reason over object geometry, spatial affordances, and instruction semantics. A distribution adjustment module further enforces physical contact consistency and semantic alignment. We evaluate AffordGrasp across four instruction-augmented benchmarks derived from HO-3D, OakInk, GRAB, and AffordPose, and observe substantial improvements over state-of-the-art methods in grasp quality, semantic accuracy, and diversity.
>
---
#### [replaced 026] VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.22674](https://arxiv.org/pdf/2601.22674)**

> **作者:** Hanxun Yu; Wentong Li; Xuan Qu; Song Wang; Junbo Chen; Jianke Zhu
>
> **备注:** ICLR2026, Code Link: this https URL
>
> **摘要:** Multimodal large language models (MLLMs) suffer from high computational costs due to excessive visual tokens, particularly in high-resolution and video-based scenarios. Existing token reduction methods typically focus on isolated pipeline components and often neglect textual alignment, leading to performance degradation. In this paper, we propose VisionTrim, a unified framework for training-free MLLM acceleration, integrating two effective plug-and-play modules: 1) the Dominant Vision Token Selection (DVTS) module, which preserves essential visual tokens via a global-local view, and 2) the Text-Guided Vision Complement (TGVC) module, which facilitates context-aware token merging guided by textual cues. Extensive experiments across diverse image and video multimodal benchmarks demonstrate the performance superiority of our VisionTrim, advancing practical MLLM deployment in real-world applications. The code is available at: this https URL.
>
---
#### [replaced 027] Minimizing the Pretraining Gap: Domain-aligned Text-Based Person Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.10195](https://arxiv.org/pdf/2507.10195)**

> **作者:** Shuyu Yang; Yaxiong Wang; Yongrui Li; Li Zhu; Zhedong Zheng
>
> **摘要:** In this work, we focus on text-based person retrieval, which identifies individuals based on textual descriptions. Despite advancements enabled by synthetic data for pretraining, a significant domain gap, due to variations in lighting, color, and viewpoint, limits the effectiveness of the pretrain-finetune paradigm. To overcome this issue, we propose a unified pipeline incorporating domain adaptation at both image and region levels. Our method features two key components: Domain-aware Diffusion (DaD) for image-level adaptation, which aligns image distributions between synthetic and real-world domains, e.g., CUHK-PEDES, and Multi-granularity Relation Alignment (MRA) for region-level adaptation, which aligns visual regions with descriptive sentences, thereby addressing disparities at a finer granularity. This dual-level strategy effectively bridges the domain gap, achieving state-of-the-art performance on CUHK-PEDES, ICFG-PEDES, and RSTPReid datasets. The dataset, model, and code are available at this https URL.
>
---
#### [replaced 028] WikiCLIP: An Efficient Contrastive Baseline for Open-domain Visual Entity Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09921](https://arxiv.org/pdf/2603.09921)**

> **作者:** Shan Ning; Longtian Qiu; Jiaxuan Sun; Xuming He
>
> **备注:** Accepted by CVPR26, codes and weights are publicly available
>
> **摘要:** Open-domain visual entity recognition (VER) seeks to associate images with entities in encyclopedic knowledge bases such as Wikipedia. Recent generative methods tailored for VER demonstrate strong performance but incur high computational costs, limiting their scalability and practical deployment. In this work, we revisit the contrastive paradigm for VER and introduce WikiCLIP, a simple yet effective framework that establishes a strong and efficient baseline for open-domain VER. WikiCLIP leverages large language model embeddings as knowledge-rich entity representations and enhances them with a Vision-Guided Knowledge Adaptor (VGKA) that aligns textual semantics with visual cues at the patch level. To further encourage fine-grained discrimination, a Hard Negative Synthesis Mechanism generates visually similar but semantically distinct negatives during training. Experimental results on popular open-domain VER benchmarks, such as OVEN, demonstrate that WikiCLIP significantly outperforms strong baselines. Specifically, WikiCLIP achieves a 16\% improvement on the challenging OVEN unseen set, while reducing inference latency by nearly 100 times compared with the leading generative model, AutoVER. The project page is available at this https URL
>
---
#### [replaced 029] DriveVGGT: Calibration-Constrained Visual Geometry Transformers for Multi-Camera Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22264](https://arxiv.org/pdf/2511.22264)**

> **作者:** Xiaosong Jia; Yanhao Liu; Yu Hong; Renqiu Xia; Junqi You; Bin Sun; Zhihui Hao; Junchi Yan
>
> **摘要:** Feed-forward reconstruction has been progressed rapidly, with the Visual Geometry Grounded Transformer (VGGT) being a notable baseline. However, directly applying VGGT to autonomous driving (AD) fails to capture three domain-specific priors: (i) Sparse Spatial Overlap: the overlap among mutli-view cameras is minimal due to $360^{\circ}$ coverage requirements under budget control, which renders global attention among all images inefficient; (ii) Calibrated Geometric Constraints: the absolute distance among cameras is generally accessible for AD data with calibration process before driving. Standard VGGT is unable to directly utilize such information for absolute scale scene reconstruction; (iii) Rigid Extrinsic Constancy: relative poses of multi-view cameras are approximately static, i.e., the ego-motion is the same for all cameras. To bridge these gaps, we propose DriveVGGT, a scale-aware reconstruction framework that explicitly integrates these priors through three targeted components. First, for the Sparse Spatial Overlap in (i), we introduce a Temporal Video Attention (TVA) module to process multi-camera videos independently. Second, for Calibrated Geometric Constraints in (ii), a Multi-camera Consistency Attention (MCA) module is designed to directly utilize the calibration information among cameras with a scale head for absolute scale scene reconstruction. Finally, to utilize Rigid Extrinsic Constancy in (iii), we reformulate the decoding process of VGGT into factorized sequential pose head and ego motion head. On AD datasets, experiments demonstrate that DriveVGGT reduces inference time by 49.3\% while improving depth and pose estimation compared to vanilla VGGT in long-sequence scenarios. It consistently outperforms recent SOTA variants. Meanwhile, extensive ablation studies verify the effectiveness of each devised module.
>
---
#### [replaced 030] RadImageNet-VQA: A Large-Scale CT and MRI Dataset for Radiologic Visual Question Answering
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出RadImageNet-VQA数据集，用于解决医学影像中的视觉问答任务。针对现有数据集规模小、依赖文本等问题，该数据集包含大量CT和MRI图像及问答对，涵盖多种病理识别任务。**

- **链接: [https://arxiv.org/pdf/2512.17396](https://arxiv.org/pdf/2512.17396)**

> **作者:** Léo Butsanets; Charles Corbière; Julien Khlaut; Pierre Manceron; Corentin Dancette
>
> **备注:** Preprint, 33 pages, 15 figures, 11 tables
>
> **摘要:** In this work, we introduce RadImageNet-VQA, a large-scale dataset designed to advance radiologic visual question answering (VQA) on CT and MRI exams. Existing medical VQA datasets are limited in scale, dominated by X-ray imaging or biomedical illustrations, and often prone to text-based shortcuts. RadImageNet-VQA is built from expert-curated annotations and provides 750K images paired with 7.5M question-answer samples. It covers three key tasks - abnormality detection, anatomy recognition, and pathology identification - spanning eight anatomical regions and 97 pathology categories, and supports open-ended, closed-ended, and multiple-choice questions. Extensive experiments show that state-of-the-art vision-language models still struggle with fine-grained pathology identification, particularly in open-ended settings and even after fine-tuning. Text-only analysis further reveals that model performance collapses to near-random without image inputs, confirming that RadImageNet-VQA is free from linguistic shortcuts. The full dataset and benchmark are publicly available at this https URL.
>
---
#### [replaced 031] Gesture-Aware Pretraining and Token Fusion for 3D Hand Pose Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.17396](https://arxiv.org/pdf/2603.17396)**

> **作者:** Rui Hong; Jana Kosecka
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** Estimating 3D hand pose from monocular RGB images is fundamental for applications in AR/VR, human-computer interaction, and sign language understanding. In this work we focus on a scenario where a discrete set of gesture labels is available and show that gesture semantics can serve as a powerful inductive bias for 3D pose estimation. We present a two-stage framework: gesture-aware pretraining that learns an informative embedding space using coarse and fine gesture labels from InterHand2.6M, followed by a per-joint token Transformer guided by gesture embeddings as intermediate representations for final regression of MANO hand parameters. Training is driven by a layered objective over parameters, joints, and structural constraints. Experiments on InterHand2.6M demonstrate that gesture-aware pretraining consistently improves single-hand accuracy over the state-of-the-art EANet baseline, and that the benefit transfers across architectures without any modification.
>
---
#### [replaced 032] Learning Underwater Active Perception in Simulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于 underwater active perception 任务，旨在解决水下视觉质量受浊度和散射影响的问题。通过构建合成数据集和MLP模型，提升不同水况下的图像质量与可视覆盖。**

- **链接: [https://arxiv.org/pdf/2504.17817](https://arxiv.org/pdf/2504.17817)**

> **作者:** Alexandre Cardaillac; Donald G. Dansereau
>
> **摘要:** When employing underwater vehicles for the autonomous inspection of assets, it is crucial to consider and assess the water conditions. These conditions significantly impact visibility and directly affect robotic operations. Turbidity can jeopardise the mission by preventing accurate visual documentation of inspected structures. Previous works have introduced methods to adapt to turbidity and backscattering, however, they also include manoeuvring and setup constraints. We propose a simple yet efficient approach to enable high-quality image acquisition of assets in a broad range of water conditions. This active perception framework includes a multi-layer perceptron (MLP) trained to predict image quality given a distance to a target and artificial light intensity. We generate a large synthetic dataset that includes ten water types with varying levels of turbidity and backscattering. For this, we modified the modelling software Blender to better account for the underwater light propagation properties. We validated the approach in simulation and demonstrate significant improvements in visual coverage and image quality compared to traditional methods. The project code is available on our project page at this https URL.
>
---
#### [replaced 033] What-Meets-Where: Unified Learning of Action and Contact Localization in Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.09428](https://arxiv.org/pdf/2508.09428)**

> **作者:** Yuxiao Wang; Yu Lei; Wolin Liang; Weiying Xue; Zhenao Wei; Nan Zhuang; Qi Liu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** People control their bodies to establish contact with the environment. To comprehensively understand actions across diverse visual contexts, it is essential to simultaneously consider \textbf{what} action is occurring and \textbf{where} it is happening. Current methodologies, however, often inadequately capture this duality, typically failing to jointly model both action semantics and their spatial contextualization within scenes. To bridge this gap, we introduce a novel vision task that simultaneously predicts high-level action semantics and fine-grained body-part contact regions. Our proposed framework, PaIR-Net, comprises three key components: the Contact Prior Aware Module (CPAM) for identifying contact-relevant body parts, the Prior-Guided Concat Segmenter (PGCS) for pixel-wise contact segmentation, and the Interaction Inference Module (IIM) responsible for integrating global interaction relationships. To facilitate this task, we present PaIR (Part-aware Interaction Representation), a comprehensive dataset containing 13,979 images that encompass 654 actions, 80 object categories, and 17 body parts. Experimental evaluation demonstrates that PaIR-Net significantly outperforms baseline approaches, while ablation studies confirm the efficacy of each architectural component. The code and dataset will be released upon publication.
>
---
#### [replaced 034] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [https://arxiv.org/pdf/2512.22065](https://arxiv.org/pdf/2512.22065)**

> **作者:** Zhiyao Sun; Ziqiao Peng; Yifeng Ma; Yi Chen; Zhengguang Zhou; Zixiang Zhou; Guozhen Zhang; Youliang Zhang; Yuan Zhou; Qinglin Lu; Yong-Jin Liu
>
> **备注:** Accepted by CVPR 2026. Project page: this https URL
>
> **摘要:** Real-time, streaming interactive avatars represent a critical yet challenging goal in digital human research. Although diffusion-based human avatar generation methods achieve remarkable success, their non-causal architecture and high computational costs make them unsuitable for streaming. Moreover, existing interactive approaches are typically restricted to the head-and-shoulder region, limiting their ability to produce gestures and body motions. To address these challenges, we propose a two-stage autoregressive adaptation and acceleration framework that applies autoregressive distillation and adversarial refinement to adapt a high-fidelity human video diffusion model for real-time, interactive streaming. To ensure long-term stability and consistency, we introduce three key components: a Reference Sink, a Reference-Anchored Positional Re-encoding (RAPR) strategy, and a Consistency-Aware Discriminator. Building on this framework, we develop a one-shot, interactive, human avatar model capable of generating both natural talking and listening behaviors with coherent gestures. Extensive experiments demonstrate that our method achieves state-of-the-art performance, surpassing existing approaches in generation quality, real-time efficiency, and interaction naturalness. Project page: this https URL .
>
---
#### [replaced 035] CoPE-VideoLM: Leveraging Codec Primitives For Efficient Video Language Modeling
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决VideoLM中关键帧采样不全和计算冗余问题。通过利用视频编解码器原语，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.13191](https://arxiv.org/pdf/2602.13191)**

> **作者:** Sayan Deb Sarkar; Rémi Pautrat; Ondrej Miksik; Marc Pollefeys; Iro Armeni; Mahdi Rad; Mihai Dusmanu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Video Language Models (VideoLMs) enable AI systems to understand temporal dynamics in videos. To fit within the maximum context window constraint, current methods use keyframe sampling which often misses both macro-level events and micro-level details due to the sparse temporal coverage. Furthermore, processing full images and their tokens for each frame incurs substantial computational overhead. We address these limitations by leveraging video codec primitives (specifically motion vectors and residuals) which natively encode video redundancy and sparsity without requiring expensive full-image encoding for most frames. To this end, we introduce lightweight transformer-based encoders that aggregate codec primitives and align their representations with image encoder embeddings through a pre-training strategy that accelerates convergence during end-to-end fine-tuning. Our approach, CoPE-VideoLM, reduces the time-to-first-token by up to 86% and token usage by up to 93% compared to standard VideoLMs. Moreover, by varying the keyframe and codec primitive densities we maintain or exceed performance on 14 diverse video understanding benchmarks spanning general question answering, temporal and motion reasoning, long-form understanding, and spatial scene understanding.
>
---
#### [replaced 036] ControlGUI: Guiding Generative GUI Exploration through Perceptual Visual Flow
- **分类: cs.HC; cs.AI; cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2502.03330](https://arxiv.org/pdf/2502.03330)**

> **作者:** Aryan Garg; Yue Jiang; Antti Oulasvirta
>
> **摘要:** During the early stages of interface design, designers need to produce multiple sketches to explore a design space. Design tools often fail to support this critical stage, because they insist on specifying more details than necessary. Although recent advances in generative AI have raised hopes of solving this issue, in practice they fail because expressing loose ideas in a prompt is impractical. In this paper, we propose a diffusion-based approach to the low-effort generation of interface sketches. It breaks new ground by allowing flexible control of the generation process via three types of inputs: A) prompts, B) wireframes, and C) visual flows. The designer can provide any combination of these as input at any level of detail, and will get a diverse gallery of low-fidelity solutions in response. The unique benefit is that large design spaces can be explored rapidly with very little effort in input-specification. We present qualitative results for various combinations of input specifications. Additionally, we demonstrate that our model aligns more accurately with these specifications than other models.
>
---
#### [replaced 037] FontCrafter: High-Fidelity Element-Driven Artistic Font Creation with Visual In-Context Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.22054](https://arxiv.org/pdf/2603.22054)**

> **作者:** Wuyang Luo; Chengkai Tan; Chang Ge; Binye Hong; Su Yang; Yongjiu Ma
>
> **备注:** To appear in CVPR 2026
>
> **摘要:** Artistic font generation aims to synthesize stylized glyphs based on a reference style. However, existing approaches suffer from limited style diversity and coarse control. In this work, we explore the potential of element-driven artistic font generation. Elements are the fundamental visual units of a font, serving as reference images for the desired style. Conceptually, we categorize elements into object elements (e.g., flowers or stones) with distinct structures and amorphous elements (e.g., flames or clouds) with unstructured textures. We introduce FontCrafter, an element-driven framework for font creation, and construct a large-scale dataset, ElementFont, which contains diverse element types and high-quality glyph images. However, achieving high-fidelity reconstruction of both texture and structure of reference elements remains challenging. To address this, we propose an in-context generation strategy that treats element images as visual context and uses an inpainting model to transfer element styles into glyph regions at the pixel level. To further control glyph shapes, we design a lightweight Context-aware Mask Adapter (CMA) that injects shape information. Moreover, a training-free attention redirection mechanism enables region-aware style control and suppresses stroke hallucination. In addition, edge repainting is applied to make boundaries more natural. Extensive experiments demonstrate that FontCrafter achieves strong zero-shot generation performance, particularly in preserving structural and textural fidelity, while also supporting flexible controls such as style mixture.
>
---
#### [replaced 038] Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation
- **分类: cs.CV; cs.AI; cs.GR; cs.HC; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.02190](https://arxiv.org/pdf/2603.02190)**

> **作者:** Divyanshu Daiya; Aniket Bera
>
> **备注:** Accepted to CVPR 2026 Main Conference (11 pages, 8 figures)
>
> **摘要:** We present Sketch2Colab, which turns storyboard-style 2D sketches into coherent, object-aware 3D multi-human motion with fine-grained control over agents, joints, timing, and contacts. Diffusion-based motion generators offer strong realism but often rely on costly guidance for multi-entity control and degrade under strong conditioning. Sketch2Colab instead learns a sketch-conditioned diffusion prior and distills it into a rectified-flow student in latent space for fast, stable sampling. To make motion follow storyboards closely, we guide the student with differentiable objectives that enforce keyframes, paths, contacts, and physical consistency. Collaborative motion naturally involves discrete changes in interaction, such as converging, forming contact, cooperative transport, or disengaging, and a continuous flow alone struggles to sequence these shifts cleanly. We address this with a lightweight continuous-time Markov chain (CTMC) planner that tracks the active interaction regime and modulates the flow to produce clearer, synchronized coordination in human-object-human motion. Experiments on CORE4D and InterHuman show that Sketch2Colab outperforms baselines in constraint adherence and perceptual quality while sampling substantially faster than diffusion-only alternatives.
>
---
#### [replaced 039] OccuFly: A 3D Vision Benchmark for Semantic Scene Completion from the Aerial Perspective
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.20770](https://arxiv.org/pdf/2512.20770)**

> **作者:** Markus Gross; Sai B. Matha; Aya Fahmy; Rui Song; Daniel Cremers; Henri Meess
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Semantic Scene Completion (SSC) is essential for 3D perception in mobile robotics, as it enables holistic scene understanding by jointly estimating dense volumetric occupancy and per-voxel semantics. Although SSC has been widely studied in terrestrial domains such as autonomous driving, aerial settings like autonomous flying remain largely unexplored, thereby limiting progress on downstream applications. Furthermore, LiDAR sensors are the primary modality for SSC data generation, which poses challenges for most uncrewed aerial vehicles (UAVs) due to flight regulations, mass and energy constraints, and the sparsity of LiDAR point clouds from elevated viewpoints. To address these limitations, we propose a LiDAR-free, camera-based data generation framework. By leveraging classical 3D reconstruction, our framework automates semantic label transfer by lifting <10% of annotated images into the reconstructed point cloud, substantially minimizing manual 3D annotation effort. Based on this framework, we introduce OccuFly, the first real-world, camera-based aerial SSC benchmark, captured across multiple altitudes and all seasons. OccuFly provides over 20,000 samples of images, semantic voxel grids, and metric depth maps across 21 semantic classes in urban, industrial, and rural environments, and follows established data organization for seamless integration. We benchmark both SSC and metric monocular depth estimation on OccuFly, revealing fundamental limitations of current vision foundation models in aerial settings and establishing new challenges for robust 3D scene understanding in the aerial domain. Visit this https URL.
>
---
#### [replaced 040] AltChart: Enhancing VLM-based Chart Summarization Through Multi-Pretext Tasks
- **分类: cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2405.13580](https://arxiv.org/pdf/2405.13580)**

> **作者:** Omar Moured; Jiaming Zhang; M. Saquib Sarfraz; Rainer Stiefelhagen
>
> **备注:** Concerns about reproducibility of the train results and dataset availability
>
> **摘要:** Chart summarization is a crucial task for blind and visually impaired individuals as it is their primary means of accessing and interpreting graphical data. Crafting high-quality descriptions is challenging because it requires precise communication of essential details within the chart without vision perception. Many chart analysis methods, however, produce brief, unstructured responses that may contain significant hallucinations, affecting their reliability for blind people. To address these challenges, this work presents three key contributions: (1) We introduce the AltChart dataset, comprising 10,000 real chart images, each paired with a comprehensive summary that features long-context, and semantically rich annotations. (2) We propose a new method for pretraining Vision-Language Models (VLMs) to learn fine-grained chart representations through training with multiple pretext tasks, yielding a performance gain with ${\sim}2.5\%$. (3) We conduct extensive evaluations of four leading chart summarization models, analyzing how accessible their descriptions are. Our dataset and codes are publicly available on our project page: this https URL.
>
---
#### [replaced 041] Less is More: Rethinking Few-Shot Learning and Recurrent Neural Nets
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2209.14267](https://arxiv.org/pdf/2209.14267)**

> **作者:** Deborah Pereg; Martin Villiger; Brett Bouma; Polina Golland
>
> **备注:** Version 3 is focused exclusively on the first part of v1 and v2, correcting minor mathematical errors. The original co-authors have transitioned in separate follow-up works
>
> **摘要:** The statistical supervised learning framework assumes an input-output set with a joint probability distribution that is reliably represented by the training dataset. The learner is then required to output a prediction rule learned from the training dataset's input-output pairs. In this work, we provide meaningful insights into the asymptotic equipartition property (AEP) \citep{Shannon:1948} in the context of machine learning, and illuminate some of its potential ramifications for few-shot learning. We provide theoretical guarantees for reliable learning under the information-theoretic AEP, and for the generalization error with respect to the sample size. We then focus on a highly efficient recurrent neural net (RNN) framework and propose a reduced-entropy algorithm for few-shot learning. We also propose a mathematical intuition for the RNN as an approximation of a sparse coding solver. We verify the applicability, robustness, and computational efficiency of the proposed approach with image deblurring and optical coherence tomography (OCT) speckle suppression. Our experimental results demonstrate significant potential for improving learning models' sample efficiency, generalization, and time complexity, that can therefore be leveraged for practical real-time applications.
>
---
#### [replaced 042] ScenePilot-4K: A Large-Scale First-Person Dataset and Benchmark for Vision-Language Models in Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.19582](https://arxiv.org/pdf/2601.19582)**

> **作者:** Yujin Wang; Yutong Zheng; Wenxian Fan; Tianyi Wang; Hongqing Chu; Li Zhang; Bingzhao Gao; Daxin Tian; Jianqiang Wang; Hong Chen
>
> **摘要:** In this paper, we introduce ScenePilot-4K, a large-scale first-person dataset for safety-aware vision-language learning and evaluation in autonomous driving. Built from public online driving videos, ScenePilot-4K contains 3,847 hours of video and 27.7M front-view frames spanning 63 countries/regions and 1,210 cities. It jointly provides scene-level natural-language descriptions, risk assessment labels, key-participant annotations, ego trajectories, and camera parameters through a unified multi-stage annotation pipeline. Building on this dataset, we establish ScenePilot-Bench, a standardized benchmark that evaluates vision-language models along four complementary axes: scene understanding, spatial perception, motion planning, and GPT-based semantic alignment. The benchmark includes fine-grained metrics and geographic generalization settings that expose model robustness under cross-region and cross-traffic domain shifts. Baseline results on representative open-source and proprietary vision-language models show that current models remain competitive in high-level scene semantics but still exhibit substantial limitations in geometry-aware perception and planning-oriented reasoning. Beyond the released dataset itself, the proposed annotation pipeline serves as a reusable and extensible recipe for scalable dataset construction from public Internet driving videos. The codes and supplementary materials are available at: this https URL, with the dataset available at this https URL.
>
---
#### [replaced 043] OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition
- **分类: cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2512.16727](https://arxiv.org/pdf/2512.16727)**

> **作者:** Haochen Chang; Pengfei Ren; Buyuan Zhang; Da Li; Tianhao Han; Haoyang Zhang; Liang Xie; Hongbo Chen; Erwei Yin
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Online micro gesture recognition from hand skeletons is critical for VR/AR interaction but faces challenges due to limited public datasets and task-specific algorithms. Micro gestures involve subtle motion patterns, which make constructing datasets with precise skeletons and frame-level annotations difficult. To this end, we develop a multi-view self-supervised pipeline to automatically generate skeleton data, complemented by heuristic rules and expert refinement for semi-automatic annotation. Based on this pipeline, we introduce OMG-Bench, the first large-scale public benchmark for skeleton-based online micro gesture recognition. It features 40 fine-grained gesture classes with 13,948 instances across 1,272 sequences, characterized by subtle motions, rapid dynamics, and continuous execution. To tackle these challenges, we propose Hierarchical Memory-Augmented Transformer (HMATr), an end-to-end framework that unifies gesture detection and classification by leveraging hierarchical memory banks which store frame-level details and window-level semantics to preserve historical context. In addition, it employs learnable position-aware queries initialized from the memory to implicitly encode gesture positions and semantics. Experiments show that HMATr outperforms state-of-the-art methods by 7.6% in detection rate, establishing a strong baseline for online micro gesture recognition. Project page: this https URL
>
---
#### [replaced 044] Thinking with Camera: A Unified Multimodal Model for Camera-Centric Understanding and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.08673](https://arxiv.org/pdf/2510.08673)**

> **作者:** Kang Liao; Size Wu; Zhonghua Wu; Linyi Jin; Chao Wang; Yikai Wang; Fei Wang; Wei Li; Chen Change Loy
>
> **备注:** Accepted by ICLR2026. Project Page: this https URL
>
> **摘要:** Camera-centric understanding and generation are two cornerstones of spatial intelligence, yet they are typically studied in isolation. We present Puffin, a unified camera-centric multimodal model that extends spatial awareness along the camera dimension. Puffin integrates language regression and diffusion-based generation to interpret and create scenes from arbitrary viewpoints. To bridge the modality gap between cameras and vision-language, we introduce a novel paradigm that treats camera as language, enabling thinking with camera. This guides the model to align spatially grounded visual cues with photographic terminology while reasoning across geometric context. Puffin is trained on Puffin-4M, a large-scale dataset of 4 million vision-language-camera triplets. We incorporate both global camera parameters and pixel-wise camera maps, yielding flexible and reliable spatial generation. Experiments demonstrate Puffin superior performance over specialized models for camera-centric generation and understanding. With instruction tuning, Puffin generalizes to diverse cross-view tasks such as spatial imagination, world exploration, and photography guidance. We will release the code, models, dataset pipeline, and benchmark to advance multimodal spatial intelligence research.
>
---
#### [replaced 045] DTVI: Dual-Stage Textual and Visual Intervention for Safe Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.22041](https://arxiv.org/pdf/2603.22041)**

> **作者:** Binhong Tan; Zhaoxin Wang; Handing Wang
>
> **摘要:** Text-to-Image (T2I) diffusion models have demonstrated strong generation ability, but their potential to generate unsafe content raises significant safety concerns. Existing inference-time defense methods typically perform category-agnostic token-level intervention in the text embedding space, which fails to capture malicious semantics distributed across the full token sequence and remains vulnerable to adversarial prompts. In this paper, we propose DTVI, a dual-stage inference-time defense framework for safe T2I generation. Unlike existing methods that intervene on specific token embeddings, our method introduces category-aware sequence-level intervention on the full prompt embedding to better capture distributed malicious semantics, and further attenuates the remaining unsafe influences during the visual generation stage. Experimental results on real-world unsafe prompts, adversarial prompts, and multiple harmful categories show that our method achieves effective and robust defense while preserving reasonable generation quality on benign prompts, obtaining an average Defense Success Rate (DSR) of 94.43% across sexual-category benchmarks and 88.56 across seven unsafe categories, while maintaining generation quality on benign prompts.
>
---
#### [replaced 046] Habitat Classification from Ground-Level Imagery Using Deep Neural Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.04017](https://arxiv.org/pdf/2507.04017)**

> **作者:** Hongrui Shi; Lisa Norton; Lucy Ridding; Simon Rolph; Tom August; Claire M Wood; Lan Qie; Petra Bosilj; James M Brown
>
> **备注:** Accepted to Ecological Informatics. Main paper has 19 pages, 7 figures, 4 tables. Appendix has 10 pages, 8 figures, 2 tables
>
> **摘要:** Habitat assessment at local scales--critical for enhancing biodiversity and guiding conservation priorities--often relies on expert field surveys that can be costly, motivating the exploration of AI-driven tools to automate and refine this process. While most AI-driven habitat mapping depends on remote sensing, it is often constrained by sensor availability, weather, and coarse resolution. In contrast, ground-level imagery captures essential structural and compositional cues invisible from above and remains underexplored for robust, fine-grained habitat classification. This study addresses this gap by applying state-of-the-art deep neural network architectures to ground-level habitat imagery. Leveraging data from the UK Countryside Survey covering 18 broad habitat types, we evaluate two families of models - convolutional neural networks (CNNs) and vision transformers (ViTs) - under both supervised and supervised contrastive learning paradigms. Our results demonstrate that ViTs consistently outperform state-of-the-art CNN baselines on key classification metrics (Top-3 accuracy = 91%, MCC = 0.66) and offer more interpretable scene understanding tailored to ground-level images. Moreover, supervised contrastive learning significantly reduces misclassification rates among visually similar habitats (e.g., Improved vs. Neutral Grassland), driven by a more discriminative embedding space. Finally, our best model performs on par with experienced ecological experts in habitat classification from images, underscoring the promise of expert-level automated assessment. By integrating advanced AI with ecological expertise, this research establishes a scalable, cost-effective framework for ground-level habitat monitoring to accelerate biodiversity conservation and inform land-use decisions at a national scale.
>
---
#### [replaced 047] SEEC: Segmentation-Assisted Multi-Entropy Models for Learned Lossless Image Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.07704](https://arxiv.org/pdf/2509.07704)**

> **作者:** Chunhang Zheng; Zichang Ren; Dou Li
>
> **备注:** Accpeted by ICME 2026
>
> **摘要:** Recently, learned image compression has attracted considerable attention due to its superior performance over traditional methods. However, most existing approaches employ a single entropy model to estimate the probability distribution of pixel values across the entire image, which limits their ability to capture the diverse statistical characteristics of different semantic regions. To overcome this limitation, we propose Segmentation-Assisted Multi-Entropy Models for Lossless Image Compression (SEEC). Our framework utilizes semantic segmentation to guide the selection and adaptation of multiple entropy models, enabling more accurate probability distribution estimation for distinct semantic regions. Experimental results on benchmark datasets demonstrate that SEEC achieves state-of-the-art compression ratios while introducing only minimal encoding and decoding latency. With superior performance, the proposed model also supports Regions of Interest (ROIs) coding condition on the provided segmentation mask. Our code is available at this https URL.
>
---
#### [replaced 048] Dream to Recall: Imagination-Guided Experience Retrieval for Memory-Persistent Vision-and-Language Navigation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决记忆持久型VLN中记忆访问机制不足的问题。提出Memoir模型，利用想象引导检索环境与行为记忆，提升导航效果。**

- **链接: [https://arxiv.org/pdf/2510.08553](https://arxiv.org/pdf/2510.08553)**

> **作者:** Yunzhe Xu; Yiyuan Pan; Zhe Liu
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to follow natural language instructions through environments, with memory-persistent variants demanding progressive improvement through accumulated experience. Existing approaches for memory-persistent VLN face critical limitations: they lack effective memory access mechanisms, instead relying on entire memory incorporation or fixed-horizon lookup, and predominantly store only environmental observations while neglecting navigation behavioral patterns that encode valuable decision-making strategies. We present Memoir, which employs imagination as a retrieval mechanism grounded by explicit memory: a world model imagines future navigation states as queries to selectively retrieve relevant environmental observations and behavioral histories. The approach comprises: 1) a language-conditioned world model that imagines future states serving dual purposes: encoding experiences for storage and generating retrieval queries; 2) Hybrid Viewpoint-Level Memory that anchors both observations and behavioral patterns to viewpoints, enabling hybrid retrieval; and 3) an experience-augmented navigation model that integrates retrieved knowledge through specialized encoders. Extensive evaluation across diverse memory-persistent VLN benchmarks with 10 distinct testing scenarios demonstrates Memoir's effectiveness: significant improvements across all scenarios, with 5.4% SPL gains on IR2R over the best memory-persistent baseline, accompanied by 8.3x training speedup and 74% inference memory reduction. The results validate that predictive retrieval of both environmental and behavioral memories enables more effective navigation, with analysis indicating substantial headroom (73.3% vs 93.4% upper bound) for this imagination-guided paradigm.
>
---
#### [replaced 049] ImAgent: A Unified Multimodal Agent Framework for Test-Time Scalable Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.11483](https://arxiv.org/pdf/2511.11483)**

> **作者:** Kaishen Wang; Ruibo Chen; Tong Zheng; Heng Huang
>
> **备注:** 8 tables, 8 figures
>
> **摘要:** Recent text-to-image (T2I) models have made remarkable progress in generating visually realistic and semantically coherent images. However, they still suffer from randomness and inconsistency with the given prompts, particularly when textual descriptions are vague or underspecified. Existing approaches, such as prompt rewriting, best-of-N sampling, and self-refinement, can mitigate these issues but usually require additional modules and operate independently, hindering test-time scaling efficiency and increasing computational overhead. In this paper, we introduce ImAgent, a training-free unified multimodal agent that integrates reasoning, generation, and self-evaluation within a single framework for efficient test-time scaling. Guided by a policy controller, multiple generation actions dynamically interact and self-organize to enhance image fidelity and semantic alignment without relying on external models. Extensive experiments on image generation and editing tasks demonstrate that ImAgent consistently improves over the backbone and even surpasses other strong baselines where the backbone model fails, highlighting the potential of unified multimodal agents for adaptive and efficient image generation under test-time scaling.
>
---
#### [replaced 050] Aligning Multi-Dimensional Preferences via Relevance Feedback: An Effortless and Training-Free Framework for Text-to-Image Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14936](https://arxiv.org/pdf/2603.14936)**

> **作者:** Wenxi Wang; Hongbin Liu; Mingqian Li; Junyan Yuan; Junqi Zhang
>
> **摘要:** Aligning generated images with users' latent visual preferences remains a fundamental challenge in text-to-image diffusion models. Existing methods fall short: training-based approaches incur prohibitive costs and lack flexibility, while inference methods using textual feedback impose heavy cognitive burdens. Recent binary feedback methods reduce effort but force Foundation Models (FMs) to infer preferences semantically. During multi-dimensional alignment, FMs suffer from inference overload and fail to accurately attribute individual feature contributions under conflicting user signals. Consequently, a low-cost, low-cognitive-load framework for multi-dimensional alignment remains critically this http URL address this, we propose a Relevance Feedback-Driven (RFD) framework, adapting the relevance feedback mechanism from information retrieval to diffusion models. RFD replaces explicit dialogue with implicit visual feedback, enabling effortless expression of multi-dimensional preferences. To tackle inference overload, RFD decouples the process into independent single-feature preference inference tasks. Furthermore, to overcome FMs' inability to attribute features under conflicting signals, RFD employs rigorous mathematical methods (Odds Ratio and Cohen's d) to quantify feature divergence between "liked" and "disliked" images. This achieves the accurate, transparent feature attribution that FMs fundamentally this http URL, RFD operates entirely within the external text space, making it strictly training-free and model-agnostic. This provides a universal plug-and-play solution without prohibitive fine-tuning costs. Extensive experiments demonstrate that RFD effectively captures true visual intent, significantly outperforming baseline approaches.
>
---
#### [replaced 051] Improving Semantic Uncertainty Quantification in LVLMs with Semantic Gaussian Processes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14177](https://arxiv.org/pdf/2512.14177)**

> **作者:** Joseph Hoche; Andrei Bursuc; David Brellmann; Gilles Louppe; Pavel Izmailov; Angela Yao; Gianni Franchi
>
> **摘要:** Large Vision-Language Models (LVLMs) often produce plausible but unreliable outputs, making robust uncertainty estimation essential. Recent work on semantic uncertainty estimates relies on external models to cluster multiple sampled responses and measure their semantic consistency. However, these clustering methods are often fragile, highly sensitive to minor phrasing variations, and can incorrectly group or separate semantically similar answers, leading to unreliable uncertainty estimates. We propose Semantic Gaussian Process Uncertainty (SGPU), a Bayesian framework that quantifies semantic uncertainty by analyzing the geometric structure of answer embeddings, avoiding brittle clustering. SGPU maps generated answers into a dense semantic space, computes the Gram matrix of their embeddings, and summarizes their semantic configuration via the eigenspectrum. This spectral representation is then fed into a Gaussian Process Classifier that learns to map patterns of semantic consistency to predictive uncertainty, and that can be applied in both black-box and white-box settings. Across six LLMs and LVLMs on eight datasets spanning VQA, image classification, and textual QA, SGPU consistently achieves state-of-the-art calibration (ECE) and discriminative (AUROC, AUARC) performance. We further show that SGPU transfers across models and modalities, indicating that its spectral representation captures general patterns of semantic uncertainty.
>
---
#### [replaced 052] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10950](https://arxiv.org/pdf/2512.10950)**

> **作者:** Qitao Zhao; Hao Tan; Qianqian Wang; Sai Bi; Kai Zhang; Kalyan Sunkavalli; Shubham Tulsiani; Hanwen Jiang
>
> **备注:** CVPR 2026 Camera-ready. Project website: this https URL
>
> **摘要:** Self-supervised pre-training has driven rapid progress in foundation models for language, 2D images, and video, yet remains largely unexplored for learning 3D-aware representations from multi-view images. In this paper, we present E-RayZer, a self-supervised 3D vision model that learns geometrically grounded representations directly from unlabeled images. Unlike prior self-supervised methods such as RayZer, which infer 3D indirectly through latent-space view synthesis, E-RayZer operates directly in 3D space, performing self-supervised 3D reconstruction with Explicit geometry. This formulation eliminates shortcut solutions and yields representations that are 3D-aware. To ensure convergence and scalability, we introduce a fine-grained learning curriculum that organizes training from easy to hard samples and harmonizes heterogeneous data sources without any supervision. Experiments show that E-RayZer significantly outperforms RayZer on pose estimation and matches or sometimes surpasses fully supervised reconstruction models such as VGGT. Furthermore, its learned representations outperform leading visual pre-training models (e.g., DINOv3, CroCo v2, VideoMAE V2, and RayZer) on 3D downstream tasks, establishing E-RayZer as a promising paradigm for spatial visual pre-training.
>
---
#### [replaced 053] MeshSplats: Mesh-Based Rendering with Gaussian Splatting Initialization
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2502.07754](https://arxiv.org/pdf/2502.07754)**

> **作者:** Rafał Tobiasz; Grzegorz Wilczyński; Marcin Mazur; Sławomir Tadeja; Weronika Smolak-Dyżewska; Przemysław Spurek
>
> **摘要:** Gaussian Splatting (GS) is a recent and pivotal technique in 3D computer graphics. GS-based algorithms almost always bypass classical methods such as ray tracing, which offer numerous inherent advantages for rendering. For example, ray tracing can handle incoherent rays for advanced lighting effects, including shadows and reflections. To address this limitation, we introduce MeshSplats, a method which converts GS to a mesh-like format. Following the completion of training, MeshSplats transforms Gaussian elements into mesh faces, enabling rendering using ray tracing methods with all their associated benefits. Our model can be utilized immediately following transformation, yielding a mesh of slightly reduced reconstruction quality without additional training. Furthermore, we can enhance the quality by applying a dedicated optimization algorithm that operates on mesh faces rather than Gaussian components. Importantly, MeshSplats acts as a wrapper, converting pre-trained GS models into a ray-traceable format. The efficacy of our method is substantiated by experimental results, underscoring its extensive applications in computer graphics and image processing.
>
---
#### [replaced 054] Anatomical Token Uncertainty for Transformer-Guided Active MRI Acquisition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21806](https://arxiv.org/pdf/2603.21806)**

> **作者:** Lev Ayzenberg; Shady Abu-Hussein; Raja Giryes; Hayit Greenspan
>
> **摘要:** Full data acquisition in MRI is inherently slow, which limits clinical throughput and increases patient discomfort. Compressed Sensing MRI (CS-MRI) seeks to accelerate acquisition by reconstructing images from under-sampled k-space data, requiring both an optimal sampling trajectory and a high-fidelity reconstruction model. In this work, we propose a novel active sampling framework that leverages the inherent discrete structure of a pretrained medical image tokenizer and a latent transformer. By representing anatomy through a dictionary of quantized visual tokens, the model provides a well-defined probability distribution over the latent space. We utilize this distribution to derive a principled uncertainty measure via token entropy, which guides the active sampling process. We introduce two strategies to exploit this latent uncertainty: (1) Latent Entropy Selection (LES), projecting patch-wise token entropy into the $k$-space domain to identify informative sampling lines, and (2) Gradient-based Entropy Optimization (GEO), which identifies regions of maximum uncertainty reduction via the $k$-space gradient of a total latent entropy loss. We evaluate our framework on the fastMRI singlecoil Knee and Brain datasets at $\times 8$ and $\times 16$ acceleration. Our results demonstrate that our active policies outperform state-of-the-art baselines in perceptual metrics, and feature-based distances. Our code is available at this https URL.
>
---
#### [replaced 055] $ϕ$-DPO: Fairness Direct Preference Optimization Approach to Continual Learning in Large Multimodal Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22601](https://arxiv.org/pdf/2602.22601)**

> **作者:** Thanh-Dat Truong; Huu-Thien Tran; Jackson Cothren; Bhiksha Raj; Khoa Luu
>
> **备注:** Accepted to CVPR'26
>
> **摘要:** Fairness in Continual Learning for Large Multimodal Models (LMMs) is an emerging yet underexplored challenge, particularly in the presence of imbalanced data distributions that can lead to biased model updates and suboptimal performance across tasks. While recent continual learning studies have made progress in addressing catastrophic forgetting, the problem of fairness caused the imbalanced data remains largely underexplored. This paper presents a novel Fairness Direct Preference Optimization (FaiDPO or $\phi$-DPO) framework for continual learning in LMMs. In particular, we first propose a new continual learning paradigm based on Direct Preference Optimization (DPO) to mitigate catastrophic forgetting by aligning learning with pairwise preference signals. Then, we identify the limitations of conventional DPO in imbalanced data and present a new $\phi$-DPO loss that explicitly addresses distributional biases. We provide a comprehensive theoretical analysis demonstrating that our approach addresses both forgetting and data imbalance. Additionally, to enable $\phi$-DPO-based continual learning, we construct pairwise preference annotations for existing benchmarks in the context of continual learning. Extensive experiments and ablation studies show the proposed $\phi$-DPO achieves State-of-the-Art performance across multiple benchmarks, outperforming prior continual learning methods of LMMs.
>
---
#### [replaced 056] MotionCrafter: Dense Geometry and Motion Reconstruction with a 4D VAE
- **分类: cs.CV; cs.AI; cs.CG; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.08961](https://arxiv.org/pdf/2602.08961)**

> **作者:** Ruijie Zhu; Jiahao Lu; Wenbo Hu; Xiaoguang Han; Jianfei Cai; Ying Shan; Chuanxia Zheng
>
> **备注:** Project page: this https URL
>
> **摘要:** We present MotionCrafter, a framework that leverages video generators to jointly reconstruct 4D geometry and estimate dense motion from a monocular video. The key idea is a joint representation of dense 3D point maps and 3D scene flows in a shared coordinate system, together with a 4D VAE tailored to learn this representation effectively. Unlike prior work that strictly aligns 3D values and latents with RGB VAE latents-despite their fundamentally different distributions-we show that such alignment is unnecessary and can hurt performance. Instead, we propose a new data normalization and VAE training strategy that better transfers diffusion priors and greatly improves reconstruction quality. Extensive experiments on multiple datasets show that MotionCrafter achieves state-of-the-art performance in both geometry reconstruction and dense scene flow estimation, delivering 38.64% and 25.0% improvements in geometry and motion reconstruction, respectively, all without any post-optimization. Project page: this https URL
>
---
#### [replaced 057] Revisiting Adversarial Training under Hyperspectral Image
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.01014](https://arxiv.org/pdf/2510.01014)**

> **作者:** Weihua Zhang; Chengze Jiang; Minjing Dong; Jie Gui; Lu Dong; Zhipeng Gui; Yuan Yan Tang; James Tin-Yau Kwok
>
> **摘要:** Recent studies have shown that deep learning-based hyperspectral image (HSI) classification models are highly vulnerable to adversarial attacks, posing significant security risks. Although most approaches attempt to enhance robustness by optimizing network architectures, these methods often rely on customized designs with limited scalability and struggle to defend against strong attacks. To address this issue, we introduce adversarial training (AT), one of the most effective defense strategies, into the hyperspectral domain. However, unlike conventional RGB image classification, directly applying AT to HSI classification introduces unique challenges due to the high-dimensional spectral signatures and strong inter-band correlations of hyperspectral data, where discriminative information relies on subtle spectral semantics and spectral-spatial consistency that are highly sensitive to adversarial perturbations. Through extensive empirical analyses, we observe that adversarial perturbations and the non-smooth nature of adversarial examples can distort or even eliminate important spectral semantic information. To mitigate this issue, we propose two hyperspectral-specific AT methods, termed AT-HARL and AT-RA. Specifically, AT-HARL exploits spectral characteristic differences and class distribution ratios to design a novel loss function that alleviates semantic distortion caused by adversarial perturbations. Meanwhile, AT-RA introduces spectral data augmentation to enhance spectral diversity while preserving spatial smoothness. Experiments on four benchmark HSI datasets demonstrate that the proposed methods achieve competitive performance compared with state-of-the-art approaches under adversarial attacks.
>
---
#### [replaced 058] MaskDiME: Adaptive Masked Diffusion for Precise and Efficient Visual Counterfactual Explanations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18792](https://arxiv.org/pdf/2602.18792)**

> **作者:** Changlu Guo; Anders Nymark Christensen; Anders Bjorholm Dahl; Morten Rieger Hannemose
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Visual counterfactual explanations aim to reveal the minimal semantic modifications that can alter a model's prediction, providing causal and interpretable insights into deep neural networks. However, existing diffusion-based counterfactual generation methods are often computationally expensive, slow to sample, and imprecise in localizing the modified regions. To address these limitations, we propose MaskDiME, a simple, fast, yet effective diffusion framework that unifies semantic consistency and spatial precision through localized sampling. Our approach adaptively focuses on decision-relevant regions to achieve localized and semantically consistent counterfactual generation while preserving high image fidelity. Our training-free framework, MaskDiME, performs inference over 30x faster than the baseline and achieves comparable or state-of-the-art performance across five benchmark datasets spanning diverse visual domains, establishing a practical and generalizable solution for efficient counterfactual explanation.
>
---
#### [replaced 059] HSD: Training-Free Acceleration for Document Parsing Vision-Language Model with Hierarchical Speculative Decoding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.12957](https://arxiv.org/pdf/2602.12957)**

> **作者:** Wenhui Liao; Hongliang Li; Pengyu Xie; Xinyu Cai; Yufan Shen; Yi Xin; Qi Qin; Shenglong Ye; Tianbin Li; Ming Hu; Junjun He; Yihao Liu; Wenhai Wang; Min Dou; Bin Fu; Botian Shi; Yu Qiao; Lianwen Jin
>
> **摘要:** Document parsing is a fundamental task in multimodal understanding, supporting a wide range of downstream applications such as information extraction and intelligent document analysis. Benefiting from strong semantic modeling and robust generalization, VLM-based end-to-end approaches have emerged as the mainstream paradigm in recent years. However, these models often suffer from substantial inference latency, as they must autoregressively generate long, full-page sequences when processing long-form documents. While recent hybrid methods mitigate this issue via region-level parallel decoding with VLMs, independent region decoding loses full-page context and might weaken global coherence. To address this issue, we propose Hierarchical Speculative Decoding (HSD), a two-stage local-to-global framework for document parsing. HSD first employs a lightweight pipeline drafter to predict region partitions and generate coarse drafts for each region. The first stage verifies the generated region-level drafts in parallel for efficiency, while the second stage further performs page-level verification on these refined outputs to preserve full-page coherence. Experimental results show that our HSD achieves a 2.78x near-lossless speedup with HunyuanOCR on OmniDocBench v1.5 and up to 7.04x speedup on long-document parsing tasks, demonstrating the effectiveness of our proposed method. We will release our code to facilitate reproducibility.
>
---
#### [replaced 060] TimeFlow: Temporal Conditioning for Longitudinal Brain MRI Registration and Aging Analysis
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2501.08667](https://arxiv.org/pdf/2501.08667)**

> **作者:** Bailiang Jian; Jiazhen Pan; Yitong Li; Fabian Bongratz; Ruochen Li; Daniel Rueckert; Benedikt Wiestler; Christian Wachinger
>
> **摘要:** Longitudinal brain analysis is essential for understanding healthy aging and identifying pathological deviations. Longitudinal registration of sequential brain MRI underpins such analyses. However, existing methods are limited by reliance on densely sampled time series, a trade-off between accuracy and temporal smoothness, and an inability to prospectively forecast future brain states. To overcome these challenges, we introduce \emph{TimeFlow}, a learning-based framework for longitudinal brain MRI registration. TimeFlow uses a U-Net backbone with temporal conditioning to model neuroanatomy as a continuous function of age. Given only two scans from an individual, TimeFlow estimates accurate and temporally coherent deformation fields, enabling non-linear extrapolation to predict future brain states. This is achieved by our proposed inter-/extra-polation consistency constraints applied to both the deformation fields and deformed images. Remarkably, these constraints preserve temporal consistency and continuity without requiring explicit smoothness regularizers or densely sampled sequential data. Extensive experiments demonstrate that TimeFlow outperforms state-of-the-art methods in terms of both future timepoint forecasting and registration accuracy. Moreover, TimeFlow supports novel biological brain aging analyses by differentiating neurodegenerative trajectories from normal aging without requiring segmentation, thereby eliminating the need for labor-intensive annotations and mitigating segmentation inconsistency. TimeFlow offers an accurate, data-efficient, and annotation-free framework for longitudinal analysis of brain aging and chronic diseases, capable of forecasting brain changes beyond the observed study period.
>
---
#### [replaced 061] Follow-Your-Motion: Video Motion Transfer via Efficient Spatial-Temporal Decoupled Finetuning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.05207](https://arxiv.org/pdf/2506.05207)**

> **作者:** Yue Ma; Yulong Liu; Qiyuan Zhu; Ayden Yang; Kunyu Feng; Xinhua Zhang; Zexuan Yan; Zhifeng Li; Sirui Han; Chenyang Qi; Qifeng Chen
>
> **备注:** Accepted by ICLR 2026, project page: this https URL
>
> **摘要:** Recently, breakthroughs in the video diffusion transformer have shown remarkable capabilities in diverse motion generations. As for the motion-transfer task, current methods mainly use two-stage Low-Rank Adaptations (LoRAs) finetuning to obtain better performance. However, existing adaptation-based motion transfer still suffers from motion inconsistency and tuning inefficiency when applied to large video diffusion transformers. Naive two-stage LoRA tuning struggles to maintain motion consistency between generated and input videos due to the inherent spatial-temporal coupling in the 3D attention operator. Additionally, they require time-consuming fine-tuning processes in both stages. To tackle these issues, we propose Follow-Your-Motion, an efficient two-stage video motion transfer framework that finetunes a powerful video diffusion transformer to synthesize complex motion. Specifically, we propose a spatial-temporal decoupled LoRA to decouple the attention architecture for spatial appearance and temporal motion processing. During the second training stage, we design the sparse motion sampling and adaptive RoPE to accelerate the tuning speed. To address the lack of a benchmark for this field, we introduce MotionBench, a comprehensive benchmark comprising diverse motion, including creative camera motion, single object motion, multiple object motion, and complex human motion. We show extensive evaluations on MotionBench to verify the superiority of Follow-Your-Motion.
>
---
#### [replaced 062] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.17470](https://arxiv.org/pdf/2601.17470)**

> **作者:** Chia-Ming Lee; Yu-Fan Lin; Yu-Jou Hsiao; Jin-Hui Jiang; Yu-Lun Liu; Chih-Chung Hsu
>
> **备注:** CVPR 2026 Camera Ready; Project Page: this https URL
>
> **摘要:** Shadow removal under diverse lighting conditions requires disentangling illumination from intrinsic reflectance, a challenge compounded when physical priors are not properly aligned. We propose PhaSR (Physically Aligned Shadow Removal), addressing this through dual-level prior alignment to enable robust performance from single-light shadows to multi-source ambient lighting. First, Physically Aligned Normalization (PAN) performs closed-form illumination correction via Gray-world normalization, log-domain Retinex decomposition, and dynamic range recombination, suppressing chromatic bias. Second, Geometric-Semantic Rectification Attention (GSRA) extends differential attention to cross-modal alignment, harmonizing depth-derived geometry with DINO-v2 semantic embeddings to resolve modal conflicts under varying illumination. Experiments show competitive performance in shadow removal with lower complexity and generalization to ambient lighting where traditional methods fail under multi-source illumination. Our source code is available at this https URL.
>
---
#### [replaced 063] Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07619](https://arxiv.org/pdf/2603.07619)**

> **作者:** Abin Shoby; Ta Duc Huy; Tuan Dung Nguyen; Minh Khoi Ho; Qi Chen; Anton van den Hengel; Phi Le Nguyen; Johan W. Verjans; Vu Minh Hieu Phan
>
> **备注:** CVPR2026 Findings
>
> **摘要:** Vision Language models (VLMs) often hallucinate non-existent objects. Detecting hallucination is analogous to detecting deception: a single final statement is insufficient, one must examine the underlying reasoning process. Yet existing detectors rely mostly on final-layer signals. Attention-based methods assume hallucinated tokens exhibit low attention, while entropy-based ones use final-step uncertainty. Our analysis reveals the opposite: hallucinated objects can exhibit peaked attention due to contextual priors; and models often express high confidence because intermediate layers have already converged to an incorrect hypothesis. We show that the key to hallucination detection lies within the model's thought process, not its final output. By probing decoder layers, we uncover a previously overlooked behavior, overthinking: models repeatedly revise object hypotheses across layers before committing to an incorrect answer. Once the model latches onto a confounded hypothesis, it can propagate through subsequent layers, ultimately causing hallucination. To capture this behavior, we introduce the Overthinking Score, a metric to measure how many competing hypotheses the model entertains and how unstable these hypotheses are across layers. This score significantly improves hallucination detection: 78.9% F1 on MSCOCO and 71.58% on AMBER.
>
---
#### [replaced 064] Generating Findings for Jaw Cysts in Dental Panoramic Radiographs Using a GPT-Based VLM: A Preliminary Study on Building a Two-Stage Self-Correction Loop with Structured Output (SLSO) Framework
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.02001](https://arxiv.org/pdf/2510.02001)**

> **作者:** Nanaka Hosokawa; Ryo Takahashi; Tomoya Kitano; Yukihiro Iida; Chisako Muramatsu; Tatsuro Hayashi; Yuta Seino; Xiangrong Zhou; Takeshi Hara; Akitoshi Katsumata; Hiroshi Fujita
>
> **备注:** Revised manuscript; supplementary materials added. Submitted to Diagnostics
>
> **摘要:** Vision-language models (VLMs) such as GPT (Generative Pre-Trained Transformer) have shown potential for medical image interpretation; however, challenges remain in generating reliable radiological findings in clinical practice, as exemplified by dental pathologies. This study proposes a Self-correction Loop with Structured Output (SLSO) framework as an integrated processing methodology to enhance the accuracy and reliability of AI-generated findings for jaw cysts in dental panoramic radiographs. Dental panoramic radiographs with jaw cysts were used to implement a 10-step integrated processing framework incorporating image analysis, structured data generation, tooth number extraction, consistency checking, and iterative regeneration. The framework functioned as an external validation mechanism for GPT outputs. Performance was compared against the conventional Chain-of-Thought (CoT) method across seven evaluation items: transparency, internal structure, borders, root resorption, tooth movement, relationships with other structures, and tooth number. The SLSO framework improved output accuracy for multiple items compared to the CoT method, with the most notable improvements observed in tooth number identification, tooth movement detection, and root resorption assessment. In successful cases, consistently structured outputs were achieved after up to five regenerations. The framework enforced explicit negative finding descriptions and suppressed hallucinations, although accurate identification of extensive lesions spanning multiple teeth remained limited. This investigation established the feasibility of the proposed integrated processing methodology and provided a foundation for future validation studies with larger, more diverse datasets.
>
---
#### [replaced 065] ViPRA: Video Prediction for Robot Actions
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出ViPRA，解决机器人控制中缺乏标注动作的问题。通过视频预测和隐式动作表示，实现无需大量标注的连续控制，提升泛化能力和控制频率。**

- **链接: [https://arxiv.org/pdf/2511.07732](https://arxiv.org/pdf/2511.07732)**

> **作者:** Sandeep Routray; Hengkai Pan; Unnat Jain; Shikhar Bahl; Deepak Pathak
>
> **备注:** In ICLR 2026. Website: this https URL
>
> **摘要:** Can we turn a video prediction model into a robot policy? Videos, including those of humans or teleoperated robots, capture rich physical interactions. However, most of them lack labeled actions, which limits their use in robot learning. We present Video Prediction for Robot Actions (ViPRA), a simple pretraining-finetuning framework that learns continuous robot control from these actionless videos. Instead of directly predicting actions, we train a video-language model to predict both future visual observations and motion-centric latent actions, which serve as intermediate representations of scene dynamics. We train these latent actions using perceptual losses and optical flow consistency to ensure they reflect physically grounded behavior. For downstream control, we introduce a chunked flow matching decoder that maps latent actions to robot-specific continuous action sequences, using only 100 to 200 teleoperated demonstrations. This approach avoids expensive action annotation, supports generalization across embodiments, and enables smooth, high-frequency continuous control upto 22 Hz via chunked action decoding. Unlike prior latent action works that treat pretraining as autoregressive policy learning, ViPRA explicitly models both what changes and how. Our method outperforms strong baselines, with a 16% gain on the SIMPLER benchmark and a 13% improvement across real world manipulation tasks. We have released models and code at this https URL
>
---
#### [replaced 066] Unleashing the Potential of Mamba: Boosting a LiDAR 3D Sparse Detector by Using Cross-Model Knowledge Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.11018](https://arxiv.org/pdf/2409.11018)**

> **作者:** Rui Yu; Runkai Zhao; Jiagen Li; Qingsong Zhao; HuaiCheng Yan; Meng Wang
>
> **摘要:** The LiDAR 3D object detector that strikes a balance between accuracy and speed is crucial for achieving real-time perception in autonomous driving. However, many existing LiDAR detection models depend on complex feature transformations, leading to poor real-time performance and high resource consumption, which limits their practical effectiveness. In this work, we propose a faster LiDAR 3D object detector, a framework that adaptively aligns sparse voxels to enable efficient heterogeneous knowledge distillation, called FASD. We aim to distill the Transformer sequence modeling capability into Mamba models, significantly boosting accuracy through knowledge transfer. Specifically, we first design the architecture for cross-model knowledge distillation to impart the global contextual understanding capabilities of the Transformer to Mamba. Transformer-based teacher model employ a scale-adaptive attention mechanism to enhance multiscale fusion. In contrast, Mamba-based student model leverages feature alignment through spatial-based adapters, supervised with latent space feature and span-head distillation losses, leading to improved performance and efficiency. We evaluated the FASD on the Waymo and nuScenes datasets, achieving a 4x reduction in resource consumption and a 1-2% performance improvement over the baseline, while also delivering significant gains in accuracy and efficiency in real deployment.
>
---
#### [replaced 067] PowerCLIP: Powerset Alignment for Contrastive Pre-Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.23170](https://arxiv.org/pdf/2511.23170)**

> **作者:** Masaki Kawamura; Nakamasa Inoue; Rintaro Yanagi; Hirokatsu Kataoka; Rio Yokota
>
> **摘要:** Contrastive vision-language pre-training frameworks such as CLIP have demonstrated impressive zero-shot performance across a range of vision-language tasks. Recent studies have shown that aligning individual text tokens with specific image patches or regions enhances fine-grained compositional understanding. However, it remains challenging to capture compositional semantics that span multiple image regions. To address this limitation, we propose PowerCLIP, a novel contrastive pre-training framework enhanced by powerset alignment, which exhaustively optimizes region-to-phrase alignments by minimizing the loss defined between powersets of image regions and textual parse trees. Since the naive powerset construction incurs exponential computational cost due to the combinatorial explosion in the number of region subsets, we introduce efficient non-linear aggregators (NLAs) that reduce complexity from O(2^M) to O(M) with respect to the number of regions M, while approximating the exact loss value with arbitrary precision. Our extensive experiments demonstrate that PowerCLIP outperforms state-of-the-art methods in zero-shot classification and retrieval tasks, underscoring the compositionality and robustness of our approach. Code is available at this https URL.
>
---
#### [replaced 068] SceneAdapt: Scene-aware Adaptation of Human Motion Diffusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.13044](https://arxiv.org/pdf/2510.13044)**

> **作者:** Jungbin Cho; Minsu Kim; Jisoo Kim; Ce Zheng; Laszlo A. Jeni; Ming-Hsuan Yang; Youngjae Yu; Seonjoo Kim
>
> **备注:** 15 pages
>
> **摘要:** Human motion is inherently diverse and semantically rich, while also shaped by the surrounding scene. However, existing motion generation approaches fail to generate semantically diverse motion while simultaneously respecting geometric scene constraints, since constructing large-scale datasets with both rich text-motion coverage and precise scene interactions is extremely challenging. In this work, we introduce SceneAdapt, a two-stage adaptation framework that enables semantically diverse, scene-aware human motion generation from text without large-scale paired text--scene--motion data. Our key idea is to use motion inbetweening, a learnable proxy task that requires no text, as a bridge between two disjoint resources: a text-motion dataset and a scene-motion dataset. By first adapting a text-to-motion model through inbetweening and then through scene-aware inbetweening, SceneAdapt injects geometric scene constraints into text-conditioned generation while preserving semantic diversity. To enable adaptation for inbetweening, we propose a novel Context-aware Keyframing (CaKey) layer that modulates motion latents for keyframe-conditioned synthesis while preserving the original latent manifold. To further adapt the model for scene-aware inbetweening, we introduce a Scene-conditioning (SceneCo) layer that injects geometric scene information by adaptively querying local context via cross-attention. Experimental results show that SceneAdapt effectively injects scene-awareness into text-to-motion models without sacrificing semantic diversity, and we further analyze the mechanisms through which this awareness emerges. Code and models will be released. Project page: \href{this https URL}{this http URL}
>
---
#### [replaced 069] A Benchmark for Incremental Micro-expression Recognition
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2501.19111](https://arxiv.org/pdf/2501.19111)**

> **作者:** Zhengqin Lai; Xiaopeng Hong; Yabin Wang; Xiaobai Li
>
> **摘要:** Micro-expression recognition plays a pivotal role in understanding hidden emotions and has applications across various fields. Traditional recognition methods assume access to all training data at once, but real-world scenarios involve continuously evolving data streams. To respond to the requirement of adapting to new data while retaining previously learned knowledge, we introduce the first benchmark specifically designed for incremental micro-expression recognition. Our contributions include: Firstly, we formulate the incremental learning setting tailored for micro-expression recognition. Secondly, we organize sequential datasets with carefully curated learning orders to reflect real-world scenarios. Thirdly, we define two cross-evaluation-based testing protocols, each targeting distinct evaluation objectives. Finally, we provide six baseline methods and their corresponding evaluation results. This benchmark lays the groundwork for advancing incremental micro-expression recognition research. All source code used in this study will be publicly available at this https URL.
>
---
#### [replaced 070] Few TensoRF: Enhance the Few-shot on Tensorial Radiance Fields
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.25008](https://arxiv.org/pdf/2603.25008)**

> **作者:** Thanh-Hai Le; Hoang-Hau Tran; Trong-Nghia Vu
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** This paper presents Few TensoRF, a 3D reconstruction framework that combines TensorRF's efficient tensor based representation with FreeNeRF's frequency driven few shot regularization. Using TensorRF to significantly accelerate rendering speed and introducing frequency and occlusion masks, the method improves stability and reconstruction quality under sparse input views. Experiments on the Synthesis NeRF benchmark show that Few TensoRF method improves the average PSNR from 21.45 dB (TensorRF) to 23.70 dB, with the fine tuned version reaching 24.52 dB, while maintaining TensorRF's fast \(\approx10-15\) minute training time. Experiments on the THuman 2.0 dataset further demonstrate competitive performance in human body reconstruction, achieving 27.37 - 34.00 dB with only eight input images. These results highlight Few TensoRF as an efficient and data effective solution for real-time 3D reconstruction across diverse scenes.
>
---
#### [replaced 071] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出UniLS，解决音频驱动的虚拟人对话任务中监听动作不自然的问题。通过两阶段训练，实现端到端的说话与倾听表达生成。**

- **链接: [https://arxiv.org/pdf/2512.09327](https://arxiv.org/pdf/2512.09327)**

> **作者:** Xuangeng Chu; Ruicong Liu; Yifei Huang; Yun Liu; Yichen Peng; Bo Zheng
>
> **备注:** CVPR 2026, code is available at this https URL, more demos are available at this https URL
>
> **摘要:** Generating lifelike conversational avatars requires modeling not just isolated speakers, but the dynamic, reciprocal interaction of speaking and listening. However, modeling the listener is exceptionally challenging: direct audio-driven training fails, producing stiff, static listening motions. This failure stems from a fundamental imbalance: the speaker's motion is strongly driven by speech audio, while the listener's motion primarily follows an internal motion prior and is only loosely guided by external speech. This challenge has led most methods to focus on speak-only generation. The only prior attempt at joint generation relies on extra speaker's motion to produce the listener. This design is not end-to-end, thereby hindering the real-time applicability. To address this limitation, we present UniLS, the first end-to-end framework for generating unified speak-listen expressions, driven by only dual-track audio. Our method introduces a novel two-stage training paradigm. Stage 1 first learns the internal motion prior by training an audio-free autoregressive generator, capturing the spontaneous dynamics of natural facial motion. Stage 2 then introduces the dual-track audio, fine-tuning the generator to modulate the learned motion prior based on external speech cues. Extensive evaluations show UniLS achieves state-of-the-art speaking accuracy. More importantly, it delivers up to 44.1\% improvement in listening metrics, generating significantly more diverse and natural listening expressions. This effectively mitigates the stiffness problem and provides a practical, high-fidelity audio-driven solution for interactive digital humans. Code and demos are available at this https URL.
>
---
#### [replaced 072] APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.15288](https://arxiv.org/pdf/2601.15288)**

> **作者:** Jiwon Kang; Yeji Choi; JoungBin Lee; Wooseok Jang; Jinhyeok Choi; Taekeun Kang; Yongjae Park; Myungin Kim; Seungryong Kim
>
> **备注:** Accepted at CVPR 2026. Project Page: this https URL
>
> **摘要:** Face swapping aims to transfer the identity of a source face onto a target face while preserving target-specific attributes such as pose, expression, lighting, skin tone, and makeup. However, since real ground truth for face swapping is unavailable, achieving both accurate identity transfer and high-quality attribute preservation remains challenging. Recent diffusion-based approaches attempt to improve visual fidelity through conditional inpainting on masked target images, but the masked condition removes crucial appearance cues, resulting in plausible yet misaligned attributes. To address this limitation, we propose APPLE (Attribute-Preserving Pseudo-Labeling), a fully diffusion-based teacher-student framework for attribute-preserving face swapping. Our approach introduces a teacher design to produce pseudo-labels aligned with the target attributes through (1) a conditional deblurring formulation that improves the preservation of global attributes such as skin tone and illumination, and (2) an attribute-aware inversion scheme that further enhances fine-grained attribute preservation such as makeup. APPLE conditions the student on clean pseudo-labels rather than degraded masked inputs, enabling more faithful attribute preservation. As a result, APPLE achieves state-of-the-art performance in attribute preservation while maintaining competitive identity transferability.
>
---
#### [replaced 073] NarrativeTrack: Evaluating Entity-Centric Reasoning for Narrative Understanding
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.01095](https://arxiv.org/pdf/2601.01095)**

> **作者:** Hyeonjeong Ha; Jinjin Ge; Bo Feng; Kaixin Ma; Gargi Chakraborty
>
> **备注:** Project Page: this https URL
>
> **摘要:** Multimodal large language models (MLLMs) have achieved impressive progress in vision-language reasoning, yet their ability to understand temporally unfolding narratives in videos remains underexplored. True narrative understanding requires grounding who is doing what, when, and where, maintaining coherent entity representations across dynamic visual and temporal contexts. We introduce NarrativeTrack, the first benchmark to evaluate narrative understanding in MLLMs through fine-grained entity-centric reasoning. Unlike existing benchmarks limited to short clips or coarse scene-level semantics, we decompose videos into constituent entities and examine their continuity via a Compositional Reasoning Progression (CRP), a structured evaluation framework that progressively increases narrative complexity across three dimensions: entity existence, entity changes, and entity ambiguity. CRP challenges models to advance from temporal persistence to contextual evolution and fine-grained perceptual reasoning. A fully automated entity-centric pipeline enables scalable extraction of temporally grounded entity representations, providing the foundation for CRP. Evaluations of state-of-the-art MLLMs reveal that models fail to robustly track entities across visual transitions and temporal dynamics, often hallucinating identity under context shifts. Open-source general-purpose MLLMs exhibit strong perceptual grounding but weak temporal coherence, while video-specific MLLMs capture temporal context yet hallucinate entity's contexts. These findings uncover a fundamental trade-off between perceptual grounding and temporal reasoning, indicating that narrative understanding emerges only from their integration. NarrativeTrack provides the first systematic framework to diagnose and advance temporally grounded narrative comprehension in MLLMs.
>
---
#### [replaced 074] Guidestar-Free Adaptive Optics with Asymmetric Apertures
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07029](https://arxiv.org/pdf/2602.07029)**

> **作者:** Weiyun Jiang; Haiyun Guo; Christopher A. Metzler; Ashok Veeraraghavan
>
> **备注:** Accepted to ACM Transactions on Graphics (TOG)
>
> **摘要:** This work introduces the first closed-loop adaptive optics (AO) system capable of optically correcting aberrations in real-time without a guidestar or a wavefront sensor. Nearly 40 years ago, Cederquist et al. demonstrated that asymmetric apertures enable phase retrieval (PR) algorithms to perform fully computational wavefront sensing, albeit at a high computational cost. More recently, Chimitt et al. extended this approach with machine learning and demonstrated real-time wavefront sensing using only a single (guidestar-based) point-spread-function (PSF) measurement. Inspired by these works, we introduce a guidestar-free AO framework built around asymmetric apertures and machine learning. Our approach combines three key elements: (1) an asymmetric aperture placed at the system's pupil plane that enables PR-based wavefront sensing, (2) a pair of machine learning algorithms that estimate the PSF from natural scene measurements and reconstruct phase aberrations, and (3) a spatial light modulator that performs optical correction. We experimentally validate this framework on dense natural scenes imaged through unknown obscurants. Our method outperforms state-of-the-art guidestar-free wavefront shaping methods, using an order of magnitude fewer measurements and three orders of magnitude less computation.
>
---
#### [replaced 075] Self-Supervised Learning for Knee Osteoarthritis: Diagnostic Limitations and Prognostic Value of Hospital Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24903](https://arxiv.org/pdf/2603.24903)**

> **作者:** Haresh Rengaraj Rajamohan; Yuxuan Chen; Kyunghyun Cho; Cem M. Deniz
>
> **摘要:** This study assesses whether self-supervised learning (SSL) improves knee osteoarthritis (OA) modeling for diagnosis and prognosis relative to ImageNet-pretrained initialization. We compared (i) image-only SSL pretrained on knee radiographs from the OAI, MOST, and NYU cohorts, and (ii) multimodal image-text SSL pretrained on hospital knee radiographs paired with radiologist impressions. For diagnostic Kellgren-Lawrence (KL) grade prediction, SSL yielded mixed results. While image-only SSL improved accuracy during linear probing (frozen encoder), it did not outperform ImageNet pretraining during full fine-tuning. Similarly, multimodal SSL failed to improve grading performance. A likely explanation is mismatch between the hospital pretraining corpus and the downstream diagnostic task: the hospital image-text dataset was restricted to knees from patients with clinically identified OA in routine care, rather than a cohort spanning the full spectrum from normal to severe disease needed for balanced KL grading. In addition, radiology impressions do not explicitly encode KL grade, limiting supervision for learning KL-specific decision boundaries. In contrast, this same multimodal initialization significantly improved prognostic modeling. It outperformed ImageNet baselines in predicting 4-year structural incidence and progression, including on external validation (MOST AUROC: 0.701 vs. 0.599 at 10\% labeled data). Overall, these results suggest that our hospital image-text data may be less effective for diagnostic grading when the pretraining cohort is limited to OA knees, but can provide a strong signal for prognostic modeling when the downstream task is better aligned with the pretraining data distribution.
>
---
#### [replaced 076] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2408.13516](https://arxiv.org/pdf/2408.13516)**

> **作者:** Yujin Lee; Sewon Kim; Daeun Moon; Seoyoon Jang; Hyunsoo Yoon
>
> **备注:** accepted to CVPR 2026
>
> **摘要:** Few-shot multi-class anomaly detection is crucial in real industrial settings, where only a few normal samples are available while numerous object types must be inspected. This setting is challenging as defect patterns vary widely across categories while normal samples remain scarce. Existing vision-language model-based approaches typically depend on class-specific anomaly descriptions or auxiliary modules, limiting both scalability and computational efficiency. In this work, we propose AnoPLe, a lightweight multimodal prompt learning framework that removes reliance on anomaly-type textual descriptions and avoids any external modules. AnoPLe employs bidirectional interactions between textual and visual prompts, allowing class semantics and instance-level cues to refine one another and form class-conditioned representations that capture shared normal patterns across categories. To enhance localization, we design a scale-aware prefix trained on both global and local views, enabling the prompts to capture both global context and fine-grained details. In addition, alignment loss propagates local anomaly evidence to global features, strengthening the consistency between pixel- and image-level predictions. Despite its simplicity, AnoPLe achieves strong performance on MVTec-AD, VisA, and Real-IAD under the few-shot multi-class setting, surpassing prior approaches while remaining efficient and free from expert-crafted anomaly descriptions. Moreover, AnoPLe generalizes well to unseen anomalies and extends effectively to the medical domain.
>
---
#### [replaced 077] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2512.10652](https://arxiv.org/pdf/2512.10652)**

> **作者:** Jian-Yu Jiang-Lin; Kang-Yang Huang; Ling Zou; Ling Lo; Sheng-Ping Yang; Yu-Wen Tseng; Kun-Hsiang Lin; Chia-Ling Chen; Yu-Ting Ta; Yan-Tsung Wang; Po-Ching Chen; Hongxia Xie; Hong-Han Shuai; Wen-Huang Cheng
>
> **备注:** CVPR 2026
>
> **摘要:** Advances in generative modeling have made it increasingly easy to fabricate realistic portrayals of individuals, creating serious risks for security, communication, and public trust. Detecting such person-driven manipulations requires systems that not only distinguish altered content from authentic media but also provide clear and reliable reasoning. In this paper, we introduce TriDF, a comprehensive benchmark for interpretable DeepFake detection. TriDF contains high-quality forgeries from advanced synthesis models, covering 16 DeepFake types across image, video, and audio modalities. The benchmark evaluates three key aspects: Perception, which measures the ability of a model to identify fine-grained manipulation artifacts using human-annotated evidence; Detection, which assesses classification performance across diverse forgery families and generators; and Hallucination, which quantifies the reliability of model-generated explanations. Experiments on state-of-the-art multimodal large language models show that accurate perception is essential for reliable detection, but hallucination can severely disrupt decision-making, revealing the interdependence of these three aspects. TriDF provides a unified framework for understanding the interaction between detection accuracy, evidence identification, and explanation reliability, offering a foundation for building trustworthy systems that address real-world synthetic media threats.
>
---
#### [replaced 078] PAVAS: Physics-Aware Video-to-Audio Synthesis
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视频到音频生成任务，旨在解决现有模型忽略物理因素的问题。提出PAVAS方法，结合物理推理生成更真实的音频。**

- **链接: [https://arxiv.org/pdf/2512.08282](https://arxiv.org/pdf/2512.08282)**

> **作者:** Oh Hyun-Bin; Yuhta Takida; Toshimitsu Uesaka; Tae-Hyun Oh; Yuki Mitsufuji
>
> **摘要:** Recent advances in Video-to-Audio (V2A) generation have achieved impressive perceptual quality and temporal synchronization, yet most models remain appearance-driven, capturing visual-acoustic correlations without considering the physical factors that shape real-world sounds. We present Physics-Aware Video-to-Audio Synthesis (PAVAS), a method that incorporates physical reasoning into a latent diffusion-based V2A generation through the Physics-Driven Audio Adapter (Phy-Adapter). The adapter receives object-level physical parameters estimated by the Physical Parameter Estimator (PPE), which uses a Vision-Language Model (VLM) to infer the moving-object mass and a segmentation-based dynamic 3D reconstruction module to recover its motion trajectory for velocity computation. These physical cues enable the model to synthesize sounds that reflect underlying physical factors. To assess physical realism, we curate VGG-Impact, a benchmark focusing on object-object interactions, and introduce Audio-Physics Correlation Coefficient (APCC), an evaluation metric that measures consistency between physical and auditory attributes. Comprehensive experiments show that PAVAS produces physically plausible and perceptually coherent audio, outperforming existing V2A models in both quantitative and qualitative evaluations. Visit this https URL for demo videos.
>
---
#### [replaced 079] Scaling Self-Supervised and Cross-Modal Pretraining for Volumetric CT Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17209](https://arxiv.org/pdf/2511.17209)**

> **作者:** Cris Claessens; Christiaan Viviers; Giacomo D'Amicantonio; Egor Bondarev; Fons van der Sommen
>
> **摘要:** We introduce SPECTRE, a fully transformer-based foundation model for volumetric computed tomography (CT). Our Self-Supervised & Cross-Modal Pretraining for CT Representation Extraction (SPECTRE) approach utilizes scalable 3D Vision Transformer architectures and modern self-supervised and vision-language pretraining strategies to learn general-purpose CT representations. Volumetric CT poses unique challenges, such as extreme token scaling, geometric anisotropy, and weak or noisy clinical supervision, that make standard transformer and contrastive learning recipes ineffective out of the box. The framework jointly optimizes a local transformer for high-resolution volumetric feature extraction and a global transformer for whole-scan context modeling, making large-scale 3D attention computationally tractable. Notably, SPECTRE is trained exclusively on openly available CT datasets, demonstrating that high-performing, generalizable representations can be achieved without relying on private data. Pretraining combines DINO-style self-distillation with SigLIP-based vision-language alignment using paired radiology reports, yielding features that are both geometrically consistent and clinically meaningful. Across multiple CT benchmarks, SPECTRE consistently outperforms prior CT foundation models in both zero-shot and fine-tuned settings, establishing SPECTRE as a scalable, open, and fully transformer-based foundation model for 3D medical imaging.
>
---
#### [replaced 080] Image-Adaptive GAN based Reconstruction
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/1906.05284](https://arxiv.org/pdf/1906.05284)**

> **作者:** Shady Abu Hussein; Tom Tirer; Raja Giryes
>
> **备注:** Published to AAAI 2020. Code available at this https URL
>
> **摘要:** In the recent years, there has been a significant improvement in the quality of samples produced by (deep) generative models such as variational auto-encoders and generative adversarial networks. However, the representation capabilities of these methods still do not capture the full distribution for complex classes of images, such as human faces. This deficiency has been clearly observed in previous works that use pre-trained generative models to solve imaging inverse problems. In this paper, we suggest to mitigate the limited representation capabilities of generators by making them image-adaptive and enforcing compliance of the restoration with the observations via back-projections. We empirically demonstrate the advantages of our proposed approach for image super-resolution and compressed sensing.
>
---
#### [replaced 081] Tracking by Detection and Query: An Efficient End-to-End Framework for Multi-Object Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.06197](https://arxiv.org/pdf/2411.06197)**

> **作者:** Shukun Jia; Shiyu Hu; Yichao Cao; Feng Yang; Xin Lu; Xiaobo Lu
>
> **备注:** Accepted by Pattern Recognition
>
> **摘要:** Multi-object tracking (MOT) is primarily dominated by two paradigms: tracking-by-detection (TBD) and tracking-by-query (TBQ). While TBD offers modular efficiency, its fragmented association pipeline often limits robustness in complex scenarios. Conversely, TBQ enhances semantic modeling end-to-end but suffers from high training costs and slow inference due to the tight coupling of detection and association. In this work, we propose the tracking-by-detection-and-query framework, TBDQ-Net, to advance the synergy between TBD and TBQ paradigms. By integrating a frozen detector with a lightweight associator, this architecture ensures intrinsic efficiency. Within this streamlined framework, we introduce tailored designs to address MOT-specific challenges. Concretely, we alleviate task conflicts and occlusions through the dual-stream update of the Basic Information Interaction (BII) module. The Content-Position Alignment (CPA) module further refines both content and positional components, providing well-aligned representations for association decoding. Extensive evaluations on DanceTrack, SportsMOT, and MOT20 benchmarks demonstrate that TBDQ-Net achieves a favorable efficiency-accuracy trade-off in challenging scenarios. Specifically, TBDQ-Net outperforms leading TBD methods by 6.0 IDF1 points on DanceTrack and achieves the best performance among TBQ methods in the crowded MOT20 benchmark. Relative to MOTRv2, TBDQ-Net reduces trainable parameters by approximately 80% while accelerating practical inference by 37.5%. These results highlight TBDQ-Net as an efficient alternative to heavy architectures, showcasing the efficacy of lightweight design. Source code is publicly available at this https URL.
>
---
#### [replaced 082] BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.18387](https://arxiv.org/pdf/2509.18387)**

> **作者:** Thomas Gossard; Filip Radovic; Andreas Ziegler; Andreas Zell
>
> **备注:** Accepted to CVPRW 2026 (CVsports)
>
> **摘要:** Motion blur reduces the clarity of fast-moving objects, posing challenges for detection systems, especially in racket sports, where balls often appear as streaks rather than distinct points. Existing labeling conventions mark the ball at the leading edge of the blur, introducing asymmetry and ignoring valuable motion cues correlated with velocity. This paper introduces a new labeling strategy that places the ball at the center of the blur streak and explicitly annotates blur attributes. Using this convention, we release a new table tennis ball detection dataset. We demonstrate that this labeling approach consistently enhances detection performance across various models. Furthermore, we introduce BlurBall, a model that jointly estimates ball position and motion blur attributes. By incorporating attention mechanisms such as Squeeze-and-Excitation over multi-frame inputs, we achieve state-of-the-art results in ball detection. Leveraging blur not only improves detection accuracy but also enables more reliable trajectory prediction, benefiting real-time sports analytics.
>
---
#### [replaced 083] Efficient Encoder-Free Fourier-based 3D Large Multimodal Model
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.23153](https://arxiv.org/pdf/2602.23153)**

> **作者:** Guofeng Mei; Wei Lin; Luigi Riz; Yujiao Wu; Yiming Wang; Fabio Poiesi
>
> **摘要:** Large Multimodal Models (LMMs) that process 3D data typically rely on heavy, pre-trained visual encoders to extract geometric features. While recent 2D LMMs have begun to eliminate such encoders for efficiency and scalability, extending this paradigm to 3D remains challenging due to the unordered and large-scale nature of point clouds. This leaves a critical unanswered question: How can we design an LMM that tokenizes unordered 3D data effectively and efficiently without a cumbersome encoder? We propose Fase3D, the first efficient encoder-free Fourier-based 3D scene LMM. Fase3D tackles the challenges of scalability and permutation invariance with a novel tokenizer that combines point cloud serialization and the Fast Fourier Transform (FFT) to approximate self-attention. This design enables an effective and computationally minimal architecture, built upon three key innovations: First, we represent large scenes compactly via structured superpoints. Second, our space-filling curve serialization followed by an FFT enables efficient global context modeling and graph-based token merging. Lastly, our Fourier-augmented LoRA adapters inject global frequency-aware interactions into the LLMs at a negligible cost. Fase3D achieves performance comparable to encoder-based 3D LMMs while being significantly more efficient in computation and parameters. Project website: this https URL.
>
---
#### [replaced 084] Disrupting Hierarchical Reasoning: Adversarial Protection for Geographic Privacy in Multimodal Reasoning Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.08503](https://arxiv.org/pdf/2512.08503)**

> **作者:** Jiaming Zhang; Che Wang; Yang Cao; Longtao Huang; Wei Yang Bryan Lim
>
> **备注:** ICLR 2026
>
> **摘要:** Multi-modal large reasoning models (MLRMs) pose significant privacy risks by inferring precise geographic locations from personal images through hierarchical chain-of-thought reasoning. Existing privacy protection techniques, primarily designed for perception-based models, prove ineffective against MLRMs' sophisticated multi-step reasoning processes that analyze environmental cues. We introduce \textbf{ReasonBreak}, a novel adversarial framework specifically designed to disrupt hierarchical reasoning in MLRMs through concept-aware perturbations. Our approach is founded on the key insight that effective disruption of geographic reasoning requires perturbations aligned with conceptual hierarchies rather than uniform noise. ReasonBreak strategically targets critical conceptual dependencies within reasoning chains, generating perturbations that invalidate specific inference steps and cascade through subsequent reasoning stages. To facilitate this approach, we contribute \textbf{GeoPrivacy-6K}, a comprehensive dataset comprising 6,341 ultra-high-resolution images ($\geq$2K) with hierarchical concept annotations. Extensive evaluation across seven state-of-the-art MLRMs (including GPT-o3, GPT-5, Gemini 2.5 Pro) demonstrates ReasonBreak's superior effectiveness, achieving a 14.4\% improvement in tract-level protection (33.8\% vs 19.4\%) and nearly doubling block-level protection (33.5\% vs 16.8\%). This work establishes a new paradigm for privacy protection against reasoning-based threats.
>
---
#### [replaced 085] FUSAR-GPT : A Spatiotemporal Feature-Embedded and Two-Stage Decoupled Visual Language Model for SAR Imagery
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.19190](https://arxiv.org/pdf/2602.19190)**

> **作者:** Xiaokun Zhang; Yi Yang; Ziqi Ye; Baiyun; Xiaorong Guo; Qingchen Fang; Ruyi Zhang; Xinpeng Zhou; Haipeng Wang
>
> **摘要:** Research on the intelligent interpretation of all-weather, all-time Synthetic Aperture Radar (SAR) is crucial for advancing remote sensing applications. In recent years, although Visual Language Models (VLMs) have demonstrated strong open-world understanding capabilities on RGB images, their performance is severely limited when directly applied to the SAR field due to the complexity of the imaging mechanism, sensitivity to scattering features, and the scarcity of high-quality text corpora. To systematically address this issue, we constructed the inaugural SAR Image-Text-AlphaEarth feature triplet dataset and developed FUSAR-GPT, a VLM specifically for SAR. FUSAR-GPT innovatively introduces a geospatial baseline model as a 'world knowledge' prior and embeds multi-source remote-sensing temporal features into the model's visual backbone via 'spatiotemporal anchors', enabling dynamic compensation for the sparse representation of targets in SAR images. Furthermore, we designed a two-stage SFT strategy to decouple the knowledge injection and task execution of large models. The spatiotemporal feature embedding and the two-stage decoupling paradigm enable FUSAR-GPT to achieve state-of-the-art performance across several typical remote sensing visual-language benchmark tests, significantly outperforming mainstream baseline models by over 10%.
>
---
#### [replaced 086] BCMDA: Bidirectional Correlation Maps Domain Adaptation for Mixed Domain Semi-Supervised Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24691](https://arxiv.org/pdf/2603.24691)**

> **作者:** Bentao Song; Jun Huang; Qingfeng Wang
>
> **备注:** Accepted at Neural Networks
>
> **摘要:** In mixed domain semi-supervised medical image segmentation (MiDSS), achieving superior performance under domain shift and limited annotations is challenging. This scenario presents two primary issues: (1) distributional differences between labeled and unlabeled data hinder effective knowledge transfer, and (2) inefficient learning from unlabeled data causes severe confirmation bias. In this paper, we propose the bidirectional correlation maps domain adaptation (BCMDA) framework to overcome these issues. On the one hand, we employ knowledge transfer via virtual domain bridging (KTVDB) to facilitate cross-domain learning. First, to construct a distribution-aligned virtual domain, we leverage bidirectional correlation maps between labeled and unlabeled data to synthesize both labeled and unlabeled images, which are then mixed with the original images to generate virtual images using two strategies, a fixed ratio and a progressive dynamic MixUp. Next, dual bidirectional CutMix is used to enable initial knowledge transfer within the fixed virtual domain and gradual knowledge transfer from the dynamically transitioning labeled domain to the real unlabeled domains. On the other hand, to alleviate confirmation bias, we adopt prototypical alignment and pseudo label correction (PAPLC), which utilizes learnable prototype cosine similarity classifiers for bidirectional prototype alignment between the virtual and real domains, yielding smoother and more compact feature representations. Finally, we use prototypical pseudo label correction to generate more reliable pseudo labels. Empirical evaluations on three public multi-domain datasets demonstrate the superiority of our method, particularly showing excellent performance even with very limited labeled samples. Code available at this https URL.
>
---
#### [replaced 087] Cross-Modal Urban Sensing: Evaluating Sound-Vision Alignment Across Street-Level and Aerial Imagery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.03388](https://arxiv.org/pdf/2506.03388)**

> **作者:** Pengyu Chen; Xiao Huang; Teng Fei; Sicheng Wang
>
> **备注:** 18 pages, 13 figures
>
> **摘要:** Environmental soundscapes convey substantial ecological and social information regarding urban environments; however, their potential remains largely untapped in large-scale geographic analysis. In this study, we investigate the extent to which urban sounds correspond with visual scenes by comparing various visual representation strategies in capturing acoustic semantics. We employ a multimodal approach that integrates geo-referenced sound recordings with both street-level and remote sensing imagery across three major global cities: London, New York, and Tokyo. Utilizing the AST model for audio, along with CLIP and RemoteCLIP for imagery, as well as CLIPSeg and Seg-Earth OV for semantic segmentation, we extract embeddings and class-level features to evaluate cross-modal similarity. The results indicate that street view embeddings demonstrate stronger alignment with environmental sounds compared to segmentation outputs, whereas remote sensing segmentation is more effective in interpreting ecological categories through a Biophony--Geophony--Anthrophony (BGA) framework. These findings imply that embedding-based models offer superior semantic alignment, while segmentation-based methods provide interpretable links between visual structure and acoustic ecology. This work advances the burgeoning field of multimodal urban sensing by offering novel perspectives for incorporating sound into geospatial analysis.
>
---
#### [replaced 088] Can Generalist Vision Language Models (VLMs) Rival Specialist Medical VLMs? Benchmarking and Strategic Insights
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.17337](https://arxiv.org/pdf/2506.17337)**

> **作者:** Yuan Zhong; Ruinan Jin; Qi Dou; Xiaoxiao Li
>
> **备注:** version 3
>
> **摘要:** Vision Language Models (VLMs) have shown promise in automating image diagnosis and interpretation in clinical settings. However, developing specialist medical VLMs requires substantial computational resources and carefully curated datasets, and it remains unclear under which conditions generalist and specialist medical VLMs each perform best. This study highlights the complementary strengths of specialist medical and generalist VLMs. Specialists remain valuable in modality-aligned use cases, but we find that efficiently fine-tuned generalist VLMs can achieve comparable or even superior performance in most tasks, particularly when transferring to unseen or rare OOD medical modalities. These results suggest that generalist VLMs, rather than being constrained by their lack of specialist medical pretraining, may offer a scalable and cost-effective pathway for advancing clinical AI development.
>
---
#### [replaced 089] From Exploration to Exploitation: A Two-Stage Entropy RLVR Approach for Noise-Tolerant MLLM Training
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07738](https://arxiv.org/pdf/2511.07738)**

> **作者:** Donglai Xu; Hongzheng Yang; Yuzhi Zhao; Pingping Zhang; Jinpeng Chen; Wenao Ma; Zhijian Hou; Mengyang Wu; Xiaolei Li; Senkang Hu; Ziyi Guan; Jason Chun Lok Li; Lai Man Po
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) for Multimodal Large Language Models (MLLMs) is highly dependent on high-quality labeled data, which is often scarce and prone to substantial annotation noise in real-world scenarios. Existing unsupervised RLVR methods, including pure entropy minimization, can overfit to incorrect labels and limit the crucial reward ranking signal for Group-Relative Policy Optimization (GRPO). To address these challenges and enhance noise tolerance, we propose a novel two-stage, token-level entropy optimization method for RLVR. This approach dynamically guides the model from exploration to exploitation during training. In the initial exploration phase, token-level entropy maximization promotes diverse and stochastic output generation, serving as a strong regularizer that prevents premature convergence to noisy labels and ensures sufficient intra-group variation, which enables more reliable reward gradient estimation in GRPO. As training progresses, the method transitions into the exploitation phase, where token-level entropy minimization encourages the model to produce confident and deterministic outputs, thereby consolidating acquired knowledge and refining prediction accuracy. Empirically, across three MLLM backbones - Qwen2-VL-2B, Qwen2-VL-7B, and Qwen2.5-VL-3B - spanning diverse noise settings and multiple tasks, our phased strategy consistently outperforms prior approaches by unifying and enhancing external, internal, and entropy-based methods, delivering robust and superior performance across the board.
>
---
#### [replaced 090] GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.20331](https://arxiv.org/pdf/2601.20331)**

> **作者:** Mai Su; Qihan Yu; Zhongtao Wang; Yilong Li; Chengwei Pan; Yisong Chen; Guoping Wang; Fei Zhu
>
> **摘要:** 3D Gaussian Splatting (3DGS) enables efficient rendering, yet accurate surface reconstruction remains challenging due to unreliable geometric supervision. Existing approaches predominantly rely on depth-based reprojection to infer visibility and enforce multi-view consistency, leading to a fundamental circular dependency: visibility estimation requires accurate depth, while depth supervision itself is conditioned on visibility. In this work, we revisit multi-view geometric supervision from the perspective of visibility modeling. Instead of inferring visibility from pixel-wise depth consistency, we explicitly model visibility at the level of Gaussian primitives. We introduce a Gaussian visibility-aware multi-view geometric consistency (GVMV) formulation, which aggregates cross-view visibility of shared Gaussians to construct reliable supervision over co-visible regions. To further incorporate monocular priors, we propose a progressive quadtree-calibrated depth alignment (QDC) strategy that performs block-wise affine calibration under visibility-aware guidance, effectively mitigating scale ambiguity while preserving local geometric structures. Extensive experiments on DTU and Tanks and Temples demonstrate that our method consistently improves reconstruction accuracy over prior Gaussian-based approaches. Our code is fully open-sourced and available at an anonymous repository: this https URL.
>
---
#### [replaced 091] iiANET: Inception Inspired Attention Hybrid Network for efficient Long-Range Dependency
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.07603](https://arxiv.org/pdf/2407.07603)**

> **作者:** Haruna Yunusa; Adamu Lawan; Abdulganiyu Abdu Yusuf
>
> **备注:** 17 pages, 7 figures. Published in Transactions on Machine Learning Research (TMLR). Available at this https URL
>
> **摘要:** The recent emergence of hybrid models has introduced a transformative approach to computer vision, gradually moving beyond conventional convolutional neural networks and vision transformers. However, efficiently combining these two approaches to better capture long-range dependencies in complex images remains a challenge. In this paper, we present iiANET (Inception Inspired Attention Network), an efficient hybrid visual backbone designed to improve the modeling of long-range dependencies in complex visual recognition tasks. The core innovation of iiANET is the iiABlock, a unified building block that integrates a modified global r-MHSA (Multi-Head Self-Attention) and convolutional layers in parallel. This design enables iiABlock to simultaneously capture global context and local details, making it effective for extracting rich and diverse features. By efficiently fusing these complementary representations, iiABlock allows iiANET to achieve strong feature interaction while maintaining computational efficiency. Extensive qualitative and quantitative evaluations on some SOTA benchmarks demonstrate improved performance.
>
---
#### [replaced 092] DIFEM: Key-points Interaction based Feature Extraction Module for Violence Recognition in Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.05386](https://arxiv.org/pdf/2412.05386)**

> **作者:** Himanshu Mittal; Suvramalya Basak; Anjali Gautam
>
> **备注:** Accepted in Signal Image and Video Processing
>
> **摘要:** Violence detection in surveillance videos is a critical task for ensuring public safety. As a result, there is increasing need for efficient and lightweight systems for automatic detection of violent behaviours. In this work, we propose an effective method which leverages human skeleton key-points to capture inherent properties of violence, such as rapid movement of specific joints and their close proximity. At the heart of our method is our novel Dynamic Interaction Feature Extraction Module (DIFEM) which captures features such as velocity, and joint intersections, effectively capturing the dynamics of violent behavior. With the features extracted by our DIFEM, we use various classification algorithms such as Random Forest, Decision tree, AdaBoost and k-Nearest Neighbor. Our approach has substantially lesser amount of parameter expense than the existing state-of-the-art (SOTA) methods employing deep learning techniques. We perform extensive experiments on three standard violence recognition datasets, showing promising performance in all three datasets. Our proposed method surpasses several SOTA violence recognition methods.
>
---
#### [replaced 093] Training-free Motion Factorization for Compositional Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09104](https://arxiv.org/pdf/2603.09104)**

> **作者:** Zixuan Wang; Ziqin Zhou; Feng Chen; Duo Peng; Yixin Hu; Changsheng Li; Yinjie Lei
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Compositional video generation aims to synthesize multiple instances with diverse appearance and motion. However, current approaches mainly focus on binding semantics, neglecting to understand diverse motion categories specified in prompts. In this paper, we propose a motion factorization framework that decomposes complex motion into three primary categories: motionlessness, rigid motion, and non-rigid motion. Specifically, our framework follows a planning before generation paradigm. (1) During planning, we reason about motion laws on the motion graph to obtain frame-wise changes in the shape and position of each instance. This alleviates semantic ambiguities in the user prompt by organizing it into a structured representation of instances and their interactions. (2) During generation, we modulate the synthesis of distinct motion categories in a disentangled manner. Conditioned on the motion cues, guidance branches stabilize appearance in motionless regions, preserve rigid-body geometry, and regularize local non-rigid deformations. Crucially, our two modules are model-agnostic, which can be seamlessly incorporated into various diffusion model architectures. Extensive experiments demonstrate that our framework achieves impressive performance in motion synthesis on real-world benchmarks. Code is available at this https URL.
>
---
#### [replaced 094] FigEx2: Visual-Conditioned Panel Detection and Captioning for Scientific Compound Figures
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出FigEx2，解决科学复合图无 caption 或 captions 短的问题，通过视觉条件检测面板并生成描述，提升图文本对齐效果。**

- **链接: [https://arxiv.org/pdf/2601.08026](https://arxiv.org/pdf/2601.08026)**

> **作者:** Jifeng Song; Arun Das; Pan Wang; Hui Ji; Kun Zhao; Yufei Huang
>
> **摘要:** Scientific compound figures combine multiple labeled panels into a single image. However, in a PMC-scale crawl of 346,567 compound figures, 16.3% have no caption and 1.8% only have captions shorter than ten words, causing them to be discarded by existing caption-decomposition pipelines. We propose FigEx2, a visual-conditioned framework that localizes panels and generates panel-wise captions directly from the image, converting otherwise unusable figures into aligned panel-text pairs for downstream pretraining and retrieval. To mitigate linguistic variance in open-ended captioning, we introduce a noise-aware gated fusion module that adaptively controls how caption features condition the detection query space, and employ a staged SFT+RL strategy with CLIP-based alignment and BERTScore-based semantic rewards. To support high-quality supervision, we curate BioSci-Fig-Cap, a refined benchmark for panel-level grounding, alongside cross-disciplinary test suites in physics and chemistry. FigEx2 achieves 0.728 mAP@0.5:0.95 for detection, outperforms Qwen3-VL-8B by 0.44 in METEOR and 0.22 in BERTScore, and transfers zero-shot to out-of-distribution scientific domains without fine-tuning.
>
---
#### [replaced 095] Dual Band Thermal Videography: Separating Time-Varying Reflection and Emission Near Ambient Conditions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.11334](https://arxiv.org/pdf/2509.11334)**

> **作者:** Sriram Narayanan; Mani Ramanagopal; Srinivasa G. Narasimhan
>
> **备注:** CVPR 2026. Project Page: this https URL
>
> **摘要:** Long-wave infrared radiation captured by a thermal camera includes (a) emission from an object governed by its temperature and emissivity, and (b) reflected radiation from the surrounding environment. Separating these components is a long-standing challenge in thermography. Even when using multiple bands, the problem is under-determined without priors on emissivity. This difficulty is amplified in near ambient conditions, where emitted and reflected signals are of comparable magnitude. We present a dual-band thermal videography framework that reduces this ambiguity by combining two complementary ideas at a per-pixel level: (i) spectral cues (ratio of emissivity between bands is unknown but fixed), and (ii) temporal cues (object radiation changes smoothly while background radiation changes rapidly). We derive an image formation model and an algorithm to jointly estimate the object's emissivity at each band, and the time-varying object and background temperatures. Experiments with calibrated and uncalibrated emissivities in everyday scenes (e.g., coffee pot heating up, palm print on mirrors, reflections of moving people), demonstrate robust separation and recovery of temperature fields.
>
---
#### [replaced 096] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.12533](https://arxiv.org/pdf/2603.12533)**

> **作者:** Yura Choi; Roy Miles; Rolandos Alexandros Potamias; Ismail Elezi; Jiankang Deng; Stefanos Zafeiriou
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Understanding and answering questions based on a user's pointing gesture is essential for next-generation egocentric AI assistants. However, current Multimodal Large Language Models (MLLMs) struggle with such tasks due to the lack of gesture-rich data and their limited ability to infer fine-grained pointing intent from egocentric video. To address this, we introduce EgoPointVQA, a dataset and benchmark for gesture-grounded egocentric question answering, comprising 4000 synthetic and 400 real-world videos across multiple deictic reasoning tasks. Built upon it, we further propose Hand Intent Tokens (HINT), which encodes tokens derived from 3D hand keypoints using an off-the-shelf reconstruction model and interleaves them with the model input to provide explicit spatial and temporal context for interpreting pointing intent. We show that our model outperforms others in different backbones and model sizes. In particular, HINT-14B achieves 68.1% accuracy, on average over 6 tasks, surpassing the state-of-the-art, InternVL3-14B, by 6.6%. To further facilitate the open research, we will release the code, model, and dataset. Project page: this https URL
>
---
#### [replaced 097] AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03100](https://arxiv.org/pdf/2508.03100)**

> **作者:** Yogesh Kulkarni; Pooyan Fazli
>
> **备注:** CVPR 2026
>
> **摘要:** Multimodal reasoning over long-horizon video is challenging due to the need for precise spatiotemporal fusion and alignment across modalities. While recent methods such as Group Relative Policy Optimization (GRPO) have shown promise in this domain, they suffer from three key limitations: (1) data inefficiency from their on-policy design, (2) a vanishing advantage problem, where identical or near-identical rewards within a group eliminate the learning signal by producing zero-valued advantages, and (3) uniform credit assignment that fails to emphasize critical reasoning steps. We introduce $\textbf{AVATAR}$ ($\textbf{A}$udio-$\textbf{V}$ideo $\textbf{A}$gen$\textbf{t}$ for $\textbf{A}$lignment and $\textbf{R}$easoning), a framework that addresses these limitations through two core components: (1) an off-policy training architecture that improves sample efficiency and resolves vanishing advantages by reusing past experiences with greater reward diversity, and (2) Temporal Advantage Shaping (TAS), a credit assignment strategy that emphasizes early (planning) and late (synthesis) reasoning phases. $\textbf{AVATAR}$ achieves strong performance across various benchmarks, outperforming the Qwen2.5-Omni baseline by $\mathbf{+5.4}$ on MMVU, $\mathbf{+4.9}$ on OmniBench, and $\mathbf{+4.5}$ on Video-Holmes. Furthermore, it surpasses standard GRPO by $\mathbf{+3.7}$ on OmniBench and $\mathbf{+1.9}$ on Video-Holmes, while demonstrating $\textbf{$5$$\times$ sample efficiency}$, requiring $80\%$ fewer generated completions to reach target performance.
>
---
#### [replaced 098] THEval. Evaluation Framework for Talking Head Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.04520](https://arxiv.org/pdf/2511.04520)**

> **作者:** Nabyl Quignon; Baptiste Chopin; Yaohui Wang; Antitza Dantcheva
>
> **备注:** CVPR 2025 Findings, Project Page: this https URL
>
> **摘要:** Video generation has achieved remarkable progress, with generated videos increasingly resembling real ones. However, the rapid advance in generation has outpaced the development of adequate evaluation metrics. Currently, the assessment of talking head generation primarily relies on limited metrics, evaluating general video quality, lip synchronization, and on conducting user studies. Motivated by this, we propose a new evaluation framework comprising 8 metrics related to three dimensions (i) quality, (ii) naturalness, and (iii) synchronization. In selecting the metrics, we place emphasis on efficiency, as well as alignment with human preferences. Based on this considerations, we streamline to analyze fine-grained dynamics of head, mouth, and eyebrows, as well as face quality. Our extensive experiments on 85,000 videos generated by 17 state-of-the-art models suggest that while many algorithms excel in lip synchronization, they face challenges with generating expressiveness and artifact-free details. These videos were generated based on a novel real dataset, that we have curated, in order to mitigate bias of training data. Our proposed benchmark framework is aimed at evaluating the improvement of generative methods. Original code, dataset and leaderboards will be publicly released and regularly updated with new methods, in order to reflect progress in the field.
>
---
#### [replaced 099] CLIP Is Shortsighted: Paying Attention Beyond the First Sentence
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22419](https://arxiv.org/pdf/2602.22419)**

> **作者:** Marc-Antoine Lavoie; Anas Mahmoud; Aldo Zaimi; Arsene Fansi Tchango; Steven L. Waslander
>
> **备注:** 20 pages, 15 figures, to be published in the CVPR 2026 proceedings
>
> **摘要:** CLIP models learn transferable multi-modal features via image-text contrastive learning on internet-scale data. They are widely used in zero-shot classification, multi-modal retrieval, text-to-image diffusion, and as image encoders in large vision-language models. However, CLIP's pretraining is dominated by images paired with short captions, biasing the model toward encoding simple descriptions of salient objects and leading to coarse alignment on complex scenes and dense descriptions. While recent work mitigates this by fine-tuning on small-scale long-caption datasets, we identify an important common bias: both human- and LLM-generated long captions typically begin with a one-sentence summary followed by a detailed description. We show that this acts as a shortcut during training, concentrating attention on the opening sentence and early tokens and weakening alignment over the rest of the caption. To resolve this, we introduce DeBias-CLIP, which removes the summary sentence during training and applies sentence sub-sampling and text token padding to distribute supervision across all token positions. DeBias-CLIP achieves state-of-the-art long-text retrieval, improves short-text retrieval, and is less sensitive to sentence order permutations. It is a drop-in replacement for Long-CLIP with no additional trainable parameters.
>
---
#### [replaced 100] SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.04361](https://arxiv.org/pdf/2602.04361)**

> **作者:** Zekun Li; Ning Wang; Tongxin Bai; Changwang Mei; Peisong Wang; Shuang Qiu; Jian Cheng
>
> **备注:** CVPR 2026
>
> **摘要:** Visual AutoRegressive (VAR) modeling has garnered significant attention for its innovative next-scale prediction paradigm. However, mainstream VAR paradigms attend to all tokens across historical scales at each autoregressive step. As the next scale resolution grows, the computational complexity of attention increases quartically with resolution, causing substantial latency. Prior accelerations often skip high-resolution scales, which speeds up inference but discards high-frequency details and harms image quality. To address these problems, we present \textbf{SparVAR}, a training-free acceleration framework that exploits three properties of VAR attention: \textbf{(i) strong attention sinks}, \textbf{(ii) cross-scale activation similarity}, and \textbf{(iii) pronounced locality}. Specifically, we dynamically predict the sparse attention pattern of later high-resolution scales from a sparse decision scale, and construct scale self-similar sparse attention via an efficient index-mapping mechanism, enabling high-efficiency sparse attention computation at large scales. Furthermore, we propose cross-scale local sparse attention and implement an efficient block-wise sparse kernel, which achieves $\mathbf{> 5\times}$ faster forward speed than FlashAttention. Extensive experiments demonstrate that the proposed SparVAR can reduce the generation time of an 8B model producing $1024\times1024$ high-resolution images to the \textbf{1s}, \textbf{without skipping the last scales}. Compared with the VAR baseline accelerated by FlashAttention, our method achieves a $\mathbf{1.57\times}$ speed-up while preserving almost all high-frequency details. When combined with existing scale-skipping strategies, SparVAR attains up to a $\mathbf{2.28\times}$ acceleration, while maintaining competitive visual generation quality. Code is available at \href{this https URL}{SparVAR}.
>
---
#### [replaced 101] OMG-Avatar: One-shot Multi-LOD Gaussian Head Avatar
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01506](https://arxiv.org/pdf/2603.01506)**

> **作者:** Jianqiang Ren; Lin Liu; Steven Hoi
>
> **摘要:** We propose OMG-Avatar, a novel One-shot method that leverages a Multi-LOD (Level-of-Detail) Gaussian representation for animatable 3D head reconstruction from a single image in 0.2s. Our method enables LOD head avatar modeling using a unified model that accommodates diverse hardware capabilities and inference speed requirements. To capture both global and local facial characteristics, we employ a transformer-based architecture for global feature extraction and projection-based sampling for local feature acquisition. These features are effectively fused under the guidance of a depth buffer, ensuring occlusion plausibility. We further introduce a coarse-to-fine learning paradigm to support Level-of-Detail functionality and enhance the perception of hierarchical details. To address the limitations of 3DMMs in modeling non-head regions such as the shoulders, we introduce a multi-region decomposition scheme in which the head and shoulders are predicted separately and then integrated through cross-region combination. Extensive experiments demonstrate that OMG-Avatar outperforms state-of-the-art methods in reconstruction quality, reenactment performance, and computational efficiency. The project homepage is this https URL .
>
---
#### [replaced 102] VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VLM-3R，解决视觉-语言模型在3D空间理解上的不足，通过单目视频和3D重建指令微调，提升空间与时间推理能力。**

- **链接: [https://arxiv.org/pdf/2505.20279](https://arxiv.org/pdf/2505.20279)**

> **作者:** Zhiwen Fan; Jian Zhang; Renjie Li; Junge Zhang; Runjin Chen; Hezhen Hu; Kevin Wang; Huaizhi Qu; Shijie Zhou; Dilin Wang; Zhicheng Yan; Hongyu Xu; Justin Theiss; Tianlong Chen; Jiachen Li; Zhengzhong Tu; Zhangyang Wang; Rakesh Ranjan
>
> **备注:** Project Page: this https URL
>
> **摘要:** The rapid advancement of Large Multimodal Models (LMMs) for 2D images and videos has motivated extending these models to understand 3D scenes, aiming for human-like visual-spatial intelligence. Nevertheless, achieving deep spatial understanding comparable to human capabilities poses significant challenges in model encoding and data acquisition. Existing methods frequently depend on external depth sensors for geometry capture or utilize off-the-shelf algorithms for pre-constructing 3D maps, thereby limiting their scalability, especially with prevalent monocular video inputs and for time-sensitive applications. In this work, we introduce VLM-3R, a unified framework for Vision-Language Models (VLMs) that incorporates 3D Reconstructive instruction tuning. VLM-3R processes monocular video frames by employing a geometry encoder to derive implicit 3D tokens that represent spatial understanding. Leveraging our Spatial-Visual-View Fusion and over 200K curated 3D reconstructive instruction tuning question-answer (QA) pairs, VLM-3R effectively aligns real-world spatial context with language instructions. This enables monocular 3D spatial assistance and embodied reasoning. To facilitate the evaluation of temporal reasoning, we introduce the Vision-Spatial-Temporal Intelligence benchmark, featuring over 138.6K QA pairs across five distinct tasks focused on evolving spatial relationships. Extensive experiments demonstrate that our model, VLM-3R, not only facilitates robust visual-spatial reasoning but also enables the understanding of temporal 3D context changes, excelling in both accuracy and scalability.
>
---
#### [replaced 103] Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.03821](https://arxiv.org/pdf/2505.03821)**

> **作者:** Gracjan Góral; Alicja Ziarko; Piotr Miłoś; Michał Nauman; Maciej Wołczyk; Michał Kosiński
>
> **备注:** Accepted at CVPR 2026 Findings
>
> **摘要:** We investigate the ability of Vision Language Models (VLMs) to perform visual perspective taking using a new set of visual tasks inspired by established human tests. Our approach leverages carefully controlled scenes in which a single humanoid minifigure is paired with a single object. By systematically varying spatial configurations -- such as object position relative to the minifigure and the minifigure's orientation -- and using both bird's-eye and surface-level views, we created 144 unique visual tasks. Each task is paired with a series of 7 diagnostic questions designed to assess three levels of visual cognition: scene understanding, spatial reasoning, and visual perspective taking. We evaluate several high-performing models, including Gemini Robotics-ER 1.5, Llama-3.2-11B-Vision-Instruct, and variants of Claude Sonnet, GPT-4, and Qwen3, and find that while they excel at scene understanding, performance declines markedly on spatial reasoning and deteriorates further on perspective taking. Our analysis suggests a gap between surface-level object recognition and the deeper spatial and perspective reasoning required for complex visual tasks, pointing to the need for integrating explicit geometric representations and tailored training protocols in future VLM development.
>
---
#### [replaced 104] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）预训练任务，旨在从工业视频中自动提取结构化数据。通过动作分割和语义聚类，解决无监督数据利用问题，提升制造中的具身AI性能。**

- **链接: [https://arxiv.org/pdf/2511.21428](https://arxiv.org/pdf/2511.21428)**

> **作者:** Jiajie Zhang; Sören Schwertfeger; Alexander Kleiner
>
> **备注:** 10 pages, 5 figures, Accepted to CVPR 2026
>
> **摘要:** We present a novel unsupervised framework to unlock vast unlabeled human demonstration data from continuous industrial video streams for Vision-Language-Action (VLA) model pre-training. Our method first trains a lightweight motion tokenizer to encode motion dynamics, then employs an unsupervised action segmenter leveraging a novel "Latent Action Energy" metric to discover and segment semantically coherent action primitives. The pipeline outputs both segmented video clips and their corresponding latent action sequences, providing structured data directly suitable for VLA pre-training. Evaluations on public benchmarks and a proprietary electric motor assembly dataset demonstrate effective segmentation of key tasks performed by humans at workstations. Further clustering and quantitative assessment via a Vision-Language Model confirm the semantic coherence of the discovered action primitives. To our knowledge, this is the first fully automated end-to-end system for extracting and organizing VLA pre-training data from unstructured industrial videos, offering a scalable solution for embodied AI integration in manufacturing.
>
---
#### [replaced 105] CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.26425](https://arxiv.org/pdf/2603.26425)**

> **作者:** Moritz Nottebaum; Matteo Dunnhofer; Christian Micheloni
>
> **备注:** Accepted at CVPR Findings 2026
>
> **摘要:** Recent research on vision backbone architectures has predominantly focused on optimizing efficiency for hardware platforms with high parallel processing capabilities. This category increasingly includes embedded systems such as mobile phones and embedded AI accelerator modules. In contrast, CPUs do not have the possibility to parallelize operations in the same manner, wherefore models benefit from a specific design philosophy that balances amount of operations (MACs) and hardware-efficient execution by having high MACs per second (MACpS). In pursuit of this, we investigate two modifications to standard convolutions, aimed at reducing computational cost: grouping convolutions and reducing kernel sizes. While both adaptations substantially decrease the total number of MACs required for inference, sustaining low latency necessitates preserving hardware-efficiency. Our experiments across diverse CPU devices confirm that these adaptations successfully retain high hardware-efficiency on CPUs. Based on these insights, we introduce CPUBone, a new family of vision backbone models optimized for CPU-based inference. CPUBone achieves state-of-the-art Speed-Accuracy Trade-offs (SATs) across a wide range of CPU devices and effectively transfers its efficiency to downstream tasks such as object detection and semantic segmentation. Models and code are available at this https URL.
>
---
#### [replaced 106] Rolling Sink: Bridging Limited-Horizon Training and Open-Ended Testing in Autoregressive Video Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.07775](https://arxiv.org/pdf/2602.07775)**

> **作者:** Haodong Li; Shaoteng Liu; Zhe Lin; Manmohan Chandraker
>
> **备注:** v5: Fix some typos. Figures were compressed to 150 dpi to comply with arXiv's submission size limit. Project page: this https URL
>
> **摘要:** Recently, autoregressive (AR) video diffusion models have achieved remarkable performance. However, due to their limited training durations, a train-test gap emerges when testing at longer horizons, leading to rapid visual degradations. Following Self Forcing, which studies the train-test gap within the training duration, this work studies the train-test gap beyond the training duration, i.e., the gap between the limited horizons during training and open-ended horizons during testing. Since open-ended testing can extend beyond any finite training window, and long-video training is computationally expensive, we pursue a training-free solution to bridge this gap. To explore a training-free solution, we conduct a systematic analysis of AR cache maintenance. These insights lead to Rolling Sink. Built on Self Forcing (trained on only 5s clips), Rolling Sink effectively scales the AR video synthesis to ultra-long durations (e.g., 5-30 minutes at 16 FPS) at test time, with consistent subjects, stable colors, coherent structures, and smooth motions. As demonstrated by extensive experiments, Rolling Sink achieves superior long-horizon visual fidelity and temporal consistency compared to SOTA baselines. Project page: this https URL
>
---
#### [replaced 107] SAM 3: Segment Anything with Concepts
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.16719](https://arxiv.org/pdf/2511.16719)**

> **作者:** Nicolas Carion; Laura Gustafson; Yuan-Ting Hu; Shoubhik Debnath; Ronghang Hu; Didac Suris; Chaitanya Ryali; Kalyan Vasudev Alwala; Haitham Khedr; Andrew Huang; Jie Lei; Tengyu Ma; Baishan Guo; Arpit Kalla; Markus Marks; Joseph Greer; Meng Wang; Peize Sun; Roman Rädle; Triantafyllos Afouras; Effrosyni Mavroudi; Katherine Xu; Tsung-Han Wu; Yu Zhou; Liliane Momeni; Rishi Hazra; Shuangrui Ding; Sagar Vaze; Francois Porcher; Feng Li; Siyuan Li; Aishwarya Kamath; Ho Kei Cheng; Piotr Dollár; Nikhila Ravi; Kate Saenko; Pengchuan Zhang; Christoph Feichtenhofer
>
> **摘要:** We present Segment Anything Model (SAM) 3, a unified model that detects, segments, and tracks objects in images and videos based on concept prompts, which we define as either short noun phrases (e.g., "yellow school bus"), image exemplars, or a combination of both. Promptable Concept Segmentation (PCS) takes such prompts and returns segmentation masks and unique identities for all matching object instances. To advance PCS, we build a scalable data engine that produces a high-quality dataset with 4M unique concept labels, including hard negatives, across images and videos. Our model consists of an image-level detector and a memory-based video tracker that share a single backbone. Recognition and localization are decoupled with a presence head, which boosts detection accuracy. SAM 3 doubles the accuracy of existing systems in both image and video PCS, and improves previous SAM capabilities on visual segmentation tasks. We open source SAM 3 along with our new Segment Anything with Concepts (SA-Co) benchmark for promptable concept segmentation.
>
---
#### [replaced 108] FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21356](https://arxiv.org/pdf/2603.21356)**

> **作者:** Yuqiu Liu; Jialin Song; Marissa Ramirez de Chanlatte; Rochishnu Chowdhury; Rushil Paresh Desai; Wuyang Chen; Daniel Martin; Michael W. Mahoney
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Real objects that inhabit the physical world follow physical laws and thus behave plausibly during interaction with other physical objects. However, current methods that perform 3D reconstructions of real-world scenes from multi-view 2D images optimize primarily for visual fidelity, i.e., they train with photometric losses and reason about uncertainty in the image or representation space. This appearance-centric view overlooks body contacts and couplings, conflates function-critical regions (e.g., aerodynamic or hydrodynamic surfaces) with ornamentation, and reconstructs structures suboptimally, even when physical regularizers are added. All these can lead to unphysical and implausible interactions. To address this, we consider the question: How can 3D reconstruction become aware of real-world interactions and underlying object functionality, beyond visual cues? To answer this question, we propose FluidGaussian, a plug-and-play method that tightly couples geometry reconstruction with ubiquitous fluid-structure interactions to assess surface quality at high granularity. We define a simulation-based uncertainty metric induced by fluid simulations and integrate it with active learning to prioritize views that improve both visual and physical fidelity. In an empirical evaluation on NeRF Synthetic (Blender), Mip-NeRF 360, and DrivAerNet++, our FluidGaussian method yields up to +8.6% visual PSNR (Peak Signal-to-Noise Ratio) and -62.3% velocity divergence during fluid simulations. Our code is available at this https URL.
>
---
#### [replaced 109] Listen to Rhythm, Choose Movements: Autoregressive Multimodal Dance Generation via Diffusion and Mamba with Decoupled Dance Dataset
- **分类: cs.GR; cs.CV; cs.HC; cs.LG; cs.SD**

- **简介: 该论文属于舞蹈生成任务，旨在解决现有方法语义控制不足和长序列不连贯的问题。提出LRCM框架，结合扩散模型与Mamba模块，实现多模态引导的自回归舞蹈生成。**

- **链接: [https://arxiv.org/pdf/2601.03323](https://arxiv.org/pdf/2601.03323)**

> **作者:** Oran Duan; Yinghua Shen; Yingzhu Lv; Luyang Jie; Yaxin Liu; Qiong Wu
>
> **备注:** 12 pages, 13 figures
>
> **摘要:** Advances in generative models and sequence learning have greatly promoted research in dance motion generation, yet current methods still suffer from coarse semantic control and poor coherence in long sequences. In this work, we present Listen to Rhythm, Choose Movements (LRCM), a multimodal-guided diffusion framework supporting both diverse input modalities and autoregressive dance motion generation. We explore a feature decoupling paradigm for dance datasets and generalize it to the Motorica Dance dataset, separating motion capture data, audio rhythm, and professionally annotated global and local text descriptions. Our diffusion architecture integrates an audio-latent Conformer and a text-latent Cross-Conformer, and incorporates a Motion Temporal Mamba Module (MTMM) to enable smooth, long-duration autoregressive synthesis. Experimental results indicate that LRCM delivers strong performance in both functional capability and quantitative metrics, demonstrating notable potential in multimodal input scenarios and extended sequence generation. We will release the full codebase, dataset, and pretrained models publicly upon acceptance.
>
---
#### [replaced 110] Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18174](https://arxiv.org/pdf/2511.18174)**

> **作者:** Mukai Yu; Mosam Dabhi; Liuyue Xie; Sebastian Scherer; László A. Jeni
>
> **备注:** Accepted to CVPR 2026. Camera-ready version. Added computation benchmark
>
> **摘要:** Modern perception increasingly relies on fisheye, panoramic, and other wide field-of-view (FoV) cameras, yet most pipelines still apply planar CNNs designed for pinhole imagery on 2D grids, where pixel-space neighborhoods misrepresent physical adjacency and models are sensitive to global rotations. Traditional spherical CNNs partially address this mismatch but require costly spherical harmonic transform that constrains resolution and efficiency. We present Unified Spherical Frontend (USF), a distortion-free lens-agnostic framework that transforms images from any calibrated camera onto the unit sphere via ray-direction correspondences, and performs spherical resampling, convolution, and pooling canonically in the spatial domain. USF is modular: projection, location sampling, value interpolation, and resolution control are fully decoupled. Its configurable distance-only convolution kernels offer rotation-equivariance, mirroring translation-equivariance in planar CNNs while avoiding harmonic transforms entirely. We compare multiple standard planar backbones with their spherical counterparts across classification, detection, and segmentation tasks on synthetic (Spherical MNIST) and real-world (PANDORA, Stanford 2D-3D-S) datasets, and stress-test robustness to extreme lens distortions, varying FoV, and arbitrary rotations. USF scales efficiently to high-resolution spherical imagery and maintains less than 1% performance drop under random test-time rotations without training-time rotational augmentation, and enables zero-shot generalization to any unseen (wide-FoV) lenses with minimal performance degradation.
>
---
#### [replaced 111] Measuring the (Un)Faithfulness of Concept-Based Explanations
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.10833](https://arxiv.org/pdf/2504.10833)**

> **作者:** Shubham Kumar; Narendra Ahuja
>
> **备注:** To appear in CVPR 2026
>
> **摘要:** Deep vision models perform input-output computations that are hard to interpret. Concept-based explanation methods (CBEMs) increase interpretability by re-expressing parts of the model with human-understandable semantic units, or concepts. Checking if the derived explanations are faithful -- that is, they represent the model's internal computation -- requires a surrogate that combines concepts to compute the output. Simplifications made for interpretability inevitably reduce faithfulness, resulting in a tradeoff between the two. State-of-the-art unsupervised CBEMs (U-CBEMs) are seemingly more interpretable, while also being more faithful to the model. However, we observe that the reported improvement in faithfulness artificially results from either (1) using overly complex surrogates, which introduces an unmeasured cost to the explanation's interpretability, or (2) relying on deletion-based approaches that, as we demonstrate, do not properly measure faithfulness. We propose Surrogate Faithfulness (SURF), which (1) replaces prior complex surrogates with a simple, linear surrogate that measures faithfulness without changing the explanation's interpretability and (2) introduces well-motivated metrics that assess loss across all output classes, not just the predicted class. We validate SURF with a measure-over-measure study by proposing a simple sanity check -- explanations with random concepts should be less faithful -- which prior surrogates fail. SURF enables the first reliable faithfulness benchmark of U-CBEMs, revealing that many visually compelling U-CBEMs are not faithful. Code is released at this https URL .
>
---
#### [replaced 112] LitePT: Lighter Yet Stronger Point Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13689](https://arxiv.org/pdf/2512.13689)**

> **作者:** Yuanwen Yue; Damien Robert; Jianyuan Wang; Sunghwan Hong; Jan Dirk Wegner; Christian Rupprecht; Konrad Schindler
>
> **备注:** CVPR 2026, Project page: this https URL
>
> **摘要:** Modern neural architectures for 3D point cloud processing contain both convolutional layers and attention blocks, but the best way to assemble them remains unclear. We analyse the role of different computational blocks in 3D point cloud networks and find an intuitive behaviour: convolution is adequate to extract low-level geometry at high-resolution in early layers, where attention is expensive without bringing any benefits; attention captures high-level semantics and context in low-resolution, deep layers more efficiently, where convolution inflates the parameter count. Guided by this design principle, we propose a new, improved 3D point cloud backbone that employs convolutions in early stages and switches to attention for deeper layers. To avoid the loss of spatial layout information when discarding redundant convolution layers, we introduce a novel, parameter-free 3D positional encoding, PointROPE. The resulting LitePT model has $3.6\times$ fewer parameters, runs $2\times$ faster, and uses $2\times$ less memory than the state-of-the-art Point Transformer V3, but nonetheless matches or outperforms it on a range of tasks and datasets. Code and models are available at: this https URL.
>
---
#### [replaced 113] Match Stereo Videos via Bidirectional Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.20283](https://arxiv.org/pdf/2409.20283)**

> **作者:** Junpeng Jing; Ye Mao; Anlan Qiu; Krystian Mikolajczyk
>
> **备注:** TPAMI 2026
>
> **摘要:** Video stereo matching is the task of estimating consistent disparity maps from rectified stereo videos. There is considerable scope for improvement in both datasets and methods within this area. Recent learning-based methods often focus on optimizing performance for independent stereo pairs, leading to temporal inconsistencies in videos. Existing video methods typically employ sliding window operation over time dimension, which can result in low-frequency oscillations corresponding to the window size. To address these challenges, we propose a bidirectional alignment mechanism for adjacent frames as a fundamental operation. Building on this, we introduce a novel video processing framework, BiDAStereo, and a plugin stabilizer network, BiDAStabilizer, compatible with general image-based methods. Regarding datasets, current synthetic object-based and indoor datasets are commonly used for training and benchmarking, with a lack of outdoor nature scenarios. To bridge this gap, we present a realistic synthetic dataset and benchmark focused on natural scenes, along with a real-world dataset captured by a stereo camera in diverse urban scenes for qualitative evaluation. Extensive experiments on in-domain, out-of-domain, and robustness evaluation demonstrate the contribution of our methods and datasets, showcasing improvements in prediction quality and achieving state-of-the-art results on various commonly used benchmarks. The project page, demos, code, and datasets are available at: this https URL.
>
---
#### [replaced 114] PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.26068](https://arxiv.org/pdf/2603.26068)**

> **作者:** Elkhan Ismayilzada; Yufei Zhang; Zijun Cui
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Significant advancements made in reconstructing hands from images have delivered accurate single-frame estimates, yet they often lack physics consistency and provide no notion of how confidently the motion satisfies physics. In this paper, we propose a novel physics-aware conditional diffusion framework that refines noisy pose sequences into physically plausible hand motion while estimating the physics variance in motion estimates. Building on a MeshCNN-Transformer backbone, we formulate Euler-Lagrange dynamics for articulated hands. Unlike prior works that enforce zero residuals, we treat the resulting dynamic residuals as virtual observables to more effectively integrate physics. Through a last-layer Laplace approximation, our method produces per-joint, per-time variances that measure physics consistency and offers interpretable variance maps indicating where physical consistency weakens. Experiments on two well-known hand datasets show consistent gains over strong image-based initializations and competitive video-based methods. Qualitative results confirm that our variance estimations are aligned with the physical plausibility of the motion in image-based estimates.
>
---
#### [replaced 115] ReSAM: Refine, Requery, and Reinforce: Self-Prompting Point-Supervised Segmentation for Remote Sensing Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21606](https://arxiv.org/pdf/2511.21606)**

> **作者:** M.Naseer Subhani
>
> **摘要:** Interactive segmentation models such as the Segment Anything Model (SAM) have demonstrated remarkable generalization on natural images, but they perform suboptimally on remote sensing imagery (RSI) due to severe domain shifts and the scarcity of dense annotations. To address this limitation, we propose a point-supervised, self-prompting framework that adapts SAM to RSI using only sparse point annotations. Our method employs a Refine-Requery-Reinforce loop, in which coarse pseudo-masks are generated from initial points (Refine), improved with self-constructed box prompts (Requery), and embeddings are aligned with Soft Semantic Alignment (SSA) to mitigate error propagation. (Reinforce). Without relying on full-mask supervision, our approach progressively enhances SAM's segmentation quality and domain robustness through self-guided prompt adaptation. We evaluate our proposed method on three RSI benchmark datasets, WHU, HRSID, and NWPU VHR-10, demonstrating that it consistently outperforms pretrained SAM and recent point-supervised segmentation methods. Compared to the fully supervised model, our approach reduces the performance gap to 1.3% (WHU), 4.9% (HRSID), and 8.5% (NWPU) while relying only on 1-point annotations. Our results demonstrate that self-prompting and semantic alignment provide an efficient path towards scalable, point-level adaptation of foundation segmentation models for remote sensing applications.
>
---
#### [replaced 116] Clinical application of HEDI for biomechanical evaluation and visualisation in incisional hernia repair
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2307.01502](https://arxiv.org/pdf/2307.01502)**

> **作者:** Philipp D. Lösel; Jacob J. Relle; Samuel Voß; Ramesch Raschidi; Regine Nessel; Johannes Görich; Mark O. Wielpütz; Thorsten Löffler; Vincent Heuveline; Friedrich Kallinowski
>
> **备注:** 15 pages, 6 figures, this is the author's accepted manuscript of an article published in Communications Medicine (2026). The final version is available online at: this https URL
>
> **摘要:** Background: Abdominal wall defects, such as incisional hernias, are a common source of pain and discomfort and often require repeated surgical interventions. Traditional mesh repair techniques typically rely on fixed overlap based on defect size, without considering important biomechanical factors like muscle activity, internal pressure, and tissue elasticity. This study aims to introduce a biomechanical approach to incisional hernia repair that accounts for abdominal wall instability and to evaluate a visualisation tool designed to support surgical planning. Methods: We developed HEDI, a tool that uses computed tomography with Valsalva maneuver to automatically assess hernia size, volume, and abdominal wall instability. This tool was applied in the preoperative evaluation of 31 patients undergoing incisional hernia repair. Surgeries were performed concurrently with the development of the tool, and patient outcomes were monitored over a three-year period. Results: Here we show that all 31 patients remain free of pain and hernia recurrence three years after surgery. The tool provides valuable visual insights into abdominal wall dynamics, supporting surgical decision-making. However, it should be used as an adjunct rather than a standalone guide. Conclusions: This study presents a biomechanical strategy for hernia repair and introduces a visualisation tool that enhances preoperative assessment. While early results are promising, the tool's evolving nature and its role as a visual aid should be considered when interpreting outcomes. Further research is needed to validate its broader clinical utility.
>
---
#### [replaced 117] SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting
- **分类: cs.CV; cs.GR; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.02172](https://arxiv.org/pdf/2512.02172)**

> **作者:** Pranav Asthana; Alex Hanson; Allen Tu; Tom Goldstein; Matthias Zwicker; Amitabh Varshney
>
> **备注:** Project Page: this https URL
>
> **摘要:** 3D Gaussian Splatting (3DGS) enables high-quality novel view synthesis, motivating interest in generating higher-resolution renders than those available during training. A natural strategy is to apply super-resolution (SR) to low-resolution (LR) input views, but independently enhancing each image introduces multi-view inconsistencies, leading to blurry renders. Prior methods attempt to mitigate these inconsistencies through learned neural components, temporally consistent video priors, or joint optimization on LR and SR views, but all uniformly apply SR across every image. In contrast, our key insight is that close-up LR views may contain high-frequency information for regions also captured in more distant views and that we can use the camera pose relative to scene geometry to inform where to add SR content. Building on this insight, we propose SplatSuRe, a method that selectively applies SR content only in undersampled regions lacking high-frequency supervision, yielding sharper and more consistent results. Across Tanks & Temples, Deep Blending, and Mip-NeRF 360, our approach surpasses baselines in both fidelity and perceptual quality. Notably, our gains are most significant in localized foreground regions where higher detail is desired.
>
---
#### [replaced 118] Omni-Weather: A Unified Multimodal Model for Weather Radar Understanding and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.21643](https://arxiv.org/pdf/2512.21643)**

> **作者:** Zhiwang Zhou; Yuandong Pu; Xuming He; Yidi Liu; Yixin Chen; Junchao Gong; Xiang Zhuang; Wanghan Xu; Qinglong Cao; Shixiang Tang; Yihao Liu; Wenlong Zhang; Lei Bai
>
> **摘要:** Weather modeling requires both accurate prediction and mechanistic interpretation, yet existing methods treat these goals in isolation, separating generation from understanding. To address this gap, we present Omni-Weather, the first multimodal foundation model that unifies weather generation and understanding within a single architecture. Omni-Weather integrates a radar encoder for weather generation tasks, followed by unified processing using a shared self-attention mechanism. Moreover, we construct a Chain-of-Thought dataset for causal reasoning in weather generation, enabling interpretable outputs and improved perceptual quality. Extensive experiments show Omni-Weather achieves state-of-the-art performance in both weather generation and understanding. Our findings further indicate that generative and understanding tasks in the weather domain can mutually enhance each other. Omni-Weather also demonstrates the feasibility and value of unifying weather generation and understanding.
>
---
#### [replaced 119] LAMP: Language-Assisted Motion Planning for Controllable Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03619](https://arxiv.org/pdf/2512.03619)**

> **作者:** Muhammed Burak Kizil; Enes Sanli; Niloy J. Mitra; Erkut Erdem; Aykut Erdem; Duygu Ceylan
>
> **备注:** CVPR 2026. Project Page: this https URL
>
> **摘要:** Video generation has achieved remarkable progress in visual fidelity and controllability, enabling conditioning on text, layout, or motion. Among these, motion control - specifying object dynamics and camera trajectories - is essential for composing complex, cinematic scenes, yet existing interfaces remain limited. We introduce LAMP that leverages large language models (LLMs) as motion planners to translate natural language descriptions into explicit 3D trajectories for dynamic objects and (relatively defined) cameras. LAMP defines a motion domain-specific language (DSL), inspired by cinematography conventions. By harnessing program synthesis capabilities of LLMs, LAMP generates structured motion programs from natural language, which are deterministically mapped to 3D trajectories. We construct a large-scale procedural dataset pairing natural text descriptions with corresponding motion programs and 3D trajectories. Experiments demonstrate LAMP's improved performance in motion controllability and alignment with user intent compared to state-of-the-art alternatives establishing the first framework for generating both object and camera motions directly from natural language specifications. Code, models and data are available on our project page.
>
---
#### [replaced 120] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.14255](https://arxiv.org/pdf/2510.14255)**

> **作者:** Liao Shen; Wentao Jiang; Yiran Zhu; Jiahe Li; Tiezheng Ge; Zhiguo Cao; Bo Zheng
>
> **备注:** accepted by CVPR 2026
>
> **摘要:** Recent advances in image-to-video (I2V) generation have achieved remarkable progress in synthesizing high-quality, temporally coherent videos from static images. Among all the applications of I2V, human-centric video generation includes a large portion. However, existing I2V models encounter difficulties in maintaining identity consistency between the input human image and the generated video, especially when the person in the video exhibits significant expression changes and movements. This issue becomes critical when the human face occupies merely a small fraction of the image. Since humans are highly sensitive to identity variations, this poses a critical yet under-explored challenge in I2V generation. In this paper, we propose Identity-Preserving Reward-guided Optimization (IPRO), a novel video diffusion framework based on reinforcement learning to enhance identity preservation. Instead of introducing auxiliary modules or altering model architectures, our approach introduces a direct and effective tuning algorithm that optimizes diffusion models using a face identity scorer. To improve performance and accelerate convergence, our method backpropagates the reward signal through the last steps of the sampling chain, enabling richer gradient feedback. We also propose a novel facial scoring mechanism that treats faces in ground-truth videos as facial feature pools, providing multi-angle facial information to enhance generalization. A KL-divergence regularization is further incorporated to stabilize training and prevent overfitting to the reward signal. Extensive experiments on Wan 2.2 I2V model and our in-house I2V model demonstrate the effectiveness of our method. Our project and code are available at this https URL.
>
---
#### [replaced 121] ConfIC-RCA: Statistically Grounded Efficient Estimation of Segmentation Quality
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.04522](https://arxiv.org/pdf/2503.04522)**

> **作者:** Matias Cosarinsky; Ramiro Billot; Lucas Mansilla; Gabriel Jimenez; Nicolas Gaggión; Guanghui Fu; Tom Tirer; Enzo Ferrante
>
> **备注:** Accepted for publication at TMI
>
> **摘要:** Assessing the quality of automatic image segmentation is crucial in clinical practice, but often very challenging due to the limited availability of ground truth annotations. Reverse Classification Accuracy (RCA) is an approach that estimates the quality of new predictions on unseen samples by training a segmenter on those predictions, and then evaluating it against existing annotated images. In this work we introduce ConfIC-RCA (Conformal In-Context RCA), a novel method for automatically estimating segmentation quality with statistical guarantees in the absence of ground-truth annotations, which consists of two main innovations. First, In-Context RCA, which leverages recent in-context learning models for image segmentation and incorporates retrieval-augmentation techniques to select the most relevant reference images. This approach enables efficient quality estimation with minimal reference data while avoiding the need of training additional models. Second, we introduce Conformal RCA, which extends both the original RCA framework and In-Context RCA to go beyond point estimation. Using tools from split conformal prediction, Conformal RCA produces prediction intervals for segmentation quality providing statistical guarantees that the true score lies within the estimated interval with a user-specified probability. Validated across 10 different medical imaging tasks in various organs and modalities, our methods demonstrate robust performance and computational efficiency, offering a promising solution for automated quality control in clinical workflows, where fast and reliable segmentation assessment is essential. The code is available at this https URL
>
---
#### [replaced 122] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.24862](https://arxiv.org/pdf/2505.24862)**

> **作者:** Cailin Zhuang; Ailin Huang; Yaoqi Hu; Jingwei Wu; Wei Cheng; Jiaqi Liao; Hongyuan Wang; Xinyao Liao; Weiwei Cai; Hengyuan Xu; Xuanyang Zhang; Xianfang Zeng; Zhewei Huang; Gang Yu; Chi Zhang
>
> **备注:** Accepted by CVPR 2026. 44 Pages, Project Page: this https URL, Code: this https URL, Dataset: this https URL
>
> **摘要:** Story visualization aims to generate coherent image sequences that faithfully represent a narrative and match given character references. Despite progress in generative models, existing benchmarks remain narrow in scope, often limited to short prompts, lacking character references, or single-image cases, failing to reflect real-world narrative complexity and obscuring true model this http URL introduce ViStoryBench, a comprehensive benchmark designed to evaluate story visualization models across varied narrative structures, visual styles, and character settings. It features richly annotated multi-shot scripts derived from curated stories spanning literature, film, and folklore. Large language models assist in story summarization and script generation, with all outputs verified by humans for coherence and fidelity. Character references are carefully curated to maintain consistency across different artistic styles. ViStoryBench proposes a suite of multi-dimensional automated metrics to evaluate character consistency, style similarity, prompt alignment, aesthetic quality, and artifacts like copy-paste behavior. These metrics are validated through human studies and used to assess a broad range of open-source and commercial models, enabling systematic analysis and encouraging advances in visual storytelling.
>
---
#### [replaced 123] Overcoming the Curvature Bottleneck in MeanFlow
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.23342](https://arxiv.org/pdf/2511.23342)**

> **作者:** Xinxi Zhang; Shiwei Tan; Quang Nguyen; Quan Dao; Ligong Han; Xiaoxiao He; Tunyu Zhang; Chengzhi Mao; Dimitris Metaxas; Vladimir Pavlovic
>
> **摘要:** MeanFlow offers a promising framework for one-step generative modeling by directly learning a mean-velocity field, bypassing expensive numerical integration. However, we find that the highly curved generative trajectories of existing models induce a noisy loss landscape, severely bottlenecking convergence and model quality. We leverage a fundamental geometric principle to overcome this: mean-velocity estimation is drastically simpler along straight paths. Building on this insight, we propose Rectified MeanFlow, a self-distillation approach that learns the mean-velocity field over a straightened velocity field, induced by rectified couplings from a pretrained model. To further promote linearity, we introduce a distance-based truncation heuristic that prunes residual high-curvature pairs. By smoothing the optimization landscape, our method achieves strong one-step generation performance. We improve the FID of baseline MeanFlow models from 30.9 to 8.6 under same training budget, and outperform the recent 2-rectified flow++ by 33.4% in FID while running 26x faster. Our work suggests that the difficulty of one-step flow generation stems partially from the rugged optimization landscapes induced by curved trajectories. Code is available at this https URL.
>
---
#### [replaced 124] DeH4R: A Decoupled and Hybrid Method for Road Network Graph Extraction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.13669](https://arxiv.org/pdf/2508.13669)**

> **作者:** Dengxian Gong; Shunping Ji
>
> **备注:** Accepted for publication in the IEEE Transactions on Geoscience and Remote Sensing (TGRS)
>
> **摘要:** The automated extraction of complete and precise road network graphs from remote sensing imagery remains a critical challenge in geospatial computer vision. Segmentation-based approaches, while effective in pixel-level recognition, struggle to maintain topology fidelity after vectorization postprocessing. Graph-growing methods build more topologically faithful graphs but suffer from computationally prohibitive iterative ROI cropping. Graph-generating methods first predict global static candidate road network vertices, and then infer possible edges between vertices. They achieve fast topology-aware inference, but limits the dynamic insertion of vertices. To address these challenges, we propose DeH4R, a novel hybrid model that combines graph-generating efficiency and graph-growing dynamics. This is achieved by decoupling the task into candidate vertex detection, adjacent vertex prediction, initial graph contruction, and graph expansion. This architectural innovation enables dynamic vertex (edge) insertions while retaining fast inference speed and enhancing both topology fidelity and spatial consistency. Comprehensive evaluations on CityScale and SpaceNet benchmarks demonstrate state-of-the-art (SOTA) performance. DeH4R outperforms the prior SOTA graph-growing method RNGDet++ by 4.62 APLS and 10.18 IoU on CityScale, while being approximately 10 $\times$ faster. The code will be made publicly available at this https URL.
>
---
#### [replaced 125] SUG-Occ: Explicit Semantics and Uncertainty Guided Sparse Learning for Efficient 3D Occupancy Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11396](https://arxiv.org/pdf/2601.11396)**

> **作者:** Hanlin Wu; Pengfei Lin; Ehsan Javanmardi; Naren Bao; Bo Qian; Hao Si; Manabu Tsukada
>
> **摘要:** 3D semantic occupancy prediction has emerged as a critical perception task for autonomous driving due to its ability to offer voxel-level semantic and geometric understanding of the environment. However, such a refined representation for large-scale scenes incurs prohibitive computation, posing a significant challenge to practical real-time deployment. To address this, we propose SUGOcc, an explicit semantics and uncertainty guided sparse learning framework for efficient occupancy prediction, which exploits the inherent sparsity of 3D scenes to reduce redundant computation while maintaining geometric and semantic integrity. Specifically, we first utilize semantic and uncertainty priors to suppress image projections from free space while employing explicit unsigned distance encoding to enhance geometric consistency, thereby producing a structurally sparse representation. Secondly, we introduce a cascade sparse completion module to enable efficient coarse-to-fine reasoning over the sparse representation via hyper cross sparse convolution, generative upsampling and adaptive pruning. Finally, we propose an object contextual representation (OCR) based mask decoder that refines the voxel-wise predictions through lightweight query-context interactions, thereby avoiding expensive attention operations over volumetric features. Extensive experiments on SemanticKITTI and Occ3D-Nuscenes benchmark demonstrate that the proposed approach outperforms the baselines, achieving notable improvements in both accuracy and efficiency across datasets.
>
---
#### [replaced 126] PointCNN++: Performant Convolution on Native Points
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.23227](https://arxiv.org/pdf/2511.23227)**

> **作者:** Lihan Li; Haofeng Zhong; Rui Bu; Mingchao Sun; Wenzheng Chen; Baoquan Chen; Yangyan Li
>
> **摘要:** Existing convolutional learning methods for 3D point cloud data are divided into two paradigms: point-based methods that preserve geometric precision but often face performance challenges, and voxel-based methods that achieve high efficiency through quantization at the cost of geometric fidelity. This loss of precision is a critical bottleneck for tasks such as point cloud registration. We propose PointCNN++, a novel architectural design that fundamentally mitigates this precision-performance trade-off. It $\textbf{generalizes sparse convolution from voxels to points}$, treating voxel-based convolution as a specialized, degraded case of our more general point-based convolution. First, we introduce a point-centric convolution where the receptive field is centered on the original, high-precision point coordinates. Second, to make this high-fidelity operation performant, we design a computational strategy that operates $\textbf{natively}$ on points. We formulate the convolution on native points as a Matrix-Vector Multiplication and Reduction (MVMR) problem, for which we develop a dedicated, highly-optimized GPU kernel. Experiments demonstrate that PointCNN++ $\textbf{uses an order of magnitude less memory and is several times faster}$ than representative point-based methods. Furthermore, when used as a simple replacement for the voxel-based backbones it generalizes, it $\textbf{significantly improves point cloud registration accuracies while proving both more memory-efficient and faster}$. PointCNN++ shows that preserving geometric detail and achieving high performance are not mutually exclusive, paving the way for a new class of 3D learning with high fidelity and efficiency. Our code will be open sourced.
>
---
#### [replaced 127] AG-VAS: Anchor-Guided Zero-Shot Visual Anomaly Segmentation with Large Multimodal Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.01305](https://arxiv.org/pdf/2603.01305)**

> **作者:** Zhen Qu; Xian Tao; Xiaoyi Bao; Dingrong Wang; ShiChen Qu; Zhengtao Zhang; Xingang Wang
>
> **摘要:** Large multimodal models (LMMs) exhibit strong task generalization capabilities, offering new opportunities for zero-shot visual anomaly segmentation (ZSAS). However, existing LMM-based segmentation approaches still face fundamental limitations: anomaly concepts are inherently abstract and context-dependent, lacking stable visual prototypes, and the weak alignment between high-level semantic embeddings and pixel-level spatial features hinders precise anomaly localization. To address these challenges, we present AG-VAS (Anchor-Guided Visual Anomaly Segmentation), a new framework that expands the LMM vocabulary with three learnable semantic anchor tokens-[SEG], [NOR], and [ANO], establishing a unified anchor-guided segmentation paradigm. Specifically, [SEG] serves as an absolute semantic anchor that translates abstract anomaly semantics into explicit, spatially grounded visual entities (e.g., holes or scratches), while [NOR] and [ANO] act as relative anchors that model the contextual contrast between normal and abnormal patterns across categories. To further enhance cross-modal alignment, we introduce a Semantic-Pixel Alignment Module (SPAM) that aligns language-level semantic embeddings with high-resolution visual features, along with an Anchor-Guided Mask Decoder (AGMD) that performs anchor-conditioned mask prediction for precise anomaly localization. In addition, we curate Anomaly-Instruct20K, a large-scale instruction dataset that organizes anomaly knowledge into structured descriptions of appearance, shape, and spatial attributes, facilitating effective learning and integration of the proposed semantic anchors. Extensive experiments on six industrial and medical benchmarks demonstrate that AG-VAS achieves consistent state-of-the-art performance in the zero-shot setting.
>
---
#### [replaced 128] NewtonGen: Physics-Consistent and Controllable Text-to-Video Generation via Neural Newtonian Dynamics
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.21309](https://arxiv.org/pdf/2509.21309)**

> **作者:** Yu Yuan; Xijun Wang; Tharindu Wickremasinghe; Zeeshan Nadir; Bole Ma; Stanley H. Chan
>
> **备注:** Accepted by ICLR 2026. Camera-ready version. Project Page: this https URL
>
> **摘要:** A primary bottleneck in large-scale text-to-video generation today is physical consistency and controllability. Despite recent advances, state-of-the-art models often produce unrealistic motions, such as objects falling upward, or abrupt changes in velocity and direction. Moreover, these models lack precise parameter control, struggling to generate physically consistent dynamics under different initial conditions. We argue that this fundamental limitation stems from current models learning motion distributions solely from appearance, while lacking an understanding of the underlying dynamics. In this work, we propose NewtonGen, a framework that integrates data-driven synthesis with learnable physical principles. At its core lies trainable Neural Newtonian Dynamics (NND), which can model and predict a variety of Newtonian motions, thereby injecting latent dynamical constraints into the video generation process. By jointly leveraging data priors and dynamical guidance, NewtonGen enables physically consistent video synthesis with precise parameter control. All data and code are available at this https URL
>
---
#### [replaced 129] Target-aware Image Editing via Cycle-consistent Constraints
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.20212](https://arxiv.org/pdf/2510.20212)**

> **作者:** Yanghao Wang; Zhen Wang; Long Chen
>
> **摘要:** Recent pre-trained text-to-image flow models have enabled remarkable progress in text-based image editing. Mainstream approaches adopt a corruption-then-restoration paradigm, where the source image is first corrupted into an editable ``intermediate state'' and then restored to the target image under the prompt guidance. However, current methods construct this intermediate state in a target-agnostic manner, i.e., they mainly focus on realizing source image reconstruction while neglecting the semantic gaps towards the specific editing target. This design inherently results in limited editability or inconsistency when the desired modifications substantially deviate from the source. In this paper, we argue that the intermediate state should be target-aware, i.e., selectively corrupting editing-relevant contents while preserving editing-irrelevant ones. Thus, we propose FlowCycle, an inversion-free and flow-based editing framework that parameterizes corruption with learnable noises and optimizes them through a cycle-consistent process. By iteratively editing the source to the target and recovering back to the source with dual consistency constraints, FlowCycle learns to produce a target-aware intermediate state, enabling faithful modifications while preserving source consistency. For efficiency, we further accelerate the optimization by dynamically adjusting the sampling steps. Extensive ablations demonstrated that FlowCycle achieves superior editing performance.
>
---
#### [replaced 130] RefTon: Reference person shot assist virtual Try-on
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00956](https://arxiv.org/pdf/2511.00956)**

> **作者:** Liuzhuozheng Li; Yue Gong; Shanyuan Liu; Bo Cheng; Yuhang Ma; Leibucha Wu; Dengyang Jiang; Zanyi Wang; Dawei Leng; Yuhui Yin
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** We introduce RefTon, a flux-based person-to-person virtual try-on framework that enhances garment realism through unpaired visual references. Unlike conventional approaches that rely on complex auxiliary inputs such as body parsing and warped mask or require finely designed extract branches to process various input conditions, RefTon streamlines the process by directly generating try-on results from a source image and a target garment, without the need for structural guidance or auxiliary components to handle diverse inputs. Moreover, inspired by human clothing selection behavior, RefTon leverages additional reference images (the target garment worn on different individuals) to provide powerful guidance for refining texture alignment and maintaining the garment details. To enable this capability, we built a dataset containing unpaired reference images for training. Extensive experiments on public benchmarks demonstrate that RefTon achieves competitive or superior performance compared to state-of-the-art methods, while maintaining a simple and efficient person-to-person design.
>
---
#### [replaced 131] Echoes of ownership: Adversarial-guided dual injection for copyright protection in MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18845](https://arxiv.org/pdf/2602.18845)**

> **作者:** Chengwei Xia; Fan Ma; Ruijie Quan; Yunqiu Xu; Kun Zhan; Yi Yang
>
> **备注:** Accepted to CVPR 2026!
>
> **摘要:** With the rapid deployment of multimodal large language models (MLLMs), disputes regarding model ownership have become increasingly frequent, raising significant concerns about intellectual property protection. In this paper, we propose a framework for generating copyright triggers for MLLMs, enabling model publishers to embed verifiable ownership information into the model. The goal is to construct trigger images that elicit ownership-related textual responses exclusively in fine-tuned derivatives, while remaining inert in other non-derivative models. Our method constructs a tracking trigger image by treating the image as a learnable tensor, performing adversarial optimization with dual-injection of ownership-relevant semantic information. The first injection is achieved by enforcing textual consistency between the output of an auxiliary MLLM and a predefined ownership-relevant target text; the consistency loss is backpropagated to inject this ownership-related information into the image. The second injection is performed at the semantic-level by minimizing the distance between the CLIP features of the image and those of the target text. Furthermore, we introduce an additional adversarial training stage involving the auxiliary model. It is specifically trained to resist generating ownership-relevant target text, thereby enhancing robustness in heavily fine-tuned derivative models. Extensive experiments demonstrate the effectiveness of our dual-injection approach in tracking model lineage under various fine-tuning and domain-shift scenarios. Code is at this https URL
>
---
#### [replaced 132] Self-Attention And Beyond the Infinite: Towards Linear Transformers with Infinite Self-Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00175](https://arxiv.org/pdf/2603.00175)**

> **作者:** Giorgio Roffo; Hazem Abdelkawy; Nilli Lavie; Luke Palmer
>
> **备注:** This work was initiated and primarily carried out while working at MindVisionLabs. We gratefully acknowledge the support of Toyota Motor Europe (TME) and Equixly API Security for this work
>
> **摘要:** The quadratic cost of softmax attention limits Transformer scalability in high-resolution vision. We introduce Infinite Self-Attention (InfSA), a spectral reformulation that treats each attention layer as a diffusion step on a content-adaptive token graph, accumulating multi-hop interactions through a discounted Neumann series over attention matrices. This links self-attention to classical graph centrality (Katz, PageRank, eigenvector centrality) for interpretable token weighting. We also show the Neumann kernel equals the fundamental matrix of an absorbing Markov chain, so a token's centrality is its expected number of random-walk visits before absorption. We then propose Linear-InfSA, a linear-time variant that approximates the principal eigenvector of the implicit attention operator without forming the full attention matrix. It keeps an auxiliary state of fixed size proportional to per-head dimension dh (independent of sequence length N), is drop-in compatible with Vision Transformers, and supports stable training at 4096 by 4096 and inference at 9216 by 9216 (about 332k tokens). In a 4-layer ViT (53.5M parameters, 59 GFLOPs at 224 by 224), Linear-InfSA reaches 84.7% top-1 on ImageNet-1K, a +3.2 point architectural gain over an equal-depth softmax ViT trained with the same recipe. On ImageNet-V2, InfViT variants outperform all compared baselines (up to 79.8% vs 76.8%), indicating robustness under distribution shift. On an A100 40GB GPU, Linear-InfViT runs at 231 images/s and 0.87 J/image (13x better throughput and energy than equal-depth ViT) and is the only tested model to complete 9216 by 9216 inference without out-of-memory. The linear approximation closely matches the dominant eigenvector of the quadratic operator (cosine 0.985).
>
---
#### [replaced 133] GenHOI: Generalized Hand-Object Pose Estimation with Occlusion Awareness
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.19013](https://arxiv.org/pdf/2603.19013)**

> **作者:** Hui Yang; Wei Sun; Jian Liu; Jian Xiao; Tao Xie; Hossein Rahmani; Ajmal Saeed Mian; Nicu Sebe; Gim Hee Lee
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** Generalized 3D hand-object pose estimation from a single RGB image remains challenging due to the large variations in object appearances and interaction patterns, especially under heavy occlusion. We propose GenHOI, a framework for generalized hand-object pose estimation with occlusion awareness. GenHOI integrates hierarchical semantic knowledge with hand priors to enhance model generalization under challenging occlusion conditions. Specifically, we introduce a hierarchical semantic prompt that encodes object states, hand configurations, and interaction patterns via textual descriptions. This enables the model to learn abstract high-level representations of hand-object interactions for generalization to unseen objects and novel interactions while compensating for missing or ambiguous visual cues. To enable robust occlusion reasoning, we adopt a multi-modal masked modeling strategy over RGB images, predicted point clouds, and textual descriptions. Moreover, we leverage hand priors as stable spatial references to extract implicit interaction constraints. This allows reliable pose inference even under significant variations in object shapes and interaction patterns. Extensive experiments on the challenging DexYCB and HO3Dv2 benchmarks demonstrate that our method achieves state-of-the-art performance in hand-object pose estimation.
>
---
#### [replaced 134] The Prism Hypothesis: Harmonizing Semantic and Pixel Representations via Unified Autoencoding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19693](https://arxiv.org/pdf/2512.19693)**

> **作者:** Weichen Fan; Haiwen Diao; Quan Wang; Dahua Lin; Ziwei Liu
>
> **备注:** Code link: this https URL
>
> **摘要:** Deep representations across modalities are inherently intertwined. In this paper, we systematically analyze the spectral characteristics of various semantic and pixel encoders. Interestingly, our study uncovers a highly inspiring and rarely explored correspondence between an encoder's feature spectrum and its functional role: semantic encoders primarily capture low-frequency components that encode abstract meaning, whereas pixel encoders additionally retain high-frequency information that conveys fine-grained detail. This heuristic finding offers a unifying perspective that ties encoder behavior to its underlying spectral structure. We define it as the Prism Hypothesis, where each data modality can be viewed as a projection of the natural world onto a shared feature spectrum, just like the prism. Building on this insight, we propose Unified Autoencoding (UAE), a model that harmonizes semantic structure and pixel details via an innovative frequency-band modulator, enabling their seamless coexistence. Extensive experiments demonstrate that UAE effectively unifies semantic abstraction and pixel-level fidelity within a single latent space, achieving state-of-the-art performance. Moreover, we show that UAE can be directly applied to pixel-space modeling, significantly improving both FID and IS over the vanilla JIT baseline. Our code is avaliable at: this https URL.
>
---
#### [replaced 135] Towards Holistic Modeling for Video Frame Interpolation with Auto-regressive Diffusion Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.14959](https://arxiv.org/pdf/2601.14959)**

> **作者:** Xinyu Peng; Han Li; Yuyang Huang; Ziyang Zheng; Yaoming Wang; Xin Chen; Wenrui Dai; Chenglin Li; Junni Zou; Hongkai Xiong
>
> **摘要:** Existing video frame interpolation (VFI) methods often adopt a frame-centric approach, processing videos as independent short segments (e.g., triplets), which leads to temporal inconsistencies and motion artifacts. To overcome this, we propose a holistic, video-centric paradigm named Local Diffusion Forcing for Video Frame Interpolation (LDF-VFI). Our framework is built upon an auto-regressive diffusion transformer that models the entire video sequence to ensure long-range temporal coherence. To mitigate error accumulation inherent in auto-regressive generation, we introduce a novel skip-concatenate sampling strategy that effectively maintains temporal stability. Furthermore, LDF-VFI incorporates sparse, local attention and tiled VAE encoding, a combination that not only enables efficient processing of long sequences but also allows generalization to arbitrary spatial resolutions (e.g., 4K) at inference without retraining. An enhanced conditional VAE decoder, which leverages multi-scale features from the input video, further improves reconstruction fidelity. Empirically, LDF-VFI achieves state-of-the-art performance on challenging VFI benchmarks, demonstrating superior per-frame quality and temporal consistency, especially in scenes with large motion. The source code is available at this https URL.
>
---
#### [replaced 136] Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.25716](https://arxiv.org/pdf/2603.25716)**

> **作者:** Kaijin Chen; Dingkang Liang; Xin Zhou; Yikang Ding; Xiaoqiang Liu; Pengfei Wan; Xiang Bai
>
> **备注:** Project Page: this https URL Code: this https URL
>
> **摘要:** Video world models have shown immense potential in simulating the physical world, yet existing memory mechanisms primarily treat environments as static canvases. When dynamic subjects hide out of sight and later re-emerge, current methods often struggle, leading to frozen, distorted, or vanishing subjects. To address this, we introduce Hybrid Memory, a novel paradigm requiring models to simultaneously act as precise archivists for static backgrounds and vigilant trackers for dynamic subjects, ensuring motion continuity during out-of-view intervals. To facilitate research in this direction, we construct HM-World, the first large-scale video dataset dedicated to hybrid memory. It features 59K high-fidelity clips with decoupled camera and subject trajectories, encompassing 17 diverse scenes, 49 distinct subjects, and meticulously designed exit-entry events to rigorously evaluate hybrid coherence. Furthermore, we propose HyDRA, a specialized memory architecture that compresses memory into tokens and utilizes a spatiotemporal relevance-driven retrieval mechanism. By selectively attending to relevant motion cues, HyDRA effectively preserves the identity and motion of hidden subjects. Extensive experiments on HM-World demonstrate that our method significantly outperforms state-of-the-art approaches in both dynamic subject consistency and overall generation quality. Code is publicly available at this https URL.
>
---
#### [replaced 137] BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.10932](https://arxiv.org/pdf/2512.10932)**

> **作者:** Shengao Wang; Wenqi Wang; Zecheng Wang; Max Whitton; Michael Wakeham; Arjun Chandra; Joey Huang; Pengyue Zhu; Helen Chen; David Li; Jeffrey Li; Shawn Li; Andrew Zagula; Amy Zhao; Andrew Zhu; Sayaka Nakamura; Yuki Yamamoto; Jerry Jun Yokono; Aaron Mueller; Bryan A. Plummer; Kate Saenko; Venkatesh Saligrama; Boqing Gong
>
> **备注:** Accepted to CVPR 2026 main track
>
> **摘要:** Early children's developmental trajectories set up a natural goal for sample-efficient pretraining of vision foundation models. We introduce BabyVLM-V2, a developmentally grounded framework for infant-inspired vision-language modeling that extensively improves upon BabyVLM-V1 through a longitudinal, multifaceted pretraining set, a versatile model, and, most importantly, DevCV Toolbox for cognitive evaluation. The pretraining set maximizes coverage while minimizing curation of a longitudinal, infant-centric audiovisual corpus, yielding video-utterance, image-utterance, and multi-turn conversational data that mirror infant experiences. DevCV Toolbox adapts all vision-related measures of the recently released NIH Baby Toolbox into a benchmark suite of ten multimodal tasks, covering spatial reasoning, memory, and vocabulary understanding aligned with early children's capabilities. Experimental results show that a compact model pretrained from scratch can achieve competitive performance on DevCV Toolbox, outperforming GPT-4o on some tasks. We hope the principled, unified BabyVLM-V2 framework will accelerate research in developmentally plausible pretraining of vision foundation models.
>
---
#### [replaced 138] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24984](https://arxiv.org/pdf/2603.24984)**

> **作者:** Dohwan Ko; Jinyoung Park; Seoung Choi; Sanghyeok Lee; Seohyun Lee; Hyunwoo J. Kim
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Mixture-of-Experts (MoE) has emerged as an effective approach to reduce the computational overhead of Transformer architectures by sparsely activating a subset of parameters for each token while preserving high model capacity. This paradigm has recently been extended to Vision-Language Models (VLMs), enabling scalable multi-modal understanding with reduced computational cost. However, the widely adopted deterministic top-K routing mechanism may overlook more optimal expert combinations and lead to expert overfitting. To address this limitation and improve the diversity of expert selection, we propose MoE-GRPO, a reinforcement learning (RL)-based framework for optimizing expert routing in MoE-based VLMs. Specifically, we formulate expert selection as a sequential decision-making problem and optimize it using Group Relative Policy Optimization (GRPO), allowing the model to learn adaptive expert routing policies through exploration and reward-based feedback. Furthermore, we introduce a modality-aware router guidance that enhances training stability and efficiency by discouraging the router from exploring experts that are infrequently activated for a given modality. Extensive experiments on multi-modal image and video benchmarks show that MoE-GRPO consistently outperforms standard top-K routing and its variants by promoting more diverse expert selection, thereby mitigating expert overfitting and enabling a task-level expert specialization.
>
---
#### [replaced 139] Seeing Isn't Orienting: A Cognitively Grounded Benchmark Reveals Systematic Orientation Failures in MLLMs Supplementary
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.11410](https://arxiv.org/pdf/2603.11410)**

> **作者:** Nazia Tasnim; Keanu Nichols; Yuting Yang; Nicholas Ikechukwu; Elva Zou; Deepti Ghadiyaram; Bryan A. Plummer
>
> **备注:** This is a replacement and updated version for submission arXiv:2505.21649 : Right Side Up? Disentangling Orientation Understanding in MLLMs with Fine-grained Multi-axis Perception Tasks
>
> **摘要:** Humans learn object orientation progressively, from recognizing which way an object faces, to mentally rotating it, to reasoning about orientations between objects. Current vision-language benchmarks largely conflate orientation with position and general scene understanding. We introduce Discriminative Orientation Reasoning Intelligence (DORI), a cognitively grounded hierarchical benchmark that makes object orientation the primary target. Inspired by stages of human orientation cognition, DORI decomposes orientation into four dimensions, each evaluated at coarse (categorical) and granular (metric) levels. Composed from 13,652 images across 14 sources, DORI provides 33,656 multiple-choice questions covering 67 object categories in real-world and synthetic settings. Its coarse-to-granular design isolates orientation from confounds such as object recognition difficulty, scene clutter, and linguistic ambiguity via bounding-box isolation, standardized spatial reference frames, and structured prompts. Evaluating 24 state-of-the-art vision-language models shows a clear pattern: models that perform well on general spatial benchmarks are near-random on object-centric orientation tasks. The best models reach only 54.2% on coarse and 45.0% on granular judgments, with largest failures on compound rotations and shifts in inter-object reference frames. Large coarse-to-granular gaps reveal reliance on categorical heuristics rather than geometric reasoning, a limitation hidden by existing benchmarks. These results identify orientation understanding as an unsolved challenge for multimodal systems, with implications for robotic manipulation, 3D scene reconstruction, and human-AI interaction.
>
---
#### [replaced 140] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely $F_1$
- **分类: cs.PF; cs.AI; cs.CV; cs.LG; stat.ML**

- **链接: [https://arxiv.org/pdf/2511.22442](https://arxiv.org/pdf/2511.22442)**

> **作者:** Sébastien Piérard; Adrien Deliège; Marc Van Droogenbroeck
>
> **备注:** CVPR 2026
>
> **摘要:** Ranking methods or models based on their performance is of prime importance but is tricky because performance is fundamentally multidimensional. In the case of classification, precision and recall are scores with probabilistic interpretations that are both important to consider and complementary. The rankings induced by these two scores are often in partial contradiction. In practice, therefore, it is extremely useful to establish a compromise between the two views to obtain a single, global ranking. Over the last fifty years or so, it has been proposed to take a weighted harmonic mean, known as the F-score, F-measure, or $F_\beta$. Generally speaking, by averaging basic scores, we obtain a score that is intermediate in terms of values. However, there is no guarantee that these scores lead to meaningful rankings and no guarantee that the rankings are good tradeoffs between these base scores. Given the ubiquity of $F_\beta$ scores in the literature, some clarification is in order. Concretely: (1) We establish that $F_\beta$-induced rankings are meaningful and define a shortest path between precision- and recall-induced rankings. (2) We frame the problem of finding a tradeoff between two scores as an optimization problem expressed with Kendall rank correlations. We show that $F_1$ and its skew-insensitive version are far from being optimal in that regard. (3) We provide theoretical tools and a closed-form expression to find the optimal value for $\beta$ for any distribution or set of performances, and we illustrate their use on six case studies. Code is available at this https URL.
>
---
#### [replaced 141] Mixture of Style Experts for Diverse Image Stylization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16649](https://arxiv.org/pdf/2603.16649)**

> **作者:** Shihao Zhu; Ziheng Ouyang; Yijia Kang; Qilong Wang; Mi Zhou; Bo Li; Ming-Ming Cheng; Qibin Hou
>
> **备注:** 24 pages, 16 figures
>
> **摘要:** Diffusion-based stylization has advanced significantly, yet existing methods are limited to color-driven transformations, neglecting complex semantics and material details. We introduce StyleExpert, a semantic-aware framework based on the Mixture of Experts (MoE). Our framework employs a unified style encoder, trained on our large-scale dataset of content-style-stylized triplets, to embed diverse styles into a consistent latent space. This embedding is then used to condition a similarity-aware gating mechanism, which dynamically routes styles to specialized experts within the MoE architecture. Leveraging this MoE architecture, our method adeptly handles diverse styles spanning multiple semantic levels, from shallow textures to deep semantics. Extensive experiments show that StyleExpert outperforms existing approaches in preserving semantics and material details, while generalizing to unseen styles. Our code and collected images are available at the project page: this https URL.
>
---
#### [replaced 142] POLY-SIM: Polyglot Speaker Identification with Missing Modality Grand Challenge 2026 Evaluation Plan
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24569](https://arxiv.org/pdf/2603.24569)**

> **作者:** Marta Moscati; Muhammad Saad Saeed; Marina Zanoni; Mubashir Noman; Rohan Kumar Das; Monorama Swain; Yufang Hou; Elisabeth Andre; Khalid Mahmood Malik; Markus Schedl; Shah Nawaz
>
> **备注:** Grand challenge at ACM MM 2026
>
> **摘要:** Multimodal speaker identification systems typically assume the availability of complete and homogeneous audio-visual modalities during both training and testing. However, in real-world applications, such assumptions often do not hold. Visual information may be missing due to occlusions, camera failures, or privacy constraints, while multilingual speakers introduce additional complexity due to linguistic variability across languages. These challenges significantly affect the robustness and generalization of multimodal speaker identification systems. The POLY-SIM Grand Challenge 2026 aims to advance research in multimodal speaker identification under missing-modality and cross-lingual conditions. Specifically, the Grand Challenge encourages the development of robust methods that can effectively leverage incomplete multimodal inputs while maintaining strong performance across different languages. This report presents the design and organization of the POLY-SIM Grand Challenge 2026, including the dataset, task formulation, evaluation protocol, and baseline model. By providing a standardized benchmark and evaluation framework, the challenge aims to foster progress toward more robust and practical multimodal speaker identification systems.
>
---
#### [replaced 143] Securing the Skies: A Comprehensive Survey on Anti-UAV Methods, Benchmarking, and Future Directions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 本文属于反无人机任务，旨在解决UAV的安全威胁问题。综述了分类、检测与跟踪方法，分析了现有技术及挑战，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2504.11967](https://arxiv.org/pdf/2504.11967)**

> **作者:** Yifei Dong; Fengyi Wu; Sanjian Zhang; Guangyu Chen; Yuzhi Hu; Masumi Yano; Jingdong Sun; Siyu Huang; Feng Liu; Qi Dai; Zhi-Qi Cheng
>
> **备注:** Accepted to CVPR 2025 Anti-UAV Workshop (Best Paper Award), 16 pages
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are indispensable for infrastructure inspection, surveillance, and related tasks, yet they also introduce critical security challenges. This survey provides a wide-ranging examination of the anti-UAV domain, centering on three core objectives-classification, detection, and tracking-while detailing emerging methodologies such as diffusion-based data synthesis, multi-modal fusion, vision-language modeling, self-supervised learning, and reinforcement learning. We systematically evaluate state-of-the-art solutions across both single-modality and multi-sensor pipelines (spanning RGB, infrared, audio, radar, and RF) and discuss large-scale as well as adversarially oriented benchmarks. Our analysis reveals persistent gaps in real-time performance, stealth detection, and swarm-based scenarios, underscoring pressing needs for robust, adaptive anti-UAV systems. By highlighting open research directions, we aim to foster innovation and guide the development of next-generation defense strategies in an era marked by the extensive use of UAVs.
>
---
#### [replaced 144] SASNet: Spatially-Adaptive Sinusoidal Networks for INRs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.09750](https://arxiv.org/pdf/2503.09750)**

> **作者:** Haoan Feng; Diana Aldana; Tiago Novello; Leila De Floriani
>
> **备注:** CVPR2026, 10 pages, 10 figures, suppl included
>
> **摘要:** Sinusoidal neural networks (SIRENs) are powerful implicit neural representations (INRs) for low-dimensional signals in vision and graphics. By encoding input coordinates with sinusoidal functions, they enable high-frequency image and surface reconstruction. However, training SIRENs is often unstable and highly sensitive to frequency initialization: small frequencies produce overly smooth reconstructions in detailed regions, whereas large ones introduce spurious high-frequency components that manifest as noise in smooth areas such as image backgrounds. To address these challenges, we propose SASNet, a Spatially-Adaptive Sinusoidal Network that couples a frozen frequency embedding layer, which explicitly fixes the network's frequency support, with jointly learned spatial masks that localize neuron influence across the domain. This pairing stabilizes optimization, sharpens edges, and suppresses noise in smooth areas. Experiments on 2D image and 3D volumetric data fitting as well as signed distance field (SDF) reconstruction benchmarks demonstrate that SASNet achieves faster convergence, superior reconstruction quality, and robust frequency localization -- assigning low- and high-frequency neurons to smooth and detailed regions respectively -- while maintaining parameter efficiency. Code available here: this https URL.
>
---
#### [replaced 145] SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出SimULi，解决多传感器实时模拟问题，支持任意相机模型和LiDAR数据，提升模拟速度与精度。**

- **链接: [https://arxiv.org/pdf/2510.12901](https://arxiv.org/pdf/2510.12901)**

> **作者:** Haithem Turki; Qi Wu; Xin Kang; Janick Martinez Esturo; Shengyu Huang; Ruilong Li; Zan Gojcic; Riccardo de Lutio
>
> **备注:** ICLR 2026 - project page: this https URL
>
> **摘要:** Rigorous testing of autonomous robots, such as self-driving vehicles, is essential to ensure their safety in real-world deployments. This requires building high-fidelity simulators to test scenarios beyond those that can be safely or exhaustively collected in the real-world. Existing neural rendering methods based on NeRF and 3DGS hold promise but suffer from low rendering speeds or can only render pinhole camera models, hindering their suitability to applications that commonly require high-distortion lenses and LiDAR data. Multi-sensor simulation poses additional challenges as existing methods handle cross-sensor inconsistencies by favoring the quality of one modality at the expense of others. To overcome these limitations, we propose SimULi, the first method capable of rendering arbitrary camera models and LiDAR data in real-time. Our method extends 3DGUT, which natively supports complex camera models, with LiDAR support, via an automated tiling strategy for arbitrary spinning LiDAR models and ray-based culling. To address cross-sensor inconsistencies, we design a factorized 3D Gaussian representation and anchoring strategy that reduces mean camera and depth error by up to 40% compared to existing methods. SimULi renders 10-20x faster than ray tracing approaches and 1.5-10x faster than prior rasterization-based work (and handles a wider range of camera models). When evaluated on two widely benchmarked autonomous driving datasets, SimULi matches or exceeds the fidelity of existing state-of-the-art methods across numerous camera and LiDAR metrics.
>
---
#### [replaced 146] Toward Phonology-Guided Sign Language Motion Generation: A Diffusion Baseline and Conditioning Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.17388](https://arxiv.org/pdf/2603.17388)**

> **作者:** Rui Hong; Jana Kosecka
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Generating natural, correct, and visually smooth 3D avatar sign language motion conditioned on the text inputs continues to be very challenging. In this work, we train a generative model of 3D body motion and explore the role of phonological attribute conditioning for sign language motion generation, using ASL-LEX 2.0 annotations such as hand shape, hand location and movement. We first establish a strong diffusion baseline using an Human Motion MDM-style diffusion model with SMPL-X representation, which outperforms SignAvatar, a state-of-the-art CVAE method, on gloss discriminability metrics. We then systematically study the role of text conditioning using different text encoders (CLIP vs. T5), conditioning modes (gloss-only vs. gloss+phonological attributes), and attribute notation format (symbolic vs. natural language). Our analysis reveals that translating symbolic ASL-LEX notations to natural language is a necessary condition for effective CLIP-based attribute conditioning, while T5 is largely unaffected by this translation. Furthermore, our best-performing variant (CLIP with mapped attributes) outperforms SignAvatar across all metrics. These findings highlight input representation as a critical factor for text-encoder-based attribute conditioning, and motivate structured conditioning approaches where gloss and phonological attributes are encoded through independent pathways.
>
---
#### [replaced 147] SpeeDe3DGS: Speedy Deformable 3D Gaussian Splatting with Temporal Pruning and Motion Grouping
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.07917](https://arxiv.org/pdf/2506.07917)**

> **作者:** Allen Tu; Haiyang Ying; Alex Hanson; Yonghan Lee; Tom Goldstein; Matthias Zwicker
>
> **备注:** Project Page: this https URL
>
> **摘要:** Dynamic extensions of 3D Gaussian Splatting (3DGS) achieve high-quality reconstructions through neural motion fields, but per-Gaussian neural inference makes these models computationally expensive. Building on DeformableGS, we introduce Speedy Deformable 3D Gaussian Splatting (SpeeDe3DGS), which bridges this efficiency-fidelity gap through three complementary modules: Temporal Sensitivity Pruning (TSP) removes low-impact Gaussians via temporally aggregated sensitivity analysis, Temporal Sensitivity Sampling (TSS) perturbs timestamps to suppress floaters and improve temporal coherence, and GroupFlow distills the learned deformation field into shared SE(3) transformations for efficient groupwise motion. On the 50 dynamic scenes in MonoDyGauBench, integrating TSP and TSS into DeformableGS accelerates rendering by 6.78$\times$ on average while maintaining neural-field fidelity and using 10$\times$ fewer primitives. Adding GroupFlow culminates in 13.71$\times$ faster rendering and 2.53$\times$ shorter training, surpassing all baselines in speed while preserving superior image quality.
>
---
#### [replaced 148] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.25706](https://arxiv.org/pdf/2603.25706)**

> **作者:** Jinbo Xing; Zeyinzi Jiang; Yuxiang Tuo; Chaojie Mao; Xiaotang Gai; Xi Chen; Jingfeng Zhang; Yulin Pan; Zhen Han; Jie Xiao; Keyu Yan; Chenwei Xie; Chongyang Zhong; Kai Zhu; Tong Shen; Lianghua Huang; Yu Liu; Yujiu Yang
>
> **备注:** CVPR 2026 Camera-ready, Webpage: this https URL
>
> **摘要:** Recent unified models have made unprecedented progress in both understanding and generation. However, while most of them accept multi-modal inputs, they typically produce only single-modality outputs. This challenge of producing interleaved content is mainly due to training data scarcity and the difficulty of modeling long-range cross-modal context. To address this issue, we decompose interleaved generation into textual planning and visual consistency modeling, and introduce a framework consisting of a planner and a visualizer. The planner produces dense textual descriptions for visual content, while the visualizer synthesizes images accordingly. Under this guidance, we construct large-scale textual-proxy interleaved data (where visual content is represented in text) to train the planner, and curate reference-guided image data to train the visualizer. These designs give rise to Wan-Weaver, which exhibits emergent interleaved generation ability with long-range textual coherence and visual consistency. Meanwhile, the integration of diverse understanding and generation data into planner training enables Wan-Weaver to achieve robust task reasoning and generation proficiency. To assess the model's capability in interleaved generation, we further construct a benchmark that spans a wide range of use cases across multiple dimensions. Extensive experiments demonstrate that, even without access to any real interleaved data, Wan-Weaver achieves superior performance over existing methods.
>
---
#### [replaced 149] Thermal Topology Collapse: Universal Physical Patch Attacks on Infrared Vision Systems
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21876](https://arxiv.org/pdf/2603.21876)**

> **作者:** Chengyin Hu; Yikun Guo; Yuxian Dong; Qike Zhang; Kalibinuer Tiliwalidi; Yiwei Wei; Haitao Shi; Jiujiang Guo; Jiahuan Long; Xiang Chen
>
> **摘要:** Although infrared pedestrian detectors have been widely deployed in visual perception tasks, their vulnerability to physical adversarial attacks is becoming increasingly apparent. Existing physical attack methods predominantly rely on instance-specific online optimization and rigid pattern design, leading to high deployment costs and insufficient physical robustness. To address these limitations, this work proposes the Universal Physical Patch Attack (UPPA), the first universal physical attack method in the infrared domain. This method employs geometrically constrained parameterized Bezier blocks to model perturbations and utilizes the Particle Swarm Optimization (PSO) algorithm to perform unified optimization across the global data distribution, thus maintaining topological stability under dynamic deformations. In the physical deployment phase, we materialize the optimized digital perturbations into physical cold patches, achieving a continuous and smooth low-temperature distribution that naturally aligns with the thermal radiation characteristics of infrared imaging. Extensive experiments demonstrate that UPPA achieves an outstanding physical attack success rate without any online computational overhead, while also exhibiting strong cross-domain generalization and reliable black-box transferability.
>
---
#### [replaced 150] Adaptive Anchor Policies for Efficient 4D Gaussian Streaming
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.17227](https://arxiv.org/pdf/2603.17227)**

> **作者:** Ashim Dahal; Rabab Abdelfattah; Nick Rahimi
>
> **摘要:** Dynamic scene reconstruction with Gaussian Splatting has enabled efficient streaming for real-time rendering and free-viewpoint video. However, most pipelines rely on fixed anchor selection such as Farthest Point Sampling (FPS), typically using 8,192 anchors regardless of scene complexity, which over-allocates computation under strict budgets. We propose Efficient Gaussian Streaming (EGS), a plug-in, budget-aware anchor sampler that replaces FPS with a reinforcement-learned policy while keeping the Gaussian streaming reconstruction backbone unchanged. The policy jointly selects an anchor budget and a subset of informative anchors under discrete constraints, balancing reconstruction quality and runtime using spatial features of the Gaussian representation. We evaluate EGS in two settings: fast rendering, which prioritizes runtime efficiency, and high-quality refinement, which enables additional optimization. Experiments on dynamic multi-view datasets show consistent improvements in the quality--efficiency trade-off over FPS sampling. On unseen data, in fast rendering at 256 anchors ($32\times$ fewer than 8,192), EGS improves PSNR by $+0.52$--$0.61$\,dB while running $1.29$--$1.35\times$ faster than IGS@8192 (N3DV and MeetingRoom). In high-quality refinement, EGS remains competitive with the full-anchor baseline at substantially lower anchor budgets. \emph{Code and pretrained checkpoints will be released upon acceptance.} \keywords{4D Gaussian Splatting \and 4D Gaussian Streaming \and Reinforcement Learning}
>
---
#### [replaced 151] Effort-Optimized, Accuracy-Driven Labelling and Validation of Test Inputs for DL Systems: A Mixed-Integer Linear Programming Approach
- **分类: cs.CV; cs.SE**

- **链接: [https://arxiv.org/pdf/2507.04990](https://arxiv.org/pdf/2507.04990)**

> **作者:** Mohammad Hossein Amini; Mehrdad Sabetzadeh; Shiva Nejati
>
> **备注:** Accepted in the Empirical Software Engineering (EMSE) Journal (2026)
>
> **摘要:** Software systems increasingly include AI components based on deep learning (DL). Reliable testing of such systems requires near-perfect test-input validity and label accuracy, with minimal human effort. Yet, the DL community has largely overlooked the need to build highly accurate datasets with minimal effort, since DL training is generally tolerant of labelling errors. This challenge, instead, reflects concerns more familiar to software engineering, where a central goal is to construct high-accuracy test inputs, with accuracy as close to 100% as possible, while keeping associated costs in check. In this article we introduce OPAL, a human-assisted labelling method that can be configured to target a desired accuracy level while minimizing the manual effort required for labelling. The main contribution of OPAL is a mixed-integer linear programming (MILP) formulation that minimizes labelling effort subject to a specified accuracy target. To evaluate OPAL we instantiate it for two tasks in the context of testing vision systems: automatic labelling of test inputs and automated validation of test inputs. Our evaluation, based on more than 2500 experiments performed on nine datasets, comparing OPAL with eight baseline methods, shows that OPAL, relying on its MILP formulation, achieves an average accuracy of 98.8%, while cutting manual labelling by more than half. OPAL significantly outperforms automated labelling baselines in labelling accuracy across all nine datasets, when all methods are provided with the same manual-labelling budget. For automated test-input validation, on average, OPAL reduces manual effort by 28.8% while achieving 4.5% higher accuracy than the SOTA test-input validation baselines. Finally, we show that augmenting OPAL with an active-learning loop leads to an additional 4.5% reduction in required manual labelling, without compromising accuracy.
>
---
#### [replaced 152] Adaptive Attention Distillation for Robust Few-Shot Segmentation under Environmental Perturbations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.03596](https://arxiv.org/pdf/2601.03596)**

> **作者:** Qianyu Guo; Jingrong Wu; Jieji Ren; Weifeng Ge; Wenqiang Zhang
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Few-shot segmentation (FSS) aims to rapidly learn novel class concepts from limited examples to segment specific targets in unseen images, and has been widely applied in areas such as medical diagnosis and industrial inspection. However, existing studies largely overlook the complex environmental factors encountered in real world scenarios-such as illumination, background, and camera viewpoint-which can substantially increase the difficulty of test images. As a result, models trained under laboratory conditions often fall short of practical deployment requirements. To bridge this gap, in this paper, an environment-robust FSS setting is introduced that explicitly incorporates challenging test cases arising from complex environments-such as motion blur, small objects, and camouflaged targets-to enhance model's robustness under realistic, dynamic conditions. An environment robust FSS benchmark (ER-FSS) is established, covering eight datasets across multiple real world scenarios. In addition, an Adaptive Attention Distillation (AAD) method is proposed, which repeatedly contrasts and distills key shared semantics between known (support) and unknown (query) images to derive class-specific attention for novel categories. This strengthens the model's ability to focus on the correct targets in complex environments, thereby improving environmental robustness. Comparative experiments show that AAD improves mIoU by 3.3% - 8.5% across all datasets and settings, demonstrating superior performance and strong generalization. The source code and dataset are available at: this https URL.
>
---
#### [replaced 153] LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13680](https://arxiv.org/pdf/2512.13680)**

> **作者:** Tianye Ding; Yiming Xie; Yiqing Liang; Moitreya Chatterjee; Pedro Miraldo; Huaizu Jiang
>
> **备注:** CVPR 2026, 16 pages
>
> **摘要:** Recent feed-forward reconstruction models like VGGT and $\pi^3$ achieve impressive reconstruction quality but cannot process streaming videos due to quadratic memory complexity, limiting their practical deployment. While existing streaming methods address this through learned memory mechanisms or causal attention, they require extensive retraining and may not fully leverage the strong geometric priors of state-of-the-art offline models. We propose LASER, a training-free framework that converts an offline reconstruction model into a streaming system by aligning predictions across consecutive temporal windows. We observe that simple similarity transformation ($\mathrm{Sim}(3)$) alignment fails due to layer depth misalignment: monocular scale ambiguity causes relative depth scales of different scene layers to vary inconsistently between windows. To address this, we introduce layer-wise scale alignment, which segments depth predictions into discrete layers, computes per-layer scale factors, and propagates them across both adjacent windows and timestamps. Extensive experiments show that LASER achieves state-of-the-art performance on camera pose estimation and point map reconstruction %quality with offline models while operating at 14 FPS with 6 GB peak memory on a RTX A6000 GPU, enabling practical deployment for kilometer-scale streaming videos. Project website: this https URL
>
---
#### [replaced 154] Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.00667](https://arxiv.org/pdf/2603.00667)**

> **作者:** Wentao Huang; Weimin Lyu; Peiliang Lou; Qingqiao Hu; Xiaoling Hu; Shahira Abousamra; Wenchao Han; Ruifeng Guo; Jiawei Zhou; Chao Chen; Chen Wang
>
> **备注:** 14 pages, 8 figures. Accepted by CVPR'26
>
> **摘要:** Computational pathology has advanced rapidly in recent years, driven by domain-specific image encoders and growing interest in using vision-language models to answer natural-language questions about diseases. Yet, the core problem behind pathology question-answering remains unsolved, considering that a gigapixel slide contains far more information than necessary for a given question. Pathologists naturally navigate tissue and morphology complexity by scanning broadly, and zooming in selectively according to the clinical questions. Current models, in contrast, rely on uniform patch sampling or broad attention maps, often attending equally to irrelevant regions while overlooking key visual evidence. In this work, we try to bring models closer to how humans actually examine slides. We propose a question-guided, tissue-aware, and coarse-to-fine retrieval framework, HistoSelect, that consists of two key components: a group sampler that identifies question-relevant tissue regions, followed by a patch selector that retrieves the most informative patches within those regions. By selecting only the most informative patches, our method becomes significantly more efficient: reducing visual token usage by 70% on average, while improving accuracy across three pathology QA tasks. Evaluated on 356,000 question-answer pairs, our approach outperforms existing methods and produces answers grounded in interpretable, pathologist-consistent regions. Our results suggest that bringing human-like search and attention patterns into WSI reasoning is a promising direction for building practical and reliable pathology this http URL is available at this https URL.
>
---
#### [replaced 155] TIGeR: A Unified Framework for Time, Images and Geo-location Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24749](https://arxiv.org/pdf/2603.24749)**

> **作者:** David G. Shatwell; Sirnam Swetha; Mubarak Shah
>
> **备注:** Accepted in CVPR 2026
>
> **摘要:** Many real-world applications in digital forensics, urban monitoring, and environmental analysis require jointly reasoning about visual appearance, location, and time. Beyond standard geo-localization and time-of-capture prediction, these applications increasingly demand more complex capabilities, such as retrieving an image captured at the same location as a query image but at a specified target time. We formalize this problem as Geo-Time Aware Image Retrieval and propose TIGeR, a unified framework for Time, Images and Geo-location Retrieval. TIGeR supports flexible input configurations (single-modality and multi-modality queries) and uses the same representation to perform (i) geo-localization, (ii) time-of-capture prediction, and (iii) geo-time-aware retrieval. By preserving the underlying location identity despite large appearance changes, TIGeR enables retrieval based on where and when a scene was captured, rather than purely on visual similarity. To support this task, we design a multistage data curation pipeline and propose a new diverse dataset of 4.5M paired image-location-time triplets for training and 86k high-quality triplets for evaluation. Extensive experiments show that TIGeR consistently outperforms strong baselines and state-of-the-art methods by up to 16% on time-of-year, 8% time-of-day prediction, and 14% in geo-time aware retrieval recall, highlighting the benefits of unified geo-temporal modeling.
>
---
#### [replaced 156] R3DP: Real-Time 3D-Aware Policy for Embodied Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出R3DP，解决实时具身操作中的3D感知问题。通过融合大模型3D先验，提升操作成功率并保持实时性。**

- **链接: [https://arxiv.org/pdf/2603.14498](https://arxiv.org/pdf/2603.14498)**

> **作者:** Yuhao Zhang; Wanxi Dong; Yue Shi; Yi Liang; Jingnan Gao; Qiaochu Yang; Yaxing Lyu; Zhixuan Liang; Yibin Liu; Congsheng Xu; Xianda Guo; Wei Sui; Yaohui Jin; Xiaokang Yang; Yanyan Xu; Yao Mu
>
> **备注:** Project Page: this https URL Github Repo: this https URL
>
> **摘要:** Embodied manipulation requires accurate 3D understanding of objects and their spatial relations to plan and execute contact-rich actions. While large-scale 3D vision models provide strong priors, their computational cost incurs prohibitive latency for real-time control. We propose Real-time 3D-aware Policy (R3DP), which integrates powerful 3D priors into manipulation policies without sacrificing real-time performance. A core innovation of R3DP is the asynchronous fast-slow collaboration module, which seamlessly integrates large-scale 3D priors into the policy without compromising real-time performance. The system maintains real-time efficiency by querying the pre-trained slow system (VGGT) only on sparse key frames, while simultaneously employing a lightweight Temporal Feature Prediction Network (TFPNet) to predict features for all intermediate frames. By leveraging historical data to exploit temporal correlations, TFPNet explicitly improves task success rates through consistent feature estimation. Additionally, to enable more effective multi-view fusion, we introduce a Multi-View Feature Fuser (MVFF) that aggregates features across views by explicitly incorporating camera intrinsics and extrinsics. R3DP offers a plug-and-play solution for integrating large models into real-time inference systems. We evaluate R3DP against multiple baselines across different visual configurations. R3DP effectively harnesses large-scale 3D priors to achieve superior results, outperforming single-view and multi-view DP by 32.9% and 51.4% in average success rate, respectively. Furthermore, by decoupling heavy 3D reasoning from policy execution, R3DP achieves a 44.8% reduction in inference time compared to a naive DP+VGGT integration.
>
---
#### [replaced 157] PhysVid: Physics Aware Local Conditioning for Generative Video Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.26285](https://arxiv.org/pdf/2603.26285)**

> **作者:** Saurabh Pathak; Elahe Arani; Mykola Pechenizkiy; Bahram Zonooz
>
> **备注:** Accepted for publication in CVPR 2026
>
> **摘要:** Generative video models achieve high visual fidelity but often violate basic physical principles, limiting reliability in real-world settings. Prior attempts to inject physics rely on conditioning: frame-level signals are domain-specific and short-horizon, while global text prompts are coarse and noisy, missing fine-grained dynamics. We present PhysVid, a physics-aware local conditioning scheme that operates over temporally contiguous chunks of frames. Each chunk is annotated with physics-grounded descriptions of states, interactions, and constraints, which are fused with the global prompt via chunk-aware cross-attention during training. At inference, we introduce negative physics prompts (descriptions of locally relevant law violations) to steer generation away from implausible trajectories. On VideoPhy, PhysVid improves physical commonsense scores by $\approx 33\%$ over baseline video generators, and by up to $\approx 8\%$ on VideoPhy2. These results show that local, physics-aware guidance substantially increases physical plausibility in generative video and marks a step toward physics-grounded video models.
>
---
#### [replaced 158] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.21655](https://arxiv.org/pdf/2602.21655)**

> **作者:** Zhijiang Tang; Linhua Wang; Jiaxin Qi; Weihao Jiang; Peng Hou; Anxiang Zeng; Jianqiang Huang
>
> **备注:** Accept by CVPR 2026
>
> **摘要:** Image captioning remains a fundamental task for vision language understanding, yet ground-truth supervision still relies predominantly on human-annotated references. Because human annotations reflect subjective preferences and expertise, ground-truth captions are often incomplete or even incorrect, which in turn limits caption models. We argue that caption quality should be assessed by two objective aspects: completeness (does the caption cover all salient visual facts?) and correctness (are the descriptions true with respect to the image?). To this end, we introduce CCCaption: a dual-reward reinforcement learning framework with a dedicated fine-tuning corpus that explicitly optimizes these properties to generate \textbf{C}omplete and \textbf{C}orrect \textbf{Captions}. For completeness, we use diverse LVLMs to disentangle the image into a set of visual queries, and reward captions that answer more of these queries, with a dynamic query sampling strategy to improve training efficiency. For correctness, we penalize captions that contain hallucinations by validating the authenticity of sub-caption queries, which are derived from the caption decomposition. Our symmetric dual-reward optimization jointly maximizes completeness and correctness, guiding models toward captions that better satisfy these objective criteria. Extensive experiments across standard captioning benchmarks show consistent improvements, offering a principled path to training caption models beyond human-annotation imitation.
>
---
#### [replaced 159] FA-Seg: A Fast and Accurate Diffusion-Based Method for Open-Vocabulary Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.23323](https://arxiv.org/pdf/2506.23323)**

> **作者:** Huy Che; Vinh-Tiep Nguyen
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) aims to segment objects from arbitrary text categories without requiring densely annotated datasets. Although contrastive learning based models enable zero-shot segmentation, they often lose fine spatial precision at pixel level, due to global representation bias. In contrast, diffusion-based models naturally encode fine-grained spatial features via attention mechanisms that capture both global context and local details. However, they often face challenges in balancing the computation costs and the quality of the segmentation mask. In this work, we present FA-Seg, a Fast and Accurate training-free framework for open-vocabulary segmentation based on diffusion models. FA-Seg performs segmentation using only a (1+1)-step from a pretrained diffusion model. Moreover, instead of running multiple times for different classes, FA-Seg performs segmentation for all classes at once. To further enhance the segmentation quality, FA-Seg introduces three key components: (i) a dual-prompt mechanism for discriminative, class-aware attention extraction, (ii) a Hierarchical Attention Refinement Method (HARD) that enhances semantic precision via multi-resolution attention fusion, and (iii) a Test-Time Flipping (TTF) scheme designed to improve spatial consistency. Extensive experiments show that FA-Seg achieves state-of-the-art training-free performance, obtaining 43.8% average mIoU across PASCAL VOC, PASCAL Context, and COCO Object benchmarks while maintaining superior inference efficiency. Our results demonstrate that FA-Seg provides a strong foundation for extendability, bridging the gap between segmentation quality and inference efficiency. The source code is available at this https URL.
>
---
#### [replaced 160] Resolving Spatio-Temporal Entanglement in Video Prediction via Multi-Modal Attention
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视频预测任务，旨在解决传统模型在长期时间一致性和高分辨率细节上的不足。提出MAUCell架构，结合多模态注意力机制，提升视频生成的准确性和实时性。**

- **链接: [https://arxiv.org/pdf/2501.16997](https://arxiv.org/pdf/2501.16997)**

> **作者:** Shreyam Gupta; P. Agrawal; Priyam Gupta
>
> **备注:** 11 pages, 3 figures, 5 tables, and 3 Algorithms
>
> **摘要:** The fast progress in computer vision has necessitated more advanced methods for temporal sequence modeling. This area is essential for the operation of autonomous systems, real-time surveillance, and predicting anomalies. As the demand for accurate video prediction increases, the limitations of traditional deterministic models, particularly their struggle to maintain long-term temporal coherence while providing high-frequency spatial detail, have become very clear. This report provides an exhaustive analysis of the Multi-Attention Unit Cell (MAUCell), a novel architectural framework that represents a significant leap forward in video frame prediction. By synergizing Generative Adversarial Networks (GANs) with a hierarchical "STAR-GAN" processing strategy and a triad of specialized attention mechanisms (Temporal, Spatial, and Pixel-wise), the MAUCell addresses the persistent "deep-in-time" dilemma that plagues Recurrent Neural Networks (RNNs). Our analysis shows that the MAUCell framework successfully establishes a new state-of-the-art benchmark, especially in its ability to produce realistic video sequences that closely resemble real-world footage while ensuring efficient inference for real-time deployment. Through rigorous evaluation on datasets: Moving MNIST, KTH Action, and CASIA-B, the framework shows superior performance metrics, especially in Learned Perceptual Image Patch Similarity (LPIPS) and Structural Similarity Index (SSIM). This success confirms its dual-pathway information transformation system. This report details the theoretical foundations, detailed structure and broader significance of MAUCell, presenting it as a valuable solution for video forecasting tasks that require high precision and limited resources.
>
---
#### [replaced 161] AnyHand: A Large-Scale Synthetic Dataset for RGB(-D) Hand Pose Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.25726](https://arxiv.org/pdf/2603.25726)**

> **作者:** Chen Si; Yulin Liu; Bo Ai; Jianwen Xie; Rolandos Alexandros Potamias; Chuanxia Zheng; Hao Su
>
> **摘要:** We present AnyHand, a large-scale synthetic dataset designed to advance the state of the art in 3D hand pose estimation from both RGB-only and RGB-D inputs. While recent works with foundation approaches have shown that an increase in the quantity and diversity of training data can markedly improve performance and robustness in hand pose estimation, existing real-world-collected datasets on this task are limited in coverage, and prior synthetic datasets rarely provide occlusions, arm details, and aligned depth together at scale. To address this bottleneck, our AnyHand contains 2.5M single-hand and 4.1M hand-object interaction RGB-D images, with rich geometric annotations. In the RGB-only setting, we show that extending the original training sets of existing baselines with AnyHand yields significant gains on multiple benchmarks (FreiHAND and HO-3D), even when keeping the architecture and training scheme fixed. More impressively, the model trained with AnyHand shows stronger generalization to the out-of-domain HO-Cap dataset, without any fine-tuning. We also contribute a lightweight depth fusion module that can be easily integrated into existing RGB-based models. Trained with AnyHand, the resulting RGB-D model achieves superior performance on the HO-3D benchmark, showing the benefits of depth integration and the effectiveness of our synthetic data.
>
---
#### [replaced 162] OmniStyle2: Learning to Stylize by Learning to Destylize
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.05970](https://arxiv.org/pdf/2509.05970)**

> **作者:** Ye Wang; Zili Yi; Yibo Zhang; Peng Zheng; Xuping Xie; Jiang Lin; Yijun Li; Yilin Wang; Rui Ma
>
> **备注:** Our project page: this https URL
>
> **摘要:** This paper introduces a scalable paradigm for supervised style transfer by inverting the problem: instead of learning to stylize directly, we learn to destylize, reducing stylistic elements from artistic images to recover their natural counterparts and thereby producing authentic, pixel-aligned training pairs at scale. To realize this paradigm, we propose DeStylePipe, a progressive, multi-stage destylization framework that begins with global general destylization, advances to category-wise instruction adaptation, and ultimately deploys specialized model adaptation for complex styles that prompt engineering alone cannot handle. Tightly integrated into this pipeline, DestyleCoT-Filter employs Chain-of-Thought reasoning to assess content preservation and style removal at each stage, routing challenging samples forward while discarding persistently low-quality pairs. Built on this framework, we construct DeStyle-350K, a large-scale dataset aligning diverse artistic styles with their underlying content. We further introduce BCS-Bench, a benchmark featuring balanced content generality and style diversity for systematic evaluation. Extensive experiments demonstrate that models trained on DeStyle-350K achieve superior stylization quality, validating destylization as a reliable and scalable supervision paradigm for style transfer.
>
---
#### [replaced 163] FaceLinkGen: Rethinking Identity Leakage in Privacy-Preserving Face Recognition with Identity Extraction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.02914](https://arxiv.org/pdf/2602.02914)**

> **作者:** Wenqi Guo; Shan Du
>
> **摘要:** Transformation-based privacy-preserving face recognition (PPFR) aims to verify identities while hiding facial data from attackers and malicious service providers. Existing evaluations mostly treat privacy as resistance to pixel-level reconstruction, measured by PSNR and SSIM. We show that this reconstruction-centric view fails. We present FaceLinkGen, an identity extraction attack that performs linkage/matching and face regeneration directly from protected templates without recovering original pixels. On three recent PPFR systems, FaceLinkGen reaches over 98.5\% matching accuracy and above 96\% regeneration success, and still exceeds 92\% matching and 94\% regeneration in a near zero knowledge setting. These results expose a structural gap between pixel distortion metrics, which are widely used in PPFR evaluation, and real privacy. We show that visual obfuscation leaves identity information broadly exposed to both external intruders and untrusted service providers.
>
---
#### [replaced 164] Chain of Event-Centric Causal Thought for Physically Plausible Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09094](https://arxiv.org/pdf/2603.09094)**

> **作者:** Zixuan Wang; Yixin Hu; Haolan Wang; Feng Chen; Yan Liu; Wen Li; Yinjie Lei
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Physically Plausible Video Generation (PPVG) has emerged as a promising avenue for modeling real-world physical phenomena. PPVG requires an understanding of commonsense knowledge, which remains a challenge for video diffusion models. Current approaches leverage commonsense reasoning capability of large language models to embed physical concepts into prompts. However, generation models often render physical phenomena as a single moment defined by prompts, due to the lack of conditioning mechanisms for modeling causal progression. In this paper, we view PPVG as generating a sequence of causally connected and dynamically evolving events. To realize this paradigm, we design two key modules: (1) Physics-driven Event Chain Reasoning. This module decomposes the physical phenomena described in prompts into multiple elementary event units, leveraging chain-of-thought reasoning. To mitigate causal ambiguity, we embed physical formulas as constraints to impose deterministic causal dependencies during reasoning. (2) Transition-aware Cross-modal Prompting (TCP). To maintain continuity between events, this module transforms causal event units into temporally aligned vision-language prompts. It summarizes discrete event descriptions to obtain causally consistent narratives, while progressively synthesizing visual keyframes of individual events by interactive editing. Comprehensive experiments on PhyGenBench and VideoPhy benchmarks demonstrate that our framework achieves superior performance in generating physically plausible videos across diverse physical domains. Code is available at this https URL.
>
---
#### [replaced 165] Relightable Holoported Characters: Capturing and Relighting Dynamic Human Performance from Sparse Views
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.00255](https://arxiv.org/pdf/2512.00255)**

> **作者:** Kunwar Maheep Singh; Jianchun Chen; Vladislav Golyanik; Stephan J. Garbin; Thabo Beeler; Rishabh Dabral; Marc Habermann; Christian Theobalt
>
> **摘要:** We present Relightable Holoported Characters (RHC), a novel person-specific method for free-view rendering and relighting of full-body and highly dynamic humans solely observed from sparse-view RGB videos at inference. In contrast to classical one-light-at-a-time (OLAT)-based human relighting, our transformer-based RelightNet predicts relit appearance within a single network pass, avoiding costly OLAT-basis capture and generation. For training such a model, we introduce a new capture strategy and dataset recorded in a multi-view lightstage, where we alternate frames lit by random environment maps with uniformly lit tracking frames, simultaneously enabling accurate motion tracking and diverse illumination as well as dynamics coverage. Inspired by the rendering equation, we derive physics-informed features that encode geometry, albedo, shading, and the virtual camera view from a coarse human mesh proxy and the input views. Our RelightNet then takes these features as input and cross-attends them with a novel lighting condition, and regresses the relit appearance in the form of texel-aligned 3D Gaussian splats attached to the coarse mesh proxy. Consequently, our RelightNet implicitly learns to efficiently compute the rendering equation for novel lighting conditions within a single feed-forward pass. Experiments demonstrate our method's superior visual fidelity and lighting reproduction compared to state-of-the-art approaches. Project page: this https URL
>
---
#### [replaced 166] More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.25848](https://arxiv.org/pdf/2509.25848)**

> **作者:** Xinyu Tian; Shu Zou; Zhaoyuan Yang; Mengqi He; Fabian Waschkowski; Lukas Wesemann; Peter Tu; Jing Zhang
>
> **备注:** Accepted to ICLR2026
>
> **摘要:** Reasoning has emerged as a pivotal capability in Large Language Models (LLMs). Through Reinforcement Learning (RL), typically Group Relative Policy Optimization (GRPO), these models are able to solve complex tasks such as mathematics and code generation. Building on these advances, recent research has sought to extend reasoning to Vision-Language Models (VLMs), yielding promising results across diverse visual tasks. Despite this progress, our study uncovers the dual nature of multimodal reasoning: while it substantially enhances logical inference and facilitates performance on challenging problems, it may gradually impair perceptual grounding, leading to recognition failures on otherwise basic visual questions. Through further analysis, we attribute this phenomenon to visual forgetting, wherein prolonged reasoning causes the model to increasingly disregard visual input. To address this, we propose Vision-Anchored Policy Optimization (VAPO), a simple yet effective method that explicitly steers the reasoning process toward visually grounded trajectories. Our result model, VAPO-Thinker-7B, significantly strengthens the model's reliance on visual information and achieves new state-of-the-art results on a wide range of established benchmarks. Project page: this https URL
>
---
#### [replaced 167] EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21332](https://arxiv.org/pdf/2603.21332)**

> **作者:** Haolan Xu; Keli Cheng; Lei Wang; Ning Bi; Xiaoming Liu
>
> **备注:** Accepted by CVPR 2026. Page: this https URL
>
> **摘要:** Audio-driven 3D talking head synthesis has advanced rapidly with Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). By leveraging rich pre-trained priors, few-shot methods enable instant personalization from just a few seconds of video. However, under expressive facial motion, existing few-shot approaches often suffer from geometric instability and audio-emotion mismatch, highlighting the need for more effective emotion-aware motion modeling. In this work, we present EmoTaG, a few-shot emotion-aware 3D talking head synthesis framework built on the Pretrain-and-Adapt paradigm. Our key insight is to reformulate motion prediction in a structured FLAME parameter space rather than directly deforming 3D Gaussians, thereby introducing explicit geometric priors that improve motion stability. Building upon this, we propose a Gated Residual Motion Network (GRMN), which captures emotional prosody from audio while supplementing head pose and upper-face cues absent from audio, enabling expressive and coherent motion generation. Extensive experiments demonstrate that EmoTaG achieves state-of-the-art performance in emotional expressiveness, lip synchronization, visual realism, and motion stability.
>
---
#### [replaced 168] Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.22150](https://arxiv.org/pdf/2601.22150)**

> **作者:** Xiaoxiao Sun; Mingyang Li; Kun Yuan; Min Woo Sun; Mark Endo; Shengguang Wu; Changlin Li; Yuhui Zhang; Zeyu Wang; Serena Yeung-Levy
>
> **备注:** 26 pages, 31 figures, 13 tables. Project Page: this https URL
>
> **摘要:** Large Vision-Language Models (VLMs) often answer classic visual illusions "correctly" on original images, yet persist with the same responses when illusion factors are inverted, even though the visual change is obvious to humans. This raises a fundamental question: do VLMs perceive visual changes or merely recall memorized patterns? While several studies have noted this phenomenon, the underlying causes remain unclear. To move from observations to systematic understanding, this paper introduces VI-Probe, a controllable visual-illusion framework with graded perturbations and matched visual controls (without illusion inducer) that disentangles visually grounded perception from language-driven recall. Unlike prior work that focuses on averaged accuracy, we measure stability and sensitivity using Polarity-Flip Consistency, Template Fixation Index, and an illusion multiplier normalized against matched controls. Experiments across different families reveal that response persistence arises from heterogeneous causes rather than a single mechanism. For instance, GPT-5 exhibits memory override, Claude-Opus-4.1 shows perception-memory competition, while Qwen variants suggest visual-processing limits. Our findings challenge single-cause views and motivate probing-based evaluation that measures both knowledge and sensitivity to controlled visual change. Data and code are available at this https URL
>
---
#### [replaced 169] DyaDiT: A Multi-Modal Diffusion Transformer for Socially Favorable Dyadic Gesture Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23165](https://arxiv.org/pdf/2602.23165)**

> **作者:** Yichen Peng; Jyun-Ting Song; Siyeol Jung; Ruofan Liu; Haiyang Liu; Xuangeng Chu; Ruicong Liu; Erwin Wu; Hideki Koike; Kris Kitani
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Generating realistic conversational gestures are essential for achieving natural, socially engaging interactions with digital humans. However, existing methods typically map a single audio stream to a single speaker's motion, without considering social context or modeling the mutual dynamics between two people engaging in conversation. We present DyaDiT, a multi-modal diffusion transformer that generates contextually appropriate human motion from dyadic audio signals. Trained on Seamless Interaction Dataset, DyaDiT takes dyadic audio with optional social-context tokens to produce context-appropriate motion. It fuses information from both speakers to capture interaction dynamics, uses a motion dictionary to encode motion priors, and can optionally utilize the conversational partner's gestures to produce more responsive motion. We evaluate DyaDiT on standard motion generation metrics and conduct quantitative user studies, demonstrating that it not only surpasses existing methods on objective metrics but is also strongly preferred by users, highlighting its robustness and socially favorable motion generation. Code and models will be released upon acceptance.
>
---
#### [replaced 170] Reconstruct Anything Model: a lightweight general model for computational imaging
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08915](https://arxiv.org/pdf/2503.08915)**

> **作者:** Matthieu Terris; Samuel Hurault; Maxime Song; Julian Tachella
>
> **摘要:** Most existing learning-based methods for solving imaging inverse problems can be roughly divided into two classes: iterative algorithms, such as plug-and-play and diffusion methods leveraging pretrained denoisers, and unrolled architectures that are trained end-to-end for specific imaging problems. Iterative methods in the first class are computationally costly and often yield suboptimal reconstruction performance, whereas unrolled architectures are generally problem-specific and require expensive training. In this work, we propose a novel non-iterative, lightweight architecture that incorporates knowledge about the forward operator (acquisition physics and noise parameters) without relying on unrolling. Our model is trained to solve a wide range of inverse problems, such as deblurring, magnetic resonance imaging, computed tomography, inpainting, and super-resolution, and handles arbitrary image sizes and channels, such as grayscale, complex, and color data. The proposed model can be easily adapted to unseen inverse problems or datasets with a few fine-tuning steps (up to a few images) in a self-supervised way, without ground-truth references. Throughout a series of experiments, we demonstrate state-of-the-art performance from medical imaging to low-photon imaging and microscopy. Our code is available at this https URL.
>
---
#### [replaced 171] WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24836](https://arxiv.org/pdf/2603.24836)**

> **作者:** Yihan Wang; Jia Deng
>
> **摘要:** We introduce WAFT-Stereo, a simple and effective warping-based method for stereo matching. WAFT-Stereo demonstrates that cost volumes, a common design used in many leading methods, are not necessary for strong performance and can be replaced by warping with improved efficiency. WAFT-Stereo ranks first on ETH3D (BP-0.5), Middlebury (RMSE), and KITTI (all metrics), reducing the zero-shot error by 81% on ETH3D, while being 1.8-6.7x faster than competitive methods. Code and model weights are available at this https URL.
>
---
#### [replaced 172] VRR-QA: Visual Relational Reasoning in Videos Beyond Explicit Cues
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.21742](https://arxiv.org/pdf/2506.21742)**

> **作者:** Sirnam Swetha; Rohit Gupta; Parth Parag Kulkarni; David G Shatwell; Jeffrey A Chan Santiago; Nyle Siddiqui; Joseph Fioresi; Mubarak Shah
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Video Question Answering (VideoQA) has made significant strides by leveraging multimodal learning to align visual and textual modalities. However, current benchmarks overwhelmingly focus on questions answerable through explicit visual content - actions, objects, and events - directly observable within individual frames or short clips. To truly understand videos as humans do, models must go beyond what is directly shown, inferring hidden relationships and contextual cues that are only implied across frames. Current benchmarks fail to capture this essential aspect of video understanding. To address this gap, we introduce VRR-QA, a benchmark for Visual Relational Reasoning Beyond Explicit Cues. We curate our benchmark from creative and cinematic videos such as movies, that deliberately employ storytelling techniques which omit direct depictions of certain events or relations, requiring viewers to infer them. VRR-QA comprises 1K meticulously expert-annotated QA pairs drawn from 1K creative video clips covering 15 genres across 7 decades of content, from both live-action and animated titles. Our extensive evaluations on 14 leading VideoQA models reveals consistent and significant performance degradation, underscoring their reliance on surface-level visual cues and highlighting the difficulty of implicit reasoning. Even the best model substantially underperforms human baselines with only 64% accuracy. Performance variations across models further illustrate the complexity and diversity of the challenges presented by VRR-QA. By releasing both dataset and data collection framework, VRR-QA establishes a rigorous, diverse, and reproducible testbed for advancing VideoQA: this https URL.
>
---
#### [replaced 173] ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19575](https://arxiv.org/pdf/2602.19575)**

> **作者:** Minseo Kim; Minchan Kwon; Dongyeun Lee; Yunho Jeon; Junmo Kim
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Personalized text-to-image (T2I) generation has emerged as a key application for creating user-specific concepts from a few reference images. The core challenge is concept disentanglement: separating the target concept from irrelevant residual information. Lacking such disentanglement, capturing high-fidelity features often incorporates undesired attributes that conflict with user prompts, compromising the trade-off between concept fidelity and text alignment. While existing methods rely on manual guidance, they often fail to represent intricate visual details and lack scalability. We introduce ConceptPrism, a framework that extracts shared features exclusively through cross-image comparison without external information. We jointly optimize a target token and image-wise residual tokens via reconstruction and exclusion losses. By suppressing shared information in residual tokens, the exclusion loss creates an information vacuum that forces the target token to capture the common concept. Extensive evaluations demonstrate that ConceptPrism achieves accurate concept disentanglement and significantly improves overall performance across diverse and complex visual concepts. The code is available at this https URL.
>
---
#### [replaced 174] Event-based Facial Keypoint Alignment via Cross-Modal Fusion Attention and Self-Supervised Multi-Event Representation Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24968](https://arxiv.org/pdf/2509.24968)**

> **作者:** Donghwa Kang; Junho Kim; Dongwoo Kang
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** Event cameras offer unique advantages for facial keypoint alignment under challenging conditions, such as low light and rapid motion, due to their high temporal resolution and robustness to varying illumination. However, existing RGB facial keypoint alignment methods do not perform well on event data, and training solely on event data often leads to suboptimal performance because of its limited spatial information. Moreover, the lack of comprehensive labeled event datasets further hinders progress in this area. To address these issues, we propose a novel framework based on cross-modal fusion attention (CMFA) and self-supervised multi-event representation learning (SSMER) for event-based facial keypoint alignment. Our framework employs CMFA to integrate corresponding RGB data, guiding the model to extract robust facial features from event input images. In parallel, SSMER enables effective feature learning from unlabeled event data, overcoming spatial limitations. Extensive experiments on our real-event E-SIE dataset and a synthetic-event version of the public WFLW-V benchmark show that our approach consistently surpasses state-of-the-art methods across multiple evaluation metrics.
>
---
#### [replaced 175] Vega: Learning to Drive with Natural Language Instructions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Vega模型，解决自动驾驶中根据自然语言指令进行个性化驾驶的问题。通过构建大规模数据集并融合视觉、语言和动作模态，提升指令跟随与路径规划能力。**

- **链接: [https://arxiv.org/pdf/2603.25741](https://arxiv.org/pdf/2603.25741)**

> **作者:** Sicheng Zuo; Yuxuan Li; Wenzhao Zheng; Zheng Zhu; Jie Zhou; Jiwen Lu
>
> **备注:** Code is available at this https URL
>
> **摘要:** Vision-language-action models have reshaped autonomous driving to incorporate languages into the decision-making process. However, most existing pipelines only utilize the language modality for scene descriptions or reasoning and lack the flexibility to follow diverse user instructions for personalized driving. To address this, we first construct a large-scale driving dataset (InstructScene) containing around 100,000 scenes annotated with diverse driving instructions with the corresponding trajectories. We then propose a unified Vision-Language-World-Action model, Vega, for instruction-based generation and planning. We employ the autoregressive paradigm to process visual inputs (vision) and language instructions (language) and the diffusion paradigm to generate future predictions (world modeling) and trajectories (action). We perform joint attention to enable interactions between the modalities and use individual projection layers for different modalities for more capabilities. Extensive experiments demonstrate that our method not only achieves superior planning performance but also exhibits strong instruction-following abilities, paving the way for more intelligent and personalized driving systems.
>
---
#### [replaced 176] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15508](https://arxiv.org/pdf/2512.15508)**

> **作者:** Arthur Moreau; Richard Shaw; Michal Nazarczuk; Jisu Shin; Thomas Tanay; Zhensong Zhang; Songcen Xu; Eduardo Pérez-Pellitero
>
> **备注:** CVPR 2026 camera ready version
>
> **摘要:** Feed-forward 3D Gaussian Splatting (3DGS) models enable real-time scene generation but are hindered by suboptimal pixel-aligned primitive placement, which relies on a dense, rigid grid that limits both quality and efficiency. We introduce a new feed-forward architecture that detects 3D Gaussian primitives at a sub-pixel level, replacing the pixel grid with an adaptive, ``Off-The-Grid" distribution. Inspired by keypoint detection, our decoder learns to locally distribute primitives across image patches. We also provide an Adaptive Density mechanism by assigning varying number of primitives per patch based on Shannon entropy. We combine the proposed decoder with a pre-trained 3D reconstruction backbone and train them end-to-end using photometric supervision without any 3D annotation. The resulting pose-free model generates photorealistic 3DGS scenes in seconds, achieving state-of-the-art novel view synthesis for feed-forward models. It outperforms competitors while using far fewer primitives, demonstrating a more accurate and efficient allocation that captures fine details and reduces artifacts. Project page: this https URL.
>
---
#### [replaced 177] PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2601.17354](https://arxiv.org/pdf/2601.17354)**

> **作者:** Wenzhi Guo; Guangchi Fang; Shu Yang; Bing Wang
>
> **摘要:** Efficient and high-fidelity 3D scene modeling is a long-standing pursuit in computer graphics. While recent 3D Gaussian Splatting (3DGS) methods achieve impressive real-time modeling performance, they rely on resource-unconstrained training assumptions that fail on mobile devices, which are limited by minute-scale training budgets and hardware-available peak-memory. We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high perceptual fidelity. Our method resolves the fundamental contradictions of standard 3DGS through three co-designed operators: G builds geometry-faithful point-cloud priors; I injects local surface statistics to seed anisotropic Gaussians, thereby reducing early conditioning gaps; and T unrolls alpha compositing with cached intermediates and index-mapped gradient scattering for stable mobile backpropagation. Collectively, these operators satisfy the competing requirements of training efficiency, memory compactness, and modeling fidelity. Extensive experiments demonstrate that PocketGS is able to outperform the powerful mainstream workstation 3DGS baseline to deliver high-quality reconstructions, enabling a fully on-device, practical capture-to-rendering workflow.
>
---
#### [replaced 178] See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2509.02028](https://arxiv.org/pdf/2509.02028)**

> **作者:** Halima Bouzidi; Haoyu Liu; Mohammad Abdullah Al Faruque
>
> **备注:** Accepted to the NeurIPS 2025 Workshop on Reliable ML from Unreliable Data
>
> **摘要:** Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.
>
---
#### [replaced 179] SciEGQA: A Dataset for Scientific Evidence-Grounded Question Answering and Reasoning
- **分类: cs.DB; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15090](https://arxiv.org/pdf/2511.15090)**

> **作者:** Wenhan Yu; Zhaoxi Zhang; Wang Chen; Guanqiang Qi; Weikang Li; Lei Sha; Deguo Xia; Jizhou Huang
>
> **备注:** 8 pages, 4 figures, 3 tables
>
> **摘要:** Scientific documents contain complex multimodal structures, which makes evidence localization and scientific reasoning in Document Visual Question Answering particularly challenging. However, most existing benchmarks evaluate models only at the page level without explicitly annotating the evidence regions that support the answer, which limits both interpretability and the reliability of evaluation. To address this limitation, we introduce SciEGQA, a scientific document question answering and reasoning dataset with semantic evidence grounding, where supporting evidence is represented as semantically coherent document regions annotated with bounding boxes. SciEGQA consists of two components: a **human-annotated fine-grained benchmark** containing 1,623 high-quality question--answer pairs, and a **large-scale automatically constructed training set** with over 30K QA pairs generated through an automated data construction pipeline. Extensive experiments on a wide range of Vision-Language Models (VLMs) show that existing models still struggle with evidence localization and evidence-based question answering in scientific documents. Training on the proposed dataset significantly improves the scientific reasoning capabilities of VLMs. The project page is available at this https URL.
>
---
#### [replaced 180] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态任务，旨在解决Omni LLMs中的跨模态幻觉问题。提出MoD-DPO框架，通过模态解耦优化提升模型对不同模态的准确感知和抗幻觉能力。**

- **链接: [https://arxiv.org/pdf/2603.03192](https://arxiv.org/pdf/2603.03192)**

> **作者:** Ashutosh Chaubey; Jiacheng Pang; Mohammad Soleymani
>
> **备注:** CVPR 2026. Project Page: this https URL
>
> **摘要:** Omni-modal large language models (omni LLMs) have recently achieved strong performance across audiovisual understanding tasks, yet they remain highly susceptible to cross-modal hallucinations arising from spurious correlations and dominant language priors. In this work, we propose Modality-Decoupled Direct Preference Optimization (MoD-DPO), a simple and effective framework for improving modality grounding in omni LLMs. MoD-DPO introduces modality-aware regularization terms that explicitly enforce invariance to corruptions in irrelevant modalities and sensitivity to perturbations in relevant modalities, thereby reducing unintended cross-modal interactions. To further mitigate over-reliance on textual priors, we incorporate a language-prior debiasing penalty that discourages hallucination-prone text-only responses. Extensive experiments across multiple audiovisual hallucination benchmarks demonstrate that MoD-DPO consistently improves perception accuracy and hallucination resistance, outperforming previous preference optimization baselines under similar training budgets. Our findings underscore the importance of modality-faithful alignment and demonstrate a scalable path toward more reliable and resilient multimodal foundation models.
>
---
#### [replaced 181] Hierarchical Concept Embedding & Pursuit for Interpretable Image Classification
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.11448](https://arxiv.org/pdf/2602.11448)**

> **作者:** Nghia Nguyen; Tianjiao Ding; René Vidal
>
> **备注:** To be published in Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Interpretable-by-design models are gaining traction in computer vision because they provide faithful explanations for their predictions. In image classification, these models typically recover human-interpretable concepts from an image and use them for classification. Sparse concept recovery methods leverage the latent space of vision-language models to represent image embeddings as sparse combinations of concept embeddings. However, by ignoring the hierarchical structure of semantic concepts, these methods may produce correct predictions with explanations that are inconsistent with the hierarchy. In this work, we propose Hierarchical Concept Embedding & Pursuit (HCEP), a framework that induces a hierarchy of concept embeddings in the latent space and performs hierarchical sparse coding to recover the concepts present in an image. Given a hierarchy of semantic concepts, we introduce a geometric construction for the corresponding hierarchy of embeddings. Under the assumption that the true concepts form a rooted path in the hierarchy, we derive sufficient conditions for their recovery in the embedding space. We further show that hierarchical sparse coding reliably recovers hierarchical concept embeddings, whereas standard sparse coding fails. Experiments on real-world datasets show that HCEP improves concept precision and recall compared to existing methods while maintaining competitive classification accuracy. Moreover, when the number of samples available for concept estimation and classifier training is limited, HCEP achieves superior classification accuracy and concept recovery. Our results demonstrate that incorporating hierarchical structure into sparse concept recovery leads to more faithful and interpretable image classification models.
>
---
#### [replaced 182] vGamba: Attentive State Space Bottleneck for efficient Long-range Dependencies in Visual Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.21262](https://arxiv.org/pdf/2503.21262)**

> **作者:** Yunusa Haruna; Adamu Lawan; Shamsuddeen Hassan Muhammad; Jiaquan Zhang; Chaoning Zhang
>
> **摘要:** Capturing long-range dependencies (LRD) efficiently is a core challenge in visual recognition, and state-space models (SSMs) have recently emerged as a promising alternative to self-attention for addressing it. However, adapting SSMs into CNN-based bottlenecks remains challenging, as existing approaches require complex pre-processing and multiple SSM replicas per block, limiting their practicality. We propose vGamba, a hybrid vision backbone that replaces the standard bottleneck convolution with a single lightweight SSM block, the Gamba cell, which incorporates 2D positional awareness and an attentive spatial context (ASC) module for efficient LRD modeling. Results on diverse downstream vision tasks demonstrate competitive accuracy against SSM-based models such as VMamba and ViM, while achieving significantly improved computation and memory efficiency over Bottleneck Transformer (BotNet). For example, at $2048 \times 2048$ resolution, vGamba is $2.07 \times$ faster than BotNet and reduces peak GPU memory by 93.8% (1.03GB vs. 16.78GB), scaling near-linearly with resolution comparable to ResNet-50. These results demonstrate that Gamba Bottleneck effectively overcomes the memory and compute constraints of BotNet global modeling, establishing it as a practical and scalable backbone for high-resolution vision tasks.
>
---
#### [replaced 183] AVERY: Intent-Driven Adaptive VLM Split Computing via Embodied Self-Awareness for Efficient Disaster Response Systems
- **分类: cs.DC; cs.AR; cs.CV; cs.LG; cs.NI**

- **链接: [https://arxiv.org/pdf/2511.18151](https://arxiv.org/pdf/2511.18151)**

> **作者:** Rajat Bhattacharjya; Sing-Yao Wu; Hyunwoo Oh; Chaewon Nam; Suyeon Koo; Mohsen Imani; Elaheh Bozorgzadeh; Nikil Dutt
>
> **备注:** Paper is currently under review. Authors' version posted for personal use and not for redistribution. Previous version of the preprint was titled: 'AVERY: Adaptive VLM Split Computing through Embodied Self-Awareness for Efficient Disaster Response Systems'
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) in disaster response require complex, queryable intelligence that onboard CNNs cannot provide. While Vision-Language Models (VLMs) offer this semantic reasoning, their high resource demands make on-device deployment infeasible, and naive cloud offloading fails under the low-bandwidth, unstable networks endemic to disaster zones. We present AVERY, an intent-driven adaptive split computing framework for efficient VLM deployment on resource-constrained platforms. AVERY is motivated by the observation that operator intent must be treated as a first-class system objective, since missions such as broad situational monitoring and precise, spatially grounded investigation require different semantic products, latency targets, and resource allocations. To reflect this, AVERY advances split computing beyond traditional depth-wise partitioning through a functional, cognitive-inspired dual-stream split: a high-frequency, low-resolution Context stream for real-time awareness, and a low-frequency, high-fidelity Insight stream for deep analysis. This design enables a hierarchical split strategy: computation is first separated by function, then partitioned depth-wise across edge and cloud when the Insight stream is required. A lightweight, self-aware onboard controller monitors network conditions and operator intent to select from pre-trained compression models, navigating the accuracy-throughput trade-off at runtime. Evaluated using LISA-7B in an edge-cloud setting under fluctuating network conditions, AVERY achieves 11.2% higher accuracy than raw image compression, 93.98% lower energy consumption than full-edge execution, and average accuracy within 0.75% of the static High-Accuracy baseline during dynamic adaptation. Overall, AVERY enhances mission efficiency and enables real-time, queryable intelligence in dynamic disaster environments.
>
---
#### [replaced 184] Image Generation Models: A Technical History
- **分类: cs.CV; cs.AI; cs.CL; cs.GR**

- **简介: 该论文属于图像生成任务，旨在系统梳理各类生成模型及其发展，解决模型碎片化问题，综述模型原理、优化方法及应用挑战。**

- **链接: [https://arxiv.org/pdf/2603.07455](https://arxiv.org/pdf/2603.07455)**

> **作者:** Rouzbeh Shirvani
>
> **摘要:** Image generation has advanced rapidly over the past decade, yet the literature seems fragmented across different models and application domains. This paper aims to offer a comprehensive survey of breakthrough image generation models, including variational autoencoders (VAEs), generative adversarial networks (GANs), normalizing flows, autoregressive and transformer-based generators, and diffusion-based methods. We provide a detailed technical walkthrough of each model type, including their underlying objectives, architectural building blocks, and algorithmic training steps. For each model type, we present the optimization techniques as well as common failure modes and limitations. We also go over recent developments in video generation and present the research works that made it possible to go from still frames to high quality videos. Lastly, we cover the growing importance of robustness and responsible deployment of these models, including deepfake risks, detection, artifacts, and watermarking.
>
---
#### [replaced 185] The Quest for Generalizable Motion Generation: Data, Model, and Evaluation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.26794](https://arxiv.org/pdf/2510.26794)**

> **作者:** Jing Lin; Ruisi Wang; Junzhe Lu; Ziqi Huang; Guorui Song; Ailing Zeng; Xian Liu; Chen Wei; Wanqi Yin; Qingping Sun; Zhongang Cai; Lei Yang; Ziwei Liu
>
> **备注:** Homepage: this https URL
>
> **摘要:** Despite recent advances in 3D human motion generation (MoGen) on standard benchmarks, existing text-to-motion models still face a fundamental bottleneck in their generalization capability. In contrast, adjacent generative fields, most notably video generation (ViGen), have demonstrated remarkable generalization in modeling human behaviors, highlighting transferable insights that MoGen can leverage. Motivated by this observation, we present a comprehensive framework that systematically transfers knowledge from ViGen to MoGen across three key pillars: data, modeling, and evaluation. First, we introduce ViMoGen-228K, a large-scale dataset comprising 228,000 high-quality motion samples that integrates high-fidelity optical MoCap data with semantically annotated motions from web videos and synthesized samples generated by state-of-the-art ViGen models. The dataset includes both text-motion pairs and text-video-motion triplets, substantially expanding semantic diversity. Second, we propose ViMoGen, a flow-matching-based diffusion transformer that unifies priors from MoCap data and ViGen models through gated multimodal conditioning. To enhance efficiency, we further develop ViMoGen-light, a distilled variant that eliminates video generation dependencies while preserving strong generalization. Finally, we present MBench, a hierarchical benchmark designed for fine-grained evaluation across motion quality, prompt fidelity, and generalization ability. Extensive experiments show that our framework significantly outperforms existing approaches in both automatic and human evaluations. The code, data, and benchmark will be made publicly available. Homepage: this https URL.
>
---
#### [replaced 186] Multimodal Graph Network Modeling for Human-Object Interaction Detection with PDE Graph Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.12554](https://arxiv.org/pdf/2509.12554)**

> **作者:** Wenxuan Ji; Haichao Shi; Xiao-Yu Zhang
>
> **摘要:** Existing GNN-based Human-Object Interaction (HOI) detection methods rely on simple MLPs to fuse instance features and propagate information. However, this mechanism is largely empirical and lack of targeted information propagation process. To address this problem, we propose Multimodal Graph Network Modeling (MGNM) for HOI detection with Partial Differential Equation (PDE) graph diffusion. Specifically, we first design a multimodal graph network framework that explicitly models the HOI detection task within a four-stage graph structure. Next, we propose a novel PDE diffusion mechanism to facilitate information propagation within this graph. This mechanism leverages multimodal features to propaganda information via a white-box PDE diffusion equation. Furthermore, we design a variational information squeezing (VIS) mechanism to further refine the multimodal features extracted from CLIP, thereby mitigating the impact of noise inherent in pretrained Vision-Language Models. Extensive experiments demonstrate that our MGNM achieves state-of-the-art performance on two widely used benchmarks: HICO-DET and V-COCO. Moreover, when integrated with a more advanced object detector, our method yields significant performance gains while maintaining an effective balance between rare and non-rare categories.
>
---
#### [replaced 187] MALLVI: A Multi-Agent Framework for Integrated Generalized Robotics Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出MALLVI框架，解决机器人操作中的任务规划问题。通过多智能体协作实现闭环反馈，提升动态环境下的操作成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.16898](https://arxiv.org/pdf/2602.16898)**

> **作者:** Mehrshad Taji; Arad Mahdinezhad Kashani; Iman Ahmadi; AmirHossein Jadidi; Saina Kashani; Babak Khalaj
>
> **摘要:** Task planning for robotic manipulation with large language models (LLMs) is an emerging area. Prior approaches rely on specialized models, fine tuning, or prompt tuning, and often operate in an open loop manner without robust environmental feedback, making them fragile in dynamic settings. MALLVI presents a Multi Agent Large Language and Vision framework that enables closed-loop feedback driven robotic manipulation. Given a natural language instruction and an image of the environment, MALLVI generates executable atomic actions for a robot manipulator. After action execution, a Vision Language Model (VLM) evaluates environmental feedback and decides whether to repeat the process or proceed to the next step. Rather than using a single model, MALLVI coordinates specialized agents, Decomposer, Localizer, Thinker, and Reflector, to manage perception, localization, reasoning, and high level planning. An optional Descriptor agent provides visual memory of the initial state. The Reflector supports targeted error detection and recovery by reactivating only relevant agents, avoiding full replanning. Experiments in simulation and real-world settings show that iterative closed loop multi agent coordination improves generalization and increases success rates in zero shot manipulation tasks. Code available at this https URL .
>
---
#### [replaced 188] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10281](https://arxiv.org/pdf/2603.10281)**

> **作者:** Rajesh Shrestha; Xiao Fu
>
> **摘要:** While score-based generative models have emerged as powerful priors for solving inverse problems, directly integrating them into optimization algorithms such as ADMM remains nontrivial. Two central challenges arise: i) the mismatch between the noisy data manifolds used to train the score functions and the geometry of ADMM iterates, especially due to the influence of dual variables, and ii) the lack of convergence understanding when ADMM is equipped with score-based denoisers. To address the manifold mismatch issue, we propose ADMM plug-and-play (ADMM-PnP) with the AC-DC denoiser, a new framework that embeds a three-stage denoiser into ADMM: (1) auto-correction (AC) via additive Gaussian noise, (2) directional correction (DC) using conditional Langevin dynamics, and (3) score-based denoising. In terms of convergence, we establish two results: first, under proper denoiser parameters, each ADMM iteration is a weakly nonexpansive operator, ensuring high-probability fixed-point $\textit{ball convergence}$ using a constant step size; second, under more relaxed conditions, the AC-DC denoiser is a bounded denoiser, which leads to convergence under an adaptive step size schedule. Experiments on a range of inverse problems demonstrate that our method consistently improves solution quality over a variety of baselines.
>
---
#### [replaced 189] Corruption-Aware Training of Latent Video Diffusion Models for Robust Text-to-Video Generation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2505.21545](https://arxiv.org/pdf/2505.21545)**

> **作者:** Chika Maduabuchi; Hao Chen; Yujin Han; Jindong Wang
>
> **备注:** ICLR 2026 ReALM-GEN
>
> **摘要:** Latent Video Diffusion Models (LVDMs) have achieved state-of-the-art generative quality for image and video generation; however, they remain brittle under noisy conditioning, where small perturbations in text or multimodal embeddings can cascade over timesteps and cause semantic drift. Existing corruption strategies from image diffusion (Gaussian, Uniform) fail in video settings because static noise disrupts temporal fidelity. In this paper, we propose CAT-LVDM, a corruption-aware training framework with structured, data-aligned noise injection tailored for video diffusion. Our two operators, Batch-Centered Noise Injection (BCNI) and Spectrum-Aware Contextual Noise (SACN), align perturbations with batch semantics or spectral dynamics to preserve coherence. CAT-LVDM yields substantial gains: BCNI reduces FVD by 31.9 percent on WebVid-2M, MSR-VTT, and MSVD, while SACN improves UCF-101 by 12.3 percent, outperforming Gaussian, Uniform, and even large diffusion baselines like DEMO (2.3B) and Lavie (3B) despite training on 5x less data. Ablations confirm the unique value of low-rank, data-aligned noise, and theory establishes why these operators tighten robustness and generalization bounds. CAT-LVDM thus sets a new framework for robust video diffusion, and our experiments show that it can also be extended to autoregressive generation and multimodal video understanding LLMs. Code, models, and samples are available at this https URL
>
---
#### [replaced 190] Efficient Mixture-of-Expert for Video-based Driver State and Physiological Multi-task Estimation in Conditional Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2410.21086](https://arxiv.org/pdf/2410.21086)**

> **作者:** Jiyao Wang; Xiao Yang; Zhenyu Wang; Ximeng Wei; Ange Wang; Dengbo He; Kaishun Wu
>
> **摘要:** Road safety remains a critical challenge worldwide, with approximately 1.35 million fatalities annually attributed to traffic accidents, often due to human errors. As we advance towards higher levels of vehicle automation, challenges still exist, as driving with automation can cognitively over-demand drivers if they engage in non-driving-related tasks (NDRTs), or lead to drowsiness if driving was the sole task. This calls for the urgent need for an effective Driver Monitoring System (DMS) that can evaluate cognitive load and drowsiness in SAE Level-2/3 autonomous driving contexts. In this study, we propose a novel multi-task DMS, termed VDMoE, which leverages RGB video input to monitor driver states non-invasively. By utilizing key facial features to minimize computational load and integrating remote Photoplethysmography (rPPG) for physiological insights, our approach enhances detection accuracy while maintaining efficiency. Additionally, we optimize the Mixture-of-Experts (MoE) framework to accommodate multi-modal inputs and improve performance across different tasks. A novel prior-inclusive regularization method is introduced to align model outputs with statistical priors, thus accelerating convergence and mitigating overfitting risks. We validate our method with the creation of a new dataset (MCDD), which comprises RGB video and physiological indicators from 42 participants, and two public datasets. Our findings demonstrate the effectiveness of VDMoE in monitoring driver states, contributing to safer autonomous driving systems. The code and data will be released.
>
---
#### [replaced 191] RISE: Single Static Radar-based Indoor Scene Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14019](https://arxiv.org/pdf/2511.14019)**

> **作者:** Kaichen Zhou; Laura Dodds; Sayed Saad Afzal; Fadel Adib
>
> **摘要:** Robust and privacy-preserving indoor scene understanding remains a fundamental open problem. While optical sensors such as RGB and LiDAR offer high spatial fidelity, they suffer from severe occlusions and introduce privacy risks in indoor environments. In contrast, millimeter-wave (mmWave) radar preserves privacy and penetrates obstacles, but its inherently low spatial resolution makes reliable geometric reasoning difficult. We introduce RISE, the first benchmark and system for single-static-radar indoor scene understanding, jointly targeting layout reconstruction and object detection. RISE is built upon the key insight that multipath reflections-traditionally treated as noise-encode rich geometric cues. To exploit this, we propose a Bi-Angular Multipath Enhancement that explicitly models Angle-of-Arrival and Angle-of-Departure to recover secondary (ghost) reflections and reveal invisible structures. On top of these enhanced observations, a simulation-to-reality Hierarchical Diffusion framework transforms fragmented radar responses into complete layout reconstruction and object detection. Our benchmark contains 50,000 frames collected across 100 real indoor trajectories, forming the first large-scale dataset dedicated to single, static, radar-based indoor scene understanding. Extensive experiments show that RISE reduces the Chamfer Distance by 60% (down to 16 cm) compared to the state of the art in mmWave layout reconstruction, and delivers the first mmWave-based object detection, achieving 58% IoU. These results establish RISE as a new foundation for geometry-aware and privacy-preserving indoor scene understanding using a single static radar. Our website and code are available at this https URL.
>
---
#### [replaced 192] AutoRegressive Generation with B-rep Holistic Token Sequence Representation
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.16771](https://arxiv.org/pdf/2601.16771)**

> **作者:** Jiahao Li; Yunpeng Bai; Yongkang Dai; Hao Guo; Hongping Gan; Yilei Shi
>
> **摘要:** Previous representation and generation approaches for the B-rep relied on graph-based representations that disentangle geometric and topological features through decoupled computational pipelines, thereby precluding the application of sequence-based generative frameworks, such as transformer architectures that have demonstrated remarkable performance. In this paper, we propose BrepARG, the first attempt to encode B-rep's geometry and topology into a holistic token sequence representation, enabling sequence-based B-rep generation with an autoregressive architecture. Specifically, BrepARG encodes B-rep into 3 types of tokens: geometry and position tokens representing geometric features, and face index tokens representing topology. Then the holistic token sequence is constructed hierarchically, starting with constructing the geometry blocks (i.e., faces and edges) using the above tokens, followed by geometry block sequencing. Finally, we assemble the holistic sequence representation for the entire B-rep. We also construct a transformer-based autoregressive model that learns the distribution over holistic token sequences via next-token prediction, using a multi-layer decoder-only architecture with causal masking. Experiments demonstrate that BrepARG achieves state-of-the-art (SOTA) performance. BrepARG validates the feasibility of representing B-rep as holistic token sequences, opening new directions for B-rep generation.
>
---
#### [replaced 193] UniGame: Turning a Unified Multimodal Model Into Its Own Adversary
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19413](https://arxiv.org/pdf/2511.19413)**

> **作者:** Zhaolong Su; Wang Lu; Hao Chen; Sharon Li; Jindong Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Unified Multimodal Models (UMMs) have shown impressive performance in both understanding and generation with a single architecture. However, UMMs still exhibit a fundamental inconsistency: understanding favors compact embeddings, whereas generation favors reconstruction-rich representations. This structural trade-off produces misaligned decision boundaries, degraded cross-modal coherence, and heightened vulnerability under distributional and adversarial shifts. In this paper, we present UniGame, a self-adversarial post-training framework that directly targets the inconsistencies. By applying a lightweight perturber at the shared token interface, UniGame enables the generation branch to actively seek and challenge fragile understanding, turning the model itself into its own adversary. Experiments demonstrate that UniGame significantly improves the consistency (+4.6%). Moreover, it also achieves substantial improvements in understanding (+3.6%), generation (+0.02)on GenEval, out-of-distribution and adversarial robustness (+4.8% and +6.2% on NaturalBench and AdVQA). The framework is architecture-agnostic, introduces less than 1% additional parameters, and is complementary to existing post-training methods. These results position adversarial self-play as a general and effective principle for enhancing the coherence, stability, and unified competence of future multimodal foundation models. The official code is available at: this https URL
>
---
#### [replaced 194] Jacobian-aware Posterior Sampling for Inverse Problems
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18471](https://arxiv.org/pdf/2511.18471)**

> **作者:** Liav Hen; Tom Tirer; Raja Giryes; Shady Abu-Hussein
>
> **摘要:** Diffusion models provide powerful generative priors for solving inverse problems by sampling from a posterior distribution conditioned on corrupted measurements. Existing methods primarily follow two paradigms: direct methods, which approximate the likelihood term, and proximal methods, which incorporate intermediate solutions satisfying measurement constraints into the sampling process. We demonstrate that these approaches differ fundamentally in their treatment of the diffusion denoiser's Jacobian within the likelihood term. While this Jacobian encodes critical prior knowledge of the data distribution, training-induced non-idealities can degrade performance in zero-shot settings. In this work, we bridge direct and proximal approaches by proposing a principled Jacobian-Aware Posterior Sampler (JAPS). JAPS leverages the Jacobian's prior knowledge while mitigating its detrimental effects through a corresponding proximal solution, requiring no additional computational cost. Our method enhances reconstruction quality across diverse linear and nonlinear noisy imaging tasks, outperforming existing diffusion-based baselines in perceptual quality while maintaining or improving distortion metrics.
>
---
#### [replaced 195] Self-Corrected Flow Distillation for Consistent One-Step and Few-Step Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.16906](https://arxiv.org/pdf/2412.16906)**

> **作者:** Quan Dao; Hao Phung; Trung Dao; Dimitris Metaxas; Anh Tran
>
> **备注:** Accepted to AAAI 2025. Code: this https URL
>
> **摘要:** Flow matching has emerged as a promising framework for training generative models, demonstrating impressive empirical performance while offering relative ease of training compared to diffusion-based models. However, this method still requires numerous function evaluations in the sampling process. To address these limitations, we introduce a self-corrected flow distillation method that effectively integrates consistency models and adversarial training within the flow-matching framework. This work is a pioneer in achieving consistent generation quality in both few-step and one-step sampling. Our extensive experiments validate the effectiveness of our method, yielding superior results both quantitatively and qualitatively on CelebA-HQ and zero-shot benchmarks on the COCO dataset. Our implementation is released at this https URL.
>
---
#### [replaced 196] AnthroTAP: Learning Point Tracking with Real-World Motion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.06233](https://arxiv.org/pdf/2507.06233)**

> **作者:** Inès Hyeonsu Kim; Seokju Cho; Jahyeok Koo; Junghyun Park; Jiahui Huang; Honglak Lee; Joon-Young Lee; Seungryong Kim
>
> **备注:** CVPR 2026. Project Page: this https URL
>
> **摘要:** Point tracking models often struggle to generalize to real-world videos because large-scale training data is predominantly synthetic$\unicode{x2014}$the only source currently feasible to produce at scale. Collecting real-world annotations, however, is prohibitively expensive, as it requires tracking hundreds of points across frames. We introduce \textbf{AnthroTAP}, an automated pipeline that generates large-scale pseudo-labeled point tracking data from real human motion videos. Leveraging the structured complexity of human movement$\unicode{x2014}$non-rigid deformations, articulated motion, and frequent occlusions$\unicode{x2014}$AnthroTAP fits Skinned Multi-Person Linear (SMPL) models to detected humans, projects mesh vertices onto image planes, resolves occlusions via ray-casting, and filters unreliable tracks using optical flow consistency. A model trained on the AnthroTAP dataset achieves state-of-the-art performance on TAP-Vid, a challenging general-domain benchmark for tracking any point on diverse rigid and non-rigid objects (e.g., humans, animals, robots, and vehicles). Our approach outperforms recent self-training methods trained on vastly larger real datasets, while requiring only one day of training on 4 GPUs. AnthroTAP shows that structured human motion offers a scalable and effective source of real-world supervision for point tracking.
>
---
#### [replaced 197] Monitoring Simulated Physical Weakness Using Detailed Behavioral Features and Personalized Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.10045](https://arxiv.org/pdf/2406.10045)**

> **作者:** Chen Long-fei; Muhammad Ahmed Raza; Craig Innes; Subramanian Ramamoorthy; Robert B. Fisher
>
> **摘要:** Aging and chronic conditions affect older adults' daily lives, making the early detection of developing health issues crucial. Weakness, which is common across many conditions, can subtly alter physical movements and daily activities. However, these behavioral changes can be difficult to detect because they are gradual and often masked by natural day-to-day variability. To isolate the behavioral phenotype of weakness while controlling for confounding factors, this study simulates physical weakness in healthy adults through exercise-induced fatigue, providing interpretable insights into potential behavioral indicators for long-term monitoring. A non-intrusive camera sensor is used to monitor individuals' daily sitting and relaxing activities over multiple days, allowing us to observe behavioral changes before and after simulated weakness. The system captures fine-grained features related to body motion, inactivity, and environmental context in real time while prioritizing privacy. A Bayesian Network models the relationships among activities, contextual factors, and behavioral indicators. Fine-grained features, including non-dominant upper-body motion speed and scale, together with inactivity distribution, are most effective when used with a 300-second window. Personalized models achieve 0.97 accuracy at distinguishing simulated weak days from normal days, and no universal set of optimal features or activities is observed across participants.
>
---
#### [replaced 198] InternVideo-Next: Towards General Video Foundation Models without Video-Text Supervision
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01342](https://arxiv.org/pdf/2512.01342)**

> **作者:** Chenting Wang; Yuhan Zhu; Yicheng Xu; Jiange Yang; Lang Lin; Ziang Yan; Yali Wang; Yi Wang; Limin Wang
>
> **摘要:** Large-scale video-text pretraining achieves strong performance but depends on noisy, synthetic captions with limited semantic coverage, often overlooking implicit world knowledge such as object motion, 3D geometry, and physical cues. In contrast, masked video modeling (MVM) directly exploits spatiotemporal structures but trails text-supervised methods on general tasks. We find this gap arises from overlooked architectural issues: pixel-level reconstruction struggles with convergence and its low-level requirement often conflicts with semantics, while latent prediction often encourages shortcut learning. To address these, we disentangle the traditional encoder-decoder design into an Encoder-Predictor-Decoder (EPD) framework, where the predictor acts as a latent world model, and propose InternVideo-Next, a two-stage pretraining scheme that builds a semantically consistent yet detail-preserving latent space for this world model. First, conventional linear decoder in pixel MVM enforces the predictor output latent to be linearly projected to, thus separable in pixel space, causing the conflict with semantic abstraction. Our Stage 1 proposes a conditional diffusion decoder and injects reliable image-level semantic priors to enhance semantics and convergence, thus bridging pixel-level fidelity with high-level semantic abstraction. Stage 2 further learns world knowledge by predicting frozen Stage 1 targets within this space, mitigating shortcut learning. Trained on public, unlabeled videos, InternVideo-Next achieves state-of-the-art results across benchmarks and provides a scalable path toward general video representation learning.
>
---
#### [replaced 199] MAN++: Scaling Momentum Auxiliary Network for Supervised Local Learning in Vision Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.16279](https://arxiv.org/pdf/2507.16279)**

> **作者:** Junhao Su; Feiyu Zhu; Hengyu Shi; Tianyang Han; Yurui Qiu; Junfeng Luo; Xiaoming Wei; Jialin Gao
>
> **备注:** Accepted by TPAMI
>
> **摘要:** Deep learning typically relies on end-to-end backpropagation for training, a method that inherently suffers from issues such as update locking during parameter optimization, high GPU memory consumption, and a lack of biological plausibility. In contrast, supervised local learning seeks to mitigate these challenges by partitioning the network into multiple local blocks and designing independent auxiliary networks to update each block separately. However, because gradients are propagated solely within individual local blocks, performance degradation occurs, preventing supervised local learning from supplanting end-to-end backpropagation. To address these limitations and facilitate inter-block information flow, we propose the Momentum Auxiliary Network++ (MAN++). MAN++ introduces a dynamic interaction mechanism by employing the Exponential Moving Average (EMA) of parameters from adjacent blocks to enhance communication across the network. The auxiliary network, updated via EMA, effectively bridges the information gap between blocks. Notably, we observed that directly applying EMA parameters can be suboptimal due to feature discrepancies between local blocks. To resolve this issue, we introduce a learnable scaling bias that balances feature differences, thereby further improving performance. We validate MAN++ through extensive experiments on tasks that include image classification, object detection, and image segmentation, utilizing multiple network architectures. The experimental results demonstrate that MAN++ achieves performance comparable to end-to-end training while significantly reducing GPU memory usage. Consequently, MAN++ offers a novel perspective for supervised local learning and presents a viable alternative to conventional training methods.
>
---
#### [replaced 200] Person-Centric Annotations of LAION-400M: Auditing Bias and Its Transfer to Models
- **分类: cs.CV; cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于视觉-语言模型的偏见分析任务，旨在解决数据集偏差对模型影响的问题。通过为LAION-400M添加人物级标注，揭示数据中的种族和性别不平衡，并验证其对模型偏见的影响。**

- **链接: [https://arxiv.org/pdf/2510.03721](https://arxiv.org/pdf/2510.03721)**

> **作者:** Leander Girrbach; Stephan Alaniz; Genevieve Smith; Trevor Darrell; Zeynep Akata
>
> **备注:** ICLR 2026
>
> **摘要:** Vision-language models trained on large-scale multimodal datasets show strong demographic biases, but the role of training data in producing these biases remains unclear. A major barrier has been the lack of demographic annotations in web-scale datasets such as LAION-400M. We address this gap by creating person-centric annotations for the full dataset, including over 276 million bounding boxes, perceived gender and race/ethnicity labels, and automatically generated captions. These annotations are produced through validated automatic labeling pipelines combining object detection, multimodal captioning, and finetuned classifiers. Using them, we uncover demographic imbalances and harmful associations, such as the disproportionate linking of men and individuals perceived as Black or Middle Eastern with crime-related and negative content. We also show that a linear fit predicts 60-70% of gender bias in CLIP and Stable Diffusion from direct co-occurrences in the data. Our resources establish the first large-scale empirical link between dataset composition and downstream model bias. Code is available at this https URL.
>
---
#### [replaced 201] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.14015](https://arxiv.org/pdf/2412.14015)**

> **作者:** Haotong Lin; Sida Peng; Jingxiao Chen; Songyou Peng; Jiaming Sun; Minghuan Liu; Hujun Bao; Jiashi Feng; Xiaowei Zhou; Bingyi Kang
>
> **备注:** CVPR 2025; Project page: this https URL
>
> **摘要:** Prompts play a critical role in unleashing the power of language and vision foundation models for specific tasks. For the first time, we introduce prompting into depth foundation models, creating a new paradigm for metric depth estimation termed Prompt Depth Anything. Specifically, we use a low-cost LiDAR as the prompt to guide the Depth Anything model for accurate metric depth output, achieving up to 4K resolution. Our approach centers on a concise prompt fusion design that integrates the LiDAR at multiple scales within the depth decoder. To address training challenges posed by limited datasets containin both LiDAR depth and precise GT depth, we propose a scalable data pipeline that includes synthetic data LiDAR simulation and real data pseudo GT depth generation. Our approach sets new state-of-the-arts on the ARKitScenes and ScanNet++ datasets and benefits downstream applications, including 3D reconstruction and generalized robotic grasping.
>
---
#### [replaced 202] A Hyperbolic Perspective on Hierarchical Structure in Object-Centric Scene Representations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14022](https://arxiv.org/pdf/2603.14022)**

> **作者:** Neelu Madan; Àlex Pujol; Andreas Møgelmose; Sergio Escalera; Kamal Nasrollahi; Graham W. Taylor; Thomas B. Moeslund
>
> **备注:** accepted at CVPR Workshops 2026
>
> **摘要:** Slot attention has emerged as a powerful framework for unsupervised object-centric learning, decomposing visual scenes into a small set of compact vector representations called \emph{slots}, each capturing a distinct region or object. However, these slots are learned in Euclidean space, which provides no geometric inductive bias for the hierarchical relationships that naturally structure visual scenes. In this work, we propose a simple post-hoc pipeline to project Euclidean slot embeddings onto the Lorentz hyperboloid of hyperbolic space, without modifying the underlying training pipeline. We construct five-level visual hierarchies directly from slot attention masks and analyse whether hyperbolic geometry reveals latent hierarchical structure that remains invisible in Euclidean space. Integrating our pipeline with SPOT (images), VideoSAUR (video), and SlotContrast (video), We find that hyperbolic projection exposes a consistent scene-level to object-level organisation, where coarse slots occupy greater manifold depth than fine slots, which is absent in Euclidean space. We further identify a "curvature--task tradeoff": low curvature ($c{=}0.2$) matches or outperforms Euclidean on parent slot retrieval, while moderate curvature ($c{=}0.5$) achieves better inter-level separation. Together, these findings suggest that slot representations already encode latent hierarchy that hyperbolic geometry reveals, motivating end-to-end hyperbolic training as a natural next step. Code and models are available at \href{this https URL}{this http URL}.
>
---
#### [replaced 203] P$^2$HCT: Plug-and-Play Hierarchical C2F Transformer for Multi-Scale Feature Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.12772](https://arxiv.org/pdf/2505.12772)**

> **作者:** Junyi Hu; Tian Bai; Fengyi Wu; Zhenming Peng; Yi Zhang
>
> **备注:** 12 pages, 6 figures, ICME2026
>
> **摘要:** Feature fusion plays a pivotal role in achieving high performance in vision models, yet existing attention-based fusion techniques often suffer from substantial computational overhead and implementation complexity, particularly in resource-constrained settings. To address these limitations, we introduce the Plug-and-Play Hierarchical C2F Transformer (P$^2$HCT), a lightweight module that combines coarse-to-fine token selection with shared attention parameters to preserve spatial details while reducing inference cost. P$^2$HCT is trainable using coarse attention alone and can be seamlessly activated at inference to enhance accuracy without retraining. Integrated into real-time detectors such as YOLOv11-N/S/M, P$^2$HCT achieves mAP gains of 0.9\%, 0.5\%, and 0.4\% on MS COCO with minimal latency increase. Similarly, embedding P$^2$HCT into ResNet-18/50/101 backbones improves ImageNet top-1 accuracy by 6.5\%, 1.7\%, and 1.0\%, respectively. These results underscore P$^2$HCT's effectiveness as a hardware-friendly and general-purpose enhancement for both detection and classification tasks.
>
---
#### [replaced 204] VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.05138](https://arxiv.org/pdf/2601.05138)**

> **作者:** Sixiao Zheng; Minghao Yin; Wenbo Hu; Xiaoyu Li; Ying Shan; Yanwei Fu
>
> **备注:** Project Page: this https URL, Accepted by CVPR 2026
>
> **摘要:** Video world models aim to simulate dynamic, real-world environments, yet existing methods struggle to provide unified and precise control over camera and multi-object motion, as videos inherently capture dynamics in the projected 2D image plane. To bridge this gap, we introduce VerseCrafter, a geometry-driven video world model that generates dynamic, realistic videos from a unified 4D geometric world state. Our approach is centered on a novel 4D Geometric Control representation, which encodes the world state as a static background point cloud and per-object 3D Gaussian trajectories. This representation captures each object's motion path and probabilistic 3D occupancy over time, providing a flexible, category-agnostic alternative to rigid bounding boxes and parametric models. We render 4D Geometric Control into 4D control maps for a pretrained video diffusion model, enabling high-fidelity, view-consistent video generation that faithfully follows the specified dynamics. To enable training at scale, we develop an automatic data engine and construct VerseControl4D, a real-world dataset of 35K training samples with automatically derived prompts and rendered 4D control maps. Extensive experiments show that VerseCrafter achieves superior visual quality and more accurate control over camera and multi-object motion than prior methods.
>
---
#### [replaced 205] From Unlearning to UNBRANDING: A Benchmark for Trademark-Safe Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13953](https://arxiv.org/pdf/2512.13953)**

> **作者:** Dawid Malarz; Filip Manjak; Maciej Zięba; Przemysław Spurek; Artur Kasymov
>
> **摘要:** The rapid progress of text-to-image diffusion models raises significant concerns regarding the unauthorized reproduction of trademarked content. While prior work targets general concepts (e.g., styles, celebrities), it fails to address specific brand identifiers. Brand recognition is multi-dimensional, extending beyond explicit logos to encompass distinctive structural features (e.g., a car's front grille). To tackle this, we introduce unbranding, a novel task for the fine-grained removal of both trademarks and subtle structural brand features, while preserving semantic coherence. We construct a benchmark dataset and introduce a novel evaluation framework combining Vision Language Models (VLMs) with segmentation-based classifiers trained on human annotations of logos and trade dress features, addressing the limitations of existing brand detectors that fail to capture abstract trade dress. Furthermore, we observe that newer, higher-fidelity systems (SDXL, FLUX) synthesize brand identifiers more readily than older models, highlighting the urgency of this challenge. Our results confirm that unbranding is a distinct problem requiring specialized techniques. Project Page: this https URL.
>
---
#### [replaced 206] FastVMT: Eliminating Redundancy in Video Motion Transfer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05551](https://arxiv.org/pdf/2602.05551)**

> **作者:** Yue Ma; Zhikai Wang; Tianhao Ren; Mingzhe Zheng; Hongyu Liu; Jiayi Guo; Kunyu Feng; Yuxuan Xue; Zixiang Zhao; Konrad Schindler; Qifeng Chen; Linfeng Zhang
>
> **备注:** Accepted by ICLR2026, Project page: this http URL, Code: this https URL
>
> **摘要:** Video motion transfer aims to synthesize videos by generating visual content according to a text prompt while transferring the motion pattern observed in a reference video. Recent methods predominantly use the Diffusion Transformer (DiT) architecture. To achieve satisfactory runtime, several methods attempt to accelerate the computations in the DiT, but fail to address structural sources of inefficiency. In this work, we identify and remove two types of computational redundancy in earlier work: motion redundancy arises because the generic DiT architecture does not reflect the fact that frame-to-frame motion is small and smooth; gradient redundancy occurs if one ignores that gradients change slowly along the diffusion trajectory. To mitigate motion redundancy, we mask the corresponding attention layers to a local neighborhood such that interaction weights are not computed unnecessarily distant image regions. To exploit gradient redundancy, we design an optimization scheme that reuses gradients from previous diffusion steps and skips unwarranted gradient computations. On average, FastVMT achieves a 3.43x speedup without degrading the visual fidelity or the temporal consistency of the generated videos.
>
---
