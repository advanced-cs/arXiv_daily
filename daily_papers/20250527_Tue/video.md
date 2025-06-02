# 计算机视觉 cs.CV

- **最新发布 318 篇**

- **更新 197 篇**

## 最新发布

#### [new 001] HyperFake: Hyperspectral Reconstruction and Attention-Guided Analysis for Advanced Deepfake Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于深度伪造检测任务，旨在解决现有方法跨技术/数据集泛化能力差及RGB数据局限性问题。提出HyperFake pipeline：通过改进MST++重建31通道超光谱视频，结合光谱注意力机制提取关键特征，再以EfficientNet分类器提升检测精度，实现无需超光谱相机的通用化深伪检测。**

- **链接: [http://arxiv.org/pdf/2505.18587v1](http://arxiv.org/pdf/2505.18587v1)**

> **作者:** Pavan C Shekar; Pawan Soni; Vivek Kanhangad
>
> **备注:** 6 pages, 3 figures, 1 table. Preliminary results on FaceForensics++ dataset. First approach to use hyperspectral reconstruction for deepfake detection
>
> **摘要:** Deepfakes pose a significant threat to digital media security, with current detection methods struggling to generalize across different manipulation techniques and datasets. While recent approaches combine CNN-based architectures with Vision Transformers or leverage multi-modal learning, they remain limited by the inherent constraints of RGB data. We introduce HyperFake, a novel deepfake detection pipeline that reconstructs 31-channel hyperspectral data from standard RGB videos, revealing hidden manipulation traces invisible to conventional methods. Using an improved MST++ architecture, HyperFake enhances hyperspectral reconstruction, while a spectral attention mechanism selects the most critical spectral features for deepfake detection. The refined spectral data is then processed by an EfficientNet-based classifier optimized for spectral analysis, enabling more accurate and generalizable detection across different deepfake styles and datasets, all without the need for expensive hyperspectral cameras. To the best of our knowledge, this is the first approach to leverage hyperspectral imaging reconstruction for deepfake detection, opening new possibilities for detecting increasingly sophisticated manipulations.
>
---
#### [new 002] Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought
- **分类: cs.CV**

- **简介: 该论文提出视频异常推理（VAR）任务，针对现有VAD方法缺乏深度推理的缺陷，设计Vad-R1框架，包含感知到认知的思维链（P2C-CoT）、专用数据集Vad-Reasoning及改进的强化学习算法AVA-GRPO，通过结构化推理与自验证机制提升异常分析能力，实验显示优于现有模型。**

- **链接: [http://arxiv.org/pdf/2505.19877v1](http://arxiv.org/pdf/2505.19877v1)**

> **作者:** Chao Huang; Benfeng Wang; Jie Wen; Chengliang Liu; Wei Wang; Li Shen; Xiaochun Cao
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Recent advancements in reasoning capability of Multimodal Large Language Models (MLLMs) demonstrate its effectiveness in tackling complex visual tasks. However, existing MLLM-based Video Anomaly Detection (VAD) methods remain limited to shallow anomaly descriptions without deep reasoning. In this paper, we propose a new task named Video Anomaly Reasoning (VAR), which aims to enable deep analysis and understanding of anomalies in the video by requiring MLLMs to think explicitly before answering. To this end, we propose Vad-R1, an end-to-end MLLM-based framework for VAR. Specifically, we design a Perception-to-Cognition Chain-of-Thought (P2C-CoT) that simulates the human process of recognizing anomalies, guiding the MLLM to reason anomaly step-by-step. Based on the structured P2C-CoT, we construct Vad-Reasoning, a dedicated dataset for VAR. Furthermore, we propose an improved reinforcement learning algorithm AVA-GRPO, which explicitly incentivizes the anomaly reasoning capability of MLLMs through a self-verification mechanism with limited annotations. Experimental results demonstrate that Vad-R1 achieves superior performance, outperforming both open-source and proprietary models on VAD and VAR tasks. Codes and datasets will be released at https://github.com/wbfwonderful/Vad-R1.
>
---
#### [new 003] ReasonPlan: Unified Scene Prediction and Decision Reasoning for Closed-loop Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO; 68T40(Primary), 68T45, 68T50(Secondary); I.2.9; I.2.10; I.5.1**

- **简介: 该论文属于自动驾驶闭环规划任务，针对多模态大模型在闭环系统中效果不佳及决策不透明的问题，提出ReasonPlan框架：通过自监督场景预测与监督决策思维链的双机制，提升模型视觉-驾驶语境对齐与可解释决策能力，并构建规划导向数据集PDR。实验显示其显著优于传统模仿学习方法，且具备零样本泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.20024v1](http://arxiv.org/pdf/2505.20024v1)**

> **作者:** Xueyi Liu; Zuodong Zhong; Yuxin Guo; Yun-Fu Liu; Zhiguo Su; Qichao Zhang; Junli Wang; Yinfeng Gao; Yupeng Zheng; Qiao Lin; Huiyong Chen; Dongbin Zhao
>
> **备注:** 18 pages; 9 figures; https://github.com/Liuxueyi/ReasonPlan
>
> **摘要:** Due to the powerful vision-language reasoning and generalization abilities, multimodal large language models (MLLMs) have garnered significant attention in the field of end-to-end (E2E) autonomous driving. However, their application to closed-loop systems remains underexplored, and current MLLM-based methods have not shown clear superiority to mainstream E2E imitation learning approaches. In this work, we propose ReasonPlan, a novel MLLM fine-tuning framework designed for closed-loop driving through holistic reasoning with a self-supervised Next Scene Prediction task and supervised Decision Chain-of-Thought process. This dual mechanism encourages the model to align visual representations with actionable driving context, while promoting interpretable and causally grounded decision making. We curate a planning-oriented decision reasoning dataset, namely PDR, comprising 210k diverse and high-quality samples. Our method outperforms the mainstream E2E imitation learning method by a large margin of 19% L2 and 16.1 driving score on Bench2Drive benchmark. Furthermore, ReasonPlan demonstrates strong zero-shot generalization on unseen DOS benchmark, highlighting its adaptability in handling zero-shot corner cases. Code and dataset will be found in https://github.com/Liuxueyi/ReasonPlan.
>
---
#### [new 004] FusionTrack: End-to-End Multi-Object Tracking in Arbitrary Multi-View Environment
- **分类: cs.CV**

- **简介: 该论文属于多视角多目标跟踪（MVMOT）任务，旨在解决现有系统缺乏灵活自由视角跟踪的问题。工作包括构建首个无人机多目标跟踪数据集MDMOT，并提出FusionTrack框架，通过端到端整合跟踪与重识别，利用多视角信息提升轨迹关联 robust性，实验显示其性能达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.18727v1](http://arxiv.org/pdf/2505.18727v1)**

> **作者:** Xiaohe Li; Pengfei Li; Zide Fan; Ying Geng; Fangli Mou; Haohua Wu; Yunping Ge
>
> **摘要:** Multi-view multi-object tracking (MVMOT) has found widespread applications in intelligent transportation, surveillance systems, and urban management. However, existing studies rarely address genuinely free-viewpoint MVMOT systems, which could significantly enhance the flexibility and scalability of cooperative tracking systems. To bridge this gap, we first construct the Multi-Drone Multi-Object Tracking (MDMOT) dataset, captured by mobile drone swarms across diverse real-world scenarios, initially establishing the first benchmark for multi-object tracking in arbitrary multi-view environment. Building upon this foundation, we propose \textbf{FusionTrack}, an end-to-end framework that reasonably integrates tracking and re-identification to leverage multi-view information for robust trajectory association. Extensive experiments on our MDMOT and other benchmark datasets demonstrate that FusionTrack achieves state-of-the-art performance in both single-view and multi-view tracking.
>
---
#### [new 005] An Interpretable Representation Learning Approach for Diffusion Tensor Imaging
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于扩散张量成像(DTI)的可解释表示学习任务，旨在解决DTI数据在深度学习中的有效表征与可解释性问题。提出将 tract-level 分数各向异性(FA)值编码为9×9灰度图，结合Beta-TC VAE与空间广播解码器学习解纠缠潜在表征，通过多任务学习和对比学习验证，较1D/3D基线模型在性别分类任务中F1值提升15.74%，且表征解纠缠性更优。**

- **链接: [http://arxiv.org/pdf/2505.19110v1](http://arxiv.org/pdf/2505.19110v1)**

> **作者:** Vishwa Mohan Singh; Alberto Gaston Villagran Asiares; Luisa Sophie Schuhmacher; Kate Rendall; Simon Weißbrod; David Rügamer; Inga Körte
>
> **备注:** Accepted for publication at MIDL 2025
>
> **摘要:** Diffusion Tensor Imaging (DTI) tractography offers detailed insights into the structural connectivity of the brain, but presents challenges in effective representation and interpretation in deep learning models. In this work, we propose a novel 2D representation of DTI tractography that encodes tract-level fractional anisotropy (FA) values into a 9x9 grayscale image. This representation is processed through a Beta-Total Correlation Variational Autoencoder with a Spatial Broadcast Decoder to learn a disentangled and interpretable latent embedding. We evaluate the quality of this embedding using supervised and unsupervised representation learning strategies, including auxiliary classification, triplet loss, and SimCLR-based contrastive learning. Compared to the 1D Group deep neural network (DNN) baselines, our approach improves the F1 score in a downstream sex classification task by 15.74% and shows a better disentanglement than the 3D representation.
>
---
#### [new 006] VL-SAM-V2: Open-World Object Detection with General and Specific Query Fusion
- **分类: cs.CV**

- **简介: 该论文属于开放世界物体检测任务，旨在解决现有模型在开放环境下依赖人工输入或性能不足的问题。提出VL-SAM-V2框架，融合开放集与开放端模型的查询，设计交互模块、排序可学习查询及去噪训练策略，提升检测性能，尤其在罕见物体上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.18986v1](http://arxiv.org/pdf/2505.18986v1)**

> **作者:** Zhiwei Lin; Yongtao Wang
>
> **摘要:** Current perception models have achieved remarkable success by leveraging large-scale labeled datasets, but still face challenges in open-world environments with novel objects. To address this limitation, researchers introduce open-set perception models to detect or segment arbitrary test-time user-input categories. However, open-set models rely on human involvement to provide predefined object categories as input during inference. More recently, researchers have framed a more realistic and challenging task known as open-ended perception that aims to discover unseen objects without requiring any category-level input from humans at inference time. Nevertheless, open-ended models suffer from low performance compared to open-set models. In this paper, we present VL-SAM-V2, an open-world object detection framework that is capable of discovering unseen objects while achieving favorable performance. To achieve this, we combine queries from open-set and open-ended models and propose a general and specific query fusion module to allow different queries to interact. By adjusting queries from open-set models, we enable VL-SAM-V2 to be evaluated in the open-set or open-ended mode. In addition, to learn more diverse queries, we introduce ranked learnable queries to match queries with proposals from open-ended models by sorting. Moreover, we design a denoising point training strategy to facilitate the training process. Experimental results on LVIS show that our method surpasses the previous open-set and open-ended methods, especially on rare objects.
>
---
#### [new 007] SaSi: A Self-augmented and Self-interpreted Deep Learning Approach for Few-shot Cryo-ET Particle Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出SaSi方法，用于冷冻电镜（cryo-ET）的少样本3D颗粒检测。针对标注数据稀缺、低信噪比及缺失楔形伪影导致的定位难题，其通过自增强技术提升数据利用率，结合自解释分割降低标注依赖，实验显示优于现有方法，为结构生物学少样本学习设新基准。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19948v1](http://arxiv.org/pdf/2505.19948v1)**

> **作者:** Gokul Adethya; Bhanu Pratyush Mantha; Tianyang Wang; Xingjian Li; Min Xu
>
> **摘要:** Cryo-electron tomography (cryo-ET) has emerged as a powerful technique for imaging macromolecular complexes in their near-native states. However, the localization of 3D particles in cellular environments still presents a significant challenge due to low signal-to-noise ratios and missing wedge artifacts. Deep learning approaches have shown great potential, but they need huge amounts of data, which can be a challenge in cryo-ET scenarios where labeled data is often scarce. In this paper, we propose a novel Self-augmented and Self-interpreted (SaSi) deep learning approach towards few-shot particle detection in 3D cryo-ET images. Our method builds upon self-augmentation techniques to further boost data utilization and introduces a self-interpreted segmentation strategy for alleviating dependency on labeled data, hence improving generalization and robustness. As demonstrated by experiments conducted on both simulated and real-world cryo-ET datasets, the SaSi approach significantly outperforms existing state-of-the-art methods for particle localization. This research increases understanding of how to detect particles with very few labels in cryo-ET and thus sets a new benchmark for few-shot learning in structural biology.
>
---
#### [new 008] SD-OVON: A Semantics-aware Dataset and Benchmark Generation Pipeline for Open-Vocabulary Object Navigation in Dynamic Scenes
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SD-OVON流水线，针对开放词汇物体导航在动态场景中的训练与评估任务，解决现有数据集局限于静态环境的问题。通过多模态模型生成符合现实语义的无限场景变体，并构建含动态元素的SD-OVON-3k/10k数据集，提供基准与工具，促进机器人跨现实-模拟应用。**

- **链接: [http://arxiv.org/pdf/2505.18881v1](http://arxiv.org/pdf/2505.18881v1)**

> **作者:** Dicong Qiu; Jiadi You; Zeying Gong; Ronghe Qiu; Hui Xiong; Junwei Liang
>
> **备注:** Preprint. 21 pages
>
> **摘要:** We present the Semantics-aware Dataset and Benchmark Generation Pipeline for Open-vocabulary Object Navigation in Dynamic Scenes (SD-OVON). It utilizes pretraining multimodal foundation models to generate infinite unique photo-realistic scene variants that adhere to real-world semantics and daily commonsense for the training and the evaluation of navigation agents, accompanied with a plugin for generating object navigation task episodes compatible to the Habitat simulator. In addition, we offer two pre-generated object navigation task datasets, SD-OVON-3k and SD-OVON-10k, comprising respectively about 3k and 10k episodes of the open-vocabulary object navigation task, derived from the SD-OVON-Scenes dataset with 2.5k photo-realistic scans of real-world environments and the SD-OVON-Objects dataset with 0.9k manually inspected scanned and artist-created manipulatable object models. Unlike prior datasets limited to static environments, SD-OVON covers dynamic scenes and manipulatable objects, facilitating both real-to-sim and sim-to-real robotic applications. This approach enhances the realism of navigation tasks, the training and the evaluation of open-vocabulary object navigation agents in complex settings. To demonstrate the effectiveness of our pipeline and datasets, we propose two baselines and evaluate them along with state-of-the-art baselines on SD-OVON-3k. The datasets, benchmark and source code are publicly available.
>
---
#### [new 009] Modeling Beyond MOS: Quality Assessment Models Must Integrate Context, Reasoning, and Multimodality
- **分类: cs.CV; cs.MM; eess.IV**

- **简介: 该论文属多媒体质量评估任务，指出传统MOS评分忽视上下文、推理和多模态信息的局限性，提出模型需整合上下文感知、可解释推理及多模态融合能力，并建议构建含语境元数据的新数据集及评估指标，推动更鲁棒、人性化的评估系统。**

- **链接: [http://arxiv.org/pdf/2505.19696v1](http://arxiv.org/pdf/2505.19696v1)**

> **作者:** Mohamed Amine Kerkouri; Marouane Tliba; Aladine Chetouani; Nour Aburaed; Alessandro Bruno
>
> **备注:** Under review
>
> **摘要:** This position paper argues that Mean Opinion Score (MOS), while historically foundational, is no longer sufficient as the sole supervisory signal for multimedia quality assessment models. MOS reduces rich, context-sensitive human judgments to a single scalar, obscuring semantic failures, user intent, and the rationale behind quality decisions. We contend that modern quality assessment models must integrate three interdependent capabilities: (1) context-awareness, to adapt evaluations to task-specific goals and viewing conditions; (2) reasoning, to produce interpretable, evidence-grounded justifications for quality judgments; and (3) multimodality, to align perceptual and semantic cues using vision-language models. We critique the limitations of current MOS-centric benchmarks and propose a roadmap for reform: richer datasets with contextual metadata and expert rationales, and new evaluation metrics that assess semantic alignment, reasoning fidelity, and contextual sensitivity. By reframing quality assessment as a contextual, explainable, and multimodal modeling task, we aim to catalyze a shift toward more robust, human-aligned, and trustworthy evaluation systems.
>
---
#### [new 010] Aggregated Structural Representation with Large Language Models for Human-Centric Layout Generation
- **分类: cs.CV**

- **简介: 该论文提出基于ASR模块的布局生成方法，解决现有技术结构信息丢失与生成能力不足的问题。通过融合图网络与LLM，用图特征替代ViT模块，实现结构保留、可编辑的布局生成，在RICO数据集验证效果优异。**

- **链接: [http://arxiv.org/pdf/2505.19554v1](http://arxiv.org/pdf/2505.19554v1)**

> **作者:** Jiongchao Jin; Shengchu Zhao; Dajun Chen; Wei Jiang; Yong Li
>
> **摘要:** Time consumption and the complexity of manual layout design make automated layout generation a critical task, especially for multiple applications across different mobile devices. Existing graph-based layout generation approaches suffer from limited generative capability, often resulting in unreasonable and incompatible outputs. Meanwhile, vision based generative models tend to overlook the original structural information, leading to component intersections and overlaps. To address these challenges, we propose an Aggregation Structural Representation (ASR) module that integrates graph networks with large language models (LLMs) to preserve structural information while enhancing generative capability. This novel pipeline utilizes graph features as hierarchical prior knowledge, replacing the traditional Vision Transformer (ViT) module in multimodal large language models (MLLM) to predict full layout information for the first time. Moreover, the intermediate graph matrix used as input for the LLM is human editable, enabling progressive, human centric design generation. A comprehensive evaluation on the RICO dataset demonstrates the strong performance of ASR, both quantitatively using mean Intersection over Union (mIoU), and qualitatively through a crowdsourced user study. Additionally, sampling on relational features ensures diverse layout generation, further enhancing the adaptability and creativity of the proposed approach.
>
---
#### [new 011] ChartSketcher: Reasoning with Multimodal Feedback and Reflection for Chart Understanding
- **分类: cs.CV**

- **简介: 该论文属于图表理解任务，旨在解决现有模型因缺乏多模态交互导致的视觉推理错误无法修正问题。提出ChartSketcher方法，通过Sketch-CoT让模型在图表上标注中间推理步骤并迭代反馈，结合两阶段训练策略提升性能，实现更准确、可解释的图表分析。**

- **链接: [http://arxiv.org/pdf/2505.19076v1](http://arxiv.org/pdf/2505.19076v1)**

> **作者:** Muye Huang; Lingling Zhang; Jie Ma; Han Lai; Fangzhi Xu; Yifei Li; Wenjun Wu; Yaqiang Wu; Jun Liu
>
> **备注:** 23 pages, 9 figures
>
> **摘要:** Charts are high-density visualization carriers for complex data, serving as a crucial medium for information extraction and analysis. Automated chart understanding poses significant challenges to existing multimodal large language models (MLLMs) due to the need for precise and complex visual reasoning. Current step-by-step reasoning models primarily focus on text-based logical reasoning for chart understanding. However, they struggle to refine or correct their reasoning when errors stem from flawed visual understanding, as they lack the ability to leverage multimodal interaction for deeper comprehension. Inspired by human cognitive behavior, we propose ChartSketcher, a multimodal feedback-driven step-by-step reasoning method designed to address these limitations. ChartSketcher is a chart understanding model that employs Sketch-CoT, enabling MLLMs to annotate intermediate reasoning steps directly onto charts using a programmatic sketching library, iteratively feeding these visual annotations back into the reasoning process. This mechanism enables the model to visually ground its reasoning and refine its understanding over multiple steps. We employ a two-stage training strategy: a cold start phase to learn sketch-based reasoning patterns, followed by off-policy reinforcement learning to enhance reflection and generalization. Experiments demonstrate that ChartSketcher achieves promising performance on chart understanding benchmarks and general vision tasks, providing an interactive and interpretable approach to chart comprehension.
>
---
#### [new 012] CSTrack: Enhancing RGB-X Tracking via Compact Spatiotemporal Features
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CSTrack，针对RGB-X目标跟踪中现有方法计算复杂、跨模态与时空建模不足的问题，通过空间紧凑模块整合RGB与X模态特征，时序模块优化热图表示，实现高效跟踪，达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.19434v1](http://arxiv.org/pdf/2505.19434v1)**

> **作者:** X. Feng; D. Zhang; S. Hu; X. Li; M. Wu; J. Zhang; X. Chen; K. Huang
>
> **备注:** Accepted by ICML25!
>
> **摘要:** Effectively modeling and utilizing spatiotemporal features from RGB and other modalities (\eg, depth, thermal, and event data, denoted as X) is the core of RGB-X tracker design. Existing methods often employ two parallel branches to separately process the RGB and X input streams, requiring the model to simultaneously handle two dispersed feature spaces, which complicates both the model structure and computation process. More critically, intra-modality spatial modeling within each dispersed space incurs substantial computational overhead, limiting resources for inter-modality spatial modeling and temporal modeling. To address this, we propose a novel tracker, CSTrack, which focuses on modeling Compact Spatiotemporal features to achieve simple yet effective tracking. Specifically, we first introduce an innovative Spatial Compact Module that integrates the RGB-X dual input streams into a compact spatial feature, enabling thorough intra- and inter-modality spatial modeling. Additionally, we design an efficient Temporal Compact Module that compactly represents temporal features by constructing the refined target distribution heatmap. Extensive experiments validate the effectiveness of our compact spatiotemporal modeling method, with CSTrack achieving new SOTA results on mainstream RGB-X benchmarks. The code and models will be released at: https://github.com/XiaokunFeng/CSTrack.
>
---
#### [new 013] PathBench: A comprehensive comparison benchmark for pathology foundation models towards precision oncology
- **分类: cs.CV**

- **简介: 该论文提出PathBench基准，解决病理基础模型（PFMs）临床转化中的挑战：模型跨癌症类型表现差异、数据泄露风险及缺乏标准化评估。通过多中心数据（10家医院15,888张切片）、全临床任务覆盖（64项诊断/预后）、严格防泄漏设计，评估19个PFMs，为模型优化与临床应用提供依据。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20202v1](http://arxiv.org/pdf/2505.20202v1)**

> **作者:** Jiabo Ma; Yingxue Xu; Fengtao Zhou; Yihui Wang; Cheng Jin; Zhengrui Guo; Jianfeng Wu; On Ki Tang; Huajun Zhou; Xi Wang; Luyang Luo; Zhengyu Zhang; Du Cai; Zizhao Gao; Wei Wang; Yueping Liu; Jiankun He; Jing Cui; Zhenhui Li; Jing Zhang; Feng Gao; Xiuming Zhang; Li Liang; Ronald Cheong Kin Chan; Zhe Wang; Hao Chen
>
> **备注:** 35 pages, 9 figures
>
> **摘要:** The emergence of pathology foundation models has revolutionized computational histopathology, enabling highly accurate, generalized whole-slide image analysis for improved cancer diagnosis, and prognosis assessment. While these models show remarkable potential across cancer diagnostics and prognostics, their clinical translation faces critical challenges including variability in optimal model across cancer types, potential data leakage in evaluation, and lack of standardized benchmarks. Without rigorous, unbiased evaluation, even the most advanced PFMs risk remaining confined to research settings, delaying their life-saving applications. Existing benchmarking efforts remain limited by narrow cancer-type focus, potential pretraining data overlaps, or incomplete task coverage. We present PathBench, the first comprehensive benchmark addressing these gaps through: multi-center in-hourse datasets spanning common cancers with rigorous leakage prevention, evaluation across the full clinical spectrum from diagnosis to prognosis, and an automated leaderboard system for continuous model assessment. Our framework incorporates large-scale data, enabling objective comparison of PFMs while reflecting real-world clinical complexity. All evaluation data comes from private medical providers, with strict exclusion of any pretraining usage to avoid data leakage risks. We have collected 15,888 WSIs from 8,549 patients across 10 hospitals, encompassing over 64 diagnosis and prognosis tasks. Currently, our evaluation of 19 PFMs shows that Virchow2 and H-Optimus-1 are the most effective models overall. This work provides researchers with a robust platform for model development and offers clinicians actionable insights into PFM performance across diverse clinical scenarios, ultimately accelerating the translation of these transformative technologies into routine pathology practice.
>
---
#### [new 014] Alchemist: Turning Public Text-to-Image Data into Generative Gold
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有监督微调（SFT）数据集领域狭窄、成本高且难以获取高质量通用样本的问题。提出利用预训练模型评估高价值样本的新方法，构建了紧凑高效的通用SFT数据集Alchemist（3,350样本），显著提升五种公开模型的生成质量并保持多样性，同时公开数据集和微调模型。**

- **链接: [http://arxiv.org/pdf/2505.19297v1](http://arxiv.org/pdf/2505.19297v1)**

> **作者:** Valerii Startsev; Alexander Ustyuzhanin; Alexey Kirillov; Dmitry Baranchuk; Sergey Kastryulin
>
> **摘要:** Pre-training equips text-to-image (T2I) models with broad world knowledge, but this alone is often insufficient to achieve high aesthetic quality and alignment. Consequently, supervised fine-tuning (SFT) is crucial for further refinement. However, its effectiveness highly depends on the quality of the fine-tuning dataset. Existing public SFT datasets frequently target narrow domains (e.g., anime or specific art styles), and the creation of high-quality, general-purpose SFT datasets remains a significant challenge. Current curation methods are often costly and struggle to identify truly impactful samples. This challenge is further complicated by the scarcity of public general-purpose datasets, as leading models often rely on large, proprietary, and poorly documented internal data, hindering broader research progress. This paper introduces a novel methodology for creating general-purpose SFT datasets by leveraging a pre-trained generative model as an estimator of high-impact training samples. We apply this methodology to construct and release Alchemist, a compact (3,350 samples) yet highly effective SFT dataset. Experiments demonstrate that Alchemist substantially improves the generative quality of five public T2I models while preserving diversity and style. Additionally, we release the fine-tuned models' weights to the public.
>
---
#### [new 015] Sparse2DGS: Sparse-View Surface Reconstruction using 2D Gaussian Splatting with Dense Point Cloud
- **分类: cs.CV**

- **简介: 该论文提出Sparse2DGS方法，解决仅用三张图像进行3D重建精度不足的问题。传统Gaussian Splatting因稀疏点云初始化效果差，该方法结合DUSt3R与COLMAP生成稠密点云优化初始化，提升稀疏视角下的重建精度。**

- **链接: [http://arxiv.org/pdf/2505.19854v1](http://arxiv.org/pdf/2505.19854v1)**

> **作者:** Natsuki Takama; Shintaro Ito; Koichi Ito; Hwann-Tzong Chen; Takafumi Aoki
>
> **备注:** Accepted to ICIP 2025
>
> **摘要:** Gaussian Splatting (GS) has gained attention as a fast and effective method for novel view synthesis. It has also been applied to 3D reconstruction using multi-view images and can achieve fast and accurate 3D reconstruction. However, GS assumes that the input contains a large number of multi-view images, and therefore, the reconstruction accuracy significantly decreases when only a limited number of input images are available. One of the main reasons is the insufficient number of 3D points in the sparse point cloud obtained through Structure from Motion (SfM), which results in a poor initialization for optimizing the Gaussian primitives. We propose a new 3D reconstruction method, called Sparse2DGS, to enhance 2DGS in reconstructing objects using only three images. Sparse2DGS employs DUSt3R, a fundamental model for stereo images, along with COLMAP MVS to generate highly accurate and dense 3D point clouds, which are then used to initialize 2D Gaussians. Through experiments on the DTU dataset, we show that Sparse2DGS can accurately reconstruct the 3D shapes of objects using just three images.
>
---
#### [new 016] Can Visual Encoder Learn to See Arrows?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型（VLM）改进任务，旨在解决VLM因依赖文本和位置偏见而难以识别图表中边的问题。研究构建无偏图表数据集，通过对比学习训练图像编码器，并在探针、检索和captioning任务中验证，结果优于CLIP及GPT-4o等模型，证明消除偏见可提升边识别能力。**

- **链接: [http://arxiv.org/pdf/2505.19944v1](http://arxiv.org/pdf/2505.19944v1)**

> **作者:** Naoyuki Terashita; Yusuke Tozaki; Hideaki Omote; Congkha Nguyen; Ryosuke Nakamoto; Yuta Koreeda; Hiroaki Ozaki
>
> **备注:** This work has been accepted for poster presentation at the Second Workshop on Visual Concepts in CVPR 2025
>
> **摘要:** The diagram is a visual representation of a relationship illustrated with edges (lines or arrows), which is widely used in industrial and scientific communication. Although recognizing diagrams is essential for vision language models (VLMs) to comprehend domain-specific knowledge, recent studies reveal that many VLMs fail to identify edges in images. We hypothesize that these failures stem from an over-reliance on textual and positional biases, preventing VLMs from learning explicit edge features. Based on this idea, we empirically investigate whether the image encoder in VLMs can learn edge representation through training on a diagram dataset in which edges are biased neither by textual nor positional information. To this end, we conduct contrastive learning on an artificially generated diagram--caption dataset to train an image encoder and evaluate its diagram-related features on three tasks: probing, image retrieval, and captioning. Our results show that the finetuned model outperforms pretrained CLIP in all tasks and surpasses zero-shot GPT-4o and LLaVA-Mistral in the captioning task. These findings confirm that eliminating textual and positional biases fosters accurate edge recognition in VLMs, offering a promising path for advancing diagram understanding.
>
---
#### [new 017] MM-Prompt: Cross-Modal Prompt Tuning for Continual Visual Question Answering
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对持续视觉问答任务，解决现有方法因分离构建视觉与文本提示导致的模态失衡及性能退化问题。提出MM-Prompt框架，通过跨模态提示查询与恢复机制，结合迭代交互和对齐损失，实现平衡模态学习并提升性能。**

- **链接: [http://arxiv.org/pdf/2505.19455v1](http://arxiv.org/pdf/2505.19455v1)**

> **作者:** Xu Li; Fan Lyu
>
> **摘要:** Continual Visual Question Answering (CVQA) based on pre-trained models(PTMs) has achieved promising progress by leveraging prompt tuning to enable continual multi-modal learning. However, most existing methods adopt cross-modal prompt isolation, constructing visual and textual prompts separately, which exacerbates modality imbalance and leads to degraded performance over time. To tackle this issue, we propose MM-Prompt, a novel framework incorporating cross-modal prompt query and cross-modal prompt recovery. The former enables balanced prompt selection by incorporating cross-modal signals during query formation, while the latter promotes joint prompt reconstruction through iterative cross-modal interactions, guided by an alignment loss to prevent representational drift. Extensive experiments show that MM-Prompt surpasses prior approaches in accuracy and knowledge retention, while maintaining balanced modality engagement throughout continual learning.
>
---
#### [new 018] Locality-Aware Zero-Shot Human-Object Interaction Detection
- **分类: cs.CV**

- **简介: 该论文属于零样本Human-Object Interaction（HOI）检测任务，旨在解决CLIP模型在未见类别上因忽略细粒度信息导致的互动识别不足问题。提出LAIN框架，通过聚合局部邻域补丁信息增强空间细节感知，并捕捉人-物间互动模式，提升CLIP表征的局部与交互意识，实验显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19503v1](http://arxiv.org/pdf/2505.19503v1)**

> **作者:** Sanghyun Kim; Deunsol Jung; Minsu Cho
>
> **备注:** Accepted to CVPR2025; Code is available at: https://github.com/OreoChocolate/LAIN
>
> **摘要:** Recent methods for zero-shot Human-Object Interaction (HOI) detection typically leverage the generalization ability of large Vision-Language Model (VLM), i.e., CLIP, on unseen categories, showing impressive results on various zero-shot settings. However, existing methods struggle to adapt CLIP representations for human-object pairs, as CLIP tends to overlook fine-grained information necessary for distinguishing interactions. To address this issue, we devise, LAIN, a novel zero-shot HOI detection framework enhancing the locality and interaction awareness of CLIP representations. The locality awareness, which involves capturing fine-grained details and the spatial structure of individual objects, is achieved by aggregating the information and spatial priors of adjacent neighborhood patches. The interaction awareness, which involves identifying whether and how a human is interacting with an object, is achieved by capturing the interaction pattern between the human and the object. By infusing locality and interaction awareness into CLIP representation, LAIN captures detailed information about the human-object pairs. Our extensive experiments on existing benchmarks show that LAIN outperforms previous methods on various zero-shot settings, demonstrating the importance of locality and interaction awareness for effective zero-shot HOI detection.
>
---
#### [new 019] A Responsible Face Recognition Approach for Small and Mid-Scale Systems Through Personalized Neural Networks
- **分类: cs.CV; cs.AI**

- **简介: 论文提出基于个性化神经网络的MOTE方法改进中小型人脸识别系统的公平性与隐私。针对传统固定模板缺乏可解释性及公平性问题，该方法为每个身份生成专属二分类器，通过单样本训练与合成数据平衡，提升个体公平性与隐私保护，适用于中小规模系统。**

- **链接: [http://arxiv.org/pdf/2505.19920v1](http://arxiv.org/pdf/2505.19920v1)**

> **作者:** Sebastian Groß; Stefan Heindorf; Philipp Terhörst
>
> **摘要:** Traditional face recognition systems rely on extracting fixed face representations, known as templates, to store and verify identities. These representations are typically generated by neural networks that often lack explainability and raise concerns regarding fairness and privacy. In this work, we propose a novel model-template (MOTE) approach that replaces vector-based face templates with small personalized neural networks. This design enables more responsible face recognition for small and medium-scale systems. During enrollment, MOTE creates a dedicated binary classifier for each identity, trained to determine whether an input face matches the enrolled identity. Each classifier is trained using only a single reference sample, along with synthetically balanced samples to allow adjusting fairness at the level of a single individual during enrollment. Extensive experiments across multiple datasets and recognition systems demonstrate substantial improvements in fairness and particularly in privacy. Although the method increases inference time and storage requirements, it presents a strong solution for small- and mid-scale applications where fairness and privacy are critical.
>
---
#### [new 020] Weather-Magician: Reconstruction and Rendering Framework for 4D Weather Synthesis In Real Time
- **分类: cs.CV**

- **简介: 该论文提出基于高斯散斑的实时4D天气合成框架Weather-Magician，解决传统方法在复杂天气场景重建与渲染中成本高、效果差的问题。通过高斯建模与渲染技术，实现多种天气效果的动态模拟与细节控制，支持低硬件需求下的实时渲染。**

- **链接: [http://arxiv.org/pdf/2505.19919v1](http://arxiv.org/pdf/2505.19919v1)**

> **作者:** Chen Sang; Yeqiang Qian; Jiale Zhang; Chunxiang Wang; Ming Yang
>
> **备注:** Project homepage: https://weathermagician.github.io
>
> **摘要:** For tasks such as urban digital twins, VR/AR/game scene design, or creating synthetic films, the traditional industrial approach often involves manually modeling scenes and using various rendering engines to complete the rendering process. This approach typically requires high labor costs and hardware demands, and can result in poor quality when replicating complex real-world scenes. A more efficient approach is to use data from captured real-world scenes, then apply reconstruction and rendering algorithms to quickly recreate the authentic scene. However, current algorithms are unable to effectively reconstruct and render real-world weather effects. To address this, we propose a framework based on gaussian splatting, that can reconstruct real scenes and render them under synthesized 4D weather effects. Our work can simulate various common weather effects by applying Gaussians modeling and rendering techniques. It supports continuous dynamic weather changes and can easily control the details of the effects. Additionally, our work has low hardware requirements and achieves real-time rendering performance. The result demos can be accessed on our project homepage: weathermagician.github.io
>
---
#### [new 021] MMIG-Bench: Towards Comprehensive and Explainable Evaluation of Multi-Modal Image Generation Models
- **分类: cs.CV**

- **简介: 该论文提出多模态图像生成评估基准MMIG-Bench，解决现有工具分散、忽略组合语义的问题。通过整合4850条文本提示与1750幅参考图，设计三级评估体系（视觉指标、AMS匹配分数、美学指标），测试17个模型，推动统一评估与生成模型创新。**

- **链接: [http://arxiv.org/pdf/2505.19415v1](http://arxiv.org/pdf/2505.19415v1)**

> **作者:** Hang Hua; Ziyun Zeng; Yizhi Song; Yunlong Tang; Liu He; Daniel Aliaga; Wei Xiong; Jiebo Luo
>
> **摘要:** Recent multimodal image generators such as GPT-4o, Gemini 2.0 Flash, and Gemini 2.5 Pro excel at following complex instructions, editing images and maintaining concept consistency. However, they are still evaluated by disjoint toolkits: text-to-image (T2I) benchmarks that lacks multi-modal conditioning, and customized image generation benchmarks that overlook compositional semantics and common knowledge. We propose MMIG-Bench, a comprehensive Multi-Modal Image Generation Benchmark that unifies these tasks by pairing 4,850 richly annotated text prompts with 1,750 multi-view reference images across 380 subjects, spanning humans, animals, objects, and artistic styles. MMIG-Bench is equipped with a three-level evaluation framework: (1) low-level metrics for visual artifacts and identity preservation of objects; (2) novel Aspect Matching Score (AMS): a VQA-based mid-level metric that delivers fine-grained prompt-image alignment and shows strong correlation with human judgments; and (3) high-level metrics for aesthetics and human preference. Using MMIG-Bench, we benchmark 17 state-of-the-art models, including Gemini 2.5 Pro, FLUX, DreamBooth, and IP-Adapter, and validate our metrics with 32k human ratings, yielding in-depth insights into architecture and data design. We will release the dataset and evaluation code to foster rigorous, unified evaluation and accelerate future innovations in multi-modal image generation.
>
---
#### [new 022] Restoring Real-World Images with an Internal Detail Enhancement Diffusion Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像修复任务，针对真实世界退化图像（如旧照片）的高保真恢复与物体级颜色控制问题。提出内部细节增强扩散模型，利用预训练Stable Diffusion，在潜空间通过IIDE技术模拟退化过程，增强结构纹理细节并支持文本引导修复，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.18674v1](http://arxiv.org/pdf/2505.18674v1)**

> **作者:** Peng Xiao; Hongbo Zhao; Yijun Wang; Jianxin Lin
>
> **摘要:** Restoring real-world degraded images, such as old photographs or low-resolution images, presents a significant challenge due to the complex, mixed degradations they exhibit, such as scratches, color fading, and noise. Recent data-driven approaches have struggled with two main challenges: achieving high-fidelity restoration and providing object-level control over colorization. While diffusion models have shown promise in generating high-quality images with specific controls, they often fail to fully preserve image details during restoration. In this work, we propose an internal detail-preserving diffusion model for high-fidelity restoration of real-world degraded images. Our method utilizes a pre-trained Stable Diffusion model as a generative prior, eliminating the need to train a model from scratch. Central to our approach is the Internal Image Detail Enhancement (IIDE) technique, which directs the diffusion model to preserve essential structural and textural information while mitigating degradation effects. The process starts by mapping the input image into a latent space, where we inject the diffusion denoising process with degradation operations that simulate the effects of various degradation factors. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art models in both qualitative assessments and perceptual quantitative evaluations. Additionally, our approach supports text-guided restoration, enabling object-level colorization control that mimics the expertise of professional photo editing.
>
---
#### [new 023] Align and Surpass Human Camouflaged Perception: Visual Refocus Reinforcement Fine-Tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态模型在伪装物体检测与分类任务中与人类视觉认知的偏差问题，提出基于策略优化的视觉再聚焦强化框架。通过动态调整注意力机制和逐步推理，提升模型对隐藏目标的定位能力，实验显示其性能超越监督微调基线，实现与超越人类伪装感知的对齐。**

- **链接: [http://arxiv.org/pdf/2505.19611v1](http://arxiv.org/pdf/2505.19611v1)**

> **作者:** Ruolin Shen; Xiaozhong Ji; Kai WU; Jiangning Zhang; Yijun He; HaiHua Yang; Xiaobin Hu; Xiaoyu Sun
>
> **备注:** Project Website: \url{https://github.com/HUuxiaobin/VRRF}
>
> **摘要:** Current multi-modal models exhibit a notable misalignment with the human visual system when identifying objects that are visually assimilated into the background. Our observations reveal that these multi-modal models cannot distinguish concealed objects, demonstrating an inability to emulate human cognitive processes which effectively utilize foreground-background similarity principles for visual analysis. To analyze this hidden human-model visual thinking discrepancy, we build a visual system that mimicks human visual camouflaged perception to progressively and iteratively `refocus' visual concealed content. The refocus is a progressive guidance mechanism enabling models to logically localize objects in visual images through stepwise reasoning. The localization process of concealed objects requires hierarchical attention shifting with dynamic adjustment and refinement of prior cognitive knowledge. In this paper, we propose a visual refocus reinforcement framework via the policy optimization algorithm to encourage multi-modal models to think and refocus more before answering, and achieve excellent reasoning abilities to align and even surpass human camouflaged perception systems. Our extensive experiments on camouflaged perception successfully demonstrate the emergence of refocus visual phenomena, characterized by multiple reasoning tokens and dynamic adjustment of the detection box. Besides, experimental results on both camouflaged object classification and detection tasks exhibit significantly superior performance compared to Supervised Fine-Tuning (SFT) baselines.
>
---
#### [new 024] SerendibCoins: Exploring The Sri Lankan Coins Dataset
- **分类: cs.CV**

- **简介: 该论文属于硬币分类任务，旨在提升斯里兰卡硬币识别准确性。构建了该国硬币图像数据集，对比传统机器学习（KNN、SVM、随机森林）与自定义CNN模型。实验显示SVM传统方法最优，CNN达近完美精度，为区域货币自动化识别提供数据基础与方法验证。**

- **链接: [http://arxiv.org/pdf/2505.18634v1](http://arxiv.org/pdf/2505.18634v1)**

> **作者:** NH Wanigasingha; ES Sithpahan; MKA Ariyaratne; PRS De Silva
>
> **备注:** 20 pages
>
> **摘要:** The recognition and classification of coins are essential in numerous financial and automated systems. This study introduces a comprehensive Sri Lankan coin image dataset and evaluates its impact on machine learning model accuracy for coin classification. We experiment with traditional machine learning classifiers K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forest as well as a custom Convolutional Neural Network (CNN) to benchmark performance at different levels of classification. Our results show that SVM outperforms KNN and Random Forest in traditional classification approaches, while the CNN model achieves near-perfect classification accuracy with minimal misclassifications. The dataset demonstrates significant potential in enhancing automated coin recognition systems, offering a robust foundation for future research in regional currency classification and deep learning applications.
>
---
#### [new 025] Advancing Limited-Angle CT Reconstruction Through Diffusion-Based Sinogram Completion
- **分类: cs.CV; cs.AI**

- **简介: 论文针对有限角CT重建任务，提出基于MR-SDE的sinogram补全方法，通过扩散模型填充缺失角度数据，结合蒸馏与伪逆约束加速重建，并采用后处理优化图像质量，解决角度缺失导致的伪影问题。**

- **链接: [http://arxiv.org/pdf/2505.19385v1](http://arxiv.org/pdf/2505.19385v1)**

> **作者:** Jiaqi Guo; Santiago Lopez-Tapia; Aggelos K. Katsaggelos
>
> **备注:** Accepted at the 2025 IEEE International Conference on Image Processing (Oral)
>
> **摘要:** Limited Angle Computed Tomography (LACT) often faces significant challenges due to missing angular information. Unlike previous methods that operate in the image domain, we propose a new method that focuses on sinogram inpainting. We leverage MR-SDEs, a variant of diffusion models that characterize the diffusion process with mean-reverting stochastic differential equations, to fill in missing angular data at the projection level. Furthermore, by combining distillation with constraining the output of the model using the pseudo-inverse of the inpainting matrix, the diffusion process is accelerated and done in a step, enabling efficient and accurate sinogram completion. A subsequent post-processing module back-projects the inpainted sinogram into the image domain and further refines the reconstruction, effectively suppressing artifacts while preserving critical structural details. Quantitative experimental results demonstrate that the proposed method achieves state-of-the-art performance in both perceptual and fidelity quality, offering a promising solution for LACT reconstruction in scientific and clinical applications.
>
---
#### [new 026] HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters
- **分类: cs.CV**

- **简介: 该论文属于音频驱动多人物动画生成任务，解决动态视频中角色一致性、情绪对齐及多角色同步驱动难题。提出HunyuanVideo-Avatar模型，通过角色图像注入模块提升动态一致性，音频情感模块实现情绪精准控制，面部感知适配器支持多角色独立驱动，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20156v1](http://arxiv.org/pdf/2505.20156v1)**

> **作者:** Yi Chen; Sen Liang; Zixiang Zhou; Ziyao Huang; Yifeng Ma; Junshu Tang; Qin Lin; Yuan Zhou; Qinglin Lu
>
> **摘要:** Recent years have witnessed significant progress in audio-driven human animation. However, critical challenges remain in (i) generating highly dynamic videos while preserving character consistency, (ii) achieving precise emotion alignment between characters and audio, and (iii) enabling multi-character audio-driven animation. To address these challenges, we propose HunyuanVideo-Avatar, a multimodal diffusion transformer (MM-DiT)-based model capable of simultaneously generating dynamic, emotion-controllable, and multi-character dialogue videos. Concretely, HunyuanVideo-Avatar introduces three key innovations: (i) A character image injection module is designed to replace the conventional addition-based character conditioning scheme, eliminating the inherent condition mismatch between training and inference. This ensures the dynamic motion and strong character consistency; (ii) An Audio Emotion Module (AEM) is introduced to extract and transfer the emotional cues from an emotion reference image to the target generated video, enabling fine-grained and accurate emotion style control; (iii) A Face-Aware Audio Adapter (FAA) is proposed to isolate the audio-driven character with latent-level face mask, enabling independent audio injection via cross-attention for multi-character scenarios. These innovations empower HunyuanVideo-Avatar to surpass state-of-the-art methods on benchmark datasets and a newly proposed wild dataset, generating realistic avatars in dynamic, immersive scenarios.
>
---
#### [new 027] Sampling Strategies for Efficient Training of Deep Learning Object Detection Algorithms
- **分类: cs.CV; cs.IT; math.IT**

- **简介: 该论文研究物体检测模型训练效率优化任务，针对减少标注数据需求问题，提出两种采样策略：均匀采样（空间均衡）和帧差采样（利用视频时序冗余），实验表明其能以少量标注数据提升训练效果。**

- **链接: [http://arxiv.org/pdf/2505.18302v1](http://arxiv.org/pdf/2505.18302v1)**

> **作者:** Gefei Shen; Yung-Hong Sun; Yu Hen Hu; Hongrui Jiang
>
> **摘要:** Two sampling strategies are investigated to enhance efficiency in training a deep learning object detection model. These sampling strategies are employed under the assumption of Lipschitz continuity of deep learning models. The first strategy is uniform sampling which seeks to obtain samples evenly yet randomly through the state space of the object dynamics. The second strategy of frame difference sampling is developed to explore the temporal redundancy among successive frames in a video. Experiment result indicates that these proposed sampling strategies provide a dataset that yields good training performance while requiring relatively few manually labelled samples.
>
---
#### [new 028] ZooplanktonBench: A Geo-Aware Zooplankton Recognition and Classification Dataset from Marine Observations
- **分类: cs.CV**

- **简介: 该论文构建了ZooplanktonBench数据集，针对海洋观测中浮游动物识别的挑战（如背景干扰、形态相似等），提供含地理空间元数据的图像/视频，设计检测、分类、跟踪等任务，推动计算机视觉在动态海洋环境中的应用。**

- **链接: [http://arxiv.org/pdf/2505.18477v1](http://arxiv.org/pdf/2505.18477v1)**

> **作者:** Fukun Liu; Adam T. Greer; Gengchen Mai; Jin Sun
>
> **摘要:** Plankton are small drifting organisms found throughout the world's oceans. One component of this plankton community is the zooplankton, which includes gelatinous animals and crustaceans (e.g. shrimp), as well as the early life stages (i.e., eggs and larvae) of many commercially important fishes. Being able to monitor zooplankton abundances accurately and understand how populations change in relation to ocean conditions is invaluable to marine science research, with important implications for future marine seafood productivity. While new imaging technologies generate massive amounts of video data of zooplankton, analyzing them using general-purpose computer vision tools developed for general objects turns out to be highly challenging due to the high similarity in appearance between the zooplankton and its background (e.g., marine snow). In this work, we present the ZooplanktonBench, a benchmark dataset containing images and videos of zooplankton associated with rich geospatial metadata (e.g., geographic coordinates, depth, etc.) in various water ecosystems. ZooplanktonBench defines a collection of tasks to detect, classify, and track zooplankton in challenging settings, including highly cluttered environments, living vs non-living classification, objects with similar shapes, and relatively small objects. Our dataset presents unique challenges and opportunities for state-of-the-art computer vision systems to evolve and improve visual understanding in a dynamic environment with huge variations and be geo-aware.
>
---
#### [new 029] Monocular Marker-free Patient-to-Image Intraoperative Registration for Cochlear Implant Surgery
- **分类: cs.CV**

- **简介: 该论文提出单目无标记耳蜗手术术中配准方法，解决传统依赖外部追踪设备或标记点的问题。通过轻量级神经网络学习合成显微数据，将术前CT与术中图像实时配准，估计6D相机位姿。临床验证显示角度误差<10°，提升临床实用性。**

- **链接: [http://arxiv.org/pdf/2505.18381v1](http://arxiv.org/pdf/2505.18381v1)**

> **作者:** Yike Zhang; Eduardo Davalos Anaya; Jack H. Noble
>
> **摘要:** This paper presents a novel method for monocular patient-to-image intraoperative registration, specifically designed to operate without any external hardware tracking equipment or fiducial point markers. Leveraging a synthetic microscopy surgical scene dataset with a wide range of transformations, our approach directly maps preoperative CT scans to 2D intraoperative surgical frames through a lightweight neural network for real-time cochlear implant surgery guidance via a zero-shot learning approach. Unlike traditional methods, our framework seamlessly integrates with monocular surgical microscopes, making it highly practical for clinical use without additional hardware dependencies and requirements. Our method estimates camera poses, which include a rotation matrix and a translation vector, by learning from the synthetic dataset, enabling accurate and efficient intraoperative registration. The proposed framework was evaluated on nine clinical cases using a patient-specific and cross-patient validation strategy. Our results suggest that our approach achieves clinically relevant accuracy in predicting 6D camera poses for registering 3D preoperative CT scans to 2D surgical scenes with an angular error within 10 degrees in most cases, while also addressing limitations of traditional methods, such as reliance on external tracking systems or fiducial markers.
>
---
#### [new 030] Modality Curation: Building Universal Embeddings for Advanced Multimodal Information Retrieval
- **分类: cs.CV; cs.IR; cs.MM**

- **简介: 该论文属多模态信息检索任务，旨在解决模态异构性及跨模态对齐挑战。提出UNITE框架，通过模态数据筛选与定制化训练配置，并采用MAMCL方法缓解模态间竞争，实现跨模态检索性能提升，达当前最优水平。**

- **链接: [http://arxiv.org/pdf/2505.19650v1](http://arxiv.org/pdf/2505.19650v1)**

> **作者:** Fanheng Kong; Jingyuan Zhang; Yahui Liu; Hongzhi Zhang; Shi Feng; Xiaocui Yang; Daling Wang; Yu Tian; Qi Wang; Fuzheng Zhang; Guorui Zhou
>
> **备注:** 26 pages, project page: https://friedrichor.github.io/projects/UNITE
>
> **摘要:** Multimodal information retrieval (MIR) faces inherent challenges due to the heterogeneity of data sources and the complexity of cross-modal alignment. While previous studies have identified modal gaps in feature spaces, a systematic approach to address these challenges remains unexplored. In this work, we introduce UNITE, a universal framework that tackles these challenges through two critical yet underexplored aspects: data curation and modality-aware training configurations. Our work provides the first comprehensive analysis of how modality-specific data properties influence downstream task performance across diverse scenarios. Moreover, we propose Modal-Aware Masked Contrastive Learning (MAMCL) to mitigate the competitive relationships among the instances of different modalities. Our framework achieves state-of-the-art results on multiple multimodal retrieval benchmarks, outperforming existing methods by notable margins. Through extensive experiments, we demonstrate that strategic modality curation and tailored training protocols are pivotal for robust cross-modal representation learning. This work not only advances MIR performance but also provides a foundational blueprint for future research in multimodal systems. Our project is available at https://friedrichor.github.io/projects/UNITE.
>
---
#### [new 031] EvdCLIP: Improving Vision-Language Retrieval with Entity Visual Descriptions from Large Language Models
- **分类: cs.CV; cs.IR**

- **简介: 该论文属于视觉语言检索（VLR）任务，旨在解决现有方法忽视实体视觉语义导致检索错误的问题。提出EvdCLIP，利用大语言模型生成实体视觉描述（EVD）增强查询，并设计可训练的EVD感知重写器（EaRW）优化查询质量，提升检索性能。**

- **链接: [http://arxiv.org/pdf/2505.18594v1](http://arxiv.org/pdf/2505.18594v1)**

> **作者:** GuangHao Meng; Sunan He; Jinpeng Wang; Tao Dai; Letian Zhang; Jieming Zhu; Qing Li; Gang Wang; Rui Zhang; Yong Jiang
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Vision-language retrieval (VLR) has attracted significant attention in both academia and industry, which involves using text (or images) as queries to retrieve corresponding images (or text). However, existing methods often neglect the rich visual semantics knowledge of entities, thus leading to incorrect retrieval results. To address this problem, we propose the Entity Visual Description enhanced CLIP (EvdCLIP), designed to leverage the visual knowledge of entities to enrich queries. Specifically, since humans recognize entities through visual cues, we employ a large language model (LLM) to generate Entity Visual Descriptions (EVDs) as alignment cues to complement textual data. These EVDs are then integrated into raw queries to create visually-rich, EVD-enhanced queries. Furthermore, recognizing that EVD-enhanced queries may introduce noise or low-quality expansions, we develop a novel, trainable EVD-aware Rewriter (EaRW) for vision-language retrieval tasks. EaRW utilizes EVD knowledge and the generative capabilities of the language model to effectively rewrite queries. With our specialized training strategy, EaRW can generate high-quality and low-noise EVD-enhanced queries. Extensive quantitative and qualitative experiments on image-text retrieval benchmarks validate the superiority of EvdCLIP on vision-language retrieval tasks.
>
---
#### [new 032] Co-AttenDWG: Co-Attentive Dimension-Wise Gating and Expert Fusion for Multi-Modal Offensive Content Detection
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态攻击性内容检测任务，解决现有方法跨模态交互不足和静态融合策略的问题。提出Co-AttenDWG模型，采用双路径编码、协同注意力与维度门控机制及专家融合模块，动态调节特征交互与融合，提升跨模态对齐与检测性能。**

- **链接: [http://arxiv.org/pdf/2505.19010v1](http://arxiv.org/pdf/2505.19010v1)**

> **作者:** Md. Mithun Hossain; Md. Shakil Hossain; Sudipto Chaki; M. F. Mridha
>
> **摘要:** Multi-modal learning has become a critical research area because integrating text and image data can significantly improve performance in tasks such as classification, retrieval, and scene understanding. However, despite progress with pre-trained models, current approaches are limited by inadequate cross-modal interactions and static fusion strategies that do not fully exploit the complementary nature of different modalities. To address these shortcomings, we introduce a novel multi-modal Co-AttenDWG architecture that leverages dual-path encoding, co-attention with dimension-wise gating, and advanced expert fusion. Our approach begins by projecting text and image features into a common embedding space, where a dedicated co-attention mechanism enables simultaneous, fine-grained interactions between modalities. This mechanism is further enhanced by a dimension-wise gating network that adaptively regulates the feature contributions at the channel level, ensuring that only the most relevant information is emphasized. In parallel, dual-path encoders refine the representations by processing cross-modal information separately before an additional cross-attention layer further aligns modalities. The refined features are then aggregated via an expert fusion module that combines learned gating and self-attention to produce a robust, unified representation. We validate our approach on the MIMIC and SemEval Memotion 1.0, where experimental results demonstrate significant improvements in cross-modal alignment and state-of-the-art performance, underscoring the potential of our model for a wide range of multi-modal applications.
>
---
#### [new 033] Cross-Sequence Semi-Supervised Learning for Multi-Parametric MRI-Based Visual Pathway Delineation
- **分类: cs.CV; cs.CE**

- **简介: 该论文属于医学影像分析任务，旨在解决多参数MRI视觉通路勾勒中跨序列信息融合不足及标注数据依赖问题。提出基于相关约束特征分解（CFD）处理序列间关系，并开发一致性样本增强（CSE）模块利用无标注数据，提升勾勒精度。**

- **链接: [http://arxiv.org/pdf/2505.19733v1](http://arxiv.org/pdf/2505.19733v1)**

> **作者:** Alou Diakite; Cheng Li; Lei Xie; Yuanjing Feng; Ruoyou Wu; Jianzhong He; Hairong Zheng; Shanshan Wang
>
> **摘要:** Accurately delineating the visual pathway (VP) is crucial for understanding the human visual system and diagnosing related disorders. Exploring multi-parametric MR imaging data has been identified as an important way to delineate VP. However, due to the complex cross-sequence relationships, existing methods cannot effectively model the complementary information from different MRI sequences. In addition, these existing methods heavily rely on large training data with labels, which is labor-intensive and time-consuming to obtain. In this work, we propose a novel semi-supervised multi-parametric feature decomposition framework for VP delineation. Specifically, a correlation-constrained feature decomposition (CFD) is designed to handle the complex cross-sequence relationships by capturing the unique characteristics of each MRI sequence and easing the multi-parametric information fusion process. Furthermore, a consistency-based sample enhancement (CSE) module is developed to address the limited labeled data issue, by generating and promoting meaningful edge information from unlabeled data. We validate our framework using two public datasets, and one in-house Multi-Shell Diffusion MRI (MDM) dataset. Experimental results demonstrate the superiority of our approach in terms of delineation performance when compared to seven state-of-the-art approaches.
>
---
#### [new 034] Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于可泛化神经辐射场（NeRF）重建任务。针对高分辨率渲染计算效率低的问题，提出深度引导束采样方法，通过分组相邻光线共享采样并基于深度置信度动态分配样本，减少冗余计算。在DTU数据集上提升PSNR 1.27dB，加速47%，实现更快更优渲染。**

- **链接: [http://arxiv.org/pdf/2505.19793v1](http://arxiv.org/pdf/2505.19793v1)**

> **作者:** Li Fang; Hao Zhu; Longlong Chen; Fei Hu; Long Ye; Zhan Ma
>
> **备注:** CVPR 2025
>
> **摘要:** Recent advancements in generalizable novel view synthesis have achieved impressive quality through interpolation between nearby views. However, rendering high-resolution images remains computationally intensive due to the need for dense sampling of all rays. Recognizing that natural scenes are typically piecewise smooth and sampling all rays is often redundant, we propose a novel depth-guided bundle sampling strategy to accelerate rendering. By grouping adjacent rays into a bundle and sampling them collectively, a shared representation is generated for decoding all rays within the bundle. To further optimize efficiency, our adaptive sampling strategy dynamically allocates samples based on depth confidence, concentrating more samples in complex regions while reducing them in smoother areas. When applied to ENeRF, our method achieves up to a 1.27 dB PSNR improvement and a 47% increase in FPS on the DTU dataset. Extensive experiments on synthetic and real-world datasets demonstrate state-of-the-art rendering quality and up to 2x faster rendering compared to existing generalizable methods. Code is available at https://github.com/KLMAV-CUC/GDB-NeRF.
>
---
#### [new 035] Seeing is Believing, but How Much? A Comprehensive Analysis of Verbalized Calibration in Vision-Language Models
- **分类: cs.CV**

- **简介: 论文评估视觉语言模型（VLMs）的verbalized不确定性校准，发现其普遍存在偏差，尤其视觉推理模型表现较优。提出两阶段提示策略（Visual Confidence-Aware Prompting）改进校准，强调模态对齐对可靠多模态系统的重要性。任务聚焦VLMs不确定性量化，解决其校准不足问题，通过跨模型/任务的评估与方法优化实现目标。**

- **链接: [http://arxiv.org/pdf/2505.20236v1](http://arxiv.org/pdf/2505.20236v1)**

> **作者:** Weihao Xuan; Qingcheng Zeng; Heli Qi; Junjue Wang; Naoto Yokoya
>
> **摘要:** Uncertainty quantification is essential for assessing the reliability and trustworthiness of modern AI systems. Among existing approaches, verbalized uncertainty, where models express their confidence through natural language, has emerged as a lightweight and interpretable solution in large language models (LLMs). However, its effectiveness in vision-language models (VLMs) remains insufficiently studied. In this work, we conduct a comprehensive evaluation of verbalized confidence in VLMs, spanning three model categories, four task domains, and three evaluation scenarios. Our results show that current VLMs often display notable miscalibration across diverse tasks and settings. Notably, visual reasoning models (i.e., thinking with images) consistently exhibit better calibration, suggesting that modality-specific reasoning is critical for reliable uncertainty estimation. To further address calibration challenges, we introduce Visual Confidence-Aware Prompting, a two-stage prompting strategy that improves confidence alignment in multimodal settings. Overall, our study highlights the inherent miscalibration in VLMs across modalities. More broadly, our findings underscore the fundamental importance of modality alignment and model faithfulness in advancing reliable multimodal systems.
>
---
#### [new 036] CONCORD: Concept-Informed Diffusion for Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文属于数据集蒸馏任务，旨在解决现有方法缺乏样本级概念控制及细节缺失问题。提出CONCORD方法，利用大语言模型提取细粒度概念指导扩散模型生成，通过概念驱动的去噪优化关键细节，提升可控性和可解释性，无需预训练分类器，在ImageNet等数据集达最优效果。**

- **链接: [http://arxiv.org/pdf/2505.18358v1](http://arxiv.org/pdf/2505.18358v1)**

> **作者:** Jianyang Gu; Haonan Wang; Ruoxi Jia; Saeed Vahidian; Vyacheslav Kungurtsev; Wei Jiang; Yiran Chen
>
> **摘要:** Dataset distillation (DD) has witnessed significant progress in creating small datasets that encapsulate rich information from large original ones. Particularly, methods based on generative priors show promising performance, while maintaining computational efficiency and cross-architecture generalization. However, the generation process lacks explicit controllability for each sample. Previous distillation methods primarily match the real distribution from the perspective of the entire dataset, whereas overlooking concept completeness at the instance level. The missing or incorrectly represented object details cannot be efficiently compensated due to the constrained sample amount typical in DD settings. To this end, we propose incorporating the concept understanding of large language models (LLMs) to perform Concept-Informed Diffusion (CONCORD) for dataset distillation. Specifically, distinguishable and fine-grained concepts are retrieved based on category labels to inform the denoising process and refine essential object details. By integrating these concepts, the proposed method significantly enhances both the controllability and interpretability of the distilled image generation, without relying on pre-trained classifiers. We demonstrate the efficacy of CONCORD by achieving state-of-the-art performance on ImageNet-1K and its subsets. The code implementation is released in https://github.com/vimar-gu/CONCORD.
>
---
#### [new 037] MIND-Edit: MLLM Insight-Driven Editing via Language-Vision Projection
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，解决现有方法在复杂场景中精度不足及文本-视觉语义不匹配的问题。提出MIND-Edit框架，结合扩散模型与多模态语言模型，通过优化文本指令和利用模型视觉理解能力生成编辑指引，并采用联合训练提升编辑精度与语义一致性。**

- **链接: [http://arxiv.org/pdf/2505.19149v1](http://arxiv.org/pdf/2505.19149v1)**

> **作者:** Shuyu Wang; Weiqi Li; Qian Wang; Shijie Zhao; Jian Zhang
>
> **摘要:** Recent advances in AI-generated content (AIGC) have significantly accelerated image editing techniques, driving increasing demand for diverse and fine-grained edits. Despite these advances, existing image editing methods still face challenges in achieving high precision and semantic accuracy in complex scenarios. Recent studies address this issue by incorporating multimodal large language models (MLLMs) into image editing pipelines. However, current MLLM-based methods mainly rely on interpreting textual instructions, leaving the intrinsic visual understanding of large models largely unexplored, thus resulting in insufficient alignment between textual semantics and visual outcomes. To overcome these limitations, we propose MIND-Edit, an end-to-end image-editing framework integrating pretrained diffusion model with MLLM. MIND-Edit introduces two complementary strategies: (1) a text instruction optimization strategy that clarifies ambiguous user instructions based on semantic reasoning from the MLLM, and (2) an MLLM insight-driven editing strategy that explicitly leverages the intrinsic visual understanding capability of the MLLM to infer editing intent and guide the diffusion process via generated visual embeddings. Furthermore, we propose a joint training approach to effectively integrate both strategies, allowing them to reinforce each other for more accurate instruction interpretation and visually coherent edits aligned with user intent. Extensive experiments demonstrate that MIND-Edit outperforms state-of-the-art image editing methods in both quantitative metrics and visual quality, particularly under complex and challenging scenarios.
>
---
#### [new 038] Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉推理任务，旨在解决现有方法依赖高成本标注及LVLMs推理不可靠的问题。提出Ground-R1框架，通过强化学习实现无需显式标注的grounded推理，分阶段生成证据区域与答案，提升性能与可解释性。**

- **链接: [http://arxiv.org/pdf/2505.20272v1](http://arxiv.org/pdf/2505.20272v1)**

> **作者:** Meng Cao; Haoze Zhao; Can Zhang; Xiaojun Chang; Ian Reid; Xiaodan Liang
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated impressive general capabilities across a wide range of multi-modal tasks. However, the reasoning processes of LVLMs often suffer from unreliable outputs and limited interpretability. To address this, grounded visual reasoning has emerged as a promising paradigm that enforces responses anchored on salient visual evidence regions. However, existing approaches typically rely on costly supervision such as bounding box annotations, chain-of-thought rationale or external tool calls, limiting their scalability. In this work, we propose Ground-R1, a reinforcement learning framework that enables grounded visual reasoning without requiring explicit evidence or rationale annotations. Ground-R1 consists of a grounding phase that generates evidence region rollouts based on format constraints, and an answering phase that produces responses guided by both answer correctness and format adherence rewards. Extensive experiments across multiple visual reasoning benchmarks manifest that Ground-R1 achieves superior performance and exhibits emergent cognitive behaviors such as uncertainty awareness, spatial perception, and iterative refinement, offering a scalable and interpretable alternative to existing approaches.
>
---
#### [new 039] Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual Reasoning from Transit Maps
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉推理基准测试任务，旨在评估MLLMs在交通地图细粒度空间推理中的能力。针对现有模型在此类任务的不足，提出ReasonMap基准，包含30城高分辨率地图及1008道题，并设计两级评估流程。测试15种MLLMs发现开源模型基础版优于推理版，闭源相反，且视觉输入遮罩会降性能，揭示模型依赖真实视觉感知的必要性。**

- **链接: [http://arxiv.org/pdf/2505.18675v1](http://arxiv.org/pdf/2505.18675v1)**

> **作者:** Sicheng Feng; Song Wang; Shuyi Ouyang; Lingdong Kong; Zikai Song; Jianke Zhu; Huan Wang; Xinchao Wang
>
> **摘要:** Multimodal large language models (MLLMs) have recently achieved significant progress in visual tasks, including semantic scene understanding and text-image alignment, with reasoning variants enhancing performance on complex tasks involving mathematics and logic. However, their capacity for reasoning tasks involving fine-grained visual understanding remains insufficiently evaluated. To address this gap, we introduce ReasonMap, a benchmark designed to assess the fine-grained visual understanding and spatial reasoning abilities of MLLMs. ReasonMap encompasses high-resolution transit maps from 30 cities across 13 countries and includes 1,008 question-answer pairs spanning two question types and three templates. Furthermore, we design a two-level evaluation pipeline that properly assesses answer correctness and quality. Comprehensive evaluations of 15 popular MLLMs, including both base and reasoning variants, reveal a counterintuitive pattern: among open-source models, base models outperform reasoning ones, while the opposite trend is observed in closed-source models. Additionally, performance generally degrades when visual inputs are masked, indicating that while MLLMs can leverage prior knowledge to answer some questions, fine-grained visual reasoning tasks still require genuine visual perception for strong performance. Our benchmark study offers new insights into visual reasoning and contributes to investigating the gap between open-source and closed-source models.
>
---
#### [new 040] The Eye of Sherlock Holmes: Uncovering User Private Attribute Profiling via Vision-Language Model Agentic Framework
- **分类: cs.CV**

- **简介: 论文研究视觉语言模型（VLM）通过用户图像集推断隐私属性（如年龄、健康、性格）的隐私风险。针对缺乏多图标注数据集及模型处理抽象属性能力不足问题，构建PAPI数据集（2510图/3012标注）并提出HolmesEye框架，结合VLM与LLM，推理准确率超SOTA 10.8%，抽象属性预测超人类15%，推动隐私保护研究。**

- **链接: [http://arxiv.org/pdf/2505.19139v1](http://arxiv.org/pdf/2505.19139v1)**

> **作者:** Feiran Liu; Yuzhe Zhang; Xinyi Huang; Yinan Peng; Xinfeng Li; Lixu Wang; Yutong Shen; Ranjie Duan; Simeng Qin; Xiaojun Jia; Qingsong Wen; Wei Dong
>
> **摘要:** Our research reveals a new privacy risk associated with the vision-language model (VLM) agentic framework: the ability to infer sensitive attributes (e.g., age and health information) and even abstract ones (e.g., personality and social traits) from a set of personal images, which we term "image private attribute profiling." This threat is particularly severe given that modern apps can easily access users' photo albums, and inference from image sets enables models to exploit inter-image relations for more sophisticated profiling. However, two main challenges hinder our understanding of how well VLMs can profile an individual from a few personal photos: (1) the lack of benchmark datasets with multi-image annotations for private attributes, and (2) the limited ability of current multimodal large language models (MLLMs) to infer abstract attributes from large image collections. In this work, we construct PAPI, the largest dataset for studying private attribute profiling in personal images, comprising 2,510 images from 251 individuals with 3,012 annotated privacy attributes. We also propose HolmesEye, a hybrid agentic framework that combines VLMs and LLMs to enhance privacy inference. HolmesEye uses VLMs to extract both intra-image and inter-image information and LLMs to guide the inference process as well as consolidate the results through forensic analysis, overcoming existing limitations in long-context visual reasoning. Experiments reveal that HolmesEye achieves a 10.8% improvement in average accuracy over state-of-the-art baselines and surpasses human-level performance by 15.0% in predicting abstract attributes. This work highlights the urgency of addressing privacy risks in image-based profiling and offers both a new dataset and an advanced framework to guide future research in this area.
>
---
#### [new 041] Align Beyond Prompts: Evaluating World Knowledge Alignment in Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成评估任务，旨在解决模型忽视现实知识（超出用户提示）对齐的问题。提出ABP基准（含2000+提示，覆盖六场景）及ABPScore指标，评估模型表现；发现先进模型仍存在不足，并提出无需训练的ITKI策略，使ABPScore提升43%。**

- **链接: [http://arxiv.org/pdf/2505.18730v1](http://arxiv.org/pdf/2505.18730v1)**

> **作者:** Wenchao Zhang; Jiahe Tian; Runze He; Jizhong Han; Jiao Dai; Miaomiao Feng; Wei Mi; Xiaodan Zhang
>
> **备注:** Code: https://github.com/smile365317/ABP
>
> **摘要:** Recent text-to-image (T2I) generation models have advanced significantly, enabling the creation of high-fidelity images from textual prompts. However, existing evaluation benchmarks primarily focus on the explicit alignment between generated images and prompts, neglecting the alignment with real-world knowledge beyond prompts. To address this gap, we introduce Align Beyond Prompts (ABP), a comprehensive benchmark designed to measure the alignment of generated images with real-world knowledge that extends beyond the explicit user prompts. ABP comprises over 2,000 meticulously crafted prompts, covering real-world knowledge across six distinct scenarios. We further introduce ABPScore, a metric that utilizes existing Multimodal Large Language Models (MLLMs) to assess the alignment between generated images and world knowledge beyond prompts, which demonstrates strong correlations with human judgments. Through a comprehensive evaluation of 8 popular T2I models using ABP, we find that even state-of-the-art models, such as GPT-4o, face limitations in integrating simple real-world knowledge into generated images. To mitigate this issue, we introduce a training-free strategy within ABP, named Inference-Time Knowledge Injection (ITKI). By applying this strategy to optimize 200 challenging samples, we achieved an improvement of approximately 43% in ABPScore. The dataset and code are available in https://github.com/smile365317/ABP.
>
---
#### [new 042] DepthMatch: Semi-Supervised RGB-D Scene Parsing through Depth-Guided Regularization
- **分类: cs.CV**

- **简介: 该论文属于RGB-D场景解析任务，旨在解决监督学习依赖大量标注数据的问题。提出DepthMatch框架，通过互补块混合增强、轻量级空间先验模块及深度引导边界损失，提升半监督学习效率与边界预测精度，在NYUv2和KITTI数据集取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.20041v1](http://arxiv.org/pdf/2505.20041v1)**

> **作者:** Jianxin Huang; Jiahang Li; Sergey Vityazev; Alexander Dvorkovich; Rui Fan
>
> **备注:** 5 pages, 2 figures, accepted by IEEE Signal Processing Letters
>
> **摘要:** RGB-D scene parsing methods effectively capture both semantic and geometric features of the environment, demonstrating great potential under challenging conditions such as extreme weather and low lighting. However, existing RGB-D scene parsing methods predominantly rely on supervised training strategies, which require a large amount of manually annotated pixel-level labels that are both time-consuming and costly. To overcome these limitations, we introduce DepthMatch, a semi-supervised learning framework that is specifically designed for RGB-D scene parsing. To make full use of unlabeled data, we propose complementary patch mix-up augmentation to explore the latent relationships between texture and spatial features in RGB-D image pairs. We also design a lightweight spatial prior injector to replace traditional complex fusion modules, improving the efficiency of heterogeneous feature fusion. Furthermore, we introduce depth-guided boundary loss to enhance the model's boundary prediction capabilities. Experimental results demonstrate that DepthMatch exhibits high applicability in both indoor and outdoor scenes, achieving state-of-the-art results on the NYUv2 dataset and ranking first on the KITTI Semantics benchmark.
>
---
#### [new 043] C3R: Channel Conditioned Cell Representations for unified evaluation in microscopy imaging
- **分类: cs.CV; cs.LG; q-bio.QM**

- **简介: 该论文提出C3R框架，解决免疫组化图像因通道配置不一致导致的模型跨数据集泛化问题。任务为统一处理IHC图像的分布内/外评估，通过将通道分为"背景-概念"结构，设计自适应编码器与知识蒸馏策略，实现无需数据集特适性调整的零样本迁移。**

- **链接: [http://arxiv.org/pdf/2505.18745v1](http://arxiv.org/pdf/2505.18745v1)**

> **作者:** Umar Marikkar; Syed Sameed Husain; Muhammad Awais; Sara Atito
>
> **摘要:** Immunohistochemical (IHC) images reveal detailed information about structures and functions at the subcellular level. However, unlike natural images, IHC datasets pose challenges for deep learning models due to their inconsistencies in channel count and configuration, stemming from varying staining protocols across laboratories and studies. Existing approaches build channel-adaptive models, which unfortunately fail to support out-of-distribution (OOD) evaluation across IHC datasets and cannot be applied in a true zero-shot setting with mismatched channel counts. To address this, we introduce a structured view of cellular image channels by grouping them into either context or concept, where we treat the context channels as a reference to the concept channels in the image. We leverage this context-concept principle to develop Channel Conditioned Cell Representations (C3R), a framework designed for unified evaluation on in-distribution (ID) and OOD datasets. C3R is a two-fold framework comprising a channel-adaptive encoder architecture and a masked knowledge distillation training strategy, both built around the context-concept principle. We find that C3R outperforms existing benchmarks on both ID and OOD tasks, while a trivial implementation of our core idea also outperforms the channel-adaptive methods reported on the CHAMMI benchmark. Our method opens a new pathway for cross-dataset generalization between IHC datasets, without requiring dataset-specific adaptation or retraining.
>
---
#### [new 044] ThinkVideo: High-Quality Reasoning Video Segmentation with Chain of Thoughts
- **分类: cs.CV**

- **简介: 该论文提出ThinkVideo框架，针对推理视频目标分割任务，解决现有方法在处理时间敏感查询时时空信息整合不足的问题。通过零样本思维链（CoT）提取关键帧对象特征，结合分割模型与SAM2视频处理器生成掩码序列，实现无需训练且支持在线视频流动态更新，实验显示显著优于先前方法。**

- **链接: [http://arxiv.org/pdf/2505.18561v1](http://arxiv.org/pdf/2505.18561v1)**

> **作者:** Shiu-hong Kao; Yu-Wing Tai; Chi-Keung Tang
>
> **备注:** Project page: https://cse.hkust.edu.hk/~skao/thinkvideo.html
>
> **摘要:** Reasoning Video Object Segmentation is a challenging task, which generates a mask sequence from an input video and an implicit, complex text query. Existing works probe into the problem by finetuning Multimodal Large Language Models (MLLM) for segmentation-based output, while still falling short in difficult cases on videos given temporally-sensitive queries, primarily due to the failure to integrate temporal and spatial information. In this paper, we propose ThinkVideo, a novel framework which leverages the zero-shot Chain-of-Thought (CoT) capability of MLLM to address these challenges. Specifically, ThinkVideo utilizes the CoT prompts to extract object selectivities associated with particular keyframes, then bridging the reasoning image segmentation model and SAM2 video processor to output mask sequences. The ThinkVideo framework is training-free and compatible with closed-source MLLMs, which can be applied to Reasoning Video Instance Segmentation. We further extend the framework for online video streams, where the CoT is used to update the object of interest when a better target starts to emerge and becomes visible. We conduct extensive experiments on video object segmentation with explicit and implicit queries. The results show that ThinkVideo significantly outperforms previous works in both cases, qualitatively and quantitatively.
>
---
#### [new 045] FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities
- **分类: cs.CV**

- **简介: 该论文属多模态统一理解和生成任务，解决自回归模型图像生成顺序依赖及推理局限问题。提出FUDOKI模型，基于离散流匹配与动力学最优速度路径，实现迭代修正和双向上下文整合，通过预训练模型初始化提升效率，实验性能媲美现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20147v1](http://arxiv.org/pdf/2505.20147v1)**

> **作者:** Jin Wang; Yao Lai; Aoxue Li; Shifeng Zhang; Jiacheng Sun; Ning Kang; Chengyue Wu; Zhenguo Li; Ping Luo
>
> **备注:** 37 pages, 12 figures
>
> **摘要:** The rapid progress of large language models (LLMs) has catalyzed the emergence of multimodal large language models (MLLMs) that unify visual understanding and image generation within a single framework. However, most existing MLLMs rely on autoregressive (AR) architectures, which impose inherent limitations on future development, such as the raster-scan order in image generation and restricted reasoning abilities in causal context modeling. In this work, we challenge the dominance of AR-based approaches by introducing FUDOKI, a unified multimodal model purely based on discrete flow matching, as an alternative to conventional AR paradigms. By leveraging metric-induced probability paths with kinetic optimal velocities, our framework goes beyond the previous masking-based corruption process, enabling iterative refinement with self-correction capability and richer bidirectional context integration during generation. To mitigate the high cost of training from scratch, we initialize FUDOKI from pre-trained AR-based MLLMs and adaptively transition to the discrete flow matching paradigm. Experimental results show that FUDOKI achieves performance comparable to state-of-the-art AR-based MLLMs across both visual understanding and image generation tasks, highlighting its potential as a foundation for next-generation unified multimodal models. Furthermore, we show that applying test-time scaling techniques to FUDOKI yields significant performance gains, further underscoring its promise for future enhancement through reinforcement learning.
>
---
#### [new 046] Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations
- **分类: cs.CV**

- **简介: 该论文针对视觉对应任务，解决Diffusion Transformers（DiTs）因“大量激活”导致性能下降的问题。提出DiTF框架，通过AdaLN定位并归一化异常激活，结合通道丢弃策略消除其负面影响，显著提升DiTs表现，超越现有模型，在Spair-71k等任务中达新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.18584v1](http://arxiv.org/pdf/2505.18584v1)**

> **作者:** Chaofan Gan; Yuanpeng Tu; Xi Chen; Tieyuan Chen; Yuxi Li; Mehrtash Harandi; Weiyao Lin
>
> **备注:** Under Review
>
> **摘要:** Pre-trained stable diffusion models (SD) have shown great advances in visual correspondence. In this paper, we investigate the capabilities of Diffusion Transformers (DiTs) for accurate dense correspondence. Distinct from SD, DiTs exhibit a critical phenomenon in which very few feature activations exhibit significantly larger values than others, known as \textit{massive activations}, leading to uninformative representations and significant performance degradation for DiTs. The massive activations consistently concentrate at very few fixed dimensions across all image patch tokens, holding little local information. We trace these dimension-concentrated massive activations and find that such concentration can be effectively localized by the zero-initialized Adaptive Layer Norm (AdaLN-zero). Building on these findings, we propose Diffusion Transformer Feature (DiTF), a training-free framework designed to extract semantic-discriminative features from DiTs. Specifically, DiTF employs AdaLN to adaptively localize and normalize massive activations with channel-wise modulation. In addition, we develop a channel discard strategy to further eliminate the negative impacts from massive activations. Experimental results demonstrate that our DiTF outperforms both DINO and SD-based models and establishes a new state-of-the-art performance for DiTs in different visual correspondence tasks (\eg, with +9.4\% on Spair-71k and +4.4\% on AP-10K-C.S.).
>
---
#### [new 047] PHI: Bridging Domain Shift in Long-Term Action Quality Assessment via Progressive Hierarchical Instruction
- **分类: cs.CV**

- **简介: 该论文属于长期动作质量评估（AQA）任务，旨在解决预训练模型与AQA任务间的领域迁移问题。针对特征级和任务级领域偏移，提出PHI框架：通过渐进流匹配（GMF）减少特征差异，结合时序注意力模块捕捉长依赖，并用列表对比正则化（LCR）对齐特征，提升长期视频动作质量评估性能。**

- **链接: [http://arxiv.org/pdf/2505.19972v1](http://arxiv.org/pdf/2505.19972v1)**

> **作者:** Kanglei Zhou; Hubert P. H. Shum; Frederick W. B. Li; Xingxing Zhang; Xiaohui Liang
>
> **备注:** Accepted by IEEE Transactions on Image Processing
>
> **摘要:** Long-term Action Quality Assessment (AQA) aims to evaluate the quantitative performance of actions in long videos. However, existing methods face challenges due to domain shifts between the pre-trained large-scale action recognition backbones and the specific AQA task, thereby hindering their performance. This arises since fine-tuning resource-intensive backbones on small AQA datasets is impractical. We address this by identifying two levels of domain shift: task-level, regarding differences in task objectives, and feature-level, regarding differences in important features. For feature-level shifts, which are more detrimental, we propose Progressive Hierarchical Instruction (PHI) with two strategies. First, Gap Minimization Flow (GMF) leverages flow matching to progressively learn a fast flow path that reduces the domain gap between initial and desired features across shallow to deep layers. Additionally, a temporally-enhanced attention module captures long-range dependencies essential for AQA. Second, List-wise Contrastive Regularization (LCR) facilitates coarse-to-fine alignment by comprehensively comparing batch pairs to learn fine-grained cues while mitigating domain shift. Integrating these modules, PHI offers an effective solution. Experiments demonstrate that PHI achieves state-of-the-art performance on three representative long-term AQA datasets, proving its superiority in addressing the domain shift for long-term AQA.
>
---
#### [new 048] Burst Image Super-Resolution via Multi-Cross Attention Encoding and Multi-Scan State-Space Decoding
- **分类: cs.CV**

- **简介: 该论文属于burst图像超分辨率任务，针对传统注意力机制视野局限导致特征对齐与聚合不足的问题，提出重叠交叉窗口注意力和跨帧注意力机制，并设计多扫描状态空间模块，提升多帧子像素信息精准提取与高效融合。**

- **链接: [http://arxiv.org/pdf/2505.19668v1](http://arxiv.org/pdf/2505.19668v1)**

> **作者:** Tengda Huang; Yu Zhang; Tianren Li; Yufu Qu; Fulin Liu; Zhenzhong Wei
>
> **备注:** 32 pages, 13 figures, submitted to 'Image and Vision Computing'
>
> **摘要:** Multi-image super-resolution (MISR) can achieve higher image quality than single-image super-resolution (SISR) by aggregating sub-pixel information from multiple spatially shifted frames. Among MISR tasks, burst super-resolution (BurstSR) has gained significant attention due to its wide range of applications. Recent methods have increasingly adopted Transformers over convolutional neural networks (CNNs) in super-resolution tasks, due to their superior ability to capture both local and global context. However, most existing approaches still rely on fixed and narrow attention windows that restrict the perception of features beyond the local field. This limitation hampers alignment and feature aggregation, both of which are crucial for high-quality super-resolution. To address these limitations, we propose a novel feature extractor that incorporates two newly designed attention mechanisms: overlapping cross-window attention and cross-frame attention, enabling more precise and efficient extraction of sub-pixel information across multiple frames. Furthermore, we introduce a Multi-scan State-Space Module with the cross-frame attention mechanism to enhance feature aggregation. Extensive experiments on both synthetic and real-world benchmarks demonstrate the superiority of our approach. Additional evaluations on ISO 12233 resolution test charts further confirm its enhanced super-resolution performance.
>
---
#### [new 049] The Missing Point in Vision Transformers for Universal Image Segmentation
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文属于图像分割任务，针对现有方法在模糊边界和类别不平衡下的分类精度不足问题，提出ViT-P框架：分两阶段处理，先生成无类别掩码提案，再通过基于ViT的中心点分类模型优化预测，支持预训练模型免改造适配，利用粗标注降低成本，实现多数据集SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.19795v1](http://arxiv.org/pdf/2505.19795v1)**

> **作者:** Sajjad Shahabodini; Mobina Mansoori; Farnoush Bayatmakou; Jamshid Abouei; Konstantinos N. Plataniotis; Arash Mohammadi
>
> **摘要:** Image segmentation remains a challenging task in computer vision, demanding robust mask generation and precise classification. Recent mask-based approaches yield high-quality masks by capturing global context. However, accurately classifying these masks, especially in the presence of ambiguous boundaries and imbalanced class distributions, remains an open challenge. In this work, we introduce ViT-P, a novel two-stage segmentation framework that decouples mask generation from classification. The first stage employs a proposal generator to produce class-agnostic mask proposals, while the second stage utilizes a point-based classification model built on the Vision Transformer (ViT) to refine predictions by focusing on mask central points. ViT-P serves as a pre-training-free adapter, allowing the integration of various pre-trained vision transformers without modifying their architecture, ensuring adaptability to dense prediction tasks. Furthermore, we demonstrate that coarse and bounding box annotations can effectively enhance classification without requiring additional training on fine annotation datasets, reducing annotation costs while maintaining strong performance. Extensive experiments across COCO, ADE20K, and Cityscapes datasets validate the effectiveness of ViT-P, achieving state-of-the-art results with 54.0 PQ on ADE20K panoptic segmentation, 87.4 mIoU on Cityscapes semantic segmentation, and 63.6 mIoU on ADE20K semantic segmentation. The code and pretrained models are available at: https://github.com/sajjad-sh33/ViT-P}{https://github.com/sajjad-sh33/ViT-P.
>
---
#### [new 050] Saliency-guided Emotion Modeling: Predicting Viewer Reactions from Video Stimuli
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视频驱动情绪预测任务，旨在解决传统方法忽视视觉显著性的问题。通过提取视频的显著区域面积和数量，结合HD2S和OpenFace模型分析，揭示显著性分布与情绪（愉悦度/唤醒度）的关联规律，提出基于视觉显著性的高效可解释情绪建模新方法。**

- **链接: [http://arxiv.org/pdf/2505.19178v1](http://arxiv.org/pdf/2505.19178v1)**

> **作者:** Akhila Yaragoppa; Siddharth
>
> **备注:** Accepted for publication at IBPRIA 2025 Conference in Coimbra, Portugal
>
> **摘要:** Understanding the emotional impact of videos is crucial for applications in content creation, advertising, and Human-Computer Interaction (HCI). Traditional affective computing methods rely on self-reported emotions, facial expression analysis, and biosensing data, yet they often overlook the role of visual saliency -- the naturally attention-grabbing regions within a video. In this study, we utilize deep learning to introduce a novel saliency-based approach to emotion prediction by extracting two key features: saliency area and number of salient regions. Using the HD2S saliency model and OpenFace facial action unit analysis, we examine the relationship between video saliency and viewer emotions. Our findings reveal three key insights: (1) Videos with multiple salient regions tend to elicit high-valence, low-arousal emotions, (2) Videos with a single dominant salient region are more likely to induce low-valence, high-arousal responses, and (3) Self-reported emotions often misalign with facial expression-based emotion detection, suggesting limitations in subjective reporting. By leveraging saliency-driven insights, this work provides a computationally efficient and interpretable alternative for emotion modeling, with implications for content creation, personalized media experiences, and affective computing research.
>
---
#### [new 051] Beyond Segmentation: Confidence-Aware and Debiased Estimation of Ratio-based Biomarkers
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，针对现有比率型生物标志物（如肿瘤坏死比例）缺乏不确定性量化的问题，提出置信感知框架。通过分析误差传播，发现模型校准是主因，引入轻量级后校准模块及可调参数Q生成可信置信区间，提升临床决策可靠性。**

- **链接: [http://arxiv.org/pdf/2505.19585v1](http://arxiv.org/pdf/2505.19585v1)**

> **作者:** Jiameng Li; Teodora Popordanoska; Sebastian G. Gruber; Frederik Maes; Matthew B. Blaschko
>
> **备注:** 9 pages
>
> **摘要:** Ratio-based biomarkers -- such as the proportion of necrotic tissue within a tumor -- are widely used in clinical practice to support diagnosis, prognosis and treatment planning. These biomarkers are typically estimated from soft segmentation outputs by computing region-wise ratios. Despite the high-stakes nature of clinical decision making, existing methods provide only point estimates, offering no measure of uncertainty. In this work, we propose a unified \textit{confidence-aware} framework for estimating ratio-based biomarkers. We conduct a systematic analysis of error propagation in the segmentation-to-biomarker pipeline and identify model miscalibration as the dominant source of uncertainty. To mitigate this, we incorporate a lightweight, post-hoc calibration module that can be applied using internal hospital data without retraining. We leverage a tunable parameter $Q$ to control the confidence level of the derived bounds, allowing adaptation towards clinical practice. Extensive experiments show that our method produces statistically sound confidence intervals, with tunable confidence levels, enabling more trustworthy application of predictive biomarkers in clinical workflows.
>
---
#### [new 052] CENet: Context Enhancement Network for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学影像分割任务，旨在解决多领域场景下边界不精确、器官形态变异及下采样信息丢失问题。提出CENet框架，包含Dual Selective Enhancement Block（增强边界与小器官检测）和Context Feature Attention Module（多尺度设计减少冗余并保持空间完整性），实验显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.18423v1](http://arxiv.org/pdf/2505.18423v1)**

> **作者:** Afshin Bozorgpour; Sina Ghorbani Kolahi; Reza Azad; Ilker Hacihaliloglu; Dorit Merhof
>
> **备注:** Provisionally accepted at MICCAI-2025
>
> **摘要:** Medical image segmentation, particularly in multi-domain scenarios, requires precise preservation of anatomical structures across diverse representations. While deep learning has advanced this field, existing models often struggle with accurate boundary representation, variability in organ morphology, and information loss during downsampling, limiting their accuracy and robustness. To address these challenges, we propose the Context Enhancement Network (CENet), a novel segmentation framework featuring two key innovations. First, the Dual Selective Enhancement Block (DSEB) integrated into skip connections enhances boundary details and improves the detection of smaller organs in a context-aware manner. Second, the Context Feature Attention Module (CFAM) in the decoder employs a multi-scale design to maintain spatial integrity, reduce feature redundancy, and mitigate overly enhanced representations. Extensive evaluations on both radiology and dermoscopic datasets demonstrate that CENet outperforms state-of-the-art (SOTA) methods in multi-organ segmentation and boundary detail preservation, offering a robust and accurate solution for complex medical image analysis tasks. The code is publicly available at https://github.com/xmindflow/cenet.
>
---
#### [new 053] OmniFall: A Unified Staged-to-Wild Benchmark for Human Fall Detection
- **分类: cs.CV; I.2.10; I.5.4**

- **简介: 该论文提出OmniFall基准，整合8个跌倒检测数据集，解决现有小规模人工场景数据集领域偏差大、无法评估真实性能的问题。工作包括建立统一分类与评估协议、创建真实事故数据集OOPS-Fall，实验显示模型在真实场景性能显著下降，强调提升鲁棒性的必要性。**

- **链接: [http://arxiv.org/pdf/2505.19889v1](http://arxiv.org/pdf/2505.19889v1)**

> **作者:** David Schneider; Zdravko Marinov; Rafael Baur; Zeyun Zhong; Rodi Düger; Rainer Stiefelhagen
>
> **摘要:** Current video-based fall detection research mostly relies on small, staged datasets with significant domain biases concerning background, lighting, and camera setup resulting in unknown real-world performance. We introduce OmniFall, unifying eight public fall detection datasets (roughly 14 h of recordings, roughly 42 h of multiview data, 101 subjects, 29 camera views) under a consistent ten-class taxonomy with standardized evaluation protocols. Our benchmark provides complete video segmentation labels and enables fair cross-dataset comparison previously impossible with incompatible annotation schemes. For real-world evaluation we curate OOPS-Fall from genuine accident videos and establish a staged-to-wild protocol measuring generalization from controlled to uncontrolled environments. Experiments with frozen pre-trained backbones such as I3D or VideoMAE reveal significant performance gaps between in-distribution and in-the-wild scenarios, highlighting critical challenges in developing robust fall detection systems. OmniFall Dataset at https://huggingface.co/datasets/simplexsigil2/omnifall , Code at https://github.com/simplexsigil/omnifall-experiments
>
---
#### [new 054] NTIRE 2025 Challenge on Video Quality Enhancement for Video Conferencing: Datasets, Methods and Results
- **分类: cs.CV**

- **简介: 该论文介绍NTIRE 2025视频会议质量增强挑战赛，属视频质量增强（VQE）任务，旨在通过提升光照、颜色、降噪和锐度改善视频会议画质。工作包括构建数据集、提供可微分质量评估模型，组织竞赛并分析10个有效方案的众包评估结果。**

- **链接: [http://arxiv.org/pdf/2505.18988v1](http://arxiv.org/pdf/2505.18988v1)**

> **作者:** Varun Jain; Zongwei Wu; Quan Zou; Louis Florentin; Henrik Turbell; Sandeep Siddhartha; Radu Timofte; others
>
> **摘要:** This paper presents a comprehensive review of the 1st Challenge on Video Quality Enhancement for Video Conferencing held at the NTIRE workshop at CVPR 2025, and highlights the problem statement, datasets, proposed solutions, and results. The aim of this challenge was to design a Video Quality Enhancement (VQE) model to enhance video quality in video conferencing scenarios by (a) improving lighting, (b) enhancing colors, (c) reducing noise, and (d) enhancing sharpness - giving a professional studio-like effect. Participants were given a differentiable Video Quality Assessment (VQA) model, training, and test videos. A total of 91 participants registered for the challenge. We received 10 valid submissions that were evaluated in a crowdsourced framework.
>
---
#### [new 055] Zero-Shot Pseudo Labels Generation Using SAM and CLIP for Semi-Supervised Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于半监督语义分割任务，旨在解决标注成本高的问题。提出结合SAM和CLIP生成零样本伪标签，并通过UniMatch优化标签质量，提升模型训练效果，实验在PASCAL和MS COCO数据集验证方法有效性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19846v1](http://arxiv.org/pdf/2505.19846v1)**

> **作者:** Nagito Saito; Shintaro Ito; Koichi Ito; Takafumi Aoki
>
> **备注:** Accepted to ICIP 2025
>
> **摘要:** Semantic segmentation is a fundamental task in medical image analysis and autonomous driving and has a problem with the high cost of annotating the labels required in training. To address this problem, semantic segmentation methods based on semi-supervised learning with a small number of labeled data have been proposed. For example, one approach is to train a semantic segmentation model using images with annotated labels and pseudo labels. In this approach, the accuracy of the semantic segmentation model depends on the quality of the pseudo labels, and the quality of the pseudo labels depends on the performance of the model to be trained and the amount of data with annotated labels. In this paper, we generate pseudo labels using zero-shot annotation with the Segment Anything Model (SAM) and Contrastive Language-Image Pretraining (CLIP), improve the accuracy of the pseudo labels using the Unified Dual-Stream Perturbations Approach (UniMatch), and use them as enhanced labels to train a semantic segmentation model. The effectiveness of the proposed method is demonstrated through the experiments using the public datasets: PASCAL and MS COCO.
>
---
#### [new 056] Deformable Attentive Visual Enhancement for Referring Segmentation Using Vision-Language Model
- **分类: cs.CV**

- **简介: 该论文属于指针式图像分割任务，旨在通过自然语言表达定位目标对象，解决视觉与语言信息融合及分割精度问题。提出SegVLM模型，融合SE块、可变形卷积与残差连接优化特征学习，设计新型RAF损失函数平衡区域对齐、边界精度与类别不平衡，实验验证各组件有效性及模型泛化性。**

- **链接: [http://arxiv.org/pdf/2505.19242v1](http://arxiv.org/pdf/2505.19242v1)**

> **作者:** Alaa Dalaq; Muzammil Behzad
>
> **摘要:** Image segmentation is a fundamental task in computer vision, aimed at partitioning an image into semantically meaningful regions. Referring image segmentation extends this task by using natural language expressions to localize specific objects, requiring effective integration of visual and linguistic information. In this work, we propose SegVLM, a vision-language model that incorporates architectural improvements to enhance segmentation accuracy and cross-modal alignment. The model integrates squeeze-and-excitation (SE) blocks for dynamic feature recalibration, deformable convolutions for geometric adaptability, and residual connections for deep feature learning. We also introduce a novel referring-aware fusion (RAF) loss that balances region-level alignment, boundary precision, and class imbalance. Extensive experiments and ablation studies demonstrate that each component contributes to consistent performance improvements. SegVLM also shows strong generalization across diverse datasets and referring expression scenarios.
>
---
#### [new 057] What You Perceive Is What You Conceive: A Cognition-Inspired Framework for Open Vocabulary Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于开放词汇图像分割任务，旨在解决现有方法中区域分割与目标概念对齐不足的问题。提出受认知启发的框架，通过生成视觉语言模型、概念增强模块及解码器，模仿人类先理解概念再感知物体的流程，提升分割精度与灵活性，实现跨数据集性能提升。**

- **链接: [http://arxiv.org/pdf/2505.19569v1](http://arxiv.org/pdf/2505.19569v1)**

> **作者:** Jianghang Lin; Yue Hu; Jiangtao Shen; Yunhang Shen; Liujuan Cao; Shengchuan Zhang; Rongrong Ji
>
> **摘要:** Open vocabulary image segmentation tackles the challenge of recognizing dynamically adjustable, predefined novel categories at inference time by leveraging vision-language alignment. However, existing paradigms typically perform class-agnostic region segmentation followed by category matching, which deviates from the human visual system's process of recognizing objects based on semantic concepts, leading to poor alignment between region segmentation and target concepts. To bridge this gap, we propose a novel Cognition-Inspired Framework for open vocabulary image segmentation that emulates the human visual recognition process: first forming a conceptual understanding of an object, then perceiving its spatial extent. The framework consists of three core components: (1) A Generative Vision-Language Model (G-VLM) that mimics human cognition by generating object concepts to provide semantic guidance for region segmentation. (2) A Concept-Aware Visual Enhancer Module that fuses textual concept features with global visual representations, enabling adaptive visual perception based on target concepts. (3) A Cognition-Inspired Decoder that integrates local instance features with G-VLM-provided semantic cues, allowing selective classification over a subset of relevant categories. Extensive experiments demonstrate that our framework achieves significant improvements, reaching $27.2$ PQ, $17.0$ mAP, and $35.3$ mIoU on A-150. It further attains $56.2$, $28.2$, $15.4$, $59.2$, $18.7$, and $95.8$ mIoU on Cityscapes, Mapillary Vistas, A-847, PC-59, PC-459, and PAS-20, respectively. In addition, our framework supports vocabulary-free segmentation, offering enhanced flexibility in recognizing unseen categories. Code will be public.
>
---
#### [new 058] Medical Large Vision Language Models with Multi-Image Visual Ability
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦医学多图像视觉语言模型任务，解决现有模型处理多图像临床场景（如时间推理、跨图像比较）能力不足的问题。工作包括构建含8.3万组多图像QA的Med-MIM数据集，微调Mantis和LLaVA-Med模型，开发评估基准，并验证新模型在医学多图像理解上的优越性。**

- **链接: [http://arxiv.org/pdf/2505.19031v1](http://arxiv.org/pdf/2505.19031v1)**

> **作者:** Xikai Yang; Juzheng Miao; Yuchen Yuan; Jiaze Wang; Qi Dou; Jinpeng Li; Pheng-Ann Heng
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Medical large vision-language models (LVLMs) have demonstrated promising performance across various single-image question answering (QA) benchmarks, yet their capability in processing multi-image clinical scenarios remains underexplored. Unlike single image based tasks, medical tasks involving multiple images often demand sophisticated visual understanding capabilities, such as temporal reasoning and cross-modal analysis, which are poorly supported by current medical LVLMs. To bridge this critical gap, we present the Med-MIM instruction dataset, comprising 83.2K medical multi-image QA pairs that span four types of multi-image visual abilities (temporal understanding, reasoning, comparison, co-reference). Using this dataset, we fine-tune Mantis and LLaVA-Med, resulting in two specialized medical VLMs: MIM-LLaVA-Med and Med-Mantis, both optimized for multi-image analysis. Additionally, we develop the Med-MIM benchmark to comprehensively evaluate the medical multi-image understanding capabilities of LVLMs. We assess eight popular LVLMs, including our two models, on the Med-MIM benchmark. Experimental results show that both Med-Mantis and MIM-LLaVA-Med achieve superior performance on the held-in and held-out subsets of the Med-MIM benchmark, demonstrating that the Med-MIM instruction dataset effectively enhances LVLMs' multi-image understanding capabilities in the medical domain.
>
---
#### [new 059] VPGS-SLAM: Voxel-based Progressive 3D Gaussian SLAM in Large-Scale Scenes
- **分类: cs.CV**

- **简介: 该论文提出VPGS-SLAM，解决3DGS在大规模场景中内存爆炸和位姿漂移问题。采用体素渐进式多子图映射、2D-3D融合追踪及闭环方法，实现室内外大场景鲁棒SLAM，通过在线蒸馏子图融合保证全局一致性。**

- **链接: [http://arxiv.org/pdf/2505.18992v1](http://arxiv.org/pdf/2505.18992v1)**

> **作者:** Tianchen Deng; Wenhua Wu; Junjie He; Yue Pan; Xirui Jiang; Shenghai Yuan; Danwei Wang; Hesheng Wang; Weidong Chen
>
> **摘要:** 3D Gaussian Splatting has recently shown promising results in dense visual SLAM. However, existing 3DGS-based SLAM methods are all constrained to small-room scenarios and struggle with memory explosion in large-scale scenes and long sequences. To this end, we propose VPGS-SLAM, the first 3DGS-based large-scale RGBD SLAM framework for both indoor and outdoor scenarios. We design a novel voxel-based progressive 3D Gaussian mapping method with multiple submaps for compact and accurate scene representation in large-scale and long-sequence scenes. This allows us to scale up to arbitrary scenes and improves robustness (even under pose drifts). In addition, we propose a 2D-3D fusion camera tracking method to achieve robust and accurate camera tracking in both indoor and outdoor large-scale scenes. Furthermore, we design a 2D-3D Gaussian loop closure method to eliminate pose drift. We further propose a submap fusion method with online distillation to achieve global consistency in large-scale scenes when detecting a loop. Experiments on various indoor and outdoor datasets demonstrate the superiority and generalizability of the proposed framework. The code will be open source on https://github.com/dtc111111/vpgs-slam.
>
---
#### [new 060] DriveCamSim: Generalizable Camera Simulation via Explicit Camera Modeling for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出DriveCamSim框架，解决现有相机模拟受限于固定视角和帧率的问题。通过显式相机建模（ECM）建立像素级跨视图与时序对应，解耦模型对特定参数依赖，并优化控制机制，提升可控性与时序一致性，实现灵活适应不同场景的高质量自动驾驶相机模拟。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19692v1](http://arxiv.org/pdf/2505.19692v1)**

> **作者:** Wenchao Sun; Xuewu Lin; Keyu Chen; Zixiang Pei; Yining Shi; Chuang Zhang; Sifa Zheng
>
> **摘要:** Camera sensor simulation serves as a critical role for autonomous driving (AD), e.g. evaluating vision-based AD algorithms. While existing approaches have leveraged generative models for controllable image/video generation, they remain constrained to generating multi-view video sequences with fixed camera viewpoints and video frequency, significantly limiting their downstream applications. To address this, we present a generalizable camera simulation framework DriveCamSim, whose core innovation lies in the proposed Explicit Camera Modeling (ECM) mechanism. Instead of implicit interaction through vanilla attention, ECM establishes explicit pixel-wise correspondences across multi-view and multi-frame dimensions, decoupling the model from overfitting to the specific camera configurations (intrinsic/extrinsic parameters, number of views) and temporal sampling rates presented in the training data. For controllable generation, we identify the issue of information loss inherent in existing conditional encoding and injection pipelines, proposing an information-preserving control mechanism. This control mechanism not only improves conditional controllability, but also can be extended to be identity-aware to enhance temporal consistency in foreground object rendering. With above designs, our model demonstrates superior performance in both visual quality and controllability, as well as generalization capability across spatial-level (camera parameters variations) and temporal-level (video frame rate variations), enabling flexible user-customizable camera simulation tailored to diverse application scenarios. Code will be avaliable at https://github.com/swc-17/DriveCamSim for facilitating future research.
>
---
#### [new 061] Veta-GS: View-dependent deformable 3D Gaussian Splatting for thermal infrared Novel-view Synthesis
- **分类: cs.CV**

- **简介: 该论文属于热红外新型视图合成任务，针对传输效应、低分辨率导致的浮动物和模糊问题，提出Veta-GS方法。其通过视图相关变形场捕捉热变化，结合热特征提取器（TFE）和MonoSSIM损失优化外观、边缘与频率，提升合成图像的鲁棒性，实验验证优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19138v1](http://arxiv.org/pdf/2505.19138v1)**

> **作者:** Myeongseok Nam; Wongi Park; Minsol Kim; Hyejin Hur; Soomok Lee
>
> **摘要:** Recently, 3D Gaussian Splatting (3D-GS) based on Thermal Infrared (TIR) imaging has gained attention in novel-view synthesis, showing real-time rendering. However, novel-view synthesis with thermal infrared images suffers from transmission effects, emissivity, and low resolution, leading to floaters and blur effects in rendered images. To address these problems, we introduce Veta-GS, which leverages a view-dependent deformation field and a Thermal Feature Extractor (TFE) to precisely capture subtle thermal variations and maintain robustness. Specifically, we design view-dependent deformation field that leverages camera position and viewing direction, which capture thermal variations. Furthermore, we introduce the Thermal Feature Extractor (TFE) and MonoSSIM loss, which consider appearance, edge, and frequency to maintain robustness. Extensive experiments on the TI-NSD benchmark show that our method achieves better performance over existing methods.
>
---
#### [new 062] Underwater Diffusion Attention Network with Contrastive Language-Image Joint Learning for Underwater Image Enhancement
- **分类: cs.CV**

- **简介: 该论文提出UDAN-CLIP模型，针对水下图像增强任务，解决合成数据偏差与领域偏移导致的不真实增强问题。方法结合扩散网络、视觉语言分类器、空间注意力模块及CLIP-Diffusion损失，提升图像自然性与细节保留，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.19895v1](http://arxiv.org/pdf/2505.19895v1)**

> **作者:** Afrah Shaahid; Muzammil Behzad
>
> **摘要:** Underwater images are often affected by complex degradations such as light absorption, scattering, color casts, and artifacts, making enhancement critical for effective object detection, recognition, and scene understanding in aquatic environments. Existing methods, especially diffusion-based approaches, typically rely on synthetic paired datasets due to the scarcity of real underwater references, introducing bias and limiting generalization. Furthermore, fine-tuning these models can degrade learned priors, resulting in unrealistic enhancements due to domain shifts. To address these challenges, we propose UDAN-CLIP, an image-to-image diffusion framework pre-trained on synthetic underwater datasets and enhanced with a customized classifier based on vision-language model, a spatial attention module, and a novel CLIP-Diffusion loss. The classifier preserves natural in-air priors and semantically guides the diffusion process, while the spatial attention module focuses on correcting localized degradations such as haze and low contrast. The proposed CLIP-Diffusion loss further strengthens visual-textual alignment and helps maintain semantic consistency during enhancement. The proposed contributions empower our UDAN-CLIP model to perform more effective underwater image enhancement, producing results that are not only visually compelling but also more realistic and detail-preserving. These improvements are consistently validated through both quantitative metrics and qualitative visual comparisons, demonstrating the model's ability to correct distortions and restore natural appearance in challenging underwater conditions.
>
---
#### [new 063] Spiking Transformers Need High Frequency Information
- **分类: cs.CV**

- **简介: 该论文针对脉冲变压器性能不足问题，揭示高频信息衰减是主因，提出Max-Former通过Max-Pooling和深度卷积增强高频信号，显著提升Cifar-100和ImageNet准确率，推动脉冲神经网络研究。**

- **链接: [http://arxiv.org/pdf/2505.18608v1](http://arxiv.org/pdf/2505.18608v1)**

> **作者:** Yuetong Fang; Deming Zhou; Ziqing Wang; Hongwei Ren; ZeCui Zeng; Lusong Li; Shibo Zhou; Renjing Xu
>
> **摘要:** Spiking Transformers offer an energy-efficient alternative to conventional deep learning by transmitting information solely through binary (0/1) spikes. However, there remains a substantial performance gap compared to artificial neural networks. A common belief is that their binary and sparse activation transmission leads to information loss, thus degrading feature representation and accuracy. In this work, however, we reveal for the first time that spiking neurons preferentially propagate low-frequency information. We hypothesize that the rapid dissipation of high-frequency components is the primary cause of performance degradation. For example, on Cifar-100, adopting Avg-Pooling (low-pass) for token mixing lowers performance to 76.73%; interestingly, replacing it with Max-Pooling (high-pass) pushes the top-1 accuracy to 79.12%, surpassing the well-tuned Spikformer baseline by 0.97%. Accordingly, we introduce Max-Former that restores high-frequency signals through two frequency-enhancing operators: extra Max-Pooling in patch embedding and Depth-Wise Convolution in place of self-attention. Notably, our Max-Former (63.99 M) hits the top-1 accuracy of 82.39% on ImageNet, showing a +7.58% improvement over Spikformer with comparable model size (74.81%, 66.34 M). We hope this simple yet effective solution inspires future research to explore the distinctive nature of spiking neural networks, beyond the established practice in standard deep learning.
>
---
#### [new 064] Two Causally Related Needles in a Video Haystack
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出视频语言模型评估基准Causal2Needles，解决现有方法无法测试模型在长视频中跨时空关联因果事件及整合多位置信息的问题。通过设计需联合分析因果行为的"双针"任务，揭示当前VLMs在处理远距离因果关系时的性能缺陷。**

- **链接: [http://arxiv.org/pdf/2505.19853v1](http://arxiv.org/pdf/2505.19853v1)**

> **作者:** Miaoyu Li; Qin Chao; Boyang Li
>
> **摘要:** Evaluating the video understanding capabilities of Video-Language Models (VLMs) remains a significant challenge. We propose a long-context video understanding benchmark, Causal2Needles, that assesses two crucial abilities insufficiently evaluated by existing benchmarks: (1) the ability to extract information from two separate locations in a long video and understand them jointly, and (2) the ability to model the world in terms of cause and effect in human behaviors. Specifically, Causal2Needles introduces 2-needle questions, which require extracting information from both the cause and effect human-behavior events in a long video and the associated narration text. To prevent textual bias, these questions comprise two complementary formats: one asking to identify the video clip containing the answer, and one asking for the textual description of an unrelated visual detail from that video clip. Our experiments reveal that models excelling in pre-existing benchmarks struggle with 2-needle visual grounding, and the model performance is negatively correlated with the distance between the two needles. These findings highlight critical limitations in current VLMs.
>
---
#### [new 065] SPARS: Self-Play Adversarial Reinforcement Learning for Segmentation of Liver Tumours
- **分类: cs.CV**

- **简介: 该论文提出SPARS框架，通过自博弈对抗强化学习实现弱监督肝肿瘤分割，利用少量患者级癌症存在标签替代体素级标注，在CT图像中取得77.3% Dice系数，优于其他弱监督方法，接近全监督模型，减少标注需求，提升临床检测效率。**

- **链接: [http://arxiv.org/pdf/2505.18989v1](http://arxiv.org/pdf/2505.18989v1)**

> **作者:** Catalina Tan; Yipeng Hu; Shaheer U. Saeed
>
> **备注:** Accepted at Medical Image Understanding and Analysis (MIUA) 2025
>
> **摘要:** Accurate tumour segmentation is vital for various targeted diagnostic and therapeutic procedures for cancer, e.g., planning biopsies or tumour ablations. Manual delineation is extremely labour-intensive, requiring substantial expert time. Fully-supervised machine learning models aim to automate such localisation tasks, but require a large number of costly and often subjective 3D voxel-level labels for training. The high-variance and subjectivity in such labels impacts model generalisability, even when large datasets are available. Histopathology labels may offer more objective labels but the infeasibility of acquiring pixel-level annotations to develop tumour localisation methods based on histology remains challenging in-vivo. In this work, we propose a novel weakly-supervised semantic segmentation framework called SPARS (Self-Play Adversarial Reinforcement Learning for Segmentation), which utilises an object presence classifier, trained on a small number of image-level binary cancer presence labels, to localise cancerous regions on CT scans. Such binary labels of patient-level cancer presence can be sourced more feasibly from biopsies and histopathology reports, enabling a more objective cancer localisation on medical images. Evaluating with real patient data, we observed that SPARS yielded a mean dice score of $77.3 \pm 9.4$, which outperformed other weakly-supervised methods by large margins. This performance was comparable with recent fully-supervised methods that require voxel-level annotations. Our results demonstrate the potential of using SPARS to reduce the need for extensive human-annotated labels to detect cancer in real-world healthcare settings.
>
---
#### [new 066] DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing
- **分类: cs.CV**

- **简介: 该论文属于红外小目标分解任务，解决紧密排列小目标信号重叠导致的检测难题。提出DISTA-Net模型，通过动态生成卷积权重和阈值参数实现亚像素级精准分离，并构建开源生态系统（含数据集、评估指标和工具包），首次建立深度学习方法处理该问题的框架。**

- **链接: [http://arxiv.org/pdf/2505.19148v1](http://arxiv.org/pdf/2505.19148v1)**

> **作者:** Shengdong Han; Shangdong Yang; Xin Zhang; Yuxuan Li; Xiang Li; Jian Yang; Ming-Ming Cheng; Yimian Dai
>
> **摘要:** Resolving closely-spaced small targets in dense clusters presents a significant challenge in infrared imaging, as the overlapping signals hinder precise determination of their quantity, sub-pixel positions, and radiation intensities. While deep learning has advanced the field of infrared small target detection, its application to closely-spaced infrared small targets has not yet been explored. This gap exists primarily due to the complexity of separating superimposed characteristics and the lack of an open-source infrastructure. In this work, we propose the Dynamic Iterative Shrinkage Thresholding Network (DISTA-Net), which reconceptualizes traditional sparse reconstruction within a dynamic framework. DISTA-Net adaptively generates convolution weights and thresholding parameters to tailor the reconstruction process in real time. To the best of our knowledge, DISTA-Net is the first deep learning model designed specifically for the unmixing of closely-spaced infrared small targets, achieving superior sub-pixel detection accuracy. Moreover, we have established the first open-source ecosystem to foster further research in this field. This ecosystem comprises three key components: (1) CSIST-100K, a publicly available benchmark dataset; (2) CSO-mAP, a custom evaluation metric for sub-pixel detection; and (3) GrokCSO, an open-source toolkit featuring DISTA-Net and other models. Our code and dataset are available at https://github.com/GrokCV/GrokCSO.
>
---
#### [new 067] Toward Patient-specific Partial Point Cloud to Surface Completion for Pre- to Intra-operative Registration in Image-guided Liver Interventions
- **分类: cs.CV**

- **简介: 该论文提出患者特异性点云补全方法，解决术中点云部分可见导致的术前-术中配准难题。通过VN-OccNet生成完整肝表面（利用术前模拟形变训练），并集成到Go-ICP算法提升初始刚性配准精度，验证旋转等变特性与表面重建对配准的改进效果。**

- **链接: [http://arxiv.org/pdf/2505.19518v1](http://arxiv.org/pdf/2505.19518v1)**

> **作者:** Nakul Poudel; Zixin Yang; Kelly Merrell; Richard Simon; Cristian A. Linte
>
> **摘要:** Intra-operative data captured during image-guided surgery lacks sub-surface information, where key regions of interest, such as vessels and tumors, reside. Image-to-physical registration enables the fusion of pre-operative information and intra-operative data, typically represented as a point cloud. However, this registration process struggles due to partial visibility of the intra-operative point cloud. In this research, we propose a patient-specific point cloud completion approach to assist with the registration process. Specifically, we leverage VN-OccNet to generate a complete liver surface from a partial intra-operative point cloud. The network is trained in a patient-specific manner, where simulated deformations from the pre-operative model are used to train the model. First, we conduct an in-depth analysis of VN-OccNet's rotation-equivariant property and its effectiveness in recovering complete surfaces from partial intra-operative surfaces. Next, we integrate the completed intra-operative surface into the Go-ICP registration algorithm to demonstrate its utility in improving initial rigid registration outcomes. Our results highlight the promise of this patient-specific completion approach in mitigating the challenges posed by partial intra-operative visibility. The rotation equivariant and surface generation capabilities of VN-OccNet hold strong promise for developing robust registration frameworks for variations of the intra-operative point cloud.
>
---
#### [new 068] PolyPose: Localizing Deformable Anatomy in 3D from Sparse 2D X-ray Images using Polyrigid Transforms
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于医学影像2D/3D配准任务，解决介入手术中仅凭少量X光片难以准确定位3D解剖结构的问题。提出PolyPose方法，通过将解剖变形建模为分段刚性变换组合，利用骨骼运动的生物约束，实现稀疏视角下精准配准，避免传统方法对复杂正则化的依赖。**

- **链接: [http://arxiv.org/pdf/2505.19256v1](http://arxiv.org/pdf/2505.19256v1)**

> **作者:** Vivek Gopalakrishnan; Neel Dey; Polina Golland
>
> **摘要:** Determining the 3D pose of a patient from a limited set of 2D X-ray images is a critical task in interventional settings. While preoperative volumetric imaging (e.g., CT and MRI) provides precise 3D localization and visualization of anatomical targets, these modalities cannot be acquired during procedures, where fast 2D imaging (X-ray) is used instead. To integrate volumetric guidance into intraoperative procedures, we present PolyPose, a simple and robust method for deformable 2D/3D registration. PolyPose parameterizes complex 3D deformation fields as a composition of rigid transforms, leveraging the biological constraint that individual bones do not bend in typical motion. Unlike existing methods that either assume no inter-joint movement or fail outright in this under-determined setting, our polyrigid formulation enforces anatomically plausible priors that respect the piecewise rigid nature of human movement. This approach eliminates the need for expensive deformation regularizers that require patient- and procedure-specific hyperparameter optimization. Across extensive experiments on diverse datasets from orthopedic surgery and radiotherapy, we show that this strong inductive bias enables PolyPose to successfully align the patient's preoperative volume to as few as two X-ray images, thereby providing crucial 3D guidance in challenging sparse-view and limited-angle settings where current registration methods fail.
>
---
#### [new 069] RTime-QA: A Benchmark for Atomic Temporal Event Understanding in Large Multi-modal Models
- **分类: cs.CV**

- **简介: 该论文提出RTime-QA基准及RTime-IT数据集，旨在评估和提升大型多模态模型（LMMs）对原子时间事件的理解能力。针对现有视频-语言任务无法有效测试时间理解的问题，构建含822个标注视频-文本问题的基准，并通过14k指令调优数据集优化模型，实验显示模型性能显著提升（从34.6%至65.9%）。**

- **链接: [http://arxiv.org/pdf/2505.19125v1](http://arxiv.org/pdf/2505.19125v1)**

> **作者:** Yuqi Liu; Qin Jin; Tianyuan Qu; Xuan Liu; Yang Du; Bei Yu; Jiaya Jia
>
> **摘要:** Understanding accurate atomic temporal event is essential for video comprehension. However, current video-language benchmarks often fall short to evaluate Large Multi-modal Models' (LMMs) temporal event understanding capabilities, as they can be effectively addressed using image-language models. In this paper, we introduce RTime-QA, a novel benchmark specifically designed to assess the atomic temporal event understanding ability of LMMs. RTime-QA comprises 822 high-quality, carefully-curated video-text questions, each meticulously annotated by human experts. Each question features a video depicting an atomic temporal event, paired with both correct answers and temporal negative descriptions, specifically designed to evaluate temporal understanding. To advance LMMs' temporal event understanding ability, we further introduce RTime-IT, a 14k instruction-tuning dataset that employs a similar annotation process as RTime-QA. Extensive experimental analysis demonstrates that RTime-QA presents a significant challenge for LMMs: the state-of-the-art model Qwen2-VL achieves only 34.6 on strict-ACC metric, substantially lagging behind human performance. Furthermore, our experiments reveal that RTime-IT effectively enhance LMMs' capacity in temporal understanding. By fine-tuning on RTime-IT, our Qwen2-VL achieves 65.9 on RTime-QA.
>
---
#### [new 070] Regularized Personalization of Text-to-Image Diffusion Models without Distributional Drift
- **分类: cs.CV**

- **简介: 该论文属于文本到图像扩散模型的个性化适配任务，旨在解决模型在少量数据适配新主题时避免遗忘原有生成能力的问题。通过分析标准训练目标与个性化目标的不匹配，提出基于Lipschitz约束的训练方法，显式限制输出分布偏移，实现在数据稀缺下保持生成多样性和高质量。**

- **链接: [http://arxiv.org/pdf/2505.19519v1](http://arxiv.org/pdf/2505.19519v1)**

> **作者:** Gihoon Kim; Hyungjin Park; Taesup Kim
>
> **摘要:** Personalization using text-to-image diffusion models involves adapting a pretrained model to novel subjects with only a few image examples. This task presents a fundamental challenge, as the model must not only learn the new subject effectively but also preserve its ability to generate diverse and coherent outputs across a wide range of prompts. In other words, successful personalization requires integrating new concepts without forgetting previously learned generative capabilities. Forgetting denotes unintended distributional drift, where the model's output distribution deviates from that of the original pretrained model. In this paper, we provide an analysis of this issue and identify a mismatch between standard training objectives and the goals of personalization. To address this, we propose a new training objective based on a Lipschitz-bounded formulation that explicitly constrains deviation from the pretrained distribution. Our method provides improved control over distributional drift and performs well even in data-scarce scenarios. Experimental results demonstrate that our approach consistently outperforms existing personalization methods, achieving higher CLIP-T, CLIP-I, and DINO scores.
>
---
#### [new 071] RAISE: Realness Assessment for Image Synthesis and Evaluation
- **分类: cs.CV; cs.AI; cs.MM; eess.IV**

- **简介: 论文提出RAISE数据集，通过人类研究评估AI生成图像的逼真度，解决其主观性评估难题。构建带真实度评分的图像数据集，并训练模型建立预测基准，验证深度模型特征可有效捕捉感知真实度。**

- **链接: [http://arxiv.org/pdf/2505.19233v1](http://arxiv.org/pdf/2505.19233v1)**

> **作者:** Aniruddha Mukherjee; Spriha Dubey; Somdyuti Paul
>
> **摘要:** The rapid advancement of generative AI has enabled the creation of highly photorealistic visual content, offering practical substitutes for real images and videos in scenarios where acquiring real data is difficult or expensive. However, reliably substituting real visual content with AI-generated counterparts requires robust assessment of the perceived realness of AI-generated visual content, a challenging task due to its inherent subjective nature. To address this, we conducted a comprehensive human study evaluating the perceptual realness of both real and AI-generated images, resulting in a new dataset, containing images paired with subjective realness scores, introduced as RAISE in this paper. Further, we develop and train multiple models on RAISE to establish baselines for realness prediction. Our experimental results demonstrate that features derived from deep foundation vision models can effectively capture the subjective realness. RAISE thus provides a valuable resource for developing robust, objective models of perceptual realness assessment.
>
---
#### [new 072] Multimodal Machine Translation with Visual Scene Graph Pruning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态机器翻译任务，针对视觉信息冗余问题，提出视觉场景图剪枝方法（PSG），利用语言场景图信息去除冗余节点以减少噪声。实验表明该方法有效，推动MMT发展。**

- **链接: [http://arxiv.org/pdf/2505.19507v1](http://arxiv.org/pdf/2505.19507v1)**

> **作者:** Chenyu Lu; Shiliang Sun; Jing Zhao; Nan Zhang; Tengfei Song; Hao Yang
>
> **摘要:** Multimodal machine translation (MMT) seeks to address the challenges posed by linguistic polysemy and ambiguity in translation tasks by incorporating visual information. A key bottleneck in current MMT research is the effective utilization of visual data. Previous approaches have focused on extracting global or region-level image features and using attention or gating mechanisms for multimodal information fusion. However, these methods have not adequately tackled the issue of visual information redundancy in MMT, nor have they proposed effective solutions. In this paper, we introduce a novel approach--multimodal machine translation with visual Scene Graph Pruning (PSG), which leverages language scene graph information to guide the pruning of redundant nodes in visual scene graphs, thereby reducing noise in downstream translation tasks. Through extensive comparative experiments with state-of-the-art methods and ablation studies, we demonstrate the effectiveness of the PSG model. Our results also highlight the promising potential of visual information pruning in advancing the field of MMT.
>
---
#### [new 073] Dynamics of Affective States During Takeover Requests in Conditionally Automated Driving Among Older Adults with and without Cognitive Impairment
- **分类: cs.CV; cs.HC**

- **简介: 该研究探讨认知障碍对老年人在有条件自动驾驶接管请求（TOR）时情绪反应的影响。通过分析面部表情的效价和唤醒度，比较认知正常与受损老年人在不同驾驶条件下的情绪差异，发现后者情绪反应较低，提示需开发适应性车辆系统以提升接管安全性。**

- **链接: [http://arxiv.org/pdf/2505.18416v1](http://arxiv.org/pdf/2505.18416v1)**

> **作者:** Gelareh Hajian; Ali Abedi; Bing Ye; Jennifer Campos; Alex Mihailidis
>
> **备注:** 16 pages, 3 figures, 2 tables
>
> **摘要:** Driving is a key component of independence and quality of life for older adults. However, cognitive decline associated with conditions such as mild cognitive impairment and dementia can compromise driving safety and often lead to premature driving cessation. Conditionally automated vehicles, which require drivers to take over control when automation reaches its operational limits, offer a potential assistive solution. However, their effectiveness depends on the driver's ability to respond to takeover requests (TORs) in a timely and appropriate manner. Understanding emotional responses during TORs can provide insight into drivers' engagement, stress levels, and readiness to resume control, particularly in cognitively vulnerable populations. This study investigated affective responses, measured via facial expression analysis of valence and arousal, during TORs among cognitively healthy older adults and those with cognitive impairment. Facial affect data were analyzed across different road geometries and speeds to evaluate within- and between-group differences in affective states. Within-group comparisons using the Wilcoxon signed-rank test revealed significant changes in valence and arousal during TORs for both groups. Cognitively healthy individuals showed adaptive increases in arousal under higher-demand conditions, while those with cognitive impairment exhibited reduced arousal and more positive valence in several scenarios. Between-group comparisons using the Mann-Whitney U test indicated that cognitively impaired individuals displayed lower arousal and higher valence than controls across different TOR conditions. These findings suggest reduced emotional response and awareness in cognitively impaired drivers, highlighting the need for adaptive vehicle systems that detect affective states and support safe handovers for vulnerable users.
>
---
#### [new 074] Category-Agnostic Neural Object Rigging
- **分类: cs.CV; I.2.10**

- **简介: 该论文提出一种无类别依赖的神经绑定方法，解决传统rigging技术依赖专家知识且难以扩展的问题。通过设计基于空间blob和实例特征体积的新型表示，自动捕捉物体变形的低维结构，实现跨类别3D姿态的直观操控，同时保留实例细节。**

- **链接: [http://arxiv.org/pdf/2505.20283v1](http://arxiv.org/pdf/2505.20283v1)**

> **作者:** Guangzhao He; Chen Geng; Shangzhe Wu; Jiajun Wu
>
> **备注:** Accepted to CVPR 2025. Project Page: https://guangzhaohe.com/canor
>
> **摘要:** The motion of deformable 4D objects lies in a low-dimensional manifold. To better capture the low dimensionality and enable better controllability, traditional methods have devised several heuristic-based methods, i.e., rigging, for manipulating dynamic objects in an intuitive fashion. However, such representations are not scalable due to the need for expert knowledge of specific categories. Instead, we study the automatic exploration of such low-dimensional structures in a purely data-driven manner. Specifically, we design a novel representation that encodes deformable 4D objects into a sparse set of spatially grounded blobs and an instance-aware feature volume to disentangle the pose and instance information of the 3D shape. With such a representation, we can manipulate the pose of 3D objects intuitively by modifying the parameters of the blobs, while preserving rich instance-specific information. We evaluate the proposed method on a variety of object categories and demonstrate the effectiveness of the proposed framework. Project page: https://guangzhaohe.com/canor
>
---
#### [new 075] LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出LangDAug，一种基于能量模型与Langevin动力学的多源领域泛化数据增强方法，用于提升医学图像分割的跨领域泛化能力。针对现有方法缺乏理论保障或泛化效果不足的问题，通过生成中间领域样本并理论证明其正则化效果，实验显示优于现有方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19659v1](http://arxiv.org/pdf/2505.19659v1)**

> **作者:** Piyush Tiwary; Kinjawl Bhattacharyya; Prathosh A. P
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Medical image segmentation models often struggle to generalize across different domains due to various reasons. Domain Generalization (DG) methods overcome this either through representation learning or data augmentation (DAug). While representation learning methods seek domain-invariant features, they often rely on ad-hoc techniques and lack formal guarantees. DAug methods, which enrich model representations through synthetic samples, have shown comparable or superior performance to representation learning approaches. We propose LangDAug, a novel $\textbf{Lang}$evin $\textbf{D}$ata $\textbf{Aug}$mentation for multi-source domain generalization in 2D medical image segmentation. LangDAug leverages Energy-Based Models (EBMs) trained via contrastive divergence to traverse between source domains, generating intermediate samples through Langevin dynamics. Theoretical analysis shows that LangDAug induces a regularization effect, and for GLMs, it upper-bounds the Rademacher complexity by the intrinsic dimensionality of the data manifold. Through extensive experiments on Fundus segmentation and 2D MRI prostate segmentation benchmarks, we show that LangDAug outperforms state-of-the-art domain generalization methods and effectively complements existing domain-randomization approaches. The codebase for our method is available at https://github.com/backpropagator/LangDAug.
>
---
#### [new 076] Towards Generalized Proactive Defense against Face Swappingwith Contour-Hybrid Watermark
- **分类: cs.CV**

- **简介: 该论文提出"轮廓混合水印"(CMark)方法，属于主动防御任务，旨在对抗未知换脸技术。针对现有检测方法依赖已知攻击或难以识别精细篡改的问题，通过在面部轮廓区域嵌入融合纹理与身份信息的水印，实现通用防御，无需预存消息或训练数据。实验显示其在8种换脸技术上表现优异，平衡了图像质量与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.19081v1](http://arxiv.org/pdf/2505.19081v1)**

> **作者:** Ruiyang Xia; Dawei Zhou; Decheng Liu; Lin Yuan; Jie Li; Nannan Wang; Xinbo Gao
>
> **备注:** 16 pages, 11 figures, under review
>
> **摘要:** Face swapping, recognized as a privacy and security concern, has prompted considerable defensive research. With the advancements in AI-generated content, the discrepancies between the real and swapped faces have become nuanced. Considering the difficulty of forged traces detection, we shift the focus to the face swapping purpose and proactively embed elaborate watermarks against unknown face swapping techniques. Given that the constant purpose is to swap the original face identity while preserving the background, we concentrate on the regions surrounding the face to ensure robust watermark generation, while embedding the contour texture and face identity information to achieve progressive image determination. The watermark is located in the facial contour and contains hybrid messages, dubbed the contour-hybrid watermark (CMark). Our approach generalizes face swapping detection without requiring any swapping techniques during training and the storage of large-scale messages in advance. Experiments conducted across 8 face swapping techniques demonstrate the superiority of our approach compared with state-of-the-art passive and proactive detectors while achieving a favorable balance between the image quality and watermark robustness.
>
---
#### [new 077] Dynamic-I2V: Exploring Image-to-Video Generaion Models via Multimodal LLM
- **分类: cs.CV**

- **简介: 该论文属于图像到视频生成任务，旨在解决现有方法在复杂动态场景中运动控制和时序连贯性不足的问题。提出Dynamic-I2V框架，结合多模态大模型与扩散Transformer提升生成质量，并创建DIVE评估基准解决现有指标偏重低动态视频的缺陷，实验显示性能提升显著。**

- **链接: [http://arxiv.org/pdf/2505.19901v1](http://arxiv.org/pdf/2505.19901v1)**

> **作者:** Peng Liu; Xiaoming Ren; Fengkai Liu; Qingsong Xie; Quanlong Zheng; Yanhao Zhang; Haonan Lu; Yujiu Yang
>
> **摘要:** Recent advancements in image-to-video (I2V) generation have shown promising performance in conventional scenarios. However, these methods still encounter significant challenges when dealing with complex scenes that require a deep understanding of nuanced motion and intricate object-action relationships. To address these challenges, we present Dynamic-I2V, an innovative framework that integrates Multimodal Large Language Models (MLLMs) to jointly encode visual and textual conditions for a diffusion transformer (DiT) architecture. By leveraging the advanced multimodal understanding capabilities of MLLMs, our model significantly improves motion controllability and temporal coherence in synthesized videos. The inherent multimodality of Dynamic-I2V further enables flexible support for diverse conditional inputs, extending its applicability to various downstream generation tasks. Through systematic analysis, we identify a critical limitation in current I2V benchmarks: a significant bias towards favoring low-dynamic videos, stemming from an inadequate balance between motion complexity and visual quality metrics. To resolve this evaluation gap, we propose DIVE - a novel assessment benchmark specifically designed for comprehensive dynamic quality measurement in I2V generation. In conclusion, extensive quantitative and qualitative experiments confirm that Dynamic-I2V attains state-of-the-art performance in image-to-video generation, particularly revealing significant improvements of 42.5%, 7.9%, and 11.8% in dynamic range, controllability, and quality, respectively, as assessed by the DIVE metric in comparison to existing methods.
>
---
#### [new 078] Deep Learning for Breast Cancer Detection: Comparative Analysis of ConvNeXT and EfficientNet
- **分类: cs.CV**

- **简介: 该论文对比ConvNeXT和EfficientNet模型在乳腺X光片乳腺癌检测中的性能，旨在通过深度学习提升早期诊断准确性。研究通过图像预处理、分类及多指标评估，发现ConvNeXT在AUC（94.33%）、准确率（93.36%）和F1分数（95.13%）上优于EfficientNet，助力早期筛查以降低死亡率。**

- **链接: [http://arxiv.org/pdf/2505.18725v1](http://arxiv.org/pdf/2505.18725v1)**

> **作者:** Mahmudul Hasan
>
> **摘要:** Breast cancer is the most commonly occurring cancer worldwide. This cancer caused 670,000 deaths globally in 2022, as reported by the WHO. Yet since health officials began routine mammography screening in age groups deemed at risk in the 1980s, breast cancer mortality has decreased by 40% in high-income nations. Every day, a greater and greater number of people are receiving a breast cancer diagnosis. Reducing cancer-related deaths requires early detection and treatment. This paper compares two convolutional neural networks called ConvNeXT and EfficientNet to predict the likelihood of cancer in mammograms from screening exams. Preprocessing of the images, classification, and performance evaluation are main parts of the whole procedure. Several evaluation metrics were used to compare and evaluate the performance of the models. The result shows that ConvNeXT generates better results with a 94.33% AUC score, 93.36% accuracy, and 95.13% F-score compared to EfficientNet with a 92.34% AUC score, 91.47% accuracy, and 93.06% F-score on RSNA screening mammography breast cancer dataset.
>
---
#### [new 079] Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation
- **分类: cs.CV**

- **简介: 该论文属于视频生成加速任务，针对Diffusion Transformers因注意力机制导致的高延迟问题。提出SVG2框架，通过语义感知排列（基于k-means的语义相似性聚类与重排token）优化稀疏注意力，减少计算浪费并提升质量。结合动态预算控制和定制内核，实现速度提升（2.3-1.89倍）同时保持PSNR达30/26。**

- **链接: [http://arxiv.org/pdf/2505.18875v1](http://arxiv.org/pdf/2505.18875v1)**

> **作者:** Shuo Yang; Haocheng Xi; Yilong Zhao; Muyang Li; Jintao Zhang; Han Cai; Yujun Lin; Xiuyu Li; Chenfeng Xu; Kelly Peng; Jianfei Chen; Song Han; Kurt Keutzer; Ion Stoica
>
> **摘要:** Diffusion Transformers (DiTs) are essential for video generation but suffer from significant latency due to the quadratic complexity of attention. By computing only critical tokens, sparse attention reduces computational costs and offers a promising acceleration approach. However, we identify that existing methods fail to approach optimal generation quality under the same computation budget for two reasons: (1) Inaccurate critical token identification: current methods cluster tokens based on position rather than semantics, leading to imprecise aggregated representations. (2) Excessive computation waste: critical tokens are scattered among non-critical ones, leading to wasted computation on GPUs, which are optimized for processing contiguous tokens. In this paper, we propose SVG2, a training-free framework that maximizes identification accuracy and minimizes computation waste, achieving a Pareto frontier trade-off between generation quality and efficiency. The core of SVG2 is semantic-aware permutation, which clusters and reorders tokens based on semantic similarity using k-means. This approach ensures both a precise cluster representation, improving identification accuracy, and a densified layout of critical tokens, enabling efficient computation without padding. Additionally, SVG2 integrates top-p dynamic budget control and customized kernel implementations, achieving up to 2.30x and 1.89x speedup while maintaining a PSNR of up to 30 and 26 on HunyuanVideo and Wan 2.1, respectively.
>
---
#### [new 080] So-Fake: Benchmarking and Explaining Social Media Image Forgery Detection
- **分类: cs.CV**

- **简介: 该论文聚焦社交媒体图像伪造检测任务，解决现有数据集规模小、多样性不足及检测模型泛化性差的问题。构建含200万张图像的So-Fake-Set数据集及10万张跨领域测试集，提出基于强化学习的So-Fake-R1框架，提升检测与定位精度并开源资源。**

- **链接: [http://arxiv.org/pdf/2505.18660v1](http://arxiv.org/pdf/2505.18660v1)**

> **作者:** Zhenglin Huang; Tianxiao Li; Xiangtai Li; Haiquan Wen; Yiwei He; Jiangning Zhang; Hao Fei; Xi Yang; Xiaowei Huang; Bei Peng; Guangliang Cheng
>
> **摘要:** Recent advances in AI-powered generative models have enabled the creation of increasingly realistic synthetic images, posing significant risks to information integrity and public trust on social media platforms. While robust detection frameworks and diverse, large-scale datasets are essential to mitigate these risks, existing academic efforts remain limited in scope: current datasets lack the diversity, scale, and realism required for social media contexts, while detection methods struggle with generalization to unseen generative technologies. To bridge this gap, we introduce So-Fake-Set, a comprehensive social media-oriented dataset with over 2 million high-quality images, diverse generative sources, and photorealistic imagery synthesized using 35 state-of-the-art generative models. To rigorously evaluate cross-domain robustness, we establish a novel and large-scale (100K) out-of-domain benchmark (So-Fake-OOD) featuring synthetic imagery from commercial models explicitly excluded from the training distribution, creating a realistic testbed for evaluating real-world performance. Leveraging these resources, we present So-Fake-R1, an advanced vision-language framework that employs reinforcement learning for highly accurate forgery detection, precise localization, and explainable inference through interpretable visual rationales. Extensive experiments show that So-Fake-R1 outperforms the second-best method, with a 1.3% gain in detection accuracy and a 4.5% increase in localization IoU. By integrating a scalable dataset, a challenging OOD benchmark, and an advanced detection framework, this work establishes a new foundation for social media-centric forgery detection research. The code, models, and datasets will be released publicly.
>
---
#### [new 081] Harnessing the Power of Training-Free Techniques in Text-to-2D Generation for Text-to-3D Generation via Score Distillation Sampling
- **分类: cs.CV**

- **简介: 该论文属于文本到3D生成任务，旨在通过2D提升方法优化Score Distillation Sampling（SDS）中无训练技术（如CFG、FreeU）的应用。针对现有SDS对无训练技术探索不足导致的几何缺陷、纹理与表面平滑度失衡问题，论文分析了CFG/FreeU参数对生成效果的影响，并提出动态调整策略，平衡纹理细节、表面平滑度及几何准确性，提升3D生成质量。**

- **链接: [http://arxiv.org/pdf/2505.19868v1](http://arxiv.org/pdf/2505.19868v1)**

> **作者:** Junhong Lee; Seungwook Kim; Minsu Cho
>
> **摘要:** Recent studies show that simple training-free techniques can dramatically improve the quality of text-to-2D generation outputs, e.g. Classifier-Free Guidance (CFG) or FreeU. However, these training-free techniques have been underexplored in the lens of Score Distillation Sampling (SDS), which is a popular and effective technique to leverage the power of pretrained text-to-2D diffusion models for various tasks. In this paper, we aim to shed light on the effect such training-free techniques have on SDS, via a particular application of text-to-3D generation via 2D lifting. We present our findings, which show that varying the scales of CFG presents a trade-off between object size and surface smoothness, while varying the scales of FreeU presents a trade-off between texture details and geometric errors. Based on these findings, we provide insights into how we can effectively harness training-free techniques for SDS, via a strategic scaling of such techniques in a dynamic manner with respect to the timestep or optimization iteration step. We show that using our proposed scheme strikes a favorable balance between texture details and surface smoothness in text-to-3D generations, while preserving the size of the output and mitigating the occurrence of geometric defects.
>
---
#### [new 082] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance
- **分类: cs.CV**

- **简介: 该论文属于图像去退化任务，旨在解决人类中心图像在传输中同时存在的通用退化和人体运动模糊（HMB）问题。提出HAODiff方法，通过三分支双提示引导（结合高清图、残差噪声和HMB掩码），在单步扩散中生成自适应提示对，提升分类器自由引导效果。实验在新基准MPII-Test上验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19742v1](http://arxiv.org/pdf/2505.19742v1)**

> **作者:** Jue Gong; Tingyu Yang; Jingkai Wang; Zheng Chen; Xing Liu; Hong Gu; Yulun Zhang; Xiaokang Yang
>
> **备注:** 9 pages, 8 figures. The code and model will be available at https://github.com/gobunu/HAODiff
>
> **摘要:** Human-centered images often suffer from severe generic degradation during transmission and are prone to human motion blur (HMB), making restoration challenging. Existing research lacks sufficient focus on these issues, as both problems often coexist in practice. To address this, we design a degradation pipeline that simulates the coexistence of HMB and generic noise, generating synthetic degraded data to train our proposed HAODiff, a human-aware one-step diffusion. Specifically, we propose a triple-branch dual-prompt guidance (DPG), which leverages high-quality images, residual noise (LQ minus HQ), and HMB segmentation masks as training targets. It produces a positive-negative prompt pair for classifier-free guidance (CFG) in a single diffusion step. The resulting adaptive dual prompts let HAODiff exploit CFG more effectively, boosting robustness against diverse degradations. For fair evaluation, we introduce MPII-Test, a benchmark rich in combined noise and HMB cases. Extensive experiments show that our HAODiff surpasses existing state-of-the-art (SOTA) methods in terms of both quantitative metrics and visual quality on synthetic and real-world datasets, including our introduced MPII-Test. Code is available at: https://github.com/gobunu/HAODiff.
>
---
#### [new 083] Recent Deep Learning in Crowd Behaviour Analysis: A Brief Review
- **分类: cs.CV**

- **简介: 该论文综述了深度学习在人群行为分析中的最新进展，聚焦行为预测与识别任务。旨在解决如何通过深度神经网络（含物理融合模型）提升行为分析效果的问题，总结了典型方法、对比研究，并探讨未来方向，为研究者提供领域概览。**

- **链接: [http://arxiv.org/pdf/2505.18401v1](http://arxiv.org/pdf/2505.18401v1)**

> **作者:** Jiangbei Yue; He Wang
>
> **备注:** 51 pages, 7 figures, Book Chapter
>
> **摘要:** Crowd behaviour analysis is essential to numerous real-world applications, such as public safety and urban planning, and therefore has been studied for decades. In the last decade or so, the development of deep learning has significantly propelled the research on crowd behaviours. This chapter reviews recent advances in crowd behaviour analysis using deep learning. We mainly review the research in two core tasks in this field, crowd behaviour prediction and recognition. We broadly cover how different deep neural networks, after first being proposed in machine learning, are applied to analysing crowd behaviours. This includes pure deep neural network models as well as recent development of methodologies combining physics with deep learning. In addition, representative studies are discussed and compared in detail. Finally, we discuss the effectiveness of existing methods and future research directions in this rapidly evolving field. This chapter aims to provide a high-level summary of the ongoing deep learning research in crowd behaviour analysis. It intends to help new researchers who just entered this field to obtain an overall understanding of the ongoing research, as well as to provide a retrospective analysis for existing researchers to identify possible future directions
>
---
#### [new 084] Agentic 3D Scene Generation with Spatially Contextualized VLMs
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出基于空间上下文的VLM新范式，解决其在结构化3D场景生成中的不足。通过构建包含场景蓝图、语义点云及超图的空间上下文，结合迭代生成流程，实现高质量3D场景生成与编辑，并支持交互式编辑和路径规划等任务。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20129v1](http://arxiv.org/pdf/2505.20129v1)**

> **作者:** Xinhang Liu; Yu-Wing Tai; Chi-Keung Tang
>
> **摘要:** Despite recent advances in multimodal content generation enabled by vision-language models (VLMs), their ability to reason about and generate structured 3D scenes remains largely underexplored. This limitation constrains their utility in spatially grounded tasks such as embodied AI, immersive simulations, and interactive 3D applications. We introduce a new paradigm that enables VLMs to generate, understand, and edit complex 3D environments by injecting a continually evolving spatial context. Constructed from multimodal input, this context consists of three components: a scene portrait that provides a high-level semantic blueprint, a semantically labeled point cloud capturing object-level geometry, and a scene hypergraph that encodes rich spatial relationships, including unary, binary, and higher-order constraints. Together, these components provide the VLM with a structured, geometry-aware working memory that integrates its inherent multimodal reasoning capabilities with structured 3D understanding for effective spatial reasoning. Building on this foundation, we develop an agentic 3D scene generation pipeline in which the VLM iteratively reads from and updates the spatial context. The pipeline features high-quality asset generation with geometric restoration, environment setup with automatic verification, and ergonomic adjustment guided by the scene hypergraph. Experiments show that our framework can handle diverse and challenging inputs, achieving a level of generalization not observed in prior work. Further results demonstrate that injecting spatial context enables VLMs to perform downstream tasks such as interactive scene editing and path planning, suggesting strong potential for spatially intelligent systems in computer graphics, 3D vision, and embodied applications.
>
---
#### [new 085] Certainty and Uncertainty Guided Active Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于主动域适应任务，针对现有方法仅关注不确定样本而忽略高置信度有效样本的问题，提出结合高斯过程主动采样（GPAS）和伪标签确定采样（PLCS）的协作框架，通过利用自信预测优化搜索空间，逐步提升域适应效果，在多个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19421v1](http://arxiv.org/pdf/2505.19421v1)**

> **作者:** Bardia Safaei; Vibashan VS; Vishal M. Patel
>
> **备注:** Accepted at IEEE ICIP 2025
>
> **摘要:** Active Domain Adaptation (ADA) adapts models to target domains by selectively labeling a few target samples. Existing ADA methods prioritize uncertain samples but overlook confident ones, which often match ground-truth. We find that incorporating confident predictions into the labeled set before active sampling reduces the search space and improves adaptation. To address this, we propose a collaborative framework that labels uncertain samples while treating highly confident predictions as ground truth. Our method combines Gaussian Process-based Active Sampling (GPAS) for identifying uncertain samples and Pseudo-Label-based Certain Sampling (PLCS) for confident ones, progressively enhancing adaptation. PLCS refines the search space, and GPAS reduces the domain gap, boosting the proportion of confident samples. Extensive experiments on Office-Home and DomainNet show that our approach outperforms state-of-the-art ADA methods.
>
---
#### [new 086] Decomposing Complex Visual Comprehension into Atomic Visual Skills for Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于Vision-Language Models（VLM）评估任务，旨在解决VLM在基础视觉任务上的不足。通过系统分类2D几何中的原子视觉技能，构建AVSD数据集，发现先进VLM在简单视觉推理上表现差于人类，强调需专用数据集训练基础视觉能力。**

- **链接: [http://arxiv.org/pdf/2505.20021v1](http://arxiv.org/pdf/2505.20021v1)**

> **作者:** Hyunsik Chae; Seungwoo Yoon; Jaden Park; Chloe Yewon Chun; Yongin Cho; Mu Cai; Yong Jae Lee; Ernest K. Ryu
>
> **备注:** 69 pages, 16 figures
>
> **摘要:** Recent Vision-Language Models (VLMs) have demonstrated impressive multimodal comprehension and reasoning capabilities, yet they often struggle with trivially simple visual tasks. In this work, we focus on the domain of basic 2D Euclidean geometry and systematically categorize the fundamental, indivisible visual perception skills, which we refer to as atomic visual skills. We then introduce the Atomic Visual Skills Dataset (AVSD) for evaluating VLMs on the atomic visual skills. Using AVSD, we benchmark state-of-the-art VLMs and find that they struggle with these tasks, despite being trivial for adult humans. Our findings highlight the need for purpose-built datasets to train and evaluate VLMs on atomic, rather than composite, visual perception tasks.
>
---
#### [new 087] Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter
- **分类: cs.CV**

- **简介: 该论文属于个性化文本到图像生成任务，旨在解决现有方法无法有效处理抽象概念（如姿势、光线）且需耗时微调的问题。提出Mod-Adapter，通过调制适配器预测概念特定的文本标记调制方向，结合视觉语言交叉注意力和专家混合层映射特征，并采用VLM引导预训练策略，实现无需微调的多概念个性化生成，达当前最优性能。**

- **链接: [http://arxiv.org/pdf/2505.18612v1](http://arxiv.org/pdf/2505.18612v1)**

> **作者:** Weizhi Zhong; Huan Yang; Zheng Liu; Huiguo He; Zijian He; Xuesong Niu; Di Zhang; Guanbin Li
>
> **备注:** Project page: https://weizhi-zhong.github.io/Mod-Adapter
>
> **摘要:** Personalized text-to-image generation aims to synthesize images of user-provided concepts in diverse contexts. Despite recent progress in multi-concept personalization, most are limited to object concepts and struggle to customize abstract concepts (e.g., pose, lighting). Some methods have begun exploring multi-concept personalization supporting abstract concepts, but they require test-time fine-tuning for each new concept, which is time-consuming and prone to overfitting on limited training images. In this work, we propose a novel tuning-free method for multi-concept personalization that can effectively customize both object and abstract concepts without test-time fine-tuning. Our method builds upon the modulation mechanism in pretrained Diffusion Transformers (DiTs) model, leveraging the localized and semantically meaningful properties of the modulation space. Specifically, we propose a novel module, Mod-Adapter, to predict concept-specific modulation direction for the modulation process of concept-related text tokens. It incorporates vision-language cross-attention for extracting concept visual features, and Mixture-of-Experts (MoE) layers that adaptively map the concept features into the modulation space. Furthermore, to mitigate the training difficulty caused by the large gap between the concept image space and the modulation space, we introduce a VLM-guided pretraining strategy that leverages the strong image understanding capabilities of vision-language models to provide semantic supervision signals. For a comprehensive comparison, we extend a standard benchmark by incorporating abstract concepts. Our method achieves state-of-the-art performance in multi-concept personalization, supported by quantitative, qualitative, and human evaluations.
>
---
#### [new 088] ProphetDWM: A Driving World Model for Rolling Out Future Actions and Videos
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶领域未来动作与视频预测任务。针对现有模型需依赖等长动作序列且忽略动作动态规律的问题，提出ProphetDWM模型，通过联合学习动作模块与扩散模型转移模块，实现动作控制、预测与视频生成的端到端联合优化，提升长期预测精度与视频一致性。**

- **链接: [http://arxiv.org/pdf/2505.18650v1](http://arxiv.org/pdf/2505.18650v1)**

> **作者:** Xiaodong Wang; Peixi Peng
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Real-world driving requires people to observe the current environment, anticipate the future, and make appropriate driving decisions. This requirement is aligned well with the capabilities of world models, which understand the environment and predict the future. However, recent world models in autonomous driving are built explicitly, where they could predict the future by controllable driving video generation. We argue that driving world models should have two additional abilities: action control and action prediction. Following this line, previous methods are limited because they predict the video requires given actions of the same length as the video and ignore the dynamical action laws. To address these issues, we propose ProphetDWM, a novel end-to-end driving world model that jointly predicts future videos and actions. Our world model has an action module to learn latent action from the present to the future period by giving the action sequence and observations. And a diffusion-model-based transition module to learn the state distribution. The model is jointly trained by learning latent actions given finite states and predicting action and video. The joint learning connects the action dynamics and states and enables long-term future prediction. We evaluate our method in video generation and action prediction tasks on the Nuscenes dataset. Compared to the state-of-the-art methods, our method achieves the best video consistency and best action prediction accuracy, while also enabling high-quality long-term video and action generation.
>
---
#### [new 089] Manifold-aware Representation Learning for Degradation-agnostic Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，解决现有方法未能有效建模多样退化结构的问题。提出MIRAGE框架，通过分解特征空间为全局上下文、局部纹理和通道统计三分支，并结合SPD流形上的对比学习，提升跨退化类型的泛化与效率，实现轻量化全场景恢复。**

- **链接: [http://arxiv.org/pdf/2505.18679v1](http://arxiv.org/pdf/2505.18679v1)**

> **作者:** Bin Ren; Yawei Li; Xu Zheng; Yuqian Fu; Danda Pani Paudel; Ming-Hsuan Yang; Luc Van Gool; Nicu Sebe
>
> **备注:** ALl-in-One Image Restoration, low-level vision
>
> **摘要:** Image Restoration (IR) aims to recover high quality images from degraded inputs affected by various corruptions such as noise, blur, haze, rain, and low light conditions. Despite recent advances, most existing approaches treat IR as a direct mapping problem, relying on shared representations across degradation types without modeling their structural diversity. In this work, we present MIRAGE, a unified and lightweight framework for all in one IR that explicitly decomposes the input feature space into three semantically aligned parallel branches, each processed by a specialized module attention for global context, convolution for local textures, and MLP for channel-wise statistics. This modular decomposition significantly improves generalization and efficiency across diverse degradations. Furthermore, we introduce a cross layer contrastive learning scheme that aligns shallow and latent features to enhance the discriminability of shared representations. To better capture the underlying geometry of feature representations, we perform contrastive learning in a Symmetric Positive Definite (SPD) manifold space rather than the conventional Euclidean space. Extensive experiments show that MIRAGE not only achieves new state of the art performance across a variety of degradation types but also offers a scalable solution for challenging all-in-one IR scenarios. Our code and models will be publicly available at https://amazingren.github.io/MIRAGE/.
>
---
#### [new 090] MGD$^3$: Mode-Guided Dataset Distillation using Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出MGD³方法，属于数据集蒸馏任务。针对现有方法依赖微调且无法保证样本多样性的缺陷，其通过三阶段流程（模式发现、模式引导、停止引导）提升合成数据多样性及代表性，无需微调扩散模型，显著降低计算成本，性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.18963v1](http://arxiv.org/pdf/2505.18963v1)**

> **作者:** Jeffrey A. Chan-Santiago; Praveen Tirupattur; Gaurav Kumar Nayak; Gaowen Liu; Mubarak Shah
>
> **摘要:** Dataset distillation has emerged as an effective strategy, significantly reducing training costs and facilitating more efficient model deployment. Recent advances have leveraged generative models to distill datasets by capturing the underlying data distribution. Unfortunately, existing methods require model fine-tuning with distillation losses to encourage diversity and representativeness. However, these methods do not guarantee sample diversity, limiting their performance. We propose a mode-guided diffusion model leveraging a pre-trained diffusion model without the need to fine-tune with distillation losses. Our approach addresses dataset diversity in three stages: Mode Discovery to identify distinct data modes, Mode Guidance to enhance intra-class diversity, and Stop Guidance to mitigate artifacts in synthetic samples that affect performance. Our approach outperforms state-of-the-art methods, achieving accuracy gains of 4.4%, 2.9%, 1.6%, and 1.6% on ImageNette, ImageIDC, ImageNet-100, and ImageNet-1K, respectively. Our method eliminates the need for fine-tuning diffusion models with distillation losses, significantly reducing computational costs. Our code is available on the project webpage: https://jachansantiago.github.io/mode-guided-distillation/
>
---
#### [new 091] StyleGuard: Preventing Text-to-Image-Model-based Style Mimicry Attacks by Style Perturbations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像模型防御任务，旨在解决现有防御方法易被净化攻击及跨模型迁移性差的问题。提出StyleGuard方法，通过设计样式损失扰动潜在空间的风格特征提升跨模型鲁棒性，并引入集成净化器的 upscale 损失增强抗净化能力，实验显示其有效对抗多种模仿攻击。**

- **链接: [http://arxiv.org/pdf/2505.18766v1](http://arxiv.org/pdf/2505.18766v1)**

> **作者:** Yanjie Li; Wenxuan Zhang; Xinqi Lyu; Yihao Liu; Bin Xiao
>
> **备注:** submitted to NIPS2025
>
> **摘要:** Recently, text-to-image diffusion models have been widely used for style mimicry and personalized customization through methods such as DreamBooth and Textual Inversion. This has raised concerns about intellectual property protection and the generation of deceptive content. Recent studies, such as Glaze and Anti-DreamBooth, have proposed using adversarial noise to protect images from these attacks. However, recent purification-based methods, such as DiffPure and Noise Upscaling, have successfully attacked these latest defenses, showing the vulnerabilities of these methods. Moreover, present methods show limited transferability across models, making them less effective against unknown text-to-image models. To address these issues, we propose a novel anti-mimicry method, StyleGuard. We propose a novel style loss that optimizes the style-related features in the latent space to make it deviate from the original image, which improves model-agnostic transferability. Additionally, to enhance the perturbation's ability to bypass diffusion-based purification, we designed a novel upscale loss that involves ensemble purifiers and upscalers during training. Extensive experiments on the WikiArt and CelebA datasets demonstrate that StyleGuard outperforms existing methods in robustness against various transformations and purifications, effectively countering style mimicry in various models. Moreover, StyleGuard is effective on different style mimicry methods, including DreamBooth and Textual Inversion.
>
---
#### [new 092] MoMBS: Mixed-order minibatch sampling enhances model training from diverse-quality images
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对图像分类和医学检测中多样质量样本（如标签噪声、类别分布不均）的训练问题，提出MoMBS方法。通过结合损失与不确定性衡量样本特性，区分"难样本"为标注错误/ minority或过拟合类别，采用混合顺序小批量采样优化梯度贡献，解决传统方法对样本利用失衡的问题。**

- **链接: [http://arxiv.org/pdf/2505.18741v1](http://arxiv.org/pdf/2505.18741v1)**

> **作者:** Han Li; Hu Han; S. Kevin Zhou
>
> **备注:** 16 pages,8 figures
>
> **摘要:** Natural images exhibit label diversity (clean vs. noisy) in noisy-labeled image classification and prevalence diversity (abundant vs. sparse) in long-tailed image classification. Similarly, medical images in universal lesion detection (ULD) exhibit substantial variations in image quality, encompassing attributes such as clarity and label correctness. How to effectively leverage training images with diverse qualities becomes a problem in learning deep models. Conventional training mechanisms, such as self-paced curriculum learning (SCL) and online hard example mining (OHEM), relieve this problem by reweighting images with high loss values. Despite their success, these methods still confront two challenges: (i) the loss-based measure of sample hardness is imprecise, preventing optimum handling of different cases, and (ii) there exists under-utilization in SCL or over-utilization OHEM with the identified hard samples. To address these issues, this paper revisits the minibatch sampling (MBS), a technique widely used in deep network training but largely unexplored concerning the handling of diverse-quality training samples. We discover that the samples within a minibatch influence each other during training; thus, we propose a novel Mixed-order Minibatch Sampling (MoMBS) method to optimize the use of training samples with diverse qualities. MoMBS introduces a measure that takes both loss and uncertainty into account to surpass a sole reliance on loss and allows for a more refined categorization of high-loss samples by distinguishing them as either poorly labeled and under represented or well represented and overfitted. We prioritize under represented samples as the main gradient contributors in a minibatch and keep them from the negative influences of poorly labeled or overfitted samples with a mixed-order minibatch sampling design.
>
---
#### [new 093] DART$^3$: Leveraging Distance for Test Time Adaptation in Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于行人重识别（ReID）领域，旨在解决摄像头视角偏移导致的性能下降问题。提出DART³框架，通过距离敏感的目标函数优化测试时适应，无需额外数据或模型修改，有效缓解跨摄像头识别偏差，在基准测试中超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.18337v1](http://arxiv.org/pdf/2505.18337v1)**

> **作者:** Rajarshi Bhattacharya; Shakeeb Murtaza; Christian Desrosiers; Jose Dolz; Maguelonne Heritier; Eric Granger
>
> **摘要:** Person re-identification (ReID) models are known to suffer from camera bias, where learned representations cluster according to camera viewpoints rather than identity, leading to significant performance degradation under (inter-camera) domain shifts in real-world surveillance systems when new cameras are added to camera networks. State-of-the-art test-time adaptation (TTA) methods, largely designed for classification tasks, rely on classification entropy-based objectives that fail to generalize well to ReID, thus making them unsuitable for tackling camera bias. In this paper, we introduce DART$^3$, a TTA framework specifically designed to mitigate camera-induced domain shifts in person ReID. DART$^3$ (Distance-Aware Retrieval Tuning at Test Time) leverages a distance-based objective that aligns better with image retrieval tasks like ReID by exploiting the correlation between nearest-neighbor distance and prediction error. Unlike prior ReID-specific domain adaptation methods, DART$^3$ requires no source data, architectural modifications, or retraining, and can be deployed in both fully black-box and hybrid settings. Empirical evaluations on multiple ReID benchmarks indicate that DART$^3$ and DART$^3$ LITE, a lightweight alternative to the approach, consistently outperforms state-of-the-art TTA baselines, making for a viable option to online learning to mitigate the adverse effects of camera bias.
>
---
#### [new 094] Focus on What Matters: Enhancing Medical Vision-Language Models with Automatic Attention Alignment Tuning
- **分类: cs.CV**

- **简介: 该论文属于医疗视觉语言模型（Med-LVLM）优化任务，旨在解决其视觉注意力分布不均导致输出不准确的问题。提出A³Tune框架，利用SAM生成弱标签并结合BioMedCLIP优化标签，针对性调整视觉关键注意力头，并引入A³MoE模块实现参数自适应，提升医疗VQA和报告生成性能。**

- **链接: [http://arxiv.org/pdf/2505.18503v1](http://arxiv.org/pdf/2505.18503v1)**

> **作者:** Aofei Chang; Le Huang; Alex James Boyd; Parminder Bhatia; Taha Kass-Hout; Cao Xiao; Fenglong Ma
>
> **备注:** Accepted to ACL2025 (main)
>
> **摘要:** Medical Large Vision-Language Models (Med-LVLMs) often exhibit suboptimal attention distribution on visual inputs, leading to hallucinated or inaccurate outputs. Existing mitigation methods primarily rely on inference-time interventions, which are limited in attention adaptation or require additional supervision. To address this, we propose A$^3$Tune, a novel fine-tuning framework for Automatic Attention Alignment Tuning. A$^3$Tune leverages zero-shot weak labels from SAM, refines them into prompt-aware labels using BioMedCLIP, and then selectively modifies visually-critical attention heads to improve alignment while minimizing interference. Additionally, we introduce a A$^3$MoE module, enabling adaptive parameter selection for attention tuning across diverse prompts and images. Extensive experiments on medical VQA and report generation benchmarks show that A$^3$Tune outperforms state-of-the-art baselines, achieving enhanced attention distributions and performance in Med-LVLMs.
>
---
#### [new 095] Long-Context State-Space Video World Models
- **分类: cs.CV**

- **简介: 该论文属于视频世界建模任务，旨在解决长序列预测中计算成本高导致的长期记忆不足问题。提出基于状态空间模型的分块扫描架构，结合局部注意力平衡时空一致性，提升长期推理效率，在迷宫和MineCraft任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.20171v1](http://arxiv.org/pdf/2505.20171v1)**

> **作者:** Ryan Po; Yotam Nitzan; Richard Zhang; Berlin Chen; Tri Dao; Eli Shechtman; Gordon Wetzstein; Xun Huang
>
> **备注:** Project website: https://ryanpo.com/ssm_wm
>
> **摘要:** Video diffusion models have recently shown promise for world modeling through autoregressive frame prediction conditioned on actions. However, they struggle to maintain long-term memory due to the high computational cost associated with processing extended sequences in attention layers. To overcome this limitation, we propose a novel architecture leveraging state-space models (SSMs) to extend temporal memory without compromising computational efficiency. Unlike previous approaches that retrofit SSMs for non-causal vision tasks, our method fully exploits the inherent advantages of SSMs in causal sequence modeling. Central to our design is a block-wise SSM scanning scheme, which strategically trades off spatial consistency for extended temporal memory, combined with dense local attention to ensure coherence between consecutive frames. We evaluate the long-term memory capabilities of our model through spatial retrieval and reasoning tasks over extended horizons. Experiments on Memory Maze and Minecraft datasets demonstrate that our approach surpasses baselines in preserving long-range memory, while maintaining practical inference speeds suitable for interactive applications.
>
---
#### [new 096] SuperGS: Consistent and Detailed 3D Super-Resolution Scene Reconstruction via Gaussian Splatting
- **分类: cs.CV**

- **简介: 论文提出SuperGS，针对高分辨率3D场景重建与新颖视角合成任务，解决低分辨率输入导致粗糙及多视图不一致问题。通过两阶段训练框架：低分辨率阶段用潜伏特征场初始化，高分辨率阶段采用基于误差图和多视图投票的致密化策略，结合不确定性建模优化伪标签，提升细节与一致性。**

- **链接: [http://arxiv.org/pdf/2505.18649v1](http://arxiv.org/pdf/2505.18649v1)**

> **作者:** Shiyun Xie; Zhiru Wang; Yinghao Zhu; Xu Wang; Chengwei Pan; Xiwang Dong
>
> **摘要:** Recently, 3D Gaussian Splatting (3DGS) has excelled in novel view synthesis (NVS) with its real-time rendering capabilities and superior quality. However, it encounters challenges for high-resolution novel view synthesis (HRNVS) due to the coarse nature of primitives derived from low-resolution input views. To address this issue, we propose SuperGS, an expansion of Scaffold-GS designed with a two-stage coarse-to-fine training framework. In the low-resolution stage, we introduce a latent feature field to represent the low-resolution scene, which serves as both the initialization and foundational information for super-resolution optimization. In the high-resolution stage, we propose a multi-view consistent densification strategy that backprojects high-resolution depth maps based on error maps and employs a multi-view voting mechanism, mitigating ambiguities caused by multi-view inconsistencies in the pseudo labels provided by 2D prior models while avoiding Gaussian redundancy. Furthermore, we model uncertainty through variational feature learning and use it to guide further scene representation refinement and adjust the supervisory effect of pseudo-labels, ensuring consistent and detailed scene reconstruction. Extensive experiments demonstrate that SuperGS outperforms state-of-the-art HRNVS methods on both forward-facing and 360-degree datasets.
>
---
#### [new 097] ParticleGS: Particle-Based Dynamics Modeling of 3D Gaussians for Prior-free Motion Extrapolation
- **分类: cs.CV**

- **简介: 该论文属于3D动态重建与运动外推任务，旨在解决现有方法依赖物理先验或无法有效建模动态导致的外推能力不足问题。提出ParticleGS框架，通过引入动力学潜势向量、编码器提取初始状态及Neural ODEs建模高斯粒子时序演化，实现无先验的动态建模与未来帧预测。**

- **链接: [http://arxiv.org/pdf/2505.20270v1](http://arxiv.org/pdf/2505.20270v1)**

> **作者:** Jinsheng Quan; Chunshi Wang; Yawei Luo
>
> **摘要:** This paper aims to model the dynamics of 3D Gaussians from visual observations to support temporal extrapolation. Existing dynamic 3D reconstruction methods often struggle to effectively learn underlying dynamics or rely heavily on manually defined physical priors, which limits their extrapolation capabilities. To address this issue, we propose a novel dynamic 3D Gaussian Splatting prior-free motion extrapolation framework based on particle dynamics systems. The core advantage of our method lies in its ability to learn differential equations that describe the dynamics of 3D Gaussians, and follow them during future frame extrapolation. Instead of simply fitting to the observed visual frame sequence, we aim to more effectively model the gaussian particle dynamics system. To this end, we introduce a dynamics latent state vector into the standard Gaussian kernel and design a dynamics latent space encoder to extract initial state. Subsequently, we introduce a Neural ODEs-based dynamics module that models the temporal evolution of Gaussian in dynamics latent space. Finally, a Gaussian kernel space decoder is used to decode latent state at the specific time step into the deformation. Experimental results demonstrate that the proposed method achieves comparable rendering quality with existing approaches in reconstruction tasks, and significantly outperforms them in future frame extrapolation. Our code is available at https://github.com/QuanJinSheng/ParticleGS.
>
---
#### [new 098] CTRL-GS: Cascaded Temporal Residue Learning for 4D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出CTRL-GS方法，改进4D高斯散射用于动态场景的新视角合成。通过分层分解视频-段-帧结构，结合残差学习建模信号（视频常量+段常量+帧残差），解决复杂运动、遮挡导致的退化问题，实现高质量实时渲染。**

- **链接: [http://arxiv.org/pdf/2505.18306v1](http://arxiv.org/pdf/2505.18306v1)**

> **作者:** Karly Hou; Wanhua Li; Hanspeter Pfister
>
> **备注:** Accepted to 4D Vision Workshop @ CVPR 2025
>
> **摘要:** Recently, Gaussian Splatting methods have emerged as a desirable substitute for prior Radiance Field methods for novel-view synthesis of scenes captured with multi-view images or videos. In this work, we propose a novel extension to 4D Gaussian Splatting for dynamic scenes. Drawing on ideas from residual learning, we hierarchically decompose the dynamic scene into a "video-segment-frame" structure, with segments dynamically adjusted by optical flow. Then, instead of directly predicting the time-dependent signals, we model the signal as the sum of video-constant values, segment-constant values, and frame-specific residuals, as inspired by the success of residual learning. This approach allows more flexible models that adapt to highly variable scenes. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets, with the greatest improvements on complex scenes with large movements, occlusions, and fine details, where current methods degrade most.
>
---
#### [new 099] Multimodal Reasoning Agent for Zero-Shot Composed Image Retrieval
- **分类: cs.CV; cs.IR**

- **简介: 该论文提出Multimodle Reasoning Agent（MRA）框架，解决零样本组合图像检索（ZS-CIR）中现有方法依赖中间文本导致误差累积的问题。通过直接构建<参考图+修改文本+目标图>三元组并利用无标注数据训练，消除文本中介，提升检索性能，在三个基准数据集上显著优于基线。**

- **链接: [http://arxiv.org/pdf/2505.19952v1](http://arxiv.org/pdf/2505.19952v1)**

> **作者:** Rong-Cheng Tu; Wenhao Sun; Hanzhe You; Yingjie Wang; Jiaxing Huang; Li Shen; Dacheng Tao
>
> **摘要:** Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images given a compositional query, consisting of a reference image and a modifying text-without relying on annotated training data. Existing approaches often generate a synthetic target text using large language models (LLMs) to serve as an intermediate anchor between the compositional query and the target image. Models are then trained to align the compositional query with the generated text, and separately align images with their corresponding texts using contrastive learning. However, this reliance on intermediate text introduces error propagation, as inaccuracies in query-to-text and text-to-image mappings accumulate, ultimately degrading retrieval performance. To address these problems, we propose a novel framework by employing a Multimodal Reasoning Agent (MRA) for ZS-CIR. MRA eliminates the dependence on textual intermediaries by directly constructing triplets, <reference image, modification text, target image>, using only unlabeled image data. By training on these synthetic triplets, our model learns to capture the relationships between compositional queries and candidate images directly. Extensive experiments on three standard CIR benchmarks demonstrate the effectiveness of our approach. On the FashionIQ dataset, our method improves Average R@10 by at least 7.5\% over existing baselines; on CIRR, it boosts R@1 by 9.6\%; and on CIRCO, it increases mAP@5 by 9.5\%.
>
---
#### [new 100] VTBench: Comprehensive Benchmark Suite Towards Real-World Virtual Try-on Models
- **分类: cs.CV**

- **简介: 该论文提出VTBench，针对真实场景虚拟试穿模型的评估基准。任务为建立全面评估框架，解决现有指标脱离人类感知、测试场景单一及缺乏指导性问题。工作包括构建分层基准，涵盖图像质量、纹理保留等五维度，提供测试集、评估标准及人类标注，并分析室内外场景性能差异，开源资源推动研究。**

- **链接: [http://arxiv.org/pdf/2505.19571v1](http://arxiv.org/pdf/2505.19571v1)**

> **作者:** Hu Xiaobin; Liang Yujie; Luo Donghao; Peng Xu; Zhang Jiangning; Zhu Junwei; Wang Chengjie; Fu Yanwei
>
> **备注:** Project Websit: \url{https://github.com/HUuxiaobin/VTBench}
>
> **摘要:** While virtual try-on has achieved significant progress, evaluating these models towards real-world scenarios remains a challenge. A comprehensive benchmark is essential for three key reasons:(1) Current metrics inadequately reflect human perception, particularly in unpaired try-on settings;(2)Most existing test sets are limited to indoor scenarios, lacking complexity for real-world evaluation; and (3) An ideal system should guide future advancements in virtual try-on generation. To address these needs, we introduce VTBench, a hierarchical benchmark suite that systematically decomposes virtual image try-on into hierarchical, disentangled dimensions, each equipped with tailored test sets and evaluation criteria. VTBench exhibits three key advantages:1) Multi-Dimensional Evaluation Framework: The benchmark encompasses five critical dimensions for virtual try-on generation (e.g., overall image quality, texture preservation, complex background consistency, cross-category size adaptability, and hand-occlusion handling). Granular evaluation metrics of corresponding test sets pinpoint model capabilities and limitations across diverse, challenging scenarios.2) Human Alignment: Human preference annotations are provided for each test set, ensuring the benchmark's alignment with perceptual quality across all evaluation dimensions. (3) Valuable Insights: Beyond standard indoor settings, we analyze model performance variations across dimensions and investigate the disparity between indoor and real-world try-on scenarios. To foster the field of virtual try-on towards challenging real-world scenario, VTBench will be open-sourced, including all test sets, evaluation protocols, generated results, and human annotations.
>
---
#### [new 101] Holistic White-light Polyp Classification via Alignment-free Dense Distillation of Auxiliary Optical Chromoendoscopy
- **分类: cs.CV**

- **简介: 该论文属于结直肠息肉分类任务，旨在提升白光成像（WLI）在资源受限场景下的诊断性能。针对现有方法依赖病灶裁剪导致的误差及上下文信息缺失问题，提出无需定位的整体分类框架，创新性地设计无对齐密集知识蒸馏（ADD）模块，通过像素级特征映射与CAM筛选，实现跨模态（WLI-NBI）知识迁移，显著提升分类精度。**

- **链接: [http://arxiv.org/pdf/2505.19319v1](http://arxiv.org/pdf/2505.19319v1)**

> **作者:** Qiang Hu; Qimei Wang; Jia Chen; Xuantao Ji; Qiang Li; Zhiwei Wang
>
> **备注:** Early Accepted by MICCAI 2025. Code and models: https://github.com/Huster-Hq/ADD
>
> **摘要:** White Light Imaging (WLI) and Narrow Band Imaging (NBI) are the two main colonoscopic modalities for polyp classification. While NBI, as optical chromoendoscopy, offers valuable vascular details, WLI remains the most common and often the only available modality in resource-limited settings. However, WLI-based methods typically underperform, limiting their clinical applicability. Existing approaches transfer knowledge from NBI to WLI through global feature alignment but often rely on cropped lesion regions, which are susceptible to detection errors and neglect contextual and subtle diagnostic cues. To address this, this paper proposes a novel holistic classification framework that leverages full-image diagnosis without requiring polyp localization. The key innovation lies in the Alignment-free Dense Distillation (ADD) module, which enables fine-grained cross-domain knowledge distillation regardless of misalignment between WLI and NBI images. Without resorting to explicit image alignment, ADD learns pixel-wise cross-domain affinities to establish correspondences between feature maps, guiding the distillation along the most relevant pixel connections. To further enhance distillation reliability, ADD incorporates Class Activation Mapping (CAM) to filter cross-domain affinities, ensuring the distillation path connects only those semantically consistent regions with equal contributions to polyp diagnosis. Extensive results on public and in-house datasets show that our method achieves state-of-the-art performance, relatively outperforming the other approaches by at least 2.5% and 16.2% in AUC, respectively. Code is available at: https://github.com/Huster-Hq/ADD.
>
---
#### [new 102] TUNA: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos
- **分类: cs.CV; cs.DB; cs.MM**

- **简介: 该论文提出TUNA基准，针对密集动态视频的细粒度时间理解任务，解决现有方法对视频时空元素综合分析不足的问题。通过构建包含视频描述和问答的双任务评测体系，揭示模型在动作描述、多主体理解及相机运动感知上的局限性，为改进视频理解模型提供数据与分析支持。**

- **链接: [http://arxiv.org/pdf/2505.20124v1](http://arxiv.org/pdf/2505.20124v1)**

> **作者:** Fanheng Kong; Jingyuan Zhang; Hongzhi Zhang; Shi Feng; Daling Wang; Linhao Yu; Xingguang Ji; Yu Tian; Qi Wang; Fuzheng Zhang
>
> **备注:** Accepted to CVPR 2025 Main. Project page: https://friedrichor.github.io/projects/TUNA
>
> **摘要:** Videos are unique in their integration of temporal elements, including camera, scene, action, and attribute, along with their dynamic relationships over time. However, existing benchmarks for video understanding often treat these properties separately or narrowly focus on specific aspects, overlooking the holistic nature of video content. To address this, we introduce TUNA, a temporal-oriented benchmark for fine-grained understanding on dense dynamic videos, with two complementary tasks: captioning and QA. Our TUNA features diverse video scenarios and dynamics, assisted by interpretable and robust evaluation criteria. We evaluate several leading models on our benchmark, providing fine-grained performance assessments across various dimensions. This evaluation reveals key challenges in video temporal understanding, such as limited action description, inadequate multi-subject understanding, and insensitivity to camera motion, offering valuable insights for improving video understanding models. The data and code are available at https://friedrichor.github.io/projects/TUNA.
>
---
#### [new 103] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Behavioural Change
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出首个用于视频中多模态矛盾/犹豫（A/H）识别的BAH数据集，解决缺乏相关数据导致机器学习模型开发受阻的问题。收集224名加拿大多族群参与者视频（8.26小时），标注时间戳、帧/视频级A/H线索及元数据，并提供基线模型验证任务挑战性。**

- **链接: [http://arxiv.org/pdf/2505.19328v1](http://arxiv.org/pdf/2505.19328v1)**

> **作者:** Manuela González-González; Soufiane Belharbi; Muhammad Osama Zeeshan; Masoumeh Sharafi; Muhammad Haseeb Aslam; Marco Pedersoli; Alessandro Lameiras Koerich; Simon L Bacon; Eric Granger
>
> **备注:** 41 pages, 13 figures, under review
>
> **摘要:** Recognizing complex emotions linked to ambivalence and hesitancy (A/H) can play a critical role in the personalization and effectiveness of digital behaviour change interventions. These subtle and conflicting emotions are manifested by a discord between multiple modalities, such as facial and vocal expressions, and body language. Although experts can be trained to identify A/H, integrating them into digital interventions is costly and less effective. Automatic learning systems provide a cost-effective alternative that can adapt to individual users, and operate seamlessly within real-time, and resource-limited environments. However, there are currently no datasets available for the design of ML models to recognize A/H. This paper introduces a first Behavioural Ambivalence/Hesitancy (BAH) dataset collected for subject-based multimodal recognition of A/H in videos. It contains videos from 224 participants captured across 9 provinces in Canada, with different age, and ethnicity. Through our web platform, we recruited participants to answer 7 questions, some of which were designed to elicit A/H while recording themselves via webcam with microphone. BAH amounts to 1,118 videos for a total duration of 8.26 hours with 1.5 hours of A/H. Our behavioural team annotated timestamp segments to indicate where A/H occurs, and provide frame- and video-level annotations with the A/H cues. Video transcripts and their timestamps are also included, along with cropped and aligned faces in each frame, and a variety of participants meta-data. We include results baselines for BAH at frame- and video-level recognition in multi-modal setups, in addition to zero-shot prediction, and for personalization using unsupervised domain adaptation. The limited performance of baseline models highlights the challenges of recognizing A/H in real-world videos. The data, code, and pretrained weights are available.
>
---
#### [new 104] InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 论文提出InstructPart基准，针对任务导向的部件分割问题，解决现有模型忽视物体部件功能的不足。通过含标注数据和指令的基准及微调基线，提升Vision-Language Models在机器人等领域的部件级任务执行能力。**

- **链接: [http://arxiv.org/pdf/2505.18291v1](http://arxiv.org/pdf/2505.18291v1)**

> **作者:** Zifu Wan; Yaqi Xie; Ce Zhang; Zhiqiu Lin; Zihan Wang; Simon Stepputtis; Deva Ramanan; Katia Sycara
>
> **备注:** Accepted by ACL 2025 Main. Project page: https://zifuwan.github.io/InstructPart/
>
> **摘要:** Large multimodal foundation models, particularly in the domains of language and vision, have significantly advanced various tasks, including robotics, autonomous driving, information retrieval, and grounding. However, many of these models perceive objects as indivisible, overlooking the components that constitute them. Understanding these components and their associated affordances provides valuable insights into an object's functionality, which is fundamental for performing a wide range of tasks. In this work, we introduce a novel real-world benchmark, InstructPart, comprising hand-labeled part segmentation annotations and task-oriented instructions to evaluate the performance of current models in understanding and executing part-level tasks within everyday contexts. Through our experiments, we demonstrate that task-oriented part segmentation remains a challenging problem, even for state-of-the-art Vision-Language Models (VLMs). In addition to our benchmark, we introduce a simple baseline that achieves a twofold performance improvement through fine-tuning with our dataset. With our dataset and benchmark, we aim to facilitate research on task-oriented part segmentation and enhance the applicability of VLMs across various domains, including robotics, virtual reality, information retrieval, and other related fields. Project website: https://zifuwan.github.io/InstructPart/.
>
---
#### [new 105] Plug-and-Play Context Feature Reuse for Efficient Masked Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，针对masked生成模型（MGMs）因多步迭代导致推理效率低的问题，提出ReCAP模块，通过复用已解码的上下文特征减少计算，在保持生成质量前提下加速推理（如ImageNet256提升2.4倍速度）。**

- **链接: [http://arxiv.org/pdf/2505.19089v1](http://arxiv.org/pdf/2505.19089v1)**

> **作者:** Xuejie Liu; Anji Liu; Guy Van den Broeck; Yitao Liang
>
> **摘要:** Masked generative models (MGMs) have emerged as a powerful framework for image synthesis, combining parallel decoding with strong bidirectional context modeling. However, generating high-quality samples typically requires many iterative decoding steps, resulting in high inference costs. A straightforward way to speed up generation is by decoding more tokens in each step, thereby reducing the total number of steps. However, when many tokens are decoded simultaneously, the model can only estimate the univariate marginal distributions independently, failing to capture the dependency among them. As a result, reducing the number of steps significantly compromises generation fidelity. In this work, we introduce ReCAP (Reused Context-Aware Prediction), a plug-and-play module that accelerates inference in MGMs by constructing low-cost steps via reusing feature embeddings from previously decoded context tokens. ReCAP interleaves standard full evaluations with lightweight steps that cache and reuse context features, substantially reducing computation while preserving the benefits of fine-grained, iterative generation. We demonstrate its effectiveness on top of three representative MGMs (MaskGIT, MAGE, and MAR), including both discrete and continuous token spaces and covering diverse architectural designs. In particular, on ImageNet256 class-conditional generation, ReCAP achieves up to 2.4x faster inference than the base model with minimal performance drop, and consistently delivers better efficiency-fidelity trade-offs under various generation settings.
>
---
#### [new 106] Step-level Reward for Free in RL-based T2I Diffusion Model Fine-tuning
- **分类: cs.CV**

- **简介: 该论文属于RL驱动的文本到图像扩散模型微调任务，旨在解决现有方法因单次延迟奖励导致的步骤级动作归因不准及训练效率低问题。提出动态分配密集奖励框架，通过跟踪中间与最终图像的余弦相似度变化量化每步贡献，并利用奖励塑形突出关键去噪阶段，提升样本效率与泛化性。**

- **链接: [http://arxiv.org/pdf/2505.19196v1](http://arxiv.org/pdf/2505.19196v1)**

> **作者:** Xinyao Liao; Wei Wei; Xiaoye Qu; Yu Cheng
>
> **摘要:** Recent advances in text-to-image (T2I) diffusion model fine-tuning leverage reinforcement learning (RL) to align generated images with learnable reward functions. The existing approaches reformulate denoising as a Markov decision process for RL-driven optimization. However, they suffer from reward sparsity, receiving only a single delayed reward per generated trajectory. This flaw hinders precise step-level attribution of denoising actions, undermines training efficiency. To address this, we propose a simple yet effective credit assignment framework that dynamically distributes dense rewards across denoising steps. Specifically, we track changes in cosine similarity between intermediate and final images to quantify each step's contribution on progressively reducing the distance to the final image. Our approach avoids additional auxiliary neural networks for step-level preference modeling and instead uses reward shaping to highlight denoising phases that have a greater impact on image quality. Our method achieves 1.25 to 2 times higher sample efficiency and better generalization across four human preference reward functions, without compromising the original optimal policy.
>
---
#### [new 107] MotionPro: A Precise Motion Controller for Image-to-Video Generation
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出MotionPro，用于图像到视频生成的精准运动控制。针对现有方法运动控制粗糙且无法区分物体与相机运动的问题，其通过区域轨迹和运动掩码实现细粒度运动调节与分类，并构建MC-Bench基准进行评估，增强视频生成质量。**

- **链接: [http://arxiv.org/pdf/2505.20287v1](http://arxiv.org/pdf/2505.20287v1)**

> **作者:** Zhongwei Zhang; Fuchen Long; Zhaofan Qiu; Yingwei Pan; Wu Liu; Ting Yao; Tao Mei
>
> **备注:** CVPR 2025. Project page: https://zhw-zhang.github.io/MotionPro-page/
>
> **摘要:** Animating images with interactive motion control has garnered popularity for image-to-video (I2V) generation. Modern approaches typically rely on large Gaussian kernels to extend motion trajectories as condition without explicitly defining movement region, leading to coarse motion control and failing to disentangle object and camera moving. To alleviate these, we present MotionPro, a precise motion controller that novelly leverages region-wise trajectory and motion mask to regulate fine-grained motion synthesis and identify target motion category (i.e., object or camera moving), respectively. Technically, MotionPro first estimates the flow maps on each training video via a tracking model, and then samples the region-wise trajectories to simulate inference scenario. Instead of extending flow through large Gaussian kernels, our region-wise trajectory approach enables more precise control by directly utilizing trajectories within local regions, thereby effectively characterizing fine-grained movements. A motion mask is simultaneously derived from the predicted flow maps to capture the holistic motion dynamics of the movement regions. To pursue natural motion control, MotionPro further strengthens video denoising by incorporating both region-wise trajectories and motion mask through feature modulation. More remarkably, we meticulously construct a benchmark, i.e., MC-Bench, with 1.1K user-annotated image-trajectory pairs, for the evaluation of both fine-grained and object-level I2V motion control. Extensive experiments conducted on WebVid-10M and MC-Bench demonstrate the effectiveness of MotionPro. Please refer to our project page for more results: https://zhw-zhang.github.io/MotionPro-page/.
>
---
#### [new 108] GoLF-NRT: Integrating Global Context and Local Geometry for Few-Shot View Synthesis
- **分类: cs.CV**

- **简介: 该论文属于少样本视图合成任务，旨在解决传统NeRF模型在输入视图较少时渲染质量下降的问题。提出GoLF-NRT方法，融合全局场景上下文（3D稀疏注意力Transformer）与局部几何特征（沿极线提取），并引入自适应采样策略，实现仅用1-3张输入图像即可高质量重建场景。**

- **链接: [http://arxiv.org/pdf/2505.19813v1](http://arxiv.org/pdf/2505.19813v1)**

> **作者:** You Wang; Li Fang; Hao Zhu; Fei Hu; Long Ye; Zhan Ma
>
> **备注:** CVPR 2025
>
> **摘要:** Neural Radiance Fields (NeRF) have transformed novel view synthesis by modeling scene-specific volumetric representations directly from images. While generalizable NeRF models can generate novel views across unknown scenes by learning latent ray representations, their performance heavily depends on a large number of multi-view observations. However, with limited input views, these methods experience significant degradation in rendering quality. To address this limitation, we propose GoLF-NRT: a Global and Local feature Fusion-based Neural Rendering Transformer. GoLF-NRT enhances generalizable neural rendering from few input views by leveraging a 3D transformer with efficient sparse attention to capture global scene context. In parallel, it integrates local geometric features extracted along the epipolar line, enabling high-quality scene reconstruction from as few as 1 to 3 input views. Furthermore, we introduce an adaptive sampling strategy based on attention weights and kernel regression, improving the accuracy of transformer-based neural rendering. Extensive experiments on public datasets show that GoLF-NRT achieves state-of-the-art performance across varying numbers of input views, highlighting the effectiveness and superiority of our approach. Code is available at https://github.com/KLMAV-CUC/GoLF-NRT.
>
---
#### [new 109] SuperAD: A Training-free Anomaly Classification and Segmentation Method for CVPR 2025 VAND 3.0 Workshop Challenge Track 1: Adapt & Detect
- **分类: cs.CV**

- **简介: 论文提出SuperAD，一种无需训练的异常分类与分割方法，针对工业场景中透明/反光表面、遮挡等复杂异常检测问题。基于DINOv2模型提取特征构建记忆库，通过最近邻匹配实现异常分割，在MVTec AD2数据集取得竞争力结果。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19750v1](http://arxiv.org/pdf/2505.19750v1)**

> **作者:** Huaiyuan Zhang; Hang Chen; Yu Cheng; Shunyi Wu; Linghao Sun; Linao Han; Zeyu Shi; Lei Qi
>
> **摘要:** In this technical report, we present our solution to the CVPR 2025 Visual Anomaly and Novelty Detection (VAND) 3.0 Workshop Challenge Track 1: Adapt & Detect: Robust Anomaly Detection in Real-World Applications. In real-world industrial anomaly detection, it is crucial to accurately identify anomalies with physical complexity, such as transparent or reflective surfaces, occlusions, and low-contrast contaminations. The recently proposed MVTec AD 2 dataset significantly narrows the gap between publicly available benchmarks and anomalies found in real-world industrial environments. To address the challenges posed by this dataset--such as complex and varying lighting conditions and real anomalies with large scale differences--we propose a fully training-free anomaly detection and segmentation method based on feature extraction using the DINOv2 model named SuperAD. Our method carefully selects a small number of normal reference images and constructs a memory bank by leveraging the strong representational power of DINOv2. Anomalies are then segmented by performing nearest neighbor matching between test image features and the memory bank. Our method achieves competitive results on both test sets of the MVTec AD 2 dataset.
>
---
#### [new 110] WeedNet: A Foundation Model-Based Global-to-Local AI Approach for Real-Time Weed Species Identification and Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出WeedNet模型，解决杂草识别中数据不足与形态复杂问题。通过自监督学习和全局到局部策略，实现跨1593种杂草91.02%的识别准确率，区域性模型达97.38%。适用于无人机及农业机器人，提供智能咨询工具。**

- **链接: [http://arxiv.org/pdf/2505.18930v1](http://arxiv.org/pdf/2505.18930v1)**

> **作者:** Yanben Shen; Timilehin T. Ayanlade; Venkata Naresh Boddepalli; Mojdeh Saadati; Ashlyn Rairdin; Zi K. Deng; Muhammad Arbab Arshad; Aditya Balu; Daren Mueller; Asheesh K Singh; Wesley Everman; Nirav Merchant; Baskar Ganapathysubramanian; Meaghan Anderson; Soumik Sarkar; Arti Singh
>
> **摘要:** Early identification of weeds is essential for effective management and control, and there is growing interest in automating the process using computer vision techniques coupled with AI methods. However, challenges associated with training AI-based weed identification models, such as limited expert-verified data and complexity and variability in morphological features, have hindered progress. To address these issues, we present WeedNet, the first global-scale weed identification model capable of recognizing an extensive set of weed species, including noxious and invasive plant species. WeedNet is an end-to-end real-time weed identification pipeline and uses self-supervised learning, fine-tuning, and enhanced trustworthiness strategies. WeedNet achieved 91.02% accuracy across 1,593 weed species, with 41% species achieving 100% accuracy. Using a fine-tuning strategy and a Global-to-Local approach, the local Iowa WeedNet model achieved an overall accuracy of 97.38% for 85 Iowa weeds, most classes exceeded a 90% mean accuracy per class. Testing across intra-species dissimilarity (developmental stages) and inter-species similarity (look-alike species) suggests that diversity in the images collected, spanning all the growth stages and distinguishable plant characteristics, is crucial in driving model performance. The generalizability and adaptability of the Global WeedNet model enable it to function as a foundational model, with the Global-to-Local strategy allowing fine-tuning for region-specific weed communities. Additional validation of drone- and ground-rover-based images highlights the potential of WeedNet for integration into robotic platforms. Furthermore, integration with AI for conversational use provides intelligent agricultural and ecological conservation consulting tools for farmers, agronomists, researchers, land managers, and government agencies across diverse landscapes.
>
---
#### [new 111] Syn3DTxt: Embedding 3D Cues for Scene Text Generation
- **分类: cs.CV**

- **简介: 论文属于场景文本生成任务，针对现有2D数据缺乏3D几何线索导致空间建模不足的问题，提出通过添加表面法线构建新合成数据集标准，增强三维特征表示，实验验证其提升复杂3D场景文本渲染效果。**

- **链接: [http://arxiv.org/pdf/2505.18479v1](http://arxiv.org/pdf/2505.18479v1)**

> **作者:** Li-Syun Hsiung; Jun-Kai Tu; Kuan-Wu Chu; Yu-Hsuan Chiu; Yan-Tsung Peng; Sheng-Luen Chung; Gee-Sern Jison Hsu
>
> **备注:** CVPR workshop 2025: SyntaGen
>
> **摘要:** This study aims to investigate the challenge of insufficient three-dimensional context in synthetic datasets for scene text rendering. Although recent advances in diffusion models and related techniques have improved certain aspects of scene text generation, most existing approaches continue to rely on 2D data, sourcing authentic training examples from movie posters and book covers, which limits their ability to capture the complex interactions among spatial layout and visual effects in real-world scenes. In particular, traditional 2D datasets do not provide the necessary geometric cues for accurately embedding text into diverse backgrounds. To address this limitation, we propose a novel standard for constructing synthetic datasets that incorporates surface normals to enrich three-dimensional scene characteristic. By adding surface normals to conventional 2D data, our approach aims to enhance the representation of spatial relationships and provide a more robust foundation for future scene text rendering methods. Extensive experiments demonstrate that datasets built under this new standard offer improved geometric context, facilitating further advancements in text rendering under complex 3D-spatial conditions.
>
---
#### [new 112] AdaTP: Attention-Debiased Token Pruning for Video Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AdaTP方法，针对视频大语言模型中视觉注意力偏差问题（全局两端偏好与局部空间过聚焦），设计双去偏模块实现无监督视觉token剪枝，在LLaVA-OneVision-7B上仅用27.3%算力即保持原性能。**

- **链接: [http://arxiv.org/pdf/2505.20100v1](http://arxiv.org/pdf/2505.20100v1)**

> **作者:** Fengyuan Sun; Leqi Shen; Hui Chen; Sicheng Zhao; Jungong Han; Guiguang Ding
>
> **摘要:** Video Large Language Models (Video LLMs) have achieved remarkable results in video understanding tasks. However, they often suffer from heavy computational overhead due to the large number of visual tokens generated from multiple video frames. Existing visual token compression methods often rely on attention scores from language models as guidance. However, these scores exhibit inherent biases: global bias reflects a tendency to focus on the two ends of the visual token sequence, while local bias leads to an over-concentration on the same spatial positions across different frames. To address the issue of attention bias, we propose $\textbf{A}$ttention-$\textbf{D}$ebi$\textbf{a}$sed $\textbf{T}$oken $\textbf{P}$runing for Video Large Language Models ($\textbf{AdaTP}$), a novel token pruning pipeline for Video LLMs. AdaTP integrates two dedicated debiasing modules into the pipeline, targeting global attention bias and local attention bias, respectively. Without the need for additional training, our method significantly reduces the computational overhead of Video LLMs while retaining the performance of vanilla models. Extensive evaluation shows that AdaTP achieves state-of-the-art performance in various commonly used video understanding benchmarks. In particular, on LLaVA-OneVision-7B, AdaTP maintains performance without degradation while using only up to $27.3\%$ FLOPs compared to the vanilla model. Our code will be released soon.
>
---
#### [new 113] TK-Mamba: Marrying KAN with Mamba for Text-Driven 3D Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出TK-Mamba框架，针对3D医学图像分割中高维数据与空间建模效率低的问题，结合Mamba序列模型与KAN网络，创新性地引入EGSC模块、3D-GR-KAN架构及双分支文本驱动策略，提升分割精度与效率，属文本驱动的3D医学图像分割任务。**

- **链接: [http://arxiv.org/pdf/2505.18525v1](http://arxiv.org/pdf/2505.18525v1)**

> **作者:** Haoyu Yang; Yuxiang Cai; Jintao Chen; Xuhong Zhang; Wenhui Lei; Xiaoming Shi; Jianwei Yin; Yankai Jiang
>
> **摘要:** 3D medical image segmentation is vital for clinical diagnosis and treatment but is challenged by high-dimensional data and complex spatial dependencies. Traditional single-modality networks, such as CNNs and Transformers, are often limited by computational inefficiency and constrained contextual modeling in 3D settings. We introduce a novel multimodal framework that leverages Mamba and Kolmogorov-Arnold Networks (KAN) as an efficient backbone for long-sequence modeling. Our approach features three key innovations: First, an EGSC (Enhanced Gated Spatial Convolution) module captures spatial information when unfolding 3D images into 1D sequences. Second, we extend Group-Rational KAN (GR-KAN), a Kolmogorov-Arnold Networks variant with rational basis functions, into 3D-Group-Rational KAN (3D-GR-KAN) for 3D medical imaging - its first application in this domain - enabling superior feature representation tailored to volumetric data. Third, a dual-branch text-driven strategy leverages CLIP's text embeddings: one branch swaps one-hot labels for semantic vectors to preserve inter-organ semantic relationships, while the other aligns images with detailed organ descriptions to enhance semantic alignment. Experiments on the Medical Segmentation Decathlon (MSD) and KiTS23 datasets show our method achieving state-of-the-art performance, surpassing existing approaches in accuracy and efficiency. This work highlights the power of combining advanced sequence modeling, extended network architectures, and vision-language synergy to push forward 3D medical image segmentation, delivering a scalable solution for clinical use. The source code is openly available at https://github.com/yhy-whu/TK-Mamba.
>
---
#### [new 114] Multi-Timescale Motion-Decoupled Spiking Transformer for Audio-Visual Zero-Shot Learning
- **分类: cs.CV**

- **简介: 该论文针对音频视觉零样本学习（ZSL）中背景偏差和运动细节不足的问题，提出双流多时间尺度脉冲变换器MDST++。通过解耦语义（循环跨模态联合学习）与动态运动信息（RGB转事件表征），结合音频运动分析模块及动态神经元阈值调节，提升时序与运动特征鲁棒性，实验显示其HM和ZSL准确率分别提升26.2%和39.9%。**

- **链接: [http://arxiv.org/pdf/2505.19938v1](http://arxiv.org/pdf/2505.19938v1)**

> **作者:** Wenrui Li; Penghong Wang; Xingtao Wang; Wangmeng Zuo; Xiaopeng Fan; Yonghong Tian
>
> **备注:** Accepted by IEEE TCSVT
>
> **摘要:** Audio-visual zero-shot learning (ZSL) has been extensively researched for its capability to classify video data from unseen classes during training. Nevertheless, current methodologies often struggle with background scene biases and inadequate motion detail. This paper proposes a novel dual-stream Multi-Timescale Motion-Decoupled Spiking Transformer (MDST++), which decouples contextual semantic information and sparse dynamic motion information. The recurrent joint learning unit is proposed to extract contextual semantic information and capture joint knowledge across various modalities to understand the environment of actions. By converting RGB images to events, our method captures motion information more accurately and mitigates background scene biases. Moreover, we introduce a discrepancy analysis block to model audio motion information. To enhance the robustness of SNNs in extracting temporal and motion cues, we dynamically adjust the threshold of Leaky Integrate-and-Fire neurons based on global motion and contextual semantic information. Our experiments validate the effectiveness of MDST++, demonstrating their consistent superiority over state-of-the-art methods on mainstream benchmarks. Additionally, incorporating motion and multi-timescale information significantly improves HM and ZSL accuracy by 26.2\% and 39.9\%.
>
---
#### [new 115] Enhancing Visual Reliance in Text Generation: A Bayesian Perspective on Mitigating Hallucination in Large Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对大型视觉语言模型（LVLM）文本生成与视觉输入不符的幻觉问题，提出三方面改进：筛选冗余视觉标记、修正先验分布、后验崩溃时终止生成，从贝叶斯视角增强视觉依赖以减少幻觉，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.19498v1](http://arxiv.org/pdf/2505.19498v1)**

> **作者:** Nanxing Hu; Xiaoyue Duan; Jinchao Zhang; Guoliang Kang
>
> **摘要:** Large Vision-Language Models (LVLMs) usually generate texts which satisfy context coherence but don't match the visual input. Such a hallucination issue hinders LVLMs' applicability in the real world. The key to solving hallucination in LVLM is to make the text generation rely more on the visual content. Most previous works choose to enhance/adjust the features/output of a specific modality (i.e., visual or textual) to alleviate hallucinations in LVLM, which do not explicitly or systematically enhance the visual reliance. In this paper, we comprehensively investigate the factors which may degenerate the visual reliance in text generation of LVLM from a Bayesian perspective. Based on our observations, we propose to mitigate hallucination in LVLM from three aspects. Firstly, we observe that not all visual tokens are informative in generating meaningful texts. We propose to evaluate and remove redundant visual tokens to avoid their disturbance. Secondly, LVLM may encode inappropriate prior information, making it lean toward generating unexpected words. We propose a simple yet effective way to rectify the prior from a Bayesian perspective. Thirdly, we observe that starting from certain steps, the posterior of next-token prediction conditioned on visual tokens may collapse to a prior distribution which does not depend on any informative visual tokens at all. Thus, we propose to stop further text generation to avoid hallucination. Extensive experiments on three benchmarks including POPE, CHAIR, and MME demonstrate that our method can consistently mitigate the hallucination issue of LVLM and performs favorably against previous state-of-the-arts.
>
---
#### [new 116] HF-VTON: High-Fidelity Virtual Try-On via Consistent Geometric and Semantic Alignment
- **分类: cs.CV**

- **简介: 该论文提出HF-VTON框架，解决虚拟试衣中多姿势下的几何扭曲、语义不匹配及细节丢失问题。通过几何对齐、语义增强和多模态生成三模块，实现跨姿态的高保真服装合成，并构建SAMP-VTONS数据集提升评估全面性，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19638v1](http://arxiv.org/pdf/2505.19638v1)**

> **作者:** Ming Meng; Qi Dong; Jiajie Li; Zhe Zhu; Xingyu Wang; Zhaoxin Fan; Wei Zhao; Wenjun Wu
>
> **摘要:** Virtual try-on technology has become increasingly important in the fashion and retail industries, enabling the generation of high-fidelity garment images that adapt seamlessly to target human models. While existing methods have achieved notable progress, they still face significant challenges in maintaining consistency across different poses. Specifically, geometric distortions lead to a lack of spatial consistency, mismatches in garment structure and texture across poses result in semantic inconsistency, and the loss or distortion of fine-grained details diminishes visual fidelity. To address these challenges, we propose HF-VTON, a novel framework that ensures high-fidelity virtual try-on performance across diverse poses. HF-VTON consists of three key modules: (1) the Appearance-Preserving Warp Alignment Module (APWAM), which aligns garments to human poses, addressing geometric deformations and ensuring spatial consistency; (2) the Semantic Representation and Comprehension Module (SRCM), which captures fine-grained garment attributes and multi-pose data to enhance semantic representation, maintaining structural, textural, and pattern consistency; and (3) the Multimodal Prior-Guided Appearance Generation Module (MPAGM), which integrates multimodal features and prior knowledge from pre-trained models to optimize appearance generation, ensuring both semantic and geometric consistency. Additionally, to overcome data limitations in existing benchmarks, we introduce the SAMP-VTONS dataset, featuring multi-pose pairs and rich textual annotations for a more comprehensive evaluation. Experimental results demonstrate that HF-VTON outperforms state-of-the-art methods on both VITON-HD and SAMP-VTONS, excelling in visual fidelity, semantic consistency, and detail preservation.
>
---
#### [new 117] Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Chain-of-Zoom（CoZ）框架，解决单图像超分辨率模型在远超训练尺度时效果崩溃的问题。通过自回归链分解放大过程，复用基础模型并结合多尺度文本提示（由视觉语言模型生成和优化），实现256倍以上高质量放大。任务为极端超分辨率，核心是可扩展性与视觉保真度的平衡。**

- **链接: [http://arxiv.org/pdf/2505.18600v1](http://arxiv.org/pdf/2505.18600v1)**

> **作者:** Bryan Sangwoo Kim; Jeongsol Kim; Jong Chul Ye
>
> **摘要:** Modern single-image super-resolution (SISR) models deliver photo-realistic results at the scale factors on which they are trained, but collapse when asked to magnify far beyond that regime. We address this scalability bottleneck with Chain-of-Zoom (CoZ), a model-agnostic framework that factorizes SISR into an autoregressive chain of intermediate scale-states with multi-scale-aware prompts. CoZ repeatedly re-uses a backbone SR model, decomposing the conditional probability into tractable sub-problems to achieve extreme resolutions without additional training. Because visual cues diminish at high magnifications, we augment each zoom step with multi-scale-aware text prompts generated by a vision-language model (VLM). The prompt extractor itself is fine-tuned using Generalized Reward Policy Optimization (GRPO) with a critic VLM, aligning text guidance towards human preference. Experiments show that a standard 4x diffusion SR model wrapped in CoZ attains beyond 256x enlargement with high perceptual quality and fidelity.
>
---
#### [new 118] VORTA: Efficient Video Diffusion via Routing Sparse Attention
- **分类: cs.CV**

- **简介: 该论文提出VORTA，加速视频扩散模型（VDiTs）的视频生成。针对其高计算成本及冗余长程注意力问题，设计稀疏注意力机制捕捉长程依赖，并通过自适应路由策略替换全3D注意力，实现1.76倍加速且无质量损失，结合其他方法达14.41倍加速，提升VDiTs实用性。**

- **链接: [http://arxiv.org/pdf/2505.18809v1](http://arxiv.org/pdf/2505.18809v1)**

> **作者:** Wenhao Sun; Rong-Cheng Tu; Yifu Ding; Zhao Jin; Jingyi Liao; Shunyu Liu; Dacheng Tao
>
> **备注:** 19 pages, 15 figures. The code is available at https://github.com/wenhao728/VORTA
>
> **摘要:** Video Diffusion Transformers (VDiTs) have achieved remarkable progress in high-quality video generation, but remain computationally expensive due to the quadratic complexity of attention over high-dimensional video sequences. Recent attention acceleration methods leverage the sparsity of attention patterns to improve efficiency; however, they often overlook inefficiencies of redundant long-range interactions. To address this problem, we propose \textbf{VORTA}, an acceleration framework with two novel components: 1) a sparse attention mechanism that efficiently captures long-range dependencies, and 2) a routing strategy that adaptively replaces full 3D attention with specialized sparse attention variants throughout the sampling process. It achieves a $1.76\times$ end-to-end speedup without quality loss on VBench. Furthermore, VORTA can seamlessly integrate with various other acceleration methods, such as caching and step distillation, reaching up to $14.41\times$ speedup with negligible performance degradation. VORTA demonstrates its efficiency and enhances the practicality of VDiTs in real-world settings.
>
---
#### [new 119] Dual-Path Stable Soft Prompt Generation for Domain Generalization
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于领域泛化任务，针对现有提示生成方法中因随机种子导致的提示不稳定问题（Prompt Variability），提出DPSPG框架。其通过引入负学习的双路径生成结构，稳定并优化动态提示，提升模型跨领域泛化性能。**

- **链接: [http://arxiv.org/pdf/2505.18770v1](http://arxiv.org/pdf/2505.18770v1)**

> **作者:** Yuedi Zhang; Shuanghao Bai; Wanqi Zhou; Zhirong Luan; Badong Chen
>
> **摘要:** Domain generalization (DG) aims to learn a model using data from one or multiple related but distinct source domains that can generalize well to unseen out-of-distribution target domains. Inspired by the success of large pre-trained vision-language models (VLMs), prompt tuning has emerged as an effective generalization strategy. However, it often struggles to capture domain-specific features due to its reliance on manually or fixed prompt inputs. Recently, some prompt generation methods have addressed this limitation by dynamically generating instance-specific and domain-specific prompts for each input, enriching domain information and demonstrating potential for enhanced generalization. Through further investigation, we identify a notable issue in existing prompt generation methods: the same input often yields significantly different and suboptimal prompts across different random seeds, a phenomenon we term Prompt Variability. To address this, we introduce negative learning into the prompt generation process and propose Dual-Path Stable Soft Prompt Generation (DPSPG), a transformer-based framework designed to improve both the stability and generalization of prompts. Specifically, DPSPG incorporates a complementary prompt generator to produce negative prompts, thereby reducing the risk of introducing misleading information. Both theoretical and empirical analyses demonstrate that negative learning leads to more robust and effective prompts by increasing the effective margin and reducing the upper bound of the gradient norm. Extensive experiments on five DG benchmark datasets show that DPSPG consistently outperforms state-of-the-art methods while maintaining prompt stability.
>
---
#### [new 120] Few-Shot Class-Incremental Learning For Efficient SAR Automatic Target Recognition
- **分类: cs.CV**

- **简介: 该论文提出基于双分支架构的 Few-Shot 类增量学习框架，解决 SAR 目标识别中的数据稀缺与增量学习挑战。通过离散傅里叶变换、全局滤波器及轻量注意力机制提取局部与全局特征，结合焦损失与中心损失优化类别边界，实验显示其在 MSTAR 数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19565v1](http://arxiv.org/pdf/2505.19565v1)**

> **作者:** George Karantaidis; Athanasios Pantsios; Ioannis Kompatsiaris; Symeon Papadopoulos
>
> **摘要:** Synthetic aperture radar automatic target recognition (SAR-ATR) systems have rapidly evolved to tackle incremental recognition challenges in operational settings. Data scarcity remains a major hurdle that conventional SAR-ATR techniques struggle to address. To cope with this challenge, we propose a few-shot class-incremental learning (FSCIL) framework based on a dual-branch architecture that focuses on local feature extraction and leverages the discrete Fourier transform and global filters to capture long-term spatial dependencies. This incorporates a lightweight cross-attention mechanism that fuses domain-specific features with global dependencies to ensure robust feature interaction, while maintaining computational efficiency by introducing minimal scale-shift parameters. The framework combines focal loss for class distinction under imbalance and center loss for compact intra-class distributions to enhance class separation boundaries. Experimental results on the MSTAR benchmark dataset demonstrate that the proposed framework consistently outperforms state-of-the-art methods in FSCIL SAR-ATR, attesting to its effectiveness in real-world scenarios.
>
---
#### [new 121] OpenS2V-Nexus: A Detailed Benchmark and Million-Scale Dataset for Subject-to-Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于Subject-to-Video（S2V）生成任务，旨在解决现有基准评估粗放及缺乏大规模数据的问题。提出OpenS2V-Nexus框架，包含细粒度评估基准OpenS2V-Eval（含180个测试提示及3个新指标）和5M规模数据集，提升视频主体一致性、自然度及文本相关性评估，推动S2V研究。**

- **链接: [http://arxiv.org/pdf/2505.20292v1](http://arxiv.org/pdf/2505.20292v1)**

> **作者:** Shenghai Yuan; Xianyi He; Yufan Deng; Yang Ye; Jinfa Huang; Bin Lin; Chongyang Ma; Jiebo Luo; Li Yuan
>
> **备注:** Code and Dataset: https://github.com/PKU-YuanGroup/OpenS2V-Nexus
>
> **摘要:** Subject-to-Video (S2V) generation aims to create videos that faithfully incorporate reference content, providing enhanced flexibility in the production of videos. To establish the infrastructure for S2V generation, we propose OpenS2V-Nexus, consisting of (i) OpenS2V-Eval, a fine-grained benchmark, and (ii) OpenS2V-5M, a million-scale dataset. In contrast to existing S2V benchmarks inherited from VBench that focus on global and coarse-grained assessment of generated videos, OpenS2V-Eval focuses on the model's ability to generate subject-consistent videos with natural subject appearance and identity fidelity. For these purposes, OpenS2V-Eval introduces 180 prompts from seven major categories of S2V, which incorporate both real and synthetic test data. Furthermore, to accurately align human preferences with S2V benchmarks, we propose three automatic metrics, NexusScore, NaturalScore and GmeScore, to separately quantify subject consistency, naturalness, and text relevance in generated videos. Building on this, we conduct a comprehensive evaluation of 16 representative S2V models, highlighting their strengths and weaknesses across different content. Moreover, we create the first open-source large-scale S2V generation dataset OpenS2V-5M, which consists of five million high-quality 720P subject-text-video triples. Specifically, we ensure subject-information diversity in our dataset by (1) segmenting subjects and building pairing information via cross-video associations and (2) prompting GPT-Image-1 on raw frames to synthesize multi-view representations. Through OpenS2V-Nexus, we deliver a robust infrastructure to accelerate future S2V generation research.
>
---
#### [new 122] Deep Spectral Prior
- **分类: cs.CV; cs.NA; math.NA**

- **简介: 该论文属于图像重建任务，旨在解决传统Deep Image Prior（DIP）依赖像素损失和早停导致过拟合的问题。提出Deep Spectral Prior（DSP），通过直接匹配输出与观测的傅里叶系数，引入频谱一致性偏置，抑制高频噪声并消除早停需求。理论分析其频谱正则化特性，并在去噪、修复和超分辨率任务中验证优于传统方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19873v1](http://arxiv.org/pdf/2505.19873v1)**

> **作者:** Yanqi Cheng; Tieyong Zeng; Pietro Lio; Carola-Bibiane Schönlieb; Angelica I Aviles-Rivero
>
> **摘要:** We introduce Deep Spectral Prior (DSP), a new formulation of Deep Image Prior (DIP) that redefines image reconstruction as a frequency-domain alignment problem. Unlike traditional DIP, which relies on pixel-wise loss and early stopping to mitigate overfitting, DSP directly matches Fourier coefficients between the network output and observed measurements. This shift introduces an explicit inductive bias towards spectral coherence, aligning with the known frequency structure of images and the spectral bias of convolutional neural networks. We provide a rigorous theoretical framework demonstrating that DSP acts as an implicit spectral regulariser, suppressing high-frequency noise by design and eliminating the need for early stopping. Our analysis spans four core dimensions establishing smooth convergence dynamics, local stability, and favourable bias-variance tradeoffs. We further show that DSP naturally projects reconstructions onto a frequency-consistent manifold, enhancing interpretability and robustness. These theoretical guarantees are supported by empirical results across denoising, inpainting, and super-resolution tasks, where DSP consistently outperforms classical DIP and other unsupervised baselines.
>
---
#### [new 123] Words as Geometric Features: Estimating Homography using Optical Character Recognition as Compressed Image Representation
- **分类: cs.CV**

- **简介: 该论文提出一种基于OCR文本的文档配准方法，解决传统图像特征对原始图像依赖的问题。通过利用OCR检测的词语空间位置和文本内容估计单应性矩阵，并采用RANSAC处理OCR噪声，实验证明其精度与效率优于传统图像方法，适用于无原始图像的场景。**

- **链接: [http://arxiv.org/pdf/2505.18925v1](http://arxiv.org/pdf/2505.18925v1)**

> **作者:** Ross Greer; Alisha Ukani; Katherine Izhikevich; Earlence Fernandes; Stefan Savage; Alex C. Snoeren
>
> **摘要:** Document alignment and registration play a crucial role in numerous real-world applications, such as automated form processing, anomaly detection, and workflow automation. Traditional methods for document alignment rely on image-based features like keypoints, edges, and textures to estimate geometric transformations, such as homographies. However, these approaches often require access to the original document images, which may not always be available due to privacy, storage, or transmission constraints. This paper introduces a novel approach that leverages Optical Character Recognition (OCR) outputs as features for homography estimation. By utilizing the spatial positions and textual content of OCR-detected words, our method enables document alignment without relying on pixel-level image data. This technique is particularly valuable in scenarios where only OCR outputs are accessible. Furthermore, the method is robust to OCR noise, incorporating RANSAC to handle outliers and inaccuracies in the OCR data. On a set of test documents, we demonstrate that our OCR-based approach even performs more accurately than traditional image-based methods, offering a more efficient and scalable solution for document registration tasks. The proposed method facilitates applications in document processing, all while reducing reliance on high-dimensional image data.
>
---
#### [new 124] AniCrafter: Customizing Realistic Human-Centric Animation via Avatar-Background Conditioning in Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出AniCrafter，属于human-centric视频动画任务。针对现有方法依赖基础结构条件导致动态背景和复杂人体姿态下效果差的问题，创新性地引入avatar-背景联合建模机制，将动画生成重构为修复任务，基于I2V扩散模型实现角色与动态场景的稳定融合，实验验证了方法优势。**

- **链接: [http://arxiv.org/pdf/2505.20255v1](http://arxiv.org/pdf/2505.20255v1)**

> **作者:** Muyao Niu; Mingdeng Cao; Yifan Zhan; Qingtian Zhu; Mingze Ma; Jiancheng Zhao; Yanhong Zeng; Zhihang Zhong; Xiao Sun; Yinqiang Zheng
>
> **备注:** Github: https://github.com/MyNiuuu/AniCrafter
>
> **摘要:** Recent advances in video diffusion models have significantly improved character animation techniques. However, current approaches rely on basic structural conditions such as DWPose or SMPL-X to animate character images, limiting their effectiveness in open-domain scenarios with dynamic backgrounds or challenging human poses. In this paper, we introduce $\textbf{AniCrafter}$, a diffusion-based human-centric animation model that can seamlessly integrate and animate a given character into open-domain dynamic backgrounds while following given human motion sequences. Built on cutting-edge Image-to-Video (I2V) diffusion architectures, our model incorporates an innovative "avatar-background" conditioning mechanism that reframes open-domain human-centric animation as a restoration task, enabling more stable and versatile animation outputs. Experimental results demonstrate the superior performance of our method. Codes will be available at https://github.com/MyNiuuu/AniCrafter.
>
---
#### [new 125] Reasoning Segmentation for Images and Videos: A Survey
- **分类: cs.CV**

- **简介: 该论文属图像/视频推理分割综述任务，解决传统分割依赖固定类别的局限，通过自然语言查询实现需推理的知识整合分割。工作包括分析26种方法、29个数据集及指标，探讨应用并指出研究缺口与未来方向。**

- **链接: [http://arxiv.org/pdf/2505.18816v1](http://arxiv.org/pdf/2505.18816v1)**

> **作者:** Yiqing Shen; Chenjia Li; Fei Xiong; Jeong-O Jeong; Tianpeng Wang; Michael Latman; Mathias Unberath
>
> **摘要:** Reasoning Segmentation (RS) aims to delineate objects based on implicit text queries, the interpretation of which requires reasoning and knowledge integration. Unlike the traditional formulation of segmentation problems that relies on fixed semantic categories or explicit prompting, RS bridges the gap between visual perception and human-like reasoning capabilities, facilitating more intuitive human-AI interaction through natural language. Our work presents the first comprehensive survey of RS for image and video processing, examining 26 state-of-the-art methods together with a review of the corresponding evaluation metrics, as well as 29 datasets and benchmarks. We also explore existing applications of RS across diverse domains and identify their potential extensions. Finally, we identify current research gaps and highlight promising future directions.
>
---
#### [new 126] VisualToolAgent (VisTA): A Reinforcement Learning Framework for Visual Tool Selection
- **分类: cs.CV**

- **简介: 该论文提出VisTA框架，解决现有视觉工具选择方法缺乏主动探索、工具多样性不足及依赖人工监督的问题。通过强化学习和GRPO算法，实现自主动态选择工具，提升任务表现与泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.20289v1](http://arxiv.org/pdf/2505.20289v1)**

> **作者:** Zeyi Huang; Yuyang Ji; Anirudh Sundara Rajan; Zefan Cai; Wen Xiao; Junjie Hu; Yong Jae Lee
>
> **摘要:** We introduce VisTA, a new reinforcement learning framework that empowers visual agents to dynamically explore, select, and combine tools from a diverse library based on empirical performance. Existing methods for tool-augmented reasoning either rely on training-free prompting or large-scale fine-tuning; both lack active tool exploration and typically assume limited tool diversity, and fine-tuning methods additionally demand extensive human supervision. In contrast, VisTA leverages end-to-end reinforcement learning to iteratively refine sophisticated, query-specific tool selection strategies, using task outcomes as feedback signals. Through Group Relative Policy Optimization (GRPO), our framework enables an agent to autonomously discover effective tool-selection pathways without requiring explicit reasoning supervision. Experiments on the ChartQA, Geometry3K, and BlindTest benchmarks demonstrate that VisTA achieves substantial performance gains over training-free baselines, especially on out-of-distribution examples. These results highlight VisTA's ability to enhance generalization, adaptively utilize diverse tools, and pave the way for flexible, experience-driven visual reasoning systems.
>
---
#### [new 127] ChartGalaxy: A Dataset for Infographic Chart Understanding and Generation
- **分类: cs.CV; cs.CL**

- **简介: 论文提出ChartGalaxy百万级数据集，用于信息图表的多模态理解和生成。针对现有LVLM难以处理复杂图表的问题，其通过归纳75种类型、330变体及68布局模板生成合成数据，并用于模型微调、代码生成基准及示例生成，提升多模态推理能力。**

- **链接: [http://arxiv.org/pdf/2505.18668v1](http://arxiv.org/pdf/2505.18668v1)**

> **作者:** Zhen Li; Yukai Guo; Duan Li; Xinyuan Guo; Bowen Li; Lanxi Xiao; Shenyu Qiao; Jiashu Chen; Zijian Wu; Hui Zhang; Xinhuan Shu; Shixia Liu
>
> **备注:** 63 pages, submitted to NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** Infographic charts are a powerful medium for communicating abstract data by combining visual elements (e.g., charts, images) with textual information. However, their visual and structural richness poses challenges for large vision-language models (LVLMs), which are typically trained on plain charts. To bridge this gap, we introduce ChartGalaxy, a million-scale dataset designed to advance the understanding and generation of infographic charts. The dataset is constructed through an inductive process that identifies 75 chart types, 330 chart variations, and 68 layout templates from real infographic charts and uses them to create synthetic ones programmatically. We showcase the utility of this dataset through: 1) improving infographic chart understanding via fine-tuning, 2) benchmarking code generation for infographic charts, and 3) enabling example-based infographic chart generation. By capturing the visual and structural complexity of real design, ChartGalaxy provides a useful resource for enhancing multimodal reasoning and generation in LVLMs.
>
---
#### [new 128] TNG-CLIP:Training-Time Negation Data Generation for Negation Awareness of CLIP
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TNG-CLIP，旨在提升CLIP对否定概念（如"无"或"排除"）的理解。针对现有方法依赖LLM生成否定数据且计算成本高、评估范围窄的问题，提出训练时高效生成否定标注数据（增加2.5%时间）及首个评估文本到图像生成的否定基准Neg-TtoI。实验显示TNG-CLIP在图文匹配、检索及图像生成任务中达最优性能。**

- **链接: [http://arxiv.org/pdf/2505.18434v1](http://arxiv.org/pdf/2505.18434v1)**

> **作者:** Yuliang Cai; Jesse Thomason; Mohammad Rostami
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Vision-language models (VLMs), such as CLIP, have demonstrated strong performance across a range of downstream tasks. However, CLIP is still limited in negation understanding: the ability to recognize the absence or exclusion of a concept. Existing methods address the problem by using a large language model (LLM) to generate large-scale data of image captions containing negation for further fine-tuning CLIP. However, these methods are both time- and compute-intensive, and their evaluations are typically restricted to image-text matching tasks. To expand the horizon, we (1) introduce a training-time negation data generation pipeline such that negation captions are generated during the training stage, which only increases 2.5% extra training time, and (2) we propose the first benchmark, Neg-TtoI, for evaluating text-to-image generation models on prompts containing negation, assessing model's ability to produce semantically accurate images. We show that our proposed method, TNG-CLIP, achieves SOTA performance on diverse negation benchmarks of image-to-text matching, text-to-image retrieval, and image generation.
>
---
#### [new 129] How Do Images Align and Complement LiDAR? Towards a Harmonized Multi-modal 3D Panoptic Segmentation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于多模态3D全景分割任务，旨在解决LiDAR数据稀疏导致的小/远物体识别困难及多模态融合中的对齐与后处理依赖问题。提出IAL框架，包含同步数据增强（PieAug）、几何引导特征融合（GTF）及先验查询生成（PQG）模块，直接预测分割结果，提升准确性并达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.18956v1](http://arxiv.org/pdf/2505.18956v1)**

> **作者:** Yining Pan; Qiongjie Cui; Xulei Yang; Na Zhao
>
> **备注:** Accepted at the 2025 International Conference on Machine Learning (ICML)
>
> **摘要:** LiDAR-based 3D panoptic segmentation often struggles with the inherent sparsity of data from LiDAR sensors, which makes it challenging to accurately recognize distant or small objects. Recently, a few studies have sought to overcome this challenge by integrating LiDAR inputs with camera images, leveraging the rich and dense texture information provided by the latter. While these approaches have shown promising results, they still face challenges, such as misalignment during data augmentation and the reliance on post-processing steps. To address these issues, we propose Image-Assists-LiDAR (IAL), a novel multi-modal 3D panoptic segmentation framework. In IAL, we first introduce a modality-synchronized data augmentation strategy, PieAug, to ensure alignment between LiDAR and image inputs from the start. Next, we adopt a transformer decoder to directly predict panoptic segmentation results. To effectively fuse LiDAR and image features into tokens for the decoder, we design a Geometric-guided Token Fusion (GTF) module. Additionally, we leverage the complementary strengths of each modality as priors for query initialization through a Prior-based Query Generation (PQG) module, enhancing the decoder's ability to generate accurate instance masks. Our IAL framework achieves state-of-the-art performance compared to previous multi-modal 3D panoptic segmentation methods on two widely used benchmarks. Code and models are publicly available at <https://github.com/IMPL-Lab/IAL.git>.
>
---
#### [new 130] A Regularization-Guided Equivariant Approach for Image Restoration
- **分类: cs.CV**

- **简介: 该论文针对图像修复任务中传统对称性模型精度不足及对严格对称假设依赖的问题，提出旋转等变正则化方法EQ-Reg。通过自监督学习与特征图的空间旋转/通道变换，自适应调整等变约束，在保持表示能力的同时提升精度与泛化性，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19799v1](http://arxiv.org/pdf/2505.19799v1)**

> **作者:** Yulu Bai; Jiahong Fu; Qi Xie; Deyu Meng
>
> **摘要:** Equivariant and invariant deep learning models have been developed to exploit intrinsic symmetries in data, demonstrating significant effectiveness in certain scenarios. However, these methods often suffer from limited representation accuracy and rely on strict symmetry assumptions that may not hold in practice. These limitations pose a significant drawback for image restoration tasks, which demands high accuracy and precise symmetry representation. To address these challenges, we propose a rotation-equivariant regularization strategy that adaptively enforces the appropriate symmetry constraints on the data while preserving the network's representational accuracy. Specifically, we introduce EQ-Reg, a regularizer designed to enhance rotation equivariance, which innovatively extends the insights of data-augmentation-based and equivariant-based methodologies. This is achieved through self-supervised learning and the spatial rotation and cyclic channel shift of feature maps deduce in the equivariant framework. Our approach firstly enables a non-strictly equivariant network suitable for image restoration, providing a simple and adaptive mechanism for adjusting equivariance based on task. Extensive experiments across three low-level tasks demonstrate the superior accuracy and generalization capability of our method, outperforming state-of-the-art approaches.
>
---
#### [new 131] StyleAR: Customizing Multimodal Autoregressive Model for Style-Aligned Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于风格对齐的文本到图像生成任务，旨在解决缺乏大规模风格化三元组数据的难题。提出StyleAR方法，通过二进制数据训练，结合CLIP编码器与风格增强技术，利用混合数据提升风格一致性，实现高质量生成。**

- **链接: [http://arxiv.org/pdf/2505.19874v1](http://arxiv.org/pdf/2505.19874v1)**

> **作者:** Yi Wu; Lingting Zhu; Shengju Qian; Lei Liu; Wandi Qiao; Lequan Yu; Bin Li
>
> **摘要:** In the current research landscape, multimodal autoregressive (AR) models have shown exceptional capabilities across various domains, including visual understanding and generation. However, complex tasks such as style-aligned text-to-image generation present significant challenges, particularly in data acquisition. In analogy to instruction-following tuning for image editing of AR models, style-aligned generation requires a reference style image and prompt, resulting in a text-image-to-image triplet where the output shares the style and semantics of the input. However, acquiring large volumes of such triplet data with specific styles is considerably more challenging than obtaining conventional text-to-image data used for training generative models. To address this issue, we propose StyleAR, an innovative approach that combines a specially designed data curation method with our proposed AR models to effectively utilize text-to-image binary data for style-aligned text-to-image generation. Our method synthesizes target stylized data using a reference style image and prompt, but only incorporates the target stylized image as the image modality to create high-quality binary data. To facilitate binary data training, we introduce a CLIP image encoder with a perceiver resampler that translates the image input into style tokens aligned with multimodal tokens in AR models and implement a style-enhanced token technique to prevent content leakage which is a common issue in previous work. Furthermore, we mix raw images drawn from large-scale text-image datasets with stylized images to enhance StyleAR's ability to extract richer stylistic features and ensure style consistency. Extensive qualitative and quantitative experiments demonstrate our superior performance.
>
---
#### [new 132] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型（LVLMs）高效化任务，旨在解决冗余视觉token导致的计算成本问题。现有方法依赖单层注意力分数剪枝，但无法捕捉层间信息交互。作者提出FlowCut框架，基于信息流分析（发现CLS token的中继作用、冗余动态生成及单层判据矛盾），实现更精准剪枝。实验显示其在LLaVA模型中减少94.4%视觉token同时提升性能，推理加速3.2倍。**

- **链接: [http://arxiv.org/pdf/2505.19536v1](http://arxiv.org/pdf/2505.19536v1)**

> **作者:** Jintao Tong; Wenwei Jin; Pengda Qin; Anqi Li; Yixiong Zou; Yuhong Li; Yuhua Li; Ruixuan Li
>
> **备注:** 19 pages, 11 figures
>
> **摘要:** Large vision-language models (LVLMs) excel at multimodal understanding but suffer from high computational costs due to redundant vision tokens. Existing pruning methods typically rely on single-layer attention scores to rank and prune redundant visual tokens to solve this inefficiency. However, as the interaction between tokens and layers is complicated, this raises a basic question: Is such a simple single-layer criterion sufficient to identify redundancy? To answer this question, we rethink the emergence of redundant visual tokens from a fundamental perspective: information flow, which models the interaction between tokens and layers by capturing how information moves between tokens across layers. We find (1) the CLS token acts as an information relay, which can simplify the complicated flow analysis; (2) the redundancy emerges progressively and dynamically via layer-wise attention concentration; and (3) relying solely on attention scores from single layers can lead to contradictory redundancy identification. Based on this, we propose FlowCut, an information-flow-aware pruning framework, mitigating the insufficiency of the current criterion for identifying redundant tokens and better aligning with the model's inherent behaviors. Extensive experiments show that FlowCut achieves superior results, outperforming SoTA by 1.6% on LLaVA-1.5-7B with 88.9% token reduction, and by 4.3% on LLaVA-NeXT-7B with 94.4% reduction, delivering 3.2x speed-up in the prefilling stage. Our code is available at https://github.com/TungChintao/FlowCut
>
---
#### [new 133] ImgEdit: A Unified Image Editing Dataset and Benchmark
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决开源模型因数据不足和基准缺失落后的问题。工作包括构建含120万高质量编辑配对的ImgEdit数据集，开发基于视觉语言模型的ImgEdit-E1，及设计ImgEdit-Bench评估基准，全面评测模型性能。**

- **链接: [http://arxiv.org/pdf/2505.20275v1](http://arxiv.org/pdf/2505.20275v1)**

> **作者:** Yang Ye; Xianyi He; Zongjian Li; Bin Lin; Shenghai Yuan; Zhiyuan Yan; Bohan Hou; Li Yuan
>
> **摘要:** Recent advancements in generative models have enabled high-fidelity text-to-image generation. However, open-source image-editing models still lag behind their proprietary counterparts, primarily due to limited high-quality data and insufficient benchmarks. To overcome these limitations, we introduce ImgEdit, a large-scale, high-quality image-editing dataset comprising 1.2 million carefully curated edit pairs, which contain both novel and complex single-turn edits, as well as challenging multi-turn tasks. To ensure the data quality, we employ a multi-stage pipeline that integrates a cutting-edge vision-language model, a detection model, a segmentation model, alongside task-specific in-painting procedures and strict post-processing. ImgEdit surpasses existing datasets in both task novelty and data quality. Using ImgEdit, we train ImgEdit-E1, an editing model using Vision Language Model to process the reference image and editing prompt, which outperforms existing open-source models on multiple tasks, highlighting the value of ImgEdit and model design. For comprehensive evaluation, we introduce ImgEdit-Bench, a benchmark designed to evaluate image editing performance in terms of instruction adherence, editing quality, and detail preservation. It includes a basic testsuite, a challenging single-turn suite, and a dedicated multi-turn suite. We evaluate both open-source and proprietary models, as well as ImgEdit-E1, providing deep analysis and actionable insights into the current behavior of image-editing models. The source data are publicly available on https://github.com/PKU-YuanGroup/ImgEdit.
>
---
#### [new 134] Beyond Domain Randomization: Event-Inspired Perception for Visually Robust Adversarial Imitation from Videos
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视觉模仿学习任务，旨在解决专家演示与学习环境视觉差异（如光照、颜色）导致的模仿失败问题。提出事件启发的感知方法，将RGB视频转为基于时间强度变化的稀疏事件流，消除静态外观干扰，分离运动与视觉风格，实现跨领域鲁棒模仿，经实验验证有效。**

- **链接: [http://arxiv.org/pdf/2505.18899v1](http://arxiv.org/pdf/2505.18899v1)**

> **作者:** Andrea Ramazzina; Vittorio Giammarino; Matteo El-Hariry; Mario Bijelic
>
> **摘要:** Imitation from videos often fails when expert demonstrations and learner environments exhibit domain shifts, such as discrepancies in lighting, color, or texture. While visual randomization partially addresses this problem by augmenting training data, it remains computationally intensive and inherently reactive, struggling with unseen scenarios. We propose a different approach: instead of randomizing appearances, we eliminate their influence entirely by rethinking the sensory representation itself. Inspired by biological vision systems that prioritize temporal transients (e.g., retinal ganglion cells) and by recent sensor advancements, we introduce event-inspired perception for visually robust imitation. Our method converts standard RGB videos into a sparse, event-based representation that encodes temporal intensity gradients, discarding static appearance features. This biologically grounded approach disentangles motion dynamics from visual style, enabling robust visual imitation from observations even in the presence of visual mismatches between expert and agent environments. By training policies on event streams, we achieve invariance to appearance-based distractors without requiring computationally expensive and environment-specific data augmentation techniques. Experiments across the DeepMind Control Suite and the Adroit platform for dynamic dexterous manipulation show the efficacy of our method. Our code is publicly available at Eb-LAIfO.
>
---
#### [new 135] A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking
- **分类: cs.CV**

- **简介: 该论文属于视频融合任务，针对现有方法忽视视频时序关联导致闪烁和不连贯的问题，提出UniVF框架（结合多帧学习与光流特征扭曲）及首个综合基准VF-Bench（含四类视频融合任务），实现高质量时序一致的视频融合。**

- **链接: [http://arxiv.org/pdf/2505.19858v1](http://arxiv.org/pdf/2505.19858v1)**

> **作者:** Zixiang Zhao; Haowen Bai; Bingxin Ke; Yukun Cui; Lilun Deng; Yulun Zhang; Kai Zhang; Konrad Schindler
>
> **摘要:** The real world is dynamic, yet most image fusion methods process static frames independently, ignoring temporal correlations in videos and leading to flickering and temporal inconsistency. To address this, we propose Unified Video Fusion (UniVF), a novel framework for temporally coherent video fusion that leverages multi-frame learning and optical flow-based feature warping for informative, temporally coherent video fusion. To support its development, we also introduce Video Fusion Benchmark (VF-Bench), the first comprehensive benchmark covering four video fusion tasks: multi-exposure, multi-focus, infrared-visible, and medical fusion. VF-Bench provides high-quality, well-aligned video pairs obtained through synthetic data generation and rigorous curation from existing datasets, with a unified evaluation protocol that jointly assesses the spatial quality and temporal consistency of video fusion. Extensive experiments show that UniVF achieves state-of-the-art results across all tasks on VF-Bench. Project page: https://vfbench.github.io.
>
---
#### [new 136] EmoNet-Face: An Expert-Annotated Benchmark for Synthetic Emotion Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于情感识别任务，针对现有数据集情感覆盖窄、图像质量差及多样性不足的问题，构建了包含40类情感分类的EmoNet-Face基准，含三个AI生成数据集、专家标注和高性能模型，提升AI对复杂人类情绪的理解能力。**

- **链接: [http://arxiv.org/pdf/2505.20033v1](http://arxiv.org/pdf/2505.20033v1)**

> **作者:** Christoph Schuhmann; Robert Kaczmarczyk; Gollam Rabby; Maurice Kraus; Felix Friedrich; Huu Nguyen; Krishna Kalyan; Kourosh Nadi; Kristian Kersting; Sören Auer
>
> **摘要:** Effective human-AI interaction relies on AI's ability to accurately perceive and interpret human emotions. Current benchmarks for vision and vision-language models are severely limited, offering a narrow emotional spectrum that overlooks nuanced states (e.g., bitterness, intoxication) and fails to distinguish subtle differences between related feelings (e.g., shame vs. embarrassment). Existing datasets also often use uncontrolled imagery with occluded faces and lack demographic diversity, risking significant bias. To address these critical gaps, we introduce EmoNet Face, a comprehensive benchmark suite. EmoNet Face features: (1) A novel 40-category emotion taxonomy, meticulously derived from foundational research to capture finer details of human emotional experiences. (2) Three large-scale, AI-generated datasets (EmoNet HQ, Binary, and Big) with explicit, full-face expressions and controlled demographic balance across ethnicity, age, and gender. (3) Rigorous, multi-expert annotations for training and high-fidelity evaluation. (4) We build Empathic Insight Face, a model achieving human-expert-level performance on our benchmark. The publicly released EmoNet Face suite - taxonomy, datasets, and model - provides a robust foundation for developing and evaluating AI systems with a deeper understanding of human emotions.
>
---
#### [new 137] Enhancing Text-to-Image Diffusion Transformer via Split-Text Conditioning
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像生成任务中扩散Transformer（DiT）因完整文本输入导致的语义理解缺陷，提出DiT-ST框架。其通过拆分文本为简化句子，利用大语言模型分层次注入不同去噪阶段，提升语义表示学习，解决细节忽略与混淆问题。**

- **链接: [http://arxiv.org/pdf/2505.19261v1](http://arxiv.org/pdf/2505.19261v1)**

> **作者:** Yu Zhang; Jialei Zhou; Xinchen Li; Qi Zhang; Zhongwei Wan; Tianyu Wang; Duoqian Miao; Changwei Wang; Longbing Cao
>
> **备注:** 21 pages
>
> **摘要:** Current text-to-image diffusion generation typically employs complete-text conditioning. Due to the intricate syntax, diffusion transformers (DiTs) inherently suffer from a comprehension defect of complete-text captions. One-fly complete-text input either overlooks critical semantic details or causes semantic confusion by simultaneously modeling diverse semantic primitive types. To mitigate this defect of DiTs, we propose a novel split-text conditioning framework named DiT-ST. This framework converts a complete-text caption into a split-text caption, a collection of simplified sentences, to explicitly express various semantic primitives and their interconnections. The split-text caption is then injected into different denoising stages of DiT-ST in a hierarchical and incremental manner. Specifically, DiT-ST leverages Large Language Models to parse captions, extracting diverse primitives and hierarchically sorting out and constructing these primitives into a split-text input. Moreover, we partition the diffusion denoising process according to its differential sensitivities to diverse semantic primitive types and determine the appropriate timesteps to incrementally inject tokens of diverse semantic primitive types into input tokens via cross-attention. In this way, DiT-ST enhances the representation learning of specific semantic primitive types across different stages. Extensive experiments validate the effectiveness of our proposed DiT-ST in mitigating the complete-text comprehension defect.
>
---
#### [new 138] Visualized Text-to-Image Retrieval
- **分类: cs.CV; cs.CL**

- **简介: 论文提出VisRet方法，针对文本到图像（T2I）检索任务，解决现有跨模态嵌入对齐不足的问题。通过先将文本生成图像，再在图像模态内检索，规避跨模态检索器对视觉-空间特征识别的局限。实验显示NDCG@10提升24.5%-32.7%，且兼容现有检索器，增强知识密集型多模态系统效果。**

- **链接: [http://arxiv.org/pdf/2505.20291v1](http://arxiv.org/pdf/2505.20291v1)**

> **作者:** Di Wu; Yixin Wan; Kai-Wei Chang
>
> **备注:** Work in Progress
>
> **摘要:** We propose Visualize-then-Retrieve (VisRet), a new paradigm for Text-to-Image (T2I) retrieval that mitigates the limitations of cross-modal similarity alignment of existing multi-modal embeddings. VisRet first projects textual queries into the image modality via T2I generation. Then, it performs retrieval within the image modality to bypass the weaknesses of cross-modal retrievers in recognizing subtle visual-spatial features. Experiments on three knowledge-intensive T2I retrieval benchmarks, including a newly introduced multi-entity benchmark, demonstrate that VisRet consistently improves T2I retrieval by 24.5% to 32.7% NDCG@10 across different embedding models. VisRet also significantly benefits downstream visual question answering accuracy when used in retrieval-augmented generation pipelines. The method is plug-and-play and compatible with off-the-shelf retrievers, making it an effective module for knowledge-intensive multi-modal systems. Our code and the new benchmark are publicly available at https://github.com/xiaowu0162/Visualize-then-Retrieve.
>
---
#### [new 139] In-Context Brush: Zero-shot Customized Subject Insertion with Context-Aware Latent Space Manipulation
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文提出零样本定制主体插入框架"In-Context Brush"，解决现有方法在图像生成中插入指定对象时保真度低、与文本提示对齐差的问题。通过跨模态示例与双层潜在空间操作（特征偏移+注意力重加权），在预训练模型基础上动态调整注意力机制，实现高质量无监督图像 inpainting。**

- **链接: [http://arxiv.org/pdf/2505.20271v1](http://arxiv.org/pdf/2505.20271v1)**

> **作者:** Yu Xu; Fan Tang; You Wu; Lin Gao; Oliver Deussen; Hongbin Yan; Jintao Li; Juan Cao; Tong-Yee Lee
>
> **摘要:** Recent advances in diffusion models have enhanced multimodal-guided visual generation, enabling customized subject insertion that seamlessly "brushes" user-specified objects into a given image guided by textual prompts. However, existing methods often struggle to insert customized subjects with high fidelity and align results with the user's intent through textual prompts. In this work, we propose "In-Context Brush", a zero-shot framework for customized subject insertion by reformulating the task within the paradigm of in-context learning. Without loss of generality, we formulate the object image and the textual prompts as cross-modal demonstrations, and the target image with the masked region as the query. The goal is to inpaint the target image with the subject aligning textual prompts without model tuning. Building upon a pretrained MMDiT-based inpainting network, we perform test-time enhancement via dual-level latent space manipulation: intra-head "latent feature shifting" within each attention head that dynamically shifts attention outputs to reflect the desired subject semantics and inter-head "attention reweighting" across different heads that amplifies prompt controllability through differential attention prioritization. Extensive experiments and applications demonstrate that our approach achieves superior identity preservation, text alignment, and image quality compared to existing state-of-the-art methods, without requiring dedicated training or additional data collection.
>
---
#### [new 140] ViewCraft3D: High-Fidelity and View-Consistent 3D Vector Graphics Synthesis
- **分类: cs.CV**

- **简介: 该论文属于3D矢量图形合成任务，旨在解决现有方法处理速度慢和视角不一致的问题。提出ViewCraft3D方法，通过3D结构分析、几何提取算法及视角一致优化流程，高效生成高质量且视角稳定的3D矢量图形，实验显示其性能与效率优于先前方法。**

- **链接: [http://arxiv.org/pdf/2505.19492v1](http://arxiv.org/pdf/2505.19492v1)**

> **作者:** Chuang Wang; Haitao Zhou; Ling Luo; Qian Yu
>
> **摘要:** 3D vector graphics play a crucial role in various applications including 3D shape retrieval, conceptual design, and virtual reality interactions due to their ability to capture essential structural information with minimal representation. While recent approaches have shown promise in generating 3D vector graphics, they often suffer from lengthy processing times and struggle to maintain view consistency. To address these limitations, we propose ViewCraft3D (VC3D), an efficient method that leverages 3D priors to generate 3D vector graphics. Specifically, our approach begins with 3D object analysis, employs a geometric extraction algorithm to fit 3D vector graphics to the underlying structure, and applies view-consistent refinement process to enhance visual quality. Our comprehensive experiments demonstrate that VC3D outperforms previous methods in both qualitative and quantitative evaluations, while significantly reducing computational overhead. The resulting 3D sketches maintain view consistency and effectively capture the essential characteristics of the original objects.
>
---
#### [new 141] COLORA: Efficient Fine-Tuning for Convolutional Models with a Study Case on Optical Coherence Tomography Image Classification
- **分类: cs.CV; cs.AI; 68T07; I.1.2; I.4.0; I.4.10; I.4.0**

- **简介: 该论文提出CoLoRA方法，解决CNN微调参数多、效率低的问题。通过扩展LoRA到卷积架构，减少参数，提升训练速度与稳定性。在OCTMNIST视网膜疾病分类任务中，微调后的CNN准确率超传统方法近1%，性能媲美Vision Transformer等模型。（98字）**

- **链接: [http://arxiv.org/pdf/2505.18315v1](http://arxiv.org/pdf/2505.18315v1)**

> **作者:** Mariano Rivera; Angello Hoyos
>
> **备注:** 15 pages, 12 figures. Submitted to Jou. Pattern Recognition
>
> **摘要:** We introduce the Convolutional Low-Rank Adaptation (CoLoRA) method, designed explicitly to overcome the inefficiencies found in current CNN fine-tuning methods. CoLoRA can be seen as a natural extension of the convolutional architectures of the Low-Rank Adaptation (LoRA) technique. We demonstrate the capabilities of our method by developing and evaluating models using the widely adopted CNN backbone pre-trained on ImageNet. We observed that this strategy results in a stable and accurate coarse-tuning procedure. Moreover, this strategy is computationally efficient and significantly reduces the number of parameters required for fine-tuning compared to traditional methods. Furthermore, our method substantially improves the speed and stability of training. Our case study focuses on classifying retinal diseases from optical coherence tomography (OCT) images, specifically using the OCTMNIST dataset. Experimental results demonstrate that a CNN backbone fine-tuned with CoLoRA surpasses nearly 1\% in accuracy. Such a performance is comparable to the Vision Transformer, State-space discrete, and Kolmogorov-Arnold network models.
>
---
#### [new 142] InfoChartQA: A Benchmark for Multimodal Question Answering on Infographic Charts
- **分类: cs.CV; cs.AI**

- **简介: 论文提出InfoChartQA基准，评估多模态模型对信息图表（含象形图、图标等设计元素）的问答能力。针对现有数据集缺乏配对图表及视觉元素问题，构建5,642对图文数据，设计基于视觉设计的问答任务，揭示模型在隐喻类问题上的不足，推动模型对信息图表的理解能力提升。**

- **链接: [http://arxiv.org/pdf/2505.19028v1](http://arxiv.org/pdf/2505.19028v1)**

> **作者:** Minzhi Lin; Tianchi Xie; Mengchen Liu; Yilin Ye; Changjian Chen; Shixia Liu
>
> **摘要:** Understanding infographic charts with design-driven visual elements (e.g., pictograms, icons) requires both visual recognition and reasoning, posing challenges for multimodal large language models (MLLMs). However, existing visual-question answering benchmarks fall short in evaluating these capabilities of MLLMs due to the lack of paired plain charts and visual-element-based questions. To bridge this gap, we introduce InfoChartQA, a benchmark for evaluating MLLMs on infographic chart understanding. It includes 5,642 pairs of infographic and plain charts, each sharing the same underlying data but differing in visual presentations. We further design visual-element-based questions to capture their unique visual designs and communicative intent. Evaluation of 20 MLLMs reveals a substantial performance decline on infographic charts, particularly for visual-element-based questions related to metaphors. The paired infographic and plain charts enable fine-grained error analysis and ablation studies, which highlight new opportunities for advancing MLLMs in infographic chart understanding. We release InfoChartQA at https://github.com/CoolDawnAnt/InfoChartQA.
>
---
#### [new 143] ViTaPEs: Visuotactile Position Encodings for Cross-Modal Alignment in Multimodal Transformers
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出ViTaPEs框架，解决视觉-触觉多模态融合及跨任务泛化问题。通过设计多尺度位置编码 scheme 捕捉模内结构与跨模态关联，确保编码的注入性与等变性，并验证其在识别任务、零样本迁移及机器人抓取中的优越性能。**

- **链接: [http://arxiv.org/pdf/2505.20032v1](http://arxiv.org/pdf/2505.20032v1)**

> **作者:** Fotios Lygerakis; Ozan Özdenizci; Elmar Rückert
>
> **摘要:** Tactile sensing provides local essential information that is complementary to visual perception, such as texture, compliance, and force. Despite recent advances in visuotactile representation learning, challenges remain in fusing these modalities and generalizing across tasks and environments without heavy reliance on pre-trained vision-language models. Moreover, existing methods do not study positional encodings, thereby overlooking the multi-scale spatial reasoning needed to capture fine-grained visuotactile correlations. We introduce ViTaPEs, a transformer-based framework that robustly integrates visual and tactile input data to learn task-agnostic representations for visuotactile perception. Our approach exploits a novel multi-scale positional encoding scheme to capture intra-modal structures, while simultaneously modeling cross-modal cues. Unlike prior work, we provide provable guarantees in visuotactile fusion, showing that our encodings are injective, rigid-motion-equivariant, and information-preserving, validating these properties empirically. Experiments on multiple large-scale real-world datasets show that ViTaPEs not only surpasses state-of-the-art baselines across various recognition tasks but also demonstrates zero-shot generalization to unseen, out-of-domain scenarios. We further demonstrate the transfer-learning strength of ViTaPEs in a robotic grasping task, where it outperforms state-of-the-art baselines in predicting grasp success. Project page: https://sites.google.com/view/vitapes
>
---
#### [new 144] SMART-PC: Skeletal Model Adaptation for Robust Test-Time Training in Point Clouds
- **分类: cs.CV**

- **简介: SMART-PC属于3D点云分类任务，解决测试时分布偏移及现有方法计算效率低的问题。提出骨骼模型框架，通过预训练提取几何骨骼特征提升鲁棒性，并仅更新BatchNorm统计量替代反向传播，实现高效实时适应，在多个数据集上达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.19546v1](http://arxiv.org/pdf/2505.19546v1)**

> **作者:** Ali Bahri; Moslem Yazdanpanah; Sahar Dastani; Mehrdad Noori; Gustavo Adolfo Vargas Hakim; David Osowiechi; Farzad Beizaee; Ismail Ben Ayed; Christian Desrosiers
>
> **摘要:** Test-Time Training (TTT) has emerged as a promising solution to address distribution shifts in 3D point cloud classification. However, existing methods often rely on computationally expensive backpropagation during adaptation, limiting their applicability in real-world, time-sensitive scenarios. In this paper, we introduce SMART-PC, a skeleton-based framework that enhances resilience to corruptions by leveraging the geometric structure of 3D point clouds. During pre-training, our method predicts skeletal representations, enabling the model to extract robust and meaningful geometric features that are less sensitive to corruptions, thereby improving adaptability to test-time distribution shifts. Unlike prior approaches, SMART-PC achieves real-time adaptation by eliminating backpropagation and updating only BatchNorm statistics, resulting in a lightweight and efficient framework capable of achieving high frame-per-second rates while maintaining superior classification performance. Extensive experiments on benchmark datasets, including ModelNet40-C, ShapeNet-C, and ScanObjectNN-C, demonstrate that SMART-PC achieves state-of-the-art results, outperforming existing methods such as MATE in terms of both accuracy and computational efficiency. The implementation is available at: https://github.com/AliBahri94/SMART-PC.
>
---
#### [new 145] OmniConsistency: Learning Style-Agnostic Consistency from Paired Stylization Data
- **分类: cs.CV**

- **简介: 该论文提出OmniConsistency，用于图像风格化任务，解决复杂场景下保持一致性及风格LoRAs退化问题。通过配对数据训练的扩散Transformer框架，采用两阶段学习和插件设计，提升视觉质量，性能接近GPT-4o。**

- **链接: [http://arxiv.org/pdf/2505.18445v1](http://arxiv.org/pdf/2505.18445v1)**

> **作者:** Yiren Song; Cheng Liu; Mike Zheng Shou
>
> **摘要:** Diffusion models have advanced image stylization significantly, yet two core challenges persist: (1) maintaining consistent stylization in complex scenes, particularly identity, composition, and fine details, and (2) preventing style degradation in image-to-image pipelines with style LoRAs. GPT-4o's exceptional stylization consistency highlights the performance gap between open-source methods and proprietary models. To bridge this gap, we propose \textbf{OmniConsistency}, a universal consistency plugin leveraging large-scale Diffusion Transformers (DiTs). OmniConsistency contributes: (1) an in-context consistency learning framework trained on aligned image pairs for robust generalization; (2) a two-stage progressive learning strategy decoupling style learning from consistency preservation to mitigate style degradation; and (3) a fully plug-and-play design compatible with arbitrary style LoRAs under the Flux framework. Extensive experiments show that OmniConsistency significantly enhances visual coherence and aesthetic quality, achieving performance comparable to commercial state-of-the-art model GPT-4o.
>
---
#### [new 146] Multimodal LLM-Guided Semantic Correction in Text-to-Image Diffusion
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于文本到图像生成任务，旨在解决扩散模型生成过程中存在的语义不一致问题（如物体混淆、空间错误等）。提出PPAD框架，首次在推理阶段引入多模态大语言模型实时分析中间结果，通过语义反馈主动修正生成轨迹，支持训练增强与轻量化部署，提升生成质量和提示对齐。**

- **链接: [http://arxiv.org/pdf/2505.20053v1](http://arxiv.org/pdf/2505.20053v1)**

> **作者:** Zheqi Lv; Junhao Chen; Qi Tian; Keting Yin; Shengyu Zhang; Fei Wu
>
> **摘要:** Diffusion models have become the mainstream architecture for text-to-image generation, achieving remarkable progress in visual quality and prompt controllability. However, current inference pipelines generally lack interpretable semantic supervision and correction mechanisms throughout the denoising process. Most existing approaches rely solely on post-hoc scoring of the final image, prompt filtering, or heuristic resampling strategies-making them ineffective in providing actionable guidance for correcting the generative trajectory. As a result, models often suffer from object confusion, spatial errors, inaccurate counts, and missing semantic elements, severely compromising prompt-image alignment and image quality. To tackle these challenges, we propose MLLM Semantic-Corrected Ping-Pong-Ahead Diffusion (PPAD), a novel framework that, for the first time, introduces a Multimodal Large Language Model (MLLM) as a semantic observer during inference. PPAD performs real-time analysis on intermediate generations, identifies latent semantic inconsistencies, and translates feedback into controllable signals that actively guide the remaining denoising steps. The framework supports both inference-only and training-enhanced settings, and performs semantic correction at only extremely few diffusion steps, offering strong generality and scalability. Extensive experiments demonstrate PPAD's significant improvements.
>
---
#### [new 147] FruitNeRF++: A Generalized Multi-Fruit Counting Method Utilizing Contrastive Learning and Neural Radiance Fields
- **分类: cs.CV; cs.LG**

- **简介: 论文提出FruitNeRF++，解决原有FruitNeRF需针对每种水果调整的局限。任务为果园图像多水果计数，通过结合对比学习与神经辐射场，利用视觉模型预测实例掩码生成实例嵌入，实现通用点云聚类计数。在合成及真实数据中表现优异。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19863v1](http://arxiv.org/pdf/2505.19863v1)**

> **作者:** Lukas Meyer; Andrei-Timotei Ardelean; Tim Weyrich; Marc Stamminger
>
> **备注:** for project website, see https://meyerls.github.io/fruit_nerfpp
>
> **摘要:** We introduce FruitNeRF++, a novel fruit-counting approach that combines contrastive learning with neural radiance fields to count fruits from unstructured input photographs of orchards. Our work is based on FruitNeRF, which employs a neural semantic field combined with a fruit-specific clustering approach. The requirement for adaptation for each fruit type limits the applicability of the method, and makes it difficult to use in practice. To lift this limitation, we design a shape-agnostic multi-fruit counting framework, that complements the RGB and semantic data with instance masks predicted by a vision foundation model. The masks are used to encode the identity of each fruit as instance embeddings into a neural instance field. By volumetrically sampling the neural fields, we extract a point cloud embedded with the instance features, which can be clustered in a fruit-agnostic manner to obtain the fruit count. We evaluate our approach using a synthetic dataset containing apples, plums, lemons, pears, peaches, and mangoes, as well as a real-world benchmark apple dataset. Our results demonstrate that FruitNeRF++ is easier to control and compares favorably to other state-of-the-art methods.
>
---
#### [new 148] Jodi: Unification of Visual Generation and Understanding via Joint Modeling
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Jodi框架，通过联合建模统一视觉生成与理解任务，解决传统方法分离处理导致的效率与性能不足。其构建扩散模型与角色切换机制，实现联合生成、可控生成及图像感知，并发布Joint-1.6M数据集，实验验证其跨领域有效性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19084v1](http://arxiv.org/pdf/2505.19084v1)**

> **作者:** Yifeng Xu; Zhenliang He; Meina Kan; Shiguang Shan; Xilin Chen
>
> **备注:** Code: https://github.com/VIPL-GENUN/Jodi
>
> **摘要:** Visual generation and understanding are two deeply interconnected aspects of human intelligence, yet they have been traditionally treated as separate tasks in machine learning. In this paper, we propose Jodi, a diffusion framework that unifies visual generation and understanding by jointly modeling the image domain and multiple label domains. Specifically, Jodi is built upon a linear diffusion transformer along with a role switch mechanism, which enables it to perform three particular types of tasks: (1) joint generation, where the model simultaneously generates images and multiple labels; (2) controllable generation, where images are generated conditioned on any combination of labels; and (3) image perception, where multiple labels can be predicted at once from a given image. Furthermore, we present the Joint-1.6M dataset, which contains 200,000 high-quality images collected from public sources, automatic labels for 7 visual domains, and LLM-generated captions. Extensive experiments demonstrate that Jodi excels in both generation and understanding tasks and exhibits strong extensibility to a wider range of visual domains. Code is available at https://github.com/VIPL-GENUN/Jodi.
>
---
#### [new 149] Rethinking Causal Mask Attention for Vision-Language Inference
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言推理任务，提出未来感知注意力机制，解决现有因果掩码对视觉token未来上下文过度屏蔽的问题。通过池化整合未来视觉信息到当前表示，增强跨模态依赖，在多个任务中提升推理性能。**

- **链接: [http://arxiv.org/pdf/2505.18605v1](http://arxiv.org/pdf/2505.18605v1)**

> **作者:** Xiaohuan Pei; Tao Huang; YanXiang Ma; Chang Xu
>
> **摘要:** Causal attention has become a foundational mechanism in autoregressive vision-language models (VLMs), unifying textual and visual inputs under a single generative framework. However, existing causal mask-based strategies are inherited from large language models (LLMs) where they are tailored for text-only decoding, and their adaptation to vision tokens is insufficiently addressed in the prefill stage. Strictly masking future positions for vision queries introduces overly rigid constraints, which hinder the model's ability to leverage future context that often contains essential semantic cues for accurate inference. In this work, we empirically investigate how different causal masking strategies affect vision-language inference and then propose a family of future-aware attentions tailored for this setting. We first empirically analyze the effect of previewing future tokens for vision queries and demonstrate that rigid masking undermines the model's capacity to capture useful contextual semantic representations. Based on these findings, we propose a lightweight attention family that aggregates future visual context into past representations via pooling, effectively preserving the autoregressive structure while enhancing cross-token dependencies. We evaluate a range of causal masks across diverse vision-language inference settings and show that selectively compressing future semantic context into past representations benefits the inference.
>
---
#### [new 150] M3DHMR: Monocular 3D Hand Mesh Recovery
- **分类: cs.CV**

- **简介: 该论文属于单目3D手部网格恢复任务，解决手部高自由度、2D-3D歧义及自遮挡导致的3D顶点预测困难问题。提出M3DHMR方法，通过螺旋解码器（含动态螺旋卷积层和ROI层）直接预测网格顶点，利用自适应权重与区域分割优化，在FreiHAND数据集上超越实时方法。**

- **链接: [http://arxiv.org/pdf/2505.20058v1](http://arxiv.org/pdf/2505.20058v1)**

> **作者:** Yihong Lin; Xianjia Wu; Xilai Wang; Jianqiao Hu; Songju Lei; Xiandong Li; Wenxiong Kang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Monocular 3D hand mesh recovery is challenging due to high degrees of freedom of hands, 2D-to-3D ambiguity and self-occlusion. Most existing methods are either inefficient or less straightforward for predicting the position of 3D mesh vertices. Thus, we propose a new pipeline called Monocular 3D Hand Mesh Recovery (M3DHMR) to directly estimate the positions of hand mesh vertices. M3DHMR provides 2D cues for 3D tasks from a single image and uses a new spiral decoder consist of several Dynamic Spiral Convolution (DSC) Layers and a Region of Interest (ROI) Layer. On the one hand, DSC Layers adaptively adjust the weights based on the vertex positions and extract the vertex features in both spatial and channel dimensions. On the other hand, ROI Layer utilizes the physical information and refines mesh vertices in each predefined hand region separately. Extensive experiments on popular dataset FreiHAND demonstrate that M3DHMR significantly outperforms state-of-the-art real-time methods.
>
---
#### [new 151] Think Twice before Adaptation: Improving Adaptability of DeepFake Detection via Online Test-Time Adaptation
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于Deepfake检测任务，旨在解决后处理操作和分布偏移导致检测器性能下降的问题。提出T²A方法，通过不确定性感知负学习目标、不确定样本优先级策略及梯度掩码技术，在线提升模型适应性，无需额外数据或标签，实现SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.18787v1](http://arxiv.org/pdf/2505.18787v1)**

> **作者:** Hong-Hanh Nguyen-Le; Van-Tuan Tran; Dinh-Thuc Nguyen; Nhien-An Le-Khac
>
> **备注:** Accepted at 34th International Joint Conference on Artificial Intelligence (IJCAI-25)
>
> **摘要:** Deepfake (DF) detectors face significant challenges when deployed in real-world environments, particularly when encountering test samples deviated from training data through either postprocessing manipulations or distribution shifts. We demonstrate postprocessing techniques can completely obscure generation artifacts presented in DF samples, leading to performance degradation of DF detectors. To address these challenges, we propose Think Twice before Adaptation (\texttt{T$^2$A}), a novel online test-time adaptation method that enhances the adaptability of detectors during inference without requiring access to source training data or labels. Our key idea is to enable the model to explore alternative options through an Uncertainty-aware Negative Learning objective rather than solely relying on its initial predictions as commonly seen in entropy minimization (EM)-based approaches. We also introduce an Uncertain Sample Prioritization strategy and Gradients Masking technique to improve the adaptation by focusing on important samples and model parameters. Our theoretical analysis demonstrates that the proposed negative learning objective exhibits complementary behavior to EM, facilitating better adaptation capability. Empirically, our method achieves state-of-the-art results compared to existing test-time adaptation (TTA) approaches and significantly enhances the resilience and generalization of DF detectors during inference. Code is available \href{https://github.com/HongHanh2104/T2A-Think-Twice-Before-Adaptation}{here}.
>
---
#### [new 152] VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文研究多模态大模型安全，解决其视觉推理能力引发的越狱风险。提出VisCRA框架，通过视觉注意力遮蔽和两阶段推理策略，利用模型自身推理链漏洞突破安全机制，实验显示对主流模型攻击成功率超56%-76%，揭示视觉能力的双刃剑效应。**

- **链接: [http://arxiv.org/pdf/2505.19684v1](http://arxiv.org/pdf/2505.19684v1)**

> **作者:** Bingrui Sima; Linhua Cong; Wenxuan Wang; Kun He
>
> **摘要:** The emergence of Multimodal Large Language Models (MLRMs) has enabled sophisticated visual reasoning capabilities by integrating reinforcement learning and Chain-of-Thought (CoT) supervision. However, while these enhanced reasoning capabilities improve performance, they also introduce new and underexplored safety risks. In this work, we systematically investigate the security implications of advanced visual reasoning in MLRMs. Our analysis reveals a fundamental trade-off: as visual reasoning improves, models become more vulnerable to jailbreak attacks. Motivated by this critical finding, we introduce VisCRA (Visual Chain Reasoning Attack), a novel jailbreak framework that exploits the visual reasoning chains to bypass safety mechanisms. VisCRA combines targeted visual attention masking with a two-stage reasoning induction strategy to precisely control harmful outputs. Extensive experiments demonstrate VisCRA's significant effectiveness, achieving high attack success rates on leading closed-source MLRMs: 76.48% on Gemini 2.0 Flash Thinking, 68.56% on QvQ-Max, and 56.60% on GPT-4o. Our findings highlight a critical insight: the very capability that empowers MLRMs -- their visual reasoning -- can also serve as an attack vector, posing significant security risks.
>
---
#### [new 153] Rotation-Equivariant Self-Supervised Method in Image Denoising
- **分类: cs.CV**

- **简介: 该论文属于图像去噪任务，旨在通过引入旋转等变先验提升自监督方法性能。针对传统CNN仅利用平移等变性的局限，提出旋转等变卷积网络，并设计融合机制结合传统CNN，构建自适应框架，经实验验证有效。**

- **链接: [http://arxiv.org/pdf/2505.19618v1](http://arxiv.org/pdf/2505.19618v1)**

> **作者:** Hanze Liu; Jiahong Fu; Qi Xie; Deyu Meng
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Self-supervised image denoising methods have garnered significant research attention in recent years, for this kind of method reduces the requirement of large training datasets. Compared to supervised methods, self-supervised methods rely more on the prior embedded in deep networks themselves. As a result, most of the self-supervised methods are designed with Convolution Neural Networks (CNNs) architectures, which well capture one of the most important image prior, translation equivariant prior. Inspired by the great success achieved by the introduction of translational equivariance, in this paper, we explore the way to further incorporate another important image prior. Specifically, we first apply high-accuracy rotation equivariant convolution to self-supervised image denoising. Through rigorous theoretical analysis, we have proved that simply replacing all the convolution layers with rotation equivariant convolution layers would modify the network into its rotation equivariant version. To the best of our knowledge, this is the first time that rotation equivariant image prior is introduced to self-supervised image denoising at the network architecture level with a comprehensive theoretical analysis of equivariance errors, which offers a new perspective to the field of self-supervised image denoising. Moreover, to further improve the performance, we design a new mask mechanism to fusion the output of rotation equivariant network and vanilla CNN-based network, and construct an adaptive rotation equivariant framework. Through extensive experiments on three typical methods, we have demonstrated the effectiveness of the proposed method.
>
---
#### [new 154] LlamaSeg: Image Segmentation via Autoregressive Mask Generation
- **分类: cs.CV**

- **简介: 该论文提出LlamaSeg，一种基于自回归掩码生成的视觉框架，统一多图像分割任务。通过将分割重新定义为视觉生成问题，使用LLaMA式Transformer预测掩码，并构建包含2M标注的SA-OVRS数据集支持训练。同时提出结合IoU和AHD的评估指标，提升分割精度与细节。实验显示优于现有生成模型。**

- **链接: [http://arxiv.org/pdf/2505.19422v1](http://arxiv.org/pdf/2505.19422v1)**

> **作者:** Jiru Deng; Tengjin Weng; Tianyu Yang; Wenhan Luo; Zhiheng Li; Wenhao Jiang
>
> **摘要:** We present LlamaSeg, a visual autoregressive framework that unifies multiple image segmentation tasks via natural language instructions. We reformulate image segmentation as a visual generation problem, representing masks as "visual" tokens and employing a LLaMA-style Transformer to predict them directly from image inputs. By adhering to the next-token prediction paradigm, our approach naturally integrates segmentation tasks into autoregressive architectures. To support large-scale training, we introduce a data annotation pipeline and construct the SA-OVRS dataset, which contains 2M segmentation masks annotated with over 5,800 open-vocabulary labels or diverse textual descriptions, covering a wide spectrum of real-world scenarios. This enables our model to localize objects in images based on text prompts and to generate fine-grained masks. To more accurately evaluate the quality of masks produced by visual generative models, we further propose a composite metric that combines Intersection over Union (IoU) with Average Hausdorff Distance (AHD), offering a more precise assessment of contour fidelity. Experimental results demonstrate that our method surpasses existing generative models across multiple datasets and yields more detailed segmentation masks.
>
---
#### [new 155] Rethinking Direct Preference Optimization in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文针对文本到图像扩散模型与人类偏好对齐任务，解决现有方法探索不足及时间步奖励不平衡问题。提出稳定参考模型更新策略与时间步感知训练方法，提升偏好优化效果。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18736v1](http://arxiv.org/pdf/2505.18736v1)**

> **作者:** Junyong Kang; Seohyun Lim; Kyungjune Baek; Hyunjung Shim
>
> **备注:** 21 pages, 12 figures, preprint
>
> **摘要:** Aligning text-to-image (T2I) diffusion models with human preferences has emerged as a critical research challenge. While recent advances in this area have extended preference optimization techniques from large language models (LLMs) to the diffusion setting, they often struggle with limited exploration. In this work, we propose a novel and orthogonal approach to enhancing diffusion-based preference optimization. First, we introduce a stable reference model update strategy that relaxes the frozen reference model, encouraging exploration while maintaining a stable optimization anchor through reference model regularization. Second, we present a timestep-aware training strategy that mitigates the reward scale imbalance problem across timesteps. Our method can be integrated into various preference optimization algorithms. Experimental results show that our approach improves the performance of state-of-the-art methods on human preference evaluation benchmarks.
>
---
#### [new 156] PAMD: Plausibility-Aware Motion Diffusion Model for Long Dance Generation
- **分类: cs.CV**

- **简介: 该论文属于长舞蹈生成任务，旨在解决现有扩散模型生成舞蹈动作物理不真实的问题。提出PAMD框架，通过Plausible Motion Constraint（神经距离场建模真实姿势流形）、Prior Motion Guidance（结合站立姿势与音乐特征）和Motion Refinement with Foot-ground Contact（修正脚滑 artifacts）三模块，提升舞蹈的物理合理性与音乐契合度。**

- **链接: [http://arxiv.org/pdf/2505.20056v1](http://arxiv.org/pdf/2505.20056v1)**

> **作者:** Hongsong Wang; Yin Zhu; Qiuxia Lai; Yang Zhang; Guo-Sen Xie; Xin Geng
>
> **备注:** This project page is available at: https://mucunzhuzhu.github.io/PAMD-page/
>
> **摘要:** Computational dance generation is crucial in many areas, such as art, human-computer interaction, virtual reality, and digital entertainment, particularly for generating coherent and expressive long dance sequences. Diffusion-based music-to-dance generation has made significant progress, yet existing methods still struggle to produce physically plausible motions. To address this, we propose Plausibility-Aware Motion Diffusion (PAMD), a framework for generating dances that are both musically aligned and physically realistic. The core of PAMD lies in the Plausible Motion Constraint (PMC), which leverages Neural Distance Fields (NDFs) to model the actual pose manifold and guide generated motions toward a physically valid pose manifold. To provide more effective guidance during generation, we incorporate Prior Motion Guidance (PMG), which uses standing poses as auxiliary conditions alongside music features. To further enhance realism for complex movements, we introduce the Motion Refinement with Foot-ground Contact (MRFC) module, which addresses foot-skating artifacts by bridging the gap between the optimization objective in linear joint position space and the data representation in nonlinear rotation space. Extensive experiments show that PAMD significantly improves musical alignment and enhances the physical plausibility of generated motions. This project page is available at: https://mucunzhuzhu.github.io/PAMD-page/.
>
---
#### [new 157] WeakMCN: Multi-task Collaborative Network for Weakly Supervised Referring Expression Comprehension and Segmentation
- **分类: cs.CV**

- **简介: 该论文提出WeakMCN，一种多任务协作网络，用于弱监督指代表达理解和分割（WREC/WRES）。针对传统单独建模导致信息利用不足的问题，通过双分支架构整合任务，设计动态视觉增强（DVFE）和一致性模块（CCM），实现跨任务协同，实验显示显著提升性能。**

- **链接: [http://arxiv.org/pdf/2505.18686v1](http://arxiv.org/pdf/2505.18686v1)**

> **作者:** Yang Liu; Silin Cheng; Xinwei He; Sebastien Ourselin; Lei Tan; Gen Luo
>
> **备注:** Accepted by CVPR2025
>
> **摘要:** Weakly supervised referring expression comprehension(WREC) and segmentation(WRES) aim to learn object grounding based on a given expression using weak supervision signals like image-text pairs. While these tasks have traditionally been modeled separately, we argue that they can benefit from joint learning in a multi-task framework. To this end, we propose WeakMCN, a novel multi-task collaborative network that effectively combines WREC and WRES with a dual-branch architecture. Specifically, the WREC branch is formulated as anchor-based contrastive learning, which also acts as a teacher to supervise the WRES branch. In WeakMCN, we propose two innovative designs to facilitate multi-task collaboration, namely Dynamic Visual Feature Enhancement(DVFE) and Collaborative Consistency Module(CCM). DVFE dynamically combines various pre-trained visual knowledge to meet different task requirements, while CCM promotes cross-task consistency from the perspective of optimization. Extensive experimental results on three popular REC and RES benchmarks, i.e., RefCOCO, RefCOCO+, and RefCOCOg, consistently demonstrate performance gains of WeakMCN over state-of-the-art single-task alternatives, e.g., up to 3.91% and 13.11% on RefCOCO for WREC and WRES tasks, respectively. Furthermore, experiments also validate the strong generalization ability of WeakMCN in both semi-supervised REC and RES settings against existing methods, e.g., +8.94% for semi-REC and +7.71% for semi-RES on 1% RefCOCO. The code is publicly available at https://github.com/MRUIL/WeakMCN.
>
---
#### [new 158] SAIL: Self-supervised Albedo Estimation from Real Images with a Latent Diffusion Model
- **分类: cs.CV**

- **简介: 该论文提出SAIL方法，解决真实图像中基于自监督的反照率估计问题。针对现有方法依赖合成数据或光照一致性差的缺陷，利用潜扩散模型在隐空间分解图像，通过正则化约束光照相关/无关组件，实现稳定反照率预测，适用于多场景且仅需无标注多光照数据。**

- **链接: [http://arxiv.org/pdf/2505.19751v1](http://arxiv.org/pdf/2505.19751v1)**

> **作者:** Hala Djeghim; Nathan Piasco; Luis Roldão; Moussab Bennehar; Dzmitry Tsishkou; Céline Loscos; Désiré Sidibé
>
> **摘要:** Intrinsic image decomposition aims at separating an image into its underlying albedo and shading components, isolating the base color from lighting effects to enable downstream applications such as virtual relighting and scene editing. Despite the rise and success of learning-based approaches, intrinsic image decomposition from real-world images remains a significant challenging task due to the scarcity of labeled ground-truth data. Most existing solutions rely on synthetic data as supervised setups, limiting their ability to generalize to real-world scenes. Self-supervised methods, on the other hand, often produce albedo maps that contain reflections and lack consistency under different lighting conditions. To address this, we propose SAIL, an approach designed to estimate albedo-like representations from single-view real-world images. We repurpose the prior knowledge of a latent diffusion model for unconditioned scene relighting as a surrogate objective for albedo estimation. To extract the albedo, we introduce a novel intrinsic image decomposition fully formulated in the latent space. To guide the training of our latent diffusion model, we introduce regularization terms that constrain both the lighting-dependent and independent components of our latent image decomposition. SAIL predicts stable albedo under varying lighting conditions and generalizes to multiple scenes, using only unlabeled multi-illumination data available online.
>
---
#### [new 159] Translation-Equivariance of Normalization Layers and Aliasing in Convolutional Neural Networks
- **分类: cs.CV**

- **简介: 该论文属于卷积神经网络（CNN）架构优化任务，研究归一化层的平移等变性问题。提出理论框架分析其对离散和平移的等变条件，通过实验验证，提升模型在科学计算中的物理准确性。**

- **链接: [http://arxiv.org/pdf/2505.19805v1](http://arxiv.org/pdf/2505.19805v1)**

> **作者:** Jérémy Scanvic; Quentin Barthélemy; Julián Tachella
>
> **摘要:** The design of convolutional neural architectures that are exactly equivariant to continuous translations is an active field of research. It promises to benefit scientific computing, notably by making existing imaging systems more physically accurate. Most efforts focus on the design of downsampling/pooling layers, upsampling layers and activation functions, but little attention is dedicated to normalization layers. In this work, we present a novel theoretical framework for understanding the equivariance of normalization layers to discrete shifts and continuous translations. We also determine necessary and sufficient conditions for normalization layers to be equivariant in terms of the dimensions they operate on. Using real feature maps from ResNet-18 and ImageNet, we test those theoretical results empirically and find that they are consistent with our predictions.
>
---
#### [new 160] NEXT: Multi-Grained Mixture of Experts via Text-Modulation for Multi-Modal Object Re-ID
- **分类: cs.CV**

- **简介: 该论文属于多模态目标重识别（ReID）任务，旨在解决跨模态特征融合不足及复杂场景下细粒度识别精度低的问题。提出NEXT框架，通过文本调节的语义专家（TMSE）和结构专家（CSSE）分支分别捕捉细粒度语义与跨模态结构特征，并采用统一融合策略提升身份表征质量。**

- **链接: [http://arxiv.org/pdf/2505.20001v1](http://arxiv.org/pdf/2505.20001v1)**

> **作者:** Shihao Li; Chenglong Li; Aihua Zheng; Andong Lu; Jin Tang; Jixin Ma
>
> **摘要:** Multi-modal object re-identification (ReID) aims to extract identity features across heterogeneous spectral modalities to enable accurate recognition and retrieval in complex real-world scenarios. However, most existing methods rely on implicit feature fusion structures, making it difficult to model fine-grained recognition strategies under varying challenging conditions. Benefiting from the powerful semantic understanding capabilities of Multi-modal Large Language Models (MLLMs), the visual appearance of an object can be effectively translated into descriptive text. In this paper, we propose a reliable multi-modal caption generation method based on attribute confidence, which significantly reduces the unknown recognition rate of MLLMs in multi-modal semantic generation and improves the quality of generated text. Additionally, we propose a novel ReID framework NEXT, the Multi-grained Mixture of Experts via Text-Modulation for Multi-modal Object Re-Identification. Specifically, we decouple the recognition problem into semantic and structural expert branches to separately capture modality-specific appearance and intrinsic structure. For semantic recognition, we propose the Text-Modulated Semantic-sampling Experts (TMSE), which leverages randomly sampled high-quality semantic texts to modulate expert-specific sampling of multi-modal features and mining intra-modality fine-grained semantic cues. Then, to recognize coarse-grained structure features, we propose the Context-Shared Structure-aware Experts (CSSE) that focuses on capturing the holistic object structure across modalities and maintains inter-modality structural consistency through a soft routing mechanism. Finally, we propose the Multi-Modal Feature Aggregation (MMFA), which adopts a unified feature fusion strategy to simply and effectively integrate semantic and structural expert outputs into the final identity representations.
>
---
#### [new 161] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文属于3D医学影像诊断任务，旨在评估视觉语言模型（VLMs）在肿瘤中心视觉问答（VQA）中的临床适用性。针对VLMs在3D解剖结构中小肿瘤识别、跨模态推理等临床需求中的不足，构建了含9,262例CT数据及39.5万专家级问题的DeepTumorVQA基准集，测试四类VLMs，发现其在测量任务表现较好但识别和推理能力不足，强调多模态预训练及模型设计对3D感知的关键作用。**

- **链接: [http://arxiv.org/pdf/2505.18915v1](http://arxiv.org/pdf/2505.18915v1)**

> **作者:** Yixiong Chen; Wenjie Xiao; Pedro R. A. S. Bassi; Xinze Zhou; Sezgin Er; Ibrahim Ethem Hamamci; Zongwei Zhou; Alan Yuille
>
> **备注:** NeurIPS 2025 datasets&benchmarks track submission
>
> **摘要:** Vision-Language Models (VLMs) have shown promise in various 2D visual tasks, yet their readiness for 3D clinical diagnosis remains unclear due to stringent demands for recognition precision, reasoning ability, and domain knowledge. To systematically evaluate these dimensions, we present DeepTumorVQA, a diagnostic visual question answering (VQA) benchmark targeting abdominal tumors in CT scans. It comprises 9,262 CT volumes (3.7M slices) from 17 public datasets, with 395K expert-level questions spanning four categories: Recognition, Measurement, Visual Reasoning, and Medical Reasoning. DeepTumorVQA introduces unique challenges, including small tumor detection and clinical reasoning across 3D anatomy. Benchmarking four advanced VLMs (RadFM, M3D, Merlin, CT-CHAT), we find current models perform adequately on measurement tasks but struggle with lesion recognition and reasoning, and are still not meeting clinical needs. Two key insights emerge: (1) large-scale multimodal pretraining plays a crucial role in DeepTumorVQA testing performance, making RadFM stand out among all VLMs. (2) Our dataset exposes critical differences in VLM components, where proper image preprocessing and design of vision modules significantly affect 3D perception. To facilitate medical multimodal research, we have released DeepTumorVQA as a rigorous benchmark: https://github.com/Schuture/DeepTumorVQA.
>
---
#### [new 162] Improving Novel view synthesis of 360$^\circ$ Scenes in Extremely Sparse Views by Jointly Training Hemisphere Sampled Synthetic Images
- **分类: cs.CV**

- **简介: 该论文属于360°场景新型视图合成任务，解决极稀疏输入（如4视图）下的过拟合与重建质量差问题。工作包括：用DUSt3R估计相机位姿并生成点云，通过采样上半球空间合成密集视图，结合稀疏实拍图像训练3D高斯散射模型，再用扩散模型优化图像质量，显著提升极稀疏条件下的视图合成效果。**

- **链接: [http://arxiv.org/pdf/2505.19264v1](http://arxiv.org/pdf/2505.19264v1)**

> **作者:** Guangan Chen; Anh Minh Truong; Hanhe Lin; Michiel Vlaminck; Wilfried Philips; Hiep Luong
>
> **摘要:** Novel view synthesis in 360$^\circ$ scenes from extremely sparse input views is essential for applications like virtual reality and augmented reality. This paper presents a novel framework for novel view synthesis in extremely sparse-view cases. As typical structure-from-motion methods are unable to estimate camera poses in extremely sparse-view cases, we apply DUSt3R to estimate camera poses and generate a dense point cloud. Using the poses of estimated cameras, we densely sample additional views from the upper hemisphere space of the scenes, from which we render synthetic images together with the point cloud. Training 3D Gaussian Splatting model on a combination of reference images from sparse views and densely sampled synthetic images allows a larger scene coverage in 3D space, addressing the overfitting challenge due to the limited input in sparse-view cases. Retraining a diffusion-based image enhancement model on our created dataset, we further improve the quality of the point-cloud-rendered images by removing artifacts. We compare our framework with benchmark methods in cases of only four input views, demonstrating significant improvement in novel view synthesis under extremely sparse-view conditions for 360$^\circ$ scenes.
>
---
#### [new 163] UltraVSR: Achieving Ultra-Realistic Video Super-Resolution with Efficient One-Step Diffusion Space
- **分类: cs.CV**

- **简介: 该论文属于视频超分辨率（VSR）任务，解决扩散模型在视频超分辨率中的随机性与时序不连贯问题。提出UltraVSR框架，通过降质感知恢复计划（DRS）实现单步无噪重建，结合时移模块（RTS）与时空蒸馏策略，确保时序一致性和细节保真度，提升效率与效果。**

- **链接: [http://arxiv.org/pdf/2505.19958v1](http://arxiv.org/pdf/2505.19958v1)**

> **作者:** Yong Liu; Jinshan Pan; Yinchuan Li; Qingji Dong; Chao Zhu; Yu Guo; Fei Wang
>
> **备注:** Under review, 10 pages, 7 figures
>
> **摘要:** Diffusion models have shown great potential in generating realistic image detail. However, adapting these models to video super-resolution (VSR) remains challenging due to their inherent stochasticity and lack of temporal modeling. In this paper, we propose UltraVSR, a novel framework that enables ultra-realistic and temporal-coherent VSR through an efficient one-step diffusion space. A central component of UltraVSR is the Degradation-aware Restoration Schedule (DRS), which estimates a degradation factor from the low-resolution input and transforms iterative denoising process into a single-step reconstruction from from low-resolution to high-resolution videos. This design eliminates randomness from diffusion noise and significantly speeds up inference. To ensure temporal consistency, we propose a lightweight yet effective Recurrent Temporal Shift (RTS) module, composed of an RTS-convolution unit and an RTS-attention unit. By partially shifting feature components along the temporal dimension, these two units collaboratively facilitate effective feature propagation, fusion, and alignment across neighboring frames, without relying on explicit temporal layers. The RTS module is integrated into a pretrained text-to-image diffusion model and is further enhanced through Spatio-temporal Joint Distillation (SJD), which improves temporal coherence while preserving realistic details. Additionally, we introduce a Temporally Asynchronous Inference (TAI) strategy to capture long-range temporal dependencies under limited memory constraints. Extensive experiments show that UltraVSR achieves state-of-the-art performance, both qualitatively and quantitatively, in a single sampling step.
>
---
#### [new 164] JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型（VLM）安全攻击任务，解决现有越狱攻击方法效果差、忽略跨模态交互的问题。提出JailBound框架，通过探测融合层潜在空间的安全边界并联合优化图文扰动，提升攻击成功率，揭示VLMs隐含的安全风险。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19610v1](http://arxiv.org/pdf/2505.19610v1)**

> **作者:** Jiaxin Song; Yixu Wang; Jie Li; Rui Yu; Yan Teng; Xingjun Ma; Yingchun Wang
>
> **摘要:** Vision-Language Models (VLMs) exhibit impressive performance, yet the integration of powerful vision encoders has significantly broadened their attack surface, rendering them increasingly susceptible to jailbreak attacks. However, lacking well-defined attack objectives, existing jailbreak methods often struggle with gradient-based strategies prone to local optima and lacking precise directional guidance, and typically decouple visual and textual modalities, thereby limiting their effectiveness by neglecting crucial cross-modal interactions. Inspired by the Eliciting Latent Knowledge (ELK) framework, we posit that VLMs encode safety-relevant information within their internal fusion-layer representations, revealing an implicit safety decision boundary in the latent space. This motivates exploiting boundary to steer model behavior. Accordingly, we propose JailBound, a novel latent space jailbreak framework comprising two stages: (1) Safety Boundary Probing, which addresses the guidance issue by approximating decision boundary within fusion layer's latent space, thereby identifying optimal perturbation directions towards the target region; and (2) Safety Boundary Crossing, which overcomes the limitations of decoupled approaches by jointly optimizing adversarial perturbations across both image and text inputs. This latter stage employs an innovative mechanism to steer the model's internal state towards policy-violating outputs while maintaining cross-modal semantic consistency. Extensive experiments on six diverse VLMs demonstrate JailBound's efficacy, achieves 94.32% white-box and 67.28% black-box attack success averagely, which are 6.17% and 21.13% higher than SOTA methods, respectively. Our findings expose a overlooked safety risk in VLMs and highlight the urgent need for more robust defenses. Warning: This paper contains potentially sensitive, harmful and offensive content.
>
---
#### [new 165] Remote Sensing Image Classification with Decoupled Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分类任务，旨在解决模型参数过多导致部署困难的问题。提出采用G-GhostNet通过特征重用减少冗余参数，并结合解耦知识蒸馏分离目标/非目标类提升精度，在保持高分类准确率的同时将参数量减少6.24倍，实现轻量化与性能的平衡。**

- **链接: [http://arxiv.org/pdf/2505.19111v1](http://arxiv.org/pdf/2505.19111v1)**

> **作者:** Yaping He; Jianfeng Cai; Qicong Hu; Peiqing Wang
>
> **备注:** 7
>
> **摘要:** To address the challenges posed by the large number of parameters in existing remote sensing image classification models, which hinder deployment on resource-constrained devices, this paper proposes a lightweight classification method based on knowledge distillation. Specifically, G-GhostNet is adopted as the backbone network, leveraging feature reuse to reduce redundant parameters and significantly improve inference efficiency. In addition, a decoupled knowledge distillation strategy is employed, which separates target and non-target classes to effectively enhance classification accuracy. Experimental results on the RSOD and AID datasets demonstrate that, compared with the high-parameter VGG-16 model, the proposed method achieves nearly equivalent Top-1 accuracy while reducing the number of parameters by 6.24 times. This approach strikes an excellent balance between model size and classification performance, offering an efficient solution for deployment on resource-limited devices.
>
---
#### [new 166] SpikeStereoNet: A Brain-Inspired Framework for Stereo Depth Estimation from Spike Streams
- **分类: cs.CV**

- **简介: 该论文属于立体深度估计任务，针对传统相机在动态场景下性能不足及脉冲相机缺乏专用算法的问题。提出SpikeStereoNet框架，首次通过RSNN融合双视角脉冲流并迭代优化，同时构建合成与真实数据集。方法在弱纹理、极端光照场景表现优异且数据效率高。**

- **链接: [http://arxiv.org/pdf/2505.19487v1](http://arxiv.org/pdf/2505.19487v1)**

> **作者:** Zhuoheng Gao; Yihao Li; Jiyao Zhang; Rui Zhao; Tong Wu; Hao Tang; Zhaofei Yu; Hao Dong; Guozhang Chen; Tiejun Huang
>
> **摘要:** Conventional frame-based cameras often struggle with stereo depth estimation in rapidly changing scenes. In contrast, bio-inspired spike cameras emit asynchronous events at microsecond-level resolution, providing an alternative sensing modality. However, existing methods lack specialized stereo algorithms and benchmarks tailored to the spike data. To address this gap, we propose SpikeStereoNet, a brain-inspired framework and the first to estimate stereo depth directly from raw spike streams. The model fuses raw spike streams from two viewpoints and iteratively refines depth estimation through a recurrent spiking neural network (RSNN) update module. To benchmark our approach, we introduce a large-scale synthetic spike stream dataset and a real-world stereo spike dataset with dense depth annotations. SpikeStereoNet outperforms existing methods on both datasets by leveraging spike streams' ability to capture subtle edges and intensity shifts in challenging regions such as textureless surfaces and extreme lighting conditions. Furthermore, our framework exhibits strong data efficiency, maintaining high accuracy even with substantially reduced training data. The source code and datasets will be publicly available.
>
---
#### [new 167] Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对Video-LLMs处理长视频时因自回归特性导致的高延迟问题，提出Sparse-to-Dense（StD）策略。通过结合稀疏top-K注意力快速解码与密集注意力验证，实现无损加速（1.94倍），无需调参且易集成，解决视频理解任务中的效率瓶颈。**

- **链接: [http://arxiv.org/pdf/2505.19155v1](http://arxiv.org/pdf/2505.19155v1)**

> **作者:** Xuan Zhang; Cunxiao Du; Sicheng Yu; Jiawei Wu; Fengzhuo Zhang; Wei Gao; Qian Liu
>
> **摘要:** Due to the auto-regressive nature of current video large language models (Video-LLMs), the inference latency increases as the input sequence length grows, posing challenges for the efficient processing of video sequences that are usually very long. We observe that during decoding, the attention scores of most tokens in Video-LLMs tend to be sparse and concentrated, with only certain tokens requiring comprehensive full attention. Based on this insight, we introduce Sparse-to-Dense (StD), a novel decoding strategy that integrates two distinct modules: one leveraging sparse top-K attention and the other employing dense full attention. These modules collaborate to accelerate Video-LLMs without loss. The fast (sparse) model speculatively decodes multiple tokens, while the slow (dense) model verifies them in parallel. StD is a tuning-free, plug-and-play solution that achieves up to a 1.94$\times$ walltime speedup in video processing. It maintains model performance while enabling a seamless transition from a standard Video-LLM to a sparse Video-LLM with minimal code modifications.
>
---
#### [new 168] K-Buffers: A Plug-in Method for Enhancing Neural Fields with Multiple Buffers
- **分类: cs.CV**

- **简介: 论文属于3D视觉/计算机图形学任务，旨在优化神经场渲染过程。现有方法侧重场景表示，而本文提出K-Buffers方法，通过多缓冲区生成像素特征图，经融合网络合并并解码生成图像，辅以加速策略。实验显示其有效提升神经点场和3DGS的渲染性能。**

- **链接: [http://arxiv.org/pdf/2505.19564v1](http://arxiv.org/pdf/2505.19564v1)**

> **作者:** Haofan Ren; Zunjie Zhu; Xiang Chen; Ming Lu; Rongfeng Lu; Chenggang Yan
>
> **备注:** 15 pages, 9 figures, IJCAI 2025
>
> **摘要:** Neural fields are now the central focus of research in 3D vision and computer graphics. Existing methods mainly focus on various scene representations, such as neural points and 3D Gaussians. However, few works have studied the rendering process to enhance the neural fields. In this work, we propose a plug-in method named K-Buffers that leverages multiple buffers to improve the rendering performance. Our method first renders K buffers from scene representations and constructs K pixel-wise feature maps. Then, We introduce a K-Feature Fusion Network (KFN) to merge the K pixel-wise feature maps. Finally, we adopt a feature decoder to generate the rendering image. We also introduce an acceleration strategy to improve rendering speed and quality. We apply our method to well-known radiance field baselines, including neural point fields and 3D Gaussian Splatting (3DGS). Extensive experiments demonstrate that our method effectively enhances the rendering performance of neural point fields and 3DGS.
>
---
#### [new 169] Freqformer: Image-Demoiréing Transformer via Efficient Frequency Decomposition
- **分类: cs.CV**

- **简介: 该论文提出Freqformer，针对图像去摩尔纹任务，解决现有方法难以分离摩尔纹导致的纹理与色彩失真问题。通过频率分解将摩尔纹拆解为高频纹理和低频色彩失真，采用双分支架构分别处理，并设计FCT模块融合结果及SA-CA优化注意力，实现高效高精度修复，模型轻量且性能最优。**

- **链接: [http://arxiv.org/pdf/2505.19120v1](http://arxiv.org/pdf/2505.19120v1)**

> **作者:** Xiaoyang Liu; Bolin Qiu; Jiezhang Cao; Zheng Chen; Yulun Zhang; Xiaokang Yang
>
> **摘要:** Image demoir\'eing remains a challenging task due to the complex interplay between texture corruption and color distortions caused by moir\'e patterns. Existing methods, especially those relying on direct image-to-image restoration, often fail to disentangle these intertwined artifacts effectively. While wavelet-based frequency-aware approaches offer a promising direction, their potential remains underexplored. In this paper, we present Freqformer, a Transformer-based framework specifically designed for image demoir\'eing through targeted frequency separation. Our method performs an effective frequency decomposition that explicitly splits moir\'e patterns into high-frequency spatially-localized textures and low-frequency scale-robust color distortions, which are then handled by a dual-branch architecture tailored to their distinct characteristics. We further propose a learnable Frequency Composition Transform (FCT) module to adaptively fuse the frequency-specific outputs, enabling consistent and high-fidelity reconstruction. To better aggregate the spatial dependencies and the inter-channel complementary information, we introduce a Spatial-Aware Channel Attention (SA-CA) module that refines moir\'e-sensitive regions without incurring high computational cost. Extensive experiments on various demoir\'eing benchmarks demonstrate that Freqformer achieves state-of-the-art performance with a compact model size. The code is publicly available at https://github.com/xyLiu339/Freqformer.
>
---
#### [new 170] ErpGS: Equirectangular Image Rendering enhanced with 3D Gaussian Regularization
- **分类: cs.CV**

- **简介: 该论文属于3D重建与新型视图合成任务，针对360度相机畸变导致3DGS方法渲染精度低的问题，提出ErpGS：通过几何正则化、尺度正则化、畸变感知权重及遮挡抑制mask优化3D高斯分布，提升equirectangular图像渲染精度，实验验证其优势。**

- **链接: [http://arxiv.org/pdf/2505.19883v1](http://arxiv.org/pdf/2505.19883v1)**

> **作者:** Shintaro Ito; Natsuki Takama; Koichi Ito; Hwann-Tzong Chen; Takafumi Aoki
>
> **备注:** Accepted to ICIP2025
>
> **摘要:** The use of multi-view images acquired by a 360-degree camera can reconstruct a 3D space with a wide area. There are 3D reconstruction methods from equirectangular images based on NeRF and 3DGS, as well as Novel View Synthesis (NVS) methods. On the other hand, it is necessary to overcome the large distortion caused by the projection model of a 360-degree camera when equirectangular images are used. In 3DGS-based methods, the large distortion of the 360-degree camera model generates extremely large 3D Gaussians, resulting in poor rendering accuracy. We propose ErpGS, which is Omnidirectional GS based on 3DGS to realize NVS addressing the problems. ErpGS introduce some rendering accuracy improvement techniques: geometric regularization, scale regularization, and distortion-aware weights and a mask to suppress the effects of obstacles in equirectangular images. Through experiments on public datasets, we demonstrate that ErpGS can render novel view images more accurately than conventional methods.
>
---
#### [new 171] Guard Me If You Know Me: Protecting Specific Face-Identity from Deepfakes
- **分类: cs.CV**

- **简介: 该论文属于个性化深度伪造检测任务，旨在保护知名人物面部身份免受深度伪造攻击。针对现有方法忽略先验身份知识且缺乏可解释性的问题，提出VIPGuard框架：通过多模态模型学习面部细节、身份级判别学习区分细微差异，并定制化建模目标身份特征，结合语义推理实现精准可解释检测，同时构建VIPBench基准进行评估。**

- **链接: [http://arxiv.org/pdf/2505.19582v1](http://arxiv.org/pdf/2505.19582v1)**

> **作者:** Kaiqing Lin; Zhiyuan Yan; Ke-Yue Zhang; Li Hao; Yue Zhou; Yuzhen Lin; Weixiang Li; Taiping Yao; Shouhong Ding; Bin Li
>
> **摘要:** Securing personal identity against deepfake attacks is increasingly critical in the digital age, especially for celebrities and political figures whose faces are easily accessible and frequently targeted. Most existing deepfake detection methods focus on general-purpose scenarios and often ignore the valuable prior knowledge of known facial identities, e.g., "VIP individuals" whose authentic facial data are already available. In this paper, we propose \textbf{VIPGuard}, a unified multimodal framework designed to capture fine-grained and comprehensive facial representations of a given identity, compare them against potentially fake or similar-looking faces, and reason over these comparisons to make accurate and explainable predictions. Specifically, our framework consists of three main stages. First, fine-tune a multimodal large language model (MLLM) to learn detailed and structural facial attributes. Second, we perform identity-level discriminative learning to enable the model to distinguish subtle differences between highly similar faces, including real and fake variations. Finally, we introduce user-specific customization, where we model the unique characteristics of the target face identity and perform semantic reasoning via MLLM to enable personalized and explainable deepfake detection. Our framework shows clear advantages over previous detection works, where traditional detectors mainly rely on low-level visual cues and provide no human-understandable explanations, while other MLLM-based models often lack a detailed understanding of specific face identities. To facilitate the evaluation of our method, we built a comprehensive identity-aware benchmark called \textbf{VIPBench} for personalized deepfake detection, involving the latest 7 face-swapping and 7 entire face synthesis techniques for generation.
>
---
#### [new 172] Self-Supervised and Generalizable Tokenization for CLIP-Based 3D Understanding
- **分类: cs.CV**

- **简介: 该论文属于CLIP驱动的3D场景理解任务，旨在解决现有3D分桶方法因空间尺度敏感导致的跨领域泛化差问题。提出S4Token，通过超点分组+坐标尺度归一化实现尺度不变学习，结合无监督掩码建模、聚类目标及跨模态蒸馏训练，并设计特征传播模块恢复点云细节。**

- **链接: [http://arxiv.org/pdf/2505.18819v1](http://arxiv.org/pdf/2505.18819v1)**

> **作者:** Guofeng Mei; Bin Ren; Juan Liu; Luigi Riz; Xiaoshui Huang; Xu Zheng; Yongshun Gong; Ming-Hsuan Yang; Nicu Sebe; Fabio Poiesi
>
> **备注:** 10 pages, tokenizer
>
> **摘要:** Vision-language models like CLIP can offer a promising foundation for 3D scene understanding when extended with 3D tokenizers. However, standard approaches, such as k-nearest neighbor or radius-based tokenization, struggle with cross-domain generalization due to sensitivity to dataset-specific spatial scales. We present a universal 3D tokenizer designed for scale-invariant representation learning with a frozen CLIP backbone. We show that combining superpoint-based grouping with coordinate scale normalization consistently outperforms conventional methods through extensive experimental analysis. Specifically, we introduce S4Token, a tokenization pipeline that produces semantically-informed tokens regardless of scene scale. Our tokenizer is trained without annotations using masked point modeling and clustering-based objectives, along with cross-modal distillation to align 3D tokens with 2D multi-view image features. For dense prediction tasks, we propose a superpoint-level feature propagation module to recover point-level detail from sparse tokens.
>
---
#### [new 173] JEDI: The Force of Jensen-Shannon Divergence in Disentangling Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出JEDI方法，属于扩散模型测试时适应任务。针对语义纠缠和复杂场景下组合对齐问题，通过Jensen-Shannon散度优化注意力图，结合对抗优化减少计算步数，并提出轻量级CLIP-free评估指标，提升模型分离效果且兼容主流架构。**

- **链接: [http://arxiv.org/pdf/2505.19166v1](http://arxiv.org/pdf/2505.19166v1)**

> **作者:** Eric Tillmann Bill; Enis Simsar; Thomas Hofmann
>
> **摘要:** We introduce JEDI, a test-time adaptation method that enhances subject separation and compositional alignment in diffusion models without requiring retraining or external supervision. JEDI operates by minimizing semantic entanglement in attention maps using a novel Jensen-Shannon divergence based objective. To improve efficiency, we leverage adversarial optimization, reducing the number of updating steps required. JEDI is model-agnostic and applicable to architectures such as Stable Diffusion 1.5 and 3.5, consistently improving prompt alignment and disentanglement in complex scenes. Additionally, JEDI provides a lightweight, CLIP-free disentanglement score derived from internal attention distributions, offering a principled benchmark for compositional alignment under test-time conditions. We will publicly release the implementation of our method.
>
---
#### [new 174] Benchmarking Large Multimodal Models for Ophthalmic Visual Question Answering with OphthalWeChat
- **分类: cs.CV; cs.AI**

- **简介: 该论文构建了眼科双语多模态VQA基准OphthalWeChat，评估视觉语言模型（VLMs）在眼科图像问答中的性能。解决领域专用评估基准缺失问题，收集微信图文数据生成中英QA对，分为六类子集，测试GPT-4o、Gemini 2.0等模型，分析其分类与开放问答表现差异，推动眼科AI应用发展。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19624v1](http://arxiv.org/pdf/2505.19624v1)**

> **作者:** Pusheng Xu; Xia Gong; Xiaolan Chen; Weiyi Zhang; Jiancheng Yang; Bingjie Yan; Meng Yuan; Yalin Zheng; Mingguang He; Danli Shi
>
> **摘要:** Purpose: To develop a bilingual multimodal visual question answering (VQA) benchmark for evaluating VLMs in ophthalmology. Methods: Ophthalmic image posts and associated captions published between January 1, 2016, and December 31, 2024, were collected from WeChat Official Accounts. Based on these captions, bilingual question-answer (QA) pairs in Chinese and English were generated using GPT-4o-mini. QA pairs were categorized into six subsets by question type and language: binary (Binary_CN, Binary_EN), single-choice (Single-choice_CN, Single-choice_EN), and open-ended (Open-ended_CN, Open-ended_EN). The benchmark was used to evaluate the performance of three VLMs: GPT-4o, Gemini 2.0 Flash, and Qwen2.5-VL-72B-Instruct. Results: The final OphthalWeChat dataset included 3,469 images and 30,120 QA pairs across 9 ophthalmic subspecialties, 548 conditions, 29 imaging modalities, and 68 modality combinations. Gemini 2.0 Flash achieved the highest overall accuracy (0.548), outperforming GPT-4o (0.522, P < 0.001) and Qwen2.5-VL-72B-Instruct (0.514, P < 0.001). It also led in both Chinese (0.546) and English subsets (0.550). Subset-specific performance showed Gemini 2.0 Flash excelled in Binary_CN (0.687), Single-choice_CN (0.666), and Single-choice_EN (0.646), while GPT-4o ranked highest in Binary_EN (0.717), Open-ended_CN (BLEU-1: 0.301; BERTScore: 0.382), and Open-ended_EN (BLEU-1: 0.183; BERTScore: 0.240). Conclusions: This study presents the first bilingual VQA benchmark for ophthalmology, distinguished by its real-world context and inclusion of multiple examinations per patient. The dataset reflects authentic clinical decision-making scenarios and enables quantitative evaluation of VLMs, supporting the development of accurate, specialized, and trustworthy AI systems for eye care.
>
---
#### [new 175] On Denoising Walking Videos for Gait Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于步态识别任务，旨在去除行走视频中与身份无关的干扰因素（如衣物纹理、颜色）。提出DenoisingGait方法，结合生成扩散模型与几何驱动特征匹配模块，通过背景消除和特征浓缩生成Gait Feature Field表征，减少噪声并提升识别精度，实验显示达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.18582v1](http://arxiv.org/pdf/2505.18582v1)**

> **作者:** Dongyang Jin; Chao Fan; Jingzhe Ma; Jingkai Zhou; Weihua Chen; Shiqi Yu
>
> **备注:** 8pages, 4 figures
>
> **摘要:** To capture individual gait patterns, excluding identity-irrelevant cues in walking videos, such as clothing texture and color, remains a persistent challenge for vision-based gait recognition. Traditional silhouette- and pose-based methods, though theoretically effective at removing such distractions, often fall short of high accuracy due to their sparse and less informative inputs. Emerging end-to-end methods address this by directly denoising RGB videos using human priors. Building on this trend, we propose DenoisingGait, a novel gait denoising method. Inspired by the philosophy that "what I cannot create, I do not understand", we turn to generative diffusion models, uncovering how they partially filter out irrelevant factors for gait understanding. Additionally, we introduce a geometry-driven Feature Matching module, which, combined with background removal via human silhouettes, condenses the multi-channel diffusion features at each foreground pixel into a two-channel direction vector. Specifically, the proposed within- and cross-frame matching respectively capture the local vectorized structures of gait appearance and motion, producing a novel flow-like gait representation termed Gait Feature Field, which further reduces residual noise in diffusion features. Experiments on the CCPG, CASIA-B*, and SUSTech1K datasets demonstrate that DenoisingGait achieves a new SoTA performance in most cases for both within- and cross-domain evaluations. Code is available at https://github.com/ShiqiYu/OpenGait.
>
---
#### [new 176] DVD-Quant: Data-free Video Diffusion Transformers Quantization
- **分类: cs.CV**

- **简介: 该论文提出DVD-Quant框架，解决视频扩散Transformer模型量化中的数据依赖和性能下降问题。通过渐进有界量化、自适应旋转量化及动态位宽分配技术，在无需校准数据情况下实现高效量化，使HunyuanVideo速度提升2倍且保持画质，首次实现W4A4量化部署。**

- **链接: [http://arxiv.org/pdf/2505.18663v1](http://arxiv.org/pdf/2505.18663v1)**

> **作者:** Zhiteng Li; Hanxuan Li; Junyi Wu; Kai Liu; Linghe Kong; Guihai Chen; Yulun Zhang; Xiaokang Yang
>
> **备注:** Code and models will be available at \url{https://github.com/lhxcs/DVD-Quant}
>
> **摘要:** Diffusion Transformers (DiTs) have emerged as the state-of-the-art architecture for video generation, yet their computational and memory demands hinder practical deployment. While post-training quantization (PTQ) presents a promising approach to accelerate Video DiT models, existing methods suffer from two critical limitations: (1) dependence on lengthy, computation-heavy calibration procedures, and (2) considerable performance deterioration after quantization. To address these challenges, we propose DVD-Quant, a novel Data-free quantization framework for Video DiTs. Our approach integrates three key innovations: (1) Progressive Bounded Quantization (PBQ) and (2) Auto-scaling Rotated Quantization (ARQ) for calibration data-free quantization error reduction, as well as (3) $\delta$-Guided Bit Switching ($\delta$-GBS) for adaptive bit-width allocation. Extensive experiments across multiple video generation benchmarks demonstrate that DVD-Quant achieves an approximately 2$\times$ speedup over full-precision baselines on HunyuanVideo while maintaining visual fidelity. Notably, DVD-Quant is the first to enable W4A4 PTQ for Video DiTs without compromising video quality. Code and models will be available at https://github.com/lhxcs/DVD-Quant.
>
---
#### [new 177] Affective Image Editing: Shaping Emotional Factors via Text Descriptions
- **分类: cs.CV**

- **简介: 该论文提出AIEdiT模型，属于情感驱动的图像编辑任务。旨在解决现有文本编辑方法无法有效处理用户情感需求的问题。工作包括构建情感光谱提取细腻需求、设计情感映射器转化抽象情感为视觉语义，并结合多语言模型监督训练及情感对齐数据集，实现按文本指令精准调整图像情感表达。**

- **链接: [http://arxiv.org/pdf/2505.18699v1](http://arxiv.org/pdf/2505.18699v1)**

> **作者:** Peixuan Zhang; Shuchen Weng; Chengxuan Zhu; Binghao Tang; Zijian Jia; Si Li; Boxin Shi
>
> **摘要:** In daily life, images as common affective stimuli have widespread applications. Despite significant progress in text-driven image editing, there is limited work focusing on understanding users' emotional requests. In this paper, we introduce AIEdiT for Affective Image Editing using Text descriptions, which evokes specific emotions by adaptively shaping multiple emotional factors across the entire images. To represent universal emotional priors, we build the continuous emotional spectrum and extract nuanced emotional requests. To manipulate emotional factors, we design the emotional mapper to translate visually-abstract emotional requests to visually-concrete semantic representations. To ensure that editing results evoke specific emotions, we introduce an MLLM to supervise the model training. During inference, we strategically distort visual elements and subsequently shape corresponding emotional factors to edit images according to users' instructions. Additionally, we introduce a large-scale dataset that includes the emotion-aligned text and image pair set for training and evaluation. Extensive experiments demonstrate that AIEdiT achieves superior performance, effectively reflecting users' emotional requests.
>
---
#### [new 178] ADD-SLAM: Adaptive Dynamic Dense SLAM with Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于动态环境下的稠密SLAM任务，旨在解决动态物体干扰定位与建图的问题。提出ADD-SLAM框架，通过几何/纹理差异分析实现无先验动态识别，设计动态-静态分离建图策略，利用时序高斯模型在线增量建模动态场景，提升定位精度与动态分割效果。**

- **链接: [http://arxiv.org/pdf/2505.19420v1](http://arxiv.org/pdf/2505.19420v1)**

> **作者:** Wenhua Wu; Chenpeng Su; Siting Zhu; Tianchen Deng; Zhe Liu; Hesheng Wang
>
> **摘要:** Recent advancements in Neural Radiance Fields (NeRF) and 3D Gaussian-based Simultaneous Localization and Mapping (SLAM) methods have demonstrated exceptional localization precision and remarkable dense mapping performance. However, dynamic objects introduce critical challenges by disrupting scene consistency, leading to tracking drift and mapping artifacts. Existing methods that employ semantic segmentation or object detection for dynamic identification and filtering typically rely on predefined categorical priors, while discarding dynamic scene information crucial for robotic applications such as dynamic obstacle avoidance and environmental interaction. To overcome these challenges, we propose ADD-SLAM: an Adaptive Dynamic Dense SLAM framework based on Gaussian splitting. We design an adaptive dynamic identification mechanism grounded in scene consistency analysis, comparing geometric and textural discrepancies between real-time observations and historical maps. Ours requires no predefined semantic category priors and adaptively discovers scene dynamics. Precise dynamic object recognition effectively mitigates interference from moving targets during localization. Furthermore, we propose a dynamic-static separation mapping strategy that constructs a temporal Gaussian model to achieve online incremental dynamic modeling. Experiments conducted on multiple dynamic datasets demonstrate our method's flexible and accurate dynamic segmentation capabilities, along with state-of-the-art performance in both localization and mapping.
>
---
#### [new 179] Weakly-supervised Mamba-Based Mastoidectomy Shape Prediction for Cochlear Implant Surgery Using 3D T-Distribution Loss
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决弱监督下乳突切除术区域预测的鲁棒性不足问题。提出基于Mamba网络的弱监督框架，采用3D T-分布损失函数处理解剖结构变异，并利用自监督网络的分割结果作为弱标签，提升术前规划准确性与临床适用性。**

- **链接: [http://arxiv.org/pdf/2505.18368v1](http://arxiv.org/pdf/2505.18368v1)**

> **作者:** Yike Zhang; Jack H. Noble
>
> **摘要:** Cochlear implant surgery is a treatment for individuals with severe hearing loss. It involves inserting an array of electrodes inside the cochlea to electrically stimulate the auditory nerve and restore hearing sensation. A crucial step in this procedure is mastoidectomy, a surgical intervention that removes part of the mastoid region of the temporal bone, providing a critical pathway to the cochlea for electrode placement. Accurate prediction of the mastoidectomy region from preoperative imaging assists presurgical planning, reduces surgical risks, and improves surgical outcomes. In previous work, a self-supervised network was introduced to predict the mastoidectomy region using only preoperative CT scans. While promising, the method suffered from suboptimal robustness, limiting its practical application. To address this limitation, we propose a novel weakly-supervised Mamba-based framework to predict accurate mastoidectomy regions directly from preoperative CT scans. Our approach utilizes a 3D T-Distribution loss function inspired by the Student-t distribution, which effectively handles the complex geometric variability inherent in mastoidectomy shapes. Weak supervision is achieved using the segmentation results from the prior self-supervised network to eliminate the need for manual data cleaning or labeling throughout the training process. The proposed method is extensively evaluated against state-of-the-art approaches, demonstrating superior performance in predicting accurate and clinically relevant mastoidectomy regions. Our findings highlight the robustness and efficiency of the weakly-supervised learning framework with the proposed novel 3D T-Distribution loss.
>
---
#### [new 180] Applications and Effect Evaluation of Generative Adversarial Networks in Semi-Supervised Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于半监督学习任务，针对图像分类中标注数据不足的问题，提出基于GAN的模型，通过生成器、判别器与分类器的协同训练，有效利用有限标注和大量无标注数据，提升图像生成质量和分类精度，解决复杂环境下的图像识别难题。**

- **链接: [http://arxiv.org/pdf/2505.19522v1](http://arxiv.org/pdf/2505.19522v1)**

> **作者:** Jiyu Hu; Haijiang Zeng; Zhen Tian
>
> **摘要:** In recent years, image classification, as a core task in computer vision, relies on high-quality labelled data, which restricts the wide application of deep learning models in practical scenarios. To alleviate the problem of insufficient labelled samples, semi-supervised learning has gradually become a research hotspot. In this paper, we construct a semi-supervised image classification model based on Generative Adversarial Networks (GANs), and through the introduction of the collaborative training mechanism of generators, discriminators and classifiers, we achieve the effective use of limited labelled data and a large amount of unlabelled data, improve the quality of image generation and classification accuracy, and provide an effective solution for the task of image recognition in complex environments.
>
---
#### [new 181] Objective, Absolute and Hue-aware Metrics for Intrinsic Image Decomposition on Real-World Scenes: A Proof of Concept
- **分类: cs.CV**

- **简介: 该论文属于内在图像分解（IID）任务，旨在解决真实场景中因缺乏 ground truth 导致的IID质量客观评估难题。针对现有方法依赖主观标注且忽略色调的问题，提出基于高光谱成像与LiDAR强度计算反照率的定量评估指标，并引入光谱相似性驱动的反照率增强方法，在实验室验证了客观、绝对及色调感知评估的可行性。**

- **链接: [http://arxiv.org/pdf/2505.19500v1](http://arxiv.org/pdf/2505.19500v1)**

> **作者:** Shogo Sato; Masaru Tsuchida; Mariko Yamaguchi; Takuhiro Kaneko; Kazuhiko Murasaki; Taiga Yoshida; Ryuichi Tanida
>
> **备注:** copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Intrinsic image decomposition (IID) is the task of separating an image into albedo and shade. In real-world scenes, it is difficult to quantitatively assess IID quality due to the unavailability of ground truth. The existing method provides the relative reflection intensities based on human-judged annotations. However, these annotations have challenges in subjectivity, relative evaluation, and hue non-assessment. To address these, we propose a concept of quantitative evaluation with a calculated albedo from a hyperspectral imaging and light detection and ranging (LiDAR) intensity. Additionally, we introduce an optional albedo densification approach based on spectral similarity. This paper conducted a concept verification in a laboratory environment, and suggested the feasibility of an objective, absolute, and hue-aware assessment. (This paper is accepted by IEEE ICIP 2025. )
>
---
#### [new 182] Rehabilitation Exercise Quality Assessment and Feedback Generation Using Large Language Models with Prompt Engineering
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于康复训练质量评估与反馈生成任务，针对传统方法在AI反馈能力不足及缺乏文本反馈数据的问题，提出基于大语言模型的新方法：提取患者骨骼特征输入LLM，结合零样本、链式推理等提示策略生成自然语言反馈，在两个公开数据集验证有效。**

- **链接: [http://arxiv.org/pdf/2505.18412v1](http://arxiv.org/pdf/2505.18412v1)**

> **作者:** Jessica Tang; Ali Abedi; Tracey J. F. Colella; Shehroz S. Khan
>
> **备注:** 16 pages, 3 figures, 5 tables
>
> **摘要:** Exercise-based rehabilitation improves quality of life and reduces morbidity, mortality, and rehospitalization, though transportation constraints and staff shortages lead to high dropout rates from rehabilitation programs. Virtual platforms enable patients to complete prescribed exercises at home, while AI algorithms analyze performance, deliver feedback, and update clinicians. Although many studies have developed machine learning and deep learning models for exercise quality assessment, few have explored the use of large language models (LLMs) for feedback and are limited by the lack of rehabilitation datasets containing textual feedback. In this paper, we propose a new method in which exercise-specific features are extracted from the skeletal joints of patients performing rehabilitation exercises and fed into pre-trained LLMs. Using a range of prompting techniques, such as zero-shot, few-shot, chain-of-thought, and role-play prompting, LLMs are leveraged to evaluate exercise quality and provide feedback in natural language to help patients improve their movements. The method was evaluated through extensive experiments on two publicly available rehabilitation exercise assessment datasets (UI-PRMD and REHAB24-6) and showed promising results in exercise assessment, reasoning, and feedback generation. This approach can be integrated into virtual rehabilitation platforms to help patients perform exercises correctly, support recovery, and improve health outcomes.
>
---
#### [new 183] PosePilot: An Edge-AI Solution for Posture Correction in Physical Exercises
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于AI驱动的健身辅助任务，旨在解决传统系统在实时姿势纠正中的不足。提出PosePilot系统，结合LSTM与注意力机制，实现边缘设备上的实时姿势识别与个性化反馈，通过瑜伽案例验证其精准捕捉肢体角度、提供动态纠错的能力，并构建专用数据集提升模型鲁棒性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19186v1](http://arxiv.org/pdf/2505.19186v1)**

> **作者:** Rushiraj Gadhvi; Priyansh Desai; Siddharth
>
> **备注:** Accepted for publication at IBPRIA 2025 Conference in Coimbra, Portugal
>
> **摘要:** Automated pose correction remains a significant challenge in AI-driven fitness systems, despite extensive research in activity recognition. This work presents PosePilot, a novel system that integrates pose recognition with real-time personalized corrective feedback, overcoming the limitations of traditional fitness solutions. Using Yoga, a discipline requiring precise spatio-temporal alignment as a case study, we demonstrate PosePilot's ability to analyze complex physical movements. Designed for deployment on edge devices, PosePilot can be extended to various at-home and outdoor exercises. We employ a Vanilla LSTM, allowing the system to capture temporal dependencies for pose recognition. Additionally, a BiLSTM with multi-head Attention enhances the model's ability to process motion contexts, selectively focusing on key limb angles for accurate error detection while maintaining computational efficiency. As part of this work, we introduce a high-quality video dataset used for evaluating our models. Most importantly, PosePilot provides instant corrective feedback at every stage of a movement, ensuring precise posture adjustments throughout the exercise routine. The proposed approach 1) performs automatic human posture recognition, 2) provides personalized posture correction feedback at each instant which is crucial in Yoga, and 3) offers a lightweight and robust posture correction model feasible for deploying on edge devices in real-world environments.
>
---
#### [new 184] OmniGenBench: A Benchmark for Omnipotent Multimodal Generation across 50+ Tasks
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出多模态生成基准OmniGenBench，解决现有评估标准广度深度不足的问题。设计包含57个真实场景任务的评测体系，通过视觉工具和LLM评估者双模式协议，全面评估GPT-4o等模型的感知与认知能力。**

- **链接: [http://arxiv.org/pdf/2505.18775v1](http://arxiv.org/pdf/2505.18775v1)**

> **作者:** Jiayu Wang; Yang Jiao; Yue Yu; Tianwen Qian; Shaoxiang Chen; Jingjing Chen; Yu-Gang Jiang
>
> **摘要:** Recent breakthroughs in large multimodal models (LMMs), such as the impressive GPT-4o-Native, have demonstrated remarkable proficiency in following general-purpose instructions for image generation. However, current benchmarks often lack the necessary breadth and depth to fully evaluate the diverse capabilities of these models. To overcome this limitation, we introduce OmniGenBench, a novel and comprehensive benchmark meticulously designed to assess the instruction-following abilities of state-of-the-art LMMs across both perception-centric and cognition-centric dimensions. Our OmniGenBench includes 57 diverse sub-tasks grounded in real-world scenarios, systematically categorized according to the specific model capabilities they demand. For rigorous evaluation, we further employ a dual-mode protocol. This protocol utilizes off-the-shelf visual parsing tools for perception-centric tasks and a powerful LLM-based judger for cognition-centric tasks to assess the alignment between generated images and user instructions. Using OmniGenBench, we evaluate mainstream generative models, including prevalent models like GPT-4o, Gemini-2.0-Flash, and Seedream, and provide in-depth comparisons and analyses of their performance.Code and data are available at https://github.com/emilia113/OmniGenBench.
>
---
#### [new 185] Hard Negative Contrastive Learning for Fine-Grained Geometric Understanding in Large Multimodal Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于几何推理任务，旨在提升大模型的细粒度几何理解能力。针对对比学习在几何问题中推理能力不足的问题，提出结合生成与规则-based硬负样本的对比学习框架，训练多模态模型MMGeoLM，显著提升几何推理性能，超越现有开源模型并接近闭源模型水平。**

- **链接: [http://arxiv.org/pdf/2505.20152v1](http://arxiv.org/pdf/2505.20152v1)**

> **作者:** Kai Sun; Yushi Bai; Zhen Yang; Jiajie Zhang; Ji Qi; Lei Hou; Juanzi Li
>
> **摘要:** Benefiting from contrastively trained visual encoders on large-scale natural scene images, Large Multimodal Models (LMMs) have achieved remarkable performance across various visual perception tasks. However, the inherent limitations of contrastive learning upon summarized descriptions fundamentally restrict the capabilities of models in meticulous reasoning, particularly in crucial scenarios of geometric problem-solving. To enhance geometric understanding, we propose a novel hard negative contrastive learning framework for the vision encoder, which combines image-based contrastive learning using generation-based hard negatives created by perturbing diagram generation code, and text-based contrastive learning using rule-based negatives derived from modified geometric descriptions and retrieval-based negatives selected based on caption similarity. We train CLIP using our strong negative learning method, namely MMCLIP (Multimodal Math CLIP), and subsequently train an LMM for geometric problem-solving. Experiments show that our trained model, MMGeoLM, significantly outperforms other open-source models on three geometric reasoning benchmarks. Even with a size of 7B, it can rival powerful closed-source models like GPT-4o. We further study the impact of different negative sample construction methods and the number of negative samples on the geometric reasoning performance of LMM, yielding fruitful conclusions. The code and dataset are available at https://github.com/THU-KEG/MMGeoLM.
>
---
#### [new 186] Absolute Coordinates Make Motion Generation Easy
- **分类: cs.CV**

- **简介: 该论文属于文本到动作生成任务，针对现有局部相对坐标表示在扩散模型中的限制及下游任务适配问题，提出采用全局绝对坐标表示，提升动作保真度与文本对齐，支持直接编辑、控制及生成SMPL-H网格顶点，简化模型设计且无需额外损失。**

- **链接: [http://arxiv.org/pdf/2505.19377v1](http://arxiv.org/pdf/2505.19377v1)**

> **作者:** Zichong Meng; Zeyu Han; Xiaogang Peng; Yiming Xie; Huaizu Jiang
>
> **备注:** Preprint
>
> **摘要:** State-of-the-art text-to-motion generation models rely on the kinematic-aware, local-relative motion representation popularized by HumanML3D, which encodes motion relative to the pelvis and to the previous frame with built-in redundancy. While this design simplifies training for earlier generation models, it introduces critical limitations for diffusion models and hinders applicability to downstream tasks. In this work, we revisit the motion representation and propose a radically simplified and long-abandoned alternative for text-to-motion generation: absolute joint coordinates in global space. Through systematic analysis of design choices, we show that this formulation achieves significantly higher motion fidelity, improved text alignment, and strong scalability, even with a simple Transformer backbone and no auxiliary kinematic-aware losses. Moreover, our formulation naturally supports downstream tasks such as text-driven motion control and temporal/spatial editing without additional task-specific reengineering and costly classifier guidance generation from control signals. Finally, we demonstrate promising generalization to directly generate SMPL-H mesh vertices in motion from text, laying a strong foundation for future research and motion-related applications.
>
---
#### [new 187] EventEgoHands: Event-based Egocentric 3D Hand Mesh Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出EventEgoHands方法，解决事件相机在动态背景下进行自中心3D手部网格重建时的噪声干扰问题。通过手部分割模块抑制背景事件，提升重建精度，在N-HOT3D数据集上MPJPE改善超4.3cm。任务属事件驱动的3D手部形状估计，针对传统相机在低光/运动模糊下的不足及事件相机对动态环境敏感的局限进行优化。**

- **链接: [http://arxiv.org/pdf/2505.19169v1](http://arxiv.org/pdf/2505.19169v1)**

> **作者:** Ryosei Hara; Wataru Ikeda; Masashi Hatano; Mariko Isogawa
>
> **备注:** IEEE International Conference on Image Processing 2025
>
> **摘要:** Reconstructing 3D hand mesh is challenging but an important task for human-computer interaction and AR/VR applications. In particular, RGB and/or depth cameras have been widely used in this task. However, methods using these conventional cameras face challenges in low-light environments and during motion blur. Thus, to address these limitations, event cameras have been attracting attention in recent years for their high dynamic range and high temporal resolution. Despite their advantages, event cameras are sensitive to background noise or camera motion, which has limited existing studies to static backgrounds and fixed cameras. In this study, we propose EventEgoHands, a novel method for event-based 3D hand mesh reconstruction in an egocentric view. Our approach introduces a Hand Segmentation Module that extracts hand regions, effectively mitigating the influence of dynamic background events. We evaluated our approach and demonstrated its effectiveness on the N-HOT3D dataset, improving MPJPE by approximately more than 4.5 cm (43%).
>
---
#### [new 188] Domain and Task-Focused Example Selection for Data-Efficient Contrastive Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决标注数据稀缺及跨领域分割效果差的问题。提出PolyCL框架，结合对比自监督学习与SAM模型，通过创新代理任务提取领域相关特征，并利用SAM进行后处理和三维分割传播，在有限数据下提升分割精度。**

- **链接: [http://arxiv.org/pdf/2505.19208v1](http://arxiv.org/pdf/2505.19208v1)**

> **作者:** Tyler Ward; Aaron Moseley; Abdullah-Al-Zubaer Imran
>
> **摘要:** Segmentation is one of the most important tasks in the medical imaging pipeline as it influences a number of image-based decisions. To be effective, fully supervised segmentation approaches require large amounts of manually annotated training data. However, the pixel-level annotation process is expensive, time-consuming, and error-prone, hindering progress and making it challenging to perform effective segmentations. Therefore, models must learn efficiently from limited labeled data. Self-supervised learning (SSL), particularly contrastive learning via pre-training on unlabeled data and fine-tuning on limited annotations, can facilitate such limited labeled image segmentation. To this end, we propose a novel self-supervised contrastive learning framework for medical image segmentation, leveraging inherent relationships of different images, dubbed PolyCL. Without requiring any pixel-level annotations or unreasonable data augmentations, our PolyCL learns and transfers context-aware discriminant features useful for segmentation from an innovative surrogate, in a task-related manner. Additionally, we integrate the Segment Anything Model (SAM) into our framework in two novel ways: as a post-processing refinement module that improves the accuracy of predicted masks using bounding box prompts derived from coarse outputs, and as a propagation mechanism via SAM 2 that generates volumetric segmentations from a single annotated 2D slice. Experimental evaluations on three public computed tomography (CT) datasets demonstrate that PolyCL outperforms fully-supervised and self-supervised baselines in both low-data and cross-domain scenarios. Our code is available at https://github.com/tbwa233/PolyCL.
>
---
#### [new 189] Why Not Replace? Sustaining Long-Term Visual Localization via Handcrafted-Learned Feature Collaboration on CPU
- **分类: cs.CV**

- **简介: 该论文属于长期视觉定位任务，针对工业环境中的光照敏感、计算效率及环境约束问题，提出结合手工与深度学习特征的分层框架：实时手工特征用于连续跟踪，选择性使用学习特征进行绝对定位，实现CPU高效运行。实验显示其在光照变化下误差降低47%，提升定位一致性。**

- **链接: [http://arxiv.org/pdf/2505.18652v1](http://arxiv.org/pdf/2505.18652v1)**

> **作者:** Yicheng Lin; Yunlong Jiang; Xujia Jiao; Bin Han
>
> **备注:** 8 pages, 6 gifures
>
> **摘要:** Robust long-term visual localization in complex industrial environments is critical for mobile robotic systems. Existing approaches face limitations: handcrafted features are illumination-sensitive, learned features are computationally intensive, and semantic- or marker-based methods are environmentally constrained. Handcrafted and learned features share similar representations but differ functionally. Handcrafted features are optimized for continuous tracking, while learned features excel in wide-baseline matching. Their complementarity calls for integration rather than replacement. Building on this, we propose a hierarchical localization framework. It leverages real-time handcrafted feature extraction for relative pose estimation. In parallel, it employs selective learned keypoint detection on optimized keyframes for absolute positioning. This design enables CPU-efficient, long-term visual localization. Experiments systematically progress through three validation phases: Initially establishing feature complementarity through comparative analysis, followed by computational latency profiling across algorithm stages on CPU platforms. Final evaluation under photometric variations (including seasonal transitions and diurnal cycles) demonstrates 47% average error reduction with significantly improved localization consistency. The code implementation is publicly available at https://github.com/linyicheng1/ORB_SLAM3_localization.
>
---
#### [new 190] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models
- **分类: cs.CV**

- **简介: 论文提出SAMA模型，解决视频多轮参考性对话中的细粒度时空理解问题。针对现有方法孤立处理视频指代理解与定位且数据不足的问题，团队构建SAMA-239K数据集，开发集成时空聚合器与Segment Anything Model的SAMA模型，并建立SAMA-Bench评估基准，提升视频对话与定位性能。**

- **链接: [http://arxiv.org/pdf/2505.18812v1](http://arxiv.org/pdf/2505.18812v1)**

> **作者:** Ye Sun; Hao Zhang; Henghui Ding; Tiehua Zhang; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** Achieving fine-grained spatio-temporal understanding in videos remains a major challenge for current Video Large Multimodal Models (Video LMMs). Addressing this challenge requires mastering two core capabilities: video referring understanding, which captures the semantics of video regions, and video grounding, which segments object regions based on natural language descriptions. However, most existing approaches tackle these tasks in isolation, limiting progress toward unified, referentially grounded video interaction. We identify a key bottleneck in the lack of high-quality, unified video instruction data and a comprehensive benchmark for evaluating referentially grounded video chat. To address these challenges, we contribute in three core aspects: dataset, model, and benchmark. First, we introduce SAMA-239K, a large-scale dataset comprising 15K videos specifically curated to enable joint learning of video referring understanding, grounding, and multi-turn video chat. Second, we propose the SAMA model, which incorporates a versatile spatio-temporal context aggregator and a Segment Anything Model to jointly enhance fine-grained video comprehension and precise grounding capabilities. Finally, we establish SAMA-Bench, a meticulously designed benchmark consisting of 5,067 questions from 522 videos, to comprehensively evaluate the integrated capabilities of Video LMMs in multi-turn, spatio-temporal referring understanding and grounded dialogue. Extensive experiments and benchmarking results show that SAMA not only achieves strong performance on SAMA-Bench but also sets a new state-of-the-art on general grounding benchmarks, while maintaining highly competitive performance on standard visual understanding benchmarks.
>
---
#### [new 191] Beyond Editing Pairs: Fine-Grained Instructional Image Editing via Multi-Scale Learnable Regions
- **分类: cs.CV**

- **简介: 该论文属于指令驱动图像编辑任务，旨在解决依赖编辑配对数据集耗时低质或无数据集方法编辑能力受限的问题。提出通过多尺度可学习区域定位编辑区域，利用文本-图像对数据提升精准编辑与指令一致性，实现高性能、强适配的图像编辑方法。**

- **链接: [http://arxiv.org/pdf/2505.19352v1](http://arxiv.org/pdf/2505.19352v1)**

> **作者:** Chenrui Ma; Xi Xiao; Tianyang Wang; Yanning Shen
>
> **摘要:** Current text-driven image editing methods typically follow one of two directions: relying on large-scale, high-quality editing pair datasets to improve editing precision and diversity, or exploring alternative dataset-free techniques. However, constructing large-scale editing datasets requires carefully designed pipelines, is time-consuming, and often results in unrealistic samples or unwanted artifacts. Meanwhile, dataset-free methods may suffer from limited instruction comprehension and restricted editing capabilities. Faced with these challenges, the present work develops a novel paradigm for instruction-driven image editing that leverages widely available and enormous text-image pairs, instead of relying on editing pair datasets. Our approach introduces a multi-scale learnable region to localize and guide the editing process. By treating the alignment between images and their textual descriptions as supervision and learning to generate task-specific editing regions, our method achieves high-fidelity, precise, and instruction-consistent image editing. Extensive experiments demonstrate that the proposed approach attains state-of-the-art performance across various tasks and benchmarks, while exhibiting strong adaptability to various types of generative models.
>
---
#### [new 192] OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Model
- **分类: cs.CV**

- **简介: 该论文提出OpenHOI框架，解决开放世界3D手物交互合成的泛化问题。针对现有方法难以处理新对象及复杂指令，其整合多模态大语言模型实现语义分解与可操作区域定位，并结合扩散模型与物理优化，生成新型对象的长期交互序列。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18947v1](http://arxiv.org/pdf/2505.18947v1)**

> **作者:** Zhenhao Zhang; Ye Shi; Lingxiao Yang; Suting Ni; Qi Ye; Jingya Wang
>
> **摘要:** Understanding and synthesizing realistic 3D hand-object interactions (HOI) is critical for applications ranging from immersive AR/VR to dexterous robotics. Existing methods struggle with generalization, performing well on closed-set objects and predefined tasks but failing to handle unseen objects or open-vocabulary instructions. We introduce OpenHOI, the first framework for open-world HOI synthesis, capable of generating long-horizon manipulation sequences for novel objects guided by free-form language commands. Our approach integrates a 3D Multimodal Large Language Model (MLLM) fine-tuned for joint affordance grounding and semantic task decomposition, enabling precise localization of interaction regions (e.g., handles, buttons) and breakdown of complex instructions (e.g., "Find a water bottle and take a sip") into executable sub-tasks. To synthesize physically plausible interactions, we propose an affordance-driven diffusion model paired with a training-free physics refinement stage that minimizes penetration and optimizes affordance alignment. Evaluations across diverse scenarios demonstrate OpenHOI's superiority over state-of-the-art methods in generalizing to novel object categories, multi-stage tasks, and complex language instructions. Our project page at \href{https://openhoi.github.io}
>
---
#### [new 193] GRE Suite: Geo-localization Inference via Fine-Tuned Vision-Language Models and Enhanced Reasoning Chains
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出GRE Suite框架，解决视觉语言模型在地理定位任务中多粒度视觉线索与外部知识融合不足的问题。通过构建GRE30K数据集、多阶段推理模型及评估基准，增强模型推理能力与可解释性，提升粗细粒度定位精度，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.18700v1](http://arxiv.org/pdf/2505.18700v1)**

> **作者:** Chun Wang; Xiaoran Pan; Zihao Pan; Haofan Wang; Yiren Song
>
> **摘要:** Recent advances in Visual Language Models (VLMs) have demonstrated exceptional performance in visual reasoning tasks. However, geo-localization presents unique challenges, requiring the extraction of multigranular visual cues from images and their integration with external world knowledge for systematic reasoning. Current approaches to geo-localization tasks often lack robust reasoning mechanisms and explainability, limiting their effectiveness. To address these limitations, we propose the Geo Reason Enhancement (GRE) Suite, a novel framework that augments VLMs with structured reasoning chains for accurate and interpretable location inference. The GRE Suite is systematically developed across three key dimensions: dataset, model, and benchmark. First, we introduce GRE30K, a high-quality geo-localization reasoning dataset designed to facilitate fine-grained visual and contextual analysis. Next, we present the GRE model, which employs a multi-stage reasoning strategy to progressively infer scene attributes, local details, and semantic features, thereby narrowing down potential geographic regions with enhanced precision. Finally, we construct the Geo Reason Evaluation Benchmark (GREval-Bench), a comprehensive evaluation framework that assesses VLMs across diverse urban, natural, and landmark scenes to measure both coarse-grained (e.g., country, continent) and fine-grained (e.g., city, street) localization performance. Experimental results demonstrate that GRE significantly outperforms existing methods across all granularities of geo-localization tasks, underscoring the efficacy of reasoning-augmented VLMs in complex geographic inference. Code and data will be released at https://github.com/Thorin215/GRE.
>
---
#### [new 194] HonestFace: Towards Honest Face Restoration with One-Step Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于人脸修复任务，旨在解决现有方法在低质量输入恢复中身份不一致、纹理失真及伪影问题。提出HonestFace模型，通过身份嵌入器保留关键特征，结合掩码对齐优化细节，设计仿射变换驱动的评估指标，整合至一步扩散模型中，提升修复真实性与保真度，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.18469v1](http://arxiv.org/pdf/2505.18469v1)**

> **作者:** Jingkai Wang; Wu Miao; Jue Gong; Zheng Chen; Xing Liu; Hong Gu; Yutong Liu; Yulun Zhang
>
> **摘要:** Face restoration has achieved remarkable advancements through the years of development. However, ensuring that restored facial images exhibit high fidelity, preserve authentic features, and avoid introducing artifacts or biases remains a significant challenge. This highlights the need for models that are more "honest" in their reconstruction from low-quality inputs, accurately reflecting original characteristics. In this work, we propose HonestFace, a novel approach designed to restore faces with a strong emphasis on such honesty, particularly concerning identity consistency and texture realism. To achieve this, HonestFace incorporates several key components. First, we propose an identity embedder to effectively capture and preserve crucial identity features from both the low-quality input and multiple reference faces. Second, a masked face alignment method is presented to enhance fine-grained details and textural authenticity, thereby preventing the generation of patterned or overly synthetic textures and improving overall clarity. Furthermore, we present a new landmark-based evaluation metric. Based on affine transformation principles, this metric improves the accuracy compared to conventional L2 distance calculations for facial feature alignment. Leveraging these contributions within a one-step diffusion model framework, HonestFace delivers exceptional restoration results in terms of facial fidelity and realism. Extensive experiments demonstrate that our approach surpasses existing state-of-the-art methods, achieving superior performance in both visual quality and quantitative assessments. The code and pre-trained models will be made publicly available at https://github.com/jkwang28/HonestFace .
>
---
#### [new 195] Improving Heart Rejection Detection in XPCI Images Using Synthetic Data Augmentation
- **分类: cs.CV**

- **简介: 该论文针对心脏移植后高分级排斥（3R）样本稀缺导致的分类模型训练难题，提出基于StyleGAN的合成数据增强方法。通过直方图均衡化预处理真实3R图像，生成1万张合成样本，并与真实无排斥（0R）数据组合训练ResNet-18分类器。实验表明，混合真实与合成数据的模型在分类精度和召回率上表现最优，验证了GAN增强在医疗图像分析中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.19746v1](http://arxiv.org/pdf/2505.19746v1)**

> **作者:** Jakov Samardžija; Donik Vršnak; Sven Lončarić
>
> **摘要:** Accurate identification of acute cellular rejection (ACR) in endomyocardial biopsies is essential for effective management of heart transplant patients. However, the rarity of high-grade rejection cases (3R) presents a significant challenge for training robust deep learning models. This work addresses the class imbalance problem by leveraging synthetic data generation using StyleGAN to augment the limited number of real 3R images. Prior to GAN training, histogram equalization was applied to standardize image appearance and improve the consistency of tissue representation. StyleGAN was trained on available 3R biopsy patches and subsequently used to generate 10,000 realistic synthetic images. These were combined with real 0R samples, that is samples without rejection, in various configurations to train ResNet-18 classifiers for binary rejection classification. Three classifier variants were evaluated: one trained on real 0R and synthetic 3R images, another using both synthetic and additional real samples, and a third trained solely on real data. All models were tested on an independent set of real biopsy images. Results demonstrate that synthetic data improves classification performance, particularly when used in combination with real samples. The highest-performing model, which used both real and synthetic images, achieved strong precision and recall for both classes. These findings underscore the value of hybrid training strategies and highlight the potential of GAN-based data augmentation in biomedical image analysis, especially in domains constrained by limited annotated datasets.
>
---
#### [new 196] Advancing Video Self-Supervised Learning via Image Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于视频自监督学习任务，旨在利用预训练图像基础模型（IFMs）降低视频表示学习的训练成本。针对直接使用IFMs的潜力未被充分挖掘的问题，提出AdViSe方法：在IFMs中加入3D ResNet时序模块，冻结IFM部分，仅通过"播放速率感知"任务训练时序模块。实验显示其性能媲美SOTA，但训练时间与显存分别减少3.4和8.2倍。**

- **链接: [http://arxiv.org/pdf/2505.19218v1](http://arxiv.org/pdf/2505.19218v1)**

> **作者:** Jingwei Wu; Zhewei Huang; Chang Liu
>
> **摘要:** In the past decade, image foundation models (IFMs) have achieved unprecedented progress. However, the potential of directly using IFMs for video self-supervised representation learning has largely been overlooked. In this study, we propose an advancing video self-supervised learning (AdViSe) approach, aimed at significantly reducing the training overhead of video representation models using pre-trained IFMs. Specifically, we first introduce temporal modeling modules (ResNet3D) to IFMs, constructing a video representation model. We then employ a video self-supervised learning approach, playback rate perception, to train temporal modules while freezing the IFM components. Experiments on UCF101 demonstrate that AdViSe achieves performance comparable to state-of-the-art methods while reducing training time by $3.4\times$ and GPU memory usage by $8.2\times$. This study offers fresh insights into low-cost video self-supervised learning based on pre-trained IFMs. Code is available at https://github.com/JingwWu/advise-video-ssl.
>
---
#### [new 197] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出DriveX，一种自监督世界模型，通过整合3D点云预测、2D语义及图像生成的多模态监督，解决自动驾驶任务专用模型泛化性差的问题。其核心包括Omni Scene Modeling模块、解耦的latent世界建模及动态感知采样，提升场景动态理解与多任务（如占用预测、流估计）性能，实现通用自动驾驶框架。**

- **链接: [http://arxiv.org/pdf/2505.19239v1](http://arxiv.org/pdf/2505.19239v1)**

> **作者:** Chen Shi; Shaoshuai Shi; Kehua Sheng; Bo Zhang; Li Jiang
>
> **摘要:** Data-driven learning has advanced autonomous driving, yet task-specific models struggle with out-of-distribution scenarios due to their narrow optimization objectives and reliance on costly annotated data. We present DriveX, a self-supervised world model that learns generalizable scene dynamics and holistic representations (geometric, semantic, and motion) from large-scale driving videos. DriveX introduces Omni Scene Modeling (OSM), a module that unifies multimodal supervision-3D point cloud forecasting, 2D semantic representation, and image generation-to capture comprehensive scene evolution. To simplify learning complex dynamics, we propose a decoupled latent world modeling strategy that separates world representation learning from future state decoding, augmented by dynamic-aware ray sampling to enhance motion modeling. For downstream adaptation, we design Future Spatial Attention (FSA), a unified paradigm that dynamically aggregates spatiotemporal features from DriveX's predictions to enhance task-specific inference. Extensive experiments demonstrate DriveX's effectiveness: it achieves significant improvements in 3D future point cloud prediction over prior work, while attaining state-of-the-art results on diverse tasks including occupancy prediction, flow estimation, and end-to-end driving. These results validate DriveX's capability as a general-purpose world model, paving the way for robust and unified autonomous driving frameworks.
>
---
#### [new 198] Revolutionizing Wildfire Detection with Convolutional Neural Networks: A VGG16 Model Approach
- **分类: cs.CV; cs.LG**

- **简介: 论文提出基于VGG16的CNN模型，用于野火检测的二分类任务。针对数据不平衡、低分辨率及实时需求，采用数据增强与模型优化，降低假阴性率，提升早期预警准确性。**

- **链接: [http://arxiv.org/pdf/2505.19479v1](http://arxiv.org/pdf/2505.19479v1)**

> **作者:** Lakshmi Aishwarya Malladi; Navarun Gupta; Ahmed El-Sayed; Xingguo Xiong
>
> **备注:** Conference at ASEE 2025
>
> **摘要:** Over 8,024 wildfire incidents have been documented in 2024 alone, affecting thousands of fatalities and significant damage to infrastructure and ecosystems. Wildfires in the United States have inflicted devastating losses. Wildfires are becoming more frequent and intense, which highlights how urgently efficient warning systems are needed to avoid disastrous outcomes. The goal of this study is to enhance the accuracy of wildfire detection by using Convolutional Neural Network (CNN) built on the VGG16 architecture. The D-FIRE dataset, which includes several kinds of wildfire and non-wildfire images, was employed in the study. Low-resolution images, dataset imbalance, and the necessity for real-time applicability are some of the main challenges. These problems were resolved by enriching the dataset using data augmentation techniques and optimizing the VGG16 model for binary classification. The model produced a low false negative rate, which is essential for reducing unexplored fires, despite dataset boundaries. In order to help authorities execute fast responses, this work shows that deep learning models such as VGG16 can offer a reliable, automated approach for early wildfire recognition. For the purpose of reducing the impact of wildfires, our future work will concentrate on connecting to systems with real-time surveillance networks and enlarging the dataset to cover more varied fire situations.
>
---
#### [new 199] Eye-See-You: Reverse Pass-Through VR and Head Avatars
- **分类: cs.CV**

- **简介: 该论文提出RevAvatar框架，利用AI生成高保真2D/3D头像，解决VR头显遮挡用户面部导致的社交障碍。通过生成模型从部分可见区域重建面部，并构建VR-Face数据集（20万样本），促进虚实交互，提升VR社交体验。属于AI增强VR交互技术，解决视觉沟通问题。**

- **链接: [http://arxiv.org/pdf/2505.18869v1](http://arxiv.org/pdf/2505.18869v1)**

> **作者:** Ankan Dash; Jingyi Gu; Guiling Wang; Chen Chen
>
> **备注:** 34th International Joint Conference on Artificial Intelligence, IJCAI 2025
>
> **摘要:** Virtual Reality (VR) headsets, while integral to the evolving digital ecosystem, present a critical challenge: the occlusion of users' eyes and portions of their faces, which hinders visual communication and may contribute to social isolation. To address this, we introduce RevAvatar, an innovative framework that leverages AI methodologies to enable reverse pass-through technology, fundamentally transforming VR headset design and interaction paradigms. RevAvatar integrates state-of-the-art generative models and multimodal AI techniques to reconstruct high-fidelity 2D facial images and generate accurate 3D head avatars from partially observed eye and lower-face regions. This framework represents a significant advancement in AI4Tech by enabling seamless interaction between virtual and physical environments, fostering immersive experiences such as VR meetings and social engagements. Additionally, we present VR-Face, a novel dataset comprising 200,000 samples designed to emulate diverse VR-specific conditions, including occlusions, lighting variations, and distortions. By addressing fundamental limitations in current VR systems, RevAvatar exemplifies the transformative synergy between AI and next-generation technologies, offering a robust platform for enhancing human connection and interaction in virtual environments.
>
---
#### [new 200] Taming Diffusion for Dataset Distillation with High Representativeness
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于数据集蒸馏任务，旨在解决扩散模型生成数据分布偏差及代表性不足的问题。提出D^3HR框架，通过DDIM反演将数据集潜在空间映射到高正态分布域，并优化采样方案，生成高代表性的紧凑数据集，提升模型精度。**

- **链接: [http://arxiv.org/pdf/2505.18399v1](http://arxiv.org/pdf/2505.18399v1)**

> **作者:** Lin Zhao; Yushu Wu; Xinru Jiang; Jianyang Gu; Yanzhi Wang; Xiaolin Xu; Pu Zhao; Xue Lin
>
> **备注:** The paper is accepted by ICML 2025
>
> **摘要:** Recent deep learning models demand larger datasets, driving the need for dataset distillation to create compact, cost-efficient datasets while maintaining performance. Due to the powerful image generation capability of diffusion, it has been introduced to this field for generating distilled images. In this paper, we systematically investigate issues present in current diffusion-based dataset distillation methods, including inaccurate distribution matching, distribution deviation with random noise, and separate sampling. Building on this, we propose D^3HR, a novel diffusion-based framework to generate distilled datasets with high representativeness. Specifically, we adopt DDIM inversion to map the latents of the full dataset from a low-normality latent domain to a high-normality Gaussian domain, preserving information and ensuring structural consistency to generate representative latents for the distilled dataset. Furthermore, we propose an efficient sampling scheme to better align the representative latents with the high-normality Gaussian distribution. Our comprehensive experiments demonstrate that D^3HR can achieve higher accuracy across different model architectures compared with state-of-the-art baselines in dataset distillation. Source code: https://github.com/lin-zhao-resoLve/D3HR.
>
---
#### [new 201] LLM-Guided Taxonomy and Hierarchical Uncertainty for 3D Point CLoud Active Learning
- **分类: cs.CV**

- **简介: 该论文属于3D点云语义分割的主动学习任务，旨在解决传统方法忽略语义层级结构导致标注效率低的问题。提出LLM生成多级语义分类，并通过层次化不确定性传播机制选择样本，在低标注预算下提升分割性能。**

- **链接: [http://arxiv.org/pdf/2505.18924v1](http://arxiv.org/pdf/2505.18924v1)**

> **作者:** Chenxi Li; Nuo Chen; Fengyun Tan; Yantong Chen; Bochun Yuan; Tianrui Li; Chongshou Li
>
> **摘要:** We present a novel active learning framework for 3D point cloud semantic segmentation that, for the first time, integrates large language models (LLMs) to construct hierarchical label structures and guide uncertainty-based sample selection. Unlike prior methods that treat labels as flat and independent, our approach leverages LLM prompting to automatically generate multi-level semantic taxonomies and introduces a recursive uncertainty projection mechanism that propagates uncertainty across hierarchy levels. This enables spatially diverse, label-aware point selection that respects the inherent semantic structure of 3D scenes. Experiments on S3DIS and ScanNet v2 show that our method achieves up to 4% mIoU improvement under extremely low annotation budgets (e.g., 0.02%), substantially outperforming existing baselines. Our results highlight the untapped potential of LLMs as knowledge priors in 3D vision and establish hierarchical uncertainty modeling as a powerful paradigm for efficient point cloud annotation.
>
---
#### [new 202] OB3D: A New Dataset for Benchmarking Omnidirectional 3D Reconstruction Using Blender
- **分类: cs.CV**

- **简介: 该论文提出OB3D数据集，用于全向3D重建的基准测试。针对现有方法在处理360度图像几何畸变（如等距投影极区失真）时精度不足及缺乏针对性数据的问题，该数据集通过Blender生成复杂场景，提供全向RGB图像、精确相机参数及像素对齐的深度、法线图等地面真实数据，以推动方法优化。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20126v1](http://arxiv.org/pdf/2505.20126v1)**

> **作者:** Shintaro Ito; Natsuki Takama; Toshiki Watanabe; Koichi Ito; Hwann-Tzong Chen; Takafumi Aoki
>
> **摘要:** Recent advancements in radiance field rendering, exemplified by Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have significantly progressed 3D modeling and reconstruction. The use of multiple 360-degree omnidirectional images for these tasks is increasingly favored due to advantages in data acquisition and comprehensive scene capture. However, the inherent geometric distortions in common omnidirectional representations, such as equirectangular projection (particularly severe in polar regions and varying with latitude), pose substantial challenges to achieving high-fidelity 3D reconstructions. Current datasets, while valuable, often lack the specific focus, scene composition, and ground truth granularity required to systematically benchmark and drive progress in overcoming these omnidirectional-specific challenges. To address this critical gap, we introduce Omnidirectional Blender 3D (OB3D), a new synthetic dataset curated for advancing 3D reconstruction from multiple omnidirectional images. OB3D features diverse and complex 3D scenes generated from Blender 3D projects, with a deliberate emphasis on challenging scenarios. The dataset provides comprehensive ground truth, including omnidirectional RGB images, precise omnidirectional camera parameters, and pixel-aligned equirectangular maps for depth and normals, alongside evaluation metrics. By offering a controlled yet challenging environment, OB3Daims to facilitate the rigorous evaluation of existing methods and prompt the development of new techniques to enhance the accuracy and reliability of 3D reconstruction from omnidirectional images.
>
---
#### [new 203] The Role of Video Generation in Enhancing Data-Limited Action Understanding
- **分类: cs.CV**

- **简介: 该论文针对数据有限的视频动作理解任务，提出通过文本到视频扩散模型生成无限规模标注数据，并设计信息增强和不确定性标签平滑策略优化训练，提升零样本动作识别效果，在多个数据集上达最优性能。**

- **链接: [http://arxiv.org/pdf/2505.19495v1](http://arxiv.org/pdf/2505.19495v1)**

> **作者:** Wei Li; Dezhao Luo; Dongbao Yang; Zhenhang Li; Weiping Wang; Yu Zhou
>
> **备注:** IJCAI2025
>
> **摘要:** Video action understanding tasks in real-world scenarios always suffer data limitations. In this paper, we address the data-limited action understanding problem by bridging data scarcity. We propose a novel method that employs a text-to-video diffusion transformer to generate annotated data for model training. This paradigm enables the generation of realistic annotated data on an infinite scale without human intervention. We proposed the information enhancement strategy and the uncertainty-based label smoothing tailored to generate sample training. Through quantitative and qualitative analysis, we observed that real samples generally contain a richer level of information than generated samples. Based on this observation, the information enhancement strategy is proposed to enhance the informative content of the generated samples from two aspects: the environments and the characters. Furthermore, we observed that some low-quality generated samples might negatively affect model training. To address this, we devised the uncertainty-based label smoothing strategy to increase the smoothing of these samples, thus reducing their impact. We demonstrate the effectiveness of the proposed method on four datasets across five tasks and achieve state-of-the-art performance for zero-shot action recognition.
>
---
#### [new 204] ReDDiT: Rehashing Noise for Discrete Visual Generation
- **分类: cs.CV**

- **简介: 论文提出ReDDiT框架，改进离散扩散模型的视觉生成任务。针对现有模型因噪声设计和采样策略导致性能不足的问题，通过随机多索引腐败扩展吸收态并设计逆向采样器，提升生成多样性和质量。实验显示其显著优于基线（gFID从6.18降至1.61），效率更高且质量媲美连续模型。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19656v1](http://arxiv.org/pdf/2505.19656v1)**

> **作者:** Tianren Ma; Xiaosong Zhang; Boyu Yang; Junlan Feng; Qixiang Ye
>
> **备注:** Preprint, under development
>
> **摘要:** Discrete diffusion models are gaining traction in the visual generative area for their efficiency and compatibility. However, the pioneered attempts still fall behind the continuous counterparts, which we attribute to the noise (absorbing state) design and sampling heuristics. In this study, we propose the rehashing noise framework for discrete diffusion transformer, termed ReDDiT, to extend absorbing states and improve expressive capacity of discrete diffusion models. ReDDiT enriches the potential paths that latent variables can traverse during training with randomized multi-index corruption. The derived rehash sampler, which reverses the randomized absorbing paths, guarantees the diversity and low discrepancy of the generation process. These reformulations lead to more consistent and competitive generation quality, mitigating the need for heavily tuned randomness. Experiments show that ReDDiT significantly outperforms the baseline (reducing gFID from 6.18 to 1.61) and is on par with the continuous counterparts with higher efficiency.
>
---
#### [new 205] DiSa: Directional Saliency-Aware Prompt Learning for Generalizable Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型适配任务，解决现有方法过拟合导致泛化性差的问题。提出DiSa框架，通过交叉互动正则化（CIR）与方向正则化策略，引导模型关注关键区域并保持特征方向一致性，提升跨模态对齐及在新类别、领域和少样本场景的泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.19373v1](http://arxiv.org/pdf/2505.19373v1)**

> **作者:** Niloufar Alipour Talemi; Hossein Kashiani; Hossein R. Nowdeh; Fatemeh Afghah
>
> **备注:** Accepted at the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2025)
>
> **摘要:** Prompt learning has emerged as a powerful paradigm for adapting vision-language models such as CLIP to downstream tasks. However, existing methods often overfit to seen data, leading to significant performance degradation when generalizing to novel classes or unseen domains. To address this limitation, we propose DiSa, a Directional Saliency-Aware Prompt Learning framework that integrates two complementary regularization strategies to enhance generalization. First, our Cross-Interactive Regularization (CIR) fosters cross-modal alignment by enabling cooperative learning between prompted and frozen encoders. Within CIR, a saliency-aware masking strategy guides the image encoder to prioritize semantically critical image regions, reducing reliance on less informative patches. Second, we introduce a directional regularization strategy that aligns visual embeddings with class-wise prototype features in a directional manner to prioritize consistency in feature orientation over strict proximity. This approach ensures robust generalization by leveraging stable prototype directions derived from class-mean statistics. Extensive evaluations on 11 diverse image classification benchmarks demonstrate that DiSa consistently outperforms state-of-the-art prompt learning methods across various settings, including base-to-novel generalization, cross-dataset transfer, domain generalization, and few-shot learning.
>
---
#### [new 206] TDVE-Assessor: Benchmarking and Evaluating the Quality of Text-Driven Video Editing with LMMs
- **分类: cs.CV**

- **简介: 该论文属于文本驱动视频编辑质量评估任务，解决现有方法无法精准评估编辑质量的问题。工作包括构建含3857个视频的TDVE-DB基准数据集，评估12种编辑模型，并提出TDVE-Assessor模型，整合时空特征与大语言模型提升评估效果，性能超现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19535v1](http://arxiv.org/pdf/2505.19535v1)**

> **作者:** Juntong Wang; Jiarui Wang; Huiyu Duan; Guangtao Zhai; Xiongkuo Min
>
> **备注:** 25 pages, 14 figures, 8 tables
>
> **摘要:** Text-driven video editing is rapidly advancing, yet its rigorous evaluation remains challenging due to the absence of dedicated video quality assessment (VQA) models capable of discerning the nuances of editing quality. To address this critical gap, we introduce TDVE-DB, a large-scale benchmark dataset for text-driven video editing. TDVE-DB consists of 3,857 edited videos generated from 12 diverse models across 8 editing categories, and is annotated with 173,565 human subjective ratings along three crucial dimensions, i.e., edited video quality, editing alignment, and structural consistency. Based on TDVE-DB, we first conduct a comprehensive evaluation for the 12 state-of-the-art editing models revealing the strengths and weaknesses of current video techniques, and then benchmark existing VQA methods in the context of text-driven video editing evaluation. Building on these insights, we propose TDVE-Assessor, a novel VQA model specifically designed for text-driven video editing assessment. TDVE-Assessor integrates both spatial and temporal video features into a large language model (LLM) for rich contextual understanding to provide comprehensive quality assessment. Extensive experiments demonstrate that TDVE-Assessor substantially outperforms existing VQA models on TDVE-DB across all three evaluation dimensions, setting a new state-of-the-art. Both TDVE-DB and TDVE-Assessor will be released upon the publication.
>
---
#### [new 207] MMP-2K: A Benchmark Multi-Labeled Macro Photography Image Quality Assessment Database
- **分类: cs.CV**

- **简介: 该论文构建了多标签微距摄影图像质量评估数据库MMP-2K，解决现有MPIQA数据不足的问题。通过筛选1.57万张图片并收集2000张图像的质量评分及失真详情，构建基准数据库，验证了现有通用IQA指标在微距图像上的局限性。**

- **链接: [http://arxiv.org/pdf/2505.19065v1](http://arxiv.org/pdf/2505.19065v1)**

> **作者:** Jiashuo Chang; Zhengyi Li; Jianxun Lou; Zhen Qiu; Hanhe Lin
>
> **备注:** Accepted to the IEEE International Conference on Image Processing, IEEE ICIP 2025
>
> **摘要:** Macro photography (MP) is a specialized field of photography that captures objects at an extremely close range, revealing tiny details. Although an accurate macro photography image quality assessment (MPIQA) metric can benefit macro photograph capturing, which is vital in some domains such as scientific research and medical applications, the lack of MPIQA data limits the development of MPIQA metrics. To address this limitation, we conducted a large-scale MPIQA study. Specifically, to ensure diversity both in content and quality, we sampled 2,000 MP images from 15,700 MP images, collected from three public image websites. For each MP image, 17 (out of 21 after outlier removal) quality ratings and a detailed quality report of distortion magnitudes, types, and positions are gathered by a lab study. The images, quality ratings, and quality reports form our novel multi-labeled MPIQA database, MMP-2k. Experimental results showed that the state-of-the-art generic IQA metrics underperform on MP images. The database and supplementary materials are available at https://github.com/Future-IQA/MMP-2k.
>
---
#### [new 208] CDPDNet: Integrating Text Guidance with Hybrid Vision Encoders for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决多数据集训练中因标注不全导致的模型学习不足及视觉模型难以捕捉复杂解剖关系的问题。提出CDPDNet，结合CNN与DINOv2提取多尺度视觉特征，通过CLIP文本嵌入和任务特定提示模块增强器官间关系建模及标注不全适应性。**

- **链接: [http://arxiv.org/pdf/2505.18958v1](http://arxiv.org/pdf/2505.18958v1)**

> **作者:** Jiong Wu; Yang Xing; Boxiao Yu; Wei Shao; Kuang Gong
>
> **摘要:** Most publicly available medical segmentation datasets are only partially labeled, with annotations provided for a subset of anatomical structures. When multiple datasets are combined for training, this incomplete annotation poses challenges, as it limits the model's ability to learn shared anatomical representations among datasets. Furthermore, vision-only frameworks often fail to capture complex anatomical relationships and task-specific distinctions, leading to reduced segmentation accuracy and poor generalizability to unseen datasets. In this study, we proposed a novel CLIP-DINO Prompt-Driven Segmentation Network (CDPDNet), which combined a self-supervised vision transformer with CLIP-based text embedding and introduced task-specific text prompts to tackle these challenges. Specifically, the framework was constructed upon a convolutional neural network (CNN) and incorporated DINOv2 to extract both fine-grained and global visual features, which were then fused using a multi-head cross-attention module to overcome the limited long-range modeling capability of CNNs. In addition, CLIP-derived text embeddings were projected into the visual space to help model complex relationships among organs and tumors. To further address the partial label challenge and enhance inter-task discriminative capability, a Text-based Task Prompt Generation (TTPG) module that generated task-specific prompts was designed to guide the segmentation. Extensive experiments on multiple medical imaging datasets demonstrated that CDPDNet consistently outperformed existing state-of-the-art segmentation methods. Code and pretrained model are available at: https://github.com/wujiong-hub/CDPDNet.git.
>
---
#### [new 209] Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视觉生成任务，旨在解决传统自回归模型在图像生成中无法有效利用全局上下文的问题。提出分层掩码自回归模型Hi-MAR，通过低分辨率标记作为枢轴分阶段预测：首阶段预测低分辨率标记构建全局结构，次阶段利用枢轴增强密集图像标记生成，并设计扩散Transformer强化全局上下文，实现更优生成效果与更低计算成本。**

- **链接: [http://arxiv.org/pdf/2505.20288v1](http://arxiv.org/pdf/2505.20288v1)**

> **作者:** Guangting Zheng; Yehao Li; Yingwei Pan; Jiajun Deng; Ting Yao; Yanyong Zhang; Tao Mei
>
> **备注:** ICML 2025. Source code is available at https://github.com/HiDream-ai/himar
>
> **摘要:** Autoregressive models have emerged as a powerful generative paradigm for visual generation. The current de-facto standard of next token prediction commonly operates over a single-scale sequence of dense image tokens, and is incapable of utilizing global context especially for early tokens prediction. In this paper, we introduce a new autoregressive design to model a hierarchy from a few low-resolution image tokens to the typical dense image tokens, and delve into a thorough hierarchical dependency across multi-scale image tokens. Technically, we present a Hierarchical Masked Autoregressive models (Hi-MAR) that pivot on low-resolution image tokens to trigger hierarchical autoregressive modeling in a multi-phase manner. Hi-MAR learns to predict a few image tokens in low resolution, functioning as intermediary pivots to reflect global structure, in the first phase. Such pivots act as the additional guidance to strengthen the next autoregressive modeling phase by shaping global structural awareness of typical dense image tokens. A new Diffusion Transformer head is further devised to amplify the global context among all tokens for mask token prediction. Extensive evaluations on both class-conditional and text-to-image generation tasks demonstrate that Hi-MAR outperforms typical AR baselines, while requiring fewer computational costs. Code is available at https://github.com/HiDream-ai/himar.
>
---
#### [new 210] Erasing Concepts, Steering Generations: A Comprehensive Survey of Concept Suppression
- **分类: cs.CV**

- **简介: 该论文综述文本到图像模型的概念擦除技术，解决生成敏感/有害内容的伦理与安全问题。系统分类现有方法，按干预层级、优化结构、语义范围分析，讨论评估标准与挑战，提出未来方向，推动负责任的生成AI发展。**

- **链接: [http://arxiv.org/pdf/2505.19398v1](http://arxiv.org/pdf/2505.19398v1)**

> **作者:** Yiwei Xie; Ping Liu; Zheng Zhang
>
> **摘要:** Text-to-Image (T2I) models have demonstrated impressive capabilities in generating high-quality and diverse visual content from natural language prompts. However, uncontrolled reproduction of sensitive, copyrighted, or harmful imagery poses serious ethical, legal, and safety challenges. To address these concerns, the concept erasure paradigm has emerged as a promising direction, enabling the selective removal of specific semantic concepts from generative models while preserving their overall utility. This survey provides a comprehensive overview and in-depth synthesis of concept erasure techniques in T2I diffusion models. We systematically categorize existing approaches along three key dimensions: intervention level, which identifies specific model components targeted for concept removal; optimization structure, referring to the algorithmic strategies employed to achieve suppression; and semantic scope, concerning the complexity and nature of the concepts addressed. This multi-dimensional taxonomy enables clear, structured comparisons across diverse methodologies, highlighting fundamental trade-offs between erasure specificity, generalization, and computational complexity. We further discuss current evaluation benchmarks, standardized metrics, and practical datasets, emphasizing gaps that limit comprehensive assessment, particularly regarding robustness and practical effectiveness. Finally, we outline major challenges and promising future directions, including disentanglement of concept representations, adaptive and incremental erasure strategies, adversarial robustness, and new generative architectures. This survey aims to guide researchers toward safer, more ethically aligned generative models, providing foundational knowledge and actionable recommendations to advance responsible development in generative AI.
>
---
#### [new 211] Structured Initialization for Vision Transformers
- **分类: cs.CV**

- **简介: 该论文提出一种结构化初始化方法，将CNN的归纳偏置融入Vision Transformer（ViT），解决其在小数据集上泛化不足的问题。通过设计基于随机脉冲滤波器的初始化策略，提升ViT在小/中数据集性能，同时保持大数据集表现，并扩展至其他Transformer架构。**

- **链接: [http://arxiv.org/pdf/2505.19985v1](http://arxiv.org/pdf/2505.19985v1)**

> **作者:** Jianqiao Zheng; Xueqian Li; Hemanth Saratchandran; Simon Lucey
>
> **摘要:** Convolutional Neural Networks (CNNs) inherently encode strong inductive biases, enabling effective generalization on small-scale datasets. In this paper, we propose integrating this inductive bias into ViTs, not through an architectural intervention but solely through initialization. The motivation here is to have a ViT that can enjoy strong CNN-like performance when data assets are small, but can still scale to ViT-like performance as the data expands. Our approach is motivated by our empirical results that random impulse filters can achieve commensurate performance to learned filters within a CNN. We improve upon current ViT initialization strategies, which typically rely on empirical heuristics such as using attention weights from pretrained models or focusing on the distribution of attention weights without enforcing structures. Empirical results demonstrate that our method significantly outperforms standard ViT initialization across numerous small and medium-scale benchmarks, including Food-101, CIFAR-10, CIFAR-100, STL-10, Flowers, and Pets, while maintaining comparative performance on large-scale datasets such as ImageNet-1K. Moreover, our initialization strategy can be easily integrated into various transformer-based architectures such as Swin Transformer and MLP-Mixer with consistent improvements in performance.
>
---
#### [new 212] MEBench: A Novel Benchmark for Understanding Mutual Exclusivity Bias in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出MEBench，用于评估视觉语言模型（VLMs）在互斥偏移（ME）及空间推理上的表现。针对传统ME任务缺乏空间挑战与真实场景的问题，构建结合空间推理的基准，设计新评估指标，并开发灵活可扩展的数据生成pipeline。**

- **链接: [http://arxiv.org/pdf/2505.20122v1](http://arxiv.org/pdf/2505.20122v1)**

> **作者:** Anh Thai; Stefan Stojanov; Zixuan Huang; Bikram Boote; James M. Rehg
>
> **摘要:** This paper introduces MEBench, a novel benchmark for evaluating mutual exclusivity (ME) bias, a cognitive phenomenon observed in children during word learning. Unlike traditional ME tasks, MEBench further incorporates spatial reasoning to create more challenging and realistic evaluation settings. We assess the performance of state-of-the-art vision-language models (VLMs) on this benchmark using novel evaluation metrics that capture key aspects of ME-based reasoning. To facilitate controlled experimentation, we also present a flexible and scalable data generation pipeline that supports the construction of diverse annotated scenes.
>
---
#### [new 213] Inference Compute-Optimal Video Vision Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究在固定推理计算预算下，优化视频视觉语言模型中语言模型大小、帧数及每帧视觉标记数量的资源配置问题。通过大规模实验与参数建模分析任务性能与数据量的影响，确定计算最优前沿并提供建议。**

- **链接: [http://arxiv.org/pdf/2505.18855v1](http://arxiv.org/pdf/2505.18855v1)**

> **作者:** Peiqi Wang; ShengYun Peng; Xuewen Zhang; Hanchao Yu; Yibo Yang; Lifu Huang; Fujun Liu; Qifan Wang
>
> **备注:** Annual Meeting of the Association for Computational Linguistics (ACL), 2025
>
> **摘要:** This work investigates the optimal allocation of inference compute across three key scaling factors in video vision language models: language model size, frame count, and the number of visual tokens per frame. While prior works typically focuses on optimizing model efficiency or improving performance without considering resource constraints, we instead identify optimal model configuration under fixed inference compute budgets. We conduct large-scale training sweeps and careful parametric modeling of task performance to identify the inference compute-optimal frontier. Our experiments reveal how task performance depends on scaling factors and finetuning data size, as well as how changes in data size shift the compute-optimal frontier. These findings translate to practical tips for selecting these scaling factors.
>
---
#### [new 214] FHGS: Feature-Homogenized Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出FHGS框架，解决3DGS中高斯体素各向异性渲染与语义特征各向同性需求的矛盾。通过构建通用特征融合架构、非可微分各向同性分布机制及电势场双驱动优化策略，实现跨视角语义一致性与实时渲染的平衡。**

- **链接: [http://arxiv.org/pdf/2505.19154v1](http://arxiv.org/pdf/2505.19154v1)**

> **作者:** Q. G. Duan; Benyun Zhao; Mingqiao Han Yijun Huang; Ben M. Chen
>
> **摘要:** Scene understanding based on 3D Gaussian Splatting (3DGS) has recently achieved notable advances. Although 3DGS related methods have efficient rendering capabilities, they fail to address the inherent contradiction between the anisotropic color representation of gaussian primitives and the isotropic requirements of semantic features, leading to insufficient cross-view feature consistency. To overcome the limitation, we proposes $\textit{FHGS}$ (Feature-Homogenized Gaussian Splatting), a novel 3D feature fusion framework inspired by physical models, which can achieve high-precision mapping of arbitrary 2D features from pre-trained models to 3D scenes while preserving the real-time rendering efficiency of 3DGS. Specifically, our $\textit{FHGS}$ introduces the following innovations: Firstly, a universal feature fusion architecture is proposed, enabling robust embedding of large-scale pre-trained models' semantic features (e.g., SAM, CLIP) into sparse 3D structures. Secondly, a non-differentiable feature fusion mechanism is introduced, which enables semantic features to exhibit viewpoint independent isotropic distributions. This fundamentally balances the anisotropic rendering of gaussian primitives and the isotropic expression of features; Thirdly, a dual-driven optimization strategy inspired by electric potential fields is proposed, which combines external supervision from semantic feature fields with internal primitive clustering guidance. This mechanism enables synergistic optimization of global semantic alignment and local structural consistency. More interactive results can be accessed on: https://fhgs.cuastro.org/.
>
---
#### [new 215] Structure Disruption: Subverting Malicious Diffusion-Based Inpainting via Self-Attention Query Perturbation
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 该论文属图像编辑安全任务，旨在防御扩散模型对敏感区域的恶意修复。针对现有全局扰动方法在mask引导编辑中失效问题，提出Structure Disruption Attack（SDA），通过干扰自注意力机制的初始查询，破坏轮廓生成过程，阻止结构生成。实验显示其防护效果达SOTA且鲁棒性强。**

- **链接: [http://arxiv.org/pdf/2505.19425v1](http://arxiv.org/pdf/2505.19425v1)**

> **作者:** Yuhao He; Jinyu Tian; Haiwei Wu; Jianqing Li
>
> **摘要:** The rapid advancement of diffusion models has enhanced their image inpainting and editing capabilities but also introduced significant societal risks. Adversaries can exploit user images from social media to generate misleading or harmful content. While adversarial perturbations can disrupt inpainting, global perturbation-based methods fail in mask-guided editing tasks due to spatial constraints. To address these challenges, we propose Structure Disruption Attack (SDA), a powerful protection framework for safeguarding sensitive image regions against inpainting-based editing. Building upon the contour-focused nature of self-attention mechanisms of diffusion models, SDA optimizes perturbations by disrupting queries in self-attention during the initial denoising step to destroy the contour generation process. This targeted interference directly disrupts the structural generation capability of diffusion models, effectively preventing them from producing coherent images. We validate our motivation through visualization techniques and extensive experiments on public datasets, demonstrating that SDA achieves state-of-the-art (SOTA) protection performance while maintaining strong robustness.
>
---
#### [new 216] CA3D: Convolutional-Attentional 3D Nets for Efficient Video Activity Recognition on the Edge
- **分类: cs.CV**

- **简介: 该论文提出CA3D模型，针对边缘设备视频活动识别任务，通过结合卷积与线性注意力机制及新型量化技术，解决传统模型计算成本高、效率低的问题，在保持准确率的同时降低计算需求，适用于智能家居等对效率和隐私要求高的场景。**

- **链接: [http://arxiv.org/pdf/2505.19928v1](http://arxiv.org/pdf/2505.19928v1)**

> **作者:** Gabriele Lagani; Fabrizio Falchi; Claudio Gennaro; Giuseppe Amato
>
> **摘要:** In this paper, we introduce a deep learning solution for video activity recognition that leverages an innovative combination of convolutional layers with a linear-complexity attention mechanism. Moreover, we introduce a novel quantization mechanism to further improve the efficiency of our model during both training and inference. Our model maintains a reduced computational cost, while preserving robust learning and generalization capabilities. Our approach addresses the issues related to the high computing requirements of current models, with the goal of achieving competitive accuracy on consumer and edge devices, enabling smart home and smart healthcare applications where efficiency and privacy issues are of concern. We experimentally validate our model on different established and publicly available video activity recognition benchmarks, improving accuracy over alternative models at a competitive computing cost.
>
---
#### [new 217] A Smart Healthcare System for Monkeypox Skin Lesion Detection and Tracking
- **分类: cs.CV; cs.AI; cs.ET; cs.LG**

- **简介: 该论文提出ITMAINN系统，属于猴痘皮肤损伤检测与追踪任务。针对全球猴痘疫情下诊断资源不足问题，开发基于深度学习的AI模型（如MobileViT、ResNetViT），实现高精度分类；部署移动端应用进行检测与症状跟踪，并构建实时监控仪表盘辅助公共卫生决策，提升诊断效率与疫情响应能力。**

- **链接: [http://arxiv.org/pdf/2505.19023v1](http://arxiv.org/pdf/2505.19023v1)**

> **作者:** Huda Alghoraibi; Nuha Alqurashi; Sarah Alotaibi; Renad Alkhudaydi; Bdoor Aldajani; Lubna Alqurashi; Jood Batweel; Maha A. Thafar
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** Monkeypox is a viral disease characterized by distinctive skin lesions and has been reported in many countries. The recent global outbreak has emphasized the urgent need for scalable, accessible, and accurate diagnostic solutions to support public health responses. In this study, we developed ITMAINN, an intelligent, AI-driven healthcare system specifically designed to detect Monkeypox from skin lesion images using advanced deep learning techniques. Our system consists of three main components. First, we trained and evaluated several pretrained models using transfer learning on publicly available skin lesion datasets to identify the most effective models. For binary classification (Monkeypox vs. non-Monkeypox), the Vision Transformer, MobileViT, Transformer-in-Transformer, and VGG16 achieved the highest performance, each with an accuracy and F1-score of 97.8%. For multiclass classification, which contains images of patients with Monkeypox and five other classes (chickenpox, measles, hand-foot-mouth disease, cowpox, and healthy), ResNetViT and ViT Hybrid models achieved 92% accuracy, with F1 scores of 92.24% and 92.19%, respectively. The best-performing and most lightweight model, MobileViT, was deployed within the mobile application. The second component is a cross-platform smartphone application that enables users to detect Monkeypox through image analysis, track symptoms, and receive recommendations for nearby healthcare centers based on their location. The third component is a real-time monitoring dashboard designed for health authorities to support them in tracking cases, analyzing symptom trends, guiding public health interventions, and taking proactive measures. This system is fundamental in developing responsive healthcare infrastructure within smart cities. Our solution, ITMAINN, is part of revolutionizing public health management.
>
---
#### [new 218] BiomechGPT: Towards a Biomechanically Fluent Multimodal Foundation Model for Clinically Relevant Motion Tasks
- **分类: cs.CV**

- **简介: 该论文提出BiomechGPT，一种多模态生物力学-语言模型，解决临床运动分析中下游任务（如活动识别、损伤诊断）自动化问题。针对传统方法需为各任务定制代码且无法利用运动共性的问题，作者收集近500人数据，构建运动问答数据集，训练模型并验证其在临床评估任务中的高性能。**

- **链接: [http://arxiv.org/pdf/2505.18465v1](http://arxiv.org/pdf/2505.18465v1)**

> **作者:** Ruize Yang; Ann Kennedy; R. James Cotton
>
> **摘要:** Advances in markerless motion capture are expanding access to biomechanical movement analysis, making it feasible to obtain high-quality movement data from outpatient clinics, inpatient hospitals, therapy, and even home. Expanding access to movement data in these diverse contexts makes the challenge of performing downstream analytics all the more acute. Creating separate bespoke analysis code for all the tasks end users might want is both intractable and does not take advantage of the common features of human movement underlying them all. Recent studies have shown that fine-tuning language models to accept tokenized movement as an additional modality enables successful descriptive captioning of movement. Here, we explore whether such a multimodal motion-language model can answer detailed, clinically meaningful questions about movement. We collected over 30 hours of biomechanics from nearly 500 participants, many with movement impairments from a variety of etiologies, performing a range of movements used in clinical outcomes assessments. After tokenizing these movement trajectories, we created a multimodal dataset of motion-related questions and answers spanning a range of tasks. We developed BiomechGPT, a multimodal biomechanics-language model, on this dataset. Our results show that BiomechGPT demonstrates high performance across a range of tasks such as activity recognition, identifying movement impairments, diagnosis, scoring clinical outcomes, and measuring walking. BiomechGPT provides an important step towards a foundation model for rehabilitation movement data.
>
---
#### [new 219] Disentangled Human Body Representation Based on Unsupervised Semantic-Aware Learning
- **分类: cs.CV**

- **简介: 该论文提出一种无监督语义感知的解纠缠3D人体表示方法，旨在解决现有方法依赖手工约束且缺乏可控语义的问题。通过设计骨骼分组解纠缠策略、基于模板的残差学习及部分感知解码器，在无监督框架下实现人体形状与姿态的精准可控重建，适用于姿态迁移等应用。**

- **链接: [http://arxiv.org/pdf/2505.19049v1](http://arxiv.org/pdf/2505.19049v1)**

> **作者:** Lu Wang; Xishuai Peng; S. Kevin Zhou
>
> **备注:** 8 pages
>
> **摘要:** In recent years, more and more attention has been paid to the learning of 3D human representation. However, the complexity of lots of hand-defined human body constraints and the absence of supervision data limit that the existing works controllably and accurately represent the human body in views of semantics and representation ability. In this paper, we propose a human body representation with controllable fine-grained semantics and high precison of reconstruction in an unsupervised learning framework. In particularly, we design a whole-aware skeleton-grouped disentangle strategy to learn a correspondence between geometric semantical measurement of body and latent codes, which facilitates the control of shape and posture of human body by modifying latent coding paramerers. With the help of skeleton-grouped whole-aware encoder and unsupervised disentanglement losses, our representation model is learned by an unsupervised manner. Besides, a based-template residual learning scheme is injected into the encoder to ease of learning human body latent parameter in complicated body shape and pose spaces. Because of the geometrically meaningful latent codes, it can be used in a wide range of applications, from human body pose transfer to bilinear latent code interpolation. Further more, a part-aware decoder is utlized to promote the learning of controllable fine-grained semantics. The experimental results on public 3D human datasets show that the method has the ability of precise reconstruction.
>
---
#### [new 220] Geometry-guided Online 3D Video Synthesis with Multi-View Temporal Consistency
- **分类: cs.CV**

- **简介: 该论文属于在线3D视频合成任务，旨在解决传统方法计算成本高或选择性输入方法导致的多视角和时间不一致问题（如闪烁）。提出几何引导方法：通过时序颜色差分优化深度图，结合截断符号距离场（TSDF）积累深度信息，指导预训练网络融合多视角图像，实现高效、高质量且时空一致的视频合成。**

- **链接: [http://arxiv.org/pdf/2505.18932v1](http://arxiv.org/pdf/2505.18932v1)**

> **作者:** Hyunho Ha; Lei Xiao; Christian Richardt; Thu Nguyen-Phuoc; Changil Kim; Min H. Kim; Douglas Lanman; Numair Khan
>
> **备注:** Accepted by CVPR 2025. Project website: https://nkhan2.github.io/projects/geometry-guided-2025/index.html
>
> **摘要:** We introduce a novel geometry-guided online video view synthesis method with enhanced view and temporal consistency. Traditional approaches achieve high-quality synthesis from dense multi-view camera setups but require significant computational resources. In contrast, selective-input methods reduce this cost but often compromise quality, leading to multi-view and temporal inconsistencies such as flickering artifacts. Our method addresses this challenge to deliver efficient, high-quality novel-view synthesis with view and temporal consistency. The key innovation of our approach lies in using global geometry to guide an image-based rendering pipeline. To accomplish this, we progressively refine depth maps using color difference masks across time. These depth maps are then accumulated through truncated signed distance fields in the synthesized view's image space. This depth representation is view and temporally consistent, and is used to guide a pre-trained blending network that fuses multiple forward-rendered input-view images. Thus, the network is encouraged to output geometrically consistent synthesis results across multiple views and time. Our approach achieves consistent, high-quality video synthesis, while running efficiently in an online manner.
>
---
#### [new 221] A Joint Learning Framework with Feature Reconstruction and Prediction for Incomplete Satellite Image Time Series in Agricultural Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文针对农业语义分割中卫星图像时间序列（SITS）因云污染导致数据缺失的问题，提出联合学习框架，同步进行特征重建与预测。通过时间掩码模拟缺失，结合教师模型指导，选择性重建关键特征并抑制噪声，提升模型对不同缺失模式的泛化性，实验显示分割精度显著提升。**

- **链接: [http://arxiv.org/pdf/2505.19159v1](http://arxiv.org/pdf/2505.19159v1)**

> **作者:** Yuze Wang; Mariana Belgiu; Haiyang Wu; Dandan Zhong; Yangyang Cao; Chao Tao
>
> **摘要:** Satellite Image Time Series (SITS) is crucial for agricultural semantic segmentation. However, Cloud contamination introduces time gaps in SITS, disrupting temporal dependencies and causing feature shifts, leading to degraded performance of models trained on complete SITS. Existing methods typically address this by reconstructing the entire SITS before prediction or using data augmentation to simulate missing data. Yet, full reconstruction may introduce noise and redundancy, while the data-augmented model can only handle limited missing patterns, leading to poor generalization. We propose a joint learning framework with feature reconstruction and prediction to address incomplete SITS more effectively. During training, we simulate data-missing scenarios using temporal masks. The two tasks are guided by both ground-truth labels and the teacher model trained on complete SITS. The prediction task constrains the model from selectively reconstructing critical features from masked inputs that align with the teacher's temporal feature representations. It reduces unnecessary reconstruction and limits noise propagation. By integrating reconstructed features into the prediction task, the model avoids learning shortcuts and maintains its ability to handle varied missing patterns and complete SITS. Experiments on SITS from Hunan Province, Western France, and Catalonia show that our method improves mean F1-scores by 6.93% in cropland extraction and 7.09% in crop classification over baselines. It also generalizes well across satellite sensors, including Sentinel-2 and PlanetScope, under varying temporal missing rates and model backbones.
>
---
#### [new 222] Mitigating Context Bias in Domain Adaptation for Object Detection using Mask Pooling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测领域自适应（DAOD）任务，旨在解决模型因训练中前景与背景关联导致的上下文偏差问题。通过因果分析指出池化操作为偏差来源，提出Mask Pooling方法，利用前景掩码分离前景背景池化，提升跨领域检测鲁棒性，并设计基准测试验证效果。**

- **链接: [http://arxiv.org/pdf/2505.18446v1](http://arxiv.org/pdf/2505.18446v1)**

> **作者:** Hojun Son; Asma Almutairi; Arpan Kusari
>
> **摘要:** Context bias refers to the association between the foreground objects and background during the object detection training process. Various methods have been proposed to minimize the context bias when applying the trained model to an unseen domain, known as domain adaptation for object detection (DAOD). But a principled approach to understand why the context bias occurs and how to remove it has been missing. In this work, we provide a causal view of the context bias, pointing towards the pooling operation in the convolution network architecture as the possible source of this bias. We present an alternative, Mask Pooling, which uses an additional input of foreground masks, to separate the pooling process in the respective foreground and background regions and show that this process leads the trained model to detect objects in a more robust manner under different domains. We also provide a benchmark designed to create an ultimate test for DAOD, using foregrounds in the presence of absolute random backgrounds, to analyze the robustness of the intended trained models. Through these experiments, we hope to provide a principled approach for minimizing context bias under domain shift.
>
---
#### [new 223] Training-free Stylized Text-to-Image Generation with Fast Inference
- **分类: cs.CV**

- **简介: 该论文属于风格化文本到图像生成任务，旨在解决现有方法依赖耗时的文本反转或微调的问题。提出无需训练的OmniPainter方法，通过提取参考图像的风格统计量及自注意力混合机制，引导预训练扩散模型生成符合风格分布的图像，实验证明其性能更优。**

- **链接: [http://arxiv.org/pdf/2505.19063v1](http://arxiv.org/pdf/2505.19063v1)**

> **作者:** Xin Ma; Yaohui Wang; Xinyuan Chen; Tien-Tsin Wong; Cunjian Chen
>
> **备注:** Project Page: https://maxin-cn.github.io/omnipainter_project
>
> **摘要:** Although diffusion models exhibit impressive generative capabilities, existing methods for stylized image generation based on these models often require textual inversion or fine-tuning with style images, which is time-consuming and limits the practical applicability of large-scale diffusion models. To address these challenges, we propose a novel stylized image generation method leveraging a pre-trained large-scale diffusion model without requiring fine-tuning or any additional optimization, termed as OmniPainter. Specifically, we exploit the self-consistency property of latent consistency models to extract the representative style statistics from reference style images to guide the stylization process. Additionally, we then introduce the norm mixture of self-attention, which enables the model to query the most relevant style patterns from these statistics for the intermediate output content features. This mechanism also ensures that the stylized results align closely with the distribution of the reference style images. Our qualitative and quantitative experimental results demonstrate that the proposed method outperforms state-of-the-art approaches.
>
---
#### [new 224] HaloGS: Loose Coupling of Compact Geometry and Gaussian Splats for 3D Scenes
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建与渲染任务，旨在解决现有方法在几何精度与渲染保真度间的效率权衡问题。提出HaloGS方法，通过松散耦合简洁三角形几何与高斯图元外观，实现紧凑模型下高精度几何重建与照片级渲染，适用于复杂室内外场景。**

- **链接: [http://arxiv.org/pdf/2505.20267v1](http://arxiv.org/pdf/2505.20267v1)**

> **作者:** Changjian Jiang; Kerui Ren; Linning Xu; Jiong Chen; Jiangmiao Pang; Yu Zhang; Bo Dai; Mulin Yu
>
> **摘要:** High fidelity 3D reconstruction and rendering hinge on capturing precise geometry while preserving photo realistic detail. Most existing methods either fuse these goals into a single cumbersome model or adopt hybrid schemes whose uniform primitives lead to a trade off between efficiency and fidelity. In this paper, we introduce HaloGS, a dual representation that loosely couples coarse triangles for geometry with Gaussian primitives for appearance, motivated by the lightweight classic geometry representations and their proven efficiency in real world applications. Our design yields a compact yet expressive model capable of photo realistic rendering across both indoor and outdoor environments, seamlessly adapting to varying levels of scene complexity. Experiments on multiple benchmark datasets demonstrate that our method yields both compact, accurate geometry and high fidelity renderings, especially in challenging scenarios where robust geometric structure make a clear difference.
>
---
#### [new 225] SATORI-R1: Incentivizing Multimodal Reasoning with Spatial Grounding and Verifiable Rewards
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉问答（VQA）任务中多模态推理的两个问题：推理链扩散视觉焦点及不可验证步骤增加计算成本，提出SATORI-R1方法。通过分解任务为图像描述、区域定位和答案预测三个可验证阶段，并引入12k标注数据集VQA-Verify，提升模型对关键区域的注意力及准确率（最高提升15.7%）。**

- **链接: [http://arxiv.org/pdf/2505.19094v1](http://arxiv.org/pdf/2505.19094v1)**

> **作者:** Chuming Shen; Wei Wei; Xiaoye Qu; Yu Cheng
>
> **备注:** Under review
>
> **摘要:** DeepSeek-R1 has demonstrated powerful reasoning capabilities in the text domain through stable reinforcement learning (RL). Recently, in the multimodal domain, works have begun to directly apply RL to generate R1-like free-form reasoning for Visual Question Answering (VQA) tasks. However, multimodal tasks share an intrinsically different nature from textual tasks, which heavily rely on the understanding of the input image to solve the problem. Therefore, such free-form reasoning faces two critical limitations in the VQA task: (1) Extended reasoning chains diffuse visual focus away from task-critical regions, degrading answer accuracy. (2) Unverifiable intermediate steps amplify policy-gradient variance and computational costs overhead. To address these issues, in this paper, we introduce SATORI ($\textbf{S}patially$ $\textbf{A}nchored$ $\textbf{T}ask$ $\textbf{O}ptimization$ with $\textbf{R}e\textbf{I}nforcement$ Learning), which decomposes VQA into three verifiable stages, including global image captioning, region localization, and answer prediction, each supplying explicit reward signals. Furthermore, we also introduce VQA-Verify, a 12k dataset annotated with answer-aligned captions and bounding-boxes to facilitate training. Experiments demonstrate consistent performance improvements across seven VQA benchmarks, achieving up to $15.7\%$ improvement in accuracy in accuracy compared to the R1-like baseline. Our analysis of the attention map confirms enhanced focus on critical regions, which brings improvements in accuracy. Our code is available at https://github.com/justairr/SATORI-R1.
>
---
#### [new 226] Kernel Space Diffusion Model for Efficient Remote Sensing Pansharpening
- **分类: cs.CV**

- **简介: 该论文针对遥感影像融合中全局信息捕获不足与扩散模型推理速度慢的问题，提出Kernel Space Diffusion Model（KSDiff）。通过在潜在空间利用扩散过程生成含全局上下文的卷积核，结合低秩张量生成器、统一因子生成器及结构感知注意力机制，并采用两阶段训练策略，提升融合质量同时加速推理，在多个数据集验证了有效性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18991v1](http://arxiv.org/pdf/2505.18991v1)**

> **作者:** Hancong Jin; Zihan Cao; Liangjian Deng
>
> **摘要:** Pansharpening is a fundamental task in remote sensing that integrates high-resolution panchromatic imagery (PAN) with low-resolution multispectral imagery (LRMS) to produce an enhanced image with both high spatial and spectral resolution. Despite significant progress in deep learning-based approaches, existing methods often fail to capture the global priors inherent in remote sensing data distributions. Diffusion-based models have recently emerged as promising solutions due to their powerful distribution mapping capabilities; however, they suffer from significant inference latency, which limits their practical applicability. In this work, we propose the Kernel Space Diffusion Model (KSDiff), a novel approach that leverages diffusion processes in a latent space to generate convolutional kernels enriched with global contextual information, thereby improving pansharpening quality while enabling faster inference. Specifically, KSDiff constructs these kernels through the integration of a low-rank core tensor generator and a unified factor generator, orchestrated by a structure-aware multi-head attention mechanism. We further introduce a two-stage training strategy tailored for pansharpening, enabling KSDiff to serve as a framework for enhancing existing pansharpening architectures. Experiments on three widely used datasets, including WorldView-3, GaoFen-2, and QuickBird, demonstrate the superior performance of KSDiff both qualitatively and quantitatively. Code will be released upon possible acceptance.
>
---
#### [new 227] DiSA: Diffusion Step Annealing in Autoregressive Image Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对自回归图像生成任务，解决扩散采样步骤过多导致的低效问题。提出DiSA方法：随生成进度逐步减少扩散步数（如初始50步渐减至后期5步），提升推理速度（MAR/Harmon快5-10倍，FlowAR/xAR快1.4-2.5倍），同时保持生成质量，且无需额外训练。**

- **链接: [http://arxiv.org/pdf/2505.20297v1](http://arxiv.org/pdf/2505.20297v1)**

> **作者:** Qinyu Zhao; Jaskirat Singh; Ming Xu; Akshay Asthana; Stephen Gould; Liang Zheng
>
> **备注:** Our code is available at https://github.com/Qinyu-Allen-Zhao/DiSA
>
> **摘要:** An increasing number of autoregressive models, such as MAR, FlowAR, xAR, and Harmon adopt diffusion sampling to improve the quality of image generation. However, this strategy leads to low inference efficiency, because it usually takes 50 to 100 steps for diffusion to sample a token. This paper explores how to effectively address this issue. Our key motivation is that as more tokens are generated during the autoregressive process, subsequent tokens follow more constrained distributions and are easier to sample. To intuitively explain, if a model has generated part of a dog, the remaining tokens must complete the dog and thus are more constrained. Empirical evidence supports our motivation: at later generation stages, the next tokens can be well predicted by a multilayer perceptron, exhibit low variance, and follow closer-to-straight-line denoising paths from noise to tokens. Based on our finding, we introduce diffusion step annealing (DiSA), a training-free method which gradually uses fewer diffusion steps as more tokens are generated, e.g., using 50 steps at the beginning and gradually decreasing to 5 steps at later stages. Because DiSA is derived from our finding specific to diffusion in autoregressive models, it is complementary to existing acceleration methods designed for diffusion alone. DiSA can be implemented in only a few lines of code on existing models, and albeit simple, achieves $5-10\times$ faster inference for MAR and Harmon and $1.4-2.5\times$ for FlowAR and xAR, while maintaining the generation quality.
>
---
#### [new 228] ToDRE: Visual Token Pruning via Diversity and Task Awareness for Efficient Large Vision-Language Models
- **分类: cs.CV**

- **简介: 论文提出ToDRE框架，针对大型视觉-语言模型（LVLMs）视觉输入计算开销大的问题。现有方法依赖标记重要性，而ToDRE通过视觉标记多样性和任务相关性两阶段剪枝，在视觉编码器后用贪心k-中心算法选取多样化标记，并在解码器中移除无关标记，减少90%视觉标记，加速2.6倍，性能保持95.1%。**

- **链接: [http://arxiv.org/pdf/2505.18757v1](http://arxiv.org/pdf/2505.18757v1)**

> **作者:** Duo Li; Zuhao Yang; Shijian Lu
>
> **备注:** 21 pages, 7 figures
>
> **摘要:** The representation of visual inputs of large vision-language models (LVLMs) usually involves substantially more tokens than that of textual inputs, leading to significant computational overhead. Several recent studies strive to mitigate this issue by either conducting token compression to prune redundant visual tokens or guiding them to bypass certain computational stages. While most existing work exploits token importance as the redundancy indicator, our study reveals that two largely neglected factors, namely, the diversity of retained visual tokens and their task relevance, often offer more robust criteria in token pruning. To this end, we design ToDRE, a two-stage and training-free token compression framework that achieves superior performance by pruning Tokens based on token Diversity and token-task RElevance. Instead of pruning redundant tokens, ToDRE introduces a greedy k-center algorithm to select and retain a small subset of diverse visual tokens after the vision encoder. Additionally, ToDRE addresses the "information migration" by further eliminating task-irrelevant visual tokens within the decoder of large language model (LLM). Extensive experiments show that ToDRE effectively reduces 90% of visual tokens after vision encoder and adaptively prunes all visual tokens within certain LLM's decoder layers, leading to a 2.6x speed-up in total inference time while maintaining 95.1% of model performance and excellent compatibility with efficient attention operators.
>
---
#### [new 229] Can Multimodal Large Language Models Understand Spatial Relations?
- **分类: cs.CV; cs.MM**

- **简介: 该论文评估多模态大模型（MLLMs）的空间关系推理能力，针对现有基准依赖边界框、忽略视角或无需图像理解的问题，构建了基于COCO2017的SpatialMQA数据集（5392样本），实验显示最优模型准确率48.14%（人类98.40%），指明未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.19015v1](http://arxiv.org/pdf/2505.19015v1)**

> **作者:** Jingping Liu; Ziyan Liu; Zhedong Cen; Yan Zhou; Yinan Zou; Weiyan Zhang; Haiyun Jiang; Tong Ruan
>
> **备注:** 13 pages, 19 figures
>
> **摘要:** Spatial relation reasoning is a crucial task for multimodal large language models (MLLMs) to understand the objective world. However, current benchmarks have issues like relying on bounding boxes, ignoring perspective substitutions, or allowing questions to be answered using only the model's prior knowledge without image understanding. To address these issues, we introduce SpatialMQA, a human-annotated spatial relation reasoning benchmark based on COCO2017, which enables MLLMs to focus more on understanding images in the objective world. To ensure data quality, we design a well-tailored annotation procedure, resulting in SpatialMQA consisting of 5,392 samples. Based on this benchmark, a series of closed- and open-source MLLMs are implemented and the results indicate that the current state-of-the-art MLLM achieves only 48.14% accuracy, far below the human-level accuracy of 98.40%. Extensive experimental analyses are also conducted, suggesting the future research directions. The benchmark and codes are available at https://github.com/ziyan-xiaoyu/SpatialMQA.git.
>
---
#### [new 230] Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Pose Splatter，一种基于3D高斯散射和形状雕刻的模型，用于动物姿态与外观量化。针对传统方法细节不足、依赖标注及计算成本高的问题，该方法无需先验几何、逐帧优化或人工标注，通过旋转不变嵌入编码姿态和外观。实验表明其能精准捕捉细微动作，提供更优低维嵌入，并支持大规模行为分析。**

- **链接: [http://arxiv.org/pdf/2505.18342v1](http://arxiv.org/pdf/2505.18342v1)**

> **作者:** Jack Goffinet; Youngjo Min; Carlo Tomasi; David E. Carlson
>
> **备注:** 19 pages, 13 figures
>
> **摘要:** Accurate and scalable quantification of animal pose and appearance is crucial for studying behavior. Current 3D pose estimation techniques, such as keypoint- and mesh-based techniques, often face challenges including limited representational detail, labor-intensive annotation requirements, and expensive per-frame optimization. These limitations hinder the study of subtle movements and can make large-scale analyses impractical. We propose Pose Splatter, a novel framework leveraging shape carving and 3D Gaussian splatting to model the complete pose and appearance of laboratory animals without prior knowledge of animal geometry, per-frame optimization, or manual annotations. We also propose a novel rotation-invariant visual embedding technique for encoding pose and appearance, designed to be a plug-in replacement for 3D keypoint data in downstream behavioral analyses. Experiments on datasets of mice, rats, and zebra finches show Pose Splatter learns accurate 3D animal geometries. Notably, Pose Splatter represents subtle variations in pose, provides better low-dimensional pose embeddings over state-of-the-art as evaluated by humans, and generalizes to unseen data. By eliminating annotation and per-frame optimization bottlenecks, Pose Splatter enables analysis of large-scale, longitudinal behavior needed to map genotype, neural activity, and micro-behavior at unprecedented resolution.
>
---
#### [new 231] Benchmarking Laparoscopic Surgical Image Restoration and Beyond
- **分类: cs.CV**

- **简介: 该论文属于腹腔镜手术图像修复任务，旨在解决烟雾、镜头雾化和污染导致的视觉退化问题。构建了含1020张图像的SurgClean数据集，评估22种修复算法，揭示临床需求差距，并分析手术场景与自然场景差异，推动专用算法发展。**

- **链接: [http://arxiv.org/pdf/2505.19161v1](http://arxiv.org/pdf/2505.19161v1)**

> **作者:** Jialun Pei; Diandian Guo; Donghui Yang; Zhixi Li; Yuxin Feng; Long Ma; Bo Du; Pheng-Ann Heng
>
> **摘要:** In laparoscopic surgery, a clear and high-quality visual field is critical for surgeons to make accurate intraoperative decisions. However, persistent visual degradation, including smoke generated by energy devices, lens fogging from thermal gradients, and lens contamination due to blood or tissue fluid splashes during surgical procedures, severely impair visual clarity. These degenerations can seriously hinder surgical workflow and pose risks to patient safety. To systematically investigate and address various forms of surgical scene degradation, we introduce a real-world open-source surgical image restoration dataset covering laparoscopic environments, called SurgClean, which involves multi-type image restoration tasks, e.g., desmoking, defogging, and desplashing. SurgClean comprises 1,020 images with diverse degradation types and corresponding paired reference labels. Based on SurgClean, we establish a standardized evaluation benchmark and provide performance for 22 representative generic task-specific image restoration approaches, including 12 generic and 10 task-specific image restoration approaches. Experimental results reveal substantial performance gaps relative to clinical requirements, highlighting a critical opportunity for algorithm advancements in intelligent surgical restoration. Furthermore, we explore the degradation discrepancies between surgical and natural scenes from structural perception and semantic understanding perspectives, providing fundamental insights for domain-specific image restoration research. Our work aims to empower the capabilities of restoration algorithms to increase surgical environments and improve the efficiency of clinical procedures.
>
---
#### [new 232] TextDiffuser-RL: Efficient and Robust Text Layout Optimization for High-Fidelity Text-to-Image Synthesis
- **分类: cs.CV; cs.AI; 68T05, 68T07, 68U10 68T05, 68T07, 68U10 68T05, 68T07, 68U10; I.2.6; I.2.7; I.2.10; I.5.1; I.4.9**

- **简介: 该论文属于文本到图像合成任务，针对现有方法计算资源消耗大、运行效率低的问题，提出基于强化学习的两阶段pipeline。通过优化文本布局生成，减少重叠并加速渲染，实现在CPU/GPU高效运行，保持高质量同时提升速度（快97.64%）与低内存需求（2MB）。**

- **链接: [http://arxiv.org/pdf/2505.19291v1](http://arxiv.org/pdf/2505.19291v1)**

> **作者:** Kazi Mahathir Rahman; Showrin Rahman; Sharmin Sultana Srishty
>
> **备注:** 14 pages, 26 figures. Submitted to arXiv for dissemination. Intended for future submission to a Generative AI conference
>
> **摘要:** Text-embedded image generation plays a critical role in industries such as graphic design, advertising, and digital content creation. Text-to-Image generation methods leveraging diffusion models, such as TextDiffuser-2, have demonstrated promising results in producing images with embedded text. TextDiffuser-2 effectively generates bounding box layouts that guide the rendering of visual text, achieving high fidelity and coherence. However, existing approaches often rely on resource-intensive processes and are limited in their ability to run efficiently on both CPU and GPU platforms. To address these challenges, we propose a novel two-stage pipeline that integrates reinforcement learning (RL) for rapid and optimized text layout generation with a diffusion-based image synthesis model. Our RL-based approach significantly accelerates the bounding box prediction step while reducing overlaps, allowing the system to run efficiently on both CPUs and GPUs. Extensive evaluations demonstrate that our framework maintains or surpasses TextDiffuser-2's quality in text placement and image synthesis, with markedly faster runtime and increased flexibility. Extensive evaluations demonstrate that our framework maintains or surpasses TextDiffuser-2's quality in text placement and image synthesis, with markedly faster runtime and increased flexibility. Our approach has been evaluated on the MARIOEval benchmark, achieving OCR and CLIPScore metrics close to state-of-the-art models, while being 97.64% more faster and requiring only 2MB of memory to run.
>
---
#### [new 233] Exploring Magnitude Preservation and Rotation Modulation in Diffusion Transformers
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于扩散模型优化任务，旨在解决Diffusion Transformer（DiT）训练不稳定问题。提出无归一化层的幅度保持设计，并引入旋转调制（用学习旋转替代传统缩放/平移）作为新条件方法。实验表明其显著提升生成性能（FID降12.8%），且参数较AdaLN减少5.4%。**

- **链接: [http://arxiv.org/pdf/2505.19122v1](http://arxiv.org/pdf/2505.19122v1)**

> **作者:** Eric Tillman Bill; Cristian Perez Jensen; Sotiris Anagnostidis; Dimitri von Rütte
>
> **摘要:** Denoising diffusion models exhibit remarkable generative capabilities, but remain challenging to train due to their inherent stochasticity, where high-variance gradient estimates lead to slow convergence. Previous works have shown that magnitude preservation helps with stabilizing training in the U-net architecture. This work explores whether this effect extends to the Diffusion Transformer (DiT) architecture. As such, we propose a magnitude-preserving design that stabilizes training without normalization layers. Motivated by the goal of maintaining activation magnitudes, we additionally introduce rotation modulation, which is a novel conditioning method using learned rotations instead of traditional scaling or shifting. Through empirical evaluations and ablation studies on small-scale models, we show that magnitude-preserving strategies significantly improve performance, notably reducing FID scores by $\sim$12.8%. Further, we show that rotation modulation combined with scaling is competitive with AdaLN, while requiring $\sim$5.4% fewer parameters. This work provides insights into conditioning strategies and magnitude control. We will publicly release the implementation of our method.
>
---
#### [new 234] GLEAM: Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scenes
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出GLEAM方法及基准GLEAM-Bench，解决复杂3D室内场景中移动机器人主动建图的泛化性不足问题。针对现有方法数据依赖强、探索策略保守的缺陷，GLEAM通过语义表征、长期导航目标和随机策略提升泛化能力，在1152场景数据集上实现66.50%覆盖率达州-of-the-art。**

- **链接: [http://arxiv.org/pdf/2505.20294v1](http://arxiv.org/pdf/2505.20294v1)**

> **作者:** Xiao Chen; Tai Wang; Quanyi Li; Tao Huang; Jiangmiao Pang; Tianfan Xue
>
> **备注:** Project page: https://xiao-chen.tech/gleam/
>
> **摘要:** Generalizable active mapping in complex unknown environments remains a critical challenge for mobile robots. Existing methods, constrained by insufficient training data and conservative exploration strategies, exhibit limited generalizability across scenes with diverse layouts and complex connectivity. To enable scalable training and reliable evaluation, we introduce GLEAM-Bench, the first large-scale benchmark designed for generalizable active mapping with 1,152 diverse 3D scenes from synthetic and real-scan datasets. Building upon this foundation, we propose GLEAM, a unified generalizable exploration policy for active mapping. Its superior generalizability comes mainly from our semantic representations, long-term navigable goals, and randomized strategies. It significantly outperforms state-of-the-art methods, achieving 66.50% coverage (+9.49%) with efficient trajectories and improved mapping accuracy on 128 unseen complex scenes. Project page: https://xiao-chen.tech/gleam/.
>
---
#### [new 235] Progressive Scaling Visual Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉目标跟踪任务，旨在解决扩展训练数据量、模型规模和输入分辨率时优化效果不佳及迭代优化受限的问题。提出渐进式扩展训练策略DT-Training，通过小模型知识迁移与双分支对齐提升模型潜力，在多数据集验证中超越现有方法，并拓展至其他任务。**

- **链接: [http://arxiv.org/pdf/2505.19990v1](http://arxiv.org/pdf/2505.19990v1)**

> **作者:** Jack Hong; Shilin Yan; Zehao Xiao; Jiayin Cai; Xiaolong Jiang; Yao Hu; Henghui Ding
>
> **摘要:** In this work, we propose a progressive scaling training strategy for visual object tracking, systematically analyzing the influence of training data volume, model size, and input resolution on tracking performance. Our empirical study reveals that while scaling each factor leads to significant improvements in tracking accuracy, naive training suffers from suboptimal optimization and limited iterative refinement. To address this issue, we introduce DT-Training, a progressive scaling framework that integrates small teacher transfer and dual-branch alignment to maximize model potential. The resulting scaled tracker consistently outperforms state-of-the-art methods across multiple benchmarks, demonstrating strong generalization and transferability of the proposed method. Furthermore, we validate the broader applicability of our approach to additional tasks, underscoring its versatility beyond tracking.
>
---
#### [new 236] Echo Planning for Autonomous Driving: From Current Observations to Future Trajectories and Back
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶路径规划任务，旨在解决现有系统因缺乏时序一致性导致预测误差累积的问题。提出Echo Planning框架，通过建立Current-Future-Current闭环循环，利用双向一致性约束（前向预测轨迹与反向重建场景的cycle loss）优化轨迹预测，提升路径合理性与安全性，无需额外监督即达最优效果。**

- **链接: [http://arxiv.org/pdf/2505.18945v1](http://arxiv.org/pdf/2505.18945v1)**

> **作者:** Jintao Sun; Hu Zhang; Gangyi Ding; Zhedong Zheng
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Modern end-to-end autonomous driving systems suffer from a critical limitation: their planners lack mechanisms to enforce temporal consistency between predicted trajectories and evolving scene dynamics. This absence of self-supervision allows early prediction errors to compound catastrophically over time. We introduce Echo Planning, a novel self-correcting framework that establishes a closed-loop Current - Future - Current (CFC) cycle to harmonize trajectory prediction with scene coherence. Our key insight is that plausible future trajectories must be bi-directionally consistent, ie, not only generated from current observations but also capable of reconstructing them. The CFC mechanism first predicts future trajectories from the Bird's-Eye-View (BEV) scene representation, then inversely maps these trajectories back to estimate the current BEV state. By enforcing consistency between the original and reconstructed BEV representations through a cycle loss, the framework intrinsically penalizes physically implausible or misaligned trajectories. Experiments on nuScenes demonstrate state-of-the-art performance, reducing L2 error by 0.04 m and collision rate by 0.12% compared to one-shot planners. Crucially, our method requires no additional supervision, leveraging the CFC cycle as an inductive bias for robust planning. This work offers a deployable solution for safety-critical autonomous systems.
>
---
#### [new 237] TESSER: Transfer-Enhancing Adversarial Attacks from Vision Transformers via Spectral and Semantic Regularization
- **分类: cs.CV**

- **简介: 该论文属于对抗攻击迁移性增强任务，旨在解决Vision Transformer（ViT）到CNN等模型的对抗样本转移性差的问题。提出TESSER框架，通过特征敏感梯度缩放（FSGS）和频谱平滑正则化（SSR），生成语义相关且频谱光滑的扰动，提升跨模型攻击成功率，实验显示其在ImageNet上比现有方法提升10.9%（CNN）和7.2%（ViTs）。**

- **链接: [http://arxiv.org/pdf/2505.19613v1](http://arxiv.org/pdf/2505.19613v1)**

> **作者:** Amira Guesmi; Bassem Ouni; Muhammad Shafique
>
> **摘要:** Adversarial transferability remains a critical challenge in evaluating the robustness of deep neural networks. In security-critical applications, transferability enables black-box attacks without access to model internals, making it a key concern for real-world adversarial threat assessment. While Vision Transformers (ViTs) have demonstrated strong adversarial performance, existing attacks often fail to transfer effectively across architectures, especially from ViTs to Convolutional Neural Networks (CNNs) or hybrid models. In this paper, we introduce \textbf{TESSER} -- a novel adversarial attack framework that enhances transferability via two key strategies: (1) \textit{Feature-Sensitive Gradient Scaling (FSGS)}, which modulates gradients based on token-wise importance derived from intermediate feature activations, and (2) \textit{Spectral Smoothness Regularization (SSR)}, which suppresses high-frequency noise in perturbations using a differentiable Gaussian prior. These components work in tandem to generate perturbations that are both semantically meaningful and spectrally smooth. Extensive experiments on ImageNet across 12 diverse architectures demonstrate that TESSER achieves +10.9\% higher attack succes rate (ASR) on CNNs and +7.2\% on ViTs compared to the state-of-the-art Adaptive Token Tuning (ATT) method. Moreover, TESSER significantly improves robustness against defended models, achieving 53.55\% ASR on adversarially trained CNNs. Qualitative analysis shows strong alignment between TESSER's perturbations and salient visual regions identified via Grad-CAM, while frequency-domain analysis reveals a 12\% reduction in high-frequency energy, confirming the effectiveness of spectral regularization.
>
---
#### [new 238] MLLM-Guided VLM Fine-Tuning with Joint Inference for Zero-Shot Composed Image Retrieval
- **分类: cs.CV; cs.IR**

- **简介: 该论文针对零样本组合图像检索（ZS-CIR）任务，提出MVFT-JI方法，解决现有方法因未优化组合查询表示导致的检索性能差的问题。通过联合训练多模态语言模型（MLLM）构建的两个无监督任务（目标文本检索与文本-图像检索），并结合推理时生成目标文本与图像相似度计算，提升复杂视觉场景的检索效果。**

- **链接: [http://arxiv.org/pdf/2505.19707v1](http://arxiv.org/pdf/2505.19707v1)**

> **作者:** Rong-Cheng Tu; Zhao Jin; Jingyi Liao; Xiao Luo; Yingjie Wang; Li Shen; Dacheng Tao
>
> **摘要:** Existing Zero-Shot Composed Image Retrieval (ZS-CIR) methods typically train adapters that convert reference images into pseudo-text tokens, which are concatenated with the modifying text and processed by frozen text encoders in pretrained VLMs or LLMs. While this design leverages the strengths of large pretrained models, it only supervises the adapter to produce encoder-compatible tokens that loosely preserve visual semantics. Crucially, it does not directly optimize the composed query representation to capture the full intent of the composition or to align with the target semantics, thereby limiting retrieval performance, particularly in cases involving fine-grained or complex visual transformations. To address this problem, we propose MLLM-Guided VLM Fine-Tuning with Joint Inference (MVFT-JI), a novel approach that leverages a pretrained multimodal large language model (MLLM) to construct two complementary training tasks using only unlabeled images: target text retrieval taskand text-to-image retrieval task. By jointly optimizing these tasks, our method enables the VLM to inherently acquire robust compositional retrieval capabilities, supported by the provided theoretical justifications and empirical validation. Furthermore, during inference, we further prompt the MLLM to generate target texts from composed queries and compute retrieval scores by integrating similarities between (i) the composed query and candidate images, and (ii) the MLLM-generated target text and candidate images. This strategy effectively combines the VLM's semantic alignment strengths with the MLLM's reasoning capabilities.
>
---
#### [new 239] Localizing Knowledge in Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于生成模型内部知识定位任务，旨在解决Diffusion Transformer（DiT）模型中知识分布不明确的问题。提出一种模型与知识无关的定位方法，识别DiT块中特定知识的位置，并验证其有效性；进一步应用于模型个性化和知识删除，实现高效、精准的模型编辑，减少计算并保留通用能力。**

- **链接: [http://arxiv.org/pdf/2505.18832v1](http://arxiv.org/pdf/2505.18832v1)**

> **作者:** Arman Zarei; Samyadeep Basu; Keivan Rezaei; Zihao Lin; Sayan Nag; Soheil Feizi
>
> **摘要:** Understanding how knowledge is distributed across the layers of generative models is crucial for improving interpretability, controllability, and adaptation. While prior work has explored knowledge localization in UNet-based architectures, Diffusion Transformer (DiT)-based models remain underexplored in this context. In this paper, we propose a model- and knowledge-agnostic method to localize where specific types of knowledge are encoded within the DiT blocks. We evaluate our method on state-of-the-art DiT-based models, including PixArt-alpha, FLUX, and SANA, across six diverse knowledge categories. We show that the identified blocks are both interpretable and causally linked to the expression of knowledge in generated outputs. Building on these insights, we apply our localization framework to two key applications: model personalization and knowledge unlearning. In both settings, our localized fine-tuning approach enables efficient and targeted updates, reducing computational cost, improving task-specific performance, and better preserving general model behavior with minimal interference to unrelated or surrounding content. Overall, our findings offer new insights into the internal structure of DiTs and introduce a practical pathway for more interpretable, efficient, and controllable model editing.
>
---
#### [new 240] MSLAU-Net: A Hybird CNN-Transformer Network for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决CNN全局特征不足与Transformer局部建模及计算复杂度高的问题。提出MSLAU-Net，结合CNN-Transformer架构，采用多尺度线性注意力机制高效捕获多尺度特征与长程依赖，并通过轻量级顶向下特征聚合恢复空间分辨率，实验验证其性能优越。**

- **链接: [http://arxiv.org/pdf/2505.18823v1](http://arxiv.org/pdf/2505.18823v1)**

> **作者:** Libin Lan; Yanxin Li; Xiaojuan Liu; Juan Zhou; Jianxun Zhang; Nannan Huang; Yudong Zhang
>
> **备注:** 13 pages, 7 figures, 7 tables
>
> **摘要:** Both CNN-based and Transformer-based methods have achieved remarkable success in medical image segmentation tasks. However, CNN-based methods struggle to effectively capture global contextual information due to the inherent limitations of convolution operations. Meanwhile, Transformer-based methods suffer from insufficient local feature modeling and face challenges related to the high computational complexity caused by the self-attention mechanism. To address these limitations, we propose a novel hybrid CNN-Transformer architecture, named MSLAU-Net, which integrates the strengths of both paradigms. The proposed MSLAU-Net incorporates two key ideas. First, it introduces Multi-Scale Linear Attention, designed to efficiently extract multi-scale features from medical images while modeling long-range dependencies with low computational complexity. Second, it adopts a top-down feature aggregation mechanism, which performs multi-level feature aggregation and restores spatial resolution using a lightweight structure. Extensive experiments conducted on benchmark datasets covering three imaging modalities demonstrate that the proposed MSLAU-Net outperforms other state-of-the-art methods on nearly all evaluation metrics, validating the superiority, effectiveness, and robustness of our approach. Our code is available at https://github.com/Monsoon49/MSLAU-Net.
>
---
#### [new 241] REGen: Multimodal Retrieval-Embedded Generation for Long-to-Short Video Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于长视频到短视频编辑任务，解决现有方法无法兼顾连贯叙事与插入原视频片段的问题。提出REGen框架，先用微调语言模型生成带占位符的脚本，再通过检索模型选取匹配视频片段嵌入，用于纪录片预告生成，实验验证其叙事连贯性和片段插入效果更优。**

- **链接: [http://arxiv.org/pdf/2505.18880v1](http://arxiv.org/pdf/2505.18880v1)**

> **作者:** Weihan Xu; Yimeng Ma; Jingyue Huang; Yang Li; Wenye Ma; Taylor Berg-Kirkpatrick; Julian McAuley; Paul Pu Liang; Hao-Wen Dong
>
> **摘要:** Short videos are an effective tool for promoting contents and improving knowledge accessibility. While existing extractive video summarization methods struggle to produce a coherent narrative, existing abstractive methods cannot `quote' from the input videos, i.e., inserting short video clips in their outputs. In this work, we explore novel video editing models for generating shorts that feature a coherent narrative with embedded video insertions extracted from a long input video. We propose a novel retrieval-embedded generation framework that allows a large language model to quote multimodal resources while maintaining a coherent narrative. Our proposed REGen system first generates the output story script with quote placeholders using a finetuned large language model, and then uses a novel retrieval model to replace the quote placeholders by selecting a video clip that best supports the narrative from a pool of candidate quotable video clips. We examine the proposed method on the task of documentary teaser generation, where short interview insertions are commonly used to support the narrative of a documentary. Our objective evaluations show that the proposed method can effectively insert short video clips while maintaining a coherent narrative. In a subjective survey, we show that our proposed method outperforms existing abstractive and extractive approaches in terms of coherence, alignment, and realism in teaser generation.
>
---
#### [new 242] Attention! You Vision Language Model Could Be Maliciously Manipulated
- **分类: cs.CV**

- **简介: 该论文属于对抗攻击任务，研究视觉语言模型（VLM）易受图像对抗样本攻击的问题。提出VMA方法，通过结合优化技术生成微小图像扰动，精准操控模型输出，实现越狱、隐私泄露等攻击及水印注入，并验证其跨场景有效性。**

- **链接: [http://arxiv.org/pdf/2505.19911v1](http://arxiv.org/pdf/2505.19911v1)**

> **作者:** Xiaosen Wang; Shaokang Wang; Zhijin Ge; Yuyang Luo; Shudong Zhang
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets.
>
---
#### [new 243] Point-RFT: Improving Multimodal Reasoning with Visually Grounded Reinforcement Finetuning
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决视觉语言任务中视觉幻觉和多模态整合不足的问题。提出Point-RFT框架，通过分阶段微调（格式微调+视觉接地强化学习），利用视觉元素锚定的推理步骤提升模型性能，在ChartQA等任务中准确率显著提升，展现跨领域泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.19702v1](http://arxiv.org/pdf/2505.19702v1)**

> **作者:** Minheng Ni; Zhengyuan Yang; Linjie Li; Chung-Ching Lin; Kevin Lin; Wangmeng Zuo; Lijuan Wang
>
> **摘要:** Recent advances in large language models have significantly improved textual reasoning through the effective use of Chain-of-Thought (CoT) and reinforcement learning. However, extending these successes to vision-language tasks remains challenging due to inherent limitations in text-only CoT, such as visual hallucinations and insufficient multimodal integration. In this paper, we introduce Point-RFT, a multimodal reasoning framework explicitly designed to leverage visually grounded CoT reasoning for visual document understanding. Our approach consists of two stages: First, we conduct format finetuning using a curated dataset of 71K diverse visual reasoning problems, each annotated with detailed, step-by-step rationales explicitly grounded to corresponding visual elements. Second, we employ reinforcement finetuning targeting visual document understanding. On ChartQA, our approach improves accuracy from 70.88% (format-finetuned baseline) to 90.04%, surpassing the 83.92% accuracy achieved by reinforcement finetuning relying solely on text-based CoT. The result shows that our grounded CoT is more effective for multimodal reasoning compared with the text-only CoT. Moreover, Point-RFT exhibits superior generalization capability across several out-of-domain visual document reasoning benchmarks, including CharXiv, PlotQA, IconQA, TabMWP, etc., and highlights its potential in complex real-world scenarios.
>
---
#### [new 244] Guiding the Experts: Semantic Priors for Efficient and Focused MoE Routing
- **分类: cs.CV**

- **简介: 该论文针对Soft MoE模型中专家路由未充分利用语义结构的问题，提出 foreground-guided策略。通过添加空间辅助损失和LayerScale机制，引导专家激活与语义前景对齐，提升路由效率和可解释性，在图像分类任务中取得性能提升。**

- **链接: [http://arxiv.org/pdf/2505.18586v1](http://arxiv.org/pdf/2505.18586v1)**

> **作者:** Chengxi Min; Wei Wang; Yahui Liu; Weixin Ye; Enver Sangineto; Qi Wang; Yao Zhao
>
> **摘要:** Mixture-of-Experts (MoE) models have emerged as a promising direction for scaling vision architectures efficiently. Among them, Soft MoE improves training stability by assigning each token to all experts via continuous dispatch weights. However, current designs overlook the semantic structure which is implicitly encoded in these weights, resulting in suboptimal expert routing. In this paper, we discover that dispatch weights in Soft MoE inherently exhibit segmentation-like patterns but are not explicitly aligned with semantic regions. Motivated by this observation, we propose a foreground-guided enhancement strategy. Specifically, we introduce a spatially aware auxiliary loss that encourages expert activation to align with semantic foreground regions. To further reinforce this supervision, we integrate a lightweight LayerScale mechanism that improves information flow and stabilizes optimization in skip connections. Our method necessitates only minor architectural adjustments and can be seamlessly integrated into prevailing Soft MoE frameworks. Comprehensive experiments on ImageNet-1K and multiple smaller-scale classification benchmarks not only showcase consistent performance enhancements but also reveal more interpretable expert routing mechanisms.
>
---
#### [new 245] Efficient Multi-modal Long Context Learning for Training-free Adaptation
- **分类: cs.CV**

- **简介: 该论文属于多模态模型任务适应任务，解决长上下文输入导致的计算与内存开销问题。提出EMLoC方法，通过分块压缩与层自适应剪枝技术，将长输入转化为紧凑表示，减少推理复杂度同时保持性能，实现高效零样本适配。**

- **链接: [http://arxiv.org/pdf/2505.19812v1](http://arxiv.org/pdf/2505.19812v1)**

> **作者:** Zehong Ma; Shiliang Zhang; Longhui Wei; Qi Tian
>
> **备注:** Accepted to ICML2025
>
> **摘要:** Traditional approaches to adapting multi-modal large language models (MLLMs) to new tasks have relied heavily on fine-tuning. This paper introduces Efficient Multi-Modal Long Context Learning (EMLoC), a novel training-free alternative that embeds demonstration examples directly into the model input. EMLoC offers a more efficient, flexible, and scalable solution for task adaptation. Because extremely lengthy inputs introduce prohibitive computational and memory overhead, EMLoC contributes a chunk-wise compression mechanism combined with layer-wise adaptive pruning. It condenses long-context multimodal inputs into compact, task-specific memory representations. By adaptively pruning tokens at each layer under a Jensen-Shannon divergence constraint, our method achieves a dramatic reduction in inference complexity without sacrificing performance. This approach is the first to seamlessly integrate compression and pruning techniques for multi-modal long-context learning, offering a scalable and efficient solution for real-world applications. Extensive experiments on diverse vision-language benchmarks demonstrate that EMLoC achieves performance on par with or superior to naive long-context approaches. Our results highlight the potential of EMLoC as a groundbreaking framework for efficient and flexible adaptation of multi-modal models in resource-constrained environments. Codes are publicly available at https://github.com/Zehong-Ma/EMLoC.
>
---
#### [new 246] CreatiDesign: A Unified Multi-Conditional Diffusion Transformer for Creative Graphic Design
- **分类: cs.CV**

- **简介: 该论文提出CreatiDesign，解决多条件控制在自动化图形设计中的挑战。现有模型或局限于单一条件，或无法精准控制多条件和谐。论文设计统一扩散Transformer架构，引入多模态注意力掩码机制，实现异构元素精准整合，并构建400K样本数据集，实验显示显著优势。**

- **链接: [http://arxiv.org/pdf/2505.19114v1](http://arxiv.org/pdf/2505.19114v1)**

> **作者:** Hui Zhang; Dexiang Hong; Maoke Yang; Yutao Chen; Zhao Zhang; Jie Shao; Xinglong Wu; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Graphic design plays a vital role in visual communication across advertising, marketing, and multimedia entertainment. Prior work has explored automated graphic design generation using diffusion models, aiming to streamline creative workflows and democratize design capabilities. However, complex graphic design scenarios require accurately adhering to design intent specified by multiple heterogeneous user-provided elements (\eg images, layouts, and texts), which pose multi-condition control challenges for existing methods. Specifically, previous single-condition control models demonstrate effectiveness only within their specialized domains but fail to generalize to other conditions, while existing multi-condition methods often lack fine-grained control over each sub-condition and compromise overall compositional harmony. To address these limitations, we introduce CreatiDesign, a systematic solution for automated graphic design covering both model architecture and dataset construction. First, we design a unified multi-condition driven architecture that enables flexible and precise integration of heterogeneous design elements with minimal architectural modifications to the base diffusion model. Furthermore, to ensure that each condition precisely controls its designated image region and to avoid interference between conditions, we propose a multimodal attention mask mechanism. Additionally, we develop a fully automated pipeline for constructing graphic design datasets, and introduce a new dataset with 400K samples featuring multi-condition annotations, along with a comprehensive benchmark. Experimental results show that CreatiDesign outperforms existing models by a clear margin in faithfully adhering to user intent.
>
---
#### [new 247] VLM-3R: Vision-Language Models Augmented with Instruction-Aligned 3D Reconstruction
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VLM-3R框架，通过整合3D重建指令调优，解决现有视觉语言模型依赖外部传感器和预建3D地图导致的扩展性及单目视频处理问题。其采用几何编码器生成3D空间特征，结合空间-视觉融合与20万QA对，实现空间-语言对齐，并构建新基准评估时空推理，提升3D场景理解与时间变化分析能力。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20279v1](http://arxiv.org/pdf/2505.20279v1)**

> **作者:** Zhiwen Fan; Jian Zhang; Renjie Li; Junge Zhang; Runjin Chen; Hezhen Hu; Kevin Wang; Huaizhi Qu; Dilin Wang; Zhicheng Yan; Hongyu Xu; Justin Theiss; Tianlong Chen; Jiachen Li; Zhengzhong Tu; Zhangyang Wang; Rakesh Ranjan
>
> **摘要:** The rapid advancement of Large Multimodal Models (LMMs) for 2D images and videos has motivated extending these models to understand 3D scenes, aiming for human-like visual-spatial intelligence. Nevertheless, achieving deep spatial understanding comparable to human capabilities poses significant challenges in model encoding and data acquisition. Existing methods frequently depend on external depth sensors for geometry capture or utilize off-the-shelf algorithms for pre-constructing 3D maps, thereby limiting their scalability, especially with prevalent monocular video inputs and for time-sensitive applications. In this work, we introduce VLM-3R, a unified framework for Vision-Language Models (VLMs) that incorporates 3D Reconstructive instruction tuning. VLM-3R processes monocular video frames by employing a geometry encoder to derive implicit 3D tokens that represent spatial understanding. Leveraging our Spatial-Visual-View Fusion and over 200K curated 3D reconstructive instruction tuning question-answer (QA) pairs, VLM-3R effectively aligns real-world spatial context with language instructions. This enables monocular 3D spatial assistance and embodied reasoning. To facilitate the evaluation of temporal reasoning, we introduce the Vision-Spatial-Temporal Intelligence benchmark, featuring over 138.6K QA pairs across five distinct tasks focused on evolving spatial relationships. Extensive experiments demonstrate that our model, VLM-3R, not only facilitates robust visual-spatial reasoning but also enables the understanding of temporal 3D context changes, excelling in both accuracy and scalability.
>
---
#### [new 248] Less is More: Efficient Point Cloud Reconstruction via Multi-Head Decoders
- **分类: cs.CV**

- **简介: 该论文属于点云重建任务，挑战"更深解码器必优"的假设，指出过深结构易过拟合。提出多头解码器，通过多独立头处理不同点子集并融合预测，提升重建多样性和精度。实验显示其在CD/EMD等指标上超越单头基线，证明多样性设计比单纯深度更重要。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19057v1](http://arxiv.org/pdf/2505.19057v1)**

> **作者:** Pedro Alonso; Tianrui Li; Chongshou Li
>
> **摘要:** We challenge the common assumption that deeper decoder architectures always yield better performance in point cloud reconstruction. Our analysis reveals that, beyond a certain depth, increasing decoder complexity leads to overfitting and degraded generalization. Additionally, we propose a novel multi-head decoder architecture that exploits the inherent redundancy in point clouds by reconstructing complete shapes from multiple independent heads, each operating on a distinct subset of points. The final output is obtained by concatenating the predictions from all heads, enhancing both diversity and fidelity. Extensive experiments on ModelNet40 and ShapeNetPart demonstrate that our approach achieves consistent improvements across key metrics--including Chamfer Distance (CD), Hausdorff Distance (HD), Earth Mover's Distance (EMD), and F1-score--outperforming standard single-head baselines. Our findings highlight that output diversity and architectural design can be more critical than depth alone for effective and efficient point cloud reconstruction.
>
---
#### [new 249] From Data to Modeling: Fully Open-vocabulary Scene Graph Generation
- **分类: cs.CV**

- **简介: 该论文属于开放词汇场景图生成任务，旨在解决传统闭集模型无法处理现实中新概念的问题。提出OvSGTR框架，采用DETR-like架构融合视觉与语义特征，通过关系感知预训练（三种弱监督pipeline）及视觉概念保留机制，实现全开放词汇的节点与边预测，在多个场景取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.20106v1](http://arxiv.org/pdf/2505.20106v1)**

> **作者:** Zuyao Chen; Jinlin Wu; Zhen Lei; Chang Wen Chen
>
> **摘要:** We present OvSGTR, a novel transformer-based framework for fully open-vocabulary scene graph generation that overcomes the limitations of traditional closed-set models. Conventional methods restrict both object and relationship recognition to a fixed vocabulary, hindering their applicability to real-world scenarios where novel concepts frequently emerge. In contrast, our approach jointly predicts objects (nodes) and their inter-relationships (edges) beyond predefined categories. OvSGTR leverages a DETR-like architecture featuring a frozen image backbone and text encoder to extract high-quality visual and semantic features, which are then fused via a transformer decoder for end-to-end scene graph prediction. To enrich the model's understanding of complex visual relations, we propose a relation-aware pre-training strategy that synthesizes scene graph annotations in a weakly supervised manner. Specifically, we investigate three pipelines--scene parser-based, LLM-based, and multimodal LLM-based--to generate transferable supervision signals with minimal manual annotation. Furthermore, we address the common issue of catastrophic forgetting in open-vocabulary settings by incorporating a visual-concept retention mechanism coupled with a knowledge distillation strategy, ensuring that the model retains rich semantic cues during fine-tuning. Extensive experiments on the VG150 benchmark demonstrate that OvSGTR achieves state-of-the-art performance across multiple settings, including closed-set, open-vocabulary object detection-based, relation-based, and fully open-vocabulary scenarios. Our results highlight the promise of large-scale relation-aware pre-training and transformer architectures for advancing scene graph generation towards more generalized and reliable visual understanding.
>
---
#### [new 250] Rethinking Metrics and Benchmarks of Video Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦视频异常检测（VAD），针对现有评估指标和基准的三大问题——单标注偏差、无法奖励早期检测、无法评估场景过拟合，提出多轮标注平均AUC/AP、延迟感知平均精度（LaAP）及两个硬正常基准（UCF-HN、MSAD-HN），并对比十种最新方法，为模型开发提供新视角。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19022v1](http://arxiv.org/pdf/2505.19022v1)**

> **作者:** Zihao Liu; Xiaoyu Wu; Wenna Li; Linlin Yang
>
> **摘要:** Video Anomaly Detection (VAD), which aims to detect anomalies that deviate from expectation, has attracted increasing attention in recent years. Existing advancements in VAD primarily focus on model architectures and training strategies, while devoting insufficient attention to evaluation metrics and benchmarks. In this paper, we rethink VAD evaluation protocols through comprehensive experimental analyses, revealing three critical limitations in current practices: 1) existing metrics are significantly influenced by single annotation bias; 2) current metrics fail to reward early detection of anomalies; 3) available benchmarks lack the capability to evaluate scene overfitting. To address these limitations, we propose three novel evaluation methods: first, we establish averaged AUC/AP metrics over multi-round annotations to mitigate single annotation bias; second, we develop a Latency-aware Average Precision (LaAP) metric that rewards early and accurate anomaly detection; and finally, we introduce two hard normal benchmarks (UCF-HN, MSAD-HN) with videos specifically designed to evaluate scene overfitting. We report performance comparisons of ten state-of-the-art VAD approaches using our proposed evaluation methods, providing novel perspectives for future VAD model development.
>
---
#### [new 251] Omni-R1: Reinforcement Learning for Omnimodal Reasoning via Two-System Collaboration
- **分类: cs.CV**

- **简介: 该论文提出Omni-R1框架，通过双系统协作解决长时视频-音频推理与高精度像素定位的矛盾。全局系统以强化学习选择关键帧并重定义任务，细节系统处理高分辨率片段。实验表明其优于现有方法，提升跨域泛化并减少模态幻觉，首次实现RL在大规模多模态推理的成功应用。（99字）**

- **链接: [http://arxiv.org/pdf/2505.20256v1](http://arxiv.org/pdf/2505.20256v1)**

> **作者:** Hao Zhong; Muzhi Zhu; Zongze Du; Zheng Huang; Canyu Zhao; Mingyu Liu; Wen Wang; Hao Chen; Chunhua Shen
>
> **备注:** Project page: https://aim-uofa.github.io/OmniR1
>
> **摘要:** Long-horizon video-audio reasoning and fine-grained pixel understanding impose conflicting requirements on omnimodal models: dense temporal coverage demands many low-resolution frames, whereas precise grounding calls for high-resolution inputs. We tackle this trade-off with a two-system architecture: a Global Reasoning System selects informative keyframes and rewrites the task at low spatial cost, while a Detail Understanding System performs pixel-level grounding on the selected high-resolution snippets. Because ``optimal'' keyframe selection and reformulation are ambiguous and hard to supervise, we formulate them as a reinforcement learning (RL) problem and present Omni-R1, an end-to-end RL framework built on Group Relative Policy Optimization. Omni-R1 trains the Global Reasoning System through hierarchical rewards obtained via online collaboration with the Detail Understanding System, requiring only one epoch of RL on small task splits. Experiments on two challenging benchmarks, namely Referring Audio-Visual Segmentation (RefAVS) and Reasoning Video Object Segmentation (REVOS), show that Omni-R1 not only surpasses strong supervised baselines but also outperforms specialized state-of-the-art models, while substantially improving out-of-domain generalization and mitigating multimodal hallucination. Our results demonstrate the first successful application of RL to large-scale omnimodal reasoning and highlight a scalable path toward universally foundation models.
>
---
#### [new 252] Data-Free Class-Incremental Gesture Recognition with Prototype-Guided Pseudo Feature Replay
- **分类: cs.CV**

- **简介: 该论文属于无数据的类增量手势识别任务，解决模型在逐步学习新手势时遗忘旧类别的问题。提出PGPFR框架，通过伪特征生成、原型回放、截断交叉熵及分类器重训练四组件，动态生成多样化伪特征并约束原型一致性，抑制灾难性遗忘，提升新旧类别识别精度，实验显示显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.20049v1](http://arxiv.org/pdf/2505.20049v1)**

> **作者:** Hongsong Wang; Ao Sun; Jie Gui; Liang Wang
>
> **备注:** Code is on https://github.com/sunao-101/PGPFR-3/
>
> **摘要:** Gesture recognition is an important research area in the field of computer vision. Most gesture recognition efforts focus on close-set scenarios, thereby limiting the capacity to effectively handle unseen or novel gestures. We aim to address class-incremental gesture recognition, which entails the ability to accommodate new and previously unseen gestures over time. Specifically, we introduce a Prototype-Guided Pseudo Feature Replay (PGPFR) framework for data-free class-incremental gesture recognition. This framework comprises four components: Pseudo Feature Generation with Batch Prototypes (PFGBP), Variational Prototype Replay (VPR) for old classes, Truncated Cross-Entropy (TCE) for new classes, and Continual Classifier Re-Training (CCRT). To tackle the issue of catastrophic forgetting, the PFGBP dynamically generates a diversity of pseudo features in an online manner, leveraging class prototypes of old classes along with batch class prototypes of new classes. Furthermore, the VPR enforces consistency between the classifier's weights and the prototypes of old classes, leveraging class prototypes and covariance matrices to enhance robustness and generalization capabilities. The TCE mitigates the impact of domain differences of the classifier caused by pseudo features. Finally, the CCRT training strategy is designed to prevent overfitting to new classes and ensure the stability of features extracted from old classes. Extensive experiments conducted on two widely used gesture recognition datasets, namely SHREC 2017 3D and EgoGesture 3D, demonstrate that our approach outperforms existing state-of-the-art methods by 11.8\% and 12.8\% in terms of mean global accuracy, respectively. The code is available on https://github.com/sunao-101/PGPFR-3/.
>
---
#### [new 253] Triangle Splatting for Real-Time Radiance Field Rendering
- **分类: cs.CV**

- **简介: 该论文提出三角形可微分渲染方法（Triangle Splatting），用于实时辐射场渲染。针对传统方法效率低、收敛慢的问题，将三角形转化为可微分图元，结合经典图形学与现代优化框架，提升视觉质量与渲染速度，在Garden场景实现2400 FPS，优于NeRF等方法。**

- **链接: [http://arxiv.org/pdf/2505.19175v1](http://arxiv.org/pdf/2505.19175v1)**

> **作者:** Jan Held; Renaud Vandeghen; Adrien Deliege; Abdullah Hamdi; Silvio Giancola; Anthony Cioppa; Andrea Vedaldi; Bernard Ghanem; Andrea Tagliasacchi; Marc Van Droogenbroeck
>
> **备注:** 18 pages, 13 figures, 10 tables
>
> **摘要:** The field of computer graphics was revolutionized by models such as Neural Radiance Fields and 3D Gaussian Splatting, displacing triangles as the dominant representation for photogrammetry. In this paper, we argue for a triangle comeback. We develop a differentiable renderer that directly optimizes triangles via end-to-end gradients. We achieve this by rendering each triangle as differentiable splats, combining the efficiency of triangles with the adaptive density of representations based on independent primitives. Compared to popular 2D and 3D Gaussian Splatting methods, our approach achieves higher visual fidelity, faster convergence, and increased rendering throughput. On the Mip-NeRF360 dataset, our method outperforms concurrent non-volumetric primitives in visual fidelity and achieves higher perceptual quality than the state-of-the-art Zip-NeRF on indoor scenes. Triangles are simple, compatible with standard graphics stacks and GPU hardware, and highly efficient: for the \textit{Garden} scene, we achieve over 2,400 FPS at 1280x720 resolution using an off-the-shelf mesh renderer. These results highlight the efficiency and effectiveness of triangle-based representations for high-quality novel view synthesis. Triangles bring us closer to mesh-based optimization by combining classical computer graphics with modern differentiable rendering frameworks. The project page is https://trianglesplatting.github.io/
>
---
#### [new 254] Towards Understanding the Mechanisms of Classifier-Free Guidance
- **分类: cs.CV**

- **简介: 该论文研究Classifier-Free Guidance（CFG）机制，解析其提升图像生成质量的三个组件（均值偏移、正负对比主成分），并通过线性与非线性扩散模型验证，揭示其作用原理。**

- **链接: [http://arxiv.org/pdf/2505.19210v1](http://arxiv.org/pdf/2505.19210v1)**

> **作者:** Xiang Li; Rongrong Wang; Qing Qu
>
> **摘要:** Classifier-free guidance (CFG) is a core technique powering state-of-the-art image generation systems, yet its underlying mechanisms remain poorly understood. In this work, we begin by analyzing CFG in a simplified linear diffusion model, where we show its behavior closely resembles that observed in the nonlinear case. Our analysis reveals that linear CFG improves generation quality via three distinct components: (i) a mean-shift term that approximately steers samples in the direction of class means, (ii) a positive Contrastive Principal Components (CPC) term that amplifies class-specific features, and (iii) a negative CPC term that suppresses generic features prevalent in unconditional data. We then verify that these insights in real-world, nonlinear diffusion models: over a broad range of noise levels, linear CFG resembles the behavior of its nonlinear counterpart. Although the two eventually diverge at low noise levels, we discuss how the insights from the linear analysis still shed light on the CFG's mechanism in the nonlinear regime.
>
---
#### [new 255] Knowledge-Aligned Counterfactual-Enhancement Diffusion Perception for Unsupervised Cross-Domain Visual Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文提出无监督跨领域视觉情绪识别（UCDVER）任务，解决跨领域情绪表达差异与分布偏移问题。通过KCDP框架，利用视觉语言模型对齐情感表征，结合扩散模型增强感知，并生成伪标签，显著提升跨领域情绪识别性能。**

- **链接: [http://arxiv.org/pdf/2505.19694v1](http://arxiv.org/pdf/2505.19694v1)**

> **作者:** Wen Yin; Yong Wang; Guiduo Duan; Dongyang Zhang; Xin Hu; Yuan-Fang Li; Tao He
>
> **备注:** Accepted at CVPR 2025
>
> **摘要:** Visual Emotion Recognition (VER) is a critical yet challenging task aimed at inferring emotional states of individuals based on visual cues. However, existing works focus on single domains, e.g., realistic images or stickers, limiting VER models' cross-domain generalizability. To fill this gap, we introduce an Unsupervised Cross-Domain Visual Emotion Recognition (UCDVER) task, which aims to generalize visual emotion recognition from the source domain (e.g., realistic images) to the low-resource target domain (e.g., stickers) in an unsupervised manner. Compared to the conventional unsupervised domain adaptation problems, UCDVER presents two key challenges: a significant emotional expression variability and an affective distribution shift. To mitigate these issues, we propose the Knowledge-aligned Counterfactual-enhancement Diffusion Perception (KCDP) framework. Specifically, KCDP leverages a VLM to align emotional representations in a shared knowledge space and guides diffusion models for improved visual affective perception. Furthermore, a Counterfactual-Enhanced Language-image Emotional Alignment (CLIEA) method generates high-quality pseudo-labels for the target domain. Extensive experiments demonstrate that our model surpasses SOTA models in both perceptibility and generalization, e.g., gaining 12% improvements over the SOTA VER model TGCA-PVT. The project page is at https://yinwen2019.github.io/ucdver.
>
---
#### [new 256] Improved Immiscible Diffusion: Accelerate Diffusion Training by Reducing Its Miscibility
- **分类: cs.CV**

- **简介: 该论文属于扩散模型训练优化任务，旨在解决其高计算成本问题。通过改进Immiscible Diffusion方法，提出KNN噪声选择、图像缩放等技术减少扩散轨迹混合，加速训练（提速超4倍），并分析最优传输作用，揭示轨迹混合为关键瓶颈，为高效训练开辟新方向。**

- **链接: [http://arxiv.org/pdf/2505.18521v1](http://arxiv.org/pdf/2505.18521v1)**

> **作者:** Yiheng Li; Feng Liang; Dan Kondratyuk; Masayoshi Tomizuka; Kurt Keutzer; Chenfeng Xu
>
> **摘要:** The substantial training cost of diffusion models hinders their deployment. Immiscible Diffusion recently showed that reducing diffusion trajectory mixing in the noise space via linear assignment accelerates training by simplifying denoising. To extend immiscible diffusion beyond the inefficient linear assignment under high batch sizes and high dimensions, we refine this concept to a broader miscibility reduction at any layer and by any implementation. Specifically, we empirically demonstrate the bijective nature of the denoising process with respect to immiscible diffusion, ensuring its preservation of generative diversity. Moreover, we provide thorough analysis and show step-by-step how immiscibility eases denoising and improves efficiency. Extending beyond linear assignment, we propose a family of implementations including K-nearest neighbor (KNN) noise selection and image scaling to reduce miscibility, achieving up to >4x faster training across diverse models and tasks including unconditional/conditional generation, image editing, and robotics planning. Furthermore, our analysis of immiscibility offers a novel perspective on how optimal transport (OT) enhances diffusion training. By identifying trajectory miscibility as a fundamental bottleneck, we believe this work establishes a potentially new direction for future research into high-efficiency diffusion training. The code is available at https://github.com/yhli123/Immiscible-Diffusion.
>
---
#### [new 257] Rep3D: Re-parameterize Large 3D Kernels with Low-Rank Receptive Modeling for Medical Imaging
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D医学图像分割任务，解决大核卷积优化不稳定与性能退化问题。提出Rep3D框架，通过可学习空间先验与轻量调制网络生成感受野偏置掩码，自适应调整核参数更新，采用深度可分离卷积简化结构，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2505.19603v1](http://arxiv.org/pdf/2505.19603v1)**

> **作者:** Ho Hin Lee; Quan Liu; Shunxing Bao; Yuankai Huo; Bennett A. Landman
>
> **备注:** 14 pages
>
> **摘要:** In contrast to vision transformers, which model long-range dependencies through global self-attention, large kernel convolutions provide a more efficient and scalable alternative, particularly in high-resolution 3D volumetric settings. However, naively increasing kernel size often leads to optimization instability and degradation in performance. Motivated by the spatial bias observed in effective receptive fields (ERFs), we hypothesize that different kernel elements converge at variable rates during training. To support this, we derive a theoretical connection between element-wise gradients and first-order optimization, showing that structurally re-parameterized convolution blocks inherently induce spatially varying learning rates. Building on this insight, we introduce Rep3D, a 3D convolutional framework that incorporates a learnable spatial prior into large kernel training. A lightweight two-stage modulation network generates a receptive-biased scaling mask, adaptively re-weighting kernel updates and enabling local-to-global convergence behavior. Rep3D adopts a plain encoder design with large depthwise convolutions, avoiding the architectural complexity of multi-branch compositions. We evaluate Rep3D on five challenging 3D segmentation benchmarks and demonstrate consistent improvements over state-of-the-art baselines, including transformer-based and fixed-prior re-parameterization methods. By unifying spatial inductive bias with optimization-aware learning, Rep3D offers an interpretable, and scalable solution for 3D medical image analysis. The source code is publicly available at https://github.com/leeh43/Rep3D.
>
---
#### [new 258] Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成与物理交互控制任务，解决模型对物理力控制信号的学习与泛化问题。提出"力提示"方法，通过局部点力或全局力场（如戳、风）引导视频生成，利用预训练模型的视觉先验，无需3D资产或物理引擎。使用Blender合成数据克服真实数据稀缺问题，验证模型在多样场景的泛化能力，并分析视觉多样性和关键词对性能的关键作用。**

- **链接: [http://arxiv.org/pdf/2505.19386v1](http://arxiv.org/pdf/2505.19386v1)**

> **作者:** Nate Gillman; Charles Herrmann; Michael Freeman; Daksh Aggarwal; Evan Luo; Deqing Sun; Chen Sun
>
> **备注:** Project page: https://force-prompting.github.io/
>
> **摘要:** Recent advances in video generation models have sparked interest in world models capable of simulating realistic environments. While navigation has been well-explored, physically meaningful interactions that mimic real-world forces remain largely understudied. In this work, we investigate using physical forces as a control signal for video generation and propose force prompts which enable users to interact with images through both localized point forces, such as poking a plant, and global wind force fields, such as wind blowing on fabric. We demonstrate that these force prompts can enable videos to respond realistically to physical control signals by leveraging the visual and motion prior in the original pretrained model, without using any 3D asset or physics simulator at inference. The primary challenge of force prompting is the difficulty in obtaining high quality paired force-video training data, both in the real world due to the difficulty of obtaining force signals, and in synthetic data due to limitations in the visual quality and domain diversity of physics simulators. Our key finding is that video generation models can generalize remarkably well when adapted to follow physical force conditioning from videos synthesized by Blender, even with limited demonstrations of few objects. Our method can generate videos which simulate forces across diverse geometries, settings, and materials. We also try to understand the source of this generalization and perform ablations that reveal two key elements: visual diversity and the use of specific text keywords during training. Our approach is trained on only around 15k training examples for a single day on four A100 GPUs, and outperforms existing methods on force adherence and physics realism, bringing world models closer to real-world physics interactions. We release all datasets, code, weights, and interactive video demos at our project page.
>
---
#### [new 259] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态模型优化任务，针对传统DPO方法仅对完整回复进行二元奖励导致细粒度推理不足的问题，提出ASPO方法：通过动态计算句子级自适应奖励实现细粒度评估，提升多模态对齐效果，无需额外模型或参数。**

- **链接: [http://arxiv.org/pdf/2505.19100v1](http://arxiv.org/pdf/2505.19100v1)**

> **作者:** Yeyuan Wang; Dehong Gao; Rujiao Long; Lei Yi; Linbo Jin; Libin Yang; Xiaoyan Cai
>
> **备注:** Accepted by ACL 2025 findings
>
> **摘要:** Direct Preference Optimization (DPO) has gained significant attention for its simplicity and computational efficiency in aligning large language models (LLMs). Recent advancements have extended DPO to multimodal scenarios, achieving strong performance. However, traditional DPO relies on binary preference optimization, rewarding or penalizing entire responses without considering fine-grained segment correctness, leading to suboptimal solutions. The root of this issue lies in the absence of fine-grained supervision during the optimization process. To address this, we propose Adaptive Sentence-level Preference Optimization (ASPO), which evaluates individual sentences for more precise preference optimization. By dynamically calculating adaptive rewards at the sentence level based on model predictions, ASPO enhances response content assessment without additional models or parameters. This significantly improves the alignment of multimodal features. Extensive experiments show that ASPO substantially enhances the overall performance of multimodal models.
>
---
#### [new 260] Unsupervised cell segmentation by fast Gaussian Processes
- **分类: stat.AP; cs.CV**

- **简介: 该论文提出一种无监督细胞分割算法，基于快速高斯过程解决传统监督方法依赖标注数据及参数调优的问题。通过自适应阈值分离背景并结合分水岭分割处理接触细胞，适用于噪声显微图像，实验验证其准确性和扩展性。**

- **链接: [http://arxiv.org/pdf/2505.18902v1](http://arxiv.org/pdf/2505.18902v1)**

> **作者:** Laura Baracaldo; Blythe King; Haoran Yan; Yizi Lin; Nina Miolane; Mengyang Gu
>
> **摘要:** Cell boundary information is crucial for analyzing cell behaviors from time-lapse microscopy videos. Existing supervised cell segmentation tools, such as ImageJ, require tuning various parameters and rely on restrictive assumptions about the shape of the objects. While recent supervised segmentation tools based on convolutional neural networks enhance accuracy, they depend on high-quality labelled images, making them unsuitable for segmenting new types of objects not in the database. We developed a novel unsupervised cell segmentation algorithm based on fast Gaussian processes for noisy microscopy images without the need for parameter tuning or restrictive assumptions about the shape of the object. We derived robust thresholding criteria adaptive for heterogeneous images containing distinct brightness at different parts to separate objects from the background, and employed watershed segmentation to distinguish touching cell objects. Both simulated studies and real-data analysis of large microscopy images demonstrate the scalability and accuracy of our approach compared with the alternatives.
>
---
#### [new 261] FieldWorkArena: Agentic AI Benchmark for Real Field Work Tasks
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出FieldWorkArena基准，针对真实工作环境中的agentic AI评估，解决现有方法局限于网络任务的问题。通过定义新动作空间、改进多模态LLM评估方法，基于工厂/仓库实拍视频和文档设计任务，验证了评估可行性，并公开数据集和代码。**

- **链接: [http://arxiv.org/pdf/2505.19662v1](http://arxiv.org/pdf/2505.19662v1)**

> **作者:** Atsunori Moteki; Shoichi Masui; Fan Yang; Yueqi Song; Yonatan Bisk; Graham Neubig; Ikuo Kusajima; Yasuto Watanabe; Hiroyuki Ishida; Jun Takahashi; Shan Jiang
>
> **备注:** 6 pages, 2 figures, 4 tables
>
> **摘要:** This paper proposes FieldWorkArena, a benchmark for agentic AI targeting real-world field work. With the recent increase in demand for agentic AI, they are required to monitor and report safety and health incidents, as well as manufacturing-related incidents, that may occur in real-world work environments. Existing agentic AI benchmarks have been limited to evaluating web tasks and are insufficient for evaluating agents in real-world work environments, where complexity increases significantly. In this paper, we define a new action space that agentic AI should possess for real world work environment benchmarks and improve the evaluation function from previous methods to assess the performance of agentic AI in diverse real-world tasks. The dataset consists of videos captured on-site and documents actually used in factories and warehouses, and tasks were created based on interviews with on-site workers and managers. Evaluation results confirmed that performance evaluation considering the characteristics of Multimodal LLM (MLLM) such as GPT-4o is feasible. Additionally, the effectiveness and limitations of the proposed new evaluation method were identified. The complete dataset (HuggingFace) and evaluation program (GitHub) can be downloaded from the following website: https://en-documents.research.global.fujitsu.com/fieldworkarena/.
>
---
#### [new 262] Understanding Generalization in Diffusion Models via Probability Flow Distance
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对扩散模型泛化能力评估困难问题，提出概率流距离（PFD）——通过比较分布的噪声-数据映射量化差异。采用教师-学生协议，揭示模型泛化行为（如记忆-泛化缩放、双下降动态等），并分析偏差-方差分解，为理论与实证研究奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.20123v1](http://arxiv.org/pdf/2505.20123v1)**

> **作者:** Huijie Zhang; Zijian Huang; Siyi Chen; Jinfan Zhou; Zekai Zhang; Peng Wang; Qing Qu
>
> **备注:** 41 pages, 14 figures
>
> **摘要:** Diffusion models have emerged as a powerful class of generative models, capable of producing high-quality samples that generalize beyond the training data. However, evaluating this generalization remains challenging: theoretical metrics are often impractical for high-dimensional data, while no practical metrics rigorously measure generalization. In this work, we bridge this gap by introducing probability flow distance ($\texttt{PFD}$), a theoretically grounded and computationally efficient metric to measure distributional generalization. Specifically, $\texttt{PFD}$ quantifies the distance between distributions by comparing their noise-to-data mappings induced by the probability flow ODE. Moreover, by using $\texttt{PFD}$ under a teacher-student evaluation protocol, we empirically uncover several key generalization behaviors in diffusion models, including: (1) scaling behavior from memorization to generalization, (2) early learning and double descent training dynamics, and (3) bias-variance decomposition. Beyond these insights, our work lays a foundation for future empirical and theoretical studies on generalization in diffusion models.
>
---
#### [new 263] Exploring the Possibility of TypiClust for Low-Budget Federated Active Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文研究低预算联邦主动学习（FAL）任务，解决在数据异构等挑战下如何高效标注的问题。通过验证TypiClust在FAL中的有效性，对比其他方法，分析其对分布偏移和特征提取的鲁棒性，证明其在有限标注场景下的适用性。**

- **链接: [http://arxiv.org/pdf/2505.19404v1](http://arxiv.org/pdf/2505.19404v1)**

> **作者:** Yuta Ono; Hiroshi Nakamura; Hideki Takase
>
> **备注:** 6 pages. Accepted at COMPSAC 2025
>
> **摘要:** Federated Active Learning (FAL) seeks to reduce the burden of annotation under the realistic constraints of federated learning by leveraging Active Learning (AL). As FAL settings make it more expensive to obtain ground truth labels, FAL strategies that work well in low-budget regimes, where the amount of annotation is very limited, are needed. In this work, we investigate the effectiveness of TypiClust, a successful low-budget AL strategy, in low-budget FAL settings. Our empirical results show that TypiClust works well even in low-budget FAL settings contrasted with relatively low performances of other methods, although these settings present additional challenges, such as data heterogeneity, compared to AL. In addition, we show that FAL settings cause distribution shifts in terms of typicality, but TypiClust is not very vulnerable to the shifts. We also analyze the sensitivity of TypiClust to feature extraction methods, and it suggests a way to perform FAL even in limited data situations.
>
---
#### [new 264] WorldEval: World Model as Real-World Robot Policies Evaluator
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人策略评估任务，旨在解决真实场景下机器人策略评估耗时、难扩展的问题。提出Policy2Vec方法将视频生成模型转化为遵循潜在动作的模拟器，并构建WorldEval自动化在线评估系统，可排序策略、检测安全风险，实验证明其性能与真实场景强相关且优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19017v1](http://arxiv.org/pdf/2505.19017v1)**

> **作者:** Yaxuan Li; Yichen Zhu; Junjie Wen; Chaomin Shen; Yi Xu
>
> **备注:** The project page is available at https://worldeval.github.io
>
> **摘要:** The field of robotics has made significant strides toward developing generalist robot manipulation policies. However, evaluating these policies in real-world scenarios remains time-consuming and challenging, particularly as the number of tasks scales and environmental conditions change. In this work, we demonstrate that world models can serve as a scalable, reproducible, and reliable proxy for real-world robot policy evaluation. A key challenge is generating accurate policy videos from world models that faithfully reflect the robot actions. We observe that directly inputting robot actions or using high-dimensional encoding methods often fails to generate action-following videos. To address this, we propose Policy2Vec, a simple yet effective approach to turn a video generation model into a world simulator that follows latent action to generate the robot video. We then introduce WorldEval, an automated pipeline designed to evaluate real-world robot policies entirely online. WorldEval effectively ranks various robot policies and individual checkpoints within a single policy, and functions as a safety detector to prevent dangerous actions by newly developed robot models. Through comprehensive paired evaluations of manipulation policies in real-world environments, we demonstrate a strong correlation between policy performance in WorldEval and real-world scenarios. Furthermore, our method significantly outperforms popular methods such as real-to-sim approach.
>
---
#### [new 265] SRDiffusion: Accelerate Video Diffusion Inference via Sketching-Rendering Cooperation
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决扩散模型推理计算成本高的问题。提出SRDiffusion框架，通过大模型（高噪声阶段保证语义/运动）与小模型（低噪声阶段细化细节）协作，实现3倍加速且质量无损，提供新型高效视频生成方案。**

- **链接: [http://arxiv.org/pdf/2505.19151v1](http://arxiv.org/pdf/2505.19151v1)**

> **作者:** Shenggan Cheng; Yuanxin Wei; Lansong Diao; Yong Liu; Bujiao Chen; Lianghua Huang; Yu Liu; Wenyuan Yu; Jiangsu Du; Wei Lin; Yang You
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Leveraging the diffusion transformer (DiT) architecture, models like Sora, CogVideoX and Wan have achieved remarkable progress in text-to-video, image-to-video, and video editing tasks. Despite these advances, diffusion-based video generation remains computationally intensive, especially for high-resolution, long-duration videos. Prior work accelerates its inference by skipping computation, usually at the cost of severe quality degradation. In this paper, we propose SRDiffusion, a novel framework that leverages collaboration between large and small models to reduce inference cost. The large model handles high-noise steps to ensure semantic and motion fidelity (Sketching), while the smaller model refines visual details in low-noise steps (Rendering). Experimental results demonstrate that our method outperforms existing approaches, over 3$\times$ speedup for Wan with nearly no quality loss for VBench, and 2$\times$ speedup for CogVideoX. Our method is introduced as a new direction orthogonal to existing acceleration strategies, offering a practical solution for scalable video generation.
>
---
#### [new 266] CageNet: A Meta-Framework for Learning on Wild Meshes
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文提出CageNet，一种处理复杂wild meshes（多部件、非流形等）的元框架。通过构建单部件流形cage包裹目标网格，并利用广义重心坐标映射，解决现有方法难以处理不规则网格的问题。在分割和蒙皮权重学习任务中，性能优于现有技术。（98字）**

- **链接: [http://arxiv.org/pdf/2505.18772v1](http://arxiv.org/pdf/2505.18772v1)**

> **作者:** Michal Edelstein; Hsueh-Ti Derek Liu; Mirela Ben-Chen
>
> **备注:** 11 pages, 13 figures (excluding supplementary material)
>
> **摘要:** Learning on triangle meshes has recently proven to be instrumental to a myriad of tasks, from shape classification, to segmentation, to deformation and animation, to mention just a few. While some of these applications are tackled through neural network architectures which are tailored to the application at hand, many others use generic frameworks for triangle meshes where the only customization required is the modification of the input features and the loss function. Our goal in this paper is to broaden the applicability of these generic frameworks to "wild", i.e. meshes in-the-wild which often have multiple components, non-manifold elements, disrupted connectivity, or a combination of these. We propose a configurable meta-framework based on the concept of caged geometry: Given a mesh, a cage is a single component manifold triangle mesh that envelopes it closely. Generalized barycentric coordinates map between functions on the cage, and functions on the mesh, allowing us to learn and test on a variety of data, in different applications. We demonstrate this concept by learning segmentation and skinning weights on difficult data, achieving better performance to state of the art techniques on wild meshes.
>
---
#### [new 267] Reinforcement Fine-Tuning Powers Reasoning Capability of Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态大模型优化任务，旨在通过强化微调（RFT）提升多模态大语言模型（MLLMs）的推理能力。研究总结了RFT在多模态数据、跨任务/领域适配、训练算法、评估基准和工程框架五方面的改进，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.18536v1](http://arxiv.org/pdf/2505.18536v1)**

> **作者:** Haoyuan Sun; Jiaqi Wu; Bo Xia; Yifu Luo; Yifei Zhao; Kai Qin; Xufei Lv; Tiantian Zhang; Yongzhe Chang; Xueqian Wang
>
> **摘要:** Standing in 2025, at a critical juncture in the pursuit of Artificial General Intelligence (AGI), reinforcement fine-tuning (RFT) has demonstrated significant potential in enhancing the reasoning capability of large language models (LLMs) and has led to the development of cutting-edge AI models such as OpenAI-o1 and DeepSeek-R1. Moreover, the efficient application of RFT to enhance the reasoning capability of multimodal large language models (MLLMs) has attracted widespread attention from the community. In this position paper, we argue that reinforcement fine-tuning powers the reasoning capability of multimodal large language models. To begin with, we provide a detailed introduction to the fundamental background knowledge that researchers interested in this field should be familiar with. Furthermore, we meticulously summarize the improvements of RFT in powering reasoning capability of MLLMs into five key points: diverse modalities, diverse tasks and domains, better training algorithms, abundant benchmarks and thriving engineering frameworks. Finally, we propose five promising directions for future research that the community might consider. We hope that this position paper will provide valuable insights to the community at this pivotal stage in the advancement toward AGI. Summary of works done on RFT for MLLMs is available at https://github.com/Sun-Haoyuan23/Awesome-RL-based-Reasoning-MLLMs.
>
---
#### [new 268] Generative RLHF-V: Learning Principles from Multi-modal Human Preference
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLMs）对齐任务，旨在解决传统奖励模型精度低、泛化差及可解释性弱的问题。提出Generative RLHF-V框架，结合生成式奖励模型（GRM）与多模态RLHF，通过RL引导GRM捕获人类意图并评分，再利用分组响应对比优化RL精度，提升7项基准测试性能18.1%。**

- **链接: [http://arxiv.org/pdf/2505.18531v1](http://arxiv.org/pdf/2505.18531v1)**

> **作者:** Jiayi Zhou; Jiaming Ji; Boyuan Chen; Jiapeng Sun; Wenqi Chen; Donghai Hong; Sirui Han; Yike Guo; Yaodong Yang
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Training multi-modal large language models (MLLMs) that align with human intentions is a long-term challenge. Traditional score-only reward models for alignment suffer from low accuracy, weak generalization, and poor interpretability, blocking the progress of alignment methods, e.g., reinforcement learning from human feedback (RLHF). Generative reward models (GRMs) leverage MLLMs' intrinsic reasoning capabilities to discriminate pair-wise responses, but their pair-wise paradigm makes it hard to generalize to learnable rewards. We introduce Generative RLHF-V, a novel alignment framework that integrates GRMs with multi-modal RLHF. We propose a two-stage pipeline: $\textbf{multi-modal generative reward modeling from RL}$, where RL guides GRMs to actively capture human intention, then predict the correct pair-wise scores; and $\textbf{RL optimization from grouped comparison}$, which enhances multi-modal RL scoring precision by grouped responses comparison. Experimental results demonstrate that, besides out-of-distribution generalization of RM discrimination, our framework improves 4 MLLMs' performance across 7 benchmarks by $18.1\%$, while the baseline RLHF is only $5.3\%$. We further validate that Generative RLHF-V achieves a near-linear improvement with an increasing number of candidate responses. Our code and models can be found at https://generative-rlhf-v.github.io.
>
---
#### [new 269] Towards Video to Piano Music Generation with Chain-of-Perform Support Benchmarks
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于视频到钢琴音乐生成任务，解决现有数据集无法准确评估音画同步复杂性的问题。提出CoP基准数据集，含多模态标注与评估框架，并开源促进研究。**

- **链接: [http://arxiv.org/pdf/2505.20038v1](http://arxiv.org/pdf/2505.20038v1)**

> **作者:** Chang Liu; Haomin Zhang; Shiyu Xia; Zihao Chen; Chaofan Ding; Xin Yue; Huizhe Chen; Xinhan Di
>
> **备注:** 4 pages, 1 figure, accepted by CVPR 2025 MMFM Workshop
>
> **摘要:** Generating high-quality piano audio from video requires precise synchronization between visual cues and musical output, ensuring accurate semantic and temporal alignment.However, existing evaluation datasets do not fully capture the intricate synchronization required for piano music generation. A comprehensive benchmark is essential for two primary reasons: (1) existing metrics fail to reflect the complexity of video-to-piano music interactions, and (2) a dedicated benchmark dataset can provide valuable insights to accelerate progress in high-quality piano music generation. To address these challenges, we introduce the CoP Benchmark Dataset-a fully open-sourced, multimodal benchmark designed specifically for video-guided piano music generation. The proposed Chain-of-Perform (CoP) benchmark offers several compelling features: (1) detailed multimodal annotations, enabling precise semantic and temporal alignment between video content and piano audio via step-by-step Chain-of-Perform guidance; (2) a versatile evaluation framework for rigorous assessment of both general-purpose and specialized video-to-piano generation tasks; and (3) full open-sourcing of the dataset, annotations, and evaluation protocols. The dataset is publicly available at https://github.com/acappemin/Video-to-Audio-and-Piano, with a continuously updated leaderboard to promote ongoing research in this domain.
>
---
#### [new 270] Tropical Geometry Based Edge Detection Using Min-Plus and Max-Plus Algebra
- **分类: math.AG; cs.CV; 14T90, 14-04**

- **简介: 该论文提出基于热带几何的边缘检测框架，利用Min-Plus/Max-Plus代数重构卷积与梯度运算，解决传统方法在低对比度和纹理区域边缘模糊的问题。通过多尺度处理、Hessian滤波等技术结合经典算子（如Canny），增强边缘连续性与清晰度，实验验证其在图像分析中的有效性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18625v1](http://arxiv.org/pdf/2505.18625v1)**

> **作者:** Shivam Kumar Jha S; Jaya NN Iyer
>
> **摘要:** This paper proposes a tropical geometry-based edge detection framework that reformulates convolution and gradient computations using min-plus and max-plus algebra. The tropical formulation emphasizes dominant intensity variations, contributing to sharper and more continuous edge representations. Three variants are explored: an adaptive threshold-based method, a multi-kernel min-plus method, and a max-plus method emphasizing structural continuity. The framework integrates multi-scale processing, Hessian filtering, and wavelet shrinkage to enhance edge transitions while maintaining computational efficiency. Experiments on MATLAB built-in grayscale and color images suggest that tropical formulations integrated with classical operators, such as Canny and LoG, can improve boundary detection in low-contrast and textured regions. Quantitative evaluation using standard edge metrics indicates favorable edge clarity and structural coherence. These results highlight the potential of tropical algebra as a scalable and noise-aware formulation for edge detection in practical image analysis tasks.
>
---
#### [new 271] Shifting AI Efficiency From Model-Centric to Data-Centric Compression
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出数据压缩（token压缩）作为AI效率新方向，解决模型规模硬件限制及自注意力长序列计算瓶颈。通过建立统一框架分析现有方法，论证token压缩优势，探讨挑战与未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.19147v1](http://arxiv.org/pdf/2505.19147v1)**

> **作者:** Xuyang Liu; Zichen Wen; Shaobo Wang; Junjie Chen; Zhishan Tao; Yubo Wang; Xiangqi Jin; Chang Zou; Yiyu Wang; Chenfei Liao; Xu Zheng; Honggang Chen; Weijia Li; Xuming Hu; Conghui He; Linfeng Zhang
>
> **备注:** Project: \url{https://github.com/xuyang-liu16/Awesome-Token-level-Model-Compression}
>
> **摘要:** The rapid advancement of large language models (LLMs) and multi-modal LLMs (MLLMs) has historically relied on model-centric scaling through increasing parameter counts from millions to hundreds of billions to drive performance gains. However, as we approach hardware limits on model size, the dominant computational bottleneck has fundamentally shifted to the quadratic cost of self-attention over long token sequences, now driven by ultra-long text contexts, high-resolution images, and extended videos. In this position paper, \textbf{we argue that the focus of research for efficient AI is shifting from model-centric compression to data-centric compression}. We position token compression as the new frontier, which improves AI efficiency via reducing the number of tokens during model training or inference. Through comprehensive analysis, we first examine recent developments in long-context AI across various domains and establish a unified mathematical framework for existing model efficiency strategies, demonstrating why token compression represents a crucial paradigm shift in addressing long-context overhead. Subsequently, we systematically review the research landscape of token compression, analyzing its fundamental benefits and identifying its compelling advantages across diverse scenarios. Furthermore, we provide an in-depth analysis of current challenges in token compression research and outline promising future directions. Ultimately, our work aims to offer a fresh perspective on AI efficiency, synthesize existing research, and catalyze innovative developments to address the challenges that increasing context lengths pose to the AI community's advancement.
>
---
#### [new 272] STRICT: Stress Test of Rendering Images Containing Text
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出STRICT基准，评估扩散模型生成图像中文字的性能，解决其在文本一致性、可读性及指令遵循上的不足。通过测试最长可读文本长度、文本准确性及指令遵守率，揭示模型局限，推动多模态生成研究。**

- **链接: [http://arxiv.org/pdf/2505.18985v1](http://arxiv.org/pdf/2505.18985v1)**

> **作者:** Tianyu Zhang; Xinyu Wang; Zhenghan Tai; Lu Li; Jijun Chi; Jingrui Tian; Hailin He; Suyuchen Wang
>
> **备注:** 13 pages
>
> **摘要:** While diffusion models have revolutionized text-to-image generation with their ability to synthesize realistic and diverse scenes, they continue to struggle to generate consistent and legible text within images. This shortcoming is commonly attributed to the locality bias inherent in diffusion-based generation, which limits their ability to model long-range spatial dependencies. In this paper, we introduce $\textbf{STRICT}$, a benchmark designed to systematically stress-test the ability of diffusion models to render coherent and instruction-aligned text in images. Our benchmark evaluates models across multiple dimensions: (1) the maximum length of readable text that can be generated; (2) the correctness and legibility of the generated text, and (3) the ratio of not following instructions for generating text. We evaluate several state-of-the-art models, including proprietary and open-source variants, and reveal persistent limitations in long-range consistency and instruction-following capabilities. Our findings provide insights into architectural bottlenecks and motivate future research directions in multimodal generative modeling. We release our entire evaluation pipeline at https://github.com/tianyu-z/STRICT-Bench.
>
---
#### [new 273] How to build a consistency model: Learning flow maps via self-distillation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成模型优化任务，旨在提升一致性模型效率并消除预训练需求。通过自蒸馏学习流映射，提出将传统蒸馏方案转化为直接训练算法的方法，并针对不同任务设计目标函数（高维任务避免时空导数，低维任务引入高阶导数），实验证明其有效性。**

- **链接: [http://arxiv.org/pdf/2505.18825v1](http://arxiv.org/pdf/2505.18825v1)**

> **作者:** Nicholas M. Boffi; Michael S. Albergo; Eric Vanden-Eijnden
>
> **摘要:** Building on the framework proposed in Boffi et al. (2024), we present a systematic approach for learning flow maps associated with flow and diffusion models. Flow map-based models, commonly known as consistency models, encompass recent efforts to improve the efficiency of generative models based on solutions to differential equations. By exploiting a relationship between the velocity field underlying a continuous-time flow and the instantaneous rate of change of the flow map, we show how to convert existing distillation schemes into direct training algorithms via self-distillation, eliminating the need for pre-trained models. We empirically evaluate several instantiations of our framework, finding that high-dimensional tasks like image synthesis benefit from objective functions that avoid temporal and spatial derivatives of the flow map, while lower-dimensional tasks can benefit from objectives incorporating higher-order derivatives to capture sharp features.
>
---
#### [new 274] Doc-CoB: Enhancing Multi-Modal Document Understanding with Visual Chain-of-Boxes Reasoning
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态文档理解任务，解决现有模型处理整图导致关键区域聚焦不足的问题。提出Doc-CoB机制，通过自主选择相关区域并聚焦，结合布局分析生成带监督数据，添加框识别与推理任务提升效果，实验验证方法有效。**

- **链接: [http://arxiv.org/pdf/2505.18603v1](http://arxiv.org/pdf/2505.18603v1)**

> **作者:** Ye Mo; Zirui Shao; Kai Ye; Xianwei Mao; Bo Zhang; Hangdi Xing; Peng Ye; Gang Huang; Kehan Chen; Zhou Huan; Zixu Yan; Sheng Zhou
>
> **摘要:** Multimodal large language models (MLLMs) have made significant progress in document understanding. However, the information-dense nature of document images still poses challenges, as most queries depend on only a few relevant regions, with the rest being redundant. Existing one-pass MLLMs process entire document images without considering query relevance, often failing to focus on critical regions and producing unfaithful responses. Inspired by the human coarse-to-fine reading pattern, we introduce Doc-CoB (Chain-of-Box), a simple-yet-effective mechanism that integrates human-style visual reasoning into MLLM without modifying its architecture. Our method allows the model to autonomously select the set of regions (boxes) most relevant to the query, and then focus attention on them for further understanding. We first design a fully automatic pipeline, integrating a commercial MLLM with a layout analyzer, to generate 249k training samples with intermediate visual reasoning supervision. Then we incorporate two enabling tasks that improve box identification and box-query reasoning, which together enhance document understanding. Extensive experiments on seven benchmarks with four popular models show that Doc-CoB significantly improves performance, demonstrating its effectiveness and wide applicability. All code, data, and models will be released publicly.
>
---
#### [new 275] Brightness-Invariant Tracking Estimation in Tagged MRI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决MRI标签成像中因亮度变化导致的运动追踪误差问题。提出BRITE方法，结合扩散概率模型分离解剖结构与标签图案，并利用物理信息神经网络估计生物合理运动，提升追踪精度且抗标签褪色。**

- **链接: [http://arxiv.org/pdf/2505.18365v1](http://arxiv.org/pdf/2505.18365v1)**

> **作者:** Zhangxing Bian; Shuwen Wei; Xiao Liang; Yuan-Chiao Lu; Samuel W. Remedios; Fangxu Xing; Jonghye Woo; Dzung L. Pham; Aaron Carass; Philip V. Bayly; Jiachen Zhuo; Ahmed Alshareef; Jerry L. Prince
>
> **备注:** Accepted by IPMI 2025
>
> **摘要:** Magnetic resonance (MR) tagging is an imaging technique for noninvasively tracking tissue motion in vivo by creating a visible pattern of magnetization saturation (tags) that deforms with the tissue. Due to longitudinal relaxation and progression to steady-state, the tags and tissue brightnesses change over time, which makes tracking with optical flow methods error-prone. Although Fourier methods can alleviate these problems, they are also sensitive to brightness changes as well as spectral spreading due to motion. To address these problems, we introduce the brightness-invariant tracking estimation (BRITE) technique for tagged MRI. BRITE disentangles the anatomy from the tag pattern in the observed tagged image sequence and simultaneously estimates the Lagrangian motion. The inherent ill-posedness of this problem is addressed by leveraging the expressive power of denoising diffusion probabilistic models to represent the probabilistic distribution of the underlying anatomy and the flexibility of physics-informed neural networks to estimate biologically-plausible motion. A set of tagged MR images of a gel phantom was acquired with various tag periods and imaging flip angles to demonstrate the impact of brightness variations and to validate our method. The results show that BRITE achieves more accurate motion and strain estimates as compared to other state of the art methods, while also being resistant to tag fading.
>
---
#### [new 276] MangaVQA and MangaLMM: A Benchmark and Specialized Model for Multimodal Manga Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态漫画理解任务，旨在解决现有模型对日漫复杂图文叙事理解不足的问题。提出MangaOCR（文本识别基准）和MangaVQA（视觉问答基准），并开发基于Qwen2.5-VL的专用模型MangaLMM，通过实验评估模型性能，为漫画领域多模态研究提供基础。**

- **链接: [http://arxiv.org/pdf/2505.20298v1](http://arxiv.org/pdf/2505.20298v1)**

> **作者:** Jeonghun Baek; Kazuki Egashira; Shota Onohara; Atsuyuki Miyai; Yuki Imajuku; Hikaru Ikuta; Kiyoharu Aizawa
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Manga, or Japanese comics, is a richly multimodal narrative form that blends images and text in complex ways. Teaching large multimodal models (LMMs) to understand such narratives at a human-like level could help manga creators reflect on and refine their stories. To this end, we introduce two benchmarks for multimodal manga understanding: MangaOCR, which targets in-page text recognition, and MangaVQA, a novel benchmark designed to evaluate contextual understanding through visual question answering. MangaVQA consists of 526 high-quality, manually constructed question-answer pairs, enabling reliable evaluation across diverse narrative and visual scenarios. Building on these benchmarks, we develop MangaLMM, a manga-specialized model finetuned from the open-source LMM Qwen2.5-VL to jointly handle both tasks. Through extensive experiments, including comparisons with proprietary models such as GPT-4o and Gemini 2.5, we assess how well LMMs understand manga. Our benchmark and model provide a comprehensive foundation for evaluating and advancing LMMs in the richly narrative domain of manga.
>
---
#### [new 277] Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对文本到多视角生成任务，解决少步扩散模型加速时图像保真度与视角一致性下降的问题。提出RL微调框架MVC-ZigAL，包含多视角统一决策过程、ZMV-Sampling推理技术、MV-ZigAL优化策略及约束优化，平衡保真度与一致性，提升生成质量同时保持高效。**

- **链接: [http://arxiv.org/pdf/2505.20107v1](http://arxiv.org/pdf/2505.20107v1)**

> **作者:** Ziyi Zhang; Li Shen; Deheng Ye; Yong Luo; Huangxuan Zhao; Lefei Zhang
>
> **摘要:** Text-to-multiview (T2MV) generation, which produces coherent multiview images from a single text prompt, remains computationally intensive, while accelerated T2MV methods using few-step diffusion models often sacrifice image fidelity and view consistency. To address this, we propose a novel reinforcement learning (RL) finetuning framework tailored for few-step T2MV diffusion models to jointly optimize per-view fidelity and cross-view consistency. Specifically, we first reformulate T2MV denoising across all views as a single unified Markov decision process, enabling multiview-aware policy optimization driven by a joint-view reward objective. Next, we introduce ZMV-Sampling, a test-time T2MV sampling technique that adds an inversion-denoising pass to reinforce both viewpoint and text conditioning, resulting in improved T2MV generation at the cost of inference time. To internalize its performance gains into the base sampling policy, we develop MV-ZigAL, a novel policy optimization strategy that uses reward advantages of ZMV-Sampling over standard sampling as learning signals for policy updates. Finally, noting that the joint-view reward objective under-optimizes per-view fidelity but naively optimizing single-view metrics neglects cross-view alignment, we reframe RL finetuning for T2MV diffusion models as a constrained optimization problem that maximizes per-view fidelity subject to an explicit joint-view constraint, thereby enabling more efficient and balanced policy updates. By integrating this constrained optimization paradigm with MV-ZigAL, we establish our complete RL finetuning framework, referred to as MVC-ZigAL, which effectively refines the few-step T2MV diffusion baseline in both fidelity and consistency while preserving its few-step efficiency.
>
---
#### [new 278] LatentLLM: Attention-Aware Joint Tensor Compression
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属模型压缩任务，针对大语言/多模态模型资源消耗高的问题，提出LatentLLM框架，通过全局注意力感知联合张量分解构建降维潜在结构，提升压缩后模型精度，实现计算/内存效率优化，适用于多模态推理等场景。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18413v1](http://arxiv.org/pdf/2505.18413v1)**

> **作者:** Toshiaki Koike-Akino; Xiangyu Chen; Jing Liu; Ye Wang; Pu; Wang; Matthew Brand
>
> **备注:** 37 pages, 16 figures
>
> **摘要:** Modern foundation models such as large language models (LLMs) and large multi-modal models (LMMs) require a massive amount of computational and memory resources. We propose a new framework to convert such LLMs/LMMs into a reduced-dimension latent structure. Our method extends a local activation-aware tensor decomposition to a global attention-aware joint tensor de-composition. Our framework can significantly improve the model accuracy over the existing model compression methods when reducing the latent dimension to realize computationally/memory-efficient LLMs/LLMs. We show the benefit on several benchmark including multi-modal reasoning tasks.
>
---
#### [new 279] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出CoreMatching框架，针对视觉语言模型（VLMs）推理效率低的问题。通过揭示核心token与神经元间的协同作用，整合两者的稀疏化策略，实现计算加速。实验显示其在多任务和硬件上优于现有方法，如Titan Xp上速度提升10倍。**

- **链接: [http://arxiv.org/pdf/2505.19235v1](http://arxiv.org/pdf/2505.19235v1)**

> **作者:** Qinsi Wang; Hancheng Ye; Ming-Yu Chung; Yudong Liu; Yueqian Lin; Martin Kuo; Mingyuan Ma; Jianyi Zhang; Yiran Chen
>
> **备注:** ICML 2025
>
> **摘要:** Vision-Language Models (VLMs) excel across diverse tasks but suffer from high inference costs in time and memory. Token sparsity mitigates inefficiencies in token usage, while neuron sparsity reduces high-dimensional computations, both offering promising solutions to enhance efficiency. Recently, these two sparsity paradigms have evolved largely in parallel, fostering the prevailing assumption that they function independently. However, a fundamental yet underexplored question remains: Do they truly operate in isolation, or is there a deeper underlying interplay that has yet to be uncovered? In this paper, we conduct the first comprehensive investigation into this question. By introducing and analyzing the matching mechanism between Core Neurons and Core Tokens, we found that key neurons and tokens for inference mutually influence and reinforce each other. Building on this insight, we propose CoreMatching, a co-adaptive sparse inference framework, which leverages the synergy between token and neuron sparsity to enhance inference efficiency. Through theoretical analysis and efficiency evaluations, we demonstrate that the proposed method surpasses state-of-the-art baselines on ten image understanding tasks and three hardware devices. Notably, on the NVIDIA Titan Xp, it achieved 5x FLOPs reduction and a 10x overall speedup. Code is released at https://github.com/wangqinsi1/2025-ICML-CoreMatching/tree/main.
>
---
#### [new 280] ICDM: Interference Cancellation Diffusion Models for Wireless Semantic Communications
- **分类: cs.IT; cs.AI; cs.CV; math.IT**

- **简介: 该论文提出ICDM模型，针对无线语义通信中干扰消除任务，解决扩散模型处理信号与干扰共存的难题。通过MAP估计分解信号、干扰先验及信道概率，用扩散模型学习梯度并结合迭代算法，实验证明可显著降低MSE并提升感知质量。**

- **链接: [http://arxiv.org/pdf/2505.19983v1](http://arxiv.org/pdf/2505.19983v1)**

> **作者:** Tong Wu; Zhiyong Chen; Dazhi He; Feng Yang; Meixia Tao; Xiaodong Xu; Wenjun Zhang; Ping Zhang
>
> **备注:** submitted to IEEE journal
>
> **摘要:** Diffusion models (DMs) have recently achieved significant success in wireless communications systems due to their denoising capabilities. The broadcast nature of wireless signals makes them susceptible not only to Gaussian noise, but also to unaware interference. This raises the question of whether DMs can effectively mitigate interference in wireless semantic communication systems. In this paper, we model the interference cancellation problem as a maximum a posteriori (MAP) problem over the joint posterior probability of the signal and interference, and theoretically prove that the solution provides excellent estimates for the signal and interference. To solve this problem, we develop an interference cancellation diffusion model (ICDM), which decomposes the joint posterior into independent prior probabilities of the signal and interference, along with the channel transition probablity. The log-gradients of these distributions at each time step are learned separately by DMs and accurately estimated through deriving. ICDM further integrates these gradients with advanced numerical iteration method, achieving accurate and rapid interference cancellation. Extensive experiments demonstrate that ICDM significantly reduces the mean square error (MSE) and enhances perceptual quality compared to schemes without ICDM. For example, on the CelebA dataset under the Rayleigh fading channel with a signal-to-noise ratio (SNR) of $20$ dB and signal to interference plus noise ratio (SINR) of 0 dB, ICDM reduces the MSE by 4.54 dB and improves the learned perceptual image patch similarity (LPIPS) by 2.47 dB.
>
---
#### [new 281] Evaluation in EEG Emotion Recognition: State-of-the-Art Review and Unified Framework
- **分类: eess.SP; cs.AI; cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于EEG情绪识别评估方法研究，旨在解决领域内缺乏统一评估协议的问题。通过分析216篇论文，揭示了现有评估中的不一致性（如ground truth定义、指标选择、数据划分方式等），并提出开源框架EEGain，标准化预处理、数据划分、评估指标及数据集加载，支持六大数据集和四种常用方法验证，推动领域可重复性与可比性。**

- **链接: [http://arxiv.org/pdf/2505.18175v1](http://arxiv.org/pdf/2505.18175v1)**

> **作者:** Natia Kukhilava; Tatia Tsmindashvili; Rapael Kalandadze; Anchit Gupta; Sofio Katamadze; François Brémond; Laura M. Ferrari; Philipp Müller; Benedikt Emanuel Wirth
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Electroencephalography-based Emotion Recognition (EEG-ER) has become a growing research area in recent years. Analyzing 216 papers published between 2018 and 2023, we uncover that the field lacks a unified evaluation protocol, which is essential to fairly define the state of the art, compare new approaches and to track the field's progress. We report the main inconsistencies between the used evaluation protocols, which are related to ground truth definition, evaluation metric selection, data splitting types (e.g., subject-dependent or subject-independent) and the use of different datasets. Capitalizing on this state-of-the-art research, we propose a unified evaluation protocol, EEGain (https://github.com/EmotionLab/EEGain), which enables an easy and efficient evaluation of new methods and datasets. EEGain is a novel open source software framework, offering the capability to compare - and thus define - state-of-the-art results. EEGain includes standardized methods for data pre-processing, data splitting, evaluation metrics, and the ability to load the six most relevant datasets (i.e., AMIGOS, DEAP, DREAMER, MAHNOB-HCI, SEED, SEED-IV) in EEG-ER with only a single line of code. In addition, we have assessed and validated EEGain using these six datasets on the four most common publicly available methods (EEGNet, DeepConvNet, ShallowConvNet, TSception). This is a significant step to make research on EEG-ER more reproducible and comparable, thereby accelerating the overall progress of the field.
>
---
#### [new 282] Multimodal Federated Learning With Missing Modalities through Feature Imputation Network
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对医疗多模态联邦学习中因模态缺失导致的模型性能下降问题，提出轻量级特征翻译网络重建缺失模态的瓶颈特征，通过三个医疗数据集实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.20232v1](http://arxiv.org/pdf/2505.20232v1)**

> **作者:** Pranav Poudel; Aavash Chhetri; Prashnna Gyawali; Georgios Leontidis; Binod Bhattarai
>
> **备注:** MIUA 2025
>
> **摘要:** Multimodal federated learning holds immense potential for collaboratively training models from multiple sources without sharing raw data, addressing both data scarcity and privacy concerns, two key challenges in healthcare. A major challenge in training multimodal federated models in healthcare is the presence of missing modalities due to multiple reasons, including variations in clinical practice, cost and accessibility constraints, retrospective data collection, privacy concerns, and occasional technical or human errors. Previous methods typically rely on publicly available real datasets or synthetic data to compensate for missing modalities. However, obtaining real datasets for every disease is impractical, and training generative models to synthesize missing modalities is computationally expensive and prone to errors due to the high dimensionality of medical data. In this paper, we propose a novel, lightweight, low-dimensional feature translator to reconstruct bottleneck features of the missing modalities. Our experiments on three different datasets (MIMIC-CXR, NIH Open-I, and CheXpert), in both homogeneous and heterogeneous settings consistently improve the performance of competitive baselines. The code and implementation details are available at: https://github.com/bhattarailab/FedFeatGen
>
---
#### [new 283] LORE: Lagrangian-Optimized Robust Embeddings for Visual Encoders
- **分类: cs.LG; cs.AI; cs.CV; math.OC**

- **简介: 该论文属于视觉编码器对抗鲁棒性优化任务。针对现有方法对抗训练不稳定及鲁棒性与清洁数据准确率权衡问题，提出LORE框架，利用拉格朗日约束优化嵌入空间邻近性，平衡两者目标，显著提升零样本对抗鲁棒性，同时减少对清洁数据准确率的损害。**

- **链接: [http://arxiv.org/pdf/2505.18884v1](http://arxiv.org/pdf/2505.18884v1)**

> **作者:** Borna Khodabandeh; Amirabbas Afzali; Amirhossein Afsharrad; Seyed Shahabeddin Mousavi; Sanjay Lall; Sajjad Amini; Seyed-Mohsen Moosavi-Dezfooli
>
> **摘要:** Visual encoders have become fundamental components in modern computer vision pipelines. However, ensuring robustness against adversarial perturbations remains a critical challenge. Recent efforts have explored both supervised and unsupervised adversarial fine-tuning strategies. We identify two key limitations in these approaches: (i) they often suffer from instability, especially during the early stages of fine-tuning, resulting in suboptimal convergence and degraded performance on clean data, and (ii) they exhibit a suboptimal trade-off between robustness and clean data accuracy, hindering the simultaneous optimization of both objectives. To overcome these challenges, we propose Lagrangian-Optimized Robust Embeddings (LORE), a novel unsupervised adversarial fine-tuning framework. LORE utilizes constrained optimization, which offers a principled approach to balancing competing goals, such as improving robustness while preserving nominal performance. By enforcing embedding-space proximity constraints, LORE effectively maintains clean data performance throughout adversarial fine-tuning. Extensive experiments show that LORE significantly improves zero-shot adversarial robustness with minimal degradation in clean data accuracy. Furthermore, we demonstrate the effectiveness of the adversarially fine-tuned CLIP image encoder in out-of-distribution generalization and enhancing the interpretability of image embeddings.
>
---
#### [new 284] Diversity-Driven Generative Dataset Distillation Based on Diffusion Model with Self-Adaptive Memory
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于数据集蒸馏任务，解决生成数据集分布多样性不足导致性能下降的问题。提出基于扩散模型的多样性驱动方法，通过自适应内存对齐蒸馏数据与原始数据分布，提升生成数据的多样性，在实验中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19469v1](http://arxiv.org/pdf/2505.19469v1)**

> **作者:** Mingzhuo Li; Guang Li; Jiafeng Mao; Takahiro Ogawa; Miki Haseyama
>
> **备注:** Accepted by ICIP 2025
>
> **摘要:** Dataset distillation enables the training of deep neural networks with comparable performance in significantly reduced time by compressing large datasets into small and representative ones. Although the introduction of generative models has made great achievements in this field, the distributions of their distilled datasets are not diverse enough to represent the original ones, leading to a decrease in downstream validation accuracy. In this paper, we present a diversity-driven generative dataset distillation method based on a diffusion model to solve this problem. We introduce self-adaptive memory to align the distribution between distilled and real datasets, assessing the representativeness. The degree of alignment leads the diffusion model to generate more diverse datasets during the distillation process. Extensive experiments show that our method outperforms existing state-of-the-art methods in most situations, proving its ability to tackle dataset distillation tasks.
>
---
#### [new 285] Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多模态模型优化任务，旨在解决模态干扰问题，即模型在处理任务时无法有效区分相关与无关模态信息导致性能下降。研究提出通过扰动基诊断实验量化干扰，并设计结合对抗性数据增强（如PGD）和一致性正则化的微调框架，提升模型单模态推理与跨模态任务的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.19616v1](http://arxiv.org/pdf/2505.19616v1)**

> **作者:** Rui Cai; Bangzheng Li; Xiaofei Wen; Muhao Chen; Zhe Zhao
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across tasks, yet they often exhibit difficulty in distinguishing task-relevant from irrelevant signals, particularly in tasks like Visual Question Answering (VQA), which can lead to susceptibility to misleading or spurious inputs. We refer to this broader limitation as the Cross-Modality Competency Problem: the model's inability to fairly evaluate all modalities. This vulnerability becomes more evident in modality-specific tasks such as image classification or pure text question answering, where models are expected to rely solely on one modality. In such tasks, spurious information from irrelevant modalities often leads to significant performance degradation. We refer to this failure as Modality Interference, which serves as a concrete and measurable instance of the cross-modality competency problem. We further design a perturbation-based causal diagnostic experiment to verify and quantify this problem. To mitigate modality interference, we propose a novel framework to fine-tune MLLMs, including perturbation-based data augmentations with both heuristic perturbations and adversarial perturbations via Projected Gradient Descent (PGD), and a consistency regularization strategy applied to model outputs with original and perturbed inputs. Experiments on multiple benchmark datasets (image-heavy, text-heavy, and VQA tasks) and multiple model families with different scales demonstrate significant improvements in robustness and cross-modality competency, indicating our method's effectiveness in boosting unimodal reasoning ability while enhancing performance on multimodal tasks.
>
---
#### [new 286] GC-KBVQA: A New Four-Stage Framework for Enhancing Knowledge Based Visual Question Answering Performance
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于基于知识的视觉问答（KB-VQA）任务，旨在解决现有方法辅助信息不相关或误导的问题。提出四阶段框架GC-KBVQA，通过生成问题相关的紧凑图像描述并结合外部知识，为LLM构建有效提示，实现零样本VQA，无需任务微调，提升性能。**

- **链接: [http://arxiv.org/pdf/2505.19354v1](http://arxiv.org/pdf/2505.19354v1)**

> **作者:** Mohammad Mahdi Moradi; Sudhir Mudur
>
> **摘要:** Knowledge-Based Visual Question Answering (KB-VQA) methods focus on tasks that demand reasoning with information extending beyond the explicit content depicted in the image. Early methods relied on explicit knowledge bases to provide this auxiliary information. Recent approaches leverage Large Language Models (LLMs) as implicit knowledge sources. While KB-VQA methods have demonstrated promising results, their potential remains constrained as the auxiliary text provided may not be relevant to the question context, and may also include irrelevant information that could misguide the answer predictor. We introduce a novel four-stage framework called Grounding Caption-Guided Knowledge-Based Visual Question Answering (GC-KBVQA), which enables LLMs to effectively perform zero-shot VQA tasks without the need for end-to-end multimodal training. Innovations include grounding question-aware caption generation to move beyond generic descriptions and have compact, yet detailed and context-rich information. This is combined with knowledge from external sources to create highly informative prompts for the LLM. GC-KBVQA can address a variety of VQA tasks, and does not require task-specific fine-tuning, thus reducing both costs and deployment complexity by leveraging general-purpose, pre-trained LLMs. Comparison with competing KB-VQA methods shows significantly improved performance. Our code will be made public.
>
---
#### [new 287] A Contrastive Learning Foundation Model Based on Perfectly Aligned Sample Pairs for Remote Sensing Images
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出PerA方法，属于自监督对比学习预训练遥感基础模型任务。针对遥感领域对比学习因域差距效果受限的问题，其通过空间分离掩码生成语义对齐但外观差异的样本对，并优化教师-学生模型一致性，提升特征质量与内存效率。实验表明其在有限模型规模下性能优异，推动遥感图像分析应用。**

- **链接: [http://arxiv.org/pdf/2505.19447v1](http://arxiv.org/pdf/2505.19447v1)**

> **作者:** Hengtong Shen; Haiyan Gu; Haitao Li; Yi Yang; Agen qiu
>
> **摘要:** Self-Supervised Learning (SSL) enables us to pre-train foundation models without costly labeled data. Among SSL methods, Contrastive Learning (CL) methods are better at obtaining accurate semantic representations in noise interference. However, due to the significant domain gap, while CL methods have achieved great success in many computer vision tasks, they still require specific adaptation for Remote Sensing (RS) images. To this end, we present a novel self-supervised method called PerA, which produces all-purpose RS features through semantically Perfectly Aligned sample pairs. Specifically, PerA obtains features from sampled views by applying spatially disjoint masks to augmented images rather than random cropping. With disjoint masks, we divide patches from different views into different parts that are semantically aligned but inconsistent in appearance. Our framework provides high-quality features by ensuring consistency between teacher and student and predicting learnable mask tokens. Compared to previous contrastive methods, our method demonstrates higher memory efficiency and can be trained with larger batches due to its sparse inputs. We also collect an unlabeled pre-training dataset, which contains about 5 million RS images. We conducted experiments on multiple downstream task datasets and achieved performance comparable to previous state-of-the-art methods with a limited model scale, which verified the superiority of our method. We hope this work will contribute to practical remote sensing interpretation works.
>
---
#### [new 288] AI- Enhanced Stethoscope in Remote Diagnostics for Cardiopulmonary Diseases
- **分类: eess.SP; cs.CV**

- **简介: 论文提出基于AI的低成本听诊器模型，结合MFCC特征提取与GRU-CNN混合网络，分析心肺听诊音频，实时诊断六类肺病及五类心脏病，解决偏远地区医疗资源不足导致的诊断延迟问题。**

- **链接: [http://arxiv.org/pdf/2505.18184v1](http://arxiv.org/pdf/2505.18184v1)**

> **作者:** Hania Ghouse; Juveria Tanveen; Abdul Muqtadir Ahmed; Uma N. Dulhare
>
> **摘要:** The increase in cardiac and pulmonary diseases presents an alarming and pervasive health challenge on a global scale responsible for unexpected and premature mortalities. In spite of how serious these conditions are, existing methods of detection and treatment encounter challenges, particularly in achieving timely diagnosis for effective medical intervention. Manual screening processes commonly used for primary detection of cardiac and respiratory problems face inherent limitations, increased by a scarcity of skilled medical practitioners in remote or under-resourced areas. To address this, our study introduces an innovative yet efficient model which integrates AI for diagnosing lung and heart conditions concurrently using the auscultation sounds. Unlike the already high-priced digital stethoscope, our proposed model has been particularly designed to deploy on low-cost embedded devices and thus ensure applicability in under-developed regions that actually face an issue of accessing medical care. Our proposed model incorporates MFCC feature extraction and engineering techniques to ensure that the signal is well analyzed for accurate diagnostics through the hybrid model combining Gated Recurrent Unit with CNN in processing audio signals recorded from the low-cost stethoscope. Beyond its diagnostic capabilities, the model generates digital audio records that facilitate in classifying six pulmonary and five cardiovascular diseases. Hence, the integration of a cost effective stethoscope with an efficient AI empowered model deployed on a web app providing real-time analysis, represents a transformative step towards standardized healthcare
>
---
#### [new 289] ReadBench: Measuring the Dense Text Visual Reading Ability of Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 论文提出ReadBench，评估视觉语言模型（VLMs）处理文本密集图像的阅读能力。针对现有基准未充分测试长文本视觉推理的问题，该工作将纯文本任务转为图像形式，发现模型在长文本或多页场景性能显著下降，揭示需改进视觉文本推理能力。**

- **链接: [http://arxiv.org/pdf/2505.19091v1](http://arxiv.org/pdf/2505.19091v1)**

> **作者:** Benjamin Clavié; Florian Brand
>
> **摘要:** Recent advancements in Large Vision-Language Models (VLMs), have greatly enhanced their capability to jointly process text and images. However, despite extensive benchmarks evaluating visual comprehension (e.g., diagrams, color schemes, OCR tasks...), there is limited assessment of VLMs' ability to read and reason about text-rich images effectively. To fill this gap, we introduce ReadBench, a multimodal benchmark specifically designed to evaluate the reading comprehension capabilities of VLMs. ReadBench transposes contexts from established text-only benchmarks into images of text while keeping textual prompts and questions intact. Evaluating leading VLMs with ReadBench, we find minimal-but-present performance degradation on short, text-image inputs, while performance sharply declines for longer, multi-page contexts. Our experiments further reveal that text resolution has negligible effects on multimodal performance. These findings highlight needed improvements in VLMs, particularly their reasoning over visually presented extensive textual content, a capability critical for practical applications. ReadBench is available at https://github.com/answerdotai/ReadBench .
>
---
#### [new 290] VerIPO: Cultivating Long Reasoning in Video-LLMs via Verifier-Gudied Iterative Policy Optimization
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视频大语言模型（Video-LLMs）长期推理任务，解决现有RL方法（如GRPO）因数据噪声/成本高导致长推理链质量不稳定的问题。提出VerIPO方法，通过Rollout-Aware Verifier构建GRPO-Verifier-DPO训练循环，利用小模型评估推理逻辑生成高质量对比数据，提升长推理链的长度与一致性，实验显示其效率与效果优于基线模型。**

- **链接: [http://arxiv.org/pdf/2505.19000v1](http://arxiv.org/pdf/2505.19000v1)**

> **作者:** Yunxin Li; Xinyu Chen; Zitao Li; Zhenyu Liu; Longyue Wang; Wenhan Luo; Baotian Hu; Min Zhang
>
> **备注:** 19 pages, 9 figures, Project Link: https://github.com/HITsz-TMG/VerIPO
>
> **摘要:** Applying Reinforcement Learning (RL) to Video Large Language Models (Video-LLMs) shows significant promise for complex video reasoning. However, popular Reinforcement Fine-Tuning (RFT) methods, such as outcome-based Group Relative Policy Optimization (GRPO), are limited by data preparation bottlenecks (e.g., noise or high cost) and exhibit unstable improvements in the quality of long chain-of-thoughts (CoTs) and downstream performance.To address these limitations, we propose VerIPO, a Verifier-guided Iterative Policy Optimization method designed to gradually improve video LLMs' capacity for generating deep, long-term reasoning chains. The core component is Rollout-Aware Verifier, positioned between the GRPO and Direct Preference Optimization (DPO) training phases to form the GRPO-Verifier-DPO training loop. This verifier leverages small LLMs as a judge to assess the reasoning logic of rollouts, enabling the construction of high-quality contrastive data, including reflective and contextually consistent CoTs. These curated preference samples drive the efficient DPO stage (7x faster than GRPO), leading to marked improvements in reasoning chain quality, especially in terms of length and contextual consistency. This training loop benefits from GRPO's expansive search and DPO's targeted optimization. Experimental results demonstrate: 1) Significantly faster and more effective optimization compared to standard GRPO variants, yielding superior performance; 2) Our trained models exceed the direct inference of large-scale instruction-tuned Video-LLMs, producing long and contextually consistent CoTs on diverse video reasoning tasks; and 3) Our model with one iteration outperforms powerful LMMs (e.g., Kimi-VL) and long reasoning models (e.g., Video-R1), highlighting its effectiveness and stability.
>
---
#### [new 291] RGC-Bent: A Novel Dataset for Bent Radio Galaxy Classification
- **分类: astro-ph.GA; cs.CV**

- **简介: 论文提出RGC-Bent数据集，用于分类弯曲射电星系中的NAT和WAT类型，解决现有数据稀缺及分类困难问题。通过处理射电天文观测数据构建数据集，并评估CNN和transformer模型，ConvNeXT表现最佳，推动AGN分类与星系演化研究。**

- **链接: [http://arxiv.org/pdf/2505.19249v1](http://arxiv.org/pdf/2505.19249v1)**

> **作者:** Mir Sazzat Hossain; Khan Muhammad Bin Asad; Payaswini Saikia; Adrita Khan; Md Akil Raihan Iftee; Rakibul Hasan Rajib; Arshad Momen; Md Ashraful Amin; Amin Ahsan Ali; AKM Mahbubur Rahman
>
> **备注:** 6 pages, 3 figures, 2 tables, Accepted In ICIP 2025
>
> **摘要:** We introduce a novel machine learning dataset tailored for the classification of bent radio active galactic nuclei (AGN) in astronomical observations. Bent radio AGN, distinguished by their curved jet structures, provide critical insights into galaxy cluster dynamics, interactions within the intracluster medium, and the broader physics of AGN. Despite their astrophysical significance, the classification of bent radio AGN remains a challenge due to the scarcity of specialized datasets and benchmarks. To address this, we present a dataset, derived from a well-recognized radio astronomy survey, that is designed to support the classification of NAT (Narrow-Angle Tail) and WAT (Wide-Angle Tail) categories, along with detailed data processing steps. We further evaluate the performance of state-of-the-art deep learning models on the dataset, including Convolutional Neural Networks (CNNs), and transformer-based architectures. Our results demonstrate the effectiveness of advanced machine learning models in classifying bent radio AGN, with ConvNeXT achieving the highest F1-scores for both NAT and WAT sources. By sharing this dataset and benchmarks, we aim to facilitate the advancement of research in AGN classification, galaxy cluster environments and galaxy evolution.
>
---
#### [new 292] WQLCP: Weighted Adaptive Conformal Prediction for Robust Uncertainty Quantification Under Distribution Shifts
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于分布偏移下的不确定性量化任务，解决传统符合性预测（CP）在数据分布变化时覆盖不足和预测集过大的问题。提出RLSCP方法，利用VAE重建损失调整得分函数；进一步提出WQLCP，通过加权分位损失动态调整校准阈值，提升鲁棒性，实验显示其在保持覆盖的同时减小预测集。**

- **链接: [http://arxiv.org/pdf/2505.19587v1](http://arxiv.org/pdf/2505.19587v1)**

> **作者:** Shadi Alijani; Homayoun Najjaran
>
> **摘要:** Conformal prediction (CP) provides a framework for constructing prediction sets with guaranteed coverage, assuming exchangeable data. However, real-world scenarios often involve distribution shifts that violate exchangeability, leading to unreliable coverage and inflated prediction sets. To address this challenge, we first introduce Reconstruction Loss-Scaled Conformal Prediction (RLSCP), which utilizes reconstruction losses derived from a Variational Autoencoder (VAE) as an uncertainty metric to scale score functions. While RLSCP demonstrates performance improvements, mainly resulting in better coverage, it quantifies quantiles based on a fixed calibration dataset without considering the discrepancies between test and train datasets in an unexchangeable setting. In the next step, we propose Weighted Quantile Loss-scaled Conformal Prediction (WQLCP), which refines RLSCP by incorporating a weighted notion of exchangeability, adjusting the calibration quantile threshold based on weights with respect to the ratio of calibration and test loss values. This approach improves the CP-generated prediction set outputs in the presence of distribution shifts. Experiments on large-scale datasets, including ImageNet variants, demonstrate that WQLCP outperforms existing baselines by consistently maintaining coverage while reducing prediction set sizes, providing a robust solution for CP under distribution shifts.
>
---
#### [new 293] Memory-Efficient Super-Resolution of 3D Micro-CT Images Using Octree-Based GANs: Enhancing Resolution and Segmentation Accuracy
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文提出基于Octree的GAN模型，解决3D微CT图像超分辨率与分割的内存瓶颈问题。通过Octree结构优化3D卷积层，结合未配对2D数据训练，实现16倍超分辨率（7→0.44μm/体素），提升矿物分割精度与孔隙表征，突破立方级内存限制。**

- **链接: [http://arxiv.org/pdf/2505.18664v1](http://arxiv.org/pdf/2505.18664v1)**

> **作者:** Evgeny Ugolkov; Xupeng He; Hyung Kwak; Hussein Hoteit
>
> **备注:** 31 pages, 15 figures
>
> **摘要:** We present a memory-efficient algorithm for significantly enhancing the quality of segmented 3D micro-Computed Tomography (micro-CT) images of rocks using a generative model. The proposed model achieves a 16x increase in resolution and corrects inaccuracies in segmentation caused by the overlapping X-ray attenuation in micro-CT measurements across different minerals. The generative model employed is a 3D Octree-based convolutional Wasserstein generative adversarial network with gradient penalty. To address the challenge of high memory consumption inherent in standard 3D convolutional layers, we implemented an Octree structure within the 3D progressive growing generator model. This enabled the use of memory-efficient 3D Octree-based convolutional layers. The approach is pivotal in overcoming the long-standing memory bottleneck in volumetric deep learning, making it possible to reach 16x super-resolution in 3D, a scale that is challenging to attain due to cubic memory scaling. For training, we utilized segmented 3D low-resolution micro-CT images along with unpaired segmented complementary 2D high-resolution laser scanning microscope images. Post-training, resolution improved from 7 to 0.44 micro-m/voxel with accurate segmentation of constituent minerals. Validated on Berea sandstone, this framework demonstrates substantial improvements in pore characterization and mineral differentiation, offering a robust solution to one of the primary computational limitations in modern geoscientific imaging.
>
---
#### [new 294] Improvement Strategies for Few-Shot Learning in OCT Image Classification of Rare Retinal Diseases
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对视网膜OCT图像分类中罕见与常见疾病类别样本不均衡问题，提出改进Few-Shot学习策略。采用GAN增强（U-GAT-IT）、数据平衡技术及CBAM注意力机制优化InceptionV3模型，实现97.85%分类准确率，显著提升罕见病诊断效果。**

- **链接: [http://arxiv.org/pdf/2505.20149v1](http://arxiv.org/pdf/2505.20149v1)**

> **作者:** Cheng-Yu Tai; Ching-Wen Chen; Chi-Chin Wu; Bo-Chen Chiu; Cheng-Hung; Lin; Cheng-Kai Lu; Jia-Kang Wang; Tzu-Lun Huang
>
> **摘要:** This paper focuses on using few-shot learning to improve the accuracy of classifying OCT diagnosis images with major and rare classes. We used the GAN-based augmentation strategy as a baseline and introduced several novel methods to further enhance our model. The proposed strategy contains U-GAT-IT for improving the generative part and uses the data balance technique to narrow down the skew of accuracy between all categories. The best model obtained was built with CBAM attention mechanism and fine-tuned InceptionV3, and achieved an overall accuracy of 97.85%, representing a significant improvement over the original baseline.
>
---
#### [new 295] From Single Images to Motion Policies via Video-Generation Environment Representations
- **分类: cs.RO; cs.CV; cs.GR; cs.LG**

- **简介: 该论文属于机器人运动规划任务，旨在从单张RGB图像生成符合环境几何结构的无碰撞运动策略。针对单目深度估计的截锥形误差问题，提出VGER框架：通过视频生成模型创建多视角视频，生成密集点云，训练多尺度环境表征及运动模型，实现单图驱动的几何感知路径规划。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19306v1](http://arxiv.org/pdf/2505.19306v1)**

> **作者:** Weiming Zhi; Ziyong Ma; Tianyi Zhang; Matthew Johnson-Roberson
>
> **摘要:** Autonomous robots typically need to construct representations of their surroundings and adapt their motions to the geometry of their environment. Here, we tackle the problem of constructing a policy model for collision-free motion generation, consistent with the environment, from a single input RGB image. Extracting 3D structures from a single image often involves monocular depth estimation. Developments in depth estimation have given rise to large pre-trained models such as DepthAnything. However, using outputs of these models for downstream motion generation is challenging due to frustum-shaped errors that arise. Instead, we propose a framework known as Video-Generation Environment Representation (VGER), which leverages the advances of large-scale video generation models to generate a moving camera video conditioned on the input image. Frames of this video, which form a multiview dataset, are then input into a pre-trained 3D foundation model to produce a dense point cloud. We then introduce a multi-scale noise approach to train an implicit representation of the environment structure and build a motion generation model that complies with the geometry of the representation. We extensively evaluate VGER over a diverse set of indoor and outdoor environments. We demonstrate its ability to produce smooth motions that account for the captured geometry of a scene, all from a single RGB input image.
>
---
#### [new 296] AmorLIP: Efficient Language-Image Pretraining via Amortization
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于语言-图像预训练任务，旨在解决CLIP方法需依赖大规模batch导致计算成本高昂的问题。提出AmorLIP框架，通过轻量神经网络 amortization 替代批量对比计算，并设计新型能量模型优化目标，提升训练效率与稳定性。实验显示其零样本任务性能较CLIP提升12.24%。**

- **链接: [http://arxiv.org/pdf/2505.18983v1](http://arxiv.org/pdf/2505.18983v1)**

> **作者:** Haotian Sun; Yitong Li; Yuchen Zhuang; Niao He; Hanjun Dai; Bo Dai
>
> **摘要:** Contrastive Language-Image Pretraining (CLIP) has demonstrated strong zero-shot performance across diverse downstream text-image tasks. Existing CLIP methods typically optimize a contrastive objective using negative samples drawn from each minibatch. To achieve robust representation learning, these methods require extremely large batch sizes and escalate computational demands to hundreds or even thousands of GPUs. Prior approaches to mitigate this issue often compromise downstream performance, prolong training duration, or face scalability challenges with very large datasets. To overcome these limitations, we propose AmorLIP, an efficient CLIP pretraining framework that amortizes expensive computations involved in contrastive learning through lightweight neural networks, which substantially improves training efficiency and performance. Leveraging insights from a spectral factorization of energy-based models, we introduce novel amortization objectives along with practical techniques to improve training stability. Extensive experiments across 38 downstream tasks demonstrate the superior zero-shot classification and retrieval capabilities of AmorLIP, consistently outperforming standard CLIP baselines with substantial relative improvements of up to 12.24%.
>
---
#### [new 297] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出Diffusion Blend方法，解决推理时多偏好对齐问题，允许用户通过线性组合预定义奖励函数及KL正则化强度，灵活控制生成结果。其通过融合多任务微调模型的扩散过程，避免额外训练，包含DB-MPA（多奖励对齐）和DB-KLA（正则化控制）算法，实验显示性能接近单独微调模型。**

- **链接: [http://arxiv.org/pdf/2505.18547v1](http://arxiv.org/pdf/2505.18547v1)**

> **作者:** Min Cheng; Fatemeh Doudi; Dileep Kalathil; Mohammad Ghavamzadeh; Panganamala R. Kumar
>
> **摘要:** Reinforcement learning (RL) algorithms have been used recently to align diffusion models with downstream objectives such as aesthetic quality and text-image consistency by fine-tuning them to maximize a single reward function under a fixed KL regularization. However, this approach is inherently restrictive in practice, where alignment must balance multiple, often conflicting objectives. Moreover, user preferences vary across prompts, individuals, and deployment contexts, with varying tolerances for deviation from a pre-trained base model. We address the problem of inference-time multi-preference alignment: given a set of basis reward functions and a reference KL regularization strength, can we design a fine-tuning procedure so that, at inference time, it can generate images aligned with any user-specified linear combination of rewards and regularization, without requiring additional fine-tuning? We propose Diffusion Blend, a novel approach to solve inference-time multi-preference alignment by blending backward diffusion processes associated with fine-tuned models, and we instantiate this approach with two algorithms: DB-MPA for multi-reward alignment and DB-KLA for KL regularization control. Extensive experiments show that Diffusion Blend algorithms consistently outperform relevant baselines and closely match or exceed the performance of individually fine-tuned models, enabling efficient, user-driven alignment at inference-time. The code is available at https://github.com/bluewoods127/DB-2025}{github.com/bluewoods127/DB-2025.
>
---
#### [new 298] Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments
- **分类: cs.AI; cs.CV; cs.LG; cs.LO**

- **简介: 该论文属模型集成任务，解决多模型在新环境因分布偏移导致性能下降及现有方法降低召回的问题。提出基于一致性 abduction 推理的方法，通过逻辑规则和两种算法（IP/HS）整合预测，最大化覆盖并控制不一致率，实验显示显著提升准确率和F1值。**

- **链接: [http://arxiv.org/pdf/2505.19361v1](http://arxiv.org/pdf/2505.19361v1)**

> **作者:** Mario Leiva; Noel Ngu; Joshua Shay Kricheli; Aditya Taparia; Ransalu Senanayake; Paulo Shakarian; Nathaniel Bastian; John Corcoran; Gerardo Simari
>
> **摘要:** The deployment of pre-trained perception models in novel environments often leads to performance degradation due to distributional shifts. Although recent artificial intelligence approaches for metacognition use logical rules to characterize and filter model errors, improving precision often comes at the cost of reduced recall. This paper addresses the hypothesis that leveraging multiple pre-trained models can mitigate this recall reduction. We formulate the challenge of identifying and managing conflicting predictions from various models as a consistency-based abduction problem. The input predictions and the learned error detection rules derived from each model are encoded in a logic program. We then seek an abductive explanation--a subset of model predictions--that maximizes prediction coverage while ensuring the rate of logical inconsistencies (derived from domain constraints) remains below a specified threshold. We propose two algorithms for this knowledge representation task: an exact method based on Integer Programming (IP) and an efficient Heuristic Search (HS). Through extensive experiments on a simulated aerial imagery dataset featuring controlled, complex distributional shifts, we demonstrate that our abduction-based framework outperforms individual models and standard ensemble baselines, achieving, for instance, average relative improvements of approximately 13.6% in F1-score and 16.6% in accuracy across 15 diverse test datasets when compared to the best individual model. Our results validate the use of consistency-based abduction as an effective mechanism to robustly integrate knowledge from multiple imperfect reasoners in challenging, novel scenarios.
>
---
#### [new 299] Don't Look Only Once: Towards Multimodal Interactive Reasoning with Selective Visual Revisitation
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决现有模型仅一次性处理视觉信息、缺乏动态视觉回溯的问题。提出v1方法，通过点选-复制机制让模型在推理中动态检索相关图像区域，并构建v1g数据集训练该能力，实验显示其在数学推理任务上显著提升性能。**

- **链接: [http://arxiv.org/pdf/2505.18842v1](http://arxiv.org/pdf/2505.18842v1)**

> **作者:** Jiwan Chung; Junhyeok Kim; Siyeol Kim; Jaeyoung Lee; Min Soo Kim; Youngjae Yu
>
> **摘要:** We present v1, a lightweight extension to Multimodal Large Language Models (MLLMs) that enables selective visual revisitation during inference. While current MLLMs typically consume visual input only once and reason purely over internal memory, v1 introduces a simple point-and-copy mechanism that allows the model to dynamically retrieve relevant image regions throughout the reasoning process. This mechanism augments existing architectures with minimal modifications, enabling contextual access to visual tokens based on the model's evolving hypotheses. To train this capability, we construct v1g, a dataset of 300K multimodal reasoning traces with interleaved visual grounding annotations. Experiments on three multimodal mathematical reasoning benchmarks -- MathVista, MathVision, and MathVerse -- demonstrate that v1 consistently improves performance over comparable baselines, particularly on tasks requiring fine-grained visual reference and multi-step reasoning. Our results suggest that dynamic visual access is a promising direction for enhancing grounded multimodal reasoning. Code, models, and data will be released to support future research.
>
---
#### [new 300] GraphAU-Pain: Graph-based Action Unit Representation for Pain Intensity Estimation
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出GraphAU-Pain，针对现有面部疼痛检测方法在可解释性与疼痛程度量化上的不足，通过构建以动作单元为节点、共现关系为边的图结构，并采用关系图神经网络建模其交互，提升疼痛强度估计的性能与可解释性。在UNBC数据集上取得66.21% F1和87.61%准确率。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19802v1](http://arxiv.org/pdf/2505.19802v1)**

> **作者:** Zhiyu Wang; Yang Liu; Hatice Gunes
>
> **摘要:** Understanding pain-related facial behaviors is essential for digital healthcare in terms of effective monitoring, assisted diagnostics, and treatment planning, particularly for patients unable to communicate verbally. Existing data-driven methods of detecting pain from facial expressions are limited due to interpretability and severity quantification. To this end, we propose GraphAU-Pain, leveraging a graph-based framework to model facial Action Units (AUs) and their interrelationships for pain intensity estimation. AUs are represented as graph nodes, with co-occurrence relationships as edges, enabling a more expressive depiction of pain-related facial behaviors. By utilizing a relational graph neural network, our framework offers improved interpretability and significant performance gains. Experiments conducted on the publicly available UNBC dataset demonstrate the effectiveness of the GraphAU-Pain, achieving an F1-score of 66.21% and accuracy of 87.61% in pain intensity estimation.
>
---
#### [new 301] ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文提出ScienceBoard框架，评估多模态代理在真实科学工作流中的能力。针对现有代理在复杂科研任务中可靠性不足的问题，构建了多领域交互环境及169项人类验证的科学任务基准，发现顶级模型成功率仅15%，并分析其局限性以指导改进设计。**

- **链接: [http://arxiv.org/pdf/2505.19897v1](http://arxiv.org/pdf/2505.19897v1)**

> **作者:** Qiushi Sun; Zhoumianze Liu; Chang Ma; Zichen Ding; Fangzhi Xu; Zhangyue Yin; Haiteng Zhao; Zhenyu Wu; Kanzhi Cheng; Zhaoyang Liu; Jianing Wang; Qintong Li; Xiangru Tang; Tianbao Xie; Xiachong Feng; Xiang Li; Ben Kao; Wenhai Wang; Biqing Qi; Lingpeng Kong; Zhiyong Wu
>
> **备注:** work in progress
>
> **摘要:** Large Language Models (LLMs) have extended their impact beyond Natural Language Processing, substantially fostering the development of interdisciplinary research. Recently, various LLM-based agents have been developed to assist scientific discovery progress across multiple aspects and domains. Among these, computer-using agents, capable of interacting with operating systems as humans do, are paving the way to automated scientific problem-solving and addressing routines in researchers' workflows. Recognizing the transformative potential of these agents, we introduce ScienceBoard, which encompasses two complementary contributions: (i) a realistic, multi-domain environment featuring dynamic and visually rich scientific workflows with integrated professional software, where agents can autonomously interact via different interfaces to accelerate complex research tasks and experiments; and (ii) a challenging benchmark of 169 high-quality, rigorously validated real-world tasks curated by humans, spanning scientific-discovery workflows in domains such as biochemistry, astronomy, and geoinformatics. Extensive evaluations of agents with state-of-the-art backbones (e.g., GPT-4o, Claude 3.7, UI-TARS) show that, despite some promising results, they still fall short of reliably assisting scientists in complex workflows, achieving only a 15% overall success rate. In-depth analysis further provides valuable insights for addressing current agent limitations and more effective design principles, paving the way to build more capable agents for scientific discovery. Our code, environment, and benchmark are at https://qiushisun.github.io/ScienceBoard-Home/.
>
---
#### [new 302] CardioCoT: Hierarchical Reasoning for Multimodal Survival Analysis
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态生存分析任务，旨在解决急性心梗患者心血管不良事件复发风险预测中模型可解释性不足及数据复杂性问题。提出CardioCoT框架：第一阶段通过自优化机制生成医学影像与文本的分层推理路径，第二阶段融合影像数据训练预测模型，兼顾预测性能与临床可解释性。**

- **链接: [http://arxiv.org/pdf/2505.19195v1](http://arxiv.org/pdf/2505.19195v1)**

> **作者:** Shaohao Rui; Haoyang Su; Jinyi Xiang; Lian-Ming Wu; Xiaosong Wang
>
> **摘要:** Accurate prediction of major adverse cardiovascular events recurrence risk in acute myocardial infarction patients based on postoperative cardiac MRI and associated clinical notes is crucial for precision treatment and personalized intervention. Existing methods primarily focus on risk stratification capability while overlooking the need for intermediate robust reasoning and model interpretability in clinical practice. Moreover, end-to-end risk prediction using LLM/VLM faces significant challenges due to data limitations and modeling complexity. To bridge this gap, we propose CardioCoT, a novel two-stage hierarchical reasoning-enhanced survival analysis framework designed to enhance both model interpretability and predictive performance. In the first stage, we employ an evidence-augmented self-refinement mechanism to guide LLM/VLMs in generating robust hierarchical reasoning trajectories based on associated radiological findings. In the second stage, we integrate the reasoning trajectories with imaging data for risk model training and prediction. CardioCoT demonstrates superior performance in MACE recurrence risk prediction while providing interpretable reasoning processes, offering valuable insights for clinical decision-making.
>
---
#### [new 303] DiffVLA: Vision-Language Guided Diffusion Planning for Autonomous Driving
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于端到端自动驾驶任务，旨在解决现有方法中BEV计算成本高、动作多样性不足及复杂场景决策不佳的问题。提出DiffVLA方法，结合稀疏-密集扩散策略与视觉语言模型，通过跨模态交互优化多模态行为与轨迹生成，提升复杂场景决策性能。**

- **链接: [http://arxiv.org/pdf/2505.19381v1](http://arxiv.org/pdf/2505.19381v1)**

> **作者:** Anqing Jiang; Yu Gao; Zhigang Sun; Yiru Wang; Jijun Wang; Jinghao Chai; Qian Cao; Yuweng Heng; Hao Jiang; Zongzheng Zhang; Xianda Guo; Hao Sun; Hao Zhao
>
> **备注:** 4pages
>
> **摘要:** Research interest in end-to-end autonomous driving has surged owing to its fully differentiable design integrating modular tasks, i.e. perception, prediction and planing, which enables optimization in pursuit of the ultimate goal. Despite the great potential of the end-to-end paradigm, existing methods suffer from several aspects including expensive BEV (bird's eye view) computation, action diversity, and sub-optimal decision in complex real-world scenarios. To address these challenges, we propose a novel hybrid sparse-dense diffusion policy, empowered by a Vision-Language Model (VLM), called Diff-VLA. We explore the sparse diffusion representation for efficient multi-modal driving behavior. Moreover, we rethink the effectiveness of VLM driving decision and improve the trajectory generation guidance through deep interaction across agent, map instances and VLM output. Our method shows superior performance in Autonomous Grand Challenge 2025 which contains challenging real and reactive synthetic scenarios. Our methods achieves 45.0 PDMS.
>
---
#### [new 304] Grounding Language with Vision: A Conditional Mutual Information Calibrated Decoding Strategy for Reducing Hallucinations in LVLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对大视觉语言模型（LVLMs）因过度依赖语言先验而忽视图像信息导致幻觉的问题，提出基于条件点互信息（C-PMI）的解码策略。通过联合建模文本与视觉token的相关性，动态优化解码过程，提升生成内容与输入图像的关联性，减少幻觉同时保持效率。**

- **链接: [http://arxiv.org/pdf/2505.19678v1](http://arxiv.org/pdf/2505.19678v1)**

> **作者:** Hao Fang; Changle Zhou; Jiawei Kong; Kuofeng Gao; Bin Chen; Tao Liang; Guojun Ma; Shu-Tao Xia
>
> **摘要:** Large Vision-Language Models (LVLMs) are susceptible to hallucinations, where generated responses seem semantically plausible yet exhibit little or no relevance to the input image. Previous studies reveal that this issue primarily stems from LVLMs' over-reliance on language priors while disregarding the visual information during decoding. To alleviate this issue, we introduce a novel Conditional Pointwise Mutual Information (C-PMI) calibrated decoding strategy, which adaptively strengthens the mutual dependency between generated texts and input images to mitigate hallucinations. Unlike existing methods solely focusing on text token sampling, we propose to jointly model the contributions of visual and textual tokens to C-PMI, formulating hallucination mitigation as a bi-level optimization problem aimed at maximizing mutual information. To solve it, we design a token purification mechanism that dynamically regulates the decoding process by sampling text tokens remaining maximally relevant to the given image, while simultaneously refining image tokens most pertinent to the generated response. Extensive experiments across various benchmarks reveal that the proposed method significantly reduces hallucinations in LVLMs while preserving decoding efficiency.
>
---
#### [new 305] Grounding Bodily Awareness in Visual Representations for Efficient Policy Learning
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人操作策略学习任务，旨在解决复杂身体动力学下视觉表征学习效率低的问题。提出ICon方法，通过对比学习分离Vision Transformer中代理与环境的token特征，生成以代理为中心的视觉表征，提升策略性能并支持跨机器人迁移。**

- **链接: [http://arxiv.org/pdf/2505.18487v1](http://arxiv.org/pdf/2505.18487v1)**

> **作者:** Junlin Wang; Zhiyun Lin
>
> **备注:** A preprint version
>
> **摘要:** Learning effective visual representations for robotic manipulation remains a fundamental challenge due to the complex body dynamics involved in action execution. In this paper, we study how visual representations that carry body-relevant cues can enable efficient policy learning for downstream robotic manipulation tasks. We present $\textbf{I}$nter-token $\textbf{Con}$trast ($\textbf{ICon}$), a contrastive learning method applied to the token-level representations of Vision Transformers (ViTs). ICon enforces a separation in the feature space between agent-specific and environment-specific tokens, resulting in agent-centric visual representations that embed body-specific inductive biases. This framework can be seamlessly integrated into end-to-end policy learning by incorporating the contrastive loss as an auxiliary objective. Our experiments show that ICon not only improves policy performance across various manipulation tasks but also facilitates policy transfer across different robots. The project website: https://github.com/HenryWJL/icon
>
---
#### [new 306] Optimizing edge AI models on HPC systems with the edge in the loop
- **分类: cs.DC; cs.CV; I.2.6; D.1.3; I.2.8; I.5.1**

- **简介: 该研究属于边缘AI模型优化任务，针对边缘设备需快速高精度推理的问题，提出硬件感知NAS方法，结合HPC系统与边缘实时延迟评估，加速架构探索。在AM质量控制中，较人工模型提升8.8倍推理速度与1.35倍精度。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19995v1](http://arxiv.org/pdf/2505.19995v1)**

> **作者:** Marcel Aach; Cyril Blanc; Andreas Lintermann; Kurt De Grave
>
> **备注:** 13 pages, accepted for oral presentation at Computational Aspects of Deep Learning 2025 (at ISC 2025)
>
> **摘要:** Artificial intelligence and machine learning models deployed on edge devices, e.g., for quality control in Additive Manufacturing (AM), are frequently small in size. Such models usually have to deliver highly accurate results within a short time frame. Methods that are commonly employed in literature start out with larger trained models and try to reduce their memory and latency footprint by structural pruning, knowledge distillation, or quantization. It is, however, also possible to leverage hardware-aware Neural Architecture Search (NAS), an approach that seeks to systematically explore the architecture space to find optimized configurations. In this study, a hardware-aware NAS workflow is introduced that couples an edge device located in Belgium with a powerful High-Performance Computing system in Germany, to train possible architecture candidates as fast as possible while performing real-time latency measurements on the target hardware. The approach is verified on a use case in the AM domain, based on the open RAISE-LPBF dataset, achieving ~8.8 times faster inference speed while simultaneously enhancing model quality by a factor of ~1.35, compared to a human-designed baseline.
>
---
#### [new 307] GQKVA: Efficient Pre-training of Transformers by Grouping Queries, Keys, and Values
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于transformer模型优化任务，旨在解决预训练效率低和过参数化问题。提出GQKVA方法，通过分组查询、键、值实现加速与压缩。实验显示，在图像分类任务中，ViT模型大小减少4%-15%，准确率仅小幅下降甚至提升，证明了该方法的有效性。**

- **链接: [http://arxiv.org/pdf/2311.03426v2](http://arxiv.org/pdf/2311.03426v2)**

> **作者:** Farnoosh Javadi; Walid Ahmed; Habib Hajimolahoseini; Foozhan Ataiefard; Mohammad Hassanpour; Saina Asani; Austin Wen; Omar Mohamed Awad; Kangling Liu; Yang Liu
>
> **摘要:** Massive transformer-based models face several challenges, including slow and computationally intensive pre-training and over-parametrization. This paper addresses these challenges by proposing a versatile method called GQKVA, which generalizes query, key, and value grouping techniques. GQKVA is designed to speed up transformer pre-training while reducing the model size. Our experiments with various GQKVA variants highlight a clear trade-off between performance and model size, allowing for customized choices based on resource and time limitations. Our findings also indicate that the conventional multi-head attention approach is not always the best choice, as there are lighter and faster alternatives available. We tested our method on ViT, which achieved an approximate 0.3% increase in accuracy while reducing the model size by about 4% in the task of image classification. Additionally, our most aggressive model reduction experiment resulted in a reduction of approximately 15% in model size, with only around a 1% drop in accuracy.
>
---
#### [new 308] Large Language Model-Driven Distributed Integrated Multimodal Sensing and Semantic Communications
- **分类: eess.SP; cs.AI; cs.CV**

- **简介: 该论文提出LLM-DiSAC框架，解决传统单模态传感系统在复杂环境中的局限性。通过多设备协作，融合RF与视觉数据，设计RF-视觉融合网络、LLM语义传输网络及自适应聚合模型，并采用分布式学习保护隐私。实验验证其提升感知与通信性能。**

- **链接: [http://arxiv.org/pdf/2505.18194v1](http://arxiv.org/pdf/2505.18194v1)**

> **作者:** Yubo Peng; Luping Xiang; Bingxin Zhang; Kun Yang
>
> **摘要:** Traditional single-modal sensing systems-based solely on either radio frequency (RF) or visual data-struggle to cope with the demands of complex and dynamic environments. Furthermore, single-device systems are constrained by limited perspectives and insufficient spatial coverage, which impairs their effectiveness in urban or non-line-of-sight scenarios. To overcome these challenges, we propose a novel large language model (LLM)-driven distributed integrated multimodal sensing and semantic communication (LLM-DiSAC) framework. Specifically, our system consists of multiple collaborative sensing devices equipped with RF and camera modules, working together with an aggregation center to enhance sensing accuracy. First, on sensing devices, LLM-DiSAC develops an RF-vision fusion network (RVFN), which employs specialized feature extractors for RF and visual data, followed by a cross-attention module for effective multimodal integration. Second, a LLM-based semantic transmission network (LSTN) is proposed to enhance communication efficiency, where the LLM-based decoder leverages known channel parameters, such as transceiver distance and signal-to-noise ratio (SNR), to mitigate semantic distortion. Third, at the aggregation center, a transformer-based aggregation model (TRAM) with an adaptive aggregation attention mechanism is developed to fuse distributed features and enhance sensing accuracy. To preserve data privacy, a two-stage distributed learning strategy is introduced, allowing local model training at the device level and centralized aggregation model training using intermediate features. Finally, evaluations on a synthetic multi-view RF-visual dataset generated by the Genesis simulation engine show that LLM-DiSAC achieves a good performance.
>
---
#### [new 309] I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多模态学习任务，针对传统融合方法无法捕捉异构模态交互及缺乏可解释性的问题，提出I2MoE框架。通过交互专家网络结合弱监督损失学习多模态交互模式，并利用动态加权模型分配专家权重，实现样本级与全局可解释性，提升融合性能。**

- **链接: [http://arxiv.org/pdf/2505.19190v1](http://arxiv.org/pdf/2505.19190v1)**

> **作者:** Jiayi Xin; Sukwon Yun; Jie Peng; Inyoung Choi; Jenna L. Ballard; Tianlong Chen; Qi Long
>
> **备注:** ICML 2025 Poster
>
> **摘要:** Modality fusion is a cornerstone of multimodal learning, enabling information integration from diverse data sources. However, vanilla fusion methods are limited by (1) inability to account for heterogeneous interactions between modalities and (2) lack of interpretability in uncovering the multimodal interactions inherent in the data. To this end, we propose I2MoE (Interpretable Multimodal Interaction-aware Mixture of Experts), an end-to-end MoE framework designed to enhance modality fusion by explicitly modeling diverse multimodal interactions, as well as providing interpretation on a local and global level. First, I2MoE utilizes different interaction experts with weakly supervised interaction losses to learn multimodal interactions in a data-driven way. Second, I2MoE deploys a reweighting model that assigns importance scores for the output of each interaction expert, which offers sample-level and dataset-level interpretation. Extensive evaluation of medical and general multimodal datasets shows that I2MoE is flexible enough to be combined with different fusion techniques, consistently improves task performance, and provides interpretation across various real-world scenarios. Code is available at https://github.com/Raina-Xin/I2MoE.
>
---
#### [new 310] Multiplicity is an Inevitable and Inherent Challenge in Multimodal Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于多模态学习任务，指出当前方法忽视模态间固有多对多关系（Multiplicity），导致训练、评估缺陷。分析其成因（语义抽象等），呼吁开发新框架与数据协议解决此问题。**

- **链接: [http://arxiv.org/pdf/2505.19614v1](http://arxiv.org/pdf/2505.19614v1)**

> **作者:** Sanghyuk Chun
>
> **摘要:** Multimodal learning has seen remarkable progress, particularly with the emergence of large-scale pre-training across various modalities. However, most current approaches are built on the assumption of a deterministic, one-to-one alignment between modalities. This oversimplifies real-world multimodal relationships, where their nature is inherently many-to-many. This phenomenon, named multiplicity, is not a side-effect of noise or annotation error, but an inevitable outcome of semantic abstraction, representational asymmetry, and task-dependent ambiguity in multimodal tasks. This position paper argues that multiplicity is a fundamental bottleneck that manifests across all stages of the multimodal learning pipeline: from data construction to training and evaluation. This paper examines the causes and consequences of multiplicity, and highlights how multiplicity introduces training uncertainty, unreliable evaluation, and low dataset quality. This position calls for new research directions on multimodal learning: novel multiplicity-aware learning frameworks and dataset construction protocols considering multiplicity.
>
---
#### [new 311] Learning without Isolation: Pathway Protection for Continual Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决参数保护导致存储效率低的问题。提出LwI框架，通过图匹配保护旧任务路径而非隔离参数，利用网络稀疏性自适应分配新任务路径，高效缓解灾难性遗忘。**

- **链接: [http://arxiv.org/pdf/2505.18568v1](http://arxiv.org/pdf/2505.18568v1)**

> **作者:** Zhikang Chen; Abudukelimu Wuerkaixi; Sen Cui; Haoxuan Li; Ding Li; Jingfeng Zhang; Bo Han; Gang Niu; Houfang Liu; Yi Yang; Sifan Yang; Changshui Zhang; Tianling Ren
>
> **备注:** 23 pages
>
> **摘要:** Deep networks are prone to catastrophic forgetting during sequential task learning, i.e., losing the knowledge about old tasks upon learning new tasks. To this end, continual learning(CL) has emerged, whose existing methods focus mostly on regulating or protecting the parameters associated with the previous tasks. However, parameter protection is often impractical, since the size of parameters for storing the old-task knowledge increases linearly with the number of tasks, otherwise it is hard to preserve the parameters related to the old-task knowledge. In this work, we bring a dual opinion from neuroscience and physics to CL: in the whole networks, the pathways matter more than the parameters when concerning the knowledge acquired from the old tasks. Following this opinion, we propose a novel CL framework, learning without isolation(LwI), where model fusion is formulated as graph matching and the pathways occupied by the old tasks are protected without being isolated. Thanks to the sparsity of activation channels in a deep network, LwI can adaptively allocate available pathways for a new task, realizing pathway protection and addressing catastrophic forgetting in a parameter-efficient manner. Experiments on popular benchmark datasets demonstrate the superiority of the proposed LwI.
>
---
#### [new 312] ReflectGAN: Modeling Vegetation Effects for Soil Carbon Estimation from Satellite Imagery
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于土壤有机碳（SOC）估算任务，旨在解决植被覆盖导致卫星光谱污染影响土壤反射率测量的问题。提出ReflectGAN框架，通过GAN学习植被与裸土反射率的光谱转换关系，重建高质量土壤反射率数据，提升SOC模型精度。实验显示其性能优于现有方法，如PMM-SU，在LUCAS和Sentinel-2数据集上表现更优。**

- **链接: [http://arxiv.org/pdf/2505.18546v1](http://arxiv.org/pdf/2505.18546v1)**

> **作者:** Dristi Datta; Manoranjan Paul; Manzur Murshed; Shyh Wei Teng; Leigh M. Schmidtke
>
> **摘要:** Soil organic carbon (SOC) is a critical indicator of soil health, but its accurate estimation from satellite imagery is hindered in vegetated regions due to spectral contamination from plant cover, which obscures soil reflectance and reduces model reliability. This study proposes the Reflectance Transformation Generative Adversarial Network (ReflectGAN), a novel paired GAN-based framework designed to reconstruct accurate bare soil reflectance from vegetated soil satellite observations. By learning the spectral transformation between vegetated and bare soil reflectance, ReflectGAN facilitates more precise SOC estimation under mixed land cover conditions. Using the LUCAS 2018 dataset and corresponding Landsat 8 imagery, we trained multiple learning-based models on both original and ReflectGAN-reconstructed reflectance inputs. Models trained on ReflectGAN outputs consistently outperformed those using existing vegetation correction methods. For example, the best-performing model (RF) achieved an $R^2$ of 0.54, RMSE of 3.95, and RPD of 2.07 when applied to the ReflectGAN-generated signals, representing a 35\% increase in $R^2$, a 43\% reduction in RMSE, and a 43\% improvement in RPD compared to the best existing method (PMM-SU). The performance of the models with ReflectGAN is also better compared to their counterparts when applied to another dataset, i.e., Sentinel-2 imagery. These findings demonstrate the potential of ReflectGAN to improve SOC estimation accuracy in vegetated landscapes, supporting more reliable soil monitoring.
>
---
#### [new 313] MedITok: A Unified Tokenizer for Medical Image Synthesis and Interpretation
- **分类: eess.IV; cs.CV**

- **简介: 论文提出MedITok，首个统一医学图像合成与解释的tokenizer。解决现有模型无法兼顾图像结构细节与临床语义的问题。通过两阶段训练框架平衡重建与语义，训练于3000万医学图像及200万图-文本对数据，实现跨模态任务SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.19225v1](http://arxiv.org/pdf/2505.19225v1)**

> **作者:** Chenglong Ma; Yuanfeng Ji; Jin Ye; Zilong Li; Chenhui Wang; Junzhi Ning; Wei Li; Lihao Liu; Qiushan Guo; Tianbin Li; Junjun He; Hongming Shan
>
> **摘要:** Advanced autoregressive models have reshaped multimodal AI. However, their transformative potential in medical imaging remains largely untapped due to the absence of a unified visual tokenizer -- one capable of capturing fine-grained visual structures for faithful image reconstruction and realistic image synthesis, as well as rich semantics for accurate diagnosis and image interpretation. To this end, we present MedITok, the first unified tokenizer tailored for medical images, encoding both low-level structural details and high-level clinical semantics within a unified latent space. To balance these competing objectives, we introduce a novel two-stage training framework: a visual representation alignment stage that cold-starts the tokenizer reconstruction learning with a visual semantic constraint, followed by a textual semantic representation alignment stage that infuses detailed clinical semantics into the latent space. Trained on the meticulously collected large-scale dataset with over 30 million medical images and 2 million image-caption pairs, MedITok achieves state-of-the-art performance on more than 30 datasets across 9 imaging modalities and 4 different tasks. By providing a unified token space for autoregressive modeling, MedITok supports a wide range of tasks in clinical diagnostics and generative healthcare applications. Model and code will be made publicly available at: https://github.com/Masaaki-75/meditok.
>
---
#### [new 314] Advancements in Medical Image Classification through Fine-Tuning Natural Domain Foundation Models
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文研究医疗图像分类任务，旨在通过微调自然领域预训练模型（如DINOv2、SAM2等）提升医疗图像分类效果，解决医疗数据标注量少的问题。作者测试了6种基础模型在乳腺X光、皮肤病变等4个医疗数据集上的表现，发现AIMv2等模型显著提升分类性能，验证了自然领域模型在医疗领域的迁移潜力。**

- **链接: [http://arxiv.org/pdf/2505.19779v1](http://arxiv.org/pdf/2505.19779v1)**

> **作者:** Mobina Mansoori; Sajjad Shahabodini; Farnoush Bayatmakou; Jamshid Abouei; Konstantinos N. Plataniotis; Arash Mohammadi
>
> **摘要:** Using massive datasets, foundation models are large-scale, pre-trained models that perform a wide range of tasks. These models have shown consistently improved results with the introduction of new methods. It is crucial to analyze how these trends impact the medical field and determine whether these advancements can drive meaningful change. This study investigates the application of recent state-of-the-art foundation models, DINOv2, MAE, VMamba, CoCa, SAM2, and AIMv2, for medical image classification. We explore their effectiveness on datasets including CBIS-DDSM for mammography, ISIC2019 for skin lesions, APTOS2019 for diabetic retinopathy, and CHEXPERT for chest radiographs. By fine-tuning these models and evaluating their configurations, we aim to understand the potential of these advancements in medical image classification. The results indicate that these advanced models significantly enhance classification outcomes, demonstrating robust performance despite limited labeled data. Based on our results, AIMv2, DINOv2, and SAM2 models outperformed others, demonstrating that progress in natural domain training has positively impacted the medical domain and improved classification outcomes. Our code is publicly available at: https://github.com/sajjad-sh33/Medical-Transfer-Learning.
>
---
#### [new 315] Improving Resnet-9 Generalization Trained on Small Datasets
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文针对小数据集下图像分类任务，提出结合锐度感知优化、标签平滑、梯度集中、输入块白化及元学习等方法，提升ResNet-9模型泛化能力，解决在10分钟内利用CIFAR-10的10%数据（5000图）高效训练高精度模型的问题，最终达88%测试准确率。**

- **链接: [http://arxiv.org/pdf/2309.03965v1](http://arxiv.org/pdf/2309.03965v1)**

> **作者:** Omar Mohamed Awad; Habib Hajimolahoseini; Michael Lim; Gurpreet Gosal; Walid Ahmed; Yang Liu; Gordon Deng
>
> **摘要:** This paper presents our proposed approach that won the first prize at the ICLR competition on Hardware Aware Efficient Training. The challenge is to achieve the highest possible accuracy in an image classification task in less than 10 minutes. The training is done on a small dataset of 5000 images picked randomly from CIFAR-10 dataset. The evaluation is performed by the competition organizers on a secret dataset with 1000 images of the same size. Our approach includes applying a series of technique for improving the generalization of ResNet-9 including: sharpness aware optimization, label smoothing, gradient centralization, input patch whitening as well as metalearning based training. Our experiments show that the ResNet-9 can achieve the accuracy of 88% while trained only on a 10% subset of CIFAR-10 dataset in less than 10 minuets
>
---
#### [new 316] Probabilistic Kernel Function for Fast Angle Testing
- **分类: cs.LG; cs.AI; cs.CV; cs.DB; cs.DS**

- **简介: 该论文针对高维空间角度测试任务，提出两种基于确定性投影的核函数（角度比较与阈值化），解决现有方法依赖高斯随机投影及渐近假设的问题，提升计算效率。实验显示其在ANNS中QPS达HNSW的2.5-3倍。**

- **链接: [http://arxiv.org/pdf/2505.20274v1](http://arxiv.org/pdf/2505.20274v1)**

> **作者:** Kejing Lu; Chuan Xiao; Yoshiharu Ishikawa
>
> **摘要:** In this paper, we study the angle testing problem in high-dimensional Euclidean spaces and propose two projection-based probabilistic kernel functions, one designed for angle comparison and the other for angle thresholding. Unlike existing approaches that rely on random projection vectors drawn from Gaussian distributions, our approach leverages reference angles and employs a deterministic structure for the projection vectors. Notably, our kernel functions do not require asymptotic assumptions, such as the number of projection vectors tending to infinity, and can be both theoretically and experimentally shown to outperform Gaussian-distribution-based kernel functions. We further apply the proposed kernel function to Approximate Nearest Neighbor Search (ANNS) and demonstrate that our approach achieves a 2.5X ~ 3X higher query-per-second (QPS) throughput compared to the state-of-the-art graph-based search algorithm HNSW.
>
---
#### [new 317] How We Won the ISLES'24 Challenge by Preprocessing
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 论文针对中风病变分割任务，解决仅用CT预测不可见病灶的挑战。提出基于深度学习的预处理（去骨、定制窗宽）结合残差nnU-Net，提升分割精度（Dice 28.5）。**

- **链接: [http://arxiv.org/pdf/2505.18424v1](http://arxiv.org/pdf/2505.18424v1)**

> **作者:** Tianyi Ren; Juampablo E. Heras Rivera; Hitender Oswal; Yutong Pan; William Henry; Jacob Ruzevick; Mehmet Kurt
>
> **摘要:** Stroke is among the top three causes of death worldwide, and accurate identification of stroke lesion boundaries is critical for diagnosis and treatment. Supervised deep learning methods have emerged as the leading solution for stroke lesion segmentation but require large, diverse, and annotated datasets. The ISLES'24 challenge addresses this need by providing longitudinal stroke imaging data, including CT scans taken on arrival to the hospital and follow-up MRI taken 2-9 days from initial arrival, with annotations derived from follow-up MRI. Importantly, models submitted to the ISLES'24 challenge are evaluated using only CT inputs, requiring prediction of lesion progression that may not be visible in CT scans for segmentation. Our winning solution shows that a carefully designed preprocessing pipeline including deep-learning-based skull stripping and custom intensity windowing is beneficial for accurate segmentation. Combined with a standard large residual nnU-Net architecture for segmentation, this approach achieves a mean test Dice of 28.5 with a standard deviation of 21.27.
>
---
#### [new 318] OmniCharacter: Towards Immersive Role-Playing Agents with Seamless Speech-Language Personality Interaction
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于角色扮演代理（RPAs）任务，解决现有方法忽视语音特质导致沉浸感不足及高延迟问题。提出OmniCharacter模型，整合语音与语言的人格交互，构建含20类角色和10K对话的OmniCharacter-10K数据集，实现低延迟（289ms）的多模态响应，提升内容与风格表现。**

- **链接: [http://arxiv.org/pdf/2505.20277v1](http://arxiv.org/pdf/2505.20277v1)**

> **作者:** Haonan Zhang; Run Luo; Xiong Liu; Yuchuan Wu; Ting-En Lin; Pengpeng Zeng; Qiang Qu; Feiteng Fang; Min Yang; Lianli Gao; Jingkuan Song; Fei Huang; Yongbin Li
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Role-Playing Agents (RPAs), benefiting from large language models, is an emerging interactive AI system that simulates roles or characters with diverse personalities. However, existing methods primarily focus on mimicking dialogues among roles in textual form, neglecting the role's voice traits (e.g., voice style and emotions) as playing a crucial effect in interaction, which tends to be more immersive experiences in realistic scenarios. Towards this goal, we propose OmniCharacter, a first seamless speech-language personality interaction model to achieve immersive RPAs with low latency. Specifically, OmniCharacter enables agents to consistently exhibit role-specific personality traits and vocal traits throughout the interaction, enabling a mixture of speech and language responses. To align the model with speech-language scenarios, we construct a dataset named OmniCharacter-10K, which involves more distinctive characters (20), richly contextualized multi-round dialogue (10K), and dynamic speech response (135K). Experimental results showcase that our method yields better responses in terms of both content and style compared to existing RPAs and mainstream speech-language models, with a response latency as low as 289ms. Code and dataset are available at https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/OmniCharacter.
>
---
## 更新

#### [replaced 001] DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14362v2](http://arxiv.org/pdf/2505.14362v2)**

> **作者:** Ziwei Zheng; Michael Yang; Jack Hong; Chenxiao Zhao; Guohai Xu; Le Yang; Chao Shen; Xing Yu
>
> **备注:** Ziwei, Michael, Jack, and Chenxiao are equal-contribution. The list order is random
>
> **摘要:** Large Vision-Language Models (VLMs) have shown strong capabilities in multimodal understanding and reasoning, yet they are primarily constrained by text-based reasoning processes. However, achieving seamless integration of visual and textual reasoning which mirrors human cognitive processes remains a significant challenge. In particular, effectively incorporating advanced visual input processing into reasoning mechanisms is still an open question. Thus, in this paper, we explore the interleaved multimodal reasoning paradigm and introduce DeepEyes, a model with "thinking with images" capabilities incentivized through end-to-end reinforcement learning without the need for cold-start SFT. Notably, this ability emerges natively within the model itself, leveraging its inherent grounding ability as a tool instead of depending on separate specialized models. Specifically, we propose a tool-use-oriented data selection mechanism and a reward strategy to encourage successful tool-assisted reasoning trajectories. DeepEyes achieves significant performance gains on fine-grained perception and reasoning benchmarks and also demonstrates improvement in grounding, hallucination, and mathematical reasoning tasks. Interestingly, we observe the distinct evolution of tool-calling behavior from initial exploration to efficient and accurate exploitation, and diverse thinking patterns that closely mirror human visual reasoning processes. Code is available at https://github.com/Visual-Agent/DeepEyes.
>
---
#### [replaced 002] EDTformer: An Efficient Decoder Transformer for Visual Place Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.00784v2](http://arxiv.org/pdf/2412.00784v2)**

> **作者:** Tong Jin; Feng Lu; Shuyu Hu; Chun Yuan; Yunpeng Liu
>
> **备注:** Accepted by T-CSVT2025
>
> **摘要:** Visual place recognition (VPR) aims to determine the general geographical location of a query image by retrieving visually similar images from a large geo-tagged database. To obtain a global representation for each place image, most approaches typically focus on the aggregation of deep features extracted from a backbone through using current prominent architectures (e.g., CNNs, MLPs, pooling layer, and transformer encoder), giving little attention to the transformer decoder. However, we argue that its strong capability to capture contextual dependencies and generate accurate features holds considerable potential for the VPR task. To this end, we propose an Efficient Decoder Transformer (EDTformer) for feature aggregation, which consists of several stacked simplified decoder blocks followed by two linear layers to directly produce robust and discriminative global representations. Specifically, we do this by formulating deep features as the keys and values, as well as a set of learnable parameters as the queries. Our EDTformer can fully utilize the contextual information within deep features, then gradually decode and aggregate the effective features into the learnable queries to output the global representations. Moreover, to provide more powerful deep features for EDTformer and further facilitate the robustness, we use the foundation model DINOv2 as the backbone and propose a Low-rank Parallel Adaptation (LoPA) method to enhance its performance in VPR, which can refine the intermediate features of the backbone progressively in a memory- and parameter-efficient way. As a result, our method not only outperforms single-stage VPR methods on multiple benchmark datasets, but also outperforms two-stage VPR methods which add a re-ranking with considerable cost. Code will be available at https://github.com/Tong-Jin01/EDTformer.
>
---
#### [replaced 003] Dynamic Multimodal Evaluation with Flexible Complexity by Vision-Language Bootstrapping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.08695v3](http://arxiv.org/pdf/2410.08695v3)**

> **作者:** Yue Yang; Shuibai Zhang; Wenqi Shao; Kaipeng Zhang; Yi Bin; Yu Wang; Ping Luo
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across multimodal tasks such as visual perception and reasoning, leading to good performance on various multimodal evaluation benchmarks. However, these benchmarks keep a static nature and overlap with the pre-training data, resulting in fixed complexity constraints and data contamination issues. This raises the concern regarding the validity of the evaluation. To address these two challenges, we introduce a dynamic multimodal evaluation protocol called Vision-Language Bootstrapping (VLB). VLB provides a robust and comprehensive assessment for LVLMs with reduced data contamination and flexible complexity. To this end, VLB dynamically generates new visual question-answering samples through a multimodal bootstrapping module that modifies both images and language, while ensuring that newly generated samples remain consistent with the original ones by a judge module. By composing various bootstrapping strategies, VLB offers dynamic variants of existing benchmarks with diverse complexities, enabling the evaluation to co-evolve with the ever-evolving capabilities of LVLMs. Extensive experimental results across multiple benchmarks, including SEEDBench, MMBench, and MME, show that VLB significantly reduces data contamination and exposes performance limitations of LVLMs.
>
---
#### [replaced 004] Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation
- **分类: cs.NE; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.13610v2](http://arxiv.org/pdf/2412.13610v2)**

> **作者:** Zecheng Hao; Qichao Ma; Kang Chen; Yi Zhang; Zhaofei Yu; Tiejun Huang
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Spiking Neural Network (SNN), as a brain-inspired and energy-efficient network, is currently facing the pivotal challenge of exploring a suitable and efficient learning framework. The predominant training methodologies, namely Spatial-Temporal Back-propagation (STBP) and ANN-SNN Conversion, are encumbered by substantial training overhead or pronounced inference latency, which impedes the advancement of SNNs in scaling to larger networks and navigating intricate application domains. In this work, we propose a novel parallel conversion learning framework, which establishes a mathematical mapping relationship between each time-step of the parallel spiking neurons and the cumulative spike firing rate. We theoretically validate the lossless and sorting properties of the conversion process, as well as pointing out the optimal shifting distance for each step. Furthermore, by integrating the above framework with the distribution-aware error calibration technique, we can achieve efficient conversion towards more general activation functions or training-free circumstance. Extensive experiments have confirmed the significant performance advantages of our method for various conversion cases under ultra-low time latency. To our best knowledge, this is the first work which jointly utilizes parallel spiking calculation and ANN-SNN Conversion, providing a highly promising approach for SNN supervised training. Code is available at https://github.com/hzc1208/Parallel_Conversion.
>
---
#### [replaced 005] DAE-Fuse: An Adaptive Discriminative Autoencoder for Multi-Modality Image Fusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.10080v3](http://arxiv.org/pdf/2409.10080v3)**

> **作者:** Yuchen Guo; Ruoxiang Xu; Rongcheng Li; Weifeng Su
>
> **摘要:** In extreme scenarios such as nighttime or low-visibility environments, achieving reliable perception is critical for applications like autonomous driving, robotics, and surveillance. Multi-modality image fusion, particularly integrating infrared imaging, offers a robust solution by combining complementary information from different modalities to enhance scene understanding and decision-making. However, current methods face significant limitations: GAN-based approaches often produce blurry images that lack fine-grained details, while AE-based methods may introduce bias toward specific modalities, leading to unnatural fusion results. To address these challenges, we propose DAE-Fuse, a novel two-phase discriminative autoencoder framework that generates sharp and natural fused images. Furthermore, We pioneer the extension of image fusion techniques from static images to the video domain while preserving temporal consistency across frames, thus advancing the perceptual capabilities required for autonomous navigation. Extensive experiments on public datasets demonstrate that DAE-Fuse achieves state-of-the-art performance on multiple benchmarks, with superior generalizability to tasks like medical image fusion.
>
---
#### [replaced 006] PromptHMR: Promptable Human Mesh Recovery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06397v2](http://arxiv.org/pdf/2504.06397v2)**

> **作者:** Yufu Wang; Yu Sun; Priyanka Patel; Kostas Daniilidis; Michael J. Black; Muhammed Kocabas
>
> **备注:** CVPR 2025. Project website: https://yufu-wang.github.io/phmr-page
>
> **摘要:** Human pose and shape (HPS) estimation presents challenges in diverse scenarios such as crowded scenes, person-person interactions, and single-view reconstruction. Existing approaches lack mechanisms to incorporate auxiliary "side information" that could enhance reconstruction accuracy in such challenging scenarios. Furthermore, the most accurate methods rely on cropped person detections and cannot exploit scene context while methods that process the whole image often fail to detect people and are less accurate than methods that use crops. While recent language-based methods explore HPS reasoning through large language or vision-language models, their metric accuracy is well below the state of the art. In contrast, we present PromptHMR, a transformer-based promptable method that reformulates HPS estimation through spatial and semantic prompts. Our method processes full images to maintain scene context and accepts multiple input modalities: spatial prompts like bounding boxes and masks, and semantic prompts like language descriptions or interaction labels. PromptHMR demonstrates robust performance across challenging scenarios: estimating people from bounding boxes as small as faces in crowded scenes, improving body shape estimation through language descriptions, modeling person-person interactions, and producing temporally coherent motions in videos. Experiments on benchmarks show that PromptHMR achieves state-of-the-art performance while offering flexible prompt-based control over the HPS estimation process.
>
---
#### [replaced 007] RSTeller: Scaling Up Visual Language Modeling in Remote Sensing with Rich Linguistic Semantics from Openly Available Data and Large Language Models
- **分类: cs.CV; cs.AI; I.4.8; I.2.10**

- **链接: [http://arxiv.org/pdf/2408.14744v4](http://arxiv.org/pdf/2408.14744v4)**

> **作者:** Junyao Ge; Xu Zhang; Yang Zheng; Kaitai Guo; Jimin Liang
>
> **备注:** Published on ISPRS, minor typos corrected
>
> **摘要:** Abundant, well-annotated multimodal data in remote sensing are pivotal for aligning complex visual remote sensing (RS) scenes with human language, enabling the development of specialized vision language models across diverse RS interpretation tasks. However, annotating RS images with rich linguistic semantics at scale demands expertise in RS and substantial human labor, making it costly and often impractical. In this study, we propose a workflow that leverages large language models (LLMs) to generate multimodal datasets with semantically rich captions at scale from plain OpenStreetMap (OSM) data for images sourced from the Google Earth Engine (GEE) platform. This approach facilitates the generation of paired remote sensing data and can be readily scaled up using openly available data. Within this framework, we present RSTeller, a multimodal dataset comprising over 1.3 million RS images, each accompanied by two descriptive captions. Extensive experiments demonstrate that RSTeller enhances the performance of multiple existing vision language models for RS scene understanding through continual pre-training. Our methodology significantly reduces the manual effort and expertise needed for annotating remote sensing imagery while democratizing access to high-quality annotated data. This advancement fosters progress in visual language modeling and encourages broader participation in remote sensing research and applications. The RSTeller dataset is available at https://github.com/SlytherinGe/RSTeller.
>
---
#### [replaced 008] STAF: Sinusoidal Trainable Activation Functions for Implicit Neural Representation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00869v2](http://arxiv.org/pdf/2502.00869v2)**

> **作者:** Alireza Morsali; MohammadJavad Vaez; Mohammadhossein Soltani; Amirhossein Kazerouni; Babak Taati; Morteza Mohammad-Noori
>
> **摘要:** Implicit Neural Representations (INRs) have emerged as a powerful framework for modeling continuous signals. The spectral bias of ReLU-based networks is a well-established limitation, restricting their ability to capture fine-grained details in target signals. While previous works have attempted to mitigate this issue through frequency-based encodings or architectural modifications, these approaches often introduce additional complexity and do not fully address the underlying challenge of learning high-frequency components efficiently. We introduce Sinusoidal Trainable Activation Functions (STAF), designed to directly tackle this limitation by enabling networks to adaptively learn and represent complex signals with higher precision and efficiency. STAF inherently modulates its frequency components, allowing for self-adaptive spectral learning. This capability significantly improves convergence speed and expressivity, making STAF highly effective for both signal representations and inverse problems. Through extensive evaluations across a range of tasks, including signal representation (shape, image, audio) and inverse problems (super-resolution, denoising), as well as neural radiance fields (NeRF), we demonstrate that STAF consistently outperforms state-of-the-art methods in accuracy and reconstruction fidelity. These results establish STAF as a robust solution to spectral bias and the capacity--convergence tradeoff, with broad applicability in computer vision and graphics. Our codebase is publicly accessible at https://github.com/AlirezaMorsali/STAF.
>
---
#### [replaced 009] Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.13377v2](http://arxiv.org/pdf/2503.13377v2)**

> **作者:** Ye Wang; Ziheng Wang; Boshen Xu; Yang Du; Kejun Lin; Zihan Xiao; Zihao Yue; Jianzhong Ju; Liang Zhang; Dingyi Yang; Xiangnan Fang; Zewen He; Zhenbo Luo; Wenxuan Wang; Junqi Lin; Jian Luan; Qin Jin
>
> **备注:** Project Page: https://xuboshen.github.io/Time-R1/
>
> **摘要:** Temporal Video Grounding (TVG), the task of locating specific video segments based on language queries, is a core challenge in long-form video understanding. While recent Large Vision-Language Models (LVLMs) have shown early promise in tackling TVG through supervised fine-tuning (SFT), their abilities to generalize remain limited. To address this, we propose a novel post-training framework that enhances the generalization capabilities of LVLMs via reinforcement learning (RL). Specifically, our contributions span three key directions: (1) Time-R1: we introduce a reasoning-guided post-training framework via RL with verifiable reward to enhance the capabilities of LVLMs on the TVG task. (2) TimeRFT: we explore data-efficient post-training strategies on our curated RL-friendly dataset, which trains the model to progressively comprehend difficult samples, leading to better generalization. (3) TVGBench: we carefully construct a small yet comprehensive benchmark for LVLM evaluation, assessing 11 types of queries and featuring balanced distributions across both videos and queries. Extensive experiments demonstrate that Time-R1 achieves state-of-the-art performance across multiple downstream datasets using only 2.5K training data, while improving its general video understanding capabilities.
>
---
#### [replaced 010] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.06510v2](http://arxiv.org/pdf/2404.06510v2)**

> **作者:** Yuan-Hong Liao; Rafid Mahmood; Sanja Fidler; David Acuna
>
> **备注:** Accepted at CVPR 2025. 22 pages, 16 figures
>
> **摘要:** Enhancing semantic grounding abilities in Vision-Language Models (VLMs) often involves collecting domain-specific training data, refining the network architectures, or modifying the training recipes. In this work, we venture into an orthogonal direction and explore whether VLMs can improve their semantic grounding by "receiving" feedback, without requiring in-domain data, fine-tuning, or modifications to the network architectures. We systematically analyze this hypothesis using a feedback mechanism composed of a binary signal. We find that if prompted appropriately, VLMs can utilize feedback both in a single step and iteratively, showcasing the potential of feedback as an alternative technique to improve grounding in internet-scale VLMs. Furthermore, VLMs, like LLMs, struggle to self-correct errors out-of-the-box. However, we find that this issue can be mitigated via a binary verification mechanism. Finally, we explore the potential and limitations of amalgamating these findings and applying them iteratively to automatically enhance VLMs' grounding performance, showing grounding accuracy consistently improves using automated feedback across all models in all settings investigated. Overall, our iterative framework improves semantic grounding in VLMs by more than 15 accuracy points under noise-free feedback and up to 5 accuracy points under a simple automated binary verification mechanism. The project website is hosted at https://andrewliao11.github.io/vlms_feedback
>
---
#### [replaced 011] Glioma Multimodal MRI Analysis System for Tumor Layered Diagnosis via Multi-task Semi-supervised Learning
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.17758v2](http://arxiv.org/pdf/2501.17758v2)**

> **作者:** Yihao Liu; Zhihao Cui; Liming Li; Junjie You; Xinle Feng; Jianxin Wang; Xiangyu Wang; Qing Liu; Minghua Wu
>
> **备注:** 22 pages
>
> **摘要:** Gliomas are the most common primary tumors of the central nervous system. Multimodal MRI is widely used for the preliminary screening of gliomas and plays a crucial role in auxiliary diagnosis, therapeutic efficacy, and prognostic evaluation. Currently, the computer-aided diagnostic studies of gliomas using MRI have focused on independent analysis events such as tumor segmentation, grading, and radiogenomic classification, without studying inter-dependencies among these events. In this study, we propose a Glioma Multimodal MRI Analysis System (GMMAS) that utilizes a deep learning network for processing multiple events simultaneously, leveraging their inter-dependencies through an uncertainty-based multi-task learning architecture and synchronously outputting tumor region segmentation, glioma histological subtype, IDH mutation genotype, and 1p/19q chromosome disorder status. Compared with the reported single-task analysis models, GMMAS improves the precision across tumor layered diagnostic tasks. Additionally, we have employed a two-stage semi-supervised learning method, enhancing model performance by fully exploiting both labeled and unlabeled MRI samples. Further, by utilizing an adaptation module based on knowledge self-distillation and contrastive learning for cross-modal feature extraction, GMMAS exhibited robustness in situations of modality absence and revealed the differing significance of each MRI modal. Finally, based on the analysis outputs of the GMMAS, we created a visual and user-friendly platform for doctors and patients, introducing GMMAS-GPT to generate personalized prognosis evaluations and suggestions.
>
---
#### [replaced 012] Stay-Positive: A Case for Ignoring Real Image Features in Fake Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07778v2](http://arxiv.org/pdf/2502.07778v2)**

> **作者:** Anirudh Sundara Rajan; Yong Jae Lee
>
> **摘要:** Detecting AI generated images is a challenging yet essential task. A primary difficulty arises from the detectors tendency to rely on spurious patterns, such as compression artifacts, which can influence its decisions. These issues often stem from specific patterns that the detector associates with the real data distribution, making it difficult to isolate the actual generative traces. We argue that an image should be classified as fake if and only if it contains artifacts introduced by the generative model. Based on this premise, we propose Stay Positive, an algorithm designed to constrain the detectors focus to generative artifacts while disregarding those associated with real data. Experimental results demonstrate that detectors trained with Stay Positive exhibit reduced susceptibility to spurious correlations, leading to improved generalization and robustness to post processing. Additionally, unlike detectors that associate artifacts with real images, those that focus purely on fake artifacts are better at detecting inpainted real images.
>
---
#### [replaced 013] SITCOM: Step-wise Triple-Consistent Diffusion Sampling for Inverse Problems
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.04479v2](http://arxiv.org/pdf/2410.04479v2)**

> **作者:** Ismail Alkhouri; Shijun Liang; Cheng-Han Huang; Jimmy Dai; Qing Qu; Saiprasad Ravishankar; Rongrong Wang
>
> **摘要:** Diffusion models (DMs) are a class of generative models that allow sampling from a distribution learned over a training set. When applied to solving inverse problems, the reverse sampling steps are modified to approximately sample from a measurement-conditioned distribution. However, these modifications may be unsuitable for certain settings (e.g., presence of measurement noise) and non-linear tasks, as they often struggle to correct errors from earlier steps and generally require a large number of optimization and/or sampling steps. To address these challenges, we state three conditions for achieving measurement-consistent diffusion trajectories. Building on these conditions, we propose a new optimization-based sampling method that not only enforces standard data manifold measurement consistency and forward diffusion consistency, as seen in previous studies, but also incorporates our proposed step-wise and network-regularized backward diffusion consistency that maintains a diffusion trajectory by optimizing over the input of the pre-trained model at every sampling step. By enforcing these conditions (implicitly or explicitly), our sampler requires significantly fewer reverse steps. Therefore, we refer to our method as Step-wise Triple-Consistent Sampling (SITCOM). Compared to SOTA baselines, our experiments across several linear and non-linear tasks (with natural and medical images) demonstrate that SITCOM achieves competitive or superior results in terms of standard similarity metrics and run-time.
>
---
#### [replaced 014] GenAnalysis: Joint Shape Analysis by Learning Man-Made Shape Generators with Deformation Regularizations
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.00807v2](http://arxiv.org/pdf/2503.00807v2)**

> **作者:** Yuezhi Yang; Haitao Yang; Kiyohiro Nakayama; Xiangru Huang; Leonidas Guibas; Qixing Huang
>
> **备注:** 19 pages, 24 figures
>
> **摘要:** We present GenAnalysis, an implicit shape generation framework that allows joint analysis of man-made shapes, including shape matching and joint shape segmentation. The key idea is to enforce an as-affine-as-possible (AAAP) deformation between synthetic shapes of the implicit generator that are close to each other in the latent space, which we achieve by designing a regularization loss. It allows us to understand the shape variation of each shape in the context of neighboring shapes and also offers structure-preserving interpolations between the input shapes. We show how to extract these shape variations by recovering piecewise affine vector fields in the tangent space of each shape. These vector fields provide single-shape segmentation cues. We then derive shape correspondences by iteratively propagating AAAP deformations across a sequence of intermediate shapes. These correspondences are then used to aggregate single-shape segmentation cues into consistent segmentations. We conduct experiments on the ShapeNet dataset to show superior performance in shape matching and joint shape segmentation over previous methods.
>
---
#### [replaced 015] SpecDETR: A Transformer-based Hyperspectral Point Object Detection Network
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.10148v3](http://arxiv.org/pdf/2405.10148v3)**

> **作者:** Zhaoxu Li; Wei An; Gaowei Guo; Longguang Wang; Yingqian Wang; Zaiping Lin
>
> **摘要:** Hyperspectral target detection (HTD) aims to identify specific materials based on spectral information in hyperspectral imagery and can detect extremely small-sized objects, some of which occupy a smaller than one-pixel area. However, existing HTD methods are developed based on per-pixel binary classification, neglecting the three-dimensional cube structure of hyperspectral images (HSIs) that integrates both spatial and spectral dimensions. The synergistic existence of spatial and spectral features in HSIs enable objects to simultaneously exhibit both, yet the per-pixel HTD framework limits the joint expression of these features. In this paper, we rethink HTD from the perspective of spatial-spectral synergistic representation and propose hyperspectral point object detection as an innovative task framework. We introduce SpecDETR, the first specialized network for hyperspectral multi-class point object detection, which eliminates dependence on pre-trained backbone networks commonly required by vision-based object detectors. SpecDETR uses a multi-layer Transformer encoder with self-excited subpixel-scale attention modules to directly extract deep spatial-spectral joint features from hyperspectral cubes. We develop a simulated hyperspectral point object detection benchmark termed SPOD, and for the first time, evaluate and compare the performance of visual object detection networks and HTD methods on hyperspectral point object detection. Extensive experiments demonstrate that our proposed SpecDETR outperforms SOTA visual object detection networks and HTD methods. Our code and dataset are available at https://github.com/ZhaoxuLi123/SpecDETR.
>
---
#### [replaced 016] Ocular Authentication: Fusion of Gaze and Periocular Modalities
- **分类: cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.17343v2](http://arxiv.org/pdf/2505.17343v2)**

> **作者:** Dillon Lohr; Michael J. Proulx; Mehedi Hasan Raju; Oleg V. Komogortsev
>
> **备注:** Supplementary material is available
>
> **摘要:** This paper investigates the feasibility of fusing two eye-centric authentication modalities-eye movements and periocular images-within a calibration-free authentication system. While each modality has independently shown promise for user authentication, their combination within a unified gaze-estimation pipeline has not been thoroughly explored at scale. In this report, we propose a multimodal authentication system and evaluate it using a large-scale in-house dataset comprising 9202 subjects with an eye tracking (ET) signal quality equivalent to a consumer-facing virtual reality (VR) device. Our results show that the multimodal approach consistently outperforms both unimodal systems across all scenarios, surpassing the FIDO benchmark. The integration of a state-of-the-art machine learning architecture contributed significantly to the overall authentication performance at scale, driven by the model's ability to capture authentication representations and the complementary discriminative characteristics of the fused modalities.
>
---
#### [replaced 017] Role Bias in Text-to-Image Diffusion Models: Diagnosing and Mitigating Compositional Failures through Intermediate Decomposition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10037v2](http://arxiv.org/pdf/2503.10037v2)**

> **作者:** Sina Malakouti; Adriana Kovashka
>
> **摘要:** Text-to-image (T2I) diffusion models exhibit impressive photorealistic image generation capabilities, yet they struggle in compositional image generation. In this work, we introduce RoleBench, a benchmark focused on evaluating compositional generalization in action-based relations (e.g., "mouse chasing cat"). We show that state-of-the-art T2I models and compositional approaches consistently default to frequent reversed relations (i.e., cat chasing mouse), a phenomenon we call RoleCollapse. Related works attribute this to the model's architectural limitation or being underrepresented in the data. Our key insight reveals that while models fail on rare compositions when their inversions are common, they can successfully generate similar intermediate compositions (e.g., "mouse chasing boy"), suggesting that this limitation is due to the presence of frequent counterparts rather than the absence of rare compositions. Motivated by this, we hypothesize that directional decomposition can gradually mitigate role collapse. We test this via ReBind, a lightweight framework that teaches role bindings using carefully selected active/passive intermediaries. Experiments suggest that intermediate compositions through intermediate fine-tuning can significantly mitigate role bias, with humans preferring more than 78% compared to state-of-the-art methods. Our findings highlight the role of distributional asymmetries in compositional failures and offer a simple, effective path to improving generalization.
>
---
#### [replaced 018] From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22976v4](http://arxiv.org/pdf/2503.22976v4)**

> **作者:** Jiahui Zhang; Yurui Chen; Yanpeng Zhou; Yueming Xu; Ze Huang; Jilin Mei; Junhui Chen; Yu-Jie Yuan; Xinyue Cai; Guowei Huang; Xingyue Quan; Hang Xu; Li Zhang
>
> **备注:** Project page: https://fudan-zvg.github.io/spar
>
> **摘要:** Recent advances in LVLMs have improved vision-language understanding, but they still struggle with spatial perception, limiting their ability to reason about complex 3D scenes. Unlike previous approaches that incorporate 3D representations into models to improve spatial understanding, we aim to unlock the potential of VLMs by leveraging spatially relevant image data. To this end, we introduce a novel 2D spatial data generation and annotation pipeline built upon scene data with 3D ground-truth. This pipeline enables the creation of a diverse set of spatial tasks, ranging from basic perception tasks to more complex reasoning tasks. Leveraging this pipeline, we construct SPAR-7M, a large-scale dataset generated from thousands of scenes across multiple public datasets. In addition, we introduce SPAR-Bench, a benchmark designed to offer a more comprehensive evaluation of spatial capabilities compared to existing spatial benchmarks, supporting both single-view and multi-view inputs. Training on both SPAR-7M and large-scale 2D datasets enables our models to achieve state-of-the-art performance on 2D spatial benchmarks. Further fine-tuning on 3D task-specific datasets yields competitive results, underscoring the effectiveness of our dataset in enhancing spatial reasoning.
>
---
#### [replaced 019] Rethinking Edge Detection through Perceptual Asymmetry: The SWBCE Loss
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.13365v2](http://arxiv.org/pdf/2501.13365v2)**

> **作者:** Hao Shu
>
> **备注:** 27 pages
>
> **摘要:** Edge detection (ED) is a fundamental component in many computer vision tasks, yet achieving both high quantitative accuracy and perceptual quality remains a significant challenge. In this paper, we propose the Symmetrization Weighted Binary Cross-Entropy (SWBCE) loss function, a novel approach that addresses this issue by leveraging the inherent asymmetry in human edge perception, where edge decisions require stronger justification than non-edge ones. By balancing label-guided and prediction-guided learning, SWBCE maintains high edge recall while effectively suppressing false positives. Extensive experiments across multiple datasets and baseline models, along with comparisons to prior loss functions, demonstrate that our method consistently improves both the quantitative metrics and perceptual quality of ED results. These findings underscore the effectiveness of SWBCE for high-quality edge prediction and its potential applicability to related vision tasks.
>
---
#### [replaced 020] VersatileMotion: A Unified Framework for Motion Synthesis and Comprehension
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17335v2](http://arxiv.org/pdf/2411.17335v2)**

> **作者:** Zeyu Ling; Bo Han; Shiyang Li; Jikang Cheng; Hongdeng Shen; Changqing Zou
>
> **摘要:** Large language models (LLMs) are, by design, inherently capable of multi-task learning: through a unified next-token prediction paradigm, they can naturally address a wide variety of downstream tasks. Prior work in the motion domain has demonstrated some generality by adapting LLMs via a Motion Tokenizer coupled with an autoregressive Transformer to generate and understand human motion. However, this generality remains limited in scope and yields only modest performance gains. We introduce VersatileMotion, a unified multimodal motion LLM that combines a novel motion tokenizer, integrating VQ-VAE with flow matching, and an autoregressive transformer backbone to seamlessly support at least nine distinct motion-related tasks. VersatileMotion is the first method to handle single-agent and multi-agent motions in a single framework and enable cross-modal conversion between motion, text, music, and speech, achieving state-of-the-art performance on seven of these tasks. Each sequence in MotionHub may include one or more of the following annotations: natural-language captions, music or audio clips, speech transcripts, and multi-agent interaction data. To facilitate evaluation, we define and release benchmark splits covering nine core tasks. Extensive experiments demonstrate the superior performance, versatility, and potential of VersatileMotion as a foundational model for future understanding and generation of motion.
>
---
#### [replaced 021] TrackRAD2025 challenge dataset: Real-time tumor tracking for MRI-guided radiotherapy
- **分类: physics.med-ph; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19119v2](http://arxiv.org/pdf/2503.19119v2)**

> **作者:** Yiling Wang; Elia Lombardo; Adrian Thummerer; Tom Blöcker; Yu Fan; Yue Zhao; Christianna Iris Papadopoulou; Coen Hurkmans; Rob H. N. Tijssen; Pia A. W. Görts; Shyama U. Tetar; Davide Cusumano; Martijn P. W. Intven; Pim Borman; Marco Riboldi; Denis Dudáš; Hilary Byrne; Lorenzo Placidi; Marco Fusella; Michael Jameson; Miguel Palacios; Paul Cobussen; Tobias Finazzi; Cornelis J. A. Haasbeek; Paul Keall; Christopher Kurz; Guillaume Landry; Matteo Maspero
>
> **备注:** 10 pages, 5 figures, 2 tables; submitted to Medical Physics, tentatively accepted
>
> **摘要:** Purpose: Magnetic resonance imaging (MRI) to visualize anatomical motion is becoming increasingly important when treating cancer patients with radiotherapy. Hybrid MRI-linear accelerator (MRI-linac) systems allow real-time motion management during irradiation. This paper presents a multi-institutional real-time MRI time series dataset from different MRI-linac vendors. The dataset is designed to support developing and evaluating real-time tumor localization (tracking) algorithms for MRI-guided radiotherapy within the TrackRAD2025 challenge (https://trackrad2025.grand-challenge.org/). Acquisition and validation methods: The dataset consists of sagittal 2D cine MRIs in 585 patients from six centers (3 Dutch, 1 German, 1 Australian, and 1 Chinese). Tumors in the thorax, abdomen, and pelvis acquired on two commercially available MRI-linacs (0.35 T and 1.5 T) were included. For 108 cases, irradiation targets or tracking surrogates were manually segmented on each temporal frame. The dataset was randomly split into a public training set of 527 cases (477 unlabeled and 50 labeled) and a private testing set of 58 cases (all labeled). Data Format and Usage Notes: The data is publicly available under the TrackRAD2025 collection: https://doi.org/10.57967/hf/4539. Both the images and segmentations for each patient are available in metadata format. Potential Applications: This novel clinical dataset will enable the development and evaluation of real-time tumor localization algorithms for MRI-guided radiotherapy. By enabling more accurate motion management and adaptive treatment strategies, this dataset has the potential to advance the field of radiotherapy significantly.
>
---
#### [replaced 022] DiMeR: Disentangled Mesh Reconstruction Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17670v2](http://arxiv.org/pdf/2504.17670v2)**

> **作者:** Lutao Jiang; Jiantao Lin; Kanghao Chen; Wenhang Ge; Xin Yang; Yifan Jiang; Yuanhuiyi Lyu; Xu Zheng; Yinchuan Li; Yingcong Chen
>
> **备注:** Project Page: https://lutao2021.github.io/DiMeR_page/
>
> **摘要:** We propose DiMeR, a novel geometry-texture disentangled feed-forward model with 3D supervision for sparse-view mesh reconstruction. Existing methods confront two persistent obstacles: (i) textures can conceal geometric errors, i.e., visually plausible images can be rendered even with wrong geometry, producing multiple ambiguous optimization objectives in geometry-texture mixed solution space for similar objects; and (ii) prevailing mesh extraction methods are redundant, unstable, and lack 3D supervision. To solve these challenges, we rethink the inductive bias for mesh reconstruction. First, we disentangle the unified geometry-texture solution space, where a single input admits multiple feasible solutions, into geometry and texture spaces individually. Specifically, given that normal maps are strictly consistent with geometry and accurately capture surface variations, the normal maps serve as the sole input for geometry prediction in DiMeR, while the texture is estimated from RGB images. Second, we streamline the algorithm of mesh extraction by eliminating modules with low performance/cost ratios and redesigning regularization losses with 3D supervision. Notably, DiMeR still accepts raw RGB images as input by leveraging foundation models for normal prediction. Extensive experiments demonstrate that DiMeR generalises across sparse-view-, single-image-, and text-to-3D tasks, consistently outperforming baselines. On the GSO and OmniObject3D datasets, DiMeR significantly reduces Chamfer Distance by more than 30%.
>
---
#### [replaced 023] Human-Aligned Bench: Fine-Grained Assessment of Reasoning Ability in MLLMs vs. Humans
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11141v2](http://arxiv.org/pdf/2505.11141v2)**

> **作者:** Yansheng Qiu; Li Xiao; Zhaopan Xu; Pengfei Zhou; Zheng Wang; Kaipeng Zhang
>
> **摘要:** The goal of achieving Artificial General Intelligence (AGI) is to imitate humans and surpass them. Models such as OpenAI's o1, o3, and DeepSeek's R1 have demonstrated that large language models (LLMs) with human-like reasoning capabilities exhibit exceptional performance and are being gradually integrated into multimodal large language models (MLLMs). However, whether these models possess capabilities comparable to humans in handling reasoning tasks remains unclear at present. In this paper, we propose Human-Aligned Bench, a benchmark for fine-grained alignment of multimodal reasoning with human performance. Specifically, we collected 9,794 multimodal questions that solely rely on contextual reasoning, including bilingual (Chinese and English) multimodal questions and pure text-based questions, encompassing four question types: visual reasoning, definition judgment, analogical reasoning, and logical judgment. More importantly, each question is accompanied by human success rates and options that humans are prone to choosing incorrectly. Extensive experiments on the Human-Aligned Bench reveal notable differences between the performance of current MLLMs in multimodal reasoning and human performance. The findings on our benchmark provide insights into the development of the next-generation models.
>
---
#### [replaced 024] Cancer-Net PCa-Seg: Benchmarking Deep Learning Models for Prostate Cancer Segmentation Using Synthetic Correlated Diffusion Imaging
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.09185v2](http://arxiv.org/pdf/2501.09185v2)**

> **作者:** Jarett Dewbury; Chi-en Amy Tai; Alexander Wong
>
> **备注:** 8 pages, 2 figures, to be published in Studies in Computational Intelligence. This paper introduces Cancer-Net PCa-Seg, a comprehensive evaluation of deep learning models for prostate cancer segmentation using synthetic correlated diffusion imaging (CDI$^s$). We benchmark five state-of-the-art architectures: U-Net, SegResNet, Swin UNETR, Attention U-Net, and LightM-UNet
>
> **摘要:** Prostate cancer (PCa) is the most prevalent cancer among men in the United States, accounting for nearly 300,000 cases, 29\% of all diagnoses and 35,000 total deaths in 2024. Traditional screening methods such as prostate-specific antigen (PSA) testing and magnetic resonance imaging (MRI) have been pivotal in diagnosis, but have faced limitations in specificity and generalizability. In this paper, we explore the potential of enhancing PCa gland segmentation using a novel MRI modality called synthetic correlated diffusion imaging (CDI$^s$). We employ several state-of-the-art deep learning models, including U-Net, SegResNet, Swin UNETR, Attention U-Net, and LightM-UNet, to segment prostate glands from a 200 CDI$^s$ patient cohort. We find that SegResNet achieved superior segmentation performance with a Dice-Sorensen coefficient (DSC) of $76.68 \pm 0.8$. Notably, the Attention U-Net, while slightly less accurate (DSC $74.82 \pm 2.0$), offered a favorable balance between accuracy and computational efficiency. Our findings demonstrate the potential of deep learning models in improving prostate gland segmentation using CDI$^s$ to enhance PCa management and clinical support.
>
---
#### [replaced 025] One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11131v2](http://arxiv.org/pdf/2505.11131v2)**

> **作者:** Feiran Li; Qianqian Xu; Shilong Bao; Zhiyong Yang; Xiaochun Cao; Qingming Huang
>
> **备注:** This paper has been accepeted to ICML 2025
>
> **摘要:** Concept erasing has recently emerged as an effective paradigm to prevent text-to-image diffusion models from generating visually undesirable or even harmful content. However, current removal methods heavily rely on manually crafted text prompts, making it challenging to achieve a high erasure (efficacy) while minimizing the impact on other benign concepts (usability). In this paper, we attribute the limitations to the inherent gap between the text and image modalities, which makes it hard to transfer the intricately entangled concept knowledge from text prompts to the image generation process. To address this, we propose a novel solution by directly integrating visual supervision into the erasure process, introducing the first text-image Collaborative Concept Erasing (Co-Erasing) framework. Specifically, Co-Erasing describes the concept jointly by text prompts and the corresponding undesirable images induced by the prompts, and then reduces the generating probability of the target concept through negative guidance. This approach effectively bypasses the knowledge gap between text and image, significantly enhancing erasure efficacy. Additionally, we design a text-guided image concept refinement strategy that directs the model to focus on visual features most relevant to the specified text concept, minimizing disruption to other benign concepts. Finally, comprehensive experiments suggest that Co-Erasing outperforms state-of-the-art erasure approaches significantly with a better trade-off between efficacy and usability. Codes are available at https://github.com/Ferry-Li/Co-Erasing.
>
---
#### [replaced 026] Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11492v3](http://arxiv.org/pdf/2502.11492v3)**

> **作者:** Kung-Hsiang Huang; Can Qin; Haoyi Qiu; Philippe Laban; Shafiq Joty; Caiming Xiong; Chien-Sheng Wu
>
> **备注:** Code and data are available at https://github.com/SalesforceAIResearch/CogAlign
>
> **摘要:** Vision Language Models (VLMs) have achieved remarkable progress in multimodal tasks, yet they often struggle with visual arithmetic, seemingly simple capabilities like object counting or length comparison, which are essential for relevant complex tasks like chart understanding and geometric reasoning. In this work, we first investigate the root causes of this deficiency through a suite of probing tasks focusing on basic visual arithmetic. Our analysis reveals that while pre-trained vision encoders typically capture sufficient information, the text decoder often fails to decode it correctly for arithmetic reasoning. To address this, we propose CogAlign, a novel post-training strategy inspired by Piaget's theory of cognitive development. CogAlign trains VLMs to recognize invariant properties under visual transformations. We demonstrate that this approach significantly improves the performance of three diverse VLMs on our proposed probing tasks. Furthermore, CogAlign enhances performance by an average of 4.6% on CHOCOLATE and 2.9% on MATH-VISION, outperforming or matching supervised fine-tuning methods while requiring only 60% less training data. These results highlight the effectiveness and generalizability of CogAlign in improving fundamental visual arithmetic capabilities and their transfer to downstream tasks.
>
---
#### [replaced 027] RefinedFields: Radiance Fields Refinement for Planar Scene Representations
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.00639v4](http://arxiv.org/pdf/2312.00639v4)**

> **作者:** Karim Kassab; Antoine Schnepf; Jean-Yves Franceschi; Laurent Caraffa; Jeremie Mary; Valérie Gouet-Brunet
>
> **备注:** Accepted at TMLR
>
> **摘要:** Planar scene representations have recently witnessed increased interests for modeling scenes from images, as their lightweight planar structure enables compatibility with image-based models. Notably, K-Planes have gained particular attention as they extend planar scene representations to support in-the-wild scenes, in addition to object-level scenes. However, their visual quality has recently lagged behind that of state-of-the-art techniques. To reduce this gap, we propose RefinedFields, a method that leverages pre-trained networks to refine K-Planes scene representations via optimization guidance using an alternating training procedure. We carry out extensive experiments and verify the merit of our method on synthetic data and real tourism photo collections. RefinedFields enhances rendered scenes with richer details and improves upon its base representation on the task of novel view synthesis. Our project page can be found at https://refinedfields.github.io .
>
---
#### [replaced 028] NeuRadar: Neural Radiance Fields for Automotive Radar Point Clouds
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.00859v3](http://arxiv.org/pdf/2504.00859v3)**

> **作者:** Mahan Rafidashti; Ji Lan; Maryam Fatemi; Junsheng Fu; Lars Hammarstrand; Lennart Svensson
>
> **摘要:** Radar is an important sensor for autonomous driving (AD) systems due to its robustness to adverse weather and different lighting conditions. Novel view synthesis using neural radiance fields (NeRFs) has recently received considerable attention in AD due to its potential to enable efficient testing and validation but remains unexplored for radar point clouds. In this paper, we present NeuRadar, a NeRF-based model that jointly generates radar point clouds, camera images, and lidar point clouds. We explore set-based object detection methods such as DETR, and propose an encoder-based solution grounded in the NeRF geometry for improved generalizability. We propose both a deterministic and a probabilistic point cloud representation to accurately model the radar behavior, with the latter being able to capture radar's stochastic behavior. We achieve realistic reconstruction results for two automotive datasets, establishing a baseline for NeRF-based radar point cloud simulation models. In addition, we release radar data for ZOD's Sequences and Drives to enable further research in this field. To encourage further development of radar NeRFs, we release the source code for NeuRadar.
>
---
#### [replaced 029] Visual Program Distillation with Template-Based Augmentation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.08564v3](http://arxiv.org/pdf/2412.08564v3)**

> **作者:** Michal Shlapentokh-Rothman; Yu-Xiong Wang; Derek Hoiem
>
> **摘要:** Adapting visual programming or prompting large language models (LLMs) to generate executable code for visual tasks like visual question answering (VQA) for specialized tasks or domains remains challenging due to high annotation and inference costs. We propose a low-cost visual program distillation method that can be used for models with at most 1 billion parameters and requires no human-generated program annotations. We achieve this through synthetic data augmentation based on decoupling programs into higher-level skills, called templates, and their corresponding arguments. Experimental results show that, with a relatively small amount of question/answer data, small language models can generate high-quality specialized visual programs with the added benefit of much faster inference
>
---
#### [replaced 030] OpenAD: Open-World Autonomous Driving Benchmark for 3D Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17761v2](http://arxiv.org/pdf/2411.17761v2)**

> **作者:** Zhongyu Xia; Jishuo Li; Zhiwei Lin; Xinhao Wang; Yongtao Wang; Ming-Hsuan Yang
>
> **摘要:** Open-world perception aims to develop a model adaptable to novel domains and various sensor configurations and can understand uncommon objects and corner cases. However, current research lacks sufficiently comprehensive open-world 3D perception benchmarks and robust generalizable methodologies. This paper introduces OpenAD, the first real open-world autonomous driving benchmark for 3D object detection. OpenAD is built upon a corner case discovery and annotation pipeline that integrates with a multimodal large language model (MLLM). The proposed pipeline annotates corner case objects in a unified format for five autonomous driving perception datasets with 2000 scenarios. In addition, we devise evaluation methodologies and evaluate various open-world and specialized 2D and 3D models. Moreover, we propose a vision-centric 3D open-world object detection baseline and further introduce an ensemble method by fusing general and specialized models to address the issue of lower precision in existing open-world methods for the OpenAD benchmark. We host an online challenge on EvalAI. Data, toolkit codes, and evaluation codes are available at https://github.com/VDIGPKU/OpenAD.
>
---
#### [replaced 031] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13055v2](http://arxiv.org/pdf/2504.13055v2)**

> **作者:** Xiangyan Liu; Jinjie Ni; Zijian Wu; Chao Du; Longxu Dou; Haonan Wang; Tianyu Pang; Michael Qizhe Shieh
>
> **备注:** Technical Report
>
> **摘要:** Recent advances in reinforcement learning (RL) have strengthened the reasoning capabilities of vision-language models (VLMs). However, enhancing policy exploration to better scale test-time compute remains largely underexplored. In addition, VLMs continue to struggle with imperfect visual perception, which in turn affects the subsequent reasoning process. To this end, we propose NoisyRollout, a simple yet effective data augmentation method that mixes trajectories from both clean and moderately distorted images during RL training. By injecting targeted diversity in visual perception and the resulting reasoning patterns, NoisyRollout promotes better policy exploration through vision-oriented inductive biases, ultimately leading to more robust reasoning behaviors. We further adopt a noise annealing schedule that gradually reduces distortion strength over training, leveraging noisy signals early on while ensuring training stability in later stages. Crucially, our method is easy-to-adopt--requiring no additional training cost and no modifications to the RL objective. Extensive experiments on $2$ distinct training datasets demonstrate that NoisyRollout achieves state-of-the-art performance among open-source RL-tuned models across $5$ out-of-domain reasoning and perception benchmarks. Furthermore, we validate the effectiveness of NoisyRollout across model sizes ($7$B and $32$B) and data scales (from $1$K to $6$K), highlighting its generalizability and scalability.
>
---
#### [replaced 032] Unsupervised Anomaly Detection Using Diffusion Trend Analysis for Display Inspection
- **分类: cs.CV; cs.LG; 68T45 (Primary) 68T27 (Secondary); I.2.10**

- **链接: [http://arxiv.org/pdf/2407.09578v2](http://arxiv.org/pdf/2407.09578v2)**

> **作者:** Eunwoo Kim; Un Yang; Cheol Lae Roh; Stefano Ermon
>
> **备注:** 4 pages, 5 figures, 2 tables. To be published in the SID Digest of Technical Papers
>
> **摘要:** Reconstruction-based anomaly detection via denoising diffusion model has limitations in determining appropriate noise parameters that can degrade anomalies while preserving normal characteristics. Also, normal regions can fluctuate considerably during reconstruction, resulting in false detection. In this paper, we propose a method to detect anomalies by analysis of reconstruction trend depending on the degree of degradation, effectively solving the both problems that impede practical application in display inspection.
>
---
#### [replaced 033] Polynomial, trigonometric, and tropical activations
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; math.AG**

- **链接: [http://arxiv.org/pdf/2502.01247v2](http://arxiv.org/pdf/2502.01247v2)**

> **作者:** Ismail Khalfaoui-Hassani; Stefan Kesselheim
>
> **摘要:** Which functions can be used as activations in deep neural networks? This article explores families of functions based on orthonormal bases, including the Hermite polynomial basis and the Fourier trigonometric basis, as well as a basis resulting from the tropicalization of a polynomial basis. Our study shows that, through simple variance-preserving initialization and without additional clamping mechanisms, these activations can successfully be used to train deep models, such as GPT-2 for next-token prediction on OpenWebText and ConvNeXt for image classification on ImageNet. Our work addresses the issue of exploding and vanishing activations and gradients, particularly prevalent with polynomial activations, and opens the door for improving the efficiency of large-scale learning tasks. Furthermore, our approach provides insight into the structure of neural networks, revealing that networks with polynomial activations can be interpreted as multivariate polynomial mappings. Finally, using Hermite interpolation, we show that our activations can closely approximate classical ones in pre-trained models by matching both the function and its derivative, making them especially useful for fine-tuning tasks. These activations are available in the torchortho library, which can be accessed via: https://github.com/K-H-Ismail/torchortho.
>
---
#### [replaced 034] Efficient Flow Matching using Latent Variables
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.04486v2](http://arxiv.org/pdf/2505.04486v2)**

> **作者:** Anirban Samaddar; Yixuan Sun; Viktor Nilsson; Sandeep Madireddy
>
> **摘要:** Flow matching models have shown great potential in image generation tasks among probabilistic generative models. However, most flow matching models in the literature do not explicitly model the underlying structure/manifold in the target data when learning the flow from a simple source distribution like the standard Gaussian. This leads to inefficient learning, especially for many high-dimensional real-world datasets, which often reside in a low-dimensional manifold. Existing strategies of incorporating manifolds, including data with underlying multi-modal distribution, often require expensive training and hence frequently lead to suboptimal performance. To this end, we present $\texttt{Latent-CFM}$, which provides simplified training/inference strategies to incorporate multi-modal data structures using pretrained deep latent variable models. Through experiments on multi-modal synthetic data and widely used image benchmark datasets, we show that $\texttt{Latent-CFM}$ exhibits improved generation quality with significantly less training (up to $\sim 50\%$ less) and computation than state-of-the-art flow matching models by incorporating extracted data features using pretrained lightweight latent variable models. Moving beyond natural images to generating fields arising from processes governed by physics, using a 2d Darcy flow dataset, we demonstrate that our approach generates more physically accurate samples than competitive approaches. In addition, through latent space analysis, we demonstrate that our approach can be used for conditional image generation conditioned on latent features, which adds interpretability to the generation process.
>
---
#### [replaced 035] Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18578v3](http://arxiv.org/pdf/2503.18578v3)**

> **作者:** Tianyu Chen; Xingcheng Fu; Yisen Gao; Haodong Qian; Yuecen Wei; Kun Yan; Haoyi Zhou; Jianxin Li
>
> **备注:** CVPR(Highlight)
>
> **摘要:** Modern vision-language models (VLMs) develop patch embedding and convolution backbone within vector space, especially Euclidean ones, at the very founding. When expanding VLMs to a galaxy scale for understanding astronomical phenomena, the integration of spherical space for planetary orbits and hyperbolic spaces for black holes raises two formidable challenges. a) The current pre-training model is confined to Euclidean space rather than a comprehensive geometric embedding. b) The predominant architecture lacks suitable backbones for anisotropic physical geometries. In this paper, we introduced Galaxy-Walker, a geometry-aware VLM, for the universe-level vision understanding tasks. We proposed the geometry prompt that generates geometry tokens by random walks across diverse spaces on a multi-scale physical graph, along with a geometry adapter that compresses and reshapes the space anisotropy in a mixture-of-experts manner. Extensive experiments demonstrate the effectiveness of our approach, with Galaxy-Walker achieving state-of-the-art performance in both galaxy property estimation ($R^2$ scores up to $0.91$) and morphology classification tasks (up to $+0.17$ F1 improvement in challenging features), significantly outperforming both domain-specific models and general-purpose VLMs.
>
---
#### [replaced 036] Fast Video Generation with Sliding Tile Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.04507v2](http://arxiv.org/pdf/2502.04507v2)**

> **作者:** Peiyuan Zhang; Yongqi Chen; Runlong Su; Hangliang Ding; Ion Stoica; Zhengzhong Liu; Hao Zhang
>
> **备注:** Accepted by ICML 25
>
> **摘要:** Diffusion Transformers (DiTs) with 3D full attention power state-of-the-art video generation, but suffer from prohibitive compute cost -- when generating just a 5-second 720P video, attention alone takes 800 out of 945 seconds of total inference time. This paper introduces sliding tile attention (STA) to address this challenge. STA leverages the observation that attention scores in pretrained video diffusion models predominantly concentrate within localized 3D windows. By sliding and attending over the local spatial-temporal region, STA eliminates redundancy from full attention. Unlike traditional token-wise sliding window attention (SWA), STA operates tile-by-tile with a novel hardware-aware sliding window design, preserving expressiveness while being hardware-efficient. With careful kernel-level optimizations, STA offers the first efficient 2D/3D sliding-window-like attention implementation, achieving 58.79% MFU. Precisely, STA accelerates attention by 2.8-17x over FlashAttention-2 (FA2) and 1.6-10x over FlashAttention-3 (FA3). On the leading video DiT, HunyuanVideo, STA reduces end-to-end latency from 945s (FA3) to 685s without quality degradation, requiring no training. Enabling finetuning further lowers latency to 268s with only a 0.09% drop on VBench. We make our codebase public at https://github.com/hao-ai-lab/FastVideo.
>
---
#### [replaced 037] FastVID: Dynamic Density Pruning for Fast Video Large Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11187v2](http://arxiv.org/pdf/2503.11187v2)**

> **作者:** Leqi Shen; Guoqiang Gong; Tao He; Yifeng Zhang; Pengzhang Liu; Sicheng Zhao; Guiguang Ding
>
> **摘要:** Video Large Language Models have demonstrated strong video understanding capabilities, yet their practical deployment is hindered by substantial inference costs caused by redundant video tokens. Existing pruning techniques fail to fully exploit the spatiotemporal redundancy inherent in video data. To bridge this gap, we perform a systematic analysis of video redundancy from two perspectives: temporal context and visual context. Leveraging these insights, we propose Dynamic Density Pruning for Fast Video LLMs termed FastVID. Specifically, FastVID dynamically partitions videos into temporally ordered segments to preserve temporal structure and applies a density-based token pruning strategy to maintain essential visual information. Our method significantly reduces computational overhead while maintaining temporal and visual integrity. Extensive evaluations show that FastVID achieves state-of-the-art performance across various short- and long-video benchmarks on leading Video LLMs, including LLaVA-OneVision and LLaVA-Video. Notably, on LLaVA-OneVision-7B, FastVID effectively prunes $\textbf{90.3%}$ of video tokens, reduces FLOPs to $\textbf{8.3%}$, and accelerates the prefilling stage by $\textbf{7.1}\times$, while maintaining $\textbf{98.0%}$ of the original accuracy. The code is available at https://github.com/LunarShen/FastVID.
>
---
#### [replaced 038] Recursive InPainting (RIP): how much information is lost under recursive inferences?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.09549v2](http://arxiv.org/pdf/2407.09549v2)**

> **作者:** Javier Conde; Miguel González; Gonzalo Martínez; Fernando Moral; Elena Merino-Gómez; Pedro Reviriego
>
> **备注:** AI & Soc (2025)
>
> **摘要:** The rapid adoption of generative artificial intelligence (AI) is accelerating content creation and modification. For example, variations of a given content, be it text or images, can be created almost instantly and at a low cost. This will soon lead to the majority of text and images being created directly by AI models or by humans assisted by AI. This poses new risks; for example, AI-generated content may be used to train newer AI models and degrade their performance, or information may be lost in the transformations made by AI which could occur when the same content is processed over and over again by AI tools. An example of AI image modifications is inpainting in which an AI model completes missing fragments of an image. The incorporation of inpainting tools into photo editing programs promotes their adoption and encourages their recursive use to modify images. Inpainting can be applied recursively, starting from an image, removing some parts, applying inpainting to reconstruct the image, revising it, and then starting the inpainting process again on the reconstructed image, etc. This paper presents an empirical evaluation of recursive inpainting when using one of the most widely used image models: Stable Diffusion. The inpainting process is applied by randomly selecting a fragment of the image, reconstructing it, selecting another fragment, and repeating the process a predefined number of iterations. The images used in the experiments are taken from a publicly available art data set and correspond to different styles and historical periods. Additionally, photographs are also evaluated as a reference. The modified images are compared with the original ones by both using quantitative metrics and performing a qualitative analysis. The results show that recursive inpainting in some cases modifies the image so that it still resembles the original one while in others leads to degeneration.
>
---
#### [replaced 039] Task-Oriented Communications for Visual Navigation with Edge-Aerial Collaboration in Low Altitude Economy
- **分类: cs.CV; cs.NI**

- **链接: [http://arxiv.org/pdf/2504.18317v4](http://arxiv.org/pdf/2504.18317v4)**

> **作者:** Zhengru Fang; Zhenghao Liu; Jingjing Wang; Senkang Hu; Yu Guo; Yiqin Deng; Yuguang Fang
>
> **备注:** Code and dataset will be made publicly available: https://github.com/fangzr/TOC-Edge-Aerial
>
> **摘要:** To support the Low Altitude Economy (LAE), it is essential to achieve precise localization of unmanned aerial vehicles (UAVs) in urban areas where global positioning system (GPS) signals are unavailable. Vision-based methods offer a viable alternative but face severe bandwidth, memory and processing constraints on lightweight UAVs. Inspired by mammalian spatial cognition, we propose a task-oriented communication framework, where UAVs equipped with multi-camera systems extract compact multi-view features and offload localization tasks to edge servers. We introduce the Orthogonally-constrained Variational Information Bottleneck encoder (O-VIB), which incorporates automatic relevance determination (ARD) to prune non-informative features while enforcing orthogonality to minimize redundancy. This enables efficient and accurate localization with minimal transmission cost. Extensive evaluation on a dedicated LAE UAV dataset shows that O-VIB achieves high-precision localization under stringent bandwidth budgets. Code and dataset will be made publicly available at: github.com/fangzr/TOC-Edge-Aerial.
>
---
#### [replaced 040] PointOBB-v3: Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.13898v2](http://arxiv.org/pdf/2501.13898v2)**

> **作者:** Peiyuan Zhang; Junwei Luo; Xue Yang; Yi Yu; Qingyun Li; Yue Zhou; Xiaosong Jia; Xudong Lu; Jingdong Chen; Xiang Li; Junchi Yan; Yansheng Li
>
> **备注:** 33 pages, 7 figures, 11 tables
>
> **摘要:** With the growing demand for oriented object detection (OOD), recent studies on point-supervised OOD have attracted significant interest. In this paper, we propose PointOBB-v3, a stronger single point-supervised OOD framework. Compared to existing methods, it generates pseudo rotated boxes without additional priors and incorporates support for the end-to-end paradigm. PointOBB-v3 functions by integrating three unique image views: the original view, a resized view, and a rotated/flipped (rot/flp) view. Based on the views, a scale augmentation module and an angle acquisition module are constructed. In the first module, a Scale-Sensitive Consistency (SSC) loss and a Scale-Sensitive Feature Fusion (SSFF) module are introduced to improve the model's ability to estimate object scale. To achieve precise angle predictions, the second module employs symmetry-based self-supervised learning. Additionally, we introduce an end-to-end version that eliminates the pseudo-label generation process by integrating a detector branch and introduces an Instance-Aware Weighting (IAW) strategy to focus on high-quality predictions. We conducted extensive experiments on the DIOR-R, DOTA-v1.0/v1.5/v2.0, FAIR1M, STAR, and RSAR datasets. Across all these datasets, our method achieves an average improvement in accuracy of 3.56% in comparison to previous state-of-the-art methods. The code will be available at https://github.com/ZpyWHU/PointOBB-v3.
>
---
#### [replaced 041] Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17412v2](http://arxiv.org/pdf/2505.17412v2)**

> **作者:** Shuang Wu; Youtian Lin; Feihu Zhang; Yifei Zeng; Yikang Yang; Yajie Bao; Jiachen Qian; Siyu Zhu; Xun Cao; Philip Torr; Yao Yao
>
> **备注:** Project page: https://www.neural4d.com/research/direct3d-s2
>
> **摘要:** Generating high-resolution 3D shapes using volumetric representations such as Signed Distance Functions (SDFs) presents substantial computational and memory challenges. We introduce Direct3D-S2, a scalable 3D generation framework based on sparse volumes that achieves superior output quality with dramatically reduced training costs. Our key innovation is the Spatial Sparse Attention (SSA) mechanism, which greatly enhances the efficiency of Diffusion Transformer (DiT) computations on sparse volumetric data. SSA allows the model to effectively process large token sets within sparse volumes, substantially reducing computational overhead and achieving a 3.9x speedup in the forward pass and a 9.6x speedup in the backward pass. Our framework also includes a variational autoencoder (VAE) that maintains a consistent sparse volumetric format across input, latent, and output stages. Compared to previous methods with heterogeneous representations in 3D VAE, this unified design significantly improves training efficiency and stability. Our model is trained on public available datasets, and experiments demonstrate that Direct3D-S2 not only surpasses state-of-the-art methods in generation quality and efficiency, but also enables training at 1024 resolution using only 8 GPUs, a task typically requiring at least 32 GPUs for volumetric representations at 256 resolution, thus making gigascale 3D generation both practical and accessible. Project page: https://www.neural4d.com/research/direct3d-s2.
>
---
#### [replaced 042] IGL-DT: Iterative Global-Local Feature Learning with Dual-Teacher Semantic Segmentation Framework under Limited Annotation Scheme
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.09797v2](http://arxiv.org/pdf/2504.09797v2)**

> **作者:** Dinh Dai Quan Tran; Hoang-Thien Nguyen; Thanh-Huy Nguyen; Gia-Van To; Tien-Huy Nguyen; Quan Nguyen
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Semi-Supervised Semantic Segmentation (SSSS) aims to improve segmentation accuracy by leveraging a small set of labeled images alongside a larger pool of unlabeled data. Recent advances primarily focus on pseudo-labeling, consistency regularization, and co-training strategies. However, existing methods struggle to balance global semantic representation with fine-grained local feature extraction. To address this challenge, we propose a novel tri-branch semi-supervised segmentation framework incorporating a dual-teacher strategy, named IGL-DT. Our approach employs SwinUnet for high-level semantic guidance through Global Context Learning and ResUnet for detailed feature refinement via Local Regional Learning. Additionally, a Discrepancy Learning mechanism mitigates over-reliance on a single teacher, promoting adaptive feature learning. Extensive experiments on benchmark datasets demonstrate that our method outperforms state-of-the-art approaches, achieving superior segmentation performance across various data regimes.
>
---
#### [replaced 043] HPPP: Halpern-type Preconditioned Proximal Point Algorithms and Applications to Image Restoration
- **分类: cs.CV; math.OC**

- **链接: [http://arxiv.org/pdf/2407.13120v4](http://arxiv.org/pdf/2407.13120v4)**

> **作者:** Shuchang Zhang; Hui Zhang; Hongxia Wang
>
> **摘要:** Recently, the degenerate preconditioned proximal point (PPP) method provides a unified and flexible framework for designing and analyzing operator-splitting algorithms such as Douglas-Rachford (DR). However, the degenerate PPP method exhibits weak convergence in the infinite-dimensional Hilbert space and lacks accelerated variants. To address these issues, we propose a Halpern-type PPP (HPPP) algorithm, which leverages the strong convergence and acceleration properties of Halpern's iteration method. Moreover, we propose a novel algorithm for image restoration by combining HPPP with denoiser priors such as Plug-and-Play (PnP) prior, which can be viewed as an accelerated PnP method. Finally, numerical experiments including several toy examples and image restoration validate the effectiveness of our proposed algorithms.
>
---
#### [replaced 044] V-RoAst: Visual Road Assessment. Can VLM be a Road Safety Assessor Using the iRAP Standard?
- **分类: cs.CV; cs.AI; cs.ET**

- **链接: [http://arxiv.org/pdf/2408.10872v3](http://arxiv.org/pdf/2408.10872v3)**

> **作者:** Natchapon Jongwiriyanurak; Zichao Zeng; June Moh Goo; James Haworth; Xinglei Wang; Kerkritt Sriroongvikrai; Nicola Christie; Ilya Ilyankou; Meihui Wang; Huanfa Chen
>
> **摘要:** Road traffic crashes result in millions of deaths annually and significant economic burdens, particularly on Low- and Middle-Income Countries (LMICs). Road safety assessments traditionally rely on human-labelled data, which is labour-intensive and time-consuming. While Convolutional Neural Networks (CNNs) have advanced automated road safety assessments, they typically demand large labelled datasets and often require fine-tuning for each new geographic context. This study explores whether Vision Language Models (VLMs) with zero-shot capability can overcome these limitations to serve as effective road safety assessors using the International Road Assessment Programme (iRAP) standard. Our approach, V-RoAst (Visual question answering for Road Assessment), leverages advanced VLMs, such as Gemini-1.5-flash and GPT-4o-mini, to analyse road safety attributes without requiring any labelled training data. By optimising prompt engineering and utilising crowdsourced imagery from Mapillary, V-RoAst provides a scalable, cost-effective, and automated solution for global road safety assessments. Preliminary results show that while VLMs achieve lower performance than CNN-based models, they are capable of Visual Question Answering (VQA) and show potential in predicting star ratings from crowdsourced imagery. However, their performance is poor when key visual features are absent in the imagery, emphasising the need for human labelling to address these gaps. Advancements in VLMs, alongside in-context learning such as chain-of-thought and few-shot learning, and parameter-efficient fine-tuning, present opportunities for improvement, making VLMs promising tools for road assessment tasks. Designed for resource-constrained stakeholders, this framework holds the potential to save lives and reduce economic burdens worldwide. Code and dataset are available at: https://github.com/PongNJ/V-RoAst.
>
---
#### [replaced 045] Efficient Lung Ultrasound Severity Scoring Using Dedicated Feature Extractor
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12524v3](http://arxiv.org/pdf/2501.12524v3)**

> **作者:** Jiaqi Guo; Yunan Wu; Evangelos Kaimakamis; Georgios Petmezas; Vasileios E. Papageorgiou; Nicos Maglaveras; Aggelos K. Katsaggelos
>
> **备注:** Accepted by IEEE ISBI 2025 (Selected for oral presentation); 2025/4/15 (v2): Corrected a notation error in Figure 2
>
> **摘要:** With the advent of the COVID-19 pandemic, ultrasound imaging has emerged as a promising technique for COVID-19 detection, due to its non-invasive nature, affordability, and portability. In response, researchers have focused on developing AI-based scoring systems to provide real-time diagnostic support. However, the limited size and lack of proper annotation in publicly available ultrasound datasets pose significant challenges for training a robust AI model. This paper proposes MeDiVLAD, a novel pipeline to address the above issue for multi-level lung-ultrasound (LUS) severity scoring. In particular, we leverage self-knowledge distillation to pretrain a vision transformer (ViT) without label and aggregate frame-level features via dual-level VLAD aggregation. We show that with minimal finetuning, MeDiVLAD outperforms conventional fully-supervised methods in both frame- and video-level scoring, while offering classification reasoning with exceptional quality. This superior performance enables key applications such as the automatic identification of critical lung pathology areas and provides a robust solution for broader medical video classification tasks.
>
---
#### [replaced 046] SCHEME: Scalable Channel Mixer for Vision Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.00412v4](http://arxiv.org/pdf/2312.00412v4)**

> **作者:** Deepak Sridhar; Yunsheng Li; Nuno Vasconcelos
>
> **摘要:** Vision Transformers have achieved impressive performance in many vision tasks. While the token mixer or attention block has been studied in great detail, much less research has been devoted to the channel mixer or feature mixing block (FFN or MLP), which accounts for a significant portion of the model parameters and computation. In this work, we show that the dense MLP connections can be replaced with a sparse block diagonal MLP structure that supports larger expansion ratios by splitting MLP features into groups. To improve the feature clusters formed by this structure we propose the use of a lightweight, parameter-free, channel covariance attention (CCA) mechanism as a parallel branch during training. This enables gradual feature mixing across channel groups during training whose contribution decays to zero as the training progresses to convergence. As a result, the CCA block can be discarded during inference, enabling enhanced performance at no additional computational cost. The resulting $\textit{Scalable CHannEl MixEr}$ (SCHEME) can be plugged into any ViT architecture to obtain a gamut of models with different trade-offs between complexity and performance by controlling the block diagonal MLP structure. This is shown by the introduction of a new family of SCHEMEformer models. Experiments on image classification, object detection, and semantic segmentation, with $\textbf{12 different ViT backbones}$, consistently demonstrate substantial accuracy/latency gains (upto $\textbf{1.5\% /20\%})$ over existing designs, especially for lower complexity regimes. The SCHEMEformer family is shown to establish new Pareto frontiers for accuracy vs FLOPS, accuracy vs model size, and accuracy vs throughput, especially for fast transformers of small size.
>
---
#### [replaced 047] When Are Concepts Erased From Diffusion Models?
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17013v2](http://arxiv.org/pdf/2505.17013v2)**

> **作者:** Kevin Lu; Nicky Kriplani; Rohit Gandikota; Minh Pham; David Bau; Chinmay Hegde; Niv Cohen
>
> **备注:** Project Page: https://nyu-dice-lab.github.io/when-are-concepts-erased/
>
> **摘要:** Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.
>
---
#### [replaced 048] Auto-nnU-Net: Towards Automated Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16561v2](http://arxiv.org/pdf/2505.16561v2)**

> **作者:** Jannis Becktepe; Leona Hennig; Steffen Oeltze-Jafra; Marius Lindauer
>
> **备注:** 31 pages, 19 figures. Accepted for publication at AutoML 2025
>
> **摘要:** Medical Image Segmentation (MIS) includes diverse tasks, from bone to organ segmentation, each with its own challenges in finding the best segmentation model. The state-of-the-art AutoML-related MIS-framework nnU-Net automates many aspects of model configuration but remains constrained by fixed hyperparameters and heuristic design choices. As a full-AutoML framework for MIS, we propose Auto-nnU-Net, a novel nnU-Net variant enabling hyperparameter optimization (HPO), neural architecture search (NAS), and hierarchical NAS (HNAS). Additionally, we propose Regularized PriorBand to balance model accuracy with the computational resources required for training, addressing the resource constraints often faced in real-world medical settings that limit the feasibility of extensive training procedures. We evaluate our approach across diverse MIS datasets from the well-established Medical Segmentation Decathlon, analyzing the impact of AutoML techniques on segmentation performance, computational efficiency, and model design choices. The results demonstrate that our AutoML approach substantially improves the segmentation performance of nnU-Net on 6 out of 10 datasets and is on par on the other datasets while maintaining practical resource requirements. Our code is available at https://github.com/LUH-AI/AutonnUNet.
>
---
#### [replaced 049] Variational Control for Guidance in Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.03686v2](http://arxiv.org/pdf/2502.03686v2)**

> **作者:** Kushagra Pandey; Farrin Marouf Sofian; Felix Draxler; Theofanis Karaletsos; Stephan Mandt
>
> **备注:** Camera-Ready. Accepted at ICML 2025. 9 pages in main text. Total of 26 pages
>
> **摘要:** Diffusion models exhibit excellent sample quality, but existing guidance methods often require additional model training or are limited to specific tasks. We revisit guidance in diffusion models from the perspective of variational inference and control, introducing Diffusion Trajectory Matching (DTM) that enables guiding pretrained diffusion trajectories to satisfy a terminal cost. DTM unifies a broad class of guidance methods and enables novel instantiations. We introduce a new method within this framework that achieves state-of-the-art results on several linear, non-linear, and blind inverse problems without requiring additional model training or specificity to pixel or latent space diffusion models. Our code will be available at https://github.com/czi-ai/oc-guidance
>
---
#### [replaced 050] Exploring 3D Activity Reasoning and Planning: From Implicit Human Intentions to Route-Aware Planning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.12974v2](http://arxiv.org/pdf/2503.12974v2)**

> **作者:** Xueying Jiang; Wenhao Li; Xiaoqin Zhang; Ling Shao; Shijian Lu
>
> **摘要:** 3D activity reasoning and planning has attracted increasing attention in human-robot interaction and embodied AI thanks to the recent advance in multimodal learning. However, most existing studies are facing two common challenges: 1) heavy reliance on explicit instructions with little reasoning on implicit user intention; 2) negligence of inter-step route planning on robot moves. We address the above challenges by proposing 3D activity reasoning and planning, a novel 3D task that reasons the intended activities from implicit instructions and decomposes them into steps with inter-step routes and planning under the guidance of fine-grained 3D object shapes and locations from scene segmentation. We tackle the new 3D task from two perspectives. First, we construct ReasonPlan3D, a large-scale benchmark that covers diverse 3D scenes with rich implicit instructions and detailed annotations for multi-step task planning, inter-step route planning, and fine-grained segmentation. Second, we design a novel framework that introduces progressive plan generation with contextual consistency across multiple steps, as well as a scene graph that is updated dynamically for capturing critical objects and their spatial relations. Extensive experiments demonstrate the effectiveness of our benchmark and framework in reasoning activities from implicit human instructions, producing accurate stepwise task plans and seamlessly integrating route planning for multi-step moves. The dataset and code will be released.
>
---
#### [replaced 051] Parrot: Multilingual Visual Instruction Tuning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.02539v3](http://arxiv.org/pdf/2406.02539v3)**

> **作者:** Hai-Long Sun; Da-Wei Zhou; Yang Li; Shiyin Lu; Chao Yi; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang; De-Chuan Zhan; Han-Jia Ye
>
> **备注:** Accepted to ICML 2025. Code and dataset are available at: https://github.com/AIDC-AI/Parrot
>
> **摘要:** The rapid development of Multimodal Large Language Models (MLLMs), such as GPT-4o, marks a significant step toward artificial general intelligence. Existing methods typically align vision encoders with LLMs via supervised fine-tuning (SFT), but this often deteriorates their ability to handle multiple languages as training progresses. We empirically observe that imbalanced SFT datasets, largely English-centric, degrade performance on non-English languages due to the failure in multilingual token alignment. To address this, we propose PARROT, a novel approach that leverages textual guidance for visual token alignment at the language level. PARROT conditions visual tokens on diverse language inputs and uses Mixture-of-Experts (MoE) to align multilingual tokens. By computing cross-attention between initial visual features and textual embeddings, we select the most relevant experts, converting visual tokens into language-specific representations. Additionally, we introduce the Massive Multilingual Multimodal Benchmark (MMMB), a new benchmark comprising 6 languages, 15 categories, and 12,000 questions, to assess multilingual capabilities. PARROT achieves state-of-the-art performance on both the multilingual benchmarks and a wide range of multimodal tasks. Code and dataset are available at: https://github.com/AIDC-AI/Parrot
>
---
#### [replaced 052] OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16091v2](http://arxiv.org/pdf/2505.16091v2)**

> **作者:** Jinpei Guo; Yifei Ji; Zheng Chen; Kai Liu; Min Liu; Wang Rao; Wenbo Li; Yong Guo; Yulun Zhang
>
> **摘要:** Pretrained latent diffusion models have shown strong potential for lossy image compression, owing to their powerful generative priors. Most existing diffusion-based methods reconstruct images by iteratively denoising from random noise, guided by compressed latent representations. While these approaches have achieved high reconstruction quality, their multi-step sampling process incurs substantial computational overhead. Moreover, they typically require training separate models for different compression bit-rates, leading to significant training and storage costs. To address these challenges, we propose a one-step diffusion codec across multiple bit-rates. termed OSCAR. Specifically, our method views compressed latents as noisy variants of the original latents, where the level of distortion depends on the bit-rate. This perspective allows them to be modeled as intermediate states along a diffusion trajectory. By establishing a mapping from the compression bit-rate to a pseudo diffusion timestep, we condition a single generative model to support reconstructions at multiple bit-rates. Meanwhile, we argue that the compressed latents retain rich structural information, thereby making one-step denoising feasible. Thus, OSCAR replaces iterative sampling with a single denoising pass, significantly improving inference efficiency. Extensive experiments demonstrate that OSCAR achieves superior performance in both quantitative and visual quality metrics. The code and models will be released at https://github.com/jp-guo/OSCAR.
>
---
#### [replaced 053] Do Vision-Language Models Really Understand Visual Language?
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.00193v3](http://arxiv.org/pdf/2410.00193v3)**

> **作者:** Yifan Hou; Buse Giledereli; Yilei Tu; Mrinmaya Sachan
>
> **备注:** ICML 2025
>
> **摘要:** Visual language is a system of communication that conveys information through symbols, shapes, and spatial arrangements. Diagrams are a typical example of a visual language depicting complex concepts and their relationships in the form of an image. The symbolic nature of diagrams presents significant challenges for building models capable of understanding them. Recent studies suggest that Large Vision-Language Models (LVLMs) can even tackle complex reasoning tasks involving diagrams. In this paper, we investigate this phenomenon by developing a comprehensive test suite to evaluate the diagram comprehension capability of LVLMs. Our test suite uses a variety of questions focused on concept entities and their relationships over a set of synthetic as well as real diagrams across domains to evaluate the recognition and reasoning abilities of models. Our evaluation of LVLMs shows that while they can accurately identify and reason about entities, their ability to understand relationships is notably limited. Further testing reveals that the decent performance on diagram understanding largely stems from leveraging their background knowledge as shortcuts to identify and reason about the relational information. Thus, we conclude that LVLMs have a limited capability for genuine diagram understanding, and their impressive performance in diagram reasoning is an illusion emanating from other confounding factors, such as the background knowledge in the models.
>
---
#### [replaced 054] LiDAR-EDIT: LiDAR Data Generation by Editing the Object Layouts in Real-World Scenes
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.00592v3](http://arxiv.org/pdf/2412.00592v3)**

> **作者:** Shing-Hei Ho; Bao Thach; Minghan Zhu
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA). 6 pages, 7 figures
>
> **摘要:** We present LiDAR-EDIT, a novel paradigm for generating synthetic LiDAR data for autonomous driving. Our framework edits real-world LiDAR scans by introducing new object layouts while preserving the realism of the background environment. Compared to end-to-end frameworks that generate LiDAR point clouds from scratch, LiDAR-EDIT offers users full control over the object layout, including the number, type, and pose of objects, while keeping most of the original real-world background. Our method also provides object labels for the generated data. Compared to novel view synthesis techniques, our framework allows for the creation of counterfactual scenarios with object layouts significantly different from the original real-world scene. LiDAR-EDIT uses spherical voxelization to enforce correct LiDAR projective geometry in the generated point clouds by construction. During object removal and insertion, generative models are employed to fill the unseen background and object parts that were occluded in the original real LiDAR scans. Experimental results demonstrate that our framework produces realistic LiDAR scans with practical value for downstream tasks.
>
---
#### [replaced 055] LLaVA-ReID: Selective Multi-image Questioner for Interactive Person Re-Identification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10174v3](http://arxiv.org/pdf/2504.10174v3)**

> **作者:** Yiding Lu; Mouxing Yang; Dezhong Peng; Peng Hu; Yijie Lin; Xi Peng
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Traditional text-based person ReID assumes that person descriptions from witnesses are complete and provided at once. However, in real-world scenarios, such descriptions are often partial or vague. To address this limitation, we introduce a new task called interactive person re-identification (Inter-ReID). Inter-ReID is a dialogue-based retrieval task that iteratively refines initial descriptions through ongoing interactions with the witnesses. To facilitate the study of this new task, we construct a dialogue dataset that incorporates multiple types of questions by decomposing fine-grained attributes of individuals. We further propose LLaVA-ReID, a question model that generates targeted questions based on visual and textual contexts to elicit additional details about the target person. Leveraging a looking-forward strategy, we prioritize the most informative questions as supervision during training. Experimental results on both Inter-ReID and text-based ReID benchmarks demonstrate that LLaVA-ReID significantly outperforms baselines.
>
---
#### [replaced 056] FreSca: Scaling in Frequency Space Enhances Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02154v2](http://arxiv.org/pdf/2504.02154v2)**

> **作者:** Chao Huang; Susan Liang; Yunlong Tang; Jing Bi; Li Ma; Yapeng Tian; Chenliang Xu
>
> **备注:** Project page: https://wikichao.github.io/FreSca/
>
> **摘要:** Latent diffusion models (LDMs) have achieved remarkable success in a variety of image tasks, yet achieving fine-grained, disentangled control over global structures versus fine details remains challenging. This paper explores frequency-based control within latent diffusion models. We first systematically analyze frequency characteristics across pixel space, VAE latent space, and internal LDM representations. This reveals that the "noise difference" term, derived from classifier-free guidance at each step t, is a uniquely effective and semantically rich target for manipulation. Building on this insight, we introduce FreSca, a novel and plug-and-play framework that decomposes noise difference into low- and high-frequency components and applies independent scaling factors to them via spatial or energy-based cutoffs. Essentially, FreSca operates without any model retraining or architectural change, offering model- and task-agnostic control. We demonstrate its versatility and effectiveness in improving generation quality and structural emphasis on multiple architectures (e.g., SD3, SDXL) and across applications including image generation, editing, depth estimation, and video synthesis, thereby unlocking a new dimension of expressive control within LDMs.
>
---
#### [replaced 057] Towards Better Robustness: Pose-Free 3D Gaussian Splatting for Arbitrarily Long Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.15096v2](http://arxiv.org/pdf/2501.15096v2)**

> **作者:** Zhen-Hui Dong; Sheng Ye; Yu-Hui Wen; Nannan Li; Yong-Jin Liu
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a powerful representation due to its efficiency and high-fidelity rendering. 3DGS training requires a known camera pose for each input view, typically obtained by Structure-from-Motion (SfM) pipelines. Pioneering works have attempted to relax this restriction but still face difficulties when handling long sequences with complex camera trajectories. In this paper, we propose Rob-GS, a robust framework to progressively estimate camera poses and optimize 3DGS for arbitrarily long video inputs. In particular, by leveraging the inherent continuity of videos, we design an adjacent pose tracking method to ensure stable pose estimation between consecutive frames. To handle arbitrarily long inputs, we propose a Gaussian visibility retention check strategy to adaptively split the video sequence into several segments and optimize them separately. Extensive experiments on Tanks and Temples, ScanNet, and a self-captured dataset show that Rob-GS outperforms the state-of-the-arts.
>
---
#### [replaced 058] Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16990v2](http://arxiv.org/pdf/2505.16990v2)**

> **作者:** Runpeng Yu; Xinyin Ma; Xinchao Wang
>
> **摘要:** In this work, we propose Dimple, the first Discrete Diffusion Multimodal Large Language Model (DMLLM). We observe that training with a purely discrete diffusion approach leads to significant training instability, suboptimal performance, and severe length bias issues. To address these challenges, we design a novel training paradigm that combines an initial autoregressive phase with a subsequent diffusion phase. This approach yields the Dimple-7B model, trained on the same dataset and using a similar training pipeline as LLaVA-NEXT. Dimple-7B ultimately surpasses LLaVA-NEXT in performance by 3.9%, demonstrating that DMLLM can achieve performance comparable to that of autoregressive models. To improve inference efficiency, we propose a decoding strategy termed confident decoding, which dynamically adjusts the number of tokens generated at each step, significantly reducing the number of generation iterations. In autoregressive models, the number of forward iterations during generation equals the response length. With confident decoding, however, the number of iterations needed by Dimple is even only $\frac{\text{response length}}{3}$. We also re-implement the prefilling technique in autoregressive models and demonstrate that it does not significantly impact performance on most benchmark evaluations, while offering a speedup of 1.5x to 7x. Additionally, we explore Dimple's capability to precisely control its response using structure priors. These priors enable structured responses in a manner distinct from instruction-based or chain-of-thought prompting, and allow fine-grained control over response format and length, which is difficult to achieve in autoregressive models. Overall, this work validates the feasibility and advantages of DMLLM and enhances its inference efficiency and controllability. Code and models are available at https://github.com/yu-rp/Dimple.
>
---
#### [replaced 059] Pixels2Points: Fusing 2D and 3D Features for Facial Skin Segmentation
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19718v3](http://arxiv.org/pdf/2504.19718v3)**

> **作者:** Victoria Yue Chen; Daoye Wang; Stephan Garbin; Jan Bednarik; Sebastian Winberg; Timo Bolkart; Thabo Beeler
>
> **备注:** 4 pages, 4 figures, published in Eurographics 2025 as a short paper
>
> **摘要:** Face registration deforms a template mesh to closely fit a 3D face scan, the quality of which commonly degrades in non-skin regions (e.g., hair, beard, accessories), because the optimized template-to-scan distance pulls the template mesh towards the noisy scan surface. Improving registration quality requires a clean separation of skin and non-skin regions on the scan mesh. Existing image-based (2D) or scan-based (3D) segmentation methods however perform poorly. Image-based segmentation outputs multi-view inconsistent masks, and they cannot account for scan inaccuracies or scan-image misalignment, while scan-based methods suffer from lower spatial resolution compared to images. In this work, we introduce a novel method that accurately separates skin from non-skin geometry on 3D human head scans. For this, our method extracts features from multi-view images using a frozen image foundation model and aggregates these features in 3D. These lifted 2D features are then fused with 3D geometric features extracted from the scan mesh, to then predict a segmentation mask directly on the scan mesh. We show that our segmentations improve the registration accuracy over pure 2D or 3D segmentation methods by 8.89% and 14.3%, respectively. Although trained only on synthetic data, our model generalizes well to real data.
>
---
#### [replaced 060] Without Paired Labeled Data: End-to-End Self-Supervised Learning for Drone-view Geo-Localization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11381v3](http://arxiv.org/pdf/2502.11381v3)**

> **作者:** Zhongwei Chen; Zhao-Xu Yang; Hai-Jun Rong
>
> **摘要:** Drone-view Geo-Localization (DVGL) aims to achieve accurate localization of drones by retrieving the most relevant GPS-tagged satellite images. However, most existing methods heavily rely on strictly pre-paired drone-satellite images for supervised learning. When the target region shifts, new paired samples are typically required to adapt to the distribution changes. The high cost of annotation and the limited transferability of these methods significantly hinder the practical deployment of DVGL in open-world scenarios. To address these limitations, we propose an end-to-end self-supervised learning method with a shallow backbone network. It employs a clustering algorithm to generate pseudo-labels and adopts a dual-path contrastive learning framework to learn discriminative intra-view representations. Furthermore, our method incorporates two core modules, including the dynamic hierarchical memory learning module (DHML) and the information consistency evolution learning module (ICEL). The DHML combines short-term and long-term memory to enhance intra-view feature consistency and discriminability. Meanwhile, the ICEL module utilizes a neighborhood-driven dynamic constraint mechanism to systematically capture implicit cross-view semantic correlations, consequently improving cross-view feature alignment. To further stabilize and strengthen the self-supervised training process, a pseudo-label enhancement strategy is introduced to enhance the quality of pseudo supervision. Extensive experiments on three public benchmark datasets demonstrate that the proposed method consistently outperforms existing self-supervised methods and even surpasses several state-of-the-art supervised methods. {Our code is available at https://github.com/ISChenawei/DMNIL.
>
---
#### [replaced 061] SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.17161v2](http://arxiv.org/pdf/2501.17161v2)**

> **作者:** Tianzhe Chu; Yuexiang Zhai; Jihan Yang; Shengbang Tong; Saining Xie; Dale Schuurmans; Quoc V. Le; Sergey Levine; Yi Ma
>
> **备注:** Website at https://tianzhechu.com/SFTvsRL
>
> **摘要:** Supervised fine-tuning (SFT) and reinforcement learning (RL) are widely used post-training techniques for foundation models. However, their roles in enhancing model generalization capabilities remain unclear. This paper studies the difference between SFT and RL on generalization and memorization, focusing on text-based rule variants and visual variants. We introduce GeneralPoints, an arithmetic reasoning card game, and adopt V-IRL, a real-world navigation environment, to assess how models trained with SFT and RL generalize to unseen variants in both textual and visual domains. We show that RL, especially when trained with an outcome-based reward, generalizes across both rule-based textual and visual variants. SFT, in contrast, tends to memorize training data and struggles to generalize out-of-distribution scenarios. Further analysis reveals that RL improves the model's underlying visual recognition capabilities, contributing to its enhanced generalization in the visual domain. Despite RL's superior generalization, we show that SFT remains essential for effective RL training; SFT stabilizes the model's output format, enabling subsequent RL to achieve its performance gains. These findings demonstrates the capability of RL for acquiring generalizable knowledge in complex, multi-modal tasks.
>
---
#### [replaced 062] HybridTrack: A Hybrid Approach for Robust Multi-Object Tracking
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.01275v2](http://arxiv.org/pdf/2501.01275v2)**

> **作者:** Leandro Di Bella; Yangxintong Lyu; Bruno Cornelis; Adrian Munteanu
>
> **备注:** IEEE ROBOTICS AND AUTOMATION LETTERS. ACCEPTED MAY, 2025
>
> **摘要:** The evolution of Advanced Driver Assistance Systems (ADAS) has increased the need for robust and generalizable algorithms for multi-object tracking. Traditional statistical model-based tracking methods rely on predefined motion models and assumptions about system noise distributions. Although computationally efficient, they often lack adaptability to varying traffic scenarios and require extensive manual design and parameter tuning. To address these issues, we propose a novel 3D multi-object tracking approach for vehicles, HybridTrack, which integrates a data-driven Kalman Filter (KF) within a tracking-by-detection paradigm. In particular, it learns the transition residual and Kalman gain directly from data, which eliminates the need for manual motion and stochastic parameter modeling. Validated on the real-world KITTI dataset, HybridTrack achieves 82.72% HOTA accuracy, significantly outperforming state-of-the-art methods. We also evaluate our method under different configurations, achieving the fastest processing speed of 112 FPS. Consequently, HybridTrack eliminates the dependency on scene-specific designs while improving performance and maintaining real-time efficiency. The code is publicly available at: https://github.com/leandro-svg/HybridTrack.
>
---
#### [replaced 063] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09271v2](http://arxiv.org/pdf/2503.09271v2)**

> **作者:** Chiara Cappellino; Gianluca Mancusi; Matteo Mosconi; Angelo Porrello; Simone Calderara; Rita Cucchiara
>
> **摘要:** Open-Vocabulary object detectors can generalize to an unrestricted set of categories through simple textual prompting. However, adapting these models to rare classes or reinforcing their abilities on multiple specialized domains remains essential. While recent methods rely on monolithic adaptation strategies with a single set of weights, we embrace modular deep learning. We introduce DitHub, a framework designed to build and maintain a library of efficient adaptation modules. Inspired by Version Control Systems, DitHub manages expert modules as branches that can be fetched and merged as needed. This modular approach allows us to conduct an in-depth exploration of the compositional properties of adaptation modules, marking the first such study in Object Detection. Our method achieves state-of-the-art performance on the ODinW-13 benchmark and ODinW-O, a newly introduced benchmark designed to assess class reappearance. For more details, visit our project page: https://aimagelab.github.io/DitHub/
>
---
#### [replaced 064] Expanding Zero-Shot Object Counting with Rich Prompts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15398v2](http://arxiv.org/pdf/2505.15398v2)**

> **作者:** Huilin Zhu; Senyao Li; Jingling Yuan; Zhengwei Yang; Yu Guo; Wenxuan Liu; Xian Zhong; Shengfeng He
>
> **摘要:** Expanding pre-trained zero-shot counting models to handle unseen categories requires more than simply adding new prompts, as this approach does not achieve the necessary alignment between text and visual features for accurate counting. We introduce RichCount, the first framework to address these limitations, employing a two-stage training strategy that enhances text encoding and strengthens the model's association with objects in images. RichCount improves zero-shot counting for unseen categories through two key objectives: (1) enriching text features with a feed-forward network and adapter trained on text-image similarity, thereby creating robust, aligned representations; and (2) applying this refined encoder to counting tasks, enabling effective generalization across diverse prompts and complex images. In this manner, RichCount goes beyond simple prompt expansion to establish meaningful feature alignment that supports accurate counting across novel categories. Extensive experiments on three benchmark datasets demonstrate the effectiveness of RichCount, achieving state-of-the-art performance in zero-shot counting and significantly enhancing generalization to unseen categories in open-world scenarios.
>
---
#### [replaced 065] Corrupted but Not Broken: Understanding and Mitigating the Negative Impacts of Corrupted Data in Visual Instruction Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12635v2](http://arxiv.org/pdf/2502.12635v2)**

> **作者:** Yunhao Gou; Hansi Yang; Zhili Liu; Kai Chen; Yihan Zeng; Lanqing Hong; Zhenguo Li; Qun Liu; Bo Han; James T. Kwok; Yu Zhang
>
> **摘要:** Visual Instruction Tuning (VIT) aims to enhance Multimodal Large Language Models (MLLMs), yet its effectiveness is often compromised by corrupted datasets with issues such as hallucinated content, incorrect responses, and poor OCR quality. Previous approaches to address these challenges have focused on refining datasets through high-quality data collection or rule-based filtering that can be costly or limited in scope. In this paper, we conduct a systematic investigation into the impact of corrupted data on MLLMs and discover that, although corrupted data degrade model performance, such adverse effects are largely reversible, and MLLMs are {\bf corrupted but not broken}. Specifically, we find that disabling a small subset of parameters can almost fully restore performance. Moreover, corrupted MLLMs inherently possess the capability to differentiate between clean and corrupted samples, facilitating dataset cleaning without external intervention. Building on these insights, we introduce a corruption-robust training paradigm that significantly surpasses existing strategies for mitigating the effects of corrupted data.
>
---
#### [replaced 066] Understanding Generative AI Capabilities in Everyday Image Editing Tasks
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16181v2](http://arxiv.org/pdf/2505.16181v2)**

> **作者:** Mohammad Reza Taesiri; Brandon Collins; Logan Bolton; Viet Dac Lai; Franck Dernoncourt; Trung Bui; Anh Totti Nguyen
>
> **备注:** Code and qualitative examples are available at: https://psrdataset.github.io
>
> **摘要:** Generative AI (GenAI) holds significant promise for automating everyday image editing tasks, especially following the recent release of GPT-4o on March 25, 2025. However, what subjects do people most often want edited? What kinds of editing actions do they want to perform (e.g., removing or stylizing the subject)? Do people prefer precise edits with predictable outcomes or highly creative ones? By understanding the characteristics of real-world requests and the corresponding edits made by freelance photo-editing wizards, can we draw lessons for improving AI-based editors and determine which types of requests can currently be handled successfully by AI editors? In this paper, we present a unique study addressing these questions by analyzing 83k requests from the past 12 years (2013-2025) on the Reddit community, which collected 305k PSR-wizard edits. According to human ratings, approximately only 33% of requests can be fulfilled by the best AI editors (including GPT-4o, Gemini-2.0-Flash, SeedEdit). Interestingly, AI editors perform worse on low-creativity requests that require precise editing than on more open-ended tasks. They often struggle to preserve the identity of people and animals, and frequently make non-requested touch-ups. On the other side of the table, VLM judges (e.g., o1) perform differently from human judges and may prefer AI edits more than human edits. Code and qualitative examples are available at: https://psrdataset.github.io
>
---
#### [replaced 067] Little Data, Big Impact: Privacy-Aware Visual Language Models via Minimal Tuning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.17423v3](http://arxiv.org/pdf/2405.17423v3)**

> **作者:** Laurens Samson; Nimrod Barazani; Sennay Ghebreab; Yuki M. Asano
>
> **备注:** preprint
>
> **摘要:** As Visual Language Models (VLMs) become increasingly embedded in everyday applications, ensuring they can recognize and appropriately handle privacy-sensitive content is essential. We conduct a comprehensive evaluation of ten state-of-the-art VLMs and identify limitations in their understanding of visual privacy. Existing datasets suffer from label inconsistencies, limiting their reliability. To address this, we introduce two compact, high-quality benchmarks, PrivBench and PrivBench-H, that focus on commonly recognized privacy categories aligned with the General Data Protection Regulation (GDPR). Additionally, we present PrivTune, an instruction-tuning dataset specifically curated to improve privacy sensitivity. We obtain a Privacy VLM by fine-tuning an off-the-shelf VLM on only 100 samples from PrivTune, which leads to substantial gains on all benchmarks, surpassing GPT-4, while maintaining strong performance on other tasks. Our findings show that privacy-awareness in VLMs can be substantially improved with minimal data and careful dataset design, setting the stage for safer, more privacy-aligned AI systems.
>
---
#### [replaced 068] RDEIC: Accelerating Diffusion-Based Extreme Image Compression with Relay Residual Diffusion
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.02640v3](http://arxiv.org/pdf/2410.02640v3)**

> **作者:** Zhiyuan Li; Yanhui Zhou; Hao Wei; Chenyang Ge; Ajmal Mian
>
> **摘要:** Diffusion-based extreme image compression methods have achieved impressive performance at extremely low bitrates. However, constrained by the iterative denoising process that starts from pure noise, these methods are limited in both fidelity and efficiency. To address these two issues, we present Relay Residual Diffusion Extreme Image Compression (RDEIC), which leverages compressed feature initialization and residual diffusion. Specifically, we first use the compressed latent features of the image with added noise, instead of pure noise, as the starting point to eliminate the unnecessary initial stages of the denoising process. Second, we directly derive a novel residual diffusion equation from Stable Diffusion's original diffusion equation that reconstructs the raw image by iteratively removing the added noise and the residual between the compressed and target latent features. In this way, we effectively combine the efficiency of residual diffusion with the powerful generative capability of Stable Diffusion. Third, we propose a fixed-step fine-tuning strategy to eliminate the discrepancy between the training and inference phases, thereby further improving the reconstruction quality. Extensive experiments demonstrate that the proposed RDEIC achieves state-of-the-art visual quality and outperforms existing diffusion-based extreme image compression methods in both fidelity and efficiency. The source code and pre-trained models are available at https://github.com/huai-chang/RDEIC.
>
---
#### [replaced 069] Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.18984v2](http://arxiv.org/pdf/2501.18984v2)**

> **作者:** Zhengrui Guo; Qichen Sun; Jiabo Ma; Lishuang Feng; Jinzhuo Wang; Hao Chen
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Whole slide image (WSI) analysis presents significant computational challenges due to the massive number of patches in gigapixel images. While transformer architectures excel at modeling long-range correlations through self-attention, their quadratic computational complexity makes them impractical for computational pathology applications. Existing solutions like local-global or linear self-attention reduce computational costs but compromise the strong modeling capabilities of full self-attention. In this work, we propose Querent, i.e., the query-aware long contextual dynamic modeling framework, which achieves a theoretically bounded approximation of full self-attention while delivering practical efficiency. Our method adaptively predicts which surrounding regions are most relevant for each patch, enabling focused yet unrestricted attention computation only with potentially important contexts. By using efficient region-wise metadata computation and importance estimation, our approach dramatically reduces computational overhead while preserving global perception to model fine-grained patch correlations. Through comprehensive experiments on biomarker prediction, gene mutation prediction, cancer subtyping, and survival analysis across over 10 WSI datasets, our method demonstrates superior performance compared to the state-of-the-art approaches. Codes are available at https://github.com/dddavid4real/Querent.
>
---
#### [replaced 070] A Survey of LLM-based Agents in Medicine: How far are we from Baymax?
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11211v2](http://arxiv.org/pdf/2502.11211v2)**

> **作者:** Wenxuan Wang; Zizhan Ma; Zheng Wang; Chenghan Wu; Jiaming Ji; Wenting Chen; Xiang Li; Yixuan Yuan
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) are transforming healthcare through the development of LLM-based agents that can understand, reason about, and assist with medical tasks. This survey provides a comprehensive review of LLM-based agents in medicine, examining their architectures, applications, and challenges. We analyze the key components of medical agent systems, including system profiles, clinical planning mechanisms, medical reasoning frameworks, and external capacity enhancement. The survey covers major application scenarios such as clinical decision support, medical documentation, training simulations, and healthcare service optimization. We discuss evaluation frameworks and metrics used to assess these agents' performance in healthcare settings. While LLM-based agents show promise in enhancing healthcare delivery, several challenges remain, including hallucination management, multimodal integration, implementation barriers, and ethical considerations. The survey concludes by highlighting future research directions, including advances in medical reasoning inspired by recent developments in LLM architectures, integration with physical systems, and improvements in training simulations. This work provides researchers and practitioners with a structured overview of the current state and future prospects of LLM-based agents in medicine.
>
---
#### [replaced 071] Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.09032v3](http://arxiv.org/pdf/2410.09032v3)**

> **作者:** Pratinav Seth; Michelle Lin; Brefo Dwamena Yaw; Jade Boutot; Mary Kang; David Rolnick
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Millions of abandoned oil and gas wells are scattered across the world, leaching methane into the atmosphere and toxic compounds into the groundwater. Many of these locations are unknown, preventing the wells from being plugged and their polluting effects averted. Remote sensing is a relatively unexplored tool for pinpointing abandoned wells at scale. We introduce the first large-scale benchmark dataset for this problem, leveraging medium-resolution multi-spectral satellite imagery from Planet Labs. Our curated dataset comprises over 213,000 wells (abandoned, suspended, and active) from Alberta, a region with especially high well density, sourced from the Alberta Energy Regulator and verified by domain experts. We evaluate baseline algorithms for well detection and segmentation, showing the promise of computer vision approaches but also significant room for improvement.
>
---
#### [replaced 072] Boosting Convolution with Efficient MLP-Permutation for Volumetric Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2303.13111v4](http://arxiv.org/pdf/2303.13111v4)**

> **作者:** Yi Lin; Xiao Fang; Dong Zhang; Kwang-Ting Cheng; Hao Chen
>
> **摘要:** Recently, the advent of vision Transformer (ViT) has brought substantial advancements in 3D dataset benchmarks, particularly in 3D volumetric medical image segmentation (Vol-MedSeg). Concurrently, multi-layer perceptron (MLP) network has regained popularity among researchers due to their comparable results to ViT, albeit with the exclusion of the resource-intensive self-attention module. In this work, we propose a novel permutable hybrid network for Vol-MedSeg, named PHNet, which capitalizes on the strengths of both convolution neural networks (CNNs) and MLP. PHNet addresses the intrinsic isotropy problem of 3D volumetric data by employing a combination of 2D and 3D CNNs to extract local features. Besides, we propose an efficient multi-layer permute perceptron (MLPP) module that captures long-range dependence while preserving positional information. This is achieved through an axis decomposition operation that permutes the input tensor along different axes, thereby enabling the separate encoding of the positional information. Furthermore, MLPP tackles the resolution sensitivity issue of MLP in Vol-MedSeg with a token segmentation operation, which divides the feature into smaller tokens and processes them individually. Extensive experimental results validate that PHNet outperforms the state-of-the-art methods with lower computational costs on the widely-used yet challenging COVID-19-20 and Synapse benchmarks. The ablation study also demonstrates the effectiveness of PHNet in harnessing the strengths of both CNNs and MLP. The code is available on Github: \href{https://github.com/xiaofang007/PHNet}{https://github.com/xiaofang007/PHNet}.
>
---
#### [replaced 073] A Review of Pseudo-Labeling for Computer Vision
- **分类: cs.CV; cs.LG; I.2.0; I.5.4; I.4.0**

- **链接: [http://arxiv.org/pdf/2408.07221v3](http://arxiv.org/pdf/2408.07221v3)**

> **作者:** Patrick Kage; Jay C. Rothenberger; Pavlos Andreadis; Dimitrios I. Diochnos
>
> **备注:** 40 pages, 4 figures, 2 tables
>
> **摘要:** Deep neural models have achieved state of the art performance on a wide range of problems in computer science, especially in computer vision. However, deep neural networks often require large datasets of labeled samples to generalize effectively, and an important area of active research is semi-supervised learning, which attempts to instead utilize large quantities of (easily acquired) unlabeled samples. One family of methods in this space is pseudo-labeling, a class of algorithms that use model outputs to assign labels to unlabeled samples which are then used as labeled samples during training. Such assigned labels, called pseudo-labels, are most commonly associated with the field of semi-supervised learning. In this work we explore a broader interpretation of pseudo-labels within both self-supervised and unsupervised methods. By drawing the connection between these areas we identify new directions when advancements in one area would likely benefit others, such as curriculum learning and self-supervised regularization.
>
---
#### [replaced 074] Tracking the Flight: Exploring a Computational Framework for Analyzing Escape Responses in Plains Zebra (Equus quagga)
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16882v2](http://arxiv.org/pdf/2505.16882v2)**

> **作者:** Isla Duporge; Sofia Minano; Nikoloz Sirmpilatze; Igor Tatarnikov; Scott Wolf; Adam L. Tyson; Daniel Rubenstein
>
> **备注:** Accepted to the CV4Animals workshop at CVPR 2025
>
> **摘要:** Ethological research increasingly benefits from the growing affordability and accessibility of drones, which enable the capture of high-resolution footage of animal movement at fine spatial and temporal scales. However, analyzing such footage presents the technical challenge of separating animal movement from drone motion. While non-trivial, computer vision techniques such as image registration and Structure-from-Motion (SfM) offer practical solutions. For conservationists, open-source tools that are user-friendly, require minimal setup, and deliver timely results are especially valuable for efficient data interpretation. This study evaluates three approaches: a bioimaging-based registration technique, an SfM pipeline, and a hybrid interpolation method. We apply these to a recorded escape event involving 44 plains zebras, captured in a single drone video. Using the best-performing method, we extract individual trajectories and identify key behavioral patterns: increased alignment (polarization) during escape, a brief widening of spacing just before stopping, and tighter coordination near the group's center. These insights highlight the method's effectiveness and its potential to scale to larger datasets, contributing to broader investigations of collective animal behavior.
>
---
#### [replaced 075] GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04119v2](http://arxiv.org/pdf/2505.04119v2)**

> **作者:** Zixiang Ai; Zichen Liu; Yuanhang Lei; Zhenyu Cui; Xu Zou; Jiahuan Zhou
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Pre-trained 3D vision models have gained significant attention for their promising performance on point cloud data. However, fully fine-tuning these models for downstream tasks is computationally expensive and storage-intensive. Existing parameter-efficient fine-tuning (PEFT) approaches, which focus primarily on input token prompting, struggle to achieve competitive performance due to their limited ability to capture the geometric information inherent in point clouds. To address this challenge, we propose a novel Geometry-Aware Point Cloud Prompt (GAPrompt) that leverages geometric cues to enhance the adaptability of 3D vision models. First, we introduce a Point Prompt that serves as an auxiliary input alongside the original point cloud, explicitly guiding the model to capture fine-grained geometric details. Additionally, we present a Point Shift Prompter designed to extract global shape information from the point cloud, enabling instance-specific geometric adjustments at the input level. Moreover, our proposed Prompt Propagation mechanism incorporates the shape information into the model's feature extraction process, further strengthening its ability to capture essential geometric characteristics. Extensive experiments demonstrate that GAPrompt significantly outperforms state-of-the-art PEFT methods and achieves competitive results compared to full fine-tuning on various benchmarks, while utilizing only 2.19% of trainable parameters. Our code is available at https://github.com/zhoujiahuan1991/ICML2025-VGP.
>
---
#### [replaced 076] Vision Graph Prompting via Semantic Low-Rank Decomposition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04121v2](http://arxiv.org/pdf/2505.04121v2)**

> **作者:** Zixiang Ai; Zichen Liu; Jiahuan Zhou
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Vision GNN (ViG) demonstrates superior performance by representing images as graph structures, providing a more natural way to capture irregular semantic patterns beyond traditional grid or sequence-based representations. To efficiently adapt ViG to downstream tasks, parameter-efficient fine-tuning techniques like visual prompting become increasingly essential. However, existing prompting methods are primarily designed for Transformer-based models, neglecting the rich topological relationships among nodes and edges in graph-based representations, limiting their capacity to model complex semantics. In this paper, we propose Vision Graph Prompting (VGP), a novel framework tailored for vision graph structures. Our core insight reveals that semantically connected components in the graph exhibit low-rank properties. Building on this observation, we introduce a semantic low-rank prompting method that decomposes low-rank semantic features and integrates them with prompts on vision graph topologies, capturing both global structural patterns and fine-grained semantic dependencies. Extensive experiments demonstrate our method significantly improves ViG's transfer performance on diverse downstream tasks, achieving results comparable to full fine-tuning while maintaining parameter efficiency. Our code is available at https://github.com/zhoujiahuan1991/ICML2025-VGP.
>
---
#### [replaced 077] SUFFICIENT: A scan-specific unsupervised deep learning framework for high-resolution 3D isotropic fetal brain MRI reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17472v2](http://arxiv.org/pdf/2505.17472v2)**

> **作者:** Jiangjie Wu; Lixuan Chen; Zhenghao Li; Xin Li; Saban Ozturk; Lihui Wang; Rongpin Wang; Hongjiang Wei; Yuyao Zhang
>
> **摘要:** High-quality 3D fetal brain MRI reconstruction from motion-corrupted 2D slices is crucial for clinical diagnosis. Reliable slice-to-volume registration (SVR)-based motion correction and super-resolution reconstruction (SRR) methods are essential. Deep learning (DL) has demonstrated potential in enhancing SVR and SRR when compared to conventional methods. However, it requires large-scale external training datasets, which are difficult to obtain for clinical fetal MRI. To address this issue, we propose an unsupervised iterative SVR-SRR framework for isotropic HR volume reconstruction. Specifically, SVR is formulated as a function mapping a 2D slice and a 3D target volume to a rigid transformation matrix, which aligns the slice to the underlying location in the target volume. The function is parameterized by a convolutional neural network, which is trained by minimizing the difference between the volume slicing at the predicted position and the input slice. In SRR, a decoding network embedded within a deep image prior framework is incorporated with a comprehensive image degradation model to produce the high-resolution (HR) volume. The deep image prior framework offers a local consistency prior to guide the reconstruction of HR volumes. By performing a forward degradation model, the HR volume is optimized by minimizing loss between predicted slices and the observed slices. Comprehensive experiments conducted on large-magnitude motion-corrupted simulation data and clinical data demonstrate the superior performance of the proposed framework over state-of-the-art fetal brain reconstruction frameworks.
>
---
#### [replaced 078] IT$^3$: Idempotent Test-Time Training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.04201v2](http://arxiv.org/pdf/2410.04201v2)**

> **作者:** Nikita Durasov; Assaf Shocher; Doruk Oner; Gal Chechik; Alexei A. Efros; Pascal Fua
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Deep learning models often struggle when deployed in real-world settings due to distribution shifts between training and test data. While existing approaches like domain adaptation and test-time training (TTT) offer partial solutions, they typically require additional data or domain-specific auxiliary tasks. We present Idempotent Test-Time Training (IT$^3$), a novel approach that enables on-the-fly adaptation to distribution shifts using only the current test instance, without any auxiliary task design. Our key insight is that enforcing idempotence -- where repeated applications of a function yield the same result -- can effectively replace domain-specific auxiliary tasks used in previous TTT methods. We theoretically connect idempotence to prediction confidence and demonstrate that minimizing the distance between successive applications of our model during inference leads to improved out-of-distribution performance. Extensive experiments across diverse domains (including image classification, aerodynamics prediction, and aerial segmentation) and architectures (MLPs, CNNs, GNNs) show that IT$^3$ consistently outperforms existing approaches while being simpler and more widely applicable. Our results suggest that idempotence provides a universal principle for test-time adaptation that generalizes across domains and architectures.
>
---
#### [replaced 079] NovelSeek: When Agent Becomes the Scientist -- Building Closed-Loop System from Hypothesis to Verification
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16938v2](http://arxiv.org/pdf/2505.16938v2)**

> **作者:** NovelSeek Team; Bo Zhang; Shiyang Feng; Xiangchao Yan; Jiakang Yuan; Zhiyin Yu; Xiaohan He; Songtao Huang; Shaowei Hou; Zheng Nie; Zhilong Wang; Jinyao Liu; Runmin Ma; Tianshuo Peng; Peng Ye; Dongzhan Zhou; Shufei Zhang; Xiaosong Wang; Yilan Zhang; Meng Li; Zhongying Tu; Xiangyu Yue; Wangli Ouyang; Bowen Zhou; Lei Bai
>
> **备注:** HomePage: https://alpha-innovator.github.io/NovelSeek-project-page
>
> **摘要:** Artificial Intelligence (AI) is accelerating the transformation of scientific research paradigms, not only enhancing research efficiency but also driving innovation. We introduce NovelSeek, a unified closed-loop multi-agent framework to conduct Autonomous Scientific Research (ASR) across various scientific research fields, enabling researchers to tackle complicated problems in these fields with unprecedented speed and precision. NovelSeek highlights three key advantages: 1) Scalability: NovelSeek has demonstrated its versatility across 12 scientific research tasks, capable of generating innovative ideas to enhance the performance of baseline code. 2) Interactivity: NovelSeek provides an interface for human expert feedback and multi-agent interaction in automated end-to-end processes, allowing for the seamless integration of domain expert knowledge. 3) Efficiency: NovelSeek has achieved promising performance gains in several scientific fields with significantly less time cost compared to human efforts. For instance, in reaction yield prediction, it increased from 27.6% to 35.4% in just 12 hours; in enhancer activity prediction, accuracy rose from 0.65 to 0.79 with only 4 hours of processing; and in 2D semantic segmentation, precision advanced from 78.8% to 81.0% in a mere 30 hours.
>
---
#### [replaced 080] CLIP-UP: A Simple and Efficient Mixture-of-Experts CLIP Training Recipe with Sparse Upcycling
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.00965v2](http://arxiv.org/pdf/2502.00965v2)**

> **作者:** Xinze Wang; Chen Chen; Yinfei Yang; Hong-You Chen; Bowen Zhang; Aditya Pal; Xiangxin Zhu; Xianzhi Du
>
> **摘要:** Mixture-of-Experts (MoE) models are crucial for scaling model capacity while controlling inference costs. While integrating MoE into multimodal models like CLIP improves performance, training these models is notoriously challenging and expensive. We propose CLIP-Upcycling (CLIP-UP), an efficient alternative training strategy that converts a pre-trained dense CLIP model into a sparse MoE architecture. Through extensive experimentation with various settings and auxiliary losses, we demonstrate that CLIP-UP significantly reduces training complexity and cost. Remarkably, our sparse CLIP B/16 model, trained with CLIP-UP, outperforms its dense counterpart by 7.2% and 6.6% on COCO and Flickr30k text-to-image Recall@1 benchmarks respectively. It even surpasses the larger CLIP L/14 model on this task while using only 30% of the inference FLOPs. We further demonstrate the generalizability of our training recipe across different scales, establishing sparse upcycling as a practical and scalable approach for building efficient, high-performance CLIP models.
>
---
#### [replaced 081] Learning Transformer-based World Models with Contrastive Predictive Coding
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.04416v2](http://arxiv.org/pdf/2503.04416v2)**

> **作者:** Maxime Burchi; Radu Timofte
>
> **摘要:** The DreamerV3 algorithm recently obtained remarkable performance across diverse environment domains by learning an accurate world model based on Recurrent Neural Networks (RNNs). Following the success of model-based reinforcement learning algorithms and the rapid adoption of the Transformer architecture for its superior training efficiency and favorable scaling properties, recent works such as STORM have proposed replacing RNN-based world models with Transformer-based world models using masked self-attention. However, despite the improved training efficiency of these methods, their impact on performance remains limited compared to the Dreamer algorithm, struggling to learn competitive Transformer-based world models. In this work, we show that the next state prediction objective adopted in previous approaches is insufficient to fully exploit the representation capabilities of Transformers. We propose to extend world model predictions to longer time horizons by introducing TWISTER (Transformer-based World model wIth contraSTivE Representations), a world model using action-conditioned Contrastive Predictive Coding to learn high-level temporal feature representations and improve the agent performance. TWISTER achieves a human-normalized mean score of 162% on the Atari 100k benchmark, setting a new record among state-of-the-art methods that do not employ look-ahead search.
>
---
#### [replaced 082] Domain-Agnostic Stroke Lesion Segmentation Using Physics-Constrained Synthetic Data
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2412.03318v2](http://arxiv.org/pdf/2412.03318v2)**

> **作者:** Liam Chalcroft; Jenny Crinion; Cathy J. Price; John Ashburner
>
> **摘要:** Segmenting stroke lesions in MRI is challenging due to diverse acquisition protocols that limit model generalisability. In this work, we introduce two physics-constrained approaches to generate synthetic quantitative MRI (qMRI) images that improve segmentation robustness across heterogeneous domains. Our first method, $\texttt{qATLAS}$, trains a neural network to estimate qMRI maps from standard MPRAGE images, enabling the simulation of varied MRI sequences with realistic tissue contrasts. The second method, $\texttt{qSynth}$, synthesises qMRI maps directly from tissue labels using label-conditioned Gaussian mixture models, ensuring physical plausibility. Extensive experiments on multiple out-of-domain datasets show that both methods outperform a baseline UNet, with $\texttt{qSynth}$ notably surpassing previous synthetic data approaches. These results highlight the promise of integrating MRI physics into synthetic data generation for robust, generalisable stroke lesion segmentation. Code is available at https://github.com/liamchalcroft/qsynth
>
---
#### [replaced 083] LinGen: Towards High-Resolution Minute-Length Text-to-Video Generation with Linear Computational Complexity
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.09856v2](http://arxiv.org/pdf/2412.09856v2)**

> **作者:** Hongjie Wang; Chih-Yao Ma; Yen-Cheng Liu; Ji Hou; Tao Xu; Jialiang Wang; Felix Juefei-Xu; Yaqiao Luo; Peizhao Zhang; Tingbo Hou; Peter Vajda; Niraj K. Jha; Xiaoliang Dai
>
> **备注:** Accepted to IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025
>
> **摘要:** Text-to-video generation enhances content creation but is highly computationally intensive: The computational cost of Diffusion Transformers (DiTs) scales quadratically in the number of pixels. This makes minute-length video generation extremely expensive, limiting most existing models to generating videos of only 10-20 seconds length. We propose a Linear-complexity text-to-video Generation (LinGen) framework whose cost scales linearly in the number of pixels. For the first time, LinGen enables high-resolution minute-length video generation on a single GPU without compromising quality. It replaces the computationally-dominant and quadratic-complexity block, self-attention, with a linear-complexity block called MATE, which consists of an MA-branch and a TE-branch. The MA-branch targets short-to-long-range correlations, combining a bidirectional Mamba2 block with our token rearrangement method, Rotary Major Scan, and our review tokens developed for long video generation. The TE-branch is a novel TEmporal Swin Attention block that focuses on temporal correlations between adjacent tokens and medium-range tokens. The MATE block addresses the adjacency preservation issue of Mamba and improves the consistency of generated videos significantly. Experimental results show that LinGen outperforms DiT (with a 75.6% win rate) in video quality with up to 15$\times$ (11.5$\times$) FLOPs (latency) reduction. Furthermore, both automatic metrics and human evaluation demonstrate our LinGen-4B yields comparable video quality to state-of-the-art models (with a 50.5%, 52.1%, 49.1% win rate with respect to Gen-3, LumaLabs, and Kling, respectively). This paves the way to hour-length movie generation and real-time interactive video generation. We provide 68s video generation results and more examples in our project website: https://lineargen.github.io/.
>
---
#### [replaced 084] GlobalGeoTree: A Multi-Granular Vision-Language Dataset for Global Tree Species Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12513v2](http://arxiv.org/pdf/2505.12513v2)**

> **作者:** Yang Mu; Zhitong Xiong; Yi Wang; Muhammad Shahzad; Franz Essl; Mark van Kleunen; Xiao Xiang Zhu
>
> **摘要:** Global tree species mapping using remote sensing data is vital for biodiversity monitoring, forest management, and ecological research. However, progress in this field has been constrained by the scarcity of large-scale, labeled datasets. To address this, we introduce GlobalGeoTree, a comprehensive global dataset for tree species classification. GlobalGeoTree comprises 6.3 million geolocated tree occurrences, spanning 275 families, 2,734 genera, and 21,001 species across the hierarchical taxonomic levels. Each sample is paired with Sentinel-2 image time series and 27 auxiliary environmental variables, encompassing bioclimatic, geographic, and soil data. The dataset is partitioned into GlobalGeoTree-6M for model pretraining and curated evaluation subsets, primarily GlobalGeoTree-10kEval for zero-shot and few-shot benchmarking. To demonstrate the utility of the dataset, we introduce a baseline model, GeoTreeCLIP, which leverages paired remote sensing data and taxonomic text labels within a vision-language framework pretrained on GlobalGeoTree-6M. Experimental results show that GeoTreeCLIP achieves substantial improvements in zero- and few-shot classification on GlobalGeoTree-10kEval over existing advanced models. By making the dataset, models, and code publicly available, we aim to establish a benchmark to advance tree species classification and foster innovation in biodiversity research and ecological applications.
>
---
#### [replaced 085] Navigating Conflicting Views: Harnessing Trust for Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.00958v3](http://arxiv.org/pdf/2406.00958v3)**

> **作者:** Jueqing Lu; Wray Buntine; Yuanyuan Qi; Joanna Dipnall; Belinda Gabbe; Lan Du
>
> **摘要:** Resolving conflicts is critical for improving the reliability of multi-view classification. While prior work focuses on learning consistent and informative representations across views, it often assumes perfect alignment and equal importance of all views, an assumption rarely met in real-world scenarios, as some views may express distinct information. To address this, we develop a computational trust-based discounting method that enhances the Evidential Multi-view framework by accounting for the instance-wise reliability of each view through a probability-sensitive trust mechanism. We evaluate our method on six real-world datasets using Top-1 Accuracy, Fleiss' Kappa, and a new metric, Multi-View Agreement with Ground Truth, to assess prediction reliability. We also assess the effectiveness of uncertainty in indicating prediction correctness via AUROC.Additionally, we test the scalability of our method through end-to-end training on a large-scale dataset. The experimental results show that computational trust can effectively resolve conflicts, paving the way for more reliable multi-view classification models in real-world applications.
>
---
#### [replaced 086] Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.04395v2](http://arxiv.org/pdf/2502.04395v2)**

> **作者:** Siru Zhong; Weilin Ruan; Ming Jin; Huan Li; Qingsong Wen; Yuxuan Liang
>
> **备注:** 20 pages
>
> **摘要:** Recent advancements in time series forecasting have explored augmenting models with text or vision modalities to improve accuracy. While text provides contextual understanding, it often lacks fine-grained temporal details. Conversely, vision captures intricate temporal patterns but lacks semantic context, limiting the complementary potential of these modalities. To address this, we propose \method, a novel multimodal framework that leverages pre-trained Vision-Language Models (VLMs) to bridge temporal, visual, and textual modalities for enhanced forecasting. Our framework comprises three key components: (1) a Retrieval-Augmented Learner, which extracts enriched temporal features through memory bank interactions; (2) a Vision-Augmented Learner, which encodes time series as informative images; and (3) a Text-Augmented Learner, which generates contextual textual descriptions. These components collaborate with frozen pre-trained VLMs to produce multimodal embeddings, which are then fused with temporal features for final prediction. Extensive experiments demonstrate that Time-VLM achieves superior performance, particularly in few-shot and zero-shot scenarios, thereby establishing a new direction for multimodal time series forecasting. Code is available at https://github.com/CityMind-Lab/ICML25-TimeVLM.
>
---
#### [replaced 087] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding
- **分类: cs.LG; cs.AI; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.15946v2](http://arxiv.org/pdf/2505.15946v2)**

> **作者:** Yuxiang Wei; Yanteng Zhang; Xi Xiao; Tianyang Wang; Xiao Wang; Vince D. Calhoun
>
> **摘要:** Decoding visual experiences from fMRI offers a powerful avenue to understand human perception and develop advanced brain-computer interfaces. However, current progress often prioritizes maximizing reconstruction fidelity while overlooking interpretability, an essential aspect for deriving neuroscientific insight. To address this gap, we propose MoRE-Brain, a neuro-inspired framework designed for high-fidelity, adaptable, and interpretable visual reconstruction. MoRE-Brain uniquely employs a hierarchical Mixture-of-Experts architecture where distinct experts process fMRI signals from functionally related voxel groups, mimicking specialized brain networks. The experts are first trained to encode fMRI into the frozen CLIP space. A finetuned diffusion model then synthesizes images, guided by expert outputs through a novel dual-stage routing mechanism that dynamically weighs expert contributions across the diffusion process. MoRE-Brain offers three main advancements: First, it introduces a novel Mixture-of-Experts architecture grounded in brain network principles for neuro-decoding. Second, it achieves efficient cross-subject generalization by sharing core expert networks while adapting only subject-specific routers. Third, it provides enhanced mechanistic insight, as the explicit routing reveals precisely how different modeled brain regions shape the semantic and spatial attributes of the reconstructed image. Extensive experiments validate MoRE-Brain's high reconstruction fidelity, with bottleneck analyses further demonstrating its effective utilization of fMRI signals, distinguishing genuine neural decoding from over-reliance on generative priors. Consequently, MoRE-Brain marks a substantial advance towards more generalizable and interpretable fMRI-based visual decoding. Code will be publicly available soon: https://github.com/yuxiangwei0808/MoRE-Brain.
>
---
#### [replaced 088] Marmot: Multi-Agent Reasoning for Multi-Object Self-Correcting in Improving Image-Text Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20054v2](http://arxiv.org/pdf/2504.20054v2)**

> **作者:** Jiayang Sun; Hongbo Wang; Jie Cao; Huaibo Huang; Ran He
>
> **摘要:** While diffusion models excel at generating high-quality images, they often struggle with accurate counting, attributes, and spatial relationships in complex multi-object scenes. One potential approach is to utilize Multimodal Large Language Model (MLLM) as an AI agent to build a self-correction framework. However, these approaches are highly dependent on the capabilities of the employed MLLM, often failing to account for all objects within the image. To address these challenges, we propose Marmot, a novel and generalizable framework that employs Multi-Agent Reasoning for Multi-Object Self-Correcting, enhancing image-text alignment and facilitating more coherent multi-object image editing. Our framework adopts a divide-and-conquer strategy, decomposing the self-correction task into object-level subtasks according to three critical dimensions: counting, attributes, and spatial relationships. We construct a multi-agent self-correcting system featuring a decision-execution-verification mechanism, effectively mitigating inter-object interference and enhancing editing reliability. To resolve the problem of subtask integration, we propose a Pixel-Domain Stitching Smoother that employs mask-guided two-stage latent space optimization. This innovation enables parallel processing of subtask results, thereby enhancing runtime efficiency while eliminating multi-stage distortion accumulation. Extensive experiments demonstrate that Marmot significantly improves accuracy in object counting, attribute assignment, and spatial relationships for image generation tasks.
>
---
#### [replaced 089] EVM-Fusion: An Explainable Vision Mamba Architecture with Neural Algorithmic Fusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17367v2](http://arxiv.org/pdf/2505.17367v2)**

> **作者:** Zichuan Yang
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Medical image classification is critical for clinical decision-making, yet demands for accuracy, interpretability, and generalizability remain challenging. This paper introduces EVM-Fusion, an Explainable Vision Mamba architecture featuring a novel Neural Algorithmic Fusion (NAF) mechanism for multi-organ medical image classification. EVM-Fusion leverages a multipath design, where DenseNet and U-Net based pathways, enhanced by Vision Mamba (Vim) modules, operate in parallel with a traditional feature pathway. These diverse features are dynamically integrated via a two-stage fusion process: cross-modal attention followed by the iterative NAF block, which learns an adaptive fusion algorithm. Intrinsic explainability is embedded through path-specific spatial attention, Vim {\Delta}-value maps, traditional feature SE-attention, and cross-modal attention weights. Experiments on a diverse 9-class multi-organ medical image dataset demonstrate EVM-Fusion's strong classification performance, achieving 99.75% test accuracy and provide multi-faceted insights into its decision-making process, highlighting its potential for trustworthy AI in medical diagnostics.
>
---
#### [replaced 090] FairREAD: Re-fusing Demographic Attributes after Disentanglement for Fair Medical Image Classification
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2412.16373v2](http://arxiv.org/pdf/2412.16373v2)**

> **作者:** Yicheng Gao; Jinkui Hao; Bo Zhou
>
> **备注:** This work has been submitted to Medical Image Analysis, Elsevier for possible publication
>
> **摘要:** Recent advancements in deep learning have shown transformative potential in medical imaging, yet concerns about fairness persist due to performance disparities across demographic subgroups. Existing methods aim to address these biases by mitigating sensitive attributes in image data; however, these attributes often carry clinically relevant information, and their removal can compromise model performance-a highly undesirable outcome. To address this challenge, we propose Fair Re-fusion After Disentanglement (FairREAD), a novel, simple, and efficient framework that mitigates unfairness by re-integrating sensitive demographic attributes into fair image representations. FairREAD employs orthogonality constraints and adversarial training to disentangle demographic information while using a controlled re-fusion mechanism to preserve clinically relevant details. Additionally, subgroup-specific threshold adjustments ensure equitable performance across demographic groups. Comprehensive evaluations on a large-scale clinical X-ray dataset demonstrate that FairREAD significantly reduces unfairness metrics while maintaining diagnostic accuracy, establishing a new benchmark for fairness and performance in medical image classification.
>
---
#### [replaced 091] AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.11706v3](http://arxiv.org/pdf/2412.11706v3)**

> **作者:** Wenhao Sun; Rong-Cheng Tu; Jingyi Liao; Zhao Jin; Dacheng Tao
>
> **备注:** 18 pages, 14 figures. Accepted by ICML 2025. The code is available at https://github.com/wenhao728/AsymRnR
>
> **摘要:** Diffusion Transformers (DiTs) have proven effective in generating high-quality videos but are hindered by high computational costs. Existing video DiT sampling acceleration methods often rely on costly fine-tuning or exhibit limited generalization capabilities. We propose Asymmetric Reduction and Restoration (AsymRnR), a training-free and model-agnostic method to accelerate video DiTs. It builds on the observation that redundancies of feature tokens in DiTs vary significantly across different model blocks, denoising steps, and feature types. Our AsymRnR asymmetrically reduces redundant tokens in the attention operation, achieving acceleration with negligible degradation in output quality and, in some cases, even improving it. We also tailored a reduction schedule to distribute the reduction across components adaptively. To further accelerate this process, we introduce a matching cache for more efficient reduction. Backed by theoretical foundations and extensive experimental validation, AsymRnR integrates into state-of-the-art video DiTs and offers substantial speedup.
>
---
#### [replaced 092] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2410.09403v3](http://arxiv.org/pdf/2410.09403v3)**

> **作者:** Haoyang Su; Renqi Chen; Shixiang Tang; Zhenfei Yin; Xinzhe Zheng; Jinzhe Li; Biqing Qi; Qi Wu; Hui Li; Wanli Ouyang; Philip Torr; Bowen Zhou; Nanqing Dong
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** The rapid advancement of scientific progress requires innovative tools that can accelerate knowledge discovery. Although recent AI methods, particularly large language models (LLMs), have shown promise in tasks such as hypothesis generation and experimental design, they fall short of replicating the collaborative nature of real-world scientific practices, where diverse experts work together in teams to tackle complex problems. To address the limitations, we propose an LLM-based multi-agent system, i.e., Virtual Scientists (VirSci), designed to mimic the teamwork inherent in scientific research. VirSci organizes a team of agents to collaboratively generate, evaluate, and refine research ideas. Through comprehensive experiments, we demonstrate that this multi-agent approach outperforms the state-of-the-art method in producing novel scientific ideas. We further investigate the collaboration mechanisms that contribute to its tendency to produce ideas with higher novelty, offering valuable insights to guide future research and illuminating pathways toward building a robust system for autonomous scientific discovery. The code is available at https://github.com/open-sciencelab/Virtual-Scientists.
>
---
#### [replaced 093] TokBench: Evaluating Your Visual Tokenizer before Visual Generation
- **分类: cs.CV; cs.DB**

- **链接: [http://arxiv.org/pdf/2505.18142v2](http://arxiv.org/pdf/2505.18142v2)**

> **作者:** Junfeng Wu; Dongliang Luo; Weizhi Zhao; Zhihao Xie; Yuanhao Wang; Junyi Li; Xudong Xie; Yuliang Liu; Xiang Bai
>
> **备注:** Benchmark, homepagee: https://wjf5203.github.io/TokBench
>
> **摘要:** In this work, we reveal the limitations of visual tokenizers and VAEs in preserving fine-grained features, and propose a benchmark to evaluate reconstruction performance for two challenging visual contents: text and face. Visual tokenizers and VAEs have significantly advanced visual generation and multimodal modeling by providing more efficient compressed or quantized image representations. However, while helping production models reduce computational burdens, the information loss from image compression fundamentally limits the upper bound of visual generation quality. To evaluate this upper bound, we focus on assessing reconstructed text and facial features since they typically: 1) exist at smaller scales, 2) contain dense and rich textures, 3) are prone to collapse, and 4) are highly sensitive to human vision. We first collect and curate a diverse set of clear text and face images from existing datasets. Unlike approaches using VLM models, we employ established OCR and face recognition models for evaluation, ensuring accuracy while maintaining an exceptionally lightweight assessment process <span style="font-weight: bold; color: rgb(214, 21, 21);">requiring just 2GB memory and 4 minutes</span> to complete. Using our benchmark, we analyze text and face reconstruction quality across various scales for different image tokenizers and VAEs. Our results show modern visual tokenizers still struggle to preserve fine-grained features, especially at smaller scales. We further extend this evaluation framework to video, conducting comprehensive analysis of video tokenizers. Additionally, we demonstrate that traditional metrics fail to accurately reflect reconstruction performance for faces and text, while our proposed metrics serve as an effective complement.
>
---
#### [replaced 094] Envisioning Beyond the Pixels: Benchmarking Reasoning-Informed Visual Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02826v3](http://arxiv.org/pdf/2504.02826v3)**

> **作者:** Xiangyu Zhao; Peiyuan Zhang; Kexian Tang; Xiaorong Zhu; Hao Li; Wenhao Chai; Zicheng Zhang; Xiaqiu Ren; Guangtao Zhai; Junchi Yan; Hua Yang; Xue Yang; Haodong Duan
>
> **摘要:** Large Multi-modality Models (LMMs) have made significant progress in visual understanding and generation, but they still face challenges in General Visual Editing, particularly in following complex instructions, preserving appearance consistency, and supporting flexible input formats. To study this gap, we introduce RISEBench, the first benchmark for evaluating Reasoning-Informed viSual Editing (RISE). RISEBench focuses on four key reasoning categories: Temporal, Causal, Spatial, and Logical Reasoning. We curate high-quality test cases for each category and propose an robust evaluation framework that assesses Instruction Reasoning, Appearance Consistency, and Visual Plausibility with both human judges and the LMM-as-a-judge approach. We conducted experiments evaluating nine prominent visual editing models, comprising both open-source and proprietary models. The evaluation results demonstrate that current models face significant challenges in reasoning-based editing tasks. Even the most powerful model evaluated, GPT-4o-Image, achieves an accuracy of merely 28.8%. RISEBench effectively highlights the limitations of contemporary editing models, provides valuable insights, and indicates potential future directions for the field of reasoning-aware visual editing. Our code and data have been released at https://github.com/PhoenixZ810/RISEBench.
>
---
#### [replaced 095] Live Video Captioning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.14206v2](http://arxiv.org/pdf/2406.14206v2)**

> **作者:** Eduardo Blanco-Fernández; Carlos Gutiérrez-Álvarez; Nadia Nasri; Saturnino Maldonado-Bascón; Roberto J. López-Sastre
>
> **摘要:** Dense video captioning involves detecting and describing events within video sequences. Traditional methods operate in an offline setting, assuming the entire video is available for analysis. In contrast, in this work we introduce a groundbreaking paradigm: Live Video Captioning (LVC), where captions must be generated for video streams in an online manner. This shift brings unique challenges, including processing partial observations of the events and the need for a temporal anticipation of the actions. We formally define the novel problem of LVC and propose innovative evaluation metrics specifically designed for this online scenario, demonstrating their advantages over traditional metrics. To address the novel complexities of LVC, we present a new model that combines deformable transformers with temporal filtering, enabling effective captioning over video streams. Extensive experiments on the ActivityNet Captions dataset validate the proposed approach, showcasing its superior performance in the LVC setting compared to state-of-the-art offline methods. To foster further research, we provide the results of our model and an evaluation toolkit with the new metrics integrated at: https://github.com/gramuah/lvc.
>
---
#### [replaced 096] ViSIR: Vision Transformer Single Image Reconstruction Method for Earth System Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06741v3](http://arxiv.org/pdf/2502.06741v3)**

> **作者:** Ehsan Zeraatkar; Salah Faroughi; Jelena Tešić
>
> **摘要:** Purpose: Earth system models (ESMs) integrate the interactions of the atmosphere, ocean, land, ice, and biosphere to estimate the state of regional and global climate under a wide variety of conditions. The ESMs are highly complex; thus, deep neural network architectures are used to model the complexity and store the down-sampled data. This paper proposes the Vision Transformer Sinusoidal Representation Networks (ViSIR) to improve the ESM data's single image SR (SR) reconstruction task. Methods: ViSIR combines the SR capability of Vision Transformers (ViT) with the high-frequency detail preservation of the Sinusoidal Representation Network (SIREN) to address the spectral bias observed in SR tasks. Results: The ViSIR outperforms SRCNN by 2.16 db, ViT by 6.29 dB, SIREN by 8.34 dB, and SR-Generative Adversarial (SRGANs) by 7.93 dB PSNR on average for three different measurements. Conclusion: The proposed ViSIR is evaluated and compared with state-of-the-art methods. The results show that the proposed algorithm is outperforming other methods in terms of Mean Square Error(MSE), Peak-Signal-to-Noise-Ratio(PSNR), and Structural Similarity Index Measure(SSIM).
>
---
#### [replaced 097] Compositional Physical Reasoning of Objects and Events from Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.02687v2](http://arxiv.org/pdf/2408.02687v2)**

> **作者:** Zhenfang Chen; Shilong Dong; Kexin Yi; Yunzhu Li; Mingyu Ding; Antonio Torralba; Joshua B. Tenenbaum; Chuang Gan
>
> **备注:** Accepted by TPAMI 2025. arXiv admin note: text overlap with arXiv:2205.01089
>
> **摘要:** Understanding and reasoning about objects' physical properties in the natural world is a fundamental challenge in artificial intelligence. While some properties like colors and shapes can be directly observed, others, such as mass and electric charge, are hidden from the objects' visual appearance. This paper addresses the unique challenge of inferring these hidden physical properties from objects' motion and interactions and predicting corresponding dynamics based on the inferred physical properties. We first introduce the Compositional Physical Reasoning (ComPhy) dataset. For a given set of objects, ComPhy includes limited videos of them moving and interacting under different initial conditions. The model is evaluated based on its capability to unravel the compositional hidden properties, such as mass and charge, and use this knowledge to answer a set of questions. Besides the synthetic videos from simulators, we also collect a real-world dataset to show further test physical reasoning abilities of different models. We evaluate state-of-the-art video reasoning models on ComPhy and reveal their limited ability to capture these hidden properties, which leads to inferior performance. We also propose a novel neuro-symbolic framework, Physical Concept Reasoner (PCR), that learns and reasons about both visible and hidden physical properties from question answering. After training, PCR demonstrates remarkable capabilities. It can detect and associate objects across frames, ground visible and hidden physical properties, make future and counterfactual predictions, and utilize these extracted representations to answer challenging questions.
>
---
#### [replaced 098] Beyond One-Hot Labels: Semantic Mixing for Model Calibration
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.13548v2](http://arxiv.org/pdf/2504.13548v2)**

> **作者:** Haoyang Luo; Linwei Tao; Minjing Dong; Chang Xu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Model calibration seeks to ensure that models produce confidence scores that accurately reflect the true likelihood of their predictions being correct. However, existing calibration approaches are fundamentally tied to datasets of one-hot labels implicitly assuming full certainty in all the annotations. Such datasets are effective for classification but provides insufficient knowledge of uncertainty for model calibration, necessitating the curation of datasets with numerically rich ground-truth confidence values. However, due to the scarcity of uncertain visual examples, such samples are not easily available as real datasets. In this paper, we introduce calibration-aware data augmentation to create synthetic datasets of diverse samples and their ground-truth uncertainty. Specifically, we present \textbf{Calibration-aware Semantic Mixing (CSM)}, a novel framework that generates training samples with mixed class characteristics and annotates them with distinct confidence scores via diffusion models. Based on this framework, we propose calibrated reannotation to tackle the misalignment between the annotated confidence score and the mixing ratio during the diffusion reverse process. Besides, we explore the loss functions that better fit the new data representation paradigm. Experimental results demonstrate that CSM achieves superior calibration compared to the state-of-the-art calibration approaches. Our code is \href{https://github.com/E-Galois/CSM}{available here}.
>
---
#### [replaced 099] Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17726v2](http://arxiv.org/pdf/2505.17726v2)**

> **作者:** Donghwan Chi; Hyomin Kim; Yoonjin Oh; Yongjin Kim; Donghoon Lee; Daejin Jo; Jongmin Kim; Junyeob Baek; Sungjin Ahn; Sungwoong Kim
>
> **摘要:** Recently, multimodal large language models (MLLMs) have emerged as a key approach in achieving artificial general intelligence. In particular, vision-language MLLMs have been developed to generate not only text but also visual outputs from multimodal inputs. This advancement requires efficient image tokens that LLMs can process effectively both in input and output. However, existing image tokenization methods for MLLMs typically capture only global abstract concepts or uniformly segmented image patches, restricting MLLMs' capability to effectively understand or generate detailed visual content, particularly at the object level. To address this limitation, we propose an object-centric visual tokenizer based on Slot Attention specifically for MLLMs. In particular, based on the Q-Former encoder, diffusion decoder, and residual vector quantization, our proposed discretized slot tokens can encode local visual details while maintaining high-level semantics, and also align with textual data to be integrated seamlessly within a unified next-token prediction framework of LLMs. The resulting Slot-MLLM demonstrates significant performance improvements over baselines with previous visual tokenizers across various vision-language tasks that entail local detailed comprehension and generation. Notably, this work is the first demonstration of the feasibility of object-centric slot attention performed with MLLMs and in-the-wild natural images.
>
---
#### [replaced 100] STAR-R1: Spatial TrAnsformation Reasoning by Reinforcing Multimodal LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15804v2](http://arxiv.org/pdf/2505.15804v2)**

> **作者:** Zongzhao Li; Zongyang Ma; Mingze Li; Songyou Li; Yu Rong; Tingyang Xu; Ziqi Zhang; Deli Zhao; Wenbing Huang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across diverse tasks, yet they lag significantly behind humans in spatial reasoning. We investigate this gap through Transformation-Driven Visual Reasoning (TVR), a challenging task requiring identification of object transformations across images under varying viewpoints. While traditional Supervised Fine-Tuning (SFT) fails to generate coherent reasoning paths in cross-view settings, sparse-reward Reinforcement Learning (RL) suffers from inefficient exploration and slow convergence. To address these limitations, we propose STAR-R1, a novel framework that integrates a single-stage RL paradigm with a fine-grained reward mechanism tailored for TVR. Specifically, STAR-R1 rewards partial correctness while penalizing excessive enumeration and passive inaction, enabling efficient exploration and precise reasoning. Comprehensive evaluations demonstrate that STAR-R1 achieves state-of-the-art performance across all 11 metrics, outperforming SFT by 23% in cross-view scenarios. Further analysis reveals STAR-R1's anthropomorphic behavior and highlights its unique ability to compare all objects for improving spatial reasoning. Our work provides critical insights in advancing the research of MLLMs and reasoning models. The codes, model weights, and data will be publicly available at https://github.com/zongzhao23/STAR-R1.
>
---
#### [replaced 101] DMAGaze: Gaze Estimation Based on Feature Disentanglement and Multi-Scale Attention
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.11160v2](http://arxiv.org/pdf/2504.11160v2)**

> **作者:** Haohan Chen; Hongjia Liu; Shiyong Lan; Wenwu Wang; Yixin Qiao; Yao Li; Guonan Deng
>
> **摘要:** Gaze estimation, which predicts gaze direction, commonly faces the challenge of interference from complex gaze-irrelevant information in face images. In this work, we propose DMAGaze, a novel gaze estimation framework that exploits information from facial images in three aspects: gaze-relevant global features (disentangled from facial image), local eye features (extracted from cropped eye patch), and head pose estimation features, to improve overall performance. Firstly, we design a new continuous mask-based Disentangler to accurately disentangle gaze-relevant and gaze-irrelevant information in facial images by achieving the dual-branch disentanglement goal through separately reconstructing the eye and non-eye regions. Furthermore, we introduce a new cascaded attention module named Multi-Scale Global Local Attention Module (MS-GLAM). Through a customized cascaded attention structure, it effectively focuses on global and local information at multiple scales, further enhancing the information from the Disentangler. Finally, the global gaze-relevant features disentangled by the upper face branch, combined with head pose and local eye features, are passed through the detection head for high-precision gaze estimation. Our proposed DMAGaze has been extensively validated on two mainstream public datasets, achieving state-of-the-art performance.
>
---
#### [replaced 102] HazyDet: Open-Source Benchmark for Drone-View Object Detection with Depth-Cues in Hazy Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.19833v2](http://arxiv.org/pdf/2409.19833v2)**

> **作者:** Changfeng Feng; Zhenyuan Chen; Xiang Li; Chunping Wang; Jian Yang; Ming-Ming Cheng; Yimian Dai; Qiang Fu
>
> **备注:** We have updated our method, resulting in a large improvement in detection performance
>
> **摘要:** Object detection from aerial platforms under adverse atmospheric conditions, particularly haze, is paramount for robust drone autonomy. Yet, this domain remains largely underexplored, primarily hindered by the absence of specialized benchmarks. To bridge this gap, we present \textit{HazyDet}, the first, large-scale benchmark specifically designed for drone-view object detection in hazy conditions. Comprising 383,000 real-world instances derived from both naturally hazy captures and synthetically hazed scenes augmented from clear images, HazyDet provides a challenging and realistic testbed for advancing detection algorithms. To address the severe visual degradation induced by haze, we propose the Depth-Conditioned Detector (DeCoDet), a novel architecture that integrates a Depth-Conditioned Kernel to dynamically modulate feature representations based on depth cues. The practical efficacy and robustness of DeCoDet are further enhanced by its training with a Progressive Domain Fine-Tuning (PDFT) strategy to navigate synthetic-to-real domain shifts, and a Scale-Invariant Refurbishment Loss (SIRLoss) to ensure resilient learning from potentially noisy depth annotations. Comprehensive empirical validation on HazyDet substantiates the superiority of our unified DeCoDet framework, which achieves state-of-the-art performance, surpassing the closest competitor by a notable +1.5\% mAP on challenging real-world hazy test scenarios. Our dataset and toolkit are available at https://github.com/GrokCV/HazyDet.
>
---
#### [replaced 103] Open the Eyes of MPNN: Vision Enhances MPNN in Link Prediction
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.08266v2](http://arxiv.org/pdf/2505.08266v2)**

> **作者:** Yanbin Wei; Xuehao Wang; Zhan Zhuang; Yang Chen; Shuhao Chen; Yulong Zhang; Yu Zhang; James Kwok
>
> **备注:** ICML 2025
>
> **摘要:** Message-passing graph neural networks (MPNNs) and structural features (SFs) are cornerstones for the link prediction task. However, as a common and intuitive mode of understanding, the potential of visual perception has been overlooked in the MPNN community. For the first time, we equip MPNNs with vision structural awareness by proposing an effective framework called Graph Vision Network (GVN), along with a more efficient variant (E-GVN). Extensive empirical results demonstrate that with the proposed frameworks, GVN consistently benefits from the vision enhancement across seven link prediction datasets, including challenging large-scale graphs. Such improvements are compatible with existing state-of-the-art (SOTA) methods and GVNs achieve new SOTA results, thereby underscoring a promising novel direction for link prediction.
>
---
#### [replaced 104] Adaptive Rank, Reduced Forgetting: Knowledge Retention in Continual Learning Vision-Language Models with Dynamic Rank-Selective LoRA
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01004v5](http://arxiv.org/pdf/2412.01004v5)**

> **作者:** Haodong Lu; Chongyang Zhao; Jason Xue; Lina Yao; Kristen Moore; Dong Gong
>
> **备注:** Preprint
>
> **摘要:** Continual learning (CL) aims to accumulate knowledge from sequential data and task streams. Leveraging their strong generalization and flexibility, pre-trained vision-language embedding models such as CLIP (Contrastive Language-Image Pre-training) have been widely adopted and validated in CL. In addition to learning new knowledge, we investigate whether the pre-trained knowledge in CLIP, can be retained or even enhanced, in CL, while incorporating new knowledge from a data stream. Existing CL methods primarily focus on continual downstream adaptation using components isolated from the pre-trained model (PTM), increasing inference complexity and limiting improvements to the PTM itself; some also retain knowledge by relying on additional reference data, resulting in high training costs. To address these limitations, we propose a universal and efficient CL approach for CLIP based on Dynamic Rank-Selective LoRA (CoDyRA), which directly improves the PTMs while preserving the existing knowledge from both pre-training and CL. By analyzing how LoRA rank and placement affect learning and forgetting in CL, we design CoDyRA that adaptively performs rank-minimized parameter updates in different modules, based on their importance to the current data. This ensures a balance between knowledge acquisition (plasticity) and forgetting mitigation (stability). Our method operates without explicit domain or distribution prediction and does not rely on reference data, enabling seamless task integration while maintaining pre-trained capabilities. Moreover, CoDyRA preserves the original model architecture and deployment pipeline, introducing no additional inference overhead. Extensive experiments show that our approach enhances representations for new downstream data while retaining pre-trained knowledge, achieving state-of-the-art results.
>
---
#### [replaced 105] VSA: Faster Video Diffusion with Trainable Sparse Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13389v3](http://arxiv.org/pdf/2505.13389v3)**

> **作者:** Peiyuan Zhang; Haofeng Huang; Yongqi Chen; Will Lin; Zhengzhong Liu; Ion Stoica; Eric Xing; Hao Zhang
>
> **摘要:** Scaling video diffusion transformers (DiTs) is limited by their quadratic 3D attention, even though most of the attention mass concentrates on a small subset of positions. We turn this observation into VSA, a trainable, hardware-efficient sparse attention that replaces full attention at \emph{both} training and inference. In VSA, a lightweight coarse stage pools tokens into tiles and identifies high-weight \emph{critical tokens}; a fine stage computes token-level attention only inside those tiles subjecting to block computing layout to ensure hard efficiency. This leads to a single differentiable kernel that trains end-to-end, requires no post-hoc profiling, and sustains 85\% of FlashAttention3 MFU. We perform a large sweep of ablation studies and scaling-law experiments by pretraining DiTs from 60M to 1.4B parameters. VSA reaches a Pareto point that cuts training FLOPS by 2.53$\times$ with no drop in diffusion loss. Retrofitting the open-source Wan-2.1 model speeds up attention time by 6$\times$ and lowers end-to-end generation time from 31s to 18s with comparable quality. These results establish trainable sparse attention as a practical alternative to full attention and a key enabler for further scaling of video diffusion models. Code will be available at https://github.com/hao-ai-lab/FastVideo.
>
---
#### [replaced 106] Multimodal 3D Reasoning Segmentation with Complex Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.13927v3](http://arxiv.org/pdf/2411.13927v3)**

> **作者:** Xueying Jiang; Lewei Lu; Ling Shao; Shijian Lu
>
> **摘要:** The recent development in multimodal learning has greatly advanced the research in 3D scene understanding in various real-world tasks such as embodied AI. However, most existing studies are facing two common challenges: 1) they are short of reasoning ability for interaction and interpretation of human intentions and 2) they focus on scenarios with single-category objects and over-simplified textual descriptions and neglect multi-object scenarios with complicated spatial relations among objects. We address the above challenges by proposing a 3D reasoning segmentation task for reasoning segmentation with multiple objects in scenes. The task allows producing 3D segmentation masks and detailed textual explanations as enriched by 3D spatial relations among objects. To this end, we create ReasonSeg3D, a large-scale and high-quality benchmark that integrates 3D segmentation masks and 3D spatial relations with generated question-answer pairs. In addition, we design MORE3D, a novel 3D reasoning network that works with queries of multiple objects and is tailored for 3D scene understanding. MORE3D learns detailed explanations on 3D relations and employs them to capture spatial information of objects and reason textual outputs. Extensive experiments show that MORE3D excels in reasoning and segmenting complex multi-object 3D scenes. In addition, the created ReasonSeg3D offers a valuable platform for future exploration of 3D reasoning segmentation. The data and code will be released.
>
---
#### [replaced 107] Mirror: Multimodal Cognitive Reframing Therapy for Rolling with Resistance
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.13211v2](http://arxiv.org/pdf/2504.13211v2)**

> **作者:** Subin Kim; Hoonrae Kim; Jihyun Lee; Yejin Jeon; Gary Geunbae Lee
>
> **摘要:** Recent studies have explored the use of large language models (LLMs) in psychotherapy; however, text-based cognitive behavioral therapy (CBT) models often struggle with client resistance, which can weaken therapeutic alliance. To address this, we propose a multimodal approach that incorporates nonverbal cues, which allows the AI therapist to better align its responses with the client's negative emotional state. Specifically, we introduce a new synthetic dataset, Mirror (Multimodal Interactive Rolling with Resistance), which is a novel synthetic dataset that pairs each client's statements with corresponding facial images. Using this dataset, we train baseline vision language models (VLMs) so that they can analyze facial cues, infer emotions, and generate empathetic responses to effectively manage client resistance. These models are then evaluated in terms of both their counseling skills as a therapist, and the strength of therapeutic alliance in the presence of client resistance. Our results demonstrate that Mirror significantly enhances the AI therapist's ability to handle resistance, which outperforms existing text-based CBT approaches. Human expert evaluations further confirm the effectiveness of our approach in managing client resistance and fostering therapeutic alliance.
>
---
#### [replaced 108] Inductive Gradient Adjustment For Spectral Bias In Implicit Neural Representations
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.13271v2](http://arxiv.org/pdf/2410.13271v2)**

> **作者:** Kexuan Shi; Hai Chen; Leheng Zhang; Shuhang Gu
>
> **备注:** Accepted to ICML 2025. Code available at https://github.com/LabShuHangGU/IGA-INR
>
> **摘要:** Implicit Neural Representations (INRs), as a versatile representation paradigm, have achieved success in various computer vision tasks. Due to the spectral bias of the vanilla multi-layer perceptrons (MLPs), existing methods focus on designing MLPs with sophisticated architectures or repurposing training techniques for highly accurate INRs. In this paper, we delve into the linear dynamics model of MLPs and theoretically identify the empirical Neural Tangent Kernel (eNTK) matrix as a reliable link between spectral bias and training dynamics. Based on this insight, we propose a practical Inductive Gradient Adjustment (IGA) method, which could purposefully improve the spectral bias via inductive generalization of eNTK-based gradient transformation matrix. Theoretical and empirical analyses validate impacts of IGA on spectral bias. Further, we evaluate our method on different INRs tasks with various INR architectures and compare to existing training techniques. The superior and consistent improvements clearly validate the advantage of our IGA. Armed with our gradient adjustment method, better INRs with more enhanced texture details and sharpened edges can be learned from data by tailored impacts on spectral bias.
>
---
#### [replaced 109] PMQ-VE: Progressive Multi-Frame Quantization for Video Enhancement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12266v2](http://arxiv.org/pdf/2505.12266v2)**

> **作者:** ZhanFeng Feng; Long Peng; Xin Di; Yong Guo; Wenbo Li; Yulun Zhang; Renjing Pei; Yang Wang; Yang Cao; Zheng-Jun Zha
>
> **摘要:** Multi-frame video enhancement tasks aim to improve the spatial and temporal resolution and quality of video sequences by leveraging temporal information from multiple frames, which are widely used in streaming video processing, surveillance, and generation. Although numerous Transformer-based enhancement methods have achieved impressive performance, their computational and memory demands hinder deployment on edge devices. Quantization offers a practical solution by reducing the bit-width of weights and activations to improve efficiency. However, directly applying existing quantization methods to video enhancement tasks often leads to significant performance degradation and loss of fine details. This stems from two limitations: (a) inability to allocate varying representational capacity across frames, which results in suboptimal dynamic range adaptation; (b) over-reliance on full-precision teachers, which limits the learning of low-bit student models. To tackle these challenges, we propose a novel quantization method for video enhancement: Progressive Multi-Frame Quantization for Video Enhancement (PMQ-VE). This framework features a coarse-to-fine two-stage process: Backtracking-based Multi-Frame Quantization (BMFQ) and Progressive Multi-Teacher Distillation (PMTD). BMFQ utilizes a percentile-based initialization and iterative search with pruning and backtracking for robust clipping bounds. PMTD employs a progressive distillation strategy with both full-precision and multiple high-bit (INT) teachers to enhance low-bit models' capacity and quality. Extensive experiments demonstrate that our method outperforms existing approaches, achieving state-of-the-art performance across multiple tasks and benchmarks.The code will be made publicly available at: https://github.com/xiaoBIGfeng/PMQ-VE.
>
---
#### [replaced 110] What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.08279v4](http://arxiv.org/pdf/2502.08279v4)**

> **作者:** Dongqi Liu; Chenxi Whitehouse; Xi Yu; Louis Mahon; Rohit Saxena; Zheng Zhao; Yifu Qiu; Mirella Lapata; Vera Demberg
>
> **备注:** ACL 2025 Main & Long Conference Paper
>
> **摘要:** Transforming recorded videos into concise and accurate textual summaries is a growing challenge in multimodal learning. This paper introduces VISTA, a dataset specifically designed for video-to-text summarization in scientific domains. VISTA contains 18,599 recorded AI conference presentations paired with their corresponding paper abstracts. We benchmark the performance of state-of-the-art large models and apply a plan-based framework to better capture the structured nature of abstracts. Both human and automated evaluations confirm that explicit planning enhances summary quality and factual consistency. However, a considerable gap remains between models and human performance, highlighting the challenges of our dataset. This study aims to pave the way for future research on scientific video-to-text summarization.
>
---
#### [replaced 111] SPKLIP: Aligning Spike Video Streams with Natural Language
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12656v2](http://arxiv.org/pdf/2505.12656v2)**

> **作者:** Yongchang Gao; Meiling Jin; Zhaofei Yu; Tiejun Huang; Guozhang Chen
>
> **摘要:** Spike cameras offer unique sensing capabilities but their sparse, asynchronous output challenges semantic understanding, especially for Spike Video-Language Alignment (Spike-VLA) where models like CLIP underperform due to modality mismatch. We introduce SPKLIP, the first architecture specifically for Spike-VLA. SPKLIP employs a hierarchical spike feature extractor that adaptively models multi-scale temporal dynamics in event streams, and uses spike-text contrastive learning to directly align spike video with language, enabling effective few-shot learning. A full-spiking visual encoder variant, integrating SNN components into our pipeline, demonstrates enhanced energy efficiency. Experiments show state-of-the-art performance on benchmark spike datasets and strong few-shot generalization on a newly contributed real-world dataset. SPKLIP's energy efficiency highlights its potential for neuromorphic deployment, advancing event-based multimodal research. The source code and dataset are available at [link removed for anonymity].
>
---
#### [replaced 112] WaveGuard: Robust Deepfake Detection and Source Tracing via Dual-Tree Complex Wavelet and Graph Neural Networks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08614v3](http://arxiv.org/pdf/2505.08614v3)**

> **作者:** Ziyuan He; Zhiqing Guo; Liejun Wang; Gaobo Yang; Yunfeng Diao; Dan Ma
>
> **备注:** 12 pages, 6 figures, 5 tables
>
> **摘要:** Deepfake technology poses increasing risks such as privacy invasion and identity theft. To address these threats, we propose WaveGuard, a proactive watermarking framework that enhances robustness and imperceptibility via frequency-domain embedding and graph-based structural consistency. Specifically, we embed watermarks into high-frequency sub-bands using Dual-Tree Complex Wavelet Transform (DT-CWT) and employ a Structural Consistency Graph Neural Network (SC-GNN) to preserve visual quality. We also design an attention module to refine embedding precision. Experimental results on face swap and reenactment tasks demonstrate that WaveGuard outperforms state-of-the-art methods in both robustness and visual quality. Code is available at https://github.com/vpsg-research/WaveGuard.
>
---
#### [replaced 113] Dynamic Snake Upsampling Operater and Boundary-Skeleton Weighted Loss for Tubular Structure Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08525v2](http://arxiv.org/pdf/2505.08525v2)**

> **作者:** Yiqi Chen; Ganghai Huang; Sheng Zhang; Jianglin Dai
>
> **摘要:** Accurate segmentation of tubular topological structures (e.g., fissures and vasculature) is critical in various fields to guarantee dependable downstream quantitative analysis and modeling. However, in dense prediction tasks such as semantic segmentation and super-resolution, conventional upsampling operators cannot accommodate the slenderness of tubular structures and the curvature of morphology. This paper introduces a dynamic snake upsampling operators and a boundary-skeleton weighted loss tailored for topological tubular structures. Specifically, we design a snake upsampling operators based on an adaptive sampling domain, which dynamically adjusts the sampling stride according to the feature map and selects a set of subpixel sampling points along the serpentine path, enabling more accurate subpixel-level feature recovery for tubular structures. Meanwhile, we propose a skeleton-to-boundary increasing weighted loss that trades off main body and boundary weight allocation based on mask class ratio and distance field, preserving main body overlap while enhancing focus on target topological continuity and boundary alignment precision. Experiments across various domain datasets and backbone networks show that this plug-and-play dynamic snake upsampling operator and boundary-skeleton weighted loss boost both pixel-wise segmentation accuracy and topological consistency of results.
>
---
#### [replaced 114] Improving Compositional Generation with Diffusion Models Using Lift Scores
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13740v2](http://arxiv.org/pdf/2505.13740v2)**

> **作者:** Chenning Yu; Sicun Gao
>
> **摘要:** We introduce a novel resampling criterion using lift scores, for improving compositional generation in diffusion models. By leveraging the lift scores, we evaluate whether generated samples align with each single condition and then compose the results to determine whether the composed prompt is satisfied. Our key insight is that lift scores can be efficiently approximated using only the original diffusion model, requiring no additional training or external modules. We develop an optimized variant that achieves relatively lower computational overhead during inference while maintaining effectiveness. Through extensive experiments, we demonstrate that lift scores significantly improved the condition alignment for compositional generation across 2D synthetic data, CLEVR position tasks, and text-to-image synthesis. Our code is available at http://rainorangelemon.github.io/complift.
>
---
#### [replaced 115] Probabilistic Interactive 3D Segmentation with Hierarchical Neural Processes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01726v2](http://arxiv.org/pdf/2505.01726v2)**

> **作者:** Jie Liu; Pan Zhou; Zehao Xiao; Jiayi Shen; Wenzhe Yin; Jan-Jakob Sonke; Efstratios Gavves
>
> **备注:** ICML 2025 Proceedings
>
> **摘要:** Interactive 3D segmentation has emerged as a promising solution for generating accurate object masks in complex 3D scenes by incorporating user-provided clicks. However, two critical challenges remain underexplored: (1) effectively generalizing from sparse user clicks to produce accurate segmentation, and (2) quantifying predictive uncertainty to help users identify unreliable regions. In this work, we propose NPISeg3D, a novel probabilistic framework that builds upon Neural Processes (NPs) to address these challenges. Specifically, NPISeg3D introduces a hierarchical latent variable structure with scene-specific and object-specific latent variables to enhance few-shot generalization by capturing both global context and object-specific characteristics. Additionally, we design a probabilistic prototype modulator that adaptively modulates click prototypes with object-specific latent variables, improving the model's ability to capture object-aware context and quantify predictive uncertainty. Experiments on four 3D point cloud datasets demonstrate that NPISeg3D achieves superior segmentation performance with fewer clicks while providing reliable uncertainty estimations.
>
---
#### [replaced 116] Reenact Anything: Semantic Video Motion Transfer Using Motion-Textual Inversion
- **分类: cs.CV; cs.GR; cs.LG; I.3.3; I.4**

- **链接: [http://arxiv.org/pdf/2408.00458v2](http://arxiv.org/pdf/2408.00458v2)**

> **作者:** Manuel Kansy; Jacek Naruniec; Christopher Schroers; Markus Gross; Romann M. Weber
>
> **备注:** Added more evaluation and analyses since first version. Accepted to SIGGRAPH 2025 (Conference Track). Project page: https://mkansy.github.io/reenact-anything/
>
> **摘要:** Recent years have seen a tremendous improvement in the quality of video generation and editing approaches. While several techniques focus on editing appearance, few address motion. Current approaches using text, trajectories, or bounding boxes are limited to simple motions, so we specify motions with a single motion reference video instead. We further propose to use a pre-trained image-to-video model rather than a text-to-video model. This approach allows us to preserve the exact appearance and position of a target object or scene and helps disentangle appearance from motion. Our method, called motion-textual inversion, leverages our observation that image-to-video models extract appearance mainly from the (latent) image input, while the text/image embedding injected via cross-attention predominantly controls motion. We thus represent motion using text/image embedding tokens. By operating on an inflated motion-text embedding containing multiple text/image embedding tokens per frame, we achieve a high temporal motion granularity. Once optimized on the motion reference video, this embedding can be applied to various target images to generate videos with semantically similar motions. Our approach does not require spatial alignment between the motion reference video and target image, generalizes across various domains, and can be applied to various tasks such as full-body and face reenactment, as well as controlling the motion of inanimate objects and the camera. We empirically demonstrate the effectiveness of our method in the semantic video motion transfer task, significantly outperforming existing methods in this context. Project website: https://mkansy.github.io/reenact-anything/
>
---
#### [replaced 117] X-GRM: Large Gaussian Reconstruction Model for Sparse-view X-rays to Computed Tomography
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15235v2](http://arxiv.org/pdf/2505.15235v2)**

> **作者:** Yifan Liu; Wuyang Li; Weihao Yu; Chenxin Li; Alexandre Alahi; Max Meng; Yixuan Yuan
>
> **摘要:** Computed Tomography serves as an indispensable tool in clinical workflows, providing non-invasive visualization of internal anatomical structures. Existing CT reconstruction works are limited to small-capacity model architecture and inflexible volume representation. In this work, we present X-GRM (X-ray Gaussian Reconstruction Model), a large feedforward model for reconstructing 3D CT volumes from sparse-view 2D X-ray projections. X-GRM employs a scalable transformer-based architecture to encode sparse-view X-ray inputs, where tokens from different views are integrated efficiently. Then, these tokens are decoded into a novel volume representation, named Voxel-based Gaussian Splatting (VoxGS), which enables efficient CT volume extraction and differentiable X-ray rendering. This combination of a high-capacity model and flexible volume representation, empowers our model to produce high-quality reconstructions from various testing inputs, including in-domain and out-domain X-ray projections. Our codes are available at: https://github.com/CUHK-AIM-Group/X-GRM.
>
---
#### [replaced 118] ManipLVM-R1: Reinforcement Learning for Reasoning in Embodied Manipulation with Large Vision-Language Models
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16517v2](http://arxiv.org/pdf/2505.16517v2)**

> **作者:** Zirui Song; Guangxian Ouyang; Mingzhe Li; Yuheng Ji; Chenxi Wang; Zixiang Xu; Zeyu Zhang; Xiaoqing Zhang; Qian Jiang; Zhenhao Chen; Zhongzhi Li; Rui Yan; Xiuying Chen
>
> **备注:** 14pages
>
> **摘要:** Large Vision-Language Models (LVLMs) have recently advanced robotic manipulation by leveraging vision for scene perception and language for instruction following. However, existing methods rely heavily on costly human-annotated training datasets, which limits their generalization and causes them to struggle in out-of-domain (OOD) scenarios, reducing real-world adaptability. To address these challenges, we propose ManipLVM-R1, a novel reinforcement learning framework that replaces traditional supervision with Reinforcement Learning using Verifiable Rewards (RLVR). By directly optimizing for task-aligned outcomes, our method enhances generalization and physical reasoning while removing the dependence on costly annotations. Specifically, we design two rule-based reward functions targeting key robotic manipulation subtasks: an Affordance Perception Reward to enhance localization of interaction regions, and a Trajectory Match Reward to ensure the physical plausibility of action paths. These rewards provide immediate feedback and impose spatial-logical constraints, encouraging the model to go beyond shallow pattern matching and instead learn deeper, more systematic reasoning about physical interactions.
>
---
#### [replaced 119] Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15966v2](http://arxiv.org/pdf/2505.15966v2)**

> **作者:** Alex Su; Haozhe Wang; Weiming Ren; Fangzhen Lin; Wenhu Chen
>
> **备注:** Project Page: https://tiger-ai-lab.github.io/Pixel-Reasoner/, Hands-on Demo: https://huggingface.co/spaces/TIGER-Lab/Pixel-Reasoner
>
> **摘要:** Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks. Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework.
>
---
#### [replaced 120] Self-Guidance: Boosting Flow and Diffusion Generation on Their Own
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05827v3](http://arxiv.org/pdf/2412.05827v3)**

> **作者:** Tiancheng Li; Weijian Luo; Zhiyang Chen; Liyuan Ma; Guo-Jun Qi
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Proper guidance strategies are essential to achieve high-quality generation results without retraining diffusion and flow-based text-to-image models. Existing guidance either requires specific training or strong inductive biases of diffusion model networks, potentially limiting their applications. Motivated by the observation that artifact outliers can be detected by a significant decline in the density from a noisier to a cleaner noise level, we propose Self-Guidance (SG), which improves the image quality by suppressing the generation of low-quality samples. SG only relies on the sampling probabilities of its own diffusion model at different noise levels with no need of any guidance-specific training. This makes it flexible to be used in a plug-and-play manner with other sampling algorithms. We also introduce a more efficient approximation of SG, named SG-prev, which reuses the output from the immediately previous diffusion step to avoid doubling sampling time. We conduct experiments on text-to-image and text-to-video generation with different architectures, including UNet and transformer models. With open-sourced diffusion models such as Stable Diffusion 3.5 and FLUX, SG exceeds existing algorithms on multiple metrics, including both FID and Human Preference Score. SG-prev also achieves strong results over both the baseline and the SG with only one forward pass. Moreover, we find that SG and SG-prev both have a surprisingly positive effect on the generation of physiologically correct human body structures such as hands, faces, and arms, showing their ability of eliminating human body artifacts with minimal efforts. We will release our code along with this paper.
>
---
#### [replaced 121] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?
- **分类: cs.CL; cs.AI; cs.CR; cs.CV; cs.SE**

- **链接: [http://arxiv.org/pdf/2505.10924v2](http://arxiv.org/pdf/2505.10924v2)**

> **作者:** Ada Chen; Yongjiang Wu; Junyuan Zhang; Jingyu Xiao; Shu Yang; Jen-tse Huang; Kun Wang; Wenxuan Wang; Shuai Wang
>
> **摘要:** Recently, AI-driven interactions with computing devices have advanced from basic prototype tools to sophisticated, LLM-based systems that emulate human-like operations in graphical user interfaces. We are now witnessing the emergence of \emph{Computer-Using Agents} (CUAs), capable of autonomously performing tasks such as navigating desktop applications, web pages, and mobile apps. However, as these agents grow in capability, they also introduce novel safety and security risks. Vulnerabilities in LLM-driven reasoning, with the added complexity of integrating multiple software components and multimodal inputs, further complicate the security landscape. In this paper, we present a systematization of knowledge on the safety and security threats of CUAs. We conduct a comprehensive literature review and distill our findings along four research objectives: \textit{\textbf{(i)}} define the CUA that suits safety analysis; \textit{\textbf{(ii)} } categorize current safety threats among CUAs; \textit{\textbf{(iii)}} propose a comprehensive taxonomy of existing defensive strategies; \textit{\textbf{(iv)}} summarize prevailing benchmarks, datasets, and evaluation metrics used to assess the safety and performance of CUAs. Building on these insights, our work provides future researchers with a structured foundation for exploring unexplored vulnerabilities and offers practitioners actionable guidance in designing and deploying secure Computer-Using Agents.
>
---
#### [replaced 122] Robust multi-coil MRI reconstruction via self-supervised denoising
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.12919v4](http://arxiv.org/pdf/2411.12919v4)**

> **作者:** Asad Aali; Marius Arvinte; Sidharth Kumar; Yamin I. Arefeen; Jonathan I. Tamir
>
> **摘要:** We study the effect of incorporating self-supervised denoising as a pre-processing step for training deep learning (DL) based reconstruction methods on data corrupted by Gaussian noise. K-space data employed for training are typically multi-coil and inherently noisy. Although DL-based reconstruction methods trained on fully sampled data can enable high reconstruction quality, obtaining large, noise-free datasets is impractical. We leverage Generalized Stein's Unbiased Risk Estimate (GSURE) for denoising. We evaluate two DL-based reconstruction methods: Diffusion Probabilistic Models (DPMs) and Model-Based Deep Learning (MoDL). We evaluate the impact of denoising on the performance of these DL-based methods in solving accelerated multi-coil magnetic resonance imaging (MRI) reconstruction. The experiments were carried out on T2-weighted brain and fat-suppressed proton-density knee scans. We observed that self-supervised denoising enhances the quality and efficiency of MRI reconstructions across various scenarios. Specifically, employing denoised images rather than noisy counterparts when training DL networks results in lower normalized root mean squared error (NRMSE), higher structural similarity index measure (SSIM) and peak signal-to-noise ratio (PSNR) across different SNR levels, including 32dB, 22dB, and 12dB for T2-weighted brain data, and 24dB, 14dB, and 4dB for fat-suppressed knee data. Overall, we showed that denoising is an essential pre-processing technique capable of improving the efficacy of DL-based MRI reconstruction methods under diverse conditions. By refining the quality of input data, denoising enables training more effective DL networks, potentially bypassing the need for noise-free reference MRI scans.
>
---
#### [replaced 123] ASGrasp: Generalizable Transparent Object Reconstruction and 6-DoF Grasp Detection from RGB-D Active Stereo Camera
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.05648v2](http://arxiv.org/pdf/2405.05648v2)**

> **作者:** Jun Shi; Yong A; Yixiang Jin; Dingzhe Li; Haoyu Niu; Zhezhu Jin; He Wang
>
> **备注:** IEEE International Conference on Robotics and Automation (ICRA), 2024
>
> **摘要:** In this paper, we tackle the problem of grasping transparent and specular objects. This issue holds importance, yet it remains unsolved within the field of robotics due to failure of recover their accurate geometry by depth cameras. For the first time, we propose ASGrasp, a 6-DoF grasp detection network that uses an RGB-D active stereo camera. ASGrasp utilizes a two-layer learning-based stereo network for the purpose of transparent object reconstruction, enabling material-agnostic object grasping in cluttered environments. In contrast to existing RGB-D based grasp detection methods, which heavily depend on depth restoration networks and the quality of depth maps generated by depth cameras, our system distinguishes itself by its ability to directly utilize raw IR and RGB images for transparent object geometry reconstruction. We create an extensive synthetic dataset through domain randomization, which is based on GraspNet-1Billion. Our experiments demonstrate that ASGrasp can achieve over 90% success rate for generalizable transparent object grasping in both simulation and the real via seamless sim-to-real transfer. Our method significantly outperforms SOTA networks and even surpasses the performance upper bound set by perfect visible point cloud inputs.Project page: https://pku-epic.github.io/ASGrasp
>
---
#### [replaced 124] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.19385v3](http://arxiv.org/pdf/2503.19385v3)**

> **作者:** Jaihoon Kim; Taehoon Yoon; Jisung Hwang; Minhyuk Sung
>
> **备注:** Project page: https://flow-inference-time-scaling.github.io/
>
> **摘要:** We propose an inference-time scaling approach for pretrained flow models. Recently, inference-time scaling has gained significant attention in LLMs and diffusion models, improving sample quality or better aligning outputs with user preferences by leveraging additional computation. For diffusion models, particle sampling has allowed more efficient scaling due to the stochasticity at intermediate denoising steps. On the contrary, while flow models have gained popularity as an alternative to diffusion models--offering faster generation and high-quality outputs in state-of-the-art image and video generative models--efficient inference-time scaling methods used for diffusion models cannot be directly applied due to their deterministic generative process. To enable efficient inference-time scaling for flow models, we propose three key ideas: 1) SDE-based generation, enabling particle sampling in flow models, 2) Interpolant conversion, broadening the search space and enhancing sample diversity, and 3) Rollover Budget Forcing (RBF), an adaptive allocation of computational resources across timesteps to maximize budget utilization. Our experiments show that SDE-based generation, particularly variance-preserving (VP) interpolant-based generation, improves the performance of particle sampling methods for inference-time scaling in flow models. Additionally, we demonstrate that RBF with VP-SDE achieves the best performance, outperforming all previous inference-time scaling approaches.
>
---
#### [replaced 125] Inverse Problem Sampling in Latent Space Using Sequential Monte Carlo
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.05908v2](http://arxiv.org/pdf/2502.05908v2)**

> **作者:** Idan Achituve; Hai Victor Habi; Amir Rosenfeld; Arnon Netzer; Idit Diamant; Ethan Fetaya
>
> **摘要:** In image processing, solving inverse problems is the task of finding plausible reconstructions of an image that was corrupted by some (usually known) degradation operator. Commonly, this process is done using a generative image model that can guide the reconstruction towards solutions that appear natural. The success of diffusion models over the last few years has made them a leading candidate for this task. However, the sequential nature of diffusion models makes this conditional sampling process challenging. Furthermore, since diffusion models are often defined in the latent space of an autoencoder, the encoder-decoder transformations introduce additional difficulties. To address these challenges, we suggest a novel sampling method based on sequential Monte Carlo (SMC) in the latent space of diffusion models. We name our method LD-SMC. We define a generative model for the data using additional auxiliary observations and perform posterior inference with SMC sampling based on a backward diffusion process. Empirical evaluations on ImageNet and FFHQ show the benefits of LD-SMC over competing methods in various inverse problem tasks and especially in challenging inpainting tasks.
>
---
#### [replaced 126] Human-Aligned Image Models Improve Visual Decoding from the Brain
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.03081v2](http://arxiv.org/pdf/2502.03081v2)**

> **作者:** Nona Rajabi; Antônio H. Ribeiro; Miguel Vasco; Farzaneh Taleb; Mårten Björkman; Danica Kragic
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Decoding visual images from brain activity has significant potential for advancing brain-computer interaction and enhancing the understanding of human perception. Recent approaches align the representation spaces of images and brain activity to enable visual decoding. In this paper, we introduce the use of human-aligned image encoders to map brain signals to images. We hypothesize that these models more effectively capture perceptual attributes associated with the rapid visual stimuli presentations commonly used in visual brain data recording experiments. Our empirical results support this hypothesis, demonstrating that this simple modification improves image retrieval accuracy by up to 21% compared to state-of-the-art methods. Comprehensive experiments confirm consistent performance improvements across diverse EEG architectures, image encoders, alignment methods, participants, and brain imaging modalities
>
---
#### [replaced 127] One Step Diffusion via Shortcut Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.12557v2](http://arxiv.org/pdf/2410.12557v2)**

> **作者:** Kevin Frans; Danijar Hafner; Sergey Levine; Pieter Abbeel
>
> **摘要:** Diffusion models and flow-matching models have enabled generating diverse and realistic images by learning to transfer noise to data. However, sampling from these models involves iterative denoising over many neural network passes, making generation slow and expensive. Previous approaches for speeding up sampling require complex training regimes, such as multiple training phases, multiple networks, or fragile scheduling. We introduce shortcut models, a family of generative models that use a single network and training phase to produce high-quality samples in a single or multiple sampling steps. Shortcut models condition the network not only on the current noise level but also on the desired step size, allowing the model to skip ahead in the generation process. Across a wide range of sampling step budgets, shortcut models consistently produce higher quality samples than previous approaches, such as consistency models and reflow. Compared to distillation, shortcut models reduce complexity to a single network and training phase and additionally allow varying step budgets at inference time.
>
---
#### [replaced 128] Emerging Properties in Unified Multimodal Pretraining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14683v2](http://arxiv.org/pdf/2505.14683v2)**

> **作者:** Chaorui Deng; Deyao Zhu; Kunchang Li; Chenhui Gou; Feng Li; Zeyu Wang; Shu Zhong; Weihao Yu; Xiaonan Nie; Ziang Song; Guang Shi; Haoqi Fan
>
> **备注:** 37 pages, 17 figures
>
> **摘要:** Unifying multimodal understanding and generation has shown impressive capabilities in cutting-edge proprietary systems. In this work, we introduce BAGEL, an open-source foundational model that natively supports multimodal understanding and generation. BAGEL is a unified, decoder-only model pretrained on trillions of tokens curated from large-scale interleaved text, image, video, and web data. When scaled with such diverse multimodal interleaved data, BAGEL exhibits emerging capabilities in complex multimodal reasoning. As a result, it significantly outperforms open-source unified models in both multimodal generation and understanding across standard benchmarks, while exhibiting advanced multimodal reasoning abilities such as free-form image manipulation, future frame prediction, 3D manipulation, and world navigation. In the hope of facilitating further opportunities for multimodal research, we share the key findings, pretraining details, data creation protocal, and release our code and checkpoints to the community. The project page is at https://bagel-ai.org/
>
---
#### [replaced 129] GUARD: Role-playing to Generate Natural-language Jailbreakings to Test Guideline Adherence of Large Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2402.03299v5](http://arxiv.org/pdf/2402.03299v5)**

> **作者:** Haibo Jin; Ruoxi Chen; Peiyan Zhang; Andy Zhou; Yang Zhang; Haohan Wang
>
> **备注:** 28 papges
>
> **摘要:** The discovery of "jailbreaks" to bypass safety filters of Large Language Models (LLMs) and harmful responses have encouraged the community to implement safety measures. One major safety measure is to proactively test the LLMs with jailbreaks prior to the release. Therefore, such testing will require a method that can generate jailbreaks massively and efficiently. In this paper, we follow a novel yet intuitive strategy to generate jailbreaks in the style of the human generation. We propose a role-playing system that assigns four different roles to the user LLMs to collaborate on new jailbreaks. Furthermore, we collect existing jailbreaks and split them into different independent characteristics using clustering frequency and semantic patterns sentence by sentence. We organize these characteristics into a knowledge graph, making them more accessible and easier to retrieve. Our system of different roles will leverage this knowledge graph to generate new jailbreaks, which have proved effective in inducing LLMs to generate unethical or guideline-violating responses. In addition, we also pioneer a setting in our system that will automatically follow the government-issued guidelines to generate jailbreaks to test whether LLMs follow the guidelines accordingly. We refer to our system as GUARD (Guideline Upholding through Adaptive Role-play Diagnostics). We have empirically validated the effectiveness of GUARD on three cutting-edge open-sourced LLMs (Vicuna-13B, LongChat-7B, and Llama-2-7B), as well as a widely-utilized commercial LLM (ChatGPT). Moreover, our work extends to the realm of vision language models (MiniGPT-v2 and Gemini Vision Pro), showcasing GUARD's versatility and contributing valuable insights for the development of safer, more reliable LLM-based applications across diverse modalities.
>
---
#### [replaced 130] OneDiff: A Generalist Model for Image Difference Captioning
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2407.05645v4](http://arxiv.org/pdf/2407.05645v4)**

> **作者:** Erdong Hu; Longteng Guo; Tongtian Yue; Zijia Zhao; Shuning Xue; Jing Liu
>
> **摘要:** In computer vision, Image Difference Captioning (IDC) is crucial for accurately describing variations between closely related images. Traditional IDC methods often rely on specialist models, which restrict their applicability across varied contexts. This paper introduces the OneDiff model, a novel generalist approach that utilizes a robust vision-language model architecture, integrating a siamese image encoder with a Visual Delta Module. This innovative configuration allows for the precise detection and articulation of fine-grained differences between image pairs. OneDiff is trained through a dual-phase strategy, encompassing Coupled Sample Training and multi-task learning across a diverse array of data types, supported by our newly developed DiffCap Dataset. This dataset merges real-world and synthetic data, enhancing the training process and bolstering the model's robustness. Extensive testing on diverse IDC benchmarks, such as Spot-the-Diff, Image-Editing-Request, and Birds-to-Words, shows that OneDiff consistently outperforms existing state-of-the-art models in accuracy and adaptability, achieving improvements of up to 97% CIDEr points in average. By setting a new benchmark in IDC, OneDiff paves the way for more versatile and effective applications in detecting and describing visual differences. The code, models, and data will be made publicly available.
>
---
#### [replaced 131] Efficient Training-Free High-Resolution Synthesis with Energy Rectification in Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.02537v3](http://arxiv.org/pdf/2503.02537v3)**

> **作者:** Zhen Yang; Guibao Shen; Minyang Li; Liang Hou; Mushui Liu; Luozhou Wang; Xin Tao; Pengfei Wan; Di Zhang; Ying-Cong Chen
>
> **备注:** Project Page: https://zhenyangcs.github.io/RectifiedHR-Diffusion/
>
> **摘要:** Diffusion models have achieved remarkable progress across various visual generation tasks. However, their performance significantly declines when generating content at resolutions higher than those used during training. Although numerous methods have been proposed to enable high-resolution generation, they all suffer from inefficiency. In this paper, we propose RectifiedHR, a straightforward and efficient solution for training-free high-resolution synthesis. Specifically, we propose a noise refresh strategy that unlocks the model's training-free high-resolution synthesis capability and improves efficiency. Additionally, we are the first to observe the phenomenon of energy decay, which may cause image blurriness during the high-resolution synthesis process. To address this issue, we introduce average latent energy analysis and find that tuning the classifier-free guidance hyperparameter can significantly improve generation performance. Our method is entirely training-free and demonstrates efficient performance. Furthermore, we show that RectifiedHR is compatible with various diffusion model techniques, enabling advanced features such as image editing, customized generation, and video synthesis. Extensive comparisons with numerous baseline methods validate the superior effectiveness and efficiency of RectifiedHR.
>
---
#### [replaced 132] Cross-Modal Bidirectional Interaction Model for Referring Remote Sensing Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.08613v2](http://arxiv.org/pdf/2410.08613v2)**

> **作者:** Zhe Dong; Yuzhe Sun; Tianzhu Liu; Wangmeng Zuo; Yanfeng Gu
>
> **摘要:** Given a natural language expression and a remote sensing image, the goal of referring remote sensing image segmentation (RRSIS) is to generate a pixel-level mask of the target object identified by the referring expression. In contrast to natural scenarios, expressions in RRSIS often involve complex geospatial relationships, with target objects of interest that vary significantly in scale and lack visual saliency, thereby increasing the difficulty of achieving precise segmentation. To address the aforementioned challenges, a novel RRSIS framework is proposed, termed the cross-modal bidirectional interaction model (CroBIM). Specifically, a context-aware prompt modulation (CAPM) module is designed to integrate spatial positional relationships and task-specific knowledge into the linguistic features, thereby enhancing the ability to capture the target object. Additionally, a language-guided feature aggregation (LGFA) module is introduced to integrate linguistic information into multi-scale visual features, incorporating an attention deficit compensation mechanism to enhance feature aggregation. Finally, a mutual-interaction decoder (MID) is designed to enhance cross-modal feature alignment through cascaded bidirectional cross-attention, thereby enabling precise segmentation mask prediction. To further forster the research of RRSIS, we also construct RISBench, a new large-scale benchmark dataset comprising 52,472 image-language-label triplets. Extensive benchmarking on RISBench and two other prevalent datasets demonstrates the superior performance of the proposed CroBIM over existing state-of-the-art (SOTA) methods. The source code for CroBIM and the RISBench dataset will be publicly available at https://github.com/HIT-SIRS/CroBIM
>
---
#### [replaced 133] Semantic Correspondence: Unified Benchmarking and a Strong Baseline
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18060v2](http://arxiv.org/pdf/2505.18060v2)**

> **作者:** Kaiyan Zhang; Xinghui Li; Jingyi Lu; Kai Han
>
> **摘要:** Establishing semantic correspondence is a challenging task in computer vision, aiming to match keypoints with the same semantic information across different images. Benefiting from the rapid development of deep learning, remarkable progress has been made over the past decade. However, a comprehensive review and analysis of this task remains absent. In this paper, we present the first extensive survey of semantic correspondence methods. We first propose a taxonomy to classify existing methods based on the type of their method designs. These methods are then categorized accordingly, and we provide a detailed analysis of each approach. Furthermore, we aggregate and summarize the results of methods in literature across various benchmarks into a unified comparative table, with detailed configurations to highlight performance variations. Additionally, to provide a detailed understanding on existing methods for semantic matching, we thoroughly conduct controlled experiments to analyse the effectiveness of the components of different methods. Finally, we propose a simple yet effective baseline that achieves state-of-the-art performance on multiple benchmarks, providing a solid foundation for future research in this field. We hope this survey serves as a comprehensive reference and consolidated baseline for future development. Code is publicly available at: https://github.com/Visual-AI/Semantic-Correspondence.
>
---
#### [replaced 134] NuGrounding: A Multi-View 3D Visual Grounding Framework in Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22436v2](http://arxiv.org/pdf/2503.22436v2)**

> **作者:** Fuhao Li; Huan Jin; Bin Gao; Liaoyuan Fan; Lihui Jiang; Long Zeng
>
> **摘要:** Multi-view 3D visual grounding is critical for autonomous driving vehicles to interpret natural languages and localize target objects in complex environments. However, existing datasets and methods suffer from coarse-grained language instructions, and inadequate integration of 3D geometric reasoning with linguistic comprehension. To this end, we introduce NuGrounding, the first large-scale benchmark for multi-view 3D visual grounding in autonomous driving. We present a Hierarchy of Grounding (HoG) method to construct NuGrounding to generate hierarchical multi-level instructions, ensuring comprehensive coverage of human instruction patterns. To tackle this challenging dataset, we propose a novel paradigm that seamlessly combines instruction comprehension abilities of multi-modal LLMs (MLLMs) with precise localization abilities of specialist detection models. Our approach introduces two decoupled task tokens and a context query to aggregate 3D geometric information and semantic instructions, followed by a fusion decoder to refine spatial-semantic feature fusion for precise localization. Extensive experiments demonstrate that our method significantly outperforms the baselines adapted from representative 3D scene understanding methods by a significant margin and achieves 0.59 in precision and 0.64 in recall, with improvements of 50.8% and 54.7%.
>
---
#### [replaced 135] Interspatial Attention for Efficient 4D Human Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15800v2](http://arxiv.org/pdf/2505.15800v2)**

> **作者:** Ruizhi Shao; Yinghao Xu; Yujun Shen; Ceyuan Yang; Yang Zheng; Changan Chen; Yebin Liu; Gordon Wetzstein
>
> **备注:** Project page: https://dsaurus.github.io/isa4d/
>
> **摘要:** Generating photorealistic videos of digital humans in a controllable manner is crucial for a plethora of applications. Existing approaches either build on methods that employ template-based 3D representations or emerging video generation models but suffer from poor quality or limited consistency and identity preservation when generating individual or multiple digital humans. In this paper, we introduce a new interspatial attention (ISA) mechanism as a scalable building block for modern diffusion transformer (DiT)--based video generation models. ISA is a new type of cross attention that uses relative positional encodings tailored for the generation of human videos. Leveraging a custom-developed video variation autoencoder, we train a latent ISA-based diffusion model on a large corpus of video data. Our model achieves state-of-the-art performance for 4D human video synthesis, demonstrating remarkable motion consistency and identity preservation while providing precise control of the camera and body poses. Our code and model are publicly released at https://dsaurus.github.io/isa4d/.
>
---
#### [replaced 136] RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21692v2](http://arxiv.org/pdf/2503.21692v2)**

> **作者:** Daniel Bermuth; Alexander Poeppel; Wolfgang Reif
>
> **摘要:** The integration of multi-view imaging and pose estimation represents a significant advance in computer vision applications, offering new possibilities for understanding human movement and interactions. This work presents a new algorithm that improves multi-view multi-person pose estimation, focusing on fast triangulation speeds and good generalization capabilities. The approach extends to whole-body pose estimation, capturing details from facial expressions to finger movements across multiple individuals and viewpoints. Adaptability to different settings is demonstrated through strong performance across unseen datasets and configurations. To support further progress in this field, all of this work is publicly accessible.
>
---
#### [replaced 137] FERGI: Automatic Scoring of User Preferences for Text-to-Image Generation from Spontaneous Facial Expression Reaction
- **分类: cs.CV; cs.AI; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.03187v4](http://arxiv.org/pdf/2312.03187v4)**

> **作者:** Shuangquan Feng; Junhua Ma; Virginia R. de Sa
>
> **摘要:** Researchers have proposed to use data of human preference feedback to fine-tune text-to-image generative models. However, the scalability of human feedback collection has been limited by its reliance on manual annotation. Therefore, we develop and test a method to automatically score user preferences from their spontaneous facial expression reaction to the generated images. We collect a dataset of Facial Expression Reaction to Generated Images (FERGI) and show that the activations of multiple facial action units (AUs) are highly correlated with user evaluations of the generated images. We develop an FAU-Net (Facial Action Units Neural Network), which receives inputs from an AU estimation model, to automatically score user preferences for text-to-image generation based on their facial expression reactions, which is complementary to the pre-trained scoring models based on the input text prompts and generated images. Integrating our FAU-Net valence score with the pre-trained scoring models improves their consistency with human preferences. This method of automatic annotation with facial expression analysis can be potentially generalized to other generation tasks. The code is available at https://github.com/ShuangquanFeng/FERGI, and the dataset is also available at the same link for research purposes.
>
---
#### [replaced 138] Token Sampling Uncertainty Does Not Explain Homogeneity Bias in Large Language Models
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.19337v2](http://arxiv.org/pdf/2501.19337v2)**

> **作者:** Messi H. J. Lee; Soyeon Jeon
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Homogeneity bias is one form of stereotyping in AI models where certain groups are represented as more similar to each other than other groups. This bias is a major obstacle to creating equitable language technologies. We test whether the bias is driven by systematic differences in token-sampling uncertainty across six large language models. While we observe the presence of homogeneity bias using sentence similarity, we find very little difference in token sampling uncertainty across groups. This finding elucidates why temperature-based sampling adjustments fail to mitigate homogeneity bias. It suggests researchers should prioritize interventions targeting representation learning mechanisms and training corpus composition rather than inference-time output manipulations.
>
---
#### [replaced 139] A Survey on Efficient Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09724v2](http://arxiv.org/pdf/2504.09724v2)**

> **作者:** Gaurav Shinde; Anuradha Ravi; Emon Dey; Shadman Sakib; Milind Rampure; Nirmalya Roy
>
> **备注:** 41 pages, 18 figures
>
> **摘要:** Vision-language models (VLMs) integrate visual and textual information, enabling a wide range of applications such as image captioning and visual question answering, making them crucial for modern AI systems. However, their high computational demands pose challenges for real-time applications. This has led to a growing focus on developing efficient vision language models. In this survey, we review key techniques for optimizing VLMs on edge and resource-constrained devices. We also explore compact VLM architectures, frameworks and provide detailed insights into the performance-memory trade-offs of efficient VLMs. Furthermore, we establish a GitHub repository at https://github.com/MPSCUMBC/Efficient-Vision-Language-Models-A-Survey to compile all surveyed papers, which we will actively update. Our objective is to foster deeper research in this area.
>
---
#### [replaced 140] Unlocking Text Capabilities in Vision Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10981v2](http://arxiv.org/pdf/2503.10981v2)**

> **作者:** Fawaz Sammani; Jonas Fischer; Nikos Deligiannis
>
> **摘要:** Visual classifiers provide high-dimensional feature representations that are challenging to interpret and analyze. Text, in contrast, provides a more expressive and human-friendly interpretable medium for understanding and analyzing model behavior. We propose a simple, yet powerful method for reformulating any pretrained visual classifier so that it can be queried with free-form text without compromising its original performance. Our approach is label-free, data and compute-efficient, and is trained to preserve the underlying classifiers distribution and decision-making processes. Our method unlocks several zero-shot text interpretability applications for any visual classifier. We apply our method on 40 visual classifiers and demonstrate two primary applications: 1) building both label-free and zero-shot concept bottleneck models and therefore converting any visual classifier to be inherently-interpretable and 2) zero-shot decoding of visual features into natural language sentences. In both tasks we establish new state-of-the-art results, outperforming existing works and surpassing CLIP-based baselines with ImageNet-only trained classifiers, while using up to 400x fewer images and 400,000x less text during training.
>
---
#### [replaced 141] Zero4D: Training-Free 4D Video Generation From Single Video Using Off-the-Shelf Video Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22622v2](http://arxiv.org/pdf/2503.22622v2)**

> **作者:** Jangho Park; Taesung Kwon; Jong Chul Ye
>
> **备注:** project page: https://zero4dvid.github.io/
>
> **摘要:** Recently, multi-view or 4D video generation has emerged as a significant research topic. Nonetheless, recent approaches to 4D generation still struggle with fundamental limitations, as they primarily rely on harnessing multiple video diffusion models with additional training or compute-intensive training of a full 4D diffusion model with limited real-world 4D data and large computational costs. To address these challenges, here we propose the first training-free 4D video generation method that leverages the off-the-shelf video diffusion models to generate multi-view videos from a single input video. Our approach consists of two key steps: (1) By designating the edge frames in the spatio-temporal sampling grid as key frames, we first synthesize them using a video diffusion model, leveraging a depth-based warping technique for guidance. This approach ensures structural consistency across the generated frames, preserving spatial and temporal coherence. (2) We then interpolate the remaining frames using a video diffusion model, constructing a fully populated and temporally coherent sampling grid while preserving spatial and temporal consistency. Through this approach, we extend a single video into a multi-view video along novel camera trajectories while maintaining spatio-temporal consistency. Our method is training-free and fully utilizes an off-the-shelf video diffusion model, offering a practical and effective solution for multi-view video generation.
>
---
#### [replaced 142] Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15660v2](http://arxiv.org/pdf/2505.15660v2)**

> **作者:** Jiaming Zhou; Ke Ye; Jiayi Liu; Teli Ma; Zifan Wang; Ronghe Qiu; Kun-Yu Lin; Zhilin Zhao; Junwei Liang
>
> **备注:** Project Page: https://jiaming-zhou.github.io/AGNOSTOS
>
> **摘要:** The generalization capabilities of vision-language-action (VLA) models to unseen tasks are crucial to achieving general-purpose robotic manipulation in open-world settings. However, the cross-task generalization capabilities of existing VLA models remain significantly underexplored. To address this gap, we introduce AGNOSTOS, a novel simulation benchmark designed to rigorously evaluate cross-task zero-shot generalization in manipulation. AGNOSTOS comprises 23 unseen manipulation tasks for testing, distinct from common training task distributions, and incorporates two levels of generalization difficulty to assess robustness. Our systematic evaluation reveals that current VLA models, despite being trained on diverse datasets, struggle to generalize effectively to these unseen tasks. To overcome this limitation, we propose Cross-Task In-Context Manipulation (X-ICM), a method that conditions large language models (LLMs) on in-context demonstrations from seen tasks to predict action sequences for unseen tasks. Additionally, we introduce a dynamics-guided sample selection strategy that identifies relevant demonstrations by capturing cross-task dynamics. On AGNOSTOS, X-ICM significantly improves cross-task zero-shot generalization performance over leading VLAs. We believe AGNOSTOS and X-ICM will serve as valuable tools for advancing general-purpose robotic manipulation.
>
---
#### [replaced 143] WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.04326v2](http://arxiv.org/pdf/2502.04326v2)**

> **作者:** Jack Hong; Shilin Yan; Jiayin Cai; Xiaolong Jiang; Yao Hu; Weidi Xie
>
> **摘要:** We introduce WorldSense, the first benchmark to assess the multi-modal video understanding, that simultaneously encompasses visual, audio, and text inputs. In contrast to existing benchmarks, our WorldSense has several features: (i) collaboration of omni-modality, we design the evaluation tasks to feature a strong coupling of audio and video, requiring models to effectively utilize the synergistic perception of omni-modality; (ii) diversity of videos and tasks, WorldSense encompasses a diverse collection of 1,662 audio-visual synchronised videos, systematically categorized into 8 primary domains and 67 fine-grained subcategories to cover the broad scenarios, and 3,172 multi-choice QA pairs across 26 distinct tasks to enable the comprehensive evaluation; (iii) high-quality annotations, all the QA pairs are manually labeled by 80 expert annotators with multiple rounds of correction to ensure quality. Based on our WorldSense, we extensively evaluate various state-of-the-art models. The experimental results indicate that existing models face significant challenges in understanding real-world scenarios (48.0% best accuracy). By analyzing the limitations of current models, we aim to provide valuable insight to guide development of real-world understanding. We hope our WorldSense can provide a platform for evaluating the ability in constructing and understanding coherent contexts from omni-modality.
>
---
#### [replaced 144] Explanatory Instructions: Towards Unified Vision Tasks Understanding and Zero-shot Generalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.18525v3](http://arxiv.org/pdf/2412.18525v3)**

> **作者:** Yang Shen; Xiu-Shen Wei; Yifan Sun; Yuxin Song; Tao Yuan; Jian Jin; Heyang Xu; Yazhou Yao; Errui Ding
>
> **备注:** ICML'25, 44 pages
>
> **摘要:** Computer Vision (CV) has yet to fully achieve the zero-shot task generalization observed in Natural Language Processing (NLP), despite following many of the milestones established in NLP, such as large transformer models, extensive pre-training, and the auto-regression paradigm, among others. In this paper, we explore the idea that CV adopts discrete and terminological task definitions (\eg, ``image segmentation''), which may be a key barrier to zero-shot task generalization. Our hypothesis is that without truly understanding previously-seen tasks--due to these terminological definitions--deep models struggle to generalize to novel tasks. To verify this, we introduce Explanatory Instructions, which provide an intuitive way to define CV task objectives through detailed linguistic transformations from input images to outputs. We create a large-scale dataset comprising 12 million ``image input $\to$ explanatory instruction $\to$ output'' triplets, and train an auto-regressive-based vision-language model (AR-based VLM) that takes both images and explanatory instructions as input. By learning to follow these instructions, the AR-based VLM achieves instruction-level zero-shot capabilities for previously-seen tasks and demonstrates strong zero-shot generalization for unseen CV tasks. Code and dataset will be openly available on our GitHub repository.
>
---
#### [replaced 145] AnchorFormer: Differentiable Anchor Attention for Efficient Vision Transformer
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.16463v2](http://arxiv.org/pdf/2505.16463v2)**

> **作者:** Jiquan Shan; Junxiao Wang; Lifeng Zhao; Liang Cai; Hongyuan Zhang; Ioannis Liritzis
>
> **摘要:** Recently, vision transformers (ViTs) have achieved excellent performance on vision tasks by measuring the global self-attention among the image patches. Given $n$ patches, they will have quadratic complexity such as $\mathcal{O}(n^2)$ and the time cost is high when splitting the input image with a small granularity. Meanwhile, the pivotal information is often randomly gathered in a few regions of an input image, some tokens may not be helpful for the downstream tasks. To handle this problem, we introduce an anchor-based efficient vision transformer (AnchorFormer), which employs the anchor tokens to learn the pivotal information and accelerate the inference. Firstly, by estimating the bipartite attention between the anchors and tokens, the complexity will be reduced from $\mathcal{O}(n^2)$ to $\mathcal{O}(mn)$, where $m$ is an anchor number and $m < n$. Notably, by representing the anchors with the neurons in a neural layer, we can differentiable learn these distributions and approximate global self-attention through the Markov process. Moreover, we extend the proposed model to three downstream tasks including classification, detection, and segmentation. Extensive experiments show the effectiveness of our AnchorFormer, e.g., achieving up to a 9.0% higher accuracy or 46.7% FLOPs reduction on ImageNet classification, 81.3% higher mAP on COCO detection under comparable FLOPs, as compared to the current baselines.
>
---
#### [replaced 146] Unsupervised Detection of Distribution Shift in Inverse Problems using Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11482v3](http://arxiv.org/pdf/2505.11482v3)**

> **作者:** Shirin Shoushtari; Edward P. Chandler; Yuanhao Wang; M. Salman Asif; Ulugbek S. Kamilov
>
> **摘要:** Diffusion models are widely used as priors in imaging inverse problems. However, their performance often degrades under distribution shifts between the training and test-time images. Existing methods for identifying and quantifying distribution shifts typically require access to clean test images, which are almost never available while solving inverse problems (at test time). We propose a fully unsupervised metric for estimating distribution shifts using only indirect (corrupted) measurements and score functions from diffusion models trained on different datasets. We theoretically show that this metric estimates the KL divergence between the training and test image distributions. Empirically, we show that our score-based metric, using only corrupted measurements, closely approximates the KL divergence computed from clean images. Motivated by this result, we show that aligning the out-of-distribution score with the in-distribution score -- using only corrupted measurements -- reduces the KL divergence and leads to improved reconstruction quality across multiple inverse problems.
>
---
#### [replaced 147] The One RING: a Robotic Indoor Navigation Generalist
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.14401v2](http://arxiv.org/pdf/2412.14401v2)**

> **作者:** Ainaz Eftekhar; Rose Hendrix; Luca Weihs; Jiafei Duan; Ege Caglar; Jordi Salvador; Alvaro Herrasti; Winson Han; Eli VanderBil; Aniruddha Kembhavi; Ali Farhadi; Ranjay Krishna; Kiana Ehsani; Kuo-Hao Zeng
>
> **摘要:** Modern robots vary significantly in shape, size, and sensor configurations used to perceive and interact with their environments. However, most navigation policies are embodiment-specific--a policy trained on one robot typically fails to generalize to another, even with minor changes in body size or camera viewpoint. As custom hardware becomes increasingly common, there is a growing need for a single policy that generalizes across embodiments, eliminating the need to retrain for each specific robot. In this paper, we introduce RING (Robotic Indoor Navigation Generalist), an embodiment-agnostic policy that turns any mobile robot into an effective indoor semantic navigator. Trained entirely in simulation, RING leverages large-scale randomization over robot embodiments to enable robust generalization to many real-world platforms. To support this, we augment the AI2-THOR simulator to instantiate robots with controllable configurations, varying in body size, rotation pivot point, and camera parameters. On the visual object-goal navigation task, RING achieves strong cross-embodiment (XE) generalization--72.1% average success rate across five simulated embodiments (a 16.7% absolute improvement on the Chores-S benchmark) and 78.9% across four real-world platforms, including Stretch RE-1, LoCoBot, and Unitree Go1--matching or even surpassing embodiment-specific policies. We further deploy RING on the RB-Y1 wheeled humanoid in a real-world kitchen environment, showcasing its out-of-the-box potential for mobile manipulation platforms. (Project website: https://one-ring-policy.allen.ai)
>
---
#### [replaced 148] Part-aware Prompted Segment Anything Model for Adaptive Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.05433v2](http://arxiv.org/pdf/2403.05433v2)**

> **作者:** Chenhui Zhao; Liyue Shen
>
> **备注:** TMLR 2025
>
> **摘要:** Precision medicine, such as patient-adaptive treatments assisted by medical image analysis, poses new challenges for segmentation algorithms in adapting to new patients, due to the large variability across different patients and the limited availability of annotated data for each patient. In this work, we propose a data-efficient segmentation algorithm, namely Part-aware Prompted Segment Anything Model ($P^2SAM$). Without any model fine-tuning, $P^2SAM$ enables seamless adaptation to any new patients relying only on one-shot patient-specific data. We introduce a novel part-aware prompt mechanism to select multiple-point prompts based on the part-level features of the one-shot data, which can be extensively integrated into different promptable segmentation models, such as SAM and SAM 2. Moreover, to determine the optimal number of parts for each specific case, we propose a distribution-guided retrieval approach that further enhances the robustness of the part-aware prompt mechanism. $P^2SAM$ improves the performance by +8.0% and +2.0% mean Dice score for two different patient-adaptive segmentation applications, respectively. In addition, $P^2SAM$ also exhibits impressive generalizability in other adaptive segmentation tasks in the natural image domain, e.g., +6.4% mIoU within personalized object segmentation task. The code is available at: https://github.com/Zch0414/p2sam
>
---
#### [replaced 149] ViEEG: Hierarchical Neural Coding with Cross-Modal Progressive Enhancement for EEG-Based Visual Decoding
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.12408v2](http://arxiv.org/pdf/2505.12408v2)**

> **作者:** Minxu Liu; Donghai Guan; Chuhang Zheng; Chunwei Tian; Jie Wen; Qi Zhu
>
> **备注:** 24 pages, 18 figures
>
> **摘要:** Understanding and decoding brain activity into visual representations is a fundamental challenge at the intersection of neuroscience and artificial intelligence. While EEG-based visual decoding has shown promise due to its non-invasive, low-cost nature and millisecond-level temporal resolution, existing methods are limited by their reliance on flat neural representations that overlook the brain's inherent visual hierarchy. In this paper, we introduce ViEEG, a biologically inspired hierarchical EEG decoding framework that aligns with the Hubel-Wiesel theory of visual processing. ViEEG decomposes each visual stimulus into three biologically aligned components-contour, foreground object, and contextual scene-serving as anchors for a three-stream EEG encoder. These EEG features are progressively integrated via cross-attention routing, simulating cortical information flow from V1 to IT to the association cortex. We further adopt hierarchical contrastive learning to align EEG representations with CLIP embeddings, enabling zero-shot object recognition. Extensive experiments on the THINGS-EEG dataset demonstrate that ViEEG achieves state-of-the-art performance, with 40.9% Top-1 accuracy in subject-dependent and 22.9% Top-1 accuracy in cross-subject settings, surpassing existing methods by over 45%. Our framework not only advances the performance frontier but also sets a new paradigm for biologically grounded brain decoding in AI.
>
---
#### [replaced 150] ALMA: a mathematics-driven approach for determining tuning parameters in generalized LASSO problems, with applications to MRI
- **分类: eess.IV; cs.CV; eess.SP; physics.med-ph; 92C55, 62J07, 65K10; I.4.2; I.4.5; J.2; J.3**

- **链接: [http://arxiv.org/pdf/2406.19239v2](http://arxiv.org/pdf/2406.19239v2)**

> **作者:** Gianluca Giacchi; Isidoros Iakovidis; Bastien Milani; Micah Murray; Benedetta Franceschiello
>
> **备注:** Modified pictures, authors and fixed some typo
>
> **摘要:** Magnetic Resonance Imaging (MRI) is a powerful technique employed for non-invasive in vivo visualization of internal structures. Sparsity is often deployed to accelerate the signal acquisition or overcome the presence of motion artifacts, improving the quality of image reconstruction. Image reconstruction algorithms use TV-regularized LASSO (Total Variation-regularized LASSO) to retrieve the missing information of undersampled signals, by cleaning the data of noise and while optimizing sparsity. A tuning parameter moderates the balance between these two aspects; its choice affecting the quality of the reconstructions. Currently, there is a lack of general deterministic techniques to choose these parameters, which are oftentimes manually selected and thus hinder the reliability of the reconstructions. Here, we present ALMA (Algorithm for Lagrange Multipliers Approximation), an iterative mathematics-inspired technique that computes tuning parameters for generalized LASSO problems during MRI reconstruction. We analyze quantitatively the performance of these parameters for imaging reconstructions via TV-LASSO in an MRI context on phantoms. Although our study concentrates on TV-LASSO, the techniques developed here hold significant promise for a wide array of applications. ALMA is not only adaptable to more generalized LASSO problems but is also robust to accommodate other forms of regularization beyond total variation. Moreover, it extends effectively to handle non-Cartesian sampling trajectories, broadening its utility in complex data reconstruction scenarios. More generally, ALMA provides a powerful tool for numerically solving constrained optimization problems across various disciplines, offering a versatile and impactful solution for advanced computational challenges.
>
---
#### [replaced 151] Semantic-Space-Intervened Diffusive Alignment for Visual Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05721v2](http://arxiv.org/pdf/2505.05721v2)**

> **作者:** Zixuan Li; Lei Meng; Guoqing Chao; Wei Wu; Xiaoshuo Yan; Yimeng Yang; Zhuang Qi; Xiangxu Meng
>
> **摘要:** Cross-modal alignment is an effective approach to improving visual classification. Existing studies typically enforce a one-step mapping that uses deep neural networks to project the visual features to mimic the distribution of textual features. However, they typically face difficulties in finding such a projection due to the two modalities in both the distribution of class-wise samples and the range of their feature values. To address this issue, this paper proposes a novel Semantic-Space-Intervened Diffusive Alignment method, termed SeDA, models a semantic space as a bridge in the visual-to-textual projection, considering both types of features share the same class-level information in classification. More importantly, a bi-stage diffusion framework is developed to enable the progressive alignment between the two modalities. Specifically, SeDA first employs a Diffusion-Controlled Semantic Learner to model the semantic features space of visual features by constraining the interactive features of the diffusion model and the category centers of visual features. In the later stage of SeDA, the Diffusion-Controlled Semantic Translator focuses on learning the distribution of textual features from the semantic space. Meanwhile, the Progressive Feature Interaction Network introduces stepwise feature interactions at each alignment step, progressively integrating textual information into mapped features. Experimental results show that SeDA achieves stronger cross-modal feature alignment, leading to superior performance over existing methods across multiple scenarios.
>
---
#### [replaced 152] VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02492v2](http://arxiv.org/pdf/2502.02492v2)**

> **作者:** Hila Chefer; Uriel Singer; Amit Zohar; Yuval Kirstain; Adam Polyak; Yaniv Taigman; Lior Wolf; Shelly Sheynin
>
> **摘要:** Despite tremendous recent progress, generative video models still struggle to capture real-world motion, dynamics, and physics. We show that this limitation arises from the conventional pixel reconstruction objective, which biases models toward appearance fidelity at the expense of motion coherence. To address this, we introduce VideoJAM, a novel framework that instills an effective motion prior to video generators, by encouraging the model to learn a joint appearance-motion representation. VideoJAM is composed of two complementary units. During training, we extend the objective to predict both the generated pixels and their corresponding motion from a single learned representation. During inference, we introduce Inner-Guidance, a mechanism that steers the generation toward coherent motion by leveraging the model's own evolving motion prediction as a dynamic guidance signal. Notably, our framework can be applied to any video model with minimal adaptations, requiring no modifications to the training data or scaling of the model. VideoJAM achieves state-of-the-art performance in motion coherence, surpassing highly competitive proprietary models while also enhancing the perceived visual quality of the generations. These findings emphasize that appearance and motion can be complementary and, when effectively integrated, enhance both the visual quality and the coherence of video generation. Project website: https://hila-chefer.github.io/videojam-paper.github.io/
>
---
#### [replaced 153] Towards user-centered interactive medical image segmentation in VR with an assistive AI agent
- **分类: cs.HC; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07214v3](http://arxiv.org/pdf/2505.07214v3)**

> **作者:** Pascal Spiegler; Arash Harirpoush; Yiming Xiao
>
> **摘要:** Crucial in disease analysis and surgical planning, manual segmentation of volumetric medical scans (e.g. MRI, CT) is laborious, error-prone, and challenging to master, while fully automatic algorithms can benefit from user feedback. Therefore, with the complementary power of the latest radiological AI foundation models and virtual reality (VR)'s intuitive data interaction, we propose SAMIRA, a novel conversational AI agent for medical VR that assists users with localizing, segmenting, and visualizing 3D medical concepts. Through speech-based interaction, the agent helps users understand radiological features, locate clinical targets, and generate segmentation masks that can be refined with just a few point prompts. The system also supports true-to-scale 3D visualization of segmented pathology to enhance patient-specific anatomical understanding. Furthermore, to determine the optimal interaction paradigm under near-far attention-switching for refining segmentation masks in an immersive, human-in-the-loop workflow, we compare VR controller pointing, head pointing, and eye tracking as input modes. With a user study, evaluations demonstrated a high usability score (SUS=90.0 $\pm$ 9.0), low overall task load, as well as strong support for the proposed VR system's guidance, training potential, and integration of AI in radiological segmentation tasks.
>
---
#### [replaced 154] EvAnimate: Event-conditioned Image-to-Video Generation for Human Animation
- **分类: cs.CV; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.18552v2](http://arxiv.org/pdf/2503.18552v2)**

> **作者:** Qiang Qu; Ming Li; Xiaoming Chen; Tongliang Liu
>
> **摘要:** Conditional human animation traditionally animates static reference images using pose-based motion cues extracted from video data. However, these video-derived cues often suffer from low temporal resolution, motion blur, and unreliable performance under challenging lighting conditions. In contrast, event cameras inherently provide robust and high temporal-resolution motion information, offering resilience to motion blur, low-light environments, and exposure variations. In this paper, we propose EvAnimate, the first method leveraging event streams as robust and precise motion cues for conditional human image animation. Our approach is fully compatible with diffusion-based generative models, enabled by encoding asynchronous event data into a specialized three-channel representation with adaptive slicing rates and densities. High-quality and temporally coherent animations are achieved through a dual-branch architecture explicitly designed to exploit event-driven dynamics, significantly enhancing performance under challenging real-world conditions. Enhanced cross-subject generalization is further achieved using specialized augmentation strategies. To facilitate future research, we establish a new benchmarking, including simulated event data for training and validation, and a real-world event dataset capturing human actions under normal and challenging scenarios. The experiment results demonstrate that EvAnimate achieves high temporal fidelity and robust performance in scenarios where traditional video-derived cues fall short.
>
---
#### [replaced 155] ZeroPur: Succinct Training-Free Adversarial Purification
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2406.03143v2](http://arxiv.org/pdf/2406.03143v2)**

> **作者:** Erhu Liu; Zonglin Yang; Bo Liu; Bin Xiao; Xiuli Bi
>
> **备注:** 17 pages, 7 figures, under review
>
> **摘要:** Adversarial purification is a kind of defense technique that can defend against various unseen adversarial attacks without modifying the victim classifier. Existing methods often depend on external generative models or cooperation between auxiliary functions and victim classifiers. However, retraining generative models, auxiliary functions, or victim classifiers relies on the domain of the fine-tuned dataset and is computation-consuming. In this work, we suppose that adversarial images are outliers of the natural image manifold, and the purification process can be considered as returning them to this manifold. Following this assumption, we present a simple adversarial purification method without further training to purify adversarial images, called ZeroPur. ZeroPur contains two steps: given an adversarial example, Guided Shift obtains the shifted embedding of the adversarial example by the guidance of its blurred counterparts; after that, Adaptive Projection constructs a directional vector by this shifted embedding to provide momentum, projecting adversarial images onto the manifold adaptively. ZeroPur is independent of external models and requires no retraining of victim classifiers or auxiliary functions, relying solely on victim classifiers themselves to achieve purification. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) using various classifier architectures (ResNet, WideResNet) demonstrate that our method achieves state-of-the-art robust performance. The code will be publicly available.
>
---
#### [replaced 156] On the Fairness, Diversity and Reliability of Text-to-Image Generative Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.13981v2](http://arxiv.org/pdf/2411.13981v2)**

> **作者:** Jordan Vice; Naveed Akhtar; Leonid Sigal; Richard Hartley; Ajmal Mian
>
> **备注:** This research is supported by the NISDRG project #20100007, funded by the Australian Government
>
> **摘要:** The rapid proliferation of multimodal generative models has sparked critical discussions on their reliability, fairness and potential for misuse. While text-to-image models excel at producing high-fidelity, user-guided content, they often exhibit unpredictable behaviors and vulnerabilities that can be exploited to manipulate class or concept representations. To address this, we propose an evaluation framework to assess model reliability by analyzing responses to global and local perturbations in the embedding space, enabling the identification of inputs that trigger unreliable or biased behavior. Beyond social implications, fairness and diversity are fundamental to defining robust and trustworthy model behavior. Our approach offers deeper insights into these essential aspects by evaluating: (i) generative diversity, measuring the breadth of visual representations for learned concepts, and (ii) generative fairness, which examines the impact that removing concepts from input prompts has on control, under a low guidance setup. Beyond these evaluations, our method lays the groundwork for detecting unreliable, bias-injected models and tracing the provenance of embedded biases. Our code is publicly available at https://github.com/JJ-Vice/T2I_Fairness_Diversity_Reliability. Keywords: Fairness, Reliability, AI Ethics, Bias, Text-to-Image Models
>
---
#### [replaced 157] ImageRAG: Enhancing Ultra High Resolution Remote Sensing Imagery Analysis with ImageRAG
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.07688v4](http://arxiv.org/pdf/2411.07688v4)**

> **作者:** Zilun Zhang; Haozhan Shen; Tiancheng Zhao; Zian Guan; Bin Chen; Yuhao Wang; Xu Jia; Yuxiang Cai; Yongheng Shang; Jianwei Yin
>
> **备注:** Accepted by IEEE Geoscience and Remote Sensing Magazine
>
> **摘要:** Ultra High Resolution (UHR) remote sensing imagery (RSI) (e.g. 100,000 $\times$ 100,000 pixels or more) poses a significant challenge for current Remote Sensing Multimodal Large Language Models (RSMLLMs). If choose to resize the UHR image to standard input image size, the extensive spatial and contextual information that UHR images contain will be neglected. Otherwise, the original size of these images often exceeds the token limits of standard RSMLLMs, making it difficult to process the entire image and capture long-range dependencies to answer the query based on the abundant visual context. In this paper, we introduce ImageRAG for RS, a training-free framework to address the complexities of analyzing UHR remote sensing imagery. By transforming UHR remote sensing image analysis task to image's long context selection task, we design an innovative image contextual retrieval mechanism based on the Retrieval-Augmented Generation (RAG) technique, denoted as ImageRAG. ImageRAG's core innovation lies in its ability to selectively retrieve and focus on the most relevant portions of the UHR image as visual contexts that pertain to a given query. Fast path and slow path are proposed in this framework to handle this task efficiently and effectively. ImageRAG allows RSMLLMs to manage extensive context and spatial information from UHR RSI, ensuring the analysis is both accurate and efficient. Codebase will be released in https://github.com/om-ai-lab/ImageRAG
>
---
#### [replaced 158] On the status of current quantum machine learning software
- **分类: quant-ph; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.08962v2](http://arxiv.org/pdf/2503.08962v2)**

> **作者:** Manish K. Gupta; Tomasz Rybotycki; Piotr Gawron
>
> **备注:** 8 pages, 1 figure, 1 table
>
> **摘要:** The recent advancements in noisy intermediate-scale quantum (NISQ) devices implementation allow us to study their application to real-life computational problems. However, hardware challenges are not the only ones that hinder our quantum computation capabilities. Software limitations are the other, less explored side of this medal. Using satellite image segmentation as a task example, we investigated how difficult it is to run a hybrid quantum-classical model on a real, publicly available quantum device. We also analyzed the costs of such endeavor and the change in quality of model.
>
---
#### [replaced 159] The Double-Ellipsoid Geometry of CLIP
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.14517v3](http://arxiv.org/pdf/2411.14517v3)**

> **作者:** Meir Yossef Levi; Guy Gilboa
>
> **备注:** Accepted to ICML 2025. This version matches the camera-ready version
>
> **摘要:** Contrastive Language-Image Pre-Training (CLIP) is highly instrumental in machine learning applications within a large variety of domains. We investigate the geometry of this embedding, which is still not well understood. We examine the raw unnormalized embedding and show that text and image reside on linearly separable ellipsoid shells, not centered at the origin. We explain the benefits of having this structure, allowing to better embed instances according to their uncertainty during contrastive training. Frequent concepts in the dataset yield more false negatives, inducing greater uncertainty. A new notion of conformity is introduced, which measures the average cosine similarity of an instance to any other instance within a representative data set. We show this measure can be accurately estimated by simply computing the cosine similarity to the modality mean vector. Furthermore, we find that CLIP's modality gap optimizes the matching of the conformity distributions of image and text.
>
---
#### [replaced 160] UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15725v2](http://arxiv.org/pdf/2505.15725v2)**

> **作者:** Xiangyu Wang; Donglin Yang; Yue Liao; Wenhao Zheng; wenjun wu; Bin Dai; Hongsheng Li; Si Liu
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are evolving into language-interactive platforms, enabling more intuitive forms of human-drone interaction. While prior works have primarily focused on high-level planning and long-horizon navigation, we shift attention to language-guided fine-grained trajectory control, where UAVs execute short-range, reactive flight behaviors in response to language instructions. We formalize this problem as the Flying-on-a-Word (Flow) task and introduce UAV imitation learning as an effective approach. In this framework, UAVs learn fine-grained control policies by mimicking expert pilot trajectories paired with atomic language instructions. To support this paradigm, we present UAV-Flow, the first real-world benchmark for language-conditioned, fine-grained UAV control. It includes a task formulation, a large-scale dataset collected in diverse environments, a deployable control framework, and a simulation suite for systematic evaluation. Our design enables UAVs to closely imitate the precise, expert-level flight trajectories of human pilots and supports direct deployment without sim-to-real gap. We conduct extensive experiments on UAV-Flow, benchmarking VLN and VLA paradigms. Results show that VLA models are superior to VLN baselines and highlight the critical role of spatial grounding in the fine-grained Flow setting.
>
---
#### [replaced 161] What Makes a Scene ? Scene Graph-based Evaluation and Feedback for Controllable Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15435v2](http://arxiv.org/pdf/2411.15435v2)**

> **作者:** Zuyao Chen; Jinlin Wu; Zhen Lei; Chang Wen Chen
>
> **摘要:** While text-to-image generation has been extensively studied, generating images from scene graphs remains relatively underexplored, primarily due to challenges in accurately modeling spatial relationships and object interactions. To fill this gap, we introduce Scene-Bench, a comprehensive benchmark designed to evaluate and enhance the factual consistency in generating natural scenes. Scene-Bench comprises MegaSG, a large-scale dataset of one million images annotated with scene graphs, facilitating the training and fair comparison of models across diverse and complex scenes. Additionally, we propose SGScore, a novel evaluation metric that leverages chain-of-thought reasoning capabilities of multimodal large language models (LLMs) to assess both object presence and relationship accuracy, offering a more effective measure of factual consistency than traditional metrics like FID and CLIPScore. Building upon this evaluation framework, we develop a scene graph feedback pipeline that iteratively refines generated images by identifying and correcting discrepancies between the scene graph and the image. Extensive experiments demonstrate that Scene-Bench provides a more comprehensive and effective evaluation framework compared to existing benchmarks, particularly for complex scene generation. Furthermore, our feedback strategy significantly enhances the factual consistency of image generation models, advancing the field of controllable image generation.
>
---
#### [replaced 162] VISTANet: VIsual Spoken Textual Additive Net for Interpretable Multimodal Emotion Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2208.11450v4](http://arxiv.org/pdf/2208.11450v4)**

> **作者:** Puneet Kumar; Sarthak Malik; Balasubramanian Raman; Xiaobai Li
>
> **摘要:** This paper proposes a multimodal emotion recognition system, VIsual Spoken Textual Additive Net (VISTANet), to classify emotions reflected by input containing image, speech, and text into discrete classes. A new interpretability technique, K-Average Additive exPlanation (KAAP), has been developed that identifies important visual, spoken, and textual features leading to predicting a particular emotion class. The VISTANet fuses information from image, speech, and text modalities using a hybrid of intermediate and late fusion. It automatically adjusts the weights of their intermediate outputs while computing the weighted average. The KAAP technique computes the contribution of each modality and corresponding features toward predicting a particular emotion class. To mitigate the insufficiency of multimodal emotion datasets labelled with discrete emotion classes, we have constructed the IIT-R MMEmoRec dataset consisting of images, corresponding speech and text, and emotion labels ('angry,' 'happy,' 'hate,' and 'sad'). The VISTANet has resulted in an overall emotion recognition accuracy of 80.11% on the IIT-R MMEmoRec dataset using visual, spoken, and textual modalities, outperforming single or dual-modality configurations. The code and data can be accessed at https://github.com/MIntelligence-Group/MMEmoRec.
>
---
#### [replaced 163] Pruning the Paradox: How CLIP's Most Informative Heads Enhance Performance While Amplifying Bias
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11103v2](http://arxiv.org/pdf/2503.11103v2)**

> **作者:** Avinash Madasu; Vasudev Lal; Phillip Howard
>
> **摘要:** CLIP is one of the most popular foundational models and is heavily used for many vision-language tasks. However, little is known about the inner workings of CLIP. While recent work has proposed decomposition-based interpretability methods for identifying textual descriptions of attention heads in CLIP, the implications of conceptual consistency in these text labels on interpretability and model performance has not been explored. To bridge this gap, we study the conceptual consistency of text descriptions for attention heads in CLIP-like models. We conduct extensive experiments on six different models from OpenAI and OpenCLIP which vary by size, type of pre-training data and patch size. We propose Concept Consistency Score (CCS), a novel interpretability metric that measures how consistently individual attention heads in CLIP models align with specific concepts. To assign concept labels to heads, we use in-context learning with ChatGPT, guided by a few manually-curated examples, and validate these labels using an LLM-as-a-judge approach. Our soft-pruning experiments reveal that high CCS heads are critical for preserving model performance, as pruning them leads to a significantly larger performance drop than pruning random or low CCS heads. Notably, we find that high CCS heads capture essential concepts and play a key role in out-of-domain detection, concept-specific reasoning, and video-language understanding. Moreover, we prove that high CCS heads learn spurious correlations amplifying social biases. These results position CCS as a powerful interpretability metric exposing the paradox of performance and social biases in CLIP models.
>
---
#### [replaced 164] Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12892v2](http://arxiv.org/pdf/2502.12892v2)**

> **作者:** Thomas Fel; Ekdeep Singh Lubana; Jacob S. Prince; Matthew Kowal; Victor Boutin; Isabel Papadimitriou; Binxu Wang; Martin Wattenberg; Demba Ba; Talia Konkle
>
> **摘要:** Sparse Autoencoders (SAEs) have emerged as a powerful framework for machine learning interpretability, enabling the unsupervised decomposition of model representations into a dictionary of abstract, human-interpretable concepts. However, we reveal a fundamental limitation: existing SAEs exhibit severe instability, as identical models trained on similar datasets can produce sharply different dictionaries, undermining their reliability as an interpretability tool. To address this issue, we draw inspiration from the Archetypal Analysis framework introduced by Cutler & Breiman (1994) and present Archetypal SAEs (A-SAE), wherein dictionary atoms are constrained to the convex hull of data. This geometric anchoring significantly enhances the stability of inferred dictionaries, and their mildly relaxed variants RA-SAEs further match state-of-the-art reconstruction abilities. To rigorously assess dictionary quality learned by SAEs, we introduce two new benchmarks that test (i) plausibility, if dictionaries recover "true" classification directions and (ii) identifiability, if dictionaries disentangle synthetic concept mixtures. Across all evaluations, RA-SAEs consistently yield more structured representations while uncovering novel, semantically meaningful concepts in large-scale vision models.
>
---
#### [replaced 165] Super-Resolution Generative Adversarial Networks based Video Enhancement
- **分类: cs.CV; cs.AI; eess.IV; I.4.3**

- **链接: [http://arxiv.org/pdf/2505.10589v3](http://arxiv.org/pdf/2505.10589v3)**

> **作者:** Kağan ÇETİN
>
> **备注:** 28 pages, 14 figures, 3 tables
>
> **摘要:** This study introduces an enhanced approach to video super-resolution by extending ordinary Single-Image Super-Resolution (SISR) Super-Resolution Generative Adversarial Network (SRGAN) structure to handle spatio-temporal data. While SRGAN has proven effective for single-image enhancement, its design does not account for the temporal continuity required in video processing. To address this, a modified framework that incorporates 3D Non-Local Blocks is proposed, which is enabling the model to capture relationships across both spatial and temporal dimensions. An experimental training pipeline is developed, based on patch-wise learning and advanced data degradation techniques, to simulate real-world video conditions and learn from both local and global structures and details. This helps the model generalize better and maintain stability across varying video content while maintaining the general structure besides the pixel-wise correctness. Two model variants-one larger and one more lightweight-are presented to explore the trade-offs between performance and efficiency. The results demonstrate improved temporal coherence, sharper textures, and fewer visual artifacts compared to traditional single-image methods. This work contributes to the development of practical, learning-based solutions for video enhancement tasks, with potential applications in streaming, gaming, and digital restoration.
>
---
#### [replaced 166] Leveraging Knowledge Graphs for Zero-Shot Object-agnostic State Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2307.12179v2](http://arxiv.org/pdf/2307.12179v2)**

> **作者:** Filipos Gouidis; Theodore Patkos; Antonis Argyros; Dimitris Plexousakis
>
> **备注:** This is the authors' version of the paper published at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025. The definitive version is available at: https://openaccess.thecvf.com/content/WACV2025/html/Gouidis_Recognizing_Unseen_States_of_Unknown_Objects_by_Leveraging_Knowledge_Graphs_WACV_2025_paper.html
>
> **摘要:** We investigate the problem of Object State Classification (OSC) as a zero-shot learning problem. Specifically, we propose the first Object-agnostic State Classification (OaSC) method that infers the state of a certain object without relying on the knowledge or the estimation of the object class. In that direction, we capitalize on Knowledge Graphs (KGs) for structuring and organizing knowledge, which, in combination with visual information, enable the inference of the states of objects in object/state pairs that have not been encountered in the method's training set. A series of experiments investigate the performance of the proposed method in various settings, against several hypotheses and in comparison with state of the art approaches for object attribute classification. The experimental results demonstrate that the knowledge of an object class is not decisive for the prediction of its state. Moreover, the proposed OaSC method outperforms existing methods in all datasets and benchmarks by a great margin.
>
---
#### [replaced 167] Exploring Generalized Gait Recognition: Reducing Redundancy and Noise within Indoor and Outdoor Datasets
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15176v3](http://arxiv.org/pdf/2505.15176v3)**

> **作者:** Qian Zhou; Xianda Guo; Jilong Wang; Chuanfu Shen; Zhongyuan Wang; Hua Zou; Qin Zou; Chao Liang; Long Chen; Gang Wu
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Generalized gait recognition, which aims to achieve robust performance across diverse domains, remains a challenging problem due to severe domain shifts in viewpoints, appearances, and environments. While mixed-dataset training is widely used to enhance generalization, it introduces new obstacles including inter-dataset optimization conflicts and redundant or noisy samples, both of which hinder effective representation learning. To address these challenges, we propose a unified framework that systematically improves cross-domain gait recognition. First, we design a disentangled triplet loss that isolates supervision signals across datasets, mitigating gradient conflicts during optimization. Second, we introduce a targeted dataset distillation strategy that filters out the least informative 20\% of training samples based on feature redundancy and prediction uncertainty, enhancing data efficiency. Extensive experiments on CASIA-B, OU-MVLP, Gait3D, and GREW demonstrate that our method significantly improves cross-dataset recognition for both GaitBase and DeepGaitV2 backbones, without sacrificing source-domain accuracy. Code will be released at https://github.com/li1er3/Generalized_Gait.
>
---
#### [replaced 168] FLASH: Latent-Aware Semi-Autoregressive Speculative Decoding for Multimodal Tasks
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.12728v2](http://arxiv.org/pdf/2505.12728v2)**

> **作者:** Zihua Wang; Ruibo Li; Haozhe Du; Joey Tianyi Zhou; Yu Zhang; Xu Yang
>
> **备注:** This preprint is under review
>
> **摘要:** Large language and multimodal models (LLMs and LMMs) exhibit strong inference capabilities but are often limited by slow decoding speeds. This challenge is especially acute in LMMs, where visual inputs typically comprise more tokens with lower information density than text -- an issue exacerbated by recent trends toward finer-grained visual tokenizations to boost performance. Speculative decoding has been effective in accelerating LLM inference by using a smaller draft model to generate candidate tokens, which are then selectively verified by the target model, improving speed without sacrificing output quality. While this strategy has been extended to LMMs, existing methods largely overlook the unique properties of visual inputs and depend solely on text-based draft models. In this work, we propose \textbf{FLASH} (Fast Latent-Aware Semi-Autoregressive Heuristics), a speculative decoding framework designed specifically for LMMs, which leverages two key properties of multimodal data to design the draft model. First, to address redundancy in visual tokens, we propose a lightweight latent-aware token compression mechanism. Second, recognizing that visual objects often co-occur within a scene, we employ a semi-autoregressive decoding strategy to generate multiple tokens per forward pass. These innovations accelerate draft decoding while maintaining high acceptance rates, resulting in faster overall inference. Experiments show that FLASH significantly outperforms prior speculative decoding approaches in both unimodal and multimodal settings, achieving up to \textbf{2.68$\times$} speed-up on video captioning and \textbf{2.55$\times$} on visual instruction tuning tasks compared to the original LMM. Our code is available \href{https://github.com/ZihuaEvan/FlashSD/}{[here]}.
>
---
#### [replaced 169] REAL: Representation Enhanced Analytic Learning for Exemplar-free Class-incremental Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.13522v2](http://arxiv.org/pdf/2403.13522v2)**

> **作者:** Run He; Di Fang; Yizhu Chen; Kai Tong; Cen Chen; Yi Wang; Lap-pui Chau; Huiping Zhuang
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Exemplar-free class-incremental learning (EFCIL) aims to mitigate catastrophic forgetting in class-incremental learning (CIL) without available historical training samples as exemplars. Compared with its exemplar-based CIL counterpart that stores exemplars, EFCIL suffers more from forgetting issues. Recently, a new EFCIL branch named Analytic Continual Learning (ACL) introduces a gradient-free paradigm via Recursive Least-Square, achieving a forgetting-resistant classifier training with a frozen backbone during CIL. However, existing ACL suffers from ineffective representations and insufficient utilization of backbone knowledge. In this paper, we propose a representation-enhanced analytic learning (REAL) to address these problems. To enhance the representation, REAL constructs a dual-stream base pretraining followed by representation enhancing distillation process. The dual-stream base pretraining combines self-supervised contrastive learning for general features and supervised learning for class-specific knowledge, followed by the representation enhancing distillation to merge both streams, enhancing representations for subsequent CIL paradigm. To utilize more knowledge from the backbone, REAL presents a feature fusion buffer to multi-layer backbone features, providing informative features for the subsequent classifier training. Our method can be incorporated into existing ACL techniques and provides more competitive performance. Empirical results demonstrate that, REAL achieves state-of-the-art performance on CIFAR-100, ImageNet-100 and ImageNet-1k benchmarks, outperforming exemplar-free methods and rivaling exemplar-based approaches.
>
---
#### [replaced 170] Diff-PCR: Diffusion-Based Correspondence Searching in Doubly Stochastic Matrix Space for Point Cloud Registration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.00436v5](http://arxiv.org/pdf/2401.00436v5)**

> **作者:** Haihua Shi; Qianliang Wu
>
> **摘要:** Efficiently finding optimal correspondences between point clouds is crucial for solving both rigid and non-rigid point cloud registration problems. Existing methods often rely on geometric or semantic feature embedding to establish correspondences and estimate transformations or flow fields. Recently, state-of-the-art methods have employed RAFT-like iterative updates to refine the solution. However, these methods have certain limitations. Firstly, their iterative refinement design lacks transparency, and their iterative updates follow a fixed path during the refinement process, which can lead to suboptimal results. Secondly, these methods overlook the importance of refining or optimizing correspondences (or matching matrices) as a precursor to solving transformations or flow fields. They typically compute candidate correspondences based on distances in the point feature space. However, they only project the candidate matching matrix into some matrix space once with Sinkhorn or dual softmax operations to obtain final correspondences. This one-shot projected matching matrix may be far from the globally optimal one, and these approaches do not consider the distribution of the target matching matrix. In this paper, we propose a novel approach that exploits the Denoising Diffusion Model to predict a searching gradient for the optimal matching matrix within the Doubly Stochastic Matrix Space. During the reverse denoising process, our method iteratively searches for better solutions along this denoising gradient, which points towards the maximum likelihood direction of the target matching matrix. Our method offers flexibility by allowing the search to start from any initial matching matrix provided by the online backbone or white noise. Experimental evaluations on the 3DMatch/3DLoMatch and 4DMatch/4DLoMatch datasets demonstrate the effectiveness of our newly designed framework.
>
---
#### [replaced 171] CauSkelNet: Causal Representation Learning for Human Behaviour Analysis
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.15564v3](http://arxiv.org/pdf/2409.15564v3)**

> **作者:** Xingrui Gu; Chuyi Jiang; Erte Wang; Zekun Wu; Qiang Cui; Leimin Tian; Lianlong Wu; Siyang Song; Chuang Yu
>
> **备注:** Accepted by 19th IEEE Automatic Face and Gesture Recognition 2025 (Oral)
>
> **摘要:** Traditional machine learning methods for movement recognition often struggle with limited model interpretability and a lack of insight into human movement dynamics. This study introduces a novel representation learning framework based on causal inference to address these challenges. Our two-stage approach combines the Peter-Clark (PC) algorithm and Kullback-Leibler (KL) divergence to identify and quantify causal relationships between human joints. By capturing joint interactions, the proposed causal Graph Convolutional Network (GCN) produces interpretable and robust representations. Experimental results on the EmoPain dataset demonstrate that the causal GCN outperforms traditional GCNs in accuracy, F1 score, and recall, particularly in detecting protective behaviors. This work contributes to advancing human motion analysis and lays a foundation for adaptive and intelligent healthcare solutions.
>
---
#### [replaced 172] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.18096v2](http://arxiv.org/pdf/2505.18096v2)**

> **作者:** Ziqiao Peng; Yanbo Fan; Haoyu Wu; Xuan Wang; Hongyan Liu; Jun He; Zhaoxin Fan
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** In face-to-face conversations, individuals need to switch between speaking and listening roles seamlessly. Existing 3D talking head generation models focus solely on speaking or listening, neglecting the natural dynamics of interactive conversation, which leads to unnatural interactions and awkward transitions. To address this issue, we propose a new task -- multi-round dual-speaker interaction for 3D talking head generation -- which requires models to handle and generate both speaking and listening behaviors in continuous conversation. To solve this task, we introduce DualTalk, a novel unified framework that integrates the dynamic behaviors of speakers and listeners to simulate realistic and coherent dialogue interactions. This framework not only synthesizes lifelike talking heads when speaking but also generates continuous and vivid non-verbal feedback when listening, effectively capturing the interplay between the roles. We also create a new dataset featuring 50 hours of multi-round conversations with over 1,000 characters, where participants continuously switch between speaking and listening roles. Extensive experiments demonstrate that our method significantly enhances the naturalness and expressiveness of 3D talking heads in dual-speaker conversations. We recommend watching the supplementary video: https://ziqiaopeng.github.io/dualtalk.
>
---
#### [replaced 173] TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding
- **分类: cs.AI; cs.CL; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.19400v2](http://arxiv.org/pdf/2502.19400v2)**

> **作者:** Max Ku; Thomas Chong; Jonathan Leung; Krish Shah; Alvin Yu; Wenhu Chen
>
> **备注:** accepted to ACL 2025 main, camera ready
>
> **摘要:** Understanding domain-specific theorems often requires more than just text-based reasoning; effective communication through structured visual explanations is crucial for deeper comprehension. While large language models (LLMs) demonstrate strong performance in text-based theorem reasoning, their ability to generate coherent and pedagogically meaningful visual explanations remains an open challenge. In this work, we introduce TheoremExplainAgent, an agentic approach for generating long-form theorem explanation videos (over 5 minutes) using Manim animations. To systematically evaluate multimodal theorem explanations, we propose TheoremExplainBench, a benchmark covering 240 theorems across multiple STEM disciplines, along with 5 automated evaluation metrics. Our results reveal that agentic planning is essential for generating detailed long-form videos, and the o3-mini agent achieves a success rate of 93.8% and an overall score of 0.77. However, our quantitative and qualitative studies show that most of the videos produced exhibit minor issues with visual element layout. Furthermore, multimodal explanations expose deeper reasoning flaws that text-based explanations fail to reveal, highlighting the importance of multimodal explanations.
>
---
#### [replaced 174] Multi-Level Embedding and Alignment Network with Consistency and Invariance Learning for Cross-View Geo-Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.14819v4](http://arxiv.org/pdf/2412.14819v4)**

> **作者:** Zhongwei Chen; Zhao-Xu Yang; Hai-Jun Rong
>
> **备注:** Accepted by TGRS 2025
>
> **摘要:** Cross-View Geo-Localization (CVGL) involves determining the localization of drone images by retrieving the most similar GPS-tagged satellite images. However, the imaging gaps between platforms are often significant and the variations in viewpoints are substantial, which limits the ability of existing methods to effectively associate cross-view features and extract consistent and invariant characteristics. Moreover, existing methods often overlook the problem of increased computational and storage requirements when improving model performance. To handle these limitations, we propose a lightweight enhanced alignment network, called the Multi-Level Embedding and Alignment Network (MEAN). The MEAN network uses a progressive multi-level enhancement strategy, global-to-local associations, and cross-domain alignment, enabling feature communication across levels. This allows MEAN to effectively connect features at different levels and learn robust cross-view consistent mappings and modality-invariant features. Moreover, MEAN adopts a shallow backbone network combined with a lightweight branch design, effectively reducing parameter count and computational complexity. Experimental results on the University-1652 and SUES-200 datasets demonstrate that MEAN reduces parameter count by 62.17% and computational complexity by 70.99% compared to state-of-the-art models, while maintaining competitive or even superior performance. Our code and models will be released on https://github.com/ISChenawei/MEAN.
>
---
#### [replaced 175] Seeing Through Deception: Uncovering Misleading Creator Intent in Multimodal News with Vision-Language Models
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.15489v2](http://arxiv.org/pdf/2505.15489v2)**

> **作者:** Jiaying Wu; Fanxiao Li; Min-Yen Kan; Bryan Hooi
>
> **摘要:** The real-world impact of misinformation stems from the underlying misleading narratives that creators seek to convey. As such, interpreting misleading creator intent is essential for multimodal misinformation detection (MMD) systems aimed at effective information governance. In this paper, we introduce an automated framework that simulates real-world multimodal news creation by explicitly modeling creator intent through two components: the desired influence and the execution plan. Using this framework, we construct DeceptionDecoded, a large-scale benchmark comprising 12,000 image-caption pairs aligned with trustworthy reference articles. The dataset captures both misleading and non-misleading intents and spans manipulations across visual and textual modalities. We conduct a comprehensive evaluation of 14 state-of-the-art vision-language models (VLMs) on three intent-centric tasks: (1) misleading intent detection, (2) misleading source attribution, and (3) creator desire inference. Despite recent advances, we observe that current VLMs fall short in recognizing misleading intent, often relying on spurious cues such as superficial cross-modal consistency, stylistic signals, and heuristic authenticity hints. Our findings highlight the pressing need for intent-aware modeling in MMD and open new directions for developing systems capable of deeper reasoning about multimodal misinformation.
>
---
#### [replaced 176] Paper Copilot Position: The Artificial Intelligence and Machine Learning Community Should Adopt a More Transparent and Regulated Peer Review Process
- **分类: cs.DL; cs.AI; cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.00874v2](http://arxiv.org/pdf/2502.00874v2)**

> **作者:** Jing Yang
>
> **备注:** ICML 2025; https://papercopilot.com/
>
> **摘要:** The rapid growth of submissions to top-tier Artificial Intelligence (AI) and Machine Learning (ML) conferences has prompted many venues to transition from closed to open review platforms. Some have fully embraced open peer reviews, allowing public visibility throughout the process, while others adopt hybrid approaches, such as releasing reviews only after final decisions or keeping reviews private despite using open peer review systems. In this work, we analyze the strengths and limitations of these models, highlighting the growing community interest in transparent peer review. To support this discussion, we examine insights from Paper Copilot, a website launched two years ago to aggregate and analyze AI / ML conference data while engaging a global audience. The site has attracted over 200,000 early-career researchers, particularly those aged 18-34 from 177 countries, many of whom are actively engaged in the peer review process. Drawing on our findings, this position paper advocates for a more transparent, open, and well-regulated peer review aiming to foster greater community involvement and propel advancements in the field.
>
---
#### [replaced 177] Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.16776v2](http://arxiv.org/pdf/2403.16776v2)**

> **作者:** Sophie Starck; Vasiliki Sideri-Lampretsa; Bernhard Kainz; Martin J. Menten; Tamara T. Mueller; Daniel Rueckert
>
> **摘要:** Anatomical atlases are widely used for population studies and analysis. Conditional atlases target a specific sub-population defined via certain conditions, such as demographics or pathologies, and allow for the investigation of fine-grained anatomical differences like morphological changes associated with ageing or disease. Existing approaches use either registration-based methods that are often unable to handle large anatomical variations or generative adversarial models, which are challenging to train since they can suffer from training instabilities. Instead of generating atlases directly in as intensities, we propose using latent diffusion models to generate deformation fields, which transform a general population atlas into one representing a specific sub-population. Our approach ensures structural integrity, enhances interpretability and avoids hallucinations that may arise during direct image synthesis by generating this deformation field and regularising it using a neighbourhood of images. We compare our method to several state-of-the-art atlas generation methods using brain MR images from the UK Biobank. Our method generates highly realistic atlases with smooth transformations and high anatomical fidelity, outperforming existing baselines. We demonstrate the quality of these atlases through comprehensive evaluations, including quantitative metrics for anatomical accuracy, perceptual similarity, and qualitative analyses displaying the consistency and realism of the generated atlases.
>
---
#### [replaced 178] Distilling Textual Priors from LLM to Efficient Image Fusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07029v3](http://arxiv.org/pdf/2504.07029v3)**

> **作者:** Ran Zhang; Xuanhua He; Ke Cao; Liu Liu; Li Zhang; Man Zhou; Jie Zhang
>
> **备注:** Change to TCSVT format
>
> **摘要:** Multi-modality image fusion aims to synthesize a single, comprehensive image from multiple source inputs. Traditional approaches, such as CNNs and GANs, offer efficiency but struggle to handle low-quality or complex inputs. Recent advances in text-guided methods leverage large model priors to overcome these limitations, but at the cost of significant computational overhead, both in memory and inference time. To address this challenge, we propose a novel framework for distilling large model priors, eliminating the need for text guidance during inference while dramatically reducing model size. Our framework utilizes a teacher-student architecture, where the teacher network incorporates large model priors and transfers this knowledge to a smaller student network via a tailored distillation process. Additionally, we introduce spatial-channel cross-fusion module to enhance the model's ability to leverage textual priors across both spatial and channel dimensions. Our method achieves a favorable trade-off between computational efficiency and fusion quality. The distilled network, requiring only 10% of the parameters and inference time of the teacher network, retains 90% of its performance and outperforms existing SOTA methods. Extensive experiments demonstrate the effectiveness of our approach. The implementation will be made publicly available as an open-source resource.
>
---
#### [replaced 179] Natural Language Generation from Visual Events: Challenges and Future Directions
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13034v2](http://arxiv.org/pdf/2502.13034v2)**

> **作者:** Aditya K Surikuchi; Raquel Fernández; Sandro Pezzelle
>
> **摘要:** The ability to use natural language to talk about visual events is at the core of human intelligence and a crucial feature of any artificial intelligence system. In recent years, a substantial body of work in visually grounded NLP has focused on describing content depicted in single images. By contrast, comparatively less attention has been devoted to exhaustively modeling scenarios in which natural language is employed to interpret and talk about events presented through videos or sequences of images. In this position paper, we argue that any NLG task dealing with sequences of images or frames is an instance of the broader, more general problem of modeling the intricate relationships between visual events unfolding over time and the features of the language used to interpret, describe, or narrate them. Therefore, solving these tasks requires models to be capable of identifying and managing such intricacies. We consider five seemingly different tasks, which we argue are compelling instances of this broader multimodal problem. Consistently, we claim that these tasks pose a common set of challenges and share similarities in terms of modeling and evaluation approaches. Building on this perspective, we identify key open questions and propose several research directions for future investigation. We claim that improving language-and-vision models' understanding of visual events is both timely and essential, given their growing applications. Additionally, this challenge offers significant scientific insight, advancing model development through principles of human cognition and language use.
>
---
#### [replaced 180] VERDI: VLM-Embedded Reasoning for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15925v2](http://arxiv.org/pdf/2505.15925v2)**

> **作者:** Bowen Feng; Zhiting Mei; Baiang Li; Julian Ost; Roger Girgis; Anirudha Majumdar; Felix Heide
>
> **摘要:** While autonomous driving (AD) stacks struggle with decision making under partial observability and real-world complexity, human drivers are capable of commonsense reasoning to make near-optimal decisions with limited information. Recent work has attempted to leverage finetuned Vision-Language Models (VLMs) for trajectory planning at inference time to emulate human behavior. Despite their success in benchmark evaluations, these methods are often impractical to deploy (a 70B parameter VLM inference at merely 8 tokens per second requires more than 160G of memory), and their monolithic network structure prohibits safety decomposition. To bridge this gap, we propose VLM-Embedded Reasoning for autonomous Driving (VERDI), a training-time framework that distills the reasoning process and commonsense knowledge of VLMs into the AD stack. VERDI augments modular differentiable end-to-end (e2e) AD models by aligning intermediate module outputs at the perception, prediction, and planning stages with text features explaining the driving reasoning process produced by VLMs. By encouraging alignment in latent space, VERDI enables the modular AD stack to internalize structured reasoning, without incurring the inference-time costs of large VLMs. We demonstrate the effectiveness of our method on the NuScenes dataset and find that VERDI outperforms existing e2e methods that do not embed reasoning by 10% in $\ell_{2}$ distance, while maintaining high inference speed.
>
---
#### [replaced 181] Dynamic Angle Selection in X-Ray CT: A Reinforcement Learning Approach to Optimal Stopping
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.12688v2](http://arxiv.org/pdf/2503.12688v2)**

> **作者:** Tianyuan Wang; Felix Lucka; Daniël M. Pelt; K. Joost Batenburg; Tristan van Leeuwen
>
> **摘要:** In industrial X-ray Computed Tomography (CT), the need for rapid in-line inspection is critical. Sparse-angle tomography plays a significant role in this by reducing the required number of projections, thereby accelerating processing and conserving resources. Most existing methods aim to balance reconstruction quality and scanning time, typically relying on fixed scan durations. Adaptive adjustment of the number of angles is essential; for instance, more angles may be required for objects with complex geometries or noisier projections. The concept of optimal stopping, which dynamically adjusts this balance according to varying industrial needs, remains overlooked. Building on our previous work, we integrate optimal stopping into sequential Optimal Experimental Design (sOED) and Reinforcement Learning (RL). We propose a novel method for computing the policy gradient within the Actor-Critic framework, enabling the development of adaptive policies for informative angle selection and scan termination. Additionally, we investigate the gap between simulation and real-world applications in the context of the developed learning-based method. Our trained model, developed using synthetic data, demonstrates reliable performance when applied to experimental X-ray CT data. This approach enhances the flexibility of CT operations and expands the applicability of sparse-angle tomography in industrial settings.
>
---
#### [replaced 182] Compile Scene Graphs with Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13617v4](http://arxiv.org/pdf/2504.13617v4)**

> **作者:** Zuyao Chen; Jinlin Wu; Zhen Lei; Marc Pollefeys; Chang Wen Chen
>
> **摘要:** Next-token prediction is the fundamental principle for training large language models (LLMs), and reinforcement learning (RL) further enhances their reasoning performance. As an effective way to model language, image, video, and other modalities, the use of LLMs for end-to-end extraction of structured visual representations, such as scene graphs, remains underexplored. It requires the model to accurately produce a set of objects and relationship triplets, rather than generating text token by token. To achieve this, we introduce R1-SGG, a multimodal LLM (M-LLM) initially trained via supervised fine-tuning (SFT) on the scene graph dataset and subsequently refined using reinforcement learning to enhance its ability to generate scene graphs in an end-to-end manner. The SFT follows a conventional prompt-response paradigm, while RL requires the design of effective reward signals. We design a set of graph-centric rewards, including three recall-based variants -- Hard Recall, Hard Recall+Relax, and Soft Recall -- which evaluate semantic and spatial alignment between predictions and ground truth at the object and relation levels. A format consistency reward further ensures that outputs follow the expected structural schema. Extensive experiments on the VG150 and PSG benchmarks show that R1-SGG substantially reduces failure rates and achieves strong performance in Recall and mean Recall, surpassing traditional SGG models and existing multimodal language models. Our code is available at https://github.com/gpt4vision/R1-SGG
>
---
#### [replaced 183] Describe Anything in Medical Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05804v2](http://arxiv.org/pdf/2505.05804v2)**

> **作者:** Xi Xiao; Yunbei Zhang; Thanh-Huy Nguyen; Ba-Thinh Lam; Janet Wang; Lin Zhao; Jihun Hamm; Tianyang Wang; Xingjian Li; Xiao Wang; Hao Xu; Tianming Liu; Min Xu
>
> **摘要:** Localized image captioning has made significant progress with models like the Describe Anything Model (DAM), which can generate detailed region-specific descriptions without explicit region-text supervision. However, such capabilities have yet to be widely applied to specialized domains like medical imaging, where diagnostic interpretation relies on subtle regional findings rather than global understanding. To mitigate this gap, we propose MedDAM, the first comprehensive framework leveraging large vision-language models for region-specific captioning in medical images. MedDAM employs medical expert-designed prompts tailored to specific imaging modalities and establishes a robust evaluation benchmark comprising a customized assessment protocol, data pre-processing pipeline, and specialized QA template library. This benchmark evaluates both MedDAM and other adaptable large vision-language models, focusing on clinical factuality through attribute-level verification tasks, thereby circumventing the absence of ground-truth region-caption pairs in medical datasets. Extensive experiments on the VinDr-CXR, LIDC-IDRI, and SkinCon datasets demonstrate MedDAM's superiority over leading peers (including GPT-4o, Claude 3.7 Sonnet, LLaMA-3.2 Vision, Qwen2.5-VL, GPT-4Rol, and OMG-LLaVA) in the task, revealing the importance of region-level semantic alignment in medical image understanding and establishing MedDAM as a promising foundation for clinical vision-language integration.
>
---
#### [replaced 184] OmniSVG: A Unified Scalable Vector Graphics Generation Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06263v2](http://arxiv.org/pdf/2504.06263v2)**

> **作者:** Yiying Yang; Wei Cheng; Sijin Chen; Xianfang Zeng; Fukun Yin; Jiaxu Zhang; Liao Wang; Gang Yu; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** 18 pages; Project Page: https://omnisvg.github.io/
>
> **摘要:** Scalable Vector Graphics (SVG) is an important image format widely adopted in graphic design because of their resolution independence and editability. The study of generating high-quality SVG has continuously drawn attention from both designers and researchers in the AIGC community. However, existing methods either produces unstructured outputs with huge computational cost or is limited to generating monochrome icons of over-simplified structures. To produce high-quality and complex SVG, we propose OmniSVG, a unified framework that leverages pre-trained Vision-Language Models (VLMs) for end-to-end multimodal SVG generation. By parameterizing SVG commands and coordinates into discrete tokens, OmniSVG decouples structural logic from low-level geometry for efficient training while maintaining the expressiveness of complex SVG structure. To further advance the development of SVG synthesis, we introduce MMSVG-2M, a multimodal dataset with two million richly annotated SVG assets, along with a standardized evaluation protocol for conditional SVG generation tasks. Extensive experiments show that OmniSVG outperforms existing methods and demonstrates its potential for integration into professional SVG design workflows.
>
---
#### [replaced 185] PillarHist: A Quantization-aware Pillar Feature Encoder based on Height-aware Histogram
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.18734v4](http://arxiv.org/pdf/2405.18734v4)**

> **作者:** Sifan Zhou; Zhihang Yuan; Dawei Yang; Ziyu Zhao; Jian Qian; Xing Hu
>
> **摘要:** Real-time and high-performance 3D object detection plays a critical role in autonomous driving and robotics. Recent pillar-based 3D object detectors have gained significant attention due to their compact representation and low computational overhead, making them suitable for onboard deployment and quantization. However, existing pillar-based detectors still suffer from information loss along height dimension and large numerical distribution difference during pillar feature encoding (PFE), which severely limits their performance and quantization potential. To address above issue, we first unveil the importance of different input information during PFE and identify the height dimension as a key factor in enhancing 3D detection performance. Motivated by this observation, we propose a height-aware pillar feature encoder, called PillarHist. Specifically, PillarHist statistics the discrete distribution of points at different heights within one pillar with the information entropy guidance. This simple yet effective design greatly preserves the information along the height dimension while significantly reducing the computation overhead of the PFE. Meanwhile, PillarHist also constrains the arithmetic distribution of PFE input to a stable range, making it quantization-friendly. Notably, PillarHist operates exclusively within the PFE stage to enhance performance, enabling seamless integration into existing pillar-based methods without introducing complex operations. Extensive experiments show the effectiveness of PillarHist in terms of both efficiency and performance.
>
---
#### [replaced 186] NFIG: Autoregressive Image Generation with Next-Frequency Prediction
- **分类: cs.CV; cs.AI; 68T07; I.2.10; I.2.6**

- **链接: [http://arxiv.org/pdf/2503.07076v2](http://arxiv.org/pdf/2503.07076v2)**

> **作者:** Zhihao Huang; Xi Qiu; Yukuo Ma; Yifu Zhou; Junjie Chen; Hongyuan Zhang; Chi Zhang; Xuelong Li
>
> **备注:** 10 pages, 7 figures, 2 tables
>
> **摘要:** Autoregressive models have achieved promising results in natural language processing. However, for image generation tasks, they encounter substantial challenges in effectively capturing long-range dependencies, managing computational costs, and most crucially, defining meaningful autoregressive sequences that reflect natural image hierarchies. To address these issues, we present \textbf{N}ext-\textbf{F}requency \textbf{I}mage \textbf{G}eneration (\textbf{NFIG}), a novel framework that decomposes the image generation process into multiple frequency-guided stages. Our approach first generates low-frequency components to establish global structure with fewer tokens, then progressively adds higher-frequency details, following the natural spectral hierarchy of images. This principled autoregressive sequence not only improves the quality of generated images by better capturing true causal relationships between image components, but also significantly reduces computational overhead during inference. Extensive experiments demonstrate that NFIG achieves state-of-the-art performance with fewer steps, offering a more efficient solution for image generation, with 1.25$\times$ speedup compared to VAR-d20 while achieving better performance (FID: 2.81) on the ImageNet-256 benchmark. We hope that our insight of incorporating frequency-domain knowledge to guide autoregressive sequence design will shed light on future research. We will make our code publicly available upon acceptance of the paper.
>
---
#### [replaced 187] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.14974v3](http://arxiv.org/pdf/2411.14974v3)**

> **作者:** Jan Held; Renaud Vandeghen; Abdullah Hamdi; Adrien Deliege; Anthony Cioppa; Silvio Giancola; Andrea Vedaldi; Bernard Ghanem; Marc Van Droogenbroeck
>
> **备注:** Accepted at CVPR 2025 as Highlight. 13 pages, 13 figures, 10 tables
>
> **摘要:** Recent advances in radiance field reconstruction, such as 3D Gaussian Splatting (3DGS), have achieved high-quality novel view synthesis and fast rendering by representing scenes with compositions of Gaussian primitives. However, 3D Gaussians present several limitations for scene reconstruction. Accurately capturing hard edges is challenging without significantly increasing the number of Gaussians, creating a large memory footprint. Moreover, they struggle to represent flat surfaces, as they are diffused in space. Without hand-crafted regularizers, they tend to disperse irregularly around the actual surface. To circumvent these issues, we introduce a novel method, named 3D Convex Splatting (3DCS), which leverages 3D smooth convexes as primitives for modeling geometrically-meaningful radiance fields from multi-view images. Smooth convex shapes offer greater flexibility than Gaussians, allowing for a better representation of 3D scenes with hard edges and dense volumes using fewer primitives. Powered by our efficient CUDA-based rasterizer, 3DCS achieves superior performance over 3DGS on benchmarks such as Mip-NeRF360, Tanks and Temples, and Deep Blending. Specifically, our method attains an improvement of up to 0.81 in PSNR and 0.026 in LPIPS compared to 3DGS while maintaining high rendering speeds and reducing the number of required primitives. Our results highlight the potential of 3D Convex Splatting to become the new standard for high-quality scene reconstruction and novel view synthesis. Project page: convexsplatting.github.io.
>
---
#### [replaced 188] DIAGen: Semantically Diverse Image Augmentation with Generative Models for Few-Shot Learning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.14584v2](http://arxiv.org/pdf/2408.14584v2)**

> **作者:** Tobias Lingenberg; Markus Reuter; Gopika Sudhakaran; Dominik Gojny; Stefan Roth; Simone Schaub-Meyer
>
> **备注:** Published in GCPR 2024
>
> **摘要:** Simple data augmentation techniques, such as rotations and flips, are widely used to enhance the generalization power of computer vision models. However, these techniques often fail to modify high-level semantic attributes of a class. To address this limitation, researchers have explored generative augmentation methods like the recently proposed DA-Fusion. Despite some progress, the variations are still largely limited to textural changes, thus falling short on aspects like varied viewpoints, environment, weather conditions, or even class-level semantic attributes (eg, variations in a dog's breed). To overcome this challenge, we propose DIAGen, building upon DA-Fusion. First, we apply Gaussian noise to the embeddings of an object learned with Textual Inversion to diversify generations using a pre-trained diffusion model's knowledge. Second, we exploit the general knowledge of a text-to-text generative model to guide the image generation of the diffusion model with varied class-specific prompts. Finally, we introduce a weighting mechanism to mitigate the impact of poorly generated samples. Experimental results across various datasets show that DIAGen not only enhances semantic diversity but also improves the performance of subsequent classifiers. The advantages of DIAGen over standard augmentations and the DA-Fusion baseline are particularly pronounced with out-of-distribution samples.
>
---
#### [replaced 189] OpenOmni: Advancing Open-Source Omnimodal Large Language Models with Progressive Multimodal Alignment and Real-Time Self-Aware Emotional Speech Synthesis
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.04561v5](http://arxiv.org/pdf/2501.04561v5)**

> **作者:** Run Luo; Ting-En Lin; Haonan Zhang; Yuchuan Wu; Xiong Liu; Min Yang; Yongbin Li; Longze Chen; Jiaming Li; Lei Zhang; Yangyi Chen; Xiaobo Xia; Hamid Alinejad-Rokny; Fei Huang
>
> **摘要:** Recent advancements in omnimodal learning have significantly improved understanding and generation across images, text, and speech, yet these developments remain predominantly confined to proprietary models. The lack of high-quality omnimodal datasets and the challenges of real-time emotional speech synthesis have notably hindered progress in open-source research. To address these limitations, we introduce \name, a two-stage training framework that integrates omnimodal alignment and speech generation to develop a state-of-the-art omnimodal large language model. In the alignment phase, a pre-trained speech model undergoes further training on text-image tasks, enabling (near) zero-shot generalization from vision to speech, outperforming models trained on tri-modal datasets. In the speech generation phase, a lightweight decoder is trained on speech tasks with direct preference optimization, enabling real-time emotional speech synthesis with high fidelity. Experiments show that \name surpasses state-of-the-art models across omnimodal, vision-language, and speech-language benchmarks. It achieves a 4-point absolute improvement on OmniBench over the leading open-source model VITA, despite using 5x fewer training samples and a smaller model size (7B vs. 7x8B). Additionally, \name achieves real-time speech generation with <1s latency at non-autoregressive mode, reducing inference time by 5x compared to autoregressive methods, and improves emotion classification accuracy by 7.7\%
>
---
#### [replaced 190] VR-Robo: A Real-to-Sim-to-Real Framework for Visual Robot Navigation and Locomotion
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01536v2](http://arxiv.org/pdf/2502.01536v2)**

> **作者:** Shaoting Zhu; Linzhan Mou; Derun Li; Baijun Ye; Runhan Huang; Hang Zhao
>
> **备注:** Project Page: https://vr-robo.github.io/
>
> **摘要:** Recent success in legged robot locomotion is attributed to the integration of reinforcement learning and physical simulators. However, these policies often encounter challenges when deployed in real-world environments due to sim-to-real gaps, as simulators typically fail to replicate visual realism and complex real-world geometry. Moreover, the lack of realistic visual rendering limits the ability of these policies to support high-level tasks requiring RGB-based perception like ego-centric navigation. This paper presents a Real-to-Sim-to-Real framework that generates photorealistic and physically interactive "digital twin" simulation environments for visual navigation and locomotion learning. Our approach leverages 3D Gaussian Splatting (3DGS) based scene reconstruction from multi-view images and integrates these environments into simulations that support ego-centric visual perception and mesh-based physical interactions. To demonstrate its effectiveness, we train a reinforcement learning policy within the simulator to perform a visual goal-tracking task. Extensive experiments show that our framework achieves RGB-only sim-to-real policy transfer. Additionally, our framework facilitates the rapid adaptation of robot policies with effective exploration capability in complex new environments, highlighting its potential for applications in households and factories.
>
---
#### [replaced 191] L2RSI: Cross-view LiDAR-based Place Recognition for Large-scale Urban Scenes via Remote Sensing Imagery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11245v2](http://arxiv.org/pdf/2503.11245v2)**

> **作者:** Ziwei Shi; Xiaoran Zhang; Yan Xia; Yu Zang; Siqi Shen; Cheng Wang
>
> **摘要:** We tackle the challenge of LiDAR-based place recognition, which traditionally depends on costly and time-consuming prior 3D maps. To overcome this, we first construct XA-L\&RSI dataset, which encompasses approximately $110,000$ remote sensing submaps and $13,000$ LiDAR point cloud submaps captured in urban scenes, and propose a novel method, L2RSI, for cross-view LiDAR place recognition using high-resolution Remote Sensing Imagery. This approach enables large-scale localization capabilities at a reduced cost by leveraging readily available overhead images as map proxies. L2RSI addresses the dual challenges of cross-view and cross-modal place recognition by learning feature alignment between point cloud submaps and remote sensing submaps in the semantic domain. Additionally, we introduce a novel probability propagation method based on particle estimation to refine position predictions, effectively leveraging temporal and spatial information. This approach enables large-scale retrieval and cross-scene generalization without fine-tuning. Extensive experiments on XA-L\&RSI demonstrate that, within a $100km^2$ retrieval range, L2RSI accurately localizes $83.27\%$ of point cloud submaps within a $30m$ radius for top-$1$ retrieved location. We provide a video to more vividly display the place recognition results of L2RSI at https://shizw695.github.io/L2RSI/.
>
---
#### [replaced 192] CompMarkGS: Robust Watermarking for Compressed 3D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12836v4](http://arxiv.org/pdf/2503.12836v4)**

> **作者:** Sumin In; Youngdong Jang; Utae Jeong; MinHyuk Jang; Hyeongcheol Park; Eunbyung Park; Sangpil Kim
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** 3D Gaussian Splatting (3DGS) enables rapid differentiable rendering for 3D reconstruction and novel view synthesis, leading to its widespread commercial use. Consequently, copyright protection via watermarking has become critical. However, because 3DGS relies on millions of Gaussians, which require gigabytes of storage, efficient transfer and storage require compression. Existing 3DGS watermarking methods are vulnerable to quantization-based compression, often resulting in the loss of the embedded watermark. To address this challenge, we propose a novel watermarking method that ensures watermark robustness after model compression while maintaining high rendering quality. In detail, we incorporate a quantization distortion layer that simulates compression during training, preserving the watermark under quantization-based compression. Also, we propose a learnable watermark embedding feature that embeds the watermark into the anchor feature, ensuring structural consistency and seamless integration into the 3D scene. Furthermore, we present a frequency-aware anchor growing mechanism to enhance image quality in high-frequency regions by effectively identifying Guassians within these regions. Experimental results confirm that our method preserves the watermark and maintains superior image quality under high compression, validating it as a promising approach for a secure 3DGS model.
>
---
#### [replaced 193] Generalizable Prompt Learning of CLIP: A Brief Overview
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01263v5](http://arxiv.org/pdf/2503.01263v5)**

> **作者:** Fangming Cui; Yonggang Zhang; Xuan Wang; Xule Wang; Liang Xiao
>
> **摘要:** Existing vision-language models (VLMs) such as CLIP have showcased an impressive capability to generalize well across various downstream tasks. These models leverage the synergy between visual and textual information, enabling them to understand and reason about the content present in images and text in a unified manner. This article provides a brief overview of CLIP based on few-shot prompt learning, including experimental data and technical characteristics of some methods. The purpose of this review is to provide a reference for researchers who have just started their research in generalizable prompting of CLIP through few-shot training for classification across 15 datasets and also to facilitate the integration of this field by researchers in other downstream tasks.
>
---
#### [replaced 194] CGI: Identifying Conditional Generative Models with Example Images
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.13991v2](http://arxiv.org/pdf/2501.13991v2)**

> **作者:** Zhi Zhou; Hao-Zhe Tan; Peng-Xiao Song; Lan-Zhe Guo
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** Generative models have achieved remarkable performance recently, and thus model hubs have emerged. Existing model hubs typically assume basic text matching is sufficient to search for models. However, in reality, due to different abstractions and the large number of models in model hubs, it is not easy for users to review model descriptions and example images, choosing which model best meets their needs. Therefore, it is necessary to describe model functionality wisely so that future users can efficiently search for the most suitable model for their needs. Efforts to address this issue remain limited. In this paper, we propose Conditional Generative Model Identification (CGI), which aims to provide an effective way to identify the most suitable model using user-provided example images rather than requiring users to manually review a large number of models with example images. To address this problem, we propose the PromptBased Model Identification (PMI) , which can adequately describe model functionality and precisely match requirements with specifications. To evaluate PMI approach and promote related research, we provide a benchmark comprising 65 models and 9100 identification tasks. Extensive experimental and human evaluation results demonstrate that PMI is effective. For instance, 92% of models are correctly identified with significantly better FID scores when four example images are provided.
>
---
#### [replaced 195] 3D Visual Illusion Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13061v3](http://arxiv.org/pdf/2505.13061v3)**

> **作者:** Chengtang Yao; Zhidan Liu; Jiaxi Zeng; Lidong Yu; Yuwei Wu; Yunde Jia
>
> **备注:** Project: https://github.com/YaoChengTang/3D-Visual-Illusion-Depth-Estimation
>
> **摘要:** 3D visual illusion is a perceptual phenomenon where a two-dimensional plane is manipulated to simulate three-dimensional spatial relationships, making a flat artwork or object look three-dimensional in the human visual system. In this paper, we reveal that the machine visual system is also seriously fooled by 3D visual illusions, including monocular and binocular depth estimation. In order to explore and analyze the impact of 3D visual illusion on depth estimation, we collect a large dataset containing almost 3k scenes and 200k images to train and evaluate SOTA monocular and binocular depth estimation methods. We also propose a robust depth estimation framework that uses common sense from a vision-language model to adaptively select reliable depth from binocular disparity and monocular depth. Experiments show that SOTA monocular, binocular, and multi-view depth estimation approaches are all fooled by various 3D visual illusions, while our method achieves SOTA performance.
>
---
#### [replaced 196] AW-GATCN: Adaptive Weighted Graph Attention Convolutional Network for Event Camera Data Joint Denoising and Object Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11232v2](http://arxiv.org/pdf/2505.11232v2)**

> **作者:** Haiyu Li; Charith Abhayaratne
>
> **备注:** Accepted to International Joint Conference on Neural Networks(IJCNN) 2025
>
> **摘要:** Event cameras, which capture brightness changes with high temporal resolution, inherently generate a significant amount of redundant and noisy data beyond essential object structures. The primary challenge in event-based object recognition lies in effectively removing this noise without losing critical spatial-temporal information. To address this, we propose an Adaptive Graph-based Noisy Data Removal framework for Event-based Object Recognition. Specifically, our approach integrates adaptive event segmentation based on normalized density analysis, a multifactorial edge-weighting mechanism, and adaptive graph-based denoising strategies. These innovations significantly enhance the integration of spatiotemporal information, effectively filtering noise while preserving critical structural features for robust recognition. Experimental evaluations on four challenging datasets demonstrate that our method achieves superior recognition accuracies of 83.77%, 76.79%, 99.30%, and 96.89%, surpassing existing graph-based methods by up to 8.79%, and improving noise reduction performance by up to 19.57%, with an additional accuracy gain of 6.26% compared to traditional Euclidean-based techniques.
>
---
#### [replaced 197] Graph Neural Networks for Knowledge Enhanced Visual Representation of Paintings
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2105.08190v2](http://arxiv.org/pdf/2105.08190v2)**

> **作者:** Athanasios Efthymiou; Stevan Rudinac; Monika Kackovic; Marcel Worring; Nachoem Wijnberg
>
> **备注:** Published in the 29th ACM International Conference on Multimedia (MM '21). This is the camera-ready version. 10 pages, 4 figures
>
> **摘要:** We propose ArtSAGENet, a novel multimodal architecture that integrates Graph Neural Networks (GNNs) and Convolutional Neural Networks (CNNs), to jointly learn visual and semantic-based artistic representations. First, we illustrate the significant advantages of multi-task learning for fine art analysis and argue that it is conceptually a much more appropriate setting in the fine art domain than the single-task alternatives. We further demonstrate that several GNN architectures can outperform strong CNN baselines in a range of fine art analysis tasks, such as style classification, artist attribution, creation period estimation, and tag prediction, while training them requires an order of magnitude less computational time and only a small amount of labeled data. Finally, through extensive experimentation we show that our proposed ArtSAGENet captures and encodes valuable relational dependencies between the artists and the artworks, surpassing the performance of traditional methods that rely solely on the analysis of visual content. Our findings underline a great potential of integrating visual content and semantics for fine art analysis and curation.
>
---
