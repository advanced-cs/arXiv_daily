# 计算机视觉 cs.CV

- **最新发布 120 篇**

- **更新 68 篇**

## 最新发布

#### [new 001] H3Former: Hypergraph-based Semantic-Aware Aggregation via Hyperbolic Hierarchical Contrastive Loss for Fine-Grained Visual Classification
- **分类: cs.CV; cs.AI**

- **简介: H3Former面向细粒度视觉分类，解决传统方法难以全面捕捉判别特征且冗余高的问题，提出基于双曲分层对比损失的超图语义聚合框架，实现更精准的区域特征建模与类别区分。**

- **链接: [https://arxiv.org/pdf/2511.10260v1](https://arxiv.org/pdf/2511.10260v1)**

> **作者:** Yongji Zhang; Siqi Li; Kuiyang Huang; Yue Gao; Yu Jiang
>
> **摘要:** Fine-Grained Visual Classification (FGVC) remains a challenging task due to subtle inter-class differences and large intra-class variations. Existing approaches typically rely on feature-selection mechanisms or region-proposal strategies to localize discriminative regions for semantic analysis. However, these methods often fail to capture discriminative cues comprehensively while introducing substantial category-agnostic redundancy. To address these limitations, we propose H3Former, a novel token-to-region framework that leverages high-order semantic relations to aggregate local fine-grained representations with structured region-level modeling. Specifically, we propose the Semantic-Aware Aggregation Module (SAAM), which exploits multi-scale contextual cues to dynamically construct a weighted hypergraph among tokens. By applying hypergraph convolution, SAAM captures high-order semantic dependencies and progressively aggregates token features into compact region-level representations. Furthermore, we introduce the Hyperbolic Hierarchical Contrastive Loss (HHCL), which enforces hierarchical semantic constraints in a non-Euclidean embedding space. The HHCL enhances inter-class separability and intra-class consistency while preserving the intrinsic hierarchical relationships among fine-grained categories. Comprehensive experiments conducted on four standard FGVC benchmarks validate the superiority of our H3Former framework.
>
---
#### [new 002] Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints
- **分类: cs.CV**

- **简介: 该论文面向语音伴随动作生成任务，解决局部旋转累积误差问题，提出GlobalDiff框架，首次在全局旋转空间进行扩散生成，并引入多级结构约束，显著提升动作稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.10076v1](https://arxiv.org/pdf/2511.10076v1)**

> **作者:** Xiangyue Zhang; Jianfang Li; Jianqiang Ren; Jiaxu Zhang
>
> **备注:** AAAI 2026
>
> **摘要:** Reliable co-speech motion generation requires precise motion representation and consistent structural priors across all joints. Existing generative methods typically operate on local joint rotations, which are defined hierarchically based on the skeleton structure. This leads to cumulative errors during generation, manifesting as unstable and implausible motions at end-effectors. In this work, we propose GlobalDiff, a diffusion-based framework that operates directly in the space of global joint rotations for the first time, fundamentally decoupling each joint's prediction from upstream dependencies and alleviating hierarchical error accumulation. To compensate for the absence of structural priors in global rotation space, we introduce a multi-level constraint scheme. Specifically, a joint structure constraint introduces virtual anchor points around each joint to better capture fine-grained orientation. A skeleton structure constraint enforces angular consistency across bones to maintain structural integrity. A temporal structure constraint utilizes a multi-scale variational encoder to align the generated motion with ground-truth temporal patterns. These constraints jointly regularize the global diffusion process and reinforce structural awareness. Extensive evaluations on standard co-speech benchmarks show that GlobalDiff generates smooth and accurate motions, improving the performance by 46.0 % compared to the current SOTA under multiple speaker identities.
>
---
#### [new 003] Robust Object Detection with Pseudo Labels from VLMs using Per-Object Co-teaching
- **分类: cs.CV**

- **简介: 该论文面向自动驾驶中的零样本目标检测任务，利用VLM生成伪标签，提出每目标协同教学策略，过滤噪声边界框，训练高效YOLO检测器，显著提升检测精度并降低对人工标注的依赖。**

- **链接: [https://arxiv.org/pdf/2511.09955v1](https://arxiv.org/pdf/2511.09955v1)**

> **作者:** Uday Bhaskar; Rishabh Bhattacharya; Avinash Patel; Sarthak Khoche; Praveen Anil Kulkarni; Naresh Manwani
>
> **摘要:** Foundation models, especially vision-language models (VLMs), offer compelling zero-shot object detection for applications like autonomous driving, a domain where manual labelling is prohibitively expensive. However, their detection latency and tendency to hallucinate predictions render them unsuitable for direct deployment. This work introduces a novel pipeline that addresses this challenge by leveraging VLMs to automatically generate pseudo-labels for training efficient, real-time object detectors. Our key innovation is a per-object co-teaching-based training strategy that mitigates the inherent noise in VLM-generated labels. The proposed per-object coteaching approach filters noisy bounding boxes from training instead of filtering the entire image. Specifically, two YOLO models learn collaboratively, filtering out unreliable boxes from each mini-batch based on their peers' per-object loss values. Overall, our pipeline provides an efficient, robust, and scalable approach to train high-performance object detectors for autonomous driving, significantly reducing reliance on costly human annotation. Experimental results on the KITTI dataset demonstrate that our method outperforms a baseline YOLOv5m model, achieving a significant mAP@0.5 boost ($31.12\%$ to $46.61\%$) while maintaining real-time detection latency. Furthermore, we show that supplementing our pseudo-labelled data with a small fraction of ground truth labels ($10\%$) leads to further performance gains, reaching $57.97\%$ mAP@0.5 on the KITTI dataset. We observe similar performance improvements for the ACDC and BDD100k datasets.
>
---
#### [new 004] Feature Quality and Adaptability of Medical Foundation Models: A Comparative Evaluation for Radiographic Classification and Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文评估医学与通用领域基础模型在胸片分类与分割任务中的特征质量，发现医学预训练显著提升性能，但对细微病变分割效果有限，且监督模型仍具竞争力，揭示了特征泛化能力的局限与架构关键性。**

- **链接: [https://arxiv.org/pdf/2511.09742v1](https://arxiv.org/pdf/2511.09742v1)**

> **作者:** Frank Li; Theo Dapamede; Mohammadreza Chavoshi; Young Seok Jeon; Bardia Khosravi; Abdulhameed Dere; Beatrice Brown-Mulry; Rohan Satya Isaac; Aawez Mansuri; Chiratidzo Sanyika; Janice Newsome; Saptarshi Purkayastha; Imon Banerjee; Hari Trivedi; Judy Gichoya
>
> **备注:** 7 figures, 3 tables
>
> **摘要:** Foundation models (FMs) promise to generalize medical imaging, but their effectiveness varies. It remains unclear how pre-training domain (medical vs. general), paradigm (e.g., text-guided), and architecture influence embedding quality, hindering the selection of optimal encoders for specific radiology tasks. To address this, we evaluate vision encoders from eight medical and general-domain FMs for chest X-ray analysis. We benchmark classification (pneumothorax, cardiomegaly) and segmentation (pneumothorax, cardiac boundary) using linear probing and fine-tuning. Our results show that domain-specific pre-training provides a significant advantage; medical FMs consistently outperformed general-domain models in linear probing, establishing superior initial feature quality. However, feature utility is highly task-dependent. Pre-trained embeddings were strong for global classification and segmenting salient anatomy (e.g., heart). In contrast, for segmenting complex, subtle pathologies (e.g., pneumothorax), all FMs performed poorly without significant fine-tuning, revealing a critical gap in localizing subtle disease. Subgroup analysis showed FMs use confounding shortcuts (e.g., chest tubes for pneumothorax) for classification, a strategy that fails for precise segmentation. We also found that expensive text-image alignment is not a prerequisite; image-only (RAD-DINO) and label-supervised (Ark+) FMs were among top performers. Notably, a supervised, end-to-end baseline remained highly competitive, matching or exceeding the best FMs on segmentation tasks. These findings show that while medical pre-training is beneficial, architectural choices (e.g., multi-scale) are critical, and pre-trained features are not universally effective, especially for complex localization tasks where supervised models remain a strong alternative.
>
---
#### [new 005] Equivariant Sampling for Improving Diffusion Model-based Image Restoration
- **分类: cs.CV**

- **简介: 该论文面向图像修复任务，针对现有扩散模型采样过程未充分利用先验的问题，提出EquS方法，通过双轨迹等变采样与时间步感知调度TAS，提升修复性能且不增计算开销。**

- **链接: [https://arxiv.org/pdf/2511.09965v1](https://arxiv.org/pdf/2511.09965v1)**

> **作者:** Chenxu Wu; Qingpeng Kong; Peiang Zhao; Wendi Yang; Wenxin Ma; Fenghe Tang; Zihang Jiang; S. Kevin Zhou
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Recent advances in generative models, especially diffusion models, have significantly improved image restoration (IR) performance. However, existing problem-agnostic diffusion model-based image restoration (DMIR) methods face challenges in fully leveraging diffusion priors, resulting in suboptimal performance. In this paper, we address the limitations of current problem-agnostic DMIR methods by analyzing their sampling process and providing effective solutions. We introduce EquS, a DMIR method that imposes equivariant information through dual sampling trajectories. To further boost EquS, we propose the Timestep-Aware Schedule (TAS) and introduce EquS$^+$. TAS prioritizes deterministic steps to enhance certainty and sampling efficiency. Extensive experiments on benchmarks demonstrate that our method is compatible with previous problem-agnostic DMIR methods and significantly boosts their performance without increasing computational costs. Our code is available at https://github.com/FouierL/EquS.
>
---
#### [new 006] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction
- **分类: cs.CV**

- **简介: 该论文针对稠密视频字幕生成任务，解决现有方法隐式建模时缺乏时序连贯性与语义完整性的问题，提出CACMI框架，通过跨模态帧聚合与上下文感知特征增强，显式建模时空语义，实现SOTA性能。**

- **链接: [https://arxiv.org/pdf/2511.10134v1](https://arxiv.org/pdf/2511.10134v1)**

> **作者:** Mingda Jia; Weiliang Meng; Zenghuang Fu; Yiheng Li; Qi Zeng; Yifan Zhang; Ju Xin; Rongtao Xu; Jiguang Zhang; Xiaopeng Zhang
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Dense video captioning jointly localizes and captions salient events in untrimmed videos. Recent methods primarily focus on leveraging additional prior knowledge and advanced multi-task architectures to achieve competitive performance. However, these pipelines rely on implicit modeling that uses frame-level or fragmented video features, failing to capture the temporal coherence across event sequences and comprehensive semantics within visual contexts. To address this, we propose an explicit temporal-semantic modeling framework called Context-Aware Cross-Modal Interaction (CACMI), which leverages both latent temporal characteristics within videos and linguistic semantics from text corpus. Specifically, our model consists of two core components: Cross-modal Frame Aggregation aggregates relevant frames to extract temporally coherent, event-aligned textual features through cross-modal retrieval; and Context-aware Feature Enhancement utilizes query-guided attention to integrate visual dynamics with pseudo-event semantics. Extensive experiments on the ActivityNet Captions and YouCook2 datasets demonstrate that CACMI achieves the state-of-the-art performance on dense video captioning task.
>
---
#### [new 007] Anomagic: Crossmodal Prompt-driven Zero-shot Anomaly Generation
- **分类: cs.CV; cs.AI**

- **简介: Anomagic提出一种零样本异常生成方法，通过跨模态提示融合视觉与文本信息，无需异常样本即可生成语义合理的异常，并借助对比优化提升生成精度，构建了首个通用异常生成基础模型。**

- **链接: [https://arxiv.org/pdf/2511.10020v1](https://arxiv.org/pdf/2511.10020v1)**

> **作者:** Yuxin Jiang; Wei Luo; Hui Zhang; Qiyu Chen; Haiming Yao; Weiming Shen; Yunkang Cao
>
> **摘要:** We propose Anomagic, a zero-shot anomaly generation method that produces semantically coherent anomalies without requiring any exemplar anomalies. By unifying both visual and textual cues through a crossmodal prompt encoding scheme, Anomagic leverages rich contextual information to steer an inpainting-based generation pipeline. A subsequent contrastive refinement strategy enforces precise alignment between synthesized anomalies and their masks, thereby bolstering downstream anomaly detection accuracy. To facilitate training, we introduce AnomVerse, a collection of 12,987 anomaly-mask-caption triplets assembled from 13 publicly available datasets, where captions are automatically generated by multimodal large language models using structured visual prompts and template-based textual hints. Extensive experiments demonstrate that Anomagic trained on AnomVerse can synthesize more realistic and varied anomalies than prior methods, yielding superior improvements in downstream anomaly detection. Furthermore, Anomagic can generate anomalies for any normal-category image using user-defined prompts, establishing a versatile foundation model for anomaly generation.
>
---
#### [new 008] Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts
- **分类: cs.CV; cs.CY**

- **简介: 该论文针对卫星图像中贫民窟检测的跨区域泛化难题，提出GRAM框架，利用混合专家模型结合测试时自适应，在无目标区域标签下实现高精度、可扩展的全球贫民窟识别。**

- **链接: [https://arxiv.org/pdf/2511.10300v1](https://arxiv.org/pdf/2511.10300v1)**

> **作者:** Sumin Lee; Sungwon Park; Jeasurk Yang; Jihee Kim; Meeyoung Cha
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Satellite-based slum segmentation holds significant promise in generating global estimates of urban poverty. However, the morphological heterogeneity of informal settlements presents a major challenge, hindering the ability of models trained on specific regions to generalize effectively to unseen locations. To address this, we introduce a large-scale high-resolution dataset and propose GRAM (Generalized Region-Aware Mixture-of-Experts), a two-phase test-time adaptation framework that enables robust slum segmentation without requiring labeled data from target regions. We compile a million-scale satellite imagery dataset from 12 cities across four continents for source training. Using this dataset, the model employs a Mixture-of-Experts architecture to capture region-specific slum characteristics while learning universal features through a shared backbone. During adaptation, prediction consistency across experts filters out unreliable pseudo-labels, allowing the model to generalize effectively to previously unseen regions. GRAM outperforms state-of-the-art baselines in low-resource settings such as African cities, offering a scalable and label-efficient solution for global slum mapping and data-driven urban planning.
>
---
#### [new 009] MuSc-V2: Zero-Shot Multimodal Industrial Anomaly Classification and Segmentation with Mutual Scoring of Unlabeled Samples
- **分类: cs.CV**

- **简介: MuSc-V2提出一种零样本多模态工业异常分类与分割方法，利用正常样本的高相似性与异常样本的孤立性，通过互评分机制融合2D/3D特征，显著提升检测精度，超越现有零样本及多数少样本方法。**

- **链接: [https://arxiv.org/pdf/2511.10047v1](https://arxiv.org/pdf/2511.10047v1)**

> **作者:** Xurui Li; Feng Xue; Yu Zhou
>
> **摘要:** Zero-shot anomaly classification (AC) and segmentation (AS) methods aim to identify and outline defects without using any labeled samples. In this paper, we reveal a key property that is overlooked by existing methods: normal image patches across industrial products typically find many other similar patches, not only in 2D appearance but also in 3D shapes, while anomalies remain diverse and isolated. To explicitly leverage this discriminative property, we propose a Mutual Scoring framework (MuSc-V2) for zero-shot AC/AS, which flexibly supports single 2D/3D or multimodality. Specifically, our method begins by improving 3D representation through Iterative Point Grouping (IPG), which reduces false positives from discontinuous surfaces. Then we use Similarity Neighborhood Aggregation with Multi-Degrees (SNAMD) to fuse 2D/3D neighborhood cues into more discriminative multi-scale patch features for mutual scoring. The core comprises a Mutual Scoring Mechanism (MSM) that lets samples within each modality to assign score to each other, and Cross-modal Anomaly Enhancement (CAE) that fuses 2D and 3D scores to recover modality-specific missing anomalies. Finally, Re-scoring with Constrained Neighborhood (RsCon) suppresses false classification based on similarity to more representative samples. Our framework flexibly works on both the full dataset and smaller subsets with consistently robust performance, ensuring seamless adaptability across diverse product lines. In aid of the novel framework, MuSc-V2 achieves significant performance improvements: a $\textbf{+23.7\%}$ AP gain on the MVTec 3D-AD dataset and a $\textbf{+19.3\%}$ boost on the Eyecandies dataset, surpassing previous zero-shot benchmarks and even outperforming most few-shot methods. The code will be available at The code will be available at \href{https://github.com/HUST-SLOW/MuSc-V2}{https://github.com/HUST-SLOW/MuSc-V2}.
>
---
#### [new 010] FOUND: Fourier-based von Mises Distribution for Robust Single Domain Generalization in Object Detection
- **分类: cs.CV**

- **简介: 该论文针对单域泛化目标检测任务，提出融合vMF分布与傅里叶变换的CLIP增强框架，通过建模方向特征与频域扰动提升模型跨域鲁棒性，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10352v1](https://arxiv.org/pdf/2511.10352v1)**

> **作者:** Mengzhu Wang; Changyuan Deng; Shanshan Wang; Nan Yin; Long Lan; Liang Yang
>
> **摘要:** Single Domain Generalization (SDG) for object detection aims to train a model on a single source domain that can generalize effectively to unseen target domains. While recent methods like CLIP-based semantic augmentation have shown promise, they often overlook the underlying structure of feature distributions and frequency-domain characteristics that are critical for robustness. In this paper, we propose a novel framework that enhances SDG object detection by integrating the von Mises-Fisher (vMF) distribution and Fourier transformation into a CLIP-guided pipeline. Specifically, we model the directional features of object representations using vMF to better capture domain-invariant semantic structures in the embedding space. Additionally, we introduce a Fourier-based augmentation strategy that perturbs amplitude and phase components to simulate domain shifts in the frequency domain, further improving feature robustness. Our method not only preserves the semantic alignment benefits of CLIP but also enriches feature diversity and structural consistency across domains. Extensive experiments on the diverse weather-driving benchmark demonstrate that our approach outperforms the existing state-of-the-art method.
>
---
#### [new 011] Right Looks, Wrong Reasons: Compositional Fidelity in Text-to-Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，揭示模型在逻辑组合（否定、计数、空间关系）上表现崩溃，归因于数据缺失、注意力架构不适用和评估偏差，指出需根本性改进表示与推理，而非渐进优化。**

- **链接: [https://arxiv.org/pdf/2511.10136v1](https://arxiv.org/pdf/2511.10136v1)**

> **作者:** Mayank Vatsa; Aparna Bharati; Richa Singh
>
> **备注:** Accepted in AAAI 2026
>
> **摘要:** The architectural blueprint of today's leading text-to-image models contains a fundamental flaw: an inability to handle logical composition. This survey investigates this breakdown across three core primitives-negation, counting, and spatial relations. Our analysis reveals a dramatic performance collapse: models that are accurate on single primitives fail precipitously when these are combined, exposing severe interference. We trace this failure to three key factors. First, training data show a near-total absence of explicit negations. Second, continuous attention architectures are fundamentally unsuitable for discrete logic. Third, evaluation metrics reward visual plausibility over constraint satisfaction. By analyzing recent benchmarks and methods, we show that current solutions and simple scaling cannot bridge this gap. Achieving genuine compositionality, we conclude, will require fundamental advances in representation and reasoning rather than incremental adjustments to existing architectures.
>
---
#### [new 012] Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉-语言模型微调中几何结构失真问题，提出Difference Vector Equalization（DiVE），通过约束预训练与微调模型的嵌入差向量一致，保留几何结构，在不牺牲零样本与分布外性能的前提下实现鲁棒微调。**

- **链接: [https://arxiv.org/pdf/2511.09973v1](https://arxiv.org/pdf/2511.09973v1)**

> **作者:** Satoshi Suzuki; Shin'ya Yamaguchi; Shoichiro Takeda; Taiga Yamane; Naoki Makishima; Naotaka Kawata; Mana Ihori; Tomohiro Tanaka; Shota Orihashi; Ryo Masumura
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Contrastive pre-trained vision-language models, such as CLIP, demonstrate strong generalization abilities in zero-shot classification by leveraging embeddings extracted from image and text encoders. This paper aims to robustly fine-tune these vision-language models on in-distribution (ID) data without compromising their generalization abilities in out-of-distribution (OOD) and zero-shot settings. Current robust fine-tuning methods tackle this challenge by reusing contrastive learning, which was used in pre-training, for fine-tuning. However, we found that these methods distort the geometric structure of the embeddings, which plays a crucial role in the generalization of vision-language models, resulting in limited OOD and zero-shot performance. To address this, we propose Difference Vector Equalization (DiVE), which preserves the geometric structure during fine-tuning. The idea behind DiVE is to constrain difference vectors, each of which is obtained by subtracting the embeddings extracted from the pre-trained and fine-tuning models for the same data sample. By constraining the difference vectors to be equal across various data samples, we effectively preserve the geometric structure. Therefore, we introduce two losses: average vector loss (AVL) and pairwise vector loss (PVL). AVL preserves the geometric structure globally by constraining difference vectors to be equal to their weighted average. PVL preserves the geometric structure locally by ensuring a consistent multimodal alignment. Our experiments demonstrate that DiVE effectively preserves the geometric structure, achieving strong results across ID, OOD, and zero-shot metrics.
>
---
#### [new 013] MonkeyOCR v1.5 Technical Report: Unlocking Robust Document Parsing for Complex Patterns
- **分类: cs.CV; cs.AI**

- **简介: MonkeyOCR v1.5提出一种视觉语言框架，解决复杂文档中布局混乱、跨页表格与图文混排的OCR难题，通过两阶段解析与强化学习提升识别精度，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10390v1](https://arxiv.org/pdf/2511.10390v1)**

> **作者:** Jiarui Zhang; Yuliang Liu; Zijun Wu; Guosheng Pang; Zhili Ye; Yupei Zhong; Junteng Ma; Tao Wei; Haiyang Xu; Weikai Chen; Zeen Wang; Qiangjun Ji; Fanxi Zhou; Qi Zhang; Yuanrui Hu; Jiahao Liu; Zhang Li; Ziyang Zhang; Qiang Liu; Xiang Bai
>
> **摘要:** Document parsing is a core task in document intelligence, supporting applications such as information extraction, retrieval-augmented generation, and automated document analysis. However, real-world documents often feature complex layouts with multi-level tables, embedded images or formulas, and cross-page structures, which remain challenging for existing OCR systems. We introduce MonkeyOCR v1.5, a unified vision-language framework that enhances both layout understanding and content recognition through a two-stage parsing pipeline. The first stage employs a large multimodal model to jointly predict document layout and reading order, leveraging visual information to ensure structural and sequential consistency. The second stage performs localized recognition of text, formulas, and tables within detected regions, maintaining high visual fidelity while reducing error propagation. To address complex table structures, we propose a visual consistency-based reinforcement learning scheme that evaluates recognition quality via render-and-compare alignment, improving structural accuracy without manual annotations. Additionally, two specialized modules, Image-Decoupled Table Parsing and Type-Guided Table Merging, are introduced to enable reliable parsing of tables containing embedded images and reconstruction of tables crossing pages or columns. Comprehensive experiments on OmniDocBench v1.5 demonstrate that MonkeyOCR v1.5 achieves state-of-the-art performance, outperforming PPOCR-VL and MinerU 2.5 while showing exceptional robustness in visually complex document scenarios.
>
---
#### [new 014] Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies
- **分类: cs.CV**

- **简介: 该论文针对LVLMs中ROPE导致的长程注意力衰减问题，提出三种推理阶段无训练的衰减恢复策略（T-DRS），提升模型对远距离上下文的记忆能力，在VQA任务上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.09868v1](https://arxiv.org/pdf/2511.09868v1)**

> **作者:** Peng Gao; Yujian Lee; Xiaofeng Zhang; Zailong Chen; Hui Zhang
>
> **备注:** Accepted in AAAI 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved impressive performance across a wide range of multimodal tasks. However, they still face critical challenges in modeling long-range dependencies under the usage of Rotary Positional Encoding (ROPE). Although it can facilitate precise modeling of token positions, it induces progressive attention decay as token distance increases, especially with progressive attention decay over distant token pairs, which severely impairs the model's ability to remember global context. To alleviate this issue, we propose inference-only Three-step Decay Resilience Strategies (T-DRS), comprising (1) Semantic-Driven DRS (SD-DRS), amplifying semantically meaningful but distant signals via content-aware residuals, (2) Distance-aware Control DRS (DC-DRS), which can purify attention by smoothly modulating weights based on positional distances, suppressing noise while preserving locality, and (3) re-Reinforce Distant DRS (reRD-DRS), consolidating the remaining informative remote dependencies to maintain global coherence. Together, the T-DRS recover suppressed long-range token pairs without harming local inductive biases. Extensive experiments on Vision Question Answering (VQA) benchmarks demonstrate that T-DRS can consistently improve performance in a training-free manner. The code can be accessed in https://github.com/labixiaoq-qq/Remember-me
>
---
#### [new 015] CLIP4VI-ReID: Learning Modality-shared Representations via CLIP Semantic Bridge for Visible-Infrared Person Re-identification
- **分类: cs.CV**

- **简介: 论文针对可见光-红外行人重识别（VI-ReID）任务，提出CLIP4VI-ReID，利用CLIP文本语义作为桥梁，通过文本生成、红外特征校正和语义对齐，实现跨模态共享表征学习，缓解模态差异问题，提升识别精度。**

- **链接: [https://arxiv.org/pdf/2511.10309v1](https://arxiv.org/pdf/2511.10309v1)**

> **作者:** Xiaomei Yang; Xizhan Gao; Sijie Niu; Fa Zhu; Guang Feng; Xiaofeng Qu; David Camacho
>
> **摘要:** This paper proposes a novel CLIP-driven modality-shared representation learning network named CLIP4VI-ReID for VI-ReID task, which consists of Text Semantic Generation (TSG), Infrared Feature Embedding (IFE), and High-level Semantic Alignment (HSA). Specifically, considering the huge gap in the physical characteristics between natural images and infrared images, the TSG is designed to generate text semantics only for visible images, thereby enabling preliminary visible-text modality alignment. Then, the IFE is proposed to rectify the feature embeddings of infrared images using the generated text semantics. This process injects id-related semantics into the shared image encoder, enhancing its adaptability to the infrared modality. Besides, with text serving as a bridge, it enables indirect visible-infrared modality alignment. Finally, the HSA is established to refine the high-level semantic alignment. This process ensures that the fine-tuned text semantics only contain id-related information, thereby achieving more accurate cross-modal alignment and enhancing the discriminability of the learned modal-shared representations. Extensive experimental results demonstrate that the proposed CLIP4VI-ReID achieves superior performance than other state-of-the-art methods on some widely used VI-ReID datasets.
>
---
#### [new 016] 3DFETUS: Standardizing Fetal Facial Planes in 3D Ultrasound
- **分类: cs.CV**

- **简介: 该论文针对胎儿面部超声平面获取困难的问题，提出GT++算法与3DFETUS模型，实现3D超声中标准面部平面的自动化精准定位，显著提升定位准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2511.10412v1](https://arxiv.org/pdf/2511.10412v1)**

> **作者:** Alomar Antonia; Rubio Ricardo; Albaiges Gerard; Salort-Benejam Laura; Caminal Julia; Prat Maria; Rueda Carolina; Cortes Berta; Piella Gemma; Sukno Federico
>
> **摘要:** Acquiring standard facial planes during routine fetal ultrasound (US) examinations is often challenging due to fetal movement, variability in orientation, and operator-dependent expertise. These factors contribute to inconsistencies, increased examination time, and potential diagnostic bias. To address these challenges in the context of facial assessment, we present: 1) GT++, a robust algorithm that estimates standard facial planes from 3D US volumes using annotated anatomical landmarks; and 2) 3DFETUS, a deep learning model that automates and standardizes their localization in 3D fetal US volumes. We evaluated our methods both qualitatively, through expert clinical review, and quantitatively. The proposed approach achieved a mean translation error of 4.13 mm and a mean rotation error of 7.93 degrees per plane, outperforming other state-of-the-art methods on 3D US volumes. Clinical assessments further confirmed the effectiveness of both GT++ and 3DFETUS, demonstrating statistically significant improvements in plane estimation accuracy.
>
---
#### [new 017] LoG3D: Ultra-High-Resolution 3D Shape Modeling via Local-to-Global Partitioning
- **分类: cs.CV**

- **简介: 论文提出LoG3D，一种基于无符号距离场（UDF）的3D变分自编码器，通过局部-全局分区架构实现超高分辨率（2048³）形状建模，解决传统方法在复杂拓扑和细节保留上的局限。**

- **链接: [https://arxiv.org/pdf/2511.10040v1](https://arxiv.org/pdf/2511.10040v1)**

> **作者:** Xinran Yang; Shuichang Lai; Jiangjing Lyu; Hongjie Li; Bowen Pan; Yuanqi Li; Jie Guo; Zhou Zhengkang; Yanwen Guo
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Generating high-fidelity 3D contents remains a fundamental challenge due to the complexity of representing arbitrary topologies-such as open surfaces and intricate internal structures-while preserving geometric details. Prevailing methods based on signed distance fields (SDFs) are hampered by costly watertight preprocessing and struggle with non-manifold geometries, while point-cloud representations often suffer from sampling artifacts and surface discontinuities. To overcome these limitations, we propose a novel 3D variational autoencoder (VAE) framework built upon unsigned distance fields (UDFs)-a more robust and computationally efficient representation that naturally handles complex and incomplete shapes. Our core innovation is a local-to-global (LoG) architecture that processes the UDF by partitioning it into uniform subvolumes, termed UBlocks. This architecture couples 3D convolutions for capturing local detail with sparse transformers for enforcing global coherence. A Pad-Average strategy further ensures smooth transitions at subvolume boundaries during reconstruction. This modular design enables seamless scaling to ultra-high resolutions up to 2048^3-a regime previously unattainable for 3D VAEs. Experiments demonstrate state-of-the-art performance in both reconstruction accuracy and generative quality, yielding superior surface smoothness and geometric flexibility.
>
---
#### [new 018] Utility of Pancreas Surface Lobularity as a CT Biomarker for Opportunistic Screening of Type 2 Diabetes
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于CT的全自动方法，利用胰腺表面小叶状结构（PSL）作为生物标志物，实现2型糖尿病的机遇性筛查。通过深度学习分割胰腺并构建预测模型，PSL显著区分糖尿病患者，AUC达0.90。**

- **链接: [https://arxiv.org/pdf/2511.10484v1](https://arxiv.org/pdf/2511.10484v1)**

> **作者:** Tejas Sudharshan Mathai; Anisa V. Prasad; Xinya Wang; Praveen T. S. Balamuralikrishna; Yan Zhuang; Abhinav Suri; Jianfei Liu; Perry J. Pickhardt; Ronald M. Summers
>
> **备注:** Submitted to IEEE ISBI 2026
>
> **摘要:** Type 2 Diabetes Mellitus (T2DM) is a chronic metabolic disease that affects millions of people worldwide. Early detection is crucial as it can alter pancreas function through morphological changes and increased deposition of ectopic fat, eventually leading to organ damage. While studies have shown an association between T2DM and pancreas volume and fat content, the role of increased pancreatic surface lobularity (PSL) in patients with T2DM has not been fully investigated. In this pilot work, we propose a fully automated approach to delineate the pancreas and other abdominal structures, derive CT imaging biomarkers, and opportunistically screen for T2DM. Four deep learning-based models were used to segment the pancreas in an internal dataset of 584 patients (297 males, 437 non-diabetic, age: 45$\pm$15 years). PSL was automatically detected and it was higher for diabetic patients (p=0.01) at 4.26 $\pm$ 8.32 compared to 3.19 $\pm$ 3.62 for non-diabetic patients. The PancAP model achieved the highest Dice score of 0.79 $\pm$ 0.17 and lowest ASSD error of 1.94 $\pm$ 2.63 mm (p$<$0.05). For predicting T2DM, a multivariate model trained with CT biomarkers attained 0.90 AUC, 66.7\% sensitivity, and 91.9\% specificity. Our results suggest that PSL is useful for T2DM screening and could potentially help predict the early onset of T2DM.
>
---
#### [new 019] Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对视觉语言模型中的对象幻觉问题，提出RUDDER框架，通过单次前向传播提取视觉残差方向，并用自适应门控机制低开销修正生成，实现高效抑制幻觉，兼顾效果与推理效率。**

- **链接: [https://arxiv.org/pdf/2511.10292v1](https://arxiv.org/pdf/2511.10292v1)**

> **作者:** Zhengtao Zou; Ya Gao; Jiarui Guan; Bin Li; Pekka Marttinen
>
> **备注:** Under review
>
> **摘要:** Large Vision-Language Models (LVLMs) often suffer from object hallucination, generating text inconsistent with visual inputs, which can critically undermine their reliability. Existing inference-time interventions to mitigate this issue present a challenging trade-off: while methods that steer internal states or adjust output logits can be effective, they often incur substantial computational overhead, typically requiring extra forward passes. This efficiency bottleneck can limit their practicality for real-world, latency-sensitive deployments. In this work, we aim to address this trade-off with Residual-Update Directed DEcoding Regulation (RUDDER), a low-overhead framework that steers LVLMs towards visually-grounded generation. RUDDER is built on two key innovations: (1) Contextual Activation Residual Direction (CARD) vector, a per-sample visual evidence vector extracted from the residual update of a self-attention layer during a single, standard forward pass. (2) A Bayesian-inspired adaptive gate that performs token-wise injection, applying a corrective signal whose strength is conditioned on the model's deviation from the visual context. Extensive experiments on key hallucination benchmarks, including POPE and CHAIR, indicate that RUDDER achieves performance comparable to state-of-the-art methods while introducing negligible computational latency, validating RUDDER as a pragmatic and effective approach for improving LVLMs' reliability without a significant compromise on efficiency.
>
---
#### [new 020] SliderEdit: Continuous Image Editing with Fine-Grained Instruction Control
- **分类: cs.CV**

- **简介: SliderEdit提出一种连续细粒度指令控制框架，解决现有图像编辑模型无法动态调整多指令强度的问题，通过全局训练的低秩适配矩阵，实现单滑块独立控制编辑强度，提升可解释性与交互性。**

- **链接: [https://arxiv.org/pdf/2511.09715v1](https://arxiv.org/pdf/2511.09715v1)**

> **作者:** Arman Zarei; Samyadeep Basu; Mobina Pournemat; Sayan Nag; Ryan Rossi; Soheil Feizi
>
> **摘要:** Instruction-based image editing models have recently achieved impressive performance, enabling complex edits to an input image from a multi-instruction prompt. However, these models apply each instruction in the prompt with a fixed strength, limiting the user's ability to precisely and continuously control the intensity of individual edits. We introduce SliderEdit, a framework for continuous image editing with fine-grained, interpretable instruction control. Given a multi-part edit instruction, SliderEdit disentangles the individual instructions and exposes each as a globally trained slider, allowing smooth adjustment of its strength. Unlike prior works that introduced slider-based attribute controls in text-to-image generation, typically requiring separate training or fine-tuning for each attribute or concept, our method learns a single set of low-rank adaptation matrices that generalize across diverse edits, attributes, and compositional instructions. This enables continuous interpolation along individual edit dimensions while preserving both spatial locality and global semantic consistency. We apply SliderEdit to state-of-the-art image editing models, including FLUX-Kontext and Qwen-Image-Edit, and observe substantial improvements in edit controllability, visual consistency, and user steerability. To the best of our knowledge, we are the first to explore and propose a framework for continuous, fine-grained instruction control in instruction-based image editing models. Our results pave the way for interactive, instruction-driven image manipulation with continuous and compositional control.
>
---
#### [new 021] Fragile by Design: On the Limits of Adversarial Defenses in Personalized Generation
- **分类: cs.CV**

- **简介: 该论文研究个性化生成中的隐私泄露问题，揭示现有对抗防御（如Anti-DreamBooth）易被简单滤波器清除，导致身份保护失效，提出新评估框架AntiDB_Purify，证明当前方法脆弱且不实用。**

- **链接: [https://arxiv.org/pdf/2511.10382v1](https://arxiv.org/pdf/2511.10382v1)**

> **作者:** Zhen Chen; Yi Zhang; Xiangyu Yin; Chengxuan Qin; Xingyu Zhao; Xiaowei Huang; Wenjie Ruan
>
> **摘要:** Personalized AI applications such as DreamBooth enable the generation of customized content from user images, but also raise significant privacy concerns, particularly the risk of facial identity leakage. Recent defense mechanisms like Anti-DreamBooth attempt to mitigate this risk by injecting adversarial perturbations into user photos to prevent successful personalization. However, we identify two critical yet overlooked limitations of these methods. First, the adversarial examples often exhibit perceptible artifacts such as conspicuous patterns or stripes, making them easily detectable as manipulated content. Second, the perturbations are highly fragile, as even a simple, non-learned filter can effectively remove them, thereby restoring the model's ability to memorize and reproduce user identity. To investigate this vulnerability, we propose a novel evaluation framework, AntiDB_Purify, to systematically evaluate existing defenses under realistic purification threats, including both traditional image filters and adversarial purification. Results reveal that none of the current methods maintains their protective effectiveness under such threats. These findings highlight that current defenses offer a false sense of security and underscore the urgent need for more imperceptible and robust protections to safeguard user identity in personalized generation.
>
---
#### [new 022] SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出SHRUG-FM框架，用于提升地球观测基础模型的可靠性，解决其在训练未覆盖环境中的性能下降问题。通过融合输入、嵌入空间的分布外检测与不确定性估计，识别并过滤低置信预测，揭示失败集中于特定地理区域。**

- **链接: [https://arxiv.org/pdf/2511.10370v1](https://arxiv.org/pdf/2511.10370v1)**

> **作者:** Kai-Hendrik Cohrs; Zuzanna Osika; Maria Gonzalez-Calabuig; Vishal Nedungadi; Ruben Cartuyvels; Steffen Knoblauch; Joppe Massant; Shruti Nath; Patrick Ebel; Vasileios Sitokonstantinou
>
> **摘要:** Geospatial foundation models for Earth observation often fail to perform reliably in environments underrepresented during pretraining. We introduce SHRUG-FM, a framework for reliability-aware prediction that integrates three complementary signals: out-of-distribution (OOD) detection in the input space, OOD detection in the embedding space and task-specific predictive uncertainty. Applied to burn scar segmentation, SHRUG-FM shows that OOD scores correlate with lower performance in specific environmental conditions, while uncertainty-based flags help discard many poorly performing predictions. Linking these flags to land cover attributes from HydroATLAS shows that failures are not random but concentrated in certain geographies, such as low-elevation zones and large river areas, likely due to underrepresentation in pretraining data. SHRUG-FM provides a pathway toward safer and more interpretable deployment of GFMs in climate-sensitive applications, helping bridge the gap between benchmark performance and real-world reliability.
>
---
#### [new 023] Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space
- **分类: cs.CV**

- **简介: 该论文提出Split-Layer，用于提升隐式神经表示（INR）的表达能力。通过将MLP分枝并用Hadamard积融合，构建高维特征空间，显著增强建模能力，同时避免计算开销激增，在图像、三维形状等任务中表现优越。**

- **链接: [https://arxiv.org/pdf/2511.10142v1](https://arxiv.org/pdf/2511.10142v1)**

> **作者:** Zhicheng Cai; Hao Zhu; Linsen Chen; Qiu Shen; Xun Cao
>
> **备注:** AAAI 2026
>
> **摘要:** Implicit neural representation (INR) models signals as continuous functions using neural networks, offering efficient and differentiable optimization for inverse problems across diverse disciplines. However, the representational capacity of INR defined by the range of functions the neural network can characterize, is inherently limited by the low-dimensional feature space in conventional multilayer perceptron (MLP) architectures. While widening the MLP can linearly increase feature space dimensionality, it also leads to a quadratic growth in computational and memory costs. To address this limitation, we propose the split-layer, a novel reformulation of MLP construction. The split-layer divides each layer into multiple parallel branches and integrates their outputs via Hadamard product, effectively constructing a high-degree polynomial space. This approach significantly enhances INR's representational capacity by expanding the feature space dimensionality without incurring prohibitive computational overhead. Extensive experiments demonstrate that the split-layer substantially improves INR performance, surpassing existing methods across multiple tasks, including 2D image fitting, 2D CT reconstruction, 3D shape representation, and 5D novel view synthesis.
>
---
#### [new 024] Benchmarking Diversity in Image Generation via Attribute-Conditional Human Evaluation
- **分类: cs.CV; cs.LG**

- **简介: 该论文面向文本到图像生成任务，解决模型输出多样性不足的问题，提出基于属性条件的人类评估框架，构建专用提示集与评估模板，通过统计检验比较模型多样性，实现可量化的多样性排名。**

- **链接: [https://arxiv.org/pdf/2511.10547v1](https://arxiv.org/pdf/2511.10547v1)**

> **作者:** Isabela Albuquerque; Ira Ktena; Olivia Wiles; Ivana Kajić; Amal Rannen-Triki; Cristina Vasconcelos; Aida Nematzadeh
>
> **摘要:** Despite advances in generation quality, current text-to-image (T2I) models often lack diversity, generating homogeneous outputs. This work introduces a framework to address the need for robust diversity evaluation in T2I models. Our framework systematically assesses diversity by evaluating individual concepts and their relevant factors of variation. Key contributions include: (1) a novel human evaluation template for nuanced diversity assessment; (2) a curated prompt set covering diverse concepts with their identified factors of variation (e.g. prompt: An image of an apple, factor of variation: color); and (3) a methodology for comparing models in terms of human annotations via binomial tests. Furthermore, we rigorously compare various image embeddings for diversity measurement. Notably, our principled approach enables ranking of T2I models by diversity, identifying categories where they particularly struggle. This research offers a robust methodology and insights, paving the way for improvements in T2I model diversity and metric development.
>
---
#### [new 025] STORM: Segment, Track, and Object Re-Localization from a Single 3D Model
- **分类: cs.CV**

- **简介: STORM提出一种无需人工标注的实时6D位姿估计系统，通过视觉-语言理解与自监督特征匹配，实现单3D模型下的目标分割、跟踪与重定位，有效应对遮挡与高速运动问题。**

- **链接: [https://arxiv.org/pdf/2511.09771v1](https://arxiv.org/pdf/2511.09771v1)**

> **作者:** Yu Deng; Teng Cao; Hikaru Shindo; Jiahong Xue; Quentin Delfosse; Kristian Kersting
>
> **摘要:** Accurate 6D pose estimation and tracking are fundamental capabilities for physical AI systems such as robots. However, existing approaches typically rely on a manually annotated segmentation mask of the target in the first frame, which is labor-intensive and leads to reduced performance when faced with occlusions or rapid movement. To address these limi- tations, we propose STORM (Segment, Track, and Object Re-localization from a single 3D Model), an open-source robust real-time 6D pose estimation system that requires no manual annotation. STORM employs a novel three-stage pipeline combining vision-language understanding with self-supervised feature matching: contextual object descriptions guide localization, self-cross-attention mechanisms identify candidate regions, and a segmentation model produces precise masks for accurate pose estimation. Another key innovation is our automatic re-registration mechanism that detects tracking failures through feature similarity monitoring and recovers from severe occlusions or rapid motion. STORM achieves state-of-the-art accuracy on challenging industrial datasets featuring multi-object occlusions, high-speed motion, and varying illumination, while operating at real-time speeds without additional training. This annotation-free approach significantly reduces deployment overhead, providing a practical solution for modern applications, such as flexible manufacturing and intelligent quality control.
>
---
#### [new 026] GrounDiff: Diffusion-Based Ground Surface Generation from Digital Surface Models
- **分类: cs.CV**

- **简介: 论文提出GrounDiff，首个基于扩散模型的DSM到DTM生成框架，通过去噪迭代移除非地面结构，结合置信度引导与先验拼接技术，显著提升地面提取精度与平滑性，超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10391v1](https://arxiv.org/pdf/2511.10391v1)**

> **作者:** Oussema Dhaouadi; Johannes Meier; Jacques Kaiser; Daniel Cremers
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** Digital Terrain Models (DTMs) represent the bare-earth elevation and are important in numerous geospatial applications. Such data models cannot be directly measured by sensors and are typically generated from Digital Surface Models (DSMs) derived from LiDAR or photogrammetry. Traditional filtering approaches rely on manually tuned parameters, while learning-based methods require well-designed architectures, often combined with post-processing. To address these challenges, we introduce Ground Diffusion (GrounDiff), the first diffusion-based framework that iteratively removes non-ground structures by formulating the problem as a denoising task. We incorporate a gated design with confidence-guided generation that enables selective filtering. To increase scalability, we further propose Prior-Guided Stitching (PrioStitch), which employs a downsampled global prior automatically generated using GrounDiff to guide local high-resolution predictions. We evaluate our method on the DSM-to-DTM translation task across diverse datasets, showing that GrounDiff consistently outperforms deep learning-based state-of-the-art methods, reducing RMSE by up to 93% on ALS2DTM and up to 47% on USGS benchmarks. In the task of road reconstruction, which requires both high precision and smoothness, our method achieves up to 81% lower distance error compared to specialized techniques on the GeRoD benchmark, while maintaining competitive surface smoothness using only DSM inputs, without task-specific optimization. Our variant for road reconstruction, GrounDiff+, is specifically designed to produce even smoother surfaces, further surpassing state-of-the-art methods. The project page is available at https://deepscenario.github.io/GrounDiff/.
>
---
#### [new 027] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding
- **分类: cs.CV**

- **简介: 论文针对弱监督时空视频定位（STVG）任务，解决传统方法因文本与轨迹独立生成导致的识别与跟踪不一致问题。提出TubeRMC框架，通过文本条件生成候选轨迹，并利用轨迹反演重建查询线索，引入多维互约束提升精度。**

- **链接: [https://arxiv.org/pdf/2511.10241v1](https://arxiv.org/pdf/2511.10241v1)**

> **作者:** Jinxuan Li; Yi Zhang; Jian-Fang Hu; Chaolei Tan; Tianming Liang; Beihao Xia
>
> **摘要:** Spatio-Temporal Video Grounding (STVG) aims to localize a spatio-temporal tube that corresponds to a given language query in an untrimmed video. This is a challenging task since it involves complex vision-language understanding and spatiotemporal reasoning. Recent works have explored weakly-supervised setting in STVG to eliminate reliance on fine-grained annotations like bounding boxes or temporal stamps. However, they typically follow a simple late-fusion manner, which generates tubes independent of the text description, often resulting in failed target identification and inconsistent target tracking. To address this limitation, we propose a Tube-conditioned Reconstruction with Mutual Constraints (\textbf{TubeRMC}) framework that generates text-conditioned candidate tubes with pre-trained visual grounding models and further refine them via tube-conditioned reconstruction with spatio-temporal constraints. Specifically, we design three reconstruction strategies from temporal, spatial, and spatio-temporal perspectives to comprehensively capture rich tube-text correspondences. Each strategy is equipped with a Tube-conditioned Reconstructor, utilizing spatio-temporal tubes as condition to reconstruct the key clues in the query. We further introduce mutual constraints between spatial and temporal proposals to enhance their quality for reconstruction. TubeRMC outperforms existing methods on two public benchmarks VidSTG and HCSTVG. Further visualization shows that TubeRMC effectively mitigates both target identification errors and inconsistent tracking.
>
---
#### [new 028] Enhancing the Outcome Reward-based RL Training of MLLMs with Self-Consistency Sampling
- **分类: cs.CV**

- **简介: 该论文针对多模态大模型基于结果奖励的强化学习中“错误推理获高奖”问题，提出自一致性采样（SCS），通过视觉扰动与轨迹一致性评分过滤不可靠推理，显著提升模型准确率。**

- **链接: [https://arxiv.org/pdf/2511.10648v1](https://arxiv.org/pdf/2511.10648v1)**

> **作者:** Jiahao Wang; Weiye Xu; Aijun Yang; Wengang Zhou; Lewei Lu; Houqiang Li; Xiaohua Wang; Jinguo Zhu
>
> **备注:** Accepted to NeurIPS 2025 (The Thirty-Ninth Annual Conference on Neural Information Processing Systems)
>
> **摘要:** Outcome-reward reinforcement learning (RL) is a common and increasingly significant way to refine the step-by-step reasoning of multimodal large language models (MLLMs). In the multiple-choice setting - a dominant format for multimodal reasoning benchmarks - the paradigm faces a significant yet often overlooked obstacle: unfaithful trajectories that guess the correct option after a faulty chain of thought receive the same reward as genuine reasoning, which is a flaw that cannot be ignored. We propose Self-Consistency Sampling (SCS) to correct this issue. For each question, SCS (i) introduces small visual perturbations and (ii) performs repeated truncation and resampling of an initial trajectory; agreement among the resulting trajectories yields a differentiable consistency score that down-weights unreliable traces during policy updates. Based on Qwen2.5-VL-7B-Instruct, plugging SCS into RLOO, GRPO, and REINFORCE++ series improves accuracy by up to 7.7 percentage points on six multimodal benchmarks with negligible extra computation. SCS also yields notable gains on both Qwen2.5-VL-3B-Instruct and InternVL3-8B, offering a simple, general remedy for outcome-reward RL in MLLMs.
>
---
#### [new 029] Depth-Consistent 3D Gaussian Splatting via Physical Defocus Modeling and Multi-View Geometric Supervision
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对3D高斯溅射中深浅区域深度不一致问题，融合物理景深建模与多视角几何约束，通过景深损失和特征匹配优化，提升远近区域深度精度，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10316v1](https://arxiv.org/pdf/2511.10316v1)**

> **作者:** Yu Deng; Baozhu Zhao; Junyan Su; Xiaohan Zhang; Qi Liu
>
> **摘要:** Three-dimensional reconstruction in scenes with extreme depth variations remains challenging due to inconsistent supervisory signals between near-field and far-field regions. Existing methods fail to simultaneously address inaccurate depth estimation in distant areas and structural degradation in close-range regions. This paper proposes a novel computational framework that integrates depth-of-field supervision and multi-view consistency supervision to advance 3D Gaussian Splatting. Our approach comprises two core components: (1) Depth-of-field Supervision employs a scale-recovered monocular depth estimator (e.g., Metric3D) to generate depth priors, leverages defocus convolution to synthesize physically accurate defocused images, and enforces geometric consistency through a novel depth-of-field loss, thereby enhancing depth fidelity in both far-field and near-field regions; (2) Multi-View Consistency Supervision employing LoFTR-based semi-dense feature matching to minimize cross-view geometric errors and enforce depth consistency via least squares optimization of reliable matched points. By unifying defocus physics with multi-view geometric constraints, our method achieves superior depth fidelity, demonstrating a 0.8 dB PSNR improvement over the state-of-the-art method on the Waymo Open Dataset. This framework bridges physical imaging principles and learning-based depth regularization, offering a scalable solution for complex depth stratification in urban environments.
>
---
#### [new 030] Soiling detection for Advanced Driver Assistance Systems
- **分类: cs.CV; cs.AI**

- **简介: 该论文将车载摄像头污损检测建模为语义分割任务，对比了多种分割方法与分类方法的性能，并发现Woodscape数据集存在数据泄露和标注不准确问题，为此构建了更优的子集，显著提升训练效率。**

- **链接: [https://arxiv.org/pdf/2511.09740v1](https://arxiv.org/pdf/2511.09740v1)**

> **作者:** Filip Beránek; Václav Diviš; Ivan Gruber
>
> **备注:** Published at ICMV 2024
>
> **摘要:** Soiling detection for automotive cameras is a crucial part of advanced driver assistance systems to make them more robust to external conditions like weather, dust, etc. In this paper, we regard the soiling detection as a semantic segmentation problem. We provide a comprehensive comparison of popular segmentation methods and show their superiority in performance while comparing them to tile-level classification approaches. Moreover, we present an extensive analysis of the Woodscape dataset showing that the original dataset contains a data-leakage and imprecise annotations. To address these problems, we create a new data subset, which, despite being much smaller, provides enough information for the segmentation method to reach comparable results in a much shorter time. All our codes and dataset splits are available at https://github.com/filipberanek/woodscape_revision.
>
---
#### [new 031] Image Aesthetic Reasoning via HCM-GRPO: Empowering Compact Model for Superior Performance
- **分类: cs.CV**

- **简介: 该论文聚焦图像美学推理任务，解决MLLMs因数据匮乏与能力不足导致的性能低下问题。提出128k样本数据集与HCM-GRPO方法，仅用小模型即超越主流大模型性能。**

- **链接: [https://arxiv.org/pdf/2511.10055v1](https://arxiv.org/pdf/2511.10055v1)**

> **作者:** Zhiyuan Hu; Zheng Sun; Yi Wei; Long Yu
>
> **摘要:** The performance of image generation has been significantly improved in recent years. However, the study of image screening is rare and its performance with Multimodal Large Language Models (MLLMs) is unsatisfactory due to the lack of data and the weak image aesthetic reasoning ability in MLLMs. In this work, we propose a complete solution to address these problems in terms of data and methodology. For data, we collect a comprehensive image screening dataset with over 128k samples, about 640k images. Each sample consists of an original image, four generated images. The dataset evaluates the image aesthetic reasoning ability under four aspects: appearance deformation, physical shadow, placement layout, and extension rationality. Regarding data annotation, we investigate multiple approaches, including purely manual, fully automated, and answer-driven annotations, to acquire high-quality chains of thought (CoT) data in the most cost-effective manner. Methodologically, we introduce a Hard Cases Mining (HCM) strategy with a Dynamic Proportional Accuracy (DPA) reward into the Group Relative Policy Optimization (GRPO) framework, called HCM-GRPO. This enhanced method demonstrates superior image aesthetic reasoning capabilities compared to the original GRPO. Our experimental results reveal that even state-of-the-art closed-source MLLMs, such as GPT4o and Qwen-VL-Max, exhibit performance akin to random guessing in image aesthetic reasoning. In contrast, by leveraging the HCM-GRPO, we are able to surpass the scores of both large-scale open-source and leading closed-source models with a much smaller model.
>
---
#### [new 032] MOBA: A Material-Oriented Backdoor Attack against LiDAR-based 3D Object Detection Systems
- **分类: cs.CV**

- **简介: 该论文提出MOBA，一种面向材料的物理后门攻击方法，解决LiDAR三维目标检测系统中数字触发器难以物理实现的问题。通过选材TiO₂与改进BRDF仿真，实现高鲁棒性、高成功率（93.5%）的物理后门攻击。**

- **链接: [https://arxiv.org/pdf/2511.09999v1](https://arxiv.org/pdf/2511.09999v1)**

> **作者:** Saket S. Chaturvedi; Gaurav Bagwe; Lan Zhang; Pan He; Xiaoyong Yuan
>
> **备注:** Accepted at AAAI 2026 Conference
>
> **摘要:** LiDAR-based 3D object detection is widely used in safety-critical systems. However, these systems remain vulnerable to backdoor attacks that embed hidden malicious behaviors during training. A key limitation of existing backdoor attacks is their lack of physical realizability, primarily due to the digital-to-physical domain gap. Digital triggers often fail in real-world settings because they overlook material-dependent LiDAR reflection properties. On the other hand, physically constructed triggers are often unoptimized, leading to low effectiveness or easy detectability.This paper introduces Material-Oriented Backdoor Attack (MOBA), a novel framework that bridges the digital-physical gap by explicitly modeling the material properties of real-world triggers. MOBA tackles two key challenges in physical backdoor design: 1) robustness of the trigger material under diverse environmental conditions, 2) alignment between the physical trigger's behavior and its digital simulation. First, we propose a systematic approach to selecting robust trigger materials, identifying titanium dioxide (TiO_2) for its high diffuse reflectivity and environmental resilience. Second, to ensure the digital trigger accurately mimics the physical behavior of the material-based trigger, we develop a novel simulation pipeline that features: (1) an angle-independent approximation of the Oren-Nayar BRDF model to generate realistic LiDAR intensities, and (2) a distance-aware scaling mechanism to maintain spatial consistency across varying depths. We conduct extensive experiments on state-of-the-art LiDAR-based and Camera-LiDAR fusion models, showing that MOBA achieves a 93.50% attack success rate, outperforming prior methods by over 41%. Our work reveals a new class of physically realizable threats and underscores the urgent need for defenses that account for material-level properties in real-world environments.
>
---
#### [new 033] PROPA: Toward Process-level Optimization in Visual Reasoning via Reinforcement Learning
- **分类: cs.CV**

- **简介: PROPA针对视觉推理中多步错误累积问题，提出一种无需人工标注的强化学习框架，结合MCTS与GRPO生成过程级奖励，优化中间推理步骤，显著提升模型在在域与跨域任务中的推理与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.10279v1](https://arxiv.org/pdf/2511.10279v1)**

> **作者:** Yanbei Jiang; Chao Lei; Yihao Ding; Krista Ehinger; Jey Han Lau
>
> **摘要:** Despite significant progress, Vision-Language Models (VLMs) still struggle with complex visual reasoning, where multi-step dependencies cause early errors to cascade through the reasoning chain. Existing post-training paradigms are limited: Supervised Fine-Tuning (SFT) relies on costly step-level annotations, while Reinforcement Learning with Verifiable Rewards (RLVR) methods like GRPO provide only sparse, outcome-level feedback, hindering stable optimization. We introduce PROPA (Process-level Reasoning Optimization with interleaved Policy Alignment), a novel framework that integrates Monte Carlo Tree Search (MCTS) with GRPO to generate dense, process-level rewards and optimize reasoning at each intermediate step without human annotations. To overcome the cold-start problem, PROPA interleaves GRPO updates with SFT, enabling the model to learn from both successful and failed reasoning trajectories. A Process Reward Model (PRM) is further trained to guide inference-time search, aligning the test-time search with the training signal. Across seven benchmarks and four VLM backbones, PROPA consistently outperforms both SFT- and RLVR-based baselines. It achieves up to 17.0% gains on in-domain tasks and 21.0% gains on out-of-domain tasks compared to existing state-of-the-art, establishing a strong reasoning and generalization capability for visual reasoning tasks. The code isavailable at: https://github.com/YanbeiJiang/PROPA.
>
---
#### [new 034] FineSkiing: A Fine-grained Benchmark for Skiing Action Quality Assessment
- **分类: cs.CV; cs.AI; cs.HC**

- **简介: 该论文面向滑雪动作质量评估（AQA）任务，解决现有方法缺乏细粒度标注与可解释性问题。构建首个含扣分项细粒度标注的滑雪数据集FineSkiing，并提出JudgeMind模型，通过阶段分割、特征增强与知识引导解码提升评分准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2511.10250v1](https://arxiv.org/pdf/2511.10250v1)**

> **作者:** Yongji Zhang; Siqi Li; Yue Gao; Yu Jiang
>
> **摘要:** Action Quality Assessment (AQA) aims to evaluate and score sports actions, which has attracted widespread interest in recent years. Existing AQA methods primarily predict scores based on features extracted from the entire video, resulting in limited interpretability and reliability. Meanwhile, existing AQA datasets also lack fine-grained annotations for action scores, especially for deduction items and sub-score annotations. In this paper, we construct the first AQA dataset containing fine-grained sub-score and deduction annotations for aerial skiing, which will be released as a new benchmark. For the technical challenges, we propose a novel AQA method, named JudgeMind, which significantly enhances performance and reliability by simulating the judgment and scoring mindset of professional referees. Our method segments the input action video into different stages and scores each stage to enhance accuracy. Then, we propose a stage-aware feature enhancement and fusion module to boost the perception of stage-specific key regions and enhance the robustness to visual changes caused by frequent camera viewpoints switching. In addition, we propose a knowledge-based grade-aware decoder to incorporate possible deduction items as prior knowledge to predict more accurate and reliable scores. Experimental results demonstrate that our method achieves state-of-the-art performance.
>
---
#### [new 035] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging
- **分类: cs.CV; cs.AI**

- **简介: MIRNet面向医学影像诊断，解决标注稀缺、标签不平衡与临床合理性问题，融合自监督预训练、图注意力推理与临床约束优化，并构建TongueAtlas-4K基准，实现舌诊的高精度自动分析。**

- **链接: [https://arxiv.org/pdf/2511.10013v1](https://arxiv.org/pdf/2511.10013v1)**

> **作者:** Shufeng Kong; Zijie Wang; Nuan Cui; Hao Tang; Yihan Meng; Yuanyuan Wei; Feifan Chen; Yingheng Wang; Zhuo Cai; Yaonan Wang; Yulong Zhang; Yuzheng Li; Zibin Zheng; Caihua Liu
>
> **备注:** To appear at AAAI-26
>
> **摘要:** Automated interpretation of medical images demands robust modeling of complex visual-semantic relationships while addressing annotation scarcity, label imbalance, and clinical plausibility constraints. We introduce MIRNet (Medical Image Reasoner Network), a novel framework that integrates self-supervised pre-training with constrained graph-based reasoning. Tongue image diagnosis is a particularly challenging domain that requires fine-grained visual and semantic understanding. Our approach leverages self-supervised masked autoencoder (MAE) to learn transferable visual representations from unlabeled data; employs graph attention networks (GAT) to model label correlations through expert-defined structured graphs; enforces clinical priors via constraint-aware optimization using KL divergence and regularization losses; and mitigates imbalance using asymmetric loss (ASL) and boosting ensembles. To address annotation scarcity, we also introduce TongueAtlas-4K, a comprehensive expert-curated benchmark comprising 4,000 images annotated with 22 diagnostic labels--representing the largest public dataset in tongue analysis. Validation shows our method achieves state-of-the-art performance. While optimized for tongue diagnosis, the framework readily generalizes to broader diagnostic medical imaging tasks.
>
---
#### [new 036] Regional Attention-Enhanced Swin Transformer for Clinically Relevant Medical Image Captioning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出一种增强型Swin-BART模型，用于医学图像自动生成临床描述。通过轻量级区域注意力机制聚焦诊断关键区域，在ROCO数据集上显著提升语义准确性与可解释性，支持人机协同诊断。**

- **链接: [https://arxiv.org/pdf/2511.09893v1](https://arxiv.org/pdf/2511.09893v1)**

> **作者:** Zubia Naz; Farhan Asghar; Muhammad Ishfaq Hussain; Yahya Hadadi; Muhammad Aasim Rafique; Wookjin Choi; Moongu Jeon
>
> **摘要:** Automated medical image captioning translates complex radiological images into diagnostic narratives that can support reporting workflows. We present a Swin-BART encoder-decoder system with a lightweight regional attention module that amplifies diagnostically salient regions before cross-attention. Trained and evaluated on ROCO, our model achieves state-of-the-art semantic fidelity while remaining compact and interpretable. We report results as mean$\pm$std over three seeds and include $95\%$ confidence intervals. Compared with baselines, our approach improves ROUGE (proposed 0.603, ResNet-CNN 0.356, BLIP2-OPT 0.255) and BERTScore (proposed 0.807, BLIP2-OPT 0.645, ResNet-CNN 0.623), with competitive BLEU, CIDEr, and METEOR. We further provide ablations (regional attention on/off and token-count sweep), per-modality analysis (CT/MRI/X-ray), paired significance tests, and qualitative heatmaps that visualize the regions driving each description. Decoding uses beam search (beam size $=4$), length penalty $=1.1$, $no\_repeat\_ngram\_size$ $=3$, and max length $=128$. The proposed design yields accurate, clinically phrased captions and transparent regional attributions, supporting safe research use with a human in the loop.
>
---
#### [new 037] Dynamic Avatar-Scene Rendering from Human-centric Context
- **分类: cs.CV**

- **简介: 该论文提出Separate-then-Map（StM）策略，解决单目视频中人与场景动态重建的时空不一致问题。通过共享高斯属性变换函数，统一建模人与场景，提升边界处渲染质量与准确性。**

- **链接: [https://arxiv.org/pdf/2511.10539v1](https://arxiv.org/pdf/2511.10539v1)**

> **作者:** Wenqing Wang; Haosen Yang; Josef Kittler; Xiatian Zhu
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Reconstructing dynamic humans interacting with real-world environments from monocular videos is an important and challenging task. Despite considerable progress in 4D neural rendering, existing approaches either model dynamic scenes holistically or model scenes and backgrounds separately aim to introduce parametric human priors. However, these approaches either neglect distinct motion characteristics of various components in scene especially human, leading to incomplete reconstructions, or ignore the information exchange between the separately modeled components, resulting in spatial inconsistencies and visual artifacts at human-scene boundaries. To address this, we propose {\bf Separate-then-Map} (StM) strategy that introduces a dedicated information mapping mechanism to bridge separately defined and optimized models. Our method employs a shared transformation function for each Gaussian attribute to unify separately modeled components, enhancing computational efficiency by avoiding exhaustive pairwise interactions while ensuring spatial and visual coherence between humans and their surroundings. Extensive experiments on monocular video datasets demonstrate that StM significantly outperforms existing state-of-the-art methods in both visual quality and rendering accuracy, particularly at challenging human-scene interaction boundaries.
>
---
#### [new 038] LLM-YOLOMS: Large Language Model-based Semantic Interpretation and Fault Diagnosis for Wind Turbine Components
- **分类: cs.CV**

- **简介: 该论文提出LLM-YOLOMS框架，融合YOLOMS视觉检测与领域微调LLM，解决风电部件故障诊断缺乏语义解释性的问题，实现高精度检测与可解释维护建议生成。**

- **链接: [https://arxiv.org/pdf/2511.10394v1](https://arxiv.org/pdf/2511.10394v1)**

> **作者:** Yaru Li; Yanxue Wang; Meng Li; Xinming Li; Jianbo Feng
>
> **备注:** Journal resubmission
>
> **摘要:** The health condition of wind turbine (WT) components is crucial for ensuring stable and reliable operation. However, existing fault detection methods are largely limited to visual recognition, producing structured outputs that lack semantic interpretability and fail to support maintenance decision-making. To address these limitations, this study proposes an integrated framework that combines YOLOMS with a large language model (LLM) for intelligent fault analysis and diagnosis. Specifically, YOLOMS employs multi-scale detection and sliding-window cropping to enhance fault feature extraction, while a lightweight key-value (KV) mapping module bridges the gap between visual outputs and textual inputs. This module converts YOLOMS detection results into structured textual representations enriched with both qualitative and quantitative attributes. A domain-tuned LLM then performs semantic reasoning to generate interpretable fault analyses and maintenance recommendations. Experiments on real-world datasets demonstrate that the proposed framework achieves a fault detection accuracy of 90.6\% and generates maintenance reports with an average accuracy of 89\%, thereby improving the interpretability of diagnostic results and providing practical decision support for the operation and maintenance of wind turbines.
>
---
#### [new 039] Physics informed Transformer-VAE for biophysical parameter estimation: PROSAIL model inversion in Sentinel-2 imagery
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出一种物理信息Transformer-VAE模型，用于在无实测数据条件下，基于模拟数据反演PROSAIL辐射传输模型，从Sentinel-2影像中估算叶面积指数和叶绿素含量，实现无需标定的高精度植被参数反演。**

- **链接: [https://arxiv.org/pdf/2511.10387v1](https://arxiv.org/pdf/2511.10387v1)**

> **作者:** Prince Mensah; Pelumi Victor Aderinto; Ibrahim Salihu Yusuf; Arnu Pretorius
>
> **备注:** 10 pages, 6 figures, uses fancyhdr.sty
>
> **摘要:** Accurate retrieval of vegetation biophysical variables from satellite imagery is crucial for ecosystem monitoring and agricultural management. In this work, we propose a physics-informed Transformer-VAE architecture to invert the PROSAIL radiative transfer model for simultaneous estimation of key canopy parameters from Sentinel-2 data. Unlike previous hybrid approaches that require real satellite images for self-supevised training. Our model is trained exclusively on simulated data, yet achieves performance on par with state-of-the-art methods that utilize real imagery. The Transformer-VAE incorporates the PROSAIL model as a differentiable physical decoder, ensuring that inferred latent variables correspond to physically plausible leaf and canopy properties. We demonstrate retrieval of leaf area index (LAI) and canopy chlorophyll content (CCC) on real-world field datasets (FRM4Veg and BelSAR) with accuracy comparable to models trained with real Sentinel-2 data. Our method requires no in-situ labels or calibration on real images, offering a cost-effective and self-supervised solution for global vegetation monitoring. The proposed approach illustrates how integrating physical models with advanced deep networks can improve the inversion of RTMs, opening new prospects for large-scale, physically-constrained remote sensing of vegetation traits.
>
---
#### [new 040] SAMIRO: Spatial Attention Mutual Information Regularization with a Pre-trained Model as Oracle for Lane Detection
- **分类: cs.CV**

- **简介: 该论文针对车道检测任务，解决复杂环境下数据依赖强、泛化差的问题，提出SAMIRO：一种利用预训练模型作为先验的时空互信息正则化方法，可即插即用提升多种主流模型性能。**

- **链接: [https://arxiv.org/pdf/2511.10385v1](https://arxiv.org/pdf/2511.10385v1)**

> **作者:** Hyunjong Lee; Jangho Lee; Jaekoo Lee
>
> **备注:** 7 pages, 4 figures, paper in press
>
> **摘要:** Lane detection is an important topic in the future mobility solutions. Real-world environmental challenges such as background clutter, varying illumination, and occlusions pose significant obstacles to effective lane detection, particularly when relying on data-driven approaches that require substantial effort and cost for data collection and annotation. To address these issues, lane detection methods must leverage contextual and global information from surrounding lanes and objects. In this paper, we propose a Spatial Attention Mutual Information Regularization with a pre-trained model as an Oracle, called SAMIRO. SAMIRO enhances lane detection performance by transferring knowledge from a pretrained model while preserving domain-agnostic spatial information. Leveraging SAMIRO's plug-and-play characteristic, we integrate it into various state-of-the-art lane detection approaches and conduct extensive experiments on major benchmarks such as CULane, Tusimple, and LLAMAS. The results demonstrate that SAMIRO consistently improves performance across different models and datasets. The code will be made available upon publication.
>
---
#### [new 041] Density Estimation and Crowd Counting
- **分类: cs.CV**

- **简介: 该论文面向视频人群计数任务，解决传统方法时序效率低与密度估计不准问题。提出融合扩散去噪、窄高斯核、回归分支与光流采样的新框架，提升密度图精度并减少冗余帧，实现高效实时人群监控。**

- **链接: [https://arxiv.org/pdf/2511.09723v1](https://arxiv.org/pdf/2511.09723v1)**

> **作者:** Balachandra Devarangadi Sunil; Rakshith Venkatesh; Shantanu Todmal
>
> **摘要:** This study enhances a crowd density estimation algorithm originally designed for image-based analysis by adapting it for video-based scenarios. The proposed method integrates a denoising probabilistic model that utilizes diffusion processes to generate high-quality crowd density maps. To improve accuracy, narrow Gaussian kernels are employed, and multiple density map outputs are generated. A regression branch is incorporated into the model for precise feature extraction, while a consolidation mechanism combines these maps based on similarity scores to produce a robust final result. An event-driven sampling technique, utilizing the Farneback optical flow algorithm, is introduced to selectively capture frames showing significant crowd movements, reducing computational load and storage by focusing on critical crowd dynamics. Through qualitative and quantitative evaluations, including overlay plots and Mean Absolute Error (MAE), the model demonstrates its ability to effectively capture crowd dynamics in both dense and sparse settings. The efficiency of the sampling method is further assessed, showcasing its capability to decrease frame counts while maintaining essential crowd events. By addressing the temporal challenges unique to video analysis, this work offers a scalable and efficient framework for real-time crowd monitoring in applications such as public safety, disaster response, and event management.
>
---
#### [new 042] Simulating Distribution Dynamics: Liquid Temporal Feature Evolution for Single-Domain Generalized Object Detection
- **分类: cs.CV**

- **简介: 该论文针对单域泛化目标检测任务，提出Liquid Temporal Feature Evolution方法，通过时序建模与液态神经网络模拟特征连续演化，缓解源域与未知域间分布差距，提升模型对动态域偏移的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.09909v1](https://arxiv.org/pdf/2511.09909v1)**

> **作者:** Zihao Zhang; Yang Li; Aming Wu; Yahong Han
>
> **摘要:** In this paper, we focus on Single-Domain Generalized Object Detection (Single-DGOD), aiming to transfer a detector trained on one source domain to multiple unknown domains. Existing methods for Single-DGOD typically rely on discrete data augmentation or static perturbation methods to expand data diversity, thereby mitigating the lack of access to target domain data. However, in real-world scenarios such as changes in weather or lighting conditions, domain shifts often occur continuously and gradually. Discrete augmentations and static perturbations fail to effectively capture the dynamic variation of feature distributions, thereby limiting the model's ability to perceive fine-grained cross-domain differences. To this end, we propose a new method, Liquid Temporal Feature Evolution, which simulates the progressive evolution of features from the source domain to simulated latent distributions by incorporating temporal modeling and liquid neural network-driven parameter adjustment. Specifically, we introduce controllable Gaussian noise injection and multi-scale Gaussian blurring to simulate initial feature perturbations, followed by temporal modeling and a liquid parameter adjustment mechanism to generate adaptive modulation parameters, enabling a smooth and continuous adaptation across domains. By capturing progressive cross-domain feature evolution and dynamically regulating adaptation paths, our method bridges the source-unknown domain distribution gap, significantly boosting generalization and robustness to unseen shifts. Significant performance improvements on the Diverse Weather dataset and Real-to-Art benchmark demonstrate the superiority of our method. Our code is available at https://github.com/2490o/LTFE.
>
---
#### [new 043] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition
- **分类: cs.CV**

- **简介: 论文提出SUGAR框架，利用视觉-运动知识引导骨骼表征学习，使预训练大语言模型无需微调即可理解骨骼序列并完成动作识别与描述，解决LLM无法直接解析骨骼数据的问题。**

- **链接: [https://arxiv.org/pdf/2511.10091v1](https://arxiv.org/pdf/2511.10091v1)**

> **作者:** Qilang Ye; Yu Zhou; Lian He; Jie Zhang; Xuanming Guo; Jiayu Zhang; Mingkui Tan; Weicheng Xie; Yue Sun; Tao Tan; Xiaochen Yuan; Ghada Khoriba; Zitong Yu
>
> **备注:** Accepted by AAAI 2026 Main Track
>
> **摘要:** Large Language Models (LLMs) hold rich implicit knowledge and powerful transferability. In this paper, we explore the combination of LLMs with the human skeleton to perform action classification and description. However, when treating LLM as a recognizer, two questions arise: 1) How can LLMs understand skeleton? 2) How can LLMs distinguish among actions? To address these problems, we introduce a novel paradigm named learning Skeleton representation with visUal-motion knowledGe for Action Recognition (SUGAR). In our pipeline, we first utilize off-the-shelf large-scale video models as a knowledge base to generate visual, motion information related to actions. Then, we propose to supervise skeleton learning through this prior knowledge to yield discrete representations. Finally, we use the LLM with untouched pre-training weights to understand these representations and generate the desired action targets and descriptions. Notably, we present a Temporal Query Projection (TQP) module to continuously model the skeleton signals with long sequences. Experiments on several skeleton-based action classification benchmarks demonstrate the efficacy of our SUGAR. Moreover, experiments on zero-shot scenarios show that SUGAR is more versatile than linear-based methods.
>
---
#### [new 044] MTAttack: Multi-Target Backdoor Attacks against Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出MTAttack，首个针对大视觉语言模型的多目标后门攻击方法，解决单触发器攻击的局限性，通过双约束优化实现多个触发器精准映射不同恶意目标，显著提升攻击成功率与泛化性。**

- **链接: [https://arxiv.org/pdf/2511.10098v1](https://arxiv.org/pdf/2511.10098v1)**

> **作者:** Zihan Wang; Guansong Pang; Wenjun Miao; Jin Zheng; Xiao Bai
>
> **备注:** AAAI2026, with supplementary material
>
> **摘要:** Recent advances in Large Visual Language Models (LVLMs) have demonstrated impressive performance across various vision-language tasks by leveraging large-scale image-text pretraining and instruction tuning. However, the security vulnerabilities of LVLMs have become increasingly concerning, particularly their susceptibility to backdoor attacks. Existing backdoor attacks focus on single-target attacks, i.e., targeting a single malicious output associated with a specific trigger. In this work, we uncover multi-target backdoor attacks, where multiple independent triggers corresponding to different attack targets are added in a single pass of training, posing a greater threat to LVLMs in real-world applications. Executing such attacks in LVLMs is challenging since there can be many incorrect trigger-target mappings due to severe feature interference among different triggers. To address this challenge, we propose MTAttack, the first multi-target backdoor attack framework for enforcing accurate multiple trigger-target mappings in LVLMs. The core of MTAttack is a novel optimization method with two constraints, namely Proxy Space Partitioning constraint and Trigger Prototype Anchoring constraint. It jointly optimizes multiple triggers in the latent space, with each trigger independently mapping clean images to a unique proxy class while at the same time guaranteeing their separability. Experiments on popular benchmarks demonstrate a high success rate of MTAttack for multi-target attacks, substantially outperforming existing attack methods. Furthermore, our attack exhibits strong generalizability across datasets and robustness against backdoor defense strategies. These findings highlight the vulnerability of LVLMs to multi-target backdoor attacks and underscore the urgent need for mitigating such threats. Code is available at https://github.com/mala-lab/MTAttack.
>
---
#### [new 045] GEA: Generation-Enhanced Alignment for Text-to-Image Person Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对文本到图像人物检索（TIPR）任务，解决文本描述不完整与模态差距导致的对齐困难问题。提出GEA方法，通过生成图像增强文本语义，并融合生成与原始图像特征，提升跨模态对齐效果。**

- **链接: [https://arxiv.org/pdf/2511.10154v1](https://arxiv.org/pdf/2511.10154v1)**

> **作者:** Hao Zou; Runqing Zhang; Xue Zhou; Jianxiao Zou
>
> **备注:** 8pages,3figures
>
> **摘要:** Text-to-Image Person Retrieval (TIPR) aims to retrieve person images based on natural language descriptions. Although many TIPR methods have achieved promising results, sometimes textual queries cannot accurately and comprehensively reflect the content of the image, leading to poor cross-modal alignment and overfitting to limited datasets. Moreover, the inherent modality gap between text and image further amplifies these issues, making accurate cross-modal retrieval even more challenging. To address these limitations, we propose the Generation-Enhanced Alignment (GEA) from a generative perspective. GEA contains two parallel modules: (1) Text-Guided Token Enhancement (TGTE), which introduces diffusion-generated images as intermediate semantic representations to bridge the gap between text and visual patterns. These generated images enrich the semantic representation of text and facilitate cross-modal alignment. (2) Generative Intermediate Fusion (GIF), which combines cross-attention between generated images, original images, and text features to generate a unified representation optimized by triplet alignment loss. We conduct extensive experiments on three public TIPR datasets, CUHK-PEDES, RSTPReid, and ICFG-PEDES, to evaluate the performance of GEA. The results justify the effectiveness of our method. More implementation details and extended results are available at https://github.com/sugelamyd123/Sup-for-GEA.
>
---
#### [new 046] Scale-Aware Relay and Scale-Adaptive Loss for Tiny Object Detection in Aerial Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对航拍图像中微小目标检测难题，提出尺度感知中继层（SARL）和尺度自适应损失（SAL），增强特征表达并平衡目标回归权重，显著提升YOLOv5/YOLOx等框架的检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.09891v1](https://arxiv.org/pdf/2511.09891v1)**

> **作者:** Jinfu Li; Yuqi Huang; Hong Song; Ting Wang; Jianghan Xia; Yucong Lin; Jingfan Fan; Jian Yang
>
> **摘要:** Recently, despite the remarkable advancements in object detection, modern detectors still struggle to detect tiny objects in aerial images. One key reason is that tiny objects carry limited features that are inevitably degraded or lost during long-distance network propagation. Another is that smaller objects receive disproportionately greater regression penalties than larger ones during training. To tackle these issues, we propose a Scale-Aware Relay Layer (SARL) and a Scale-Adaptive Loss (SAL) for tiny object detection, both of which are seamlessly compatible with the top-performing frameworks. Specifically, SARL employs a cross-scale spatial-channel attention to progressively enrich the meaningful features of each layer and strengthen the cross-layer feature sharing. SAL reshapes the vanilla IoU-based losses so as to dynamically assign lower weights to larger objects. This loss is able to focus training on tiny objects while reducing the influence on large objects. Extensive experiments are conducted on three benchmarks (\textit{i.e.,} AI-TOD, DOTA-v2.0 and VisDrone2019), and the results demonstrate that the proposed method boosts the generalization ability by 5.5\% Average Precision (AP) when embedded in YOLOv5 (anchor-based) and YOLOx (anchor-free) baselines. Moreover, it also promotes the robust performance with 29.0\% AP on the real-world noisy dataset (\textit{i.e.,} AI-TOD-v2.0).
>
---
#### [new 047] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection
- **分类: cs.CV**

- **简介: 论文提出SAM-DAQ，用于RGB-D视频显著目标检测，解决SAM依赖人工提示、内存高和计算重的问题。通过深度引导并行编码器与查询驱动时序记忆模块，实现无提示、高效、多模态融合的视频显著性分割。**

- **链接: [https://arxiv.org/pdf/2511.09870v1](https://arxiv.org/pdf/2511.09870v1)**

> **作者:** Jia Lin; Xiaofei Zhou; Jiyuan Liu; Runmin Cong; Guodao Zhang; Zhi Liu; Jiyong Zhang
>
> **备注:** Accepted to 40th AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Recently segment anything model (SAM) has attracted widespread concerns, and it is often treated as a vision foundation model for universal segmentation. Some researchers have attempted to directly apply the foundation model to the RGB-D video salient object detection (RGB-D VSOD) task, which often encounters three challenges, including the dependence on manual prompts, the high memory consumption of sequential adapters, and the computational burden of memory attention. To address the limitations, we propose a novel method, namely Segment Anything Model with Depth-guided Adaptive Queries (SAM-DAQ), which adapts SAM2 to pop-out salient objects from videos by seamlessly integrating depth and temporal cues within a unified framework. Firstly, we deploy a parallel adapter-based multi-modal image encoder (PAMIE), which incorporates several depth-guided parallel adapters (DPAs) in a skip-connection way. Remarkably, we fine-tune the frozen SAM encoder under prompt-free conditions, where the DPA utilizes depth cues to facilitate the fusion of multi-modal features. Secondly, we deploy a query-driven temporal memory (QTM) module, which unifies the memory bank and prompt embeddings into a learnable pipeline. Concretely, by leveraging both frame-level queries and video-level queries simultaneously, the QTM module can not only selectively extract temporal consistency features but also iteratively update the temporal representations of the queries. Extensive experiments are conducted on three RGB-D VSOD datasets, and the results show that the proposed SAM-DAQ consistently outperforms state-of-the-art methods in terms of all evaluation metrics.
>
---
#### [new 048] DGFusion: Dual-guided Fusion for Robust Multi-Modal 3D Object Detection
- **分类: cs.CV**

- **简介: 论文针对自动驾驶中多模态3D目标检测的硬实例（远、小、遮挡）检测难题，提出DGFusion，通过双引导融合机制与困难感知匹配器，协同利用点云与图像模态优势，显著提升检测性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.10035v1](https://arxiv.org/pdf/2511.10035v1)**

> **作者:** Feiyang Jia; Caiyan Jia; Ailin Liu; Shaoqing Xu; Qiming Xia; Lin Liu; Lei Yang; Yan Gong; Ziying Song
>
> **摘要:** As a critical task in autonomous driving perception systems, 3D object detection is used to identify and track key objects, such as vehicles and pedestrians. However, detecting distant, small, or occluded objects (hard instances) remains a challenge, which directly compromises the safety of autonomous driving systems. We observe that existing multi-modal 3D object detection methods often follow a single-guided paradigm, failing to account for the differences in information density of hard instances between modalities. In this work, we propose DGFusion, based on the Dual-guided paradigm, which fully inherits the advantages of the Point-guide-Image paradigm and integrates the Image-guide-Point paradigm to address the limitations of the single paradigms. The core of DGFusion, the Difficulty-aware Instance Pair Matcher (DIPM), performs instance-level feature matching based on difficulty to generate easy and hard instance pairs, while the Dual-guided Modules exploit the advantages of both pair types to enable effective multi-modal feature fusion. Experimental results demonstrate that our DGFusion outperforms the baseline methods, with respective improvements of +1.0\% mAP, +0.8\% NDS, and +1.3\% average recall on nuScenes. Extensive experiments demonstrate consistent robustness gains for hard instance detection across ego-distance, size, visibility, and small-scale training scenarios.
>
---
#### [new 049] Rethinking Visual Information Processing in Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出LLaViT，将大语言模型作为视觉编码器，通过三重改进解决多模态LLM中视觉与文本特征融合不佳的问题，显著超越LLaVA等基线方法，在视觉-语言任务上表现更优。**

- **链接: [https://arxiv.org/pdf/2511.10301v1](https://arxiv.org/pdf/2511.10301v1)**

> **作者:** Dongwan Kim; Viresh Ranjan; Takashi Nagata; Arnab Dhua; Amit Kumar K C
>
> **摘要:** Despite the remarkable success of the LLaVA architecture for vision-language tasks, its design inherently struggles to effectively integrate visual features due to the inherent mismatch between text and vision modalities. We tackle this issue from a novel perspective in which the LLM not only serves as a language model but also a powerful vision encoder. To this end, we present LLaViT - Large Language Models as extended Vision Transformers - which enables the LLM to simultaneously function as a vision encoder through three key modifications: (1) learning separate QKV projections for vision modality, (2) enabling bidirectional attention on visual tokens, and (3) incorporating both global and local visual representations. Through extensive controlled experiments on a wide range of LLMs, we demonstrate that LLaViT significantly outperforms the baseline LLaVA method on a multitude of benchmarks, even surpassing models with double its parameter count, establishing a more effective approach to vision-language modeling.
>
---
#### [new 050] MosaicDoc: A Large-Scale Bilingual Benchmark for Visually Rich Document Understanding
- **分类: cs.CV**

- **简介: 该论文提出MosaicDoc，一个中英双语、大规模视觉丰富文档理解基准，解决现有数据集布局简单、语言单一问题。通过多智能体生成72K图文样本与60万QA对，支持OCR、VQA等多任务，推动真实文档理解研究。**

- **链接: [https://arxiv.org/pdf/2511.09919v1](https://arxiv.org/pdf/2511.09919v1)**

> **作者:** Ketong Chen; Yuhao Chen; Yang Xue
>
> **摘要:** Despite the rapid progress of Vision-Language Models (VLMs), their capabilities are inadequately assessed by existing benchmarks, which are predominantly English-centric, feature simplistic layouts, and support limited tasks. Consequently, they fail to evaluate model performance for Visually Rich Document Understanding (VRDU), a critical challenge involving complex layouts and dense text. To address this, we introduce DocWeaver, a novel multi-agent pipeline that leverages Large Language Models to automatically generate a new benchmark. The result is MosaicDoc, a large-scale, bilingual (Chinese and English) resource designed to push the boundaries of VRDU. Sourced from newspapers and magazines, MosaicDoc features diverse and complex layouts (including multi-column and non-Manhattan), rich stylistic variety from 196 publishers, and comprehensive multi-task annotations (OCR, VQA, reading order, and localization). With 72K images and over 600K QA pairs, MosaicDoc serves as a definitive benchmark for the field. Our extensive evaluation of state-of-the-art models on this benchmark reveals their current limitations in handling real-world document complexity and charts a clear path for future research.
>
---
#### [new 051] Histology-informed tiling of whole tissue sections improves the interpretability and predictability of cancer relapse and genetic alterations
- **分类: cs.CV; q-bio.QM; q-bio.TO**

- **简介: 该论文提出组织学引导的切片方法（HIT），利用语义分割提取前列腺癌组织中的腺体结构，替代传统网格切片，提升多实例学习模型对癌症复发和基因变异的预测准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.10432v1](https://arxiv.org/pdf/2511.10432v1)**

> **作者:** Willem Bonnaffé; Yang Hu; Andrea Chatrian; Mengran Fan; Stefano Malacrino; Sandy Figiel; CRUK ICGC Prostate Group; Srinivasa R. Rao; Richard Colling; Richard J. Bryant; Freddie C. Hamdy; Dan J. Woodcock; Ian G. Mills; Clare Verrill; Jens Rittscher
>
> **备注:** 26 pages, 6 figures
>
> **摘要:** Histopathologists establish cancer grade by assessing histological structures, such as glands in prostate cancer. Yet, digital pathology pipelines often rely on grid-based tiling that ignores tissue architecture. This introduces irrelevant information and limits interpretability. We introduce histology-informed tiling (HIT), which uses semantic segmentation to extract glands from whole slide images (WSIs) as biologically meaningful input patches for multiple-instance learning (MIL) and phenotyping. Trained on 137 samples from the ProMPT cohort, HIT achieved a gland-level Dice score of 0.83 +/- 0.17. By extracting 380,000 glands from 760 WSIs across ICGC-C and TCGA-PRAD cohorts, HIT improved MIL models AUCs by 10% for detecting copy number variation (CNVs) in genes related to epithelial-mesenchymal transitions (EMT) and MYC, and revealed 15 gland clusters, several of which were associated with cancer relapse, oncogenic mutations, and high Gleason. Therefore, HIT improved the accuracy and interpretability of MIL predictions, while streamlining computations by focussing on biologically meaningful structures during feature extraction.
>
---
#### [new 052] CephRes-MHNet: A Multi-Head Residual Network for Accurate and Robust Cephalometric Landmark Detection
- **分类: cs.CV**

- **简介: 该论文提出CephRes-MHNet，用于2D头颅侧位X光片的自动地标检测，解决低对比度与解剖复杂性导致的定位不准问题。通过多头残差网络与双注意力机制，实现更高精度与参数效率，超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10173v1](https://arxiv.org/pdf/2511.10173v1)**

> **作者:** Ahmed Jaheen; Islam Hassan; Mohanad Abouserie; Abdelaty Rehab; Adham Elasfar; Knzy Elmasry; Mostafa El-Dawlatly; Seif Eldawlatly
>
> **备注:** 5 Pages, Under Review at The IEEE International Symposium on Biomedical Imaging (ISBI 2026)
>
> **摘要:** Accurate localization of cephalometric landmarks from 2D lateral skull X-rays is vital for orthodontic diagnosis and treatment. Manual annotation is time-consuming and error-prone, whereas automated approaches often struggle with low contrast and anatomical complexity. This paper introduces CephRes-MHNet, a multi-head residual convolutional network for robust and efficient cephalometric landmark detection. The architecture integrates residual encoding, dual-attention mechanisms, and multi-head decoders to enhance contextual reasoning and anatomical precision. Trained on the Aariz Cephalometric dataset of 1,000 radiographs, CephRes-MHNet achieved a mean radial error (MRE) of 1.23 mm and a success detection rate (SDR) @ 2.0 mm of 85.5%, outperforming all evaluated models. In particular, it exceeded the strongest baseline, the attention-driven AFPF-Net (MRE = 1.25 mm, SDR @ 2.0 mm = 84.1%), while using less than 25% of its parameters. These results demonstrate that CephRes-MHNet attains state-of-the-art accuracy through architectural efficiency, providing a practical solution for real-world orthodontic analysis.
>
---
#### [new 053] Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification
- **分类: cs.CV**

- **简介: 该论文针对高光谱图像分类中置信度误导问题，提出CABIN框架，通过感知不确定性、引导采样与细粒度伪标签校正，实现闭环学习，提升稀疏标注下的泛化能力与标注效率。**

- **链接: [https://arxiv.org/pdf/2511.10068v1](https://arxiv.org/pdf/2511.10068v1)**

> **作者:** Muzhou Yang; Wuzhou Quan; Mingqiang Wei
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Confidence alone is often misleading in hyperspectral image classification, as models tend to mistake high predictive scores for correctness while lacking awareness of uncertainty. This leads to confirmation bias, especially under sparse annotations or class imbalance, where models overfit confident errors and fail to generalize. We propose CABIN (Cognitive-Aware Behavior-Informed learNing), a semi-supervised framework that addresses this limitation through a closed-loop learning process of perception, action, and correction. CABIN first develops perceptual awareness by estimating epistemic uncertainty, identifying ambiguous regions where errors are likely to occur. It then acts by adopting an Uncertainty-Guided Dual Sampling Strategy, selecting uncertain samples for exploration while anchoring confident ones as stable pseudo-labels to reduce bias. To correct noisy supervision, CABIN introduces a Fine-Grained Dynamic Assignment Strategy that categorizes pseudo-labeled data into reliable, ambiguous, and noisy subsets, applying tailored losses to enhance generalization. Experimental results show that a wide range of state-of-the-art methods benefit from the integration of CABIN, with improved labeling efficiency and performance.
>
---
#### [new 054] PriVi: Towards A General-Purpose Video Model For Primate Behavior In The Wild
- **分类: cs.CV; cs.LG**

- **简介: 论文提出PriVi，首个灵长类行为专用视频预训练数据集，通过数据-centric方法预训练V-JEPA模型，提升在多野外灵长类行为数据集上的泛化与数据效率，显著超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.09675v1](https://arxiv.org/pdf/2511.09675v1)**

> **作者:** Felix B. Mueller; Jan F. Meier; Timo Lueddecke; Richard Vogg; Roger L. Freixanet; Valentin Hassler; Tiffany Bosshard; Elif Karakoc; William J. O'Hearn; Sofia M. Pereira; Sandro Sehner; Kaja Wierucka; Judith Burkart; Claudia Fichtel; Julia Fischer; Alexander Gail; Catherine Hobaiter; Julia Ostner; Liran Samuni; Oliver Schülke; Neda Shahidi; Erin G. Wessling; Alexander S. Ecker
>
> **摘要:** Non-human primates are our closest living relatives, and analyzing their behavior is central to research in cognition, evolution, and conservation. Computer vision could greatly aid this research, but existing methods often rely on human-centric pretrained models and focus on single datasets, which limits generalization. We address this limitation by shifting from a model-centric to a data-centric approach and introduce PriVi, a large-scale primate-centric video pretraining dataset. PriVi contains 424 hours of curated video, combining 174 hours from behavioral research across 11 settings with 250 hours of diverse web-sourced footage, assembled through a scalable data curation pipeline. We pretrain V-JEPA on PriVi to learn primate-specific representations and evaluate it using a lightweight frozen classifier. Across four benchmark datasets, ChimpACT, BaboonLand, PanAf500, and ChimpBehave, our approach consistently outperforms prior work, including fully finetuned baselines, and scales favorably with fewer labels. These results demonstrate that primate-centric pretraining substantially improves data efficiency and generalization, making it a promising approach for low-label applications. Code, models, and the majority of the dataset will be made available.
>
---
#### [new 055] RobIA: Robust Instance-aware Continual Test-time Adaptation for Deep Stereo
- **分类: cs.CV**

- **简介: 论文提出RobIA，用于深度立体视觉的持续在线自适应任务，解决动态域偏移下标注稀疏与适应僵化问题。通过实例感知的AttEx-MoE与Robust AdaptBN教师模型，实现高效、鲁棒的持续自适应。**

- **链接: [https://arxiv.org/pdf/2511.10107v1](https://arxiv.org/pdf/2511.10107v1)**

> **作者:** Jueun Ko; Hyewon Park; Hyesong Choi; Dongbo Min
>
> **备注:** Accepted by Neural Information Processing Systems (NeurIPS) 2025
>
> **摘要:** Stereo Depth Estimation in real-world environments poses significant challenges due to dynamic domain shifts, sparse or unreliable supervision, and the high cost of acquiring dense ground-truth labels. While recent Test-Time Adaptation (TTA) methods offer promising solutions, most rely on static target domain assumptions and input-invariant adaptation strategies, limiting their effectiveness under continual shifts. In this paper, we propose RobIA, a novel Robust, Instance-Aware framework for Continual Test-Time Adaptation (CTTA) in stereo depth estimation. RobIA integrates two key components: (1) Attend-and-Excite Mixture-of-Experts (AttEx-MoE), a parameter-efficient module that dynamically routes input to frozen experts via lightweight self-attention mechanism tailored to epipolar geometry, and (2) Robust AdaptBN Teacher, a PEFT-based teacher model that provides dense pseudo-supervision by complementing sparse handcrafted labels. This strategy enables input-specific flexibility, broad supervision coverage, improving generalization under domain shift. Extensive experiments demonstrate that RobIA achieves superior adaptation performance across dynamic target domains while maintaining computational efficiency.
>
---
#### [new 056] SPOT: Sparsification with Attention Dynamics via Token Relevance in Vision Transformers
- **分类: cs.CV; eess.IV**

- **简介: SPOT提出一种轻量级框架，通过动态分析Vision Transformer中token的相关性，在早期阶段稀疏化冗余token，显著降低计算开销（最高40%），同时保持或提升模型精度。**

- **链接: [https://arxiv.org/pdf/2511.10488v1](https://arxiv.org/pdf/2511.10488v1)**

> **作者:** Oded Schlesinger; Amirhossein Farzam; J. Matias Di Martino; Guillermo Sapiro
>
> **备注:** Project repository: https://github.com/odedsc/SPOT
>
> **摘要:** While Vision Transformers (ViT) have demonstrated remarkable performance across diverse tasks, their computational demands are substantial, scaling quadratically with the number of processed tokens. Compact attention representations, reflecting token interaction distributions, can guide early detection and reduction of less salient tokens prior to attention computation. Motivated by this, we present SParsification with attentiOn dynamics via Token relevance (SPOT), a framework for early detection of redundant tokens within ViTs that leverages token embeddings, interactions, and attention dynamics across layers to infer token importance, resulting in a more context-aware and interpretable relevance detection process. SPOT informs token sparsification and facilitates the elimination of such tokens, improving computational efficiency without sacrificing performance. SPOT employs computationally lightweight predictors that can be plugged into various ViT architectures and learn to derive effective input-specific token prioritization across layers. Its versatile design supports a range of performance levels adaptable to varying resource constraints. Empirical evaluations demonstrate significant efficiency gains of up to 40% compared to standard ViTs, while maintaining or even improving accuracy. Code and models are available at https://github.com/odedsc/SPOT .
>
---
#### [new 057] VLF-MSC: Vision-Language Feature-Based Multimodal Semantic Communication System
- **分类: cs.CV; eess.SY**

- **简介: 论文提出VLF-MSC，一种基于视觉-语言联合表征的多模态语义通信系统，解决传统方法多模态独立传输效率低的问题，通过预训练VLM编码为统一语义特征，实现单流传输下图像与文本的协同重建，提升低信噪比下的频谱效率与语义保真度。**

- **链接: [https://arxiv.org/pdf/2511.10074v1](https://arxiv.org/pdf/2511.10074v1)**

> **作者:** Gwangyeon Ahn; Jiwan Seo; Joonhyuk Kang
>
> **备注:** To appear in the AI4NextG Workshop at NeurIPS 2025
>
> **摘要:** We propose Vision-Language Feature-based Multimodal Semantic Communication (VLF-MSC), a unified system that transmits a single compact vision-language representation to support both image and text generation at the receiver. Unlike existing semantic communication techniques that process each modality separately, VLF-MSC employs a pre-trained vision-language model (VLM) to encode the source image into a vision-language semantic feature (VLF), which is transmitted over the wireless channel. At the receiver, a decoder-based language model and a diffusion-based image generator are both conditioned on the VLF to produce a descriptive text and a semantically aligned image. This unified representation eliminates the need for modality-specific streams or retransmissions, improving spectral efficiency and adaptability. By leveraging foundation models, the system achieves robustness to channel noise while preserving semantic fidelity. Experiments demonstrate that VLF-MSC outperforms text-only and image-only baselines, achieving higher semantic accuracy for both modalities under low SNR with significantly reduced bandwidth.
>
---
#### [new 058] AHA! Animating Human Avatars in Diverse Scenes with Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出基于高斯溅射（3DGS）的新人物动画框架，解决传统方法依赖网格/点云、难以实现自由视角交互的问题。通过解耦运动合成与渲染，利用高斯结构引导姿态对齐与场景交互，实现无配对数据的几何一致人体动画。**

- **链接: [https://arxiv.org/pdf/2511.09827v1](https://arxiv.org/pdf/2511.09827v1)**

> **作者:** Aymen Mir; Jian Wang; Riza Alp Guler; Chuan Guo; Gerard Pons-Moll; Bing Zhou
>
> **摘要:** We present a novel framework for animating humans in 3D scenes using 3D Gaussian Splatting (3DGS), a neural scene representation that has recently achieved state-of-the-art photorealistic results for novel-view synthesis but remains under-explored for human-scene animation and interaction. Unlike existing animation pipelines that use meshes or point clouds as the underlying 3D representation, our approach introduces the use of 3DGS as the 3D representation to the problem of animating humans in scenes. By representing humans and scenes as Gaussians, our approach allows for geometry-consistent free-viewpoint rendering of humans interacting with 3D scenes. Our key insight is that the rendering can be decoupled from the motion synthesis and each sub-problem can be addressed independently, without the need for paired human-scene data. Central to our method is a Gaussian-aligned motion module that synthesizes motion without explicit scene geometry, using opacity-based cues and projected Gaussian structures to guide human placement and pose alignment. To ensure natural interactions, we further propose a human-scene Gaussian refinement optimization that enforces realistic contact and navigation. We evaluate our approach on scenes from Scannet++ and the SuperSplat library, and on avatars reconstructed from sparse and dense multi-view human capture. Finally, we demonstrate that our framework allows for novel applications such as geometry-consistent free-viewpoint rendering of edited monocular RGB videos with new animated humans, showcasing the unique advantage of 3DGS for monocular video-based human animation.
>
---
#### [new 059] From 2D to 3D Without Extra Baggage: Data-Efficient Cancer Detection in Digital Breast Tomosynthesis
- **分类: cs.CV**

- **简介: 该论文针对乳腺断层合成（DBT）数据稀缺问题，提出M&M-3D架构，在不增加参数的前提下实现3D特征学习，通过复用2D模型权重，提升癌症检测的定位与分类性能，显著优于传统2D与复杂3D方法。**

- **链接: [https://arxiv.org/pdf/2511.10597v1](https://arxiv.org/pdf/2511.10597v1)**

> **作者:** Yen Nhi Truong Vu; Dan Guo; Sripad Joshi; Harshit Kumar; Jason Su; Thomas Paul Matthews
>
> **摘要:** Digital Breast Tomosynthesis (DBT) enhances finding visibility for breast cancer detection by providing volumetric information that reduces the impact of overlapping tissues; however, limited annotated data has constrained the development of deep learning models for DBT. To address data scarcity, existing methods attempt to reuse 2D full-field digital mammography (FFDM) models by either flattening DBT volumes or processing slices individually, thus discarding volumetric information. Alternatively, 3D reasoning approaches introduce complex architectures that require more DBT training data. Tackling these drawbacks, we propose M&M-3D, an architecture that enables learnable 3D reasoning while remaining parameter-free relative to its FFDM counterpart, M&M. M&M-3D constructs malignancy-guided 3D features, and 3D reasoning is learned through repeatedly mixing these 3D features with slice-level information. This is achieved by modifying operations in M&M without adding parameters, thus enabling direct weight transfer from FFDM. Extensive experiments show that M&M-3D surpasses 2D projection and 3D slice-based methods by 11-54% for localization and 3-10% for classification. Additionally, M&M-3D outperforms complex 3D reasoning variants by 20-47% for localization and 2-10% for classification in the low-data regime, while matching their performance in high-data regime. On the popular BCS-DBT benchmark, M&M-3D outperforms previous top baseline by 4% for classification and 10% for localization.
>
---
#### [new 060] FedeCouple: Fine-Grained Balancing of Global-Generalization and Local-Adaptability in Federated Learning
- **分类: cs.CV**

- **简介: FedeCouple面向联邦学习中的异构数据场景，解决全局泛化与本地适应性失衡问题，通过细粒度协同学习特征表示、动态知识蒸馏和隐私保护锚点，提升模型性能与安全性。**

- **链接: [https://arxiv.org/pdf/2511.09599v1](https://arxiv.org/pdf/2511.09599v1)**

> **作者:** Ming Yang; Dongrun Li; Xin Wang; Feng Li; Lisheng Fan; Chunxiao Wang; Xiaoming Wu; Peng Cheng
>
> **摘要:** In privacy-preserving mobile network transmission scenarios with heterogeneous client data, personalized federated learning methods that decouple feature extractors and classifiers have demonstrated notable advantages in enhancing learning capability. However, many existing approaches primarily focus on feature space consistency and classification personalization during local training, often neglecting the local adaptability of the extractor and the global generalization of the classifier. This oversight results in insufficient coordination and weak coupling between the components, ultimately degrading the overall model performance. To address this challenge, we propose FedeCouple, a federated learning method that balances global generalization and local adaptability at a fine-grained level. Our approach jointly learns global and local feature representations while employing dynamic knowledge distillation to enhance the generalization of personalized classifiers. We further introduce anchors to refine the feature space; their strict locality and non-transmission inherently preserve privacy and reduce communication overhead. Furthermore, we provide a theoretical analysis proving that FedeCouple converges for nonconvex objectives, with iterates approaching a stationary point as the number of communication rounds increases. Extensive experiments conducted on five image-classification datasets demonstrate that FedeCouple consistently outperforms nine baseline methods in effectiveness, stability, scalability, and security. Notably, in experiments evaluating effectiveness, FedeCouple surpasses the best baseline by a significant margin of 4.3%.
>
---
#### [new 061] MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
- **分类: cs.CV; cs.RO**

- **简介: 论文提出MSGNav，面向零样本具身导航任务，解决传统3D场景图丢失视觉信息与词汇受限问题，引入多模态3D场景图（M3DSG）并设计四模块系统，实现开放词汇、高效推理与精准终点选择，性能达SOTA。**

- **链接: [https://arxiv.org/pdf/2511.10376v1](https://arxiv.org/pdf/2511.10376v1)**

> **作者:** Xun Huang; Shijia Zhao; Yunxiang Wang; Xin Lu; Wanfa Zhang; Rongsheng Qu; Weixin Li; Yunhong Wang; Chenglu Wen
>
> **备注:** 10 pages
>
> **摘要:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relational edges with dynamically assigned images. Built on M3DSG, we propose MSGNav, a zero-shot navigation system that includes a Key Subgraph Selection module for efficient reasoning, an Adaptive Vocabulary Update module for open vocabulary support, and a Closed-Loop Reasoning module for accurate exploration reasoning. Additionally, we further identify the last-mile problem in zero-shot navigation - determining the feasible target location with a suitable final viewpoint, and propose a Visibility-based Viewpoint Decision module to explicitly resolve it. Comprehensive experimental results demonstrate that MSGNav achieves state-of-the-art performance on GOAT-Bench and HM3D-OVON datasets. The open-source code will be publicly available.
>
---
#### [new 062] One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出Latent Upscaler Adapter（LUA），在扩散模型的潜在空间中直接实现超分辨率，解决高分辨率生成慢、后处理失真问题，无需修改原模型，仅单次前向传播即可高效提升分辨率，显著加速且保持画质。**

- **链接: [https://arxiv.org/pdf/2511.10629v1](https://arxiv.org/pdf/2511.10629v1)**

> **作者:** Aleksandr Razin; Danil Kazantsev; Ilya Makarov
>
> **摘要:** Diffusion models struggle to scale beyond their training resolutions, as direct high-resolution sampling is slow and costly, while post-hoc image super-resolution (ISR) introduces artifacts and additional latency by operating after decoding. We present the Latent Upscaler Adapter (LUA), a lightweight module that performs super-resolution directly on the generator's latent code before the final VAE decoding step. LUA integrates as a drop-in component, requiring no modifications to the base model or additional diffusion stages, and enables high-resolution synthesis through a single feed-forward pass in latent space. A shared Swin-style backbone with scale-specific pixel-shuffle heads supports 2x and 4x factors and remains compatible with image-space SR baselines, achieving comparable perceptual quality with nearly 3x lower decoding and upscaling time (adding only +0.42 s for 1024 px generation from 512 px, compared to 1.87 s for pixel-space SR using the same SwinIR architecture). Furthermore, LUA shows strong generalization across the latent spaces of different VAEs, making it easy to deploy without retraining from scratch for each new decoder. Extensive experiments demonstrate that LUA closely matches the fidelity of native high-resolution generation while offering a practical and efficient path to scalable, high-fidelity image synthesis in modern diffusion pipelines.
>
---
#### [new 063] Beyond Cosine Similarity Magnitude-Aware CLIP for No-Reference Image Quality Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向无参考图像质量评估（NR-IQA），发现CLIP图像特征的幅值与感知质量强相关，提出一种无需训练的幅值感知融合框架，结合归一化幅值与余弦相似度，显著提升评估性能。**

- **链接: [https://arxiv.org/pdf/2511.09948v1](https://arxiv.org/pdf/2511.09948v1)**

> **作者:** Zhicheng Liao; Dongxu Wu; Zhenshan Shi; Sijie Mai; Hanwei Zhu; Lingyu Zhu; Yuncheng Jiang; Baoliang Chen
>
> **摘要:** Recent efforts have repurposed the Contrastive Language-Image Pre-training (CLIP) model for No-Reference Image Quality Assessment (NR-IQA) by measuring the cosine similarity between the image embedding and textual prompts such as "a good photo" or "a bad photo." However, this semantic similarity overlooks a critical yet underexplored cue: the magnitude of the CLIP image features, which we empirically find to exhibit a strong correlation with perceptual quality. In this work, we introduce a novel adaptive fusion framework that complements cosine similarity with a magnitude-aware quality cue. Specifically, we first extract the absolute CLIP image features and apply a Box-Cox transformation to statistically normalize the feature distribution and mitigate semantic sensitivity. The resulting scalar summary serves as a semantically-normalized auxiliary cue that complements cosine-based prompt matching. To integrate both cues effectively, we further design a confidence-guided fusion scheme that adaptively weighs each term according to its relative strength. Extensive experiments on multiple benchmark IQA datasets demonstrate that our method consistently outperforms standard CLIP-based IQA and state-of-the-art baselines, without any task-specific training.
>
---
#### [new 064] Classifying Phonotrauma Severity from Vocal Fold Images with Soft Ordinal Regression
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出一种软序回归方法，自动从声带图像分类音创伤严重程度，解决临床评估成本高、主观性强的问题，通过处理标注分布提升预测精度与不确定性校准，逼近专家水平。**

- **链接: [https://arxiv.org/pdf/2511.09702v1](https://arxiv.org/pdf/2511.09702v1)**

> **作者:** Katie Matton; Purvaja Balaji; Hamzeh Ghasemzadeh; Jameson C. Cooper; Daryush D. Mehta; Jarrad H. Van Stan; Robert E. Hillman; Rosalind Picard; John Guttag; S. Mazdak Abulnaga
>
> **备注:** 16 pages, 9 figures, 5 tables; ML4H 2025; Proceedings of Machine Learning Research 297, 2025
>
> **摘要:** Phonotrauma refers to vocal fold tissue damage resulting from exposure to forces during voicing. It occurs on a continuum from mild to severe, and treatment options can vary based on severity. Assessment of severity involves a clinician's expert judgment, which is costly and can vary widely in reliability. In this work, we present the first method for automatically classifying phonotrauma severity from vocal fold images. To account for the ordinal nature of the labels, we adopt a widely used ordinal regression framework. To account for label uncertainty, we propose a novel modification to ordinal regression loss functions that enables them to operate on soft labels reflecting annotator rating distributions. Our proposed soft ordinal regression method achieves predictive performance approaching that of clinical experts, while producing well-calibrated uncertainty estimates. By providing an automated tool for phonotrauma severity assessment, our work can enable large-scale studies of phonotrauma, ultimately leading to improved clinical understanding and patient care.
>
---
#### [new 065] A Style is Worth One Code: Unlocking Code-to-Style Image Generation with Discrete Style Space
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出“代码到风格图像生成”新任务，解决现有方法风格不一致、创意受限问题。首创开源方法CoTyle，通过离散风格码本与自回归生成器，仅用数值编码即可可控生成多样风格图像。**

- **链接: [https://arxiv.org/pdf/2511.10555v1](https://arxiv.org/pdf/2511.10555v1)**

> **作者:** Huijie Liu; Shuhao Cui; Haoxiang Cao; Shuai Ma; Kai Wu; Guoliang Kang
>
> **备注:** 16 pages, 13 figures, 5 tables
>
> **摘要:** Innovative visual stylization is a cornerstone of artistic creation, yet generating novel and consistent visual styles remains a significant challenge. Existing generative approaches typically rely on lengthy textual prompts, reference images, or parameter-efficient fine-tuning to guide style-aware image generation, but often struggle with style consistency, limited creativity, and complex style representations. In this paper, we affirm that a style is worth one numerical code by introducing the novel task, code-to-style image generation, which produces images with novel, consistent visual styles conditioned solely on a numerical style code. To date, this field has only been primarily explored by the industry (e.g., Midjourney), with no open-source research from the academic community. To fill this gap, we propose CoTyle, the first open-source method for this task. Specifically, we first train a discrete style codebook from a collection of images to extract style embeddings. These embeddings serve as conditions for a text-to-image diffusion model (T2I-DM) to generate stylistic images. Subsequently, we train an autoregressive style generator on the discrete style embeddings to model their distribution, allowing the synthesis of novel style embeddings. During inference, a numerical style code is mapped to a unique style embedding by the style generator, and this embedding guides the T2I-DM to generate images in the corresponding style. Unlike existing methods, our method offers unparalleled simplicity and diversity, unlocking a vast space of reproducible styles from minimal input. Extensive experiments validate that CoTyle effectively turns a numerical code into a style controller, demonstrating a style is worth one code.
>
---
#### [new 066] GridPrune: From "Where to Look" to "What to Select" in Visual Token Pruning for MLLMs
- **分类: cs.CV**

- **简介: GridPrune针对MLLMs视觉标记剪枝中的效率问题，提出“先定区域、再选标记”的两阶段策略，替代传统全局Top-K方法，通过文本引导动态分配各区域预算，显著提升剪枝效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.10081v1](https://arxiv.org/pdf/2511.10081v1)**

> **作者:** Yuxiang Duan; Ao Li; Yingqin Li; Luyu Li; Pengwei Wang
>
> **摘要:** Multimodal large language models (MLLMs) have shown remarkable capabilities in a wide range of vision-language tasks. However, the large number of visual tokens introduces significant computational overhead. To address this issue, visual token pruning has emerged as a key technique for enhancing the efficiency of MLLMs. In cognitive science, humans tend to first determine which regions of a scene to attend to ("where to look") before deciding which specific elements within those regions to process in detail ("what to select"). This two-stage strategy enables the visual system to efficiently allocate attention at a coarse spatial level before performing fine-grained selection. However, existing pruning methods primarily focus on directly optimizing "what to select", typically using attention scores or similarity metrics. They rarely consider "where to look", which has been shown to lead to inefficient spatial allocation, positional bias, and the retention of irrelevant or redundant tokens. In this paper, we propose GridPrune, a method that replaces the global Top-K mechanism with a "guide-globally, select-locally" zonal selection system. GridPrune splits the pruning process into two steps: first, it uses text-conditional guidance to dynamically allocate a token budget across spatial zones; and then, it performs local selection within each budgeted zone. Experimental results demonstrate that GridPrune achieves superior performance across various MLLM architectures. On LLaVA-NeXT-7B, GridPrune retains 96.98% of the full performance while using 11.1% of the tokens, outperforming the best-performing baseline by 2.34% at the same pruning rate.
>
---
#### [new 067] HeatV2X: Scalable Heterogeneous Collaborative Perception via Efficient Alignment and Interaction
- **分类: cs.CV**

- **简介: 论文提出HeatV2X，面向车路协同感知任务，解决多模态异构代理间特征对齐难与扩展性差的问题，通过局部与全局自适应适配器实现高效协同，显著提升性能并降低训练开销。**

- **链接: [https://arxiv.org/pdf/2511.10211v1](https://arxiv.org/pdf/2511.10211v1)**

> **作者:** Yueran Zhao; Zhang Zhang; Chao Sun; Tianze Wang; Chao Yue; Nuoran Li
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Vehicle-to-Everything (V2X) collaborative perception extends sensing beyond single vehicle limits through transmission. However, as more agents participate, existing frameworks face two key challenges: (1) the participating agents are inherently multi-modal and heterogeneous, and (2) the collaborative framework must be scalable to accommodate new agents. The former requires effective cross-agent feature alignment to mitigate heterogeneity loss, while the latter renders full-parameter training impractical, highlighting the importance of scalable adaptation. To address these issues, we propose Heterogeneous Adaptation (HeatV2X), a scalable collaborative framework. We first train a high-performance agent based on heterogeneous graph attention as the foundation for collaborative learning. Then, we design Local Heterogeneous Fine-Tuning and Global Collaborative Fine-Tuning to achieve effective alignment and interaction among heterogeneous agents. The former efficiently extracts modality-specific differences using Hetero-Aware Adapters, while the latter employs the Multi-Cognitive Adapter to enhance cross-agent collaboration and fully exploit the fusion potential. These designs enable substantial performance improvement of the collaborative framework with minimal training cost. We evaluate our approach on the OPV2V-H and DAIR-V2X datasets. Experimental results demonstrate that our method achieves superior perception performance with significantly reduced training overhead, outperforming existing state-of-the-art approaches. Our implementation will be released soon.
>
---
#### [new 068] Social LSTM with Dynamic Occupancy Modeling for Realistic Pedestrian Trajectory Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对行人轨迹预测任务，提出一种改进的Social LSTM模型，引入动态占用空间损失函数，兼顾碰撞规避与位置精度，有效降低拥挤场景下的碰撞率并提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2511.09735v1](https://arxiv.org/pdf/2511.09735v1)**

> **作者:** Ahmed Alia; Mohcine Chraibi; Armin Seyfried
>
> **备注:** 19 pages, 9 figures, 4 tables
>
> **摘要:** In dynamic and crowded environments, realistic pedestrian trajectory prediction remains a challenging task due to the complex nature of human motion and the mutual influences among individuals. Deep learning models have recently achieved promising results by implicitly learning such patterns from 2D trajectory data. However, most approaches treat pedestrians as point entities, ignoring the physical space that each person occupies. To address these limitations, this paper proposes a novel deep learning model that enhances the Social LSTM with a new Dynamic Occupied Space loss function. This loss function guides Social LSTM in learning to avoid realistic collisions without increasing displacement error across different crowd densities, ranging from low to high, in both homogeneous and heterogeneous density settings. Such a function achieves this by combining the average displacement error with a new collision penalty that is sensitive to scene density and individual spatial occupancy. For efficient training and evaluation, five datasets were generated from real pedestrian trajectories recorded during the Festival of Lights in Lyon 2022. Four datasets represent homogeneous crowd conditions -- low, medium, high, and very high density -- while the fifth corresponds to a heterogeneous density distribution. The experimental findings indicate that the proposed model not only lowers collision rates but also enhances displacement prediction accuracy in each dataset. Specifically, the model achieves up to a 31% reduction in the collision rate and reduces the average displacement error and the final displacement error by 5% and 6%, respectively, on average across all datasets compared to the baseline. Moreover, the proposed model consistently outperforms several state-of-the-art deep learning models across most test sets.
>
---
#### [new 069] Towards Blind and Low-Vision Accessibility of Lightweight VLMs and Custom LLM-Evals
- **分类: cs.CV; cs.CL**

- **简介: 该论文面向盲低视用户，研究轻量级视觉语言模型（VLMs）的可访问性，提出双评估框架（多上下文与导航辅助），评估SmolVLM2不同规模模型在移动设备上的描述质量与部署性能，优化资源受限环境下的无障碍描述生成。**

- **链接: [https://arxiv.org/pdf/2511.10615v1](https://arxiv.org/pdf/2511.10615v1)**

> **作者:** Shruti Singh Baghel; Yash Pratap Singh Rathore; Sushovan Jena; Anurag Pradhan; Amit Shukla; Arnav Bhavsar; Pawan Goyal
>
> **备注:** 8 pages
>
> **摘要:** Large Vision-Language Models (VLMs) excel at understanding and generating video descriptions but their high memory, computation, and deployment demands hinder practical use particularly for blind and low-vision (BLV) users who depend on detailed, context-aware descriptions. To study the effect of model size on accessibility-focused description quality, we evaluate SmolVLM2 variants with 500M and 2.2B parameters across two diverse datasets: AVCaps (outdoor), and Charades (indoor). In this work, we introduce two novel evaluation frameworks specifically designed for BLV accessibility assessment: the Multi-Context BLV Framework evaluating spatial orientation, social interaction, action events, and ambience contexts; and the Navigational Assistance Framework focusing on mobility-critical information. Additionally, we conduct a systematic evaluation of four different prompt design strategies and deploy both models on a smartphone, evaluating FP32 and INT8 precision variants to assess real-world performance constraints on resource-limited mobile devices.
>
---
#### [new 070] From Street to Orbit: Training-Free Cross-View Retrieval via Location Semantics and LLM Guidance
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种无需训练的街景-卫星图像跨视图检索方法，利用LLM与预训练视觉编码器，通过地理语义推理自动生成卫星查询，实现零样本高精度匹配，并自动构建对齐数据集。**

- **链接: [https://arxiv.org/pdf/2511.09820v1](https://arxiv.org/pdf/2511.09820v1)**

> **作者:** Jeongho Min; Dongyoung Kim; Jaehyup Lee
>
> **备注:** Accepted to WACV 2026, 10pages, 4 figures
>
> **摘要:** Cross-view image retrieval, particularly street-to-satellite matching, is a critical task for applications such as autonomous navigation, urban planning, and localization in GPS-denied environments. However, existing approaches often require supervised training on curated datasets and rely on panoramic or UAV-based images, which limits real-world deployment. In this paper, we present a simple yet effective cross-view image retrieval framework that leverages a pretrained vision encoder and a large language model (LLM), requiring no additional training. Given a monocular street-view image, our method extracts geographic cues through web-based image search and LLM-based location inference, generates a satellite query via geocoding API, and retrieves matching tiles using a pretrained vision encoder (e.g., DINOv2) with PCA-based whitening feature refinement. Despite using no ground-truth supervision or finetuning, our proposed method outperforms prior learning-based approaches on the benchmark dataset under zero-shot settings. Moreover, our pipeline enables automatic construction of semantically aligned street-to-satellite datasets, which is offering a scalable and cost-efficient alternative to manual annotation. All source codes will be made publicly available at https://jeonghomin.github.io/street2orbit.github.io/.
>
---
#### [new 071] OmniVGGT: Omni-Modality Driven Visual Geometry Grounded
- **分类: cs.CV**

- **简介: OmniVGGT提出一种多模态几何感知的3D视觉基础模型，解决现有模型忽略相机参数与深度等几何信息的问题。通过GeoAdapter高效融合任意数量几何模态，并结合随机多模态融合策略，显著提升3D感知性能，且在RGB-only下达SOTA。**

- **链接: [https://arxiv.org/pdf/2511.10560v1](https://arxiv.org/pdf/2511.10560v1)**

> **作者:** Haosong Peng; Hao Li; Yalun Dai; Yushi Lan; Yihang Luo; Tianyu Qi; Zhengshen Zhang; Yufeng Zhan; Junfei Zhang; Wenchao Xu; Ziwei Liu
>
> **备注:** Project Page: https://livioni.github.io/OmniVGGT-offcial/
>
> **摘要:** General 3D foundation models have started to lead the trend of unifying diverse vision tasks, yet most assume RGB-only inputs and ignore readily available geometric cues (e.g., camera intrinsics, poses, and depth maps). To address this issue, we introduce OmniVGGT, a novel framework that can effectively benefit from an arbitrary number of auxiliary geometric modalities during both training and inference. In our framework, a GeoAdapter is proposed to encode depth and camera intrinsics/extrinsics into a spatial foundation model. It employs zero-initialized convolutions to progressively inject geometric information without disrupting the foundation model's representation space. This design ensures stable optimization with negligible overhead, maintaining inference speed comparable to VGGT even with multiple additional inputs. Additionally, a stochastic multimodal fusion regimen is proposed, which randomly samples modality subsets per instance during training. This enables an arbitrary number of modality inputs during testing and promotes learning robust spatial representations instead of overfitting to auxiliary cues. Comprehensive experiments on monocular/multi-view depth estimation, multi-view stereo, and camera pose estimation demonstrate that OmniVGGT outperforms prior methods with auxiliary inputs and achieves state-of-the-art results even with RGB-only input. To further highlight its practical utility, we integrated OmniVGGT into vision-language-action (VLA) models. The enhanced VLA model by OmniVGGT not only outperforms the vanilla point-cloud-based baseline on mainstream benchmarks, but also effectively leverages accessible auxiliary inputs to achieve consistent gains on robotic tasks.
>
---
#### [new 072] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data
- **分类: cs.CV**

- **简介: 论文提出STELLAR，用于低资源语言和真实场景文本编辑，解决风格保留不足、数据缺失与评估缺失问题。通过语言自适应编码器、多阶段训练和新数据集STIPLAR，提升视觉一致性与识别准确率，并引入新指标TAS评估风格相似性。**

- **链接: [https://arxiv.org/pdf/2511.09977v1](https://arxiv.org/pdf/2511.09977v1)**

> **作者:** Yongdeuk Seo; Hyun-seok Min; Sungchul Choi
>
> **备注:** Accepted to AAAI Workshop (Artificial Intelligence with Biased or Scarce Data)
>
> **摘要:** Scene Text Editing (STE) is the task of modifying text content in an image while preserving its visual style, such as font, color, and background. While recent diffusion-based approaches have shown improvements in visual quality, key limitations remain: lack of support for low-resource languages, domain gap between synthetic and real data, and the absence of appropriate metrics for evaluating text style preservation. To address these challenges, we propose STELLAR (Scene Text Editor for Low-resource LAnguages and Real-world data). STELLAR enables reliable multilingual editing through a language-adaptive glyph encoder and a multi-stage training strategy that first pre-trains on synthetic data and then fine-tunes on real images. We also construct a new dataset, STIPLAR(Scene Text Image Pairs of Low-resource lAnguages and Real-world data), for training and evaluation. Furthermore, we propose Text Appearance Similarity (TAS), a novel metric that assesses style preservation by independently measuring font, color, and background similarity, enabling robust evaluation even without ground truth. Experimental results demonstrate that STELLAR outperforms state-of-the-art models in visual consistency and recognition accuracy, achieving an average TAS improvement of 2.2% across languages over the baselines.
>
---
#### [new 073] TSPE-GS: Probabilistic Depth Extraction for Semi-Transparent Surface Reconstruction via 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: TSPE-GS针对3D高斯溅射在半透明表面重建中单深度假设的局限，提出概率性多模态深度提取方法，通过透射率采样与符号距离函数融合，实现内外表面分离重建，无需额外训练，显著提升半透明场景重建质量。**

- **链接: [https://arxiv.org/pdf/2511.09944v1](https://arxiv.org/pdf/2511.09944v1)**

> **作者:** Zhiyuan Xu; Nan Min; Yuhang Guo; Tong Wei
>
> **备注:** AAAI26 Poster
>
> **摘要:** 3D Gaussian Splatting offers a strong speed-quality trade-off but struggles to reconstruct semi-transparent surfaces because most methods assume a single depth per pixel, which fails when multiple surfaces are visible. We propose TSPE-GS (Transparent Surface Probabilistic Extraction for Gaussian Splatting), which uniformly samples transmittance to model a pixel-wise multi-modal distribution of opacity and depth, replacing the prior single-peak assumption and resolving cross-surface depth ambiguity. By progressively fusing truncated signed distance functions, TSPE-GS reconstructs external and internal surfaces separately within a unified framework. The method generalizes to other Gaussian-based reconstruction pipelines without extra training overhead. Extensive experiments on public and self-collected semi-transparent and opaque datasets show TSPE-GS significantly improves semi-transparent geometry reconstruction while maintaining performance on opaque scenes.
>
---
#### [new 074] CertMask: Certifiable Defense Against Adversarial Patches via Theoretically Optimal Mask Coverage
- **分类: cs.CV; cs.AI**

- **简介: CertMask提出一种高效可证明的对抗补丁防御方法，通过理论最优的单轮掩码覆盖，确保每个潜在补丁位置被覆盖至少k次，显著提升认证鲁棒性，同时保持高效推理（O(n)复杂度）。**

- **链接: [https://arxiv.org/pdf/2511.09834v1](https://arxiv.org/pdf/2511.09834v1)**

> **作者:** Xuntao Lyu; Ching-Chi Lin; Abdullah Al Arafat; Georg von der Brüggen; Jian-Jia Chen; Zhishan Guo
>
> **摘要:** Adversarial patch attacks inject localized perturbations into images to mislead deep vision models. These attacks can be physically deployed, posing serious risks to real-world applications. In this paper, we propose CertMask, a certifiably robust defense that constructs a provably sufficient set of binary masks to neutralize patch effects with strong theoretical guarantees. While the state-of-the-art approach (PatchCleanser) requires two rounds of masking and incurs $O(n^2)$ inference cost, CertMask performs only a single round of masking with $O(n)$ time complexity, where $n$ is the cardinality of the mask set to cover an input image. Our proposed mask set is computed using a mathematically rigorous coverage strategy that ensures each possible patch location is covered at least $k$ times, providing both efficiency and robustness. We offer a theoretical analysis of the coverage condition and prove its sufficiency for certification. Experiments on ImageNet, ImageNette, and CIFAR-10 show that CertMask improves certified robust accuracy by up to +13.4\% over PatchCleanser, while maintaining clean accuracy nearly identical to the vanilla model.
>
---
#### [new 075] AdaptViG: Adaptive Vision GNN with Exponential Decay Gating
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出AdaptViG，一种高效视觉图神经网络，解决传统ViG计算开销大的问题。通过自适应图卷积与指数衰减门控机制，动态权衡长程依赖，在保持高精度的同时显著降低参数与计算量。**

- **链接: [https://arxiv.org/pdf/2511.09942v1](https://arxiv.org/pdf/2511.09942v1)**

> **作者:** Mustafa Munir; Md Mostafijur Rahman; Radu Marculescu
>
> **备注:** Accepted in 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2026)
>
> **摘要:** Vision Graph Neural Networks (ViGs) offer a new direction for advancements in vision architectures. While powerful, ViGs often face substantial computational challenges stemming from their graph construction phase, which can hinder their efficiency. To address this issue we propose AdaptViG, an efficient and powerful hybrid Vision GNN that introduces a novel graph construction mechanism called Adaptive Graph Convolution. This mechanism builds upon a highly efficient static axial scaffold and a dynamic, content-aware gating strategy called Exponential Decay Gating. This gating mechanism selectively weighs long-range connections based on feature similarity. Furthermore, AdaptViG employs a hybrid strategy, utilizing our efficient gating mechanism in the early stages and a full Global Attention block in the final stage for maximum feature aggregation. Our method achieves a new state-of-the-art trade-off between accuracy and efficiency among Vision GNNs. For instance, our AdaptViG-M achieves 82.6% top-1 accuracy, outperforming ViG-B by 0.3% while using 80% fewer parameters and 84% fewer GMACs. On downstream tasks, AdaptViG-M obtains 45.8 mIoU, 44.8 APbox, and 41.1 APmask, surpassing the much larger EfficientFormer-L7 by 0.7 mIoU, 2.2 APbox, and 2.1 APmask, respectively, with 78% fewer parameters.
>
---
#### [new 076] Decoupling Bias, Aligning Distributions: Synergistic Fairness Optimization for Deepfake Detection
- **分类: cs.CV**

- **简介: 该论文针对深度伪造检测中的公平性问题，提出双机制优化框架：结构解耦敏感通道与特征分布对齐，在提升跨群体公平性的同时保持检测精度，实现公平与准确的协同优化。**

- **链接: [https://arxiv.org/pdf/2511.10150v1](https://arxiv.org/pdf/2511.10150v1)**

> **作者:** Feng Ding; Wenhui Yi; Yunpeng Zhou; Xinan He; Hong Rao; Shu Hu
>
> **摘要:** Fairness is a core element in the trustworthy deployment of deepfake detection models, especially in the field of digital identity security. Biases in detection models toward different demographic groups, such as gender and race, may lead to systemic misjudgments, exacerbating the digital divide and social inequities. However, current fairness-enhanced detectors often improve fairness at the cost of detection accuracy. To address this challenge, we propose a dual-mechanism collaborative optimization framework. Our proposed method innovatively integrates structural fairness decoupling and global distribution alignment: decoupling channels sensitive to demographic groups at the model architectural level, and subsequently reducing the distance between the overall sample distribution and the distributions corresponding to each demographic group at the feature level. Experimental results demonstrate that, compared with other methods, our framework improves both inter-group and intra-group fairness while maintaining overall detection accuracy across domains.
>
---
#### [new 077] When Eyes and Ears Disagree: Can MLLMs Discern Audio-Visual Confusion?
- **分类: cs.CV**

- **简介: 该论文研究多模态大模型在音视频冲突场景中的幻觉问题，提出AV-ConfuseBench基准与RL-CoMM方法，通过音频语言模型引导和强化学习优化，提升模型对缺失音频的辨别能力，准确率提升10~30%。**

- **链接: [https://arxiv.org/pdf/2511.10059v1](https://arxiv.org/pdf/2511.10059v1)**

> **作者:** Qilang Ye; Wei Zeng; Meng Liu; Jie Zhang; Yupeng Hu; Zitong Yu; Yu Zhou
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Can Multimodal Large Language Models (MLLMs) discern confused objects that are visually present but audio-absent? To study this, we introduce a new benchmark, AV-ConfuseBench, which simulates an ``Audio-Visual Confusion'' scene by modifying the corresponding sound of an object in the video, e.g., mute the sounding object and ask MLLMs Is there a/an muted-object sound''. Experimental results reveal that MLLMs, such as Qwen2.5-Omni and Gemini 2.5, struggle to discriminate non-existent audio due to visually dominated reasoning. Motivated by this observation, we introduce RL-CoMM, a Reinforcement Learning-based Collaborative Multi-MLLM that is built upon the Qwen2.5-Omni foundation. RL-CoMM includes two stages: 1) To alleviate visually dominated ambiguities, we introduce an external model, a Large Audio Language Model (LALM), as the reference model to generate audio-only reasoning. Then, we design a Step-wise Reasoning Reward function that enables MLLMs to self-improve audio-visual reasoning with the audio-only reference. 2) To ensure an accurate answer prediction, we introduce Answer-centered Confidence Optimization to reduce the uncertainty of potential heterogeneous reasoning differences. Extensive experiments on audio-visual question answering and audio-visual hallucination show that RL-CoMM improves the accuracy by 10~30\% over the baseline model with limited training data. Follow: https://github.com/rikeilong/AVConfusion.
>
---
#### [new 078] Gradient-Guided Exploration of Generative Model's Latent Space for Controlled Iris Image Augmentations
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出一种梯度引导的生成模型潜在空间遍历方法，用于可控的虹膜图像增强，在保持身份不变的前提下，精准调控虹膜纹理、尺寸等属性，解决真实多样性数据稀缺问题。**

- **链接: [https://arxiv.org/pdf/2511.09749v1](https://arxiv.org/pdf/2511.09749v1)**

> **作者:** Mahsa Mitcheff; Siamul Karim Khan; Adam Czajka
>
> **摘要:** Developing reliable iris recognition and presentation attack detection methods requires diverse datasets that capture realistic variations in iris features and a wide spectrum of anomalies. Because of the rich texture of iris images, which spans a wide range of spatial frequencies, synthesizing same-identity iris images while controlling specific attributes remains challenging. In this work, we introduce a new iris image augmentation strategy by traversing a generative model's latent space toward latent codes that represent same-identity samples but with some desired iris image properties manipulated. The latent space traversal is guided by a gradient of specific geometrical, textural, or quality-related iris image features (e.g., sharpness, pupil size, iris size, or pupil-to-iris ratio) and preserves the identity represented by the image being manipulated. The proposed approach can be easily extended to manipulate any attribute for which a differentiable loss term can be formulated. Additionally, our approach can use either randomly generated images using either a pre-train GAN model or real-world iris images. We can utilize GAN inversion to project any given iris image into the latent space and obtain its corresponding latent code.
>
---
#### [new 079] PANDA - Patch And Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning
- **分类: cs.CV; eess.IV**

- **简介: 该论文面向无样本持续学习任务，解决真实数据长尾分布导致的灾难性遗忘问题。提出PANDA框架，通过CLIP感知补丁与分布感知增强，平衡任务内/间类别偏差，提升预训练模型的公平学习能力。**

- **链接: [https://arxiv.org/pdf/2511.09791v1](https://arxiv.org/pdf/2511.09791v1)**

> **作者:** Siddeshwar Raghavan; Jiangpeng He; Fengqing Zhu
>
> **备注:** Accepted in AAAI 2026 Main Technical Track
>
> **摘要:** Exemplar-Free Continual Learning (EFCL) restricts the storage of previous task data and is highly susceptible to catastrophic forgetting. While pre-trained models (PTMs) are increasingly leveraged for EFCL, existing methods often overlook the inherent imbalance of real-world data distributions. We discovered that real-world data streams commonly exhibit dual-level imbalances, dataset-level distributions combined with extreme or reversed skews within individual tasks, creating both intra-task and inter-task disparities that hinder effective learning and generalization. To address these challenges, we propose PANDA, a Patch-and-Distribution-Aware Augmentation framework that integrates seamlessly with existing PTM-based EFCL methods. PANDA amplifies low-frequency classes by using a CLIP encoder to identify representative regions and transplanting those into frequent-class samples within each task. Furthermore, PANDA incorporates an adaptive balancing strategy that leverages prior task distributions to smooth inter-task imbalances, reducing the overall gap between average samples across tasks and enabling fairer learning with frozen PTMs. Extensive experiments and ablation studies demonstrate PANDA's capability to work with existing PTM-based CL methods, improving accuracy and reducing catastrophic forgetting.
>
---
#### [new 080] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文面向类增量学习任务，解决预训练ViT序列微调中的分布漂移问题，提出SLDC方法，通过潜在空间过渡算子对齐特征分布，并结合知识蒸馏，显著提升性能，逼近联合训练效果。**

- **链接: [https://arxiv.org/pdf/2511.09926v1](https://arxiv.org/pdf/2511.09926v1)**

> **作者:** Xuan Rao; Simian Xu; Zheng Li; Bo Zhao; Derong Liu; Mingming Ha; Cesare Alippi
>
> **备注:** The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Recent advances have shown that sequential fine-tuning (SeqFT) of pre-trained vision transformers (ViTs), followed by classifier refinement using approximate distributions of class features, can be an effective strategy for class-incremental learning (CIL). However, this approach is susceptible to distribution drift, caused by the sequential optimization of shared backbone parameters. This results in a mismatch between the distributions of the previously learned classes and that of the updater model, ultimately degrading the effectiveness of classifier performance over time. To address this issue, we introduce a latent space transition operator and propose Sequential Learning with Drift Compensation (SLDC). SLDC aims to align feature distributions across tasks to mitigate the impact of drift. First, we present a linear variant of SLDC, which learns a linear operator by solving a regularized least-squares problem that maps features before and after fine-tuning. Next, we extend this with a weakly nonlinear SLDC variant, which assumes that the ideal transition operator lies between purely linear and fully nonlinear transformations. This is implemented using learnable, weakly nonlinear mappings that balance flexibility and generalization. To further reduce representation drift, we apply knowledge distillation (KD) in both algorithmic variants. Extensive experiments on standard CIL benchmarks demonstrate that SLDC significantly improves the performance of SeqFT. Notably, by combining KD to address representation drift with SLDC to compensate distribution drift, SeqFT achieves performance comparable to joint training across all evaluated datasets. Code: https://github.com/raoxuan98-hash/sldc.git.
>
---
#### [new 081] IPCD: Intrinsic Point-Cloud Decomposition
- **分类: cs.CV**

- **简介: 论文提出IPCD方法，首次在点云上直接分解反照率与阴影，解决非网格结构与全局光照缺失问题。设计IPCD-Net与PLD模块，提升分解精度，并在合成与真实数据上验证了其在重光照、纹理编辑等任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.09866v1](https://arxiv.org/pdf/2511.09866v1)**

> **作者:** Shogo Sato; Takuhiro Kaneko; Shoichiro Takeda; Tomoyasu Shimada; Kazuhiko Murasaki; Taiga Yoshida; Ryuichi Tanida; Akisato Kimura
>
> **备注:** Accepted in WACV2026
>
> **摘要:** Point clouds are widely used in various fields, including augmented reality (AR) and robotics, where relighting and texture editing are crucial for realistic visualization. Achieving these tasks requires accurately separating albedo from shade. However, performing this separation on point clouds presents two key challenges: (1) the non-grid structure of point clouds makes conventional image-based decomposition models ineffective, and (2) point-cloud models designed for other tasks do not explicitly consider global-light direction, resulting in inaccurate shade. In this paper, we introduce \textbf{Intrinsic Point-Cloud Decomposition (IPCD)}, which extends image decomposition to the direct decomposition of colored point clouds into albedo and shade. To overcome challenge (1), we propose \textbf{IPCD-Net} that extends image-based model with point-wise feature aggregation for non-grid data processing. For challenge (2), we introduce \textbf{Projection-based Luminance Distribution (PLD)} with a hierarchical feature refinement, capturing global-light ques via multi-view projection. For comprehensive evaluation, we create a synthetic outdoor-scene dataset. Experimental results demonstrate that IPCD-Net reduces cast shadows in albedo and enhances color accuracy in shade. Furthermore, we showcase its applications in texture editing, relighting, and point-cloud registration under varying illumination. Finally, we verify the real-world applicability of IPCD-Net.
>
---
#### [new 082] Multitask GLocal OBIA-Mamba for Sentinel-2 Landcover Mapping
- **分类: cs.CV; cs.LG**

- **简介: 该论文面向Sentinel-2遥感影像土地覆盖分类任务，解决空间异质性与语义模糊问题。提出Multitask GLocal OBIA-Mamba模型，融合超像素Mamba令牌、全局-局部双分支架构与多任务优化，提升分类精度与细节保留。**

- **链接: [https://arxiv.org/pdf/2511.10604v1](https://arxiv.org/pdf/2511.10604v1)**

> **作者:** Zack Dewis; Yimin Zhu; Zhengsen Xu; Mabel Heffring; Saeid Taleghanidoozdoozan; Kaylee Xiao; Motasem Alkayid; Lincoln Linlin Xu
>
> **摘要:** Although Sentinel-2 based land use and land cover (LULC) classification is critical for various environmental monitoring applications, it is a very difficult task due to some key data challenges (e.g., spatial heterogeneity, context information, signature ambiguity). This paper presents a novel Multitask Glocal OBIA-Mamba (MSOM) for enhanced Sentinel-2 classification with the following contributions. First, an object-based image analysis (OBIA) Mamba model (OBIA-Mamba) is designed to reduce redundant computation without compromising fine-grained details by using superpixels as Mamba tokens. Second, a global-local (GLocal) dual-branch convolutional neural network (CNN)-mamba architecture is designed to jointly model local spatial detail and global contextual information. Third, a multitask optimization framework is designed to employ dual loss functions to balance local precision with global consistency. The proposed approach is tested on Sentinel-2 imagery in Alberta, Canada, in comparison with several advanced classification approaches, and the results demonstrate that the proposed approach achieves higher classification accuracy and finer details that the other state-of-the-art methods.
>
---
#### [new 083] Multivariate Gaussian Representation Learning for Medical Action Evaluation
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对医疗动作评估中数据稀缺与动态建模不足的问题，提出GaussMedAct框架，利用多变量高斯编码建模时空动作特征，在自建CPREval-6k基准上实现92.1%准确率，显著优于基线模型。**

- **链接: [https://arxiv.org/pdf/2511.10060v1](https://arxiv.org/pdf/2511.10060v1)**

> **作者:** Luming Yang; Haoxian Liu; Siqing Li; Alper Yilmaz
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Fine-grained action evaluation in medical vision faces unique challenges due to the unavailability of comprehensive datasets, stringent precision requirements, and insufficient spatiotemporal dynamic modeling of very rapid actions. To support development and evaluation, we introduce CPREval-6k, a multi-view, multi-label medical action benchmark containing 6,372 expert-annotated videos with 22 clinical labels. Using this dataset, we present GaussMedAct, a multivariate Gaussian encoding framework, to advance medical motion analysis through adaptive spatiotemporal representation learning. Multivariate Gaussian Representation projects the joint motions to a temporally scaled multi-dimensional space, and decomposes actions into adaptive 3D Gaussians that serve as tokens. These tokens preserve motion semantics through anisotropic covariance modeling while maintaining robustness to spatiotemporal noise. Hybrid Spatial Encoding, employing a Cartesian and Vector dual-stream strategy, effectively utilizes skeletal information in the form of joint and bone features. The proposed method achieves 92.1% Top-1 accuracy with real-time inference on the benchmark, outperforming the ST-GCN baseline by +5.9% accuracy with only 10% FLOPs. Cross-dataset experiments confirm the superiority of our method in robustness.
>
---
#### [new 084] LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers
- **分类: cs.CV**

- **简介: 论文提出LampQ，针对Vision Transformers的层间混合精度量化问题，解决均匀量化导致的精度损失。通过层粒度敏感度度量与整数规划优化比特分配，实现高精度、低开销量化，显著提升多任务性能。**

- **链接: [https://arxiv.org/pdf/2511.10004v1](https://arxiv.org/pdf/2511.10004v1)**

> **作者:** Minjun Kim; Jaeri Lee; Jongjin Kim; Jeongin Yun; Yongmo Kwon; U Kang
>
> **备注:** AAAI 2026
>
> **摘要:** How can we accurately quantize a pre-trained Vision Transformer model? Quantization algorithms compress Vision Transformers (ViTs) into low-bit formats, reducing memory and computation demands with minimal accuracy degradation. However, existing methods rely on uniform precision, ignoring the diverse sensitivity of ViT components to quantization. Metric-based Mixed Precision Quantization (MPQ) is a promising alternative, but previous MPQ methods for ViTs suffer from three major limitations: 1) coarse granularity, 2) mismatch in metric scale across component types, and 3) quantization-unaware bit allocation. In this paper, we propose LampQ (Layer-wise Mixed Precision Quantization for Vision Transformers), an accurate metric-based MPQ method for ViTs to overcome these limitations. LampQ performs layer-wise quantization to achieve both fine-grained control and efficient acceleration, incorporating a type-aware Fisher-based metric to measure sensitivity. Then, LampQ assigns bit-widths optimally through integer linear programming and further updates them iteratively. Extensive experiments show that LampQ provides the state-of-the-art performance in quantizing ViTs pre-trained on various tasks such as image classification, object detection, and zero-shot quantization.
>
---
#### [new 085] DermAI: Clinical dermatology acquisition through quality-driven image collection for AI classification in mobile
- **分类: cs.CV; cs.AI**

- **简介: DermAI是一款移动端皮肤病变检测应用，旨在解决AI皮肤病诊断中数据偏差与质量低的问题。通过实时采集多样化皮肤图像并进行本地质量校验与模型微调，提升模型在真实临床场景中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.10367v1](https://arxiv.org/pdf/2511.10367v1)**

> **作者:** Thales Bezerra; Emanoel Thyago; Kelvin Cunha; Rodrigo Abreu; Fábio Papais; Francisco Mauro; Natália Lopes; Érico Medeiros; Jéssica Guido; Shirley Cruz; Paulo Borba; Tsang Ing Ren
>
> **备注:** 4 pages, 2 figures, 1 table, submitted on ISBI
>
> **摘要:** AI-based dermatology adoption remains limited by biased datasets, variable image quality, and limited validation. We introduce DermAI, a lightweight, smartphone-based application that enables real-time capture, annotation, and classification of skin lesions during routine consultations. Unlike prior dermoscopy-focused tools, DermAI performs on-device quality checks, and local model adaptation. The DermAI clinical dataset, encompasses a wide range of skin tones, ethinicity and source devices. In preliminary experiments, models trained on public datasets failed to generalize to our samples, while fine-tuning with local data improved performance. These results highlight the importance of standardized, diverse data collection aligned with healthcare needs and oriented to machine learning development.
>
---
#### [new 086] RodEpil: A Video Dataset of Laboratory Rodents for Seizure Detection and Benchmark Evaluation
- **分类: cs.CV**

- **简介: 论文提出RodEpil视频数据集，用于实验室啮齿动物癫痫发作的非侵入式视频检测。任务为二分类（发作/正常），通过TimeSformer模型实现97% F1-score，公开数据与代码支持可复现研究。**

- **链接: [https://arxiv.org/pdf/2511.10431v1](https://arxiv.org/pdf/2511.10431v1)**

> **作者:** Daniele Perlo; Vladimir Despotovic; Selma Boudissa; Sang-Yoon Kim; Petr Nazarov; Yanrong Zhang; Max Wintermark; Olivier Keunen
>
> **摘要:** We introduce a curated video dataset of laboratory rodents for automatic detection of convulsive events. The dataset contains short (10~s) top-down and side-view video clips of individual rodents, labeled at clip level as normal activity or seizure. It includes 10,101 negative samples and 2,952 positive samples collected from 19 subjects. We describe the data curation, annotation protocol and preprocessing pipeline, and report baseline experiments using a transformer-based video classifier (TimeSformer). Experiments employ five-fold cross-validation with strict subject-wise partitioning to prevent data leakage (no subject appears in more than one fold). Results show that the TimeSformer architecture enables discrimination between seizure and normal activity with an average F1-score of 97%. The dataset and baseline code are publicly released to support reproducible research on non-invasive, video-based monitoring in preclinical epilepsy research. RodEpil Dataset access - DOI: 10.5281/zenodo.17601357
>
---
#### [new 087] HCC-3D: Hierarchical Compensatory Compression for 98% 3D Token Reduction in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文面向3D视觉-语言模型，解决3D点云令牌计算开销大的问题，提出分层补偿压缩（HCC-3D），通过全局结构压缩与自适应细节挖掘，实现98%令牌压缩且性能提升。**

- **链接: [https://arxiv.org/pdf/2511.09883v1](https://arxiv.org/pdf/2511.09883v1)**

> **作者:** Liheng Zhang; Jin Wang; Hui Li; Bingfeng Zhang; Weifeng Liu
>
> **摘要:** 3D understanding has drawn significant attention recently, leveraging Vision-Language Models (VLMs) to enable multi-modal reasoning between point cloud and text data. Current 3D-VLMs directly embed the 3D point clouds into 3D tokens, following large 2D-VLMs with powerful reasoning capabilities. However, this framework has a great computational cost limiting its application, where we identify that the bottleneck lies in processing all 3D tokens in the Large Language Model (LLM) part. This raises the question: how can we reduce the computational overhead introduced by 3D tokens while preserving the integrity of their essential information? To address this question, we introduce Hierarchical Compensatory Compression (HCC-3D) to efficiently compress 3D tokens while maintaining critical detail retention. Specifically, we first propose a global structure compression (GSC), in which we design global queries to compress all 3D tokens into a few key tokens while keeping overall structural information. Then, to compensate for the information loss in GSC, we further propose an adaptive detail mining (ADM) module that selectively recompresses salient but under-attended features through complementary scoring. Extensive experiments demonstrate that HCC-3D not only achieves extreme compression ratios (approximately 98%) compared to previous 3D-VLMs, but also achieves new state-of-the-art performance, showing the great improvements on both efficiency and performance.
>
---
#### [new 088] LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures
- **分类: cs.CV**

- **简介: LiNeXt提出一种非扩散架构，用于激光雷达点云补全任务，解决扩散模型计算慢的问题。通过Noise-to-Coarse与Refine模块实现单次高效补全，并引入距离感知策略提升分布均匀性，显著提升速度与精度。**

- **链接: [https://arxiv.org/pdf/2511.10209v1](https://arxiv.org/pdf/2511.10209v1)**

> **作者:** Wenzhe He; Xiaojun Chen; Ruiqi Wang; Ruihui Li; Huilong Pi; Jiapeng Zhang; Zhuo Tang; Kenli Li
>
> **备注:** 18 pages, 13 figures, Accepted to AAAI 2026
>
> **摘要:** 3D LiDAR scene completion from point clouds is a fundamental component of perception systems in autonomous vehicles. Previous methods have predominantly employed diffusion models for high-fidelity reconstruction. However, their multi-step iterative sampling incurs significant computational overhead, limiting its real-time applicability. To address this, we propose LiNeXt-a lightweight, non-diffusion network optimized for rapid and accurate point cloud completion. Specifically, LiNeXt first applies the Noise-to-Coarse (N2C) Module to denoise the input noisy point cloud in a single pass, thereby obviating the multi-step iterative sampling of diffusion-based methods. The Refine Module then takes the coarse point cloud and its intermediate features from the N2C Module to perform more precise refinement, further enhancing structural completeness. Furthermore, we observe that LiDAR point clouds exhibit a distance-dependent spatial distribution, being densely sampled at proximal ranges and sparsely sampled at distal ranges. Accordingly, we propose the Distance-aware Selected Repeat strategy to generate a more uniformly distributed noisy point cloud. On the SemanticKITTI dataset, LiNeXt achieves a 199.8x speedup in inference, reduces Chamfer Distance by 50.7%, and uses only 6.1% of the parameters compared with LiDiff. These results demonstrate the superior efficiency and effectiveness of LiNeXt for real-time scene completion.
>
---
#### [new 089] VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: VISTA提出一种视觉与意图感知的社会注意力框架，用于多智能体轨迹预测，解决现有方法忽视长期目标与细粒度交互的问题，通过递归目标条件Transformer实现更真实、可解释且安全的轨迹生成。**

- **链接: [https://arxiv.org/pdf/2511.10203v1](https://arxiv.org/pdf/2511.10203v1)**

> **作者:** Stephane Da Silva Martins; Emanuel Aldea; Sylvie Le Hégarat-Mascle
>
> **备注:** Paper accepted at WACV 2026
>
> **摘要:** Multi-agent trajectory prediction is crucial for autonomous systems operating in dense, interactive environments. Existing methods often fail to jointly capture agents' long-term goals and their fine-grained social interactions, which leads to unrealistic multi-agent futures. We propose VISTA, a recursive goal-conditioned transformer for multi-agent trajectory forecasting. VISTA combines (i) a cross-attention fusion module that integrates long-horizon intent with past motion, (ii) a social-token attention mechanism for flexible interaction modeling across agents, and (iii) pairwise attention maps that make social influence patterns interpretable at inference time. Our model turns single-agent goal-conditioned prediction into a coherent multi-agent forecasting framework. Beyond standard displacement metrics, we evaluate trajectory collision rates as a measure of joint realism. On the high-density MADRAS benchmark and on SDD, VISTA achieves state-of-the-art accuracy and substantially fewer collisions. On MADRAS, it reduces the average collision rate of strong baselines from 2.14 to 0.03 percent, and on SDD it attains zero collisions while improving ADE, FDE, and minFDE. These results show that VISTA generates socially compliant, goal-aware, and interpretable trajectories, making it promising for safety-critical autonomous systems.
>
---
#### [new 090] Physically Interpretable Multi-Degradation Image Restoration via Deep Unfolding and Explainable Convolution
- **分类: cs.CV**

- **简介: 该论文针对多退化图像恢复任务，提出InterIR模型，结合深度展开与可解释卷积，将优化算法物理可解释化，并模拟人脑自适应机制，实现多类型退化（如雨、噪、雾）的高效恢复，兼顾性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.10166v1](https://arxiv.org/pdf/2511.10166v1)**

> **作者:** Hu Gao; Xiaoning Lei; Xichen Xu; Depeng Dang; Lizhuang Ma
>
> **摘要:** Although image restoration has advanced significantly, most existing methods target only a single type of degradation. In real-world scenarios, images often contain multiple degradations simultaneously, such as rain, noise, and haze, requiring models capable of handling diverse degradation types. Moreover, methods that improve performance through module stacking often suffer from limited interpretability. In this paper, we propose a novel interpretability-driven approach for multi-degradation image restoration, built upon a deep unfolding network that maps the iterative process of a mathematical optimization algorithm into a learnable network structure. Specifically, we employ an improved second-order semi-smooth Newton algorithm to ensure that each module maintains clear physical interpretability. To further enhance interpretability and adaptability, we design an explainable convolution module inspired by the human brain's flexible information processing and the intrinsic characteristics of images, allowing the network to flexibly leverage learned knowledge and autonomously adjust parameters for different input. The resulting tightly integrated architecture, named InterIR, demonstrates excellent performance in multi-degradation restoration while remaining highly competitive on single-degradation tasks.
>
---
#### [new 091] Depth Anything 3: Recovering the Visual Space from Any Views
- **分类: cs.CV**

- **简介: Depth Anything 3 提出一种通用深度估计模型，仅用简单Transformer即可从任意数量图像恢复一致三维几何，无需已知相机位姿，显著提升位姿与几何估计精度，超越前代模型DA2及SOTA方法。**

- **链接: [https://arxiv.org/pdf/2511.10647v1](https://arxiv.org/pdf/2511.10647v1)**

> **作者:** Haotong Lin; Sili Chen; Junhao Liew; Donny Y. Chen; Zhenyu Li; Guang Shi; Jiashi Feng; Bingyi Kang
>
> **备注:** https://depth-anything-3.github.io/
>
> **摘要:** We present Depth Anything 3 (DA3), a model that predicts spatially consistent geometry from an arbitrary number of visual inputs, with or without known camera poses. In pursuit of minimal modeling, DA3 yields two key insights: a single plain transformer (e.g., vanilla DINO encoder) is sufficient as a backbone without architectural specialization, and a singular depth-ray prediction target obviates the need for complex multi-task learning. Through our teacher-student training paradigm, the model achieves a level of detail and generalization on par with Depth Anything 2 (DA2). We establish a new visual geometry benchmark covering camera pose estimation, any-view geometry and visual rendering. On this benchmark, DA3 sets a new state-of-the-art across all tasks, surpassing prior SOTA VGGT by an average of 44.3% in camera pose accuracy and 25.1% in geometric accuracy. Moreover, it outperforms DA2 in monocular depth estimation. All models are trained exclusively on public academic datasets.
>
---
#### [new 092] PALMS+: Modular Image-Based Floor Plan Localization Leveraging Depth Foundation Model
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: PALMS+提出一种无基础设施的视觉定位方法，利用单目深度模型重建3D点云，通过几何匹配实现高精度室内外定位，无需训练，显著优于PALMS和F3Loc。**

- **链接: [https://arxiv.org/pdf/2511.09724v1](https://arxiv.org/pdf/2511.09724v1)**

> **作者:** Yunqian Cheng; Benjamin Princen; Roberto Manduchi
>
> **备注:** Accepted to IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026, Application Track. Main paper: 8 pages, 5 figures. Supplementary material included
>
> **摘要:** Indoor localization in GPS-denied environments is crucial for applications like emergency response and assistive navigation. Vision-based methods such as PALMS enable infrastructure-free localization using only a floor plan and a stationary scan, but are limited by the short range of smartphone LiDAR and ambiguity in indoor layouts. We propose PALMS$+$, a modular, image-based system that addresses these challenges by reconstructing scale-aligned 3D point clouds from posed RGB images using a foundation monocular depth estimation model (Depth Pro), followed by geometric layout matching via convolution with the floor plan. PALMS$+$ outputs a posterior over the location and orientation, usable for direct or sequential localization. Evaluated on the Structured3D and a custom campus dataset consisting of 80 observations across four large campus buildings, PALMS$+$ outperforms PALMS and F3Loc in stationary localization accuracy -- without requiring any training. Furthermore, when integrated with a particle filter for sequential localization on 33 real-world trajectories, PALMS$+$ achieved lower localization errors compared to other methods, demonstrating robustness for camera-free tracking and its potential for infrastructure-free applications. Code and data are available at https://github.com/Head-inthe-Cloud/PALMS-Plane-based-Accessible-Indoor-Localization-Using-Mobile-Smartphones
>
---
#### [new 093] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 论文提出Fine-grained 3D Embodied Reasoning任务，旨在让AI代理根据指令精准定位3D场景中可交互元素的空间位置、运动类型与轴向。作者构建AffordBot框架，结合多模态大模型与链式推理，通过全景渲染实现3D到2D的对齐，显著提升物理交互推理能力。**

- **链接: [https://arxiv.org/pdf/2511.10017v1](https://arxiv.org/pdf/2511.10017v1)**

> **作者:** Xinyi Wang; Xun Yang; Yanlong Xu; Yuchen Wu; Zhen Li; Na Zhao
>
> **备注:** NeurIPS 2025
>
> **摘要:** Effective human-agent collaboration in physical environments requires understanding not only what to act upon, but also where the actionable elements are and how to interact with them. Existing approaches often operate at the object level or disjointedly handle fine-grained affordance reasoning, lacking coherent, instruction-driven grounding and reasoning. In this work, we introduce a new task: Fine-grained 3D Embodied Reasoning, which requires an agent to predict, for each referenced affordance element in a 3D scene, a structured triplet comprising its spatial location, motion type, and motion axis, based on a task instruction. To solve this task, we propose AffordBot, a novel framework that integrates Multimodal Large Language Models (MLLMs) with a tailored chain-of-thought (CoT) reasoning paradigm. To bridge the gap between 3D input and 2D-compatible MLLMs, we render surround-view images of the scene and project 3D element candidates into these views, forming a rich visual representation aligned with the scene geometry. Our CoT pipeline begins with an active perception stage, prompting the MLLM to select the most informative viewpoint based on the instruction, before proceeding with step-by-step reasoning to localize affordance elements and infer plausible interaction motions. Evaluated on the SceneFun3D dataset, AffordBot achieves state-of-the-art performance, demonstrating strong generalization and physically grounded reasoning with only 3D point cloud input and MLLMs.
>
---
#### [new 094] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Spectrum-Aware Test-Time Steering (STS)，用于视觉-语言模型在测试时的零样本泛化。通过在潜空间中仅调整少量参数，引导文本嵌入的谱子空间以降低熵，无需反向传播或修改编码器，实现高效轻量级自适应。**

- **链接: [https://arxiv.org/pdf/2511.09809v1](https://arxiv.org/pdf/2511.09809v1)**

> **作者:** Konstantinos M. Dafnis; Dimitris N. Metaxas
>
> **备注:** NeurIPS 2025
>
> **摘要:** Vision-Language Models (VLMs) excel at zero-shot inference but often degrade under test-time domain shifts. For this reason, episodic test-time adaptation strategies have recently emerged as powerful techniques for adapting VLMs to a single unlabeled image. However, existing adaptation strategies, such as test-time prompt tuning, typically require backpropagating through large encoder weights or altering core model components. In this work, we introduce Spectrum-Aware Test-Time Steering (STS), a lightweight adaptation framework that extracts a spectral subspace from the textual embeddings to define principal semantic directions and learns to steer latent representations in a spectrum-aware manner by adapting a small number of per-sample shift parameters to minimize entropy across augmented views. STS operates entirely at inference in the latent space, without backpropagation through or modification of the frozen encoders. Building on standard evaluation protocols, our comprehensive experiments demonstrate that STS largely surpasses or compares favorably against state-of-the-art test-time adaptation methods, while introducing only a handful of additional parameters and achieving inference speeds up to 8x faster with a 12x smaller memory footprint than conventional test-time prompt tuning. The code is available at https://github.com/kdafnis/STS.
>
---
#### [new 095] DBGroup: Dual-Branch Point Grouping for Weakly Supervised 3D Instance Segmentation
- **分类: cs.CV**

- **简介: 论文提出DBGroup，用于弱监督3D实例分割，解决标注成本高的问题。利用场景级标注，通过双分支点分组生成伪标签，并结合自训练与过滤策略提升分割精度，性能超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10003v1](https://arxiv.org/pdf/2511.10003v1)**

> **作者:** Xuexun Liu; Xiaoxu Xu; Qiudan Zhang; Lin Ma; Xu Wang
>
> **摘要:** Weakly supervised 3D instance segmentation is essential for 3D scene understanding, especially as the growing scale of data and high annotation costs associated with fully supervised approaches. Existing methods primarily rely on two forms of weak supervision: one-thing-one-click annotations and bounding box annotations, both of which aim to reduce labeling efforts. However, these approaches still encounter limitations, including labor-intensive annotation processes, high complexity, and reliance on expert annotators. To address these challenges, we propose \textbf{DBGroup}, a two-stage weakly supervised 3D instance segmentation framework that leverages scene-level annotations as a more efficient and scalable alternative. In the first stage, we introduce a Dual-Branch Point Grouping module to generate pseudo labels guided by semantic and mask cues extracted from multi-view images. To further improve label quality, we develop two refinement strategies: Granularity-Aware Instance Merging and Semantic Selection and Propagation. The second stage involves multi-round self-training on an end-to-end instance segmentation network using the refined pseudo-labels. Additionally, we introduce an Instance Mask Filter strategy to address inconsistencies within the pseudo labels. Extensive experiments demonstrate that DBGroup achieves competitive performance compared to sparse-point-level supervised 3D instance segmentation methods, while surpassing state-of-the-art scene-level supervised 3D semantic segmentation approaches. Code is available at https://github.com/liuxuexun/DBGroup.
>
---
#### [new 096] Revisiting Evaluation of Deep Neural Networks for Pedestrian Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对行人检测模型评估不足的问题，提出八类错误分类与新评估指标，利用图像分割实现细粒度性能分析，提升安全关键场景下的模型对比可靠性，并在CityPersons上以简单架构达到SOTA。**

- **链接: [https://arxiv.org/pdf/2511.10308v1](https://arxiv.org/pdf/2511.10308v1)**

> **作者:** Patrick Feifel; Benedikt Franke; Frank Bonarens; Frank Köster; Arne Raulf; Friedhelm Schwenker
>
> **摘要:** Reliable pedestrian detection represents a crucial step towards automated driving systems. However, the current performance benchmarks exhibit weaknesses. The currently applied metrics for various subsets of a validation dataset prohibit a realistic performance evaluation of a DNN for pedestrian detection. As image segmentation supplies fine-grained information about a street scene, it can serve as a starting point to automatically distinguish between different types of errors during the evaluation of a pedestrian detector. In this work, eight different error categories for pedestrian detection are proposed and new metrics are proposed for performance comparison along these error categories. We use the new metrics to compare various backbones for a simplified version of the APD, and show a more fine-grained and robust way to compare models with each other especially in terms of safety-critical performance. We achieve SOTA on CityPersons-reasonable (without extra training data) by using a rather simple architecture.
>
---
#### [new 097] Lumos3D: A Single-Forward Framework for Low-Light 3D Scene Restoration
- **分类: cs.CV**

- **简介: Lumos3D提出一种无位姿、单次训练的前向框架，解决低光条件下3D场景重建的可扩展性难题，通过几何引导的高斯表示与跨光照蒸馏，实现高质量、泛化性强的光照与结构恢复。**

- **链接: [https://arxiv.org/pdf/2511.09818v1](https://arxiv.org/pdf/2511.09818v1)**

> **作者:** Hanzhou Liu; Peng Jiang; Jia Huang; Mi Lu
>
> **摘要:** Restoring 3D scenes captured under low-light con- ditions remains a fundamental yet challenging problem. Most existing approaches depend on precomputed camera poses and scene-specific optimization, which greatly restricts their scala- bility to dynamic real-world environments. To overcome these limitations, we introduce Lumos3D, a generalizable pose-free framework for 3D low-light scene restoration. Trained once on a single dataset, Lumos3D performs inference in a purely feed- forward manner, directly restoring illumination and structure from unposed, low-light multi-view images without any per- scene training or optimization. Built upon a geometry-grounded backbone, Lumos3D reconstructs a normal-light 3D Gaussian representation that restores illumination while faithfully pre- serving structural details. During training, a cross-illumination distillation scheme is employed, where the teacher network is distilled on normal-light ground truth to transfer accurate geometric information, such as depth, to the student model. A dedicated Lumos loss is further introduced to promote photomet- ric consistency within the reconstructed 3D space. Experiments on real-world datasets demonstrate that Lumos3D achieves high- fidelity low-light 3D scene restoration with accurate geometry and strong generalization to unseen cases. Furthermore, the framework naturally extends to handle over-exposure correction, highlighting its versatility for diverse lighting restoration tasks.
>
---
#### [new 098] FreDFT: Frequency Domain Fusion Transformer for Visible-Infrared Object Detection
- **分类: cs.CV**

- **简介: 该论文针对可见光-红外目标检测中的模态信息不平衡与跨模态融合不足问题，提出FreDFT，通过频域注意力与混合尺度特征融合策略，提升多模态特征表达与检测性能。**

- **链接: [https://arxiv.org/pdf/2511.10046v1](https://arxiv.org/pdf/2511.10046v1)**

> **作者:** Wencong Wu; Xiuwei Zhang; Hanlin Yin; Shun Dai; Hongxi Zhang; Yanning Zhang
>
> **摘要:** Visible-infrared object detection has gained sufficient attention due to its detection performance in low light, fog, and rain conditions. However, visible and infrared modalities captured by different sensors exist the information imbalance problem in complex scenarios, which can cause inadequate cross-modal fusion, resulting in degraded detection performance. \textcolor{red}{Furthermore, most existing methods use transformers in the spatial domain to capture complementary features, ignoring the advantages of developing frequency domain transformers to mine complementary information.} To solve these weaknesses, we propose a frequency domain fusion transformer, called FreDFT, for visible-infrared object detection. The proposed approach employs a novel multimodal frequency domain attention (MFDA) to mine complementary information between modalities and a frequency domain feed-forward layer (FDFFL) via a mixed-scale frequency feature fusion strategy is designed to better enhance multimodal features. To eliminate the imbalance of multimodal information, a cross-modal global modeling module (CGMM) is constructed to perform pixel-wise inter-modal feature interaction in a spatial and channel manner. Moreover, a local feature enhancement module (LFEM) is developed to strengthen multimodal local feature representation and promote multimodal feature fusion by using various convolution layers and applying a channel shuffle. Extensive experimental results have verified that our proposed FreDFT achieves excellent performance on multiple public datasets compared with other state-of-the-art methods. The code of our FreDFT is linked at https://github.com/WenCongWu/FreDFT.
>
---
#### [new 099] OpenSR-SRGAN: A Flexible Super-Resolution Framework for Multispectral Earth Observation Data
- **分类: cs.CV; cs.LG**

- **简介: 论文提出OpenSR-SRGAN，一个面向多光谱遥感数据的可配置超分辨率框架，解决SRGAN模型部署复杂的问题，通过配置文件实现模型、损失函数和训练流程的灵活切换，降低使用门槛，支持可复现对比与工程部署。**

- **链接: [https://arxiv.org/pdf/2511.10461v1](https://arxiv.org/pdf/2511.10461v1)**

> **作者:** Simon Donike; Cesar Aybar; Julio Contreras; Luis Gómez-Chova
>
> **摘要:** We present OpenSR-SRGAN, an open and modular framework for single-image super-resolution in Earth Observation. The software provides a unified implementation of SRGAN-style models that is easy to configure, extend, and apply to multispectral satellite data such as Sentinel-2. Instead of requiring users to modify model code, OpenSR-SRGAN exposes generators, discriminators, loss functions, and training schedules through concise configuration files, making it straightforward to switch between architectures, scale factors, and band setups. The framework is designed as a practical tool and benchmark implementation rather than a state-of-the-art model. It ships with ready-to-use configurations for common remote sensing scenarios, sensible default settings for adversarial training, and built-in hooks for logging, validation, and large-scene inference. By turning GAN-based super-resolution into a configuration-driven workflow, OpenSR-SRGAN lowers the entry barrier for researchers and practitioners who wish to experiment with SRGANs, compare models in a reproducible way, and deploy super-resolution pipelines across diverse Earth-observation datasets.
>
---
#### [new 100] Utilizing a Geospatial Foundation Model for Coastline Delineation in Small Sandy Islands
- **分类: cs.CV; cs.AI**

- **简介: 该论文利用Prithvi-EO-2.0地理空间基础模型，面向小沙岛海岸线提取任务，仅用少量标注图像（最少5张）即实现高精度分割（F1=0.94），验证了其在数据匮乏地区的迁移能力，并公开发布了225幅标注图像数据集。**

- **链接: [https://arxiv.org/pdf/2511.10177v1](https://arxiv.org/pdf/2511.10177v1)**

> **作者:** Tishya Chhabra; Manisha Bajpai; Walter Zesk; Skylar Tibbits
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** We present an initial evaluation of NASA and IBM's Prithvi-EO-2.0 geospatial foundation model on shoreline delineation of small sandy islands using satellite images. We curated and labeled a dataset of 225 multispectral images of two Maldivian islands, which we publicly release, and fine-tuned both the 300M and 600M parameter versions of Prithvi on training subsets ranging from 5 to 181 images. Our experiments show that even with as few as 5 training images, the models achieve high performance (F1 of 0.94, IoU of 0.79). Our results demonstrate the strong transfer learning capability of Prithvi, underscoring the potential of such models to support coastal monitoring in data-poor regions.
>
---
#### [new 101] CORONA-Fields: Leveraging Foundation Models for Classification of Solar Wind Phenomena
- **分类: cs.CV; astro-ph.IM; astro-ph.SR**

- **简介: 该论文将太阳物理基础模型嵌入与航天器位置、磁场连接性结合，构建神经场模型，实现太阳风结构的分类，首次验证了基础模型用于原位太阳风数据分析的可行性，助力空间天气预测。**

- **链接: [https://arxiv.org/pdf/2511.09843v1](https://arxiv.org/pdf/2511.09843v1)**

> **作者:** Daniela Martin; Jinsu Hong; Connor O'Brien; Valmir P Moraes Filho; Jasmine R. Kobayashi; Evangelia Samara; Joseph Gallego
>
> **摘要:** Space weather at Earth, driven by the solar activity, poses growing risks to satellites around our planet as well as to critical ground-based technological infrastructure. Major space weather contributors are the solar wind and coronal mass ejections whose variable density, speed, temperature, and magnetic field make the automated classification of those structures challenging. In this work, we adapt a foundation model for solar physics, originally trained on Solar Dynamics Observatory imagery, to create embeddings suitable for solar wind structure analysis. These embeddings are concatenated with the spacecraft position and solar magnetic connectivity encoded using Fourier features which generates a neural field-based model. The full deep learning architecture is fine-tuned bridging the gap between remote sensing and in situ observations. Labels are derived from Parker Solar Probe measurements, forming a downstream classification task that maps plasma properties to solar wind structures. Although overall classification performance is modest, likely due to coarse labeling, class imbalance, and limited transferability of the pretrained model, this study demonstrates the feasibility of leveraging foundation model embeddings for in situ solar wind tasks. As a first proof-of-concept, it lays the groundwork for future improvements toward more reliable space weather predictions. The code and configuration files used in this study are publicly available to support reproducibility.
>
---
#### [new 102] Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文针对行人重识别（ReID）的对抗攻击鲁棒性问题，提出去偏双不变防御框架，通过扩散模型平衡数据与双对抗自元训练，提升对未知身份和攻击类型的泛化能力，显著超越现有方法。**

- **链接: [https://arxiv.org/pdf/2511.09933v1](https://arxiv.org/pdf/2511.09933v1)**

> **作者:** Yuhang Zhou; Yanxiang Zhao; Zhongyun Hua; Zhipu Liu; Zhaoquan Gu; Qing Liao; Leo Yu Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Person re-identification (ReID) is a fundamental task in many real-world applications such as pedestrian trajectory tracking. However, advanced deep learning-based ReID models are highly susceptible to adversarial attacks, where imperceptible perturbations to pedestrian images can cause entirely incorrect predictions, posing significant security threats. Although numerous adversarial defense strategies have been proposed for classification tasks, their extension to metric learning tasks such as person ReID remains relatively unexplored. Moreover, the several existing defenses for person ReID fail to address the inherent unique challenges of adversarially robust ReID. In this paper, we systematically identify the challenges of adversarial defense in person ReID into two key issues: model bias and composite generalization requirements. To address them, we propose a debiased dual-invariant defense framework composed of two main phases. In the data balancing phase, we mitigate model bias using a diffusion-model-based data resampling strategy that promotes fairness and diversity in training data. In the bi-adversarial self-meta defense phase, we introduce a novel metric adversarial training approach incorporating farthest negative extension softening to overcome the robustness degradation caused by the absence of classifier. Additionally, we introduce an adversarially-enhanced self-meta mechanism to achieve dual-generalization for both unseen identities and unseen attack types. Experiments demonstrate that our method significantly outperforms existing state-of-the-art defenses.
>
---
#### [new 103] Next-Frame Feature Prediction for Multimodal Deepfake Detection and Temporal Localization
- **分类: cs.CV**

- **简介: 该论文提出一种单阶段多模态深度伪造检测框架，通过预测下一帧特征并引入窗口注意力机制，提升对未知伪造手法的泛化能力，并精准定位伪造片段，解决现有方法忽视单模态伪影与泛化不足的问题。**

- **链接: [https://arxiv.org/pdf/2511.10212v1](https://arxiv.org/pdf/2511.10212v1)**

> **作者:** Ashutosh Anshul; Shreyas Gopal; Deepu Rajan; Eng Siong Chng
>
> **备注:** Under Review, Multimodal Deepfake detection
>
> **摘要:** Recent multimodal deepfake detection methods designed for generalization conjecture that single-stage supervised training struggles to generalize across unseen manipulations and datasets. However, such approaches that target generalization require pretraining over real samples. Additionally, these methods primarily focus on detecting audio-visual inconsistencies and may overlook intra-modal artifacts causing them to fail against manipulations that preserve audio-visual alignment. To address these limitations, we propose a single-stage training framework that enhances generalization by incorporating next-frame prediction for both uni-modal and cross-modal features. Additionally, we introduce a window-level attention mechanism to capture discrepancies between predicted and actual frames, enabling the model to detect local artifacts around every frame, which is crucial for accurately classifying fully manipulated videos and effectively localizing deepfake segments in partially spoofed samples. Our model, evaluated on multiple benchmark datasets, demonstrates strong generalization and precise temporal localization.
>
---
#### [new 104] MMaDA-Parallel: Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation
- **分类: cs.CV**

- **简介: 该论文针对思维感知图像生成中的文本-图像对齐失效问题，提出MMaDA-Parallel框架，通过并行扩散模型与并行强化学习实现文本与图像的双向交互，显著提升跨模态一致性，超越SOTA模型Bagel 6.9%。**

- **链接: [https://arxiv.org/pdf/2511.09611v1](https://arxiv.org/pdf/2511.09611v1)**

> **作者:** Ye Tian; Ling Yang; Jiongfan Yang; Anran Wang; Yu Tian; Jiani Zheng; Haochen Wang; Zhiyang Teng; Zhuochen Wang; Yinjie Wang; Yunhai Tong; Mengdi Wang; Xiangtai Li
>
> **备注:** Project Page: https://tyfeld.github.io/mmadaparellel.github.io/
>
> **摘要:** While thinking-aware generation aims to improve performance on complex tasks, we identify a critical failure mode where existing sequential, autoregressive approaches can paradoxically degrade performance due to error propagation. To systematically analyze this issue, we propose ParaBench, a new benchmark designed to evaluate both text and image output modalities. Our analysis using ParaBench reveals that this performance degradation is strongly correlated with poor alignment between the generated reasoning and the final image. To resolve this, we propose a parallel multimodal diffusion framework, MMaDA-Parallel, that enables continuous, bidirectional interaction between text and images throughout the entire denoising trajectory. MMaDA-Parallel is trained with supervised finetuning and then further optimized by Parallel Reinforcement Learning (ParaRL), a novel strategy that applies semantic rewards along the trajectory to enforce cross-modal consistency. Experiments validate that our model significantly improves cross-modal alignment and semantic consistency, achieving a 6.9\% improvement in Output Alignment on ParaBench compared to the state-of-the-art model, Bagel, establishing a more robust paradigm for thinking-aware image synthesis. Our code is open-sourced at https://github.com/tyfeld/MMaDA-Parallel
>
---
#### [new 105] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: SemanticVLA面向机器人操作任务，解决视觉冗余与语义对齐不足问题，提出语义对齐的稀疏化与增强框架，通过三模块协同提升效率与性能，在LIBERO基准上显著超越OpenVLA。**

- **链接: [https://arxiv.org/pdf/2511.10518v1](https://arxiv.org/pdf/2511.10518v1)**

> **作者:** Wei Li; Renshan Zhang; Rui Shao; Zhijian Fang; Kaiwen Zhou; Zhuotao Tian; Liqiang Nie
>
> **备注:** Accepted to AAAI 2026 (Oral), Project Page: https://github.com/JiuTian-VL/SemanticVLA
>
> **摘要:** Vision-Language-Action (VLA) models have advanced in robotic manipulation, yet practical deployment remains hindered by two key limitations: 1) perceptual redundancy, where irrelevant visual inputs are processed inefficiently, and 2) superficial instruction-vision alignment, which hampers semantic grounding of actions. In this paper, we propose SemanticVLA, a novel VLA framework that performs Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation. Specifically: 1) To sparsify redundant perception while preserving semantic alignment, Semantic-guided Dual Visual Pruner (SD-Pruner) performs: Instruction-driven Pruner (ID-Pruner) extracts global action cues and local semantic anchors in SigLIP; Spatial-aggregation Pruner (SA-Pruner) compacts geometry-rich features into task-adaptive tokens in DINOv2. 2) To exploit sparsified features and integrate semantics with spatial geometry, Semantic-complementary Hierarchical Fuser (SH-Fuser) fuses dense patches and sparse tokens across SigLIP and DINOv2 for coherent representation. 3) To enhance the transformation from perception to action, Semantic-conditioned Action Coupler (SA-Coupler) replaces the conventional observation-to-DoF approach, yielding more efficient and interpretable behavior modeling for manipulation tasks. Extensive experiments on simulation and real-world tasks show that SemanticVLA sets a new SOTA in both performance and efficiency. SemanticVLA surpasses OpenVLA on LIBERO benchmark by 21.1% in success rate, while reducing training cost and inference latency by 3.0-fold and 2.7-fold.SemanticVLA is open-sourced and publicly available at https://github.com/JiuTian-VL/SemanticVLA
>
---
#### [new 106] Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment
- **分类: cs.CV**

- **简介: 该论文针对弱监督视频异常检测中的类别混淆与正常模式挖掘不足问题，提出DSANet，通过粗粒度自引导正常建模与细粒度解耦对比语义对齐，分离异常与正常特征，提升检测与分类性能。**

- **链接: [https://arxiv.org/pdf/2511.10334v1](https://arxiv.org/pdf/2511.10334v1)**

> **作者:** Wenti Yin; Huaxin Zhang; Xiang Wang; Yuqing Lu; Yicheng Zhang; Bingquan Gong; Jialong Zuo; Li Yu; Changxin Gao; Nong Sang
>
> **备注:** Accepted to AAAI 2026. Code is available at https://github.com/lessiYin/DSANet
>
> **摘要:** Recent advancements in weakly-supervised video anomaly detection have achieved remarkable performance by applying the multiple instance learning paradigm based on multimodal foundation models such as CLIP to highlight anomalous instances and classify categories. However, their objectives may tend to detect the most salient response segments, while neglecting to mine diverse normal patterns separated from anomalies, and are prone to category confusion due to similar appearance, leading to unsatisfactory fine-grained classification results. Therefore, we propose a novel Disentangled Semantic Alignment Network (DSANet) to explicitly separate abnormal and normal features from coarse-grained and fine-grained aspects, enhancing the distinguishability. Specifically, at the coarse-grained level, we introduce a self-guided normality modeling branch that reconstructs input video features under the guidance of learned normal prototypes, encouraging the model to exploit normality cues inherent in the video, thereby improving the temporal separation of normal patterns and anomalous events. At the fine-grained level, we present a decoupled contrastive semantic alignment mechanism, which first temporally decomposes each video into event-centric and background-centric components using frame-level anomaly scores and then applies visual-language contrastive learning to enhance class-discriminative representations. Comprehensive experiments on two standard benchmarks, namely XD-Violence and UCF-Crime, demonstrate that DSANet outperforms existing state-of-the-art methods.
>
---
#### [new 107] Learnable Total Variation with Lambda Mapping for Low-Dose CT Denoising
- **分类: cs.CV**

- **简介: 该论文针对低剂量CT去噪任务，提出可学习总变分（LTV）框架，通过数据驱动的LambdaNet动态生成像素级正则化图，实现空间自适应平滑，突破传统TV对固定参数的依赖，提升去噪性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.10500v1](https://arxiv.org/pdf/2511.10500v1)**

> **作者:** Yusuf Talha Basak; Mehmet Ozan Unal; Metin Ertas; Isa Yildirim
>
> **摘要:** Although Total Variation (TV) performs well in noise reduction and edge preservation on images, its dependence on the lambda parameter limits its efficiency and makes it difficult to use effectively. In this study, we present a Learnable Total Variation (LTV) framework that couples an unrolled TV solver with a data-driven Lambda Mapping Network (LambdaNet) predicting a per-pixel regularization map. The pipeline is trained end-to-end so that reconstruction and regularization are optimized jointly, yielding spatially adaptive smoothing: strong in homogeneous regions, relaxed near anatomical boundaries. Experiments on the DeepLesion dataset, using a realistic noise model adapted from the LoDoPaB-CT methodology, show consistent gains over classical TV and FBP+U-Net: +2.9 dB PSNR and +6% SSIM on average. LTV provides an interpretable alternative to black-box CNNs and a basis for 3D and data-consistency-driven reconstruction. Our codes are available at: https://github.com/itu-biai/deep_tv_for_ldct
>
---
#### [new 108] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis
- **分类: cs.CV**

- **简介: 论文提出Facial-R1，面向面部情绪分析（FEA）任务，解决VLMs推理幻觉与推理-识别脱节问题，通过三阶段对齐框架（指令微调、强化学习、数据合成）实现可解释、高精度情绪分析，并构建FEA-20K基准。**

- **链接: [https://arxiv.org/pdf/2511.10254v1](https://arxiv.org/pdf/2511.10254v1)**

> **作者:** Jiulong Wu; Yucheng Shen; Lingyong Yan; Haixin Sun; Deguo Xia; Jizhou Huang; Min Cao
>
> **备注:** This paper has been accepted by AAAI 2026. 16 pages, 3 figures, 10 tables
>
> **摘要:** Facial Emotion Analysis (FEA) extends traditional facial emotion recognition by incorporating explainable, fine-grained reasoning. The task integrates three subtasks: emotion recognition, facial Action Unit (AU) recognition, and AU-based emotion reasoning to model affective states jointly. While recent approaches leverage Vision-Language Models (VLMs) and achieve promising results, they face two critical limitations: (1) hallucinated reasoning, where VLMs generate plausible but inaccurate explanations due to insufficient emotion-specific knowledge; and (2) misalignment between emotion reasoning and recognition, caused by fragmented connections between observed facial features and final labels. We propose Facial-R1, a three-stage alignment framework that effectively addresses both challenges with minimal supervision. First, we employ instruction fine-tuning to establish basic emotional reasoning capability. Second, we introduce reinforcement training guided by emotion and AU labels as reward signals, which explicitly aligns the generated reasoning process with the predicted emotion. Third, we design a data synthesis pipeline that iteratively leverages the prior stages to expand the training dataset, enabling scalable self-improvement of the model. Built upon this framework, we introduce FEA-20K, a benchmark dataset comprising 17,737 training and 1,688 test samples with fine-grained emotion analysis annotations. Extensive experiments across eight standard benchmarks demonstrate that Facial-R1 achieves state-of-the-art performance in FEA, with strong generalization and robust interpretability.
>
---
#### [new 109] RWKV-PCSSC: Exploring RWKV Model for Point Cloud Semantic Scene Completion
- **分类: cs.CV**

- **简介: 论文提出RWKV-PCSSC，用于点云语义场景补全任务，解决传统方法参数多、计算重的问题。通过引入RWKV-SG和RWKV-PD模块，实现轻量化建模，在保持SOTA性能的同时大幅降低参数量与内存开销。**

- **链接: [https://arxiv.org/pdf/2511.09878v1](https://arxiv.org/pdf/2511.09878v1)**

> **作者:** Wenzhe He; Xiaojun Chen; Wentang Chen; Hongyu Wang; Ying Liu; Ruihui Li
>
> **备注:** 13 pages, 8 figures, published to ACM MM
>
> **摘要:** Semantic Scene Completion (SSC) aims to generate a complete semantic scene from an incomplete input. Existing approaches often employ dense network architectures with a high parameter count, leading to increased model complexity and resource demands. To address these limitations, we propose RWKV-PCSSC, a lightweight point cloud semantic scene completion network inspired by the Receptance Weighted Key Value (RWKV) mechanism. Specifically, we introduce a RWKV Seed Generator (RWKV-SG) module that can aggregate features from a partial point cloud to produce a coarse point cloud with coarse features. Subsequently, the point-wise feature of the point cloud is progressively restored through multiple stages of the RWKV Point Deconvolution (RWKV-PD) modules. By leveraging a compact and efficient design, our method achieves a lightweight model representation. Experimental results demonstrate that RWKV-PCSSC reduces the parameter count by 4.18$\times$ and improves memory efficiency by 1.37$\times$ compared to state-of-the-art methods PointSSC. Furthermore, our network achieves state-of-the-art performance on established indoor (SSC-PC, NYUCAD-PC) and outdoor (PointSSC) scene dataset, as well as on our proposed datasets (NYUCAD-PC-V2, 3D-FRONT-PC).
>
---
#### [new 110] EgoEMS: A High-Fidelity Multimodal Egocentric Dataset for Cognitive Assistance in Emergency Medical Services
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 论文提出EgoEMS，首个面向急救医疗的高保真多模态自拍数据集，用于支持AI认知辅助系统开发，解决急救人员认知负荷过重问题，包含20+小时真实场景数据与多维度标注，并构建了实时关键步骤识别与动作质量评估基准。**

- **链接: [https://arxiv.org/pdf/2511.09894v1](https://arxiv.org/pdf/2511.09894v1)**

> **作者:** Keshara Weerasinghe; Xueren Ge; Tessa Heick; Lahiru Nuwan Wijayasingha; Anthony Cortez; Abhishek Satpathy; John Stankovic; Homa Alemzadeh
>
> **备注:** Accepted to AAAI 2026 (Preprint), 45 pages, 29 figures
>
> **摘要:** Emergency Medical Services (EMS) are critical to patient survival in emergencies, but first responders often face intense cognitive demands in high-stakes situations. AI cognitive assistants, acting as virtual partners, have the potential to ease this burden by supporting real-time data collection and decision making. In pursuit of this vision, we introduce EgoEMS, the first end-to-end, high-fidelity, multimodal, multiperson dataset capturing over 20 hours of realistic, procedural EMS activities from an egocentric view in 233 simulated emergency scenarios performed by 62 participants, including 46 EMS professionals. Developed in collaboration with EMS experts and aligned with national standards, EgoEMS is captured using an open-source, low-cost, and replicable data collection system and is annotated with keysteps, timestamped audio transcripts with speaker diarization, action quality metrics, and bounding boxes with segmentation masks. Emphasizing realism, the dataset includes responder-patient interactions reflecting real-world emergency dynamics. We also present a suite of benchmarks for real-time multimodal keystep recognition and action quality estimation, essential for developing AI support tools for EMS. We hope EgoEMS inspires the research community to push the boundaries of intelligent EMS systems and ultimately contribute to improved patient outcomes.
>
---
#### [new 111] Impact of Layer Norm on Memorization and Generalization in Transformers
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究LayerNorm在Transformer中对记忆与泛化的影响，对比Pre-和Post-LayerNorm架构，发现其在前者稳定学习、后者抑制记忆，且早期层作用最关键，通过13模型6数据集验证。**

- **链接: [https://arxiv.org/pdf/2511.10566v1](https://arxiv.org/pdf/2511.10566v1)**

> **作者:** Rishi Singhal; Jung-Eun Kim
>
> **备注:** NeurIPS 2025
>
> **摘要:** Layer Normalization (LayerNorm) is one of the fundamental components in transformers that stabilizes training and improves optimization. In recent times, Pre-LayerNorm transformers have become the preferred choice over Post-LayerNorm transformers due to their stable gradient flow. However, the impact of LayerNorm on learning and memorization across these architectures remains unclear. In this work, we investigate how LayerNorm influences memorization and learning for Pre- and Post-LayerNorm transformers. We identify that LayerNorm serves as a key factor for stable learning in Pre-LayerNorm transformers, while in Post-LayerNorm transformers, it impacts memorization. Our analysis reveals that eliminating LayerNorm parameters in Pre-LayerNorm models exacerbates memorization and destabilizes learning, while in Post-LayerNorm models, it effectively mitigates memorization by restoring genuine labels. We further precisely identify that early layers LayerNorm are the most critical over middle/later layers and their influence varies across Pre and Post LayerNorm models. We have validated it through 13 models across 6 Vision and Language datasets. These insights shed new light on the role of LayerNorm in shaping memorization and learning in transformers.
>
---
#### [new 112] eXIAA: eXplainable Injections for Adversarial Attack
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出eXIAA，一种黑盒、模型无关的对抗攻击方法，旨在在不改变预测结果的前提下，隐蔽篡改XAI生成的解释图（如显著图），暴露现有可解释性方法的脆弱性。**

- **链接: [https://arxiv.org/pdf/2511.10088v1](https://arxiv.org/pdf/2511.10088v1)**

> **作者:** Leonardo Pesce; Jiawen Wei; Gianmarco Mengaldo
>
> **摘要:** Post-hoc explainability methods are a subset of Machine Learning (ML) that aim to provide a reason for why a model behaves in a certain way. In this paper, we show a new black-box model-agnostic adversarial attack for post-hoc explainable Artificial Intelligence (XAI), particularly in the image domain. The goal of the attack is to modify the original explanations while being undetected by the human eye and maintain the same predicted class. In contrast to previous methods, we do not require any access to the model or its weights, but only to the model's computed predictions and explanations. Additionally, the attack is accomplished in a single step while significantly changing the provided explanations, as demonstrated by empirical evaluation. The low requirements of our method expose a critical vulnerability in current explainability methods, raising concerns about their reliability in safety-critical applications. We systematically generate attacks based on the explanations generated by post-hoc explainability methods (saliency maps, integrated gradients, and DeepLIFT SHAP) for pretrained ResNet-18 and ViT-B16 on ImageNet. The results show that our attacks could lead to dramatically different explanations without changing the predictive probabilities. We validate the effectiveness of our attack, compute the induced change based on the explanation with mean absolute difference, and verify the closeness of the original image and the corrupted one with the Structural Similarity Index Measure (SSIM).
>
---
#### [new 113] Trapped by Their Own Light: Deployable and Stealth Retroreflective Patch Attacks on Traffic Sign Recognition Systems
- **分类: cs.CR; cs.CV**

- **简介: 该论文针对自动驾驶交通标志识别系统的安全漏洞，提出一种隐蔽的逆反射贴片攻击（ARP），利用车灯激活材料实现高成功率与高隐蔽性，并设计极化滤镜防御方案DPR Shield，有效抵御此类攻击。**

- **链接: [https://arxiv.org/pdf/2511.10050v1](https://arxiv.org/pdf/2511.10050v1)**

> **作者:** Go Tsuruoka; Takami Sato; Qi Alfred Chen; Kazuki Nomoto; Ryunosuke Kobayashi; Yuna Tanaka; Tatsuya Mori
>
> **摘要:** Traffic sign recognition plays a critical role in ensuring safe and efficient transportation of autonomous vehicles but remain vulnerable to adversarial attacks using stickers or laser projections. While existing attack vectors demonstrate security concerns, they suffer from visual detectability or implementation constraints, suggesting unexplored vulnerability surfaces in TSR systems. We introduce the Adversarial Retroreflective Patch (ARP), a novel attack vector that combines the high deployability of patch attacks with the stealthiness of laser projections by utilizing retroreflective materials activated only under victim headlight illumination. We develop a retroreflection simulation method and employ black-box optimization to maximize attack effectiveness. ARP achieves $\geq$93.4\% success rate in dynamic scenarios at 35 meters and $\geq$60\% success rate against commercial TSR systems in real-world conditions. Our user study demonstrates that ARP attacks maintain near-identical stealthiness to benign signs while achieving $\geq$1.9\% higher stealthiness scores than previous patch attacks. We propose the DPR Shield defense, employing strategically placed polarized filters, which achieves $\geq$75\% defense success rates for stop signs and speed limit signs against micro-prism patches.
>
---
#### [new 114] PRISM: Diversifying Dataset Distillation by Decoupling Architectural Priors
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: PRISM提出一种多教师解耦的_dataset_蒸馏框架，通过分离logit匹配与BN对齐任务，缓解单教师偏差导致的样本同质化问题，显著提升低样本量下数据多样性与泛化性能。**

- **链接: [https://arxiv.org/pdf/2511.09905v1](https://arxiv.org/pdf/2511.09905v1)**

> **作者:** Brian B. Moser; Shalini Strode; Federico Raue; Stanislav Frolov; Krzysztof Adamkiewicz; Arundhati Shanbhag; Joachim Folk; Tobias C. Nauen; Andreas Dengel
>
> **摘要:** Dataset distillation (DD) promises compact yet faithful synthetic data, but existing approaches often inherit the inductive bias of a single teacher model. As dataset size increases, this bias drives generation toward overly smooth, homogeneous samples, reducing intra-class diversity and limiting generalization. We present PRISM (PRIors from diverse Source Models), a framework that disentangles architectural priors during synthesis. PRISM decouples the logit-matching and regularization objectives, supervising them with different teacher architectures: a primary model for logits and a stochastic subset for batch-normalization (BN) alignment. On ImageNet-1K, PRISM consistently and reproducibly outperforms single-teacher methods (e.g., SRe2L) and recent multi-teacher variants (e.g., G-VBSM) at low- and mid-IPC regimes. The generated data also show significantly richer intra-class diversity, as reflected by a notable drop in cosine similarity between features. We further analyze teacher selection strategies (pre- vs. intra-distillation) and introduce a scalable cross-class batch formation scheme for fast parallel synthesis. Code will be released after the review period.
>
---
#### [new 115] Learning to Pose Problems: Reasoning-Driven and Solver-Adaptive Data Synthesis for Large Reasoning Models
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出一种推理驱动的合成数据方法，通过动态适配求解器能力生成高质量问题，解决现有方法生成低价值或浅层问题的缺陷，实现生成器与求解器的协同进化，提升大推理模型性能。**

- **链接: [https://arxiv.org/pdf/2511.09907v1](https://arxiv.org/pdf/2511.09907v1)**

> **作者:** Yongxian Wei; Yilin Zhao; Li Shen; Xinrui Chen; Runxi Cheng; Sinan Du; Hao Yu; Gang Liu; Jiahong Yan; Chun Yuan; Dian Li
>
> **摘要:** Data synthesis for training large reasoning models offers a scalable alternative to limited, human-curated datasets, enabling the creation of high-quality data. However, existing approaches face several challenges: (i) indiscriminate generation that ignores the solver's ability and yields low-value problems, or reliance on complex data pipelines to balance problem difficulty; and (ii) a lack of reasoning in problem generation, leading to shallow problem variants. In this paper, we develop a problem generator that reasons explicitly to plan problem directions before synthesis and adapts difficulty to the solver's ability. Specifically, we construct related problem pairs and augment them with intermediate problem-design CoT produced by a reasoning model. These data bootstrap problem-design strategies from the generator. Then, we treat the solver's feedback on synthetic problems as a reward signal, enabling the generator to calibrate difficulty and produce complementary problems near the edge of the solver's competence. Extensive experiments on 10 mathematical and general reasoning benchmarks show that our method achieves an average improvement of 2.5% and generalizes to both language and vision-language models. Moreover, a solver trained on the synthesized data provides improved rewards for continued generator training, enabling co-evolution and yielding a further 0.7% performance gain. Our code will be made publicly available here.
>
---
#### [new 116] How does My Model Fail? Automatic Identification and Interpretation of Physical Plausibility Failure Modes with Matryoshka Transcoders
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出Matryoshka Transcoders框架，自动识别并解释生成模型中的物理合理性失效模式，解决现有评估方法无法检测物理错误的问题，通过多粒度特征学习与多模态解释，实现无手工特征的失效模式发现与基准构建。**

- **链接: [https://arxiv.org/pdf/2511.10094v1](https://arxiv.org/pdf/2511.10094v1)**

> **作者:** Yiming Tang; Abhijeet Sinha; Dianbo Liu
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Although recent generative models are remarkably capable of producing instruction-following and realistic outputs, they remain prone to notable physical plausibility failures. Though critical in applications, these physical plausibility errors often escape detection by existing evaluation methods. Furthermore, no framework exists for automatically identifying and interpreting specific physical error patterns in natural language, preventing targeted model improvements. We introduce Matryoshka Transcoders, a novel framework for the automatic discovery and interpretation of physical plausibility features in generative models. Our approach extends the Matryoshka representation learning paradigm to transcoder architectures, enabling hierarchical sparse feature learning at multiple granularity levels. By training on intermediate representations from a physical plausibility classifier and leveraging large multimodal models for interpretation, our method identifies diverse physics-related failure modes without manual feature engineering, achieving superior feature relevance and feature accuracy compared to existing approaches. We utilize the discovered visual patterns to establish a benchmark for evaluating physical plausibility in generative models. Our analysis of eight state-of-the-art generative models provides valuable insights into how these models fail to follow physical constraints, paving the way for further model improvements.
>
---
#### [new 117] Intrinsic Dimensionality as a Model-Free Measure of Class Imbalance
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出用数据内在维度（ID）作为无模型的类别不平衡度量，替代传统基于样本数量的方法。在五类数据集上验证，ID显著优于重加权与重采样技术，且与样本数结合可进一步提升性能。**

- **链接: [https://arxiv.org/pdf/2511.10475v1](https://arxiv.org/pdf/2511.10475v1)**

> **作者:** Çağrı Eser; Zeynep Sonat Baltacı; Emre Akbaş; Sinan Kalkan
>
> **备注:** 45 pages, 11 figures
>
> **摘要:** Imbalance in classification tasks is commonly quantified by the cardinalities of examples across classes. This, however, disregards the presence of redundant examples and inherent differences in the learning difficulties of classes. Alternatively, one can use complex measures such as training loss and uncertainty, which, however, depend on training a machine learning model. Our paper proposes using data Intrinsic Dimensionality (ID) as an easy-to-compute, model-free measure of imbalance that can be seamlessly incorporated into various imbalance mitigation methods. Our results across five different datasets with a diverse range of imbalance ratios show that ID consistently outperforms cardinality-based re-weighting and re-sampling techniques used in the literature. Moreover, we show that combining ID with cardinality can further improve performance. Code: https://github.com/cagries/IDIM.
>
---
#### [new 118] Querying Labeled Time Series Data with Scenario Programs
- **分类: cs.AI; cs.CV; cs.FL; cs.LG**

- **简介: 该论文针对仿真与现实间的“sim-to-real”差距，提出基于Scenic语言的场景程序，定义并实现了一种高效查询标注时序传感器数据中匹配场景的算法，验证仿真失败场景在真实数据中的可重现性。**

- **链接: [https://arxiv.org/pdf/2511.10627v1](https://arxiv.org/pdf/2511.10627v1)**

> **作者:** Edward Kim; Devan Shanker; Varun Bharadwaj; Hongbeen Park; Jinkyu Kim; Hazem Torfah; Daniel J Fremont; Sanjit A Seshia
>
> **摘要:** Simulation-based testing has become a crucial complement to road testing for ensuring the safety of cyber physical systems (CPS). As a result, significant research efforts have been directed toward identifying failure scenarios within simulation environments. However, a critical question remains. Are the AV failure scenarios discovered in simulation reproducible on actual systems in the real world? The sim-to-real gap caused by differences between simulated and real sensor data means that failure scenarios identified in simulation might either be artifacts of synthetic sensor data or actual issues that also occur with real sensor data. To address this, an effective approach to validating simulated failure scenarios is to locate occurrences of these scenarios within real-world datasets and verify whether the failure persists on the datasets. To this end, we introduce a formal definition of how labeled time series sensor data can match an abstract scenario, represented as a scenario program using the Scenic probabilistic programming language. We present a querying algorithm that, given a scenario program and a labeled dataset, identifies the subset of data that matches the specified scenario. Our experiment shows that our algorithm is more accurate and orders of magnitude faster in querying scenarios than the state-of-the-art commercial vision large language models, and can scale with the duration of queried time series data.
>
---
#### [new 119] Efficient Automated Diagnosis of Retinopathy of Prematurity by Customize CNN Models
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对早产儿视网膜病变（ROP）诊断问题，构建定制化CNN模型，提升检测精度与效率，降低计算成本，并通过投票系统优化性能，验证了其在临床部署的可行性。**

- **链接: [https://arxiv.org/pdf/2511.10023v1](https://arxiv.org/pdf/2511.10023v1)**

> **作者:** Farzan Saeedi; Sanaz Keshvari; Nasser Shoeibi
>
> **摘要:** This paper encompasses an in-depth examination of Retinopathy of Prematurity (ROP) diagnosis, employing advanced deep learning methodologies. Our focus centers on refining and evaluating CNN-based approaches for precise and efficient ROP detection. We navigate the complexities of dataset curation, preprocessing strategies, and model architecture, aligning with research objectives encompassing model effectiveness, computational cost analysis, and time complexity assessment. Results underscore the supremacy of tailored CNN models over pre-trained counterparts, evident in heightened accuracy and F1-scores. Implementation of a voting system further enhances performance. Additionally, our study reveals the potential of the proposed customized CNN model to alleviate computational burdens associated with deep neural networks. Furthermore, we showcase the feasibility of deploying these models within dedicated software and hardware configurations, highlighting their utility as valuable diagnostic aids in clinical settings. In summary, our discourse significantly contributes to ROP diagnosis, unveiling the efficacy of deep learning models in enhancing diagnostic precision and efficiency.
>
---
#### [new 120] VEDA: 3D Molecular Generation via Variance-Exploding Diffusion with Annealing
- **分类: physics.chem-ph; cs.AI; cs.CV**

- **简介: VEDA提出一种基于方差爆炸扩散与退火机制的SE(3)等变框架，用于高效生成高精度3D分子构象，解决传统扩散模型采样慢、流模型结构不准的问题，在100步内实现媲美流模型的速度与卓越几何稳定性。**

- **链接: [https://arxiv.org/pdf/2511.09568v1](https://arxiv.org/pdf/2511.09568v1)**

> **作者:** Peining Zhang; Jinbo Bi; Minghu Song
>
> **摘要:** Diffusion models show promise for 3D molecular generation, but face a fundamental trade-off between sampling efficiency and conformational accuracy. While flow-based models are fast, they often produce geometrically inaccurate structures, as they have difficulty capturing the multimodal distributions of molecular conformations. In contrast, denoising diffusion models are more accurate but suffer from slow sampling, a limitation attributed to sub-optimal integration between diffusion dynamics and SE(3)-equivariant architectures. To address this, we propose VEDA, a unified SE(3)-equivariant framework that combines variance-exploding diffusion with annealing to efficiently generate conformationally accurate 3D molecular structures. Specifically, our key technical contributions include: (1) a VE schedule that enables noise injection functionally analogous to simulated annealing, improving 3D accuracy and reducing relaxation energy; (2) a novel preconditioning scheme that reconciles the coordinate-predicting nature of SE(3)-equivariant networks with a residual-based diffusion objective, and (3) a new arcsin-based scheduler that concentrates sampling in critical intervals of the logarithmic signal-to-noise ratio. On the QM9 and GEOM-DRUGS datasets, VEDA matches the sampling efficiency of flow-based models, achieving state-of-the-art valency stability and validity with only 100 sampling steps. More importantly, VEDA's generated structures are remarkably stable, as measured by their relaxation energy during GFN2-xTB optimization. The median energy change is only 1.72 kcal/mol, significantly lower than the 32.3 kcal/mol from its architectural baseline, SemlaFlow. Our framework demonstrates that principled integration of VE diffusion with SE(3)-equivariant architectures can achieve both high chemical accuracy and computational efficiency.
>
---
## 更新

#### [replaced 001] Generating Physically Stable and Buildable Brick Structures from Text
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.05469v3](https://arxiv.org/pdf/2505.05469v3)**

> **作者:** Ava Pun; Kangle Deng; Ruixuan Liu; Deva Ramanan; Changliu Liu; Jun-Yan Zhu
>
> **备注:** Project page: https://avalovelace1.github.io/BrickGPT/
>
> **摘要:** We introduce BrickGPT, the first approach for generating physically stable interconnecting brick assembly models from text prompts. To achieve this, we construct a large-scale, physically stable dataset of brick structures, along with their associated captions, and train an autoregressive large language model to predict the next brick to add via next-token prediction. To improve the stability of the resulting designs, we employ an efficient validity check and physics-aware rollback during autoregressive inference, which prunes infeasible token predictions using physics laws and assembly constraints. Our experiments show that BrickGPT produces stable, diverse, and aesthetically pleasing brick structures that align closely with the input text prompts. We also develop a text-based brick texturing method to generate colored and textured designs. We show that our designs can be assembled manually by humans and automatically by robotic arms. We release our new dataset, StableText2Brick, containing over 47,000 brick structures of over 28,000 unique 3D objects accompanied by detailed captions, along with our code and models at the project website: https://avalovelace1.github.io/BrickGPT/.
>
---
#### [replaced 002] Attri-Net: A Globally and Locally Inherently Interpretable Model for Multi-Label Classification Using Class-Specific Counterfactuals
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2406.05477v2](https://arxiv.org/pdf/2406.05477v2)**

> **作者:** Susu Sun; Stefano Woerner; Andreas Maier; Lisa M. Koch; Christian F. Baumgartner
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) https://melba-journal.org/2025:028
>
> **摘要:** Interpretability is crucial for machine learning algorithms in high-stakes medical applications. However, high-performing neural networks typically cannot explain their predictions. Post-hoc explanation methods provide a way to understand neural networks but have been shown to suffer from conceptual problems. Moreover, current research largely focuses on providing local explanations for individual samples rather than global explanations for the model itself. In this paper, we propose Attri-Net, an inherently interpretable model for multi-label classification that provides local and global explanations. Attri-Net first counterfactually generates class-specific attribution maps to highlight the disease evidence, then performs classification with logistic regression classifiers based solely on the attribution maps. Local explanations for each prediction can be obtained by interpreting the attribution maps weighted by the classifiers' weights. Global explanation of whole model can be obtained by jointly considering learned average representations of the attribution maps for each class (called the class centers) and the weights of the linear classifiers. To ensure the model is ``right for the right reason", we further introduce a mechanism to guide the model's explanations to align with human knowledge. Our comprehensive evaluations show that Attri-Net can generate high-quality explanations consistent with clinical knowledge while not sacrificing classification performance.
>
---
#### [replaced 003] A Bayesian Approach to Segmentation with Noisy Labels via Spatially Correlated Distributions
- **分类: eess.IV; cs.CV; cs.LG; stat.ML**

- **链接: [https://arxiv.org/pdf/2504.14795v3](https://arxiv.org/pdf/2504.14795v3)**

> **作者:** Ryu Tadokoro; Tsukasa Takagi; Shin-ichi Maeda
>
> **摘要:** In semantic segmentation, the accuracy of models heavily depends on the high-quality annotations. However, in many practical scenarios, such as medical imaging and remote sensing, obtaining true annotations is not straightforward and usually requires significant human labor. Relying on human labor often introduces annotation errors, including mislabeling, omissions, and inconsistency between annotators. In the case of remote sensing, differences in procurement time can lead to misaligned ground-truth annotations. These label errors are not independently distributed, and instead usually appear in spatially connected regions where adjacent pixels are more likely to share the same errors. To address these issues, we propose an approximate Bayesian estimation based on a probabilistic model that assumes training data include label errors, incorporating the tendency for these errors to occur with spatial correlations between adjacent pixels. However, Bayesian inference for such spatially correlated discrete variables is notoriously intractable. To overcome this fundamental challenge, we introduce a novel class of probabilistic models, which we term the ELBO-Computable Correlated Discrete Distribution (ECCD). By representing the discrete dependencies through a continuous latent Gaussian field with a Kac-Murdock-Szegö (KMS) structured covariance, our framework enables scalable and efficient variational inference for problems previously considered computationally prohibitive. Through experiments on multiple segmentation tasks, we confirm that leveraging the spatial correlation of label errors significantly improves performance. Notably, in specific tasks such as lung segmentation, the proposed method achieves performance comparable to training with clean labels under moderate noise levels. Code is available at https://github.com/pfnet-research/Bayesian_SpatialCorr.
>
---
#### [replaced 004] Dual-Mode Deep Anomaly Detection for Medical Manufacturing: Structural Similarity and Feature Distance
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.05796v3](https://arxiv.org/pdf/2509.05796v3)**

> **作者:** Julio Zanon Diaz; Georgios Siogkas; Peter Corcoran
>
> **备注:** 12 pages, 3 figures, 3 tables
>
> **摘要:** Automated visual inspection in medical-device manufacturing faces unique challenges, including extremely low defect rates, limited annotated data, hardware restrictions on production lines, and the need for validated, explainable artificial-intelligence systems. This paper presents two attention-guided autoencoder architectures that address these constraints through complementary anomaly-detection strategies. The first employs a multi-scale structural-similarity (4-MS-SSIM) index for inline inspection, enabling interpretable, real-time defect detection on constrained hardware. The second applies a Mahalanobis-distance analysis of randomly reduced latent features for efficient feature-space monitoring and lifecycle verification. Both approaches share a lightweight backbone optimised for high-resolution imagery for typical manufacturing conditions. Evaluations on the Surface Seal Image (SSI) dataset-representing sterile-barrier packaging inspection-demonstrate that the proposed methods outperform reference baselines, including MOCCA, CPCAE, and RAG-PaDiM, under realistic industrial constraints. Cross-domain validation on the MVTec-Zipper benchmark confirms comparable accuracy to state-of-the-art anomaly-detection methods. The dual-mode framework integrates inline anomaly detection and supervisory monitoring, advancing explainable AI architectures toward greater reliability, observability, and lifecycle monitoring in safety-critical manufacturing environments. To facilitate reproducibility, the source code developed for the experiments has been released in the project repository, while the datasets were obtained from publicly available sources.
>
---
#### [replaced 005] MVU-Eval: Towards Multi-Video Understanding Evaluation for Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.07250v2](https://arxiv.org/pdf/2511.07250v2)**

> **作者:** Tianhao Peng; Haochen Wang; Yuanxing Zhang; Zekun Wang; Zili Wang; Gavin Chang; Jian Yang; Shihao Li; Yanghai Wang; Xintao Wang; Houyi Li; Wei Ji; Pengfei Wan; Steven Huang; Zhaoxiang Zhang; Jiaheng Liu
>
> **摘要:** The advent of Multimodal Large Language Models (MLLMs) has expanded AI capabilities to visual modalities, yet existing evaluation benchmarks remain limited to single-video understanding, overlooking the critical need for multi-video understanding in real-world scenarios (e.g., sports analytics and autonomous driving). To address this significant gap, we introduce MVU-Eval, the first comprehensive benchmark for evaluating Multi-Video Understanding for MLLMs. Specifically, our MVU-Eval mainly assesses eight core competencies through 1,824 meticulously curated question-answer pairs spanning 4,959 videos from diverse domains, addressing both fundamental perception tasks and high-order reasoning tasks. These capabilities are rigorously aligned with real-world applications such as multi-sensor synthesis in autonomous systems and cross-angle sports analytics. Through extensive evaluation of state-of-the-art open-source and closed-source models, we reveal significant performance discrepancies and limitations in current MLLMs' ability to perform understanding across multiple videos. The benchmark will be made publicly available to foster future research.
>
---
#### [replaced 006] ForAug: Recombining Foregrounds and Backgrounds to Improve Vision Transformer Training with Bias Mitigation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.09399v2](https://arxiv.org/pdf/2503.09399v2)**

> **作者:** Tobias Christian Nauen; Brian Moser; Federico Raue; Stanislav Frolov; Andreas Dengel
>
> **备注:** v2: added DeiT, added ablation vs simple copy-paste
>
> **摘要:** Transformers, particularly Vision Transformers (ViTs), have achieved state-of-the-art performance in large-scale image classification. However, they often require large amounts of data and can exhibit biases that limit their robustness and generalizability. This paper introduces ForAug, a novel data augmentation scheme that addresses these challenges and explicitly includes inductive biases, which commonly are part of the neural network architecture, into the training data. ForAug is constructed by using pretrained foundation models to separate and recombine foreground objects with different backgrounds, enabling fine-grained control over image composition during training. It thus increases the data diversity and effective number of training samples. We demonstrate that training on ForNet, the application of ForAug to ImageNet, significantly improves the accuracy of ViTs and other architectures by up to 4.5 percentage points (p.p.) on ImageNet and 7.3 p.p. on downstream tasks. Importantly, ForAug enables novel ways of analyzing model behavior and quantifying biases. Namely, we introduce metrics for background robustness, foreground focus, center bias, and size bias and show that training on ForNet substantially reduces these biases compared to training on ImageNet. In summary, ForAug provides a valuable tool for analyzing and mitigating biases, enabling the development of more robust and reliable computer vision models. Our code and dataset are publicly available at https://github.com/tobna/ForAug.
>
---
#### [replaced 007] Latent Knowledge-Guided Video Diffusion for Scientific Phenomena Generation from a Single Initial Frame
- **分类: cs.CV; stat.AP**

- **链接: [https://arxiv.org/pdf/2411.11343v2](https://arxiv.org/pdf/2411.11343v2)**

> **作者:** Qinglong Cao; Xirui Li; Ding Wang; Chao Ma; Yuntian Chen; Xiaokang Yang
>
> **摘要:** Video diffusion models have achieved impressive results in natural scene generation, yet they struggle to generalize to scientific phenomena such as fluid simulations and meteorological processes, where underlying dynamics are governed by scientific laws. These tasks pose unique challenges, including severe domain gaps, limited training data, and the lack of descriptive language annotations. To handle this dilemma, we extracted the latent scientific phenomena knowledge and further proposed a fresh framework that teaches video diffusion models to generate scientific phenomena from a single initial frame. Particularly, static knowledge is extracted via pre-trained masked autoencoders, while dynamic knowledge is derived from pre-trained optical flow prediction. Subsequently, based on the aligned spatial relations between the CLIP vision and language encoders, the visual embeddings of scientific phenomena, guided by latent scientific phenomena knowledge, are projected to generate the pseudo-language prompt embeddings in both spatial and frequency domains. By incorporating these prompts and fine-tuning the video diffusion model, we enable the generation of videos that better adhere to scientific laws. Extensive experiments on both computational fluid dynamics simulations and real-world typhoon observations demonstrate the effectiveness of our approach, achieving superior fidelity and consistency across diverse scientific scenarios.
>
---
#### [replaced 008] VasoMIM: Vascular Anatomy-Aware Masked Image Modeling for Vessel Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10794v2](https://arxiv.org/pdf/2508.10794v2)**

> **作者:** De-Xing Huang; Xiao-Hu Zhou; Mei-Jiang Gui; Xiao-Liang Xie; Shi-Qi Liu; Shuang-Yi Wang; Tian-Yu Xiang; Rui-Ze Ma; Nu-Fang Xiao; Zeng-Guang Hou
>
> **备注:** Accepted by the Annual AAAI Conference on Artificial Intelligence (AAAI). Extended version
>
> **摘要:** Accurate vessel segmentation in X-ray angiograms is crucial for numerous clinical applications. However, the scarcity of annotated data presents a significant challenge, which has driven the adoption of self-supervised learning (SSL) methods such as masked image modeling (MIM) to leverage large-scale unlabeled data for learning transferable representations. Unfortunately, conventional MIM often fails to capture vascular anatomy because of the severe class imbalance between vessel and background pixels, leading to weak vascular representations. To address this, we introduce Vascular anatomy-aware Masked Image Modeling (VasoMIM), a novel MIM framework tailored for X-ray angiograms that explicitly integrates anatomical knowledge into the pre-training process. Specifically, it comprises two complementary components: anatomy-guided masking strategy and anatomical consistency loss. The former preferentially masks vessel-containing patches to focus the model on reconstructing vessel-relevant regions. The latter enforces consistency in vascular semantics between the original and reconstructed images, thereby improving the discriminability of vascular representations. Empirically, VasoMIM achieves state-of-the-art performance across three datasets. These findings highlight its potential to facilitate X-ray angiogram analysis.
>
---
#### [replaced 009] WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.26125v3](https://arxiv.org/pdf/2510.26125v3)**

> **作者:** Runsheng Xu; Hubert Lin; Wonseok Jeon; Hao Feng; Yuliang Zou; Liting Sun; John Gorman; Ekaterina Tolstaya; Sarah Tang; Brandyn White; Ben Sapp; Mingxing Tan; Jyh-Jing Hwang; Dragomir Anguelov
>
> **摘要:** Vision-based end-to-end (E2E) driving has garnered significant interest in the research community due to its scalability and synergy with multimodal large language models (MLLMs). However, current E2E driving benchmarks primarily feature nominal scenarios, failing to adequately test the true potential of these systems. Furthermore, existing open-loop evaluation metrics often fall short in capturing the multi-modal nature of driving or effectively evaluating performance in long-tail scenarios. To address these gaps, we introduce the Waymo Open Dataset for End-to-End Driving (WOD-E2E). WOD-E2E contains 4,021 driving segments (approximately 12 hours), specifically curated for challenging long-tail scenarios that that are rare in daily life with an occurring frequency of less than 0.03%. Concretely, each segment in WOD-E2E includes the high-level routing information, ego states, and 360-degree camera views from 8 surrounding cameras. To evaluate the E2E driving performance on these long-tail situations, we propose a novel open-loop evaluation metric: Rater Feedback Score (RFS). Unlike conventional metrics that measure the distance between predicted way points and the logs, RFS measures how closely the predicted trajectory matches rater-annotated trajectory preference labels. We have released rater preference labels for all WOD-E2E validation set segments, while the held out test set labels have been used for the 2025 WOD-E2E Challenge. Through our work, we aim to foster state of the art research into generalizable, robust, and safe end-to-end autonomous driving agents capable of handling complex real-world situations.
>
---
#### [replaced 010] Drifting Away from Truth: GenAI-Driven News Diversity Challenges LVLM-Based Misinformation Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12711v2](https://arxiv.org/pdf/2508.12711v2)**

> **作者:** Fanxiao Li; Jiaying Wu; Tingchao Fu; Yunyun Dong; Bingbing Song; Wei Zhou
>
> **摘要:** The proliferation of multimodal misinformation poses growing threats to public discourse and societal trust. While Large Vision-Language Models (LVLMs) have enabled recent progress in multimodal misinformation detection (MMD), the rise of generative AI (GenAI) tools introduces a new challenge: GenAI-driven news diversity, characterized by highly varied and complex content. We show that this diversity induces multi-level drift, comprising (1) model-level misperception drift, where stylistic variations disrupt a model's internal reasoning, and (2) evidence-level drift, where expression diversity degrades the quality or relevance of retrieved external evidence. These drifts significantly degrade the robustness of current LVLM-based MMD systems. To systematically study this problem, we introduce DriftBench, a large-scale benchmark comprising 16,000 news instances across six categories of diversification. We design three evaluation tasks: (1) robustness of truth verification under multi-level drift; (2) susceptibility to adversarial evidence contamination generated by GenAI; and (3) analysis of reasoning consistency across diverse inputs. Experiments with six state-of-the-art LVLM-based detectors show substantial performance drops (average F1 -14.8%) and increasingly unstable reasoning traces, with even more severe failures under adversarial evidence injection. Our findings uncover fundamental vulnerabilities in existing MMD systems and suggest an urgent need for more resilient approaches in the GenAI era.
>
---
#### [replaced 011] MCM: Multi-layer Concept Map for Efficient Concept Learning from Masked Images
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2502.00266v2](https://arxiv.org/pdf/2502.00266v2)**

> **作者:** Yuwei Sun; Lu Mi; Ippei Fujisawa; Ruiqiao Mei; Jimin Chen; Siyu Zhu; Ryota Kanai
>
> **摘要:** Masking strategies commonly employed in natural language processing are still underexplored in vision tasks such as concept learning, where conventional methods typically rely on full images. However, using masked images diversifies perceptual inputs, potentially offering significant advantages in concept learning with large-scale Transformer models. To this end, we propose Multi-layer Concept Map (MCM), the first work to devise an efficient concept learning method based on masked images. In particular, we introduce an asymmetric concept learning architecture by establishing correlations between different encoder and decoder layers, updating concept tokens using backward gradients from reconstruction tasks. The learned concept tokens at various levels of granularity help either reconstruct the masked image patches by filling in gaps or guide the reconstruction results in a direction that reflects specific concepts. Moreover, we present both quantitative and qualitative results across a wide range of metrics, demonstrating that MCM significantly reduces computational costs by training on fewer than 75% of the total image patches while enhancing concept prediction performance. Additionally, editing specific concept tokens in the latent space enables targeted image generation from masked images, aligning both the visible contextual patches and the provided concepts. By further adjusting the testing time mask ratio, we could produce a range of reconstructions that blend the visible patches with the provided concepts, proportional to the chosen ratios.
>
---
#### [replaced 012] Feature-EndoGaussian: Feature Distilled Gaussian Splatting in Surgical Deformable Scene Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.06161v2](https://arxiv.org/pdf/2503.06161v2)**

> **作者:** Kai Li; Junhao Wang; William Han; Ding Zhao
>
> **备注:** 17 pages, 5 figures; Accepted to ML4H 2025
>
> **摘要:** Minimally invasive surgery (MIS) requires high-fidelity, real-time visual feedback of dynamic and low-texture surgical scenes. To address these requirements, we introduce FeatureEndo-4DGS (FE-4DGS), the first real time pipeline leveraging feature-distilled 4D Gaussian Splatting for simultaneous reconstruction and semantic segmentation of deformable surgical environments. Unlike prior feature-distilled methods restricted to static scenes, and existing 4D approaches that lack semantic integration, FE-4DGS seamlessly leverages pre-trained 2D semantic embeddings to produce a unified 4D representation-where semantics also deform with tissue motion. This unified approach enables the generation of real-time RGB and semantic outputs through a single, parallelized rasterization process. Despite the additional complexity from feature distillation, FE-4DGS sustains real-time rendering (61 FPS) with a compact footprint, achieves state-of-the-art rendering fidelity on EndoNeRF (39.1 PSNR) and SCARED (27.3 PSNR), and delivers competitive EndoVis18 segmentation, matching or exceeding strong 2D baselines for binary segmentation tasks (0.93 DSC) and remaining competitive for multi-label segmentation (0.77 DSC).
>
---
#### [replaced 013] DICE: Discrete Inversion Enabling Controllable Editing for Multinomial Diffusion and Masked Generative Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2410.08207v3](https://arxiv.org/pdf/2410.08207v3)**

> **作者:** Xiaoxiao He; Quan Dao; Ligong Han; Song Wen; Minhao Bai; Di Liu; Han Zhang; Martin Renqiang Min; Felix Juefei-Xu; Chaowei Tan; Bo Liu; Kang Li; Hongdong Li; Junzhou Huang; Faez Ahmed; Akash Srivastava; Dimitris Metaxas
>
> **备注:** Project webpage: https://hexiaoxiao-cs.github.io/DICE/. This paper was accepted to CVPR 2025 but later desk-rejected post camera-ready, due to a withdrawal from ICLR made 14 days before reviewer assignment
>
> **摘要:** Discrete diffusion models have achieved success in tasks like image generation and masked language modeling but face limitations in controlled content editing. We introduce DICE (Discrete Inversion for Controllable Editing), the first approach to enable precise inversion for discrete diffusion models, including multinomial diffusion and masked generative models. By recording noise sequences and masking patterns during the reverse diffusion process, DICE enables accurate reconstruction and flexible editing of discrete data without the need for predefined masks or attention manipulation. We demonstrate the effectiveness of DICE across both image and text domains, evaluating it on models such as VQ-Diffusion, Paella, and RoBERTa. Our results show that DICE preserves high data fidelity while enhancing editing capabilities, offering new opportunities for fine-grained content manipulation in discrete spaces.
>
---
#### [replaced 014] Improving the generalization of gait recognition with limited datasets
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15176v4](https://arxiv.org/pdf/2505.15176v4)**

> **作者:** Qian Zhou; Xianda Guo; Jilong Wang; Chuanfu Shen; Zhongyuan Wang; Zhen Han; Qin Zou; Shiqi Yu
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Generalized gait recognition remains challenging due to significant domain shifts in viewpoints, appearances, and environments. Mixed-dataset training has recently become a practical route to improve cross-domain robustness, but it introduces underexplored issues: 1) inter-dataset supervision conflicts, which distract identity learning, and 2) redundant or noisy samples, which reduce data efficiency and may reinforce dataset-specific patterns. To address these challenges, we introduce a unified paradigm for cross-dataset gait learning that simultaneously improves motion-signal quality and supervision consistency. We first increase the reliability of training data by suppressing sequences dominated by redundant gait cycles or unstable silhouettes, guided by representation redundancy and prediction uncertainty. This refinement concentrates learning on informative gait dynamics when mixing heterogeneous datasets. In parallel, we stabilize supervision by disentangling metric learning across datasets, forming triplets within each source to prevent destructive cross-domain gradients while preserving transferable identity cues. These components act in synergy to stabilize optimization and strengthen generalization without modifying network architectures or requiring extra annotations. Experiments on CASIA-B, OU-MVLP, Gait3D, and GREW with both GaitBase and DeepGaitV2 backbones consistently show improved cross-domain performance without sacrificing in-domain accuracy. These results demonstrate that data selection and aligning supervision effectively enables scalable mixed-dataset gait learning.
>
---
#### [replaced 015] HD$^2$-SSC: High-Dimension High-Density Semantic Scene Completion for Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07925v2](https://arxiv.org/pdf/2511.07925v2)**

> **作者:** Zhiwen Yang; Yuxin Peng
>
> **备注:** 10 pages, 6 figures, accepted by AAAI 2026
>
> **摘要:** Camera-based 3D semantic scene completion (SSC) plays a crucial role in autonomous driving, enabling voxelized 3D scene understanding for effective scene perception and decision-making. Existing SSC methods have shown efficacy in improving 3D scene representations, but suffer from the inherent input-output dimension gap and annotation-reality density gap, where the 2D planner view from input images with sparse annotated labels leads to inferior prediction of real-world dense occupancy with a 3D stereoscopic view. In light of this, we propose the corresponding High-Dimension High-Density Semantic Scene Completion (HD$^2$-SSC) framework with expanded pixel semantics and refined voxel occupancies. To bridge the dimension gap, a High-dimension Semantic Decoupling module is designed to expand 2D image features along a pseudo third dimension, decoupling coarse pixel semantics from occlusions, and then identify focal regions with fine semantics to enrich image features. To mitigate the density gap, a High-density Occupancy Refinement module is devised with a "detect-and-refine" architecture to leverage contextual geometric and semantic structures for enhanced semantic density with the completion of missing voxels and correction of erroneous ones. Extensive experiments and analyses on the SemanticKITTI and SSCBench-KITTI-360 datasets validate the effectiveness of our HD$^2$-SSC framework.
>
---
#### [replaced 016] Understanding while Exploring: Semantics-driven Active Mapping
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.00225v2](https://arxiv.org/pdf/2506.00225v2)**

> **作者:** Liyan Chen; Huangying Zhan; Hairong Yin; Yi Xu; Philippos Mordohai
>
> **摘要:** Effective robotic autonomy in unknown environments demands proactive exploration and precise understanding of both geometry and semantics. In this paper, we propose ActiveSGM, an active semantic mapping framework designed to predict the informativeness of potential observations before execution. Built upon a 3D Gaussian Splatting (3DGS) mapping backbone, our approach employs semantic and geometric uncertainty quantification, coupled with a sparse semantic representation, to guide exploration. By enabling robots to strategically select the most beneficial viewpoints, ActiveSGM efficiently enhances mapping completeness, accuracy, and robustness to noisy semantic data, ultimately supporting more adaptive scene exploration. Our experiments on the Replica and Matterport3D datasets highlight the effectiveness of ActiveSGM in active semantic mapping tasks.
>
---
#### [replaced 017] Wi-CBR: Salient-aware Adaptive WiFi Sensing for Cross-domain Behavior Recognition
- **分类: cs.CV; eess.SP**

- **链接: [https://arxiv.org/pdf/2506.11616v3](https://arxiv.org/pdf/2506.11616v3)**

> **作者:** Ruobei Zhang; Shengeng Tang; Huan Yan; Xiang Zhang; Jiabao Guo
>
> **摘要:** The challenge in WiFi-based cross-domain Behavior Recognition lies in the significant interference of domain-specific signals on gesture variation. However, previous methods alleviate this interference by mapping the phase from multiple domains into a common feature space. If the Doppler Frequency Shift (DFS) signal is used to dynamically supplement the phase features to achieve better generalization, it enables the model to not only explore a wider feature space but also to avoid potential degradation of gesture semantic information. Specifically, we propose a novel Salient-aware Adaptive WiFi Sensing for Cross-domain Behavior Recognition (Wi-CBR), which constructs a dual-branch self-attention module that captures temporal features from phase information reflecting dynamic path length variations while extracting kinematic features from DFS correlated with motion velocity. Moreover, we design a Saliency Guidance Module that employs group attention mechanisms to mine critical activity features and utilizes gating mechanisms to optimize information entropy, facilitating feature fusion and enabling effective interaction between salient and non-salient behavioral characteristics. Extensive experiments on two large-scale public datasets (Widar3.0 and XRF55) demonstrate the superior performance of our method in both in-domain and cross-domain scenarios.
>
---
#### [replaced 018] STATIC : Surface Temporal Affine for TIme Consistency in Video Monocular Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.01090v2](https://arxiv.org/pdf/2412.01090v2)**

> **作者:** Sunghun Yang; Minhyeok Lee; Suhwan Cho; Jungho Lee; Sangyoun Lee
>
> **摘要:** Video monocular depth estimation is essential for applications such as autonomous driving, AR/VR, and robotics. Recent transformer-based single-image monocular depth estimation models perform well on single images but struggle with depth consistency across video frames. Traditional methods aim to improve temporal consistency using multi-frame temporal modules or prior information like optical flow and camera parameters. However, these approaches face issues such as high memory use, reduced performance with dynamic or irregular motion, and limited motion understanding. We propose STATIC, a novel model that independently learns temporal consistency in static and dynamic area without additional information. A difference mask from surface normals identifies static and dynamic area by measuring directional variance. For static area, the Masked Static (MS) module enhances temporal consistency by focusing on stable regions. For dynamic area, the Surface Normal Similarity (SNS) module aligns areas and enhances temporal consistency by measuring feature similarity between frames. A final refinement integrates the independently learned static and dynamic area, enabling STATIC to achieve temporal consistency across the entire sequence. Our method achieves state-of-the-art video depth estimation on the KITTI and NYUv2 datasets without additional information.
>
---
#### [replaced 019] Generating Attribute-Aware Human Motions from Textual Prompt
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2506.21912v2](https://arxiv.org/pdf/2506.21912v2)**

> **作者:** Xinghan Wang; Kun Xu; Fei Li; Cao Sheng; Jiazhong Yu; Yadong Mu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Text-driven human motion generation has recently attracted considerable attention, allowing models to generate human motions based on textual descriptions. However, current methods neglect the influence of human attributes-such as age, gender, weight, and height-which are key factors shaping human motion patterns. This work represents a pilot exploration for bridging this gap. We conceptualize each motion as comprising both attribute information and action semantics, where textual descriptions align exclusively with action semantics. To achieve this, a new framework inspired by Structural Causal Models is proposed to decouple action semantics from human attributes, enabling text-to-semantics prediction and attribute-controlled generation. The resulting model is capable of generating attribute-aware motion aligned with the user's text and attribute inputs. For evaluation, we introduce a comprehensive dataset containing attribute annotations for text-motion pairs, setting the first benchmark for attribute-aware motion generation. Extensive experiments validate our model's effectiveness.
>
---
#### [replaced 020] Beyond Frequency: Seeing Subtle Cues Through the Lens of Spatial Decomposition for Fine-Grained Visual Classification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.06959v2](https://arxiv.org/pdf/2508.06959v2)**

> **作者:** Qin Xu; Lili Zhu; Xiaoxia Cheng; Bo Jiang
>
> **备注:** After supplementary experiments and careful review, minor inconsistencies in prompt template configuration and partial experimental parameter records were identified. To ensure research accuracy, rigor, and reproducibility, we will revise technical descriptions, verify results with standardized parameters, and resubmit a polished version soon. Apologies for any inconvenience
>
> **摘要:** The crux of resolving fine-grained visual classification (FGVC) lies in capturing discriminative and class-specific cues that correspond to subtle visual characteristics. Recently, frequency decomposition/transform based approaches have attracted considerable interests since its appearing discriminative cue mining ability. However, the frequency-domain methods are based on fixed basis functions, lacking adaptability to image content and unable to dynamically adjust feature extraction according to the discriminative requirements of different images. To address this, we propose a novel method for FGVC, named Subtle-Cue Oriented Perception Engine (SCOPE), which adaptively enhances the representational capability of low-level details and high-level semantics in the spatial domain, breaking through the limitations of fixed scales in the frequency domain and improving the flexibility of multi-scale fusion. The core of SCOPE lies in two modules: the Subtle Detail Extractor (SDE), which dynamically enhances subtle details such as edges and textures from shallow features, and the Salient Semantic Refiner (SSR), which learns semantically coherent and structure-aware refinement features from the high-level features guided by the enhanced shallow features. The SDE and SSR are cascaded stage-by-stage to progressively combine local details with global semantics. Extensive experiments demonstrate that our method achieves new state-of-the-art on four popular fine-grained image classification benchmarks.
>
---
#### [replaced 021] Test-Time Reinforcement Learning for GUI Grounding via Region Consistency
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2508.05615v2](https://arxiv.org/pdf/2508.05615v2)**

> **作者:** Yong Du; Yuchen Yan; Fei Tang; Zhengxi Lu; Chang Zong; Weiming Lu; Shengpei Jiang; Yongliang Shen
>
> **备注:** [Accepted by AAAI2026] Project Page: https://zju-real.github.io/gui-rcpo Code: https://github.com/zju-real/gui-rcpo
>
> **摘要:** Graphical User Interface (GUI) grounding, the task of mapping natural language instructions to precise screen coordinates, is fundamental to autonomous GUI agents. While existing methods achieve strong performance through extensive supervised training or reinforcement learning with labeled rewards, they remain constrained by the cost and availability of pixel-level annotations. We observe that when models generate multiple predictions for the same GUI element, the spatial overlap patterns reveal implicit confidence signals that can guide more accurate localization. Leveraging this insight, we propose GUI-RC (Region Consistency), a test-time scaling method that constructs spatial voting grids from multiple sampled predictions to identify consensus regions where models show highest agreement. Without any training, GUI-RC improves accuracy by 2-3% across various architectures on ScreenSpot benchmarks. We further introduce GUI-RCPO (Region Consistency Policy Optimization), transforming these consistency patterns into rewards for test-time reinforcement learning. By computing how well each prediction aligns with the collective consensus, GUI-RCPO enables models to iteratively refine their outputs on unlabeled data during inference. Extensive experiments demonstrate the generality of our approach: using only 1,272 unlabeled data, GUI-RCPO achieves 3-6% accuracy improvements across various architectures on ScreenSpot benchmarks. Our approach reveals the untapped potential of test-time scaling and test-time reinforcement learning for GUI grounding, offering a promising path toward more data-efficient GUI agents.
>
---
#### [replaced 022] VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.20322v2](https://arxiv.org/pdf/2509.20322v2)**

> **作者:** Shaofeng Yin; Yanjie Ze; Hong-Xing Yu; C. Karen Liu; Jiajun Wu
>
> **备注:** Website: https://visualmimic.github.io
>
> **摘要:** Humanoid loco-manipulation in unstructured environments demands tight integration of egocentric perception and whole-body control. However, existing approaches either depend on external motion capture systems or fail to generalize across diverse tasks. We introduce VisualMimic, a visual sim-to-real framework that unifies egocentric vision with hierarchical whole-body control for humanoid robots. VisualMimic combines a task-agnostic low-level keypoint tracker -- trained from human motion data via a teacher-student scheme -- with a task-specific high-level policy that generates keypoint commands from visual and proprioceptive input. To ensure stable training, we inject noise into the low-level policy and clip high-level actions using human motion statistics. VisualMimic enables zero-shot transfer of visuomotor policies trained in simulation to real humanoid robots, accomplishing a wide range of loco-manipulation tasks such as box lifting, pushing, football dribbling, and kicking. Beyond controlled laboratory settings, our policies also generalize robustly to outdoor environments. Videos are available at: https://visualmimic.github.io .
>
---
#### [replaced 023] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04369v4](https://arxiv.org/pdf/2508.04369v4)**

> **作者:** Canhui Tang; Zifan Han; Hongbo Sun; Sanping Zhou; Xuchong Zhang; Xin Wei; Ye Yuan; Huayu Zhang; Jinglin Xu; Hao Sun
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated significant progress in vision-language tasks, yet they still face challenges when processing long-duration video inputs. The limitation arises from MLLMs' context limit and training costs, necessitating sparse frame sampling before feeding videos into MLLMs. However, building a trainable sampling method remains challenging due to the unsupervised and non-differentiable nature of sparse frame sampling in Video-MLLMs. To address these problems, we propose Temporal Sampling Policy Optimization (TSPO), advancing MLLMs' long-form video-language understanding via reinforcement learning. Specifically, we first propose a trainable event-aware temporal agent, which captures event-query correlation for performing probabilistic keyframe selection. Then, we propose the TSPO reinforcement learning paradigm, which models keyframe selection and language generation as a joint decision-making process, enabling end-to-end group relative optimization for the temporal sampling policy. Furthermore, we propose a dual-style long video training data construction pipeline, balancing comprehensive temporal understanding and key segment localization. Finally, we incorporate rule-based answering accuracy and temporal locating reward mechanisms to optimize the temporal sampling policy. Comprehensive experiments show that our TSPO achieves state-of-the-art performance across multiple long video understanding benchmarks, and shows transferable ability across different cutting-edge Video-MLLMs. Our code is available at https://github.com/Hui-design/TSPO
>
---
#### [replaced 024] ImageSet2Text: Describing Sets of Images through Text
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.19361v2](https://arxiv.org/pdf/2503.19361v2)**

> **作者:** Piera Riccio; Francesco Galati; Kajetan Schweighofer; Noa Garcia; Nuria Oliver
>
> **摘要:** In the era of large-scale visual data, understanding collections of images is a challenging yet important task. To this end, we introduce ImageSet2Text, a novel method to automatically generate natural language descriptions of image sets. Based on large language models, visual-question answering chains, an external lexical graph, and CLIP-based verification, ImageSet2Text iteratively extracts key concepts from image subsets and organizes them into a structured concept graph. We conduct extensive experiments evaluating the quality of the generated descriptions in terms of accuracy, completeness, and user satisfaction. We also examine the method's behavior through ablation studies, scalability assessments, and failure analyses. Results demonstrate that ImageSet2Text combines data-driven AI and symbolic representations to reliably summarize large image collections for a wide range of applications.
>
---
#### [replaced 025] LPLC: A Dataset for License Plate Legibility Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.18425v2](https://arxiv.org/pdf/2508.18425v2)**

> **作者:** Lucas Wojcik; Gabriel E. Lima; Valfride Nascimento; Eduil Nascimento; Rayson Laroca; David Menotti
>
> **备注:** Accepted for presentation at the Conference on Graphics, Patterns and Images (SIBGRAPI) 2025
>
> **摘要:** Automatic License Plate Recognition (ALPR) faces a major challenge when dealing with illegible license plates (LPs). While reconstruction methods such as super-resolution (SR) have emerged, the core issue of recognizing these low-quality LPs remains unresolved. To optimize model performance and computational efficiency, image pre-processing should be applied selectively to cases that require enhanced legibility. To support research in this area, we introduce a novel dataset comprising 10,210 images of vehicles with 12,687 annotated LPs for legibility classification (the LPLC dataset). The images span a wide range of vehicle types, lighting conditions, and camera/image quality levels. We adopt a fine-grained annotation strategy that includes vehicle- and LP-level occlusions, four legibility categories (perfect, good, poor, and illegible), and character labels for three categories (excluding illegible LPs). As a benchmark, we propose a classification task using three image recognition networks to determine whether an LP image is good enough, requires super-resolution, or is completely unrecoverable. The overall F1 score, which remained below 80% for all three baseline models (ViT, ResNet, and YOLO), together with the analyses of SR and LP recognition methods, highlights the difficulty of the task and reinforces the need for further research. The proposed dataset is publicly available at https://github.com/lmlwojcik/lplc-dataset.
>
---
#### [replaced 026] Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.11881v4](https://arxiv.org/pdf/2505.11881v4)**

> **作者:** Giyeong Oh; Woohyun Cho; Siyeol Kim; Suhwan Choi; Youngjae Yu
>
> **备注:** 27 pages, maybe final final version
>
> **摘要:** Residual connections are pivotal for deep neural networks, enabling greater depth by mitigating vanishing gradients. However, in standard residual updates, the module's output is directly added to the input stream. This can lead to updates that predominantly reinforce or modulate the existing stream direction, potentially underutilizing the module's capacity for learning entirely novel features. In this work, we introduce Orthogonal Residual Update: we decompose the module's output relative to the input stream and add only the component orthogonal to this stream. This design aims to guide modules to contribute primarily new representational directions, fostering richer feature learning while promoting more efficient training. We demonstrate that our orthogonal update strategy improves generalization accuracy and training stability across diverse architectures (ResNetV2, Vision Transformers) and datasets (CIFARs, TinyImageNet, ImageNet-1k), achieving, for instance, a +3.78 pp top-1 accuracy gain for ViT-B on ImageNet-1k.
>
---
#### [replaced 027] MatchAttention: Matching the Relative Positions for High-Resolution Cross-View Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.14260v2](https://arxiv.org/pdf/2510.14260v2)**

> **作者:** Tingman Yan; Tao Liu; Xilian Yang; Qunfei Zhao; Zeyang Xia
>
> **摘要:** Cross-view matching is fundamentally achieved through cross-attention mechanisms. However, matching of high-resolution images remains challenging due to the quadratic complexity and lack of explicit matching constraints in the existing cross-attention. This paper proposes an attention mechanism, MatchAttention, that dynamically matches relative positions. The relative position determines the attention sampling center of the key-value pairs given a query. Continuous and differentiable sliding-window attention sampling is achieved by the proposed BilinearSoftmax. The relative positions are iteratively updated through residual connections across layers by embedding them into the feature channels. Since the relative position is exactly the learning target for cross-view matching, an efficient hierarchical cross-view decoder, MatchDecoder, is designed with MatchAttention as its core component. To handle cross-view occlusions, gated cross-MatchAttention and a consistency-constrained loss are proposed. These two components collectively mitigate the impact of occlusions in both forward and backward passes, allowing the model to focus more on learning matching relationships. When applied to stereo matching, MatchStereo-B ranked 1st in average error on the public Middlebury benchmark and requires only 29ms for KITTI-resolution inference. MatchStereo-T can process 4K UHD images in 0.1 seconds using only 3GB of GPU memory. The proposed models also achieve state-of-the-art performance on KITTI 2012, KITTI 2015, ETH3D, and Spring flow datasets. The combination of high accuracy and low computational complexity makes real-time, high-resolution, and high-accuracy cross-view matching possible. Project page: https://github.com/TingmanYan/MatchAttention.
>
---
#### [replaced 028] Redundant Queries in DETR-Based 3D Detection Methods: Unnecessary and Prunable
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.02054v3](https://arxiv.org/pdf/2412.02054v3)**

> **作者:** Lizhen Xu; Zehao Wu; Wenzhao Qiu; Shanmin Pang; Xiuxiu Bai; Kuizhi Mei; Jianru Xue
>
> **备注:** AAAI 2026
>
> **摘要:** Query-based models are extensively used in 3D object detection tasks, with a wide range of pre-trained checkpoints readily available online. However, despite their popularity, these models often require an excessive number of object queries, far surpassing the actual number of objects to detect. The redundant queries result in unnecessary computational and memory costs. In this paper, we find that not all queries contribute equally -- a significant portion of queries have a much smaller impact compared to others. Based on this observation, we propose an embarrassingly simple approach called Gradually Pruning Queries (GPQ), which prunes queries incrementally based on their classification scores. A key advantage of GPQ is that it requires no additional learnable parameters. It is straightforward to implement in any query-based method, as it can be seamlessly integrated as a fine-tuning step using an existing checkpoint after training. With GPQ, users can easily generate multiple models with fewer queries, starting from a checkpoint with an excessive number of queries. Experiments on various advanced 3D detectors show that GPQ effectively reduces redundant queries while maintaining performance. Using our method, model inference on desktop GPUs can be accelerated by up to 1.35x. Moreover, after deployment on edge devices, it achieves up to a 67.86% reduction in FLOPs and a 65.16% decrease in inference time. The code will be available at https://github.com/iseri27/Gpq.
>
---
#### [replaced 029] Depth Matters: Multimodal RGB-D Perception for Robust Autonomous Agents
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.16711v3](https://arxiv.org/pdf/2503.16711v3)**

> **作者:** Mihaela-Larisa Clement; Mónika Farsang; Felix Resch; Mihai-Teodor Stanusoiu; Radu Grosu
>
> **摘要:** Autonomous agents that rely purely on perception to make real-time control decisions require efficient and robust architectures. In this work, we demonstrate that augmenting RGB input with depth information significantly enhances our agents' ability to predict steering commands compared to using RGB alone. We benchmark lightweight recurrent controllers that leverage the fused RGB-D features for sequential decision-making. To train our models, we collect high-quality data using a small-scale autonomous car controlled by an expert driver via a physical steering wheel, capturing varying levels of steering difficulty. Our models were successfully deployed on real hardware and inherently avoided dynamic and static obstacles, under out-of-distribution conditions. Specifically, our findings reveal that the early fusion of depth data results in a highly robust controller, which remains effective even with frame drops and increased noise levels, without compromising the network's focus on the task.
>
---
#### [replaced 030] LLM-Guided Probabilistic Fusion for Label-Efficient Document Layout Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08903v2](https://arxiv.org/pdf/2511.08903v2)**

> **作者:** Ibne Farabi Shihab; Sanjeda Akter; Anuj Sharma
>
> **摘要:** Document layout understanding remains data-intensive despite advances in semi-supervised learning. We present a framework that enhances semi-supervised detection by fusing visual predictions with structural priors from text-pretrained LLMs via principled probabilistic weighting. Given unlabeled documents, an OCR-LLM pipeline infers hierarchical regions which are combined with teacher detector outputs through inverse-variance fusion to generate refined pseudo-labels.Our method demonstrates consistent gains across model scales. With a lightweight SwiftFormer backbone (26M params), we achieve 88.2$\pm$0.3 AP using only 5\% labels on PubLayNet. When applied to document-pretrained LayoutLMv3 (133M params), our fusion framework reaches 89.7$\pm$0.4 AP, surpassing both LayoutLMv3 with standard semi-supervised learning (89.1$\pm$0.4 AP, p=0.02) and matching UDOP~\cite{udop} (89.8 AP) which requires 100M+ pages of multimodal pretraining. This demonstrates that LLM structural priors are complementary to both lightweight and pretrained architectures. Key findings include: (1) learned instance-adaptive gating improves over fixed weights by +0.9 AP with data-dependent PAC bounds correctly predicting convergence; (2) open-source LLMs enable privacy-preserving deployment with minimal loss (Llama-3-70B: 87.1 AP lightweight, 89.4 AP with LayoutLMv3); (3) LLMs provide targeted semantic disambiguation (18.7\% of cases, +3.8 AP gain) beyond simple text heuristics.Total system cost includes \$12 for GPT-4o-mini API or 17 GPU-hours for local Llama-3-70B per 50K pages, amortized across training runs.
>
---
#### [replaced 031] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.13813v2](https://arxiv.org/pdf/2505.13813v2)**

> **作者:** Matthew Raffel; Lizhong Chen
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** The Kolmogorov-Arnold Network (KAN) has been gaining popularity as an alternative to the multi-layer perceptron (MLP) with its increased expressiveness and interpretability. Even so, the KAN suffers from being orders of magnitude slower due to its increased computational cost and training instability, limiting its applicability to larger-scale tasks. Recently, the Kolmogorov-Arnold Transformer (KAT) has been proposed, which can achieve FLOPs similar to the traditional Transformer with MLPs by leveraging Group-Rational KAN (GR-KAN). Unfortunately, despite the comparable FLOPs, our testing reveals that the KAT is still 123x slower in training speeds, indicating that there are other performance bottlenecks beyond FLOPs. In this paper, we conduct a series of experiments to understand the root cause of the slowdown in KAT. We uncover that the slowdown can be isolated to memory stalls, linked more specifically to inefficient gradient accumulations in the backward pass of GR-KAN. To address this memory bottleneck, we propose FlashKAT, which minimizes accesses to slow memory and the usage of atomic adds through a restructured kernel. Evaluations demonstrate that FlashKAT can achieve a training speedup of 86.5x compared with the state-of-the-art KAT, while reducing rounding errors in the computation of the gradients.
>
---
#### [replaced 032] UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2510.12174v2](https://arxiv.org/pdf/2510.12174v2)**

> **作者:** Yusen Xie; Zhenmin Huang; Jianhao Jiao; Dimitrios Kanoulas; Jun Ma
>
> **摘要:** In this paper, we propose UniGS, a unified map representation and differentiable framework for high-fidelity multimodal 3D reconstruction based on 3D Gaussian Splatting. Our framework integrates a CUDA-accelerated rasterization pipeline capable of rendering photo-realistic RGB images, geometrically accurate depth maps, consistent surface normals, and semantic logits simultaneously. We redesign the rasterization to render depth via differentiable ray-ellipsoid intersection rather than using Gaussian centers, enabling effective optimization of rotation and scale attribute through analytic depth gradients. Furthermore, we derive the analytic gradient formulation for surface normal rendering, ensuring geometric consistency among reconstructed 3D scenes. To improve computational and storage efficiency, we introduce a learnable attribute that enables differentiable pruning of Gaussians with minimal contribution during training. Quantitative and qualitative experiments demonstrate state-of-the-art reconstruction accuracy across all modalities, validating the efficacy of our geometry-aware paradigm. Source code and multimodal viewer will be available on GitHub.
>
---
#### [replaced 033] Graph-Theoretic Consistency for Robust and Topology-Aware Semi-Supervised Histopathology Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22689v3](https://arxiv.org/pdf/2509.22689v3)**

> **作者:** Ha-Hieu Pham; Minh Le; Han Huynh; Nguyen Quoc Khanh Le; Huy-Hieu Pham
>
> **备注:** Accepted to the AAAI 2026 Student Abstract and Poster Program
>
> **摘要:** Semi-supervised semantic segmentation (SSSS) is vital in computational pathology, where dense annotations are costly and limited. Existing methods often rely on pixel-level consistency, which propagates noisy pseudo-labels and produces fragmented or topologically invalid masks. We propose Topology Graph Consistency (TGC), a framework that integrates graph-theoretic constraints by aligning Laplacian spectra, component counts, and adjacency statistics between prediction graphs and references. This enforces global topology and improves segmentation accuracy. Experiments on GlaS and CRAG demonstrate that TGC achieves state-of-the-art performance under 5-10% supervision and significantly narrows the gap to full supervision.
>
---
#### [replaced 034] SphereDiff: Tuning-free 360° Static and Dynamic Panorama Generation via Spherical Latent Representation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.14396v2](https://arxiv.org/pdf/2504.14396v2)**

> **作者:** Minho Park; Taewoong Kang; Jooyeol Yun; Sungwon Hwang; Jaegul Choo
>
> **备注:** Accepted to AAAI 2026 (Oral)
>
> **摘要:** The increasing demand for AR/VR applications has highlighted the need for high-quality content, such as 360° live wallpapers. However, generating high-quality 360° panoramic contents remains a challenging task due to the severe distortions introduced by equirectangular projection (ERP). Existing approaches either fine-tune pretrained diffusion models on limited ERP datasets or adopt tuning-free methods that still rely on ERP latent representations, often resulting in distracting distortions near the poles. In this paper, we introduce SphereDiff, a novel approach for synthesizing 360° static and live wallpaper with state-of-the-art diffusion models without additional tuning. We define a spherical latent representation that ensures consistent quality across all perspectives, including near the poles. Then, we extend MultiDiffusion to spherical latent representation and propose a dynamic spherical latent sampling method to enable direct use of pretrained diffusion models. Moreover, we introduce distortion-aware weighted averaging to further improve the generation quality. Our method outperforms existing approaches in generating 360° static and live wallpaper, making it a robust solution for immersive AR/VR applications. The code is available here. https://github.com/pmh9960/SphereDiff
>
---
#### [replaced 035] TUS-REC2024: A Challenge to Reconstruct 3D Freehand Ultrasound Without External Tracker
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.21765v2](https://arxiv.org/pdf/2506.21765v2)**

> **作者:** Qi Li; Shaheer U. Saeed; Yuliang Huang; Mingyuan Luo; Zhongnuo Yan; Jiongquan Chen; Xin Yang; Dong Ni; Nektarios Winter; Phuc Nguyen; Lucas Steinberger; Caelan Haney; Yuan Zhao; Mingjie Jiang; Bowen Ren; SiYeoul Lee; Seonho Kim; MinKyung Seo; MinWoo Kim; Yimeng Dou; Zhiwei Zhang; Yin Li; Tomy Varghese; Dean C. Barratt; Matthew J. Clarkson; Tom Vercauteren; Yipeng Hu
>
> **摘要:** Trackerless freehand ultrasound reconstruction aims to reconstruct 3D volumes from sequences of 2D ultrasound images without relying on external tracking systems. By eliminating the need for optical or electromagnetic trackers, this approach offers a low-cost, portable, and widely deployable alternative to more expensive volumetric ultrasound imaging systems, particularly valuable in resource-constrained clinical settings. However, predicting long-distance transformations and handling complex probe trajectories remain challenging. The TUS-REC2024 Challenge establishes the first benchmark for trackerless 3D freehand ultrasound reconstruction by providing a large publicly available dataset, along with a baseline model and a rigorous evaluation framework. By the submission deadline, the Challenge had attracted 43 registered teams, of which 6 teams submitted 21 valid dockerized solutions. The submitted methods span a wide range of approaches, including the state space model, the recurrent model, the registration-driven volume refinement, the attention mechanism, and the physics-informed model. This paper provides a comprehensive background introduction and literature review in the field, presents an overview of the challenge design and dataset, and offers a comparative analysis of submitted methods across multiple evaluation metrics. These analyses highlight both the progress and the current limitations of state-of-the-art approaches in this domain and provide insights for future research directions. All data and code are publicly available to facilitate ongoing development and reproducibility. As a live and evolving benchmark, it is designed to be continuously iterated and improved. The Challenge was held at MICCAI 2024 and is organised again at MICCAI 2025, reflecting its sustained commitment to advancing this field.
>
---
#### [replaced 036] VADB: A Large-Scale Video Aesthetic Database with Professional and Multi-Dimensional Annotations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.25238v2](https://arxiv.org/pdf/2510.25238v2)**

> **作者:** Qianqian Qiao; DanDan Zheng; Yihang Bo; Bao Peng; Heng Huang; Longteng Jiang; Huaye Wang; Jingdong Chen; Jun Zhou; Xin Jin
>
> **摘要:** Video aesthetic assessment, a vital area in multimedia computing, integrates computer vision with human cognition. Its progress is limited by the lack of standardized datasets and robust models, as the temporal dynamics of video and multimodal fusion challenges hinder direct application of image-based methods. This study introduces VADB, the largest video aesthetic database with 10,490 diverse videos annotated by 37 professionals across multiple aesthetic dimensions, including overall and attribute-specific aesthetic scores, rich language comments and objective tags. We propose VADB-Net, a dual-modal pre-training framework with a two-stage training strategy, which outperforms existing video quality assessment models in scoring tasks and supports downstream video aesthetic assessment tasks. The dataset and source code are available at https://github.com/BestiVictory/VADB.
>
---
#### [replaced 037] Agent Journey Beyond RGB: Hierarchical Semantic-Spatial Representation Enrichment for Vision-and-Language Navigation
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2412.06465v5](https://arxiv.org/pdf/2412.06465v5)**

> **作者:** Xuesong Zhang; Yunbo Xu; Jia Li; Ruonan Liu; Zhenzhen Hu
>
> **备注:** AAAI2026, I14 pages, 12 figures, 11 tables
>
> **摘要:** Navigating unseen environments from natural language instructions remains challenging for egocentric agents in Vision-and-Language Navigation (VLN). Humans naturally ground concrete semantic knowledge within spatial layouts during indoor navigation. Although prior work has introduced diverse environment representations to improve reasoning, auxiliary modalities are often naively concatenated with RGB features, which underutilizes each modality's distinct contribution. We propose a hierarchical Semantic Understanding and Spatial Awareness (SUSA) architecture to enable agents to perceive and ground environments at multiple scales. Specifically, the Textual Semantic Understanding (TSU) module supports local action prediction by generating view-level descriptions, capturing fine-grained semantics and narrowing the modality gap between instructions and environments. Complementarily, the Depth Enhanced Spatial Perception (DSP) module incrementally builds a trajectory-level depth exploration map, providing a coarse-grained representation of global spatial layout. Extensive experiments show that the hierarchical representation enrichment of SUSA significantly improves navigation performance over the baseline on discrete VLN benchmarks (REVERIE, R2R, and SOON) and generalizes better to the continuous R2R-CE benchmark.
>
---
#### [replaced 038] Organizing Unstructured Image Collections using Natural Language
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.05217v5](https://arxiv.org/pdf/2410.05217v5)**

> **作者:** Mingxuan Liu; Zhun Zhong; Jun Li; Gianni Franchi; Subhankar Roy; Elisa Ricci
>
> **备注:** Preprint. Project webpage: https://oatmealliu.github.io/xcluster.html
>
> **摘要:** In this work, we introduce and study the novel task of Open-ended Semantic Multiple Clustering (OpenSMC). Given a large, unstructured image collection, the goal is to automatically discover several, diverse semantic clustering criteria (e.g., Activity or Location) from the images, and subsequently organize them according to the discovered criteria, without requiring any human input. Our framework, X-Cluster: eXploratory Clustering, treats text as a reasoning proxy: it concurrently scans the entire image collection, proposes candidate criteria in natural language, and groups images into meaningful clusters per criterion. This radically differs from previous works, which either assume predefined clustering criteria or fixed cluster counts. To evaluate X-Cluster, we create two new benchmarks, COCO-4C and Food-4C, each annotated with four distinct grouping criteria and corresponding cluster labels. Experiments show that X-Cluster can effectively reveal meaningful partitions on several datasets. Finally, we use X-Cluster to achieve various real-world applications, including uncovering hidden biases in text-to-image (T2I) generative models and analyzing image virality on social media. Code and datasets will be open-sourced for future research.
>
---
#### [replaced 039] Boosting Adversarial Transferability via Ensemble Non-Attention
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.08937v2](https://arxiv.org/pdf/2511.08937v2)**

> **作者:** Yipeng Zou; Qin Liu; Jie Wu; Yu Peng; Guo Chen; Hui Zhou; Guanghui Ye
>
> **备注:** 16 pages, 11 figures, accepted by AAAI 2026
>
> **摘要:** Ensemble attacks integrate the outputs of surrogate models with diverse architectures, which can be combined with various gradient-based attacks to improve adversarial transferability. However, previous work shows unsatisfactory attack performance when transferring across heterogeneous model architectures. The main reason is that the gradient update directions of heterogeneous surrogate models differ widely, making it hard to reduce the gradient variance of ensemble models while making the best of individual model. To tackle this challenge, we design a novel ensemble attack, NAMEA, which for the first time integrates the gradients from the non-attention areas of ensemble models into the iterative gradient optimization process. Our design is inspired by the observation that the attention areas of heterogeneous models vary sharply, thus the non-attention areas of ViTs are likely to be the focus of CNNs and vice versa. Therefore, we merge the gradients respectively from the attention and non-attention areas of ensemble models so as to fuse the transfer information of CNNs and ViTs. Specifically, we pioneer a new way of decoupling the gradients of non-attention areas from those of attention areas, while merging gradients by meta-learning. Empirical evaluations on ImageNet dataset indicate that NAMEA outperforms AdaEA and SMER, the state-of-the-art ensemble attacks by an average of 15.0% and 9.6%, respectively. This work is the first attempt to explore the power of ensemble non-attention in boosting cross-architecture transferability, providing new insights into launching ensemble attacks.
>
---
#### [replaced 040] RangeSAM: On the Potential of Visual Foundation Models for Range-View represented LiDAR segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.15886v3](https://arxiv.org/pdf/2509.15886v3)**

> **作者:** Paul Julius Kühn; Duc Anh Nguyen; Arjan Kuijper; Holger Graf; Saptarshi Neil Sinha
>
> **摘要:** Point cloud segmentation is central to autonomous driving and 3D scene understanding. While voxel- and point-based methods dominate recent research due to their compatibility with deep architectures and ability to capture fine-grained geometry, they often incur high computational cost, irregular memory access, and limited real-time efficiency. In contrast, range-view methods, though relatively underexplored - can leverage mature 2D semantic segmentation techniques for fast and accurate predictions. Motivated by the rapid progress in Visual Foundation Models (VFMs) for captioning, zero-shot recognition, and multimodal tasks, we investigate whether SAM2, the current state-of-the-art VFM for segmentation tasks, can serve as a strong backbone for LiDAR point cloud segmentation in the range view. We present , to our knowledge, the first range-view framework that adapts SAM2 to 3D segmentation, coupling efficient 2D feature extraction with standard projection/back-projection to operate on point clouds. To optimize SAM2 for range-view representations, we implement several architectural modifications to the encoder: (1) a novel module that emphasizes horizontal spatial dependencies inherent in LiDAR range images, (2) a customized configuration of tailored to the geometric properties of spherical projections, and (3) an adapted mechanism in the encoder backbone specifically designed to capture the unique spatial patterns and discontinuities present in range-view pseudo-images. Our approach achieves competitive performance on SemanticKITTI while benefiting from the speed, scalability, and deployment simplicity of 2D-centric pipelines. This work highlights the viability of VFMs as general-purpose backbones for 3D perception and opens a path toward unified, foundation-model-driven LiDAR segmentation. Results lets us conclude that range-view segmentation methods using VFMs leads to promising results.
>
---
#### [replaced 041] Abn-BLIP: Abnormality-aligned Bootstrapping Language-Image Pre-training for Pulmonary Embolism Diagnosis and Report Generation from CTPA
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.02034v3](https://arxiv.org/pdf/2503.02034v3)**

> **作者:** Zhusi Zhong; Yuli Wang; Lulu Bi; Zhuoqi Ma; Sun Ho Ahn; Christopher J. Mullin; Colin F. Greineder; Michael K. Atalay; Scott Collins; Grayson L. Baird; Cheng Ting Lin; Webster Stayman; Todd M. Kolb; Ihab Kamel; Harrison X. Bai; Zhicheng Jiao
>
> **摘要:** Medical imaging plays a pivotal role in modern healthcare, with computed tomography pulmonary angiography (CTPA) being a critical tool for diagnosing pulmonary embolism and other thoracic conditions. However, the complexity of interpreting CTPA scans and generating accurate radiology reports remains a significant challenge. This paper introduces Abn-BLIP (Abnormality-aligned Bootstrapping Language-Image Pretraining), an advanced diagnosis model designed to align abnormal findings to generate the accuracy and comprehensiveness of radiology reports. By leveraging learnable queries and cross-modal attention mechanisms, our model demonstrates superior performance in detecting abnormalities, reducing missed findings, and generating structured reports compared to existing methods. Our experiments show that Abn-BLIP outperforms state-of-the-art medical vision-language models and 3D report generation methods in both accuracy and clinical relevance. These results highlight the potential of integrating multimodal learning strategies for improving radiology reporting. The source code is available at https://github.com/zzs95/abn-blip.
>
---
#### [replaced 042] Two Heads are Better than One: Robust Learning Meets Multi-branch Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2208.08083v4](https://arxiv.org/pdf/2208.08083v4)**

> **作者:** Zongyuan Zhang; Qingwen Bu; Tianyang Duan; Zheng Lin; Yuhao Qing; Zihan Fang; Heming Cui; Dong Huang
>
> **备注:** Camera-ready version for ICPADS 2025
>
> **摘要:** Deep neural networks (DNNs) are vulnerable to adversarial examples, in which DNNs are misled to false outputs due to inputs containing imperceptible perturbations. Adversarial training, a reliable and effective method of defense, may significantly reduce the vulnerability of neural networks and becomes the de facto standard for robust learning. While many recent works practice the data-centric philosophy, such as how to generate better adversarial examples or use generative models to produce additional training data, we look back to the models themselves and revisit the adversarial robustness from the perspective of deep feature distribution as an insightful complementarity. In this paper, we propose \textit{Branch Orthogonality adveRsarial Training} (BORT) to obtain state-of-the-art performance with solely the original dataset for adversarial training. To practice our design idea of integrating multiple orthogonal solution spaces, we leverage a simple multi-branch neural network and propose a corresponding loss function, branch-orthogonal loss, to make each solution space of the multi-branch model orthogonal. We evaluate our approach on CIFAR-10, CIFAR-100 and SVHN against $\ell_{\infty}$ norm-bounded perturbations of size $ε= 8/255$, respectively. Exhaustive experiments are conducted to show that our method goes beyond all state-of-the-art methods without any tricks. Compared to all methods that do not use additional data for training, our models achieve 67.3\% and 41.5\% robust accuracy on CIFAR-10 and CIFAR-100 (improving upon the state-of-the-art by +7.23\% and +9.07\%).
>
---
#### [replaced 043] LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts
- **分类: cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2505.13928v3](https://arxiv.org/pdf/2505.13928v3)**

> **作者:** Qifeng Cai; Hao Liang; Hejun Dong; Meiyi Qiang; Ruichuan An; Zhaoyang Han; Zhengzhou Zhu; Bin Cui; Wentao Zhang
>
> **摘要:** Long videos contain a vast amount of information, making video-text retrieval an essential and challenging task in multimodal learning. However, existing benchmarks suffer from limited video duration, low-quality captions, and coarse annotation granularity, which hinder the evaluation of advanced video-text retrieval methods. To address these limitations, we introduce LoVR, a benchmark specifically designed for long video-text retrieval. LoVR contains 467 long videos and over 40,804 fine-grained clips with high-quality captions. To overcome the issue of poor machine-generated annotations, we propose an efficient caption generation framework that integrates VLM automatic generation, caption quality scoring, and dynamic refinement. This pipeline improves annotation accuracy while maintaining scalability. Furthermore, we introduce a semantic fusion method to generate coherent full-video captions without losing important contextual information. Our benchmark introduces longer videos, more detailed captions, and a larger-scale dataset, presenting new challenges for video understanding and retrieval. Extensive experiments on various advanced embedding models demonstrate that LoVR is a challenging benchmark, revealing the limitations of current approaches and providing valuable insights for future research. We release the code and dataset link at https://github.com/TechNomad-ds/LoVR-benchmark
>
---
#### [replaced 044] Multi-view Structural Convolution Network for Domain-Invariant Point Cloud Recognition of Autonomous Vehicles
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.16289v5](https://arxiv.org/pdf/2501.16289v5)**

> **作者:** Younggun Kim; Mohamed Abdel-Aty; Beomsik Cho; Seonghoon Ryoo; Soomok Lee
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Point cloud representation has recently become a research hotspot in the field of computer vision and has been utilized for autonomous vehicles. However, adapting deep learning networks for point cloud data recognition is challenging due to the variability in datasets and sensor technologies. This variability underscores the necessity for adaptive techniques to maintain accuracy under different conditions. In this paper, we present the Multi-View Structural Convolution Network (MSCN) designed for domain-invariant point cloud recognition. MSCN comprises Structural Convolution Layers (SCL) that extract local context geometric features from point clouds and Structural Aggregation Layers (SAL) that extract and aggregate both local and overall context features from point clouds. Furthermore, MSCN enhances feature robustness by training with unseen domain point clouds generated from the source domain, enabling the model to acquire domain-invariant representations. Extensive cross-domain experiments demonstrate that MSCN achieves an average accuracy of 82.0%, surpassing the strong baseline PointTransformer by 15.8%, confirming its effectiveness under real-world domain shifts. Our code is available at https://github.com/MLMLab/MSCN.
>
---
#### [replaced 045] Intraoperative 2D/3D Registration via Spherical Similarity Learning and Inference-Time Differentiable Levenberg-Marquardt Optimization
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2509.06890v2](https://arxiv.org/pdf/2509.06890v2)**

> **作者:** Minheng Chen; Youyong Kong
>
> **备注:** WACV 2026 Accepted
>
> **摘要:** Intraoperative 2D/3D registration aligns preoperative 3D volumes with real-time 2D radiographs, enabling accurate localization of instruments and implants. A recent fully differentiable similarity learning framework approximates geodesic distances on SE(3), expanding the capture range of registration and mitigating the effects of substantial disturbances, but existing Euclidean approximations distort manifold structure and slow convergence. To address these limitations, we explore similarity learning in non-Euclidean spherical feature spaces to better capture and fit complex manifold structure. We extract feature embeddings using a CNN-Transformer encoder, project them into spherical space, and approximate their geodesic distances with Riemannian distances in the bi-invariant SO(4) space. This enables a more expressive and geometrically consistent deep similarity metric, enhancing the ability to distinguish subtle pose differences. During inference, we replace gradient descent with fully differentiable Levenberg-Marquardt optimization to accelerate convergence. Experiments on real and synthetic datasets show superior accuracy in both patient-specific and patient-agnostic scenarios.
>
---
#### [replaced 046] Seeing the Unseen in Low-light Spike Streams
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23304v2](https://arxiv.org/pdf/2509.23304v2)**

> **作者:** Liwen Hu; Yang Li; Mianzhi Liu; Yijia Guo; Shenghao Xie; Ziluo Ding; Tiejun Huang; Lei Ma
>
> **摘要:** Spike camera, a type of neuromorphic sensor with high-temporal resolution, shows great promise for high-speed visual tasks. Unlike traditional cameras, spike camera continuously accumulates photons and fires asynchronous spike streams. Due to unique data modality, spike streams require reconstruction methods to become perceptible to the human eye. However, lots of methods struggle to handle spike streams in low-light high-speed scenarios due to severe noise and sparse information. In this work, we propose Diff-SPK, a diffusion-based reconstruction method. Diff-SPK effectively leverages generative priors to supplement texture information under diverse low-light conditions. Specifically, it first employs an Enhanced Texture from Inter-spike Interval (ETFI) to aggregate sparse information from low-light spike streams. Then, the encoded ETFI by a suitable encoder serve as the input of ControlNet for high-speed scenes generation. To improve the quality of results, we introduce an ETFI-based feature fusion module during the generation process.
>
---
#### [replaced 047] Image-based Outlier Synthesis With Training Data
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.10794v4](https://arxiv.org/pdf/2411.10794v4)**

> **作者:** Sudarshan Regmi
>
> **备注:** Code: https://github.com/sudarshanregmi/ASCOOD/
>
> **摘要:** Out-of-distribution (OOD) detection is critical to ensure the safe deployment of deep learning models in critical applications. Deep learning models can often misidentify OOD samples as in-distribution (ID) samples. This vulnerability worsens in the presence of spurious correlation in the training set. Likewise, in fine-grained classification settings, detection of fine-grained OOD samples becomes inherently challenging due to their high similarity to ID samples. However, current research on OOD detection has focused instead largely on relatively easier (conventional) cases. Even the few recent works addressing these challenging cases rely on carefully curated or synthesized outliers, ultimately requiring external data. This motivates our central research question: ``Can we innovate OOD detection training framework for fine-grained and spurious settings \textbf{without requiring any external data at all?}" In this work, we present a unified \textbf{A}pproach to \textbf{S}purious, fine-grained, and \textbf{C}onventional \textbf{OOD D}etection (\textbf{\ASCOOD}) that eliminates the reliance on external data. First, we synthesize virtual outliers from ID data by approximating the destruction of invariant features. Specifically, we propose to add gradient attribution values to ID inputs to disrupt invariant features while amplifying true-class logit, thereby synthesizing challenging near-manifold virtual outliers. Then, we simultaneously incentivize ID classification and predictive uncertainty towards virtual outliers. For this, we further propose to leverage standardized features with z-score normalization. ASCOOD effectively mitigates impact of spurious correlations and encourages capturing fine-grained attributes. Extensive experiments across \textbf{7} datasets and and comparisons with \textbf{30+} methods demonstrate merit of ASCOOD in spurious, fine-grained and conventional settings.
>
---
#### [replaced 048] LISA: A Layer-wise Integration and Suppression Approach for Hallucination Mitigation in Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.19110v2](https://arxiv.org/pdf/2507.19110v2)**

> **作者:** Zhihui Guo; Xin Man; Hui Xu; Jie Shao; Zhiguo Jiang; Xianchao Zhang; Heng Tao Shen
>
> **摘要:** Multimodal Large Language Models (MLLMs) excel in vision-language tasks such as image captioning but remain prone to object hallucinations, where they describe objects that do not appear in the image. To mitigate this, we propose LISA, a Layer-wise Integration and Suppression Approach. LISA leverages the layer-wise functional roles in MLLMs: shallow layers provide visual grounding, middle layers encode semantics, and deep layers tend to amplify spurious signals. First, layer-wise spectral modulation stabilizes attention by suppressing over-amplified activations in deeper layers while preserving alignment cues in earlier layers. Second, token-level logits from selected layers are fused via anchor-based routing, with token-wise anchor selection and soft logit fusion enabling adaptive integration during decoding. LISA is fully plug-and-play and can be seamlessly integrated into existing MLLMs, including Qwen2.5-VL. Experiments on multiple benchmarks show that LISA reduces hallucinations by up to 53.6% in $\text{CHAIR}_\text{I}$ and improves POPE F1 by up to 5.1%, demonstrating strong generalization across models and tasks. Our code is available at https://github.com/zhlisa1010-eng/LISA.
>
---
#### [replaced 049] Interpretable and Granular Video-Based Quantification of Motor Characteristics from the Finger Tapping Test in Parkinson Disease
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.18925v3](https://arxiv.org/pdf/2506.18925v3)**

> **作者:** Tahereh Zarrat Ehsan; Michael Tangermann; Yağmur Güçlütürk; Bastiaan R. Bloem; Luc J. W. Evers
>
> **摘要:** Accurately quantifying motor characteristics in Parkinson disease (PD) is crucial for monitoring disease progression and optimizing treatment strategies. The finger-tapping test is a standard motor assessment. Clinicians visually evaluate a patient's tapping performance and assign an overall severity score based on tapping amplitude, speed, and irregularity. However, this subjective evaluation is prone to inter- and intra-rater variability, and does not offer insights into individual motor characteristics captured during this test. This paper introduces a granular computer vision-based method for quantifying PD motor characteristics from video recordings. Four sets of clinically relevant features are proposed to characterize hypokinesia, bradykinesia, sequence effect, and hesitation-halts. We evaluate our approach on video recordings and clinical evaluations of 74 PD patients from the Personalized Parkinson Project. Principal component analysis with varimax rotation shows that the video-based features corresponded to the four deficits. Additionally, video-based analysis has allowed us to identify further granular distinctions within sequence effect and hesitation-halts deficits. In the following, we have used these features to train machine learning classifiers to estimate the Movement Disorder Society Unified Parkinson Disease Rating Scale (MDS-UPDRS) finger-tapping score. Compared to state-of-the-art approaches, our method achieves a higher accuracy in MDS-UPDRS score prediction, while still providing an interpretable quantification of individual finger-tapping motor characteristics. In summary, the proposed framework provides a practical solution for the objective assessment of PD motor characteristics, that can potentially be applied in both clinical and remote settings. Future work is needed to assess its responsiveness to symptomatic treatment and disease progression.
>
---
#### [replaced 050] Cameras as Relative Positional Encoding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.10496v2](https://arxiv.org/pdf/2507.10496v2)**

> **作者:** Ruilong Li; Brent Yi; Junchen Liu; Hang Gao; Yi Ma; Angjoo Kanazawa
>
> **备注:** Project Page: https://www.liruilong.cn/prope/
>
> **摘要:** Transformers are increasingly prevalent for multi-view computer vision tasks, where geometric relationships between viewpoints are critical for 3D perception. To leverage these relationships, multi-view transformers must use camera geometry to ground visual tokens in 3D space. In this work, we compare techniques for conditioning transformers on cameras: token-level raymap encodings, attention-level relative pose encodings, and a new relative encoding we propose -- Projective Positional Encoding (PRoPE) -- that captures complete camera frustums, both intrinsics and extrinsics, as a relative positional encoding. Our experiments begin by showing how relative camera conditioning improves performance in feedforward novel view synthesis, with further gains from PRoPE. This holds across settings: scenes with both shared and varying intrinsics, when combining token- and attention-level conditioning, and for generalization to inputs with out-of-distribution sequence lengths and camera intrinsics. We then verify that these benefits persist for different tasks, stereo depth estimation and discriminative spatial cognition, as well as larger model sizes.
>
---
#### [replaced 051] Xiaoice: Training-Free Video Understanding via Self-Supervised Spatio-Temporal Clustering of Semantic Features
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2510.16781v2](https://arxiv.org/pdf/2510.16781v2)**

> **作者:** Shihao Ji; Zihui Song
>
> **备注:** This paper is being withdrawn because we have identified a significant error in the implementation of our self-supervised clustering approach. Specifically, our feature aggregation step inadvertently leaked temporal information across frames, which violates the core assumption of our training-free method. We sincerely apologize to the research community
>
> **摘要:** The remarkable zero-shot reasoning capabilities of large-scale Visual Language Models (VLMs) on static images have yet to be fully translated to the video domain. Conventional video understanding models often rely on extensive, task-specific training on annotated datasets, a process that is both costly and limited in scalability. This paper introduces a novel, training-free framework for video understanding that circumvents end-to-end training by synergistically combining the rich semantic priors of pre-trained VLMs with classic machine learning algorithms for pattern discovery. Our core idea is to reframe video understanding as a self-supervised spatio-temporal clustering problem within a high-dimensional semantic feature space. The proposed pipeline first transforms a video stream into a semantic feature trajectory using the frozen visual encoder of a pre-trained VLM. Subsequently, we employ Kernel Temporal Segmentation (KTS), a robust machine learning technique, to partition the continuous feature stream into discrete, semantically coherent event segments. These segments are then subjected to unsupervised density-based clustering to identify recurring macroscopic scenes and themes throughout the video. By selecting representative keyframes from each discovered cluster and leveraging the VLM's generative capabilities for textual description, our framework automatically produces a structured, multi-modal summary of the video content. This approach provides an effective, interpretable, and model-agnostic pathway for zero-shot, automated structural analysis of video content.
>
---
#### [replaced 052] Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection
- **分类: cs.CV; cs.CL**

- **链接: [https://arxiv.org/pdf/2509.03113v3](https://arxiv.org/pdf/2509.03113v3)**

> **作者:** Shan Wang; Maying Shen; Nadine Chang; Chuong Nguyen; Hongdong Li; Jose M. Alvarez
>
> **摘要:** Multimodal large language models achieve strong performance across diverse tasks but remain prone to hallucinations, where outputs are not grounded in visual inputs. This issue can be attributed to two main biases: text-visual bias, the overreliance on prompts and prior outputs, and co-occurrence bias, spurious correlations between frequently paired objects. We propose Gradient-based Influence-Aware Constrained Decoding (GACD), an inference-based method, that addresses both biases without auxiliary models, and is readily applicable to existing models without finetuning. The core of our approach is bias estimation, which uses first-order Taylor gradients to understand the contribution of individual tokens-visual features and text tokens-to the current output. Based on this analysis, GACD mitigates hallucinations through two components: (1) suppressing spurious visual features correlated with the output objects, and (2) rebalancing cross-modal contributions by strengthening visual features relative to text. Experiments across multiple benchmarks demonstrate that GACD effectively reduces hallucinations and improves the visual grounding of MLLM outputs.
>
---
#### [replaced 053] Temporal Zoom Networks: Distance Regression and Continuous Depth for Efficient Action Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.03943v3](https://arxiv.org/pdf/2511.03943v3)**

> **作者:** Ibne Farabi Shihab; Sanjeda Akter; Anuj Sharma
>
> **摘要:** Temporal action localization requires both precise boundary detection and computational efficiency. Current methods apply uniform computation across all temporal positions, wasting resources on easy boundaries while struggling with ambiguous ones. We address this through two complementary innovations: Boundary Distance Regression (BDR), which replaces classification-based boundary detection with signed-distance regression achieving 3.3--16.7$\times$ lower variance; and Adaptive Temporal Refinement (ATR), which allocates transformer depth continuously ($τ\in[0,1]$) to concentrate computation near difficult boundaries. On THUMOS14, our method achieves 56.5\% mAP@0.7 and 58.2\% average mAP@[0.3:0.7] with 151G FLOPs, using 36\% fewer FLOPs than ActionFormer++ (55.7\% mAP@0.7 at 235G). Compared to uniform baselines, we achieve +2.9\% mAP@0.7 (+1.8\% avg mAP, 5.4\% relative) with 24\% fewer FLOPs and 29\% lower latency, with particularly strong gains on short actions (+4.2\%, 8.6\% relative). Training requires 1.29$\times$ baseline FLOPs, but this one-time cost is amortized over many inference runs; knowledge distillation further reduces this to 1.1$\times$ while retaining 99.5\% accuracy. Our contributions include: (i) a theoretically-grounded distance formulation with information-theoretic analysis showing optimal variance scaling; (ii) a continuous depth allocation mechanism avoiding discrete routing complexity; and (iii) consistent improvements across four datasets with gains correlating with boundary heterogeneity.
>
---
#### [replaced 054] Mitigating Perception Bias: A Training-Free Approach to Enhance LMM for Image Quality Assessment
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2411.12791v2](https://arxiv.org/pdf/2411.12791v2)**

> **作者:** Baoliang Chen; Siyi Pan; Dongxu Wu; Liang Xie; Xiangjie Sui; Lingyu Zhu; Hanwei Zhu
>
> **摘要:** Despite the impressive performance of large multimodal models (LMMs) in high-level visual tasks, their capacity for image quality assessment (IQA) remains limited. One main reason is that LMMs are primarily trained for high-level tasks (e.g., image captioning), emphasizing unified image semantics extraction under varied quality. Such semantic-aware yet quality-insensitive perception bias inevitably leads to a heavy reliance on image semantics when those LMMs are forced for quality rating. In this paper, instead of retraining or tuning an LMM costly, we propose a training-free debiasing framework, in which the image quality prediction is rectified by mitigating the bias caused by image semantics. Specifically, we first explore several semantic-preserving distortions that can significantly degrade image quality while maintaining identifiable semantics. By applying these specific distortions to the query or test images, we ensure that the degraded images are recognized as poor quality while their semantics mainly remain. During quality inference, both a query image and its corresponding degraded version are fed to the LMM along with a prompt indicating that the query image quality should be inferred under the condition that the degraded one is deemed poor quality. This prior condition effectively aligns the LMM's quality perception, as all degraded images are consistently rated as poor quality, regardless of their semantic variance. Finally, the quality scores of the query image inferred under different prior conditions (degraded versions) are aggregated using a conditional probability model. Extensive experiments on various IQA datasets show that our debiasing framework could consistently enhance the LMM performance.
>
---
#### [replaced 055] Text-to-Scene with Large Reasoning Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.26091v2](https://arxiv.org/pdf/2509.26091v2)**

> **作者:** Frédéric Berdoz; Luca A. Lanzendörfer; Nick Tuninga; Roger Wattenhofer
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Prompt-driven scene synthesis allows users to generate complete 3D environments from textual descriptions. Current text-to-scene methods often struggle with complex geometries and object transformations, and tend to show weak adherence to complex instructions. We address these limitations by introducing Reason-3D, a text-to-scene model powered by large reasoning models (LRMs). Reason-3D integrates object retrieval using captions covering physical, functional, and contextual attributes. Reason-3D then places the selected objects based on implicit and explicit layout constraints, and refines their positions with collision-aware spatial reasoning. Evaluated on instructions ranging from simple to complex indoor configurations, Reason-3D significantly outperforms previous methods in human-rated visual fidelity, adherence to constraints, and asset retrieval quality. Beyond its contribution to the field of text-to-scene generation, our work showcases the advanced spatial reasoning abilities of modern LRMs. Additionally, we release the codebase to further the research in object retrieval and placement with LRMs.
>
---
#### [replaced 056] Explainable Cross-Disease Reasoning for Cardiovascular Risk Assessment from LDCT
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.06625v2](https://arxiv.org/pdf/2511.06625v2)**

> **作者:** Yifei Zhang; Jiashuo Zhang; Mojtaba Safari; Xiaofeng Yang; Liang Zhao
>
> **摘要:** Low-dose chest computed tomography (LDCT) inherently captures both pulmonary and cardiac structures, offering a unique opportunity for joint assessment of lung and cardiovascular health. However, most existing approaches treat these domains as independent tasks, overlooking their physiological interplay and shared imaging biomarkers. We propose an Explainable Cross-Disease Reasoning Framework that enables interpretable cardiopulmonary risk assessment from a single LDCT scan. The framework introduces an agentic reasoning process that emulates clinical diagnostic thinking-first perceiving pulmonary findings, then reasoning through established medical knowledge, and finally deriving a cardiovascular judgment with explanatory rationale. It integrates three synergistic components: a pulmonary perception module that summarizes lung abnormalities, a knowledge-guided reasoning module that infers their cardiovascular implications, and a cardiac representation module that encodes structural biomarkers. Their outputs are fused to produce a holistic cardiovascular risk prediction that is both accurate and physiologically grounded. Experiments on the NLST cohort demonstrate that the proposed framework achieves state-of-the-art performance for CVD screening and mortality prediction, outperforming single-disease and purely image-based baselines. Beyond quantitative gains, the framework provides human-verifiable reasoning that aligns with cardiological understanding, revealing coherent links between pulmonary abnormalities and cardiac stress mechanisms. Overall, this work establishes a unified and explainable paradigm for cardiovascular analysis from LDCT, bridging the gap between image-based prediction and mechanism-based medical interpretation.
>
---
#### [replaced 057] Zero-Shot Referring Expression Comprehension via Vison-Language True/False Verification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.09958v3](https://arxiv.org/pdf/2509.09958v3)**

> **作者:** Jeffrey Liu; Rongbin Hu
>
> **摘要:** Referring Expression Comprehension (REC) is usually addressed with task-trained grounding models. We show that a zero-shot workflow, without any REC-specific training, can achieve competitive or superior performance. Our approach reformulates REC as box-wise visual-language verification: given proposals from a COCO-clean generic detector (YOLO-World), a general-purpose VLM independently answers True/False queries for each region. This simple procedure reduces cross-box interference, supports abstention and multiple matches, and requires no fine-tuning. On RefCOCO, RefCOCO+, and RefCOCOg, our method not only surpasses a zero-shot GroundingDINO baseline but also exceeds reported results for GroundingDINO trained on REC and GroundingDINO+CRG. Controlled studies with identical proposals confirm that verification significantly outperforms selection-based prompting, and results hold with open VLMs. Overall, we show that workflow design, rather than task-specific pretraining, drives strong zero-shot REC performance.
>
---
#### [replaced 058] PAN: A World Model for General, Interactable, and Long-Horizon World Simulation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.09057v2](https://arxiv.org/pdf/2511.09057v2)**

> **作者:** PAN Team; Jiannan Xiang; Yi Gu; Zihan Liu; Zeyu Feng; Qiyue Gao; Yiyan Hu; Benhao Huang; Guangyi Liu; Yichi Yang; Kun Zhou; Davit Abrahamyan; Arif Ahmad; Ganesh Bannur; Junrong Chen; Kimi Chen; Mingkai Deng; Ruobing Han; Xinqi Huang; Haoqiang Kang; Zheqi Li; Enze Ma; Hector Ren; Yashowardhan Shinde; Rohan Shingre; Ramsundar Tanikella; Kaiming Tao; Dequan Yang; Xinle Yu; Cong Zeng; Binglin Zhou; Zhengzhong Liu; Zhiting Hu; Eric P. Xing
>
> **摘要:** A world model enables an intelligent agent to imagine, predict, and reason about how the world evolves in response to its actions, and accordingly to plan and strategize. While recent video generation models produce realistic visual sequences, they typically operate in the prompt-to-full-video manner without causal control, interactivity, or long-horizon consistency required for purposeful reasoning. Existing world modeling efforts, on the other hand, often focus on restricted domains (e.g., physical, game, or 3D-scene dynamics) with limited depth and controllability, and struggle to generalize across diverse environments and interaction formats. In this work, we introduce PAN, a general, interactable, and long-horizon world model that predicts future world states through high-quality video simulation conditioned on history and natural language actions. PAN employs the Generative Latent Prediction (GLP) architecture that combines an autoregressive latent dynamics backbone based on a large language model (LLM), which grounds simulation in extensive text-based knowledge and enables conditioning on language-specified actions, with a video diffusion decoder that reconstructs perceptually detailed and temporally coherent visual observations, to achieve a unification between latent space reasoning (imagination) and realizable world dynamics (reality). Trained on large-scale video-action pairs spanning diverse domains, PAN supports open-domain, action-conditioned simulation with coherent, long-term dynamics. Extensive experiments show that PAN achieves strong performance in action-conditioned world simulation, long-horizon forecasting, and simulative reasoning compared to other video generators and world models, taking a step towards general world models that enable predictive simulation of future world states for reasoning and acting.
>
---
#### [replaced 059] MAUGIF: Mechanism-Aware Unsupervised General Image Fusion via Dual Cross-Image Autoencoders
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08272v3](https://arxiv.org/pdf/2511.08272v3)**

> **作者:** Kunjing Yang; Zhiwei Wang; Minru Bai
>
> **摘要:** Image fusion aims to integrate structural and complementary information from multi-source images. However, existing fusion methods are often either highly task-specific, or general frameworks that apply uniform strategies across diverse tasks, ignoring their distinct fusion mechanisms. To address this issue, we propose a mechanism-aware unsupervised general image fusion (MAUGIF) method based on dual cross-image autoencoders. Initially, we introduce a classification of additive and multiplicative fusion according to the inherent mechanisms of different fusion tasks. Then, dual encoders map source images into a shared latent space, capturing common content while isolating modality-specific details. During the decoding phase, dual decoders act as feature injectors, selectively reintegrating the unique characteristics of each modality into the shared content for reconstruction. The modality-specific features are injected into the source image in the fusion process, generating the fused image that integrates information from both modalities. The architecture of decoders varies according to their fusion mechanisms, enhancing both performance and interpretability. Extensive experiments are conducted on diverse fusion tasks to validate the effectiveness and generalization ability of our method. The code is available at https://anonymous.4open.science/r/MAUGIF.
>
---
#### [replaced 060] LayerPeeler: Autoregressive Peeling for Layer-wise Image Vectorization
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2505.23740v3](https://arxiv.org/pdf/2505.23740v3)**

> **作者:** Ronghuan Wu; Wanchao Su; Jing Liao
>
> **备注:** Project Page: https://layerpeeler.github.io/
>
> **摘要:** Image vectorization is a powerful technique that converts raster images into vector graphics, enabling enhanced flexibility and interactivity. However, popular image vectorization tools struggle with occluded regions, producing incomplete or fragmented shapes that hinder editability. While recent advancements have explored optimization-based and learning-based layer-wise image vectorization, these methods face limitations in vectorization quality and flexibility. In this paper, we introduce LayerPeeler, a novel layer-wise image vectorization approach that addresses these challenges through a progressive simplification paradigm. The key to LayerPeeler's success lies in its autoregressive peeling strategy: by identifying and removing the topmost non-occluded layers while recovering underlying content, we generate vector graphics with complete paths and coherent layer structures. Our method leverages vision-language models to construct a layer graph that captures occlusion relationships among elements, enabling precise detection and description for non-occluded layers. These descriptive captions are used as editing instructions for a finetuned image diffusion model to remove the identified layers. To ensure accurate removal, we employ localized attention control that precisely guides the model to target regions while faithfully preserving the surrounding content. To support this, we contribute a large-scale dataset specifically designed for layer peeling tasks. Extensive quantitative and qualitative experiments demonstrate that LayerPeeler significantly outperforms existing techniques, producing vectorization results with superior path semantics, geometric regularity, and visual fidelity.
>
---
#### [replaced 061] Self-Supervised Training For Low Dose CT Reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2010.13232v3](https://arxiv.org/pdf/2010.13232v3)**

> **作者:** Mehmet Ozan Unal; Metin Ertas; Isa Yildirim
>
> **摘要:** Ionizing radiation has been the biggest concern in CT imaging. To reduce the dose level without compromising the image quality, low-dose CT reconstruction has been offered with the availability of compressed sensing based reconstruction methods. Recently, data-driven methods got attention with the rise of deep learning, the availability of high computational power, and big datasets. Deep learning based methods have also been used in low-dose CT reconstruction problem in different manners. Usually, the success of these methods depends on labeled data. However, recent studies showed that training can be achieved successfully with noisy datasets. In this study, we defined a training scheme to use low-dose sinograms as their own training targets. We applied the self-supervision principle in the projection domain where the noise is element-wise independent which is a requirement for self-supervised training methods. Using the self-supervised training, the filtering part of the FBP method and the parameters of a denoiser neural network are optimized. We demonstrate that our method outperforms both conventional and compressed sensing based iterative reconstruction methods qualitatively and quantitatively in the reconstruction of analytic CT phantoms and real-world CT images in low-dose CT reconstruction task.
>
---
#### [replaced 062] Caption This, Reason That: VLMs Caught in the Middle
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.21538v2](https://arxiv.org/pdf/2505.21538v2)**

> **作者:** Zihan Weng; Lucas Gomez; Taylor Whittington Webb; Pouya Bashivan
>
> **备注:** Paper accepted by nips 2025
>
> **摘要:** Vision-Language Models (VLMs) have shown remarkable progress in visual understanding in recent years. Yet, they still lag behind human capabilities in specific visual tasks such as counting or relational reasoning. To understand the underlying limitations, we adopt methodologies from cognitive science, analyzing VLM performance along core cognitive axes: Perception, Attention, and Memory. Using a suite of tasks targeting these abilities, we evaluate state-of-the-art VLMs, including GPT-4o. Our analysis reveals distinct cognitive profiles: while advanced models approach ceiling performance on some tasks (e.g. category identification), a significant gap persists, particularly in tasks requiring spatial understanding or selective attention. Investigating the source of these failures and potential methods for improvement, we employ a vision-text decoupling analysis, finding that models struggling with direct visual reasoning show marked improvement when reasoning over their own generated text captions. These experiments reveal a strong need for improved VLM Chain-of-Thought (CoT) abilities, even in models that consistently exceed human performance. Furthermore, we demonstrate the potential of targeted fine-tuning on composite visual reasoning tasks and show that fine-tuning smaller VLMs substantially improves core cognitive abilities. While this improvement does not translate to large enhancements on challenging, out-of-distribution benchmarks, we show broadly that VLM performance on our datasets strongly correlates with performance on these other benchmarks. Our work provides a detailed analysis of VLM cognitive strengths and weaknesses and identifies key bottlenecks in simultaneous perception and reasoning while also providing an effective and simple solution.
>
---
#### [replaced 063] Towards Consistent and Efficient Dataset Distillation via Diffusion-Driven Selection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.09959v4](https://arxiv.org/pdf/2412.09959v4)**

> **作者:** Xinhao Zhong; Shuoyang Sun; Xulin Gu; Zhaoyang Xu; Yaowei Wang; Min Zhang; Bin Chen
>
> **摘要:** Dataset distillation provides an effective approach to reduce memory and computational costs by optimizing a compact dataset that achieves performance comparable to the full original. However, for large-scale datasets and complex deep networks (e.g., ImageNet-1K with ResNet-101), the vast optimization space hinders distillation effectiveness, limiting practical applications. Recent methods leverage pre-trained diffusion models to directly generate informative images, thereby bypassing pixel-level optimization and achieving promising results. Nonetheless, these approaches often suffer from distribution shifts between the pre-trained diffusion prior and target datasets, as well as the need for multiple distillation steps under varying settings. To overcome these challenges, we propose a novel framework that is orthogonal to existing diffusion-based distillation techniques by utilizing the diffusion prior for patch selection rather than generation. Our method predicts noise from the diffusion model conditioned on input images and optional text prompts (with or without label information), and computes the associated loss for each image-patch pair. Based on the loss differences, we identify distinctive regions within the original images. Furthermore, we apply intra-class clustering and ranking on the selected patches to enforce diversity constraints. This streamlined pipeline enables a one-step distillation process. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art methods across various metrics and settings.
>
---
#### [replaced 064] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.09147v2](https://arxiv.org/pdf/2511.09147v2)**

> **作者:** Jiayue Yuan; Fangting Xie; Guangwen Ouyang; Changhai Ma; Ziyu Wu; Heyu Ding; Quan Wan; Yi Ke; Yuchen Wu; Xiaohui Cai
>
> **备注:** Accepted by AAAI-2026
>
> **摘要:** Multi-person global human mesh recovery (HMR) is crucial for understanding crowd dynamics and interactions. Traditional vision-based HMR methods sometimes face limitations in real-world scenarios due to mutual occlusions, insufficient lighting, and privacy concerns. Human-floor tactile interactions offer an occlusion-free and privacy-friendly alternative for capturing human motion. Existing research indicates that pressure signals acquired from tactile mats can effectively estimate human pose in single-person scenarios. However, when multiple individuals walk randomly on the mat simultaneously, how to distinguish intermingled pressure signals generated by different persons and subsequently acquire individual temporal pressure data remains a pending challenge for extending pressure-based HMR to the multi-person situation. In this paper, we present \textbf{PressTrack-HMR}, a top-down pipeline that recovers multi-person global human meshes solely from pressure signals. This pipeline leverages a tracking-by-detection strategy to first identify and segment each individual's pressure signal from the raw pressure data, and subsequently performs HMR for each extracted individual signal. Furthermore, we build a multi-person interaction pressure dataset \textbf{MIP}, which facilitates further research into pressure-based human motion analysis in multi-person scenarios. Experimental results demonstrate that our method excels in multi-person HMR using pressure data, with 89.2 $mm$ MPJPE and 112.6 $mm$ WA-MPJPE$_{100}$, and these showcase the potential of tactile mats for ubiquitous, privacy-preserving multi-person action recognition. Our dataset & code are available at https://github.com/Jiayue-Yuan/PressTrack-HMR.
>
---
#### [replaced 065] ManipDreamer3D : Synthesizing Plausible Robotic Manipulation Video with Occupancy-aware 3D Trajectory
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.05314v2](https://arxiv.org/pdf/2509.05314v2)**

> **作者:** Ying Li; Xiaobao Wei; Xiaowei Chi; Yuming Li; Zhongyu Zhao; Hao Wang; Ningning Ma; Ming Lu; Sirui Han; Shanghang Zhang
>
> **备注:** 7pages; 7figures; 3 tables
>
> **摘要:** Data scarcity continues to be a major challenge in the field of robotic manipulation. Although diffusion models provide a promising solution for generating robotic manipulation videos, existing methods largely depend on 2D trajectories, which inherently face issues with 3D spatial ambiguity. In this work, we present a novel framework named ManipDreamer3D for generating plausible 3D-aware robotic manipulation videos from the input image and the text instruction. Our method combines 3D trajectory planning with a reconstructed 3D occupancy map created from a third-person perspective, along with a novel trajectory-to-video diffusion model. Specifically, ManipDreamer3D first reconstructs the 3D occupancy representation from the input image and then computes an optimized 3D end-effector trajectory, minimizing path length while avoiding collisions. Next, we employ a latent editing technique to create video sequences from the initial image latent and the optimized 3D trajectory. This process conditions our specially trained trajectory-to-video diffusion model to produce robotic pick-and-place videos. Our method generates robotic videos with autonomously planned plausible 3D trajectories, significantly reducing human intervention requirements. Experimental results demonstrate superior visual quality compared to existing methods.
>
---
#### [replaced 066] vMFCoOp: Towards Equilibrium on a Unified Hyperspherical Manifold for Prompting Biomedical VLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09540v2](https://arxiv.org/pdf/2511.09540v2)**

> **作者:** Minye Shao; Sihan Guo; Xinrun Li; Xingyu Miao; Haoran Duan; Yang Long
>
> **备注:** Accepted as an Oral Presentation at AAAI 2026 Main Technical Track (this version is not peer-reviewed; it is the extended version)
>
> **摘要:** Recent advances in context optimization (CoOp) guided by large language model (LLM)-distilled medical semantic priors offer a scalable alternative to manual prompt engineering and full fine-tuning for adapting biomedical CLIP-based vision-language models (VLMs). However, prompt learning in this context is challenged by semantic misalignment between LLMs and CLIP variants due to divergent training corpora and model architectures; it further lacks scalability across continuously evolving families of foundation models. More critically, pairwise multimodal alignment via conventional Euclidean-space optimization lacks the capacity to model unified representations or apply localized geometric constraints, which tends to amplify modality gaps in complex biomedical imaging and destabilize few-shot adaptation. In this work, we propose vMFCoOp, a framework that inversely estimates von Mises-Fisher (vMF) distributions on a shared Hyperspherical Manifold, aligning semantic biases between arbitrary LLMs and CLIP backbones via Unified Semantic Anchors to achieve robust biomedical prompting and superior few-shot classification. Grounded in three complementary constraints, vMFCoOp demonstrates consistent improvements across 14 medical datasets, 12 medical imaging modalities, and 13 anatomical regions, outperforming state-of-the-art methods in accuracy, generalization, and clinical applicability. This work aims to continuously expand to encompass more downstream applications, and the corresponding resources are intended to be shared through https://github.com/VinyehShaw/UniEqui.
>
---
#### [replaced 067] Enhanced Structured Lasso Pruning with Class-wise Information
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.09125v2](https://arxiv.org/pdf/2502.09125v2)**

> **作者:** Xiang Liu; Mingchen Li; Xia Li; Leigang Qu; Guangsu Wang; Zifan Peng; Yijun Song; Zemin Liu; Linshan Jiang; Jialin Li
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Modern applications require lightweight neural network models. Most existing neural network pruning methods focus on removing unimportant filters; however, these may result in the loss of statistical information after pruning due to failing to consider the class-wise information. In this paper, we employ the structured lasso from the perspective of utilizing precise class-wise information for model pruning with the help of Information Bottleneck theory, which guides us to ensure the retention of statistical information before and after pruning. With these techniques, we propose two novel adaptive network pruning schemes in parallel: sparse graph-structured lasso pruning with Information Bottleneck (sGLP-IB) and sparse tree-guided lasso pruning with Information Bottleneck (sTLP-IB). The key component is that we prune the model filters utilizing sGLP-IB and sTLP-IB with more precise structured class-wise relatedness. Compared to multiple state-of-the-art methods, our approaches achieve the best performance across three datasets and six model structures on extensive experiments. For example, with the VGG16 model based on the CIFAR-10 dataset, we can reduce the parameters by 85%, decrease the FLOPs by 61%, and maintain an accuracy of 94.10% (0.14% better than the original). For large-scale ImageNet, we can reduce the parameters by 55% while keeping the accuracy at 76.12% (only drop 0.03%) using the ResNet architecture. In summary, we succeed in reducing the model size and computational resource usage while maintaining the effectiveness of accuracy.
>
---
#### [replaced 068] Remodeling Semantic Relationships in Vision-Language Fine-Tuning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.08238v2](https://arxiv.org/pdf/2511.08238v2)**

> **作者:** Xiangyang Wu; Liu Liu; Baosheng Yu; Jiayan Qiu; Zhenwei Shi
>
> **摘要:** Vision-language fine-tuning has emerged as an efficient paradigm for constructing multimodal foundation models. While textual context often highlights semantic relationships within an image, existing fine-tuning methods typically overlook this information when aligning vision and language, thus leading to suboptimal performance. Toward solving this problem, we propose a method that can improve multimodal alignment and fusion based on both semantics and relationships.Specifically, we first extract multilevel semantic features from different vision encoder to capture more visual cues of the relationships. Then, we learn to project the vision features to group related semantics, among which are more likely to have relationships. Finally, we fuse the visual features with the textual by using inheritable cross-attention, where we globally remove the redundant visual relationships by discarding visual-language feature pairs with low correlation. We evaluate our proposed method on eight foundation models and two downstream tasks, visual question answering and image captioning, and show that it outperforms all existing methods.
>
---
