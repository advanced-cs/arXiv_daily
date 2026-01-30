# 计算机视觉 cs.CV

- **最新发布 115 篇**

- **更新 79 篇**

## 最新发布

#### [new 001] Multimodal Visual Surrogate Compression for Alzheimer's Disease Classification
- **分类: cs.CV**

- **简介: 该论文属于阿尔茨海默病分类任务，旨在解决sMRI特征提取效率与效果的问题。提出MVSC方法，将3D脑图像压缩为2D特征，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.21673v1](https://arxiv.org/pdf/2601.21673v1)**

> **作者:** Dexuan Ding; Ciyuan Peng; Endrowednes Kuantama; Jingcai Guo; Jia Wu; Jian Yang; Amin Beheshti; Ming-Hsuan Yang; Yuankai Qi
>
> **摘要:** High-dimensional structural MRI (sMRI) images are widely used for Alzheimer's Disease (AD) diagnosis. Most existing methods for sMRI representation learning rely on 3D architectures (e.g., 3D CNNs), slice-wise feature extraction with late aggregation, or apply training-free feature extractions using 2D foundation models (e.g., DINO). However, these three paradigms suffer from high computational cost, loss of cross-slice relations, and limited ability to extract discriminative features, respectively. To address these challenges, we propose Multimodal Visual Surrogate Compression (MVSC). It learns to compress and adapt large 3D sMRI volumes into compact 2D features, termed as visual surrogates, which are better aligned with frozen 2D foundation models to extract powerful representations for final AD classification. MVSC has two key components: a Volume Context Encoder that captures global cross-slice context under textual guidance, and an Adaptive Slice Fusion module that aggregates slice-level information in a text-enhanced, patch-wise manner. Extensive experiments on three large-scale Alzheimer's disease benchmarks demonstrate our MVSC performs favourably on both binary and multi-class classification tasks compared against state-of-the-art methods.
>
---
#### [new 002] Gaussian Belief Propagation Network for Depth Completion
- **分类: cs.CV**

- **简介: 该论文属于深度补全任务，旨在从稀疏深度数据中预测稠密深度图。提出GBPN框架，结合深度学习与概率图模型，有效处理稀疏数据，提升性能。**

- **链接: [https://arxiv.org/pdf/2601.21291v1](https://arxiv.org/pdf/2601.21291v1)**

> **作者:** Jie Tang; Pingping Xie; Jian Li; Ping Tan
>
> **摘要:** Depth completion aims to predict a dense depth map from a color image with sparse depth measurements. Although deep learning methods have achieved state-of-the-art (SOTA), effectively handling the sparse and irregular nature of input depth data in deep networks remains a significant challenge, often limiting performance, especially under high sparsity. To overcome this limitation, we introduce the Gaussian Belief Propagation Network (GBPN), a novel hybrid framework synergistically integrating deep learning with probabilistic graphical models for end-to-end depth completion. Specifically, a scene-specific Markov Random Field (MRF) is dynamically constructed by the Graphical Model Construction Network (GMCN), and then inferred via Gaussian Belief Propagation (GBP) to yield the dense depth distribution. Crucially, the GMCN learns to construct not only the data-dependent potentials of MRF but also its structure by predicting adaptive non-local edges, enabling the capture of complex, long-range spatial dependencies. Furthermore, we enhance GBP with a serial \& parallel message passing scheme, designed for effective information propagation, particularly from sparse measurements. Extensive experiments demonstrate that GBPN achieves SOTA performance on the NYUv2 and KITTI benchmarks. Evaluations across varying sparsity levels, sparsity patterns, and datasets highlight GBPN's superior performance, notable robustness, and generalizable capability.
>
---
#### [new 003] Spava: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决长视频推理效率低的问题。通过提出Spava框架，实现多GPU并行加速，提升处理能力且不损失性能。**

- **链接: [https://arxiv.org/pdf/2601.21444v1](https://arxiv.org/pdf/2601.21444v1)**

> **作者:** Yuxiang Huang; Mingye Li; Xu Han; Chaojun Xiao; Weilin Zhao; Ao Sun; Ziqi Yuan; Hao Zhou; Fandong Meng; Zhiyuan Liu
>
> **备注:** Preprint
>
> **摘要:** The efficiency of long-video inference remains a critical bottleneck, mainly due to the dense computation in the prefill stage of Large Multimodal Models (LMMs). Existing methods either compress visual embeddings or apply sparse attention on a single GPU, yielding limited acceleration or degraded performance and restricting LMMs from handling longer, more complex videos. To overcome these issues, we propose Spava, a sequence-parallel framework with optimized attention that accelerates long-video inference across multiple GPUs. By distributing approximate attention, Spava reduces computation and increases parallelism, enabling efficient processing of more visual embeddings without compression and thereby improving task performance. System-level optimizations, such as load balancing and fused forward passes, further unleash the potential of Spava, delivering speedups of 12.72x, 1.70x, and 1.18x over FlashAttn, ZigZagRing, and APB, without notable performance loss. Code available at https://github.com/thunlp/APB
>
---
#### [new 004] Urban Neural Surface Reconstruction from Constrained Sparse Aerial Imagery with 3D SAR Fusion
- **分类: cs.CV**

- **简介: 该论文属于城市三维重建任务，解决稀疏航拍影像下的几何模糊问题，通过融合3D SAR点云提升重建精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.22045v1](https://arxiv.org/pdf/2601.22045v1)**

> **作者:** Da Li; Chen Yao; Tong Mao; Jiacheng Bao; Houjun Sun
>
> **摘要:** Neural surface reconstruction (NSR) has recently shown strong potential for urban 3D reconstruction from multi-view aerial imagery. However, existing NSR methods often suffer from geometric ambiguity and instability, particularly under sparse-view conditions. This issue is critical in large-scale urban remote sensing, where aerial image acquisition is limited by flight paths, terrain, and cost. To address this challenge, we present the first urban NSR framework that fuses 3D synthetic aperture radar (SAR) point clouds with aerial imagery for high-fidelity reconstruction under constrained, sparse-view settings. 3D SAR can efficiently capture large-scale geometry even from a single side-looking flight path, providing robust priors that complement photometric cues from images. Our framework integrates radar-derived spatial constraints into an SDF-based NSR backbone, guiding structure-aware ray selection and adaptive sampling for stable and efficient optimization. We also construct the first benchmark dataset with co-registered 3D SAR point clouds and aerial imagery, facilitating systematic evaluation of cross-modal 3D reconstruction. Extensive experiments show that incorporating 3D SAR markedly enhances reconstruction accuracy, completeness, and robustness compared with single-modality baselines under highly sparse and oblique-view conditions, highlighting a viable route toward scalable high-fidelity urban reconstruction with advanced airborne and spaceborne optical-SAR sensing.
>
---
#### [new 005] Semantic-Guided Dynamic Sparsification for Pre-Trained Model-based Class-Incremental Learning
- **分类: cs.CV**

- **简介: 该论文属于类增量学习任务，旨在解决模型在学习新类时遗忘旧类的问题。通过引入语义引导的动态稀疏化方法，有效减少任务间干扰，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.21345v1](https://arxiv.org/pdf/2601.21345v1)**

> **作者:** Ruiqi Liu; Boyu Diao; Zijia An; Runjie Shao; Zhulin An; Fei Wang; Yongjun Xu
>
> **摘要:** Class-Incremental Learning (CIL) requires a model to continually learn new classes without forgetting old ones. A common and efficient solution freezes a pre-trained model and employs lightweight adapters, whose parameters are often forced to be orthogonal to prevent inter-task interference. However, we argue that this parameter-constraining method is detrimental to plasticity. To this end, we propose Semantic-Guided Dynamic Sparsification (SGDS), a novel method that proactively guides the activation space by governing the orientation and rank of its subspaces through targeted sparsification. Specifically, SGDS promotes knowledge transfer by encouraging similar classes to share a compact activation subspace, while simultaneously preventing interference by assigning non-overlapping activation subspaces to dissimilar classes. By sculpting class-specific sparse subspaces in the activation space, SGDS effectively mitigates interference without imposing rigid constraints on the parameter space. Extensive experiments on various benchmark datasets demonstrate the state-of-the-art performance of SGDS.
>
---
#### [new 006] BookNet: Book Image Rectification via Cross-Page Attention Network
- **分类: cs.CV**

- **简介: 该论文属于文档图像校正任务，解决书籍双页图像因装订导致的几何失真问题。提出BookNet模型，通过跨页注意力机制建模左右页相互影响，提升校正效果。**

- **链接: [https://arxiv.org/pdf/2601.21938v1](https://arxiv.org/pdf/2601.21938v1)**

> **作者:** Shaokai Liu; Hao Feng; Bozhi Luan; Min Hou; Jiajun Deng; Wengang Zhou
>
> **摘要:** Book image rectification presents unique challenges in document image processing due to complex geometric distortions from binding constraints, where left and right pages exhibit distinctly asymmetric curvature patterns. However, existing single-page document image rectification methods fail to capture the coupled geometric relationships between adjacent pages in books. In this work, we introduce BookNet, the first end-to-end deep learning framework specifically designed for dual-page book image rectification. BookNet adopts a dual-branch architecture with cross-page attention mechanisms, enabling it to estimate warping flows for both individual pages and the complete book spread, explicitly modeling how left and right pages influence each other. Moreover, to address the absence of specialized datasets, we present Book3D, a large-scale synthetic dataset for training, and Book100, a comprehensive real-world benchmark for evaluation. Extensive experiments demonstrate that BookNet outperforms existing state-of-the-art methods on book image rectification. Code and dataset will be made publicly available.
>
---
#### [new 007] MultiModal Fine-tuning with Synthetic Captions
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决预训练与微调阶段多模态能力不匹配的问题。通过生成合成文本描述，将单模态数据转为多模态，提升微调效果。**

- **链接: [https://arxiv.org/pdf/2601.21426v1](https://arxiv.org/pdf/2601.21426v1)**

> **作者:** Shohei Enomoto; Shin'ya Yamaguchi
>
> **摘要:** In this paper, we address a fundamental gap between pre-training and fine-tuning of deep neural networks: while pre-training has shifted from unimodal to multimodal learning with enhanced visual understanding, fine-tuning predominantly remains unimodal, limiting the benefits of rich pre-trained representations. To bridge this gap, we propose a novel approach that transforms unimodal datasets into multimodal ones using Multimodal Large Language Models (MLLMs) to generate synthetic image captions for fine-tuning models with a multimodal objective. Our method employs carefully designed prompts incorporating class labels and domain context to produce high-quality captions tailored for classification tasks. Furthermore, we introduce a supervised contrastive loss function that explicitly encourages clustering of same-class representations during fine-tuning, along with a new inference technique that leverages class-averaged text embeddings from multiple synthetic captions per image. Extensive experiments across 13 image classification benchmarks demonstrate that our approach outperforms baseline methods, with particularly significant improvements in few-shot learning scenarios. Our work establishes a new paradigm for dataset enhancement that effectively bridges the gap between multimodal pre-training and fine-tuning. Our code is available at https://github.com/s-enmt/MMFT.
>
---
#### [new 008] Vision KAN: Towards an Attention-Free Backbone for Vision with Kolmogorov-Arnold Networks
- **分类: cs.CV**

- **简介: 该论文提出ViK，一种无需注意力机制的视觉骨干网络，解决注意力机制复杂度高、难以解释的问题。通过Kolmogorov-Arnold网络实现高效token混合。**

- **链接: [https://arxiv.org/pdf/2601.21541v1](https://arxiv.org/pdf/2601.21541v1)**

> **作者:** Zhuoqin Yang; Jiansong Zhang; Xiaoling Luo; Xu Wu; Zheng Lu; Linlin Shen
>
> **摘要:** Attention mechanisms have become a key module in modern vision backbones due to their ability to model long-range dependencies. However, their quadratic complexity in sequence length and the difficulty of interpreting attention weights limit both scalability and clarity. Recent attention-free architectures demonstrate that strong performance can be achieved without pairwise attention, motivating the search for alternatives. In this work, we introduce Vision KAN (ViK), an attention-free backbone inspired by the Kolmogorov-Arnold Networks. At its core lies MultiPatch-RBFKAN, a unified token mixer that combines (a) patch-wise nonlinear transform with Radial Basis Function-based KANs, (b) axis-wise separable mixing for efficient local propagation, and (c) low-rank global mapping for long-range interaction. Employing as a drop-in replacement for attention modules, this formulation tackles the prohibitive cost of full KANs on high-resolution features by adopting a patch-wise grouping strategy with lightweight operators to restore cross-patch dependencies. Experiments on ImageNet-1K show that ViK achieves competitive accuracy with linear complexity, demonstrating the potential of KAN-based token mixing as an efficient and theoretically grounded alternative to attention.
>
---
#### [new 009] TraceRouter: Robust Safety for Large Foundation Models via Path-Level Intervention
- **分类: cs.CV; cs.AI; cs.CY; cs.MM**

- **简介: 该论文属于模型安全任务，旨在解决大模型易受攻击的问题。提出TraceRouter框架，通过路径级干预切断有害信息传播，提升安全性与通用性。**

- **链接: [https://arxiv.org/pdf/2601.21900v1](https://arxiv.org/pdf/2601.21900v1)**

> **作者:** Chuancheng Shi; Shangze Li; Wenjun Lu; Wenhua Wu; Cong Wang; Zifeng Cheng; Fei Shen; Tat-Seng Chua
>
> **摘要:** Despite their capabilities, large foundation models (LFMs) remain susceptible to adversarial manipulation. Current defenses predominantly rely on the "locality hypothesis", suppressing isolated neurons or features. However, harmful semantics act as distributed, cross-layer circuits, rendering such localized interventions brittle and detrimental to utility. To bridge this gap, we propose \textbf{TraceRouter}, a path-level framework that traces and disconnects the causal propagation circuits of illicit semantics. TraceRouter operates in three stages: (1) it pinpoints a sensitive onset layer by analyzing attention divergence; (2) it leverages sparse autoencoders (SAEs) and differential activation analysis to disentangle and isolate malicious features; and (3) it maps these features to downstream causal pathways via feature influence scores (FIS) derived from zero-out interventions. By selectively suppressing these causal chains, TraceRouter physically severs the flow of harmful information while leaving orthogonal computation routes intact. Extensive experiments demonstrate that TraceRouter significantly outperforms state-of-the-art baselines, achieving a superior trade-off between adversarial robustness and general utility. Our code will be publicly released. WARNING: This paper contains unsafe model responses.
>
---
#### [new 010] AI-based Prediction of Biochemical Recurrence from Biopsy and Prostatectomy Samples
- **分类: cs.CV**

- **简介: 该论文属于医学预测任务，旨在解决前列腺癌术后生化复发的精准预测问题。通过AI分析活检样本，建立预测模型，并验证其在多个队列中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.21022v1](https://arxiv.org/pdf/2601.21022v1)**

> **作者:** Andrea Camilloni; Chiara Micoli; Nita Mulliqi; Erik Everett Palm; Thorgerdur Palsdottir; Kelvin Szolnoky; Xiaoyi Ji; Sol Erika Boman; Andrea Discacciati; Henrik Grönberg; Lars Egevad; Tobias Nordström; Kimmo Kartasalo; Martin Eklund
>
> **备注:** 39 pages, 6 tables, 11 figures
>
> **摘要:** Biochemical recurrence (BCR) after radical prostatectomy (RP) is a surrogate marker for aggressive prostate cancer with adverse outcomes, yet current prognostic tools remain imprecise. We trained an AI-based model on diagnostic prostate biopsy slides from the STHLM3 cohort (n = 676) to predict patient-specific risk of BCR, using foundation models and attention-based multiple instance learning. Generalizability was assessed across three external RP cohorts: LEOPARD (n = 508), CHIMERA (n = 95), and TCGA-PRAD (n = 379). The image-based approach achieved 5-year time-dependent AUCs of 0.64, 0.70, and 0.70, respectively. Integrating clinical variables added complementary prognostic value and enabled statistically significant risk stratification. Compared with guideline-based CAPRA-S, AI incrementally improved postoperative prognostication. These findings suggest biopsy-trained histopathology AI can generalize across specimen types to support preoperative and postoperative decision making, but the added value of AI-based multimodal approaches over simpler predictive models should be critically scrutinized in further studies.
>
---
#### [new 011] Hypersolid: Emergent Vision Representations via Short-Range Repulsion
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于自监督学习任务，旨在解决表示崩溃问题。通过短距离排斥机制，提升特征多样性，增强细粒度和低分辨率分类性能。**

- **链接: [https://arxiv.org/pdf/2601.21255v1](https://arxiv.org/pdf/2601.21255v1)**

> **作者:** Esteban Rodríguez-Betancourt; Edgar Casasola-Murillo
>
> **备注:** 17 pages, 16 figures
>
> **摘要:** A recurring challenge in self-supervised learning is preventing representation collapse. Existing solutions typically rely on global regularization, such as maximizing distances, decorrelating dimensions or enforcing certain distributions. We instead reinterpret representation learning as a discrete packing problem, where preserving information simplifies to maintaining injectivity. We operationalize this in Hypersolid, a method using short-range hard-ball repulsion to prevent local collisions. This constraint results in a high-separation geometric regime that preserves augmentation diversity, excelling on fine-grained and low-resolution classification tasks.
>
---
#### [new 012] Towards Mitigating Modality Bias in Vision-Language Models for Temporal Action Localization
- **分类: cs.CV**

- **简介: 该论文属于时序动作定位任务，旨在解决视觉-语言模型中的模态偏差问题。通过引入去偏重加权和残差聚合策略，提升模型的视觉主导性和时间推理能力。**

- **链接: [https://arxiv.org/pdf/2601.21078v1](https://arxiv.org/pdf/2601.21078v1)**

> **作者:** Jiaqi Li; Guangming Wang; Shuntian Zheng; Minzhe Ni; Xiaoman Lu; Guanghui Ye; Yu Guan
>
> **摘要:** Temporal Action Localization (TAL) requires identifying both the boundaries and categories of actions in untrimmed videos. While vision-language models (VLMs) offer rich semantics to complement visual evidence, existing approaches tend to overemphasize linguistic priors at the expense of visual performance, leading to a pronounced modality bias. We propose ActionVLM, a vision-language aggregation framework that systematically mitigates modality bias in TAL. Our key insight is to preserve vision as the dominant signal while adaptively exploiting language only when beneficial. To this end, we introduce (i) a debiasing reweighting module that estimates the language advantage-the incremental benefit of language over vision-only predictions-and dynamically reweights language modality accordingly, and (ii) a residual aggregation strategy that treats language as a complementary refinement rather than the primary driver. This combination alleviates modality bias, reduces overconfidence from linguistic priors, and strengthens temporal reasoning. Experiments on THUMOS14 show that our model outperforms state-of-the-art by up to 3.2% mAP.
>
---
#### [new 013] VideoAesBench: Benchmarking the Video Aesthetics Perception Capabilities of Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于视频美学评估任务，旨在解决LMMs在视频美学感知能力上的不足。构建了VideoAesBench基准，涵盖多种视频类型和问题形式，评估23个模型，发现其能力仍不完善。**

- **链接: [https://arxiv.org/pdf/2601.21915v1](https://arxiv.org/pdf/2601.21915v1)**

> **作者:** Yunhao Li; Sijing Wu; Zhilin Gao; Zicheng Zhang; Qi Jia; Huiyu Duan; Xiongkuo Min; Guangtao Zhai
>
> **摘要:** Large multimodal models (LMMs) have demonstrated outstanding capabilities in various visual perception tasks, which has in turn made the evaluation of LMMs significant. However, the capability of video aesthetic quality assessment, which is a fundamental ability for human, remains underexplored for LMMs. To address this, we introduce VideoAesBench, a comprehensive benchmark for evaluating LMMs' understanding of video aesthetic quality. VideoAesBench has several significant characteristics: (1) Diverse content including 1,804 videos from multiple video sources including user-generated (UGC), AI-generated (AIGC), compressed, robotic-generated (RGC), and game videos. (2) Multiple question formats containing traditional single-choice questions, multi-choice questions, True or False questions, and a novel open-ended questions for video aesthetics description. (3) Holistic video aesthetics dimensions including visual form related questions from 5 aspects, visual style related questions from 4 aspects, and visual affectiveness questions from 3 aspects. Based on VideoAesBench, we benchmark 23 open-source and commercial large multimodal models. Our findings show that current LMMs only contain basic video aesthetics perception ability, their performance remains incomplete and imprecise. We hope our VideoAesBench can be served as a strong testbed and offer insights for explainable video aesthetics assessment.
>
---
#### [new 014] Do Pathology Foundation Models Encode Disease Progression? A Pseudotime Analysis of Visual Representations
- **分类: cs.CV**

- **简介: 该论文属于计算机病理学任务，旨在验证视觉基础模型是否能编码疾病进展过程。通过伪时间分析，发现模型能有效捕捉疾病状态的连续性，提升对疾病过渡特征的定量分析能力。**

- **链接: [https://arxiv.org/pdf/2601.21334v1](https://arxiv.org/pdf/2601.21334v1)**

> **作者:** Pritika Vig; Ren-Chin Wu; William Lotter
>
> **备注:** 21 pages, 17 figures. Appendix included
>
> **摘要:** Vision foundation models trained on discretely sampled images achieve strong performance on classification benchmarks, yet whether their representations encode the continuous processes underlying their training data remains unclear. This question is especially pertinent in computational pathology, where we posit that models whose latent representations implicitly capture continuous disease progression may better reflect underlying biology, support more robust generalization, and enable quantitative analyses of features associated with disease transitions. Using diffusion pseudotime, a method developed to infer developmental trajectories from single-cell transcriptomics, we probe whether foundation models organize disease states along coherent progression directions in representation space. Across four cancer progressions and six models, we find that all pathology-specific models recover trajectory orderings significantly exceeding null baselines, with vision-only models achieving the highest fidelities $(τ> 0.78$ on CRC-Serrated). Model rankings by trajectory fidelity on reference diseases strongly predict few-shot classification performance on held-out diseases ($ρ= 0.92$), and exploratory analysis shows cell-type composition varies smoothly along inferred trajectories in patterns consistent with known stromal remodeling. Together, these results demonstrate that vision foundation models can implicitly learn to represent continuous processes from independent static observations, and that trajectory fidelity provides a complementary measure of representation quality beyond downstream performance. While demonstrated in pathology, this framework could be applied to other domains where continuous processes are observed through static snapshots.
>
---
#### [new 015] Trajectory-Guided Diffusion for Foreground-Preserving Background Generation in Multi-Layer Documents
- **分类: cs.CV**

- **简介: 该论文属于多层文档背景生成任务，解决前景保留和跨页风格一致性问题。通过设计潜在空间轨迹，实现自然避让前景并保持风格稳定。**

- **链接: [https://arxiv.org/pdf/2601.21857v1](https://arxiv.org/pdf/2601.21857v1)**

> **作者:** Taewon Kang
>
> **备注:** 47 pages, 36 figures
>
> **摘要:** We present a diffusion-based framework for document-centric background generation that achieves foreground preservation and multi-page stylistic consistency through latent-space design rather than explicit constraints. Instead of suppressing diffusion updates or applying masking heuristics, our approach reinterprets diffusion as the evolution of stochastic trajectories through a structured latent space. By shaping the initial noise and its geometric alignment, background generation naturally avoids designated foreground regions, allowing readable content to remain intact without auxiliary mechanisms. To address the long-standing issue of stylistic drift across pages, we decouple style control from text conditioning and introduce cached style directions as persistent vectors in latent space. Once selected, these directions constrain diffusion trajectories to a shared stylistic subspace, ensuring consistent appearance across pages and editing iterations. This formulation eliminates the need for repeated prompt-based style specification and provides a more stable foundation for multi-page generation. Our framework admits a geometric and physical interpretation, where diffusion paths evolve on a latent manifold shaped by preferred directions, and foreground regions are rarely traversed as a consequence of trajectory initialization rather than explicit exclusion. The proposed method is training-free, compatible with existing diffusion backbones, and produces visually coherent, foreground-preserving results across complex documents. By reframing diffusion as trajectory design in latent space, we offer a principled approach to consistent and structured generative modeling.
>
---
#### [new 016] BadDet+: Robust Backdoor Attacks for Object Detection
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于目标检测任务，旨在解决后门攻击在目标检测中的脆弱性问题。提出BadDet+框架，提升攻击的物理鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.21066v1](https://arxiv.org/pdf/2601.21066v1)**

> **作者:** Kealan Dunnett; Reza Arablouei; Dimity Miller; Volkan Dedeoglu; Raja Jurdak
>
> **摘要:** Backdoor attacks pose a severe threat to deep learning, yet their impact on object detection remains poorly understood compared to image classification. While attacks have been proposed, we identify critical weaknesses in existing detection-based methods, specifically their reliance on unrealistic assumptions and a lack of physical validation. To bridge this gap, we introduce BadDet+, a penalty-based framework that unifies Region Misclassification Attacks (RMA) and Object Disappearance Attacks (ODA). The core mechanism utilizes a log-barrier penalty to suppress true-class predictions for triggered inputs, resulting in (i) position and scale invariance, and (ii) enhanced physical robustness. On real-world benchmarks, BadDet+ achieves superior synthetic-to-physical transfer compared to existing RMA and ODA baselines while preserving clean performance. Theoretical analysis confirms the proposed penalty acts within a trigger-specific feature subspace, reliably inducing attacks without degrading standard inference. These results highlight significant vulnerabilities in object detection and the necessity for specialized defenses.
>
---
#### [new 017] ChartE$^{3}$: A Comprehensive Benchmark for End-to-End Chart Editing
- **分类: cs.CV**

- **简介: 该论文提出ChartE$^{3}$基准，用于评估端到端图表编辑能力。解决图表编辑中局部与全局修改的挑战，通过高质量数据集提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.21694v1](https://arxiv.org/pdf/2601.21694v1)**

> **作者:** Shuo Li; Jiajun Sun; Zhekai Wang; Xiaoran Fan; Hui Li; Dingwen Yang; Zhiheng Xi; Yijun Wang; Zifei Shan; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** Our benchmark will be publicly available at https://github.com/galactic123/ChartE3
>
> **摘要:** Charts are a fundamental visualization format for structured data analysis. Enabling end-to-end chart editing according to user intent is of great practical value, yet remains challenging due to the need for both fine-grained control and global structural consistency. Most existing approaches adopt pipeline-based designs, where natural language or code serves as an intermediate representation, limiting their ability to faithfully execute complex edits. We introduce ChartE$^{3}$, an End-to-End Chart Editing benchmark that directly evaluates models without relying on intermediate natural language programs or code-level supervision. ChartE$^{3}$ focuses on two complementary editing dimensions: local editing, which involves fine-grained appearance changes such as font or color adjustments, and global editing, which requires holistic, data-centric transformations including data filtering and trend line addition. ChartE$^{3}$ contains over 1,200 high-quality samples constructed via a well-designed data pipeline with human curation. Each sample is provided as a triplet of a chart image, its underlying code, and a multimodal editing instruction, enabling evaluation from both objective and subjective perspectives. Extensive benchmarking of state-of-the-art multimodal large language models reveals substantial performance gaps, particularly on global editing tasks, highlighting critical limitations in current end-to-end chart editing capabilities.
>
---
#### [new 018] Generative Recall, Dense Reranking: Learning Multi-View Semantic IDs for Efficient Text-to-Video Retrieval
- **分类: cs.CV**

- **简介: 该论文属于文本到视频检索任务，解决两阶段检索中召回模型性能不足的问题。提出GRDR方法，通过多视图语义ID提升召回质量，实现高效准确的检索。**

- **链接: [https://arxiv.org/pdf/2601.21193v1](https://arxiv.org/pdf/2601.21193v1)**

> **作者:** Zecheng Zhao; Zhi Chen; Zi Huang; Shazia Sadiq; Tong Chen
>
> **备注:** 10 pages
>
> **摘要:** Text-to-Video Retrieval (TVR) is essential in video platforms. Dense retrieval with dual-modality encoders leads in accuracy, but its computation and storage scale poorly with corpus size. Thus, real-time large-scale applications adopt two-stage retrieval, where a fast recall model gathers a small candidate pool, which is reranked by an advanced dense retriever. Due to hugely reduced candidates, the reranking model can use any off-the-shelf dense retriever without hurting efficiency, meaning the recall model bounds two-stage TVR performance. Recently, generative retrieval (GR) replaces dense video embeddings with discrete semantic IDs and retrieves by decoding text queries into ID tokens. GR offers near-constant inference and storage complexity, and its semantic IDs capture high-level video features via quantization, making it ideal for quickly eliminating irrelevant candidates during recall. However, as a recall model in two-stage TVR, GR suffers from (i) semantic ambiguity, where each video satisfies diverse queries but is forced into one semantic ID; and (ii) cross-modal misalignment, as semantic IDs are solely derived from visual features without text supervision. We propose Generative Recall and Dense Reranking (GRDR), designing a novel GR method to uplift recalled candidate quality. GRDR assigns multiple semantic IDs to each video using a query-guided multi-view tokenizer exposing diverse semantic access paths, and jointly trains the tokenizer and generative retriever via a shared codebook to cast semantic IDs as the semantic bridge between texts and videos. At inference, trie-constrained decoding generates a compact candidate set reranked by a dense model for fine-grained matching. Experiments on TVR benchmarks show GRDR matches strong dense retrievers in accuracy while reducing index storage by an order of magnitude and accelerating up to 300$\times$ in full-corpus retrieval.
>
---
#### [new 019] Deep Models, Shallow Alignment: Uncovering the Granularity Mismatch in Neural Decoding
- **分类: cs.CV**

- **简介: 该论文属于神经视觉解码任务，旨在解决人类与机器视觉在细节和语义粒度上的不匹配问题。通过提出浅层对齐方法，提升解码性能。**

- **链接: [https://arxiv.org/pdf/2601.21948v1](https://arxiv.org/pdf/2601.21948v1)**

> **作者:** Yang Du; Siyuan Dai; Yonghao Song; Paul M. Thompson; Haoteng Tang; Liang Zhan
>
> **备注:** 29 pages, 13 figures
>
> **摘要:** Neural visual decoding is a central problem in brain computer interface research, aiming to reconstruct human visual perception and to elucidate the structure of neural representations. However, existing approaches overlook a fundamental granularity mismatch between human and machine vision, where deep vision models emphasize semantic invariance by suppressing local texture information, whereas neural signals preserve an intricate mixture of low-level visual attributes and high-level semantic content. To address this mismatch, we propose Shallow Alignment, a novel contrastive learning strategy that aligns neural signals with intermediate representations of visual encoders rather than their final outputs, thereby striking a better balance between low-level texture details and high-level semantic features. Extensive experiments across multiple benchmarks demonstrate that Shallow Alignment significantly outperforms standard final-layer alignment, with performance gains ranging from 22% to 58% across diverse vision backbones. Notably, our approach effectively unlocks the scaling law in neural visual decoding, enabling decoding performance to scale predictably with the capacity of pre-trained vision backbones. We further conduct systematic empirical analyses to shed light on the mechanisms underlying the observed performance gains.
>
---
#### [new 020] Unifying Heterogeneous Degradations: Uncertainty-Aware Diffusion Bridge Model for All-in-One Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决异质退化下的统一恢复问题。提出UDBM模型，通过不确定性感知的扩散桥方法，有效处理多种退化类型，提升恢复效果。**

- **链接: [https://arxiv.org/pdf/2601.21592v1](https://arxiv.org/pdf/2601.21592v1)**

> **作者:** Luwei Tu; Jiawei Wu; Xing Luo; Zhi Jin
>
> **摘要:** All-in-One Image Restoration (AiOIR) faces the fundamental challenge in reconciling conflicting optimization objectives across heterogeneous degradations. Existing methods are often constrained by coarse-grained control mechanisms or fixed mapping schedules, yielding suboptimal adaptation. To address this, we propose an Uncertainty-Aware Diffusion Bridge Model (UDBM), which innovatively reformulates AiOIR as a stochastic transport problem steered by pixel-wise uncertainty. By introducing a relaxed diffusion bridge formulation which replaces the strict terminal constraint with a relaxed constraint, we model the uncertainty of degradations while theoretically resolving the drift singularity inherent in standard diffusion bridges. Furthermore, we devise a dual modulation strategy: the noise schedule aligns diverse degradations into a shared high-entropy latent space, while the path schedule adaptively regulates the transport trajectory motivated by the viscous dynamics of entropy regularization. By effectively rectifying the transport geometry and dynamics, UDBM achieves state-of-the-art performance across diverse restoration tasks within a single inference step.
>
---
#### [new 021] Mining Forgery Traces from Reconstruction Error: A Weakly Supervised Framework for Multimodal Deepfake Temporal Localization
- **分类: cs.CV**

- **简介: 该论文属于深度伪造时间定位任务，解决弱监督下的精细时间定位问题。提出RT-DeepLoc框架，通过重建误差和对比损失实现有效伪造检测。**

- **链接: [https://arxiv.org/pdf/2601.21458v1](https://arxiv.org/pdf/2601.21458v1)**

> **作者:** Midou Guo; Qilin Yin; Wei Lu; Xiangyang Luo; Rui Yang
>
> **摘要:** Modern deepfakes have evolved into localized and intermittent manipulations that require fine-grained temporal localization. The prohibitive cost of frame-level annotation makes weakly supervised methods a practical necessity, which rely only on video-level labels. To this end, we propose Reconstruction-based Temporal Deepfake Localization (RT-DeepLoc), a weakly supervised temporal forgery localization framework that identifies forgeries via reconstruction errors. Our framework uses a Masked Autoencoder (MAE) trained exclusively on authentic data to learn its intrinsic spatiotemporal patterns; this allows the model to produce significant reconstruction discrepancies for forged segments, effectively providing the missing fine-grained cues for localization. To robustly leverage these indicators, we introduce a novel Asymmetric Intra-video Contrastive Loss (AICL). By focusing on the compactness of authentic features guided by these reconstruction cues, AICL establishes a stable decision boundary that enhances local discrimination while preserving generalization to unseen forgeries. Extensive experiments on large-scale datasets, including LAV-DF, demonstrate that RT-DeepLoc achieves state-of-the-art performance in weakly-supervised temporal forgery localization.
>
---
#### [new 022] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型量化任务，针对ARVG模型在量化过程中出现的三个挑战，提出PTQ4ARVG框架，实现高效低精度量化。**

- **链接: [https://arxiv.org/pdf/2601.21238v1](https://arxiv.org/pdf/2601.21238v1)**

> **作者:** Xuewen Liu; Zhikai Li; Jing Zhang; Mengjuan Chen; Qingyi Gu
>
> **备注:** ICLR 2026
>
> **摘要:** AutoRegressive Visual Generation (ARVG) models retain an architecture compatible with language models, while achieving performance comparable to diffusion-based models. Quantization is commonly employed in neural networks to reduce model size and computational latency. However, applying quantization to ARVG remains largely underexplored, and existing quantization methods fail to generalize effectively to ARVG models. In this paper, we explore this issue and identify three key challenges: (1) severe outliers at channel-wise level, (2) highly dynamic activations at token-wise level, and (3) mismatched distribution information at sample-wise level. To these ends, we propose PTQ4ARVG, a training-free post-training quantization (PTQ) framework consisting of: (1) Gain-Projected Scaling (GPS) mitigates the channel-wise outliers, which expands the quantization loss via a Taylor series to quantify the gain of scaling for activation-weight quantization, and derives the optimal scaling factor through differentiation.(2) Static Token-Wise Quantization (STWQ) leverages the inherent properties of ARVG, fixed token length and position-invariant distribution across samples, to address token-wise variance without incurring dynamic calibration overhead.(3) Distribution-Guided Calibration (DGC) selects samples that contribute most to distributional entropy, eliminating the sample-wise distribution mismatch. Extensive experiments show that PTQ4ARVG can effectively quantize the ARVG family models to 8-bit and 6-bit while maintaining competitive performance. Code is available at http://github.com/BienLuky/PTQ4ARVG .
>
---
#### [new 023] Low performing pixel correction in computed tomography with unrolled network and synthetic data training
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决CT中低性能像素导致的伪影问题。通过合成数据训练，提出一种双域校正方法，无需真实数据即可有效去除伪影。**

- **链接: [https://arxiv.org/pdf/2601.20995v1](https://arxiv.org/pdf/2601.20995v1)**

> **作者:** Hongxu Yang; Levente Lippenszky; Edina Timko; Lehel Ferenczi; Gopal Avinash
>
> **备注:** ISBI 2026 accepted
>
> **摘要:** Low performance pixels (LPP) in Computed Tomography (CT) detectors would lead to ring and streak artifacts in the reconstructed images, making them clinically unusable. In recent years, several solutions have been proposed to correct LPP artifacts, either in the image domain or in the sinogram domain using supervised deep learning methods. However, these methods require dedicated datasets for training, which are expensive to collect. Moreover, existing approaches focus solely either on image-space or sinogram-space correction, ignoring the intrinsic correlations from the forward operation of the CT geometry. In this work, we propose an unrolled dual-domain method based on synthetic data to correct LPP artifacts. Specifically, the intrinsic correlations of LPP between the sinogram and image domains are leveraged through synthetic data generated from natural images, enabling the trained model to correct artifacts without requiring any real-world clinical data. In experiments simulating 1-2% detectors defect near the isocenter, the proposed method outperformed the state-of-the-art approaches by a large margin. The results indicate that our solution can correct LPP artifacts without the cost of data collection for model training, and it is adaptable to different scanner settings for software-based applications.
>
---
#### [new 024] Mam-App: A Novel Parameter-Efficient Mamba Model for Apple Leaf Disease Classification
- **分类: cs.CV**

- **简介: 该论文属于苹果叶片疾病分类任务，旨在解决深度学习模型参数过多导致效率低的问题。提出Mam-App模型，在保持高精度的同时显著减少参数量。**

- **链接: [https://arxiv.org/pdf/2601.21307v1](https://arxiv.org/pdf/2601.21307v1)**

> **作者:** Md Nadim Mahamood; Md Imran Hasan; Md Rasheduzzaman; Ausrukona Ray; Md Shafi Ud Doula; Kamrul Hasan
>
> **备注:** 18 Pages, 7 Tables, 5 Figures
>
> **摘要:** The rapid growth of the global population, alongside exponential technological advancement, has intensified the demand for food production. Meeting this demand depends not only on increasing agricultural yield but also on minimizing food loss caused by crop diseases. Diseases account for a substantial portion of apple production losses, despite apples being among the most widely produced and nutritionally valuable fruits worldwide. Previous studies have employed machine learning techniques for feature extraction and early diagnosis of apple leaf diseases, and more recently, deep learning-based models have shown remarkable performance in disease recognition. However, most state-of-the-art deep learning models are highly parameter-intensive, resulting in increased training and inference time. Although lightweight models are more suitable for user-friendly and resource-constrained applications, they often suffer from performance degradation. To address the trade-off between efficiency and performance, we propose Mam-App, a parameter-efficient Mamba-based model for feature extraction and leaf disease classification. The proposed approach achieves competitive state-of-the-art performance on the PlantVillage Apple Leaf Disease dataset, attaining 99.58% accuracy, 99.30% precision, 99.14% recall, and a 99.22% F1-score, while using only 0.051M parameters. This extremely low parameter count makes the model suitable for deployment on drones, mobile devices, and other low-resource platforms. To demonstrate the robustness and generalizability of the proposed model, we further evaluate it on the PlantVillage Corn Leaf Disease and Potato Leaf Disease datasets. The model achieves 99.48%, 99.20%, 99.34%, and 99.27% accuracy, precision, recall, and F1-score on the corn dataset and 98.46%, 98.91%, 95.39%, and 97.01% on the potato dataset, respectively.
>
---
#### [new 025] Causal World Modeling for Robot Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决长期操作和数据效率问题。提出LingBot-VA模型，结合视觉与动作信息，实现高效控制与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.21998v1](https://arxiv.org/pdf/2601.21998v1)**

> **作者:** Lin Li; Qihang Zhang; Yiming Luo; Shuai Yang; Ruilin Wang; Fei Han; Mingrui Yu; Zelin Gao; Nan Xue; Xing Zhu; Yujun Shen; Yinghao Xu
>
> **备注:** Project page: https://technology.robbyant.com/lingbot-va Code: https://github.com/robbyant/lingbot-va
>
> **摘要:** This work highlights that video world modeling, alongside vision-language pre-training, establishes a fresh and independent foundation for robot learning. Intuitively, video world models provide the ability to imagine the near future by understanding the causality between actions and visual dynamics. Inspired by this, we introduce LingBot-VA, an autoregressive diffusion framework that learns frame prediction and policy execution simultaneously. Our model features three carefully crafted designs: (1) a shared latent space, integrating vision and action tokens, driven by a Mixture-of-Transformers (MoT) architecture, (2) a closed-loop rollout mechanism, allowing for ongoing acquisition of environmental feedback with ground-truth observations, (3) an asynchronous inference pipeline, parallelizing action prediction and motor execution to support efficient control. We evaluate our model on both simulation benchmarks and real-world scenarios, where it shows significant promise in long-horizon manipulation, data efficiency in post-training, and strong generalizability to novel configurations. The code and model are made publicly available to facilitate the community.
>
---
#### [new 026] Beyond Global Alignment: Fine-Grained Motion-Language Retrieval via Pyramidal Shapley-Taylor Learning
- **分类: cs.CV**

- **简介: 该论文属于运动与语言检索任务，旨在解决全局对齐忽略局部细节的问题。提出Pyramidal Shapley-Taylor框架，通过分层对齐提升细粒度匹配效果。**

- **链接: [https://arxiv.org/pdf/2601.21904v1](https://arxiv.org/pdf/2601.21904v1)**

> **作者:** Hanmo Chen; Guangtao Lyu; Chenghao Xu; Jiexi Yan; Xu Yang; Cheng Deng
>
> **摘要:** As a foundational task in human-centric cross-modal intelligence, motion-language retrieval aims to bridge the semantic gap between natural language and human motion, enabling intuitive motion analysis, yet existing approaches predominantly focus on aligning entire motion sequences with global textual representations. This global-centric paradigm overlooks fine-grained interactions between local motion segments and individual body joints and text tokens, inevitably leading to suboptimal retrieval performance. To address this limitation, we draw inspiration from the pyramidal process of human motion perception (from joint dynamics to segment coherence, and finally to holistic comprehension) and propose a novel Pyramidal Shapley-Taylor (PST) learning framework for fine-grained motion-language retrieval. Specifically, the framework decomposes human motion into temporal segments and spatial body joints, and learns cross-modal correspondences through progressive joint-wise and segment-wise alignment in a pyramidal fashion, effectively capturing both local semantic details and hierarchical structural relationships. Extensive experiments on multiple public benchmark datasets demonstrate that our approach significantly outperforms state-of-the-art methods, achieving precise alignment between motion segments and body joints and their corresponding text tokens. The code of this work will be released upon acceptance.
>
---
#### [new 027] Past- and Future-Informed KV Cache Policy with Salience Estimation in Autoregressive Video Diffusion
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决长序列生成中KV缓存策略效率低的问题。提出PaFu-KV方法，结合显著性估计优化缓存，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2601.21896v1](https://arxiv.org/pdf/2601.21896v1)**

> **作者:** Hanmo Chen; Chenghao Xu; Xu Yang; Xuan Chen; Cheng Deng
>
> **摘要:** Video generation is pivotal to digital media creation, and recent advances in autoregressive video generation have markedly enhanced the efficiency of real-time video synthesis. However, existing approaches generally rely on heuristic KV Cache policies, which ignore differences in token importance in long-term video generation. This leads to the loss of critical spatiotemporal information and the accumulation of redundant, invalid cache, thereby degrading video generation quality and efficiency. To address this limitation, we first observe that token contributions to video generation are highly time-heterogeneous and accordingly propose a novel Past- and Future-Informed KV Cache Policy (PaFu-KV). Specifically, PaFu-KV introduces a lightweight Salience Estimation Head distilled from a bidirectional teacher to estimate salience scores, allowing the KV cache to retain informative tokens while discarding less relevant ones. This policy yields a better quality-efficiency trade-off by shrinking KV cache capacity and reducing memory footprint at inference time. Extensive experiments on benchmarks demonstrate that our method preserves high-fidelity video generation quality while enables accelerated inference, thereby enabling more efficient long-horizon video generation. Our code will be released upon paper acceptance.
>
---
#### [new 028] MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Methods
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决开源视觉语言模型缺乏高质量推理数据的问题。通过构建大规模多模态推理数据集MMFineReason，并进行模型微调，提升了模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2601.21821v1](https://arxiv.org/pdf/2601.21821v1)**

> **作者:** Honglin Lin; Zheng Liu; Yun Zhu; Chonghan Qin; Juekai Lin; Xiaoran Shang; Conghui He; Wentao Zhang; Lijun Wu
>
> **摘要:** Recent advances in Vision Language Models (VLMs) have driven significant progress in visual reasoning. However, open-source VLMs still lag behind proprietary systems, largely due to the lack of high-quality reasoning data. Existing datasets offer limited coverage of challenging domains such as STEM diagrams and visual puzzles, and lack consistent, long-form Chain-of-Thought (CoT) annotations essential for eliciting strong reasoning capabilities. To bridge this gap, we introduce MMFineReason, a large-scale multimodal reasoning dataset comprising 1.8M samples and 5.1B solution tokens, featuring high-quality reasoning annotations distilled from Qwen3-VL-235B-A22B-Thinking. The dataset is established via a systematic three-stage pipeline: (1) large-scale data collection and standardization, (2) CoT rationale generation, and (3) comprehensive selection based on reasoning quality and difficulty awareness. The resulting dataset spans STEM problems, visual puzzles, games, and complex diagrams, with each sample annotated with visually grounded reasoning traces. We fine-tune Qwen3-VL-Instruct on MMFineReason to develop MMFineReason-2B/4B/8B versions. Our models establish new state-of-the-art results for their size class. Notably, MMFineReason-4B succesfully surpasses Qwen3-VL-8B-Thinking, and MMFineReason-8B even outperforms Qwen3-VL-30B-A3B-Thinking while approaching Qwen3-VL-32B-Thinking, demonstrating remarkable parameter efficiency. Crucially, we uncover a "less is more" phenomenon via our difficulty-aware filtering strategy: a subset of just 7\% (123K samples) achieves performance comparable to the full dataset. Notably, we reveal a synergistic effect where reasoning-oriented data composition simultaneously boosts general capabilities.
>
---
#### [new 029] NFCDS: A Plug-and-Play Noise Frequency-Controlled Diffusion Sampling Strategy for Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，解决扩散采样中数据保真与感知质量的矛盾。提出NFCDS方法，通过频域滤波控制噪声，提升恢复效果。**

- **链接: [https://arxiv.org/pdf/2601.21248v1](https://arxiv.org/pdf/2601.21248v1)**

> **作者:** Zhen Wang; Hongyi Liu; Jianing Li; Zhihui Wei
>
> **摘要:** Diffusion sampling-based Plug-and-Play (PnP) methods produce images with high perceptual quality but often suffer from reduced data fidelity, primarily due to the noise introduced during reverse diffusion. To address this trade-off, we propose Noise Frequency-Controlled Diffusion Sampling (NFCDS), a spectral modulation mechanism for reverse diffusion noise. We show that the fidelity-perception conflict can be fundamentally understood through noise frequency: low-frequency components induce blur and degrade fidelity, while high-frequency components drive detail generation. Based on this insight, we design a Fourier-domain filter that progressively suppresses low-frequency noise and preserves high-frequency content. This controlled refinement injects a data-consistency prior directly into sampling, enabling fast convergence to results that are both high-fidelity and perceptually convincing--without additional training. As a PnP module, NFCDS seamlessly integrates into existing diffusion-based restoration frameworks and improves the fidelity-perception balance across diverse zero-shot tasks.
>
---
#### [new 030] PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出PLANING，用于单目视频流的3D重建任务，解决传统方法难以兼顾几何精度与渲染质量的问题，通过耦合三角形和高斯分布实现高效重建。**

- **链接: [https://arxiv.org/pdf/2601.22046v1](https://arxiv.org/pdf/2601.22046v1)**

> **作者:** Changjian Jiang; Kerui Ren; Xudong Li; Kaiwen Song; Linning Xu; Tao Lu; Junting Dong; Yu Zhang; Bo Dai; Mulin Yu
>
> **摘要:** Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of \modelname~make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
>
---
#### [new 031] BLO-Inst: Bi-Level Optimization Based Alignment of YOLO and SAM for Robust Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文属于实例分割任务，解决YOLO与SAM对齐问题。通过双层优化框架BLO-Inst，提升检测器生成高质量分割提示的能力。**

- **链接: [https://arxiv.org/pdf/2601.22061v1](https://arxiv.org/pdf/2601.22061v1)**

> **作者:** Li Zhang; Pengtao Xie
>
> **摘要:** The Segment Anything Model has revolutionized image segmentation with its zero-shot capabilities, yet its reliance on manual prompts hinders fully automated deployment. While integrating object detectors as prompt generators offers a pathway to automation, existing pipelines suffer from two fundamental limitations: objective mismatch, where detectors optimized for geometric localization do not correspond to the optimal prompting context required by SAM, and alignment overfitting in standard joint training, where the detector simply memorizes specific prompt adjustments for training samples rather than learning a generalizable policy. To bridge this gap, we introduce BLO-Inst, a unified framework that aligns detection and segmentation objectives by bi-level optimization. We formulate the alignment as a nested optimization problem over disjoint data splits. In the lower level, the SAM is fine-tuned to maximize segmentation fidelity given the current detection proposals on a subset ($D_1$). In the upper level, the detector is updated to generate bounding boxes that explicitly minimize the validation loss of the fine-tuned SAM on a separate subset ($D_2$). This effectively transforms the detector into a segmentation-aware prompt generator, optimizing the bounding boxes not just for localization accuracy, but for downstream mask quality. Extensive experiments demonstrate that BLO-Inst achieves superior performance, outperforming standard baselines on tasks in general and biomedical domains.
>
---
#### [new 032] Towards Geometry-Aware and Motion-Guided Video Human Mesh Recovery
- **分类: cs.CV**

- **简介: 该论文属于视频人体网格重建任务，旨在解决现有方法生成结果不真实的问题。提出HMRMamba框架，结合几何感知和运动引导机制，提升重建精度与时间一致性。**

- **链接: [https://arxiv.org/pdf/2601.21376v1](https://arxiv.org/pdf/2601.21376v1)**

> **作者:** Hongjun Chen; Huan Zheng; Wencheng Han; Jianbing Shen
>
> **摘要:** Existing video-based 3D Human Mesh Recovery (HMR) methods often produce physically implausible results, stemming from their reliance on flawed intermediate 3D pose anchors and their inability to effectively model complex spatiotemporal dynamics. To overcome these deep-rooted architectural problems, we introduce HMRMamba, a new paradigm for HMR that pioneers the use of Structured State Space Models (SSMs) for their efficiency and long-range modeling prowess. Our framework is distinguished by two core contributions. First, the Geometry-Aware Lifting Module, featuring a novel dual-scan Mamba architecture, creates a robust foundation for reconstruction. It directly grounds the 2D-to-3D pose lifting process with geometric cues from image features, producing a highly reliable 3D pose sequence that serves as a stable anchor. Second, the Motion-guided Reconstruction Network leverages this anchor to explicitly process kinematic patterns over time. By injecting this crucial temporal awareness, it significantly enhances the final mesh's coherence and robustness, particularly under occlusion and motion blur. Comprehensive evaluations on 3DPW, MPI-INF-3DHP, and Human3.6M benchmarks confirm that HMRMamba sets a new state-of-the-art, outperforming existing methods in both reconstruction accuracy and temporal consistency while offering superior computational efficiency.
>
---
#### [new 033] FRISM: Fine-Grained Reasoning Injection via Subspace-Level Model Merging for Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉语言模型的推理增强任务，旨在解决融合大推理模型时牺牲视觉能力的问题。通过子空间级模型融合实现细粒度推理注入，提升推理能力同时保持视觉性能。**

- **链接: [https://arxiv.org/pdf/2601.21187v1](https://arxiv.org/pdf/2601.21187v1)**

> **作者:** Chenyu Huang; Peng Ye; Xudong Tan; Jinhan Mu; Shenghe Zheng; Li Shen; Tao Chen
>
> **备注:** 23 pages, 8 figures
>
> **摘要:** Efficiently enhancing the reasoning capabilities of Vision-Language Models (VLMs) by merging them with Large Reasoning Models (LRMs) has emerged as a promising direction. However, existing methods typically operate at a coarse-grained layer level, which often leads to a trade-off between injecting reasoning capabilities and preserving visual capabilities. To address this limitation, we propose {FRISM} (Fine-grained Reasoning Injection via Subspace-level model Merging), a fine-grained reasoning injection framework based on subspace-level model merging. Observing that reasoning capabilities are encoded in distinct subspaces, FRISM decomposes LRM task vectors via Singular Value Decomposition (SVD) and adaptively tunes the scaling coefficients of each subspace through learning to realize fine-grained reasoning injection. Furthermore, we introduce a label-free self-distillation learning strategy with a dual-objective optimization using common vision-language perception datasets. Extensive experiments demonstrate that FRISM effectively improves reasoning capabilities without compromising the model's original visual capabilities by consistently achieving state-of-the-art performance across diverse visual reasoning benchmarks.
>
---
#### [new 034] Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决驾驶轨迹预测中的多模态问题。通过结合视频预训练与轨迹蒸馏，提出Drive-JEPA框架，提升驾驶决策性能。**

- **链接: [https://arxiv.org/pdf/2601.22032v1](https://arxiv.org/pdf/2601.22032v1)**

> **作者:** Linhan Wang; Zichong Yang; Chen Bai; Guoxiang Zhang; Xiaotong Liu; Xiaoyin Zheng; Xiao-Xiao Long; Chang-Tien Lu; Cheng Lu
>
> **摘要:** End-to-end autonomous driving increasingly leverages self-supervised video pretraining to learn transferable planning representations. However, pretraining video world models for scene understanding has so far brought only limited improvements. This limitation is compounded by the inherent ambiguity of driving: each scene typically provides only a single human trajectory, making it difficult to learn multimodal behaviors. In this work, we propose Drive-JEPA, a framework that integrates Video Joint-Embedding Predictive Architecture (V-JEPA) with multimodal trajectory distillation for end-to-end driving. First, we adapt V-JEPA for end-to-end driving, pretraining a ViT encoder on large-scale driving videos to produce predictive representations aligned with trajectory planning. Second, we introduce a proposal-centric planner that distills diverse simulator-generated trajectories alongside human trajectories, with a momentum-aware selection mechanism to promote stable and safe behavior. When evaluated on NAVSIM, the V-JEPA representation combined with a simple transformer-based decoder outperforms prior methods by 3 PDMS in the perception-free setting. The complete Drive-JEPA framework achieves 93.3 PDMS on v1 and 87.8 EPDMS on v2, setting a new state-of-the-art.
>
---
#### [new 035] RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决单图参考生成的局限性，通过引入3D资产提升生成多样性与一致性。工作包括设计双分支模型，联合建模RGB与点云信息。**

- **链接: [https://arxiv.org/pdf/2601.22094v1](https://arxiv.org/pdf/2601.22094v1)**

> **作者:** Hanzhuo Huang; Qingyang Bao; Zekai Gu; Zhongshuo Du; Cheng Lin; Yuan Liu; Sibei Yang
>
> **备注:** ICLR 2026. Project page: https://judgementh.github.io/RefAny3D Codes: https://github.com/JudgementH/RefAny3D
>
> **摘要:** In this paper, we propose a 3D asset-referenced diffusion model for image generation, exploring how to integrate 3D assets into image diffusion models. Existing reference-based image generation methods leverage large-scale pretrained diffusion models and demonstrate strong capability in generating diverse images conditioned on a single reference image. However, these methods are limited to single-image references and cannot leverage 3D assets, constraining their practical versatility. To address this gap, we present a cross-domain diffusion model with dual-branch perception that leverages multi-view RGB images and point maps of 3D assets to jointly model their colors and canonical-space coordinates, achieving precise consistency between generated images and the 3D references. Our spatially aligned dual-branch generation architecture and domain-decoupled generation mechanism ensure the simultaneous generation of two spatially aligned but content-disentangled outputs, RGB images and point maps, linking 2D image attributes with 3D asset attributes. Experiments show that our approach effectively uses 3D assets as references to produce images consistent with the given assets, opening new possibilities for combining diffusion models with 3D content creation.
>
---
#### [new 036] Token Entropy Regularization for Multi-modal Antenna Affiliation Identification
- **分类: cs.CV**

- **简介: 该论文属于多模态分类任务，旨在解决基站天线归属识别问题。通过融合视频、几何特征和PCI信号，提出Token Entropy Regularization提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.21280v1](https://arxiv.org/pdf/2601.21280v1)**

> **作者:** Dong Chen; Ruoyu Li; Xinyan Zhang; Jialei Xu; Ruoseng Zhao; Zhikang Zhang; Lingyun Li; Zizhuang Wei
>
> **摘要:** Accurate antenna affiliation identification is crucial for optimizing and maintaining communication networks. Current practice, however, relies on the cumbersome and error-prone process of manual tower inspections. We propose a novel paradigm shift that fuses video footage of base stations, antenna geometric features, and Physical Cell Identity (PCI) signals, transforming antenna affiliation identification into multi-modal classification and matching tasks. Publicly available pretrained transformers struggle with this unique task due to a lack of analogous data in the communications domain, which hampers cross-modal alignment. To address this, we introduce a dedicated training framework that aligns antenna images with corresponding PCI signals. To tackle the representation alignment challenge, we propose a novel Token Entropy Regularization module in the pretraining stage. Our experiments demonstrate that TER accelerates convergence and yields significant performance gains. Further analysis reveals that the entropy of the first token is modality-dependent. Code will be made available upon publication.
>
---
#### [new 037] MetricAnything: Scaling Metric Depth Pretraining with Noisy Heterogeneous Sources
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于度量深度估计任务，解决传感器噪声、相机偏差和数据歧义问题。提出Metric Anything框架，通过稀疏度量提示实现高效预训练，提升多种深度相关任务性能。**

- **链接: [https://arxiv.org/pdf/2601.22054v1](https://arxiv.org/pdf/2601.22054v1)**

> **作者:** Baorui Ma; Jiahui Yang; Donglin Di; Xuancheng Zhang; Jianxun Cui; Hao Li; Yan Xie; Wei Chen
>
> **备注:** Project Page: https://metric-anything.github.io/metric-anything-io/
>
> **摘要:** Scaling has powered recent advances in vision foundation models, yet extending this paradigm to metric depth estimation remains challenging due to heterogeneous sensor noise, camera-dependent biases, and metric ambiguity in noisy cross-source 3D data. We introduce Metric Anything, a simple and scalable pretraining framework that learns metric depth from noisy, diverse 3D sources without manually engineered prompts, camera-specific modeling, or task-specific architectures. Central to our approach is the Sparse Metric Prompt, created by randomly masking depth maps, which serves as a universal interface that decouples spatial reasoning from sensor and camera biases. Using about 20M image-depth pairs spanning reconstructed, captured, and rendered 3D data across 10000 camera models, we demonstrate-for the first time-a clear scaling trend in the metric depth track. The pretrained model excels at prompt-driven tasks such as depth completion, super-resolution and Radar-camera fusion, while its distilled prompt-free student achieves state-of-the-art results on monocular depth estimation, camera intrinsics recovery, single/multi-view metric 3D reconstruction, and VLA planning. We also show that using pretrained ViT of Metric Anything as a visual encoder significantly boosts Multimodal Large Language Model capabilities in spatial intelligence. These results show that metric depth estimation can benefit from the same scaling laws that drive modern foundation models, establishing a new path toward scalable and efficient real-world metric perception. We open-source MetricAnything at http://metric-anything.github.io/metric-anything-io/ to support community research.
>
---
#### [new 038] Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于深度学习中的OOD检测任务，旨在解决DNN对分布外输入的过度自信问题。通过最优传输理论生成语义模糊样本，并训练模型降低不确定区域的置信度。**

- **链接: [https://arxiv.org/pdf/2601.21320v1](https://arxiv.org/pdf/2601.21320v1)**

> **作者:** Keke Tang; Ziyong Du; Xiaofei Wang; Weilong Peng; Peican Zhu; Zhihong Tian
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Deep neural networks (DNNs) often produce overconfident predictions on out-of-distribution (OOD) inputs, undermining their reliability in open-world environments. Singularities in semi-discrete optimal transport (OT) mark regions of semantic ambiguity, where classifiers are particularly prone to unwarranted high-confidence predictions. Motivated by this observation, we propose a principled framework to mitigate OOD overconfidence by leveraging the geometry of OT-induced singular boundaries. Specifically, we formulate an OT problem between a continuous base distribution and the latent embeddings of training data, and identify the resulting singular boundaries. By sampling near these boundaries, we construct a class of OOD inputs, termed optimal transport-induced OOD samples (OTIS), which are geometrically grounded and inherently semantically ambiguous. During training, a confidence suppression loss is applied to OTIS to guide the model toward more calibrated predictions in structurally uncertain regions. Extensive experiments show that our method significantly alleviates OOD overconfidence and outperforms state-of-the-art methods.
>
---
#### [new 039] PathReasoner-R1: Instilling Structured Reasoning into Pathology Vision-Language Model via Knowledge-Guided Policy Optimization
- **分类: cs.CV**

- **简介: 该论文属于医学视觉语言任务，旨在解决病理诊断缺乏可验证推理的问题。通过构建大规模数据集并引入知识引导的强化学习方法，提升模型的结构化推理能力。**

- **链接: [https://arxiv.org/pdf/2601.21617v1](https://arxiv.org/pdf/2601.21617v1)**

> **作者:** Songhan Jiang; Fengchun Liu; Ziyue Wang; Linghan Cai; Yongbing Zhang
>
> **摘要:** Vision-Language Models (VLMs) are advancing computational pathology with superior visual understanding capabilities. However, current systems often reduce diagnosis to directly output conclusions without verifiable evidence-linked reasoning, which severely limits clinical trust and hinders expert error rectification. To address these barriers, we construct PathReasoner, the first large-scale dataset of whole-slide image (WSI) reasoning. Unlike previous work reliant on unverified distillation, we develop a rigorous knowledge-guided generation pipeline. By leveraging medical knowledge graphs, we explicitly align structured pathological findings and clinical reasoning with diagnoses, generating over 20K high-quality instructional samples. Based on the database, we propose PathReasoner-R1, which synergizes trajectory-masked supervised fine-tuning with reasoning-oriented reinforcement learning to instill structured chain-of-thought capabilities. To ensure medical rigor, we engineer a knowledge-aware multi-granular reward function incorporating an Entity Reward mechanism strictly aligned with knowledge graphs. This effectively guides the model to optimize for logical consistency rather than mere outcome matching, thereby enhancing robustness. Extensive experiments demonstrate that PathReasoner-R1 achieves state-of-the-art performance on both PathReasoner and public benchmarks across various image scales, equipping pathology models with transparent, clinically grounded reasoning capabilities. Dataset and code are available at https://github.com/cyclexfy/PathReasoner-R1.
>
---
#### [new 040] Hypernetwork-Based Adaptive Aggregation for Multimodal Multiple-Instance Learning in Predicting Coronary Calcium Debulking
- **分类: cs.CV**

- **简介: 该论文属于多模态多实例学习任务，旨在根据CT图像和患者数据预测冠状动脉钙化剥脱的必要性。提出HyperAdAgFormer模型，通过超网络自适应调整特征聚合策略。**

- **链接: [https://arxiv.org/pdf/2601.21479v1](https://arxiv.org/pdf/2601.21479v1)**

> **作者:** Kaito Shiku; Ichika Seo; Tetsuya Matoba; Rissei Hino; Yasuhiro Nakano; Ryoma Bise
>
> **备注:** Accepted to ISBI 2026
>
> **摘要:** In this paper, we present the first attempt to estimate the necessity of debulking coronary artery calcifications from computed tomography (CT) images. We formulate this task as a Multiple-instance Learning (MIL) problem. The difficulty of this task lies in that physicians adjust their focus and decision criteria for device usage according to tabular data representing each patient's condition. To address this issue, we propose a hypernetwork-based adaptive aggregation transformer (HyperAdAgFormer), which adaptively modifies the feature aggregation strategy for each patient based on tabular data through a hypernetwork. The experiments using the clinical dataset demonstrated the effectiveness of HyperAdAgFormer. The code is publicly available at https://github.com/Shiku-Kaito/HyperAdAgFormer.
>
---
#### [new 041] Understanding Multimodal Complementarity for Single-Frame Action Anticipation
- **分类: cs.CV**

- **简介: 该论文属于动作预见任务，研究在单帧图像中如何有效预测未来动作。通过融合多模态信息，提升单帧动作预见性能，验证其在多个基准上的有效性。**

- **链接: [https://arxiv.org/pdf/2601.22039v1](https://arxiv.org/pdf/2601.22039v1)**

> **作者:** Manuel Benavent-Lledo; Konstantinos Bacharidis; Konstantinos Papoutsakis; Antonis Argyros; Jose Garcia-Rodriguez
>
> **摘要:** Human action anticipation is commonly treated as a video understanding problem, implicitly assuming that dense temporal information is required to reason about future actions. In this work, we challenge this assumption by investigating what can be achieved when action anticipation is constrained to a single visual observation. We ask a fundamental question: how much information about the future is already encoded in a single frame, and how can it be effectively exploited? Building on our prior work on Action Anticipation at a Glimpse (AAG), we conduct a systematic investigation of single-frame action anticipation enriched with complementary sources of information. We analyze the contribution of RGB appearance, depth-based geometric cues, and semantic representations of past actions, and investigate how different multimodal fusion strategies, keyframe selection policies and past-action history sources influence anticipation performance. Guided by these findings, we consolidate the most effective design choices into AAG+, a refined single-frame anticipation framework. Despite operating on a single frame, AAG+ consistently improves upon the original AAG and achieves performance comparable to, or exceeding, that of state-of-the-art video-based methods on challenging anticipation benchmarks including IKEA-ASM, Meccano and Assembly101. Our results offer new insights into the limits and potential of single-frame action anticipation, and clarify when dense temporal modeling is necessary and when a carefully selected glimpse is sufficient.
>
---
#### [new 042] Similarity of Processing Steps in Vision Model Representations
- **分类: cs.CV**

- **简介: 该论文研究视觉模型表示的处理步骤相似性，探讨不同模型是否收敛到相同中间过程。任务是分析模型表示的演化路径，解决模型间差异与相似性问题。工作包括量化不同阶段的表示距离，识别关键差异步骤。**

- **链接: [https://arxiv.org/pdf/2601.21621v1](https://arxiv.org/pdf/2601.21621v1)**

> **作者:** Matéo Mahaut; Marco Baroni
>
> **摘要:** Recent literature suggests that the bigger the model, the more likely it is to converge to similar, ``universal'' representations, despite different training objectives, datasets, or modalities. While this literature shows that there is an area where model representations are similar, we study here how vision models might get to those representations--in particular, do they also converge to the same intermediate steps and operations? We therefore study the processes that lead to convergent representations in different models. First, we quantify distance between different model representations at different stages. We follow the evolution of distances between models throughout processing, identifying the processing steps which are most different between models. We find that while layers at similar positions in different models have the most similar representations, strong differences remain. Classifier models, unlike the others, will discard information about low-level image statistics in their final layers. CNN- and transformer-based models also behave differently, with transformer models applying smoother changes to representations from one layer to the next. These distinctions clarify the level and nature of convergence between model representations, and enables a more qualitative account of the underlying processes in image models.
>
---
#### [new 043] Enhancing Underwater Light Field Images via Global Geometry-aware Diffusion Process
- **分类: cs.CV**

- **简介: 该论文属于 underwater image enhancement 任务，旨在解决水下4-D光场图像质量差的问题。提出GeoDiff-LF框架，结合扩散模型与光场几何结构，提升图像质量。**

- **链接: [https://arxiv.org/pdf/2601.21179v1](https://arxiv.org/pdf/2601.21179v1)**

> **作者:** Yuji Lin; Qian Zhao; Zongsheng Yue; Junhui Hou; Deyu Meng
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** This work studies the challenging problem of acquiring high-quality underwater images via 4-D light field (LF) imaging. To this end, we propose GeoDiff-LF, a novel diffusion-based framework built upon SD-Turbo to enhance underwater 4-D LF imaging by leveraging its spatial-angular structure. GeoDiff-LF consists of three key adaptations: (1) a modified U-Net architecture with convolutional and attention adapters to model geometric cues, (2) a geometry-guided loss function using tensor decomposition and progressive weighting to regularize global structure, and (3) an optimized sampling strategy with noise prediction to improve efficiency. By integrating diffusion priors and LF geometry, GeoDiff-LF effectively mitigates color distortion in underwater scenes. Extensive experiments demonstrate that our framework outperforms existing methods across both visual fidelity and quantitative performance, advancing the state-of-the-art in enhancing underwater imaging. The code will be publicly available at https://github.com/linlos1234/GeoDiff-LF.
>
---
#### [new 044] CG-MLLM: Captioning and Generating 3D content via Multi-modal Large Language Models
- **分类: cs.CV**

- **简介: 该论文提出CG-MLLM，解决3D内容生成任务中分辨率低、细节不足的问题，通过多模态架构实现高保真3D生成。**

- **链接: [https://arxiv.org/pdf/2601.21798v1](https://arxiv.org/pdf/2601.21798v1)**

> **作者:** Junming Huang; Weiwei Xu
>
> **摘要:** Large Language Models(LLMs) have revolutionized text generation and multimodal perception, but their capabilities in 3D content generation remain underexplored. Existing methods compromise by producing either low-resolution meshes or coarse structural proxies, failing to capture fine-grained geometry natively. In this paper, we propose CG-MLLM, a novel Multi-modal Large Language Model (MLLM) capable of 3D captioning and high-resolution 3D generation in a single framework. Leveraging the Mixture-of-Transformer architecture, CG-MLLM decouples disparate modeling needs, where the Token-level Autoregressive (TokenAR) Transformer handles token-level content, and the Block-level Autoregressive (BlockAR) Transformer handles block-level content. By integrating a pre-trained vision-language backbone with a specialized 3D VAE latent space, CG-MLLM facilitates long-context interactions between standard tokens and spatial blocks within a single integrated architecture. Experimental results show that CG-MLLM significantly outperforms existing MLLMs in generating high-fidelity 3D objects, effectively bringing high-resolution 3D content creation into the mainstream LLM paradigm.
>
---
#### [new 045] A Tilted Seesaw: Revisiting Autoencoder Trade-off for Controllable Diffusion
- **分类: cs.CV**

- **简介: 论文探讨了扩散模型中自编码器的权衡问题，指出传统评估指标偏向生成效果而忽视重建质量，导致可控性下降。研究旨在提升可控扩散模型的可靠性，通过分析不同指标对条件保持的影响，提出更合理的评估方法。**

- **链接: [https://arxiv.org/pdf/2601.21633v1](https://arxiv.org/pdf/2601.21633v1)**

> **作者:** Pu Cao; Yiyang Ma; Feng Zhou; Xuedan Yin; Qing Song; Lu Yang
>
> **备注:** work in progress
>
> **摘要:** In latent diffusion models, the autoencoder (AE) is typically expected to balance two capabilities: faithful reconstruction and a generation-friendly latent space (e.g., low gFID). In recent ImageNet-scale AE studies, we observe a systematic bias toward generative metrics in handling this trade-off: reconstruction metrics are increasingly under-reported, and ablation-based AE selection often favors the best-gFID configuration even when reconstruction fidelity degrades. We theoretically analyze why this gFID-dominant preference can appear unproblematic for ImageNet generation, yet becomes risky when scaling to controllable diffusion: AEs can induce condition drift, which limits achievable condition alignment. Meanwhile, we find that reconstruction fidelity, especially instance-level measures, better indicates controllability. We empirically validate the impact of tilted autoencoder evaluation on controllability by studying several recent ImageNet AEs. Using a multi-dimensional condition-drift evaluation protocol reflecting controllable generation tasks, we find that gFID is only weakly predictive of condition preservation, whereas reconstruction-oriented metrics are substantially more aligned. ControlNet experiments further confirm that controllability tracks condition preservation rather than gFID. Overall, our results expose a gap between ImageNet-centric AE evaluation and the requirements of scalable controllable diffusion, offering practical guidance for more reliable benchmarking and model selection.
>
---
#### [new 046] Early and Prediagnostic Detection of Pancreatic Cancer from Computed Tomography
- **分类: cs.CV**

- **简介: 该论文属于胰腺癌早期检测任务，旨在解决晚期诊断问题。通过开发AI系统ePAI，提高CT影像中早期胰腺癌的检出率和定位精度。**

- **链接: [https://arxiv.org/pdf/2601.22134v1](https://arxiv.org/pdf/2601.22134v1)**

> **作者:** Wenxuan Li; Pedro R. A. S. Bassi; Lizhou Wu; Xinze Zhou; Yuxuan Zhao; Qi Chen; Szymon Plotka; Tianyu Lin; Zheren Zhu; Marisa Martin; Justin Caskey; Shanshan Jiang; Xiaoxi Chen; Jaroslaw B. Ćwikla; Artur Sankowski; Yaping Wu; Sergio Decherchi; Andrea Cavalli; Chandana Lall; Cristian Tomasetti; Yaxing Guo; Xuan Yu; Yuqing Cai; Hualin Qiao; Jie Bao; Chenhan Hu; Ximing Wang; Arkadiusz Sitek; Kai Ding; Heng Li; Meiyun Wang; Dexin Yu; Guang Zhang; Yang Yang; Kang Wang; Alan L. Yuille; Zongwei Zhou
>
> **摘要:** Pancreatic ductal adenocarcinoma (PDAC), one of the deadliest solid malignancies, is often detected at a late and inoperable stage. Retrospective reviews of prediagnostic CT scans, when conducted by expert radiologists aware that the patient later developed PDAC, frequently reveal lesions that were previously overlooked. To help detecting these lesions earlier, we developed an automated system named ePAI (early Pancreatic cancer detection with Artificial Intelligence). It was trained on data from 1,598 patients from a single medical center. In the internal test involving 1,009 patients, ePAI achieved an area under the receiver operating characteristic curve (AUC) of 0.939-0.999, a sensitivity of 95.3%, and a specificity of 98.7% for detecting small PDAC less than 2 cm in diameter, precisely localizing PDAC as small as 2 mm. In an external test involving 7,158 patients across 6 centers, ePAI achieved an AUC of 0.918-0.945, a sensitivity of 91.5%, and a specificity of 88.0%, precisely localizing PDAC as small as 5 mm. Importantly, ePAI detected PDACs on prediagnostic CT scans obtained 3 to 36 months before clinical diagnosis that had originally been overlooked by radiologists. It successfully detected and localized PDACs in 75 of 159 patients, with a median lead time of 347 days before clinical diagnosis. Our multi-reader study showed that ePAI significantly outperformed 30 board-certified radiologists by 50.3% (P < 0.05) in sensitivity while maintaining a comparable specificity of 95.4% in detecting PDACs early and prediagnostic. These findings suggest its potential of ePAI as an assistive tool to improve early detection of pancreatic cancer.
>
---
#### [new 047] From Implicit Ambiguity to Explicit Solidity: Diagnosing Interior Geometric Degradation in Neural Radiance Fields for Dense 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，解决神经辐射场在密集自遮挡场景中的几何退化问题。通过引入显式几何方法提升重建准确性。**

- **链接: [https://arxiv.org/pdf/2601.21421v1](https://arxiv.org/pdf/2601.21421v1)**

> **作者:** Jiangsan Zhao; Jakob Geipel; Kryzysztof Kusnierek
>
> **摘要:** Neural Radiance Fields (NeRFs) have emerged as a powerful paradigm for multi-view reconstruction, complementing classical photogrammetric pipelines based on Structure-from-Motion (SfM) and Multi-View Stereo (MVS). However, their reliability for quantitative 3D analysis in dense, self-occluding scenes remains poorly understood. In this study, we identify a fundamental failure mode of implicit density fields under heavy occlusion, which we term Interior Geometric Degradation (IGD). We show that transmittance-based volumetric optimization satisfies photometric supervision by reconstructing hollow or fragmented structures rather than solid interiors, leading to systematic instance undercounting. Through controlled experiments on synthetic datasets with increasing occlusion, we demonstrate that state-of-the-art mask-supervised NeRFs saturate at approximately 89% instance recovery in dense scenes, despite improved surface coherence and mask quality. To overcome this limitation, we introduce an explicit geometric pipeline based on Sparse Voxel Rasterization (SVRaster), initialized from SfM feature geometry. By projecting 2D instance masks onto an explicit voxel grid and enforcing geometric separation via recursive splitting, our approach preserves physical solidity and achieves a 95.8% recovery rate in dense clusters. A sensitivity analysis using degraded segmentation masks further shows that explicit SfM-based geometry is substantially more robust to supervision failure, recovering 43% more instances than implicit baselines. These results demonstrate that explicit geometric priors are a prerequisite for reliable quantitative analysis in highly self-occluding 3D scenes.
>
---
#### [new 048] Generation Enhances Understanding in Unified Multimodal Models via Multi-Representation Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态理解与生成任务，旨在解决生成对理解的促进问题。通过引入多表示生成任务，增强模型对视觉输入的深度理解。**

- **链接: [https://arxiv.org/pdf/2601.21406v1](https://arxiv.org/pdf/2601.21406v1)**

> **作者:** Zihan Su; Hongyang Wei; Kangrui Cen; Yong Wang; Guanhua Chen; Chun Yuan; Xiangxiang Chu
>
> **摘要:** Unified Multimodal Models (UMMs) integrate both visual understanding and generation within a single framework. Their ultimate aspiration is to create a cycle where understanding and generation mutually reinforce each other. While recent post-training methods have successfully leveraged understanding to enhance generation, the reverse direction of utilizing generation to improve understanding remains largely unexplored. In this work, we propose UniMRG (Unified Multi-Representation Generation), a simple yet effective architecture-agnostic post-training method. UniMRG enhances the understanding capabilities of UMMs by incorporating auxiliary generation tasks. Specifically, we train UMMs to generate multiple intrinsic representations of input images, namely pixel (reconstruction), depth (geometry), and segmentation (structure), alongside standard visual understanding objectives. By synthesizing these diverse representations, UMMs capture complementary information regarding appearance, spatial relations, and structural layout. Consequently, UMMs develop a deeper and more comprehensive understanding of visual inputs. Extensive experiments across diverse UMM architectures demonstrate that our method notably enhances fine-grained perception, reduces hallucinations, and improves spatial understanding, while simultaneously boosting generation capabilities.
>
---
#### [new 049] Variance & Greediness: A comparative study of metric-learning losses
- **分类: cs.CV**

- **简介: 该论文属于度量学习任务，旨在比较不同损失函数对嵌入空间的影响。通过分析七种损失函数在五组数据集上的表现，揭示了效率与粒度的权衡，为实际应用提供指导。**

- **链接: [https://arxiv.org/pdf/2601.21450v1](https://arxiv.org/pdf/2601.21450v1)**

> **作者:** Donghuo Zeng; Hao Niu; Zhi Li; Masato Taya
>
> **备注:** 5 pages, 2 figures, 3 tables. Accepted by ICASSP 2026
>
> **摘要:** Metric learning is central to retrieval, yet its effects on embedding geometry and optimization dynamics are not well understood. We introduce a diagnostic framework, VARIANCE (intra-/inter-class variance) and GREEDINESS (active ratio and gradient norms), to compare seven representative losses, i.e., Contrastive, Triplet, N-pair, InfoNCE, ArcFace, SCL, and CCL, across five image-retrieval datasets. Our analysis reveals that Triplet and SCL preserve higher within-class variance and clearer inter-class margins, leading to stronger top-1 retrieval in fine-grained settings. In contrast, Contrastive and InfoNCE compact embeddings are achieved quickly through many small updates, accelerating convergence but potentially oversimplifying class structures. N-pair achieves a large mean separation but with uneven spacing. These insights reveal a form of efficiency-granularity trade-off and provide practical guidance: prefer Triplet/SCL when diversity preservation and hard-sample discrimination are critical, and Contrastive/InfoNCE when faster embedding compaction is desired.
>
---
#### [new 050] Text controllable PET denoising
- **分类: cs.CV**

- **简介: 该论文属于PET图像去噪任务，旨在解决PET图像噪声影响诊断的问题。通过引入文本引导的去噪方法，提升不同计数水平下的图像质量。**

- **链接: [https://arxiv.org/pdf/2601.20990v1](https://arxiv.org/pdf/2601.20990v1)**

> **作者:** Xuehua Ye; Hongxu Yang; Adam J. Schwarz
>
> **备注:** SPIE Medical Imaging 2026
>
> **摘要:** Positron Emission Tomography (PET) imaging is a vital tool in medical diagnostics, offering detailed insights into molecular processes within the human body. However, PET images often suffer from complicated noise, which can obscure critical diagnostic information. The quality of the PET image is impacted by various factors including scanner hardware, image reconstruction, tracer properties, dose/count level, and acquisition time. In this study, we propose a novel text-guided denoising method capable of enhancing PET images across a wide range of count levels within a single model. The model utilized the features from a pretrained CLIP model with a U-Net based denoising model. Experimental results demonstrate that the proposed model leads significant improvements in both qualitative and quantitative assessments. The flexibility of the model shows the potential for helping more complicated denoising demands or reducing the acquisition time.
>
---
#### [new 051] Bi-Anchor Interpolation Solver for Accelerating Generative Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生成建模任务，旨在解决流匹配模型计算延迟高的问题。提出BA-solver，在保持高保真度的同时显著减少神经函数评估次数。**

- **链接: [https://arxiv.org/pdf/2601.21542v1](https://arxiv.org/pdf/2601.21542v1)**

> **作者:** Hongxu Chen; Hongxiang Li; Zhen Wang; Long Chen
>
> **摘要:** Flow Matching (FM) models have emerged as a leading paradigm for high-fidelity synthesis. However, their reliance on iterative Ordinary Differential Equation (ODE) solving creates a significant latency bottleneck. Existing solutions face a dichotomy: training-free solvers suffer from significant performance degradation at low Neural Function Evaluations (NFEs), while training-based one- or few-steps generation methods incur prohibitive training costs and lack plug-and-play versatility. To bridge this gap, we propose the Bi-Anchor Interpolation Solver (BA-solver). BA-solver retains the versatility of standard training-free solvers while achieving significant acceleration by introducing a lightweight SideNet (1-2% backbone size) alongside the frozen backbone. Specifically, our method is founded on two synergistic components: \textbf{1) Bidirectional Temporal Perception}, where the SideNet learns to approximate both future and historical velocities without retraining the heavy backbone; and 2) Bi-Anchor Velocity Integration, which utilizes the SideNet with two anchor velocities to efficiently approximate intermediate velocities for batched high-order integration. By utilizing the backbone to establish high-precision ``anchors'' and the SideNet to densify the trajectory, BA-solver enables large interval sizes with minimized error. Empirical results on ImageNet-256^2 demonstrate that BA-solver achieves generation quality comparable to 100+ NFEs Euler solver in just 10 NFEs and maintains high fidelity in as few as 5 NFEs, incurring negligible training costs. Furthermore, BA-solver ensures seamless integration with existing generative pipelines, facilitating downstream tasks such as image editing.
>
---
#### [new 052] Bidirectional Cross-Perception for Open-Vocabulary Semantic Segmentation in Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文属于遥感图像语义分割任务，解决高密度、复杂边界下的开放词汇分割问题。提出SDCI框架，融合双分支特征并优化边界，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2601.21159v1](https://arxiv.org/pdf/2601.21159v1)**

> **作者:** Jianzheng Wang; Huan Ni
>
> **摘要:** High-resolution remote sensing imagery is characterized by densely distributed land-cover objects and complex boundaries, which places higher demands on both geometric localization and semantic prediction. Existing training-free open-vocabulary semantic segmentation (OVSS) methods typically fuse CLIP and vision foundation models (VFMs) using "one-way injection" and "shallow post-processing" strategies, making it difficult to satisfy these requirements. To address this issue, we propose a spatial-regularization-aware dual-branch collaborative inference framework for training-free OVSS, termed SDCI. First, during feature encoding, SDCI introduces a cross-model attention fusion (CAF) module, which guides collaborative inference by injecting self-attention maps into each other. Second, we propose a bidirectional cross-graph diffusion refinement (BCDR) module that enhances the reliability of dual-branch segmentation scores through iterative random-walk diffusion. Finally, we incorporate low-level superpixel structures and develop a convex-optimization-based superpixel collaborative prediction (CSCP) mechanism to further refine object boundaries. Experiments on multiple remote sensing semantic segmentation benchmarks demonstrate that our method achieves better performance than existing approaches. Moreover, ablation studies further confirm that traditional object-based remote sensing image analysis methods leveraging superpixel structures remain effective within deep learning frameworks. Code: https://github.com/yu-ni1989/SDCI.
>
---
#### [new 053] HiFi-Mesh: High-Fidelity Efficient 3D Mesh Generation via Compact Autoregressive Dependence
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D mesh生成任务，旨在解决现有方法效率低、细节表达不足的问题。提出LANE和AdaGraph，提升生成速度与质量。**

- **链接: [https://arxiv.org/pdf/2601.21314v1](https://arxiv.org/pdf/2601.21314v1)**

> **作者:** Yanfeng Li; Tao Tan; Qingquan Gao; Zhiwen Cao; Xiaohong liu; Yue Sun
>
> **摘要:** High-fidelity 3D meshes can be tokenized into one-dimension (1D) sequences and directly modeled using autoregressive approaches for faces and vertices. However, existing methods suffer from insufficient resource utilization, resulting in slow inference and the ability to handle only small-scale sequences, which severely constrains the expressible structural details. We introduce the Latent Autoregressive Network (LANE), which incorporates compact autoregressive dependencies in the generation process, achieving a $6\times$ improvement in maximum generatable sequence length compared to existing methods. To further accelerate inference, we propose the Adaptive Computation Graph Reconfiguration (AdaGraph) strategy, which effectively overcomes the efficiency bottleneck of traditional serial inference through spatiotemporal decoupling in the generation process. Experimental validation demonstrates that LANE achieves superior performance across generation speed, structural detail, and geometric consistency, providing an effective solution for high-quality 3D mesh generation.
>
---
#### [new 054] Unsupervised Decomposition and Recombination with Discriminator-Driven Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于无监督表示学习任务，旨在分解数据为可重用因子并实现高质量组合生成。通过引入对抗训练信号，提升因子发现与生成质量。**

- **链接: [https://arxiv.org/pdf/2601.22057v1](https://arxiv.org/pdf/2601.22057v1)**

> **作者:** Archer Wang; Emile Anand; Yilun Du; Marin Soljačić
>
> **备注:** 28 pages, 16 figures, 4 tables
>
> **摘要:** Decomposing complex data into factorized representations can reveal reusable components and enable synthesizing new samples via component recombination. We investigate this in the context of diffusion-based models that learn factorized latent spaces without factor-level supervision. In images, factors can capture background, illumination, and object attributes; in robotic videos, they can capture reusable motion components. To improve both latent factor discovery and quality of compositional generation, we introduce an adversarial training signal via a discriminator trained to distinguish between single-source samples and those generated by recombining factors across sources. By optimizing the generator to fool this discriminator, we encourage physical and semantic consistency in the resulting recombinations. Our method outperforms implementations of prior baselines on CelebA-HQ, Virtual KITTI, CLEVR, and Falcor3D, achieving lower FID scores and better disentanglement as measured by MIG and MCC. Furthermore, we demonstrate a novel application to robotic video trajectories: by recombining learned action components, we generate diverse sequences that significantly increase state-space coverage for exploration on the LIBERO benchmark.
>
---
#### [new 055] Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型研究，旨在探究VLM是感知还是回忆图像。通过设计VI-Probe框架，分析模型对视觉幻觉的响应机制，揭示其感知与记忆的混合行为。**

- **链接: [https://arxiv.org/pdf/2601.22150v1](https://arxiv.org/pdf/2601.22150v1)**

> **作者:** Xiaoxiao Sun; Mingyang Li; Kun yuan; Min Woo Sun; Mark Endo; Shengguang Wu; Changlin Li; Yuhui Zhang; Zeyu Wang; Serena Yeung-Levy
>
> **备注:** 26 pages, 31 figures, 13 tables. Project Page: https://sites.google.com/view/vi-probe/
>
> **摘要:** Large Vision-Language Models (VLMs) often answer classic visual illusions "correctly" on original images, yet persist with the same responses when illusion factors are inverted, even though the visual change is obvious to humans. This raises a fundamental question: do VLMs perceive visual changes or merely recall memorized patterns? While several studies have noted this phenomenon, the underlying causes remain unclear. To move from observations to systematic understanding, this paper introduces VI-Probe, a controllable visual-illusion framework with graded perturbations and matched visual controls (without illusion inducer) that disentangles visually grounded perception from language-driven recall. Unlike prior work that focuses on averaged accuracy, we measure stability and sensitivity using Polarity-Flip Consistency, Template Fixation Index, and an illusion multiplier normalized against matched controls. Experiments across different families reveal that response persistence arises from heterogeneous causes rather than a single mechanism. For instance, GPT-5 exhibits memory override, Claude-Opus-4.1 shows perception-memory competition, while Qwen variants suggest visual-processing limits. Our findings challenge single-cause views and motivate probing-based evaluation that measures both knowledge and sensitivity to controlled visual change. Data and code are available at https://sites.google.com/view/vi-probe/.
>
---
#### [new 056] When Gradient Optimization Is Not Enough: $\dagger$ Dispersive and Anchoring Geometric Regularizer for Multimodal Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态学习任务，旨在解决模型表示几何问题。提出一种轻量级正则化框架，通过分散和锚定机制提升表示多样性与跨模态一致性。**

- **链接: [https://arxiv.org/pdf/2601.21670v1](https://arxiv.org/pdf/2601.21670v1)**

> **作者:** Zixuan Xia; Hao Wang; Pengcheng Weng; Yanyu Qian; Yangxin Xu; William Dan; Fei Wang
>
> **摘要:** Multimodal learning aims to integrate complementary information from heterogeneous modalities, yet strong optimization alone does not guaranty well-structured representations. Even under carefully balanced training schemes, multimodal models often exhibit geometric pathologies, including intra-modal representation collapse and sample-level cross-modal inconsistency, which degrade both unimodal robustness and multimodal fusion. We identify representation geometry as a missing control axis in multimodal learning and propose \regName, a lightweight geometry-aware regularization framework. \regName enforces two complementary constraints on intermediate embeddings: an intra-modal dispersive regularization that promotes representation diversity, and an inter-modal anchoring regularization that bounds sample-level cross-modal drift without rigid alignment. The proposed regularizer is plug-and-play, requires no architectural modifications, and is compatible with various training paradigms. Extensive experiments across multiple multimodal benchmarks demonstrate consistent improvements in both multimodal and unimodal performance, showing that explicitly regulating representation geometry effectively mitigates modality trade-offs.
>
---
#### [new 057] Improving Classifier-Free Guidance of Flow Matching via Manifold Projection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像生成任务，旨在解决CFG方法对引导尺度敏感的问题。通过优化视角重新解释CFG，提出基于流形投影的改进方法，提升生成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.21892v1](https://arxiv.org/pdf/2601.21892v1)**

> **作者:** Jian-Feng Cai; Haixia Liu; Zhengyi Su; Chao Wang
>
> **备注:** 24 pages, 14 figures
>
> **摘要:** Classifier-free guidance (CFG) is a widely used technique for controllable generation in diffusion and flow-based models. Despite its empirical success, CFG relies on a heuristic linear extrapolation that is often sensitive to the guidance scale. In this work, we provide a principled interpretation of CFG through the lens of optimization. We demonstrate that the velocity field in flow matching corresponds to the gradient of a sequence of smoothed distance functions, which guides latent variables toward the scaled target image set. This perspective reveals that the standard CFG formulation is an approximation of this gradient, where the prediction gap, the discrepancy between conditional and unconditional outputs, governs guidance sensitivity. Leveraging this insight, we reformulate the CFG sampling as a homotopy optimization with a manifold constraint. This formulation necessitates a manifold projection step, which we implement via an incremental gradient descent scheme during sampling. To improve computational efficiency and stability, we further enhance this iterative process with Anderson Acceleration without requiring additional model evaluations. Our proposed methods are training-free and consistently refine generation fidelity, prompt alignment, and robustness to the guidance scale. We validate their effectiveness across diverse benchmarks, demonstrating significant improvements on large-scale models such as DiT-XL-2-256, Flux, and Stable Diffusion 3.5.
>
---
#### [new 058] Creative Image Generation with Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在提升生成图像的创造性。通过扩散模型，将图像导向CLIP空间中低概率区域，以生成独特、富有想象力的图像。**

- **链接: [https://arxiv.org/pdf/2601.22125v1](https://arxiv.org/pdf/2601.22125v1)**

> **作者:** Kunpeng Song; Ahmed Elgammal
>
> **备注:** Project page: https://creative-t2i.github.io
>
> **摘要:** Creative image generation has emerged as a compelling area of research, driven by the need to produce novel and high-quality images that expand the boundaries of imagination. In this work, we propose a novel framework for creative generation using diffusion models, where creativity is associated with the inverse probability of an image's existence in the CLIP embedding space. Unlike prior approaches that rely on a manual blending of concepts or exclusion of subcategories, our method calculates the probability distribution of generated images and drives it towards low-probability regions to produce rare, imaginative, and visually captivating outputs. We also introduce pullback mechanisms, achieving high creativity without sacrificing visual fidelity. Extensive experiments on text-to-image diffusion models demonstrate the effectiveness and efficiency of our creative generation framework, showcasing its ability to produce unique, novel, and thought-provoking images. This work provides a new perspective on creativity in generative models, offering a principled method to foster innovation in visual content synthesis.
>
---
#### [new 059] Shape of Thought: Progressive Object Assembly via Visual Chain-of-Thought
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决结构约束下的生成问题。提出SoT框架，通过视觉思维链实现渐进式形状组装，提升结构准确性和组件数量准确性。**

- **链接: [https://arxiv.org/pdf/2601.21081v1](https://arxiv.org/pdf/2601.21081v1)**

> **作者:** Yu Huo; Siyu Zhang; Kun Zeng; Haoyue Liu; Owen Lee; Junlin Chen; Yuquan Lu; Yifu Guo; Yaodong Liang; Xiaoying Tang
>
> **备注:** The code is available at https://anonymous.4open.science/r/16FE/
>
> **摘要:** Multimodal models for text-to-image generation have achieved strong visual fidelity, yet they remain brittle under compositional structural constraints-notably generative numeracy, attribute binding, and part-level relations. To address these challenges, we propose Shape-of-Thought (SoT), a visual CoT framework that enables progressive shape assembly via coherent 2D projections without external engines at inference time. SoT trains a unified multimodal autoregressive model to generate interleaved textual plans and rendered intermediate states, helping the model capture shape-assembly logic without producing explicit geometric representations. To support this paradigm, we introduce SoT-26K, a large-scale dataset of grounded assembly traces derived from part-based CAD hierarchies, and T2S-CompBench, a benchmark for evaluating structural integrity and trace faithfulness. Fine-tuning on SoT-26K achieves 88.4% on component numeracy and 84.8% on structural topology, outperforming text-only baselines by around 20%. SoT establishes a new paradigm for transparent, process-supervised compositional generation. The code is available at https://anonymous.4open.science/r/16FE/. The SoT-26K dataset will be released upon acceptance.
>
---
#### [new 060] WorldBench: Disambiguating Physics for Diagnostic Evaluation of World Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决现有基准测试中物理概念混杂的问题。提出WorldBench，实现物理概念的解耦评估，提升对世界模型物理推理能力的诊断精度。**

- **链接: [https://arxiv.org/pdf/2601.21282v1](https://arxiv.org/pdf/2601.21282v1)**

> **作者:** Rishi Upadhyay; Howard Zhang; Jim Solomon; Ayush Agrawal; Pranay Boreddy; Shruti Satya Narayana; Yunhao Ba; Alex Wong; Celso M de Melo; Achuta Kadambi
>
> **备注:** Webpage: https://world-bench.github.io/
>
> **摘要:** Recent advances in generative foundational models, often termed "world models," have propelled interest in applying them to critical tasks like robotic planning and autonomous system training. For reliable deployment, these models must exhibit high physical fidelity, accurately simulating real-world dynamics. Existing physics-based video benchmarks, however, suffer from entanglement, where a single test simultaneously evaluates multiple physical laws and concepts, fundamentally limiting their diagnostic capability. We introduce WorldBench, a novel video-based benchmark specifically designed for concept-specific, disentangled evaluation, allowing us to rigorously isolate and assess understanding of a single physical concept or law at a time. To make WorldBench comprehensive, we design benchmarks at two different levels: 1) an evaluation of intuitive physical understanding with concepts such as object permanence or scale/perspective, and 2) an evaluation of low-level physical constants and material properties such as friction coefficients or fluid viscosity. When SOTA video-based world models are evaluated on WorldBench, we find specific patterns of failure in particular physics concepts, with all tested models lacking the physical consistency required to generate reliable real-world interactions. Through its concept-specific evaluation, WorldBench offers a more nuanced and scalable framework for rigorously evaluating the physical reasoning capabilities of video generation and world models, paving the way for more robust and generalizable world-model-driven learning.
>
---
#### [new 061] HydroSense: A Dual-Microcontroller IoT Framework for Real-Time Multi-Parameter Water Quality Monitoring with Edge Processing and Cloud Analytics
- **分类: cs.CV**

- **简介: 论文提出HydroSense框架，用于实时多参数水质监测。解决资源受限环境下的监测难题，通过双微控制器架构实现精准测量与数据传输。**

- **链接: [https://arxiv.org/pdf/2601.21595v1](https://arxiv.org/pdf/2601.21595v1)**

> **作者:** Abdul Hasib; A. S. M. Ahsanul Sarkar Akib; Anish Giri
>
> **摘要:** The global water crisis necessitates affordable, accurate, and real-time water quality monitoring solutions. Traditional approaches relying on manual sampling or expensive commercial systems fail to address accessibility challenges in resource-constrained environments. This paper presents HydroSense, an innovative Internet of Things framework that integrates six critical water quality parameters including pH, dissolved oxygen (DO), temperature, total dissolved solids (TDS), estimated nitrogen, and water level into a unified monitoring system. HydroSense employs a novel dual-microcontroller architecture, utilizing Arduino Uno for precision analog measurements with five-point calibration algorithms and ESP32 for wireless connectivity, edge processing, and cloud integration. The system implements advanced signal processing techniques including median filtering for TDS measurement, temperature compensation algorithms, and robust error handling. Experimental validation over 90 days demonstrates exceptional performance metrics: pH accuracy of plus or minus 0.08 units across the 0 to 14 range, DO measurement stability within plus or minus 0.2 mg/L, TDS accuracy of plus or minus 1.9 percent across 0 to 1000 ppm, and 99.8 percent cloud data transmission reliability. With a total implementation cost of 32,983 BDT (approximately 300 USD), HydroSense achieves an 85 percent cost reduction compared to commercial systems while providing enhanced connectivity through the Firebase real-time database. This research establishes a new paradigm for accessible environmental monitoring, demonstrating that professional-grade water quality assessment can be achieved through intelligent system architecture and cost-effective component selection.
>
---
#### [new 062] PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing
- **分类: cs.CV**

- **简介: 该论文提出PaddleOCR-VL-1.5，解决文档解析任务中的鲁棒性问题，提升多任务处理能力，保持模型高效。**

- **链接: [https://arxiv.org/pdf/2601.21957v1](https://arxiv.org/pdf/2601.21957v1)**

> **作者:** Cheng Cui; Ting Sun; Suyin Liang; Tingquan Gao; Zelun Zhang; Jiaxuan Liu; Xueqing Wang; Changda Zhou; Hongen Liu; Manhui Lin; Yue Zhang; Yubo Zhang; Yi Liu; Dianhai Yu; Yanjun Ma
>
> **摘要:** We introduce PaddleOCR-VL-1.5, an upgraded model achieving a new state-of-the-art (SOTA) accuracy of 94.5% on OmniDocBench v1.5. To rigorously evaluate robustness against real-world physical distortions, including scanning, skew, warping, screen-photography, and illumination, we propose the Real5-OmniDocBench benchmark. Experimental results demonstrate that this enhanced model attains SOTA performance on the newly curated benchmark. Furthermore, we extend the model's capabilities by incorporating seal recognition and text spotting tasks, while remaining a 0.9B ultra-compact VLM with high efficiency. Code: https://github.com/PaddlePaddle/PaddleOCR
>
---
#### [new 063] An AI Framework for Microanastomosis Motion Assessment
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决微血管吻合术技能评估主观性强的问题。通过构建AI框架，实现手术器械的自动检测、跟踪与操作评估。**

- **链接: [https://arxiv.org/pdf/2601.21120v1](https://arxiv.org/pdf/2601.21120v1)**

> **作者:** Yan Meng; Eduardo J. Torres-Rodríguez; Marcelle Altshuler; Nishanth Gowda; Arhum Naeem; Recai Yilmaz; Omar Arnaout; Daniel A. Donoho
>
> **备注:** Accepted by IEEE/EMBS NER 2025. \c{opyright} 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
>
> **摘要:** Proficiency in microanastomosis is a fundamental competency across multiple microsurgical disciplines. These procedures demand exceptional precision and refined technical skills, making effective, standardized assessment methods essential. Traditionally, the evaluation of microsurgical techniques has relied heavily on the subjective judgment of expert raters. They are inherently constrained by limitations such as inter-rater variability, lack of standardized evaluation criteria, susceptibility to cognitive bias, and the time-intensive nature of manual review. These shortcomings underscore the urgent need for an objective, reliable, and automated system capable of assessing microsurgical performance with consistency and scalability. To bridge this gap, we propose a novel AI framework for the automated assessment of microanastomosis instrument handling skills. The system integrates four core components: (1) an instrument detection module based on the You Only Look Once (YOLO) architecture; (2) an instrument tracking module developed from Deep Simple Online and Realtime Tracking (DeepSORT); (3) an instrument tip localization module employing shape descriptors; and (4) a supervised classification module trained on expert-labeled data to evaluate instrument handling proficiency. Experimental results demonstrate the effectiveness of the framework, achieving an instrument detection precision of 97%, with a mean Average Precision (mAP) of 96%, measured by Intersection over Union (IoU) thresholds ranging from 50% to 95% (mAP50-95).
>
---
#### [new 064] SR$^{2}$-Net: A General Plug-and-Play Model for Spectral Refinement in Hyperspectral Image Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像超分辨率任务，旨在解决现有方法忽视光谱一致性导致的伪影问题。提出SR²-Net，通过增强与校正流程提升光谱保真度和重建质量。**

- **链接: [https://arxiv.org/pdf/2601.21338v1](https://arxiv.org/pdf/2601.21338v1)**

> **作者:** Ji-Xuan He; Guohang Zhuang; Junge Bo; Tingyi Li; Chen Ling; Yanan Qiao
>
> **摘要:** HSI-SR aims to enhance spatial resolution while preserving spectrally faithful and physically plausible characteristics. Recent methods have achieved great progress by leveraging spatial correlations to enhance spatial resolution. However, these methods often neglect spectral consistency across bands, leading to spurious oscillations and physically implausible artifacts. While spectral consistency can be addressed by designing the network architecture, it results in a loss of generality and flexibility. To address this issue, we propose a lightweight plug-and-play rectifier, physically priors Spectral Rectification Super-Resolution Network (SR$^{2}$-Net), which can be attached to a wide range of HSI-SR models without modifying their architectures. SR$^{2}$-Net follows an enhance-then-rectify pipeline consisting of (i) Hierarchical Spectral-Spatial Synergy Attention (H-S$^{3}$A) to reinforce cross-band interactions and (ii) Manifold Consistency Rectification (MCR) to constrain the reconstructed spectra to a compact, physically plausible spectral manifold. In addition, we introduce a degradation-consistency loss to enforce data fidelity by encouraging the degraded SR output to match the observed low resolution input. Extensive experiments on multiple benchmarks and diverse backbones demonstrate consistent improvements in spectral fidelity and overall reconstruction quality with negligible computational overhead. Our code will be released upon publication.
>
---
#### [new 065] SimGraph: A Unified Framework for Scene Graph-Based Image Generation and Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SimGraph，解决图像生成与编辑任务中对象关系和空间一致性问题，通过统一框架实现精准控制。**

- **链接: [https://arxiv.org/pdf/2601.21498v1](https://arxiv.org/pdf/2601.21498v1)**

> **作者:** Thanh-Nhan Vo; Trong-Thuan Nguyen; Tam V. Nguyen; Minh-Triet Tran
>
> **摘要:** Recent advancements in Generative Artificial Intelligence (GenAI) have significantly enhanced the capabilities of both image generation and editing. However, current approaches often treat these tasks separately, leading to inefficiencies and challenges in maintaining spatial consistency and semantic coherence between generated content and edits. Moreover, a major obstacle is the lack of structured control over object relationships and spatial arrangements. Scene graph-based methods, which represent objects and their interrelationships in a structured format, offer a solution by providing greater control over composition and interactions in both image generation and editing. To address this, we introduce SimGraph, a unified framework that integrates scene graph-based image generation and editing, enabling precise control over object interactions, layouts, and spatial coherence. In particular, our framework integrates token-based generation and diffusion-based editing within a single scene graph-driven model, ensuring high-quality and consistent results. Through extensive experiments, we empirically demonstrate that our approach outperforms existing state-of-the-art methods.
>
---
#### [new 066] GeoRC: A Benchmark for Geolocation Reasoning Chains
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型的地理推理任务，旨在解决模型无法准确解释地理位置判断依据的问题。通过构建基准数据集，评估模型生成推理链的能力。**

- **链接: [https://arxiv.org/pdf/2601.21278v1](https://arxiv.org/pdf/2601.21278v1)**

> **作者:** Mohit Talreja; Joshua Diao; Jim Thannikary James; Radu Casapu; Tejas Santanam; Ethan Mendes; Alan Ritter; Wei Xu; James Hays
>
> **摘要:** Vision Language Models (VLMs) are good at recognizing the global location of a photograph -- their geolocation prediction accuracy rivals the best human experts. But many VLMs are startlingly bad at explaining which image evidence led to their prediction, even when their location prediction is correct. The reasoning chains produced by VLMs frequently hallucinate scene attributes to support their location prediction (e.g. phantom writing, imagined infrastructure, misidentified flora). In this paper, we introduce the first benchmark for geolocation reasoning chains. We focus on the global location prediction task in the popular GeoGuessr game which draws from Google Street View spanning more than 100 countries. We collaborate with expert GeoGuessr players, including the reigning world champion, to produce 800 ground truth reasoning chains for 500 query scenes. These expert reasoning chains address hundreds of different discriminative visual attributes such as license plate shape, architecture, and soil properties to name just a few. We evaluate LLM-as-a-judge and VLM-as-a-judge strategies for scoring VLM-generated reasoning chains against our expert reasoning chains and find that Qwen 3 LLM-as-a-judge correlates best with human scoring. Our benchmark reveals that while large, closed-source VLMs such as Gemini and GPT 5 rival human experts at prediction locations, they still lag behind human experts when it comes to producing auditable reasoning chains. Open weights VLMs such as Llama and Qwen catastrophically fail on our benchmark -- they perform only slightly better than a baseline in which an LLM hallucinates a reasoning chain with oracle knowledge of the photo location but no visual information at all. We believe the gap between human experts and VLMs on this task points to VLM limitations at extracting fine-grained visual attributes from high resolution images.
>
---
#### [new 067] One-step Latent-free Image Generation with Pixel Mean Flows
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决传统扩散/流模型需多步采样和依赖潜在空间的问题。提出pMF方法，实现单步无潜在空间的图像生成。**

- **链接: [https://arxiv.org/pdf/2601.22158v1](https://arxiv.org/pdf/2601.22158v1)**

> **作者:** Yiyang Lu; Susie Lu; Qiao Sun; Hanhong Zhao; Zhicheng Jiang; Xianbang Wang; Tianhong Li; Zhengyang Geng; Kaiming He
>
> **备注:** Technical report
>
> **摘要:** Modern diffusion/flow-based models for image generation typically exhibit two core characteristics: (i) using multi-step sampling, and (ii) operating in a latent space. Recent advances have made encouraging progress on each aspect individually, paving the way toward one-step diffusion/flow without latents. In this work, we take a further step towards this goal and propose "pixel MeanFlow" (pMF). Our core guideline is to formulate the network output space and the loss space separately. The network target is designed to be on a presumed low-dimensional image manifold (i.e., x-prediction), while the loss is defined via MeanFlow in the velocity space. We introduce a simple transformation between the image manifold and the average velocity field. In experiments, pMF achieves strong results for one-step latent-free generation on ImageNet at 256x256 resolution (2.22 FID) and 512x512 resolution (2.48 FID), filling a key missing piece in this regime. We hope that our study will further advance the boundaries of diffusion/flow-based generative models.
>
---
#### [new 068] PI-Light: Physics-Inspired Diffusion for Full-Image Relighting
- **分类: cs.CV**

- **简介: 该论文属于图像重光照任务，解决真实场景下图像重光照的难题。提出PI-Light框架，结合物理启发的扩散模型，提升光照一致性与真实性。**

- **链接: [https://arxiv.org/pdf/2601.22135v1](https://arxiv.org/pdf/2601.22135v1)**

> **作者:** Zhexin Liang; Zhaoxi Chen; Yongwei Chen; Tianyi Wei; Tengfei Wang; Xingang Pan
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Full-image relighting remains a challenging problem due to the difficulty of collecting large-scale structured paired data, the difficulty of maintaining physical plausibility, and the limited generalizability imposed by data-driven priors. Existing attempts to bridge the synthetic-to-real gap for full-scene relighting remain suboptimal. To tackle these challenges, we introduce Physics-Inspired diffusion for full-image reLight ($π$-Light, or PI-Light), a two-stage framework that leverages physics-inspired diffusion models. Our design incorporates (i) batch-aware attention, which improves the consistency of intrinsic predictions across a collection of images, (ii) a physics-guided neural rendering module that enforces physically plausible light transport, (iii) physics-inspired losses that regularize training dynamics toward a physically meaningful landscape, thereby enhancing generalizability to real-world image editing, and (iv) a carefully curated dataset of diverse objects and scenes captured under controlled lighting conditions. Together, these components enable efficient finetuning of pretrained diffusion models while also providing a solid benchmark for downstream evaluation. Experiments demonstrate that $π$-Light synthesizes specular highlights and diffuse reflections across a wide variety of materials, achieving superior generalization to real-world scenes compared with prior approaches.
>
---
#### [new 069] LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models
- **分类: cs.CV**

- **简介: 该论文研究多图像视觉语言模型的对抗攻击问题，提出LAMP方法生成通用扰动，有效破坏模型跨图像信息融合，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2601.21220v1](https://arxiv.org/pdf/2601.21220v1)**

> **作者:** Alvi Md Ishmam; Najibul Haque Sarker; Zaber Ibn Abdul Hakim; Chris Thomas
>
> **备注:** Accepted in main technical track AAAI 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable performance across vision-language tasks. Recent advancements allow these models to process multiple images as inputs. However, the vulnerabilities of multi-image MLLMs remain unexplored. Existing adversarial attacks focus on single-image settings and often assume a white-box threat model, which is impractical in many real-world scenarios. This paper introduces LAMP, a black-box method for learning Universal Adversarial Perturbations (UAPs) targeting multi-image MLLMs. LAMP applies an attention-based constraint that prevents the model from effectively aggregating information across images. LAMP also introduces a novel cross-image contagious constraint that forces perturbed tokens to influence clean tokens, spreading adversarial effects without requiring all inputs to be modified. Additionally, an index-attention suppression loss enables a robust position-invariant attack. Experimental results show that LAMP outperforms SOTA baselines and achieves the highest attack success rates across multiple vision-language tasks and models.
>
---
#### [new 070] MA-LipNet: Multi-Dimensional Attention Networks for Robust Lipreading
- **分类: cs.CV**

- **简介: 该论文属于唇读任务，旨在提升唇读模型的鲁棒性。针对现有方法特征区分度低、泛化能力差的问题，提出MA-LipNet，通过多维注意力机制优化视觉特征。**

- **链接: [https://arxiv.org/pdf/2601.20881v1](https://arxiv.org/pdf/2601.20881v1)**

> **作者:** Matteo Rossi
>
> **摘要:** Lipreading, the technology of decoding spoken content from silent videos of lip movements, holds significant application value in fields such as public security. However, due to the subtle nature of articulatory gestures, existing lipreading methods often suffer from limited feature discriminability and poor generalization capabilities. To address these challenges, this paper delves into the purification of visual features from temporal, spatial, and channel dimensions. We propose a novel method named Multi-Attention Lipreading Network(MA-LipNet). The core of MA-LipNet lies in its sequential application of three dedicated attention modules. Firstly, a \textit{Channel Attention (CA)} module is employed to adaptively recalibrate channel-wise features, thereby mitigating interference from less informative channels. Subsequently, two spatio-temporal attention modules with distinct granularities-\textit{Joint Spatial-Temporal Attention (JSTA)} and \textit{Separate Spatial-Temporal Attention (SSTA)}-are leveraged to suppress the influence of irrelevant pixels and video frames. The JSTA module performs a coarse-grained filtering by computing a unified weight map across the spatio-temporal dimensions, while the SSTA module conducts a more fine-grained refinement by separately modeling temporal and spatial attentions. Extensive experiments conducted on the CMLR and GRID datasets demonstrate that MA-LipNet significantly reduces the Character Error Rate (CER) and Word Error Rate (WER), validating its effectiveness and superiority over several state-of-the-art methods. Our work highlights the importance of multi-dimensional feature refinement for robust visual speech recognition.
>
---
#### [new 071] Non-Markov Multi-Round Conversational Image Generation with History-Conditioned MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多轮对话图像生成任务，解决非马尔可夫场景下的多轮一致性问题。提出数据构建策略、历史条件框架及优化方法，提升图像重建与个性化能力。**

- **链接: [https://arxiv.org/pdf/2601.20911v1](https://arxiv.org/pdf/2601.20911v1)**

> **作者:** Haochen Zhang; Animesh Sinha; Felix Juefei-Xu; Haoyu Ma; Kunpeng Li; Zhipeng Fan; Meng Dong; Xiaoliang Dai; Tingbo Hou; Peizhao Zhang; Zecheng He
>
> **备注:** 19 pages, 19 figures, plan for TIP
>
> **摘要:** Conversational image generation requires a model to follow user instructions across multiple rounds of interaction, grounded in interleaved text and images that accumulate as chat history. While recent multimodal large language models (MLLMs) can generate and edit images, most existing multi-turn benchmarks and training recipes are effectively Markov: the next output depends primarily on the most recent image, enabling shortcut solutions that ignore long-range history. In this work we formalize and target the more challenging non-Markov setting, where a user may refer back to earlier states, undo changes, or reference entities introduced several rounds ago. We present (i) non-Markov multi-round data construction strategies, including rollback-style editing that forces retrieval of earlier visual states and name-based multi-round personalization that binds names to appearances across rounds; (ii) a history-conditioned training and inference framework with token-level caching to prevent multi-round identity drift; and (iii) enabling improvements for high-fidelity image reconstruction and editable personalization, including a reconstruction-based DiT detokenizer and a multi-stage fine-tuning curriculum. We demonstrate that explicitly training for non-Markov interactions yields substantial improvements in multi-round consistency and instruction compliance, while maintaining strong single-round editing and personalization.
>
---
#### [new 072] Few-Shot Domain Adaptation with Temporal References and Static Priors for Glacier Calving Front Delineation
- **分类: cs.CV**

- **简介: 该论文属于冰川断裂线分割任务，解决模型在新区域泛化能力不足的问题。通过少量样本域适应、静态先验和季节参考图像，显著提升分割精度。**

- **链接: [https://arxiv.org/pdf/2601.21663v1](https://arxiv.org/pdf/2601.21663v1)**

> **作者:** Marcel Dreier; Nora Gourmelon; Dakota Pyles; Thorsten Seehaus; Matthias H. Braun; Andreas Maier; Vincent Christlein
>
> **摘要:** During benchmarking, the state-of-the-art model for glacier calving front delineation achieves near-human performance. However, when applied in a real-world setting at a novel study site, its delineation accuracy is insufficient for calving front products intended for further scientific analyses. This site represents an out-of-distribution domain for a model trained solely on the benchmark dataset. By employing a few-shot domain adaptation strategy, incorporating spatial static prior knowledge, and including summer reference images in the input time series, the delineation error is reduced from 1131.6 m to 68.7 m without any architectural modifications. These methodological advancements establish a framework for applying deep learning-based calving front segmentation to novel study sites, enabling calving front monitoring on a global scale.
>
---
#### [new 073] Thinker: A vision-language foundation model for embodied intelligence
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Thinker，一个用于具身智能的视觉语言基础模型，解决机器人在视频理解中的视角混淆和时间推理问题。通过构建专用数据集和融合关键帧与完整视频输入，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.21199v1](https://arxiv.org/pdf/2601.21199v1)**

> **作者:** Baiyu Pan; Daqin Luo; Junpeng Yang; Jiyuan Wang; Yixuan Zhang; Hailin Shi; Jichao Jiao
>
> **备注:** IROS 2025, 4 pages, 3 figures
>
> **摘要:** When large vision-language models are applied to the field of robotics, they encounter problems that are simple for humans yet error-prone for models. Such issues include confusion between third-person and first-person perspectives and a tendency to overlook information in video endings during temporal reasoning. To address these challenges, we propose Thinker, a large vision-language foundation model designed for embodied intelligence. We tackle the aforementioned issues from two perspectives. Firstly, we construct a large-scale dataset tailored for robotic perception and reasoning, encompassing ego-view videos, visual grounding, spatial understanding, and chain-of-thought data. Secondly, we introduce a simple yet effective approach that substantially enhances the model's capacity for video comprehension by jointly incorporating key frames and full video sequences as inputs. Our model achieves state-of-the-art results on two of the most commonly used benchmark datasets in the field of task planning.
>
---
#### [new 074] OCRVerse: Towards Holistic OCR in End-to-End Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出OCRVerse，解决文本和视觉信息共同存在的OCR任务，整合文本和视觉识别，提升跨领域准确性。**

- **链接: [https://arxiv.org/pdf/2601.21639v1](https://arxiv.org/pdf/2601.21639v1)**

> **作者:** Yufeng Zhong; Lei Chen; Xuanle Zhao; Wenkang Han; Liming Zheng; Jing Huang; Deyang Jiang; Yilin Cao; Lin Ma; Zhixiong Zeng
>
> **摘要:** The development of large vision language models drives the demand for managing, and applying massive amounts of multimodal data, making OCR technology, which extracts information from visual images, increasingly popular. However, existing OCR methods primarily focus on recognizing text elements from images or scanned documents (\textbf{Text-centric OCR}), neglecting the identification of visual elements from visually information-dense image sources (\textbf{Vision-centric OCR}), such as charts, web pages and science plots. In reality, these visually information-dense images are widespread on the internet and have significant real-world application value, such as data visualization and web page analysis. In this technical report, we propose \textbf{OCRVerse}, the first holistic OCR method in end-to-end manner that enables unified text-centric OCR and vision-centric OCR. To this end, we constructe comprehensive data engineering to cover a wide range of text-centric documents, such as newspapers, magazines and books, as well as vision-centric rendered composites, including charts, web pages and scientific plots. Moreover, we propose a two-stage SFT-RL multi-domain training method for OCRVerse. SFT directly mixes cross-domain data to train and establish initial domain knowledge, while RL focuses on designing personalized reward strategies for the characteristics of each domain. Specifically, since different domains require various output formats and expected outputs, we provide sufficient flexibility in the RL stage to customize flexible reward signals for each domain, thereby improving cross-domain fusion and avoiding data conflicts. Experimental results demonstrate the effectiveness of OCRVerse, achieving competitive results across text-centric and vision-centric data types, even comparable to large-scale open-source and closed-source models.
>
---
#### [new 075] Just Noticeable Difference Modeling for Deep Visual Features
- **分类: cs.CV**

- **简介: 该论文属于视觉特征质量控制任务，旨在解决如何量化深度视觉特征的可感知差异问题。提出FeatJND模型，用于预测特征扰动边界，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2601.21933v1](https://arxiv.org/pdf/2601.21933v1)**

> **作者:** Rui Zhao; Wenrui Li; Lin Zhu; Yajing Zheng; Weisi Lin
>
> **摘要:** Deep visual features are increasingly used as the interface in vision systems, motivating the need to describe feature characteristics and control feature quality for machine perception. Just noticeable difference (JND) characterizes the maximum imperceptible distortion for images under human or machine vision. Extending it to deep visual features naturally meets the above demand by providing a task-aligned tolerance boundary in feature space, offering a practical reference for controlling feature quality under constrained resources. We propose FeatJND, a task-aligned JND formulation that predicts the maximum tolerable per-feature perturbation map while preserving downstream task performance. We propose a FeatJND estimator at standardized split points and validate it across image classification, detection, and instance segmentation. Under matched distortion strength, FeatJND-based distortions consistently preserve higher task performance than unstructured Gaussian perturbations, and attribution visualizations suggest FeatJND can suppress non-critical feature regions. As an application, we further apply FeatJND to token-wise dynamic quantization and show that FeatJND-guided step-size allocation yields clear gains over random step-size permutation and global uniform step size under the same noise budget. Our code will be released after publication.
>
---
#### [new 076] UEval: A Benchmark for Unified Multimodal Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出UEval，一个评估统一多模态生成模型的基准，解决多模态生成质量评估难题。通过设计评分体系，实现细粒度自动评分。**

- **链接: [https://arxiv.org/pdf/2601.22155v1](https://arxiv.org/pdf/2601.22155v1)**

> **作者:** Bo Li; Yida Yin; Wenhao Chai; Xingyu Fu; Zhuang Liu
>
> **摘要:** We introduce UEval, a benchmark to evaluate unified models, i.e., models capable of generating both images and text. UEval comprises 1,000 expert-curated questions that require both images and text in the model output, sourced from 8 real-world tasks. Our curated questions cover a wide range of reasoning types, from step-by-step guides to textbook explanations. Evaluating open-ended multimodal generation is non-trivial, as simple LLM-as-a-judge methods can miss the subtleties. Different from previous works that rely on multimodal Large Language Models (MLLMs) to rate image quality or text accuracy, we design a rubric-based scoring system in UEval. For each question, reference images and text answers are provided to a MLLM to generate an initial rubric, consisting of multiple evaluation criteria, and human experts then refine and validate these rubrics. In total, UEval contains 10,417 validated rubric criteria, enabling scalable and fine-grained automatic scoring. UEval is challenging for current unified models: GPT-5-Thinking scores only 66.4 out of 100, while the best open-source model reaches merely 49.1. We observe that reasoning models often outperform non-reasoning ones, and transferring reasoning traces from a reasoning model to a non-reasoning model significantly narrows the gap. This suggests that reasoning may be important for tasks requiring complex multimodal understanding and generation.
>
---
#### [new 077] Dynamic Topology Awareness: Breaking the Granularity Rigidity in Vision-Language Navigation
- **分类: cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决拓扑地图构建中的“粒度刚性”问题。通过动态调节地图密度和连接性，提升导航精度与安全性。**

- **链接: [https://arxiv.org/pdf/2601.21751v1](https://arxiv.org/pdf/2601.21751v1)**

> **作者:** Jiankun Peng; Jianyuan Guo; Ying Xu; Yue Liu; Jiashuang Yan; Xuanwei Ye; Houhua Li; Xiaoming Wang
>
> **摘要:** Vision-Language Navigation in Continuous Environments (VLN-CE) presents a core challenge: grounding high-level linguistic instructions into precise, safe, and long-horizon spatial actions. Explicit topological maps have proven to be a vital solution for providing robust spatial memory in such tasks. However, existing topological planning methods suffer from a "Granularity Rigidity" problem. Specifically, these methods typically rely on fixed geometric thresholds to sample nodes, which fails to adapt to varying environmental complexities. This rigidity leads to a critical mismatch: the model tends to over-sample in simple areas, causing computational redundancy, while under-sampling in high-uncertainty regions, increasing collision risks and compromising precision. To address this, we propose DGNav, a framework for Dynamic Topological Navigation, introducing a context-aware mechanism to modulate map density and connectivity on-the-fly. Our approach comprises two core innovations: (1) A Scene-Aware Adaptive Strategy that dynamically modulates graph construction thresholds based on the dispersion of predicted waypoints, enabling "densification on demand" in challenging environments; (2) A Dynamic Graph Transformer that reconstructs graph connectivity by fusing visual, linguistic, and geometric cues into dynamic edge weights, enabling the agent to filter out topological noise and enhancing instruction adherence. Extensive experiments on the R2R-CE and RxR-CE benchmarks demonstrate DGNav exhibits superior navigation performance and strong generalization capabilities. Furthermore, ablation studies confirm that our framework achieves an optimal trade-off between navigation efficiency and safe exploration. The code is available at https://github.com/shannanshouyin/DGNav.
>
---
#### [new 078] Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态任务，旨在解决MLLM在复杂视觉搜索中的能力不足问题。通过提出Vision-DeepResearch，实现多轮、多实体、多尺度的深度搜索，提升模型在噪声环境下的表现。**

- **链接: [https://arxiv.org/pdf/2601.22060v1](https://arxiv.org/pdf/2601.22060v1)**

> **作者:** Wenxuan Huang; Yu Zeng; Qiuchen Wang; Zhen Fang; Shaosheng Cao; Zheng Chu; Qingyu Yin; Shuang Chen; Zhenfei Yin; Lin Chen; Zehui Chen; Yao Hu; Philip Torr; Feng Zhao; Wanli Ouyang
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable success across a broad range of vision tasks. However, constrained by the capacity of their internal world knowledge, prior work has proposed augmenting MLLMs by ``reasoning-then-tool-call'' for visual and textual search engines to obtain substantial gains on tasks requiring extensive factual information. However, these approaches typically define multimodal search in a naive setting, assuming that a single full-level or entity-level image query and few text query suffices to retrieve the key evidence needed to answer the question, which is unrealistic in real-world scenarios with substantial visual noise. Moreover, they are often limited in the reasoning depth and search breadth, making it difficult to solve complex questions that require aggregating evidence from diverse visual and textual sources. Building on this, we propose Vision-DeepResearch, which proposes one new multimodal deep-research paradigm, i.e., performs multi-turn, multi-entity and multi-scale visual and textual search to robustly hit real-world search engines under heavy noise. Our Vision-DeepResearch supports dozens of reasoning steps and hundreds of engine interactions, while internalizing deep-research capabilities into the MLLM via cold-start supervision and RL training, resulting in a strong end-to-end multimodal deep-research MLLM. It substantially outperforming existing multimodal deep-research MLLMs, and workflows built on strong closed-source foundation model such as GPT-5, Gemini-2.5-pro and Claude-4-Sonnet. The code will be released in https://github.com/Osilly/Vision-DeepResearch.
>
---
#### [new 079] Dynamical Adapter Fusion: Constructing A Global Adapter for Pre-Trained Model-based Class-Incremental Learning
- **分类: cs.CV**

- **简介: 该论文属于类增量学习任务，解决模型在持续学习新类时遗忘旧类的问题。通过构建全局适配器，融合任务特定参数与初始化参数，提升知识迁移能力。**

- **链接: [https://arxiv.org/pdf/2601.21341v1](https://arxiv.org/pdf/2601.21341v1)**

> **作者:** Ruiqi Liu; Boyu Diao; Zijia An; Zhulin An; Fei Wang; Yongjun Xu
>
> **摘要:** Class-Incremental Learning (CIL) requires models to continuously acquire new classes without forgetting previously learned ones. A dominant paradigm involves freezing a pre-trained model and training lightweight, task-specific adapters. However, maintaining task-specific parameters hinders knowledge transfer and incurs high retrieval costs, while naive parameter fusion often leads to destructive interference and catastrophic forgetting. To address these challenges, we propose Dynamical Adapter Fusion (DAF) to construct a single robust global adapter. Grounded in the PAC-Bayes theorem, we derive a fusion mechanism that explicitly integrates three components: the optimized task-specific adapter parameters, the previous global adapter parameters, and the initialization parameters. We utilize the Taylor expansion of the loss function to derive the optimal fusion coefficients, dynamically achieving the best balance between stability and plasticity. Furthermore, we propose a Robust Initialization strategy to effectively capture global knowledge patterns. Experiments on multiple CIL benchmarks demonstrate that DAF achieves state-of-the-art (SOTA) performance.
>
---
#### [new 080] Rectifying Geometry-Induced Similarity Distortions for Real-World Aerial-Ground Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于航空-地面行人重识别任务，解决因视角和距离差异导致的几何失真问题。提出GIQT模块，通过几何条件调整相似性计算，提升匹配性能。**

- **链接: [https://arxiv.org/pdf/2601.21405v1](https://arxiv.org/pdf/2601.21405v1)**

> **作者:** Kailash A. Hambarde; Hugo Proença
>
> **摘要:** Aerial-ground person re-identification (AG-ReID) is fundamentally challenged by extreme viewpoint and distance discrepancies between aerial and ground cameras, which induce severe geometric distortions and invalidate the assumption of a shared similarity space across views. Existing methods primarily rely on geometry-aware feature learning or appearance-conditioned prompting, while implicitly assuming that the geometry-invariant dot-product similarity used in attention mechanisms remains reliable under large viewpoint and scale variations. We argue that this assumption does not hold. Extreme camera geometry systematically distorts the query-key similarity space and degrades attention-based matching, even when feature representations are partially aligned. To address this issue, we introduce Geometry-Induced Query-Key Transformation (GIQT), a lightweight low-rank module that explicitly rectifies the similarity space by conditioning query-key interactions on camera geometry. Rather than modifying feature representations or the attention formulation itself, GIQT adapts the similarity computation to compensate for dominant geometry-induced anisotropic distortions. Building on this local similarity rectification, we further incorporate a geometry-conditioned prompt generation mechanism that provides global, view-adaptive representation priors derived directly from camera geometry. Experiments on four aerial-ground person re-identification benchmarks demonstrate that the proposed framework consistently improves robustness under extreme and previously unseen geometric conditions, while introducing minimal computational overhead compared to state-of-the-art methods.
>
---
#### [new 081] RSGround-R1: Rethinking Remote Sensing Visual Grounding through Spatial Reasoning
- **分类: cs.CV**

- **简介: 该论文属于遥感视觉定位任务，解决MLLM在复杂场景中空间推理不足的问题。提出RSGround-R1框架，通过位置感知训练和强化学习提升定位准确性。**

- **链接: [https://arxiv.org/pdf/2601.21634v1](https://arxiv.org/pdf/2601.21634v1)**

> **作者:** Shiqi Huang; Shuting He; Bihan Wen
>
> **摘要:** Remote Sensing Visual Grounding (RSVG) aims to localize target objects in large-scale aerial imagery based on natural language descriptions. Owing to the vast spatial scale and high semantic ambiguity of remote sensing scenes, these descriptions often rely heavily on positional cues, posing unique challenges for Multimodal Large Language Models (MLLMs) in spatial reasoning. To leverage this unique feature, we propose a reasoning-guided, position-aware post-training framework, dubbed \textbf{RSGround-R1}, to progressively enhance spatial understanding. Specifically, we first introduce Chain-of-Thought Supervised Fine-Tuning (CoT-SFT) using synthetically generated RSVG reasoning data to establish explicit position awareness. Reinforcement Fine-Tuning (RFT) is then applied, augmented by our newly designed positional reward that provides continuous and distance-aware guidance toward accurate localization. Moreover, to mitigate incoherent localization behaviors across rollouts, we introduce a spatial consistency guided optimization scheme that dynamically adjusts policy updates based on their spatial coherence, ensuring stable and robust convergence. Extensive experiments on RSVG benchmarks demonstrate superior performance and generalization of our model.
>
---
#### [new 082] Zero-Shot Video Restoration and Enhancement with Assistance of Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于视频修复与增强任务，解决视频修复中出现的帧间闪烁问题。通过融合文本到视频扩散模型，提升视频时序一致性。**

- **链接: [https://arxiv.org/pdf/2601.21922v1](https://arxiv.org/pdf/2601.21922v1)**

> **作者:** Cong Cao; Huanjing Yue; Shangbin Xie; Xin Liu; Jingyu Yang
>
> **摘要:** Although diffusion-based zero-shot image restoration and enhancement methods have achieved great success, applying them to video restoration or enhancement will lead to severe temporal flickering. In this paper, we propose the first framework that utilizes the rapidly-developed video diffusion model to assist the image-based method in maintaining more temporal consistency for zero-shot video restoration and enhancement. We propose homologous latents fusion, heterogenous latents fusion, and a COT-based fusion ratio strategy to utilize both homologous and heterogenous text-to-video diffusion models to complement the image method. Moreover, we propose temporal-strengthening post-processing to utilize the image-to-video diffusion model to further improve temporal consistency. Our method is training-free and can be applied to any diffusion-based image restoration and enhancement methods. Experimental results demonstrate the superiority of the proposed method.
>
---
#### [new 083] CAF-Mamba: Mamba-Based Cross-Modal Adaptive Attention Fusion for Multimodal Depression Detection
- **分类: cs.CV; cs.CY; cs.HC**

- **简介: 该论文属于多模态抑郁检测任务，旨在解决现有方法特征类型有限、忽略跨模态交互及融合方式简单的问题。提出CAF-Mamba框架，通过自适应注意力机制实现更有效的多模态融合。**

- **链接: [https://arxiv.org/pdf/2601.21648v1](https://arxiv.org/pdf/2601.21648v1)**

> **作者:** Bowen Zhou; Marc-André Fiedler; Ayoub Al-Hamadi
>
> **备注:** The paper contains a total of 5 pages and 3 figures. This paper has been accepted for publication in the proceedings of 2026 IEEE ICASSP Conference
>
> **摘要:** Depression is a prevalent mental health disorder that severely impairs daily functioning and quality of life. While recent deep learning approaches for depression detection have shown promise, most rely on limited feature types, overlook explicit cross-modal interactions, and employ simple concatenation or static weighting for fusion. To overcome these limitations, we propose CAF-Mamba, a novel Mamba-based cross-modal adaptive attention fusion framework. CAF-Mamba not only captures cross-modal interactions explicitly and implicitly, but also dynamically adjusts modality contributions through a modality-wise attention mechanism, enabling more effective multimodal fusion. Experiments on two in-the-wild benchmark datasets, LMVD and D-Vlog, demonstrate that CAF-Mamba consistently outperforms existing methods and achieves state-of-the-art performance.
>
---
#### [new 084] HERS: Hidden-Pattern Expert Learning for Risk-Specific Vehicle Damage Adaptation in Diffusion Models
- **分类: cs.CV**

- **简介: 该论文提出HERS框架，解决扩散模型生成车辆损伤图像的可靠性问题，提升图像真实性与可控性，用于保险领域风险适配。**

- **链接: [https://arxiv.org/pdf/2601.21517v1](https://arxiv.org/pdf/2601.21517v1)**

> **作者:** Teerapong Panboonyuen
>
> **备注:** 26 pages
>
> **摘要:** Recent advances in text-to-image (T2I) diffusion models have enabled increasingly realistic synthesis of vehicle damage, raising concerns about their reliability in automated insurance workflows. The ability to generate crash-like imagery challenges the boundary between authentic and synthetic data, introducing new risks of misuse in fraud or claim manipulation. To address these issues, we propose HERS (Hidden-Pattern Expert Learning for Risk-Specific Damage Adaptation), a framework designed to improve fidelity, controllability, and domain alignment of diffusion-generated damage images. HERS fine-tunes a base diffusion model via domain-specific expert adaptation without requiring manual annotation. Using self-supervised image-text pairs automatically generated by a large language model and T2I pipeline, HERS models each damage category, such as dents, scratches, broken lights, or cracked paint, as a separate expert. These experts are later integrated into a unified multi-damage model that balances specialization with generalization. We evaluate HERS across four diffusion backbones and observe consistent improvements: plus 5.5 percent in text faithfulness and plus 2.3 percent in human preference ratings compared to baselines. Beyond image fidelity, we discuss implications for fraud detection, auditability, and safe deployment of generative models in high-stakes domains. Our findings highlight both the opportunities and risks of domain-specific diffusion, underscoring the importance of trustworthy generation in safety-critical applications such as auto insurance.
>
---
#### [new 085] EditYourself: Audio-Driven Generation and Manipulation of Talking Head Videos with Diffusion Transformers
- **分类: cs.CV; cs.GR; cs.LG; cs.MM**

- **简介: 该论文提出EditYourself，解决视频编辑中语音驱动的 talking head 视频修改问题，通过扩散变换器实现精准唇形同步和内容重构。**

- **链接: [https://arxiv.org/pdf/2601.22127v1](https://arxiv.org/pdf/2601.22127v1)**

> **作者:** John Flynn; Wolfgang Paier; Dimitar Dinev; Sam Nhut Nguyen; Hayk Poghosyan; Manuel Toribio; Sandipan Banerjee; Guy Gafni
>
> **备注:** Project page: https://edit-yourself.github.io/
>
> **摘要:** Current generative video models excel at producing novel content from text and image prompts, but leave a critical gap in editing existing pre-recorded videos, where minor alterations to the spoken script require preserving motion, temporal coherence, speaker identity, and accurate lip synchronization. We introduce EditYourself, a DiT-based framework for audio-driven video-to-video (V2V) editing that enables transcript-based modification of talking head videos, including the seamless addition, removal, and retiming of visually spoken content. Building on a general-purpose video diffusion model, EditYourself augments its V2V capabilities with audio conditioning and region-aware, edit-focused training extensions. This enables precise lip synchronization and temporally coherent restructuring of existing performances via spatiotemporal inpainting, including the synthesis of realistic human motion in newly added segments, while maintaining visual fidelity and identity consistency over long durations. This work represents a foundational step toward generative video models as practical tools for professional video post-production.
>
---
#### [new 086] Synthetic-to-Real Domain Bridging for Single-View 3D Reconstruction of Ships for Maritime Monitoring
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文属于单视角船舶3D重建任务，解决真实场景下缺乏多视角监督和3D标注的问题。通过合成数据训练，结合自定义数据集和后处理，实现高效、实时的船舶3D重建与可视化。**

- **链接: [https://arxiv.org/pdf/2601.21786v1](https://arxiv.org/pdf/2601.21786v1)**

> **作者:** Borja Carrillo-Perez; Felix Sattler; Angel Bueno Rodriguez; Maurice Stephan; Sarah Barnes
>
> **摘要:** Three-dimensional (3D) reconstruction of ships is an important part of maritime monitoring, allowing improved visualization, inspection, and decision-making in real-world monitoring environments. However, most state-ofthe-art 3D reconstruction methods require multi-view supervision, annotated 3D ground truth, or are computationally intensive, making them impractical for real-time maritime deployment. In this work, we present an efficient pipeline for single-view 3D reconstruction of real ships by training entirely on synthetic data and requiring only a single view at inference. Our approach uses the Splatter Image network, which represents objects as sparse sets of 3D Gaussians for rapid and accurate reconstruction from single images. The model is first fine-tuned on synthetic ShapeNet vessels and further refined with a diverse custom dataset of 3D ships, bridging the domain gap between synthetic and real-world imagery. We integrate a state-of-the-art segmentation module based on YOLOv8 and custom preprocessing to ensure compatibility with the reconstruction network. Postprocessing steps include real-world scaling, centering, and orientation alignment, followed by georeferenced placement on an interactive web map using AIS metadata and homography-based mapping. Quantitative evaluation on synthetic validation data demonstrates strong reconstruction fidelity, while qualitative results on real maritime images from the ShipSG dataset confirm the potential for transfer to operational maritime settings. The final system provides interactive 3D inspection of real ships without requiring real-world 3D annotations. This pipeline provides an efficient, scalable solution for maritime monitoring and highlights a path toward real-time 3D ship visualization in practical applications. Interactive demo: https://dlr-mi.github.io/ship3d-demo/.
>
---
#### [new 087] DreamActor-M2: Universal Character Image Animation via Spatiotemporal In-Context Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于角色图像动画任务，解决运动注入与身份保持的平衡问题及对姿态先验的依赖问题，提出DreamActor-M2框架提升动画质量与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.21716v1](https://arxiv.org/pdf/2601.21716v1)**

> **作者:** Mingshuang Luo; Shuang Liang; Zhengkun Rong; Yuxuan Luo; Tianshu Hu; Ruibing Hou; Hong Chang; Yong Li; Yuan Zhang; Mingyuan Gao
>
> **摘要:** Character image animation aims to synthesize high-fidelity videos by transferring motion from a driving sequence to a static reference image. Despite recent advancements, existing methods suffer from two fundamental challenges: (1) suboptimal motion injection strategies that lead to a trade-off between identity preservation and motion consistency, manifesting as a "see-saw", and (2) an over-reliance on explicit pose priors (e.g., skeletons), which inadequately capture intricate dynamics and hinder generalization to arbitrary, non-humanoid characters. To address these challenges, we present DreamActor-M2, a universal animation framework that reimagines motion conditioning as an in-context learning problem. Our approach follows a two-stage paradigm. First, we bridge the input modality gap by fusing reference appearance and motion cues into a unified latent space, enabling the model to jointly reason about spatial identity and temporal dynamics by leveraging the generative prior of foundational models. Second, we introduce a self-bootstrapped data synthesis pipeline that curates pseudo cross-identity training pairs, facilitating a seamless transition from pose-dependent control to direct, end-to-end RGB-driven animation. This strategy significantly enhances generalization across diverse characters and motion scenarios. To facilitate comprehensive evaluation, we further introduce AW Bench, a versatile benchmark encompassing a wide spectrum of characters types and motion scenarios. Extensive experiments demonstrate that DreamActor-M2 achieves state-of-the-art performance, delivering superior visual fidelity and robust cross-domain generalization. Project Page: https://grisoon.github.io/DreamActor-M2/
>
---
#### [new 088] MPF-Net: Exposing High-Fidelity AI-Generated Video Forgeries via Hierarchical Manifold Deviation and Micro-Temporal Fluctuations
- **分类: cs.CV**

- **简介: 该论文属于视频伪造检测任务，旨在识别高保真AI生成视频。通过分析像素结构和时间波动，提出MPF-Net框架以暴露伪造痕迹。**

- **链接: [https://arxiv.org/pdf/2601.21408v1](https://arxiv.org/pdf/2601.21408v1)**

> **作者:** Xinan He; Kaiqing Lin; Yue Zhou; Jiaming Zhong; Wei Ye; Wenhui Yi; Bing Fan; Feng Ding; Haodong Li; Bo Cao; Bin Li
>
> **摘要:** With the rapid advancement of video generation models such as Veo and Wan, the visual quality of synthetic content has reached a level where macro-level semantic errors and temporal inconsistencies are no longer prominent. However, this does not imply that the distinction between real and cutting-edge high-fidelity fake is untraceable. We argue that AI-generated videos are essentially products of a manifold-fitting process rather than a physical recording. Consequently, the pixel composition logic of consecutive adjacent frames residual in AI videos exhibits a structured and homogenous characteristic. We term this phenomenon `Manifold Projection Fluctuations' (MPF). Driven by this insight, we propose a hierarchical dual-path framework that operates as a sequential filtering process. The first, the Static Manifold Deviation Branch, leverages the refined perceptual boundaries of Large-Scale Vision Foundation Models (VFMs) to capture residual spatial anomalies or physical violations that deviate from the natural real-world manifold (off-manifold). For the remaining high-fidelity videos that successfully reside on-manifold and evade spatial detection, we introduce the Micro-Temporal Fluctuation Branch as a secondary, fine-grained filter. By analyzing the structured MPF that persists even in visually perfect sequences, our framework ensures that forgeries are exposed regardless of whether they manifest as global real-world manifold deviations or subtle computational fingerprints.
>
---
#### [new 089] SINA: A Circuit Schematic Image-to-Netlist Generator Using Artificial Intelligence
- **分类: cs.CV; cs.AI; eess.SY**

- **简介: 该论文属于电路图到网表生成任务，解决组件识别与连接推理问题。提出SINA系统，结合深度学习、CCL、OCR和VLM，提升网表生成准确率至96.47%。**

- **链接: [https://arxiv.org/pdf/2601.22114v1](https://arxiv.org/pdf/2601.22114v1)**

> **作者:** Saoud Aldowaish; Yashwanth Karumanchi; Kai-Chen Chiang; Soroosh Noorzad; Morteza Fayazi
>
> **摘要:** Current methods for converting circuit schematic images into machine-readable netlists struggle with component recognition and connectivity inference. In this paper, we present SINA, an open-source, fully automated circuit schematic image-to-netlist generator. SINA integrates deep learning for accurate component detection, Connected-Component Labeling (CCL) for precise connectivity extraction, and Optical Character Recognition (OCR) for component reference designator retrieval, while employing a Vision-Language Model (VLM) for reliable reference designator assignments. In our experiments, SINA achieves 96.47% overall netlist-generation accuracy, which is 2.72x higher than state-of-the-art approaches.
>
---
#### [new 090] Lightweight High-Fidelity Low-Bitrate Talking Face Compression for 3D Video Conference
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D视频会议任务，旨在解决低比特率下高保真人脸压缩的问题。通过结合FLAME模型与3DGS神经渲染，提出轻量高效压缩框架。**

- **链接: [https://arxiv.org/pdf/2601.21269v1](https://arxiv.org/pdf/2601.21269v1)**

> **作者:** Jianglong Li; Jun Xu; Bingcong Lu; Zhengxue Cheng; Hongwei Hu; Ronghua Wu; Li Song
>
> **摘要:** The demand for immersive and interactive communication has driven advancements in 3D video conferencing, yet achieving high-fidelity 3D talking face representation at low bitrates remains a challenge. Traditional 2D video compression techniques fail to preserve fine-grained geometric and appearance details, while implicit neural rendering methods like NeRF suffer from prohibitive computational costs. To address these challenges, we propose a lightweight, high-fidelity, low-bitrate 3D talking face compression framework that integrates FLAME-based parametric modeling with 3DGS neural rendering. Our approach transmits only essential facial metadata in real time, enabling efficient reconstruction with a Gaussian-based head model. Additionally, we introduce a compact representation and compression scheme, including Gaussian attribute compression and MLP optimization, to enhance transmission efficiency. Experimental results demonstrate that our method achieves superior rate-distortion performance, delivering high-quality facial rendering at extremely low bitrates, making it well-suited for real-time 3D video conferencing applications.
>
---
#### [new 091] WMVLM: Evaluating Diffusion Model Image Watermarking via Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于图像水印评估任务，旨在解决现有方法在统一性、可解释性和安全性方面的不足。提出WMVLM框架，利用视觉语言模型实现更准确的水印评价。**

- **链接: [https://arxiv.org/pdf/2601.21610v1](https://arxiv.org/pdf/2601.21610v1)**

> **作者:** Zijin Yang; Yu Sun; Kejiang Chen; Jiawei Zhao; Jun Jiang; Weiming Zhang; Nenghai Yu
>
> **摘要:** Digital watermarking is essential for securing generated images from diffusion models. Accurate watermark evaluation is critical for algorithm development, yet existing methods have significant limitations: they lack a unified framework for both residual and semantic watermarks, provide results without interpretability, neglect comprehensive security considerations, and often use inappropriate metrics for semantic watermarks. To address these gaps, we propose WMVLM, the first unified and interpretable evaluation framework for diffusion model image watermarking via vision-language models (VLMs). We redefine quality and security metrics for each watermark type: residual watermarks are evaluated by artifact strength and erasure resistance, while semantic watermarks are assessed through latent distribution shifts. Moreover, we introduce a three-stage training strategy to progressively enable the model to achieve classification, scoring, and interpretable text generation. Experiments show WMVLM outperforms state-of-the-art VLMs with strong generalization across datasets, diffusion models, and watermarking methods.
>
---
#### [new 092] From Global to Granular: Revealing IQA Model Performance via Correlation Surface
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像质量评估（IQA）任务，旨在解决传统全局相关性指标无法反映模型在局部质量区间表现差异的问题。提出GMC方法，通过构建相关性表面实现细粒度性能分析。**

- **链接: [https://arxiv.org/pdf/2601.21738v1](https://arxiv.org/pdf/2601.21738v1)**

> **作者:** Baoliang Chen; Danni Huang; Hanwei Zhu; Lingyu Zhu; Wei Zhou; Shiqi Wang; Yuming Fang; Weisi Lin
>
> **摘要:** Evaluation of Image Quality Assessment (IQA) models has long been dominated by global correlation metrics, such as Pearson Linear Correlation Coefficient (PLCC) and Spearman Rank-Order Correlation Coefficient (SRCC). While widely adopted, these metrics reduce performance to a single scalar, failing to capture how ranking consistency varies across the local quality spectrum. For example, two IQA models may achieve identical SRCC values, yet one ranks high-quality images (related to high Mean Opinion Score, MOS) more reliably, while the other better discriminates image pairs with small quality/MOS differences (related to $|Δ$MOS$|$). Such complementary behaviors are invisible under global metrics. Moreover, SRCC and PLCC are sensitive to test-sample quality distributions, yielding unstable comparisons across test sets. To address these limitations, we propose \textbf{Granularity-Modulated Correlation (GMC)}, which provides a structured, fine-grained analysis of IQA performance. GMC includes: (1) a \textbf{Granularity Modulator} that applies Gaussian-weighted correlations conditioned on absolute MOS values and pairwise MOS differences ($|Δ$MOS$|$) to examine local performance variations, and (2) a \textbf{Distribution Regulator} that regularizes correlations to mitigate biases from non-uniform quality distributions. The resulting \textbf{correlation surface} maps correlation values as a joint function of MOS and $|Δ$MOS$|$, providing a 3D representation of IQA performance. Experiments on standard benchmarks show that GMC reveals performance characteristics invisible to scalar metrics, offering a more informative and reliable paradigm for analyzing, comparing, and deploying IQA models. Codes are available at https://github.com/Dniaaa/GMC.
>
---
#### [new 093] DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于动态物体操作任务，解决VLA模型在动态场景中的感知、预测和控制问题。提出DynamicVLA框架，结合时间推理与闭环适应，提升动态操作性能。**

- **链接: [https://arxiv.org/pdf/2601.22153v1](https://arxiv.org/pdf/2601.22153v1)**

> **作者:** Haozhe Xie; Beichen Wen; Jiarui Zheng; Zhaoxi Chen; Fangzhou Hong; Haiwen Diao; Ziwei Liu
>
> **备注:** Project Page: https://www.infinitescript.com/project/dynamic-vla/ GitHub: https://github.com/hzxie/DynamicVLA
>
> **摘要:** Manipulating dynamic objects remains an open challenge for Vision-Language-Action (VLA) models, which, despite strong generalization in static manipulation, struggle in dynamic scenarios requiring rapid perception, temporal anticipation, and continuous control. We present DynamicVLA, a framework for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptation through three key designs: 1) a compact 0.4B VLA using a convolutional vision encoder for spatially efficient, structurally faithful encoding, enabling fast multimodal inference; 2) Continuous Inference, enabling overlapping reasoning and execution for lower latency and timely adaptation to object motion; and 3) Latent-aware Action Streaming, which bridges the perception-execution gap by enforcing temporally aligned action execution. To fill the missing foundation of dynamic manipulation data, we introduce the Dynamic Object Manipulation (DOM) benchmark, built from scratch with an auto data collection pipeline that efficiently gathers 200K synthetic episodes across 2.8K scenes and 206 objects, and enables fast collection of 2K real-world episodes without teleoperation. Extensive evaluations demonstrate remarkable improvements in response speed, perception, and generalization, positioning DynamicVLA as a unified framework for general dynamic object manipulation across embodiments.
>
---
#### [new 094] ViTMAlis: Towards Latency-Critical Mobile Video Analytics with Vision Transformers
- **分类: cs.NI; cs.CV; cs.MM**

- **简介: 该论文属于移动视频分析任务，解决ViT在延迟敏感场景中的高推理延迟问题。提出动态混合分辨率策略和ViTMAlis框架，降低传输与推理延迟，提升准确性。**

- **链接: [https://arxiv.org/pdf/2601.21362v1](https://arxiv.org/pdf/2601.21362v1)**

> **作者:** Miao Zhang; Guanzhen Wu; Hao Fang; Yifei Zhu; Fangxin Wang; Ruixiao Zhang; Jiangchuan Liu
>
> **摘要:** Edge-assisted mobile video analytics (MVA) applications are increasingly shifting from using vision models based on convolutional neural networks (CNNs) to those built on vision transformers (ViTs) to leverage their superior global context modeling and generalization capabilities. However, deploying these advanced models in latency-critical MVA scenarios presents significant challenges. Unlike traditional CNN-based offloading paradigms where network transmission is the primary bottleneck, ViT-based systems are constrained by substantial inference delays, particularly for dense prediction tasks where the need for high-resolution inputs exacerbates the inherent quadratic computational complexity of ViTs. To address these challenges, we propose a dynamic mixed-resolution inference strategy tailored for ViT-backboned dense prediction models, enabling flexible runtime trade-offs between speed and accuracy. Building on this, we introduce ViTMAlis, a ViT-native device-to-edge offloading framework that dynamically adapts to network conditions and video content to jointly reduce transmission and inference delays. We implement a fully functional prototype of ViTMAlis on commodity mobile and edge devices. Extensive experiments demonstrate that, compared to state-of-the-art accuracy-centric, content-aware, and latency-adaptive baselines, ViTMAlis significantly reduces end-to-end offloading latency while improving user-perceived rendering accuracy, providing a practical foundation for next-generation mobile intelligence.
>
---
#### [new 095] JUST-DUB-IT: Video Dubbing via Joint Audio-Visual Diffusion
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于视频配音任务，解决传统方法依赖复杂流程的问题。提出一种基于音频-视觉扩散模型的单模型方法，通过轻量LoRA实现高质量配音，提升唇形同步与视觉质量。**

- **链接: [https://arxiv.org/pdf/2601.22143v1](https://arxiv.org/pdf/2601.22143v1)**

> **作者:** Anthony Chen; Naomi Ken Korem; Tavi Halperin; Matan Ben Yosef; Urska Jelercic; Ofir Bibi; Or Patashnik; Daniel Cohen-Or
>
> **备注:** Project webpage available at https://justdubit.github.io
>
> **摘要:** Audio-Visual Foundation Models, which are pretrained to jointly generate sound and visual content, have recently shown an unprecedented ability to model multi-modal generation and editing, opening new opportunities for downstream tasks. Among these tasks, video dubbing could greatly benefit from such priors, yet most existing solutions still rely on complex, task-specific pipelines that struggle in real-world settings. In this work, we introduce a single-model approach that adapts a foundational audio-video diffusion model for video-to-video dubbing via a lightweight LoRA. The LoRA enables the model to condition on an input audio-video while jointly generating translated audio and synchronized facial motion. To train this LoRA, we leverage the generative model itself to synthesize paired multilingual videos of the same speaker. Specifically, we generate multilingual videos with language switches within a single clip, and then inpaint the face and audio in each half to match the language of the other half. By leveraging the rich generative prior of the audio-visual model, our approach preserves speaker identity and lip synchronization while remaining robust to complex motion and real-world dynamics. We demonstrate that our approach produces high-quality dubbed videos with improved visual fidelity, lip synchronization, and robustness compared to existing dubbing pipelines.
>
---
#### [new 096] Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data
- **分类: cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于模型剪枝任务，旨在解决数据异质性下单一子网络性能不足的问题。提出RTL框架，发现多个适应不同类别的子网络，提升准确率并减少参数。**

- **链接: [https://arxiv.org/pdf/2601.22141v1](https://arxiv.org/pdf/2601.22141v1)**

> **作者:** Grzegorz Stefanski; Alberto Presta; Michal Byra
>
> **摘要:** In pruning, the Lottery Ticket Hypothesis posits that large networks contain sparse subnetworks, or winning tickets, that can be trained in isolation to match the performance of their dense counterparts. However, most existing approaches assume a single universal winning ticket shared across all inputs, ignoring the inherent heterogeneity of real-world data. In this work, we propose Routing the Lottery (RTL), an adaptive pruning framework that discovers multiple specialized subnetworks, called adaptive tickets, each tailored to a class, semantic cluster, or environmental condition. Across diverse datasets and tasks, RTL consistently outperforms single- and multi-model baselines in balanced accuracy and recall, while using up to 10 times fewer parameters than independent models and exhibiting semantically aligned. Furthermore, we identify subnetwork collapse, a performance drop under aggressive pruning, and introduce a subnetwork similarity score that enables label-free diagnosis of oversparsification. Overall, our results recast pruning as a mechanism for aligning model structure with data heterogeneity, paving the way toward more modular and context-aware deep learning.
>
---
#### [new 097] Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于统计评估任务，解决LLM在不完美裁判下的可靠性验证问题。通过构建噪声有效框架，提升评估的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.20913v1](https://arxiv.org/pdf/2601.20913v1)**

> **作者:** Chen Feng; Minghe Shen; Ananth Balashankar; Carsten Gerner-Beuerle; Miguel R. D. Rodrigues
>
> **备注:** Accepted to ICLR2026
>
> **摘要:** Reliable certification of Large Language Models (LLMs)-verifying that failure rates are below a safety threshold-is critical yet challenging. While "LLM-as-a-Judge" offers scalability, judge imperfections, noise, and bias can invalidate statistical guarantees. We introduce a "Noisy but Valid" hypothesis testing framework to address this. By leveraging a small human-labelled calibration set to estimate the judge's True Positive and False Positive Rates (TPR/FPR), we derive a variance-corrected critical threshold applied to a large judge-labelled dataset. Crucially, our framework theoretically guarantees finite-sample Type-I error control (validity) despite calibration uncertainty. This distinguishes our work from Prediction-Powered Inference (PPI), positioning our method as a diagnostic tool that explicitly models judge behavior rather than a black-box estimator. Our contributions include: (1) Theoretical Guarantees: We derive the exact conditions under which noisy testing yields higher statistical power than direct evaluation; (2) Empirical Validation: Experiments on Jigsaw Comment, Hate Speech and SafeRLHF confirm our theory; (3) The Oracle Gap: We reveal a significant performance gap between practical methods and the theoretical "Oracle" (perfectly known judge parameters), quantifying the cost of estimation. Specifically, we provide the first systematic treatment of the imperfect-judge setting, yielding interpretable diagnostics of judge reliability and clarifying how evaluation power depends on judge quality, dataset size, and certification levels. Together, these results sharpen understanding of statistical evaluation with LLM judges, and highlight trade-offs among competing inferential tools.
>
---
#### [new 098] 4D-CAAL: 4D Radar-Camera Calibration and Auto-Labeling for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多模态感知任务，旨在解决4D雷达与相机的标定及自动标注问题。提出4D-CAAL框架，设计双用途标定靶，实现精准标定与高效标注。**

- **链接: [https://arxiv.org/pdf/2601.21454v1](https://arxiv.org/pdf/2601.21454v1)**

> **作者:** Shanliang Yao; Zhuoxiao Li; Runwei Guan; Kebin Cao; Meng Xia; Fuping Hu; Sen Xu; Yong Yue; Xiaohui Zhu; Weiping Ding; Ryan Wen Liu
>
> **摘要:** 4D radar has emerged as a critical sensor for autonomous driving, primarily due to its enhanced capabilities in elevation measurement and higher resolution compared to traditional 3D radar. Effective integration of 4D radar with cameras requires accurate extrinsic calibration, and the development of radar-based perception algorithms demands large-scale annotated datasets. However, existing calibration methods often employ separate targets optimized for either visual or radar modalities, complicating correspondence establishment. Furthermore, manually labeling sparse radar data is labor-intensive and unreliable. To address these challenges, we propose 4D-CAAL, a unified framework for 4D radar-camera calibration and auto-labeling. Our approach introduces a novel dual-purpose calibration target design, integrating a checkerboard pattern on the front surface for camera detection and a corner reflector at the center of the back surface for radar detection. We develop a robust correspondence matching algorithm that aligns the checkerboard center with the strongest radar reflection point, enabling accurate extrinsic calibration. Subsequently, we present an auto-labeling pipeline that leverages the calibrated sensor relationship to transfer annotations from camera-based segmentations to radar point clouds through geometric projection and multi-feature optimization. Extensive experiments demonstrate that our method achieves high calibration accuracy while significantly reducing manual annotation effort, thereby accelerating the development of robust multi-modal perception systems for autonomous driving.
>
---
#### [new 099] Learning to Communicate Across Modalities: Perceptual Heterogeneity in Multi-Agent Systems
- **分类: cs.MA; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究多智能体系统中跨模态通信问题，探讨感知异质性下的信息编码与传递机制。**

- **链接: [https://arxiv.org/pdf/2601.22041v1](https://arxiv.org/pdf/2601.22041v1)**

> **作者:** Naomi Pitzer; Daniela Mihai
>
> **备注:** To be published in EvoLang XVI proceedings. 15 pages, 17 figures
>
> **摘要:** Emergent communication offers insight into how agents develop shared structured representations, yet most research assumes homogeneous modalities or aligned representational spaces, overlooking the perceptual heterogeneity of real-world settings. We study a heterogeneous multi-step binary communication game where agents differ in modality and lack perceptual grounding. Despite perceptual misalignment, multimodal systems converge to class-consistent messages grounded in perceptual input. Unimodal systems communicate more efficiently, using fewer bits and achieving lower classification entropy, while multimodal agents require greater information exchange and exhibit higher uncertainty. Bit perturbation experiments provide strong evidence that meaning is encoded in a distributional rather than compositional manner, as each bit's contribution depends on its surrounding pattern. Finally, interoperability analyses show that systems trained in different perceptual worlds fail to directly communicate, but limited fine-tuning enables successful cross-system communication. This work positions emergent communication as a framework for studying how agents adapt and transfer representations across heterogeneous modalities, opening new directions for both theory and experimentation.
>
---
#### [new 100] SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出SONIC-O1基准，用于评估多模态大语言模型在音视频理解中的表现。针对现有研究多聚焦静态图像、缺乏真实场景下音视频处理评估的问题，论文构建了涵盖13个对话领域的高质量数据集，涵盖多种任务，揭示模型性能差异与社会公平性问题。**

- **链接: [https://arxiv.org/pdf/2601.21666v1](https://arxiv.org/pdf/2601.21666v1)**

> **作者:** Ahmed Y. Radwan; Christos Emmanouilidis; Hina Tabassum; Deval Pandya; Shaina Raza
>
> **摘要:** Multimodal Large Language Models (MLLMs) are a major focus of recent AI research. However, most prior work focuses on static image understanding, while their ability to process sequential audio-video data remains underexplored. This gap highlights the need for a high-quality benchmark to systematically evaluate MLLM performance in a real-world setting. We introduce SONIC-O1, a comprehensive, fully human-verified benchmark spanning 13 real-world conversational domains with 4,958 annotations and demographic metadata. SONIC-O1 evaluates MLLMs on key tasks, including open-ended summarization, multiple-choice question (MCQ) answering, and temporal localization with supporting rationales (reasoning). Experiments on closed- and open-source models reveal limitations. While the performance gap in MCQ accuracy between two model families is relatively small, we observe a substantial 22.6% performance difference in temporal localization between the best performing closed-source and open-source models. Performance further degrades across demographic groups, indicating persistent disparities in model behavior. Overall, SONIC-O1 provides an open evaluation suite for temporally grounded and socially robust multimodal understanding. We release SONIC-O1 for reproducibility and research: Project page: https://vectorinstitute.github.io/sonic-o1/ Dataset: https://huggingface.co/datasets/vector-institute/sonic-o1 Github: https://github.com/vectorinstitute/sonic-o1 Leaderboard: https://huggingface.co/spaces/vector-institute/sonic-o1-leaderboard
>
---
#### [new 101] Hybrid Foveated Path Tracing with Peripheral Gaussians for Immersive Anatomy
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于医学可视化任务，旨在解决传统方法在交互性与视觉质量上的不足。提出混合渲染方法，结合路径追踪与高斯点云，实现实时、高质量的解剖可视化。**

- **链接: [https://arxiv.org/pdf/2601.22026v1](https://arxiv.org/pdf/2601.22026v1)**

> **作者:** Constantin Kleinbeck; Luisa Theelke; Hannah Schieber; Ulrich Eck; Rüdiger von Eisenhart-Rothe; Daniel Roth
>
> **备注:** Scheduled for publication in the Proceedings of IEEE VR 2026
>
> **摘要:** Volumetric medical imaging offers great potential for understanding complex pathologies. Yet, traditional 2D slices provide little support for interpreting spatial relationships, forcing users to mentally reconstruct anatomy into three dimensions. Direct volumetric path tracing and VR rendering can improve perception but are computationally expensive, while precomputed representations, like Gaussian Splatting, require planning ahead. Both approaches limit interactive use. We propose a hybrid rendering approach for high-quality, interactive, and immersive anatomical visualization. Our method combines streamed foveated path tracing with a lightweight Gaussian Splatting approximation of the periphery. The peripheral model generation is optimized with volume data and continuously refined using foveal renderings, enabling interactive updates. Depth-guided reprojection further improves robustness to latency and allows users to balance fidelity with refresh rate. We compare our method against direct path tracing and Gaussian Splatting. Our results highlight how their combination can preserve strengths in visual quality while re-generating the peripheral model in under a second, eliminating extensive preprocessing and approximations. This opens new options for interactive medical visualization.
>
---
#### [new 102] InspecSafe-V1: A Multimodal Benchmark for Safety Assessment in Industrial Inspection Scenarios
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出InspecSafe-V1基准数据集，用于工业检测安全评估。解决真实场景下多模态感知与安全分析的问题，包含多种传感器数据和精细标注。**

- **链接: [https://arxiv.org/pdf/2601.21173v1](https://arxiv.org/pdf/2601.21173v1)**

> **作者:** Zeyi Liu; Shuang Liu; Jihai Min; Zhaoheng Zhang; Jun Cen; Pengyu Han; Songqiao Hu; Zihan Meng; Xiao He; Donghua Zhou
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** With the rapid development of industrial intelligence and unmanned inspection, reliable perception and safety assessment for AI systems in complex and dynamic industrial sites has become a key bottleneck for deploying predictive maintenance and autonomous inspection. Most public datasets remain limited by simulated data sources, single-modality sensing, or the absence of fine-grained object-level annotations, which prevents robust scene understanding and multimodal safety reasoning for industrial foundation models. To address these limitations, InspecSafe-V1 is released as the first multimodal benchmark dataset for industrial inspection safety assessment that is collected from routine operations of real inspection robots in real-world environments. InspecSafe-V1 covers five representative industrial scenarios, including tunnels, power facilities, sintering equipment, oil and gas petrochemical plants, and coal conveyor trestles. The dataset is constructed from 41 wheeled and rail-mounted inspection robots operating at 2,239 valid inspection sites, yielding 5,013 inspection instances. For each instance, pixel-level segmentation annotations are provided for key objects in visible-spectrum images. In addition, a semantic scene description and a corresponding safety level label are provided according to practical inspection tasks. Seven synchronized sensing modalities are further included, including infrared video, audio, depth point clouds, radar point clouds, gas measurements, temperature, and humidity, to support multimodal anomaly recognition, cross-modal fusion, and comprehensive safety assessment in industrial environments.
>
---
#### [new 103] Visual-Guided Key-Token Regularization for Multimodal Large Language Model Unlearning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于多模态大语言模型的遗忘任务，旨在解决模型泄露私有信息的问题。提出ViKeR方法，通过视觉引导优化关键标记的遗忘过程，提升效果并减少遗忘。**

- **链接: [https://arxiv.org/pdf/2601.22020v1](https://arxiv.org/pdf/2601.22020v1)**

> **作者:** Chengyi Cai; Zesheng Ye; Peike Li; Bo Han; Jianzhong Qi; Feng Liu
>
> **摘要:** Unlearning in Multimodal Large Language Models (MLLMs) prevents the model from revealing private information when queried about target images. Existing MLLM unlearning methods largely adopt approaches developed for LLMs. They treat all answer tokens uniformly, disregarding their varying importance in the unlearning process. Moreover, these methods focus exclusively on the language modality, disregarding visual cues that indicate key tokens in answers. In this paper, after formulating the problem of unlearning in multimodal question answering for MLLMs, we propose Visual-Guided Key-Token Regularization (ViKeR). We leverage irrelevant visual inputs to predict ideal post-unlearning token-level distributions and use these distributions to regularize the unlearning process, thereby prioritizing key tokens. Further, we define key tokens in unlearning via information entropy and discuss ViKeR's effectiveness through token-level gradient reweighting, which amplifies updates on key tokens. Experiments on MLLMU and CLEAR benchmarks demonstrate that our method effectively performs unlearning while mitigating forgetting and maintaining response coherence.
>
---
#### [new 104] Lossy Common Information in a Learnable Gray-Wyner Network
- **分类: cs.LG; cs.CV; cs.IT**

- **简介: 该论文属于多任务视觉编码领域，旨在解决冗余表示问题。通过构建可学习的Gray-Wyner网络，分离共享与任务特定信息，提升编码效率。**

- **链接: [https://arxiv.org/pdf/2601.21424v1](https://arxiv.org/pdf/2601.21424v1)**

> **作者:** Anderson de Andrade; Alon Harell; Ivan V. Bajić
>
> **摘要:** Many computer vision tasks share substantial overlapping information, yet conventional codecs tend to ignore this, leading to redundant and inefficient representations. The Gray-Wyner network, a classical concept from information theory, offers a principled framework for separating common and task-specific information. Inspired by this idea, we develop a learnable three-channel codec that disentangles shared information from task-specific details across multiple vision tasks. We characterize the limits of this approach through the notion of lossy common information, and propose an optimization objective that balances inherent tradeoffs in learning such representations. Through comparisons of three codec architectures on two-task scenarios spanning six vision benchmarks, we demonstrate that our approach substantially reduces redundancy and consistently outperforms independent coding. These results highlight the practical value of revisiting Gray-Wyner theory in modern machine learning contexts, bridging classic information theory with task-driven representation learning.
>
---
#### [new 105] Learning Transient Convective Heat Transfer with Geometry Aware World Models
- **分类: physics.flu-dyn; cs.CV**

- **简介: 该论文属于物理模拟任务，旨在解决实时PDE仿真计算成本高的问题。通过改进的生成模型学习瞬态对流传热，提升模拟效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.22086v1](https://arxiv.org/pdf/2601.22086v1)**

> **作者:** Onur T. Doganay; Alexander Klawonn; Martin Eigel; Hanno Gottschalk
>
> **备注:** 36 pages, 18 figures, 2 tables
>
> **摘要:** Partial differential equation (PDE) simulations are fundamental to engineering and physics but are often computationally prohibitive for real-time applications. While generative AI offers a promising avenue for surrogate modeling, standard video generation architectures lack the specific control and data compatibility required for physical simulations. This paper introduces a geometry aware world model architecture, derived from a video generation architecture (LongVideoGAN), designed to learn transient physics. We introduce two key architecture elements: (1) a twofold conditioning mechanism incorporating global physical parameters and local geometric masks, and (2) an architectural adaptation to support arbitrary channel dimensions, moving beyond standard RGB constraints. We evaluate this approach on a 2D transient computational fluid dynamics (CFD) problem involving convective heat transfer from buoyancy-driven flow coupled to a heat flow in a solid structure. We demonstrate that the conditioned model successfully reproduces complex temporal dynamics and spatial correlations of the training data. Furthermore, we assess the model's generalization capabilities on unseen geometric configurations, highlighting both its potential for controlled simulation synthesis and current limitations in spatial precision for out-of-distribution samples.
>
---
#### [new 106] Revisiting Diffusion Model Predictions Through Dimensionality
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成模型任务，解决扩散模型预测目标选择问题。通过理论分析与数据驱动方法，提出k-Diff框架，提升生成性能。**

- **链接: [https://arxiv.org/pdf/2601.21419v1](https://arxiv.org/pdf/2601.21419v1)**

> **作者:** Qing Jin; Chaoyang Wang
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Recent advances in diffusion and flow matching models have highlighted a shift in the preferred prediction target -- moving from noise ($\varepsilon$) and velocity (v) to direct data (x) prediction -- particularly in high-dimensional settings. However, a formal explanation of why the optimal target depends on the specific properties of the data remains elusive. In this work, we provide a theoretical framework based on a generalized prediction formulation that accommodates arbitrary output targets, of which $\varepsilon$-, v-, and x-prediction are special cases. We derive the analytical relationship between data's geometry and the optimal prediction target, offering a rigorous justification for why x-prediction becomes superior when the ambient dimension significantly exceeds the data's intrinsic dimension. Furthermore, while our theory identifies dimensionality as the governing factor for the optimal prediction target, the intrinsic dimension of manifold-bound data is typically intractable to estimate in practice. To bridge this gap, we propose k-Diff, a framework that employs a data-driven approach to learn the optimal prediction parameter k directly from data, bypassing the need for explicit dimension estimation. Extensive experiments in both latent-space and pixel-space image generation demonstrate that k-Diff consistently outperforms fixed-target baselines across varying architectures and data scales, providing a principled and automated approach to enhancing generative performance.
>
---
#### [new 107] Thinking in Frames: How Visual Context and Test-Time Scaling Empower Video Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉推理任务，旨在解决视频中细粒度空间理解和连续动作规划问题。通过视频生成模型，探索视觉上下文与测试时缩放对推理能力的影响。**

- **链接: [https://arxiv.org/pdf/2601.21037v1](https://arxiv.org/pdf/2601.21037v1)**

> **作者:** Chengzu Li; Zanyi Wang; Jiaang Li; Yi Xu; Han Zhou; Huanyu Zhang; Ruichuan An; Dengyang Jiang; Zhaochong An; Ivan Vulić; Serge Belongie; Anna Korhonen
>
> **备注:** 8 pages, 3 figures, 3 tables (26 pages, 13 figures, 6 tables including references and appendices)
>
> **摘要:** Vision-Language Models have excelled at textual reasoning, but they often struggle with fine-grained spatial understanding and continuous action planning, failing to simulate the dynamics required for complex visual reasoning. In this work, we formulate visual reasoning by means of video generation models, positing that generated frames can act as intermediate reasoning steps between initial states and solutions. We evaluate their capacity in two distinct regimes: Maze Navigation for sequential discrete planning with low visual change and Tangram Puzzle for continuous manipulation with high visual change. Our experiments reveal three critical insights: (1) Robust Zero-Shot Generalization: In both tasks, the model demonstrates strong performance on unseen data distributions without specific finetuning. (2) Visual Context: The model effectively uses visual context as explicit control, such as agent icons and tangram shapes, enabling it to maintain high visual consistency and adapt its planning capability robustly to unseen patterns. (3) Visual Test-Time Scaling: We observe a test-time scaling law in sequential planning; increasing the generated video length (visual inference budget) empowers better zero-shot generalization to spatially and temporally complex paths. These findings suggest that video generation is not merely a media tool, but a scalable, generalizable paradigm for visual reasoning.
>
---
#### [new 108] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 该论文研究视觉语言模型在视觉令牌压缩下的对抗鲁棒性问题，提出CAGE攻击方法，以更准确评估压缩模型的脆弱性。**

- **链接: [https://arxiv.org/pdf/2601.21531v1](https://arxiv.org/pdf/2601.21531v1)**

> **作者:** Xinwei Zhang; Hangcheng Liu; Li Bai; Hao Wang; Qingqing Ye; Tianwei Zhang; Haibo Hu
>
> **备注:** Under Review, 20 pages
>
> **摘要:** Visual token compression is widely used to accelerate large vision-language models (LVLMs) by pruning or merging visual tokens, yet its adversarial robustness remains unexplored. We show that existing encoder-based attacks can substantially overestimate the robustness of compressed LVLMs, due to an optimization-inference mismatch: perturbations are optimized on the full-token representation, while inference is performed through a token-compression bottleneck. To address this gap, we propose the Compression-AliGnEd attack (CAGE), which aligns perturbation optimization with compression inference without assuming access to the deployed compression mechanism or its token budget. CAGE combines (i) expected feature disruption, which concentrates distortion on tokens likely to survive across plausible budgets, and (ii) rank distortion alignment, which actively aligns token distortions with rank scores to promote the retention of highly distorted evidence. Across diverse representative plug-and-play compression mechanisms and datasets, our results show that CAGE consistently achieves lower robust accuracy than the baseline. This work highlights that robustness assessments ignoring compression can be overly optimistic, calling for compression-aware security evaluation and defenses for efficient LVLMs.
>
---
#### [new 109] Drive-KD: Multi-Teacher Distillation for VLMs in Autonomous Driving
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决大模型资源消耗高与小模型性能不足的问题。通过知识蒸馏方法，提升小模型性能，实现高效自动驾驶视觉语言模型。**

- **链接: [https://arxiv.org/pdf/2601.21288v1](https://arxiv.org/pdf/2601.21288v1)**

> **作者:** Weitong Lian; Zecong Tang; Haoran Li; Tianjian Gao; Yifei Wang; Zixu Wang; Lingyi Meng; Tengju Ru; Zhejun Cui; Yichen Zhu; Hangshuo Cao; Qi Kang; Tianxing Chen; Yusen Qin; Kaixuan Wang; Yu Zhang
>
> **备注:** Preprint. 23 pages, 14 figures
>
> **摘要:** Autonomous driving is an important and safety-critical task, and recent advances in LLMs/VLMs have opened new possibilities for reasoning and planning in this domain. However, large models demand substantial GPU memory and exhibit high inference latency, while conventional supervised fine-tuning (SFT) often struggles to bridge the capability gaps of small models. To address these limitations, we propose Drive-KD, a framework that decomposes autonomous driving into a "perception-reasoning-planning" triad and transfers these capabilities via knowledge distillation. We identify layer-specific attention as the distillation signal to construct capability-specific single-teacher models that outperform baselines. Moreover, we unify these single-teacher settings into a multi-teacher distillation framework and introduce asymmetric gradient projection to mitigate cross-capability gradient conflicts. Extensive evaluations validate the generalization of our method across diverse model families and scales. Experiments show that our distilled InternVL3-1B model, with ~42 times less GPU memory and ~11.4 times higher throughput, achieves better overall performance than the pretrained 78B model from the same family on DriveBench, and surpasses GPT-5.1 on the planning dimension, providing insights toward efficient autonomous driving VLMs.
>
---
#### [new 110] From Instruction to Event: Sound-Triggered Mobile Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究声音触发的移动操作任务，解决传统指令驱动方式限制自主性的问题。通过构建数据平台和基线模型，使智能体能主动感知并响应声音事件。**

- **链接: [https://arxiv.org/pdf/2601.21667v1](https://arxiv.org/pdf/2601.21667v1)**

> **作者:** Hao Ju; Shaofei Huang; Hongyu Li; Zihan Ding; Si Liu; Meng Wang; Zhedong Zheng
>
> **摘要:** Current mobile manipulation research predominantly follows an instruction-driven paradigm, where agents rely on predefined textual commands to execute tasks. However, this setting confines agents to a passive role, limiting their autonomy and ability to react to dynamic environmental events. To address these limitations, we introduce sound-triggered mobile manipulation, where agents must actively perceive and interact with sound-emitting objects without explicit action instructions. To support these tasks, we develop Habitat-Echo, a data platform that integrates acoustic rendering with physical interaction. We further propose a baseline comprising a high-level task planner and low-level policy models to complete these tasks. Extensive experiments show that the proposed baseline empowers agents to actively detect and respond to auditory events, eliminating the need for case-by-case instructions. Notably, in the challenging dual-source scenario, the agent successfully isolates the primary source from overlapping acoustic interference to execute the first interaction, and subsequently proceeds to manipulate the secondary object, verifying the robustness of the baseline.
>
---
#### [new 111] Denoising and Baseline Correction of Low-Scan FTIR Spectra: A Benchmark of Deep Learning Models Against Traditional Signal Processing
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于FTIR光谱处理任务，旨在解决低扫描数据的降噪与基线校正问题。通过提出物理信息级联Unet模型，提升诊断级成像速度与质量。**

- **链接: [https://arxiv.org/pdf/2601.20905v1](https://arxiv.org/pdf/2601.20905v1)**

> **作者:** Azadeh Mokari; Shravan Raghunathan; Artem Shydliukh; Oleg Ryabchykov; Christoph Krafft; Thomas Bocklitz
>
> **摘要:** High-quality Fourier Transform Infrared (FTIR) imaging usually needs extensive signal averaging to reduce noise and drift which severely limits clinical speed. Deep learning can accelerate imaging by reconstructing spectra from rapid, single-scan inputs. However, separating noise and baseline drift simultaneously without ground truth is an ill-posed inverse problem. Standard black-box architectures often rely on statistical approximations that introduce spectral hallucinations or fail to generalize to unstable atmospheric conditions. To solve these issues we propose a physics-informed cascade Unet that separates denoising and baseline correction tasks using a new, deterministic Physics Bridge. This architecture forces the network to separate random noise from chemical signals using an embedded SNIP layer to enforce spectroscopic constraints instead of learning statistical approximations. We benchmarked this approach against a standard single Unet and a traditional Savitzky-Golay/SNIP workflow. We used a dataset of human hypopharyngeal carcinoma cells (FaDu). The cascade model outperformed all other methods, achieving a 51.3% reduction in RMSE compared to raw single-scan inputs, surpassing both the single Unet (40.2%) and the traditional workflow (33.7%). Peak-aware metrics show that the cascade architecture eliminates spectral hallucinations found in standard deep learning. It also preserves peak intensity with much higher fidelity than traditional smoothing. These results show that the cascade Unet is a robust solution for diagnostic-grade FTIR imaging. It enables imaging speeds 32 times faster than current methods.
>
---
#### [new 112] Adversarial Vulnerability Transcends Computational Paradigms: Feature Engineering Provides No Defense Against Neural Adversarial Transfer
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于机器学习安全领域，研究对抗样本在不同模型间的迁移问题。工作包括评估特征工程对防御对抗攻击的效果，发现其无法有效抵御攻击。**

- **链接: [https://arxiv.org/pdf/2601.21323v1](https://arxiv.org/pdf/2601.21323v1)**

> **作者:** Achraf Hsain; Ahmed Abdelkader; Emmanuel Baldwin Mbaya; Hamoud Aljamaan
>
> **摘要:** Deep neural networks are vulnerable to adversarial examples--inputs with imperceptible perturbations causing misclassification. While adversarial transfer within neural networks is well-documented, whether classical ML pipelines using handcrafted features inherit this vulnerability when attacked via neural surrogates remains unexplored. Feature engineering creates information bottlenecks through gradient quantization and spatial binning, potentially filtering high-frequency adversarial signals. We evaluate this hypothesis through the first comprehensive study of adversarial transfer from DNNs to HOG-based classifiers. Using VGG16 as a surrogate, we generate FGSM and PGD adversarial examples and test transfer to four classical classifiers (KNN, Decision Tree, Linear SVM, Kernel SVM) and a shallow neural network across eight HOG configurations on CIFAR-10. Our results strongly refute the protective hypothesis: all classifiers suffer 16.6%-59.1% relative accuracy drops, comparable to neural-to-neural transfer. More surprisingly, we discover attack hierarchy reversal--contrary to patterns where iterative PGD dominates FGSM within neural networks, FGSM causes greater degradation than PGD in 100% of classical ML cases, suggesting iterative attacks overfit to surrogate-specific features that don't survive feature extraction. Block normalization provides partial but insufficient mitigation. These findings demonstrate that adversarial vulnerability is not an artifact of end-to-end differentiability but a fundamental property of image classification systems, with implications for security-critical deployments across computational paradigms.
>
---
#### [new 113] Blind Ultrasound Image Enhancement via Self-Supervised Physics-Guided Degradation Modeling
- **分类: eess.IV; cs.CV; stat.ML**

- **简介: 该论文属于医学图像增强任务，解决超声图像中的模糊和噪声问题。提出一种自监督框架，结合物理退化模型，实现无监督的去卷积与降噪。**

- **链接: [https://arxiv.org/pdf/2601.21856v1](https://arxiv.org/pdf/2601.21856v1)**

> **作者:** Shujaat Khan; Syed Muhammad Atif; Jaeyoung Huh; Syed Saad Azhar
>
> **备注:** 11 pages, 13 figures
>
> **摘要:** Ultrasound (US) interpretation is hampered by multiplicative speckle, acquisition blur from the point-spread function (PSF), and scanner- and operator-dependent artifacts. Supervised enhancement methods assume access to clean targets or known degradations; conditions rarely met in practice. We present a blind, self-supervised enhancement framework that jointly deconvolves and denoises B-mode images using a Swin Convolutional U-Net trained with a \emph{physics-guided} degradation model. From each training frame, we extract rotated/cropped patches and synthesize inputs by (i) convolving with a Gaussian PSF surrogate and (ii) injecting noise via either spatial additive Gaussian noise or complex Fourier-domain perturbations that emulate phase/magnitude distortions. For US scans, clean-like targets are obtained via non-local low-rank (NLLR) denoising, removing the need for ground truth; for natural images, the originals serve as targets. Trained and validated on UDIAT~B, JNU-IFM, and XPIE Set-P, and evaluated additionally on a 700-image PSFHS test set, the method achieves the highest PSNR/SSIM across Gaussian and speckle noise levels, with margins that widen under stronger corruption. Relative to MSANN, Restormer, and DnCNN, it typically preserves an extra $\sim$1--4\,dB PSNR and 0.05--0.15 SSIM in heavy Gaussian noise, and $\sim$2--5\,dB PSNR and 0.05--0.20 SSIM under severe speckle. Controlled PSF studies show reduced FWHM and higher peak gradients, evidence of resolution recovery without edge erosion. Used as a plug-and-play preprocessor, it consistently boosts Dice for fetal head and pubic symphysis segmentation. Overall, the approach offers a practical, assumption-light path to robust US enhancement that generalizes across datasets, scanners, and degradation types.
>
---
#### [new 114] From Consistency to Complementarity: Aligned and Disentangled Multi-modal Learning for Time Series Understanding and Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于时间序列理解与推理任务，解决多模态对齐与语义解耦问题，提出MADI模型提升跨模态融合效果。**

- **链接: [https://arxiv.org/pdf/2601.21436v1](https://arxiv.org/pdf/2601.21436v1)**

> **作者:** Hang Ni; Weijia Zhang; Fei Wang; Zezhi Shao; Hao Liu
>
> **摘要:** Advances in multi-modal large language models (MLLMs) have inspired time series understanding and reasoning tasks, that enable natural language querying over time series, producing textual analyses of complex temporal dynamics. Recent attempts hybridize numerical time series with their visualized plots, facilitating precise value reasoning and visual structure comprehension for comprehensive time series understanding of MLLMs. However, effective cross-modal integration remains challenging due to fine-grained temporal misalignment across modalities and severe entanglement between shared and modality-specific semantics, which hinder localized interpretation and complementary reasoning. To address these issues, we propose MADI, a multi-modal LLM enhanced with fine-grained alignment and disentangled interaction, featuring (1) Patch-level Alignment, which enforces physically grounded fine-grained correspondence across heterogeneous modalities, (2) Discrete Disentangled Interaction, which separates modality-common semantics into compact discrete latents and adaptively synergizes the purified modality-unique information, and (3) Critical-token Highlighting, which emphasizes informative, query-relevant signals for robust reasoning. Experiments on synthetic and real-world benchmarks show that MADI consistently outperforms general-purpose LLMs and time-series-specialized MLLMs.
>
---
#### [new 115] Lossless Copyright Protection via Intrinsic Model Fingerprinting
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于模型版权保护任务，解决扩散模型被非法复制的问题。提出TrajPrint框架，通过提取模型内在指纹实现无损验证。**

- **链接: [https://arxiv.org/pdf/2601.21252v1](https://arxiv.org/pdf/2601.21252v1)**

> **作者:** Lingxiao Chen; Liqin Wang; Wei Lu; Xiangyang Luo
>
> **摘要:** The exceptional performance of diffusion models establishes them as high-value intellectual property but exposes them to unauthorized replication. Existing protection methods either modify the model to embed watermarks, which impairs performance, or extract model fingerprints by manipulating the denoising process, rendering them incompatible with black-box APIs. In this paper, we propose TrajPrint, a completely lossless and training-free framework that verifies model copyright by extracting unique manifold fingerprints formed during deterministic generation. Specifically, we first utilize a watermarked image as an anchor and exactly trace the path back to its trajectory origin, effectively locking the model fingerprint mapped by this path. Subsequently, we implement a joint optimization strategy that employs dual-end anchoring to synthesize a specific fingerprint noise, which strictly adheres to the target manifold for robust watermark recovery. As input, it enables the protected target model to recover the watermarked image, while failing on non-target models. Finally, we achieved verification via atomic inference and statistical hypothesis testing. Extensive experiments demonstrate that TrajPrint achieves lossless verification in black-box API scenarios with superior robustness against model modifications.
>
---
## 更新

#### [replaced 001] VidLaDA: Bidirectional Diffusion Large Language Models for Efficient Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.17868v2](https://arxiv.org/pdf/2601.17868v2)**

> **作者:** Zhihao He; Tieyuan Chen; Kangyu Wang; Ziran Qin; Yang Shao; Chaofan Gan; Shijie Li; Zuxuan Wu; Weiyao Lin
>
> **摘要:** Current Video Large Language Models (Video LLMs) typically encode frames via a vision encoder and employ an autoregressive (AR) LLM for understanding and generation. However, this AR paradigm inevitably faces a dual efficiency bottleneck: strictly unidirectional attention compromises understanding efficiency by hindering global spatiotemporal aggregation, while serial decoding restricts generation efficiency. To address this, we propose VidLaDA, a Video LLM based on Diffusion Language Models (DLMs) that leverages bidirectional attention to unlock comprehensive spatiotemporal modeling and decode tokens in parallel. To further mitigate the computational overhead of diffusion decoding, we introduce MARS-Cache, an acceleration strategy that prunes redundancy by combining asynchronous visual cache refreshing with frame-wise chunk attention. Experiments show VidLaDA rivals state-of-the-art AR baselines (e.g., Qwen2.5-VL and LLaVA-Video) and outperforms DLM baselines, with MARS-Cache delivering over 12x speedup without compromising accuracy. Code and checkpoints are open-sourced at https://github.com/ziHoHe/VidLaDA.
>
---
#### [replaced 002] GR3EN: Generative Relighting for 3D Environments
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.16272v2](https://arxiv.org/pdf/2601.16272v2)**

> **作者:** Xiaoyan Xing; Philipp Henzler; Junhwa Hur; Runze Li; Jonathan T. Barron; Pratul P. Srinivasan; Dor Verbin
>
> **备注:** project page: https://gr3en-relight.github.io/
>
> **摘要:** We present a method for relighting 3D reconstructions of large room-scale environments. Existing solutions for 3D scene relighting often require solving under-determined or ill-conditioned inverse rendering problems, and are as such unable to produce high-quality results on complex real-world scenes. Though recent progress in using generative image and video diffusion models for relighting has been promising, these techniques are either limited to 2D image and video relighting or 3D relighting of individual objects. Our approach enables controllable 3D relighting of room-scale scenes by distilling the outputs of a video-to-video relighting diffusion model into a 3D reconstruction. This side-steps the need to solve a difficult inverse rendering problem, and results in a flexible system that can relight 3D reconstructions of complex real-world scenes. We validate our approach on both synthetic and real-world datasets to show that it can faithfully render novel views of scenes under new lighting conditions.
>
---
#### [replaced 003] Progressively Deformable 2D Gaussian Splatting for Video Representation at Arbitrary Resolutions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.05600v2](https://arxiv.org/pdf/2503.05600v2)**

> **作者:** Mufan Liu; Qi Yang; Miaoran Zhao; He Huang; Le Yang; Zhu Li; Yiling Xu
>
> **摘要:** Implicit neural representations (INRs) enable fast video compression and effective video processing, but a single model rarely offers scalable decoding across rates and resolutions. In practice, multi-resolution typically relies on retraining or multi-branch designs, and structured pruning failed to provide a permutation-invariant progressive transmission order. Motivated by the explicit structure and efficiency of Gaussian splatting, we propose D2GV-AR, a deformable 2D Gaussian video representation that enables \emph{arbitrary-scale} rendering and \emph{any-ratio} progressive coding within a single model. We partition each video into fixed-length Groups of Pictures and represent each group with a canonical set of 2D Gaussian primitives, whose temporal evolution is modeled by a neural ordinary differential equation. During training and rendering, we apply scale-aware grouping according to Nyquist sampling theorem to form a nested hierarchy across resolutions. Once trained, primitives can be pruned via a D-optimal subset objective to enable any-ratio progressive coding. Extensive experiments show that D2GV-AR renders at over 250 FPS while matching or surpassing recent INR baselines, enabling multiscale continuous rate--quality adaptation.
>
---
#### [replaced 004] LaTo: Landmark-tokenized Diffusion Transformer for Fine-grained Human Face Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25731v2](https://arxiv.org/pdf/2509.25731v2)**

> **作者:** Zhenghao Zhang; Ziying Zhang; Junchao Liao; Xiangyu Meng; Qiang Hu; Siyu Zhu; Xiaoyun Zhang; Long Qin; Weizhi Wang
>
> **摘要:** Recent multimodal models for instruction-based face editing enable semantic manipulation but still struggle with precise attribute control and identity preservation. Structural facial representations such as landmarks are effective for intermediate supervision, yet most existing methods treat them as rigid geometric constraints, which can degrade identity when conditional landmarks deviate significantly from the source (e.g., large expression or pose changes, inaccurate landmark estimates). To address these limitations, we propose LaTo, a landmark-tokenized diffusion transformer for fine-grained, identity-preserving face editing. Our key innovations include: (1) a landmark tokenizer that directly quantizes raw landmark coordinates into discrete facial tokens, obviating the need for dense pixel-wise correspondence; (2) a location-mapped positional encoding and a landmark-aware classifier-free guidance that jointly facilitate flexible yet decoupled interactions among instruction, geometry, and appearance, enabling strong identity preservation; and (3) a landmark predictor that leverages vision-language models to infer target landmarks from instructions and source images, whose structured chain-of-thought improves estimation accuracy and interactive control. To mitigate data scarcity, we curate HFL-150K, to our knowledge the largest benchmark for this task, containing over 150K real face pairs with fine-grained instructions. Extensive experiments show that LaTo outperforms state-of-the-art methods by 7.8% in identity preservation and 4.6% in semantic consistency. Code and dataset will be made publicly available upon acceptance.
>
---
#### [replaced 005] SkyReels-V3 Technique Report
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.17323v2](https://arxiv.org/pdf/2601.17323v2)**

> **作者:** Debang Li; Zhengcong Fei; Tuanhui Li; Yikun Dou; Zheng Chen; Jiangping Yang; Mingyuan Fan; Jingtao Xu; Jiahua Wang; Baoxuan Gu; Mingshan Chang; Wenjing Cai; Yuqiang Xie; Binjie Mao; Youqiang Zhang; Nuo Pang; Hao Zhang; Yuzhe Jin; Zhiheng Xu; Dixuan Lin; Guibin Chen; Yahui Zhou
>
> **摘要:** Video generation serves as a cornerstone for building world models, where multimodal contextual inference stands as the defining test of capability. In this end, we present SkyReels-V3, a conditional video generation model, built upon a unified multimodal in-context learning framework with diffusion Transformers. SkyReels-V3 model supports three core generative paradigms within a single architecture: reference images-to-video synthesis, video-to-video extension and audio-guided video generation. (i) reference images-to-video model is designed to produce high-fidelity videos with strong subject identity preservation, temporal coherence, and narrative consistency. To enhance reference adherence and compositional stability, we design a comprehensive data processing pipeline that leverages cross frame pairing, image editing, and semantic rewriting, effectively mitigating copy paste artifacts. During training, an image video hybrid strategy combined with multi-resolution joint optimization is employed to improve generalization and robustness across diverse scenarios. (ii) video extension model integrates spatio-temporal consistency modeling with large-scale video understanding, enabling both seamless single-shot continuation and intelligent multi-shot switching with professional cinematographic patterns. (iii) Talking avatar model supports minute-level audio-conditioned video generation by training first-and-last frame insertion patterns and reconstructing key-frame inference paradigms. On the basis of ensuring visual quality, synchronization of audio and videos has been optimized. Extensive evaluations demonstrate that SkyReels-V3 achieves state-of-the-art or near state-of-the-art performance on key metrics including visual quality, instruction following, and specific aspect metrics, approaching leading closed-source systems. Github: https://github.com/SkyworkAI/SkyReels-V3.
>
---
#### [replaced 006] Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19616v4](https://arxiv.org/pdf/2505.19616v4)**

> **作者:** Rui Cai; Bangzheng Li; Xiaofei Wen; Muhao Chen; Zhe Zhao
>
> **摘要:** Multimodal Large Language Models demonstrate strong performance on multimodal benchmarks, yet often exhibit poor robustness when exposed to spurious modality interference, such as irrelevant text in vision understanding, or irrelevant visual content in question answering. At its core, modality interference refers to cases where spurious signals from non-essential modalities distort model decisions, which we systematically analyze through causal, perturbation-based diagnostic experiments. To address this problem, we propose a unified finetuning framework that combines heuristic and adversarial perturbation-based data augmentation with output-level consistency regularization between original and perturbed inputs. Extensive experiments across image-heavy, text-heavy, and multimodal benchmarks, spanning multiple MLLM architectures and model scales, demonstrate consistent improvements in unimodal robustness and generalization, while improving standard multimodal performance.
>
---
#### [replaced 007] The Algorithmic Gaze: An Audit and Ethnography of the LAION-Aesthetics Predictor Model
- **分类: cs.HC; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.09896v2](https://arxiv.org/pdf/2601.09896v2)**

> **作者:** Jordan Taylor; William Agnew; Maarten Sap; Sarah E. Fox; Haiyi Zhu
>
> **摘要:** Visual generative AI models are trained using a one-size-fits-all measure of aesthetic appeal. However, what is deemed "aesthetic" is inextricably linked to personal taste and cultural values, raising the question of whose taste is represented in visual generative AI models. In this work, we study an aesthetic evaluation model--LAION Aesthetic Predictor (LAP)--that is widely used to curate datasets to train visual generative image models, like Stable Diffusion, and evaluate the quality of AI-generated images. To understand what LAP measures, we audited the model across three datasets. First, we examined the impact of aesthetic filtering on the LAION-Aesthetics Dataset (approximately 1.2B images), which was curated from LAION-5B using LAP. We find that the LAP disproportionally filters in images with captions mentioning women, while filtering out images with captions mentioning men or LGBTQ+ people. Then, we used LAP to score approximately 330k images across two art datasets, finding the model rates realistic images of landscapes, cityscapes, and portraits from western and Japanese artists most highly. In doing so, the algorithmic gaze of this aesthetic evaluation model reinforces the imperial and male gazes found within western art history. In order to understand where these biases may have originated, we performed a digital ethnography of public materials related to the creation of LAP. We find that the development of LAP reflects the biases we found in our audits, such as the aesthetic scores used to train LAP primarily coming from English-speaking photographers and western AI-enthusiasts. In response, we discuss how aesthetic evaluation can perpetuate representational harms and call on AI developers to shift away from prescriptive measures of "aesthetics" toward more pluralistic evaluation.
>
---
#### [replaced 008] SpatialV2A: Visual-Guided High-fidelity Spatial Audio Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.15017v2](https://arxiv.org/pdf/2601.15017v2)**

> **作者:** Yanan Wang; Linjie Ren; Zihao Li; Junyi Wang; Tian Gan
>
> **摘要:** While video-to-audio generation has achieved remarkable progress in semantic and temporal alignment, most existing studies focus solely on these aspects, paying limited attention to the spatial perception and immersive quality of the synthesized audio. This limitation stems largely from current models' reliance on mono audio datasets, which lack the binaural spatial information needed to learn visual-to-spatial audio mappings. To address this gap, we introduce two key contributions: we construct BinauralVGGSound, the first large-scale video-binaural audio dataset designed to support spatially aware video-to-audio generation; and we propose a end-to-end spatial audio generation framework guided by visual cues, which explicitly models spatial features. Our framework incorporates a visual-guided audio spatialization module that ensures the generated audio exhibits realistic spatial attributes and layered spatial depth while maintaining semantic and temporal alignment. Experiments show that our approach substantially outperforms state-of-the-art models in spatial fidelity and delivers a more immersive auditory experience, without sacrificing temporal or semantic consistency. The demo page can be accessed at https://github.com/renlinjie868-web/SpatialV2A.
>
---
#### [replaced 009] EROAM: Event-based Camera Rotational Odometry and Mapping in Real-time
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EROAM，用于实时事件相机旋转定位与建图，解决高动态场景下的精度与效率问题。通过球面事件表示和优化框架提升性能。**

- **链接: [https://arxiv.org/pdf/2411.11004v2](https://arxiv.org/pdf/2411.11004v2)**

> **作者:** Wanli Xing; Shijie Lin; Linhan Yang; Zeqing Zhang; Yanjun Du; Maolin Lei; Yipeng Pan; Chen Wang; Jia Pan
>
> **备注:** Accepted by IEEE Transactions on Robotics (T-RO), 2026. Project page: https://wlxing1901.github.io/eroam/
>
> **摘要:** This paper presents EROAM, a novel event-based rotational odometry and mapping system that achieves real-time, accurate camera rotation estimation. Unlike existing approaches that rely on event generation models or contrast maximization, EROAM employs a spherical event representation by projecting events onto a unit sphere and introduces Event Spherical Iterative Closest Point (ES-ICP), a novel geometric optimization framework designed specifically for event camera data. The spherical representation simplifies rotational motion formulation while operating in a continuous spherical domain, enabling enhanced spatial resolution. Our system features an efficient map management approach using incremental k-d tree structures and intelligent regional density control, ensuring optimal computational performance during long-term operation. Combined with parallel point-to-line optimization, EROAM achieves efficient computation without compromising accuracy. Extensive experiments on both synthetic and real-world datasets show that EROAM significantly outperforms state-of-the-art methods in terms of accuracy, robustness, and computational efficiency. Our method maintains consistent performance under challenging conditions, including high angular velocities and extended sequences, where other methods often fail or show significant drift. Additionally, EROAM produces high-quality panoramic reconstructions with preserved fine structural details.
>
---
#### [replaced 010] Semantic Router: On the Feasibility of Hijacking MLLMs via a Single Adversarial Perturbation
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [https://arxiv.org/pdf/2511.20002v2](https://arxiv.org/pdf/2511.20002v2)**

> **作者:** Changyue Li; Jiaying Li; Youliang Yuan; Jiaming He; Zhicong Huang; Pinjia He
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly deployed in stateless systems, such as autonomous driving and robotics. This paper investigates a novel threat: Semantic-Aware Hijacking. We explore the feasibility of hijacking multiple stateless decisions simultaneously using a single universal perturbation. We introduce the Semantic-Aware Universal Perturbation (SAUP), which acts as a semantic router, "actively" perceiving input semantics and routing them to distinct, attacker-defined targets. To achieve this, we conduct theoretical and empirical analysis on the geometric properties in the latent space. Guided by these insights, we propose the Semantic-Oriented (SORT) optimization strategy and annotate a new dataset with fine-grained semantics to evaluate performance. Extensive experiments on three representative MLLMs demonstrate the fundamental feasibility of this attack, achieving a 66% attack success rate over five targets using a single frame against Qwen.
>
---
#### [replaced 011] Soft Masked Transformer for Point Cloud Processing with Skip Attention-Based Upsampling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2403.14124v2](https://arxiv.org/pdf/2403.14124v2)**

> **作者:** Yong He; Hongshan Yu; Chaoxu Mu; Mingtao Feng; Tongjia Chen; Zechuan Li; Anwaar Ulhaq; Ajmal Mian
>
> **备注:** Conditionally accepted by IEEE Transactions on Automation Science and Engineering
>
> **摘要:** Point cloud processing methods leverage local and global point features %at the feature level to cater to downstream tasks, yet they often overlook the task-level context inherent in point clouds during the encoding stage. We argue that integrating task-level information into the encoding stage significantly enhances performance. To that end, we propose SMTransformer which incorporates task-level information into a vector-based transformer by utilizing a soft mask generated from task-level queries and keys to learn the attention weights. Additionally, to facilitate effective communication between features from the encoding and decoding layers in high-level tasks such as segmentation, we introduce a skip-attention-based up-sampling block. This block dynamically fuses features from various resolution points across the encoding and decoding layers. To mitigate the increase in network parameters and training time resulting from the complexity of the aforementioned blocks, we propose a novel shared position encoding strategy. This strategy allows various transformer blocks to share the same position information over the same resolution points, thereby reducing network parameters and training time without compromising accuracy.Experimental comparisons with existing methods on multiple datasets demonstrate the efficacy of SMTransformer and skip-attention-based up-sampling for point cloud processing tasks, including semantic segmentation and classification. In particular, we achieve state-of-the-art semantic segmentation results of 73.4% mIoU on S3DIS Area 5 and 62.4% mIoU on SWAN dataset
>
---
#### [replaced 012] Testing of Deep Learning Model in Real World Clinical Setting: A Case Study in Obstetric Ultrasound
- **分类: cs.HC; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2404.00032v2](https://arxiv.org/pdf/2404.00032v2)**

> **作者:** Chun Kit Wong; Mary Ngo; Manxi Lin; Zahra Bashir; Amihai Heen; Morten Bo Søndergaard Svendsen; Martin Grønnebæk Tolsgaard; Anders Nymark Christensen; Aasa Feragen
>
> **备注:** 5 pages; ISBI 2026
>
> **摘要:** Despite the rapid development of AI models in medical image analysis, their validation in real-world clinical settings remains limited. To address this, we introduce a generic framework designed for deploying image-based AI models in such settings. Using this framework, we deployed a trained model for fetal ultrasound standard plane detection, and evaluated it in real-time sessions with both novice and expert users. Feedback from these sessions revealed that while the model offers potential benefits to medical practitioners, the need for navigational guidance was identified as a key area for improvement. These findings underscore the importance of early deployment of AI models in real-world settings, leading to insights that can guide the refinement of the model and system based on actual user feedback.
>
---
#### [replaced 013] MORPH: PDE Foundation Models with Arbitrary Data Modality
- **分类: cs.CV; cs.AI; cs.LG; physics.comp-ph**

- **链接: [https://arxiv.org/pdf/2509.21670v4](https://arxiv.org/pdf/2509.21670v4)**

> **作者:** Mahindra Singh Rautela; Alexander Most; Siddharth Mansingh; Bradley C. Love; Alexander Scheinker; Diane Oyen; Nathan Debardeleben; Earl Lawrence; Ayan Biswas
>
> **摘要:** We introduce MORPH, a modality-agnostic, autoregressive foundation model for partial differential equations (PDEs). MORPH is built on a convolutional vision transformer backbone that seamlessly handles heterogeneous spatiotemporal datasets of varying data modality (1D--3D) at different resolutions, and multiple fields with mixed scalar and vector components. The architecture combines (i) component-wise convolution, which jointly processes scalar and vector channels to capture local interactions, (ii) inter-field cross-attention, which models and selectively propagates information between different physical fields, (iii) axial attentions, which factorize full spatiotemporal self-attention along individual spatial and temporal axes to reduce computational burden while retaining expressivity. We pretrain multiple model variants on a diverse collection of heterogeneous PDE datasets and evaluate transfer to a range of downstream prediction tasks. Using both full-model fine-tuning and parameter-efficient low-rank adapters, MORPH outperforms models trained from scratch. Across extensive evaluations, MORPH matches or surpasses strong baselines and recent state-of-the-art models. Collectively, these capabilities present a flexible and powerful backbone for learning from the heterogeneous and multimodal nature of scientific observations, charting a path toward scalable and data-efficient scientific machine learning. The source code, datasets, and models are publicly available at https://github.com/lanl/MORPH.
>
---
#### [replaced 014] Enhancing Visual Prompting through Expanded Transformation Space and Overfitting Mitigation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.07823v2](https://arxiv.org/pdf/2510.07823v2)**

> **作者:** Shohei Enomoto
>
> **备注:** Accepted to NeurIPS2025
>
> **摘要:** Visual prompting (VP) has emerged as a promising parameter-efficient fine-tuning approach for adapting pre-trained vision models to downstream tasks without modifying model parameters. Despite offering advantages like negligible computational overhead and compatibility with black-box models, conventional VP methods typically achieve lower accuracy than other adaptation approaches. Our analysis reveals two critical limitations: the restricted expressivity of simple additive transformation and a tendency toward overfitting when the parameter count increases. To address these challenges, we propose ACAVP (Affine, Color, and Additive Visual Prompting), which enhances VP's expressive power by introducing complementary transformation operations: affine transformation for creating task-specific prompt regions while preserving original image information, and color transformation for emphasizing task-relevant visual features. Additionally, we identify that overfitting is a critical issue in VP training and introduce TrivialAugment as an effective data augmentation, which not only benefits our approach but also significantly improves existing VP methods, with performance gains of up to 12 percentage points on certain datasets. This demonstrates that appropriate data augmentation is universally beneficial for VP training. Extensive experiments across twelve diverse image classification datasets with two different model architectures demonstrate that ACAVP achieves state-of-the-art accuracy among VP methods, surpasses linear probing in average accuracy, and exhibits superior robustness to distribution shifts, all while maintaining minimal computational overhead during inference. Our code is available at https://github.com/s-enmt/ACAVP.
>
---
#### [replaced 015] RAVE: Rate-Adaptive Visual Encoding for 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.07052v2](https://arxiv.org/pdf/2512.07052v2)**

> **作者:** Hoang-Nhat Tran; Francesco Di Sario; Gabriele Spadaro; Giuseppe Valenzise; Enzo Tartaglione
>
> **摘要:** Recent advances in neural scene representations have transformed immersive multimedia, with 3D Gaussian Splatting (3DGS) enabling real-time photorealistic rendering. Despite its efficiency, 3DGS suffers from large memory requirements and costly training procedures, motivating efforts toward compression. Existing approaches, however, operate at fixed rates, limiting adaptability to varying bandwidth and device constraints. In this work, we propose a flexible compression scheme for 3DGS that supports interpolation at any rate between predefined bounds. Our method is computationally lightweight, requires no retraining for any rate, and preserves rendering quality across a broad range of operating points. Experiments demonstrate that the approach achieves efficient, high-quality compression while offering dynamic rate control, making it suitable for practical deployment in immersive applications. The code is available at https://github.com/inspiros/RAVE.
>
---
#### [replaced 016] OREHAS: A fully automated deep-learning pipeline for volumetric endolymphatic hydrops quantification in MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.18368v2](https://arxiv.org/pdf/2601.18368v2)**

> **作者:** Caterina Fuster-Barceló; Claudia Castrillón; Laura Rodrigo-Muñoz; Victor Manuel Suárez-Vega; Nicolás Pérez-Fernández; Gorka Bastarrika; Arrate Muñoz-Barrutia
>
> **摘要:** We present OREHAS (Optimized Recognition & Evaluation of volumetric Hydrops in the Auditory System), the first fully automatic pipeline for volumetric quantification of endolymphatic hydrops (EH) from routine 3D-SPACE-MRC and 3D-REAL-IR MRI. The system integrates three components -- slice classification, inner ear localization, and sequence-specific segmentation -- into a single workflow that computes per-ear endolymphatic-to-vestibular volume ratios (ELR) directly from whole MRI volumes, eliminating the need for manual intervention. Trained with only 3 to 6 annotated slices per patient, OREHAS generalized effectively to full 3D volumes, achieving Dice scores of 0.90 for SPACE-MRC and 0.75 for REAL-IR. In an external validation cohort with complete manual annotations, OREHAS closely matched expert ground truth (VSI = 74.3%) and substantially outperformed the clinical syngo.via software (VSI = 42.5%), which tended to overestimate endolymphatic volumes. Across 19 test patients, vestibular measurements from OREHAS were consistent with syngo.via, while endolymphatic volumes were systematically smaller and more physiologically realistic. These results show that reliable and reproducible EH quantification can be achieved from standard MRI using limited supervision. By combining efficient deep-learning-based segmentation with a clinically aligned volumetric workflow, OREHAS reduces operator dependence, ensures methodological consistency. Besides, the results are compatible with established imaging protocols. The approach provides a robust foundation for large-scale studies and for recalibrating clinical diagnostic thresholds based on accurate volumetric measurements of the inner ear.
>
---
#### [replaced 017] REST: Diffusion-based Real-time End-to-end Streaming Talking Head Generation via ID-Context Caching and Asynchronous Streaming Distillation
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于 Talking Head Generation 任务，旨在解决扩散模型在实时生成中的速度慢和非自回归问题。通过引入ID-Context Cache和ASD策略，提升生成效率与一致性。**

- **链接: [https://arxiv.org/pdf/2512.11229v3](https://arxiv.org/pdf/2512.11229v3)**

> **作者:** Haotian Wang; Yuzhe Weng; Jun Du; Haoran Xu; Xiaoyan Wu; Shan He; Bing Yin; Cong Liu; Qingfeng Liu
>
> **备注:** 27 pages, 10 figures
>
> **摘要:** Diffusion models have significantly advanced the field of talking head generation (THG). However, slow inference speeds and prevalent non-autoregressive paradigms severely constrain the application of diffusion-based THG models. In this study, we propose REST, a pioneering diffusion-based, real-time, end-to-end streaming audio-driven talking head generation framework. To support real-time end-to-end generation, a compact video latent space is first learned through a spatiotemporal variational autoencoder with a high compression ratio. Additionally, to enable semi-autoregressive streaming within the compact video latent space, we introduce an ID-Context Cache mechanism, which integrates ID-Sink and Context-Cache principles into key-value caching for maintaining identity consistency and temporal coherence during long-term streaming generation. Furthermore, an Asynchronous Streaming Distillation (ASD) strategy is proposed to mitigate error accumulation and enhance temporal consistency in streaming generation, leveraging a non-streaming teacher with an asynchronous noise schedule to supervise the streaming student. REST bridges the gap between autoregressive and diffusion-based approaches, achieving a breakthrough in efficiency for applications requiring real-time THG. Experimental results demonstrate that REST outperforms state-of-the-art methods in both generation speed and overall performance.
>
---
#### [replaced 018] GUIGuard: Toward a General Framework for Privacy-Preserving GUI Agents
- **分类: cs.CR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.18842v2](https://arxiv.org/pdf/2601.18842v2)**

> **作者:** Yanxi Wang; Zhiling Zhang; Wenbo Zhou; Weiming Zhang; Jie Zhang; Qiannan Zhu; Yu Shi; Shuxin Zheng; Jiyan He
>
> **摘要:** GUI agents enable end-to-end automation through direct perception of and interaction with on-screen interfaces. However, these agents frequently access interfaces containing sensitive personal information, and screenshots are often transmitted to remote models, creating substantial privacy risks. These risks are particularly severe in GUI workflows: GUIs expose richer, more accessible private information, and privacy risks depend on interaction trajectories across sequential scenes. We propose GUIGuard, a three-stage framework for privacy-preserving GUI agents: (1) privacy recognition, (2) privacy protection, and (3) task execution under protection. We further construct GUIGuard-Bench, a cross-platform benchmark with 630 trajectories and 13,830 screenshots, annotated with region-level privacy grounding and fine-grained labels of risk level, privacy category, and task necessity. Evaluations reveal that existing agents exhibit limited privacy recognition, with state-of-the-art models achieving only 13.3% accuracy on Android and 1.4% on PC. Under privacy protection, task-planning semantics can still be maintained, with closed-source models showing stronger semantic consistency than open-source ones. Case studies on MobileWorld show that carefully designed protection strategies achieve higher task accuracy while preserving privacy. Our results highlight privacy recognition as a critical bottleneck for practical GUI agents. Project: https://futuresis.github.io/GUIGuard-page/
>
---
#### [replaced 019] CacheFlow: Fast Human Motion Prediction by Cached Normalizing Flow
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.13140v2](https://arxiv.org/pdf/2505.13140v2)**

> **作者:** Takahiro Maeda; Jinkun Cao; Norimichi Ukita; Kris Kitani
>
> **备注:** Accepted at Transactions on Machine Learning Research (TMLR). See https://openreview.net/forum?id=y8dGdBJkWa
>
> **摘要:** Many density estimation techniques for 3D human motion prediction require a significant amount of inference time, often exceeding the duration of the predicted time horizon. To address the need for faster density estimation for 3D human motion prediction, we introduce a novel flow-based method for human motion prediction called CacheFlow. Unlike previous conditional generative models that suffer from poor time efficiency, CacheFlow takes advantage of an unconditional flow-based generative model that transforms a Gaussian mixture into the density of future motions. The results of the computation of the flow-based generative model can be precomputed and cached. Then, for conditional prediction, we seek a mapping from historical trajectories to samples in the Gaussian mixture. This mapping can be done by a much more lightweight model, thus saving significant computation overhead compared to a typical conditional flow model. In such a two-stage fashion and by caching results from the slow flow model computation, we build our CacheFlow without loss of prediction accuracy and model expressiveness. This inference process is completed in approximately one millisecond, making it 4 times faster than previous VAE methods and 30 times faster than previous diffusion-based methods on standard benchmarks such as Human3.6M and AMASS datasets. Furthermore, our method demonstrates improved density estimation accuracy and comparable prediction accuracy to a SOTA method on Human3.6M. Our code and models are available at https://github.com/meaten/CacheFlow.
>
---
#### [replaced 020] ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.07720v3](https://arxiv.org/pdf/2412.07720v3)**

> **作者:** Jinyi Hu; Shengding Hu; Yuxuan Song; Yufei Huang; Mingxuan Wang; Hao Zhou; Zhiyuan Liu; Wei-Ying Ma; Maosong Sun
>
> **备注:** TMLR camera-ready version
>
> **摘要:** Autoregressive and diffusion models have achieved remarkable progress in language models and visual generation, respectively. We present ACDiT, a novel Autoregressive blockwise Conditional Diffusion Transformer, that innovatively combines autoregressive and diffusion paradigms for continuous visual information. By introducing a block-wise autoregressive unit, ACDiT offers a flexible interpolation between token-wise autoregression and full-sequence diffusion, bypassing the limitations of discrete tokenization. The generation of each block is formulated as a conditional diffusion process, conditioned on prior blocks. ACDiT is easy to implement, as simple as applying a specially designed Skip-Causal Attention Mask on the standard diffusion transformer during training. During inference, the process iterates between diffusion denoising and autoregressive decoding that can make full use of KV-Cache. We validate the effectiveness of ACDiT on image, video, and text generation and show that ACDiT performs best among all autoregressive baselines under similar model scales on visual generation tasks. We also demonstrate that, benefiting from autoregressive modeling, pretrained ACDiT can be transferred in visual understanding tasks despite being trained with the generative objective. The analysis of the trade-off between autoregressive and diffusion demonstrates the potential of ACDiT to be used in long-horizon visual generation tasks. We hope that ACDiT offers a novel perspective on visual autoregressive generation and sheds light on new avenues for unified models.
>
---
#### [replaced 021] City Navigation in the Wild: Exploring Emergent Navigation from Web-Scale Knowledge in MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15933v2](https://arxiv.org/pdf/2512.15933v2)**

> **作者:** Dwip Dalal; Utkarsh Mishra; Narendra Ahuja; Nebojsa Jojic
>
> **备注:** Accepted at EACL 2026
>
> **摘要:** Leveraging multimodal large language models (MLLMs) to develop embodied agents offers significant promise for addressing complex real-world tasks. However, current evaluation benchmarks remain predominantly language-centric or heavily reliant on simulated environments, rarely probing the nuanced, knowledge-intensive reasoning essential for practical, real-world scenarios. To bridge this critical gap, we introduce the task of Sparsely Grounded Visual Navigation, explicitly designed to evaluate the sequential decision-making abilities of MLLMs in challenging, knowledge-intensive real-world environment. We operationalize this task with CityNav, a comprehensive benchmark encompassing four diverse global cities, specifically constructed to assess raw MLLM-driven agents in city navigation. Agents are required to rely solely on visual inputs and internal multimodal reasoning to sequentially navigate 50+ decision points without additional environmental annotations or specialized architectural modifications. Crucially, agents must autonomously achieve localization through interpreting city-specific cues and recognizing landmarks, perform spatial reasoning, and strategically plan and execute routes to their destinations. Through extensive evaluations, we demonstrate that current state-of-the-art MLLMs, reasoning techniques (e.g., GEPA, chain-of-thought, reflection) and competitive baseline PReP significantly underperform in this challenging setting. To address this, we propose Verbalization of Path(VoP), which explicitly grounds the agent's internal reasoning by probing city-scale cognitive maps (key landmarks and directions toward the destination) from the MLLM, substantially enhancing navigation success. Project Webpage: https://dwipddalal.github.io/AgentNav/
>
---
#### [replaced 022] Are We Truly Forgetting? A Critical Re-examination of Machine Unlearning Evaluation Protocols
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.06991v3](https://arxiv.org/pdf/2503.06991v3)**

> **作者:** Yongwoo Kim; Sungmin Cha; Donghyun Kim
>
> **备注:** Accepted to Engineering Applications of Artificial Intelligence
>
> **摘要:** Machine unlearning is a process to remove specific data points from a trained model while maintaining the performance on the retain data, addressing privacy or legal requirements. Despite its importance, existing unlearning evaluations tend to focus on logit-based metrics under small-scale scenarios. We observe that this could lead to a false sense of security in unlearning approaches under real-world scenarios. In this paper, we conduct a comprehensive evaluation that employs representation-based evaluations of the unlearned model under large-scale scenarios to verify whether the unlearning approaches truly eliminate the targeted data from the model's representation perspective. Our analysis reveals that current state-of-the-art unlearning approaches either completely degrade the representational quality of the unlearned model or merely modify the classifier, thereby achieving superior logit-based performance while maintaining representational similarity to the original model. Furthermore, we introduce a novel unlearning evaluation scenario in which the forgetting classes exhibit semantic similarity to downstream task classes, necessitating that feature representations diverge significantly from those of the original model, thus enabling a more thorough evaluation from a representation perspective. We hope our benchmark will serve as a standardized protocol for evaluating unlearning algorithms under realistic conditions.
>
---
#### [replaced 023] Edge Collaborative Gaussian Splatting with Integrated Rendering and Communication
- **分类: cs.IT; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.22718v2](https://arxiv.org/pdf/2510.22718v2)**

> **作者:** Yujie Wan; Chenxuan Liu; Shuai Wang; Tong Zhang; James Jianqiao Yu; Kejiang Ye; Dusit Niyato; Chengzhong Xu
>
> **备注:** IEEE ICASSP, Barcelona, Spain, 2026
>
> **摘要:** Gaussian splatting (GS) struggles with degraded rendering quality on low-cost devices. To address this issue, we present edge collaborative GS (ECO-GS), where each user can switch between a local small GS model to guarantee timeliness and a remote large GS model to guarantee fidelity. However, deciding how to engage the large GS model is nontrivial, due to the interdependency between rendering requirements and resource conditions. To this end, we propose integrated rendering and communication (IRAC), which jointly optimizes collaboration status (i.e., deciding whether to engage large GS) and edge power allocation (i.e., enabling remote rendering) under communication constraints across different users by minimizing a newly-derived GS switching function. Despite the nonconvexity of the problem, we propose an efficient penalty majorization minimization (PMM) algorithm to obtain the critical point solution. Furthermore, we develop an imitation learning optimization (ILO) algorithm, which reduces the computational time by over 100x compared to PMM. Experiments demonstrate the superiority of PMM and the real-time execution capability of ILO.
>
---
#### [replaced 024] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2502.20120v4](https://arxiv.org/pdf/2502.20120v4)**

> **作者:** QingYuan Jiang; Longfei Huang; Yang Yang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Multimodal learning (MML) is significantly constrained by modality imbalance, leading to suboptimal performance in practice. While existing approaches primarily focus on balancing the learning of different modalities to address this issue, they fundamentally overlook the inherent disproportion in model classification ability, which serves as the primary cause of this phenomenon. In this paper, we propose a novel multimodal learning approach to dynamically balance the classification ability of weak and strong modalities by incorporating the principle of boosting. Concretely, we first propose a sustained boosting algorithm in multimodal learning by simultaneously optimizing the classification and residual errors. Subsequently, we introduce an adaptive classifier assignment strategy to dynamically facilitate the classification performance of the weak modality. Furthermore, we theoretically analyze the convergence property of the cross-modal gap function, ensuring the effectiveness of the proposed boosting scheme. To this end, the classification ability of strong and weak modalities is expected to be balanced, thereby mitigating the imbalance issue. Empirical experiments on widely used datasets reveal the superiority of our method through comparison with various state-of-the-art (SOTA) multimodal learning baselines. The source code is available at https://github.com/njustkmg/NeurIPS25-AUG.
>
---
#### [replaced 025] Uni-Parser Technical Report
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.15098v2](https://arxiv.org/pdf/2512.15098v2)**

> **作者:** Xi Fang; Haoyi Tao; Shuwen Yang; Chaozheng Huang; Suyang Zhong; Haocheng Lu; Han Lyu; Xinyu Li; Linfeng Zhang; Guolin Ke
>
> **摘要:** This technical report introduces Uni-Parser, an industrial-grade document parsing engine tailored for scientific literature and patents, delivering high throughput, robust accuracy, and cost efficiency. Unlike pipeline-based document parsing methods, Uni-Parser employs a modular, loosely coupled multi-expert architecture that preserves fine-grained cross-modal alignments across text, equations, tables, figures, and chemical structures, while remaining easily extensible to emerging modalities. The system incorporates adaptive GPU load balancing, distributed inference, dynamic module orchestration, and configurable modes that support either holistic or modality-specific parsing. Optimized for large-scale cloud deployment, Uni-Parser achieves a processing rate of up to 20 PDF pages per second on 8 x NVIDIA RTX 4090D GPUs, enabling cost-efficient inference across billions of pages. This level of scalability facilitates a broad spectrum of downstream applications, ranging from literature retrieval and summarization to the extraction of chemical structures, reaction schemes, and bioactivity data, as well as the curation of large-scale corpora for training next-generation large language models and AI4Science models.
>
---
#### [replaced 026] A New Dataset and Framework for Robust Road Surface Classification via Camera-IMU Fusion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.20847v2](https://arxiv.org/pdf/2601.20847v2)**

> **作者:** Willams de Lima Costa; Thifany Ketuli Silva de Souza; Jonas Ferreira Silva; Carlos Gabriel Bezerra Pereira; Bruno Reis Vila Nova; Leonardo Silvino Brito; Rafael Raider Leoni; Juliano Silva Filho; Valter Ferreira; Sibele Miguel Soares Neto; Samantha Uehara; Daniel Giacometti Amaral; João Marcelo Teixeira; Veronica Teichrieb; Cristiano Coelho de Araújo
>
> **摘要:** Road surface classification (RSC) is a key enabler for environment-aware predictive maintenance systems. However, existing RSC techniques often fail to generalize beyond narrow operational conditions due to limited sensing modalities and datasets that lack environmental diversity. This work addresses these limitations by introducing a multimodal framework that fuses images and inertial measurements using a lightweight bidirectional cross-attention module followed by an adaptive gating layer that adjusts modality contributions under domain shifts. Given the limitations of current benchmarks, especially regarding lack of variability, we introduce ROAD, a new dataset composed of three complementary subsets: (i) real-world multimodal recordings with RGB-IMU streams synchronized using a gold-standard industry datalogger, captured across diverse lighting, weather, and surface conditions; (ii) a large vision-only subset designed to assess robustness under adverse illumination and heterogeneous capture setups; and (iii) a synthetic subset generated to study out-of-distribution generalization in scenarios difficult to obtain in practice. Experiments show that our method achieves a +1.4 pp improvement over the previous state-of-the-art on the PVS benchmark and an +11.6 pp improvement on our multimodal ROAD subset, with consistently higher F1-scores on minority classes. The framework also demonstrates stable performance across challenging visual conditions, including nighttime, heavy rain, and mixed-surface transitions. These findings indicate that combining affordable camera and IMU sensors with multimodal attention mechanisms provides a scalable, robust foundation for road surface understanding, particularly relevant for regions where environmental variability and cost constraints limit the adoption of high-end sensing suites.
>
---
#### [replaced 027] BiPO: Bidirectional Partial Occlusion Network for Text-to-Motion Synthesis
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2412.00112v4](https://arxiv.org/pdf/2412.00112v4)**

> **作者:** Seong-Eun Hong; Soobin Lim; Juyeong Hwang; Minwook Chang; Hyeongyeop Kang
>
> **备注:** 18 pages, 11 figures. Accepted to WACV 2026 (Oral)
>
> **摘要:** Generating natural and expressive human motions from textual descriptions is challenging due to the complexity of coordinating full-body dynamics and capturing nuanced motion patterns over extended sequences that accurately reflect the given text. To address this, we introduce BiPO, Bidirectional Partial Occlusion Network for Text-to-Motion Synthesis, a novel model that enhances text-to-motion synthesis by integrating part-based generation with a bidirectional autoregressive architecture. This integration allows BiPO to consider both past and future contexts during generation while enhancing detailed control over individual body parts without requiring ground-truth motion length. To relax the interdependency among body parts caused by the integration, we devise the Partial Occlusion technique, which probabilistically occludes the certain motion part information during training. In our comprehensive experiments, BiPO achieves state-of-the-art performance on the HumanML3D dataset, outperforming recent methods such as ParCo, MoMask, and BAMM in terms of FID scores and overall motion quality. Notably, BiPO excels not only in the text-to-motion generation task but also in motion editing tasks that synthesize motion based on partially generated motion sequences and textual descriptions. These results reveal the BiPO's effectiveness in advancing text-to-motion synthesis and its potential for practical applications.
>
---
#### [replaced 028] MultiHateLoc: Towards Temporal Localisation of Multimodal Hate Content in Online Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10408v3](https://arxiv.org/pdf/2512.10408v3)**

> **作者:** Qiyue Sun; Tailin Chen; Yinghui Zhang; Yuchen Zhang; Jiangbei Yue; Jianbo Jiao; Zeyu Fu
>
> **备注:** In Proceedings of the ACM Web Conference 2026 (WWW 2026)
>
> **摘要:** The rapid growth of video content on platforms such as TikTok and YouTube has intensified the spread of multimodal hate speech, where harmful cues emerge subtly and asynchronously across visual, acoustic, and textual streams. Existing research primarily focuses on video-level classification, leaving the practically crucial task of temporal localisation, identifying when hateful segments occur, largely unaddressed. This challenge is even more noticeable under weak supervision, where only video-level labels are available, and static fusion or classification-based architectures struggle to capture cross-modal and temporal dynamics. To address these challenges, we propose MultiHateLoc, the first framework designed for weakly-supervised multimodal hate localisation. MultiHateLoc incorporates (1) modality-aware temporal encoders to model heterogeneous sequential patterns, including a tailored text-based preprocessing module for feature enhancement; (2) dynamic cross-modal fusion to adaptively emphasise the most informative modality at each moment and a cross-modal contrastive alignment strategy to enhance multimodal feature consistency; (3) a modality-aware MIL objective to identify discriminative segments under video-level supervision. Despite relying solely on coarse labels, MultiHateLoc produces fine-grained, interpretable frame-level predictions. Experiments on HateMM and MultiHateClip show that our method achieves state-of-the-art performance in the localisation task.
>
---
#### [replaced 029] OrthoInsight: Rib Fracture Diagnosis and Report Generation Based on Multi-Modal Large Models
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.13993v4](https://arxiv.org/pdf/2507.13993v4)**

> **作者:** Ningyong Wu; Jiangbo Zhang; Wenhong Zhao; Jinzhi Wang; Chenzhan Yu; Zhigang Xiu; Duwei Dai; Ziyu Xu; Yongli Yang
>
> **摘要:** The growing volume of medical imaging data has increased the need for automated diagnostic tools, especially for musculoskeletal injuries like rib fractures, commonly detected via CT scans. Manual interpretation is time-consuming and error-prone. We propose OrthoInsight, a multi-modal deep learning framework for rib fracture diagnosis and report generation. It integrates a YOLOv9 model for fracture detection, a medical knowledge graph for retrieving clinical context, and a fine-tuned LLaVA language model for generating diagnostic reports. OrthoInsight combines visual features from CT images with expert textual data to deliver clinically useful outputs. Evaluated on 28,675 annotated CT images and expert reports, it achieves high performance across Diagnostic Accuracy, Content Completeness, Logical Coherence, and Clinical Guidance Value, with an average score of 4.28, outperforming models like GPT-4 and Claude-3. This study demonstrates the potential of multi-modal learning in transforming medical image analysis and providing effective support for radiologists.
>
---
#### [replaced 030] Learning Stochastic Bridges for Video Object Removal via Video-to-Video Translation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.12066v3](https://arxiv.org/pdf/2601.12066v3)**

> **作者:** Zijie Lou; Xiangwei Feng; Jiaxin Wang; Jiangtao Yao; Fei Che; Tianbao Liu; Chengjing Wu; Xiaochao Qu; Luoqi Liu; Ting Liu
>
> **摘要:** Existing video object removal methods predominantly rely on diffusion models following a noise-to-data paradigm, where generation starts from uninformative Gaussian noise. This approach discards the rich structural and contextual priors present in the original input video. Consequently, such methods often lack sufficient guidance, leading to incomplete object erasure or the synthesis of implausible content that conflicts with the scene's physical logic. In this paper, we reformulate video object removal as a video-to-video translation task via a stochastic bridge model. Unlike noise-initialized methods, our framework establishes a direct stochastic path from the source video (with objects) to the target video (objects removed). This bridge formulation effectively leverages the input video as a strong structural prior, guiding the model to perform precise removal while ensuring that the filled regions are logically consistent with the surrounding environment. To address the trade-off where strong bridge priors hinder the removal of large objects, we propose a novel adaptive mask modulation strategy. This mechanism dynamically modulates input embeddings based on mask characteristics, balancing background fidelity with generative flexibility. Extensive experiments demonstrate that our approach significantly outperforms existing methods in both visual quality and temporal consistency. The project page is https://bridgeremoval.github.io/.
>
---
#### [replaced 031] EndoAgent: A Memory-Guided Reflective Agent for Intelligent Endoscopic Vision-to-Decision Reasoning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出EndoAgent，用于内镜视觉到决策的智能分析，解决多步骤临床任务中的协调与推理问题。通过双记忆设计和工具集成，提升诊断准确性与灵活性。**

- **链接: [https://arxiv.org/pdf/2508.07292v2](https://arxiv.org/pdf/2508.07292v2)**

> **作者:** Yi Tang; Kaini Wang; Yang Chen; Guangquan Zhou
>
> **备注:** This paper is withdrawn due to the identification of a methodological flaw in the experimental evaluation protocol (Section 5), which may lead to unreliable performance comparisons. The authors are re-examining the evaluation design and will release a corrected version in the future
>
> **摘要:** Developing general artificial intelligence (AI) systems to support endoscopic image diagnosis is an emerging research priority. Existing methods based on large-scale pretraining often lack unified coordination across tasks and struggle to handle the multi-step processes required in complex clinical workflows. While AI agents have shown promise in flexible instruction parsing and tool integration across domains, their potential in endoscopy remains underexplored. To address this gap, we propose EndoAgent, the first memory-guided agent for vision-to-decision endoscopic analysis that integrates iterative reasoning with adaptive tool selection and collaboration. Built on a dual-memory design, it enables sophisticated decision-making by ensuring logical coherence through short-term action tracking and progressively enhancing reasoning acuity through long-term experiential learning. To support diverse clinical tasks, EndoAgent integrates a suite of expert-designed tools within a unified reasoning loop. We further introduce EndoAgentBench, a benchmark of 5,709 visual question-answer pairs that assess visual understanding and language generation capabilities in realistic scenarios. Extensive experiments show that EndoAgent consistently outperforms both general and medical multimodal models, exhibiting its strong flexibility and reasoning capabilities.
>
---
#### [replaced 032] Revisiting Reweighted Risk for Calibration: AURC, Focal, and Inverse Focal Loss
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23463v5](https://arxiv.org/pdf/2505.23463v5)**

> **作者:** Han Zhou; Sebastian G. Gruber; Teodora Popordanoska; Matthew B. Blaschko
>
> **摘要:** Several variants of reweighted risk functionals, such as focal loss, inverse focal loss, and the Area Under the Risk Coverage Curve (AURC), have been proposed for improving model calibration; yet their theoretical connections to calibration errors remain under-explored. In this paper, we revisit a broad class of weighted risk functions and find a principled connection between calibration error and selective classification. We show that minimizing calibration error is closely linked to the selective classification paradigm and demonstrate that optimizing selective risk in low confidence regions naturally improves calibration. Our proposed loss shares a similar reweighting strategy with dual focal loss but offers greater flexibility through the choice of confidence score functions (CSFs). Furthermore, our approach utilizes a bin-based cumulative distribution function (CDF) approximation, enabling efficient gradient-based optimization with O(nM) complexity for n samples and M bins. Empirical evaluations demonstrate that our method achieves competitive calibration performance across a range of datasets and model architectures.
>
---
#### [replaced 033] Memento 2: Learning by Stateful Reflective Memory
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.22716v3](https://arxiv.org/pdf/2512.22716v3)**

> **作者:** Jun Wang
>
> **备注:** 35 pages, four figures
>
> **摘要:** We present a theoretical study of continual and experiential learning in large language model agents that combine episodic memory with reinforcement learning. We argue that the key mechanism for continual adaptation, without updating model parameters, is reflection: the agent's ability to use past experience to guide future actions. Empirical findings suggest that episodic, experience-driven reflection enables generalised adaptation across a wide range of open-ended, long-horizon tasks. This indicates that efficient learning can occur during deployment and weakens the traditional separation between training and testing. Motivated by this, we introduce the Stateful Reflective Decision Process, a formal model of reflective memory dynamics. In this abstraction, an agent maintains an episodic memory and performs two core operations. Writing stores interaction outcomes and plays the role of policy evaluation. Reading retrieves relevant past cases to inform decisions and plays the role of policy improvement. This perspective treats reflective memory as a control object that can be analysed using classical reinforcement learning tools. We then develop a read-write reflective learning framework by integrating retrieval into soft policy iteration and establish convergence guarantees. We show that as memory grows and provides denser coverage of the state space, the resulting composite policy converges to the optimal solution. Overall, this framework connects practical memory-based methods with principled reinforcement learning, providing a rigorous mathematical basis for building reflective, memory-embedded agents capable of continual general-purpose learning.
>
---
#### [replaced 034] Hierarchical Transformers for Unsupervised 3D Shape Abstraction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.27088v2](https://arxiv.org/pdf/2510.27088v2)**

> **作者:** Aditya Vora; Lily Goli; Andrea Tagliasacchi; Hao Zhang
>
> **备注:** Accepted to 3DV'26, 16 pages, 13 figures
>
> **摘要:** We introduce HiT, a novel hierarchical neural field representation for 3D shapes that learns general hierarchies in a coarse-to-fine manner across different shape categories in an unsupervised setting. Our key contribution is a hierarchical transformer (HiT), where each level learns parent-child relationships of the tree hierarchy using a compressed codebook. This codebook enables the network to automatically identify common substructures across potentially diverse shape categories. Unlike previous works that constrain the task to a fixed hierarchical structure (e.g., binary), we impose no such restriction, except for limiting the total number of nodes at each tree level. This flexibility allows our method to infer the hierarchical structure directly from data, over multiple shape categories, and representing more general and complex hierarchies than prior approaches. When trained at scale with a reconstruction loss, our model captures meaningful containment relationships between parent and child nodes. We demonstrate its effectiveness through an unsupervised shape segmentation task over all 55 ShapeNet categories, where our method successfully segments shapes into multiple levels of granularity.
>
---
#### [replaced 035] Practical Insights into Semi-Supervised Object Detection Approaches
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.13380v2](https://arxiv.org/pdf/2601.13380v2)**

> **作者:** Chaoxin Wang; Bharaneeshwar Balasubramaniyam; Anurag Sangem; Nicolais Guevara; Doina Caragea
>
> **摘要:** Learning in data-scarce settings has recently gained significant attention in the research community. Semi-supervised object detection(SSOD) aims to improve detection performance by leveraging a large number of unlabeled images alongside a limited number of labeled images(a.k.a.,few-shot learning). In this paper, we present a comprehensive comparison of three state-of-the-art SSOD approaches, including MixPL, Semi-DETR and Consistent-Teacher, with the goal of understanding how performance varies with the number of labeled images. We conduct experiments using the MS-COCO and Pascal VOC datasets, two popular object detection benchmarks which allow for standardized evaluation. In addition, we evaluate the SSOD approaches on a custom Beetle dataset which enables us to gain insights into their performance on specialized datasets with a smaller number of object categories. Our findings highlight the trade-offs between accuracy, model size, and latency, providing insights into which methods are best suited for low-data regimes.
>
---
#### [replaced 036] MindGrab for BrainChop: Fast and Accurate Skull Stripping for Command Line and Browser
- **分类: eess.IV; cs.AI; cs.CV; cs.NE**

- **链接: [https://arxiv.org/pdf/2506.11860v2](https://arxiv.org/pdf/2506.11860v2)**

> **作者:** Armina Fani; Mike Doan; Isabelle Le; Alex Fedorov; Malte Hoffmann; Chris Rorden; Sergey Plis
>
> **备注:** 17 pages, 1 table, 5 figures. 2 supplementary tables. Brainchop-cli: https://pypi.org/project/brainchop/ . Brainchop web: https://brainchop.org/
>
> **摘要:** Deployment complexity and specialized hardware requirements hinder the adoption of deep learning models in neuroimaging. We present MindGrab, a lightweight, fully convolutional model for volumetric skull stripping across all imaging modalities. MindGrab's architecture is designed from first principles using a spectral interpretation of dilated convolutions, and demonstrates state-of-the-art performance (mean Dice score across datasets and modalities: 95.9 with SD 1.6), with up to 40-fold speedups and substantially lower memory demands compared to established methods. Its minimal footprint allows for fast, full-volume processing in resource-constrained environments, including direct in-browser execution. MindGrab is delivered via the BrainChop platform as both a simple command-line tool (pip install brainchop) and a zero-installation web application (brainchop.org). By removing traditional deployment barriers without sacrificing accuracy, MindGrab makes state-of-the-art neuroimaging analysis broadly accessible.
>
---
#### [replaced 037] Align & Invert: Solving Inverse Problems with Diffusion and Flow-based Models via Representation Alignment
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.16870v2](https://arxiv.org/pdf/2511.16870v2)**

> **作者:** Loukas Sfountouris; Giannis Daras; Paris Giampouras
>
> **摘要:** Enforcing alignment between the internal representations of diffusion or flow-based generative models and those of pretrained self-supervised encoders has recently been shown to provide a powerful inductive bias, improving both convergence and sample quality. In this work, we extend this idea to inverse problems, where pretrained generative models are employed as priors. We propose applying representation alignment (REPA) between diffusion or flow-based models and a DINOv2 visual encoder, to guide the reconstruction process at inference time. Although ground-truth signals are unavailable in inverse problems, we empirically show that aligning model representations of approximate target features can substantially enhance reconstruction quality and perceptual realism. We provide theoretical results showing (a) that REPA regularization can be viewed as a variational approach for minimizing a divergence measure in the DINOv2 embedding space, and (b) how under certain regularity assumptions REPA updates steer the latent diffusion states toward those of the clean image. These results offer insights into the role of REPA in improving perceptual fidelity. Finally, we demonstrate the generality of our approach by We integrate REPA into multiple state-of-the-art inverse problem solvers, and provide extensive experiments on super-resolution, box inpainting, Gaussian deblurring, and motion deblurring confirming that our method consistently improves reconstruction quality, while also providing efficiency gains reducing the number of required discretization steps.
>
---
#### [replaced 038] FreeFuse: Multi-Subject LoRA Fusion via Adaptive Token-Level Routing at Test Time
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.23515v2](https://arxiv.org/pdf/2510.23515v2)**

> **作者:** Yaoli Liu; Yao-Xiang Ding; Kun Zhou
>
> **摘要:** This paper proposes FreeFuse, a training-free framework for multi-subject text-to-image generation through automatic fusion of multiple subject LoRAs. In contrast to prior studies that focus on retraining LoRA to alleviate feature conflicts, our analysis reveals that simply spatially confining the subject LoRA's output to its target region and preventing other LoRAs from directly intruding into this area is sufficient for effective mitigation. Accordingly, we implement Adaptive Token-Level Routing during the inference phase. We introduce FreeFuseAttn, a mechanism that exploits the flow matching model's intrinsic semantic alignment to dynamically match subject-specific tokens to their corresponding spatial regions at early denoising timesteps, thereby bypassing the need for external segmentors. FreeFuse distinguishes itself through high practicality: it necessitates no additional training, model modifications, or user-defined masks spatial conditions. Users need only provide subject activation words to achieve seamless integration into standard workflows. Extensive experiments validate that FreeFuse outperforms existing approaches in both identity preservation and compositional fidelity. Our code is available at https://github.com/yaoliliu/FreeFuse.
>
---
#### [replaced 039] Online Navigation Refinement: Achieving Lane-Level Guidance by Associating Standard-Definition and Online Perception Maps
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.07487v4](https://arxiv.org/pdf/2507.07487v4)**

> **作者:** Jiaxu Wan; Xu Wang; Mengwei Xie; Xinyuan Chang; Xinran Liu; Zheng Pan; Mu Xu; Hong Zhang; Ding Yuan; Yifan Yang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Lane-level navigation is critical for geographic information systems and navigation-based tasks, offering finer-grained guidance than road-level navigation by standard definition (SD) maps. However, it currently relies on expansive global HD maps that cannot adapt to dynamic road conditions. Recently, online perception (OP) maps have become research hotspots, providing real-time geometry as an alternative, but lack the global topology needed for navigation. To address these issues, Online Navigation Refinement (ONR), a new mission is introduced that refines SD-map-based road-level routes into accurate lane-level navigation by associating SD maps with OP maps. The map-to-map association to handle many-to-one lane-to-road mappings under two key challenges: (1) no public dataset provides lane-to-road correspondences; (2) severe misalignment from spatial fluctuations, semantic disparities, and OP map noise invalidates traditional map matching. For these challenges, We contribute: (1) Online map association dataset (OMA), the first ONR benchmark with 30K scenarios and 2.6M annotated lane vectors; (2) MAT, a transformer with path-aware attention to aligns topology despite spatial fluctuations and semantic disparities and spatial attention for integrates noisy OP features via global context; and (3) NR P-R, a metric evaluating geometric and semantic alignment. Experiments show that MAT outperforms existing methods at 34 ms latency, enabling low-cost and up-to-date lane-level navigation.
>
---
#### [replaced 040] Beyond Retraining: Training-Free Unknown Class Filtering for Source-Free Open Set Domain Adaptation of Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.14224v2](https://arxiv.org/pdf/2504.14224v2)**

> **作者:** Yongguang Li; Jindong Li; Qi Wang; Qianli Xing; Runliang Niu; Shengsheng Wang; Menglin Yang
>
> **备注:** Core methods unchanged; title updated and full-text narrative refined for clarity and logical coherence. No changes to key findings and conclusions
>
> **摘要:** Vision-language models (VLMs) have gained widespread attention for their strong zero-shot capabilities across numerous downstream tasks. However, these models assume that each test image's class label is drawn from a predefined label set and lack a reliable mechanism to reject samples from emerging unknown classes when only unlabeled data are available. To address this gap, open-set domain adaptation methods retrain models to push potential unknowns away from known clusters. Yet, some unknown samples remain stably anchored to specific known classes in the VLM feature space due to semantic relevance, which is termed as Semantic Affinity Anchoring (SAA). Forcibly repelling these samples unavoidably distorts the native geometry of VLMs and degrades performance. Meanwhile, existing score-based unknown detectors use simplistic thresholds and suffer from threshold sensitivity, resulting in sub-optimal performance. To address aforementioned issues, we propose VLM-OpenXpert, which comprises two training-free, plug-and-play inference modules. SUFF performs SVD on high-confidence unknowns to extract a low-rank "unknown subspace". Each sample's projection onto this subspace is weighted and softly removed from its feature, suppressing unknown components while preserving semantics. BGAT corrects score skewness via a Box-Cox transform, then fits a bimodal Gaussian mixture to adaptively estimate the optimal threshold balancing known-class recognition and unknown-class rejection. Experiments on 9 benchmarks and three backbones (CLIP, SigLIP, ALIGN) under source-free OSDA settings show that our training-free pipeline matches or outperforms retraining-heavy state-of-the-art methods, establishing a powerful lightweight inference calibration paradigm for open-set VLM deployment.
>
---
#### [replaced 041] Causality-guided Prompt Learning for Vision-language Models via Visual Granulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.03803v4](https://arxiv.org/pdf/2509.03803v4)**

> **作者:** Mengyu Gao; Qiulei Dong
>
> **备注:** Updated version
>
> **摘要:** Prompt learning has recently attracted much attention for adapting pre-trained vision-language models (e.g., CLIP) to downstream recognition tasks. However, most of the existing CLIP-based prompt learning methods only show a limited ability for handling fine-grained datasets. To address this issue, we propose a causality-guided text prompt learning method via visual granulation for CLIP, called CaPL, where the explored visual granulation technique could construct sets of visual granules for the text prompt to capture subtle discrepancies among different fine-grained classes through casual inference. The CaPL method contains the following two modules: (1) An attribute disentanglement module is proposed to decompose visual features into non-individualized attributes (shared by some classes) and individualized attributes (specific to single classes) using a Brownian Bridge Diffusion Model; (2) A granule learning module is proposed to construct visual granules by integrating the aforementioned attributes for recognition under two causal inference strategies. Thanks to the learned visual granules, more discriminative text prompt is expected to be learned. Extensive experimental results on 15 datasets demonstrate that our CaPL method significantly outperforms the state-of-the-art prompt learning methods, especially on fine-grained datasets.
>
---
#### [replaced 042] DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.20766v2](https://arxiv.org/pdf/2510.20766v2)**

> **作者:** Noam Issachar; Guy Yariv; Sagie Benaim; Yossi Adi; Dani Lischinski; Raanan Fattal
>
> **摘要:** Diffusion Transformer models can generate images with remarkable fidelity and detail, yet training them at ultra-high resolutions remains extremely costly due to the self-attention mechanism's quadratic scaling with the number of image tokens. In this paper, we introduce Dynamic Position Extrapolation (DyPE), a novel, training-free method that enables pre-trained diffusion transformers to synthesize images at resolutions far beyond their training data, with no additional sampling cost. DyPE takes advantage of the spectral progression inherent to the diffusion process, where low-frequency structures converge early, while high-frequencies take more steps to resolve. Specifically, DyPE dynamically adjusts the model's positional encoding at each diffusion step, matching their frequency spectrum with the current stage of the generative process. This approach allows us to generate images at resolutions that exceed the training resolution dramatically, e.g., 16 million pixels using FLUX. On multiple benchmarks, DyPE consistently improves performance and achieves state-of-the-art fidelity in ultra-high-resolution image generation, with gains becoming even more pronounced at higher resolutions. Project page is available at https://noamissachar.github.io/DyPE/.
>
---
#### [replaced 043] PCICF: A Pedestrian Crossing Identification and Classification Framework
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24386v3](https://arxiv.org/pdf/2509.24386v3)**

> **作者:** Junyi Gu; Beatriz Cabrero-Daniel; Ali Nouri; Lydia Armini; Christian Berger
>
> **摘要:** We have recently observed the commercial roll-out of robotaxis in various countries. They are deployed within an operational design domain (ODD) on specific routes and environmental conditions, and are subject to continuous monitoring to regain control in safety-critical situations. Since ODDs typically cover urban areas, robotaxis must reliably detect vulnerable road users (VRUs) such as pedestrians, bicyclists, or e-scooter riders. To better handle such varied traffic situations, end-to-end AI, which directly compute vehicle control actions from multi-modal sensor data instead of only for perception, is on the rise. High quality data is needed for systematically training and evaluating such systems within their OOD. In this work, we propose PCICF, a framework to systematically identify and classify VRU situations to support ODD's incident analysis. We base our work on the existing synthetic dataset SMIRK, and enhance it by extending its single-pedestrian-only design into the MoreSMIRK dataset, a structured dictionary of multi-pedestrian crossing situations constructed systematically. We then use space-filling curves (SFCs) to transform multi-dimensional features of scenarios into characteristic patterns, which we match with corresponding entries in MoreSMIRK. We evaluate PCICF with the large real-world dataset PIE, which contains more than 150 manually annotated pedestrian crossing videos. We show that PCICF can successfully identify and classify complex pedestrian crossings, even when groups of pedestrians merge or split. By leveraging computationally efficient components like SFCs, PCICF has even potential to be used onboard of robotaxis for OOD detection for example. We share an open-source replication package for PCICF containing its algorithms, the complete MoreSMIRK dataset and dictionary, as well as our experiment results presented in: https://github.com/Claud1234/PCICF
>
---
#### [replaced 044] UniHash: Unifying Pointwise and Pairwise Hashing Paradigms
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.09828v3](https://arxiv.org/pdf/2601.09828v3)**

> **作者:** Xiaoxu Ma; Runhao Li; Xiangbo Zhang; Zhenyu Weng
>
> **摘要:** Effective retrieval across both seen and unseen categories is crucial for modern image retrieval systems. Retrieval on seen categories ensures precise recognition of known classes, while retrieval on unseen categories promotes generalization to novel classes with limited supervision. However, most existing deep hashing methods are confined to a single training paradigm, either pointwise or pairwise, where the former excels on seen categories and the latter generalizes better to unseen ones. To overcome this limitation, we propose Unified Hashing (UniHash), a dual-branch framework that unifies the strengths of both paradigms to achieve balanced retrieval performance across seen and unseen categories. UniHash consists of two complementary branches: a center-based branch following the pointwise paradigm and a pairwise branch following the pairwise paradigm. A novel hash code learning method is introduced to enable bidirectional knowledge transfer between branches, improving hash code discriminability and generalization. It employs a mutual learning loss to align hash representations and introduces a Split-Merge Mixture of Hash Experts (SM-MoH) module to enhance cross-branch exchange of hash representations. Theoretical analysis substantiates the effectiveness of UniHash, and extensive experiments on CIFAR-10, MSCOCO, and ImageNet demonstrate that UniHash consistently achieves state-of-the-art performance in both seen and unseen image retrieval scenarios.
>
---
#### [replaced 045] Efficient Spike-driven Transformer for High-performance Drone-View Geo-Localization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19365v2](https://arxiv.org/pdf/2512.19365v2)**

> **作者:** Zhongwei Chen; Hai-Jun Rong; Zhao-Xu Yang; Guoqi Li
>
> **摘要:** Traditional drone-view geo-localization (DVGL) methods based on artificial neural networks (ANNs) have achieved remarkable performance. However, ANNs rely on dense computation, which results in high power consumption. In contrast, spiking neural networks (SNNs), which benefit from spike-driven computation, inherently provide low power consumption. Regrettably, the potential of SNNs for DVGL has yet to be thoroughly investigated. Meanwhile, the inherent sparsity of spike-driven computation for representation learning scenarios also results in loss of critical information and difficulties in learning long-range dependencies when aligning heterogeneous visual data sources. To address these, we propose SpikeViMFormer, the first SNN framework designed for DVGL. In this framework, a lightweight spike-driven transformer backbone is adopted to extract coarse-grained features. To mitigate the loss of critical information, the spike-driven selective attention (SSA) block is designed, which uses a spike-driven gating mechanism to achieve selective feature enhancement and highlight discriminative regions. Furthermore, a spike-driven hybrid state space (SHS) block is introduced to learn long-range dependencies using a hybrid state space. Moreover, only the backbone is utilized during the inference stage to reduce computational cost. To ensure backbone effectiveness, a novel hierarchical re-ranking alignment learning (HRAL) strategy is proposed. It refines features via neighborhood re-ranking and maintains cross-batch consistency to directly optimize the backbone. Experimental results demonstrate that SpikeViMFormer outperforms state-of-the-art SNNs. Compared with advanced ANNs, it also achieves competitive performance.Our code is available at https://github.com/ISChenawei/SpikeViMFormer
>
---
#### [replaced 046] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.03779v3](https://arxiv.org/pdf/2507.03779v3)**

> **作者:** Jiaqi Zhang; Juntuo Wang; Zhixin Sun; John Zou; Randall Balestriero
>
> **备注:** Accepted by 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Large-scale vision foundation models such as DINOv2 boast impressive performances by leveraging massive architectures and training datasets. But numerous scenarios require practitioners to reproduce those pre-training solutions, such as on private data, new modalities, or simply for scientific questioning--which is currently extremely demanding computation-wise. We thus propose a novel pre-training strategy for DINOv2 that simultaneously accelerates convergence--and strengthens robustness to common corruptions as a by-product. Our approach involves a frequency filtering curriculum--low-frequency being seen first--and the Gaussian noise patching augmentation. Applied to a ViT-B/16 backbone trained on ImageNet-1K, while pre-training time and FLOPs are reduced by 1.6x and 2.25x, our method still achieves matching robustness in corruption benchmarks (ImageNet-C) and maintains competitive linear probing performance compared with baseline. This dual benefit of efficiency and robustness makes large-scale self-supervised foundation modeling more attainable, while opening the door to novel exploration around data curriculum and augmentation as means to improve self-supervised learning models robustness. The code is available at https://github.com/KevinZ0217/fast_dinov2
>
---
#### [replaced 047] Reliable Deep Learning for Small-Scale Classifications: Experiments on Real-World Image Datasets from Bangladesh
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11911v2](https://arxiv.org/pdf/2601.11911v2)**

> **作者:** Alfe Suny; MD Sakib Ul Islam; Md. Imran Hossain
>
> **摘要:** Convolutional neural networks (CNNs) have achieved state-of-the-art performance in image recognition tasks but often involve complex architectures that may overfit on small datasets. In this study, we evaluate a compact CNN across five publicly available, real-world image datasets from Bangladesh, including urban encroachment, vehicle detection, road damage, and agricultural crops. The network demonstrates high classification accuracy, efficient convergence, and low computational overhead. Quantitative metrics and saliency analyses indicate that the model effectively captures discriminative features and generalizes robustly across diverse scenarios, highlighting the suitability of streamlined CNN architectures for small-class image classification tasks.
>
---
#### [replaced 048] InternSVG: Towards Unified SVG Tasks with Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.11341v3](https://arxiv.org/pdf/2510.11341v3)**

> **作者:** Haomin Wang; Jinhui Yin; Qi Wei; Wenguang Zeng; Lixin Gu; Shenglong Ye; Zhangwei Gao; Yaohui Wang; Yanting Zhang; Yuanqi Li; Yanwen Guo; Wenhai Wang; Kai Chen; Yu Qiao; Hongjie Zhang
>
> **摘要:** General SVG modeling remains challenging due to fragmented datasets, limited transferability of methods across tasks, and the difficulty of handling structural complexity. In response, we leverage the strong transfer and generalization capabilities of multimodal large language models (MLLMs) to achieve unified modeling for SVG understanding, editing, and generation. We present the InternSVG family, an integrated data-benchmark-model suite. At its core is SAgoge, the largest and most comprehensive multimodal dataset for SVG tasks, encompassing both static graphics and dynamic animations. It covers icons, long-sequence illustrations, scientific diagrams, and dynamic animations, supporting tasks of varied difficulty levels and providing deeper hierarchies with richer attributes compared to previous datasets. Based on this resource, we introduce SArena, a companion benchmark with comprehensive task definitions and standardized evaluation that aligns with the domains and difficulty spectrum covered by SAgoge. Building on these foundations, we propose InternSVG, a unified MLLM for SVG understanding, editing, and generation with SVG-specific special tokens, subword-based embedding initialization, and a two-stage training strategy that progresses from short static SVGs to long-sequence illustrations and complex animations. This unified formulation induces positive transfer and improves overall performance. Experiments on SArena and prior benchmark confirm that InternSVG achieves substantial gains and consistently outperforms leading open and proprietary counterparts.
>
---
#### [replaced 049] TIPO: Text to Image with Text Presampling for Prompt Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.08127v5](https://arxiv.org/pdf/2411.08127v5)**

> **作者:** Shih-Ying Yeh; Sang-Hyun Park; Yi Li; Giyeong Oh; Xuehai Wang; Min Song; Youngjae Yu; Shang-Hong Lai
>
> **备注:** 50 pages, 28 figures
>
> **摘要:** TIPO (Text-to-Image Prompt Optimization) introduces an efficient approach for automatic prompt refinement in text-to-image (T2I) generation. Starting from simple user prompts, TIPO leverages a lightweight pre-trained model to expand these prompts into richer and more detailed versions. Conceptually, TIPO samples refined prompts from a targeted sub-distribution within the broader semantic space, preserving the original intent while significantly improving visual quality, coherence, and detail. Unlike resource-intensive methods based on large language models (LLMs) or reinforcement learning (RL), TIPO offers strong computational efficiency and scalability, opening new possibilities for effective automated prompt engineering in T2I tasks. Extensive experiments across multiple domains demonstrate that TIPO achieves stronger text alignment, reduced visual artifacts, and consistently higher human preference rates, while maintaining competitive aesthetic quality. These results highlight the effectiveness of distribution-aligned prompt engineering and point toward broader opportunities for scalable, automated refinement in text-to-image generation.
>
---
#### [replaced 050] Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.20354v2](https://arxiv.org/pdf/2601.20354v2)**

> **作者:** Zengbin Wang; Xuecai Hu; Yong Wang; Feng Xiong; Man Zhang; Xiangxiang Chu
>
> **备注:** Accepted by ICLR 2026, URL: https://github.com/AMAP-ML/SpatialGenEval
>
> **摘要:** Text-to-image (T2I) models have achieved remarkable success in generating high-fidelity images, but they often fail in handling complex spatial relationships, e.g., spatial perception, reasoning, or interaction. These critical aspects are largely overlooked by current benchmarks due to their short or information-sparse prompt design. In this paper, we introduce SpatialGenEval, a new benchmark designed to systematically evaluate the spatial intelligence of T2I models, covering two key aspects: (1) SpatialGenEval involves 1,230 long, information-dense prompts across 25 real-world scenes. Each prompt integrates 10 spatial sub-domains and corresponding 10 multi-choice question-answer pairs, ranging from object position and layout to occlusion and causality. Our extensive evaluation of 21 state-of-the-art models reveals that higher-order spatial reasoning remains a primary bottleneck. (2) To demonstrate that the utility of our information-dense design goes beyond simple evaluation, we also construct the SpatialT2I dataset. It contains 15,400 text-image pairs with rewritten prompts to ensure image consistency while preserving information density. Fine-tuned results on current foundation models (i.e., Stable Diffusion-XL, Uniworld-V1, OmniGen2) yield consistent performance gains (+4.2%, +5.7%, +4.4%) and more realistic effects in spatial relations, highlighting a data-centric paradigm to achieve spatial intelligence in T2I models.
>
---
#### [replaced 051] MuSLR: Multimodal Symbolic Logical Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25851v2](https://arxiv.org/pdf/2509.25851v2)**

> **作者:** Jundong Xu; Hao Fei; Yuhui Zhang; Liangming Pan; Qijun Huang; Qian Liu; Preslav Nakov; Min-Yen Kan; William Yang Wang; Mong-Li Lee; Wynne Hsu
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Multimodal symbolic logical reasoning, which aims to deduce new facts from multimodal input via formal logic, is critical in high-stakes applications such as autonomous driving and medical diagnosis, as its rigorous, deterministic reasoning helps prevent serious consequences. To evaluate such capabilities of current state-of-the-art vision language models (VLMs), we introduce the first benchmark MuSLR for multimodal symbolic logical reasoning grounded in formal logical rules. MuSLR comprises 1,093 instances across 7 domains, including 35 atomic symbolic logic and 976 logical combinations, with reasoning depths ranging from 2 to 9. We evaluate 7 state-of-the-art VLMs on MuSLR and find that they all struggle with multimodal symbolic reasoning, with the best model, GPT-4.1, achieving only 46.8%. Thus, we propose LogiCAM, a modular framework that applies formal logical rules to multimodal inputs, boosting GPT-4.1's Chain-of-Thought performance by 14.13%, and delivering even larger gains on complex logics such as first-order logic. We also conduct a comprehensive error analysis, showing that around 70% of failures stem from logical misalignment between modalities, offering key insights to guide future improvements. All data and code are publicly available at https://llm-symbol.github.io/MuSLR.
>
---
#### [replaced 052] CycleDiff: Cycle Diffusion Models for Unpaired Image-to-image Translation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.06625v3](https://arxiv.org/pdf/2508.06625v3)**

> **作者:** Shilong Zou; Yuhang Huang; Renjiao Yi; Chenyang Zhu; Kai Xu
>
> **备注:** Accepted by IEEE TIP 2026
>
> **摘要:** We introduce a diffusion-based cross-domain image translator in the absence of paired training data. Unlike GAN-based methods, our approach integrates diffusion models to learn the image translation process, allowing for more coverable modeling of the data distribution and performance improvement of the cross-domain translation. However, incorporating the translation process within the diffusion process is still challenging since the two processes are not aligned exactly, i.e., the diffusion process is applied to the noisy signal while the translation process is conducted on the clean signal. As a result, recent diffusion-based studies employ separate training or shallow integration to learn the two processes, yet this may cause the local minimal of the translation optimization, constraining the effectiveness of diffusion models. To address the problem, we propose a novel joint learning framework that aligns the diffusion and the translation process, thereby improving the global optimality. Specifically, we propose to extract the image components with diffusion models to represent the clean signal and employ the translation process with the image components, enabling an end-to-end joint learning manner. On the other hand, we introduce a time-dependent translation network to learn the complex translation mapping, resulting in effective translation learning and significant performance improvement. Benefiting from the design of joint learning, our method enables global optimization of both processes, enhancing the optimality and achieving improved fidelity and structural consistency. We have conducted extensive experiments on RGB$\leftrightarrow$RGB and diverse cross-modality translation tasks including RGB$\leftrightarrow$Edge, RGB$\leftrightarrow$Semantics and RGB$\leftrightarrow$Depth, showcasing better generative performances than the state of the arts.
>
---
#### [replaced 053] SEGA: A Transferable Signed Ensemble Gaussian Black-Box Attack against No-Reference Image Quality Assessment Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.18546v2](https://arxiv.org/pdf/2509.18546v2)**

> **作者:** Yujia Liu; Dingquan Li; Zhixuan Li; Tiejun Huang
>
> **摘要:** No-Reference Image Quality Assessment (NR-IQA) models play an important role in various real-world applications. Recently, adversarial attacks against NR-IQA models have attracted increasing attention, as they provide valuable insights for revealing model vulnerabilities and guiding robust system design. Some effective attacks have been proposed against NR-IQA models in white-box settings, where the attacker has full access to the target model. However, these attacks often suffer from poor transferability to unknown target models in more realistic black-box scenarios, where the target model is inaccessible. This work makes the first attempt to address the challenge of low transferability in attacking NR-IQA models by proposing a transferable Signed Ensemble Gaussian black-box Attack (SEGA). The main idea is to approximate the gradient of the target model by applying Gaussian smoothing to source models and ensembling their smoothed gradients. To ensure the imperceptibility of adversarial perturbations, SEGA further removes inappropriate perturbations using a specially designed perturbation filter mask. Experimental results on the CLIVE dataset demonstrate the superior transferability of SEGA, validating its effectiveness in enabling successful transfer-based black-box attacks against NR-IQA models.
>
---
#### [replaced 054] MARE: Multimodal Alignment and Reinforcement for Explainable Deepfake Detection via Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.20433v2](https://arxiv.org/pdf/2601.20433v2)**

> **作者:** Wenbo Xu; Wei Lu; Xiangyang Luo; Jiantao Zhou
>
> **摘要:** Deepfake detection is a widely researched topic that is crucial for combating the spread of malicious content, with existing methods mainly modeling the problem as classification or spatial localization. The rapid advancements in generative models impose new demands on Deepfake detection. In this paper, we propose multimodal alignment and reinforcement for explainable Deepfake detection via vision-language models, termed MARE, which aims to enhance the accuracy and reliability of Vision-Language Models (VLMs) in Deepfake detection and reasoning. Specifically, MARE designs comprehensive reward functions, incorporating reinforcement learning from human feedback (RLHF), to incentivize the generation of text-spatially aligned reasoning content that adheres to human preferences. Besides, MARE introduces a forgery disentanglement module to capture intrinsic forgery traces from high-level facial semantics, thereby improving its authenticity detection capability. We conduct thorough evaluations on the reasoning content generated by MARE. Both quantitative and qualitative experimental results demonstrate that MARE achieves state-of-the-art performance in terms of accuracy and reliability.
>
---
#### [replaced 055] CMOOD: Concept-based Multi-label OOD Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.13578v3](https://arxiv.org/pdf/2411.13578v3)**

> **作者:** Zhendong Liu; Yi Nian; Yuehan Qin; Henry Peng Zou; Li Li; Xiyang Hu; Yue Zhao
>
> **摘要:** How can models effectively detect out-of-distribution (OOD) samples in complex, multi-label settings without extensive retraining? Existing OOD detection methods struggle to capture the intricate semantic relationships and label co-occurrences inherent in multi-label settings, often requiring large amounts of training data and failing to generalize to unseen label combinations. While large language models have revolutionized zero-shot OOD detection, they primarily focus on single-label scenarios, leaving a critical gap in handling real-world tasks where samples can be associated with multiple interdependent labels. To address these challenges, we introduce COOD, a novel zero-shot multi-label OOD detection framework. COOD leverages pre-trained vision-language models, enhancing them with a concept-based label expansion strategy and a new scoring function. By enriching the semantic space with both positive and negative concepts for each label, our approach models complex label dependencies, precisely differentiating OOD samples without the need for additional training. Extensive experiments demonstrate that our method significantly outperforms existing approaches, achieving approximately 95% average AUROC on both VOC and COCO datasets, while maintaining robust performance across varying numbers of labels and different types of OOD samples.
>
---
#### [replaced 056] iPEAR: Iterative Pyramid Estimation with Attention and Residuals for Deformable Medical Image Registration
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.07666v2](https://arxiv.org/pdf/2510.07666v2)**

> **作者:** Heming Wu; Di Wang; Tai Ma; Peng Zhao; Yubin Xiao; Zhongke Wu; Xing-Ce Wang; Xuan Wu; You Zhou
>
> **摘要:** Existing pyramid registration networks may accumulate anatomical misalignments and lack an effective mechanism to dynamically determine the number of optimization iterations under varying deformation requirements across images, leading to degraded performance. To solve these limitations, we propose iPEAR. Specifically, iPEAR adopts our proposed Fused Attention-Residual Module (FARM) for decoding, which comprises an attention pathway and a residual pathway to alleviate the accumulation of anatomical misalignment. We further propose a dual-stage Threshold-Controlled Iterative (TCI) strategy that adaptively determines the number of optimization iterations for varying images by evaluating registration stability and convergence. Extensive experiments on three public brain MRI datasets and one public abdomen CT dataset show that iPEAR outperforms state-of-the-art (SOTA) registration networks in terms of accuracy, while achieving on-par inference speed and model parameter size. Generalization and ablation studies further validate the effectiveness of the proposed FARM and TCI.
>
---
#### [replaced 057] A Coreset Selection of Coreset Selection Literature: Introduction and Recent Advances
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.17799v2](https://arxiv.org/pdf/2505.17799v2)**

> **作者:** Brian B. Moser; Arundhati S. Shanbhag; Stanislav Frolov; Federico Raue; Joachim Folz; Andreas Dengel
>
> **摘要:** Coreset selection targets the challenge of finding a small, representative subset of a large dataset that preserves essential patterns for effective machine learning. Although several surveys have examined data reduction strategies before, most focus narrowly on either classical geometry-based methods or active learning techniques. In contrast, this survey presents a more comprehensive view by unifying three major lines of coreset research, namely, training-free, training-oriented, and label-free approaches, into a single taxonomy. We present subfields often overlooked by existing work, including submodular formulations, bilevel optimization, and recent progress in pseudo-labeling for unlabeled datasets. Additionally, we examine how pruning strategies influence generalization and neural scaling laws, offering new insights that are absent from prior reviews. Finally, we compare these methods under varying computational, robustness, and performance demands and highlight open challenges, such as robustness, outlier filtering, and adapting coreset selection to foundation models, for future research.
>
---
#### [replaced 058] RestoRect: Degraded Image Restoration via Latent Rectified Flow & Feature Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23480v2](https://arxiv.org/pdf/2509.23480v2)**

> **作者:** Shourya Verma; Mengbo Wang; Nadia Atallah Lanman; Ananth Grama
>
> **摘要:** Current approaches for restoration of degraded images face a trade-off: high-performance models are slow for practical use, while fast models produce poor results. Knowledge distillation transfers teacher knowledge to students, but existing static feature matching methods cannot capture how modern transformer architectures dynamically generate features. We propose a novel Latent Rectified Flow Feature Distillation method for restoring degraded images called \textbf{'RestoRect'}. We apply rectified flow to reformulate feature distillation as a generative process where students learn to synthesize teacher-quality features through learnable trajectories in latent space. Our framework combines Retinex decomposition with learnable anisotropic diffusion constraints, and trigonometric color space polarization. We introduce a Feature Layer Extraction loss for robust knowledge transfer between different network architectures through cross-normalized transformer feature alignment with percentile-based outlier detection. RestoRect achieves better training stability, and faster convergence and inference while preserving restoration quality, demonstrating superior results across 15 image restoration datasets, covering 4 tasks, on 10 metrics against baselines.
>
---
#### [replaced 059] PRISM: A Framework Harnessing Unsupervised Visual Representations and Textual Prompts for Explainable MACE Survival Prediction from Cardiac Cine MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.19325v2](https://arxiv.org/pdf/2508.19325v2)**

> **作者:** Haoyang Su; Jin-Yi Xiang; Shaohao Rui; Yifan Gao; Xingyu Chen; Tingxuan Yin; Shaoting Zhang; Xiaosong Wang; Lian-Ming Wu
>
> **摘要:** Accurate prediction of major adverse cardiac events (MACE) remains a central challenge in cardiovascular prognosis. We present PRISM (Prompt-guided Representation Integration for Survival Modeling), a self-supervised framework that integrates visual representations from non-contrast cardiac cine magnetic resonance imaging with structured electronic health records (EHRs) for survival analysis. PRISM extracts temporally synchronized imaging features through motion-aware multi-view distillation and modulates them using medically informed textual prompts to enable fine-grained risk prediction. Across four independent clinical cohorts, PRISM consistently surpasses classical survival prediction models and state-of-the-art (SOTA) deep learning baselines under internal and external validation. Further clinical findings demonstrate that the combined imaging and EHR representations derived from PRISM provide valuable insights into cardiac risk across diverse cohorts. Three distinct imaging signatures associated with elevated MACE risk are uncovered, including lateral wall dyssynchrony, inferior wall hypersensitivity, and anterior elevated focus during diastole. Prompt-guided attribution further identifies hypertension, diabetes, and smoking as dominant contributors among clinical and physiological EHR factors.
>
---
#### [replaced 060] An explainable vision transformer with transfer learning based efficient drought stress identification
- **分类: cs.CV; cs.AI; cs.ET; cs.LG**

- **链接: [https://arxiv.org/pdf/2407.21666v3](https://arxiv.org/pdf/2407.21666v3)**

> **作者:** Aswini Kumar Patra; Ankit Varshney; Lingaraj Sahoo
>
> **备注:** 33 pages, 7 figures, 8 tables
>
> **摘要:** Early detection of drought stress is critical for taking timely measures for reducing crop loss before the drought impact becomes irreversible. The subtle phenotypical and physiological changes in response to drought stress are captured by non-invasive imaging techniques and these imaging data serve as valuable resource for machine learning methods to identify drought stress. While convolutional neural networks (CNNs) are in wide use, vision transformers (ViTs) present a promising alternative in capturing long-range dependencies and intricate spatial relationships, thereby enhancing the detection of subtle indicators of drought stress. We propose an explainable deep learning pipeline that leverages the power of ViTs for drought stress detection in potato crops using aerial imagery. We applied two distinct approaches: a synergistic combination of ViT and support vector machine (SVM), where ViT extracts intricate spatial features from aerial images, and SVM classifies the crops as stressed or healthy and an end-to-end approach using a dedicated classification layer within ViT to directly detect drought stress. Our key findings explain the ViT model's decision-making process by visualizing attention maps. These maps highlight the specific spatial features within the aerial images that the ViT model focuses as the drought stress signature. Our findings demonstrate that the proposed methods not only achieve high accuracy in drought stress identification but also shedding light on the diverse subtle plant features associated with drought stress. This offers a robust and interpretable solution for drought stress monitoring for farmers to undertake informed decisions for improved crop management.
>
---
#### [replaced 061] MiLDEdit: Reasoning-Based Multi-Layer Design Document Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.04589v2](https://arxiv.org/pdf/2601.04589v2)**

> **作者:** Zihao Lin; Wanrong Zhu; Jiuxiang Gu; Jihyung Kil; Christopher Tensmeyer; Lin Zhang; Shilong Liu; Ruiyi Zhang; Lifu Huang; Vlad I. Morariu; Tong Sun
>
> **摘要:** Real-world design documents (e.g., posters) are inherently multi-layered, combining decoration, text, and images. Editing them from natural-language instructions requires fine-grained, layer-aware reasoning to identify relevant layers and coordinate modifications. Prior work largely overlooks multi-layer design document editing, focusing instead on single-layer image editing or multi-layer generation, which assume a flat canvas and lack the reasoning needed to determine what and where to modify. To address this gap, we introduce the Multi-Layer Document Editing Agent (MiLDEAgent), a reasoning-based framework that combines an RL-trained multimodal reasoner for layer-wise understanding with an image editor for targeted modifications. To systematically benchmark this setting, we introduce the MiLDEBench, a human-in-the-loop corpus of over 20K design documents paired with diverse editing instructions. The benchmark is complemented by a task-specific evaluation protocol, MiLDEEval, which spans four dimensions including instruction following, layout consistency, aesthetics, and text rendering. Extensive experiments on 14 open-source and 2 closed-source models reveal that existing approaches fail to generalize: open-source models often cannot complete multi-layer document editing tasks, while closed-source models suffer from format violations. In contrast, MiLDEAgent achieves strong layer-aware reasoning and precise editing, significantly outperforming all open-source baselines and attaining performance comparable to closed-source models, thereby establishing the first strong baseline for multi-layer document editing.
>
---
#### [replaced 062] From Limited Labels to Open Domains:An Efficient Learning Method for Drone-view Geo-Localization
- **分类: cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2503.07520v4](https://arxiv.org/pdf/2503.07520v4)**

> **作者:** Zhongwei Chen; Zhao-Xu Yang; Hai-Jun Rong; Jiawei Lang; Guoqi Li
>
> **摘要:** Traditional supervised drone-view geo-localization (DVGL) methods heavily depend on paired training data and encounter difficulties in learning cross-view correlations from unpaired data. Moreover, when deployed in a new domain, these methods require obtaining the new paired data and subsequent retraining for model adaptation, which significantly increases computational overhead. Existing unsupervised methods have enabled to generate pseudo-labels based on cross-view similarity to infer the pairing relationships. However, geographical similarity and spatial continuity often cause visually analogous features at different geographical locations. The feature confusion compromises the reliability of pseudo-label generation, where incorrect pseudo-labels drive negative optimization. Given these challenges inherent in both supervised and unsupervised DVGL methods, we propose a novel cross-domain invariant knowledge transfer network (CDIKTNet) with limited supervision, whose architecture consists of a cross-domain invariance sub-network (CDIS) and a cross-domain transfer sub-network (CDTS). This architecture facilitates a closed-loop framework for invariance feature learning and knowledge transfer. The CDIS is designed to learn cross-view structural and spatial invariance from a small amount of paired data that serves as prior knowledge. It endows the shared feature space of unpaired data with similar implicit cross-view correlations at initialization, which alleviates feature confusion. Based on this, the CDTS employs dual-path contrastive learning to further optimize each subspace while preserving consistency in a shared feature space. Extensive experiments demonstrate that CDIKTNet achieves state-of-the-art performance under full supervision compared with those supervised methods, and further surpasses existing unsupervised methods in both few-shot and cross-domain initialization.
>
---
#### [replaced 063] Continual GUI Agents
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2601.20732v2](https://arxiv.org/pdf/2601.20732v2)**

> **作者:** Ziwei Liu; Borui Kang; Hangjie Yuan; Zixiang Zhao; Wei Li; Yifan Zhu; Tao Feng
>
> **摘要:** As digital environments (data distribution) are in flux, with new GUI data arriving over time-introducing new domains or resolutions-agents trained on static environments deteriorate in performance. In this work, we introduce Continual GUI Agents, a new task that requires GUI agents to perform continual learning under shifted domains and resolutions. We find existing methods fail to maintain stable grounding as GUI distributions shift over time, due to the diversity of UI interaction points and regions in fluxing scenarios. To address this, we introduce GUI-Anchoring in Flux (GUI-AiF), a new reinforcement fine-tuning framework that stabilizes continual learning through two novel rewards: Anchoring Point Reward in Flux (APR-iF) and Anchoring Region Reward in Flux (ARR-iF). These rewards guide the agents to align with shifting interaction points and regions, mitigating the tendency of existing reward strategies to over-adapt to static grounding cues (e.g., fixed coordinates or element scales). Extensive experiments show GUI-AiF surpasses state-of-the-art baselines. Our work establishes the first continual learning framework for GUI agents, revealing the untapped potential of reinforcement fine-tuning for continual GUI Agents.
>
---
#### [replaced 064] SSCATeR: Sparse Scatter-Based Convolution Algorithm with Temporal Data Recycling for Real-Time 3D Object Detection in LiDAR Point Clouds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.08557v3](https://arxiv.org/pdf/2512.08557v3)**

> **作者:** Alexander Dow; Manduhu Manduhu; Matheus Santos; Ben Bartlett; Gerard Dooly; James Riordan
>
> **备注:** 23 Pages, 27 Figures, This work has been accepted for publication by the IEEE Sensors Journal. Please see the first page of the article PDF for copyright information
>
> **摘要:** This work leverages the continuous sweeping motion of LiDAR scanning to concentrate object detection efforts on specific regions that receive a change in point data from one frame to another. We achieve this by using a sliding time window with short strides and consider the temporal dimension by storing convolution results between passes. This allows us to ignore unchanged regions, significantly reducing the number of convolution operations per forward pass without sacrificing accuracy. This data reuse scheme introduces extreme sparsity to detection data. To exploit this sparsity, we extend our previous work on scatter-based convolutions to allow for data reuse, and as such propose Sparse Scatter-Based Convolution Algorithm with Temporal Data Recycling (SSCATeR). This operation treats incoming LiDAR data as a continuous stream and acts only on the changing parts of the point cloud. By doing so, we achieve the same results with as much as a 6.61-fold reduction in processing time. Our test results show that the feature maps output by our method are identical to those produced by traditional sparse convolution techniques, whilst greatly increasing the computational efficiency of the network.
>
---
#### [replaced 065] Efficient4D: Fast Dynamic 3D Object Generation from a Single-view Video
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2401.08742v5](https://arxiv.org/pdf/2401.08742v5)**

> **作者:** Zijie Pan; Zeyu Yang; Xiatian Zhu; Li Zhang
>
> **备注:** IJCV version
>
> **摘要:** Generating dynamic 3D object from a single-view video is challenging due to the lack of 4D labeled data. An intuitive approach is to extend previous image-to-3D pipelines by transferring off-the-shelf image generation models such as score distillation sampling.However, this approach would be slow and expensive to scale due to the need for back-propagating the information-limited supervision signals through a large pretrained model. To address this, we propose an efficient video-to-4D object generation framework called Efficient4D. It generates high-quality spacetime-consistent images under different camera views, and then uses them as labeled data to directly reconstruct the 4D content through a 4D Gaussian splatting model. Importantly, our method can achieve real-time rendering under continuous camera trajectories. To enable robust reconstruction under sparse views, we introduce inconsistency-aware confidence-weighted loss design, along with a lightly weighted score distillation loss. Extensive experiments on both synthetic and real videos show that Efficient4D offers a remarkable 10-fold increase in speed when compared to prior art alternatives while preserving the quality of novel view synthesis. For example, Efficient4D takes only 10 minutes to model a dynamic object, vs 120 minutes by the previous art model Consistent4D.
>
---
#### [replaced 066] DrivIng: A Large-Scale Multimodal Driving Dataset with Full Digital Twin Integration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.15260v2](https://arxiv.org/pdf/2601.15260v2)**

> **作者:** Dominik Rößle; Xujun Xie; Adithya Mohan; Venkatesh Thirugnana Sambandham; Daniel Cremers; Torsten Schön
>
> **备注:** Copyright 2026 IEEE. This is the accepted manuscript (postprint), not the final published version. For code and dataset, see https://github.com/cvims/DrivIng
>
> **摘要:** Perception is a cornerstone of autonomous driving, enabling vehicles to understand their surroundings and make safe, reliable decisions. Developing robust perception algorithms requires large-scale, high-quality datasets that cover diverse driving conditions and support thorough evaluation. Existing datasets often lack a high-fidelity digital twin, limiting systematic testing, edge-case simulation, sensor modification, and sim-to-real evaluations. To address this gap, we present DrivIng, a large-scale multimodal dataset with a complete geo-referenced digital twin of a ~18 km route spanning urban, suburban, and highway segments. Our dataset provides continuous recordings from six RGB cameras, one LiDAR, and high-precision ADMA-based localization, captured across day, dusk, and night. All sequences are annotated at 10 Hz with 3D bounding boxes and track IDs across 12 classes, yielding ~1.2 million annotated instances. Alongside the benefits of a digital twin, DrivIng enables a 1-to-1 transfer of real traffic into simulation, preserving agent interactions while enabling realistic and flexible scenario testing. To support reproducible research and robust validation, we benchmark DrivIng with state-of-the-art perception models and publicly release the dataset, digital twin, HD map, and codebase.
>
---
#### [replaced 067] JointDiff: Bridging Continuous and Discrete in Multi-Agent Trajectory Generation
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22522v3](https://arxiv.org/pdf/2509.22522v3)**

> **作者:** Guillem Capellera; Luis Ferraz; Antonio Rubio; Alexandre Alahi; Antonio Agudo
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Generative models often treat continuous data and discrete events as separate processes, creating a gap in modeling complex systems where they interact synchronously. To bridge this gap, we introduce JointDiff, a novel diffusion framework designed to unify these two processes by simultaneously generating continuous spatio-temporal data and synchronous discrete events. We demonstrate its efficacy in the sports domain by simultaneously modeling multi-agent trajectories and key possession events. This joint modeling is validated with non-controllable generation and two novel controllable generation scenarios: weak-possessor-guidance, which offers flexible semantic control over game dynamics through a simple list of intended ball possessors, and text-guidance, which enables fine-grained, language-driven generation. To enable the conditioning with these guidance signals, we introduce CrossGuid, an effective conditioning operation for multi-agent domains. We also share a new unified sports benchmark enhanced with textual descriptions for soccer and football datasets. JointDiff achieves state-of-the-art performance, demonstrating that joint modeling is crucial for building realistic and controllable generative models for interactive systems. https://guillem-cf.github.io/JointDiff/
>
---
#### [replaced 068] ViSurf: Visual Supervised-and-Reinforcement Fine-Tuning for Large Vision-and-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.10606v3](https://arxiv.org/pdf/2510.10606v3)**

> **作者:** Yuqi Liu; Liangyu Chen; Jiazhen Liu; Mingkang Zhu; Zhisheng Zhong; Bei Yu; Jiaya Jia
>
> **摘要:** Post-training Large Vision-and-Language Models (LVLMs) typically involves Supervised Fine-Tuning (SFT) for knowledge injection or Reinforcement Learning with Verifiable Rewards (RLVR) for performance enhancement. However, SFT often leads to sub-optimal performance, while RLVR remains constrained by the model's internal knowledge base. While a sequential SFT $\rightarrow$ RLVR pipeline can be used, it introduces significant computational overhead and suffers from catastrophic forgetting. To address these limitations, we propose ViSurf (\textbf{Vi}sual \textbf{Su}pervised-and-\textbf{R}einforcement \textbf{F}ine-Tuning), a unified, single-stage paradigm that integrates the strengths of both SFT and RLVR. By analyzing their training objectives, we establish a unified framework that injects ground-truth labels directly into RLVR rollouts, facilitating simultaneous external supervision and internal reinforcement. Furthermore, we introduce three novel reward control strategies to ensure training stability and optimization. Extensive experiments demonstrate that ViSurf consistently outperforms standalone SFT, RLVR, and the traditional two-stage pipeline across diverse benchmarks. In-depth analysis corroborates these findings, validating the derivation and design principles of ViSurf.
>
---
#### [replaced 069] Can Large Language Models Capture Video Game Engagement?
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于情感识别任务，旨在探究大语言模型能否准确捕捉视频游戏中的用户参与度。通过大量实验，评估LLMs在多模态输入下的表现，并分析影响因素。**

- **链接: [https://arxiv.org/pdf/2502.04379v2](https://arxiv.org/pdf/2502.04379v2)**

> **作者:** David Melhart; Matthew Barthet; Georgios N. Yannakakis
>
> **备注:** This work has been submitted to the IEEE for publication
>
> **摘要:** Can out-of-the-box pretrained Large Language Models (LLMs) detect human affect successfully when observing a video? To address this question, for the first time, we evaluate comprehensively the capacity of popular LLMs for successfully predicting continuous affect annotations of videos when prompted by a sequence of text and video frames in a multimodal fashion. In this paper, we test LLMs' ability to correctly label changes of in-game engagement in 80 minutes of annotated videogame footage from 20 first-person shooter games of the GameVibe corpus. We run over 4,800 experiments to investigate the impact of LLM architecture, model size, input modality, prompting strategy, and ground truth processing method on engagement prediction. Our findings suggest that while LLMs rightfully claim human-like performance across multiple domains and able to outperform traditional machine learning baselines, they generally fall behind continuous experience annotations provided by humans. We examine some of the underlying causes for a fluctuating performance across games, highlight the cases where LLMs exceed expectations, and draw a roadmap for the further exploration of automated emotion labelling via LLMs.
>
---
#### [replaced 070] Large Vision Models Can Solve Mental Rotation Problems
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.15271v2](https://arxiv.org/pdf/2509.15271v2)**

> **作者:** Sebastian Ray Mason; Anders Gjølbye; Phillip Chavarria Højbjerg; Lenka Tětková; Lars Kai Hansen
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Mental rotation is a key test of spatial reasoning in humans and has been central to understanding how perception supports cognition. Despite the success of modern vision transformers, it is still unclear how well these models develop similar abilities. In this work, we present a systematic evaluation of ViT, CLIP, DINOv2, and DINOv3 across a range of mental-rotation tasks, from simple block structures similar to those used by Shepard and Metzler to study human cognition, to more complex block figures, three types of text, and photo-realistic objects. By probing model representations layer by layer, we examine where and how these networks succeed. We find that i) self-supervised ViTs capture geometric structure better than supervised ViTs; ii) intermediate layers perform better than final layers; iii) task difficulty increases with rotation complexity and occlusion, mirroring human reaction times and suggesting similar constraints in embedding space representations.
>
---
#### [replaced 071] Visual Localization via Semantic Structures in Autonomous Photovoltaic Power Plant Inspection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自主导航任务，解决光伏电站巡检中无人机精确定位问题。通过整合模块检测与导航，实现精准定位，并评估不同分割方法的性能。**

- **链接: [https://arxiv.org/pdf/2501.14587v2](https://arxiv.org/pdf/2501.14587v2)**

> **作者:** Viktor Kozák; Karel Košnar; Jan Chudoba; Miroslav Kulich; Libor Přeučil
>
> **备注:** 50 pages, 23 figures. Submitted for review to Array
>
> **摘要:** Inspection systems utilizing unmanned aerial vehicles (UAVs) equipped with thermal cameras are increasingly popular for the maintenance of photovoltaic (PV) power plants. However, automation of the inspection task is a challenging problem as it requires precise navigation to capture images from optimal distances and viewing angles. This paper presents a novel localization pipeline that directly integrates PV module detection with UAV navigation, allowing precise positioning during inspection. The detections are used to identify the power plant structures in the image. These are associated with the power plant model and used to infer the UAV position relative to the inspected PV installation. We define visually recognizable anchor points for the initial association and use object tracking to discern global associations. Additionally, we present three different methods for visual segmentation of PV modules and evaluate their performance in relation to the proposed localization pipeline. The presented methods were verified and evaluated using custom aerial inspection data sets, demonstrating their robustness and applicability for real-time navigation. Additionally, we evaluate the influence of the power plant model precision on the localization methods.
>
---
#### [replaced 072] MOTION: ML-Assisted On-Device Low-Latency Motion Recognition
- **分类: cs.CV; cs.AI; cs.HC**

- **链接: [https://arxiv.org/pdf/2512.00008v2](https://arxiv.org/pdf/2512.00008v2)**

> **作者:** Veeramani Pugazhenthi; Wei-Hsiang Chu; Junwei Lu; Jadyn N. Miyahira; Mahdi Eslamimehr; Pratik Satam; Rozhin Yasaei; Soheil Salehi
>
> **摘要:** The use of tiny devices capable of low-latency gesture recognition is gaining momentum in everyday human-computer interaction and especially in medical monitoring fields. Embedded solutions such as fall detection, rehabilitation tracking, and patient supervision require fast and efficient tracking of movements while avoiding unwanted false alarms. This study presents an efficient solution on how to build very efficient motion-based models only using triaxial accelerometer sensors. We explore the capability of the AutoML pipelines to extract the most important features from the data segments. This approach also involves training multiple lightweight machine learning algorithms using the extracted features. We use WeBe Band, a multi-sensor wearable device that is equipped with a powerful enough MCU to effectively perform gesture recognition entirely on the device. Of the models explored, we found that the neural network provided the best balance between accuracy, latency, and memory use. Our results also demonstrate that reliable real-time gesture recognition can be achieved in WeBe Band, with great potential for real-time medical monitoring solutions that require a secure and fast response time.
>
---
#### [replaced 073] Scale-Equivariant Imaging: Self-Supervised Learning for Image Super-Resolution and Deblurring
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2312.11232v4](https://arxiv.org/pdf/2312.11232v4)**

> **作者:** Jérémy Scanvic; Mike Davies; Patrice Abry; Julián Tachella
>
> **摘要:** Self-supervised methods have recently proved to be nearly as effective as supervised ones in various imaging inverse problems, paving the way for learning-based approaches in scientific and medical imaging applications where ground truth data is hard or expensive to obtain. These methods critically rely on invariance to translations and/or rotations of the image distribution to learn from incomplete measurement data alone. However, existing approaches fail to obtain competitive performances in the problems of image super-resolution and deblurring, which play a key role in most imaging systems. In this work, we show that invariance to roto-translations is insufficient to learn from measurements that only contain low-frequency information. Instead, we propose scale-equivariant imaging, a new self-supervised approach that leverages the fact that many image distributions are approximately scale-invariant, enabling the recovery of high-frequency information lost in the measurement process. We demonstrate throughout a series of experiments on real datasets that the proposed method outperforms other self-supervised approaches, and obtains performances on par with fully supervised learning.
>
---
#### [replaced 074] Bridging Weakly-Supervised Learning and VLM Distillation: Noisy Partial Label Learning for Efficient Downstream Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.03229v3](https://arxiv.org/pdf/2506.03229v3)**

> **作者:** Qian-Wei Wang; Yaguang Song; Shu-Tao Xia
>
> **摘要:** In the context of noisy partial label learning (NPLL), each training sample is associated with a set of candidate labels annotated by multiple noisy annotators. With the emergence of high-performance pre-trained vision-language models (VLMs) such as CLIP, LLaVA, and GPT-4V, leveraging these models to replace time-consuming manual annotation and enable annotation-free training has become a promising research direction. This paper studies learning from noisy partial labels generated by pre-trained VLMs and proposes a collaborative consistency regularization (Co-Reg) framework. Unlike symmetric noise commonly assumed in traditional noisy label learning, VLM-generated noise is instance-dependent and reflects the intrinsic biases of pre-trained models, posing greater challenges. To address this issue, we jointly train two neural networks to perform collaborative label purification via a co-pseudo-labeling mechanism, while enforcing consistency regularization in both label and feature representation spaces. In addition, multiple anti-overfitting strategies are introduced, including alternating optimization of contrastive representations and pseudo-labels, as well as maintaining class prototypes in a shared feature space. The proposed method can further incorporate few-shot manually annotated labels for performance enhancement. Extensive experiments under various settings demonstrate the effectiveness of our approach and highlight the potential of integrating weakly supervised learning into the knowledge distillation of pre-trained models.
>
---
#### [replaced 075] OmniLens: Towards Universal Lens Aberration Correction via LensLib-to-Specific Domain Adaptation
- **分类: physics.optics; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2409.05809v3](https://arxiv.org/pdf/2409.05809v3)**

> **作者:** Qi Jiang; Yao Gao; Shaohua Gao; Zhonghua Yi; Xiaolong Qian; Hao Shi; Kailun Yang; Lei Sun; Kaiwei Wang; Jian Bai
>
> **备注:** Accepted to Optics & Laser Technology (JOLT). The code and data will be available at https://github.com/zju-jiangqi/OmniLens
>
> **摘要:** Emerging universal Computational Aberration Correction (CAC) paradigms provide an inspiring solution to light-weight and high-quality imaging with a universal model trained on a lens library (LensLib) to address arbitrary lens optical aberrations blindly. However, the limited coverage of existing LensLibs leads to poor generalization of the trained models to unseen lenses, whose fine-tuning pipeline is also confined to the lens-descriptions-known case. In this work, we introduce OmniLens, a flexible solution to universal CAC via (i) establishing a convincing LensLib with comprehensive coverage for pre-training a robust base model, and (ii) adapting the model to any specific lens designs with unknown lens descriptions via fast LensLib-to-specific domain adaptation. To achieve these, an Evolution-based Automatic Optical Design (EAOD) pipeline is proposed to generate a rich variety of lens samples with realistic aberration behaviors. Then, we design an unsupervised regularization term for efficient domain adaptation on a few easily accessible real-captured images based on the statistical observation of dark channel priors in degradation induced by lens aberrations. Extensive experiments demonstrate that the LensLib generated by EAOD effectively develops a universal CAC model with strong generalization capabilities, which can also improve the non-blind lens-specific methods by 0.35~1.81dB in PSNR. Additionally, the proposed domain adaptation method significantly improves the base model, especially in severe aberration cases (at most 2.59dB in PSNR). The code and data will be available at https://github.com/zju-jiangqi/OmniLens.
>
---
#### [replaced 076] Entropy Guided Dynamic Patch Segmentation for Time Series Transformers
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.26157v2](https://arxiv.org/pdf/2509.26157v2)**

> **作者:** Sachith Abeywickrama; Emadeldeen Eldele; Min Wu; Xiaoli Li; Chau Yuen
>
> **备注:** Preprint. Under Review
>
> **摘要:** Patch-based transformers have emerged as efficient and improved long-horizon modeling architectures for time series modeling. Yet, existing approaches rely on temporally-agnostic patch construction, where arbitrary starting positions and fixed lengths fracture temporal coherence by splitting natural transitions across boundaries. This naive segmentation often disrupts short-term dependencies and weakens representation learning. We propose a novel Entropy-Guided Dynamic Patch Encoder (EntroPE), as a temporally informed framework that dynamically detects transition points via conditional entropy and dynamically places patch boundaries. This preserves temporal structure while retaining the computational benefits of patching. EntroPE consists of two key modules, namely an Entropy-based Dynamic Patcher (EDP) that applies information-theoretic criteria to locate natural temporal shifts and determine patch boundaries, and an Adaptive Patch Encoder (APE) that employs pooling and cross-attention to capture intra-patch dependencies and produce fixed-size latent representations. Extensive experiments on long-term forecasting, classification, and anomaly detection demonstrate that the proposed method improves both accuracy and efficiency, establishing entropy-guided dynamic patching as a promising new paradigm for time series modeling. Code is available at https://github.com/Sachithx/EntroPE.
>
---
#### [replaced 077] Enhancing Semantic Segmentation with Continual Self-Supervised Pre-training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17816v2](https://arxiv.org/pdf/2509.17816v2)**

> **作者:** Brown Ebouky; Ajad Chhatkuli; Cristiano Malossi; Christoph Studer; Roy Assaf; Andrea Bartezzaghi
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** Self-supervised learning (SSL) has emerged as a central paradigm for training foundation models by leveraging large-scale unlabeled datasets, often producing representations with strong generalization capabilities. These models are typically pre-trained on general-purpose datasets such as ImageNet and subsequently adapted to various downstream tasks through finetuning. While prior work has investigated parameter-efficient adaptation methods like adapters, LoRA, and prompt tuning, primarily targeting downstream finetuning, extending the SSL pre-training itself in a continual manner to new domains under limited data remains largely underexplored, especially for downstream dense prediction tasks like semantic segmentation. In this work, we address the challenge of adapting vision foundation models to low-data target domains through continual self-supervised pre-training, specifically targeting downstream semantic segmentation. We propose GLARE (Global Local and Regional Enforcement), a novel continual self-supervised pre-training task designed to enhance downstream semantic segmentation performance. GLARE introduces patch-level augmentations to encourage local consistency and incorporates a regional consistency constraint that leverages spatial semantics in the data. For efficient continual pre-training, we initialize Vision Transformers (ViTs) with weights from existing SSL models and update only lightweight adapter modules specifically UniAdapter - while keeping the rest of the backbone frozen. Experiments across multiple semantic segmentation benchmarks on different domains demonstrate that GLARE consistently improves downstream performance with minimal computational and parameter overhead.
>
---
#### [replaced 078] BIR-Adapter: A parameter-efficient diffusion adapter for blind image restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.06904v2](https://arxiv.org/pdf/2509.06904v2)**

> **作者:** Cem Eteke; Alexander Griessel; Wolfgang Kellerer; Eckehard Steinbach
>
> **摘要:** We introduce the BIR-Adapter, a parameter-efficient diffusion adapter for blind image restoration. Diffusion-based restoration methods have demonstrated promising performance in addressing this fundamental problem in computer vision, typically relying on auxiliary feature extractors or extensive fine-tuning of pre-trained models. Motivated by the observation that large-scale pretrained diffusion models can retain informative representations under common image degradations, BIR-Adapter introduces a parameter-efficient, plug-and-play attention mechanism that substantially reduces the number of trained parameters. To further improve reliability, we propose a sampling guidance mechanism that mitigates hallucinations during the restoration process. Experiments on synthetic and real-world degradations demonstrate that BIR-Adapter achieves competitive, and in several settings superior, performance compared to state-of-the-art methods while requiring up to 36x fewer trained parameters. Moreover, the adapter-based design enables seamless integration into existing models. We validate this generality by extending a super-resolution-only diffusion model to handle additional unknown degradations, highlighting the adaptability of our approach for broader image restoration tasks.
>
---
#### [replaced 079] Temporally-Similar Structure-Aware Spatiotemporal Fusion of Satellite Images
- **分类: eess.SP; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11259v2](https://arxiv.org/pdf/2508.11259v2)**

> **作者:** Ryosuke Isono; Shunsuke Ono
>
> **备注:** Submitted to IEEE Transactions on Geoscience and Remote Sensing. arXiv admin note: text overlap with arXiv:2308.00500
>
> **摘要:** This paper proposes a spatiotemporal (ST) fusion framework robust against diverse noise for satellite images, named Temporally-Similar Structure-Aware ST fusion (TSSTF). ST fusion is a promising approach to address the trade-off between the spatial and temporal resolution of satellite images. In real-world scenarios, observed satellite images are severely degraded by noise due to measurement equipment and environmental conditions. Consequently, some recent studies have focused on enhancing the robustness of ST fusion methods against noise. However, existing noise-robust ST fusion approaches often fail to capture fine spatial structure, leading to oversmoothing and artifacts. To address this issue, TSSTF introduces two key mechanisms: Temporally-Guided Total Variation (TGTV) and Temporally-Guided Edge Constraint (TGEC). TGTV is a weighted total variation-based regularization that promotes spatial piecewise smoothness while preserving structural details, guided by a reference high spatial resolution image acquired on a nearby date. TGEC enforces consistency in edge locations between two temporally adjacent images, while allowing for spectral variations. We formulate the ST fusion task as a constrained optimization problem incorporating TGTV and TGEC, and develop an efficient algorithm based on a preconditioned primal-dual splitting method. Experimental results demonstrate that TSSTF performs comparably to state-of-the-art methods under noise-free conditions and outperforms them under noisy conditions.
>
---
