# 计算机视觉 cs.CV

- **最新发布 149 篇**

- **更新 80 篇**

## 最新发布

#### [new 001] Control-DINO: Feature Space Conditioning for Controllable Image-to-Video Diffusion
- **分类: cs.CV**

- **简介: 该论文提出Control-DINO，用于图像到视频扩散模型的可控生成。解决如何利用自监督学习特征进行视频风格迁移和3D生成的问题，通过解耦外观与其他特征提升控制能力。**

- **链接: [https://arxiv.org/pdf/2604.01761](https://arxiv.org/pdf/2604.01761)**

> **作者:** Edoardo A. Dominici; Thomas Deixelberger; Konstantinos Vardis; Markus Steinberger
>
> **备注:** project page this https URL
>
> **摘要:** Video models have recently been applied with success to problems in content generation, novel view synthesis, and, more broadly, world simulation. Many applications in generation and transfer rely on conditioning these models, typically through perceptual, geometric, or simple semantic signals, fundamentally using them as generative renderers. At the same time, high-dimensional features obtained from large-scale self-supervised learning on images or point clouds are increasingly used as a general-purpose interface for vision models. The connection between the two has been explored for subject specific editing, aligning and training video diffusion models, but not in the role of a more general conditioning signal for pretrained video diffusion models. Features obtained through self-supervised learning like DINO, contain a lot of entangled information about style, lighting and semantics of the scene. This makes them great at reconstruction tasks but limits their generative capabilities. In this paper, we show how we can use the features for tasks such as video domain transfer and video-from-3D generation. We introduce a lightweight architecture and training strategy that decouples appearance from other features that we wish to preserve, enabling robust control for appearance changes such as stylization and relighting. Furthermore, we show that low spatial resolution can be compensated by higher feature dimensionality, improving controllability in generative rendering from explicit spatial representations.
>
---
#### [new 002] MTLSI-Net: A Linear Semantic Interaction Network for Parameter-Efficient Multi-Task Dense Prediction
- **分类: cs.CV**

- **简介: 该论文提出MTLSI-Net，解决多任务密集预测中的跨任务交互问题，通过线性注意力机制实现高效参数处理。**

- **链接: [https://arxiv.org/pdf/2604.01995](https://arxiv.org/pdf/2604.01995)**

> **作者:** Chen Liu; Hengyu Man; Xiaopeng Fan; Debin Zhao
>
> **备注:** accepted by ICME 2026, to be published
>
> **摘要:** Multi-task dense prediction aims to perform multiple pixel-level tasks simultaneously. However, capturing global cross-task interactions remains non-trivial due to the quadratic complexity of standard self-attention on high-resolution features. To address this limitation, we propose a Multi-Task Linear Semantic Interaction Network (MTLSI-Net), which facilitates cross-task interaction through linear attention. Specifically, MTLSI-Net incorporates three key components: a Multi-Task Multi-scale Query Linear Fusion Block, which captures cross-task dependencies across multiple scales with linear complexity using a shared global context matrix; a Semantic Token Distiller that compresses redundant features into compact semantic tokens, distilling essential cross-task knowledge; and a Cross-Window Integrated attention Block that injects global semantics into local features via a dual-branch architecture, preserving both global consistency and spatial precision. These components collectively enable the network to capture comprehensive cross-task interactions at linear complexity with reduced parameters. Extensive experiments on NYUDv2 and PASCAL-Context demonstrate that MTLSI-Net achieves state-of-the-art performance, validating its effectiveness and efficiency in multi-task learning.
>
---
#### [new 003] HieraVid: Hierarchical Token Pruning for Fast Video Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频理解任务，解决视频大语言模型计算负担重的问题。提出HieraVid框架，通过分层剪枝减少冗余，提升效率。**

- **链接: [https://arxiv.org/pdf/2604.01881](https://arxiv.org/pdf/2604.01881)**

> **作者:** Yansong Guo; Chaoyang Zhu; Jiayi Ji; Jianghang Lin; Liujuan Cao
>
> **摘要:** Video Large Language Models (VideoLLMs) have demonstrated impressive capabilities in video understanding, yet the massive number of input video tokens incurs a significant computational burden for deployment. Existing methods mainly prune video tokens at input level while neglecting the inherent information structure embedded in videos and large language models (LLMs). To address this, we propose HieraVid, a hierarchical pruning framework that progressively and dynamically reduces visual redundancy. Based on two observations that videos possess the segment-frame structure and LLMs internally propagate multi-modal information unidirectionally, we decompose pruning into three levels: 1) segment-level, where video tokens are first temporally segmented and spatially merged; 2) frame-level, where similar frames within the same segment are jointly pruned to preserve diversity; 3) layer-level, redundancy gradually shrinks as LLM layer increases w/o compromising performance. We conduct extensive experiments on four widely used video understanding benchmarks to comprehensively evaluate the effectiveness of HieraVid. Remarkably, with only 30% of tokens retained, HieraVid achieves new state-of-the-art performance, while maintaining over 98% and 99% of the performance of LLaVA-Video-7B and LLaVA-OneVision-7B, respectively.
>
---
#### [new 004] From Understanding to Erasing: Towards Complete and Stable Video Object Removal
- **分类: cs.CV**

- **简介: 该论文属于视频目标移除任务，旨在消除目标物体并保持时空一致性。针对物体引发的副作用问题，提出外部知识蒸馏和内部上下文注意力机制，提升移除效果与整体连贯性。**

- **链接: [https://arxiv.org/pdf/2604.01693](https://arxiv.org/pdf/2604.01693)**

> **作者:** Dingming Liu; Wenjing Wang; Chen Li; Jing Lyu
>
> **摘要:** Video object removal aims to eliminate target objects from videos while plausibly completing missing regions and preserving spatio-temporal consistency. Although diffusion models have recently advanced this task, it remains challenging to remove object-induced side effects (e.g., shadows, reflections, and illumination changes) without compromising overall coherence. This limitation stems from the insufficient physical and semantic understanding of the target object and its interactions with the scene. In this paper, we propose to introduce understanding into erasing from two complementary perspectives. Externally, we introduce a distillation scheme that transfers the relationships between objects and their induced effects from vision foundation models to video diffusion models. Internally, we propose a framewise context cross-attention mechanism that grounds each denoising block in informative, unmasked context surrounding the target region. External and internal guidance jointly enable our model to understand the target object, its induced effects, and the global background context, resulting in clear and coherent object removal. Extensive experiments demonstrate our state-of-the-art performance, and we establish the first real-world benchmark for video object removal to facilitate future research and community progress. Our code, data, and models are available at: this https URL.
>
---
#### [new 005] Beyond Referring Expressions: Scenario Comprehension Visual Grounding
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，解决传统基准难以反映真实场景理解的问题。提出RSC基准和ScenGround方法，提升模型在复杂场景下的推理能力。**

- **链接: [https://arxiv.org/pdf/2604.02323](https://arxiv.org/pdf/2604.02323)**

> **作者:** Ruozhen He; Nisarg A. Shah; Qihua Dong; Zilin Xiao; Jaywon Koo; Vicente Ordonez
>
> **备注:** 20 pages, 18 figures, Project Page: this https URL
>
> **摘要:** Existing visual grounding benchmarks primarily evaluate alignment between image regions and literal referring expressions, where models can often succeed by matching a prominent named category. We explore a complementary and more challenging setting of scenario-based visual grounding, where the target must be inferred from roles, intentions, and relational context rather than explicit naming. We introduce Referring Scenario Comprehension (RSC), a benchmark designed for this setting. The queries in this benchmark are paragraph-length texts that describe object roles, user goals, and contextual cues, including deliberate references to distractor objects that often require deep understanding to resolve. Each instance is annotated with interpretable difficulty tags for uniqueness, clutter, size, overlap, and position which expose distinct failure modes and support fine-grained analysis. RSC contains approximately 31k training examples, 4k in-domain test examples, and a 3k out-of-distribution split with unseen object categories. We further propose ScenGround, a curriculum reasoning method serving as a reference point for this setting, combining supervised warm-starting with difficulty-aware reinforcement learning. Experiments show that scenario-based queries expose systematic failures in current models that standard benchmarks do not reveal, and that curriculum training improves performance on challenging slices and transfers to standard benchmarks.
>
---
#### [new 006] Are VLMs Lost Between Sky and Space? LinkS$^2$Bench for UAV-Satellite Dynamic Cross-View Spatial Intelligence
- **分类: cs.CV**

- **简介: 该论文属于跨视角空间智能任务，旨在解决UAV与卫星图像间动态对齐问题。构建了LinkS$^2$Bench基准，包含17.9k问答对，评估VLMs的跨视图推理能力。**

- **链接: [https://arxiv.org/pdf/2604.02020](https://arxiv.org/pdf/2604.02020)**

> **作者:** Dian Liu; Jie Feng; Di Li; Yuhui Zheng; Guanbin Li; Weisheng Dong; Guangming Shi
>
> **摘要:** Synergistic spatial intelligence between UAVs and satellites is indispensable for emergency response and security operations, as it uniquely integrates macro-scale global coverage with dynamic, real-time local perception. However, the capacity of Vision-Language Models (VLMs) to master this complex interplay remains largely unexplored. This gap persists primarily because existing benchmarks are confined to isolated Unmanned Aerial Vehicle (UAV) videos or static satellite imagery, failing to evaluate the dynamic local-to-global spatial mapping essential for comprehensive cross-view reasoning. To bridge this gap, we introduce LinkS$^2$Bench, the first comprehensive benchmark designed to evaluate VLMs' wide-area, dynamic cross-view spatial intelligence. LinkS$^2$Bench links 1,022 minutes of dynamic UAV footage with high-resolution satellite imagery covering over 200 km$^2$. Through an LMM-assisted pipeline and rigorous human annotation, we constructed 17.9k high-quality question-answer pairs comprising 12 fine-grained tasks across four dimensions: perception, localization, relation, and reasoning. Evaluations of 18 representative VLMs reveal a substantial gap compared to human baselines, identifying accurate cross-view dynamic alignment as the critical bottleneck. To alleviate this, we design a Cross-View Alignment Adapter, demonstrating that explicit alignment significantly improves model performance. Furthermore, fine-tuning experiments underscore the potential of LinkS$^2$Bench in advancing VLM adaptation for complex spatial reasoning.
>
---
#### [new 007] A Self supervised learning framework for imbalanced medical imaging datasets
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决数据稀缺和类别不平衡问题。提出AMIMV方法增强SSL效果，并在多个数据集上验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.01947](https://arxiv.org/pdf/2604.01947)**

> **作者:** Yash Kumar Sharma; Charan Ramtej Kodi; Vineet Padmanabhan
>
> **摘要:** Two problems often plague medical imaging analysis: 1) Non-availability of large quantities of labeled training data, and 2) Dealing with imbalanced data, i.e., abundant data are available for frequent classes, whereas data are highly limited for the rare class. Self supervised learning (SSL) methods have been proposed to deal with the first problem to a certain extent, but the issue of investigating the robustness of SSL to imbalanced data has rarely been addressed in the domain of medical image classification. In this work, we make the following contributions: 1) The MIMV method proposed by us in an earlier work is extended with a new augmentation strategy to construct asymmetric multi-image, multi-view (AMIMV) pairs to address both data scarcity and dataset imbalance in medical image classification. 2) We carry out a data analysis to evaluate the robustness of AMIMV under varying degrees of class imbalance in medical imaging . 3) We evaluate eight representative SSL methods in 11 medical imaging datasets (MedMNIST) under long-tailed distributions and limited supervision. Our experimental results on the MedMNIST dataset show an improvement of 4.25% on retinaMNIST, 1.88% on tissueMNIST, and 3.1% on DermaMNIST.
>
---
#### [new 008] ProDiG: Progressive Diffusion-Guided Gaussian Splatting for Aerial to Ground Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出ProDiG，解决从航空影像重建地面视图和3D模型的问题。通过渐进式扩散引导的高斯点云优化，提升几何一致性与视觉质量。**

- **链接: [https://arxiv.org/pdf/2604.02003](https://arxiv.org/pdf/2604.02003)**

> **作者:** Sirshapan Mitra; Yogesh S. Rawat
>
> **摘要:** Generating ground-level views and coherent 3D site models from aerial-only imagery is challenging due to extreme viewpoint changes, missing intermediate observations, and large scale variations. Existing methods either refine renderings post-hoc, often producing geometrically inconsistent results, or rely on multi-altitude ground-truth, which is rarely available. Gaussian Splatting and diffusion-based refinements improve fidelity under small variations but fail under wide aerial-to-ground gaps. To address these limitations, we introduce ProDiG (Progressive Altitude Gaussian Splatting), a diffusion-guided framework that progressively transforms aerial 3D representations toward ground-level fidelity. ProDiG synthesizes intermediate-altitude views and refines the Gaussian representation at each stage using a geometry-aware causal attention module that injects epipolar structure into reference-view diffusion. A distance-adaptive Gaussian module dynamically adjusts Gaussian scale and opacity based on camera distance, ensuring stable reconstruction across large viewpoint gaps. Together, these components enable progressive, geometrically grounded refinement without requiring additional ground-truth viewpoints. Extensive experiments on synthetic and real-world datasets demonstrate that ProDiG produces visually realistic ground-level renderings and coherent 3D geometry, significantly outperforming existing approaches in terms of visual quality, geometric consistency, and robustness to extreme viewpoint changes.
>
---
#### [new 009] CXR-LT 2026 Challenge: Projection-Aware Multi-Label and Zero-Shot Chest X-Ray Classification
- **分类: cs.CV**

- **简介: 该论文属于胸部X光图像的多标签与零样本分类任务，旨在解决已知病变的多标签分类和未知病变的零样本识别问题。通过集成投影特定模型和改进的双分支架构提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.02185](https://arxiv.org/pdf/2604.02185)**

> **作者:** Juno Cho; Dohui Kim; Mingeon Kim; Hyunseo Jang; Chang Sun Lee; Jong Chul Ye
>
> **备注:** 5 pages, 3 figures. Accepted to the IEEE ISBI 2026 CXR-LT Challenge
>
> **摘要:** This challenge tackles multi-label classification for known chest X-ray (CXR) lesions and zero-shot classification for unseen ones. To handle diverse CXR projections, we integrate projection-specific models via a classification network into a unified framework. For zero-shot classification (Task 2), we extend CheXzero with a novel dual-branch architecture that combines contrastive learning, Asymmetric Loss (ASL), and LLM-generated descriptive prompts. This effectively mitigates severe long-tail imbalances and maximizes zero-shot generalization. Additionally, strong data and test-time augmentations (TTA) ensure robustness across both tasks.
>
---
#### [new 010] Prototype-Based Low Altitude UAV Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于低空无人机图像语义分割任务，解决尺度变化大、边界复杂和计算资源有限的问题。提出PBSeg框架，结合原型注意力和多尺度特征提取，提升分割精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.01550](https://arxiv.org/pdf/2604.01550)**

> **作者:** Da Zhang; Gao Junyu; Zhao Zhiyuan
>
> **备注:** Accepted to ICME 2026
>
> **摘要:** Semantic segmentation of low-altitude UAV imagery presents unique challenges due to extreme scale variations, complex object boundaries, and limited computational resources on edge devices. Existing transformer-based segmentation methods achieve remarkable performance but incur high computational overhead, while lightweight approaches struggle to capture fine-grained details in high-resolution aerial scenes. To address these limitations, we propose PBSeg, an efficient prototype-based segmentation framework tailored for UAV applications. PBSeg introduces a novel prototype-based cross-attention (PBCA) that exploits feature redundancy to reduce computational complexity while maintaining segmentation quality. The framework incorporates an efficient multi-scale feature extraction module that combines deformable convolutions (DConv) with context-aware modulation (CAM) to capture both local details and global semantics. Experiments on two challenging UAV datasets demonstrate the effectiveness of the proposed approach. PBSeg achieves 71.86\% mIoU on UAVid and 80.92\% mIoU on UDD6, establishing competitive performance while maintaining computational efficiency. Code is available at this https URL.
>
---
#### [new 011] Language-Pretraining-Induced Bias: A Strong Foundation for General Vision Tasks
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文研究语言与视觉模态间的跨模态适应问题，旨在解决语言预训练模型在视觉任务中的应用难题。通过引入随机标签桥接训练，实现参数对齐，验证了部分微调的有效性。**

- **链接: [https://arxiv.org/pdf/2604.01833](https://arxiv.org/pdf/2604.01833)**

> **作者:** Yaxin Luo; Zhiqiang Shen
>
> **摘要:** The ratio of outlier parameters in language pre-training models and vision pre-training models differs significantly, making cross-modality (language and vision) inherently more challenging than cross-domain adaptation. As a result, many prior studies have focused on cross-domain transfer rather than attempting to bridge language and vision modalities, assuming that language pre-trained models are unsuitable for downstream visual tasks due to disparate parameter spaces. Contrary to this assumption, we show that adding a bridge training stage as a modality adaptation learner can effectively align Large Language Model (LLM) parameters with vision tasks. Specifically, we propose a simple yet powerful solution random label bridge training that requires no manual labeling and helps LLM parameters adapt to vision foundation tasks. Moreover, our findings reveal that partial bridge training is often advantageous, as certain layers in LLMs exhibit strong foundational properties that remain beneficial even without fine-tuning for visual tasks. This surprising discovery opens up new avenues for leveraging language pre-trained parameters directly within vision models and highlights the potential of partial bridge training as a practical pathway to cross-modality adaptation.
>
---
#### [new 012] Tex3D: Objects as Attack Surfaces via Adversarial 3D Textures for Vision-Language-Action Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于机器人视觉-语言-动作系统安全研究，旨在解决VLA模型对物理3D对抗纹理攻击的脆弱性问题。提出Tex3D框架，实现3D对抗纹理的端到端优化，有效降低模型性能。**

- **链接: [https://arxiv.org/pdf/2604.01618](https://arxiv.org/pdf/2604.01618)**

> **作者:** Jiawei Chen; Simin Huang; Jiawei Du; Shuaihang Chen; Yu Tian; Mingjie Wei; Chao Yu; Zhaoxia Yin
>
> **摘要:** Vision-language-action (VLA) models have shown strong performance in robotic manipulation, yet their robustness to physically realizable adversarial attacks remains underexplored. Existing studies reveal vulnerabilities through language perturbations and 2D visual attacks, but these attack surfaces are either less representative of real deployment or limited in physical realism. In contrast, adversarial 3D textures pose a more physically plausible and damaging threat, as they are naturally attached to manipulated objects and are easier to deploy in physical environments. Bringing adversarial 3D textures to VLA systems is nevertheless nontrivial. A central obstacle is that standard 3D simulators do not provide a differentiable optimization path from the VLA objective function back to object appearance, making it difficult to optimize through an end-to-end manner. To address this, we introduce Foreground-Background Decoupling (FBD), which enables differentiable texture optimization through dual-renderer alignment while preserving the original simulation environment. To further ensure that the attack remains effective across long-horizon and diverse viewpoints in the physical world, we propose Trajectory-Aware Adversarial Optimization (TAAO), which prioritizes behaviorally critical frames and stabilizes optimization with a vertex-based parameterization. Built on these designs, we present Tex3D, the first framework for end-to-end optimization of 3D adversarial textures directly within the VLA simulation environment. Experiments in both simulation and real-robot settings show that Tex3D significantly degrades VLA performance across multiple manipulation tasks, achieving task failure rates of up to 96.7\%. Our empirical results expose critical vulnerabilities of VLA systems to physically grounded 3D adversarial attacks and highlight the need for robustness-aware training.
>
---
#### [new 013] Learning Spatial Structure from Pre-Beamforming Per-Antenna Range-Doppler Radar Data via Visibility-Aware Cross-Modal Supervision
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究如何从雷达原始数据中直接学习空间结构，解决传统需先进行波束成形的问题。通过端到端方法，无需角度域构建即可恢复几何信息。**

- **链接: [https://arxiv.org/pdf/2604.01921](https://arxiv.org/pdf/2604.01921)**

> **作者:** George Sebastian; Philipp Berthold; Bianca Forkel; Leon Pohl; Mirko Maehlisch
>
> **摘要:** Automotive radar perception pipelines commonly construct angle-domain representations via beamforming before applying learning-based models. This work instead investigates a representational question: can meaningful spatial structure be learned directly from pre-beamforming per-antenna range-Doppler (RD) measurements? Experiments are conducted on a 6-TX x 8-RX (48 virtual antennas) commodity automotive radar employing an A/B chirp-sequence frequency-modulated continuous-wave (CS-FMCW) transmit scheme, in which the effective transmit aperture varies between chirps (single-TX vs. multi-TX), enabling controlled analysis of chirp-dependent transmit configurations. We operate on pre-beamforming per-antenna RD tensors using a dual-chirp shared-weight encoder trained in an end-to-end, fully data-driven manner, and evaluate spatial recoverability using bird's-eye-view (BEV) occupancy as a geometric probe rather than a performance-driven objective. Supervision is visibility-aware and cross-modal, derived from LiDAR with explicit modeling of the radar field-of-view and occlusion-aware LiDAR observability via ray-based visibility. Through chirp ablations (A-only, B-only, A+B), range-band analysis, and physics-aligned baselines, we assess how transmit configurations affect geometric recoverability. The results indicate that spatial structure can be learned directly from pre-beamforming per-antenna RD tensors without explicit angle-domain construction or hand-crafted signal-processing stages.
>
---
#### [new 014] LatentUM: Unleashing the Potential of Interleaved Cross-Modal Reasoning via a Latent-Space Unified Model
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出LatentUM，解决统一模型中视觉理解与生成的分离问题，通过共享语义潜在空间实现跨模态推理与生成。**

- **链接: [https://arxiv.org/pdf/2604.02097](https://arxiv.org/pdf/2604.02097)**

> **作者:** Jiachun Jin; Zetong Zhou; Xiao Yang; Hao Zhang; Pengfei Liu; Jun Zhu; Zhijie Deng
>
> **摘要:** Unified models (UMs) hold promise for their ability to understand and generate content across heterogeneous modalities. Compared to merely generating visual content, the use of UMs for interleaved cross-modal reasoning is more promising and valuable, e.g., for solving understanding problems that require dense visual thinking, improving visual generation through self-reflection, or modeling visual dynamics of the physical world guided by stepwise action interventions. However, existing UMs necessitate pixel decoding as a bridge due to their disjoint visual representations for understanding and generation, which is both ineffective and inefficient. In this paper, we introduce LatentUM, a novel unified model that represents all modalities within a shared semantic latent space, eliminating the need for pixel-space mediation between visual understanding and generation. This design naturally enables flexible interleaved cross-modal reasoning and generation. Beyond improved computational efficiency, the shared representation substantially alleviates codec bias and strengthens cross-modal alignment, allowing LatentUM to achieve state-of-the-art performance on the Visual Spatial Planning benchmark, push the limits of visual generation through self-reflection, and support world modeling by predicting future visual states within the shared semantic latent space.
>
---
#### [new 015] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding
- **分类: cs.CV**

- **简介: 该论文属于医学图像-文本理解任务，旨在解决现有模型难以适配超声数据的问题。构建了大规模数据集和知识框架，提出Ultrasound-CLIP模型提升超声图像与文本的对齐效果。**

- **链接: [https://arxiv.org/pdf/2604.01749](https://arxiv.org/pdf/2604.01749)**

> **作者:** Jiayun Jin; Haolong Chai; Xueying Huang; Xiaoqing Guo; Zengwei Zheng; Zhan Zhou; Junmei Wang; Xinyu Wang; Jie Liu; Binbin Zhou
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Ultrasound imaging is widely used in clinical diagnostics due to its real-time capability and radiation-free nature. However, existing vision-language pre-training models, such as CLIP, are primarily designed for other modalities, and are difficult to directly apply to ultrasound data, which exhibit heterogeneous anatomical structures and diverse diagnostic attributes. To bridge this gap, we construct US-365K, a large-scale ultrasound image-text dataset containing 365k paired samples across 52 anatomical categories. We establish Ultrasonographic Diagnostic Taxonomy (UDT) containing two hierarchical knowledge frameworks. Ultrasonographic Hierarchical Anatomical Taxonomy standardizes anatomical organization, and Ultrasonographic Diagnostic Attribute Framework formalizes nine diagnostic dimensions, including body system, organ, diagnosis, shape, margins, echogenicity, internal characteristics, posterior acoustic phenomena, and vascularity. Building upon these foundations, we propose Ultrasound-CLIP, a semantic-aware contrastive learning framework that introduces semantic soft labels and semantic loss to refine sample discrimination. Moreover, we construct a heterogeneous graph modality derived from UDAF's textual representations, enabling structured reasoning over lesion-attribute relations. Extensive experiments with patient-level data splitting demonstrate that our approach achieves state-of-the-art performance on classification and retrieval benchmarks, while also delivering strong generalization to zero-shot, linear probing, and fine-tuning tasks.
>
---
#### [new 016] Perceptual misalignment of texture representations in convolutional neural networks
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉领域，研究CNN纹理表示与人类感知的对齐问题。通过分析不同CNN的特征相关性，发现其纹理表示与人类感知无显著关联，揭示了现有模型在纹理建模上的局限性。**

- **链接: [https://arxiv.org/pdf/2604.01341](https://arxiv.org/pdf/2604.01341)**

> **作者:** Ludovica de Paolis; Fabio Anselmi; Alessio Ansuini; Eugenio Piasini
>
> **摘要:** Mathematical modeling of visual textures traces back to Julesz's intuition that texture perception in humans is based on local correlations between image features. An influential approach for texture analysis and generation generalizes this notion to linear correlations between the nonlinear features computed by convolutional neural networks (CNNs), compiled into Gram matrices. Given that CNNs are often used as models for the visual system, it is natural to ask whether such "texture representations" spontaneously align with the textures' perceptual content, and in particular whether those CNNs that are regarded as better models for the visual system also possess more human-like texture representations. Here we compare the perceptual content captured by feature correlations computed for a diverse pool of CNNs, and we compare it to the models' perceptual alignment with the mammalian visual system as measured by Brain-Score. Surprisingly, we find that there is no connection between conventional measures of CNN quality as a model of the visual system and its alignment with human texture perception. We conclude that texture perception involves mechanisms that are distinct from those that are commonly modeled using approaches based on CNNs trained on object recognition, possibly depending on the integration of contextual information.
>
---
#### [new 017] Light-ResKAN: A Parameter-Sharing Lightweight KAN with Gram Polynomials for Efficient SAR Image Recognition
- **分类: cs.CV**

- **简介: 该论文属于SAR图像识别任务，旨在解决轻量化模型在精度与计算效率间的平衡问题。提出Light-ResKAN，结合KAN和Gram多项式，实现高效特征提取。**

- **链接: [https://arxiv.org/pdf/2604.01903](https://arxiv.org/pdf/2604.01903)**

> **作者:** Pan Yi; Weijie Li; Xiaodong Chen; Jiehua Zhang; Li Liu; Yongxiang Liu
>
> **备注:** 16 pages, 8 figures, accepted by JSTARS
>
> **摘要:** Synthetic Aperture Radar (SAR) image recognition is vital for disaster monitoring, military reconnaissance, and ocean observation. However, large SAR image sizes hinder deep learning deployment on resource-constrained edge devices, and existing lightweight models struggle to balance high-precision feature extraction with low computational requirements. The emerging Kolmogorov-Arnold Network (KAN) enhances fitting by replacing fixed activations with learnable ones, reducing parameters and computation. Inspired by KAN, we propose Light-ResKAN to achieve a better balance between precision and efficiency. First, Light-ResKAN modifies ResNet by replacing convolutions with KAN convolutions, enabling adaptive feature extraction for SAR images. Second, we use Gram Polynomials as activations, which are well-suited for SAR data to capture complex non-linear relationships. Third, we employ a parameter-sharing strategy: each kernel shares parameters per channel, preserving unique features while reducing parameters and FLOPs. Our model achieves 99.09%, 93.01%, and 97.26% accuracy on MSTAR, FUSAR-Ship, and SAR-ACD datasets, respectively. Experiments on MSTAR resized to $1024 \times 1024$ show that compared to VGG16, our model reduces FLOPs by $82.90 \times$ and parameters by $163.78 \times$. This work establishes an efficient solution for edge SAR image recognition.
>
---
#### [new 018] CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦于3D affordance grounding任务，解决在功能相似物体中根据意图精准识别目标对象的问题。提出CompassAD基准和CompassNet框架，提升多物体场景下的意图驱动定位效果。**

- **链接: [https://arxiv.org/pdf/2604.02060](https://arxiv.org/pdf/2604.02060)**

> **作者:** Jingliang Li; Jindou Jia; Tuo An; Chuhao Zhou; Xiangyu Chen; Shilin Shan; Boyu Ma; Bofan Lyu; Gen Li; Jianfei Yang
>
> **备注:** Code available at: this http URL
>
> **摘要:** When told to "cut the apple," a robot must choose the knife over nearby scissors, despite both objects affording the same cutting function. In real-world scenes, multiple objects may share identical affordances, yet only one is appropriate under the given task context. We call such cases confusing pairs. However, existing 3D affordance methods largely sidestep this challenge by evaluating isolated single objects, often with explicit category names provided in the query. We formalize Multi-Object Affordance Grounding under Intent-Driven Instructions, a new 3D affordance setting that requires predicting a per-point affordance mask on the correct object within a cluttered multi-object point cloud, conditioned on implicit natural language intent. To study this problem, we construct CompassAD, the first benchmark centered on implicit intent in confusable multi-object scenes. It comprises 30 confusing object pairs spanning 16 affordance types, 6,422 scenes, and 88K+ query-answer pairs. Furthermore, we propose CompassNet, a framework that incorporates two dedicated modules tailored to this task. Instance-bounded Cross Injection (ICI) constrains language-geometry alignment within object boundaries to prevent cross-object semantic leakage. Bi-level Contrastive Refinement (BCR) enforces discrimination at both geometric-group and point levels, sharpening distinctions between target and confusable surfaces. Extensive experiments demonstrate state-of-the-art results on both seen and unseen queries, and deployment on a robotic manipulator confirms effective transfer to real-world grasping in confusing multi-object scenes.
>
---
#### [new 019] Attention at Rest Stays at Rest: Breaking Visual Inertia for Cognitive Hallucination Mitigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态大语言模型任务，旨在解决认知幻觉问题。通过分析视觉注意力惯性，提出IVE方法促进动态关系推理，提升模型的组合理解能力。**

- **链接: [https://arxiv.org/pdf/2604.01989](https://arxiv.org/pdf/2604.01989)**

> **作者:** Boyang Gong; Yu Zheng; Fanye Kong; Jie Zhou; Jiwen Lu
>
> **摘要:** Like a body at rest that stays at rest, we find that visual attention in multimodal large language models (MLLMs) exhibits pronounced inertia, remaining largely static once settled during early decoding steps and failing to support the compositional understanding required for cognitive inference. While existing hallucination mitigation methods mainly target perceptual hallucinations concerning object existence or attributes, they remain inadequate for such cognitive hallucinations that require inter-object relational deduction. Through token-wise attention analysis, we identify this visual inertia as a key factor: attention to semantically critical regions remains persistently focused and fails to dynamically support relational inference. We thereby propose a training-free Inertia-aware Visual Excitation (IVE) method that breaks this inertial pattern by modeling cognitive inference as the dynamic responsiveness of visual attention. Specifically, IVE selects visual tokens that are dynamically emerging relative to historical attention trends while distinguishing tokens exhibiting inertial behavior. To further facilitate compositional inference, IVE introduces an inertia-aware penalty that discourages over-concentration and limits the persistence of attention within localized regions. Extensive experiments show that IVE is effective across various base MLLMs and multiple hallucination benchmarks, particularly for cognitive hallucinations.
>
---
#### [new 020] DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data
- **分类: cs.CV**

- **简介: 该论文提出DynaVid，解决视频生成中动态运动和精细控制的问题。通过合成运动数据训练，提升视频 realism 和可控性。**

- **链接: [https://arxiv.org/pdf/2604.01666](https://arxiv.org/pdf/2604.01666)**

> **作者:** Wonjoon Jin; Jiyun Won; Janghyeok Han; Qi Dai; Chong Luo; Seung-Hwan Baek; Sunghyun Cho
>
> **备注:** Accepted to CVPR 2026. Website: this https URL
>
> **摘要:** Despite recent progress, video diffusion models still struggle to synthesize realistic videos involving highly dynamic motions or requiring fine-grained motion controllability. A central limitation lies in the scarcity of such examples in commonly used training datasets. To address this, we introduce DynaVid, a video synthesis framework that leverages synthetic motion data in training, which is represented as optical flow and rendered using computer graphics pipelines. This approach offers two key advantages. First, synthetic motion offers diverse motion patterns and precise control signals that are difficult to obtain from real data. Second, unlike rendered videos with artificial appearances, rendered optical flow encodes only motion and is decoupled from appearance, thereby preventing models from reproducing the unnatural look of synthetic videos. Building on this idea, DynaVid adopts a two-stage generation framework: a motion generator first synthesizes motion, and then a motion-guided video generator produces video frames conditioned on that motion. This decoupled formulation enables the model to learn dynamic motion patterns from synthetic data while preserving visual realism from real-world videos. We validate our framework on two challenging scenarios, vigorous human motion generation and extreme camera motion control, where existing datasets are particularly limited. Extensive experiments demonstrate that DynaVid improves the realism and controllability in dynamic motion generation and camera motion control.
>
---
#### [new 021] GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决3D高斯泼溅的高内存消耗问题。通过优化高斯点的空间分布和动态剪枝，提升渲染质量并减少内存占用。**

- **链接: [https://arxiv.org/pdf/2604.01884](https://arxiv.org/pdf/2604.01884)**

> **作者:** Xianben Yang; Tao Wang; Yuxuan Li; Yi Jin; Haibin Ling
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated breakthrough performance in novel view synthesis and real-time rendering. Nevertheless, its practicality is constrained by the high memory cost due to a huge number of Gaussian points. Many pruning-based 3DGS variants have been proposed for memory saving, but often compromise spatial consistency and may lead to rendering artifacts. To address this issue, we propose graph-based spatial distribution optimization for compact 3D Gaussian Splatting (GS\textasciicircum2), which enhances reconstruction quality by optimizing the spatial distribution of Gaussian points. Specifically, we introduce an evidence lower bound (ELBO)-based adaptive densification strategy that automatically controls the densification process. In addition, an opacity-aware progressive pruning strategy is proposed to further reduce memory consumption by dynamically removing low-opacity Gaussian points. Furthermore, we propose a graph-based feature encoding module to adjust the spatial distribution via feature-guided point shifting. Extensive experiments validate that GS\textasciicircum2 achieves a compact Gaussian representation while delivering superior rendering quality. Compared with 3DGS, it achieves higher PSNR with only about 12.5\% Gaussian points. Furthermore, it outperforms all compared baselines in both rendering quality and memory efficiency.
>
---
#### [new 022] Can Video Diffusion Models Predict Past Frames? Bidirectional Cycle Consistency for Reversible Interpolation
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频帧插值任务，旨在解决运动漂移和时序不一致问题。提出双向框架，通过循环一致性约束提升生成质量与可逆性。**

- **链接: [https://arxiv.org/pdf/2604.01700](https://arxiv.org/pdf/2604.01700)**

> **作者:** Lingyu Liu; Yaxiong Wang; Li Zhu; Zhedong Zheng
>
> **摘要:** Video frame interpolation aims to synthesize realistic intermediate frames between given endpoints while adhering to specific motion semantics. While recent generative models have improved visual fidelity, they predominantly operate in a unidirectional manner, lacking mechanisms to self-verify temporal consistency. This often leads to motion drift, directional ambiguity, and boundary misalignment, especially in long-range sequences. Inspired by the principle of temporal cycle-consistency in self-supervised learning, we propose a novel bidirectional framework that enforces symmetry between forward and backward generation trajectories. Our approach introduces learnable directional tokens to explicitly condition a shared backbone on temporal orientation, enabling the model to jointly optimize forward synthesis and backward reconstruction within a single unified architecture. This cycle-consistent supervision acts as a powerful regularizer, ensuring that generated motion paths are logically reversible. Furthermore, we employ a curriculum learning strategy that progressively trains the model from short to long sequences, stabilizing dynamics across varying durations. Crucially, our cyclic constraints are applied only during training; inference requires a single forward pass, maintaining the high efficiency of the base model. Extensive experiments show that our method achieves state-of-the-art performance in imaging quality, motion smoothness, and dynamic control on both 37-frame and 73-frame tasks, outperforming strong baselines while incurring no additional computational overhead.
>
---
#### [new 023] EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors
- **分类: cs.CV**

- **简介: 该论文提出EventHub，用于训练通用事件立体网络，无需主动传感器标注。通过标准彩色图像生成代理标注，提升模型泛化能力。任务为事件立体视觉，解决标注成本高问题。**

- **链接: [https://arxiv.org/pdf/2604.02331](https://arxiv.org/pdf/2604.02331)**

> **作者:** Luca Bartolomei; Fabio Tosi; Matteo Poggi; Stefano Mattoccia; Guillermo Gallego
>
> **备注:** CVPR 2026. Project Page: this https URL Code: this https URL
>
> **摘要:** We propose EventHub, a novel framework for training deep-event stereo networks without ground truth annotations from costly active sensors, relying instead on standard color images. From these images, we derive either proxy annotations and proxy events through state-of-the-art novel view synthesis techniques, or simply proxy annotations when images are already paired with event data. Using the training set generated by our data factory, we repurpose state-of-the-art stereo models from RGB literature to process event data, obtaining new event stereo models with unprecedented generalization capabilities. Experiments on widely used event stereo datasets support the effectiveness of EventHub and show how the same data distillation mechanism can improve the accuracy of RGB stereo foundation models in challenging conditions such as nighttime scenes.
>
---
#### [new 024] Night Eyes: A Reproducible Framework for Constellation-Based Corneal Reflection Matching
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于眼动追踪任务，解决多光点检测与匹配的可重复性问题。提出基于星座的2D几何方法，实现稳定匹配与清晰评估。**

- **链接: [https://arxiv.org/pdf/2604.01909](https://arxiv.org/pdf/2604.01909)**

> **作者:** Virmarie Maquiling; Yasmeen Abdrabou; Enkelejda Kasneci
>
> **备注:** 6 pages, 3 figures, 2 algorithms, ETRA26
>
> **摘要:** Corneal reflection (glint) detection plays an important role in pupil-corneal reflection (P-CR) eye tracking, but in practice it is often handled as heuristics embedded within larger systems, making reproducibility difficult across hardware setups. We introduce a 2D geometry-driven, constellation-based pipeline for mulit-glint detection and matching, focusing on reproducibility and clear evaluation. Inspired by lost-in-space star identification, we treat glints as structured constellations rather than independent blobs. We propose a Similarity-Layout Alignment (SLA) procedure which adapts constellation matching to the specific constraints of multi-LED eye tracking. The framework brings together controlled over-detection, adaptive candidate fallback, appearance-aware scoring, and optional semantic layout priors while keeping detection and correspondence explicitly separated. Evaluated on a public multi-LED dataset, the system provides stable identity-preserving correspondence under noisy conditions. We release code, presets, and evaluation scripts to enable transparent replication, comparison, and dataset annotation.
>
---
#### [new 025] Ranking-Guided Semi-Supervised Domain Adaptation for Severity Classification
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析中的领域自适应任务，旨在解决严重程度分类中因类别边界模糊带来的域迁移问题。提出一种基于排名的半监督域适应方法，通过排序和分布对齐实现跨域样本对齐。**

- **链接: [https://arxiv.org/pdf/2604.01834](https://arxiv.org/pdf/2604.01834)**

> **作者:** Shota Harada; Ryoma Bise; Kiyohito Tanaka; Seiichi Uchida
>
> **摘要:** Semi-supervised domain adaptation leverages a few labeled and many unlabeled target samples, making it promising for addressing domain shifts in medical image analysis. However, existing methods struggle with severity classification due to unclear class boundaries. Severity classification involves naturally ordered class labels, complicating adaptation. We propose a novel method that aligns source and target domains using rank scores learned via ranking with class order. Specifically, Cross-Domain Ranking ranks sample pairs across domains, while Continuous Distribution Alignment aligns rank score distributions. Experiments on ulcerative colitis and diabetic retinopathy classification validate the effectiveness of our approach, demonstrating successful alignment of class-specific rank score distributions.
>
---
#### [new 026] Satellite-Free Training for Drone-View Geo-Localization
- **分类: cs.CV**

- **简介: 该论文属于无人机视角地理定位任务，解决GPS缺失环境下定位问题。通过无卫星训练框架，利用多视角无人机图像重建三维场景并生成兼容卫星图的特征表示。**

- **链接: [https://arxiv.org/pdf/2604.01581](https://arxiv.org/pdf/2604.01581)**

> **作者:** Tao Liu; Yingzhi Zhang; Kan Ren; Xiaoqi Zhao
>
> **摘要:** Drone-view geo-localization (DVGL) aims to determine the location of drones in GPS-denied environments by retrieving the corresponding geotagged satellite tile from a reference gallery given UAV observations of a location. In many existing formulations, these observations are represented by a single oblique UAV image. In contrast, our satellite-free setting is designed for multi-view UAV sequences, which are used to construct a geometry-normalized UAV-side location representation before cross-view retrieval. Existing approaches rely on satellite imagery during training, either through paired supervision or unsupervised alignment, which limits practical deployment when satellite data are unavailable or restricted. In this paper, we propose a satellite-free training (SFT) framework that converts drone imagery into cross-view compatible representations through three main stages: drone-side 3D scene reconstruction, geometry-based pseudo-orthophoto generation, and satellite-free feature aggregation for retrieval. Specifically, we first reconstruct dense 3D scenes from multi-view drone images using 3D Gaussian splatting and project the reconstructed geometry into pseudo-orthophotos via PCA-guided orthographic projection. This rendering stage operates directly on reconstructed scene geometry without requiring camera parameters at rendering time. Next, we refine these orthophotos with lightweight geometry-guided inpainting to obtain texture-complete drone-side views. Finally, we extract DINOv3 patch features from the generated orthophotos, learn a Fisher vector aggregation model solely from drone data, and reuse it at test time to encode satellite tiles for cross-view retrieval. Experimental results on University-1652 and SUES-200 show that our SFT framework substantially outperforms satellite-free generalization baselines and narrows the gap to methods trained with satellite imagery.
>
---
#### [new 027] AffordTissue: Dense Affordance Prediction for Tool-Action Specific Tissue Interaction
- **分类: cs.CV; cs.AI; cs.RO; eess.IV**

- **简介: 该论文提出AffordTissue，解决手术中工具与组织交互区域预测问题，通过多模态框架实现工具-动作特定的密集 affordance 预测，提升手术自动化安全性。**

- **链接: [https://arxiv.org/pdf/2604.01371](https://arxiv.org/pdf/2604.01371)**

> **作者:** Aiza Maksutova; Lalithkumar Seenivasan; Hao Ding; Jiru Xu; Chenhao Yu; Chenyan Jing; Yiqing Shen; Mathias Unberath
>
> **摘要:** Surgical action automation has progressed rapidly toward achieving surgeon-like dexterous control, driven primarily by advances in learning from demonstration and vision-language-action models. While these have demonstrated success in table-top experiments, translating them to clinical deployment remains challenging: current methods offer limited predictability on where instruments will interact on tissue surfaces and lack explicit conditioning inputs to enforce tool-action-specific safe interaction regions. Addressing this gap, we introduce AffordTissue, a multimodal framework for predicting tool-action specific tissue affordance regions as dense heatmaps during cholecystectomy. Our approach combines a temporal vision encoder capturing tool motion and tissue dynamics across multiple viewpoints, language conditioning enabling generalization across diverse instrument-action pairs, and a DiT-style decoder for dense affordance prediction. We establish the first tissue affordance benchmark by curating and annotating 15,638 video clips across 103 cholecystectomy procedures, covering six unique tool-action pairs involving four instruments (hook, grasper, scissors, clipper) and their associated tasks: dissection, grasping, clipping, and cutting. Experiments demonstrate substantial improvement over vision-language model baselines (20.6 px ASSD vs. 60.2 px for Molmo-VLM), showing that our task-specific architecture outperforms large-scale foundation models for dense surgical affordance prediction. By predicting tool-action specific tissue affordance regions, AffordTissue provides explicit spatial reasoning for safe surgical automation, potentially unlocking explicit policy guidance toward appropriate tissue regions and early safe stop when instruments deviate outside predicted safe zones.
>
---
#### [new 028] SHOE: Semantic HOI Open-Vocabulary Evaluation Metric
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人-物交互检测任务，解决传统评估指标无法处理语义相似但词面不同的预测问题。提出SHOE框架，通过语义相似性评估提升开放词汇检测的准确性。**

- **链接: [https://arxiv.org/pdf/2604.01586](https://arxiv.org/pdf/2604.01586)**

> **作者:** Maja Noack; Qinqian Lei; Taipeng Tian; Bihan Dong; Robby T. Tan; Yixin Chen; John Young; Saijun Zhang; Bo Wang
>
> **备注:** Accepted to GRAIL-V Workshop at CVPR 2026
>
> **摘要:** Open-vocabulary human-object interaction (HOI) detection is a step towards building scalable systems that generalize to unseen interactions in real-world scenarios and support grounded multimodal systems that reason about human-object relationships. However, standard evaluation metrics, such as mean Average Precision (mAP), treat HOI classes as discrete categorical labels and fail to credit semantically valid but lexically different predictions (e.g., "lean on couch" vs. "sit on couch"), limiting their applicability for evaluating open-vocabulary predictions that go beyond any predefined set of HOI labels. We introduce SHOE (Semantic HOI Open-Vocabulary Evaluation), a new evaluation framework that incorporates semantic similarity between predicted and ground-truth HOI labels. SHOE decomposes each HOI prediction into its verb and object components, estimates their semantic similarity using the average of multiple large language models (LLMs), and combines them into a similarity score to evaluate alignment beyond exact string match. This enables a flexible and scalable evaluation of both existing HOI detection methods and open-ended generative models using standard benchmarks such as HICO-DET. Experimental results show that SHOE scores align more closely with human judgments than existing metrics, including LLM-based and embedding-based baselines, achieving an agreement of 85.73% with the average human ratings. Our work underscores the need for semantically grounded HOI evaluation that better mirrors human understanding of interactions. We will release our evaluation metric to the public to facilitate future research.
>
---
#### [new 029] Omni123: Exploring 3D Native Foundation Models with Limited 3D Data by Unifying Text to 2D and 3D Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Omni123，解决3D生成中数据不足的问题，通过统一文本到2D和3D生成，提升3D表示。**

- **链接: [https://arxiv.org/pdf/2604.02289](https://arxiv.org/pdf/2604.02289)**

> **作者:** Chongjie Ye; Cheng Cao; Chuanyu Pan; Yiming Hao; Yihao Zhi; Yuanming Hu; Xiaoguang Han
>
> **摘要:** Recent multimodal large language models have achieved strong performance in unified text and image understanding and generation, yet extending such native capability to 3D remains challenging due to limited data. Compared to abundant 2D imagery, high-quality 3D assets are scarce, making 3D synthesis under-constrained. Existing methods often rely on indirect pipelines that edit in 2D and lift results into 3D via optimization, sacrificing geometric consistency. We present Omni123, a 3D-native foundation model that unifies text-to-2D and text-to-3D generation within a single autoregressive framework. Our key insight is that cross-modal consistency between images and 3D can serve as an implicit structural constraint. By representing text, images, and 3D as discrete tokens in a shared sequence space, the model leverages abundant 2D data as a geometric prior to improve 3D representations. We introduce an interleaved X-to-X training paradigm that coordinates diverse cross-modal tasks over heterogeneous paired datasets without requiring fully aligned text-image-3D triplets. By traversing semantic-visual-geometric cycles (e.g., text to image to 3D to image) within autoregressive sequences, the model jointly enforces semantic alignment, appearance fidelity, and multi-view geometric consistency. Experiments show that Omni123 significantly improves text-guided 3D generation and editing, demonstrating a scalable path toward multimodal 3D world models.
>
---
#### [new 030] VideoZeroBench: Probing the Limits of Video MLLMs with Spatio-Temporal Evidence Verification
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频多模态问答任务，旨在解决现有评估体系无法准确验证模型时空证据识别的问题。提出VideoZeroBench基准，通过严格验证时空证据提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2604.01569](https://arxiv.org/pdf/2604.01569)**

> **作者:** Jiahao Meng; Tan Yue; Qi Xu; Haochen Wang; Zhongwei Ren; Weisong Liu; Yuhao Wang; Renrui Zhang; Yunhai Tong; Haodong Duan
>
> **摘要:** Recent video multimodal large language models achieve impressive results across various benchmarks. However, current evaluations suffer from two critical limitations: (1) inflated scores can mask deficiencies in fine-grained visual understanding and reasoning, and (2) answer correctness is often measured without verifying whether models identify the precise spatio-temporal evidence supporting their predictions. To address this, we present VideoZeroBench, a hierarchical benchmark designed for challenging long-video question answering that rigorously verifies spatio-temporal evidence. It comprises 500 manually annotated questions across 13 domains, paired with temporal intervals and spatial bounding boxes as evidence. To disentangle answering generation, temporal grounding, and spatial grounding, we introduce a five-level evaluation protocol that progressively tightens evidence requirements. Experiments show that even Gemini-3-Pro correctly answers fewer than 17% of questions under the standard end-to-end QA setting (Level-3). When grounding constraints are imposed, performance drops sharply: No model exceeds 1% accuracy when both correct answering and accurate spatio-temporal localization are required (Level-5), with most failing to achieve any correct grounded predictions. These results expose a significant gap between surface-level answer correctness and genuine evidence-based reasoning, revealing that grounded video understanding remains a bottleneck for long-video QA. We further analyze performance across minimal evidence spans, atomic abilities, and inference paradigms, providing insights for future research in grounded video reasoning. The benchmark and code will be made publicly available.
>
---
#### [new 031] UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决视觉-语言-动作模型中空间感知与语义推理的冲突。提出UniDriveVLA模型，通过专家解耦提升性能。**

- **链接: [https://arxiv.org/pdf/2604.02190](https://arxiv.org/pdf/2604.02190)**

> **作者:** Yongkang Li; Lijun Zhou; Sixu Yan; Bencheng Liao; Tianyi Yan; Kaixin Xiong; Long Chen; Hongwei Xie; Bing Wang; Guang Chen; Hangjun Ye; Wenyu Liu; Haiyang Sun; Xinggang Wang
>
> **备注:** code has been released at this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged in autonomous driving, with the promise of leveraging rich world knowledge to improve the cognitive capabilities of driving systems. However, adapting such models for driving tasks currently faces a critical dilemma between spatial perception and semantic reasoning. Consequently, existing VLA systems are forced into suboptimal compromises: directly adopting 2D Vision-Language Models yields limited spatial perception, whereas enhancing them with 3D spatial representations often impairs the native reasoning capacity of VLMs. We argue that this dilemma largely stems from the coupled optimization of spatial perception and semantic reasoning within shared model parameters. To overcome this, we propose UniDriveVLA, a Unified Driving Vision-Language-Action model based on Mixture-of-Transformers that addresses the perception-reasoning conflict via expert decoupling. Specifically, it comprises three experts for driving understanding, scene perception, and action planning, which are coordinated through masked joint attention. In addition, we combine a sparse perception paradigm with a three-stage progressive training strategy to improve spatial perception while maintaining semantic reasoning capability. Extensive experiments show that UniDriveVLA achieves state-of-the-art performance in open-loop evaluation on nuScenes and closed-loop evaluation on Bench2Drive. Moreover, it demonstrates strong performance across a broad range of perception, prediction, and understanding tasks, including 3D detection, online mapping, motion forecasting, and driving-oriented VQA, highlighting its broad applicability as a unified model for autonomous driving. Code and model have been released at this https URL
>
---
#### [new 032] Rethinking Representations for Cross-Domain Infrared Small Target Detection: A Generalizable Perspective from the Frequency Domain
- **分类: cs.CV**

- **简介: 该论文属于跨域红外小目标检测任务，旨在解决模型在不同域间性能下降的问题。通过频率域视角重构表示，提出S²CPNet网络提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.01934](https://arxiv.org/pdf/2604.01934)**

> **作者:** Yimin Fu; Songbo Wang; Feiyan Wu; Jialin Lyu; Zhunga Liu; Michael K. Ng
>
> **备注:** The code will be released at this https URL upon acceptance
>
> **摘要:** The accurate target-background separation in infrared small target detection (IRSTD) highly depends on the discriminability of extracted representations. However, most existing methods are confined to domain-consistent settings, while overlooking whether such discriminability can generalize to unseen domains. In practice, distribution shifts between training and testing data are inevitable due to variations in observational conditions and environmental factors. Meanwhile, the intrinsic indistinctiveness of infrared small targets aggravates overfitting to domain-specific patterns. Consequently, the detection performance of models trained on source domains can be severely degraded when deployed in unseen domains. To address this challenge, we propose a spatial-spectral collaborative perception network (S$^2$CPNet) for cross-domain IRSTD. Moving beyond conventional spatial learning pipelines, we rethink IRSTD representations from a frequency perspective and reveal inconsistencies in spectral phase as the primary manifestation of domain discrepancies. Based on this insight, we develop a phase rectification module (PRM) to derive generalizable target awareness. Then, we employ an orthogonal attention mechanism (OAM) in skip connections to preserve positional information while refining informative representations. Moreover, the bias toward domain-specific patterns is further mitigated through selective style recomposition (SSR). Extensive experiments have been conducted on three IRSTD datasets, and the proposed method consistently achieves state-of-the-art performance under diverse cross-domain settings.
>
---
#### [new 033] TOL: Textual Localization with OpenStreetMap
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出TOL任务，解决文本到OSM的定位问题，通过TOLoc框架实现城市环境下的精准定位。**

- **链接: [https://arxiv.org/pdf/2604.01644](https://arxiv.org/pdf/2604.01644)**

> **作者:** Youqi Liao; Shuhao Kang; Jingyu Xu; Olaf Wysocki; Yan Xia; Jianping Li; Zhen Dong; Bisheng Yang; Xieyuanli Chen
>
> **备注:** Tech repo
>
> **摘要:** Natural language provides an intuitive way to express spatial intent in geospatial applications. While existing localization methods often rely on dense point cloud maps or high-resolution imagery, OpenStreetMap (OSM) offers a compact and freely available map representation that encodes rich semantic and structural information, making it well suited for large-scale localization. However, text-to-OSM (T2O) localization remains largely unexplored. In this paper, we formulate the T2O global localization task, which aims to estimate accurate 2 degree-of-freedom (DoF) positions in urban environments from textual scene descriptions without relying on geometric observations or GNSS-based initial location. To support the proposed task, we introduce TOL, a large-scale benchmark spanning multiple continents and diverse urban environments. TOL contains approximately 121K textual queries paired with OSM map tiles and covers about 316 km of road trajectories across Boston, Karlsruhe, and Singapore. We further propose TOLoc, a coarse-to-fine localization framework that explicitly models the semantics of surrounding objects and their directional information. In the coarse stage, direction-aware features are extracted from both textual descriptions and OSM tiles to construct global descriptors, which are used to retrieve candidate locations for the query. In the fine stage, the query text and top-1 retrieved tile are jointly processed, where a dedicated alignment module fuses textual descriptor and local map features to regress the 2-DoF pose. Experimental results demonstrate that TOLoc achieves strong localization performance, outperforming the best existing method by 6.53%, 9.93%, and 8.31% at 5m, 10m, and 25m thresholds, respectively, and shows strong generalization to unseen environments. Dataset, code and models will be publicly available at: this https URL.
>
---
#### [new 034] Nonlinear Methods for Analyzing Pose in Behavioral Research
- **分类: cs.CV**

- **简介: 该论文属于行为分析任务，旨在解决姿态数据高维、噪声和时间复杂性带来的挑战。提出一个通用分析流程，结合预处理、降维和递归分析，以提取运动动态中的模式。**

- **链接: [https://arxiv.org/pdf/2604.01453](https://arxiv.org/pdf/2604.01453)**

> **作者:** Carter Sale; Margaret C. Macpherson; Gaurav Patil; Kelly Miles; Rachel W. Kallen; Sebastian Wallot; Michael J. Richardson
>
> **备注:** 40 pages, 13 figures
>
> **摘要:** Advances in markerless pose estimation have made it possible to capture detailed human movement in naturalistic settings using standard video, enabling new forms of behavioral analysis at scale. However, the high dimensionality, noise, and temporal complexity of pose data raise significant challenges for extracting meaningful patterns of coordination and behavioral change. This paper presents a general-purpose analysis pipeline for human pose data, designed to support both linear and nonlinear characterizations of movement across diverse experimental contexts. The pipeline combines principled preprocessing, dimensionality reduction, and recurrence-based time series analysis to quantify the temporal structure of movement dynamics. To illustrate the pipeline's flexibility, we present three case studies spanning facial and full-body movement, 2D and 3D data, and individual versus multi-agent behavior. Together, these examples demonstrate how the same analytic workflow can be adapted to extract theoretically meaningful insights from complex pose time series.
>
---
#### [new 035] ActionParty: Multi-Subject Action Binding in Generative Video Games
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出ActionParty，解决多主体动作绑定问题，属于生成式视频游戏领域。通过引入主体状态标记，实现多主体动作控制与场景渲染分离。**

- **链接: [https://arxiv.org/pdf/2604.02330](https://arxiv.org/pdf/2604.02330)**

> **作者:** Alexander Pondaven; Ziyi Wu; Igor Gilitschenski; Philip Torr; Sergey Tulyakov; Fabio Pizzati; Aliaksandr Siarohin
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent advances in video diffusion have enabled the development of "world models" capable of simulating interactive environments. However, these models are largely restricted to single-agent settings, failing to control multiple agents simultaneously in a scene. In this work, we tackle a fundamental issue of action binding in existing video diffusion models, which struggle to associate specific actions with their corresponding subjects. For this purpose, we propose ActionParty, an action controllable multi-subject world model for generative video games. It introduces subject state tokens, i.e. latent variables that persistently capture the state of each subject in the scene. By jointly modeling state tokens and video latents with a spatial biasing mechanism, we disentangle global video frame rendering from individual action-controlled subject updates. We evaluate ActionParty on the Melting Pot benchmark, demonstrating the first video world model capable of controlling up to seven players simultaneously across 46 diverse environments. Our results show significant improvements in action-following accuracy and identity consistency, while enabling robust autoregressive tracking of subjects through complex interactions.
>
---
#### [new 036] Unifying UAV Cross-View Geo-Localization via 3D Geometric Perception
- **分类: cs.CV**

- **简介: 该论文属于UAV跨视角地理定位任务，解决GNSS拒止环境下图像与卫星图几何差异问题。提出一种基于3D几何感知的统一框架，结合多视角重建与BEV渲染，提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.01747](https://arxiv.org/pdf/2604.01747)**

> **作者:** Haoyuan Li; Wen Yang; Fang Xu; Hong Tan; Haijian Zhang; Shengyang Li; Gui-Song Xia
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Cross-view geo-localization for Unmanned Aerial Vehicles (UAVs) operating in GNSS-denied environments remains challenging due to the severe geometric discrepancy between oblique UAV imagery and orthogonal satellite maps. Most existing methods address this problem through a decoupled pipeline of place retrieval and pose estimation, implicitly treating perspective distortion as appearance noise rather than an explicit geometric transformation. In this work, we propose a geometry-aware UAV geo-localization framework that explicitly models the 3D scene geometry to unify coarse place recognition and fine-grained pose estimation within a single inference pipeline. Our approach reconstructs a local 3D scene from multi-view UAV image sequences using a Visual Geometry Grounded Transformer (VGGT), and renders a virtual Bird's-Eye View (BEV) representation that orthorectifies the UAV perspective to align with satellite imagery. This BEV serves as a geometric intermediary that enables robust cross-view retrieval and provides spatial priors for accurate 3 Degrees of Freedom (3-DoF) pose regression. To efficiently handle multiple location hypotheses, we introduce a Satellite-wise Attention Block that isolates the interaction between each satellite candidate and the reconstructed UAV scene, preventing inter-candidate interference while maintaining linear computational complexity. In addition, we release a recalibrated version of the University-1652 dataset with precise coordinate annotations and spatial overlap analysis, enabling rigorous evaluation of end-to-end localization accuracy. Extensive experiments on the refined University-1652 benchmark and SUES-200 demonstrate that our method significantly outperforms state-of-the-art baselines, achieving robust meter-level localization accuracy and improved generalization in complex urban environments.
>
---
#### [new 037] Large-scale Codec Avatars: The Unreasonable Effectiveness of Large-scale Avatar Pretraining
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D人体建模任务，旨在解决高保真与泛化能力的矛盾。通过大规模预训练和微调，提出LCA模型，实现高效、高质量的全身虚拟人像生成。**

- **链接: [https://arxiv.org/pdf/2604.02320](https://arxiv.org/pdf/2604.02320)**

> **作者:** Junxuan Li; Rawal Khirodkar; Chengan He; Zhongshi Jiang; Giljoo Nam; Lingchen Yang; Jihyun Lee; Egor Zakharov; Zhaoen Su; Rinat Abdrashitov; Yuan Dong; Julieta Martinez; Kai Li; Qingyang Tan; Takaaki Shiratori; Matthew Hu; Peihong Guo; Xuhua Huang; Ariyan Zarei; Marco Pesavento; Yichen Xu; He Wen; Teng Deng; Wyatt Borsos; Anjali Thakrar; Jean-Charles Bazin; Carsten Stoll; Ginés Hidalgo; James Booth; Lucy Wang; Xiaowen Ma; Yu Rong; Sairanjith Thalanki; Chen Cao; Christian Häne; Abhishek Kar; Sofien Bouaziz; Jason Saragih; Yaser Sheikh; Shunsuke Saito
>
> **备注:** Accepted in CVPR2026. Website: this https URL
>
> **摘要:** High-quality 3D avatar modeling faces a critical trade-off between fidelity and generalization. On the one hand, multi-view studio data enables high-fidelity modeling of humans with precise control over expressions and poses, but it struggles to generalize to real-world data due to limited scale and the domain gap between the studio environment and the real world. On the other hand, recent large-scale avatar models trained on millions of in-the-wild samples show promise for generalization across a wide range of identities, yet the resulting avatars are often of low-quality due to inherent 3D ambiguities. To address this, we present Large-Scale Codec Avatars (LCA), a high-fidelity, full-body 3D avatar model that generalizes to world-scale populations in a feedforward manner, enabling efficient inference. Inspired by the success of large language models and vision foundation models, we present, for the first time, a pre/post-training paradigm for 3D avatar modeling at scale: we pretrain on 1M in-the-wild videos to learn broad priors over appearance and geometry, then post-train on high-quality curated data to enhance expressivity and fidelity. LCA generalizes across hair styles, clothing, and demographics while providing precise, fine-grained facial expressions and finger-level articulation control, with strong identity preservation. Notably, we observe emergent generalization to relightability and loose garment support to unconstrained inputs, and zero-shot robustness to stylized imagery, despite the absence of direct supervision.
>
---
#### [new 038] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于人-物交互检测任务，旨在提升对图像中人与物体交互的识别效果。针对现有方法未能充分利用场景上下文的问题，提出InCoM-Net框架，融合视觉语言模型语义与实例特征，增强交互推理能力。**

- **链接: [https://arxiv.org/pdf/2604.02071](https://arxiv.org/pdf/2604.02071)**

> **作者:** Soo Won Seo; KyungChae Lee; Hyungchan Cho; Taein Son; Nam Ik Cho; Jun Won Choi
>
> **备注:** Accepted to CVPR 2026. Code: this https URL
>
> **摘要:** Human-Object Interaction (HOI) detection aims to localize human-object pairs and classify their interactions from a single image, a task that demands strong visual understanding and nuanced contextual reasoning. Recent approaches have leveraged Vision-Language Models (VLMs) to introduce semantic priors, significantly improving HOI detection performance. However, existing methods often fail to fully capitalize on the diverse contextual cues distributed across the entire scene. To overcome these limitations, we propose the Instance-centric Context Mining Network (InCoM-Net)-a novel framework that effectively integrates rich semantic knowledge extracted from VLMs with instance-specific features produced by an object detector. This design enables deeper interaction reasoning by modeling relationships not only within each detected instance but also across instances and their surrounding scene context. InCoM-Net comprises two core components: Instancecentric Context Refinement (ICR), which separately extracts intra-instance, inter-instance, and global contextual cues from VLM-derived features, and Progressive Context Aggregation (ProCA), which iteratively fuses these multicontext features with instance-level detector features to support high-level HOI reasoning. Extensive experiments on the HICO-DET and V-COCO benchmarks show that InCoM-Net achieves state-of-the-art performance, surpassing previous HOI detection methods. Code is available at this https URL.
>
---
#### [new 039] Reflection Generation for Composite Image Using Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于图像合成任务，解决反射生成问题。通过引入先验信息和类型感知设计，构建了首个大规模反射数据集，提升了反射的物理一致性和视觉真实性。**

- **链接: [https://arxiv.org/pdf/2604.02168](https://arxiv.org/pdf/2604.02168)**

> **作者:** Haonan Zhao; Qingyang Liu; Jiaxuan Chen; Li Niu
>
> **摘要:** Image composition involves inserting a foreground object into the background while synthesizing environment-consistent effects such as shadows and reflections. Although shadow generation has been extensively studied, reflection generation remains largely underexplored. In this work, we focus on reflection generation. We inject the prior information of reflection placement and reflection appearance into foundation diffusion model. We also divide reflections into two types and adopt type-aware model design. To support training, we construct the first large-scale object reflection dataset DEROBA. Experiments demonstrate that our method generates reflections that are physically coherent and visually realistic, establishing a new benchmark for reflection generation.
>
---
#### [new 040] Semantic Segmentation of Textured Non-manifold 3D Meshes using Transformers
- **分类: cs.CV**

- **简介: 该论文属于3D网格语义分割任务，解决纹理非流形网格的分割问题。提出一种融合纹理和几何信息的Transformer模型，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.01836](https://arxiv.org/pdf/2604.01836)**

> **作者:** Mohammadreza Heidarianbaei; Max Mehltretter; Franz Rottensteiner
>
> **摘要:** Textured 3D meshes jointly represent geometry, topology, and appearance, yet their irregular structure poses significant challenges for deep-learning-based semantic segmentation. While a few recent methods operate directly on meshes without imposing geometric constraints, they typically overlook the rich textural information also provided by such meshes. We introduce a texture-aware transformer that learns directly from raw pixels associated with each mesh face, coupled with a new hierarchical learning scheme for multi-scale feature aggregation. A texture branch summarizes all face-level pixels into a learnable token, which is fused with geometrical descriptors and processed by a stack of Two-Stage Transformer Blocks (TSTB), which allow for both a local and a global information flow. We evaluate our model on the Semantic Urban Meshes (SUM) benchmark and a newly curated cultural-heritage dataset comprising textured roof tiles with triangle-level annotations for damage types. Our method achieves 81.9\% mF1 and 94.3\% OA on SUM and 49.7\% mF1 and 72.8\% OA on the new dataset, substantially outperforming existing approaches.
>
---
#### [new 041] IndoorCrowd: A Multi-Scene Dataset for Human Detection, Segmentation, and Tracking with an Automated Annotation Pipeline
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出 IndoorCrowd 数据集，用于解决室内人群检测、分割和跟踪任务，通过多场景数据和自动化标注方法提升研究效果。**

- **链接: [https://arxiv.org/pdf/2604.02032](https://arxiv.org/pdf/2604.02032)**

> **作者:** Sebastian-Ion Nae; Radu Moldoveanu; Alexandra Stefania Ghita; Adina Magda Florea
>
> **备注:** Accepted at Conference on Computer Vision and Pattern Recognition Workshops 2026
>
> **摘要:** Understanding human behaviour in crowded indoor environments is central to surveillance, smart buildings, and human-robot interaction, yet existing datasets rarely capture real-world indoor complexity at scale. We introduce IndoorCrowd, a multi-scene dataset for indoor human detection, instance segmentation, and multi-object tracking, collected across four campus locations (ACS-EC, ACS-EG, IE-Central, R-Central). It comprises $31$ videos ($9{,}913$ frames at $5$fps) with human-verified, per-instance segmentation masks. A $620$-frame control subset benchmarks three foundation-model auto-annotators: SAM3, GroundingSAM, and EfficientGroundingSAM, against human labels using Cohen's $\kappa$, AP, precision, recall, and mask IoU. A further $2{,}552$-frame subset supports multi-object tracking with continuous identity tracks in MOTChallenge format. We establish detection, segmentation, and tracking baselines using YOLOv8n, YOLOv26n, and RT-DETR-L paired with ByteTrack, BoT-SORT, and OC-SORT. Per-scene analysis reveals substantial difficulty variation driven by crowd density, scale, and occlusion: ACS-EC, with $79.3\%$ dense frames and a mean instance scale of $60.8$px, is the most challenging scene. The project page is available at this https URL.
>
---
#### [new 042] Resonance4D: Frequency-Domain Motion Supervision for Preset-Free Physical Parameter Learning in 4D Dynamic Physical Scene Simulation
- **分类: cs.CV**

- **简介: 该论文提出Resonance4D，解决4D动态物理场景模拟中运动监督成本高、参数优化受限的问题。通过双域运动监督和零样本分割，实现高效物理仿真。**

- **链接: [https://arxiv.org/pdf/2604.01994](https://arxiv.org/pdf/2604.01994)**

> **作者:** Changshe Zhang; Jie Feng; Siyu Chen; Guanbin Li; Ronghua Shang; Junpeng Zhang
>
> **摘要:** Physics-driven 4D dynamic simulation from static 3D scenes remains constrained by an overlooked contradiction: reliable motion supervision often relies on online video diffusion or optical-flow pipelines whose computational cost exceeds that of the simulator itself. Existing methods further simplify inverse physical modeling by optimizing only partial material parameters, limiting realism in scenes with complex materials and dynamics. We present Resonance4D, a physics-driven 4D dynamic simulation framework that couples 3D Gaussian Splatting with the Material Point Method through lightweight yet physically expressive supervision. Our key insight is that dynamic consistency can be enforced without dense temporal generation by jointly constraining motion in complementary domains. To this end, we introduce Dual-domain Motion Supervision (DMS), which combines spatial structural consistency for local deformation with frequency-domain spectral consistency for oscillatory and global dynamic patterns, substantially reducing training cost and memory overhead while preserving physically meaningful motion cues. To enable stable full-parameter physical recovery, we further combine zero-shot text-prompted segmentation with simulation-guided initialization to automatically decompose Gaussians into object-part-level regions and support joint optimization of full material parameters. Experiments on both synthetic and real scenes show that Resonance4D achieves strong physical fidelity and motion consistency while reducing peak GPU memory from over 35\,GB to around 20\,GB, enabling high-fidelity physics-driven 4D simulation on a single consumer-grade GPU.
>
---
#### [new 043] Prime Once, then Reprogram Locally: An Efficient Alternative to Black-Box Service Model Adaptation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于模型适应任务，解决封闭式API模型优化效率低的问题。提出AReS方法，通过一次API交互和本地重编程实现高效模型适配。**

- **链接: [https://arxiv.org/pdf/2604.01474](https://arxiv.org/pdf/2604.01474)**

> **作者:** Yunbei Zhang; Chengyi Cai; Feng Liu; Jihun Hamm
>
> **备注:** CVPR 2026
>
> **摘要:** Adapting closed-box service models (i.e., APIs) for target tasks typically relies on reprogramming via Zeroth-Order Optimization (ZOO). However, this standard strategy is known for extensive, costly API calls and often suffers from slow, unstable optimization. Furthermore, we observe that this paradigm faces new challenges with modern APIs (e.g., GPT-4o). These models can be less sensitive to the input perturbations ZOO relies on, thereby hindering performance gains. To address these limitations, we propose an Alternative efficient Reprogramming approach for Service models (AReS). Instead of direct, continuous closed-box optimization, AReS initiates a single-pass interaction with the service API to prime an amenable local pre-trained encoder. This priming stage trains only a lightweight layer on top of the local encoder, making it highly receptive to the subsequent glass-box (white-box) reprogramming stage performed directly on the local model. Consequently, all subsequent adaptation and inference rely solely on this local proxy, eliminating all further API costs. Experiments demonstrate AReS's effectiveness where prior ZOO-based methods struggle: on GPT-4o, AReS achieves a +27.8% gain over the zero-shot baseline, a task where ZOO-based methods provide little to no improvement. Broadly, across ten diverse datasets, AReS outperforms state-of-the-art methods (+2.5% for VLMs, +15.6% for standard VMs) while reducing API calls by over 99.99%. AReS thus provides a robust and practical solution for adapting modern closed-box models.
>
---
#### [new 044] Semantic Richness or Geometric Reasoning? The Fragility of VLM's Visual Invariance
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型研究，探讨其在几何变换下的空间不变性问题。工作揭示了VLMs在简单旋转、缩放等操作下表现脆弱，指出其空间推理能力不足。**

- **链接: [https://arxiv.org/pdf/2604.01848](https://arxiv.org/pdf/2604.01848)**

> **作者:** Jason Qiu; Zachary Meurer; Xavier Thomas; Deepti Ghadiyaram
>
> **摘要:** This work investigates the fundamental fragility of state-of-the-art Vision-Language Models (VLMs) under basic geometric transformations. While modern VLMs excel at semantic tasks such as recognizing objects in canonical orientations and describing complex scenes, they exhibit systematic failures at a more fundamental level: lack of robust spatial invariance and equivariance required to reliably determine object identity under simple rotations, scaling, and identity transformations. We demonstrate this limitation through a systematic evaluation across diverse visual domains, including symbolic sketches, natural photographs, and abstract art. Performance drops sharply as semantic content becomes sparse, and this behavior is observed across architectures, model capacities, and prompting strategies. Overall, our results reveal a systematic gap between semantic understanding and spatial reasoning in current VLMs, highlighting the need for stronger geometric grounding in future multimodal systems.
>
---
#### [new 045] Modular Energy Steering for Safe Text-to-Image Generation with Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成的安全控制任务，旨在解决生成内容安全性问题。提出一种无需训练的推理阶段引导框架，利用预训练模型的语义信息实现安全控制。**

- **链接: [https://arxiv.org/pdf/2604.02265](https://arxiv.org/pdf/2604.02265)**

> **作者:** Yaoteng Tan; Zikui Cai; M. Salman Asif
>
> **摘要:** Controlling the behavior of text-to-image generative models is critical for safe and practical deployment. Existing safety approaches typically rely on model fine-tuning or curated datasets, which can degrade generation quality or limit scalability. We propose an inference-time steering framework that leverages gradient feedback from frozen pretrained foundation models to guide the generation process without modifying the underlying generator. Our key observation is that vision-language foundation models encode rich semantic representations that can be repurposed as off-the-shelf supervisory signals during generation. By injecting such feedback through clean latent estimates at each sampling step, our method formulates safety steering as an energy-based sampling problem. This design enables modular, training-free safety control that is compatible with both diffusion and flow-matching models and can generalize across diverse visual concepts. Experiments demonstrate state-of-the-art robustness against NSFW red-teaming benchmarks and effective multi-target steering, while preserving high generation quality on benign non-targeted prompts. Our framework provides a principled approach for utilizing foundation models as semantic energy estimators, enabling reliable and scalable safety control for text-to-image generation.
>
---
#### [new 046] Test-Time Adaptation for Height Completion via Self-Supervised ViT Features and Monocular Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于高度补全任务，解决DSM不完整问题。通过自监督ViT和单目模型，在测试时进行适应，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2604.02009](https://arxiv.org/pdf/2604.02009)**

> **作者:** Osher Rafaeli; Tal Svoray; Ariel Nahlieli
>
> **摘要:** Accurate digital surface models (DSMs) are essential for many geospatial applications, including urban monitoring, environmental analyses, infrastructure management, and change detection. However, large-scale DSMs frequently contain incomplete or outdated regions due to acquisition limitations, reconstruction artifacts, or changes in the built environment. Traditional height completion approaches primarily rely on spatial interpolation or which assume spatial continuity and therefore fail when objects are missing. Recent learning-based approaches improve reconstruction quality but typically require supervised training on sensor-specific datasets, limiting their generalization across domains and sensing conditions. We propose Prior2DSM, a training-free framework for metric DSM completion that operates entirely at test time by leveraging foundation models. Unlike previous height completion approaches that require task-specific training, the proposed method combines self-supervised Vision Transformer (ViT) features from DINOv3 with monocular depth foundation models to propagate metric information from incomplete height priors through semantic feature-space correspondence. Test-time adaptation (TTA) is performed using parameter-efficient low-rank adaptation (LoRA) together with a lightweight multilayer perceptron (MLP), which predicts spatially varying scale and shift parameters to convert relative depth estimates into metric heights. Experiments demonstrate consistent improvements over interpolation based methods, prior-based rescaling height approaches, and state-of-the-art monocular depth estimation models. Prior2DSM reduces reconstruction error while preserving structural fidelity, achieving up to a 46% reduction in RMSE compared to linear fitting of MDE, and further enables DSM updating and coupled RGB-DSM generation.
>
---
#### [new 047] ProVG: Progressive Visual Grounding via Language Decoupling for Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文提出ProVG，解决遥感图像中基于自然语言的物体定位问题。通过解耦语言表达，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2604.01893](https://arxiv.org/pdf/2604.01893)**

> **作者:** Ke Li; Ting Wang; Di Wang; Yongshan Zhu; Yiming Zhang; Tao Lei; Quan Wang
>
> **摘要:** Remote sensing visual grounding (RSVG) aims to localize objects in remote sensing imagery according to natural language expressions. Previous methods typically rely on sentence-level vision-language alignment, which struggles to exploit fine-grained linguistic cues, such as \textit{spatial relations} and \textit{object attributes}, that are crucial for distinguishing objects with similar characteristics. Importantly, these cues play distinct roles across different grounding stages and should be leveraged accordingly to provide more explicit guidance. In this work, we propose \textbf{ProVG}, a novel RSVG framework that improves localization accuracy by decoupling language expressions into global context, spatial relations, and object attributes. To integrate these linguistic cues, ProVG employs a simple yet effective progressive cross-modal modulator, which dynamically modulates visual attention through a \textit{survey-locate-verify} scheme, enabling coarse-to-fine vision-language alignment. In addition, ProVG incorporates a cross-scale fusion module to mitigate the large-scale variations in remote sensing imagery, along with a language-guided calibration decoder to refine cross-modal alignment during prediction. A unified multi-task head further enables ProVG to support both referring expression comprehension and segmentation tasks. Extensive experiments on two benchmarks, \textit{i.e.}, RRSIS-D and RISBench, demonstrate that ProVG consistently outperforms existing methods, achieving new state-of-the-art performance.
>
---
#### [new 048] FlowSlider: Training-Free Continuous Image Editing via Fidelity-Steering Decomposition
- **分类: cs.CV**

- **简介: 该论文提出FlowSlider，解决无训练的连续图像编辑问题，通过分解更新项实现稳定控制，提升编辑质量。**

- **链接: [https://arxiv.org/pdf/2604.02088](https://arxiv.org/pdf/2604.02088)**

> **作者:** Taichi Endo; Guoqing Hao; Kazuhiko Sumi
>
> **备注:** HuggingFace Space: this https URL
>
> **摘要:** Continuous image editing aims to provide slider-style control of edit strength while preserving source-image fidelity and maintaining a consistent edit direction. Existing learning-based slider methods typically rely on auxiliary modules trained with synthetic or proxy supervision. This introduces additional training overhead and couples slider behavior to the training distribution, which can reduce reliability under distribution shifts in edits or domains. We propose \textit{FlowSlider}, a training-free method for continuous editing in Rectified Flow that requires no post-training. \textit{FlowSlider} decomposes FlowEdit's update into (i) a fidelity term, which acts as a source-conditioned stabilizer that preserves identity and structure, and (ii) a steering term that drives semantic transition toward the target edit. Geometric analysis and empirical measurements show that these terms are approximately orthogonal, enabling stable strength control by scaling only the steering term while keeping the fidelity term unchanged. As a result, \textit{FlowSlider} provides smooth and reliable control without post-training, improving continuous editing quality across diverse tasks.
>
---
#### [new 049] GPA: Learning GUI Process Automation from Demonstrations
- **分类: cs.CV; cs.AI; cs.SE**

- **简介: 该论文提出GPA，解决GUI自动化任务中传统RPA脆弱和视觉语言模型不可靠的问题，通过序列蒙特卡洛定位、准备校准等方法提升鲁棒性与安全性。**

- **链接: [https://arxiv.org/pdf/2604.01676](https://arxiv.org/pdf/2604.01676)**

> **作者:** Zirui Zhao; Jun Hao Liew; Yan Yang; Wenzhuo Yang; Ziyang Luo; Doyen Sahoo; Silvio Savarese; Junnan Li
>
> **摘要:** GUI Process Automation (GPA) is a lightweight but general vision-based Robotic Process Automation (RPA), which enables fast and stable process replay with only a single demo. Addressing the fragility of traditional RPA and the non-deterministic risks of current vision language model-based GUI agents, GPA introduces three core benefits: (1) Robustness via Sequential Monte Carlo-based localization to handle rescaling and detection uncertainty; (2) Deterministic and Reliability safeguarded by readiness calibration; and (3) Privacy through fast, fully local execution. This approach delivers the adaptability, robustness, and security required for enterprise workflows. It can also be used as an MCP/CLI tool by other agents with coding capabilities so that the agent only reasons and orchestrates while GPA handles the GUI execution. We conducted a pilot experiment to compare GPA with Gemini 3 Pro (with CUA tools) and found that GPA achieves higher success rate with 10 times faster execution speed in finishing long-horizon GUI tasks.
>
---
#### [new 050] NEMESIS: Noise-suppressed Efficient MAE with Enhanced Superpatch Integration Strategy
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出NEMESIS，用于3D医学影像的自监督学习任务，解决标注成本高和内存消耗大的问题，通过局部超块设计提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.01612](https://arxiv.org/pdf/2604.01612)**

> **作者:** Kyeonghun Kim; Hyeonseok Jung; Youngung Han; Hyunsu Go; Eunseob Choi; Seongbin Park; Junsu Lim; Jiwon Yang; Sumin Lee; Insung Hwang; Ken Ying-Kai Liao; Nam-Joon Kim
>
> **备注:** 5 pages, 5 figures, 5 tables
>
> **摘要:** Volumetric CT imaging is essential for clinical diagnosis, yet annotating 3D volumes is expensive and time-consuming, motivating self-supervised learning (SSL) from unlabeled data. However, applying SSL to 3D CT remains challenging due to the high memory cost of full-volume transformers and the anisotropic spatial structure of CT data, which is not well captured by conventional masking strategies. We propose NEMESIS, a masked autoencoder (MAE) framework that operates on local 128x128x128 superpatches, enabling memory-efficient training while preserving anatomical detail. NEMESIS introduces three key components: (i) noise-enhanced reconstruction as a pretext task, (ii) Masked Anatomical Transformer Blocks (MATB) that perform dual-masking through parallel plane-wise and axis-wise token removal, and (iii) NEMESIS Tokens (NT) for cross-scale context aggregation. On the BTCV multi-organ classification benchmark, NEMESIS with a frozen backbone and a linear classifier achieves a mean AUROC of 0.9633, surpassing fully fine-tuned SuPreM (0.9493) and VoCo (0.9387). Under a low-label regime with only 10% of available annotations, it retains an AUROC of 0.9075, demonstrating strong label efficiency. Furthermore, the superpatch-based design reduces computational cost to 31.0 GFLOPs per forward pass, compared to 985.8 GFLOPs for the full-volume baseline, providing a scalable and robust foundation for 3D medical imaging.
>
---
#### [new 051] Moiré Video Authentication: A Physical Signature Against AI Video Generation
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于视频真实性验证任务，旨在解决AI生成视频难以辨别的问题。通过利用莫尔效应的物理特性，提取视频中的特征进行验证，区分真实与AI生成视频。**

- **链接: [https://arxiv.org/pdf/2604.01654](https://arxiv.org/pdf/2604.01654)**

> **作者:** Yuan Qing; Kunyu Zheng; Lingxiao Li; Boqing Gong; Chang Xiao
>
> **备注:** 17 pages, 14 figures
>
> **摘要:** Recent advances in video generation have made AI-synthesized content increasingly difficult to distinguish from real footage. We propose a physics-based authentication signature that real cameras produce naturally, but that generative models cannot faithfully reproduce. Our approach exploits the Moiré effect: the interference fringes formed when a camera views a compact two-layer grating structure. We derive the Moiré motion invariant, showing that fringe phase and grating image displacement are linearly coupled by optical geometry, independent of viewing distance and grating structure. A verifier extracts both signals from video and tests their correlation. We validate the invariant on both real-captured and AI-generated videos from multiple state-of-the-art generators, and find that real and AI-generated videos produce significantly different correlation signatures, suggesting a robust means of differentiating them. Our work demonstrates that deterministic optical phenomena can serve as physically grounded, verifiable signatures against AI-generated video.
>
---
#### [new 052] End-to-End Shared Attention Estimation via Group Detection with Feedback Refinement
- **分类: cs.CV**

- **简介: 该论文属于共享注意力估计任务，旨在解决传统方法未检测实际关注群体或假设单一焦点的问题。通过两步流程实现群体检测与注意力估计的联合优化。**

- **链接: [https://arxiv.org/pdf/2604.01714](https://arxiv.org/pdf/2604.01714)**

> **作者:** Chihiro Nakatani; Norimichi Ukita; Jean-Marc Odobez
>
> **备注:** Accepted to CVPR2026 Workshop (GAZE 2026)
>
> **摘要:** This paper proposes an end-to-end shared attention estimation method via group detection. Most previous methods estimate shared attention (SA) without detecting the actual group of people focusing on it, or assume that there is a single SA point in a given image. These issues limit the applicability of SA detection in practice and impact performance. To address them, we propose to simultaneously achieve group detection and shared attention estimation using a two step process: (i) the generation of SA heatmaps relying on individual gaze attention heatmaps and group membership scalars estimated in a group inference; (ii) a refinement of the initial group memberships allowing to account for the initial SA heatmaps, and the final prediction of the SA heatmap. Experiments demonstrate that our method outperforms other methods in group detection and shared attention estimation. Additional analyses validate the effectiveness of the proposed components. Code: this https URL.
>
---
#### [new 053] Robust Embodied Perception in Dynamic Environments via Disentangled Weight Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于 embodied perception 任务，解决动态环境中模型过拟合与遗忘问题。提出无领域标识和样本的增量学习框架，通过解耦表示和权重融合提升模型泛化与适应能力。**

- **链接: [https://arxiv.org/pdf/2604.01669](https://arxiv.org/pdf/2604.01669)**

> **作者:** Juncen Guo; Xiaoguang Zhu; Jingyi Wu; Jingyu Zhang; Jingnan Cai; Zhenghao Niu; Liang Song
>
> **备注:** Accepted by ICME2026
>
> **摘要:** Embodied perception systems face severe challenges of dynamic environment distribution drift when they continuously interact in open physical spaces. However, the existing domain incremental awareness methods often rely on the domain id obtained in advance during the testing phase, which limits their practicability in unknown interaction scenarios. At the same time, the model often overfits to the context-specific perceptual noise, which leads to insufficient generalization ability and catastrophic forgetting. To address these limitations, we propose a domain-id and exemplar-free incremental learning framework for embodied multimedia systems, which aims to achieve robust continuous environment adaptation. This method designs a disentangled representation mechanism to remove non-essential environmental style interference, and guide the model to focus on extracting semantic intrinsic features shared across scenes, thereby eliminating perceptual uncertainty and improving generalization. We further use the weight fusion strategy to dynamically integrate the old and new environment knowledge in the parameter space, so as to ensure that the model adapts to the new distribution without storing historical data and maximally retains the discrimination ability of the old environment. Extensive experiments on multiple standard benchmark datasets show that the proposed method significantly reduces catastrophic forgetting in a completely exemplar-free and domain-id free setting, and its accuracy is better than the existing state-of-the-art methods.
>
---
#### [new 054] FSKD: Monocular Forest Structure Inference via LiDAR-to-RGBI Knowledge Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FSKD框架，通过LiDAR到RGBI的知识蒸馏，解决单目森林结构推断问题，实现高精度CHM、PAI和FHD预测。**

- **链接: [https://arxiv.org/pdf/2604.01766](https://arxiv.org/pdf/2604.01766)**

> **作者:** Taimur Khan; Hannes Feilhauer; Muhammad Jazib Zafar
>
> **备注:** Paper in-review
>
> **摘要:** Very High Resolution (VHR) forest structure data at individual-tree scale is essential for carbon, biodiversity, and ecosystem monitoring. Still, airborne LiDAR remains costly and infrequent despite being the reference for forest structure metrics like Canopy Height Model (CHM), Plant Area Index (PAI), and Foliage Height Diversity (FHD). We propose FSKD: a LiDAR-to-RGB-Infrared (RGBI) knowledge distillation (KD) framework in which a multi-modal teacher fuses RGBI imagery with LiDAR-derived planar metrics and vertical profiles via cross-attention, and an RGBI-only SegFormer student learns to reproduce these outputs. Trained on 384 $km^2$ of forests in Saxony, Germany (20 cm ground sampling distance (GSD)) and evaluated on eight geographically distinct test tiles, the student achieves state-of-the-art (SOTA) zero-shot CHM performance (MedAE 4.17 m, $R^2$=0.51, IoU 0.87), outperforming HRCHM/DAC baselines by 29--46% in MAE (5.81 m vs. 8.14--10.84 m) with stronger correlation coefficients (0.713 vs. 0.166--0.652). Ablations show that multi-modal fusion improves performance by 10--26% over RGBI-only training, and that asymmetric distillation with appropriate model capacity is critical. The method jointly predicts CHM, PAI, and FHD, a multi-metric capability not provided by current monocular CHM estimators, although PAI/FHD transfer remains region-dependent and benefits from local calibration. The framework also remains effective under temporal mismatch (winter LiDAR, summer RGBI), removing strict co-acquisition constraints and enabling scalable 20 cm operational monitoring for workflows such as Digital Twin Germany and national Digital Orthophoto programs.
>
---
#### [new 055] Mitigating the ID-OOD Tradeoff in Open-Set Test-Time Adaptation
- **分类: cs.CV**

- **简介: 该论文属于开放集测试时适应任务，解决模型在分布偏移下同时提升ID分类和OOD检测的问题。通过引入角度损失和特征范数损失，提出ROSETTA方法，有效缓解熵最小化与最大化的冲突。**

- **链接: [https://arxiv.org/pdf/2604.01589](https://arxiv.org/pdf/2604.01589)**

> **作者:** Wenjie Zhao; Jia Li; Xin Dong; Yapeng Tian; Yu Xiang; Yunhui Guo
>
> **摘要:** Open-set test-time adaptation (OSTTA) addresses the challenge of adapting models to new environments where out-of-distribution (OOD) samples coexist with in-distribution (ID) samples affected by distribution shifts. In such settings, covariate shift-for example, changes in weather conditions such as snow-can alter ID samples, reducing model reliability. Consequently, models must not only correctly classify covariate-shifted ID (csID) samples but also effectively reject covariate-shifted OOD (csOOD) samples. Entropy minimization is a common strategy in test-time adaptation to maintain ID performance under distribution shifts, while entropy maximization is widely applied to enhance OOD detection. Several studies have sought to combine these objectives to tackle the challenges of OSTTA. However, the intrinsic conflict between entropy minimization and maximization inevitably leads to a trade-off between csID classification and csOOD detection. In this paper, we first analyze the limitations of entropy maximization in OSTTA and then introduce an angular loss to regulate feature norm magnitudes, along with a feature-norm loss to suppress csOOD logits, thereby improving OOD detection. These objectives form ROSETTA, a $\underline{r}$obust $\underline{o}$pen-$\underline{se}$t $\underline{t}$est-$\underline{t}$ime $\underline{a}$daptation. Our method achieves strong OOD detection while maintaining high ID classification performance on CIFAR-10-C, CIFAR-100-C, Tiny-ImageNet-C and ImageNet-C. Furthermore, experiments on the Cityscapes validate the method's effectiveness in real-world semantic segmentation, and results on the HAC dataset demonstrate its applicability across different open-set TTA setups.
>
---
#### [new 056] Modulate-and-Map: Crossmodal Feature Mapping with Cross-View Modulation for 3D Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文提出ModMap，用于3D异常检测与分割任务，通过跨模态特征映射和视图调制解决多视角、多模态数据中的异常识别问题。**

- **链接: [https://arxiv.org/pdf/2604.02328](https://arxiv.org/pdf/2604.02328)**

> **作者:** Alex Costanzino; Pierluigi Zama Ramirez; Giuseppe Lisanti; Luigi Di Stefano
>
> **备注:** Accepted at CVPR Findings 2026
>
> **摘要:** We present ModMap, a natively multiview and multimodal framework for 3D anomaly detection and segmentation. Unlike existing methods that process views independently, our method draws inspiration from the crossmodal feature mapping paradigm to learn to map features across both modalities and views, while explicitly modelling view-dependent relationships through feature-wise modulation. We introduce a cross-view training strategy that leverages all possible view combinations, enabling effective anomaly scoring through multiview ensembling and aggregation. To process high-resolution 3D data, we train and publicly release a foundational depth encoder tailored to industrial datasets. Experiments on SiM3D, a recent benchmark that introduces the first multiview and multimodal setup for 3D anomaly detection and segmentation, demonstrate that ModMap attains state-of-the-art performance by surpassing previous methods by wide margins.
>
---
#### [new 057] Cross-Domain Vessel Segmentation via Latent Similarity Mining and Iterative Co-Optimization
- **分类: cs.CV**

- **简介: 该论文属于跨域视网膜血管分割任务，旨在解决训练与测试数据域差异导致的性能下降问题。通过潜在相似性挖掘和迭代优化，提升分割准确性和跨域适应性。**

- **链接: [https://arxiv.org/pdf/2604.01553](https://arxiv.org/pdf/2604.01553)**

> **作者:** Zhanqiang Guo; Jianjiang Feng; Jie Zhou
>
> **摘要:** Retinal vessel segmentation serves as a critical prerequisite for automated diagnosis of retinal pathologies. While recent advances in Convolutional Neural Networks (CNNs) have demonstrated promising performance in this task, significant performance degradation occurs when domain shifts exist between training and testing data. To address these limitations, we propose a novel domain transfer framework that leverages latent vascular similarity across domains and iterative co-optimization of generation and segmentation networks. Specifically, we first pre-train generation networks for source and target domains. Subsequently, the pretrained source-domain conditional diffusion model performs deterministic inversion to establish intermediate latent representations of vascular images, creating domain-agnostic prototypes for target synthesis. Finally, we develop an iterative refinement strategy where segmentation network and generative model undergo mutual optimization through cyclic parameter updating. This co-evolution process enables simultaneous enhancement of cross-domain image synthesis quality and segmentation accuracy. Experiments demonstrate that our framework achieves state-of-the-art performance in cross-domain retinal vessel segmentation, particularly in challenging clinical scenarios with significant modality discrepancies.
>
---
#### [new 058] Lightweight Spatiotemporal Highway Lane Detection via 3D-ResNet and PINet with ROI-Aware Attention
- **分类: cs.CV**

- **简介: 该论文属于道路车道检测任务，旨在提升实时驾驶场景下的检测精度与效率。提出两种轻量级模型，结合3D-ResNet与PINet，引入ROI注意力机制，优化特征表达并减少计算量。**

- **链接: [https://arxiv.org/pdf/2604.02188](https://arxiv.org/pdf/2604.02188)**

> **作者:** Sorna Shanmuga Raja; Abdelhafid Zenati
>
> **摘要:** This paper presents a lightweight, end-to-end highway lane detection architecture that jointly captures spatial and temporal information for robust performance in real-world driving scenarios. Building on the strengths of 3D convolutional neural networks and instance segmentation, we propose two models that integrate a 3D-ResNet encoder with a Point Instance Network (PINet) decoder. The first model enhances multi-scale feature representation using a Feature Pyramid Network (FPN) and Self-Attention mechanism to refine spatial dependencies. The second model introduces a Region of Interest (ROI) detection head to selectively focus on lane-relevant regions, thereby improving precision and reducing computational complexity. Experiments conducted on the TuSimple dataset (highway driving scenarios) demonstrate that the proposed second model achieves 93.40% accuracy while significantly reducing false negatives. Compared to existing 2D and 3D baselines, our approach achieves improved performance with fewer parameters and reduced latency. The architecture has been validated through offline training and real-time inference in the Autonomous Systems Laboratory at City, St George's University of London. These results suggest that the proposed models are well-suited for integration into Advanced Driver Assistance Systems (ADAS), with potential scalability toward full Lane Assist Systems (LAS).
>
---
#### [new 059] LESV: Language Embedded Sparse Voxel Fusion for Open-Vocabulary 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于开放词汇3D场景理解任务，解决3DGS中的空间和语义模糊问题。提出LESV框架，利用稀疏体素渲染实现精准特征注册，提升细粒度查询性能。**

- **链接: [https://arxiv.org/pdf/2604.01388](https://arxiv.org/pdf/2604.01388)**

> **作者:** Fusang Wang; Nathan Piasco; Moussab Bennehar; Luis Roldão; Dzmitry Tsishkou; Fabien Moutarde
>
> **摘要:** Recent advancements in open-vocabulary 3D scene understanding heavily rely on 3D Gaussian Splatting (3DGS) to register vision-language features into 3D space. However, we identify two critical limitations in these approaches: the spatial ambiguity arising from unstructured, overlapping Gaussians which necessitates probabilistic feature registration, and the multi-level semantic ambiguity caused by pooling features over object-level masks, which dilutes fine-grained details. To address these challenges, we present a novel framework that leverages Sparse Voxel Rasterization (SVRaster) as a structured, disjoint geometry representation. By regularizing SVRaster with monocular depth and normal priors, we establish a stable geometric foundation. This enables a deterministic, confidence-aware feature registration process and suppresses the semantic bleeding artifact common in 3DGS. Furthermore, we resolve multi-level ambiguity by exploiting the emerging dense alignment properties of foundation model AM-RADIO, avoiding the computational overhead of hierarchical training methods. Our approach achieves state-of-the-art performance on Open Vocabulary 3D Object Retrieval and Point Cloud Understanding benchmarks, particularly excelling on fine-grained queries where registration methods typically fail.
>
---
#### [new 060] Riemannian and Symplectic Geometry for Hierarchical Text-Driven Place Recognition
- **分类: cs.CV**

- **简介: 该论文属于文本到点云定位任务，解决信息丢失和场景结构不清晰的问题。提出SympLoc框架，通过多层级对齐实现更精确的跨模态检索。**

- **链接: [https://arxiv.org/pdf/2604.01598](https://arxiv.org/pdf/2604.01598)**

> **作者:** Tianyi Shang; Zhenyu Li
>
> **备注:** 9 pages
>
> **摘要:** Text-to-point-cloud localization enables robots to understand spatial positions through natural language descriptions, which is crucial for human-robot collaboration in applications such as autonomous driving and last-mile delivery. However, existing methods employ pooled global descriptors for similarity retrieval, which suffer from severe information loss and fail to capture discriminative scene structures. To address these issues, we propose SympLoc, a novel coarse-to-fine localization framework with multi-level alignment in the coarse stage. Different from previous methods that rely solely on global descriptors, our coarse stage consists of three complementary alignment levels: 1) Instance-level alignment establishes direct correspondence between individual object instances in point clouds and textual hints through Riemannian self-attention in hyperbolic space; 2) Relation-level alignment explicitly models pairwise spatial relationships between objects using the Information-Symplectic Relation Encoder (ISRE), which reformulates relation features through Fisher-Rao metric and Hamiltonian dynamics for uncertainty-aware geometrically consistent propagation; 3) Global-level alignment synthesizes discriminative global descriptors via the Spectral Manifold Transform (SMT) that extracts structural invariants through graph spectral analysis. This hierarchical alignment strategy progressively captures fine-grained to coarse-grained scene semantics, enabling robust cross-modal retrieval. Extensive experiments on the KITTI360Pose dataset demonstrate that SympLoc achieves a 19% improvement in Top-1 recall@10m compared to existing state-of-the-art approaches.
>
---
#### [new 061] Center-Aware Detection with Swin-based Co-DETR Framework for Cervical Cytology
- **分类: cs.CV**

- **简介: 该论文属于宫颈细胞图像检测任务，解决密集细胞分布和复杂形态带来的分析难题。提出基于Swin-Large的Co-DETR框架，通过中心点预测和优化策略提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.02090](https://arxiv.org/pdf/2604.02090)**

> **作者:** Yan Kong; Yuan Yin; Hongan Chen; Yuqi Fang; Caifeng Shan
>
> **备注:** ISBI 2026 Accepted Paper & Winning Solution for the RIVA Cervical Cytology Challenge
>
> **摘要:** Automated analysis of Pap smear images is critical for cervical cancer screening but remains challenging due to dense cell distribution and complex morphology. In this paper, we present our winning solution for the RIVA Cervical Cytology Challenge, achieving 1st place in Track B and 2nd place in Track A. Our approach leverages a powerful baseline, integrating the Co-DINO framework with a Swin-Large backbone for robust multi-scale feature extraction. To address the dataset's unique fixed-size bounding box annotations, we formulate the detection task as a center-point prediction problem. Tailoring our approach to this formulation, we introduce a center-preserving data augmentation strategy and an analytical geometric box optimization to effectively absorb localization jitter. Finally, we apply track-specific loss tuning to adapt the loss weights for each task. Experiments demonstrate that our targeted optimizations improve detection performance, providing an effective pipeline for cytology image analysis. Our code is available at this https URL.
>
---
#### [new 062] Look Twice: Training-Free Evidence Highlighting in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态问答任务，旨在解决MLLM在知识密集型问题中难以识别关键视觉和文本证据的问题。提出LoT框架，在不训练的情况下通过注意力机制突出相关证据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.01280](https://arxiv.org/pdf/2604.01280)**

> **作者:** Marco Morini; Sara Sarto; Marcella Cornia; Lorenzo Baraldi
>
> **备注:** Project Page: this https URL
>
> **摘要:** Answering questions about images often requires combining visual understanding with external knowledge. Multimodal Large Language Models (MLLMs) provide a natural framework for this setting, but they often struggle to identify the most relevant visual and textual evidence when answering knowledge-intensive queries. In such scenarios, models must integrate visual cues with retrieved textual evidence that is often noisy or only partially relevant, while also localizing fine-grained visual information in the image. In this work, we introduce Look Twice (LoT), a training-free inference-time framework that improves how pretrained MLLMs utilize multimodal evidence. Specifically, we exploit the model attention patterns to estimate which visual regions and retrieved textual elements are relevant to a query, and then generate the answer conditioned on this highlighted evidence. The selected cues are highlighted through lightweight prompt-level markers that encourage the model to re-attend to the relevant evidence during generation. Experiments across multiple knowledge-based VQA benchmarks show consistent improvements over zero-shot MLLMs. Additional evaluations on vision-centric and hallucination-oriented benchmarks further demonstrate that visual evidence highlighting alone improves model performance in settings without textual context, all without additional training or architectural modifications. Source code will be publicly released.
>
---
#### [new 063] FTPFusion: Frequency-Aware Infrared and Visible Video Fusion with Temporal Perturbation
- **分类: cs.CV**

- **简介: 该论文属于红外与可见光视频融合任务，解决时空稳定性与细节保留问题。提出FTPFusion方法，通过频率分解和时间扰动策略提升融合效果。**

- **链接: [https://arxiv.org/pdf/2604.01900](https://arxiv.org/pdf/2604.01900)**

> **作者:** Xilai Li; Chusheng Fang; Xiaosong Li
>
> **摘要:** Infrared and visible video fusion plays a critical role in intelligent surveillance and low-light monitoring. However, maintaining temporal stability while preserving spatial detail remains a fundamental challenge. Existing methods either focus on frame-wise enhancement with limited temporal modeling or rely on heavy spatio-temporal aggregation that often sacrifices high-frequency details. In this paper, we propose FTPFusion, a frequency-aware infrared and visible video fusion method based on temporal perturbation and sparse cross-modal interaction. Specifically, FTPFusion decomposes the feature representations into high-frequency and low-frequency components for collaborative modeling. The high-frequency branch performs sparse cross-modal spatio-temporal interaction to capture motion-related context and complementary details. The low-frequency branch introduces a temporal perturbation strategy to enhance robustness against complex video variations, such as flickering, jitter, and local misalignment. Furthermore, we design an offset-aware temporal consistency constraint to explicitly stabilize cross-frame representations under temporal disturbances. Extensive experiments on multiple public benchmarks demonstrate that FTPFusion consistently outperforms state-of-the-art methods across multiple metrics in both spatial fidelity and temporal consistency. The source code will be available at this https URL.
>
---
#### [new 064] Lifting Unlabeled Internet-level Data for 3D Scene Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D场景理解任务，旨在解决标注数据稀缺的问题。通过利用互联网上的未标注视频生成训练数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.01907](https://arxiv.org/pdf/2604.01907)**

> **作者:** Yixin Chen; Yaowei Zhang; Huangyue Yu; Junchao He; Yan Wang; Jiangyong Huang; Hongyu Shen; Junfeng Ni; Shaofei Wang; Baoxiong Jia; Song-Chun Zhu; Siyuan Huang
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Annotated 3D scene data is scarce and expensive to acquire, while abundant unlabeled videos are readily available on the internet. In this paper, we demonstrate that carefully designed data engines can leverage web-curated, unlabeled videos to automatically generate training data, to facilitate end-to-end models in 3D scene understanding alongside human-annotated datasets. We identify and analyze bottlenecks in automated data generation, revealing critical factors that determine the efficiency and effectiveness of learning from unlabeled data. To validate our approach across different perception granularities, we evaluate on three tasks spanning low-level perception, i.e., 3D object detection and instance segmentation, to high-evel reasoning, i.e., 3D spatial Visual Question Answering (VQA) and Vision-Lanugage Navigation (VLN). Models trained on our generated data demonstrate strong zero-shot performance and show further improvement after finetuning. This demonstrates the viability of leveraging readily available web data as a path toward more capable scene understanding systems.
>
---
#### [new 065] ReFlow: Self-correction Motion Learning for Dynamic Scene Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于单目动态场景重建任务，解决动态区域初始化不完整导致的重建不稳定问题。提出ReFlow框架，通过自校正机制实现精准运动学习与场景重建。**

- **链接: [https://arxiv.org/pdf/2604.01561](https://arxiv.org/pdf/2604.01561)**

> **作者:** Yanzhe Liang; Ruijie Zhu; Hanzhi Chang; Zhuoyuan Li; Jiahao Lu; Tianzhu Zhang
>
> **备注:** Project page: this https URL {this https URL}
>
> **摘要:** We present ReFlow, a unified framework for monocular dynamic scene reconstruction that learns 3D motion in a novel self-correction manner from raw video. Existing methods often suffer from incomplete scene initialization for dynamic regions, leading to unstable reconstruction and motion estimation, which often resorts to external dense motion guidance such as pre-computed optical flow to further stabilize and constrain the reconstruction of dynamic components. However, this introduces additional complexity and potential error propagation. To address these issues, ReFlow integrates a Complete Canonical Space Construction module for enhanced initialization of both static and dynamic regions, and a Separation-Based Dynamic Scene Modeling module that decouples static and dynamic components for targeted motion supervision. The core of ReFlow is a novel self-correction flow matching mechanism, consisting of Full Flow Matching to align 3D scene flow with time-varying 2D observations, and Camera Flow Matching to enforce multi-view consistency for static objects. Together, these modules enable robust and accurate dynamic scene reconstruction. Extensive experiments across diverse scenarios demonstrate that ReFlow achieves superior reconstruction quality and robustness, establishing a novel self-correction paradigm for monocular 4D reconstruction.
>
---
#### [new 066] MAR-MAER: Metric-Aware and Ambiguity-Adaptive Autoregressive Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决生成图像质量不高和处理模糊提示困难的问题。提出MAR-MAER框架，结合度量感知和概率潜变量模型，提升生成质量和多样性。**

- **链接: [https://arxiv.org/pdf/2604.01864](https://arxiv.org/pdf/2604.01864)**

> **作者:** Kai Dong; Tingting Bai
>
> **备注:** Accepted by AMME 2025
>
> **摘要:** Autoregressive (AR) models have demonstrated significant success in the realm of text-to-image generation. However, they usually face two major challenges. Firstly, the generated images may not always meet the quality standards expected by humans. Furthermore, these models face difficulty when dealing with ambiguous prompts that could be interpreted in several valid ways. To address these issues, we introduce MAR-MAER, an innovative hierarchical autoregressive framework. It combines two main components. It is a metric-aware embedding regularization method. The other one is a probabilistic latent model used for handling ambiguous semantics. Our method utilizes a lightweight projection head, which is trained with an adaptive kernel regression loss function. This aligns the model's internal representations with human-preferred quality metrics, such as CLIPScore and HPSv2. As a result, the embedding space that is learned more accurately reflects human judgment. We are also introducing a conditional variational module. This approach incorporates an aspect of controlled randomness within the hierarchical token generation process. This capability allows the model to produce a diverse array of coherent images based on ambiguous or open-ended prompts. We conducted extensive experiments using COCO and a newly developed Ambiguous-Prompt Benchmark. The results show that MAR-MAER achieves excellent performance in both metric consistency and semantic flexibility. It exceeds the baseline Hi-MAR model's performance, showing an improvement of +1.6 in CLIPScore and +5.3 in HPSv2. For unclear inputs, it produces a notably wider range of outputs. These findings have been confirmed through both human evaluation and automated metrics.
>
---
#### [new 067] Beyond the Fold: Quantifying Split-Level Noise and the Case for Leave-One-Dataset-Out AU Evaluation
- **分类: cs.CV**

- **简介: 该论文属于面部动作单元检测任务，旨在解决交叉验证带来的评估偏差问题。通过分析不同分割方式的噪声，提出使用留一数据集验证以提高结果稳定性。**

- **链接: [https://arxiv.org/pdf/2604.02162](https://arxiv.org/pdf/2604.02162)**

> **作者:** Saurabh Hinduja; Gurmeet Kaur; Maneesh Bilalpur; Jeffrey Cohn; Shaun Canavan
>
> **备注:** CVPR 2026
>
> **摘要:** Subject-exclusive cross-validation is the standard evaluation protocol for facial Action Unit (AU) detection, yet reported improvements are often small. We show that cross-validation itself introduces measurable stochastic variance. On BP4D+, repeated 3-fold subject-exclusive splits produce an empirical noise floor of $\pm 0.065$ in average F1, with substantially larger variation for low-prevalence AUs. Operating-point metrics such as F1 fluctuate more than threshold-independent measures such as AUC, and model ranking can change under different fold assignments. We further evaluate cross-dataset robustness using a Leave-One-Dataset-Out (LODO) protocol across five AU datasets. LODO removes partition randomness and exposes domain-level instability that is not visible under single-dataset cross-validation. Together, these results suggest that gains often reported in cross-fold validation may fall within protocol variance. Leave-one-dataset-out cross-validation yields more stable and interpretable findings
>
---
#### [new 068] Better Rigs, Not Bigger Networks: A Body Model Ablation for Gaussian Avatars
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D人体重建任务，旨在提升虚拟角色的重建质量。通过替换身体模型，减少训练复杂度，验证了身体模型表达能力对重建效果的关键作用。**

- **链接: [https://arxiv.org/pdf/2604.01447](https://arxiv.org/pdf/2604.01447)**

> **作者:** Derek Austin
>
> **摘要:** Recent 3D Gaussian splatting methods built atop SMPL achieve remarkable visual fidelity while continually increasing the complexity of the overall training architecture. We demonstrate that much of this complexity is unnecessary: by replacing SMPL with the Momentum Human Rig (MHR), estimated via SAM-3D-Body, a minimal pipeline with no learned deformations or pose-dependent corrections achieves the highest reported PSNR and competitive or superior LPIPS and SSIM on PeopleSnapshot and ZJU-MoCap. To disentangle pose estimation quality from body model representational capacity, we perform two controlled ablations: translating SAM-3D-Body meshes to SMPL-X, and translating the original dataset's SMPL poses into MHR both retrained under identical conditions. These ablations confirm that body model expressiveness has been a primary bottleneck in avatar reconstruction, with both mesh representational capacity and pose estimation quality contributing meaningfully to the full pipeline's gains.
>
---
#### [new 069] True to Tone? Quantifying Skin Tone Fidelity and Bias in Photographic-to-Virtual Human Pipelines
- **分类: cs.CV**

- **简介: 该论文属于虚拟人渲染任务，旨在解决皮肤色调准确性和偏差问题。通过自动流程评估不同提取策略的肤色保真度，发现深色皮肤存在更高误差。**

- **链接: [https://arxiv.org/pdf/2604.02055](https://arxiv.org/pdf/2604.02055)**

> **作者:** Gabriel Ferri Schneider; Erick Menezes; Rafael Mecenas; Paulo Knob; Victor Araujo; Soraia Raupp Musse
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Accurate reproduction of facial skin tone is essential for realism, identity preservation, and fairness in Virtual Human (VH) rendering. However, most accessible avatar creation pipelines rely on photographic inputs that lack colorimetric calibration, which can introduce inconsistencies and bias. We propose a fully automatic and scalable methodology to systematically evaluate skin tone fidelity across the VH generation pipeline. Our approach defines a full workflow that integrates skin color and illumination extraction, texture recolorization, real-time rendering, and quantitative color analysis. Using facial images from the Chicago Face Database (CFD), we compare skin tone extraction strategies based on cheek-region sampling, following the literature, and multidimensional masking derived from full-face analysis. Additionally, we test both strategies with lighting isolation, using the pre-trained TRUST framework, employed without any training or optimization within our pipeline. Extracted skin tones are applied to MetaHuman textures and rendered under multiple lighting configurations. Skin tone consistency is evaluated objectively in the CIELAB color space using the $\Delta E$ metric and the Individual Typology Angle (ITA). The proposed methodology operates without manual intervention and, with the exception of pre-trained illumination compensation modules, the pipeline does not include learning or training stages, enabling low computational cost and large-scale evaluation. Using this framework, we generate and analyze approximately 19,848 rendered instances. Our results show phenotype-dependent behavior of extraction strategies and consistently higher colorimetric errors for darker skin tones.
>
---
#### [new 070] Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation
- **分类: cs.CV**

- **简介: 该论文属于时间动作分割任务，旨在提升细粒度分割质量。通过引入边界回归和段级正则化损失，实现更精确的时序定位与段内一致性，无需复杂架构。**

- **链接: [https://arxiv.org/pdf/2604.01859](https://arxiv.org/pdf/2604.01859)**

> **作者:** Hinako Mitsuoka; Kazuhiro Hotta
>
> **备注:** Accepted by CVPR2026 Workshop "AI-driven Skilled Activity Understanding, Assessment & Feedback Generation (SAUAFG)"
>
> **摘要:** Recent progress in Temporal Action Segmentation (TAS) has increasingly relied on complex architectures, which can hinder practical deployment. We present a lightweight dual-loss training framework that improves fine-grained segmentation quality with only one additional output channel and two auxiliary loss terms, requiring minimal architectural modification. Our approach combines a boundary-regression loss that promotes accurate temporal localization via a single-channel boundary prediction and a CDF-based segment-level regularization loss that encourages coherent within-segment structure by matching cumulative distributions over predicted and ground-truth segments. The framework is architecture-agnostic and can be integrated into existing TAS models (e.g., MS-TCN, C2F-TCN, FACT) as a training-time loss function. Across three benchmark datasets, the proposed method improves segment-level consistency and boundary quality, yielding higher F1 and Edit scores across three different models. Frame-wise accuracy remains largely unchanged, highlighting that precise segmentation can be achieved through simple loss design rather than heavier architectures or inference-time refinements.
>
---
#### [new 071] Towards Minimal Focal Stack in Shape from Focus
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决SFF方法依赖大量焦堆的问题。通过引入辅助图像和EOD图，结合深度网络，实现用两幅图像精确重建深度。**

- **链接: [https://arxiv.org/pdf/2604.01603](https://arxiv.org/pdf/2604.01603)**

> **作者:** Khurram Ashfaq; Muhammad Tariq Mahmood
>
> **备注:** Accepted to CVPRW 2026 (3DMV)
>
> **摘要:** Shape from Focus (SFF) is a depth reconstruction technique that estimates scene structure from focus variations observed across a focal stack, that is, a sequence of images captured at different focus settings. A key limitation of SFF methods is their reliance on densely sampled, large focal stacks, which limits their practical applicability. In this study, we propose a focal stack augmentation that enables SFF methods to estimate depth using a reduced stack of just two images, without sacrificing precision. We introduce a simple yet effective physics-based focal stack augmentation that enriches the stack with two auxiliary cues: an all-in-focus (AiF) image estimated from two input images, and Energy-of-Difference (EOD) maps, computed as the energy of differences between the AiF and input images. Furthermore, we propose a deep network that computes a deep focus volume from the augmented focal stacks and iteratively refines depth using convolutional Gated Recurrent Units (ConvGRUs) at multiple scales. Extensive experiments on both synthetic and real-world datasets demonstrate that the proposed augmentation benefits existing state-of-the-art SFF models, enabling them to achieve comparable accuracy. The results also show that our approach maintains state-of-the-art performance with a minimal stack size.
>
---
#### [new 072] Generative World Renderer
- **分类: cs.CV**

- **简介: 该论文提出一个大规模动态数据集，用于改进生成式渲染任务中的真实感和时间一致性问题，解决合成数据与现实场景的差距。**

- **链接: [https://arxiv.org/pdf/2604.02329](https://arxiv.org/pdf/2604.02329)**

> **作者:** Zheng-Hui Huang; Zhixiang Wang; Jiaming Tan; Ruihan Yu; Yidan Zhang; Bo Zheng; Yu-Lun Liu; Yung-Yu Chuang; Kaipeng Zhang
>
> **备注:** Project page: this https URL
>
> **摘要:** Scaling generative inverse and forward rendering to real-world scenarios is bottlenecked by the limited realism and temporal coherence of existing synthetic datasets. To bridge this persistent domain gap, we introduce a large-scale, dynamic dataset curated from visually complex AAA games. Using a novel dual-screen stitched capture method, we extracted 4M continuous frames (720p/30 FPS) of synchronized RGB and five G-buffer channels across diverse scenes, visual effects, and environments, including adverse weather and motion-blur variants. This dataset uniquely advances bidirectional rendering: enabling robust in-the-wild geometry and material decomposition, and facilitating high-fidelity G-buffer-guided video generation. Furthermore, to evaluate the real-world performance of inverse rendering without ground truth, we propose a novel VLM-based assessment protocol measuring semantic, spatial, and temporal consistency. Experiments demonstrate that inverse renderers fine-tuned on our data achieve superior cross-dataset generalization and controllable generation, while our VLM evaluation strongly correlates with human judgment. Combined with our toolkit, our forward renderer enables users to edit styles of AAA games from G-buffers using text prompts.
>
---
#### [new 073] IGLOSS: Image Generation for Lidar Open-vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D点云的开放词汇语义分割任务，旨在解决VLMs存在的模态差距问题。通过文本生成图像并匹配特征，实现高效的点云分类。**

- **链接: [https://arxiv.org/pdf/2604.01361](https://arxiv.org/pdf/2604.01361)**

> **作者:** Nermin Samet; Gilles Puy; Renaud Marlet
>
> **摘要:** This paper presents a new method for the zero-shot open-vocabulary semantic segmentation (OVSS) of 3D automotive lidar data. To circumvent the recognized image-text modality gap that is intrinsic to approaches based on Vision Language Models (VLMs) such as CLIP, our method relies instead on image generation from text, to create prototype images. Given a 3D network distilled from a 2D Vision Foundation Model (VFM), we then label a point cloud by matching 3D point features with 2D image features of these prototypes. Our method is state-of-the-art for OVSS on nuScenes and SemanticKITTI. Code, pre-trained models, and generated images are available at this https URL.
>
---
#### [new 074] AdamFlow: Adam-based Wasserstein Gradient Flows for Surface Registration in Medical Imaging
- **分类: cs.CV; math.OC**

- **简介: 该论文属于医学影像中的表面配准任务，旨在解决效率与鲁棒性的平衡问题。通过将表面网格建模为概率测度，使用AdamFlow优化方法实现快速准确的配准。**

- **链接: [https://arxiv.org/pdf/2604.02290](https://arxiv.org/pdf/2604.02290)**

> **作者:** Qiang Ma; Qingjie Meng; Xin Hu; Yicheng Wu; Wenjia Bai
>
> **摘要:** Surface registration plays an important role for anatomical shape analysis in medical imaging. Existing surface registration methods often face a trade-off between efficiency and robustness. Local point matching methods are computationally efficient, but vulnerable to noise and initialisation. Methods designed for global point set alignment tend to incur a high computational cost. To address the challenge, here we present a fast surface registration method, which formulates surface meshes as probability measures and surface registration as a distributional optimisation problem. The discrepancy between two meshes is measured using an efficient sliced Wasserstein distance with log-linear computational complexity. We propose a novel optimisation method, AdamFlow, which generalises the well-known Adam optimisation method from the Euclidean space to the probability space for minimising the sliced Wasserstein distance. We theoretically analyse the asymptotic convergence of AdamFlow and empirically demonstrate its superior performance in both affine and non-rigid surface registration across various anatomical structures.
>
---
#### [new 075] COMPASS: Complete Multimodal Fusion via Proxy Tokens and Shared Spaces for Ubiquitous Sensing
- **分类: cs.CV**

- **简介: 该论文属于多模态感知任务，解决缺失模态导致的融合不完整问题。提出COMPASS框架，通过生成代理令牌保持固定输入结构，提升跨模态交互效果。**

- **链接: [https://arxiv.org/pdf/2604.02056](https://arxiv.org/pdf/2604.02056)**

> **作者:** Hao Wang; Yanyu Qian; Pengcheng Weng; Zixuan Xia; William Dan; Yangxin Xu; Fei Wang
>
> **摘要:** Missing modalities remain a major challenge for multimodal sensing, because most existing methods adapt the fusion process to the observed subset by dropping absent branches, using subset-specific fusion, or reconstructing missing features. As a result, the fusion head often receives an input structure different from the one seen during training, leading to incomplete fusion and degraded cross-modal interaction. We propose COMPASS, a missing-modality fusion framework built on the principle of fusion completeness: the fusion head always receives a fixed N-slot multimodal input, with one token per modality slot. For each missing modality, COMPASS synthesizes a target-specific proxy token from the observed modalities using pairwise source-to-target generators in a shared latent space, and aggregates them into a single replacement token. To make these proxies both representation-compatible and task-informative, we combine proxy alignment, shared-space regularization, and per-proxy discriminative supervision. Experiments on XRF55, MM-Fi, and OctoNet under diverse single- and multiple-missing settings show that COMPASS outperforms prior methods on the large majority of scenarios. Our results suggest that preserving a modality-complete fusion interface is a simple and effective design principle for robust multimodal sensing.
>
---
#### [new 076] Investigating Permutation-Invariant Discrete Representation Learning for Spatially Aligned Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决离散表示中位置信息依赖的问题。提出PI-VQ模型，使潜在代码无位置信息，提升语义特征表达并支持直接插值生成图像。**

- **链接: [https://arxiv.org/pdf/2604.01843](https://arxiv.org/pdf/2604.01843)**

> **作者:** Jamie S. J. Stirling; Noura Al-Moubayed; Hubert P. H. Shum
>
> **备注:** 15 pages plus references; 5 figures; supplementary appended; accepted to ICPR 2026
>
> **摘要:** Vector quantization approaches (VQ-VAE, VQ-GAN) learn discrete neural representations of images, but these representations are inherently position-dependent: codes are spatially arranged and contextually entangled, requiring autoregressive or diffusion-based priors to model their dependencies at sample time. In this work, we ask whether positional information is necessary for discrete representations of spatially aligned data. We propose the permutation-invariant vector-quantized autoencoder (PI-VQ), in which latent codes are constrained to carry no positional information. We find that this constraint encourages codes to capture global, semantic features, and enables direct interpolation between images without a learned prior. To address the reduced information capacity of permutation-invariant representations, we introduce matching quantization, a vector quantization algorithm based on optimal bipartite matching that increases effective bottleneck capacity by $3.5\times$ relative to naive nearest-neighbour quantization. The compositional structure of the learned codes further enables interpolation-based sampling, allowing synthesis of novel images in a single forward pass. We evaluate PI-VQ on CelebA, CelebA-HQ and FFHQ, obtaining competitive precision, density and coverage metrics for images synthesised with our approach. We discuss the trade-offs inherent to position-free representations, including separability and interpretability of the latent codes, pointing to numerous directions for future work.
>
---
#### [new 077] Camouflage-aware Image-Text Retrieval via Expert Collaboration
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像-文本检索任务，旨在解决伪装场景下的跨模态对齐问题。通过构建数据集和提出CECNet模型，提升伪装目标的检索准确率。**

- **链接: [https://arxiv.org/pdf/2604.01251](https://arxiv.org/pdf/2604.01251)**

> **作者:** Yao Jiang; Zhongkuan Mao; Xuan Wu; Keren Fu; Qijun Zhao
>
> **摘要:** Camouflaged scene understanding (CSU) has attracted significant attention due to its broad practical implications. However, in this field, robust image-text cross-modal alignment remains under-explored, hindering deeper understanding of camouflaged scenarios and their related applications. To this end, we focus on the typical image-text retrieval task, and formulate a new task dubbed ``camouflage-aware image-text retrieval'' (CA-ITR). We first construct a dedicated camouflage image-text retrieval dataset (CamoIT), comprising $\sim$10.5K samples with multi-granularity textual annotations. Benchmark results conducted on CamoIT reveal the underlying challenges of CA-ITR for existing cutting-edge retrieval techniques, which are mainly caused by objects' camouflage properties as well as those complex image contents. As a solution, we propose a camouflage-expert collaborative network (CECNet), which features a dual-branch visual encoder: one branch captures holistic image representations, while the other incorporates a dedicated model to inject representations of camouflaged objects. A novel confidence-conditioned graph attention (C\textsuperscript{2}GA) mechanism is incorporated to exploit the complementarity across branches. Comparative experiments show that CECNet achieves $\sim$29% overall CA-ITR accuracy boost, surpassing seven representative retrieval models. The dataset and code will be available at this https URL.
>
---
#### [new 078] VOID: Video Object and Interaction Deletion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频对象移除任务，解决物体移除后物理交互不一致的问题。提出VOID框架，通过因果推理生成物理合理的场景变化。**

- **链接: [https://arxiv.org/pdf/2604.02296](https://arxiv.org/pdf/2604.02296)**

> **作者:** Saman Motamed; William Harvey; Benjamin Klein; Luc Van Gool; Zhuoning Yuan; Ta-Ying Cheng
>
> **摘要:** Existing video object removal methods excel at inpainting content "behind" the object and correcting appearance-level artifacts such as shadows and reflections. However, when the removed object has more significant interactions, such as collisions with other objects, current models fail to correct them and produce implausible results. We present VOID, a video object removal framework designed to perform physically-plausible inpainting in these complex scenarios. To train the model, we generate a new paired dataset of counterfactual object removals using Kubric and HUMOTO, where removing an object requires altering downstream physical interactions. During inference, a vision-language model identifies regions of the scene affected by the removed object. These regions are then used to guide a video diffusion model that generates physically consistent counterfactual outcomes. Experiments on both synthetic and real data show that our approach better preserves consistent scene dynamics after object removal compared to prior video object removal methods. We hope this framework sheds light on how to make video editing models better simulators of the world through high-level causal reasoning.
>
---
#### [new 079] GeoAI Agency Primitives
- **分类: cs.CV**

- **简介: 该论文属于GIS与AI交叉任务，旨在解决GeoAI助手在实际工作流中效率低的问题。提出9种代理原语及基准，提升GIS工作者的生产力。**

- **链接: [https://arxiv.org/pdf/2604.01869](https://arxiv.org/pdf/2604.01869)**

> **作者:** Akram Zaytar; Rohan Sawahn; Caleb Robinson; Gilles Q. Hacheme; Girmaw A. Tadesse; Inbal Becker-Reshef; Rahul Dodhia; Juan Lavista Ferres
>
> **摘要:** We present ongoing research on agency primitives for GeoAI assistants -- core capabilities that connect Foundation models to the artifact-centric, human-in-the-loop workflows where GIS practitioners actually work. Despite advances in satellite image captioning, visual question answering, and promptable segmentation, these capabilities have not translated into productivity gains for practitioners who spend most of their time producing vector layers, raster maps, and cartographic products. The gap is not model capability alone but the absence of an agency layer that supports iterative collaboration. We propose a vocabulary of $9$ primitives for such a layer -- including navigation, perception, geo-referenced memory, and dual modeling -- along with a benchmark that measures human productivity. Our goal is a vocabulary that makes agentic assistance in GIS implementable, testable, and comparable.
>
---
#### [new 080] GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出GRAZE方法，用于在无监督情况下精确定位美式足球训练视频中的首次接触帧。任务是事件定位，解决接触时刻识别难题。**

- **链接: [https://arxiv.org/pdf/2604.01383](https://arxiv.org/pdf/2604.01383)**

> **作者:** Syed Ahsan Masud Zaidi; Lior Shamir; William Hsu; Scott Dietrich; Talha Zaidi
>
> **备注:** 9 pages, 5 figures, accepted to the CVPR 2026 Workshop on Computer Vision in Sports (CVSports) code: this https URL
>
> **摘要:** American football practice generates video at scale, yet the interaction of interest occupies only a brief window of each long, untrimmed clip. Reliable biomechanical analysis, therefore, depends on spatiotemporal localization that identifies both the interacting entities and the onset of contact. We study First Point of Contact (FPOC), defined as the first frame in which a player physically touches a tackle dummy, in unconstrained practice footage with camera motion, clutter, multiple similarly equipped athletes, and rapid pose changes around impact. We present GRAZE, a training-free pipeline for FPOC localization that requires no labeled tackle-contact examples. GRAZE uses Grounding DINO to discover candidate player-dummy interactions, refines them with motion-aware temporal reasoning, and uses SAM2 as an explicit pixel-level verifier of contact rather than relying on detection confidence alone. This separation between candidate discovery and contact confirmation makes the approach robust to cluttered scenes and unstable grounding near impact. On 738 tackle-practice videos, GRAZE produces valid outputs for 97.4% of clips and localizes FPOC within $\pm$ 10 frames on 77.5% of all clips and within $\pm$ 20 frames on 82.7% of all clips. These results show that frame-accurate contact onset localization in real-world practice footage is feasible without task-specific training.
>
---
#### [new 081] Interactive Tracking: A Human-in-the-Loop Paradigm with Memory-Augmented Adaptation
- **分类: cs.CV**

- **简介: 该论文属于目标跟踪任务，旨在解决传统跟踪器无法支持人类实时交互的问题。提出InteractTrack基准和IMAT模型，实现用户通过自然语言指令引导跟踪。**

- **链接: [https://arxiv.org/pdf/2604.01974](https://arxiv.org/pdf/2604.01974)**

> **作者:** Yuqing Huang; Guotian Zeng; Zhenqiao Yuan; Zhenyu He; Xin Li; Yaowei Wang; Ming-Hsuan Yang
>
> **摘要:** Existing visual trackers mainly operate in a non-interactive, fire-and-forget manner, making them impractical for real-world scenarios that require human-in-the-loop adaptation. To overcome this limitation, we introduce Interactive Tracking, a new paradigm that allows users to guide the tracker at any time using natural language commands. To support research in this direction, we make three main contributions. First, we present InteractTrack, the first large-scale benchmark for interactive tracking, containing 150 videos with dense bounding box annotations and timestamped language instructions. Second, we propose a comprehensive evaluation protocol and evaluate 25 representative trackers, showing that state-of-the-art methods fail in interactive scenarios; strong performance on conventional benchmarks does not transfer. Third, we introduce Interactive Memory-Augmented Tracking (IMAT), a new baseline that employs a dynamic memory mechanism to learn from user feedback and update tracking behavior accordingly. Our benchmark, protocol, and baseline establish a foundation for developing more intelligent, adaptive, and collaborative tracking systems, bridging the gap between automated perception and human guidance. The full benchmark, tracking results, and analysis are available at this https URL.
>
---
#### [new 082] A Simple Baseline for Streaming Video Understanding
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频流处理问题。提出简单滑动窗口方法SimpleStream，无需复杂记忆机制即可达到优秀效果，揭示了记忆与实时感知的权衡。**

- **链接: [https://arxiv.org/pdf/2604.02317](https://arxiv.org/pdf/2604.02317)**

> **作者:** Yujiao Shen; Shulin Tian; Jingkang Yang; Ziwei Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent streaming video understanding methods increasingly rely on complex memory mechanisms to handle long video streams. We challenge this trend with a simple finding: a sliding-window baseline that feeds only the most recent N frames to an off-the-shelf VLM already matches or surpasses published streaming models. We formalize this baseline as SimpleStream and evaluate it against 13 major offline and online video LLM baselines on OVO-Bench and StreamingBench. Despite its simplicity, SimpleStream delivers consistently strong performance. With only 4 recent frames, it reaches 67.7% average accuracy on OVO-Bench and 80.59% on StreamingBench. Controlled ablations further show that the value of longer context is backbone-dependent rather than uniformly increasing with model scale, and reveal a consistent perception-memory trade-off: adding more historical context can improve recall, but often weakens real-time perception. This suggests that stronger memory, retrieval, or compression modules should not be taken as evidence of progress unless they clearly outperform SimpleStream under the same protocol. We therefore argue that future streaming benchmarks should separate recent-scene perception from long-range memory, so that performance improvements from added complexity can be evaluated more clearly.
>
---
#### [new 083] Universal computational thermal imaging overcoming the ghosting effect
- **分类: cs.CV; physics.optics**

- **简介: 该论文属于计算机视觉任务，旨在解决热成像中的鬼影效应问题。通过提出TAG框架，实现高保真夜视成像，提升纹理和表情恢复效果。**

- **链接: [https://arxiv.org/pdf/2604.01542](https://arxiv.org/pdf/2604.01542)**

> **作者:** Hongyi Xu; Du Wang; Chenjun Zhao; Jiashuo Chen; Jiale Lin; Liqin Cao; Yanfei Zhong; Yiyuan She; Fanglin Bao
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Thermal imaging is crucial for night vision but fundamentally hampered by the ghosting effect, a loss of detailed texture in cluttered photon streams. While conventional ghosting mitigation has relied on data post-processing, the recent breakthrough in heat-assisted detection and ranging (HADAR) opens a promising frontier for hyperspectral computational thermal imaging that produces night vision with day-like visibility. However, universal anti-ghosting imaging remains elusive, as state-of-the-art HADAR applies only to limited scenes with uniform materials, whereas material non-uniformity is ubiquitous in the real world. Here, we propose a universal computational thermal imaging framework, TAG (thermal anti-ghosting), to address material non-uniformity and overcome ghosting for high-fidelity night vision. TAG takes hyperspectral photon streams for nonparametric texture recovery, enabling our experimental demonstration of unprecedented expression recovery in thus-far-elusive ghostly human faces -- the archetypal, long-recognized ghosting phenomenon. Strikingly, TAG not only universally outperforms HADAR across various scenes, but also reveals the influence of material non-uniformity, shedding light on HADAR's effectiveness boundary. We extensively test facial texture and expression recovery across day and night, and demonstrate, for the first time, thermal 3D topological alignment and mood detection. This work establishes a universal foundation for high-fidelity computational night vision, with potential applications in autonomous navigation, reconnaissance, healthcare, and wildlife monitoring.
>
---
#### [new 084] Hidden Meanings in Plain Sight: RebusBench for Evaluating Cognitive Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视觉推理任务，旨在解决模型在隐含信息理解上的不足。通过构建RebusBench基准，评估模型整合视觉与知识进行抽象推理的能力。**

- **链接: [https://arxiv.org/pdf/2604.01764](https://arxiv.org/pdf/2604.01764)**

> **作者:** Seyed Amir Kasaei; Arash Marioriyad; Mahbod Khaleti; MohammadAmin Fazli; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **备注:** Accepted at ICLR 2026 Workshop: From Human Cognition to AI Reasoning (HCAIR)
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved remarkable proficiency in explicit visual recognition, effectively describing what is directly visible in an image. However, a critical cognitive gap emerges when the visual input serves only as a clue rather than the answer. We identify that current models struggle with the complex, multi-step reasoning required to solve problems where information is not explicitly depicted. Successfully solving a rebus puzzle requires a distinct cognitive workflow: the model must extract visual and textual attributes, retrieve linguistic prior knowledge (such as idioms), and perform abstract mapping to synthesize these elements into a meaning that exists outside the pixel space. To evaluate this neurosymbolic capability, we introduce RebusBench, a benchmark of 1,164 puzzles designed to test this specific integration of perception and knowledge. Our evaluation of state-of-the-art models (including Qwen, InternVL, and LLaVA) shows a severe deficiency: performance saturates below 10% Exact Match and 20% semantic accuracy, with no significant improvement observed from model scaling or In-Context Learning (ICL). These findings suggest that while models possess the necessary visual and linguistic components, they lack the cognitive reasoning glue to connect them. Project page available at this https URL.
>
---
#### [new 085] A deep learning pipeline for PAM50 subtype classification using histopathology images and multi-objective patch selection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于乳腺癌PAM50亚型分类任务，旨在通过病理图像预测基因亚型，减少对昂贵分子检测的依赖。研究提出一种结合优化算法和不确定性估计的深度学习方法，提升分类性能与效率。**

- **链接: [https://arxiv.org/pdf/2604.01798](https://arxiv.org/pdf/2604.01798)**

> **作者:** Arezoo Borji; Gernot Kronreif; Bernhard Angermayr; Francisco Mario Calisto; Wolfgang Birkfellner; Inna Servetnyk; Yinyin Yuan; Sepideh Hatamikia
>
> **摘要:** Breast cancer is a highly heterogeneous disease with diverse molecular profiles. The PAM50 gene signature is widely recognized as a standard for classifying breast cancer into intrinsic subtypes, enabling more personalized treatment strategies. In this study, we introduce a novel optimization-driven deep learning framework that aims to reduce reliance on costly molecular assays by directly predicting PAM50 subtypes from H&E-stained whole-slide images (WSIs). Our method jointly optimizes patch informativeness, spatial diversity, uncertainty, and patch count by combining the non-dominated sorting genetic algorithm II (NSGA-II) with Monte Carlo dropout-based uncertainty estimation. The proposed method can identify a small but highly informative patch subset for classification. We used a ResNet18 backbone for feature extraction and a custom CNN head for classification. For evaluation, we used the internal TCGA-BRCA dataset as the training cohort and the external CPTAC-BRCA dataset as the test cohort. On the internal dataset, an F1-score of 0.8812 and an AUC of 0.9841 using 627 WSIs from the TCGA-BRCA cohort were achieved. The performance of the proposed approach on the external validation dataset showed an F1-score of 0.7952 and an AUC of 0.9512. These findings indicate that the proposed optimization-guided, uncertainty-aware patch selection can achieve high performance and improve the computational efficiency of histopathology-based PAM50 classification compared to existing methods, suggesting a scalable imaging-based replacement that has the potential to support clinical decision-making.
>
---
#### [new 086] SPAR: Single-Pass Any-Resolution ViT for Open-vocabulary Segmentation
- **分类: cs.CV**

- **简介: 该论文针对开放词汇分割任务，解决ViT在高分辨率下计算成本高的问题。提出SPAR模型，通过单次处理实现高效高分辨率推理。**

- **链接: [https://arxiv.org/pdf/2604.02252](https://arxiv.org/pdf/2604.02252)**

> **作者:** Naomi Kombol; Ivan Martinović; Siniša Šegvić; Giorgos Tolias
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Foundational Vision Transformers (ViTs) have limited effectiveness in tasks requiring fine-grained spatial understanding, due to their fixed pre-training resolution and inherently coarse patch-level representations. These challenges are especially pronounced in dense prediction scenarios, such as open-vocabulary segmentation with ViT-based vision-language models, where high-resolution inputs are essential for accurate pixel-level reasoning. Existing approaches typically process large-resolution images using a sliding-window strategy at the pre-training resolution. While this improves accuracy through finer strides, it comes at a significant computational cost. We introduce SPAR: Single-Pass Any-Resolution ViT, a resolution-agnostic dense feature extractor designed for efficient high-resolution inference. We distill the spatial reasoning capabilities of a finely-strided, sliding-window teacher into a single-pass student using a feature regression loss, without requiring architectural changes or pixel-level supervision. Applied to open-vocabulary segmentation, SPAR improves single-pass baselines by up to 10.5 mIoU and even surpasses the teacher, demonstrating effectiveness in efficient, high-resolution reasoning. Code: this https URL
>
---
#### [new 087] Sparse Spectral LoRA: Routed Experts for Medical VLMs
- **分类: cs.CV**

- **简介: 该论文属于医疗视觉语言模型任务，解决医学影像中模型泛化性和持续学习问题。提出MedQwen，结合谱路由专家和低秩更新，提升性能并减少遗忘。**

- **链接: [https://arxiv.org/pdf/2604.01310](https://arxiv.org/pdf/2604.01310)**

> **作者:** Omid Nejati Manzari; Hojat Asgariandehkordi; Taha Koleilat; Yiming Xiao; Hassan Rivaz
>
> **摘要:** Large vision-language models (VLMs) excel on general benchmarks but often lack robustness in medical imaging, where heterogeneous supervision induces cross-dataset interference and sensitivity to data regime (i.e., how the supervisory signals are mixed). In realistic clinical workflows, data and tasks arrive sequentially, so naive continual training further leads to catastrophic forgetting. To address these challenges, we propose MedQwen, a parameter-efficient medical VLM that couples a spectrally routed Mixture-of-Experts (MoE) with a theoretically grounded scaling rule that aligns low-rank updates with a full-rank, fully fine-tuned MoE, without changing the base architecture. Concretely, we initialize each expert from non-overlapping singular value decomposition (SVD) segments of the pretrained weight and introduce a residual compensation and scaling scheme to enable stable expert specialization and consistent routing under distribution shift. Across 23 medical datasets covering visual question answering, report generation, radiology classification, and hallucination mitigation, MedQwen achieves strong, reliable performance: it approaches full fine-tuning on zero-shot classification with 339$\times$ fewer trainable parameters, and reduces sequential forgetting to $\sim$5\% where strong baselines degrade by $>$20-50\%.
>
---
#### [new 088] Low-Effort Jailbreak Attacks Against Text-to-Image Safety Filters
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成领域的安全研究，旨在解决安全过滤机制被低努力攻击绕过的问题。通过分析提示策略，提出多种视觉越狱技术，验证了现有防护的不足。**

- **链接: [https://arxiv.org/pdf/2604.01888](https://arxiv.org/pdf/2604.01888)**

> **作者:** Ahmed B Mustafa; Zihan Ye; Yang Lu; Michael P Pound; Shreyank N Gowda
>
> **备注:** Text-to-Image version of the Anyone can Jailbreak paper. Accepted in CVPR-W AIMS 2026
>
> **摘要:** Text-to-image generative models are widely deployed in creative tools and online platforms. To mitigate misuse, these systems rely on safety filters and moderation pipelines that aim to block harmful or policy violating content. In this work we show that modern text-to-image models remain vulnerable to low-effort jailbreak attacks that require only natural language prompts. We present a systematic study of prompt-based strategies that bypass safety filters without model access, optimization, or adversarial training. We introduce a taxonomy of visual jailbreak techniques including artistic reframing, material substitution, pseudo-educational framing, lifestyle aesthetic camouflage, and ambiguous action substitution. These strategies exploit weaknesses in prompt moderation and visual safety filtering by masking unsafe intent within benign semantic contexts. We evaluate these attacks across several state-of-the-art text-to-image systems and demonstrate that simple linguistic modifications can reliably evade existing safeguards and produce restricted imagery. Our findings highlight a critical gap between surface-level prompt filtering and the semantic understanding required to detect adversarial intent in generative media systems. Across all tested models and attack categories we observe an attack success rate (ASR) of up to 74.47%.
>
---
#### [new 089] SafeRoPE: Risk-specific Head-wise Embedding Rotation for Safe Generation in Rectified Flow Transformers
- **分类: cs.CV**

- **简介: 该论文属于文本生成图像任务，解决Transformer模型中不安全语义的问题。通过SafeRoPE框架，实现对不安全内容的精准抑制，同时保持生成质量。**

- **链接: [https://arxiv.org/pdf/2604.01826](https://arxiv.org/pdf/2604.01826)**

> **作者:** Xiang Yang; Feifei Li; Mi Zhang; Geng Hong; Xiaoyu You; Min Yang
>
> **备注:** CVPR26
>
> **摘要:** Recent Text-to-Image (T2I) models based on rectified-flow transformers (e.g., SD3, FLUX) achieve high generative fidelity but remain vulnerable to unsafe semantics, especially when triggered by multi-token interactions. Existing mitigation methods largely rely on fine-tuning or attention modulation for concept unlearning; however, their expensive computational overhead and design tailored to U-Net-based denoisers hinder direct adaptation to transformer-based diffusion models (e.g., MMDiT). In this paper, we conduct an in-depth analysis of the attention mechanism in MMDiT and find that unsafe semantics concentrate within interpretable, low-dimensional subspaces at head level, where a finite set of safety-critical heads is responsible for unsafe feature extraction. We further observe that perturbing the Rotary Positional Embedding (RoPE) applied to the query and key vectors can effectively modify some specific concepts in the generated images. Motivated by these insights, we propose SafeRoPE, a lightweight and fine-grained safe generation framework for MMDiT. Specifically, SafeRoPE first constructs head-wise unsafe subspaces by decomposing unsafe embeddings within safety-critical heads, and computes a Latent Risk Score (LRS) for each input vector via projection onto these subspaces. We then introduce head-wise RoPE perturbations that can suppress unsafe semantics without degrading benign content or image quality. SafeRoPE combines both head-wise LRS and RoPE perturbations to perform risk-specific head-wise rotation on query and key vector embeddings, enabling precise suppression of unsafe outputs while maintaining generation fidelity. Extensive experiments demonstrate that SafeRoPE achieves SOTA performance in balancing effective harmful content mitigation and utility preservation for safe generation of MMDiT. Codes are available at this https URL.
>
---
#### [new 090] Human Pose Estimation in Trampoline Gymnastics: Improving Performance Using a New Synthetic Dataset
- **分类: cs.CV**

- **简介: 该论文属于人体姿态估计任务，旨在解决跳马项目中极端姿势识别困难的问题。通过构建合成数据集STP并微调模型，提升了2D和3D姿态估计的准确性。**

- **链接: [https://arxiv.org/pdf/2604.01322](https://arxiv.org/pdf/2604.01322)**

> **作者:** Léa Drolet-Roy; Victor Nogues; Sylvain Gaudet; Eve Charbonneau; Mickaël Begon; Lama Séoud
>
> **摘要:** Trampoline gymnastics involves extreme human poses and uncommon viewpoints, on which state-of-the art pose estimation models tend to under-perform. We demonstrate that this problem can be addressed by fine-tuning a pose estimation model on a dataset of synthetic trampoline poses (STP). STP is generated from motion capture recordings of trampoline routines. We develop a pipeline to fit noisy motion capture data to a parametric human model, then generate multiview realistic images. We use this data to fine-tune a ViTPose model, and test it on real multi-view trampoline images. The resulting model exhibits accuracy improvements in 2D which translates to improved 3D triangulation. In 2D, we obtain state-of-the-art results on such challenging data, bridging the performance gap between common and extreme poses. In 3D, we reduce the MPJPE by 12.5 mm with our best model, which represents an improvement of 19.6% compared to the pretrained ViTPose model.
>
---
#### [new 091] Setup-Independent Full Projector Compensation
- **分类: cs.CV**

- **简介: 该论文属于投影补偿任务，解决现有方法依赖特定设置的问题。提出SIComp框架，实现无需微调的通用投影补偿，通过分离几何与光度处理提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.01736](https://arxiv.org/pdf/2604.01736)**

> **作者:** Haibo Li; Qingyue Deng; Jijiang Li; Haibin Ling; Bingyao Huang
>
> **备注:** 16 pages,17 figures
>
> **摘要:** Projector compensation seeks to correct geometric and photometric distortions that occur when images are projected onto nonplanar or textured surfaces. However, most existing methods are highly setup-dependent, requiring fine-tuning or retraining whenever the surface, lighting, or projector-camera pose changes. Progress has been limited by two key challenges: (1) the absence of large, diverse training datasets and (2) existing geometric correction models are typically constrained by specific spatial setups; without further retraining or fine-tuning, they often fail to generalize directly to novel geometric configurations. We introduce SIComp, the first Setup-Independent framework for full projector Compensation, capable of generalizing to unseen setups without fine-tuning or retraining. To enable this, we construct a large-scale real-world dataset spanning 277 distinct projector-camera setups. SIComp adopts a co-adaptive design that decouples geometry and photometry: A carefully tailored optical flow module performs online geometric correction, while a novel photometric network handles photometric compensation. To further enhance robustness under varying illumination, we integrate intensity-varying surface priors into the network design. Extensive experiments demonstrate that SIComp consistently produces high-quality compensation across diverse unseen setups, substantially outperforming existing methods in terms of generalization ability and establishing the first generalizable solution to projector compensation. The code and dataset are available on our project page: this https URL
>
---
#### [new 092] SteerFlow: Steering Rectified Flows for Faithful Inversion-Based Image Editing
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决现有方法在保留源图像真实性上的不足。提出SteerFlow框架，通过优化逆过程和轨迹插值提升编辑质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.01715](https://arxiv.org/pdf/2604.01715)**

> **作者:** Thinh Dao; Zhen Wang; Kien T.Pham; Long Chen
>
> **摘要:** Recent advances in flow-based generative models have enabled training-free, text-guided image editing by inverting an image into its latent noise and regenerating it under a new target conditional guidance. However, existing methods struggle to preserve source fidelity: higher-order solvers incur additional model inferences, truncated inversion constrains editability, and feature injection methods lack architectural transferability. To address these limitations, we propose SteerFlow, a model-agnostic editing framework with strong theoretical guarantees on source fidelity. In the forward process, we introduce an Amortized Fixed-Point Solver that implicitly straightens the forward trajectory by enforcing velocity consistency across consecutive timesteps, yielding a high-fidelity inverted latent. In the backward process, we introduce Trajectory Interpolation, which adaptively blends target-editing and source-reconstruction velocities to keep the editing trajectory anchored to the source. To further improve background preservation, we introduce an Adaptive Masking mechanism that spatially constrains the editing signal with concept-guided segmentation and source-target velocity differences. Extensive experiments on FLUX.1-dev and Stable Diffusion 3.5 Medium demonstrate that SteerFlow consistently achieves better editing quality than existing methods. Finally, we show that SteerFlow extends naturally to a complex multi-turn editing paradigm without accumulating drift.
>
---
#### [new 093] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，旨在解决连续帧间深度不一致的问题。通过结合轮式里程计和光流，提升深度预测的稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2604.01791](https://arxiv.org/pdf/2604.01791)**

> **作者:** Leezy Han; Seunggyu Kim; Dongseok Shim; Hyeonbeom Lee
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Monocular depth estimation (MDE) has been widely adopted in the perception systems of autonomous vehicles and mobile robots. However, existing approaches often struggle to maintain temporal consistency in depth estimation across consecutive frames. This inconsistency not only causes jitter but can also lead to estimation failures when the depth range changes abruptly. To address these challenges, this paper proposes a consistency-aware monocular depth estimation framework that leverages wheel odometry from a mobile robot to achieve stable and coherent depth predictions over time. Specifically, we estimate camera pose and sparse depth from triangulation using optical flow between consecutive frames. The sparse depth estimates are used to update a recursive Bayesian estimate of the metric scale, which is then applied to rescale the relative depth predicted by a pre-trained depth estimation foundation model. The proposed method is evaluated on the KITTI, TartanAir, MS2, and our own dataset, demonstrating robust and accurate depth estimation performance.
>
---
#### [new 094] FaCT-GS: Fast and Scalable CT Reconstruction with Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于CT重建任务，旨在解决Gaussian Splatting（GS）方法速度慢、扩展性差的问题。提出FaCT-GS框架，优化了体素化和光栅化流程，显著提升速度并支持大规模重建。**

- **链接: [https://arxiv.org/pdf/2604.01844](https://arxiv.org/pdf/2604.01844)**

> **作者:** Pawel Tomasz Pieta; Rasmus Juul Pedersen; Sina Borgi; Jakob Sauer Jørgensen; Jens Wenzel Andreasen; Vedrana Andersen Dahl
>
> **摘要:** Gaussian Splatting (GS) has emerged as a dominating technique for image rendering and has quickly been adapted for the X-ray Computed Tomography (CT) reconstruction task. However, despite being on par or better than many of its predecessors, the benefits of GS are typically not substantial enough to motivate a transition from well-established reconstruction algorithms. This paper addresses the most significant remaining limitations of the GS-based approach by introducing FaCT-GS, a framework for fast and flexible CT reconstruction. Enabled by an in-depth optimization of the voxelization and rasterization pipelines, our new method is significantly faster than its predecessors and scales well with projection and output volume size. Furthermore, the improved voxelization enables rapid fitting of Gaussians to pre-existing volumes, which can serve as a prior for warm-starting the reconstruction, or simply as an alternative, compressed representation. FaCT-GS is over 4X faster than the State of the Art GS CT reconstruction on standard 512x512 projections, and over 13X faster on 2k projections. Implementation available at: this https URL.
>
---
#### [new 095] Cosine-Normalized Attention for Hyperspectral Image Classification
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像分类任务，旨在解决传统注意力机制因混合特征幅度与方向而效果不佳的问题。通过引入余弦归一化注意力，强调角度关系，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.01763](https://arxiv.org/pdf/2604.01763)**

> **作者:** Muhammad Ahmad; Manuel Mazzara
>
> **摘要:** Transformer-based methods have improved hyperspectral image classification (HSIC) by modeling long-range spatial-spectral dependencies; however, their attention mechanisms typically rely on dot-product similarity, which mixes feature magnitude and orientation and may be suboptimal for hyperspectral data. This work revisits attention scoring from a geometric perspective and introduces a cosine-normalized attention formulation that aligns similarity computation with the angular structure of hyperspectral signatures. By projecting query and key embeddings onto a unit hypersphere and applying a squared cosine similarity, the proposed method emphasizes angular relationships while reducing sensitivity to magnitude variations. The formulation is integrated into a spatial-spectral Transformer and evaluated under extremely limited supervision. Experiments on three benchmark datasets demonstrate that the proposed approach consistently achieves higher performance, outperforming several recent Transformer- and Mamba-based models despite using a lightweight backbone. In addition, a controlled analysis of multiple attention score functions shows that cosine-based scoring provides a reliable inductive bias for hyperspectral representation learning.
>
---
#### [new 096] SHARC: Reference point driven Spherical Harmonic Representation for Complex Shapes
- **分类: cs.CV; cs.CG**

- **简介: 该论文提出SHARC框架，用于复杂形状的合成与重建。任务是生成高精度三维形状，解决传统方法在细节和效率上的不足。通过球面谐波表示和优化参考点，提升几何保真度与计算效率。**

- **链接: [https://arxiv.org/pdf/2604.01894](https://arxiv.org/pdf/2604.01894)**

> **作者:** Panagiotis Sapoutzoglou; George Terzakis; Maria Pateraki
>
> **备注:** Accepted at ICPR 2026
>
> **摘要:** We propose SHARC, a novel framework that synthesizes arbitrary, genus-agnostic shapes by means of a collection of Spherical Harmonic (SH) representations of distance fields. These distance fields are anchored at optimally placed reference points in the interior volume of the surface in a way that maximizes learning of the finer details of the surface. To achieve this, we employ a cost function that jointly maximizes sparsity and centrality in terms of positioning, as well as visibility of the surface from their location. For each selected reference point, we sample the visible distance field to the surface geometry via ray-casting and compute the SH coefficients using the Fast Spherical Harmonic Transform (FSHT). To enhance geometric fidelity, we apply a configurable low-pass filter to the coefficients and refine the output using a local consistency constraint based on proximity. Evaluation of SHARC against state-of-the-art methods demonstrates that the proposed method outperforms existing approaches in both reconstruction accuracy and time efficiency without sacrificing model parsimony. The source code is available at this https URL.
>
---
#### [new 097] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation
- **分类: cs.CV**

- **简介: 该论文提出EgoFlow，解决egocentric视频中6DoF物体运动生成问题，通过流匹配框架结合物理约束，提升运动的物理合理性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.01421](https://arxiv.org/pdf/2604.01421)**

> **作者:** Abhishek Saroha; Huajian Zeng; Xingxing Zuo; Daniel Cremers; Xi Wang
>
> **备注:** CVPR 2026: this https URL
>
> **摘要:** Understanding and predicting object motion from egocentric video is fundamental to embodied perception and interaction. However, generating physically consistent 6DoF trajectories remains challenging due to occlusions, fast motion, and the lack of explicit physical reasoning in existing generative models. We present EgoFlow, a flow-matching framework that synthesizes realistic and physically plausible trajectories conditioned on multimodal egocentric observations. EgoFlow employs a hybrid Mamba-Transformer-Perceiver architecture to jointly model temporal dynamics, scene geometry, and semantic intent, while a gradient-guided inference process enforces differentiable physical constraints such as collision avoidance and motion smoothness. This combination yields coherent and controllable motion generation without post-hoc filtering or additional supervision. Experiments on real-world datasets HD-EPIC, EgoExo4D, and HOT3D show that EgoFlow outperforms diffusion-based and transformer baselines in accuracy, generalization, and physical realism, reducing collision rates by up to 79%, and strong generalization to unseen scenes. Our results highlight the promise of flow-based generative modeling for scalable and physically grounded egocentric motion understanding.
>
---
#### [new 098] CoRegOVCD: Consistency-Regularized Open-Vocabulary Change Detection
- **分类: cs.CV**

- **简介: 该论文属于遥感变化检测任务，解决开放词汇变化检测中概念响应不一致的问题。提出CoRegOVCD框架，通过后验校准和空间一致性约束，提升变化检测的准确性和语义可靠性。**

- **链接: [https://arxiv.org/pdf/2604.02160](https://arxiv.org/pdf/2604.02160)**

> **作者:** Weidong Tang; Hanbin Sun; Zihan Li; Yikai Wang; Feifan Zhang
>
> **摘要:** Remote sensing change detection (CD) aims to identify where land-cover semantics change across time, but most existing methods still assume a fixed label space and therefore cannot answer arbitrary user-defined queries. Open-vocabulary change detection (OVCD) instead asks for the change mask of a queried concept. In the fully training-free setting, however, dense concept responses are difficult to compare directly across dates: appearance variation, weak cross-concept competition, and the spatial continuity of many land-cover categories often produce noisy, fragmented, and semantically unreliable change evidence. We propose Consistency-Regularized Open-Vocabulary Change Detection (CoRegOVCD), a training-free dense inference framework that reformulates concept-specific change as calibrated posterior discrepancy. Competitive Posterior Calibration (CPC) and the Semantic Posterior Delta (SPD) convert raw concept responses into competition-aware queried-concept posteriors and quantify their cross-temporal discrepancy, making semantic change evidence more comparable without explicit instance matching. Geometry-Token Consistency Gate (GeoGate) and Regional Consensus Discrepancy (RCD) further suppress unsupported responses and improve spatial coherence through geometry-aware structural verification and regional consensus. Across four benchmarks spanning building-oriented and multi-class settings, CoRegOVCD consistently improves over the strongest previous training-free baseline by 2.24 to 4.98 F1$_C$ points and reaches a six-class average of 47.50% F1$_C$ on SECOND.
>
---
#### [new 099] ViT-Explainer: An Interactive Walkthrough of the Vision Transformer Pipeline
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于模型解释任务，旨在解决Vision Transformer推理过程不透明的问题。提出ViT-Explainer系统，提供从图像分块到分类的可视化分析工具。**

- **链接: [https://arxiv.org/pdf/2604.02182](https://arxiv.org/pdf/2604.02182)**

> **作者:** Juan Manuel Hernandez; Mariana Fernandez-Espinosa; Denis Parra; Diego Gomez-Zara
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Transformer-based architectures have become the shared backbone of natural language processing and computer vision. However, understanding how these models operate remains challenging, particularly in vision settings, where images are processed as sequences of patch tokens. Existing interpretability tools often focus on isolated components or expert-oriented analysis, leaving a gap in guided, end-to-end understanding of the full inference pipeline. To bridge this gap, we present ViT-Explainer, a web-based interactive system that provides an integrated visualization of Vision Transformer inference, from patch tokenization to final classification. The system combines animated walkthroughs, patch-level attention overlays, and a vision-adapted Logit Lens within both guided and free exploration modes. A user study with six participants suggests that ViT-Explainer is easy to learn and use, helping users interpret and understand Vision Transformer behavior.
>
---
#### [new 100] LivingWorld: Interactive 4D World Generation with Environmental Dynamics
- **分类: cs.CV**

- **简介: 该论文提出LivingWorld，解决4D世界生成中环境动态一致性问题。通过构建全局运动场，实现交互式、连贯的动态场景生成。**

- **链接: [https://arxiv.org/pdf/2604.01641](https://arxiv.org/pdf/2604.01641)**

> **作者:** Hyeongju Mun; In-Hwan Jin; Sohyeong Kim; Kyeongbo Kong
>
> **摘要:** We introduce LivingWorld, an interactive framework for generating 4D worlds with environmental dynamics from a single image. While recent advances in 3D scene generation enable large-scale environment creation, most approaches focus primarily on reconstructing static geometry, leaving scene-scale environmental dynamics such as clouds, water, or smoke largely unexplored. Modeling such dynamics is challenging because motion must remain coherent across an expanding scene while supporting low-latency user feedback. LivingWorld addresses this challenge by progressively constructing a globally coherent motion field as the scene expands. To maintain global consistency during expansion, we introduce a geometry-aware alignment module that resolves directional and scale ambiguities across views. We further represent motion using a compact hash-based motion field, enabling efficient querying and stable propagation of dynamics throughout the scene. This representation also supports bidirectional motion propagation during rendering, producing long and temporally coherent 4D sequences without relying on expensive video-based refinement. On a single RTX 5090 GPU, generating each new scene expansion step requires 9 seconds, followed by 3 seconds for motion alignment and motion field updates, enabling interactive 4D world generation with globally coherent environmental dynamics. Video demonstrations are available at this http URL.
>
---
#### [new 101] UAV-Track VLA: Embodied Aerial Tracking via Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机视觉跟踪任务，解决动态场景下多模态跟踪问题。提出UAV-Track VLA模型，提升跟踪性能与实时性。**

- **链接: [https://arxiv.org/pdf/2604.02241](https://arxiv.org/pdf/2604.02241)**

> **作者:** Qiyao Zhang; Shuhua Zheng; Jianli Sun; Chengxiang Li; Xianke Wu; Zihan Song; Zhiyong Cui; Yisheng Lv; Yonglin Tian
>
> **摘要:** Embodied visual tracking is crucial for Unmanned Aerial Vehicles (UAVs) executing complex real-world tasks. In dynamic urban scenarios with complex semantic requirements, Vision-Language-Action (VLA) models show great promise due to their cross-modal fusion and continuous action generation capabilities. To benchmark multimodal tracking in such environments, we construct a dedicated evaluation benchmark and a large-scale dataset encompassing over 890K frames, 176 tasks, and 85 diverse objects. Furthermore, to address temporal feature redundancy and the lack of spatial geometric priors in existing VLA models, we propose an improved VLA tracking model, UAV-Track VLA. Built upon the $\pi_{0.5}$ architecture, our model introduces a temporal compression net to efficiently capture inter-frame dynamics. Additionally, a parallel dual-branch decoder comprising a spatial-aware auxiliary grounding head and a flow matching action expert is designed to decouple cross-modal features and generate fine-grained continuous actions. Systematic experiments in the CARLA simulator validate the superior end-to-end performance of our method. Notably, in challenging long-distance pedestrian tracking tasks, UAV-Track VLA achieves a 61.76\% success rate and 269.65 average tracking frames, significantly outperforming existing baselines. Furthermore, it demonstrates robust zero-shot generalization in unseen environments and reduces single-step inference latency by 33.4\% (to 0.0571s) compared to the original $\pi_{0.5}$, enabling highly efficient, real-time UAV control. Data samples and demonstration videos are available at: this https URL\_VLA.
>
---
#### [new 102] Bias mitigation in graph diffusion models
- **分类: cs.CV**

- **简介: 该论文属于图扩散模型任务，解决模型中的逆向起始偏差和曝光偏差问题，通过设计新的采样算法和分数修正机制进行优化。**

- **链接: [https://arxiv.org/pdf/2604.01709](https://arxiv.org/pdf/2604.01709)**

> **作者:** Meng Yu; Kun Zhan
>
> **备注:** Accepted to ICLR 2025!
>
> **摘要:** Most existing graph diffusion models have significant bias problems. We observe that the forward diffusion's maximum perturbation distribution in most models deviates from the standard Gaussian distribution, while reverse sampling consistently starts from a standard Gaussian distribution, which results in a reverse-starting bias. Together with the inherent exposure bias of diffusion models, this results in degraded generation quality. This paper proposes a comprehensive approach to mitigate both biases. To mitigate reverse-starting bias, we employ a newly designed Langevin sampling algorithm to align with the forward maximum perturbation distribution, establishing a new reverse-starting point. To address the exposure bias, we introduce a score correction mechanism based on a newly defined score difference. Our approach, which requires no network modifications, is validated across multiple models, datasets, and tasks, achieving state-of-the-art this http URL is at this https URL
>
---
#### [new 103] Director: Instance-aware Gaussian Splatting for Dynamic Scene Modeling and Understanding
- **分类: cs.CV**

- **简介: 该论文属于动态场景建模任务，旨在解决传统方法在实例级结构和语义理解上的不足。通过引入统一的时空高斯表示，实现精准的4D重建与实例分割。**

- **链接: [https://arxiv.org/pdf/2604.01678](https://arxiv.org/pdf/2604.01678)**

> **作者:** Yuheng Jiang; Yiwen Cai; Zihao Wang; Yize Wu; Sicheng Li; Zhuo Su; Shaohui Jiao; Lan Xu
>
> **备注:** Project page: this https URL
>
> **摘要:** Volumetric video seeks to model dynamic scenes as temporally coherent 4D representations. While recent Gaussian-based approaches achieve impressive rendering fidelity, they primarily emphasize appearance but are largely agnostic to instance-level structure, limiting stable tracking and semantic reasoning in highly dynamic scenarios. In this paper, we present Director, a unified spatio-temporal Gaussian representation that jointly models human performance, high-fidelity rendering, and instance-level semantics. Our key insight is that embedding instance-consistent semantics naturally complements 4D modeling, enabling more accurate scene decomposition while supporting robust dynamic scene understanding. To this end, we leverage temporally aligned instance masks and sentence embeddings derived from Multimodal Large Language Models to supervise the learnable semantic features of each Gaussian via two MLP decoders, enabling language-aligned 4D representations and enforcing identity consistency over time. To enhance temporal stability, we bridge 2D optical flow with 4D Gaussians and finetune their motions, yielding reliable initialization and reducing drift. For the training, we further introduce a geometry-aware SDF constraints, along with regularization terms that enforces surface continuity, enhancing temporal coherence in dynamic foreground modeling. Experiments demonstrate that Director achieves temporally coherent 4D reconstructions while simultaneously enabling instance segmentation and open-vocabulary querying.
>
---
#### [new 104] MAVFusion: Efficient Infrared and Visible Video Fusion via Motion-Aware Sparse Interaction
- **分类: cs.CV**

- **简介: 该论文属于红外与可见光视频融合任务，解决视频帧间运动处理效率低的问题。提出MAVFusion框架，通过动态区域稀疏交互提升融合效率与质量。**

- **链接: [https://arxiv.org/pdf/2604.01958](https://arxiv.org/pdf/2604.01958)**

> **作者:** Xilai Li; Weijun Jiang; Xiaosong Li; Yang Liu; Hongbin Wang; Tao Ye; Huafeng Li; Haishu Tan
>
> **摘要:** Infrared and visible video fusion combines the object saliency from infrared images with the texture details from visible images to produce semantically rich fusion results. However, most existing methods are designed for static image fusion and cannot effectively handle frame-to-frame motion in videos. Current video fusion methods improve temporal consistency by introducing interactions across frames, but they often require high computational cost. To mitigate these challenges, we propose MAVFusion, an end-to-end video fusion framework featuring a motion-aware sparse interaction mechanism that enhances efficiency while maintaining superior fusion quality. Specifically, we leverage optical flow to identify dynamic regions in multi-modal sequences, adaptively allocating computationally intensive cross-modal attention to these sparse areas to capture salient transitions and facilitate inter-modal information exchange. For static background regions, a lightweight weak interaction module is employed to maintain structural and appearance integrity. By decoupling the processing of dynamic and static regions, MAVFusion simultaneously preserves temporal consistency and fine-grained details while significantly accelerating inference. Extensive experiments demonstrate that MAVFusion achieves state-of-the-art performance on multiple infrared and visible video benchmarks, achieving a speed of 14.16\,FPS at $640 \times 480$ resolution. The source code will be available at this https URL.
>
---
#### [new 105] Automatic Image-Level Morphological Trait Annotation for Organismal Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像特征标注任务，旨在解决生物形态特征自动标注难题。通过构建数据集和引入视觉语言提示方法，实现高效、可解释的形态描述生成。**

- **链接: [https://arxiv.org/pdf/2604.01619](https://arxiv.org/pdf/2604.01619)**

> **作者:** Vardaan Pahuja; Samuel Stevens; Alyson East; Sydne Record; Yu Su
>
> **备注:** ICLR 2026
>
> **摘要:** Morphological traits are physical characteristics of biological organisms that provide vital clues on how organisms interact with their environment. Yet extracting these traits remains a slow, expert-driven process, limiting their use in large-scale ecological studies. A major bottleneck is the absence of high-quality datasets linking biological images to trait-level annotations. In this work, we demonstrate that sparse autoencoders trained on foundation-model features yield monosemantic, spatially grounded neurons that consistently activate on meaningful morphological parts. Leveraging this property, we introduce a trait annotation pipeline that localizes salient regions and uses vision-language prompting to generate interpretable trait descriptions. Using this approach, we construct Bioscan-Traits, a dataset of 80K trait annotations spanning 19K insect images from BIOSCAN-5M. Human evaluation confirms the biological plausibility of the generated morphological descriptions. We assess design sensitivity through a comprehensive ablation study, systematically varying key design choices and measuring their impact on the quality of the resulting trait descriptions. By annotating traits with a modular pipeline rather than prohibitively expensive manual efforts, we offer a scalable way to inject biologically meaningful supervision into foundation models, enable large-scale morphological analyses, and bridge the gap between ecological relevance and machine-learning practicality.
>
---
#### [new 106] Harmonized Tabular-Image Fusion via Gradient-Aligned Alternating Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态表格-图像融合任务，解决模态间梯度冲突问题。提出GAAL方法通过梯度对齐和交替学习提升融合性能。**

- **链接: [https://arxiv.org/pdf/2604.01579](https://arxiv.org/pdf/2604.01579)**

> **作者:** Longfei Huang; Yang Yang
>
> **备注:** ICME 26
>
> **摘要:** Multimodal tabular-image fusion is an emerging task that has received increasing attention in various domains. However, existing methods may be hindered by gradient conflicts between modalities, misleading the optimization of the unimodal learner. In this paper, we propose a novel Gradient-Aligned Alternating Learning (GAAL) paradigm to address this issue by aligning modality gradients. Specifically, GAAL adopts an alternating unimodal learning and shared classifier to decouple the multimodal gradient and facilitate interaction. Furthermore, we design uncertainty-based cross-modal gradient surgery to selectively align cross-modal gradients, thereby steering the shared parameters to benefit all modalities. As a result, GAAL can provide effective unimodal assistance and help boost the overall fusion performance. Empirical experiments on widely used datasets reveal the superiority of our method through comparison with various state-of-the-art (SoTA) tabular-image fusion baselines and test-time tabular missing baselines. The source code is available at this https URL.
>
---
#### [new 107] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多智能体3D重建任务，解决分布式场景下几何不一致和通信开销问题。提出F3DGS框架，通过联邦优化实现共享几何结构的分布式重建。**

- **链接: [https://arxiv.org/pdf/2604.01605](https://arxiv.org/pdf/2604.01605)**

> **作者:** Morui Zhu; Mohammad Dehghani Tezerjani; Mátyás Szántó; Márton Vaitkus; Song Fu; Qing Yang
>
> **备注:** Accepted to the CVPR 2026 SPAR-3D Workshop
>
> **摘要:** We present F3DGS, a federated 3D Gaussian Splatting framework for decentralized multi-agent 3D reconstruction. Existing 3DGS pipelines assume centralized access to all observations, which limits their applicability in distributed robotic settings where agents operate independently, and centralized data aggregation may be restricted. Directly extending centralized training to multi-agent systems introduces communication overhead and geometric inconsistency. F3DGS first constructs a shared geometric scaffold by registering locally merged LiDAR point clouds from multiple clients to initialize a global 3DGS model. During federated optimization, Gaussian positions are fixed to preserve geometric alignment, while each client updates only appearance-related attributes, including covariance, opacity, and spherical harmonic coefficients. The server aggregates these updates using visibility-aware aggregation, weighting each client's contribution by how frequently it observed each Gaussian, resolving the partial-observability challenge inherent to multi-agent exploration. To evaluate decentralized reconstruction, we collect a multi-sequence indoor dataset with synchronized LiDAR, RGB, and IMU measurements. Experiments show that F3DGS achieves reconstruction quality comparable to centralized training while enabling distributed optimization across agents. The dataset, development kit, and source code will be publicly released.
>
---
#### [new 108] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding
- **分类: cs.CV**

- **简介: 该论文属于视频时序定位任务，解决现有模型采样不均导致关键帧丢失的问题。提出GroundVTS架构，通过查询引导的视觉令牌采样和优化策略，提升时序建模与定位精度。**

- **链接: [https://arxiv.org/pdf/2604.02093](https://arxiv.org/pdf/2604.02093)**

> **作者:** Rong Fan; Kaiyan Xiao; Minghao Zhu; Liuyi Wang; Kai Dai; Zhao Yang
>
> **备注:** Published as a conference paper at CVPR 2026
>
> **摘要:** Video temporal grounding (VTG) is a critical task in video understanding and a key capability for extending video large language models (Vid-LLMs) to broader applications. However, existing Vid-LLMs rely on uniform frame sampling to extract video information, resulting in a sparse distribution of key frames and the loss of crucial temporal cues. To address this limitation, we propose Grounded Visual Token Sampling (GroundVTS), a Vid-LLM architecture that focuses on the most informative temporal segments. GroundVTS employs a fine-grained, query-guided mechanism to filter visual tokens before feeding them into the LLM, thereby preserving essential spatio-temporal information and maintaining temporal coherence. Futhermore, we introduce a progressive optimization strategy that enables the LLM to effectively adapt to the non-uniform distribution of visual features, enhancing its ability to model temporal dependencies and achieve precise video localization. We comprehensively evaluate GroundVTS on three standard VTG benchmarks, where it outperforms existing methods, achieving a 7.7-point improvement in mIoU for moment retrieval and 12.0-point improvement in mAP for highlight detection. Code is available at this https URL.
>
---
#### [new 109] Network Structure in UK Payment Flows: Evidence on Economic Interdependencies and Implications for Real-Time Measurement
- **分类: cs.CV; econ.EM**

- **简介: 该论文属于经济监测任务，旨在通过支付网络分析揭示经济关联性。研究利用图论方法提升支付流预测精度，尤其在经济危机期间表现突出，为实时经济监控提供新思路。**

- **链接: [https://arxiv.org/pdf/2604.02068](https://arxiv.org/pdf/2604.02068)**

> **作者:** Aditya Humnabadkar
>
> **备注:** Accepted for Poster presentation at the ESCoE Conference on Economic Measurement 2026
>
> **摘要:** Network analysis of inter-industry payment flows reveals structural economic relationships invisible to traditional bilateral measurement approaches, with significant implications for real-time economic monitoring. Analysing 532,346 UK payment records (2017--2024) across 89 industry sectors, we demonstrate that graph-theoretic features which include centrality measures and clustering coefficients improve payment flow forecasting by 8.8 percentage points beyond traditional time-series methods. Critically, network features prove most valuable during economic disruptions: during the COVID-19 pandemic, when traditional forecasting accuracy collapsed (R2} falling from 0.38 to 0.19), network-enhanced models maintained substantially better performance, with network contributions reaching +13.8 percentage points. The analysis identifies Financial Services, Wholesale Trade, and Professional Services as structurally central industries whose network positions indicate systemic importance beyond their transaction volumes. Network density increased 12.5\% over the sample period, with visible disruption during 2020 followed by recovery exceeding pre-pandemic integration levels. These findings suggest payment network monitoring could enhance official statistics production by providing leading indicators of structural economic change and improving nowcasting accuracy during periods when traditional temporal patterns prove unreliable.
>
---
#### [new 110] ViTs for Action Classification in Videos: An Approach to Risky Tackle Detection in American Football Practice Videos
- **分类: cs.CV**

- **简介: 该论文属于视频动作分类任务，旨在检测美式橄榄球训练中的危险擒抱动作。通过引入扩展数据集和视觉Transformer模型，提升危险动作的识别效果。**

- **链接: [https://arxiv.org/pdf/2604.01318](https://arxiv.org/pdf/2604.01318)**

> **作者:** Syed Ahsan Masud Zaidi; William Hsu; Scott Dietrich
>
> **备注:** 15 pages, 4 figures. Accepted to ICPR 2026 (28th International Conference on Pattern Recognition)
>
> **摘要:** Early identification of hazardous actions in contact sports enables timely intervention and improves player safety. We present a method for detecting risky tackles in American football practice videos and introduce a substantially expanded dataset for this task. Our work contains 733 single-athlete-dummy tackle clips, each temporally localized around first point contact and labeled with a strike zone component of the standardized Assessment for Tackling Technique (SATT-3), extending prior work that reported 178 annotated videos. Using a Vision transformer-based model with imbalance-aware training, we obtain risky recall of 0.67 and Risky F1 of 0.59 under crossvalidation. Relative to the previous baseline in a smaller subset (risky recall of 0.58; Risky F1 0.56 ), our approach improves risky recall by more than 8% points on a much larger dataset. These results indicate that the vision transformer-based video analysis, coupled with careful handling of class imbalance, can reliably detect rare but safety-critical tackling patterns, offering a practical pathway toward coach-centered injury prevention tools.
>
---
#### [new 111] PLUME: Latent Reasoning Based Universal Multimodal Embedding
- **分类: cs.CV**

- **简介: 该论文提出PLUME，解决通用多模态嵌入（UME）中的推理效率问题，通过隐式推理替代显式思维链，提升检索性能与速度。**

- **链接: [https://arxiv.org/pdf/2604.02073](https://arxiv.org/pdf/2604.02073)**

> **作者:** Chenwei He; Xiangzhao Hao; Tianyu Yang; Yuxiang Ma; Yuheng Jia; Lingxiang Wu; Chaoyang Zhao; Haiyun Guo; Jinqiao Wang
>
> **摘要:** Universal multimodal embedding (UME) maps heterogeneous inputs into a shared retrieval space with a single model. Recent approaches improve UME by generating explicit chain-of-thought (CoT) rationales before extracting embeddings, enabling multimodal large language models to better infer complex query intent. However, explicit CoT incurs substantial inference overhead and can compress rich multimodal evidence into a narrow textual bottleneck. We propose PLUME, a latent reasoning framework that advances UME by replacing verbalized CoT with a short autoregressive rollout of continuous latent states. To support diverse multimodal queries, PLUME further introduces a semantic-anchor-guided transition adapter that steers latent rollout along different reasoning trajectories under the same fixed computation budget. To stabilize training, PLUME adopts a progressive explicit-to-latent curriculum that uses verbalized reasoning only as a temporary training scaffold and gradually transfers this behavior into hidden-state computation, eliminating explicit CoT at inference. On the 78-task MMEB-v2 benchmark, PLUME outperforms strong explicit-CoT UME baselines while reducing reasoning from hundreds of generated tokens to fewer than 10 latent steps, delivering over 30x faster inference. PLUME is especially well suited to retrieval settings where relevant evidence is dense, structurally complex, and difficult to organize through verbalized intermediate rationales, such as video and visual document retrieval. These results show that structured latent computation can preserve the benefits of intermediate reasoning without the overhead of explicit rationale generation, providing a stronger and more efficient paradigm for practical retrieval systems.
>
---
#### [new 112] DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Planning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DriveDreamer-Policy，属于驾驶场景下的世界-动作建模任务，旨在提升物理环境中的几何感知与规划能力。通过整合深度生成、视频预测和路径规划，增强驾驶决策的准确性与连贯性。**

- **链接: [https://arxiv.org/pdf/2604.01765](https://arxiv.org/pdf/2604.01765)**

> **作者:** Yang Zhou; Xiaofeng Wang; Hao Shao; Letian Wang; Guosheng Zhao; Jiangnan Shao; Jiagang Zhu; Tingdong Yu; Zheng Zhu; Guan Huang; Steven L. Waslander
>
> **备注:** 11 pages, 4 figures; Project Website: this https URL
>
> **摘要:** Recently, world-action models (WAM) have emerged to bridge vision-language-action (VLA) models and world models, unifying their reasoning and instruction-following capabilities and spatio-temporal world modeling. However, existing WAM approaches often focus on modeling 2D appearance or latent representations, with limited geometric grounding-an essential element for embodied systems operating in the physical world. We present DriveDreamer-Policy, a unified driving world-action model that integrates depth generation, future video generation, and motion planning within a single modular architecture. The model employs a large language model to process language instructions, multi-view images, and actions, followed by three lightweight generators that produce depth, future video, and actions. By learning a geometry-aware world representation and using it to guide both future prediction and planning within a unified framework, the proposed model produces more coherent imagined futures and more informed driving actions, while maintaining modularity and controllable latency. Experiments on the Navsim v1 and v2 benchmarks demonstrate that DriveDreamer-Policy achieves strong performance on both closed-loop planning and world generation tasks. In particular, our model reaches 89.2 PDMS on Navsim v1 and 88.7 EPDMS on Navsim v2, outperforming existing world-model-based approaches while producing higher-quality future video and depth predictions. Ablation studies further show that explicit depth learning provides complementary benefits to video imagination and improves planning robustness.
>
---
#### [new 113] Jagle: Building a Large-Scale Japanese Multimodal Post-Training Dataset for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出Jagle，一个大规模日语多模态数据集，解决非英语视觉-语言模型训练数据不足的问题。通过多种策略生成VQA对，提升日语任务性能。**

- **链接: [https://arxiv.org/pdf/2604.02048](https://arxiv.org/pdf/2604.02048)**

> **作者:** Issa Sugiura; Keito Sasagawa; Keisuke Nakao; Koki Maeda; Ziqi Yin; Zhishen Yang; Shuhei Kurita; Yusuke Oda; Ryoko Tokuhisa; Daisuke Kawahara; Naoaki Okazaki
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Developing vision-language models (VLMs) that generalize across diverse tasks requires large-scale training datasets with diverse content. In English, such datasets are typically constructed by aggregating and curating numerous existing visual question answering (VQA) resources. However, this strategy does not readily extend to other languages, where VQA datasets remain limited in both scale and domain coverage, posing a major obstacle to building high-quality multilingual and non-English VLMs. In this work, we introduce Jagle, the largest Japanese multimodal post-training dataset to date, comprising approximately 9.2 million instances across diverse tasks. Rather than relying on existing VQA datasets, we collect heterogeneous source data, including images, image-text pairs, and PDF documents, and generate VQA pairs through multiple strategies such as VLM-based QA generation, translation, and text rendering. Experiments demonstrate that a 2.2B model trained with Jagle achieves strong performance on Japanese tasks, surpassing InternVL3.5-2B in average score across ten Japanese evaluation tasks and approaching within five points of Qwen3-VL-2B-Instruct. Furthermore, combining Jagle with FineVision does not degrade English performance; instead, it improves English performance compared to training with FineVision alone. To facilitate reproducibility and future research, we release the dataset, trained models, and code.
>
---
#### [new 114] Regularizing Attention Scores with Bootstrapping
- **分类: cs.CV; cs.AI; cs.LG; stat.ME; stat.ML**

- **简介: 该论文属于视觉Transformer的可解释性任务，旨在解决注意力分数噪声多、可解释性差的问题。通过引入自助法（bootstrapping）生成基准分布，量化注意力分数的显著性，实现注意力图的正则化。**

- **链接: [https://arxiv.org/pdf/2604.01339](https://arxiv.org/pdf/2604.01339)**

> **作者:** Neo Christopher Chung; Maxim Laletin
>
> **摘要:** Vision transformers (ViT) rely on attention mechanism to weigh input features, and therefore attention scores have naturally been considered as explanations for its decision-making process. However, attention scores are almost always non-zero, resulting in noisy and diffused attention maps and limiting interpretability. Can we quantify uncertainty measures of attention scores and obtain regularized attention scores? To this end, we consider attention scores of ViT in a statistical framework where independent noise would lead to insignificant yet non-zero scores. Leveraging statistical learning techniques, we introduce the bootstrapping for attention scores which generates a baseline distribution of attention scores by resampling input features. Such a bootstrap distribution is then used to estimate significances and posterior probabilities of attention scores. In natural and medical images, the proposed \emph{Attention Regularization} approach demonstrates a straightforward removal of spurious attention arising from noise, drastically improving shrinkage and sparsity. Quantitative evaluations are conducted using both simulation and real-world datasets. Our study highlights bootstrapping as a practical regularization tool when using attention scores as explanations for ViT. Code available: this https URL
>
---
#### [new 115] SDesc3D: Towards Layout-Aware 3D Indoor Scene Generation from Short Descriptions
- **分类: cs.CV**

- **简介: 该论文属于3D室内场景生成任务，旨在解决短文本描述下场景物理合理性与细节不足的问题。提出SDesc3D框架，结合多视角结构先验和功能区域推理，提升场景布局的准确性与丰富性。**

- **链接: [https://arxiv.org/pdf/2604.01972](https://arxiv.org/pdf/2604.01972)**

> **作者:** Jie Feng; Jiawei Shen; Junjia Huang; Junpeng Zhang; Mingtao Feng; Weisheng Dong; Guanbin Li
>
> **摘要:** 3D indoor scene generation conditioned on short textual descriptions provides a promising avenue for interactive 3D environment construction without the need for labor-intensive layout specification. Despite recent progress in text-conditioned 3D scene generation, existing works suffer from poor physical plausibility and insufficient detail richness in such semantic condensation cases, largely due to their reliance on explicit semantic cues about compositional objects and their spatial relationships. This limitation highlights the need for enhanced 3D reasoning capabilities, particularly in terms of prior integration and spatial this http URL by this, we propose SDesc3D, a short-text conditioned 3D indoor scene generation framework, that leverages multi-view structural priors and regional functionality implications to enable 3D layout reasoning under sparse textual this http URL, we introduce a Multi-view scene prior augmentation that enriches underspecified textual inputs with aggregated multi-view structural knowledge, shifting from inaccessible semantic relation cues to multi-view relational prior aggregation. Building on this, we design a Functionality-aware layout grounding, employing regional functionality grounding for implicit spatial anchors and conducting hierarchical layout reasoning to enhance scene organization and semantic this http URL, an Iterative reflection-rectification scheme is employed for progressive structural plausibility refinement via this http URL experiments show that our method outperforms existing approaches on short-text conditioned 3D indoor scene this http URL will be publicly available.
>
---
#### [new 116] UniRecGen: Unifying Multi-View 3D Reconstruction and Generation
- **分类: cs.CV**

- **简介: 该论文提出UniRecGen，解决稀疏视角下3D重建与生成的矛盾问题。整合重建与生成模型，提升模型完整性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.01479](https://arxiv.org/pdf/2604.01479)**

> **作者:** Zhisheng Huang; Jiahao Chen; Cheng Lin; Chenyu Hu; Hanzhuo Huang; Zhengming Yu; Mengfei Li; Yuheng Liu; Zekai Gu; Zibo Zhao; Yuan Liu; Xin Li; Wenping Wang
>
> **摘要:** Sparse-view 3D modeling represents a fundamental tension between reconstruction fidelity and generative plausibility. While feed-forward reconstruction excels in efficiency and input alignment, it often lacks the global priors needed for structural completeness. Conversely, diffusion-based generation provides rich geometric details but struggles with multi-view consistency. We present UniRecGen, a unified framework that integrates these two paradigms into a single cooperative system. To overcome inherent conflicts in coordinate spaces, 3D representations, and training objectives, we align both models within a shared canonical space. We employ disentangled cooperative learning, which maintains stable training while enabling seamless collaboration during inference. Specifically, the reconstruction module is adapted to provide canonical geometric anchors, while the diffusion generator leverages latent-augmented conditioning to refine and complete the geometric structure. Experimental results demonstrate that UniRecGen achieves superior fidelity and robustness, outperforming existing methods in creating complete and consistent 3D models from sparse observations.
>
---
#### [new 117] NearID: Identity Representation Learning via Near-identity Distractors
- **分类: cs.CV**

- **简介: 该论文属于身份表示学习任务，解决视觉编码器混淆身份与背景的问题。通过引入NearID数据集和对比学习方法，提升身份区分能力。**

- **链接: [https://arxiv.org/pdf/2604.01973](https://arxiv.org/pdf/2604.01973)**

> **作者:** Aleksandar Cvejic; Rameen Abdal; Abdelrahman Eldesokey; Bernard Ghanem; Peter Wonka
>
> **备注:** Code at this https URL
>
> **摘要:** When evaluating identity-focused tasks such as personalized generation and image editing, existing vision encoders entangle object identity with background context, leading to unreliable representations and metrics. We introduce the first principled framework to address this vulnerability using Near-identity (NearID) distractors, where semantically similar but distinct instances are placed on the exact same background as a reference image, eliminating contextual shortcuts and isolating identity as the sole discriminative signal. Based on this principle, we present the NearID dataset (19K identities, 316K matched-context distractors) together with a strict margin-based evaluation protocol. Under this setting, pre-trained encoders perform poorly, achieving Sample Success Rates (SSR), a strict margin-based identity discrimination metric, as low as 30.7% and often ranking distractors above true cross-view matches. We address this by learning identity-aware representations on a frozen backbone using a two-tier contrastive objective enforcing the hierarchy: same identity > NearID distractor > random negative. This improves SSR to 99.2%, enhances part-level discrimination by 28.0%, and yields stronger alignment with human judgments on DreamBench++, a human-aligned benchmark for personalization. Project page: this https URL
>
---
#### [new 118] Enhancing Medical Visual Grounding via Knowledge-guided Spatial Prompts
- **分类: cs.CV**

- **简介: 该论文属于医学视觉定位任务，旨在提升模型对医学影像中相关区域的精确定位。针对现有模型空间精度不足的问题，提出KnowMVG框架，结合知识引导提示和全局-局部注意力机制，增强空间感知能力。**

- **链接: [https://arxiv.org/pdf/2604.01915](https://arxiv.org/pdf/2604.01915)**

> **作者:** Yifan Gao; Tao Zhou; Yi Zhou; Ke Zou; Yizhe Zhang; Huazhu Fu
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Medical Visual Grounding (MVG) aims to identify diagnostically relevant phrases from free-text radiology reports and localize their corresponding regions in medical images, providing interpretable visual evidence to support clinical decision-making. Although recent Vision-Language Models (VLMs) exhibit promising multimodal reasoning ability, their grounding remains insufficient spatial precision, largely due to a lack of explicit localization priors when relying solely on latent embeddings. In this work, we analyze this limitation from an attention perspective and propose KnowMVG, a Knowledge-prior and global-local attention enhancement framework for MVG in VLMs that explicitly strengthens spatial awareness during decoding. Specifically, we present a knowledge-enhanced prompting strategy that encodes phrase related medical knowledge into compact embeddings, together with a global-local attention that jointly leverages coarse global information and refined local cues to guide precise region localization. localization. This design bridges high-level semantic understanding and fine-grained visual perception without introducing extra textual reasoning overhead. Extensive experiments on four MVG benchmarks demonstrate that our KnowMVG consistently outperforms existing approaches, achieving gains of 3.0% in AP50 and 2.6% in mIoU over prior state-of-the-art methods. Qualitative and ablation studies further validate the effectiveness of each component.
>
---
#### [new 119] Dense Point-to-Mask Optimization with Reinforced Point Selection for Crowd Instance Segmentation
- **分类: cs.CV**

- **简介: 该论文针对人群实例分割任务，解决密集人群下点标注转分割的问题。提出DPMO和RPS框架，提升分割与计数精度。**

- **链接: [https://arxiv.org/pdf/2604.01742](https://arxiv.org/pdf/2604.01742)**

> **作者:** Hongru Chen; Jiyang Huang; Jia Wan; Antoni B.Chan
>
> **摘要:** Crowd instance segmentation is a crucial task with a wide range of applications, including surveillance and transportation. Currently, point labels are common in crowd datasets, while region labels (e.g., boxes) are rare and inaccurate. The masks obtained through segmentation help to improve the accuracy of region labels and resolve the correspondence between individual location coordinates and crowd density maps. However, directly applying currently popular large foundation models such as SAM does not yield ideal results in dense crowds. To this end, we first propose Dense Point-to-Mask Optimization (DPMO), which integrates SAM with the Nearest Neighbor Exclusive Circle (NNEC) constraint to generate dense instance segmentation from point annotations. With DPMO and manual correction, we obtain mask annotations from the existing point annotations for traditional crowd datasets. Then, to predict instance segmentation in dense crowds, we propose a Reinforced Point Selection (RPS) framework trained with Group Relative Policy Optimization (GRPO), which selects the best predicted point from a sampling of the initial point prediction. Through extensive experiments, we achieve state-of-the-art crowd instance segmentation performance on ShanghaiTech, UCF-QNRF, JHU-CROWD++, and NWPU-Crowd datasets. Furthermore, we design new loss functions supervised by masks that boost counting performance across different models, demonstrating the significant role of mask annotations in enhancing counting accuracy.
>
---
#### [new 120] MonoSAOD: Monocular 3D Object Detection with Sparsely Annotated Label
- **分类: cs.CV**

- **简介: 该论文属于单目3D目标检测任务，解决稀疏标注下的检测难题。提出RAPA和PBF模块，结合几何保持增强与原型引导伪标签，提升稀疏监督下的检测性能。**

- **链接: [https://arxiv.org/pdf/2604.01646](https://arxiv.org/pdf/2604.01646)**

> **作者:** Junyoung Jung; Seokwon Kim; Jun Uk Kim
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Monocular 3D object detection has achieved impressive performance on densely annotated datasets. However, it struggles when only a fraction of objects are labeled due to the high cost of 3D annotation. This sparsely annotated setting is common in real-world scenarios where annotating every object is impractical. To address this, we propose a novel framework for sparsely annotated monocular 3D object detection with two key modules. First, we propose Road-Aware Patch Augmentation (RAPA), which leverages sparse annotations by augmenting segmented object patches onto road regions while preserving 3D geometric consistency. Second, we propose Prototype-Based Filtering (PBF), which generates high-quality pseudo-labels by filtering predictions through prototype similarity and depth uncertainty. It maintains global 2D RoI feature prototypes and selects pseudo-labels that are both feature-consistent with learned prototypes and have reliable depth estimates. Our training strategy combines geometry-preserving augmentation with prototype-guided pseudo-labeling to achieve robust detection under sparse supervision. Extensive experiments demonstrate the effectiveness of the proposed method. The source code is available at this https URL .
>
---
#### [new 121] Automated Prostate Gland Segmentation in MRI Using nnU-Net
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决前列腺自动分割问题。针对手动分割耗时且不一致的问题，提出基于nnU-Net的多模态方法，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.01964](https://arxiv.org/pdf/2604.01964)**

> **作者:** Pablo Rodriguez-Belenguer; Gloria Ribas; Javier Aquerreta Escribano; Rafael Moreno-Calatayud; Leonor Cerda-Alberich; Luis Marti-Bonmati
>
> **备注:** 9 pages, 2 tables, 1 figure
>
> **摘要:** Accurate segmentation of the prostate gland in multiparametric MRI (mpMRI) is a fundamental step for a wide range of clinical and research applications, including image registration, volume estimation, and radiomic analysis. However, manual delineation is time-consuming and subject to inter-observer variability, while general-purpose segmentation tools often fail to provide sufficient accuracy for prostate-specific tasks. In this work, we propose a dedicated deep learning-based approach for automatic prostate gland segmentation using the nnU-Net v2 framework. The model leverages multimodal mpMRI data, including T2-weighted imaging, diffusion-weighted imaging (DWI), and apparent diffusion coefficient (ADC) maps, to exploit complementary tissue information. Training was performed on 981 cases from the PI-CAI dataset using whole-gland annotations, and model performance was assessed through 5-fold cross-validation and external validation on an independent cohort of 54 patients from Hospital La Fe. The proposed model achieved a mean Dice score of 0.96 +/- 0.00 in cross-validation and 0.82 on the external test set, demonstrating strong generalization despite domain shift. In comparison, a general-purpose approach (TotalSegmentator) showed substantially lower performance, with a Dice score of 0.15, primarily due to under-segmentation of the gland. These results highlight the importance of task-specific, multimodal segmentation strategies and demonstrate the potential of the proposed approach for reliable integration into clinical research workflows. To facilitate reproducibility and deployment, the model has been fully containerized and is available as a ready-to-use inference tool.
>
---
#### [new 122] SCALE: Semantic- and Confidence-Aware Conditional Variational Autoencoder for Zero-shot Skeleton-based Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于零样本骨架动作识别任务，解决未见类别识别问题。提出SCALE框架，通过语义和置信度感知的条件变分自编码器，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2604.02222](https://arxiv.org/pdf/2604.02222)**

> **作者:** Soroush Oraki; Feng Ding; Jie Liang
>
> **备注:** Accepted to ICPR 2026
>
> **摘要:** Zero-shot skeleton-based action recognition (ZSAR) aims to recognize action classes without any training skeletons from those classes, relying instead on auxiliary semantics from text. Existing approaches frequently depend on explicit skeleton-text alignment, which can be brittle when action names underspecify fine-grained dynamics and when unseen classes are semantically confusable. We propose SCALE, a lightweight and deterministic Semantic- and Confidence-Aware Listwise Energy-based framework that formulates ZSAR as class-conditional energy ranking. SCALE builds a text-conditioned Conditional Variational Autoencoder where frozen text representations parameterize both the latent prior and the decoder, enabling likelihood-based evaluation for unseen classes without generating samples at test time. To separate competing hypotheses, we introduce a semantic- and confidence-aware listwise energy loss that emphasizes semantically similar hard negatives and incorporates posterior uncertainty to adapt decision margins and reweight ambiguous training instances. Additionally, we utilize a latent prototype contrast objective to align posterior means with text-derived latent prototypes, improving semantic organization and class separability without direct feature matching. Experiments on NTU-60 and NTU-120 datasets show that SCALE consistently improves over prior VAE- and alignment-based baselines while remaining competitive with diffusion-based methods.
>
---
#### [new 123] DOne: Decoupling Structure and Rendering for High-Fidelity Design-to-Code Generation
- **分类: cs.CV; cs.SE**

- **简介: 该论文属于设计到代码生成任务，解决结构与细节不匹配导致的布局错误问题。提出DOne框架，通过解耦结构与渲染提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.01226](https://arxiv.org/pdf/2604.01226)**

> **作者:** Xinhao Huang; Jinke Yu; Wenhao Xu; Zeyi Wen; Ying Zhou; Junzhuo Liu; Junhao Ji; Zulong Chen
>
> **摘要:** While Vision Language Models (VLMs) have shown promise in Design-to-Code generation, they suffer from a "holistic bottleneck-failing to reconcile high-level structural hierarchy with fine-grained visual details, often resulting in layout distortions or generic placeholders. To bridge this gap, we propose DOne, an end-to-end framework that decouples structure understanding from element rendering. DOne introduces (1) a learned layout segmentation module to decompose complex designs, avoiding the limitations of heuristic cropping; (2) a specialized hybrid element retriever to handle the extreme aspect ratios and densities of UI components; and (3) a schema-guided generation paradigm that bridges layout and code. To rigorously assess performance, we introduce HiFi2Code, a benchmark featuring significantly higher layout complexity than existing datasets. Extensive evaluations on the HiFi2Code demonstrate that DOne outperforms exiting methods in both high-level visual similarity (e.g., over 10% in GPT Score) and fine-grained element alignment. Human evaluations confirm a 3 times productivity gain with higher visual fidelity.
>
---
#### [new 124] Steerable Visual Representations
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Steerable Visual Representations，解决视觉表示难以通过语言引导的问题。通过早期融合文本与视觉特征，实现对视觉特征的精准控制，提升任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.02327](https://arxiv.org/pdf/2604.02327)**

> **作者:** Jona Ruthardt; Manu Gaur; Deva Ramanan; Makarand Tapaswi; Yuki M. Asano
>
> **备注:** preprint
>
> **摘要:** Pretrained Vision Transformers (ViTs) such as DINOv2 and MAE provide generic image features that can be applied to a variety of downstream tasks such as retrieval, classification, and segmentation. However, such representations tend to focus on the most salient visual cues in the image, with no way to direct them toward less prominent concepts of interest. In contrast, Multimodal LLMs can be guided with textual prompts, but the resulting representations tend to be language-centric and lose their effectiveness for generic visual tasks. To address this, we introduce Steerable Visual Representations, a new class of visual representations, whose global and local features can be steered with natural language. While most vision-language models (e.g., CLIP) fuse text with visual features after encoding (late fusion), we inject text directly into the layers of the visual encoder (early fusion) via lightweight cross-attention. We introduce benchmarks for measuring representational steerability, and demonstrate that our steerable visual features can focus on any desired objects in an image while preserving the underlying representation quality. Our method also matches or outperforms dedicated approaches on anomaly detection and personalized object discrimination, exhibiting zero-shot generalization to out-of-distribution tasks.
>
---
#### [new 125] BTS-rPPG: Orthogonal Butterfly Temporal Shifting for Remote Photoplethysmography
- **分类: cs.CV**

- **简介: 该论文属于远程光电容积描记（rPPG）任务，旨在解决时间动态建模不足的问题。提出BTS-rPPG框架，通过正交蝴蝶时间转移实现更长的时域建模和高效信息传播。**

- **链接: [https://arxiv.org/pdf/2604.01679](https://arxiv.org/pdf/2604.01679)**

> **作者:** Ba-Thinh Nguyen; Thi-Duyen Ngo; Thanh-Trung Huynh; Thanh-Ha Le; Huy-Hieu Pham
>
> **摘要:** Remote photoplethysmography (rPPG) enables contactless physiological sensing from facial videos by analyzing subtle appearance variations induced by blood circulation. However, modeling the temporal dynamics of these signals remains challenging, as many deep learning methods rely on temporal shifting or convolutional operators that aggregate information primarily from neighboring frames, resulting in predominantly local temporal modeling and limited temporal receptive fields. To address this limitation, we propose BTS-rPPG, a temporal modeling framework based on Orthogonal Butterfly Temporal Shifting (BTS). Inspired by the butterfly communication pattern in the Fast Fourier Transform (FFT), BTS establishes structured frame interactions via an XOR-based butterfly pairing schedule, progressively expanding the temporal receptive field and enabling efficient propagation of information across distant frames. Furthermore, we introduce an orthogonal feature transfer mechanism (OFT) that filters the source feature with respect to the target context before temporal shifting, retaining only the orthogonal component for cross-frame transmission. This reduces redundant feature propagation and encourages complementary temporal interaction. Extensive experiments on multiple benchmark datasets demonstrate that BTS-rPPG improves long-range temporal modeling of physiological dynamics and consistently outperforms existing temporal modeling strategies for rPPG estimation.
>
---
#### [new 126] Efficient Reasoning via Thought Compression for Language Segmentation
- **分类: cs.CV**

- **简介: 该论文属于语言分割任务，旨在解决链式推理计算成本高的问题。通过引入WISE框架，生成简洁推理过程，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2604.02040](https://arxiv.org/pdf/2604.02040)**

> **作者:** Qing Zhou; Shiyu Zhang; Yuyu Jia; Junyu Gao; Weiping Ni; Junzheng Wu; Qi Wang
>
> **摘要:** Chain-of-thought (CoT) reasoning has significantly improved the performance of large multimodal models in language-guided segmentation, yet its prohibitive computational cost, stemming from generating verbose rationales, limits real-world applicability. We introduce WISE (Wisdom from Internal Self-Exploration), a novel paradigm for efficient reasoning guided by the principle of \textit{thinking twice -- once for learning, once for speed}. WISE trains a model to generate a structured sequence: a concise rationale, the final answer, and then a detailed explanation. By placing the concise rationale first, our method leverages autoregressive conditioning to enforce that the concise rationale acts as a sufficient summary for generating the detailed explanation. This structure is reinforced by a self-distillation objective that jointly rewards semantic fidelity and conciseness, compelling the model to internalize its detailed reasoning into a compact form. At inference, the detailed explanation is omitted. To address the resulting conditional distribution shift, our inference strategy, WISE-S, employs a simple prompting technique that injects a brevity-focused instruction into the user's query. This final adjustment facilitates the robust activation of the learned concise policy, unlocking the full benefits of our framework. Extensive experiments show that WISE-S achieves state-of-the-art zero-shot performance on the ReasonSeg benchmark with 58.3 cIoU, while reducing the average reasoning length by nearly \textbf{5$\times$} -- from 112 to just 23 tokens. Code is available at \href{this https URL}{WISE}.
>
---
#### [new 127] Curia-2: Scaling Self-Supervised Learning for Radiology Foundation Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Curia-2，优化放射学数据的自监督学习，提升模型性能。属于医学影像分析任务，解决放射科医生工作负担重的问题。通过改进预训练策略和模型架构实现性能提升。**

- **链接: [https://arxiv.org/pdf/2604.01987](https://arxiv.org/pdf/2604.01987)**

> **作者:** Antoine Saporta; Baptiste Callard; Corentin Dancette; Julien Khlaut; Charles Corbière; Leo Butsanets; Amaury Prat; Pierre Manceron
>
> **摘要:** The rapid growth of medical imaging has fueled the development of Foundation Models (FMs) to reduce the growing, unsustainable workload on radiologists. While recent FMs have shown the power of large-scale pre-training to CT and MRI analysis, there remains significant room to optimize how these models learn from complex radiological volumes. Building upon the Curia framework, this work introduces Curia-2, which significantly improves the original pre-training strategy and representation quality to better capture the specificities of radiological data. The proposed methodology enables scaling the architecture up to billion-parameter Vision Transformers, marking a first for multi-modal CT and MRI FMs. Furthermore, we formalize the evaluation of these models by extending and restructuring CuriaBench into two distinct tracks: a 2D track tailored for slice-based vision models and a 3D track for volumetric benchmarking. Our results demonstrate that Curia-2 outperforms all FMs on vision-focused tasks and fairs competitively to vision-language models on clinically complex tasks such as finding detection. Weights will be made publicly available to foster further research.
>
---
#### [new 128] HOT: Harmonic-Constrained Optimal Transport for Remote Photoplethysmography Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于rPPG领域适应任务，解决域移导致的性能下降问题。通过频率域适应和HOT方法，提升模型对光照等外观变化的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.01675](https://arxiv.org/pdf/2604.01675)**

> **作者:** Ba-Thinh Nguyen; Thi-Duyen Ngo; Thanh-Trung Huynh; Thanh-Ha Le; Huy-Hieu Pham
>
> **摘要:** Remote photoplethysmography (rPPG) enables non-contact physiological measurement from facial videos; however, its practical deployment is often hindered by substantial performance degradation under domain shift. While recent deep learning-based rPPG methods have achieved strong performance on individual datasets, they frequently overfit to appearance-related factors, such as illumination, camera characteristics, and color response, that vary significantly across domains. To address this limitation, we introduce frequency domain adaptation (FDA) as a principled strategy for modeling appearance variation in rPPG. By transferring low-frequency spectral components that encode domain-dependent appearance characteristics, FDA encourages rPPG models to learn invariance to appearance variations while retaining cardiac-induced signals. To further support physiologically consistent alignment under such appearance variation, we propose Harmonic-Constrained Optimal Transport (HOT), which leverages the harmonic property of cardiac signals to guide alignment between original and FDA-transferred representations. Extensive cross-dataset experiments demonstrate that the proposed FDA and HOT framework effectively enhances the robustness and generalization of rPPG models across diverse datasets.
>
---
#### [new 129] Reinforcing Consistency in Video MLLMs with Structured Rewards
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决MLLM生成结果缺乏视觉和时间一致性的问题。通过引入结构化奖励机制，提升模型对事实和时序的准确把握。**

- **链接: [https://arxiv.org/pdf/2604.01460](https://arxiv.org/pdf/2604.01460)**

> **作者:** Yihao Quan; Zeru Shi; Jinman Zhao; Ruixiang Tang
>
> **摘要:** Multimodal large language models (MLLMs) have achieved remarkable progress in video understanding. However, seemingly plausible outputs often suffer from poor visual and temporal grounding: a model may fabricate object existence, assign incorrect attributes, or collapse repeated events while still producing a globally reasonable caption or answer. We study this failure mode through a compositional consistency audit that decomposes a caption into supporting factual and temporal claims, investigating whether a correct high-level prediction is actually backed by valid lower-level evidence. Our top-down audit reveals that even correct root relational claims often lack reliable attribute and existence support. This indicates that standard sentence-level supervision is a weak proxy for faithful video understanding. Furthermore, when turning to reinforcement learning (RL) for better alignment, standard sentence-level rewards often prove too coarse to accurately localize specific grounding failures. To address this, we replace generic sentence-level rewards with a structured reward built from factual and temporal units. Our training objective integrates three complementary components: (1) an instance-aware scene-graph reward for factual objects, attributes, and relations; (2) a temporal reward for event ordering and repetition; and (3) a video-grounded VQA reward for hierarchical self-verification. Across temporal, general video understanding, and hallucination-oriented benchmarks, this objective yields consistent gains on open-source backbones. These results suggest that structured reward shaping is a practical route to more faithful video understanding.
>
---
#### [new 130] Decouple and Rectify: Semantics-Preserving Structural Enhancement for Open-Vocabulary Remote Sensing Segmentation
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分割任务，旨在解决开放词汇语义分割中语义与结构细节不匹配的问题。通过解耦CLIP特征并引入DINO增强结构，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.02010](https://arxiv.org/pdf/2604.02010)**

> **作者:** Jie Feng; Fengze Li; Junpeng Zhang; Siyu Chen; Yuping Liang; Junying Chen; Ronghua Shang
>
> **摘要:** Open-vocabulary semantic segmentation in the remote sensing (RS) field requires both language-aligned recognition and fine-grained spatial delineation. Although CLIP offers robust semantic generalization, its global-aligned visual representations inherently struggle to capture structural details. Recent methods attempt to compensate for this by introducing RS-pretrained DINO features. However, these methods treat CLIP representations as a monolithic semantic space and cannot localize where structural enhancement is required, failing to effectively delineate boundaries while risking the disruption of CLIP's semantic integrity. To address this limitation, we propose DR-Seg, a novel decouple-and-rectify framework in this paper. Our method is motivated by the key observation that CLIP feature channels exhibit distinct functional heterogeneity rather than forming a uniform semantic space. Building on this insight, DR-Seg decouples CLIP features into semantics-dominated and structure-dominated subspaces, enabling targeted structural enhancement by DINO without distorting language-aligned semantics. Subsequently, a prior-driven graph rectification module injects high-fidelity structural priors under DINO guidance to form a refined branch, while an uncertainty-guided adaptive fusion module dynamically integrates this refined branch with the original CLIP branch for final prediction. Comprehensive experiments across eight benchmarks demonstrate that DR-Seg establishes a new state-of-the-art.
>
---
#### [new 131] STRIVE: Structured Spatiotemporal Exploration for Reinforcement Learning in Video Question Answering
- **分类: cs.CV**

- **简介: 该论文提出STRIVE框架，解决视频问答中的奖励方差低问题。通过构建时空变体和重要性采样，提升多模态强化学习的稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2604.01824](https://arxiv.org/pdf/2604.01824)**

> **作者:** Emad Bahrami; Olga Zatsarynna; Parth Pathak; Sunando Sengupta; Juergen Gall; Mohsen Fayyaz
>
> **摘要:** We introduce STRIVE (SpatioTemporal Reinforcement with Importance-aware Variant Exploration), a structured reinforcement learning framework for video question answering. While group-based policy optimization methods have shown promise in large multimodal models, they often suffer from low reward variance when responses exhibit similar correctness, leading to weak or unstable advantage estimates. STRIVE addresses this limitation by constructing multiple spatiotemporal variants of each input video and performing joint normalization across both textual generations and visual variants. By expanding group comparisons beyond linguistic diversity to structured visual perturbations, STRIVE enriches reward signals and promotes more stable and informative policy updates. To ensure exploration remains semantically grounded, we introduce an importance-aware sampling mechanism that prioritizes frames most relevant to the input question while preserving temporal coverage. This design encourages robust reasoning across complementary visual perspectives rather than overfitting to a single spatiotemporal configuration. Experiments on six challenging video reasoning benchmarks including VideoMME, TempCompass, VideoMMMU, MMVU, VSI-Bench, and PerceptionTest demonstrate consistent improvements over strong reinforcement learning baselines across multiple large multimodal models. Our results highlight the role of structured spatiotemporal exploration as a principled mechanism for stabilizing multimodal reinforcement learning and improving video reasoning performance.
>
---
#### [new 132] A3R: Agentic Affordance Reasoning via Cross-Dimensional Evidence in 3D Gaussian Scenes
- **分类: cs.CV**

- **简介: 该论文属于3D场景中的具身推理任务，解决复杂环境中因证据不足导致的推理失败问题。提出A3R框架，通过跨维度证据逐步提升推理精度。**

- **链接: [https://arxiv.org/pdf/2604.01882](https://arxiv.org/pdf/2604.01882)**

> **作者:** Di Li; Jie Feng; Guanbin Li; Ronghua Shang; Yuhui Zheng; Weisheng Dong; Guangming Shi
>
> **摘要:** Affordance reasoning in 3D Gaussian scenes aims to identify the region that supports the action specified by a given text instruction in complex environments. Existing methods typically cast this problem as one-shot prediction from static scene observations, assuming sufficient evidence is already available for reasoning. However, in complex 3D scenes, many failure cases arise not from weak prediction capacity, but from incomplete task-relevant evidence under fixed observations. To address this limitation, we reformulate fine-grained affordance reasoning as a sequential evidence acquisition process, where ambiguity is progressively reduced through complementary 3D geometric and 2D semantic evidence. Building on this formulation, we propose A3R, an agentic affordance reasoning framework that enables an MLLM-based policy to iteratively select evidence acquisition actions and update the affordance belief through cross-dimensional evidence acquisition. To optimize such sequential decision making, we further introduce a GRPO-based policy learning strategy that improves evidence acquisition efficiency and reasoning accuracy. Extensive experiments on scene-level benchmarks show that A3R consistently surpasses static one-shot baselines, demonstrating the advantage of agentic cross-dimensional evidence acquisition for fine-grained affordance reasoning in complex 3D Gaussian scenes.
>
---
#### [new 133] Ego-Grounding for Personalized Question-Answering in Egocentric Videos
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于个性化问答任务，旨在解决egocentric视频中ego-grounding问题。提出MyEgo数据集，评估模型对“我”的理解与记忆能力，发现现有模型表现不佳，强调长时记忆的重要性。**

- **链接: [https://arxiv.org/pdf/2604.01966](https://arxiv.org/pdf/2604.01966)**

> **作者:** Junbin Xiao; Shenglang Zhang; Pengxiang Zhu; Angela Yao
>
> **备注:** To appear at CVPR'26
>
> **摘要:** We present the first systematic analysis of multimodal large language models (MLLMs) in personalized question-answering requiring ego-grounding - the ability to understand the camera-wearer in egocentric videos. To this end, we introduce MyEgo, the first egocentric VideoQA dataset designed to evaluate MLLMs' ability to understand, remember, and reason about the camera wearer. MyEgo comprises 541 long videos and 5K personalized questions asking about "my things", "my activities", and "my past". Benchmarking reveals that competitive MLLMs across variants, including open-source vs. proprietary, thinking vs. non-thinking, small vs. large scales all struggle on MyEgo. Top closed- and open-source models (e.g., GPT-5 and Qwen3-VL) achieve only~46% and 36% accuracy, trailing human performance by near 40% and 50% respectively. Surprisingly, neither explicit reasoning nor model scaling yield consistent improvements. Models improve when relevant evidence is explicitly provided, but gains drop over time, indicating limitations in tracking and remembering "me" and "my past". These findings collectively highlight the crucial role of ego-grounding and long-range memory in enabling personalized QA in egocentric videos. We hope MyEgo and our analyses catalyze further progress in these areas for egocentric personalized assistance. Data and code are available at this https URL
>
---
#### [new 134] GardenDesigner: Encoding Aesthetic Principles into Jiangnan Garden Construction via a Chain of Agents
- **分类: cs.CV**

- **简介: 该论文属于数字艺术生成任务，旨在解决江南园林手动建模效率低的问题。通过编码美学原则和代理链，实现自动化园林设计与交互式构建。**

- **链接: [https://arxiv.org/pdf/2604.01777](https://arxiv.org/pdf/2604.01777)**

> **作者:** Mengtian Li; Fan Yang; Ruixue Xiong; Yiyan Fan; Zhifeng Xie; Zeyu Wang
>
> **备注:** CVPR 2026, Project page: this https URL
>
> **摘要:** Jiangnan gardens, a prominent style of Chinese classical gardens, hold great potential as digital assets for film and game production and digital tourism. However, manual modeling of Jiangnan gardens heavily relies on expert experience for layout design and asset creation, making the process time-consuming. To address this gap, we propose GardenDesigner, a novel framework that encodes aesthetic principles for Jiangnan garden construction and integrates a chain of agents based on procedural modeling. The water-centric terrain and explorative pathway rules are applied by terrain distribution and road generation agents. Selection and spatial layout of garden assets follow the aesthetic and cultural constraints. Consequently, we propose asset selection and layout optimization agents to select and arrange objects for each area in the garden. Additionally, we introduce GardenVerse for Jiangnan garden construction, including expert-annotated garden knowledge to enhance the asset arrangement process. To enable interaction and editing, we develop an interactive interface and tools in Unity, in which non-expert users can construct Jiangnan gardens via text input within one minute. Experiments and human evaluations demonstrate that GardenDesigner can generate diverse and aesthetically pleasing Jiangnan gardens. Project page is available at this https URL.
>
---
#### [new 135] CASHG: Context-Aware Stylized Online Handwriting Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于在线手写生成任务，旨在解决句子级手写风格一致性问题。提出CASHG模型，显式建模字符连接性，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.02103](https://arxiv.org/pdf/2604.02103)**

> **作者:** Jinsu Shin; Sungeun Hong; Jin Yeong Bak
>
> **备注:** 42 pages, 19 figures
>
> **摘要:** Online handwriting represents strokes as time-ordered trajectories, which makes handwritten content easier to transform and reuse in a wide range of applications. However, generating natural sentence-level online handwriting that faithfully reflects a writer's style remains challenging, since sentence synthesis demands context-dependent characters with stroke continuity and spacing. Prior methods treat these boundary properties as implicit outcomes of sequence modeling, which becomes unreliable at the sentence scale and under limited compositional diversity. We propose CASHG, a context-aware stylized online handwriting generator that explicitly models inter-character connectivity for style-consistent sentence-level trajectory synthesis. CASHG uses a Character Context Encoder to obtain character identity and sentence-dependent context memory and fuses them in a bigram-aware sliding-window Transformer decoder that emphasizes local predecessor--current transitions, complemented by gated context fusion for sentence-level this http URL proceeds through a three-stage curriculum from isolated glyphs to full sentences, improving robustness under sparse transition coverage. We further introduce Connectivity and Spacing Metrics (CSM), a boundary-aware evaluation suite that quantifies cursive connectivity and spacing similarity. Under benchmark-matched evaluation protocols, CASHG consistently improves CSM over comparison methods while remaining competitive in DTW-based trajectory similarity, with gains corroborated by a human evaluation.
>
---
#### [new 136] CLPIPS: A Personalized Metric for AI-Generated Image Similarity
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文提出CLPIPS，用于提升AI生成图像相似度度量与人类判断的一致性。针对文本到图像生成中的感知对齐问题，通过轻量级人类增强微调优化度量模型。**

- **链接: [https://arxiv.org/pdf/2604.01234](https://arxiv.org/pdf/2604.01234)**

> **作者:** Khoi Trinh; Jay Rothenberger; Scott Seidenberger; Dimitrios Diochnos; Anindya Maiti
>
> **摘要:** Iterative prompt refinement is central to reproducing target images with text to image generative models. Previous studies have incorporated image similarity metrics (ISMs) as additional feedback to human users. Existing ISMs such as LPIPS and CLIP provide objective measures of image likeness but often fail to align with human judgments, particularly in context specific or user driven tasks. In this paper, we introduce Customized Learned Perceptual Image Patch Similarity (CLPIPS), a customized extension of LPIPS that adapts a metric's notion of similarity directly to human judgments. We aim to explore whether lightweight, human augmented fine tuning can meaningfully improve perceptual alignment, positioning similarity metrics as adaptive components for human in the loop workflows with text to image tools. We evaluate CLPIPS on a human subject dataset in which participants iteratively regenerate target images and rank generated outputs by perceived similarity. Using margin ranking loss on human ranked image pairs, we fine tune only the LPIPS layer combination weights and assess alignment via Spearman rank correlation and Intraclass Correlation Coefficient. Our results show that CLPIPS achieves stronger correlation and agreement with human judgments than baseline LPIPS. Rather than optimizing absolute metric performance, our work emphasizes improving alignment consistency between metric predictions and human ranks, demonstrating that even limited human specific fine tuning can meaningfully enhance perceptual alignment in human in the loop text to image workflows.
>
---
#### [new 137] Captioning Daily Activity Images in Early Childhood Education: Benchmark and Algorithm
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像描述生成任务，旨在解决早期儿童教育场景下图像描述不精准的问题。通过构建ECAC数据集和提出RSRS训练框架，提升模型对专业物体的识别与描述能力。**

- **链接: [https://arxiv.org/pdf/2604.01941](https://arxiv.org/pdf/2604.01941)**

> **作者:** Sixing Li; Zhibin Gu; Ziqi Zhang; Weiguo Pan; Bing Li; Ying Wang; Hongzhe Liu
>
> **摘要:** Image captioning for Early Childhood Education (ECE) is essential for automated activity understanding and educational assessment. However, existing methods face two key challenges. First, the lack of large-scale, domain-specific datasets limits the model's ability to capture fine-grained semantic concepts unique to ECE scenarios, resulting in generic and imprecise descriptions. Second, conventional training paradigms exhibit limitations in enhancing professional object description capability, as supervised learning tends to favor high-frequency expressions, while reinforcement learning may suffer from unstable optimization on difficult samples. To address these limitations, we introduce ECAC, a large-scale benchmark for ECE daily activity image captioning, comprising 256,121 real-world images annotated with expert-level captions and fine-grained labels. ECAC is further equipped with a domain-oriented evaluation protocol, the Teaching Toy Recognition Score (TTS), to explicitly measure professional object naming accuracy. Furthermore, we propose RSRS (Reward-Conditional Switch of Reinforcement Learning and Supervised Fine-Tuning), a hybrid training framework that dynamically alternates between RL and supervised optimization. By rerouting hard samples with zero rewards to supervised fine-tuning, RSRS effectively mitigates advantage collapse and enables stable optimization for fine-grained recognition. Leveraging ECAC and RSRS, we develop KinderMM-Cap-3B, a domain-adapted multimodal large language model. Extensive experiments demonstrate that our model achieves a TTS of 51.06, substantially outperforming state-of-the-art baselines while maintaining superior caption quality, highlighting its potential for specialized educational applications.
>
---
#### [new 138] Rare-Aware Autoencoding: Reconstructing Spatially Imbalanced Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像重建任务，解决空间不平衡数据的重构问题。通过自熵损失和样本传播机制，提升对罕见空间位置的重建效果。**

- **链接: [https://arxiv.org/pdf/2604.02031](https://arxiv.org/pdf/2604.02031)**

> **作者:** Alejandro Castañeda Garcia; Jan van Gemert; Daan Brinks; Nergis Tömen
>
> **摘要:** Autoencoders can be challenged by spatially non-uniform sampling of image content. This is common in medical imaging, biology, and physics, where informative patterns occur rarely at specific image coordinates, as background dominates these locations in most samples, biasing reconstructions toward the majority appearance. In practice, autoencoders are biased toward dominant patterns resulting in the loss of fine-grained detail and causing blurred reconstructions for rare spatial inputs especially under spatial data imbalance. We address spatial imbalance by two complementary components: (i) self-entropy-based loss that upweights statistically uncommon spatial locations and (ii) Sample Propagation, a replay mechanism that selectively re-exposes the model to hard to reconstruct samples across batches during training. We benchmark existing data balancing strategies, originally developed for supervised classification, in the unsupervised reconstruction setting. Drawing on the limitations of these approaches, our method specifically targets spatial imbalance by encouraging models to focus on statistically rare locations, improving reconstruction consistency compared to existing baselines. We validate in a simulated dataset with controlled spatial imbalance conditions, and in three, uncontrolled, diverse real-world datasets spanning physical, biological, and astronomical domains. Our approach outperforms baselines on various reconstruction metrics, particularly under spatial imbalance distributions. These results highlight the importance of data representation in a batch and emphasize rare samples in unsupervised image reconstruction. We will make all code and related data available.
>
---
#### [new 139] Deep Neural Network Based Roadwork Detection for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于道路施工检测任务，旨在解决自动驾驶中动态施工区域识别问题。通过融合YOLO与LiDAR数据，实现施工区域的实时检测与定位。**

- **链接: [https://arxiv.org/pdf/2604.02282](https://arxiv.org/pdf/2604.02282)**

> **作者:** Sebastian Wullrich; Nicolai Steinke; Daniel Goehring
>
> **备注:** 7 pages, 10 figures
>
> **摘要:** Road construction sites create major challenges for both autonomous vehicles and human drivers due to their highly dynamic and heterogeneous nature. This paper presents a real-time system that detects and localizes roadworks by combining a YOLO neural network with LiDAR data. The system identifies individual roadwork objects while driving, merges them into coherent construction sites and records their outlines in world coordinates. The model training was based on an adapted US dataset and a new dataset collected from test drives with a prototype vehicle in Berlin, Germany. Evaluations on real-world road construction sites showed a localization accuracy below 0.5 m. The system can support traffic authorities with up-to-date roadwork data and could enable autonomous vehicles to navigate construction sites more safely in the future.
>
---
#### [new 140] M3D-BFS: a Multi-stage Dynamic Fusion Strategy for Sample-Adaptive Multi-Modal Brain Network Analysis
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于多模态脑网络分析任务，解决静态融合方法无法适应不同样本的问题。提出M3D-BFS动态融合策略，通过分阶段训练提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.01667](https://arxiv.org/pdf/2604.01667)**

> **作者:** Rui Dong; Xiaotong Zhang; Jiaxing Li; Yueying Li; Jiayin Wei; Youyong Kong
>
> **摘要:** Multi-modal fusion is of great significance in neuroscience which integrates information from different modalities and can achieve better performance than uni-modal methods in downstream tasks. Current multi-modal fusion methods in brain networks, which mainly focus on structural connectivity (SC) and functional connectivity (FC) modalities, are static in nature. They feed different samples into the same model with identical computation, ignoring inherent difference between input samples. This lack of sample adaptation hinders model's further performance. To this end, we innovatively propose a multi-stage dynamic fusion strategy (M3D-BFS) for sample-adaptive multi-modal brain network analysis. Unlike other static fusion methods, we design different mixture-of-experts (MoEs) for uni- and multi-modal representations where modules can adaptively change as input sample changes during inference. To alleviate issue of MoE where training of experts may be collapsed, we divide our method into 3 stages. We first train uni-modal encoders respectively, then pretrain single experts of MoEs before finally finetuning the whole model. A multi-modal disentanglement loss is designed to enhance the final representations. To the best of our knowledge, this is the first work for dynamic fusion for multi-modal brain network analysis. Extensive experiments on different real-world datasets demonstrates the superiority of M3D-BFS.
>
---
#### [new 141] Country-wide, high-resolution monitoring of forest browning with Sentinel-2
- **分类: stat.AP; cs.CV**

- **简介: 该论文属于森林健康监测任务，旨在解决大范围森林退化检测问题。通过Sentinel-2数据构建模型，识别森林绿色度异常，实现高分辨率的森林退化量化分析。**

- **链接: [https://arxiv.org/pdf/2604.02074](https://arxiv.org/pdf/2604.02074)**

> **作者:** Samantha Biegel; David Brüggemann; Francesco Grossi; Michele Volpi; Konrad Schindler; Benjamin D. Stocker
>
> **备注:** 9 pages, 7 figures, to be published in the ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences (ISPRS Congress)
>
> **摘要:** Natural and anthropogenic disturbances are impacting the health of forests worldwide. Monitoring forest disturbances at scale is important to inform conservation efforts. Here, we present a scalable approach for country-wide mapping of forest greenness anomalies at the 10 m resolution of Sentinel-2. Using relevant ecological and topographical context and an established representation of the vegetation cycle, we learn a predictive quantile model of the normalised difference vegetation index (NDVI) derived from Sentinel-2 data. The resulting expected seasonal cycles are used to detect NDVI anomalies across Switzerland between April 2017 and August 2025. Goodness-of-fit evaluations show that the conditional model explains 65% of the observed variations in the median seasonal cycle. The model consistently benefits from the local context information, particularly during the green-up period. The approach produces coherent spatial anomaly patterns and enables country-wide quantification of forest browning. Case studies with independent reference data from known events illustrate that the model reliably detects different types of disturbances.
>
---
#### [new 142] Efficient Equivariant Transformer for Self-Driving Agent Modeling
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶中的智能体行为建模任务，旨在解决SE(2)-equivariance的高效建模问题。提出DriveGATr架构，通过几何代数方法实现等变性，避免了高成本的显式位置编码。**

- **链接: [https://arxiv.org/pdf/2604.01466](https://arxiv.org/pdf/2604.01466)**

> **作者:** Scott Xu; Dian Chen; Kelvin Wong; Chris Zhang; Kion Fallah; Raquel Urtasun
>
> **备注:** CVPR 2026
>
> **摘要:** Accurately modeling agent behaviors is an important task in self-driving. It is also a task with many symmetries, such as equivariance to the order of agents and objects in the scene or equivariance to arbitrary roto-translations of the entire scene as a whole; i.e., SE(2)-equivariance. The transformer architecture is a ubiquitous tool for modeling these symmetries. While standard self-attention is inherently permutation equivariant, explicit pairwise relative positional encodings have been the standard for introducing SE(2)-equivariance. However, this approach introduces an additional cost that is quadratic in the number of agents, limiting its scalability to larger scenes and batch sizes. In this work, we propose DriveGATr, a novel transformer-based architecture for agent modeling that achieves SE(2)-equivariance without the computational cost of existing methods. Inspired by recent advances in geometric deep learning, DriveGATr encodes scene elements as multivectors in the 2D projective geometric algebra $\mathbb{R}^*_{2,0,1}$ and processes them with a stack of equivariant transformer blocks. Crucially, DriveGATr models geometric relationships using standard attention between multivectors, eliminating the need for costly explicit pairwise relative positional encodings. Experiments on the Waymo Open Motion Dataset demonstrate that DriveGATr is comparable to the state-of-the-art in traffic simulation and establishes a superior Pareto front for performance vs computational cost.
>
---
#### [new 143] Enhanced Polarization Locking in VCSELs
- **分类: physics.optics; cs.CV**

- **简介: 该论文属于激光器优化任务，旨在解决VCSEL polarization locking的挑战。通过设计氧化孔和电流调节，提升锁定性能。**

- **链接: [https://arxiv.org/pdf/2604.01857](https://arxiv.org/pdf/2604.01857)**

> **作者:** Zifeng Yuan; Dewen Zhang; Lei Shi; Yutong Liu; Aaron Danner
>
> **摘要:** While optical injection locking (OIL) of vertical-cavity surface-emitting lasers (VCSELs) has been widely studied in the past, the polarization dynamics of OIL have received far less attention. Recent studies suggest that polarization locking via OIL could enable novel computational applications such as polarization-encoded Ising computers. However, the inherent polarization preference and limited polarization switchability of VCSELs hinder their use for such purposes. To address these challenges, we fabricate VCSELs with tailored oxide aperture designs and combine these with bias current tuning to study the overall impact on polarization locking. Experimental results demonstrate that this approach reduces the required injection power (to as low as 3.6 {\mu}W) and expands the locking range. To investigate the impact of the approach, the spin-flip model (SFM) is used to analyze the effects of amplitude anisotropy and bias current on polarization locking, demonstrating strong coherence with experimental results.
>
---
#### [new 144] Non-Rigid 3D Shape Correspondences: From Foundations to Open Challenges and Opportunities
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于非刚性3D形状对应任务，解决变形形状间对应关系估计问题，综述了谱方法、组合方法和变形方法，并探讨了未来研究方向。**

- **链接: [https://arxiv.org/pdf/2604.01274](https://arxiv.org/pdf/2604.01274)**

> **作者:** Aleksei Zhuravlev; Lennart Bastian; Dongliang Cao; Nafie El Amrani; Paul Roetzer; Viktoria Ehm; Riccardo Marin; Hiroki Nishizawa; Shigeo Morishima; Christian Theobalt; Nassir Navab; Daniel Cremers; Florian Bernard; Zorah Lähner; Vladislav Golyanik
>
> **备注:** 35 pages and 15 figures; Eurographics 2026 STAR; Project page: this https URL
>
> **摘要:** Estimating correspondences between deformed shape instances is a long-standing problem in computer graphics; numerous applications, from texture transfer to statistical modelling, rely on recovering an accurate correspondence map. Many methods have thus been proposed to tackle this challenging problem from varying perspectives, depending on the downstream application. This state-of-the-art report is geared towards researchers, practitioners, and students seeking to understand recent trends and advances in the field. We categorise developments into three paradigms: spectral methods based on functional maps, combinatorial formulations that impose discrete constraints, and deformation-based methods that directly recover a global alignment. Each school of thought offers different advantages and disadvantages, which we discuss throughout the report. Meanwhile, we highlight the latest developments in each area and suggest new potential research directions. Finally, we provide an overview of emerging challenges and opportunities in this growing field, including the recent use of vision foundation models for zero-shot correspondence and the particularly challenging task of matching partial shapes.
>
---
#### [new 145] SECURE: Stable Early Collision Understanding via Robust Embeddings in Autonomous Driving
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于自动驾驶中的事故预测任务，旨在解决模型对输入扰动的不稳定性问题。提出SECURE框架，通过多目标损失提升模型鲁棒性与性能。**

- **链接: [https://arxiv.org/pdf/2604.01337](https://arxiv.org/pdf/2604.01337)**

> **作者:** Wenjing Wang; Wenxuan Wang; Songning Lai
>
> **备注:** 13 pages, 2 figures
>
> **摘要:** While deep learning has significantly advanced accident anticipation, the robustness of these safety-critical systems against real-world perturbations remains a major challenge. We reveal that state-of-the-art models like CRASH, despite their high performance, exhibit significant instability in predictions and latent representations when faced with minor input perturbations, posing serious reliability risks. To address this, we introduce SECURE - Stable Early Collision Understanding Robust Embeddings, a framework that formally defines and enforces model robustness. SECURE is founded on four key attributes: consistency and stability in both prediction space and latent feature space. We propose a principled training methodology that fine-tunes a baseline model using a multi-objective loss, which minimizes divergence from a reference model and penalizes sensitivity to adversarial perturbations. Experiments on DAD and CCD datasets demonstrate that our approach not only significantly enhances robustness against various perturbations but also improves performance on clean data, achieving new state-of-the-art results.
>
---
#### [new 146] Why Instruction-Based Unlearning Fails in Diffusion Models?
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于模型可解释性与可控生成任务，研究指令式遗忘在扩散模型中的失效问题，通过实验和分析发现自然语言指令无法有效抑制目标概念。**

- **链接: [https://arxiv.org/pdf/2604.01514](https://arxiv.org/pdf/2604.01514)**

> **作者:** Zeliang Zhang; Rui Sun; Jiani Liu; Qi Wu; Chenliang Xu
>
> **摘要:** Instruction-based unlearning has proven effective for modifying the behavior of large language models at inference time, but whether this paradigm extends to other generative models remains unclear. In this work, we investigate instruction-based unlearning in diffusion-based image generation models and show, through controlled experiments across multiple concepts and prompt variants, that diffusion models systematically fail to suppress targeted concepts when guided solely by natural-language unlearning instructions. By analyzing both the CLIP text encoder and cross-attention dynamics during the denoising process, we find that unlearning instructions do not induce sustained reductions in attention to the targeted concept tokens, causing the targeted concept representations to persist throughout generation. These results reveal a fundamental limitation of prompt-level instruction in diffusion models and suggest that effective unlearning requires interventions beyond inference-time language control.
>
---
#### [new 147] Novel Memory Forgetting Techniques for Autonomous AI Agents: Balancing Relevance and Efficiency
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于对话系统任务，解决长期对话中记忆冗余与错误传播问题。提出一种自适应遗忘框架，通过相关性评分和优化策略，提升记忆效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.02280](https://arxiv.org/pdf/2604.02280)**

> **作者:** Payal Fofadiya; Sunil Tiwari
>
> **摘要:** Long-horizon conversational agents require persistent memory for coherent reasoning, yet uncontrolled accumulation causes temporal decay and false memory propagation. Benchmarks such as LOCOMO and LOCCO report performance degradation from 0.455 to 0.05 across stages, while MultiWOZ shows 78.2% accuracy with 6.8% false memory rate under persistent retention. This work introduces an adaptive budgeted forgetting framework that regulates memory through relevanceguided scoring and bounded optimization. The approach integrates recency, frequency, and semantic alignment to maintain stability under constrained context. Comparative analysis demonstrates improved long-horizon F1 beyond 0.583 baseline levels, higher retention consistency, and reduced false memory behavior without increasing context usage. These findings confirm that structured forgetting preserves reasoning performance while preventing unbounded memory growth in extended conversational settings.
>
---
#### [new 148] Stop Wandering: Efficient Vision-Language Navigation via Metacognitive Reasoning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决传统方法效率低的问题。提出MetaNav，通过元认知机制提升导航效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.02318](https://arxiv.org/pdf/2604.02318)**

> **作者:** Xueying Li; Feng Lyu; Hao Wu; Mingliu Liu; Jia-Nan Liu; Guozi Liu
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Training-free Vision-Language Navigation (VLN) agents powered by foundation models can follow instructions and explore 3D environments. However, existing approaches rely on greedy frontier selection and passive spatial memory, leading to inefficient behaviors such as local oscillation and redundant revisiting. We argue that this stems from a lack of metacognitive capabilities: the agent cannot monitor its exploration progress, diagnose strategy failures, or adapt accordingly. To address this, we propose MetaNav, a metacognitive navigation agent integrating spatial memory, history-aware planning, and reflective correction. Spatial memory builds a persistent 3D semantic map. History-aware planning penalizes revisiting to improve efficiency. Reflective correction detects stagnation and uses an LLM to generate corrective rules that guide future frontier selection. Experiments on GOAT-Bench, HM3D-OVON, and A-EQA show that MetaNav achieves state-of-the-art performance while reducing VLM queries by 20.7%, demonstrating that metacognitive reasoning significantly improves robustness and efficiency.
>
---
#### [new 149] DenOiS: Dual-Domain Denoising of Observation and Solution in Ultrasound Image Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像重建任务，旨在解决噪声和模型不准确导致的图像质量下降问题。提出DenOiS框架，同时去噪观测数据和重建结果，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2604.02105](https://arxiv.org/pdf/2604.02105)**

> **作者:** Can Deniz Bezek; Orcun Goksel
>
> **摘要:** Medical imaging aims to recover underlying tissue properties, using inexact (simplified/linearized) imaging models and often from inaccurate and incomplete measurements. Analytical reconstruction methods rely on hand-crafted regularization, sensitive to noise assumptions and parameter tuning. Among deep learning alternatives, plug-and-play (PnP) approaches learn regularization while incorporating imaging physics during inference, outperforming purely data-driven methods. The performance of all these approaches, however, still strongly depends on measurement quality and imaging model accuracy. In this work, we propose DenOiS, a framework that denoises both input observations and resulting solution in their respective domains. It consists of an observation refinement strategy that corrects degraded measurements while compensating for imaging model simplifications, and a diffusion-based PnP reconstruction approach that remains robust under missing measurements. DenOiS enables generalization to real data from training only in simulations, resulting in high-fidelity image reconstruction with noisy observations and inexact imaging models. We demonstrate this for speed-of-sound imaging as a challenging setting of quantitative ultrasound image reconstruction.
>
---
## 更新

#### [replaced 001] Think, Act, Build: An Agentic Framework with Vision Language Models for Zero-Shot 3D Visual Grounding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.00528](https://arxiv.org/pdf/2604.00528)**

> **作者:** Haibo Wang; Zihao Lin; Zhiyang Xu; Lifu Huang
>
> **摘要:** 3D Visual Grounding (3D-VG) aims to localize objects in 3D scenes via natural language descriptions. While recent advancements leveraging Vision-Language Models (VLMs) have explored zero-shot possibilities, they typically suffer from a static workflow relying on preprocessed 3D point clouds, essentially degrading grounding into proposal matching. To bypass this reliance, our core motivation is to decouple the task: leveraging 2D VLMs to resolve complex spatial semantics, while relying on deterministic multi-view geometry to instantiate the 3D structure. Driven by this insight, we propose "Think, Act, Build (TAB)", a dynamic agentic framework that reformulates 3D-VG tasks as a generative 2D-to-3D reconstruction paradigm operating directly on raw RGB-D streams. Specifically, guided by a specialized 3D-VG skill, our VLM agent dynamically invokes visual tools to track and reconstruct the target across 2D frames. Crucially, to overcome the multi-view coverage deficit caused by strict VLM semantic tracking, we introduce the Semantic-Anchored Geometric Expansion, a mechanism that first anchors the target in a reference video clip and then leverages multi-view geometry to propagate its spatial location across unobserved frames. This enables the agent to "Build" the target's 3D representation by aggregating these multi-view features via camera parameters, directly mapping 2D visual cues to 3D coordinates. Furthermore, to ensure rigorous assessment, we identify flaws such as reference ambiguity and category errors in existing benchmarks and manually refine the incorrect queries. Extensive experiments on ScanRefer and Nr3D demonstrate that our framework, relying entirely on open-source models, significantly outperforms previous zero-shot methods and even surpasses fully supervised baselines.
>
---
#### [replaced 002] EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23205](https://arxiv.org/pdf/2602.23205)**

> **作者:** Wenjia Wang; Liang Pan; Huaijin Pi; Yuke Lou; Xuqian Ren; Yifan Wu; Zhouyingcheng Liao; Lei Yang; Rishabh Dabral; Christian Theobalt; Taku Komura
>
> **摘要:** Human behaviors in the real world naturally encode rich, long-term contextual information that can be leveraged to train embodied agents for perception, understanding, and acting. However, existing capture systems typically rely on costly studio setups and wearable devices, limiting the large-scale collection of scene-conditioned human motion data in the wild. To address this, we propose EmbodMocap, a portable and affordable data collection pipeline using two moving iPhones. Our key idea is to jointly calibrate dual RGB-D sequences to reconstruct both humans and scenes within a unified metric world coordinate frame. The proposed method allows metric-scale and scene-consistent capture in everyday environments without static cameras or markers, bridging human motion and scene geometry seamlessly. Compared with optical capture ground truth, we demonstrate that the dual-view setting exhibits a remarkable ability to mitigate depth ambiguity, achieving superior alignment and reconstruction performance over single iphone or monocular models. Based on the collected data, we empower three embodied AI tasks: monocular human-scene-reconstruction, where we fine-tune on feedforward models that output metric-scale, world-space aligned humans and scenes; physics-based character animation, where we prove our data could be used to scale human-object interaction skills and scene-aware motion tracking; and robot motion control, where we train a humanoid robot via sim-to-real RL to replicate human motions depicted in videos. Experimental results validate the effectiveness of our pipeline and its contributions towards advancing embodied AI research.
>
---
#### [replaced 003] PPEDCRF: Privacy-Preserving Enhanced Dynamic CRF for Location-Privacy Protection for Sequence Videos with Minimal Detection Degradation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01593](https://arxiv.org/pdf/2603.01593)**

> **作者:** Bo Ma; Jinsong Wu; Weiqi Yan; Catherine Shi; Minh Nguyen
>
> **备注:** We would like to withdraw this paper due to identified issues in the experimental design and insufficient supporting data, which affect the reliability of the reported results. A substantially revised version with corrected experiments and extended evaluations will be prepared and submitted in the future
>
> **摘要:** Dashcam videos collected by autonomous or assisted-driving systems are increasingly shared for safety auditing and model improvement. Even when explicit GPS metadata are removed, an attacker can still infer the recording location by matching background visual cues (e.g., buildings and road layouts) against large-scale street-view imagery. This paper studies location-privacy leakage under a background-based retrieval attacker, and proposes PPEDCRF, a privacy-preserving enhanced dynamic conditional random field framework that injects calibrated perturbations only into inferred location-sensitive background regions while preserving foreground detection utility. PPEDCRF consists of three components: (i) a dynamic CRF that enforces temporal consistency to discover and track location sensitive regions across frames, (ii) a normalized control penalty (NCP) that allocates perturbation strength according to a hierarchical sensitivity model, and (iii) a utility-preserving noise injection module that minimizes interference to object detection and segmentation. Experiments on public driving datasets demonstrate that PPEDCRF significantly reduces location-retrieval attack success (e.g., Top-k retrieval accuracy) while maintaining competitive detection performance (e.g., mAP and segmentation metrics) compared with common baselines such as global noise, white-noise masking, and feature-based anonymization. The source code is in this https URL
>
---
#### [replaced 004] Be Tangential to Manifold: Discovering Riemannian Metric for Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.05509](https://arxiv.org/pdf/2510.05509)**

> **作者:** Shinnosuke Saito; Takashi Matsubara
>
> **摘要:** Diffusion models are powerful deep generative models, but unlike classical models, they lack an explicit low-dimensional latent space that parameterizes the data manifold. This absence makes it difficult to perform manifold-aware operations, such as geometrically faithful interpolation or conditional guidance that respects the learned manifold. We propose a training-free Riemannian metric on the noise space, derived from the Jacobian of the score function. The key insight is that the spectral structure of this Jacobian separates tangent and normal directions of the data manifold; our metric leverages this separation to encourage paths to stay tangential to the manifold rather than drift toward high-density regions. To validate that our metric faithfully captures the manifold geometry, we examine it from two complementary angles. First, geodesics under our metric yield perceptually more natural interpolations than existing methods on synthetic, image, and video frame datasets. Second, the tangent-normal decomposition induced by our metric prevents classifier-free guidance from deviating off the manifold, improving generation quality while preserving text-image alignment.
>
---
#### [replaced 005] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2512.14870](https://arxiv.org/pdf/2512.14870)**

> **作者:** Dan Ben-Ami; Gabriele Serussi; Kobi Cohen; Chaim Baskin
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Video Large Language Models (Video-LLMs) are improving rapidly, yet current Video Question Answering (VideoQA) benchmarks often admit single-cue shortcuts, under-testing reasoning that must integrate evidence across time. We introduce HERBench, a benchmark designed to make multi-evidence integration unavoidable: each question requires at least three non-overlapping cues drawn from distinct video segments. HERBench contains 26,806 five-way multiple-choice questions across 12 compositional tasks. To make evidential demand measurable, we introduce the Minimum Required Frame-Set (MRFS), the smallest number of frames a model must fuse to answer correctly, and show that HERBench imposes higher evidential demand than prior benchmarks. Evaluating 13 state-of-the-art Video-LLMs yields only 31-42% accuracy, only modestly above the 20\% random-guess baseline. We disentangle this failure into two critical bottlenecks: (1) a retrieval deficit, where frame selectors overlook key evidence, and (2) a fusion deficit, where models fail to integrate information even when all necessary evidence is provided. HERBench thus provides a principled benchmark for studying robust multi-evidence video understanding.
>
---
#### [replaced 006] Understanding the Risks of Asphalt Art to the Reliability of Vision-Based Perception Systems
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02530](https://arxiv.org/pdf/2508.02530)**

> **作者:** Jin Ma; Abyad Enan; Long Cheng; Mashrur Chowdhury
>
> **备注:** J. Ma and A. Enan are co-first authors; they have contributed equally. This second revised version has been resubmitted to the Transportation Research Record: Journal of the Transportation Research Board after addressing the reviewers' comments and is currently awaiting the final decision
>
> **摘要:** Artistic crosswalks featuring asphalt art, introduced by different organizations in recent years, aim to enhance the visibility and safety of pedestrians. However, their visual complexity may interfere with surveillance systems that rely on vision-based object detection models. In this study, we investigate the impact of asphalt art on pedestrian detection performance of a pretrained vision-based object detection model. We construct realistic crosswalk scenarios by compositing various street art patterns into a fixed surveillance scene and evaluate the model's performance in detecting pedestrians on asphalt-arted crosswalks under both benign and adversarial conditions. A benign case refers to pedestrian crosswalks painted with existing normal asphalt art, whereas an adversarial case involves digitally crafted or altered asphalt art perpetrated by an attacker. Our results show that while simple, color-based designs have minimal effect, complex artistic patterns, particularly those with high visual salience, can significantly degrade pedestrian detection performance. Furthermore, we demonstrate that adversarially crafted asphalt art can be exploited to deliberately obscure real pedestrians or generate non-existent pedestrian detections. These findings highlight a potential vulnerability in urban vision-based pedestrian surveillance systems, and underscore the importance of accounting for environmental visual variations when designing robust pedestrian perception models.
>
---
#### [replaced 007] PMMA: The Polytechnique Montreal Mobility Aids Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.10259](https://arxiv.org/pdf/2602.10259)**

> **作者:** Qingwu Liu; Nicolas Saunier; Guillaume-Alexandre Bilodeau
>
> **备注:** Submitted to the journal IEEE Open Journal Intelligent Transportation Systems, under review
>
> **摘要:** This study introduces a new object detection dataset of pedestrians using mobility aids, named PMMA. The dataset was collected in an outdoor environment, where volunteers used wheelchairs, canes, and walkers, resulting in nine categories of pedestrians: pedestrians, cane users, two types of walker users, whether walking or resting, five types of wheelchair users, including wheelchair users, people pushing empty wheelchairs, and three types of users pushing occupied wheelchairs, including the entire pushing group, the pusher and the person seated on the wheelchair. To establish a benchmark, seven object detection models (Faster R-CNN, CenterNet, YOLOX, DETR, Deformable DETR, DINO, and RT-DETR) and three tracking algorithms (ByteTrack, BOT-SORT, and OC-SORT) were implemented under the MMDetection framework. Experimental results show that YOLOX, Deformable DETR, and Faster R-CNN achieve the best detection performance, while the differences among the three trackers are relatively small. The PMMA dataset is publicly available at this https URL, and the video processing and model training code is available at this https URL.
>
---
#### [replaced 008] Human Insights Driven Latent Space for Different Driving Perspectives: A Unified Encoder for Efficient Multi-Task Inference
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.10095](https://arxiv.org/pdf/2409.10095)**

> **作者:** Huy-Dung Nguyen; Anass Bairouk; Mirjana Maras; Wei Xiao; Tsun-Hsuan Wang; Patrick Chareyre; Ramin Hasani; Marc Blanchon; Daniela Rus
>
> **摘要:** Autonomous driving systems require a comprehensive understanding of the environment, achieved by extracting visual features essential for perception, planning, and control. However, models trained solely on single-task objectives or generic datasets often lack the contextual information needed for robust performance in complex driving scenarios. In this work, we propose a unified encoder trained on multiple computer vision tasks crucial for urban driving, including depth, pose, and 3D scene flow estimation, as well as semantic, instance, panoptic, and motion segmentation. By integrating these diverse visual cues-similar to human perceptual mechanisms-the encoder captures rich features that enhance navigation-related predictions. We evaluate the model on steering estimation as a downstream task, leveraging its dense latent space. To ensure efficient multi-task learning, we introduce a multi-scale feature network for pose estimation and apply knowledge distillation from a multi-backbone teacher model. Our findings highlight two key findings: (1) the unified encoder achieves competitive performance across all visual perception tasks, demonstrating strong generalization capabilities; and (2) for steering estimation, the frozen unified encoder-leveraging dense latent representations-outperforms both its fine-tuned counterpart and the same frozen model pretrained on generic datasets like ImageNet. These results underline the significance of task-specific visual features and demonstrate the promise of multi-task learning in advancing autonomous driving systems. More details and the pretrained model are available at this https URL.
>
---
#### [replaced 009] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.01390](https://arxiv.org/pdf/2512.01390)**

> **作者:** Seungho Choi; Jeahun Sung; Jihyong Oh
>
> **备注:** CVPR 2026 (camera ready ver.). Please visit our project page at this https URL
>
> **摘要:** Real-image super-resolution (Real-ISR) seeks to recover HR images from LR inputs with mixed, unknown degradations. While diffusion models surpass GANs in perceptual quality, they under-reconstruct high-frequency (HF) details due to a low-frequency (LF) bias and a depth-wise "low-first, high-later" hierarchy. We introduce FRAMER, a plug-and-play training scheme that exploits diffusion priors without changing the backbone or inference. At each denoising step, the final-layer feature map teaches all intermediate layers. Teacher and student feature maps are decomposed into LF/HF bands via FFT masks to align supervision with the model's internal frequency hierarchy. For LF, an Intra Contrastive Loss (IntraCL) stabilizes globally shared structure. For HF, an Inter Contrastive Loss (InterCL) sharpens instance-specific details using random-layer and in-batch negatives. Two adaptive modulators, Frequency-based Adaptive Weight (FAW) and Frequency-based Alignment Modulation (FAM), reweight per-layer LF/HF signals and gate distillation by current similarity. Across U-Net and DiT backbones (e.g., Stable Diffusion 2, 3), FRAMER consistently improves PSNR/SSIM and perceptual metrics (LPIPS, NIQE, MANIQA, MUSIQ). Ablations validate the final-layer teacher and random-layer negatives.
>
---
#### [replaced 010] DeDelayed: Deleting Remote Inference Delay via On-Device Correction
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.13714](https://arxiv.org/pdf/2510.13714)**

> **作者:** Dan Jacobellis; Mateen Ulhaq; Fabien Racapé; Hyomin Choi; Neeraja J. Yadwadkar
>
> **备注:** CVPR 2026
>
> **摘要:** Video comprises the vast majority of bits that are generated daily, and is the primary signal driving current innovations in robotics, remote sensing, and wearable technology. Yet, the most powerful video understanding models are too expensive for the resource-constrained platforms used in these applications. One approach is to offload inference to the cloud; this gives access to GPUs capable of processing high-resolution videos in real time. But even with reliable, high-bandwidth communication channels, the combined latency of video encoding, model inference, and round-trip communication prohibits use for certain real-time applications. The alternative is to use fully local inference; but this places extreme constraints on computational and power costs, requiring smaller models and lower resolution, leading to degraded accuracy. To address these challenges, we propose Dedelayed, a real-time inference system that divides computation between a remote model operating on delayed video frames and a local model with access to the current frame. The remote model is trained to make predictions on anticipated future frames, which the local model incorporates into its prediction for the current frame. The local and remote models are jointly optimized with an autoencoder that limits the transmission bitrate required by the available downlink communication channel. We evaluate Dedelayed on the task of real-time streaming video segmentation using the BDD100k driving dataset. For a round trip delay of 100 ms, Dedelayed improves performance by 6.4 mIoU compared to fully local inference and 9.8 mIoU compared to remote inference -- an equivalent improvement to using a model ten times larger. We release our training code, pretrained models, and python library at this https URL .
>
---
#### [replaced 011] GCond: Gradient Conflict Resolution via Accumulation-based Stabilization for Large-Scale Multi-Task Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.07252](https://arxiv.org/pdf/2509.07252)**

> **作者:** Evgeny Alves Limarenko; Anastasiia Studenikina; Svetlana Illarionova; Maxim Sharaev
>
> **备注:** Published in IEEE Access. This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License (CC BY-NC-ND 4.0)
>
> **摘要:** In multi-task learning (MTL), gradient conflict poses a significant challenge. Effective methods for addressing this problem, including PCGrad, CAGrad, and GradNorm, in their original implementations are computationally demanding, which significantly limits their application in modern large models such as transformers. We propose Gradient Conductor (GCond), a method that builds upon PCGrad principles by combining them with gradient accumulation and an adaptive arbitration mechanism. We evaluated GCond on self-supervised multi-task learning tasks using MobileNetV3-Small and ConvNeXt architectures on the ImageNet 1K dataset and a combined head and neck CT scan dataset, comparing the proposed method against baseline linear combinations and state-of-the-art gradient conflict resolution methods. The classical and stochastic approaches of GCond were analyzed. The stochastic mode of GCond achieved a two-fold computational speedup while maintaining optimization quality, and demonstrated superior performance across all evaluated metrics, achieving lower L1 and SSIM losses compared to other methods on both datasets, and demonstrating superior generalization in heterogeneous scenarios: GCond improved ImageNet Top-1 Accuracy by 4.5% over baselines and prevented confidence overfitting in medical diagnosis tasks. GCond exhibited high scalability, being successfully applied to both compact models: MobileNetV3-Small and ConvNeXt-tiny; and large architecture ConvNeXtV2-Base. It also showed compatibility with modern optimizers such as AdamW and Lion/LARS. Therefore, GCond offers a scalable and efficient solution to the problem of gradient conflicts in multi-task learning.
>
---
#### [replaced 012] GridVAD: Open-Set Video Anomaly Detection via Spatial Reasoning over Stratified Frame Grids
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.25467](https://arxiv.org/pdf/2603.25467)**

> **作者:** Mohamed Eltahir; Ahmed O. Ibrahim; Obada Siralkhatim; Tabarak Abdallah; Sondos Mohamed
>
> **摘要:** Vision-Language Models (VLMs) are powerful open-set reasoners, yet their direct use as anomaly detectors in video surveillance is fragile: without calibrated anomaly priors, they alternate between missed detections and hallucinated false alarms. We argue the problem is not the VLM itself but how it is used. VLMs should function as anomaly proposers, generating open-set candidate descriptions that are then grounded and tracked by purpose-built spatial and temporal modules. We instantiate this propose-ground-propagate principle in GridVAD, a training-free pipeline that produces pixel-level anomaly masks without any domain-specific training. A VLM reasons over stratified grid representations of video clips to generate natural-language anomaly proposals. Self-Consistency Consolidation (SCC) filters hallucinations by retaining only proposals that recur across multiple independent samplings. Grounding DINO anchors each surviving proposal to a bounding box, and SAM2 propagates it as a dense mask through the anomaly interval. The per-clip VLM budget is fixed at M+1 calls regardless of video length, where M can be set according to the proposals needed. On UCSD Ped2, GridVAD achieves the highest Pixel-AUROC (77.59) among all compared methods, surpassing even the partially fine-tuned TAO (75.11) and outperforms other zero-shot approaches on object-level RBDC by over 5x. Ablations reveal that SCC provides a controllable precision-recall tradeoff: filtering improves all pixel level metrics at a modest cost in object-level recall. Efficiency experiments show GridVAD is 2.7x more call-efficient than uniform per-frame VLM querying while additionally producing dense segmentation this http URL and qualitative video results are available at this https URL.
>
---
#### [replaced 013] OOD-SEG: Exploiting out-of-distribution detection techniques for learning image segmentation from sparse multi-class positive-only annotations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.09553](https://arxiv.org/pdf/2411.09553)**

> **作者:** Junwen Wang; Zhonghao Wang; Oscar MacCormac; Jonathan Shapey; Tom Vercauteren
>
> **备注:** Accepted in MedIA
>
> **摘要:** Despite significant advancements, segmentation based on deep neural networks in medical and surgical imaging faces several challenges, two of which we aim to address in this work. First, acquiring complete pixel-level segmentation labels for medical images is time-consuming and requires domain expertise. Second, typical segmentation pipelines cannot detect out-of-distribution (OOD) pixels, leaving them prone to spurious outputs during deployment. In this work, we propose a novel segmentation approach which broadly falls within the positive-unlabelled (PU) learning paradigm and exploits tools from OOD detection techniques. Our framework learns only from sparsely annotated pixels from multiple positive-only classes and does not use any annotation for the background class. These multi-class positive annotations naturally fall within the in-distribution (ID) set. Unlabelled pixels may contain positive classes but also negative ones, including what is typically referred to as \emph{background} in standard segmentation formulations. To the best of our knowledge, this work is the first to formulate multi-class segmentation with sparse positive-only annotations as a pixel-wise PU learning problem and to address it using OOD detection techniques. Here, we forgo the need for background annotation and consider these together with any other unseen classes as part of the OOD set. Our framework can integrate, at a pixel-level, any OOD detection approaches designed for classification tasks. To address the lack of existing OOD datasets and established evaluation metric for medical image segmentation, we propose a cross-validation strategy that treats held-out labelled classes as OOD. Extensive experiments on both multi-class hyperspectral and RGB surgical imaging datasets demonstrate the robustness and generalisation capability of our proposed framework.
>
---
#### [replaced 014] Pixel Motion Diffusion is What We Need for Robot Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DAWN框架，用于机器人控制任务，解决高、低层控制间的映射问题。通过扩散模型和像素运动表示，实现端到端学习与可解释的中间抽象。**

- **链接: [https://arxiv.org/pdf/2509.22652](https://arxiv.org/pdf/2509.22652)**

> **作者:** E-Ro Nguyen; Yichi Zhang; Kanchana Ranasinghe; Xiang Li; Michael S. Ryoo
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** We present DAWN (Diffusion is All We Need for robot control), a unified diffusion-based framework for language-conditioned robotic manipulation that bridges high-level motion intent and low-level robot action via structured pixel motion representation. In DAWN, both the high-level and low-level controllers are modeled as diffusion processes, yielding a fully trainable, end-to-end system with interpretable intermediate motion abstractions. DAWN achieves state-of-the-art results on the challenging CALVIN benchmark, demonstrating strong multi-task performance, and further validates its effectiveness on MetaWorld. Despite the substantial domain gap between simulation and reality and limited real-world data, we demonstrate reliable real-world transfer with only minimal finetuning, illustrating the practical viability of diffusion-based motion abstractions for robotic control. Our results show the effectiveness of combining diffusion modeling with motion-centric representations as a strong baseline for scalable and robust robot learning. Project page: this https URL
>
---
#### [replaced 015] OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24458](https://arxiv.org/pdf/2603.24458)**

> **作者:** Kaihang Pan; Qi Tian; Jianwei Zhang; Weijie Kong; Jiangfeng Xiong; Yanxin Long; Shixue Zhang; Haiyi Qiu; Tan Wang; Zheqi Lv; Yue Wu; Liefeng Bo; Siliang Tang; Zhao Zhong
>
> **备注:** 32 pages, 22 figures. Project Page: this https URL. Github: this https URL. Model: this https URL
>
> **摘要:** While proprietary systems such as Seedance-2.0 have achieved remarkable success in omni-capable video generation, open-source alternatives significantly lag behind. Most academic models remain heavily fragmented, and the few existing efforts toward unified video generation still struggle to seamlessly integrate diverse tasks within a single framework. To bridge this gap, we propose OmniWeaving, an omni-level video generation model featuring powerful multimodal composition and reasoning-informed capabilities. By leveraging a massive-scale pretraining dataset that encompasses diverse compositional and reasoning-augmented scenarios, OmniWeaving learns to temporally bind interleaved text, multi-image, and video inputs while acting as an intelligent agent to infer complex user intentions for sophisticated video creation. Furthermore, we introduce IntelligentVBench, the first comprehensive benchmark designed to rigorously assess next-level intelligent unified video generation. Extensive experiments demonstrate that OmniWeaving achieves SoTA performance among open-source unified models. The codes and model have already been publicly available. Project Page: this https URL.
>
---
#### [replaced 016] Seeing without Pixels: Perception from Camera Trajectories
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21681](https://arxiv.org/pdf/2511.21681)**

> **作者:** Zihui Xue; Kristen Grauman; Dima Damen; Andrew Zisserman; Tengda Han
>
> **备注:** Accepted by CVPR 2026, Project website: this https URL
>
> **摘要:** Can one perceive a video's content without seeing its pixels, just from the camera trajectory-the path it carves through space? This paper is the first to systematically investigate this seemingly implausible question. Towards this end, we propose a contrastive learning framework to train CamFormer, a dedicated encoder that projects camera pose trajectories into a joint embedding space, aligning them with natural language. We find that, contrary to its apparent simplicity, the camera trajectory is a remarkably informative signal to uncover video content. In other words, "how you move" can indeed provide valuable cues about "what you are doing" (egocentric) or "observing" (exocentric). We demonstrate the versatility of our learned CamFormer embeddings on a diverse suite of downstream tasks, ranging from cross-modal alignment to classification and temporal analysis. Importantly, our representations are robust across diverse camera pose estimation methods, including both high-fidelity multi-sensored and standard RGB-only estimators. Our findings establish camera trajectory as a lightweight, robust, and versatile modality for perceiving video content.
>
---
#### [replaced 017] Robust Adaptation of Foundation Models with Black-Box Visual Prompting
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2407.17491](https://arxiv.org/pdf/2407.17491)**

> **作者:** Changdae Oh; Gyeongdeok Seo; Geunyoung Jung; Zhi-Qi Cheng; Hosik Choi; Jiyoung Jung; Kyungwoo Song
>
> **备注:** Accepted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 2026
>
> **摘要:** With a surge of large-scale pre-trained models, parameter-efficient transfer learning (PETL) of large models has garnered significant attention. While promising, they commonly rely on two optimistic assumptions: 1) full access to the parameters of a PTM, and 2) sufficient memory capacity to cache all intermediate activations for gradient computation. However, in most real-world applications, PTMs serve as black-box APIs or proprietary software without full parameter accessibility. Besides, it is hard to meet a large memory requirement for modern PTMs. This work proposes black-box visual prompting (BlackVIP), which efficiently adapts the PTMs without knowledge of their architectures or parameters. BlackVIP has two components: 1) Coordinator and 2) simultaneous perturbation stochastic approximation with gradient correction (SPSA-GC). The Coordinator designs input-dependent visual prompts, which allow the target PTM to adapt in the wild. SPSA-GC efficiently estimates the gradient of PTM to update Coordinator. Besides, we introduce a variant, BlackVIP-SE, which significantly reduces the runtime and computational cost of BlackVIP. Extensive experiments on 19 datasets demonstrate that BlackVIPs enable robust adaptation to diverse domains and tasks with minimal memory requirements. We further provide a theoretical analysis on the generalization of visual prompting methods by presenting their connection to the certified robustness of randomized smoothing, and presenting an empirical support for improved robustness.
>
---
#### [replaced 018] Deterministic World Models for Verification of Closed-loop Vision-based Systems
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.08991](https://arxiv.org/pdf/2512.08991)**

> **作者:** Yuang Geng; Zhuoyang Zhou; Zhongzheng Zhang; Siyuan Pan; Hoang-Dung Tran; Ivan Ruchkin
>
> **备注:** Significantly revised version with additional experiments and updated results. Submitted to EMSOFT 2026
>
> **摘要:** Verifying closed-loop vision-based control systems remains a fundamental challenge due to the high dimensionality of images and the difficulty of modeling visual environments. While generative models are increasingly used as camera surrogates in verification, their reliance on stochastic latent variables introduces unnecessary overapproximation error. To address this bottleneck, we propose a Deterministic World Model (DWM) that maps system states directly to generative images, effectively eliminating uninterpretable latent variables to ensure precise input bounds. The DWM is trained with a dual-objective loss function that combines pixel-level reconstruction accuracy with a control difference loss to maintain behavioral consistency with the real system. We integrate DWM into a verification pipeline utilizing Star-based reachability analysis (StarV) and employ conformal prediction to derive rigorous statistical bounds on the trajectory deviation between the world model and the actual vision-based system. Experiments on standard benchmarks show that our approach yields significantly tighter reachable sets and better verification performance than a latent-variable baseline.
>
---
#### [replaced 019] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20985](https://arxiv.org/pdf/2602.20985)**

> **作者:** Munish Monga; Vishal Chudasama; Pankaj Wasnik; C.V. Jawahar
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Real-world object detection must operate in evolving environments where new classes emerge, domains shift, and unseen objects must be identified as "unknown": all without accessing prior data. We introduce Evolving World Object Detection (EWOD), a paradigm coupling incremental learning, domain adaptation, and unknown detection under exemplar-free constraints. To tackle EWOD, we propose EW-DETR framework that augments DETR-based detectors with three synergistic modules: Incremental LoRA Adapters for exemplar-free incremental learning under evolving domains; a Query-Norm Objectness Adapter that decouples objectness-aware features from DETR decoder queries; and Entropy-Aware Unknown Mixing for calibrated unknown detection. This framework generalises across DETR-based detectors, enabling state-of-the-art RF-DETR to operate effectively in evolving-world settings. We also introduce FOGS (Forgetting, Openness, Generalisation Score) to holistically evaluate performance across these dimensions. Extensive experiments on Pascal Series and Diverse Weather benchmarks show EW-DETR outperforms other methods, improving FOGS by 57.24%.
>
---
#### [replaced 020] Which Way Does Time Flow? A Psychophysics-Grounded Evaluation for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型的时序理解任务，旨在评估模型对视频时间方向的判断能力。通过构建基准测试，发现现有模型在时间推理上表现不佳，揭示了多模态系统在因果理解和时间连续性上的不足。**

- **链接: [https://arxiv.org/pdf/2510.26241](https://arxiv.org/pdf/2510.26241)**

> **作者:** Shiho Matta; Lis Kanashiro Pereira; Peitao Han; Fei Cheng; Shigeru Kitazawa
>
> **备注:** 12 pages
>
> **摘要:** Modern vision-language models (VLMs) excel at many multimodal tasks, yet their grasp of temporal information in video remains weak and has not been adequately evaluated. We probe this gap with a deceptively simple but revealing challenge: judging the arrow of time (AoT)-whether a short clip is played forward or backward. We introduce AoT-PsyPhyBENCH, a psychophysically validated benchmark that tests whether VLMs can infer temporal direction in natural videos using the same stimuli and behavioral baselines established for humans. Our comprehensive evaluation of open-weight and proprietary, reasoning and non-reasoning VLMs reveals that most models perform near chance, and even the best model lags far behind human accuracy on physically irreversible processes (e.g., free fall, diffusion/explosion) and causal manual actions (division/addition) that humans recognize almost instantly. These results highlight a fundamental gap in current multimodal systems: while they capture rich visual-semantic correlations, they lack the inductive biases required for temporal continuity and causal understanding. We release the code and data for AoT-PsyPhyBENCH to encourage further progress in the physical and temporal reasoning capabilities of VLMs.
>
---
#### [replaced 021] Structure is Supervision: Multiview Masked Autoencoders for Radiology
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.22294](https://arxiv.org/pdf/2511.22294)**

> **作者:** Sonia Laguna; Andrea Agostini; Alain Ryser; Samuel Ruiperez-Campillo; Irene Cannistraci; Moritz Vandenhirtz; Stephan Mandt; Nicolas Deperrois; Farhad Nooralahzadeh; Michael Krauthammer; Thomas M. Sutter; Julia E. Vogt
>
> **摘要:** Building robust medical machine learning systems requires pretraining strategies that exploit the intrinsic structure present in clinical data. We introduce Multiview Masked Autoencoder (MVMAE), a self-supervised framework that leverages the natural multi-view organization of radiology studies to learn view-invariant and disease-relevant representations. MVMAE combines masked image reconstruction with cross-view alignment, transforming clinical redundancy across projections into a powerful self-supervisory signal. We further extend this approach with MVMAE-V2T, which incorporates radiology reports as an auxiliary text-based learning signal to enhance semantic grounding while preserving fully vision-based inference. Evaluated on a downstream disease classification task on three large-scale public datasets, MIMIC-CXR, CheXpert, and PadChest, MVMAE consistently outperforms supervised and vision-language baselines. Furthermore, MVMAE-V2T provides additional gains, particularly in low-label regimes where structured textual supervision is most beneficial. Together, these results establish the importance of structural and textual supervision as complementary paths toward scalable, clinically grounded medical foundation models.
>
---
#### [replaced 022] Tackling Non-IIDness in HAPS-Aided Federated Learning
- **分类: cs.NI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2401.05308](https://arxiv.org/pdf/2401.05308)**

> **作者:** Amin Farajzadeh; Animesh Yadav; Halim Yanikomeroglu
>
> **备注:** Submitted to IEEE for possible publication
>
> **摘要:** High-altitude platform stations (HAPS) enable large-scale federated learning (FL) in non-terrestrial networks (NTN) by providing wide-area coverage and predominantly line-of-sight (LoS) connectivity to many ground users. However, practical deployments face heterogeneous and non-independently and identically distributed (non-IID) client data, which degrades accuracy and slows convergence. We propose a weighted attribute-based client selection strategy that leverages server-side indicators: historical traffic behavior, instantaneous channel quality, computational capability, and prior-round learning contribution. At each round, the HAPS computes a composite score and selects the top clients, while adapting attribute weights online based on their correlation with validation-loss improvement. We further provide theoretical justification that traffic-derived uniformity can serve as a proxy for latent data heterogeneity, enabling selection of client subsets with reduced expected non-IIDness. Simulations demonstrate improved test accuracy, faster convergence, and lower training loss compared with random, resource-only, and single-attribute baselines.
>
---
#### [replaced 023] MOON3.0: Reasoning-aware Multimodal Representation Learning for E-commerce Product Understanding
- **分类: cs.LG; cs.AI; cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2604.00513](https://arxiv.org/pdf/2604.00513)**

> **作者:** Junxian Wu; Chenghan Fu; Zhanheng Nie; Daoze Zhang; Bowen Wan; Wanxian Guan; Chuan Yu; Jian Xu; Bo Zheng
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** With the rapid growth of e-commerce, exploring general representations rather than task-specific ones has attracted increasing attention. Although recent multimodal large language models (MLLMs) have driven significant progress in product understanding, they are typically employed as feature extractors that implicitly encode product information into global embeddings, thereby limiting their ability to capture fine-grained attributes. Therefore, we argue that leveraging the reasoning capabilities of MLLMs to explicitly model fine-grained product attributes holds significant potential. Nevertheless, achieving this goal remains non-trivial due to several key challenges: (i) long-context reasoning tends to dilute the model's attention to salient information in the raw input; (ii) supervised fine-tuning (SFT) primarily encourages rigid imitation, limiting the exploration of effective reasoning strategies; and (iii) fine-grained details are progressively attenuated during forward propagation. To address these issues, we propose MOON3.0, the first reasoning-aware MLLM-based model for product representation learning. Our method (1) employs a multi-head modality fusion module to adaptively integrate raw signals; (2) incorporates a joint contrastive and reinforcement learning framework to autonomously explore more effective reasoning strategies; and (3) introduces a fine-grained residual enhancement module to progressively preserve local details throughout the network. Additionally, we release a large-scale multimodal e-commerce benchmark MBE3.0. Experimentally, our model demonstrates state-of-the-art zero-shot performance across various downstream tasks on both our benchmark and public datasets.
>
---
#### [replaced 024] Monocular Building Height Estimation from PhiSat-2 Imagery: Dataset and Method
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29245](https://arxiv.org/pdf/2603.29245)**

> **作者:** Yanjiao Song; Bowen Cai; Timo Balz; Zhenfeng Shao; Neema Simon Sumari; James Magidi; Walter Musakwa
>
> **摘要:** Monocular building height estimation from optical imagery is important for urban morphology characterization but remains challenging due to ambiguous height cues, large inter-city variations in building morphology, and the long-tailed distribution of building heights. PhiSat-2 is a promising open-access data source for this task because of its global coverage, 4.75 m spatial resolution, and seven-band spectral observations, yet its potential has not been systematically evaluated. To address this gap, we construct a PhiSat-2-Height dataset (PHDataset) and propose a Two-Stream Ordinal Network (TSONet). PHDataset contains 9,475 co-registered image-label patch pairs from 26 cities worldwide. TSONet jointly models footprint segmentation and height estimation, and introduces a Cross-Stream Exchange Module (CSEM) and a Feature-Enhanced Bin Refinement (FEBR) module for footprint-aware feature interaction and ordinal height refinement. Experiments on PHDataset show that TSONet achieves the best overall performance, reducing MAE and RMSE by 13.2% and 9.7%, and improving IoU and F1-score by 14.0% and 10.1% over the strongest competing results. Ablation studies further verify the effectiveness of CSEM, FEBR, and the joint use of ordinal regression and footprint assistance. Additional analyses and patch-level comparison with publicly available building height products indicate that PhiSat-2 benefits monocular building height estimation through its balanced combination of building-relevant spatial detail and multispectral observations. Overall, this study confirms the potential of PhiSat-2 for monocular building height estimation and provides a dedicated dataset and an effective method for future research.
>
---
#### [replaced 025] ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.11508](https://arxiv.org/pdf/2601.11508)**

> **作者:** Emily Steiner; Jianhao Zheng; Henry Howard-Jenkins; Chris Xie; Iro Armeni
>
> **备注:** CVPR 2026
>
> **摘要:** Indoor environments evolve as objects move, appear, or leave the scene. Capturing these dynamics requires maintaining temporally consistent instance identities across intermittently captured 3D scans, even when changes are unobserved. We introduce and formalize the task of temporally sparse 4D indoor semantic instance segmentation (SIS), which jointly segments, identifies, and temporally associates object instances. This setting poses a challenge for existing 3DSIS methods, which require a discrete matching step due to their lack of temporal reasoning, and for 4D LiDAR approaches, which perform poorly due to their reliance on high-frequency temporal measurements that are uncommon in the longer-horizon evolution of indoor environments. We propose ReScene4D, a novel method that adapts 3DSIS architectures for 4DSIS without needing dense observations. Our method enables temporal information sharing--using spatiotemporal contrastive loss, masking, and serialization--to adaptively leverage geometric and semantic priors across observations. This shared context enables consistent instance tracking and improves standard 3DSIS performance. To evaluate this task, we define a new metric, t-mAP, that extends mAP to reward temporal identity consistency. ReScene4D achieves state-of-the-art performance on the 3RScan dataset, establishing a new benchmark for understanding evolving indoor scenes.
>
---
#### [replaced 026] GPA-VGGT:Adapting VGGT to Large Scale Localization by Self-Supervised Learning with Geometry and Physics Aware Loss
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决VGGT模型在无标签场景下的适应问题。通过自监督学习和几何物理损失，提升其大尺度环境中的定位能力。**

- **链接: [https://arxiv.org/pdf/2601.16885](https://arxiv.org/pdf/2601.16885)**

> **作者:** Yangfan Xu; Lilian Zhang; Xiaofeng He; Pengdong Wu; Wenqi Wu; Jun Mao
>
> **摘要:** Transformer-based general visual geometry frameworks have shown promising performance in camera pose estimation and 3D scene understanding. Recent advancements in Visual Geometry Grounded Transformer (VGGT) models have shown great promise in camera pose estimation and 3D reconstruction. However, these models typically rely on ground truth labels for training, posing challenges when adapting to unlabeled and unseen scenes. In this paper, we propose a self-supervised framework to train VGGT with unlabeled data, thereby enhancing its localization capability in large-scale environments. To achieve this, we extend conventional pair-wise relations to sequence-wise geometric constraints for self-supervised learning. Specifically, in each sequence, we sample multiple source frames and geometrically project them onto different target frames, which improves temporal feature consistency. We formulate physical photometric consistency and geometric constraints as a joint optimization loss to circumvent the requirement for hard labels. By training the model with this proposed method, not only the local and global cross-view attention layers but also the camera and depth heads can effectively capture the underlying multi-view geometry. Experiments demonstrate that the model converges within hundreds of iterations and achieves significant improvements in large-scale localization. Our code will be released at this https URL.
>
---
#### [replaced 027] LaVR: Scene Latent Conditioned Generative Video Trajectory Re-Rendering using Large 4D Reconstruction Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.14674](https://arxiv.org/pdf/2601.14674)**

> **作者:** Mingyang Xie; Numair Khan; Tianfu Wang; Naina Dhingra; Seonghyeon Nam; Haitao Yang; Zhuo Hui; Christopher Metzler; Andrea Vedaldi; Hamed Pirsiavash; Lei Luo
>
> **摘要:** Given a monocular video, the goal of video re-rendering is to generate views of the scene from a novel camera trajectory. Existing methods face two distinct challenges. Geometrically unconditioned models lack spatial awareness, leading to drift and deformation under viewpoint changes. On the other hand, geometrically-conditioned models depend on estimated depth and explicit reconstruction, making them susceptible to depth inaccuracies and calibration errors. We propose to address these challenges by using the implicit geometric knowledge embedded in the latent space of a large 4D reconstruction model to condition the video generation process. These latents capture scene structure in a continuous space without explicit reconstruction. Therefore, they provide a flexible representation that allows the pretrained diffusion prior to regularize errors more effectively. By jointly conditioning on these latents and source camera poses, we demonstrate that our model achieves state-of-the-art results on the video re-rendering task. Project webpage is this https URL.
>
---
#### [replaced 028] FastSurfer-CC: A robust, accurate, and comprehensive framework for corpus callosum morphometry
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16471](https://arxiv.org/pdf/2511.16471)**

> **作者:** Clemens Pollak; Kersten Diers; Santiago Estrada; David Kügler; Martin Reuter
>
> **摘要:** The corpus callosum, the largest commissural structure in the human brain, is a central focus in research on aging and neurological diseases. It is also a critical target for interventions such as deep brain stimulation and serves as an important biomarker in clinical trials, including those investigating remyelination therapies. Despite extensive research on corpus callosum segmentation, few publicly available tools provide a comprehensive and automated analysis pipeline. To address this gap, we present FastSurfer-CC, an efficient and fully automated framework for corpus callosum morphometry. FastSurfer-CC automatically identifies mid-sagittal slices, segments the corpus callosum and fornix, localizes the anterior and posterior commissures to standardize head positioning, generates thickness profiles and subdivisions, and extracts eight shape metrics for statistical analysis. We demonstrate that FastSurfer-CC outperforms existing specialized tools across the individual tasks. Moreover, our method reveals statistically significant differences between Huntington's disease patients and healthy controls that are not detected by the current state-of-the-art.
>
---
#### [replaced 029] Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.01351](https://arxiv.org/pdf/2507.01351)**

> **作者:** Chaoxiang Cai; Longrong Yang; Minghe Weng; Xuewei Li; Zequn Qin; Xi Li
>
> **摘要:** The mixture-of-experts (MoE) architecture, which replaces dense networks with sparse ones, has attracted significant attention in large vision-language models (LVLMs) for achieving comparable performance while activating far fewer parameters. Existing MoE architectures for LVLMs primarily focus on token-to-expert routing (TER), encouraging different experts to specialize in processing specific tokens. However, these methods typically rely on the load balancing mechanism, neglecting the inherent distributional differences between vision and language modalities. To address this limitation, we propose the Long-Tailed Distribution-aware Router (LTDR) for vision-language TER, which tackles two key challenges: (1) Modality-specific distribution-aware routing. We observe that language TER generally follows a relatively uniform distribution, whereas vision TER exhibits a long-tailed distribution. This modality discrepancy motivates the design of specialized routing strategies for each modality. (2) Vision-specific dynamic expert activation. Recognizing the importance of high-information vision tail tokens, we introduce a data-augmentation-inspired strategy that increases the number of activated experts, ensuring sufficient learning for these rare but informative tokens. On vision-language and vision benchmarks, our approach achieves consistent improvements, boosting performance by 1.2% / 2.1% on vision-language and 1.6% on vision benchmarks.
>
---
#### [replaced 030] MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出MaskAdapt，解决物理角色的灵活运动适应问题。通过两阶段学习，实现部分身体动作的精准调整，提升运动鲁棒性与多样性。**

- **链接: [https://arxiv.org/pdf/2603.29272](https://arxiv.org/pdf/2603.29272)**

> **作者:** Soomin Park; Eunseong Lee; Kwang Bin Lee; Sung-Hee Lee
>
> **备注:** CVPR 2026
>
> **摘要:** We present MaskAdapt, a framework for flexible motion adaptation in physics-based humanoid control. The framework follows a two-stage residual learning paradigm. In the first stage, we train a mask-invariant base policy using stochastic body-part masking and a regularization term that enforces consistent action distributions across masking conditions. This yields a robust motion prior that remains stable under missing observations, anticipating later adaptation in those regions. In the second stage, a residual policy is trained atop the frozen base controller to modify only the targeted body parts while preserving the original behaviors elsewhere. We demonstrate the versatility of this design through two applications: (i) motion composition, where varying masks enable multi-part adaptation within a single sequence, and (ii) text-driven partial goal tracking, where designated body parts follow kinematic targets provided by a pre-trained text-conditioned autoregressive motion generator. Through experiments, MaskAdapt demonstrates strong robustness and adaptability, producing diverse behaviors under masked observations and delivering superior targeted motion adaptation compared to prior work.
>
---
#### [replaced 031] Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.23711](https://arxiv.org/pdf/2603.23711)**

> **作者:** Morui Zhu; Yongqi Zhu; Song Fu; Qing Yang
>
> **备注:** CVPR 2026 camera-ready version (minor revision & supplementary included)
>
> **摘要:** Autonomous trucking poses unique challenges due to articulated tractor-trailer geometry, and time-varying sensor poses caused by the fifth-wheel joint and trailer flex. Existing perception and calibration methods assume static baselines or rely on high-parallax and texture-rich scenes, limiting their reliability under real-world settings. We propose dCAP (dynamic Calibration and Articulated Perception), a vision-based framework that continuously estimates the 6-DoF (degree of freedom) relative pose between tractor and trailer cameras. dCAP employs a transformer with cross-view and temporal attention to robustly aggregate spatial cues while maintaining temporal consistency, enabling accurate perception under rapid articulation and occlusion. Integrated with BEVFormer, dCAP improves 3D object detection by replacing static calibration with dynamically predicted extrinsics. To facilitate evaluation, we introduce STT4AT, a CARLA-based benchmark simulating semi-trailer trucks with synchronized multi-sensor suites and time-varying inter-rig geometry across diverse environments. Experiments demonstrate that dCAP achieves stable, accurate perception while addressing the limitations of static calibration in autonomous trucking. The dataset, development kit, and source code will be publicly released.
>
---
#### [replaced 032] ThinkGeo: Evaluating Tool-Augmented Agents for Remote Sensing Tasks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.23752](https://arxiv.org/pdf/2505.23752)**

> **作者:** Akashah Shabbir; Muhammad Akhtar Munir; Akshay Dudhane; Muhammad Umer Sheikh; Muhammad Haris Khan; Paolo Fraccaro; Juan Bernabe Moreno; Fahad Shahbaz Khan; Salman Khan
>
> **摘要:** Recent progress in large language models (LLMs) has enabled tool-augmented agents capable of solving complex real-world tasks through step-by-step reasoning. However, existing evaluations often focus on general-purpose or multimodal scenarios, leaving a gap in domain-specific benchmarks that assess tool-use capabilities in complex remote sensing use cases. We present ThinkGeo, an agentic benchmark designed to evaluate LLM-driven agents on remote sensing tasks via structured tool use and multi-step planning. Inspired by tool-interaction paradigms, ThinkGeo includes human-curated queries spanning a wide range of real-world applications such as urban planning, disaster assessment and change analysis, environmental monitoring, transportation analysis, aviation monitoring, recreational infrastructure, and industrial site analysis. Queries are grounded in satellite or aerial imagery, including both optical RGB and SAR data, and require agents to reason through a diverse toolset. We implement a ReAct-style interaction loop and evaluate both open and closed-source LLMs (e.g., GPT-4o, Qwen2.5) on 486 structured agentic tasks with 1,778 expert-verified reasoning steps. The benchmark reports both step-wise execution metrics and final answer correctness. Our analysis reveals notable disparities in tool accuracy and planning consistency across models. ThinkGeo provides the first extensive testbed for evaluating how tool-enabled LLMs handle spatial reasoning in remote sensing.
>
---
#### [replaced 033] ScrollScape: Unlocking 32K Image Generation With Video Diffusion Priors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24270](https://arxiv.org/pdf/2603.24270)**

> **作者:** Haodong Yu; Yabo Zhang; Donglin Di; Ruyi Zhang; Wangmeng Zuo
>
> **摘要:** While diffusion models excel at generating images with conventional dimensions, pushing them to synthesize ultra-high-resolution imagery at extreme aspect ratios (EAR) often triggers catastrophic structural failures, such as object repetition and spatial fragmentation. This limitation fundamentally stems from a lack of robust spatial priors, as static text-to-image models are primarily trained on image distributions with conventional dimensions. To overcome this bottleneck, we present ScrollScape, a novel framework that reformulates EAR image synthesis into a continuous video generation process through two core innovations. By mapping the spatial expansion of a massive canvas to the temporal evolution of video frames, ScrollScape leverages the inherent temporal consistency of video models as a powerful global constraint to ensure long-range structural integrity. Specifically, Scanning Positional Encoding (ScanPE) distributes global coordinates across frames to act as a flexible moving camera, while Scrolling Super-Resolution (ScrollSR) leverages video super-resolution priors to circumvent memory bottlenecks, efficiently scaling outputs to an unprecedented 32K resolution. Fine-tuned on a curated 3K multi-ratio image dataset, ScrollScape effectively aligns pre-trained video priors with the EAR generation task. Extensive evaluations demonstrate that it significantly outperforms existing image-diffusion baselines by eliminating severe localized artifacts. Consequently, our method overcomes inherent structural bottlenecks to ensure exceptional global coherence and visual fidelity across diverse domains at extreme scales.
>
---
#### [replaced 034] Aleatoric Uncertainty Medical Image Segmentation Estimation via Flow Matching
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.22418](https://arxiv.org/pdf/2507.22418)**

> **作者:** Phi Van Nguyen; Ngoc Huynh Trinh; Duy Minh Lam Nguyen; Phu Loc Nguyen; Quoc Long Tran
>
> **摘要:** Quantifying aleatoric uncertainty in medical image segmentation is critical since it is a reflection of the natural variability observed among expert annotators. A conventional approach is to model the segmentation distribution using the generative model, but current methods limit the expression ability of generative models. While current diffusion-based approaches have demonstrated impressive performance in approximating the data distribution, their inherent stochastic sampling process and inability to model exact densities limit their effectiveness in accurately capturing uncertainty. In contrast, our proposed method leverages conditional flow matching, a simulation-free flow-based generative model that learns an exact density, to produce highly accurate segmentation results. By guiding the flow model on the input image and sampling multiple data points, our approach synthesizes segmentation samples whose pixel-wise variance reliably reflects the underlying data distribution. This sampling strategy captures uncertainties in regions with ambiguous boundaries, offering robust quantification that mirrors inter-annotator differences. Experimental results demonstrate that our method not only achieves competitive segmentation accuracy but also generates uncertainty maps that provide deeper insights into the reliability of the segmentation outcomes. The code for this paper is freely available at this https URL
>
---
#### [replaced 035] Adaptive Reinforcement for Open-ended Medical Reasoning via Semantic-Guided Reward Collapse Mitigation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12957](https://arxiv.org/pdf/2508.12957)**

> **作者:** Yizhou Liu; Dingkang Yang; Zizhi Chen; Minghao Han; Xukun Zhang; Keliang Liu; Jingwei Wei; Lihua Zhang
>
> **备注:** Accept to 2026 CVPR Findings
>
> **摘要:** Reinforcement learning (RL) with rule-based reward functions has recently shown great promise in enhancing the reasoning depth and generalization ability of vision-language models (VLMs), while maintaining computational efficiency. In spite of these advances, its adoption in medical imaging remains limited. Current reinforcement fine-tuning (RFT) efforts in this field mainly focus on closed-ended visual question answering (VQA), restricting their applicability to realistic clinical reasoning. However, open-ended medical VQA better mirrors clinical diagnostic workflows but remains underexplored. Although several studies have attempted to bridge the two formats through semantically guided RL, model-driven semantic rewards often suffer from reward collapse, where responses with distinct semantics yield nearly identical scores. To overcome this limitation, we introduce Adaptive Reinforcement for Medical Reasoning (ARMed), a novel RL framework tailored for open-ended medical VQA. ARMed first injects domain expertise through supervised fine-tuning (SFT) on chain-of-thought annotations, followed by reinforcement optimization using textual correctness and adaptive semantic rewards to refine reasoning consistency and factual accuracy. Extensive experiments on six challenging medical VQA benchmarks demonstrate that ARMed substantially improves both accuracy and generalization. These findings underscore the importance of reward discriminability in medical RL and highlight the potential of adaptive semantic rewards for building robust, clinically reliable multimodal reasoning systems.
>
---
#### [replaced 036] Cryo-Bench: Benchmarking Foundation Models for Cryosphere Applications
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.01576](https://arxiv.org/pdf/2603.01576)**

> **作者:** Saurabh Kaushik; Lalit Maurya; Beth Tellman
>
> **摘要:** Geo-Foundation Models (GFMs) have been evaluated across diverse Earth observation task including multiple domains and have demonstrated strong potential of producing reliable maps even with sparse labels. However, benchmarking GFMs for Cryosphere applications has remained limited, primarily due to the lack of suitable evaluation datasets. To address this gap, we introduce \textbf{Cryo-Bench}, a benchmark compiled to evaluate GFM performance across key Cryospheric components. Cryo-Bench includes debris-covered glaciers, glacial lakes, sea ice, and calving fronts, spanning multiple sensors and broad geographic regions. We evaluate 14 GFMs alongside UNet and ViT baselines to assess their advantages, limitations, and optimal usage strategies. With a frozen encoder, UNet achieves the highest average mIoU of \textbf{66.38}, followed by TerraMind at \textbf{64.02} across five evluation dataset included in Cryo-Bench. In the few-shot setting (10\% input data), GFMs such as DOFA and TerraMind outperform UNet, achieving mIoU scores of \textbf{59.53}, \textbf{56.62}, and \textbf{56.60}, respectively, comapred to U-Net's 56.60. When fully finetuning GFMs, we observe inconsistent performance across datasets and models. However, tuning learning rate along with finetuning substantially improves GFM performance. For example, evaluation on two representative datasets (GLID and CaFFe) shows an average relative improvement of \textbf{12.77\%}. Despite having minimal Cryosphere representation in their pretraining data, GFMs exhibit notable domain adaptation capabilities and produce meaningful results across tasks. Based on our findings, We recommend encoder fine-tuning with hyperparameter optimization optimization to achieve the best possible performance, while using frozen encoders when users need quick results without extensive experimentation.(\href{this https URL}{GitHub}).
>
---
#### [replaced 037] When Surfaces Lie: Exploiting Wrinkle-Induced Attention Shift to Attack Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27759](https://arxiv.org/pdf/2603.27759)**

> **作者:** Chengyin Hu; Xuemeng Sun; Jiajun Han; Qike Zhang; Xiang Chen; Xin Wang; Yiwei Wei; Jiahua Long
>
> **摘要:** Visual-Language Models (VLMs) have demonstrated exceptional cross-modal understanding across various tasks, including zero-shot classification, image captioning, and visual question answering. However, their robustness to physically plausible non-rigid deformations-such as wrinkles on flexible surfaces-remains poorly understood. In this work, we propose a parametric structural perturbation method inspired by the mechanics of three-dimensional fabric wrinkles. Specifically, our method generates photorealistic non-rigid perturbations by constructing multi-scale wrinkle fields and integrating displacement field distortion with surface-consistent appearance variations. To achieve an optimal balance between visual naturalness and adversarial effectiveness, we design a hierarchical fitness function in a low-dimensional parameter space and employ an optimization-based search strategy. We evaluate our approach using a two-stage framework: perturbations are first optimized on a zero-shot classification proxy task and subsequently assessed for transferability on generative tasks. Experimental results demonstrate that our method significantly degrades the performance of various state-of-the-art VLMs, consistently outperforming baselines in both image captioning and visual question-answering tasks.
>
---
#### [replaced 038] AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.17979](https://arxiv.org/pdf/2603.17979)**

> **作者:** Jinho Park; Se Young Chun; Mingoo Seok
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** Radar is a critical perception modality in autonomous driving systems due to its all-weather characteristics and ability to measure range and Doppler velocity. However, the sheer volume of high-dimensional raw radar data saturates the communication link to the computing engine (e.g., an NPU), which is often a low-bandwidth interface with data rate provisioned only for a few low-resolution range-Doppler frames. A generalized codec for utilizing high-dimensional radar data is notably absent, while existing image-domain approaches are unsuitable, as they typically operate at fixed compression ratios and fail to adapt to varying or adversarial conditions. In light of this, we propose radar data compression with adaptive feedback. It dynamically adjusts the compression ratio by performing gradient descent from the proxy gradient of detection confidence with respect to the compression rate. We employ a zeroth-order gradient approximation as it enables gradient computation even with non-differentiable core operations--pruning and quantization. This also avoids transmitting the gradient tensors over the band-limited link, which, if estimated, would be as large as the original radar data. In addition, we have found that radar feature maps are heavily concentrated on a few frequency components. Thus, we apply the discrete cosine transform to the radar data cubes and selectively prune out the coefficients effectively. We preserve the dynamic range of each radar patch through scaled quantization. Combining those techniques, our proposed online adaptive compression scheme achieves over 100x feature size reduction at minimal performance drop (~1%p). We validate our results on the RADIal, CARRADA, and Radatron datasets.
>
---
#### [replaced 039] Assessing Multimodal Chronic Wound Embeddings with Expert Triplet Agreement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29376](https://arxiv.org/pdf/2603.29376)**

> **作者:** Fabian Kabus; Julia Hindel; Jelena Bratulić; Meropi Karakioulaki; Ayush Gupta; Cristina Has; Thomas Brox; Abhinav Valada; Harald Binder
>
> **摘要:** Recessive dystrophic epidermolysis bullosa (RDEB) is a rare genetic skin disorder for which clinicians greatly benefit from finding similar cases using images and clinical text. However, off-the-shelf foundation models do not reliably capture clinically meaningful features for this heterogeneous, long-tail disease, and structured measurement of agreement with experts is challenging. To address these gaps, we propose evaluating embedding spaces with expert ordinal comparisons (triplet judgments), which are fast to collect and encode implicit clinical similarity knowledge. We further introduce TriDerm, a multimodal framework that learns interpretable wound representations from small cohorts by integrating wound imagery, boundary masks, and expert reports. On the vision side, TriDerm adapts visual foundation models to RDEB using wound-level attention pooling and non-contrastive representation learning. For text, we prompt large language models with comparison queries and recover medically meaningful representations via soft ordinal embeddings (SOE). We show that visual and textual modalities capture complementary aspects of wound phenotype, and that fusing both modalities yields 73.5% agreement with experts, outperforming the best off-the-shelf single-modality foundation model by over 5.6 percentage points. We make the expert annotation tool, model code and representative dataset samples publicly available.
>
---
#### [replaced 040] Learning Fine-Grained Geometry for Sparse-View Splatting via Cascade Depth Loss
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.22279](https://arxiv.org/pdf/2505.22279)**

> **作者:** Wenjun Lu; Haodong Chen; Anqi Yi; Guoxi Huang; Yuk Ying Chung; Kun Hu; Zhiyong Wang
>
> **摘要:** Novel view synthesis is a fundamental task in 3D computer vision that aims to reconstruct photorealistic images from novel viewpoints given a set of posed images. However, reconstruction quality degrades sharply under sparse-view conditions due to insufficient geometric cues. Existing methods, including Neural Radiance Fields (NeRF) and more recent 3D Gaussian Splatting (3DGS), often exhibit blurred details and structural artifacts when trained from sparse observations. Recent works have identified rendered depth quality as a key factor in mitigating these artifacts, as it directly affects geometric accuracy and view consistency. However, effectively leveraging depth under sparse views remains challenging. Depth priors can be noisy or misaligned with rendered geometry, and single-scale supervision often fails to capture both global structure and fine details. To address these challenges, we introduce Hierarchical Depth-Guided Splatting (HDGS), a depth supervision framework that progressively refines geometry from coarse to fine levels. Central to HDGS is our novel Cascade Pearson Correlation Loss (CPCL), which enforces consistency between rendered and estimated depth priors across multiple spatial scales. By enforcing multi-scale depth consistency, our method improves structural fidelity in sparse-view reconstruction. Experiments on LLFF and DTU demonstrate state-of-the-art performance under sparse-view settings.
>
---
#### [replaced 041] Improvise, Adapt, Overcome -- Telescopic Adapters for Efficient Fine-tuning of Vision Language Models in Medical Imaging
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.13855](https://arxiv.org/pdf/2512.13855)**

> **作者:** Ujjwal Mishra; Vinita Shukla; Praful Hambarde; Amit Shukla
>
> **备注:** Accepted at the IEEE/CVF winter conference on applications of computer vision (WACV 2026)
>
> **摘要:** Adapting Vision Language Segmentation Models (VLSMs) to medical imaging domains requires significant computational overhead when using conventional fine-tuning approaches. Existing Parameter-Efficient Fine-Tuning (PEFT) methods apply uniform adapter dimensions across all transformer layers, leading to suboptimal parameter allocation and reduced adaptation efficiency. We introduce Telescopic Adapters, a novel PEFT framework that employs depth-aware scaling to progressively increase adapter capacity from shallow to deep transformer layers. Our method integrates lightweight bottleneck modules within CLIPSeg's vision and text encoders, with adapter dimensions dynamically scaled based on layer depth and semantic relevance. Using only 613k trainable parameters--244x fewer than end-to-end fine-tuning, Telescopic Adapters achieve superior performance across five diverse medical datasets spanning polyp segmentation, skin lesion detection, and breast ultrasound imaging. Comprehensive ablation studies demonstrate that deeper layers require substantially more adaptation capacity than shallow layers, validating our telescopic scaling hypothesis. Our approach establishes a new paradigm for efficient medical VLSM fine-tuning, enabling deployment in resource-constrained clinical environments while maintaining competitive segmentation accuracy. Our source code is publicly available at this https URL
>
---
#### [replaced 042] Bias Is a Subspace, Not a Coordinate: A Geometric Rethinking of Post-hoc Debiasing in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言模型的公平性研究，解决模型中的偏见问题。通过几何方法识别并去除偏见子空间，提升模型公平性。**

- **链接: [https://arxiv.org/pdf/2511.18123](https://arxiv.org/pdf/2511.18123)**

> **作者:** Dachuan Zhao; Weiyue Li; Zhenda Shen; Yushu Qiu; Bowen Xu; Haoyu Chen; Yongchao Chen
>
> **备注:** Accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Vision-Language Models (VLMs) have become indispensable for multimodal reasoning, yet their representations often encode and amplify demographic biases, resulting in biased associations and misaligned predictions in downstream tasks. Such behavior undermines fairness and distorts the intended alignment between vision and language. Recent post-hoc approaches attempt to mitigate bias by replacing the most attribute-correlated embedding coordinates with neutral values. However, our systematic analysis reveals three critical limitations of this coordinate-wise approach: feature entanglement, poor cross-dataset generalization, and incomplete bias removal. We find that bias is not localized to a few coordinates but is instead distributed across a few linear subspaces. To address these limitations, we propose $\textbf{S}$ubspace $\textbf{P}$rojection $\textbf{D}$ebiasing ($\textbf{SPD}$), a geometrically principled framework that identifies and removes the entire subspace of linearly decodable bias while reinserting a neutral mean component to preserve semantic fidelity. Extensive experiments across zero-shot classification, text-to-image retrieval, and image generation validate the effectiveness of SPD: our method achieves more robust debiasing with an average improvement of $18.5\%$ across four fairness metrics, while maintaining minimal loss in task performance compared to the best debiasing baseline.
>
---
#### [replaced 043] Fourier Splatting: Generalized Fourier encoded primitives for scalable radiance fields
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.19834](https://arxiv.org/pdf/2603.19834)**

> **作者:** Mihnea-Bogdan Jurca; Bert Van hauwermeiren; Adrian Munteanu
>
> **摘要:** Novel view synthesis has recently been revolutionized by 3D Gaussian Splatting (3DGS), which enables real-time rendering through explicit primitive rasterization. However, existing methods tie visual fidelity strictly to the number of primitives: quality downscaling is achieved only through pruning primitives. We propose the first inherently scalable primitive for radiance field rendering. Fourier Splatting employs scalable primitives with arbitrary closed shapes obtained by parameterizing planar surfels with Fourier encoded descriptors. This formulation allows a single trained model to be rendered at varying levels of detail simply by truncating Fourier coefficients at runtime. To facilitate stable optimization, we employ a straight-through estimator for gradient extension beyond the primitive boundary, and introduce HYDRA, a densification strategy that decomposes complex primitives into simpler constituents within the MCMC framework. Our method achieves state-of-the-art rendering quality among planar-primitive frameworks and comparable perceptual metrics compared to leading volumetric representations on standard benchmarks, providing a versatile solution for bandwidth-constrained high-fidelity rendering.
>
---
#### [replaced 044] Maximally Useful and Minimally Redundant: The Key to Self Supervised Learning for Imbalanced Data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.08469](https://arxiv.org/pdf/2509.08469)**

> **作者:** Yash Kumar Sharma; Vineet Padmanabhan
>
> **摘要:** Contrastive self supervised learning(CSSL) usually makes use of the multi-view assumption which states that all relevant information must be shared between all views. The main objective of CSSL is to maximize the mutual information(MI) between representations of different views and at the same time compress irrelevant information in each representation. Recently, as part of future work, Schwartz Ziv & Yan LeCun pointed out that, when the multi-view assumption is violated, one of the most significant challenges in SSL is in identifying new methods to separate relevant from irrelevant information based on alternative assumptions. Taking a cue from this intuition we make the following contributions in this paper: 1) We develop a CSSL framework wherein multiple images and multiple views(MIMV) are considered as input, which is different from the traditional multi-view assumption 2) We adopt a novel augmentation strategy that includes both normalized (invertible) and augmented (non-invertible) views so that complete information of one image can be preserved and hard augmentation can be chosen for the other image 3) An Information bottleneck(IB) principle is outlined for MIMV to produce optimal representations 4) We introduce a loss function that helps to learn better representations by filtering out extreme features 5) The robustness of our proposed framework is established by applying it to the imbalanced dataset problem wherein we achieve a new state-of-the-art accuracy (2% improvement in Cifar10-LT using Resnet-18, 5% improvement in Cifar100-LT using Resnet-18 and 3% improvement in Imagenet-LT (1k) using Resnet-50).
>
---
#### [replaced 045] Towards Physically Realizable Adversarial Attenuation Patch against SAR Object Detection
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2604.00887](https://arxiv.org/pdf/2604.00887)**

> **作者:** Yiming Zhang; Weibo Qin; Feng Wang
>
> **备注:** 5 pages, 4 figures. Source code is available at this https URL
>
> **摘要:** Deep neural networks have demonstrated excellent performance in SAR target detection tasks but remain susceptible to adversarial attacks. Existing SAR-specific attack methods can effectively deceive detectors; however, they often introduce noticeable perturbations and are largely confined to digital domain, neglecting physical implementation constrains for attacking SAR systems. In this paper, a novel Adversarial Attenuation Patch (AAP) method is proposed that employs energy-constrained optimization strategy coupled with an attenuation-based deployment framework to achieve a seamless balance between attack effectiveness and stealthiness. More importantly, AAP exhibits strong potential for physical realization by aligning with signal-level electronic jamming mechanisms. Experimental results show that AAP effectively degrades detection performance while preserving high imperceptibility, and shows favorable transferability across different models. This study provides a physical grounded perspective for adversarial attacks on SAR target detection systems and facilitates the design of more covert and practically deployable attack strategies. The source code is made available at this https URL.
>
---
#### [replaced 046] RehearsalNeRF: Decoupling Intrinsic Neural Fields of Dynamic Illuminations for Scene Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27948](https://arxiv.org/pdf/2603.27948)**

> **作者:** Changyeon Won; Hyunjun Jung; Jungu Cho; Seonmi Park; Chi-Hoon Lee; Hae-Gon Jeon
>
> **备注:** Accepted to the International Journal of Computer Vision (IJCV). Changyeon Won and Hyunjun Jung contributed equally to this work
>
> **摘要:** Although there has been significant progress in neural radiance fields, an issue on dynamic illumination changes still remains unsolved. Different from relevant works that parameterize time-variant/-invariant components in scenes, subjects' radiance is highly entangled with their own emitted radiance and lighting colors in spatio-temporal domain. In this paper, we present a new effective method to learn disentangled neural fields under the severe illumination changes, named RehearsalNeRF. Our key idea is to leverage scenes captured under stable lighting like rehearsal stages, easily taken before dynamic illumination occurs, to enforce geometric consistency between the different lighting conditions. In particular, RehearsalNeRF employs a learnable vector for lighting effects which represents illumination colors in a temporal dimension and is used to disentangle projected light colors from scene radiance. Furthermore, our RehearsalNeRF is also able to reconstruct the neural fields of dynamic objects by simply adopting off-the-shelf interactive masks. To decouple the dynamic objects, we propose a new regularization leveraging optical flow, which provides coarse supervision for the color disentanglement. We demonstrate the effectiveness of RehearsalNeRF by showing robust performances on novel view synthesis and scene editing under dynamic illumination conditions. Our source code and video datasets will be publicly available.
>
---
#### [replaced 047] PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis
- **分类: cs.GR; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.02794](https://arxiv.org/pdf/2506.02794)**

> **作者:** Mijeong Kim; Gunhee Kim; Jungyoon Choi; Wonjae Roh; Bohyung Han
>
> **备注:** Accepted at CVPR 2026; Project page: this http URL Dataset: this https URL
>
> **摘要:** We introduce PhysGaia, a novel physics-aware benchmark for Dynamic Novel View Synthesis (DyNVS) that encompasses both structured objects and unstructured physical phenomena. While existing datasets primarily focus on photorealistic appearance, PhysGaia is specifically designed to support physics-consistent dynamic reconstruction. Our benchmark features complex scenarios with rich multi-body interactions, where objects realistically collide and exchange forces. Furthermore, it incorporates a diverse range of materials, including liquid, gas, textile, and rheological substance, moving beyond the rigid-body assumptions prevalent in prior work. To ensure physical fidelity, all scenes in PhysGaia are generated using material-specific physics solvers that strictly adhere to fundamental physical laws. We provide comprehensive ground-truth information, including 3D particle trajectories and physical parameters (e.g., viscosity), enabling the quantitative evaluation of physical modeling. To facilitate research adoption, we also provide integration pipelines for recent 4D Gaussian Splatting models along with our dataset and their results. By addressing the critical shortage of physics-aware benchmarks, PhysGaia can significantly advance research in dynamic view synthesis, physics-based scene understanding, and the integration of deep learning with physical simulation, ultimately enabling more faithful reconstruction and interpretation of complex dynamic scenes.
>
---
#### [replaced 048] MS-Mix: Sentiment-Guided Adaptive Augmentation for Multimodal Sentiment Analysis
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.11579](https://arxiv.org/pdf/2510.11579)**

> **作者:** Hongyu Zhu; Lin Chen; Xin Jin; Mingsheng Shang
>
> **备注:** Under Review
>
> **摘要:** Multimodal Sentiment Analysis (MSA) integrates complementary features from text, video, and audio for robust emotion understanding in human interactions. However, models suffer from severe data scarcity and high annotation costs, severely limiting real-world deployment in social media analytics and human-computer systems. Existing Mixup-based augmentation techniques, when naively applied to MSA, often produce semantically inconsistent samples and amplified label noise by ignoring emotional semantics across modalities. To address these challenges, we propose MS-Mix, an adaptive emotion-sensitive augmentation framework that automatically optimizes data quality in multimodal settings. Its key components are: (1) Sentiment-aware sample selection strategy that filters incompatible pairs via latent-space semantic similarity to prevent contradictory emotion mixing. (2) Sentiment intensity guided module with multi-head self-attention for computing modality-specific mixing ratios conditioned on emotional salience dynamically. (3) Sentiment alignment loss based on Kullback-Leibler divergence to align predicted sentiment distributions across modalities with ground-truth labels, improving discrimination and consistency. Extensive experiments on two public datasets with six state-of-the-art backbones confirm that MS-Mix consistently outperforms prior methods, significantly improving robustness and practical applicability for MSA. The source code is available at an anonymous link: this https URL.
>
---
#### [replaced 049] A Luminance-Aware Multi-Scale Network for Polarization Image Fusion with a Multi-Scene Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.24379](https://arxiv.org/pdf/2510.24379)**

> **作者:** Zhuangfan Huang; Xiaosong Li; Gao Wang; Tao Ye; Haishu Tan; Huafeng Li
>
> **摘要:** Polarization image fusion combines S0 and DOLP images to reveal surface roughness and material properties through complementary texture features, which has important applications in camouflage recognition, tissue pathology analysis, surface defect detection and other fields. To intergrate coL-Splementary information from different polarized images in complex luminance environment, we propose a luminance-aware multi-scale network (MLSN). In the encoder stage, we propose a multi-scale spatial weight matrix through a brightness-branch , which dynamically weighted inject the luminance into the feature maps, solving the problem of inherent contrast difference in polarized images. The global-local feature fusion mechanism is designed at the bottleneck layer to perform windowed self-attention computation, to balance the global context and local details through residual linking in the feature dimension restructuring stage. In the decoder stage, to further improve the adaptability to complex lighting, we propose a Brightness-Enhancement module, establishing the mapping relationship between luminance distribution and texture features, realizing the nonlinear luminance correction of the fusion result. We also present MSP, an 1000 pairs of polarized images that covers 17 types of indoor and outdoor complex lighting scenes. MSP provides four-direction polarization raw maps, solving the scarcity of high-quality datasets in polarization image fusion. Extensive experiment on MSP, PIF and GAND datasets verify that the proposed MLSN outperms the state-of-the-art methods in subjective and objective evaluations, and the MS-SSIM and SD metircs are higher than the average values of other methods by 8.57%, 60.64%, 10.26%, 63.53%, 22.21%, and 54.31%, respectively. The source code and dataset is avalable at this https URL.
>
---
#### [replaced 050] Learning to Translate Noise for Robust Image Denoising
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2412.04727](https://arxiv.org/pdf/2412.04727)**

> **作者:** Inju Ha; Donghun Ryou; Seonguk Seo; Bohyung Han
>
> **备注:** Project page: this https URL Accepted to CVPR 2026 Findings
>
> **摘要:** Deep learning-based image denoising techniques often struggle with poor generalization performance to out-of-distribution real-world noise. To tackle this challenge, we propose a novel noise translation framework that performs denoising on an image with translated noise rather than directly denoising an original noisy image. Specifically, our approach translates complex, unknown real-world noise into Gaussian noise, which is spatially uncorrelated and independent of image content, through a noise translation network. The translated noisy images are then processed by an image denoising network pretrained to effectively remove Gaussian noise, enabling robust and consistent denoising performance. We also design well-motivated loss functions and architectures for the noise translation network by leveraging the mathematical properties of Gaussian noise. Experimental results demonstrate that the proposed method substantially improves robustness and generalizability, outperforming state-of-the-art methods across diverse benchmarks. Visualized denoising results and the source code are available on our project page.
>
---
#### [replaced 051] SALAD: Achieve High-Sparsity Attention via Efficient Linear Attention Tuning for Video Diffusion Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.16515](https://arxiv.org/pdf/2601.16515)**

> **作者:** Tongcheng Fang; Hanling Zhang; Ruiqi Xie; Zhuo Han; Xin Tao; Tianchen Zhao; Pengfei Wan; Wenbo Ding; Wanli Ouyang; Xuefei Ning; Yu Wang
>
> **摘要:** Diffusion Transformers have demonstrated remarkable performance in video generation. However, their long input sequences incur substantial latency due to the quadratic complexity of full attention. Various sparse attention mechanisms have been proposed. Training-free approaches are limited to moderate sparsity and thus yield only modest acceleration, whereas training-based methods can reach much higher sparsity but demand substantial data and computation. In this work, we propose SALAD, introducing a lightweight linear attention branch in parallel with the sparse attention. Leveraging a Multi-level Static-Dynamic Scaling Strategy to balance the two branches, our method attains up to 90% sparsity and 1.52-2.03x inference speedup across different models and sequence lengths, while maintaining generation quality comparable to the full attention baseline. Moreover, our finetuning process is highly efficient, requiring only 2,000 video samples, fewer than 1,600 training steps, and no more than 30 GPU hours with a batch size of 8.
>
---
#### [replaced 052] SPAGS: Sparse-View Articulated Object Reconstruction from Single State via Planar Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17092](https://arxiv.org/pdf/2511.17092)**

> **作者:** Di Wu; Liu Liu; Xueyu Yuan; Wenxiao Chen; Lijun Yue; Liuzhu Chen; Yiming Tang; Meng Wang
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** Articulated objects are ubiquitous in daily environments, and their 3D reconstruction holds great significance across various fields. However, existing articulated object reconstruction methods typically require costly inputs such as multi-stage and multi-view observations. To address the limitations, we propose a category-agnostic articulated object reconstruction framework via planar Gaussian Splatting, which only uses sparse-view RGB images from a single state. Specifically, we first introduce a Gaussian information field to perceive the optimal sparse viewpoints from candidate camera poses. To ensure precise geometric fidelity, we constrain traditional 3D Gaussians into planar primitives, facilitating accurate normal and depth estimation. The planar Gaussians are then optimized in a coarse-to-fine manner, regularized by depth smoothness and few-shot diffusion priors. Furthermore, we leverage a Vision-Language Model (VLM) via visual prompting to achieve open-vocabulary part segmentation and joint parameter estimation. Extensive experiments on both synthetic and real-world datasets demonstrate that our approach significantly outperforms existing baselines, achieving superior part-level surface reconstruction fidelity. Code and data are provided in the supplementary material.
>
---
#### [replaced 053] Catalogue Grounded Multimodal Attribution for Museum Video under Resource and Regulatory Constraints
- **分类: cs.MM; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.11147](https://arxiv.org/pdf/2603.11147)**

> **作者:** Minsak Nanang; Adrian Hilton; Armin Mustafa
>
> **备注:** Demo video url: this https URL
>
> **摘要:** Audiovisual (AV) archives in museums and galleries are growing rapidly, but much of this material remains effectively locked away because it lacks consistent, searchable metadata. Existing method for archiving requires extensive manual effort. We address this by automating the most labour intensive part of the workflow: catalogue style metadata curation for in gallery video, grounded in an existing collection database. Concretely, we propose catalogue-grounded multimodal attribution for museum AV content using an open, locally deployable video language model. We design a multi pass pipeline that (i) summarises artworks in a video, (ii) generates catalogue style descriptions and genre labels, and (iii) attempts to attribute title and artist via conservative similarity matching to the structured catalogue. Early deployments on a painting catalogue suggest that this framework can improve AV archive discoverability while respecting resource constraints, data sovereignty, and emerging regulation, offering a transferable template for application-driven machine learning in other high-stakes domains.
>
---
#### [replaced 054] Fair Benchmarking of Emerging One-Step Generative Models Against Multistep Diffusion and Flow Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.14186](https://arxiv.org/pdf/2603.14186)**

> **作者:** Advaith Ravishankar; Serena Liu; Mingyang Wang; Todd Zhou; Jeffrey Zhou; Arnav Sharma; Ziling Hu; Léopold Das; Abdulaziz Sobirov; Faizaan Siddique; Freddy Yu; Seungjoo Baek; Yan Luo; Mengyu Wang
>
> **摘要:** State-of-the-art text-to-image models produce high-quality images, but inference remains expensive as generation requires several sequential ODE or denoising steps. Native one-step models aim to reduce this cost by mapping noise to an image in a single step, yet fair comparisons to multi-step systems are difficult because studies use mismatched sampling steps and different classifier-free guidance (CFG) settings, where CFG can shift FID, Inception Score, and CLIP-based alignment in opposing directions. It is also unclear how well one-step models scale to multi-step inference, and there is limited standardized out-of-distribution evaluation for label-ID-conditioned generators beyond ImageNet. To address this, We benchmark eight models spanning one-step flows (MeanFlow, Improved MeanFlow, SoFlow), multi-step baselines (RAE, Scale-RAE), and established systems (SiT, Stable Diffusion 3.5, FLUX.1) under a controlled class-conditional protocol on ImageNet validation, ImageNetV2, and reLAIONet, our new proofread out-of-distribution dataset aligned to ImageNet label IDs. Using FID, Inception Score, CLIP Score, and Pick Score, we show that FID-focused model development and CFG selection can be misleading in few-step regimes, where guidance changes can improve FID while degrading text-image alignment and human preference signals and worsening perceived quality. We further show that leading one-step models benefit from step scaling and become substantially more competitive under multi-step inference, although they still exhibit characteristic local distortions. To capture these tradeoffs, we introduce MinMax Harmonic Mean (MMHM), a composite proxy over all four metrics that stabilizes hyperparameter selection across guidance and step sweeps.
>
---
#### [replaced 055] FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.23791](https://arxiv.org/pdf/2602.23791)**

> **作者:** Hyejin Park; Jiwon Yoon; Sumin Park; Suree Kim; Sinae Jang; Eunsoo Lee; Dongmin Kang; Dongbo Min
>
> **备注:** Accepted at CVPR 2026, Project Page: this https URL
>
> **摘要:** Accurate focus quality assessment (FQA) in fluorescence microscopy is challenging due to stain-dependent optical variations that induce heterogeneous focus behavior across images. Existing methods, however, treat focus quality as a stain-agnostic problem, assuming a shared global ordering. We formulate stain-aware FQA for fluorescence microscopy, showing that focus-rank relationships vary substantially across stains due to stain-dependent imaging characteristics and invalidate this assumption. To support this formulation, we introduce FluoMix, the first dataset for stain-aware FQA spanning multiple tissues, fluorescent stains, and focus levels. We further propose FluoCLIP, a two-stage vision-language framework that grounds stain semantics and enables stain-conditioned ordinal reasoning for focus prediction, effectively decoupling stain representation from ordinal structure. By explicitly modeling stain-dependent focus behavior, FluoCLIP consistently outperforms both conventional FQA methods and recent vision-language baselines, demonstrating strong generalization across diverse fluorescence microscopy conditions. Code and dataset are publicly available at this https URL.
>
---
#### [replaced 056] Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文介绍Intern-S1-Pro，一个万亿参数的科学多模态基础模型，解决通用与科学领域任务。通过大规模训练和高效框架，提升推理与专业任务性能。**

- **链接: [https://arxiv.org/pdf/2603.25040](https://arxiv.org/pdf/2603.25040)**

> **作者:** Yicheng Zou; Dongsheng Zhu; Lin Zhu; Tong Zhu; Yunhua Zhou; Peiheng Zhou; Xinyu Zhou; Dongzhan Zhou; Zhiwang Zhou; Yuhao Zhou; Bowen Zhou; Zhanping Zhong; Zhijie Zhong; Haiteng Zhao; Penghao Zhao; Xiaomeng Zhao; Zhiyuan Zhao; Yechen Zhang; Jin Zhang; Wenwei Zhang; Hongjie Zhang; Zhuo Zhang; Wenlong Zhang; Bo Zhang; Chao Zhang; Chen Zhang; Yuhang Zang; Fei Yuan; Jiakang Yuan; Jiashuo Yu; Jinhui Yin; Haochen Ye; Qian Yao; Bowen Yang; Danni Yang; Kaichen Yang; Ziang Yan; Jun Xu; Yicheng Xu; Wanghan Xu; Xuenan Xu; Chao Xu; Ruiliang Xu; Shuhao Xing; Long Xing; Xinchen Xie; Ling-I Wu; Zijian Wu; Zhenyu Wu; Lijun Wu; Yue Wu; Jianyu Wu; Wen Wu; Fan Wu; Xilin Wei; Qi Wei; Bingli Wang; Rui Wang; Ziyi Wang; Zun Wang; Yi Wang; Haomin Wang; Yizhou Wang; Lintao Wang; Yiheng Wang; Longjiang Wang; Bin Wang; Jian Tong; Zhongbo Tian; Huanze Tang; Chen Tang; Shixiang Tang; Yu Sun; Qiushi Sun; Xuerui Su; Qisheng Su; Chenlin Su; Demin Song; Jin Shi; Fukai Shang; Yuchen Ren; Pengli Ren; Xiaoye Qu; Yuan Qu; Jiantao Qiu; Yu Qiao; Biqing Qi; Runyu Peng; Tianshuo Peng; Jiahui Peng; Qizhi Pei; Zhuoshi Pan; Linke Ouyang; Wenchang Ning; Yichuan Ma; Zerun Ma; Ningsheng Ma; Runyuan Ma; Chengqi Lyu; Haijun Lv
>
> **摘要:** We introduce Intern-S1-Pro, the first one-trillion-parameter scientific multimodal foundation model. Scaling to this unprecedented size, the model delivers a comprehensive enhancement across both general and scientific domains. Beyond stronger reasoning and image-text understanding capabilities, its intelligence is augmented with advanced agent capabilities. Simultaneously, its scientific expertise has been vastly expanded to master over 100 specialized tasks across critical science fields, including chemistry, materials, life sciences, and earth sciences. Achieving this massive scale is made possible by the robust infrastructure support of XTuner and LMDeploy, which facilitates highly efficient Reinforcement Learning (RL) training at the 1-trillion parameter level while ensuring strict precision consistency between training and inference. By seamlessly integrating these advancements, Intern-S1-Pro further fortifies the fusion of general and specialized intelligence, working as a Specializable Generalist, demonstrating its position in the top tier of open-source models for general capabilities, while outperforming proprietary models in the depth of specialized scientific tasks.
>
---
#### [replaced 057] One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image
- **分类: cs.CL; cs.CR; cs.CV; cs.IR**

- **简介: 该论文研究视觉文档检索增强生成（VD-RAG）系统的安全问题，探讨如何通过单张图像进行 poisoning 攻击。任务属于网络安全领域，旨在揭示VD-RAG的脆弱性并提出攻击方法。**

- **链接: [https://arxiv.org/pdf/2504.02132](https://arxiv.org/pdf/2504.02132)**

> **作者:** Ezzeldin Shereen; Dan Ristea; Shae McFadden; Burak Hasircioglu; Vasilios Mavroudis; Chris Hicks
>
> **备注:** Published in Transactions on Machine Learning Research (03/2026)
>
> **摘要:** Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.
>
---
#### [replaced 058] GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.20331](https://arxiv.org/pdf/2601.20331)**

> **作者:** Mai Su; Qihan Yu; Zhongtao Wang; Yilong Li; Chengwei Pan; Yisong Chen; Guoping Wang; Fei Zhu
>
> **摘要:** 3D Gaussian Splatting (3DGS) enables efficient rendering, yet accurate surface reconstruction remains challenging due to unreliable geometric supervision. Existing approaches predominantly rely on depth-based reprojection to infer visibility and enforce multi-view consistency, leading to a fundamental circular dependency: visibility estimation requires accurate depth, while depth supervision itself is conditioned on visibility. In this work, we revisit multi-view geometric supervision from the perspective of visibility modeling. Instead of inferring visibility from pixel-wise depth consistency, we explicitly model visibility at the level of Gaussian primitives. We introduce a Gaussian visibility-aware multi-view geometric consistency (GVMV) formulation, which aggregates cross-view visibility of shared Gaussians to construct reliable supervision over co-visible regions. To further incorporate monocular priors, we propose a progressive quadtree-calibrated depth alignment (QDC) strategy that performs block-wise affine calibration under visibility-aware guidance, effectively mitigating scale ambiguity while preserving local geometric structures. Extensive experiments on DTU and Tanks and Temples demonstrate that our method consistently improves reconstruction accuracy over prior Gaussian-based approaches. Our code is fully open-sourced and available at an anonymous repository: this https URL.
>
---
#### [replaced 059] Human-Centric Perception for Child Sexual Abuse Imagery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27290](https://arxiv.org/pdf/2603.27290)**

> **作者:** Camila Laranjeira; João Macedo; Sandra Avila; Fabrício Benevenuto; Jefersson A. dos Santos
>
> **备注:** submitted to IEEE Transactions on Information Forensics and Security (TIFS)
>
> **摘要:** Law enforcement agencies and non-gonvernmental organizations handling reports of Child Sexual Abuse Imagery (CSAI) are overwhelmed by large volumes of data, requiring the aid of automation tools. However, defining sexual abuse in images of children is inherently challenging, encompassing sexually explicit activities and hints of sexuality conveyed by the individual's pose, or their attire. CSAI classification methods often rely on black-box approaches, targeting broad and abstract concepts such as pornography. Thus, our work is an in-depth exploration of tasks from the literature on Human-Centric Perception, across the domains of safe images, adult pornography, and CSAI, focusing on targets that enable more objective and explainable pipelines for CSAI classification in the future. We introduce the Body-Keypoint-Part Dataset (BKPD), gathering images of people from varying age groups and sexual explicitness to approximate the domain of CSAI, along with manually curated hierarchically structured labels for skeletal keypoints and bounding boxes for person and body parts, including head, chest, hip, and hands. We propose two methods, namely BKP-Association and YOLO-BKP, for simultaneous pose estimation and detection, with targets associated per individual for a comprehensive decomposed representation of each person. Our methods are benchmarked on COCO-Keypoints and COCO-HumanParts, as well as our human-centric dataset, achieving competitive results with models that jointly perform all tasks. Cross-domain ablation studies on BKPD and a case study on RCPD highlight the challenges posed by sexually explicit domains. Our study addresses previously unexplored targets in the CSAI domain, paving the way for novel research opportunities.
>
---
#### [replaced 060] Generation Is Compression: Zero-Shot Video Coding via Stochastic Rectified Flow
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.26571](https://arxiv.org/pdf/2603.26571)**

> **作者:** Ziyue Zeng; Xun Su; Haoyuan Liu; Bingyu Lu; Yui Tatsumi; Hiroshi Watanabe
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Recent advances in generative modeling have enabled perceptual video compression at ultra-low bitrates, yet existing methods predominantly treat the generative model as a refinement or reconstruction module attached to a separately designed codec backbone. We propose \emph{Generative Video Codebook Codec} (GVCC), a zero-shot framework that turns a pretrained video generative model into the codec itself: the transmitted bitstream directly specifies the generative decoding trajectory, with no retraining required. To enable this, we convert the deterministic rectified-flow ODE of modern video foundation models into an equivalent SDE at inference time, unlocking per-step stochastic injection points for codebook-driven compression. Building on this unified backbone, we instantiate three complementary conditioning strategies -- \emph{Image-to-Video} (I2V) with autoregressive GOP chaining, tail latent residual correction, and adaptive atom allocation, \emph{Text-to-Video} (T2V) operating at near-zero side information as a pure generative prior, and \emph{First-Last-Frame-to-Video} (FLF2V) with boundary-sharing GOP chaining for dual-anchor temporal control. Together, these variants span a principled trade-off space between spatial fidelity, temporal coherence, and compression efficiency. Experiments on standard benchmarks show that GVCC achieves high-quality reconstruction below 0.002\,bpp while supporting flexible bitrate control through a single hyperparameter.
>
---
#### [replaced 061] Scaling Video Pretraining for Surgical Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29966](https://arxiv.org/pdf/2603.29966)**

> **作者:** Sicheng Lu; Zikai Xiao; Jianhui Wei; Danyu Sun; Qi Lu; Keli Hu; Yang Feng; Jian Wu; Zongxin Yang; Zuozhu Liu
>
> **摘要:** Surgical video understanding is essential for computer-assisted interventions, yet existing surgical foundation models remain constrained by limited data scale, procedural diversity, and inconsistent evaluation, often lacking a reproducible training pipeline. We propose SurgRec, a scalable and reproducible pretraining recipe for surgical video understanding, instantiated with two variants: SurgRec-MAE and SurgRec-JEPA. We curate a large multi-source corpus of 10,535 videos and 214.5M frames spanning endoscopy, laparoscopy, cataract, and robotic surgery. Building on this corpus, we develop a unified pretraining pipeline with balanced sampling and standardize a reproducible benchmark across 16 downstream datasets and four clinical domains with consistent data splits. Across extensive comparisons against SSL baselines and vision-language models, SurgRec consistently achieves superior performance across downstream datasets. In contrast, VLMs prove unreliable for fine-grained temporal recognition, exhibiting both performance gaps and sensitivity to prompt phrasing. Our work provides a reproducible, scalable foundation for the community to build more general surgical video models. All code, models, and data will be publicly released.
>
---
#### [replaced 062] SkinGenBench: Generative Model and Preprocessing Effects for Synthetic Dermoscopic Augmentation in Melanoma Diagnosis
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.17585](https://arxiv.org/pdf/2512.17585)**

> **作者:** N. A. Adarsh Pritam; Jeba Shiney O; Sanyam Jain
>
> **摘要:** This work introduces SkinGenBench, a systematic biomedical imaging benchmark that investigates how preprocessing complexity interacts with generative model choice for synthetic dermoscopic image augmentation and downstream melanoma diagnosis. Using a curated dataset of $14,116$ dermoscopic images from HAM10000 and MILK10K across five lesion classes, we evaluate the two representative generative paradigms: StyleGAN2-ADA and Denoising Diffusion Probabilistic Models (DDPMs) under basic geometric augmentation and advanced artifact removal pipelines. Synthetic melanoma images are assessed using established perceptual and distributional metrics (FID, KID, IS), feature space analysis, and their impact on diagnostic performance across five downstream classifiers. Experimental results demonstrate that generative architecture choice has a stronger influence on both image fidelity and diagnostic utility than preprocessing complexity. StyleGAN2-ADA consistently produced synthetic images more closely aligned with real data distributions, achieving the lowest FID ($\approx 65.5$) and KID ($\approx 0.05$), while diffusion models generated higher variance samples at the cost of reduced perceptual fidelity and class anchoring. Advanced artifact removal yielded only marginal improvements in generative metrics and provided limited downstream diagnostic gains, suggesting possible suppression of clinically relevant texture cues. In contrast, synthetic data augmentation substantially improved melanoma detection with $8$-$15$\% absolute gains in melanoma F1-score, and ViT-B/16 achieving F1 $\approx 0.88$ and ROC-AUC $\approx 0.98$, representing an improvement of approximately $14\%$ over non-augmented baselines. Our code can be found at this https URL
>
---
#### [replaced 063] Grounding Everything in Tokens for Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10554](https://arxiv.org/pdf/2512.10554)**

> **作者:** Xiangxuan Ren; Zhongdao Wang; Liping Hou; Pin Tang; Guoqing Wang; Chao Ma
>
> **备注:** 19 pages, 16 figures, 12 Tables
>
> **摘要:** Multimodal large language models (MLLMs) have made significant advancements in vision understanding and reasoning. However, the autoregressive Transformer architecture used by MLLMs requries tokenization on input images, which limits their ability to accurately ground objects within the 2D image space. This raises an important question: how can sequential language tokens be improved to better ground objects in 2D spatial space for MLLMs? To address this, we present a spatial representation method for grounding objects, namely GETok, that integrates a specialized vocabulary of learnable tokens into MLLMs. GETok first uses grid tokens to partition the image plane into structured spatial anchors, and then exploits offset tokens to enable precise and iterative refinement of localization predictions. By embedding spatial relationships directly into tokens, GETok significantly advances MLLMs in native 2D space reasoning without modifying the autoregressive architecture. Extensive experiments demonstrate that GETok achieves superior performance over the state-of-the-art methods across various referring tasks in both supervised fine-tuning and reinforcement learning settings.
>
---
#### [replaced 064] A Novel FACS-Aligned Anatomical Text Description Paradigm for Fine-Grained Facial Behavior Synthesis
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2603.18588](https://arxiv.org/pdf/2603.18588)**

> **作者:** Jiahe Wang; Cong Liang; Xuandong Huang; Yuxin Wang; Xin Yun; Yi Wu; Yanan Chang; Shangfei Wang
>
> **摘要:** Facial behavior constitutes the primary medium of human nonverbal communication. Existing synthesis methods predominantly follow two paradigms: coarse emotion category labels or one-hot Action Unit (AU) vectors from the Facial Action Coding System (FACS). Neither paradigm reliably renders fine-grained facial behaviors nor resolves anatomically implausible artifacts caused by conflicting AUs. Therefore, we propose a novel task paradigm: anatomically grounded facial behavior synthesis from FACS-based AU descriptions. This paradigm explicitly encodes FACS-defined muscle movement rules, inter-AU interactions, and conflict resolution mechanisms into natural language control signals. To enable systematic research, we develop a dynamic AU text processor, a FACS rule-based module that converts raw AU annotations into anatomically consistent natural language descriptions. Using this processor, we construct BP4D-AUText, the first large-scale text-image paired dataset for fine-grained facial behavior synthesis, comprising over 302K high-quality samples. Given that existing general semantic consistency metrics cannot capture the alignment between anatomical facial descriptions and synthesized muscle movements, we propose the Alignment Accuracy of AU Probability Distributions (AAAD), a task-specific metric that quantifies semantic consistency. Finally, we design VQ-AUFace, a robust baseline framework incorporating anatomical priors and progressive cross-modal alignment, to validate the paradigm. Extensive quantitative experiments and user studies demonstrate the paradigm significantly outperforms state-of-the-art methods, particularly in challenging conflicting AU scenarios, achieving superior anatomical fidelity, semantic consistency, and visual quality.
>
---
#### [replaced 065] PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.22193](https://arxiv.org/pdf/2603.22193)**

> **作者:** Mingju Gao; Kaisen Yang; Huan-ang Gao; Bohan Li; Ao Ding; Wenyi Li; Yangcheng Yu; Jinkun Liu; Shaocong Xu; Yike Niu; Haohan Chi; Hao Chen; Hao Tang; Yu Zhang; Li Yi; Hao Zhao
>
> **备注:** Accepted to CVPR 2026 Code: this https URL
>
> **摘要:** Hand-object interaction (HOI) reconstruction and synthesis are becoming central to embodied AI and AR/VR. Yet, despite rapid progress, existing HOI generation research remains fragmented across three disjoint tracks: (1) pose-only synthesis that predicts MANO trajectories without producing pixels; (2) single-image HOI generation that hallucinates appearance from masks or 2D cues but lacks dynamics; and (3) video generation methods that require both the entire pose sequence and the ground-truth first frame as inputs, preventing true sim-to-real deployment. Inspired by the philosophy of Joo et al. (2018), we think that HOI generation requires a unified engine that brings together pose, appearance, and motion within one coherent framework. Thus we introduce PAM: a Pose-Appearance-Motion Engine for controllable HOI video generation. The performance of our engine is validated by: (1) On DexYCB, we obtain an FVD of 29.13 (vs. 38.83 for InterDyn), and MPJPE of 19.37 mm (vs. 30.05 mm for CosHand), while generating higher-resolution 480x720 videos compared to 256x256 and 256x384 baselines. (2) On OAKINK2, our full multi-condition model improves FVD from 68.76 to 46.31. (3) An ablation over input conditions on DexYCB shows that combining depth, segmentation, and keypoints consistently yields the best results. (4) For a downstream hand pose estimation task using SimpleHand, augmenting training with 3,400 synthetic videos (207k frames) allows a model trained on only 50% of the real data plus our synthetic data to match the 100% real baseline.
>
---
#### [replaced 066] Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.10611](https://arxiv.org/pdf/2601.10611)**

> **作者:** Christopher Clark; Jieyu Zhang; Zixian Ma; Jae Sung Park; Mohammadreza Salehi; Rohun Tripathi; Sangho Lee; Zhongzheng Ren; Chris Dongjoo Kim; Yinuo Yang; Vincent Shao; Yue Yang; Weikai Huang; Ziqi Gao; Taira Anderson; Jianrui Zhang; Jitesh Jain; George Stoica; Winson Han; Ali Farhadi; Ranjay Krishna
>
> **备注:** Updated first authors
>
> **摘要:** Today's strongest video-language models (VLMs) remain proprietary. The strongest open-weight models either rely on synthetic data from proprietary VLMs, effectively distilling from them, or do not disclose their training data or recipe. As a result, the open-source community lacks the foundations needed to improve on the state-of-the-art video (and image) language models. Crucially, many downstream applications require more than just high-level video understanding; they require grounding -- either by pointing or by tracking in pixels. Even proprietary models lack this capability. We present Molmo2, a new family of VLMs that are state-of-the-art among open-source models and demonstrate exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks. Our key contribution is a collection of 7 new video datasets and 2 multi-image datasets, including a dataset of highly detailed video captions for pre-training, a free-form video Q&A dataset for fine-tuning, a new object tracking dataset with complex queries, and an innovative new video pointing dataset, all collected without the use of closed VLMs. We also present a training recipe for this data utilizing an efficient packing and message-tree encoding scheme, and show bi-directional attention on vision tokens and a novel token-weight strategy improves performance. Our best-in-class 8B model outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos. On video-grounding Molmo2 significantly outperforms existing open-weight models like Qwen3-VL (35.5 vs 29.6 accuracy on video counting) and surpasses proprietary models like Gemini 3 Pro on some tasks (38.4 vs 20.0 F1 on video pointing and 56.2 vs 41.1 J&F on video tracking).
>
---
#### [replaced 067] Toward Personalized Darts Training: A Data-Driven Framework Based on Skeleton-Based Biomechanical Analysis and Motion Modeling
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2604.01130](https://arxiv.org/pdf/2604.01130)**

> **作者:** Zhantao Chen; Dongyi He; Jin Fang; Xi Chen; Yishuo Liu; Xiaozhen Zhong; Xuejun Hu
>
> **摘要:** As sports training becomes more data-driven, traditional dart coaching based mainly on experience and visual observation is increasingly inadequate for high-precision, goal-oriented movements. Although prior studies have highlighted the importance of release parameters, joint motion, and coordination in dart throwing, most quantitative methods still focus on local variables, single-release metrics, or static template matching. These approaches offer limited support for personalized training and often overlook useful movement variability. This paper presents a data-driven dart training assistance system. The system creates a closed-loop framework spanning motion capture, feature modeling, and personalized feedback. Dart-throwing data were collected in markerless conditions using a Kinect 2.0 depth sensor and an optical camera. Eighteen kinematic features were extracted from four biomechanical dimensions: three-link coordination, release velocity, multi-joint angular configuration, and postural stability. Two modules were developed: a personalized optimal throwing trajectory model that combines historical high-quality samples with the minimum jerk criterion, and a motion deviation diagnosis and recommendation model based on z-scores and hierarchical logic. A total of 2,396 throwing samples from professional and non-professional athletes were collected. Results show that the system generates smooth personalized reference trajectories consistent with natural human movement. Case studies indicate that it can detect poor trunk stability, abnormal elbow displacement, and imbalanced velocity control, then provide targeted recommendations. The framework shifts dart evaluation from deviation from a uniform standard to deviation from an individual's optimal control range, improving personalization and interpretability for darts training and other high-precision target sports.
>
---
#### [replaced 068] InTraGen: Trajectory-controlled Video Generation for Object Interactions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.16804](https://arxiv.org/pdf/2411.16804)**

> **作者:** Zuhao Liu; Aleksandar Yanev; Ahmad Mahmood; Ivan Nikolov; Saman Motamed; Wei-Shi Zheng; Xi Wang; Lei Sun; Luc Van Gool; Danda Pani Paudel
>
> **摘要:** Advances in video generation have significantly improved the realism and quality of created scenes. This has fueled interest in developing intuitive tools that let users leverage video generation as world simulators. Text-to-video (T2V) generation is one such approach, enabling video creation from text descriptions only. Yet, due to the inherent ambiguity in texts and the limited temporal information offered by text prompts, researchers have explored additional control signals like trajectory-guided systems, for more accurate T2V generation. Nonetheless, methods to evaluate whether T2V models can generate realistic interactions between multiple objects are lacking. We introduce InTraGen, a pipeline for improved trajectory-based generation of object interaction scenarios. We propose 4 new datasets and a novel trajectory quality metric to evaluate the performance of the proposed InTraGen. To achieve object interaction, we introduce a multi-modal interaction encoding pipeline with an object ID injection mechanism that enriches object-environment interactions. Our results demonstrate improvements in both visual fidelity and quantitative performance. Code and datasets are available at this https URL
>
---
#### [replaced 069] InvZW: Invariant Feature Learning via Noise-Adversarial Training for Robust Image Zero-Watermarking
- **分类: cs.CV; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2506.20370](https://arxiv.org/pdf/2506.20370)**

> **作者:** Abdullah All Tanvir; Frank Y. Shih; Xin Zhong
>
> **备注:** This paper has been accepted for publication by the Frontiers in Signal Processing
>
> **摘要:** This paper introduces a novel deep learning framework for robust image zero-watermarking based on distortion-invariant feature learning. As a zero-watermarking scheme, our method leaves the original image unaltered and learns a reference signature through optimization in the feature space. The proposed framework consists of two key modules. In the first module, a feature extractor is trained via noise-adversarial learning to generate representations that are both invariant to distortions and semantically expressive. This is achieved by combining adversarial supervision against a distortion discriminator and a reconstruction constraint to retain image content. In the second module, we design a learning-based multibit zero-watermarking scheme where the trained invariant features are projected onto a set of trainable reference codes optimized to match a target binary message. Extensive experiments on diverse image datasets and a wide range of distortions show that our method achieves state-of-the-art robustness in both feature stability and watermark recovery. Comparative evaluations against existing self-supervised and deep watermarking techniques further highlight the superiority of our framework in generalization and robustness.
>
---
#### [replaced 070] Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.15475](https://arxiv.org/pdf/2601.15475)**

> **作者:** Yunshan Qi; Lin Zhu; Nan Bao; Yifan Zhao; Jia Li
>
> **备注:** Accepted by the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2026. Project Page: this https URL. Our code and datasets are publicly available at this https URL
>
> **摘要:** Novel view synthesis from low dynamic range (LDR) blurry images, which are common in the wild, struggles to recover high dynamic range (HDR) and sharp 3D representations in extreme lighting conditions. Although existing methods employ event data to address this issue, they ignore the sensor-physics mismatches between the camera output and physical world radiance, resulting in suboptimal HDR and deblurring results. To cope with this problem, we propose a unified sensor-physics grounded NeRF framework for sharp HDR novel view synthesis from single-exposure blurry LDR images and corresponding events. We employ NeRF to directly represent the actual radiance of the 3D scene in the HDR domain and model raw HDR scene rays hitting the sensor pixels as in the physical world. A 2D pixel-wise RGB CRF model is introduced to align the NeRF rendered pixel values with the sensor-recorded LDR pixel values of the input images. A novel event CRF model is also designed to bridge the gap between physical scene dynamics and event sensor output. The two models are jointly optimized with the NeRF network, leveraging the spatial and temporal dynamic information in events to enhance the sharp HDR 3D representation learning. Experiments on the collected and public datasets demonstrate that our method achieves state-of-the-art HDR and deblurring novel view synthesis results with single-exposure blurry LDR images and corresponding events.
>
---
#### [replaced 071] Demystifying Transition Matching: When and Why It Can Beat Flow Matching
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.17991](https://arxiv.org/pdf/2510.17991)**

> **作者:** Jaihoon Kim; Rajarshi Saha; Minhyuk Sung; Youngsuk Park
>
> **备注:** AISTATS 2026
>
> **摘要:** Flow Matching (FM) underpins many state-of-the-art generative models, yet recent results indicate that Transition Matching (TM) can achieve higher quality with fewer sampling steps. This work answers the question of when and why TM outperforms FM. First, when the target is a unimodal Gaussian distribution, we prove that TM attains strictly lower KL divergence than FM for finite number of steps. The improvement arises from stochastic difference latent updates in TM, which preserve target covariance that deterministic FM underestimates. We then characterize convergence rates, showing that TM achieves faster convergence than FM under a fixed compute budget, establishing its advantage in the unimodal Gaussian setting. Second, we extend the analysis to Gaussian mixtures and identify local-unimodality regimes in which the sampling dynamics approximate the unimodal case, where TM can outperform FM. The approximation error decreases as the minimal distance between component means increases, highlighting that TM is favored when the modes are well separated. However, when the target variance approaches zero, each TM update converges to the FM update, and the performance advantage of TM diminishes. In summary, we show that TM outperforms FM when the target distribution has well-separated modes and non-negligible variances. We validate our theoretical results with controlled experiments on Gaussian distributions, and extend the comparison to real-world applications in image and video generation.
>
---
#### [replaced 072] ICM-SR: Image-Conditioned Manifold Regularization for Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.22048](https://arxiv.org/pdf/2511.22048)**

> **作者:** Junoh Kang; Donghun Ryou; Bohyung Han
>
> **摘要:** Real world image super-resolution (Real-ISR) often leverages the powerful generative priors of text-to-image diffusion models by regularizing the output to lie on their learned manifold. However, existing methods often overlook the importance of the regularizing manifold, typically defaulting to a text-conditioned manifold. This approach suffers from two key limitations. Conceptually, it is misaligned with the Real-ISR task, which is to generate high quality (HQ) images directly tied to the low quality (LQ) images. Practically, the teacher model often reconstructs images with color distortions and blurred edges, indicating a flawed generative prior for this task. To correct these flaws and ensure conceptual alignment, a more suitable manifold must incorporate information from the images. While the most straightforward approach is to condition directly on the raw input images, their high information densities make the regularization process numerically unstable. To resolve this, we propose image-conditioned manifold regularization (ICM), a method that regularizes the output towards a manifold conditioned on the sparse yet essential structural information: a combination of colormap and Canny edges. ICM provides a task-aligned and stable regularization signal, thereby avoiding the instability of dense-conditioning and enhancing the final super-resolution quality. Our experiments confirm that the proposed regularization significantly enhances super-resolution performance, particularly in perceptual quality, demonstrating its effectiveness for real-world applications. We will release the source code of our work for reproducibility.
>
---
#### [replaced 073] KARL: Knowledge-Aware Reasoning and Reinforcement Learning for Knowledge-Intensive Visual Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于知识密集型视觉定位任务，解决MLLM在特定概念定位中无法有效利用内部知识的问题。通过构建知识引导数据和KARL框架提升模型性能。**

- **链接: [https://arxiv.org/pdf/2503.12797](https://arxiv.org/pdf/2503.12797)**

> **作者:** Xinyu Ma; Ziyang Ding; Zhicong Luo; Chi Chen; Zonghao Guo; Derek F. Wong; Zhen Zhao; Xiaoyi Feng; Maosong Sun
>
> **摘要:** Knowledge-Intensive Visual Grounding (KVG) requires models to localize objects using fine-grained, domain-specific entity names rather than generic referring expressions. Although Multimodal Large Language Models (MLLMs) possess rich entity knowledge and strong generic grounding capabilities, they often fail to effectively utilize such knowledge when grounding specialized concepts, revealing a knowledge-grounding gap between internal knowledge and grounding predictions. To address this challenge, we propose a knowledge-aware training paradigm for KVG. Our approach first constructs knowledge-guided reasoning data to encourage models to activate domain-relevant entity knowledge during grounding, and then introduces KARL, a Knowledge-Aware Reinforcement Learning framework that adaptively modulates reward signals according to the model's estimated knowledge mastery of different entities. To facilitate systematic evaluation, we introduce KVG-Bench, a benchmark spanning 10 domains with 1.3K curated test cases covering 531 images and 882 entities. Extensive experiments show that our approach consistently outperforms a wide range of baseline models and achieves substantially stronger cross-domain generalization on unseen categories. The data, codes, and models are released at this https URL.
>
---
#### [replaced 074] CARE: Confidence-aware Ratio Estimation for Medical Biomarkers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19585](https://arxiv.org/pdf/2505.19585)**

> **作者:** Jiameng Li; Teodora Popordanoska; Aleksei Tiulpin; Sebastian G. Gruber; Frederik Maes; Matthew B. Blaschko
>
> **备注:** 12 pages
>
> **摘要:** Ratio-based biomarkers (RBBs), such as the proportion of necrotic tissue within a tumor, are widely used in clinical practice to support diagnosis, prognosis, and treatment planning. These biomarkers are typically estimated from segmentation outputs by computing region-wise ratios. Despite the high-stakes nature of clinical decision making, existing methods provide only point estimates, offering no measure of uncertainty. In this work, we propose a unified confidence-aware framework for estimating ratio-based biomarkers. Our uncertainty analysis stems from two observations: (1) the probability ratio estimator inherently admits a statistical confidence interval regarding local randomness (bias and variance); (2) the segmentation network is not perfectly calibrated (calibration error).We perform a systematic analysis of error propagation in the segmentation-to-biomarker pipeline and identify model miscalibration as the dominant source of uncertainty. Extensive experiments show that our method produces statistically sound confidence intervals, with tunable confidence levels, enabling more trustworthy application of segmentation-derived RBBs in clinical workflows.
>
---
#### [replaced 075] Seeing Through the Chain: Mitigate Hallucination in Multimodal Reasoning Models via CoT Compression and Contrastive Preference Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03380](https://arxiv.org/pdf/2602.03380)**

> **作者:** Hao Fang; Jinyu Li; Jiawei Kong; Tianqu Zhuang; Kuofeng Gao; Bin Chen; Shu-Tao Xia
>
> **摘要:** While multimodal reasoning models (MLRMs) have exhibited impressive capabilities, they remain prone to hallucinations, and effective solutions are still underexplored. In this paper, we experimentally analyze the hallucination cause and propose C3PO, a training-based mitigation framework comprising \textbf{C}hain-of-Thought \textbf{C}ompression and \textbf{C}ontrastive \textbf{P}reference \textbf{O}ptimization. Firstly, we identify that introducing reasoning mechanisms exacerbates models' reliance on language priors while overlooking visual inputs, which can produce CoTs with reduced visual cues but redundant text tokens. To this end, we propose to selectively filter redundant thinking tokens for a more compact and signal-efficient CoT representation that preserves task-relevant information while suppressing noise. In addition, we observe that the quality of the reasoning trace largely determines whether hallucination emerges in subsequent responses. To leverage this insight, we introduce a reasoning-enhanced preference tuning scheme that constructs training pairs using high-quality AI feedback. We further design a multimodal hallucination-inducing mechanism that elicits models' inherent hallucination patterns via carefully crafted inducers, yielding informative negative signals for contrastive correction. We provide theoretical justification for the effectiveness and demonstrate consistent hallucination reduction across diverse MLRMs and benchmarks.
>
---
#### [replaced 076] Seamless High-Resolution Terrain Reconstruction: A Prior-Based Vision Transformer Approach
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2507.09681](https://arxiv.org/pdf/2507.09681)**

> **作者:** Osher Rafaeli; Tal Svoray; Ariel Nahlieli
>
> **摘要:** High-resolution elevation data is essential for hydrological modeling, hazard assessment, and environmental monitoring; however, globally consistent, fine-scale Digital Elevation Models (DEMs) remain unavailable. Very high-resolution single-view imagery enables the extraction of topographic information at the pixel level, allowing the reconstruction of fine terrain details over large spatial extents. In this paper, we present single-view-based DEM reconstruction shown to support practical analysis in GIS environments across multiple sub-national jurisdictions. Specifically, we produce high-resolution DEMs for large-scale basins, representing a substantial improvement over the 30 m resolution of globally available Shuttle Radar Topography Mission (SRTM) data. The DEMs are generated using a prior-based monocular depth foundation (MDE) model, extended in this work to the remote sensing height domain for high-resolution, globally consistent elevation reconstruction. We fine-tune the model by integrating low-resolution SRTM data as a global prior with high-resolution RGB imagery from the National Agriculture Imagery Program (NAIP), producing DEMs with near LiDAR-level accuracy. Our method achieves a 100x resolution enhancement (from 30 m to 30 cm), exceeding existing super-resolution approaches by an order of magnitude. Across two diverse landscapes, the model generalizes robustly, resolving fine-scale terrain features with a mean absolute error of less than 5 m relative to LiDAR and improving upon SRTM by up to 18 %. Hydrological analyses at both catchment and hillslope scales confirm the method's utility for hazard assessment and environmental monitoring, demonstrating improved streamflow representation and catchment delineation. Finally, we demonstrate the scalability of the framework by applying it across large geographic regions.
>
---
#### [replaced 077] Towards Faithful Reasoning in Comics for Small MLLMs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.02991](https://arxiv.org/pdf/2601.02991)**

> **作者:** Chengcheng Feng; Haojie Yin; Yucheng Jin; Kaizhu Huang
>
> **摘要:** Comic understanding presents a significant challenge for Multimodal Large Language Models (MLLMs), as the intended meaning of a comic often emerges from the joint interpretation of visual, textual, and social cues. This naturally motivates Chain-of-Thought (CoT) prompting, since explicit intermediate reasoning appears promising for integrating such heterogeneous signals. However, existing CoT methods are poorly matched to this structure: they tend to force interpretation into a single reasoning path before multiple cues have been jointly considered, often degrading performance, especially for small MLLMs. Our key idea is to explicitly preserve multi-cue interpretation during supervision construction, rather than collapsing comic understanding into a single reasoning chain. To this end, we propose a two-stage framework for faithful comic reasoning in small MLLMs. First, we introduce MoCoT, a modular supervision construction framework that preserves multi-cue interpretation and turns it into more faithful supervision. Second, we propose VERA, a structured reward mechanism that turns such supervision into faithful reasoning behavior by aligning optimization with both reasoning faithfulness and answer correctness. Extensive experiments on five benchmarks spanning comic understanding and broader humor-centric and abstract visual reasoning tasks demonstrate that our framework achieves strong results in the $\leq$ 4B regime, surpasses several 7B baselines, improves four small MLLMs by an average of $\mathbf{12.1%}$ as a plug-in, and consistently enhances reasoning faithfulness while preserving inference efficiency.
>
---
#### [replaced 078] VOIC: Visible-Occluded Integrated Guidance for 3D Semantic Scene Completion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.18954](https://arxiv.org/pdf/2512.18954)**

> **作者:** Zaidao Han; Risa Higashita; Jiang Liu
>
> **备注:** Novelty less
>
> **摘要:** Camera-based 3D Semantic Scene Completion (SSC) is a critical task for autonomous driving and robotic scene understanding. It aims to infer a complete 3D volumetric representation of both semantics and geometry from a single image. Existing methods typically focus on end-to-end 2D-to-3D feature lifting and voxel completion. However, they often overlook the interference between high-confidence visible-region perception and low-confidence occluded-region reasoning caused by single-image input, which can lead to feature dilution and error propagation. To address these challenges, we introduce an offline Visible Region Label Extraction (VRLE) strategy that explicitly separates and extracts voxel-level supervision for visible regions from dense 3D ground truth. This strategy purifies the supervisory space for two complementary sub-tasks: visible-region perception and occluded-region reasoning. Building on this idea, we propose the Visible-Occluded Interactive Completion Network (VOIC), a novel dual-decoder framework that explicitly decouples SSC into visible-region semantic perception and occluded-region scene completion. VOIC first constructs a base 3D voxel representation by fusing image features with depth-derived occupancy. The visible decoder focuses on generating high-fidelity geometric and semantic priors, while the occlusion decoder leverages these priors together with cross-modal interaction to perform coherent global scene reasoning. Extensive experiments on the SemanticKITTI and SSCBench-KITTI360 benchmarks demonstrate that VOIC outperforms existing monocular SSC methods in both geometric completion and semantic segmentation accuracy, achieving state-of-the-art performance.
>
---
#### [replaced 079] A multi-weight self-matching visual explanation for cnns on sar images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02344](https://arxiv.org/pdf/2512.02344)**

> **作者:** Siyuan Sun; Yongping Zhang; Hongcheng Zeng; Yamin Wang; Wei Yang; Wanting Yang; Jie Chen
>
> **摘要:** In recent years, convolutional neural networks (CNNs) have achieved significant success in various synthetic aperture radar (SAR) tasks. However, the complexity and opacity of their internal mechanisms hinder the fulfillment of high-reliability requirements, thereby limiting their application in SAR. Improving the interpretability of CNNs is thus of great importance for their development and deployment in SAR. In this paper, a visual explanation method termed multi-weight self-matching class activation mapping (MS-CAM) is proposed. MS-CAM matches SAR images with the feature maps and corresponding gradients extracted by the CNN, and combines both channel-wise and element-wise weights to visualize the decision basis learned by the model in SAR images. Extensive experiments conducted on a self-constructed SAR target classification dataset demonstrate that MS-CAM more accurately highlights the network's regions of interest and captures detailed target feature information, thereby enhancing network interpretability. Furthermore, the feasibility of applying MS-CAM to weakly-supervised obiect localization is validated. Key factors affecting localization accuracy, such as pixel thresholds, are analyzed in depth to inform future work.
>
---
#### [replaced 080] AutoWeather4D: Autonomous Driving Video Weather Conversion via G-Buffer Dual-Pass Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.26546](https://arxiv.org/pdf/2603.26546)**

> **作者:** Tianyu Liu; Weitao Xiong; Kunming Luo; Manyuan Zhang; Peng Li; Yuan Liu; Ping Tan
>
> **备注:** Project Page: this https URL | Github: this https URL
>
> **摘要:** Generative video models have significantly advanced the photorealistic synthesis of adverse weather for autonomous driving; however, they consistently demand massive datasets to learn rare weather scenarios. While 3D-aware editing methods alleviate these data constraints by augmenting existing video footage, they are fundamentally bottlenecked by costly per-scene optimization and suffer from inherent geometric and illumination entanglement. In this work, we introduce AutoWeather4D, a feed-forward 3D-aware weather editing framework designed to explicitly decouple geometry and illumination. At the core of our approach is a G-buffer Dual-pass Editing mechanism. The Geometry Pass leverages explicit structural foundations to enable surface-anchored physical interactions, while the Light Pass analytically resolves light transport, accumulating the contributions of local illuminants into the global illumination to enable dynamic 3D local relighting. Extensive experiments demonstrate that AutoWeather4D achieves comparable photorealism and structural consistency to generative baselines while enabling fine-grained parametric physical control, serving as a practical data engine for autonomous driving.
>
---
