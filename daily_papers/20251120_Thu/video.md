# 计算机视觉 cs.CV

- **最新发布 121 篇**

- **更新 95 篇**

## 最新发布

#### [new 001] Jointly Conditioned Diffusion Model for Multi-View Pose-Guided Person Image Synthesis
- **分类: cs.CV**

- **简介: 该论文提出JCDM模型，用于多视角姿态引导的人像合成任务。针对单视角纹理不完整和跨视角交互缺失问题，设计外观先验模块和联合条件注入机制，融合多视角信息并保持身份一致性，实现高质量、跨视角一致的图像生成。**

- **链接: [https://arxiv.org/pdf/2511.15092v1](https://arxiv.org/pdf/2511.15092v1)**

> **作者:** Chengyu Xie; Zhi Gong; Junchi Ren; Linkun Yu; Si Shen; Fei Shen; Xiaoyu Du
>
> **摘要:** Pose-guided human image generation is limited by incomplete textures from single reference views and the absence of explicit cross-view interaction. We present jointly conditioned diffusion model (JCDM), a jointly conditioned diffusion framework that exploits multi-view priors. The appearance prior module (APM) infers a holistic identity preserving prior from incomplete references, and the joint conditional injection (JCI) mechanism fuses multi-view cues and injects shared conditioning into the denoising backbone to align identity, color, and texture across poses. JCDM supports a variable number of reference views and integrates with standard diffusion backbones with minimal and targeted architectural modifications. Experiments demonstrate state of the art fidelity and cross-view consistency.
>
---
#### [new 002] CKDA: Cross-modality Knowledge Disentanglement and Alignment for Visible-Infrared Lifelong Person Re-identification
- **分类: cs.CV**

- **简介: 该论文针对可见光-红外持续人物再识别（VI-LReID）任务，解决多模态知识干扰导致的灾难性遗忘问题。提出CKDA方法，通过解耦模态特有与共有知识并加以对齐，实现跨模态知识平衡保留，提升连续学习性能。**

- **链接: [https://arxiv.org/pdf/2511.15016v1](https://arxiv.org/pdf/2511.15016v1)**

> **作者:** Zhenyu Cui; Jiahuan Zhou; Yuxin Peng
>
> **摘要:** Lifelong person Re-IDentification (LReID) aims to match the same person employing continuously collected individual data from different scenarios. To achieve continuous all-day person matching across day and night, Visible-Infrared Lifelong person Re-IDentification (VI-LReID) focuses on sequential training on data from visible and infrared modalities and pursues average performance over all data. To this end, existing methods typically exploit cross-modal knowledge distillation to alleviate the catastrophic forgetting of old knowledge. However, these methods ignore the mutual interference of modality-specific knowledge acquisition and modality-common knowledge anti-forgetting, where conflicting knowledge leads to collaborative forgetting. To address the above problems, this paper proposes a Cross-modality Knowledge Disentanglement and Alignment method, called CKDA, which explicitly separates and preserves modality-specific knowledge and modality-common knowledge in a balanced way. Specifically, a Modality-Common Prompting (MCP) module and a Modality-Specific Prompting (MSP) module are proposed to explicitly disentangle and purify discriminative information that coexists and is specific to different modalities, avoiding the mutual interference between both knowledge. In addition, a Cross-modal Knowledge Alignment (CKA) module is designed to further align the disentangled new knowledge with the old one in two mutually independent inter- and intra-modality feature spaces based on dual-modality prototypes in a balanced manner. Extensive experiments on four benchmark datasets verify the effectiveness and superiority of our CKDA against state-of-the-art methods. The source code of this paper is available at https://github.com/PKU-ICST-MIPL/CKDA-AAAI2026.
>
---
#### [new 003] D4C: Data-free Quantization for Contrastive Language-Image Pre-training Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对视觉语言模型CLIP的无数据量化问题，提出D4C框架。通过语义注入、结构对比生成和扰动增强三组件，合成高质量伪图像，显著提升量化后模型性能，解决现有方法因语义不足和多样性差导致的精度下降问题。**

- **链接: [https://arxiv.org/pdf/2511.15411v1](https://arxiv.org/pdf/2511.15411v1)**

> **作者:** Wenlun Zhang; Yunshan Zhong; Zihao Ding; Xinyu Li; Kentaro Yoshioka
>
> **摘要:** Data-Free Quantization (DFQ) offers a practical solution for model compression without requiring access to real data, making it particularly attractive in privacy-sensitive scenarios. While DFQ has shown promise for unimodal models, its extension to Vision-Language Models such as Contrastive Language-Image Pre-training (CLIP) models remains underexplored. In this work, we reveal that directly applying existing DFQ techniques to CLIP results in substantial performance degradation due to two key limitations: insufficient semantic content and low intra-image diversity in synthesized samples. To tackle these challenges, we propose D4C, the first DFQ framework tailored for CLIP. D4C synthesizes semantically rich and structurally diverse pseudo images through three key components: (1) Prompt-Guided Semantic Injection aligns generated images with real-world semantics using text prompts; (2) Structural Contrastive Generation reproduces compositional structures of natural images by leveraging foreground-background contrastive synthesis; and (3) Perturbation-Aware Enhancement applies controlled perturbations to improve sample diversity and robustness. These components jointly empower D4C to synthesize images that are both semantically informative and structurally diverse, effectively bridging the performance gap of DFQ on CLIP. Extensive experiments validate the effectiveness of D4C, showing significant performance improvements on various bit-widths and models. For example, under the W4A8 setting with CLIP ResNet-50 and ViT-B/32, D4C achieves Top-1 accuracy improvement of 12.4% and 18.9% on CIFAR-10, 6.8% and 19.7% on CIFAR-100, and 1.4% and 5.7% on ImageNet-1K in zero-shot classification, respectively.
>
---
#### [new 004] When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦于牙科龋齿分割任务，旨在解决低对比度、形态多样性和数据稀缺带来的挑战。通过在DC1000数据集上对比12种先进架构，发现CNN模型（如DoubleU-Net）表现最优，表明任务特性比模型复杂度更重要。**

- **链接: [https://arxiv.org/pdf/2511.14860v1](https://arxiv.org/pdf/2511.14860v1)**

> **作者:** Aashish Ghimire; Jun Zeng; Roshan Paudel; Nikhil Kumar Tomar; Deepak Ranjan Nayak; Harshith Reddy Nalla; Vivek Jha; Glenda Reynolds; Debesh Jha
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Accurate identification and segmentation of dental caries in panoramic radiographs are critical for early diagnosis and effective treatment planning. Automated segmentation remains challenging due to low lesion contrast, morphological variability, and limited annotated data. In this study, we present the first comprehensive benchmarking of convolutional neural networks, vision transformers and state-space mamba architectures for automated dental caries segmentation on panoramic radiographs through a DC1000 dataset. Twelve state-of-the-art architectures, including VMUnet, MambaUNet, VMUNetv2, RMAMamba-S, TransNetR, PVTFormer, DoubleU-Net, and ResUNet++, were trained under identical configurations. Results reveal that, contrary to the growing trend toward complex attention based architectures, the CNN-based DoubleU-Net achieved the highest dice coefficient of 0.7345, mIoU of 0.5978, and precision of 0.8145, outperforming all transformer and Mamba variants. In the study, the top 3 results across all performance metrics were achieved by CNN-based architectures. Here, Mamba and transformer-based methods, despite their theoretical advantage in global context modeling, underperformed due to limited data and weaker spatial priors. These findings underscore the importance of architecture-task alignment in domain-specific medical image segmentation more than model complexity. Our code is available at: https://github.com/JunZengz/dental-caries-segmentation.
>
---
#### [new 005] A Multimodal Transformer Approach for UAV Detection and Aerial Object Recognition Using Radar, Audio, and Video Data
- **分类: cs.CV**

- **简介: 论文提出一种多模态Transformer模型，融合雷达、视频（RGB/红外）和音频数据，用于无人机检测与空中目标识别。解决单模态方法精度不足问题，实现高准确率与实时性，显著提升复杂空域下的检测性能。**

- **链接: [https://arxiv.org/pdf/2511.15312v1](https://arxiv.org/pdf/2511.15312v1)**

> **作者:** Mauro Larrat; Claudomiro Sales
>
> **备注:** 23 pages, 7 figures
>
> **摘要:** Unmanned aerial vehicle (UAV) detection and aerial object recognition are critical for modern surveillance and security, prompting a need for robust systems that overcome limitations of single-modality approaches. This research addresses these challenges by designing and rigorously evaluating a novel multimodal Transformer model that integrates diverse data streams: radar, visual band video (RGB), infrared (IR) video, and audio. The architecture effectively fuses distinct features from each modality, leveraging the Transformer's self-attention mechanisms to learn comprehensive, complementary, and highly discriminative representations for classification. The model demonstrated exceptional performance on an independent test set, achieving macro-averaged metrics of 0.9812 accuracy, 0.9873 recall, 0.9787 precision, 0.9826 F1-score, and 0.9954 specificity. Notably, it exhibited particularly high precision and recall in distinguishing drones from other aerial objects. Furthermore, computational analysis confirmed its efficiency, with 1.09 GFLOPs, 1.22 million parameters, and an inference speed of 41.11 FPS, highlighting its suitability for real-time applications. This study presents a significant advancement in aerial object classification, validating the efficacy of multimodal data fusion via a Transformer architecture for achieving state-of-the-art performance, thereby offering a highly accurate and resilient solution for UAV detection and monitoring in complex airspace.
>
---
#### [new 006] Multi-Stage Residual-Aware Unsupervised Deep Learning Framework for Consistent Ultrasound Strain Elastography
- **分类: cs.CV**

- **简介: 该论文提出MUSSE-Net框架，用于超声应变弹性成像的稳定估计。针对噪声干扰和应变不一致问题，设计多阶段残差感知网络，提升图像质量和诊断可靠性。**

- **链接: [https://arxiv.org/pdf/2511.15640v1](https://arxiv.org/pdf/2511.15640v1)**

> **作者:** Shourov Joarder; Tushar Talukder Showrav; Md. Kamrul Hasan
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Ultrasound Strain Elastography (USE) is a powerful non-invasive imaging technique for assessing tissue mechanical properties, offering crucial diagnostic value across diverse clinical applications. However, its clinical application remains limited by tissue decorrelation noise, scarcity of ground truth, and inconsistent strain estimation under different deformation conditions. Overcoming these barriers, we propose MUSSE-Net, a residual-aware, multi-stage unsupervised sequential deep learning framework designed for robust and consistent strain estimation. At its backbone lies our proposed USSE-Net, an end-to-end multi-stream encoder-decoder architecture that parallelly processes pre- and post-deformation RF sequences to estimate displacement fields and axial strains. The novel architecture incorporates Context-Aware Complementary Feature Fusion (CACFF)-based encoder with Tri-Cross Attention (TCA) bottleneck with a Cross-Attentive Fusion (CAF)-based sequential decoder. To ensure temporal coherence and strain stability across varying deformation levels, this architecture leverages a tailored consistency loss. Finally, with the MUSSE-Net framework, a secondary residual refinement stage further enhances accuracy and suppresses noise. Extensive validation on simulation, in vivo, and private clinical datasets from Bangladesh University of Engineering and Technology (BUET) medical center, demonstrates MUSSE-Net's outperformed existing unsupervised approaches. On MUSSE-Net achieves state-of-the-art performance with a target SNR of 24.54, background SNR of 132.76, CNR of 59.81, and elastographic SNR of 9.73 on simulation data. In particular, on the BUET dataset, MUSSE-Net produces strain maps with enhanced lesion-to-background contrast and significant noise suppression yielding clinically interpretable strain patterns.
>
---
#### [new 007] AVATAAR: Agentic Video Answering via Temporal Adaptive Alignment and Reasoning
- **分类: cs.CV**

- **简介: 该论文提出AVATAAR框架，用于长视频问答任务，解决LVLM在复杂查询上理解不足的问题。通过全局局部上下文融合、预检索思维代理和重思模块的反馈循环，提升准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.15578v1](https://arxiv.org/pdf/2511.15578v1)**

> **作者:** Urjitkumar Patel; Fang-Chun Yeh; Chinmay Gondhalekar
>
> **备注:** Accepted in the 5th IEEE Big Data Workshop on Multimodal AI (MMAI 2025), Dec 8-11, Macau, China, 2025 (Preprint Copy)
>
> **摘要:** With the increasing prevalence of video content, effectively understanding and answering questions about long form videos has become essential for numerous applications. Although large vision language models (LVLMs) have enhanced performance, they often face challenges with nuanced queries that demand both a comprehensive understanding and detailed analysis. To overcome these obstacles, we introduce AVATAAR, a modular and interpretable framework that combines global and local video context, along with a Pre Retrieval Thinking Agent and a Rethink Module. AVATAAR creates a persistent global summary and establishes a feedback loop between the Rethink Module and the Pre Retrieval Thinking Agent, allowing the system to refine its retrieval strategies based on partial answers and replicate human-like iterative reasoning. On the CinePile benchmark, AVATAAR demonstrates significant improvements over a baseline, achieving relative gains of +5.6% in temporal reasoning, +5% in technical queries, +8% in theme-based questions, and +8.2% in narrative comprehension. Our experiments confirm that each module contributes positively to the overall performance, with the feedback loop being crucial for adaptability. These findings highlight AVATAAR's effectiveness in enhancing video understanding capabilities. Ultimately, AVATAAR presents a scalable solution for long-form Video Question Answering (QA), merging accuracy, interpretability, and extensibility.
>
---
#### [new 008] Reasoning via Video: The First Evaluation of Video Models' Reasoning Abilities through Maze-Solving Tasks
- **分类: cs.CV; cs.AI**

- **简介: 论文提出视频模型通过迷宫求解任务进行空间推理的新范式，构建VR-Bench基准评估其推理能力。发现SFT可有效激发推理能力，视频模型在空间感知上优于视觉语言模型，并具良好泛化性与测试时缩放效应。**

- **链接: [https://arxiv.org/pdf/2511.15065v1](https://arxiv.org/pdf/2511.15065v1)**

> **作者:** Cheng Yang; Haiyuan Wan; Yiran Peng; Xin Cheng; Zhaoyang Yu; Jiayi Zhang; Junchi Yu; Xinlei Yu; Xiawu Zheng; Dongzhan Zhou; Chenglin Wu
>
> **摘要:** Video Models have achieved remarkable success in high-fidelity video generation with coherent motion dynamics. Analogous to the development from text generation to text-based reasoning in language modeling, the development of video models motivates us to ask: Can video models reason via video generation? Compared with the discrete text corpus, video grounds reasoning in explicit spatial layouts and temporal continuity, which serves as an ideal substrate for spatial reasoning. In this work, we explore the reasoning via video paradigm and introduce VR-Bench -- a comprehensive benchmark designed to systematically evaluate video models' reasoning capabilities. Grounded in maze-solving tasks that inherently require spatial planning and multi-step reasoning, VR-Bench contains 7,920 procedurally generated videos across five maze types and diverse visual styles. Our empirical analysis demonstrates that SFT can efficiently elicit the reasoning ability of video model. Video models exhibit stronger spatial perception during reasoning, outperforming leading VLMs and generalizing well across diverse scenarios, tasks, and levels of complexity. We further discover a test-time scaling effect, where diverse sampling during inference improves reasoning reliability by 10--20%. These findings highlight the unique potential and scalability of reasoning via video for spatial reasoning tasks.
>
---
#### [new 009] Taming Generative Synthetic Data for X-ray Prohibited Item Detection
- **分类: cs.CV**

- **简介: 论文提出Xsyn，一种无需额外人工成本的一阶段X-ray违禁品检测图像合成方法，解决数据稀缺问题。通过交叉注意力优化与背景遮挡建模提升合成图像质量，显著改善检测性能。**

- **链接: [https://arxiv.org/pdf/2511.15299v1](https://arxiv.org/pdf/2511.15299v1)**

> **作者:** Jialong Sun; Hongguang Zhu; Weizhe Liu; Yunda Sun; Renshuai Tao; Yunchao Wei
>
> **摘要:** Training prohibited item detection models requires a large amount of X-ray security images, but collecting and annotating these images is time-consuming and laborious. To address data insufficiency, X-ray security image synthesis methods composite images to scale up datasets. However, previous methods primarily follow a two-stage pipeline, where they implement labor-intensive foreground extraction in the first stage and then composite images in the second stage. Such a pipeline introduces inevitable extra labor cost and is not efficient. In this paper, we propose a one-stage X-ray security image synthesis pipeline (Xsyn) based on text-to-image generation, which incorporates two effective strategies to improve the usability of synthetic images. The Cross-Attention Refinement (CAR) strategy leverages the cross-attention map from the diffusion model to refine the bounding box annotation. The Background Occlusion Modeling (BOM) strategy explicitly models background occlusion in the latent space to enhance imaging complexity. To the best of our knowledge, compared with previous methods, Xsyn is the first to achieve high-quality X-ray security image synthesis without extra labor cost. Experiments demonstrate that our method outperforms all previous methods with 1.2% mAP improvement, and the synthetic images generated by our method are beneficial to improve prohibited item detection performance across various X-ray security datasets and detectors. Code is available at https://github.com/pILLOW-1/Xsyn/.
>
---
#### [new 010] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking
- **分类: cs.CV; cs.AI**

- **简介: 论文提出CompTrack框架，用于LiDAR点云中的3D单目标跟踪任务。针对点云稀疏导致的空间与信息冗余问题，设计空间前景预测模块和信息瓶颈引导的动态令牌压缩模块，实现高效精准跟踪。**

- **链接: [https://arxiv.org/pdf/2511.15580v1](https://arxiv.org/pdf/2511.15580v1)**

> **作者:** Sifan Zhou; Yichao Cao; Jiahao Nie; Yuqian Fu; Ziyu Zhao; Xiaobo Lu; Shuo Wang
>
> **备注:** Accepted by AAAI 2026 (Oral)
>
> **摘要:** 3D single object tracking (SOT) in LiDAR point clouds is a critical task in computer vision and autonomous driving. Despite great success having been achieved, the inherent sparsity of point clouds introduces a dual-redundancy challenge that limits existing trackers: (1) vast spatial redundancy from background noise impairs accuracy, and (2) informational redundancy within the foreground hinders efficiency. To tackle these issues, we propose CompTrack, a novel end-to-end framework that systematically eliminates both forms of redundancy in point clouds. First, CompTrack incorporates a Spatial Foreground Predictor (SFP) module to filter out irrelevant background noise based on information entropy, addressing spatial redundancy. Subsequently, its core is an Information Bottleneck-guided Dynamic Token Compression (IB-DTC) module that eliminates the informational redundancy within the foreground. Theoretically grounded in low-rank approximation, this module leverages an online SVD analysis to adaptively compress the redundant foreground into a compact and highly informative set of proxy tokens. Extensive experiments on KITTI, nuScenes and Waymo datasets demonstrate that CompTrack achieves top-performing tracking performance with superior efficiency, running at a real-time 90 FPS on a single RTX 3090 GPU.
>
---
#### [new 011] Artificial intelligence approaches for energy-efficient laser cutting machines
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 论文提出基于深度学习的自适应控制方法，解决激光切割机能耗高、环保差的问题。通过材料识别与烟雾检测模型动态调节抽风泵功率，实现20%–50%节能。**

- **链接: [https://arxiv.org/pdf/2511.14952v1](https://arxiv.org/pdf/2511.14952v1)**

> **作者:** Mohamed Abdallah Salem; Hamdy Ahmed Ashour; Ahmed Elshenawy
>
> **摘要:** This research addresses the significant challenges of energy consumption and environmental impact in laser cutting by proposing novel deep learning (DL) methodologies to achieve energy reduction. Recognizing the current lack of adaptive control and the open-loop nature of CO2 laser suction pumps, this study utilizes closed-loop configurations that dynamically adjust pump power based on both the material being cut and the smoke level generated. To implement this adaptive system, diverse material classification methods are introduced, including techniques leveraging lens-less speckle sensing with a customized Convolutional Neural Network (CNN) and an approach using a USB camera with transfer learning via the pre-trained VGG16 CNN model. Furthermore, a separate DL model for smoke level detection is employed to simultaneously refine the pump's power output. This integration prompts the exhaust suction pump to automatically halt during inactive times and dynamically adjust power during operation, leading to experimentally proven and remarkable energy savings, with results showing a 20% to 50% reduction in the smoke suction pump's energy consumption, thereby contributing substantially to sustainable development in the manufacturing sector.
>
---
#### [new 012] InstructMix2Mix: Consistent Sparse-View Editing Through Multi-View Model Personalization
- **分类: cs.CV**

- **简介: 论文提出InstructMix2Mix框架，解决稀疏视图下多视角图像编辑的一致性问题。通过将2D扩散模型的编辑能力蒸馏到多视角扩散模型中，利用其3D先验实现跨视角一致性，改进了现有方法在编辑质量和一致性上的不足。**

- **链接: [https://arxiv.org/pdf/2511.14899v1](https://arxiv.org/pdf/2511.14899v1)**

> **作者:** Daniel Gilo; Or Litany
>
> **摘要:** We address the task of multi-view image editing from sparse input views, where the inputs can be seen as a mix of images capturing the scene from different viewpoints. The goal is to modify the scene according to a textual instruction while preserving consistency across all views. Existing methods, based on per-scene neural fields or temporal attention mechanisms, struggle in this setting, often producing artifacts and incoherent edits. We propose InstructMix2Mix (I-Mix2Mix), a framework that distills the editing capabilities of a 2D diffusion model into a pretrained multi-view diffusion model, leveraging its data-driven 3D prior for cross-view consistency. A key contribution is replacing the conventional neural field consolidator in Score Distillation Sampling (SDS) with a multi-view diffusion student, which requires novel adaptations: incremental student updates across timesteps, a specialized teacher noise scheduler to prevent degeneration, and an attention modification that enhances cross-view coherence without additional cost. Experiments demonstrate that I-Mix2Mix significantly improves multi-view consistency while maintaining high per-frame edit quality.
>
---
#### [new 013] Generating Natural-Language Surgical Feedback: From Structured Representation to Domain-Grounded Evaluation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出结构感知的手术反馈生成方法，解决自动化生成高质量、临床可信反馈的问题。通过挖掘IAT三元组构建手术动作本体，结合视频理解与GPT-4o生成反馈，显著提升反馈的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.15159v1](https://arxiv.org/pdf/2511.15159v1)**

> **作者:** Firdavs Nasriddinov; Rafal Kocielnik; Anima Anandkumar; Andrew J. Hung
>
> **备注:** Accepted as proceedings paper for ML4H 2025
>
> **摘要:** High-quality intraoperative feedback from a surgical trainer is pivotal for improving trainee performance and long-term skill acquisition. Automating natural, trainer-style feedback promises timely, accessible, and consistent guidance at scale but requires models that understand clinically relevant representations. We present a structure-aware pipeline that learns a surgical action ontology from real trainer-to-trainee transcripts (33 surgeries) and uses it to condition feedback generation. We contribute by (1) mining Instrument-Action-Target (IAT) triplets from real-world feedback text and clustering surface forms into normalized categories, (2) fine-tuning a video-to-IAT model that leverages the surgical procedure and task contexts as well as fine-grained temporal instrument motion, and (3) demonstrating how to effectively use IAT triplet representations to guide GPT-4o in generating clinically grounded, trainer-style feedback. We show that, on Task 1: Video-to-IAT recognition, our context injection and temporal tracking deliver consistent AUC gains (Instrument: 0.67 to 0.74; Action: 0.60 to 0.63; Tissue: 0.74 to 0.79). For Task 2: feedback text generation (rated on a 1-5 fidelity rubric where 1 = opposite/unsafe, 3 = admissible, and 5 = perfect match to a human trainer), GPT-4o from video alone scores 2.17, while IAT conditioning reaches 2.44 (+12.4%), doubling the share of admissible generations with score >= 3 from 21% to 42%. Traditional text-similarity metrics also improve: word error rate decreases by 15-31% and ROUGE (phrase/substring overlap) increases by 9-64%. Grounding generation in explicit IAT structure improves fidelity and yields clinician-verifiable rationales, supporting auditable use in surgical training.
>
---
#### [new 014] MambaTrack3D: A State Space Model Framework for LiDAR-Based Object Tracking under High Temporal Variation
- **分类: cs.CV**

- **简介: 论文提出MambaTrack3D，用于LiDAR点云中高时间变化环境下的3D目标跟踪任务。针对现有方法计算复杂度高、冗余信息多等问题，设计了基于Mamba的跨帧传播模块和分组特征增强模块，提升效率与精度，兼顾HTV与常规场景表现。**

- **链接: [https://arxiv.org/pdf/2511.15077v1](https://arxiv.org/pdf/2511.15077v1)**

> **作者:** Shengjing Tian; Yinan Han; Xiantong Zhao; Xuehu Liu; Qi Lang
>
> **备注:** This work has been submitted to a journal for possible publication
>
> **摘要:** Dynamic outdoor environments with high temporal variation (HTV) pose significant challenges for 3D single object tracking in LiDAR point clouds. Existing memory-based trackers often suffer from quadratic computational complexity, temporal redundancy, and insufficient exploitation of geometric priors. To address these issues, we propose MambaTrack3D, a novel HTV-oriented tracking framework built upon the state space model Mamba. Specifically, we design a Mamba-based Inter-frame Propagation (MIP) module that replaces conventional single-frame feature extraction with efficient inter-frame propagation, achieving near-linear complexity while explicitly modeling spatial relations across historical frames. Furthermore, a Grouped Feature Enhancement Module (GFEM) is introduced to separate foreground and background semantics at the channel level, thereby mitigating temporal redundancy in the memory bank. Extensive experiments on KITTI-HTV and nuScenes-HTV benchmarks demonstrate that MambaTrack3D consistently outperforms both HTV-oriented and normal-scenario trackers, achieving improvements of up to 6.5 success and 9.5 precision over HVTrack under moderate temporal gaps. On the standard KITTI dataset, MambaTrack3D remains highly competitive with state-of-the-art normal-scenario trackers, confirming its strong generalization ability. Overall, MambaTrack3D achieves a superior accuracy-efficiency trade-off, delivering robust performance across both specialized HTV and conventional tracking scenarios.
>
---
#### [new 015] FinCriticalED: A Visual Benchmark for Financial Fact-Level OCR Evaluation
- **分类: cs.CV**

- **简介: 论文提出FinCriticalED，一个用于金融文档OCR和视觉语言模型的事实级评估基准，解决传统指标无法捕捉关键事实错误的问题。工作包括构建500个图像-HTML对、专家标注700+关键事实，并开发LLM-as-Judge验证流程，推动视觉事实精度提升。**

- **链接: [https://arxiv.org/pdf/2511.14998v1](https://arxiv.org/pdf/2511.14998v1)**

> **作者:** Yueru He; Xueqing Peng; Yupeng Cao; Yan Wang; Lingfei Qian; Haohang Li; Yi Han; Ruoyu Xiang; Mingquan Lin; Prayag Tiwari; Jimin Huang; Guojun Xiong; Sophia Ananiadou
>
> **备注:** Yueru He, Xueqing Peng: These two authors contributed equally to this work
>
> **摘要:** We introduce FinCriticalED (Financial Critical Error Detection), a visual benchmark for evaluating OCR and vision language models on financial documents at the fact level. Financial documents contain visually dense and table heavy layouts where numerical and temporal information is tightly coupled with structure. In high stakes settings, small OCR mistakes such as sign inversion or shifted dates can lead to materially different interpretations, while traditional OCR metrics like ROUGE and edit distance capture only surface level text similarity. \ficriticaled provides 500 image-HTML pairs with expert annotated financial facts covering over seven hundred numerical and temporal facts. It introduces three key contributions. First, it establishes the first fact level evaluation benchmark for financial document understanding, shifting evaluation from lexical overlap to domain critical factual correctness. Second, all annotations are created and verified by financial experts with strict quality control over signs, magnitudes, and temporal expressions. Third, we develop an LLM-as-Judge evaluation pipeline that performs structured fact extraction and contextual verification for visually complex financial documents. We benchmark OCR systems, open source vision language models, and proprietary models on FinCriticalED. Results show that although the strongest proprietary models achieve the highest factual accuracy, substantial errors remain in visually intricate numerical and temporal contexts. Through quantitative evaluation and expert case studies, FinCriticalED provides a rigorous foundation for advancing visual factual precision in financial and other precision critical domains.
>
---
#### [new 016] GEO-Bench-2: From Performance to Capability, Rethinking Evaluation in Geospatial AI
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出GEO-Bench-2，用于评估地理空间基础模型（GeoFMs）的能力。针对缺乏标准化评估的问题，构建涵盖多任务的基准框架，并按数据特性分组能力，揭示不同模型在特定任务上的优势，指导模型选择与改进方向。**

- **链接: [https://arxiv.org/pdf/2511.15658v1](https://arxiv.org/pdf/2511.15658v1)**

> **作者:** Naomi Simumba; Nils Lehmann; Paolo Fraccaro; Hamed Alemohammad; Geeth De Mel; Salman Khan; Manil Maskey; Nicolas Longepe; Xiao Xiang Zhu; Hannah Kerner; Juan Bernabe-Moreno; Alexander Lacoste
>
> **摘要:** Geospatial Foundation Models (GeoFMs) are transforming Earth Observation (EO), but evaluation lacks standardized protocols. GEO-Bench-2 addresses this with a comprehensive framework spanning classification, segmentation, regression, object detection, and instance segmentation across 19 permissively-licensed datasets. We introduce ''capability'' groups to rank models on datasets that share common characteristics (e.g., resolution, bands, temporality). This enables users to identify which models excel in each capability and determine which areas need improvement in future work. To support both fair comparison and methodological innovation, we define a prescriptive yet flexible evaluation protocol. This not only ensures consistency in benchmarking but also facilitates research into model adaptation strategies, a key and open challenge in advancing GeoFMs for downstream tasks. Our experiments show that no single model dominates across all tasks, confirming the specificity of the choices made during architecture design and pretraining. While models pretrained on natural images (ConvNext ImageNet, DINO V3) excel on high-resolution tasks, EO-specific models (TerraMind, Prithvi, and Clay) outperform them on multispectral applications such as agriculture and disaster response. These findings demonstrate that optimal model choice depends on task requirements, data modalities, and constraints. This shows that the goal of a single GeoFM model that performs well across all tasks remains open for future research. GEO-Bench-2 enables informed, reproducible GeoFM evaluation tailored to specific use cases. Code, data, and leaderboard for GEO-Bench-2 are publicly released under a permissive license.
>
---
#### [new 017] nnMIL: A generalizable multiple instance learning framework for computational pathology
- **分类: cs.CV**

- **简介: 该论文提出nnMIL框架，解决病理图像中patch到slide的聚合问题，提升模型泛化性和可靠性。通过随机采样和轻量聚合器，实现高效训练与不确定性估计，在35项临床任务中优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.14907v1](https://arxiv.org/pdf/2511.14907v1)**

> **作者:** Xiangde Luo; Jinxi Xiang; Yuanfeng Ji; Ruijiang Li
>
> **备注:** A conceptual evaluation work; more studies are in progress; examples are here (https://github.com/Luoxd1996/nnMIL)
>
> **摘要:** Computational pathology holds substantial promise for improving diagnosis and guiding treatment decisions. Recent pathology foundation models enable the extraction of rich patch-level representations from large-scale whole-slide images (WSIs), but current approaches for aggregating these features into slide-level predictions remain constrained by design limitations that hinder generalizability and reliability. Here, we developed nnMIL, a simple yet broadly applicable multiple-instance learning framework that connects patch-level foundation models to robust slide-level clinical inference. nnMIL introduces random sampling at both the patch and feature levels, enabling large-batch optimization, task-aware sampling strategies, and efficient and scalable training across datasets and model architectures. A lightweight aggregator performs sliding-window inference to generate ensemble slide-level predictions and supports principled uncertainty estimation. Across 40,000 WSIs encompassing 35 clinical tasks and four pathology foundation models, nnMIL consistently outperformed existing MIL methods for disease diagnosis, histologic subtyping, molecular biomarker detection, and pan- cancer prognosis prediction. It further demonstrated strong cross-model generalization, reliable uncertainty quantification, and robust survival stratification in multiple external cohorts. In conclusion, nnMIL offers a practical and generalizable solution for translating pathology foundation models into clinically meaningful predictions, advancing the development and deployment of reliable AI systems in real-world settings.
>
---
#### [new 018] B-Rep Distance Functions (BR-DF): How to Represent a B-Rep Model by Volumetric Distance Functions?
- **分类: cs.CV; cs.AI**

- **简介: 论文提出B-Rep Distance Functions（BR-DF），将CAD模型的几何与拓扑信息编码为体积距离函数，通过改进的Marching Cubes算法实现100%成功率生成闭合B-Rep模型，并利用多分支扩散模型联合生成SDF和面级UDF。该方法解决了CAD模型生成中失败率高的问题，属于CAD生成任务。**

- **链接: [https://arxiv.org/pdf/2511.14870v1](https://arxiv.org/pdf/2511.14870v1)**

> **作者:** Fuyang Zhang; Pradeep Kumar Jayaraman; Xiang Xu; Yasutaka Furukawa
>
> **备注:** Project page: https://zhangfuyang.github.io/brdf/
>
> **摘要:** This paper presents a novel geometric representation for CAD Boundary Representation (B-Rep) based on volumetric distance functions, dubbed B-Rep Distance Functions (BR-DF). BR-DF encodes the surface mesh geometry of a CAD model as signed distance function (SDF). B-Rep vertices, edges, faces and their topology information are encoded as per-face unsigned distance functions (UDFs). An extension of the Marching Cubes algorithm converts BR-DF directly into watertight CAD B-Rep model (strictly speaking a faceted B-Rep model). A surprising characteristic of BR-DF is that this conversion process never fails. Leveraging the volumetric nature of BR-DF, we propose a multi-branch latent diffusion with 3D U-Net backbone for jointly generating the SDF and per-face UDFs of a BR-DF model. Our approach achieves comparable CAD generation performance against SOTA methods while reaching the unprecedented 100% success rate in producing (faceted) B-Rep models.
>
---
#### [new 019] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
- **分类: cs.CV; cs.CL**

- **简介: 论文提出MoDES框架，用于加速多模态大语言模型中的专家混合（MoE）推理。针对现有方法在多模态场景下性能下降的问题，MoDES通过全局调制局部门控和双模态阈值策略，实现精准专家跳过，显著提升效率与准确率。**

- **链接: [https://arxiv.org/pdf/2511.15690v1](https://arxiv.org/pdf/2511.15690v1)**

> **作者:** Yushi Huang; Zining Wang; Zhihang Yuan; Yifu Ding; Ruihao Gong; Jinyang Guo; Xianglong Liu; Jun Zhang
>
> **备注:** Code will be released upon acceptance
>
> **摘要:** Mixture-of-Experts (MoE) Multimodal large language models (MLLMs) excel at vision-language tasks, but they suffer from high computational inefficiency. To reduce inference overhead, expert skipping methods have been proposed to deactivate redundant experts based on the current input tokens. However, we find that applying these methods-originally designed for unimodal large language models (LLMs)-to MLLMs results in considerable performance degradation. This is primarily because such methods fail to account for the heterogeneous contributions of experts across MoE layers and modality-specific behaviors of tokens within these layers. Motivated by these findings, we propose MoDES, the first training-free framework that adaptively skips experts to enable efficient and accurate MoE MLLM inference. It incorporates a globally-modulated local gating (GMLG) mechanism that integrates global layer-wise importance into local routing probabilities to accurately estimate per-token expert importance. A dual-modality thresholding (DMT) method is then applied, which processes tokens from each modality separately, to derive the skipping schedule. To set the optimal thresholds, we introduce a frontier search algorithm that exploits monotonicity properties, cutting convergence time from several days to a few hours. Extensive experiments for 3 model series across 13 benchmarks demonstrate that MoDES far outperforms previous approaches. For instance, when skipping 88% experts for Qwen3-VL-MoE-30B-A3B-Instruct, the performance boost is up to 10.67% (97.33% vs. 86.66%). Furthermore, MoDES significantly enhances inference speed, improving the prefilling time by 2.16$\times$ and the decoding time by 1.26$\times$.
>
---
#### [new 020] GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization
- **分类: cs.CV**

- **简介: 论文提出GeoVista，一个用于地理定位的智能体模型，通过整合图像缩放与网络搜索工具，提升多模态推理能力。针对现有基准不足，构建GeoBench，并采用监督微调与强化学习训练，显著优于开源模型，接近闭源模型表现。**

- **链接: [https://arxiv.org/pdf/2511.15705v1](https://arxiv.org/pdf/2511.15705v1)**

> **作者:** Yikun Wang; Zuyan Liu; Ziyi Wang; Pengfei Liu; Han Hu; Yongming Rao
>
> **摘要:** Current research on agentic visual reasoning enables deep multimodal understanding but primarily focuses on image manipulation tools, leaving a gap toward more general-purpose agentic models. In this work, we revisit the geolocalization task, which requires not only nuanced visual grounding but also web search to confirm or refine hypotheses during reasoning. Since existing geolocalization benchmarks fail to meet the need for high-resolution imagery and the localization challenge for deep agentic reasoning, we curate GeoBench, a benchmark that includes photos and panoramas from around the world, along with a subset of satellite images of different cities to rigorously evaluate the geolocalization ability of agentic models. We also propose GeoVista, an agentic model that seamlessly integrates tool invocation within the reasoning loop, including an image-zoom-in tool to magnify regions of interest and a web-search tool to retrieve related web information. We develop a complete training pipeline for it, including a cold-start supervised fine-tuning (SFT) stage to learn reasoning patterns and tool-use priors, followed by a reinforcement learning (RL) stage to further enhance reasoning ability. We adopt a hierarchical reward to leverage multi-level geographical information and improve overall geolocalization performance. Experimental results show that GeoVista surpasses other open-source agentic models on the geolocalization task greatly and achieves performance comparable to closed-source models such as Gemini-2.5-flash and GPT-5 on most metrics.
>
---
#### [new 021] MambaIO: Global-Coordinate Inertial Odometry for Pedestrians via Multi-Scale Frequency-Decoupled Modeling
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于行人惯性里程计（IO）任务，旨在解决传统全局坐标系下定位精度不足的问题。通过理论分析与实验验证，提出MambaIO模型，利用多尺度频域解耦策略，结合Mamba与卷积结构分别处理低频上下文和高频细节，显著提升定位精度。**

- **链接: [https://arxiv.org/pdf/2511.15645v1](https://arxiv.org/pdf/2511.15645v1)**

> **作者:** Shanshan Zhang
>
> **摘要:** Inertial Odometry (IO) enables real-time localization using only acceleration and angular velocity measurements from an Inertial Measurement Unit (IMU), making it a promising solution for localization in consumer-grade applications. Traditionally, IMU measurements in IO have been processed under two coordinate system paradigms: the body coordinate frame and the global coordinate frame, with the latter being widely adopted. However, recent studies in drone scenarios have demonstrated that the body frame can significantly improve localization accuracy, prompting a re-evaluation of the suitability of the global frame for pedestrian IO. To address this issue, this paper systematically evaluates the effectiveness of the global coordinate frame in pedestrian IO through theoretical analysis, qualitative inspection, and quantitative experiments. Building upon these findings, we further propose MambaIO, which decomposes IMU measurements into high-frequency and low-frequency components using a Laplacian pyramid. The low-frequency component is processed by a Mamba architecture to extract implicit contextual motion cues, while the high-frequency component is handled by a convolutional structure to capture fine-grained local motion details. Experiments on multiple public datasets show that MambaIO substantially reduces localization error and achieves state-of-the-art (SOTA) performance. To the best of our knowledge, this is the first application of the Mamba architecture to the inertial odometry task.
>
---
#### [new 022] HULFSynth : An INR based Super-Resolution and Ultra Low-Field MRI Synthesis via Contrast factor estimation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出HULFSynth，基于隐式神经表示（INR）实现高场（HF）与超低场（ULF）MRI图像的双向无监督合成。解决HF到ULF图像转换及超分辨率问题，通过估计组织类型信噪比模拟对比度变化，提升WM-GM对比度并验证模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.14897v1](https://arxiv.org/pdf/2511.14897v1)**

> **作者:** Pranav Indrakanti; Ivor Simpson
>
> **备注:** Submitted to ISBI 2026
>
> **摘要:** We present an unsupervised single image bidirectional Magnetic Resonance Image (MRI) synthesizer that synthesizes an Ultra-Low Field (ULF) like image from a High-Field (HF) magnitude image and vice-versa. Unlike existing MRI synthesis models, our approach is inspired by the physics that drives contrast changes between HF and ULF MRIs. Our forward model simulates a HF to ULF transformation by estimating the tissue-type Signal-to-Noise ratio (SNR) values based on target contrast values. For the Super-Resolution task, we used an Implicit Neural Representation (INR) network to synthesize HF image by simultaneously predicting tissue-type segmentations and image intensity without observed HF data. The proposed method is evaluated using synthetic ULF-like data from generated from standard 3T T$_1$-weighted images for qualitative assessments and paired 3T-64mT T$_1$-weighted images for validation experiments. WM-GM contrast improved by 52% in synthetic ULF-like images and 37% in 64mT images. Sensitivity experiments demonstrated the robustness of our forward model to variations in target contrast, noise and initial seeding.
>
---
#### [new 023] Multi-Text Guided Few-Shot Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文提出MTGNet，用于少样本语义分割任务，解决单一文本提示导致目标区域激活不全、跨模态交互弱及支持特征噪声等问题。通过多文本引导机制增强文本先验与视觉先验的协同优化，提升分割精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.15515v1](https://arxiv.org/pdf/2511.15515v1)**

> **作者:** Qiang Jiao; Bin Yan; Yi Yang; Mengrui Shi; Qiang Zhang
>
> **摘要:** Recent CLIP-based few-shot semantic segmentation methods introduce class-level textual priors to assist segmentation by typically using a single prompt (e.g., a photo of class). However, these approaches often result in incomplete activation of target regions, as a single textual description cannot fully capture the semantic diversity of complex categories. Moreover, they lack explicit cross-modal interaction and are vulnerable to noisy support features, further degrading visual prior quality. To address these issues, we propose the Multi-Text Guided Few-Shot Semantic Segmentation Network (MTGNet), a dual-branch framework that enhances segmentation performance by fusing diverse textual prompts to refine textual priors and guide the cross-modal optimization of visual priors. Specifically, we design a Multi-Textual Prior Refinement (MTPR) module that suppresses interference and aggregates complementary semantic cues to enhance foreground activation and expand semantic coverage for structurally complex objects. We introduce a Text Anchor Feature Fusion (TAFF) module, which leverages multi-text embeddings as semantic anchors to facilitate the transfer of discriminative local prototypes from support images to query images, thereby improving semantic consistency and alleviating intra-class variations. Furthermore, a Foreground Confidence-Weighted Attention (FCWA) module is presented to enhance visual prior robustness by leveraging internal self-similarity within support foreground features. It adaptively down-weights inconsistent regions and effectively suppresses interference in the query segmentation process. Extensive experiments on standard FSS benchmarks validate the effectiveness of MTGNet. In the 1-shot setting, it achieves 76.8% mIoU on PASCAL-5i and 57.4% on COCO-20i, with notable improvements in folds exhibiting high intra-class variations.
>
---
#### [new 024] Hyperspectral Image Classification using Spectral-Spatial Mixer Network
- **分类: cs.CV**

- **简介: 该论文提出SS-MixNet模型，用于高光谱图像分类任务，旨在解决小样本下准确分类难题。通过融合3D卷积与并行MLP混合块，捕获局部与长程依赖，并引入轻量注意力机制提升性能，在仅1%标注数据下实现高精度分类。**

- **链接: [https://arxiv.org/pdf/2511.15692v1](https://arxiv.org/pdf/2511.15692v1)**

> **作者:** Mohammed Q. Alkhatib
>
> **备注:** Accepted for WHISPERS2025
>
> **摘要:** This paper introduces SS-MixNet, a lightweight and effective deep learning model for hyperspectral image (HSI) classification. The architecture integrates 3D convolutional layers for local spectral-spatial feature extraction with two parallel MLP-style mixer blocks that capture long-range dependencies in spectral and spatial dimensions. A depthwise convolution-based attention mechanism is employed to enhance discriminative capability with minimal computational overhead. The model is evaluated on the QUH-Tangdaowan and QUH-Qingyun datasets using only 1% of labeled data for training and validation. SS-MixNet achieves the highest performance among compared methods, including 2D-CNN, 3D-CNN, IP-SWIN, SimPoolFormer, and HybridKAN, reaching 95.68% and 93.86% overall accuracy on the Tangdaowan and Qingyun datasets, respectively. The results, supported by quantitative metrics and classification maps, confirm the model's effectiveness in delivering accurate and robust predictions with limited supervision. The code will be made publicly available at: https://github.com/mqalkhatib/SS-MixNet
>
---
#### [new 025] UniHOI: Unified Human-Object Interaction Understanding via Unified Token Space
- **分类: cs.CV; cs.AI**

- **简介: 论文提出UniHOI，统一建模人-物体交互检测与生成任务，通过统一token空间实现知识共享，解决两者分离导致的泛化不足问题，提升长尾检测准确率和开放词汇生成性能。**

- **链接: [https://arxiv.org/pdf/2511.15046v1](https://arxiv.org/pdf/2511.15046v1)**

> **作者:** Panqi Yang; Haodong Jing; Nanning Zheng; Yongqiang Ma
>
> **备注:** Accepted by AAAI 2026,9 pages, 4 figures
>
> **摘要:** In the field of human-object interaction (HOI), detection and generation are two dual tasks that have traditionally been addressed separately, hindering the development of comprehensive interaction understanding. To address this, we propose UniHOI, which jointly models HOI detection and generation via a unified token space, thereby effectively promoting knowledge sharing and enhancing generalization. Specifically, we introduce a symmetric interaction-aware attention module and a unified semi-supervised learning paradigm, enabling effective bidirectional mapping between images and interaction semantics even under limited annotations. Extensive experiments demonstrate that UniHOI achieves state-of-the-art performance in both HOI detection and generation. Specifically, UniHOI improves accuracy by 4.9% on long-tailed HOI detection and boosts interaction metrics by 42.0% on open-vocabulary generation tasks.
>
---
#### [new 026] First Frame Is the Place to Go for Video Content Customization
- **分类: cs.CV**

- **简介: 论文研究视频生成模型中第一帧的作用，发现其可作为概念记忆缓冲区，用于参考式视频定制。通过少量样本（20–50）实现多样场景下的内容定制，无需架构改动或大规模微调。**

- **链接: [https://arxiv.org/pdf/2511.15700v1](https://arxiv.org/pdf/2511.15700v1)**

> **作者:** Jingxi Chen; Zongxia Li; Zhichao Liu; Guangyao Shi; Xiyang Wu; Fuxiao Liu; Cornelia Fermuller; Brandon Y. Feng; Yiannis Aloimonos
>
> **备注:** Project Website: https://firstframego.github.io/
>
> **摘要:** What role does the first frame play in video generation models? Traditionally, it's viewed as the spatial-temporal starting point of a video, merely a seed for subsequent animation. In this work, we reveal a fundamentally different perspective: video models implicitly treat the first frame as a conceptual memory buffer that stores visual entities for later reuse during generation. Leveraging this insight, we show that it's possible to achieve robust and generalized video content customization in diverse scenarios, using only 20-50 training examples without architectural changes or large-scale finetuning. This unveils a powerful, overlooked capability of video generation models for reference-based video customization.
>
---
#### [new 027] Text2Loc++: Generalizing 3D Point Cloud Localization from Natural Language
- **分类: cs.CV**

- **简介: 论文提出Text2Loc++，解决基于自然语言描述的3D点云子图定位问题。通过粗到精框架实现跨模态对齐，引入新数据集与多级语言复杂度，提升模型在城市场景中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.15308v1](https://arxiv.org/pdf/2511.15308v1)**

> **作者:** Yan Xia; Letian Shi; Yilin Di; Joao F. Henriques; Daniel Cremers
>
> **备注:** This paper builds upon and extends our earlier conference paper Text2Loc presented at CVPR 2024
>
> **摘要:** We tackle the problem of localizing 3D point cloud submaps using complex and diverse natural language descriptions, and present Text2Loc++, a novel neural network designed for effective cross-modal alignment between language and point clouds in a coarse-to-fine localization pipeline. To support benchmarking, we introduce a new city-scale dataset covering both color and non-color point clouds from diverse urban scenes, and organize location descriptions into three levels of linguistic complexity. In the global place recognition stage, Text2Loc++ combines a pretrained language model with a Hierarchical Transformer with Max pooling (HTM) for sentence-level semantics, and employs an attention-based point cloud encoder for spatial understanding. We further propose Masked Instance Training (MIT) to filter out non-aligned objects and improve multimodal robustness. To enhance the embedding space, we introduce Modality-aware Hierarchical Contrastive Learning (MHCL), incorporating cross-modal, submap-, text-, and instance-level losses. In the fine localization stage, we completely remove explicit text-instance matching and design a lightweight yet powerful framework based on Prototype-based Map Cloning (PMC) and a Cascaded Cross-Attention Transformer (CCAT). Extensive experiments on the KITTI360Pose dataset show that Text2Loc++ outperforms existing methods by up to 15%. In addition, the proposed model exhibits robust generalization when evaluated on the new dataset, effectively handling complex linguistic expressions and a wide variety of urban environments. The code and dataset will be made publicly available.
>
---
#### [new 028] Zero-Shot Open-Vocabulary Human Motion Grounding with Test-Time Training
- **分类: cs.CV**

- **简介: 论文提出ZOMG框架，解决开放词汇下人体动作分割问题，无需标注或微调。通过语言语义划分和软掩码优化，实现零样本动作 grounding，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.15379v1](https://arxiv.org/pdf/2511.15379v1)**

> **作者:** Yunjiao Zhou; Xinyan Chen; Junlang Qian; Lihua Xie; Jianfei Yang
>
> **摘要:** Understanding complex human activities demands the ability to decompose motion into fine-grained, semantic-aligned sub-actions. This motion grounding process is crucial for behavior analysis, embodied AI and virtual reality. Yet, most existing methods rely on dense supervision with predefined action classes, which are infeasible in open-vocabulary, real-world settings. In this paper, we propose ZOMG, a zero-shot, open-vocabulary framework that segments motion sequences into semantically meaningful sub-actions without requiring any annotations or fine-tuning. Technically, ZOMG integrates (1) language semantic partition, which leverages large language models to decompose instructions into ordered sub-action units, and (2) soft masking optimization, which learns instance-specific temporal masks to focus on frames critical to sub-actions, while maintaining intra-segment continuity and enforcing inter-segment separation, all without altering the pretrained encoder. Experiments on three motion-language datasets demonstrate state-of-the-art effectiveness and efficiency of motion grounding performance, outperforming prior methods by +8.7\% mAP on HumanML3D benchmark. Meanwhile, significant improvements also exist in downstream retrieval, establishing a new paradigm for annotation-free motion understanding.
>
---
#### [new 029] A Comprehensive Study on Visual Token Redundancy for Discrete Diffusion-based Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 论文研究离散扩散多模态大模型（dMLLM）中的视觉token冗余问题，旨在提升推理效率。发现冗余仅出现在从头训练模型的长回答任务中，提出针对性剪枝策略：层跳过加速AR转扩散模型，渐进剪枝优化从头训练模型。**

- **链接: [https://arxiv.org/pdf/2511.15098v1](https://arxiv.org/pdf/2511.15098v1)**

> **作者:** Duo Li; Zuhao Yang; Xiaoqin Zhang; Ling Shao; Shijian Lu
>
> **备注:** 14 pages, 2 figures
>
> **摘要:** Discrete diffusion-based multimodal large language models (dMLLMs) have emerged as a promising alternative to autoregressive MLLMs thanks to their advantages in parallel decoding and bidirectional context modeling, but most existing dMLLMs incur significant computational overhead during inference due to the full-sequence attention computation in each denoising step. Pioneer studies attempt to resolve this issue from a modality-agnostic perspective via key-value cache optimization or efficient sampling but most of them overlook modality-specific visual token redundancy. In this work, we conduct a comprehensive study on how visual token redundancy evolves with different dMLLM architectures and tasks and how visual token pruning affects dMLLM responses and efficiency. Specifically, our study reveals that visual redundancy emerges only in from-scratch dMLLMs while handling long-answer tasks. In addition, we validate that visual token pruning introduces non-negligible information loss in dMLLMs and only from-scratch dMLLMs can recover the lost information progressively during late denoising steps. Furthermore, our study shows that layer-skipping is promising for accelerating AR-to-diffusion dMLLMs, whereas progressive or late-step pruning is more effective for from-scratch dMLLMs. Overall, this work offers a new perspective on efficiency optimization for dMLLMs, greatly advancing their applicability across various multimodal understanding tasks.
>
---
#### [new 030] EGSA-PT:Edge-Guided Spatial Attention with Progressive Training for Monocular Depth Estimation and Segmentation of Transparent Objects
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对透明物体的单目深度估计与分割任务，提出Edge-Guided Spatial Attention（EGSA）融合机制和渐进式训练策略，通过边界信息减少任务间干扰，提升透明区域感知性能。**

- **链接: [https://arxiv.org/pdf/2511.14970v1](https://arxiv.org/pdf/2511.14970v1)**

> **作者:** Gbenga Omotara; Ramy Farag; Seyed Mohamad Ali Tousi; G. N. DeSouza
>
> **摘要:** Transparent object perception remains a major challenge in computer vision research, as transparency confounds both depth estimation and semantic segmentation. Recent work has explored multi-task learning frameworks to improve robustness, yet negative cross-task interactions often hinder performance. In this work, we introduce Edge-Guided Spatial Attention (EGSA), a fusion mechanism designed to mitigate destructive interactions by incorporating boundary information into the fusion between semantic and geometric features. On both Syn-TODD and ClearPose benchmarks, EGSA consistently improved depth accuracy over the current state of the art method (MODEST), while preserving competitive segmentation performance, with the largest improvements appearing in transparent regions. Besides our fusion design, our second contribution is a multi-modal progressive training strategy, where learning transitions from edges derived from RGB images to edges derived from predicted depth images. This approach allows the system to bootstrap learning from the rich textures contained in RGB images, and then switch to more relevant geometric content in depth maps, while it eliminates the need for ground-truth depth at training time. Together, these contributions highlight edge-guided fusion as a robust approach capable of improving transparent object perception.
>
---
#### [new 031] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models
- **分类: cs.CV**

- **简介: 论文提出Uni-Adapter，一种无需训练的在线测试时适应方法，用于提升3D视觉语言模型在分布偏移下的性能。通过动态原型学习和熵加权融合，显著改善了点云分类任务的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.15311v1](https://arxiv.org/pdf/2511.15311v1)**

> **作者:** Mehran Tamjidi; Hamidreza Dastmalchi; Mohammadreza Alimoradijazi; Ali Cheraghian; Aijun An; Morteza Saberi
>
> **备注:** Accepted by AAAI 2026. 7 pages, 4 figures
>
> **摘要:** 3D Vision-Language Foundation Models (VLFMs) have shown strong generalization and zero-shot recognition capabilities in open-world point cloud processing tasks. However, these models often underperform in practical scenarios where data are noisy, incomplete, or drawn from a different distribution than the training data. To address this, we propose Uni-Adapter, a novel training-free online test-time adaptation (TTA) strategy for 3D VLFMs based on dynamic prototype learning. We define a 3D cache to store class-specific cluster centers as prototypes, which are continuously updated to capture intra-class variability in heterogeneous data distributions. These dynamic prototypes serve as anchors for cache-based logit computation via similarity scoring. Simultaneously, a graph-based label smoothing module captures inter-prototype similarities to enforce label consistency among similar prototypes. Finally, we unify predictions from the original 3D VLFM and the refined 3D cache using entropy-weighted aggregation for reliable adaptation. Without retraining, Uni-Adapter effectively mitigates distribution shifts, achieving state-of-the-art performance on diverse 3D benchmarks over different 3D VLFMs, improving ModelNet-40C by 10.55%, ScanObjectNN-C by 8.26%, and ShapeNet-C by 4.49% over the source 3D VLFMs.
>
---
#### [new 032] ShelfOcc: Native 3D Supervision beyond LiDAR for Vision-Based Occupancy Estimation
- **分类: cs.CV**

- **简介: 论文提出ShelfOcc，一种无需LiDAR的视觉占位估计方法，通过视频生成一致的3D语义体素标签，解决2D监督导致的几何不一致和深度泄漏问题，显著提升弱监督占位估计性能。**

- **链接: [https://arxiv.org/pdf/2511.15396v1](https://arxiv.org/pdf/2511.15396v1)**

> **作者:** Simon Boeder; Fabian Gigengack; Simon Roesler; Holger Caesar; Benjamin Risse
>
> **摘要:** Recent progress in self- and weakly supervised occupancy estimation has largely relied on 2D projection or rendering-based supervision, which suffers from geometric inconsistencies and severe depth bleeding. We thus introduce ShelfOcc, a vision-only method that overcomes these limitations without relying on LiDAR. ShelfOcc brings supervision into native 3D space by generating metrically consistent semantic voxel labels from video, enabling true 3D supervision without any additional sensors or manual 3D annotations. While recent vision-based 3D geometry foundation models provide a promising source of prior knowledge, they do not work out of the box as a prediction due to sparse or noisy and inconsistent geometry, especially in dynamic driving scenes. Our method introduces a dedicated framework that mitigates these issues by filtering and accumulating static geometry consistently across frames, handling dynamic content and propagating semantic information into a stable voxel representation. This data-centric shift in supervision for weakly/shelf-supervised occupancy estimation allows the use of essentially any SOTA occupancy model architecture without relying on LiDAR data. We argue that such high-quality supervision is essential for robust occupancy learning and constitutes an important complementary avenue to architectural innovation. On the Occ3D-nuScenes benchmark, ShelfOcc substantially outperforms all previous weakly/shelf-supervised methods (up to a 34% relative improvement), establishing a new data-driven direction for LiDAR-free 3D scene understanding.
>
---
#### [new 033] Towards Unbiased Cross-Modal Representation Learning for Food Image-to-Recipe Retrieval
- **分类: cs.CV; cs.MM**

- **简介: 论文针对食物图像到菜谱检索任务中的偏见问题，提出基于因果理论的去偏方法。通过识别食材为混杂因素并引入反事实干预，改进模型以减少视觉-文本对齐偏差，提升检索准确性，在Recipe1M数据集上达到SOTA性能。**

- **链接: [https://arxiv.org/pdf/2511.15201v1](https://arxiv.org/pdf/2511.15201v1)**

> **作者:** Qing Wang; Chong-Wah Ngo; Ee-Peng Lim
>
> **摘要:** This paper addresses the challenges of learning representations for recipes and food images in the cross-modal retrieval problem. As the relationship between a recipe and its cooked dish is cause-and-effect, treating a recipe as a text source describing the visual appearance of a dish for learning representation, as the existing approaches, will create bias misleading image-and-recipe similarity judgment. Specifically, a food image may not equally capture every detail in a recipe, due to factors such as the cooking process, dish presentation, and image-capturing conditions. The current representation learning tends to capture dominant visual-text alignment while overlooking subtle variations that determine retrieval relevance. In this paper, we model such bias in cross-modal representation learning using causal theory. The causal view of this problem suggests ingredients as one of the confounder sources and a simple backdoor adjustment can alleviate the bias. By causal intervention, we reformulate the conventional model for food-to-recipe retrieval with an additional term to remove the potential bias in similarity judgment. Based on this theory-informed formulation, we empirically prove the oracle performance of retrieval on the Recipe1M dataset to be MedR=1 across the testing data sizes of 1K, 10K, and even 50K. We also propose a plug-and-play neural module, which is essentially a multi-label ingredient classifier for debiasing. New state-of-the-art search performances are reported on the Recipe1M dataset.
>
---
#### [new 034] MF-GCN: A Multi-Frequency Graph Convolutional Network for Tri-Modal Depression Detection Using Eye-Tracking, Facial, and Acoustic Features
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MF-GCN模型，用于三模态抑郁检测任务，解决现有图模型仅关注低频信息的问题。通过多频滤波模块融合眼动、面部和声学特征，提升抑郁识别准确率，在多个数据集上表现优异。**

- **链接: [https://arxiv.org/pdf/2511.15675v1](https://arxiv.org/pdf/2511.15675v1)**

> **作者:** Sejuti Rahman; Swakshar Deb; MD. Sameer Iqbal Chowdhury; MD. Jubair Ahmed Sourov; Mohammad Shamsuddin
>
> **摘要:** Eye tracking data quantifies the attentional bias towards negative stimuli that is frequently observed in depressed groups. Audio and video data capture the affective flattening and psychomotor retardation characteristic of depression. Statistical validation confirmed their significant discriminative power in distinguishing depressed from non depressed groups. We address a critical limitation of existing graph-based models that focus on low-frequency information and propose a Multi-Frequency Graph Convolutional Network (MF-GCN). This framework consists of a novel Multi-Frequency Filter Bank Module (MFFBM), which can leverage both low and high frequency signals. Extensive evaluation against traditional machine learning algorithms and deep learning frameworks demonstrates that MF-GCN consistently outperforms baselines. In binary (depressed and non depressed) classification, the model achieved a sensitivity of 0.96 and F2 score of 0.94. For the 3 class (no depression, mild to moderate depression and severe depression) classification task, the proposed method achieved a sensitivity of 0.79 and specificity of 0.87 and siginificantly suprassed other models. To validate generalizability, the model was also evaluated on the Chinese Multimodal Depression Corpus (CMDC) dataset and achieved a sensitivity of 0.95 and F2 score of 0.96. These results confirm that our trimodal, multi frequency framework effectively captures cross modal interaction for accurate depression detection.
>
---
#### [new 035] Think Visually, Reason Textually: Vision-Language Synergy in ARC
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 论文研究ARC-AGI任务，解决基础模型从少量示例中抽象推理规则的难题。提出视觉与语言协同策略（VLSR和MSSC），利用视觉抽象与语言精确执行的优势，提升模型在ARC-AGI上的表现。**

- **链接: [https://arxiv.org/pdf/2511.15703v1](https://arxiv.org/pdf/2511.15703v1)**

> **作者:** Beichen Zhang; Yuhang Zang; Xiaoyi Dong; Yuhang Cao; Haodong Duan; Dahua Lin; Jiaqi Wang
>
> **摘要:** Abstract reasoning from minimal examples remains a core unsolved problem for frontier foundation models such as GPT-5 and Grok 4. These models still fail to infer structured transformation rules from a handful of examples, which is a key hallmark of human intelligence. The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) provides a rigorous testbed for this capability, demanding conceptual rule induction and transfer to novel tasks. Most existing methods treat ARC-AGI as a purely textual reasoning task, overlooking the fact that humans rely heavily on visual abstraction when solving such puzzles. However, our pilot experiments reveal a paradox: naively rendering ARC-AGI grids as images degrades performance due to imprecise rule execution. This leads to our central hypothesis that vision and language possess complementary strengths across distinct reasoning stages: vision supports global pattern abstraction and verification, whereas language specializes in symbolic rule formulation and precise execution. Building on this insight, we introduce two synergistic strategies: (1) Vision-Language Synergy Reasoning (VLSR), which decomposes ARC-AGI into modality-aligned subtasks; and (2) Modality-Switch Self-Correction (MSSC), which leverages vision to verify text-based reasoning for intrinsic error correction. Extensive experiments demonstrate that our approach yields up to a 4.33% improvement over text-only baselines across diverse flagship models and multiple ARC-AGI tasks. Our findings suggest that unifying visual abstraction with linguistic reasoning is a crucial step toward achieving generalizable, human-like intelligence in future foundation models. Source code will be released soon.
>
---
#### [new 036] Learning to Expand Images for Efficient Visual Autoregressive Modeling
- **分类: cs.CV**

- **简介: 论文提出EAR框架，用于高效视觉自回归建模。针对现有方法效率低的问题，设计螺旋式中心向外生成策略与动态长度解码，提升生成质量和速度，适用于图像生成任务。**

- **链接: [https://arxiv.org/pdf/2511.15499v1](https://arxiv.org/pdf/2511.15499v1)**

> **作者:** Ruiqing Yang; Kaixin Zhang; Zheng Zhang; Shan You; Tao Huang
>
> **备注:** 16 pages, 18 figures, includes appendix with additional visualizations, submitted as arXiv preprint
>
> **摘要:** Autoregressive models have recently shown great promise in visual generation by leveraging discrete token sequences akin to language modeling. However, existing approaches often suffer from inefficiency, either due to token-by-token decoding or the complexity of multi-scale representations. In this work, we introduce Expanding Autoregressive Representation (EAR), a novel generation paradigm that emulates the human visual system's center-outward perception pattern. EAR unfolds image tokens in a spiral order from the center and progressively expands outward, preserving spatial continuity and enabling efficient parallel decoding. To further enhance flexibility and speed, we propose a length-adaptive decoding strategy that dynamically adjusts the number of tokens predicted at each step. This biologically inspired design not only reduces computational cost but also improves generation quality by aligning the generation order with perceptual relevance. Extensive experiments on ImageNet demonstrate that EAR achieves state-of-the-art trade-offs between fidelity and efficiency on single-scale autoregressive models, setting a new direction for scalable and cognitively aligned autoregressive image generation.
>
---
#### [new 037] TiCAL:Typicality-Based Consistency-Aware Learning for Multimodal Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文研究多模态情感识别任务，针对不同模态间情感冲突的问题，提出TiCAL框架。通过伪单模态标签和典型性估计动态评估样本一致性，并在双曲空间中增强情感表示，提升模型在不一致样本上的识别准确率。**

- **链接: [https://arxiv.org/pdf/2511.15085v1](https://arxiv.org/pdf/2511.15085v1)**

> **作者:** Wen Yin; Siyu Zhan; Cencen Liu; Xin Hu; Guiduo Duan; Xiurui Xie; Yuan-Fang Li; Tao He
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Multimodal Emotion Recognition (MER) aims to accurately identify human emotional states by integrating heterogeneous modalities such as visual, auditory, and textual data. Existing approaches predominantly rely on unified emotion labels to supervise model training, often overlooking a critical challenge: inter-modal emotion conflicts, wherein different modalities within the same sample may express divergent emotional tendencies. In this work, we address this overlooked issue by proposing a novel framework, Typicality-based Consistent-aware Multimodal Emotion Recognition (TiCAL), inspired by the stage-wise nature of human emotion perception. TiCAL dynamically assesses the consistency of each training sample by leveraging pseudo unimodal emotion labels alongside a typicality estimation. To further enhance emotion representation, we embed features in a hyperbolic space, enabling the capture of fine-grained distinctions among emotional categories. By incorporating consistency estimates into the learning process, our method improves model performance, particularly on samples exhibiting high modality inconsistency. Extensive experiments on benchmark datasets, e.g, CMU-MOSEI and MER2023, validate the effectiveness of TiCAL in mitigating inter-modal emotional conflicts and enhancing overall recognition accuracy, e.g., with about 2.6% improvements over the state-of-the-art DMD.
>
---
#### [new 038] MMCM: Multimodality-aware Metric using Clustering-based Modes for Probabilistic Human Motion Prediction
- **分类: cs.CV**

- **简介: 该论文针对人类运动预测中的多模态评估问题，提出MMCM指标。它通过聚类定义运动模式，分别评估预测结果的覆盖性（分布于多模式）与有效性（符合真实运动规律），从而更准确衡量概率性预测质量。**

- **链接: [https://arxiv.org/pdf/2511.15179v1](https://arxiv.org/pdf/2511.15179v1)**

> **作者:** Kyotaro Tokoro; Hiromu Taketsugu; Norimichi Ukita
>
> **备注:** Accepted to WACV2026
>
> **摘要:** This paper proposes a novel metric for Human Motion Prediction (HMP). Since a single past sequence can lead to multiple possible futures, a probabilistic HMP method predicts such multiple motions. While a single motion predicted by a deterministic method is evaluated only with the difference from its ground truth motion, multiple predicted motions should also be evaluated based on their distribution. For this evaluation, this paper focuses on the following two criteria. \textbf{(a) Coverage}: motions should be distributed among multiple motion modes to cover diverse possibilities. \textbf{(b) Validity}: motions should be kinematically valid as future motions observable from a given past motion. However, existing metrics simply appreciate widely distributed motions even if these motions are observed in a single mode and kinematically invalid. To resolve these disadvantages, this paper proposes a Multimodality-aware Metric using Clustering-based Modes (MMCM). For (a) coverage, MMCM divides a motion space into several clusters, each of which is regarded as a mode. These modes are used to explicitly evaluate whether predicted motions are distributed among multiple modes. For (b) validity, MMCM identifies valid modes by collecting possible future motions from a motion dataset. Our experiments validate that our clustering yields sensible mode definitions and that MMCM accurately scores multimodal predictions. Code: https://github.com/placerkyo/MMCM
>
---
#### [new 039] Learning Depth from Past Selves: Self-Evolution Contrast for Robust Depth Estimation
- **分类: cs.CV; cs.AI**

- **简介: 论文提出SEC-Depth框架，用于自监督深度估计任务，解决恶劣天气下模型性能下降问题。通过构建时序演化延迟模型并设计自进化对比损失，提升模型鲁棒性，无需人工干预即可适应天气变化。**

- **链接: [https://arxiv.org/pdf/2511.15167v1](https://arxiv.org/pdf/2511.15167v1)**

> **作者:** Jing Cao; Kui Jiang; Shenyi Li; Xiaocheng Feng; Yong Huang
>
> **摘要:** Self-supervised depth estimation has gained significant attention in autonomous driving and robotics. However, existing methods exhibit substantial performance degradation under adverse weather conditions such as rain and fog, where reduced visibility critically impairs depth prediction. To address this issue, we propose a novel self-evolution contrastive learning framework called SEC-Depth for self-supervised robust depth estimation tasks. Our approach leverages intermediate parameters generated during training to construct temporally evolving latency models. Using these, we design a self-evolution contrastive scheme to mitigate performance loss under challenging conditions. Concretely, we first design a dynamic update strategy of latency models for the depth estimation task to capture optimization states across training stages. To effectively leverage latency models, we introduce a self-evolution contrastive Loss (SECL) that treats outputs from historical latency models as negative samples. This mechanism adaptively adjusts learning objectives while implicitly sensing weather degradation severity, reducing the needs for manual intervention. Experiments show that our method integrates seamlessly into diverse baseline models and significantly enhances robustness in zero-shot evaluations.
>
---
#### [new 040] Physics-Based Benchmarking Metrics for Multimodal Synthetic Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多模态合成图像评估中现有指标无法准确捕捉语义和结构信息的问题，提出PCMDE方法。通过融合视觉语言模型与大语言模型的物理约束推理，实现更精准的图像内容验证。任务为多模态数据评估，解决领域特定场景下的准确性不足问题。**

- **链接: [https://arxiv.org/pdf/2511.15204v1](https://arxiv.org/pdf/2511.15204v1)**

> **作者:** Kishor Datta Gupta; Marufa Kamal; Md. Mahfuzur Rahman; Fahad Rahman; Mohd Ariful Haque; Sunzida Siddique
>
> **摘要:** Current state of the art measures like BLEU, CIDEr, VQA score, SigLIP-2 and CLIPScore are often unable to capture semantic or structural accuracy, especially for domain-specific or context-dependent scenarios. For this, this paper proposes a Physics-Constrained Multimodal Data Evaluation (PCMDE) metric combining large language models with reasoning, knowledge based mapping and vision-language models to overcome these limitations. The architecture is comprised of three main stages: (1) feature extraction of spatial and semantic information with multimodal features through object detection and VLMs; (2) Confidence-Weighted Component Fusion for adaptive component-level validation; and (3) physics-guided reasoning using large language models for structural and relational constraints (e.g., alignment, position, consistency) enforcement.
>
---
#### [new 041] SceneEdited: A City-Scale Benchmark for 3D HD Map Updating via Image-Guided Change Detection
- **分类: cs.CV**

- **简介: 论文提出SceneEdited数据集，用于3D高清地图更新任务，解决2D图像变化检测与3D地图更新之间的鸿沟问题。通过城市级点云和图像数据，支持基于图像的变化检测与地图更新研究。**

- **链接: [https://arxiv.org/pdf/2511.15153v1](https://arxiv.org/pdf/2511.15153v1)**

> **作者:** Chun-Jung Lin; Tat-Jun Chin; Sourav Garg; Feras Dayoub
>
> **备注:** accepted by WACV 2026
>
> **摘要:** Accurate, up-to-date High-Definition (HD) maps are critical for urban planning, infrastructure monitoring, and autonomous navigation. However, these maps quickly become outdated as environments evolve, creating a need for robust methods that not only detect changes but also incorporate them into updated 3D representations. While change detection techniques have advanced significantly, there remains a clear gap between detecting changes and actually updating 3D maps, particularly when relying on 2D image-based change detection. To address this gap, we introduce SceneEdited, the first city-scale dataset explicitly designed to support research on HD map maintenance through 3D point cloud updating. SceneEdited contains over 800 up-to-date scenes covering 73 km of driving and approximate 3 $\text{km}^2$ of urban area, with more than 23,000 synthesized object changes created both manually and automatically across 2000+ out-of-date versions, simulating realistic urban modifications such as missing roadside infrastructure, buildings, overpasses, and utility poles. Each scene includes calibrated RGB images, LiDAR scans, and detailed change masks for training and evaluation. We also provide baseline methods using a foundational image-based structure-from-motion pipeline for updating outdated scenes, as well as a comprehensive toolkit supporting scalability, trackability, and portability for future dataset expansion and unification of out-of-date object annotations. Both the dataset and the toolkit are publicly available at https://github.com/ChadLin9596/ScenePoint-ETK, establising a standardized benchmark for 3D map updating research.
>
---
#### [new 042] From Low-Rank Features to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers
- **分类: cs.CV**

- **简介: 论文研究视觉Transformer中特征蒸馏失效问题，揭示其因编码不匹配导致。通过低秩分析与令牌级能量分布研究，提出两种轻量级修复策略，显著提升学生模型性能，为紧凑ViT设计提供指导。**

- **链接: [https://arxiv.org/pdf/2511.15572v1](https://arxiv.org/pdf/2511.15572v1)**

> **作者:** Huiyuan Tian; Bonan Xu; Shijian Li; Xin Jin
>
> **摘要:** Feature-map knowledge distillation (KD) is highly effective for convolutional networks but often fails for Vision Transformers (ViTs). To understand this failure and guide method design, we conduct a two-view representation analysis of ViTs. First, a layer-wise Singular Value Decomposition (SVD) of full feature matrices shows that final-layer representations are globally low-rank: for CaiT-S24, only $121/61/34/14$ dimensions suffice to capture $99\%/95\%/90\%/80\%$ of the energy. In principle, this suggests that a compact student plus a simple linear projector should be enough for feature alignment, contradicting the weak empirical performance of standard feature KD. To resolve this paradox, we introduce a token-level Spectral Energy Pattern (SEP) analysis that measures how each token uses channel capacity. SEP reveals that, despite the global low-rank structure, individual tokens distribute energy over most channels, forming a high-bandwidth encoding pattern. This results in an encoding mismatch between wide teachers and narrow students. Motivated by this insight, we propose two minimal, mismatch-driven strategies: (1) post-hoc feature lifting with a lightweight projector retained during inference, or (2) native width alignment that widens only the student's last block to the teacher's width. On ImageNet-1K, these strategies reactivate simple feature-map distillation in ViTs, raising DeiT-Tiny accuracy from $74.86\%$ to $77.53\%$ and $78.23\%$ when distilling from CaiT-S24, while also improving standalone students trained without any teacher. Our analysis thus explains why ViT feature distillation fails and shows how exploiting low-rank structure yields effective, interpretable remedies and concrete design guidance for compact ViTs.
>
---
#### [new 043] FunnyNodules: A Customizable Medical Dataset Tailored for Evaluating Explainable AI
- **分类: cs.CV**

- **简介: 论文提出FunnyNodules，一个可定制的合成医学图像数据集，用于评估AI模型的可解释性。它通过控制结节形状属性来模拟诊断决策，解决缺乏带推理标注的医疗数据问题，支持模型对属性关联、注意力对齐等分析。**

- **链接: [https://arxiv.org/pdf/2511.15481v1](https://arxiv.org/pdf/2511.15481v1)**

> **作者:** Luisa Gallée; Yiheng Xiong; Meinrad Beer; Michael Götz
>
> **摘要:** Densely annotated medical image datasets that capture not only diagnostic labels but also the underlying reasoning behind these diagnoses are scarce. Such reasoning-related annotations are essential for developing and evaluating explainable AI (xAI) models that reason similarly to radiologists: making correct predictions for the right reasons. To address this gap, we introduce FunnyNodules, a fully parameterized synthetic dataset designed for systematic analysis of attribute-based reasoning in medical AI models. The dataset generates abstract, lung nodule-like shapes with controllable visual attributes such as roundness, margin sharpness, and spiculation. Target class is derived from a predefined attribute combination, allowing full control over the decision rule that links attributes to the diagnostic class. We demonstrate how FunnyNodules can be used in model-agnostic evaluations to assess whether models learn correct attribute-target relations, to interpret over- or underperformance in attribute prediction, and to analyze attention alignment with attribute-specific regions of interest. The framework is fully customizable, supporting variations in dataset complexity, target definitions, class balance, and beyond. With complete ground truth information, FunnyNodules provides a versatile foundation for developing, benchmarking, and conducting in-depth analyses of explainable AI methods in medical image analysis.
>
---
#### [new 044] CellGenNet: A Knowledge-Distilled Framework for Robust Cell Segmentation in Cancer Tissues
- **分类: cs.CV**

- **简介: 该论文提出CellGenNet，一种基于知识蒸馏的细胞分割框架，用于解决癌症组织切片中核分割的挑战。通过师生架构和混合损失函数，在少量标注数据下提升分割准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.15054v1](https://arxiv.org/pdf/2511.15054v1)**

> **作者:** Srijan Ray; Bikesh K. Nirala; Jason T. Yustein; Sundaresh Ram
>
> **备注:** 4 pages, 3 figures, Submitted to IEEE SSIAI 2026
>
> **摘要:** Accurate nuclei segmentation in microscopy whole slide images (WSIs) remains challenging due to variability in staining, imaging conditions, and tissue morphology. We propose CellGenNet, a knowledge distillation framework for robust cross-tissue cell segmentation under limited supervision. CellGenNet adopts a student-teacher architecture, where a capacity teacher is trained on sparse annotations and generates soft pseudo-labels for unlabeled regions. The student is optimized using a joint objective that integrates ground-truth labels, teacher-derived probabilistic targets, and a hybrid loss function combining binary cross-entropy and Tversky loss, enabling asymmetric penalties to mitigate class imbalance and better preserve minority nuclear structures. Consistency regularization and layerwise dropout further stabilize feature representations and promote reliable feature transfer. Experiments across diverse cancer tissue WSIs show that CellGenNet improves segmentation accuracy and generalization over supervised and semi-supervised baselines, supporting scalable and reproducible histopathology analysis.
>
---
#### [new 045] Hyperspectral Super-Resolution with Inter-Image Variability via Degradation-based Low-Rank and Residual Fusion Method
- **分类: cs.CV**

- **简介: 该论文属于遥感图像融合任务，旨在解决因成像条件不同导致的高光谱与多光谱图像间光谱和空间差异问题。提出DLRRF模型，通过降解建模和低秩残差分解提升融合精度，并用PnP框架优化求解。**

- **链接: [https://arxiv.org/pdf/2511.15052v1](https://arxiv.org/pdf/2511.15052v1)**

> **作者:** Yue Wen; Kunjing Yang; Minru Bai
>
> **摘要:** The fusion of hyperspectral image (HSI) with multispectral image (MSI) provides an effective way to enhance the spatial resolution of HSI. However, due to different acquisition conditions, there may exist spectral variability and spatially localized changes between HSI and MSI, referred to as inter-image variability, which can significantly affect the fusion performance. Existing methods typically handle inter-image variability by applying direct transformations to the images themselves, which can exacerbate the ill-posedness of the fusion model. To address this challenge, we propose a Degradation-based Low-Rank and Residual Fusion (DLRRF) model. First, we model the spectral variability as change in the spectral degradation operator. Second, to recover the lost spatial details caused by spatially localized changes, we decompose the target HSI into low rank and residual components, where the latter is used to capture the lost details. By exploiting the spectral correlation within the images, we perform dimensionality reduction on both components. Additionally, we introduce an implicit regularizer to utilize the spatial prior information from the images. The proposed DLRRF model is solved using the Proximal Alternating Optimization (PAO) algorithm within a Plug-and-Play (PnP) framework, where the subproblem regarding implicit regularizer is addressed by an external denoiser. We further provide a comprehensive convergence analysis of the algorithm. Finally, extensive numerical experiments demonstrate that DLRRF achieves superior performance in fusing HSI and MSI with inter-image variability.
>
---
#### [new 046] Evaluating Multimodal Large Language Models on Vertically Written Japanese Text
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态大模型在竖排日文文本上的阅读能力。针对现有模型在竖排日文上表现不佳的问题，作者构建了合成与真实场景的评测数据集，并通过训练提升模型性能。任务为文档图像理解中的文本识别。**

- **链接: [https://arxiv.org/pdf/2511.15059v1](https://arxiv.org/pdf/2511.15059v1)**

> **作者:** Keito Sasagawa; Shuhei Kurita; Daisuke Kawahara
>
> **备注:** 17pages, 8 figures
>
> **摘要:** Multimodal Large Language Models (MLLMs) have seen rapid advances in recent years and are now being applied to visual document understanding tasks. They are expected to process a wide range of document images across languages, including Japanese. Understanding documents from images requires models to read what are written in them. Since some Japanese documents are written vertically, support for vertical writing is essential. However, research specifically focused on vertically written Japanese text remains limited. In this study, we evaluate the reading capability of existing MLLMs on vertically written Japanese text. First, we generate a synthetic Japanese OCR dataset by rendering Japanese texts into images, and use it for both model fine-tuning and evaluation. This dataset includes Japanese text in both horizontal and vertical writing. We also create an evaluation dataset sourced from the real-world document images containing vertically written Japanese text. Using these datasets, we demonstrate that the existing MLLMs perform worse on vertically written Japanese text than on horizontally written Japanese text. Furthermore, we show that training MLLMs on our synthesized Japanese OCR dataset results in improving the performance of models that previously could not handle vertical writing. The datasets and code are publicly available https://github.com/llm-jp/eval_vertical_ja.
>
---
#### [new 047] SIGMMA: Hierarchical Graph-Based Multi-Scale Multi-modal Contrastive Alignment of Histopathology Image and Spatial Transcriptome
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出Sigmma框架，解决病理图像与空间转录组数据在多尺度下对齐不足的问题。通过层次化图结构建模细胞交互，实现跨模态对比对齐，提升基因表达预测与跨模态检索性能。**

- **链接: [https://arxiv.org/pdf/2511.15464v1](https://arxiv.org/pdf/2511.15464v1)**

> **作者:** Dabin Jeong; Amirhossein Vahidi; Ciro Ramírez-Suástegui; Marie Moullet; Kevin Ly; Mohammad Vali Sanian; Sebastian Birk; Yinshui Chang; Adam Boxall; Daniyal Jafree; Lloyd Steele; Vijaya Baskar MS; Muzlifah Haniffa; Mohammad Lotfollahi
>
> **摘要:** Recent advances in computational pathology have leveraged vision-language models to learn joint representations of Hematoxylin and Eosin (HE) images with spatial transcriptomic (ST) profiles. However, existing approaches typically align HE tiles with their corresponding ST profiles at a single scale, overlooking fine-grained cellular structures and their spatial organization. To address this, we propose Sigmma, a multi-modal contrastive alignment framework for learning hierarchical representations of HE images and spatial transcriptome profiles across multiple scales. Sigmma introduces multi-scale contrastive alignment, ensuring that representations learned at different scales remain coherent across modalities. Furthermore, by representing cell interactions as a graph and integrating inter- and intra-subgraph relationships, our approach effectively captures cell-cell interactions, ranging from fine to coarse, within the tissue microenvironment. We demonstrate that Sigmm learns representations that better capture cross-modal correspondences, leading to an improvement of avg. 9.78\% in the gene-expression prediction task and avg. 26.93\% in the cross-modal retrieval task across datasets. We further show that it learns meaningful multi-tissue organization in downstream analyses.
>
---
#### [new 048] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 论文提出Gaussian Blending，改进3D高斯溅射中的alpha混合方式，解决缩放时出现的模糊和阶梯状伪影问题，提升新视角合成质量，且保持实时渲染速度。**

- **链接: [https://arxiv.org/pdf/2511.15102v1](https://arxiv.org/pdf/2511.15102v1)**

> **作者:** Junseo Koo; Jinseo Jeong; Gunhee Kim
>
> **备注:** AAAI 2026
>
> **摘要:** The recent introduction of 3D Gaussian Splatting (3DGS) has significantly advanced novel view synthesis. Several studies have further improved the rendering quality of 3DGS, yet they still exhibit noticeable visual discrepancies when synthesizing views at sampling rates unseen during training. Specifically, they suffer from (i) erosion-induced blurring artifacts when zooming in and (ii) dilation-induced staircase artifacts when zooming out. We speculate that these artifacts arise from the fundamental limitation of the alpha blending adopted in 3DGS methods. Instead of the conventional alpha blending that computes alpha and transmittance as scalar quantities over a pixel, we propose to replace it with our novel Gaussian Blending that treats alpha and transmittance as spatially varying distributions. Thus, transmittances can be updated considering the spatial distribution of alpha values across the pixel area, allowing nearby background splats to contribute to the final rendering. Our Gaussian Blending maintains real-time rendering speed and requires no additional memory cost, while being easily integrated as a drop-in replacement into existing 3DGS-based or other NVS frameworks. Extensive experiments demonstrate that Gaussian Blending effectively captures fine details at various sampling rates unseen during training, consistently outperforming existing novel view synthesis models across both unseen and seen sampling rates.
>
---
#### [new 049] Adaptive thresholding pattern for fingerprint forgery detection
- **分类: cs.CV**

- **简介: 论文提出一种自适应阈值模式用于指纹伪造检测，解决真实与伪造指纹区分难题。通过多层小波变换与各向异性扩散提取特征，结合SVM分类器，在多种失真下表现优越，准确率提升显著。**

- **链接: [https://arxiv.org/pdf/2511.15322v1](https://arxiv.org/pdf/2511.15322v1)**

> **作者:** Zahra Farzadpour; Masoumeh Azghani
>
> **备注:** 25 pages, 10 figures, Journal paper
>
> **摘要:** Fingerprint liveness detection systems have been affected by spoofing, which is a severe threat for fingerprint-based biometric systems. Therefore, it is crucial to develop some techniques to distinguish the fake fingerprints from the real ones. The software based techniques can detect the fingerprint forgery automatically. Also, the scheme shall be resistant against various distortions such as noise contamination, pixel missing and block missing, so that the forgers cannot deceive the detector by adding some distortions to the faked fingerprint. In this paper, we propose a fingerprint forgery detection algorithm based on a suggested adaptive thresholding pattern. The anisotropic diffusion of the input image is passed through three levels of the wavelet transform. The coefficients of different layers are adaptively thresholded and concatenated to produce the feature vector which is classified using the SVM classifier. Another contribution of the paper is to investigate the effect of various distortions such as pixel missing, block missing, and noise contamination. Our suggested approach includes a novel method that exhibits improved resistance against a range of distortions caused by environmental phenomena or manipulations by malicious users. In quantitative comparisons, our proposed method outperforms its counterparts by approximately 8% and 5% in accuracy for missing pixel scenarios of 90% and block missing scenarios of size 70x70 , respectively. This highlights the novelty approach in addressing such challenges.
>
---
#### [new 050] Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance
- **分类: cs.CV**

- **简介: 论文提出动态梯度引导方法，解决多模态大模型持续学习中的灾难性遗忘问题。通过几何梯度近似和伯努利采样策略，在不扩展模型的情况下有效平衡稳定性和可塑性，实现高效多模态任务连续适应。**

- **链接: [https://arxiv.org/pdf/2511.15164v1](https://arxiv.org/pdf/2511.15164v1)**

> **作者:** Songze Li; Mingyu Gao; Tonghua Su; Xu-Yao Zhang; Zhongjie Wang
>
> **摘要:** Multimodal continual instruction tuning enables multimodal large language models to sequentially adapt to new tasks while building upon previously acquired knowledge. However, this continual learning paradigm faces the significant challenge of catastrophic forgetting, where learning new tasks leads to performance degradation on previous ones. In this paper, we introduce a novel insight into catastrophic forgetting by conceptualizing it as a problem of missing gradients from old tasks during new task learning. Our approach approximates these missing gradients by leveraging the geometric properties of the parameter space, specifically using the directional vector between current parameters and previously optimal parameters as gradient guidance. This approximated gradient can be further integrated with real gradients from a limited replay buffer and regulated by a Bernoulli sampling strategy that dynamically balances model stability and plasticity. Extensive experiments on multimodal continual instruction tuning datasets demonstrate that our method achieves state-of-the-art performance without model expansion, effectively mitigating catastrophic forgetting while maintaining a compact architecture.
>
---
#### [new 051] Controlling False Positives in Image Segmentation via Conformal Prediction
- **分类: cs.CV; cs.LG**

- **简介: 论文提出一种基于置信度预测的图像分割方法，用于控制假阳性错误。针对临床场景中过分割的风险，该方法通过校准集选择最优阈值，在不重新训练模型的前提下提供有限样本统计保证，实现风险可控的分割。**

- **链接: [https://arxiv.org/pdf/2511.15406v1](https://arxiv.org/pdf/2511.15406v1)**

> **作者:** Luca Mossina; Corentin Friedrich
>
> **摘要:** Reliable semantic segmentation is essential for clinical decision making, yet deep models rarely provide explicit statistical guarantees on their errors. We introduce a simple post-hoc framework that constructs confidence masks with distribution-free, image-level control of false-positive predictions. Given any pretrained segmentation model, we define a nested family of shrunken masks obtained either by increasing the score threshold or by applying morphological erosion. A labeled calibration set is used to select a single shrink parameter via conformal prediction, ensuring that, for new images that are exchangeable with the calibration data, the proportion of false positives retained in the confidence mask stays below a user-specified tolerance with high probability. The method is model-agnostic, requires no retraining, and provides finite-sample guarantees regardless of the underlying predictor. Experiments on a polyp-segmentation benchmark demonstrate target-level empirical validity. Our framework enables practical, risk-aware segmentation in settings where over-segmentation can have clinical consequences. Code at https://github.com/deel-ai-papers/conseco.
>
---
#### [new 052] Transferable Dual-Domain Feature Importance Attack against AI-Generated Image Detector
- **分类: cs.CV; cs.CR**

- **简介: 论文提出DuFIA攻击方法，针对AI生成图像检测器，通过联合建模空间与频域特征重要性，生成具有高迁移性的对抗样本，提升检测器安全性评估能力。**

- **链接: [https://arxiv.org/pdf/2511.15571v1](https://arxiv.org/pdf/2511.15571v1)**

> **作者:** Weiheng Zhu; Gang Cao; Jing Liu; Lifang Yu; Shaowei Weng
>
> **摘要:** Recent AI-generated image (AIGI) detectors achieve impressive accuracy under clean condition. In view of antiforensics, it is significant to develop advanced adversarial attacks for evaluating the security of such detectors, which remains unexplored sufficiently. This letter proposes a Dual-domain Feature Importance Attack (DuFIA) scheme to invalidate AIGI detectors to some extent. Forensically important features are captured by the spatially interpolated gradient and frequency-aware perturbation. The adversarial transferability is enhanced by jointly modeling spatial and frequency-domain feature importances, which are fused to guide the optimization-based adversarial example generation. Extensive experiments across various AIGI detectors verify the cross-model transferability, transparency and robustness of DuFIA.
>
---
#### [new 053] Representation Space Constrained Learning with Modality Decoupling for Multimodal Object Detection
- **分类: cs.CV**

- **简介: 论文针对多模态目标检测中的融合退化问题，提出RSC-MD方法。通过理论分析发现梯度抑制和模态不平衡是主因，设计约束表示空间与解耦模态模块，提升各模态优化效果，显著改善检测性能。**

- **链接: [https://arxiv.org/pdf/2511.15433v1](https://arxiv.org/pdf/2511.15433v1)**

> **作者:** YiKang Shao; Tao Shi
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Multimodal object detection has attracted significant attention in both academia and industry for its enhanced robustness. Although numerous studies have focused on improving modality fusion strategies, most neglect fusion degradation, and none provide a theoretical analysis of its underlying causes. To fill this gap, this paper presents a systematic theoretical investigation of fusion degradation in multimodal detection and identifies two key optimization deficiencies: (1) the gradients of unimodal branch backbones are severely suppressed under multimodal architectures, resulting in under-optimization of the unimodal branches; (2) disparities in modality quality cause weaker modalities to experience stronger gradient suppression, which in turn results in imbalanced modality learning. To address these issues, this paper proposes a Representation Space Constrained Learning with Modality Decoupling (RSC-MD) method, which consists of two modules. The RSC module and the MD module are designed to respectively amplify the suppressed gradients and eliminate inter-modality coupling interference as well as modality imbalance, thereby enabling the comprehensive optimization of each modality-specific backbone. Extensive experiments conducted on the FLIR, LLVIP, M3FD, and MFAD datasets demonstrate that the proposed method effectively alleviates fusion degradation and achieves state-of-the-art performance across multiple benchmarks. The code and training procedures will be released at https://github.com/yikangshao/RSC-MD.
>
---
#### [new 054] Driving in Spikes: An Entropy-Guided Object Detector for Spike Cameras
- **分类: cs.CV**

- **简介: 论文提出EASD，一种端到端的脉冲相机目标检测方法，解决运动模糊和极端光照下检测难题。通过双分支结构融合全局语义与局部细节，并构建首个面向驾驶场景的脉冲数据集DSEC Spike。**

- **链接: [https://arxiv.org/pdf/2511.15459v1](https://arxiv.org/pdf/2511.15459v1)**

> **作者:** Ziyan Liu; Qi Su; Lulu Tang; Zhaofei Yu; Tiejun Huang
>
> **摘要:** Object detection in autonomous driving suffers from motion blur and saturation under fast motion and extreme lighting. Spike cameras, offer microsecond latency and ultra high dynamic range for object detection by using per pixel asynchronous integrate and fire. However, their sparse, discrete output cannot be processed by standard image-based detectors, posing a critical challenge for end to end spike stream detection. We propose EASD, an end to end spike camera detector with a dual branch design: a Temporal Based Texture plus Feature Fusion branch for global cross slice semantics, and an Entropy Selective Attention branch for object centric details. To close the data gap, we introduce DSEC Spike, the first driving oriented simulated spike detection benchmark.
>
---
#### [new 055] Graph Query Networks for Object Detection with Automotive Radar
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对汽车雷达目标检测任务，解决雷达点云稀疏不规则导致传统方法性能差的问题。提出图查询网络（GQN），通过图结构建模和动态注意力机制，实现关系推理与上下文聚合，显著提升检测精度并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2511.15271v1](https://arxiv.org/pdf/2511.15271v1)**

> **作者:** Loveneet Saini; Hasan Tercan; Tobias Meisen
>
> **备注:** Accepted in WACV 2026 Main Conference
>
> **摘要:** Object detection with 3D radar is essential for 360-degree automotive perception, but radar's long wavelengths produce sparse and irregular reflections that challenge traditional grid and sequence-based convolutional and transformer detectors. This paper introduces Graph Query Networks (GQN), an attention-based framework that models objects sensed by radar as graphs, to extract individualized relational and contextual features. GQN employs a novel concept of graph queries to dynamically attend over the bird's-eye view (BEV) space, constructing object-specific graphs processed by two novel modules: EdgeFocus for relational reasoning and DeepContext Pooling for contextual aggregation. On the NuScenes dataset, GQN improves relative mAP by up to +53%, including a +8.2% gain over the strongest prior radar method, while reducing peak graph construction overhead by 80% with moderate FLOPs cost.
>
---
#### [new 056] GeoSceneGraph: Geometric Scene Graph Diffusion Model for Text-guided 3D Indoor Scene Synthesis
- **分类: cs.CV**

- **简介: 论文提出GeoSceneGraph，用于文本引导的3D室内场景合成。解决现有方法忽视场景图结构或依赖人工标注关系的问题，通过等变图神经网络实现无需预定义关系类的高质量场景生成。**

- **链接: [https://arxiv.org/pdf/2511.14884v1](https://arxiv.org/pdf/2511.14884v1)**

> **作者:** Antonio Ruiz; Tao Wu; Andrew Melnik; Qing Cheng; Xuqin Wang; Lu Liu; Yongliang Wang; Yanfeng Zhang; Helge Ritter
>
> **摘要:** Methods that synthesize indoor 3D scenes from text prompts have wide-ranging applications in film production, interior design, video games, virtual reality, and synthetic data generation for training embodied agents. Existing approaches typically either train generative models from scratch or leverage vision-language models (VLMs). While VLMs achieve strong performance, particularly for complex or open-ended prompts, smaller task-specific models remain necessary for deployment on resource-constrained devices such as extended reality (XR) glasses or mobile phones. However, many generative approaches that train from scratch overlook the inherent graph structure of indoor scenes, which can limit scene coherence and realism. Conversely, methods that incorporate scene graphs either demand a user-provided semantic graph, which is generally inconvenient and restrictive, or rely on ground-truth relationship annotations, limiting their capacity to capture more varied object interactions. To address these challenges, we introduce GeoSceneGraph, a method that synthesizes 3D scenes from text prompts by leveraging the graph structure and geometric symmetries of 3D scenes, without relying on predefined relationship classes. Despite not using ground-truth relationships, GeoSceneGraph achieves performance comparable to methods that do. Our model is built on equivariant graph neural networks (EGNNs), but existing EGNN approaches are typically limited to low-dimensional conditioning and are not designed to handle complex modalities such as text. We propose a simple and effective strategy for conditioning EGNNs on text features, and we validate our design through ablation studies.
>
---
#### [new 057] Logit-Based Losses Limit the Effectiveness of Feature Knowledge Distillation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文研究知识蒸馏任务，针对现有方法依赖logit损失限制效果的问题，提出仅用特征损失训练学生模型的新框架，并引入知识质量指标选择最优教师层。在多个数据集和模型对上实现显著性能提升。**

- **链接: [https://arxiv.org/pdf/2511.14981v1](https://arxiv.org/pdf/2511.14981v1)**

> **作者:** Nicholas Cooper; Lijun Chen; Sailesh Dwivedy; Danna Gurari
>
> **备注:** NeurIPS Workshop on Symmetry and Geometry in Neural Representations (NeurReps), December 2025
>
> **摘要:** Knowledge distillation (KD) methods can transfer knowledge of a parameter-heavy teacher model to a light-weight student model. The status quo for feature KD methods is to utilize loss functions based on logits (i.e., pre-softmax class scores) and intermediate layer features (i.e., latent representations). Unlike previous approaches, we propose a feature KD framework for training the student's backbone using feature-based losses exclusively (i.e., without logit-based losses such as cross entropy). Leveraging recent discoveries about the geometry of latent representations, we introduce a knowledge quality metric for identifying which teacher layers provide the most effective knowledge for distillation. Experiments on three image classification datasets with four diverse student-teacher pairs, spanning convolutional neural networks and vision transformers, demonstrate our KD method achieves state-of-the-art performance, delivering top-1 accuracy boosts of up to 15% over standard approaches. We publically share our code to facilitate future work at https://github.com/Thegolfingocto/KD_wo_CE.
>
---
#### [new 058] SkinGPT-R1: Adapter-Only Dual Distillation for Efficient Dermatology Reasoning
- **分类: cs.CV**

- **简介: 论文提出SkinGPT-R1，一种专注皮肤科的视觉语言模型，通过显式链式思维推理提升诊断可解释性。解决皮肤科诊断中模型推理不透明的问题，构建DermCoT语料库和DermBench基准，实现更高准确率与更高质量的诊断推理。**

- **链接: [https://arxiv.org/pdf/2511.15242v1](https://arxiv.org/pdf/2511.15242v1)**

> **作者:** Yuhao Shen; Jiahe Qian; Zhangtianyi Chen; Yuanhao He; Juexiao Zhou
>
> **摘要:** We present SkinGPT-R1, a dermatology focused vision language model that makes diagnostic chain of thought reasoning explicit, step by step, and verifiable. To support skin specific reasoning, we build DermCoT, a corpus of standardized dermatologic chain of thought narratives that combines 10,000 DermEval filtered training cases with 3,000 dermatologist scored certified cases, and we define DermEval as a physician aligned six dimensional evaluator and DermBench as the corresponding benchmark for dermatologic chain of thought quality. On DermBench, across 14 general, reasoning, and medical vision language models, SkinGPT-R1 achieves an average score of 4.031 out of 5 over the six clinician defined dimensions, ranks 1st among all systems, and improves the average score over Vision-R1 by about 41%. On three dermatology classification benchmarks, SkinGPT-R1 delivers stable accuracy gains over Vision-R1 and remains competitive among strong vision language models. Ablation results further show that DermCoT based chain of thought supervision provides substantial improvements over the base model and that adding dermatology aware visual distillation yields consistent additional gains in both narrative quality and recognition.
>
---
#### [new 059] SplitFlux: Learning to Decouple Content and Style from a Single Image
- **分类: cs.CV**

- **简介: 论文提出SplitFlux，解决单图内容与风格解耦问题。通过分析Flux模型发现早期块控内容、后期块控风格，设计Rank-Constrained Adaptation和Visual-Gated LoRA，实现高质量内容保留与风格迁移。**

- **链接: [https://arxiv.org/pdf/2511.15258v1](https://arxiv.org/pdf/2511.15258v1)**

> **作者:** Yitong Yang; Yinglin Wang; Changshuo Wang; Yongjun Zhang; Ziyang Chen; Shuting He
>
> **摘要:** Disentangling image content and style is essential for customized image generation. Existing SDXL-based methods struggle to achieve high-quality results, while the recently proposed Flux model fails to achieve effective content-style separation due to its underexplored characteristics. To address these challenges, we conduct a systematic analysis of Flux and make two key observations: (1) Single Dream Blocks are essential for image generation; and (2) Early single stream blocks mainly control content, whereas later blocks govern style. Based on these insights, we propose SplitFlux, which disentangles content and style by fine-tuning the single dream blocks via LoRA, enabling the disentangled content to be re-embedded into new contexts. It includes two key components: (1) Rank-Constrained Adaptation. To preserve content identity and structure, we compress the rank and amplify the magnitude of updates within specific blocks, preventing content leakage into style blocks. (2) Visual-Gated LoRA. We split the content LoRA into two branches with different ranks, guided by image saliency. The high-rank branch preserves primary subject information, while the low-rank branch encodes residual details, mitigating content overfitting and enabling seamless re-embedding. Extensive experiments demonstrate that SplitFlux consistently outperforms state-of-the-art methods, achieving superior content preservation and stylization quality across diverse scenarios.
>
---
#### [new 060] IPTQ-ViT: Post-Training Quantization of Non-linear Functions for Integer-only Vision Transformers
- **分类: cs.CV; cs.AI**

- **简介: 论文提出IPTQ-ViT框架，解决视觉Transformer在后训练量化中非线性函数难以全整数实现的问题。通过设计多项式GELU和位移Softmax近似函数，并引入统一指标选择最优近似方案，实现无需重训练的全整数推理，显著提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2511.15369v1](https://arxiv.org/pdf/2511.15369v1)**

> **作者:** Gihwan Kim; Jemin Lee; Hyungshin Kim
>
> **备注:** accepted in WACV 2026 (10 pages)
>
> **摘要:** Previous Quantization-Aware Training (QAT) methods for vision transformers rely on expensive retraining to recover accuracy loss in non-linear layer quantization, limiting their use in resource-constrained environments. In contrast, existing Post-Training Quantization (PTQ) methods either partially quantize non-linear functions or adjust activation distributions to maintain accuracy but fail to achieve fully integer-only inference. In this paper, we introduce IPTQ-ViT, a novel PTQ framework for fully integer-only vision transformers without retraining. We present approximation functions: a polynomial-based GELU optimized for vision data and a bit-shifting-based Softmax designed to improve approximation accuracy in PTQ. In addition, we propose a unified metric integrating quantization sensitivity, perturbation, and computational cost to select the optimal approximation function per activation layer. IPTQ-ViT outperforms previous PTQ methods, achieving up to 6.44\%p (avg. 1.78\%p) top-1 accuracy improvement for image classification, 1.0 mAP for object detection. IPTQ-ViT outperforms partial floating-point PTQ methods under W8A8 and W4A8, and achieves accuracy and latency comparable to integer-only QAT methods. We plan to release our code https://github.com/gihwan-kim/IPTQ-ViT.git.
>
---
#### [new 061] Skin-R1: Toward Trustworthy Clinical Reasoning for Dermatological Diagnosis
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出SkinR1，一个用于皮肤科诊断的视觉语言模型，解决数据异质性、缺乏可解释推理和泛化能力弱的问题。通过教科书式推理生成器和强化学习框架，提升诊断准确性和可信度。**

- **链接: [https://arxiv.org/pdf/2511.14900v1](https://arxiv.org/pdf/2511.14900v1)**

> **作者:** Zehao Liu; Wejieying Ren; Jipeng Zhang; Tianxiang Zhao; Jingxi Zhu; Xiaoting Li; Vasant G. Honavar
>
> **摘要:** The emergence of vision-language models (VLMs) has opened new possibilities for clinical reasoning and has shown promising performance in dermatological diagnosis. However, their trustworthiness and clinical utility are often limited by three major factors: (1) Data heterogeneity, where diverse datasets lack consistent diagnostic labels and clinical concept annotations; (2) Absence of grounded diagnostic rationales, leading to a scarcity of reliable reasoning supervision; and (3) Limited scalability and generalization, as models trained on small, densely annotated datasets struggle to transfer nuanced reasoning to large, sparsely-annotated ones. To address these limitations, we propose SkinR1, a novel dermatological VLM that combines deep, textbook-based reasoning with the broad generalization capabilities of reinforcement learning (RL). SkinR1 systematically resolves the key challenges through a unified, end-to-end framework. First, we design a textbook-based reasoning generator that synthesizes high-fidelity, hierarchy-aware, and differential-diagnosis (DDx)-informed trajectories, providing reliable expert-level supervision. Second, we leverage the constructed trajectories for supervised fine-tuning (SFT) empowering the model with grounded reasoning ability. Third, we develop a novel RL paradigm that, by incorporating the hierarchical structure of diseases, effectively transfers these grounded reasoning patterns to large-scale, sparse data. Extensive experiments on multiple dermatology datasets demonstrate that SkinR1 achieves superior diagnostic accuracy. The ablation study demonstrates the importance of the reasoning foundation instilled by SFT.
>
---
#### [new 062] Unsupervised Discovery of Long-Term Spatiotemporal Periodic Workflows in Human Activities
- **分类: cs.CV**

- **简介: 论文提出了一种无监督方法，用于发现人类活动中长期周期性工作流。针对现有研究多关注短周期、高对比度活动的问题，作者构建了首个包含580个多元序列的基准数据集，并设计轻量级无需训练的基线模型，在检测、追踪和异常识别任务中表现优异，无需标注即可达到传统监督方法效果。**

- **链接: [https://arxiv.org/pdf/2511.14945v1](https://arxiv.org/pdf/2511.14945v1)**

> **作者:** Fan Yang; Quanting Xie; Atsunori Moteki; Shoichi Masui; Shan Jiang; Yonatan Bisk; Graham Neubig
>
> **备注:** accepted to WACV 2026
>
> **摘要:** Periodic human activities with implicit workflows are common in manufacturing, sports, and daily life. While short-term periodic activities -- characterized by simple structures and high-contrast patterns -- have been widely studied, long-term periodic workflows with low-contrast patterns remain largely underexplored. To bridge this gap, we introduce the first benchmark comprising 580 multimodal human activity sequences featuring long-term periodic workflows. The benchmark supports three evaluation tasks aligned with real-world applications: unsupervised periodic workflow detection, task completion tracking, and procedural anomaly detection. We also propose a lightweight, training-free baseline for modeling diverse periodic workflow patterns. Experiments show that: (i) our benchmark presents significant challenges to both unsupervised periodic detection methods and zero-shot approaches based on powerful large language models (LLMs); (ii) our baseline outperforms competing methods by a substantial margin in all evaluation tasks; and (iii) in real-world applications, our baseline demonstrates deployment advantages on par with traditional supervised workflow detection approaches, eliminating the need for annotation and retraining. Our project page is https://sites.google.com/view/periodicworkflow.
>
---
#### [new 063] When to Think and When to Look: Uncertainty-Guided Lookback
- **分类: cs.CV; cs.CL**

- **简介: 论文研究视觉语言模型中测试时思考（thinking）的策略，解决“盲目增加思考步骤未必提升性能”的问题。通过分析发现短回看短语能增强视觉 grounding，提出无需训练的不确定性引导回看策略，显著提升多任务表现，包括 MMMU 和其他五个基准。**

- **链接: [https://arxiv.org/pdf/2511.15613v1](https://arxiv.org/pdf/2511.15613v1)**

> **作者:** Jing Bi; Filippos Bellos; Junjia Guo; Yayuan Li; Chao Huang; Yunlong; Tang; Luchuan Song; Susan Liang; Zhongfei; Zhang; Jason J. Corso; Chenliang Xu
>
> **摘要:** Test-time thinking (that is, generating explicit intermediate reasoning chains) is known to boost performance in large language models and has recently shown strong gains for large vision language models (LVLMs). However, despite these promising results, there is still no systematic analysis of how thinking actually affects visual reasoning. We provide the first such analysis with a large scale, controlled comparison of thinking for LVLMs, evaluating ten variants from the InternVL3.5 and Qwen3-VL families on MMMU-val under generous token budgets and multi pass decoding. We show that more thinking is not always better; long chains often yield long wrong trajectories that ignore the image and underperform the same models run in standard instruct mode. A deeper analysis reveals that certain short lookback phrases, which explicitly refer back to the image, are strongly enriched in successful trajectories and correlate with better visual grounding. Building on this insight, we propose uncertainty guided lookback, a training free decoding strategy that combines an uncertainty signal with adaptive lookback prompts and breadth search. Our method improves overall MMMU performance, delivers the largest gains in categories where standard thinking is weak, and outperforms several strong decoding baselines, setting a new state of the art under fixed model families and token budgets. We further show that this decoding strategy generalizes, yielding consistent improvements on five additional benchmarks, including two broad multimodal suites and math focused visual reasoning datasets.
>
---
#### [new 064] Computer Vision Modeling of the Development of Geometric and Numerical Concepts in Humans
- **分类: cs.CV**

- **简介: 该论文研究计算机视觉模型在几何和数字概念发展上的发育一致性。通过分析ResNet-50模型，发现其对部分几何与数字概念的学习轨迹与儿童发展相似，揭示CV模型在理解人类数学认知发展中的潜力。**

- **链接: [https://arxiv.org/pdf/2511.15029v1](https://arxiv.org/pdf/2511.15029v1)**

> **作者:** Zekun Wang; Sashank Varma
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Mathematical thinking is a fundamental aspect of human cognition. Cognitive scientists have investigated the mechanisms that underlie our ability to thinking geometrically and numerically, to take two prominent examples, and developmental scientists have documented the trajectories of these abilities over the lifespan. Prior research has shown that computer vision (CV) models trained on the unrelated task of image classification nevertheless learn latent representations of geometric and numerical concepts similar to those of adults. Building on this demonstrated cognitive alignment, the current study investigates whether CV models also show developmental alignment: whether their performance improvements across training to match the developmental progressions observed in children. In a detailed case study of the ResNet-50 model, we show that this is the case. For the case of geometry and topology, we find developmental alignment for some classes of concepts (Euclidean Geometry, Geometrical Figures, Metric Properties, Topology) but not others (Chiral Figures, Geometric Transformations, Symmetrical Figures). For the case of number, we find developmental alignment in the emergence of a human-like ``mental number line'' representation with experience. These findings show the promise of computer vision models for understanding the development of mathematical understanding in humans. They point the way to future research exploring additional model architectures and building larger benchmarks.
>
---
#### [new 065] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling
- **分类: cs.CV**

- **简介: 论文提出ProPL框架，解决超声图像分割任务中模型泛化能力弱的问题。通过提示引导的伪标签机制和不确定性校准模块，实现多器官、多任务的半监督分割，提升临床实用性。**

- **链接: [https://arxiv.org/pdf/2511.15057v1](https://arxiv.org/pdf/2511.15057v1)**

> **作者:** Yaxiong Chen; Qicong Wang; Chunlei Li; Jingliang Hu; Yilei Shi; Shengwu Xiong; Xiao Xiang Zhu; Lichao Mou
>
> **备注:** AAAI 2026
>
> **摘要:** Existing approaches for the problem of ultrasound image segmentation, whether supervised or semi-supervised, are typically specialized for specific anatomical structures or tasks, limiting their practical utility in clinical settings. In this paper, we pioneer the task of universal semi-supervised ultrasound image segmentation and propose ProPL, a framework that can handle multiple organs and segmentation tasks while leveraging both labeled and unlabeled data. At its core, ProPL employs a shared vision encoder coupled with prompt-guided dual decoders, enabling flexible task adaptation through a prompting-upon-decoding mechanism and reliable self-training via an uncertainty-driven pseudo-label calibration (UPLC) module. To facilitate research in this direction, we introduce a comprehensive ultrasound dataset spanning 5 organs and 8 segmentation tasks. Extensive experiments demonstrate that ProPL outperforms state-of-the-art methods across various metrics, establishing a new benchmark for universal ultrasound image segmentation.
>
---
#### [new 066] US-X Complete: A Multi-Modal Approach to Anatomical 3D Shape Recovery
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像处理任务，旨在解决超声成像因骨性阴影导致椎体结构不完整的问题。通过融合单张X光片与3D超声数据，提出多模态深度学习方法实现椎体形状补全，提升重建精度，无需术前CT配准。**

- **链接: [https://arxiv.org/pdf/2511.15600v1](https://arxiv.org/pdf/2511.15600v1)**

> **作者:** Miruna-Alexandra Gafencu; Yordanka Velikova; Nassir Navab; Mohammad Farid Azampour
>
> **备注:** Accepted at the Workshop on Shape in Medical Imaging at MICCAI 2025
>
> **摘要:** Ultrasound offers a radiation-free, cost-effective solution for real-time visualization of spinal landmarks, paraspinal soft tissues and neurovascular structures, making it valuable for intraoperative guidance during spinal procedures. However, ultrasound suffers from inherent limitations in visualizing complete vertebral anatomy, in particular vertebral bodies, due to acoustic shadowing effects caused by bone. In this work, we present a novel multi-modal deep learning method for completing occluded anatomical structures in 3D ultrasound by leveraging complementary information from a single X-ray image. To enable training, we generate paired training data consisting of: (1) 2D lateral vertebral views that simulate X-ray scans, and (2) 3D partial vertebrae representations that mimic the limited visibility and occlusions encountered during ultrasound spine imaging. Our method integrates morphological information from both imaging modalities and demonstrates significant improvements in vertebral reconstruction (p < 0.001) compared to state of art in 3D ultrasound vertebral completion. We perform phantom studies as an initial step to future clinical translation, and achieve a more accurate, complete volumetric lumbar spine visualization overlayed on the ultrasound scan without the need for registration with preoperative modalities such as computed tomography. This demonstrates that integrating a single X-ray projection mitigates ultrasound's key limitation while preserving its strengths as the primary imaging modality. Code and data can be found at https://github.com/miruna20/US-X-Complete
>
---
#### [new 067] What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs
- **分类: cs.CV**

- **简介: 论文提出FIA-Flow框架，针对Split DNNs中中间特征泄露隐私的问题，通过语义对齐和分布校正实现高保真图像重建，揭示更严重的隐私风险。**

- **链接: [https://arxiv.org/pdf/2511.15316v1](https://arxiv.org/pdf/2511.15316v1)**

> **作者:** Zhihan Ren; Lijun He; Jiaxi Liang; Xinzhu Fu; Haixia Bi; Fan Li
>
> **摘要:** Split DNNs enable edge devices by offloading intensive computation to a cloud server, but this paradigm exposes privacy vulnerabilities, as the intermediate features can be exploited to reconstruct the private inputs via Feature Inversion Attack (FIA). Existing FIA methods often produce limited reconstruction quality, making it difficult to assess the true extent of privacy leakage. To reveal the privacy risk of the leaked features, we introduce FIA-Flow, a black-box FIA framework that achieves high-fidelity image reconstruction from intermediate features. To exploit the semantic information within intermediate features, we design a Latent Feature Space Alignment Module (LFSAM) to bridge the semantic gap between the intermediate feature space and the latent space. Furthermore, to rectify distributional mismatch, we develop Deterministic Inversion Flow Matching (DIFM), which projects off-manifold features onto the target manifold with one-step inference. This decoupled design simplifies learning and enables effective training with few image-feature pairs. To quantify privacy leakage from a human perspective, we also propose two metrics based on a large vision-language model. Experiments show that FIA-Flow achieves more faithful and semantically aligned feature inversion across various models (AlexNet, ResNet, Swin Transformer, DINO, and YOLO11) and layers, revealing a more severe privacy threat in Split DNNs than previously recognized.
>
---
#### [new 068] Unbiased Semantic Decoding with Vision Foundation Models for Few-shot Segmentation
- **分类: cs.CV**

- **简介: 论文针对少样本图像分割任务，解决现有方法依赖显式提示导致的解码偏差问题。提出无偏语义解码策略（USD），结合SAM与CLIP模型，通过支持集和查询集联合提取目标信息，增强语义判别力，并生成聚焦目标的提示嵌入，提升对未知类别的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.15118v1](https://arxiv.org/pdf/2511.15118v1)**

> **作者:** Jin Wang; Bingfeng Zhang; Jian Pang; Weifeng Liu; Baodi Liu; Honglong Chen
>
> **摘要:** Few-shot segmentation has garnered significant attention. Many recent approaches attempt to introduce the Segment Anything Model (SAM) to handle this task. With the strong generalization ability and rich object-specific extraction ability of the SAM model, such a solution shows great potential in few-shot segmentation. However, the decoding process of SAM highly relies on accurate and explicit prompts, making previous approaches mainly focus on extracting prompts from the support set, which is insufficient to activate the generalization ability of SAM, and this design is easy to result in a biased decoding process when adapting to the unknown classes. In this work, we propose an Unbiased Semantic Decoding (USD) strategy integrated with SAM, which extracts target information from both the support and query set simultaneously to perform consistent predictions guided by the semantics of the Contrastive Language-Image Pre-training (CLIP) model. Specifically, to enhance the unbiased semantic discrimination of SAM, we design two feature enhancement strategies that leverage the semantic alignment capability of CLIP to enrich the original SAM features, mainly including a global supplement at the image level to provide a generalize category indicate with support image and a local guidance at the pixel level to provide a useful target location with query image. Besides, to generate target-focused prompt embeddings, a learnable visual-text target prompt generator is proposed by interacting target text embeddings and clip visual features. Without requiring re-training of the vision foundation models, the features with semantic discrimination draw attention to the target region through the guidance of prompt with rich target information.
>
---
#### [new 069] Gaussian See, Gaussian Do: Semantic 3D Motion Transfer from Multiview Video
- **分类: cs.CV**

- **简介: 该论文提出Gaussian See, Gaussian Do，解决跨类别、无骨骼的语义3D运动迁移问题。通过条件反演提取运动嵌入，结合动态3D高斯泼溅重建，实现跨视角一致且结构稳定的运动迁移。**

- **链接: [https://arxiv.org/pdf/2511.14848v1](https://arxiv.org/pdf/2511.14848v1)**

> **作者:** Yarin Bekor; Gal Michael Harari; Or Perel; Or Litany
>
> **备注:** SIGGRAPH Asia 2025
>
> **摘要:** We present Gaussian See, Gaussian Do, a novel approach for semantic 3D motion transfer from multiview video. Our method enables rig-free, cross-category motion transfer between objects with semantically meaningful correspondence. Building on implicit motion transfer techniques, we extract motion embeddings from source videos via condition inversion, apply them to rendered frames of static target shapes, and use the resulting videos to supervise dynamic 3D Gaussian Splatting reconstruction. Our approach introduces an anchor-based view-aware motion embedding mechanism, ensuring cross-view consistency and accelerating convergence, along with a robust 4D reconstruction pipeline that consolidates noisy supervision videos. We establish the first benchmark for semantic 3D motion transfer and demonstrate superior motion fidelity and structural consistency compared to adapted baselines. Code and data for this paper available at https://gsgd-motiontransfer.github.io/
>
---
#### [new 070] WaveFuse-AL: Cyclical and Performance-Adaptive Multi-Strategy Active Learning for Medical Images
- **分类: cs.CV; cs.LG**

- **简介: 论文提出WaveFuse-AL框架，用于医疗图像中的主动学习任务，解决单一策略在不同训练阶段表现不稳定的问题。通过周期性与性能自适应融合多种采样策略，提升标注效率与模型性能。**

- **链接: [https://arxiv.org/pdf/2511.15132v1](https://arxiv.org/pdf/2511.15132v1)**

> **作者:** Nishchala Thakur; Swati Kochhar; Deepti R. Bathula; Sukrit Gupta
>
> **摘要:** Active learning reduces annotation costs in medical imaging by strategically selecting the most informative samples for labeling. However, individual acquisition strategies often exhibit inconsistent behavior across different stages of the active learning cycle. We propose Cyclical and Performance-Adaptive Multi-Strategy Active Learning (WaveFuse-AL), a novel framework that adaptively fuses multiple established acquisition strategies-BALD, BADGE, Entropy, and CoreSet throughout the learning process. WaveFuse-AL integrates cyclical (sinusoidal) temporal priors with performance-driven adaptation to dynamically adjust strategy importance over time. We evaluate WaveFuse-AL on three medical imaging benchmarks: APTOS-2019 (multi-class classification), RSNA Pneumonia Detection (binary classification), and ISIC-2018 (skin lesion segmentation). Experimental results demonstrate that WaveFuse-AL consistently outperforms both single-strategy and alternating-strategy baselines, achieving statistically significant performance improvements (on ten out of twelve metric measurements) while maximizing the utility of limited annotation budgets.
>
---
#### [new 071] FlashMesh: Faster and Better Autoregressive Mesh Synthesis via Structured Speculation
- **分类: cs.CV**

- **简介: 论文提出FlashMesh，用于加速和提升3D网格生成质量。针对自回归模型推理慢的问题，利用网格结构相关性实现多标记并行推测，显著提速且保持高保真度。**

- **链接: [https://arxiv.org/pdf/2511.15618v1](https://arxiv.org/pdf/2511.15618v1)**

> **作者:** Tingrui Shen; Yiheng Zhang; Chen Tang; Chuan Ping; Zixing Zhao; Le Wan; Yuwang Wang; Ronggang Wang; Shengfeng He
>
> **摘要:** Autoregressive models can generate high-quality 3D meshes by sequentially producing vertices and faces, but their token-by-token decoding results in slow inference, limiting practical use in interactive and large-scale applications. We present FlashMesh, a fast and high-fidelity mesh generation framework that rethinks autoregressive decoding through a predict-correct-verify paradigm. The key insight is that mesh tokens exhibit strong structural and geometric correlations that enable confident multi-token speculation. FlashMesh leverages this by introducing a speculative decoding scheme tailored to the commonly used hourglass transformer architecture, enabling parallel prediction across face, point, and coordinate levels. Extensive experiments show that FlashMesh achieves up to a 2 x speedup over standard autoregressive models while also improving generation fidelity. Our results demonstrate that structural priors in mesh data can be systematically harnessed to accelerate and enhance autoregressive generation.
>
---
#### [new 072] RS-CA-HSICT: A Residual and Spatial Channel Augmented CNN Transformer Framework for Monkeypox Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出RS-CA-HSICT框架，用于猴痘（MPox）图像分类任务，旨在提升检测准确率。通过融合CNN与Transformer优势，增强特征表达能力，解决复杂纹理和局部差异识别难题。**

- **链接: [https://arxiv.org/pdf/2511.15476v1](https://arxiv.org/pdf/2511.15476v1)**

> **作者:** Rashid Iqbal; Saddam Hussain Khan
>
> **备注:** 33 Pages, 12 Figure, 4 Tables
>
> **摘要:** This work proposes a hybrid deep learning approach, namely Residual and Spatial Learning based Channel Augmented Integrated CNN-Transformer architecture, that leverages the strengths of CNN and Transformer towards enhanced MPox detection. The proposed RS-CA-HSICT framework is composed of an HSICT block, a residual CNN module, a spatial CNN block, and a CA, which enhances the diverse feature space, detailed lesion information, and long-range dependencies. The new HSICT module first integrates an abstract representation of the stem CNN and customized ICT blocks for efficient multihead attention and structured CNN layers with homogeneous (H) and structural (S) operations. The customized ICT blocks learn global contextual interactions and local texture extraction. Additionally, H and S layers learn spatial homogeneity and fine structural details by reducing noise and modeling complex morphological variations. Moreover, inverse residual learning enhances vanishing gradient, and stage-wise resolution reduction ensures scale invariance. Furthermore, the RS-CA-HSICT framework augments the learned HSICT channels with the TL-driven Residual and Spatial CNN maps for enhanced multiscale feature space capturing global and localized structural cues, subtle texture, and contrast variations. These channels, preceding augmentation, are refined through the Channel-Fusion-and-Attention block, which preserves discriminative channels while suppressing redundant ones, thereby enabling efficient computation. Finally, the spatial attention mechanism refines pixel selection to detect subtle patterns and intra-class contrast variations in Mpox. Experimental results on both the Kaggle benchmark and a diverse MPox dataset reported classification accuracy as high as 98.30% and an F1-score of 98.13%, which outperforms the existing CNNs and ViTs.
>
---
#### [new 073] HV-Attack: Hierarchical Visual Attack for Multimodal Retrieval Augmented Generation
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 论文提出HV-Attack，一种针对多模态检索增强生成（MRAG）系统的层级视觉攻击方法，通过在用户图像输入中添加不可察觉的扰动，破坏检索与生成环节的对齐，从而降低MRAG性能。**

- **链接: [https://arxiv.org/pdf/2511.15435v1](https://arxiv.org/pdf/2511.15435v1)**

> **作者:** Linyin Luo; Yujuan Ding; Yunshan Ma; Wenqi Fan; Hanjiang Lai
>
> **摘要:** Advanced multimodal Retrieval-Augmented Generation (MRAG) techniques have been widely applied to enhance the capabilities of Large Multimodal Models (LMMs), but they also bring along novel safety issues. Existing adversarial research has revealed the vulnerability of MRAG systems to knowledge poisoning attacks, which fool the retriever into recalling injected poisoned contents. However, our work considers a different setting: visual attack of MRAG by solely adding imperceptible perturbations at the image inputs of users, without manipulating any other components. This is challenging due to the robustness of fine-tuned retrievers and large-scale generators, and the effect of visual perturbation may be further weakened by propagation through the RAG chain. We propose a novel Hierarchical Visual Attack that misaligns and disrupts the two inputs (the multimodal query and the augmented knowledge) of MRAG's generator to confuse its generation. We further design a hierarchical two-stage strategy to obtain misaligned augmented knowledge. We disrupt the image input of the retriever to make it recall irrelevant knowledge from the original database, by optimizing the perturbation which first breaks the cross-modal alignment and then disrupts the multimodal semantic alignment. We conduct extensive experiments on two widely-used MRAG datasets: OK-VQA and InfoSeek. We use CLIP-based retrievers and two LMMs BLIP-2 and LLaVA as generators. Results demonstrate the effectiveness of our visual attack on MRAG through the significant decrease in both retrieval and generation performance.
>
---
#### [new 074] Hierarchical Semantic Tree Anchoring for CLIP-Based Class-Incremental Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于类增量学习（CIL）任务，旨在解决CLIP模型在持续学习中新类导致旧类遗忘的问题。提出HASTEN方法，通过超球面嵌入和梯度投影抑制遗忘，显式建模视觉与语言概念的层次结构，提升模型稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2511.15633v1](https://arxiv.org/pdf/2511.15633v1)**

> **作者:** Tao Hu; Lan Li; Zhen-Hao Xie; Da-Wei Zhou
>
> **摘要:** Class-Incremental Learning (CIL) enables models to learn new classes continually while preserving past knowledge. Recently, vision-language models like CLIP offer transferable features via multi-modal pre-training, making them well-suited for CIL. However, real-world visual and linguistic concepts are inherently hierarchical: a textual concept like "dog" subsumes fine-grained categories such as "Labrador" and "Golden Retriever," and each category entails its images. But existing CLIP-based CIL methods fail to explicitly capture this inherent hierarchy, leading to fine-grained class features drift during incremental updates and ultimately to catastrophic forgetting. To address this challenge, we propose HASTEN (Hierarchical Semantic Tree Anchoring) that anchors hierarchical information into CIL to reduce catastrophic forgetting. First, we employ an external knowledge graph as supervision to embed visual and textual features in hyperbolic space, effectively preserving hierarchical structure as data evolves. Second, to mitigate catastrophic forgetting, we project gradients onto the null space of the shared hyperbolic mapper, preventing interference with prior tasks. These two steps work synergistically to enable the model to resist forgetting by maintaining hierarchical relationships. Extensive experiments show that HASTEN consistently outperforms existing methods while providing a unified structured representation.
>
---
#### [new 075] Edge-Centric Relational Reasoning for 3D Scene Graph Prediction
- **分类: cs.CV**

- **简介: 该论文针对3D场景图预测任务，解决现有方法难以捕捉高阶关系依赖的问题。提出LEO框架，通过线图神经网络实现从关系级到物体级的渐进推理，增强关系预测准确性，并可兼容现有方法。**

- **链接: [https://arxiv.org/pdf/2511.15288v1](https://arxiv.org/pdf/2511.15288v1)**

> **作者:** Yanni Ma; Hao Liu; Yulan Guo; Theo Gevers; Martin R. Oswald
>
> **摘要:** 3D scene graph prediction aims to abstract complex 3D environments into structured graphs consisting of objects and their pairwise relationships. Existing approaches typically adopt object-centric graph neural networks, where relation edge features are iteratively updated by aggregating messages from connected object nodes. However, this design inherently restricts relation representations to pairwise object context, making it difficult to capture high-order relational dependencies that are essential for accurate relation prediction. To address this limitation, we propose a Link-guided Edge-centric relational reasoning framework with Object-aware fusion, namely LEO, which enables progressive reasoning from relation-level context to object-level understanding. Specifically, LEO first predicts potential links between object pairs to suppress irrelevant edges, and then transforms the original scene graph into a line graph where each relation is treated as a node. A line graph neural network is applied to perform edge-centric relational reasoning to capture inter-relation context. The enriched relation features are subsequently integrated into the original object-centric graph to enhance object-level reasoning and improve relation prediction. Our framework is model-agnostic and can be integrated with any existing object-centric method. Experiments on the 3DSSG dataset with two competitive baselines show consistent improvements, highlighting the effectiveness of our edge-to-object reasoning paradigm.
>
---
#### [new 076] BrainRotViT: Transformer-ResNet Hybrid for Explainable Modeling of Brain Aging from 3D sMRI
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出BrainRotViT模型，用于3D结构MRI脑龄预测任务，解决传统方法泛化差、解释性弱的问题。通过混合Transformer与CNN架构，实现高效、可解释的脑龄估计，并揭示与神经退行性疾病相关的脑区变化。**

- **链接: [https://arxiv.org/pdf/2511.15188v1](https://arxiv.org/pdf/2511.15188v1)**

> **作者:** Wasif Jalal; Md Nafiu Rahman; M. Sohel Rahman
>
> **摘要:** Accurate brain age estimation from structural MRI is a valuable biomarker for studying aging and neurodegeneration. Traditional regression and CNN-based methods face limitations such as manual feature engineering, limited receptive fields, and overfitting on heterogeneous data. Pure transformer models, while effective, require large datasets and high computational cost. We propose Brain ResNet over trained Vision Transformer (BrainRotViT), a hybrid architecture that combines the global context modeling of vision transformers (ViT) with the local refinement of residual CNNs. A ViT encoder is first trained on an auxiliary age and sex classification task to learn slice-level features. The frozen encoder is then applied to all sagittal slices to generate a 2D matrix of embedding vectors, which is fed into a residual CNN regressor that incorporates subject sex at the final fully-connected layer to estimate continuous brain age. Our method achieves an MAE of 3.34 years (Pearson $r=0.98$, Spearman $ρ=0.97$, $R^2=0.95$) on validation across 11 MRI datasets encompassing more than 130 acquisition sites, outperforming baseline and state-of-the-art models. It also generalizes well across 4 independent cohorts with MAEs between 3.77 and 5.04 years. Analyses on the brain age gap (the difference between the predicted age and actual age) show that aging patterns are associated with Alzheimer's disease, cognitive impairment, and autism spectrum disorder. Model attention maps highlight aging-associated regions of the brain, notably the cerebellar vermis, precentral and postcentral gyri, temporal lobes, and medial superior frontal gyrus. Our results demonstrate that this method provides an efficient, interpretable, and generalizable framework for brain-age prediction, bridging the gap between CNN- and transformer-based approaches while opening new avenues for aging and neurodegeneration research.
>
---
#### [new 077] Insert In Style: A Zero-Shot Generative Framework for Harmonious Cross-Domain Object Composition
- **分类: cs.CV**

- **简介: 该论文提出Insert In Style，一个零样本生成框架，用于在风格化领域中和谐地插入真实物体。解决现有方法在跨域物体组合时缺乏生成保真度或需繁琐微调的问题。通过多阶段训练和掩码注意力机制实现身份、风格与构图的解耦，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.15197v1](https://arxiv.org/pdf/2511.15197v1)**

> **作者:** Raghu Vamsi Chittersu; Yuvraj Singh Rathore; Pranav Adlinge; Kunal Swami
>
> **摘要:** Reference-based object composition methods fail when inserting real-world objects into stylized domains. This under-explored problem is currently split between practical "blenders" that lack generative fidelity and "generators" that require impractical, per-subject online finetuning. In this work, we introduce Insert In Style, the first zero-shot generative framework that is both practical and high-fidelity. Our core contribution is a unified framework with two key innovations: (i) a novel multi-stage training protocol that disentangles representations for identity, style, and composition, and (ii) a specialized masked-attention architecture that surgically enforces this disentanglement during generation. This approach prevents the concept interference common in general-purpose, unified-attention models. Our framework is trained on a new 100k sample dataset, curated from a novel data pipeline. This pipeline couples large-scale generation with a rigorous, two-stage filtering process to ensure both high-fidelity semantic identity and style coherence. Unlike prior work, our model is truly zero-shot and requires no text prompts. We also introduce a new public benchmark for stylized composition. We demonstrate state-of-the-art performance, significantly outperforming existing methods on both identity and style metrics, a result strongly corroborated by user studies.
>
---
#### [new 078] BokehFlow: Depth-Free Controllable Bokeh Rendering via Flow Matching
- **分类: cs.CV**

- **简介: 该论文提出BokehFlow，一种无需深度图的可控虚化渲染方法。解决现有方法依赖深度信息或控制能力弱的问题，通过流匹配和文本提示实现精准语义控制，提升渲染质量和效率。**

- **链接: [https://arxiv.org/pdf/2511.15066v1](https://arxiv.org/pdf/2511.15066v1)**

> **作者:** Yachuan Huang; Xianrui Luo; Qiwen Wang; Liao Shen; Jiaqi Li; Huiqiang Sun; Zihao Huang; Wei Jiang; Zhiguo Cao
>
> **摘要:** Bokeh rendering simulates the shallow depth-of-field effect in photography, enhancing visual aesthetics and guiding viewer attention to regions of interest. Although recent approaches perform well, rendering controllable bokeh without additional depth inputs remains a significant challenge. Existing classical and neural controllable methods rely on accurate depth maps, while generative approaches often struggle with limited controllability and efficiency. In this paper, we propose BokehFlow, a depth-free framework for controllable bokeh rendering based on flow matching. BokehFlow directly synthesizes photorealistic bokeh effects from all-in-focus images, eliminating the need for depth inputs. It employs a cross-attention mechanism to enable semantic control over both focus regions and blur intensity via text prompts. To support training and evaluation, we collect and synthesize four datasets. Extensive experiments demonstrate that BokehFlow achieves visually compelling bokeh effects and offers precise control, outperforming existing depth-dependent and generative methods in both rendering quality and efficiency.
>
---
#### [new 079] Deep Learning for Accurate Vision-based Catch Composition in Tropical Tuna Purse Seiners
- **分类: cs.CV**

- **简介: 论文提出基于深度学习的视觉方法，解决热带金枪鱼围网渔船中鱼种组成估算难题。通过多阶段pipeline实现个体分割、跟踪与分类，结合YOLOv9-SAM2与分层分类策略，提升准确率至84.8%，误差仅4.5%。**

- **链接: [https://arxiv.org/pdf/2511.15468v1](https://arxiv.org/pdf/2511.15468v1)**

> **作者:** Xabier Lekunberri; Ahmad Kamal; Izaro Goienetxea; Jon Ruiz; Iñaki Quincoces; Jaime Valls Miro; Ignacio Arganda-Carreras; Jose A. Fernandes-Salvador
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** Purse seiners play a crucial role in tuna fishing, as approximately 69% of the world's tropical tuna is caught using this gear. All tuna Regional Fisheries Management Organizations have established minimum standards to use electronic monitoring (EM) in fisheries in addition to traditional observers. The EM systems produce a massive amount of video data that human analysts must process. Integrating artificial intelligence (AI) into their workflow can decrease that workload and improve the accuracy of the reports. However, species identification still poses significant challenges for AI, as achieving balanced performance across all species requires appropriate training data. Here, we quantify the difficulty experts face to distinguish bigeye tuna (BET, Thunnus Obesus) from yellowfin tuna (YFT, Thunnus Albacares) using images captured by EM systems. We found inter-expert agreements of 42.9% $\pm$ 35.6% for BET and 57.1% $\pm$ 35.6% for YFT. We then present a multi-stage pipeline to estimate the species composition of the catches using a reliable ground-truth dataset based on identifications made by observers on board. Three segmentation approaches are compared: Mask R-CNN, a combination of DINOv2 with SAM2, and a integration of YOLOv9 with SAM2. We found that the latest performs the best, with a validation mean average precision of 0.66 $\pm$ 0.03 and a recall of 0.88 $\pm$ 0.03. Segmented individuals are tracked using ByteTrack. For classification, we evaluate a standard multiclass classification model and a hierarchical approach, finding a superior generalization by the hierarchical. All our models were cross-validated during training and tested on fishing operations with fully known catch composition. Combining YOLOv9-SAM2 with the hierarchical classification produced the best estimations, with 84.8% of the individuals being segmented and classified with a mean average error of 4.5%.
>
---
#### [new 080] INQUIRE-Search: A Framework for Interactive Discovery in Large-Scale Biodiversity Databases
- **分类: cs.CV**

- **简介: 论文提出INQUIRE-Search框架，解决大规模生物多样性图像中生态信息难以高效利用的问题。通过自然语言交互式搜索，支持科学家快速发现、验证和分析图像数据，提升科研效率与可探索性。**

- **链接: [https://arxiv.org/pdf/2511.15656v1](https://arxiv.org/pdf/2511.15656v1)**

> **作者:** Edward Vendrow; Julia Chae; Rupa Kurinchi-Vendhan; Isaac Eckert; Jazlynn Hall; Marta Jarzyna; Reymond Miyajima; Ruth Oliver; Laura Pollock; Lauren Schrack; Scott Yanco; Oisin Mac Aodha; Sara Beery
>
> **备注:** EV, JC, RKV contributed equally
>
> **摘要:** Large community science platforms such as iNaturalist contain hundreds of millions of biodiversity images that often capture ecological context on behaviors, interactions, phenology, and habitat. Yet most ecological workflows rely on metadata filtering or manual inspection, leaving this secondary information inaccessible at scale. We introduce INQUIRE-Search, an open-source system that enables scientists to rapidly and interactively search within an ecological image database for specific concepts using natural language, verify and export relevant observations, and utilize this discovered data for novel scientific analysis. Compared to traditional methods, INQUIRE-Search takes a fraction of the time, opening up new possibilities for scientific questions that can be explored. Through five case studies, we show the diversity of scientific applications that a tool like INQUIRE-Search can support, from seasonal variation in behavior across species to forest regrowth after wildfires. These examples demonstrate a new paradigm for interactive, efficient, and scalable scientific discovery that can begin to unlock previously inaccessible scientific value in large-scale biodiversity datasets. Finally, we emphasize using such AI-enabled discovery tools for science call for experts to reframe the priorities of the scientific process and develop novel methods for experiment design, data collection, survey effort, and uncertainty analysis.
>
---
#### [new 081] RoMa v2: Harder Better Faster Denser Feature Matching
- **分类: cs.CV**

- **简介: 论文提出RoMa v2，改进密集特征匹配任务，解决现有方法在复杂场景下准确率低、速度慢的问题。通过新架构、损失函数、训练管道和CUDA优化，实现更准确、高效、低内存的匹配效果。**

- **链接: [https://arxiv.org/pdf/2511.15706v1](https://arxiv.org/pdf/2511.15706v1)**

> **作者:** Johan Edstedt; David Nordström; Yushan Zhang; Georg Bökman; Jonathan Astermark; Viktor Larsson; Anders Heyden; Fredrik Kahl; Mårten Wadenbäck; Michael Felsberg
>
> **摘要:** Dense feature matching aims to estimate all correspondences between two images of a 3D scene and has recently been established as the gold-standard due to its high accuracy and robustness. However, existing dense matchers still fail or perform poorly for many hard real-world scenarios, and high-precision models are often slow, limiting their applicability. In this paper, we attack these weaknesses on a wide front through a series of systematic improvements that together yield a significantly better model. In particular, we construct a novel matching architecture and loss, which, combined with a curated diverse training distribution, enables our model to solve many complex matching tasks. We further make training faster through a decoupled two-stage matching-then-refinement pipeline, and at the same time, significantly reduce refinement memory usage through a custom CUDA kernel. Finally, we leverage the recent DINOv3 foundation model along with multiple other insights to make the model more robust and unbiased. In our extensive set of experiments we show that the resulting novel matcher sets a new state-of-the-art, being significantly more accurate than its predecessors. Code is available at https://github.com/Parskatt/romav2
>
---
#### [new 082] WarNav: An Autonomous Driving Benchmark for Segmentation of Navigable Zones in War Scenes
- **分类: cs.CV**

- **简介: 该论文提出WarNav数据集，用于战争场景下自主车辆可行驶区域分割任务，解决极端环境下标注数据稀缺与模型泛化能力不足的问题。通过构建真实战场图像数据集并测试主流分割模型，提供基准结果与无标注训练策略，推动高风险场景下自动驾驶的鲁棒性发展。**

- **链接: [https://arxiv.org/pdf/2511.15429v1](https://arxiv.org/pdf/2511.15429v1)**

> **作者:** Marc-Emmanuel Coupvent des Graviers; Hejer Ammar; Christophe Guettier; Yann Dumortier; Romaric Audigier
>
> **备注:** Accepted at CAID (Conference on Artificial Intelligence for Defence)
>
> **摘要:** We introduce WarNav, a novel real-world dataset constructed from images of the open-source DATTALION repository, specifically tailored to enable the development and benchmarking of semantic segmentation models for autonomous ground vehicle navigation in unstructured, conflict-affected environments. This dataset addresses a critical gap between conventional urban driving resources and the unique operational scenarios encountered by unmanned systems in hazardous and damaged war-zones. We detail the methodological challenges encountered, ranging from data heterogeneity to ethical considerations, providing guidance for future efforts that target extreme operational contexts. To establish performance references, we report baseline results on WarNav using several state-of-the-art semantic segmentation models trained on structured urban scenes. We further analyse the impact of training data environments and propose a first step towards effective navigability in challenging environments with the constraint of having no annotation of the targeted images. Our goal is to foster impactful research that enhances the robustness and safety of autonomous vehicles in high-risk scenarios while being frugal in annotated data.
>
---
#### [new 083] DCL-SE: Dynamic Curriculum Learning for Spatiotemporal Encoding of Brain Imaging
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出DCL-SE框架，解决脑影像分析中时空保真度低和模型适应性差的问题。通过数据驱动的时空编码与动态课程学习策略，提升分类、分割和预测任务的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.15151v1](https://arxiv.org/pdf/2511.15151v1)**

> **作者:** Meihua Zhou; Xinyu Tong; Jiarui Zhao; Min Cheng; Li Yang; Lei Tian; Nan Wan
>
> **摘要:** High-dimensional neuroimaging analyses for clinical diagnosis are often constrained by compromises in spatiotemporal fidelity and by the limited adaptability of large-scale, general-purpose models. To address these challenges, we introduce Dynamic Curriculum Learning for Spatiotemporal Encoding (DCL-SE), an end-to-end framework centered on data-driven spatiotemporal encoding (DaSE). We leverage Approximate Rank Pooling (ARP) to efficiently encode three-dimensional volumetric brain data into information-rich, two-dimensional dynamic representations, and then employ a dynamic curriculum learning strategy, guided by a Dynamic Group Mechanism (DGM), to progressively train the decoder, refining feature extraction from global anatomical structures to fine pathological details. Evaluated across six publicly available datasets, including Alzheimer's disease and brain tumor classification, cerebral artery segmentation, and brain age prediction, DCL-SE consistently outperforms existing methods in accuracy, robustness, and interpretability. These findings underscore the critical importance of compact, task-specific architectures in the era of large-scale pretrained networks.
>
---
#### [new 084] Kandinsky 5.0: A Family of Foundation Models for Image and Video Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出Kandinsky 5.0，一组用于高分辨率图像和10秒视频生成的前沿基础模型，解决高质量生成与高效推理问题。工作包括多阶段训练优化、数据处理流程改进及架构创新，实现性能与速度的显著提升。**

- **链接: [https://arxiv.org/pdf/2511.14993v1](https://arxiv.org/pdf/2511.14993v1)**

> **作者:** Vladimir Arkhipkin; Vladimir Korviakov; Nikolai Gerasimenko; Denis Parkhomenko; Viacheslav Vasilev; Alexey Letunovskiy; Maria Kovaleva; Nikolai Vaulin; Ivan Kirillov; Lev Novitskiy; Denis Koposov; Nikita Kiselev; Alexander Varlamov; Dmitrii Mikhailov; Vladimir Polovnikov; Andrey Shutkin; Ilya Vasiliev; Julia Agafonova; Anastasiia Kargapoltseva; Anna Dmitrienko; Anastasia Maltseva; Anna Averchenkova; Olga Kim; Tatiana Nikulina; Denis Dimitrov
>
> **备注:** Website: https://kandinskylab.ai/
>
> **摘要:** This report introduces Kandinsky 5.0, a family of state-of-the-art foundation models for high-resolution image and 10-second video synthesis. The framework comprises three core line-up of models: Kandinsky 5.0 Image Lite - a line-up of 6B parameter image generation models, Kandinsky 5.0 Video Lite - a fast and lightweight 2B parameter text-to-video and image-to-video models, and Kandinsky 5.0 Video Pro - 19B parameter models that achieves superior video generation quality. We provide a comprehensive review of the data curation lifecycle - including collection, processing, filtering and clustering - for the multi-stage training pipeline that involves extensive pre-training and incorporates quality-enhancement techniques such as self-supervised fine-tuning (SFT) and reinforcement learning (RL)-based post-training. We also present novel architectural, training, and inference optimizations that enable Kandinsky 5.0 to achieve high generation speeds and state-of-the-art performance across various tasks, as demonstrated by human evaluation. As a large-scale, publicly available generative framework, Kandinsky 5.0 leverages the full potential of its pre-training and subsequent stages to be adapted for a wide range of generative applications. We hope that this report, together with the release of our open-source code and training checkpoints, will substantially advance the development and accessibility of high-quality generative models for the research community.
>
---
#### [new 085] RocSync: Millisecond-Accurate Temporal Synchronization for Heterogeneous Camera Systems
- **分类: cs.CV**

- **简介: 论文提出RocSync方法，解决异构相机系统（RGB/IR）的毫秒级时间同步问题。通过自建LED时钟编码时间信息，实现多相机视频流的高精度对齐，提升3D重建与姿态估计等下游任务性能。**

- **链接: [https://arxiv.org/pdf/2511.14948v1](https://arxiv.org/pdf/2511.14948v1)**

> **作者:** Jaro Meyer; Frédéric Giraud; Joschua Wüthrich; Marc Pollefeys; Philipp Fürnstahl; Lilian Calvet
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Accurate spatiotemporal alignment of multi-view video streams is essential for a wide range of dynamic-scene applications such as multi-view 3D reconstruction, pose estimation, and scene understanding. However, synchronizing multiple cameras remains a significant challenge, especially in heterogeneous setups combining professional and consumer-grade devices, visible and infrared sensors, or systems with and without audio, where common hardware synchronization capabilities are often unavailable. This limitation is particularly evident in real-world environments, where controlled capture conditions are not feasible. In this work, we present a low-cost, general-purpose synchronization method that achieves millisecond-level temporal alignment across diverse camera systems while supporting both visible (RGB) and infrared (IR) modalities. The proposed solution employs a custom-built \textit{LED Clock} that encodes time through red and infrared LEDs, allowing visual decoding of the exposure window (start and end times) from recorded frames for millisecond-level synchronization. We benchmark our method against hardware synchronization and achieve a residual error of 1.34~ms RMSE across multiple recordings. In further experiments, our method outperforms light-, audio-, and timecode-based synchronization approaches and directly improves downstream computer vision tasks, including multi-view pose estimation and 3D reconstruction. Finally, we validate the system in large-scale surgical recordings involving over 25 heterogeneous cameras spanning both IR and RGB modalities. This solution simplifies and streamlines the synchronization pipeline and expands access to advanced vision-based sensing in unconstrained environments, including industrial and clinical applications.
>
---
#### [new 086] A Dataset and Baseline for Deep Learning-Based Visual Quality Inspection in Remanufacturing
- **分类: cs.CV**

- **简介: 论文针对再制造中视觉质量检测难题，提出一个齿轮箱部件图像数据集和基线模型。通过对比正则化损失提升深度学习模型对新部件的泛化能力，解决缺陷检测模型在不同产品变体上性能下降的问题。**

- **链接: [https://arxiv.org/pdf/2511.15440v1](https://arxiv.org/pdf/2511.15440v1)**

> **作者:** Johannes C. Bauer; Paul Geng; Stephan Trattnig; Petr Dokládal; Rüdiger Daub
>
> **摘要:** Remanufacturing describes a process where worn products are restored to like-new condition and it offers vast ecological and economic potentials. A key step is the quality inspection of disassembled components, which is mostly done manually due to the high variety of parts and defect patterns. Deep neural networks show great potential to automate such visual inspection tasks but struggle to generalize to new product variants, components, or defect patterns. To tackle this challenge, we propose a novel image dataset depicting typical gearbox components in good and defective condition from two automotive transmissions. Depending on the train-test split of the data, different distribution shifts are generated to benchmark the generalization ability of a classification model. We evaluate different models using the dataset and propose a contrastive regularization loss to enhance model robustness. The results obtained demonstrate the ability of the loss to improve generalisation to unseen types of components.
>
---
#### [new 087] Breaking Expert Knowledge Limits: Self-Pruning for Large Language Models
- **分类: cs.CV**

- **简介: 论文提出AutoPrune方法，解决大语言模型剪枝依赖专家知识和高剪枝率下性能下降的问题。通过自动生成剪枝算法和动态稀疏分配策略，实现无需人工干预的高效模型压缩。**

- **链接: [https://arxiv.org/pdf/2511.15390v1](https://arxiv.org/pdf/2511.15390v1)**

> **作者:** Haidong Kang; Lihong Lin; Enneng Yang; Hongning Dai; Hao Wang
>
> **摘要:** Large language models (LLMs) have achieved remarkable performance on a wide range of tasks, hindering real-world deployment due to their massive size. Existing pruning methods (e.g., Wanda) tailored for LLMs rely heavily on manual design pruning algorithms, thereby leading to \textit{huge labor costs} and \textit{requires expert knowledge}. Furthermore, we are the first to identify the serious \textit{outlier value issue} behind dramatic performance degradation under high pruning ratios that are caused by uniform sparsity, raising an additional concern about how to design adaptive pruning sparsity ideal for LLMs. Can LLMs prune by themselves? In this work, we introduce an affirmative answer by proposing a novel pruning method called \textbf{AutoPrune}, which first overcomes expert knowledge limits by leveraging LLMs to design optimal pruning algorithms for themselves automatically without any expert knowledge. Specifically, to mitigate the black-box nature of LLMs, we propose a Graph-driven Chain-of-Thought (GCoT) to optimize prompts, significantly enhancing the reasoning process in learning the pruning algorithm and enabling us to generate pruning algorithms with superior performance and interpretability in the next generation. Finally, grounded in insights of outlier value issue, we introduce Skew-aware Dynamic Sparsity Allocation (SDSA) to overcome the outlier value issue, mitigating performance degradation under high pruning ratios. We conduct extensive experiments on mainstream LLMs benchmarks, demonstrating the superiority of AutoPrune, which consistently excels state-of-the-art competitors. The code is available at: https://anonymous.4open.science/r/AutoPrune.
>
---
#### [new 088] VisPlay: Self-Evolving Vision-Language Models from Images
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 论文提出VisPlay框架，通过自监督强化学习让视觉语言模型从无标签图像中自主提升推理能力，解决人工标注成本高、难以扩展的问题。通过角色交互与GRPO优化，实现视觉问答质量与难度的平衡，显著提升多任务表现。**

- **链接: [https://arxiv.org/pdf/2511.15661v1](https://arxiv.org/pdf/2511.15661v1)**

> **作者:** Yicheng He; Chengsong Huang; Zongxia Li; Jiaxin Huang; Yonghui Yang
>
> **摘要:** Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained with Group Relative Policy Optimization (GRPO), which incorporates diversity and difficulty rewards to balance the complexity of generated questions with the quality of the silver answers. VisPlay scales efficiently across two model families. When trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks, including MM-Vet and MMMU, demonstrating a scalable path toward self-evolving multimodal intelligence. The project page is available at https://bruno686.github.io/VisPlay/
>
---
#### [new 089] X-WIN: Building Chest Radiograph World Model via Predictive Sensing
- **分类: cs.CV**

- **简介: 该论文提出X-WIN模型，用于构建胸部X光的世界模型。针对CXRs因2D投影导致的3D结构信息缺失问题，通过学习CT体积在潜在空间中的2D投影来捕获3D知识，提升疾病诊断和图像重建能力。**

- **链接: [https://arxiv.org/pdf/2511.14918v1](https://arxiv.org/pdf/2511.14918v1)**

> **作者:** Zefan Yang; Ge Wang; James Hendler; Mannudeep K. Kalra; Pingkun Yan
>
> **摘要:** Chest X-ray radiography (CXR) is an essential medical imaging technique for disease diagnosis. However, as 2D projectional images, CXRs are limited by structural superposition and hence fail to capture 3D anatomies. This limitation makes representation learning and disease diagnosis challenging. To address this challenge, we propose a novel CXR world model named X-WIN, which distills volumetric knowledge from chest computed tomography (CT) by learning to predict its 2D projections in latent space. The core idea is that a world model with internalized knowledge of 3D anatomical structure can predict CXRs under various transformations in 3D space. During projection prediction, we introduce an affinity-guided contrastive alignment loss that leverages mutual similarities to capture rich, correlated information across projections from the same volume. To improve model adaptability, we incorporate real CXRs into training through masked image modeling and employ a domain classifier to encourage statistically similar representations for real and simulated CXRs. Comprehensive experiments show that X-WIN outperforms existing foundation models on diverse downstream tasks using linear probing and few-shot fine-tuning. X-WIN also demonstrates the ability to render 2D projections for reconstructing a 3D CT volume.
>
---
#### [new 090] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset
- **分类: cs.CV**

- **简介: 论文提出指令引导的肺部X光图像病灶分割方法（ILS），解决现有模型依赖专家标注和有限病灶类型的问题。构建了首个大规模自动标注数据集MIMIC-ILS，并训练了可响应简单指令并生成分割结果与解释的模型ROSALIA。**

- **链接: [https://arxiv.org/pdf/2511.15186v1](https://arxiv.org/pdf/2511.15186v1)**

> **作者:** Geon Choi; Hangyul Yoon; Hyunju Shin; Hyunki Park; Sang Hoon Seo; Eunho Yang; Edward Choi
>
> **摘要:** The applicability of current lesion segmentation models for chest X-rays (CXRs) has been limited both by a small number of target labels and the reliance on long, detailed expert-level text inputs, creating a barrier to practical use. To address these limitations, we introduce a new paradigm: instruction-guided lesion segmentation (ILS), which is designed to segment diverse lesion types based on simple, user-friendly instructions. Under this paradigm, we construct MIMIC-ILS, the first large-scale instruction-answer dataset for CXR lesion segmentation, using our fully automated multimodal pipeline that generates annotations from chest X-ray images and their corresponding reports. MIMIC-ILS contains 1.1M instruction-answer pairs derived from 192K images and 91K unique segmentation masks, covering seven major lesion types. To empirically demonstrate its utility, we introduce ROSALIA, a vision-language model fine-tuned on MIMIC-ILS. ROSALIA can segment diverse lesions and provide textual explanations in response to user instructions. The model achieves high segmentation and textual accuracy in our newly proposed task, highlighting the effectiveness of our pipeline and the value of MIMIC-ILS as a foundational resource for pixel-level CXR lesion grounding.
>
---
#### [new 091] Fast Post-Hoc Confidence Fusion for 3-Class Open-Set Aerial Object Detection
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 论文提出一种轻量级后处理框架，用于无人机航空目标检测中的三类分类：已知目标、未知目标和背景。解决开放集检测中混淆未知对象与背景的问题，通过融合多 confidence 估计提升准确率，同时保持检测性能。**

- **链接: [https://arxiv.org/pdf/2511.15343v1](https://arxiv.org/pdf/2511.15343v1)**

> **作者:** Spyridon Loukovitis; Vasileios Karampinis; Athanasios Voulodimos
>
> **摘要:** Developing reliable UAV navigation systems requires robust air-to-air object detectors capable of distinguishing between objects seen during training and previously unseen objects. While many methods address closed-set detection and achieve high-confidence recognition of in-domain (ID) targets, they generally do not tackle open-set detection, which requires simultaneous handling of both ID and out-of-distribution (OOD) objects. Existing open-set approaches typically rely on a single uncertainty score with thresholding, limiting flexibility and often conflating OOD objects with background clutter. In contrast, we propose a lightweight, model-agnostic post-processing framework that explicitly separates background from unknown objects while preserving the base detector's performance. Our approach extends open-set detection beyond binary ID/OOD classification to real-time three-way classification among ID targets, OOD objects, and background. To this end, we employ a fusion scheme that aggregates multiple confidence estimates and per-detection features using a compact multilayer perceptron (MLP). Incorporating different logit variants into the MLP consistently enhances performance across both binary and three-class classification without compromising throughput. Extensive ablation and comparative experiments confirm that our method surpasses threshold-based baselines in two-class classification by an average of 2.7% AUROC, while retaining or improving open-set mAP. Furthermore, our study uniquely enables robust three-class classification, a critical capability for safe UAV navigation, where OOD objects must be actively avoided and background regions safely ignored. Comparative analysis highlights that our method surpasses competitive techniques in AUROC across datasets, while improving closed-set mAP by up to 9 points, an 18% relative gain.
>
---
#### [new 092] Learning from Mistakes: Loss-Aware Memory Enhanced Continual Learning for LiDAR Place Recognition
- **分类: cs.CV**

- **简介: 论文针对LiDAR place recognition中的灾难性遗忘问题，提出KDF+框架，通过损失感知采样和记忆增强机制，提升模型在新环境中持续学习能力，显著改善长期知识保留与性能稳定性。**

- **链接: [https://arxiv.org/pdf/2511.15597v1](https://arxiv.org/pdf/2511.15597v1)**

> **作者:** Xufei Wang; Junqiao Zhao; Siyue Tao; Qiwen Gu; Wonbong Kim; Tiantian Feng
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** LiDAR place recognition plays a crucial role in SLAM, robot navigation, and autonomous driving. However, existing LiDAR place recognition methods often struggle to adapt to new environments without forgetting previously learned knowledge, a challenge widely known as catastrophic forgetting. To address this issue, we propose KDF+, a novel continual learning framework for LiDAR place recognition that extends the KDF paradigm with a loss-aware sampling strategy and a rehearsal enhancement mechanism. The proposed sampling strategy estimates the learning difficulty of each sample via its loss value and selects samples for replay according to their estimated difficulty. Harder samples, which tend to encode more discriminative information, are sampled with higher probability while maintaining distributional coverage across the dataset. In addition, the rehearsal enhancement mechanism encourages memory samples to be further refined during new-task training by slightly reducing their loss relative to previous tasks, thereby reinforcing long-term knowledge retention. Extensive experiments across multiple benchmarks demonstrate that KDF+ consistently outperforms existing continual learning methods and can be seamlessly integrated into state-of-the-art continual learning for LiDAR place recognition frameworks to yield significant and stable performance gains. The code will be available at https://github.com/repo/KDF-plus.
>
---
#### [new 093] An Event-triggered System for Social Persuasion and Danger Alert in Elder Home Monitoring
- **分类: cs.CV; cs.MM**

- **简介: 论文提出一种事件触发系统，用于养老院中老人身心状态监测，解决安全预警与社交沟通问题。通过GMM检测行为、SVM分析图像，实现看护、危险提醒和照片分享功能，设计直观操作促进老人与亲属互动。**

- **链接: [https://arxiv.org/pdf/2511.15117v1](https://arxiv.org/pdf/2511.15117v1)**

> **作者:** Jun-Yi Liu; Chung-Hao Chen; Ya-Chi Tsao; Ssu-Yao Wu; Yu-Ting Tsao; Lyn Chao-ling Chen
>
> **备注:** Accepted in the 35th IPPR Conference on Computer Vision, Graphics, and Image Processing (CVGIP2022)
>
> **摘要:** In the study, the physical state and mental state of elders are both considered, and an event-triggered system has developed to detect events: watch dog, danger notice and photo link. By adopting GMM background modeling, the motion behavior of visitors and elders can be detected in the watch dog event and danger notice event respectively. Experiments set in home scenarios and 5 families participated in the experiments for detecting and recording three types of events from their life activities. In addition, the captured images were analyzed using SVM machine learning. For lack of technical experiences of elders, an intuitive operation as normal life activity was designed to create communication between elder and relatives via social media.
>
---
#### [new 094] Scriboora: Rethinking Human Pose Forecasting
- **分类: cs.CV**

- **简介: 论文提出Scriboora，针对人体姿态预测任务，解决现有方法可复现性差、噪声鲁棒性不足的问题。通过统一训练评估流程，引入语音模型类比提升性能，并在真实噪声数据上验证了无监督微调的有效性。**

- **链接: [https://arxiv.org/pdf/2511.15565v1](https://arxiv.org/pdf/2511.15565v1)**

> **作者:** Daniel Bermuth; Alexander Poeppel; Wolfgang Reif
>
> **摘要:** Human pose forecasting predicts future poses based on past observations, and has many significant applications in areas such as action recognition, autonomous driving or human-robot interaction. This paper evaluates a wide range of pose forecasting algorithms in the task of absolute pose forecasting, revealing many reproducibility issues, and provides a unified training and evaluation pipeline. After drawing a high-level analogy to the task of speech understanding, it is shown that recent speech models can be efficiently adapted to the task of pose forecasting, and improve current state-of-the-art performance. At last the robustness of the models is evaluated, using noisy joint coordinates obtained from a pose estimator model, to reflect a realistic type of noise, which is more close to real-world applications. For this a new dataset variation is introduced, and it is shown that estimated poses result in a substantial performance degradation, and how much of it can be recovered again by unsupervised finetuning.
>
---
#### [new 095] A Hybrid CNN-ViT-GNN Framework with GAN-Based Augmentation for Intelligent Weed Detection in Precision Agriculture
- **分类: cs.CV**

- **简介: 论文提出混合CNN-ViT-GNN框架结合GAN数据增强与自监督预训练，用于精准农业中的杂草检测任务，解决小样本和多场景下识别准确率低的问题，实现高精度、实时、可部署的智能杂草识别。**

- **链接: [https://arxiv.org/pdf/2511.15535v1](https://arxiv.org/pdf/2511.15535v1)**

> **作者:** Pandiyaraju V; Abishek Karthik; Sreya Mynampati; Poovarasan L; D. Saraswathi
>
> **摘要:** The task of weed detection is an essential element of precision agriculture since accurate species identification allows a farmer to selectively apply herbicides and fits into sustainable agriculture crop management. This paper proposes a hybrid deep learning framework recipe for weed detection that utilizes Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and Graph Neural Networks (GNNs) to build robustness to multiple field conditions. A Generative Adversarial Network (GAN)-based augmentation method was imposed to balance class distributions and better generalize the model. Further, a self-supervised contrastive pre-training method helps to learn more features from limited annotated data. Experimental results yield superior results with 99.33% accuracy, precision, recall, and F1-score on multi-benchmark datasets. The proposed model architecture enables local, global, and relational feature representations and offers high interpretability and adaptability. Practically, the framework allows real-time, efficient deployment to edge devices for automated weed detecting, reducing over-reliance on herbicides and providing scalable, sustainable precision-farming options.
>
---
#### [new 096] Complex-Valued 2D Gaussian Representation for Computer-Generated Holography
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 论文提出基于复值二维高斯基元的全息表示方法，用于计算机生成全息（CGH），解决传统方法参数空间大、显存占用高和重建质量差的问题。通过结构化表示与可微光栅化器，实现更高效、高质量的全息图生成与优化。**

- **链接: [https://arxiv.org/pdf/2511.15022v1](https://arxiv.org/pdf/2511.15022v1)**

> **作者:** Yicheng Zhan; Xiangjun Gao; Long Quan; Kaan Akşit
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** We propose a new hologram representation based on structured complex-valued 2D Gaussian primitives, which replaces per-pixel information storage and reduces the parameter search space by up to 10:1. To enable end-to-end training, we develop a differentiable rasterizer for our representation, integrated with a GPU-optimized light propagation kernel in free space. Our extensive experiments show that our method achieves up to 2.5x lower VRAM usage and 50% faster optimization while producing higher-fidelity reconstructions than existing methods. We further introduce a conversion procedure that adapts our representation to practical hologram formats, including smooth and random phase-only holograms. Our experiments show that this procedure can effectively suppress noise artifacts observed in previous methods. By reducing the hologram parameter search space, our representation enables a more scalable hologram estimation in the next-generation computer-generated holography systems.
>
---
#### [new 097] The SA-FARI Dataset: Segment Anything in Footage of Animals for Recognition and Identification
- **分类: cs.CV; cs.AI**

- **简介: 论文提出SA-FARI数据集，解决野生动物多动物跟踪（MAT）缺乏大规模、多样化标注数据的问题。该数据集包含11,609个视频、99种物种、46小时密集标注，支持通用MAT模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2511.15622v1](https://arxiv.org/pdf/2511.15622v1)**

> **作者:** Dante Francisco Wasmuht; Otto Brookes; Maximillian Schall; Pablo Palencia; Chris Beirne; Tilo Burghardt; Majid Mirmehdi; Hjalmar Kühl; Mimi Arandjelovic; Sam Pottie; Peter Bermant; Brandon Asheim; Yi Jin Toh; Adam Elzinga; Jason Holmberg; Andrew Whitworth; Eleanor Flatt; Laura Gustafson; Chaitanya Ryali; Yuan-Ting Hu; Baishan Guo; Andrew Westbury; Kate Saenko; Didac Suris
>
> **摘要:** Automated video analysis is critical for wildlife conservation. A foundational task in this domain is multi-animal tracking (MAT), which underpins applications such as individual re-identification and behavior recognition. However, existing datasets are limited in scale, constrained to a few species, or lack sufficient temporal and geographical diversity - leaving no suitable benchmark for training general-purpose MAT models applicable across wild animal populations. To address this, we introduce SA-FARI, the largest open-source MAT dataset for wild animals. It comprises 11,609 camera trap videos collected over approximately 10 years (2014-2024) from 741 locations across 4 continents, spanning 99 species categories. Each video is exhaustively annotated culminating in ~46 hours of densely annotated footage containing 16,224 masklet identities and 942,702 individual bounding boxes, segmentation masks, and species labels. Alongside the task-specific annotations, we publish anonymized camera trap locations for each video. Finally, we present comprehensive benchmarks on SA-FARI using state-of-the-art vision-language models for detection and tracking, including SAM 3, evaluated with both species-specific and generic animal prompts. We also compare against vision-only methods developed specifically for wildlife analysis. SA-FARI is the first large-scale dataset to combine high species diversity, multi-region coverage, and high-quality spatio-temporal annotations, offering a new foundation for advancing generalizable multianimal tracking in the wild. The dataset is available at $\href{https://www.conservationxlabs.com/sa-fari}{\text{conservationxlabs.com/SA-FARI}}$.
>
---
#### [new 098] MaskMed: Decoupled Mask and Class Prediction for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文提出MaskMed，用于医学图像分割任务，解决传统方法中类别与掩码耦合导致特征共享受限的问题。通过解耦掩码和类别预测，并引入全尺度感知可变形Transformer实现高效融合，显著提升分割精度。**

- **链接: [https://arxiv.org/pdf/2511.15603v1](https://arxiv.org/pdf/2511.15603v1)**

> **作者:** Bin Xie; Gady Agam
>
> **摘要:** Medical image segmentation typically adopts a point-wise convolutional segmentation head to predict dense labels, where each output channel is heuristically tied to a specific class. This rigid design limits both feature sharing and semantic generalization. In this work, we propose a unified decoupled segmentation head that separates multi-class prediction into class-agnostic mask prediction and class label prediction using shared object queries. Furthermore, we introduce a Full-Scale Aware Deformable Transformer module that enables low-resolution encoder features to attend across full-resolution encoder features via deformable attention, achieving memory-efficient and spatially aligned full-scale fusion. Our proposed method, named MaskMed, achieves state-of-the-art performance, surpassing nnUNet by +2.0% Dice on AMOS 2022 and +6.9% Dice on BTCV.
>
---
#### [new 099] Evaluating Low-Light Image Enhancement Across Multiple Intensity Levels
- **分类: cs.CV; cs.AI**

- **简介: 论文聚焦低光照图像增强任务，解决现有方法在不同光照强度下性能不稳定的问题。作者构建了多光照强度数据集MILL，系统评估主流算法并提出改进策略，显著提升不同设备在多种光照条件下的增强效果。**

- **链接: [https://arxiv.org/pdf/2511.15496v1](https://arxiv.org/pdf/2511.15496v1)**

> **作者:** Maria Pilligua; David Serrano-Lozano; Pai Peng; Ramon Baldrich; Michael S. Brown; Javier Vazquez-Corral
>
> **摘要:** Imaging in low-light environments is challenging due to reduced scene radiance, which leads to elevated sensor noise and reduced color saturation. Most learning-based low-light enhancement methods rely on paired training data captured under a single low-light condition and a well-lit reference. The lack of radiance diversity limits our understanding of how enhancement techniques perform across varying illumination intensities. We introduce the Multi-Illumination Low-Light (MILL) dataset, containing images captured at diverse light intensities under controlled conditions with fixed camera settings and precise illuminance measurements. MILL enables comprehensive evaluation of enhancement algorithms across variable lighting conditions. We benchmark several state-of-the-art methods and reveal significant performance variations across intensity levels. Leveraging the unique multi-illumination structure of our dataset, we propose improvements that enhance robustness across diverse illumination scenarios. Our modifications achieve up to 10 dB PSNR improvement for DSLR and 2 dB for the smartphone on Full HD images.
>
---
#### [new 100] FarSLIP: Discovering Effective CLIP Adaptation for Fine-Grained Remote Sensing Understanding
- **分类: cs.CV**

- **简介: 该论文针对遥感图像细粒度理解任务，解决CLIP模型空间感知不足问题。提出FarSLIP框架，构建多粒度遥感图文数据集MGRS-200k，采用patch-to-patch蒸馏与CLS-token区域对齐策略，提升细粒度视觉语言对齐效果，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.14901v1](https://arxiv.org/pdf/2511.14901v1)**

> **作者:** Zhenshi Li; Weikang Yu; Dilxat Muhtar; Xueliang Zhang; Pengfeng Xiao; Pedram Ghamisi; Xiao Xiang Zhu
>
> **摘要:** As CLIP's global alignment limits its ability to capture fine-grained details, recent efforts have focused on enhancing its region-text alignment. However, current remote sensing (RS)-specific CLIP variants still inherit this limited spatial awareness. We identify two key limitations behind this: (1) current RS image-text datasets generate global captions from object-level labels, leaving the original object-level supervision underutilized; (2) despite the success of region-text alignment methods in general domain, their direct application to RS data often leads to performance degradation. To address these, we construct the first multi-granularity RS image-text dataset, MGRS-200k, featuring rich object-level textual supervision for RS region-category alignment. We further investigate existing fine-grained CLIP tuning strategies and find that current explicit region-text alignment methods, whether in a direct or indirect way, underperform due to severe degradation of CLIP's semantic coherence. Building on these, we propose FarSLIP, a Fine-grained Aligned RS Language-Image Pretraining framework. Rather than the commonly used patch-to-CLS self-distillation, FarSLIP employs patch-to-patch distillation to align local and global visual cues, which improves feature discriminability while preserving semantic coherence. Additionally, to effectively utilize region-text supervision, it employs simple CLS token-based region-category alignment rather than explicit patch-level alignment, further enhancing spatial awareness. FarSLIP features improved fine-grained vision-language alignment in RS domain and sets a new state of the art not only on RS open-vocabulary semantic segmentation, but also on image-level tasks such as zero-shot classification and image-text retrieval. Our dataset, code, and models are available at https://github.com/NJU-LHRS/FarSLIP.
>
---
#### [new 101] CPSL: Representing Volumetric Video via Content-Promoted Scene Layers
- **分类: cs.CV; cs.MM**

- **简介: 论文提出CPSL，一种2.5D视频表示方法，用于将2D视频转化为可支持自由视角和真实运动视差的沉浸式媒体。该方法通过内容引导的分层结构实现高效存储与实时渲染，解决现有体积视频在捕获、计算和渲染上的高成本问题。**

- **链接: [https://arxiv.org/pdf/2511.14927v1](https://arxiv.org/pdf/2511.14927v1)**

> **作者:** Kaiyuan Hu; Yili Jin; Junhua Liu; Xize Duan; Hong Kang; Xue Liu
>
> **摘要:** Volumetric video enables immersive and interactive visual experiences by supporting free viewpoint exploration and realistic motion parallax. However, existing volumetric representations from explicit point clouds to implicit neural fields, remain costly in capture, computation, and rendering, which limits their scalability for on-demand video and reduces their feasibility for real-time communication. To bridge this gap, we propose Content-Promoted Scene Layers (CPSL), a compact 2.5D video representation that brings the perceptual benefits of volumetric video to conventional 2D content. Guided by per-frame depth and content saliency, CPSL decomposes each frame into a small set of geometry-consistent layers equipped with soft alpha bands and an edge-depth cache that jointly preserve occlusion ordering and boundary continuity. These lightweight, 2D-encodable assets enable parallax-corrected novel-view synthesis via depth-weighted warping and front-to-back alpha compositing, bypassing expensive 3D reconstruction. Temporally, CPSL maintains inter-frame coherence using motion-guided propagation and per-layer encoding, supporting real-time playback with standard video codecs. Across multiple benchmarks, CPSL achieves superior perceptual quality and boundary fidelity compared with layer-based and neural-field baselines while reducing storage and rendering cost by several folds. Our approach offer a practical path from 2D video to scalable 2.5D immersive media.
>
---
#### [new 102] Computer-Use Agents as Judges for Generative User Interface
- **分类: cs.CV; cs.CL; cs.HC**

- **简介: 论文提出Cuer-CUA协作框架，让计算机使用代理（CUA）作为裁判评估并优化由代码模型生成的GUI设计，提升界面的代理友好性与任务可执行性。**

- **链接: [https://arxiv.org/pdf/2511.15567v1](https://arxiv.org/pdf/2511.15567v1)**

> **作者:** Kevin Qinghong Lin; Siyuan Hu; Linjie Li; Zhengyuan Yang; Lijuan Wang; Philip Torr; Mike Zheng Shou
>
> **备注:** Project: https://showlab.github.io/AUI Github: https://github.com/showlab/AUI
>
> **摘要:** Computer-Use Agents (CUA) are becoming increasingly capable of autonomously operating digital environments through Graphical User Interfaces (GUI). Yet, most GUI remain designed primarily for humans--prioritizing aesthetics and usability--forcing agents to adopt human-oriented behaviors that are unnecessary for efficient task execution. At the same time, rapid advances in coding-oriented language models (Coder) have transformed automatic GUI design. This raises a fundamental question: Can CUA as judges to assist Coder for automatic GUI design? To investigate, we introduce AUI-Gym, a benchmark for Automatic GUI development spanning 52 applications across diverse domains. Using language models, we synthesize 1560 tasks that simulate real-world scenarios. To ensure task reliability, we further develop a verifier that programmatically checks whether each task is executable within its environment. Building on this, we propose a Coder-CUA in Collaboration framework: the Coder acts as Designer, generating and revising websites, while the CUA serves as Judge, evaluating functionality and refining designs. Success is measured not by visual appearance, but by task solvability and CUA navigation success rate. To turn CUA feedback into usable guidance, we design a CUA Dashboard that compresses multi-step navigation histories into concise visual summaries, offering interpretable guidance for iterative redesign. By positioning agents as both designers and judges, our framework shifts interface design toward agent-native efficiency and reliability. Our work takes a step toward shifting agents from passive use toward active participation in digital environments. Our code and dataset are available at https://github.com/showlab/AUI.
>
---
#### [new 103] GRPO-RM: Fine-Tuning Representation Models via GRPO-Driven Reinforcement Learning
- **分类: cs.LG; cs.CV**

- **简介: 论文提出GRPO-RM方法，将GRPO强化学习用于表示模型微调，解决LLM中GRPO难以直接应用于表示模型的问题。通过预定义输出集和定制奖励函数，提升表示模型性能，在多个数据集上验证有效性。**

- **链接: [https://arxiv.org/pdf/2511.15256v1](https://arxiv.org/pdf/2511.15256v1)**

> **作者:** Yanchen Xu; Ziheng Jiao; Hongyuan Zhang; Xuelong Li
>
> **摘要:** The Group Relative Policy Optimization (GRPO), a reinforcement learning method used to fine-tune large language models (LLMs), has proved its effectiveness in practical applications such as DeepSeek-R1. It raises a question whether GRPO can be generalized to representation learning models. In this paper, we propose Group Relative Policy Optimization for Representation Model (GRPO-RM), and investigate the performance of GRPO-like policy in post-training representation models. Specifically, our method establishes a predefined output set to functionally replace token sequence sampling in LLMs, thereby generating an output group, which is essential for the probability-driven optimization of GRPO. In addition, a specialized reward function is designed to accommodate the properties of representation models. Extensive experiments are conducted on various real-world datasets to validate the effectiveness of our proposed method.
>
---
#### [new 104] NTK-Guided Implicit Neural Teaching
- **分类: cs.LG; cs.CV**

- **简介: 论文提出NINT方法，用于加速隐式神经表示（INR）的训练。针对高分辨率信号优化计算成本高的问题，利用NTK动态选择能最大化全局更新的坐标，实现更快收敛与更优质量。**

- **链接: [https://arxiv.org/pdf/2511.15487v1](https://arxiv.org/pdf/2511.15487v1)**

> **作者:** Chen Zhang; Wei Zuo; Bingyang Cheng; Yikun Wang; Wei-Bin Kou; Yik Chung WU; Ngai Wong
>
> **备注:** Preprint
>
> **摘要:** Implicit Neural Representations (INRs) parameterize continuous signals via multilayer perceptrons (MLPs), enabling compact, resolution-independent modeling for tasks like image, audio, and 3D reconstruction. However, fitting high-resolution signals demands optimizing over millions of coordinates, incurring prohibitive computational costs. To address it, we propose NTK-Guided Implicit Neural Teaching (NINT), which accelerates training by dynamically selecting coordinates that maximize global functional updates. Leveraging the Neural Tangent Kernel (NTK), NINT scores examples by the norm of their NTK-augmented loss gradients, capturing both fitting errors and heterogeneous leverage (self-influence and cross-coordinate coupling). This dual consideration enables faster convergence compared to existing methods. Through extensive experiments, we demonstrate that NINT significantly reduces training time by nearly half while maintaining or improving representation quality, establishing state-of-the-art acceleration among recent sampling-based strategies.
>
---
#### [new 105] Deep Pathomic Learning Defines Prognostic Subtypes and Molecular Drivers in Colorectal Cancer
- **分类: cs.LG; cs.AI; cs.CV; q-bio.GN**

- **简介: 该论文属于医学AI任务，旨在解决结直肠癌预后分层不精准的问题。作者提出TDAM-CRC模型，基于病理图像实现风险分层，结合多组学分析发现MRPL37为关键预后基因，构建了可解释的临床决策工具。**

- **链接: [https://arxiv.org/pdf/2511.15067v1](https://arxiv.org/pdf/2511.15067v1)**

> **作者:** Zisong Wang; Xuanyu Wang; Hang Chen; Haizhou Wang; Yuxin Chen; Yihang Xu; Yunhe Yuan; Lihuan Luo; Xitong Ling; Xiaoping Liu
>
> **摘要:** Precise prognostic stratification of colorectal cancer (CRC) remains a major clinical challenge due to its high heterogeneity. The conventional TNM staging system is inadequate for personalized medicine. We aimed to develop and validate a novel multiple instance learning model TDAM-CRC using histopathological whole-slide images for accurate prognostic prediction and to uncover its underlying molecular mechanisms. We trained the model on the TCGA discovery cohort (n=581), validated it in an independent external cohort (n=1031), and further we integrated multi-omics data to improve model interpretability and identify novel prognostic biomarkers. The results demonstrated that the TDAM-CRC achieved robust risk stratification in both cohorts. Its predictive performance significantly outperformed the conventional clinical staging system and multiple state-of-the-art models. The TDAM-CRC risk score was confirmed as an independent prognostic factor in multivariable analysis. Multi-omics analysis revealed that the high-risk subtype is closely associated with metabolic reprogramming and an immunosuppressive tumor microenvironment. Through interaction network analysis, we identified and validated Mitochondrial Ribosomal Protein L37 (MRPL37) as a key hub gene linking deep pathomic features to clinical prognosis. We found that high expression of MRPL37, driven by promoter hypomethylation, serves as an independent biomarker of favorable prognosis. Finally, we constructed a nomogram incorporating the TDAM-CRC risk score and clinical factors to provide a precise and interpretable clinical decision-making tool for CRC patients. Our AI-driven pathological model TDAM-CRC provides a robust tool for improved CRC risk stratification, reveals new molecular targets, and facilitates personalized clinical decision-making.
>
---
#### [new 106] Application of Graph Based Vision Transformers Architectures for Accurate Temperature Prediction in Fiber Specklegram Sensors
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于温度预测任务，旨在解决光纤散斑传感器中非线性数据导致的预测精度问题。通过引入基于图结构的视觉Transformer模型（如ViT、MAP-ViGAT等），提升了预测准确性，并结合XAI技术增强模型可解释性。**

- **链接: [https://arxiv.org/pdf/2511.14792v1](https://arxiv.org/pdf/2511.14792v1)**

> **作者:** Abhishek Sebastian
>
> **摘要:** Fiber Specklegram Sensors (FSS) are highly effective for environmental monitoring, particularly for detecting temperature variations. However, the nonlinear nature of specklegram data presents significant challenges for accurate temperature prediction. This study investigates the use of transformer-based architectures, including Vision Transformers (ViTs), Swin Transformers, and emerging models such as Learnable Importance Non-Symmetric Attention Vision Transformers (LINA-ViT) and Multi-Adaptive Proximity Vision Graph Attention Transformers (MAP-ViGAT), to predict temperature from specklegram data over a range of 0 to 120 Celsius. The results show that ViTs achieved a Mean Absolute Error (MAE) of 1.15, outperforming traditional models such as CNNs. GAT-ViT and MAP-ViGAT variants also demonstrated competitive accuracy, highlighting the importance of adaptive attention mechanisms and graph-based structures in capturing complex modal interactions and phase shifts in specklegram data. Additionally, this study incorporates Explainable AI (XAI) techniques, including attention maps and saliency maps, to provide insights into the decision-making processes of the transformer models, improving interpretability and transparency. These findings establish transformer architectures as strong benchmarks for optical fiber-based temperature sensing and offer promising directions for industrial monitoring and structural health assessment applications.
>
---
#### [new 107] In-N-On: Scaling Egocentric Manipulation with in-the-wild and on-task Data
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出In-N-On方法，通过分类和利用野生与任务相关的人类视角数据，训练出可语言控制、少样本学习且鲁棒的操纵策略Human0，解决人类数据在机器人操纵中利用率低的问题。**

- **链接: [https://arxiv.org/pdf/2511.15704v1](https://arxiv.org/pdf/2511.15704v1)**

> **作者:** Xiongyi Cai; Ri-Zhao Qiu; Geng Chen; Lai Wei; Isabella Liu; Tianshu Huang; Xuxin Cheng; Xiaolong Wang
>
> **备注:** Project webpage: https://xiongyicai.github.io/In-N-On/
>
> **摘要:** Egocentric videos are a valuable and scalable data source to learn manipulation policies. However, due to significant data heterogeneity, most existing approaches utilize human data for simple pre-training, which does not unlock its full potential. This paper first provides a scalable recipe for collecting and using egocentric data by categorizing human data into two categories: in-the-wild and on-task alongside with systematic analysis on how to use the data. We first curate a dataset, PHSD, which contains over 1,000 hours of diverse in-the-wild egocentric data and over 20 hours of on-task data directly aligned to the target manipulation tasks. This enables learning a large egocentric language-conditioned flow matching policy, Human0. With domain adaptation techniques, Human0 minimizes the gap between humans and humanoids. Empirically, we show Human0 achieves several novel properties from scaling human data, including language following of instructions from only human data, few-shot learning, and improved robustness using on-task data. Project website: https://xiongyicai.github.io/In-N-On/
>
---
#### [new 108] Knowledge Graphs as Structured Memory for Embedding Spaces: From Training Clusters to Explainable Inference
- **分类: cs.LG; cs.CV**

- **简介: 论文提出Graph Memory（GM），一种基于知识图谱的非参数框架，用于增强嵌入空间中的推理与解释。它通过原型节点和关系边构建结构化记忆，统一实例检索、原型推理与标签传播，解决传统方法在样本效率、校准性和决策边界平滑性上的不足。**

- **链接: [https://arxiv.org/pdf/2511.14961v1](https://arxiv.org/pdf/2511.14961v1)**

> **作者:** Artur A. Oliveira; Mateus Espadoto; Roberto M. Cesar; Roberto Hirata
>
> **备注:** Submitted to GRIVAPP 2026 (21st International Conference on Computer Graphics, Interaction, Visualization Theory and Applications), Marbella, Spain, March 9-11 2026
>
> **摘要:** We introduce Graph Memory (GM), a structured non-parametric framework that augments embedding-based inference with a compact, relational memory over region-level prototypes. Rather than treating each training instance in isolation, GM summarizes the embedding space into prototype nodes annotated with reliability indicators and connected by edges that encode geometric and contextual relations. This design unifies instance retrieval, prototype-based reasoning, and graph-based label propagation within a single inductive model that supports both efficient inference and faithful explanation. Experiments on synthetic and real datasets including breast histopathology (IDC) show that GM achieves accuracy competitive with $k$NN and Label Spreading while offering substantially better calibration and smoother decision boundaries, all with an order of magnitude fewer samples. By explicitly modeling reliability and relational structure, GM provides a principled bridge between local evidence and global consistency in non-parametric learning.
>
---
#### [new 109] Dynamic Nested Hierarchies: Pioneering Self-Evolution in Machine Learning Architectures for Lifelong Intelligence
- **分类: cs.LG; cs.CV**

- **简介: 论文提出动态嵌套层次结构，解决静态模型在非平稳环境中难以持续学习的问题。通过自适应调整优化层级与更新频率，实现模型自我演化，提升长期学习能力与环境适应性。**

- **链接: [https://arxiv.org/pdf/2511.14823v1](https://arxiv.org/pdf/2511.14823v1)**

> **作者:** Akbar Anbar Jafari; Cagri Ozcinar; Gholamreza Anbarjafari
>
> **备注:** 12 pages, 1 figure
>
> **摘要:** Contemporary machine learning models, including large language models, exhibit remarkable capabilities in static tasks yet falter in non-stationary environments due to rigid architectures that hinder continual adaptation and lifelong learning. Building upon the nested learning paradigm, which decomposes models into multi-level optimization problems with fixed update frequencies, this work proposes dynamic nested hierarchies as the next evolutionary step in advancing artificial intelligence and machine learning. Dynamic nested hierarchies empower models to autonomously adjust the number of optimization levels, their nesting structures, and update frequencies during training or inference, inspired by neuroplasticity to enable self-evolution without predefined constraints. This innovation addresses the anterograde amnesia in existing models, facilitating true lifelong learning by dynamically compressing context flows and adapting to distribution shifts. Through rigorous mathematical formulations, theoretical proofs of convergence, expressivity bounds, and sublinear regret in varying regimes, alongside empirical demonstrations of superior performance in language modeling, continual learning, and long-context reasoning, dynamic nested hierarchies establish a foundational advancement toward adaptive, general-purpose intelligence.
>
---
#### [new 110] A Novel CustNetGC Boosted Model with Spectral Features for Parkinson's Disease Prediction
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于医学诊断分类任务，旨在提高帕金森病（PD）的早期预测准确性。通过提取语音信号的光谱特征（L-mHP和频谱斜率），结合CNN与CatBoost的CustNetGC模型，在公开数据集上实现99.06%准确率，提升诊断效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.15485v1](https://arxiv.org/pdf/2511.15485v1)**

> **作者:** Abishek Karthik; Pandiyaraju V; Dominic Savio M; Rohit Swaminathan S
>
> **摘要:** Parkinson's disease is a neurodegenerative disorder that can be very tricky to diagnose and treat. Such early symptoms can include tremors, wheezy breathing, and changes in voice quality as critical indicators of neural damage. Notably, there has been growing interest in utilizing changes in vocal attributes as markers for the detection of PD early on. Based on this understanding, the present paper was designed to focus on the acoustic feature analysis based on voice recordings of patients diagnosed with PD and healthy controls (HC). In this paper, we introduce a novel classification and visualization model known as CustNetGC, combining a Convolutional Neural Network (CNN) with Custom Network Grad-CAM and CatBoost to enhance the efficiency of PD diagnosis. We use a publicly available dataset from Figshare, including voice recordings of 81 participants: 40 patients with PD and 41 healthy controls. From these recordings, we extracted the key spectral features: L-mHP and Spectral Slopes. The L-mHP feature combines three spectrogram representations: Log-Mel spectrogram, harmonic spectrogram, and percussive spectrogram, which are derived using Harmonic-Percussive Source Separation (HPSS). Grad-CAM was used to highlight the important regions in the data, thus making the PD predictions interpretable and effective. Our proposed CustNetGC model achieved an accuracy of 99.06% and precision of 95.83%, with the area under the ROC curve (AUC) recorded at 0.90 for the PD class and 0.89 for the HC class. Additionally, the combination of CatBoost, a gradient boosting algorithm, enhanced the robustness and the prediction performance by properly classifying PD and non-PD samples. Therefore, the results provide the potential improvement in the CustNetGC system in enhancing diagnostic accuracy and the interpretability of the Parkinson's Disease prediction model.
>
---
#### [new 111] Data-driven Prediction of Species-Specific Plant Responses to Spectral-Shifting Films from Leaf Phenotypic and Photosynthetic Traits
- **分类: q-bio.QM; cs.CV; cs.LG; eess.IV**

- **简介: 该论文属于预测任务，旨在解决光谱转换膜（SF）对不同作物产量影响难以预测的问题。通过收集多类植物性状数据并使用变分自编码器增强数据，训练了多种机器学习模型，其中前馈神经网络准确率达91.4%，可有效预测SF对作物产量的显著影响。**

- **链接: [https://arxiv.org/pdf/2511.15173v1](https://arxiv.org/pdf/2511.15173v1)**

> **作者:** Jun Hyeun Kang; Jung Eek Son; Tae In Ahn
>
> **摘要:** The application of spectral-shifting films in greenhouses to shift green light to red light has shown variable growth responses across crop species. However, the yield enhancement of crops under altered light quality is related to the collective effects of the specific biophysical characteristics of each species. Considering only one attribute of a crop has limitations in understanding the relationship between sunlight quality adjustments and crop growth performance. Therefore, this study aims to comprehensively link multiple plant phenotypic traits and daily light integral considering the physiological responses of crops to their growth outcomes under SF using artificial intelligence. Between 2021 and 2024, various leafy, fruiting, and root crops were grown in greenhouses covered with either PEF or SF, and leaf reflectance, leaf mass per area, chlorophyll content, daily light integral, and light saturation point were measured from the plants cultivated in each condition. 210 data points were collected, but there was insufficient data to train deep learning models, so a variational autoencoder was used for data augmentation. Most crop yields showed an average increase of 22.5% under SF. These data were used to train several models, including logistic regression, decision tree, random forest, XGBoost, and feedforward neural network (FFNN), aiming to binary classify whether there was a significant effect on yield with SF application. The FFNN achieved a high classification accuracy of 91.4% on a test dataset that was not used for training. This study provide insight into the complex interactions between leaf phenotypic and photosynthetic traits, environmental conditions, and solar spectral components by improving the ability to predict solar spectral shift effects using SF.
>
---
#### [new 112] Context Cascade Compression: Exploring the Upper Limits of Text Compression
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出Context Cascade Compression（C3），用于长文本压缩任务，解决LLM处理长上下文时的计算与内存挑战。通过两级LLM架构实现高比例压缩（最高40倍），在保持高解码准确率（93%）的同时，验证了纯文本压缩的可行性与上限。**

- **链接: [https://arxiv.org/pdf/2511.15244v1](https://arxiv.org/pdf/2511.15244v1)**

> **作者:** Fanfan Liu; Haibo Qiu
>
> **摘要:** Million-level token inputs in long-context tasks pose significant computational and memory challenges for Large Language Models (LLMs). Recently, DeepSeek-OCR conducted research into the feasibility of Contexts Optical Compression and achieved preliminary results. Inspired by this, we introduce Context Cascade Compression C3 to explore the upper limits of text compression. Our method cascades two LLMs of different sizes to handle the compression and decoding tasks. Specifically, a small LLM, acting as the first stage, performs text compression by condensing a long context into a set of latent tokens (e.g., 32 or 64 in length), achieving a high ratio of text tokens to latent tokens. A large LLM, as the second stage, then executes the decoding task on this compressed context. Experiments show that at a 20x compression ratio (where the number of text tokens is 20 times the number of latent tokens), our model achieves 98% decoding accuracy, compared to approximately 60% for DeepSeek-OCR. When we further increase the compression ratio to 40x, the accuracy is maintained at around 93%. This indicates that in the domain of context compression, C3 Compression demonstrates superior performance and feasibility over optical character compression. C3 uses a simpler, pure-text pipeline that ignores factors like layout, color, and information loss from a visual encoder. This also suggests a potential upper bound for compression ratios in future work on optical character compression, OCR, and related fields. Codes and model weights are publicly accessible at https://github.com/liufanfanlff/C3-Context-Cascade-Compression
>
---
#### [new 113] Multimodal Evaluation of Russian-language Architectures
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Mera Multi，一个面向俄语的多模态评估框架，解决俄语领域缺乏多模态基准的问题。工作包括构建18个新任务、统一评测标准、提供基线结果及防泄露方法，为多模态模型评估提供可复用方案。**

- **链接: [https://arxiv.org/pdf/2511.15552v1](https://arxiv.org/pdf/2511.15552v1)**

> **作者:** Artem Chervyakov; Ulyana Isaeva; Anton Emelyanov; Artem Safin; Maria Tikhonova; Alexander Kharitonov; Yulia Lyakh; Petr Surovtsev; Denis Shevelev Vildan Saburov; Vasily Konovalov; Elisei Rykov; Ivan Sviridov; Amina Miftakhova; Ilseyar Alimova; Alexander Panchenko; Alexander Kapitanov; Alena Fenogenova
>
> **摘要:** Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce Mera Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (image-to-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking and licenses for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family.
>
---
#### [new 114] Look, Zoom, Understand: The Robotic Eyeball for Embodied Perception
- **分类: cs.RO; cs.CV**

- **简介: 论文提出EyeVLA，一种用于具身感知的主动视觉系统，解决固定摄像头难以兼顾广域覆盖与细节获取的问题。通过将动作离散化为动作令牌并融合视觉语言模型，实现指令驱动的旋转与缩放，提升环境感知能力。**

- **链接: [https://arxiv.org/pdf/2511.15279v1](https://arxiv.org/pdf/2511.15279v1)**

> **作者:** Jiashu Yang; Yifan Han; Yucheng Xie; Ning Guo; Wenzhao Lian
>
> **摘要:** In embodied AI perception systems, visual perception should be active: the goal is not to passively process static images, but to actively acquire more informative data within pixel and spatial budget constraints. Existing vision models and fixed RGB-D camera systems fundamentally fail to reconcile wide-area coverage with fine-grained detail acquisition, severely limiting their efficacy in open-world robotic applications. To address this issue, we propose EyeVLA, a robotic eyeball for active visual perception that can take proactive actions based on instructions, enabling clear observation of fine-grained target objects and detailed information across a wide spatial extent. EyeVLA discretizes action behaviors into action tokens and integrates them with vision-language models (VLMs) that possess strong open-world understanding capabilities, enabling joint modeling of vision, language, and actions within a single autoregressive sequence. By using the 2D bounding box coordinates to guide the reasoning chain and applying reinforcement learning to refine the viewpoint selection policy, we transfer the open-world scene understanding capability of the VLM to a vision language action (VLA) policy using only minimal real-world data. Experiments show that our system efficiently performs instructed scenes in real-world environments and actively acquires more accurate visual information through instruction-driven actions of rotation and zoom, thereby achieving strong environmental perception capabilities. EyeVLA introduces a novel robotic vision system that leverages detailed and spatially rich, large-scale embodied data, and actively acquires highly informative visual observations for downstream embodied tasks.
>
---
#### [new 115] Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration
- **分类: cs.AI; cs.CV**

- **简介: 论文提出Octopus框架，解决多模态推理模型缺乏自主探索和动态能力选择的问题。通过六种核心能力的协同调度，实现更灵活的多模态智能体推理，在自建基准上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.15351v1](https://arxiv.org/pdf/2511.15351v1)**

> **作者:** Yifu Guo; Zishan Xu; Zhiyuan Yao; Yuquan Lu; Jiaye Lin; Sen Hu; Zhenheng Tang; Yingchao Li; Huacan Wang; Ronghao Chen
>
> **摘要:** Existing multimodal reasoning models and frameworks suffer from fundamental architectural limitations: most lack the human-like ability to autonomously explore diverse reasoning pathways-whether in direct inference, tool-driven visual exploration, programmatic visual manipulation, or intrinsic visual imagination. Consequently, they struggle to adapt to dynamically changing capability requirements in real-world tasks. Meanwhile, humans exhibit a complementary set of thinking abilities when addressing such tasks, whereas existing methods typically cover only a subset of these dimensions. Inspired by this, we propose Octopus: Agentic Multimodal Reasoning with Six-Capability Orchestration, a new paradigm for multimodal agentic reasoning. We define six core capabilities essential for multimodal reasoning and organize a comprehensive evaluation benchmark, Octopus-Bench, accordingly. Octopus is capable of autonomously exploring during reasoning and dynamically selecting the most appropriate capability based on the current state. Experimental results show that Octopus achieves the best performance on the vast majority of tasks in Octopus-Bench, highlighting the crucial role of capability coordination in agentic multimodal reasoning.
>
---
#### [new 116] BBox DocVQA: A Large Scale Bounding Box Grounded Dataset for Enhancing Reasoning in Document Visual Question Answer
- **分类: cs.DB; cs.AI; cs.CV**

- **简介: 论文提出BBox DocVQA，一个大规模边界框标注的文档视觉问答数据集，旨在提升视觉语言模型的空间推理与证据定位能力。通过自动化构建流程和人工验证，解决现有数据集缺乏细粒度空间接地的问题，显著增强模型的空间理解与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.15090v1](https://arxiv.org/pdf/2511.15090v1)**

> **作者:** Wenhan Yu; Wang Chen; Guanqiang Qi; Weikang Li; Yang Li; Lei Sha; Deguo Xia; Jizhou Huang
>
> **备注:** 22 pages, 4 figures
>
> **摘要:** Document Visual Question Answering (DocVQA) is a fundamental task for multimodal document understanding and a key testbed for vision language reasoning. However, most existing DocVQA datasets are limited to the page level and lack fine grained spatial grounding, constraining the interpretability and reasoning capability of Vision Language Models (VLMs). To address this gap, we introduce BBox DocVQA a large scale, bounding box grounded dataset designed to enhance spatial reasoning and evidence localization in visual documents. We further present an automated construction pipeline, Segment Judge and Generate, which integrates a segment model for region segmentation, a VLM for semantic judgment, and another advanced VLM for question answer generation, followed by human verification for quality assurance. The resulting dataset contains 3.6 K diverse documents and 32 K QA pairs, encompassing single and multi region as well as single and multi page scenarios. Each QA instance is grounded on explicit bounding boxes, enabling fine grained evaluation of spatial semantic alignment. Benchmarking multiple state of the art VLMs (e.g., GPT 5, Qwen2.5 VL, and InternVL) on BBox DocVQA reveals persistent challenges in spatial grounding and reasoning accuracy. Furthermore, fine tuning on BBox DocVQA substantially improves both bounding box localization and answer generation, validating its effectiveness for enhancing the reasoning ability of VLMs. Our dataset and code will be publicly released to advance research on interpretable and spatially grounded vision language reasoning.
>
---
#### [new 117] SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出SRPO框架，用于视觉-语言-动作模型的强化学习训练。针对专家示范依赖和奖励稀疏问题，利用模型自身成功轨迹作为自参考，通过世界模型潜空间表示衡量行为进展，实现高效、无监督的策略优化，在LIBERO基准上显著提升成功率。**

- **链接: [https://arxiv.org/pdf/2511.15605v1](https://arxiv.org/pdf/2511.15605v1)**

> **作者:** Senyu Fei; Siyin Wang; Li Ji; Ao Li; Shiduo Zhang; Liming Liu; Jinlong Hou; Jingjing Gong; Xianzhong Zhao; Xipeng Qiu
>
> **摘要:** Vision-Language-Action (VLA) models excel in robotic manipulation but are constrained by their heavy reliance on expert demonstrations, leading to demonstration bias and limiting performance. Reinforcement learning (RL) is a vital post-training strategy to overcome these limits, yet current VLA-RL methods, including group-based optimization approaches, are crippled by severe reward sparsity. Relying on binary success indicators wastes valuable information in failed trajectories, resulting in low training efficiency. To solve this, we propose Self-Referential Policy Optimization (SRPO), a novel VLA-RL framework. SRPO eliminates the need for external demonstrations or manual reward engineering by leveraging the model's own successful trajectories, generated within the current training batch, as a self-reference. This allows us to assign a progress-wise reward to failed attempts. A core innovation is the use of latent world representations to measure behavioral progress robustly. Instead of relying on raw pixels or requiring domain-specific fine-tuning, we utilize the compressed, transferable encodings from a world model's latent space. These representations naturally capture progress patterns across environments, enabling accurate, generalized trajectory comparison. Empirical evaluations on the LIBERO benchmark demonstrate SRPO's efficiency and effectiveness. Starting from a supervised baseline with 48.9% success, SRPO achieves a new state-of-the-art success rate of 99.2% in just 200 RL steps, representing a 103% relative improvement without any extra supervision. Furthermore, SRPO shows substantial robustness, achieving a 167% performance improvement on the LIBERO-Plus benchmark.
>
---
#### [new 118] IPR-1: Interactive Physical Reasoner
- **分类: cs.AI; cs.CV**

- **简介: 论文提出IPR-1，一种通过交互学习物理推理的模型。针对现有方法在互动中缺乏前瞻性和物理理解的问题，该工作结合世界模型与视觉语言模型，引入物理导向的动作编码，在多样游戏中实现从生存到好奇心的多层推理，并展示持续改进和零样本迁移能力。**

- **链接: [https://arxiv.org/pdf/2511.15407v1](https://arxiv.org/pdf/2511.15407v1)**

> **作者:** Mingyu Zhang; Lifeng Zhuo; Tianxi Tan; Guocan Xie; Xian Nie; Yan Li; Renjie Zhao; Zizhu He; Ziyu Wang; Jiting Cai; Yong-Lu Li
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Humans learn by observing, interacting with environments, and internalizing physics and causality. Here, we aim to ask whether an agent can similarly acquire human-like reasoning from interaction and keep improving with more experience. We study this in a Game-to-Unseen (G2U) setting, curating 1,000+ heterogeneous games with diverse physical and causal mechanisms, and evaluate at three human-like levels: Survival, Curiosity, Utility, from primitive intuition to goal-driven reasoning. Our analysis reveals complementary failures: VLM/VLA agents reason but lack look-ahead in interactive settings, while world models imagine but imitate visual patterns rather than analyze physics and causality. We therefore propose IPR (Interactive Physical Reasoner), using world-model rollouts to score and reinforce a VLM's policy, and introduce PhysCode, a physics-centric action code aligning semantic intent with dynamics to provide a shared action space for prediction and reasoning. Pretrained on 1,000+ games, our IPR performs robustly on three levels, matches GPT-5 overall, and surpasses it on Curiosity. We find that performance improves with more training games and interaction steps, and that the model also zero-shot transfers to unseen games. These results support physics-centric interaction as a path to steadily improving physical reasoning.
>
---
#### [new 119] MHR: Momentum Human Rig
- **分类: cs.GR; cs.CV**

- **简介: 论文提出MHR，一种结合ATLAS骨架/形状解耦与Momentum库灵活绑定系统的参数化人体模型，用于实现更自然的人体动画，解决传统模型在AR/VR中表达力不足和姿势校正不灵活的问题。**

- **链接: [https://arxiv.org/pdf/2511.15586v1](https://arxiv.org/pdf/2511.15586v1)**

> **作者:** Aaron Ferguson; Ahmed A. A. Osman; Berta Bescos; Carsten Stoll; Chris Twigg; Christoph Lassner; David Otte; Eric Vignola; Federica Bogo; Igor Santesteban; Javier Romero; Jenna Zarate; Jeongseok Lee; Jinhyung Park; Jinlong Yang; John Doublestein; Kishore Venkateshan; Kris Kitani; Ladislav Kavan; Marco Dal Farra; Matthew Hu; Matthew Cioffi; Michael Fabris; Michael Ranieri; Mohammad Modarres; Petr Kadlecek; Rinat Abdrashitov; Romain Prévost; Roman Rajbhandari; Ronald Mallet; Russel Pearsall; Sandy Kao; Sanjeev Kumar; Scott Parrish; Te-Li Wang; Tony Tung; Yuan Dong; Yuhua Chen; Yuanlu Xu; Yuting Ye; Zhongshi Jiang
>
> **摘要:** We present MHR, a parametric human body model that combines the decoupled skeleton/shape paradigm of ATLAS with a flexible, modern rig and pose corrective system inspired by the Momentum library. Our model enables expressive, anatomically plausible human animation, supporting non-linear pose correctives, and is designed for robust integration in AR/VR and graphics pipelines.
>
---
#### [new 120] Image Denoising Using Transformed L1 (TL1) Regularization via ADMM
- **分类: eess.IV; cs.CV; math.OC**

- **简介: 论文提出基于TL1正则化的图像去噪方法，解决传统TV模型产生的阶梯伪影和对比度损失问题。通过ADMM算法优化，结合闭式近似算子与FFT加速，有效去噪并保边增强对比度。**

- **链接: [https://arxiv.org/pdf/2511.15060v1](https://arxiv.org/pdf/2511.15060v1)**

> **作者:** Nabiha Choudhury; Jianqing Jia; Yifei Lou
>
> **摘要:** Total variation (TV) regularization is a classical tool for image denoising, but its convex $\ell_1$ formulation often leads to staircase artifacts and loss of contrast. To address these issues, we introduce the Transformed $\ell_1$ (TL1) regularizer applied to image gradients. In particular, we develop a TL1-regularized denoising model and solve it using the Alternating Direction Method of Multipliers (ADMM), featuring a closed-form TL1 proximal operator and an FFT-based image update under periodic boundary conditions. Experimental results demonstrate that our approach achieves superior denoising performance, effectively suppressing noise while preserving edges and enhancing image contrast.
>
---
#### [new 121] Attacking Autonomous Driving Agents with Adversarial Machine Learning: A Holistic Evaluation with the CARLA Leaderboard
- **分类: cs.CR; cs.CV; cs.LG; cs.RO**

- **简介: 论文研究对抗性攻击对自动驾驶系统的影响，旨在评估攻击是否能引发有害驾驶行为。作者在CARLA模拟器中测试针对多个自动驾驶代理的对抗补丁攻击，发现部分代理因控制器逻辑可抵消模型误判，从而提升安全性。**

- **链接: [https://arxiv.org/pdf/2511.14876v1](https://arxiv.org/pdf/2511.14876v1)**

> **作者:** Henry Wong; Clement Fung; Weiran Lin; Karen Li; Stanley Chen; Lujo Bauer
>
> **备注:** 12 pages
>
> **摘要:** To autonomously control vehicles, driving agents use outputs from a combination of machine-learning (ML) models, controller logic, and custom modules. Although numerous prior works have shown that adversarial examples can mislead ML models used in autonomous driving contexts, it remains unclear if these attacks are effective at producing harmful driving actions for various agents, environments, and scenarios. To assess the risk of adversarial examples to autonomous driving, we evaluate attacks against a variety of driving agents, rather than against ML models in isolation. To support this evaluation, we leverage CARLA, an urban driving simulator, to create and evaluate adversarial examples. We create adversarial patches designed to stop or steer driving agents, stream them into the CARLA simulator at runtime, and evaluate them against agents from the CARLA Leaderboard, a public repository of best-performing autonomous driving agents from an annual research competition. Unlike prior work, we evaluate attacks against autonomous driving systems without creating or modifying any driving-agent code and against all parts of the agent included with the ML model. We perform a case-study investigation of two attack strategies against three open-source driving agents from the CARLA Leaderboard across multiple driving scenarios, lighting conditions, and locations. Interestingly, we show that, although some attacks can successfully mislead ML models into predicting erroneous stopping or steering commands, some driving agents use modules, such as PID control or GPS-based rules, that can overrule attacker-manipulated predictions from ML models.
>
---
## 更新

#### [replaced 001] CompAgent: An Agentic Framework for Visual Compliance Verification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00171v2](https://arxiv.org/pdf/2511.00171v2)**

> **作者:** Rahul Ghosh; Baishali Chaudhury; Hari Prasanna Das; Meghana Ashok; Ryan Razkenari; Sungmin Hong; Chun-Hao Liu
>
> **备注:** Under review
>
> **摘要:** Visual compliance verification is a critical yet underexplored problem in computer vision, especially in domains such as media, entertainment, and advertising where content must adhere to complex and evolving policy rules. Existing methods often rely on task-specific deep learning models trained on manually labeled datasets, which are costly to build and limited in generalizability. While recent Multimodal Large Language Models (MLLMs) offer broad real-world knowledge and policy understanding, they struggle to reason over fine-grained visual details and apply structured compliance rules effectively on their own. In this paper, we propose CompAgent, the first agentic framework for visual compliance verification. CompAgent augments MLLMs with a suite of visual tools-such as object detectors, face analyzers, NSFW detectors, and captioning models-and introduces a planning agent that dynamically selects appropriate tools based on the compliance policy. A compliance verification agent then integrates image, tool outputs, and policy context to perform multimodal reasoning. Experiments on public benchmarks show that CompAgent outperforms specialized classifiers, direct MLLM prompting, and curated routing baselines, achieving up to 76% F1 score and a 10% improvement over the state-of-the-art on the UnsafeBench dataset. Our results demonstrate the effectiveness of agentic planning and robust tool-augmented reasoning for scalable, accurate, and adaptable visual compliance verification.
>
---
#### [replaced 002] DeepContrast: Deep Tissue Contrast Enhancement using Synthetic Data Degradations and OOD Model Predictions
- **分类: eess.IV; cs.CV; q-bio.TO**

- **链接: [https://arxiv.org/pdf/2308.08365v2](https://arxiv.org/pdf/2308.08365v2)**

> **作者:** Nuno Pimpão Martins; Yannis Kalaidzidis; Marino Zerial; Florian Jug
>
> **备注:** 8 pages, 7 figures, 1 table
>
> **摘要:** Microscopy images are crucial for life science research, allowing detailed inspection and characterization of cellular and tissue-level structures and functions. However, microscopy data are unavoidably affected by image degradations, such as noise, blur, or others. Many such degradations also contribute to a loss of image contrast, which becomes especially pronounced in deeper regions of thick samples. Today, best performing methods to increase the quality of images are based on Deep Learning approaches, which typically require ground truth (GT) data during training. Our inability to counteract blurring and contrast loss when imaging deep into samples prevents the acquisition of such clean GT data. The fact that the forward process of blurring and contrast loss deep into tissue can be modeled, allowed us to propose a new method that can circumvent the problem of unobtainable GT data. To this end, we first synthetically degraded the quality of microscopy images even further by using an approximate forward model for deep tissue image degradations. Then we trained a neural network that learned the inverse of this degradation function from our generated pairs of raw and degraded images. We demonstrated that networks trained in this way can be used out-of-distribution (OOD) to improve the quality of less severely degraded images, e.g. the raw data imaged in a microscope. Since the absolute level of degradation in such microscopy images can be stronger than the additional degradation introduced by our forward model, we also explored the effect of iterative predictions. Here, we observed that in each iteration the measured image contrast kept improving while detailed structures in the images got increasingly removed. Therefore, dependent on the desired downstream analysis, a balance between contrast improvement and retention of image details has to be found.
>
---
#### [replaced 003] Integration of nested cross-validation, automated hyperparameter optimization, high-performance computing to reduce and quantify the variance of test performance estimation of deep learning models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08589v2](https://arxiv.org/pdf/2503.08589v2)**

> **作者:** Paul Calle; Averi Bates; Justin C. Reynolds; Yunlong Liu; Haoyang Cui; Sinaro Ly; Chen Wang; Qinghao Zhang; Alberto J. de Armendi; Shashank S. Shettar; Kar Ming Fung; Qinggong Tang; Chongle Pan
>
> **摘要:** Background and Objectives: The variability and biases in the real-world performance benchmarking of deep learning models for medical imaging compromise their trustworthiness for real-world deployment. The common approach of holding out a single fixed test set fails to quantify the variance in the estimation of test performance metrics. This study introduces NACHOS (Nested and Automated Cross-validation and Hyperparameter Optimization using Supercomputing) to reduce and quantify the variance of test performance metrics of deep learning models. Methods: NACHOS integrates Nested Cross-Validation (NCV) and Automated Hyperparameter Optimization (AHPO) within a parallelized high-performance computing (HPC) framework. NACHOS was demonstrated on a chest X-ray repository and an Optical Coherence Tomography (OCT) dataset under multiple data partitioning schemes. Beyond performance estimation, DACHOS (Deployment with Automated Cross-validation and Hyperparameter Optimization using Supercomputing) is introduced to leverage AHPO and cross-validation to build the final model on the full dataset, improving expected deployment performance. Results: The findings underscore the importance of NCV in quantifying and reducing estimation variance, AHPO in optimizing hyperparameters consistently across test folds, and HPC in ensuring computational feasibility. Conclusions: By integrating these methodologies, NACHOS and DACHOS provide a scalable, reproducible, and trustworthy framework for DL model evaluation and deployment in medical imaging. To maximize public availability, the full open-source codebase is provided at https://github.com/thepanlab/NACHOS
>
---
#### [replaced 004] Beyond Diagnosis: Evaluating Multimodal LLMs for Pathology Localization in Chest Radiographs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.18015v2](https://arxiv.org/pdf/2509.18015v2)**

> **作者:** Advait Gosai; Arun Kavishwar; Stephanie L. McNamara; Soujanya Samineni; Renato Umeton; Alexander Chowdhury; William Lotter
>
> **备注:** Proceedings of the 5th Machine Learning for Health (ML4H) Symposium
>
> **摘要:** Recent work has shown promising performance of frontier large language models (LLMs) and their multimodal counterparts in medical quizzes and diagnostic tasks, highlighting their potential for broad clinical utility given their accessible, general-purpose nature. However, beyond diagnosis, a fundamental aspect of medical image interpretation is the ability to localize pathological findings. Evaluating localization not only has clinical and educational relevance but also provides insight into a model's spatial understanding of anatomy and disease. Here, we systematically assess two general-purpose MLLMs (GPT-4 and GPT-5) and a domain-specific model (MedGemma) in their ability to localize pathologies on chest radiographs, using a prompting pipeline that overlays a spatial grid and elicits coordinate-based predictions. Averaged across nine pathologies in the CheXlocalize dataset, GPT-5 exhibited a localization accuracy of 49.7%, followed by GPT-4 (39.1%) and MedGemma (17.7%), all lower than a task-specific CNN baseline (59.9%) and a radiologist benchmark (80.1%). Despite modest performance, error analysis revealed that GPT-5's predictions were largely in anatomically plausible regions, just not always precisely localized. GPT-4 performed well on pathologies with fixed anatomical locations, but struggled with spatially variable findings and exhibited anatomically implausible predictions more frequently. MedGemma demonstrated the lowest performance on all pathologies, but showed improvements when provided examples through few shot prompting. Our findings highlight both the promise and limitations of current MLLMs in medical imaging and underscore the importance of integrating them with task-specific tools for reliable use.
>
---
#### [replaced 005] Interpretable Retinal Disease Prediction Using Biology-Informed Heterogeneous Graph Representations
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2502.16697v2](https://arxiv.org/pdf/2502.16697v2)**

> **作者:** Laurin Lux; Alexander H. Berger; Maria Romeo Tricas; Richard Rosen; Alaa E. Fayed; Sobha Sivaprasada; Linus Kreitner; Jonas Weidner; Martin J. Menten; Daniel Rueckert; Johannes C. Paetzold
>
> **摘要:** Interpretability is crucial to enhance trust in machine learning models for medical diagnostics. However, most state-of-the-art image classifiers based on neural networks are not interpretable. As a result, clinicians often resort to known biomarkers for diagnosis, although biomarker-based classification typically performs worse than large neural networks. This work proposes a method that surpasses the performance of established machine learning models while simultaneously improving prediction interpretability for diabetic retinopathy staging from optical coherence tomography angiography (OCTA) images. Our method is based on a novel biology-informed heterogeneous graph representation that models retinal vessel segments, intercapillary areas, and the foveal avascular zone (FAZ) in a human-interpretable way. This graph representation allows us to frame diabetic retinopathy staging as a graph-level classification task, which we solve using an efficient graph neural network. We benchmark our method against well-established baselines, including classical biomarker-based classifiers, convolutional neural networks (CNNs), and vision transformers. Our model outperforms all baselines on two datasets. Crucially, we use our biology-informed graph to provide explanations of unprecedented detail. Our approach surpasses existing methods in precisely localizing and identifying critical vessels or intercapillary areas. In addition, we give informative and human-interpretable attributions to critical characteristics. Our work contributes to the development of clinical decision-support tools in ophthalmology.
>
---
#### [replaced 006] Underage Detection through a Multi-Task and MultiAge Approach for Screening Minors in Unconstrained Imagery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.10689v2](https://arxiv.org/pdf/2506.10689v2)**

> **作者:** Christopher Gaul; Eduardo Fidalgo; Enrique Alegre; Rocío Alaiz Rodríguez; Eri Pérez Corral
>
> **摘要:** Accurate automatic screening of minors in unconstrained images requires models robust to distribution shift and resilient to the under-representation of children in public datasets. To address these issues, we propose a multi-task architecture with dedicated under/over-age discrimination tasks based on a frozen FaRL vision-language backbone joined with a compact two-layer MLP that shares features across one age-regression head and four binary underage heads (12, 15, 18, and 21 years). This design focuses on the legally critical age range while keeping the backbone frozen. Class imbalance is mitigated through an $α$-reweighted focal loss and age-balanced mini-batch sampling, while an age gap removes ambiguous samples near thresholds. Evaluation is conducted on our new Overall Underage Benchmark (303k cleaned training images, 110k test images), defining both the "ASORES-39k" restricted overall test, which removes the noisiest domains, and the age estimation wild-shifts test "ASWIFT-20k" of 20k-images, stressing extreme poses ($>$45°), expressions, and low image quality to emulate real-world shifts. Trained on the cleaned overall set with resampling and age gap, our multiage model "F" reduces the mean absolute error on ASORES-39k from 4.175 y (age-only baseline) to 4.068 y and improves under-18 detection from F2 score of 0.801 to 0.857 at 1% false-adult rate. Under the ASWIFT-20k, the same configuration nearly sustains 0.99 recall while F2 rises from 0.742 to 0.833, demonstrating robustness to domain shift.
>
---
#### [replaced 007] Gene-DML: Dual-Pathway Multi-Level Discrimination for Gene Expression Prediction from Histopathology Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.14670v2](https://arxiv.org/pdf/2507.14670v2)**

> **作者:** Yaxuan Song; Jianan Fan; Hang Chang; Weidong Cai
>
> **备注:** Accepted by The IEEE/CVF Winter Conference on Applications of Computer Vision 2026 (WACV2026). Code and data available at https://github.com/YXSong000/Gene-DML
>
> **摘要:** Accurately predicting gene expression from histopathology images offers a scalable and non-invasive approach to molecular profiling, with significant implications for precision medicine and computational pathology. However, existing methods often underutilize the cross-modal representation alignment between histopathology images and gene expression profiles across multiple representational levels, thereby limiting their prediction performance. To address this, we propose Gene-DML, a unified framework that structures latent space through Dual-pathway Multi-Level discrimination to enhance correspondence between morphological and transcriptional modalities. The multi-scale instance-level discrimination pathway aligns hierarchical histopathology representations extracted at local, neighbor, and global levels with gene expression profiles, capturing scale-aware morphological-transcriptional relationships. In parallel, the cross-level instance-group discrimination pathway enforces structural consistency between individual (image/gene) instances and modality-crossed (gene/image, respectively) groups, strengthening the alignment across modalities. By jointly modeling fine-grained and structural-level discrimination, Gene-DML is able to learn robust cross-modal representations, enhancing both predictive accuracy and generalization across diverse biological contexts. Extensive experiments on public spatial transcriptomics datasets demonstrate that Gene-DML achieves state-of-the-art performance in gene expression prediction. The code and processed datasets are available at https://github.com/YXSong000/Gene-DML.
>
---
#### [replaced 008] What Color Is It? A Text-Interference Multimodal Hallucination Benchmark
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13400v2](https://arxiv.org/pdf/2511.13400v2)**

> **作者:** Jinkun Zhao; Lei Huang; Haixin Ge; Wenjun Wu
>
> **摘要:** With the rapid advancement of Large Models, numerous text-and-vision-fused Multimodal Large Models (MLMs) have emerged. However, these MLMs remain susceptible to informational interference in visual perception, particularly in color perception, which introduces an additional risk of hallucination. To validate this hypothesis, we introduce the "What Color Is It" dataset, a novel benchmark constructed using a simple method to trigger single-modality visual hallucination in MLMs. Based on this dataset, we further investigate the underlying causes of hallucination in the visual modality of MLMs and propose potential solutions to enhance their robustness.
>
---
#### [replaced 009] TrackStudio: An Integrated Toolkit for Markerless Tracking
- **分类: cs.CV; q-bio.QM**

- **链接: [https://arxiv.org/pdf/2511.07624v2](https://arxiv.org/pdf/2511.07624v2)**

> **作者:** Hristo Dimitrov; Giulia Dominijanni; Viktorija Pavalkyte; Tamar R. Makin
>
> **备注:** 26 pages, 5 main text figures, 5 supplementary figures
>
> **摘要:** Markerless motion tracking has advanced rapidly in the past 10 years and currently offers powerful opportunities for behavioural, clinical, and biomechanical research. While several specialised toolkits provide high performance for specific tasks, using existing tools still requires substantial technical expertise. There remains a gap in accessible, integrated solutions that deliver sufficient tracking for non-experts across diverse settings. TrackStudio was developed to address this gap by combining established open-source tools into a single, modular, GUI-based pipeline that works out of the box. It provides automatic 2D and 3D tracking, calibration, preprocessing, feature extraction, and visualisation without requiring any programming skills. We supply a user guide with practical advice for video acquisition, synchronisation, and setup, alongside documentation of common pitfalls and how to avoid them. To validate the toolkit, we tested its performance across three environments using either low-cost webcams or high-resolution cameras, including challenging conditions for body position, lightning, and space and obstructions. Across 76 participants, average inter-frame correlations exceeded 0.98 and average triangulation errors remained low (<13.6mm for hand tracking), demonstrating stable and consistent tracking. We further show that the same pipeline can be extended beyond hand tracking to other body and face regions. TrackStudio provides a practical, accessible route into markerless tracking for researchers or laypeople who need reliable performance without specialist expertise.
>
---
#### [replaced 010] GeoMVD: Geometry-Enhanced Multi-View Generation Model Based on Geometric Information Extraction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12204v3](https://arxiv.org/pdf/2511.12204v3)**

> **作者:** Jiaqi Wu; Yaosen Chen; Shuyuan Zhu
>
> **摘要:** Multi-view image generation holds significant application value in computer vision, particularly in domains like 3D reconstruction, virtual reality, and augmented reality. Most existing methods, which rely on extending single images, face notable computational challenges in maintaining cross-view consistency and generating high-resolution outputs. To address these issues, we propose the Geometry-guided Multi-View Diffusion Model, which incorporates mechanisms for extracting multi-view geometric information and adjusting the intensity of geometric features to generate images that are both consistent across views and rich in detail. Specifically, we design a multi-view geometry information extraction module that leverages depth maps, normal maps, and foreground segmentation masks to construct a shared geometric structure, ensuring shape and structural consistency across different views. To enhance consistency and detail restoration during generation, we develop a decoupled geometry-enhanced attention mechanism that strengthens feature focus on key geometric details, thereby improving overall image quality and detail preservation. Furthermore, we apply an adaptive learning strategy that fine-tunes the model to better capture spatial relationships and visual coherence between the generated views, ensuring realistic results. Our model also incorporates an iterative refinement process that progressively improves the output quality through multiple stages of image generation. Finally, a dynamic geometry information intensity adjustment mechanism is proposed to adaptively regulate the influence of geometric data, optimizing overall quality while ensuring the naturalness of generated images. More details can be found on the project page: https://sobeymil.github.io/GeoMVD.com.
>
---
#### [replaced 011] PAVE: An End-to-End Dataset for Production Autonomous Vehicle Evaluation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14185v2](https://arxiv.org/pdf/2511.14185v2)**

> **作者:** Xiangyu Li; Chen Wang; Yumao Liu; Dengbo He; Jiahao Zhang; Ke Ma
>
> **摘要:** Most existing autonomous-driving datasets (e.g., KITTI, nuScenes, and the Waymo Perception Dataset), collected by human-driving mode or unidentified driving mode, can only serve as early training for the perception and prediction of autonomous vehicles (AVs). To evaluate the real behavioral safety of AVs controlled in the black box, we present the first end-to-end benchmark dataset collected entirely by autonomous-driving mode in the real world. This dataset contains over 100 hours of naturalistic data from multiple production autonomous-driving vehicle models in the market. We segment the original data into 32,727 key frames, each consisting of four synchronized camera images and high-precision GNSS/IMU data (0.8 cm localization accuracy). For each key frame, 20 Hz vehicle trajectories spanning the past 6 s and future 5 s are provided, along with detailed 2D annotations of surrounding vehicles, pedestrians, traffic lights, and traffic signs. These key frames have rich scenario-level attributes, including driver intent, area type (covering highways, urban roads, and residential areas), lighting (day, night, or dusk), weather (clear or rain), road surface (paved or unpaved), traffic and vulnerable road users (VRU) density, traffic lights, and traffic signs (warning, prohibition, and indication). To evaluate the safety of AVs, we employ an end-to-end motion planning model that predicts vehicle trajectories with an Average Displacement Error (ADE) of 1.4 m on autonomous-driving frames. The dataset continues to expand by over 10 hours of new data weekly, thereby providing a sustainable foundation for research on AV driving behavior analysis and safety evaluation. The PAVE dataset is publicly available at https://hkustgz-my.sharepoint.com/:f:/g/personal/kema_hkust-gz_edu_cn/IgDXyoHKfdGnSZ3JbbidjduMAXxs-Z3NXzm005A_Ix9tr0Q?e=9HReCu.
>
---
#### [replaced 012] Adaptive Multi-Scale Integration Unlocks Robust Cell Annotation in Histopathology Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13586v2](https://arxiv.org/pdf/2511.13586v2)**

> **作者:** Yinuo Xu; Yan Cui; Mingyao Li; Zhi Huang
>
> **摘要:** Identifying cell types and subtypes in routine histopathology is fundamental for understanding disease. Existing tile-based models capture nuclear detail but miss the broader tissue context that influences cell identity. Current human annotations are coarse-grained and uneven across studies, making fine-grained, subtype-level classification difficult. In this study, we build a marker-guided dataset from Xenium spatial transcriptomics with single-cell resolution labels for more than two million cells across eight organs and 16 classes to address the lack of high-quality annotations. Leveraging this data resource, we introduce NuClass, a pathologist workflow inspired framework for cell-wise multi-scale integration of nuclear morphology and microenvironmental context. It combines Path local, which focuses on nuclear morphology from 224x224 pixel crops, and Path global, which models the surrounding 1024x1024 pixel neighborhood, through a learnable gating module that balances local and global information. An uncertainty-guided objective directs the global path to prioritize regions where the local path is uncertain, and we provide calibrated confidence estimates and Grad-CAM maps for interpretability. Evaluated on three fully held-out cohorts, NuClass achieves up to 96 percent F1 for its best-performing class, outperforming strong baselines. Our results demonstrate that multi-scale, uncertainty-aware fusion can bridge the gap between slide-level pathological foundation models and reliable, cell-level phenotype prediction.
>
---
#### [replaced 013] Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.21704v2](https://arxiv.org/pdf/2510.21704v2)**

> **作者:** Christy Li; Josep Lopez Camuñas; Jake Thomas Touchet; Jacob Andreas; Agata Lapedriza; Antonio Torralba; Tamar Rott Shaham
>
> **备注:** 32 pages, 10 figures, Neurips 2025
>
> **摘要:** When a vision model performs image recognition, which visual attributes drive its predictions? Detecting unintended reliance on specific visual features is critical for ensuring model robustness, preventing overfitting, and avoiding spurious correlations. We introduce an automated framework for detecting such dependencies in trained vision models. At the core of our method is a self-reflective agent that systematically generates and tests hypotheses about visual attributes that a model may rely on. This process is iterative: the agent refines its hypotheses based on experimental outcomes and uses a self-evaluation protocol to assess whether its findings accurately explain model behavior. When inconsistencies arise, the agent self-reflects over its findings and triggers a new cycle of experimentation. We evaluate our approach on a novel benchmark of 130 models designed to exhibit diverse visual attribute dependencies across 18 categories. Our results show that the agent's performance consistently improves with self-reflection, with a significant performance increase over non-reflective baselines. We further demonstrate that the agent identifies real-world visual attribute dependencies in state-of-the-art models, including CLIP's vision encoder and the YOLOv8 object detector.
>
---
#### [replaced 014] Unobtrusive Monitoring of Simulated Physical Weakness Using Fine-Grained Behavioral Features and Personalized Modeling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.10045v2](https://arxiv.org/pdf/2406.10045v2)**

> **作者:** Chen Long-fei; Muhammad Ahmed Raza; Craig Innes; Subramanian Ramamoorthy; Robert B. Fisher
>
> **摘要:** Aging and chronic conditions affect older adults' daily lives, making early detection of developing health issues crucial. Weakness, common in many conditions, alters physical movements and daily activities subtly. However, detecting such changes can be challenging due to their subtle and gradual nature. To address this, we employ a non-intrusive camera sensor to monitor individuals' daily sitting and relaxing activities for signs of weakness. We simulate weakness in healthy subjects by having them perform physical exercise and observing the behavioral changes in their daily activities before and after workouts. The proposed system captures fine-grained features related to body motion, inactivity, and environmental context in real-time while prioritizing privacy. A Bayesian Network is used to model the relationships between features, activities, and health conditions. We aim to identify specific features and activities that indicate such changes and determine the most suitable time scale for observing the change. Results show 0.97 accuracy in distinguishing simulated weakness at the daily level. Fine-grained behavioral features, including non-dominant upper body motion speed and scale, and inactivity distribution, along with a 300-second window, are found most effective. However, individual-specific models are recommended as no universal set of optimal features and activities was identified across all participants.
>
---
#### [replaced 015] H-CNN-ViT: A Hierarchical Gated Attention Multi-Branch Model for Bladder Cancer Recurrence Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.13869v2](https://arxiv.org/pdf/2511.13869v2)**

> **作者:** Xueyang Li; Zongren Wang; Yuliang Zhang; Zixuan Pan; Yu-Jen Chen; Nishchal Sapkota; Gelei Xu; Danny Z. Chen; Yiyu Shi
>
> **摘要:** Bladder cancer is one of the most prevalent malignancies worldwide, with a recurrence rate of up to 78%, necessitating accurate post-operative monitoring for effective patient management. Multi-sequence contrast-enhanced MRI is commonly used for recurrence detection; however, interpreting these scans remains challenging, even for experienced radiologists, due to post-surgical alterations such as scarring, swelling, and tissue remodeling. AI-assisted diagnostic tools have shown promise in improving bladder cancer recurrence prediction, yet progress in this field is hindered by the lack of dedicated multi-sequence MRI datasets for recurrence assessment study. In this work, we first introduce a curated multi-sequence, multi-modal MRI dataset specifically designed for bladder cancer recurrence prediction, establishing a valuable benchmark for future research. We then propose H-CNN-ViT, a new Hierarchical Gated Attention Multi-Branch model that enables selective weighting of features from the global (ViT) and local (CNN) paths based on contextual demands, achieving a balanced and targeted feature fusion. Our multi-branch architecture processes each modality independently, ensuring that the unique properties of each imaging channel are optimally captured and integrated. Evaluated on our dataset, H-CNN-ViT achieves an AUC of 78.6%, surpassing state-of-the-art models. Our model is publicly available at https://github.com/XLIAaron/H-CNN-ViT.
>
---
#### [replaced 016] ANTS: Adaptive Negative Textual Space Shaping for OOD Detection via Test-Time MLLM Understanding and Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.03951v3](https://arxiv.org/pdf/2509.03951v3)**

> **作者:** Wenjie Zhu; Yabin Zhang; Xin Jin; Wenjun Zeng; Lei Zhang
>
> **摘要:** The introduction of negative labels (NLs) has proven effective in enhancing Out-of-Distribution (OOD) detection. However, existing methods often lack an understanding of OOD images, making it difficult to construct an accurate negative space. Furthermore, the absence of negative labels semantically similar to ID labels constrains their capability in near-OOD detection. To address these issues, we propose shaping an Adaptive Negative Textual Space (ANTS) by leveraging the understanding and reasoning capabilities of multimodal large language models (MLLMs). Specifically, we cache images likely to be OOD samples from the historical test images and prompt the MLLM to describe these images, generating expressive negative sentences that precisely characterize the OOD distribution and enhance far-OOD detection. For the near-OOD setting, where OOD samples resemble the in-distribution (ID) subset, we cache the subset of ID classes that are visually similar to historical test images and then leverage MLLM reasoning to generate visually similar negative labels tailored to this subset, effectively reducing false negatives and improving near-OOD detection. To balance these two types of negative textual spaces, we design an adaptive weighted score that enables the method to handle different OOD task settings (near-OOD and far-OOD), making it highly adaptable in open environments. On the ImageNet benchmark, our ANTS significantly reduces the FPR95 by 3.1\%, establishing a new state-of-the-art. Furthermore, our method is training-free and zero-shot, enabling high scalability.
>
---
#### [replaced 017] RoboTidy : A 3D Gaussian Splatting Household Tidying Benchmark for Embodied Navigation and Action
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14161v2](https://arxiv.org/pdf/2511.14161v2)**

> **作者:** Xiaoquan Sun; Ruijian Zhang; Kang Pang; Bingchen Miao; Yuxiang Tan; Zhen Yang; Ming Li; Jiayu Chen
>
> **摘要:** Household tidying is an important application area, yet current benchmarks neither model user preferences nor support mobility, and they generalize poorly, making it hard to comprehensively assess integrated language-to-action capabilities. To address this, we propose RoboTidy, a unified benchmark for language-guided household tidying that supports Vision-Language-Action (VLA) and Vision-Language-Navigation (VLN) training and evaluation. RoboTidy provides 500 photorealistic 3D Gaussian Splatting (3DGS) household scenes (covering 500 objects and containers) with collisions, formulates tidying as an "Action (Object, Container)" list, and supplies 6.4k high-quality manipulation demonstration trajectories and 1.5k naviagtion trajectories to support both few-shot and large-scale training. We also deploy RoboTidy in the real world for object tidying, establishing an end-to-end benchmark for household tidying. RoboTidy offers a scalable platform and bridges a key gap in embodied AI by enabling holistic and realistic evaluation of language-guided robots.
>
---
#### [replaced 018] PointVDP: Learning View-Dependent Projection by Fireworks Rays for 3D Point Cloud Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.06618v2](https://arxiv.org/pdf/2507.06618v2)**

> **作者:** Yang Chen; Yueqi Duan; Haowen Sun; Ziwei Wang; Jiwen Lu; Yap-Peng Tan
>
> **备注:** This version needs major revision
>
> **摘要:** In this paper, we propose view-dependent projection (VDP) to facilitate point cloud segmentation, designing efficient 3D-to-2D mapping that dynamically adapts to the spatial geometry from view variations. Existing projection-based methods leverage view-independent projection in complex scenes, relying on straight lines to generate direct rays or upward curves to reduce occlusions. However, their view independence provides projection rays that are limited to pre-defined parameters by human settings, restricting point awareness and failing to capture sufficient projection diversity across different view planes. Although multiple projections per view plane are commonly used to enhance spatial variety, the projected redundancy leads to excessive computational overhead and inefficiency in image processing. To address these limitations, we design a framework of VDP to generate data-driven projections from 3D point distributions, producing highly informative single-image inputs by predicting rays inspired by the adaptive behavior of fireworks. In addition, we construct color regularization to optimize the framework, which emphasizes essential features within semantic pixels and suppresses the non-semantic features within black pixels, thereby maximizing 2D space utilization in a projected image. As a result, our approach, PointVDP, develops lightweight projections in marginal computation costs. Experiments on S3DIS and ScanNet benchmarks show that our approach achieves competitive results, offering a resource-efficient solution for semantic understanding.
>
---
#### [replaced 019] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2503.07265v3](https://arxiv.org/pdf/2503.07265v3)**

> **作者:** Yuwei Niu; Munan Ning; Mengren Zheng; Weiyang Jin; Bin Lin; Peng Jin; Jiaqi Liao; Chaoran Feng; Kunpeng Ning; Bin Zhu; Li Yuan
>
> **备注:** Code, data and leaderboard: https://github.com/PKU-YuanGroup/WISE
>
> **摘要:** Text-to-Image (T2I) models are capable of generating high-quality artistic creations and visual content. However, existing research and evaluation standards predominantly focus on image realism and shallow text-image alignment, lacking a comprehensive assessment of complex semantic understanding and world knowledge integration in text-to-image generation. To address this challenge, we propose \textbf{WISE}, the first benchmark specifically designed for \textbf{W}orld Knowledge-\textbf{I}nformed \textbf{S}emantic \textbf{E}valuation. WISE moves beyond simple word-pixel mapping by challenging models with 1000 meticulously crafted prompts across 25 subdomains in cultural common sense, spatio-temporal reasoning, and natural science. To overcome the limitations of traditional CLIP metric, we introduce \textbf{WiScore}, a novel quantitative metric for assessing knowledge-image alignment. Through comprehensive testing of 20 models (10 dedicated T2I models and 10 unified multimodal models) using 1,000 structured prompts spanning 25 subdomains, our findings reveal significant limitations in their ability to effectively integrate and apply world knowledge during image generation, highlighting critical pathways for enhancing knowledge incorporation and application in next-generation T2I models. Code and data are available at \href{https://github.com/PKU-YuanGroup/WISE}{PKU-YuanGroup/WISE}.
>
---
#### [replaced 020] Gaussian Mapping for Evolving Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.06909v2](https://arxiv.org/pdf/2506.06909v2)**

> **作者:** Vladimir Yugay; Thies Kersten; Luca Carlone; Theo Gevers; Martin R. Oswald; Lukas Schmid
>
> **摘要:** Mapping systems with novel view synthesis (NVS) capabilities, most notably 3D Gaussian Splatting (3DGS), are widely used in computer vision and across various applications, including augmented reality, robotics, and autonomous driving. However, many current approaches are limited to static scenes. While recent works have begun addressing short-term dynamics (motion within the camera's view), long-term dynamics (the scene evolving through changes out of view) remain less explored. To overcome this limitation, we introduce a dynamic scene-adaptation mechanism that continuously updates 3DGS to reflect the latest changes. Since maintaining consistency remains challenging due to stale observations that disrupt the reconstruction process, we propose a novel keyframe management mechanism that discards outdated observations while preserving as much information as possible. We thoroughly evaluate Gaussian Mapping for Evolving Scenes (\ours) on both synthetic and real-world datasets, achieving a 29.7\% improvement in PSNR and a 3 times improvement in L1 depth error over the most competitive baseline.
>
---
#### [replaced 021] OmniSparse: Training-Aware Fine-Grained Sparse Attention for Long-Video MLLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12201v2](https://arxiv.org/pdf/2511.12201v2)**

> **作者:** Feng Chen; Yefei He; Shaoxuan He; Yuanyu He; Jing Liu; Lequan Lin; Akide Liu; Zhaoyang Li; Jiyuan Zhang; Zhenbang Sun; Bohan Zhuang; Qi Wu
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Existing sparse attention methods primarily target inference-time acceleration by selecting critical tokens under predefined sparsity patterns. However, they often fail to bridge the training-inference gap and lack the capacity for fine-grained token selection across multiple dimensions such as queries, key-values (KV), and heads, leading to suboptimal performance and limited acceleration gains. In this paper, we introduce OmniSparse, a training-aware fine-grained sparse attention framework for long-video MLLMs, which operates in both training and inference with dynamic token budget allocation. Specifically, OmniSparse contains three adaptive and complementary mechanisms: (1) query selection via lazy-active classification, retaining active queries that capture broad semantic similarity while discarding most lazy ones that focus on limited local context and exhibit high functional redundancy; (2) KV selection with head-level dynamic budget allocation, where a shared budget is determined based on the flattest head and applied uniformly across all heads to ensure attention recall; and (3) KV cache slimming to reduce head-level redundancy by selectively fetching visual KV cache according to the head-level decoding query pattern. Experimental results show that OmniSparse matches the performance of full attention while achieving up to 2.7x speedup during prefill and 2.4x memory reduction during decoding.
>
---
#### [replaced 022] Human-AI Collaboration and Explainability for 2D/3D Registration Quality Assurance
- **分类: cs.HC; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.17597v2](https://arxiv.org/pdf/2507.17597v2)**

> **作者:** Sue Min Cho; Alexander Do; Russell H. Taylor; Mathias Unberath
>
> **摘要:** Purpose: As surgery increasingly integrates advanced imaging, algorithms, and robotics to automate complex tasks, human judgment of system correctness remains a vital safeguard for patient safety. A critical example is 2D/3D registration, where small registration misalignments can lead to surgical errors. Current visualization strategies alone are insufficient to reliably enable humans to detect these misalignments, highlighting the need for enhanced decision-support tools. Methods: We propose the first artificial intelligence (AI) model tailored to 2D/3D registration quality assessment, augmented with explainable AI (XAI) mechanisms to clarify the model's predictions. Using both objective measures (e.g., accuracy, sensitivity, precision, specificity) and subjective evaluations (e.g., workload, trust, and understanding), we systematically compare decision-making across four conditions: AI-only, Human-only, Human+AI, and Human+XAI. Results: The AI-only condition achieved the highest accuracy, whereas collaborative paradigms (Human+AI and Human+XAI) improved sensitivity, precision, and specificity compared to standalone approaches. Participants experienced significantly lower workload in collaborative conditions relative to the Human-only condition. Moreover, participants reported a greater understanding of AI predictions in the Human+XAI condition than in Human+AI, although no significant differences were observed between the two collaborative paradigms in perceived trust or workload. Conclusion: Human-AI collaboration can enhance 2D/3D registration quality assurance, with explainability mechanisms improving user understanding. Future work should refine XAI designs to optimize decision-making performance and efficiency. Extending both the algorithmic design and human-XAI collaboration elements holds promise for more robust quality assurance of 2D/3D registration.
>
---
#### [replaced 023] Streaming Generation of Co-Speech Gestures via Accelerated Rolling Diffusion
- **分类: cs.LG; cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2503.10488v3](https://arxiv.org/pdf/2503.10488v3)**

> **作者:** Evgeniia Vu; Andrei Boiarov; Dmitry Vetrov
>
> **备注:** Accepted at the 40th AAAI Conference on Artificial Intelligence (AAAI-26) Main Track
>
> **摘要:** Generating co-speech gestures in real time requires both temporal coherence and efficient sampling. We introduce a novel framework for streaming gesture generation that extends Rolling Diffusion models with structured progressive noise scheduling, enabling seamless long-sequence motion synthesis while preserving realism and diversity. Our framework is universally compatible with existing diffusion-based gesture generation model, transforming them into streaming methods capable of continuous generation without requiring post-processing. We evaluate our framework on ZEGGS and BEAT, strong benchmarks for real-world applicability. Applied to state-of-the-art baselines on both datasets, it consistently outperforms them, demonstrating its effectiveness as a generalizable and efficient solution for real-time co-speech gesture synthesis. We further propose Rolling Diffusion Ladder Acceleration (RDLA), a new approach that employs a ladder-based noise scheduling strategy to simultaneously denoise multiple frames. This significantly improves sampling efficiency while maintaining motion consistency, achieving up to a 4x speedup with high visual fidelity and temporal coherence in our experiments. Comprehensive user studies further validate our framework ability to generate realistic, diverse gestures closely synchronized with the audio input.
>
---
#### [replaced 024] Visual Odometry with Transformers
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.03348v2](https://arxiv.org/pdf/2510.03348v2)**

> **作者:** Vlardimir Yugay; Duy-Kien Nguyen; Theo Gevers; Cees G. M. Snoek; Martin R. Oswald
>
> **摘要:** Despite the rapid development of large 3D models, classical optimization-based approaches dominate the field of visual odometry (VO). Thus, current approaches to VO heavily rely on camera parameters and many handcrafted components, most of which involve complex bundle adjustment and feature-matching processes. Although disregarded in the literature, we find it problematic in terms of both (1) speed, that performs bundle adjustment requires a significant amount of time, and (2) scalability, as hand-crafted components struggle to learn from large-scale training data. In this work, we introduce a simple yet efficient architecture, Visual Odometry Transformer (VoT), that formulates monocular visual odometry as a direct relative pose regression problem. Our approach streamlines the monocular visual odometry pipeline in an end-to-end manner, effectively eliminating the need for handcrafted components such as bundle adjustment, feature matching, or camera calibration. We show that VoT is up to 4 times faster than traditional approaches, yet with competitive or better performance. Compared to recent 3D foundation models, VoT runs 10 times faster with strong scaling behavior in terms of both model sizes and training data. Moreover, VoT generalizes well in both low-data regimes and previously unseen scenarios, reducing the gap between optimization-based and end-to-end approaches.
>
---
#### [replaced 025] UNIV: Unified Foundation Model for Infrared and Visible Modalities
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.15642v2](https://arxiv.org/pdf/2509.15642v2)**

> **作者:** Fangyuan Mao; Shuo Wang; Jilin Mei; Shun Lu; Chen Min; Fuyang Liu; Xiaokun Feng; Meiqi Wu; Yu Hu
>
> **摘要:** Joint RGB-infrared perception is essential for achieving robustness under diverse weather and illumination conditions. Although foundation models excel within single modalities, they suffer from substantial cross-modal degradation, an issue we attribute to a pattern shortcut, i.e., a modal bias that prioritizes superficial sensor patterns over underlying semantics. To address this problem, we introduce UNIV, a Unified foundation model for Infrared and Visible modalities. At the core of UNIV lies Patch Cross-modal Contrastive Learning (PCCL), a self-supervised contrastive learning strategy that constructs a unified cross-modal feature space. PCCL employs a frozen pre-trained model to sample pseudo patch pairs based on semantic similarity, and aligns infrared-visible representations by attracting semantically related pairs while repelling unrelated ones. This process simultaneously enhances cross-modal alignment and inter-class semantic separability, guiding the model to focus on semantic structure rather than falling into pattern shortcuts. To further enable cross-modal learning, we introduce MVIP, the most comprehensive visible-infrared benchmark to date, containing 98,992 precisely aligned image pairs across diverse scenes. Extensive experiments demonstrate UNIV's superior performance on infrared tasks (+1.7 mIoU for semantic segmentation and +0.7 mAP for detection), while maintaining competitive accuracy on RGB tasks.
>
---
#### [replaced 026] UGG-ReID: Uncertainty-Guided Graph Model for Multi-Modal Object Re-Identification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.04638v3](https://arxiv.org/pdf/2507.04638v3)**

> **作者:** Xixi Wan; Aihua Zheng; Bo Jiang; Beibei Wang; Chenglong Li; Jin Tang
>
> **摘要:** Multi-modal object Re-IDentification (ReID) has gained considerable attention with the goal of retrieving specific targets across cameras using heterogeneous visual data sources. At present, multi-modal object ReID faces two core challenges: (1) learning robust features under fine-grained local noise caused by occlusion, frame loss, and other disruptions; and (2) effectively integrating heterogeneous modalities to enhance multi-modal representation. To address the above challenges, we propose a robust approach named Uncertainty-Guided Graph model for multi-modal object ReID (UGG-ReID). UGG-ReID is designed to mitigate noise interference and facilitate effective multi-modal fusion by estimating both local and sample-level aleatoric uncertainty and explicitly modeling their dependencies. Specifically, we first propose the Gaussian patch-graph representation model that leverages uncertainty to quantify fine-grained local cues and capture their structural relationships. This process boosts the expressiveness of modal-specific information, ensuring that the generated embeddings are both more informative and robust. Subsequently, we design an uncertainty-guided mixture of experts strategy that dynamically routes samples to experts exhibiting low uncertainty. This strategy effectively suppresses noise-induced instability, leading to enhanced robustness. Meanwhile, we design an uncertainty-guided routing to strengthen the multi-modal interaction, improving the performance. UGG-ReID is comprehensively evaluated on five representative multi-modal object ReID datasets, encompassing diverse spectral modalities. Experimental results show that the proposed method achieves excellent performance on all datasets and is significantly better than current methods in terms of noise immunity. Our code is available at https://github.com/wanxixi11/UGG-ReID.
>
---
#### [replaced 027] Learning from the Right Patches: A Two-Stage Wavelet-Driven Masked Autoencoder for Histopathology Representation Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06958v2](https://arxiv.org/pdf/2511.06958v2)**

> **作者:** Raneen Younis; Louay Hamdi; Lukas Chavez; Zahra Ahmadi
>
> **摘要:** Whole-slide images are central to digital pathology, yet their extreme size and scarce annotations make self-supervised learning essential. Masked Autoencoders (MAEs) with Vision Transformer backbones have recently shown strong potential for histopathology representation learning. However, conventional random patch sampling during MAE pretraining often includes irrelevant or noisy regions, limiting the model's ability to capture meaningful tissue patterns. In this paper, we present a lightweight and domain-adapted framework that brings structure and biological relevance into MAE-based learning through a wavelet-informed patch selection strategy. WISE-MAE applies a two-step coarse-to-fine process: wavelet-based screening at low magnification to locate structurally rich regions, followed by high-resolution extraction for detailed modeling. This approach mirrors the diagnostic workflow of pathologists and improves the quality of learned representations. Evaluations across multiple cancer datasets, including lung, renal, and colorectal tissues, show that WISE-MAE achieves competitive representation quality and downstream classification performance while maintaining efficiency under weak supervision.
>
---
#### [replaced 028] SymGS : Leveraging Local Symmetries for 3D Gaussian Splatting Compression
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2511.13264v2](https://arxiv.org/pdf/2511.13264v2)**

> **作者:** Keshav Gupta; Akshat Sanghvi; Shreyas Reddy Palley; Astitva Srivastava; Charu Sharma; Avinash Sharma
>
> **备注:** Project Page: https://symgs.github.io/
>
> **摘要:** 3D Gaussian Splatting has emerged as a transformative technique in novel view synthesis, primarily due to its high rendering speed and photorealistic fidelity. However, its memory footprint scales rapidly with scene complexity, often reaching several gigabytes. Existing methods address this issue by introducing compression strategies that exploit primitive-level redundancy through similarity detection and quantization. We aim to surpass the compression limits of such methods by incorporating symmetry-aware techniques, specifically targeting mirror symmetries to eliminate redundant primitives. We propose a novel compression framework, SymGS, introducing learnable mirrors into the scene, thereby eliminating local and global reflective redundancies for compression. Our framework functions as a plug-and-play enhancement to state-of-the-art compression methods, (e.g. HAC) to achieve further compression. Compared to HAC, we achieve $1.66 \times$ compression across benchmark datasets (upto $3\times$ on large-scale scenes). On an average, SymGS enables $\bf{108\times}$ compression of a 3DGS scene, while preserving rendering quality. The project page and supplementary can be found at symgs.github.io
>
---
#### [replaced 029] SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24006v2](https://arxiv.org/pdf/2509.24006v2)**

> **作者:** Jintao Zhang; Haoxu Wang; Kai Jiang; Shuo Yang; Kaiwen Zheng; Haocheng Xi; Ziteng Wang; Hongzhou Zhu; Min Zhao; Ion Stoica; Joseph E. Gonzalez; Jun Zhu; Jianfei Chen
>
> **摘要:** In Diffusion Transformer (DiT) models, particularly for video generation, attention latency is a major bottleneck due to the long sequence length and the quadratic complexity. We find that attention weights can be separated into two parts: a small fraction of large weights with high rank and the remaining weights with very low rank. This naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second. Based on this finding, we propose SLA (Sparse-Linear Attention), a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. SLA classifies attention weights into critical, marginal, and negligible categories, applying O(N^2) attention to critical weights, O(N) attention to marginal weights, and skipping negligible ones. SLA combines these computations into a single GPU kernel and supports both forward and backward passes. With only a few fine-tuning steps using SLA, DiT models achieve a 20x reduction in attention computation, resulting in significant acceleration without loss of generation quality. Experiments show that SLA reduces attention computation by 95% without degrading end-to-end generation quality, outperforming baseline methods. In addition, we implement an efficient GPU kernel for SLA, which yields a 13.7x speedup in attention computation and a 2.2x end-to-end speedup in video generation on Wan2.1-1.3B. The code is available at https://github.com/thu-ml/SLA.
>
---
#### [replaced 030] MaskRIS: Semantic Distortion-aware Data Augmentation for Referring Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.19067v2](https://arxiv.org/pdf/2411.19067v2)**

> **作者:** Minhyun Lee; Seungho Lee; Song Park; Dongyoon Han; Byeongho Heo; Hyunjung Shim
>
> **备注:** Accepted to TMLR 2025. First two authors contributed equally
>
> **摘要:** Referring Image Segmentation (RIS) is an advanced vision-language task that involves identifying and segmenting objects within an image as described by free-form text descriptions. While previous studies focused on aligning visual and language features, exploring training techniques, such as data augmentation, remains underexplored. In this work, we explore effective data augmentation for RIS and propose a novel training framework called Masked Referring Image Segmentation (MaskRIS). We observe that the conventional image augmentations fall short of RIS, leading to performance degradation, while simple random masking significantly enhances the performance of RIS. MaskRIS uses both image and text masking, followed by Distortion-aware Contextual Learning (DCL) to fully exploit the benefits of the masking strategy. This approach can improve the model's robustness to occlusions, incomplete information, and various linguistic complexities, resulting in a significant performance improvement. Experiments demonstrate that MaskRIS can easily be applied to various RIS models, outperforming existing methods in both fully supervised and weakly supervised settings. Finally, MaskRIS achieves new state-of-the-art performance on RefCOCO, RefCOCO+, and RefCOCOg datasets. Code is available at https://github.com/naver-ai/maskris.
>
---
#### [replaced 031] Causal Representation Learning with Observational Grouping for CXR Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.20582v2](https://arxiv.org/pdf/2506.20582v2)**

> **作者:** Rajat Rasal; Avinash Kori; Ben Glocker
>
> **备注:** Proceedings of the 3rd FAIMI Workshop at the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025, Daejeon, South Korea
>
> **摘要:** Identifiable causal representation learning seeks to uncover the true causal relationships underlying a data generation process. In medical imaging, this presents opportunities to improve the generalisability and robustness of task-specific latent features. This work introduces the concept of grouping observations to learn identifiable representations for disease classification in chest X-rays via an end-to-end framework. Our experiments demonstrate that these causal representations improve generalisability and robustness across multiple classification tasks when grouping is used to enforce invariance w.r.t race, sex, and imaging views.
>
---
#### [replaced 032] UniME-V2: MLLM-as-a-Judge for Universal Multimodal Embedding Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.13515v2](https://arxiv.org/pdf/2510.13515v2)**

> **作者:** Tiancheng Gu; Kaicheng Yang; Kaichen Zhang; Xiang An; Ziyong Feng; Yueyi Zhang; Weidong Cai; Jiankang Deng; Lidong Bing
>
> **备注:** AAAI2026 Oral, Webpage:https://garygutc.github.io/UniME-v2/
>
> **摘要:** Universal multimodal embedding models are foundational to various tasks. Existing approaches typically employ in-batch negative mining by measuring the similarity of query-candidate pairs. However, these methods often struggle to capture subtle semantic differences among candidates and lack diversity in negative samples. Moreover, the embeddings exhibit limited discriminative ability in distinguishing false and hard negatives. In this paper, we leverage the advanced understanding capabilities of MLLMs to enhance representation learning and present a novel Universal Multimodal Embedding (UniME-V2) model. Our approach first constructs a potential hard negative set through global retrieval. We then introduce the MLLM-as-a-Judge mechanism, which utilizes MLLMs to assess the semantic alignment of query-candidate pairs and generate soft semantic matching scores. These scores serve as a foundation for hard negative mining, mitigating the impact of false negatives and enabling the identification of diverse, high-quality hard negatives. Furthermore, the semantic matching scores are used as soft labels to mitigate the rigid one-to-one mapping constraint. By aligning the similarity matrix with the soft semantic matching score matrix, the model learns semantic distinctions among candidates, significantly enhancing its discriminative capacity. To further improve performance, we propose UniME-V2-Reranker, a reranking model trained on our mined hard negatives through a joint pairwise and listwise optimization approach. We conduct comprehensive experiments on the MMEB benchmark and multiple retrieval tasks, demonstrating that our method achieves state-of-the-art performance on average across all tasks.
>
---
#### [replaced 033] Label-Efficient Cross-Modality Generalization for Liver Segmentation in Multi-Phase MRI
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.04705v2](https://arxiv.org/pdf/2510.04705v2)**

> **作者:** Quang-Khai Bui-Tran; Minh-Toan Dinh; Thanh-Huy Nguyen; Ba-Thinh Lam; Mai-Anh Vu; Ulas Bagci
>
> **备注:** Accepted at CARE @ MICCAI 2025
>
> **摘要:** Accurate liver segmentation in multi-phase MRI is vital for liver fibrosis assessment, yet labeled data is often scarce and unevenly distributed across imaging modalities and vendor systems. We propose a label-efficient segmentation approach that promotes cross-modality generalization under real-world conditions, where GED4 hepatobiliary-phase annotations are limited, non-contrast sequences (T1WI, T2WI, DWI) are unlabeled, and spatial misalignment and missing phases are common. Our method integrates a foundation-scale 3D segmentation backbone adapted via fine-tuning, co-training with cross pseudo supervision to leverage unlabeled volumes, and a standardized preprocessing pipeline. Without requiring spatial registration, the model learns to generalize across MRI phases and vendors, demonstrating robust segmentation performance in both labeled and unlabeled domains. Our results exhibit the effectiveness of our proposed label-efficient baseline for liver segmentation in multi-phase, multi-vendor MRI and highlight the potential of combining foundation model adaptation with co-training for real-world clinical imaging tasks.
>
---
#### [replaced 034] ReassembleNet: Learnable Keypoints and Diffusion for 2D Fresco Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.21117v3](https://arxiv.org/pdf/2505.21117v3)**

> **作者:** Adeela Islam; Stefano Fiorini; Stuart James; Pietro Morerio; Alessio Del Bue
>
> **摘要:** The task of reassembly is a significant challenge across multiple domains, including archaeology, genomics, and molecular docking, requiring the precise placement and orientation of elements to reconstruct an original structure. In this work, we address key limitations in state-of-the-art Deep Learning methods for reassembly, namely i) scalability; ii) multimodality; and iii) real-world applicability: beyond square or simple geometric shapes, realistic and complex erosion, or other real-world problems. We propose ReassembleNet, a method that reduces complexity by representing each input piece as a set of contour keypoints and learning to select the most informative ones by Graph Neural Networks pooling inspired techniques. ReassembleNet effectively lowers computational complexity while enabling the integration of features from multiple modalities, including both geometric and texture data. Further enhanced through pretraining on a semi-synthetic dataset. We then apply diffusion-based pose estimation to recover the original structure. We improve on prior methods by 57% and 87% for RMSE Rotation and Translation, respectively.
>
---
#### [replaced 035] The Role of Radiographic Knee Alignment in Total Knee Replacement Outcomes and Opportunities for Artificial Intelligence-Driven Assessment
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10941v2](https://arxiv.org/pdf/2508.10941v2)**

> **作者:** Zhisen Hu; Dominic Cullen; David S. Johnson; Aleksei Tiulpin; Timothy F. Cootes; Claudia Lindner
>
> **摘要:** Knee osteoarthritis (OA) is one of the most widespread and burdensome health problems [1-4]. Total knee replacement (TKR) may be offered as treatment for end-stage knee OA. Nevertheless, TKR is an invasive procedure involving prosthesis implantation at the knee joint, and around 10% of patients are dissatisfied following TKR [5,6]. Dissatisfaction is often assessed through patient-reported outcome measures (PROMs) [7], which are usually completed by patients and assessed by health professionals to evaluate the condition of TKR patients. In clinical practice, predicting poor TKR outcomes in advance could help optimise patient selection and improve management strategies. Radiographic knee alignment is an important biomarker for predicting TKR outcomes and long-term joint health. Abnormalities such as femoral or tibial deformities can directly influence surgical planning, implant selection, and postoperative recovery [8,9]. Traditional alignment measurement is manual, time-consuming, and requires long-leg radiographs, which are not always undertaken in clinical practice. Instead, standard anteroposterior (AP) knee radiographs are often the main imaging modality. Automated methods for alignment assessment in standard knee radiographs are potentially clinically valuable for improving efficiency in the knee OA treatment pathway.
>
---
#### [replaced 036] CST Anti-UAV: A Thermal Infrared Benchmark for Tiny UAV Tracking in Complex Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.23473v2](https://arxiv.org/pdf/2507.23473v2)**

> **作者:** Bin Xie; Congxuan Zhang; Fagan Wang; Peng Liu; Feng Lu; Zhen Chen; Weiming Hu
>
> **备注:** Accepted by ICCVW2025
>
> **摘要:** The widespread application of Unmanned Aerial Vehicles (UAVs) has raised serious public safety and privacy concerns, making UAV perception crucial for anti-UAV tasks. However, existing UAV tracking datasets predominantly feature conspicuous objects and lack diversity in scene complexity and attribute representation, limiting their applicability to real-world scenarios. To overcome these limitations, we present the CST Anti-UAV, a new thermal infrared dataset specifically designed for Single Object Tracking (SOT) in Complex Scenes with Tiny UAVs (CST). It contains 220 video sequences with over 240k high-quality bounding box annotations, highlighting two key properties: a significant number of tiny-sized UAV targets and the diverse and complex scenes. To the best of our knowledge, CST Anti-UAV is the first dataset to incorporate complete manual frame-level attribute annotations, enabling precise evaluations under varied challenges. To conduct an in-depth performance analysis for CST Anti-UAV, we evaluate 20 existing SOT methods on the proposed dataset. Experimental results demonstrate that tracking tiny UAVs in complex environments remains a challenge, as the state-of-the-art method achieves only 35.92% state accuracy, much lower than the 67.69% observed on the Anti-UAV410 dataset. These findings underscore the limitations of existing benchmarks and the need for further advancements in UAV tracking research. The CST Anti-UAV benchmark is about to be publicly released, which not only fosters the development of more robust SOT methods but also drives innovation in anti-UAV systems.
>
---
#### [replaced 037] ViewBridge:Revisiting Cross-View Localization from Image Matching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10716v2](https://arxiv.org/pdf/2508.10716v2)**

> **作者:** Panwang Xia; Qiong Wu; Lei Yu; Yi Liu; Mingtao Xiong; Xudong Lu; Yi Liu; Haoyu Guo; Yongxiang Yao; Junjian Zhang; Xiangyuan Cai; Hongwei Hu; Zhi Zheng; Yongjun Zhang; Yi Wan
>
> **摘要:** Cross-view localization aims to estimate the 3-DoF pose of a ground-view image by aligning it with aerial or satellite imagery. Existing methods typically address this task through direct regression or feature alignment in a shared bird's-eye view (BEV) space. Although effective for coarse alignment, these methods fail to establish fine-grained and geometrically reliable correspondences under large viewpoint variations, thereby limiting both the accuracy and interpretability of localization results. Consequently, we revisit cross-view localization from the perspective of image matching and propose a unified framework that enhances both matching and localization. Specifically, we introduce a Surface Model that constrains BEV feature projection to physically valid regions for geometric consistency, and a SimRefiner that adaptively refines similarity distributions to enhance match reliability. To further support research in this area, we present CVFM, the first benchmark with 32,509 cross-view image pairs annotated with pixel-level correspondences. Extensive experiments demonstrate that our approach achieves geometry-consistent and fine-grained correspondences across extreme viewpoints and further improves the accuracy and stability of cross-view localization.
>
---
#### [replaced 038] One Latent Space to Rule All Degradations: Unifying Restoration Knowledge for Image Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.07033v3](https://arxiv.org/pdf/2503.07033v3)**

> **作者:** Haolong Ma; Hui Li; Chunyang Cheng; Zeyang Zhang; Xiaoqing Luo; Xiaoning Song; Xiao-Jun Wu
>
> **摘要:** All-in-One Degradation-Aware Fusion Models (ADFMs) as one of multi-modal image fusion models, which aims to address complex scenes by mitigating degradations from source images and generating high-quality fused images. Mainstream ADFMs rely on end-to-end learning and heavily synthesized datasets to achieve degradation awareness and fusion. This rough learning strategy and non-real world scenario dataset dependence often limit their upper-bound performance, leading to low-quality results. To address these limitations, we present LURE, a Learning-driven Unified REpresentation model for infrared and visible image fusion, which is degradation-aware. LURE learns a Unified Latent Feature Space (ULFS) to avoid the dependency on complex data formats inherent in previous end-to-end learning pipelines. It further improves image fusion quality by leveraging the intrinsic relationships between multi-modalities. A novel loss function is also proposed to drive the learning of unified latent representations more stable.More importantly, LURE seamlessly incorporates existing high-quality real-world image restoration datasets. To further enhance the model's representation capability, we design a simple yet effective structure, termed internal residual block, to facilitate the learning of latent features. Experiments show our method outperforms state-of-the-art (SOTA) methods across general fusion, degradation-aware fusion, and downstream tasks. The code is available in the supplementary materials.
>
---
#### [replaced 039] Systematic Evaluation and Guidelines for Segment Anything Model in Surgical Video Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.00525v2](https://arxiv.org/pdf/2501.00525v2)**

> **作者:** Cheng Yuan; Jian Jiang; Kunyi Yang; Lv Wu; Rui Wang; Zi Meng; Haonan Ping; Ziyu Xu; Yifan Zhou; Wanli Song; Hesheng Wang; Qi Dou; Yutong Ban
>
> **摘要:** Surgical video segmentation is critical for AI to interpret spatial-temporal dynamics in surgery, yet model performance is constrained by limited annotated data. The SAM2 model, pretrained on natural videos, offers potential for zero-shot surgical segmentation, but its applicability in complex surgical environments, with challenges like tissue deformation and instrument variability, remains unexplored. We present the first comprehensive evaluation of the zero-shot capability of SAM2 in 9 surgical datasets (17 surgery types), covering laparoscopic, endoscopic, and robotic procedures. We analyze various prompting (points, boxes, mask) and {finetuning (dense, sparse) strategies}, robustness to surgical challenges, and generalization across procedures and anatomies. Key findings reveal that while SAM2 demonstrates notable zero-shot adaptability in structured scenarios (e.g., instrument segmentation, {multi-organ segmentation}, and scene segmentation), its performance varies under dynamic surgical conditions, highlighting gaps in handling temporal coherence and domain-specific artifacts. These results highlight future pathways to adaptive data-efficient solutions for the surgical data science field.
>
---
#### [replaced 040] Alpha Divergence Losses for Biometric Verification
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.13621v2](https://arxiv.org/pdf/2511.13621v2)**

> **作者:** Dimitrios Koutsianos; Ladislav Mosner; Yannis Panagakis; Themos Stafylakis
>
> **备注:** Found something suboptimal in results
>
> **摘要:** Performance in face and speaker verification is largely driven by margin based softmax losses like CosFace and ArcFace. Recently introduced $α$-divergence loss functions offer a compelling alternative, particularly for their ability to induce sparse solutions (when $α>1$). However, integrating an angular margin-crucial for verification tasks-is not straightforward. We find this integration can be achieved in at least two distinct ways: via the reference measure (prior probabilities) or via the logits (unnormalized log-likelihoods). In this paper, we explore both pathways, deriving two novel margin-based $α$-divergence losses: Q-Margin (margin in the reference measure) and A3M (margin in the logits). We identify and address a critical training instability in A3M-caused by the interplay of penalized logits and sparsity-with a simple yet effective prototype re-initialization strategy. Our methods achieve significant performance gains on the challenging IJB-B and IJB-C face verification benchmarks. We demonstrate similarly strong performance in speaker verification on VoxCeleb. Crucially, our models significantly outperform strong baselines at low false acceptance rates (FAR). This capability is crucial for practical high-security applications, such as banking authentication, when minimizing false authentications is paramount.
>
---
#### [replaced 041] Gaussian Splatting-based Low-Rank Tensor Representation for Multi-Dimensional Image Recovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14270v2](https://arxiv.org/pdf/2511.14270v2)**

> **作者:** Yiming Zeng; Xi-Le Zhao; Wei-Hao Wu; Teng-Yu Ji; Chao Wang
>
> **摘要:** Tensor singular value decomposition (t-SVD) is a promising tool for multi-dimensional image representation, which decomposes a multi-dimensional image into a latent tensor and an accompanying transform matrix. However, two critical limitations of t-SVD methods persist: (1) the approximation of the latent tensor (e.g., tensor factorizations) is coarse and fails to accurately capture spatial local high-frequency information; (2) The transform matrix is composed of fixed basis atoms (e.g., complex exponential atoms in DFT and cosine atoms in DCT) and cannot precisely capture local high-frequency information along the mode-3 fibers. To address these two limitations, we propose a Gaussian Splatting-based Low-rank tensor Representation (GSLR) framework, which compactly and continuously represents multi-dimensional images. Specifically, we leverage tailored 2D Gaussian splatting and 1D Gaussian splatting to generate the latent tensor and transform matrix, respectively. The 2D and 1D Gaussian splatting are indispensable and complementary under this representation framework, which enjoys a powerful representation capability, especially for local high-frequency information. To evaluate the representation ability of the proposed GSLR, we develop an unsupervised GSLR-based multi-dimensional image recovery model. Extensive experiments on multi-dimensional image recovery demonstrate that GSLR consistently outperforms state-of-the-art methods, particularly in capturing local high-frequency information.
>
---
#### [replaced 042] Clothing agnostic Pre-inpainting Virtual Try-ON
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.17654v3](https://arxiv.org/pdf/2509.17654v3)**

> **作者:** Sehyun Kim; Hye Jun Lee; Jiwoo Lee; Taemin Lee
>
> **备注:** Github : https://github.com/DevChoco/CAP-VTON
>
> **摘要:** With the development of deep learning technology, virtual try-on technology has devel-oped important application value in the fields of e-commerce, fashion, and entertainment. The recently proposed Leffa technology has addressed the texture distortion problem of diffusion-based models, but there are limitations in that the bottom detection inaccuracy and the existing clothing silhouette persist in the synthesis results. To solve this problem, this study proposes CaP-VTON (Clothing Agnostic Pre-Inpainting Virtual Try-On). CaP-VTON integrates DressCode-based multi-category masking and Stable Diffu-sion-based skin inflation preprocessing; in particular, a generated skin module was in-troduced to solve skin restoration problems that occur when long-sleeved images are con-verted to short-sleeved or sleeveless ones, introducing a preprocessing structure that im-proves the naturalness and consistency of full-body clothing synthesis, and allowing the implementation of high-quality restoration considering human posture and color. As a result, CaP-VTON achieved 92.5%, which is 15.4% better than Leffa, in short-sleeved syn-thesis accuracy, and consistently reproduced the style and shape of the reference clothing in visual evaluation. These structures maintain model-agnostic properties and are appli-cable to various diffusion-based virtual inspection systems; they can also contribute to applications that require high-precision virtual wearing, such as e-commerce, custom styling, and avatar creation.
>
---
#### [replaced 043] TongUI: Building Generalized GUI Agents by Learning from Multimodal Web Tutorials
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.12679v3](https://arxiv.org/pdf/2504.12679v3)**

> **作者:** Bofei Zhang; Zirui Shang; Zhi Gao; Wang Zhang; Rui Xie; Xiaojian Ma; Tao Yuan; Xinxiao Wu; Song-Chun Zhu; Qing Li
>
> **备注:** AAAI 2026
>
> **摘要:** Building Graphical User Interface (GUI) agents is a promising research direction, which simulates human interaction with computers or mobile phones to perform diverse GUI tasks. However, a major challenge in developing generalized GUI agents is the lack of sufficient trajectory data across various operating systems and applications, mainly due to the high cost of manual annotations. In this paper, we propose the TongUI framework that builds generalized GUI agents by learning from rich multimodal web tutorials. Concretely, we crawl and process online GUI tutorials (such as videos and articles) into GUI agent trajectory data, through which we produce the GUI-Net dataset containing 143K trajectory data across five operating systems and more than 200 applications. We develop the TongUI agent by fine-tuning Qwen2.5-VL-3B/7B models on GUI-Net, which show remarkable performance improvements on commonly used grounding and navigation benchmarks, outperforming baseline agents about 10\% on multiple benchmarks, showing the effectiveness of the GUI-Net dataset and underscoring the significance of our TongUI framework. We will fully open-source the code, the GUI-Net dataset, and the trained models soon.
>
---
#### [replaced 044] GloTok: Global Perspective Tokenizer for Image Reconstruction and Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14184v2](https://arxiv.org/pdf/2511.14184v2)**

> **作者:** Xuan Zhao; Zhongyu Zhang; Yuge Huang; Yuxi Mi; Guodong Mu; Shouhong Ding; Jun Wang; Rizen Guo; Shuigeng Zhou
>
> **备注:** Accepted at AAAI'26
>
> **摘要:** Existing state-of-the-art image tokenization methods leverage diverse semantic features from pre-trained vision models for additional supervision, to expand the distribution of latent representations and thereby improve the quality of image reconstruction and generation. These methods employ a locally supervised approach for semantic supervision, which limits the uniformity of semantic distribution. However, VA-VAE proves that a more uniform feature distribution yields better generation performance. In this work, we introduce a Global Perspective Tokenizer (GloTok), which utilizes global relational information to model a more uniform semantic distribution of tokenized features. Specifically, a codebook-wise histogram relation learning method is proposed to transfer the semantics, which are modeled by pre-trained models on the entire dataset, to the semantic codebook. Then, we design a residual learning module that recovers the fine-grained details to minimize the reconstruction error caused by quantization. Through the above design, GloTok delivers more uniformly distributed semantic latent representations, which facilitates the training of autoregressive (AR) models for generating high-quality images without requiring direct access to pre-trained models during the training process. Experiments on the standard ImageNet-1k benchmark clearly show that our proposed method achieves state-of-the-art reconstruction performance and generation quality.
>
---
#### [replaced 045] Style Content Decomposition-based Data Augmentation for Domain Generalizable Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2502.20619v3](https://arxiv.org/pdf/2502.20619v3)**

> **作者:** Zhiqiang Shen; Peng Cao; Jinzhu Yang; Osmar R. Zaiane; Zhaolin Chen
>
> **摘要:** Due to domain shifts across diverse medical imaging modalities, learned segmentation models often suffer significant performance degradation during deployment. We posit that these domain shifts can generally be categorized into two main components: 1) "style" shifts, referring to global disparities in image properties such as illumination, contrast, and color; and 2) "content" shifts, which involve local discrepancies in anatomical structures. To address the domain shifts in medical image segmentation, we first factorize an image into style codes and content maps, explicitly modeling the "style" and "content" components. Building on this, we introduce a Style-Content decomposition-based data augmentation algorithm (StyCona), which performs augmentation on both the global style and local content of source-domain images, enabling the training of a well-generalized model for domain generalizable medical image segmentation. StyCona is a simple yet effective plug-and-play module that substantially improves model generalization without requiring additional training parameters or modifications to segmentation model architectures. Experiments on cardiac magnetic resonance imaging and fundus photography segmentation tasks, with single and multiple target domains respectively, demonstrate the effectiveness of StyCona and its superiority over state-of-the-art domain generalization methods. The code is available at https://github.com/Senyh/StyCona.
>
---
#### [replaced 046] Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.14386v2](https://arxiv.org/pdf/2511.14386v2)**

> **作者:** Kangqiao Zhao; Shuo Huai; Xurui Song; Jun Luo
>
> **摘要:** Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.
>
---
#### [replaced 047] MK-SGN: A Spiking Graph Convolutional Network with Multimodal Fusion and Knowledge Distillation for Skeleton-based Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2404.10210v5](https://arxiv.org/pdf/2404.10210v5)**

> **作者:** Naichuan Zheng; Hailun Xia; Zeyu Liang; Yuchen Du
>
> **摘要:** In recent years, multimodal Graph Convolutional Networks (GCNs) have achieved remarkable performance in skeleton-based action recognition. The reliance on high-energy-consuming continuous floating-point operations inherent in GCN-based methods poses significant challenges for deployment in energy-constrained, battery-powered edge devices. To address these limitations, MK-SGN, a Spiking Graph Convolutional Network with Multimodal Fusion and Knowledge Distillation, is proposed to leverage the energy efficiency of Spiking Neural Networks (SNNs) for skeleton-based action recognition for the first time. By integrating the energy-saving properties of SNNs with the graph representation capabilities of GCNs, MK-SGN achieves significant reductions in energy consumption while maintaining competitive recognition accuracy. Firstly, we formulate a Spiking Multimodal Fusion (SMF) module to effectively fuse multimodal skeleton data represented as spike-form features. Secondly, we propose the Self-Attention Spiking Graph Convolution (SA-SGC) module and the Spiking Temporal Convolution (STC) module, to capture spatial relationships and temporal dynamics of spike-form features. Finally, we propose an integrated knowledge distillation strategy to transfer information from the multimodal GCN to the SGN, incorporating both intermediate-layer distillation and soft-label distillation to enhance the performance of the SGN. MK-SGN exhibits substantial advantages, surpassing state-of-the-art GCN frameworks in energy efficiency and outperforming state-of-the-art SNN frameworks in recognition accuracy. The proposed method achieves a remarkable reduction in energy consumption, exceeding 98\% compared to conventional GCN-based approaches. This research establishes a robust baseline for developing high-performance, energy-efficient SNN-based models for skeleton-based action recognition
>
---
#### [replaced 048] Measuring the (Un)Faithfulness of Concept-Based Explanations
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.10833v3](https://arxiv.org/pdf/2504.10833v3)**

> **作者:** Shubham Kumar; Narendra Ahuja
>
> **备注:** Pre-print
>
> **摘要:** Deep vision models perform input-output computations that are hard to interpret. Concept-based explanation methods (CBEMs) increase interpretability by re-expressing parts of the model with human-understandable semantic units, or concepts. Checking if the derived explanations are faithful -- that is, they represent the model's internal computation -- requires a surrogate that combines concepts to compute the output. Simplifications made for interpretability inevitably reduce faithfulness, resulting in a tradeoff between the two. State-of-the-art unsupervised CBEMs (U-CBEMs) have reported increasingly interpretable concepts, while also being more faithful to the model. However, we observe that the reported improvement in faithfulness artificially results from either (1) using overly complex surrogates, which introduces an unmeasured cost to the explanation's interpretability, or (2) relying on deletion-based approaches that, as we demonstrate, do not properly measure faithfulness. We propose Surrogate Faithfulness (SURF), which (1) replaces prior complex surrogates with a simple, linear surrogate that measures faithfulness without changing the explanation's interpretability and (2) introduces well-motivated metrics that assess loss across all output classes, not just the predicted class. We validate SURF with a measure-over-measure study by proposing a simple sanity check -- explanations with random concepts should be less faithful -- which prior surrogates fail. SURF enables the first reliable faithfulness benchmark of U-CBEMs, revealing that many visually compelling U-CBEMs are not faithful. Code to be released.
>
---
#### [replaced 049] InvFusion: Bridging Supervised and Zero-shot Diffusion for Inverse Problems
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.01689v2](https://arxiv.org/pdf/2504.01689v2)**

> **作者:** Noam Elata; Hyungjin Chung; Jong Chul Ye; Tomer Michaeli; Michael Elad
>
> **摘要:** Diffusion Models have demonstrated remarkable capabilities in handling inverse problems, offering high-quality posterior-sampling-based solutions. Despite significant advances, a fundamental trade-off persists regarding the way the conditioned synthesis is employed: Zero-shot approaches can accommodate any linear degradation but rely on approximations that reduce accuracy. In contrast, training-based methods model the posterior correctly, but cannot adapt to the degradation at test-time. Here we introduce InvFusion, the first training-based degradation-aware posterior sampler. InvFusion combines the best of both worlds -- the strong performance of supervised approaches and the flexibility of zero-shot methods. This is achieved through a novel architectural design that seamlessly integrates the degradation operator directly into the diffusion denoiser. We compare InvFusion against existing general-purpose posterior samplers, both degradation-aware zero-shot techniques and blind training-based methods. Experiments on the FFHQ and ImageNet datasets demonstrate state-of-the-art performance. Beyond posterior sampling, we further demonstrate the applicability of our architecture, operating as a general Minimum Mean Square Error predictor, and as a Neural Posterior Principal Component estimator.
>
---
#### [replaced 050] Fairness-Aware Deepfake Detection: Leveraging Dual-Mechanism Optimization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10150v3](https://arxiv.org/pdf/2511.10150v3)**

> **作者:** Feng Ding; Wenhui Yi; Yunpeng Zhou; Xinan He; Hong Rao; Shu Hu
>
> **摘要:** Fairness is a core element in the trustworthy deployment of deepfake detection models, especially in the field of digital identity security. Biases in detection models toward different demographic groups, such as gender and race, may lead to systemic misjudgments, exacerbating the digital divide and social inequities. However, current fairness-enhanced detectors often improve fairness at the cost of detection accuracy. To address this challenge, we propose a dual-mechanism collaborative optimization framework. Our proposed method innovatively integrates structural fairness decoupling and global distribution alignment: decoupling channels sensitive to demographic groups at the model architectural level, and subsequently reducing the distance between the overall sample distribution and the distributions corresponding to each demographic group at the feature level. Experimental results demonstrate that, compared with other methods, our framework improves both inter-group and intra-group fairness while maintaining overall detection accuracy across domains.
>
---
#### [replaced 051] Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.25801v2](https://arxiv.org/pdf/2510.25801v2)**

> **作者:** Kun Chen; Peng Shi; Haibo Qiu; Zhixiong Zeng; Siqi Yang; Wenji Mao; Lin Ma
>
> **备注:** Project Page: https://github.com/Kwen-Chen/SPECS-VL
>
> **摘要:** Reinforcement learning (RL) with verifiable rewards has recently catalyzed a wave of "MLLM-r1" approaches that bring RL to vision language models. Most representative paradigms begin with a cold start, typically employing supervised fine-tuning (SFT), to initialize the policy before RL. However, SFT-based cold start adopts the reasoning paradigm intertwined with task solution and output format, which may induce instruction-style overfitting, weakens out-of-distribution generalization, and ultimately affects downstream RL. We revisit the cold start along two views, its training method and data construction, and introduce the Generalization Factor (GF) coefficient to quantify the generalization capability under different methods. Our empirical study finds that preference-based training methods (e.g. DPO) generalizes better than SFT-based methods in cold start. Motivated by this, we propose SPECS-a Self-distilled, Preference-based Cold Start framework that decouples multimodal learning: (1) generates introspective preference data pairs via self-distillation, avoiding reliance on larger teachers or manual annotation; (2) performs preference-based training to learn, focusing on shallow, transferable surface-form criteria (format, structure, style) rather than memorizing content; and (3) hands off to RL with verifiable rewards for deep reasoning results. Experimental results across multiple multimodal benchmarks show that our decoupling learning framework yields consistent performance gains over strong baselines, improving MEGA-Bench by 4.1% and MathVista by 12.2%. Additional experiments indicate that SPECS contributes to reducing in-distribution "stuckness," improving exploration, stabilizing training, and raising the performance ceiling.
>
---
#### [replaced 052] FireCastNet: Earth-as-a-Graph for Seasonal Fire Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.01550v2](https://arxiv.org/pdf/2502.01550v2)**

> **作者:** Dimitrios Michail; Charalampos Davalas; Konstantinos Chafis; Lefki-Ioanna Panagiotou; Ioannis Prapas; Spyros Kondylatos; Nikolaos Ioannis Bountos; Ioannis Papoutsis
>
> **摘要:** With climate change intensifying fire weather conditions globally, accurate seasonal wildfire forecasting has become critical for disaster preparedness and ecosystem management. We introduce FireCastNet, a novel deep learning architecture that combines 3D convolutional encoding with GraphCast-based Graph Neural Networks (GNNs) to model complex spatio-temporal dependencies for global wildfire prediction. Our approach leverages the SeasFire dataset, a comprehensive multivariate Earth system datacube containing climate, vegetation, and human-related variables, to forecast burned area patterns up to six months in advance. FireCastNet treats the Earth as an interconnected graph, enabling it to capture both local fire dynamics and long-range teleconnections that influence wildfire behavior across different spatial and temporal scales. Through comprehensive benchmarking against state-of-the-art models including GRU, Conv-GRU, Conv-LSTM, U-TAE, and TeleViT, we demonstrate that FireCastNet achieves superior performance in global burned area forecasting, with particularly strong results in fire-prone regions such as Africa, South America, and Southeast Asia. Our analysis reveals that longer input time-series significantly improve prediction robustness, while spatial context integration enhances model performance across extended forecasting horizons. Additionally, we implement local area modeling techniques that provide enhanced spatial resolution and accuracy for region-specific predictions. These findings highlight the importance of modeling Earth system interactions for long-term wildfire prediction.
>
---
#### [replaced 053] DINOv3 as a Frozen Encoder for CRPS-Oriented Probabilistic Rainfall Nowcasting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10894v2](https://arxiv.org/pdf/2511.10894v2)**

> **作者:** Luciano Araujo Dourado Filho; Almir Moreira da Silva Neto; Anthony Miyaguchi; Rodrigo Pereira David; Rodrigo Tripodi Calumby; Lukáš Picek
>
> **摘要:** This paper proposes a competitive and computationally efficient approach to probabilistic rainfall nowcasting. A video projector (V-JEPA Vision Transformer) associated to a lightweight probabilistic head is attached to a pre-trained satellite vision encoder (DINOv3-SAT493M) to map encoder tokens into a discrete empirical CDF (eCDF) over 4-hour accumulated rainfall. The projector-head is optimized end-to-end over the Ranked Probability Score (RPS). As an alternative, 3D-UNET baselines trained with an aggregate Rank Probability Score and a per-pixel Gamma-Hurdle objective are used. On the Weather4Cast 2025 benchmark, the proposed method achieved a promising performance, with a CRPS of 3.5102, which represents $\approx$ 26% in effectiveness gain against the best 3D-UNET.
>
---
#### [replaced 054] Distribution Matching Distillation Meets Reinforcement Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.13649v2](https://arxiv.org/pdf/2511.13649v2)**

> **作者:** Dengyang Jiang; Dongyang Liu; Zanyi Wang; Qilong Wu; Liuzhuozheng Li; Hengzhuang Li; Xin Jin; David Liu; Zhen Li; Bo Zhang; Mengmeng Wang; Steven Hoi; Peng Gao; Harry Yang
>
> **备注:** The synergy of reinforcement learning and distribution matching distillation. See more: https://github.com/vvvvvjdy/dmdr
>
> **摘要:** Distribution Matching Distillation (DMD) distills a pre-trained multi-step diffusion model to a few-step one to improve inference efficiency. However, the performance of the latter is often capped by the former. To circumvent this dilemma, we propose DMDR, a novel framework that combines Reinforcement Learning (RL) techniques into the distillation process. We show that for the RL of the few-step generator, the DMD loss itself is a more effective regularization compared to the traditional ones. In turn, RL can help to guide the mode coverage process in DMD more effectively. These allow us to unlock the capacity of the few-step generator by conducting distillation and RL simultaneously. Meanwhile, we design the dynamic distribution guidance and dynamic renoise sampling training strategies to improve the initial distillation process. The experiments demonstrate that DMDR can achieve leading visual quality, prompt coherence among few-step methods, and even exhibit performance that exceeds the multi-step teacher.
>
---
#### [replaced 055] StreamingTalker: Audio-driven 3D Facial Animation with Autoregressive Diffusion Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14223v2](https://arxiv.org/pdf/2511.14223v2)**

> **作者:** Yifan Yang; Zhi Cen; Sida Peng; Xiangwei Chen; Yifu Deng; Xinyu Zhu; Fan Jia; Xiaowei Zhou; Hujun Bao
>
> **摘要:** This paper focuses on the task of speech-driven 3D facial animation, which aims to generate realistic and synchronized facial motions driven by speech inputs. Recent methods have employed audio-conditioned diffusion models for 3D facial animation, achieving impressive results in generating expressive and natural animations. However, these methods process the whole audio sequences in a single pass, which poses two major challenges: they tend to perform poorly when handling audio sequences that exceed the training horizon and will suffer from significant latency when processing long audio inputs. To address these limitations, we propose a novel autoregressive diffusion model that processes input audio in a streaming manner. This design ensures flexibility with varying audio lengths and achieves low latency independent of audio duration. Specifically, we select a limited number of past frames as historical motion context and combine them with the audio input to create a dynamic condition. This condition guides the diffusion process to iteratively generate facial motion frames, enabling real-time synthesis with high-quality results. Additionally, we implemented a real-time interactive demo, highlighting the effectiveness and efficiency of our approach. We will release the code at https://zju3dv.github.io/StreamingTalker/.
>
---
#### [replaced 056] Efficient Document Image Dewarping via Hybrid Deep Learning and Cubic Polynomial Geometry Restoration
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.03145v3](https://arxiv.org/pdf/2501.03145v3)**

> **作者:** Valery Istomin; Oleg Pereziabov; Ilya Afanasyev
>
> **备注:** 21 pages, 4 figures
>
> **摘要:** Camera-captured document images often suffer from geometric distortions caused by paper deformation, perspective distortion, and lens aberrations, significantly reducing OCR accuracy. This study develops an efficient automated method for document image dewarping that balances accuracy with computational efficiency. We propose a hybrid approach combining deep learning for document detection with classical computer vision for geometry restoration. YOLOv8 performs initial document segmentation and mask generation. Subsequently, classical CV techniques construct a topological 2D grid through cubic polynomial interpolation of document boundaries, followed by image remapping to correct nonlinear distortions. A new annotated dataset and open-source framework are provided to facilitate reproducibility and further research. Experimental evaluation against state-of-the-art methods (RectiNet, DocGeoNet, DocTr++) and mobile applications (DocScan, CamScanner, TapScanner) demonstrates superior performance. Our method achieves the lowest median Character Error Rate (CER=0.0235), Levenshtein Distance (LD=27.8), and highest Jaro--Winkler similarity (JW=0.902), approaching the quality of scanned originals. The approach requires significantly fewer computational resources and memory compared to pure deep learning solutions while delivering better OCR readability and geometry restoration quality. The proposed hybrid methodology effectively restores document geometry with computational efficiency superior to existing deep learning approaches, making it suitable for resource-constrained applications while maintaining high-quality document digitization. Project page: https://github.com/HorizonParadox/DRCCBI
>
---
#### [replaced 057] Accelerating Local AI on Consumer GPUs: A Hardware-Aware Dynamic Strategy for YOLOv10s
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.07928v2](https://arxiv.org/pdf/2509.07928v2)**

> **作者:** Mahmudul Islam Masum; Miad Islam
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** As local AI grows in popularity, there is a critical gap between the benchmark performance of object detectors and their practical viability on consumer-grade hardware. While models like YOLOv10s promise real-time speeds, these metrics are typically achieved on high-power, desktop-class GPUs. This paper reveals that on resource-constrained systems, such as laptops with RTX 4060 GPUs, performance is not compute-bound but is instead dominated by system-level bottlenecks, as illustrated by a simple bottleneck test. To overcome this hardware-level constraint, we introduce a Two-Pass Adaptive Inference algorithm, a model-independent approach that requires no architectural changes. This study mainly focuses on adaptive inference strategies and undertakes a comparative analysis of architectural early-exit and resolution-adaptive routing, highlighting their respective trade-offs within a unified evaluation framework. The system uses a fast, low-resolution pass and only escalates to a high-resolution model pass when detection confidence is low. On a 5000-image COCO dataset, our method achieves a 1.85x speedup over a PyTorch Early-Exit baseline, with a modest mAP loss of 5.51%. This work provides a practical and reproducible blueprint for deploying high-performance, real-time AI on consumer-grade devices by shifting the focus from pure model optimization to hardware-aware inference strategies that maximize throughput.
>
---
#### [replaced 058] Detecting Out-of-Distribution Objects through Class-Conditioned Inpainting
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2402.03292v4](https://arxiv.org/pdf/2402.03292v4)**

> **作者:** Quang-Huy Nguyen; Jin Peng Zhou; Zhenzhen Liu; Khanh-Huyen Bui; Kilian Q. Weinberger; Wei-Lun Chao; Dung D. Le
>
> **备注:** Accepted in WACV 2026 (Algorithms track)
>
> **摘要:** Recent object detectors have achieved impressive accuracy in identifying objects seen during training. However, real-world deployment often introduces novel and unexpected objects, referred to as out-of-distribution (OOD) objects, posing significant challenges to model trustworthiness. Modern object detectors are typically overconfident, making it unreliable to use their predictions alone for OOD detection. To address this, we propose leveraging an auxiliary model as a complementary solution. Specifically, we utilize an off-the-shelf text-to-image generative model, such as Stable Diffusion, which is trained with objective functions distinct from those of discriminative object detectors. We hypothesize that this fundamental difference enables the detection of OOD objects by measuring inconsistencies between the models. Concretely, for a given detected object bounding box and its predicted in-distribution class label, we perform class-conditioned inpainting on the image with the object removed. If the object is OOD, the inpainted image is likely to deviate significantly from the original, making the reconstruction error a robust indicator of OOD status. Extensive experiments demonstrate that our approach consistently surpasses existing zero-shot and non-zero-shot OOD detection methods, establishing a robust framework for enhancing object detection systems in dynamic environments.
>
---
#### [replaced 059] Wave-Former: Through-Occlusion 3D Reconstruction via Wireless Shape Completion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14152v2](https://arxiv.org/pdf/2511.14152v2)**

> **作者:** Laura Dodds; Maisy Lam; Waleed Akbar; Yibo Cheng; Fadel Adib
>
> **摘要:** We present Wave-Former, a novel method capable of high-accuracy 3D shape reconstruction for completely occluded, diverse, everyday objects. This capability can open new applications spanning robotics, augmented reality, and logistics. Our approach leverages millimeter-wave (mmWave) wireless signals, which can penetrate common occlusions and reflect off hidden objects. In contrast to past mmWave reconstruction methods, which suffer from limited coverage and high noise, Wave-Former introduces a physics-aware shape completion model capable of inferring full 3D geometry. At the heart of Wave-Former's design is a novel three-stage pipeline which bridges raw wireless signals with recent advancements in vision-based shape completion by incorporating physical properties of mmWave signals. The pipeline proposes candidate geometric surfaces, employs a transformer-based shape completion model designed specifically for mmWave signals, and finally performs entropy-guided surface selection. This enables Wave-Former to be trained using entirely synthetic point-clouds, while demonstrating impressive generalization to real-world data. In head-to-head comparisons with state-of-the-art baselines, Wave-Former raises recall from 54% to 72% while maintaining a high precision of 85%.
>
---
#### [replaced 060] Multi-source-free Domain Adaptation via Uncertainty-aware Adaptive Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2402.06213v2](https://arxiv.org/pdf/2402.06213v2)**

> **作者:** Yaxuan Song; Jianan Fan; Dongnan Liu; Weidong Cai
>
> **备注:** Accepted by ISBI 2024. Code available at https://github.com/YXSong000/UAD
>
> **摘要:** Source-free domain adaptation (SFDA) alleviates the domain discrepancy among data obtained from domains without accessing the data for the awareness of data privacy. However, existing conventional SFDA methods face inherent limitations in medical contexts, where medical data are typically collected from multiple institutions using various equipment. To address this problem, we propose a simple yet effective method, named Uncertainty-aware Adaptive Distillation (UAD) for the multi-source-free unsupervised domain adaptation (MSFDA) setting. UAD aims to perform well-calibrated knowledge distillation from (i) model level to deliver coordinated and reliable base model initialisation and (ii) instance level via model adaptation guided by high-quality pseudo-labels, thereby obtaining a high-performance target domain model. To verify its general applicability, we evaluate UAD on two image-based diagnosis benchmarks among two multi-centre datasets, where our method shows a significant performance gain compared with existing works. The code is available at https://github.com/YXSong000/UAD.
>
---
#### [replaced 061] Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2511.07710v2](https://arxiv.org/pdf/2511.07710v2)**

> **作者:** Jiale Liu; Haoming Zhou; Yishu Zhu; Bingzhi Chen; Yuncheng Jiang
>
> **备注:** 10 pages, 6 figures, accepted by AAAI 2026
>
> **摘要:** Fine-grained image-text alignment is a pivotal challenge in multimodal learning, underpinning key applications such as visual question answering, image captioning, and vision-language navigation. Unlike global alignment, fine-grained alignment requires precise correspondence between localized visual regions and textual tokens, often hindered by noisy attention mechanisms and oversimplified modeling of cross-modal relationships. In this work, we identify two fundamental limitations of existing approaches: the lack of robust intra-modal mechanisms to assess the significance of visual and textual tokens, leading to poor generalization in complex scenes; and the absence of fine-grained uncertainty modeling, which fails to capture the one-to-many and many-to-one nature of region-word correspondences. To address these issues, we propose a unified approach that incorporates significance-aware and granularity-aware modeling and region-level uncertainty modeling. Our method leverages modality-specific biases to identify salient features without relying on brittle cross-modal attention, and represents region features as a mixture of Gaussian distributions to capture fine-grained uncertainty. Extensive experiments on Flickr30K and MS-COCO demonstrate that our approach achieves state-of-the-art performance across various backbone architectures, significantly enhancing the robustness and interpretability of fine-grained image-text alignment.
>
---
#### [replaced 062] MOON: Generative MLLM-based Multimodal Representation Learning for E-commerce Product Understanding
- **分类: cs.CV; cs.AI; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.11999v3](https://arxiv.org/pdf/2508.11999v3)**

> **作者:** Daoze Zhang; Chenghan Fu; Zhanheng Nie; Jianyu Liu; Wanxian Guan; Yuan Gao; Jun Song; Pengjie Wang; Jian Xu; Bo Zheng
>
> **备注:** Accepted by WSDM 2026. 11 pages, 9 figures
>
> **摘要:** With the rapid advancement of e-commerce, exploring general representations rather than task-specific ones has attracted increasing research attention. For product understanding, although existing discriminative dual-flow architectures drive progress in this field, they inherently struggle to model the many-to-one alignment between multiple images and texts of products. Therefore, we argue that generative Multimodal Large Language Models (MLLMs) hold significant potential for improving product representation learning. Nevertheless, achieving this goal still remains non-trivial due to several key challenges: the lack of multimodal and aspect-aware modeling modules in typical LLMs; the common presence of background noise in product images; and the absence of a standard benchmark for evaluation. To address these issues, we propose the first generative MLLM-based model named MOON for product representation learning. Our method (1) employs a guided Mixture-of-Experts (MoE) module for targeted modeling of multimodal and aspect-specific product content; (2) effectively detects core semantic regions in product images to mitigate the distraction and interference caused by background noise; and (3) introduces the specialized negative sampling strategy to increase the difficulty and diversity of negative samples. In addition, we release a large-scale multimodal benchmark MBE for various product understanding tasks. Experimentally, our model demonstrates competitive zero-shot performance on both our benchmark and the public dataset, showcasing strong generalization across various downstream tasks, including cross-modal retrieval, product classification, and attribute prediction. Furthermore, the case study and visualization illustrate the effectiveness of MOON for product understanding.
>
---
#### [replaced 063] DoGCLR: Dominance-Game Contrastive Learning Network for Skeleton-Based Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14179v2](https://arxiv.org/pdf/2511.14179v2)**

> **作者:** Yanshan Li; Ke Ma; Miaomiao Wei; Linhui Dai
>
> **备注:** 14 pages, 7 figures, journal
>
> **摘要:** Existing self-supervised contrastive learning methods for skeleton-based action recognition often process all skeleton regions uniformly, and adopt a first-in-first-out (FIFO) queue to store negative samples, which leads to motion information loss and non-optimal negative sample selection. To address these challenges, this paper proposes Dominance-Game Contrastive Learning network for skeleton-based action Recognition (DoGCLR), a self-supervised framework based on game theory. DoGCLR models the construction of positive and negative samples as a dynamic Dominance Game, where both sample types interact to reach an equilibrium that balances semantic preservation and discriminative strength. Specifically, a spatio-temporal dual weight localization mechanism identifies key motion regions and guides region-wise augmentations to enhance motion diversity while maintaining semantics. In parallel, an entropy-driven dominance strategy manages the memory bank by retaining high entropy (hard) negatives and replacing low-entropy (weak) ones, ensuring consistent exposure to informative contrastive signals. Extensive experiments are conducted on NTU RGB+D and PKU-MMD datasets. On NTU RGB+D 60 X-Sub/X-View, DoGCLR achieves 81.1%/89.4% accuracy, and on NTU RGB+D 120 X-Sub/X-Set, DoGCLR achieves 71.2%/75.5% accuracy, surpassing state-of-the-art methods by 0.1%, 2.7%, 1.1%, and 2.3%, respectively. On PKU-MMD Part I/Part II, DoGCLR performs comparably to the state-of-the-art methods and achieves a 1.9% higher accuracy on Part II, highlighting its strong robustness on more challenging scenarios.
>
---
#### [replaced 064] ToDRE: Effective Visual Token Pruning via Token Diversity and Task Relevance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.18757v2](https://arxiv.org/pdf/2505.18757v2)**

> **作者:** Duo Li; Zuhao Yang; Xiaoqin Zhang; Ling Shao; Shijian Lu
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Visual token pruning aims to compress and prune redundant visual tokens which play a critical role in efficient inference with large vision-language models (LVLMs). However, most existing work estimates visual redundancy using a single metric, such as cross-modal attention or visual token similarity. We show that visual token diversity and task-specific token relevance are two crucial yet orthogonal factors that complement each other in conveying useful information and should therefore be treated separately for more effective visual token pruning. Building upon this insight, we design TODRE, a two-stage and training-free framework that incorporates Token Diversity and task RElevance for effective token compression and efficient LVLM inference. Instead of pruning redundant tokens, we introduce a greedy max-sum diversification algorithm that selects and retains a subset of diverse and representative visual tokens after the vision encoder. On top of that, ToDRE leverages an "information migration" mechanism to eliminate task-irrelevant visual tokens within certain decoder layers of large language model(LLM) to further improve token pruning and LVLM inference. Extensive experiments show that ToDRE prunes 90% of visual tokens after the vision encoder as well as all visual tokens in certain LLM decoder layers, leading to a 2.6x speed-up in total inference time while maintaining 95.0% model performance plus excellent model compatibility.
>
---
#### [replaced 065] Self Pre-training with Topology- and Spatiality-aware Masked Autoencoders for 3D Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2406.10519v3](https://arxiv.org/pdf/2406.10519v3)**

> **作者:** Pengfei Gu; Huimin Li; Yejia Zhang; Chaoli Wang; Danny Z. Chen
>
> **摘要:** Masked Autoencoders (MAEs) have been shown to be effective in pre-training Vision Transformers (ViTs) for natural and medical image analysis problems. By reconstructing missing pixel/voxel information in visible patches, a ViT encoder can aggregate contextual information for downstream tasks. But, existing MAE pre-training methods, which were specifically developed with the ViT architecture, lack the ability to capture geometric shape and spatial information, which is critical for medical image segmentation tasks. In this paper, we propose a novel extension of known MAEs for self pre-training (i.e., models pre-trained on the same target dataset) for 3D medical image segmentation. (1) We propose a new topological loss to preserve geometric shape information by computing topological signatures of both the input and reconstructed volumes, learning geometric shape information. (2) We introduce a pre-text task that predicts the positions of the centers and eight corners of 3D crops, enabling the MAE to aggregate spatial information. (3) We extend the MAE pre-training strategy to a hybrid state-of-the-art (SOTA) medical image segmentation architecture and co-pretrain it alongside the ViT. (4) We develop a fine-tuned model for downstream segmentation tasks by complementing the pre-trained ViT encoder with our pre-trained SOTA model. Extensive experiments on five public 3D segmentation datasets show the effectiveness of our new approach.
>
---
#### [replaced 066] A Hybrid Multimodal Deep Learning Framework for Intelligent Fashion Recommendation
- **分类: cs.IR; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07573v2](https://arxiv.org/pdf/2511.07573v2)**

> **作者:** Kamand Kalashi; Babak Teimourpour
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** The rapid expansion of online fashion platforms has created an increasing demand for intelligent recommender systems capable of understanding both visual and textual cues. This paper proposes a hybrid multimodal deep learning framework for fashion recommendation that jointly addresses two key tasks: outfit compatibility prediction and complementary item retrieval. The model leverages the visual and textual encoders of the CLIP architecture to obtain joint latent representations of fashion items, which are then integrated into a unified feature vector and processed by a transformer encoder. For compatibility prediction, an "outfit token" is introduced to model the holistic relationships among items, achieving an AUC of 0.95 on the Polyvore dataset. For complementary item retrieval, a "target item token" representing the desired item description is used to retrieve compatible items, reaching an accuracy of 69.24% under the Fill-in-the-Blank (FITB) metric. The proposed approach demonstrates strong performance across both tasks, highlighting the effectiveness of multimodal learning for fashion recommendation.
>
---
#### [replaced 067] Spot The Ball: A Benchmark for Visual Social Inference
- **分类: cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2511.00261v2](https://arxiv.org/pdf/2511.00261v2)**

> **作者:** Neha Balamurugan; Sarah Wu; Adam Chun; Gabe Gaw; Cristobal Eyzaguirre; Tobias Gerstenberg
>
> **摘要:** Humans excel at visual social inference, the ability to infer hidden elements of a scene from subtle behavioral cues such as other people's gaze, pose, and orientation. This ability drives everyday social reasoning in humans and is critical for developing more human-like AI agents. We introduce Spot The Ball, a challenging benchmark for evaluating visual social inference in vision-language models (VLMs) using sports as a test domain. The task is to localize a removed sports ball from soccer, basketball, and volleyball images. We present a curated evaluation set with human baselines and a scalable pipeline for generating additional test items. We evaluate four state-of-the-art VLMs (Gemini, GPT, LLaMA, Qwen) using three prompting strategies, finding that humans are consistently two to three times more accurate (20-34%) than models ($\leq$ 17%) across all sports. Our analyses show that models rely on superficial spatial heuristics--such as guessing near the image center or nearby players--while humans leverage social cues like gaze direction and body pose. These findings reveal a persistent human-model gap in visual social reasoning and underscore the need for architectures that explicitly encode structured behavioral cues to achieve robust, human-like inference.
>
---
#### [replaced 068] FairJudge: MLLM Judging for Social Attributes and Prompt Image Alignment
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.22827v2](https://arxiv.org/pdf/2510.22827v2)**

> **作者:** Zahraa Al Sahili; Maryam Fetanat; Maimuna Nowaz; Ioannis Patras; Matthew Purver
>
> **摘要:** Text-to-image (T2I) systems lack simple, reproducible ways to evaluate how well images match prompts and how models treat social attributes. Common proxies -- face classifiers and contrastive similarity -- reward surface cues, lack calibrated abstention, and miss attributes only weakly visible (for example, religion, culture, disability). We present FairJudge, a lightweight protocol that treats instruction-following multimodal LLMs as fair judges. It scores alignment with an explanation-oriented rubric mapped to [-1, 1]; constrains judgments to a closed label set; requires evidence grounded in the visible content; and mandates abstention when cues are insufficient. Unlike CLIP-only pipelines, FairJudge yields accountable, evidence-aware decisions; unlike mitigation that alters generators, it targets evaluation fairness. We evaluate gender, race, and age on FairFace, PaTA, and FairCoT; extend to religion, culture, and disability; and assess profession correctness and alignment on IdenProf, FairCoT-Professions, and our new DIVERSIFY-Professions. We also release DIVERSIFY, a 469-image corpus of diverse, non-iconic scenes. Across datasets, judge models outperform contrastive and face-centric baselines on demographic prediction and improve mean alignment while maintaining high profession accuracy, enabling more reliable, reproducible fairness audits.
>
---
#### [replaced 069] A Style is Worth One Code: Unlocking Code-to-Style Image Generation with Discrete Style Space
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10555v4](https://arxiv.org/pdf/2511.10555v4)**

> **作者:** Huijie Liu; Shuhao Cui; Haoxiang Cao; Shuai Ma; Kai Wu; Guoliang Kang
>
> **备注:** Code: https://github.com/Kwai-Kolors/CoTyle Demo: https://huggingface.co/spaces/Kwai-Kolors/CoTyle Homepage: https://kwai-kolors.github.io/CoTyle/
>
> **摘要:** Innovative visual stylization is a cornerstone of artistic creation, yet generating novel and consistent visual styles remains a significant challenge. Existing generative approaches typically rely on lengthy textual prompts, reference images, or parameter-efficient fine-tuning to guide style-aware image generation, but often struggle with style consistency, limited creativity, and complex style representations. In this paper, we affirm that a style is worth one numerical code by introducing the novel task, code-to-style image generation, which produces images with novel, consistent visual styles conditioned solely on a numerical style code. To date, this field has only been primarily explored by the industry (e.g., Midjourney), with no open-source research from the academic community. To fill this gap, we propose CoTyle, the first open-source method for this task. Specifically, we first train a discrete style codebook from a collection of images to extract style embeddings. These embeddings serve as conditions for a text-to-image diffusion model (T2I-DM) to generate stylistic images. Subsequently, we train an autoregressive style generator on the discrete style embeddings to model their distribution, allowing the synthesis of novel style embeddings. During inference, a numerical style code is mapped to a unique style embedding by the style generator, and this embedding guides the T2I-DM to generate images in the corresponding style. Unlike existing methods, our method offers unparalleled simplicity and diversity, unlocking a vast space of reproducible styles from minimal input. Extensive experiments validate that CoTyle effectively turns a numerical code into a style controller, demonstrating a style is worth one code.
>
---
#### [replaced 070] RN-SDEs: Limited-Angle CT Reconstruction with Residual Null-Space Diffusion Stochastic Differential Equations
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2409.13930v2](https://arxiv.org/pdf/2409.13930v2)**

> **作者:** Jiaqi Guo; Santiago Lopez-Tapia; Wing Shun Li; Yunnan Wu; Marcelo Carignano; Martin Kröger; Vinayak P. Dravid; Igal Szleifer; Vadim Backman; Aggelos K. Katsaggelos
>
> **摘要:** Computed tomography is a widely used imaging modality with applications ranging from medical imaging to material analysis. One major challenge arises from the lack of scanning information at certain angles, leading to distorted CT images with artifacts. This results in an ill-posed problem known as the Limited Angle Computed Tomography (LACT) reconstruction problem. To address this problem, we propose Residual Null-Space Diffusion Stochastic Differential Equations (RN-SDEs), which are a variant of diffusion models that characterize the diffusion process with mean-reverting (MR) stochastic differential equations. To demonstrate the generalizability of RN-SDEs, our experiments are conducted on two different LACT datasets, i.e., ChromSTEM and C4KC-KiTS. Through extensive experiments, we show that by leveraging learned Mean-Reverting SDEs as a prior and emphasizing data consistency using Range-Null Space Decomposition (RNSD) based rectification, RN-SDEs can restore high-quality images from severe degradation and achieve state-of-the-art performance in most LACT tasks. Additionally, we present a quantitative comparison of computational complexity and runtime efficiency, highlighting the superior effectiveness of our proposed approach.
>
---
#### [replaced 071] Scale-invariant brain morphometry: application to sulcal depth
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.05436v2](https://arxiv.org/pdf/2501.05436v2)**

> **作者:** Maxime Dieudonné; Guillaume Auzias; Julien Lefèvre
>
> **备注:** GA and JL contributed equally to this work
>
> **摘要:** The geometry of the human cortex is complex and highly variable, with interactions between brain size, cortical folding, and age well-documented in the literature. However, few studies have explored how global brain size influences morphometry features of the cortical surface derived from anatomical MRI. In this work, we focus on sulcal depth, an imaging phenotype that has gained attention in both basic research and clinical applications. We make key contributions to the field by: 1) providing the first quantitative analysis of the influence of brain size on sulcal depth measurements; 2) introducing a novel, scale-invariant method for sulcal depth estimation based on an original formalization of the problem; 3) presenting a validation framework and sharing our code and benchmark data with the community; and 4) demonstrating the biological relevance of our new sulcal depth measure using a large sample of 1,987 subjects spanning the developmental period from 26 weeks post-conception to adulthood.
>
---
#### [replaced 072] FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.15488v4](https://arxiv.org/pdf/2502.15488v4)**

> **作者:** Jiangyong Yu; Changyong Shu; Sifan Zhou; Zichen Yu; Xing Hu; Yan Chen; Dawei Yang
>
> **备注:** This paper is acceptted by AAAI 2026
>
> **摘要:** Camera-based multi-view 3D detection is crucial for autonomous driving. PETR and its variants (PETRs) excel in benchmarks but face deployment challenges due to high computational cost and memory footprint. Quantization is an effective technique for compressing deep neural networks by reducing the bit width of weights and activations. However, directly applying existing quantization methods to PETRs leads to severe accuracy degradation. This issue primarily arises from two key challenges: (1) significant magnitude disparity between multi-modal features-specifically, image features and camera-ray positional embeddings (PE), and (2) the inefficiency and approximation error of quantizing non-linear operators, which commonly rely on hardware-unfriendly computations. In this paper, we propose FQ-PETR, a fully quantized framework for PETRs, featuring three key innovations: (1) Quantization-Friendly LiDAR-ray Position Embedding (QFPE): Replacing multi-point sampling with LiDAR-prior-guided single-point sampling and anchor-based embedding eliminates problematic non-linearities (e.g., inverse-sigmoid) and aligns PE scale with image features, preserving accuracy. (2) Dual-Lookup Table (DULUT): This algorithm approximates complex non-linear functions using two cascaded linear LUTs, achieving high fidelity with minimal entries and no specialized hardware. (3) Quantization After Numerical Stabilization (QANS): Performing quantization after softmax numerical stabilization mitigates attention distortion from large inputs. On PETRs (e.g. PETR, StreamPETR, PETRv2, MV2d), FQ-PETR under W8A8 achieves near-floating-point accuracy (1% degradation) while reducing latency by up to 75%, significantly outperforming existing PTQ and QAT baselines.
>
---
#### [replaced 073] Euclid's Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.24473v3](https://arxiv.org/pdf/2509.24473v3)**

> **作者:** Shijie Lian; Changti Wu; Laurence Tianruo Yang; Hang Yuan; Bin Yu; Lei Zhang; Kai Chen
>
> **摘要:** Spatial intelligence spans a rich suite of abilities, including visualising and transforming shapes, mentally rotating objects, judging relational positions and containment, and estimating numerosity. However, it still remains a critical unresolved challenge for Multimodal Large Language Models (MLLMs). To fill this gap, we propose to treat Euclidean geometry problem-solving as a surrogate task. Specifically, we meticulously constructed a curated multimodal dataset, called Euclid30K, comprising approximately 30K plane and solid geometry problems. Furthermore, to enable the model to learn and apply Euclidean principles from these geometry problems, we fine-tuned seven model variants (spanning 3--72B parameters) from the Qwen2.5VL, Qwen3VL, and RoboBrain2.0 families using Group Relative Policy Optimization (GRPO), inspiring the models to identify shapes, count, and relate entities, and perform multi-step deductive reasoning using Euclidean principles. Our experiments demonstrate that the resulting models achieve substantial zero-shot gains across four spatial reasoning benchmarks (Super-CLEVR, Omni3DBench, VSI-Bench, and MindCube) without any task-specific adaptations. Notably, after training on the Euclid30K, the mean VSI-Bench accuracy rose from 36.6\% to 41.8\% (+5.2\%), and the mean MindCube accuracy rose from 31.4\% to 38.1\% (+6.7\%). To our knowledge, this is the first systematic study showing that geometry-centric fine-tuning can confer vision-language models with broadly transferable spatial skills. Code and Euclid30K dataset can be found in \href{https://zgca-ai4edu.github.io/Euclids_Gift}{this}.
>
---
#### [replaced 074] Causal Tracing of Object Representations in Large Vision Language Models: Mechanistic Interpretability and Hallucination Mitigation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05923v3](https://arxiv.org/pdf/2511.05923v3)**

> **作者:** Qiming Li; Zekai Ye; Xiaocheng Feng; Weihong Zhong; Weitao Ma; Xiachong Feng
>
> **备注:** AAAI2026 Oral
>
> **摘要:** Despite the remarkable advancements of Large Vision-Language Models (LVLMs), the mechanistic interpretability remains underexplored. Existing analyses are insufficiently comprehensive and lack examination covering visual and textual tokens, model components, and the full range of layers. This limitation restricts actionable insights to improve the faithfulness of model output and the development of downstream tasks, such as hallucination mitigation. To address this limitation, we introduce Fine-grained Cross-modal Causal Tracing (FCCT) framework, which systematically quantifies the causal effects on visual object perception. FCCT conducts fine-grained analysis covering the full range of visual and textual tokens, three core model components including multi-head self-attention (MHSA), feed-forward networks (FFNs), and hidden states, across all decoder layers. Our analysis is the first to demonstrate that MHSAs of the last token in middle layers play a critical role in aggregating cross-modal information, while FFNs exhibit a three-stage hierarchical progression for the storage and transfer of visual object representations. Building on these insights, we propose Intermediate Representation Injection (IRI), a training-free inference-time technique that reinforces visual object information flow by precisely intervening on cross-modal representations at specific components and layers, thereby enhancing perception and mitigating hallucination. Consistent improvements across five widely used benchmarks and LVLMs demonstrate IRI achieves state-of-the-art performance, while preserving inference speed and other foundational performance.
>
---
#### [replaced 075] Drifting Away from Truth: GenAI-Driven News Diversity Challenges LVLM-Based Misinformation Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12711v3](https://arxiv.org/pdf/2508.12711v3)**

> **作者:** Fanxiao Li; Jiaying Wu; Tingchao Fu; Yunyun Dong; Bingbing Song; Wei Zhou
>
> **摘要:** The proliferation of multimodal misinformation poses growing threats to public discourse and societal trust. While Large Vision-Language Models (LVLMs) have enabled recent progress in multimodal misinformation detection (MMD), the rise of generative AI (GenAI) tools introduces a new challenge: GenAI-driven news diversity, characterized by highly varied and complex content. We show that this diversity induces multi-level drift, comprising (1) model-level misperception drift, where stylistic variations disrupt a model's internal reasoning, and (2) evidence-level drift, where expression diversity degrades the quality or relevance of retrieved external evidence. These drifts significantly degrade the robustness of current LVLM-based MMD systems. To systematically study this problem, we introduce DriftBench, a large-scale benchmark comprising 16,000 news instances across six categories of diversification. We design three evaluation tasks: (1) robustness of truth verification under multi-level drift; (2) susceptibility to adversarial evidence contamination generated by GenAI; and (3) analysis of reasoning consistency across diverse inputs. Experiments with six state-of-the-art LVLM-based detectors show substantial performance drops (average F1 -14.8%) and increasingly unstable reasoning traces, with even more severe failures under adversarial evidence injection. Our findings uncover fundamental vulnerabilities in existing MMD systems and suggest an urgent need for more resilient approaches in the GenAI era.
>
---
#### [replaced 076] AdCare-VLM: Towards a Unified and Pre-aligned Latent Representation for Healthcare Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.00275v3](https://arxiv.org/pdf/2505.00275v3)**

> **作者:** Md Asaduzzaman Jabin; Hanqi Jiang; Yiwei Li; Patrick Kaggwa; Eugene Douglass; Juliet N. Sekandi; Tianming Liu
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: 7th International Workshop on Large Scale Holistic Video Understanding: Toward Video Foundation Models
>
> **摘要:** Chronic diseases, including diabetes, hypertension, asthma, HIV-AIDS, epilepsy, and tuberculosis, necessitate rigorous adherence to medication to avert disease progression, manage symptoms, and decrease mortality rates. Adherence is frequently undermined by factors including patient behavior, caregiver support, elevated medical costs, and insufficient healthcare infrastructure. We propose AdCare-VLM, a specialized LLaVA-based multimodal large vision language model (LVLM) by introducing a unified visual latent space with pre-alignment to facilitate visual question answering (VQA) concerning medication adherence through patient videos. We employ a private dataset comprising 806 custom-annotated tuberculosis (TB) medication monitoring videos, which have been labeled by clinical experts, to fine-tune the model for adherence pattern detection. We present LLM-TB-VQA, a detailed medical adherence VQA dataset that encompasses positive, negative, and ambiguous adherence cases. Our method identifies correlations between visual features, such as the clear visibility of the patient's face, medication, water intake, and the act of ingestion, and their associated medical concepts in captions. This facilitates the integration of aligned visual-linguistic representations and improves multimodal interactions. Experimental results indicate that our method surpasses parameter-efficient fine-tuning (PEFT) enabled VLM models, such as LLaVA-V1.5 and Chat-UniVi, with absolute improvements ranging from 3.1% to 3.54% across pre-trained, regular, and low-rank adaptation (LoRA) configurations. Comprehensive ablation studies and attention map visualizations substantiate our approach, enhancing interpretability.
>
---
#### [replaced 077] Other Vehicle Trajectories Are Also Needed: A Driving World Model Unifies Ego-Other Vehicle Trajectories in Video Latent Space
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.09215v4](https://arxiv.org/pdf/2503.09215v4)**

> **作者:** Jian Zhu; Zhengyu Jia; Tian Gao; Jiaxin Deng; Shidi Li; Lang Zhang; Fu Liu; Peng Jia; Xianpeng Lang
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Advanced end-to-end autonomous driving systems predict other vehicles' motions and plan ego vehicle's trajectory. The world model that can foresee the outcome of the trajectory has been used to evaluate the autonomous driving system. However, existing world models predominantly emphasize the trajectory of the ego vehicle and leave other vehicles uncontrollable. This limitation hinders their ability to realistically simulate the interaction between the ego vehicle and the driving scenario. In this paper, we propose a driving World Model named EOT-WM, unifying Ego-Other vehicle Trajectories in videos for driving simulation. Specifically, it remains a challenge to match multiple trajectories in the BEV space with each vehicle in the video to control the video generation. We first project ego-other vehicle trajectories in the BEV space into the image coordinate for vehicle-trajectory match via pixel positions. Then, trajectory videos are encoded by the Spatial-Temporal Variational Auto Encoder to align with driving video latents spatially and temporally in the unified visual space. A trajectory-injected diffusion Transformer is further designed to denoise the noisy video latents for video generation with the guidance of ego-other vehicle trajectories. In addition, we propose a metric based on control latent similarity to evaluate the controllability of trajectories. Extensive experiments are conducted on the nuScenes dataset, and the proposed model outperforms the state-of-the-art method by 30% in FID and 55% in FVD. The model can also predict unseen driving scenes with self-produced trajectories.
>
---
#### [replaced 078] Uni-Hema: Unified Model for Digital Hematopathology
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.13889v2](https://arxiv.org/pdf/2511.13889v2)**

> **作者:** Abdul Rehman; Iqra Rasool; Ayisha Imran; Mohsen Ali; Waqas Sultani
>
> **摘要:** Digital hematopathology requires cell-level analysis across diverse disease categories, including malignant disorders (e.g., leukemia), infectious conditions (e.g., malaria), and non-malignant red blood cell disorders (e.g., sickle cell disease). Whether single-task, vision-language, WSI-optimized, or single-cell hematology models, these approaches share a key limitation, they cannot provide unified, multi-task, multi-modal reasoning across the complexities of digital hematopathology. To overcome these limitations, we propose Uni-Hema, a multi-task, unified model for digital hematopathology integrating detection, classification, segmentation, morphology prediction, and reasoning across multiple diseases. Uni-Hema leverages 46 publicly available datasets, encompassing over 700K images and 21K question-answer pairs, and is built upon Hema-Former, a multimodal module that bridges visual and textual representations at the hierarchy level for the different tasks (detection, classification, segmentation, morphology, mask language modeling and visual question answer) at different granularity. Extensive experiments demonstrate that Uni-Hema achieves comparable or superior performance to train on a single-task and single dataset models, across diverse hematological tasks, while providing interpretable, morphologically relevant insights at the single-cell level. Our framework establishes a new standard for multi-task and multi-modal digital hematopathology. The code will be made publicly available.
>
---
#### [replaced 079] Deep Spectral Prior
- **分类: cs.CV; math.NA**

- **链接: [https://arxiv.org/pdf/2505.19873v2](https://arxiv.org/pdf/2505.19873v2)**

> **作者:** Yanqi Cheng; Xuxiang Zhao; Tieyong Zeng; Pietro Lio; Carola-Bibiane Schönlieb; Angelica I Aviles-Rivero
>
> **摘要:** We introduce the Deep Spectral Prior (DSP), a new framework for unsupervised image reconstruction that operates entirely in the complex frequency domain. Unlike the Deep Image Prior (DIP), which optimises pixel-level errors and is highly sensitive to overfitting, DSP performs joint learning of amplitude and phase to capture the full spectral structure of images. We derive a rigorous theoretical characterisation of DSP's optimisation dynamics, proving that it follows frequency-dependent descent trajectories that separate informative low-frequency modes from stochastic high-frequency noise. This spectral mode separation explains DSP's self-regularising behaviour and, for the first time, formally establishes the elimination of DIP's major limitation-its reliance on manual early stopping. Moreover, DSP induces an implicit projection onto a frequency-consistent manifold, ensuring convergence to stable, physically plausible reconstructions without explicit priors or supervision. Extensive experiments on denoising, inpainting, and deblurring demonstrate that DSP consistently surpasses DIP and other unsupervised baselines, achieving superior fidelity, robustness, and theoretical interpretability within a unified, unsupervised data-free framework.
>
---
#### [replaced 080] Class-Aware PillarMix: Can Mixed Sample Data Augmentation Enhance 3D Object Detection with Radar Point Clouds?
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2503.02687v3](https://arxiv.org/pdf/2503.02687v3)**

> **作者:** Miao Zhang; Sherif Abdulatif; Benedikt Loesch; Marco Altmann; Bin Yang
>
> **备注:** 8 pages, 6 figures, 4 tables, accepted to 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025). Code: https://github.com/boschresearch/CAPMIX
>
> **摘要:** Due to the significant effort required for data collection and annotation in 3D perception tasks, mixed sample data augmentation (MSDA) has been widely studied to generate diverse training samples by mixing existing data. Recently, many MSDA techniques have been developed for point clouds, but they mainly target LiDAR data, leaving their application to radar point clouds largely unexplored. In this paper, we examine the feasibility of applying existing MSDA methods to radar point clouds and identify several challenges in adapting these techniques. These obstacles stem from the radar's irregular angular distribution, deviations from a single-sensor polar layout in multi-radar setups, and point sparsity. To address these issues, we propose Class-Aware PillarMix (CAPMix), a novel MSDA approach that applies MixUp at the pillar level in 3D point clouds, guided by class labels. Unlike methods that rely a single mix ratio to the entire sample, CAPMix assigns an independent ratio to each pillar, boosting sample diversity. To account for the density of different classes, we use class-specific distributions: for dense objects (e.g., large vehicles), we skew ratios to favor points from another sample, while for sparse objects (e.g., pedestrians), we sample more points from the original. This class-aware mixing retains critical details and enriches each sample with new information, ultimately generating more diverse training data. Experimental results demonstrate that our method not only significantly boosts performance but also outperforms existing MSDA approaches across two datasets (Bosch Street and K-Radar). We believe that this straightforward yet effective approach will spark further investigation into MSDA techniques for radar data.
>
---
#### [replaced 081] UINO-FSS: Unifying Representation Learning and Few-shot Segmentation via Hierarchical Distillation and Mamba-HyperCorrelation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.15669v3](https://arxiv.org/pdf/2504.15669v3)**

> **作者:** Wei Zhuo; Zhiyue Tang; Wufeng Xue; Hao Ding; Junkai Ji; Linlin Shen
>
> **摘要:** Few-shot semantic segmentation has attracted growing interest for its ability to generalize to novel object categories using only a few annotated samples. To address data scarcity, recent methods incorporate multiple foundation models to improve feature transferability and segmentation performance. However, they often rely on dual-branch architectures that combine pre-trained encoders to leverage complementary strengths, a design that limits flexibility and efficiency. This raises a fundamental question: can we build a unified model that integrates knowledge from different foundation architectures? Achieving this is, however, challenging due to the misalignment between class-agnostic segmentation capabilities and fine-grained discriminative representations. To this end, we present UINO-FSS, a novel framework built on the key observation that early-stage DINOv2 features exhibit distribution consistency with SAM's output embeddings. This consistency enables the integration of both models' knowledge into a single-encoder architecture via coarse-to-fine multimodal distillation. In particular, our segmenter consists of three core components: a bottleneck adapter for embedding alignment, a meta-visual prompt generator that leverages dense similarity volumes and semantic embeddings, and a mask decoder. Using hierarchical cross-model distillation, we effectively transfer SAM's knowledge into the segmenter, further enhanced by Mamba-based 4D correlation mining on support-query pairs. Extensive experiments on PASCAL-5$^i$ and COCO-20$^i$ show that UINO-FSS achieves new state-of-the-art results under the 1-shot setting, with mIoU of 80.6 (+3.8%) on PASCAL-5$^i$ and 64.5 (+4.1%) on COCO-20$^i$, demonstrating the effectiveness of our unified approach.
>
---
#### [replaced 082] Wonder3D++: Cross-domain Diffusion for High-fidelity 3D Generation from a Single Image
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.01767v2](https://arxiv.org/pdf/2511.01767v2)**

> **作者:** Yuxiao Yang; Xiao-Xiao Long; Zhiyang Dou; Cheng Lin; Yuan Liu; Qingsong Yan; Yuexin Ma; Haoqian Wang; Zhiqiang Wu; Wei Yin
>
> **备注:** 21 pages, 19 figures, accepted by TPAMI
>
> **摘要:** In this work, we introduce \textbf{Wonder3D++}, a novel method for efficiently generating high-fidelity textured meshes from single-view images. Recent methods based on Score Distillation Sampling (SDS) have shown the potential to recover 3D geometry from 2D diffusion priors, but they typically suffer from time-consuming per-shape optimization and inconsistent geometry. In contrast, certain works directly produce 3D information via fast network inferences, but their results are often of low quality and lack geometric details. To holistically improve the quality, consistency, and efficiency of single-view reconstruction tasks, we propose a cross-domain diffusion model that generates multi-view normal maps and the corresponding color images. To ensure the consistency of generation, we employ a multi-view cross-domain attention mechanism that facilitates information exchange across views and modalities. Lastly, we introduce a cascaded 3D mesh extraction algorithm that drives high-quality surfaces from the multi-view 2D representations in only about $3$ minute in a coarse-to-fine manner. Our extensive evaluations demonstrate that our method achieves high-quality reconstruction results, robust generalization, and good efficiency compared to prior works. Code available at https://github.com/xxlong0/Wonder3D/tree/Wonder3D_Plus.
>
---
#### [replaced 083] Conflict Adaptation in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [https://arxiv.org/pdf/2510.24804v2](https://arxiv.org/pdf/2510.24804v2)**

> **作者:** Xiaoyang Hu
>
> **备注:** Workshop on Interpreting Cognition in Deep Learning Models at NeurIPS 2025
>
> **摘要:** A signature of human cognitive control is conflict adaptation: improved performance on a high-conflict trial following another high-conflict trial. This phenomenon offers an account for how cognitive control, a scarce resource, is recruited. Using a sequential Stroop task, we find that 12 of 13 vision-language models (VLMs) tested exhibit behavior consistent with conflict adaptation, with the lone exception likely reflecting a ceiling effect. To understand the representational basis of this behavior, we use sparse autoencoders (SAEs) to identify task-relevant supernodes in InternVL 3.5 4B. Partially overlapping supernodes emerge for text and color in both early and late layers, and their relative sizes mirror the automaticity asymmetry between reading and color naming in humans. We further isolate a conflict-modulated supernode in layers 24-25 whose ablation significantly increases Stroop errors while minimally affecting congruent trials.
>
---
#### [replaced 084] HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12969v2](https://arxiv.org/pdf/2511.12969v2)**

> **作者:** Ziqiao Weng; Yaoyu Fang; Jiahe Qian; Xinkun Wang; Lee AD Cooper; Weidong Cai; Bo Zhou
>
> **备注:** Accepted to AAAI 2026. 7 pages (main text), 12 pages total including references and supplementary material. 6 figures
>
> **摘要:** Spatial transcriptomics (ST) bridges gene expression and tissue morphology but faces clinical adoption barriers due to technical complexity and prohibitive costs. While computational methods predict gene expression from H&E-stained whole-slide images (WSIs), existing approaches often fail to capture the intricate biological heterogeneity within spots and are susceptible to morphological noise when integrating contextual information from surrounding tissue. To overcome these limitations, we propose HiFusion, a novel deep learning framework that integrates two complementary components. First, we introduce the Hierarchical Intra-Spot Modeling module that extracts fine-grained morphological representations through multi-resolution sub-patch decomposition, guided by a feature alignment loss to ensure semantic consistency across scales. Concurrently, we present the Context-aware Cross-scale Fusion module, which employs cross-attention to selectively incorporate biologically relevant regional context, thereby enhancing representational capacity. This architecture enables comprehensive modeling of both cellular-level features and tissue microenvironmental cues, which are essential for accurate gene expression prediction. Extensive experiments on two benchmark ST datasets demonstrate that HiFusion achieves state-of-the-art performance across both 2D slide-wise cross-validation and more challenging 3D sample-specific scenarios. These results underscore HiFusion's potential as a robust, accurate, and scalable solution for ST inference from routine histopathology.
>
---
#### [replaced 085] Beacon2Science: Enhancing STEREO/HI beacon data with machine learning for efficient CME tracking
- **分类: physics.space-ph; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.15288v2](https://arxiv.org/pdf/2503.15288v2)**

> **作者:** Justin Le Louëdec; Maike Bauer; Tanja Amerstorfer; Jackie A. Davies
>
> **备注:** 25 pages, 11 figures, 1 tables, submitted to AGU Space Weather on 14th March 2025, accepted 05 June 2025, published 15 July 2025
>
> **摘要:** Observing and forecasting coronal mass ejections (CME) in real-time is crucial due to the strong geomagnetic storms they can generate that can have a potentially damaging effect, for example, on satellites and electrical devices. With its near-real-time availability, STEREO/HI beacon data is the perfect candidate for early forecasting of CMEs. However, previous work concluded that CME arrival prediction based on beacon data could not achieve the same accuracy as with high-resolution science data due to data gaps and lower quality. We present our novel machine-learning pipeline entitled ``Beacon2Science'', bridging the gap between beacon and science data to improve CME tracking. Through this pipeline, we first enhance the quality (signal-to-noise ratio and spatial resolution) of beacon data. We then increase the time resolution of enhanced beacon images through learned interpolation to match science data's 40-minute resolution. We maximize information coherence between consecutive frames with adapted model architecture and loss functions through the different steps. The improved beacon images are comparable to science data, showing better CME visibility than the original beacon data. Furthermore, we compare CMEs tracked in beacon, enhanced beacon, and science images. The tracks extracted from enhanced beacon data are closer to those from science images, with a mean average error of $\sim 0.5 ^\circ$ of elongation compared to $1^\circ$ with original beacon data. The work presented in this paper paves the way for its application to forthcoming missions such as Vigil and PUNCH.
>
---
#### [replaced 086] A Denoising Framework for Real-World Ultra-Low-Dose Lung CT Images Based on an Image Purification Strategy
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.07492v3](https://arxiv.org/pdf/2510.07492v3)**

> **作者:** Guoliang Gong; Man Yu
>
> **摘要:** Computed Tomography (CT) is a vital diagnostic tool in clinical practice, yet the health risks associated with ionizing radiation cannot be overlooked. Low-dose CT (LDCT) helps mitigate radiation exposure but simultaneously leads to reduced image quality. Consequently, researchers have sought to reconstruct clear images from LDCT scans using artificial intelligence-based image enhancement techniques. However, these studies typically rely on synthetic LDCT images for algorithm training, which introduces significant domain-shift issues and limits the practical effectiveness of these algorithms in real-world scenarios. To address this challenge, we constructed a real-world paired lung dataset, referred to as Patient-uLDCT (ultra-low-dose CT), by performing multiple scans on volunteers. The radiation dose for the low-dose images in this dataset is only 2% of the normal dose, substantially lower than the conventional 25% low-dose and 10% ultra-low-dose levels. Furthermore, to resolve the anatomical misalignment between normal-dose and uLDCT images caused by respiratory motion during acquisition, we propose a novel purification strategy to construct corresponding aligned image pairs. Finally, we introduce a Frequency-domain Flow Matching model (FFM) that achieves excellent image reconstruction performance. Code is available at https://github.com/MonkeyDadLufy/flow-matching.
>
---
#### [replaced 087] Verb Mirage: Unveiling and Assessing Verb Concept Hallucinations in Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.04939v2](https://arxiv.org/pdf/2412.04939v2)**

> **作者:** Zehao Wang; Xinpeng Liu; Xiaoqian Wu; Yudonglin Zhang; Zhou Fang; Yifan Fang; Junfu Pu; Cewu Lu; Yong-Lu Li
>
> **备注:** Accepted by AAAI-26
>
> **摘要:** Multimodal Large Language Models (MLLMs) have garnered significant attention recently and demonstrate outstanding capabilities in various tasks such as OCR, VQA, captioning, $\textit{etc}$. However, hallucination remains a persistent issue. While numerous methods have been proposed to mitigate hallucinations, achieving notable improvements, these methods primarily focus on mitigating hallucinations about $\textbf{object/noun-related}$ concepts. Verb concepts, crucial for understanding human actions, have been largely overlooked. In this paper, to the best of our knowledge, we are the $\textbf{first}$ to investigate the $\textbf{verb hallucination}$ phenomenon of MLLMs from various perspectives. Our findings reveal that most state-of-the-art MLLMs suffer from severe verb hallucination. To assess the effectiveness of existing mitigation methods for object concept hallucination on verb hallucination, we evaluated these methods and found that they do not effectively address verb hallucination. To address this issue, we propose a novel rich verb knowledge-based tuning method to mitigate verb hallucination. The experiment results demonstrate that our method significantly reduces hallucinations related to verbs.
>
---
#### [replaced 088] UniAV: Unified Audio-Visual Perception for Multi-Task Video Event Localization
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2404.03179v3](https://arxiv.org/pdf/2404.03179v3)**

> **作者:** Tiantian Geng; Teng Wang; Jinming Duan; Yanfu Zhang; Weili Guan; Feng Zheng; Ling shao
>
> **备注:** Published on IEEE TPAMI
>
> **摘要:** Video event localization tasks include temporal action localization (TAL), sound event detection (SED) and audio-visual event localization (AVEL). Existing methods tend to over-specialize on individual tasks, neglecting the equal importance of these different events for a complete understanding of video content. In this work, we aim to develop a unified framework to solve TAL, SED and AVEL tasks together to facilitate holistic video understanding. However, it is challenging since different tasks emphasize distinct event characteristics and there are substantial disparities in existing task-specific datasets (size/domain/duration). It leads to unsatisfactory results when applying a naive multi-task strategy. To tackle the problem, we introduce UniAV, a Unified Audio-Visual perception network to effectively learn and share mutually beneficial knowledge across tasks and modalities. Concretely, we propose a unified audio-visual encoder to derive generic representations from multiple temporal scales for videos from all tasks. Meanwhile, task-specific experts are designed to capture the unique knowledge specific to each task. Besides, instead of using separate prediction heads, we develop a novel unified language-aware classifier by utilizing semantic-aligned task prompts, enabling our model to flexibly localize various instances across tasks with an impressive open-set ability to localize novel categories. Extensive experiments demonstrate that UniAV, with its unified architecture, significantly outperforms both single-task models and the naive multi-task baseline across all three tasks. It achieves superior or on-par performances compared to the state-of-the-art task-specific methods on ActivityNet 1.3, DESED and UnAV-100 benchmarks.
>
---
#### [replaced 089] Arbitrary-Scale 3D Gaussian Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.16467v2](https://arxiv.org/pdf/2508.16467v2)**

> **作者:** Huimin Zeng; Yue Bai; Yun Fu
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Existing 3D Gaussian Splatting (3DGS) super-resolution methods typically perform high-resolution (HR) rendering of fixed scale factors, making them impractical for resource-limited scenarios. Directly rendering arbitrary-scale HR views with vanilla 3DGS introduces aliasing artifacts due to the lack of scale-aware rendering ability, while adding a post-processing upsampler for 3DGS complicates the framework and reduces rendering efficiency. To tackle these issues, we build an integrated framework that incorporates scale-aware rendering, generative prior-guided optimization, and progressive super-resolving to enable 3D Gaussian super-resolution of arbitrary scale factors with a single 3D model. Notably, our approach supports both integer and non-integer scale rendering to provide more flexibility. Extensive experiments demonstrate the effectiveness of our model in rendering high-quality arbitrary-scale HR views (6.59 dB PSNR gain over 3DGS) with a single model. It preserves structural consistency with LR views and across different scales, while maintaining real-time rendering speed (85 FPS at 1080p).
>
---
#### [replaced 090] RetinexDual: Retinex-based Dual Nature Approach for Generalized Ultra-High-Definition Image Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.04797v3](https://arxiv.org/pdf/2508.04797v3)**

> **作者:** Mohab Kishawy; Ali Abdellatif Hussein; Jun Chen
>
> **摘要:** Advancements in image sensing have elevated the importance of Ultra-High-Definition Image Restoration (UHD IR). Traditional methods, such as extreme downsampling or transformation from the spatial to the frequency domain, encounter significant drawbacks: downsampling induces irreversible information loss in UHD images, while our frequency analysis reveals that pure frequency-domain approaches are ineffective for spatially confined image artifacts, primarily due to the loss of degradation locality. To overcome these limitations, we present RetinexDual, a novel Retinex theory-based framework designed for generalized UHD IR tasks. RetinexDual leverages two complementary sub-networks: the Scale-Attentive maMBA (SAMBA) and the Frequency Illumination Adaptor (FIA). SAMBA, responsible for correcting the reflectance component, utilizes a coarse-to-fine mechanism to overcome the causal modeling of mamba, which effectively reduces artifacts and restores intricate details. On the other hand, FIA ensures precise correction of color and illumination distortions by operating in the frequency domain and leveraging the global context provided by it. Evaluating RetinexDual on four UHD IR tasks, namely deraining, deblurring, dehazing, and Low-Light Image Enhancement (LLIE), shows that it outperforms recent methods qualitatively and quantitatively. Ablation studies demonstrate the importance of employing distinct designs for each branch in RetinexDual, as well as the effectiveness of its various components.
>
---
#### [replaced 091] IWR-Bench: Can LVLMs reconstruct interactive webpage from a user interaction video?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.24709v3](https://arxiv.org/pdf/2509.24709v3)**

> **作者:** Yang Chen; Minghao Liu; Yufan Shen; Yunwen Li; Tianyuan Huang; Xinyu Fang; Tianyu Zheng; Wenxuan Huang; Cheng Yang; Daocheng Fu; Jianbiao Mei; Rong Wu; Yunfei Zhao; Licheng Wen; Xuemeng Yang; Song Mao; Qunshu Lin; Zhi Yu; Yongliang Shen; Yu Qiao; Botian Shi
>
> **摘要:** The webpage-to-code task requires models to understand visual representations of webpages and generate corresponding code. However, existing benchmarks primarily focus on static screenshot-to-code tasks, thereby overlooking the dynamic interactions fundamental to real-world web applications. To address this limitation, this paper introduces IWR-Bench, a novel benchmark for evaluating the capabilities of Large Vision-Language Models (LVLMs) in interactive webpage reconstruction from video. IWR-Bench comprises 113 meticulously curated tasks from 100 real-world websites, with 1,001 actions and featuring diverse interaction complexities (e.g., web games), visual styles, and domains. Aligning with standard web development practices, each task includes not only user interaction videos but also all crawled static assets (e.g., images, videos). This benchmark evaluates models on two fundamental challenges: comprehensive multi-modal reasoning to infer interaction logic from video and assets, and advanced code generation to translate this logic into functional code. An agent-as-a-judge framework with a comprehensive metric system automatically assesses the functional correctness and visual fidelity of generated webpages. Extensive experiments on 28 LVLMs reveal a significant challenge: the best model achieves an overall score of only 36.35%, as functional correctness (24.39% IFS) lags significantly behind visual fidelity (64.25% VFS). These results highlight critical limitations in current models' ability to reason about temporal dynamics and synthesize event-driven logic, establishing IWR-Bench as a challenging frontier for vision-language research. The benchmark and evaluation code will be made publicly available at https://github.com/SIGMME/IWR-Bench.
>
---
#### [replaced 092] Point Cloud Quantization through Multimodal Prompting for 3D Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12079v2](https://arxiv.org/pdf/2511.12079v2)**

> **作者:** Hongxuan Li; Wencheng Zhu; Huiying Xu; Xinzhong Zhu; Pengfei Zhu
>
> **备注:** Accepted by AAAI 2026. 11 pages, 7 figures
>
> **摘要:** Vector quantization has emerged as a powerful tool in large-scale multimodal models, unifying heterogeneous representations through discrete token encoding. However, its effectiveness hinges on robust codebook design. Current prototype-based approaches relying on trainable vectors or clustered centroids fall short in representativeness and interpretability, even as multimodal alignment demonstrates its promise in vision-language models. To address these limitations, we propose a simple multimodal prompting-driven quantization framework for point cloud analysis. Our methodology is built upon two core insights: 1) Text embeddings from pre-trained models inherently encode visual semantics through many-to-one contrastive alignment, naturally serving as robust prototype priors; and 2) Multimodal prompts enable adaptive refinement of these prototypes, effectively mitigating vision-language semantic gaps. The framework introduces a dual-constrained quantization space, enforced by compactness and separation regularization, which seamlessly integrates visual and prototype features, resulting in hybrid representations that jointly encode geometric and semantic information. Furthermore, we employ Gumbel-Softmax relaxation to achieve differentiable discretization while maintaining quantization sparsity. Extensive experiments on the ModelNet40 and ScanObjectNN datasets clearly demonstrate the superior effectiveness of the proposed method.
>
---
#### [replaced 093] Event Stream Filtering via Probability Flux Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.07503v2](https://arxiv.org/pdf/2504.07503v2)**

> **作者:** Jinze Chen; Wei Zhai; Yang Cao; Bin Li; Zheng-Jun Zha
>
> **摘要:** Event cameras asynchronously capture brightness changes with microsecond latency, offering exceptional temporal precision but suffering from severe noise and signal inconsistencies. Unlike conventional signals, events carry state information through polarities and process information through inter-event time intervals. However, existing event filters often ignore the latter, producing outputs that are sparser than the raw input and limiting the reconstruction of continuous irradiance dynamics. We propose the Event Density Flow Filter (EDFilter), a framework that models event generation as threshold-crossing probability fluxes arising from the stochastic diffusion of irradiance trajectories. EDFilter performs nonparametric, kernel-based estimation of probability flux and reconstructs the continuous event density flow using an O(1) recursive solver, enabling real-time processing. The Rotary Event Dataset (RED), featuring microsecond-resolution ground-truth irradiance flow under controlled illumination is also presented for event quality evaluation. Experiments demonstrate that EDFilter achieves high-fidelity, physically interpretable event denoising and motion reconstruction.
>
---
#### [replaced 094] Differentiable, Bit-shifting, and Scalable Quantization without training neural network from scratch
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: [https://arxiv.org/pdf/2510.16088v3](https://arxiv.org/pdf/2510.16088v3)**

> **作者:** Zia Badar
>
> **摘要:** Quantization of neural networks provides benefits of inference in less compute and memory requirements. Previous work in quantization lack two important aspects which this work provides. First almost all previous work in quantization used a non-differentiable approach and for learning; the derivative is usually set manually in backpropogation which make the learning ability of algorithm questionable, our approach is not just differentiable, we also provide proof of convergence of our approach to the optimal neural network. Second previous work in shift/logrithmic quantization either have avoided activation quantization along with weight quantization or achieved less accuracy. Learning logrithmic quantize values of form $2^n$ requires the quantization function can scale to more than 1 bit quantization which is another benifit of our quantization that it provides $n$ bits quantization as well. Our approach when tested with image classification task using imagenet dataset, resnet18 and weight quantization only achieves less than 1 percent accuracy compared to full precision accuracy while taking only 15 epochs to train using shift bit quantization and achieves comparable to SOTA approaches accuracy in both weight and activation quantization using shift bit quantization in 15 training epochs with slightly higher(only higher cpu instructions) inference cost compared to 1 bit quantization(without logrithmic quantization) and not requiring any higher precision multiplication.
>
---
#### [replaced 095] Capture Stage Matting: Challenges, Approaches, and Solutions for Offline and Real-Time Processing
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.07623v2](https://arxiv.org/pdf/2507.07623v2)**

> **作者:** Hannah Dröge; Janelle Pfeifer; Saskia Rabich; Reinhard Klein; Matthias B. Hullin; Markus Plack
>
> **摘要:** Capture stages are high-end sources of state-of-the-art recordings for downstream applications in movies, games, and other media. One crucial step in almost all pipelines is matting, i.e., separating captured performances from the background. While common matting algorithms deliver remarkable performance in other applications like teleconferencing and mobile entertainment, we found that they struggle significantly with the peculiarities of capture stage content. The goal of our work is to share insights into those challenges as a curated list of these characteristics along with a constructive discussion for proactive intervention and present a guideline to practitioners for an improved workflow to mitigate unresolved challenges. To this end, we also demonstrate an efficient pipeline to adapt state-of-the-art approaches to such custom setups without the need for extensive annotations, both offline and real-time. For an objective evaluation, we introduce a validation methodology using a state-of-the-art diffusion model to demonstrate the benefits of our approach.
>
---
